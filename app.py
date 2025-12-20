import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils import ensure_dirs, now_iso, append_jsonl
from ingest import load_pdfs_from_folder
from rag_core import (
    chunk_documents,
    build_faiss_index,
    save_index,
    load_index,
    retrieve_with_scores,
    answer_with_citations
)

LOG_PATH = "data/logs/rag_runs.jsonl"

load_dotenv()
ensure_dirs()

st.set_page_config(page_title="RAG Debugger", layout="wide")
st.title("RAG Debugger — Single Query Inspector")

st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Build/Refresh Index", "Debug Query", "Batch Evaluation", "Compare Runs"])


chunk_size = st.sidebar.slider("Chunk size", 300, 1500, 800, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, 120, 10)
top_k = st.sidebar.slider("Top K", 1, 15, 5, 1)
WEAK_RETRIEVAL_THRESHOLD = st.sidebar.slider(
    "Weak Retrieval Threshold (distance)",
    0.6, 0.9, 0.8, 0.01
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

docs_folder = "data/docs"

if mode == "Build/Refresh Index":
    st.subheader("1) Put PDFs in: data/docs/")
    st.write("Then click build. This will chunk → embed → store in FAISS.")

    if st.button("Build Index"):
        with st.spinner("Loading PDFs..."):
            docs = load_pdfs_from_folder(docs_folder)
        st.write(f"Loaded **{len(docs)}** pages.")
        if len(docs) == 0:
            st.error(
                "No text could be extracted from your PDFs. "
                "They may be scanned/image PDFs. Try a text-based PDF first, or we’ll add OCR next."
            )
            st.stop()
        st.markdown("### Extraction Preview (first 2 pages)")
        for d in docs[:2]:
            st.write(f"**{d.metadata.get('source')} — page {d.metadata.get('page')}**")
            st.write(d.page_content[:600])
            st.divider()

        with st.spinner("Chunking..."):
            chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.write(f"Created **{len(chunks)}** chunks.")

        if len(chunks) == 0:
            st.error(
                "Chunking produced 0 chunks (meaning extracted text was empty). "
                "Try a different PDF or add OCR support."
            )
            st.stop()

        with st.spinner("Embedding + Building FAISS index..."):
            vs = build_faiss_index(chunks)
            save_index(vs)

        st.success("Index built and saved ✅")
        st.info("Now switch to **Debug Query** mode.")

elif mode == 'Debug Query':
    st.subheader("Ask a question and inspect retrieval + answer")
    query = st.text_input("Question", placeholder="e.g., What is lifestyle inflation?")
    run = st.button("Run Debugger")

    if run:
        if not os.path.exists("data/indexes/faiss_index"):
            st.error("No index found. Build the index first in 'Build/Refresh Index' mode.")
        elif not query.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Loading index..."):
                vs = load_index()

            with st.spinner("Retrieving chunks..."):
                retrieved = retrieve_with_scores(vs, query, k=top_k)

            left, right = st.columns([1.2, 1])

            with left:
                st.markdown("### Retrieved Chunks (ranked)")
                for i, (doc, score) in enumerate(retrieved, start=1):
                    src = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", None)

                    st.markdown(f"**#{i} — {src} (page {page})**  \nScore: `{score}`")
                    st.write(doc.page_content[:900] + ("..." if len(doc.page_content) > 900 else ""))
                    st.divider()

            with right:
                st.markdown("### Answer")
                with st.spinner("Generating answer..."):
                    out = answer_with_citations(query, retrieved)

                st.write(out["answer"])

                st.markdown("### Citation Previews")
                for c in out["citations"]:
                    st.markdown(f"**[{c['ref']}] {c['source']} — page {c['page']}**  \nScore: `{c['score']}`")
                    st.caption(c["preview"])
                    st.divider()

            # Log run
            record = {
                "ts": now_iso(),
                "query": query,
                "config": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "top_k": top_k
                },
                "retrieved": [
                    {
                        "rank": i,
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "score": float(score),
                        "text_preview": doc.page_content[:500]
                    }
                    for i, (doc, score) in enumerate(retrieved, start=1)
                ],
                "answer": out["answer"]
            }
            append_jsonl(LOG_PATH, record)

            st.success("Run logged to data/logs/rag_runs.jsonl ✅")
            best_score = float(retrieved[0][1]) if retrieved else None
            threshold_score = WEAK_RETRIEVAL_THRESHOLD
            if best_score is not None and best_score > threshold_score:
                st.warning("⚠️ Retrieval looks weak (high distance). Answer may be incomplete or ungrounded.")

elif mode == "Batch Evaluation":
    import time
    import pandas as pd
    from eval_core import load_testset_csv, hit_at_k, summarize_hits
    from rag_core import load_index, retrieve_with_scores
    from run_store import create_run, save_query_result, save_run_metrics, stable_hash

    st.subheader("Batch Evaluation (Hit@K)")
    st.write("Runs a set of questions and measures whether the expected source appears in the retrieved top-k.")

    testset_path = "data/testset.csv"
    if not os.path.exists(testset_path):
        st.error("Missing data/testset.csv. Create it first.")
        st.stop()

    if not os.path.exists("data/indexes/faiss_index"):
        st.error("No index found. Build the index first.")
        st.stop()

    # Optional notes to identify runs in comparisons
    run_notes = st.text_input("Run notes (optional)", placeholder="e.g., k=5, chunk=800/120, threshold=0.8")

    test_rows = load_testset_csv(testset_path)
    st.write(f"Loaded **{len(test_rows)}** test questions from `{testset_path}`")

    # Build a clean config snapshot (IMPORTANT: keep primitives only)
    config = {
        "embedding_model": "text-embedding-3-large",
        "index_type": "faiss",
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "retrieval_k": int(top_k),
        "distance_metric": "l2",
        "weak_retrieval_threshold": float(WEAK_RETRIEVAL_THRESHOLD),
        "search_type": "similarity",
        "testset_path": testset_path,
    }

    # Versioning for fairness (simple + effective)
    dataset_id = os.path.basename(testset_path)
    dataset_hash = stable_hash(test_rows)  # test_rows is list[dict] -> good
    index_id = "faiss_index"
    index_path = "data/indexes/faiss_index"
    index_hash = str(os.path.getmtime(index_path)) if os.path.exists(index_path) else ""

    run_eval = st.button("Run Batch Evaluation")

    if run_eval:
        # Create a run ONLY at execution time
        run_id = create_run(
            config=config,
            notes=run_notes,
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            index_id=index_id,
            index_hash=index_hash,
        )

        vs = load_index()

        results = []
        failures = []

        with st.spinner("Evaluating..."):
            for i, row in enumerate(test_rows):
                q = (row.get("question") or "").strip()
                if not q:
                    continue

                start = time.perf_counter()
                retrieved = retrieve_with_scores(vs, q, k=top_k)
                latency_ms = (time.perf_counter() - start) * 1000

                hit_info = hit_at_k(
                    retrieved=retrieved,
                    expected_source=row.get("expected_source"),
                    expected_page=row.get("expected_page"),
                )

                best_score = float(retrieved[0][1]) if retrieved else None
                weak_retrieval = (best_score is not None and best_score > WEAK_RETRIEVAL_THRESHOLD)

                # Build stable retrieved_ids + distances (so you can compare later)
                retrieved_ids = []
                distances = []
                for rank, (doc, score) in enumerate(retrieved, start=1):
                    src = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", None)
                    retrieved_ids.append(f"{src}::page={page}::rank={rank}")
                    distances.append(float(score))

                # Compute hit@k for common K values (bounded by your top_k)
                def hit_at_limit(limit: int) -> int:
                    limit = min(int(limit), int(top_k))
                    # if your hit_at_k checks "within top_k", we can approximate by slicing
                    sliced = retrieved[:limit]
                    tmp = hit_at_k(
                        retrieved=sliced,
                        expected_source=row.get("expected_source"),
                        expected_page=row.get("expected_page"),
                    )
                    return 1 if tmp.get("hit_source") else 0

                hit_k = {
                    "hit@1": hit_at_limit(1),
                    "hit@3": hit_at_limit(3),
                    "hit@5": hit_at_limit(5),
                    f"hit@{top_k}": 1 if hit_info.get("hit_source") else 0,
                }

                # Save per-query run record (this is the main “run-save” feature)
                expected_source = row.get("expected_source")
                expected_page = row.get("expected_page")
                expected_id = f"{expected_source}::page={expected_page}"

                save_query_result(
                    run_id=run_id,
                    q_index=i,
                    query_text=q,
                    expected_id=expected_id,
                    retrieved_ids=retrieved_ids,
                    distances=distances,
                    hit_k=hit_k,
                    weak_retrieval=bool(weak_retrieval),
                    latency_ms=float(latency_ms),
                )

                res = {
                    "id": row.get("id"),
                    "question": q,
                    "expected_source": expected_source,
                    "expected_page": expected_page or None,
                    "best_score": best_score,
                    "hit_source": hit_info.get("hit_source"),
                    "hit_page": hit_info.get("hit_page"),
                    "weak_retrieval": weak_retrieval,
                    "top_sources": hit_info.get("retrieved_sources", [])[: min(len(hit_info.get("retrieved_sources", [])), 5)],
                    "latency_ms": latency_ms,
                }
                results.append(res)

                if expected_source and not hit_info.get("hit_source"):
                    failures.append(res)

        # Aggregate + save run metrics
        summary = summarize_hits(results)
        weak_rate = (sum(1 for r in results if r["weak_retrieval"]) / len(results)) if results else 0.0
        avg_latency = (sum(r["latency_ms"] for r in results) / len(results)) if results else 0.0

        metrics = {
            "num_queries": int(summary.get("n", len(results))),
            "hit_source_rate_at_top_k": float(summary.get("hit_source_rate", 0.0)),
            "hit_page_rate_at_top_k": summary.get("hit_page_rate", None),
            "weak_retrieval_rate": float(weak_rate),
            "avg_latency_ms": float(avg_latency),
            "top_k": int(top_k),
        }
        save_run_metrics(run_id, metrics)

        st.success(f"Saved run ✅  run_id={run_id[:8]}  (stored in SQLite via run_store.py)")

        # Display (your existing UI stays)
        st.markdown("### Summary")
        st.write(f"Questions evaluated: **{summary['n']}**")
        st.write(f"Hit@{top_k} (by source): **{summary['hit_source_rate']*100:.1f}%**")
        if summary["hit_page_rate"] is not None:
            st.write(f"Hit@{top_k} (by page): **{summary['hit_page_rate']*100:.1f}%**")

        st.write(f"Weak retrieval rate (score > {WEAK_RETRIEVAL_THRESHOLD}): **{weak_rate*100:.1f}%**")

        st.markdown("### Detailed Results")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        st.markdown("### Failures (expected source not retrieved)")
        if failures:
            fdf = pd.DataFrame(failures)
            st.dataframe(fdf, use_container_width=True)
        else:
            st.success("No failures 🎉")

elif mode == "Compare Runs":
    import pandas as pd
    from run_store import list_runs, load_run_details

    st.title("RAG Debugger — Compare Runs")
    st.write("Select two saved runs to compare metrics, configs, and per-query behavior.")

    runs = list_runs()
    if not runs:
        st.info("No saved runs found yet. Run a Batch Evaluation first.")
        st.stop()

    # Build friendly labels
    def run_label(r: dict) -> str:
        notes = (r.get("notes") or "").strip()
        note_part = f" | {notes}" if notes else ""
        return f'{r["created_at"]} | {r["run_id"][:8]}{note_part}'

    labels = [run_label(r) for r in runs]
    label_to_id = {run_label(r): r["run_id"] for r in runs}

    col1, col2 = st.columns(2)
    with col1:
        a_label = st.selectbox("Run A", labels, index=0)
    with col2:
        b_label = st.selectbox("Run B", labels, index=1 if len(labels) > 1 else 0)

    runA = load_run_details(label_to_id[a_label])
    runB = load_run_details(label_to_id[b_label])

    # Guardrails for fair comparison
    st.markdown("### Compatibility Check")
    if runA.get("dataset_hash") and runB.get("dataset_hash") and runA["dataset_hash"] != runB["dataset_hash"]:
        st.warning("⚠️ Different dataset_hash: comparison may be unfair.")
    else:
        st.success("Dataset looks consistent ✅")

    if runA.get("index_hash") and runB.get("index_hash") and runA["index_hash"] != runB["index_hash"]:
        st.warning("⚠️ Different index_hash: comparison may be unfair.")
    else:
        st.success("Index looks consistent ✅")

    # Metrics comparison
    st.markdown("### Metrics Comparison")
    mA = runA.get("metrics", {}) or {}
    mB = runB.get("metrics", {}) or {}

    def delta(a, b):
        if a is None or b is None:
            return None
        try:
            return float(b) - float(a)
        except Exception:
            return None

    metric_rows = []
    keys = sorted(set(mA.keys()) | set(mB.keys()))
    for k in keys:
        metric_rows.append({
            "metric": k,
            "A": mA.get(k),
            "B": mB.get(k),
            "Δ (B-A)": delta(mA.get(k), mB.get(k))
        })

    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

    # Config diff
    st.markdown("### Config Diff")
    cA = runA.get("config", {}) or {}
    cB = runB.get("config", {}) or {}

    cfg_keys = sorted(set(cA.keys()) | set(cB.keys()))
    cfg_rows = []
    for k in cfg_keys:
        if cA.get(k) != cB.get(k):
            cfg_rows.append({"key": k, "A": cA.get(k), "B": cB.get(k)})

    if cfg_rows:
        st.dataframe(pd.DataFrame(cfg_rows), use_container_width=True)
    else:
        st.success("Configs match (no differences) ✅")

    # Per-query diffs
    st.markdown("### Per-query Differences")

    qA = {q["q_index"]: q for q in runA.get("queries", [])}
    qB = {q["q_index"]: q for q in runB.get("queries", [])}
    common = sorted(set(qA.keys()) & set(qB.keys()))

    if not common:
        st.info("No overlapping queries found (q_index mismatch).")
        st.stop()

    diffs = []
    for qi in common:
        A = qA[qi]
        B = qB[qi]

        A_hit = A.get("hit_k", {}) or {}
        B_hit = B.get("hit_k", {}) or {}

        # choose the most relevant hit@key to display (prefer hit@{top_k}, else hit@5, else hit@1)
        def pick_hit(hit_dict):
            if not hit_dict:
                return None
            # try find "hit@X" where X is largest number in keys
            candidates = []
            for k, v in hit_dict.items():
                if isinstance(k, str) and k.startswith("hit@"):
                    try:
                        num = int(k.split("@")[1])
                        candidates.append((num, k, v))
                    except Exception:
                        pass
            if not candidates:
                return None
            num, k, v = sorted(candidates, key=lambda x: x[0])[-1]
            return k, v

        A_pick = pick_hit(A_hit)
        B_pick = pick_hit(B_hit)

        A_hit_key, A_hit_val = (A_pick if A_pick else (None, None))
        B_hit_key, B_hit_val = (B_pick if B_pick else (None, None))

        COMMON_KEYS = ["hit@1", "hit@3", "hit@5"]

        def normalize_hits(hit_dict):
            return {k: hit_dict.get(k) for k in COMMON_KEYS}

        A_norm = normalize_hits(A_hit)
        B_norm = normalize_hits(B_hit)

        changed = (
            A_norm != B_norm
            or bool(A.get("weak_retrieval")) != bool(B.get("weak_retrieval"))
            or (A.get("retrieved_ids")[:1] != B.get("retrieved_ids")[:1])
        )

        if changed:
            diffs.append({
                "q_index": qi,
                "question": A.get("query_text"),
                "A_hit": f"{A_hit_key}={A_hit_val}" if A_hit_key else None,
                "B_hit": f"{B_hit_key}={B_hit_val}" if B_hit_key else None,
                "A_weak": bool(A.get("weak_retrieval")),
                "B_weak": bool(B.get("weak_retrieval")),
                "A_top1": (A.get("retrieved_ids") or [None])[0],
                "B_top1": (B.get("retrieved_ids") or [None])[0],
                "A_best_dist": (A.get("distances") or [None])[0],
                "B_best_dist": (B.get("distances") or [None])[0],
            })

    st.write(f"Changed queries: **{len(diffs)}** out of **{len(common)}**")
    if diffs:
        diffs_df = pd.DataFrame(diffs)
        st.dataframe(diffs_df, use_container_width=True)

        pick = st.selectbox("Inspect a changed query (q_index)", diffs_df["q_index"].tolist())
        A = qA[int(pick)]
        B = qB[int(pick)]

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Run A")
            st.caption(a_label)
            st.json(A)
        with colB:
            st.markdown("#### Run B")
            st.caption(b_label)
            st.json(B)
    else:
        st.success("No per-query differences detected 🎉")
