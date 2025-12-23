# cli_eval.py
import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()


from rag_core import load_index, retrieve_with_scores
from eval_core import load_testset_csv, hit_at_k, summarize_hits
from run_store import (
    create_run,
    save_query_result,
    save_run_metrics,
    stable_hash,
    DB_DEFAULT_PATH,
)


# -----------------------------
# Helpers
# -----------------------------

def doc_id(doc: Document) -> str:
    """
    Stable-ish ID for a retrieved chunk. Keeps it human-readable.
    """
    src = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", ""))
    # optional chunk id if you have it; otherwise fallback to hash of content snippet
    chunk_id = str(doc.metadata.get("chunk_id", ""))

    if chunk_id:
        return f"{src}::p{page}::c{chunk_id}"

    # fallback: hash first 400 chars to avoid huge strings
    preview = (doc.page_content or "")[:400]
    preview_hash = stable_hash(preview)
    return f"{src}::p{page}::h{preview_hash[:12]}"


def is_weak_retrieval(
    retrieved: List[Tuple[Document, float]],
    hit_info: Dict[str, Any],
    weak_distance_threshold: Optional[float],
) -> bool:
    """
    Definition (MVP):
    - weak if no hit_source (and expected_source exists)
    - OR (threshold provided and best distance > threshold)
    """
    expected_source_exists = True  # because testset always has column, but may be empty
    if (hit_info.get("hit_source") is False) and expected_source_exists:
        return True

    if weak_distance_threshold is not None and retrieved:
        best_dist = float(retrieved[0][1])
        if best_dist > weak_distance_threshold:
            return True

    return False


def compute_run_metrics(
    per_q_hits: List[Dict[str, Any]],
    weak_flags: List[bool],
    latencies_ms: List[float],
) -> Dict[str, Any]:
    hits_summary = summarize_hits(per_q_hits)
    n = hits_summary.get("n", 0) or 0

    weak_rate = (sum(1 for w in weak_flags if w) / n) if n else 0.0
    avg_latency = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else None

    metrics = {
        "n": n,
        "hit_source_rate": float(hits_summary.get("hit_source_rate", 0.0) or 0.0),
        "hit_page_rate": hits_summary.get("hit_page_rate", None),
        "weak_retrieval_rate": float(weak_rate),
        "avg_latency_ms": avg_latency,
    }
    return metrics


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Run batch RAG eval + save a run (SQLite).")
    p.add_argument("--testset", required=True, help="Path to testset CSV")
    p.add_argument("--notes", default="", help="Notes for the run (e.g. chunk800_k5)")
    p.add_argument("--k", type=int, default=5, help="Top-K retrieval")
    p.add_argument("--limit", type=int, default=0, help="Limit rows (0 = all)")
    p.add_argument("--db", default=DB_DEFAULT_PATH, help="SQLite db path")
    p.add_argument(
        "--weak_distance_threshold",
        type=float,
        default=None,
        help="If set: weak if best distance > threshold (FAISS distance).",
    )
    args = p.parse_args()

    # ---- Load testset
    rows = load_testset_csv(args.testset)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    dataset_hash = stable_hash(rows)

    # ---- Load index
    vs = load_index()

    # ---- Create run
    config = {
        "k": args.k,
        "weak_distance_threshold": args.weak_distance_threshold,
        "index_path": "data/indexes/faiss_index",
        "retriever": "faiss.similarity_search_with_score",
        "distance_meaning": "lower_is_better",
    }

    run_id = create_run(
        config=config,
        notes=args.notes,
        dataset_id=args.testset,
        dataset_hash=dataset_hash,
        db_path=args.db,
    )

    per_q_hits: List[Dict[str, Any]] = []
    weak_flags: List[bool] = []
    latencies_ms: List[float] = []

    # ---- Eval loop
    for i, r in enumerate(rows):
        q = (r.get("question") or "").strip()
        expected_source = (r.get("expected_source") or "").strip()
        expected_page = (r.get("expected_page") or "").strip()

        t0 = time.perf_counter()
        retrieved = retrieve_with_scores(vs, q, k=args.k)  # [(doc, dist)]
        t1 = time.perf_counter()

        latency = (t1 - t0) * 1000.0
        latencies_ms.append(latency)

        # Hit@K (source + optional page)
        hit_info = hit_at_k(
            retrieved=retrieved,
            expected_source=expected_source,
            expected_page=expected_page if expected_page else None,
        )
        per_q_hits.append(hit_info)

        # Weak retrieval
        weak = is_weak_retrieval(
            retrieved=retrieved,
            hit_info=hit_info,
            weak_distance_threshold=args.weak_distance_threshold,
        )
        weak_flags.append(weak)

        # Store query result
        retrieved_ids = [doc_id(doc) for doc, _ in retrieved]
        distances = [float(score) for _, score in retrieved]

        # For your DB schema, hit_k expects Dict[str,int]
        # We’ll store source/page hits as 0/1 in a compact form.
        hit_k = {
            f"hit_source@{args.k}": 1 if hit_info.get("hit_source") else 0,
            f"hit_page@{args.k}": 1 if hit_info.get("hit_page") else 0
            if hit_info.get("hit_page") is not None
            else 0,
        }

        save_query_result(
            run_id=run_id,
            q_index=i,
            query_text=q,
            expected_id=r.get("id"),
            retrieved_ids=retrieved_ids,
            distances=distances,
            hit_k=hit_k,
            weak_retrieval=bool(weak),
            latency_ms=latency,
            db_path=args.db,
        )

        # Optional: lightweight progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(rows)}...")

    # ---- Save run metrics
    metrics = compute_run_metrics(per_q_hits, weak_flags, latencies_ms)
    save_run_metrics(run_id, metrics=metrics, db_path=args.db)

    # ---- Print summary + run id
    print("\n✅ Saved eval run")
    print("Run ID:", run_id)
    print("Notes:", args.notes)
    print("Metrics:")
    print(f"  n: {metrics['n']}")
    print(f"  hit_source_rate: {metrics['hit_source_rate']:.4f}")
    if metrics["hit_page_rate"] is not None:
        print(f"  hit_page_rate: {float(metrics['hit_page_rate']):.4f}")
    print(f"  weak_retrieval_rate: {metrics['weak_retrieval_rate']:.4f}")
    if metrics["avg_latency_ms"] is not None:
        print(f"  avg_latency_ms: {float(metrics['avg_latency_ms']):.2f}")


if __name__ == "__main__":
    main()
