import csv
from typing import Dict, Any, List, Tuple, Optional
from langchain.schema import Document

def load_testset_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "id": r.get("id"),
                "question": (r.get("question") or "").strip(),
                "expected_source": (r.get("expected_source") or "").strip(),
                "expected_page": (r.get("expected_page") or "").strip(),
            })
    return rows

def hit_at_k(
    retrieved: List[Tuple[Document, float]],
    expected_source: str,
    expected_page: Optional[str] = None
) -> Dict[str, Any]:
    """
    Hit@K by source, and optionally by page if provided.
    """
    expected_source = (expected_source or "").strip()
    expected_page = (expected_page or "").strip() if expected_page is not None else ""

    sources = [doc.metadata.get("source") for doc, _ in retrieved]
    pages = [str(doc.metadata.get("page")) for doc, _ in retrieved]

    hit_source = False
    hit_page = False

    if expected_source:
        hit_source = expected_source in sources

    if expected_source and expected_page:
        # must match both source and page
        for (doc, _score) in retrieved:
            if doc.metadata.get("source") == expected_source and str(doc.metadata.get("page")) == expected_page:
                hit_page = True
                break

    return {
        "hit_source": hit_source,
        "hit_page": hit_page if expected_page else None,
        "retrieved_sources": sources,
        "retrieved_pages": pages,
    }

def summarize_hits(per_question: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_question) if per_question else 0
    if n == 0:
        return {"n": 0, "hit_source_rate": 0.0, "hit_page_rate": None}

    hit_source_rate = sum(1 for r in per_question if r.get("hit_source")) / n

    page_items = [r for r in per_question if r.get("hit_page") is not None]
    hit_page_rate = None
    if page_items:
        hit_page_rate = sum(1 for r in page_items if r.get("hit_page")) / len(page_items)

    return {
        "n": n,
        "hit_source_rate": hit_source_rate,
        "hit_page_rate": hit_page_rate,
    }
