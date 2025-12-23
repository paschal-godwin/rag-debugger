# regression_gate.py
import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

from run_store import load_run_details, DB_DEFAULT_PATH


@dataclass
class Thresholds:
    # Hit source rate: higher is better
    warn_drop_hit_source: float = 0.01
    fail_drop_hit_source: float = 0.03

    # Weak retrieval rate: lower is better
    warn_increase_weak: float = 0.02
    fail_increase_weak: float = 0.05


def _get(metrics: Dict[str, Any], key: str) -> float:
    if key not in metrics:
        raise KeyError(f"Missing required metric: {key}")
    return float(metrics[key])



def main():
    p = argparse.ArgumentParser(description="RAG Regression Gate (baseline vs candidate).")
    p.add_argument("--baseline", required=True, help="Baseline run_id")
    p.add_argument("--candidate", required=True, help="Candidate run_id")
    p.add_argument("--db", default=DB_DEFAULT_PATH, help="SQLite db path")

    p.add_argument("--warn_drop_hit", type=float, default=0.01)
    p.add_argument("--fail_drop_hit", type=float, default=0.03)
    p.add_argument("--warn_inc_weak", type=float, default=0.02)
    p.add_argument("--fail_inc_weak", type=float, default=0.05)

    args = p.parse_args()

    th = Thresholds(
        warn_drop_hit_source=args.warn_drop_hit,
        fail_drop_hit_source=args.fail_drop_hit,
        warn_increase_weak=args.warn_inc_weak,
        fail_increase_weak=args.fail_inc_weak,
    )

    base = load_run_details(args.baseline, db_path=args.db)
    cand = load_run_details(args.candidate, db_path=args.db)

    bm = base["metrics"]
    cm = cand["metrics"]

    b_hit = _get(bm, "hit_source_rate_at_top_k")
    c_hit = _get(cm, "hit_source_rate_at_top_k")

    b_weak = _get(bm, "weak_retrieval_rate")
    c_weak = _get(cm, "weak_retrieval_rate")


    missing = [name for name, val in [
        ("baseline.hit_source_rate", b_hit),
        ("candidate.hit_source_rate", c_hit),
        ("baseline.weak_retrieval_rate", b_weak),
        ("candidate.weak_retrieval_rate", c_weak),
    ] if val is None]

    if missing:
        print("\n❌ Regression gate error: missing required metrics:")
        for m in missing:
            print("  -", m)
        print("\nMake sure you ran cli_eval.py for both runs so metrics exist.\n")
        sys.exit(2)

    hit_drop = b_hit - c_hit               # positive = candidate worse
    weak_increase = c_weak - b_weak        # positive = candidate worse

    status = "PASS"
    reasons = []

    if bm.get("top_k") != cm.get("top_k"):
        print("⚠️ Warning: comparing runs with different top_k values")


    # Hit@K gate
    if hit_drop >= th.fail_drop_hit_source:
        status = "FAIL"
        reasons.append(f"Hit source rate dropped by {hit_drop:.4f} (>= {th.fail_drop_hit_source:.4f})")
    elif hit_drop >= th.warn_drop_hit_source:
        status = "WARN"
        reasons.append(f"Hit source rate dropped by {hit_drop:.4f} (>= {th.warn_drop_hit_source:.4f})")

    # Weak retrieval gate
    if weak_increase >= th.fail_increase_weak:
        status = "FAIL"
        reasons.append(f"Weak retrieval rate increased by {weak_increase:.4f} (>= {th.fail_increase_weak:.4f})")
    elif weak_increase >= th.warn_increase_weak and status != "FAIL":
        status = "WARN"
        reasons.append(f"Weak retrieval rate increased by {weak_increase:.4f} (>= {th.warn_increase_weak:.4f})")

    print("\n================ RAG Regression Gate ================")
    print("Baseline  :", args.baseline, "|", base.get("notes", ""))
    print("Candidate :", args.candidate, "|", cand.get("notes", ""))
    print("\nMetrics:")
    print(f"  hit_source_rate: baseline={b_hit:.4f}  candidate={c_hit:.4f}  drop={hit_drop:.4f}")
    print(f"  weak_retrieval_rate: baseline={b_weak:.4f}  candidate={c_weak:.4f}  increase={weak_increase:.4f}")
    print("\nDecision:", status)
    if reasons:
        print("Reasons:")
        for r in reasons:
            print("  -", r)
    print("====================================================\n")

    # CI-friendly exit codes: FAIL blocks merges
    if status == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
