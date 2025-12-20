from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple


DB_DEFAULT_PATH = os.path.join("runs", "rag_debugger.sqlite")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_hash(obj: Any) -> str:
    """
    Stable SHA256 for dict/list/str primitives.
    """
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def init_db(db_path: str = DB_DEFAULT_PATH) -> None:
    _ensure_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            notes TEXT,
            dataset_id TEXT,
            dataset_hash TEXT,
            index_id TEXT,
            index_hash TEXT,
            config_json TEXT NOT NULL,
            config_hash TEXT NOT NULL
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS query_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            q_index INTEGER NOT NULL,
            query_text TEXT NOT NULL,
            expected_id TEXT,
            retrieved_ids_json TEXT NOT NULL,
            distances_json TEXT NOT NULL,
            hit_k_json TEXT NOT NULL,
            weak_retrieval INTEGER NOT NULL,
            latency_ms REAL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id TEXT PRIMARY KEY,
            metrics_json TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );
        """)
        conn.commit()


def create_run(
    config: Dict[str, Any],
    notes: str = "",
    dataset_id: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    index_id: Optional[str] = None,
    index_hash: Optional[str] = None,
    db_path: str = DB_DEFAULT_PATH,
) -> str:
    init_db(db_path)

    run_id = str(uuid.uuid4())
    created_at = _utc_now_iso()
    config_hash = stable_hash(config)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO runs (
                run_id, created_at, notes, dataset_id, dataset_hash,
                index_id, index_hash, config_json, config_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, created_at, notes or None, dataset_id, dataset_hash,
            index_id, index_hash, json.dumps(config, ensure_ascii=False),
            config_hash
        ))
        conn.commit()

    return run_id


def save_query_result(
    run_id: str,
    q_index: int,
    query_text: str,
    retrieved_ids: List[str],
    distances: List[float],
    hit_k: Dict[str, int],  # e.g. {"hit@1":0,"hit@3":1,"hit@5":1}
    weak_retrieval: bool,
    expected_id: Optional[str] = None,
    latency_ms: Optional[float] = None,
    db_path: str = DB_DEFAULT_PATH,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO query_results (
                run_id, q_index, query_text, expected_id,
                retrieved_ids_json, distances_json, hit_k_json,
                weak_retrieval, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            q_index,
            query_text,
            expected_id,
            json.dumps(retrieved_ids, ensure_ascii=False),
            json.dumps(distances),
            json.dumps(hit_k),
            1 if weak_retrieval else 0,
            latency_ms
        ))
        conn.commit()


def save_run_metrics(
    run_id: str,
    metrics: Dict[str, Any],
    db_path: str = DB_DEFAULT_PATH,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO run_metrics (run_id, metrics_json)
            VALUES (?, ?)
        """, (run_id, json.dumps(metrics, ensure_ascii=False)))
        conn.commit()


def list_runs(db_path: str = DB_DEFAULT_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT r.run_id, r.created_at, r.notes,
                   r.dataset_id, r.dataset_hash, r.index_id, r.index_hash,
                   r.config_hash, r.config_json,
                   m.metrics_json
            FROM runs r
            LEFT JOIN run_metrics m ON m.run_id = r.run_id
            ORDER BY r.created_at DESC
        """).fetchall()

    runs = []
    for row in rows:
        (run_id, created_at, notes, dataset_id, dataset_hash,
         index_id, index_hash, config_hash, config_json, metrics_json) = row
        runs.append({
            "run_id": run_id,
            "created_at": created_at,
            "notes": notes or "",
            "dataset_id": dataset_id or "",
            "dataset_hash": dataset_hash or "",
            "index_id": index_id or "",
            "index_hash": index_hash or "",
            "config_hash": config_hash,
            "config": json.loads(config_json),
            "metrics": json.loads(metrics_json) if metrics_json else {},
        })
    return runs


def load_run_details(run_id: str, db_path: str = DB_DEFAULT_PATH) -> Dict[str, Any]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        r = conn.execute("""
            SELECT run_id, created_at, notes, dataset_id, dataset_hash,
                   index_id, index_hash, config_json, config_hash
            FROM runs WHERE run_id = ?
        """, (run_id,)).fetchone()
        if not r:
            raise ValueError(f"Run not found: {run_id}")

        m = conn.execute("SELECT metrics_json FROM run_metrics WHERE run_id = ?", (run_id,)).fetchone()
        q = conn.execute("""
            SELECT q_index, query_text, expected_id,
                   retrieved_ids_json, distances_json, hit_k_json,
                   weak_retrieval, latency_ms
            FROM query_results
            WHERE run_id = ?
            ORDER BY q_index ASC
        """, (run_id,)).fetchall()

    run = {
        "run_id": r[0],
        "created_at": r[1],
        "notes": r[2] or "",
        "dataset_id": r[3] or "",
        "dataset_hash": r[4] or "",
        "index_id": r[5] or "",
        "index_hash": r[6] or "",
        "config": json.loads(r[7]),
        "config_hash": r[8],
        "metrics": json.loads(m[0]) if m else {},
        "queries": []
    }

    for row in q:
        run["queries"].append({
            "q_index": row[0],
            "query_text": row[1],
            "expected_id": row[2],
            "retrieved_ids": json.loads(row[3]),
            "distances": json.loads(row[4]),
            "hit_k": json.loads(row[5]),
            "weak_retrieval": bool(row[6]),
            "latency_ms": row[7],
        })

    return run
