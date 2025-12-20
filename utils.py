import json
import os
from datetime import datetime
from typing import Any, Dict

def ensure_dirs():
    os.makedirs("data/docs", exist_ok=True)
    os.makedirs("data/indexes", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _json_default(o):
    # Handles numpy types like float32/int64 if they appear
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass

    # fallback
    return str(o)

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
