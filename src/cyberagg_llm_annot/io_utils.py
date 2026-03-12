from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def safe_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)

def safe_write_json(path: str, obj: Any, ensure_ascii: bool = False, indent: int = 2) -> None:
    text = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
    safe_write_text(path, text)

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
