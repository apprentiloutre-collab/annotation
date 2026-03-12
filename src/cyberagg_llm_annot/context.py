from __future__ import annotations
import math
from typing import Any, Dict, Optional


def _safe_str(val: Any, default: str = "") -> str:
    """Convertit en string, remplace None/NaN par *default*."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return str(val)


def get_message_window(df, idx: int) -> Dict[str, Optional[Dict[str, Any]]]:
    """Fenêtre prev / target / next autour de l'index *idx*."""
    n = len(df)
    target = df.iloc[idx].to_dict()
    prev_msg = df.iloc[idx - 1].to_dict() if idx - 1 >= 0 else None
    next_msg = df.iloc[idx + 1].to_dict() if idx + 1 < n else None
    return {"prev": prev_msg, "target": target, "next": next_msg}


def minimal_msg_repr(msg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Représentation compacte et sûre pour injection dans le prompt."""
    if msg is None:
        return None
    return {
        "ID":   msg.get("ID"),
        "NAME": _safe_str(msg.get("NAME"), "?"),
        "TIME": _safe_str(msg.get("TIME")),
        "TEXT": _safe_str(msg.get("TEXT")),
        "ROLE": _safe_str(msg.get("ROLE")),
    }
