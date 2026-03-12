from __future__ import annotations
import math
import re
from typing import Any, Dict, Optional

NULL_PATTERN = re.compile(r"Majority:\s*NULL", re.IGNORECASE)

def is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False

def parse_cell_with_possible_null(value: Any) -> Dict[str, Any]:
    """
    Normalise une cellule provenant du XLSX.
    Cas possibles:
    - NaN / None => status="missing"
    - String contenant "Majority: NULL" => status="no_consensus" + raw
    - sinon => status="value" + value (string)
    """
    if is_nan(value):
        return {"status": "missing", "value": None, "raw": None}

    if isinstance(value, str) and NULL_PATTERN.search(value):
        return {"status": "no_consensus", "value": None, "raw": value}

    return {"status": "value", "value": value, "raw": None}

def extract_row_labels(row: Dict[str, Any], label_cols: list[str]) -> Dict[str, Dict[str, Any]]:
    """
    Retourne un dict {col: parsed_cell_dict} pour les colonnes d'annotation souhaitées.
    """
    out = {}
    for col in label_cols:
        out[col] = parse_cell_with_possible_null(row.get(col))
    return out
