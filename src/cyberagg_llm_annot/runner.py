from __future__ import annotations
import glob
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import (
    ensure_dir, append_jsonl, safe_write_json, load_json, utc_now_iso,
)

# ── Regex pour extraire le JSON d'un éventuel bloc ```json … ``` ───────────
_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)

# ── Émotions attendues dans la sortie ──────────────────────────────────────
_EXPECTED_EMOTIONS = frozenset({
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
})


# ── Progress ───────────────────────────────────────────────────────────────

def load_progress(progress_path: str) -> Dict[str, Any]:
    prog = load_json(progress_path)
    if prog is None:
        return {"last_completed_idx": -1}
    return prog


def save_progress(progress_path: str, last_completed_idx: int) -> None:
    safe_write_json(progress_path, {
        "last_completed_idx": last_completed_idx,
        "updated_at": utc_now_iso(),
    })


# ── JSON parsing robuste ──────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Retire un éventuel bloc markdown ``` autour du JSON."""
    text = text.strip()
    m = _CODEBLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Cas où il n'y a que des ``` en début/fin sans contenu capturé
    if text.startswith("```"):
        lines = text.splitlines()
        # retirer première et dernière ligne si elles sont des ```
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        return "\n".join(lines).strip()
    return text


def try_parse_json(text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Parse la réponse LLM en JSON, en tolérant un wrapper markdown."""
    cleaned = _strip_markdown(text)
    try:
        obj = json.loads(cleaned)
        return True, obj, None
    except Exception as exc:
        return False, None, str(exc)


# ── Validation structurelle ────────────────────────────────────────────────

def validate_annotation(obj: Dict[str, Any]) -> List[str]:
    """
    Vérifie que le JSON parsé respecte le schéma attendu.
    Retourne une liste de warnings (vide = tout est OK).
    """
    warnings: List[str] = []

    if not isinstance(obj, dict):
        return ["root is not a dict"]

    # metadata
    meta = obj.get("metadata")
    if meta is None:
        warnings.append("missing 'metadata'")
    else:
        if meta.get("confidence") not in ("high", "medium", "low"):
            warnings.append(f"unexpected confidence: {meta.get('confidence')}")

    # emotions
    emotions = obj.get("emotions")
    if emotions is None:
        warnings.append("missing 'emotions'")
    else:
        present = set(emotions.keys())
        missing = _EXPECTED_EMOTIONS - present
        extra   = present - _EXPECTED_EMOTIONS
        if missing:
            warnings.append(f"missing emotions: {sorted(missing)}")
        if extra:
            warnings.append(f"unexpected emotions: {sorted(extra)}")
        for k, v in emotions.items():
            if v not in (0, 1):
                warnings.append(f"'{k}' has non-binary value: {v}")

    return warnings


# ── Persistance d'une itération ────────────────────────────────────────────

def persist_iteration(
    out_dir: str,
    run_id: str,
    idx: int,
    row_id: Any,
    prompt: str,
    raw_text: str,
    llm_result: Dict[str, Any],
    parsed_json: Optional[Dict[str, Any]],
    json_ok: bool,
    json_error: Optional[str],
    validation_warnings: Optional[List[str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(out_dir)

    record: Dict[str, Any] = {
        "run_id":               run_id,
        "idx":                  idx,
        "row_id":               row_id,
        "timestamp_utc":        utc_now_iso(),
        "json_ok":              json_ok,
        "json_error":           json_error,
        "validation_warnings":  validation_warnings or [],
        "raw_text":             raw_text,
        "parsed_json":          parsed_json,
        "llm_result":           llm_result,
        "prompt":               prompt,
    }
    if extra_meta:
        record["meta"] = extra_meta

    # 1) Log JSONL append-only (artefact principal)
    append_jsonl(os.path.join(out_dir, f"{run_id}.jsonl"), record)

    # 2) Snapshot individuel temporaire (debug / reprise)
    item_dir = os.path.join(out_dir, "items")
    ensure_dir(item_dir)
    safe_write_json(
        os.path.join(item_dir, f"{run_id}__idx{idx:05d}__id{row_id}.json"),
        record,
    )


def cleanup_items_dir(out_dir: str, run_id: str) -> int:
    """
    Supprime le dossier items/ après génération du JSONL final.
    Retourne le nombre de fichiers supprimés.
    """
    item_dir = os.path.join(out_dir, "items")
    if not os.path.isdir(item_dir):
        return 0
    pattern = os.path.join(item_dir, f"{run_id}__*.json")
    files = glob.glob(pattern)
    count = len(files)
    for f in files:
        os.remove(f)
    # Supprimer le dossier s'il est vide
    try:
        os.rmdir(item_dir)
    except OSError:
        pass  # pas vide (d'autres runs y sont)
    return count
