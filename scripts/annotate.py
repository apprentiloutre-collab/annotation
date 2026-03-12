#!/usr/bin/env python3
"""
Script d'annotation automatisée des émotions via LLM.

Usage (exemples) :
    # Bedrock Claude Sonnet (défaut)
    python scripts/annotate.py \
        --xlsx data/homophobie_scenario_julie.xlsx \
        --thematique homophobie \
        --run_id homophobie_run001

    # Bedrock Mistral Pixtral
    python scripts/annotate.py \
        --xlsx data/mon_fichier.xlsx \
        --thematique racisme \
        --run_id racisme_run001 \
        --model_provider bedrock \
        --model mistral-pixtral

    # HuggingFace DeepSeek
    python scripts/annotate.py \
        --xlsx data/mon_fichier.xlsx \
        --thematique sexisme \
        --run_id sexisme_run001 \
        --model_provider huggingface \
        --model "deepseek-ai/DeepSeek-V3.2:novita"

    # Gemini Flash (Colab uniquement)
    python scripts/annotate.py \
        --xlsx data/mon_fichier.xlsx \
        --thematique homophobie \
        --run_id homophobie_gemini_run001 \
        --model_provider gemini

    # Avec annotations d'experts dans le prompt
    python scripts/annotate.py \
        --xlsx data/mon_fichier.xlsx \
        --thematique homophobie \
        --run_id homophobie_run002 \
        --use_annotations
"""

import argparse
import os
import sys
import time
import logging

logging.basicConfig(level=logging.WARNING)

# ── Résolution du chemin du repo ──────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pandas as pd
from cyberagg_llm_annot.llm_providers import get_provider
from cyberagg_llm_annot.context import get_message_window, minimal_msg_repr
from cyberagg_llm_annot.parsing import extract_row_labels
from cyberagg_llm_annot.prompt_utils import (
    SYSTEM_PROMPT, DEFAULT_LABEL_COLS,
    build_annotations_block, build_user_message,
)
from cyberagg_llm_annot.runner import (
    load_progress, save_progress, try_parse_json,
    validate_annotation, persist_iteration, cleanup_items_dir,
)
from cyberagg_llm_annot.io_utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Annotation automatisée des émotions via LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── Données ──
    p.add_argument("--xlsx", required=True,
                    help="Chemin vers le fichier XLSX à annoter")
    p.add_argument("--thematique", required=True,
                    help="Thématique du scénario (ex: homophobie)")
    p.add_argument("--run_id", required=True,
                    help="Identifiant unique du run (ex: homophobie_run001)")

    # ── Sorties ──
    p.add_argument("--out_dir", default=None,
                    help="Dossier de sortie (défaut: outputs/<thematique>)")

    # ── Modèle ──
    p.add_argument("--model_provider", default="bedrock",
                    choices=["bedrock", "gemini", "huggingface"],
                    help="Provider LLM (défaut: bedrock)")
    p.add_argument("--model", default="claude-sonnet-4-6",
                    help="Modèle à utiliser (défaut: claude-sonnet-4-6)")

    # ── Annotations d'experts ──
    p.add_argument("--use_annotations", action="store_true",
                    help="Intégrer les annotations d'experts dans le prompt")

    # ── Paramètres LLM ──
    p.add_argument("--max_tokens", type=int, default=512,
                    help="Nombre max de tokens en réponse (défaut: 512)")
    p.add_argument("--delay", type=float, default=1.0,
                    help="Délai entre appels en secondes (défaut: 1.0)")

    # ── Bedrock spécifique ──
    p.add_argument("--region", default="eu-north-1",
                    help="Région AWS Bedrock (défaut: eu-north-1)")

    # ── HuggingFace spécifique ──
    p.add_argument("--hf_token_env", default="HF_TOKEN",
                    help="Variable d'env. pour le token HF (défaut: HF_TOKEN)")

    return p.parse_args()


def main():
    args = parse_args()

    # ═══ 1. CHARGEMENT ═════════════════════════════════════════════════════
    xlsx_path = os.path.abspath(args.xlsx)
    df = pd.read_excel(xlsx_path)
    print(f"✓ {len(df)} lignes chargées depuis {os.path.basename(xlsx_path)}")

    # ═══ 2. CONFIGURATION ═════════════════════════════════════════════════
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "outputs", args.thematique)
    progress_path = os.path.join(out_dir, f"{args.run_id}__progress.json")
    ensure_dir(out_dir)

    # ═══ 3. PROVIDER LLM ══════════════════════════════════════════════════
    provider_kwargs = {}
    if args.model_provider == "bedrock":
        provider_kwargs["region_name"] = args.region
    elif args.model_provider == "huggingface":
        token = os.environ.get(args.hf_token_env)
        if token:
            provider_kwargs["hf_token"] = token

    provider = get_provider(args.model_provider, args.model, **provider_kwargs)
    print(f"✓ Provider : {args.model_provider} / {args.model}")

    # ═══ 4. REPRISE ═══════════════════════════════════════════════════════
    progress = load_progress(progress_path)
    start_idx = progress["last_completed_idx"] + 1
    total = len(df)
    print(f"▸ Reprise à idx={start_idx} / {total}")

    # ═══ 5. BOUCLE PRINCIPALE ═════════════════════════════════════════════
    errors = 0
    t0 = time.time()

    for idx in range(start_idx, total):
        # ── Fenêtre prev / target / next ──
        w = get_message_window(df, idx)
        prev_repr   = minimal_msg_repr(w["prev"])
        target_repr = minimal_msg_repr(w["target"])
        next_repr   = minimal_msg_repr(w["next"])
        row_dict    = w["target"]

        # ── Annotations (conditionnel via --use_annotations) ──
        annotations_block = None
        if args.use_annotations:
            parsed_labels = extract_row_labels(row_dict, DEFAULT_LABEL_COLS)
            annotations_block = build_annotations_block(parsed_labels)

        # ── Construction du message user ──
        user_message = build_user_message(
            thematique=args.thematique,
            prev_repr=prev_repr,
            target_repr=target_repr,
            next_repr=next_repr,
            annotations_block=annotations_block,
        )

        # ── Appel LLM ──
        llm_result = provider.invoke(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )

        raw_text = provider.extract_text(llm_result)
        is_complete, stop_reason = provider.check_stop_reason(llm_result)

        # ── Parse + validation ──
        json_ok, parsed_obj, json_error = try_parse_json(raw_text)

        validation_warnings = []
        if json_ok and parsed_obj:
            validation_warnings = validate_annotation(parsed_obj)
        if not is_complete:
            validation_warnings.append(
                f"stop_reason={stop_reason} (troncature probable)"
            )

        # ── Persistance immédiate ──
        row_id = row_dict.get("ID", idx)
        full_prompt_log = f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_message}"

        persist_iteration(
            out_dir=out_dir,
            run_id=args.run_id,
            idx=idx,
            row_id=row_id,
            prompt=full_prompt_log,
            raw_text=raw_text,
            llm_result=llm_result,
            parsed_json=parsed_obj,
            json_ok=json_ok,
            json_error=json_error,
            validation_warnings=validation_warnings,
            extra_meta={
                "thematique":      args.thematique,
                "model_provider":  args.model_provider,
                "model":           args.model,
                "stop_reason":     stop_reason,
                "target_role":     row_dict.get("ROLE"),
                "target_name":     row_dict.get("NAME"),
                "use_annotations": args.use_annotations,
            },
        )

        # ── Progression (après sauvegarde !) ──
        save_progress(progress_path, last_completed_idx=idx)
        if not json_ok:
            errors += 1

        # ── Monitoring ──
        done    = idx - start_idx + 1
        remain  = total - idx - 1
        elapsed = time.time() - t0
        avg     = elapsed / done
        eta     = avg * remain

        status = "✓" if (json_ok and not validation_warnings) else "⚠" if json_ok else "✗"
        print(
            f"[{status}] {idx}/{total-1}  ID={row_id}  "
            f"json={json_ok}  stop={stop_reason}  "
            f"warn={len(validation_warnings)}  err={errors}  "
            f"ETA={eta/60:.1f}min"
        )

        time.sleep(args.delay)

    # ═══ 6. NETTOYAGE items/ ═══════════════════════════════════════════════
    n_cleaned = cleanup_items_dir(out_dir, args.run_id)
    if n_cleaned:
        print(f"🧹 {n_cleaned} fichiers temporaires supprimés (items/)")

    # ═══ 7. RÉSUMÉ ═════════════════════════════════════════════════════════
    elapsed_total = time.time() - t0
    processed = total - start_idx
    print(f"\n{'='*60}")
    print(f"Terminé. {processed} items traités en {elapsed_total/60:.1f} min")
    print(f"Erreurs JSON : {errors}/{processed}")
    print(f"Sorties dans : {out_dir}")
    print(f"JSONL final  : {os.path.join(out_dir, args.run_id + '.jsonl')}")


if __name__ == "__main__":
    main()
