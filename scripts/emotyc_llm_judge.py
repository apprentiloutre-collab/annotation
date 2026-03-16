#!/usr/bin/env python3
"""
Script 2/3 — Annotation LLM des erreurs d'EMOTYC via Claude Sonnet 4.6.

Lit le fichier emotyc_predictions.jsonl produit par emotyc_predict.py,
puis effectue 2 passes d'annotation LLM sur les lignes divergentes :

  Passe 1 (Double-blind) : Le LLM juge entre "Annotateur A" et "Annotateur B"
  sans savoir qui est le gold, qui est EMOTYC → mesure de fiabilité.

  Passe 2 (Diagnostic) : Le LLM reçoit les probas EMOTYC + gold labels et
  produit une analyse pragmatique + classification d'erreur.

Usage :
    python scripts/emotyc_llm_judge.py \
        --predictions outputs/homophobie/emotyc_eval/emotyc_predictions.jsonl \
        --out_dir outputs/homophobie/emotyc_eval

    # Passe 1 uniquement
    python scripts/emotyc_llm_judge.py \
        --predictions outputs/homophobie/emotyc_eval/emotyc_predictions.jsonl \
        --out_dir outputs/homophobie/emotyc_eval \
        --pass blind

    # Passe 2 uniquement
    python scripts/emotyc_llm_judge.py \
        --predictions outputs/homophobie/emotyc_eval/emotyc_predictions.jsonl \
        --out_dir outputs/homophobie/emotyc_eval \
        --pass diagnostic
"""

import argparse
import json
import os
import random
import re
import sys
import time
import logging

logging.basicConfig(level=logging.WARNING)

# ── Résolution du chemin du repo ──────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from cyberagg_llm_annot.llm_providers import get_provider
from cyberagg_llm_annot.io_utils import ensure_dir, append_jsonl, utc_now_iso

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

EMOTION_ORDER = [
    "Admiration", "Colère", "Culpabilité", "Dégoût", "Embarras",
    "Fierté", "Jalousie", "Joie", "Peur", "Surprise", "Tristesse",
]

# Taxonomie fixe des types d'erreurs
ERROR_TAXONOMY = [
    "lexical_argot",
    "ironie_polarity",
    "pragmatic_confusion",
    "seuil_limite",
    "erreur_humain",
    "contexte_manquant",
    "autre",
]

RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_BLIND = """\
Tu es un linguiste expert, spécialiste en sociolinguistique (langage des 11-25 ans) \
et en pragmatique des émotions. Tu interviens comme juge impartial et intraitable.

CONTEXTE DU CORPUS : Ce sont des messages échangés par des adolescents (11-18 ans) \
dans des conversations en ligne en français. Les messages se suivent chronologiquement \
(format groupe/chat), mais ne sont pas toujours des réponses directes au message précédent. \
Le corpus traite de thématiques de cyber-harcèlement, contient de l'argot, des abréviations, \
de l'ironie et du langage coloré.

On s'intéresse UNIQUEMENT à la détection de ces 11 émotions :
[Admiration, Colère, Culpabilité, Dégoût, Embarras, Fierté, Jalousie, Joie, Peur, Surprise, Tristesse].

Tu dois être STRICT et TRANCHÉ. Ne cherche PAS d'excuses aux annotateurs. \
Si une annotation est fausse, dis-le clairement."""

SYSTEM_PROMPT_DIAGNOSTIC = """\
Tu es un linguiste expert, spécialiste en sociolinguistique (langage des 11-25 ans) \
et en pragmatique des émotions. Tu es un diagnosticien implacable des erreurs \
de modèles de classification d'émotions.

CONTEXTE DU CORPUS : Ce sont des messages échangés par des adolescents (11-18 ans) \
dans des conversations en ligne en français. Les messages se suivent chronologiquement \
(format groupe/chat), mais ne sont pas toujours des réponses directes au message précédent. \
Le corpus traite de thématiques de cyber-harcèlement.

On s'intéresse à ces 11 émotions :
[Admiration, Colère, Culpabilité, Dégoût, Embarras, Fierté, Jalousie, Joie, Peur, Surprise, Tristesse].

TAXONOMIE D'ERREURS — tu dois choisir EXACTEMENT parmi :
- "lexical_argot" : erreur due au vocabulaire / tokenizer / argot (ex: « seum » mal interprété)
- "ironie_polarity" : phrase ironique / négation sémantique mal prise en compte
- "pragmatic_confusion" : confusion entre émotion ressentie, provoquée et thématisée
- "seuil_limite" : score proche du seuil de décision
- "erreur_humain" : l'annotation humaine semble incorrecte, la machine a raison
- "contexte_manquant" : l'émotion nécessite plus de contexte que disponible
- "autre" : cas non couvert

AXES PRAGMATIQUES — pour chaque émotion, classe-la selon :
- "ressentie" : l'énonciateur RESSENT sincèrement cette émotion
- "provoquee" : l'énonciateur cherche à PROVOQUER cette émotion chez la cible
- "thematisee" : l'énonciateur PARLE de cette émotion comme concept
- "absent" : l'émotion n'est pas présente dans le texte

Sois STRICT. Ne cherche PAS d'excuses au modèle."""


def build_blind_user_message(record, rng):
    """
    Construit le prompt double-blind.
    Randomise l'attribution A/B (gold vs EMOTYC).
    """
    text = record["text"]
    text_prev = record.get("text_prev") or ""
    text_next = record.get("text_next") or ""

    # Préparer les annotations divergentes
    golds = record["golds"]
    preds = record["preds"]

    # Ne montrer que les émotions divergentes + quelques concordantes pour contexte
    annot_gold = {emo: golds[emo] for emo in EMOTION_ORDER}
    annot_pred = {emo: preds[emo] for emo in EMOTION_ORDER}

    # Randomiser A/B
    gold_is_a = rng.random() < 0.5

    if gold_is_a:
        annot_a, annot_b = annot_gold, annot_pred
    else:
        annot_a, annot_b = annot_pred, annot_gold

    # Formatter les annotations
    def fmt_annot(annot):
        present = [e for e in EMOTION_ORDER if annot[e] == 1]
        absent = [e for e in EMOTION_ORDER if annot[e] == 0]
        if present:
            return f"Émotions PRÉSENTES : {present}\nÉmotions ABSENTES : {absent}"
        return f"Aucune émotion détectée (toutes absentes)."

    eos = "</s>"
    msg = f"""Voici un extrait de texte à analyser, formaté avec son contexte :
<texte>
before:{text_prev or eos}{eos if not text_prev else ''} current:{text}{eos} after:{text_next or eos}{eos if not text_next else ''}
</texte>

Deux annotateurs indépendants (Annotateur A et Annotateur B) ont évalué les émotions \
présentes dans la phrase "current". L'un d'eux peut être une machine, l'autre un humain, \
ou les deux peuvent se tromper. Tu ne dois faire AUCUNE supposition sur leur identité.

<annotations>
Annotateur A :
{fmt_annot(annot_a)}

Annotateur B :
{fmt_annot(annot_b)}
</annotations>

TA TÂCHE :
Dans une balise <raisonnement>, effectue ton analyse étape par étape :
1. Analyse lexicale et argotique : argot, ironie, abréviations qui pourraient tromper ?
2. Analyse pragmatique de "current" (avec contexte before/after). Classe les émotions :
   - RESSENTIE : l'énonciateur ressent-il sincèrement cette émotion ?
   - PROVOQUÉE : cherche-t-il à susciter cette émotion chez la cible ?
   - THÉMATISÉE : parle-t-il de l'émotion comme d'un concept ?
3. Évaluation critique : qui a raison, qui a tort ? (les deux peuvent avoir tort/raison)
4. Nature de l'erreur s'il y a divergence.

Ensuite, fournis UNIQUEMENT un objet JSON valide dans une balise <json> :
{{
  "argot_present": true/false,
  "mots_argotiques": ["mot1", "mot2"],
  "analyse_pragmatique": {{
       "emotions_ressenties": ["Emotion1"],
       "emotions_provoquees": [],
       "emotions_thematisees": []
  }},
  "verdict_A": "Correct" | "Faux positif" | "Faux négatif" | "Partiellement correct",
  "verdict_B": "Correct" | "Faux positif" | "Faux négatif" | "Partiellement correct",
  "type_erreur_constatee": "lexical_argot" | "ironie_polarity" | "pragmatic_confusion" | "seuil_limite" | "erreur_humain" | "contexte_manquant" | "autre" | "aucune",
  "justification_stricte": "Explication brève et tranchée (2-3 phrases max)."
}}"""

    return msg, gold_is_a


def build_diagnostic_user_message(record):
    """
    Construit le prompt diagnostic pour les émotions divergentes.
    Révèle explicitement EMOTYC vs gold + probas.
    """
    text = record["text"]
    text_prev = record.get("text_prev") or ""
    text_next = record.get("text_next") or ""
    divergences = record["divergences"]
    probas = record["probas"]
    golds = record["golds"]
    preds = record["preds"]

    # Formatter chaque divergence
    div_lines = []
    for d in divergences:
        emo = d["emotion"]
        div_lines.append(
            f"  - {emo} : gold={d['gold']}, EMOTYC={d['pred']}, "
            f"proba_sigmoid={d['proba']:.4f}, seuil={d['seuil']:.4f}, "
            f"type={d['type_divergence']}"
        )
    div_block = "\n".join(div_lines)

    # Prédictions complètes EMOTYC
    pred_lines = []
    for emo in EMOTION_ORDER:
        p = probas[emo]
        g = golds[emo]
        flag = " ← DIVERGENCE" if preds[emo] != g else ""
        pred_lines.append(f"  {emo:<15s} : proba={p:.4f}, pred={preds[emo]}, gold={g}{flag}")
    pred_block = "\n".join(pred_lines)

    msg = f"""Analyse la phrase suivante issue d'un corpus de cyber-harcèlement entre adolescents :

<texte>
Contexte précédent : "{text_prev or '(aucun)'}"
Phrase à analyser  : "{text}"
Contexte suivant   : "{text_next or '(aucun)'}"
</texte>

Un modèle de classification d'émotions (EMOTYC, basé sur CamemBERT) a produit les prédictions suivantes.
Les annotations gold (vérité terrain humaine) sont aussi indiquées.

<predictions_completes>
{pred_block}
</predictions_completes>

<divergences>
{div_block}
</divergences>

Pour CHAQUE divergence listée ci-dessus, produis une analyse.

Fournis UNIQUEMENT un objet JSON valide dans une balise <json> avec cette structure :
{{
  "argot_present": true/false,
  "mots_argotiques": ["mot1"],
  "verdicts": [
    {{
      "emotion": "NomEmotion",
      "gold": 0 ou 1,
      "pred_binaire": 0 ou 1,
      "pred_proba": 0.1234,
      "type_divergence": "faux_positif" | "faux_negatif",
      "axe_pragmatique": "ressentie" | "provoquee" | "thematisee" | "absent",
      "type_erreur": "lexical_argot" | "ironie_polarity" | "pragmatic_confusion" | "seuil_limite" | "erreur_humain" | "contexte_manquant" | "autre",
      "qui_a_raison": "gold" | "emotyc" | "indecidable",
      "justification": "2 phrases max."
    }}
  ]
}}"""

    return msg


# ═══════════════════════════════════════════════════════════════════════════
#  PARSING
# ═══════════════════════════════════════════════════════════════════════════

_JSON_TAG_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.DOTALL)
_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)


def extract_json_from_response(text):
    """Extrait un objet JSON de la réponse LLM (depuis <json> ou ```json)."""
    # Essayer <json>...</json>
    m = _JSON_TAG_RE.search(text)
    if m:
        try:
            return True, json.loads(m.group(1)), None
        except json.JSONDecodeError as e:
            return False, None, f"JSON dans <json> invalide: {e}"

    # Essayer ```json...```
    m = _CODEBLOCK_RE.search(text)
    if m:
        try:
            return True, json.loads(m.group(1)), None
        except json.JSONDecodeError as e:
            return False, None, f"JSON dans codeblock invalide: {e}"

    # Essayer le texte brut
    text_stripped = text.strip()
    # Chercher le premier { et le dernier }
    start = text_stripped.find("{")
    end = text_stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return True, json.loads(text_stripped[start:end + 1]), None
        except json.JSONDecodeError as e:
            return False, None, f"JSON brut invalide: {e}"

    return False, None, "Aucun JSON trouvé dans la réponse"


# ═══════════════════════════════════════════════════════════════════════════
#  PROGRESS
# ═══════════════════════════════════════════════════════════════════════════

def load_completed_indices(jsonl_path):
    """Charge les indices déjà traités depuis un JSONL existant."""
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["idx"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Annotation LLM des erreurs d'EMOTYC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--predictions", required=True,
                    help="Chemin vers emotyc_predictions.jsonl")
    p.add_argument("--out_dir", required=True,
                    help="Dossier de sortie")
    p.add_argument("--pass", dest="pass_mode", default="both",
                    choices=["blind", "diagnostic", "both"],
                    help="Quelle passe exécuter (défaut: both)")
    p.add_argument("--model", default="claude-sonnet-4-6",
                    help="Modèle Bedrock (défaut: claude-sonnet-4-6)")
    p.add_argument("--region", default="eu-north-1",
                    help="Région AWS Bedrock (défaut: eu-north-1)")
    p.add_argument("--max-tokens", type=int, default=1024,
                    help="Max tokens par réponse LLM (défaut: 1024)")
    p.add_argument("--delay", type=float, default=1.0,
                    help="Délai entre appels API en secondes (défaut: 1.0)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help=f"Seed pour randomisation A/B (défaut: {RANDOM_SEED})")
    return p.parse_args()


def load_predictions(path):
    """Charge le JSONL de prédictions et filtre les divergentes."""
    all_records = []
    divergent = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            all_records.append(rec)
            if rec["n_divergences"] > 0:
                divergent.append(rec)
    return all_records, divergent


def run_pass(
    pass_name,
    records,
    provider,
    build_msg_fn,
    system_prompt,
    out_path,
    max_tokens,
    delay,
    rng=None,
):
    """Exécute une passe d'annotation LLM."""
    done_indices = load_completed_indices(out_path)
    todo = [r for r in records if r["idx"] not in done_indices]

    print(f"\n{'═' * 65}")
    print(f"  PASSE : {pass_name}")
    print(f"  Total divergents : {len(records)}")
    print(f"  Déjà traités     : {len(done_indices)}")
    print(f"  À traiter        : {len(todo)}")
    print(f"{'═' * 65}")

    if not todo:
        print("  ✓ Tout est déjà traité.")
        return

    errors = 0
    t0 = time.time()

    for i, rec in enumerate(todo):
        idx = rec["idx"]

        # Construire le message
        if pass_name == "blind":
            user_msg, gold_is_a = build_msg_fn(rec, rng)
            extra = {"gold_is_a": gold_is_a}
        else:
            user_msg = build_msg_fn(rec)
            extra = {}

        # Appel LLM
        try:
            llm_result = provider.invoke(
                system_prompt=system_prompt,
                user_message=user_msg,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            raw_text = provider.extract_text(llm_result)
            is_complete, stop_reason = provider.check_stop_reason(llm_result)
        except Exception as exc:
            print(f"  ✗ idx={idx} — Erreur API : {exc}")
            errors += 1
            time.sleep(delay * 2)
            continue

        # Parser le JSON
        json_ok, parsed_json, json_error = extract_json_from_response(raw_text)

        # Construire le record de sortie
        output_record = {
            "idx": idx,
            "id": rec.get("id"),
            "text": rec["text"],
            "pass": pass_name,
            "timestamp_utc": utc_now_iso(),
            "json_ok": json_ok,
            "json_error": json_error,
            "stop_reason": stop_reason,
            "is_complete": is_complete,
            "raw_response": raw_text,
            "parsed_json": parsed_json,
            "n_divergences": rec["n_divergences"],
            "divergences": rec["divergences"],
            "golds": rec["golds"],
            "preds": rec["preds"],
            "probas": rec["probas"],
        }
        output_record.update(extra)

        # Sauvegarder
        append_jsonl(out_path, output_record)

        # Monitoring
        done = i + 1
        remain = len(todo) - done
        elapsed = time.time() - t0
        avg = elapsed / done
        eta = avg * remain

        if not json_ok:
            errors += 1
        status = "✓" if json_ok else "✗"

        print(
            f"  [{status}] {done}/{len(todo)}  idx={idx}  "
            f"json={json_ok}  stop={stop_reason}  "
            f"err={errors}  ETA={eta/60:.1f}min"
        )

        time.sleep(delay)

    elapsed_total = time.time() - t0
    print(f"\n  Passe '{pass_name}' terminée : {len(todo)} items en {elapsed_total/60:.1f}min")
    print(f"  Erreurs JSON : {errors}/{len(todo)}")


def main():
    args = parse_args()

    # ── Chargement ────────────────────────────────────────────────────
    pred_path = os.path.abspath(args.predictions)
    all_records, divergent = load_predictions(pred_path)
    print(f"✓ {len(all_records)} prédictions chargées, {len(divergent)} divergentes")

    if not divergent:
        print("Aucune divergence détectée — rien à annoter.")
        return

    ensure_dir(args.out_dir)

    # ── Provider LLM ──────────────────────────────────────────────────
    provider = get_provider("bedrock", args.model, region_name=args.region)
    print(f"✓ Provider : bedrock / {args.model}")

    rng = random.Random(args.seed)

    # ── Passe 1 : Double-blind ────────────────────────────────────────
    if args.pass_mode in ("blind", "both"):
        blind_path = os.path.join(args.out_dir, "emotyc_judge_blind.jsonl")
        run_pass(
            pass_name="blind",
            records=divergent,
            provider=provider,
            build_msg_fn=build_blind_user_message,
            system_prompt=SYSTEM_PROMPT_BLIND,
            out_path=blind_path,
            max_tokens=args.max_tokens,
            delay=args.delay,
            rng=rng,
        )

    # ── Passe 2 : Diagnostic ─────────────────────────────────────────
    if args.pass_mode in ("diagnostic", "both"):
        diag_path = os.path.join(args.out_dir, "emotyc_judge_diagnostic.jsonl")
        run_pass(
            pass_name="diagnostic",
            records=divergent,
            provider=provider,
            build_msg_fn=build_diagnostic_user_message,
            system_prompt=SYSTEM_PROMPT_DIAGNOSTIC,
            out_path=diag_path,
            max_tokens=args.max_tokens,
            delay=args.delay,
        )

    print(f"\n{'═' * 65}")
    print(f"  TERMINÉ — Résultats dans : {args.out_dir}")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
