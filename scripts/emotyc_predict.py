#!/usr/bin/env python3
"""
Script 1/3 — Inférence EMOTYC locale et comparaison au gold label.

Charge le modèle EMOTYC (TextToKids/CamemBERT-base-EmoTextToKids),
applique les prédictions sur chaque ligne du gold label, compare
avec les annotations humaines, et exporte un JSONL de résultats.

Usage :
    python scripts/emotyc_predict.py \
        --xlsx outputs/homophobie/annotations_validees.xlsx \
        --out_dir outputs/homophobie/emotyc_eval

    # Sans seuils optimisés (seuil 0.5 pour tout)
    python scripts/emotyc_predict.py \
        --xlsx outputs/homophobie/annotations_validees.xlsx \
        --out_dir outputs/homophobie/emotyc_eval \
        --no-optimized-thresholds

    # Avec contexte voisin (i-1, i, i+1)
    python scripts/emotyc_predict.py \
        --xlsx outputs/homophobie/annotations_validees.xlsx \
        --out_dir outputs/homophobie/emotyc_eval \
        --use-context
"""

import argparse
import json
import os
import sys
import math

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

MODEL_NAME = "TextToKids/CamemBERT-base-EmoTextToKids"
TOKENIZER_NAME = "camembert-base"

# Les 11 émotions cibles — mapping gold label → EMOTYC model label
GOLD_TO_EMOTYC = {
    "Colère":      "Colere",       # id 9
    "Dégoût":      "Degout",       # id 11
    "Joie":        "Joie",         # id 15
    "Peur":        "Peur",         # id 16
    "Surprise":    "Surprise",     # id 17
    "Tristesse":   "Tristesse",    # id 18
    "Admiration":  "Admiration",   # id 7
    "Culpabilité": "Culpabilite",  # id 10
    "Embarras":    "Embarras",     # id 12
    "Fierté":      "Fierte",       # id 13
    "Jalousie":    "Jalousie",     # id 14
}

# Ordre canonique des 11 émotions (pour affichage cohérent)
EMOTION_ORDER = list(GOLD_TO_EMOTYC.keys())

# Seuils optimisés — issus du notebook retroIngenierie sur un corpus de 2451 phrases
# Template bca_v3 : before:</s>current:{s}</s>after:</s>
# Ces seuils servent à reproduire les outputs du modèle EMOTYC web.
OPTIMIZED_THRESHOLDS = {
    "Admiration":  0.9531926895718311,   # (swap admiration ↔ autre)
    "Colère":      0.28217218720548165,
    "Culpabilité": 0.12671495241969652,
    "Dégoût":      0.19269005632824862,
    "Embarras":    0.9548280448988165,
    "Fierté":      0.8002327448859459,
    "Jalousie":    0.017136900811277365,
    "Joie":        0.9155047132251537,
    "Peur":        0.9881862235180032,
    "Surprise":    0.9722425408373772,
    "Tristesse":   0.6984491339960737,
}

# Mapping EMOTYC label2id (from model config)
EMOTYC_LABEL2ID = {
    "Emo": 0, "Comportementale": 1, "Designee": 2, "Montree": 3,
    "Suggeree": 4, "Base": 5, "Complexe": 6, "Admiration": 7,
    "Autre": 8, "Colere": 9, "Culpabilite": 10, "Degout": 11,
    "Embarras": 12, "Fierte": 13, "Jalousie": 14, "Joie": 15,
    "Peur": 16, "Surprise": 17, "Tristesse": 18,
}

# Index des 11 émotions dans le vecteur de 19 logits
EMOTION_INDICES = {
    gold_name: EMOTYC_LABEL2ID[emotyc_name]
    for gold_name, emotyc_name in GOLD_TO_EMOTYC.items()
}

# Swap admiration ↔ autre (identifié dans retroIngenierie notebook)
SWAP_PAIRS = [(EMOTYC_LABEL2ID["Admiration"], EMOTYC_LABEL2ID["Autre"])]


# ═══════════════════════════════════════════════════════════════════════════
#  UTILS
# ═══════════════════════════════════════════════════════════════════════════

def safe_str(val, default=""):
    """Convertit en string, remplace None/NaN par default."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return str(val)


# ═══════════════════════════════════════════════════════════════════════════
#  MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

def load_model(device=None):
    """Charge le modèle EMOTYC et le tokenizer."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME)
        .to(device)
        .eval()
    )
    print(f"✓ Modèle EMOTYC chargé sur {device}")
    print(f"  {model.config.num_labels} labels, type={model.config.problem_type}")
    return tokenizer, model, device


def format_input(tokenizer, sentence, prev_sentence=None, next_sentence=None,
                 use_context=False):
    """
    Formate l'input selon le meilleur template identifié.
    
    Sans contexte (défaut) : before:</s>current:{s}</s>after:</s>
    Avec contexte          : before:{prev}</s>current:{s}</s>after:{next}</s>
    """
    eos = tokenizer.eos_token  # </s>
    if use_context:
        prev = prev_sentence or eos
        nxt = next_sentence or eos
        return f"before:{prev}{eos}current:{sentence}{eos}after:{nxt}{eos}"
    else:
        return f"before:{eos}current:{sentence}{eos}after:{eos}"


@torch.no_grad()
def predict_batch(tokenizer, model, device, texts, batch_size=16):
    """
    Inférence par batch. Retourne une matrice (N, 19) de probas sigmoid.
    """
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)
        logits = model(**encodings).logits  # (B, 19)

        # Appliquer le swap admiration ↔ autre
        for a, b in SWAP_PAIRS:
            logits[:, [a, b]] = logits[:, [b, a]]

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


# ═══════════════════════════════════════════════════════════════════════════
#  EXTRACTION DES GOLD LABELS
# ═══════════════════════════════════════════════════════════════════════════

def load_gold_labels(xlsx_path):
    """Charge le fichier gold label et extrait les colonnes pertinentes."""
    df = pd.read_excel(xlsx_path)
    print(f"✓ Gold labels : {len(df)} lignes chargées depuis {os.path.basename(xlsx_path)}")

    # Vérifier la présence des colonnes d'émotions
    missing = [e for e in EMOTION_ORDER if e not in df.columns]
    if missing:
        print(f"  ⚠ Colonnes émotions manquantes : {missing}")
        sys.exit(1)

    # Vérifier la colonne TEXT
    text_col = None
    for candidate in ("TEXT", "text", "sentence"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        print("  ✗ Colonne texte non trouvée (TEXT/text/sentence)")
        sys.exit(1)

    print(f"  Colonne texte : '{text_col}'")
    print(f"  Colonnes émotions : {EMOTION_ORDER}")

    return df, text_col


def extract_gold_matrix(df):
    """Extrait la matrice binaire (N, 11) du gold label."""
    gold = np.zeros((len(df), len(EMOTION_ORDER)), dtype=int)
    for j, emo in enumerate(EMOTION_ORDER):
        vals = pd.to_numeric(df[emo], errors="coerce").fillna(0)
        gold[:, j] = (vals >= 0.5).astype(int)
    return gold


# ═══════════════════════════════════════════════════════════════════════════
#  MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(gold, pred):
    """Calcule les métriques par émotion et globales."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        cohen_kappa_score,
    )

    results = []
    for j, emo in enumerate(EMOTION_ORDER):
        g, p = gold[:, j], pred[:, j]
        tp = int(((g == 1) & (p == 1)).sum())
        fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        tn = int(((g == 0) & (p == 0)).sum())

        acc = accuracy_score(g, p)
        try:
            kappa = cohen_kappa_score(g, p)
        except Exception:
            kappa = float("nan")
        f1 = f1_score(g, p, zero_division=0)
        prec = precision_score(g, p, zero_division=0)
        rec = recall_score(g, p, zero_division=0)

        results.append({
            "emotion": emo,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4),
            "kappa": round(kappa, 4) if not math.isnan(kappa) else None,
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "prevalence_gold": round(g.sum() / len(g), 4),
            "prevalence_pred": round(p.sum() / len(p), 4),
        })

    # Métriques globales
    macro_f1 = np.mean([r["f1"] for r in results])
    micro_f1 = f1_score(gold.ravel(), pred.ravel(), zero_division=0)
    exact_match = np.all(gold == pred, axis=1).mean()

    return results, {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "exact_match": round(float(exact_match), 4),
        "n_samples": len(gold),
        "n_emotions": len(EMOTION_ORDER),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Inférence EMOTYC locale et comparaison au gold label",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--xlsx", required=True,
                    help="Chemin vers le fichier gold label (.xlsx)")
    p.add_argument("--out_dir", required=True,
                    help="Dossier de sortie pour les résultats")
    p.add_argument("--use-context", action="store_true",
                    help="Utiliser les phrases voisines (i-1, i+1) comme contexte")
    p.add_argument("--no-optimized-thresholds", action="store_true",
                    help="Utiliser un seuil fixe de 0.5 au lieu des seuils optimisés")
    p.add_argument("--batch-size", type=int, default=16,
                    help="Taille du batch pour l'inférence (défaut: 16)")
    p.add_argument("--device", default=None,
                    help="Device PyTorch (défaut: auto-détection cuda/cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Chargement du gold ─────────────────────────────────────────
    xlsx_path = os.path.abspath(args.xlsx)
    df, text_col = load_gold_labels(xlsx_path)
    gold_matrix = extract_gold_matrix(df)
    sentences = df[text_col].astype(str).tolist()
    N = len(sentences)

    # ── 2. Chargement du modèle ───────────────────────────────────────
    device = torch.device(args.device) if args.device else None
    tokenizer, model, device = load_model(device)

    # ── 3. Préparation des inputs ─────────────────────────────────────
    use_context = args.use_context
    formatted_texts = []
    for i in range(N):
        prev_s = sentences[i - 1] if (i > 0 and use_context) else None
        next_s = sentences[i + 1] if (i < N - 1 and use_context) else None
        formatted_texts.append(
            format_input(tokenizer, sentences[i], prev_s, next_s, use_context)
        )

    template_name = "bca_v3_context" if use_context else "bca_v3_no_context"
    print(f"▸ Template : {template_name}")
    print(f"  Exemple  : {formatted_texts[0][:120]}…")

    # ── 4. Inférence ──────────────────────────────────────────────────
    print(f"\nInférence sur {N} phrases (batch_size={args.batch_size})…")
    all_probs_19 = predict_batch(
        tokenizer, model, device, formatted_texts,
        batch_size=args.batch_size,
    )
    print(f"✓ Inférence terminée — shape: {all_probs_19.shape}")

    # ── 5. Extraction des 11 émotions ─────────────────────────────────
    emotion_probs = np.zeros((N, len(EMOTION_ORDER)), dtype=np.float64)
    for j, emo in enumerate(EMOTION_ORDER):
        idx = EMOTION_INDICES[emo]
        emotion_probs[:, j] = all_probs_19[:, idx]

    # ── 6. Seuils et prédictions binaires ─────────────────────────────
    if args.no_optimized_thresholds:
        thresholds = {emo: 0.5 for emo in EMOTION_ORDER}
        threshold_mode = "fixed_0.5"
        print("▸ Seuils : 0.5 fixe pour toutes les émotions")
    else:
        thresholds = OPTIMIZED_THRESHOLDS
        threshold_mode = "optimized"
        print("▸ Seuils optimisés :")
        for emo in EMOTION_ORDER:
            print(f"    {emo:<15s} : {thresholds[emo]:.6f}")

    threshold_array = np.array([thresholds[emo] for emo in EMOTION_ORDER])
    pred_matrix = (emotion_probs >= threshold_array).astype(int)

    # ── 7. Métriques ──────────────────────────────────────────────────
    per_emotion, global_metrics = compute_metrics(gold_matrix, pred_matrix)

    print(f"\n{'═' * 75}")
    print(f"  MÉTRIQUES PAR ÉMOTION  (seuils: {threshold_mode})")
    print(f"{'═' * 75}")
    print(f"  {'Émotion':<15s} {'Acc':>7s} {'Kappa':>7s} {'F1':>7s} "
          f"{'Prec':>7s} {'Recall':>7s} {'FP':>5s} {'FN':>5s}")
    print(f"  {'-' * 68}")
    for r in per_emotion:
        k_str = f"{r['kappa']:.3f}" if r['kappa'] is not None else "  N/A"
        print(f"  {r['emotion']:<15s} {r['accuracy']:>7.3f} {k_str:>7s} "
              f"{r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['fp']:>5d} {r['fn']:>5d}")
    print(f"  {'-' * 68}")
    print(f"  Macro-F1    : {global_metrics['macro_f1']:.4f}")
    print(f"  Micro-F1    : {global_metrics['micro_f1']:.4f}")
    print(f"  Exact Match : {global_metrics['exact_match']:.4f}")
    print(f"{'═' * 75}")

    # ── 8. Export JSONL ───────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "emotyc_predictions.jsonl")

    n_divergent = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(N):
            # Identifier les divergences
            divergences = []
            for j, emo in enumerate(EMOTION_ORDER):
                g = int(gold_matrix[i, j])
                p = int(pred_matrix[i, j])
                if g != p:
                    div_type = "faux_positif" if p == 1 else "faux_negatif"
                    divergences.append({
                        "emotion": emo,
                        "gold": g,
                        "pred": p,
                        "proba": round(float(emotion_probs[i, j]), 6),
                        "seuil": round(float(threshold_array[j]), 6),
                        "type_divergence": div_type,
                    })

            if divergences:
                n_divergent += 1

            # Contexte textuel
            prev_text = sentences[i - 1] if i > 0 else None
            next_text = sentences[i + 1] if i < N - 1 else None

            record = {
                "idx": i,
                "id": safe_str(df.iloc[i].get("ID", i)),
                "text": sentences[i],
                "text_prev": prev_text,
                "text_next": next_text,
                "template_used": template_name,
                "threshold_mode": threshold_mode,
                "probas": {
                    emo: round(float(emotion_probs[i, j]), 6)
                    for j, emo in enumerate(EMOTION_ORDER)
                },
                "preds": {
                    emo: int(pred_matrix[i, j])
                    for j, emo in enumerate(EMOTION_ORDER)
                },
                "golds": {
                    emo: int(gold_matrix[i, j])
                    for j, emo in enumerate(EMOTION_ORDER)
                },
                "n_divergences": len(divergences),
                "divergences": divergences,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✓ Résultats exportés : {out_path}")
    print(f"  {N} lignes, {n_divergent} avec ≥1 divergence")

    # ── 9. Export du résumé JSON ──────────────────────────────────────
    summary = {
        "source_xlsx": os.path.basename(xlsx_path),
        "n_samples": N,
        "n_divergent_rows": n_divergent,
        "template": template_name,
        "threshold_mode": threshold_mode,
        "thresholds": {emo: round(thresholds[emo], 6) for emo in EMOTION_ORDER},
        "per_emotion": per_emotion,
        "global_metrics": global_metrics,
    }
    summary_path = os.path.join(args.out_dir, "emotyc_predictions_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Résumé : {summary_path}")


if __name__ == "__main__":
    main()
