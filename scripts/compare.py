#!/usr/bin/env python3
"""
Comparaison inter-runs d'annotations LLM.

Usage :
    python scripts/compare.py \
        --run1 outputs/homophobie/run001.jsonl \
        --run2 outputs/homophobie/run002.jsonl \
        --xlsx data/homophobie_scenario_julie.xlsx
"""

import argparse, json, os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

sns.set_theme(style="whitegrid", font_scale=1.05)

EMOTIONS = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
]


def parse_args():
    p = argparse.ArgumentParser(description="Comparaison inter-runs")
    p.add_argument("--run1", required=True, help="JSONL du run 1")
    p.add_argument("--run2", required=True, help="JSONL du run 2")
    p.add_argument("--xlsx", default=None, help="XLSX original (textes)")
    p.add_argument("--out_dir", default=None, help="Dossier de sortie")
    p.add_argument("--label_run1", default="run1", help="Label run 1")
    p.add_argument("--label_run2", default="run2", help="Label run 2")
    return p.parse_args()


def load_emotions_from_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = {"idx": rec["idx"], "row_id": rec.get("row_id"),
                   "json_ok": rec.get("json_ok", False)}
            pj = rec.get("parsed_json")
            if rec.get("json_ok") and isinstance(pj, dict):
                emo = pj.get("emotions", {})
                for e in EMOTIONS:
                    row[e] = emo.get(e, np.nan)
                row["confidence"] = pj.get("metadata", {}).get("confidence")
                row["rationale"]  = pj.get("rationale_short")
            else:
                for e in EMOTIONS:
                    row[e] = np.nan
                row["confidence"] = None
                row["rationale"]  = None
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.run1)), "comparaison_runs")
    os.makedirs(out_dir, exist_ok=True)
    lr1, lr2 = args.label_run1, args.label_run2

    df1 = load_emotions_from_jsonl(args.run1)
    df2 = load_emotions_from_jsonl(args.run2)
    print(f"Run 1 ({lr1}): {len(df1)} lignes  |  Run 2 ({lr2}): {len(df2)} lignes")

    df_text = None
    if args.xlsx and os.path.exists(args.xlsx):
        df_text = pd.read_excel(args.xlsx).reset_index(drop=True)

    merged = pd.merge(df1, df2, on="idx", how="inner", suffixes=("_r1", "_r2"))
    merged = merged[merged["json_ok_r1"] & merged["json_ok_r2"]].copy()
    N = len(merged)
    print(f"✓ {N} messages comparables")
    if N == 0:
        print("⚠ Aucun message comparable"); return

    # ── Métriques par émotion ──
    stats_rows = []
    for e in EMOTIONS:
        v1 = merged[f"{e}_r1"].values.astype(int)
        v2 = merged[f"{e}_r2"].values.astype(int)
        agree = (v1 == v2).sum()
        r1_only = ((v1 == 1) & (v2 == 0)).sum()
        r2_only = ((v1 == 0) & (v2 == 1)).sum()
        try: kappa = cohen_kappa_score(v1, v2)
        except: kappa = np.nan
        stats_rows.append({
            "emotion": e, "accord_pct": round(agree/N*100, 1),
            "kappa": round(kappa, 3) if not np.isnan(kappa) else "N/A",
            "run1_seul": r1_only, "run2_seul": r2_only,
            "total_divergences": r1_only + r2_only,
        })
    df_stats = pd.DataFrame(stats_rows)
    print(f"\n{'='*70}\n  ACCORD PAR ÉMOTION ({lr1} vs {lr2})\n{'='*70}")
    print(df_stats.to_string(index=False))

    # ── Exact match ──
    exact = [all(row[f"{e}_r1"]==row[f"{e}_r2"] for e in EMOTIONS)
             for _, row in merged.iterrows()]
    merged["exact_match"] = exact
    ndiv = [sum(1 for e in EMOTIONS if row[f"{e}_r1"]!=row[f"{e}_r2"])
            for _, row in merged.iterrows()]
    merged["n_divergences"] = ndiv
    n_exact = sum(exact)
    print(f"\nExact match: {n_exact}/{N} ({n_exact/N*100:.1f}%)")

    # ── Figures ──
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2ecc71" if v>=90 else "#f39c12" if v>=80 else "#e74c3c"
              for v in df_stats["accord_pct"]]
    bars = ax.barh(df_stats["emotion"][::-1], df_stats["accord_pct"][::-1],
                   color=colors[::-1], edgecolor="black", linewidth=0.5)
    ax.axvline(90, color="grey", ls="--", lw=0.8, label="90%")
    ax.set_xlabel("Accord (%)"); ax.set_xlim(0, 105)
    ax.set_title(f"Accord inter-runs ({lr1} vs {lr2})"); ax.legend()
    for bar, val in zip(bars, df_stats["accord_pct"][::-1]):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accord_par_emotion.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(EMOTIONS)); w = 0.35
    ax.barh(x+w/2, df_stats["run1_seul"], w, label=f"{lr1} seul", color="#3498db")
    ax.barh(x-w/2, df_stats["run2_seul"], w, label=f"{lr2} seul", color="#e67e22")
    ax.set_yticks(x); ax.set_yticklabels(df_stats["emotion"])
    ax.set_xlabel("Nb messages"); ax.set_title("Divergences directionnelles")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "divergences_direction.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(merged["n_divergences"], bins=range(0, merged["n_divergences"].max()+2),
            align="left", color="#9b59b6", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Nb émotions divergentes"); ax.set_ylabel("Nb messages")
    ax.set_title("Distribution des divergences"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_divergences.png"), dpi=150)
    plt.close("all")
    print(f"✓ Figures sauvegardées dans {out_dir}/")

    # ── Exemples divergences ──
    df_div = merged[merged["n_divergences"]>0].sort_values("n_divergences", ascending=False)
    examples = []
    for _, row in df_div.iterrows():
        idx = int(row["idx"])
        text, name, role = "N/A", "?", "?"
        if df_text is not None and idx < len(df_text):
            text = str(df_text.iloc[idx].get("TEXT", "N/A"))
            name = str(df_text.iloc[idx].get("NAME", "?"))
            role = str(df_text.iloc[idx].get("ROLE", "?"))
        dd = [{"emotion": e, "r1": int(row[f"{e}_r1"]), "r2": int(row[f"{e}_r2"])}
              for e in EMOTIONS if row[f"{e}_r1"] != row[f"{e}_r2"]]
        examples.append({"idx": idx, "row_id": row.get("row_id_r1", idx),
            "name": name, "role": role, "text": text,
            "n_divergences": int(row["n_divergences"]), "divergences": dd,
            "rationale_r1": row.get("rationale_r1"), "rationale_r2": row.get("rationale_r2"),
            "confidence_r1": row.get("confidence_r1"), "confidence_r2": row.get("confidence_r2")})

    for ex in examples[:15]:
        print(f"\n  idx={ex['idx']}  [{ex['name']}] role={ex['role']} div={ex['n_divergences']}")
        print(f"  TEXT: \"{ex['text'][:120]}\"")
        for d in ex["divergences"]:
            print(f"    {d['emotion']:15s}  {lr1}={d['r1']}  {lr2}={d['r2']}")

    # ── Export XLSX ──
    rows_exp = []
    for ex in examples:
        r = {"idx": ex["idx"], "name": ex["name"], "role": ex["role"],
             "text": ex["text"], "n_divergences": ex["n_divergences"]}
        for e in EMOTIONS:
            v1 = int(merged.loc[merged["idx"]==ex["idx"], f"{e}_r1"].values[0])
            v2 = int(merged.loc[merged["idx"]==ex["idx"], f"{e}_r2"].values[0])
            r[f"{e}_r1"], r[f"{e}_r2"] = v1, v2
            r[f"{e}_match"] = "✓" if v1==v2 else "✗"
        rows_exp.append(r)
    out_xlsx = os.path.join(out_dir, "comparaison_divergences.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        pd.DataFrame(rows_exp).to_excel(w, sheet_name="divergences", index=False)
        df_stats.to_excel(w, sheet_name="accord_par_emotion", index=False)
        cols = ["idx","exact_match","n_divergences"] + [f"{e}_{s}" for e in EMOTIONS for s in ("r1","r2")]
        merged[[c for c in cols if c in merged.columns]].to_excel(w, sheet_name="tous_messages", index=False)
        pd.DataFrame([{"messages": N, "exact_match": n_exact,
            "exact_match_pct": round(n_exact/N*100,1)}]).T.to_excel(w, sheet_name="resume", header=["valeur"])
    print(f"\n✓ XLSX exporté → {out_xlsx}")

    # ── Résumé ──
    print(f"\n{'='*70}\n  RÉSUMÉ {lr1} vs {lr2}\n{'='*70}")
    print(f"  Messages comparés : {N}")
    print(f"  Exact match       : {n_exact}/{N} ({n_exact/N*100:.1f}%)")
    print(f"  Avec divergences  : {len(df_div)} ({len(df_div)/N*100:.1f}%)")
    print(f"\n  Cohen's Kappa :")
    for _, r in df_stats.iterrows():
        bar = "█"*int(float(r["kappa"])*20) if r["kappa"]!="N/A" else "—"
        print(f"    {r['emotion']:15s}  κ={str(r['kappa']):>6s}  {bar}")


if __name__ == "__main__":
    main()
