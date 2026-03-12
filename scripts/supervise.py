#!/usr/bin/env python3
"""
Interface de supervision manuelle des annotations (widget notebook).

Usage depuis un notebook :
    %run scripts/supervise.py \
        --run1 outputs/homophobie/run001.jsonl \
        --run2 outputs/homophobie/run002.jsonl \
        --xlsx data/homophobie_scenario_julie.xlsx

Ou en CLI (affiche les stats sans widget) :
    python scripts/supervise.py \
        --run1 outputs/homophobie/run001.jsonl \
        --run2 outputs/homophobie/run002.jsonl
"""

import argparse, json, os, sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import pandas as pd

EMOTIONS = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
]


def parse_args():
    p = argparse.ArgumentParser(description="Supervision manuelle des annotations")
    p.add_argument("--run1", required=True, help="JSONL du run 1")
    p.add_argument("--run2", required=True, help="JSONL du run 2")
    p.add_argument("--xlsx", default=None, help="XLSX original")
    p.add_argument("--save_json", default=None,
                   help="Fichier de sauvegarde progression (défaut: auto)")
    p.add_argument("--out_xlsx", default=None,
                   help="Fichier d'export XLSX (défaut: auto)")
    return p.parse_args()


def load_run(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            row = {"idx": rec["idx"], "row_id": rec.get("row_id"),
                   "json_ok": rec.get("json_ok", False)}
            pj = rec.get("parsed_json")
            if rec["json_ok"] and isinstance(pj, dict):
                emo = pj.get("emotions", {})
                for e in EMOTIONS: row[e] = int(emo.get(e, 0))
                row["confidence"] = pj.get("metadata", {}).get("confidence")
                row["rationale"]  = pj.get("rationale_short")
            else:
                for e in EMOTIONS: row[e] = None
                row["confidence"] = None
                row["rationale"]  = None
            rows.append(row)
    return pd.DataFrame(rows)


# ═══ CLASSE DE SUPERVISION ═════════════════════════════════════════════════

class SupervisionUI:

    def __init__(self, df, df_orig, save_path, export_path):
        import ipywidgets as W
        from IPython.display import display
        self._W = W
        self._display = display

        self.df          = df
        self.df_orig     = df_orig
        self.save_path   = save_path
        self.export_path = export_path
        self.n           = len(df)
        self.pos         = 0
        self.decisions   = self._load()

        self.only_diverg = True
        self.filtered    = []
        self._update_filter()

        for i, fi in enumerate(self.filtered):
            if fi not in self.decisions:
                self.pos = i
                break

        # ── Widgets ──
        self.tog_filter = W.ToggleButtons(
            options=[("Tous les messages", False), ("Divergences seules", True)],
            value=True, style={"button_width": "160px"})
        self.tog_filter.observe(self._on_filter, names="value")

        self.bar_progress = W.IntProgress(min=0, max=1, bar_style="info",
                                          layout=W.Layout(width="250px"))
        self.lbl_progress = W.HTML()

        self.btn_prev = W.Button(description="◄ Préc.", layout=W.Layout(width="85px"))
        self.btn_next = W.Button(description="Suiv. ►", layout=W.Layout(width="85px"))
        self.btn_skip = W.Button(description="→ Non-revu suivant",
                                 button_style="warning", layout=W.Layout(width="170px"))
        self.lbl_pos  = W.HTML()
        self.inp_jump = W.BoundedIntText(
            value=0, min=0, max=max(1, int(df["idx"].max())),
            description="idx :", layout=W.Layout(width="160px"))
        self.btn_jump = W.Button(description="Go", layout=W.Layout(width="45px"))

        self.btn_prev.on_click(lambda _: self._go(-1))
        self.btn_next.on_click(lambda _: self._go(1))
        self.btn_skip.on_click(self._go_unreviewed)
        self.btn_jump.on_click(self._go_jump)

        self.html_msg = W.HTML()
        self.emo_parts = {}
        emo_children = [self._header_row()]
        for e in EMOTIONS:
            row_w, parts = self._emotion_row(e)
            self.emo_parts[e] = parts
            emo_children.append(row_w)
        self.vbox_emos = W.VBox(emo_children)
        self.html_rat = W.HTML()
        self.txt_notes = W.Textarea(
            placeholder="Notes du réviseur (optionnel)…",
            layout=W.Layout(width="100%", height="50px"))

        self.btn_valid  = W.Button(description="✓ Valider et suivant",
                                   button_style="success",
                                   layout=W.Layout(width="200px", height="38px"))
        self.btn_export = W.Button(description="💾 Exporter XLSX",
                                   button_style="info",
                                   layout=W.Layout(width="200px", height="38px"))
        self.lbl_status = W.HTML()

        self.btn_valid.on_click(self._on_validate)
        self.btn_export.on_click(self._on_export)

        self.ui = W.VBox([
            self.tog_filter,
            W.HBox([self.bar_progress, self.lbl_progress]),
            W.HBox([self.btn_prev, self.btn_next, self.btn_skip,
                    W.HTML("&nbsp;"), self.lbl_pos]),
            W.HBox([self.inp_jump, self.btn_jump]),
            W.HTML("<hr style='margin:4px 0'>"),
            self.html_msg,
            W.HTML("<hr style='margin:4px 0'>"),
            self.vbox_emos,
            W.HTML("<hr style='margin:4px 0'>"),
            self.html_rat, self.txt_notes,
            W.HBox([self.btn_valid, self.btn_export]),
            self.lbl_status,
        ])
        self._render()

    def _header_row(self):
        W = self._W
        S = "display:inline-block;font-weight:bold;text-align:center"
        return W.HBox([
            W.HTML(f"<span style='{S};width:18px'></span>"),
            W.HTML(f"<span style='{S};width:120px'>Émotion</span>"),
            W.HTML(f"<span style='{S};width:55px'>R1</span>"),
            W.HTML(f"<span style='{S};width:55px'>R2</span>"),
            W.HTML(f"<span style='{S};width:160px'>Votre choix</span>"),
        ])

    def _emotion_row(self, emotion):
        W = self._W
        flag   = W.HTML(layout=W.Layout(width="18px"))
        label  = W.HTML(layout=W.Layout(width="120px"))
        r1     = W.HTML(layout=W.Layout(width="55px"))
        r2     = W.HTML(layout=W.Layout(width="55px"))
        toggle = W.ToggleButtons(options=[("0", 0), ("1", 1)], value=0,
                                 style={"button_width": "45px"},
                                 layout=W.Layout(width="160px"))
        parts = dict(flag=flag, label=label, r1=r1, r2=r2, toggle=toggle)
        return W.HBox([flag, label, r1, r2, toggle]), parts

    @staticmethod
    def _badge(val, color):
        return (f"<span style='display:inline-block;width:28px;text-align:center;"
                f"background:{color};color:#fff;padding:2px 6px;border-radius:4px;"
                f"font-weight:bold;font-size:13px'>{val}</span>")

    def _load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r", encoding="utf-8") as f:
                return {int(k): v for k, v in json.load(f).items()}
        return {}

    def _save(self):
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.decisions.items()},
                      f, ensure_ascii=False, indent=2)

    def _update_filter(self):
        if self.only_diverg:
            self.filtered = [i for i in range(self.n) if self.df.iloc[i]["n_div"] > 0]
        else:
            self.filtered = list(range(self.n))
        if not self.filtered: self.pos = 0
        else: self.pos = max(0, min(self.pos, len(self.filtered) - 1))

    def _on_filter(self, change):
        self.only_diverg = change["new"]
        self._update_filter(); self.pos = 0
        self.lbl_status.value = ""; self._render()

    def _ri(self):
        return self.filtered[self.pos] if self.filtered else None

    def _go(self, delta):
        new = self.pos + delta
        if 0 <= new < len(self.filtered):
            self.pos = new; self.lbl_status.value = ""; self._render()

    def _go_unreviewed(self, _):
        for offset in range(1, len(self.filtered) + 1):
            p = (self.pos + offset) % len(self.filtered)
            if self.filtered[p] not in self.decisions:
                self.pos = p; self.lbl_status.value = ""; self._render(); return
        self.lbl_status.value = ("<span style='color:#27ae60;font-weight:bold'>"
                                 "✓ Tous les messages revus !</span>")

    def _go_jump(self, _):
        target = self.inp_jump.value
        best, best_d = 0, float("inf")
        for i, fi in enumerate(self.filtered):
            d = abs(int(self.df.iloc[fi]["idx"]) - target)
            if d < best_d: best, best_d = i, d
        self.pos = best; self.lbl_status.value = ""; self._render()

    def _render(self):
        ri = self._ri()
        if ri is None:
            self.html_msg.value = "<h3 style='color:#888'>Aucun message.</h3>"; return

        row = self.df.iloc[ri]
        is_rev = ri in self.decisions
        n_div = int(row["n_div"]); orig_idx = int(row["idx"])

        n_rev = sum(1 for fi in self.filtered if fi in self.decisions)
        self.bar_progress.max = max(1, len(self.filtered))
        self.bar_progress.value = n_rev
        pct = n_rev / len(self.filtered) * 100 if self.filtered else 0
        self.lbl_progress.value = (f"<b>{n_rev}/{len(self.filtered)}</b> revus ({pct:.0f}%)")

        tag = "✅ revu" if is_rev else "🔶 à revoir"
        self.lbl_pos.value = (f"<b>Position {self.pos+1}/{len(self.filtered)}</b>"
                              f" | idx={orig_idx} | {tag}")

        text = str(row.get("TEXT", ""))
        name = str(row.get("NAME", "?"))
        role = str(row.get("ROLE", "?"))
        div_badge = (f"<span style='background:#e74c3c;color:#fff;padding:3px 10px;"
                     f"border-radius:5px;font-weight:bold'>⚠ {n_div} divergence(s)</span>"
                     if n_div > 0 else
                     "<span style='background:#27ae60;color:#fff;padding:3px 10px;"
                     "border-radius:5px'>✓ Accord complet</span>")
        self.html_msg.value = (
            f"<div style='background:#f8f9fa;padding:12px;border-radius:8px;"
            f"border:1px solid #dee2e6'><div style='margin-bottom:8px'>"
            f"<b style='font-size:15px'>{name}</b> | Rôle: <code>{role}</code>"
            f" | {div_badge}</div>"
            f"<div style='font-size:14px;padding:10px;background:#fff;"
            f"border-radius:6px;border-left:5px solid #3498db;line-height:1.6'>"
            f"{text or '<i style=\"color:#999\">(vide)</i>'}</div></div>")

        prev_dec = self.decisions.get(ri, {})
        for e in EMOTIONS:
            wd = self.emo_parts[e]
            r1v = int(row[f"{e}_r1"]) if pd.notna(row[f"{e}_r1"]) else 0
            r2v = int(row[f"{e}_r2"]) if pd.notna(row[f"{e}_r2"]) else 0
            div = r1v != r2v
            wd["flag"].value = "⚠️" if div else "<span style='color:#27ae60'>✓</span>"
            bg = "#fff3cd" if div else "transparent"
            fw = "bold" if div else "normal"
            wd["label"].value = (f"<span style='background:{bg};padding:3px 6px;"
                                 f"border-radius:4px;font-weight:{fw}'>{e}</span>")
            c = "#3498db" if div else "#bdc3c7"
            c2 = "#e67e22" if div else "#bdc3c7"
            wd["r1"].value = self._badge(r1v, c)
            wd["r2"].value = self._badge(r2v, c2)
            wd["toggle"].value = int(prev_dec[e]) if e in prev_dec else r1v

        rat1 = row.get("rationale_r1") or ""
        rat2 = row.get("rationale_r2") or ""
        conf1 = row.get("confidence_r1") or "?"
        conf2 = row.get("confidence_r2") or "?"
        self.html_rat.value = (
            f"<div style='font-size:12px;background:#f0f0f0;padding:8px;"
            f"border-radius:6px;line-height:1.6'>"
            f"<b>Rationale R1</b> (conf: <code>{conf1}</code>): <i>{rat1}</i><br>"
            f"<b>Rationale R2</b> (conf: <code>{conf2}</code>): <i>{rat2}</i></div>")
        self.txt_notes.value = prev_dec.get("notes", "")

    def _on_validate(self, _):
        ri = self._ri()
        if ri is None: return
        decision = {}
        for e in EMOTIONS: decision[e] = int(self.emo_parts[e]["toggle"].value)
        decision["notes"] = self.txt_notes.value
        decision["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        self.decisions[ri] = decision; self._save()
        orig_idx = int(self.df.iloc[ri]["idx"])
        found = self._advance_unreviewed()
        if found:
            self.lbl_status.value = (f"<span style='color:#27ae60'>✓ idx={orig_idx} "
                                     f"sauvegardé — prochain non-revu affiché</span>")
        else:
            self.lbl_status.value = (f"<span style='color:#27ae60;font-weight:bold'>"
                                     f"✓ idx={orig_idx} — Tous revus !</span>")

    def _advance_unreviewed(self):
        for offset in range(1, len(self.filtered) + 1):
            p = (self.pos + offset) % len(self.filtered)
            if self.filtered[p] not in self.decisions:
                self.pos = p; self._render(); return True
        self._render(); return False

    def _on_export(self, _):
        rows_out = []
        for i in range(self.n):
            row = self.df.iloc[i]; orig_idx = int(row["idx"])
            out = {"idx": orig_idx}
            if self.df_orig is not None and orig_idx < len(self.df_orig):
                for col in self.df_orig.columns: out[col] = self.df_orig.iloc[orig_idx][col]
            if i in self.decisions:
                dec = self.decisions[i]
                for e in EMOTIONS: out[e] = dec.get(e)
                out["reviewed"] = True; out["reviewer_notes"] = dec.get("notes", "")
            else:
                for e in EMOTIONS:
                    r1, r2 = row.get(f"{e}_r1"), row.get(f"{e}_r2")
                    out[e] = int(r1) if (pd.notna(r1) and pd.notna(r2) and int(r1)==int(r2)) else None
                out["reviewed"] = False; out["reviewer_notes"] = ""
            for e in EMOTIONS:
                out[f"{e}_run1"] = int(row[f"{e}_r1"]) if pd.notna(row[f"{e}_r1"]) else None
                out[f"{e}_run2"] = int(row[f"{e}_r2"]) if pd.notna(row[f"{e}_r2"]) else None
            out["n_divergences"] = int(row["n_div"])
            rows_out.append(out)

        df_out = pd.DataFrame(rows_out)
        lead = [c for c in ["idx"] if c in df_out.columns]
        if self.df_orig is not None:
            lead += [c for c in self.df_orig.columns if c in df_out.columns and c not in lead]
        ordered = lead + EMOTIONS + ["reviewed","reviewer_notes","n_divergences"]
        ordered += [f"{e}_{s}" for e in EMOTIONS for s in ("run1","run2")]
        rest = [c for c in df_out.columns if c not in ordered]
        df_out = df_out[[c for c in ordered+rest if c in df_out.columns]]
        df_out.to_excel(self.export_path, index=False, engine="openpyxl")

        n_rev = len(self.decisions)
        self.lbl_status.value = (
            f"<div style='color:#155724;background:#d4edda;padding:10px;"
            f"border-radius:6px;font-weight:bold'>"
            f"✓ Exporté → <code>{self.export_path}</code> ({n_rev} revus)</div>")

    def show(self):
        self._display(self.ui)


# ═══ MAIN ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    df_r1 = load_run(args.run1)
    df_r2 = load_run(args.run2)

    merged = pd.merge(df_r1, df_r2, on="idx", how="inner", suffixes=("_r1", "_r2"))
    merged = merged[merged["json_ok_r1"] & merged["json_ok_r2"]].reset_index(drop=True)

    df_orig = None
    if args.xlsx and os.path.exists(args.xlsx):
        df_orig = pd.read_excel(args.xlsx).reset_index(drop=True)
        for col in ["TEXT", "NAME", "ROLE", "TIME"]:
            merged[col] = merged["idx"].apply(
                lambda i, c=col: str(df_orig.iloc[i].get(c, ""))
                if i < len(df_orig) else "")

    for e in EMOTIONS:
        merged[f"{e}_div"] = merged[f"{e}_r1"] != merged[f"{e}_r2"]
    merged["n_div"] = sum(merged[f"{e}_div"].astype(int) for e in EMOTIONS)

    N_TOTAL  = len(merged)
    N_DIVERG = int((merged["n_div"] > 0).sum())
    print(f"✓ {N_TOTAL} messages comparables")
    print(f"  dont {N_DIVERG} avec ≥1 divergence")

    # Chemins par défaut
    run1_dir = os.path.dirname(os.path.abspath(args.run1))
    save_path = args.save_json or os.path.join(run1_dir, "supervision_progress.json")
    export_path = args.out_xlsx or os.path.join(run1_dir, "annotations_validees.xlsx")

    if N_TOTAL == 0:
        print("⚠ Aucun message comparable"); return

    try:
        supervisor = SupervisionUI(
            df=merged, df_orig=df_orig,
            save_path=save_path, export_path=export_path)
        supervisor.show()
    except ImportError:
        print("⚠ ipywidgets non disponible — exécutez depuis un notebook Jupyter/Colab.")
        print(f"  Fichiers de sortie prévus :")
        print(f"    Progression : {save_path}")
        print(f"    Export XLSX : {export_path}")


if __name__ == "__main__":
    main()
