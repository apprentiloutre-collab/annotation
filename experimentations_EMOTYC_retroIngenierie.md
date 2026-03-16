cell
```python
import torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = (
    AutoModelForSequenceClassification
    .from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
    .to(DEVICE)
    .eval()
)

sentences = df["sentence"].astype(str).tolist()
EOS       = tokenizer.eos_token                            # </s>

# LABEL_COLS[j] = colonne XLSX correspondant à l'index j du modèle (id2label)
LABEL_COLS = [
    "emo", "emo_comportementale", "emo_designee", "emo_montree",
    "emo_suggeree", "emo_base", "emo_complexe",
    "admiration", "autre",
    "colere", "culpabilite", "degout", "embarras",
    "fierte", "jalousie", "joie", "peur", "surprise", "tristesse",
]
Y = df[LABEL_COLS].values.astype(int)                      # (101, 19)
N, L = Y.shape
print(f"Référence : {N} phrases × {L} labels = {N*L} cellules\n")

# ═══════════════════════════════════════════════════════════════
#  2. FORMATS D'ENTRÉE CANDIDATS
#     (on ne sait pas comment le web formate avant d'envoyer
#      au tokenizer → on teste plusieurs variantes)
# ═══════════════════════════════════════════════════════════════
TEMPLATES = {
    "raw":
        lambda s: s,
    "bca_eos_nospace":
        lambda s: f"before: {EOS}current: {s}{EOS}after: {EOS}",
    "bca_eos_space":
        lambda s: f"before: {EOS} current: {s} after: {EOS}",
    "bca_empty":
        lambda s: f"before:  current: {s} after: ",
    "bca_none_str":
        lambda s: f"before: None current: {s} after: None",
}

# ═══════════════════════════════════════════════════════════════
#  3. INFÉRENCE  –  sigmoid (multi-label), dropout désactivé
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_sigmoid(template_fn):
    """Renvoie la matrice (N, L) des probabilités sigmoïdes."""
    P = np.empty((N, L), dtype=np.float64)
    for i, s in enumerate(sentences):
        enc = tokenizer(
            template_fn(s),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        logits = model(**enc).logits.squeeze(0)          # (19,)
        P[i] = torch.sigmoid(logits).cpu().double().numpy()
    return P

# ═══════════════════════════════════════════════════════════════
#  4. OPTIMISATION DES SEUILS (un par label)
#     Pour chaque label j on parcourt tous les « midpoints »
#     entre valeurs consécutives de σ(logit_j) et on garde
#     celui qui minimise les erreurs binaires.
# ═══════════════════════════════════════════════════════════════
def sweep_thresholds(P, Y):
    thresholds = np.empty(L)
    total_err  = 0
    for j in range(L):
        pj, yj = P[:, j], Y[:, j]
        vals = np.sort(np.unique(pj))
        # seuils candidats : entre chaque paire de probas + bornes
        cuts = np.concatenate([
            [vals[0] - 1e-9],
            (vals[:-1] + vals[1:]) / 2.0,
            [vals[-1] + 1e-9],
        ])
        errs = np.array([
            int(((pj >= t).astype(int) != yj).sum()) for t in cuts
        ])
        idx_best = int(errs.argmin())
        thresholds[j] = cuts[idx_best]
        total_err += errs[idx_best]
    return thresholds, total_err

# ═══════════════════════════════════════════════════════════════
#  5. RECHERCHE DU MEILLEUR FORMAT
# ═══════════════════════════════════════════════════════════════
winner = dict(err=N * L + 1)

for name, fn in TEMPLATES.items():
    P   = infer_sigmoid(fn)
    thr, err_opt = sweep_thresholds(P, Y)
    err_05 = int(((P >= 0.5).astype(int) != Y).sum())

    star = " ★ PERFECT" if err_opt == 0 else ""
    print(f"  {name:20s}  err@0.5={err_05:4d}   "
          f"err@opt={err_opt:4d}  /{N*L}{star}")

    if err_opt < winner["err"]:
        winner = dict(name=name, fn=fn, P=P, thr=thr, err=err_opt)
    if err_opt == 0:                       # inutile de continuer
        break

# ═══════════════════════════════════════════════════════════════
#  6. RAPPORT DÉTAILLÉ
# ═══════════════════════════════════════════════════════════════
print(f"\n{'═'*62}")
print(f"  Meilleur template  : {winner['name']}")
print(f"  Erreurs restantes  : {winner['err']} / {N*L}")
print(f"{'═'*62}\n")

P, thr = winner["P"], winner["thr"]

print("Seuils optimaux par label :")
for j in range(L):
    preds  = (P[:, j] >= thr[j]).astype(int)
    ne     = int((preds != Y[:, j]).sum())
    status = "✓" if ne == 0 else f"✗ {ne} erreur(s)"
    print(f"  {LABEL_COLS[j]:<25s}  seuil = {thr[j]:.8f}   {status}")

Y_hat  = (P >= thr).astype(int)
row_ok = int((Y_hat == Y).all(axis=1).sum())
print(f"\nLignes parfaitement reproduites : {row_ok} / {N}")
print(f"Précision cellule par cellule  : "
      f"{(Y_hat == Y).sum()} / {N*L}  "
      f"({(Y_hat == Y).mean():.4%})")

# Détail des éventuelles erreurs résiduelles
bad = np.where(~(Y_hat == Y).all(axis=1))[0]
if bad.size:
    print(f"\nLignes en erreur ({bad.size}) :")
    for i in bad:
        for j in np.where(Y_hat[i] != Y[i])[0]:
            print(f"  row {i:3d}  {LABEL_COLS[j]:<22s}  "
                  f"prédit={Y_hat[i,j]}  vrai={Y[i,j]}  "
                  f"σ={P[i,j]:.8f}")
else:
    print("\n✅  Aucune erreur : les outputs web sont parfaitement reproduits !")

# ═══════════════════════════════════════════════════════════════
#  7. FONCTION RÉUTILISABLE  (à brancher dans votre pipeline)
# ═══════════════════════════════════════════════════════════════
FINAL_TEMPLATE   = winner["fn"]
FINAL_THRESHOLDS = winner["thr"]          # np.array de shape (19,)

@torch.no_grad()
def predict_binary(sentence: str) -> dict[str, int]:
    """
    Reproduit exactement la sortie binaire du modèle web
    pour une phrase isolée.
    """
    text = FINAL_TEMPLATE(sentence)
    enc  = tokenizer(text, return_tensors="pt",
                     truncation=True, max_length=512).to(DEVICE)
    logits = model(**enc).logits.squeeze(0)
    probs  = torch.sigmoid(logits).cpu().numpy()
    binary = (probs >= FINAL_THRESHOLDS).astype(int)
    return {LABEL_COLS[j]: int(binary[j]) for j in range(L)}


# ── Validation finale sur toutes les lignes ──
print("\n\nValidation complète :")
all_ok = True
for i in range(N):
    pred = predict_binary(sentences[i])
    true = {LABEL_COLS[j]: int(Y[i, j]) for j in range(L)}
    if pred != true:
        all_ok = False
        print(f"  ✗ row {i}: {sentences[i][:55]}…")
if all_ok:
    print("  ✓ 101/101 lignes identiques aux outputs web.")

# ── Export des seuils pour réutilisation ──
thresholds_dict = {LABEL_COLS[j]: float(FINAL_THRESHOLDS[j]) for j in range(L)}
print(f"\n# Seuils à copier-coller :\nTHRESHOLDS = {thresholds_dict}")
```
```output
config.json:   0%|          | 0.00/508 [00:00<?, ?B/s]tokenizer_config.json:   0%|          | 0.00/25.0 [00:00<?, ?B/s]sentencepiece.bpe.model:   0%|          | 0.00/811k [00:00<?, ?B/s]tokenizer.json: 0.00B [00:00, ?B/s]config.json: 0.00B [00:00, ?B/s]model.safetensors:   0%|          | 0.00/443M [00:00<?, ?B/s]Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]Référence : 2451 phrases × 19 labels = 46569 cellules

  raw                   err@0.5=3654   err@opt=3532  /46569
  bca_eos_nospace       err@0.5=2599   err@opt=2113  /46569
  bca_eos_space         err@0.5=2680   err@opt=2166  /46569
  bca_empty             err@0.5=2727   err@opt=2175  /46569
  bca_none_str          err@0.5=2723   err@opt=2157  /46569

══════════════════════════════════════════════════════════════
  Meilleur template  : bca_eos_nospace
  Erreurs restantes  : 2113 / 46569
══════════════════════════════════════════════════════════════

Seuils optimaux par label :
  emo                        seuil = 0.55359590   ✗ 312 erreur(s)
  emo_comportementale        seuil = 0.83242783   ✗ 75 erreur(s)
  emo_designee               seuil = 0.48298708   ✗ 94 erreur(s)
  emo_montree                seuil = 0.84488982   ✗ 86 erreur(s)
  emo_suggeree               seuil = 0.85283723   ✗ 347 erreur(s)
  emo_base                   seuil = 0.47861214   ✗ 303 erreur(s)
  emo_complexe               seuil = 0.97325930   ✗ 64 erreur(s)
  admiration                 seuil = 0.98975620   ✗ 303 erreur(s)
  autre                      seuil = 0.99997091   ✗ 28 erreur(s)
  colere                     seuil = 0.81989828   ✗ 127 erreur(s)
  culpabilite                seuil = 0.12354351   ✓
  degout                     seuil = 0.19585860   ✓
  embarras                   seuil = 0.95544162   ✗ 20 erreur(s)
  fierte                     seuil = 0.81253061   ✗ 10 erreur(s)
  jalousie                   seuil = 0.01689715   ✓
  joie                       seuil = 0.94541007   ✗ 95 erreur(s)
  peur                       seuil = 0.98595539   ✗ 57 erreur(s)
  surprise                   seuil = 0.97324455   ✗ 56 erreur(s)
  tristesse                  seuil = 0.64309916   ✗ 136 erreur(s)

Lignes parfaitement reproduites : 1511 / 2451
Précision cellule par cellule  : 44456 / 46569  (95.4626%)

Lignes en erreur (940) :
  row   0  emo_montree             prédit=0  vrai=1  σ=0.00941700
  row   0  emo_base                prédit=0  vrai=1  σ=0.23194258
  row   0  colere                  prédit=0  vrai=1  σ=0.35127339
  row  52  emo                     prédit=1  vrai=0  σ=0.98970556
  row  52  emo_montree             prédit=1  vrai=0  σ=0.98620439
  row  52  emo_base                prédit=1  vrai=0  σ=0.99060190
  row  52  joie                    prédit=1  vrai=0  σ=0.99606812
  row  66  admiration              prédit=0  vrai=1  σ=0.00006661
  row  67  emo_designee            prédit=0  vrai=1  σ=0.34512794
  row  67  admiration              prédit=0  vrai=1  σ=0.00030836
  row  67  colere                  prédit=0  vrai=1  σ=0.25697601
  row  67  embarras                prédit=0  vrai=1  σ=0.65186977
  row  69  colere                  prédit=0  vrai=1  σ=0.36672962
  row  69  tristesse               prédit=1  vrai=0  σ=0.76184762
  row  70  admiration              prédit=0  vrai=1  σ=0.00010354
  row  70  colere                  prédit=0  vrai=1  σ=0.59848565
  row  70  tristesse               prédit=0  vrai=1  σ=0.14251174
  row  71  emo                     prédit=0  vrai=1  σ=0.00450449
  row  71  emo_montree             prédit=0  vrai=1  σ=0.00022179
  row  71  emo_base                prédit=0  vrai=1  σ=0.00054577
  row  71  colere                  prédit=0  vrai=1  σ=0.00171229
  row  72  emo_suggeree            prédit=0  vrai=1  σ=0.01164239
  row  72  emo_complexe            prédit=0  vrai=1  σ=0.00763510
  row  72  embarras                prédit=0  vrai=1  σ=0.08505897
  row  72  tristesse               prédit=0  vrai=1  σ=0.06460778
  row  76  emo_designee            prédit=0  vrai=1  σ=0.13893500
  row  76  emo_montree             prédit=0  vrai=1  σ=0.81388682
  row  76  emo_suggeree            prédit=0  vrai=1  σ=0.23081598
  row  76  emo_complexe            prédit=0  vrai=1  σ=0.81892681
  row  76  embarras                prédit=0  vrai=1  σ=0.80849892
  row  76  tristesse               prédit=0  vrai=1  σ=0.18620898
  row  78  emo                     prédit=0  vrai=1  σ=0.00014735
  row  78  emo_montree             prédit=0  vrai=1  σ=0.00001568
  row  78  emo_base                prédit=0  vrai=1  σ=0.00009072
  row  78  colere                  prédit=0  vrai=1  σ=0.00002410
  row  85  emo                     prédit=0  vrai=1  σ=0.00232184
  row  85  emo_montree             prédit=0  vrai=1  σ=0.00158251
  row  85  emo_base                prédit=0  vrai=1  σ=0.00036005
  row  85  colere                  prédit=0  vrai=1  σ=0.00183107
  row  87  admiration              prédit=0  vrai=1  σ=0.00003705
  row  90  emo                     prédit=0  vrai=1  σ=0.00016691
  row  90  emo_base                prédit=0  vrai=1  σ=0.00010185
  row  90  colere                  prédit=0  vrai=1  σ=0.00001344
  row  91  emo_base                prédit=0  vrai=1  σ=0.00112886
  row  91  admiration              prédit=0  vrai=1  σ=0.00003083
  row  91  joie                    prédit=0  vrai=1  σ=0.00018581
  row  92  emo                     prédit=0  vrai=1  σ=0.00018959
  row  92  emo_suggeree            prédit=0  vrai=1  σ=0.00013757
  row  92  emo_base                prédit=0  vrai=1  σ=0.00006674
  row  92  colere                  prédit=0  vrai=1  σ=0.00005459
  row  93  admiration              prédit=0  vrai=1  σ=0.00009211
  row  96  emo_montree             prédit=0  vrai=1  σ=0.15173286
  row  96  emo_base                prédit=0  vrai=1  σ=0.20629905
  row  96  admiration              prédit=0  vrai=1  σ=0.00009614
  row  96  colere                  prédit=0  vrai=1  σ=0.69525909
  row  97  emo_montree             prédit=0  vrai=1  σ=0.00659943
  row  97  emo_base                prédit=0  vrai=1  σ=0.02115680
  row  97  admiration              prédit=0  vrai=1  σ=0.00002628
  row  97  colere                  prédit=0  vrai=1  σ=0.01191117
  row  99  admiration              prédit=0  vrai=1  σ=0.00011222
  row 104  emo                     prédit=0  vrai=1  σ=0.00016568
  row 104  emo_montree             prédit=0  vrai=1  σ=0.00001756
  row 104  emo_base                prédit=0  vrai=1  σ=0.00009148
  row 104  colere                  prédit=0  vrai=1  σ=0.00001950
  row 105  admiration              prédit=0  vrai=1  σ=0.00007458
  row 106  emo_suggeree            prédit=0  vrai=1  σ=0.13904212
  row 106  emo_base                prédit=0  vrai=1  σ=0.36034784
  row 106  colere                  prédit=0  vrai=1  σ=0.65060925
  row 109  admiration              prédit=0  vrai=1  σ=0.00001866
  row 112  emo_base                prédit=0  vrai=1  σ=0.37090629
  row 114  emo_designee            prédit=1  vrai=0  σ=0.76903224
  row 114  emo_suggeree            prédit=0  vrai=1  σ=0.09018456
  row 114  emo_complexe            prédit=0  vrai=1  σ=0.90200001
  row 114  admiration              prédit=0  vrai=1  σ=0.01140456
  row 114  colere                  prédit=0  vrai=1  σ=0.33615416
  row 114  embarras                prédit=0  vrai=1  σ=0.88622004
  row 115  emo_montree             prédit=0  vrai=1  σ=0.13325179
  row 115  admiration              prédit=0  vrai=1  σ=0.00001856
  row 115  colere                  prédit=0  vrai=1  σ=0.47441754
  row 116  emo                     prédit=1  vrai=0  σ=0.99898022
  row 118  colere                  prédit=0  vrai=1  σ=0.56390047
  row 119  emo_montree             prédit=0  vrai=1  σ=0.31195024
  row 119  emo_base                prédit=0  vrai=1  σ=0.13408296
  row 129  emo_suggeree            prédit=0  vrai=1  σ=0.18571293
  row 129  emo_complexe            prédit=0  vrai=1  σ=0.06066672
  row 129  embarras                prédit=0  vrai=1  σ=0.17406790
  row 130  admiration              prédit=0  vrai=1  σ=0.00010575
  row 131  emo_suggeree            prédit=0  vrai=1  σ=0.01474741
  row 131  emo_base                prédit=0  vrai=1  σ=0.06918475
  row 131  colere                  prédit=0  vrai=1  σ=0.10842619
  row 131  tristesse               prédit=0  vrai=1  σ=0.26746696
  row 133  admiration              prédit=0  vrai=1  σ=0.00002166
  row 135  emo                     prédit=0  vrai=1  σ=0.05481307
  row 135  emo_suggeree            prédit=0  vrai=1  σ=0.06228619
  row 135  emo_base                prédit=0  vrai=1  σ=0.00435079
  row 135  colere                  prédit=0  vrai=1  σ=0.00839802
  row 135  tristesse               prédit=0  vrai=1  σ=0.00021044
  row 138  emo                     prédit=0  vrai=1  σ=0.00973954
  row 138  emo_suggeree            prédit=0  vrai=1  σ=0.00008254
  row 138  emo_base                prédit=0  vrai=1  σ=0.00004825
  row 138  colere                  prédit=0  vrai=1  σ=0.00002570
  row 139  emo_suggeree            prédit=0  vrai=1  σ=0.04497145
  row 139  colere                  prédit=1  vrai=0  σ=0.99626786
  row 139  tristesse               prédit=0  vrai=1  σ=0.00061428
  row 140  emo_suggeree            prédit=0  vrai=1  σ=0.01590493
  row 140  colere                  prédit=1  vrai=0  σ=0.98751903
  row 145  colere                  prédit=0  vrai=1  σ=0.04950316
  row 145  joie                    prédit=1  vrai=0  σ=0.98682833
  row 146  admiration              prédit=0  vrai=1  σ=0.00014267
  row 149  emo_montree             prédit=0  vrai=1  σ=0.00104361
  row 149  emo_suggeree            prédit=0  vrai=1  σ=0.00130372
  row 149  emo_base                prédit=0  vrai=1  σ=0.00530296
  row 149  admiration              prédit=0  vrai=1  σ=0.00007602
  row 149  colere                  prédit=0  vrai=1  σ=0.00196590
  row 154  admiration              prédit=0  vrai=1  σ=0.00007568
  row 155  emo_comportementale     prédit=0  vrai=1  σ=0.20910835
  row 155  emo_suggeree            prédit=0  vrai=1  σ=0.10813820
  row 155  emo_complexe            prédit=0  vrai=1  σ=0.28427708
  row 156  emo                     prédit=0  vrai=1  σ=0.00438274
  row 156  emo_suggeree            prédit=0  vrai=1  σ=0.01864407
  row 156  emo_complexe            prédit=0  vrai=1  σ=0.00003632
  row 156  tristesse               prédit=0  vrai=1  σ=0.02376434
  row 158  emo_suggeree            prédit=0  vrai=1  σ=0.00009565
  row 159  colere                  prédit=0  vrai=1  σ=0.70655060
  row 161  emo_base                prédit=0  vrai=1  σ=0.08870111
  row 161  colere                  prédit=0  vrai=1  σ=0.41501194
  row 161  joie                    prédit=0  vrai=1  σ=0.05046047
  row 162  emo_suggeree            prédit=0  vrai=1  σ=0.10861066
  row 165  admiration              prédit=0  vrai=1  σ=0.00002078
  row 167  emo_montree             prédit=0  vrai=1  σ=0.03189901
  row 167  emo_base                prédit=0  vrai=1  σ=0.15137634
  row 167  admiration              prédit=0  vrai=1  σ=0.00008040
  row 167  colere                  prédit=0  vrai=1  σ=0.41143516
  row 168  admiration              prédit=0  vrai=1  σ=0.00003705
  row 170  emo                     prédit=1  vrai=0  σ=0.99792957
  row 170  emo_montree             prédit=1  vrai=0  σ=0.96238053
  row 170  emo_base                prédit=1  vrai=0  σ=0.99655652
  row 170  colere                  prédit=1  vrai=0  σ=0.99711561
  row 171  emo_suggeree            prédit=0  vrai=1  σ=0.02412403
  row 171  emo_base                prédit=0  vrai=1  σ=0.17015606
  row 172  emo_montree             prédit=0  vrai=1  σ=0.00104361
  row 172  emo_suggeree            prédit=0  vrai=1  σ=0.00130372
  row 172  emo_base                prédit=0  vrai=1  σ=0.00530296
  row 172  admiration              prédit=0  vrai=1  σ=0.00007602
  row 172  colere                  prédit=0  vrai=1  σ=0.00196590
  row 176  admiration              prédit=0  vrai=1  σ=0.00003705
  row 178  admiration              prédit=0  vrai=1  σ=0.00003705
  row 180  admiration              prédit=0  vrai=1  σ=0.00003705
  row 193  emo_designee            prédit=0  vrai=1  σ=0.29120532
  row 193  colere                  prédit=0  vrai=1  σ=0.80915493
  row 197  admiration              prédit=0  vrai=1  σ=0.00018107
  row 200  admiration              prédit=0  vrai=1  σ=0.00006180
  row 205  emo                     prédit=1  vrai=0  σ=0.99762100
  row 207  admiration              prédit=0  vrai=1  σ=0.00006751
  row 208  emo                     prédit=1  vrai=0  σ=0.55549109
  row 208  tristesse               prédit=1  vrai=0  σ=0.77903020
  row 210  admiration              prédit=0  vrai=1  σ=0.00025508
  row 213  admiration              prédit=0  vrai=1  σ=0.00005945
  row 214  emo_comportementale     prédit=0  vrai=1  σ=0.72579855
  row 214  peur                    prédit=0  vrai=1  σ=0.64914447
  row 217  emo_base                prédit=1  vrai=0  σ=0.93815589
  row 217  admiration              prédit=0  vrai=1  σ=0.00002212
… [output truncated]
```
cell
```python
# (markdown)
# ══════════════════════════════════════════════════════════════
#   Meilleur template  : bca_eos_nospace
#   Erreurs restantes  : 2113 / 46569
# ══════════════════════════════════════════════════════════════
#
# Seuils optimaux par label :
#   emo                        seuil = 0.55359590   ✗ 312 erreur(s)
#   emo_comportementale        seuil = 0.83242783   ✗ 75 erreur(s)
#   emo_designee               seuil = 0.48298708   ✗ 94 erreur(s)
#   emo_montree                seuil = 0.84488982   ✗ 86 erreur(s)
#   emo_suggeree               seuil = 0.85283723   ✗ 347 erreur(s)
#   emo_base                   seuil = 0.47861214   ✗ 303 erreur(s)
#   emo_complexe               seuil = 0.97325930   ✗ 64 erreur(s)
#   admiration                 seuil = 0.98975620   ✗ 303 erreur(s)
#   autre                      seuil = 0.99997091   ✗ 28 erreur(s)
#   colere                     seuil = 0.81989828   ✗ 127 erreur(s)
#   culpabilite                seuil = 0.12354351   ✓
#   degout                     seuil = 0.19585860   ✓
#   embarras                   seuil = 0.95544162   ✗ 20 erreur(s)
#   fierte                     seuil = 0.81253061   ✗ 10 erreur(s)
#   jalousie                   seuil = 0.01689715   ✓
#   joie                       seuil = 0.94541007   ✗ 95 erreur(s)
#   peur                       seuil = 0.98595539   ✗ 57 erreur(s)
#   surprise                   seuil = 0.97324455   ✗ 56 erreur(s)
#   tristesse                  seuil = 0.64309916   ✗ 136 erreur(s)
#
# Lignes parfaitement reproduites : 1511 / 2451
# Précision cellule par cellule  : 44456 / 46569  (95.4626%)
```
```output

```
cell
```python
# (markdown)
# # Seuils à copier-coller :
# THRESHOLDS = {'emo': 0.5535959005355835, 'emo_comportementale': 0.8324278295040131, 'emo_designee': 0.48298707604408264, 'emo_montree': 0.8448898196220398, 'emo_suggeree': 0.8528372347354889, 'emo_base': 0.47861213982105255, 'emo_complexe': 0.973259299993515, 'admiration': 0.9897561967372894, 'autre': 0.9999709139333496, 'colere': 0.8198982775211334, 'culpabilite': 0.12354350935084915, 'degout': 0.19585859875543213, 'embarras': 0.9554416239261627, 'fierte': 0.8125306069850922, 'jalousie': 0.016897154109312057, 'joie': 0.9454100728034973, 'peur': 0.9859553873538971, 'surprise': 0.9732445478439331, 'tristesse': 0.6430991590023041}
```
```output

```
cell
```python
"""
Reproduction des outputs web via :
  Phase 1 – Recherche du meilleur template
  Phase 2 – Recherche de permutations de labels
  Phase 3 – Seuil optimal σ(logit_j) ≥ t
  Phase 4 – Validation
"""

import torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from itertools import combinations

# ══════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = (
    AutoModelForSequenceClassification
    .from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
    .to(DEVICE).eval()
)

sentences = df["sentence"].astype(str).tolist()
EOS = tokenizer.eos_token

LABELS = [
    "emo","emo_comportementale","emo_designee","emo_montree",
    "emo_suggeree","emo_base","emo_complexe","admiration","autre",
    "colere","culpabilite","degout","embarras","fierte","jalousie",
    "joie","peur","surprise","tristesse"
]

Y = df[LABELS].values.astype(int)

N, K = Y.shape
print(f"{N} phrases × {K} labels")

# ══════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-500,500)))

@torch.no_grad()
def compute_logits(template):
    out = np.empty((N,K))
    for i,s in enumerate(sentences):
        enc = tokenizer(
            template(s),
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)

        out[i] = model(**enc).logits.squeeze(0).cpu().numpy()

    return out


def find_best_thresholds(probs):

    thr = np.empty(K)
    total = 0

    for j in range(K):

        p = probs[:,j]
        y = Y[:,j]

        vals = np.unique(p)

        cuts = np.concatenate([
            [vals[0]-1e-9],
            (vals[:-1]+vals[1:])/2,
            [vals[-1]+1e-9]
        ])

        errs = np.array([
            ((p>=c).astype(int)!=y).sum()
            for c in cuts
        ])

        best = errs.argmin()

        thr[j] = cuts[best]
        total += errs[best]

    return thr,total


# ══════════════════════════════════════════════════════
# PHASE 1 · Recherche du template
# ══════════════════════════════════════════════════════

templates = {

"raw":lambda s:s,
"strip":lambda s:s.strip(),
"lower":lambda s:s.lower(),

"bca_v1":lambda s:f"before: {EOS}current: {s}{EOS}after: {EOS}",
"bca_v2":lambda s:f"before: {EOS} current: {s} after: {EOS}",
"bca_v3":lambda s:f"before:{EOS}current:{s}{EOS}after:{EOS}",
"bca_v4":lambda s:f"before: {EOS} current: {s}{EOS} after: {EOS}",

"eos_wrap":lambda s:f"{EOS}{s}{EOS}",
"eos_left":lambda s:f"{EOS} {s}",
"eos_right":lambda s:f"{s}{EOS}"

}

best = dict(err=N*K+1)

for name,fn in templates.items():

    logits = compute_logits(fn)
    probs = sigmoid(logits)

    thr,err = find_best_thresholds(probs)

    print(name,err)

    if err < best["err"]:
        best=dict(
            name=name,
            fn=fn,
            logits=logits,
            probs=probs,
            err=err
        )

BEST_FMT = best["fn"]

X = best["logits"]
P = best["probs"]

print("BEST TEMPLATE:",best["name"])
print()


# ══════════════════════════════════════════════════════
# PHASE 2 · swaps de labels
# ══════════════════════════════════════════════════════

applied_swaps=[]

improving=True

while improving:

    improving=False

    _,cur=find_best_thresholds(P)

    for a,b in combinations(range(K),2):

        Ps=P.copy()
        Ps[:,[a,b]]=Ps[:,[b,a]]

        _,e=find_best_thresholds(Ps)

        if e < cur:

            print("swap",LABELS[a],LABELS[b],cur,"→",e)

            P[:,[a,b]]=P[:,[b,a]]
            X[:,[a,b]]=X[:,[b,a]]

            applied_swaps.append((a,b))

            improving=True
            break


thr,err=find_best_thresholds(P)

print("errors after swaps:",err)


# ══════════════════════════════════════════════════════
# PHASE 3 · seuils finaux
# ══════════════════════════════════════════════════════

thresholds = thr

print("\nThresholds")

for j in range(K):
    print(LABELS[j],thresholds[j])


# ══════════════════════════════════════════════════════
# PHASE 4 · validation
# ══════════════════════════════════════════════════════

def classify_batch(logits):

    probs=sigmoid(logits)

    return (probs>=thresholds).astype(int)


Y_pred = classify_batch(X)

n_err = (Y_pred!=Y).sum()

print("\nErrors:",n_err,"/",N*K)


# ══════════════════════════════════════════════════════
# fonction réutilisable
# ══════════════════════════════════════════════════════

@torch.no_grad()
def predict_web(sentence):

    enc = tokenizer(
        BEST_FMT(sentence),
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    logits=model(**enc).logits.squeeze(0).cpu().numpy()

    for a,b in applied_swaps:
        logits[a],logits[b]=logits[b],logits[a]

    probs=sigmoid(logits)

    pred=(probs>=thresholds).astype(int)

    return {LABELS[j]:int(pred[j]) for j in range(K)}
```
```output
Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]2451 phrases × 19 labels
raw 3532
strip 3532
lower 3587
bca_v1 2113
bca_v2 2166
bca_v3 2095
bca_v4 2113
eos_wrap 2672
eos_left 3214
eos_right 2826
BEST TEMPLATE: bca_v3

swap admiration autre 2095 → 1935
errors after swaps: 1935

Thresholds
emo 0.5717521566721935
emo_comportementale 0.8346100973884136
emo_designee 0.46738021166954014
emo_montree 0.8828845547904867
emo_suggeree 0.8002183776251559
emo_base 0.25774435083136205
emo_complexe 0.9798978796636761
admiration 0.9495145680696742
autre 0.9531926895718311
colere 0.28217218720548165
culpabilite 0.12671495241969652
degout 0.19269005632824862
embarras 0.9548280448988165
fierte 0.8002327448859459
jalousie 0.017136900811277365
joie 0.9155047132251537
peur 0.9881862235180032
surprise 0.9722425408373772
tristesse 0.6984491339960737

Errors: 1935 / 46569
```
