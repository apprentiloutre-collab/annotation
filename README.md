# 🏷️ Annotation LLM — Cyberharcèlement

Annotation automatisée des émotions dans des messages de cyberharcèlement (11-18 ans, français) via différents LLMs.

## Architecture

```
Annotation/
├── data/                          # Fichiers XLSX sources
├── scripts/
│   ├── annotate.py                # Annotation LLM (CLI)
│   ├── compare.py                 # Comparaison inter-runs (CLI)
│   └── supervise.py               # Supervision manuelle (notebook)
├── src/cyberagg_llm_annot/        # Bibliothèque interne
│   ├── llm_providers.py           # Providers LLM (Bedrock, Gemini, HF)
│   ├── prompt_utils.py            # Prompts et taxonomy
│   ├── runner.py                  # Boucle d'annotation + persistance
│   ├── context.py                 # Fenêtre contextuelle
│   ├── parsing.py                 # Parsing annotations experts
│   └── io_utils.py                # I/O fichiers
├── outputs/                       # Résultats (.jsonl, .xlsx)
├── notebooks/                     # Notebooks d'orchestration
└── requirements.txt
```

## Installation

```bash
git clone <repo-url> && cd Annotation
pip install -r requirements.txt
```

## Providers LLM supportés

| Provider | Modèles | Environnement |
|---|---|---|
| **AWS Bedrock** | `claude-sonnet-4-6`, `claude-opus-4-6`, `mistral-pixtral` | AWS credentials |
| **Google Gemini** | `gemini-flash` | Google Colab uniquement |
| **HuggingFace** | `deepseek-ai/DeepSeek-V3.2:novita`, etc. | Token `HF_TOKEN` |

## Utilisation

### 1. Annotation

```bash
# Bedrock Claude (défaut)
python scripts/annotate.py \
    --xlsx data/homophobie_scenario_julie.xlsx \
    --thematique homophobie \
    --run_id homophobie_run001

# Bedrock Mistral Pixtral
python scripts/annotate.py \
    --xlsx data/mon_fichier.xlsx \
    --thematique homophobie \
    --run_id run_pixtral \
    --model mistral-pixtral

# HuggingFace DeepSeek
python scripts/annotate.py \
    --xlsx data/mon_fichier.xlsx \
    --thematique racisme \
    --run_id run_deepseek \
    --model_provider huggingface \
    --model "deepseek-ai/DeepSeek-V3.2:novita"

# Gemini Flash (Colab)
python scripts/annotate.py \
    --xlsx data/mon_fichier.xlsx \
    --thematique homophobie \
    --run_id run_gemini \
    --model_provider gemini

# Avec annotations d'experts
python scripts/annotate.py \
    --xlsx data/mon_fichier.xlsx \
    --thematique homophobie \
    --run_id run002 \
    --use_annotations
```

### 2. Comparaison inter-runs

```bash
python scripts/compare.py \
    --run1 outputs/homophobie/homophobie_run001.jsonl \
    --run2 outputs/homophobie/homophobie_run002.jsonl \
    --xlsx data/homophobie_scenario_julie.xlsx \
    --label_run1 "avec experts" \
    --label_run2 "sans experts"
```

### 3. Supervision manuelle (notebook)

```python
%run scripts/supervise.py \
    --run1 outputs/homophobie/run001.jsonl \
    --run2 outputs/homophobie/run002.jsonl \
    --xlsx data/homophobie_scenario_julie.xlsx
```

## Gestion des outputs

- Le fichier `.jsonl` est l'artefact principal (append-only)
- Les JSON individuels (`items/`) sont temporaires et **supprimés automatiquement** en fin de run
- Le dossier `items/` est dans `.gitignore`

## Émotions annotées (11)

Colère · Dégoût · Joie · Peur · Surprise · Tristesse · Admiration · Culpabilité · Embarras · Fierté · Jalousie