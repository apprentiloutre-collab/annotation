from __future__ import annotations
from typing import Any, Dict, List, Optional

# ── Colonnes du corpus existant (annotations d'experts) ────────────────────
DEFAULT_LABEL_COLS = [
    "ROLE", "HATE", "TARGET", "VERBAL_ABUSE",
    "INTENTION", "CONTEXT", "SENTIMENT",
]

# ── 11 émotions cibles ─────────────────────────────────────────────────────
EMOTIONS = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
]

# ── System prompt (injecté via le paramètre "system" de Bedrock) ───────────
SYSTEM_PROMPT = """\
Tu es un annotateur expert en émotions. Ta tâche : produire des étiquettes émotionnelles (multi-label, 0/1) pour UN SEUL message cible (TARGET) écrit par des adolescents (11–18 ans) dans des conversations de cyber-harcèlement en français.

RÈGLES STRICTES
1. Annote UNIQUEMENT le message TARGET. PREV/NEXT = contexte conversationnel.
2. Étiquette les émotions EXPRIMÉES par l'auteur du TARGET \
   (pas celles de la victime, ni l'effet sur le lecteur).
3. Multi-label autorisé — active au maximum 3 émotions. \
   En cas de forte hésitation, préfère 0 et signale l'ambiguïté.
4. "mdr", "ptdr", emojis rieurs → joie OU moquerie/attaque. \
   Active "Joie" uniquement si un affect positif est réellement exprimé.
5. Si des annotations d'experts sont fournies (EXPERT_ANNOTATIONS), \
   utilise-les comme indice contextuel mais NE LES COPIE PAS aveuglément.
6. Renvoie UNIQUEMENT un objet JSON brut. \

TAXONOMIE — 11 émotions (valeurs 0 ou 1)
• Colère       — irritation, agressivité, menaces, injonctions hostiles, insultes
• Dégoût       — répulsion, mépris corporel/moral
• Joie         — amusement, plaisir (y compris rire moqueur si affect positif)
• Peur         — inquiétude, menace subie, anxiété
• Surprise     — étonnement réel ("hein?", "wtf?")
• Tristesse    — peine, découragement, désespoir
• Admiration   — estime, soutien valorisant (rare en contexte de harcèlement)
• Culpabilité  — remords, auto-reproche
• Embarras     — honte, gêne, humiliation ressentie
• Fierté       — vantardise, domination, satisfaction de soi
• Jalousie     — envie hostile/possessive, rancœur comparative

PROCÉDURE INTERNE (applique sans détailler dans la sortie)
- Décode l'argot / abréviations adolescentes.
- Identifie si le TARGET est une attaque, défense, soutien ou sarcasme.
- Décide 0/1 pour chaque émotion selon le TARGET uniquement.

FORMAT DE SORTIE — JSON strict, rien d'autre
{
  "metadata": {
    "topic": "<thématique>",
    "used_expert_annotations": true|false
  },
  "emotions": {
    "Colère": 0, "Dégoût": 0, "Joie": 0, "Peur": 0,
    "Surprise": 0, "Tristesse": 0, "Admiration": 0,
    "Culpabilité": 0, "Embarras": 0, "Fierté": 0, "Jalousie": 0
  },
  "rationale_short": "Une phrase concise citant 1-2 indices linguistiques du TARGET.",
  "ambiguities": ["indices ambigus (max 3)"]
}"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def build_annotations_block(
    parsed_labels: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Transforme les cellules parsées en bloc structuré pour le prompt."""
    block: Dict[str, Any] = {}
    for k, v in parsed_labels.items():
        if v["status"] == "value":
            block[k] = {"status": "value", "value": v["value"]}
        elif v["status"] == "no_consensus":
            block[k] = {"status": "no_consensus", "raw": v["raw"]}
        else:
            block[k] = {"status": "missing"}
    return block


def _is_block_empty(block: Dict[str, Any]) -> bool:
    """Vrai si toutes les annotations sont missing."""
    return all(v["status"] == "missing" for v in block.values())


def _fmt_msg(label: str, msg_repr: Optional[Dict[str, Any]]) -> str:
    if msg_repr is None:
        return f"{label}: (aucun message)"
    name = msg_repr.get("NAME", "?")
    role = msg_repr.get("ROLE", "")
    time = msg_repr.get("TIME", "")
    text = msg_repr.get("TEXT", "")
    header = f"[{name}]"
    if role:
        header += f" (role={role})"
    if time:
        header += f" (time={time})"
    return f'{label}: {header} "{text}"'


def build_user_message(
    thematique: str,
    prev_repr: Optional[Dict[str, Any]],
    target_repr: Dict[str, Any],
    next_repr: Optional[Dict[str, Any]],
    annotations_block: Optional[Dict[str, Any]] = None,
) -> str:
    """Construit le message *user* injecté dans la requête Bedrock."""
    lines: List[str] = [f"THÉMATIQUE: {thematique}", ""]

    # ── Fenêtre de contexte ──
    lines.append("<CONTEXT>")
    lines.append(_fmt_msg("PREV",   prev_repr))
    lines.append(_fmt_msg("TARGET", target_repr))
    lines.append(_fmt_msg("NEXT",   next_repr))
    lines.append("</CONTEXT>")

    # ── Annotations existantes (conditionnel) ──
    if annotations_block and not _is_block_empty(annotations_block):
        lines += ["", "<EXPERT_ANNOTATIONS>"]
        for col, info in annotations_block.items():
            if info["status"] == "value":
                lines.append(f"  {col}: {info['value']}")
            elif info["status"] == "no_consensus":
                lines.append(f"  {col}: NO_CONSENSUS")
            # on n'affiche pas les MISSING
        lines.append("</EXPERT_ANNOTATIONS>")

    lines += [
        "",
        "Annote le message TARGET ci-dessus. "
        "Renvoie UNIQUEMENT le JSON demandé, sans markdown ni commentaire.",
    ]
    return "\n".join(lines)
