from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, Tuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Codes d'erreur Bedrock qui méritent un retry automatique
_RETRYABLE = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
})


def make_bedrock_client(region_name: str = "eu-north-1"):
    return boto3.client("bedrock-runtime", region_name=region_name)


def invoke_claude(
    client,
    system_prompt: str,
    user_message: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    model_id: str = "global.anthropic.claude-sonnet-4-6",
    anthropic_version: str = "bedrock-2023-05-31",
    max_retries: int = 6,
    base_delay: float = 2.0,
) -> Dict[str, Any]:
    """
    Appel bloquant avec retry exponentiel sur les erreurs transitoires.
    Utilise le paramètre *system* de l'API Claude (meilleur que tout en user).
    """
    body = {
        "anthropic_version": anthropic_version,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    for attempt in range(max_retries + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
            )
            return json.loads(resp["body"].read())

        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in _RETRYABLE and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Bedrock %s (attempt %d/%d) — retry in %.1fs",
                    code, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
            else:
                raise


def extract_text(result: Dict[str, Any]) -> str:
    """Extrait le texte de la première content block."""
    return result["content"][0]["text"]


def check_stop_reason(result: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Retourne (is_complete, reason).
    is_complete=True si stop_reason=="end_turn" (le modèle a terminé normalement).
    """
    reason = result.get("stop_reason", "unknown")
    return reason == "end_turn", reason
