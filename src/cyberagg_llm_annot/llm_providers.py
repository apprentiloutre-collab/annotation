"""
Système de providers LLM unifié.

Supporte :
  - AWS Bedrock  (Claude, Mistral Pixtral)
  - Google Colab Gemini Flash
  - HuggingFace Inference Providers (OpenAI-compatible)
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Interface commune
# ═══════════════════════════════════════════════════════════════════════════

class LLMProvider(ABC):
    """Interface abstraite pour tous les providers LLM."""

    @abstractmethod
    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Appel bloquant au LLM. Retourne un dict résultat brut."""
        ...

    @abstractmethod
    def extract_text(self, result: Dict[str, Any]) -> str:
        """Extrait le texte de la réponse."""
        ...

    @abstractmethod
    def check_stop_reason(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Retourne (is_complete, reason)."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  AWS Bedrock (Claude / Mistral Pixtral)
# ═══════════════════════════════════════════════════════════════════════════

# Codes d'erreur transitoires méritant un retry
_BEDROCK_RETRYABLE = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
})

# Alias → model_id Bedrock
BEDROCK_MODEL_IDS = {
    "claude-sonnet-4-6":  "global.anthropic.claude-sonnet-4-6",
    "claude-opus-4-6":    "anthropic.claude-opus-4-6-v1",
    "mistral-pixtral":    "mistral.pixtral-large-2502-v1:0",
}


class BedrockProvider(LLMProvider):
    """Provider AWS Bedrock (Anthropic Claude & Mistral Pixtral)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        region_name: str = "eu-north-1",
    ):
        import boto3
        self.model_id = BEDROCK_MODEL_IDS.get(model, model)
        self.is_anthropic = "anthropic" in self.model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        logger.info("BedrockProvider  model=%s  region=%s", self.model_id, region_name)

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_retries: int = 6,
        base_delay: float = 2.0,
    ) -> Dict[str, Any]:
        if self.is_anthropic:
            body = self._anthropic_body(system_prompt, user_message,
                                        max_tokens, temperature)
        else:
            body = self._mistral_body(system_prompt, user_message,
                                      max_tokens, temperature)

        from botocore.exceptions import ClientError

        for attempt in range(max_retries + 1):
            try:
                resp = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                )
                return json.loads(resp["body"].read())
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code in _BEDROCK_RETRYABLE and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Bedrock %s (attempt %d/%d) — retry in %.1fs",
                        code, attempt + 1, max_retries, delay,
                    )
                    time.sleep(delay)
                else:
                    raise

    # ── Anthropic (Claude) ──────────────────────────────────────────────
    @staticmethod
    def _anthropic_body(system_prompt, user_message, max_tokens, temperature):
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

    # ── Mistral Pixtral ─────────────────────────────────────────────────
    @staticmethod
    def _mistral_body(system_prompt, user_message, max_tokens, temperature):
        return {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

    def extract_text(self, result: Dict[str, Any]) -> str:
        if self.is_anthropic:
            return result["content"][0]["text"]
        # Mistral
        return result["choices"][0]["message"]["content"]

    def check_stop_reason(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        if self.is_anthropic:
            reason = result.get("stop_reason", "unknown")
            return reason == "end_turn", reason
        # Mistral
        reason = result.get("choices", [{}])[0].get("finish_reason", "unknown")
        return reason == "stop", reason


# ═══════════════════════════════════════════════════════════════════════════
#  Google Colab Gemini Flash
# ═══════════════════════════════════════════════════════════════════════════

class ColabGeminiProvider(LLMProvider):
    """
    Provider Gemini Flash via google.colab.ai (sans clé API).
    Disponible uniquement dans un notebook Google Colab.
    """

    def __init__(self, model: str = "gemini-flash", **_kwargs):
        self.model = model
        try:
            from google.colab import ai as _ai  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "ColabGeminiProvider nécessite google.colab. "
                "Exécutez ce script depuis un notebook Google Colab."
            )
        logger.info("ColabGeminiProvider initialisé")

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        from google.colab import ai

        full_prompt = f"{system_prompt}\n\n{user_message}"
        text = ai.generate_text(full_prompt)
        return {
            "text": text,
            "stop_reason": "end_turn",
            "provider": "colab_gemini",
        }

    def extract_text(self, result: Dict[str, Any]) -> str:
        return result["text"]

    def check_stop_reason(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        return True, result.get("stop_reason", "end_turn")


# ═══════════════════════════════════════════════════════════════════════════
#  HuggingFace Inference Providers (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════

class HuggingFaceProvider(LLMProvider):
    """
    Provider HuggingFace via l'API OpenAI-compatible.
    Nécessite `openai` et un token HF.
    """

    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-V3.2:novita",
        hf_token: Optional[str] = None,
        **_kwargs,
    ):
        from openai import OpenAI

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            # Essayer via google.colab.userdata (Colab)
            try:
                from google.colab import userdata
                token = userdata.get("HF_TOKEN")
            except Exception:
                pass
        if not token:
            raise ValueError(
                "Token HuggingFace non trouvé. Définissez HF_TOKEN via "
                "variable d'environnement ou google.colab.userdata."
            )

        self.model = model
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )
        logger.info("HuggingFaceProvider  model=%s", model)

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        choice = completion.choices[0]
        return {
            "text": choice.message.content,
            "stop_reason": choice.finish_reason or "stop",
            "provider": "huggingface",
            "model": self.model,
            "usage": {
                "input_tokens": getattr(completion.usage, "prompt_tokens", None),
                "output_tokens": getattr(completion.usage, "completion_tokens", None),
            },
        }

    def extract_text(self, result: Dict[str, Any]) -> str:
        return result["text"]

    def check_stop_reason(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        reason = result.get("stop_reason", "unknown")
        return reason in ("stop", "end_turn"), reason


# ═══════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════

_PROVIDERS = {
    "bedrock":     BedrockProvider,
    "gemini":      ColabGeminiProvider,
    "huggingface": HuggingFaceProvider,
}


def get_provider(provider_name: str, model: str, **kwargs) -> LLMProvider:
    """
    Factory pour obtenir un provider LLM.

    Args:
        provider_name: 'bedrock', 'gemini', ou 'huggingface'
        model: identifiant ou alias du modèle
        **kwargs: arguments supplémentaires passés au constructeur
    """
    cls = _PROVIDERS.get(provider_name.lower())
    if cls is None:
        raise ValueError(
            f"Provider inconnu : '{provider_name}'. "
            f"Choix : {', '.join(_PROVIDERS)}"
        )
    return cls(model=model, **kwargs)
