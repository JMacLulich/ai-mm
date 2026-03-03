"""LM Studio provider implementation (OpenAI-compatible local endpoint)."""

import os
from decimal import Decimal
from typing import Optional

from .base import Provider, ProviderError, ProviderResponse


class LMStudioProvider(Provider):
    """Provider for locally hosted models exposed by LM Studio."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)

        resolved_base_url = (
            base_url or os.getenv("LMSTUDIO_BASE_URL") or "http://127.0.0.1:1234/v1"
        ).rstrip("/")
        if not resolved_base_url.endswith("/v1"):
            resolved_base_url = f"{resolved_base_url}/v1"

        self.base_url = resolved_base_url
        self.api_key = self.api_key or os.getenv("LMSTUDIO_API_KEY") or "lm-studio"

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("openai package not installed. Run: pip install openai")

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _extract_text(response) -> str:
        try:
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def complete(
        self,
        prompt: str,
        model: str = "qwen3.5:27b",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        client = self._client()

        if not system_prompt:
            system_prompt = "You are a helpful AI assistant."

        try:
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }

            if max_tokens:
                params["max_tokens"] = max_tokens

            params.update(kwargs)
            response = client.chat.completions.create(**params)

            if not getattr(response, "choices", None):
                error_details = getattr(response, "error", None)
                hint = (
                    "Check LMSTUDIO_BASE_URL "
                    "(OpenAI-compatible endpoints usually require a /v1 suffix)."
                )
                raise ProviderError(
                    "LM Studio API returned no choices. "
                    f"{hint} Raw error: {error_details or 'unknown'}"
                )

            text = self._extract_text(response)
            if not text.strip():
                hint = "Check LMSTUDIO_BASE_URL and model behavior for empty/think-only responses."
                raise ProviderError(f"LM Studio returned empty response content. {hint}")

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            return ProviderResponse(
                text=text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=Decimal("0"),
                cached=False,
            )
        except Exception as e:
            raise ProviderError(f"LM Studio API error: {e}")

    async def complete_async(
        self,
        prompt: str,
        model: str = "qwen3.5:27b",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        return self.complete(prompt, model, system_prompt, temperature, max_tokens, **kwargs)

    def get_model_info(self, model: str) -> dict:
        return {
            "provider": "lmstudio",
            "model": model,
            "pricing": {"input": 0.0, "output": 0.0},
            "context_window": 128000,
        }

    def validate_key(self) -> tuple[bool, str]:
        try:
            self.complete(prompt="Reply with OK", model="qwen3.5:27b", max_tokens=8, temperature=0)
            return True, "Running"
        except Exception as e:
            return False, str(e)
