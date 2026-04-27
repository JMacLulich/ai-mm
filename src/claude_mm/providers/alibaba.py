"""Alibaba Cloud DashScope provider implementation."""

import os
from typing import Optional

from claude_mm.pricing import get_model_pricing
from claude_mm.retry import retry_with_backoff

from .base import Provider, ProviderError, ProviderResponse

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class AlibabaProvider(Provider):
    """Provider for Alibaba Cloud DashScope OpenAI-compatible models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(api_key, **kwargs)

        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_API_KEY")

        if not self.api_key:
            raise ProviderError(
                "DASHSCOPE_API_KEY not set. Set via environment or pass to constructor."
            )

        resolved_base_url = (
            base_url
            or os.getenv("DASHSCOPE_BASE_URL")
            or os.getenv("ALIBABA_BASE_URL")
            or DEFAULT_DASHSCOPE_BASE_URL
        )
        self.base_url = resolved_base_url.rstrip("/")

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("openai package not installed. Run: pip install openai")

        timeout = float(self.config.get("timeout", 60.0))
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)

    @retry_with_backoff(max_attempts=3, initial_delay=1, max_delay=10)
    def complete(
        self,
        prompt: str,
        model: str = "qwen3.6-35b-a3b",
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
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = self.estimate_cost(input_tokens, output_tokens, model)

            return ProviderResponse(
                text=response.choices[0].message.content or "",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                cached=False,
            )
        except Exception as e:
            raise ProviderError(f"Alibaba DashScope API error: {e}")

    async def complete_async(
        self,
        prompt: str,
        model: str = "qwen3.6-35b-a3b",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        return self.complete(prompt, model, system_prompt, temperature, max_tokens, **kwargs)

    def get_model_info(self, model: str) -> dict:
        pricing = get_model_pricing("alibaba", model)
        return {
            "provider": "alibaba",
            "model": model,
            "pricing": pricing,
            "context_window": 128000,
        }

    def validate_key(self) -> tuple[bool, str]:
        try:
            self.complete(prompt="Reply with OK", model="qwen3.6-35b-a3b", max_tokens=5)
            return True, "Valid"
        except Exception as e:
            return False, str(e)
