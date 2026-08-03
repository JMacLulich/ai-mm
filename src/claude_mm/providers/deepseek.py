"""DeepSeek provider implementation using its OpenAI-compatible API."""

import asyncio
import os
from typing import Optional

from claude_mm.pricing import get_model_pricing

from .base import Provider, ProviderError, ProviderResponse

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _normalize_reasoning_effort(reasoning_effort: Optional[str]) -> tuple[bool, Optional[str]]:
    """Translate shared CLI effort names into DeepSeek's high/max contract."""
    if reasoning_effort == "none":
        return False, None
    if reasoning_effort in {"xhigh", "max"}:
        return True, "max"
    if reasoning_effort in {"minimal", "low", "medium", "high"}:
        return True, "high"
    if reasoning_effort is None:
        return True, None
    raise ProviderError(
        "Unsupported DeepSeek reasoning effort. Use none, minimal, low, medium, high, "
        "xhigh, or max."
    )


class DeepSeekProvider(Provider):
    """Provider for DeepSeek V4 Pro and Flash models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(api_key, **kwargs)
        self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "DEEPSEEK_API_KEY not set. Set it via the environment or pass it to the "
                "provider constructor."
            )

        self.base_url = (
            base_url or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
        ).rstrip("/")

    def _client(self, timeout: float):
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("openai package not installed. Run: pip install openai")

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,
        )

    def complete(
        self,
        prompt: str,
        model: str = "deepseek-v4-pro",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> ProviderResponse:
        thinking_enabled, applied_effort = _normalize_reasoning_effort(reasoning_effort)
        timeout_setting = "thinking_timeout" if applied_effort == "max" else "timeout"
        default_timeout = 300.0 if applied_effort == "max" else 90.0
        timeout = float(self.config.get(timeout_setting, default_timeout))
        client = self._client(timeout)

        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "extra_body": {
                "thinking": {"type": "enabled" if thinking_enabled else "disabled"}
            },
        }
        if applied_effort:
            params["reasoning_effort"] = applied_effort
        if not thinking_enabled:
            params["temperature"] = temperature
        if max_tokens:
            params["max_tokens"] = max_tokens
        params.update(kwargs)

        try:
            response = client.chat.completions.create(**params)
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            return ProviderResponse(
                text=response.choices[0].message.content or "",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=self.estimate_cost(input_tokens, output_tokens, model),
                cached=False,
                metadata={
                    "reasoning_effort_requested": reasoning_effort,
                    "reasoning_effort_applied": applied_effort or (
                        "provider-default" if thinking_enabled else "none"
                    ),
                },
            )
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"DeepSeek API error: {e}")

    async def complete_async(
        self,
        prompt: str,
        model: str = "deepseek-v4-pro",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> ProviderResponse:
        return await asyncio.to_thread(
            self.complete,
            prompt,
            model,
            system_prompt,
            temperature,
            max_tokens,
            reasoning_effort,
            **kwargs,
        )

    def get_model_info(self, model: str) -> dict:
        return {
            "provider": "deepseek",
            "model": model,
            "pricing": get_model_pricing("deepseek", model),
            "context_window": 1000000,
            "max_output_tokens": 384000,
        }

    def validate_key(self) -> tuple[bool, str]:
        try:
            self.complete(
                prompt="Reply with OK",
                model="deepseek-v4-flash",
                max_tokens=5,
                reasoning_effort="none",
            )
            return True, "Valid"
        except Exception as e:
            return False, str(e)
