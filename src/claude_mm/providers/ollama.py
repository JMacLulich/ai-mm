"""Ollama (local LLM) provider implementation."""

import json
import os
import urllib.error
import urllib.request
from decimal import Decimal
from typing import Optional

from claude_mm.pricing import get_model_pricing
from claude_mm.retry import retry_with_backoff

from .base import Provider, ProviderError, ProviderResponse


class OllamaProvider(Provider):
    """Provider for local Ollama models."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """
        Initialize Ollama provider.

        Args:
            api_key: Not used for Ollama (local)
            base_url: Ollama API URL (defaults to OLLAMA_BASE_URL env var)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)

        resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL")
        if not resolved_base_url:
            raise ProviderError(
                "OLLAMA_BASE_URL not set. Configure it via 'ai config' or export OLLAMA_BASE_URL."
            )

        self.base_url = resolved_base_url.rstrip("/")

    def _is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    @retry_with_backoff(max_attempts=2, initial_delay=1, max_delay=5)
    def complete(
        self,
        prompt: str,
        model: str = "qwen2.5:14b-instruct",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        """
        Synchronous Ollama completion.

        Args:
            prompt: User prompt
            model: Model identifier (e.g., 'qwen2.5:14b-instruct')
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Ollama parameters

        Returns:
            ProviderResponse with completion and usage
        """
        if not self._is_available():
            raise ProviderError("Ollama server not running. Start with: ollama serve")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            text = result.get("message", {}).get("content", "")

            eval_count = result.get("eval_count", 0) or result.get("message", {}).get(
                "eval_count", 0
            )
            prompt_eval_count = result.get("prompt_eval_count", 0) or result.get("message", {}).get(
                "prompt_eval_count", 0
            )

            input_tokens = prompt_eval_count
            output_tokens = eval_count

            return ProviderResponse(
                text=text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=Decimal("0"),
                cached=False,
            )

        except urllib.error.URLError as e:
            raise ProviderError(f"Ollama connection error: {e}")
        except json.JSONDecodeError as e:
            raise ProviderError(f"Ollama response parse error: {e}")
        except Exception as e:
            raise ProviderError(f"Ollama API error: {e}")

    async def complete_async(
        self,
        prompt: str,
        model: str = "qwen2.5:14b-instruct",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        """Async not implemented for Ollama - uses sync."""
        return self.complete(prompt, model, system_prompt, temperature, max_tokens, **kwargs)

    def get_model_info(self, model: str) -> dict:
        """Get Ollama model information."""
        pricing = get_model_pricing("ollama", model)

        return {
            "provider": "ollama",
            "model": model,
            "pricing": pricing,
            "context_window": 128000,
        }

    def validate_key(self) -> tuple[bool, str]:
        """Check if Ollama server is running."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True, "Running"
            return False, "Ollama not responding"
        except Exception as e:
            return False, f"Ollama not running: {e}"
