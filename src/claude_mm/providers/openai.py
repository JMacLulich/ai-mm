"""OpenAI provider implementation."""

import os
from typing import Optional

from claude_mm.pricing import get_model_pricing
from claude_mm.retry import retry_with_backoff

from .base import Provider, ProviderError, ProviderResponse


class OpenAIProvider(Provider):
    """Provider for OpenAI models (GPT series)."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)

        # Get API key from env if not provided
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ProviderError(
                "OPENAI_API_KEY not set. Set via environment or pass to constructor."
            )

    @retry_with_backoff(max_attempts=3, initial_delay=1, max_delay=10)
    def complete(
        self,
        prompt: str,
        model: str = "gpt-5.2",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        """
        Synchronous OpenAI completion.

        Args:
            prompt: User prompt
            model: Model identifier (e.g., 'gpt-5.2', 'gpt-4o')
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters

        Returns:
            ProviderResponse with completion and usage
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("openai package not installed. Run: pip install openai")

        client = OpenAI(api_key=self.api_key)

        if not system_prompt:
            system_prompt = "You are a helpful AI assistant."

        try:
            # Build request parameters
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }

            # GPT-5.2 models don't support temperature parameter
            if not model.startswith("gpt-5"):
                params["temperature"] = temperature

            if max_tokens:
                params["max_tokens"] = max_tokens

            # Add any additional parameters
            params.update(kwargs)

            response = client.chat.completions.create(**params)

            # Calculate cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = self.estimate_cost(input_tokens, output_tokens, model)

            return ProviderResponse(
                text=response.choices[0].message.content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                cached=False,
            )

        except Exception as e:
            raise ProviderError(f"OpenAI API error: {e}")

    async def complete_async(
        self,
        prompt: str,
        model: str = "gpt-5.2",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        """
        Asynchronous OpenAI completion.

        Args:
            prompt: User prompt
            model: Model identifier
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters

        Returns:
            ProviderResponse with completion and usage
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ProviderError("openai package not installed. Run: pip install openai")

        client = AsyncOpenAI(api_key=self.api_key)

        if not system_prompt:
            system_prompt = "You are a helpful AI assistant."

        try:
            # Build request parameters
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }

            # GPT-5.2 models don't support temperature parameter
            if not model.startswith("gpt-5"):
                params["temperature"] = temperature

            if max_tokens:
                params["max_tokens"] = max_tokens

            # Add any additional parameters
            params.update(kwargs)

            response = await client.chat.completions.create(**params)

            # Calculate cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = self.estimate_cost(input_tokens, output_tokens, model)

            return ProviderResponse(
                text=response.choices[0].message.content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                cached=False,
            )

        except Exception as e:
            raise ProviderError(f"OpenAI API error: {e}")

    def get_model_info(self, model: str) -> dict:
        """Get OpenAI model information."""
        pricing = get_model_pricing("openai", model)

        return {
            "provider": "openai",
            "model": model,
            "pricing": pricing,
            "context_window": 128000 if model.startswith("gpt-5") else 8192,
        }

    def validate_key(self) -> tuple[bool, str]:
        """Validate the OpenAI API key."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return True, "Valid"
        except Exception as e:
            return False, str(e)
