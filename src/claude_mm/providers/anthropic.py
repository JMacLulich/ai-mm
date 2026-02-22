"""Anthropic (Claude) provider implementation."""

import os
from typing import Optional

from claude_mm.pricing import get_model_pricing
from claude_mm.retry import retry_with_backoff

from .base import Provider, ProviderError, ProviderResponse


class AnthropicProvider(Provider):
    """Provider for Anthropic Claude models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)

        # Get API key from env if not provided
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY not set. Set via environment or pass to constructor."
            )

    @retry_with_backoff(max_attempts=3, initial_delay=1, max_delay=10)
    def complete(
        self,
        prompt: str,
        model: str = "claude-opus-4-6",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        """
        Synchronous Anthropic Claude completion.

        Args:
            prompt: User prompt
            model: Model identifier (e.g., 'claude-opus-4-6')
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Claude parameters

        Returns:
            ProviderResponse with completion and usage
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ProviderError("anthropic package not installed. Run: pip install anthropic")

        client = Anthropic(api_key=self.api_key)

        if not system_prompt:
            system_prompt = "You are a helpful AI assistant."

        # Default max_tokens for Claude
        if not max_tokens:
            max_tokens = 4096

        try:
            # Build request parameters
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "system": system_prompt,
                "temperature": temperature,
            }

            # Add any additional parameters
            params.update(kwargs)

            response = client.messages.create(**params)

            # Extract usage info
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0

            # Calculate cost
            cost = self.estimate_cost(input_tokens, output_tokens, model)

            # Extract text content
            text_content = ""
            for block in response.content:
                if block.type == "text":
                    text_content += block.text

            return ProviderResponse(
                text=text_content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                cached=False,
            )

        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}")

    async def complete_async(
        self,
        prompt: str,
        model: str = "claude-opus-4-6",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        """
        Asynchronous Anthropic Claude completion.

        Args:
            prompt: User prompt
            model: Model identifier
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Claude parameters

        Returns:
            ProviderResponse with completion and usage
        """
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ProviderError("anthropic package not installed. Run: pip install anthropic")

        client = AsyncAnthropic(api_key=self.api_key)

        if not system_prompt:
            system_prompt = "You are a helpful AI assistant."

        # Default max_tokens for Claude
        if not max_tokens:
            max_tokens = 4096

        try:
            # Build request parameters
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "system": system_prompt,
                "temperature": temperature,
            }

            # Add any additional parameters
            params.update(kwargs)

            response = await client.messages.create(**params)

            # Extract usage info
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0

            # Calculate cost
            cost = self.estimate_cost(input_tokens, output_tokens, model)

            # Extract text content
            text_content = ""
            for block in response.content:
                if block.type == "text":
                    text_content += block.text

            return ProviderResponse(
                text=text_content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                cached=False,
            )

        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}")

    def get_model_info(self, model: str) -> dict:
        """Get Anthropic model information."""
        pricing = get_model_pricing("anthropic", model)

        return {
            "provider": "anthropic",
            "model": model,
            "pricing": pricing,
            "context_window": 200000,
        }

    def validate_key(self) -> tuple[bool, str]:
        """Validate the Anthropic API key."""
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)
            client.messages.create(
                model="claude-opus-4-6",
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True, "Valid"
        except Exception as e:
            return False, str(e)
