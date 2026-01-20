"""Google (Gemini) provider implementation."""

import os
from typing import Optional

from pricing import get_model_pricing
from retry import retry_with_backoff

from .base import Provider, ProviderError, ProviderResponse


class GoogleProvider(Provider):
    """Provider for Google Gemini models."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Google provider.

        Args:
            api_key: Google AI API key (defaults to GOOGLE_AI_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)

        # Get API key from env if not provided
        if not self.api_key:
            self.api_key = os.getenv("GOOGLE_AI_API_KEY")

        if not self.api_key:
            raise ProviderError(
                "GOOGLE_AI_API_KEY not set. "
                "Set via environment or pass to constructor."
            )

    @retry_with_backoff(max_attempts=3, initial_delay=1, max_delay=10)
    def complete(
        self,
        prompt: str,
        model: str = "gemini-3-flash-preview",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ProviderResponse:
        """
        Synchronous Google Gemini completion.

        Args:
            prompt: User prompt
            model: Model identifier (e.g., 'gemini-3-flash-preview')
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Gemini parameters

        Returns:
            ProviderResponse with completion and usage
        """
        try:
            from google import genai
        except ImportError:
            raise ProviderError(
                "google-genai package not installed. Run: pip install google-genai"
            )

        client = genai.Client(api_key=self.api_key)

        # Gemini combines system and user prompts
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            # Build request parameters
            params = {
                "model": model,
                "contents": full_prompt,
            }

            # Add generation config if specified
            if temperature != 0.7 or max_tokens or kwargs:
                config = {}
                if temperature != 0.7:
                    config["temperature"] = temperature
                if max_tokens:
                    config["max_output_tokens"] = max_tokens
                config.update(kwargs)
                params["config"] = config

            response = client.models.generate_content(**params)

            # Extract usage info
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

            # Calculate cost
            cost = self.estimate_cost(input_tokens, output_tokens, model)

            return ProviderResponse(
                text=response.text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                cached=False,
            )

        except Exception as e:
            raise ProviderError(f"Google Gemini API error: {e}")

    async def complete_async(
        self,
        prompt: str,
        model: str = "gemini-3-flash-preview",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ProviderResponse:
        """
        Asynchronous Google Gemini completion.

        Args:
            prompt: User prompt
            model: Model identifier
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Gemini parameters

        Returns:
            ProviderResponse with completion and usage
        """
        try:
            from google import genai
        except ImportError:
            raise ProviderError(
                "google-genai package not installed. Run: pip install google-genai"
            )

        # Note: google-genai doesn't have native async support yet
        # We'll use the sync method for now and wrap it if needed
        # This is a known limitation we can address when google-genai adds async
        return self.complete(prompt, model, system_prompt, temperature, max_tokens, **kwargs)

    def get_model_info(self, model: str) -> dict:
        """Get Google Gemini model information."""
        pricing = get_model_pricing("google", model)

        return {
            "provider": "google",
            "model": model,
            "pricing": pricing,
            "context_window": 1000000,  # 1M tokens for Gemini models
        }
