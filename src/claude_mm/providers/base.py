"""
Base provider interface for LLM interactions.

All provider implementations must inherit from Provider and implement
the abstract methods for both sync and async operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass
class ProviderResponse:
    """
    Standardized response from any LLM provider.

    This ensures consistent handling of responses regardless of which
    provider is being used.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: Optional[Decimal] = None
    cached: bool = False
    metadata: Optional[dict] = None


class ProviderError(Exception):
    """Base exception for all provider-related errors."""

    pass


class Provider(ABC):
    """
    Abstract base class for LLM providers.

    Providers must implement both sync and async methods to support
    different execution contexts (CLI vs async frameworks).
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the provider.

        Args:
            api_key: API key for authentication (can be None if using env vars)
            **kwargs: Provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    def complete(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        """
        Synchronous completion request.

        Args:
            prompt: User prompt/input
            model: Model identifier
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            ProviderResponse with completion and usage info

        Raises:
            ProviderError: On API or configuration errors
        """
        pass

    @abstractmethod
    async def complete_async(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse:
        """
        Asynchronous completion request.

        Args:
            prompt: User prompt/input
            model: Model identifier
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            ProviderResponse with completion and usage info

        Raises:
            ProviderError: On API or configuration errors
        """
        pass

    @abstractmethod
    def get_model_info(self, model: str) -> dict:
        """
        Get information about a specific model.

        Args:
            model: Model identifier

        Returns:
            Dictionary with model metadata (pricing, context window, etc.)
        """
        pass

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> Decimal:
        """
        Estimate the cost of a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model identifier

        Returns:
            Estimated cost in USD
        """
        model_info = self.get_model_info(model)
        pricing = model_info.get("pricing", {})

        input_cost = (
            Decimal(str(input_tokens)) * Decimal(str(pricing.get("input", 0))) / Decimal("1000000")
        )
        output_cost = (
            Decimal(str(output_tokens))
            * Decimal(str(pricing.get("output", 0)))
            / Decimal("1000000")
        )

        return input_cost + output_cost

    def validate_key(self) -> tuple[bool, str]:
        """
        Validate that the API key is working.

        Returns:
            Tuple of (success, message)
        """
        return True, "Valid (not checked)"
