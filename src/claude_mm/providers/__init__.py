"""
LLM Provider abstraction layer.

This module provides a uniform interface for interacting with different LLM providers
(OpenAI, Google, Anthropic, etc.) with support for both sync and async operations.
"""

from .anthropic import AnthropicProvider
from .base import Provider, ProviderError, ProviderResponse
from .google import GoogleProvider
from .openai import OpenAIProvider

__all__ = [
    "Provider",
    "ProviderResponse",
    "ProviderError",
    "OpenAIProvider",
    "GoogleProvider",
    "AnthropicProvider",
]


def get_provider(name: str, **kwargs) -> Provider:
    """
    Factory function to get a provider instance.

    Args:
        name: Provider name ('openai', 'google', 'anthropic')
        **kwargs: Provider-specific configuration

    Returns:
        Provider instance

    Raises:
        ValueError: If provider name is unknown
    """
    providers = {
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "anthropic": AnthropicProvider,
    }

    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")

    return providers[name](**kwargs)
