"""
LLM Provider abstraction layer.

This module provides a uniform interface for interacting with different LLM providers
(OpenAI, DeepSeek, Google, Anthropic, Alibaba, Ollama) with support for sync and async operations.
"""

from .alibaba import AlibabaProvider
from .anthropic import AnthropicProvider
from .base import Provider, ProviderError, ProviderResponse
from .deepseek import DeepSeekProvider
from .google import GoogleProvider
from .lmstudio import LMStudioProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider

__all__ = [
    "Provider",
    "ProviderResponse",
    "ProviderError",
    "AlibabaProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "GoogleProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LMStudioProvider",
]


def get_provider(name: str, **kwargs) -> Provider:
    """
    Factory function to get a provider instance.

    Args:
        name: Provider name ('openai', 'deepseek', 'google', 'anthropic', 'alibaba', 'ollama')
        **kwargs: Provider-specific configuration

    Returns:
        Provider instance

    Raises:
        ValueError: If provider name is unknown
    """
    providers = {
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
        "alibaba": AlibabaProvider,
        "google": GoogleProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider,
    }

    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")

    return providers[name](**kwargs)
