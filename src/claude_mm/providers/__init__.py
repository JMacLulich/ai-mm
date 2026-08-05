"""LLM execution boundary.

``ai-mm`` deliberately exposes one runtime provider: the host-level Rust
``llm-router`` service. Direct vendor adapters remain implementation history and are
not selectable through this factory.
"""

from .base import Provider, ProviderError, ProviderResponse
from .router import LLMRouterProvider

__all__ = [
    "Provider",
    "ProviderResponse",
    "ProviderError",
    "LLMRouterProvider",
    "get_provider",
]


def get_provider(name: str = "llm_router", **kwargs) -> Provider:
    """Build the single approved provider client."""
    normalized = name.replace("-", "_").lower()
    if normalized not in {"llm_router", "router"}:
        raise ValueError(
            f"Direct provider '{name}' is disabled; ai-mm routes all LLM calls via llm-router"
        )
    return LLMRouterProvider(**kwargs)
