"""Architecture tests that keep routing decisions out of feature code."""

from pathlib import Path

import pytest

from claude_mm.config_tui import PROVIDERS
from claude_mm.providers import get_provider


@pytest.mark.parametrize(
    "provider",
    ["openai", "deepseek", "anthropic", "alibaba", "ollama", "lmstudio", "google"],
)
def test_direct_provider_factory_access_is_disabled(provider: str) -> None:
    with pytest.raises(ValueError, match="routes all LLM calls via llm-router"):
        get_provider(provider)


def test_runtime_hot_paths_have_no_vendor_client_or_api_model_selection() -> None:
    root = Path(__file__).parents[2]
    paths = [
        root / "bin" / "ai",
        root / "src" / "claude_mm" / "api.py",
        root / "src" / "claude_mm" / "planning.py",
        root / "src" / "claude_mm" / "rig_assist.py",
        root / "src" / "claude_mm" / "config.py",
        root / "src" / "claude_mm" / "config_tui.py",
    ]
    forbidden = [
        "OpenAIProvider",
        "DeepSeekProvider",
        "AnthropicProvider",
        "AlibabaProvider",
        "OllamaProvider",
        "LMStudioProvider",
    ]
    for path in paths:
        contents = path.read_text()
        for token in forbidden:
            assert token not in contents, f"{token} leaked into {path}"
        for provider in ("OPENAI", "DEEPSEEK", "ANTHROPIC"):
            assert f"{provider}_API_KEY" not in contents
        for local_runtime in ("OLLAMA", "LMSTUDIO"):
            assert f"{local_runtime}_BASE_URL" not in contents


def test_config_surface_contains_only_router_connection() -> None:
    assert PROVIDERS == [
        (
            "llm-router",
            "LLM_ROUTER_BASE_URL",
            "Rust routing service endpoint",
            "http(s)://...",
        )
    ]


def test_dependency_metadata_has_no_direct_vendor_sdk() -> None:
    root = Path(__file__).parents[2]
    metadata = (root / "pyproject.toml").read_text().lower()
    assert '"anthropic' not in metadata
