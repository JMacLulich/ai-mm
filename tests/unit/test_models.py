"""Unit tests for models module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm.models import (
    CLAUDE_ALIASES,
    CLAUDE_MODELS,
    GEMINI_ALIASES,
    GEMINI_MODELS,
    OPENAI_ALIASES,
    OPENAI_MODELS,
    get_model_characteristics,
    get_model_display_name,
    get_provider_for_model,
    list_all_aliases,
    list_all_models,
    normalize_model_name,
)


class TestModelRegistries:
    """Test model registries are properly defined."""

    def test_openai_models_exist(self):
        """OpenAI models registry is not empty."""
        assert len(OPENAI_MODELS) > 0
        assert "gpt-5.2-chat-latest" in OPENAI_MODELS
        assert "gpt-5.2" in OPENAI_MODELS
        assert "gpt-5.2-pro" in OPENAI_MODELS

    def test_openai_aliases_exist(self):
        """OpenAI aliases are properly defined."""
        assert len(OPENAI_ALIASES) > 0
        assert "gpt" in OPENAI_ALIASES
        assert "gpt-5.2-instant" in OPENAI_ALIASES  # Backward compatibility

    def test_gemini_models_exist(self):
        """Gemini models registry is not empty."""
        assert len(GEMINI_MODELS) > 0
        assert "gemini-3-flash-preview" in GEMINI_MODELS

    def test_gemini_aliases_exist(self):
        """Gemini aliases are properly defined."""
        assert len(GEMINI_ALIASES) > 0
        assert "gemini" in GEMINI_ALIASES

    def test_claude_models_exist(self):
        """Claude models registry is not empty."""
        assert len(CLAUDE_MODELS) > 0
        assert "claude-sonnet-4-5-20250929" in CLAUDE_MODELS

    def test_claude_aliases_exist(self):
        """Claude aliases are properly defined."""
        assert len(CLAUDE_ALIASES) > 0
        assert "claude" in CLAUDE_ALIASES


class TestGetProviderForModel:
    """Test provider detection from model names."""

    def test_openai_models(self):
        """OpenAI models resolve to 'openai' provider."""
        assert get_provider_for_model("gpt-5.2") == "openai"
        assert get_provider_for_model("gpt-5.2-chat-latest") == "openai"
        assert get_provider_for_model("gpt-5.2-pro") == "openai"

    def test_openai_aliases(self):
        """OpenAI aliases resolve to 'openai' provider."""
        assert get_provider_for_model("gpt") == "openai"
        assert get_provider_for_model("gpt-5.2-instant") == "openai"

    def test_gemini_models(self):
        """Gemini models resolve to 'google' provider."""
        assert get_provider_for_model("gemini-3-flash-preview") == "google"

    def test_gemini_aliases(self):
        """Gemini aliases resolve to 'google' provider."""
        assert get_provider_for_model("gemini") == "google"

    def test_claude_models(self):
        """Claude models resolve to 'anthropic' provider."""
        assert get_provider_for_model("claude-sonnet-4-5-20250929") == "anthropic"

    def test_claude_aliases(self):
        """Claude aliases resolve to 'anthropic' provider."""
        assert get_provider_for_model("claude") == "anthropic"

    def test_unknown_model(self):
        """Unknown models return None."""
        assert get_provider_for_model("unknown-model") is None
        assert get_provider_for_model("gpt-99") is None


class TestNormalizeModelName:
    """Test model name normalization."""

    def test_openai_direct_models(self):
        """OpenAI direct model names normalize correctly."""
        provider, model = normalize_model_name("gpt-5.2")
        assert provider == "openai"
        assert model == "gpt-5.2"

        provider, model = normalize_model_name("gpt-5.2-chat-latest")
        assert provider == "openai"
        assert model == "gpt-5.2-chat-latest"

        provider, model = normalize_model_name("gpt-5.2-pro")
        assert provider == "openai"
        assert model == "gpt-5.2-pro"

    def test_openai_aliases(self):
        """OpenAI aliases resolve to API names."""
        provider, model = normalize_model_name("gpt")
        assert provider == "openai"
        assert model == "gpt-5.2"  # Default GPT model

        # CRITICAL: Backward compatibility for old name
        provider, model = normalize_model_name("gpt-5.2-instant")
        assert provider == "openai"
        assert model == "gpt-5.2-chat-latest"

        provider, model = normalize_model_name("gpt-instant")
        assert provider == "openai"
        assert model == "gpt-5.2-chat-latest"

    def test_gemini_direct_models(self):
        """Gemini direct model names normalize correctly."""
        provider, model = normalize_model_name("gemini-3-flash-preview")
        assert provider == "google"
        assert model == "gemini-3-flash-preview"

    def test_gemini_aliases(self):
        """Gemini aliases resolve to API names."""
        provider, model = normalize_model_name("gemini")
        assert provider == "google"
        assert model == "gemini-3-pro-preview"

        provider, model = normalize_model_name("gemini-flash")
        assert provider == "google"
        assert model == "gemini-3-flash-preview"

    def test_claude_direct_models(self):
        """Claude direct model names normalize correctly."""
        provider, model = normalize_model_name("claude-sonnet-4-5-20250929")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-5-20250929"

    def test_claude_aliases(self):
        """Claude aliases resolve to API names."""
        provider, model = normalize_model_name("claude")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-5-20250929"

    def test_unknown_model_raises(self):
        """Unknown models raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            normalize_model_name("unknown-model")

        with pytest.raises(ValueError, match="Unknown model"):
            normalize_model_name("gpt-99")


class TestGetModelDisplayName:
    """Test human-readable display names."""

    def test_openai_display_names(self):
        """OpenAI models have proper display names."""
        assert get_model_display_name("gpt-5.2-chat-latest") == "GPT-5.2 Instant"
        assert get_model_display_name("gpt-5.2") == "GPT-5.2 Thinking"
        assert get_model_display_name("gpt-5.2-pro") == "GPT-5.2 Pro"

    def test_gemini_display_names(self):
        """Gemini models have proper display names."""
        assert get_model_display_name("gemini-3-flash-preview") == "Gemini 3 Flash"

    def test_claude_display_names(self):
        """Claude models have proper display names."""
        assert get_model_display_name("claude-sonnet-4-5-20250929") == "Claude Sonnet 4.5"

    def test_unknown_model_returns_original(self):
        """Unknown models return the original name."""
        assert get_model_display_name("unknown-model") == "unknown-model"


class TestGetModelCharacteristics:
    """Test model characteristics metadata."""

    def test_openai_instant_characteristics(self):
        """GPT-5.2 Instant has correct characteristics."""
        chars = get_model_characteristics("gpt-5.2-chat-latest")
        assert chars["speed"] == "fast"
        assert chars["cost_tier"] == "low"
        assert chars["context_window"] == 128000
        assert "description" in chars

    def test_openai_thinking_characteristics(self):
        """GPT-5.2 Thinking has correct characteristics."""
        chars = get_model_characteristics("gpt-5.2")
        assert chars["speed"] == "medium"
        assert chars["cost_tier"] == "medium"
        assert chars["context_window"] == 128000

    def test_openai_pro_characteristics(self):
        """GPT-5.2 Pro has correct characteristics."""
        chars = get_model_characteristics("gpt-5.2-pro")
        assert chars["speed"] == "slow"
        assert chars["cost_tier"] == "high"
        assert chars["context_window"] == 128000

    def test_gemini_characteristics(self):
        """Gemini has correct characteristics."""
        chars = get_model_characteristics("gemini-3-flash-preview")
        assert chars["speed"] == "fast"
        assert chars["cost_tier"] == "low"
        assert chars["context_window"] == 1000000

    def test_claude_characteristics(self):
        """Claude has correct characteristics."""
        chars = get_model_characteristics("claude-sonnet-4-5-20250929")
        assert chars["speed"] == "fast"
        assert chars["cost_tier"] == "medium"
        assert chars["context_window"] == 200000

    def test_unknown_model_has_defaults(self):
        """Unknown models get default characteristics."""
        chars = get_model_characteristics("unknown-model")
        assert chars["speed"] == "unknown"
        assert chars["cost_tier"] == "unknown"
        assert chars["context_window"] == 8192
        assert chars["description"] == "Unknown model"


class TestListFunctions:
    """Test listing functions."""

    def test_list_all_models(self):
        """list_all_models returns all models by provider."""
        models = list_all_models()
        assert "openai" in models
        assert "google" in models
        assert "anthropic" in models
        assert isinstance(models["openai"], list)
        assert len(models["openai"]) > 0

    def test_list_all_aliases(self):
        """list_all_aliases returns all aliases."""
        aliases = list_all_aliases()
        assert "gpt" in aliases
        assert "gpt-5.2-instant" in aliases  # Backward compatibility
        assert "gemini" in aliases
        assert "claude" in aliases
        assert isinstance(aliases, dict)


class TestBackwardCompatibility:
    """Test backward compatibility with old model names."""

    def test_gpt_5_2_instant_alias(self):
        """Old 'gpt-5.2-instant' name maps to correct API model."""
        # This is the critical test for the issue reported
        provider, model = normalize_model_name("gpt-5.2-instant")
        assert provider == "openai"
        assert model == "gpt-5.2-chat-latest"

        # Verify it's recognized as OpenAI
        assert get_provider_for_model("gpt-5.2-instant") == "openai"

    def test_all_aliases_resolve(self):
        """All defined aliases successfully resolve."""
        all_aliases = list_all_aliases()
        for alias, _ in all_aliases.items():
            # Should not raise ValueError
            provider, model = normalize_model_name(alias)
            assert provider is not None
            assert model is not None
