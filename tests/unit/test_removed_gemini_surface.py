"""Regression tests for the complete removal of the Gemini provider surface."""

import importlib
from pathlib import Path

import pytest

from claude_mm.config_tui import PROVIDERS
from claude_mm.cost_tracker import PRICING
from claude_mm.models import (
    get_provider_for_model,
    list_all_aliases,
    list_all_models,
    normalize_model_name,
)
from claude_mm.pricing import DEFAULT_PRICING
from claude_mm.providers import get_provider


@pytest.mark.parametrize(
    "model",
    [
        "gemini",
        "gemini-pro",
        "gemini-flash",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
    ],
)
def test_removed_gemini_models_are_rejected(model):
    assert get_provider_for_model(model) is None
    with pytest.raises(ValueError, match="Unknown model"):
        normalize_model_name(model)


def test_removed_google_provider_is_not_importable_or_registered():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("claude_mm.providers.google")

    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("google")


def test_removed_provider_is_absent_from_config_and_pricing():
    provider_names = {provider[0] for provider in PROVIDERS}
    config_names = {provider[1] for provider in PROVIDERS}

    assert "google" not in provider_names
    assert "GOOGLE_AI_API_KEY" not in config_names
    assert "google" not in DEFAULT_PRICING
    assert "google" not in list_all_models()
    assert all("gemini" not in model for model in PRICING)
    assert all("gemini" not in alias for alias in list_all_aliases())


def test_cli_install_and_dependency_metadata_have_no_gemini_surface():
    root = Path(__file__).parents[2]
    exposed_files = [root / "bin" / "ai", root / "commands" / "install" / "run"]

    for path in exposed_files:
        contents = path.read_text().lower()
        assert "gemini" not in contents
        assert "google_ai_api_key" not in contents

    assert "google-genai" not in (root / "pyproject.toml").read_text().lower()
