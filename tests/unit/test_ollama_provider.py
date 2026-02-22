"""Unit tests for Ollama provider configuration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm.providers.base import ProviderError
from claude_mm.providers.ollama import OllamaProvider


def test_ollama_requires_base_url(monkeypatch):
    """Provider requires OLLAMA_BASE_URL when base_url is not passed."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    with pytest.raises(ProviderError, match="OLLAMA_BASE_URL not set"):
        OllamaProvider()


def test_ollama_reads_base_url_from_env(monkeypatch):
    """Provider reads and normalizes base URL from env."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")

    provider = OllamaProvider()
    assert provider.base_url == "http://localhost:11434"
