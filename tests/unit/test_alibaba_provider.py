"""Unit tests for Alibaba Cloud DashScope provider configuration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm.providers.alibaba import DEFAULT_DASHSCOPE_BASE_URL, AlibabaProvider
from claude_mm.providers.base import ProviderError


def test_alibaba_requires_api_key(monkeypatch):
    """Provider requires DASHSCOPE_API_KEY when api_key is not passed."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ALIBABA_API_KEY", raising=False)

    with pytest.raises(ProviderError, match="DASHSCOPE_API_KEY not set"):
        AlibabaProvider()


def test_alibaba_reads_api_key_from_env(monkeypatch):
    """Provider reads DashScope API key from env."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    provider = AlibabaProvider()
    assert provider.api_key == "test-key"
    assert provider.base_url == DEFAULT_DASHSCOPE_BASE_URL


def test_alibaba_reads_base_url_from_env(monkeypatch):
    """Provider reads and normalizes DashScope base URL from env."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.test/v1/")

    provider = AlibabaProvider()
    assert provider.base_url == "https://example.test/v1"
