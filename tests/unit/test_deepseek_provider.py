"""Unit tests for DeepSeek V4 request behavior."""

import asyncio
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest

from claude_mm.providers.base import ProviderError
from claude_mm.providers.deepseek import DEFAULT_DEEPSEEK_BASE_URL, DeepSeekProvider


def _install_fake_openai(monkeypatch, captured):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
    )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **params):
            captured["request"] = params
            return response

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))


def test_xhigh_maps_to_deepseek_max_thinking(monkeypatch):
    captured = {}
    _install_fake_openai(monkeypatch, captured)
    provider = DeepSeekProvider(api_key="test-key")
    monkeypatch.setattr(provider, "estimate_cost", lambda *_args: Decimal("0"))

    result = provider.complete(
        "verify this",
        model="deepseek-v4-pro",
        system_prompt="verifier",
        reasoning_effort="xhigh",
    )

    assert result.text == "ok"
    assert captured["client"]["base_url"] == DEFAULT_DEEPSEEK_BASE_URL
    assert captured["client"]["timeout"] == 300.0
    assert captured["client"]["max_retries"] == 0
    assert captured["request"]["reasoning_effort"] == "max"
    assert captured["request"]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in captured["request"]
    assert result.metadata["reasoning_effort_applied"] == "max"


@pytest.mark.parametrize("requested", ["minimal", "low", "medium", "high"])
def test_lower_efforts_map_to_deepseek_high(monkeypatch, requested):
    captured = {}
    _install_fake_openai(monkeypatch, captured)
    provider = DeepSeekProvider(api_key="test-key")
    monkeypatch.setattr(provider, "estimate_cost", lambda *_args: Decimal("0"))

    provider.complete("review", reasoning_effort=requested)

    assert captured["request"]["reasoning_effort"] == "high"
    assert "temperature" not in captured["request"]


def test_none_disables_thinking_and_keeps_temperature(monkeypatch):
    captured = {}
    _install_fake_openai(monkeypatch, captured)
    provider = DeepSeekProvider(api_key="test-key")
    monkeypatch.setattr(provider, "estimate_cost", lambda *_args: Decimal("0"))

    provider.complete("review", reasoning_effort="none", temperature=0.2)

    assert "reasoning_effort" not in captured["request"]
    assert captured["request"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["request"]["temperature"] == 0.2


def test_async_xhigh_preserves_max_effort(monkeypatch):
    captured = {}
    _install_fake_openai(monkeypatch, captured)
    provider = DeepSeekProvider(api_key="test-key")
    monkeypatch.setattr(provider, "estimate_cost", lambda *_args: Decimal("0"))

    result = asyncio.run(provider.complete_async("verify", reasoning_effort="xhigh"))

    assert result.text == "ok"
    assert captured["request"]["reasoning_effort"] == "max"


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="DEEPSEEK_API_KEY"):
        DeepSeekProvider()
