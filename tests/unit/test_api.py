"""Unit tests for API default model behavior."""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm import api
from claude_mm.providers.base import ProviderResponse


def test_review_defaults_to_gpt_5_4(monkeypatch):
    """Review uses GPT-5.4 when no model is provided."""
    captured = {}

    monkeypatch.setattr(api, "load_config", lambda: {"default_models": {"review": "gpt-5.4"}})

    def stub_review_single(prompt, model, system_prompt, use_cache, cache_ttl):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(api, "_review_single", stub_review_single)

    result = api.review("diff --git a/test")

    assert result == "ok"
    assert captured["model"] == "gpt-5.4"


def test_review_async_defaults_to_gpt_5_4(monkeypatch):
    """Async review uses GPT-5.4 when no model is provided."""
    captured = {}

    monkeypatch.setattr(api, "load_config", lambda: {"default_models": {"review": "gpt-5.4"}})

    async def stub_review_single_async(prompt, model, system_prompt, use_cache, cache_ttl):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(api, "_review_single_async", stub_review_single_async)

    result = asyncio.run(api.review_async("diff --git a/test"))

    assert result == "ok"
    assert captured["model"] == "gpt-5.4"


def test_review_forwards_reasoning_effort_to_openai_model(monkeypatch):
    """OpenAI reasoning effort reaches the provider without an API call."""
    captured = {}

    class StubProvider:
        def complete(self, prompt, model, system_prompt=None, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs
            return ProviderResponse(
                text="ok",
                model=model,
                input_tokens=1,
                output_tokens=1,
                cost=Decimal("0"),
            )

    monkeypatch.setattr(api, "get_provider", lambda _provider: StubProvider())
    monkeypatch.setattr(api, "log_api_call", lambda **_kwargs: None)

    result = api.review(
        "diff --git a/test",
        model="sol-5.6",
        use_cache=False,
        per_model_timeout=0,
        reasoning_effort="xhigh",
    )

    assert result.text == "ok"
    assert captured == {"model": "gpt-5.6-sol", "kwargs": {"reasoning_effort": "xhigh"}}


def test_review_rejects_unknown_reasoning_effort():
    """The public API rejects unsupported effort names before provider work."""
    try:
        api.review("diff --git a/test", model="sol-5.6", reasoning_effort="ultra")
    except ValueError as exc:
        assert "reasoning_effort" in str(exc)
    else:
        raise AssertionError("expected invalid reasoning effort to fail")


def test_review_forwards_reasoning_effort_to_deepseek(monkeypatch):
    captured = {}

    class StubProvider:
        def complete(self, prompt, model, system_prompt=None, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs
            return ProviderResponse(
                text="verified",
                model=model,
                input_tokens=1,
                output_tokens=1,
                cost=Decimal("0"),
            )

    monkeypatch.setattr(api, "get_provider", lambda _provider: StubProvider())
    monkeypatch.setattr(api, "log_api_call", lambda **_kwargs: None)

    result = api.review(
        "diff --git a/test",
        model="deepseek-pro",
        focus="verification",
        use_cache=False,
        per_model_timeout=0,
        reasoning_effort="xhigh",
    )

    assert result.text == "verified"
    assert captured == {
        "model": "deepseek-v4-pro",
        "kwargs": {"reasoning_effort": "xhigh"},
    }
