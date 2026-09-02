"""Unit tests for routed API defaults and request hints."""

import asyncio
from decimal import Decimal

import pytest

from claude_mm import api
from claude_mm.providers.base import ProviderResponse


def _response(text: str, route: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        model="served-model",
        input_tokens=1,
        output_tokens=1,
        cost=Decimal("0"),
        metadata={
            "router_verified": True,
            "profile": "standard",
            "provider": "router-provider",
            "served_model": "served-model",
            "fallback_outcome": "served",
            "requested_route": route,
        },
    )


def test_review_defaults_to_review_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    monkeypatch.setattr(
        api, "load_config", lambda: {"default_models": {"review": "stage:review"}}
    )

    def stub_review_single(prompt, model, system_prompt, use_cache, cache_ttl):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(api, "_review_single", stub_review_single)
    assert api.review("diff --git a/test") == "ok"
    assert captured["model"] == "stage:review"


def test_review_async_defaults_to_review_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    monkeypatch.setattr(
        api, "load_config", lambda: {"default_models": {"review": "stage:review"}}
    )

    async def stub_review_single_async(prompt, model, system_prompt, use_cache, cache_ttl):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(api, "_review_single_async", stub_review_single_async)
    assert asyncio.run(api.review_async("diff --git a/test")) == "ok"
    assert captured["model"] == "stage:review"


def test_review_forwards_provider_neutral_effort_to_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class StubProvider:
        def complete(self, prompt, model, system_prompt=None, **kwargs):
            captured.update(model=model, kwargs=kwargs)
            return _response("verified", model)

    monkeypatch.setattr(api, "get_provider", lambda provider: StubProvider())
    monkeypatch.setattr(api, "log_api_call", lambda **kwargs: None)
    result = api.review(
        "diff --git a/test",
        model="deepseek",
        focus="verification",
        use_cache=False,
        per_model_timeout=0,
        reasoning_effort="max",
    )
    assert result.text == "verified"
    assert captured == {
        "model": "stage:audit",
        "kwargs": {"reasoning_effort": "max"},
    }


def test_review_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        api.review("diff --git a/test", model="deepseek", reasoning_effort="ultra")


def test_review_rejects_direct_api_model_selector() -> None:
    with pytest.raises(ValueError, match="Unknown LLM route"):
        api.review(
            "diff --git a/test",
            model="vendor/model-id",
            use_cache=False,
            per_model_timeout=0,
        )


def test_review_rejects_oversized_prompt_before_provider_call() -> None:
    prompt = "x" * (api.MAX_PROMPT_CHARS + 1)

    with pytest.raises(ValueError, match="prompt is too large.*Review the diff"):
        api.review(prompt, model="stage:review", per_model_timeout=0)
