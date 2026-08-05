"""Integration tests for multi-seat orchestration through one router boundary."""

from decimal import Decimal

import pytest

from claude_mm import api
from claude_mm.models import MODEL_GROUPS
from claude_mm.providers.base import ProviderResponse


def _routed_response(route: str) -> ProviderResponse:
    return ProviderResponse(
        text=f"review from {route}",
        model="served-model",
        input_tokens=10,
        output_tokens=5,
        cost=Decimal("0"),
        metadata={
            "router_verified": True,
            "profile": "standard",
            "provider": "router-provider",
            "served_model": "served-model",
            "fallback_outcome": "served",
        },
    )


def test_mm_group_invokes_semantic_router_seats(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []

    class StubRouter:
        def complete(self, prompt, model, system_prompt=None, **kwargs):
            called.append(model)
            return _routed_response(model)

    monkeypatch.setattr(api, "get_provider", lambda provider: StubRouter())
    monkeypatch.setattr(api, "log_api_call", lambda **kwargs: None)
    result = api.review(
        "diff --git a/test",
        models=MODEL_GROUPS["mm"],
        use_cache=False,
        per_model_timeout=0,
    )
    assert set(called) == set(MODEL_GROUPS["mm"])
    assert set(result.results) == set(MODEL_GROUPS["mm"])


def test_partial_router_failure_does_not_add_ai_mm_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []

    class StubRouter:
        def complete(self, prompt, model, system_prompt=None, **kwargs):
            called.append(model)
            if model == "stage:audit":
                raise RuntimeError("router cascade exhausted")
            return _routed_response(model)

    monkeypatch.setattr(api, "get_provider", lambda provider: StubRouter())
    monkeypatch.setattr(api, "log_api_call", lambda **kwargs: None)
    result = api.review(
        "diff --git a/test",
        models=["stage:review", "stage:audit"],
        use_cache=False,
        per_model_timeout=0,
    )
    assert called == ["stage:review", "stage:audit"] or called == [
        "stage:audit",
        "stage:review",
    ]
    assert result.fallback_models == set()
    assert result.errors == {"stage:audit": "router cascade exhausted"}


def test_all_router_failures_raise_without_local_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FailedRouter:
        def complete(self, prompt, model, system_prompt=None, **kwargs):
            calls.append(model)
            raise RuntimeError("unavailable")

    monkeypatch.setattr(api, "get_provider", lambda provider: FailedRouter())
    with pytest.raises(api.AllModelsFailedError, match="llm-router health"):
        api.review(
            "diff --git a/test",
            models=["stage:review", "stage:audit"],
            use_cache=False,
            per_model_timeout=0,
        )
    assert sorted(calls) == ["stage:audit", "stage:review"]
