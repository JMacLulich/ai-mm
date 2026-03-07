"""Integration tests for multimode review provider invocation."""

import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm import api
from claude_mm.models import MODEL_GROUPS
from claude_mm.providers.base import ProviderResponse


def test_mm_review_invokes_ollama_and_reports_error(monkeypatch, capsys):
    """Multimode reviews invoke local model and print Ollama errors when unavailable."""
    called_models = []

    class StubProvider:
        def complete(self, prompt, model, system_prompt=None):
            called_models.append(model)
            if model == "qwen2.5:14b-instruct":
                raise Exception("Ollama server not running. Start with: ollama serve")

            return ProviderResponse(
                text=f"ok from {model}",
                model=model,
                input_tokens=1,
                output_tokens=1,
                cost=Decimal("0"),
            )

    monkeypatch.setattr(api, "get_provider", lambda _provider_name: StubProvider())
    monkeypatch.setattr(api, "log_api_call", lambda **_kwargs: None)

    result = api.review(
        prompt="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py",
        models=["gpt-5.4", "gemini", "claude-opus-4-6", "ollama"],
        focus="architecture",
        use_cache=False,
    )

    output = capsys.readouterr().out

    assert "qwen2.5:14b-instruct" in called_models
    assert "Error reviewing with ollama:" in output
    assert "ollama" not in result.results
    assert len(result.results) == 3


def test_mm_review_raises_when_all_models_fail(monkeypatch):
    """Multimode reviews fail fast when every configured model errors out."""

    class AlwaysFailProvider:
        def complete(self, prompt, model, system_prompt=None):
            raise Exception(f"{model} unavailable")

    monkeypatch.setattr(api, "get_provider", lambda _provider_name: AlwaysFailProvider())
    monkeypatch.setattr(api, "log_api_call", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="All review models failed"):
        api.review(
            prompt="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py",
            models=MODEL_GROUPS["mm"],
            focus="architecture",
            use_cache=False,
        )


def test_mm_review_falls_back_to_lmstudio_after_two_overload_errors(monkeypatch, capsys):
    """When 2+ models fail with 503/529 overloads, fallback to LM Studio runs."""

    class StubProvider:
        def complete(self, prompt, model, system_prompt=None):
            if model in {"gpt-5.4", "gemini-3.1-pro-preview"}:
                raise Exception("503 Service Unavailable")

            if model == "lmstudio" or model == "qwen3.5:27b":
                return ProviderResponse(
                    text="ok from lmstudio",
                    model="qwen3.5:27b",
                    input_tokens=1,
                    output_tokens=1,
                    cost=Decimal("0"),
                )

            return ProviderResponse(
                text=f"ok from {model}",
                model=model,
                input_tokens=1,
                output_tokens=1,
                cost=Decimal("0"),
            )

    monkeypatch.setattr(api, "get_provider", lambda _provider_name: StubProvider())
    monkeypatch.setattr(api, "log_api_call", lambda **_kwargs: None)

    result = api.review(
        prompt="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py",
        models=["gpt-5.4", "gemini", "claude-opus-4-6"],
        focus="architecture",
        use_cache=False,
    )

    output = capsys.readouterr().out

    assert "Detected repeated provider overloads (503/529)" in output
    assert "lmstudio" in result.results


def test_mm_review_uses_local_fallback_when_external_models_fail(monkeypatch, capsys):
    """When external providers fail and no local succeeds, local fallback is attempted."""

    class StubProvider:
        def complete(self, prompt, model, system_prompt=None):
            if model in {
                "gpt-5.4",
                "gemini-3.1-pro-preview",
                "claude-opus-4-6",
                "qwen2.5:14b-instruct",
            }:
                raise Exception(f"{model} unavailable")

            if model == "qwen3.5:27b":
                return ProviderResponse(
                    text="ok from lmstudio",
                    model="qwen3.5:27b",
                    input_tokens=1,
                    output_tokens=1,
                    cost=Decimal("0"),
                )

            return ProviderResponse(
                text=f"ok from {model}",
                model=model,
                input_tokens=1,
                output_tokens=1,
                cost=Decimal("0"),
            )

    monkeypatch.setattr(api, "get_provider", lambda _provider_name: StubProvider())
    monkeypatch.setattr(api, "log_api_call", lambda **_kwargs: None)

    result = api.review(
        prompt="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py",
        models=["gpt-5.4", "gemini", "claude-opus-4-6", "ollama"],
        focus="architecture",
        use_cache=False,
    )

    output = capsys.readouterr().out

    assert "External providers failed and no local review succeeded yet" in output
    assert "lmstudio" in result.results


def test_mm_review_aggregates_results_when_one_model_times_out(monkeypatch):
    """Timed-out model should not block successful model results."""

    class StubProvider:
        def complete(self, prompt, model, system_prompt=None):
            if model == "gpt-5.4":
                time.sleep(0.2)
                return ProviderResponse(
                    text="late gpt result",
                    model=model,
                    input_tokens=1,
                    output_tokens=1,
                    cost=Decimal("0"),
                )

            return ProviderResponse(
                text=f"ok from {model}",
                model=model,
                input_tokens=1,
                output_tokens=1,
                cost=Decimal("0"),
            )

    monkeypatch.setattr(api, "get_provider", lambda _provider_name: StubProvider())
    monkeypatch.setattr(api, "log_api_call", lambda **_kwargs: None)

    result = api.review(
        prompt="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py",
        models=["gpt-5.4", "gemini"],
        focus="architecture",
        use_cache=False,
        per_model_timeout=0.05,
    )

    assert "gemini" in result.results
    assert "gpt-5.4" not in result.results
    assert result.errors["gpt-5.4"].startswith("timed out after")
