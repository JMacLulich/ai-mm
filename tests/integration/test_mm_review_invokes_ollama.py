"""Integration tests for multimode review provider invocation."""

import sys
from decimal import Decimal
from pathlib import Path

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
        models=MODEL_GROUPS["mm"],
        focus="architecture",
        use_cache=False,
    )

    output = capsys.readouterr().out

    assert "qwen2.5:14b-instruct" in called_models
    assert "Error reviewing with ollama:" in output
    assert "ollama" not in result.results
    assert len(result.results) == len(MODEL_GROUPS["mm"]) - 1
