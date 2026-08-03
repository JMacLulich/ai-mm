"""Unit tests for OpenAI-specific request options."""

import sys
from decimal import Decimal
from types import SimpleNamespace

from claude_mm.providers.openai import OpenAIProvider


def test_sol_xhigh_is_sent_as_reasoning_effort(monkeypatch):
    captured = {}

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
    )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, **params):
            captured.update(params)
            return response

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    provider = OpenAIProvider(api_key="test-key")
    monkeypatch.setattr(provider, "estimate_cost", lambda *_args: Decimal("0"))
    result = provider.complete(
        "review this",
        model="gpt-5.6-sol",
        system_prompt="reviewer",
        reasoning_effort="xhigh",
    )

    assert result.text == "ok"
    assert captured["model"] == "gpt-5.6-sol"
    assert captured["reasoning_effort"] == "xhigh"
    assert "temperature" not in captured
