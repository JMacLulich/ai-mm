from __future__ import annotations

from types import SimpleNamespace

import pytest

from claude_mm.providers.lmstudio import LMStudioProvider


def test_lmstudio_accepts_only_structured_json_from_reasoning_channel(monkeypatch) -> None:
    message = SimpleNamespace(content="", reasoning_content='{"ok":true}')
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
        model="qwen/qwen3.6-35b-a3b",
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    provider = LMStudioProvider()
    monkeypatch.setattr(provider, "_client", lambda: client)

    result = provider.complete(
        prompt="Return JSON.",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "smoke",
                "schema": {"type": "object"},
            },
        },
    )

    assert result.text == '{"ok":true}'


def test_lmstudio_does_not_expose_unstructured_reasoning(monkeypatch) -> None:
    message = SimpleNamespace(content="", reasoning_content="private chain of thought")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
        model="qwen/qwen3.6-35b-a3b",
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    provider = LMStudioProvider()
    monkeypatch.setattr(provider, "_client", lambda: client)

    with pytest.raises(Exception, match="empty response content"):
        provider.complete(
            prompt="Return JSON.",
            response_format={"type": "json_schema", "json_schema": {}},
        )
