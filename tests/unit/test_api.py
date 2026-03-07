"""Unit tests for API default model behavior."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm import api


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
