"""Unit tests for pricing metadata precedence."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm import pricing


def test_exact_user_pricing_preserves_embedded_metadata(tmp_path, monkeypatch):
    """Exact user overrides keep embedded metadata like estimated pricing."""
    pricing_file = tmp_path / "pricing.yaml"
    pricing_file.write_text(
        yaml.safe_dump(
            {
                "openai": {"gpt-5.4": {"input": 2.0, "output": 16.0}},
                "google": {},
                "anthropic": {},
                "_metadata": {"last_updated": "2026-01-01T00:00:00", "version": "1.0.0"},
            }
        )
    )
    monkeypatch.setattr(pricing, "get_pricing_file", lambda: pricing_file)

    result = pricing.get_model_pricing("openai", "gpt-5.4")

    assert result == {"input": 2.0, "output": 16.0, "estimated": True}


def test_embedded_exact_match_beats_stale_provider_fallback(tmp_path, monkeypatch):
    """New exact embedded models keep their metadata when user pricing file is stale."""
    pricing_file = tmp_path / "pricing.yaml"
    pricing_file.write_text(
        yaml.safe_dump(
            {
                "openai": {"gpt-5.2": {"input": 1.5, "output": 12.0}},
                "google": {},
                "anthropic": {},
                "_metadata": {"last_updated": "2026-01-01T00:00:00", "version": "1.0.0"},
            }
        )
    )
    monkeypatch.setattr(pricing, "get_pricing_file", lambda: pricing_file)

    result = pricing.get_model_pricing("openai", "gpt-5.4")

    assert result == {"input": 1.75, "output": 14.0, "estimated": True}
