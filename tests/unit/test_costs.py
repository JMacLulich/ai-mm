"""Unit tests for costs module."""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm.costs import (
    estimate_cost,
    estimate_cost_from_text,
    estimate_tokens,
    format_cost_warning,
    should_warn_about_cost,
)


def test_estimate_tokens():
    """Test token estimation."""
    # Basic text
    text = "Hello, world!"
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)

    # Longer text should have more tokens
    long_text = "Hello, world! " * 100
    long_tokens = estimate_tokens(long_text)
    assert long_tokens > tokens


def test_estimate_cost():
    """Test cost estimation for different models."""
    input_tokens = 1000
    output_tokens = 500

    # GPT-5.2-instant (cheap)
    cost_instant = estimate_cost("gpt-5.2-instant", input_tokens, output_tokens)
    assert cost_instant > 0

    # GPT-5.2 (more expensive)
    cost_standard = estimate_cost("gpt-5.2", input_tokens, output_tokens)
    assert cost_standard > cost_instant

    # Gemini (cheapest)
    cost_gemini = estimate_cost("gemini-3-flash-preview", input_tokens, output_tokens)
    assert cost_gemini < cost_instant


def test_estimate_cost_from_text():
    """Test cost estimation from text."""
    text = "This is a test prompt for cost estimation."
    model = "gpt-5.2-instant"

    cost = estimate_cost_from_text(text, model)
    assert cost > 0
    assert isinstance(cost, float)


def test_format_cost_warning():
    """Test cost warning formatting."""
    warning = format_cost_warning(0.05, "test operation")
    assert "$0.05" in warning
    assert "test operation" in warning


def test_should_warn_about_cost():
    """Test cost warning threshold."""
    # Small cost - no warning
    assert not should_warn_about_cost(0.01, threshold=0.10)

    # Large cost - warning
    assert should_warn_about_cost(0.15, threshold=0.10)

    # Exactly at threshold - warning
    assert should_warn_about_cost(0.10, threshold=0.10)
