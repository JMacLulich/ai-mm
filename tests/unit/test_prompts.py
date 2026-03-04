"""Unit tests for review prompt selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm.prompts import DEFAULT_REVIEW_FOCUS, get_review_system_prompt


def test_review_focus_prompt_exists() -> None:
    """Review focus returns the rigorous PR review prompt."""
    prompt = get_review_system_prompt("review")
    assert "rigorous pull request review" in prompt.lower()
    assert "final: summary for the author" in prompt.lower()


def test_testing_focus_prompt_exists() -> None:
    """Testing focus returns the QA-oriented review prompt."""
    prompt = get_review_system_prompt("testing")
    assert "testing and qa specialist" in prompt.lower()


def test_unknown_focus_falls_back_to_default() -> None:
    """Unknown focus values fall back to default focus prompt."""
    prompt = get_review_system_prompt("not-a-real-focus")
    default_prompt = get_review_system_prompt(DEFAULT_REVIEW_FOCUS)
    assert prompt == default_prompt
