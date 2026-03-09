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


def test_security_focus_prompt_includes_security_review_categories() -> None:
    """Security focus includes key application security review areas."""
    prompt = get_review_system_prompt("security").lower()
    assert "authentication and authorization" in prompt
    assert "tenant isolation" in prompt
    assert "row-level security (rls)" in prompt
    assert "secrets exposure" in prompt
    assert "sql/command/template injection" in prompt
    assert "xss/csrf/ssrf" in prompt
    assert "security test to add" in prompt


def test_architecture_focus_prompt_includes_design_principles() -> None:
    """Architecture focus includes the expanded design review context."""
    prompt = get_review_system_prompt("architecture").lower()
    assert "modularity" in prompt
    assert "loose coupling" in prompt
    assert "high cohesion" in prompt
    assert "abstraction" in prompt
    assert "ci/cd" in prompt
    assert "separation of concerns" in prompt
    assert "testability" in prompt


def test_unknown_focus_falls_back_to_default() -> None:
    """Unknown focus values fall back to default focus prompt."""
    prompt = get_review_system_prompt("not-a-real-focus")
    default_prompt = get_review_system_prompt(DEFAULT_REVIEW_FOCUS)
    assert prompt == default_prompt
