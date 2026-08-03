"""Pytest configuration and shared fixtures."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Prevent tests from accidentally using real API keys
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Provide mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
