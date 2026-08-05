"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Tests must never be redirected to a non-local router endpoint.
    os.environ.pop("LLM_ROUTER_API_KEY", None)
    os.environ["LLM_ROUTER_BASE_URL"] = "http://127.0.0.1:4000/v1"


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Provide a mock router connection for tests."""
    monkeypatch.setenv("LLM_ROUTER_BASE_URL", "http://127.0.0.1:4000/v1")
