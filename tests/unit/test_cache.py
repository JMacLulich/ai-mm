"""Unit tests for cache module."""

import json

# Import after path setup
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm.cache import (
    cache_response,
    clear_cache,
    get_cache_key,
    get_cache_stats,
    get_cached_response,
)


@pytest.fixture
def temp_cache_dir(monkeypatch, tmp_path):
    """Use temporary directory for cache during tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    return cache_dir


def test_get_cache_key():
    """Test cache key generation is deterministic."""
    key1 = get_cache_key("model1", "system", "prompt")
    key2 = get_cache_key("model1", "system", "prompt")
    key3 = get_cache_key("model2", "system", "prompt")

    assert key1 == key2  # Same inputs = same key
    assert key1 != key3  # Different model = different key
    assert len(key1) == 64  # SHA256 hex digest


def test_cache_response_and_get(temp_cache_dir):
    """Test caching and retrieving responses."""
    model = "test-model"
    system_prompt = "You are a test assistant"
    prompt = "Test prompt"
    response = "Test response"

    # Cache the response
    cache_response(model, prompt, response, system_prompt)

    # Retrieve it
    cached = get_cached_response(model, prompt, system_prompt)
    assert cached == response


def test_get_cached_response_miss(temp_cache_dir):
    """Test cache miss returns None."""
    cached = get_cached_response("model", "nonexistent prompt", "system")
    assert cached is None


def test_cache_expiry(temp_cache_dir, monkeypatch):
    """Test cache expiry after TTL."""
    model = "test-model"
    system_prompt = "system"
    prompt = "prompt"
    response = "response"

    # Cache the response
    cache_response(model, prompt, response, system_prompt)

    # Verify it's there
    assert get_cached_response(model, prompt, system_prompt) == response

    # Simulate time passing by modifying cache file timestamp
    cache_key = get_cache_key(model, prompt, system_prompt)
    cache_file = Path.home() / ".config" / "ai-mm" / "cache" / f"{cache_key}.json"

    # Set timestamp to 25 hours ago (past default 24hr TTL)
    old_time = (datetime.now() - timedelta(hours=25)).isoformat()
    with open(cache_file, "r") as f:
        data = json.load(f)
    data["timestamp"] = old_time
    with open(cache_file, "w") as f:
        json.dump(data, f)

    # Should now be expired
    cached = get_cached_response(model, prompt, system_prompt, ttl_hours=24)
    assert cached is None


def test_clear_cache(temp_cache_dir):
    """Test clearing cache."""
    # Add some cached responses
    cache_response("model1", "prompt1", "response1", "system")
    cache_response("model2", "prompt2", "response2", "system")

    # Verify they exist
    assert get_cached_response("model1", "prompt1", "system") is not None
    assert get_cached_response("model2", "prompt2", "system") is not None

    # Clear cache
    cleared = clear_cache()
    assert cleared >= 2

    # Verify they're gone
    assert get_cached_response("model1", "system", "prompt1") is None
    assert get_cached_response("model2", "system", "prompt2") is None


def test_get_cache_stats(temp_cache_dir):
    """Test cache statistics."""
    # Empty cache
    stats = get_cache_stats()
    assert stats["total_files"] == 0
    assert stats["total_size_mb"] == 0

    # Add some entries
    cache_response("model1", "prompt1", "response1", "system")
    cache_response("model2", "prompt2", "response2", "system")

    stats = get_cache_stats()
    assert stats["total_files"] == 2
    assert stats["total_size_mb"] >= 0  # Size may be 0.0 for small test files
    assert stats["oldest"] is not None
