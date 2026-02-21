#!/usr/bin/env python3
"""
Response caching for AI API calls.

Provides disk-based caching with TTL support and atomic writes to prevent race conditions.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path


def get_cache_dir() -> Path:
    """Get the cache directory path."""
    cache_dir = Path.home() / ".config" / "ai-mm" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(model: str, prompt: str, system_prompt: str = None) -> str:
    """Generate a cache key from model and prompts."""
    content = f"{model}:{system_prompt or ''}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached_response(
    model: str, prompt: str, system_prompt: str = None, ttl_hours: int = 24
) -> str:
    """
    Get cached response if available and not expired.

    Args:
        model: Model name
        prompt: User prompt
        system_prompt: System prompt (optional)
        ttl_hours: Time-to-live in hours (default: 24)

    Returns:
        Cached response text or None if not found/expired
    """
    cache_dir = get_cache_dir()
    cache_key = get_cache_key(model, prompt, system_prompt)
    cache_file = cache_dir / f"{cache_key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file) as f:
            cache_data = json.load(f)

        # Check if expired
        cached_at = datetime.fromisoformat(cache_data["timestamp"])
        if datetime.now() - cached_at > timedelta(hours=ttl_hours):
            # Expired, remove cache file
            cache_file.unlink()
            return None

        return cache_data["response"]
    except Exception:
        # If cache read fails, ignore and return None
        return None


def cache_response(model: str, prompt: str, response: str, system_prompt: str = None) -> None:
    """
    Cache an API response using atomic write to prevent race conditions.

    Args:
        model: Model name
        prompt: User prompt
        response: API response text
        system_prompt: System prompt (optional)
    """
    cache_dir = get_cache_dir()
    cache_key = get_cache_key(model, prompt, system_prompt)
    cache_file = cache_dir / f"{cache_key}.json"

    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "response": response,
    }

    try:
        # Use atomic write: write to temp file, then rename
        # This prevents corruption if multiple processes write simultaneously
        with tempfile.NamedTemporaryFile(
            mode="w", dir=cache_dir, delete=False, suffix=".tmp"
        ) as tmp_file:
            json.dump(cache_data, tmp_file)
            tmp_path = tmp_file.name

        # Atomic rename (replaces existing file if present)
        os.replace(tmp_path, cache_file)
    except Exception as e:
        # Don't fail the operation if caching fails
        import sys

        print(f"Warning: Failed to cache response: {e}", file=sys.stderr)
        # Clean up temp file if it exists
        try:
            if "tmp_path" in locals():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def clear_cache(older_than_hours: int = None) -> int:
    """
    Clear cached responses.

    Args:
        older_than_hours: Only clear cache older than N hours (None = all)

    Returns:
        Number of cache files removed
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    removed = 0
    cutoff = None
    if older_than_hours is not None:
        cutoff = datetime.now() - timedelta(hours=older_than_hours)

    for cache_file in cache_dir.glob("*.json"):
        try:
            if cutoff:
                # Check file timestamp
                with open(cache_file) as f:
                    cache_data = json.load(f)
                cached_at = datetime.fromisoformat(cache_data["timestamp"])
                if cached_at > cutoff:
                    continue

            cache_file.unlink()
            removed += 1
        except Exception:
            # Skip files we can't process
            continue

    return removed


def get_cache_stats() -> dict:
    """
    Get cache statistics.

    Returns:
        Dictionary with cache stats
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return {
            "total_files": 0,
            "total_size_mb": 0,
            "oldest": None,
            "newest": None,
        }

    total_files = 0
    total_size = 0
    oldest = None
    newest = None

    for cache_file in cache_dir.glob("*.json"):
        try:
            total_files += 1
            total_size += cache_file.stat().st_size

            with open(cache_file) as f:
                cache_data = json.load(f)
            cached_at = datetime.fromisoformat(cache_data["timestamp"])

            if oldest is None or cached_at < oldest:
                oldest = cached_at
            if newest is None or cached_at > newest:
                newest = cached_at
        except Exception:
            continue

    return {
        "total_files": total_files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "oldest": oldest.isoformat() if oldest else None,
        "newest": newest.isoformat() if newest else None,
    }
