#!/usr/bin/env python3
"""
Response caching for AI API calls.

Provides disk-based caching with TTL support and atomic writes to prevent race conditions.
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get the cache directory path, creating it with secure permissions."""
    cache_dir = Path.home() / ".config" / "ai-mm" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Enforce permissions even if directory already existed
    cache_dir.chmod(0o700)
    return cache_dir


def get_cache_key(model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """Generate a cache key from model and prompts.

    Uses JSON serialization to avoid delimiter-collision issues that arise with
    simple string concatenation when components contain the separator character.
    """
    content = json.dumps([model, system_prompt or "", prompt], ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached_response(
    model: str, prompt: str, system_prompt: Optional[str] = None, ttl_hours: int = 24
) -> Optional[str]:
    """
    Get cached response if available and not expired.

    Args:
        model: Normalized model ID (not alias) for stable cache keys
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

        # Support both new UTC-aware and legacy naive timestamps
        cached_at = datetime.fromisoformat(cache_data["timestamp"])
        if cached_at.tzinfo is None:
            age = datetime.now() - cached_at
        else:
            age = datetime.now(timezone.utc) - cached_at

        if age > timedelta(hours=ttl_hours):
            cache_file.unlink(missing_ok=True)
            return None

        return cache_data["response"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        # Corrupt, unreadable, or expired cache file — delete it and treat as miss
        cache_file.unlink(missing_ok=True)
        return None


def cache_response(
    model: str, prompt: str, response: str, system_prompt: Optional[str] = None
) -> None:
    """
    Cache an API response using atomic write to prevent race conditions.

    Cache files are written with 0o600 permissions to protect potentially sensitive content.

    Args:
        model: Normalized model ID (not alias) for stable cache keys
        prompt: User prompt
        response: API response text
        system_prompt: System prompt (optional)
    """
    cache_dir = get_cache_dir()
    cache_key = get_cache_key(model, prompt, system_prompt)
    cache_file = cache_dir / f"{cache_key}.json"

    cache_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "response": response,
    }

    tmp_path = None
    try:
        # Assign tmp_path before json.dump so cleanup works on serialization failure
        with tempfile.NamedTemporaryFile(
            mode="w", dir=cache_dir, delete=False, suffix=".tmp"
        ) as tmp_file:
            tmp_path = tmp_file.name
            json.dump(cache_data, tmp_file)

        # Restrict permissions before making the file visible
        os.chmod(tmp_path, 0o600)
        # Atomic rename (replaces existing file if present)
        os.replace(tmp_path, cache_file)
    except Exception as e:
        logger.warning("Failed to cache response: %s", e)
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def clear_cache(older_than_hours: Optional[int] = None) -> int:
    """
    Clear cached responses.

    Uses file modification time for efficient filtering without parsing JSON.

    Args:
        older_than_hours: Only clear cache older than N hours (None = all)

    Returns:
        Number of cache files removed
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    removed = 0
    cutoff_ts = None
    if older_than_hours is not None:
        cutoff_ts = datetime.now(timezone.utc).timestamp() - (older_than_hours * 3600)

    for cache_file in cache_dir.glob("*.json"):
        try:
            if cutoff_ts is not None:
                if cache_file.stat().st_mtime > cutoff_ts:
                    continue
            cache_file.unlink()
            removed += 1
        except Exception:
            continue

    return removed


def get_cache_stats() -> dict:
    """
    Get cache statistics.

    Uses file modification time for efficient scanning without parsing JSON.

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
    oldest_ts = None
    newest_ts = None

    for cache_file in cache_dir.glob("*.json"):
        try:
            stat = cache_file.stat()
            total_files += 1
            total_size += stat.st_size
            mtime = stat.st_mtime
            if oldest_ts is None or mtime < oldest_ts:
                oldest_ts = mtime
            if newest_ts is None or mtime > newest_ts:
                newest_ts = mtime
        except Exception:
            continue

    def ts_to_iso(ts: Optional[float]) -> Optional[str]:
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    return {
        "total_files": total_files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "oldest": ts_to_iso(oldest_ts),
        "newest": ts_to_iso(newest_ts),
    }
