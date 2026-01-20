#!/usr/bin/env python3
"""
Cost logging and usage tracking for AI API calls.

Provides persistent logging of API usage with cost tracking and statistics.
"""

import fcntl
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


def get_cost_log_path() -> Path:
    """Get the path to the cost log file."""
    config_dir = Path.home() / ".config" / "system-playbooks"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "costs.jsonl"


def log_api_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    operation: str = "unknown"
) -> None:
    """
    Log an API call to the cost tracking file with file locking.

    Args:
        model: Model name used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost: Actual cost in USD
        operation: Operation type (plan, review, stabilize)
    """
    log_path = get_cost_log_path()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "operation": operation,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": round(cost, 6),
    }

    try:
        # Use file locking to prevent corruption from concurrent writes
        with open(log_path, "a") as f:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        # Don't fail the operation if logging fails
        import sys
        print(f"Warning: Failed to log cost: {e}", file=sys.stderr)


def get_usage_stats(days: int = None) -> dict:
    """
    Get usage statistics from the cost log.

    Args:
        days: Number of days to look back (None = all time)

    Returns:
        Dictionary with usage statistics
    """
    import sys

    log_path = get_cost_log_path()
    if not log_path.exists():
        return {
            "total_cost": 0,
            "total_calls": 0,
            "by_model": {},
            "by_operation": {},
        }

    cutoff_date = None
    if days is not None:
        cutoff_date = datetime.now() - timedelta(days=days)

    total_cost = 0
    total_calls = 0
    by_model = {}
    by_operation = {}

    try:
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                timestamp = datetime.fromisoformat(entry["timestamp"])

                # Skip if outside date range
                if cutoff_date and timestamp < cutoff_date:
                    continue

                cost = entry["cost"]
                model = entry["model"]
                operation = entry.get("operation", "unknown")

                total_cost += cost
                total_calls += 1

                # By model
                if model not in by_model:
                    by_model[model] = {"cost": 0, "calls": 0}
                by_model[model]["cost"] += cost
                by_model[model]["calls"] += 1

                # By operation
                if operation not in by_operation:
                    by_operation[operation] = {"cost": 0, "calls": 0}
                by_operation[operation]["cost"] += cost
                by_operation[operation]["calls"] += 1
    except Exception as e:
        print(f"Warning: Failed to read cost log: {e}", file=sys.stderr)

    return {
        "total_cost": round(total_cost, 4),
        "total_calls": total_calls,
        "by_model": by_model,
        "by_operation": by_operation,
    }
