#!/usr/bin/env python3
"""
Cost tracking and estimation for OpenAI API usage.

Billing rates as of December 2025.

Caching Support:
- OpenAI: Supports prompt caching with 90% discount on cached input tokens
- Google Gemini: Supports context caching with 75% discount on cached input tokens
- Anthropic Claude: Supports prompt caching with 90% discount on cached input tokens

Pricing Accuracy:
- OpenAI GPT-5.2: Confirmed from official pricing page
- Google Gemini: Confirmed from official pricing page
- Anthropic Claude: Estimated pricing (verify with Anthropic for exact rates)
"""

# API pricing (per 1M tokens)
# Sources:
# - OpenAI: https://openai.com/api/pricing/
# - Google: https://ai.google.dev/pricing
# - Anthropic: https://www.anthropic.com/pricing (Claude prices are approximate)

# Base pricing data
_GPT_INSTANT_PRICING = {
    "input": 0.40,      # $0.40 per 1M tokens
    "output": 1.60,     # $1.60 per 1M tokens
    "cached": 0.04,     # 90% discount on cached input
    "is_estimated": False,
}

_GPT_STANDARD_PRICING = {
    "input": 1.75,      # $1.75 per 1M tokens
    "output": 14.00,    # $14.00 per 1M tokens
    "cached": 0.175,    # 90% discount on cached input
    "is_estimated": False,
}

_GPT_PRO_PRICING = {
    "input": 8.75,      # $8.75 per 1M tokens (estimated 5x standard)
    "output": 70.00,    # $70.00 per 1M tokens (estimated 5x standard)
    "cached": 0.875,    # 90% discount on cached input
    "is_estimated": True,  # Pro pricing is estimated
}

_GEMINI_FLASH_PRICING = {
    "input": 0.075,     # $0.075 per 1M tokens (Google Gemini 3 Flash)
    "output": 0.30,     # $0.30 per 1M tokens
    "cached": 0.01875,  # 75% discount on cached input
    "is_estimated": False,
}

_CLAUDE_SONNET_PRICING = {
    "input": 3.00,      # $3.00 per 1M tokens (Claude Sonnet 4.5)
    "output": 15.00,    # $15.00 per 1M tokens
    "cached": 0.30,     # 90% discount on cached input
    "is_estimated": True,  # Claude pricing is approximate
}

# Model aliases with pricing
# Note: Each model gets an independent copy to prevent shared mutable state
# Note: This dict accepts all user-facing model names (including aliases like "gpt", "gemini")
# bin/ai normalizes these to API model names before calling APIs, but cost_tracker
# accepts any alias for standalone cost estimation
PRICING = {
    # GPT models
    "gpt-5.2-instant": _GPT_INSTANT_PRICING.copy(),
    "gpt-5.2-chat-latest": _GPT_STANDARD_PRICING.copy(),
    "gpt-5.2": _GPT_STANDARD_PRICING.copy(),
    "gpt-5": _GPT_STANDARD_PRICING.copy(),
    "gpt": _GPT_STANDARD_PRICING.copy(),
    "gpt-5.2-pro": _GPT_PRO_PRICING.copy(),

    # Gemini models (gemini-3-flash-preview only)
    "gemini": _GEMINI_FLASH_PRICING.copy(),
    "gemini-3-flash-preview": _GEMINI_FLASH_PRICING.copy(),

    # Claude models (approximate pricing - verify with Anthropic)
    "claude": _CLAUDE_SONNET_PRICING.copy(),
    "claude-sonnet-4-5-20250929": _CLAUDE_SONNET_PRICING.copy(),
}

# Rough token estimation (chars / 4)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough approximation)."""
    if not text:
        return 0
    # Return at least 1 token for non-empty strings
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0
) -> float:
    """
    Estimate cost for an API call.

    Args:
        model: Model name (e.g., "gpt-5.2", "gpt-5.2-instant", "gpt-5.2-pro")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (90% discount)

    Returns:
        Estimated cost in USD
    """
    if model not in PRICING:
        available_models = ", ".join(PRICING.keys())
        raise ValueError(f"Unknown model '{model}'. Available models: {available_models}")

    pricing = PRICING[model]

    # Validate cached_tokens to prevent negative token counts
    cached_tokens = max(0, min(cached_tokens, input_tokens))

    # Calculate costs with safe lookups
    uncached_tokens = input_tokens - cached_tokens
    input_cost = (uncached_tokens / 1_000_000) * pricing.get("input", 0)
    cached_cost = (cached_tokens / 1_000_000) * pricing.get("cached", 0)
    output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)

    total_cost = input_cost + cached_cost + output_cost
    return total_cost


def estimate_cost_from_text(
    model: str,
    input_text: str,
    expected_output_tokens: int = 1000,
    cached_ratio: float = 0.0
) -> dict:
    """
    Estimate cost from input text.

    Args:
        model: Model name
        input_text: Input prompt text
        expected_output_tokens: Expected output length in tokens
        cached_ratio: Ratio of input that will be cached (0.0 to 1.0)

    Returns:
        Dictionary with cost breakdown
    """
    # Validate cached_ratio
    if not (0.0 <= cached_ratio <= 1.0):
        raise ValueError(f"cached_ratio must be between 0.0 and 1.0, got {cached_ratio}")

    input_tokens = estimate_tokens(input_text)
    cached_tokens = int(input_tokens * cached_ratio)

    cost = estimate_cost(model, input_tokens, expected_output_tokens, cached_tokens)

    pricing = PRICING.get(model, {})
    is_estimated = pricing.get("is_estimated", True)  # Default to True for safety

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": expected_output_tokens,
        "cached_tokens": cached_tokens,
        "estimated_cost": cost,
        "cost_formatted": f"${cost:.4f}",
        "is_estimated": is_estimated,
    }


def format_cost_warning(model: str, estimated_cost: float, operation: str = "operation") -> str:
    """
    Format a cost warning message.

    Args:
        model: Model name
        estimated_cost: Estimated cost in USD
        operation: Description of the operation

    Returns:
        Formatted warning string
    """
    if model == "gpt-5.2-pro":
        warning_level = "⚠️  EXPENSIVE"
    elif estimated_cost > 0.10:
        warning_level = "💰 Moderate cost"
    else:
        warning_level = "✓ Low cost"

    return f"""
{warning_level}: {operation}
Model: {model}
Estimated cost: ${estimated_cost:.4f}

Billing rates (per 1M tokens):
  Input:  ${PRICING.get(model, {}).get('input', 0):.2f}
  Output: ${PRICING.get(model, {}).get('output', 0):.2f}
"""


def should_warn_about_cost(model: str, estimated_cost: float, threshold: float = 0.10) -> bool:
    """
    Determine if we should warn the user about cost.

    Args:
        model: Model name
        estimated_cost: Estimated cost in USD
        threshold: Cost threshold in USD (default: 0.10)

    Returns:
        True if warning is needed
    """
    # Always warn for pro
    if model == "gpt-5.2-pro":
        return True

    # Warn if cost exceeds threshold
    if estimated_cost > threshold:
        return True

    return False


# ============================================================================
# Cost Logging
# ============================================================================

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path


def get_cost_log_path() -> Path:
    """Get the path to the cost log file."""
    config_dir = Path.home() / ".config" / "ai"
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
    Log an API call to the cost tracking file.

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
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Don't fail the operation if logging fails
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
    from datetime import timedelta

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


# ============================================================================
# Response Caching
# ============================================================================

def get_cache_dir() -> Path:
    """Get the cache directory path."""
    cache_dir = Path.home() / ".config" / "ai" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(model: str, prompt: str, system_prompt: str = None) -> str:
    """Generate a cache key from model and prompts."""
    content = f"{model}:{system_prompt or ''}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached_response(model: str, prompt: str, system_prompt: str = None, ttl_hours: int = 24) -> str:
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
    Cache an API response.

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
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
    except Exception as e:
        # Don't fail the operation if caching fails
        import sys
        print(f"Warning: Failed to cache response: {e}", file=sys.stderr)


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


if __name__ == "__main__":
    # Example usage
    print("Cost Estimation Examples:\n")

    # Code review
    review_input = "git diff with 500 lines of code changes"
    review_estimate = estimate_cost_from_text("gpt-5.2-instant", review_input * 100, 500)
    print(f"Code Review (gpt-5.2-instant): {review_estimate['cost_formatted']}")

    # Planning
    plan_input = "Design user authentication system with OAuth, JWT, session management"
    plan_estimate = estimate_cost_from_text("gpt-5.2", plan_input * 50, 2000)
    print(f"Planning (gpt-5.2): {plan_estimate['cost_formatted']}")

    # Stabilization (multi-round)
    stabilize_estimate = estimate_cost_from_text("gpt-5.2", plan_input * 100, 4000)
    stabilize_total = stabilize_estimate['estimated_cost'] * 4  # 4 rounds
    print(f"Stabilization 2 rounds (gpt-5.2): ${stabilize_total:.4f}")

    # Pro warning
    pro_estimate = estimate_cost_from_text("gpt-5.2-pro", plan_input * 100, 2000)
    print(f"\nPro Model (gpt-5.2-pro): {pro_estimate['cost_formatted']}")
    print(format_cost_warning("gpt-5.2-pro", pro_estimate['estimated_cost'], "complex planning"))
