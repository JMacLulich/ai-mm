#!/usr/bin/env python3
"""
Cost estimation for AI API usage.

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
# bin/ai normalizes these to API model names before calling APIs, but costs.py
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
