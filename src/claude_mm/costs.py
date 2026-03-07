#!/usr/bin/env python3
"""
Cost estimation for AI API usage.

Billing rates as of December 2025.

Caching Support:
- OpenAI: Supports prompt caching with 90% discount on cached input tokens
- Google Gemini: Supports context caching with 75% discount on cached input tokens
- Anthropic Claude: Supports prompt caching with 90% discount on cached input tokens

Pricing is loaded from pricing.py. This module adds caching discount calculations.
"""

from claude_mm.models import normalize_model_name
from claude_mm.pricing import get_model_pricing

# Caching discounts by provider
CACHE_DISCOUNTS = {
    "openai": 0.90,  # 90% discount on cached input
    "google": 0.75,  # 75% discount on cached input
    "anthropic": 0.90,  # 90% discount on cached input
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
    model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0
) -> float:
    """
    Estimate cost for an API call.

    Args:
        model: Model name (e.g., "gpt-5.4", "gpt-5.2-chat-latest", "gpt-5.2-pro")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (discount varies by provider)

    Returns:
        Estimated cost in USD
    """
    try:
        provider, api_model = normalize_model_name(model)
    except ValueError as e:
        raise ValueError(f"Unknown model '{model}': {e}")

    pricing = get_model_pricing(provider, api_model)
    if not pricing:
        raise ValueError(f"No pricing data available for {provider}/{api_model}")

    # Validate cached_tokens to prevent negative token counts
    cached_tokens = max(0, min(cached_tokens, input_tokens))

    # Calculate costs
    uncached_tokens = input_tokens - cached_tokens
    input_cost = (uncached_tokens / 1_000_000) * pricing.get("input", 0)

    # Apply caching discount
    cache_discount = CACHE_DISCOUNTS.get(provider, 0.90)
    cached_price = pricing.get("input", 0) * (1 - cache_discount)
    cached_cost = (cached_tokens / 1_000_000) * cached_price

    output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)

    total_cost = input_cost + cached_cost + output_cost
    return total_cost


def estimate_cost_from_text(
    model: str, input_text: str, expected_output_tokens: int = 1000, cached_ratio: float = 0.0
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

    # Pro models are expensive, mark as estimated
    is_estimated = "pro" in model.lower()

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
    if "pro" in model.lower():
        warning_level = "⚠️  EXPENSIVE"
    elif estimated_cost > 0.10:
        warning_level = "💰 Moderate cost"
    else:
        warning_level = "✓ Low cost"

    try:
        provider, api_model = normalize_model_name(model)
        pricing = get_model_pricing(provider, api_model)
        input_price = pricing.get("input", 0) if pricing else 0
        output_price = pricing.get("output", 0) if pricing else 0
        pricing_note = (
            "Note: Pricing for this model is estimated.\n"
            if pricing and pricing.get("estimated")
            else ""
        )
    except (ValueError, Exception):
        input_price = 0
        output_price = 0
        pricing_note = ""

    return f"""
{warning_level}: {operation}
Model: {model}
Estimated cost: ${estimated_cost:.4f}
{pricing_note}

Billing rates (per 1M tokens):
  Input:  ${input_price:.2f}
  Output: ${output_price:.2f}
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
    review_estimate = estimate_cost_from_text("gpt-5.2-chat-latest", review_input * 100, 500)
    print(f"Code Review (gpt-5.2-chat-latest): {review_estimate['cost_formatted']}")

    # Planning
    plan_input = "Design user authentication system with OAuth, JWT, session management"
    plan_estimate = estimate_cost_from_text("gpt-5.2", plan_input * 50, 2000)
    print(f"Planning (gpt-5.2): {plan_estimate['cost_formatted']}")

    # Stabilization (multi-round)
    stabilize_estimate = estimate_cost_from_text("gpt-5.2", plan_input * 100, 4000)
    stabilize_total = stabilize_estimate["estimated_cost"] * 4  # 4 rounds
    print(f"Stabilization 2 rounds (gpt-5.2): ${stabilize_total:.4f}")

    # Pro warning
    pro_estimate = estimate_cost_from_text("gpt-5.2-pro", plan_input * 100, 2000)
    print(f"\nPro Model (gpt-5.2-pro): {pro_estimate['cost_formatted']}")
    print(format_cost_warning("gpt-5.2-pro", pro_estimate["estimated_cost"], "complex planning"))
