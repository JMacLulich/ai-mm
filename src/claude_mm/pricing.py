"""
External pricing configuration with auto-update capability.

This module manages LLM pricing data in an external YAML file that can be
updated independently of the code. Includes fallback to embedded defaults.
"""

import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml

DEFAULT_PRICING = {
    "openai": {
        "gpt-5.2-chat-latest": {"input": 0.40, "output": 1.60},  # GPT-5.2 Instant (fast)
        "gpt-5.2": {"input": 1.75, "output": 14.00},  # GPT-5.2 Thinking (standard)
        "gpt-5.2-pro": {"input": 21.00, "output": 84.00},  # GPT-5.2 Pro (expensive)
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
    },
    "google": {
        "gemini-3.1-pro-preview": {"input": 1.25, "output": 10.00},
        "gemini-3-pro-preview": {"input": 1.25, "output": 10.00},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
        "gemini-pro": {"input": 0.50, "output": 1.50},
    },
    "anthropic": {
        "claude-opus-4-6": {"input": 5.00, "output": 25.00},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
        "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    },
    "_metadata": {
        "last_updated": datetime.now().isoformat(),
        "version": "1.0.0",
        "source": "embedded_defaults",
    },
}


def get_pricing_file() -> Path:
    """Get the path to the pricing configuration file."""
    config_dir = Path.home() / ".config" / "system-playbooks"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "pricing.yaml"


def load_pricing() -> Dict:
    """
    Load pricing data from external file or use defaults.

    Returns:
        Dictionary with pricing data by provider and model
    """
    pricing_file = get_pricing_file()

    # If file exists and is recent, use it
    if pricing_file.exists():
        try:
            with open(pricing_file) as f:
                pricing = yaml.safe_load(f)
                if pricing and "_metadata" in pricing:
                    return pricing
        except Exception as e:
            print(f"Warning: Failed to load pricing file: {e}")
            print("Using embedded defaults")

    # Otherwise, create with defaults
    save_pricing(DEFAULT_PRICING)
    return DEFAULT_PRICING


def save_pricing(pricing: Dict) -> None:
    """
    Save pricing data to external file.

    Args:
        pricing: Pricing dictionary to save
    """
    pricing_file = get_pricing_file()

    try:
        with open(pricing_file, "w") as f:
            yaml.dump(pricing, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Warning: Failed to save pricing file: {e}")


def get_model_pricing(provider: str, model: str) -> Optional[Dict]:
    """
    Get pricing for a specific model.

    Args:
        provider: Provider name (openai, google, anthropic)
        model: Model identifier

    Returns:
        Dictionary with 'input' and 'output' prices per 1M tokens, or None
    """
    pricing = load_pricing()
    provider_pricing = pricing.get(provider, {})

    # Try exact match first
    if model in provider_pricing:
        return provider_pricing[model]

    # Try to find a default for the provider from loaded pricing
    if provider == "openai":
        for fallback in ["gpt-5.2", "gpt-5.2-chat-latest"]:
            if fallback in provider_pricing:
                return provider_pricing[fallback]
    elif provider == "google":
        for fallback in [
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
        ]:
            if fallback in provider_pricing:
                return provider_pricing[fallback]
    elif provider == "anthropic":
        for fallback in ["claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5-20251101"]:
            if fallback in provider_pricing:
                return provider_pricing[fallback]

    # Fall back to embedded defaults when user pricing file is stale
    default_provider_pricing = DEFAULT_PRICING.get(provider, {})
    if model in default_provider_pricing:
        return default_provider_pricing[model]

    if provider == "openai":
        for fallback in ["gpt-5.2", "gpt-5.2-chat-latest"]:
            if fallback in default_provider_pricing:
                return default_provider_pricing[fallback]
    elif provider == "google":
        for fallback in [
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
        ]:
            if fallback in default_provider_pricing:
                return default_provider_pricing[fallback]
    elif provider == "anthropic":
        for fallback in ["claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5-20251101"]:
            if fallback in default_provider_pricing:
                return default_provider_pricing[fallback]

    return None


def update_pricing_from_url(url: str) -> bool:
    """
    Update pricing from a remote URL.

    This allows pricing to be updated without code changes.

    Args:
        url: URL to fetch pricing JSON from

    Returns:
        True if update succeeded, False otherwise

    Expected JSON format:
        {
            "openai": {
                "gpt-5.2": {"input": 1.75, "output": 14.00},
                ...
            },
            ...
        }
    """
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            new_pricing = json.loads(response.read().decode())

            # Add metadata
            new_pricing["_metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "version": new_pricing.get("_metadata", {}).get("version", "1.0.0"),
                "source": url,
            }

            # Validate basic structure
            required_providers = ["openai", "google", "anthropic"]
            if not all(p in new_pricing for p in required_providers):
                print("Warning: Pricing data missing required providers")
                return False

            # Save new pricing
            save_pricing(new_pricing)
            print(f"✓ Updated pricing from {url}")
            return True

    except Exception as e:
        print(f"Failed to update pricing from {url}: {e}")
        return False


def check_pricing_age() -> Optional[int]:
    """
    Check how old the pricing data is.

    Returns:
        Age in days, or None if metadata unavailable
    """
    pricing = load_pricing()
    metadata = pricing.get("_metadata", {})
    last_updated = metadata.get("last_updated")

    if not last_updated:
        return None

    try:
        updated_time = datetime.fromisoformat(last_updated)
        age = datetime.now() - updated_time
        return age.days
    except Exception:
        return None


def suggest_pricing_update() -> bool:
    """
    Check if pricing should be updated (older than 30 days).

    Returns:
        True if update is suggested
    """
    age = check_pricing_age()
    if age is None:
        return False

    return age > 30


# Example usage and CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "update":
        # Update from a URL
        url = sys.argv[2] if len(sys.argv) > 2 else None
        if url:
            success = update_pricing_from_url(url)
            sys.exit(0 if success else 1)
        else:
            print("Usage: python pricing.py update <url>")
            sys.exit(1)

    elif len(sys.argv) > 1 and sys.argv[1] == "show":
        # Show current pricing
        pricing = load_pricing()
        print(yaml.dump(pricing, default_flow_style=False, sort_keys=False))

    elif len(sys.argv) > 1 and sys.argv[1] == "age":
        # Check age
        age = check_pricing_age()
        if age is not None:
            print(f"Pricing is {age} days old")
            if suggest_pricing_update():
                print("⚠️  Consider updating pricing (>30 days old)")
        else:
            print("Pricing age unknown")

    else:
        print("Usage:")
        print("  python pricing.py show           # Display current pricing")
        print("  python pricing.py age            # Check pricing age")
        print("  python pricing.py update <url>   # Update from URL")
