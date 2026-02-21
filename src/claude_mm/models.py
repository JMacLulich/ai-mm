"""
Centralized model definitions and metadata.

This is the single source of truth for:
- Valid model names per provider
- Model aliases and mappings
- API model names vs user-facing names
- Model characteristics (speed, cost tier, etc.)
"""

from typing import Any, Dict, Optional, Tuple

# ============================================================================
# Model Registry
# ============================================================================

# OpenAI Models
OPENAI_MODELS = {
    # User-facing name -> API name
    "gpt-5.2-chat-latest": "gpt-5.2-chat-latest",  # Fast workhorse (Instant)
    "gpt-5.2": "gpt-5.2",  # Thinking model for complex work
    "gpt-5.2-pro": "gpt-5.2-pro",  # Premium model, highest quality
    "gpt-4o": "gpt-4o",  # Previous generation
    "gpt-4": "gpt-4",  # Legacy
}

# User-friendly aliases
OPENAI_ALIASES = {
    "gpt": "gpt-5.2",  # Default to thinking model
    "gpt-5": "gpt-5.2",
    "gpt-instant": "gpt-5.2-chat-latest",
    "gpt-5.2-instant": "gpt-5.2-chat-latest",  # Legacy alias (incorrect API name)
}

# Google Gemini Models
GEMINI_MODELS = {
    "gemini-3-pro-preview": "gemini-3-pro-preview",  # Latest Pro - biggest model
    "gemini-2.5-pro": "gemini-2.5-pro",  # Stable Pro
    "gemini-3-flash-preview": "gemini-3-flash-preview",  # Fast, cheap
    "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",  # Experimental
}

GEMINI_ALIASES = {
    "gemini": "gemini-3-pro-preview",  # Default to biggest model for reviews
    "gemini-pro": "gemini-2.5-pro",  # Stable pro
    "gemini-flash": "gemini-3-flash-preview",
}

# Anthropic Claude Models
CLAUDE_MODELS = {
    "claude-opus-4-5-20251101": "claude-opus-4-5-20251101",  # Latest Opus
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",  # Latest Sonnet
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",  # Latest Haiku
}

CLAUDE_ALIASES = {
    "claude": "claude-sonnet-4-5-20250929",  # Default to latest Sonnet
    "opus": "claude-opus-4-5-20251101",  # Opus alias
    "haiku": "claude-haiku-4-5-20251001",
}

# Ollama Models (local)
OLLAMA_MODELS = {
    "qwen2.5:14b-instruct": "qwen2.5:14b-instruct",  # Good balance
    "qwen2.5:7b-instruct": "qwen2.5:7b-instruct",  # Faster
    "llama3:latest": "llama3:latest",  # Meta Llama 3
}

OLLAMA_ALIASES = {
    "ollama": "qwen2.5:14b-instruct",  # Default
    "qwen": "qwen2.5:14b-instruct",
    "llama": "llama3:latest",
}

# Model groups for multi-model reviews
MODEL_GROUPS = {
    "mm": ["gpt-5.2", "gemini", "claude-opus-4-5-20251101"],  # Premium multi-model
    "fast": [
        "gpt-5.2-chat-latest",
        "gemini-3-flash-preview",
        "claude-haiku-4-5-20251001",
    ],  # Fast models
}


# ============================================================================
# Provider-Model Mappings
# ============================================================================


def get_provider_for_model(model: str) -> Optional[str]:
    """
    Get the provider name for a given model.

    Args:
        model: Model name (can be alias or API name)

    Returns:
        Provider name ("openai", "google", "anthropic", "ollama") or None if unknown
    """
    # Check OpenAI
    if model in OPENAI_MODELS or model in OPENAI_ALIASES:
        return "openai"

    # Check Gemini
    if model in GEMINI_MODELS or model in GEMINI_ALIASES:
        return "google"

    # Check Claude
    if model in CLAUDE_MODELS or model in CLAUDE_ALIASES:
        return "anthropic"

    # Check Ollama
    if model in OLLAMA_MODELS or model in OLLAMA_ALIASES:
        return "ollama"

    return None


def normalize_model_name(model: str) -> Tuple[str, str]:
    """
    Convert user-facing model name to (provider, api_model_name).

    Args:
        model: User-facing model name or alias

    Returns:
        Tuple of (provider, api_model_name)

    Raises:
        ValueError: If model is unknown

    Examples:
        >>> normalize_model_name("gpt")
        ("openai", "gpt-5.2")
        >>> normalize_model_name("gpt-5.2-instant")
        ("openai", "gpt-5.2-chat-latest")
        >>> normalize_model_name("gemini")
        ("google", "gemini-3-pro-preview")
        >>> normalize_model_name("ollama")
        ("ollama", "qwen2.5:14b-instruct")
    """
    # Try OpenAI
    if model in OPENAI_MODELS:
        return "openai", OPENAI_MODELS[model]
    if model in OPENAI_ALIASES:
        return "openai", OPENAI_MODELS[OPENAI_ALIASES[model]]

    # Try Gemini
    if model in GEMINI_MODELS:
        return "google", GEMINI_MODELS[model]
    if model in GEMINI_ALIASES:
        return "google", GEMINI_MODELS[GEMINI_ALIASES[model]]

    # Try Claude
    if model in CLAUDE_MODELS:
        return "anthropic", CLAUDE_MODELS[model]
    if model in CLAUDE_ALIASES:
        return "anthropic", CLAUDE_MODELS[CLAUDE_ALIASES[model]]

    # Try Ollama
    if model in OLLAMA_MODELS:
        return "ollama", OLLAMA_MODELS[model]
    if model in OLLAMA_ALIASES:
        return "ollama", OLLAMA_MODELS[OLLAMA_ALIASES[model]]

    raise ValueError(f"Unknown model: {model}")


def get_model_display_name(api_model: str) -> str:
    """
    Get user-friendly display name for an API model.

    Args:
        api_model: API model name (e.g., "gpt-5.2-chat-latest")

    Returns:
        Display name (e.g., "GPT-5.2 Instant")
    """
    display_names = {
        # OpenAI
        "gpt-5.2-chat-latest": "GPT-5.2 Instant",
        "gpt-5.2": "GPT-5.2 Thinking",
        "gpt-5.2-pro": "GPT-5.2 Pro",
        "gpt-4o": "GPT-4o",
        "gpt-4": "GPT-4",
        # Gemini
        "gemini-3-pro-preview": "Gemini 3 Pro",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "gemini-3-flash-preview": "Gemini 3 Flash",
        "gemini-2.0-flash-exp": "Gemini 2.0 Flash (Experimental)",
        # Claude
        "claude-opus-4-5-20251101": "Claude Opus 4.5",
        "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
        # Ollama
        "qwen2.5:14b-instruct": "Qwen 2.5 14B (Local)",
        "qwen2.5:7b-instruct": "Qwen 2.5 7B (Local)",
        "llama3:latest": "Llama 3 (Local)",
    }

    return display_names.get(api_model, api_model)


def get_model_characteristics(api_model: str) -> Dict[str, Any]:
    """
    Get model characteristics (speed, cost tier, context window, etc.).

    Args:
        api_model: API model name

    Returns:
        Dictionary with model characteristics
    """
    # Model characteristics
    chars = {
        # OpenAI
        "gpt-5.2-chat-latest": {
            "speed": "fast",
            "cost_tier": "low",
            "context_window": 128000,
            "description": "Fast workhorse for everyday tasks",
        },
        "gpt-5.2": {
            "speed": "medium",
            "cost_tier": "medium",
            "context_window": 128000,
            "description": "Thinking model for complex reasoning",
        },
        "gpt-5.2-pro": {
            "speed": "slow",
            "cost_tier": "high",
            "context_window": 128000,
            "description": "Premium model with highest quality",
        },
        "gpt-4o": {
            "speed": "medium",
            "cost_tier": "medium",
            "context_window": 128000,
            "description": "Previous generation GPT model",
        },
        "gpt-4": {
            "speed": "slow",
            "cost_tier": "high",
            "context_window": 8192,
            "description": "Legacy GPT-4 model",
        },
        # Gemini
        "gemini-3-pro-preview": {
            "speed": "medium",
            "cost_tier": "medium",
            "context_window": 1000000,
            "description": "Latest Gemini 3 Pro - biggest model",
        },
        "gemini-2.5-pro": {
            "speed": "medium",
            "cost_tier": "medium",
            "context_window": 1000000,
            "description": "Stable Gemini 2.5 Pro",
        },
        "gemini-3-flash-preview": {
            "speed": "fast",
            "cost_tier": "low",
            "context_window": 1000000,
            "description": "Fast, cheap Gemini model",
        },
        "gemini-2.0-flash-exp": {
            "speed": "fast",
            "cost_tier": "low",
            "context_window": 1000000,
            "description": "Experimental Gemini 2.0",
        },
        # Claude
        "claude-opus-4-5-20251101": {
            "speed": "moderate",
            "cost_tier": "high",
            "context_window": 200000,
            "description": "Premium model with maximum intelligence and practical performance",
        },
        "claude-sonnet-4-5-20250929": {
            "speed": "fast",
            "cost_tier": "medium",
            "context_window": 200000,
            "description": "Smart model for complex agents and coding",
        },
        "claude-haiku-4-5-20251001": {
            "speed": "fastest",
            "cost_tier": "low",
            "context_window": 200000,
            "description": "Fastest model with near-frontier intelligence",
        },
        # Ollama
        "qwen2.5:14b-instruct": {
            "speed": "medium",
            "cost_tier": "free",
            "context_window": 128000,
            "description": "Local Qwen 2.5 14B - good balance",
        },
        "qwen2.5:7b-instruct": {
            "speed": "fast",
            "cost_tier": "free",
            "context_window": 128000,
            "description": "Local Qwen 2.5 7B - faster",
        },
        "llama3:latest": {
            "speed": "medium",
            "cost_tier": "free",
            "context_window": 8192,
            "description": "Local Llama 3",
        },
    }

    return chars.get(
        api_model,
        {
            "speed": "unknown",
            "cost_tier": "unknown",
            "context_window": 8192,
            "description": "Unknown model",
        },
    )


def list_all_models() -> Dict[str, list]:
    """
    List all available models grouped by provider.

    Returns:
        Dictionary mapping provider to list of model names
    """
    return {
        "openai": list(OPENAI_MODELS.keys()),
        "google": list(GEMINI_MODELS.keys()),
        "anthropic": list(CLAUDE_MODELS.keys()),
        "ollama": list(OLLAMA_MODELS.keys()),
    }


def list_all_aliases() -> Dict[str, str]:
    """
    List all model aliases and their targets.

    Returns:
        Dictionary mapping alias to target model
    """
    all_aliases = {}
    all_aliases.update(OPENAI_ALIASES)
    all_aliases.update(GEMINI_ALIASES)
    all_aliases.update(CLAUDE_ALIASES)
    all_aliases.update(OLLAMA_ALIASES)
    return all_aliases


# ============================================================================
# CLI Usage
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        print("\n=== Available Models ===\n")
        for provider, models in list_all_models().items():
            print(f"{provider.upper()}:")
            for model in models:
                chars = get_model_characteristics(model)
                print(f"  - {model} ({chars['cost_tier']} cost, {chars['speed']})")
            print()

        print("\n=== Aliases ===\n")
        for alias, target in list_all_aliases().items():
            provider, api_name = normalize_model_name(alias)
            print(f"  {alias} → {api_name} ({provider})")

    elif len(sys.argv) > 2 and sys.argv[1] == "info":
        model = sys.argv[2]
        try:
            provider, api_name = normalize_model_name(model)
            chars = get_model_characteristics(api_name)
            display = get_model_display_name(api_name)

            print(f"\n=== {display} ===\n")
            print(f"Provider: {provider}")
            print(f"API Name: {api_name}")
            print(f"Speed: {chars['speed']}")
            print(f"Cost Tier: {chars['cost_tier']}")
            print(f"Context Window: {chars['context_window']:,} tokens")
            print(f"Description: {chars['description']}")
            print()
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    else:
        print("Usage:")
        print("  python models.py list           # List all models")
        print("  python models.py info <model>   # Get model info")
