#!/usr/bin/env python3
"""
Configuration management for AI tooling.

Loads user configuration from ~/.config/ai/config.yaml with sensible defaults.
"""

from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "ai"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Default configuration
DEFAULT_CONFIG = {
    "default_models": {
        "plan": "gpt-5.2",  # GPT-5.2 Thinking (for complex planning)
        "review": "gpt-5.4",  # GPT-5.4 for deeper, more rigorous reviews
    },
    "review_per_model_timeout_seconds": 60,
    "local_model_timeout_seconds": 240,  # longer timeout for local models (ollama, lmstudio)
    "cost_warning_threshold": 0.10,
    "cache_ttl_hours": 24,
    "planning": {
        "depth": "standard",
        "rounds": 2,
        "context_mode": "none",
        "strict": False,
        "confidence_threshold": 0.72,
    },
}


def _default_config_copy() -> dict:
    return {
        "default_models": dict(DEFAULT_CONFIG["default_models"]),
        "review_per_model_timeout_seconds": DEFAULT_CONFIG["review_per_model_timeout_seconds"],
        "cost_warning_threshold": DEFAULT_CONFIG["cost_warning_threshold"],
        "cache_ttl_hours": DEFAULT_CONFIG["cache_ttl_hours"],
        "planning": dict(DEFAULT_CONFIG["planning"]),
    }


def load_user_config(config_path: Path | None = None) -> dict:
    """Load raw user configuration from file.

    Args:
        config_path: Optional path to config file (for testing)

    Returns:
        User configuration dictionary, or an empty dict if unavailable
    """
    if config_path is None:
        config_path = CONFIG_FILE

    if not config_path.exists():
        return {}

    try:
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("Warning: pyyaml not installed, ignoring user config")
        return {}
    except Exception as e:
        print(f"Warning: Failed to load config: {e}, ignoring user config")
        return {}


def load_config(config_path: Path | None = None):
    """Load merged configuration with defaults applied."""
    user_config = load_user_config(config_path)
    config = _default_config_copy()
    config.update(user_config)
    return config


def save_user_config(config: dict, config_path: Path | None = None) -> None:
    """Save user configuration to disk."""
    if config_path is None:
        config_path = CONFIG_FILE

    config_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config():
    """Get the default configuration without loading from file."""
    return _default_config_copy()
