#!/usr/bin/env python3
"""
Configuration management for AI tooling.

Loads user configuration from ~/.config/ai/config.yaml with sensible defaults.
"""

from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "default_models": {
        "plan": "gpt-5.2",
        "review": "gpt-5.2-instant",
    },
    "cost_warning_threshold": 0.10,
    "cache_ttl_hours": 24,
}


def load_config(config_path: Path = None):
    """
    Load user configuration from file or use defaults.

    Args:
        config_path: Optional path to config file (for testing)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path.home() / ".config" / "ai" / "config.yaml"

    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    try:
        import yaml
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config
    except ImportError:
        print("Warning: pyyaml not installed, using default config")
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"Warning: Failed to load config: {e}, using defaults")
        return DEFAULT_CONFIG.copy()


def get_default_config():
    """Get the default configuration without loading from file."""
    return DEFAULT_CONFIG.copy()
