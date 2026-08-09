"""Semantic LLM route intents for the Rust ``llm-router`` service.

This module intentionally contains no provider URLs, credentials, API model IDs, or
fallback order. Those decisions belong to ``~/Development/llm-router``.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

ROUTER_PROVIDER = "llm_router"

ROUTER_STAGES = frozenset(
    {
        "adversarial",
        "architect",
        "audit",
        "chat",
        "classification",
        "code",
        "extraction",
        "planning",
        "review",
        "standard",
    }
)

# Multi-seat orchestration remains ai-mm's responsibility; every seat asks for an
# intent and lets llm-router choose and cascade provider/model attempts.
MODEL_GROUPS = {
    "mm": ["stage:review", "stage:audit", "stage:adversarial"],
    "all": ["stage:review", "stage:audit", "stage:adversarial", "stage:architect"],
    "fast": ["stage:review"],
    "local": ["profile:local_only"],
    "max": ["profile:kimi"],
}

# Stable convenience selectors name a policy boundary, not an API model.
LEGACY_ROUTE_ALIASES = {
    # Stable planning selector. The router-owned profile tries direct DeepSeek
    # V4 Flash first and local Qwen only when that remote attempt is unavailable.
    "deepseek": "profile:rig_planning",
    "local": "profile:local_only",
    "lmstudio": "profile:local_only",
    "kimi": "profile:kimi",
    "commercial": "profile:commercial_compliant",
}

_INTENT_RE = re.compile(r"^(stage|profile):[a-z0-9][a-z0-9_-]*$")


def normalize_model_name(model: str) -> Tuple[str, str]:
    """Resolve a user selector to ``(llm_router, semantic_intent)``.

    Explicit ``stage:`` and ``profile:`` values pass through. Bare known stages are
    normalized to ``stage:<name>``. Historical provider/model names are deprecated
    aliases and never select a vendor directly.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LLM route must be a non-empty string")
    selector = model.strip().lower()

    if selector in ROUTER_STAGES:
        return ROUTER_PROVIDER, f"stage:{selector}"
    if _INTENT_RE.fullmatch(selector):
        return ROUTER_PROVIDER, selector
    if selector in LEGACY_ROUTE_ALIASES:
        intent = LEGACY_ROUTE_ALIASES[selector]
        return ROUTER_PROVIDER, intent

    raise ValueError(
        f"Unknown LLM route '{model}'. Use stage:<intent>, profile:<name>, "
        f"or one of the groups: {', '.join(sorted(MODEL_GROUPS))}"
    )


def get_provider_for_model(model: str) -> Optional[str]:
    """Return the sole execution provider for a valid route, else ``None``."""
    try:
        provider, _intent = normalize_model_name(model)
        return provider
    except ValueError:
        return None


def get_model_display_name(route: str) -> str:
    """Return a human-readable route label without claiming a concrete model."""
    try:
        _provider, intent = normalize_model_name(route)
    except ValueError:
        intent = route
    kind, _, name = intent.partition(":")
    if not name:
        return intent
    return f"llm-router {kind} {name.replace('_', ' ')}"


def get_model_characteristics(route: str) -> dict[str, Any]:
    """Describe the stable client-side characteristics of a router intent."""
    _provider, intent = normalize_model_name(route)
    return {
        "display_name": get_model_display_name(intent),
        "route": intent,
        "routing_owner": "llm-router",
        "provider": ROUTER_PROVIDER,
        "dynamic": True,
    }


def list_all_models() -> dict[str, list[str]]:
    """List supported semantic stage routes, grouped under the router boundary."""
    return {ROUTER_PROVIDER: [f"stage:{stage}" for stage in sorted(ROUTER_STAGES)]}


def list_all_aliases() -> dict[str, str]:
    """Return compatibility aliases for operator visibility."""
    return dict(LEGACY_ROUTE_ALIASES)
