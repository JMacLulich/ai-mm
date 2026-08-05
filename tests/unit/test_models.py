"""Tests for semantic route normalization."""

import pytest

from claude_mm.models import (
    MODEL_GROUPS,
    ROUTER_PROVIDER,
    get_model_characteristics,
    get_model_display_name,
    get_provider_for_model,
    list_all_aliases,
    list_all_models,
    normalize_model_name,
)


def test_bare_stage_normalizes_to_router_intent() -> None:
    assert normalize_model_name("review") == (ROUTER_PROVIDER, "stage:review")


def test_explicit_stage_and_profile_pass_through() -> None:
    assert normalize_model_name("stage:audit") == (ROUTER_PROVIDER, "stage:audit")
    assert normalize_model_name("profile:kimi") == (ROUTER_PROVIDER, "profile:kimi")


def test_deepseek_is_a_provider_level_selector_for_router_owned_flash_profile() -> None:
    assert normalize_model_name("deepseek") == (ROUTER_PROVIDER, "stage:audit")


def test_lmstudio_is_a_compatibility_alias_for_router_owned_local_profile() -> None:
    assert normalize_model_name("lmstudio") == (ROUTER_PROVIDER, "profile:local_only")


def test_direct_api_model_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown LLM route"):
        normalize_model_name("vendor/model-id")
    with pytest.raises(ValueError, match="Unknown LLM route"):
        normalize_model_name("vendor-exact-version")


def test_unknown_or_empty_selector_is_rejected() -> None:
    assert get_provider_for_model("made-up-model") is None
    with pytest.raises(ValueError, match="non-empty"):
        normalize_model_name("")


def test_all_groups_contain_only_semantic_intents() -> None:
    assert set(MODEL_GROUPS) == {"all", "fast", "local", "max", "mm"}
    for routes in MODEL_GROUPS.values():
        assert routes
        assert all(route.startswith(("stage:", "profile:")) for route in routes)


def test_model_metadata_describes_dynamic_router_ownership() -> None:
    characteristics = get_model_characteristics("stage:review")
    assert characteristics["provider"] == ROUTER_PROVIDER
    assert characteristics["routing_owner"] == "llm-router"
    assert characteristics["dynamic"] is True
    assert get_model_display_name("profile:kimi") == "llm-router profile kimi"


def test_operator_lists_have_no_api_model_ids() -> None:
    models = list_all_models()
    aliases = list_all_aliases()
    assert set(models) == {ROUTER_PROVIDER}
    assert "deepseek" in aliases
    assert aliases["lmstudio"] == "profile:local_only"
    assert all(route.startswith(("stage:", "profile:")) for route in aliases.values())
