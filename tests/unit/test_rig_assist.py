from __future__ import annotations

import json

import pytest

from claude_mm.providers.base import ProviderError, ProviderResponse
from claude_mm.rig_assist import (
    RigAssistError,
    _extract_json,
    assist,
    validate_packet,
    validate_plan_response,
    validate_recovery_response,
)

HEAD = "a" * 40


@pytest.fixture(autouse=True)
def disable_cost_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("claude_mm.rig_assist.log_api_call", lambda **_kwargs: None)


def plan_packet() -> dict:
    return {
        "schema_version": 1,
        "mode": "plan",
        "project": {"name": "myapp", "head": HEAD},
        "intake_frontier": [
            {
                "document": ".planning/phases/10-feature/10-PLAN.md",
                "phase": "10",
                "intake_action": "ENQUEUE_RESIDUAL",
                "remaining_tasks": [{"number": 2, "name": "Finish it"}],
            }
        ],
        "documents": [
            {
                "path": ".planning/phases/10-feature/10-PLAN.md",
                "content": "# Phase 10\n\nTask 2 is exact and bounded.",
            }
        ],
        "active_units": [],
        "constraints": ["No shell authority"],
    }


def plan_response() -> dict:
    return {
        "decision": "PROPOSE",
        "summary": "Propose only the residual task.",
        "confidence": 0.91,
        "questions": [],
        "units": [
            {
                "id": "p10-t2",
                "files": ["src/feature.py", "tests/test_feature.py"],
                "depends_on": ["p10-t1"],
                "eligible_rigs": ["Rig B", "Rig C"],
                "verify": ["./run test tests/test_feature.py"],
                "acceptance": ["The residual behavior is proven."],
                "prompt": "Implement only Phase 10 task 2.",
                "planning": {
                    "document": ".planning/phases/10-feature/10-PLAN.md",
                    "record_id": "10-01-T2",
                    "verified_head": HEAD,
                    "verified_at": "2026-08-05T00:00:00Z",
                    "evidence": ["Phase 10 task 2"],
                    "task_numbers": [2],
                },
            }
        ],
    }


def test_plan_response_schema_and_validator_reject_noncanonical_rig_names() -> None:
    from claude_mm.rig_assist import _response_schema

    eligible_items = _response_schema("plan")["properties"]["units"]["items"]["properties"][
        "eligible_rigs"
    ]["items"]
    assert eligible_items == {
        "type": "string",
        "enum": ["Rig A", "Rig B", "Rig C"],
    }

    response = plan_response()
    response["units"][0]["eligible_rigs"] = ["rig-a"]

    with pytest.raises(RigAssistError, match=r"eligible_rigs.*Rig A.*Rig B.*Rig C"):
        validate_plan_response(response, plan_packet())


def test_plan_assist_is_schema_constrained_and_uses_deepseek_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubProvider:
        def complete(self, **kwargs):
            assert kwargs["model"] == "stage:audit"
            # Strict json_schema is rejected HTTP 400 by DeepSeek Flash, so
            # the first wire hint must be json_object.
            assert kwargs["response_format"] == {"type": "json_object"}
            assert kwargs["temperature"] == 0.1
            assert "reasoning_effort" not in kwargs
            return ProviderResponse(
                text=json.dumps(plan_response()),
                model="served-flash",
                input_tokens=100,
                output_tokens=100,
                metadata={
                    "provider": "deepseek",
                    "profile": "deepseek_v4_flash_direct",
                },
            )

    monkeypatch.setattr("claude_mm.rig_assist.get_provider", lambda name: StubProvider())

    result = assist(plan_packet(), mode="plan", model="deepseek")

    assert result["decision"] == "PROPOSE"
    assert result["units"][0]["owner"] == "ANY"
    assert result["units"][0]["status"] == "READY"
    assert result["units"][0]["planning"]["task_numbers"] == [2]
    with pytest.raises(RigAssistError, match="must be one of"):
        assist(plan_packet(), mode="plan", model="gpt")


def test_plan_assist_defaults_to_router_local_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubProvider:
        def complete(self, **kwargs):
            assert kwargs["model"] == "profile:local_only"
            assert kwargs["temperature"] == 0.1
            assert kwargs["response_format"] == {"type": "json_object"}
            assert "reasoning_effort" not in kwargs
            # The schema is carried in the prompt, not on the wire.
            assert '"additionalProperties":false' in kwargs["system_prompt"]
            return ProviderResponse(
                text=json.dumps(plan_response()),
                model="served-local",
                input_tokens=100,
                output_tokens=100,
                metadata={"provider": "ollama", "profile": "local_only"},
            )

    monkeypatch.setattr("claude_mm.rig_assist.get_provider", lambda name: StubProvider())

    result = assist(plan_packet(), mode="plan")

    assert result["assistant"]["provider"] == "ollama"
    assert result["assistant"]["route"] == "profile:local_only"
    assert result["assistant"]["cost"] == 0.0
    assert result["escalation_recommended"] is False


def test_plan_assist_degrades_response_format_when_route_rejects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LM Studio answers json_object with HTTP 400; the unconstrained rung must win.

    Before the ladder existed this raised
    ProviderError('llm-router request failed: ... http_400') and every
    `rig-v2 replan --model local` failed closed.
    """
    seen: list[object] = []

    class StubProvider:
        def complete(self, **kwargs):
            seen.append(kwargs.get("response_format"))
            if kwargs.get("response_format") is not None:
                raise ProviderError("llm-router request failed: http_400")
            return ProviderResponse(
                text="```json\n" + json.dumps(plan_response()) + "\n```",
                model="served-local",
                input_tokens=100,
                output_tokens=100,
                metadata={"provider": "lmstudio", "profile": "local_only"},
            )

    monkeypatch.setattr("claude_mm.rig_assist.get_provider", lambda name: StubProvider())

    result = assist(plan_packet(), mode="plan")

    assert seen == [{"type": "json_object"}, None]
    assert result["decision"] == "PROPOSE"


def test_plan_assist_degrades_when_route_returns_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reasoning-only 200 surfaces as empty content; it must not end the call.

    This is the exact qwen3.6 strict-json_schema failure that closed replan:
    finish_reason=stop with content=''.
    """

    class StubProvider:
        def complete(self, **kwargs):
            if kwargs.get("response_format") is not None:
                raise ProviderError("llm-router returned empty response content")
            return ProviderResponse(
                text=json.dumps(plan_response()),
                model="served-local",
                input_tokens=100,
                output_tokens=100,
                metadata={"provider": "lmstudio", "profile": "local_only"},
            )

    monkeypatch.setattr("claude_mm.rig_assist.get_provider", lambda name: StubProvider())

    assert assist(plan_packet(), mode="plan")["decision"] == "PROPOSE"


def test_plan_assist_reraises_when_every_response_format_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine outage must still fail closed, not be masked by the ladder."""

    class StubProvider:
        def complete(self, **kwargs):
            raise ProviderError("llm-router request failed: 502 all attempts failed")

    monkeypatch.setattr("claude_mm.rig_assist.get_provider", lambda name: StubProvider())

    with pytest.raises(ProviderError, match="all response_format rungs failed"):
        assist(plan_packet(), mode="plan")


def test_format_ladder_reports_every_rung_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-rung cause must survive; reporting only the first hides the real health."""
    errors = iter(
        [
            ProviderError("read timeout"),
            ProviderError("502 all attempts failed"),
        ]
    )

    class StubProvider:
        def complete(self, **kwargs):
            raise next(errors)

    monkeypatch.setattr("claude_mm.rig_assist.get_provider", lambda name: StubProvider())

    with pytest.raises(ProviderError) as excinfo:
        assist(plan_packet(), mode="plan")

    message = str(excinfo.value)
    assert "json_object: read timeout" in message
    assert "none: 502 all attempts failed" in message


def test_extract_json_recovers_object_embedded_in_prose() -> None:
    """The unconstrained rung lets a model wrap its answer in commentary."""
    payload = (
        'Here is the plan:\n{"decision": "NO_WORK", "note": "a } brace in a string",'
        ' "nested": {"a": 1}}\nHope that helps!'
    )

    assert _extract_json(payload) == {
        "decision": "NO_WORK",
        "note": "a } brace in a string",
        "nested": {"a": 1},
    }

    with pytest.raises(RigAssistError, match="not valid JSON"):
        _extract_json("no json here at all")


def test_extract_json_refuses_two_candidate_objects() -> None:
    """Two top-level objects is ambiguous; picking one silently would hide output."""
    with pytest.raises(RigAssistError, match="contains 2 top-level JSON objects"):
        _extract_json('Draft: {"decision": "NO_WORK"} Final: {"decision": "PROPOSE"}')


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda response: response["units"][0]["files"].append("../escape.py"),
            "safe repository-relative",
        ),
        (
            lambda response: response["units"][0]["planning"].update(
                {"document": ".planning/phases/99-invented/99-PLAN.md"}
            ),
            "outside the packet",
        ),
        (
            lambda response: response["units"][0]["planning"].update({"verified_head": "b" * 40}),
            "does not match packet head",
        ),
        (
            lambda response: response["units"][0]["planning"].update({"task_numbers": [True]}),
            "positive integers",
        ),
    ],
)
def test_plan_response_rejects_authority_escape(mutation, message: str) -> None:
    response = plan_response()
    mutation(response)

    with pytest.raises(RigAssistError, match=message):
        validate_plan_response(response, plan_packet())


def test_packet_is_bounded_and_requires_known_mode() -> None:
    packet = plan_packet()
    packet["documents"][0]["content"] = "x" * 160_001
    with pytest.raises(RigAssistError, match="packet exceeds"):
        validate_packet(packet, "plan")

    packet = plan_packet()
    packet["mode"] = "recovery"
    with pytest.raises(RigAssistError, match="packet.mode must be plan"):
        validate_packet(packet, "plan")


def test_recovery_advice_is_whitelisted_and_advisory_only() -> None:
    result = validate_recovery_response(
        {
            "decision": "ADVISE",
            "diagnosis": "The transport retry budget is exhausted.",
            "recommended_action": "REQUEUE_EXHAUSTED_TRANSPORT",
            "unit_id": "phase-10",
            "evidence": ["three failed delivery attempts"],
            "confidence": 0.94,
        }
    )

    assert result["recommended_action"] == "REQUEUE_EXHAUSTED_TRANSPORT"
    assert result["escalation_recommended"] is False
    with pytest.raises(RigAssistError, match="recommended_action must be one of"):
        validate_recovery_response(
            {
                "decision": "ADVISE",
                "diagnosis": "Try something unbounded.",
                "recommended_action": "RUN_ARBITRARY_SHELL",
                "unit_id": "phase-10",
                "evidence": ["none"],
                "confidence": 0.9,
            }
        )
