from __future__ import annotations

import json

import pytest

from claude_mm.providers.base import ProviderResponse
from claude_mm.rig_assist import (
    RigAssistError,
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


def test_plan_assist_is_schema_constrained_and_deepseek_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubProvider:
        def complete(self, **kwargs):
            assert kwargs["response_format"] == {"type": "json_object"}
            assert kwargs["reasoning_effort"] == "xhigh"
            return ProviderResponse(
                text=json.dumps(plan_response()),
                model="deepseek-v4-pro",
                input_tokens=100,
                output_tokens=100,
            )

    monkeypatch.setattr("claude_mm.rig_assist.get_provider", lambda name: StubProvider())

    result = assist(plan_packet(), mode="plan", model="deepseek-pro-xhigh")

    assert result["decision"] == "PROPOSE"
    assert result["units"][0]["owner"] == "ANY"
    assert result["units"][0]["status"] == "READY"
    assert result["units"][0]["planning"]["task_numbers"] == [2]
    with pytest.raises(RigAssistError, match="must be one of"):
        assist(plan_packet(), mode="plan", model="gpt")


def test_plan_assist_defaults_to_free_local_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubProvider:
        def complete(self, **kwargs):
            assert kwargs["model"] == "qwen/qwen3.6-35b-a3b"
            assert kwargs["temperature"] == 0.1
            assert kwargs["response_format"]["type"] == "json_schema"
            assert kwargs["response_format"]["json_schema"]["strict"] is True
            assert "reasoning_effort" not in kwargs
            return ProviderResponse(
                text=json.dumps(plan_response()),
                model="qwen/qwen3.6-35b-a3b",
                input_tokens=100,
                output_tokens=100,
            )

    monkeypatch.setattr("claude_mm.rig_assist.get_provider", lambda name: StubProvider())

    result = assist(plan_packet(), mode="plan")

    assert result["assistant"]["provider"] == "lmstudio"
    assert result["assistant"]["cost"] == 0.0
    assert result["escalation_recommended"] is False


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
