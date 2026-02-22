"""Unit tests for advanced planning pipeline."""

import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm import planning
from claude_mm.providers.base import ProviderResponse


def _valid_plan() -> dict:
    return {
        "summary": "Implement reliable auth migration.",
        "objective": "Migrate auth service with zero downtime.",
        "constraints": ["No production downtime"],
        "non_goals": ["No UI redesign"],
        "assumptions": ["Feature flag framework exists"],
        "unknowns": ["Exact peak traffic windows"],
        "blocking_questions": ["Who approves production migration window?"],
        "architecture_decisions": ["Use dual-write during migration"],
        "tasks": [
            {
                "id": "T1",
                "title": "Add dual-write",
                "description": "Write to old and new auth stores.",
                "depends_on": [],
                "verification": ["Unit tests for write path"],
            },
            {
                "id": "T2",
                "title": "Backfill data",
                "description": "Backfill existing users.",
                "depends_on": ["T1"],
                "verification": ["Backfill checksum report"],
            },
        ],
        "test_strategy": ["Add migration integration tests"],
        "success_criteria": ["Backfill mismatch rate < 0.1%"],
        "risks": [
            {
                "id": "R1",
                "description": "Data mismatch across stores",
                "impact": "high",
                "likelihood": "medium",
                "mitigation": "Checksum every migration batch",
                "detection": "Alert on mismatch threshold",
            }
        ],
        "rollout": ["Enable dual-write for 5% traffic"],
        "rollback": ["Disable dual-write feature flag"],
        "observability": ["Dashboard for dual-write errors"],
        "confidence": 0.86,
    }


def test_extract_json_object_handles_code_fence():
    payload = '```json\n{"summary": "ok"}\n```'
    parsed = planning.extract_json_object(payload)
    assert parsed["summary"] == "ok"


def test_validate_plan_document_detects_dependency_cycle():
    doc = _valid_plan()
    doc["tasks"][0]["depends_on"] = ["T2"]
    normalized, errors, _warnings = planning.validate_plan_document(doc)

    assert normalized["tasks"]
    assert any("dependency cycle" in message for message in errors)


def test_generate_plan_output_with_stub_provider(monkeypatch):
    class StubProvider:
        def complete(self, prompt, model, system_prompt=None, temperature=0.2):
            if "adversarial" in (system_prompt or ""):
                payload = {
                    "critical_issues": [],
                    "high_priority_issues": [],
                    "dependency_gaps": [],
                    "test_gaps": [],
                    "risk_gaps": [],
                    "blocking_questions": [],
                    "confidence_adjustment": 0.0,
                }
            else:
                payload = _valid_plan()
            return ProviderResponse(
                text=json.dumps(payload),
                model=model,
                input_tokens=11,
                output_tokens=13,
                cost=Decimal("0.01"),
                cached=False,
            )

    monkeypatch.setattr(planning, "normalize_model_name", lambda _model: ("openai", "gpt-5.2"))
    monkeypatch.setattr(planning, "get_provider", lambda _provider: StubProvider())
    monkeypatch.setattr(planning, "get_cached_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(planning, "cache_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(planning, "log_api_call", lambda **kwargs: None)

    output = planning.generate_plan_output(
        "Migrate auth service",
        "gpt",
        rounds=1,
        output_format="markdown",
        context_mode="none",
        use_cache=False,
    )

    assert "# Implementation Plan" in output.text
    assert output.model == "gpt"
    assert output.cost == Decimal("0.03")
    assert output.input_tokens == 33
    assert output.output_tokens == 39
