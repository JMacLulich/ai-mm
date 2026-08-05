"""Schema-constrained routed assistance for rig planning and recovery.

The model receives a bounded JSON packet and can only return proposals. It has
no repository, shell, mailbox, controller, Docker, or git authority.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from claude_mm.models import normalize_model_name
from claude_mm.providers import get_provider
from claude_mm.usage import log_api_call

MAX_PACKET_BYTES = 160_000
MAX_UNITS = 6
MAX_FILES_PER_UNIT = 40
MAX_COMMANDS_PER_UNIT = 12
PLAN_DECISIONS = {"PROPOSE", "NO_WORK", "BLOCKED"}
RECOVERY_DECISIONS = {"ADVISE", "NO_ACTION", "BLOCKED"}
RECOVERY_ACTIONS = {
    "NONE",
    "WAIT_ACTIVE",
    "RESTART_STALE_CONTROLLER",
    "RESET_LEASE_FREE_IDLE_PANE",
    "REQUEUE_EXHAUSTED_TRANSPORT",
    "CLEAN_PROVEN_ORPHAN_STACK",
    "ESCALATE_CONTROLLER_BUG",
}
ALLOWED_MODELS = {"local", "deepseek"}
ESCALATION_CONFIDENCE = 0.72


class RigAssistError(ValueError):
    """Raised when a rig-assist packet or model response violates the contract."""


def _response_format(mode: str) -> dict[str, Any]:
    """Return the domain-owned JSON-schema response contract."""
    if mode == "recovery":
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "decision",
                "diagnosis",
                "recommended_action",
                "unit_id",
                "evidence",
                "confidence",
            ],
            "properties": {
                "decision": {"type": "string", "enum": sorted(RECOVERY_DECISIONS)},
                "diagnosis": {"type": "string"},
                "recommended_action": {
                    "type": "string",
                    "enum": sorted(RECOVERY_ACTIONS),
                },
                "unit_id": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }
    else:
        unit_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "id",
                "files",
                "depends_on",
                "eligible_rigs",
                "verify",
                "acceptance",
                "prompt",
                "planning",
            ],
            "properties": {
                "id": {"type": "string"},
                "files": {"type": "array", "items": {"type": "string"}},
                "depends_on": {"type": "array", "items": {"type": "string"}},
                "eligible_rigs": {"type": "array", "items": {"type": "string"}},
                "verify": {"type": "array", "items": {"type": "string"}},
                "acceptance": {"type": "array", "items": {"type": "string"}},
                "prompt": {"type": "string"},
                "planning": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "document",
                        "record_id",
                        "verified_head",
                        "verified_at",
                        "evidence",
                        "task_numbers",
                    ],
                    "properties": {
                        "document": {"type": "string"},
                        "record_id": {"type": "string"},
                        "verified_head": {"type": "string"},
                        "verified_at": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "task_numbers": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "summary", "confidence", "questions", "units"],
            "properties": {
                "decision": {"type": "string", "enum": sorted(PLAN_DECISIONS)},
                "summary": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "questions": {"type": "array", "items": {"type": "string"}},
                "units": {"type": "array", "items": unit_schema},
            },
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"rig_assist_{mode}",
            "strict": True,
            "schema": schema,
        },
    }


def _string(value: Any, field: str, *, max_chars: int = 8_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RigAssistError(f"{field} must be a non-empty string")
    result = value.strip()
    if len(result) > max_chars:
        raise RigAssistError(f"{field} exceeds {max_chars} characters")
    return result


def _strings(value: Any, field: str, *, maximum: int, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise RigAssistError(f"{field} must be a {'possibly empty ' if allow_empty else ''}array")
    if len(value) > maximum:
        raise RigAssistError(f"{field} exceeds {maximum} items")
    return [_string(item, f"{field}[]", max_chars=2_000) for item in value]


def _safe_relative_path(value: Any, field: str) -> str:
    path = _string(value, field, max_chars=500).replace("\\", "/")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise RigAssistError(f"{field} must be a safe repository-relative path")
    return path.rstrip("/")


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RigAssistError("confidence must be a number between 0 and 1")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise RigAssistError("confidence must be a number between 0 and 1")
    return result


def validate_packet(packet: Any, mode: str) -> dict[str, Any]:
    if not isinstance(packet, dict):
        raise RigAssistError("packet must be a JSON object")
    encoded = json.dumps(packet, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_PACKET_BYTES:
        raise RigAssistError(f"packet exceeds {MAX_PACKET_BYTES} bytes")
    if packet.get("schema_version") != 1:
        raise RigAssistError("packet.schema_version must be 1")
    if packet.get("mode") != mode:
        raise RigAssistError(f"packet.mode must be {mode}")
    project = packet.get("project")
    if not isinstance(project, dict):
        raise RigAssistError("packet.project must be an object")
    _string(project.get("name"), "packet.project.name", max_chars=200)
    head = _string(project.get("head"), "packet.project.head", max_chars=80)
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head.lower()):
        raise RigAssistError("packet.project.head must be a 40-character git SHA")
    if mode == "plan":
        documents = packet.get("documents")
        if not isinstance(documents, list) or not documents:
            raise RigAssistError("plan packet.documents must be a non-empty array")
        if len(documents) > 8:
            raise RigAssistError("plan packet.documents exceeds 8 items")
        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                raise RigAssistError(f"packet.documents[{index}] must be an object")
            _safe_relative_path(document.get("path"), f"packet.documents[{index}].path")
            _string(document.get("content"), f"packet.documents[{index}].content", max_chars=40_000)
    return packet


def _extract_json(text: str) -> dict[str, Any]:
    payload = text.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        payload = "\n".join(lines[1:-1]) if len(lines) >= 3 else payload
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RigAssistError(f"model response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RigAssistError("model response must be a JSON object")
    return value


def validate_plan_response(response: Any, packet: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise RigAssistError("plan response must be an object")
    decision = _string(response.get("decision"), "decision", max_chars=20).upper()
    if decision not in PLAN_DECISIONS:
        raise RigAssistError(f"decision must be one of {sorted(PLAN_DECISIONS)}")
    summary = _string(response.get("summary"), "summary", max_chars=2_000)
    confidence = _confidence(response.get("confidence"))
    questions = _strings(response.get("questions", []), "questions", maximum=6, allow_empty=True)
    raw_units = response.get("units", [])
    if not isinstance(raw_units, list) or len(raw_units) > MAX_UNITS:
        raise RigAssistError(f"units must be an array with at most {MAX_UNITS} items")
    if decision == "PROPOSE" and not raw_units:
        raise RigAssistError("PROPOSE requires at least one unit")
    if decision != "PROPOSE" and raw_units:
        raise RigAssistError(f"{decision} must not include units")

    documents = {str(item["path"]): item for item in packet["documents"]}
    head = str(packet["project"]["head"])
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, dict):
            raise RigAssistError(f"units[{index}] must be an object")
        unit_id = _string(raw.get("id"), f"units[{index}].id", max_chars=160)
        if unit_id in seen:
            raise RigAssistError(f"duplicate unit id: {unit_id}")
        seen.add(unit_id)
        planning = raw.get("planning")
        if not isinstance(planning, dict):
            raise RigAssistError(f"units[{index}].planning must be an object")
        document = _safe_relative_path(
            planning.get("document"), f"units[{index}].planning.document"
        )
        if document not in documents:
            raise RigAssistError(f"unit {unit_id} cites a document outside the packet")
        verified_head = _string(
            planning.get("verified_head"), f"units[{index}].planning.verified_head", max_chars=80
        )
        if verified_head != head:
            raise RigAssistError(f"unit {unit_id} planning head does not match packet head")
        raw_task_numbers = planning.get("task_numbers", [])
        if not isinstance(raw_task_numbers, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in raw_task_numbers
        ):
            raise RigAssistError(
                f"units[{index}].planning.task_numbers must contain positive integers"
            )
        files = [
            _safe_relative_path(item, f"units[{index}].files[]")
            for item in _strings(
                raw.get("files"), f"units[{index}].files", maximum=MAX_FILES_PER_UNIT
            )
        ]
        units.append(
            {
                "id": unit_id,
                "owner": "ANY",
                "status": "READY",
                "files": files,
                "depends_on": _strings(
                    raw.get("depends_on", []),
                    f"units[{index}].depends_on",
                    maximum=20,
                    allow_empty=True,
                ),
                "eligible_rigs": _strings(
                    raw.get("eligible_rigs", []),
                    f"units[{index}].eligible_rigs",
                    maximum=6,
                    allow_empty=True,
                ),
                "verify": _strings(
                    raw.get("verify"),
                    f"units[{index}].verify",
                    maximum=MAX_COMMANDS_PER_UNIT,
                ),
                "acceptance": _strings(
                    raw.get("acceptance"),
                    f"units[{index}].acceptance",
                    maximum=20,
                ),
                "prompt": _string(raw.get("prompt"), f"units[{index}].prompt"),
                "planning": {
                    "document": document,
                    "record_id": _string(
                        planning.get("record_id"),
                        f"units[{index}].planning.record_id",
                        max_chars=300,
                    ),
                    "verified_head": verified_head,
                    "verified_at": _string(
                        planning.get("verified_at"),
                        f"units[{index}].planning.verified_at",
                        max_chars=80,
                    ),
                    "evidence": _strings(
                        planning.get("evidence"),
                        f"units[{index}].planning.evidence",
                        maximum=12,
                    ),
                    "task_numbers": list(dict.fromkeys(raw_task_numbers)),
                },
            }
        )
    return {
        "schema_version": 1,
        "decision": decision,
        "summary": summary,
        "units": units,
        "questions": questions,
        "confidence": confidence,
        "escalation_recommended": confidence < ESCALATION_CONFIDENCE or decision == "BLOCKED",
    }


def validate_recovery_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise RigAssistError("recovery response must be an object")
    decision = _string(response.get("decision"), "decision", max_chars=20).upper()
    if decision not in RECOVERY_DECISIONS:
        raise RigAssistError(f"decision must be one of {sorted(RECOVERY_DECISIONS)}")
    action = _string(response.get("recommended_action"), "recommended_action", max_chars=80).upper()
    if action not in RECOVERY_ACTIONS:
        raise RigAssistError(f"recommended_action must be one of {sorted(RECOVERY_ACTIONS)}")
    confidence = _confidence(response.get("confidence"))
    return {
        "schema_version": 1,
        "decision": decision,
        "diagnosis": _string(response.get("diagnosis"), "diagnosis", max_chars=2_000),
        "recommended_action": action,
        "unit_id": str(response.get("unit_id") or "").strip()[:160],
        "evidence": _strings(response.get("evidence"), "evidence", maximum=12),
        "confidence": confidence,
        "escalation_recommended": confidence < ESCALATION_CONFIDENCE or decision == "BLOCKED",
    }


def assist(packet: dict[str, Any], *, mode: str, model: str = "local") -> dict[str, Any]:
    packet = validate_packet(packet, mode)
    if model not in ALLOWED_MODELS:
        raise RigAssistError(f"rig assistance model must be one of {sorted(ALLOWED_MODELS)}")
    provider_name, api_model = normalize_model_name(model)
    provider = get_provider(provider_name)
    if mode == "plan":
        system_prompt = (
            "You are a bounded rig planning assistant. You have no execution authority. "
            "Use only the supplied canonical documents and state. Return JSON only. "
            "Propose small disjoint units only from packet intake_frontier; preserve phase "
            "dependencies; never revive terminal history; never invent files, commands, "
            "product decisions, or completion evidence. If exact safe units cannot be "
            "derived, return BLOCKED or NO_WORK. Each proposed unit must include id, files, "
            "depends_on, eligible_rigs, verify (exact executable commands), acceptance, "
            "prompt, and planning {document, record_id, verified_head, verified_at, evidence, "
            "task_numbers}. Maximum six units."
            " Also include confidence as a number from 0 to 1."
        )
    else:
        system_prompt = (
            "You are a bounded rig recovery diagnostician with no execution authority. "
            "Use only supplied evidence. Return JSON only with decision, diagnosis, "
            "recommended_action, unit_id, evidence. recommended_action must be one of: "
            + ", ".join(sorted(RECOVERY_ACTIONS))
            + ". Prefer WAIT_ACTIVE when bounded work is genuinely running; never infer a "
            "stall from silence alone. Also include confidence as a number from 0 to 1."
        )
    request_options: dict[str, Any] = {
        "prompt": json.dumps(packet, separators=(",", ":")),
        "model": api_model,
        "system_prompt": system_prompt,
        "max_tokens": 12_000,
        "response_format": _response_format(mode),
        "metadata": {"operation": f"rig-assist-{mode}", "stage": "planning"},
    }
    request_options["temperature"] = 0.1
    response = provider.complete(
        **request_options,
    )
    log_api_call(
        model=api_model,
        input_tokens=int(response.input_tokens or 0),
        output_tokens=int(response.output_tokens or 0),
        cost=float(response.cost or 0),
        operation=f"rig-assist-{mode}",
    )
    parsed = _extract_json(response.text)
    result = (
        validate_plan_response(parsed, packet)
        if mode == "plan"
        else validate_recovery_response(parsed)
    )
    metadata = response.metadata or {}
    result["assistant"] = {
        "provider": metadata.get("provider", provider_name),
        "profile": metadata.get("profile"),
        "route": api_model,
        "model": response.model,
        "cost": float(response.cost or 0),
    }
    return result
