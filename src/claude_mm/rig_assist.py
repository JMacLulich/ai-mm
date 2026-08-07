"""Schema-constrained routed assistance for rig planning and recovery.

The model receives a bounded JSON packet and can only return proposals. It has
no repository, shell, mailbox, controller, Docker, or git authority.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from claude_mm.models import normalize_model_name
from claude_mm.providers import ProviderError, get_provider
from claude_mm.usage import log_api_call

MAX_PACKET_BYTES = 160_000
MAX_UNITS = 6
MAX_FILES_PER_UNIT = 40
MAX_COMMANDS_PER_UNIT = 12
ELIGIBLE_RIGS = ("Rig A", "Rig B", "Rig C")
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

# Response-format degradation ladder.
#
# Neither allowed route serves OpenAI strict `json_schema`: the DeepSeek cascade
# rejects it with HTTP 400, and the LM Studio local route answers 200 with empty
# content (all budget spent on stripped reasoning), which surfaces as
# "llm-router returned empty response content". `json_object` is the inverse --
# served by DeepSeek, rejected 400 by LM Studio. The schema is therefore carried
# in the system prompt and enforced by validate_plan_response /
# validate_recovery_response, and the wire-level hint degrades per route.
RESPONSE_FORMAT_LADDER: tuple[dict[str, Any] | None, ...] = (
    {"type": "json_object"},
    None,
)


class RigAssistError(ValueError):
    """Raised when a rig-assist packet or model response violates the contract."""


def _response_schema(mode: str) -> dict[str, Any]:
    """Return the domain-owned JSON schema for a mode's response contract."""
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
                "eligible_rigs": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ELIGIBLE_RIGS)},
                },
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
    return schema


def _response_format(mode: str) -> dict[str, Any]:
    """Return the strict json_schema wrapper around the mode's response contract.

    Kept as the canonical description of the contract. It is NOT sent on the
    wire -- see RESPONSE_FORMAT_LADDER for why.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"rig_assist_{mode}",
            "strict": True,
            "schema": _response_schema(mode),
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


def _closing_brace(payload: str, start: int) -> int | None:
    """Return the index closing the object opened at ``start``, or None.

    Brace counting is string-literal aware so a brace inside a quoted value
    cannot close the object early.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(payload)):
        char = payload[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _embedded_json_objects(payload: str) -> list[dict[str, Any]]:
    """Return every top-level JSON object embedded in prose, in order.

    Needed because the unconstrained rung of RESPONSE_FORMAT_LADDER lets a model
    wrap its answer in commentary. Nested objects are skipped -- once an object
    parses, scanning resumes after its closing brace.
    """
    found: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = payload.find("{", cursor)
        if start == -1:
            return found
        end = _closing_brace(payload, start)
        if end is None:
            return found
        try:
            value = json.loads(payload[start : end + 1])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            found.append(value)
        cursor = end + 1


def _extract_json(text: str) -> dict[str, Any]:
    payload = text.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        payload = "\n".join(lines[1:-1]) if len(lines) >= 3 else payload
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        embedded = _embedded_json_objects(payload)
        if len(embedded) == 1:
            return embedded[0]
        if len(embedded) > 1:
            # Two candidate answers is ambiguous, not recoverable. Fail closed
            # rather than silently picking one and discarding the rest.
            raise RigAssistError(
                f"model response contains {len(embedded)} top-level JSON objects"
            ) from exc
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
        eligible_rigs = _strings(
            raw.get("eligible_rigs", []),
            f"units[{index}].eligible_rigs",
            maximum=6,
            allow_empty=True,
        )
        invalid_rigs = sorted(set(eligible_rigs).difference(ELIGIBLE_RIGS))
        if invalid_rigs:
            raise RigAssistError(
                f"units[{index}].eligible_rigs must contain only {list(ELIGIBLE_RIGS)}; "
                f"got {invalid_rigs}"
            )
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
                "eligible_rigs": eligible_rigs,
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


def _complete_with_format_ladder(provider: Any, request_options: dict[str, Any]) -> Any:
    """Complete the request, degrading the response-format hint per route.

    Routes disagree about which OpenAI response_format values they serve, and a
    rejection is indistinguishable from an outage at the call site. Each rung is
    attempted in order; the first provider error is re-raised (with the rungs
    tried appended) only if every rung fails, so a genuine outage still surfaces.
    """
    failures: list[tuple[str, ProviderError]] = []
    for response_format in RESPONSE_FORMAT_LADDER:
        rung = "none" if response_format is None else str(response_format.get("type"))
        options = dict(request_options)
        if response_format is not None:
            options["response_format"] = response_format
        try:
            return provider.complete(**options)
        except ProviderError as exc:
            failures.append((rung, exc))
    # Every rung failed. Report all of them -- the last rung is unconstrained, so
    # its error is the one that describes the route's real health, but a caller
    # deciding whether to retry needs to see each rung's distinct cause.
    detail = "; ".join(f"{rung}: {error}" for rung, error in failures)
    raise ProviderError(f"all response_format rungs failed ({detail})") from failures[-1][1]


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
    system_prompt += (
        " Respond with a single JSON object and nothing else -- no prose, no markdown"
        " fence. It must validate against this JSON schema: "
        + json.dumps(_response_schema(mode), separators=(",", ":"))
    )
    request_options: dict[str, Any] = {
        "prompt": json.dumps(packet, separators=(",", ":")),
        "model": api_model,
        "system_prompt": system_prompt,
        "max_tokens": 12_000,
        "metadata": {"operation": f"rig-assist-{mode}", "stage": "planning"},
        "temperature": 0.1,
    }
    response = _complete_with_format_ladder(provider, request_options)
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
