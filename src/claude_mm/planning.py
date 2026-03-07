"""Advanced planning pipeline with critique loops and schema validation."""

from __future__ import annotations

import json
import re
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from claude_mm.cache import cache_response, get_cached_response
from claude_mm.models import normalize_model_name
from claude_mm.providers import get_provider
from claude_mm.providers.base import Provider, ProviderError
from claude_mm.usage import log_api_call

DEFAULT_CONFIDENCE_THRESHOLD = 0.72
PLAN_MODEL_GROUPS = {
    "mm": ["gpt-5.2", "gemini", "claude-opus-4-6"],
}

_PLAN_SCHEMA = {
    "summary": "string",
    "objective": "string",
    "constraints": ["string"],
    "non_goals": ["string"],
    "assumptions": ["string"],
    "unknowns": ["string"],
    "blocking_questions": ["string"],
    "architecture_decisions": ["string"],
    "tasks": [
        {
            "id": "T1",
            "title": "Task title",
            "description": "What to implement",
            "depends_on": ["T0"],
            "verification": ["How to verify this task"],
        }
    ],
    "test_strategy": ["string"],
    "success_criteria": ["string"],
    "risks": [
        {
            "id": "R1",
            "description": "Risk summary",
            "impact": "low|medium|high",
            "likelihood": "low|medium|high",
            "mitigation": "How to prevent",
            "detection": "How to detect early",
        }
    ],
    "rollout": ["string"],
    "rollback": ["string"],
    "observability": ["string"],
    "confidence": 0.0,
}

_CRITIQUE_SCHEMA = {
    "critical_issues": ["string"],
    "high_priority_issues": ["string"],
    "dependency_gaps": ["string"],
    "test_gaps": ["string"],
    "risk_gaps": ["string"],
    "blocking_questions": ["string"],
    "confidence_adjustment": -0.1,
}


class PlanningError(Exception):
    """Raised when plan generation cannot satisfy strict requirements."""


@dataclass
class PlanRun:
    """Output from a single model planning pipeline run."""

    model: str
    plan: dict[str, Any]
    critiques: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]
    confidence: float
    input_tokens: int
    output_tokens: int
    cost: Decimal
    cached: bool


@dataclass
class PlanOutput:
    """Output that can be returned by API plan()."""

    text: str
    model: str
    metadata: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cost: Decimal
    cached: bool


@dataclass
class _StageResult:
    text: str
    cached: bool
    input_tokens: int
    output_tokens: int
    cost: Decimal


def infer_goal_packet(goal: str) -> dict[str, Any]:
    """Infer structured inputs from free-form goal text."""
    lines = [line.strip() for line in goal.splitlines() if line.strip()]
    if not lines:
        return {
            "objective": "",
            "constraints": [],
            "non_goals": [],
            "assumptions": [],
            "success_criteria": [],
            "unknowns": [],
        }

    objective = lines[0]
    sections = {
        "constraints": [],
        "non_goals": [],
        "assumptions": [],
        "success_criteria": [],
        "unknowns": [],
    }
    section_aliases = {
        "constraint": "constraints",
        "constraints": "constraints",
        "non-goal": "non_goals",
        "non-goals": "non_goals",
        "non goal": "non_goals",
        "non goals": "non_goals",
        "assumption": "assumptions",
        "assumptions": "assumptions",
        "success": "success_criteria",
        "success criterion": "success_criteria",
        "success criteria": "success_criteria",
        "unknown": "unknowns",
        "unknowns": "unknowns",
    }
    active_section: str | None = None

    for line in lines[1:]:
        lower = line.lower()
        match = re.match(r"^([a-z\- ]+):\s*(.*)$", lower)
        if match:
            label = match.group(1).strip()
            body = line.split(":", 1)[1].strip()
            mapped = section_aliases.get(label)
            if mapped:
                active_section = mapped
                if body:
                    sections[mapped].append(body)
                continue

        if line.startswith(("-", "*")) and active_section:
            item = line[1:].strip()
            if item:
                sections[active_section].append(item)
            continue

        if active_section:
            sections[active_section].append(line)

    return {
        "objective": objective,
        "constraints": sections["constraints"],
        "non_goals": sections["non_goals"],
        "assumptions": sections["assumptions"],
        "success_criteria": sections["success_criteria"],
        "unknowns": sections["unknowns"],
    }


def build_context_snapshot(
    mode: str = "none",
    include_files: list[str] | None = None,
    max_file_chars: int = 2500,
) -> str:
    """Build optional repository context to reduce planning hallucinations."""
    if mode != "auto":
        return ""

    cwd = Path.cwd()
    candidates: list[Path] = []
    for name in ("README.md", "pyproject.toml", "TODO.md"):
        path = cwd / name
        if path.exists() and path.is_file():
            candidates.append(path)

    for path_str in include_files or []:
        path = Path(path_str)
        if not path.is_absolute():
            path = cwd / path
        if path.exists() and path.is_file():
            candidates.append(path)

    unique_candidates = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(path)

    chunks = []
    status_output = ""
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            status_output = proc.stdout.strip()
    except Exception:
        status_output = ""

    if status_output:
        chunks.append("## Git status\n" + status_output)

    for path in unique_candidates[:8]:
        try:
            text = path.read_text(encoding="utf-8")[:max_file_chars].strip()
        except Exception:
            continue
        if text:
            chunks.append(f"## File: {path.name}\n{text}")

    return "\n\n".join(chunks)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from a model response."""
    payload = text.strip()
    if not payload:
        raise ValueError("Empty response")

    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)

    try:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        return parsed
    except json.JSONDecodeError:
        pass

    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Response does not contain a JSON object")

    parsed = json.loads(payload[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif item is not None:
                out.append(str(item).strip())
        return [item for item in out if item]
    return [str(value).strip()]


def _detect_dependency_cycle(graph: dict[str, list[str]]) -> bool:
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        node_state = state.get(node, 0)
        if node_state == 1:
            return True
        if node_state == 2:
            return False

        state[node] = 1
        for dep in graph.get(node, []):
            if dep in graph and visit(dep):
                return True
        state[node] = 2
        return False

    return any(visit(node) for node in graph)


def _is_measurable_success_criterion(item: str) -> bool:
    lower = item.lower()
    measurable_tokens = [
        "%",
        "ms",
        "seconds",
        "minutes",
        "p95",
        "p99",
        "coverage",
        "latency",
        "error rate",
        "<",
        ">",
        "<=",
        ">=",
    ]
    return any(token in lower for token in measurable_tokens) or any(ch.isdigit() for ch in lower)


def validate_plan_document(plan: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate and normalize a plan document against required schema."""
    errors: list[str] = []
    warnings: list[str] = []
    out: dict[str, Any] = {}

    for key in ["summary", "objective"]:
        value = plan.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"Missing required non-empty string field: {key}")
            out[key] = ""
        else:
            out[key] = value.strip()

    list_fields = [
        "constraints",
        "non_goals",
        "assumptions",
        "unknowns",
        "blocking_questions",
        "architecture_decisions",
        "test_strategy",
        "success_criteria",
        "rollout",
        "rollback",
        "observability",
    ]
    for key in list_fields:
        out[key] = _string_list(plan.get(key))

    if not out["test_strategy"]:
        errors.append("test_strategy must contain at least one entry")
    if not out["success_criteria"]:
        errors.append("success_criteria must contain at least one entry")
    if not out["rollback"]:
        errors.append("rollback must contain at least one step")

    for criterion in out["success_criteria"]:
        if not _is_measurable_success_criterion(criterion):
            warnings.append(f"Success criterion may not be measurable: '{criterion}'")

    tasks_raw = plan.get("tasks")
    out_tasks: list[dict[str, Any]] = []
    if not isinstance(tasks_raw, list) or not tasks_raw:
        errors.append("tasks must be a non-empty list")
    else:
        for idx, task in enumerate(tasks_raw, start=1):
            if not isinstance(task, dict):
                errors.append(f"Task #{idx} must be an object")
                continue

            task_id = str(task.get("id", f"T{idx}")).strip() or f"T{idx}"
            title = str(task.get("title", "")).strip()
            description = str(task.get("description", "")).strip()
            depends_on = _string_list(task.get("depends_on"))
            verification = _string_list(task.get("verification"))

            if not title:
                errors.append(f"Task {task_id} missing title")
            if not description:
                warnings.append(f"Task {task_id} missing description")
            if not verification:
                warnings.append(f"Task {task_id} missing verification checks")

            out_tasks.append(
                {
                    "id": task_id,
                    "title": title,
                    "description": description,
                    "depends_on": depends_on,
                    "verification": verification,
                }
            )

    out["tasks"] = out_tasks

    risks_raw = plan.get("risks")
    out_risks: list[dict[str, str]] = []
    if isinstance(risks_raw, list):
        for idx, risk in enumerate(risks_raw, start=1):
            if isinstance(risk, str):
                out_risks.append(
                    {
                        "id": f"R{idx}",
                        "description": risk,
                        "impact": "medium",
                        "likelihood": "medium",
                        "mitigation": "Define mitigation",
                        "detection": "Define early detection signal",
                    }
                )
                continue
            if not isinstance(risk, dict):
                errors.append(f"Risk #{idx} must be object or string")
                continue

            out_risks.append(
                {
                    "id": str(risk.get("id", f"R{idx}")).strip() or f"R{idx}",
                    "description": str(risk.get("description", "")).strip(),
                    "impact": str(risk.get("impact", "medium")).strip() or "medium",
                    "likelihood": str(risk.get("likelihood", "medium")).strip() or "medium",
                    "mitigation": str(risk.get("mitigation", "")).strip(),
                    "detection": str(risk.get("detection", "")).strip(),
                }
            )
    else:
        errors.append("risks must be a list")

    if not out_risks:
        errors.append("risks must contain at least one risk")
    out["risks"] = out_risks

    task_ids = [task["id"] for task in out_tasks]
    if len(task_ids) != len(set(task_ids)):
        errors.append("tasks contain duplicate ids")

    graph = {task["id"]: task["depends_on"] for task in out_tasks}
    known = set(graph.keys())
    for task in out_tasks:
        unknown = [dep for dep in task["depends_on"] if dep not in known]
        if unknown:
            errors.append(f"Task {task['id']} depends on unknown tasks: {', '.join(unknown)}")

    if graph and _detect_dependency_cycle(graph):
        errors.append("tasks contain a dependency cycle")

    task_positions = {task_id: idx for idx, task_id in enumerate(task_ids)}
    for task in out_tasks:
        for dep in task["depends_on"]:
            if dep in task_positions and task_positions[dep] > task_positions[task["id"]]:
                warnings.append(
                    "Task "
                    f"{task['id']} is listed before dependency {dep}; "
                    "ordering may confuse execution"
                )

    confidence_raw = plan.get("confidence", 0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
        warnings.append("confidence is missing or invalid; defaulted to 0.0")
    confidence = max(0.0, min(1.0, confidence))

    penalty = 0.12 * len(errors) + 0.03 * len(warnings)
    computed_confidence = max(0.0, min(1.0, confidence - penalty))
    out["confidence"] = round(confidence, 2)
    out["computed_confidence"] = round(computed_confidence, 2)

    if not out["blocking_questions"] and out["unknowns"]:
        out["blocking_questions"] = [
            f"How should we resolve: {item}" for item in out["unknowns"][:3]
        ]
        warnings.append("blocking_questions inferred from unknowns")

    return out, errors, warnings


def _validate_critique_document(data: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in [
        "critical_issues",
        "high_priority_issues",
        "dependency_gaps",
        "test_gaps",
        "risk_gaps",
        "blocking_questions",
    ]:
        out[key] = _string_list(data.get(key))

    raw_adjustment = data.get("confidence_adjustment", 0.0)
    try:
        adjustment = float(raw_adjustment)
    except (TypeError, ValueError):
        adjustment = 0.0
    out["confidence_adjustment"] = max(-0.5, min(0.5, adjustment))
    return out


def _complete_stage(
    provider: Provider,
    model_id: str,
    prompt: str,
    system_prompt: str,
    stage_name: str,
    use_cache: bool,
    cache_ttl: int,
) -> _StageResult:
    cache_model_name = f"{model_id}:{stage_name}"
    if use_cache:
        cached_text = get_cached_response(
            cache_model_name,
            prompt,
            system_prompt,
            ttl_hours=cache_ttl,
        )
        if cached_text:
            return _StageResult(
                text=cached_text,
                cached=True,
                input_tokens=0,
                output_tokens=0,
                cost=Decimal("0"),
            )

    response = provider.complete(
        prompt=prompt,
        model=model_id,
        system_prompt=system_prompt,
        temperature=0.2,
    )

    cost = response.cost if response.cost is not None else Decimal("0")
    if not isinstance(cost, Decimal):
        cost = Decimal(str(cost))

    log_api_call(
        model=model_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost=float(cost),
        operation="plan",
    )

    if use_cache:
        cache_response(cache_model_name, prompt, response.text, system_prompt)

    return _StageResult(
        text=response.text,
        cached=False,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost=cost,
    )


def _request_json_stage(
    provider: Provider,
    model_id: str,
    prompt: str,
    system_prompt: str,
    stage_name: str,
    use_cache: bool,
    cache_ttl: int,
) -> tuple[dict[str, Any], _StageResult]:
    stage_result = _complete_stage(
        provider,
        model_id,
        prompt,
        system_prompt,
        stage_name,
        use_cache,
        cache_ttl,
    )

    try:
        return extract_json_object(stage_result.text), stage_result
    except Exception as err:
        repair_prompt = (
            "Convert the following model output into valid JSON. "
            "Return ONLY JSON and preserve as much detail as possible.\n\n"
            f"OUTPUT:\n{stage_result.text}\n\nERROR:\n{err}"
        )
        repair_result = _complete_stage(
            provider,
            model_id,
            repair_prompt,
            "You are a strict JSON repair assistant.",
            f"{stage_name}:repair",
            use_cache,
            cache_ttl,
        )
        repaired = extract_json_object(repair_result.text)

        combined = _StageResult(
            text=repair_result.text,
            cached=stage_result.cached and repair_result.cached,
            input_tokens=stage_result.input_tokens + repair_result.input_tokens,
            output_tokens=stage_result.output_tokens + repair_result.output_tokens,
            cost=stage_result.cost + repair_result.cost,
        )
        return repaired, combined


def _plan_draft_system_prompt(depth: str) -> str:
    depth_note = (
        "Deep mode: maximize thoroughness, include additional fallback and rollback safeguards."
        if depth == "deep"
        else "Standard mode: thorough and concise, avoid unnecessary verbosity."
    )
    return (
        "You are a principal software architect creating an implementation plan.\n"
        "Your output MUST be valid JSON object only, with no markdown.\n"
        "The plan must be dependency-safe, test-first, risk-first, and rollback-safe.\n"
        "Always include explicit blocking questions if assumptions are uncertain.\n"
        f"{depth_note}\n"
        f"Required schema example: {json.dumps(_PLAN_SCHEMA)}"
    )


def _plan_critique_system_prompt() -> str:
    return (
        "You are an adversarial planning reviewer.\n"
        "Find contradictions, missing dependencies, weak testing strategy, and risk gaps.\n"
        "Return ONLY valid JSON object with this schema:\n"
        f"{json.dumps(_CRITIQUE_SCHEMA)}"
    )


def _plan_revision_system_prompt(depth: str) -> str:
    depth_note = "deep" if depth == "deep" else "standard"
    return (
        "You are a principal software architect revising a plan based on critique.\n"
        "Fix all critical/high issues and preserve valid details.\n"
        "Return ONLY valid JSON object.\n"
        f"Planning depth: {depth_note}.\n"
        f"Required schema example: {json.dumps(_PLAN_SCHEMA)}"
    )


def _render_plan_markdown(
    goal: str,
    plan: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    critiques: list[dict[str, Any]],
    model_scores: dict[str, float] | None = None,
) -> str:
    lines = [
        "# Implementation Plan",
        "",
        "## Goal",
        goal,
        "",
        "## Summary",
        plan.get("summary", ""),
        "",
        "## Objective",
        plan.get("objective", ""),
        "",
        "## Assumptions",
    ]
    lines.extend([f"- {item}" for item in plan.get("assumptions", [])] or ["- None provided"])

    lines.extend(["", "## Constraints"])
    lines.extend([f"- {item}" for item in plan.get("constraints", [])] or ["- None provided"])

    lines.extend(["", "## Non-Goals"])
    lines.extend([f"- {item}" for item in plan.get("non_goals", [])] or ["- None provided"])

    lines.extend(["", "## Implementation Tasks (Dependency Aware)"])
    for task in plan.get("tasks", []):
        deps = ", ".join(task.get("depends_on", [])) or "none"
        lines.append(
            f"- [{task.get('id', '?')}] {task.get('title', '(missing title)')} (depends on: {deps})"
        )
        if task.get("description"):
            lines.append(f"  - {task['description']}")
        for check in task.get("verification", []):
            lines.append(f"  - verify: {check}")

    lines.extend(["", "## Test Strategy"])
    lines.extend([f"- {item}" for item in plan.get("test_strategy", [])] or ["- None provided"])

    lines.extend(["", "## Success Criteria"])
    lines.extend([f"- {item}" for item in plan.get("success_criteria", [])] or ["- None provided"])

    lines.extend(["", "## Risks"])
    for risk in plan.get("risks", []):
        desc = risk.get("description", "")
        impact = risk.get("impact", "unknown")
        likelihood = risk.get("likelihood", "unknown")
        mitigation = risk.get("mitigation", "")
        detection = risk.get("detection", "")
        lines.append(
            f"- [{risk.get('id', '?')}] {desc} (impact: {impact}, likelihood: {likelihood})"
        )
        if mitigation:
            lines.append(f"  - mitigation: {mitigation}")
        if detection:
            lines.append(f"  - detection: {detection}")

    lines.extend(["", "## Rollout"])
    lines.extend([f"- {item}" for item in plan.get("rollout", [])] or ["- None provided"])

    lines.extend(["", "## Rollback"])
    lines.extend([f"- {item}" for item in plan.get("rollback", [])] or ["- None provided"])

    lines.extend(["", "## Observability"])
    lines.extend([f"- {item}" for item in plan.get("observability", [])] or ["- None provided"])

    lines.extend(["", "## Unknowns"])
    lines.extend([f"- {item}" for item in plan.get("unknowns", [])] or ["- None provided"])

    lines.extend(["", "## Blocking Questions"])
    lines.extend(
        [f"- {item}" for item in plan.get("blocking_questions", [])] or ["- None provided"]
    )

    lines.extend(["", "## Quality Gates"])
    lines.append(f"- model confidence: {plan.get('confidence', 0):.2f}")
    lines.append(f"- computed confidence: {plan.get('computed_confidence', 0):.2f}")
    lines.append(f"- lint errors: {len(errors)}")
    lines.append(f"- lint warnings: {len(warnings)}")

    if errors:
        lines.append("- errors:")
        lines.extend([f"  - {item}" for item in errors])
    if warnings:
        lines.append("- warnings:")
        lines.extend([f"  - {item}" for item in warnings])

    if critiques:
        lines.extend(["", "## Critique Highlights"])
        for idx, critique in enumerate(critiques, start=1):
            lines.append(f"- round {idx}")
            for key in [
                "critical_issues",
                "high_priority_issues",
                "dependency_gaps",
                "test_gaps",
                "risk_gaps",
            ]:
                for item in critique.get(key, [])[:4]:
                    lines.append(f"  - {key}: {item}")

    if model_scores:
        lines.extend(["", "## Multi-Model Confidence"])
        for model, score in sorted(model_scores.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- {model}: {score:.2f}")

    return "\n".join(lines).strip() + "\n"


def _merge_ranked_lists(items_by_model: list[list[str]], max_items: int = 12) -> list[str]:
    counts = Counter()
    first_index: dict[str, int] = {}
    display_text: dict[str, str] = {}
    for model_items in items_by_model:
        for idx, item in enumerate(model_items):
            trimmed = item.strip()
            normalized = re.sub(r"\s+", " ", trimmed.lower())
            if not normalized:
                continue
            counts[normalized] += 1
            if normalized not in first_index:
                first_index[normalized] = idx
                display_text[normalized] = trimmed

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], first_index.get(kv[0], 9999), kv[0]))
    return [display_text[item] for item, _count in ordered[:max_items]]


def _synthesize_plan_runs(runs: list[PlanRun]) -> tuple[dict[str, Any], dict[str, float]]:
    if len(runs) == 1:
        return runs[0].plan, {runs[0].model: runs[0].confidence}

    scores = {
        run.model: run.confidence - (0.12 * len(run.errors)) - (0.03 * len(run.warnings))
        for run in runs
    }
    best_model = max(scores, key=lambda model_name: scores[model_name])
    best_plan = next(run.plan for run in runs if run.model == best_model)
    merged = dict(best_plan)

    for key in [
        "constraints",
        "non_goals",
        "assumptions",
        "unknowns",
        "blocking_questions",
        "architecture_decisions",
        "test_strategy",
        "success_criteria",
        "rollout",
        "rollback",
        "observability",
    ]:
        merged[key] = _merge_ranked_lists([run.plan.get(key, []) for run in runs])

    risk_pool = []
    for run in runs:
        risk_pool.extend([risk.get("description", "") for risk in run.plan.get("risks", [])])
    top_risks = _merge_ranked_lists([risk_pool], max_items=8)
    merged["risks"] = [
        {
            "id": f"R{idx + 1}",
            "description": risk,
            "impact": "medium",
            "likelihood": "medium",
            "mitigation": "Mitigation to be refined during implementation",
            "detection": "Track via tests/monitoring",
        }
        for idx, risk in enumerate(top_risks)
    ]

    merged["confidence"] = round(
        sum(run.plan.get("confidence", 0.0) for run in runs) / len(runs),
        2,
    )
    merged["consensus"] = {
        "best_model": best_model,
        "agreements": [
            item
            for item in _merge_ranked_lists(
                [run.plan.get("success_criteria", []) for run in runs], 10
            )
        ],
    }
    return merged, scores


def _build_draft_prompt(goal: str, goal_packet: dict[str, Any], context_snapshot: str) -> str:
    payload = {
        "raw_goal": goal,
        "inferred_input": goal_packet,
        "context_snapshot": context_snapshot,
    }
    return (
        "Build a complete implementation plan using this input payload.\n"
        "Fill missing assumptions explicitly and add blocking questions when uncertain.\n"
        "Output strictly valid JSON only.\n\n"
        f"INPUT:\n{json.dumps(payload, indent=2)}"
    )


def _build_critique_prompt(current_plan: dict[str, Any]) -> str:
    return (
        "Critique this plan with focus on dependency correctness, "
        "test completeness, risk coverage, "
        "rollback safety, and hidden assumptions.\n"
        "Return JSON only.\n\n"
        f"PLAN:\n{json.dumps(current_plan, indent=2)}"
    )


def _build_revision_prompt(
    goal: str,
    goal_packet: dict[str, Any],
    current_plan: dict[str, Any],
    critique: dict[str, Any],
    context_snapshot: str,
) -> str:
    payload = {
        "raw_goal": goal,
        "inferred_input": goal_packet,
        "context_snapshot": context_snapshot,
        "current_plan": current_plan,
        "critique": critique,
    }
    return (
        "Revise the current plan by fixing all critical/high issues first, "
        "then improve test strategy "
        "and risk mitigation. Keep dependency order valid. Return JSON only.\n\n"
        f"INPUT:\n{json.dumps(payload, indent=2)}"
    )


def _run_single_model_pipeline(
    goal: str,
    model_name: str,
    depth: str,
    rounds: int,
    use_cache: bool,
    cache_ttl: int,
    context_mode: str,
    include_files: list[str] | None,
    strict: bool,
    confidence_threshold: float,
) -> PlanRun:
    provider_name, model_id = normalize_model_name(model_name)
    provider = get_provider(provider_name)

    goal_packet = infer_goal_packet(goal)
    context_snapshot = build_context_snapshot(context_mode, include_files)

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = Decimal("0")
    all_cached = True

    draft_prompt = _build_draft_prompt(goal, goal_packet, context_snapshot)
    draft_doc, draft_stage = _request_json_stage(
        provider,
        model_id,
        draft_prompt,
        _plan_draft_system_prompt(depth),
        "draft",
        use_cache,
        cache_ttl,
    )
    total_input_tokens += draft_stage.input_tokens
    total_output_tokens += draft_stage.output_tokens
    total_cost += draft_stage.cost
    all_cached = all_cached and draft_stage.cached

    current_plan, errors, warnings = validate_plan_document(draft_doc)
    if errors:
        raise PlanningError("Invalid draft plan schema: " + "; ".join(errors))

    critiques: list[dict[str, Any]] = []
    rounds = max(0, rounds)
    for round_index in range(rounds):
        critique_prompt = _build_critique_prompt(current_plan)
        critique_doc, critique_stage = _request_json_stage(
            provider,
            model_id,
            critique_prompt,
            _plan_critique_system_prompt(),
            f"critique:{round_index + 1}",
            use_cache,
            cache_ttl,
        )
        critique = _validate_critique_document(critique_doc)
        critiques.append(critique)

        total_input_tokens += critique_stage.input_tokens
        total_output_tokens += critique_stage.output_tokens
        total_cost += critique_stage.cost
        all_cached = all_cached and critique_stage.cached

        revision_prompt = _build_revision_prompt(
            goal,
            goal_packet,
            current_plan,
            critique,
            context_snapshot,
        )
        revised_doc, revised_stage = _request_json_stage(
            provider,
            model_id,
            revision_prompt,
            _plan_revision_system_prompt(depth),
            f"revise:{round_index + 1}",
            use_cache,
            cache_ttl,
        )

        total_input_tokens += revised_stage.input_tokens
        total_output_tokens += revised_stage.output_tokens
        total_cost += revised_stage.cost
        all_cached = all_cached and revised_stage.cached

        current_plan, errors, warnings = validate_plan_document(revised_doc)
        if errors:
            raise PlanningError(
                f"Invalid revised plan schema in round {round_index + 1}: " + "; ".join(errors)
            )

        adjustment = critique.get("confidence_adjustment", 0.0)
        current_plan["confidence"] = max(
            0.0,
            min(1.0, float(current_plan.get("confidence", 0.0)) + adjustment),
        )
        current_plan["computed_confidence"] = max(
            0.0,
            min(1.0, float(current_plan.get("computed_confidence", 0.0)) + adjustment),
        )

    current_plan, errors, warnings = validate_plan_document(current_plan)
    confidence = float(current_plan.get("computed_confidence", 0.0))
    blockers = current_plan.get("blocking_questions", [])

    if strict and (errors or blockers or confidence < confidence_threshold):
        reasons = []
        if errors:
            reasons.append(f"{len(errors)} lint errors")
        if blockers:
            reasons.append(f"{len(blockers)} blocking questions")
        if confidence < confidence_threshold:
            reasons.append(
                f"confidence {confidence:.2f} below threshold {confidence_threshold:.2f}"
            )
        raise PlanningError("Strict planning gate failed: " + ", ".join(reasons))

    return PlanRun(
        model=model_name,
        plan=current_plan,
        critiques=critiques,
        errors=errors,
        warnings=warnings,
        confidence=confidence,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost=total_cost,
        cached=all_cached,
    )


def generate_plan_output(
    goal: str,
    model: str,
    *,
    depth: str = "standard",
    rounds: int = 2,
    output_format: str = "markdown",
    strict: bool = False,
    context_mode: str = "none",
    include_files: list[str] | None = None,
    use_cache: bool = True,
    cache_ttl: int = 24,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> PlanOutput:
    """Generate a robust plan with schema validation and critique rounds."""
    selected_models = PLAN_MODEL_GROUPS.get(model, [model])

    if len(selected_models) == 1:
        run = _run_single_model_pipeline(
            goal,
            selected_models[0],
            depth,
            rounds,
            use_cache,
            cache_ttl,
            context_mode,
            include_files,
            strict,
            confidence_threshold,
        )
        model_scores = {run.model: run.confidence}
        final_plan = run.plan
        critiques = run.critiques
        errors = run.errors
        warnings = run.warnings
        total_input = run.input_tokens
        total_output = run.output_tokens
        total_cost = run.cost
        all_cached = run.cached
        output_model = run.model
    else:
        runs: list[PlanRun] = []
        failures: dict[str, str] = {}
        lock = threading.Lock()

        def worker(model_name: str) -> None:
            try:
                result = _run_single_model_pipeline(
                    goal,
                    model_name,
                    depth,
                    rounds,
                    use_cache,
                    cache_ttl,
                    context_mode,
                    include_files,
                    strict,
                    confidence_threshold,
                )
            except Exception as err:  # noqa: BLE001
                with lock:
                    failures[model_name] = str(err)
                return

            with lock:
                runs.append(result)

        threads = [
            threading.Thread(target=worker, args=(model_name,)) for model_name in selected_models
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if not runs:
            joined = "; ".join(f"{name}: {msg}" for name, msg in failures.items())
            raise ProviderError(f"All planning models failed: {joined}")

        final_plan, model_scores = _synthesize_plan_runs(runs)
        final_plan, errors, warnings = validate_plan_document(final_plan)
        critiques = (
            [
                {
                    "critical_issues": [f"{name}: {msg}" for name, msg in sorted(failures.items())],
                    "high_priority_issues": [],
                    "dependency_gaps": [],
                    "test_gaps": [],
                    "risk_gaps": [],
                    "blocking_questions": [],
                    "confidence_adjustment": 0,
                }
            ]
            if failures
            else []
        )
        total_input = sum(run.input_tokens for run in runs)
        total_output = sum(run.output_tokens for run in runs)
        total_cost = sum((run.cost for run in runs), Decimal("0"))
        all_cached = all(run.cached for run in runs)
        output_model = "mm"

    if output_format == "json":
        payload = {
            "plan": final_plan,
            "metadata": {
                "errors": errors,
                "warnings": warnings,
                "model_scores": model_scores,
                "confidence": final_plan.get("computed_confidence", 0.0),
                "strict": strict,
                "rounds": rounds,
                "depth": depth,
            },
        }
        text = json.dumps(payload, indent=2)
    else:
        text = _render_plan_markdown(
            goal,
            final_plan,
            errors,
            warnings,
            critiques,
            model_scores=model_scores,
        )

    return PlanOutput(
        text=text,
        model=output_model,
        metadata={
            "errors": errors,
            "warnings": warnings,
            "model_scores": model_scores,
            "confidence": final_plan.get("computed_confidence", 0.0),
            "strict": strict,
            "rounds": rounds,
            "depth": depth,
        },
        input_tokens=total_input,
        output_tokens=total_output,
        cost=total_cost,
        cached=all_cached,
    )
