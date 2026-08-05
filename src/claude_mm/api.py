"""
Public Python API for AI tooling.

This module provides a clean, programmatic interface for code review, planning,
and stabilization operations. It can be used from Python scripts, Jupyter notebooks,
or editor integrations (Emacs, VS Code, etc.).

Example usage:
    from api import review, plan, stabilize

    # Single routed review
    result = review("git diff output", model="stage:review")

    # Multi-model review
    results = review("git diff output", models=["stage:review", "deepseek"])

    # Planning
    plan_result = plan("Add user authentication", model="stage:planning")

    # Stabilization (multi-round)
    stable_plan = stabilize("Add caching layer", rounds=2)
"""

import asyncio
import atexit
import concurrent.futures
import logging
import re
import threading
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Union

from claude_mm.cache import cache_response, get_cached_response
from claude_mm.config import load_config
from claude_mm.models import get_model_display_name, normalize_model_name
from claude_mm.planning import DEFAULT_CONFIDENCE_THRESHOLD, generate_plan_output
from claude_mm.prompts import get_review_system_prompt
from claude_mm.providers import get_provider
from claude_mm.providers.base import ProviderResponse
from claude_mm.usage import log_api_call

logger = logging.getLogger(__name__)

VALID_FOCUS_VALUES = frozenset(
    {"general", "review", "verification", "security", "performance", "architecture", "testing"}
)
REASONING_EFFORT_VALUES = frozenset(
    {
        "none",
        "minimal",
        "low",
        "fast",
        "medium",
        "standard",
        "high",
        "careful",
        "xhigh",
        "max",
    }
)


_MAX_ERROR_MSG_LEN = 200  # Truncate error messages to prevent API key leakage in logs

# Regex to redact common API key patterns before logging
_API_KEY_PATTERN = re.compile(
    r"\b(sk-[A-Za-z0-9]{10,}|Bearer\s+[A-Za-z0-9._-]{10,}|api[_-]?key[=:]\s*\S{8,})",
    re.IGNORECASE,
)
_MAX_WORKERS = 32  # Cap on concurrent provider threads

# Module-level thread pool to avoid per-call executor allocation overhead.
# max_workers capped to prevent resource exhaustion from large model lists.
_REVIEW_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the module-level thread pool, creating it lazily and safely."""
    global _REVIEW_EXECUTOR
    with _EXECUTOR_LOCK:
        if _REVIEW_EXECUTOR is None:
            _REVIEW_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=_MAX_WORKERS, thread_name_prefix="mm_review"
            )
        return _REVIEW_EXECUTOR


def _shutdown_executor() -> None:
    """Shut down the module-level thread pool (called by atexit)."""
    global _REVIEW_EXECUTOR
    with _EXECUTOR_LOCK:
        if _REVIEW_EXECUTOR is not None:
            _REVIEW_EXECUTOR.shutdown(wait=False)
            _REVIEW_EXECUTOR = None


atexit.register(_shutdown_executor)


def _safe_err(exc: "Union[BaseException, str]") -> str:
    """Return a log-safe representation of an exception or string.

    Redacts common API key patterns (sk-..., Bearer ...) before truncating
    to prevent secrets from leaking into logs even when they appear early in the message.
    Accepts both BaseException and plain strings for ergonomic use at both
    call sites (exception handlers and dict comprehensions).
    """
    msg = str(exc)
    msg = _API_KEY_PATTERN.sub("[REDACTED]", msg)
    if len(msg) > _MAX_ERROR_MSG_LEN:
        return msg[:_MAX_ERROR_MSG_LEN] + "..."
    return msg


class AllModelsFailedError(RuntimeError):
    """Raised when all review models fail with no successful result.

    Attributes:
        errors: Map of model name → error string for each failed model.
    """

    def __init__(self, errors: Dict[str, str]):
        # Redact and truncate each error string to prevent API key leakage.
        # _safe_err handles both regex-based redaction and length capping.
        # (Input values may already be sanitized, but we apply it defensively here too.)
        self.errors = {m: _safe_err(e) for m, e in errors.items()}
        summary = "; ".join(f"{m}: {e}" for m, e in self.errors.items())
        super().__init__(f"All routed reviews failed. Check llm-router health. Errors: {summary}")

__all__ = [
    "review",
    "review_async",
    "plan",
    "stabilize",
    "ReviewResult",
    "MultiReviewResult",
    "AllModelsFailedError",
    "VALID_FOCUS_VALUES",
]


class ReviewResult:
    """Result from a review operation."""

    def __init__(self, response: ProviderResponse, cached: bool = False):
        self.text = response.text
        self.model = response.model
        self.input_tokens = response.input_tokens
        self.output_tokens = response.output_tokens
        self.cost = response.cost
        self.cached = cached
        self.metadata = response.metadata or {}

    def __str__(self):
        return self.text

    def __repr__(self):
        return (
            f"ReviewResult(model={self.model!r}, cached={self.cached}, "
            f"tokens={self.input_tokens}+{self.output_tokens}, cost={self.cost})"
        )


class MultiReviewResult:
    """Result from a multi-model review.

    Attributes:
        results: Map of model name → ReviewResult for all successful models.
        errors: Map of model name → error string for failed/timed-out models.
        fallback_models: Backward-compatible field that remains empty. Fallbacks are
            internal llm-router attempts and appear in result provenance.
        total_cost: Sum of costs across all successful results.
    """

    def __init__(
        self,
        results: Dict[str, "ReviewResult"],
        errors: Optional[Dict[str, str]] = None,
        fallback_models: Optional[set] = None,
    ):
        self.results = results
        # Redact and truncate error messages to prevent API key leakage
        self.errors = {
            m: _safe_err(e) for m, e in (errors or {}).items()
        }
        self.fallback_models: set = fallback_models or set()
        # Coerce to Decimal to handle providers that return float costs
        self.total_cost = sum(
            (Decimal(str(r.cost)) for r in results.values() if r.cost is not None), Decimal("0")
        )

    def __getitem__(self, model: str) -> "ReviewResult":
        return self.results[model]

    def __iter__(self):
        return iter(self.results.items())

    def __repr__(self):
        models = list(self.results.keys())
        return (
            f"MultiReviewResult(models={models!r}, errors={list(self.errors.keys())!r}, "
            f"fallbacks={list(self.fallback_models)!r}, total_cost={self.total_cost})"
        )


def _resolve_cache_ttl(cache_ttl: Optional[int], config: dict) -> int:
    """Resolve effective cache TTL from caller arg and config, with type coercion."""
    raw = cache_ttl if cache_ttl is not None else config.get("cache_ttl_hours", 24)
    try:
        value = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"cache_ttl must be an integer number of hours, got {raw!r}") from e
    if value < 0:
        raise ValueError(f"cache_ttl must be >= 0, got {value}")
    return value


MAX_PROMPT_CHARS = 1_000_000  # ~250k tokens — guard against accidental huge diffs


def _log_profile_selection(alias: str, provider_name: str, model_id: str) -> None:
    """Emit a visible audit line naming the profile/model a review is about to use.

    This is the answer to "which profile am I actually running?" — logged BEFORE
    the call so a hang/timeout is still attributable. INFO on the ``claude_mm``
    logger, which the CLI surfaces on stderr.
    """
    logger.info(
        "🧭 router → selector=%s boundary=%s intent=%s (%s)",
        alias,
        provider_name,
        model_id,
        get_model_display_name(model_id),
    )


def _audit_served_model(requested_route: str, response: "ProviderResponse") -> None:
    """Log verified router provenance for an intent-resolved response."""
    metadata = response.metadata or {}
    if not metadata.get("router_verified"):
        logger.warning("⚠️  unverified routing response for %s", requested_route)
        return
    logger.info(
        "✅ router audit: route=%s profile=%s provider=%s model=%s fallback=%s",
        requested_route,
        metadata.get("profile", "unknown"),
        metadata.get("provider", "unknown"),
        metadata.get("served_model", response.model),
        metadata.get("fallback_outcome", "unknown"),
    )


def _validate_review_args(
    model: Optional[str],
    models: Optional[List[str]],
    focus: str,
    per_model_timeout: Optional[float],
    prompt: str = "",
    local_model_timeout: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> None:
    """Validate shared review() / review_async() arguments, raising ValueError on bad input."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"prompt is too large ({len(prompt):,} chars). Maximum is {MAX_PROMPT_CHARS:,}."
        )
    if model is not None and models is not None:
        raise ValueError("Pass either 'model' or 'models', not both")
    if models is not None and not models:
        raise ValueError("'models' must not be empty")
    if focus not in VALID_FOCUS_VALUES:
        raise ValueError(f"Invalid focus '{focus}'. Must be one of: {sorted(VALID_FOCUS_VALUES)}")
    if per_model_timeout is not None and per_model_timeout < 0:
        raise ValueError("per_model_timeout must be >= 0")
    if local_model_timeout is not None and local_model_timeout < 0:
        raise ValueError("local_model_timeout must be >= 0")
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORT_VALUES:
        raise ValueError(
            f"Invalid reasoning_effort '{reasoning_effort}'. "
            f"Must be one of: {sorted(REASONING_EFFORT_VALUES)}"
        )


def _cache_variant(reasoning_effort: Optional[str]) -> Optional[str]:
    """Keep provider request options from colliding in the response cache."""
    return f"reasoning_effort={reasoning_effort}" if reasoning_effort else None


def _review_single_with_options(
    prompt: str,
    model: str,
    system_prompt: str,
    use_cache: bool,
    cache_ttl: int,
    reasoning_effort: Optional[str] = None,
) -> "ReviewResult":
    """Compatibility wrapper that only passes new options when requested."""
    if reasoning_effort is None:
        return _review_single(prompt, model, system_prompt, use_cache, cache_ttl)
    return _review_single(
        prompt,
        model,
        system_prompt,
        use_cache,
        cache_ttl,
        reasoning_effort=reasoning_effort,
    )


def review(
    prompt: str,
    model: Optional[str] = None,
    models: Optional[List[str]] = None,
    focus: str = "general",
    use_cache: bool = True,
    cache_ttl: Optional[int] = None,
    on_result: Optional[Callable[[str, "ReviewResult", float], None]] = None,
    per_model_timeout: Optional[float] = None,
    local_model_timeout: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> Union["ReviewResult", "MultiReviewResult"]:
    """
    Perform code review with one or more AI models.

    Args:
        prompt: Code or diff to review
        model: Single router selector (for example ``stage:review`` or ``deepseek``)
        models: Multiple router selectors to orchestrate in parallel.
        focus: Review focus ('general', 'review', 'security', 'performance', 'architecture',
            'testing', 'verification')
        use_cache: Whether to use cached responses
        cache_ttl: Cache TTL in hours (overrides default; 0 = expire immediately / no cache read)
        on_result: Callback invoked as each model completes (model_name, result, duration_secs).
            Only available in the multi-model sync path.
        per_model_timeout: Client deadline for one complete router cascade.
        local_model_timeout: Deprecated compatibility argument; routing and attempt
            timeouts are owned by llm-router.
        reasoning_effort: Provider-neutral quality hint forwarded to llm-router.

    Returns:
        ReviewResult for single model, MultiReviewResult for multiple models

    Raises:
        ValueError: If both model and models are provided, models is empty, focus is invalid,
            per_model_timeout is negative, or local_model_timeout is negative

    Examples:
        >>> result = review("git diff output", model="gpt")
        >>> print(result.text)

        >>> results = review("git diff output", models=["stage:review", "deepseek"])
        >>> for model, result in results:
        ...     print(f"{model}: {result.text}")
    """
    _validate_review_args(
        model,
        models,
        focus,
        per_model_timeout,
        prompt,
        local_model_timeout,
        reasoning_effort,
    )

    config = load_config()
    effective_cache_ttl = _resolve_cache_ttl(cache_ttl, config)

    if models is not None:
        model_list = list(dict.fromkeys(models))  # deduplicate preserving order
    elif model is not None:
        model_list = [model]
    else:
        model_list = [str(config.get("default_models", {}).get("review", "stage:review"))]

    system_prompt = get_review_system_prompt(focus)

    # Resolve remote model timeout
    effective_timeout = per_model_timeout
    if effective_timeout is None:
        configured_timeout = config.get("review_per_model_timeout_seconds", 600)
        try:
            effective_timeout = float(configured_timeout)
        except (TypeError, ValueError):
            effective_timeout = 600.0

    if len(model_list) == 1:
        if on_result is not None:
            logger.warning(
                "on_result callback is ignored for single-model reviews; "
                "use models=[...] for multi-model review"
            )
        applied_timeout = effective_timeout
        if applied_timeout is not None and applied_timeout > 0:
            executor = _get_executor()
            f = executor.submit(
                _review_single_with_options,
                prompt,
                model_list[0],
                system_prompt,
                use_cache,
                effective_cache_ttl,
                reasoning_effort,
            )
            try:
                return f.result(timeout=applied_timeout)
            except concurrent.futures.TimeoutError:
                f.cancel()
                raise AllModelsFailedError(
                    {model_list[0]: f"timed out after {applied_timeout:.1f}s"}
                )
        return _review_single_with_options(
            prompt,
            model_list[0],
            system_prompt,
            use_cache,
            effective_cache_ttl,
            reasoning_effort,
        )

    return _review_multi(
        prompt,
        model_list,
        system_prompt,
        use_cache,
        effective_cache_ttl,
        on_result,
        effective_timeout,
        reasoning_effort,
    )


def _review_single(
    prompt: str,
    model: str,
    system_prompt: str,
    use_cache: bool,
    cache_ttl: int,
    reasoning_effort: Optional[str] = None,
) -> "ReviewResult":
    """Internal: Single model review with cache check and side effects.

    When cache_ttl=0, both cache reads and writes are skipped (as documented).
    """
    provider_name, model_id = normalize_model_name(model)
    # cache_ttl=0 means "skip caching entirely for this call" (per documented contract)
    effective_use_cache = use_cache and cache_ttl > 0

    cache_variant = _cache_variant(reasoning_effort)
    if effective_use_cache:
        cached = get_cached_response(
            model_id,
            prompt,
            system_prompt,
            ttl_hours=cache_ttl,
            cache_variant=cache_variant,
        )
        if cached is not None:
            return ReviewResult(
                ProviderResponse(
                    text=cached,
                    model=model_id,
                    input_tokens=0,
                    output_tokens=0,
                    cost=Decimal("0"),
                    cached=True,
                ),
                cached=True,
            )

    _log_profile_selection(model, provider_name, model_id)
    provider = get_provider(provider_name)
    provider_kwargs = {}
    if reasoning_effort is not None:
        provider_kwargs["reasoning_effort"] = reasoning_effort
    response = provider.complete(
        prompt,
        model_id,
        system_prompt=system_prompt,
        **provider_kwargs,
    )
    _audit_served_model(model_id, response)

    # Non-critical side effects: log failure should not prevent returning the result
    try:
        log_api_call(
            model=model_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=float(response.cost) if response.cost is not None else 0.0,
            operation="review",
        )
    except Exception as e:
        logger.warning("Failed to log API call for %s: %s", model_id, _safe_err(e))

    if effective_use_cache:
        cache_response(
            model_id,
            prompt,
            response.text,
            system_prompt,
            cache_variant=cache_variant,
        )

    return ReviewResult(response, cached=False)


def _review_multi(
    prompt: str,
    models: List[str],
    system_prompt: str,
    use_cache: bool,
    cache_ttl: int,
    on_result: Optional[Callable[[str, "ReviewResult", float], None]] = None,
    per_model_timeout: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> "MultiReviewResult":
    """
    Internal: Multi-model review using ThreadPoolExecutor with per-model deadlines.

    Each task has one client deadline for the router's complete cascade. Attempt
    timeouts and fallback order remain inside llm-router.
    """
    results: Dict[str, ReviewResult] = {}
    errors: Dict[str, str] = {}

    def _effective_timeout_for(model_name: str) -> Optional[float]:
        """Return the client deadline for one routed task."""
        del model_name
        return (
            per_model_timeout
            if per_model_timeout is not None and per_model_timeout > 0
            else None
        )

    executor = _get_executor()
    # Key by future object directly (not id()) to avoid id-reuse bugs
    start_times: Dict[concurrent.futures.Future, float] = {}
    future_to_model: Dict[concurrent.futures.Future, str] = {}

    # Initialize pending before try so the finally block can safely iterate it
    pending: set = set()
    # Wrap future submission in try/finally so futures are always cancelled on error
    try:
        now = time.perf_counter()
        for m in models:
            logger.info("Starting review with %s...", m)
            future = executor.submit(
                _review_single_with_options,
                prompt,
                m,
                system_prompt,
                use_cache,
                cache_ttl,
                reasoning_effort,
            )
            start_times[future] = now
            future_to_model[future] = m

        # Per-route deadlines bound the complete cascade, not individual providers.
        model_deadline: Dict[str, Optional[float]] = {
            m: ((now + t) if (t := _effective_timeout_for(m)) is not None else None)
            for m in models
        }
        model_timeout_val: Dict[str, Optional[float]] = {
            m: _effective_timeout_for(m) for m in models
        }

        if any(v is not None for v in model_timeout_val.values()):
            logger.info("⏱️  Router task timeout: %.0fs", per_model_timeout or 0)

        # Collect results with event-driven waiting (no busy-poll)
        pending = set(future_to_model.keys())

        while pending:
            # Compute time until the next individual deadline among pending models
            deadlines = [
                model_deadline[future_to_model[f]]
                for f in pending
                if model_deadline.get(future_to_model[f]) is not None
            ]
            next_deadline = min(deadlines) if deadlines else None
            now = time.perf_counter()

            # If any model's deadline has already passed, handle timeouts immediately
            if next_deadline is not None and now >= next_deadline:
                for future in list(pending):
                    m = future_to_model[future]
                    d = model_deadline.get(m)
                    if d is not None and time.perf_counter() >= d:
                        future.cancel()
                        pending.discard(future)
                        t = model_timeout_val.get(m) or 0
                        errors[m] = f"timed out after {t:.1f}s"
                        logger.warning("Timed out %s after %.1fs; continuing", m, t)
                continue  # re-compute next_deadline

            remaining = max(0.0, next_deadline - now) if next_deadline is not None else None

            done, pending = concurrent.futures.wait(
                pending,
                timeout=remaining,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                model_name = future_to_model[future]
                duration = time.perf_counter() - start_times[future]
                try:
                    result = future.result()
                    results[model_name] = result
                    source = "cached" if result.cached else "live"
                    logger.info("Completed %s in %.2fs (%s)", model_name, duration, source)
                except Exception as e:
                    errors[model_name] = _safe_err(e)
                    logger.warning(
                        "Error reviewing with %s: %s (after %.2fs)",
                        model_name, _safe_err(e), duration
                    )
                    continue

                # on_result callback runs outside the result-recording try/except
                # so callback failures don't corrupt the result dict
                if on_result:
                    try:
                        on_result(model_name, result, duration)
                    except Exception as e:
                        logger.warning(
                            "on_result callback raised for %s: %s", model_name, _safe_err(e)
                        )

            # After wait, check per-model timeouts for still-pending futures
            now = time.perf_counter()
            for future in list(pending):
                m = future_to_model[future]
                d = model_deadline.get(m)
                if d is not None and now >= d:
                    future.cancel()
                    pending.discard(future)
                    t = model_timeout_val.get(m) or 0
                    errors[m] = f"timed out after {t:.1f}s"
                    logger.warning("Timed out %s after %.1fs; continuing", m, t)

            # If wait returned with no completions and we had a deadline, loop to re-check
            # (per-model check above will catch any newly expired models)

    finally:
        # Cancel all tracked futures to cover both:
        # 1. Mid-submission crash: pending is empty but future_to_model has submitted futures
        # 2. Timeout path: pending has unfinished futures
        # Cancelling an already-completed future is a safe no-op.
        for future in future_to_model:
            future.cancel()

    if not results:
        raise AllModelsFailedError(errors)

    return MultiReviewResult(results, errors=errors)


def plan(
    goal: str,
    model: Optional[str] = None,
    use_cache: bool = True,
    cache_ttl: Optional[int] = None,
    depth: str = "standard",
    rounds: int = 2,
    output_format: str = "markdown",
    strict: bool = False,
    context_mode: str = "none",
    include_files: Optional[List[str]] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> "ReviewResult":
    """
    Generate an implementation plan.

    Args:
        goal: What to plan (e.g., "Add user authentication")
        model: Model to use (defaults to configured plan model)
        use_cache: Whether to use cached responses
        cache_ttl: Cache TTL in hours (overrides default; 0 = expire immediately / no cache read)
        depth: Planning depth preset ('standard' or 'deep')
        rounds: Number of critique/revision rounds
        output_format: Output format ('markdown' or 'json')
        strict: If True, fail when blocking questions or low confidence exist
        context_mode: Context mode ('none' or 'auto')
        include_files: Optional additional files to include in planning context
        confidence_threshold: Minimum confidence when strict mode is enabled

    Returns:
        ReviewResult with the plan

    Example:
        >>> plan_result = plan("Add caching layer to API")
        >>> print(plan_result.text)
    """
    if not goal or not goal.strip():
        raise ValueError("goal must not be empty")
    if len(goal) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"goal is too large ({len(goal):,} chars). Maximum is {MAX_PROMPT_CHARS:,}."
        )
    if depth not in {"standard", "deep"}:
        raise ValueError(f"Invalid depth '{depth}'. Must be 'standard' or 'deep'")
    if output_format not in {"markdown", "json"}:
        raise ValueError(f"Invalid output_format '{output_format}'. Must be 'markdown' or 'json'")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if context_mode not in {"none", "auto"}:
        raise ValueError(f"Invalid context_mode '{context_mode}'. Must be 'none' or 'auto'")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
        )

    config = load_config()
    effective_cache_ttl = _resolve_cache_ttl(cache_ttl, config)

    selected_model = model or config.get("default_models", {}).get("plan", "stage:planning")
    selected_model = str(selected_model)

    plan_output = generate_plan_output(
        goal,
        selected_model,
        depth=depth,
        rounds=rounds,
        output_format=output_format,
        strict=strict,
        context_mode=context_mode,
        include_files=include_files,
        use_cache=use_cache,
        cache_ttl=effective_cache_ttl,
        confidence_threshold=confidence_threshold,
    )

    return ReviewResult(
        ProviderResponse(
            text=plan_output.text,
            model=plan_output.model,
            input_tokens=plan_output.input_tokens,
            output_tokens=plan_output.output_tokens,
            cost=plan_output.cost,
            cached=plan_output.cached,
            metadata=plan_output.metadata,
        ),
        cached=plan_output.cached,
    )


def stabilize(
    goal: str,
    rounds: int = 2,
    mode: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Multi-round plan stabilization with critique and revision.

    Args:
        goal: What to plan
        rounds: Number of critique/revision rounds requested
        mode: Optional mode ('migrations', 'docs', 'infra')
        use_cache: Whether to use cached responses

    Returns:
        Dictionary with:
            - final_plan: The stabilized plan (ReviewResult)
            - rounds_requested: Number of critique/revision rounds requested
            - mode: Optional mode used ('migrations', 'docs', 'infra', or None)
            - total_cost: Total cost in USD
            - metadata: Additional metadata from the plan

    Example:
        >>> result = stabilize("Add user authentication", rounds=2)
        >>> print(result['final_plan'].text)
        >>> print(f"Total cost: ${result['total_cost']:.4f}")
    """
    if not goal or not goal.strip():
        raise ValueError("goal must not be empty")
    if len(goal) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"goal is too large ({len(goal):,} chars). Maximum is {MAX_PROMPT_CHARS:,}."
        )
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    valid_modes = {"migrations", "docs", "infra", None}
    if mode not in valid_modes:
        valid_list = sorted(m for m in valid_modes if m)
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_list}")

    context_mode = "auto" if mode in {"migrations", "docs", "infra"} else "none"
    plan_result = plan(
        goal=goal,
        model=None,
        use_cache=use_cache,
        cache_ttl=None,
        depth="deep",
        rounds=rounds,
        output_format="markdown",
        strict=False,
        context_mode=context_mode,
        include_files=None,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )

    return {
        "final_plan": plan_result,
        "rounds_requested": rounds,
        "mode": mode,
        "total_cost": float(plan_result.cost or Decimal("0")),
        "metadata": plan_result.metadata,
    }


# Async API variants


async def _review_single_async_with_options(
    prompt: str,
    model: str,
    system_prompt: str,
    use_cache: bool,
    cache_ttl: int,
    reasoning_effort: Optional[str] = None,
) -> "ReviewResult":
    """Compatibility wrapper that only passes new options when requested."""
    if reasoning_effort is None:
        return await _review_single_async(prompt, model, system_prompt, use_cache, cache_ttl)
    return await _review_single_async(
        prompt,
        model,
        system_prompt,
        use_cache,
        cache_ttl,
        reasoning_effort=reasoning_effort,
    )


async def review_async(
    prompt: str,
    model: Optional[str] = None,
    models: Optional[List[str]] = None,
    focus: str = "general",
    use_cache: bool = True,
    cache_ttl: Optional[int] = None,
    per_model_timeout: Optional[float] = None,
    local_model_timeout: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> Union["ReviewResult", "MultiReviewResult"]:
    """
    Async version of review(). See review() for full documentation.

    Provider retries and fallbacks are executed by llm-router, exactly as in the
    synchronous path.

    Note: on_result callback is not supported in the async path. To stream results
    as each model completes, call review_async() multiple times concurrently with
    asyncio.gather or asyncio.as_completed.

    Raises:
        ValueError: If both model and models are provided, models is empty, focus is invalid,
            per_model_timeout is negative, or local_model_timeout is negative
    """
    _validate_review_args(
        model,
        models,
        focus,
        per_model_timeout,
        prompt,
        local_model_timeout,
        reasoning_effort,
    )

    config = load_config()
    effective_cache_ttl = _resolve_cache_ttl(cache_ttl, config)

    if models is not None:
        model_list = list(dict.fromkeys(models))  # deduplicate preserving order
    elif model is not None:
        model_list = [model]
    else:
        model_list = [str(config.get("default_models", {}).get("review", "stage:review"))]

    system_prompt = get_review_system_prompt(focus)

    # Resolve the client deadline for a complete router cascade.
    effective_timeout = per_model_timeout
    if effective_timeout is None:
        configured_timeout = config.get("review_per_model_timeout_seconds", 600)
        try:
            effective_timeout = float(configured_timeout)
        except (TypeError, ValueError):
            effective_timeout = 600.0

    if len(model_list) == 1:
        coro = _review_single_async_with_options(
            prompt,
            model_list[0],
            system_prompt,
            use_cache,
            effective_cache_ttl,
            reasoning_effort,
        )
        applied_timeout = effective_timeout
        if applied_timeout is not None and applied_timeout > 0:
            try:
                return await asyncio.wait_for(coro, timeout=applied_timeout)
            except asyncio.TimeoutError:
                raise AllModelsFailedError(
                    {model_list[0]: f"timed out after {applied_timeout:.1f}s"}
                )
        return await coro

    async def run_model(model_name: str):
        start = time.perf_counter()
        model_t = effective_timeout
        timeout_for_model = model_t if (model_t is not None and model_t > 0) else 0.0
        try:
            coro = _review_single_async_with_options(
                prompt,
                model_name,
                system_prompt,
                use_cache,
                effective_cache_ttl,
                reasoning_effort,
            )
            if timeout_for_model > 0:
                result = await asyncio.wait_for(coro, timeout=timeout_for_model)
            else:
                result = await coro

            duration = time.perf_counter() - start
            return model_name, result, duration, None
        except asyncio.TimeoutError:
            duration = time.perf_counter() - start
            return model_name, None, duration, f"timed out after {timeout_for_model:.1f}s"
        except Exception as exc:
            duration = time.perf_counter() - start
            return model_name, None, duration, _safe_err(exc)

    tasks = [run_model(m) for m in model_list]
    # run_model handles all exceptions internally and returns them as tuple values;
    # asyncio.gather's default behavior (propagate BaseException) is appropriate here.
    results_list = await asyncio.gather(*tasks)

    errors: Dict[str, str] = {}
    results: Dict[str, ReviewResult] = {}
    for model_name, result, duration, error in results_list:
        if error:
            logger.warning(
                "Error reviewing with %s: %s (after %.2fs)", model_name, error, duration
            )
            errors[model_name] = error  # already _safe_err'd in run_model
        elif result is not None:
            source = "cached" if result.cached else "live"
            logger.info("Completed %s in %.2fs (%s)", model_name, duration, source)
            results[model_name] = result
        else:
            errors[model_name] = "unknown error"

    if not results:
        raise AllModelsFailedError(errors)

    return MultiReviewResult(results, errors=errors)


async def _review_single_async(
    prompt: str,
    model: str,
    system_prompt: str,
    use_cache: bool,
    cache_ttl: int,
    reasoning_effort: Optional[str] = None,
) -> "ReviewResult":
    """Internal: Async single model review. All blocking I/O is offloaded to a thread pool.

    cache_ttl=0 skips both cache reads and writes (per documented contract).
    """
    provider_name, model_id = normalize_model_name(model)
    effective_use_cache = use_cache and cache_ttl > 0

    cache_variant = _cache_variant(reasoning_effort)
    if effective_use_cache:
        cached = await asyncio.to_thread(
            get_cached_response,
            model_id,
            prompt,
            system_prompt,
            ttl_hours=cache_ttl,
            cache_variant=cache_variant,
        )
        if cached is not None:
            return ReviewResult(
                ProviderResponse(
                    text=cached,
                    model=model_id,
                    input_tokens=0,
                    output_tokens=0,
                    cost=Decimal("0"),
                    cached=True,
                ),
                cached=True,
            )

    _log_profile_selection(model, provider_name, model_id)
    provider = get_provider(provider_name)
    provider_kwargs = {}
    if reasoning_effort is not None:
        provider_kwargs["reasoning_effort"] = reasoning_effort
    response = await provider.complete_async(
        prompt,
        model_id,
        system_prompt=system_prompt,
        **provider_kwargs,
    )
    _audit_served_model(model_id, response)

    # Non-critical side effect: log failure should not prevent returning the result
    try:
        await asyncio.to_thread(
            log_api_call,
            model=model_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=float(response.cost) if response.cost is not None else 0.0,
            operation="review",
        )
    except Exception as e:
        logger.warning("Failed to log API call for %s: %s", model_id, _safe_err(e))

    if effective_use_cache:
        await asyncio.to_thread(
            cache_response,
            model_id,
            prompt,
            response.text,
            system_prompt,
            cache_variant=cache_variant,
        )

    return ReviewResult(response, cached=False)
