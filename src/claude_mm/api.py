"""
Public Python API for AI tooling.

This module provides a clean, programmatic interface for code review, planning,
and stabilization operations. It can be used from Python scripts, Jupyter notebooks,
or editor integrations (Emacs, VS Code, etc.).

Example usage:
    from api import review, plan, stabilize

    # Single model review
    result = review("git diff output", model="gpt")

    # Multi-model review
    results = review("git diff output", models=["gpt", "gemini"])

    # Planning
    plan_result = plan("Add user authentication", model="gpt-5.2")

    # Stabilization (multi-round)
    stable_plan = stabilize("Add caching layer", rounds=2)
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from claude_mm.cache import cache_response, get_cached_response
from claude_mm.config import load_config
from claude_mm.models import normalize_model_name
from claude_mm.planning import DEFAULT_CONFIDENCE_THRESHOLD, generate_plan_output
from claude_mm.providers import get_provider
from claude_mm.providers.base import ProviderResponse
from claude_mm.usage import log_api_call


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


class MultiReviewResult:
    """Result from a multi-model review."""

    def __init__(self, results: Dict[str, ReviewResult]):
        self.results = results
        self.total_cost = sum(r.cost for r in results.values() if r.cost)

    def __getitem__(self, model: str) -> ReviewResult:
        return self.results[model]

    def __iter__(self):
        return iter(self.results.items())


def review(
    prompt: str,
    model: Optional[str] = None,
    models: Optional[List[str]] = None,
    focus: str = "general",
    use_cache: bool = True,
    cache_ttl: Optional[int] = None,
) -> Union[ReviewResult, MultiReviewResult]:
    """
    Perform code review with one or more AI models.

    Args:
        prompt: Code or diff to review
        model: Single model to use (e.g., 'gpt', 'gemini', 'claude')
        models: Multiple models to use (for parallel review)
        focus: Review focus ('general', 'security', 'performance', 'architecture')
        use_cache: Whether to use cached responses
        cache_ttl: Cache TTL in hours (overrides default)

    Returns:
        ReviewResult for single model, MultiReviewResult for multiple models

    Examples:
        >>> result = review("git diff output", model="gpt")
        >>> print(result.text)

        >>> results = review("git diff output", models=["gpt", "gemini"])
        >>> for model, result in results:
        ...     print(f"{model}: {result.text}")
    """
    config = load_config()

    # Determine which models to use
    if models:
        model_list = models
    elif model:
        model_list = [model]
    else:
        # Default to configured review model
        model_list = [config.get("default_models", {}).get("review", "gpt-5.2-chat-latest")]

    # Build system prompt based on focus
    system_prompts = {
        "general": "You are an expert code reviewer. Provide thorough, actionable feedback.",
        "security": (
            "You are a security expert. Focus on security vulnerabilities, "
            "input validation, and potential exploits."
        ),
        "performance": (
            "You are a performance expert. Focus on optimization opportunities, "
            "algorithmic efficiency, and resource usage."
        ),
        "architecture": (
            "You are a software architect conducting a code architecture review. "
            "Evaluate the code against these CRITICAL principles:\n\n"
            "**DRY (Don't Repeat Yourself)**: Identify duplicated logic, repeated patterns, "
            "and copy-paste code that should be extracted into reusable functions or modules.\n\n"
            "**Single Responsibility Principle (SRP)**: Every module, class, and function "
            "should have ONE reason to change. Flag god objects, multi-purpose classes, "
            "and functions doing too many things.\n\n"
            "**Principle of Least Astonishment (POLA)**: Code should behave as users expect. "
            "Flag surprising side effects, confusing naming, non-obvious behavior, "
            "and APIs that violate developer expectations.\n\n"
            "Also consider: modularity, separation of concerns, dependency injection, "
            "and long-term maintainability. Provide specific, actionable recommendations."
        ),
    }
    system_prompt = system_prompts.get(focus, system_prompts["general"])

    # Single model review
    if len(model_list) == 1:
        return _review_single(
            prompt,
            model_list[0],
            system_prompt,
            use_cache,
            cache_ttl or config.get("cache_ttl_hours", 24),
        )

    # Multi-model review (parallel)
    return _review_multi(
        prompt,
        model_list,
        system_prompt,
        use_cache,
        cache_ttl or config.get("cache_ttl_hours", 24),
    )


def _review_single(
    prompt: str,
    model: str,
    system_prompt: str,
    use_cache: bool,
    cache_ttl: int,
) -> ReviewResult:
    """Internal: Single model review."""
    # Check cache first
    if use_cache:
        cached = get_cached_response(model, prompt, system_prompt, ttl_hours=cache_ttl)
        if cached:
            return ReviewResult(
                ProviderResponse(
                    text=cached,
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    cost=Decimal("0"),
                    cached=True,
                ),
                cached=True,
            )

    # Get provider and call
    provider_name, model_id = normalize_model_name(model)
    provider = get_provider(provider_name)
    response = provider.complete(prompt, model_id, system_prompt)

    # Log usage
    log_api_call(
        model=model_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost=float(response.cost) if response.cost else 0.0,
        operation="review",
    )

    # Cache response
    if use_cache:
        cache_response(model, prompt, response.text, system_prompt)

    return ReviewResult(response, cached=False)


def _review_multi(
    prompt: str,
    models: List[str],
    system_prompt: str,
    use_cache: bool,
    cache_ttl: int,
) -> MultiReviewResult:
    """Internal: Multi-model review using asyncio."""
    # For now, use threading for backward compatibility
    # TODO: Switch to pure asyncio in future
    import threading

    results = {}
    errors = {}

    def review_thread(model_name):
        try:
            result = _review_single(prompt, model_name, system_prompt, use_cache, cache_ttl)
            results[model_name] = result
        except Exception as e:
            errors[model_name] = str(e)

    threads = [threading.Thread(target=review_thread, args=(m,)) for m in models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        for model, error in errors.items():
            print(f"Error reviewing with {model}: {error}")

    return MultiReviewResult(results)


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
) -> ReviewResult:
    """
    Generate an implementation plan.

    Args:
        goal: What to plan (e.g., "Add user authentication")
        model: Model to use (defaults to configured plan model)
        use_cache: Whether to use cached responses
        cache_ttl: Cache TTL in hours (overrides default)
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
    config = load_config()

    selected_model = model or config.get("default_models", {}).get("plan", "gpt-5.2")
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
        cache_ttl=cache_ttl or config.get("cache_ttl_hours", 24),
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
        rounds: Number of critique/revision rounds
        mode: Optional mode ('migrations', 'docs', 'infra')
        use_cache: Whether to use cached responses

    Returns:
        Dictionary with:
            - final_plan: The stabilized plan
            - rounds: List of round results
            - total_cost: Total cost across all rounds

    Example:
        >>> result = stabilize("Add user authentication", rounds=2)
        >>> print(result['final_plan'].text)
        >>> print(f"Total cost: ${result['total_cost']:.4f}")
    """
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
        "rounds": rounds,
        "mode": mode,
        "total_cost": float(plan_result.cost or Decimal("0")),
        "metadata": plan_result.metadata,
    }


# Async API variants
async def review_async(
    prompt: str,
    model: Optional[str] = None,
    models: Optional[List[str]] = None,
    focus: str = "general",
    use_cache: bool = True,
    cache_ttl: Optional[int] = None,
) -> Union[ReviewResult, MultiReviewResult]:
    """Async version of review(). See review() for documentation."""
    config = load_config()

    # Determine which models to use
    if models:
        model_list = models
    elif model:
        model_list = [model]
    else:
        model_list = [config.get("default_models", {}).get("review", "gpt-5.2-chat-latest")]

    # Build system prompt
    system_prompts = {
        "general": "You are an expert code reviewer. Provide thorough, actionable feedback.",
        "security": "You are a security expert. Focus on security vulnerabilities.",
        "performance": "You are a performance expert. Focus on optimization opportunities.",
        "architecture": (
            "You are a software architect conducting a code architecture review. "
            "Evaluate the code against these CRITICAL principles:\n\n"
            "**DRY (Don't Repeat Yourself)**: Identify duplicated logic, repeated patterns, "
            "and copy-paste code that should be extracted into reusable functions or modules.\n\n"
            "**Single Responsibility Principle (SRP)**: Every module, class, and function "
            "should have ONE reason to change. Flag god objects, multi-purpose classes, "
            "and functions doing too many things.\n\n"
            "**Principle of Least Astonishment (POLA)**: Code should behave as users expect. "
            "Flag surprising side effects, confusing naming, non-obvious behavior, "
            "and APIs that violate developer expectations.\n\n"
            "Also consider: modularity, separation of concerns, dependency injection, "
            "and long-term maintainability. Provide specific, actionable recommendations."
        ),
    }
    system_prompt = system_prompts.get(focus, system_prompts["general"])

    # Single model
    if len(model_list) == 1:
        return await _review_single_async(
            prompt,
            model_list[0],
            system_prompt,
            use_cache,
            cache_ttl or config.get("cache_ttl_hours", 24),
        )

    # Multi-model (parallel with asyncio)
    tasks = [
        _review_single_async(
            prompt,
            m,
            system_prompt,
            use_cache,
            cache_ttl or config.get("cache_ttl_hours", 24),
        )
        for m in model_list
    ]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Build results dict
    results = {}
    for i, model_name in enumerate(model_list):
        if isinstance(results_list[i], Exception):
            print(f"Error reviewing with {model_name}: {results_list[i]}")
        else:
            results[model_name] = results_list[i]

    return MultiReviewResult(results)


async def _review_single_async(
    prompt: str,
    model: str,
    system_prompt: str,
    use_cache: bool,
    cache_ttl: int,
) -> ReviewResult:
    """Internal: Async single model review."""
    # Check cache first (sync operation)
    if use_cache:
        cached = get_cached_response(model, prompt, system_prompt, ttl_hours=cache_ttl)
        if cached:
            return ReviewResult(
                ProviderResponse(
                    text=cached,
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    cost=Decimal("0"),
                    cached=True,
                ),
                cached=True,
            )

    # Get provider and call async
    provider_name, model_id = normalize_model_name(model)
    provider = get_provider(provider_name)
    response = await provider.complete_async(prompt, model_id, system_prompt)

    # Log usage (sync operation)
    log_api_call(
        model=model_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost=float(response.cost) if response.cost else 0.0,
        operation="review",
    )

    # Cache response (sync operation)
    if use_cache:
        cache_response(model, prompt, response.text, system_prompt)

    return ReviewResult(response, cached=False)
