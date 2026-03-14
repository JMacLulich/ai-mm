#!/usr/bin/env python3
"""
Simple retry logic with exponential backoff.

Provides retry decorators for API calls without external dependencies.
Supports both sync and async functions transparently.
"""

import asyncio
import logging
import time
from functools import wraps
from inspect import iscoroutinefunction, signature

logger = logging.getLogger(__name__)


def _extract_retry_context(func, args, kwargs):
    """Best-effort context string for retry logs (provider/model)."""
    provider_name = None
    model_name = None

    if args:
        instance = args[0]
        class_name = getattr(getattr(instance, "__class__", None), "__name__", "")
        if class_name.endswith("Provider"):
            provider_name = class_name.removesuffix("Provider").lower()

    try:
        bound = signature(func).bind_partial(*args, **kwargs)
        model_name = bound.arguments.get("model")
    except (TypeError, ValueError):
        model_name = kwargs.get("model")

    if provider_name and model_name:
        return f"{provider_name}/{model_name}"
    if provider_name:
        return provider_name
    if model_name:
        return str(model_name)
    return None


def _should_not_retry(error_msg: str) -> bool:
    """
    Return True if the error is not retriable (auth/config errors).

    Checks for specific auth-related phrases only — avoids over-matching on
    "invalid" since it appears in many retriable errors (invalid response, etc).
    """
    return any(
        x in error_msg
        for x in ["api key", "authentication", "unauthorized", "invalid api key", "403"]
    )


def retry_with_backoff(max_attempts=3, initial_delay=1, max_delay=10, backoff_factor=2):
    """
    Decorator to retry a function with exponential backoff.

    Works transparently with both sync and async functions.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay between attempts

    Example:
        @retry_with_backoff(max_attempts=3)
        def call_api():
            ...

        @retry_with_backoff(max_attempts=3)
        async def call_api_async():
            ...
    """

    def decorator(func):
        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                delay = initial_delay
                last_exception = None
                context = _extract_retry_context(func, args, kwargs)
                context_prefix = f"[{context}] " if context else ""

                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        error_msg = str(e).lower()
                        if _should_not_retry(error_msg):
                            raise

                        if attempt < max_attempts:
                            if "429" in error_msg or "rate limit" in error_msg:
                                logger.warning(
                                    "%sRate limited. Retrying in %ss... (attempt %d/%d)",
                                    context_prefix,
                                    delay,
                                    attempt,
                                    max_attempts,
                                )
                            else:
                                logger.warning(
                                    "%sAPI call failed: %s. Retrying in %ss... (attempt %d/%d)",
                                    context_prefix,
                                    e,
                                    delay,
                                    attempt,
                                    max_attempts,
                                )
                            await asyncio.sleep(delay)
                            delay = min(delay * backoff_factor, max_delay)
                        else:
                            logger.error(
                                "%sAPI call failed after %d attempts: %s",
                                context_prefix,
                                max_attempts,
                                e,
                            )

                if last_exception is None:
                    raise RuntimeError("Retry wrapper exhausted attempts without exception")
                raise last_exception

            return async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                delay = initial_delay
                last_exception = None
                context = _extract_retry_context(func, args, kwargs)
                context_prefix = f"[{context}] " if context else ""

                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        error_msg = str(e).lower()
                        if _should_not_retry(error_msg):
                            raise

                        if attempt < max_attempts:
                            if "429" in error_msg or "rate limit" in error_msg:
                                logger.warning(
                                    "%sRate limited. Retrying in %ss... (attempt %d/%d)",
                                    context_prefix,
                                    delay,
                                    attempt,
                                    max_attempts,
                                )
                            else:
                                logger.warning(
                                    "%sAPI call failed: %s. Retrying in %ss... (attempt %d/%d)",
                                    context_prefix,
                                    e,
                                    delay,
                                    attempt,
                                    max_attempts,
                                )
                            time.sleep(delay)
                            delay = min(delay * backoff_factor, max_delay)
                        else:
                            logger.error(
                                "%sAPI call failed after %d attempts: %s",
                                context_prefix,
                                max_attempts,
                                e,
                            )

                if last_exception is None:
                    raise RuntimeError("Retry wrapper exhausted attempts without exception")
                raise last_exception

            return sync_wrapper

    return decorator
