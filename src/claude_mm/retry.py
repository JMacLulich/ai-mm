#!/usr/bin/env python3
"""
Simple retry logic with exponential backoff.

Provides retry decorators for API calls without external dependencies.
Supports both sync and async functions transparently.
"""

import asyncio
import logging
import random
import re
import time
from functools import wraps
from inspect import iscoroutinefunction, signature

logger = logging.getLogger(__name__)

# Compiled pattern for non-retriable errors.
# Uses word boundaries (\b) for numeric codes to avoid false positives on
# strings like "error in gpt-4001" matching "400", or "port 4010" matching "401".
_NON_RETRIABLE_RE = re.compile(
    r"\b(400|401|403|404)\b"
    r"|api key|authentication|unauthorized",
    re.IGNORECASE,
)


def _make_context_extractor(func):
    """
    Pre-compute a fast context extractor for a decorated function.

    Binds the function's signature once at decoration time to avoid the
    cost of inspect.signature() on every invocation.
    """
    try:
        _sig = signature(func)
    except (ValueError, TypeError):
        _sig = None

    def extract(args, kwargs):
        provider_name = None
        model_name = None

        if args:
            instance = args[0]
            class_name = getattr(getattr(instance, "__class__", None), "__name__", "")
            if class_name.endswith("Provider"):
                provider_name = class_name.removesuffix("Provider").lower()

        if _sig is not None:
            try:
                bound = _sig.bind_partial(*args, **kwargs)
                model_name = bound.arguments.get("model")
            except (TypeError, ValueError):
                model_name = kwargs.get("model")
        else:
            model_name = kwargs.get("model")

        if provider_name and model_name:
            return f"{provider_name}/{model_name}"
        if provider_name:
            return provider_name
        if model_name:
            return str(model_name)
        return None

    return extract


def _should_not_retry(error_msg: str) -> bool:
    """
    Return True if the error is not retriable (auth/config/client errors).

    Uses regex with word boundaries for numeric HTTP status codes to prevent
    false positives (e.g., "4001" matching "400", or "model-gpt-4012" matching "401").

    Non-retriable categories:
    - Auth/config: api key, authentication, unauthorized
    - HTTP client errors: 400 (bad request), 401 (unauthorized), 403 (forbidden),
      404 (not found) — these are caller bugs, not transient server errors
    """
    return bool(_NON_RETRIABLE_RE.search(error_msg))


def retry_with_backoff(
    max_attempts=3, initial_delay=1, max_delay=10, backoff_factor=2, jitter=True
):
    """
    Decorator to retry a function with exponential backoff.

    Works transparently with both sync and async functions.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay between attempts
        jitter: If True, add ±25% random jitter to delays to prevent thundering herd

    Example:
        @retry_with_backoff(max_attempts=3)
        def call_api():
            ...

        @retry_with_backoff(max_attempts=3)
        async def call_api_async():
            ...
    """

    def decorator(func):
        # Pre-compute context extractor once to avoid signature() on every call
        _get_context = _make_context_extractor(func)

        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                delay = initial_delay
                last_exception = None
                context = _get_context(args, kwargs)
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
                            actual_delay = (
                                delay * (1 + random.uniform(-0.25, 0.25)) if jitter else delay
                            )
                            await asyncio.sleep(actual_delay)
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
                context = _get_context(args, kwargs)
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
                            actual_delay = (
                                delay * (1 + random.uniform(-0.25, 0.25)) if jitter else delay
                            )
                            time.sleep(actual_delay)
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
