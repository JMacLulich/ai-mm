#!/usr/bin/env python3
"""
Simple retry logic with exponential backoff.

Provides retry decorators for API calls without external dependencies.
"""

import sys
import time
from functools import wraps
from inspect import signature


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


def retry_with_backoff(max_attempts=3, initial_delay=1, max_delay=10, backoff_factor=2):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay between attempts

    Example:
        @retry_with_backoff(max_attempts=3)
        def call_api():
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            context = _extract_retry_context(func, args, kwargs)
            context_prefix = f"[{context}] " if context else ""

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Don't retry on certain errors
                    error_msg = str(e).lower()
                    if any(
                        x in error_msg
                        for x in ["api key", "authentication", "unauthorized", "invalid"]
                    ):
                        # Authentication/config errors - don't retry
                        raise

                    if attempt < max_attempts:
                        # Check if it's a rate limit error
                        if "429" in error_msg or "rate limit" in error_msg:
                            print(
                                f"⏸️  {context_prefix}Rate limited. Retrying in {delay}s... "
                                f"(attempt {attempt}/{max_attempts})",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                f"⚠️  {context_prefix}API call failed: {e}. Retrying in {delay}s... "
                                f"(attempt {attempt}/{max_attempts})",
                                file=sys.stderr,
                            )

                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        print(
                            f"❌ {context_prefix}API call failed after {max_attempts} "
                            f"attempts: {e}",
                            file=sys.stderr,
                        )

            # If we've exhausted all retries, raise the last exception
            if last_exception is None:
                raise RuntimeError("Retry wrapper exhausted attempts without exception")
            raise last_exception

        return wrapper

    return decorator
