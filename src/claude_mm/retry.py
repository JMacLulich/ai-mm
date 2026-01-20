#!/usr/bin/env python3
"""
Simple retry logic with exponential backoff.

Provides retry decorators for API calls without external dependencies.
"""

import sys
import time
from functools import wraps


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

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Don't retry on certain errors
                    error_msg = str(e).lower()
                    if any(x in error_msg for x in ['api key', 'authentication', 'unauthorized', 'invalid']):
                        # Authentication/config errors - don't retry
                        raise

                    if attempt < max_attempts:
                        # Check if it's a rate limit error
                        if '429' in error_msg or 'rate limit' in error_msg:
                            print(f"⏸️  Rate limited. Retrying in {delay}s... (attempt {attempt}/{max_attempts})", file=sys.stderr)
                        else:
                            print(f"⚠️  API call failed: {e}. Retrying in {delay}s... (attempt {attempt}/{max_attempts})", file=sys.stderr)

                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        print(f"❌ API call failed after {max_attempts} attempts", file=sys.stderr)

            # If we've exhausted all retries, raise the last exception
            raise last_exception

        return wrapper
    return decorator
