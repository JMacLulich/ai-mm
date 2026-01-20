"""Unit tests for retry module."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from claude_mm.retry import retry_with_backoff


def test_retry_with_backoff_success():
    """Test retry decorator with successful function."""
    mock_func = Mock(return_value="success")
    decorated = retry_with_backoff(max_attempts=3)(mock_func)

    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 1  # No retries needed


def test_retry_with_backoff_eventual_success():
    """Test retry decorator with eventual success."""
    mock_func = Mock(side_effect=[
        Exception("503 Service Unavailable"),  # Fail 1
        Exception("503 Service Unavailable"),  # Fail 2
        "success"  # Success on 3rd attempt
    ])
    decorated = retry_with_backoff(max_attempts=3, initial_delay=0.01)(mock_func)

    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_with_backoff_max_attempts():
    """Test retry decorator exhausts max attempts."""
    mock_func = Mock(side_effect=Exception("503 Service Unavailable"))
    decorated = retry_with_backoff(max_attempts=3, initial_delay=0.01)(mock_func)

    with pytest.raises(Exception, match="503 Service Unavailable"):
        decorated()

    assert mock_func.call_count == 3


def test_retry_with_backoff_no_retry_on_auth_error():
    """Test retry decorator doesn't retry auth errors."""
    mock_func = Mock(side_effect=Exception("401 Unauthorized"))
    decorated = retry_with_backoff(max_attempts=3)(mock_func)

    with pytest.raises(Exception, match="401 Unauthorized"):
        decorated()

    assert mock_func.call_count == 1  # No retries
