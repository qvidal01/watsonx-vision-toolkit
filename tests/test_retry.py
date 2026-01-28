"""Tests for retry utilities"""

import time
import pytest
from unittest.mock import Mock, patch

from watsonx_vision.retry import (
    RetryConfig,
    retry_with_backoff,
    retry_llm_call,
    DEFAULT_RETRY_CONFIG,
    DEFAULT_RETRY_EXCEPTIONS,
)
from watsonx_vision.exceptions import (
    LLMConnectionError,
    LLMTimeoutError,
)


class TestRetryConfig:
    """Tests for RetryConfig"""

    def test_default_config(self):
        """Test default retry configuration"""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config(self):
        """Test custom retry configuration"""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation"""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.calculate_delay(0) == 1.0   # 1 * 2^0 = 1
        assert config.calculate_delay(1) == 2.0   # 1 * 2^1 = 2
        assert config.calculate_delay(2) == 4.0   # 1 * 2^2 = 4
        assert config.calculate_delay(3) == 8.0   # 1 * 2^3 = 8

    def test_calculate_delay_max_cap(self):
        """Test that delay is capped at max_delay"""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=False
        )

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 5.0  # Capped at max_delay
        assert config.calculate_delay(10) == 5.0  # Still capped

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness to delay"""
        config = RetryConfig(base_delay=1.0, jitter=True)

        # With jitter, delay should be between base and base + 25%
        delays = [config.calculate_delay(0) for _ in range(10)]

        # All delays should be >= base delay
        assert all(d >= 1.0 for d in delays)
        # All delays should be <= base delay + 25%
        assert all(d <= 1.25 for d in delays)
        # Not all delays should be exactly the same (jitter adds randomness)
        assert len(set(delays)) > 1

    def test_custom_retry_exceptions(self):
        """Test custom retry exceptions"""
        config = RetryConfig(retry_exceptions=(ValueError, TypeError))
        assert config.retry_exceptions == (ValueError, TypeError)


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator"""

    def test_success_no_retry(self):
        """Test successful call without retry"""
        mock_func = Mock(return_value="success")

        @retry_with_backoff(RetryConfig(max_attempts=3))
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_connection_error(self):
        """Test retry on connection error"""
        mock_func = Mock(side_effect=[
            LLMConnectionError("Connection failed"),
            LLMConnectionError("Connection failed"),
            "success"
        ])

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        @retry_with_backoff(config)
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_on_timeout_error(self):
        """Test retry on timeout error"""
        mock_func = Mock(side_effect=[
            LLMTimeoutError("Timeout"),
            "success"
        ])

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        @retry_with_backoff(config)
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 2

    def test_all_retries_fail(self):
        """Test that exception is raised when all retries fail"""
        mock_func = Mock(side_effect=LLMConnectionError("Connection failed"))

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        @retry_with_backoff(config)
        def test_func():
            return mock_func()

        with pytest.raises(LLMConnectionError):
            test_func()

        assert mock_func.call_count == 3

    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are raised immediately"""
        mock_func = Mock(side_effect=ValueError("Not retryable"))

        config = RetryConfig(max_attempts=3, base_delay=0.01)

        @retry_with_backoff(config)
        def test_func():
            return mock_func()

        with pytest.raises(ValueError):
            test_func()

        # Should only be called once - no retry for ValueError
        assert mock_func.call_count == 1

    def test_on_retry_callback(self):
        """Test on_retry callback is called"""
        mock_func = Mock(side_effect=[
            LLMConnectionError("Fail 1"),
            LLMConnectionError("Fail 2"),
            "success"
        ])
        callback = Mock()

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        @retry_with_backoff(config, on_retry=callback)
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert callback.call_count == 2
        # Verify callback was called with correct attempt numbers
        calls = callback.call_args_list
        assert calls[0][0][1] == 1  # First retry, attempt 1
        assert calls[1][0][1] == 2  # Second retry, attempt 2
        # Verify exception types
        assert isinstance(calls[0][0][0], LLMConnectionError)
        assert isinstance(calls[1][0][0], LLMConnectionError)

    def test_uses_default_config(self):
        """Test that default config is used when none provided"""
        mock_func = Mock(return_value="success")

        @retry_with_backoff()
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "success"


class TestRetryLLMCall:
    """Tests for retry_llm_call function"""

    def test_success_no_retry(self):
        """Test successful call without retry"""
        mock_func = Mock(return_value="success")

        result = retry_llm_call(
            mock_func,
            config=RetryConfig(max_attempts=3, base_delay=0.01)
        )

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_with_args(self):
        """Test retry with positional arguments"""
        mock_func = Mock(side_effect=[
            LLMConnectionError("Fail"),
            "success"
        ])

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        result = retry_llm_call(mock_func, "arg1", "arg2", config=config)

        assert result == "success"
        assert mock_func.call_count == 2
        mock_func.assert_called_with("arg1", "arg2")

    def test_retry_with_kwargs(self):
        """Test retry with keyword arguments"""
        mock_func = Mock(side_effect=[
            LLMConnectionError("Fail"),
            "success"
        ])

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        result = retry_llm_call(mock_func, key1="val1", config=config, key2="val2")

        assert result == "success"
        mock_func.assert_called_with(key1="val1", key2="val2")

    def test_all_retries_fail(self):
        """Test exception raised when all retries fail"""
        mock_func = Mock(side_effect=LLMTimeoutError("Timeout"))

        config = RetryConfig(max_attempts=2, base_delay=0.01, jitter=False)

        with pytest.raises(LLMTimeoutError):
            retry_llm_call(mock_func, config=config)

        assert mock_func.call_count == 2

    def test_uses_default_config(self):
        """Test default config when none provided"""
        mock_func = Mock(return_value="success")

        result = retry_llm_call(mock_func)

        assert result == "success"


class TestDefaultRetryExceptions:
    """Tests for default retry exceptions"""

    def test_default_exceptions_include_expected_types(self):
        """Test default exceptions include expected types"""
        assert LLMConnectionError in DEFAULT_RETRY_EXCEPTIONS
        assert LLMTimeoutError in DEFAULT_RETRY_EXCEPTIONS
        assert ConnectionError in DEFAULT_RETRY_EXCEPTIONS
        assert TimeoutError in DEFAULT_RETRY_EXCEPTIONS

    def test_default_config_exists(self):
        """Test DEFAULT_RETRY_CONFIG is properly configured"""
        assert DEFAULT_RETRY_CONFIG.max_attempts == 3
        assert DEFAULT_RETRY_CONFIG.base_delay == 1.0
