"""
Retry Utilities for Watsonx Vision Toolkit

Provides retry logic with exponential backoff for handling transient LLM failures.
"""

import logging
import random
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type, TypeVar, Union

from .exceptions import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
    WatsonxVisionError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default exceptions that should trigger a retry
DEFAULT_RETRY_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    LLMConnectionError,
    LLMTimeoutError,
    ConnectionError,
    TimeoutError,
)


class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (including initial attempt)
        base_delay: Base delay in seconds between retries
        max_delay: Maximum delay in seconds (caps exponential growth)
        exponential_base: Base for exponential backoff (delay = base_delay * exponential_base^attempt)
        jitter: Whether to add random jitter to delays
        retry_exceptions: Tuple of exception types that should trigger retry
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions or DEFAULT_RETRY_EXCEPTIONS

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% random jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        config: RetryConfig instance (uses DEFAULT_RETRY_CONFIG if None)
        on_retry: Optional callback called on each retry with (exception, attempt_number)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(RetryConfig(max_attempts=3))
        ... def call_llm():
        ...     return llm.invoke(messages)

        >>> # With custom retry callback
        >>> def log_retry(exc, attempt):
        ...     print(f"Retry {attempt}: {exc}")
        >>> @retry_with_backoff(on_retry=log_retry)
        ... def call_llm():
        ...     return llm.invoke(messages)
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retry_exceptions as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed. Last error: {e}"
                        )

            # Re-raise the last exception if all retries failed
            if last_exception:
                raise last_exception

            # Should never reach here, but satisfy type checker
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator


def retry_llm_call(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    **kwargs,
) -> T:
    """
    Execute a function with retry logic (non-decorator version).

    Useful when you can't use a decorator or need dynamic retry behavior.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        config: RetryConfig instance
        on_retry: Optional callback on retry
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        The last exception if all retries fail

    Example:
        >>> result = retry_llm_call(
        ...     llm.invoke,
        ...     messages,
        ...     config=RetryConfig(max_attempts=5)
        ... )
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            return func(*args, **kwargs)
        except config.retry_exceptions as e:
            last_exception = e

            if attempt < config.max_attempts - 1:
                delay = config.calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if on_retry:
                    on_retry(e, attempt + 1)

                time.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_attempts} attempts failed. Last error: {e}"
                )

    if last_exception:
        raise last_exception

    raise RuntimeError("Unexpected retry loop exit")
