#!/usr/bin/env python3
"""
Retry Configuration Example

Demonstrates how to configure retry behavior for resilient LLM calls.
"""

import logging
import os
import time

from watsonx_vision import (
    VisionLLM,
    VisionLLMConfig,
    LLMProvider,
    RetryConfig,
    retry_with_backoff,
    retry_llm_call,
    LLMConnectionError,
    LLMTimeoutError,
)

# Enable logging to see retry attempts
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_basic_retry():
    """Basic retry configuration with VisionLLM."""
    print("\n" + "="*60)
    print("Example 1: Basic Retry Configuration")
    print("="*60)

    # Retry is enabled by default with sensible defaults
    config = VisionLLMConfig(
        provider=LLMProvider.OLLAMA,
        model_id="llava",
        url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        # Retry settings (these are the defaults)
        retry_enabled=True,
        retry_max_attempts=3,
        retry_base_delay=1.0,
        retry_max_delay=60.0,
    )

    print(f"Retry enabled: {config.retry_enabled}")
    print(f"Max attempts: {config.retry_max_attempts}")
    print(f"Base delay: {config.retry_base_delay}s")
    print(f"Max delay: {config.retry_max_delay}s")

    # llm = VisionLLM(config)
    # The LLM will automatically retry on connection errors


def example_custom_retry_config():
    """Custom retry configuration for high-availability scenarios."""
    print("\n" + "="*60)
    print("Example 2: Custom Retry Configuration")
    print("="*60)

    # More aggressive retry for critical operations
    config = VisionLLMConfig(
        provider=LLMProvider.WATSONX,
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        api_key=os.getenv("WATSONX_API_KEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        # Aggressive retry for production
        retry_enabled=True,
        retry_max_attempts=5,      # More attempts
        retry_base_delay=0.5,      # Start faster
        retry_max_delay=30.0,      # Cap at 30s
    )

    print(f"Max attempts: {config.retry_max_attempts}")
    print(f"Base delay: {config.retry_base_delay}s")
    print(f"Max delay: {config.retry_max_delay}s")

    # Calculate delays for each attempt
    retry_config = RetryConfig(
        max_attempts=config.retry_max_attempts,
        base_delay=config.retry_base_delay,
        max_delay=config.retry_max_delay,
        jitter=False  # Disable jitter to show exact delays
    )

    print("\nExpected delays (without jitter):")
    for attempt in range(config.retry_max_attempts):
        delay = retry_config.calculate_delay(attempt)
        print(f"  Attempt {attempt + 1}: {delay:.2f}s")


def example_disable_retry():
    """Disable retry for fast-fail scenarios."""
    print("\n" + "="*60)
    print("Example 3: Disable Retry")
    print("="*60)

    # Disable retry for interactive/low-latency scenarios
    config = VisionLLMConfig(
        provider=LLMProvider.OLLAMA,
        model_id="llava",
        retry_enabled=False,  # Fail fast
    )

    print(f"Retry enabled: {config.retry_enabled}")
    print("LLM will fail immediately on connection errors")


def example_retry_decorator():
    """Using the retry_with_backoff decorator directly."""
    print("\n" + "="*60)
    print("Example 4: Retry Decorator")
    print("="*60)

    # Custom retry configuration
    config = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        max_delay=5.0,
        exponential_base=2.0,
        jitter=True,
    )

    call_count = 0

    @retry_with_backoff(config)
    def flaky_operation():
        """Simulates a flaky operation that fails twice then succeeds."""
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise LLMConnectionError(f"Simulated failure {call_count}")
        return "Success!"

    try:
        result = flaky_operation()
        print(f"Result: {result}")
        print(f"Total attempts: {call_count}")
    except LLMConnectionError as e:
        print(f"All retries failed: {e}")


def example_retry_with_callback():
    """Using retry with a callback for monitoring."""
    print("\n" + "="*60)
    print("Example 5: Retry with Callback")
    print("="*60)

    def on_retry_callback(exception: Exception, attempt: int):
        """Called on each retry attempt."""
        logger.warning(f"Retry callback: attempt {attempt}, error: {exception}")
        # You could send metrics, alerts, etc. here

    config = RetryConfig(
        max_attempts=4,
        base_delay=0.1,
        jitter=False,
    )

    call_count = 0

    @retry_with_backoff(config, on_retry=on_retry_callback)
    def monitored_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise LLMTimeoutError(f"Timeout on attempt {call_count}")
        return "Completed!"

    result = monitored_operation()
    print(f"Result: {result}")


def example_retry_llm_call_function():
    """Using retry_llm_call for non-decorator usage."""
    print("\n" + "="*60)
    print("Example 6: retry_llm_call Function")
    print("="*60)

    config = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        jitter=False,
    )

    call_count = 0

    def api_call(message: str, temperature: float = 0.7):
        """Simulates an API call."""
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Network error")
        return f"Response to: {message} (temp={temperature})"

    # retry_llm_call allows passing args and kwargs
    result = retry_llm_call(
        api_call,
        "Hello, world!",
        temperature=0.5,
        config=config,
    )

    print(f"Result: {result}")
    print(f"Total attempts: {call_count}")


def example_custom_retry_exceptions():
    """Configure which exceptions trigger retry."""
    print("\n" + "="*60)
    print("Example 7: Custom Retry Exceptions")
    print("="*60)

    # Only retry on specific exceptions
    config = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        retry_exceptions=(
            LLMConnectionError,
            LLMTimeoutError,
            ConnectionError,
            TimeoutError,
            # Add custom exceptions as needed
        ),
    )

    print(f"Retry exceptions: {[e.__name__ for e in config.retry_exceptions]}")

    call_count = 0

    @retry_with_backoff(config)
    def selective_retry():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # This will trigger retry
            raise ConnectionError("Network issue")
        elif call_count == 2:
            # This will NOT trigger retry (ValueError not in list)
            raise ValueError("Invalid input")
        return "Success"

    try:
        result = selective_retry()
        print(f"Result: {result}")
    except ValueError as e:
        print(f"ValueError raised (not retried): {e}")
        print(f"Total attempts: {call_count}")


def example_jitter_comparison():
    """Compare delays with and without jitter."""
    print("\n" + "="*60)
    print("Example 8: Jitter Comparison")
    print("="*60)

    # Without jitter - deterministic delays
    config_no_jitter = RetryConfig(
        base_delay=1.0,
        exponential_base=2.0,
        jitter=False,
    )

    # With jitter - randomized delays
    config_with_jitter = RetryConfig(
        base_delay=1.0,
        exponential_base=2.0,
        jitter=True,
    )

    print("Without jitter (deterministic):")
    for attempt in range(5):
        delay = config_no_jitter.calculate_delay(attempt)
        print(f"  Attempt {attempt}: {delay:.3f}s")

    print("\nWith jitter (randomized, 3 samples each):")
    for attempt in range(5):
        delays = [config_with_jitter.calculate_delay(attempt) for _ in range(3)]
        delays_str = ", ".join(f"{d:.3f}s" for d in delays)
        print(f"  Attempt {attempt}: {delays_str}")


def main():
    """Run all examples."""
    print("Retry Configuration Examples")
    print("="*60)

    example_basic_retry()
    example_custom_retry_config()
    example_disable_retry()
    example_retry_decorator()
    example_retry_with_callback()
    example_retry_llm_call_function()
    example_custom_retry_exceptions()
    example_jitter_comparison()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
