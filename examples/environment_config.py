"""
Environment Variable Configuration Example

This example demonstrates how to configure VisionLLM using environment variables
for production deployments where hardcoding credentials is not desirable.

Environment Variables:
    # Required for Watsonx
    WATSONX_API_KEY=your-api-key
    WATSONX_URL=https://us-south.ml.cloud.ibm.com
    WATSONX_PROJECT_ID=your-project-id

    # Or for Ollama
    VISION_PROVIDER=ollama
    OLLAMA_URL=http://localhost:11434

    # Optional settings
    VISION_MODEL_ID=meta-llama/llama-4-maverick-17b-128e-instruct-fp8
    VISION_MAX_TOKENS=2000
    VISION_TEMPERATURE=0.0
    VISION_RETRY_ENABLED=true
    VISION_RETRY_MAX_ATTEMPTS=3
"""

import os

from watsonx_vision import VisionLLM, VisionLLMConfig
from watsonx_vision.vision_llm import LLMProvider


def basic_env_config():
    """Basic configuration from environment variables."""
    # Load configuration from environment
    # This will read WATSONX_API_KEY, WATSONX_URL, etc.
    config = VisionLLMConfig.from_env()

    print(f"Provider: {config.provider.value}")
    print(f"Model: {config.model_id}")
    print(f"URL: {config.url}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"Retry enabled: {config.retry_enabled}")

    # Create VisionLLM instance
    llm = VisionLLM(config)
    return llm


def prefixed_env_config():
    """
    Use prefixed environment variables for multiple configurations.

    Useful when you need different configs for dev/staging/prod.

    Environment:
        PROD_WATSONX_API_KEY=prod-key
        PROD_WATSONX_PROJECT_ID=prod-project
        DEV_WATSONX_API_KEY=dev-key
        DEV_WATSONX_PROJECT_ID=dev-project
    """
    # Load production configuration
    prod_config = VisionLLMConfig.from_env(prefix="PROD_")

    # Load development configuration
    dev_config = VisionLLMConfig.from_env(prefix="DEV_")

    print("Production config:")
    print(f"  API Key: {prod_config.api_key[:10]}..." if prod_config.api_key else "  API Key: None")
    print(f"  Project: {prod_config.project_id}")

    print("\nDevelopment config:")
    print(f"  API Key: {dev_config.api_key[:10]}..." if dev_config.api_key else "  API Key: None")
    print(f"  Project: {dev_config.project_id}")


def env_with_overrides():
    """
    Load from environment but override specific values.

    This is useful when you want env vars for credentials but
    different model settings per use case.
    """
    # Load base config from env, but override temperature for this use case
    config = VisionLLMConfig.from_env(
        temperature=0.7,  # Higher temperature for creative tasks
        max_tokens=4000,  # More tokens for detailed analysis
    )

    print(f"Temperature: {config.temperature}")  # 0.7, not from env
    print(f"Max tokens: {config.max_tokens}")  # 4000, not from env


def local_development_config():
    """
    Configuration for local development with Ollama.

    Environment:
        VISION_PROVIDER=ollama
        OLLAMA_URL=http://localhost:11434
        VISION_MODEL_ID=llava:latest
    """
    # For local dev, you might set minimal env vars
    os.environ.setdefault("VISION_PROVIDER", "ollama")
    os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
    os.environ.setdefault("VISION_MODEL_ID", "llava:latest")

    config = VisionLLMConfig.from_env()

    print(f"Provider: {config.provider.value}")
    print(f"Model: {config.model_id}")
    print(f"URL: {config.url}")


def docker_compose_config():
    """
    Example configuration for Docker Compose deployment.

    docker-compose.yml:
        services:
          app:
            environment:
              - WATSONX_API_KEY=${WATSONX_API_KEY}
              - WATSONX_URL=https://us-south.ml.cloud.ibm.com
              - WATSONX_PROJECT_ID=${WATSONX_PROJECT_ID}
              - VISION_RETRY_MAX_ATTEMPTS=5
              - VISION_MAX_TOKENS=4096
    """
    config = VisionLLMConfig.from_env()

    # In production, you'd use the config to create VisionLLM
    print("Docker configuration loaded:")
    print(f"  Provider: {config.provider.value}")
    print(f"  Max attempts: {config.retry_max_attempts}")
    print(f"  Max tokens: {config.max_tokens}")


def kubernetes_secrets_config():
    """
    Example for Kubernetes with secrets mounted as environment variables.

    Kubernetes deployment.yaml:
        env:
          - name: WATSONX_API_KEY
            valueFrom:
              secretKeyRef:
                name: watsonx-credentials
                key: api-key
          - name: WATSONX_PROJECT_ID
            valueFrom:
              secretKeyRef:
                name: watsonx-credentials
                key: project-id
    """
    # Kubernetes secrets become environment variables
    config = VisionLLMConfig.from_env()

    # Verify required credentials are present
    if not config.api_key:
        raise ValueError("WATSONX_API_KEY not set - check Kubernetes secrets")
    if not config.project_id:
        raise ValueError("WATSONX_PROJECT_ID not set - check Kubernetes secrets")

    print("Kubernetes configuration loaded successfully")
    return config


def fallback_chain_example():
    """
    Demonstrate the fallback chain for environment variables.

    The from_env() method checks variables in order:
    1. Prefixed variable (if prefix provided)
    2. Standard variable name
    3. Alternative variable name (fallbacks)
    4. Default value
    """
    # WATSONX_APIKEY is a fallback for WATSONX_API_KEY
    os.environ["WATSONX_APIKEY"] = "fallback-key"

    # OLLAMA_HOST is a fallback for OLLAMA_URL
    os.environ["OLLAMA_HOST"] = "http://ollama:11434"

    # WATSONX_MODEL is a fallback for VISION_MODEL_ID
    os.environ["WATSONX_MODEL"] = "ibm/granite-vision"

    config = VisionLLMConfig.from_env()

    print("Fallback chain results:")
    print(f"  API Key: {config.api_key}")  # fallback-key

    # For Ollama provider
    os.environ["VISION_PROVIDER"] = "ollama"
    config = VisionLLMConfig.from_env()
    print(f"  Ollama URL: {config.url}")  # http://ollama:11434

    # Cleanup
    del os.environ["WATSONX_APIKEY"]
    del os.environ["OLLAMA_HOST"]
    del os.environ["WATSONX_MODEL"]
    del os.environ["VISION_PROVIDER"]


if __name__ == "__main__":
    print("=" * 60)
    print("Environment Variable Configuration Examples")
    print("=" * 60)

    print("\n1. Basic Environment Configuration:")
    print("-" * 40)
    try:
        basic_env_config()
    except Exception as e:
        print(f"   (Set WATSONX_* environment variables to test: {e})")

    print("\n2. Local Development Configuration:")
    print("-" * 40)
    local_development_config()

    print("\n3. Environment with Overrides:")
    print("-" * 40)
    env_with_overrides()

    print("\n4. Fallback Chain Example:")
    print("-" * 40)
    fallback_chain_example()

    print("\n" + "=" * 60)
    print("See function docstrings for Docker/Kubernetes examples")
    print("=" * 60)
