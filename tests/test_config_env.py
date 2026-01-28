"""
Tests for VisionLLMConfig.from_env() environment variable loading.
"""

import os
import pytest
from unittest.mock import patch

from watsonx_vision.vision_llm import VisionLLMConfig, LLMProvider
from watsonx_vision.exceptions import ConfigurationError


class TestConfigFromEnv:
    """Test suite for VisionLLMConfig.from_env() method."""

    def test_from_env_defaults(self):
        """Test from_env with no environment variables returns defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.provider == LLMProvider.WATSONX
        assert config.model_id == "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
        assert config.api_key is None
        assert config.url is None
        assert config.project_id is None
        assert config.max_tokens == 2000
        assert config.temperature == 0.0
        assert config.top_p == 0.1
        assert config.retry_enabled is True
        assert config.retry_max_attempts == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 60.0

    def test_from_env_watsonx_provider(self):
        """Test loading Watsonx configuration from environment."""
        env = {
            "VISION_PROVIDER": "watsonx",
            "WATSONX_API_KEY": "test-api-key",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
            "WATSONX_PROJECT_ID": "test-project-id",
            "VISION_MODEL_ID": "custom-model",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.provider == LLMProvider.WATSONX
        assert config.api_key == "test-api-key"
        assert config.url == "https://us-south.ml.cloud.ibm.com"
        assert config.project_id == "test-project-id"
        assert config.model_id == "custom-model"

    def test_from_env_ollama_provider(self):
        """Test loading Ollama configuration from environment."""
        env = {
            "VISION_PROVIDER": "ollama",
            "OLLAMA_URL": "http://192.168.1.100:11434",
            "VISION_MODEL_ID": "llava:latest",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.provider == LLMProvider.OLLAMA
        assert config.url == "http://192.168.1.100:11434"
        assert config.model_id == "llava:latest"

    def test_from_env_ollama_default_url(self):
        """Test Ollama provider gets default localhost URL."""
        env = {"VISION_PROVIDER": "ollama"}
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.provider == LLMProvider.OLLAMA
        assert config.url == "http://localhost:11434"

    def test_from_env_ollama_host_fallback(self):
        """Test OLLAMA_HOST as fallback for OLLAMA_URL."""
        env = {
            "VISION_PROVIDER": "ollama",
            "OLLAMA_HOST": "http://ollama-server:11434",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.url == "http://ollama-server:11434"

    def test_from_env_watsonx_apikey_fallback(self):
        """Test WATSONX_APIKEY as fallback for WATSONX_API_KEY."""
        env = {"WATSONX_APIKEY": "fallback-key"}
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.api_key == "fallback-key"

    def test_from_env_generation_params(self):
        """Test loading generation parameters from environment."""
        env = {
            "VISION_MAX_TOKENS": "4000",
            "VISION_TEMPERATURE": "0.7",
            "VISION_TOP_P": "0.9",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.max_tokens == 4000
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_from_env_retry_config(self):
        """Test loading retry configuration from environment."""
        env = {
            "VISION_RETRY_ENABLED": "false",
            "VISION_RETRY_MAX_ATTEMPTS": "5",
            "VISION_RETRY_BASE_DELAY": "2.5",
            "VISION_RETRY_MAX_DELAY": "120.0",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.retry_enabled is False
        assert config.retry_max_attempts == 5
        assert config.retry_base_delay == 2.5
        assert config.retry_max_delay == 120.0

    def test_from_env_retry_enabled_variations(self):
        """Test various boolean values for retry_enabled."""
        true_values = ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]
        false_values = ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"]

        for val in true_values:
            with patch.dict(os.environ, {"VISION_RETRY_ENABLED": val}, clear=True):
                config = VisionLLMConfig.from_env()
                assert config.retry_enabled is True, f"Failed for value: {val}"

        for val in false_values:
            with patch.dict(os.environ, {"VISION_RETRY_ENABLED": val}, clear=True):
                config = VisionLLMConfig.from_env()
                assert config.retry_enabled is False, f"Failed for value: {val}"

    def test_from_env_invalid_provider(self):
        """Test invalid provider raises ConfigurationError."""
        env = {"VISION_PROVIDER": "invalid_provider"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                VisionLLMConfig.from_env()

        assert "Invalid provider" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)

    def test_from_env_with_prefix(self):
        """Test loading configuration with custom prefix."""
        env = {
            "PROD_VISION_PROVIDER": "watsonx",
            "PROD_WATSONX_API_KEY": "prod-api-key",
            "PROD_WATSONX_PROJECT_ID": "prod-project",
            "PROD_VISION_MAX_TOKENS": "3000",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env(prefix="PROD_")

        assert config.provider == LLMProvider.WATSONX
        assert config.api_key == "prod-api-key"
        assert config.project_id == "prod-project"
        assert config.max_tokens == 3000

    def test_from_env_prefix_fallback_to_unprefixed(self):
        """Test prefix falls back to unprefixed variables."""
        env = {
            "WATSONX_API_KEY": "global-api-key",
            "PROD_WATSONX_PROJECT_ID": "prod-project",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env(prefix="PROD_")

        # Should use prefixed project_id but fall back to global api_key
        assert config.api_key == "global-api-key"
        assert config.project_id == "prod-project"

    def test_from_env_with_overrides(self):
        """Test explicit overrides take precedence over env vars."""
        env = {
            "WATSONX_API_KEY": "env-api-key",
            "VISION_TEMPERATURE": "0.5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env(
                api_key="override-api-key",
                temperature=0.9,
            )

        assert config.api_key == "override-api-key"
        assert config.temperature == 0.9

    def test_from_env_provider_override(self):
        """Test provider can be overridden."""
        env = {"VISION_PROVIDER": "watsonx"}
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env(provider=LLMProvider.OLLAMA)

        assert config.provider == LLMProvider.OLLAMA

    def test_from_env_invalid_int_uses_default(self):
        """Test invalid integer values use defaults with warning."""
        env = {"VISION_MAX_TOKENS": "not-a-number"}
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.max_tokens == 2000  # Default value

    def test_from_env_invalid_float_uses_default(self):
        """Test invalid float values use defaults with warning."""
        env = {"VISION_TEMPERATURE": "invalid"}
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.temperature == 0.0  # Default value

    def test_from_env_watsonx_model_fallback(self):
        """Test WATSONX_MODEL as fallback for VISION_MODEL_ID."""
        env = {"WATSONX_MODEL": "ibm/granite-vision"}
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.model_id == "ibm/granite-vision"

    def test_from_env_case_insensitive_provider(self):
        """Test provider value is case-insensitive."""
        for provider_str in ["WATSONX", "Watsonx", "watsonx", "OLLAMA", "Ollama"]:
            env = {"VISION_PROVIDER": provider_str}
            with patch.dict(os.environ, env, clear=True):
                config = VisionLLMConfig.from_env()
                # Should not raise and provider should be set correctly
                assert config.provider.value == provider_str.lower()

    def test_from_env_openai_provider(self):
        """Test OpenAI provider configuration."""
        env = {
            "VISION_PROVIDER": "openai",
            "VISION_LLM_URL": "https://api.openai.com/v1",
            "VISION_MODEL_ID": "gpt-4-vision-preview",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.provider == LLMProvider.OPENAI
        assert config.url == "https://api.openai.com/v1"
        assert config.model_id == "gpt-4-vision-preview"

    def test_from_env_anthropic_provider(self):
        """Test Anthropic provider configuration."""
        env = {
            "VISION_PROVIDER": "anthropic",
            "VISION_MODEL_ID": "claude-3-opus-20240229",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model_id == "claude-3-opus-20240229"

    def test_from_env_combined_scenario(self):
        """Test realistic combined configuration scenario."""
        env = {
            "VISION_PROVIDER": "watsonx",
            "WATSONX_API_KEY": "my-secret-key",
            "WATSONX_URL": "https://eu-de.ml.cloud.ibm.com",
            "WATSONX_PROJECT_ID": "project-12345",
            "VISION_MODEL_ID": "meta-llama/llama-3-2-vision",
            "VISION_MAX_TOKENS": "4096",
            "VISION_TEMPERATURE": "0.1",
            "VISION_TOP_P": "0.95",
            "VISION_RETRY_ENABLED": "true",
            "VISION_RETRY_MAX_ATTEMPTS": "5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VisionLLMConfig.from_env()

        assert config.provider == LLMProvider.WATSONX
        assert config.api_key == "my-secret-key"
        assert config.url == "https://eu-de.ml.cloud.ibm.com"
        assert config.project_id == "project-12345"
        assert config.model_id == "meta-llama/llama-3-2-vision"
        assert config.max_tokens == 4096
        assert config.temperature == 0.1
        assert config.top_p == 0.95
        assert config.retry_enabled is True
        assert config.retry_max_attempts == 5
