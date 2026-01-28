"""Tests for async functionality"""

import asyncio
import sys
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Mock the langchain modules before importing anything that uses them
import json

_mock_human_message = type('HumanMessage', (), {'__init__': lambda self, content: setattr(self, 'content', content)})
_mock_system_message = type('SystemMessage', (), {'__init__': lambda self, content: setattr(self, 'content', content)})

# Create a mock JsonOutputParser that actually parses JSON
class MockJsonOutputParser:
    def parse(self, content):
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        return json.loads(content)

mock_langchain_ibm = MagicMock()
mock_langchain_core = MagicMock()
mock_langchain_core_messages = MagicMock()
mock_langchain_core_output_parsers = MagicMock()

mock_langchain_core_messages.HumanMessage = _mock_human_message
mock_langchain_core_messages.SystemMessage = _mock_system_message
mock_langchain_core_output_parsers.JsonOutputParser = MockJsonOutputParser

sys.modules['langchain_ibm'] = mock_langchain_ibm
sys.modules['langchain_core'] = mock_langchain_core
sys.modules['langchain_core.messages'] = mock_langchain_core_messages
sys.modules['langchain_core.output_parsers'] = mock_langchain_core_output_parsers

from watsonx_vision.retry import (
    RetryConfig,
    async_retry_with_backoff,
    async_retry_llm_call,
)
from watsonx_vision.exceptions import (
    LLMConnectionError,
    LLMTimeoutError,
)


class TestAsyncRetryWithBackoff:
    """Tests for async_retry_with_backoff decorator"""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful async call without retry"""
        mock_func = AsyncMock(return_value="success")

        @async_retry_with_backoff(RetryConfig(max_attempts=3))
        async def test_func():
            return await mock_func()

        result = await test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test async retry on connection error"""
        mock_func = AsyncMock(side_effect=[
            LLMConnectionError("Connection failed"),
            LLMConnectionError("Connection failed"),
            "success"
        ])

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        @async_retry_with_backoff(config)
        async def test_func():
            return await mock_func()

        result = await test_func()

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self):
        """Test async retry on timeout error"""
        mock_func = AsyncMock(side_effect=[
            LLMTimeoutError("Timeout"),
            "success"
        ])

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        @async_retry_with_backoff(config)
        async def test_func():
            return await mock_func()

        result = await test_func()

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        """Test that exception is raised when all async retries fail"""
        mock_func = AsyncMock(side_effect=LLMConnectionError("Connection failed"))

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        @async_retry_with_backoff(config)
        async def test_func():
            return await mock_func()

        with pytest.raises(LLMConnectionError):
            await test_func()

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are raised immediately"""
        mock_func = AsyncMock(side_effect=ValueError("Not retryable"))

        config = RetryConfig(max_attempts=3, base_delay=0.01)

        @async_retry_with_backoff(config)
        async def test_func():
            return await mock_func()

        with pytest.raises(ValueError):
            await test_func()

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback is called in async"""
        mock_func = AsyncMock(side_effect=[
            LLMConnectionError("Fail 1"),
            LLMConnectionError("Fail 2"),
            "success"
        ])
        callback = Mock()

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        @async_retry_with_backoff(config, on_retry=callback)
        async def test_func():
            return await mock_func()

        result = await test_func()

        assert result == "success"
        assert callback.call_count == 2
        calls = callback.call_args_list
        assert calls[0][0][1] == 1  # First retry, attempt 1
        assert calls[1][0][1] == 2  # Second retry, attempt 2

    @pytest.mark.asyncio
    async def test_uses_default_config(self):
        """Test that default config is used when none provided"""
        mock_func = AsyncMock(return_value="success")

        @async_retry_with_backoff()
        async def test_func():
            return await mock_func()

        result = await test_func()
        assert result == "success"


class TestAsyncRetryLLMCall:
    """Tests for async_retry_llm_call function"""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful async call without retry"""
        mock_func = AsyncMock(return_value="success")

        result = await async_retry_llm_call(
            mock_func,
            config=RetryConfig(max_attempts=3, base_delay=0.01)
        )

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_args(self):
        """Test async retry with positional arguments"""
        mock_func = AsyncMock(side_effect=[
            LLMConnectionError("Fail"),
            "success"
        ])

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        result = await async_retry_llm_call(mock_func, "arg1", "arg2", config=config)

        assert result == "success"
        assert mock_func.call_count == 2
        mock_func.assert_called_with("arg1", "arg2")

    @pytest.mark.asyncio
    async def test_retry_with_kwargs(self):
        """Test async retry with keyword arguments"""
        mock_func = AsyncMock(side_effect=[
            LLMConnectionError("Fail"),
            "success"
        ])

        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)

        result = await async_retry_llm_call(
            mock_func, key1="val1", config=config, key2="val2"
        )

        assert result == "success"
        mock_func.assert_called_with(key1="val1", key2="val2")

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        """Test exception raised when all async retries fail"""
        mock_func = AsyncMock(side_effect=LLMTimeoutError("Timeout"))

        config = RetryConfig(max_attempts=2, base_delay=0.01, jitter=False)

        with pytest.raises(LLMTimeoutError):
            await async_retry_llm_call(mock_func, config=config)

        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_default_config(self):
        """Test default config when none provided"""
        mock_func = AsyncMock(return_value="success")

        result = await async_retry_llm_call(mock_func)

        assert result == "success"


class TestVisionLLMAsync:
    """Tests for VisionLLM async methods"""

    @pytest.mark.asyncio
    async def test_analyze_image_async_success(self):
        """Test async analyze_image returns correct result"""
        from watsonx_vision import VisionLLM, VisionLLMConfig
        from watsonx_vision.vision_llm import LLMProvider

        # Create mock LLM with async support
        mock_internal_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"result": "test"}'
        mock_internal_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Configure and create VisionLLM
        mock_langchain_ibm.ChatWatsonx.return_value = mock_internal_llm

        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            api_key="test-key",
            url="https://test.com",
            project_id="test-project",
            retry_enabled=False,
        )
        llm = VisionLLM(config)

        result = await llm.analyze_image_async(
            "data:image/png;base64,abc123",
            "Test prompt"
        )

        assert result == {"result": "test"}
        mock_internal_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_document_async(self):
        """Test async document classification"""
        from watsonx_vision import VisionLLM, VisionLLMConfig
        from watsonx_vision.vision_llm import LLMProvider

        mock_internal_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"doc_type": "Passport"}'
        mock_internal_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_langchain_ibm.ChatWatsonx.return_value = mock_internal_llm

        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            api_key="test-key",
            url="https://test.com",
            project_id="test-project",
            retry_enabled=False,
        )
        llm = VisionLLM(config)

        result = await llm.classify_document_async(
            "data:image/png;base64,abc123"
        )

        assert result["doc_type"] == "Passport"

    @pytest.mark.asyncio
    async def test_extract_information_async(self):
        """Test async information extraction"""
        from watsonx_vision import VisionLLM, VisionLLMConfig
        from watsonx_vision.vision_llm import LLMProvider

        mock_internal_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"name": "John Doe", "dob": "1990-01-15"}'
        mock_internal_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_langchain_ibm.ChatWatsonx.return_value = mock_internal_llm

        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            api_key="test-key",
            url="https://test.com",
            project_id="test-project",
            retry_enabled=False,
        )
        llm = VisionLLM(config)

        result = await llm.extract_information_async(
            "data:image/png;base64,abc123"
        )

        assert result["name"] == "John Doe"
        assert result["dob"] == "1990-01-15"

    @pytest.mark.asyncio
    async def test_validate_authenticity_async(self):
        """Test async authenticity validation"""
        from watsonx_vision import VisionLLM, VisionLLMConfig
        from watsonx_vision.vision_llm import LLMProvider

        mock_internal_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''{"valid": true, "reason": "Document appears authentic",
            "layout_score": 90, "field_score": 85, "forgery_signs": []}'''
        mock_internal_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_langchain_ibm.ChatWatsonx.return_value = mock_internal_llm

        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            api_key="test-key",
            url="https://test.com",
            project_id="test-project",
            retry_enabled=False,
        )
        llm = VisionLLM(config)

        result = await llm.validate_authenticity_async(
            "data:image/png;base64,abc123"
        )

        assert result["valid"] is True
        assert result["layout_score"] == 90


class TestFraudDetectorAsync:
    """Tests for FraudDetector async methods"""

    @pytest.mark.asyncio
    async def test_validate_document_async(self):
        """Test async document validation"""
        from watsonx_vision import VisionLLM, VisionLLMConfig
        from watsonx_vision.vision_llm import LLMProvider
        from watsonx_vision.fraud_detector import FraudDetector, FraudSeverity

        mock_internal_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''{"valid": true, "reason": "Document is valid",
            "layout_score": 85, "field_score": 90, "forgery_signs": []}'''
        mock_internal_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_langchain_ibm.ChatWatsonx.return_value = mock_internal_llm

        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            api_key="test-key",
            url="https://test.com",
            project_id="test-project",
            retry_enabled=False,
        )
        vision_llm = VisionLLM(config)
        detector = FraudDetector(vision_llm)

        result = await detector.validate_document_async(
            "data:image/png;base64,abc123",
            filename="test.png"
        )

        assert result.valid is True
        assert result.filename == "test.png"
        assert result.layout_score == 85
        assert result.field_score == 90

    @pytest.mark.asyncio
    async def test_validate_batch_async_concurrent(self):
        """Test async batch validation runs concurrently"""
        from watsonx_vision import VisionLLM, VisionLLMConfig
        from watsonx_vision.vision_llm import LLMProvider
        from watsonx_vision.fraud_detector import FraudDetector

        mock_internal_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''{"valid": true, "reason": "Valid",
            "layout_score": 80, "field_score": 80, "forgery_signs": []}'''
        mock_internal_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_langchain_ibm.ChatWatsonx.return_value = mock_internal_llm

        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            api_key="test-key",
            url="https://test.com",
            project_id="test-project",
            retry_enabled=False,
        )
        vision_llm = VisionLLM(config)
        detector = FraudDetector(vision_llm)

        documents = [
            {"image_data": "data:image/png;base64,abc1", "filename": "doc1.png"},
            {"image_data": "data:image/png;base64,abc2", "filename": "doc2.png"},
            {"image_data": "data:image/png;base64,abc3", "filename": "doc3.png"},
        ]

        results = await detector.validate_batch_async(documents)

        assert len(results) == 3
        assert all(r.valid for r in results)


class TestCrossValidatorAsync:
    """Tests for CrossValidator async methods"""

    @pytest.mark.asyncio
    async def test_validate_async(self):
        """Test async cross-validation"""
        from watsonx_vision.cross_validator import CrossValidator

        # Create mock LLM and pass it directly
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''{
            "passed": true,
            "inconsistencies": [],
            "matched_fields": ["name", "dob"],
            "summary": "All fields match",
            "confidence": 95
        }'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Pass mock LLM directly to avoid import issues
        validator = CrossValidator(llm=mock_llm, retry_enabled=False)

        result = await validator.validate_async(
            application_data={"name": "John Doe", "dob": "1990-01-15"},
            document_data=[
                {"doc_type": "Passport", "name": "John Doe", "dob": "1990-01-15"}
            ]
        )

        assert result.passed is True
        assert result.confidence == 95
        assert "name" in result.matched_fields

    @pytest.mark.asyncio
    async def test_validate_batch_async(self):
        """Test async batch cross-validation"""
        from watsonx_vision.cross_validator import CrossValidator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''{
            "passed": true,
            "inconsistencies": [],
            "matched_fields": ["name"],
            "summary": "Match",
            "confidence": 90
        }'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        validator = CrossValidator(llm=mock_llm, retry_enabled=False)

        packages = [
            {
                "application_data": {"name": "John"},
                "document_data": [{"doc_type": "ID", "name": "John"}]
            },
            {
                "application_data": {"name": "Jane"},
                "document_data": [{"doc_type": "ID", "name": "Jane"}]
            }
        ]

        results = await validator.validate_batch_async(packages)

        assert len(results) == 2
        assert all(r.passed for r in results)
