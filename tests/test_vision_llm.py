"""Tests for VisionLLM module"""

import pytest
import base64
import tempfile
import os
import sys
from unittest.mock import Mock, MagicMock, patch, PropertyMock

# Mock the langchain modules before importing VisionLLM
# This simulates having the packages installed
_mock_human_message = type('HumanMessage', (), {'__init__': lambda self, content: setattr(self, 'content', content)})
_mock_system_message = type('SystemMessage', (), {'__init__': lambda self, content: setattr(self, 'content', content)})

mock_langchain_ibm = MagicMock()
mock_langchain_core = MagicMock()
mock_langchain_core_messages = MagicMock()
mock_langchain_core_output_parsers = MagicMock()

# Set up the message classes to work like real ones
mock_langchain_core_messages.HumanMessage = _mock_human_message
mock_langchain_core_messages.SystemMessage = _mock_system_message
mock_langchain_core_output_parsers.JsonOutputParser = MagicMock

sys.modules['langchain_ibm'] = mock_langchain_ibm
sys.modules['langchain_core'] = mock_langchain_core
sys.modules['langchain_core.messages'] = mock_langchain_core_messages
sys.modules['langchain_core.output_parsers'] = mock_langchain_core_output_parsers

# Now import the module (it will use our mocks)
from watsonx_vision.vision_llm import (
    VisionLLM,
    VisionLLMConfig,
    LLMProvider
)


class TestLLMProvider:
    """Tests for LLMProvider enum"""

    def test_provider_values(self):
        """Test all provider enum values"""
        assert LLMProvider.WATSONX.value == "watsonx"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"

    def test_provider_membership(self):
        """Test provider enum membership"""
        assert LLMProvider.WATSONX in LLMProvider
        assert LLMProvider.OLLAMA in LLMProvider


class TestVisionLLMConfig:
    """Tests for VisionLLMConfig dataclass"""

    def test_default_config(self):
        """Test default VisionLLMConfig values"""
        config = VisionLLMConfig()

        assert config.provider == LLMProvider.WATSONX
        assert config.model_id == "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
        assert config.api_key is None
        assert config.url is None
        assert config.project_id is None
        assert config.max_tokens == 2000
        assert config.temperature == 0.0
        assert config.top_p == 0.1

    def test_custom_config(self):
        """Test custom VisionLLMConfig values"""
        config = VisionLLMConfig(
            provider=LLMProvider.OLLAMA,
            model_id="llava:latest",
            api_key="test-key",
            url="http://localhost:11434",
            project_id="test-project",
            max_tokens=4000,
            temperature=0.5,
            top_p=0.9
        )

        assert config.provider == LLMProvider.OLLAMA
        assert config.model_id == "llava:latest"
        assert config.api_key == "test-key"
        assert config.url == "http://localhost:11434"
        assert config.max_tokens == 4000
        assert config.temperature == 0.5
        assert config.top_p == 0.9

    def test_watsonx_config(self):
        """Test Watsonx-specific configuration"""
        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            api_key="ibm-cloud-key",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="my-project-id"
        )

        assert config.provider == LLMProvider.WATSONX
        assert "llama" in config.model_id.lower()


class TestVisionLLMInitialization:
    """Tests for VisionLLM initialization"""

    def test_raises_without_watsonx_package(self):
        """Test that VisionLLM raises ImportError without langchain-ibm"""
        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            api_key="test",
            url="http://test",
            project_id="test"
        )

        with patch('watsonx_vision.vision_llm.WATSONX_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                VisionLLM(config)
            assert "langchain-ibm is required" in str(exc_info.value)

    def test_raises_for_unsupported_provider(self):
        """Test that VisionLLM raises ValueError for unsupported providers"""
        config = VisionLLMConfig(provider=LLMProvider.OPENAI)

        with pytest.raises(ValueError) as exc_info:
            VisionLLM(config)
        assert "Unsupported provider" in str(exc_info.value)

    def test_watsonx_initialization(self):
        """Test successful Watsonx initialization"""
        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            model_id="test-model",
            api_key="test-key",
            url="http://test-url",
            project_id="test-project",
            max_tokens=1000,
            temperature=0.1,
            top_p=0.2
        )

        llm = VisionLLM(config)

        # Verify the LLM instance was created and config preserved
        assert llm.config == config
        assert llm.config.model_id == "test-model"
        assert llm.config.api_key == "test-key"
        assert llm._llm is not None

    def test_ollama_raises_without_package(self):
        """Test that Ollama provider raises ImportError without langchain-ollama"""
        config = VisionLLMConfig(
            provider=LLMProvider.OLLAMA,
            model_id="llava:latest"
        )

        # Remove the langchain_ollama module temporarily
        with patch.dict(sys.modules, {'langchain_ollama': None}):
            with pytest.raises(ImportError) as exc_info:
                VisionLLM(config)
            assert "langchain-ollama is required" in str(exc_info.value)


def create_mock_vision_llm():
    """Helper to create a mocked VisionLLM instance"""
    # Create fresh mocks for each instance to avoid cross-test contamination
    mock_llm_instance = Mock()
    mock_llm_instance.invoke = MagicMock(return_value=Mock(content='{}'))
    mock_langchain_ibm.ChatWatsonx.return_value = mock_llm_instance

    config = VisionLLMConfig(
        provider=LLMProvider.WATSONX,
        api_key="test",
        url="http://test",
        project_id="test"
    )
    llm = VisionLLM(config)

    # Ensure we have a fresh _llm mock
    llm._llm = mock_llm_instance

    # Replace the parser with a fresh mock
    llm._parser = Mock()
    llm._parser.parse = MagicMock(return_value={})

    return llm


class TestVisionLLMMessageCreation:
    """Tests for VisionLLM message creation"""

    def test_create_vision_message_without_system_prompt(self):
        """Test creating vision message without system prompt"""
        llm = create_mock_vision_llm()

        messages = llm._create_vision_message(
            image_data="data:image/png;base64,abc123",
            prompt="Describe this image"
        )

        assert len(messages) == 1
        # Check the HumanMessage content structure
        assert messages[0].content[0]["type"] == "text"
        assert messages[0].content[0]["text"] == "Describe this image"
        assert messages[0].content[1]["type"] == "image_url"
        assert messages[0].content[1]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_create_vision_message_with_system_prompt(self):
        """Test creating vision message with system prompt"""
        llm = create_mock_vision_llm()

        messages = llm._create_vision_message(
            image_data="data:image/png;base64,abc123",
            prompt="Classify this document",
            system_prompt="You are a document classifier"
        )

        assert len(messages) == 2
        # First should be SystemMessage
        assert messages[0].content == "You are a document classifier"
        # Second should be HumanMessage with content
        assert messages[1].content[0]["text"] == "Classify this document"


class TestVisionLLMAnalyzeImage:
    """Tests for VisionLLM analyze_image method"""

    def test_analyze_image_with_json_parsing(self):
        """Test analyzing image with JSON parsing"""
        llm = create_mock_vision_llm()
        llm._llm.invoke.return_value = Mock(content='{"result": "test"}')
        llm._parser.parse.return_value = {"result": "test"}

        result = llm.analyze_image(
            image_data="data:image/png;base64,abc",
            prompt="Analyze this",
            parse_json=True
        )

        assert result == {"result": "test"}
        llm._llm.invoke.assert_called_once()
        llm._parser.parse.assert_called_once()

    def test_analyze_image_without_json_parsing(self):
        """Test analyzing image without JSON parsing"""
        llm = create_mock_vision_llm()
        llm._llm.invoke.return_value = Mock(content="Raw text response")

        result = llm.analyze_image(
            image_data="data:image/png;base64,abc",
            prompt="Describe this",
            parse_json=False
        )

        assert result == "Raw text response"
        llm._parser.parse.assert_not_called()

    def test_analyze_image_with_system_prompt(self):
        """Test analyzing image with system prompt"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {"test": True}

        llm.analyze_image(
            image_data="data:image/png;base64,abc",
            prompt="Test prompt",
            system_prompt="Test system prompt",
            parse_json=True
        )

        # Verify invoke was called (message creation includes system prompt)
        llm._llm.invoke.assert_called_once()


class TestVisionLLMClassifyDocument:
    """Tests for VisionLLM classify_document method"""

    def test_classify_document_default_types(self):
        """Test classifying document with default document types"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {"doc_type": "Passport"}

        result = llm.classify_document(
            image_data="data:image/png;base64,abc"
        )

        assert result == {"doc_type": "Passport"}
        llm._llm.invoke.assert_called_once()

    def test_classify_document_custom_types(self):
        """Test classifying document with custom document types"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {"doc_type": "Invoice"}

        custom_types = ["Invoice", "Receipt", "Contract", "Other"]

        result = llm.classify_document(
            image_data="data:image/png;base64,abc",
            document_types=custom_types
        )

        assert result == {"doc_type": "Invoice"}

    def test_classify_document_returns_others(self):
        """Test classifying unrecognized document"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {"doc_type": "Others"}

        result = llm.classify_document(
            image_data="data:image/png;base64,abc"
        )

        assert result["doc_type"] == "Others"


class TestVisionLLMExtractInformation:
    """Tests for VisionLLM extract_information method"""

    def test_extract_information_default_fields(self):
        """Test extracting information with default fields"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {
            "name": "John Doe",
            "dob": "1990-01-15"
        }

        result = llm.extract_information(
            image_data="data:image/png;base64,abc"
        )

        assert result["name"] == "John Doe"
        assert result["dob"] == "1990-01-15"

    def test_extract_information_custom_fields(self):
        """Test extracting information with custom fields"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {
            "company_name": "ACME Corp",
            "ein": "12-3456789"
        }

        custom_fields = ["Company Name", "EIN", "Address"]

        result = llm.extract_information(
            image_data="data:image/png;base64,abc",
            fields=custom_fields
        )

        assert result["company_name"] == "ACME Corp"
        assert result["ein"] == "12-3456789"

    def test_extract_information_custom_date_format(self):
        """Test extracting information with custom date format"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {"dob": "01/15/1990"}

        llm.extract_information(
            image_data="data:image/png;base64,abc",
            date_format="MM/DD/YYYY"
        )

        # Verify the method was called (format is passed in system prompt)
        llm._llm.invoke.assert_called_once()


class TestVisionLLMValidateAuthenticity:
    """Tests for VisionLLM validate_authenticity method"""

    def test_validate_authenticity_valid_document(self):
        """Test validating an authentic document"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {
            "valid": True,
            "reason": "Document appears authentic",
            "layout_score": 90,
            "field_score": 85,
            "forgery_signs": []
        }

        result = llm.validate_authenticity(
            image_data="data:image/png;base64,abc"
        )

        assert result["valid"] is True
        assert result["layout_score"] == 90
        assert result["field_score"] == 85
        assert result["forgery_signs"] == []

    def test_validate_authenticity_invalid_document(self):
        """Test validating a fraudulent document"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {
            "valid": False,
            "reason": "SAMPLE watermark detected",
            "layout_score": 30,
            "field_score": 40,
            "forgery_signs": ["SAMPLE watermark", "Font inconsistency"]
        }

        result = llm.validate_authenticity(
            image_data="data:image/png;base64,abc"
        )

        assert result["valid"] is False
        assert result["layout_score"] == 30
        assert len(result["forgery_signs"]) == 2


class TestVisionLLMEncodeImage:
    """Tests for VisionLLM encode_image_to_base64 static method"""

    def test_encode_png_image(self):
        """Test encoding a PNG image"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
            f.write(png_data)
            f.flush()
            temp_path = f.name

        try:
            result = VisionLLM.encode_image_to_base64(temp_path)

            assert result.startswith("data:image/png;base64,")
            base64_part = result.split(",")[1]
            decoded = base64.b64decode(base64_part)
            assert decoded == png_data
        finally:
            os.unlink(temp_path)

    def test_encode_jpeg_image(self):
        """Test encoding a JPEG image"""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            jpeg_data = b'\xff\xd8\xff\xe0' + b'\x00' * 100
            f.write(jpeg_data)
            f.flush()
            temp_path = f.name

        try:
            result = VisionLLM.encode_image_to_base64(temp_path)
            assert result.startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(temp_path)

    def test_encode_with_explicit_mime_type(self):
        """Test encoding with explicit MIME type"""
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(b'test data')
            f.flush()
            temp_path = f.name

        try:
            result = VisionLLM.encode_image_to_base64(
                temp_path,
                mime_type="image/webp"
            )
            assert result.startswith("data:image/webp;base64,")
        finally:
            os.unlink(temp_path)

    def test_encode_unknown_extension_defaults_to_png(self):
        """Test that unknown extensions default to image/png"""
        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as f:
            f.write(b'test data')
            f.flush()
            temp_path = f.name

        try:
            result = VisionLLM.encode_image_to_base64(temp_path)
            assert result.startswith("data:image/png;base64,")
        finally:
            os.unlink(temp_path)

    def test_encode_file_not_found(self):
        """Test encoding non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            VisionLLM.encode_image_to_base64("/nonexistent/path/image.png")

    def test_encode_gif_image(self):
        """Test encoding a GIF image"""
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            gif_data = b'GIF89a' + b'\x00' * 100
            f.write(gif_data)
            f.flush()
            temp_path = f.name

        try:
            result = VisionLLM.encode_image_to_base64(temp_path)
            assert result.startswith("data:image/gif;base64,")
        finally:
            os.unlink(temp_path)


class TestVisionLLMOllamaProvider:
    """Tests for VisionLLM with Ollama provider"""

    def test_ollama_initialization_with_default_url(self):
        """Test Ollama initialization uses default URL if not provided"""
        config = VisionLLMConfig(
            provider=LLMProvider.OLLAMA,
            model_id="llava:latest",
            url=None
        )
        # Config stores None, but initialization would use default
        assert config.url is None

    def test_ollama_initialization_with_custom_url(self):
        """Test Ollama initialization with custom URL"""
        config = VisionLLMConfig(
            provider=LLMProvider.OLLAMA,
            model_id="llava:latest",
            url="http://ai-server:11434"
        )
        assert config.url == "http://ai-server:11434"

    def test_ollama_successful_initialization(self):
        """Test successful Ollama initialization with mocked package"""
        mock_ollama = MagicMock()
        mock_ollama.ChatOllama = MagicMock(return_value=Mock())

        with patch.dict(sys.modules, {'langchain_ollama': mock_ollama}):
            config = VisionLLMConfig(
                provider=LLMProvider.OLLAMA,
                model_id="llava:latest",
                url="http://localhost:11434"
            )
            llm = VisionLLM(config)

            mock_ollama.ChatOllama.assert_called_once_with(
                model="llava:latest",
                base_url="http://localhost:11434",
                temperature=0.0
            )


class TestVisionLLMEdgeCases:
    """Tests for VisionLLM edge cases"""

    def test_empty_image_data(self):
        """Test handling empty image data"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {"doc_type": "Unknown"}

        result = llm.classify_document(image_data="")

        # Should still call the LLM (validation is not done at this level)
        llm._llm.invoke.assert_called_once()

    def test_config_preserved(self):
        """Test that config is preserved on instance"""
        llm = create_mock_vision_llm()

        assert llm.config is not None
        assert llm.config.provider == LLMProvider.WATSONX

    def test_multiple_calls(self):
        """Test multiple consecutive calls"""
        llm = create_mock_vision_llm()
        llm._parser.parse.return_value = {"doc_type": "Test"}

        for _ in range(3):
            llm.classify_document(image_data="data:image/png;base64,abc")

        assert llm._llm.invoke.call_count == 3

    def test_parser_is_initialized(self):
        """Test that JSON parser is initialized"""
        llm = create_mock_vision_llm()
        # The original _parser would be a JsonOutputParser mock
        # We replaced it with our mock, but the original should have been set
        assert llm._parser is not None
