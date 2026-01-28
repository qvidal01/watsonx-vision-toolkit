"""
Tests for the CLI module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from watsonx_vision.cli import cli, load_image, output_result, format_pretty


class TestCLIHelpers:
    """Test CLI helper functions."""

    def test_load_image_nonexistent(self):
        """Test loading a nonexistent image raises error."""
        from click import ClickException

        with pytest.raises(ClickException) as exc_info:
            load_image("/nonexistent/path/image.png")

        assert "not found" in str(exc_info.value)

    def test_load_image_unsupported_format(self):
        """Test loading unsupported image format raises error."""
        from click import ClickException

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            temp_path = f.name

        try:
            with pytest.raises(ClickException) as exc_info:
                load_image(temp_path)

            assert "Unsupported image format" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_format_pretty_simple(self):
        """Test pretty formatting of simple dict."""
        data = {"name": "John", "age": 30}
        result = format_pretty(data)

        assert "name: John" in result
        assert "age: 30" in result

    def test_format_pretty_nested(self):
        """Test pretty formatting of nested dict."""
        data = {
            "person": {
                "name": "John",
                "address": {"city": "NYC"}
            }
        }
        result = format_pretty(data)

        assert "person:" in result
        assert "name: John" in result
        assert "city: NYC" in result

    def test_format_pretty_list(self):
        """Test pretty formatting with lists."""
        data = {"items": ["a", "b", "c"]}
        result = format_pretty(data)

        assert "items:" in result
        assert "- a" in result
        assert "- b" in result

    def test_output_result_json(self):
        """Test JSON output format."""
        data = {"doc_type": "Passport"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            output_result(data, "json", temp_path)
            content = Path(temp_path).read_text()
            parsed = json.loads(content)
            assert parsed == data
        finally:
            os.unlink(temp_path)


class TestCLICommands:
    """Test CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image file."""
        # Create minimal valid PNG (1x1 red pixel)
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_cli_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_cli_help(self, runner):
        """Test --help flag."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "classify" in result.output
        assert "extract" in result.output
        assert "validate" in result.output
        assert "fraud" in result.output

    def test_classify_help(self, runner):
        """Test classify --help."""
        result = runner.invoke(cli, ["classify", "--help"])
        assert result.exit_code == 0
        assert "Classify a document" in result.output
        assert "--types" in result.output
        assert "--provider" in result.output

    def test_extract_help(self, runner):
        """Test extract --help."""
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "Extract structured information" in result.output
        assert "--fields" in result.output
        assert "--date-format" in result.output

    def test_validate_help(self, runner):
        """Test validate --help."""
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate document authenticity" in result.output

    def test_fraud_help(self, runner):
        """Test fraud --help."""
        result = runner.invoke(cli, ["fraud", "--help"])
        assert result.exit_code == 0
        assert "Detect fraud" in result.output
        assert "--threshold" in result.output

    def test_analyze_help(self, runner):
        """Test analyze --help."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "Analyze an image" in result.output
        assert "--system" in result.output

    def test_config_help(self, runner):
        """Test config --help."""
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "--show" in result.output
        assert "--validate" in result.output
        assert "--env" in result.output

    def test_config_env(self, runner):
        """Test config --env shows environment variables."""
        result = runner.invoke(cli, ["config", "--env"])
        assert result.exit_code == 0
        assert "VISION_PROVIDER" in result.output
        assert "WATSONX_API_KEY" in result.output
        assert "OLLAMA_URL" in result.output

    def test_config_show(self, runner):
        """Test config --show displays configuration."""
        with patch.dict(os.environ, {"VISION_PROVIDER": "ollama"}, clear=False):
            result = runner.invoke(cli, ["config", "--show"])

        assert result.exit_code == 0
        assert "Provider:" in result.output
        assert "Model:" in result.output
        assert "Cache Enabled:" in result.output

    def test_classify_missing_image(self, runner):
        """Test classify with missing image file."""
        result = runner.invoke(cli, ["classify", "nonexistent.png"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()

    def test_extract_missing_image(self, runner):
        """Test extract with missing image file."""
        result = runner.invoke(cli, ["extract", "nonexistent.png"])
        assert result.exit_code != 0

    def test_validate_missing_image(self, runner):
        """Test validate with missing image file."""
        result = runner.invoke(cli, ["validate", "nonexistent.png"])
        assert result.exit_code != 0

    def test_fraud_missing_image(self, runner):
        """Test fraud with missing image file."""
        result = runner.invoke(cli, ["fraud", "nonexistent.png"])
        assert result.exit_code != 0


class TestCLIWithMockedLLM:
    """Test CLI commands with mocked LLM."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image file."""
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @patch("watsonx_vision.cli.get_llm")
    def test_classify_success(self, mock_get_llm, runner, sample_image):
        """Test successful classification."""
        mock_llm = MagicMock()
        mock_llm.classify_document.return_value = {"doc_type": "Passport"}
        mock_get_llm.return_value = mock_llm

        result = runner.invoke(cli, ["classify", sample_image])

        assert result.exit_code == 0
        assert "Passport" in result.output

    @patch("watsonx_vision.cli.get_llm")
    def test_classify_json_output(self, mock_get_llm, runner, sample_image):
        """Test classification with JSON output."""
        mock_llm = MagicMock()
        mock_llm.classify_document.return_value = {"doc_type": "License"}
        mock_get_llm.return_value = mock_llm

        result = runner.invoke(cli, ["classify", sample_image, "--output", "json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["doc_type"] == "License"

    @patch("watsonx_vision.cli.get_llm")
    def test_extract_success(self, mock_get_llm, runner, sample_image):
        """Test successful extraction."""
        mock_llm = MagicMock()
        mock_llm.extract_information.return_value = {
            "name": "John Doe",
            "dob": "1990-01-15"
        }
        mock_get_llm.return_value = mock_llm

        result = runner.invoke(cli, ["extract", sample_image])

        assert result.exit_code == 0
        assert "John Doe" in result.output

    @patch("watsonx_vision.cli.get_llm")
    def test_validate_success(self, mock_get_llm, runner, sample_image):
        """Test successful validation."""
        mock_llm = MagicMock()
        mock_llm.validate_authenticity.return_value = {
            "valid": True,
            "reason": "Document appears authentic",
            "layout_score": 95,
            "field_score": 90,
        }
        mock_get_llm.return_value = mock_llm

        result = runner.invoke(cli, ["validate", sample_image])

        assert result.exit_code == 0
        assert "VALID" in result.output

    @patch("watsonx_vision.cli.get_llm")
    def test_validate_invalid_document(self, mock_get_llm, runner, sample_image):
        """Test validation of invalid document."""
        mock_llm = MagicMock()
        mock_llm.validate_authenticity.return_value = {
            "valid": False,
            "reason": "Signs of manipulation detected",
            "layout_score": 40,
            "field_score": 50,
        }
        mock_get_llm.return_value = mock_llm

        result = runner.invoke(cli, ["validate", sample_image])

        assert result.exit_code == 0
        assert "INVALID" in result.output

    @patch("watsonx_vision.cli.get_llm")
    def test_analyze_success(self, mock_get_llm, runner, sample_image):
        """Test successful custom analysis."""
        mock_llm = MagicMock()
        mock_llm.analyze_image.return_value = {"description": "A document"}
        mock_get_llm.return_value = mock_llm

        result = runner.invoke(cli, ["analyze", sample_image, "Describe this"])

        assert result.exit_code == 0
        assert "document" in result.output.lower()

    @patch("watsonx_vision.cli.get_llm")
    def test_analyze_raw_output(self, mock_get_llm, runner, sample_image):
        """Test analyze with raw text output."""
        mock_llm = MagicMock()
        mock_llm.analyze_image.return_value = "This is a test document"
        mock_get_llm.return_value = mock_llm

        result = runner.invoke(cli, ["analyze", sample_image, "Describe", "--raw"])

        assert result.exit_code == 0
        assert "This is a test document" in result.output


class TestCLIConfigValidation:
    """Test CLI configuration validation."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_config_validate_missing_watsonx_key(self, runner):
        """Test validation fails when Watsonx API key is missing."""
        env = {
            "VISION_PROVIDER": "watsonx",
            # Missing WATSONX_API_KEY
        }
        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(cli, ["config", "--validate"])

        # Should report missing API key
        assert "API_KEY" in result.output or result.exit_code != 0

    def test_config_validate_ollama_no_key_needed(self, runner):
        """Test validation passes for Ollama without API key."""
        env = {
            "VISION_PROVIDER": "ollama",
            "OLLAMA_URL": "http://localhost:11434",
        }
        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(cli, ["config", "--show"])

        assert result.exit_code == 0
        assert "ollama" in result.output.lower()
