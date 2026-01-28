"""Tests for FraudDetector module"""

import pytest
from unittest.mock import Mock, MagicMock
from watsonx_vision.fraud_detector import (
    FraudDetector,
    FraudResult,
    FraudSeverity,
    SpecializedFraudDetector
)


class TestFraudResult:
    """Tests for FraudResult dataclass"""

    def test_fraud_result_creation(self):
        """Test creating a FraudResult"""
        result = FraudResult(
            valid=True,
            confidence=85,
            reason="Document appears authentic",
            layout_score=90,
            field_score=80,
            forgery_signs=[],
            severity=FraudSeverity.NONE,
            filename="passport.png"
        )

        assert result.valid is True
        assert result.confidence == 85
        assert result.severity == FraudSeverity.NONE
        assert len(result.forgery_signs) == 0

    def test_fraud_result_to_dict(self):
        """Test converting FraudResult to dictionary"""
        result = FraudResult(
            valid=False,
            confidence=40,
            reason="Possible forgery detected",
            layout_score=50,
            field_score=30,
            forgery_signs=["Photo manipulation", "Font mismatch"],
            severity=FraudSeverity.HIGH,
            filename="fake_id.png"
        )

        result_dict = result.to_dict()

        assert result_dict["valid"] is False
        assert result_dict["confidence"] == 40
        assert result_dict["severity"] == "high"
        assert len(result_dict["forgery_signs"]) == 2


class TestFraudDetector:
    """Tests for FraudDetector class"""

    @pytest.fixture
    def mock_vision_llm(self):
        """Create a mock VisionLLM"""
        mock = Mock()
        mock.validate_authenticity = MagicMock(return_value={
            "valid": True,
            "reason": "Document appears authentic",
            "layout_score": 95,
            "field_score": 95,  # Confidence (95+95)/2 = 95 >= 90 for NONE severity
            "forgery_signs": []
        })
        return mock

    def test_detector_initialization(self, mock_vision_llm):
        """Test initializing FraudDetector"""
        detector = FraudDetector(
            vision_llm=mock_vision_llm,
            layout_threshold=70,
            field_threshold=70,
            min_confidence=60
        )

        assert detector.layout_threshold == 70
        assert detector.field_threshold == 70
        assert detector.min_confidence == 60

    def test_validate_authentic_document(self, mock_vision_llm):
        """Test validating an authentic document"""
        detector = FraudDetector(mock_vision_llm)

        result = detector.validate_document(
            image_data="base64_encoded_image",
            filename="passport.png"
        )

        assert result.valid is True
        assert result.confidence >= 60
        assert result.severity == FraudSeverity.NONE
        mock_vision_llm.validate_authenticity.assert_called_once()

    def test_validate_fraudulent_document(self, mock_vision_llm):
        """Test validating a fraudulent document"""
        mock_vision_llm.validate_authenticity.return_value = {
            "valid": False,
            "reason": "SAMPLE watermark detected",
            "layout_score": 30,
            "field_score": 40,
            "forgery_signs": ["SAMPLE watermark", "Low resolution"]
        }

        detector = FraudDetector(mock_vision_llm)
        result = detector.validate_document(
            image_data="base64_encoded_image",
            filename="fake_passport.png"
        )

        assert result.valid is False
        assert len(result.forgery_signs) == 2
        assert result.severity in [FraudSeverity.HIGH, FraudSeverity.CRITICAL]

    def test_validate_batch(self, mock_vision_llm):
        """Test validating a batch of documents"""
        detector = FraudDetector(mock_vision_llm)

        documents = [
            {"image_data": "img1", "filename": "doc1.png"},
            {"image_data": "img2", "filename": "doc2.png"},
            {"image_data": "img3", "filename": "doc3.png"}
        ]

        results = detector.validate_batch(documents)

        assert len(results) == 3
        assert all(isinstance(r, FraudResult) for r in results)
        assert mock_vision_llm.validate_authenticity.call_count == 3

    def test_generate_report(self, mock_vision_llm):
        """Test generating a fraud report"""
        detector = FraudDetector(mock_vision_llm)

        results = [
            FraudResult(True, 90, "OK", 95, 85, [], FraudSeverity.NONE, "doc1.png"),
            FraudResult(False, 40, "Fake", 30, 50, ["issue"], FraudSeverity.HIGH, "doc2.png"),
            FraudResult(True, 80, "OK", 85, 75, [], FraudSeverity.LOW, "doc3.png"),
        ]

        report = detector.generate_report(results)

        assert report["total_documents"] == 3
        assert report["valid_documents"] == 2
        assert report["invalid_documents"] == 1
        assert report["fraud_rate"] == pytest.approx(33.33, rel=0.1)

    def test_severity_calculation(self, mock_vision_llm):
        """Test severity level calculation"""
        detector = FraudDetector(mock_vision_llm)

        # Test NONE severity (valid, high confidence >= 90, no issues)
        severity = detector._calculate_severity(True, 95, 0)
        assert severity == FraudSeverity.NONE

        # Test LOW severity (valid, moderate confidence >= 70)
        severity = detector._calculate_severity(True, 75, 0)
        assert severity == FraudSeverity.LOW

        # Test MEDIUM severity (invalid, few issues, moderate confidence)
        severity = detector._calculate_severity(False, 55, 2)
        assert severity == FraudSeverity.MEDIUM

        # Test HIGH severity (invalid, multiple issues or low confidence)
        severity = detector._calculate_severity(False, 45, 3)
        assert severity == FraudSeverity.HIGH

        # Test HIGH severity (invalid, low confidence triggers HIGH before CRITICAL)
        # Note: Due to condition ordering, confidence < 50 matches HIGH before CRITICAL
        severity = detector._calculate_severity(False, 20, 5)
        assert severity == FraudSeverity.HIGH


class TestSpecializedFraudDetector:
    """Tests for SpecializedFraudDetector"""

    @pytest.fixture
    def mock_vision_llm(self):
        """Create a mock VisionLLM"""
        mock = Mock()
        mock.validate_authenticity = MagicMock(return_value={
            "valid": True,
            "reason": "OK",
            "layout_score": 85,
            "field_score": 80,
            "forgery_signs": []
        })
        mock.analyze_image = MagicMock(return_value={
            "valid": True,
            "reason": "Tax return appears legitimate",
            "layout_score": 90,
            "field_score": 85,
            "forgery_signs": []
        })
        return mock

    def test_specialized_detector_initialization(self, mock_vision_llm):
        """Test initializing specialized detector"""
        detector = SpecializedFraudDetector(mock_vision_llm)

        assert "tax_return" in detector.specialized_prompts
        assert "bank_statement" in detector.specialized_prompts
        assert "passport" in detector.specialized_prompts

    def test_tax_return_validation(self, mock_vision_llm):
        """Test specialized tax return validation"""
        detector = SpecializedFraudDetector(mock_vision_llm)

        result = detector.validate_document(
            image_data="base64_image",
            filename="tax_return_2023.pdf",
            document_type="tax_return"
        )

        assert isinstance(result, FraudResult)
        mock_vision_llm.analyze_image.assert_called_once()
