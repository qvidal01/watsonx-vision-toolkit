"""
Fraud Detection Module

Provides document authenticity validation using vision-based analysis.
Detects forgery, manipulation, and other fraud indicators.

Extracted from IBM Watsonx Loan Preprocessing Agents project.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from .vision_llm import VisionLLM
from .exceptions import (
    DocumentAnalysisError,
    LLMResponseError,
    LLMParseError,
    WatsonxVisionError,
)

logger = logging.getLogger(__name__)


class FraudSeverity(Enum):
    """Severity levels for fraud indicators"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FraudResult:
    """
    Result of fraud detection analysis.

    Attributes:
        valid: Whether the document appears authentic
        confidence: Confidence score (0-100)
        reason: Human-readable explanation
        layout_score: Layout consistency score (0-100)
        field_score: Field formatting score (0-100)
        forgery_signs: List of forgery indicators found
        severity: Overall fraud severity level
        filename: Original filename (if provided)
    """
    valid: bool
    confidence: int
    reason: str
    layout_score: int
    field_score: int
    forgery_signs: List[str]
    severity: FraudSeverity
    filename: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "valid": self.valid,
            "confidence": self.confidence,
            "reason": self.reason,
            "layout_score": self.layout_score,
            "field_score": self.field_score,
            "forgery_signs": self.forgery_signs,
            "severity": self.severity.value,
            "filename": self.filename
        }


class FraudDetector:
    """
    Document fraud detector using vision-based analysis.

    Analyzes documents for:
    - Layout inconsistencies
    - Field formatting issues
    - Photo manipulation
    - Text overlays
    - Font mismatches
    - Watermark detection
    - SAMPLE/SPECIMEN indicators

    Example:
        >>> from watsonx_vision import VisionLLM, VisionLLMConfig, FraudDetector
        >>> config = VisionLLMConfig(api_key="...", url="...", project_id="...")
        >>> llm = VisionLLM(config)
        >>> detector = FraudDetector(llm)
        >>> result = detector.validate_document(image_base64, filename="passport.png")
        >>> if not result.valid:
        ...     print(f"Fraud detected: {result.reason}")
    """

    def __init__(
        self,
        vision_llm: VisionLLM,
        layout_threshold: int = 70,
        field_threshold: int = 70,
        min_confidence: int = 60
    ):
        """
        Initialize fraud detector.

        Args:
            vision_llm: VisionLLM instance for image analysis
            layout_threshold: Minimum layout score for validity (0-100)
            field_threshold: Minimum field score for validity (0-100)
            min_confidence: Minimum confidence for valid result (0-100)
        """
        self.llm = vision_llm
        self.layout_threshold = layout_threshold
        self.field_threshold = field_threshold
        self.min_confidence = min_confidence

    def validate_document(
        self,
        image_data: str,
        filename: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> FraudResult:
        """
        Validate document authenticity.

        Args:
            image_data: Base64-encoded image data URI
            filename: Optional filename for result tracking
            document_type: Optional document type hint for specialized validation

        Returns:
            FraudResult with validation details

        Raises:
            DocumentAnalysisError: If document analysis fails

        Example:
            >>> result = detector.validate_document(image_base64, "passport.png")
            >>> print(f"Valid: {result.valid}, Confidence: {result.confidence}%")
        """
        logger.debug(f"Validating document: {filename or 'unnamed'}")

        try:
            # Perform vision-based validation
            raw_result = self.llm.validate_authenticity(image_data)
        except WatsonxVisionError as e:
            logger.error(f"Vision analysis failed for {filename}: {e}")
            raise DocumentAnalysisError(
                f"Failed to analyze document: {filename or 'unnamed'}",
                details={"original_error": str(e), "filename": filename}
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during document validation: {e}")
            raise DocumentAnalysisError(
                f"Unexpected error analyzing document: {filename or 'unnamed'}",
                details=str(e)
            ) from e

        # Extract scores with safe defaults
        layout_score = raw_result.get("layout_score", 0)
        field_score = raw_result.get("field_score", 0)
        forgery_signs = raw_result.get("forgery_signs", [])
        reason = raw_result.get("reason", "No reason provided")

        # Validate score types
        if not isinstance(layout_score, (int, float)):
            logger.warning(f"Invalid layout_score type: {type(layout_score)}, defaulting to 0")
            layout_score = 0
        if not isinstance(field_score, (int, float)):
            logger.warning(f"Invalid field_score type: {type(field_score)}, defaulting to 0")
            field_score = 0

        # Calculate overall confidence
        confidence = int((layout_score + field_score) / 2)

        # Determine validity
        valid = (
            raw_result.get("valid", False) and
            layout_score >= self.layout_threshold and
            field_score >= self.field_threshold and
            confidence >= self.min_confidence
        )

        # Determine severity
        severity = self._calculate_severity(
            valid, confidence, len(forgery_signs)
        )

        logger.debug(f"Document {filename}: valid={valid}, confidence={confidence}, severity={severity.value}")

        return FraudResult(
            valid=valid,
            confidence=confidence,
            reason=reason,
            layout_score=layout_score,
            field_score=field_score,
            forgery_signs=forgery_signs,
            severity=severity,
            filename=filename
        )

    def validate_batch(
        self,
        documents: List[Dict[str, str]]
    ) -> List[FraudResult]:
        """
        Validate multiple documents.

        Args:
            documents: List of dicts with 'image_data' and optional 'filename', 'doc_type'

        Returns:
            List of FraudResult objects

        Example:
            >>> docs = [
            ...     {"image_data": img1_base64, "filename": "passport.png"},
            ...     {"image_data": img2_base64, "filename": "license.png"}
            ... ]
            >>> results = detector.validate_batch(docs)
            >>> invalid_count = sum(1 for r in results if not r.valid)
        """
        results = []
        for doc in documents:
            result = self.validate_document(
                image_data=doc["image_data"],
                filename=doc.get("filename"),
                document_type=doc.get("doc_type")
            )
            results.append(result)
        return results

    def _calculate_severity(
        self,
        valid: bool,
        confidence: int,
        forgery_count: int
    ) -> FraudSeverity:
        """
        Calculate fraud severity based on validation results.

        Args:
            valid: Whether document is valid
            confidence: Confidence score (0-100)
            forgery_count: Number of forgery signs detected

        Returns:
            FraudSeverity level
        """
        if valid and confidence >= 90 and forgery_count == 0:
            return FraudSeverity.NONE

        if valid and confidence >= 70:
            return FraudSeverity.LOW

        if not valid and forgery_count <= 2 and confidence >= 50:
            return FraudSeverity.MEDIUM

        if not valid and (forgery_count > 2 or confidence < 50):
            return FraudSeverity.HIGH

        if not valid and confidence < 30:
            return FraudSeverity.CRITICAL

        return FraudSeverity.MEDIUM

    def generate_report(
        self,
        results: List[FraudResult]
    ) -> Dict:
        """
        Generate summary report from validation results.

        Args:
            results: List of FraudResult objects

        Returns:
            Dict with summary statistics and details

        Example:
            >>> results = detector.validate_batch(documents)
            >>> report = detector.generate_report(results)
            >>> print(f"Fraud rate: {report['fraud_rate']}%")
        """
        total = len(results)
        valid_count = sum(1 for r in results if r.valid)
        invalid_count = total - valid_count

        avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0

        severity_counts = {
            FraudSeverity.NONE: 0,
            FraudSeverity.LOW: 0,
            FraudSeverity.MEDIUM: 0,
            FraudSeverity.HIGH: 0,
            FraudSeverity.CRITICAL: 0
        }

        for result in results:
            severity_counts[result.severity] += 1

        return {
            "total_documents": total,
            "valid_documents": valid_count,
            "invalid_documents": invalid_count,
            "fraud_rate": round((invalid_count / total * 100) if total > 0 else 0, 2),
            "average_confidence": round(avg_confidence, 2),
            "severity_breakdown": {
                k.value: v for k, v in severity_counts.items()
            },
            "details": [r.to_dict() for r in results]
        }


class SpecializedFraudDetector(FraudDetector):
    """
    Specialized fraud detector with document-type-specific validation.

    Extends base FraudDetector with additional checks for:
    - Tax returns (altered figures, fake schedules)
    - Bank statements (fabricated transactions)
    - Identity documents (photo swaps, data manipulation)
    - Business documents (forged signatures, fake seals)
    """

    def __init__(
        self,
        vision_llm: VisionLLM,
        layout_threshold: int = 70,
        field_threshold: int = 70,
        min_confidence: int = 60
    ):
        super().__init__(vision_llm, layout_threshold, field_threshold, min_confidence)

        # Document-specific validation rules
        self.specialized_prompts = {
            "tax_return": self._get_tax_return_prompt(),
            "bank_statement": self._get_bank_statement_prompt(),
            "passport": self._get_passport_prompt(),
            "drivers_license": self._get_drivers_license_prompt()
        }

    def validate_document(
        self,
        image_data: str,
        filename: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> FraudResult:
        """
        Validate with document-type-specific rules if provided.

        Args:
            image_data: Base64-encoded image data URI
            filename: Optional filename
            document_type: Document type for specialized validation

        Returns:
            FraudResult with enhanced validation
        """
        # Use specialized validation if document type is known
        if document_type and document_type.lower().replace(" ", "_") in self.specialized_prompts:
            return self._validate_specialized(image_data, filename, document_type)

        # Fall back to generic validation
        return super().validate_document(image_data, filename, document_type)

    def _validate_specialized(
        self,
        image_data: str,
        filename: Optional[str],
        document_type: str
    ) -> FraudResult:
        """
        Perform specialized validation for specific document types.

        Raises:
            DocumentAnalysisError: If specialized analysis fails
        """
        doc_type_key = document_type.lower().replace(" ", "_")
        specialized_prompt = self.specialized_prompts.get(doc_type_key)

        logger.debug(f"Performing specialized validation for {document_type}: {filename}")

        try:
            if specialized_prompt:
                # Use custom prompt for this document type
                result = self.llm.analyze_image(
                    image_data=image_data,
                    prompt=f"Validate this {document_type} for authenticity",
                    system_prompt=specialized_prompt,
                    parse_json=True
                )
            else:
                # Fall back to generic validation
                result = self.llm.validate_authenticity(image_data)
        except WatsonxVisionError as e:
            logger.error(f"Specialized validation failed for {document_type}: {e}")
            raise DocumentAnalysisError(
                f"Failed specialized validation for {document_type}",
                details={"document_type": document_type, "filename": filename, "error": str(e)}
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in specialized validation: {e}")
            raise DocumentAnalysisError(
                f"Unexpected error in specialized validation for {document_type}",
                details=str(e)
            ) from e

        # Convert to FraudResult with safe defaults
        layout_score = result.get("layout_score", 0)
        field_score = result.get("field_score", 0)
        forgery_signs = result.get("forgery_signs", [])
        confidence = int((layout_score + field_score) / 2)
        valid = result.get("valid", False)
        severity = self._calculate_severity(valid, confidence, len(forgery_signs))

        return FraudResult(
            valid=valid,
            confidence=confidence,
            reason=result.get("reason", ""),
            layout_score=layout_score,
            field_score=field_score,
            forgery_signs=forgery_signs,
            severity=severity,
            filename=filename
        )

    def _get_tax_return_prompt(self) -> str:
        """Get specialized prompt for tax return validation"""
        return """You are an expert in detecting fraudulent tax returns. Analyze this tax return for signs of manipulation.

Check for:
1. Altered figures (inconsistent fonts, alignment issues)
2. Fake schedules or attachments
3. Inconsistent formatting across pages
4. Missing or altered IRS watermarks
5. Unusual calculations or unrealistic numbers
6. Signs of digital manipulation (cloning, overlays)

Return JSON:
{
  "valid": true/false,
  "reason": "explanation",
  "layout_score": 0-100,
  "field_score": 0-100,
  "forgery_signs": ["list of specific issues"]
}"""

    def _get_bank_statement_prompt(self) -> str:
        """Get specialized prompt for bank statement validation"""
        return """You are an expert in detecting fraudulent bank statements. Analyze this statement for authenticity.

Check for:
1. Fabricated transactions (inconsistent formatting)
2. Altered balances or dates
3. Missing bank logos or incorrect branding
4. Font inconsistencies
5. Unusual transaction patterns
6. Digital manipulation signs

Return JSON:
{
  "valid": true/false,
  "reason": "explanation",
  "layout_score": 0-100,
  "field_score": 0-100,
  "forgery_signs": ["list of specific issues"]
}"""

    def _get_passport_prompt(self) -> str:
        """Get specialized prompt for passport validation"""
        return """You are an expert in detecting fraudulent passports. Analyze this passport for authenticity.

Check for:
1. Photo manipulation or substitution
2. Altered personal data
3. Missing or fake security features
4. Inconsistent font or layout
5. SAMPLE or SPECIMEN watermarks
6. Signs of physical or digital alteration

Return JSON:
{
  "valid": true/false,
  "reason": "explanation",
  "layout_score": 0-100,
  "field_score": 0-100,
  "forgery_signs": ["list of specific issues"]
}"""

    def _get_drivers_license_prompt(self) -> str:
        """Get specialized prompt for driver's license validation"""
        return """You are an expert in detecting fraudulent driver's licenses. Analyze this license for authenticity.

Check for:
1. Photo quality and authenticity
2. Hologram or security feature presence
3. Consistent state/province formatting
4. Proper data alignment and fonts
5. SAMPLE or TEST indicators
6. Signs of alteration or manipulation

Return JSON:
{
  "valid": true/false,
  "reason": "explanation",
  "layout_score": 0-100,
  "field_score": 0-100,
  "forgery_signs": ["list of specific issues"]
}"""
