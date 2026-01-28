"""Tests for CrossValidator module"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch

# Mock the langchain modules before importing CrossValidator
mock_langchain_ibm = MagicMock()
mock_langchain_core = MagicMock()
mock_langchain_core_messages = MagicMock()
mock_langchain_core_output_parsers = MagicMock()

# Create mock classes
_mock_human_message = type('HumanMessage', (), {'__init__': lambda self, content: setattr(self, 'content', content)})
_mock_system_message = type('SystemMessage', (), {'__init__': lambda self, content: setattr(self, 'content', content)})

mock_langchain_core_messages.HumanMessage = _mock_human_message
mock_langchain_core_messages.SystemMessage = _mock_system_message
mock_langchain_core_output_parsers.JsonOutputParser = MagicMock

sys.modules['langchain_ibm'] = mock_langchain_ibm
sys.modules['langchain_core'] = mock_langchain_core
sys.modules['langchain_core.messages'] = mock_langchain_core_messages
sys.modules['langchain_core.output_parsers'] = mock_langchain_core_output_parsers

from watsonx_vision.cross_validator import (
    CrossValidator,
    FinancialCrossValidator,
    ValidationResult,
    Inconsistency,
    InconsistencySeverity
)


class TestInconsistency:
    """Tests for Inconsistency dataclass"""

    def test_inconsistency_creation(self):
        """Test creating an Inconsistency"""
        inc = Inconsistency(
            field="ssn",
            source1="123-45-6789",
            source2="987-65-4321",
            source1_doc="Application",
            source2_doc="Tax Return",
            severity=InconsistencySeverity.CRITICAL,
            explanation="SSN mismatch between documents"
        )

        assert inc.field == "ssn"
        assert inc.source1 == "123-45-6789"
        assert inc.source2 == "987-65-4321"
        assert inc.severity == InconsistencySeverity.CRITICAL

    def test_inconsistency_to_dict(self):
        """Test converting Inconsistency to dictionary"""
        inc = Inconsistency(
            field="name",
            source1="John Doe",
            source2="John D. Doe",
            source1_doc="Passport",
            source2_doc="License",
            severity=InconsistencySeverity.LOW,
            explanation="Minor name variation"
        )

        result = inc.to_dict()

        assert result["field"] == "name"
        assert result["source1"] == "John Doe"
        assert result["source2"] == "John D. Doe"
        assert result["severity"] == "low"
        assert "explanation" in result

    def test_all_severity_levels(self):
        """Test all severity levels serialize correctly"""
        for severity in InconsistencySeverity:
            inc = Inconsistency(
                field="test",
                source1="a",
                source2="b",
                source1_doc="doc1",
                source2_doc="doc2",
                severity=severity,
                explanation="test"
            )
            assert inc.to_dict()["severity"] == severity.value


class TestValidationResult:
    """Tests for ValidationResult dataclass"""

    def test_validation_result_creation_passed(self):
        """Test creating a passing ValidationResult"""
        result = ValidationResult(
            passed=True,
            total_inconsistencies=0,
            inconsistencies=[],
            matched_fields=["name", "dob", "ssn"],
            summary="All data consistent across documents",
            confidence=95
        )

        assert result.passed is True
        assert result.total_inconsistencies == 0
        assert len(result.matched_fields) == 3
        assert result.confidence == 95

    def test_validation_result_creation_failed(self):
        """Test creating a failing ValidationResult"""
        inconsistencies = [
            Inconsistency(
                "ssn", "123-45-6789", "987-65-4321",
                "Application", "Tax Return",
                InconsistencySeverity.CRITICAL,
                "SSN mismatch"
            )
        ]

        result = ValidationResult(
            passed=False,
            total_inconsistencies=1,
            inconsistencies=inconsistencies,
            matched_fields=["name", "dob"],
            summary="Critical SSN inconsistency found",
            confidence=30
        )

        assert result.passed is False
        assert result.total_inconsistencies == 1
        assert len(result.inconsistencies) == 1
        assert result.confidence == 30

    def test_validation_result_to_dict(self):
        """Test converting ValidationResult to dictionary"""
        inconsistencies = [
            Inconsistency(
                "address", "123 Main St", "123 Main Street",
                "Application", "Bank Statement",
                InconsistencySeverity.MEDIUM,
                "Address abbreviation difference"
            )
        ]

        result = ValidationResult(
            passed=True,
            total_inconsistencies=1,
            inconsistencies=inconsistencies,
            matched_fields=["name", "ssn"],
            summary="Minor address variation",
            confidence=85
        )

        result_dict = result.to_dict()

        assert result_dict["passed"] is True
        assert result_dict["total_inconsistencies"] == 1
        assert len(result_dict["inconsistencies"]) == 1
        assert result_dict["inconsistencies"][0]["severity"] == "medium"
        assert result_dict["confidence"] == 85

    def test_validation_result_empty_inconsistencies(self):
        """Test ValidationResult with no inconsistencies"""
        result = ValidationResult(
            passed=True,
            total_inconsistencies=0,
            inconsistencies=[],
            matched_fields=[],
            summary="No data to compare",
            confidence=100
        )

        result_dict = result.to_dict()
        assert result_dict["inconsistencies"] == []


def create_mock_cross_validator():
    """Helper to create a mocked CrossValidator instance"""
    mock_llm = Mock()
    mock_llm.invoke = MagicMock(return_value=Mock(
        content='{"passed": true, "inconsistencies": [], "matched_fields": ["name", "dob"], "summary": "All consistent", "confidence": 90}'
    ))

    validator = CrossValidator(llm=mock_llm)

    # Replace the parser with a mock
    validator._parser = Mock()
    validator._parser.parse = MagicMock(return_value={
        "passed": True,
        "inconsistencies": [],
        "matched_fields": ["name", "dob"],
        "summary": "All data consistent",
        "confidence": 90
    })

    return validator


class TestCrossValidator:
    """Tests for CrossValidator class"""

    def test_validator_initialization_with_llm(self):
        """Test initializing CrossValidator with pre-configured LLM"""
        mock_llm = Mock()
        validator = CrossValidator(llm=mock_llm)
        assert validator._llm == mock_llm

    def test_validator_raises_without_langchain(self):
        """Test that CrossValidator raises ImportError without langchain-ibm"""
        with patch('watsonx_vision.cross_validator.LANGCHAIN_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                CrossValidator(
                    api_key="test",
                    url="http://test",
                    project_id="test"
                )
            assert "langchain-ibm is required" in str(exc_info.value)

    def test_field_severity_mapping(self):
        """Test default field severity mappings"""
        validator = create_mock_cross_validator()

        assert validator._field_severity["ssn"] == InconsistencySeverity.CRITICAL
        assert validator._field_severity["name"] == InconsistencySeverity.HIGH
        assert validator._field_severity["address"] == InconsistencySeverity.MEDIUM
        assert validator._field_severity["phone"] == InconsistencySeverity.LOW

    def test_validate_consistent_data(self):
        """Test validating consistent application and document data"""
        validator = create_mock_cross_validator()

        application_data = {
            "name": "John Doe",
            "dob": "1990-01-15",
            "ssn": "123-45-6789"
        }
        document_data = [
            {"doc_type": "Passport", "name": "John Doe", "dob": "1990-01-15"},
            {"doc_type": "Tax Return", "name": "John Doe", "ssn": "123-45-6789"}
        ]

        result = validator.validate(application_data, document_data)

        assert result.passed is True
        assert result.total_inconsistencies == 0
        validator._llm.invoke.assert_called_once()

    def test_validate_with_inconsistencies(self):
        """Test validating data with inconsistencies"""
        validator = create_mock_cross_validator()

        # Configure mock to return inconsistencies
        validator._parser.parse.return_value = {
            "passed": False,
            "inconsistencies": [
                {
                    "field": "name",
                    "source1": "John Doe",
                    "source2": "Jane Doe",
                    "source1_doc": "Application",
                    "source2_doc": "Passport",
                    "severity": "high",
                    "explanation": "Name mismatch"
                }
            ],
            "matched_fields": ["dob"],
            "summary": "Name inconsistency found",
            "confidence": 40
        }

        application_data = {"name": "John Doe", "dob": "1990-01-15"}
        document_data = [{"doc_type": "Passport", "name": "Jane Doe", "dob": "1990-01-15"}]

        result = validator.validate(application_data, document_data)

        assert result.passed is False
        assert result.total_inconsistencies == 1
        assert result.inconsistencies[0].field == "name"
        assert result.inconsistencies[0].severity == InconsistencySeverity.HIGH

    def test_validate_with_custom_fields(self):
        """Test validating with custom field severity mappings"""
        validator = create_mock_cross_validator()

        custom_fields = {
            "employee_id": InconsistencySeverity.CRITICAL,
            "department": InconsistencySeverity.LOW
        }

        application_data = {"employee_id": "E123"}
        document_data = [{"doc_type": "Badge", "employee_id": "E123"}]

        validator.validate(application_data, document_data, custom_fields=custom_fields)

        assert validator._field_severity["employee_id"] == InconsistencySeverity.CRITICAL
        assert validator._field_severity["department"] == InconsistencySeverity.LOW

    def test_validate_batch(self):
        """Test batch validation of multiple document packages"""
        validator = create_mock_cross_validator()

        packages = [
            {
                "application_data": {"name": "John Doe"},
                "document_data": [{"doc_type": "Passport", "name": "John Doe"}]
            },
            {
                "application_data": {"name": "Jane Smith"},
                "document_data": [{"doc_type": "License", "name": "Jane Smith"}]
            },
            {
                "application_data": {"name": "Bob Johnson"},
                "document_data": [{"doc_type": "ID", "name": "Bob Johnson"}]
            }
        ]

        results = validator.validate_batch(packages)

        assert len(results) == 3
        assert all(isinstance(r, ValidationResult) for r in results)
        assert validator._llm.invoke.call_count == 3

    def test_validate_batch_with_custom_fields(self):
        """Test batch validation with custom fields in package"""
        validator = create_mock_cross_validator()

        packages = [
            {
                "application_data": {"name": "John"},
                "document_data": [{"name": "John"}],
                "custom_fields": {"custom_id": InconsistencySeverity.HIGH}
            }
        ]

        results = validator.validate_batch(packages)

        assert len(results) == 1
        assert validator._field_severity["custom_id"] == InconsistencySeverity.HIGH

    def test_format_dict(self):
        """Test dictionary formatting for LLM prompt"""
        validator = create_mock_cross_validator()

        data = {
            "name": "John Doe",
            "dob": "1990-01-15",
            "filename": "doc.pdf",  # Should be filtered
            "doc_type": "Passport"  # Should be filtered
        }

        formatted = validator._format_dict(data)

        assert "name: John Doe" in formatted
        assert "dob: 1990-01-15" in formatted
        assert "filename" not in formatted
        assert "doc_type" not in formatted

    def test_llm_validate_severity_fallback(self):
        """Test that invalid severity falls back to MEDIUM"""
        validator = create_mock_cross_validator()

        validator._parser.parse.return_value = {
            "passed": False,
            "inconsistencies": [
                {
                    "field": "test",
                    "source1": "a",
                    "source2": "b",
                    "source1_doc": "doc1",
                    "source2_doc": "doc2",
                    "severity": "invalid_severity",
                    "explanation": "test"
                }
            ],
            "matched_fields": [],
            "summary": "test",
            "confidence": 50
        }

        result = validator.validate({"test": "a"}, [{"test": "b"}])

        assert result.inconsistencies[0].severity == InconsistencySeverity.MEDIUM

    def test_llm_validate_missing_fields(self):
        """Test handling of missing fields in LLM response"""
        validator = create_mock_cross_validator()

        validator._parser.parse.return_value = {
            "passed": True,
            # Missing: inconsistencies, matched_fields, summary, confidence
        }

        result = validator.validate({"name": "John"}, [{"name": "John"}])

        assert result.passed is True
        assert result.total_inconsistencies == 0
        assert result.inconsistencies == []
        assert result.matched_fields == []
        assert result.summary == ""
        assert result.confidence == 80  # Default


class TestCrossValidatorReport:
    """Tests for CrossValidator report generation"""

    def test_generate_report_passed(self):
        """Test generating report for passing validation"""
        validator = create_mock_cross_validator()

        result = ValidationResult(
            passed=True,
            total_inconsistencies=0,
            inconsistencies=[],
            matched_fields=["name", "dob", "ssn"],
            summary="All data is consistent",
            confidence=95
        )

        report = validator.generate_report(result)

        assert "PASSED" in report
        assert "95%" in report
        assert "Total Inconsistencies: 0" in report
        assert "name, dob, ssn" in report

    def test_generate_report_failed(self):
        """Test generating report for failing validation"""
        validator = create_mock_cross_validator()

        inconsistencies = [
            Inconsistency(
                "ssn", "123-45-6789", "987-65-4321",
                "Application", "Tax Return",
                InconsistencySeverity.CRITICAL,
                "SSN numbers do not match"
            ),
            Inconsistency(
                "name", "John Doe", "John D. Doe",
                "Application", "License",
                InconsistencySeverity.LOW,
                "Minor name variation"
            )
        ]

        result = ValidationResult(
            passed=False,
            total_inconsistencies=2,
            inconsistencies=inconsistencies,
            matched_fields=["dob"],
            summary="Critical inconsistencies found",
            confidence=30
        )

        report = validator.generate_report(result)

        assert "FAILED" in report
        assert "30%" in report
        assert "Total Inconsistencies: 2" in report
        assert "SSN" in report
        assert "CRITICAL" in report.lower() or "critical" in report
        assert "NAME" in report
        assert "LOW" in report.lower() or "low" in report

    def test_generate_report_without_details(self):
        """Test generating report without detailed inconsistency list"""
        validator = create_mock_cross_validator()

        inconsistencies = [
            Inconsistency(
                "name", "John", "Jane",
                "App", "Doc",
                InconsistencySeverity.HIGH,
                "Mismatch"
            )
        ]

        result = ValidationResult(
            passed=False,
            total_inconsistencies=1,
            inconsistencies=inconsistencies,
            matched_fields=[],
            summary="Issue found",
            confidence=50
        )

        report = validator.generate_report(result, include_details=False)

        assert "Issue found" in report
        # Detailed inconsistency section should not be present
        assert "Mismatch" not in report

    def test_generate_report_severity_icons(self):
        """Test that severity icons are correctly applied"""
        validator = create_mock_cross_validator()

        inconsistencies = [
            Inconsistency("f1", "a", "b", "d1", "d2", InconsistencySeverity.CRITICAL, "e1"),
            Inconsistency("f2", "a", "b", "d1", "d2", InconsistencySeverity.HIGH, "e2"),
            Inconsistency("f3", "a", "b", "d1", "d2", InconsistencySeverity.MEDIUM, "e3"),
            Inconsistency("f4", "a", "b", "d1", "d2", InconsistencySeverity.LOW, "e4"),
        ]

        result = ValidationResult(
            passed=False,
            total_inconsistencies=4,
            inconsistencies=inconsistencies,
            matched_fields=[],
            summary="Multiple issues",
            confidence=20
        )

        report = validator.generate_report(result)

        # Check for severity indicators in report
        assert "critical" in report.lower()
        assert "high" in report.lower()
        assert "medium" in report.lower()
        assert "low" in report.lower()


def create_mock_financial_validator():
    """Helper to create a mocked FinancialCrossValidator instance"""
    mock_llm = Mock()
    mock_llm.invoke = MagicMock(return_value=Mock(
        content='{"passed": true, "inconsistencies": [], "matched_fields": ["revenue", "net_income"], "summary": "Financial data consistent", "confidence": 85}'
    ))

    validator = FinancialCrossValidator(llm=mock_llm)

    # Replace the parser with a mock
    validator._parser = Mock()
    validator._parser.parse = MagicMock(return_value={
        "passed": True,
        "inconsistencies": [],
        "matched_fields": ["revenue", "net_income"],
        "summary": "Financial data consistent",
        "confidence": 85
    })

    return validator


class TestFinancialCrossValidator:
    """Tests for FinancialCrossValidator class"""

    def test_financial_validator_initialization(self):
        """Test initializing FinancialCrossValidator"""
        validator = create_mock_financial_validator()

        # Check financial-specific severities are added
        assert validator._field_severity["revenue"] == InconsistencySeverity.HIGH
        assert validator._field_severity["gross_income"] == InconsistencySeverity.HIGH
        assert validator._field_severity["net_income"] == InconsistencySeverity.HIGH
        assert validator._field_severity["total_expenses"] == InconsistencySeverity.MEDIUM
        assert validator._field_severity["total_assets"] == InconsistencySeverity.HIGH

    def test_financial_validator_inherits_base_severities(self):
        """Test that FinancialCrossValidator inherits base severities"""
        validator = create_mock_financial_validator()

        # Base severities should still exist
        assert validator._field_severity["ssn"] == InconsistencySeverity.CRITICAL
        assert validator._field_severity["name"] == InconsistencySeverity.HIGH

    def test_validate_financials(self):
        """Test validating financial documents"""
        validator = create_mock_financial_validator()

        tax_return_data = {
            "doc_type": "Tax Return",
            "revenue": 100000,
            "net_income": 75000
        }

        bank_statements = [
            {"doc_type": "Bank Statement", "deposits": 8500, "month": "January"},
            {"doc_type": "Bank Statement", "deposits": 8200, "month": "February"}
        ]

        result = validator.validate_financials(
            tax_return_data=tax_return_data,
            bank_statements=bank_statements,
            tolerance_percent=5.0
        )

        assert isinstance(result, ValidationResult)
        validator._llm.invoke.assert_called_once()

    def test_validate_financials_with_pfs(self):
        """Test validating financials including Personal Financial Statement"""
        validator = create_mock_financial_validator()

        tax_return_data = {"doc_type": "Tax Return", "net_income": 75000}
        bank_statements = [{"doc_type": "Bank Statement", "balance": 50000}]
        pfs_data = {
            "doc_type": "Personal Financial Statement",
            "total_assets": 200000,
            "total_liabilities": 50000,
            "net_worth": 150000
        }

        result = validator.validate_financials(
            tax_return_data=tax_return_data,
            bank_statements=bank_statements,
            pfs_data=pfs_data
        )

        assert isinstance(result, ValidationResult)

    def test_validate_financials_without_pfs(self):
        """Test validating financials without Personal Financial Statement"""
        validator = create_mock_financial_validator()

        result = validator.validate_financials(
            tax_return_data={"revenue": 100000},
            bank_statements=[{"deposits": 8500}],
            pfs_data=None
        )

        assert isinstance(result, ValidationResult)


class TestInconsistencySeverityEnum:
    """Tests for InconsistencySeverity enum"""

    def test_all_severity_values(self):
        """Test all severity enum values"""
        assert InconsistencySeverity.LOW.value == "low"
        assert InconsistencySeverity.MEDIUM.value == "medium"
        assert InconsistencySeverity.HIGH.value == "high"
        assert InconsistencySeverity.CRITICAL.value == "critical"

    def test_severity_comparison(self):
        """Test that severities can be compared by identity"""
        inc1 = Inconsistency("f", "a", "b", "d1", "d2", InconsistencySeverity.HIGH, "e")
        inc2 = Inconsistency("f", "a", "b", "d1", "d2", InconsistencySeverity.HIGH, "e")

        assert inc1.severity == inc2.severity
        assert inc1.severity is InconsistencySeverity.HIGH
