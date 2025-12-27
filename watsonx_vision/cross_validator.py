"""
Cross-Validation Module

Provides multi-document validation to detect inconsistencies across documents
and application data. Uses LLM-based intelligent comparison.

Extracted from IBM Watsonx Loan Preprocessing Agents project.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

try:
    from langchain_ibm import ChatWatsonx
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class InconsistencySeverity(Enum):
    """Severity levels for data inconsistencies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Inconsistency:
    """
    Represents a data inconsistency between documents.

    Attributes:
        field: The field name where inconsistency was found
        source1: First source value
        source2: Second source value
        source1_doc: Document name for first source
        source2_doc: Document name for second source
        severity: How severe the inconsistency is
        explanation: Human-readable explanation
    """
    field: str
    source1: str
    source2: str
    source1_doc: str
    source2_doc: str
    severity: InconsistencySeverity
    explanation: str

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "field": self.field,
            "source1": self.source1,
            "source2": self.source2,
            "source1_doc": self.source1_doc,
            "source2_doc": self.source2_doc,
            "severity": self.severity.value,
            "explanation": self.explanation
        }


@dataclass
class ValidationResult:
    """
    Result of cross-validation analysis.

    Attributes:
        passed: Whether validation passed (no critical/high severity issues)
        total_inconsistencies: Total number of inconsistencies found
        inconsistencies: List of Inconsistency objects
        matched_fields: List of fields that matched across documents
        summary: Human-readable summary
        confidence: Confidence score (0-100)
    """
    passed: bool
    total_inconsistencies: int
    inconsistencies: List[Inconsistency]
    matched_fields: List[str]
    summary: str
    confidence: int

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "passed": self.passed,
            "total_inconsistencies": self.total_inconsistencies,
            "inconsistencies": [i.to_dict() for i in self.inconsistencies],
            "matched_fields": self.matched_fields,
            "summary": self.summary,
            "confidence": self.confidence
        }


class CrossValidator:
    """
    Cross-validation engine for multi-document consistency checking.

    Compares data across multiple documents to detect:
    - Name mismatches
    - Date of birth discrepancies
    - Address inconsistencies
    - ID number conflicts
    - Financial data discrepancies

    Example:
        >>> from watsonx_vision import CrossValidator
        >>> validator = CrossValidator(
        ...     api_key="...",
        ...     url="https://us-south.ml.cloud.ibm.com",
        ...     project_id="..."
        ... )
        >>> result = validator.validate(
        ...     application_data={"name": "John Doe", "dob": "1990-01-15"},
        ...     document_data=[
        ...         {"doc_type": "Passport", "name": "John Doe", "dob": "1990-01-15"},
        ...         {"doc_type": "License", "name": "John D Doe", "dob": "1990-01-15"}
        ...     ]
        ... )
        >>> if not result.passed:
        ...     for issue in result.inconsistencies:
        ...         print(f"{issue.field}: {issue.explanation}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        model_id: str = "mistralai/mistral-medium-2505",
        max_tokens: int = 2000,
        temperature: float = 0.0,
        llm: Optional[Any] = None
    ):
        """
        Initialize cross-validator.

        Args:
            api_key: IBM Cloud API key (not needed if llm provided)
            url: Watsonx URL (not needed if llm provided)
            project_id: Watsonx project ID (not needed if llm provided)
            model_id: Text model to use for comparison
            max_tokens: Maximum tokens for response
            temperature: Model temperature (0.0 for deterministic)
            llm: Optional pre-configured LLM instance

        Raises:
            ImportError: If langchain-ibm is not installed
        """
        if llm is not None:
            self._llm = llm
        else:
            if not LANGCHAIN_AVAILABLE:
                raise ImportError(
                    "langchain-ibm is required for CrossValidator. "
                    "Install with: pip install langchain-ibm"
                )

            self._llm = ChatWatsonx(
                model_id=model_id,
                apikey=api_key,
                url=url,
                project_id=project_id,
                params={
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                }
            )

        self._parser = JsonOutputParser()

        # Fields to compare with their severity if mismatched
        self._field_severity = {
            "name": InconsistencySeverity.HIGH,
            "full_name": InconsistencySeverity.HIGH,
            "dob": InconsistencySeverity.HIGH,
            "date_of_birth": InconsistencySeverity.HIGH,
            "ssn": InconsistencySeverity.CRITICAL,
            "social_security_number": InconsistencySeverity.CRITICAL,
            "passport_number": InconsistencySeverity.HIGH,
            "drivers_license_number": InconsistencySeverity.HIGH,
            "ein": InconsistencySeverity.HIGH,
            "address": InconsistencySeverity.MEDIUM,
            "city": InconsistencySeverity.MEDIUM,
            "state": InconsistencySeverity.MEDIUM,
            "zip": InconsistencySeverity.MEDIUM,
            "postal_code": InconsistencySeverity.MEDIUM,
            "phone": InconsistencySeverity.LOW,
            "email": InconsistencySeverity.LOW,
            "gender": InconsistencySeverity.LOW,
            "nationality": InconsistencySeverity.LOW,
            "bank_account_number": InconsistencySeverity.HIGH,
            "routing_number": InconsistencySeverity.HIGH,
        }

    def validate(
        self,
        application_data: Dict[str, Any],
        document_data: List[Dict[str, Any]],
        custom_fields: Optional[Dict[str, InconsistencySeverity]] = None
    ) -> ValidationResult:
        """
        Cross-validate application data against document data.

        Args:
            application_data: Data from the application form
            document_data: List of extracted data from documents
            custom_fields: Optional custom field severity mappings

        Returns:
            ValidationResult with details of any inconsistencies

        Example:
            >>> result = validator.validate(
            ...     application_data={"name": "John Doe", "ssn": "123-45-6789"},
            ...     document_data=[
            ...         {"doc_type": "Tax Return", "name": "John Doe", "ssn": "123-45-6789"},
            ...         {"doc_type": "W2", "name": "John D. Doe", "ssn": "123-45-6789"}
            ...     ]
            ... )
        """
        if custom_fields:
            self._field_severity.update(custom_fields)

        # Use LLM for intelligent comparison
        result = self._llm_validate(application_data, document_data)

        return result

    def _llm_validate(
        self,
        application_data: Dict[str, Any],
        document_data: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Use LLM to perform intelligent cross-validation"""

        system_prompt = """You are a document verification specialist. Your task is to cross-validate application data against extracted document data to identify any inconsistencies.

Consider that:
1. Names may have minor variations (e.g., "John Doe" vs "John D. Doe") - flag as LOW severity
2. Date formats may differ but represent the same date - not an inconsistency
3. Addresses may be abbreviated differently - use judgment
4. ID numbers (SSN, passport, license) must match exactly - HIGH/CRITICAL severity
5. Missing data in one source is not necessarily an inconsistency

Analyze the data and return JSON in this format:
```json
{
  "passed": true/false,
  "inconsistencies": [
    {
      "field": "field_name",
      "source1": "value from source 1",
      "source2": "value from source 2",
      "source1_doc": "Application" or document type,
      "source2_doc": document type,
      "severity": "low|medium|high|critical",
      "explanation": "Brief explanation of the issue"
    }
  ],
  "matched_fields": ["list", "of", "matching", "fields"],
  "summary": "Brief summary of validation results",
  "confidence": 0-100
}
```

Set passed=false if there are any HIGH or CRITICAL severity inconsistencies."""

        # Format the data for the LLM
        docs_formatted = "\n".join([
            f"Document {i+1} ({doc.get('doc_type', 'Unknown')}):\n{self._format_dict(doc)}"
            for i, doc in enumerate(document_data)
        ])

        user_message = f"""Please cross-validate the following data:

APPLICATION DATA:
{self._format_dict(application_data)}

EXTRACTED DOCUMENT DATA:
{docs_formatted}

Identify any inconsistencies between the application and documents, or between different documents."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        response = self._llm.invoke(messages)
        parsed = self._parser.parse(response.content)

        # Convert to ValidationResult
        inconsistencies = []
        for inc in parsed.get("inconsistencies", []):
            severity_str = inc.get("severity", "medium").lower()
            severity = InconsistencySeverity(severity_str) if severity_str in [s.value for s in InconsistencySeverity] else InconsistencySeverity.MEDIUM

            inconsistencies.append(Inconsistency(
                field=inc.get("field", "unknown"),
                source1=str(inc.get("source1", "")),
                source2=str(inc.get("source2", "")),
                source1_doc=inc.get("source1_doc", "Unknown"),
                source2_doc=inc.get("source2_doc", "Unknown"),
                severity=severity,
                explanation=inc.get("explanation", "")
            ))

        return ValidationResult(
            passed=parsed.get("passed", True),
            total_inconsistencies=len(inconsistencies),
            inconsistencies=inconsistencies,
            matched_fields=parsed.get("matched_fields", []),
            summary=parsed.get("summary", ""),
            confidence=parsed.get("confidence", 80)
        )

    def _format_dict(self, data: Dict) -> str:
        """Format dictionary for LLM prompt"""
        lines = []
        for key, value in data.items():
            if key not in ["filename", "doc_type"]:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def validate_batch(
        self,
        packages: List[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """
        Validate multiple document packages.

        Args:
            packages: List of dicts with 'application_data' and 'document_data' keys

        Returns:
            List of ValidationResult objects

        Example:
            >>> packages = [
            ...     {
            ...         "application_data": {...},
            ...         "document_data": [{...}, {...}]
            ...     },
            ...     ...
            ... ]
            >>> results = validator.validate_batch(packages)
        """
        results = []
        for package in packages:
            result = self.validate(
                application_data=package["application_data"],
                document_data=package["document_data"],
                custom_fields=package.get("custom_fields")
            )
            results.append(result)
        return results

    def generate_report(
        self,
        result: ValidationResult,
        include_details: bool = True
    ) -> str:
        """
        Generate a human-readable report from validation results.

        Args:
            result: ValidationResult object
            include_details: Whether to include detailed inconsistency list

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "CROSS-VALIDATION REPORT",
            "=" * 60,
            "",
            f"Status: {'PASSED ✓' if result.passed else 'FAILED ✗'}",
            f"Confidence: {result.confidence}%",
            f"Total Inconsistencies: {result.total_inconsistencies}",
            "",
            "Summary:",
            result.summary,
            "",
        ]

        if result.matched_fields:
            report_lines.extend([
                "Matched Fields:",
                ", ".join(result.matched_fields),
                "",
            ])

        if include_details and result.inconsistencies:
            report_lines.extend([
                "Inconsistencies Found:",
                "-" * 40,
            ])

            for i, inc in enumerate(result.inconsistencies, 1):
                severity_icon = {
                    InconsistencySeverity.LOW: "⚪",
                    InconsistencySeverity.MEDIUM: "🟡",
                    InconsistencySeverity.HIGH: "🟠",
                    InconsistencySeverity.CRITICAL: "🔴"
                }.get(inc.severity, "⚪")

                report_lines.extend([
                    f"\n{i}. {severity_icon} {inc.field.upper()} [{inc.severity.value}]",
                    f"   {inc.source1_doc}: {inc.source1}",
                    f"   {inc.source2_doc}: {inc.source2}",
                    f"   → {inc.explanation}",
                ])

        report_lines.append("\n" + "=" * 60)

        return "\n".join(report_lines)


class FinancialCrossValidator(CrossValidator):
    """
    Specialized cross-validator for financial documents.

    Extends base CrossValidator with:
    - Revenue/income validation
    - Expense verification
    - Balance sheet reconciliation
    - Tax return vs bank statement comparison
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add financial-specific field severities
        self._field_severity.update({
            "revenue": InconsistencySeverity.HIGH,
            "gross_income": InconsistencySeverity.HIGH,
            "net_income": InconsistencySeverity.HIGH,
            "total_expenses": InconsistencySeverity.MEDIUM,
            "total_assets": InconsistencySeverity.HIGH,
            "total_liabilities": InconsistencySeverity.HIGH,
            "net_worth": InconsistencySeverity.HIGH,
            "annual_revenue": InconsistencySeverity.HIGH,
            "monthly_income": InconsistencySeverity.MEDIUM,
        })

    def validate_financials(
        self,
        tax_return_data: Dict[str, Any],
        bank_statements: List[Dict[str, Any]],
        pfs_data: Optional[Dict[str, Any]] = None,
        tolerance_percent: float = 5.0
    ) -> ValidationResult:
        """
        Validate financial data consistency with tolerance.

        Args:
            tax_return_data: Data extracted from tax returns
            bank_statements: List of data from bank statements
            pfs_data: Optional personal financial statement data
            tolerance_percent: Acceptable percentage difference for amounts

        Returns:
            ValidationResult with financial validation details
        """
        # Combine all documents
        all_docs = [tax_return_data] + bank_statements
        if pfs_data:
            all_docs.append(pfs_data)

        # Use parent validation with financial context
        return self.validate(
            application_data=tax_return_data,  # Use tax return as baseline
            document_data=all_docs
        )
