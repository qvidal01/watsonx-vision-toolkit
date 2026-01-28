#!/usr/bin/env python3
"""
Complete Workflow Example

Demonstrates an end-to-end loan application processing workflow using
all components of the watsonx-vision-toolkit.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from watsonx_vision import (
    VisionLLM,
    VisionLLMConfig,
    LLMProvider,
    FraudDetector,
    FraudResult,
    CrossValidator,
    ValidationResult,
    DecisionEngine,
    Decision,
    RetryConfig,
)
from watsonx_vision.fraud_detector import FraudSeverity
from watsonx_vision.cross_validator import FieldSeverity
from watsonx_vision.decision_engine import DecisionStatus, LoanDecisionEngine


@dataclass
class LoanApplication:
    """Loan application data."""
    applicant_name: str
    date_of_birth: str
    ssn_last_4: str
    annual_income: float
    employer: str
    employment_start_date: str
    loan_amount: float
    loan_purpose: str
    address: str
    documents: dict[str, str]  # name -> file path


@dataclass
class ProcessingResult:
    """Complete processing result."""
    application: LoanApplication
    classifications: dict[str, str]
    extractions: dict[str, dict]
    fraud_results: dict[str, FraudResult]
    validation_result: ValidationResult
    decision: Decision

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "application": {
                "applicant_name": self.application.applicant_name,
                "loan_amount": self.application.loan_amount,
                "loan_purpose": self.application.loan_purpose,
            },
            "classifications": self.classifications,
            "extractions": self.extractions,
            "fraud_results": {
                name: {
                    "is_valid": r.is_valid,
                    "severity": r.severity.value,
                    "confidence": r.confidence,
                    "issues": r.issues,
                }
                for name, r in self.fraud_results.items()
            },
            "validation": {
                "is_valid": self.validation_result.is_valid,
                "overall_score": self.validation_result.overall_score,
                "discrepancy_count": len(self.validation_result.discrepancies),
            },
            "decision": {
                "status": self.decision.status.value,
                "overall_score": self.decision.overall_score,
                "reason": self.decision.reason,
                "criteria_results": {
                    k: {"score": v.score, "weight": v.weight, "passed": v.passed}
                    for k, v in self.decision.criteria_results.items()
                },
            },
        }


class LoanProcessor:
    """
    Complete loan application processor.

    Orchestrates document classification, information extraction,
    fraud detection, cross-validation, and decision making.
    """

    def __init__(self, provider: str = "ollama", verbose: bool = True):
        """
        Initialize the loan processor.

        Args:
            provider: LLM provider ("watsonx" or "ollama")
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.llm = self._create_llm(provider)
        self.fraud_detector = FraudDetector(self.llm)
        self.cross_validator = CrossValidator(self.llm)
        self.decision_engine = LoanDecisionEngine()

    def _create_llm(self, provider: str) -> VisionLLM:
        """Create VisionLLM with retry configuration."""
        if provider == "watsonx":
            config = VisionLLMConfig(
                provider=LLMProvider.WATSONX,
                model_id=os.getenv("VISION_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"),
                api_key=os.getenv("WATSONX_API_KEY"),
                project_id=os.getenv("WATSONX_PROJECT_ID"),
                url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
                # Production retry settings
                retry_enabled=True,
                retry_max_attempts=5,
                retry_base_delay=1.0,
                retry_max_delay=30.0,
            )
        else:
            config = VisionLLMConfig(
                provider=LLMProvider.OLLAMA,
                model_id=os.getenv("VISION_MODEL", "llava"),
                url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                retry_enabled=True,
                retry_max_attempts=3,
                retry_base_delay=0.5,
            )
        return VisionLLM(config)

    def _log(self, message: str):
        """Print progress message if verbose."""
        if self.verbose:
            print(f"[*] {message}")

    def process(self, application: LoanApplication) -> ProcessingResult:
        """
        Process a complete loan application.

        Args:
            application: Loan application data with document paths

        Returns:
            ProcessingResult with all analysis results
        """
        self._log(f"Processing loan application for: {application.applicant_name}")
        self._log(f"Loan amount: ${application.loan_amount:,.2f}")
        self._log(f"Documents: {list(application.documents.keys())}")

        # Step 1: Encode all documents
        self._log("\nStep 1: Encoding documents...")
        encoded_docs = {}
        for name, path in application.documents.items():
            encoded_docs[name] = VisionLLM.encode_image_to_base64(path)
            self._log(f"  - Encoded: {name}")

        # Step 2: Classify documents
        self._log("\nStep 2: Classifying documents...")
        classifications = {}
        for name, image_data in encoded_docs.items():
            result = self.llm.classify_document(image_data)
            classifications[name] = result.get("doc_type", "Unknown")
            self._log(f"  - {name}: {classifications[name]}")

        # Step 3: Extract information
        self._log("\nStep 3: Extracting information...")
        extractions = {}
        for name, image_data in encoded_docs.items():
            extracted = self.llm.extract_information(image_data)
            extractions[name] = extracted
            self._log(f"  - {name}: {len(extracted)} fields extracted")

        # Step 4: Fraud detection
        self._log("\nStep 4: Running fraud detection...")
        fraud_results = {}
        all_valid = True
        for name, image_data in encoded_docs.items():
            result = self.fraud_detector.validate_document(image_data)
            fraud_results[name] = result
            status = "VALID" if result.is_valid else f"SUSPICIOUS ({result.severity.value})"
            self._log(f"  - {name}: {status}")
            if not result.is_valid:
                all_valid = False

        # Step 5: Cross-validation
        self._log("\nStep 5: Cross-validating documents...")
        application_data = {
            "applicant_name": application.applicant_name,
            "date_of_birth": application.date_of_birth,
            "annual_income": application.annual_income,
            "employer": application.employer,
            "address": application.address,
        }
        validation_result = self.cross_validator.validate(application_data, encoded_docs)
        self._log(f"  - Valid: {validation_result.is_valid}")
        self._log(f"  - Score: {validation_result.overall_score:.1%}")
        self._log(f"  - Discrepancies: {len(validation_result.discrepancies)}")

        # Step 6: Make decision
        self._log("\nStep 6: Making loan decision...")

        # Prepare data for decision engine
        decision_data = {
            "applicant_age": self._calculate_age(application.date_of_birth),
            "annual_income": application.annual_income,
            "loan_amount": application.loan_amount,
            "employment_months": self._calculate_employment_months(
                application.employment_start_date
            ),
            "fraud_score": self._calculate_fraud_score(fraud_results),
            "validation_score": validation_result.overall_score,
            "debt_to_income": application.loan_amount / application.annual_income,
        }

        # Add custom criteria for fraud and validation
        self.decision_engine.add_criterion(
            name="fraud_check",
            evaluator=lambda d: 1.0 if d.get("fraud_score", 0) >= 0.8 else 0.0,
            weight=0.25,
            required=True,
        )
        self.decision_engine.add_criterion(
            name="document_validation",
            evaluator=lambda d: d.get("validation_score", 0),
            weight=0.15,
        )

        decision = self.decision_engine.evaluate(decision_data)

        self._log(f"  - Status: {decision.status.value}")
        self._log(f"  - Score: {decision.overall_score:.1%}")
        self._log(f"  - Reason: {decision.reason}")

        return ProcessingResult(
            application=application,
            classifications=classifications,
            extractions=extractions,
            fraud_results=fraud_results,
            validation_result=validation_result,
            decision=decision,
        )

    def _calculate_age(self, dob: str) -> int:
        """Calculate age from date of birth (YYYY-MM-DD)."""
        from datetime import datetime
        try:
            birth = datetime.strptime(dob, "%Y-%m-%d")
            today = datetime.now()
            age = today.year - birth.year
            if (today.month, today.day) < (birth.month, birth.day):
                age -= 1
            return age
        except ValueError:
            return 30  # Default age

    def _calculate_employment_months(self, start_date: str) -> int:
        """Calculate months of employment."""
        from datetime import datetime
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            today = datetime.now()
            months = (today.year - start.year) * 12 + (today.month - start.month)
            return max(0, months)
        except ValueError:
            return 24  # Default

    def _calculate_fraud_score(self, fraud_results: dict[str, FraudResult]) -> float:
        """Calculate overall fraud score (1.0 = all valid)."""
        if not fraud_results:
            return 1.0
        valid_count = sum(1 for r in fraud_results.values() if r.is_valid)
        return valid_count / len(fraud_results)


def print_report(result: ProcessingResult):
    """Print a comprehensive processing report."""
    print("\n" + "="*70)
    print("LOAN APPLICATION PROCESSING REPORT")
    print("="*70)

    # Application summary
    print(f"\nApplicant: {result.application.applicant_name}")
    print(f"Loan Amount: ${result.application.loan_amount:,.2f}")
    print(f"Purpose: {result.application.loan_purpose}")

    # Document classifications
    print(f"\n{'─'*70}")
    print("DOCUMENT CLASSIFICATIONS")
    print(f"{'─'*70}")
    for name, doc_type in result.classifications.items():
        print(f"  {name}: {doc_type}")

    # Fraud detection summary
    print(f"\n{'─'*70}")
    print("FRAUD DETECTION RESULTS")
    print(f"{'─'*70}")
    for name, fraud in result.fraud_results.items():
        status = "✓ Valid" if fraud.is_valid else f"✗ {fraud.severity.value}"
        print(f"  {name}: {status} (confidence: {fraud.confidence:.1%})")
        if fraud.issues:
            for issue in fraud.issues[:2]:  # Show first 2 issues
                print(f"    - {issue}")

    # Cross-validation summary
    print(f"\n{'─'*70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'─'*70}")
    print(f"  Overall Score: {result.validation_result.overall_score:.1%}")
    print(f"  Status: {'✓ Valid' if result.validation_result.is_valid else '✗ Invalid'}")
    if result.validation_result.discrepancies:
        print(f"  Discrepancies ({len(result.validation_result.discrepancies)}):")
        for disc in result.validation_result.discrepancies[:3]:  # Show first 3
            print(f"    - {disc.field}: expected '{disc.expected_value}', "
                  f"found '{disc.found_value}' [{disc.severity.value}]")

    # Decision
    print(f"\n{'─'*70}")
    print("LOAN DECISION")
    print(f"{'─'*70}")

    status_colors = {
        DecisionStatus.APPROVED: "\033[92m",      # Green
        DecisionStatus.REJECTED: "\033[91m",      # Red
        DecisionStatus.PENDING_REVIEW: "\033[93m", # Yellow
        DecisionStatus.NEEDS_MORE_INFO: "\033[94m", # Blue
    }
    reset = "\033[0m"

    color = status_colors.get(result.decision.status, "")
    print(f"  Status: {color}{result.decision.status.value}{reset}")
    print(f"  Score: {result.decision.overall_score:.1%}")
    print(f"  Reason: {result.decision.reason}")

    print(f"\n  Criteria Results:")
    for name, criteria in result.decision.criteria_results.items():
        passed = "✓" if criteria.passed else "✗"
        print(f"    {passed} {name}: {criteria.score:.1%} (weight: {criteria.weight:.0%})")

    print("\n" + "="*70)


def create_sample_application() -> LoanApplication:
    """Create a sample loan application for testing."""
    return LoanApplication(
        applicant_name="John Michael Doe",
        date_of_birth="1985-03-15",
        ssn_last_4="1234",
        annual_income=85000.0,
        employer="Acme Corporation",
        employment_start_date="2020-01-15",
        loan_amount=25000.0,
        loan_purpose="Home Improvement",
        address="123 Main Street, Springfield, IL 62701",
        documents={},  # Will be populated from command line
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process a complete loan application"
    )
    parser.add_argument(
        "--documents-dir",
        "-d",
        help="Directory containing document images"
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        help="Individual documents in format: name=path"
    )
    parser.add_argument(
        "--application",
        "-a",
        help="JSON file with application data"
    )
    parser.add_argument(
        "--provider",
        choices=["watsonx", "ollama"],
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    args = parser.parse_args()

    # Build document list
    documents = {}

    if args.documents_dir:
        doc_dir = Path(args.documents_dir)
        if not doc_dir.exists():
            print(f"Error: Directory not found: {args.documents_dir}")
            sys.exit(1)
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"]:
            for path in doc_dir.glob(ext):
                documents[path.stem] = str(path)

    if args.documents:
        for doc_spec in args.documents:
            if "=" not in doc_spec:
                print(f"Error: Document format should be name=path, got: {doc_spec}")
                sys.exit(1)
            name, path = doc_spec.split("=", 1)
            if not os.path.exists(path):
                print(f"Error: Document not found: {path}")
                sys.exit(1)
            documents[name] = path

    if not documents:
        print("Error: No documents provided. Use --documents-dir or --documents")
        print("\nExample:")
        print("  python complete_workflow.py -d ./loan_docs/")
        print("  python complete_workflow.py --documents id=passport.png income=payslip.png")
        sys.exit(1)

    # Load or create application data
    if args.application:
        with open(args.application) as f:
            app_data = json.load(f)
        application = LoanApplication(
            applicant_name=app_data.get("applicant_name", "Unknown"),
            date_of_birth=app_data.get("date_of_birth", "1990-01-01"),
            ssn_last_4=app_data.get("ssn_last_4", "0000"),
            annual_income=app_data.get("annual_income", 50000),
            employer=app_data.get("employer", "Unknown"),
            employment_start_date=app_data.get("employment_start_date", "2020-01-01"),
            loan_amount=app_data.get("loan_amount", 10000),
            loan_purpose=app_data.get("loan_purpose", "Personal"),
            address=app_data.get("address", "Unknown"),
            documents=documents,
        )
    else:
        application = create_sample_application()
        application.documents = documents

    # Process the application
    try:
        processor = LoanProcessor(
            provider=args.provider,
            verbose=not args.quiet
        )
        result = processor.process(application)

        # Print report
        print_report(result)

        # Save to JSON if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error processing application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
