#!/usr/bin/env python3
"""
Cross-Validation Example

Demonstrates how to validate consistency across multiple documents.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from watsonx_vision import (
    VisionLLM,
    VisionLLMConfig,
    LLMProvider,
    CrossValidator,
    ValidationResult,
)
from watsonx_vision.cross_validator import FieldSeverity, FinancialCrossValidator


def create_llm(provider: str = "ollama") -> VisionLLM:
    """Create VisionLLM instance based on provider."""
    if provider == "watsonx":
        config = VisionLLMConfig(
            provider=LLMProvider.WATSONX,
            model_id=os.getenv("VISION_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"),
            api_key=os.getenv("WATSONX_API_KEY"),
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        )
    else:
        config = VisionLLMConfig(
            provider=LLMProvider.OLLAMA,
            model_id=os.getenv("VISION_MODEL", "llava"),
            url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        )
    return VisionLLM(config)


def validate_documents(
    application_data: dict,
    documents: dict[str, str],
    provider: str = "ollama"
) -> ValidationResult:
    """
    Validate documents against application data.

    Args:
        application_data: Dict of claimed information (e.g., name, income)
        documents: Dict mapping document names to image paths
        provider: LLM provider

    Returns:
        ValidationResult with discrepancies
    """
    llm = create_llm(provider)
    validator = CrossValidator(llm)

    # Encode all documents
    encoded_docs = {}
    for name, path in documents.items():
        encoded_docs[name] = VisionLLM.encode_image_to_base64(path)

    return validator.validate(application_data, encoded_docs)


def validate_financial_documents(
    application_data: dict,
    documents: dict[str, str],
    provider: str = "ollama"
) -> ValidationResult:
    """
    Validate financial documents with specialized rules.

    Uses FinancialCrossValidator for stricter income/employment validation.

    Args:
        application_data: Dict with financial claims
        documents: Dict mapping document names to paths
        provider: LLM provider

    Returns:
        ValidationResult
    """
    llm = create_llm(provider)
    validator = FinancialCrossValidator(llm)

    encoded_docs = {}
    for name, path in documents.items():
        encoded_docs[name] = VisionLLM.encode_image_to_base64(path)

    return validator.validate(application_data, encoded_docs)


def validate_with_custom_fields(
    application_data: dict,
    documents: dict[str, str],
    field_severity: dict[str, str],
    provider: str = "ollama"
) -> ValidationResult:
    """
    Validate with custom field severity mappings.

    Args:
        application_data: Dict of claimed information
        documents: Dict mapping document names to paths
        field_severity: Dict mapping field names to severity levels
        provider: LLM provider

    Returns:
        ValidationResult
    """
    llm = create_llm(provider)

    # Convert string severity to FieldSeverity enum
    severity_map = {}
    for field, level in field_severity.items():
        severity_map[field] = FieldSeverity(level.upper())

    validator = CrossValidator(llm, field_severity=severity_map)

    encoded_docs = {}
    for name, path in documents.items():
        encoded_docs[name] = VisionLLM.encode_image_to_base64(path)

    return validator.validate(application_data, encoded_docs)


def print_result(result: ValidationResult):
    """Pretty print validation result."""
    severity_colors = {
        FieldSeverity.CRITICAL: "\033[91m",  # Red
        FieldSeverity.HIGH: "\033[91m",      # Red
        FieldSeverity.MEDIUM: "\033[93m",    # Yellow
        FieldSeverity.LOW: "\033[94m",       # Blue
    }
    reset = "\033[0m"

    status_color = "\033[92m" if result.is_valid else "\033[91m"
    status = "VALID" if result.is_valid else "INVALID"

    print(f"\n{'='*60}")
    print(f"Cross-Validation Result")
    print(f"{'='*60}")
    print(f"Status: {status_color}{status}{reset}")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Discrepancies: {len(result.discrepancies)}")

    if result.discrepancies:
        print(f"\nDiscrepancies Found:")
        print("-" * 60)
        for disc in result.discrepancies:
            color = severity_colors.get(disc.severity, "")
            print(f"\n  Field: {disc.field}")
            print(f"  Severity: {color}{disc.severity.value}{reset}")
            print(f"  Expected: {disc.expected_value}")
            print(f"  Found: {disc.found_value}")
            print(f"  Source: {disc.source_document}")
            if disc.explanation:
                print(f"  Explanation: {disc.explanation}")

    # Generate human-readable report
    llm = create_llm()
    validator = CrossValidator(llm)
    report = validator.generate_report(result)
    print(f"\n{'='*60}")
    print("Human-Readable Report")
    print(f"{'='*60}")
    print(report)


# Example application data templates
LOAN_APPLICATION_TEMPLATE = {
    "applicant_name": "John Doe",
    "ssn_last_4": "1234",
    "date_of_birth": "1985-03-15",
    "annual_income": 85000,
    "employer": "Acme Corporation",
    "employment_start_date": "2020-01-15",
    "address": "123 Main St, Springfield, IL 62701",
}


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate documents against application data"
    )
    parser.add_argument(
        "--documents",
        "-d",
        nargs="+",
        required=True,
        help="Document images in format: name=path (e.g., passport=./passport.png)"
    )
    parser.add_argument(
        "--application",
        "-a",
        help="JSON file with application data"
    )
    parser.add_argument(
        "--name",
        help="Applicant name"
    )
    parser.add_argument(
        "--income",
        type=float,
        help="Annual income"
    )
    parser.add_argument(
        "--employer",
        help="Employer name"
    )
    parser.add_argument(
        "--dob",
        help="Date of birth (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--provider",
        choices=["watsonx", "ollama"],
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--financial",
        action="store_true",
        help="Use financial document validator"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file"
    )
    args = parser.parse_args()

    # Parse documents
    documents = {}
    for doc_spec in args.documents:
        if "=" not in doc_spec:
            print(f"Error: Document format should be name=path, got: {doc_spec}")
            sys.exit(1)
        name, path = doc_spec.split("=", 1)
        if not os.path.exists(path):
            print(f"Error: Document not found: {path}")
            sys.exit(1)
        documents[name] = path

    # Build application data
    if args.application:
        with open(args.application) as f:
            application_data = json.load(f)
    else:
        application_data = {}
        if args.name:
            application_data["applicant_name"] = args.name
        if args.income:
            application_data["annual_income"] = args.income
        if args.employer:
            application_data["employer"] = args.employer
        if args.dob:
            application_data["date_of_birth"] = args.dob

    if not application_data:
        print("Error: Provide application data via --application file or individual flags")
        print("\nExample:")
        print('  python cross_validation.py -d passport=./id.png salary=./payslip.png \\')
        print('    --name "John Doe" --income 85000 --employer "Acme Corp"')
        sys.exit(1)

    print(f"Provider: {args.provider}")
    print(f"Documents: {list(documents.keys())}")
    print(f"Application Data: {json.dumps(application_data, indent=2)}")

    try:
        if args.financial:
            result = validate_financial_documents(
                application_data, documents, args.provider
            )
        else:
            result = validate_documents(
                application_data, documents, args.provider
            )

        print_result(result)

        if args.output:
            output_data = {
                "is_valid": result.is_valid,
                "overall_score": result.overall_score,
                "discrepancies": [
                    {
                        "field": d.field,
                        "severity": d.severity.value,
                        "expected_value": d.expected_value,
                        "found_value": d.found_value,
                        "source_document": d.source_document,
                        "explanation": d.explanation,
                    }
                    for d in result.discrepancies
                ],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
