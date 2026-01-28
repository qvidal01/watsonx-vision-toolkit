#!/usr/bin/env python3
"""
Fraud Detection Example

Demonstrates how to detect document fraud and manipulation using FraudDetector.
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
    FraudDetector,
    FraudResult,
)
from watsonx_vision.fraud_detector import FraudSeverity, SpecializedFraudDetector


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


def check_single_document(
    image_path: str,
    provider: str = "ollama"
) -> FraudResult:
    """
    Check a single document for fraud indicators.

    Args:
        image_path: Path to document image
        provider: LLM provider

    Returns:
        FraudResult with validation details
    """
    llm = create_llm(provider)
    detector = FraudDetector(llm)

    image_data = VisionLLM.encode_image_to_base64(image_path)
    return detector.validate_document(image_data)


def check_batch_documents(
    image_paths: list[str],
    provider: str = "ollama"
) -> list[FraudResult]:
    """
    Check multiple documents for fraud.

    Args:
        image_paths: List of paths to document images
        provider: LLM provider

    Returns:
        List of FraudResult objects
    """
    llm = create_llm(provider)
    detector = FraudDetector(llm)

    documents = {}
    for path in image_paths:
        doc_name = Path(path).stem
        documents[doc_name] = VisionLLM.encode_image_to_base64(path)

    return detector.validate_batch(documents)


def check_specialized_document(
    image_path: str,
    doc_type: str,
    provider: str = "ollama"
) -> FraudResult:
    """
    Check a document using specialized validation rules.

    Args:
        image_path: Path to document image
        doc_type: Document type (passport, license, bank_statement, tax_return)
        provider: LLM provider

    Returns:
        FraudResult with specialized validation
    """
    llm = create_llm(provider)
    detector = SpecializedFraudDetector(llm)

    image_data = VisionLLM.encode_image_to_base64(image_path)

    validators = {
        "passport": detector.validate_passport,
        "license": detector.validate_drivers_license,
        "bank_statement": detector.validate_bank_statement,
        "tax_return": detector.validate_tax_return,
    }

    if doc_type not in validators:
        raise ValueError(f"Unknown document type: {doc_type}. Use: {list(validators.keys())}")

    return validators[doc_type](image_data)


def print_result(result: FraudResult, doc_name: str = "Document"):
    """Pretty print a fraud detection result."""
    severity_colors = {
        FraudSeverity.NONE: "\033[92m",      # Green
        FraudSeverity.LOW: "\033[93m",       # Yellow
        FraudSeverity.MEDIUM: "\033[93m",    # Yellow
        FraudSeverity.HIGH: "\033[91m",      # Red
        FraudSeverity.CRITICAL: "\033[91m",  # Red
    }
    reset = "\033[0m"

    color = severity_colors.get(result.severity, "")
    status = "VALID" if result.is_valid else "SUSPICIOUS"

    print(f"\n{'='*50}")
    print(f"{doc_name}")
    print(f"{'='*50}")
    print(f"Status: {color}{status}{reset}")
    print(f"Severity: {color}{result.severity.value}{reset}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Reason: {result.reason}")

    if result.issues:
        print(f"\nIssues Found ({len(result.issues)}):")
        for issue in result.issues:
            print(f"  - {issue}")

    if result.details:
        print(f"\nDetails:")
        print(f"  Layout Score: {result.details.get('layout_score', 'N/A')}")
        print(f"  Field Score: {result.details.get('field_score', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Detect document fraud")
    parser.add_argument("images", nargs="+", help="Path(s) to document image(s)")
    parser.add_argument(
        "--provider",
        choices=["watsonx", "ollama"],
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--type",
        choices=["passport", "license", "bank_statement", "tax_return"],
        help="Use specialized validator for document type"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary report for batch processing"
    )
    args = parser.parse_args()

    # Validate all images exist
    for image in args.images:
        if not os.path.exists(image):
            print(f"Error: Image not found: {image}")
            sys.exit(1)

    print(f"Provider: {args.provider}")
    print(f"Documents: {len(args.images)}")

    try:
        results = []

        if len(args.images) == 1 and args.type:
            # Single document with specialized validator
            result = check_specialized_document(args.images[0], args.type, args.provider)
            results.append(result)
            print_result(result, Path(args.images[0]).name)

        elif len(args.images) == 1:
            # Single document with general validator
            result = check_single_document(args.images[0], args.provider)
            results.append(result)
            print_result(result, Path(args.images[0]).name)

        else:
            # Batch processing
            results = check_batch_documents(args.images, args.provider)
            for path, result in zip(args.images, results):
                print_result(result, Path(path).name)

        # Print summary if requested
        if args.summary and len(results) > 1:
            llm = create_llm(args.provider)
            detector = FraudDetector(llm)
            summary = detector.generate_summary_report(results)
            print(f"\n{'='*50}")
            print("SUMMARY REPORT")
            print(f"{'='*50}")
            print(summary)

        # Save to JSON if requested
        if args.output:
            output_data = []
            for path, result in zip(args.images, results):
                output_data.append({
                    "document": Path(path).name,
                    "is_valid": result.is_valid,
                    "severity": result.severity.value,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "issues": result.issues,
                    "details": result.details,
                })
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
