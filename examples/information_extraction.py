#!/usr/bin/env python3
"""
Information Extraction Example

Demonstrates how to extract structured information from document images.
"""

import argparse
import json
import os
import sys

from watsonx_vision import VisionLLM, VisionLLMConfig, LLMProvider


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


def extract_information(image_path: str, provider: str = "ollama") -> dict:
    """
    Extract information from a document using default fields.

    Default fields include: Name, Address, DOB, Gender, Document Number,
    Nationality, SSN, EIN, Revenue, Expenses, Net Income, etc.

    Args:
        image_path: Path to document image
        provider: LLM provider

    Returns:
        Dict with extracted fields
    """
    llm = create_llm(provider)
    image_data = VisionLLM.encode_image_to_base64(image_path)

    return llm.extract_information(image_data)


def extract_custom_fields(
    image_path: str,
    fields: list[str],
    date_format: str = "YYYY-MM-DD",
    provider: str = "ollama"
) -> dict:
    """
    Extract specific fields from a document.

    Args:
        image_path: Path to document image
        fields: List of field names to extract
        date_format: Expected date format in output
        provider: LLM provider

    Returns:
        Dict with extracted fields
    """
    llm = create_llm(provider)
    image_data = VisionLLM.encode_image_to_base64(image_path)

    return llm.extract_information(
        image_data,
        fields=fields,
        date_format=date_format
    )


# Preset field configurations for common document types
PASSPORT_FIELDS = [
    "Full Name",
    "Date of Birth",
    "Place of Birth",
    "Nationality",
    "Passport Number",
    "Date of Issue",
    "Expiry Date",
    "Issuing Authority",
    "Gender",
]

DRIVERS_LICENSE_FIELDS = [
    "Full Name",
    "Address",
    "Date of Birth",
    "License Number",
    "Class",
    "Expiry Date",
    "Issue Date",
    "Restrictions",
    "Height",
    "Eye Color",
]

BANK_STATEMENT_FIELDS = [
    "Account Holder Name",
    "Account Number",
    "Bank Name",
    "Statement Period",
    "Opening Balance",
    "Closing Balance",
    "Total Deposits",
    "Total Withdrawals",
]

TAX_RETURN_FIELDS = [
    "Taxpayer Name",
    "SSN (last 4 digits)",
    "Filing Status",
    "Tax Year",
    "Total Income",
    "Adjusted Gross Income",
    "Taxable Income",
    "Total Tax",
    "Refund Amount",
]


def main():
    parser = argparse.ArgumentParser(description="Extract information from a document")
    parser.add_argument("image", help="Path to document image")
    parser.add_argument(
        "--provider",
        choices=["watsonx", "ollama"],
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--preset",
        choices=["passport", "license", "bank", "tax"],
        help="Use preset field configuration"
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Custom fields to extract"
    )
    parser.add_argument(
        "--date-format",
        default="YYYY-MM-DD",
        help="Date format for output (default: YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file (optional)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Determine fields to extract
    fields = None
    if args.preset:
        presets = {
            "passport": PASSPORT_FIELDS,
            "license": DRIVERS_LICENSE_FIELDS,
            "bank": BANK_STATEMENT_FIELDS,
            "tax": TAX_RETURN_FIELDS,
        }
        fields = presets[args.preset]
        print(f"Using preset: {args.preset}")
    elif args.fields:
        fields = args.fields
        print(f"Custom fields: {fields}")

    print(f"Extracting from: {args.image}")
    print(f"Provider: {args.provider}")

    try:
        if fields:
            result = extract_custom_fields(
                args.image,
                fields,
                args.date_format,
                args.provider
            )
        else:
            result = extract_information(args.image, args.provider)

        # Pretty print results
        print("\nExtracted Information:")
        print("-" * 40)
        for key, value in result.items():
            print(f"  {key}: {value}")

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
