#!/usr/bin/env python3
"""
Basic Document Classification Example

Demonstrates how to classify documents into predefined types using VisionLLM.
"""

import argparse
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


def classify_document(image_path: str, provider: str = "ollama") -> dict:
    """
    Classify a document image.

    Args:
        image_path: Path to the document image
        provider: LLM provider ("watsonx" or "ollama")

    Returns:
        Classification result with doc_type
    """
    llm = create_llm(provider)

    # Encode image to base64
    image_data = VisionLLM.encode_image_to_base64(image_path)

    # Classify with default document types
    result = llm.classify_document(image_data)

    return result


def classify_with_custom_types(
    image_path: str,
    document_types: list[str],
    provider: str = "ollama"
) -> dict:
    """
    Classify a document with custom document types.

    Args:
        image_path: Path to the document image
        document_types: List of possible document types
        provider: LLM provider

    Returns:
        Classification result
    """
    llm = create_llm(provider)
    image_data = VisionLLM.encode_image_to_base64(image_path)

    result = llm.classify_document(image_data, document_types=document_types)

    return result


def main():
    parser = argparse.ArgumentParser(description="Classify a document image")
    parser.add_argument("image", help="Path to document image")
    parser.add_argument(
        "--provider",
        choices=["watsonx", "ollama"],
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        help="Custom document types (optional)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    print(f"Classifying: {args.image}")
    print(f"Provider: {args.provider}")

    try:
        if args.types:
            print(f"Custom types: {args.types}")
            result = classify_with_custom_types(args.image, args.types, args.provider)
        else:
            result = classify_document(args.image, args.provider)

        print(f"\nResult: {result['doc_type']}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
