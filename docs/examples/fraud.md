# Fraud Detection

Detect document fraud and manipulation using vision analysis.

## Basic Fraud Detection

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, FraudDetector

# Setup
config = VisionLLMConfig.from_env()
llm = VisionLLM(config)
detector = FraudDetector(llm)

# Validate a document
image_data = VisionLLM.encode_image_to_base64("passport.png")
result = detector.validate_document(image_data, filename="passport.png")

if result.valid:
    print(f"Document appears authentic (confidence: {result.confidence}%)")
else:
    print(f"Fraud detected: {result.reason}")
    print(f"Severity: {result.severity.value}")
    print(f"Issues: {result.forgery_signs}")
```

## FraudResult Properties

| Property | Type | Description |
|----------|------|-------------|
| `valid` | `bool` | Document appears authentic |
| `confidence` | `int` | 0-100 confidence score |
| `reason` | `str` | Explanation |
| `layout_score` | `int` | Layout consistency (0-100) |
| `field_score` | `int` | Field formatting (0-100) |
| `forgery_signs` | `List[str]` | Detected issues |
| `severity` | `FraudSeverity` | Overall severity |
| `filename` | `str` | Original filename |

## Configure Thresholds

Adjust sensitivity:

```python
# Stricter validation (higher thresholds)
strict_detector = FraudDetector(
    vision_llm=llm,
    layout_threshold=80,  # Default: 70
    field_threshold=80,   # Default: 70
    min_confidence=70     # Default: 60
)

# More lenient validation
lenient_detector = FraudDetector(
    vision_llm=llm,
    layout_threshold=60,
    field_threshold=60,
    min_confidence=50
)
```

## Batch Processing

Validate multiple documents:

```python
documents = [
    {"image_data": img1, "filename": "passport.png"},
    {"image_data": img2, "filename": "license.jpg"},
    {"image_data": img3, "filename": "tax_return.pdf", "doc_type": "tax_return"}
]

results = detector.validate_batch(documents)

for result in results:
    status = "PASS" if result.valid else "FAIL"
    print(f"{result.filename}: {status} ({result.confidence}%)")
```

## Generate Reports

```python
# Validate batch
results = detector.validate_batch(documents)

# Generate summary report
report = detector.generate_report(results)

print(f"Total documents: {report['total_documents']}")
print(f"Valid: {report['valid_documents']}")
print(f"Invalid: {report['invalid_documents']}")
print(f"Fraud rate: {report['fraud_rate']}%")
print(f"Average confidence: {report['average_confidence']}%")
print(f"Severity breakdown: {report['severity_breakdown']}")
```

## Specialized Detection

Use document-specific validation:

```python
from watsonx_vision.fraud_detector import SpecializedFraudDetector

detector = SpecializedFraudDetector(vision_llm=llm)

# Tax return validation
result = detector.validate_document(
    image_data,
    document_type="tax_return"
)

# Bank statement validation
result = detector.validate_document(
    image_data,
    document_type="bank_statement"
)

# Passport validation
result = detector.validate_document(
    image_data,
    document_type="passport"
)

# Driver's license validation
result = detector.validate_document(
    image_data,
    document_type="drivers_license"
)
```

## Async Processing

```python
import asyncio

async def validate_documents(paths):
    """Validate documents concurrently."""
    async def validate_one(path):
        image_data = VisionLLM.encode_image_to_base64(path)
        return await detector.validate_document_async(image_data, path)

    tasks = [validate_one(p) for p in paths]
    return await asyncio.gather(*tasks)

# Run concurrent validation
results = asyncio.run(validate_documents([
    "doc1.pdf", "doc2.pdf", "doc3.pdf"
]))

# Or use batch async
documents = [{"image_data": img, "filename": name} for img, name in ...]
results = await detector.validate_batch_async(documents, concurrent=True)
```

## Severity Levels

| Severity | Meaning | Action |
|----------|---------|--------|
| `NONE` | Valid, high confidence | Accept |
| `LOW` | Valid, moderate confidence | Accept with note |
| `MEDIUM` | Some concerns | Manual review |
| `HIGH` | Significant issues | Reject or escalate |
| `CRITICAL` | Definite fraud indicators | Reject and flag |

## Error Handling

```python
from watsonx_vision import DocumentAnalysisError, WatsonxVisionError

def safe_validate(image_path):
    """Validate with error handling."""
    try:
        image_data = VisionLLM.encode_image_to_base64(image_path)
        return detector.validate_document(image_data, image_path)

    except DocumentAnalysisError as e:
        print(f"Analysis failed for {image_path}: {e.message}")
        return None

    except WatsonxVisionError as e:
        print(f"Error validating {image_path}: {e.message}")
        return None
```

## CLI Fraud Detection

```bash
# Single document
watsonx-vision fraud passport.png

# Multiple documents
watsonx-vision fraud doc1.png doc2.jpg doc3.pdf

# With document type hint
watsonx-vision fraud tax_return.pdf --doc-type tax_return

# Custom threshold
watsonx-vision fraud invoice.png --threshold 80

# JSON output
watsonx-vision fraud *.pdf --output json --output-file results.json
```

## Complete Example

```python
#!/usr/bin/env python3
"""Fraud detection pipeline example."""

from pathlib import Path
from watsonx_vision import VisionLLM, VisionLLMConfig, FraudDetector
from watsonx_vision.fraud_detector import SpecializedFraudDetector

def main():
    # Setup
    config = VisionLLMConfig.from_env()
    llm = VisionLLM(config)
    detector = SpecializedFraudDetector(llm)

    # Document types for specialized validation
    DOC_TYPES = {
        "passport": "passport",
        "license": "drivers_license",
        "tax": "tax_return",
        "bank": "bank_statement"
    }

    # Process documents
    documents_dir = Path("documents/")
    results = []

    for doc_path in documents_dir.glob("*.*"):
        # Determine document type from filename
        doc_type = None
        for key, value in DOC_TYPES.items():
            if key in doc_path.stem.lower():
                doc_type = value
                break

        # Validate
        print(f"Validating: {doc_path.name}")
        image_data = VisionLLM.encode_image_to_base64(str(doc_path))
        result = detector.validate_document(
            image_data,
            filename=doc_path.name,
            document_type=doc_type
        )
        results.append(result)

        # Print result
        status = "✓ PASS" if result.valid else "✗ FAIL"
        print(f"  {status} (confidence: {result.confidence}%, severity: {result.severity.value})")
        if not result.valid:
            print(f"  Reason: {result.reason}")
            for sign in result.forgery_signs:
                print(f"    - {sign}")

    # Generate report
    print("\n" + "="*50)
    report = detector.generate_report(results)
    print(f"Summary:")
    print(f"  Total: {report['total_documents']}")
    print(f"  Valid: {report['valid_documents']}")
    print(f"  Invalid: {report['invalid_documents']}")
    print(f"  Fraud Rate: {report['fraud_rate']:.1f}%")

if __name__ == "__main__":
    main()
```
