# FraudDetector

Document fraud detection using vision-based analysis.

## FraudDetector

### Constructor

```python
from watsonx_vision import VisionLLM, FraudDetector

llm = VisionLLM(config)
detector = FraudDetector(
    vision_llm=llm,
    layout_threshold=70,
    field_threshold=70,
    min_confidence=60
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vision_llm` | `VisionLLM` | Required | VisionLLM instance |
| `layout_threshold` | `int` | `70` | Minimum layout score for validity |
| `field_threshold` | `int` | `70` | Minimum field score for validity |
| `min_confidence` | `int` | `60` | Minimum confidence for valid result |

---

### validate_document

Validate document authenticity.

```python
result = detector.validate_document(
    image_data="data:image/png;base64,...",
    filename="passport.png",
    document_type="passport"
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_data` | `str` | Yes | Base64-encoded image data URI |
| `filename` | `str \| None` | No | Filename for tracking |
| `document_type` | `str \| None` | No | Document type hint |

**Returns:** `FraudResult`

**Raises:**

- `DocumentAnalysisError`: If analysis fails

**Async variant:** `validate_document_async()`

---

### validate_batch

Validate multiple documents.

```python
documents = [
    {"image_data": img1, "filename": "doc1.png"},
    {"image_data": img2, "filename": "doc2.png", "doc_type": "passport"}
]
results = detector.validate_batch(documents)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `documents` | `List[Dict]` | List of document dicts |

Each document dict:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `image_data` | `str` | Yes | Base64-encoded image |
| `filename` | `str` | No | Filename for tracking |
| `doc_type` | `str` | No | Document type hint |

**Returns:** `List[FraudResult]`

**Async variant:** `validate_batch_async(documents, concurrent=False)`

---

### generate_report

Generate summary report from validation results.

```python
report = detector.generate_report(results)
```

**Returns:** `Dict` with:

```python
{
    "total_documents": 3,
    "valid_documents": 2,
    "invalid_documents": 1,
    "fraud_rate": 33.33,
    "average_confidence": 75.0,
    "severity_breakdown": {"none": 1, "low": 1, "high": 1},
    "details": [...]
}
```

---

## SpecializedFraudDetector

Extends `FraudDetector` with document-type-specific validation prompts.

```python
from watsonx_vision.fraud_detector import SpecializedFraudDetector

detector = SpecializedFraudDetector(vision_llm=llm)
result = detector.validate_document(
    image_data="...",
    document_type="tax_return"
)
```

**Supported Document Types:**

| Type | Validation Focus |
|------|------------------|
| `tax_return` | IRS formatting, form numbers, signatures |
| `bank_statement` | Bank logos, account formatting, transaction layout |
| `passport` | MRZ codes, photo integration, security features |
| `drivers_license` | State formatting, photo quality, holograms |

---

## FraudResult

Dataclass for fraud detection results.

```python
@dataclass
class FraudResult:
    valid: bool              # Document appears authentic
    confidence: int          # 0-100 confidence score
    reason: str              # Explanation
    layout_score: int        # 0-100 layout consistency
    field_score: int         # 0-100 field formatting
    forgery_signs: List[str] # Issues found
    severity: FraudSeverity  # Overall severity level
    filename: Optional[str]  # Original filename
```

### Methods

#### to_dict

Convert to dictionary.

```python
data = result.to_dict()
```

---

## FraudSeverity Enum

```python
from watsonx_vision.fraud_detector import FraudSeverity

class FraudSeverity(Enum):
    NONE = "none"        # Valid, high confidence
    LOW = "low"          # Valid, moderate confidence
    MEDIUM = "medium"    # Some concerns
    HIGH = "high"        # Significant issues
    CRITICAL = "critical" # Definite fraud indicators
```

### Severity Calculation

| Condition | Severity |
|-----------|----------|
| Valid + confidence ≥ 80 | `NONE` |
| Valid + confidence ≥ 60 | `LOW` |
| Invalid + confidence ≥ 70 | `MEDIUM` |
| Invalid + confidence ≥ 50 | `HIGH` |
| Invalid + confidence < 50 | `CRITICAL` |

---

## Example

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, FraudDetector

# Setup
config = VisionLLMConfig.from_env()
llm = VisionLLM(config)
detector = FraudDetector(llm, layout_threshold=75, field_threshold=75)

# Single document
image = VisionLLM.encode_image_to_base64("passport.png")
result = detector.validate_document(image, "passport.png", "passport")

print(f"Valid: {result.valid}")
print(f"Confidence: {result.confidence}%")
print(f"Severity: {result.severity.value}")

if not result.valid:
    print(f"Reason: {result.reason}")
    print(f"Issues: {result.forgery_signs}")

# Batch processing
documents = [
    {"image_data": img1, "filename": "doc1.png"},
    {"image_data": img2, "filename": "doc2.png"},
]
results = detector.validate_batch(documents)

# Generate report
report = detector.generate_report(results)
print(f"Fraud rate: {report['fraud_rate']}%")
```
