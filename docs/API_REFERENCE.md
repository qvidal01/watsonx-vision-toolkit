# Watsonx Vision Toolkit - API Reference

Complete API documentation for all public classes, methods, and exceptions.

---

## Table of Contents

- [VisionLLM](#visionllm)
- [FraudDetector](#frauddetector)
- [CrossValidator](#crossvalidator)
- [DecisionEngine](#decisionengine)
- [Exceptions](#exceptions)
- [Data Classes](#data-classes)
- [Enums](#enums)

---

## VisionLLM

Vision-based LLM for document analysis. Supports multiple providers with a unified interface.

### Class: `VisionLLMConfig`

Configuration dataclass for VisionLLM.

```python
from watsonx_vision import VisionLLMConfig, LLMProvider

config = VisionLLMConfig(
    provider=LLMProvider.WATSONX,  # or LLMProvider.OLLAMA
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    api_key="your-api-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project-id",
    max_tokens=2000,
    temperature=0.0,
    top_p=0.1
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `LLMProvider` | `WATSONX` | LLM provider to use |
| `model_id` | `str` | `"meta-llama/..."` | Model identifier |
| `api_key` | `str \| None` | `None` | API key for authentication |
| `url` | `str \| None` | `None` | Provider URL |
| `project_id` | `str \| None` | `None` | Project/workspace ID |
| `max_tokens` | `int` | `2000` | Maximum response tokens |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `top_p` | `float` | `0.1` | Nucleus sampling parameter |

### Class: `VisionLLM`

Main class for vision-based document analysis.

#### Constructor

```python
from watsonx_vision import VisionLLM, VisionLLMConfig

config = VisionLLMConfig(...)
llm = VisionLLM(config)
```

**Raises:**
- `ImportError`: If required provider packages are not installed
- `ConfigurationError`: If configuration is invalid

#### Method: `analyze_image`

Analyze an image with a custom prompt.

```python
result = llm.analyze_image(
    image_data="data:image/png;base64,...",
    prompt="Describe this document",
    system_prompt="You are a document analysis assistant",
    parse_json=True
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_data` | `str` | Yes | Base64-encoded image data URI |
| `prompt` | `str` | Yes | User prompt for analysis |
| `system_prompt` | `str \| None` | No | Optional system context |
| `parse_json` | `bool` | No | Parse response as JSON (default: `True`) |

**Returns:** `Dict` if `parse_json=True`, else `str`

**Raises:**
- `LLMConnectionError`: Connection to provider failed
- `LLMResponseError`: Invalid response from LLM
- `LLMParseError`: Failed to parse JSON response
- `LLMTimeoutError`: Request timed out

#### Method: `classify_document`

Classify a document image into predefined types.

```python
result = llm.classify_document(
    image_data="data:image/png;base64,...",
    document_types=["Passport", "Driver's License", "Tax Return"]
)
# Returns: {"doc_type": "Passport"}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_data` | `str` | Yes | Base64-encoded image data URI |
| `document_types` | `List[str] \| None` | No | Custom document types (defaults to common types) |

**Default Document Types:**
- Driving License, Passport, SSN, Utility Bill, Salary Slip
- ITR (Income Tax Return), Bank Account Statement, Tax Return
- Articles of Incorporation, Personal Financial Statement, Others

**Returns:** `Dict[str, str]` with `doc_type` key

#### Method: `extract_information`

Extract structured information from a document image.

```python
result = llm.extract_information(
    image_data="data:image/png;base64,...",
    fields=["Name", "Date of Birth", "Address"],
    date_format="YYYY-MM-DD"
)
# Returns: {"name": "John Doe", "dob": "1990-01-15", "address": "..."}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_data` | `str` | Yes | Base64-encoded image data URI |
| `fields` | `List[str] \| None` | No | Fields to extract (defaults to common PII) |
| `date_format` | `str` | No | Output date format (default: `"YYYY-MM-DD"`) |

**Returns:** `Dict[str, Any]` with extracted fields

#### Method: `validate_authenticity`

Validate document authenticity using vision analysis.

```python
result = llm.validate_authenticity(image_data="data:image/png;base64,...")
# Returns:
# {
#     "valid": True,
#     "reason": "Document appears authentic",
#     "layout_score": 95,
#     "field_score": 90,
#     "forgery_signs": []
# }
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_data` | `str` | Yes | Base64-encoded image data URI |

**Returns:** `Dict` with:
- `valid` (bool): Whether document appears authentic
- `reason` (str): Explanation
- `layout_score` (int): Layout consistency score (0-100)
- `field_score` (int): Field formatting score (0-100)
- `forgery_signs` (List[str]): Issues found

#### Static Method: `encode_image_to_base64`

Encode a local image file to base64 data URI.

```python
data_uri = VisionLLM.encode_image_to_base64(
    image_path="/path/to/document.png",
    mime_type="image/png"  # Optional, auto-detected
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | `str` | Yes | Path to image file |
| `mime_type` | `str \| None` | No | MIME type (auto-detected if not provided) |

**Returns:** `str` - Base64-encoded data URI

---

## FraudDetector

Document fraud detector using vision-based analysis.

### Class: `FraudDetector`

#### Constructor

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

#### Method: `validate_document`

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

#### Method: `validate_batch`

Validate multiple documents.

```python
documents = [
    {"image_data": img1, "filename": "doc1.png"},
    {"image_data": img2, "filename": "doc2.png", "doc_type": "passport"}
]
results = detector.validate_batch(documents)
```

**Returns:** `List[FraudResult]`

#### Method: `generate_report`

Generate summary report from validation results.

```python
report = detector.generate_report(results)
# Returns:
# {
#     "total_documents": 3,
#     "valid_documents": 2,
#     "invalid_documents": 1,
#     "fraud_rate": 33.33,
#     "average_confidence": 75.0,
#     "severity_breakdown": {"none": 1, "low": 1, "high": 1},
#     "details": [...]
# }
```

**Returns:** `Dict` with summary statistics

### Class: `SpecializedFraudDetector`

Extends `FraudDetector` with document-type-specific validation prompts.

**Supported Document Types:**
- `tax_return`
- `bank_statement`
- `passport`
- `drivers_license`

```python
detector = SpecializedFraudDetector(vision_llm=llm)
result = detector.validate_document(
    image_data="...",
    document_type="tax_return"
)
```

---

## CrossValidator

Cross-validation engine for multi-document consistency checking.

### Class: `CrossValidator`

#### Constructor

```python
from watsonx_vision import CrossValidator

validator = CrossValidator(
    api_key="...",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="...",
    model_id="mistralai/mistral-medium-2505",
    max_tokens=2000,
    temperature=0.0,
    llm=None  # Or provide pre-configured LLM
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | IBM Cloud API key |
| `url` | `str \| None` | `None` | Watsonx URL |
| `project_id` | `str \| None` | `None` | Project ID |
| `model_id` | `str` | `"mistralai/..."` | Text model for comparison |
| `max_tokens` | `int` | `2000` | Maximum response tokens |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `llm` | `Any \| None` | `None` | Pre-configured LLM instance |

#### Method: `validate`

Cross-validate application data against document data.

```python
result = validator.validate(
    application_data={
        "name": "John Doe",
        "dob": "1990-01-15",
        "ssn": "123-45-6789"
    },
    document_data=[
        {"doc_type": "Passport", "name": "John Doe", "dob": "1990-01-15"},
        {"doc_type": "Tax Return", "name": "John D. Doe", "ssn": "123-45-6789"}
    ],
    custom_fields={"custom_field": InconsistencySeverity.HIGH}
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `application_data` | `Dict[str, Any]` | Yes | Data from application form |
| `document_data` | `List[Dict]` | Yes | Extracted data from documents |
| `custom_fields` | `Dict[str, InconsistencySeverity] \| None` | No | Custom field severity mappings |

**Returns:** `ValidationResult`

**Raises:**
- `LLMConnectionError`: Connection failed
- `LLMResponseError`: Invalid response
- `LLMParseError`: Parse failed
- `ValidationError`: Validation logic error

#### Method: `validate_batch`

Validate multiple document packages.

```python
packages = [
    {
        "application_data": {...},
        "document_data": [{...}, {...}],
        "custom_fields": {...}  # Optional
    },
    ...
]
results = validator.validate_batch(packages)
```

**Returns:** `List[ValidationResult]`

#### Method: `generate_report`

Generate human-readable report from validation results.

```python
report = validator.generate_report(result, include_details=True)
print(report)
```

**Returns:** `str` - Formatted report with severity icons

### Class: `FinancialCrossValidator`

Extends `CrossValidator` with financial-specific field severities.

**Additional HIGH Severity Fields:**
- `revenue`, `gross_income`, `net_income`
- `total_assets`, `total_liabilities`, `net_worth`
- `annual_revenue`

**Additional MEDIUM Severity Fields:**
- `total_expenses`, `monthly_income`

#### Method: `validate_financials`

```python
result = financial_validator.validate_financials(
    tax_return_data={...},
    bank_statements=[{...}, {...}],
    pfs_data={...},  # Optional
    tolerance_percent=5.0
)
```

---

## DecisionEngine

Multi-criteria weighted decision making engine.

### Class: `DecisionEngine`

#### Constructor

```python
from watsonx_vision import DecisionEngine

engine = DecisionEngine(
    approval_threshold=80,
    rejection_threshold=40
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `approval_threshold` | `int` | `80` | Score threshold for approval |
| `rejection_threshold` | `int` | `40` | Score threshold for rejection |

#### Method: `add_criterion`

Add a custom evaluation criterion.

```python
engine.add_criterion(
    name="income_check",
    weight=30,
    evaluator=lambda data: data.get("income", 0) >= 50000
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Criterion identifier |
| `weight` | `int` | Weight (contribution to total score) |
| `evaluator` | `Callable[[Dict], bool]` | Function returning True/False |

#### Method: `decide`

Make a decision based on all criteria.

```python
decision = engine.decide(
    application_data={"name": "John", "income": 75000, "age": 35},
    fraud_result=fraud_result,
    validation_result=validation_result
)
```

**Returns:** `Decision`

### Class: `LoanDecisionEngine`

Pre-configured decision engine for loan applications.

**Built-in Criteria:**
- Age verification (≥18 years)
- Income requirement (≥$30,000)
- Debt-to-income ratio (≤43%)

```python
from watsonx_vision.decision_engine import LoanDecisionEngine

engine = LoanDecisionEngine()
decision = engine.decide(
    application_data={
        "dob": "1990-01-15",
        "annual_income": 75000,
        "monthly_debt": 1500
    },
    fraud_result=fraud_result,
    validation_result=validation_result
)
```

---

## Exceptions

All exceptions inherit from `WatsonxVisionError`.

### Exception Hierarchy

```
WatsonxVisionError (base)
├── LLMConnectionError      # Connection to LLM failed
├── LLMResponseError        # Invalid LLM response
├── LLMParseError           # JSON parsing failed
├── LLMTimeoutError         # Request timeout
├── DocumentAnalysisError   # Document analysis failed
├── ValidationError         # Validation logic error
└── ConfigurationError      # Invalid configuration
```

### Usage

```python
from watsonx_vision import (
    WatsonxVisionError,
    LLMConnectionError,
    LLMParseError,
    DocumentAnalysisError
)

try:
    result = llm.analyze_image(image_data, prompt)
except LLMConnectionError as e:
    print(f"Connection failed: {e.message}")
    print(f"Details: {e.details}")
except LLMParseError as e:
    print(f"Parse failed: {e.message}")
    # e.details contains raw_content and error
except WatsonxVisionError as e:
    print(f"General error: {e}")
```

### Exception Properties

All exceptions have:
- `message` (str): Human-readable error message
- `details` (Any | None): Additional context (dict, str, etc.)

---

## Data Classes

### `FraudResult`

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

    def to_dict(self) -> Dict: ...
```

### `ValidationResult`

```python
@dataclass
class ValidationResult:
    passed: bool                       # No critical/high issues
    total_inconsistencies: int         # Count of issues
    inconsistencies: List[Inconsistency]
    matched_fields: List[str]          # Fields that matched
    summary: str                       # Human-readable summary
    confidence: int                    # 0-100

    def to_dict(self) -> Dict: ...
```

### `Inconsistency`

```python
@dataclass
class Inconsistency:
    field: str                    # Field name
    source1: str                  # First source value
    source2: str                  # Second source value
    source1_doc: str              # First document name
    source2_doc: str              # Second document name
    severity: InconsistencySeverity
    explanation: str              # Human-readable explanation

    def to_dict(self) -> Dict: ...
```

### `Decision`

```python
@dataclass
class Decision:
    status: DecisionStatus      # APPROVED, REJECTED, etc.
    score: int                  # 0-100 overall score
    criteria_results: List[CriterionResult]
    reasons: List[str]          # Explanation list
    timestamp: datetime

    def to_dict(self) -> Dict: ...
    def summary(self) -> str: ...
```

---

## Enums

### `LLMProvider`

```python
class LLMProvider(Enum):
    WATSONX = "watsonx"
    OLLAMA = "ollama"
    OPENAI = "openai"      # Future
    ANTHROPIC = "anthropic" # Future
```

### `FraudSeverity`

```python
class FraudSeverity(Enum):
    NONE = "none"        # Valid, high confidence
    LOW = "low"          # Valid, moderate confidence
    MEDIUM = "medium"    # Some concerns
    HIGH = "high"        # Significant issues
    CRITICAL = "critical" # Definite fraud indicators
```

### `InconsistencySeverity`

```python
class InconsistencySeverity(Enum):
    LOW = "low"          # Minor (phone, email)
    MEDIUM = "medium"    # Moderate (address)
    HIGH = "high"        # Significant (name, DOB, ID numbers)
    CRITICAL = "critical" # Critical (SSN mismatch)
```

### `DecisionStatus`

```python
class DecisionStatus(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"
    NEEDS_MORE_INFO = "needs_more_info"
```

---

## Field Severity Mappings

Default field-to-severity mappings in `CrossValidator`:

| Severity | Fields |
|----------|--------|
| **CRITICAL** | `ssn`, `social_security_number` |
| **HIGH** | `name`, `full_name`, `dob`, `date_of_birth`, `passport_number`, `drivers_license_number`, `ein`, `bank_account_number`, `routing_number` |
| **MEDIUM** | `address`, `city`, `state`, `zip`, `postal_code` |
| **LOW** | `phone`, `email`, `gender`, `nationality` |

`FinancialCrossValidator` adds:

| Severity | Fields |
|----------|--------|
| **HIGH** | `revenue`, `gross_income`, `net_income`, `total_assets`, `total_liabilities`, `net_worth`, `annual_revenue` |
| **MEDIUM** | `total_expenses`, `monthly_income` |

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WATSONX_API_KEY` | IBM Cloud API key | For Watsonx provider |
| `WATSONX_URL` | Watsonx endpoint URL | For Watsonx provider |
| `WATSONX_PROJECT_ID` | Project/workspace ID | For Watsonx provider |
| `OLLAMA_BASE_URL` | Ollama server URL | For Ollama provider (default: `http://localhost:11434`) |

---

## Version Information

```python
import watsonx_vision

print(watsonx_vision.__version__)  # "0.1.0"
print(watsonx_vision.__author__)   # "AIQSO - Quinn Vidal"
```
