# CrossValidator

Cross-validation engine for multi-document consistency checking.

## CrossValidator

### Constructor

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
| `retry_config` | `RetryConfig \| None` | `None` | Retry configuration |

---

### validate

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

**Async variant:** `validate_async()`

---

### validate_batch

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

**Async variant:** `validate_batch_async(packages, concurrent=False)`

---

### generate_report

Generate human-readable report from validation results.

```python
report = validator.generate_report(result, include_details=True)
print(report)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `ValidationResult` | Required | Validation result |
| `include_details` | `bool` | `True` | Include detailed inconsistencies |

**Returns:** `str` - Formatted report with severity icons

**Example output:**

```
Cross-Validation Report
=======================

Status: FAILED
Total Inconsistencies: 2
Confidence: 75%

Summary: Found discrepancies in name and SSN fields

Inconsistencies:
  🔴 [CRITICAL] ssn
     Application: 123-45-6789
     Tax Return: 123-45-6780
     Explanation: SSN last digit mismatch

  🟡 [MEDIUM] name
     Passport: John Doe
     Tax Return: John D. Doe
     Explanation: Middle initial variation
```

---

## FinancialCrossValidator

Extends `CrossValidator` with financial-specific field severities.

```python
from watsonx_vision.cross_validator import FinancialCrossValidator

validator = FinancialCrossValidator(
    api_key="...",
    url="...",
    project_id="..."
)
```

### Additional Field Severities

| Severity | Fields |
|----------|--------|
| **HIGH** | `revenue`, `gross_income`, `net_income`, `total_assets`, `total_liabilities`, `net_worth`, `annual_revenue` |
| **MEDIUM** | `total_expenses`, `monthly_income` |

---

### validate_financials

Validate financial documents specifically.

```python
result = validator.validate_financials(
    tax_return_data={
        "gross_income": 75000,
        "net_income": 62000,
        "tax_year": 2024
    },
    bank_statements=[
        {"month": "January", "deposits": 6250, "ending_balance": 15000},
        {"month": "February", "deposits": 6250, "ending_balance": 18500}
    ],
    pfs_data={  # Optional
        "total_assets": 250000,
        "total_liabilities": 100000,
        "net_worth": 150000
    },
    tolerance_percent=5.0
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tax_return_data` | `Dict` | Yes | Tax return extracted data |
| `bank_statements` | `List[Dict]` | Yes | Bank statement data |
| `pfs_data` | `Dict \| None` | No | Personal financial statement |
| `tolerance_percent` | `float` | No | Acceptable variance (default 5%) |

**Returns:** `ValidationResult`

**Async variant:** `validate_financials_async()`

---

## ValidationResult

Dataclass for validation results.

```python
@dataclass
class ValidationResult:
    passed: bool                       # No critical/high issues
    total_inconsistencies: int         # Count of issues
    inconsistencies: List[Inconsistency]
    matched_fields: List[str]          # Fields that matched
    summary: str                       # Human-readable summary
    confidence: int                    # 0-100
```

### Methods

#### to_dict

Convert to dictionary.

```python
data = result.to_dict()
```

---

## Inconsistency

Dataclass for individual inconsistencies.

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
```

---

## InconsistencySeverity Enum

```python
from watsonx_vision.cross_validator import InconsistencySeverity

class InconsistencySeverity(Enum):
    LOW = "low"          # Minor (phone, email)
    MEDIUM = "medium"    # Moderate (address)
    HIGH = "high"        # Significant (name, DOB, ID numbers)
    CRITICAL = "critical" # Critical (SSN mismatch)
```

---

## Default Field Severity Mappings

| Severity | Fields |
|----------|--------|
| **CRITICAL** | `ssn`, `social_security_number` |
| **HIGH** | `name`, `full_name`, `dob`, `date_of_birth`, `passport_number`, `drivers_license_number`, `ein`, `bank_account_number`, `routing_number` |
| **MEDIUM** | `address`, `city`, `state`, `zip`, `postal_code` |
| **LOW** | `phone`, `email`, `gender`, `nationality` |

---

## Example

```python
from watsonx_vision import CrossValidator

validator = CrossValidator(
    api_key="your-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project"
)

# Application data from form
application_data = {
    "name": "John Doe",
    "dob": "1990-05-15",
    "ssn": "123-45-6789",
    "address": "123 Main St, Springfield, IL 62701"
}

# Extracted data from documents
document_data = [
    {
        "doc_type": "Passport",
        "name": "John Doe",
        "dob": "1990-05-15"
    },
    {
        "doc_type": "Tax Return",
        "name": "John D. Doe",
        "ssn": "123-45-6789",
        "annual_income": 75000
    }
]

# Validate
result = validator.validate(application_data, document_data)

if result.passed:
    print("All data is consistent!")
else:
    print(f"Found {result.total_inconsistencies} inconsistencies:")
    for issue in result.inconsistencies:
        print(f"  [{issue.severity.value}] {issue.field}: {issue.explanation}")

# Generate report
print(validator.generate_report(result))
```
