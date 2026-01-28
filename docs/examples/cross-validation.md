# Cross-Validation

Validate data consistency across multiple documents.

## Basic Cross-Validation

```python
from watsonx_vision import CrossValidator

validator = CrossValidator(
    api_key="your-api-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project-id"
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
    print(f"Found {result.total_inconsistencies} inconsistencies")
    for issue in result.inconsistencies:
        print(f"  [{issue.severity.value}] {issue.field}: {issue.explanation}")
```

## ValidationResult Properties

| Property | Type | Description |
|----------|------|-------------|
| `passed` | `bool` | No critical/high severity issues |
| `total_inconsistencies` | `int` | Count of issues |
| `inconsistencies` | `List[Inconsistency]` | Detailed issues |
| `matched_fields` | `List[str]` | Fields that matched |
| `summary` | `str` | Human-readable summary |
| `confidence` | `int` | 0-100 confidence score |

## Severity Levels

| Severity | Fields | Action |
|----------|--------|--------|
| **CRITICAL** | SSN | Reject |
| **HIGH** | Name, DOB, ID numbers | Manual review |
| **MEDIUM** | Address, city, state | Note discrepancy |
| **LOW** | Phone, email | Accept with note |

## Custom Field Severities

Add custom severity mappings:

```python
from watsonx_vision.cross_validator import InconsistencySeverity

result = validator.validate(
    application_data,
    document_data,
    custom_fields={
        "employee_id": InconsistencySeverity.HIGH,
        "department": InconsistencySeverity.LOW,
        "salary": InconsistencySeverity.CRITICAL
    }
)
```

## Generate Reports

```python
# Validate
result = validator.validate(application_data, document_data)

# Generate human-readable report
report = validator.generate_report(result, include_details=True)
print(report)
```

**Output:**

```
Cross-Validation Report
=======================

Status: FAILED
Total Inconsistencies: 2
Confidence: 75%

Summary: Found discrepancies in name field across documents

Matched Fields: dob, ssn

Inconsistencies:
  🟡 [MEDIUM] name
     Application: John Doe
     Tax Return: John D. Doe
     Explanation: Name variation with middle initial

  🟢 [LOW] phone
     Passport: (555) 123-4567
     Tax Return: 555-123-4567
     Explanation: Phone format variation
```

## Financial Cross-Validation

Use specialized financial validator:

```python
from watsonx_vision.cross_validator import FinancialCrossValidator

validator = FinancialCrossValidator(
    api_key="your-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project"
)

result = validator.validate_financials(
    tax_return_data={
        "gross_income": 75000,
        "net_income": 62000,
        "tax_year": 2024
    },
    bank_statements=[
        {"month": "January", "deposits": 6250},
        {"month": "February", "deposits": 6250}
    ],
    pfs_data={
        "total_assets": 250000,
        "total_liabilities": 100000,
        "net_worth": 150000
    },
    tolerance_percent=5.0  # Allow 5% variance
)
```

## Batch Validation

Validate multiple document packages:

```python
packages = [
    {
        "application_data": {"name": "John Doe", "ssn": "123-45-6789"},
        "document_data": [
            {"doc_type": "Passport", "name": "John Doe"},
            {"doc_type": "License", "name": "John Doe"}
        ]
    },
    {
        "application_data": {"name": "Jane Smith", "ssn": "987-65-4321"},
        "document_data": [
            {"doc_type": "Passport", "name": "Jane Smith"},
            {"doc_type": "Tax Return", "name": "Jane A. Smith"}
        ]
    }
]

results = validator.validate_batch(packages)

for i, result in enumerate(results):
    status = "PASS" if result.passed else "FAIL"
    print(f"Package {i+1}: {status} ({result.total_inconsistencies} issues)")
```

## Async Validation

```python
import asyncio

async def validate_packages(packages):
    """Validate packages concurrently."""
    results = await validator.validate_batch_async(packages, concurrent=True)
    return results

# Run async validation
results = asyncio.run(validate_packages(packages))
```

## Error Handling

```python
from watsonx_vision import (
    LLMConnectionError,
    LLMParseError,
    ValidationError,
    WatsonxVisionError
)

def safe_validate(app_data, doc_data):
    """Validate with error handling."""
    try:
        return validator.validate(app_data, doc_data)

    except ValidationError as e:
        print(f"Validation logic error: {e.message}")
        return None

    except LLMParseError as e:
        print(f"LLM response parse error: {e.details}")
        return None

    except LLMConnectionError:
        print("Connection to LLM failed")
        return None

    except WatsonxVisionError as e:
        print(f"Validation error: {e.message}")
        return None
```

## Complete Example

```python
#!/usr/bin/env python3
"""Cross-validation pipeline example."""

from watsonx_vision import VisionLLM, VisionLLMConfig, CrossValidator

def main():
    # Setup vision LLM for extraction
    vision_config = VisionLLMConfig.from_env()
    llm = VisionLLM(vision_config)

    # Setup cross-validator
    validator = CrossValidator(
        api_key="your-key",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="your-project"
    )

    # Application data (from web form)
    application_data = {
        "applicant_name": "John Michael Doe",
        "date_of_birth": "1990-05-15",
        "ssn": "123-45-6789",
        "annual_income": 75000,
        "employer": "Acme Corp"
    }

    # Extract data from uploaded documents
    documents = [
        ("passport.jpg", ["name", "dob", "passport_number"]),
        ("tax_return.pdf", ["name", "ssn", "gross_income", "employer"]),
        ("bank_statement.pdf", ["account_holder", "average_balance"])
    ]

    document_data = []
    for doc_path, fields in documents:
        print(f"Extracting from {doc_path}...")
        image_data = VisionLLM.encode_image_to_base64(doc_path)
        extracted = llm.extract_information(image_data, fields=fields)
        extracted["doc_type"] = doc_path.split(".")[0].replace("_", " ").title()
        document_data.append(extracted)

    # Cross-validate
    print("\nCross-validating...")
    result = validator.validate(application_data, document_data)

    # Generate and print report
    report = validator.generate_report(result)
    print(report)

    # Decision
    if result.passed:
        print("\n✅ Application data is consistent with documents")
    else:
        critical_issues = [
            i for i in result.inconsistencies
            if i.severity.value in ["critical", "high"]
        ]
        if critical_issues:
            print(f"\n❌ Found {len(critical_issues)} critical/high severity issues")
        else:
            print(f"\n⚠️ Found {result.total_inconsistencies} minor inconsistencies")

if __name__ == "__main__":
    main()
```
