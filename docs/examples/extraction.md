# Information Extraction

Extract structured data from document images.

## Basic Extraction

```python
from watsonx_vision import VisionLLM, VisionLLMConfig

config = VisionLLMConfig.from_env()
llm = VisionLLM(config)

# Load image
image_data = VisionLLM.encode_image_to_base64("passport.jpg")

# Extract with default fields
result = llm.extract_information(image_data)
print(result)
# {
#     "name": "John Doe",
#     "dob": "1990-01-15",
#     "address": "123 Main St",
#     "ssn": "XXX-XX-6789",
#     ...
# }
```

## Default Fields

Without custom fields, extracts common PII:

- Name / Full Name
- Date of Birth
- Address
- SSN (masked)
- Phone
- Email
- ID Numbers

## Custom Fields

Specify exactly what to extract:

```python
# Invoice extraction
invoice_data = llm.extract_information(
    image_data,
    fields=[
        "invoice_number",
        "date",
        "due_date",
        "vendor_name",
        "vendor_address",
        "line_items",
        "subtotal",
        "tax",
        "total"
    ]
)

# Receipt extraction
receipt_data = llm.extract_information(
    image_data,
    fields=[
        "store_name",
        "date",
        "items",
        "subtotal",
        "tax",
        "total",
        "payment_method"
    ]
)

# Medical document extraction
medical_data = llm.extract_information(
    image_data,
    fields=[
        "patient_name",
        "dob",
        "date_of_service",
        "diagnosis",
        "medications",
        "doctor_name"
    ]
)
```

## Date Format

Control output date format:

```python
# ISO format (default)
result = llm.extract_information(image_data, date_format="YYYY-MM-DD")
# dob: "1990-01-15"

# US format
result = llm.extract_information(image_data, date_format="MM/DD/YYYY")
# dob: "01/15/1990"

# European format
result = llm.extract_information(image_data, date_format="DD/MM/YYYY")
# dob: "15/01/1990"

# Long format
result = llm.extract_information(image_data, date_format="MMMM D, YYYY")
# dob: "January 15, 1990"
```

## Document-Specific Presets

### Passport

```python
passport_fields = [
    "full_name",
    "nationality",
    "date_of_birth",
    "place_of_birth",
    "sex",
    "passport_number",
    "date_of_issue",
    "date_of_expiry",
    "issuing_authority"
]

result = llm.extract_information(image_data, fields=passport_fields)
```

### Driver's License

```python
license_fields = [
    "full_name",
    "address",
    "date_of_birth",
    "license_number",
    "class",
    "issue_date",
    "expiration_date",
    "restrictions",
    "endorsements"
]

result = llm.extract_information(image_data, fields=license_fields)
```

### Tax Return (Form 1040)

```python
tax_fields = [
    "taxpayer_name",
    "ssn",
    "filing_status",
    "tax_year",
    "wages",
    "interest_income",
    "dividend_income",
    "business_income",
    "capital_gains",
    "total_income",
    "adjusted_gross_income",
    "taxable_income",
    "total_tax",
    "refund_amount"
]

result = llm.extract_information(image_data, fields=tax_fields)
```

### Bank Statement

```python
bank_fields = [
    "account_holder",
    "account_number",
    "statement_period",
    "beginning_balance",
    "total_deposits",
    "total_withdrawals",
    "ending_balance",
    "transactions"
]

result = llm.extract_information(image_data, fields=bank_fields)
```

## Batch Extraction

```python
def extract_batch(documents):
    """Extract information from multiple documents."""
    results = []

    for doc in documents:
        image_data = VisionLLM.encode_image_to_base64(doc["path"])
        data = llm.extract_information(
            image_data,
            fields=doc.get("fields"),
            date_format=doc.get("date_format", "YYYY-MM-DD")
        )
        results.append({
            "file": doc["path"],
            "data": data
        })

    return results

# Process different document types
documents = [
    {"path": "passport.jpg", "fields": passport_fields},
    {"path": "license.png", "fields": license_fields},
    {"path": "tax_return.pdf", "fields": tax_fields}
]

results = extract_batch(documents)
```

## Async Extraction

```python
import asyncio

async def extract_documents(paths, fields):
    """Extract from multiple documents concurrently."""
    async def extract_one(path):
        image_data = VisionLLM.encode_image_to_base64(path)
        return await llm.extract_information_async(image_data, fields=fields)

    tasks = [extract_one(p) for p in paths]
    return await asyncio.gather(*tasks)

# Run concurrent extraction
results = asyncio.run(extract_documents(
    ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    fields=["name", "date", "total"]
))
```

## Error Handling

```python
from watsonx_vision import LLMParseError, WatsonxVisionError

def safe_extract(image_path, fields):
    """Extract with error handling."""
    try:
        image_data = VisionLLM.encode_image_to_base64(image_path)
        return llm.extract_information(image_data, fields=fields)

    except LLMParseError as e:
        # Model returned non-JSON response
        print(f"Parse error: {e.details.get('raw_content')}")
        return {}

    except WatsonxVisionError as e:
        print(f"Extraction failed: {e.message}")
        return {}
```

## CLI Extraction

```bash
# Default extraction
watsonx-vision extract document.png

# Custom fields
watsonx-vision extract invoice.pdf --fields "invoice_number,date,total,vendor"

# Custom date format
watsonx-vision extract passport.jpg --date-format "MM/DD/YYYY"

# JSON output
watsonx-vision extract doc.png --output json --output-file data.json
```

## Complete Example

```python
#!/usr/bin/env python3
"""Information extraction example."""

from watsonx_vision import VisionLLM, VisionLLMConfig

def main():
    config = VisionLLMConfig.from_env()
    llm = VisionLLM(config)

    # Document-specific field sets
    PRESETS = {
        "passport": [
            "full_name", "nationality", "date_of_birth",
            "passport_number", "date_of_expiry"
        ],
        "invoice": [
            "invoice_number", "date", "vendor_name",
            "line_items", "total"
        ],
        "receipt": [
            "store_name", "date", "items", "total"
        ]
    }

    # Example: Extract from invoice
    image_data = VisionLLM.encode_image_to_base64("invoice.pdf")

    result = llm.extract_information(
        image_data,
        fields=PRESETS["invoice"],
        date_format="YYYY-MM-DD"
    )

    print("Extracted Data:")
    for field, value in result.items():
        print(f"  {field}: {value}")

if __name__ == "__main__":
    main()
```
