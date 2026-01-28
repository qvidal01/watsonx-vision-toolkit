# Complete Workflow

End-to-end document processing pipeline for loan applications.

## Overview

This example demonstrates a complete loan application workflow:

1. **Document Classification** - Identify document types
2. **Information Extraction** - Extract structured data
3. **Fraud Detection** - Validate document authenticity
4. **Cross-Validation** - Check data consistency
5. **Decision Making** - Automated approval decision

## Full Example

```python
#!/usr/bin/env python3
"""Complete loan application processing workflow."""

from pathlib import Path
from watsonx_vision import (
    VisionLLM, VisionLLMConfig, CacheConfig,
    FraudDetector, CrossValidator
)
from watsonx_vision.decision_engine import LoanDecisionEngine
from watsonx_vision.fraud_detector import SpecializedFraudDetector

def main():
    # ==========================================
    # 1. SETUP
    # ==========================================
    print("Setting up components...")

    # Vision LLM with caching
    vision_config = VisionLLMConfig.from_env()
    cache_config = CacheConfig(enabled=True, ttl=3600)
    vision_llm = VisionLLM(vision_config, cache_config=cache_config)

    # Fraud detector
    fraud_detector = SpecializedFraudDetector(vision_llm)

    # Cross-validator
    cross_validator = CrossValidator(
        api_key="your-key",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="your-project"
    )

    # Decision engine
    decision_engine = LoanDecisionEngine(
        min_age=18,
        min_income=30000,
        max_dti=0.43
    )

    # ==========================================
    # 2. APPLICATION DATA (from web form)
    # ==========================================
    application_data = {
        "applicant_name": "John Michael Doe",
        "date_of_birth": "1990-05-15",
        "ssn": "123-45-6789",
        "address": "123 Main Street, Springfield, IL 62701",
        "annual_income": 75000,
        "monthly_debt": 1500,
        "loan_amount": 25000,
        "loan_purpose": "Home Improvement"
    }

    # ==========================================
    # 3. DOCUMENT PROCESSING
    # ==========================================
    print("\nProcessing uploaded documents...")

    # Documents uploaded by applicant
    document_paths = [
        "passport.jpg",
        "tax_return_2024.pdf",
        "bank_statement_jan.pdf",
        "bank_statement_feb.pdf"
    ]

    # Field sets for different document types
    FIELD_SETS = {
        "Passport": ["full_name", "date_of_birth", "passport_number"],
        "Tax Return": ["name", "ssn", "gross_income", "net_income", "tax_year"],
        "Bank Account Statement": ["account_holder", "statement_period", "ending_balance", "total_deposits"]
    }

    processed_documents = []
    fraud_results = []

    for doc_path in document_paths:
        print(f"\n  Processing: {doc_path}")

        # Load image
        image_data = VisionLLM.encode_image_to_base64(doc_path)

        # Step 1: Classify document
        classification = vision_llm.classify_document(image_data)
        doc_type = classification["doc_type"]
        print(f"    Type: {doc_type}")

        # Step 2: Extract information
        fields = FIELD_SETS.get(doc_type, ["name", "date"])
        extracted_data = vision_llm.extract_information(image_data, fields=fields)
        extracted_data["doc_type"] = doc_type
        extracted_data["filename"] = doc_path
        processed_documents.append(extracted_data)
        print(f"    Extracted: {len(extracted_data)} fields")

        # Step 3: Fraud detection
        fraud_doc_type = {
            "Passport": "passport",
            "Tax Return": "tax_return",
            "Bank Account Statement": "bank_statement"
        }.get(doc_type)

        fraud_result = fraud_detector.validate_document(
            image_data,
            filename=doc_path,
            document_type=fraud_doc_type
        )
        fraud_results.append(fraud_result)

        status = "✓ PASS" if fraud_result.valid else "✗ FAIL"
        print(f"    Fraud Check: {status} ({fraud_result.confidence}%)")

    # ==========================================
    # 4. CROSS-VALIDATION
    # ==========================================
    print("\nCross-validating data...")

    validation_result = cross_validator.validate(
        application_data,
        processed_documents
    )

    if validation_result.passed:
        print("  ✓ Data is consistent across all documents")
    else:
        print(f"  ✗ Found {validation_result.total_inconsistencies} inconsistencies:")
        for issue in validation_result.inconsistencies:
            print(f"    [{issue.severity.value}] {issue.field}: {issue.explanation}")

    # ==========================================
    # 5. DECISION MAKING
    # ==========================================
    print("\nMaking loan decision...")

    decision = decision_engine.decide(
        application_data=application_data,
        fraud_results=fraud_results,
        validation_result=validation_result
    )

    # ==========================================
    # 6. RESULTS
    # ==========================================
    print("\n" + "=" * 60)
    print("LOAN APPLICATION DECISION")
    print("=" * 60)

    print(f"\nApplicant: {application_data['applicant_name']}")
    print(f"Loan Amount: ${application_data['loan_amount']:,}")
    print(f"Purpose: {application_data['loan_purpose']}")

    print(f"\nDecision: {decision.status.value.upper()}")
    print(f"Score: {decision.score}/100")

    print("\nCriteria Results:")
    for cr in decision.criteria_results:
        status = "✓" if cr.passed else "✗"
        print(f"  {status} {cr.name}: {cr.contribution}/{cr.weight} points")

    print("\nReasons:")
    for reason in decision.reasons:
        print(f"  - {reason}")

    # Fraud detection summary
    fraud_report = fraud_detector.generate_report(fraud_results)
    print(f"\nFraud Detection Summary:")
    print(f"  Documents checked: {fraud_report['total_documents']}")
    print(f"  Valid: {fraud_report['valid_documents']}")
    print(f"  Fraud rate: {fraud_report['fraud_rate']}%")

    # Cross-validation summary
    print(f"\nCross-Validation Summary:")
    print(f"  Status: {'PASSED' if validation_result.passed else 'FAILED'}")
    print(f"  Inconsistencies: {validation_result.total_inconsistencies}")
    print(f"  Matched fields: {len(validation_result.matched_fields)}")

    # Cache stats
    cache_stats = vision_llm.cache_stats()
    print(f"\nCache Performance:")
    print(f"  Hits: {cache_stats.hits}, Misses: {cache_stats.misses}")
    print(f"  Hit rate: {cache_stats.hit_rate:.1%}")

    return decision

if __name__ == "__main__":
    main()
```

## Output Example

```
Setting up components...

Processing uploaded documents...

  Processing: passport.jpg
    Type: Passport
    Extracted: 4 fields
    Fraud Check: ✓ PASS (92%)

  Processing: tax_return_2024.pdf
    Type: Tax Return
    Extracted: 6 fields
    Fraud Check: ✓ PASS (88%)

  Processing: bank_statement_jan.pdf
    Type: Bank Account Statement
    Extracted: 5 fields
    Fraud Check: ✓ PASS (85%)

  Processing: bank_statement_feb.pdf
    Type: Bank Account Statement
    Extracted: 5 fields
    Fraud Check: ✓ PASS (87%)

Cross-validating data...
  ✓ Data is consistent across all documents

Making loan decision...

============================================================
LOAN APPLICATION DECISION
============================================================

Applicant: John Michael Doe
Loan Amount: $25,000
Purpose: Home Improvement

Decision: APPROVED
Score: 95/100

Criteria Results:
  ✓ age_verification: 20/20 points
  ✓ income_requirement: 30/30 points
  ✓ debt_to_income: 25/25 points
  ✓ fraud_check: 15/15 points
  ✗ cross_validation: 5/10 points

Reasons:
  - Applicant meets minimum age requirement (33 years)
  - Annual income ($75,000) exceeds minimum ($30,000)
  - Debt-to-income ratio (24%) is within acceptable range (43%)
  - All documents passed fraud detection
  - Minor inconsistencies in cross-validation (name format)

Fraud Detection Summary:
  Documents checked: 4
  Valid: 4
  Fraud rate: 0.0%

Cross-Validation Summary:
  Status: PASSED
  Inconsistencies: 1
  Matched fields: 8

Cache Performance:
  Hits: 3, Misses: 8
  Hit rate: 27.3%
```

## Async Workflow

For better performance with many documents:

```python
import asyncio

async def process_loan_application_async(application_data, document_paths):
    """Process loan application with concurrent document processing."""

    # Process all documents concurrently
    async def process_document(path):
        image_data = VisionLLM.encode_image_to_base64(path)

        # Run classification, extraction, and fraud check concurrently
        classify_task = vision_llm.classify_document_async(image_data)
        fraud_task = fraud_detector.validate_document_async(image_data, path)

        classification, fraud_result = await asyncio.gather(
            classify_task, fraud_task
        )

        # Extract based on document type
        fields = FIELD_SETS.get(classification["doc_type"], ["name"])
        extracted = await vision_llm.extract_information_async(image_data, fields=fields)
        extracted["doc_type"] = classification["doc_type"]

        return extracted, fraud_result

    # Process all documents
    tasks = [process_document(path) for path in document_paths]
    results = await asyncio.gather(*tasks)

    processed_docs = [r[0] for r in results]
    fraud_results = [r[1] for r in results]

    # Cross-validate
    validation_result = await cross_validator.validate_async(
        application_data, processed_docs
    )

    # Decision
    decision = decision_engine.decide(
        application_data=application_data,
        fraud_results=fraud_results,
        validation_result=validation_result
    )

    return decision

# Run async workflow
decision = asyncio.run(process_loan_application_async(
    application_data,
    document_paths
))
```
