# Async Support

The toolkit provides full async/await support for concurrent operations.

## Overview

Async support enables:

- **Concurrent processing** - Process multiple documents simultaneously
- **Better throughput** - Don't wait for I/O-bound operations
- **Scalable APIs** - Build async web services
- **Efficient batch processing** - Parallel document analysis

## Async Methods

All main methods have async variants:

| Sync Method | Async Method |
|-------------|--------------|
| `analyze_image()` | `analyze_image_async()` |
| `classify_document()` | `classify_document_async()` |
| `extract_information()` | `extract_information_async()` |
| `validate_authenticity()` | `validate_authenticity_async()` |

## Quick Start

```python
import asyncio
from watsonx_vision import VisionLLM, VisionLLMConfig

config = VisionLLMConfig.from_env()
llm = VisionLLM(config)

async def main():
    image = VisionLLM.encode_image_to_base64("document.png")

    # Async classification
    result = await llm.classify_document_async(image)
    print(result)

asyncio.run(main())
```

## Concurrent Processing

Process multiple documents in parallel:

```python
import asyncio
from watsonx_vision import VisionLLM, VisionLLMConfig

config = VisionLLMConfig.from_env()
llm = VisionLLM(config)

async def classify_document(path):
    image = VisionLLM.encode_image_to_base64(path)
    return await llm.classify_document_async(image)

async def main():
    documents = ["doc1.png", "doc2.png", "doc3.png", "doc4.png"]

    # Process all documents concurrently
    tasks = [classify_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks)

    for doc, result in zip(documents, results):
        print(f"{doc}: {result['doc_type']}")

asyncio.run(main())
```

## FraudDetector Async

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, FraudDetector

config = VisionLLMConfig.from_env()
llm = VisionLLM(config)
detector = FraudDetector(llm)

async def main():
    image = VisionLLM.encode_image_to_base64("passport.png")

    # Single document
    result = await detector.validate_document_async(image)

    # Batch with concurrency
    documents = [
        {"image_data": img1, "filename": "doc1.png"},
        {"image_data": img2, "filename": "doc2.png"},
        {"image_data": img3, "filename": "doc3.png"}
    ]
    results = await detector.validate_batch_async(documents, concurrent=True)

asyncio.run(main())
```

## CrossValidator Async

```python
from watsonx_vision import CrossValidator

validator = CrossValidator(
    api_key="your-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project"
)

async def main():
    result = await validator.validate_async(
        application_data={"name": "John Doe", "dob": "1990-01-15"},
        document_data=[
            {"doc_type": "Passport", "name": "John Doe"},
            {"doc_type": "License", "name": "John D. Doe"}
        ]
    )

    # Batch validation
    packages = [
        {"application_data": {...}, "document_data": [...]},
        {"application_data": {...}, "document_data": [...]}
    ]
    results = await validator.validate_batch_async(packages, concurrent=True)

asyncio.run(main())
```

## FinancialCrossValidator Async

```python
from watsonx_vision.cross_validator import FinancialCrossValidator

validator = FinancialCrossValidator(
    api_key="your-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project"
)

async def main():
    result = await validator.validate_financials_async(
        tax_return_data={"gross_income": 75000},
        bank_statements=[{"deposits": 6250}],
        pfs_data={"net_worth": 150000}
    )

asyncio.run(main())
```

## Async Retry

Use async retry decorators:

```python
from watsonx_vision import async_retry_with_backoff, async_retry_llm_call, RetryConfig

config = RetryConfig(max_attempts=3)

@async_retry_with_backoff(config)
async def reliable_api_call():
    return await some_async_operation()

# Or function-based
async def my_operation():
    return await llm.classify_document_async(image)

result = await async_retry_llm_call(my_operation, config)
```

## With Caching

Async methods work with caching:

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig

cache_config = CacheConfig(enabled=True, ttl=3600)
config = VisionLLMConfig.from_env()
llm = VisionLLM(config, cache_config=cache_config)

async def main():
    image = VisionLLM.encode_image_to_base64("document.png")

    # First call - hits API
    result1 = await llm.classify_document_async(image)

    # Second call - returns from cache
    result2 = await llm.classify_document_async(image)

asyncio.run(main())
```

## Semaphore for Rate Limiting

Control concurrency to avoid rate limits:

```python
import asyncio
from watsonx_vision import VisionLLM, VisionLLMConfig

config = VisionLLMConfig.from_env()
llm = VisionLLM(config)

# Limit to 5 concurrent requests
semaphore = asyncio.Semaphore(5)

async def classify_with_limit(path):
    async with semaphore:
        image = VisionLLM.encode_image_to_base64(path)
        return await llm.classify_document_async(image)

async def main():
    documents = [f"doc{i}.png" for i in range(100)]

    tasks = [classify_with_limit(doc) for doc in documents]
    results = await asyncio.gather(*tasks)

    return results

asyncio.run(main())
```

## FastAPI Integration

Build async web APIs:

```python
from fastapi import FastAPI, UploadFile
from watsonx_vision import VisionLLM, VisionLLMConfig
import base64

app = FastAPI()

config = VisionLLMConfig.from_env()
llm = VisionLLM(config)

@app.post("/classify")
async def classify_document(file: UploadFile):
    content = await file.read()
    image_data = f"data:image/png;base64,{base64.b64encode(content).decode()}"

    result = await llm.classify_document_async(image_data)
    return result

@app.post("/extract")
async def extract_information(file: UploadFile):
    content = await file.read()
    image_data = f"data:image/png;base64,{base64.b64encode(content).decode()}"

    result = await llm.extract_information_async(image_data)
    return result
```

## Example: Async Document Pipeline

```python
import asyncio
from watsonx_vision import VisionLLM, VisionLLMConfig, FraudDetector

config = VisionLLMConfig.from_env()
llm = VisionLLM(config)
detector = FraudDetector(llm)

async def process_document(path):
    """Full async document processing pipeline."""
    image = VisionLLM.encode_image_to_base64(path)

    # Run classification and extraction concurrently
    classify_task = llm.classify_document_async(image)
    extract_task = llm.extract_information_async(image)
    fraud_task = detector.validate_document_async(image)

    classification, extraction, fraud_result = await asyncio.gather(
        classify_task, extract_task, fraud_task
    )

    return {
        "path": path,
        "doc_type": classification["doc_type"],
        "extracted_data": extraction,
        "fraud_check": {
            "valid": fraud_result.valid,
            "confidence": fraud_result.confidence,
            "severity": fraud_result.severity.value
        }
    }

async def main():
    documents = ["passport.png", "license.jpg", "tax_return.pdf"]

    # Process all documents with full pipeline
    tasks = [process_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"\n{result['path']}:")
        print(f"  Type: {result['doc_type']}")
        print(f"  Fraud: {'PASS' if result['fraud_check']['valid'] else 'FAIL'}")

asyncio.run(main())
```
