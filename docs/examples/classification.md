# Document Classification

Classify documents into predefined types using vision analysis.

## Basic Classification

```python
from watsonx_vision import VisionLLM, VisionLLMConfig

# Configure the LLM
config = VisionLLMConfig(
    provider="ollama",
    model_id="llava:latest",
    url="http://localhost:11434"
)

llm = VisionLLM(config)

# Load and encode image
image_data = VisionLLM.encode_image_to_base64("document.png")

# Classify with default types
result = llm.classify_document(image_data)
print(f"Document type: {result['doc_type']}")
```

## Default Document Types

When no custom types are provided, the classifier uses:

- Driving License
- Passport
- SSN (Social Security Number)
- Utility Bill
- Salary Slip
- ITR (Income Tax Return)
- Bank Account Statement
- Tax Return
- Articles of Incorporation
- Personal Financial Statement
- Others

## Custom Document Types

Specify your own document types:

```python
# Financial documents
result = llm.classify_document(
    image_data,
    document_types=[
        "Invoice",
        "Receipt",
        "Purchase Order",
        "Credit Memo",
        "Statement"
    ]
)

# Legal documents
result = llm.classify_document(
    image_data,
    document_types=[
        "Contract",
        "Agreement",
        "Power of Attorney",
        "Deed",
        "Will"
    ]
)

# Medical documents
result = llm.classify_document(
    image_data,
    document_types=[
        "Prescription",
        "Lab Report",
        "Medical Record",
        "Insurance Card",
        "Discharge Summary"
    ]
)
```

## Batch Classification

Process multiple documents:

```python
from pathlib import Path

def classify_batch(document_paths):
    """Classify multiple documents."""
    results = []

    for path in document_paths:
        image_data = VisionLLM.encode_image_to_base64(path)
        result = llm.classify_document(image_data)
        results.append({
            "file": path,
            "type": result["doc_type"]
        })

    return results

# Classify all PDFs in a folder
documents = list(Path("documents/").glob("*.pdf"))
results = classify_batch(documents)

for r in results:
    print(f"{r['file']}: {r['type']}")
```

## Async Classification

Process documents concurrently:

```python
import asyncio

async def classify_document(path):
    image_data = VisionLLM.encode_image_to_base64(path)
    return await llm.classify_document_async(image_data)

async def classify_batch_async(paths):
    tasks = [classify_document(p) for p in paths]
    return await asyncio.gather(*tasks)

# Run async batch
results = asyncio.run(classify_batch_async(document_paths))
```

## With Caching

Cache results for repeated classification:

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig

cache_config = CacheConfig(
    enabled=True,
    ttl=3600,  # Cache for 1 hour
    max_size=500
)

config = VisionLLMConfig.from_env()
llm = VisionLLM(config, cache_config=cache_config)

# First call - hits API
result1 = llm.classify_document(image_data)

# Second call - returns from cache (instant)
result2 = llm.classify_document(image_data)

# Check cache stats
stats = llm.cache_stats()
print(f"Cache hit rate: {stats.hit_rate:.1%}")
```

## Error Handling

```python
from watsonx_vision import (
    LLMConnectionError,
    LLMTimeoutError,
    LLMParseError,
    WatsonxVisionError
)

def safe_classify(image_path):
    """Classify with error handling."""
    try:
        image_data = VisionLLM.encode_image_to_base64(image_path)
        result = llm.classify_document(image_data)
        return result["doc_type"]

    except LLMConnectionError:
        print(f"Connection failed for {image_path}")
        return "Unknown"

    except LLMTimeoutError:
        print(f"Timeout for {image_path}")
        return "Unknown"

    except LLMParseError as e:
        print(f"Parse error for {image_path}: {e.details}")
        return "Unknown"

    except WatsonxVisionError as e:
        print(f"Error for {image_path}: {e.message}")
        return "Unknown"
```

## CLI Classification

Use the command line:

```bash
# Basic classification
watsonx-vision classify document.png

# Custom types
watsonx-vision classify invoice.pdf --types "Invoice,Receipt,PO"

# JSON output
watsonx-vision classify doc.png --output json

# Save to file
watsonx-vision classify doc.png --output json --output-file result.json

# With caching
watsonx-vision classify doc.png --cache --cache-ttl 7200
```

## Complete Example

```python
#!/usr/bin/env python3
"""Document classification example."""

import sys
from pathlib import Path
from watsonx_vision import VisionLLM, VisionLLMConfig, CacheConfig

def main():
    # Configuration from environment
    config = VisionLLMConfig.from_env()
    cache_config = CacheConfig(enabled=True, ttl=3600)

    llm = VisionLLM(config, cache_config=cache_config)

    # Get document path from command line
    if len(sys.argv) < 2:
        print("Usage: python classify.py <document_path>")
        sys.exit(1)

    doc_path = Path(sys.argv[1])
    if not doc_path.exists():
        print(f"File not found: {doc_path}")
        sys.exit(1)

    # Classify
    print(f"Classifying: {doc_path}")
    image_data = VisionLLM.encode_image_to_base64(str(doc_path))
    result = llm.classify_document(image_data)

    print(f"Document type: {result['doc_type']}")

    # Show cache stats
    stats = llm.cache_stats()
    print(f"Cache: {stats.hits} hits, {stats.misses} misses")

if __name__ == "__main__":
    main()
```
