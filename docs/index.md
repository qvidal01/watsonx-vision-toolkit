# Watsonx Vision Toolkit

A powerful Python toolkit for vision-based document analysis using IBM Watsonx AI or Ollama.

[![PyPI version](https://badge.fury.io/py/watsonx-vision-toolkit.svg)](https://pypi.org/project/watsonx-vision-toolkit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Vision LLM Interface** - Unified API for IBM Watsonx AI and Ollama vision models
- **Document Classification** - Automatically classify documents (passport, license, tax returns, etc.)
- **Information Extraction** - Extract structured data from document images
- **Fraud Detection** - Detect document forgery, manipulation, and authenticity issues
- **Cross-Validation** - Compare data across multiple documents for consistency
- **Decision Engine** - Multi-criteria weighted scoring for automated decisions
- **Response Caching** - LRU cache with TTL to reduce API calls
- **Async Support** - Full async/await API for concurrent operations
- **CLI Tool** - Command-line interface for quick document analysis

## Quick Example

```python
from watsonx_vision import VisionLLM, VisionLLMConfig

# Configure for Ollama (local)
config = VisionLLMConfig(
    provider="ollama",
    model_id="llava:latest",
    url="http://localhost:11434"
)

llm = VisionLLM(config)

# Classify a document
image_data = VisionLLM.encode_image_to_base64("passport.png")
result = llm.classify_document(image_data)
print(result)  # {"doc_type": "Passport"}

# Extract information
info = llm.extract_information(image_data)
print(info)  # {"name": "John Doe", "dob": "1990-01-15", ...}
```

## CLI Example

```bash
# Classify a document
watsonx-vision classify document.png

# Extract information as JSON
watsonx-vision extract passport.jpg --output json

# Detect fraud
watsonx-vision fraud invoice.pdf --provider ollama
```

## Installation

```bash
pip install watsonx-vision-toolkit[all]
```

See [Installation](getting-started/installation.md) for detailed options.

## Why This Toolkit?

This toolkit was extracted from the **IBM Watsonx Loan Preprocessing Agents** project where it proved effective in production for:

- Automated loan document processing
- Identity verification workflows
- Financial document validation
- Fraud detection pipelines

It provides a clean, well-tested API that abstracts away the complexity of working with vision LLMs for document analysis.

## Next Steps

- [Installation](getting-started/installation.md) - Get the toolkit installed
- [Quick Start](getting-started/quickstart.md) - Your first document analysis
- [CLI Reference](guide/cli.md) - Use from the command line
- [API Reference](api/vision-llm.md) - Full API documentation
