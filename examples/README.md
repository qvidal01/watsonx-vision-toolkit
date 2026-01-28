# Watsonx Vision Toolkit Examples

This directory contains practical examples demonstrating the key features of the watsonx-vision-toolkit.

## Prerequisites

1. Install the toolkit:
   ```bash
   pip install -e ".[watsonx]"  # For IBM Watsonx
   # or
   pip install -e ".[ollama]"   # For local Ollama
   ```

2. Set up your credentials (for Watsonx):
   ```bash
   export WATSONX_API_KEY="your-api-key"
   export WATSONX_PROJECT_ID="your-project-id"
   export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
   ```

3. Or start Ollama locally:
   ```bash
   ollama serve
   ollama pull llava  # Vision model
   ```

## Examples

| File | Description |
|------|-------------|
| [basic_classification.py](basic_classification.py) | Classify documents into predefined types |
| [information_extraction.py](information_extraction.py) | Extract structured data from documents |
| [fraud_detection.py](fraud_detection.py) | Detect document fraud and manipulation |
| [cross_validation.py](cross_validation.py) | Validate consistency across multiple documents |
| [retry_configuration.py](retry_configuration.py) | Configure retry behavior for resilient LLM calls |
| [complete_workflow.py](complete_workflow.py) | End-to-end loan application processing |

## Quick Start

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, LLMProvider

# Initialize with Ollama (local)
config = VisionLLMConfig(
    provider=LLMProvider.OLLAMA,
    model_id="llava",
    url="http://localhost:11434"
)
llm = VisionLLM(config)

# Classify a document
image_data = VisionLLM.encode_image_to_base64("document.png")
result = llm.classify_document(image_data)
print(f"Document type: {result['doc_type']}")
```

## Running Examples

Each example can be run directly:

```bash
# Basic classification
python examples/basic_classification.py path/to/document.png

# With custom provider
python examples/basic_classification.py document.png --provider ollama

# Complete workflow
python examples/complete_workflow.py --documents-dir ./loan_docs/
```

## Sample Documents

For testing, you can use any document images. The toolkit works with:
- PNG, JPEG, GIF, WebP images
- Scanned documents
- Photos of documents
- Screenshots

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WATSONX_API_KEY` | IBM Watsonx API key | Required for Watsonx |
| `WATSONX_PROJECT_ID` | Watsonx project ID | Required for Watsonx |
| `WATSONX_URL` | Watsonx API URL | `https://us-south.ml.cloud.ibm.com` |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `VISION_MODEL` | Model ID to use | Provider-specific default |
