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
| [environment_config.py](environment_config.py) | Load configuration from environment variables |
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

Use `VisionLLMConfig.from_env()` to load configuration from environment variables:

```python
from watsonx_vision import VisionLLM, VisionLLMConfig

# Load all settings from environment
config = VisionLLMConfig.from_env()
llm = VisionLLM(config)

# With a prefix for multiple configurations
prod_config = VisionLLMConfig.from_env(prefix="PROD_")

# With explicit overrides
config = VisionLLMConfig.from_env(temperature=0.7, max_tokens=4000)
```

### Supported Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VISION_PROVIDER` | Provider: `watsonx`, `ollama`, `openai`, `anthropic` | `watsonx` |
| `VISION_MODEL_ID` | Model ID to use | Provider-specific |
| `WATSONX_API_KEY` | IBM Watsonx API key | Required for Watsonx |
| `WATSONX_PROJECT_ID` | Watsonx project ID | Required for Watsonx |
| `WATSONX_URL` | Watsonx API URL | None |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `VISION_MAX_TOKENS` | Maximum tokens in response | `2000` |
| `VISION_TEMPERATURE` | Temperature (0.0-1.0) | `0.0` |
| `VISION_TOP_P` | Top-p sampling | `0.1` |
| `VISION_RETRY_ENABLED` | Enable automatic retries | `true` |
| `VISION_RETRY_MAX_ATTEMPTS` | Maximum retry attempts | `3` |
| `VISION_RETRY_BASE_DELAY` | Initial retry delay (seconds) | `1.0` |
| `VISION_RETRY_MAX_DELAY` | Maximum retry delay (seconds) | `60.0` |

### Alternative Variable Names

Some variables have fallbacks for compatibility:

| Primary | Fallbacks |
|---------|-----------|
| `WATSONX_API_KEY` | `WATSONX_APIKEY` |
| `VISION_MODEL_ID` | `WATSONX_MODEL` |
| `OLLAMA_URL` | `OLLAMA_HOST` |
