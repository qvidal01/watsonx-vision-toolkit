# Examples

Practical code examples for common use cases.

## Available Examples

| Example | Description |
|---------|-------------|
| [Document Classification](classification.md) | Classify documents into types |
| [Information Extraction](extraction.md) | Extract structured data from documents |
| [Fraud Detection](fraud.md) | Detect document fraud and manipulation |
| [Cross-Validation](cross-validation.md) | Validate data across multiple documents |
| [Complete Workflow](workflow.md) | End-to-end loan application processing |

## Running Examples

All examples are available in the `examples/` directory of the repository.

### Setup

```bash
# Clone the repository
git clone https://github.com/qvidal01/watsonx-vision-toolkit.git
cd watsonx-vision-toolkit

# Install with all dependencies
pip install -e ".[all]"

# Set up environment variables
export VISION_PROVIDER=ollama
export OLLAMA_URL=http://localhost:11434
export VISION_MODEL_ID=llava:latest
```

### Run an Example

```bash
# Run basic classification
python examples/basic_classification.py

# Run fraud detection
python examples/fraud_detection.py

# Run complete workflow
python examples/complete_workflow.py
```

## Example Files

```
examples/
├── README.md                  # Example documentation
├── basic_classification.py    # Document classification
├── information_extraction.py  # Data extraction with presets
├── fraud_detection.py         # Single and batch fraud detection
├── cross_validation.py        # Multi-document validation
├── retry_configuration.py     # Retry behavior configuration
├── environment_config.py      # Environment variable configuration
├── response_caching.py        # Response caching examples
└── complete_workflow.py       # End-to-end loan processing
```

## Quick Example

```python
from watsonx_vision import VisionLLM, VisionLLMConfig

# Configure
config = VisionLLMConfig(
    provider="ollama",
    model_id="llava:latest",
    url="http://localhost:11434"
)

llm = VisionLLM(config)

# Classify a document
image = VisionLLM.encode_image_to_base64("document.png")
result = llm.classify_document(image)
print(f"Document type: {result['doc_type']}")
```
