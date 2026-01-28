# Quick Start

This guide will get you analyzing documents in under 5 minutes.

## Setup

First, ensure you have the toolkit installed with your preferred provider:

```bash
pip install "watsonx-vision-toolkit[ollama]"  # For local Ollama
# or
pip install "watsonx-vision-toolkit[watsonx]" # For IBM Watsonx
```

## Configure the LLM

=== "Ollama"

    ```python
    from watsonx_vision import VisionLLM, VisionLLMConfig

    config = VisionLLMConfig(
        provider="ollama",
        model_id="llava:latest",
        url="http://localhost:11434"
    )

    llm = VisionLLM(config)
    ```

=== "IBM Watsonx"

    ```python
    from watsonx_vision import VisionLLM, VisionLLMConfig

    config = VisionLLMConfig(
        provider="watsonx",
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        api_key="your-api-key",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="your-project-id"
    )

    llm = VisionLLM(config)
    ```

=== "Environment Variables"

    ```bash
    export VISION_PROVIDER=ollama
    export OLLAMA_URL=http://localhost:11434
    export VISION_MODEL_ID=llava:latest
    ```

    ```python
    from watsonx_vision import VisionLLM, VisionLLMConfig

    config = VisionLLMConfig.from_env()
    llm = VisionLLM(config)
    ```

## Classify a Document

```python
# Load and encode the image
image_data = VisionLLM.encode_image_to_base64("document.png")

# Classify
result = llm.classify_document(image_data)
print(result)
# {"doc_type": "Passport"}
```

With custom document types:

```python
result = llm.classify_document(
    image_data,
    document_types=["Invoice", "Receipt", "Contract", "Report"]
)
```

## Extract Information

```python
# Extract with default fields
info = llm.extract_information(image_data)
print(info)
# {
#     "name": "John Doe",
#     "dob": "1990-01-15",
#     "address": "123 Main St",
#     ...
# }
```

With custom fields:

```python
info = llm.extract_information(
    image_data,
    fields=["invoice_number", "date", "total", "vendor_name"],
    date_format="MM/DD/YYYY"
)
```

## Validate Authenticity

```python
result = llm.validate_authenticity(image_data)

if result["valid"]:
    print(f"Document appears authentic")
    print(f"Layout score: {result['layout_score']}")
    print(f"Field score: {result['field_score']}")
else:
    print(f"Issues detected: {result['reason']}")
    print(f"Forgery signs: {result['forgery_signs']}")
```

## Custom Analysis

For custom prompts:

```python
result = llm.analyze_image(
    image_data,
    prompt="List all dates visible in this document",
    system_prompt="You are a document analysis assistant"
)
```

## Using the CLI

Quick analysis from the command line:

```bash
# Classify
watsonx-vision classify document.png

# Extract (JSON output)
watsonx-vision extract passport.jpg --output json

# Validate
watsonx-vision validate license.png

# Custom analysis
watsonx-vision analyze receipt.jpg "What is the total amount?"
```

## Next Steps

- [Configuration](configuration.md) - Environment variables and advanced config
- [Fraud Detection](../examples/fraud.md) - Build a fraud detection pipeline
- [CLI Reference](../guide/cli.md) - Full CLI documentation
