# Watsonx Vision Toolkit

A reusable Python toolkit for vision-based document analysis using IBM Watsonx AI or Ollama. Includes fraud detection, document classification, information extraction, cross-validation, and multi-criteria decision engines.

**Extracted from the IBM Watsonx Loan Preprocessing Agents project with proven production results.**

## Features

- **Vision LLM Interface** - Unified API for IBM Watsonx AI and Ollama vision models
- **Document Classification** - Automatically classify documents (passport, license, tax returns, etc.)
- **Information Extraction** - Extract structured data from document images (PII, financial data)
- **Fraud Detection** - Detect document forgery, manipulation, and authenticity issues
- **Cross-Validation** - Compare data across multiple documents for consistency
- **Decision Engine** - Multi-criteria weighted scoring for automated decisions

## Installation

```bash
# Basic installation (no provider dependencies)
pip install watsonx-vision-toolkit

# With IBM Watsonx AI support
pip install watsonx-vision-toolkit[watsonx]

# With Ollama support
pip install watsonx-vision-toolkit[ollama]

# With all providers
pip install watsonx-vision-toolkit[all]

# Development installation
pip install watsonx-vision-toolkit[dev]
```

### From Source

```bash
git clone https://github.com/qvidal01/watsonx-vision-toolkit.git
cd watsonx-vision-toolkit
pip install -e ".[all,dev]"
```

## Quick Start

### 1. Vision LLM - Document Classification

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, LLMProvider

# Configure for IBM Watsonx AI
config = VisionLLMConfig(
    provider=LLMProvider.WATSONX,
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    api_key="your-ibm-cloud-api-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project-id"
)

# Or configure for Ollama (local)
config = VisionLLMConfig(
    provider=LLMProvider.OLLAMA,
    model_id="llava:13b",
    url="http://localhost:11434"
)

# Initialize and use
llm = VisionLLM(config)

# Encode image
image_data = VisionLLM.encode_image_to_base64("path/to/document.png")

# Classify document
result = llm.classify_document(image_data)
print(f"Document type: {result['doc_type']}")
# Output: Document type: Passport

# Extract information
info = llm.extract_information(image_data)
print(f"Name: {info.get('name')}")
print(f"DOB: {info.get('dob')}")
# Output: Name: John Doe
# Output: DOB: 1990-05-15
```

### 2. Fraud Detection

```python
from watsonx_vision import VisionLLM, VisionLLMConfig, FraudDetector

# Initialize vision LLM (see above)
llm = VisionLLM(config)

# Create fraud detector
detector = FraudDetector(
    vision_llm=llm,
    layout_threshold=70,  # Minimum layout score
    field_threshold=70,   # Minimum field score
    min_confidence=60     # Minimum overall confidence
)

# Validate single document
image_data = VisionLLM.encode_image_to_base64("passport.png")
result = detector.validate_document(image_data, filename="passport.png")

if result.valid:
    print(f"Document is authentic (confidence: {result.confidence}%)")
else:
    print(f"Fraud detected: {result.reason}")
    print(f"Severity: {result.severity.value}")
    print(f"Issues: {result.forgery_signs}")

# Validate batch of documents
documents = [
    {"image_data": img1, "filename": "passport.png"},
    {"image_data": img2, "filename": "license.png"},
    {"image_data": img3, "filename": "bank_statement.png"}
]
results = detector.validate_batch(documents)

# Generate report
report = detector.generate_report(results)
print(f"Fraud rate: {report['fraud_rate']}%")
```

### 3. Cross-Validation

```python
from watsonx_vision import CrossValidator

# Initialize validator
validator = CrossValidator(
    api_key="your-ibm-cloud-api-key",
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
        "dob": "1990-05-15",
        "passport_number": "X12345678"
    },
    {
        "doc_type": "Tax Return",
        "name": "John D. Doe",  # Slight variation
        "ssn": "123-45-6789",
        "annual_income": 75000
    },
    {
        "doc_type": "Bank Statement",
        "name": "John Doe",
        "account_number": "****1234"
    }
]

# Validate
result = validator.validate(application_data, document_data)

if result.passed:
    print("All data is consistent!")
else:
    print(f"Found {result.total_inconsistencies} inconsistencies:")
    for issue in result.inconsistencies:
        print(f"  - {issue.field}: {issue.explanation} [{issue.severity.value}]")

# Generate human-readable report
print(validator.generate_report(result))
```

### 4. Decision Engine

```python
from watsonx_vision import DecisionEngine, LoanDecisionEngine

# Basic decision engine
engine = DecisionEngine(
    approval_threshold=75.0,
    rejection_threshold=40.0,
    fraud_weight=0.4,
    cross_validation_weight=0.3,
    custom_criteria_weight=0.3
)

# Add custom criteria
engine.add_criterion(
    "minimum_age",
    weight=0.5,
    evaluator=lambda data: data.get("age", 0) >= 18
)

engine.add_criterion(
    "income_requirement",
    weight=0.5,
    evaluator=lambda data: data.get("annual_income", 0) >= 50000
)

# Make decision
decision = engine.decide(
    fraud_results=fraud_detector_results,    # List[FraudResult]
    validation_result=cross_validation_result,  # ValidationResult
    custom_data={"age": 25, "annual_income": 75000}
)

print(decision.summary())
# Output: ✅ APPROVED (Score: 85.5/100)

# Get detailed results
print(f"Status: {decision.status.value}")
print(f"Reasons: {decision.reasons}")
print(f"Recommendations: {decision.recommendations}")

# Or use the pre-configured loan decision engine
loan_engine = LoanDecisionEngine(
    min_age=18,
    min_income=30000,
    max_dti=0.43
)

decision = loan_engine.decide(
    fraud_results=fraud_results,
    validation_result=validation_result,
    custom_data={
        "age": 35,
        "annual_income": 85000,
        "monthly_debt": 1500,
        "monthly_income": 7000
    }
)
```

## Complete Example: Loan Application Processing

```python
from watsonx_vision import (
    VisionLLM, VisionLLMConfig, LLMProvider,
    FraudDetector, CrossValidator, LoanDecisionEngine
)

# 1. Setup
config = VisionLLMConfig(
    provider=LLMProvider.WATSONX,
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    api_key="your-api-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project-id"
)

vision_llm = VisionLLM(config)
fraud_detector = FraudDetector(vision_llm)
cross_validator = CrossValidator(
    api_key="your-api-key",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your-project-id"
)
decision_engine = LoanDecisionEngine(min_age=18, min_income=30000)

# 2. Process documents
documents = ["passport.png", "tax_return.png", "bank_statement.png"]
document_data = []
fraud_results = []

for doc_path in documents:
    # Encode image
    image_data = VisionLLM.encode_image_to_base64(doc_path)

    # Classify
    doc_type = vision_llm.classify_document(image_data)

    # Extract info
    extracted = vision_llm.extract_information(image_data)
    extracted["doc_type"] = doc_type["doc_type"]
    extracted["filename"] = doc_path
    document_data.append(extracted)

    # Fraud check
    fraud_result = fraud_detector.validate_document(image_data, doc_path)
    fraud_results.append(fraud_result)

# 3. Cross-validate
application_data = {
    "name": "John Doe",
    "dob": "1990-05-15",
    "annual_income": 75000
}
validation_result = cross_validator.validate(application_data, document_data)

# 4. Make decision
decision = decision_engine.decide(
    fraud_results=fraud_results,
    validation_result=validation_result,
    custom_data=application_data
)

# 5. Output results
print(decision.summary())
print(decision.to_dict())
```

## Environment Variables

The toolkit can be configured via environment variables:

```bash
# IBM Watsonx AI
export WATSONX_APIKEY="your-api-key"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID="your-project-id"

# Ollama (optional)
export OLLAMA_URL="http://localhost:11434"
```

## Supported Models

### IBM Watsonx AI

| Model | Type | Best For |
|-------|------|----------|
| `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` | Vision | Document analysis, classification |
| `mistralai/mistral-medium-2505` | Text | Cross-validation, decision logic |
| `ibm/granite-3-8b-instruct` | Text | General processing |

### Ollama (Local)

| Model | Type | Best For |
|-------|------|----------|
| `llava:13b` | Vision | Document analysis |
| `llava:34b` | Vision | High-accuracy analysis |
| `mistral:7b` | Text | Cross-validation |

## API Reference

### VisionLLM

```python
class VisionLLM:
    def __init__(self, config: VisionLLMConfig): ...
    def classify_document(self, image_data: str, document_types: List[str] = None) -> Dict: ...
    def extract_information(self, image_data: str, fields: List[str] = None) -> Dict: ...
    def validate_authenticity(self, image_data: str) -> Dict: ...
    def analyze_image(self, image_data: str, prompt: str, system_prompt: str = None) -> Dict: ...

    @staticmethod
    def encode_image_to_base64(image_path: str, mime_type: str = None) -> str: ...
```

### FraudDetector

```python
class FraudDetector:
    def __init__(self, vision_llm: VisionLLM, layout_threshold: int = 70, ...): ...
    def validate_document(self, image_data: str, filename: str = None) -> FraudResult: ...
    def validate_batch(self, documents: List[Dict]) -> List[FraudResult]: ...
    def generate_report(self, results: List[FraudResult]) -> Dict: ...
```

### CrossValidator

```python
class CrossValidator:
    def __init__(self, api_key: str, url: str, project_id: str, ...): ...
    def validate(self, application_data: Dict, document_data: List[Dict]) -> ValidationResult: ...
    def validate_batch(self, packages: List[Dict]) -> List[ValidationResult]: ...
    def generate_report(self, result: ValidationResult) -> str: ...
```

### DecisionEngine

```python
class DecisionEngine:
    def __init__(self, approval_threshold: float = 75.0, ...): ...
    def add_criterion(self, name: str, weight: float, evaluator: Callable): ...
    def remove_criterion(self, name: str): ...
    def decide(self, fraud_results: List, validation_result: ValidationResult, ...) -> Decision: ...
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=watsonx_vision --cov-report=html

# Run specific test file
pytest tests/test_vision_llm.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file.

## Credits

- **Author**: AIQSO - Quinn Vidal (quinn@aiqso.io)
- **Extracted from**: IBM Watsonx Loan Preprocessing Agents project
- **Powered by**: IBM Watsonx AI, Ollama, LangChain

## Related Projects

- [IBM dsce-sample-apps](https://github.com/IBM/dsce-sample-apps) - Original IBM sample applications
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [Ollama](https://ollama.ai/) - Local LLM runtime
