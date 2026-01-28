# CLI Reference

The `watsonx-vision` command-line tool provides quick access to document analysis features.

## Installation

```bash
pip install "watsonx-vision-toolkit[cli]"
```

## Global Options

```bash
watsonx-vision --version  # Show version
watsonx-vision --help     # Show help
```

## Commands

### classify

Classify a document image into predefined types.

```bash
watsonx-vision classify IMAGE [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `IMAGE` | Path to image file (PNG, JPG, PDF) |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--types TEXT` | Comma-separated document types | Auto-detect |
| `--provider TEXT` | LLM provider (ollama, watsonx) | From env |
| `--model TEXT` | Model ID | Provider default |
| `--url TEXT` | Provider URL | From env |
| `--output TEXT` | Output format (pretty, json, raw) | `pretty` |
| `--output-file PATH` | Write output to file | - |
| `--cache / --no-cache` | Enable response caching | `false` |
| `--cache-ttl INT` | Cache TTL in seconds | `3600` |

**Examples:**

```bash
# Basic classification
watsonx-vision classify passport.png

# Custom document types
watsonx-vision classify doc.png --types "Invoice,Receipt,Contract"

# JSON output
watsonx-vision classify doc.png --output json

# Save to file
watsonx-vision classify doc.png --output json --output-file result.json

# Use Ollama
watsonx-vision classify doc.png --provider ollama --url http://localhost:11434
```

---

### extract

Extract structured information from a document.

```bash
watsonx-vision extract IMAGE [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--fields TEXT` | Comma-separated fields to extract | Auto-detect |
| `--date-format TEXT` | Output date format | `YYYY-MM-DD` |
| `--provider TEXT` | LLM provider | From env |
| `--output TEXT` | Output format | `pretty` |

**Examples:**

```bash
# Extract default fields
watsonx-vision extract passport.jpg

# Extract specific fields
watsonx-vision extract invoice.png --fields "invoice_number,date,total,vendor"

# Custom date format
watsonx-vision extract doc.png --date-format "MM/DD/YYYY"

# JSON output to file
watsonx-vision extract passport.jpg --output json --output-file data.json
```

---

### validate

Validate document authenticity.

```bash
watsonx-vision validate IMAGE [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--provider TEXT` | LLM provider | From env |
| `--output TEXT` | Output format | `pretty` |

**Examples:**

```bash
# Validate a document
watsonx-vision validate license.png

# JSON output
watsonx-vision validate passport.jpg --output json
```

**Output:**

```
Document Validation Result
==========================
Status: VALID
Reason: Document appears authentic with consistent formatting

Scores:
  Layout Score: 95/100
  Field Score:  90/100

Forgery Signs: None detected
```

---

### fraud

Detect fraud in document images. Supports single and batch processing.

```bash
watsonx-vision fraud IMAGES... [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `IMAGES` | One or more image paths |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--doc-type TEXT` | Document type hint | Auto-detect |
| `--threshold INT` | Confidence threshold (0-100) | `60` |
| `--provider TEXT` | LLM provider | From env |
| `--output TEXT` | Output format | `pretty` |

**Examples:**

```bash
# Single document
watsonx-vision fraud passport.png

# Batch processing
watsonx-vision fraud doc1.png doc2.jpg doc3.pdf

# With document type hint
watsonx-vision fraud tax_return.pdf --doc-type "tax_return"

# Custom threshold
watsonx-vision fraud invoice.png --threshold 80
```

**Output:**

```
Fraud Detection Results
=======================

Document: passport.png
  Valid: Yes
  Confidence: 92%
  Severity: NONE

Document: suspicious.pdf
  Valid: No
  Confidence: 45%
  Severity: HIGH
  Issues:
    - Inconsistent font rendering
    - Digital manipulation artifacts

Summary:
  Total: 2 documents
  Valid: 1 (50%)
  Invalid: 1 (50%)
```

---

### analyze

Analyze an image with a custom prompt.

```bash
watsonx-vision analyze IMAGE PROMPT [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `IMAGE` | Path to image file |
| `PROMPT` | Analysis prompt |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--system TEXT` | System prompt | - |
| `--raw` | Return raw text (no JSON parsing) | `false` |
| `--provider TEXT` | LLM provider | From env |
| `--output TEXT` | Output format | `pretty` |

**Examples:**

```bash
# Custom analysis
watsonx-vision analyze receipt.jpg "What is the total amount?"

# With system prompt
watsonx-vision analyze doc.png "List all dates" --system "You are a document analyst"

# Raw text output
watsonx-vision analyze image.png "Describe this image" --raw
```

---

### config

Show or validate configuration.

```bash
watsonx-vision config [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--show` | Display current configuration |
| `--validate` | Validate configuration |
| `--env` | Show environment variable reference |

**Examples:**

```bash
# Show current config
watsonx-vision config --show

# Validate config
watsonx-vision config --validate

# Show env var reference
watsonx-vision config --env
```

**Output (--show):**

```
Current Configuration
=====================
Provider: ollama
Model: llava:latest
URL: http://localhost:11434
Max Tokens: 2000
Temperature: 0.0
Cache Enabled: No
```

**Output (--env):**

```
Environment Variables
=====================

  VISION_PROVIDER: ollama
  VISION_MODEL_ID: llava:latest
  WATSONX_API_KEY: (not set)
  WATSONX_URL: (not set)
  WATSONX_PROJECT_ID: (not set)
  OLLAMA_URL: http://localhost:11434
  VISION_MAX_TOKENS: (not set)
  VISION_TEMPERATURE: (not set)
  VISION_CACHE_ENABLED: (not set)
  VISION_CACHE_TTL: (not set)
```

## Output Formats

### pretty (default)

Human-readable formatted output with colors and structure.

### json

Machine-readable JSON output, suitable for piping to other tools:

```bash
watsonx-vision classify doc.png --output json | jq '.doc_type'
```

### raw

Raw LLM response without parsing (for `analyze` command with `--raw`).

## Environment Variables

The CLI reads configuration from environment variables:

```bash
export VISION_PROVIDER=ollama
export OLLAMA_URL=http://localhost:11434
export VISION_MODEL_ID=llava:13b
```

Command-line options override environment variables.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (invalid input, file not found, etc.) |
| `2` | Configuration error |
