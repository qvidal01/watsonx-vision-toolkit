# Installation

## Requirements

- Python 3.9 or higher
- pip or uv package manager

## Install from PyPI

### Basic Installation

Install the core package without any provider dependencies:

```bash
pip install watsonx-vision-toolkit
```

### With Provider Support

=== "Ollama (Local)"

    ```bash
    pip install "watsonx-vision-toolkit[ollama]"
    ```

    This installs `langchain-ollama` for local LLM inference.

=== "IBM Watsonx"

    ```bash
    pip install "watsonx-vision-toolkit[watsonx]"
    ```

    This installs `langchain-ibm` and `ibm-watson-machine-learning`.

=== "All Providers"

    ```bash
    pip install "watsonx-vision-toolkit[all]"
    ```

    Installs all provider dependencies plus CLI support.

### With CLI Support

```bash
pip install "watsonx-vision-toolkit[cli]"
```

This installs the `watsonx-vision` command-line tool.

## Install from Source

Clone and install in development mode:

```bash
git clone https://github.com/qvidal01/watsonx-vision-toolkit.git
cd watsonx-vision-toolkit
pip install -e ".[all,dev]"
```

## Using uv

If you prefer [uv](https://github.com/astral-sh/uv) for faster installs:

```bash
uv pip install "watsonx-vision-toolkit[all]"
```

## Verify Installation

```python
import watsonx_vision
print(watsonx_vision.__version__)  # 0.2.0
```

Or via CLI:

```bash
watsonx-vision --version
```

## Provider Setup

### Ollama

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a vision model:
   ```bash
   ollama pull llava:latest
   # or for better accuracy:
   ollama pull llava:13b
   ```
3. Ensure Ollama is running on `http://localhost:11434`

### IBM Watsonx

1. Create an [IBM Cloud account](https://cloud.ibm.com/)
2. Create a Watsonx.ai project
3. Generate an API key
4. Note your project ID and region URL

## Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `watsonx` | langchain-ibm, ibm-watson-machine-learning | IBM Watsonx support |
| `ollama` | langchain-ollama | Ollama support |
| `cli` | click | Command-line interface |
| `all` | All of the above | Full installation |
| `dev` | pytest, ruff, mypy, black | Development tools |

## Troubleshooting

### ImportError: No module named 'langchain_ollama'

You need to install the Ollama extra:

```bash
pip install "watsonx-vision-toolkit[ollama]"
```

### ImportError: No module named 'langchain_ibm'

You need to install the Watsonx extra:

```bash
pip install "watsonx-vision-toolkit[watsonx]"
```

### Command not found: watsonx-vision

Install the CLI extra:

```bash
pip install "watsonx-vision-toolkit[cli]"
```
