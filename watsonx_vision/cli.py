"""
Watsonx Vision Toolkit CLI

Command-line interface for document analysis, fraud detection, and validation.

Usage:
    watsonx-vision classify <image>
    watsonx-vision extract <image>
    watsonx-vision validate <image>
    watsonx-vision fraud <image>
    watsonx-vision config
"""

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    print("CLI requires click package. Install with: pip install click")
    sys.exit(1)

from .vision_llm import VisionLLM, VisionLLMConfig, LLMProvider
from .fraud_detector import FraudDetector
from .cache import CacheConfig
from .exceptions import WatsonxVisionError


def get_llm(
    provider: str,
    model: Optional[str],
    url: Optional[str],
    cache_enabled: bool,
    cache_ttl: float,
) -> VisionLLM:
    """Create VisionLLM instance from CLI options or environment."""
    # Try to load from environment first, then override with CLI options
    try:
        config = VisionLLMConfig.from_env()
    except Exception:
        config = VisionLLMConfig()

    # Override with CLI options if provided
    if provider:
        config.provider = LLMProvider(provider.lower())

    if model:
        config.model_id = model

    if url:
        config.url = url

    # Setup cache if enabled
    cache_config = None
    if cache_enabled:
        cache_config = CacheConfig(enabled=True, ttl=cache_ttl)

    return VisionLLM(config, cache_config=cache_config)


def load_image(image_path: str) -> str:
    """Load and encode image to base64."""
    path = Path(image_path)
    if not path.exists():
        raise click.ClickException(f"Image file not found: {image_path}")

    if not path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif', '.webp'):
        raise click.ClickException(
            f"Unsupported image format: {path.suffix}. "
            "Supported: .png, .jpg, .jpeg, .gif, .webp"
        )

    return VisionLLM.encode_image_to_base64(str(path))


def output_result(result: dict, output_format: str, output_file: Optional[str]):
    """Output result in specified format."""
    if output_format == "json":
        output = json.dumps(result, indent=2)
    elif output_format == "pretty":
        output = format_pretty(result)
    else:
        output = str(result)

    if output_file:
        Path(output_file).write_text(output)
        click.echo(f"Output written to: {output_file}")
    else:
        click.echo(output)


def format_pretty(data: dict, indent: int = 0) -> str:
    """Format dictionary for human-readable output."""
    lines = []
    prefix = "  " * indent

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_pretty(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(format_pretty(item, indent + 1))
                else:
                    lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


# Common options
provider_option = click.option(
    "--provider", "-p",
    type=click.Choice(["watsonx", "ollama", "openai", "anthropic"]),
    help="LLM provider (default: from env or watsonx)"
)
model_option = click.option(
    "--model", "-m",
    help="Model ID to use"
)
url_option = click.option(
    "--url", "-u",
    help="Provider URL"
)
output_option = click.option(
    "--output", "-o",
    type=click.Choice(["json", "pretty", "raw"]),
    default="pretty",
    help="Output format (default: pretty)"
)
output_file_option = click.option(
    "--output-file", "-f",
    help="Write output to file instead of stdout"
)
cache_option = click.option(
    "--cache/--no-cache",
    default=False,
    help="Enable response caching"
)
cache_ttl_option = click.option(
    "--cache-ttl",
    default=3600.0,
    help="Cache TTL in seconds (default: 3600)"
)


def get_version():
    """Get package version."""
    try:
        from importlib.metadata import version
        return version("watsonx-vision-toolkit")
    except Exception:
        # Fallback for development
        return "0.2.0-dev"


@click.group()
@click.version_option(version=get_version())
def cli():
    """Watsonx Vision Toolkit - Document analysis and validation CLI.

    Analyze documents, detect fraud, and validate information using
    vision-based LLM models.

    \b
    Examples:
        watsonx-vision classify document.png
        watsonx-vision extract passport.jpg --output json
        watsonx-vision fraud invoice.pdf --provider ollama
        watsonx-vision config --show
    """
    pass


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.option(
    "--types", "-t",
    multiple=True,
    help="Document types to classify (can specify multiple)"
)
@provider_option
@model_option
@url_option
@output_option
@output_file_option
@cache_option
@cache_ttl_option
def classify(
    image: str,
    types: tuple,
    provider: Optional[str],
    model: Optional[str],
    url: Optional[str],
    output: str,
    output_file: Optional[str],
    cache: bool,
    cache_ttl: float,
):
    """Classify a document image into predefined types.

    \b
    Examples:
        watsonx-vision classify document.png
        watsonx-vision classify scan.jpg --types "Passport" --types "License"
        watsonx-vision classify form.png --provider ollama --output json
    """
    try:
        llm = get_llm(provider, model, url, cache, cache_ttl)
        image_data = load_image(image)

        document_types = list(types) if types else None
        result = llm.classify_document(image_data, document_types=document_types)

        output_result(result, output, output_file)

    except WatsonxVisionError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Classification failed: {e}")


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.option(
    "--fields", "-F",
    multiple=True,
    help="Fields to extract (can specify multiple)"
)
@click.option(
    "--date-format",
    default="YYYY-MM-DD",
    help="Date format for extraction (default: YYYY-MM-DD)"
)
@provider_option
@model_option
@url_option
@output_option
@output_file_option
@cache_option
@cache_ttl_option
def extract(
    image: str,
    fields: tuple,
    date_format: str,
    provider: Optional[str],
    model: Optional[str],
    url: Optional[str],
    output: str,
    output_file: Optional[str],
    cache: bool,
    cache_ttl: float,
):
    """Extract structured information from a document.

    \b
    Examples:
        watsonx-vision extract passport.jpg
        watsonx-vision extract form.png --fields "Name" --fields "Address"
        watsonx-vision extract license.jpg --output json --output-file data.json
    """
    try:
        llm = get_llm(provider, model, url, cache, cache_ttl)
        image_data = load_image(image)

        field_list = list(fields) if fields else None
        result = llm.extract_information(
            image_data,
            fields=field_list,
            date_format=date_format
        )

        output_result(result, output, output_file)

    except WatsonxVisionError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Extraction failed: {e}")


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@provider_option
@model_option
@url_option
@output_option
@output_file_option
@cache_option
@cache_ttl_option
def validate(
    image: str,
    provider: Optional[str],
    model: Optional[str],
    url: Optional[str],
    output: str,
    output_file: Optional[str],
    cache: bool,
    cache_ttl: float,
):
    """Validate document authenticity.

    Checks for:
    - Layout consistency
    - Field formatting
    - Signs of forgery or manipulation

    \b
    Examples:
        watsonx-vision validate document.png
        watsonx-vision validate suspicious.jpg --output json
    """
    try:
        llm = get_llm(provider, model, url, cache, cache_ttl)
        image_data = load_image(image)

        result = llm.validate_authenticity(image_data)

        # Add status indicator for pretty output
        if output == "pretty":
            status = "VALID" if result.get("valid") else "INVALID"
            click.echo(f"\nDocument Status: {status}\n")

        output_result(result, output, output_file)

    except WatsonxVisionError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Validation failed: {e}")


@cli.command()
@click.argument("images", type=click.Path(exists=True), nargs=-1, required=True)
@click.option(
    "--doc-type",
    help="Document type for specialized detection"
)
@click.option(
    "--threshold",
    default=0.7,
    type=float,
    help="Fraud detection threshold (0.0-1.0, default: 0.7)"
)
@provider_option
@model_option
@url_option
@output_option
@output_file_option
@cache_option
@cache_ttl_option
def fraud(
    images: tuple,
    doc_type: Optional[str],
    threshold: float,
    provider: Optional[str],
    model: Optional[str],
    url: Optional[str],
    output: str,
    output_file: Optional[str],
    cache: bool,
    cache_ttl: float,
):
    """Detect fraud in document images.

    Analyzes documents for signs of manipulation, forgery, or inconsistencies.
    Supports single or batch processing.

    \b
    Examples:
        watsonx-vision fraud document.png
        watsonx-vision fraud doc1.png doc2.png doc3.png
        watsonx-vision fraud passport.jpg --doc-type passport
    """
    try:
        llm = get_llm(provider, model, url, cache, cache_ttl)
        detector = FraudDetector(llm, threshold=threshold)

        if len(images) == 1:
            # Single document
            image_data = load_image(images[0])
            result = detector.validate_document(
                image_data,
                filename=Path(images[0]).name,
                document_type=doc_type,
            )

            # Convert to dict for output
            result_dict = {
                "filename": result.filename,
                "is_valid": result.is_valid,
                "confidence": result.confidence,
                "severity": result.severity.value,
                "issues": result.issues,
                "details": result.details,
            }

            if output == "pretty":
                status = "VALID" if result.is_valid else f"FRAUD DETECTED ({result.severity.value})"
                click.echo(f"\nDocument: {result.filename}")
                click.echo(f"Status: {status}")
                click.echo(f"Confidence: {result.confidence:.2%}\n")

            output_result(result_dict, output, output_file)

        else:
            # Batch processing
            documents = []
            for img_path in images:
                image_data = load_image(img_path)
                documents.append({
                    "image_data": image_data,
                    "filename": Path(img_path).name,
                    "document_type": doc_type,
                })

            results = detector.validate_batch(documents)

            # Convert to list of dicts
            results_list = []
            for r in results:
                results_list.append({
                    "filename": r.filename,
                    "is_valid": r.is_valid,
                    "confidence": r.confidence,
                    "severity": r.severity.value,
                    "issues": r.issues,
                })

            if output == "pretty":
                click.echo(f"\nBatch Results ({len(results)} documents):\n")
                for r in results:
                    status = "VALID" if r.is_valid else f"FRAUD ({r.severity.value})"
                    click.echo(f"  {r.filename}: {status} ({r.confidence:.2%})")
                click.echo()

            output_result({"results": results_list}, output, output_file)

    except WatsonxVisionError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Fraud detection failed: {e}")


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.argument("prompt")
@click.option(
    "--system", "-s",
    help="System prompt for context"
)
@click.option(
    "--raw/--json",
    default=False,
    help="Return raw text instead of parsing as JSON"
)
@provider_option
@model_option
@url_option
@output_option
@output_file_option
@cache_option
@cache_ttl_option
def analyze(
    image: str,
    prompt: str,
    system: Optional[str],
    raw: bool,
    provider: Optional[str],
    model: Optional[str],
    url: Optional[str],
    output: str,
    output_file: Optional[str],
    cache: bool,
    cache_ttl: float,
):
    """Analyze an image with a custom prompt.

    \b
    Examples:
        watsonx-vision analyze photo.jpg "Describe what you see"
        watsonx-vision analyze doc.png "List all text" --raw
        watsonx-vision analyze chart.png "Extract data" --system "You are a data analyst"
    """
    try:
        llm = get_llm(provider, model, url, cache, cache_ttl)
        image_data = load_image(image)

        result = llm.analyze_image(
            image_data,
            prompt=prompt,
            system_prompt=system,
            parse_json=not raw,
        )

        if raw or isinstance(result, str):
            click.echo(result)
        else:
            output_result(result, output, output_file)

    except WatsonxVisionError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Analysis failed: {e}")


@cli.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--validate", "validate_config", is_flag=True, help="Validate configuration")
@click.option("--env", is_flag=True, help="Show environment variables")
def config(show: bool, validate_config: bool, env: bool):
    """Show or validate configuration.

    \b
    Examples:
        watsonx-vision config --show
        watsonx-vision config --validate
        watsonx-vision config --env
    """
    import os

    if env:
        click.echo("Environment Variables:")
        click.echo()
        env_vars = [
            ("VISION_PROVIDER", "LLM Provider"),
            ("VISION_MODEL_ID", "Model ID"),
            ("WATSONX_API_KEY", "Watsonx API Key"),
            ("WATSONX_URL", "Watsonx URL"),
            ("WATSONX_PROJECT_ID", "Watsonx Project ID"),
            ("OLLAMA_URL", "Ollama URL"),
            ("VISION_MAX_TOKENS", "Max Tokens"),
            ("VISION_TEMPERATURE", "Temperature"),
            ("VISION_CACHE_ENABLED", "Cache Enabled"),
            ("VISION_CACHE_TTL", "Cache TTL"),
        ]

        for var, desc in env_vars:
            value = os.environ.get(var)
            if value:
                # Mask sensitive values
                if "KEY" in var or "SECRET" in var:
                    display = value[:8] + "..." if len(value) > 8 else "***"
                else:
                    display = value
                click.echo(f"  {var}: {display}")
            else:
                click.echo(f"  {var}: (not set)")
        return

    if show or validate_config:
        try:
            config = VisionLLMConfig.from_env()
            cache_config = CacheConfig.from_env()

            click.echo("Current Configuration:")
            click.echo()
            click.echo(f"  Provider: {config.provider.value}")
            click.echo(f"  Model: {config.model_id}")
            click.echo(f"  URL: {config.url or '(not set)'}")
            click.echo(f"  API Key: {'***' if config.api_key else '(not set)'}")
            click.echo(f"  Project ID: {config.project_id or '(not set)'}")
            click.echo()
            click.echo(f"  Max Tokens: {config.max_tokens}")
            click.echo(f"  Temperature: {config.temperature}")
            click.echo(f"  Top P: {config.top_p}")
            click.echo()
            click.echo(f"  Retry Enabled: {config.retry_enabled}")
            click.echo(f"  Retry Max Attempts: {config.retry_max_attempts}")
            click.echo()
            click.echo(f"  Cache Enabled: {cache_config.enabled}")
            click.echo(f"  Cache TTL: {cache_config.ttl}s")
            click.echo(f"  Cache Max Size: {cache_config.max_size}")

            if validate_config:
                click.echo()
                # Check for required settings based on provider
                issues = []
                if config.provider == LLMProvider.WATSONX:
                    if not config.api_key:
                        issues.append("WATSONX_API_KEY is required")
                    if not config.project_id:
                        issues.append("WATSONX_PROJECT_ID is required")

                if issues:
                    click.echo("Configuration Issues:")
                    for issue in issues:
                        click.echo(f"  - {issue}")
                    sys.exit(1)
                else:
                    click.echo("Configuration is valid.")

        except Exception as e:
            raise click.ClickException(f"Configuration error: {e}")

    if not any([show, validate_config, env]):
        # Show help by default
        ctx = click.get_current_context()
        click.echo(ctx.get_help())


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
