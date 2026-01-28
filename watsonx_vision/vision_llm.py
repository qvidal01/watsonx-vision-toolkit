"""
Vision LLM Module

Provides a unified interface for vision-based document analysis using various LLM providers.
Supports IBM Watsonx AI, local Ollama models, and other compatible providers.

Extracted from IBM Watsonx Loan Preprocessing Agents project.
"""

import base64
import logging
import mimetypes
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

from .exceptions import (
    LLMConnectionError,
    LLMResponseError,
    LLMParseError,
    LLMTimeoutError,
    ConfigurationError,
)
from .retry import (
    RetryConfig,
    retry_llm_call,
    async_retry_llm_call,
    DEFAULT_RETRY_CONFIG,
)

try:
    from langchain_ibm import ChatWatsonx
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import JsonOutputParser
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    WATSONX = "watsonx"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class VisionLLMConfig:
    """Configuration for Vision LLM"""
    provider: LLMProvider = LLMProvider.WATSONX
    model_id: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    api_key: Optional[str] = None
    url: Optional[str] = None
    project_id: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.0
    top_p: float = 0.1
    # Retry configuration
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0


class VisionLLM:
    """
    Vision-based LLM for document analysis.

    Supports multiple providers with a unified interface for:
    - Document classification
    - Information extraction
    - Visual analysis
    - OCR and text detection

    Example:
        >>> config = VisionLLMConfig(
        ...     provider=LLMProvider.WATSONX,
        ...     model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        ...     api_key="your-api-key",
        ...     url="https://us-south.ml.cloud.ibm.com",
        ...     project_id="your-project-id"
        ... )
        >>> llm = VisionLLM(config)
        >>> result = llm.classify_document(image_base64)
    """

    def __init__(self, config: VisionLLMConfig):
        """
        Initialize Vision LLM with configuration.

        Args:
            config: VisionLLMConfig object with provider settings

        Raises:
            ImportError: If required packages for provider are not installed
            ValueError: If configuration is invalid
        """
        self.config = config
        self._llm = self._initialize_llm()
        self._parser = JsonOutputParser()

        # Initialize retry configuration
        if config.retry_enabled:
            self._retry_config = RetryConfig(
                max_attempts=config.retry_max_attempts,
                base_delay=config.retry_base_delay,
                max_delay=config.retry_max_delay,
            )
        else:
            self._retry_config = None

    def _initialize_llm(self):
        """Initialize the LLM based on provider configuration"""
        if self.config.provider == LLMProvider.WATSONX:
            if not WATSONX_AVAILABLE:
                raise ImportError(
                    "langchain-ibm is required for Watsonx provider. "
                    "Install with: pip install langchain-ibm"
                )

            return ChatWatsonx(
                model_id=self.config.model_id,
                apikey=self.config.api_key,
                url=self.config.url,
                project_id=self.config.project_id,
                params={
                    "max_new_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                }
            )

        elif self.config.provider == LLMProvider.OLLAMA:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                raise ImportError(
                    "langchain-ollama is required for Ollama provider. "
                    "Install with: pip install langchain-ollama"
                )

            return ChatOllama(
                model=self.config.model_id,
                base_url=self.config.url or "http://localhost:11434",
                temperature=self.config.temperature,
            )

        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _create_vision_message(
        self,
        image_data: str,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List[Union[SystemMessage, HumanMessage]]:
        """
        Create messages for vision LLM with image and text.

        Args:
            image_data: Base64-encoded image data URI (data:image/png;base64,...)
            prompt: User prompt/question about the image
            system_prompt: Optional system prompt for context

        Returns:
            List of messages for LLM
        """
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=content))

        return messages

    def analyze_image(
        self,
        image_data: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        parse_json: bool = True
    ) -> Union[Dict, str]:
        """
        Analyze an image with a custom prompt.

        Args:
            image_data: Base64-encoded image data URI
            prompt: Analysis prompt
            system_prompt: Optional system context
            parse_json: Whether to parse response as JSON

        Returns:
            Parsed JSON dict if parse_json=True, else raw string

        Raises:
            LLMConnectionError: If connection to LLM fails
            LLMResponseError: If LLM returns invalid response
            LLMParseError: If JSON parsing fails
            LLMTimeoutError: If request times out
        """
        messages = self._create_vision_message(image_data, prompt, system_prompt)

        def invoke_llm():
            """Inner function for retry wrapper"""
            try:
                return self._llm.invoke(messages)
            except TimeoutError as e:
                logger.error(f"LLM request timed out: {e}")
                raise LLMTimeoutError(
                    "LLM request timed out",
                    details={"prompt_length": len(prompt)}
                ) from e
            except ConnectionError as e:
                logger.error(f"Failed to connect to LLM: {e}")
                raise LLMConnectionError(
                    f"Failed to connect to LLM provider: {self.config.provider.value}",
                    details=str(e)
                ) from e

        # Execute with or without retry
        try:
            if self._retry_config:
                response = retry_llm_call(invoke_llm, config=self._retry_config)
            else:
                response = invoke_llm()
        except (LLMConnectionError, LLMTimeoutError):
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise LLMResponseError(
                "LLM invocation failed",
                details=str(e)
            ) from e

        if response is None or not hasattr(response, 'content'):
            logger.error("LLM returned empty or invalid response")
            raise LLMResponseError(
                "LLM returned empty or invalid response",
                details={"response": str(response)}
            )

        if parse_json:
            try:
                return self._parser.parse(response.content)
            except Exception as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response content: {response.content[:500]}")
                raise LLMParseError(
                    "Failed to parse LLM response as JSON",
                    details={"raw_content": response.content[:500], "error": str(e)}
                ) from e

        return response.content

    def classify_document(
        self,
        image_data: str,
        document_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Classify a document image into predefined types.

        Args:
            image_data: Base64-encoded image data URI
            document_types: Optional list of expected document types
                          (defaults to common financial documents)

        Returns:
            Dict with 'doc_type' key containing classification result

        Example:
            >>> result = llm.classify_document(image_base64)
            >>> print(result['doc_type'])  # "Passport"
        """
        if document_types is None:
            document_types = [
                "Driving License",
                "Passport",
                "SSN (Social Security Number card)",
                "Utility Bill",
                "Salary Slip",
                "ITR (Income Tax Return)",
                "Bank Account Statement",
                "Tax Return",
                "Articles of Incorporation",
                "Personal Financial Statement",
                "Others"
            ]

        types_list = "\n    * ".join(document_types)

        system_prompt = f"""You are a helpful document classification assistant. You will be given an image of a document. Your task is to carefully analyze the visual and textual content of the document to determine its type.

The possible types are:
    * {types_list}

Based on the layout, text, and any visible clues, classify the document into one of these types.
Return your answer strictly in the following JSON format (without any extra text):

```json
{{
  "doc_type": "<document type>"
}}
```

Replace `<document type>` with one of the above options exactly as written."""

        return self.analyze_image(
            image_data=image_data,
            prompt="Classify this document",
            system_prompt=system_prompt,
            parse_json=True
        )

    def extract_information(
        self,
        image_data: str,
        fields: Optional[List[str]] = None,
        date_format: str = "YYYY-MM-DD"
    ) -> Dict[str, Any]:
        """
        Extract structured information from a document image.

        Args:
            image_data: Base64-encoded image data URI
            fields: Optional list of fields to extract
                   (defaults to common PII fields)
            date_format: Expected date format in output

        Returns:
            Dict with extracted fields

        Example:
            >>> result = llm.extract_information(image_base64)
            >>> print(result['name'])  # "John Doe"
        """
        if fields is None:
            fields = [
                "Name", "Address", "Date of Birth (DOB)", "Gender",
                "Document Number", "Nationality", "Issuing authority",
                "Date of issue", "Expiry date", "SSN", "EIN",
                "Revenue", "Expenses", "Net Income"
            ]

        fields_list = "\n ".join([f"- {field}" for field in fields])

        system_prompt = f"""You are a highly skilled information extraction assistant.
You will be given an image of a document. Your task is to carefully analyze the visual and textual content of the document and extract all available personal or business information.

Specifically, extract details such as (if present):
 {fields_list}

All dates should be extracted in the format {date_format}.
For document numbers, the json key should be respective to the document type (e.g., "passport_number", "driving_license_number", "ein", "ssn").
Only include information explicitly present on the document. If a field is not found, omit it from the output.
Return the extracted information strictly in a JSON format, for example:

```json
{{
  "name": "John Doe",
  "address": "123 Main Street, Springfield, IL, USA",
  "dob": "1990-05-15",
  "gender": "Male",
  "document_number": "X1234567",
  "nationality": "USA"
}}
```

If any information is missing, do not include that key in the JSON.
Provide only the JSON object as the output, with no additional text."""

        return self.analyze_image(
            image_data=image_data,
            prompt="Extract personal or business information from this document",
            system_prompt=system_prompt,
            parse_json=True
        )

    def validate_authenticity(
        self,
        image_data: str
    ) -> Dict[str, Any]:
        """
        Validate document authenticity using vision analysis.

        Checks for:
        - Layout consistency
        - Field formatting
        - Forgery signs (photo manipulation, text overlays, font mismatches)

        Args:
            image_data: Base64-encoded image data URI

        Returns:
            Dict with validation results including:
            - valid (bool): Whether document appears authentic
            - reason (str): Explanation
            - layout_score (int): 0-100
            - field_score (int): 0-100
            - forgery_signs (List[str]): List of issues found

        Example:
            >>> result = llm.validate_authenticity(image_base64)
            >>> if not result['valid']:
            ...     print(f"Fraud detected: {result['reason']}")
        """
        system_prompt = """You are a document fraud detection expert. Analyze this document image for signs of forgery or manipulation.

Check for:
1. Layout consistency - Does the layout match standard government-issued or official documents?
2. Field consistency - Are all fields properly formatted and aligned?
3. Signs of forgery - Photo manipulation, text overlays, font mismatches, etc.
4. Overall authenticity - Does this appear to be a genuine document?

Return your analysis in JSON format:
```json
{
  "valid": true/false,
  "reason": "Brief explanation of why the document is valid or fake",
  "layout_score": 0-100,
  "field_score": 0-100,
  "forgery_signs": ["list", "of", "issues"] or []
}
```"""

        return self.analyze_image(
            image_data=image_data,
            prompt="Validate the authenticity of this document",
            system_prompt=system_prompt,
            parse_json=True
        )

    # ==================== Async Methods ====================

    async def analyze_image_async(
        self,
        image_data: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        parse_json: bool = True
    ) -> Union[Dict, str]:
        """
        Async version of analyze_image.

        Analyze an image with a custom prompt asynchronously.

        Args:
            image_data: Base64-encoded image data URI
            prompt: Analysis prompt
            system_prompt: Optional system context
            parse_json: Whether to parse response as JSON

        Returns:
            Parsed JSON dict if parse_json=True, else raw string

        Raises:
            LLMConnectionError: If connection to LLM fails
            LLMResponseError: If LLM returns invalid response
            LLMParseError: If JSON parsing fails
            LLMTimeoutError: If request times out

        Example:
            >>> result = await llm.analyze_image_async(image_data, "Describe this image")
        """
        messages = self._create_vision_message(image_data, prompt, system_prompt)

        async def invoke_llm_async():
            """Inner async function for retry wrapper"""
            try:
                return await self._llm.ainvoke(messages)
            except TimeoutError as e:
                logger.error(f"LLM request timed out: {e}")
                raise LLMTimeoutError(
                    "LLM request timed out",
                    details={"prompt_length": len(prompt)}
                ) from e
            except ConnectionError as e:
                logger.error(f"Failed to connect to LLM: {e}")
                raise LLMConnectionError(
                    f"Failed to connect to LLM provider: {self.config.provider.value}",
                    details=str(e)
                ) from e

        # Execute with or without retry
        try:
            if self._retry_config:
                response = await async_retry_llm_call(
                    invoke_llm_async, config=self._retry_config
                )
            else:
                response = await invoke_llm_async()
        except (LLMConnectionError, LLMTimeoutError):
            raise
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise LLMResponseError(
                "LLM invocation failed",
                details=str(e)
            ) from e

        if response is None or not hasattr(response, 'content'):
            logger.error("LLM returned empty or invalid response")
            raise LLMResponseError(
                "LLM returned empty or invalid response",
                details={"response": str(response)}
            )

        if parse_json:
            try:
                return self._parser.parse(response.content)
            except Exception as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response content: {response.content[:500]}")
                raise LLMParseError(
                    "Failed to parse LLM response as JSON",
                    details={"raw_content": response.content[:500], "error": str(e)}
                ) from e

        return response.content

    async def classify_document_async(
        self,
        image_data: str,
        document_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Async version of classify_document.

        Classify a document image into predefined types asynchronously.

        Args:
            image_data: Base64-encoded image data URI
            document_types: Optional list of expected document types

        Returns:
            Dict with 'doc_type' key containing classification result

        Example:
            >>> result = await llm.classify_document_async(image_base64)
            >>> print(result['doc_type'])  # "Passport"
        """
        if document_types is None:
            document_types = [
                "Driving License",
                "Passport",
                "SSN (Social Security Number card)",
                "Utility Bill",
                "Salary Slip",
                "ITR (Income Tax Return)",
                "Bank Account Statement",
                "Tax Return",
                "Articles of Incorporation",
                "Personal Financial Statement",
                "Others"
            ]

        types_list = "\n    * ".join(document_types)

        system_prompt = f"""You are a helpful document classification assistant. You will be given an image of a document. Your task is to carefully analyze the visual and textual content of the document to determine its type.

The possible types are:
    * {types_list}

Based on the layout, text, and any visible clues, classify the document into one of these types.
Return your answer strictly in the following JSON format (without any extra text):

```json
{{
  "doc_type": "<document type>"
}}
```

Replace `<document type>` with one of the above options exactly as written."""

        return await self.analyze_image_async(
            image_data=image_data,
            prompt="Classify this document",
            system_prompt=system_prompt,
            parse_json=True
        )

    async def extract_information_async(
        self,
        image_data: str,
        fields: Optional[List[str]] = None,
        date_format: str = "YYYY-MM-DD"
    ) -> Dict[str, Any]:
        """
        Async version of extract_information.

        Extract structured information from a document image asynchronously.

        Args:
            image_data: Base64-encoded image data URI
            fields: Optional list of fields to extract
            date_format: Expected date format in output

        Returns:
            Dict with extracted fields

        Example:
            >>> result = await llm.extract_information_async(image_base64)
            >>> print(result['name'])  # "John Doe"
        """
        if fields is None:
            fields = [
                "Name", "Address", "Date of Birth (DOB)", "Gender",
                "Document Number", "Nationality", "Issuing authority",
                "Date of issue", "Expiry date", "SSN", "EIN",
                "Revenue", "Expenses", "Net Income"
            ]

        fields_list = "\n ".join([f"- {field}" for field in fields])

        system_prompt = f"""You are a highly skilled information extraction assistant.
You will be given an image of a document. Your task is to carefully analyze the visual and textual content of the document and extract all available personal or business information.

Specifically, extract details such as (if present):
 {fields_list}

All dates should be extracted in the format {date_format}.
For document numbers, the json key should be respective to the document type (e.g., "passport_number", "driving_license_number", "ein", "ssn").
Only include information explicitly present on the document. If a field is not found, omit it from the output.
Return the extracted information strictly in a JSON format, for example:

```json
{{
  "name": "John Doe",
  "address": "123 Main Street, Springfield, IL, USA",
  "dob": "1990-05-15",
  "gender": "Male",
  "document_number": "X1234567",
  "nationality": "USA"
}}
```

If any information is missing, do not include that key in the JSON.
Provide only the JSON object as the output, with no additional text."""

        return await self.analyze_image_async(
            image_data=image_data,
            prompt="Extract personal or business information from this document",
            system_prompt=system_prompt,
            parse_json=True
        )

    async def validate_authenticity_async(
        self,
        image_data: str
    ) -> Dict[str, Any]:
        """
        Async version of validate_authenticity.

        Validate document authenticity using vision analysis asynchronously.

        Args:
            image_data: Base64-encoded image data URI

        Returns:
            Dict with validation results

        Example:
            >>> result = await llm.validate_authenticity_async(image_base64)
            >>> if not result['valid']:
            ...     print(f"Fraud detected: {result['reason']}")
        """
        system_prompt = """You are a document fraud detection expert. Analyze this document image for signs of forgery or manipulation.

Check for:
1. Layout consistency - Does the layout match standard government-issued or official documents?
2. Field consistency - Are all fields properly formatted and aligned?
3. Signs of forgery - Photo manipulation, text overlays, font mismatches, etc.
4. Overall authenticity - Does this appear to be a genuine document?

Return your analysis in JSON format:
```json
{
  "valid": true/false,
  "reason": "Brief explanation of why the document is valid or fake",
  "layout_score": 0-100,
  "field_score": 0-100,
  "forgery_signs": ["list", "of", "issues"] or []
}
```"""

        return await self.analyze_image_async(
            image_data=image_data,
            prompt="Validate the authenticity of this document",
            system_prompt=system_prompt,
            parse_json=True
        )

    @staticmethod
    def encode_image_to_base64(
        image_path: str,
        mime_type: Optional[str] = None
    ) -> str:
        """
        Encode a local image file to base64 data URI.

        Args:
            image_path: Path to image file
            mime_type: Optional MIME type (auto-detected if not provided)

        Returns:
            Base64-encoded data URI string

        Example:
            >>> data_uri = VisionLLM.encode_image_to_base64("document.png")
            >>> result = llm.classify_document(data_uri)
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")

        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = "image/png"

        return f"data:{mime_type};base64,{base64_encoded}"
