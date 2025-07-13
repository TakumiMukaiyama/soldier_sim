import json
import os
import re
from typing import Any, Dict, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.get_logger import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Client for Gemini API with structured output support"""

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"
    ):
        """
        Initialize Gemini client

        Args:
            api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
            model_name: Gemini model name to use
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided and GEMINI_API_KEY env var not set"
            )

        self.model_name = model_name
        self.client = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.api_key,
        )

    def _clean_json_content(self, content: str) -> str:
        """Clean content to prepare for JSON parsing

        Args:
            content: Raw response content from the model

        Returns:
            Cleaned content ready for JSON parsing
        """
        # Remove code block markers and whitespace
        cleaned = content.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Remove control characters
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)

        return cleaned

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from cleaned content

        Args:
            content: Cleaned response content

        Returns:
            Parsed JSON as dict

        Raises:
            ValueError: If JSON parsing fails
        """
        try:
            # First try: extract JSON object with full regex
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL
            )
            if json_match:
                return json.loads(json_match.group(0))

            # Second try: parse entire content as JSON
            return json.loads(content)

        except json.JSONDecodeError:
            # Third try: use simpler regex for incomplete JSON
            simple_match = re.search(r'\{.*?"reason".*?\}', content, re.DOTALL)
            if simple_match:
                try:
                    return json.loads(simple_match.group(0))
                except Exception as e:
                    logger.error(f"Simple JSON extraction failed: {e}")

            raise ValueError(f"Failed to parse JSON from content: {content[:500]}...")

    def invoke_with_structured_output(
        self,
        system_prompt: str,
        user_prompt: str,
        json_structure: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke Gemini with a prompt designed to produce structured JSON output

        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt containing input variables
            json_structure: Example JSON structure to guide output format

        Returns:
            JSON response from Gemini
        """
        # Create prompt with structured output instructions
        json_str = str(json_structure)

        messages = [
            SystemMessage(
                content=f"{system_prompt}\n\nYou must respond with valid JSON matching this structure: {json_str}"
            ),
            HumanMessage(content=user_prompt),
        ]

        # Get response from model
        try:
            response = self.client.invoke(messages)
            content = self._clean_json_content(response.content)
            return self._extract_json(content)
        except ValueError as e:
            # Add original response to error message for better debugging
            raise ValueError(f"{str(e)}\nOriginal response: {response.content}")
        except Exception as e:
            raise ValueError(f"Unexpected error in LLM request: {e}")

    def __call__(self, *args, **kwargs):
        """Make the client callable like a LangChain LLM"""
        return self.client(*args, **kwargs)
