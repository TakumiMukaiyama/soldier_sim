from typing import Any, Dict, Optional
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage


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
        response = self.client.invoke(messages)

        # Parse JSON response
        try:
            return response.json()
        except Exception as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}\nResponse: {response.content}"
            )

    def __call__(self, *args, **kwargs):
        """Make the client callable like a LangChain LLM"""
        return self.client(*args, **kwargs)
