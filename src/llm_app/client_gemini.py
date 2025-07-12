import json
import os
import re
from typing import Any, Dict, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiClient:
    """Client for Gemini API with structured output support"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        """
        Initialize Gemini client

        Args:
            api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
            model_name: Gemini model name to use
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided and GEMINI_API_KEY env var not set")

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
            content = response.content.strip()

            # Clean up common formatting issues
            content = content.replace("```json", "").replace("```", "").strip()

            # Try to extract JSON from the response if it contains additional text
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up any potential formatting issues
                json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", json_str)  # Remove control characters
                return json.loads(json_str)
            else:
                # Try to parse the entire content as JSON after cleaning
                clean_content = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", content)
                return json.loads(clean_content)
        except json.JSONDecodeError as e:
            # Try one more time with a simpler regex for incomplete JSON
            simple_match = re.search(r'\{.*?"reason".*?\}', content, re.DOTALL)
            if simple_match:
                try:
                    return json.loads(simple_match.group(0))
                except:
                    pass

            raise ValueError(f"Failed to parse JSON response: {e}\nResponse content: {response.content}")
        except Exception as e:
            raise ValueError(f"Unexpected error parsing response: {e}\nResponse content: {response.content}")

    def __call__(self, *args, **kwargs):
        """Make the client callable like a LangChain LLM"""
        return self.client(*args, **kwargs)
