import json
import os
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI


class AzureGPTClient:
    """Client for Azure OpenAI API with structured output support"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        """
        Initialize Azure OpenAI client

        Args:
            api_key: Azure API key (if None, uses AZURE_API_KEY env var)
            api_version: Azure API version (if None, uses AZURE_API_VERSION env var)
            azure_endpoint: Azure endpoint URL (if None, uses AZURE_ENDPOINT env var)
            azure_deployment: Azure deployment name (if None, uses AZURE_DEPLOYMENT env var)
        """
        self.api_key = api_key or os.environ.get("AZURE_API_KEY")
        self.api_version = api_version or os.environ.get("AZURE_API_VERSION", "2023-05-15")
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_ENDPOINT")
        self.azure_deployment = azure_deployment or os.environ.get("AZURE_DEPLOYMENT")

        if not self.api_key:
            raise ValueError("Azure API key not provided and AZURE_API_KEY env var not set")

        if not self.azure_endpoint:
            raise ValueError("Azure endpoint not provided and AZURE_ENDPOINT env var not set")

        if not self.azure_deployment:
            raise ValueError("Azure deployment not provided and AZURE_DEPLOYMENT env var not set")

        self.client = AzureChatOpenAI(
            azure_deployment=self.azure_deployment,
            openai_api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
        )

    def invoke_with_structured_output(
        self,
        system_prompt: str,
        user_prompt: str,
        json_structure: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke Azure OpenAI with a prompt designed to produce structured JSON output

        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt containing input variables
            json_structure: Example JSON structure to guide output format

        Returns:
            JSON response from Azure OpenAI
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
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON response: {e}\nResponse content: {response.content}")

            raise ValueError(f"Failed to parse JSON response: {e}\nResponse content: {response.content}")
        except Exception as e:
            raise ValueError(f"Unexpected error parsing response: {e}\nResponse content: {response.content}")

    def __call__(self, *args, **kwargs):
        """Make the client callable like a LangChain LLM"""
        return self.client(*args, **kwargs)
