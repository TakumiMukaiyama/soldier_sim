from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables or .env file"""

    # Azure OpenAI settings
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_endpoint: str
    azure_openai_deployment_id: str

    # Gemini settings
    gemini_api_key: str
    gemini_model_name: str

    # LangSmith settings (optional for tracing)
    langsmith_tracing: Optional[str] = None
    langsmith_endpoint: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None

    # Data Paths
    config_path: str = "configs/default.yaml"
    personas_path: str = "data/personas.json"
    pois_path: str = "data/pois.json"

    # Configure loading from .env file
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")


app_settings = AppSettings()
