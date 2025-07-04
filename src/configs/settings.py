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

    # Simulation Settings
    simulation_days: int = 100
    poi_count: int = 20
    agent_count: int = 100
    time_steps_per_day: int = 8

    # Data Paths
    config_path: str = "configs/default.yaml"
    personas_path: str = "data/personas.json"
    pois_path: str = "data/pois.json"

    # Configure loading from .env file
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")


def get_settings():
    """Get application settings from environment variables or .env file"""
    return AppSettings()
