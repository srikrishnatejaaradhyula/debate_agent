"""
Configuration management for the Multi-Agent Debate System.

Handles environment variables, API credentials, and debate parameters.
Uses Pydantic for validation and type safety.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DebateConfig(BaseModel):
    """
    Configuration for the debate system.
    
    All settings can be overridden via environment variables or direct instantiation.
    """
    
    # OpenRouter API Configuration
    openrouter_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""),
        description="OpenRouter API key for LLM access"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    
    # Model Configuration
    model_name: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "tngtech/deepseek-r1t2-chimera:free"),
        description="LLM model to use for all agents"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model temperature for response generation"
    )
    
    # Debate Parameters
    max_rounds: int = Field(
        default_factory=lambda: int(os.getenv("MAX_ROUNDS", "3")),
        ge=1,
        le=10,
        description="Maximum number of rebuttal rounds"
    )
    max_response_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RESPONSE_LENGTH", "500")),
        ge=100,
        le=2000,
        description="Maximum words per agent response"
    )
    
    # Retry Configuration
    max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum retry attempts for failed LLM calls"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Base delay between retries (seconds)"
    )
    
    def validate_api_key(self) -> bool:
        """Check if API key is configured."""
        return bool(self.openrouter_api_key and self.openrouter_api_key != "your_openrouter_api_key_here")
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


def get_default_config() -> DebateConfig:
    """
    Factory function to create a DebateConfig with environment defaults.
    
    Returns:
        DebateConfig: Configured instance ready for use
        
    Raises:
        ValueError: If required API key is not set
    """
    config = DebateConfig()
    
    if not config.validate_api_key():
        raise ValueError(
            "OPENROUTER_API_KEY not configured. "
            "Set it in .env file or as environment variable."
        )
    
    return config
