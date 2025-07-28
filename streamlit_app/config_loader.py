"""
Configuration loader for Streamlit app
Handles environment variables from both .env files and Streamlit secrets
"""
import os
import streamlit as st
from typing import Any, Optional


def get_config(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """
    Get configuration value from Streamlit secrets or environment variables.
    
    Priority:
    1. Streamlit secrets (when available)
    2. Environment variables
    3. Default value
    
    Args:
        key: Configuration key to retrieve
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    # Check if running on Streamlit Cloud
    if hasattr(st, 'secrets'):
        try:
            # Try to get from st.secrets
            return st.secrets.get(key, default)
        except Exception:
            # Fallback to environment variables if st.secrets fails
            pass
    
    # Get from environment variables
    return os.getenv(key, default)


def is_streamlit_cloud() -> bool:
    """
    Check if the app is running on Streamlit Cloud.
    
    Returns:
        True if running on Streamlit Cloud, False otherwise
    """
    return (
        os.getenv("STREAMLIT_SHARING_MODE") is not None or 
        os.getenv("STREAMLIT_CLOUD") is not None or
        hasattr(st, 'secrets')
    )


def get_all_config() -> dict:
    """
    Get all configuration values as a dictionary.
    Useful for passing configuration to FastAPI backend.
    
    Returns:
        Dictionary of all configuration values
    """
    config = {}
    
    # List of all configuration keys
    config_keys = [
        "OPENAI_API_KEY",
        "ASSISTANT_ID",
        "ENVIRONMENT",
        "APP_NAME",
        "APP_VERSION",
        "APP_DESCRIPTION",
        "API_HOST",
        "API_PORT",
        "API_RELOAD",
        "CORS_ORIGINS",
        "API_RATE_LIMIT",
        "API_RATE_WINDOW",
        "STREAMLIT_SERVER_PORT",
        "STREAMLIT_SERVER_ADDRESS",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS",
        "BACKEND_API_URL",
        "DEFAULT_BANNER_SIZE",
        "MAX_UPLOAD_SIZE",
        "SUPPORTED_IMAGE_TYPES",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "LOG_FILE",
        "OPENAI_VISION_MODEL",
        "OPENAI_VISION_MAX_TOKENS",
        "OPENAI_VISION_TEMPERATURE",
        "OPENAI_TEXT_MODEL",
        "OPENAI_TEXT_MAX_TOKENS",
        "OPENAI_TEXT_TEMPERATURE",
        "MAX_IMAGE_WIDTH",
        "MAX_IMAGE_HEIGHT",
        "IMAGE_QUALITY",
        "ENABLE_IMAGE_OPTIMIZATION",
        "SUPPORTED_OUTPUT_FORMATS",
        "ENABLE_RATE_LIMITING",
        "MAX_REQUESTS_PER_MINUTE",
        "ENABLE_INPUT_VALIDATION",
        "MAX_PROMPT_LENGTH",
        "DATABASE_URL",
        "DATABASE_POOL_SIZE",
        "DATABASE_POOL_OVERFLOW",
        "DATABASE_POOL_TIMEOUT",
        "ENABLE_CACHING",
        "CACHE_TTL",
        "CACHE_MAX_SIZE",
        "ENABLE_MONITORING",
        "ENABLE_ERROR_TRACKING",
        "ENABLE_PERFORMANCE_TRACKING",
        "STORAGE_BACKEND",
        "STORAGE_LOCAL_PATH",
        "MAX_FILE_SIZE_MB",
        "ALLOWED_FILE_EXTENSIONS",
        "SESSION_TIMEOUT_MINUTES",
        "SESSION_COOKIE_SECURE",
        "SESSION_COOKIE_HTTPONLY",
    ]
    
    # Get all configuration values
    for key in config_keys:
        value = get_config(key)
        if value is not None:
            config[key] = value
    
    return config