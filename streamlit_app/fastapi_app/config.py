"""
Configuration helper for FastAPI app
Handles environment variables with fallback to parent config loader
"""
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Try to import parent config loader when running embedded in Streamlit
try:
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from config_loader import get_config as parent_get_config, is_streamlit_cloud
    USE_PARENT_CONFIG = True
except ImportError:
    USE_PARENT_CONFIG = False
    
    def parent_get_config(key, default=None):
        return os.getenv(key, default)
    
    def is_streamlit_cloud():
        return os.getenv("STREAMLIT_SHARING_MODE") is not None or os.getenv("STREAMLIT_CLOUD") is not None


def get_config(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """
    Get configuration value with proper fallback chain.
    
    Args:
        key: Configuration key to retrieve
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    # Use parent config loader if available (handles st.secrets)
    return parent_get_config(key, default)


# Export commonly used configuration values as constants
OPENAI_API_KEY = get_config("OPENAI_API_KEY")
ASSISTANT_ID = get_config("ASSISTANT_ID")
OPENAI_VISION_MODEL = get_config("OPENAI_VISION_MODEL", "gpt-4o")
OPENAI_VISION_MAX_TOKENS = int(get_config("OPENAI_VISION_MAX_TOKENS", "2000"))
OPENAI_VISION_TEMPERATURE = float(get_config("OPENAI_VISION_TEMPERATURE", "0.1"))
OPENAI_TEXT_MODEL = get_config("OPENAI_TEXT_MODEL", "gpt-4o-mini")
OPENAI_TEXT_MAX_TOKENS = int(get_config("OPENAI_TEXT_MAX_TOKENS", "1000"))
OPENAI_TEXT_TEMPERATURE = float(get_config("OPENAI_TEXT_TEMPERATURE", "0.3"))