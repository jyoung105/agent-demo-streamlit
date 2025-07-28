"""
Shared utilities for Banner Agent v2
"""

import base64
import os
from typing import Optional


def validate_base64_image(image_data: str) -> bool:
    """
    Validate if the provided string is valid base64 image data.
    
    Args:
        image_data: Base64 encoded image string
        
    Returns:
        bool: True if valid base64, False otherwise
    """
    try:
        base64.b64decode(image_data)
        return True
    except Exception:
        return False


def get_env_var(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with optional default.
    
    Args:
        var_name: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(var_name, default)


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."