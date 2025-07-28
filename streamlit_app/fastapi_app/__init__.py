"""
FastAPI app package initialization with proper path setup for embedded mode.
"""
import sys
from pathlib import Path

# Add the fastapi_app directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Add the parent streamlit_app directory for shared imports
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))