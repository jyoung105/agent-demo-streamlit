"""
Routers package initialization with path setup for shared modules.
"""
import sys
from pathlib import Path

# Add parent directories to Python path for shared module access
current_dir = Path(__file__).parent
fastapi_dir = current_dir.parent
parent_dir = fastapi_dir.parent

# Add paths if not already present
for path in [str(fastapi_dir), str(parent_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)