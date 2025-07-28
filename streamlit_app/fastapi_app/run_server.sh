#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting FastAPI server..."
echo "Script directory: $SCRIPT_DIR"
echo "Parent directory: $PARENT_DIR"

# Navigate to the fastapi_app directory
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include current directory AND parent directory
# Current directory is needed for relative imports (services, routers, database)
# Parent directory is needed for shared modules
export PYTHONPATH="$SCRIPT_DIR:$PARENT_DIR:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the FastAPI server from the current directory
echo "Starting uvicorn..."
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000