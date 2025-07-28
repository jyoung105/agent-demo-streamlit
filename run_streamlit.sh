#!/bin/bash
# Script to run Streamlit app with proper virtual environment activation

# Exit on any error
set -e

# Change to the project directory
cd "$(dirname "$0")"

echo "🚀 Starting Banner Agent Streamlit App..."

# Check if virtual environment exists
if [ ! -d "fastapi_app/venv" ]; then
    echo "❌ Virtual environment not found at fastapi_app/venv"
    echo "Please create a virtual environment first:"
    echo "  cd fastapi_app && python3 -m venv venv"
    echo "  source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source fastapi_app/venv/bin/activate

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python3 -c "
import sys
missing = []
required = ['streamlit', 'requests', 'pillow', 'fastapi', 'uvicorn', 'openai']
for package in required:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print(f'❌ Missing packages: {missing}')
    print('Installing missing packages...')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
else:
    print('✅ All required packages are installed')
"

# Navigate to streamlit app directory
cd streamlit_app

# Verify imports work
echo "🔗 Testing module imports..."
python3 -c "
try:
    from services.banner_api import BannerAPIClient, BannerWorkflowTracker
    print('✅ Successfully imported banner_api modules')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Run streamlit
echo "▶️  Starting Streamlit on http://localhost:8501"
streamlit run app.py --server.port 8501