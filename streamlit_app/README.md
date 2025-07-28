# Banner Agent v2 - Streamlit Cloud Deployment

## Overview
This is a complete Streamlit application with embedded FastAPI backend for AI-powered banner generation using GPT-image-1.

## Features
- 🎨 AI-powered banner generation with GPT-image-1
- 📐 Multiple size options (1792x1024, 1024x1792, 1024x1024)
- 🖼️ Reference image support for layout extraction
- ⚡ Real-time processing with status updates
- 📱 Responsive web interface
- ☁️ Streamlit Cloud ready with embedded FastAPI

## Local Development

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup
1. **Create and activate virtual environment:**
   ```bash
   cd fastapi_app
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cd ..
   ```

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Run the application:**
   ```bash
   # Option 1: Use the provided script (recommended)
   ./run_streamlit.sh
   
   # Option 2: Manual activation
   source fastapi_app/venv/bin/activate
   cd streamlit_app
   streamlit run app.py
   ```

### Troubleshooting Import Errors

If you encounter `ModuleNotFoundError: No module named 'services.banner_api'`:

1. **Ensure virtual environment is activated:**
   ```bash
   source fastapi_app/venv/bin/activate
   ```

2. **Verify you're in the correct directory:**
   ```bash
   cd streamlit_app  # Must run from this directory
   ```

3. **Check dependencies are installed:**
   ```bash
   pip list | grep -E "(streamlit|requests|pillow)"
   ```

4. **Test imports manually:**
   ```bash
   python3 -c "from services.banner_api import BannerAPIClient"
   ```

## Streamlit Cloud Deployment

### Setup on Streamlit Cloud
1. Fork/upload this repository to GitHub
2. Connect your GitHub account to Streamlit Cloud
3. Create a new app pointing to `streamlit_app/app.py`
4. Add secrets in Streamlit Cloud dashboard:
   - `OPENAI_API_KEY`: Your OpenAI API key

### Environment Variables (Optional)
- `ENVIRONMENT`: Set to "production" for cloud deployment
- `LOG_LEVEL`: Set logging level (INFO, DEBUG, WARNING, ERROR)
- `STREAMLIT_CLOUD`: Auto-detected for cloud deployment

### Files Structure
```
streamlit_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # All dependencies
├── packages.txt          # System packages for cloud
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── secrets.toml.example  # Example secrets file
├── fastapi_app/          # Complete FastAPI backend
│   ├── main.py
│   ├── routers/
│   ├── services/
│   └── database.py
└── shared/               # Shared modules
    ├── banner.py
    └── utils.py
```

## Architecture

### Local Development Mode
- **Frontend**: Streamlit app on port 8501/8502
- **Backend**: FastAPI subprocess on port 8001
- **Mode**: Subprocess mode with virtual environment

### Streamlit Cloud Mode  
- **Frontend**: Streamlit app (managed by Streamlit Cloud)
- **Backend**: FastAPI embedded in same process (threading)
- **Mode**: Embedded mode for cloud compatibility

## API Endpoints
- `GET /`: Root endpoint with server info
- `GET /api/banner/health`: Health check
- `POST /api/banner/workflow`: Complete banner generation workflow
- `GET /docs`: FastAPI documentation

## Usage
1. Enter banner requirements in the text area
2. Optionally upload a reference image
3. Select banner size
4. Click "Generate Banner" 
5. Download the generated banner

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all files are in the correct directory structure
2. **API Key**: Make sure OPENAI_API_KEY is set in secrets
3. **Timeout**: Increase timeout values for slower connections
4. **Memory**: Large images may require more memory

### Logs
Check Streamlit Cloud logs for detailed error information and FastAPI startup status.