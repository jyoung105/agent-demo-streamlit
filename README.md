# Banner Agent v2 - Streamlit Cloud Deployment

🎨 AI-powered banner creation with embedded FastAPI backend, optimized for Streamlit Cloud deployment.

## 🏗️ Architecture

This application uses **Method 1** (Embedded FastAPI) from the deployment guide:
- **Streamlit Frontend**: Main user interface (`streamlit_app/app.py`)
- **Embedded FastAPI Backend**: Runs as subprocess (`fastapi_app/main.py`)
- **Shared Models**: Common data structures (`shared/`)
- **Single Deployment**: Both services run in one Streamlit Cloud instance

## 📁 Project Structure

```
banner-agent-v2-cloud/
├── streamlit_app/
│   ├── app.py                 # Main Streamlit app with embedded FastAPI
│   ├── components/            # Original Streamlit components (legacy)
│   └── requirements.txt       # Streamlit-specific dependencies
├── fastapi_app/
│   ├── main.py               # FastAPI application
│   ├── routers/              # API route modules  
│   ├── services/             # Business logic services
│   ├── database.py           # Database connection and models
│   ├── banner_agent.db       # SQLite database
│   └── requirements.txt      # FastAPI-specific dependencies
├── shared/
│   ├── __init__.py
│   ├── banner.py             # Shared data models
│   └── utils.py              # Shared utilities
├── public/                   # Static assets (fonts, examples)
├── .streamlit/
│   ├── config.toml           # Streamlit configuration
│   └── secrets.toml          # Secrets template (not committed)
├── requirements.txt          # Root requirements for Streamlit Cloud
├── packages.txt              # System packages for Streamlit Cloud  
├── runtime.txt               # Python version specification
├── .env                      # Environment configuration
└── README.md                 # This file
```

## 🚀 Deployment to Streamlit Cloud

### Prerequisites

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/)
2. **GitHub Repository**: Push this code to a GitHub repository
3. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

### Deployment Steps

1. **Configure Secrets**:
   - Go to your Streamlit Cloud app settings
   - Add the following secrets:
     ```toml
     [openai]
     api_key = "sk-your-actual-openai-api-key"
     
     [app]
     environment = "production"
     debug = false
     ```

2. **Deploy Application**:
   - Connect your GitHub repository to Streamlit Cloud
   - Set main file path: `streamlit_app/app.py`
   - Deploy and wait for startup

3. **Verify Deployment**:
   - Check that the FastAPI backend starts successfully
   - Test the banner generation workflow
   - Monitor logs for any issues

## 🔧 Local Development

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd banner-agent-v2-cloud

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY
```

### Running Locally

```bash
# Method 1: Run Streamlit (will auto-start FastAPI)
streamlit run streamlit_app/app.py

# Method 2: Run services separately (for development)
# Terminal 1: FastAPI
cd fastapi_app
python -m uvicorn main:app --reload --port 8001

# Terminal 2: Streamlit  
streamlit run streamlit_app/app.py
```

## 🎯 Features

- **AI Banner Generation**: Powered by OpenAI DALL-E 3
- **Layout Extraction**: Analyze reference images for layout patterns
- **Prompt Optimization**: AI-enhanced prompt generation
- **Text Overlay**: Advanced text positioning and styling
- **4-Step Workflow**: Complete banner creation pipeline
- **Real-time Progress**: Live workflow status tracking
- **Download Support**: Export generated banners
- **Embedded Architecture**: Single deployment, dual services

## 🛠️ API Endpoints

The embedded FastAPI backend provides these endpoints:

- `GET /` - API information
- `GET /api/banner/health` - Health check
- `POST /api/banner/workflow` - Complete banner generation workflow
- `POST /api/banner/extract-layout` - Layout extraction only
- `POST /api/banner/optimize-prompt` - Prompt optimization only
- `POST /api/banner/generate` - Image generation only

## 🔒 Security Considerations

- **API Keys**: Store in Streamlit secrets, never in code
- **CORS**: Configured for embedded deployment
- **Input Validation**: Pydantic models validate all inputs
- **Error Handling**: Comprehensive error management
- **Process Isolation**: FastAPI runs in separate subprocess

## 📊 Monitoring

- **Health Checks**: Built-in API health monitoring
- **Logs**: Structured logging for debugging
- **Error Tracking**: Comprehensive error handling
- **Performance**: Processing time tracking for each step

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **FastAPI won't start**: Check Python version and dependencies
2. **OpenAI API errors**: Verify API key and credits
3. **Image generation fails**: Check model parameters and quotas
4. **Streamlit Cloud timeout**: Increase server timeout settings

### Support

- Check the logs in Streamlit Cloud dashboard
- Verify all environment variables are set
- Test locally before deploying
- Review the FastAPI docs at `/docs` endpoint

---

🎨 **Built with**: Streamlit, FastAPI, OpenAI DALL-E 3, SQLAlchemy, Pydantic