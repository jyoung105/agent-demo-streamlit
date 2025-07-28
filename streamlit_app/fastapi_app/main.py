"""
FastAPI backend for Banner Agent v2.
Provides REST API endpoints for banner creation workflow.
"""
import os
import sys
from pathlib import Path

# No path manipulation needed - using absolute imports

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_app.routers.banner import router as banner_router
from fastapi_app.database import init_database
from datetime import datetime
import logging
from dotenv import load_dotenv

# Try to import config loader from parent directory (when running embedded in Streamlit)
try:
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from config_loader import get_config, is_streamlit_cloud
    USE_CONFIG_LOADER = True
except ImportError:
    USE_CONFIG_LOADER = False
    # Define fallback functions
    def get_config(key, default=None):
        return os.getenv(key, default)
    
    def is_streamlit_cloud():
        return os.getenv("STREAMLIT_SHARING_MODE") is not None or os.getenv("STREAMLIT_CLOUD") is not None

# Load environment variables from .env file (for local development)
if not is_streamlit_cloud():
    load_dotenv()

# Configure logging based on environment variables
log_level = get_config("LOG_LEVEL", "INFO").upper()
log_format = get_config("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format
)

logger = logging.getLogger(__name__)

# Create FastAPI application with environment-based configuration
app_name = get_config("APP_NAME", "Banner Agent API v2")
app_version = get_config("APP_VERSION", "2.0.0")
app_description = get_config("APP_DESCRIPTION", "FastAPI backend for AI-powered banner creation using OpenAI")

app = FastAPI(
    title=app_name,
    description=app_description,
    version=app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS with environment-based origins
cors_origins_str = get_config("CORS_ORIGINS", "*")
cors_origins = cors_origins_str.split(",") if cors_origins_str != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(banner_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Global exception handler for HTTP exceptions
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Global exception handler for general exceptions
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.on_event("startup")
async def startup_event():
    """
    Application startup event
    """
    logger.info("Starting Banner Agent API v2...")
    
    # Check for required environment variables
    if not get_config("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    
    # Initialize database
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise RuntimeError(f"Database initialization failed: {e}")
    
    logger.info("Banner Agent API v2 started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event
    """
    logger.info("Shutting down Banner Agent API v2...")


@app.get("/")
async def root():
    """
    Root endpoint with environment-based information
    """
    return {
        "message": app_name,
        "version": app_version,
        "description": app_description,
        "environment": get_config("ENVIRONMENT", "development"),
        "docs": "/docs",
        "health": "/api/banner/health",
        "endpoints": {
            "extract_layout": "POST /api/banner/extract-layout",
            "optimize_prompt": "POST /api/banner/optimize-prompt", 
            "generate_banner": "POST /api/banner/generate",
            "complete_workflow": "POST /api/banner/workflow",
            "start_workflow": "POST /api/banner/start-workflow",
            "job_status": "GET /api/banner/job-status/{job_id}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = get_config("API_HOST", "0.0.0.0")
    port = int(get_config("API_PORT", "8000"))
    reload = get_config("API_RELOAD", "true").lower() == "true"
    log_level = get_config("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )