"""
FastAPI backend for Banner Agent v2.
Provides REST API endpoints for banner creation workflow.
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routers.banner import router as banner_router
from database import init_database
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging based on environment variables
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format
)

logger = logging.getLogger(__name__)

# Create FastAPI application with environment-based configuration
app_name = os.getenv("APP_NAME", "Banner Agent API v2")
app_version = os.getenv("APP_VERSION", "2.0.0")
app_description = os.getenv("APP_DESCRIPTION", "FastAPI backend for AI-powered banner creation using OpenAI")

app = FastAPI(
    title=app_name,
    description=app_description,
    version=app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS with environment-based origins
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") != "*" else ["*"]

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
    if not os.getenv("OPENAI_API_KEY"):
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
        "environment": os.getenv("ENVIRONMENT", "development"),
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
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )