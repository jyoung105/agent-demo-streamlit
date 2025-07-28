"""
Enhanced banner API routes with database integration and workflow orchestration.
Provides endpoints for complete banner generation pipeline with persistent data storage.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# No path manipulation needed - using absolute imports

from fastapi import APIRouter, HTTPException, status, Depends
from openai import OpenAI
from sqlalchemy.orm import Session
from fastapi_app.config import get_config, OPENAI_API_KEY

from shared.banner import (
    BannerLayoutRequest, BannerLayoutResponse,
    BannerPromptRequest, BannerPromptResponse,
    BannerGenerationRequest, BannerGenerationResponse,
    BannerWorkflowRequest, BannerWorkflowResponse, BannerWorkflowStep,
    HealthResponse, ErrorResponse
)
from fastapi_app.services.banner_layout import BannerLayoutService
from fastapi_app.services.banner_prompt import BannerPromptService
from fastapi_app.services.banner_generation import BannerGenerationService
from fastapi_app.services.banner_tools import BannerToolsService
from fastapi_app.services.workflow_service import BannerWorkflowService
from fastapi_app.database import get_database_session, init_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/banner", tags=["banner"])


def get_openai_client() -> OpenAI:
    """Dependency to get OpenAI client."""
    api_key = OPENAI_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured"
        )
    return OpenAI(api_key=api_key)


def get_database() -> Session:
    """Dependency to get database session."""
    return next(get_database_session())


def get_banner_layout_service(
    client: OpenAI = Depends(get_openai_client),
    db: Session = Depends(get_database)
) -> BannerLayoutService:
    """Dependency to get banner layout service."""
    return BannerLayoutService(client, db)


def get_banner_prompt_service(
    client: OpenAI = Depends(get_openai_client),
    db: Session = Depends(get_database)
) -> BannerPromptService:
    """Dependency to get banner prompt service."""
    return BannerPromptService(client, db)


def get_banner_generation_service(
    client: OpenAI = Depends(get_openai_client),
    db: Session = Depends(get_database)
) -> BannerGenerationService:
    """Dependency to get banner generation service."""
    return BannerGenerationService(client, db)


def get_banner_tools_service(
    client: OpenAI = Depends(get_openai_client),
    db: Session = Depends(get_database)
) -> BannerToolsService:
    """Dependency to get banner tools service."""
    return BannerToolsService(client, db)


def get_workflow_service(
    client: OpenAI = Depends(get_openai_client),
    db: Session = Depends(get_database)
) -> BannerWorkflowService:
    """Dependency to get workflow orchestration service."""
    return BannerWorkflowService(client, db)


@router.post("/extract-layout", response_model=BannerLayoutResponse)
async def extract_layout(
    request: BannerLayoutRequest,
    layout_service: BannerLayoutService = Depends(get_banner_layout_service)
):
    """
    Extract layout information from a reference banner image.
    
    Step 1 of the banner creation workflow.
    """
    try:
        logger.info("Starting banner layout extraction")
        result = layout_service.extract_layout(request.image_data)
        logger.info(f"Layout extraction completed in {result.processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Layout extraction endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Layout extraction failed: {str(e)}"
        )


@router.post("/optimize-prompt", response_model=BannerPromptResponse)
async def optimize_prompt(
    request: BannerPromptRequest,
    prompt_service: BannerPromptService = Depends(get_banner_prompt_service)
):
    """
    Optimize banner prompt based on layout data and user requirements.
    
    Step 2 of the banner creation workflow.
    """
    try:
        logger.info("Starting banner prompt optimization")
        result = prompt_service.optimize_prompt(request.layout_data, request.user_requirements)
        logger.info(f"Prompt optimization completed in {result.processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Prompt optimization endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt optimization failed: {str(e)}"
        )


@router.post("/generate", response_model=BannerGenerationResponse)
async def generate_banner(
    request: BannerGenerationRequest,
    generation_service: BannerGenerationService = Depends(get_banner_generation_service)
):
    """
    Generate banner image using GPT-image-1.
    
    Step 3 of the banner creation workflow.
    """
    try:
        logger.info(f"Starting banner generation with size {request.size}")
        result = generation_service.generate_banner(
            optimized_data=request.optimized_data,
            user_requirements="",  # Not needed when optimized_data is provided
            size=request.size,
            include_image_data=request.include_image_data
        )
        logger.info(f"Banner generation completed in {result.processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Banner generation endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Banner generation failed: {str(e)}"
        )


@router.post("/workflow", response_model=BannerWorkflowResponse)
async def banner_workflow(
    request: BannerWorkflowRequest,
    workflow_service: BannerWorkflowService = Depends(get_workflow_service)
):
    """
    Enhanced complete banner creation workflow with database persistence.
    
    Handles the entire 3-step process with proper data flow and error handling:
    1. Layout extraction (if reference image provided)
    2. Prompt generation and optimization
    3. Image generation with GPT-image-1
    
    Each step stores outputs that are used as inputs for subsequent steps.
    """
    try:
        logger.info("Starting enhanced banner workflow")
        result = workflow_service.execute_workflow(request)
        
        if result.success:
            logger.info(f"Workflow completed successfully in {result.total_processing_time:.2f}s")
        else:
            logger.error(f"Workflow failed after {result.total_processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Banner workflow endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Banner workflow failed: {str(e)}"
        )


@router.post("/tools-workflow", response_model=BannerGenerationResponse)
async def banner_tools_workflow(
    request: BannerWorkflowRequest,
    tools_service: BannerToolsService = Depends(get_banner_tools_service)
):
    """
    New tools-based banner creation workflow using OpenAI tools chain:
    1. extract_layout (gpt-4o) - Extract layout from reference image
    2. write_banner_prompt (gpt-4o) - Generate optimized prompt
    3. generate_banner (gpt-image-1) - Create banner image
    4. add_text_layout (gpt-4o + code) - Add text overlays
    
    This is the recommended approach for banner generation.
    """
    try:
        logger.info("Starting tools-based banner workflow")
        result = tools_service.execute_workflow(
            user_requirements=request.user_requirements,
            image_data=request.image_data,
            size=request.size
        )
        
        if result.success:
            logger.info(f"Tools workflow completed successfully in {result.processing_time:.2f}s")
        else:
            logger.error(f"Tools workflow failed after {result.processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Tools workflow endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tools workflow failed: {str(e)}"
        )


@router.get("/tools")
async def list_tools(
    tools_service: BannerToolsService = Depends(get_banner_tools_service)
) -> Dict[str, Any]:
    """
    List all available banner generation tools.
    """
    try:
        tools = tools_service.list_tools()
        return {
            "tools": tools,
            "workflow_chain": [
                "extract_layout",
                "write_banner_prompt", 
                "generate_banner",
                "add_text_layout"
            ]
        }
        
    except Exception as e:
        logger.error(f"List tools endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tools: {str(e)}"
        )


@router.post("/tools/{tool_name}")
async def execute_tool(
    tool_name: str,
    input_data: Dict[str, Any],
    tools_service: BannerToolsService = Depends(get_banner_tools_service)
) -> Dict[str, Any]:
    """
    Execute a specific banner generation tool.
    
    Available tools:
    - extract_layout: Extract layout from reference image
    - write_banner_prompt: Generate optimized banner prompt
    - generate_banner: Create banner image with GPT-image-1
    - add_text_layout: Add text overlays to banner
    """
    try:
        logger.info(f"Executing tool: {tool_name}")
        result = tools_service.execute_tool(tool_name, input_data)
        
        if result["success"]:
            logger.info(f"Tool {tool_name} completed successfully in {result.get('processing_time', 0):.2f}s")
        else:
            logger.error(f"Tool {tool_name} failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Execute tool endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}"
        )


@router.post("/start-workflow")
async def start_workflow_async(
    request: BannerWorkflowRequest,
    workflow_service: BannerWorkflowService = Depends(get_workflow_service)
) -> Dict[str, Any]:
    """
    Start banner workflow asynchronously and return job_id immediately.
    
    The workflow runs in the background while the client can poll
    /job-status/{job_id} for real-time step progress and outputs.
    
    Returns:
        dict: Contains job_id for polling progress
    """
    import asyncio
    from database import create_banner_job, get_database_session
    
    try:
        logger.info("Starting async banner workflow")
        
        # Create job in database immediately
        db = next(get_database_session())
        try:
            job = create_banner_job(
                db,
                user_requirements=request.user_requirements,
                image_size=request.size,
                original_image_data=request.image_data
            )
            
            # Create a new workflow service instance for background execution
            from openai import OpenAI
            import os
            
            def execute_in_background():
                """Execute workflow in background thread"""
                try:
                    # Create fresh instances for background execution
                    bg_client = OpenAI(api_key=OPENAI_API_KEY)
                    bg_db = next(get_database_session())
                    bg_workflow_service = BannerWorkflowService(bg_client, bg_db)
                    
                    result = bg_workflow_service.execute_workflow(request)
                    logger.info(f"Background workflow completed for job {job.id}")
                    bg_db.close()
                except Exception as e:
                    logger.error(f"Background workflow failed for job {job.id}: {e}")
            
            # Start background execution using thread pool
            import threading
            background_thread = threading.Thread(target=execute_in_background)
            background_thread.daemon = True
            background_thread.start()
            
            return {
                "success": True,
                "job_id": job.id,
                "message": "Workflow started successfully. Use /job-status/{job_id} to track progress.",
                "polling_endpoint": f"/api/banner/job-status/{job.id}"
            }
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Failed to start async workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workflow: {str(e)}"
        )


@router.get("/job-status/{job_id}")
async def get_job_status(
    job_id: str,
    workflow_service: BannerWorkflowService = Depends(get_workflow_service)
) -> Dict[str, Any]:
    """
    Get detailed status of a banner generation job with rich step outputs.
    
    Returns job information including:
    - Step-by-step progress and status
    - Processing times for each step  
    - Detailed outputs from each completed step
    - Layout data, prompt details, image metadata
    - Real-time progress for frontend polling
    """
    try:
        job_status = workflow_service.get_enhanced_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job status endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.post("/retry-job/{job_id}", response_model=BannerWorkflowResponse)
async def retry_failed_job(
    job_id: str,
    workflow_service: BannerWorkflowService = Depends(get_workflow_service)
):
    """
    Retry a failed banner generation job.
    
    Re-runs the workflow from the beginning using the same
    parameters as the original job.
    """
    try:
        logger.info(f"Retrying failed job {job_id}")
        result = workflow_service.retry_failed_job(job_id)
        
        if result.success:
            logger.info(f"Job retry completed successfully in {result.total_processing_time:.2f}s")
        else:
            logger.error(f"Job retry failed after {result.total_processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Job retry endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job retry failed: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(client: OpenAI = Depends(get_openai_client)):
    """
    Health check endpoint for banner service.
    """
    try:
        # Test OpenAI connection
        models = client.models.list()
        openai_connection = True
        
        return HealthResponse(
            status="healthy",
            openai_connection=openai_connection
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            openai_connection=False
        )