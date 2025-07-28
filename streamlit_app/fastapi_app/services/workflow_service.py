"""
Enhanced workflow orchestration service for banner generation.
Coordinates all four steps with proper data flow and error handling.
"""
import time
import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from openai import OpenAI

from shared.banner import (
    BannerWorkflowRequest, BannerWorkflowResponse, BannerWorkflowStep,
    BannerGenerationResponse, BannerLayoutData
)
from fastapi_app.database import (
    create_banner_job, get_job_by_id, update_job_status,
    JobStatus, StepStatus, BannerJob, LayoutExtraction, PromptGeneration, TextOverlay
)
from fastapi_app.services.banner_layout import BannerLayoutService
from fastapi_app.services.banner_prompt import BannerPromptService
from fastapi_app.services.banner_generation import BannerGenerationService
from fastapi_app.services.banner_text_overlay import BannerTextOverlayService

logger = logging.getLogger(__name__)


class BannerWorkflowService:
    """
    Orchestrates the complete 4-step banner generation workflow with step-by-step processing.
    """
    
    def __init__(self, openai_client: OpenAI, db_session: Session):
        self.client = openai_client
        self.db = db_session
        
        # Initialize individual services
        self.layout_service = BannerLayoutService(openai_client, db_session)
        self.prompt_service = BannerPromptService(openai_client, db_session)
        self.generation_service = BannerGenerationService(openai_client)
        self.text_overlay_service = BannerTextOverlayService(db_session)
    
    def execute_workflow(self, request: BannerWorkflowRequest) -> BannerWorkflowResponse:
        """
        Execute the complete banner generation workflow.
        
        Args:
            request: Workflow request containing user requirements and optional reference image
            
        Returns:
            BannerWorkflowResponse: Complete workflow response with step tracking
        """
        start_time = time.time()
        
        # Create job in database
        job = create_banner_job(
            self.db,
            user_requirements=request.user_requirements,
            image_size=request.size,
            original_image_data=request.image_data
        )
        
        logger.info(f"Started banner workflow for job {job.id}")
        
        # Update job status to in progress
        update_job_status(self.db, job.id, JobStatus.IN_PROGRESS)
        
        # Initialize workflow steps for 4-step workflow
        workflow_steps = [
            BannerWorkflowStep(
                step=1,
                name="layout_extraction",
                status="pending",
                result=None,
                error=None,
                processing_time=None
            ),
            BannerWorkflowStep(
                step=2,
                name="prompt_generation",
                status="pending",
                result=None,
                error=None,
                processing_time=None
            ),
            BannerWorkflowStep(
                step=3,
                name="image_generation",
                status="pending",
                result=None,
                error=None,
                processing_time=None
            ),
            BannerWorkflowStep(
                step=4,
                name="text_overlay",
                status="pending",
                result=None,
                error=None,
                processing_time=None
            )
        ]
        
        layout_data = None
        final_result = None
        
        try:
            # Step 1: Layout Extraction (if reference image provided)
            if request.image_data:
                workflow_steps[0].status = "in_progress"
                step_start = time.time()
                
                layout_response = self.layout_service.extract_layout_for_job(
                    job.id, request.image_data
                )
                
                workflow_steps[0].processing_time = time.time() - step_start
                
                if layout_response.success:
                    workflow_steps[0].status = "completed"
                    
                    # Extract detailed results from layout data
                    text_count = len(layout_response.layout_data.text) if layout_response.layout_data and layout_response.layout_data.text else 0
                    background_style = layout_response.layout_data.background.style if layout_response.layout_data and layout_response.layout_data.background else "unknown"
                    scene_description = layout_response.layout_data.background.scene if layout_response.layout_data and layout_response.layout_data.background else "No scene"
                    color_palette = layout_response.layout_data.background.color_palette if layout_response.layout_data and layout_response.layout_data.background else []
                    
                    workflow_steps[0].result = {
                        "success": True,
                        "text_elements_count": text_count,
                        "background_style": background_style,
                        "scene_description": scene_description,
                        "color_palette": color_palette,
                        "processing_model": "gpt-4.1",
                        "processing_time": layout_response.processing_time
                    }
                    layout_data = layout_response.layout_data
                    
                    # Log detailed step results
                    logger.info(f"✅ Step 1 (Layout Extraction) completed for job {job.id}")
                    logger.info(f"   📊 Results: {text_count} text elements, style: {background_style}")
                    logger.info(f"   🎨 Scene: {scene_description}")
                    logger.info(f"   ⏱️  Processing time: {layout_response.processing_time:.2f}s")
                else:
                    workflow_steps[0].status = "failed"
                    workflow_steps[0].error = layout_response.error
                    logger.error(f"Layout extraction failed for job {job.id}: {layout_response.error}")
                    
                    return BannerWorkflowResponse(
                        success=False,
                        steps=workflow_steps,
                        final_result=None,
                        total_processing_time=time.time() - start_time
                    )
            else:
                # Skip layout extraction if no reference image
                workflow_steps[0].status = "completed"
                workflow_steps[0].result = {"success": True, "skipped": "No reference image provided"}
                workflow_steps[0].processing_time = 0.1
                logger.info(f"Layout extraction skipped for job {job.id} - no reference image")
            
            # Step 2: Prompt Generation
            workflow_steps[1].status = "in_progress"
            step_start = time.time()
            
            if layout_data:
                # Use extracted layout data
                prompt_response = self.prompt_service.optimize_prompt_for_job(
                    job.id, layout_data, request.user_requirements
                )
            else:
                # Create default layout data from user requirements
                from shared.banner import Background, CameraSettings
                default_background = Background(
                    scene=f"Professional banner design for: {request.user_requirements}",
                    subjects=[],
                    style="digital illustration",
                    color_palette=["#FFFFFF", "#000000", "#808080"],
                    lighting="balanced lighting",
                    mood="professional",
                    background="clean background",
                    composition="centered layout",
                    camera=CameraSettings()
                )
                default_layout = BannerLayoutData(text=[], background=default_background)
                
                prompt_response = self.prompt_service.optimize_prompt_for_job(
                    job.id, default_layout, request.user_requirements
                )
            
            workflow_steps[1].processing_time = time.time() - step_start
            
            if prompt_response.success:
                workflow_steps[1].status = "completed"
                
                # Extract detailed results from prompt generation
                optimized_dict = prompt_response.optimized_data.dict() if prompt_response.optimized_data else {}
                final_prompt_preview = None
                
                # Get the generated prompt from database for preview
                try:
                    prompt_generation = self.db.query(PromptGeneration).filter(
                        PromptGeneration.job_id == job.id
                    ).first()
                    if prompt_generation:
                        final_prompt_preview = prompt_generation.generation_prompt[:100] + "..." if len(prompt_generation.generation_prompt) > 100 else prompt_generation.generation_prompt
                except:
                    pass
                
                workflow_steps[1].result = {
                    "success": True,
                    "optimized_data": optimized_dict,
                    "final_prompt_preview": final_prompt_preview,
                    "processing_model": "gpt-4.1",
                    "processing_time": prompt_response.processing_time,
                    "optimized_style": optimized_dict.get("background", {}).get("style", "unknown"),
                    "optimized_mood": optimized_dict.get("background", {}).get("mood", "unknown")
                }
                
                # Log detailed step results
                logger.info(f"✅ Step 2 (Prompt Generation) completed for job {job.id}")
                logger.info(f"   🎭 Style: {optimized_dict.get('background', {}).get('style', 'unknown')}")
                logger.info(f"   😊 Mood: {optimized_dict.get('background', {}).get('mood', 'unknown')}")
                if final_prompt_preview:
                    logger.info(f"   📝 Prompt preview: {final_prompt_preview}")
                logger.info(f"   ⏱️  Processing time: {prompt_response.processing_time:.2f}s")
            else:
                workflow_steps[1].status = "failed"
                workflow_steps[1].error = prompt_response.error
                logger.error(f"Prompt generation failed for job {job.id}: {prompt_response.error}")
                
                return BannerWorkflowResponse(
                    success=False,
                    steps=workflow_steps,
                    final_result=None,
                    total_processing_time=time.time() - start_time
                )
            
            # Step 3: Background Image Generation
            workflow_steps[2].status = "in_progress"
            step_start = time.time()
            
            # Get final prompt from database
            prompt_generation = self.db.query(PromptGeneration).filter(
                PromptGeneration.job_id == job.id
            ).first()
            
            if not prompt_generation:
                error_msg = "Prompt generation data not found in database"
                workflow_steps[2].status = "failed"
                workflow_steps[2].error = error_msg
                workflow_steps[2].processing_time = time.time() - step_start
                
                return BannerWorkflowResponse(
                    success=False,
                    steps=workflow_steps,
                    final_result=None,
                    total_processing_time=time.time() - start_time
                )
            
            generation_response = self.generation_service.generate_banner(
                prompt_response.optimized_data, request.user_requirements, request.size, True
            )
            
            workflow_steps[2].processing_time = time.time() - step_start
            
            if generation_response.success:
                workflow_steps[2].status = "completed"
                
                # Extract detailed results from background image generation
                image_data_size = len(generation_response.image_data) if generation_response.image_data else 0
                
                workflow_steps[2].result = {
                    "success": True,
                    "has_image_data": bool(generation_response.image_data),
                    "has_image_url": bool(generation_response.image_url),
                    "image_size": generation_response.size,
                    "image_data_size_kb": round(image_data_size / 1024, 2) if image_data_size > 0 else 0,
                    "processing_model": "gpt-image-1",
                    "processing_time": generation_response.processing_time,
                    "prompt_used": prompt_generation.generation_prompt[:50] + "..." if prompt_generation and len(prompt_generation.generation_prompt) > 50 else (prompt_generation.generation_prompt if prompt_generation else "")
                }
                
                # Log detailed step results
                logger.info(f"✅ Step 3 (Background Generation) completed for job {job.id}")
                logger.info(f"   🖼️  Image size: {generation_response.size}")
                logger.info(f"   📁 Data size: {round(image_data_size / 1024, 2)} KB" if image_data_size > 0 else "   📁 No image data")
                logger.info(f"   🤖 Model: gpt-image-1")
                logger.info(f"   ⏱️  Processing time: {generation_response.processing_time:.2f}s")
            else:
                workflow_steps[2].status = "failed"
                workflow_steps[2].error = generation_response.error
                logger.error(f"Background generation failed for job {job.id}: {generation_response.error}")
                
                return BannerWorkflowResponse(
                    success=False,
                    steps=workflow_steps,
                    final_result=None,
                    total_processing_time=time.time() - start_time
                )
            
            # Step 4: Text Overlay
            workflow_steps[3].status = "in_progress"
            step_start = time.time()
            
            # Get text elements from the prompt generation data
            if not prompt_generation.optimized_scene:
                error_msg = "No optimized scene data found for text overlay"
                workflow_steps[3].status = "failed"
                workflow_steps[3].error = error_msg
                workflow_steps[3].processing_time = time.time() - step_start
                
                return BannerWorkflowResponse(
                    success=False,
                    steps=workflow_steps,
                    final_result=None,
                    total_processing_time=time.time() - start_time
                )
            
            # Extract text elements from optimized scene data
            text_elements = []
            original_width = None
            original_height = None
            
            # Extract original dimensions from image_info if available
            if prompt_generation.optimized_scene.get("image_info"):
                image_info = prompt_generation.optimized_scene["image_info"]
                original_width = image_info.get("original_width")
                original_height = image_info.get("original_height")
                logger.info(f"Found original dimensions in optimized scene: {original_width}x{original_height}")
            
            if prompt_generation.optimized_scene.get("text"):
                from shared.banner import TextElement
                text_data = prompt_generation.optimized_scene.get("text", [])
                for elem_data in text_data:
                    if isinstance(elem_data, dict):
                        text_elem = TextElement(
                            bbox=elem_data.get("bbox", [100, 100, 200, 50]),
                            font_size=elem_data.get("font_size", 36),
                            font_style=elem_data.get("font_style", "Arial"),
                            font_color=elem_data.get("font_color", "#000000"),
                            description=elem_data.get("description", "Sample text")
                        )
                        text_elements.append(text_elem)
            
            # If no text elements from layout, create default text based on user requirements
            if not text_elements:
                from shared.banner import TextElement
                # Create a default text element in the center
                default_text = TextElement(
                    bbox=[int(request.size.split('x')[0]) // 4, int(request.size.split('x')[1]) // 2 - 25, 
                          int(request.size.split('x')[0]) // 2, 50],
                    font_size=48,
                    font_style="Arial",
                    font_color="#FFFFFF",
                    description=request.user_requirements[:50] + "..." if len(request.user_requirements) > 50 else request.user_requirements
                )
                text_elements.append(default_text)
            
            # Perform text overlay with original dimensions for scaling
            overlay_result = self.text_overlay_service.overlay_text_for_job(
                job.id, generation_response.image_data, text_elements,
                original_width=original_width, original_height=original_height
            )
            
            workflow_steps[3].processing_time = time.time() - step_start
            
            if overlay_result.get("success"):
                workflow_steps[3].status = "completed"
                
                # Extract detailed results from text overlay
                final_image_size = len(overlay_result.get("final_image_data", ""))
                
                workflow_steps[3].result = {
                    "success": True,
                    "text_elements_count": overlay_result.get("text_elements_count", 0),
                    "final_image_size_kb": round(final_image_size / 1024, 2) if final_image_size > 0 else 0,
                    "processing_time": overlay_result.get("processing_time", 0),
                    "overlay_engine": "PIL/Pillow",
                    "coordinate_scaling_applied": overlay_result.get("coordinate_scaling_applied", False),
                    "original_dimensions": f"{original_width}x{original_height}" if original_width and original_height else None
                }
                
                # Create final result with composite image
                final_result = BannerGenerationResponse(
                    success=True,
                    image_data=overlay_result.get("final_image_data"),
                    image_url=None,
                    original_prompt=generation_response.original_prompt,
                    size=generation_response.size,
                    processing_time=overlay_result.get("processing_time", 0)
                )
                
                # Log detailed step results
                logger.info(f"✅ Step 4 (Text Overlay) completed for job {job.id}")
                logger.info(f"   📝 Text elements: {overlay_result.get('text_elements_count', 0)}")
                logger.info(f"   📁 Final size: {round(final_image_size / 1024, 2)} KB" if final_image_size > 0 else "   📁 No final image data")
                logger.info(f"   🎨 Engine: PIL/Pillow")
                logger.info(f"   ⏱️  Processing time: {overlay_result.get('processing_time', 0):.2f}s")
            else:
                workflow_steps[3].status = "failed"
                workflow_steps[3].error = overlay_result.get("error", "Text overlay failed")
                logger.error(f"Text overlay failed for job {job.id}: {overlay_result.get('error')}")
                
                return BannerWorkflowResponse(
                    success=False,
                    steps=workflow_steps,
                    final_result=None,
                    total_processing_time=time.time() - start_time
                )
            
            # All steps completed successfully
            total_time = time.time() - start_time
            logger.info(f"🎉 BANNER WORKFLOW COMPLETED SUCCESSFULLY for job {job.id}")
            logger.info(f"   📊 Total processing time: {total_time:.2f}s")
            logger.info(f"   🔄 Steps completed: {sum(1 for step in workflow_steps if step.status == 'completed')}/4")
            logger.info(f"   🎯 Final banner with text overlay ready!")
            
            return BannerWorkflowResponse(
                success=True,
                steps=workflow_steps,
                final_result=final_result,
                total_processing_time=total_time
            )
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(f"Workflow failed for job {job.id}: {error_msg}")
            
            # Update job status to failed
            update_job_status(
                self.db, job.id, JobStatus.FAILED,
                error_message=error_msg
            )
            
            # Mark current step as failed
            for step in workflow_steps:
                if step.status == "in_progress":
                    step.status = "failed"
                    step.error = error_msg
                    step.processing_time = time.time() - start_time
                    break
            
            return BannerWorkflowResponse(
                success=False,
                steps=workflow_steps,
                final_result=None,
                total_processing_time=time.time() - start_time
            )
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a banner job.
        
        Args:
            job_id: Banner job ID
            
        Returns:
            dict: Job status with step details or None if not found
        """
        job = get_job_by_id(self.db, job_id)
        if not job:
            return None
        
        # Convert to dict and add step details
        job_dict = job.to_dict()
        
        # Add detailed step information
        if job.steps:
            for step_data in job_dict["steps"]:
                step_number = step_data["step_number"]
                
                # Add step-specific data based on step number
                if step_number == 1:  # Layout extraction
                    extraction = self.db.query(LayoutExtraction).filter(
                        LayoutExtraction.job_id == job_id
                    ).first()
                    if extraction:
                        step_data["extraction_data"] = extraction.to_dict()
                
                elif step_number == 2:  # Prompt generation
                    prompt_gen = self.db.query(PromptGeneration).filter(
                        PromptGeneration.job_id == job_id
                    ).first()
                    if prompt_gen:
                        step_data["prompt_data"] = prompt_gen.to_dict()
        
        return job_dict

    def get_enhanced_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get enhanced job status with rich step outputs for frontend display.
        
        Args:
            job_id: Banner job ID
            
        Returns:
            dict: Enhanced job status with detailed step outputs and display data
        """
        from database import ImageGeneration
        
        job = get_job_by_id(self.db, job_id)
        if not job:
            return None
        
        # Start with basic job dict
        job_dict = job.to_dict()
        
        # Calculate overall progress
        completed_steps = sum(1 for step in job.steps if step.status == "completed")
        total_steps = len(job.steps)
        progress_percentage = int((completed_steps / total_steps) * 100) if total_steps > 0 else 0
        
        # Add progress information
        job_dict["progress"] = {
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "percentage": progress_percentage,
            "current_step": None,
            "current_step_name": None
        }
        
        # Find current step
        for step in job.steps:
            if step.status == "in_progress":
                job_dict["progress"]["current_step"] = step.step_number
                job_dict["progress"]["current_step_name"] = step.step_name
                break
        
        # Enhance each step with detailed output data
        if job.steps:
            for i, step_data in enumerate(job_dict["steps"]):
                step_number = step_data["step_number"]
                step_name = step_data["step_name"]
                
                # Add display-friendly step information
                step_data["display_name"] = {
                    "layout_extraction": "🔍 Extract Layout",
                    "prompt_generation": "✨ Generate Prompt", 
                    "image_generation": "🖼️ Create Background",
                    "text_overlay": "📝 Overlay Text"
                }.get(step_name, step_name.replace("_", " ").title())
                
                step_data["display_description"] = {
                    "layout_extraction": "Analyzing reference image layout and extracting text elements",
                    "prompt_generation": "Optimizing prompt based on layout and requirements",
                    "image_generation": "Generating background image with AI",
                    "text_overlay": "Overlaying text elements on background image"
                }.get(step_name, "Processing step...")
                
                # Add detailed step outputs based on step number
                if step_number == 1 and step_data["status"] in ["completed", "failed"]:
                    # Layout extraction step
                    extraction = self.db.query(LayoutExtraction).filter(
                        LayoutExtraction.job_id == job_id
                    ).first()
                    
                    if extraction:
                        step_data["output_summary"] = {
                            "text_elements_count": len(extraction.text_elements) if extraction.text_elements else 0,
                            "background_style": extraction.background_data.get("style", "unknown") if extraction.background_data else "unknown",
                            "scene_description": extraction.background_data.get("scene", "No scene") if extraction.background_data else "No scene",
                            "color_palette": extraction.background_data.get("color_palette", []) if extraction.background_data else [],
                            "processing_model": extraction.extraction_model
                        }
                        
                        # Include detailed extraction data
                        step_data["detailed_output"] = {
                            "text_elements": extraction.text_elements,
                            "background_data": extraction.background_data
                        }
                
                elif step_number == 2 and step_data["status"] in ["completed", "failed"]:
                    # Prompt generation step
                    prompt_gen = self.db.query(PromptGeneration).filter(
                        PromptGeneration.job_id == job_id
                    ).first()
                    
                    if prompt_gen:
                        # Get preview of generation prompt
                        prompt_preview = prompt_gen.generation_prompt[:100] + "..." if len(prompt_gen.generation_prompt) > 100 else prompt_gen.generation_prompt
                        
                        step_data["output_summary"] = {
                            "optimized_style": prompt_gen.optimized_scene.get("background", {}).get("style", "unknown") if prompt_gen.optimized_scene else "unknown",
                            "optimized_mood": prompt_gen.optimized_scene.get("background", {}).get("mood", "unknown") if prompt_gen.optimized_scene else "unknown",
                            "prompt_preview": prompt_preview,
                            "prompt_length": len(prompt_gen.generation_prompt),
                            "processing_model": prompt_gen.prompt_model
                        }
                        
                        # Include detailed prompt data
                        step_data["detailed_output"] = {
                            "optimized_scene": prompt_gen.optimized_scene,
                            "full_prompt": prompt_gen.generation_prompt
                        }
                
                elif step_number == 3 and step_data["status"] in ["completed", "failed"]:
                    # Background image generation step
                    image_gen = self.db.query(ImageGeneration).filter(
                        ImageGeneration.job_id == job_id
                    ).first()
                    
                    if image_gen:
                        image_data_size = len(image_gen.image_data) if image_gen.image_data else 0
                        
                        step_data["output_summary"] = {
                            "has_image_data": bool(image_gen.image_data),
                            "has_image_url": bool(image_gen.image_url),
                            "image_size": image_gen.image_size,
                            "image_format": image_gen.image_format,
                            "image_data_size_kb": round(image_data_size / 1024, 2) if image_data_size > 0 else 0,
                            "processing_model": image_gen.generation_model
                        }
                        
                        # Include background image data for display (if available)
                        step_data["detailed_output"] = {
                            "image_data": image_gen.image_data if image_gen.image_data else None,
                            "image_url": image_gen.image_url,
                            "generation_settings": image_gen.generation_settings
                        }
                
                elif step_number == 4 and step_data["status"] in ["completed", "failed"]:
                    # Text overlay step
                    text_overlay = self.db.query(TextOverlay).filter(
                        TextOverlay.job_id == job_id
                    ).first()
                    
                    if text_overlay:
                        final_image_size = len(text_overlay.final_image_data) if text_overlay.final_image_data else 0
                        
                        step_data["output_summary"] = {
                            "text_elements_count": len(text_overlay.text_elements) if text_overlay.text_elements else 0,
                            "has_final_image_data": bool(text_overlay.final_image_data),
                            "has_final_image_url": bool(text_overlay.final_image_url),
                            "final_image_size_kb": round(final_image_size / 1024, 2) if final_image_size > 0 else 0,
                            "overlay_engine": text_overlay.overlay_engine,
                            "text_quality_score": text_overlay.text_quality_score
                        }
                        
                        # Include final composite image data for display (if available)
                        step_data["detailed_output"] = {
                            "final_image_data": text_overlay.final_image_data if text_overlay.final_image_data else None,
                            "final_image_url": text_overlay.final_image_url,
                            "text_elements": text_overlay.text_elements,
                            "font_settings": text_overlay.font_settings
                        }
        
        # Add final result summary if job is completed
        if job.status == "completed":
            job_dict["final_summary"] = {
                "success": True,
                "has_final_image": bool(job.final_image_data or job.final_image_url),
                "total_processing_time": job.total_processing_time,
                "step_breakdown": [
                    {
                        "step": step.step_number,
                        "name": step.step_name.replace("_", " ").title(),
                        "time": step.processing_time,
                        "status": step.status
                    }
                    for step in job.steps
                ]
            }
        elif job.status == "failed":
            job_dict["final_summary"] = {
                "success": False,
                "error": job.error_message,
                "failed_at_step": None
            }
            
            # Find which step failed
            for step in job.steps:
                if step.status == "failed":
                    job_dict["final_summary"]["failed_at_step"] = {
                        "step_number": step.step_number,
                        "step_name": step.step_name,
                        "error": step.error_message
                    }
                    break
        
        return job_dict
    
    def retry_failed_job(self, job_id: str) -> BannerWorkflowResponse:
        """
        Retry a failed job from the last successful step.
        
        Args:
            job_id: Banner job ID to retry
            
        Returns:
            BannerWorkflowResponse: Retry response
        """
        job = get_job_by_id(self.db, job_id)
        if not job:
            return BannerWorkflowResponse(
                success=False,
                steps=[],
                final_result=None,
                total_processing_time=0.0
            )
        
        if job.status != JobStatus.FAILED:
            return BannerWorkflowResponse(
                success=False,
                steps=[],
                final_result=None,
                total_processing_time=0.0
            )
        
        # Create new workflow request from job data
        workflow_request = BannerWorkflowRequest(
            image_data=job.original_image_data,
            user_requirements=job.user_requirements,
            size=job.image_size
        )
        
        logger.info(f"Retrying failed job {job_id}")
        return self.execute_workflow(workflow_request)