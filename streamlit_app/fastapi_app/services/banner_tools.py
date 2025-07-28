"""
Banner Tools Service - OpenAI Tools-based Banner Generation
Implements a tools-based workflow for banner creation with different OpenAI models.
"""
import time
import json
import logging
import os
import base64
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
from sqlalchemy.orm import Session

# Add parent directory to path for shared module access
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from shared.banner import BannerLayoutData, BannerGenerationResponse
from database import (
    PromptGeneration, LayoutExtraction, update_step_status, 
    StepStatus, JobStatus, update_job_status
)

logger = logging.getLogger(__name__)


class BannerTool:
    """Base class for banner generation tools"""
    
    def __init__(self, name: str, model: str, description: str):
        self.name = name
        self.model = model
        self.description = description
    
    def execute(self, openai_client: OpenAI, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given input data"""
        raise NotImplementedError("Subclasses must implement execute method")


class ExtractLayoutTool(BannerTool):
    """Tool for extracting layout information from images using GPT-4.1"""
    
    def __init__(self):
        super().__init__(
            name="extract_layout",
            model="gpt-4o",  # Using gpt-4o for vision capabilities
            description="Extract layout and design elements from reference images"
        )
    
    def execute(self, openai_client: OpenAI, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract layout from image using GPT-4o vision"""
        try:
            image_data = input_data.get("image_data")
            if not image_data:
                return {
                    "success": False,
                    "error": "No image data provided for layout extraction"
                }
            
            prompt = """
            Analyze this banner/poster image and extract detailed layout information in JSON format.
            
            Return a JSON object with this exact structure:
            {
                "background": {
                    "scene": "Description of the main scene or background",
                    "style": "Art style (e.g., 'modern', 'vintage', 'minimalist')",
                    "mood": "Overall mood/atmosphere",
                    "subjects": ["Primary subject", "Secondary subject"],
                    "color_palette": ["Dominant color 1", "Dominant color 2", "Accent color"],
                    "lighting": "Lighting description",
                    "composition": "Composition style",
                    "camera": {
                        "angle": "Camera angle",
                        "distance": "Shot distance", 
                        "focus": "Focus description"
                    }
                },
                "text": [
                    {
                        "bbox": [x, y, width, height],
                        "font_size": estimated_font_size,
                        "font_style": "font style description",
                        "font_color": "text color",
                        "description": "Text content or placeholder description"
                    }
                ]
            }
            """
            
            response = openai_client.chat.completions.create(
                model=self.model,
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            layout_data = json.loads(content)
            
            return {
                "success": True,
                "layout_data": layout_data,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"Layout extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model
            }


class WriteBannerPromptTool(BannerTool):
    """Tool for generating banner prompts using GPT-4.1"""
    
    def __init__(self):
        super().__init__(
            name="write_banner_prompt",
            model="gpt-4o",
            description="Generate optimized prompts for banner creation based on layout and requirements"
        )
    
    def execute(self, openai_client: OpenAI, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate banner prompt using GPT-4o"""
        try:
            user_requirements = input_data.get("user_requirements", "")
            layout_data = input_data.get("layout_data", {})
            
            if not user_requirements:
                return {
                    "success": False,
                    "error": "User requirements are required for prompt generation"
                }
            
            prompt = f"""
            As a creative director, generate a detailed and optimized prompt for GPT-image-1 to create a banner.
            
            User Requirements: "{user_requirements}"
            
            Layout Data (if available): {json.dumps(layout_data, indent=2) if layout_data else "None"}
            
            Create a comprehensive prompt that includes:
            1. Main scene description
            2. Visual style and mood
            3. Color palette
            4. Composition and layout
            5. Lighting and atmosphere
            6. Any specific elements mentioned in user requirements
            
            Return ONLY the prompt text that will be sent to GPT-image-1, without any JSON formatting or additional explanation.
            The prompt should be creative, detailed, and optimized for image generation.
            """
            
            response = openai_client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                temperature=0.8,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a creative director specializing in banner and poster design. Generate detailed, creative prompts for AI image generation."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            generated_prompt = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "generated_prompt": generated_prompt,
                "background_data": layout_data.get("background", {}),
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model
            }


class GenerateBannerTool(BannerTool):
    """Tool for generating banner images using GPT-image-1"""
    
    def __init__(self):
        super().__init__(
            name="generate_banner", 
            model="gpt-image-1",
            description="Generate banner images from text prompts"
        )
    
    def execute(self, openai_client: OpenAI, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate banner image using GPT-image-1"""
        try:
            prompt = input_data.get("generated_prompt") or input_data.get("prompt", "")
            size = input_data.get("size", "1792x1024")
            
            if not prompt:
                return {
                    "success": False,
                    "error": "No prompt provided for image generation"
                }
            
            # Validate size for GPT-image-1
            valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
            if size not in valid_sizes:
                return {
                    "success": False,
                    "error": f"Invalid size {size}. Must be one of {valid_sizes}"
                }
            
            response = openai_client.images.generate(
                model=self.model,
                prompt=prompt,
                size=size
            )
            
            image_base64 = response.data[0].b64_json
            
            return {
                "success": True,
                "image_data": image_base64,
                "original_prompt": prompt,
                "size": size,
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model
            }


class AddTextLayoutTool(BannerTool):
    """Tool for adding text overlays using code interpreter"""
    
    def __init__(self):
        super().__init__(
            name="add_text_layout",
            model="gpt-4o",  # Using GPT-4o with code execution
            description="Add text overlays to generated banners using code interpreter"
        )
    
    def execute(self, openai_client: OpenAI, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add text layout using code interpreter (GPT-4o)"""
        try:
            image_data = input_data.get("image_data")
            text_elements = input_data.get("text_elements", [])
            
            if not image_data:
                return {
                    "success": False,
                    "error": "No image data provided for text overlay"
                }
            
            if not text_elements:
                # Return original image if no text elements
                return {
                    "success": True,
                    "image_data": image_data,
                    "message": "No text elements to add",
                    "model_used": self.model
                }
            
            # For now, implement a simplified version without code interpreter
            # In production, this would use OpenAI's code interpreter capability
            prompt = f"""
            I need to add text overlays to a banner image. Here are the text elements to add:
            {json.dumps(text_elements, indent=2)}
            
            Please provide Python code using PIL/Pillow to add these text elements to the image.
            The code should:
            1. Load the image from base64 data
            2. Add each text element at the specified position with appropriate styling
            3. Return the modified image as base64
            
            Provide complete, executable Python code.
            """
            
            response = openai_client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Python developer specializing in image processing with PIL/Pillow. Provide complete, executable code for text overlay tasks."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # For now, return the original image since we're not executing code
            # In production implementation, this would execute the generated code
            return {
                "success": True,
                "image_data": image_data,
                "generated_code": response.choices[0].message.content,
                "message": "Text layout tool executed (code generation mode)",
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"Text layout addition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model
            }


class BannerToolsService:
    """Service for managing banner generation tools and workflows"""
    
    def __init__(self, openai_client: OpenAI, db_session: Session):
        self.client = openai_client
        self.db = db_session
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> Dict[str, BannerTool]:
        """Initialize all available tools"""
        return {
            "extract_layout": ExtractLayoutTool(),
            "write_banner_prompt": WriteBannerPromptTool(),
            "generate_banner": GenerateBannerTool(),
            "add_text_layout": AddTextLayoutTool()
        }
    
    def get_tool(self, tool_name: str) -> Optional[BannerTool]:
        """Get a specific tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools"""
        return [
            {
                "name": tool.name,
                "model": tool.model,
                "description": tool.description
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with input data"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        start_time = time.time()
        result = tool.execute(self.client, input_data)
        processing_time = time.time() - start_time
        
        result["processing_time"] = processing_time
        result["tool_name"] = tool_name
        
        return result
    
    def execute_workflow(self, user_requirements: str, image_data: Optional[str] = None, 
                        size: str = "1792x1024") -> BannerGenerationResponse:
        """
        Execute the complete banner generation workflow:
        extract_layout -> write_banner_prompt -> generate_banner -> add_text_layout
        """
        start_time = time.time()
        workflow_data = {}
        
        try:
            # Step 1: Extract Layout (if image provided)
            if image_data:
                logger.info("Step 1: Extracting layout from reference image")
                layout_result = self.execute_tool("extract_layout", {"image_data": image_data})
                
                if not layout_result["success"]:
                    return BannerGenerationResponse(
                        success=False,
                        error=f"Layout extraction failed: {layout_result['error']}",
                        size=size,
                        processing_time=time.time() - start_time
                    )
                
                workflow_data["layout_data"] = layout_result["layout_data"]
                workflow_data["text_elements"] = layout_result["layout_data"].get("text", [])
            else:
                workflow_data["layout_data"] = {}
                workflow_data["text_elements"] = []
            
            # Step 2: Write Banner Prompt
            logger.info("Step 2: Generating optimized banner prompt")
            prompt_result = self.execute_tool("write_banner_prompt", {
                "user_requirements": user_requirements,
                "layout_data": workflow_data["layout_data"]
            })
            
            if not prompt_result["success"]:
                return BannerGenerationResponse(
                    success=False,
                    error=f"Prompt generation failed: {prompt_result['error']}",
                    size=size,
                    processing_time=time.time() - start_time
                )
            
            workflow_data["generated_prompt"] = prompt_result["generated_prompt"]
            
            # Step 3: Generate Banner
            logger.info("Step 3: Generating banner image with GPT-image-1")
            banner_result = self.execute_tool("generate_banner", {
                "generated_prompt": workflow_data["generated_prompt"],
                "size": size
            })
            
            if not banner_result["success"]:
                return BannerGenerationResponse(
                    success=False,
                    error=f"Banner generation failed: {banner_result['error']}",
                    size=size,
                    processing_time=time.time() - start_time
                )
            
            workflow_data["image_data"] = banner_result["image_data"]
            
            # Step 4: Add Text Layout
            logger.info("Step 4: Adding text layout overlays")
            text_result = self.execute_tool("add_text_layout", {
                "image_data": workflow_data["image_data"],
                "text_elements": workflow_data["text_elements"]
            })
            
            if text_result["success"]:
                workflow_data["final_image_data"] = text_result["image_data"]
            else:
                # Use original image if text overlay fails
                workflow_data["final_image_data"] = workflow_data["image_data"]
                logger.warning(f"Text overlay failed, using original image: {text_result.get('error')}")
            
            # Return successful response
            return BannerGenerationResponse(
                success=True,
                image_data=workflow_data["final_image_data"],
                image_url=None,  # GPT-image-1 returns base64, not URL
                original_prompt=workflow_data["generated_prompt"],
                size=size,
                processing_time=time.time() - start_time,
                workflow_data=workflow_data
            )
            
        except Exception as e:
            logger.error(f"Banner workflow failed: {e}")
            return BannerGenerationResponse(
                success=False,
                error=f"Banner workflow failed: {str(e)}",
                size=size,
                processing_time=time.time() - start_time
            )