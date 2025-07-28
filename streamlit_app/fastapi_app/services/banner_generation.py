"""
Banner generation service using OpenAI GPT-image-1.
Enhanced with precise bbox coordinate system for text positioning.
Migrated from banner-agent/streamlit/tools/generate_banner.py
"""
import time
import json
import logging
import os
import sys
from typing import Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
from fastapi_app.config import get_config

# Add streamlit_app directory to path for imports
streamlit_app_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(streamlit_app_dir))

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
from openai import OpenAI
from shared.banner import BannerLayoutData, BannerGenerationResponse

# Import our new bbox system components
try:
    from bbox_extractor import BboxExtractor
    from coordinate_normalizer import CoordinateNormalizer
    from text_positioner import TextPositioner
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Bbox system components not available: {e}")
    # Create dummy classes for fallback
    class BboxExtractor:
        def extract_precise_bbox_coordinates(self, *args, **kwargs):
            return {"success": False, "error": "BboxExtractor not available"}
    
    class CoordinateNormalizer:
        def normalize_text_elements(self, *args, **kwargs):
            return []
    
    class TextPositioner:
        def position_text_elements(self, *args, **kwargs):
            return {"success": False, "error": "TextPositioner not available"}

logger = logging.getLogger(__name__)


class BannerGenerationService:
    def __init__(self, openai_client: OpenAI, db_session: Optional['Session'] = None):
        self.client = openai_client
        self.db = db_session  # Database session for future database integration
        self.model = get_config("OPENAI_IMAGE_MODEL", "gpt-image-1")
        
        # Initialize bbox system components
        self.bbox_extractor = BboxExtractor()
        self.coordinate_normalizer = CoordinateNormalizer()
        self.text_positioner = TextPositioner()
        
        # Configuration for enhanced generation
        self.use_precise_bbox = True
        self.enable_text_overlay = True
    
    def create_banner_prompt_from_data(self, optimized_data: BannerLayoutData) -> str:
        """
        Convert optimized layout data image into a comprehensive gpt-image-1 prompt.
        Uses ONLY the image JSON part for image generation - text elements are handled separately.
        
        Args:
            optimized_data: Optimized prompt data from banner_prompt service
            
        Returns:
            str: Formatted prompt for gpt-image-1 using only image data
        """
        prompt_parts = []
        
        # Add image information
        image = optimized_data.image  # Now using image instead of background
        
        # Scene and style
        if image.scene:
            prompt_parts.append(f"Scene: {image.scene}")
        
        prompt_parts.append(f"Art style: {image.style}")
        prompt_parts.append(f"Mood: {image.mood}")
        
        # Subjects
        if image.subjects:
            subjects_text = ", ".join([
                subject.description for subject in image.subjects
            ])
            prompt_parts.append(f"Main subjects: {subjects_text}")
        
        # Color palette
        if image.color_palette:
            colors_text = ", ".join(image.color_palette)
            prompt_parts.append(f"Color palette: {colors_text}")
        
        # Lighting and composition
        if image.lighting:
            prompt_parts.append(f"Lighting: {image.lighting}")
        if image.composition:
            # Emphasize centered composition for products
            if "center" in image.composition.lower() and any(s.type == "product" or "product" in s.description.lower() for s in (image.subjects or [])):
                prompt_parts.append(f"Composition: {image.composition}. IMPORTANT: The product must be perfectly centered horizontally in the frame")
            else:
                prompt_parts.append(f"Composition: {image.composition}")
        
        # Camera settings
        camera = image.camera
        camera_parts = []
        if camera.angle:
            camera_parts.append(f"angle: {camera.angle}")
        if camera.distance:
            camera_parts.append(f"distance: {camera.distance}")
        if camera.focus:
            camera_parts.append(f"focus: {camera.focus}")
        
        if camera_parts:
            prompt_parts.append(f"Camera: {', '.join(camera_parts)}")
        
        # Background elements (nested background field)
        if hasattr(image, 'background') and image.background:
            prompt_parts.append(f"Background elements: {image.background}")
        
        # Add negative prompt from image if available
        if hasattr(image, 'negative_prompt') and image.negative_prompt:
            prompt_parts.append(f"IMPORTANT - DO NOT INCLUDE: {image.negative_prompt}")
        
        # Note: Text elements are handled separately in the text overlay step
        # Only background elements are used for image generation
        
        # Combine all parts
        full_prompt = ". ".join(prompt_parts)
        
        # Add banner-specific instructions with strong NO TEXT emphasis
        banner_instructions = """Create a professional banner/poster background design that is visually striking and suitable for marketing purposes. The design should be well-balanced and eye-catching. 
        
CRITICAL REQUIREMENTS:
- Generate ONLY the visual background/scene without ANY text, letters, numbers, or typography
- NO text overlays, NO words, NO characters of any kind
- Focus entirely on the visual elements, products, and background
- Leave clean areas where text can be added later
- The image must be completely text-free"""
        
        return f"{full_prompt}. {banner_instructions}"
    
    def generate_banner_with_gpt_image(self, prompt: str, size: str = "1024x1024") -> Dict[str, Any]:
        """
        Generate a banner image using gpt-image-1.
        
        Args:
            prompt: Text prompt for image generation
            size: Image size (1024x1024, 1536x1024, or 1024x1536)
            
        Returns:
            dict: Result containing image data or error information
        """
        try:
            # Validate size for gpt-image-1
            valid_sizes = ["1024x1024", "1536x1024", "1024x1536"]
            if size not in valid_sizes:
                return {
                    "success": False,
                    "error": f"Invalid size {size}. Must be one of {valid_sizes}"
                }
            
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=size
            )
            
            # Get the image data (either base64 or URL)
            image_base64 = getattr(response.data[0], 'b64_json', None)
            image_url = getattr(response.data[0], 'url', None)
            
            return {
                "success": True,
                "image_data": image_base64,
                "image_url": image_url,
                "original_prompt": prompt,
                "size": size
            }
            
        except Exception as e:
            logger.error(f"gpt-image-1 generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_prompt": prompt
            }
    
    def generate_banner_from_text_only(self, user_requirements: str, size: str = "1536x1024") -> Dict[str, Any]:
        """
        Generate a banner from text requirements only (no reference image).
        
        Args:
            user_requirements: User's text requirements
            size: Image size for generation
            
        Returns:
            dict: Result containing image data or error information
        """
        try:
            # Create a comprehensive prompt from user requirements
            enhanced_prompt = f"""
            Create a professional banner/poster BACKGROUND design based on the following requirements: {user_requirements}
            
            Design specifications:
            - Style: Modern, visually appealing digital illustration
            - Composition: Well-balanced and eye-catching layout
            - Quality: High-resolution, suitable for marketing purposes
            - Colors: Use a harmonious color palette that matches the theme
            - Layout: Professional banner format with clear focal points for FUTURE text placement
            
            CRITICAL: Generate ONLY the visual background/scene. DO NOT include any text, letters, numbers, words, or typography of any kind. The image must be completely text-free. Focus on products, visual elements, and background only.
            
            The final design should be a polished, marketing-ready BACKGROUND that will have text added separately.
            """
            
            return self.generate_banner_with_gpt_image(enhanced_prompt, size)
            
        except Exception as e:
            logger.error(f"Text-only banner generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_banner_with_precise_text_overlay(self, optimized_data: BannerLayoutData, 
                                                 size: str = "1024x1024",
                                                 reference_image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate banner with precise text overlay using bbox coordinates.
        
        Args:
            optimized_data: Layout data with text elements and bbox coordinates
            size: Target banner size
            reference_image_data: Optional reference image for bbox validation
            
        Returns:
            dict: Result with final banner including text overlays
        """
        try:
            # Step 1: Generate background image (without text)
            background_prompt = self.create_banner_prompt_from_data(optimized_data)
            # Add extra strong instruction to generate clean background without text
            background_prompt += " ABSOLUTELY NO TEXT: Generate a completely text-free background image. No letters, no words, no numbers, no typography, no text of any kind. Only visual elements and products."
            
            background_result = self.generate_banner_with_gpt_image(background_prompt, size)
            
            if not background_result.get("success"):
                return background_result
            
            background_image_data = background_result.get("image_data")
            if not background_image_data:
                return {
                    "success": False,
                    "error": "No image data in background generation result"
                }
            
            # Step 2: Extract and validate bbox coordinates
            if self.use_precise_bbox and reference_image_data:
                # Extract bbox from reference image for validation
                bbox_result = self.bbox_extractor.extract_precise_bbox_coordinates(reference_image_data)
                
                if bbox_result.get("success"):
                    logger.info(f"Extracted {len(bbox_result.get('detected_boxes', []))} bbox regions from reference")
            
            # Step 3: Normalize text elements for target size
            text_elements = []
            if optimized_data.text:
                # Convert Pydantic models to dicts
                text_elements = [element.dict() for element in optimized_data.text]
                
                # Get source image dimensions
                source_width = optimized_data.image_info.original_width if optimized_data.image_info else 1024
                source_height = optimized_data.image_info.original_height if optimized_data.image_info else 1024
                source_size = f"{source_width}x{source_height}"
                
                # Normalize for target size if different
                if source_size != size and size in self.coordinate_normalizer.supported_sizes:
                    text_elements = self.coordinate_normalizer.normalize_text_elements(
                        text_elements, source_size, size
                    )
            
            # Step 4: Apply text overlays with precise positioning
            if self.enable_text_overlay and text_elements:
                positioning_result = self.text_positioner.position_text_elements(
                    background_image_data, text_elements, size
                )
                
                if positioning_result.get("success"):
                    return {
                        "success": True,
                        "image_data": positioning_result["final_image_data"],
                        "original_prompt": background_prompt,
                        "size": size,
                        "text_positioning": positioning_result,
                        "method": "precise_bbox_overlay"
                    }
                else:
                    logger.warning(f"Text positioning failed: {positioning_result.get('error')}")
                    # Fallback to background only
                    return {
                        "success": True,
                        "image_data": background_image_data,
                        "original_prompt": background_prompt,
                        "size": size,
                        "warning": "Text overlay failed, returning background only",
                        "method": "background_only"
                    }
            else:
                # Return background image without text overlay
                return {
                    "success": True,
                    "image_data": background_image_data,
                    "original_prompt": background_prompt,
                    "size": size,
                    "method": "background_only"
                }
            
        except Exception as e:
            logger.error(f"Enhanced banner generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_bbox_coordinates(self, layout_data: BannerLayoutData, 
                                 image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate bbox coordinates in layout data.
        
        Args:
            layout_data: Layout data with text elements
            image_data: Optional reference image for validation
            
        Returns:
            dict: Validation results and recommendations
        """
        try:
            validation_results = {
                "valid_elements": [],
                "invalid_elements": [],
                "recommendations": []
            }
            
            # Get image dimensions
            if layout_data.image_info:
                img_width = layout_data.image_info.original_width
                img_height = layout_data.image_info.original_height
            else:
                img_width = img_height = 1024
            
            # Validate each text element
            for i, text_element in enumerate(layout_data.text or []):
                bbox = text_element.bbox
                
                # Check bbox format and bounds
                is_valid = self.coordinate_normalizer.validate_bbox_proportions(
                    bbox, img_width, img_height
                )
                
                element_validation = {
                    "index": i,
                    "bbox": bbox,
                    "valid": is_valid,
                    "issues": []
                }
                
                if not is_valid:
                    # Identify specific issues
                    x, y, w, h = bbox
                    if x < 0 or y < 0:
                        element_validation["issues"].append("Negative coordinates")
                    if x + w > img_width or y + h > img_height:
                        element_validation["issues"].append("Extends beyond image bounds")
                    if w < 10 or h < 8:
                        element_validation["issues"].append("Too small for readable text")
                    
                    validation_results["invalid_elements"].append(element_validation)
                else:
                    validation_results["valid_elements"].append(element_validation)
            
            # Generate recommendations
            if validation_results["invalid_elements"]:
                validation_results["recommendations"].append(
                    "Adjust bbox coordinates to fit within image bounds"
                )
                validation_results["recommendations"].append(
                    "Ensure minimum text size for readability"
                )
            
            # Cross-reference with computer vision if image provided
            if image_data and self.use_precise_bbox:
                cv_result = self.bbox_extractor.extract_precise_bbox_coordinates(image_data)
                if cv_result.get("success"):
                    validation_results["cv_detection"] = {
                        "detected_regions": len(cv_result.get("detected_boxes", [])),
                        "high_confidence": len([
                            b for b in cv_result.get("detected_boxes", [])
                            if b.get("confidence", 0) > 0.7
                        ])
                    }
            
            validation_results["summary"] = {
                "total_elements": len(layout_data.text or []),
                "valid_elements": len(validation_results["valid_elements"]),
                "validation_score": (
                    len(validation_results["valid_elements"]) / 
                    max(1, len(layout_data.text or []))
                )
            }
            
            return {
                "success": True,
                "validation": validation_results
            }
            
        except Exception as e:
            logger.error(f"Bbox validation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_banner(self, optimized_data: Optional[BannerLayoutData], 
                      user_requirements: str, size: str = "1024x1024", 
                      include_image_data: bool = True,
                      reference_image_data: Optional[str] = None) -> BannerGenerationResponse:
        """
        Main service method to generate banner with enhanced bbox support.
        
        Args:
            optimized_data: Optimized layout data (None for text-only generation)
            user_requirements: User's requirements (fallback for text-only)
            size: Image size for generation
            include_image_data: Whether to include base64 image data
            reference_image_data: Optional reference image for bbox validation
            
        Returns:
            BannerGenerationResponse: Service response with generated banner or error
        """
        start_time = time.time()
        
        try:
            if optimized_data and optimized_data.text and self.enable_text_overlay:
                # Use enhanced generation with precise text overlay
                logger.info("Using enhanced banner generation with precise text overlay")
                result = self.generate_banner_with_precise_text_overlay(
                    optimized_data, size, reference_image_data
                )
                
                # Add workflow metadata
                if result.get("success") and result.get("text_positioning"):
                    positioning_data = result.get("text_positioning", {})
                    result["workflow_data"] = {
                        "layout_extraction": True,
                        "bbox_validation": True,
                        "text_overlay": True,
                        "positioning_statistics": positioning_data.get("statistics", {}),
                        "method": result.get("method", "precise_bbox_overlay")
                    }
                
            elif optimized_data:
                # Generate from optimized layout data (traditional method)
                logger.info("Using traditional banner generation")
                prompt = self.create_banner_prompt_from_data(optimized_data)
                result = self.generate_banner_with_gpt_image(prompt, size)
                
                if result.get("success"):
                    result["workflow_data"] = {
                        "layout_extraction": True,
                        "bbox_validation": False,
                        "text_overlay": False,
                        "method": "traditional_prompt_based"
                    }
            else:
                # Generate from text requirements only
                logger.info("Using text-only banner generation")
                result = self.generate_banner_from_text_only(user_requirements, size)
                
                if result.get("success"):
                    result["workflow_data"] = {
                        "layout_extraction": False,
                        "bbox_validation": False,
                        "text_overlay": False,
                        "method": "text_only"
                    }
            
            if not result.get("success", False):
                return BannerGenerationResponse(
                    success=False,
                    error=result.get("error", "Unknown generation error"),
                    size=size,
                    processing_time=time.time() - start_time
                )
            
            # Prepare response
            response_data = {
                "success": True,
                "image_url": result.get("image_url"),
                "original_prompt": result.get("original_prompt"),
                "size": size,
                "processing_time": time.time() - start_time
            }
            
            # Include image data if requested
            if include_image_data and result.get("image_data"):
                response_data["image_data"] = result["image_data"]
            
            return BannerGenerationResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Banner generation service failed: {e}")
            return BannerGenerationResponse(
                success=False,
                error=f"Banner generation failed: {str(e)}",
                size=size,
                processing_time=time.time() - start_time
            )