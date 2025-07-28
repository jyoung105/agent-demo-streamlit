"""
Enhanced banner layout extraction service with database integration and precise bbox extraction.
Provides step-by-step processing with persistent data storage and accurate coordinate extraction.
"""
import time
import json
import logging
import os
import hashlib
import base64
import io
import sys
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
from openai import OpenAI
from sqlalchemy.orm import Session
from pathlib import Path
from shared.banner import BannerLayoutData, BannerLayoutResponse
from database import (
    LayoutExtraction, BannerJob, update_step_status, 
    StepStatus, JobStatus, update_job_status
)

# Add streamlit_app directory to path for bbox system imports
streamlit_app_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(streamlit_app_dir))

# Import bbox system components
try:
    from bbox_extractor import BboxExtractor
    from coordinate_normalizer import CoordinateNormalizer
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Bbox system components not available: {e}")
    # Create dummy classes for fallback
    class BboxExtractor:
        def extract_precise_bbox_coordinates(self, *args, **kwargs):
            return {"success": False, "error": "BboxExtractor not available"}
        def extract_bbox_from_ai_response(self, *args, **kwargs):
            return []
    
    class CoordinateNormalizer:
        def validate_bbox_proportions(self, *args, **kwargs):
            return True

logger = logging.getLogger(__name__)


class BannerLayoutService:
    def __init__(self, openai_client: OpenAI, db_session: Session):
        self.client = openai_client
        self.db = db_session
        self.model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
        self.max_tokens = int(os.getenv("OPENAI_VISION_MAX_TOKENS", "2000"))
        self.temperature = float(os.getenv("OPENAI_VISION_TEMPERATURE", "0.1"))
        
        # Initialize bbox system components
        self.bbox_extractor = BboxExtractor()
        self.coordinate_normalizer = CoordinateNormalizer()
        
        # Enhanced extraction configuration
        self.use_cv_validation = True
        self.enable_bbox_refinement = True
    
    def get_image_dimensions(self, image_data: str) -> Tuple[int, int]:
        """
        Get original image dimensions from base64 encoded image data.
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            Tuple of (width, height) in pixels
        """
        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return image.size  # Returns (width, height)
        except Exception as e:
            logger.warning(f"Could not determine image dimensions: {e}")
            return (1024, 1024)  # Default dimensions

    def extract_layout_from_image(self, image_data: str) -> Dict[str, Any]:
        """
        Extract layout information from a base64 encoded image using GPT-4.1 Vision API.
        
        Args:
            image_data (str): Base64 encoded image data
            
        Returns:
            dict: JSON structure with 'text', 'background', and 'image_info' parameters
        """
        
        # Get original image dimensions
        original_width, original_height = self.get_image_dimensions(image_data)
        
        # Prepare the enhanced prompt for layout extraction
        prompt = f"""Analyze this banner/poster image and extract the layout information in the following exact JSON format.

IMPORTANT: The original image dimensions are {original_width}x{original_height} pixels. Use these dimensions for accurate bounding box coordinates.

STEP 1 - EXTRACT TEXT ELEMENTS FIRST:
Carefully identify and extract ALL text elements in the image. For each text element, provide precise bounding box coordinates and styling information.

STEP 2 - EXTRACT BACKGROUND DETAILS:
After extracting text, analyze the background composition in great detail, including every visual element, styling choice, and compositional technique.

{{
  "image_info": {{
    "original_width": {original_width},
    "original_height": {original_height}
  }},
  "text": [
    {{
      "bbox": [x, y, width, height],
      "font_size": 24,
      "font_style": "bold", 
      "font_color": "#000000",
      "description": "Main title text describing what the text says"
    }}
  ],
  "image": {{
    "scene": "DETAILED description of the overall scene including environment, setting, and context",
    "subjects": [
      {{
        "type": "specific type of subject (product, person, animal, object, etc.)",
        "description": "VERY detailed description including size, shape, color, texture, material",
        "pose": "exact pose, orientation, or arrangement", 
        "position": "precise position in frame with coordinates if possible (e.g., 'centered at bottom 30%', 'left side at x:200')"
      }}
    ],
    "style": "SPECIFIC art style (e.g., 'hyperrealistic 3D render', 'minimalist vector illustration', 'watercolor painting', 'professional product photography')",
    "color_palette": ["#HEX1", "#HEX2", "#HEX3"],
    "lighting": "DETAILED lighting setup (e.g., 'soft diffused light from top-left', 'dramatic rim lighting', 'golden hour natural light')",
    "mood": "SPECIFIC mood and emotional tone (e.g., 'luxurious and sophisticated', 'fresh and energetic', 'calm and trustworthy')",
    "background": "DETAILED background elements including textures, patterns, gradients (e.g., 'water ripples with caustic light reflections', 'geometric hexagon pattern')",
    "composition": "PRECISE composition technique and layout (e.g., 'rule of thirds with product at bottom intersection', 'centered symmetrical layout', 'diagonal dynamic composition')",
    "camera": {{
      "angle": "EXACT camera angle (e.g., 'straight-on eye level', '45-degree high angle', 'worm's eye view')",
      "distance": "PRECISE shot distance (e.g., 'extreme close-up showing texture', 'medium shot with 50% negative space', 'wide establishing shot')",
      "focus": "DETAILED focus and depth description (e.g., 'tack sharp focus on product with f/2.8 bokeh background', 'deep focus f/16 everything sharp')"
    }}
  }}
}}

CRITICAL INSTRUCTIONS FOR TEXT EXTRACTION:
1. FIRST, scan the ENTIRE image for ALL text elements - don't miss any!
2. For font sizes, REDUCE the detected size by 20-30% to prevent overflow:
   - If you detect 100px → report as 70-80px
   - If you detect 60px → report as 40-50px
   - If you detect 40px → report as 28-35px
   - This ensures text fits properly when regenerated
3. For bounding boxes, provide ACCURATE coordinates:
   - x: exact left edge position from left side of image
   - y: exact top edge position from top of image  
   - width: exact width of text element in pixels
   - height: exact height of text element in pixels
4. Font colors must be EXACT hex codes from the actual text pixels
5. Description must be the EXACT text content, character by character

CRITICAL INSTRUCTIONS FOR BACKGROUND EXTRACTION:
1. Describe EVERY visual element you can see
2. Be EXTREMELY detailed about:
   - Product placement and size
   - Surface textures and materials
   - Color gradients and transitions
   - Shadow and highlight details
   - Environmental elements
   - Special effects or filters
3. For composition, note:
   - Exact placement of main subject
   - Negative space distribution
   - Visual hierarchy
   - Balance and symmetry
4. Style should be very specific (not just "digital art" but "hyperrealistic 3D render with subsurface scattering")

Return ONLY the JSON object, no additional text or markdown formatting."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
                
            layout_data = json.loads(response_text)
            return layout_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error(f"Raw response: {response_text}")
            return {"error": f"JSON parsing failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Error calling OpenAI Vision API: {e}")
            return {"error": f"API call failed: {str(e)}"}
    
    def validate_and_normalize_layout_data(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize layout data structure.
        
        Args:
            layout_data: Raw layout data from OpenAI
            
        Returns:
            dict: Validated and normalized layout data
        """
        if "error" in layout_data:
            return layout_data
        
        # Ensure required structure exists
        if not isinstance(layout_data, dict):
            return {"error": f"Invalid layout data type: {type(layout_data).__name__}"}
        
        # Validate required keys and add defaults if missing
        validated_data = {
            "text": layout_data.get("text", []),
            "image": layout_data.get("image", layout_data.get("background", {})),  # Support both "image" and "background" for backward compatibility
            "image_info": layout_data.get("image_info", {"original_width": 1024, "original_height": 1024})
        }
        
        # Ensure image has required structure
        image = validated_data["image"]
        if not isinstance(image, dict):
            validated_data["image"] = {}
            image = validated_data["image"]
        
        # Set defaults for image fields
        image.setdefault("scene", "Banner design layout")
        image.setdefault("subjects", [])
        image.setdefault("style", "digital illustration")
        image.setdefault("color_palette", ["#FFFFFF", "#000000", "#808080"])
        image.setdefault("lighting", "balanced lighting")
        image.setdefault("mood", "professional")
        image.setdefault("background", "clean background")
        image.setdefault("composition", "centered layout")
        image.setdefault("camera", {
            "angle": "eye level",
            "distance": "medium shot", 
            "focus": "sharp focus"
        })
        
        # Ensure text is a list
        if not isinstance(validated_data["text"], list):
            validated_data["text"] = []
        
        # Validate each text element
        validated_text = []
        for text_elem in validated_data["text"]:
            if isinstance(text_elem, dict):
                validated_elem = {
                    "bbox": text_elem.get("bbox", [0, 0, 100, 50]),
                    "font_size": text_elem.get("font_size", 48),
                    "font_style": text_elem.get("font_style", "regular"),
                    "font_color": text_elem.get("font_color", "#000000"),
                    "description": text_elem.get("description", "Text element")
                }
                validated_text.append(validated_elem)
                
        validated_data["text"] = validated_text
        
        return validated_data
    
    def enhance_bbox_coordinates_with_cv(self, layout_data: Dict[str, Any], 
                                        image_data: str) -> Dict[str, Any]:
        """
        Enhance AI-extracted bbox coordinates using computer vision validation.
        
        Args:
            layout_data: Layout data from AI vision model
            image_data: Base64 encoded image for CV analysis
            
        Returns:
            Enhanced layout data with refined bbox coordinates
        """
        try:
            if not self.use_cv_validation:
                return layout_data
            
            # Extract CV-detected text regions
            cv_result = self.bbox_extractor.extract_precise_bbox_coordinates(image_data)
            
            if not cv_result.get("success"):
                logger.warning("CV bbox extraction failed, using AI coordinates only")
                return layout_data
            
            cv_boxes = cv_result.get("detected_boxes", [])
            ai_text_elements = layout_data.get("text", [])
            
            if not cv_boxes or not ai_text_elements:
                return layout_data
            
            # Match AI text elements with CV detections
            enhanced_text_elements = []
            
            for ai_element in ai_text_elements:
                ai_bbox = ai_element.get("bbox", [0, 0, 100, 50])
                
                # Find best matching CV detection
                best_match = self._find_best_bbox_match(ai_bbox, cv_boxes)
                
                if best_match and best_match["confidence"] > 0.5:
                    # Use CV coordinates but keep AI metadata
                    enhanced_element = ai_element.copy()
                    enhanced_element["bbox"] = best_match["bbox"]
                    enhanced_element["cv_confidence"] = best_match["confidence"]
                    enhanced_element["refinement"] = "cv_enhanced"
                    logger.info(f"Enhanced bbox with CV: {best_match['bbox']}")
                else:
                    # Keep original AI coordinates
                    enhanced_element = ai_element.copy()
                    enhanced_element["cv_confidence"] = 0.0
                    enhanced_element["refinement"] = "ai_only"
                
                enhanced_text_elements.append(enhanced_element)
            
            # Update layout data
            enhanced_layout = layout_data.copy()
            enhanced_layout["text"] = enhanced_text_elements
            enhanced_layout["cv_enhancement"] = {
                "cv_boxes_detected": len(cv_boxes),
                "ai_elements_processed": len(ai_text_elements),
                "enhanced_elements": len([e for e in enhanced_text_elements if e.get("cv_confidence", 0) > 0.5])
            }
            
            return enhanced_layout
            
        except Exception as e:
            logger.error(f"CV enhancement failed: {e}")
            return layout_data
    
    def _find_best_bbox_match(self, ai_bbox: List[float], 
                             cv_boxes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the best matching CV detection for an AI-detected bbox.
        
        Args:
            ai_bbox: AI-detected bbox [x, y, width, height]
            cv_boxes: List of CV-detected boxes
            
        Returns:
            Best matching CV box or None
        """
        if not cv_boxes:
            return None
        
        best_match = None
        best_overlap = 0.0
        
        for cv_box in cv_boxes:
            cv_bbox = cv_box.get("bbox", [])
            if len(cv_bbox) != 4:
                continue
            
            # Calculate intersection over union (IoU)
            overlap = self._calculate_bbox_overlap(ai_bbox, cv_bbox)
            
            if overlap > best_overlap and overlap > 0.3:  # Minimum 30% overlap
                best_overlap = overlap
                best_match = cv_box
        
        return best_match
    
    def _calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate intersection over union (IoU) between two bboxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def validate_and_refine_bbox_coordinates(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and refine bbox coordinates for accuracy and consistency.
        
        Args:
            layout_data: Layout data with text elements
            
        Returns:
            Refined layout data with validated coordinates
        """
        try:
            if not self.enable_bbox_refinement:
                return layout_data
            
            # Get image dimensions
            image_info = layout_data.get("image_info", {})
            img_width = image_info.get("original_width", 1024)
            img_height = image_info.get("original_height", 1024)
            
            text_elements = layout_data.get("text", [])
            refined_elements = []
            
            for i, element in enumerate(text_elements):
                bbox = element.get("bbox", [0, 0, 100, 50])
                
                # Validate bbox proportions
                is_valid = self.coordinate_normalizer.validate_bbox_proportions(
                    bbox, img_width, img_height
                )
                
                refined_element = element.copy()
                
                if not is_valid:
                    # Apply corrections
                    corrected_bbox = self._correct_bbox_coordinates(bbox, img_width, img_height)
                    refined_element["bbox"] = corrected_bbox
                    refined_element["bbox_corrected"] = True
                    logger.warning(f"Corrected invalid bbox {i}: {bbox} -> {corrected_bbox}")
                else:
                    refined_element["bbox_corrected"] = False
                
                # Add validation metadata
                refined_element["bbox_validation"] = {
                    "is_valid": is_valid,
                    "within_bounds": self._check_bbox_bounds(refined_element["bbox"], img_width, img_height),
                    "min_size_met": self._check_min_text_size(refined_element["bbox"])
                }
                
                refined_elements.append(refined_element)
            
            # Update layout data
            refined_layout = layout_data.copy()
            refined_layout["text"] = refined_elements
            refined_layout["bbox_refinement"] = {
                "total_elements": len(text_elements),
                "corrected_elements": len([e for e in refined_elements if e.get("bbox_corrected", False)]),
                "validation_applied": True
            }
            
            return refined_layout
            
        except Exception as e:
            logger.error(f"Bbox refinement failed: {e}")
            return layout_data
    
    def _correct_bbox_coordinates(self, bbox: List[float], 
                                 img_width: int, img_height: int) -> List[float]:
        """Correct invalid bbox coordinates."""
        x, y, w, h = bbox
        
        # Ensure non-negative coordinates
        x = max(0, x)
        y = max(0, y)
        
        # Ensure minimum size
        w = max(10, w)  # Minimum text width
        h = max(8, h)   # Minimum text height
        
        # Ensure bbox fits within image
        if x + w > img_width:
            if w <= img_width:
                x = img_width - w
            else:
                x = 0
                w = img_width
        
        if y + h > img_height:
            if h <= img_height:
                y = img_height - h
            else:
                y = 0
                h = img_height
        
        return [float(x), float(y), float(w), float(h)]
    
    def _check_bbox_bounds(self, bbox: List[float], img_width: int, img_height: int) -> bool:
        """Check if bbox is within image bounds."""
        x, y, w, h = bbox
        return (x >= 0 and y >= 0 and 
                x + w <= img_width and y + h <= img_height)
    
    def _check_min_text_size(self, bbox: List[float]) -> bool:
        """Check if bbox meets minimum text size requirements."""
        _, _, w, h = bbox
        return w >= 10 and h >= 8
    
    def calculate_image_hash(self, image_data: str) -> str:
        """Calculate MD5 hash of image data for deduplication."""
        return hashlib.md5(image_data.encode()).hexdigest()
    
    def get_image_dimensions(self, image_data: str) -> Tuple[int, int]:
        """
        Get original image dimensions from base64 encoded image data.
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            Tuple of (width, height) in pixels
        """
        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return image.size  # Returns (width, height)
        except Exception as e:
            logger.warning(f"Could not determine image dimensions: {e}")
            return (1024, 1024)  # Default dimensions
    
    def check_existing_extraction(self, image_hash: str) -> Optional[LayoutExtraction]:
        """Check if we already have an extraction for this image."""
        return self.db.query(LayoutExtraction).filter(
            LayoutExtraction.original_image_hash == image_hash
        ).first()
    
    def extract_layout(self, image_data: str) -> BannerLayoutResponse:
        """
        Main service method to extract layout from image.
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            BannerLayoutResponse: Service response with layout data or error
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not isinstance(image_data, str) or not image_data.strip():
                return BannerLayoutResponse(
                    success=False,
                    error="Invalid or empty image data provided",
                    processing_time=time.time() - start_time
                )
            
            # Check for existing extraction (deduplication)
            image_hash = self.calculate_image_hash(image_data)
            existing_extraction = self.check_existing_extraction(image_hash)
            
            if existing_extraction:
                logger.info(f"Using cached extraction for image hash: {image_hash}")
                layout_data_dict = {
                    "text": existing_extraction.text_elements,
                    "image": existing_extraction.background_data  # Map old background_data to new image field
                }
                layout_data = BannerLayoutData(**layout_data_dict)
                
                return BannerLayoutResponse(
                    success=True,
                    layout_data=layout_data,
                    processing_time=0.1  # Cached result
                )
            
            # Extract layout information from OpenAI
            raw_layout_data = self.extract_layout_from_image(image_data)
            
            # Validate and normalize the data
            validated_layout_data = self.validate_and_normalize_layout_data(raw_layout_data)
            
            if "error" in validated_layout_data:
                return BannerLayoutResponse(
                    success=False,
                    error=validated_layout_data["error"],
                    processing_time=time.time() - start_time
                )
            
            # Enhanced bbox processing
            enhanced_layout_data = validated_layout_data
            
            # Step 1: Apply CV enhancement if enabled
            if self.use_cv_validation:
                enhanced_layout_data = self.enhance_bbox_coordinates_with_cv(
                    enhanced_layout_data, image_data
                )
                logger.info("Applied CV enhancement to bbox coordinates")
            
            # Step 2: Apply bbox refinement and validation
            if self.enable_bbox_refinement:
                enhanced_layout_data = self.validate_and_refine_bbox_coordinates(
                    enhanced_layout_data
                )
                logger.info("Applied bbox coordinate refinement and validation")
            
            # Convert to Pydantic model for validation
            try:
                layout_data = BannerLayoutData(**enhanced_layout_data)
            except Exception as e:
                logger.error(f"Pydantic validation failed: {e}")
                return BannerLayoutResponse(
                    success=False,
                    error=f"Data validation failed: {str(e)}",
                    processing_time=time.time() - start_time
                )
            
            processing_time = time.time() - start_time
            
            return BannerLayoutResponse(
                success=True,
                layout_data=layout_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Layout extraction failed: {e}")
            return BannerLayoutResponse(
                success=False,
                error=f"Layout extraction failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def extract_layout_for_job(self, job_id: str, image_data: str) -> BannerLayoutResponse:
        """
        Extract layout for a specific job with database tracking.
        
        Args:
            job_id: Banner job ID
            image_data: Base64 encoded image data
            
        Returns:
            BannerLayoutResponse: Service response with layout data or error
        """
        start_time = time.time()
        
        # Update step status to in_progress
        update_step_status(
            self.db, job_id, 1, StepStatus.IN_PROGRESS,
            input_data={"image_data_length": len(image_data)},
            api_model=self.model
        )
        
        try:
            # Perform layout extraction
            response = self.extract_layout(image_data)
            
            if response.success:
                # Store extraction results in database
                image_hash = self.calculate_image_hash(image_data)
                
                # Prepare image data with image_info included
                image_data_with_info = response.layout_data.image.dict()
                # Add image_info to the image data for easy retrieval
                if response.layout_data.image_info:
                    image_data_with_info["image_info"] = response.layout_data.image_info.dict()
                
                extraction = LayoutExtraction(
                    job_id=job_id,
                    original_image_hash=image_hash,
                    text_elements=[text_elem.dict() for text_elem in response.layout_data.text] if response.layout_data.text else [],
                    background_data=image_data_with_info,  # Still storing as background_data in DB for backward compatibility
                    extraction_model=self.model,
                    processing_time=response.processing_time
                )
                
                self.db.add(extraction)
                self.db.commit()
                
                # Update step status to completed
                output_data = {
                    "extraction_id": extraction.id,
                    "text_elements_count": len(response.layout_data.text) if response.layout_data and response.layout_data.text else 0,
                    "image_data": response.layout_data.image.dict() if response.layout_data and response.layout_data.image else {}
                }
                
                update_step_status(
                    self.db, job_id, 1, StepStatus.COMPLETED,
                    output_data=output_data,
                    processing_time=response.processing_time
                )
                
                logger.info(f"Layout extraction completed for job {job_id}")
                
            else:
                # Update step status to failed
                update_step_status(
                    self.db, job_id, 1, StepStatus.FAILED,
                    error_message=response.error or "Unknown error",
                    processing_time=response.processing_time
                )
                
                # Update job status to failed
                update_job_status(
                    self.db, job_id, JobStatus.FAILED,
                    error_message=f"Layout extraction failed: {response.error}"
                )
                
                logger.error(f"Layout extraction failed for job {job_id}: {response.error}")
            
            return response
            
        except Exception as e:
            error_msg = f"Layout extraction service error: {str(e)}"
            logger.error(error_msg)
            
            # Update step and job status to failed
            update_step_status(
                self.db, job_id, 1, StepStatus.FAILED,
                error_message=error_msg,
                processing_time=time.time() - start_time
            )
            
            update_job_status(
                self.db, job_id, JobStatus.FAILED,
                error_message=error_msg
            )
            
            return BannerLayoutResponse(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )