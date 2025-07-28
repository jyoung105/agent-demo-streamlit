"""
Enhanced banner text overlay service with PIL/Pillow.
Handles the 4th step of banner generation: overlaying text on background images.
"""
import time
import logging
import os
import base64
import io
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session
from shared.banner import BannerLayoutData, TextElement
from database import (
    TextOverlay, ImageGeneration, update_step_status,
    StepStatus, JobStatus, update_job_status
)

logger = logging.getLogger(__name__)


class BannerTextOverlayService:
    def __init__(self, db_session: Session):
        self.db = db_session
        # Default font settings
        self.default_font_size = 72
        self.default_font_color = "#000000"
        self.default_font_style = "Arial"
        
        # Font paths for different platforms
        self.font_paths = self._get_system_font_paths()
    
    def _get_system_font_paths(self) -> Dict[str, str]:
        """Get system font paths for different operating systems."""
        import platform
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            base_path = "/System/Library/Fonts/"
            return {
                "Arial": f"{base_path}Supplemental/Arial.ttf",
                "Arial Bold": f"{base_path}Supplemental/Arial Bold.ttf",
                "Helvetica": f"{base_path}Helvetica.ttc",
                "Times": f"{base_path}Times.ttc",
                "Courier": f"{base_path}Courier New.ttf",
                "AppleGothic": "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # Korean font
                "default": "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # Use Korean font as default
            }
        elif system == "linux":
            return {
                "Arial": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "Arial Bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "Helvetica": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "Times": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                "Courier": "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "default": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            }
        else:  # Windows
            return {
                "Arial": "C:/Windows/Fonts/arial.ttf",
                "Arial Bold": "C:/Windows/Fonts/arialbd.ttf",
                "Helvetica": "C:/Windows/Fonts/arial.ttf",
                "Times": "C:/Windows/Fonts/times.ttf",
                "Courier": "C:/Windows/Fonts/cour.ttf",
                "default": "C:/Windows/Fonts/arial.ttf"
            }
    
    def _load_font(self, font_style: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Load a font with the specified style and size."""
        try:
            # Try to get the specific font
            font_path = self.font_paths.get(font_style, self.font_paths.get("default"))
            
            if font_path and os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
            else:
                logger.warning(f"Font {font_style} not found, using default")
                return ImageFont.load_default()
                
        except Exception as e:
            logger.warning(f"Error loading font {font_style}: {e}, using default")
            return ImageFont.load_default()
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int, int]:
        """Parse color string to RGBA tuple."""
        try:
            # Remove # if present
            color_str = color_str.lstrip('#')
            
            # Handle different color formats
            if len(color_str) == 6:  # RGB
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                return (r, g, b, 255)
            elif len(color_str) == 8:  # RGBA
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                a = int(color_str[6:8], 16)
                return (r, g, b, a)
            else:
                # Default to black
                return (0, 0, 0, 255)
                
        except Exception as e:
            logger.warning(f"Error parsing color {color_str}: {e}, using black")
            return (0, 0, 0, 255)
    
    def _calculate_font_size_for_width(self, text: str, font_style: str, max_width: int, 
                                        max_font_size: int = 120, min_font_size: int = 20) -> int:
        """Calculate optimal font size to fit text within width."""
        for size in range(max_font_size, min_font_size, -2):
            font = self._load_font(font_style, size)
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            if text_width <= max_width:
                return size
        return min_font_size
    
    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int, force_single_line: bool = False) -> List[str]:
        """Wrap text to fit within the specified width."""
        if force_single_line:
            # Return the entire text as a single line
            return [text]
            
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Test if adding this word would exceed the width
            test_line = ' '.join(current_line + [word])
            
            # Get text bounding box
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, add it anyway
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _draw_text_with_outline(self, draw: ImageDraw.Draw, position: Tuple[int, int], 
                               text: str, font: ImageFont.FreeTypeFont, 
                               fill_color: Tuple[int, int, int, int],
                               outline_color: Tuple[int, int, int, int] = (255, 255, 255, 128),
                               outline_width: int = 2):
        """Draw text with an outline for better visibility."""
        x, y = position
        
        # Draw outline
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                if adj_x != 0 or adj_y != 0:
                    draw.text((x + adj_x, y + adj_y), text, font=font, fill=outline_color)
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=fill_color)
    
    def overlay_text_on_image(self, background_image_data: str, text_elements: List[TextElement], 
                              original_width: Optional[int] = None, original_height: Optional[int] = None,
                              text_alignment: str = "top") -> str:
        """
        Overlay text elements on a background image with coordinate scaling.
        
        Args:
            background_image_data: Base64 encoded background image
            text_elements: List of text elements with positioning and styling
            original_width: Original image width (for scaling)
            original_height: Original image height (for scaling)
            text_alignment: Vertical alignment within bbox ("top", "center", "bottom")
            
        Returns:
            str: Base64 encoded image with text overlays
        """
        try:
            # Decode the background image
            image_bytes = base64.b64decode(background_image_data)
            background_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            
            # Get banner dimensions
            banner_width, banner_height = background_image.size
            
            # Calculate scaling ratios if original dimensions are provided
            scale_x = 1.0
            scale_y = 1.0
            if original_width and original_height:
                scale_x = banner_width / original_width
                scale_y = banner_height / original_height
                logger.info(f"Scaling coordinates: original {original_width}x{original_height} -> banner {banner_width}x{banner_height}")
                logger.info(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
            
            # Create a new image for text overlay
            text_overlay = Image.new("RGBA", background_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_overlay)
            
            # Process each text element
            for text_elem in text_elements:
                if not text_elem.description or not text_elem.bbox:
                    continue
                
                # Extract positioning and styling
                x, y, width, height = text_elem.bbox
                
                # Apply coordinate scaling
                x = int(x * scale_x)
                y = int(y * scale_y)
                width = int(width * scale_x)
                height = int(height * scale_y)
                
                # Scale font size proportionally
                font_size = text_elem.font_size or self.default_font_size
                # Apply more aggressive scaling for better visibility
                font_size = int(font_size * max(scale_x, scale_y) * 1.2)  # Use maximum scale and boost by 20%
                
                font_style = text_elem.font_style or self.default_font_style
                font_color = text_elem.font_color or self.default_font_color
                
                logger.debug(f"Text element: '{text_elem.description[:30]}...' at scaled position ({x}, {y}) with size {width}x{height}")
                
                # Load font
                font = self._load_font(font_style, font_size)
                
                # Parse color
                color_rgba = self._parse_color(font_color)
                
                # Wrap text to fit within the bounding box
                wrapped_lines = self._wrap_text(text_elem.description, font, width)
                
                # Calculate line height
                bbox = font.getbbox("Ay")
                line_height = bbox[3] - bbox[1] + 4  # Add some line spacing
                
                # Calculate total text height
                total_text_height = len(wrapped_lines) * line_height
                
                # Apply text alignment strategy
                if text_alignment == "center":
                    # Center vertically within the bounding box
                    start_y = y + (height - total_text_height) // 2
                elif text_alignment == "bottom":
                    # Align to bottom of bounding box
                    start_y = y + height - total_text_height
                else:  # Default to "top"
                    # Use the y coordinate as the top position
                    start_y = y
                
                # Draw each line
                for i, line in enumerate(wrapped_lines):
                    line_y = start_y + i * line_height
                    
                    # Get line width for horizontal centering
                    line_bbox = font.getbbox(line)
                    line_width = line_bbox[2] - line_bbox[0]
                    
                    # Center horizontally within the bounding box
                    line_x = x + (width - line_width) // 2
                    
                    # Draw text with outline for better visibility
                    self._draw_text_with_outline(
                        draw, (line_x, line_y), line, font, color_rgba
                    )
            
            # Composite the text overlay onto the background
            final_image = Image.alpha_composite(background_image, text_overlay)
            
            # Convert back to RGB for JPEG compatibility
            final_image = final_image.convert("RGB")
            
            # Save to bytes
            output_buffer = io.BytesIO()
            final_image.save(output_buffer, format="PNG", quality=95)
            output_bytes = output_buffer.getvalue()
            
            # Encode to base64
            output_base64 = base64.b64encode(output_bytes).decode('utf-8')
            
            return output_base64
            
        except Exception as e:
            logger.error(f"Error overlaying text on image: {e}")
            raise
    
    def overlay_text_for_job(self, job_id: str, background_image_data: str, 
                           text_elements: List[TextElement], original_width: Optional[int] = None,
                           original_height: Optional[int] = None) -> Dict[str, Any]:
        """
        Overlay text on background image for a specific job with database tracking.
        
        Args:
            job_id: Banner job ID
            background_image_data: Base64 encoded background image
            text_elements: List of text elements to overlay
            original_width: Original image width (for scaling)
            original_height: Original image height (for scaling)
            
        Returns:
            Dict with success status and final image data
        """
        start_time = time.time()
        
        # Update step status to in_progress
        update_step_status(
            self.db, job_id, 4, StepStatus.IN_PROGRESS,
            input_data={
                "text_elements_count": len(text_elements),
                "background_image_size": len(background_image_data) if background_image_data else 0,
                "original_dimensions": f"{original_width}x{original_height}" if original_width and original_height else None
            }
        )
        
        try:
            # Get image generation record from database
            image_generation = self.db.query(ImageGeneration).filter(
                ImageGeneration.job_id == job_id
            ).first()
            
            if not image_generation:
                raise Exception("Background image generation not found")
            
            # Get layout extraction to find original dimensions if not provided
            if not original_width or not original_height:
                from database import LayoutExtraction
                layout_extraction = self.db.query(LayoutExtraction).filter(
                    LayoutExtraction.job_id == job_id
                ).first()
                
                if layout_extraction:
                    # The image_info should be stored at the root level of the background_data
                    if layout_extraction.background_data and isinstance(layout_extraction.background_data, dict):
                        # Check if image_info is stored separately or within background_data
                        image_info = layout_extraction.background_data.get("image_info")
                        if image_info:
                            original_width = image_info.get("original_width")
                            original_height = image_info.get("original_height")
                            logger.info(f"Found dimensions in background_data.image_info: {original_width}x{original_height}")
                
                # If still not found, check in the prompt generation's optimized scene
                if not original_width or not original_height:
                    from database import PromptGeneration
                    prompt_gen = self.db.query(PromptGeneration).filter(
                        PromptGeneration.job_id == job_id
                    ).first()
                    
                    if prompt_gen and prompt_gen.optimized_scene:
                        image_info = prompt_gen.optimized_scene.get("image_info")
                        if image_info:
                            original_width = image_info.get("original_width")
                            original_height = image_info.get("original_height")
                            logger.info(f"Found dimensions in prompt generation: {original_width}x{original_height}")
                
                if original_width and original_height:
                    logger.info(f"Retrieved original dimensions: {original_width}x{original_height}")
                else:
                    logger.warning("Could not retrieve original dimensions from database")
            
            # Perform text overlay with scaling and top alignment
            final_image_data = self.overlay_text_on_image(
                background_image_data, text_elements, original_width, original_height,
                text_alignment="top"
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Store text overlay results in database
            text_overlay = TextOverlay(
                job_id=job_id,
                image_generation_id=image_generation.id,
                text_elements=[elem.dict() for elem in text_elements],
                final_image_data=final_image_data,
                processing_time=processing_time
            )
            
            self.db.add(text_overlay)
            self.db.commit()
            
            # Update step status to completed
            output_data = {
                "text_overlay_id": text_overlay.id,
                "final_image_size": len(final_image_data),
                "text_elements_overlaid": len(text_elements),
                "coordinate_scaling_applied": bool(original_width and original_height)
            }
            
            update_step_status(
                self.db, job_id, 4, StepStatus.COMPLETED,
                output_data=output_data,
                processing_time=processing_time
            )
            
            # Update job with final image data and mark as completed
            from database import BannerJob
            job = self.db.query(BannerJob).filter(BannerJob.id == job_id).first()
            if job:
                job.final_image_data = final_image_data
                job.status = JobStatus.COMPLETED
                self.db.commit()
            
            logger.info(f"Text overlay completed for job {job_id}")
            
            return {
                "success": True,
                "final_image_data": final_image_data,
                "processing_time": processing_time,
                "text_elements_count": len(text_elements),
                "coordinate_scaling_applied": bool(original_width and original_height)
            }
            
        except Exception as e:
            error_msg = f"Text overlay service error: {str(e)}"
            logger.error(error_msg)
            
            processing_time = time.time() - start_time
            
            # Update step and job status to failed
            update_step_status(
                self.db, job_id, 4, StepStatus.FAILED,
                error_message=error_msg,
                processing_time=processing_time
            )
            
            update_job_status(
                self.db, job_id, JobStatus.FAILED,
                error_message=error_msg
            )
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }