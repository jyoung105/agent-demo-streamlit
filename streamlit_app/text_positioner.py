"""
Advanced text positioning system for banner generation.
Handles precise text overlay placement using extracted bbox coordinates.
"""
import base64
import io
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import json
import os

from coordinate_normalizer import CoordinateNormalizer

logger = logging.getLogger(__name__)


class TextPositioner:
    """
    Advanced text positioning system that handles precise text overlay placement
    using extracted bbox coordinates with support for multiple font styles and layouts.
    """
    
    def __init__(self):
        self.normalizer = CoordinateNormalizer()
        
        # Font configuration
        self.font_cache = {}
        self.default_font_paths = {
            "regular": self._get_system_font("arial.ttf", "DejaVuSans.ttf"),
            "bold": self._get_system_font("arialbd.ttf", "DejaVuSans-Bold.ttf"),
            "italic": self._get_system_font("ariali.ttf", "DejaVuSans-Oblique.ttf")
        }
        
        # Text styling options
        self.text_effects = {
            "drop_shadow": True,
            "outline": True,
            "anti_aliasing": True
        }
        
        # Layout constraints
        self.min_padding = 5
        self.line_spacing_ratio = 1.2
        self.word_spacing_ratio = 0.3
    
    def _get_system_font(self, *font_names: str) -> Optional[str]:
        """Find available system font from a list of candidates."""
        system_font_paths = [
            "/System/Library/Fonts/",  # macOS
            "/Windows/Fonts/",         # Windows
            "/usr/share/fonts/",       # Linux
            "/usr/local/share/fonts/"  # Linux alternative
        ]
        
        for font_name in font_names:
            for font_path in system_font_paths:
                full_path = os.path.join(font_path, font_name)
                if os.path.exists(full_path):
                    return full_path
        
        return None  # Use PIL default font
    
    def _get_font(self, font_size: int, font_style: str = "regular") -> ImageFont.ImageFont:
        """Get or create cached font object."""
        cache_key = f"{font_style}_{font_size}"
        
        if cache_key not in self.font_cache:
            font_path = self.default_font_paths.get(font_style.lower())
            
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    # Fallback to default font
                    font = ImageFont.load_default()
                    logger.warning(f"Using default font for {font_style} {font_size}px")
                
                self.font_cache[cache_key] = font
            except Exception as e:
                logger.error(f"Font loading failed: {e}")
                self.font_cache[cache_key] = ImageFont.load_default()
        
        return self.font_cache[cache_key]
    
    def calculate_text_dimensions(self, text: str, font_size: int, 
                                 font_style: str = "regular") -> Tuple[int, int]:
        """Calculate actual text dimensions for given font settings."""
        font = self._get_font(font_size, font_style)
        
        # Create temporary image to measure text
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return width, height
    
    def fit_text_to_bbox(self, text: str, bbox: List[float], 
                        font_style: str = "regular", 
                        max_lines: int = 3) -> Dict[str, Any]:
        """
        Calculate optimal font size and layout to fit text within bbox.
        
        Args:
            text: Text content to fit
            bbox: [x, y, width, height] bounding box
            font_style: Font style to use
            max_lines: Maximum number of lines allowed
            
        Returns:
            Dictionary with fitted text parameters
        """
        x, y, width, height = bbox
        available_width = width - (2 * self.min_padding)
        available_height = height - (2 * self.min_padding)
        
        # Start with estimated font size
        estimated_font_size = self.normalizer.calculate_optimal_text_size(
            text, bbox, font_style
        )
        
        best_fit = None
        
        # Try different font sizes to find best fit
        for font_size in range(estimated_font_size, 8, -2):  # Decrease from estimate
            font = self._get_font(font_size, font_style)
            
            # Try to fit text with word wrapping
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = " ".join(current_line + [word])
                text_width, text_height = self.calculate_text_dimensions(
                    test_line, font_size, font_style
                )
                
                if text_width <= available_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                    else:
                        # Single word too long, force break
                        lines.append(word)
                        current_line = []
            
            if current_line:
                lines.append(" ".join(current_line))
            
            # Check if all lines fit
            if len(lines) <= max_lines:
                total_height = len(lines) * font_size * self.line_spacing_ratio
                
                if total_height <= available_height:
                    best_fit = {
                        "font_size": font_size,
                        "lines": lines,
                        "total_width": max(self.calculate_text_dimensions(
                            line, font_size, font_style)[0] for line in lines),
                        "total_height": total_height,
                        "line_height": font_size * self.line_spacing_ratio,
                        "fits": True
                    }
                    break
        
        if not best_fit:
            # Fallback: use minimum font size
            best_fit = {
                "font_size": 12,
                "lines": [text[:30] + "..." if len(text) > 30 else text],
                "total_width": available_width,
                "total_height": 12 * self.line_spacing_ratio,
                "line_height": 12 * self.line_spacing_ratio,
                "fits": False
            }
        
        return best_fit
    
    def apply_text_effects(self, draw: ImageDraw.Draw, text: str, 
                          position: Tuple[int, int], font: ImageFont.ImageFont,
                          color: str, effects: Dict[str, Any]) -> None:
        """Apply visual effects to text (shadow, outline, etc.)."""
        x, y = position
        
        # Apply drop shadow
        if effects.get("drop_shadow", False):
            shadow_offset = max(1, font.size // 20)
            shadow_color = "#00000088"  # Semi-transparent black
            draw.text((x + shadow_offset, y + shadow_offset), text, 
                     font=font, fill=shadow_color)
        
        # Apply outline
        if effects.get("outline", False):
            outline_width = max(1, font.size // 30)
            outline_color = "#000000" if color.upper() in ["#FFFFFF", "#FFF", "WHITE"] else "#FFFFFF"
            
            # Draw outline by drawing text in multiple positions
            for adj_x in range(-outline_width, outline_width + 1):
                for adj_y in range(-outline_width, outline_width + 1):
                    if adj_x != 0 or adj_y != 0:
                        draw.text((x + adj_x, y + adj_y), text, 
                                 font=font, fill=outline_color)
        
        # Draw main text
        draw.text(position, text, font=font, fill=color)
    
    def position_text_elements(self, image_data: str, text_elements: List[Dict[str, Any]],
                              target_size: str = "1024x1024") -> Dict[str, Any]:
        """
        Position all text elements on the image using precise bbox coordinates.
        
        Args:
            image_data: Base64 encoded background image
            text_elements: List of text elements with bbox coordinates
            target_size: Target banner size
            
        Returns:
            Dictionary with positioning results and final image
        """
        try:
            # Decode background image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize to target size if needed
            target_width, target_height = self.normalizer.supported_sizes[target_size]
            if image.size != (target_width, target_height):
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Create drawing context
            draw = ImageDraw.Draw(image)
            
            positioned_elements = []
            
            for i, element in enumerate(text_elements):
                try:
                    # Extract element properties
                    bbox = element.get("bbox", [0, 0, 100, 50])
                    font_size = element.get("font_size", 24)
                    font_style = element.get("font_style", "regular")
                    font_color = element.get("font_color", "#000000")
                    description = element.get("description", f"Text element {i+1}")
                    
                    # Generate actual text content (in real implementation, this might come from user input)
                    text_content = self._extract_text_content(description)
                    
                    # Validate bbox coordinates
                    if not self.normalizer.validate_bbox_proportions(bbox, target_width, target_height):
                        logger.warning(f"Invalid bbox for element {i}: {bbox}")
                        continue
                    
                    # Fit text to bbox
                    text_fit = self.fit_text_to_bbox(text_content, bbox, font_style)
                    
                    if not text_fit["fits"]:
                        logger.warning(f"Text doesn't fit in bbox for element {i}")
                    
                    # Position each line of text
                    x, y, width, height = bbox
                    start_y = y + self.min_padding
                    
                    for line_idx, line in enumerate(text_fit["lines"]):
                        # Calculate line position
                        line_y = start_y + (line_idx * text_fit["line_height"])
                        line_x = x + self.min_padding
                        
                        # Center text horizontally in bbox
                        line_width = self.calculate_text_dimensions(
                            line, text_fit["font_size"], font_style
                        )[0]
                        centered_x = x + (width - line_width) // 2
                        
                        # Get font for this line
                        font = self._get_font(text_fit["font_size"], font_style)
                        
                        # Apply text effects
                        effects = {
                            "drop_shadow": len(text_fit["lines"]) == 1,  # Shadow for single line
                            "outline": font_color.upper() in ["#FFFFFF", "#FFF", "WHITE"]
                        }
                        
                        self.apply_text_effects(
                            draw, line, (centered_x, int(line_y)), 
                            font, font_color, effects
                        )
                    
                    # Record positioned element
                    positioned_elements.append({
                        "original_element": element,
                        "text_content": text_content,
                        "final_bbox": bbox,
                        "font_settings": {
                            "size": text_fit["font_size"],
                            "style": font_style,
                            "color": font_color
                        },
                        "layout": {
                            "lines": text_fit["lines"],
                            "line_height": text_fit["line_height"],
                            "total_height": text_fit["total_height"]
                        },
                        "positioning_success": text_fit["fits"]
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to position text element {i}: {e}")
                    continue
            
            # Convert final image to base64
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG', quality=95)
            output_buffer.seek(0)
            final_image_b64 = base64.b64encode(output_buffer.getvalue()).decode()
            
            return {
                "success": True,
                "final_image_data": final_image_b64,
                "positioned_elements": positioned_elements,
                "target_size": target_size,
                "statistics": {
                    "total_elements": len(text_elements),
                    "successfully_positioned": len(positioned_elements),
                    "elements_that_fit": sum(1 for e in positioned_elements if e["positioning_success"])
                }
            }
            
        except Exception as e:
            logger.error(f"Text positioning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "positioned_elements": []
            }
    
    def _extract_text_content(self, description: str) -> str:
        """
        Extract or generate text content from description.
        In a full implementation, this might use AI to generate appropriate text.
        """
        # Simple implementation: use description as-is or generate placeholder
        if len(description) > 50:
            # Use first part of description
            return description[:50].strip()
        
        # Generate sample text based on description
        description_lower = description.lower()
        
        if "title" in description_lower or "heading" in description_lower:
            return "Sample Title"
        elif "subtitle" in description_lower:
            return "Sample Subtitle"
        elif "button" in description_lower:
            return "Click Here"
        elif "logo" in description_lower:
            return "LOGO"
        else:
            return description if description else "Sample Text"
    
    def create_text_positioning_preview(self, image_data: str, 
                                       text_elements: List[Dict[str, Any]],
                                       target_size: str = "1024x1024") -> Optional[str]:
        """
        Create a preview showing text positioning without final rendering.
        
        Args:
            image_data: Base64 encoded background image
            text_elements: List of text elements
            target_size: Target banner size
            
        Returns:
            Base64 encoded preview image or None if failed
        """
        try:
            # Decode background image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize to target size
            target_width, target_height = self.normalizer.supported_sizes[target_size]
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Create drawing context
            draw = ImageDraw.Draw(image)
            
            # Draw bbox outlines and text placeholders
            for i, element in enumerate(text_elements):
                bbox = element.get("bbox", [0, 0, 100, 50])
                x, y, width, height = bbox
                
                # Draw bbox outline
                draw.rectangle([x, y, x + width, y + height], 
                              outline="red", width=2)
                
                # Draw label
                label = f"Text {i+1}"
                draw.text((x + 5, y + 5), label, fill="red")
            
            # Convert to base64
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG')
            output_buffer.seek(0)
            
            return base64.b64encode(output_buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Preview creation failed: {e}")
            return None
    
    def validate_text_positioning(self, positioning_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the quality of text positioning results.
        
        Args:
            positioning_result: Result from position_text_elements
            
        Returns:
            Validation report with quality metrics
        """
        if not positioning_result.get("success"):
            return {
                "overall_quality": "failed",
                "issues": ["Positioning process failed"],
                "recommendations": ["Check input data and try again"]
            }
        
        positioned_elements = positioning_result.get("positioned_elements", [])
        stats = positioning_result.get("statistics", {})
        
        issues = []
        recommendations = []
        
        # Check positioning success rate
        success_rate = (stats.get("elements_that_fit", 0) / 
                       max(1, stats.get("total_elements", 1)))
        
        if success_rate < 0.5:
            issues.append("Low text fitting success rate")
            recommendations.append("Consider increasing bbox sizes or reducing text content")
        
        # Check for overlapping text
        overlaps = self._check_text_overlaps(positioned_elements)
        if overlaps:
            issues.append(f"Found {len(overlaps)} overlapping text elements")
            recommendations.append("Adjust bbox positions to avoid overlaps")
        
        # Determine overall quality
        if not issues:
            quality = "excellent"
        elif len(issues) == 1:
            quality = "good"
        elif len(issues) <= 2:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "overall_quality": quality,
            "success_rate": success_rate,
            "issues": issues,
            "recommendations": recommendations,
            "detailed_metrics": {
                "total_elements": stats.get("total_elements", 0),
                "positioned_elements": stats.get("successfully_positioned", 0),
                "elements_that_fit": stats.get("elements_that_fit", 0),
                "overlapping_elements": len(overlaps) if overlaps else 0
            }
        }
    
    def _check_text_overlaps(self, positioned_elements: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Check for overlapping text elements and return pairs of overlapping indices."""
        overlaps = []
        
        for i, elem1 in enumerate(positioned_elements):
            bbox1 = elem1["final_bbox"]
            for j, elem2 in enumerate(positioned_elements[i+1:], i+1):
                bbox2 = elem2["final_bbox"]
                
                # Check if bboxes overlap
                x1, y1, w1, h1 = bbox1
                x2, y2, w2, h2 = bbox2
                
                if (x1 < x2 + w2 and x1 + w1 > x2 and 
                    y1 < y2 + h2 and y1 + h1 > y2):
                    overlaps.append((i, j))
        
        return overlaps


def main():
    """Test the text positioner."""
    positioner = TextPositioner()
    
    # Test text fitting
    test_bbox = [100, 50, 300, 80]
    test_text = "This is a sample banner title that needs to fit"
    
    fit_result = positioner.fit_text_to_bbox(test_text, test_bbox, "bold")
    print(f"Text fit result: {fit_result}")
    
    # Test text dimensions calculation
    width, height = positioner.calculate_text_dimensions("Sample Text", 24, "regular")
    print(f"Text dimensions: {width}x{height}")


if __name__ == "__main__":
    main()