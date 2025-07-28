"""
Coordinate normalization system for banner generation.
Converts between absolute pixel coordinates and relative coordinates for different banner sizes.
"""
import logging
from typing import Dict, List, Tuple, Any, Optional
import math

logger = logging.getLogger(__name__)


class CoordinateNormalizer:
    """
    Handles coordinate system conversions between different banner sizes and formats.
    Supports absolute pixel coordinates, relative coordinates (0.0-1.0), and percentage-based scaling.
    """
    
    def __init__(self):
        # Standard banner sizes supported
        self.supported_sizes = {
            "1024x1024": (1024, 1024),
            "1536x1024": (1536, 1024), 
            "1024x1536": (1024, 1536)
        }
        
        # Minimum text size constraints
        self.min_font_size = 8
        self.max_font_size = 200
        self.min_text_width = 10
        self.min_text_height = 8
    
    def pixel_to_relative(self, bbox: List[float], 
                         source_width: int, source_height: int) -> List[float]:
        """
        Convert absolute pixel coordinates to relative coordinates (0.0-1.0).
        
        Args:
            bbox: [x, y, width, height] in absolute pixels
            source_width: Original image width in pixels
            source_height: Original image height in pixels
            
        Returns:
            [x_rel, y_rel, width_rel, height_rel] in relative coordinates
        """
        if source_width <= 0 or source_height <= 0:
            logger.error(f"Invalid source dimensions: {source_width}x{source_height}")
            return [0.0, 0.0, 0.1, 0.1]  # Safe fallback
        
        x, y, width, height = bbox
        
        # Convert to relative coordinates
        x_rel = x / source_width
        y_rel = y / source_height  
        width_rel = width / source_width
        height_rel = height / source_height
        
        # Clamp to valid range [0.0, 1.0]
        x_rel = max(0.0, min(1.0, x_rel))
        y_rel = max(0.0, min(1.0, y_rel))
        width_rel = max(0.001, min(1.0 - x_rel, width_rel))  # Ensure it fits in image
        height_rel = max(0.001, min(1.0 - y_rel, height_rel))
        
        return [x_rel, y_rel, width_rel, height_rel]
    
    def relative_to_pixel(self, bbox_rel: List[float], 
                         target_width: int, target_height: int) -> List[float]:
        """
        Convert relative coordinates to absolute pixel coordinates.
        
        Args:
            bbox_rel: [x_rel, y_rel, width_rel, height_rel] in relative coordinates
            target_width: Target image width in pixels
            target_height: Target image height in pixels
            
        Returns:
            [x, y, width, height] in absolute pixels
        """
        x_rel, y_rel, width_rel, height_rel = bbox_rel
        
        # Convert to absolute coordinates
        x = x_rel * target_width
        y = y_rel * target_height
        width = width_rel * target_width
        height = height_rel * target_height
        
        # Round to integers
        x = round(x)
        y = round(y)
        width = max(self.min_text_width, round(width))
        height = max(self.min_text_height, round(height))
        
        # Ensure coordinates are within bounds
        x = max(0, min(target_width - width, x))
        y = max(0, min(target_height - height, y))
        
        return [float(x), float(y), float(width), float(height)]
    
    def scale_bbox_between_sizes(self, bbox: List[float], 
                                source_size: str, target_size: str) -> List[float]:
        """
        Scale bounding box coordinates between different banner sizes.
        
        Args:
            bbox: [x, y, width, height] in source size coordinates
            source_size: Source size string (e.g., "1024x1024")
            target_size: Target size string (e.g., "1536x1024")
            
        Returns:
            Scaled bbox coordinates for target size
        """
        try:
            # Get dimensions
            source_width, source_height = self.supported_sizes[source_size]
            target_width, target_height = self.supported_sizes[target_size]
            
            # Convert to relative coordinates
            bbox_rel = self.pixel_to_relative(bbox, source_width, source_height)
            
            # Convert to target pixel coordinates
            scaled_bbox = self.relative_to_pixel(bbox_rel, target_width, target_height)
            
            return scaled_bbox
            
        except KeyError as e:
            logger.error(f"Unsupported size: {e}")
            return bbox  # Return original if scaling fails
    
    def normalize_font_size(self, font_size: int, 
                           source_width: int, source_height: int,
                           target_width: int, target_height: int) -> int:
        """
        Scale font size proportionally between different image sizes.
        
        Args:
            font_size: Original font size in pixels
            source_width: Original image width
            source_height: Original image height
            target_width: Target image width
            target_height: Target image height
            
        Returns:
            Scaled font size for target dimensions
        """
        # Calculate scaling factor based on diagonal (maintains proportions)
        source_diagonal = math.sqrt(source_width**2 + source_height**2)
        target_diagonal = math.sqrt(target_width**2 + target_height**2)
        
        scale_factor = target_diagonal / source_diagonal if source_diagonal > 0 else 1.0
        
        # Scale font size
        scaled_font_size = round(font_size * scale_factor)
        
        # Clamp to valid range
        scaled_font_size = max(self.min_font_size, min(self.max_font_size, scaled_font_size))
        
        return scaled_font_size
    
    def validate_bbox_proportions(self, bbox: List[float], 
                                 image_width: int, image_height: int) -> bool:
        """
        Validate that bbox has reasonable proportions for text.
        
        Args:
            bbox: [x, y, width, height] coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            True if bbox has valid proportions
        """
        x, y, width, height = bbox
        
        # Check if coordinates are within image bounds
        if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            return False
        
        # Check minimum size
        if width < self.min_text_width or height < self.min_text_height:
            return False
        
        # Check aspect ratio (text shouldn't be too extreme)
        aspect_ratio = width / height if height > 0 else float('inf')
        if aspect_ratio > 50 or aspect_ratio < 0.1:  # Very wide or very tall
            return False
        
        # Check if bbox takes up too much of the image (probably not text)
        area_ratio = (width * height) / (image_width * image_height)
        if area_ratio > 0.8:  # More than 80% of image
            return False
        
        return True
    
    def normalize_text_elements(self, text_elements: List[Dict[str, Any]], 
                               source_size: str, target_size: str) -> List[Dict[str, Any]]:
        """
        Normalize all text elements for a different banner size.
        
        Args:
            text_elements: List of text element dictionaries
            source_size: Source banner size (e.g., "1024x1024")
            target_size: Target banner size (e.g., "1536x1024")
            
        Returns:
            List of normalized text elements
        """
        try:
            source_width, source_height = self.supported_sizes[source_size]
            target_width, target_height = self.supported_sizes[target_size]
            
            normalized_elements = []
            
            for element in text_elements:
                # Extract bbox and font size
                bbox = element.get("bbox", [0, 0, 100, 50])
                font_size = element.get("font_size", 24)
                
                # Scale bbox coordinates
                scaled_bbox = self.scale_bbox_between_sizes(bbox, source_size, target_size)
                
                # Scale font size
                scaled_font_size = self.normalize_font_size(
                    font_size, source_width, source_height, target_width, target_height
                )
                
                # Validate scaled bbox
                if not self.validate_bbox_proportions(scaled_bbox, target_width, target_height):
                    logger.warning(f"Invalid bbox proportions after scaling: {scaled_bbox}")
                    # Use fallback positioning
                    scaled_bbox = [target_width * 0.1, target_height * 0.1, 
                                  target_width * 0.3, target_height * 0.1]
                
                # Create normalized element
                normalized_element = element.copy()
                normalized_element["bbox"] = scaled_bbox
                normalized_element["font_size"] = scaled_font_size
                
                # Add scaling metadata
                normalized_element["scaling_info"] = {
                    "source_size": source_size,
                    "target_size": target_size,
                    "original_bbox": bbox,
                    "original_font_size": font_size,
                    "scale_factor": target_width / source_width  # For reference
                }
                
                normalized_elements.append(normalized_element)
            
            return normalized_elements
            
        except Exception as e:
            logger.error(f"Text element normalization failed: {e}")
            return text_elements  # Return original on failure
    
    def create_responsive_bbox_set(self, bbox: List[float], font_size: int,
                                  source_size: str) -> Dict[str, Dict[str, Any]]:
        """
        Create bbox coordinates for all supported banner sizes.
        
        Args:
            bbox: Original bbox in source size
            font_size: Original font size
            source_size: Source banner size
            
        Returns:
            Dictionary with bbox data for each supported size
        """
        responsive_set = {}
        
        for size_name in self.supported_sizes.keys():
            if size_name == source_size:
                # Use original for source size
                responsive_set[size_name] = {
                    "bbox": bbox,
                    "font_size": font_size,
                    "is_original": True
                }
            else:
                # Scale for other sizes
                scaled_bbox = self.scale_bbox_between_sizes(bbox, source_size, size_name)
                target_width, target_height = self.supported_sizes[size_name]
                source_width, source_height = self.supported_sizes[source_size]
                scaled_font_size = self.normalize_font_size(
                    font_size, source_width, source_height, target_width, target_height
                )
                
                responsive_set[size_name] = {
                    "bbox": scaled_bbox,
                    "font_size": scaled_font_size,
                    "is_original": False
                }
        
        return responsive_set
    
    def calculate_optimal_text_size(self, text_content: str, bbox: List[float],
                                   font_style: str = "regular") -> int:
        """
        Calculate optimal font size for given text content and bbox.
        
        Args:
            text_content: Text that needs to fit
            bbox: Available space [x, y, width, height]
            font_style: Font style (affects character width estimation)
            
        Returns:
            Optimal font size in pixels
        """
        _, _, width, height = bbox
        
        # Estimate character width based on font style
        char_width_ratios = {
            "bold": 0.65,
            "condensed": 0.45,
            "extended": 0.75,
            "regular": 0.6
        }
        
        char_width_ratio = char_width_ratios.get(font_style.lower(), 0.6)
        
        # Estimate lines needed (simple word wrapping)
        avg_chars_per_line = max(1, int(width / (height * char_width_ratio)))
        estimated_lines = max(1, math.ceil(len(text_content) / avg_chars_per_line))
        
        # Calculate font size that fits
        optimal_font_size = int(height / estimated_lines * 0.8)  # 80% of line height
        
        # Clamp to valid range
        optimal_font_size = max(self.min_font_size, min(self.max_font_size, optimal_font_size))
        
        return optimal_font_size


def main():
    """Test the coordinate normalizer."""
    normalizer = CoordinateNormalizer()
    
    # Test bbox scaling
    original_bbox = [100, 50, 300, 80]  # x, y, width, height
    print(f"Original bbox (1024x1024): {original_bbox}")
    
    # Scale to different sizes
    for target_size in ["1536x1024", "1024x1536"]:
        scaled = normalizer.scale_bbox_between_sizes(original_bbox, "1024x1024", target_size)
        print(f"Scaled to {target_size}: {scaled}")
    
    # Test font size scaling
    original_font = 24
    scaled_font = normalizer.normalize_font_size(
        original_font, 1024, 1024, 1536, 1024
    )
    print(f"Font size scaled from {original_font} to {scaled_font}")
    
    # Test responsive bbox set
    responsive = normalizer.create_responsive_bbox_set(
        original_bbox, original_font, "1024x1024"
    )
    print(f"Responsive bbox set: {responsive}")


if __name__ == "__main__":
    main()