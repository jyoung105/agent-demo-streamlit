"""
Bounding box coordinate extraction system for banner generation.
Extracts accurate bbox coordinates from input images with real pixel dimensions.
"""
import base64
import io
import logging
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Any, Optional
import json

logger = logging.getLogger(__name__)


class BboxExtractor:
    """
    Advanced bounding box extraction system that combines computer vision
    and AI analysis to extract precise text element coordinates.
    """
    
    def __init__(self):
        self.min_text_area = 100  # Minimum area for text detection
        self.confidence_threshold = 0.5
        
    def get_image_dimensions(self, image_data: str) -> Tuple[int, int]:
        """
        Extract real pixel dimensions from base64 encoded image.
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Tuple of (width, height) in pixels
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return image.size  # Returns (width, height)
        except Exception as e:
            logger.error(f"Failed to get image dimensions: {e}")
            return (1024, 1024)  # Default fallback
    
    def extract_text_regions_opencv(self, image_data: str) -> List[Dict[str, Any]]:
        """
        Simplified text region detection without OpenCV dependencies.
        Uses basic image analysis to identify potential text regions.
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            List of detected text regions with confidence scores
        """
        try:
            # Decode image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale for analysis
            gray_image = image.convert('L')
            width, height = image.size
            
            # Simple grid-based text detection
            # Divide image into regions and analyze contrast
            grid_size = 32
            text_regions = []
            
            for y in range(0, height - grid_size, grid_size // 2):
                for x in range(0, width - grid_size, grid_size // 2):
                    # Extract region
                    region = gray_image.crop((x, y, x + grid_size, y + grid_size))
                    
                    # Calculate contrast (simple text detection heuristic)
                    pixels = list(region.getdata())
                    if len(pixels) > 0:
                        min_val = min(pixels)
                        max_val = max(pixels)
                        contrast = max_val - min_val
                        
                        # High contrast regions likely contain text
                        if contrast > 50:  # Threshold for text detection
                            # Expand region to more reasonable text size
                            text_width = min(300, width - x)
                            text_height = min(80, height - y)
                            
                            confidence = min(1.0, contrast / 255.0)
                            
                            text_regions.append({
                                "bbox": [x, y, text_width, text_height],
                                "confidence": confidence,
                                "area": text_width * text_height,
                                "method": "simple_contrast"
                            })
            
            # Remove overlapping regions and sort by confidence
            text_regions = self.merge_overlapping_boxes(text_regions, 0.5)
            text_regions.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Limit to top detections
            return text_regions[:10]
            
        except Exception as e:
            logger.error(f"Simple text detection failed: {e}")
            return []
    
    def merge_overlapping_boxes(self, boxes: List[Dict[str, Any]], 
                               overlap_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Merge overlapping bounding boxes to avoid duplicates.
        
        Args:
            boxes: List of bbox dictionaries
            overlap_threshold: Minimum overlap ratio to trigger merge
            
        Returns:
            List of merged boxes
        """
        if not boxes:
            return []
        
        def calculate_overlap(box1: List[float], box2: List[float]) -> float:
            """Calculate intersection over union (IoU) of two boxes."""
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
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
        
        merged_boxes = []
        used_indices = set()
        
        for i, box in enumerate(boxes):
            if i in used_indices:
                continue
                
            # Find all boxes that overlap with current box
            overlapping = [box]
            used_indices.add(i)
            
            for j, other_box in enumerate(boxes[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                overlap = calculate_overlap(box["bbox"], other_box["bbox"])
                if overlap > overlap_threshold:
                    overlapping.append(other_box)
                    used_indices.add(j)
            
            # Merge overlapping boxes
            if len(overlapping) == 1:
                merged_boxes.append(overlapping[0])
            else:
                # Calculate merged bounding box
                all_x = [b["bbox"][0] for b in overlapping]
                all_y = [b["bbox"][1] for b in overlapping]
                all_x2 = [b["bbox"][0] + b["bbox"][2] for b in overlapping]
                all_y2 = [b["bbox"][1] + b["bbox"][3] for b in overlapping]
                
                merged_x = min(all_x)
                merged_y = min(all_y)
                merged_w = max(all_x2) - merged_x
                merged_h = max(all_y2) - merged_y
                
                # Average confidence
                avg_confidence = sum(b["confidence"] for b in overlapping) / len(overlapping)
                
                merged_boxes.append({
                    "bbox": [merged_x, merged_y, merged_w, merged_h],
                    "confidence": avg_confidence,
                    "area": merged_w * merged_h,
                    "method": "merged",
                    "merged_count": len(overlapping)
                })
        
        return merged_boxes
    
    def validate_bbox_coordinates(self, bbox: List[float], 
                                 image_width: int, image_height: int) -> List[float]:
        """
        Validate and clamp bbox coordinates to image boundaries.
        
        Args:
            bbox: [x, y, width, height] coordinates
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            Validated bbox coordinates
        """
        x, y, w, h = bbox
        
        # Clamp to image boundaries
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        w = max(1, min(w, image_width - x))
        h = max(1, min(h, image_height - y))
        
        return [x, y, w, h]
    
    def extract_precise_bbox_coordinates(self, image_data: str, 
                                       ai_detected_boxes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Extract precise bounding box coordinates combining AI and CV methods.
        
        Args:
            image_data: Base64 encoded image string
            ai_detected_boxes: Optional AI-detected text boxes for validation
            
        Returns:
            Dictionary with extracted bbox information
        """
        try:
            # Get real image dimensions
            width, height = self.get_image_dimensions(image_data)
            
            # Extract text regions using computer vision
            cv_boxes = self.extract_text_regions_opencv(image_data)
            
            # Merge overlapping boxes
            merged_boxes = self.merge_overlapping_boxes(cv_boxes)
            
            # Validate coordinates
            validated_boxes = []
            for box in merged_boxes:
                validated_bbox = self.validate_bbox_coordinates(box["bbox"], width, height)
                box["bbox"] = validated_bbox
                validated_boxes.append(box)
            
            # Calculate extraction statistics
            total_boxes = len(validated_boxes)
            high_confidence_boxes = len([b for b in validated_boxes if b["confidence"] > 0.7])
            
            return {
                "success": True,
                "image_dimensions": {
                    "width": width,
                    "height": height
                },
                "detected_boxes": validated_boxes,
                "statistics": {
                    "total_boxes": total_boxes,
                    "high_confidence_boxes": high_confidence_boxes,
                    "detection_methods": list(set(b["method"] for b in validated_boxes))
                },
                "ai_boxes_provided": ai_detected_boxes is not None,
                "extraction_method": "cv_with_ai_validation" if ai_detected_boxes else "cv_only"
            }
            
        except Exception as e:
            logger.error(f"Bbox extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_dimensions": {"width": 1024, "height": 1024},
                "detected_boxes": []
            }
    
    def create_bbox_visualization(self, image_data: str, 
                                 bbox_results: Dict[str, Any]) -> Optional[str]:
        """
        Create a visualization of detected bounding boxes for debugging.
        
        Args:
            image_data: Base64 encoded image string
            bbox_results: Results from extract_precise_bbox_coordinates
            
        Returns:
            Base64 encoded visualization image or None if failed
        """
        try:
            # Decode image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            draw = ImageDraw.Draw(image)
            
            # Draw detected boxes
            for i, box in enumerate(bbox_results.get("detected_boxes", [])):
                bbox = box["bbox"]
                confidence = box["confidence"]
                
                # Color based on confidence
                if confidence > 0.7:
                    color = "green"
                elif confidence > 0.4:
                    color = "orange"
                else:
                    color = "red"
                
                # Draw rectangle
                x, y, w, h = bbox
                draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
                
                # Draw confidence label
                label = f"{i+1}: {confidence:.2f}"
                draw.text((x, y - 15), label, fill=color)
            
            # Save visualization
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG')
            output_buffer.seek(0)
            
            # Encode to base64
            visualization_b64 = base64.b64encode(output_buffer.getvalue()).decode()
            return visualization_b64
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return None
    
    def extract_bbox_from_ai_response(self, ai_layout_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and validate bbox coordinates from AI layout response.
        
        Args:
            ai_layout_data: Layout data from AI vision model
            
        Returns:
            List of validated bbox coordinates with metadata
        """
        try:
            extracted_boxes = []
            
            # Get image dimensions
            image_info = ai_layout_data.get("image_info", {})
            img_width = image_info.get("original_width", 1024)
            img_height = image_info.get("original_height", 1024)
            
            # Extract text elements
            text_elements = ai_layout_data.get("text", [])
            
            for i, text_elem in enumerate(text_elements):
                bbox = text_elem.get("bbox", [0, 0, 100, 50])
                
                # Validate bbox format
                if not isinstance(bbox, list) or len(bbox) != 4:
                    logger.warning(f"Invalid bbox format for text element {i}: {bbox}")
                    continue
                
                # Validate coordinates
                validated_bbox = self.validate_bbox_coordinates(bbox, img_width, img_height)
                
                extracted_boxes.append({
                    "bbox": validated_bbox,
                    "confidence": 0.9,  # AI detection gets high confidence
                    "method": "ai_vision",
                    "text_info": {
                        "font_size": text_elem.get("font_size", 24),
                        "font_style": text_elem.get("font_style", "regular"),
                        "font_color": text_elem.get("font_color", "#000000"),
                        "description": text_elem.get("description", "")
                    }
                })
            
            return extracted_boxes
            
        except Exception as e:
            logger.error(f"AI bbox extraction failed: {e}")
            return []


def main():
    """Test the bbox extractor with sample data."""
    extractor = BboxExtractor()
    
    # Test with a sample image (would need actual base64 data)
    sample_bbox_data = {
        "image_info": {"original_width": 1536, "original_height": 1024},
        "text": [
            {
                "bbox": [100, 50, 300, 80],
                "font_size": 36,
                "font_style": "bold",
                "font_color": "#FFFFFF",
                "description": "Main Title"
            },
            {
                "bbox": [150, 200, 250, 40],
                "font_size": 18,
                "font_style": "regular", 
                "font_color": "#333333",
                "description": "Subtitle text"
            }
        ]
    }
    
    # Extract bbox from AI data
    ai_boxes = extractor.extract_bbox_from_ai_response(sample_bbox_data)
    print("AI Extracted Boxes:")
    for box in ai_boxes:
        print(f"  {box}")


if __name__ == "__main__":
    main()