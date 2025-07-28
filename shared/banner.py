from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import base64


class TextElement(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box [x, y, width, height]")
    font_size: int = Field(..., ge=8, le=200, description="Font size in pixels")
    font_style: str = Field(..., description="Font style (bold, italic, regular, etc.)")
    font_color: str = Field(..., description="Font color as hex code or color name")
    description: str = Field(..., description="Description of what the text says")


class CameraSettings(BaseModel):
    angle: str = Field(default="eye level", description="Camera angle")
    distance: str = Field(default="medium shot", description="Shot distance")
    focus: str = Field(default="sharp focus", description="Focus description")


class Subject(BaseModel):
    type: str = Field(..., description="Type of subject (person, object, etc.)")
    description: str = Field(..., description="Detailed description of the subject")
    pose: str = Field(default="", description="Pose or position of the subject")
    position: str = Field(default="foreground", description="Position in frame")


class Background(BaseModel):
    scene: str = Field(..., description="Overall scene description")
    subjects: List[Subject] = Field(default=[], description="Main subjects in the scene")
    style: str = Field(default="digital illustration", description="Art style")
    color_palette: List[str] = Field(default=[], description="Main colors in the design")
    lighting: str = Field(default="balanced lighting", description="Lighting conditions")
    mood: str = Field(default="professional", description="Overall mood or atmosphere")
    background: str = Field(default="clean background", description="Background elements")
    composition: str = Field(default="centered layout", description="Composition technique")
    camera: CameraSettings = Field(default_factory=CameraSettings, description="Camera settings")


class ImageInfo(BaseModel):
    original_width: int = Field(default=1024, description="Original image width in pixels")
    original_height: int = Field(default=1024, description="Original image height in pixels")


class BannerLayoutData(BaseModel):
    text: List[TextElement] = Field(default=[], description="Text elements in the banner")
    background: Background = Field(..., description="Background and scene information")
    image_info: Optional[ImageInfo] = Field(default_factory=ImageInfo, description="Original image dimensions")


class BannerLayoutRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    
    @validator('image_data')
    def validate_base64(cls, v):
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 image data")


class BannerLayoutResponse(BaseModel):
    success: bool = Field(..., description="Whether the extraction was successful")
    layout_data: Optional[BannerLayoutData] = Field(None, description="Extracted layout information")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)


class BannerPromptRequest(BaseModel):
    layout_data: BannerLayoutData = Field(..., description="Layout data from extraction")
    user_requirements: str = Field(..., min_length=1, max_length=2000, description="User's banner requirements")


class BannerPromptResponse(BaseModel):
    success: bool = Field(..., description="Whether the optimization was successful")
    optimized_data: Optional[BannerLayoutData] = Field(None, description="Optimized prompt data")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)


class BannerGenerationRequest(BaseModel):
    optimized_data: BannerLayoutData = Field(..., description="Optimized prompt data")
    size: str = Field(default="1792x1024", description="Image size for generation")
    include_image_data: bool = Field(default=True, description="Whether to include base64 image data")
    
    @validator('size')
    def validate_size(cls, v):
        valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        if v not in valid_sizes:
            raise ValueError(f"Size must be one of {valid_sizes}")
        return v


class BannerGenerationResponse(BaseModel):
    success: bool = Field(..., description="Whether the generation was successful")
    image_data: Optional[str] = Field(None, description="Base64 encoded generated image")
    image_url: Optional[str] = Field(None, description="URL to the generated image")
    original_prompt: Optional[str] = Field(None, description="The prompt used for generation")
    size: str = Field(..., description="Size of the generated image")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., description="Time taken to generate in seconds")
    workflow_data: Optional[Dict[str, Any]] = Field(None, description="Workflow execution data and intermediate results")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('size')
    def validate_size(cls, v):
        valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        if v not in valid_sizes:
            raise ValueError(f"Size must be one of {valid_sizes}")
        return v


class BannerWorkflowRequest(BaseModel):
    image_data: Optional[str] = Field(None, description="Base64 encoded reference image (optional)")
    user_requirements: str = Field(..., min_length=1, max_length=2000, description="User's banner requirements")
    size: str = Field(default="1792x1024", description="Image size for generation")
    
    @validator('image_data')
    def validate_base64_optional(cls, v):
        if v is None:
            return v
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 image data")
    
    @validator('size')
    def validate_size(cls, v):
        valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        if v not in valid_sizes:
            raise ValueError(f"Size must be one of {valid_sizes}")
        return v


class BannerWorkflowStep(BaseModel):
    step: int = Field(..., description="Step number (1, 2, or 3)")
    name: str = Field(..., description="Step name")
    status: str = Field(..., description="Step status (pending, in_progress, completed, failed)")
    result: Optional[Dict[str, Any]] = Field(None, description="Step result data")
    error: Optional[str] = Field(None, description="Error message if step failed")
    processing_time: Optional[float] = Field(None, description="Time taken for this step")


class BannerWorkflowResponse(BaseModel):
    success: bool = Field(..., description="Whether the entire workflow was successful")
    steps: List[BannerWorkflowStep] = Field(..., description="Status of each workflow step")
    final_result: Optional[BannerGenerationResponse] = Field(None, description="Final banner generation result")
    total_processing_time: float = Field(..., description="Total time for entire workflow")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    openai_connection: bool = Field(..., description="Whether OpenAI API is accessible")
    version: str = Field(default="2.0.0", description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)