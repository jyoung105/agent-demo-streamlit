"""
Database models and setup for Banner Agent v2.
Provides persistent storage for workflow data and job tracking.
"""
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Integer, Boolean, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
import json

# Database URL from environment variable, fallback to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./banner_agent.db")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class BannerJob(Base):
    """
    Main job tracking table for banner generation workflows
    """
    __tablename__ = "banner_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default=JobStatus.PENDING)
    user_requirements = Column(Text, nullable=False)
    image_size = Column(String, default="1792x1024")
    original_image_data = Column(Text, nullable=True)  # Base64 encoded original image
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Processing times
    total_processing_time = Column(Float, default=0.0)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Final results
    final_image_data = Column(Text, nullable=True)  # Base64 encoded final image
    final_image_url = Column(String, nullable=True)
    
    # Relationships
    steps = relationship("BannerJobStep", back_populates="job", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "user_requirements": self.user_requirements,
            "image_size": self.image_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_processing_time": self.total_processing_time,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "has_final_image": bool(self.final_image_data or self.final_image_url),
            "steps": [step.to_dict() for step in self.steps] if self.steps else []
        }


class BannerJobStep(Base):
    """
    Individual step tracking for banner generation workflow
    """
    __tablename__ = "banner_job_steps"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("banner_jobs.id"), nullable=False)
    step_number = Column(Integer, nullable=False)  # 1, 2, 3, 4
    step_name = Column(String, nullable=False)  # "layout_extraction", "prompt_generation", "image_generation", "text_overlay"
    status = Column(String, default=StepStatus.PENDING)
    
    # Timestamps
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    processing_time = Column(Float, default=0.0)
    
    # Step inputs and outputs
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # API call metadata
    api_model_used = Column(String, nullable=True)
    api_tokens_used = Column(Integer, nullable=True)
    api_cost = Column(Float, nullable=True)
    
    # Relationship
    job = relationship("BannerJob", back_populates="steps")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "step_number": self.step_number,
            "step_name": self.step_name,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time": self.processing_time,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "api_model_used": self.api_model_used,
            "api_tokens_used": self.api_tokens_used,
            "api_cost": self.api_cost
        }


class LayoutExtraction(Base):
    """
    Storage for layout extraction results
    """
    __tablename__ = "layout_extractions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("banner_jobs.id"), nullable=False)
    
    # Original image info
    original_image_hash = Column(String, nullable=True)  # MD5 hash for deduplication
    
    # Extracted layout data
    text_elements = Column(JSON, nullable=False)  # List of text elements with bbox, styling
    background_data = Column(JSON, nullable=False)  # Scene, subjects, style, etc.
    
    # Processing metadata
    extraction_model = Column(String, default="gpt-4o")
    extraction_confidence = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, default=0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "text_elements": self.text_elements,
            "background_data": self.background_data,
            "extraction_model": self.extraction_model,
            "extraction_confidence": self.extraction_confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time": self.processing_time
        }


class PromptGeneration(Base):
    """
    Storage for generated prompts
    """
    __tablename__ = "prompt_generations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("banner_jobs.id"), nullable=False)
    
    # Input data
    layout_extraction_id = Column(String, ForeignKey("layout_extractions.id"), nullable=True)
    user_requirements = Column(Text, nullable=False)
    
    # Generated prompt data
    optimized_scene = Column(JSON, nullable=False)  # Complete optimized scene description
    generation_prompt = Column(Text, nullable=False)  # Final text prompt for image generation
    
    # Processing metadata
    prompt_model = Column(String, default="gpt-4")
    prompt_strategy = Column(String, default="scene_optimization")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, default=0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "layout_extraction_id": self.layout_extraction_id,
            "user_requirements": self.user_requirements,
            "optimized_scene": self.optimized_scene,
            "generation_prompt": self.generation_prompt,
            "prompt_model": self.prompt_model,
            "prompt_strategy": self.prompt_strategy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time": self.processing_time
        }


class ImageGeneration(Base):
    """
    Storage for generated background images (step 3)
    """
    __tablename__ = "image_generations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("banner_jobs.id"), nullable=False)
    
    # Input data
    prompt_generation_id = Column(String, ForeignKey("prompt_generations.id"), nullable=True)
    generation_prompt = Column(Text, nullable=False)
    
    # Generated background image data
    image_data = Column(Text, nullable=True)  # Base64 encoded background image
    image_url = Column(String, nullable=True)  # External URL if stored elsewhere
    image_size = Column(String, default="1792x1024")
    image_format = Column(String, default="png")
    
    # Processing metadata
    generation_model = Column(String, default="gpt-image-1")
    generation_settings = Column(JSON, nullable=True)  # Model-specific settings
    
    # Quality metrics
    generation_score = Column(Float, nullable=True)
    user_rating = Column(Integer, nullable=True)  # 1-5 rating from user
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, default=0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "prompt_generation_id": self.prompt_generation_id,
            "generation_prompt": self.generation_prompt,
            "image_size": self.image_size,
            "image_format": self.image_format,
            "generation_model": self.generation_model,
            "generation_settings": self.generation_settings,
            "generation_score": self.generation_score,
            "user_rating": self.user_rating,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time": self.processing_time,
            "has_image_data": bool(self.image_data),
            "has_image_url": bool(self.image_url)
        }


class TextOverlay(Base):
    """
    Storage for text overlay results (step 4)
    """
    __tablename__ = "text_overlays"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("banner_jobs.id"), nullable=False)
    
    # Input data
    image_generation_id = Column(String, ForeignKey("image_generations.id"), nullable=False)
    text_elements = Column(JSON, nullable=False)  # List of text elements with positioning and styling
    
    # Final composite image data
    final_image_data = Column(Text, nullable=False)  # Base64 encoded final banner with text overlays
    final_image_url = Column(String, nullable=True)  # External URL if stored elsewhere
    
    # Processing metadata
    overlay_engine = Column(String, default="PIL")
    font_settings = Column(JSON, nullable=True)  # Font configurations used
    
    # Quality metrics
    text_quality_score = Column(Float, nullable=True)
    user_rating = Column(Integer, nullable=True)  # 1-5 rating from user
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, default=0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "image_generation_id": self.image_generation_id,
            "text_elements": self.text_elements,
            "overlay_engine": self.overlay_engine,
            "font_settings": self.font_settings,
            "text_quality_score": self.text_quality_score,
            "user_rating": self.user_rating,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time": self.processing_time,
            "has_final_image_data": bool(self.final_image_data),
            "has_final_image_url": bool(self.final_image_url)
        }


def get_database_session() -> Session:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_database():
    """
    Initialize database tables
    """
    Base.metadata.create_all(bind=engine)


def get_job_by_id(db: Session, job_id: str) -> Optional[BannerJob]:
    """
    Get job by ID with all related data
    """
    return db.query(BannerJob).filter(BannerJob.id == job_id).first()


def create_banner_job(db: Session, user_requirements: str, image_size: str = "1792x1024", 
                     original_image_data: str = None) -> BannerJob:
    """
    Create a new banner job
    """
    job = BannerJob(
        user_requirements=user_requirements,
        image_size=image_size,
        original_image_data=original_image_data
    )
    db.add(job)
    db.flush()  # Flush to get the job.id before creating steps
    
    # Create initial job steps for 4-step workflow
    steps = [
        BannerJobStep(job_id=job.id, step_number=1, step_name="layout_extraction"),
        BannerJobStep(job_id=job.id, step_number=2, step_name="prompt_generation"),
        BannerJobStep(job_id=job.id, step_number=3, step_name="image_generation"),
        BannerJobStep(job_id=job.id, step_number=4, step_name="text_overlay")
    ]
    
    for step in steps:
        db.add(step)
    
    db.commit()
    db.refresh(job)
    return job


def update_job_status(db: Session, job_id: str, status: JobStatus, 
                     error_message: str = None, error_details: Dict = None) -> bool:
    """
    Update job status
    """
    job = get_job_by_id(db, job_id)
    if not job:
        return False
    
    job.status = status
    job.updated_at = datetime.utcnow()
    
    if status == JobStatus.IN_PROGRESS and not job.started_at:
        job.started_at = datetime.utcnow()
    elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
        job.completed_at = datetime.utcnow()
        if job.started_at:
            job.total_processing_time = (job.completed_at - job.started_at).total_seconds()
    
    if error_message:
        job.error_message = error_message
        job.error_details = error_details
    
    db.commit()
    return True


def update_step_status(db: Session, job_id: str, step_number: int, status: StepStatus,
                      input_data: Dict = None, output_data: Dict = None,
                      error_message: str = None, error_details: Dict = None,
                      api_model: str = None, processing_time: float = None) -> bool:
    """
    Update step status and data
    """
    step = db.query(BannerJobStep).filter(
        BannerJobStep.job_id == job_id,
        BannerJobStep.step_number == step_number
    ).first()
    
    if not step:
        return False
    
    step.status = status
    
    if status == StepStatus.IN_PROGRESS and not step.started_at:
        step.started_at = datetime.utcnow()
    elif status in [StepStatus.COMPLETED, StepStatus.FAILED]:
        step.completed_at = datetime.utcnow()
        if step.started_at and not processing_time:
            step.processing_time = (step.completed_at - step.started_at).total_seconds()
        elif processing_time:
            step.processing_time = processing_time
    
    if input_data:
        step.input_data = input_data
    if output_data:
        step.output_data = output_data
    if error_message:
        step.error_message = error_message
        step.error_details = error_details
    if api_model:
        step.api_model_used = api_model
    
    db.commit()
    return True