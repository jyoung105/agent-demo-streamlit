"""
Banner API client for Streamlit frontend.
Handles communication with the FastAPI backend.
"""
import requests
import base64
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class BannerAPIClient:
    """Client for interacting with Banner Agent API v2."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/banner"
        self.timeout = 120  # 2 minutes timeout for banner generation
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy and OpenAI is accessible."""
        try:
            response = requests.get(
                f"{self.api_base}/health",
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"API returned status {response.status_code}",
                    "detail": response.text
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return {
                "success": False,
                "error": f"Connection failed: {str(e)}"
            }
    
    def upload_file_to_base64(self, uploaded_file) -> Optional[str]:
        """Convert uploaded file to base64 string."""
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            
            # Convert to base64
            base64_data = base64.b64encode(file_bytes).decode('utf-8')
            
            # Reset file pointer for potential reuse
            uploaded_file.seek(0)
            
            return base64_data
            
        except Exception as e:
            logger.error(f"Failed to convert file to base64: {e}")
            return None
    
    def extract_layout(self, image_data: str) -> Dict[str, Any]:
        """
        Extract layout from reference image.
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            dict: API response with layout data or error
        """
        try:
            payload = {
                "image_data": image_data
            }
            
            response = requests.post(
                f"{self.api_base}/extract-layout",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                error_data = response.json() if response.content else {"error": "Unknown error"}
                return {
                    "success": False,
                    "error": error_data.get("error", f"API error {response.status_code}"),
                    "detail": error_data.get("detail", response.text)
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Layout extraction request failed: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Layout extraction failed: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def optimize_prompt(self, layout_data: Dict[str, Any], user_requirements: str) -> Dict[str, Any]:
        """
        Optimize banner prompt based on layout and user requirements.
        
        Args:
            layout_data: Layout data from extract_layout
            user_requirements: User's banner requirements
            
        Returns:
            dict: API response with optimized data or error
        """
        try:
            payload = {
                "layout_data": layout_data,
                "user_requirements": user_requirements
            }
            
            response = requests.post(
                f"{self.api_base}/optimize-prompt",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                error_data = response.json() if response.content else {"error": "Unknown error"}
                return {
                    "success": False,
                    "error": error_data.get("error", f"API error {response.status_code}"),
                    "detail": error_data.get("detail", response.text)
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Prompt optimization request failed: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def generate_banner(self, optimized_data: Dict[str, Any], 
                       size: str = "1792x1024", 
                       include_image_data: bool = True) -> Dict[str, Any]:
        """
        Generate banner using optimized data.
        
        Args:
            optimized_data: Optimized data from optimize_prompt
            size: Image size for generation
            include_image_data: Whether to include base64 image data
            
        Returns:
            dict: API response with generated banner or error
        """
        try:
            payload = {
                "optimized_data": optimized_data,
                "size": size,
                "include_image_data": include_image_data
            }
            
            response = requests.post(
                f"{self.api_base}/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                error_data = response.json() if response.content else {"error": "Unknown error"}
                return {
                    "success": False,
                    "error": error_data.get("error", f"API error {response.status_code}"),
                    "detail": error_data.get("detail", response.text)
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Banner generation request failed: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Banner generation failed: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def complete_workflow(self, user_requirements: str, 
                         image_data: Optional[str] = None,
                         size: str = "1792x1024") -> Dict[str, Any]:
        """
        Complete banner creation workflow in a single API call.
        
        Args:
            user_requirements: User's banner requirements
            image_data: Optional base64 encoded reference image
            size: Image size for generation
            
        Returns:
            dict: API response with workflow results or error
        """
        try:
            payload = {
                "user_requirements": user_requirements,
                "size": size
            }
            
            if image_data:
                payload["image_data"] = image_data
            
            response = requests.post(
                f"{self.api_base}/workflow",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                error_data = response.json() if response.content else {"error": "Unknown error"}
                return {
                    "success": False,
                    "error": error_data.get("error", f"API error {response.status_code}"),
                    "detail": error_data.get("detail", response.text)
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Workflow request failed: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def base64_to_image_bytes(self, base64_data: str) -> Optional[bytes]:
        """Convert base64 image data to bytes for display."""
        try:
            return base64.b64decode(base64_data)
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None

    def start_workflow_async(self, user_requirements: str, 
                            image_data: Optional[str] = None,
                            size: str = "1792x1024") -> Dict[str, Any]:
        """
        Start banner workflow asynchronously and return job_id immediately.
        
        Args:
            user_requirements: User's banner requirements
            image_data: Optional base64 encoded reference image
            size: Image size for generation
            
        Returns:
            dict: API response with job_id for polling progress
        """
        try:
            payload = {
                "user_requirements": user_requirements,
                "size": size
            }
            
            if image_data:
                payload["image_data"] = image_data
            
            response = requests.post(
                f"{self.api_base}/start-workflow",
                json=payload,
                timeout=30  # Shorter timeout since this should return quickly
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                error_data = response.json() if response.content else {"error": "Unknown error"}
                return {
                    "success": False,
                    "error": error_data.get("error", f"API error {response.status_code}"),
                    "detail": error_data.get("detail", response.text)
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Start workflow request failed: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Start workflow failed: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed job status with step-by-step progress and outputs.
        
        Args:
            job_id: Job ID from start_workflow_async
            
        Returns:
            dict: API response with detailed job status and step outputs
        """
        try:
            response = requests.get(
                f"{self.api_base}/job-status/{job_id}",
                timeout=10  # Quick status check
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            elif response.status_code == 404:
                return {
                    "success": False,
                    "error": f"Job {job_id} not found"
                }
            else:
                error_data = response.json() if response.content else {"error": "Unknown error"}
                return {
                    "success": False,
                    "error": error_data.get("error", f"API error {response.status_code}"),
                    "detail": error_data.get("detail", response.text)
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Job status request failed: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Job status check failed: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def poll_job_with_callback(self, job_id: str, callback_func, polling_interval: float = 2.0, timeout: int = 300) -> Dict[str, Any]:
        """
        Poll job status with callback function for real-time updates.
        
        Args:
            job_id: Job ID to monitor
            callback_func: Function to call with each status update
            polling_interval: Seconds between status checks
            timeout: Maximum time to poll in seconds
            
        Returns:
            Final job status
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_response = self.get_job_status(job_id)
            
            if not status_response["success"]:
                return {"status": "error", "error": status_response["error"]}
            
            job_data = status_response["data"]
            job_status = job_data.get("status", "unknown")
            
            # Call callback with current status
            if callback_func:
                callback_func(job_data)
            
            # Check if job is complete
            if job_status in ["completed", "failed"]:
                return job_data
            
            # Wait before next poll
            time.sleep(polling_interval)
        
        return {"status": "timeout", "error": f"Job polling timed out after {timeout}s"}


class BannerWorkflowTracker:
    """Helper class to track banner creation workflow progress."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset workflow tracking."""
        self.start_time = None
        self.steps = []
        self.current_step = 0
        self.total_steps = 3
        self.is_running = False
        self.is_complete = False
        self.final_result = None
    
    def start_workflow(self):
        """Start tracking a new workflow."""
        self.reset()
        self.start_time = time.time()
        self.is_running = True
        self.steps = [
            {"name": "Extract Layout", "status": "pending", "message": "Analyzing reference image..."},
            {"name": "Optimize Prompt", "status": "pending", "message": "Optimizing banner design..."},
            {"name": "Generate Banner", "status": "pending", "message": "Creating your banner..."}
        ]
    
    def update_step(self, step_index: int, status: str, message: str = "", result: Any = None):
        """Update the status of a workflow step."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = status
            if message:
                self.steps[step_index]["message"] = message
            if result:
                self.steps[step_index]["result"] = result
            
            if status == "completed":
                self.current_step = min(step_index + 1, self.total_steps)
    
    def complete_workflow(self, success: bool, final_result: Any = None):
        """Mark workflow as complete."""
        self.is_running = False
        self.is_complete = True
        self.final_result = final_result
        
        if success:
            # Mark all steps as completed
            for step in self.steps:
                if step["status"] == "pending":
                    step["status"] = "completed"
        
    def get_progress_percentage(self) -> int:
        """Get current progress as percentage."""
        if not self.is_running and not self.is_complete:
            return 0
        
        completed_steps = sum(1 for step in self.steps if step["status"] == "completed")
        return int((completed_steps / self.total_steps) * 100)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since workflow started."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0