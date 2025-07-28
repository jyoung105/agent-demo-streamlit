"""
Banner Agent v2 - Streamlit Frontend with Embedded FastAPI
AI-powered banner creation using embedded FastAPI backend.
"""
import streamlit as st
import requests
import subprocess
import time
import threading
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import base64
from datetime import datetime
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import time

# Add directories to path for module imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
# Insert in reverse order since insert(0, ...) adds to the beginning
sys.path.insert(0, str(current_dir / "fastapi_app"))  # Add embedded fastapi_app
sys.path.insert(0, str(parent_dir))   # Add parent for shared modules  
sys.path.insert(0, str(current_dir))  # Add current directory for local imports (highest priority)

# Import shared modules
from shared.banner import BannerWorkflowRequest

# Import local components
from components.step_output_viewer import render_step_output_viewer
from components.banner_progress import poll_job_status_with_display
from services.banner_api import BannerAPIClient

# Import FastAPI app for embedded mode
EMBEDDED_MODE = False
fastapi_app = None

def import_fastapi_app():
    """Import FastAPI app with proper directory context"""
    try:
        # Save original cwd
        import os
        import sys
        original_cwd = os.getcwd()
        
        # Add fastapi_app directory to Python path
        fastapi_dir = Path(__file__).parent / "fastapi_app"
        if str(fastapi_dir) not in sys.path:
            sys.path.insert(0, str(fastapi_dir))
        
        # Also add the parent directory for shared imports
        parent_dir = Path(__file__).parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        # Import the fastapi_app package to initialize paths
        import fastapi_app
        
        # Now import the app from main
        from fastapi_app.main import app
        
        return app, True
    except ImportError as e:
        st.error(f"Failed to import FastAPI app: {e}")
        st.error(f"Current path: {sys.path}")
        st.error(f"Current directory: {os.getcwd()}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, False

# Configure page
st.set_page_config(
    page_title="Banner Agent v2",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration loader
from config_loader import get_config, is_streamlit_cloud, get_all_config

# Constants - Dynamic port assignment for Streamlit Cloud
IS_STREAMLIT_CLOUD = is_streamlit_cloud()
API_PORT = int(get_config("PORT", "8001")) if not IS_STREAMLIT_CLOUD else 8001
API_BASE_URL = f"http://localhost:{API_PORT}"
FASTAPI_STARTUP_TIMEOUT = 30  # seconds

@st.cache_resource
def start_fastapi_server():
    """Start FastAPI server - embedded for Streamlit Cloud, subprocess for local"""
    if IS_STREAMLIT_CLOUD:
        return start_embedded_fastapi()
    else:
        return start_subprocess_fastapi()

def start_embedded_fastapi():
    """Start FastAPI server in embedded mode using threading"""
    try:
        # Import FastAPI app with proper context
        app, success = import_fastapi_app()
        if not success or app is None:
            st.error("FastAPI app not available for embedded mode")
            return None
            
        # Start FastAPI server in a background thread
        def run_fastapi():
            try:
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=API_PORT,
                    log_level="info",
                    access_log=False  # Reduce logs in embedded mode
                )
            except Exception as e:
                logger.error(f"Embedded FastAPI server error: {e}")
        
        # Start server in daemon thread
        server_thread = threading.Thread(target=run_fastapi, daemon=True)
        server_thread.start()
        
        # Wait for server to be ready
        start_time = time.time()
        server_ready = False
        
        st.info("🚀 Starting embedded FastAPI backend...")
        
        while time.time() - start_time < FASTAPI_STARTUP_TIMEOUT:
            try:
                response = requests.get(f"{API_BASE_URL}/", timeout=2)
                if response.status_code == 200:
                    server_ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        
        if server_ready:
            st.success("🚀 FastAPI backend started successfully (embedded mode)!")
            logger.info("Embedded FastAPI server started successfully")
            return server_thread
        else:
            st.error("⚠️ Embedded FastAPI server failed to start within timeout")
            return None
            
    except Exception as e:
        st.error(f"Failed to start embedded FastAPI server: {e}")
        logger.error(f"Embedded FastAPI startup error: {e}")
        return None

def start_subprocess_fastapi():
    """Start FastAPI server as subprocess (local development)"""
    try:
        # Get the path to the FastAPI app (embedded in streamlit_app)
        fastapi_path = Path(__file__).parent / "fastapi_app"
        
        # Check if main.py exists
        main_py_path = fastapi_path / "main.py"
        if not main_py_path.exists():
            st.error(f"FastAPI main.py not found at {main_py_path}")
            return None
        
        # Try virtual environment first, fall back to system python
        current_dir = Path(__file__).parent
        venv_python = current_dir / "fastapi_app" / "venv" / "bin" / "python"
        
        if venv_python.exists():
            python_cmd = str(venv_python)
        else:
            python_cmd = sys.executable
            st.warning("Virtual environment not found, using system Python")
            
        process = subprocess.Popen([
            python_cmd, 
            "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", str(API_PORT),
            "--log-level", "info"
        ], cwd=str(fastapi_path))
        
        # Wait for server to start with timeout
        start_time = time.time()
        server_ready = False
        
        st.info("🚀 Starting FastAPI backend subprocess...") 
        
        while time.time() - start_time < FASTAPI_STARTUP_TIMEOUT:
            try:
                response = requests.get(f"{API_BASE_URL}/", timeout=2)
                if response.status_code == 200:
                    server_ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        
        if server_ready:
            st.success("🚀 FastAPI backend started successfully (subprocess mode)!")
            logger.info("Subprocess FastAPI server started successfully")
            return process
        else:
            st.error("⚠️ FastAPI server failed to start within timeout")
            process.terminate()
            return None
            
    except Exception as e:
        st.error(f"Failed to start subprocess FastAPI server: {e}")
        logger.error(f"Subprocess FastAPI startup error: {e}")
        return None

def check_api_health() -> bool:
    """Check if the FastAPI server is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/banner/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def main():
    """Main Streamlit application"""
    st.title("🎨 Banner Agent v2")
    st.markdown("Start to steal banner with AI.")
    
    # Start FastAPI backend
    with st.spinner("Initializing FastAPI backend..."):
        server_process = start_fastapi_server()
        
    if server_process is None:
        st.error("❌ Failed to start FastAPI backend. Please check logs.")
        st.stop()
    
    # Check API health
    if not check_api_health():
        st.error("❌ FastAPI backend is not responding. Please check the server.")
        st.stop()
    
    st.success("✅ Backend is running and healthy!")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    st.sidebar.info(f"**API Endpoint:** {API_BASE_URL}")
    st.sidebar.info(f"**Mode:** {'Embedded' if IS_STREAMLIT_CLOUD or EMBEDDED_MODE else 'Subprocess'}")
    
    # Main interface - only Create Banner tab
    st.markdown("---")
    st.header("Create Banner")
    
    # Initialize template selection state
    if "selected_template" not in st.session_state:
        st.session_state.selected_template = None
    
    # Create container for organized inputs
    with st.container():
        # User requirements input (description)
        st.markdown("#### Description")
        user_requirements = st.text_area(
            "Describe your banner requirements:",
            placeholder="e.g., A modern tech startup banner with blue colors and professional look",
            height=100,
            label_visibility="collapsed"
        )
        
        # Image upload section (reference_image)
        st.markdown("#### Reference Image")
        uploaded_file = st.file_uploader(
            "Upload your own image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a reference image to extract layout information",
            label_visibility="collapsed"
        )
        
        # Template selection section (banner_template)
        st.markdown("#### Banner Template")
        st.caption("Click on a template to select it")
        
        template_cols = st.columns(3)
        template_paths = {
            "Example 1": "example-1.jpg",
            "Example 2": "example-2.jpg", 
            "Example 3": "example-3.jpg"
        }
        
        example_file_data = None
        for idx, (name, filename) in enumerate(template_paths.items()):
            with template_cols[idx]:
                template_path = Path(__file__).parent.parent / "public" / "banner-example" / filename
                if template_path.exists():
                    with open(template_path, "rb") as f:
                        template_data = f.read()
                    
                    # Display the template image
                    st.image(template_data, caption=name, use_container_width=True)
                    
                    # Create clickable button
                    if st.button(
                        f"Select {name}",
                        key=f"main_select_template_{idx}",
                        use_container_width=True,
                        type="primary" if st.session_state.selected_template == name else "secondary"
                    ):
                        st.session_state.selected_template = name
                        st.rerun()
                    
                    # Show selected indicator
                    if st.session_state.selected_template == name:
                        st.success(f"✓ Selected")
                        example_file_data = template_data
        
        # Clear selection button
        if st.session_state.selected_template:
            if st.button("❌ Clear Template Selection", key="main_clear_template"):
                st.session_state.selected_template = None
                st.rerun()
            st.info(f"📷 Using {st.session_state.selected_template} as reference")
        
        # Size selection
        st.markdown("#### Size")
        size_options = ["1536x1024", "1024x1024", "1024x1536"]
        selected_size = st.selectbox("Banner Size:", size_options, index=1, label_visibility="collapsed")
        
        # Generate button
        st.markdown("#### Generate")
        if st.button("🎨 Generate Banner", type="primary", use_container_width=True):
            if not user_requirements.strip():
                st.error("Please provide banner requirements")
                return
                
            with st.spinner("Creating your banner..."):
                try:
                    # Create workflow request
                    workflow_request = {
                        "user_requirements": user_requirements,
                        "size": selected_size,
                        "image_data": None
                    }
                    
                    # Add image data if uploaded or example selected
                    if uploaded_file is not None:
                        image_bytes = uploaded_file.read()
                        image_base64 = base64.b64encode(image_bytes).decode()
                        workflow_request["image_data"] = image_base64
                    elif example_file_data is not None:
                        image_base64 = base64.b64encode(example_file_data).decode()
                        workflow_request["image_data"] = image_base64
                    
                    # Call FastAPI tools-workflow endpoint (new recommended approach)
                    response = requests.post(
                        f"{API_BASE_URL}/api/banner/tools-workflow",
                        json=workflow_request,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get("success"):
                            st.success("✅ Banner generated successfully!")
                            
                            # Display generated banner
                            if result.get("image_data"):
                                image_bytes = base64.b64decode(result["image_data"])
                                st.image(image_bytes, caption="Generated Banner", use_container_width=True)
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Banner",
                                    data=image_bytes,
                                    file_name=f"banner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png"
                                )
                            
                            # Show processing info
                            with st.expander("Processing Details"):
                                processing_details = {
                                    "processing_time": f"{result.get('processing_time', 0):.2f}s",
                                    "size": result.get("size", selected_size),
                                    "model_used": result.get("model_used", "gpt-image-1")
                                }
                                
                                # Add workflow data if available
                                workflow_data = result.get("workflow_data", {})
                                if workflow_data:
                                    processing_details.update({
                                        "layout_extracted": bool(workflow_data.get("layout_data")),
                                        "prompt_generated": bool(workflow_data.get("generated_prompt")),
                                        "image_generated": bool(workflow_data.get("image_data")),
                                        "text_overlays_added": bool(workflow_data.get("final_image_data"))
                                    })
                                
                                st.json(processing_details)
                            
                            # Show all workflow steps data
                            workflow_data = result.get("workflow_data", {})
                            if workflow_data:
                                st.markdown("---")
                                st.markdown("### 🔄 Workflow Steps")
                                
                                # Create columns for the 4 steps
                                col1, col2 = st.columns(2)
                                
                                # Step 1: Layout Extraction (JSON)
                                with col1:
                                    st.markdown("#### Step 1: Layout Extraction")
                                    if workflow_data.get("layout_data"):
                                        with st.expander("📋 Layout Data (JSON)", expanded=True):
                                            st.json(workflow_data["layout_data"])
                                    else:
                                        st.info("No reference image provided - skipped layout extraction")
                                
                                # Step 2: Prompt Generation (JSON)
                                with col2:
                                    st.markdown("#### Step 2: Prompt Generation")
                                    if workflow_data.get("generated_prompt"):
                                        with st.expander("✍️ Generated Prompt", expanded=True):
                                            st.text_area(
                                                "Prompt sent to image generator:",
                                                value=workflow_data["generated_prompt"],
                                                height=200,
                                                disabled=True
                                            )
                                    else:
                                        st.warning("Prompt generation data not available")
                                
                                # Step 3: Generated Banner (Image)
                                with col1:
                                    st.markdown("#### Step 3: Banner Generation")
                                    if workflow_data.get("image_data"):
                                        try:
                                            step3_image = base64.b64decode(workflow_data["image_data"])
                                            st.image(step3_image, caption="Generated Banner (without text)", use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Error displaying Step 3 image: {e}")
                                    else:
                                        st.warning("Banner generation data not available")
                                
                                # Step 4: Final Banner with Text (Image)
                                with col2:
                                    st.markdown("#### Step 4: Text Overlay")
                                    if workflow_data.get("final_image_data"):
                                        try:
                                            step4_image = base64.b64decode(workflow_data["final_image_data"])
                                            st.image(step4_image, caption="Final Banner (with text)", use_container_width=True)
                                            
                                            # Show text elements if available
                                            if workflow_data.get("text_elements"):
                                                with st.expander("📝 Text Elements Added"):
                                                    st.json(workflow_data["text_elements"])
                                        except Exception as e:
                                            st.error(f"Error displaying Step 4 image: {e}")
                                    else:
                                        st.info("Text overlay step completed - using original image")
                        else:
                            st.error(f"❌ Banner generation failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"❌ API request failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error generating banner: {str(e)}")



if __name__ == "__main__":
    main()