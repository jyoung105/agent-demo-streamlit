"""
Banner upload component for reference image handling.
"""
import streamlit as st
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def render_banner_upload() -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Render banner upload component.
    
    Returns:
        tuple: (file_data, filename, base64_data) or (None, None, None)
    """
    st.subheader("📎 Reference Image (Optional)")
    st.markdown("Upload a reference banner image to extract layout and design elements.")
    
    uploaded_file = st.file_uploader(
        "Choose a reference image",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload a banner image to analyze its layout and style",
        key="reference_image_uploader"
    )
    
    if uploaded_file is not None:
        # Validate file size (max 20MB)
        MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ File too large ({uploaded_file.size / 1024 / 1024:.1f}MB). Maximum size is 20MB.")
            return None, None, None
        
        # Display preview
        try:
            st.image(uploaded_file, caption=f"Reference: {uploaded_file.name}", use_column_width=True)
            
            # File info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col2:
                st.metric("File Type", uploaded_file.type)
            with col3:
                st.metric("File Name", uploaded_file.name[:15] + "..." if len(uploaded_file.name) > 15 else uploaded_file.name)
            
            # Convert to base64 for API
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)  # Reset for potential reuse
                
                import base64
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                
                st.success("✅ Reference image uploaded successfully!")
                return file_bytes, uploaded_file.name, base64_data
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                logger.error(f"Image processing error: {e}")
                return None, None, None
                
        except Exception as e:
            st.error(f"❌ Error displaying image: {str(e)}")
            logger.error(f"Image display error: {e}")
            return None, None, None
    
    else:
        st.info("💡 No reference image uploaded. The system will create a banner based on your text requirements only.")
        return None, None, None


def render_banner_requirements() -> Optional[str]:
    """
    Render banner requirements input component.
    
    Returns:
        str: User requirements text or None
    """
    st.subheader("✍️ Banner Requirements")
    st.markdown("Describe what kind of banner you want to create.")
    
    # Provide example prompts
    with st.expander("💡 Example Requirements"):
        st.markdown("""
        **Product Banner Examples:**
        - "Create a modern banner for Torriden hydration lotion with clean, minimal design"
        - "Design a vibrant banner for a summer sale with bright colors and energetic feel"
        - "Make a professional banner for a tech conference with futuristic elements"
        
        **Service Banner Examples:**
        - "Create a banner for a fitness gym with motivational messaging"
        - "Design a banner for a coffee shop with warm, cozy atmosphere"
        - "Make a banner for an online course with educational theme"
        
        **Event Banner Examples:**
        - "Create a banner for a music festival with dynamic, colorful design"
        - "Design a banner for a wedding photography service with elegant style"
        - "Make a banner for a holiday promotion with festive elements"
        """)
    
    requirements = st.text_area(
        "Banner Requirements",
        placeholder="Example: Create a modern banner for Torriden hydration lotion with clean, minimal design and soft blue colors. Include product image and key benefits like 'Deep Hydration' and '24H Moisture Lock'.",
        height=120,
        help="Be specific about colors, style, text content, and overall feel you want.",
        key="banner_requirements"
    )
    
    if requirements and len(requirements.strip()) > 0:
        # Character count
        char_count = len(requirements)
        if char_count > 2000:
            st.warning(f"⚠️ Requirements too long ({char_count} characters). Maximum is 2000 characters.")
            return None
        else:
            st.info(f"📝 Requirements: {char_count} characters")
            return requirements.strip()
    else:
        st.warning("⚠️ Please enter your banner requirements to continue.")
        return None


def render_banner_settings() -> dict:
    """
    Render banner generation settings.
    
    Returns:
        dict: Settings configuration
    """
    st.subheader("⚙️ Generation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        size = st.selectbox(
            "Banner Size",
            options=["1792x1024", "1024x1792", "1024x1024"],
            index=0,
            help="Choose the dimensions for your banner",
            key="banner_size"
        )
        
        # Size descriptions
        size_descriptions = {
            "1792x1024": "Landscape (16:9) - Perfect for website headers, social media covers",
            "1024x1792": "Portrait (9:16) - Great for Instagram stories, mobile banners",
            "1024x1024": "Square (1:1) - Ideal for social media posts, profile images"
        }
        st.caption(size_descriptions[size])
    
    with col2:
        workflow_mode = st.radio(
            "Generation Mode",
            options=["Complete Workflow", "Live Streaming", "Step by Step"],
            index=0,
            help="Choose your preferred workflow mode",
            key="workflow_mode"
        )
        
        if workflow_mode == "Complete Workflow":
            st.caption("🚀 Runs all 4 steps automatically: Extract → Optimize → Background → Text Overlay")
        elif workflow_mode == "Live Streaming":
            st.caption("🌊 Real-time step updates with detailed outputs and progress")
        else:
            st.caption("🔧 Manual control over each step for fine-tuning")
    
    return {
        "size": size,
        "workflow_mode": workflow_mode
    }