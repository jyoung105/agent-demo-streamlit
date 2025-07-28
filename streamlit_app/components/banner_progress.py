"""
Banner progress tracking component.
"""
import streamlit as st
import time
from typing import Dict, Any, List
from services.banner_api import BannerWorkflowTracker


def render_workflow_progress(tracker: BannerWorkflowTracker) -> None:
    """
    Render workflow progress visualization.
    
    Args:
        tracker: BannerWorkflowTracker instance
    """
    if not tracker.is_running and not tracker.is_complete:
        return
    
    st.subheader("🔄 Banner Creation Progress")
    
    # Progress bar
    progress_percentage = tracker.get_progress_percentage()
    progress_bar = st.progress(progress_percentage / 100)
    
    # Overall status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Progress", f"{progress_percentage}%")
    with col2:
        elapsed_time = tracker.get_elapsed_time()
        st.metric("Elapsed Time", f"{elapsed_time:.1f}s")
    with col3:
        if tracker.is_complete:
            st.metric("Status", "✅ Complete")
        elif tracker.is_running:
            st.metric("Status", "🔄 Running")
        else:
            st.metric("Status", "⏸️ Pending")
    
    # Step details
    st.markdown("### Step Details")
    
    for i, step in enumerate(tracker.steps):
        step_name = step["name"]
        step_status = step["status"]
        step_message = step.get("message", "")
        
        # Status icon
        if step_status == "completed":
            status_icon = "✅"
            status_color = "green"
        elif step_status == "in_progress":
            status_icon = "🔄"
            status_color = "blue"
        elif step_status == "failed":
            status_icon = "❌"
            status_color = "red"
        else:  # pending
            status_icon = "⏳"
            status_color = "gray"
        
        # Render step
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**Step {i+1}**")
            with col2:
                st.markdown(f"{status_icon} **{step_name}**")
                if step_message:
                    st.caption(step_message)
                
                # Show loading animation for in-progress step
                if step_status == "in_progress":
                    with st.empty():
                        for _ in range(3):
                            st.markdown("🔄 Processing...")
                            time.sleep(0.5)
                            st.markdown("🔄 Processing")
                            time.sleep(0.5)


def render_step_by_step_controls(current_step: int, max_steps: int = 4) -> Dict[str, bool]:
    """
    Render step-by-step control buttons.
    
    Args:
        current_step: Current step number (0-based)
        max_steps: Maximum number of steps
        
    Returns:
        dict: Button states
    """
    st.subheader("🎛️ Step Controls")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        extract_button = st.button(
            "1️⃣ Extract Layout",
            disabled=current_step != 0,
            help="Analyze reference image layout",
            key="step1_button"
        )
    
    with col2:
        optimize_button = st.button(
            "2️⃣ Optimize Prompt",
            disabled=current_step != 1,
            help="Optimize banner design prompt",
            key="step2_button"
        )
    
    with col3:
        background_button = st.button(
            "3️⃣ Create Background",
            disabled=current_step != 2,
            help="Generate background image",
            key="step3_button"
        )
    
    with col4:
        overlay_button = st.button(
            "4️⃣ Overlay Text",
            disabled=current_step != 3,
            help="Add text overlays to create final banner",
            key="step4_button"
        )
    
    with col5:
        reset_button = st.button(
            "🔄 Reset",
            help="Reset workflow to start over",
            key="reset_button"
        )
    
    return {
        "extract": extract_button,
        "optimize": optimize_button,
        "generate": background_button,
        "overlay": overlay_button,
        "reset": reset_button
    }


def render_workflow_status_cards(workflow_data: Dict[str, Any]) -> None:
    """
    Render workflow status as cards.
    
    Args:
        workflow_data: Workflow response data
    """
    if not workflow_data:
        return
    
    st.subheader("📊 Workflow Summary")
    
    # Overall status
    success = workflow_data.get("success", False)
    total_time = workflow_data.get("total_processing_time", 0)
    
    col1, col2 = st.columns(2)
    with col1:
        if success:
            st.success(f"✅ **Workflow Completed Successfully**")
        else:
            st.error(f"❌ **Workflow Failed**")
    
    with col2:
        st.info(f"⏱️ **Total Time:** {total_time:.2f} seconds")
    
    # Step details
    steps = workflow_data.get("steps", [])
    if steps:
        st.markdown("### Step Breakdown")
        
        for step in steps:
            step_num = step.get("step", 0)
            step_name = step.get("name", "Unknown")
            step_status = step.get("status", "unknown")
            step_time = step.get("processing_time", 0)
            step_error = step.get("error")
            
            with st.expander(f"Step {step_num}: {step_name} ({step_status})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status", step_status.title())
                with col2:
                    if step_time:
                        st.metric("Processing Time", f"{step_time:.2f}s")
                
                if step_error:
                    st.error(f"Error: {step_error}")
                elif step_status == "completed":
                    st.success("Step completed successfully!")


def render_loading_animation(message: str = "Processing...") -> None:
    """
    Render a loading animation with custom message.
    
    Args:
        message: Loading message to display
    """
    with st.empty():
        for i in range(3):
            dots = "." * (i + 1)
            st.markdown(f"🔄 {message}{dots}")
            time.sleep(0.5)
        st.markdown(f"🔄 {message}")


def show_toast_notifications(step_name: str, status: str, duration: int = 3) -> None:
    """
    Show toast-style notifications for step completion.
    
    Args:
        step_name: Name of the completed step
        status: Status (completed, failed, etc.)
        duration: How long to show the notification
    """
    if status == "completed":
        st.toast(f"✅ {step_name} completed successfully!", icon="✅")
    elif status == "failed":
        st.toast(f"❌ {step_name} failed", icon="❌")
    elif status == "in_progress":
        st.toast(f"🔄 {step_name} in progress...", icon="🔄")


def render_workflow_timeline(steps: List[Dict[str, Any]]) -> None:
    """
    Render a visual timeline of workflow steps.
    
    Args:
        steps: List of step dictionaries
    """
    st.subheader("📈 Workflow Timeline")
    
    # Create timeline visualization
    timeline_html = """
    <div style="position: relative; margin: 20px 0;">
    """
    
    for i, step in enumerate(steps):
        step_name = step.get("name", f"Step {i+1}")
        step_status = step.get("status", "pending")
        step_time = step.get("processing_time", 0)
        
        # Status color
        if step_status == "completed":
            color = "#28a745"  # Green
            icon = "✅"
        elif step_status == "in_progress":
            color = "#007bff"  # Blue
            icon = "🔄"
        elif step_status == "failed":
            color = "#dc3545"  # Red
            icon = "❌"
        else:
            color = "#6c757d"  # Gray
            icon = "⏳"
        
        timeline_html += f"""
        <div style="display: flex; align-items: center; margin: 10px 0;">
            <div style="width: 30px; height: 30px; border-radius: 50%; background-color: {color}; 
                        display: flex; align-items: center; justify-content: center; color: white; 
                        font-weight: bold; margin-right: 15px;">
                {i+1}
            </div>
            <div style="flex-grow: 1;">
                <div style="font-weight: bold; color: {color};">
                    {icon} {step_name}
                </div>
                {f'<div style="font-size: 12px; color: #666;">Processing time: {step_time:.2f}s</div>' if step_time > 0 else ''}
            </div>
        </div>
        """
        
        # Add connector line (except for last step)
        if i < len(steps) - 1:
            timeline_html += f"""
            <div style="width: 2px; height: 20px; background-color: {color}; 
                        margin-left: 14px; opacity: 0.3;"></div>
            """
    
    timeline_html += "</div>"
    
    st.markdown(timeline_html, unsafe_allow_html=True)


# ===== ENHANCED STEP-BY-STEP PROGRESS DISPLAY =====

def display_enhanced_step_progress(job_status: Dict[str, Any]) -> None:
    """
    Display enhanced step-by-step progress with rich outputs and visual indicators.
    
    Args:
        job_status: Enhanced job status from backend API
    """
    if not job_status:
        st.error("No job status available")
        return
    
    # Overall progress header
    progress_info = job_status.get("progress", {})
    overall_progress = progress_info.get("percentage", 0)
    current_step = progress_info.get("current_step")
    current_step_name = progress_info.get("current_step_name", "")
    
    # Progress bar
    st.progress(overall_progress / 100.0)
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Progress", f"{overall_progress}%")
    with col2:
        st.metric("Completed Steps", f"{progress_info.get('completed_steps', 0)}/{progress_info.get('total_steps', 4)}")
    with col3:
        if current_step:
            st.metric("Current Step", f"Step {current_step}")
        else:
            st.metric("Status", job_status.get("status", "unknown").title())
    
    # Step-by-step display
    st.subheader("📋 Workflow Progress")
    
    steps = job_status.get("steps", [])
    for step in steps:
        display_enhanced_individual_step(step)
    
    # Final results if completed
    if job_status.get("status") == "completed":
        display_enhanced_final_results(job_status)
    elif job_status.get("status") == "failed":
        display_enhanced_error_summary(job_status)


def display_enhanced_individual_step(step: Dict[str, Any]) -> None:
    """Display detailed information for an individual workflow step."""
    step_number = step.get("step_number", 0)
    step_name = step.get("step_name", "unknown")
    status = step.get("status", "pending")
    display_name = step.get("display_name", step_name.replace("_", " ").title())
    description = step.get("display_description", "Processing...")
    processing_time = step.get("processing_time", 0)
    
    # Status icon mapping
    status_icons = {
        "pending": "⏳",
        "in_progress": "🔄", 
        "completed": "✅",
        "failed": "❌"
    }
    
    # Status color mapping
    status_colors = {
        "pending": "gray",
        "in_progress": "blue",
        "completed": "green", 
        "failed": "red"
    }
    
    icon = status_icons.get(status, "❓")
    color = status_colors.get(status, "gray")
    
    # Step container
    with st.container():
        st.markdown(f"### {icon} {display_name}")
        
        # Status and timing info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"*{description}*")
        with col2:
            st.markdown(f"**Status:** :{color}[{status.upper()}]")
        with col3:
            if processing_time > 0:
                st.markdown(f"**Time:** {processing_time:.1f}s")
        
        # Step outputs (if available)
        if status in ["completed", "failed"]:
            display_enhanced_step_outputs(step)
        
        st.divider()


def display_enhanced_step_outputs(step: Dict[str, Any]) -> None:
    """Display detailed outputs for a completed step."""
    step_number = step.get("step_number", 0)
    output_summary = step.get("output_summary", {})
    detailed_output = step.get("detailed_output", {})
    
    if not output_summary and not detailed_output:
        return
    
    # Expandable section for step details
    with st.expander(f"📊 Step {step_number} Results", expanded=True):
        
        if step_number == 1:  # Layout extraction
            display_layout_extraction_results(output_summary, detailed_output)
        elif step_number == 2:  # Prompt generation
            display_prompt_generation_results(output_summary, detailed_output)
        elif step_number == 3:  # Background image generation
            display_background_generation_results(output_summary, detailed_output)
        elif step_number == 4:  # Text overlay
            display_text_overlay_results(output_summary, detailed_output)


def display_layout_extraction_results(summary: Dict[str, Any], details: Dict[str, Any]) -> None:
    """Display layout extraction step output."""
    if summary:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Text Elements", summary.get("text_elements_count", 0))
        with col2:
            st.metric("Style", summary.get("background_style", "unknown"))
        with col3:
            st.metric("Model", summary.get("processing_model", "unknown"))
        
        # Scene description
        scene = summary.get("scene_description", "")
        if scene and scene != "No scene":
            st.markdown(f"**Scene:** {scene}")
        
        # Color palette
        colors = summary.get("color_palette", [])
        if colors:
            st.markdown("**Color Palette:**")
            color_cols = st.columns(len(colors))
            for i, color in enumerate(colors):
                with color_cols[i]:
                    st.markdown(f'<div style="background-color: {color}; padding: 10px; text-align: center; border-radius: 5px; color: white; font-weight: bold;">{color}</div>', unsafe_allow_html=True)
    
    # Detailed text elements
    if details and "text_elements" in details:
        text_elements = details["text_elements"]
        if text_elements:
            st.markdown("**📝 Extracted Text Elements:**")
            for i, elem in enumerate(text_elements):
                with st.expander(f"Text Element {i+1}: '{elem.get('description', 'Unknown')}'"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            "description": elem.get("description", ""),
                            "font_size": elem.get("font_size", 0),
                            "font_style": elem.get("font_style", ""),
                            "font_color": elem.get("font_color", "")
                        })
                    with col2:
                        bbox = elem.get("bbox", [0, 0, 0, 0])
                        st.json({
                            "position": {"x": bbox[0], "y": bbox[1]},
                            "size": {"width": bbox[2], "height": bbox[3]}
                        })


def display_prompt_generation_results(summary: Dict[str, Any], details: Dict[str, Any]) -> None:
    """Display prompt generation step output."""
    if summary:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Style", summary.get("optimized_style", "unknown"))
        with col2:
            st.metric("Mood", summary.get("optimized_mood", "unknown"))
        with col3:
            st.metric("Prompt Length", f"{summary.get('prompt_length', 0)} chars")
        
        # Prompt preview
        preview = summary.get("prompt_preview", "")
        if preview:
            st.markdown("**🎯 Generated Prompt Preview:**")
            st.code(preview, language="text")
    
    # Full prompt details
    if details and "full_prompt" in details:
        with st.expander("📝 Full Generated Prompt"):
            st.code(details["full_prompt"], language="text")
    
    # Optimized scene data
    if details and "optimized_scene" in details:
        with st.expander("🎨 Optimized Scene Data"):
            st.json(details["optimized_scene"])


def display_background_generation_results(summary: Dict[str, Any], details: Dict[str, Any]) -> None:
    """Display background image generation step output."""
    if summary:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Image Size", summary.get("image_size", "unknown"))
        with col2:
            st.metric("Format", summary.get("image_format", "unknown"))
        with col3:
            st.metric("Data Size", f"{summary.get('image_data_size_kb', 0)} KB")
        
        st.metric("Model", summary.get("processing_model", "unknown"))
    
    # Display generated background image
    if details and details.get("image_data"):
        st.markdown("**🖼️ Generated Background:**")
        try:
            import base64
            image_bytes = base64.b64decode(details["image_data"])
            st.image(image_bytes, caption="Generated Background Image", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to display image: {e}")
    
    # Generation settings
    if details and "generation_settings" in details:
        with st.expander("⚙️ Generation Settings"):
            st.json(details["generation_settings"])


def display_text_overlay_results(summary: Dict[str, Any], details: Dict[str, Any]) -> None:
    """Display text overlay step output."""
    if summary:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Text Elements", summary.get("text_elements_count", 0))
        with col2:
            st.metric("Overlay Engine", summary.get("overlay_engine", "unknown"))
        with col3:
            st.metric("Final Size", f"{summary.get('final_image_size_kb', 0)} KB")
        
        # Quality score if available
        quality_score = summary.get("text_quality_score")
        if quality_score is not None:
            st.metric("Quality Score", f"{quality_score:.2f}")
    
    # Display final composite image with text overlays
    if details and details.get("final_image_data"):
        st.markdown("**🎨 Final Banner with Text:**")
        try:
            import base64
            image_bytes = base64.b64decode(details["final_image_data"])
            st.image(image_bytes, caption="Final Banner with Text Overlays", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to display final image: {e}")
    
    # Text elements details
    if details and "text_elements" in details:
        text_elements = details["text_elements"]
        if text_elements:
            st.markdown("**📝 Text Overlay Details:**")
            for i, elem in enumerate(text_elements):
                with st.expander(f"Text Element {i+1}: '{elem.get('description', 'Unknown')}'"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            "description": elem.get("description", ""),
                            "font_size": elem.get("font_size", 0),
                            "font_style": elem.get("font_style", ""),
                            "font_color": elem.get("font_color", "")
                        })
                    with col2:
                        bbox = elem.get("bbox", [0, 0, 0, 0])
                        st.json({
                            "position": {"x": bbox[0], "y": bbox[1]},
                            "size": {"width": bbox[2], "height": bbox[3]}
                        })
    
    # Font settings
    if details and "font_settings" in details:
        with st.expander("🔤 Font Settings"):
            st.json(details["font_settings"])


def display_enhanced_final_results(job_status: Dict[str, Any]) -> None:
    """Display final workflow results and summary."""
    final_summary = job_status.get("final_summary", {})
    
    if not final_summary:
        return
    
    st.success("🎉 Banner Generation Completed Successfully!")
    
    # Performance metrics
    with st.expander("📊 Performance Summary", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Time", f"{final_summary.get('total_processing_time', 0):.1f}s")
            st.metric("Final Image", "✅ Ready" if final_summary.get('has_final_image') else "❌ Missing")
        
        with col2:
            step_breakdown = final_summary.get("step_breakdown", [])
            if step_breakdown:
                st.markdown("**Step Timing:**")
                for step in step_breakdown:
                    st.markdown(f"• {step['name']}: {step['time']:.1f}s ({step['status']})")


def display_enhanced_error_summary(job_status: Dict[str, Any]) -> None:
    """Display error summary for failed jobs."""
    final_summary = job_status.get("final_summary", {})
    
    if not final_summary:
        return
    
    st.error("❌ Banner Generation Failed")
    
    error_msg = final_summary.get("error", "Unknown error")
    st.markdown(f"**Error:** {error_msg}")
    
    failed_step = final_summary.get("failed_at_step")
    if failed_step:
        st.markdown(f"**Failed at:** Step {failed_step['step_number']} - {failed_step['step_name']}")
        if failed_step.get('error'):
            st.code(failed_step['error'], language="text")


def poll_job_status_with_display(api_client, job_id: str, polling_interval: float = 2.0) -> Dict[str, Any]:
    """
    Poll job status with live progress display until completion or failure.
    
    Args:
        api_client: BannerAPIClient instance
        job_id: Job ID to monitor
        polling_interval: Seconds between status checks
        
    Returns:
        Final job status
    """
    placeholder = st.empty()
    status_info = st.empty()
    
    st.info(f"🔄 Monitoring job: `{job_id}` (Updates every {polling_interval}s)")
    
    while True:
        status_response = api_client.get_job_status(job_id)
        
        if not status_response["success"]:
            st.error(f"Failed to get job status: {status_response['error']}")
            return {"status": "error", "error": status_response["error"]}
        
        job_data = status_response["data"]
        job_status = job_data.get("status", "unknown")
        
        # Display current status
        with placeholder.container():
            display_enhanced_step_progress(job_data)
        
        # Show last update time
        with status_info.container():
            st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")
        
        # Check if job is complete
        if job_status in ["completed", "failed"]:
            return job_data
        
        # Wait before next poll
        time.sleep(polling_interval)