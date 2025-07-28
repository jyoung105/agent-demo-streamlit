"""
Step-by-step output visualization component for banner generation pipeline.
Displays detailed outputs from each processing step with interactive elements.
"""
import streamlit as st
import base64
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime


def render_step_output_viewer(job_status: Dict[str, Any]) -> None:
    """
    Render comprehensive step-by-step output viewer for banner generation pipeline.
    
    Args:
        job_status: Enhanced job status from FastAPI backend with step outputs
    """
    if not job_status:
        st.error("No job status data available")
        return
    
    # Header with overall progress
    st.header("🔄 Pipeline Progress & Outputs")
    
    # Overall progress metrics
    progress_info = job_status.get("progress", {})
    overall_progress = progress_info.get("percentage", 0)
    current_step = progress_info.get("current_step")
    total_steps = progress_info.get("total_steps", 4)
    
    # Progress indicator
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Progress", f"{overall_progress}%")
    with col2:
        st.metric("Current Step", f"{current_step or 'N/A'}/{total_steps}")
    with col3:
        status = job_status.get("status", "unknown")
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌"}.get(status, "❓")
        st.metric("Status", f"{status_emoji} {status.title()}")
    with col4:
        total_time = job_status.get("total_processing_time", 0)
        st.metric("Total Time", f"{total_time:.1f}s")
    
    # Progress bar
    st.progress(overall_progress / 100.0)
    
    # Step-by-step detailed outputs
    steps = job_status.get("steps", [])
    if steps:
        st.subheader("📋 Step Details & Outputs")
        
        for step in steps:
            render_individual_step_output(step)
    
    # Final results section
    if status == "completed":
        render_final_results_section(job_status)
    elif status == "failed":
        render_error_analysis_section(job_status)


def render_individual_step_output(step: Dict[str, Any]) -> None:
    """Render detailed output for an individual pipeline step."""
    step_number = step.get("step_number", 0) 
    step_name = step.get("step_name", "unknown")
    status = step.get("status", "pending")
    display_name = step.get("display_name", step_name.replace("_", " ").title())
    processing_time = step.get("processing_time", 0)
    
    # Step status icons and colors
    status_config = {
        "pending": {"icon": "⏳", "color": "gray"},
        "in_progress": {"icon": "🔄", "color": "blue"},
        "completed": {"icon": "✅", "color": "green"},
        "failed": {"icon": "❌", "color": "red"}
    }
    
    config = status_config.get(status, {"icon": "❓", "color": "gray"})
    
    # Step container with dynamic styling
    with st.container():
        # Step header
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"### {config['icon']} Step {step_number}: {display_name}")
        with col2:
            st.markdown(f"**Status:** :{config['color']}[{status.upper()}]")
        with col3:
            if processing_time > 0:
                st.markdown(f"**Time:** {processing_time:.1f}s")
        
        # Step description
        description = step.get("display_description", "Processing step...")
        st.markdown(f"*{description}*")
        
        # Step outputs based on step type
        if status in ["completed", "failed"]:
            render_step_specific_outputs(step_number, step)
        elif status == "in_progress":
            render_in_progress_indicator(step_name)
        
        st.divider()


def render_step_specific_outputs(step_number: int, step: Dict[str, Any]) -> None:
    """Render outputs specific to each step type."""
    output_summary = step.get("output_summary", {})
    detailed_output = step.get("detailed_output", {})
    
    if not output_summary and not detailed_output:
        st.info("No output data available for this step")
        return
    
    # Expandable section for step outputs
    with st.expander(f"📊 Step {step_number} Outputs & Data", expanded=True):
        
        if step_number == 1:  # Layout Extraction
            render_layout_extraction_outputs(output_summary, detailed_output)
        elif step_number == 2:  # Prompt Generation  
            render_prompt_generation_outputs(output_summary, detailed_output)
        elif step_number == 3:  # Background Generation
            render_background_generation_outputs(output_summary, detailed_output)
        elif step_number == 4:  # Text Overlay
            render_text_overlay_outputs(output_summary, detailed_output)


def render_layout_extraction_outputs(summary: Dict[str, Any], details: Dict[str, Any]) -> None:
    """Render layout extraction step outputs."""
    st.markdown("#### 🔍 Layout Analysis Results")
    
    # Summary metrics
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Text Elements", summary.get("text_elements_count", 0))
        with col2:
            st.metric("Image Dimensions", f"{summary.get('original_width', 0)}×{summary.get('original_height', 0)}")
        with col3:
            st.metric("Style Detected", summary.get("background_style", "unknown"))
        with col4:
            st.metric("Processing Model", summary.get("processing_model", "unknown"))
        
        # Scene description
        scene = summary.get("scene_description", "")
        if scene and scene != "No scene":
            st.markdown(f"**🎬 Scene Description:** {scene}")
        
        # Color palette visualization
        colors = summary.get("color_palette", [])
        if colors:
            st.markdown("**🎨 Extracted Color Palette:**")
            color_cols = st.columns(len(colors))
            for i, color in enumerate(colors):
                with color_cols[i]:
                    st.markdown(
                        f'<div style="background-color: {color}; padding: 15px; text-align: center; '
                        f'border-radius: 5px; color: white; font-weight: bold; margin: 2px;">{color}</div>',
                        unsafe_allow_html=True
                    )
    
    # Detailed text elements
    if details and "text_elements" in details:
        text_elements = details["text_elements"]
        if text_elements:
            st.markdown("#### 📝 Extracted Text Elements")
            
            for i, elem in enumerate(text_elements):
                with st.expander(f"Text Element {i+1}: '{elem.get('description', 'Unknown')}'", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Text Properties:**")
                        st.json({
                            "description": elem.get("description", ""),
                            "font_size": elem.get("font_size", 0),
                            "font_style": elem.get("font_style", ""),
                            "font_color": elem.get("font_color", "")
                        })
                    
                    with col2:
                        st.markdown("**Position & Size:**")
                        bbox = elem.get("bbox", [0, 0, 0, 0])
                        st.json({
                            "position": {"x": bbox[0], "y": bbox[1]},
                            "dimensions": {"width": bbox[2], "height": bbox[3]}
                        })
                        
                        # Visual position indicator
                        if bbox[2] > 0 and bbox[3] > 0:
                            st.markdown("**Visual Preview:**")
                            st.markdown(
                                f'<div style="border: 2px solid {elem.get("font_color", "#000")}; '
                                f'width: {min(bbox[2]//4, 200)}px; height: {min(bbox[3]//4, 50)}px; '
                                f'margin: 5px; padding: 5px; font-size: {min(elem.get("font_size", 12), 16)}px; '
                                f'color: {elem.get("font_color", "#000")}; font-weight: {elem.get("font_style", "normal")};">'
                                f'{elem.get("description", "Text")}</div>',
                                unsafe_allow_html=True
                            )


def render_prompt_generation_outputs(summary: Dict[str, Any], details: Dict[str, Any]) -> None:
    """Render prompt generation step outputs."""
    st.markdown("#### ✨ Prompt Optimization Results")
    
    # Summary metrics
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Style", summary.get("optimized_style", "unknown"))
        with col2:
            st.metric("Mood", summary.get("optimized_mood", "unknown"))
        with col3:
            st.metric("Prompt Length", f"{summary.get('prompt_length', 0)} chars")
        with col4:
            st.metric("Processing Model", summary.get("processing_model", "unknown"))
        
        # Prompt preview
        preview = summary.get("prompt_preview", "")
        if preview:
            st.markdown("**🎯 Generated Prompt Preview:**")
            st.code(preview, language="text")
    
    # Detailed outputs
    if details:
        # Full prompt
        if "full_prompt" in details:
            with st.expander("📝 Complete Generated Prompt", expanded=False):
                st.code(details["full_prompt"], language="text")
                
                # Copy to clipboard button
                st.code(f"Prompt length: {len(details['full_prompt'])} characters")
        
        # Optimized scene data
        if "optimized_scene" in details:
            with st.expander("🎨 Optimized Scene Parameters", expanded=False):
                scene_data = details["optimized_scene"]
                
                # Background parameters
                if "background" in scene_data:
                    bg = scene_data["background"]
                    st.markdown("**Background Settings:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            "scene": bg.get("scene", ""),
                            "style": bg.get("style", ""),
                            "mood": bg.get("mood", ""),
                            "lighting": bg.get("lighting", "")
                        })
                    with col2:
                        st.json({
                            "composition": bg.get("composition", ""),
                            "color_palette": bg.get("color_palette", []),
                            "camera": bg.get("camera", {})
                        })
                
                # Text optimizations
                if "text" in scene_data:
                    st.markdown("**Text Optimizations:**")
                    for i, text_elem in enumerate(scene_data["text"]):
                        st.markdown(f"**Element {i+1}:** {text_elem.get('description', 'N/A')}")


def render_background_generation_outputs(summary: Dict[str, Any], details: Dict[str, Any]) -> None:
    """Render background image generation step outputs."""
    st.markdown("#### 🖼️ Background Generation Results")
    
    # Summary metrics
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Image Size", summary.get("image_size", "unknown"))
        with col2:
            st.metric("Format", summary.get("image_format", "unknown"))
        with col3:
            st.metric("Data Size", f"{summary.get('image_data_size_kb', 0)} KB")
        with col4:
            st.metric("Generation Model", summary.get("processing_model", "unknown"))
    
    # Generated background image
    if details and details.get("image_data"):
        st.markdown("**🎨 Generated Background Image:**")
        try:
            image_bytes = base64.b64decode(details["image_data"])
            st.image(image_bytes, caption="Generated Background", use_column_width=True)
            
            # Download button for background
            st.download_button(
                label="📥 Download Background",
                data=image_bytes,
                file_name=f"background_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                key=f"download_bg_{time.time()}"
            )
        except Exception as e:
            st.error(f"Failed to display background image: {e}")
    
    # Generation settings
    if details and "generation_settings" in details:
        with st.expander("⚙️ Generation Parameters", expanded=False):
            settings = details["generation_settings"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.json({k: v for k, v in settings.items() if k in ["model", "size", "quality", "style"]})
            with col2:
                st.json({k: v for k, v in settings.items() if k not in ["model", "size", "quality", "style"]})


def render_text_overlay_outputs(summary: Dict[str, Any], details: Dict[str, Any]) -> None:
    """Render text overlay step outputs."""
    st.markdown("#### 📝 Text Overlay Results")
    
    # Summary metrics
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Text Elements", summary.get("text_elements_count", 0))
        with col2:
            st.metric("Font System", summary.get("font_system", "Pretendard"))
        with col3:
            st.metric("Final Size", f"{summary.get('final_image_size_kb', 0)} KB")
        with col4:
            overlay_engine = summary.get("overlay_engine", "PIL")
            st.metric("Overlay Engine", overlay_engine)
    
    # Final composite image
    if details and details.get("final_image_data"):
        st.markdown("**🎨 Final Banner with Text Overlays:**")
        try:
            image_bytes = base64.b64decode(details["final_image_data"])
            st.image(image_bytes, caption="Final Banner with Text", use_column_width=True)
            
            # Download button for final banner
            st.download_button(
                label="🎯 Download Final Banner",
                data=image_bytes,
                file_name=f"final_banner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                key=f"download_final_{time.time()}"
            )
        except Exception as e:
            st.error(f"Failed to display final banner: {e}")
    
    # Text overlay details
    if details and "text_elements" in details:
        text_elements = details["text_elements"]
        if text_elements:
            st.markdown("#### 🔤 Applied Text Overlays")
            
            for i, elem in enumerate(text_elements):
                with st.expander(f"Overlay {i+1}: '{elem.get('description', 'Unknown')}'", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Text Styling:**")
                        st.json({
                            "text": elem.get("description", ""),
                            "font_size": elem.get("font_size", 0),
                            "font_style": elem.get("font_style", ""),
                            "font_color": elem.get("font_color", "")
                        })
                    
                    with col2:
                        st.markdown("**Positioning:**")
                        bbox = elem.get("bbox", [0, 0, 0, 0])
                        st.json({
                            "x": bbox[0], "y": bbox[1],
                            "width": bbox[2], "height": bbox[3]
                        })
                        
                        # Font preview
                        if elem.get("font_color"):
                            st.markdown(
                                f'<div style="color: {elem.get("font_color")}; '
                                f'font-size: 16px; font-weight: {("bold" if "bold" in elem.get("font_style", "").lower() else "normal")}; '
                                f'margin: 10px 0; padding: 8px; border-left: 3px solid {elem.get("font_color")};">'
                                f'Preview: {elem.get("description", "Sample Text")}</div>',
                                unsafe_allow_html=True
                            )
    
    # Font configuration
    if details and "font_settings" in details:
        with st.expander("🔤 Font Configuration", expanded=False):
            st.json(details["font_settings"])


def render_in_progress_indicator(step_name: str) -> None:
    """Render animated progress indicator for in-progress steps."""
    with st.empty():
        progress_messages = [
            f"🔄 {step_name.replace('_', ' ').title()} in progress...",
            f"⚡ Processing {step_name.replace('_', ' ')}...",
            f"🎯 Working on {step_name.replace('_', ' ')}..."
        ]
        
        for i in range(3):
            st.markdown(progress_messages[i % len(progress_messages)])
            time.sleep(0.8)


def render_final_results_section(job_status: Dict[str, Any]) -> None:
    """Render final results summary and download options."""
    st.success("🎉 Banner Generation Completed Successfully!")
    
    final_summary = job_status.get("final_summary", {})
    
    if final_summary:
        with st.expander("📊 Performance Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_time = final_summary.get("total_processing_time", 0)
                st.metric("Total Processing Time", f"{total_time:.1f}s")
            
            with col2:
                has_final = final_summary.get("has_final_image", False)
                st.metric("Final Banner", "✅ Ready" if has_final else "❌ Missing")
            
            with col3:
                steps_completed = len([s for s in job_status.get("steps", []) if s.get("status") == "completed"])
                st.metric("Steps Completed", f"{steps_completed}/4")
            
            # Step timing breakdown
            step_breakdown = final_summary.get("step_breakdown", [])
            if step_breakdown:
                st.markdown("**⏱️ Step Timing Breakdown:**")
                
                timing_data = []
                for step in step_breakdown:
                    timing_data.append({
                        "Step": step.get("name", "Unknown"),
                        "Time (s)": f"{step.get('time', 0):.1f}",
                        "Status": step.get("status", "unknown"),
                        "Percentage": f"{(step.get('time', 0) / total_time * 100):.1f}%" if total_time > 0 else "0%"
                    })
                
                st.dataframe(timing_data, use_container_width=True)


def render_error_analysis_section(job_status: Dict[str, Any]) -> None:
    """Render detailed error analysis for failed jobs."""
    st.error("❌ Banner Generation Failed")
    
    final_summary = job_status.get("final_summary", {})
    
    if final_summary:
        error_msg = final_summary.get("error", "Unknown error occurred")
        st.markdown(f"**Primary Error:** {error_msg}")
        
        failed_step = final_summary.get("failed_at_step")
        if failed_step:
            with st.expander("🔍 Failure Details", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Failed Step:** {failed_step['step_number']} - {failed_step['step_name']}")
                    st.markdown(f"**Processing Time:** {failed_step.get('processing_time', 0):.1f}s")
                
                with col2:
                    if failed_step.get('error'):
                        st.markdown("**Error Details:**")
                        st.code(failed_step['error'], language="text")
                
                # Recovery suggestions
                st.markdown("**💡 Recovery Suggestions:**")
                step_num = failed_step.get('step_number', 0)
                
                if step_num == 1:
                    st.info("• Try uploading a different reference image with clearer text\n• Ensure image is high resolution and good quality")
                elif step_num == 2:
                    st.info("• Simplify your requirements description\n• Try more specific visual style keywords")
                elif step_num == 3:
                    st.info("• The image generation service may be temporarily unavailable\n• Try again in a few minutes")
                elif step_num == 4:
                    st.info("• Check if font files are accessible\n• Verify text element positioning data")
                
                # Retry button
                if st.button("🔄 Retry Generation", type="primary"):
                    st.rerun()


def render_comparison_view(original_image: Optional[str], final_image: Optional[str]) -> None:
    """Render side-by-side comparison of original and generated images."""
    if not original_image and not final_image:
        return
    
    st.subheader("🔄 Before & After Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Reference**")
        if original_image:
            try:
                image_bytes = base64.b64decode(original_image)
                st.image(image_bytes, caption="Original Image", use_column_width=True)
            except Exception as e:
                st.error(f"Failed to display original image: {e}")
        else:
            st.info("No reference image provided")
    
    with col2:
        st.markdown("**Generated Banner**")
        if final_image:
            try:
                image_bytes = base64.b64decode(final_image)
                st.image(image_bytes, caption="Generated Banner", use_column_width=True)
            except Exception as e:
                st.error(f"Failed to display final image: {e}")
        else:
            st.info("Banner generation not completed")


def render_step_navigation(current_step: int, total_steps: int = 4) -> int:
    """Render navigation controls for stepping through pipeline stages."""
    st.subheader("🎛️ Step Navigation")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", disabled=current_step <= 1):
            return max(1, current_step - 1)
    
    with col2:
        step = st.slider("Go to Step", min_value=1, max_value=total_steps, value=current_step)
        return step
    
    with col3:
        if st.button("Next ➡️", disabled=current_step >= total_steps):
            return min(total_steps, current_step + 1)
    
    return current_step


def render_export_options(job_status: Dict[str, Any]) -> None:
    """Render export and download options for generated content."""
    if job_status.get("status") != "completed":
        return
    
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Export Pipeline Data"):
            # Export complete pipeline data as JSON
            export_data = {
                "job_id": job_status.get("job_id"),
                "timestamp": datetime.now().isoformat(),
                "pipeline_data": job_status
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="💾 Download Pipeline Data",
                data=json_str,
                file_name=f"pipeline_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("🖼️ Export All Images"):
            st.info("Multiple image export feature coming soon!")
    
    with col3:
        if st.button("📊 Generate Report"):
            st.info("Pipeline report generation feature coming soon!")