"""
Banner gallery and results display component.
"""
import streamlit as st
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def render_banner_result(result_data: Dict[str, Any], show_metadata: bool = True) -> None:
    """
    Render a single banner generation result.
    
    Args:
        result_data: Banner generation result data
        show_metadata: Whether to show metadata and details
    """
    if not result_data:
        st.warning("No banner result to display.")
        return
    
    success = result_data.get("success", False)
    
    if not success:
        st.error(f"❌ Banner generation failed: {result_data.get('error', 'Unknown error')}")
        return
    
    # Display the generated banner
    image_data = result_data.get("image_data")
    if image_data:
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Display image
            st.image(
                image_bytes,
                caption="🎨 Generated Banner",
                use_column_width=True
            )
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"banner_{timestamp}.png"
            
            st.download_button(
                label="💾 Download Banner",
                data=image_bytes,
                file_name=filename,
                mime="image/png",
                help="Download the generated banner image"
            )
            
        except Exception as e:
            st.error(f"❌ Failed to display banner image: {str(e)}")
            logger.error(f"Image display error: {e}")
    else:
        st.warning("⚠️ No image data available in the result.")
    
    # Show metadata if requested
    if show_metadata:
        render_banner_metadata(result_data)


def render_banner_metadata(result_data: Dict[str, Any]) -> None:
    """
    Render banner generation metadata and details.
    
    Args:
        result_data: Banner generation result data
    """
    with st.expander("📊 Banner Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            size = result_data.get("size", "Unknown")
            st.metric("Size", size)
        
        with col2:
            processing_time = result_data.get("processing_time", 0)
            st.metric("Generation Time", f"{processing_time:.2f}s")
        
        with col3:
            timestamp = result_data.get("timestamp", "")
            if timestamp:
                try:
                    # Parse ISO timestamp
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime("%H:%M:%S")
                    st.metric("Generated At", formatted_time)
                except:
                    st.metric("Generated At", "Now")
        
        # Show original prompt if available
        original_prompt = result_data.get("original_prompt")
        if original_prompt:
            st.markdown("**🎯 Generation Prompt:**")
            st.text_area(
                "Prompt used for generation",
                value=original_prompt,
                height=100,
                disabled=True,
                key=f"prompt_display_{hash(original_prompt)}"
            )
        
        # Show image URL if available
        image_url = result_data.get("image_url")
        if image_url:
            st.markdown(f"**🔗 Image URL:** [View Original]({image_url})")


def render_banner_gallery(results_history: List[Dict[str, Any]]) -> None:
    """
    Render a gallery of previously generated banners.
    
    Args:
        results_history: List of banner generation results
    """
    if not results_history:
        st.info("📁 No banners generated yet. Create your first banner to see it here!")
        return
    
    st.subheader("🖼️ Banner Gallery")
    st.markdown(f"Showing {len(results_history)} generated banner(s)")
    
    # Gallery controls
    col1, col2, col3 = st.columns(3)
    with col1:
        view_mode = st.radio(
            "View Mode",
            ["Grid", "List"],
            index=0,
            key="gallery_view_mode"
        )
    with col2:
        show_details = st.checkbox(
            "Show Details",
            value=False,
            key="gallery_show_details"
        )
    with col3:
        if st.button("🗑️ Clear Gallery", key="clear_gallery"):
            st.session_state.banner_results = []
            st.rerun()
    
    if view_mode == "Grid":
        render_gallery_grid(results_history, show_details)
    else:
        render_gallery_list(results_history, show_details)


def render_gallery_grid(results_history: List[Dict[str, Any]], show_details: bool = False) -> None:
    """
    Render gallery in grid layout.
    
    Args:
        results_history: List of banner results
        show_details: Whether to show detailed information
    """
    # Show 2 banners per row
    cols_per_row = 2
    
    for i in range(0, len(results_history), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            if i + j < len(results_history):
                result = results_history[i + j]
                
                with cols[j]:
                    render_gallery_item(result, f"banner_{i+j}", show_details, compact=True)


def render_gallery_list(results_history: List[Dict[str, Any]], show_details: bool = False) -> None:
    """
    Render gallery in list layout.
    
    Args:
        results_history: List of banner results
        show_details: Whether to show detailed information
    """
    for i, result in enumerate(results_history):
        with st.container():
            render_gallery_item(result, f"banner_{i}", show_details, compact=False)
            st.divider()


def render_gallery_item(result_data: Dict[str, Any], item_key: str, 
                       show_details: bool = False, compact: bool = False) -> None:
    """
    Render a single gallery item.
    
    Args:
        result_data: Banner result data
        item_key: Unique key for this item
        show_details: Whether to show details
        compact: Whether to use compact layout
    """
    if not result_data.get("success", False):
        return
    
    image_data = result_data.get("image_data")
    if not image_data:
        return
    
    try:
        # Decode and display image
        image_bytes = base64.b64decode(image_data)
        
        if compact:
            st.image(image_bytes, use_column_width=True)
            
            # Compact info
            size = result_data.get("size", "Unknown")
            processing_time = result_data.get("processing_time", 0)
            st.caption(f"📐 {size} • ⏱️ {processing_time:.1f}s")
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"banner_{timestamp}_{item_key}.png"
            
            st.download_button(
                "💾 Download",
                data=image_bytes,
                file_name=filename,
                mime="image/png",
                key=f"download_{item_key}",
                use_container_width=True
            )
            
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image_bytes, use_column_width=True)
                
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"banner_{timestamp}_{item_key}.png"
                
                st.download_button(
                    "💾 Download Banner",
                    data=image_bytes,
                    file_name=filename,
                    mime="image/png",
                    key=f"download_{item_key}"
                )
            
            with col2:
                # Banner info
                size = result_data.get("size", "Unknown")
                processing_time = result_data.get("processing_time", 0)
                timestamp = result_data.get("timestamp", "")
                
                st.markdown("**📊 Banner Information**")
                info_cols = st.columns(3)
                with info_cols[0]:
                    st.metric("Size", size)
                with info_cols[1]:
                    st.metric("Generation Time", f"{processing_time:.2f}s")
                with info_cols[2]:
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            formatted_time = dt.strftime("%H:%M")
                            st.metric("Generated", formatted_time)
                        except:
                            st.metric("Generated", "Recently")
                
                # Show prompt if details are enabled
                if show_details:
                    original_prompt = result_data.get("original_prompt")
                    if original_prompt:
                        st.markdown("**🎯 Generation Prompt:**")
                        st.text_area(
                            "Prompt",
                            value=original_prompt[:200] + "..." if len(original_prompt) > 200 else original_prompt,
                            height=80,
                            disabled=True,
                            key=f"prompt_{item_key}"
                        )
    
    except Exception as e:
        st.error(f"❌ Failed to display banner: {str(e)}")
        logger.error(f"Gallery item display error: {e}")


def render_export_options(results_history: List[Dict[str, Any]]) -> None:
    """
    Render export options for multiple banners.
    
    Args:
        results_history: List of banner results
    """
    if not results_history:
        return
    
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📦 Download All Banners", key="download_all"):
            try:
                import zipfile
                import io
                
                # Create zip file in memory
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for i, result in enumerate(results_history):
                        if result.get("success") and result.get("image_data"):
                            image_bytes = base64.b64decode(result["image_data"])
                            filename = f"banner_{i+1}_{result.get('size', 'unknown')}.png"
                            zip_file.writestr(filename, image_bytes)
                
                zip_buffer.seek(0)
                
                # Download button for zip file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = f"banners_collection_{timestamp}.zip"
                
                st.download_button(
                    "💾 Download ZIP File",
                    data=zip_buffer.getvalue(),
                    file_name=zip_filename,
                    mime="application/zip"
                )
                
            except Exception as e:
                st.error(f"❌ Failed to create ZIP file: {str(e)}")
    
    with col2:
        if st.button("📋 Export Metadata", key="export_metadata"):
            try:
                import json
                
                # Prepare metadata (exclude image_data for size)
                metadata = []
                for i, result in enumerate(results_history):
                    if result.get("success"):
                        meta = {
                            "banner_id": i + 1,
                            "size": result.get("size"),
                            "processing_time": result.get("processing_time"),
                            "timestamp": result.get("timestamp"),
                            "original_prompt": result.get("original_prompt", "")[:500]  # Truncate long prompts
                        }
                        metadata.append(meta)
                
                metadata_json = json.dumps(metadata, indent=2)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"banner_metadata_{timestamp}.json"
                
                st.download_button(
                    "💾 Download Metadata JSON",
                    data=metadata_json,
                    file_name=filename,
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"❌ Failed to export metadata: {str(e)}")


def show_banner_stats(results_history: List[Dict[str, Any]]) -> None:
    """
    Show statistics about generated banners.
    
    Args:
        results_history: List of banner results
    """
    if not results_history:
        return
    
    successful_results = [r for r in results_history if r.get("success", False)]
    
    if not successful_results:
        return
    
    st.subheader("📈 Banner Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Banners", len(successful_results))
    
    with col2:
        avg_time = sum(r.get("processing_time", 0) for r in successful_results) / len(successful_results)
        st.metric("Avg Generation Time", f"{avg_time:.1f}s")
    
    with col3:
        sizes = [r.get("size", "") for r in successful_results]
        most_common_size = max(set(sizes), key=sizes.count) if sizes else "Unknown"
        st.metric("Most Used Size", most_common_size)
    
    with col4:
        total_time = sum(r.get("processing_time", 0) for r in successful_results)
        st.metric("Total Time Spent", f"{total_time:.1f}s")