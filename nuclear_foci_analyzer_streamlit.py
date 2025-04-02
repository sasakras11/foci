#!/usr/bin/env python3
"""
Nuclear Foci Analyzer - Streamlit Version
This web app provides a simple interface for counting green foci within cell nuclei.
"""

import os
import sys
import io as io_module
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage import io, filters, exposure, segmentation, measure, feature
from skimage.morphology import disk, closing
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage.draw import circle_perimeter as draw_circle_perimeter
import base64

# Set page configuration
st.set_page_config(
    page_title="Nuclear Foci Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for custom styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        padding-top: 0;
    }
    .stSlider {
        padding-bottom: 1rem;
    }
    .result-container {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .footnote {
        font-size: 0.8rem;
        color: #666;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Define analysis functions

def segment_nuclei(nucleus_channel, threshold=0, min_size=500, sensitivity=0.5):
    """Segment nuclei from the selected channel"""
    try:
        # Make sure nucleus_channel is float between 0 and 1 for processing
        if nucleus_channel.dtype == np.uint8:
            nucleus_channel = nucleus_channel.astype(float) / 255.0
            
        # Preprocess: Gaussian blur
        blurred = filters.gaussian(nucleus_channel, sigma=1.0)
        
        # Enhance contrast using CLAHE
        clahe = exposure.equalize_adapthist(blurred)
        
        # Thresholding
        if threshold == 0:  # Auto threshold with Otsu
            thresh = filters.threshold_otsu(clahe)
            binary = clahe > thresh
        else:
            # Manual threshold (normalize threshold to 0-1)
            binary = clahe > (threshold / 255.0)
        
        # Clean up small objects
        cleaned = segmentation.clear_border(binary)
        
        # Close gaps
        selem = disk(3)
        closed = closing(cleaned, selem)
        
        # Distance transform for watershed
        distance = ndi.distance_transform_edt(closed)
        
        # Find local maxima using a version-compatible approach
        try:
            # Try the newer scikit-image API with indices=False
            local_max = feature.peak_local_max(
                distance, 
                min_distance=int(20 * sensitivity),
                indices=False
            )
        except TypeError:
            try:
                # Older scikit-image API - create a binary mask from coordinates
                coordinates = feature.peak_local_max(
                    distance, 
                    min_distance=int(20 * sensitivity)
                )
                local_max = np.zeros_like(distance, dtype=bool)
                for coord in coordinates:
                    local_max[coord[0], coord[1]] = True
            except Exception:
                # As a last resort, try the simplest approach
                coordinates = feature.peak_local_max(
                    distance, 
                    min_distance=int(20 * sensitivity)
                )
                local_max = np.zeros_like(distance, dtype=bool)
                for i in range(len(coordinates)):
                    y, x = coordinates[i]
                    local_max[y, x] = True
        
        # Watershed segmentation
        markers = measure.label(local_max)
        labels = watershed(-distance, markers, mask=closed)
        
        # Remove small objects
        filtered_labels = np.zeros_like(labels)
        num_kept = 0
        for region in regionprops(labels):
            if region.area >= min_size:
                filtered_labels[labels == region.label] = num_kept + 1
                num_kept += 1
        
        # If no objects are left after filtering, return the original labels
        if num_kept == 0:
            filtered_labels = labels
        
        return filtered_labels
        
    except Exception as e:
        st.error(f"Nucleus segmentation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def detect_foci(image, green_channel, min_sigma=2.0, max_sigma=10.0, threshold=0.1, image_path=None):
    """Detect foci in the green channel"""
    try:
        # Ensure green channel is float between 0-1
        if green_channel.dtype == np.uint8:
            green_channel = green_channel.astype(float) / 255.0
        
        # Preprocess: Background subtraction
        background = filters.gaussian(green_channel, sigma=50)
        bg_subtracted = green_channel - background
        bg_subtracted[bg_subtracted < 0] = 0  # Clip negative values
        
        # Normalize
        normalized = exposure.rescale_intensity(bg_subtracted)
        
        # JPEG images often need denoising
        if image_path and os.path.splitext(image_path)[1].lower() in ['.jpg', '.jpeg']:
            normalized = filters.gaussian(normalized, sigma=0.5)
        
        # Create a mask to ignore white areas
        # Use all channels to identify white regions (letters, scale bars, etc.)
        r_channel = image[:, :, 0].astype(float) / 255.0 if image[:, :, 0].dtype == np.uint8 else image[:, :, 0]
        g_channel = image[:, :, 1].astype(float) / 255.0 if image[:, :, 1].dtype == np.uint8 else image[:, :, 1]
        b_channel = image[:, :, 2].astype(float) / 255.0 if image[:, :, 2].dtype == np.uint8 else image[:, :, 2]
        
        # White areas have high values in all channels
        white_mask = (r_channel > 0.8) & (g_channel > 0.8) & (b_channel > 0.8)
        
        # Also filter out areas that are not predominantly green
        # Green areas should have higher green channel values compared to red and blue
        green_dominant = (g_channel > r_channel * 1.2) & (g_channel > b_channel * 1.2)
        
        # Apply masks - exclude white areas and focus on predominantly green areas
        valid_area_mask = ~white_mask
        normalized = normalized * valid_area_mask
        
        # Use LoG blob detection with more robust parameters
        blobs = feature.blob_log(
            normalized,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=10,
            threshold=threshold,
            exclude_border=False
        )
        
        # Filter out blobs with invalid coordinates
        height, width = green_channel.shape
        valid_blobs = []
        for blob in blobs:
            y, x, r = blob
            y_int, x_int = int(y), int(x)
            
            # Only include if in bounds and not in a white area
            if (0 <= y_int < height and 0 <= x_int < width and 
                not white_mask[y_int, x_int]):
                valid_blobs.append(blob)
        
        if valid_blobs:
            return np.array(valid_blobs)
        return []
        
    except Exception as e:
        st.error(f"Foci detection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def assign_foci_to_nuclei(nuclei_mask, foci_coordinates, green_channel):
    """Assign each focus to a nucleus"""
    if nuclei_mask is None or foci_coordinates is None or len(foci_coordinates) == 0:
        return None
    
    nuclei_props = regionprops(nuclei_mask)
    nuclei_data = []
    
    # Initialize data for each nucleus
    for prop in regionprops(nuclei_mask):
        nucleus_id = prop.label
        nucleus_area = prop.area
        
        nuclei_data.append({
            'nucleus_id': nucleus_id,
            'area': nucleus_area,
            'foci_count': 0
        })
    
    # For each focus, find which nucleus it belongs to
    height, width = nuclei_mask.shape
    for blob in foci_coordinates:
        y, x, r = blob
        
        # Ensure coordinates are within image bounds
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < width and 0 <= y_int < height:
            nucleus_id = nuclei_mask[y_int, x_int]
            
            if nucleus_id > 0:
                # Find the corresponding nucleus in our data
                for nucleus in nuclei_data:
                    if nucleus['nucleus_id'] == nucleus_id:
                        nucleus['foci_count'] += 1
    
    return nuclei_data

def create_visualization(image, foci=None, nuclei_mask=None):
    """Create a visualization image with nuclei boundaries and foci markers"""
    if image is None:
        return None
        
    # Create a copy for visualization
    display_img = image.copy()
    
    if isinstance(display_img[0,0,0], np.float64):
        display_img = (display_img * 255).astype(np.uint8)
        
    # Draw nuclei boundaries if available
    if nuclei_mask is not None:
        # Create a binary edge image of the segmentation
        edges = segmentation.find_boundaries(nuclei_mask, mode='outer')
        # Apply nuclei boundaries in cyan
        display_img[edges, 0] = 0  # Red channel
        display_img[edges, 1] = 255  # Green channel
        display_img[edges, 2] = 255  # Blue channel
    
    # Draw foci if available
    if foci is not None and len(foci) > 0:
        # Draw red circles around detected foci
        for blob in foci:
            y, x, r = blob
            y_int, x_int = int(y), int(x)
            radius = int(r)
            
            # Only draw if in bounds
            if 0 <= x_int < display_img.shape[1] and 0 <= y_int < display_img.shape[0]:
                # Draw circle
                rr, cc = draw_circle_perimeter(y_int, x_int, max(1, radius), shape=display_img.shape[:2])
                display_img[rr, cc, 0] = 255  # Red channel
                display_img[rr, cc, 1] = 0    # Green channel
                display_img[rr, cc, 2] = 0    # Blue channel
    
    return display_img

def get_csv_download_link(df):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="nuclear_foci_results.csv">Download CSV File</a>'
    return href

def main():
    """Main Streamlit application"""
    st.title("Nuclear Foci Analyzer")
    st.write("Upload an image to analyze and count green foci within cell nuclei.")
    
    # Create a two-column layout
    col1, col2 = st.columns([7, 3])
    
    # Session state for managing app state
    if 'image' not in st.session_state:
        st.session_state.image = None
        st.session_state.original_file_name = None
        st.session_state.results = None
        st.session_state.nuclei_mask = None
        st.session_state.foci = None
    
    # Left column for image display
    with col1:
        # Image display area
        if st.session_state.image is not None:
            # Check if we have visualization results
            if st.session_state.results is not None and st.session_state.nuclei_mask is not None and st.session_state.foci is not None:
                # Create visualization with detection results
                visualization = create_visualization(
                    st.session_state.image,
                    st.session_state.foci,
                    st.session_state.nuclei_mask
                )
                st.image(visualization, caption="Analysis Results", use_container_width=True)
            else:
                # Show original image
                st.image(st.session_state.image, caption=st.session_state.original_file_name, use_container_width=True)
        else:
            # Placeholder when no image is loaded
            st.info("Please upload an image to begin")
    
    # Right column for controls and results
    with col2:
        # File uploader
        uploaded_file = st.file_uploader("Upload Image", type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Read the image when a file is uploaded
            try:
                # Get file details
                file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
                st.session_state.original_file_name = uploaded_file.name
                
                # Read the image using scikit-image
                bytes_data = uploaded_file.getvalue()
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
                temp_file.write(bytes_data)
                temp_file.close()
                
                image = io.imread(temp_file.name)
                
                # Handle different image formats
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image] * 3, axis=-1)
                elif len(image.shape) == 3:
                    if image.shape[2] > 3:  # More than RGB
                        image = image[:, :, :3]  # Take first three channels
                    elif image.shape[2] < 3:  # Less than RGB
                        # Expand to 3 channels
                        channels = image.shape[2]
                        expanded = np.zeros((*image.shape[:2], 3), dtype=image.dtype)
                        for i in range(channels):
                            expanded[:, :, i] = image[:, :, i]
                        for i in range(channels, 3):
                            expanded[:, :, i] = image[:, :, channels-1]  # Duplicate last channel
                        image = expanded
                
                # Store image in session state
                st.session_state.image = image
                st.session_state.image_path = temp_file.name
                st.session_state.results = None
                st.session_state.nuclei_mask = None
                st.session_state.foci = None
                
                # Force a rerun to update the image display
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to load image: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Channel selection
        st.subheader("Channel Selection")
        nucleus_channel = st.selectbox(
            "Nucleus Channel:",
            options=["red", "green", "blue"],
            index=2  # Default to blue
        )
        
        # Detection settings
        st.subheader("Detection Settings")
        detection_threshold = st.slider(
            "Foci Detection Sensitivity:",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            format="%.2f"
        )
        
        # Analysis button
        if st.button("Count Foci", key="analyze"):
            if st.session_state.image is not None:
                with st.spinner("Running analysis..."):
                    try:
                        # Use default parameters that work well for most images
                        params = {
                            'threshold': 0,  # Auto threshold with Otsu
                            'min_size': 500,
                            'sensitivity': 0.5,
                            'min_sigma': 2.0,
                            'max_sigma': 10.0,
                            'detection_threshold': detection_threshold
                        }
                        
                        # Get selected nucleus channel data
                        if nucleus_channel == "red":
                            nucleus_channel_data = st.session_state.image[:, :, 0]
                        elif nucleus_channel == "green":
                            nucleus_channel_data = st.session_state.image[:, :, 1]
                        else:  # Default to blue
                            nucleus_channel_data = st.session_state.image[:, :, 2]
                        
                        # Run nucleus segmentation
                        progress_text = st.empty()
                        progress_text.text("Segmenting nuclei...")
                        
                        nuclei_mask = segment_nuclei(
                            nucleus_channel_data,
                            params['threshold'],
                            params['min_size'],
                            params['sensitivity']
                        )
                        
                        if nuclei_mask is None:
                            st.error("Failed to segment nuclei")
                            return
                        
                        # Count nuclei
                        num_nuclei = len(np.unique(nuclei_mask)) - 1  # Subtract 1 for background
                        if num_nuclei == 0:
                            st.error("No nuclei detected")
                            return
                        
                        # Detect foci
                        progress_text.text("Detecting foci...")
                        green_channel = st.session_state.image[:, :, 1]  # Green channel
                        foci = detect_foci(
                            st.session_state.image,
                            green_channel,
                            params['min_sigma'],
                            params['max_sigma'],
                            params['detection_threshold'],
                            st.session_state.image_path
                        )
                        
                        if foci is None or len(foci) == 0:
                            st.error("No foci detected")
                            return
                        
                        # Assign foci to nuclei
                        progress_text.text("Counting foci per nucleus...")
                        nuclei_data = assign_foci_to_nuclei(nuclei_mask, foci, green_channel)
                        if nuclei_data is None:
                            st.error("Failed to assign foci to nuclei")
                            return
                        
                        # Calculate average foci per nucleus
                        total_foci = sum(n['foci_count'] for n in nuclei_data)
                        avg_foci = total_foci / num_nuclei
                        
                        # Store results in session state
                        st.session_state.results = {
                            'nuclei_data': nuclei_data,
                            'num_nuclei': num_nuclei,
                            'total_foci': total_foci,
                            'avg_foci': avg_foci
                        }
                        st.session_state.nuclei_mask = nuclei_mask
                        st.session_state.foci = foci
                        
                        # Clear progress text and force a rerun to update display
                        progress_text.empty()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        import traceback
                        traceback.print_exc()
        
        # Results display
        if st.session_state.results is not None:
            st.subheader("Results")
            
            # Display metrics
            col1_m, col2_m = st.columns(2)
            
            with col1_m:
                st.metric("Number of Nuclei", st.session_state.results['num_nuclei'])
                st.metric("Total Foci", st.session_state.results['total_foci'])
            
            with col2_m:
                st.metric("Avg Foci per Nucleus", f"{st.session_state.results['avg_foci']:.2f}")
            
            # Export results
            st.subheader("Export")
            
            # Create DataFrame
            data = []
            for nucleus in st.session_state.results['nuclei_data']:
                data.append({
                    'Nucleus ID': nucleus['nucleus_id'],
                    'Nucleus Area': nucleus['area'],
                    'Foci Count': nucleus['foci_count']
                })
            
            # Add summary row
            data.append({
                'Nucleus ID': 'TOTAL',
                'Nucleus Area': '',
                'Foci Count': st.session_state.results['total_foci']
            })
            
            data.append({
                'Nucleus ID': 'AVERAGE',
                'Nucleus Area': '',
                'Foci Count': f"{st.session_state.results['avg_foci']:.2f}"
            })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Display download link
            st.markdown(get_csv_download_link(df), unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footnote">
        <hr>
        <p>Nuclear Foci Analyzer - Web Version</p>
        <p>This tool analyzes fluorescence microscopy images to count nuclear foci.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 