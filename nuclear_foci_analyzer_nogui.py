#!/usr/bin/env python3
"""
Nuclear Foci Analyzer (No GUI version)
This script analyzes fluorescence microscopy images to count green foci within cell nuclei.
It works without any GUI dependencies like Tkinter, making it suitable for environments
where GUI libraries aren't available.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend to avoid any window requirements
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import io, filters, exposure, segmentation, measure, feature
from skimage.morphology import disk, closing
from skimage.segmentation import watershed  # Import watershed from segmentation module
from skimage.measure import regionprops
from skimage.color import label2rgb

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Nuclear Foci Analyzer (No GUI version)")
    
    parser.add_argument("input", help="Input image file path")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--nucleus-channel", choices=["red", "green", "blue"], default="blue",
                        help="Channel to use for nucleus segmentation (default: blue)")
    
    # Nucleus segmentation parameters
    parser.add_argument("--threshold", type=int, default=0,
                        help="Threshold for nucleus segmentation (0-255, 0 for auto)")
    parser.add_argument("--min-nucleus-size", type=int, default=500,
                        help="Minimum size for nucleus segmentation (pixels)")
    parser.add_argument("--watershed-sensitivity", type=float, default=0.5,
                        help="Sensitivity for watershed segmentation (0-1)")
    
    # Foci detection parameters
    parser.add_argument("--min-sigma", type=float, default=2.0,
                        help="Minimum sigma for blob detection")
    parser.add_argument("--max-sigma", type=float, default=10.0,
                        help="Maximum sigma for blob detection")
    parser.add_argument("--detection-threshold", type=float, default=0.1,
                        help="Threshold for blob detection")
    
    return parser.parse_args()

def load_image(file_path):
    """Load an image file"""
    try:
        image = io.imread(file_path)
        
        # Handle different image formats
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] > 3:  # More than RGB
            image = image[:, :, :3]  # Take first three channels
            
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def segment_nuclei(nucleus_channel, threshold=0, min_size=500, sensitivity=0.5):
    """Segment nuclei from the selected channel"""
    try:
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
        distance = segmentation.ndi.distance_transform_edt(closed)
        
        # Find local maxima
        local_max = feature.peak_local_max(
            distance, 
            min_distance=int(20 * sensitivity),
            indices=False,
            labels=closed
        )
        
        # Watershed segmentation
        markers = measure.label(local_max)
        labels = watershed(-distance, markers, mask=closed)
        
        # Remove small objects
        for region in regionprops(labels):
            if region.area < min_size:
                labels[labels == region.label] = 0
        
        # Re-label to ensure consecutive labels
        labels = measure.label(labels > 0)
        
        return labels
        
    except Exception as e:
        print(f"Nucleus segmentation failed: {str(e)}")
        return None

def detect_foci(green_channel, min_sigma=2.0, max_sigma=10.0, threshold=0.1):
    """Detect foci in the green channel"""
    try:
        # Preprocess: Background subtraction
        # Use a large-radius gaussian blur as background estimate
        background = filters.gaussian(green_channel, sigma=50)
        bg_subtracted = green_channel - background
        bg_subtracted[bg_subtracted < 0] = 0  # Clip negative values
        
        # Normalize
        normalized = exposure.rescale_intensity(bg_subtracted)
        
        # Use LoG blob detection
        blobs = feature.blob_log(
            normalized,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=10,
            threshold=threshold
        )
        
        return blobs
        
    except Exception as e:
        print(f"Foci detection failed: {str(e)}")
        return None

def assign_foci_to_nuclei(nuclei_mask, foci_coordinates, green_channel):
    """Assign each focus to a nucleus"""
    if nuclei_mask is None or foci_coordinates is None:
        return None
    
    nuclei_props = regionprops(nuclei_mask)
    nuclei_data = []
    
    # Initialize data for each nucleus
    for prop in nuclei_props:
        nucleus_id = prop.label
        nucleus_area = prop.area
        
        nuclei_data.append({
            'nucleus_id': nucleus_id,
            'area': nucleus_area,
            'foci_count': 0,
            'foci_intensities': [],
            'avg_intensity': 0,
            'total_intensity': 0
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
                        # Get intensity from green channel
                        intensity = green_channel[y_int, x_int]
                        nucleus['foci_intensities'].append(intensity)
        
    # Calculate statistics for each nucleus
    for nucleus in nuclei_data:
        if nucleus['foci_count'] > 0:
            nucleus['avg_intensity'] = np.mean(nucleus['foci_intensities'])
            nucleus['total_intensity'] = np.sum(nucleus['foci_intensities'])
    
    return nuclei_data

def calculate_statistics(nuclei_data):
    """Calculate overall statistics from nuclei data"""
    if not nuclei_data:
        return None
    
    # Count total nuclei
    total_nuclei = len(nuclei_data)
    
    # Count total foci
    total_foci = sum(n['foci_count'] for n in nuclei_data)
    
    # Calculate average foci per nucleus
    avg_foci = total_foci / total_nuclei if total_nuclei > 0 else 0
    
    # Calculate standard deviation of foci per nucleus
    foci_counts = [n['foci_count'] for n in nuclei_data]
    stddev_foci = np.std(foci_counts) if foci_counts else 0
    
    return {
        'total_nuclei': total_nuclei,
        'total_foci': total_foci,
        'avg_foci': avg_foci,
        'stddev_foci': stddev_foci
    }

def export_results(nuclei_data, stats, output_path):
    """Export analysis results to CSV"""
    if not nuclei_data:
        return False
    
    try:
        # Prepare data for CSV
        data = []
        for nucleus in nuclei_data:
            data.append({
                'Nucleus ID': nucleus['nucleus_id'],
                'Nucleus Area': nucleus['area'],
                'Foci Count': nucleus['foci_count'],
                'Average Foci Intensity': nucleus['avg_intensity'] if nucleus['foci_count'] > 0 else 0,
                'Total Foci Intensity': nucleus['total_intensity'] if nucleus['foci_count'] > 0 else 0
            })
        
        # Write to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        # Write summary to console
        print(f"Analysis Summary:")
        print(f"  Total Nuclei: {stats['total_nuclei']}")
        print(f"  Total Foci: {stats['total_foci']}")
        print(f"  Average Foci per Nucleus: {stats['avg_foci']:.2f}")
        print(f"  StdDev Foci per Nucleus: {stats['stddev_foci']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Failed to export results: {str(e)}")
        return False

def plot_foci_histogram(nuclei_data, output_path):
    """Create a histogram of foci counts per nucleus"""
    if not nuclei_data:
        return False
    
    try:
        # Create histogram figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        # Get foci counts
        foci_counts = [n['foci_count'] for n in nuclei_data]
        
        # Create histogram
        max_foci = max(foci_counts) if foci_counts else 0
        bins = max(max_foci, 10)
        
        ax.hist(foci_counts, bins=range(0, bins+2), alpha=0.7, edgecolor='black')
        ax.set_xlabel('Foci Count per Nucleus')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Foci per Nucleus')
        ax.grid(axis='y', alpha=0.75)
        
        # Save histogram
        fig.savefig(output_path)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"Failed to create histogram: {str(e)}")
        return False

def save_segmentation_image(original_image, nuclei_mask, output_path):
    """Save an image of the segmented nuclei"""
    try:
        # Create colored labels for display
        overlay = label2rgb(nuclei_mask, bg_label=0, bg_color=(0, 0, 0))
        
        # Blend with original image
        alpha = 0.3
        blended = (1-alpha) * original_image + alpha * overlay * 255
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Save the image
        io.imsave(output_path, blended)
        
        return True
        
    except Exception as e:
        print(f"Failed to save segmentation image: {str(e)}")
        return False

def save_visualization(original_image, nuclei_mask, foci_coordinates, output_path):
    """Save a visualization of the segmentation and detected foci"""
    try:
        # Create overlay of segmented nuclei
        overlay = np.zeros_like(original_image)
        for i in range(3):
            overlay[:, :, i] = segmentation.mark_boundaries(
                original_image[:, :, i], 
                nuclei_mask, 
                color=(1, 0, 0)
            )
        
        # Create figure for visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(overlay)
        
        # Add circles for each detected focus
        for blob in foci_coordinates:
            y, x, r = blob
            radius = r * 1.5  # Make circle a bit larger for visibility
            circle = Circle((x, y), radius, color='g', fill=False, linewidth=1)
            ax.add_patch(circle)
            
        ax.set_axis_off()
        fig.tight_layout()
        
        # Save the visualization
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"Failed to save visualization: {str(e)}")
        return False

def main():
    """Main program execution"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found.")
        sys.exit(1)
    
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            print(f"Error: Could not create output directory: {str(e)}")
            sys.exit(1)
    
    # Process the image
    print(f"Loading image: {args.input}")
    image = load_image(args.input)
    if image is None:
        sys.exit(1)
    
    # Extract channels
    print("Extracting channels...")
    if len(image.shape) == 3 and image.shape[2] >= 3:
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]
    else:
        print("Error: Image must have at least 3 channels")
        sys.exit(1)
    
    # Select nucleus channel
    if args.nucleus_channel == "red":
        nucleus_channel_data = red_channel
        print("Using RED channel for nuclei")
    elif args.nucleus_channel == "green":
        nucleus_channel_data = green_channel
        print("Using GREEN channel for nuclei")
    else:  # Default to blue
        nucleus_channel_data = blue_channel
        print("Using BLUE channel for nuclei")
    
    # Segment nuclei
    print("Segmenting nuclei...")
    nuclei_mask = segment_nuclei(
        nucleus_channel_data, 
        args.threshold, 
        args.min_nucleus_size, 
        args.watershed_sensitivity
    )
    
    if nuclei_mask is None:
        sys.exit(1)
    
    # Count number of detected nuclei
    num_nuclei = len(np.unique(nuclei_mask)) - 1  # Subtract 1 for background
    print(f"Detected {num_nuclei} nuclei")
    
    # Detect foci
    print("Detecting foci...")
    foci = detect_foci(
        green_channel,
        args.min_sigma,
        args.max_sigma,
        args.detection_threshold
    )
    
    if foci is None:
        sys.exit(1)
    
    print(f"Detected {len(foci)} foci")
    
    # Assign foci to nuclei
    print("Analyzing foci per nucleus...")
    nuclei_data = assign_foci_to_nuclei(nuclei_mask, foci, green_channel)
    if nuclei_data is None:
        sys.exit(1)
    
    # Calculate statistics
    stats = calculate_statistics(nuclei_data)
    
    # Generate file names
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    csv_path = os.path.join(args.output_dir, f"{base_name}_results.csv")
    histogram_path = os.path.join(args.output_dir, f"{base_name}_histogram.png")
    segmentation_path = os.path.join(args.output_dir, f"{base_name}_segmentation.png")
    visualization_path = os.path.join(args.output_dir, f"{base_name}_visualization.png")
    
    # Export results
    print(f"Exporting results to: {csv_path}")
    export_results(nuclei_data, stats, csv_path)
    
    # Create histogram
    print(f"Creating histogram: {histogram_path}")
    plot_foci_histogram(nuclei_data, histogram_path)
    
    # Save segmentation image
    print(f"Saving segmentation image: {segmentation_path}")
    save_segmentation_image(image, nuclei_mask, segmentation_path)
    
    # Create visualization
    print(f"Creating visualization: {visualization_path}")
    save_visualization(image, nuclei_mask, foci, visualization_path)
    
    print("Analysis complete.")
    print(f"All output files are saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 