#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import io, filters, exposure, segmentation, measure, feature
from skimage.morphology import disk, watershed, closing
from skimage.color import label2rgb
from skimage.future import graph
from skimage.measure import regionprops
from PIL import Image, ImageTk

class NuclearFociAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Nuclear Foci Analyzer")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.current_image = None
        self.original_display = None
        self.nucleus_channel = None
        self.green_channel = None
        self.nuclei_mask = None
        self.foci_coordinates = None
        self.nuclei_data = None
        
        # UI setup
        self.setup_ui()
    
    def setup_ui(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.results_frame = ttk.Frame(self.root, padding=10, width=300)
        self.results_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Setup control panel
        self.setup_control_panel()
        
        # Setup image display
        self.setup_image_display()
        
        # Setup results panel
        self.setup_results_panel()

    def setup_control_panel(self):
        # File operations
        file_frame = ttk.LabelFrame(self.control_frame, text="File Operations", padding=5)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=2)
        
        # Channel selection
        channel_frame = ttk.LabelFrame(self.control_frame, text="Channel Selection", padding=5)
        channel_frame.pack(fill=tk.X, pady=5)
        
        self.nucleus_channel_var = tk.StringVar(value="blue")
        ttk.Label(channel_frame, text="Nucleus Channel:").pack(anchor=tk.W)
        ttk.Combobox(channel_frame, textvariable=self.nucleus_channel_var, 
                    values=["red", "green", "blue", "violet"], state="readonly").pack(fill=tk.X, pady=2)
        
        # Nucleus segmentation parameters
        nucleus_frame = ttk.LabelFrame(self.control_frame, text="Nucleus Segmentation", padding=5)
        nucleus_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(nucleus_frame, text="Threshold:").pack(anchor=tk.W)
        self.nucleus_threshold_var = tk.IntVar(value=128)
        ttk.Scale(nucleus_frame, from_=0, to=255, variable=self.nucleus_threshold_var, 
                orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        ttk.Label(nucleus_frame, text="Min Nucleus Size:").pack(anchor=tk.W)
        self.min_nucleus_size_var = tk.IntVar(value=500)
        ttk.Scale(nucleus_frame, from_=100, to=5000, variable=self.min_nucleus_size_var, 
                orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        ttk.Label(nucleus_frame, text="Watershed Sensitivity:").pack(anchor=tk.W)
        self.watershed_sensitivity_var = tk.DoubleVar(value=0.5)
        ttk.Scale(nucleus_frame, from_=0, to=1, variable=self.watershed_sensitivity_var, 
                orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        # Foci detection parameters
        foci_frame = ttk.LabelFrame(self.control_frame, text="Foci Detection", padding=5)
        foci_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(foci_frame, text="Min Sigma:").pack(anchor=tk.W)
        self.min_sigma_var = tk.DoubleVar(value=2.0)
        ttk.Scale(foci_frame, from_=1, to=5, variable=self.min_sigma_var, 
                orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        ttk.Label(foci_frame, text="Max Sigma:").pack(anchor=tk.W)
        self.max_sigma_var = tk.DoubleVar(value=10.0)
        ttk.Scale(foci_frame, from_=5, to=15, variable=self.max_sigma_var, 
                orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        ttk.Label(foci_frame, text="Detection Threshold:").pack(anchor=tk.W)
        self.detection_threshold_var = tk.DoubleVar(value=0.1)
        ttk.Scale(foci_frame, from_=0.01, to=0.5, variable=self.detection_threshold_var, 
                orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        # Analysis and export buttons
        analysis_frame = ttk.LabelFrame(self.control_frame, text="Analysis", padding=5)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="Run Analysis", command=self.run_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=2)

    def setup_image_display(self):
        # Create canvas for image display
        self.canvas = tk.Canvas(self.image_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add scroll bars
        h_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Placeholder text
        self.canvas.create_text(400, 300, text="Load an image to begin analysis", fill="white", font=("Arial", 16))

    def setup_results_panel(self):
        # Statistics display
        stats_frame = ttk.LabelFrame(self.results_frame, text="Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Create a frame for summary statistics
        summary_frame = ttk.Frame(stats_frame)
        summary_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(summary_frame, text="Total Nuclei:").grid(row=0, column=0, sticky=tk.W)
        self.total_nuclei_var = tk.StringVar(value="0")
        ttk.Label(summary_frame, textvariable=self.total_nuclei_var).grid(row=0, column=1, sticky=tk.E)
        
        ttk.Label(summary_frame, text="Total Foci:").grid(row=1, column=0, sticky=tk.W)
        self.total_foci_var = tk.StringVar(value="0")
        ttk.Label(summary_frame, textvariable=self.total_foci_var).grid(row=1, column=1, sticky=tk.E)
        
        ttk.Label(summary_frame, text="Avg Foci per Nucleus:").grid(row=2, column=0, sticky=tk.W)
        self.avg_foci_var = tk.StringVar(value="0")
        ttk.Label(summary_frame, textvariable=self.avg_foci_var).grid(row=2, column=1, sticky=tk.E)
        
        ttk.Label(summary_frame, text="StdDev Foci per Nucleus:").grid(row=3, column=0, sticky=tk.W)
        self.stddev_foci_var = tk.StringVar(value="0")
        ttk.Label(summary_frame, textvariable=self.stddev_foci_var).grid(row=3, column=1, sticky=tk.E)
        
        # Create a frame for the results table
        table_frame = ttk.Frame(self.results_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create Treeview for results table
        self.results_table = ttk.Treeview(table_frame, columns=("nucleus_id", "foci_count", "avg_intensity", "total_intensity"))
        self.results_table.heading("#0", text="")
        self.results_table.heading("nucleus_id", text="Nucleus ID")
        self.results_table.heading("foci_count", text="Foci Count")
        self.results_table.heading("avg_intensity", text="Avg Intensity")
        self.results_table.heading("total_intensity", text="Total Intensity")
        
        self.results_table.column("#0", width=0, stretch=tk.NO)
        self.results_table.column("nucleus_id", width=80, anchor=tk.CENTER)
        self.results_table.column("foci_count", width=80, anchor=tk.CENTER)
        self.results_table.column("avg_intensity", width=80, anchor=tk.CENTER)
        self.results_table.column("total_intensity", width=80, anchor=tk.CENTER)
        
        self.results_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar to table
        table_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.results_table.yview)
        table_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_table.configure(yscrollcommand=table_scrollbar.set)
        
        # Create a frame for the histogram
        self.histogram_frame = ttk.LabelFrame(self.results_frame, text="Foci Distribution", padding=5)
        self.histogram_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    # Core functionality methods 
    def load_image(self):
        """Load and display an image file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg")]
        )
        
        if not file_path:
            return
        
        try:
            self.image_path = file_path
            self.original_image = io.imread(file_path)
            
            # Handle different image formats
            if len(self.original_image.shape) == 2:  # Grayscale
                self.original_image = np.stack([self.original_image] * 3, axis=-1)
            elif len(self.original_image.shape) == 3 and self.original_image.shape[2] > 3:  # More than RGB
                self.original_image = self.original_image[:, :, :3]  # Take first three channels
            
            # Display the original image
            self.display_image(self.original_image)
            
            # Reset analysis results
            self.clear_results()
            
            # Enable channel selection based on image
            self.process_channels()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def process_channels(self):
        """Split image into channels"""
        if self.original_image is None:
            return
        
        # Extract channels
        if len(self.original_image.shape) == 3 and self.original_image.shape[2] >= 3:
            self.red_channel = self.original_image[:, :, 0]
            self.green_channel = self.original_image[:, :, 1]
            self.blue_channel = self.original_image[:, :, 2]
        else:
            messagebox.showerror("Error", "Image must have at least 3 channels")
            return
    
    def get_nucleus_channel(self):
        """Get the selected nucleus channel"""
        channel_name = self.nucleus_channel_var.get()
        if channel_name == "red":
            return self.red_channel
        elif channel_name == "green":
            return self.green_channel
        elif channel_name == "blue":
            return self.blue_channel
        elif channel_name == "violet" and len(self.original_image.shape) > 3 and self.original_image.shape[2] > 3:
            return self.original_image[:, :, 3]
        else:
            return self.blue_channel  # Default to blue
    
    def segment_nuclei(self):
        """Segment nuclei from the selected channel"""
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded")
            return None
        
        # Get parameters
        threshold_value = self.nucleus_threshold_var.get()
        min_size = self.min_nucleus_size_var.get()
        sensitivity = self.watershed_sensitivity_var.get()
        
        # Get nucleus channel
        nucleus_channel = self.get_nucleus_channel()
        
        try:
            # Preprocess: Gaussian blur
            blurred = filters.gaussian(nucleus_channel, sigma=1.0)
            
            # Enhance contrast using CLAHE
            clahe = exposure.equalize_adapthist(blurred)
            
            # Thresholding
            if threshold_value == 0:  # Auto threshold with Otsu
                thresh = filters.threshold_otsu(clahe)
                binary = clahe > thresh
            else:
                # Manual threshold (normalize threshold to 0-1)
                binary = clahe > (threshold_value / 255.0)
            
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
            
            # Store and return the nuclei mask
            self.nuclei_mask = labels
            return labels
            
        except Exception as e:
            messagebox.showerror("Error", f"Nucleus segmentation failed: {str(e)}")
            return None
    
    def detect_foci(self):
        """Detect foci in the green channel"""
        if self.original_image is None or self.green_channel is None:
            messagebox.showerror("Error", "No image loaded")
            return None
        
        # Get parameters
        min_sigma = self.min_sigma_var.get()
        max_sigma = self.max_sigma_var.get()
        threshold = self.detection_threshold_var.get()
        
        try:
            # Preprocess: Background subtraction
            # Use a large-radius gaussian blur as background estimate
            background = filters.gaussian(self.green_channel, sigma=50)
            bg_subtracted = self.green_channel - background
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
            
            # Store foci coordinates
            self.foci_coordinates = blobs
            return blobs
            
        except Exception as e:
            messagebox.showerror("Error", f"Foci detection failed: {str(e)}")
            return None
    
    def assign_foci_to_nuclei(self):
        """Assign each focus to a nucleus"""
        if self.nuclei_mask is None or self.foci_coordinates is None:
            return None
        
        nuclei_props = regionprops(self.nuclei_mask)
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
        height, width = self.nuclei_mask.shape
        for blob in self.foci_coordinates:
            y, x, r = blob
            
            # Ensure coordinates are within image bounds
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < width and 0 <= y_int < height:
                nucleus_id = self.nuclei_mask[y_int, x_int]
                
                if nucleus_id > 0:
                    # Find the corresponding nucleus in our data
                    for nucleus in nuclei_data:
                        if nucleus['nucleus_id'] == nucleus_id:
                            nucleus['foci_count'] += 1
                            # Get intensity from green channel
                            intensity = self.green_channel[y_int, x_int]
                            nucleus['foci_intensities'].append(intensity)
            
        # Calculate statistics for each nucleus
        for nucleus in nuclei_data:
            if nucleus['foci_count'] > 0:
                nucleus['avg_intensity'] = np.mean(nucleus['foci_intensities'])
                nucleus['total_intensity'] = np.sum(nucleus['foci_intensities'])
        
        self.nuclei_data = nuclei_data
        return nuclei_data
    
    def calculate_statistics(self):
        """Calculate overall statistics from nuclei data"""
        if self.nuclei_data is None:
            return
        
        # Count total nuclei
        total_nuclei = len(self.nuclei_data)
        
        # Count total foci
        total_foci = sum(n['foci_count'] for n in self.nuclei_data)
        
        # Calculate average foci per nucleus
        avg_foci = total_foci / total_nuclei if total_nuclei > 0 else 0
        
        # Calculate standard deviation of foci per nucleus
        foci_counts = [n['foci_count'] for n in self.nuclei_data]
        stddev_foci = np.std(foci_counts) if foci_counts else 0
        
        # Update the GUI with statistics
        self.total_nuclei_var.set(str(total_nuclei))
        self.total_foci_var.set(str(total_foci))
        self.avg_foci_var.set(f"{avg_foci:.2f}")
        self.stddev_foci_var.set(f"{stddev_foci:.2f}")
        
        # Update the results table
        self.update_results_table()
        
        # Create and display histogram
        self.plot_foci_histogram()
    
    def plot_foci_histogram(self):
        """Create a histogram of foci counts per nucleus"""
        if self.nuclei_data is None:
            return
        
        # Clear previous histogram
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()
        
        # Create histogram figure
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        
        # Get foci counts
        foci_counts = [n['foci_count'] for n in self.nuclei_data]
        
        # Determine range for histogram
        max_foci = max(foci_counts) if foci_counts else 0
        bins = max(max_foci, 10)
        
        # Create histogram
        ax.hist(foci_counts, bins=range(0, bins+2), alpha=0.7, edgecolor='black')
        ax.set_xlabel('Foci Count per Nucleus')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Foci per Nucleus')
        ax.grid(axis='y', alpha=0.75)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_results_table(self):
        """Update the results table with nuclei data"""
        # Clear existing rows
        for row in self.results_table.get_children():
            self.results_table.delete(row)
        
        # Add new data
        for nucleus in self.nuclei_data:
            self.results_table.insert(
                "", tk.END, 
                values=(
                    nucleus['nucleus_id'],
                    nucleus['foci_count'],
                    f"{nucleus['avg_intensity']:.2f}" if nucleus['foci_count'] > 0 else "0.00",
                    f"{nucleus['total_intensity']:.2f}" if nucleus['foci_count'] > 0 else "0.00"
                )
            )
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded")
            return
        
        # Step 1: Segment nuclei
        self.segment_nuclei()
        if self.nuclei_mask is None:
            return
        
        # Step 2: Detect foci
        self.detect_foci()
        if self.foci_coordinates is None:
            return
        
        # Step 3: Assign foci to nuclei
        self.assign_foci_to_nuclei()
        if self.nuclei_data is None:
            return
        
        # Step 4: Calculate statistics and update results
        self.calculate_statistics()
        
        # Step 5: Display results visually
        self.display_results()
    
    def display_image(self, image):
        """Display an image on the canvas"""
        if image is None:
            return
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            # Normalize and convert to 8-bit for display
            if image.dtype != np.uint8:
                image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
            
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Store the original display image
        self.original_display = pil_image
        
        # Create a PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Get image dimensions
        img_width, img_height = pil_image.size
        
        # Configure canvas
        self.canvas.configure(scrollregion=(0, 0, img_width, img_height))
        
        # Create image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
    
    def display_results(self):
        """Display segmentation and foci detection results"""
        if self.original_image is None or self.nuclei_mask is None or self.foci_coordinates is None:
            return
        
        # Create a copy of the original image
        result_image = self.original_image.copy()
        
        # Create colored labels for display
        overlay = label2rgb(self.nuclei_mask, bg_label=0, bg_color=(0, 0, 0))
        
        # Blend with original image
        alpha = 0.3
        blended = (1-alpha) * result_image + alpha * overlay * 255
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Draw foci as circles
        foci_image = Image.fromarray(blended)
        draw = ImageTk.Draw(foci_image)
        
        for blob in self.foci_coordinates:
            y, x, r = blob
            radius = int(r * 1.5)  # Make circle a bit larger for visibility
            
            # Convert to integer coordinates
            x, y = int(x), int(y)
            
            # Draw circle around focus
            draw.ellipse(
                [(x-radius, y-radius), (x+radius, y+radius)], 
                outline=(0, 255, 0), 
                width=2
            )
        
        # Display the result
        self.display_image(foci_image)
    
    def clear_results(self):
        """Clear all analysis results"""
        self.nuclei_mask = None
        self.foci_coordinates = None
        self.nuclei_data = None
        
        # Clear statistics
        self.total_nuclei_var.set("0")
        self.total_foci_var.set("0")
        self.avg_foci_var.set("0")
        self.stddev_foci_var.set("0")
        
        # Clear table
        for row in self.results_table.get_children():
            self.results_table.delete(row)
        
        # Clear histogram
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()
    
    def export_results(self):
        """Export analysis results to CSV and images"""
        if self.nuclei_data is None:
            messagebox.showerror("Error", "No analysis results to export")
            return
        
        # Ask for export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        try:
            # Export CSV data
            csv_path = os.path.join(export_dir, f"{base_name}_results.csv")
            
            # Prepare data for CSV
            data = []
            for nucleus in self.nuclei_data:
                data.append({
                    'Nucleus ID': nucleus['nucleus_id'],
                    'Nucleus Area': nucleus['area'],
                    'Foci Count': nucleus['foci_count'],
                    'Average Foci Intensity': nucleus['avg_intensity'] if nucleus['foci_count'] > 0 else 0,
                    'Total Foci Intensity': nucleus['total_intensity'] if nucleus['foci_count'] > 0 else 0
                })
            
            # Write to CSV
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            
            # Export processed image
            if self.tk_image:
                image_path = os.path.join(export_dir, f"{base_name}_processed.png")
                if hasattr(self, 'original_display') and self.original_display:
                    self.original_display.save(image_path)
            
            # Export histogram
            histogram_path = os.path.join(export_dir, f"{base_name}_histogram.png")
            
            # Create histogram figure
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            
            # Get foci counts
            foci_counts = [n['foci_count'] for n in self.nuclei_data]
            
            # Create histogram
            max_foci = max(foci_counts) if foci_counts else 0
            bins = max(max_foci, 10)
            
            ax.hist(foci_counts, bins=range(0, bins+2), alpha=0.7, edgecolor='black')
            ax.set_xlabel('Foci Count per Nucleus')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Foci per Nucleus')
            ax.grid(axis='y', alpha=0.75)
            
            # Save histogram
            fig.savefig(histogram_path)
            plt.close(fig)
            
            messagebox.showinfo("Export Complete", f"Results exported to {export_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = NuclearFociAnalyzer(root)
    root.mainloop() 