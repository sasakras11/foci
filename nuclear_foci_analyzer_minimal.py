#!/usr/bin/env python3
"""
Minimal Nuclear Foci Analyzer
This script provides a simple GUI for counting green foci within cell nuclei.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QSlider, QFileDialog, 
    QMessageBox, QGroupBox, QFrame, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from skimage import io, filters, exposure, segmentation, measure, feature
from skimage.morphology import disk, closing
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage.draw import circle_perimeter as draw_circle_perimeter

class AnalysisWorker(QThread):
    """Worker thread for image analysis to keep the UI responsive"""
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, image, nucleus_channel, params, image_path=None):
        super().__init__()
        self.image = image
        self.nucleus_channel = nucleus_channel
        self.params = params
        self.image_path = image_path
        self.results = None
    
    def run(self):
        try:
            # Extract parameters
            threshold = self.params['threshold']
            min_size = self.params['min_size']
            sensitivity = self.params['sensitivity']
            min_sigma = self.params['min_sigma']
            max_sigma = self.params['max_sigma']
            detection_threshold = self.params['detection_threshold']
            
            # Progress updates
            self.progress.emit("Segmenting nuclei...")
            
            # Get nucleus channel
            if self.nucleus_channel == "red":
                nucleus_channel_data = self.image[:, :, 0]
            elif self.nucleus_channel == "green":
                nucleus_channel_data = self.image[:, :, 1]
            else:  # Default to blue
                nucleus_channel_data = self.image[:, :, 2]
            
            # Segment nuclei
            nuclei_mask = self.segment_nuclei(nucleus_channel_data, threshold, min_size, sensitivity)
            if nuclei_mask is None:
                self.error.emit("Failed to segment nuclei")
                return
            
            # Count nuclei
            num_nuclei = len(np.unique(nuclei_mask)) - 1  # Subtract 1 for background
            if num_nuclei == 0:
                self.error.emit("No nuclei detected")
                return
                
            # Detect foci
            self.progress.emit("Detecting foci...")
            green_channel = self.image[:, :, 1]  # Green channel
            foci = self.detect_foci(green_channel, min_sigma, max_sigma, detection_threshold)
            if foci is None or len(foci) == 0:
                self.error.emit("No foci detected")
                return
            
            # Assign foci to nuclei
            self.progress.emit("Counting foci per nucleus...")
            nuclei_data = self.assign_foci_to_nuclei(nuclei_mask, foci, green_channel)
            if nuclei_data is None:
                self.error.emit("Failed to assign foci to nuclei")
                return
            
            # Calculate average foci per nucleus
            total_foci = sum(n['foci_count'] for n in nuclei_data)
            avg_foci = total_foci / num_nuclei
            
            # Prepare results
            self.results = {
                'nuclei_data': nuclei_data,
                'num_nuclei': num_nuclei,
                'total_foci': total_foci,
                'avg_foci': avg_foci
            }
            
            # Signal completion
            self.finished.emit(self.results)
            
        except Exception as e:
            self.error.emit(f"Analysis error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def segment_nuclei(self, nucleus_channel, threshold=0, min_size=500, sensitivity=0.5):
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
            self.error.emit(f"Nucleus segmentation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_foci(self, green_channel, min_sigma=2.0, max_sigma=10.0, threshold=0.1):
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
            if self.image_path and os.path.splitext(self.image_path)[1].lower() in ['.jpg', '.jpeg']:
                normalized = filters.gaussian(normalized, sigma=0.5)
            
            # Create a mask to ignore white areas
            # Use all channels to identify white regions (letters, scale bars, etc.)
            r_channel = self.image[:, :, 0].astype(float) / 255.0 if self.image[:, :, 0].dtype == np.uint8 else self.image[:, :, 0]
            g_channel = self.image[:, :, 1].astype(float) / 255.0 if self.image[:, :, 1].dtype == np.uint8 else self.image[:, :, 1]
            b_channel = self.image[:, :, 2].astype(float) / 255.0 if self.image[:, :, 2].dtype == np.uint8 else self.image[:, :, 2]
            
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
            self.error.emit(f"Foci detection failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def assign_foci_to_nuclei(self, nuclei_mask, foci_coordinates, green_channel):
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

class MinimalFociAnalyzer(QMainWindow):
    """Minimal UI for Nuclear Foci Analyzer that just counts dots"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal Nuclear Foci Analyzer")
        self.setGeometry(100, 100, 500, 400)
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.worker = None
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the main UI layout"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout as horizontal layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for image display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 500)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setText("Image will appear here")
        left_layout.addWidget(self.image_label)
        
        # Status label under the image
        self.status_label = QLabel("Load an image to begin")
        left_layout.addWidget(self.status_label)
        
        main_layout.addWidget(left_panel, 7)  # Image takes 70% of width
        
        # Right panel for controls and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_button)
        right_layout.addWidget(file_group)
        
        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout(channel_group)
        channel_layout.addWidget(QLabel("Nucleus Channel:"))
        self.nucleus_channel_combo = QComboBox()
        self.nucleus_channel_combo.addItems(["red", "green", "blue"])
        self.nucleus_channel_combo.setCurrentText("blue")
        channel_layout.addWidget(self.nucleus_channel_combo)
        right_layout.addWidget(channel_group)
        
        # Detection parameters
        param_group = QGroupBox("Detection Settings")
        param_layout = QVBoxLayout(param_group)
        
        # Foci detection sensitivity
        param_layout.addWidget(QLabel("Foci Detection Sensitivity:"))
        self.sensitivity_label = QLabel("0.1")
        sensitivity_layout = QHBoxLayout()
        
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(50)
        self.sensitivity_slider.setValue(10)  # Default 0.1
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(5)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity_label)
        
        sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addWidget(self.sensitivity_label)
        param_layout.addLayout(sensitivity_layout)
        
        # Add a "Reanalyze with current settings" button
        self.reanalyze_button = QPushButton("Reanalyze Current Image")
        self.reanalyze_button.clicked.connect(self.run_analysis)
        self.reanalyze_button.setEnabled(False)
        param_layout.addWidget(self.reanalyze_button)
        
        right_layout.addWidget(param_group)
        
        # Analysis button
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        self.run_button = QPushButton("Count Foci")
        self.run_button.clicked.connect(self.run_analysis)
        analysis_layout.addWidget(self.run_button)
        right_layout.addWidget(analysis_group)
        
        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        # Create labels for results
        results_layout.addWidget(QLabel("Number of Nuclei:"))
        self.nuclei_count_label = QLabel("0")
        results_layout.addWidget(self.nuclei_count_label)
        
        results_layout.addWidget(QLabel("Total Foci Count:"))
        self.foci_count_label = QLabel("0")
        results_layout.addWidget(self.foci_count_label)
        
        results_layout.addWidget(QLabel("Average Foci per Nucleus:"))
        self.avg_foci_label = QLabel("0")
        results_layout.addWidget(self.avg_foci_label)
        
        # Add export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        results_layout.addWidget(self.export_button)
        
        right_layout.addWidget(results_group)
        
        # Add right panel to main layout
        main_layout.addWidget(right_panel, 3)  # Controls take 30% of width
        
        # Adjust window size to fit 16:10 aspect ratio better
        self.setGeometry(100, 100, 1000, 625)  # 16:10 aspect ratio
    
    def load_image(self):
        """Load an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.image_path = file_path
            self.original_image = io.imread(file_path)
            
            # Handle different image formats
            if len(self.original_image.shape) == 2:  # Grayscale
                self.original_image = np.stack([self.original_image] * 3, axis=-1)
            elif len(self.original_image.shape) == 3:
                if self.original_image.shape[2] > 3:  # More than RGB
                    self.original_image = self.original_image[:, :, :3]  # Take first three channels
                elif self.original_image.shape[2] < 3:  # Less than RGB
                    # Expand to 3 channels
                    channels = self.original_image.shape[2]
                    expanded = np.zeros((*self.original_image.shape[:2], 3), dtype=self.original_image.dtype)
                    for i in range(channels):
                        expanded[:, :, i] = self.original_image[:, :, i]
                    for i in range(channels, 3):
                        expanded[:, :, i] = self.original_image[:, :, channels-1]  # Duplicate last channel
                    self.original_image = expanded
            
            # Display the loaded image
            self.display_image(self.original_image)
            
            # Reset results
            self.nuclei_count_label.setText("0")
            self.foci_count_label.setText("0")
            self.avg_foci_label.setText("0")
            self.export_button.setEnabled(False)
            
            # Enable reanalyze button if an image is loaded
            self.reanalyze_button.setEnabled(True)
            
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_image(self, image, foci=None, nuclei_mask=None):
        """Display image with optional foci markers and nuclei outlines"""
        if image is None:
            return
            
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
        
        # Convert NumPy array to QImage for display
        height, width, channels = display_img.shape
        bytesPerLine = channels * width
        qImg = QImage(display_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        
        # Create a pixmap and set it to the label
        pixmap = QPixmap.fromImage(qImg)
        
        # Scale the pixmap to fit the label while preserving aspect ratio
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
    
    def run_analysis(self):
        """Run the analysis to count foci"""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        # Get detection sensitivity from slider
        detection_threshold = self.sensitivity_slider.value() / 100.0
        
        # Use default parameters that work well for most images
        params = {
            'threshold': 0,  # Auto threshold with Otsu
            'min_size': 500,
            'sensitivity': 0.5,
            'min_sigma': 2.0,
            'max_sigma': 10.0,
            'detection_threshold': detection_threshold  # Use value from slider
        }
        
        # Get selected nucleus channel
        nucleus_channel = self.nucleus_channel_combo.currentText()
        
        # Disable UI during analysis
        self.set_ui_enabled(False)
        self.status_label.setText("Running analysis...")
        
        # Create worker thread
        self.worker = AnalysisWorker(self.original_image, nucleus_channel, params, self.image_path)
        self.worker.progress.connect(self.update_status)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.process_results)
        self.worker.start()
    
    def update_status(self, message):
        """Update status label with progress message"""
        self.status_label.setText(message)
    
    def show_error(self, message):
        """Display error message and re-enable UI"""
        QMessageBox.critical(self, "Error", message)
        self.set_ui_enabled(True)
    
    def process_results(self, results):
        """Process and display analysis results"""
        if results is None:
            self.set_ui_enabled(True)
            return
        
        # Update result labels
        self.nuclei_count_label.setText(str(results['num_nuclei']))
        self.foci_count_label.setText(str(results['total_foci']))
        self.avg_foci_label.setText(f"{results['avg_foci']:.2f}")
        
        # Create a visualization of the results
        if hasattr(self.worker, 'image') and hasattr(self.worker, 'params'):
            # Get detected foci
            green_channel = self.worker.image[:, :, 1]
            foci = self.worker.detect_foci(
                green_channel, 
                self.worker.params['min_sigma'], 
                self.worker.params['max_sigma'], 
                self.worker.params['detection_threshold']
            )
            
            # Get nucleus mask
            nucleus_channel_name = self.nucleus_channel_combo.currentText()
            if nucleus_channel_name == "red":
                nucleus_channel = self.worker.image[:, :, 0]
            elif nucleus_channel_name == "green":
                nucleus_channel = self.worker.image[:, :, 1]
            else:  # Default to blue
                nucleus_channel = self.worker.image[:, :, 2]
                
            nuclei_mask = self.worker.segment_nuclei(
                nucleus_channel,
                self.worker.params['threshold'],
                self.worker.params['min_size'],
                self.worker.params['sensitivity']
            )
            
            # Display the results
            self.display_image(self.original_image, foci, nuclei_mask)
        
        # Enable export button
        self.export_button.setEnabled(True)
        
        # Re-enable UI
        self.set_ui_enabled(True)
        self.status_label.setText("Analysis complete")
    
    def export_results(self):
        """Export analysis results to CSV"""
        if not hasattr(self.worker, 'results') or not self.worker.results:
            QMessageBox.warning(self, "Warning", "No analysis results to export")
            return
        
        # Ask for export file
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Prepare data for CSV
            data = []
            for nucleus in self.worker.results['nuclei_data']:
                data.append({
                    'Nucleus ID': nucleus['nucleus_id'],
                    'Nucleus Area': nucleus['area'],
                    'Foci Count': nucleus['foci_count']
                })
            
            # Add summary row
            data.append({
                'Nucleus ID': 'TOTAL',
                'Nucleus Area': '',
                'Foci Count': self.worker.results['total_foci']
            })
            
            data.append({
                'Nucleus ID': 'AVERAGE',
                'Nucleus Area': '',
                'Foci Count': f"{self.worker.results['avg_foci']:.2f}"
            })
            
            # Write to CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements during processing"""
        self.load_button.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        self.nucleus_channel_combo.setEnabled(enabled)
        self.sensitivity_slider.setEnabled(enabled)
        self.reanalyze_button.setEnabled(enabled and self.original_image is not None)
    
    def update_sensitivity_label(self):
        """Update the sensitivity label when slider is moved"""
        value = self.sensitivity_slider.value() / 100.0
        self.sensitivity_label.setText(f"{value:.2f}")

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern cross-platform style
    window = MinimalFociAnalyzer()
    window.show()
    sys.exit(app.exec_()) 