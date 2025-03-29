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
                if 0 <= int(y) < height and 0 <= int(x) < width:
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
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_button)
        main_layout.addWidget(file_group)
        
        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout(channel_group)
        channel_layout.addWidget(QLabel("Nucleus Channel:"))
        self.nucleus_channel_combo = QComboBox()
        self.nucleus_channel_combo.addItems(["red", "green", "blue"])
        self.nucleus_channel_combo.setCurrentText("blue")
        channel_layout.addWidget(self.nucleus_channel_combo)
        main_layout.addWidget(channel_group)
        
        # Analysis button
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        self.run_button = QPushButton("Count Foci")
        self.run_button.clicked.connect(self.run_analysis)
        analysis_layout.addWidget(self.run_button)
        main_layout.addWidget(analysis_group)
        
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
        
        main_layout.addWidget(results_group)
        
        # Status label
        self.status_label = QLabel("Load an image to begin")
        main_layout.addWidget(self.status_label)
    
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
            
            # Reset results
            self.nuclei_count_label.setText("0")
            self.foci_count_label.setText("0")
            self.avg_foci_label.setText("0")
            self.export_button.setEnabled(False)
            
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_analysis(self):
        """Run the analysis to count foci"""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        # Use default parameters that work well for most images
        params = {
            'threshold': 0,  # Auto threshold with Otsu
            'min_size': 500,
            'sensitivity': 0.5,
            'min_sigma': 2.0,
            'max_sigma': 10.0,
            'detection_threshold': 0.1
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

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern cross-platform style
    window = MinimalFociAnalyzer()
    window.show()
    sys.exit(app.exec_()) 