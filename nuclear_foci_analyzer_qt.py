#!/usr/bin/env python3
"""
Nuclear Foci Analyzer - PyQt5 Version
This script provides a GUI for analyzing fluorescence microscopy images to count green foci within cell nuclei.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QSlider, QFileDialog, QTableWidget, 
    QTableWidgetItem, QMessageBox, QGroupBox, QFrame, QSplitter,
    QScrollArea, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from skimage import io, filters, exposure, segmentation, measure, feature
from skimage.morphology import disk, closing
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.draw import circle_perimeter
from scipy import ndimage as ndi  # Add this import for distance transform

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in PyQt5"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

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
                
            # Detect foci
            self.progress.emit("Detecting foci...")
            green_channel = self.image[:, :, 1]  # Green channel
            foci = self.detect_foci(green_channel, min_sigma, max_sigma, detection_threshold)
            if foci is None:
                self.error.emit("Failed to detect foci")
                return
            
            # Assign foci to nuclei
            self.progress.emit("Analyzing foci per nucleus...")
            nuclei_data = self.assign_foci_to_nuclei(nuclei_mask, foci, green_channel)
            if nuclei_data is None:
                self.error.emit("Failed to assign foci to nuclei")
                return
            
            # Calculate statistics
            stats = self.calculate_statistics(nuclei_data)
            
            # Prepare results
            self.results = {
                'nuclei_mask': nuclei_mask,
                'foci': foci,
                'nuclei_data': nuclei_data,
                'stats': stats
            }
            
            # Signal completion
            self.finished.emit(self.results)
            
        except Exception as e:
            self.error.emit(f"Analysis error: {str(e)}")
    
    def segment_nuclei(self, nucleus_channel, threshold=0, min_size=500, sensitivity=0.5):
        """Segment nuclei from the selected channel"""
        try:
            print("Starting nucleus segmentation...")
            
            # Make sure nucleus_channel is float between 0 and 1 for processing
            if nucleus_channel.dtype == np.uint8:
                nucleus_channel = nucleus_channel.astype(float) / 255.0
                
            # Preprocess: Gaussian blur
            blurred = filters.gaussian(nucleus_channel, sigma=1.0)
            print("Applied Gaussian blur")
            
            # Enhance contrast using CLAHE
            clahe = exposure.equalize_adapthist(blurred)
            print("Applied CLAHE")
            
            # Thresholding
            if threshold == 0:  # Auto threshold with Otsu
                thresh = filters.threshold_otsu(clahe)
                binary = clahe > thresh
                print(f"Applied Otsu thresholding with value {thresh:.4f}")
            else:
                # Manual threshold (normalize threshold to 0-1)
                binary = clahe > (threshold / 255.0)
                print(f"Applied manual thresholding with value {threshold/255.0:.4f}")
            
            # Clean up small objects
            cleaned = segmentation.clear_border(binary)
            print("Cleared border objects")
            
            # Close gaps
            selem = disk(3)
            closed = closing(cleaned, selem)
            print("Applied morphological closing")
            
            # Distance transform for watershed
            distance = ndi.distance_transform_edt(closed)
            print("Calculated distance transform")
            
            # Find local maxima using a version-compatible approach
            print(f"Finding local maxima with min_distance={int(20 * sensitivity)}")
            try:
                # Try the newer scikit-image API with indices=False
                local_max = feature.peak_local_max(
                    distance, 
                    min_distance=int(20 * sensitivity),
                    indices=False
                )
                print("Used peak_local_max with indices=False")
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
                    print("Used peak_local_max with coordinate conversion")
                except Exception as e:
                    # As a last resort, try the simplest approach
                    coordinates = feature.peak_local_max(
                        distance, 
                        min_distance=int(20 * sensitivity)
                    )
                    local_max = np.zeros_like(distance, dtype=bool)
                    for i in range(len(coordinates)):
                        y, x = coordinates[i]
                        local_max[y, x] = True
                    print("Used fallback peak_local_max approach")
            
            # Watershed segmentation
            markers = measure.label(local_max)
            labels = watershed(-distance, markers, mask=closed)
            print(f"Applied watershed segmentation, found {np.max(labels)} initial regions")
            
            # Remove small objects
            filtered_labels = np.zeros_like(labels)
            num_kept = 0
            for region in regionprops(labels):
                if region.area >= min_size:
                    filtered_labels[labels == region.label] = num_kept + 1
                    num_kept += 1
            
            print(f"Removed small objects, kept {num_kept} regions of size >= {min_size}")
            
            # If no objects are left after filtering, return the original labels
            if num_kept == 0:
                print("No regions large enough! Using original labels")
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
            print("Starting foci detection...")
            
            # Ensure green channel is float between 0-1
            if green_channel.dtype == np.uint8:
                green_channel = green_channel.astype(float) / 255.0
            
            print(f"Green channel: min={green_channel.min():.4f}, max={green_channel.max():.4f}, mean={green_channel.mean():.4f}")
            
            # Preprocess: Background subtraction
            # Use a large-radius gaussian blur as background estimate
            background = filters.gaussian(green_channel, sigma=50)
            bg_subtracted = green_channel - background
            bg_subtracted[bg_subtracted < 0] = 0  # Clip negative values
            print("Subtracted background")
            
            # Normalize
            normalized = exposure.rescale_intensity(bg_subtracted)
            print(f"Normalized: min={normalized.min():.4f}, max={normalized.max():.4f}, mean={normalized.mean():.4f}")
            
            # JPEG images often need denoising
            if os.path.splitext(self.image_path)[1].lower() in ['.jpg', '.jpeg']:
                print("Applying additional denoising for JPEG image")
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
            
            print(f"Detected {len(blobs)} foci")
            
            # Filter out blobs with invalid coordinates
            height, width = green_channel.shape
            valid_blobs = []
            for blob in blobs:
                y, x, r = blob
                if 0 <= int(y) < height and 0 <= int(x) < width:
                    valid_blobs.append(blob)
                else:
                    print(f"Rejected blob at ({x}, {y}) - outside image boundaries")
            
            if len(valid_blobs) < len(blobs):
                print(f"Filtered out {len(blobs) - len(valid_blobs)} invalid blobs")
                blobs = np.array(valid_blobs)
            
            return blobs
            
        except Exception as e:
            self.error.emit(f"Foci detection failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def assign_foci_to_nuclei(self, nuclei_mask, foci_coordinates, green_channel):
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
    
    def calculate_statistics(self, nuclei_data):
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

class NuclearFociAnalyzerQt(QMainWindow):
    """Main application window for Nuclear Foci Analyzer"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nuclear Foci Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.nuclei_mask = None
        self.foci_coordinates = None
        self.nuclei_data = None
        self.worker = None
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the main UI layout"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create left panel (controls)
        self.control_panel = QWidget()
        control_layout = QVBoxLayout(self.control_panel)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_button)
        control_layout.addWidget(file_group)
        
        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout(channel_group)
        channel_layout.addWidget(QLabel("Nucleus Channel:"))
        self.nucleus_channel_combo = QComboBox()
        self.nucleus_channel_combo.addItems(["red", "green", "blue"])
        self.nucleus_channel_combo.setCurrentText("blue")
        channel_layout.addWidget(self.nucleus_channel_combo)
        control_layout.addWidget(channel_group)
        
        # Nucleus segmentation parameters
        nucleus_group = QGroupBox("Nucleus Segmentation")
        nucleus_layout = QVBoxLayout(nucleus_group)
        
        nucleus_layout.addWidget(QLabel("Threshold (0 for auto):"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        nucleus_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("128")
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_label.setText(str(v)))
        nucleus_layout.addWidget(self.threshold_label)
        
        nucleus_layout.addWidget(QLabel("Min Nucleus Size:"))
        self.min_size_slider = QSlider(Qt.Horizontal)
        self.min_size_slider.setRange(100, 5000)
        self.min_size_slider.setValue(500)
        nucleus_layout.addWidget(self.min_size_slider)
        self.min_size_label = QLabel("500")
        self.min_size_slider.valueChanged.connect(lambda v: self.min_size_label.setText(str(v)))
        nucleus_layout.addWidget(self.min_size_label)
        
        nucleus_layout.addWidget(QLabel("Watershed Sensitivity:"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_slider.setValue(50)
        nucleus_layout.addWidget(self.sensitivity_slider)
        self.sensitivity_label = QLabel("0.5")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(f"{v/100:.2f}")
        )
        nucleus_layout.addWidget(self.sensitivity_label)
        
        control_layout.addWidget(nucleus_group)
        
        # Foci detection parameters
        foci_group = QGroupBox("Foci Detection")
        foci_layout = QVBoxLayout(foci_group)
        
        foci_layout.addWidget(QLabel("Min Sigma:"))
        self.min_sigma_slider = QSlider(Qt.Horizontal)
        self.min_sigma_slider.setRange(10, 50)
        self.min_sigma_slider.setValue(20)
        foci_layout.addWidget(self.min_sigma_slider)
        self.min_sigma_label = QLabel("2.0")
        self.min_sigma_slider.valueChanged.connect(
            lambda v: self.min_sigma_label.setText(f"{v/10:.1f}")
        )
        foci_layout.addWidget(self.min_sigma_label)
        
        foci_layout.addWidget(QLabel("Max Sigma:"))
        self.max_sigma_slider = QSlider(Qt.Horizontal)
        self.max_sigma_slider.setRange(50, 150)
        self.max_sigma_slider.setValue(100)
        foci_layout.addWidget(self.max_sigma_slider)
        self.max_sigma_label = QLabel("10.0")
        self.max_sigma_slider.valueChanged.connect(
            lambda v: self.max_sigma_label.setText(f"{v/10:.1f}")
        )
        foci_layout.addWidget(self.max_sigma_label)
        
        foci_layout.addWidget(QLabel("Detection Threshold:"))
        self.detection_threshold_slider = QSlider(Qt.Horizontal)
        self.detection_threshold_slider.setRange(1, 50)
        self.detection_threshold_slider.setValue(10)
        foci_layout.addWidget(self.detection_threshold_slider)
        self.detection_threshold_label = QLabel("0.1")
        self.detection_threshold_slider.valueChanged.connect(
            lambda v: self.detection_threshold_label.setText(f"{v/100:.2f}")
        )
        foci_layout.addWidget(self.detection_threshold_label)
        
        control_layout.addWidget(foci_group)
        
        # Analysis and export buttons
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        analysis_layout.addWidget(self.run_button)
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        analysis_layout.addWidget(self.export_button)
        
        control_layout.addWidget(analysis_group)
        
        # Status label
        self.status_label = QLabel("Load an image to begin")
        control_layout.addWidget(self.status_label)
        
        # Add spacer to push everything to the top
        control_layout.addStretch()
        
        # Add control panel to splitter
        splitter.addWidget(self.control_panel)
        
        # Create middle panel (image display)
        self.image_panel = QWidget()
        image_layout = QVBoxLayout(self.image_panel)
        
        # Create scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Load an image to begin analysis")
        
        scroll_area.setWidget(self.image_label)
        image_layout.addWidget(scroll_area)
        
        splitter.addWidget(self.image_panel)
        
        # Create right panel (results)
        self.results_panel = QWidget()
        results_layout = QVBoxLayout(self.results_panel)
        
        # Summary statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        stats_grid = QHBoxLayout()
        stats_grid.addWidget(QLabel("Total Nuclei:"))
        self.total_nuclei_label = QLabel("0")
        stats_grid.addWidget(self.total_nuclei_label)
        stats_layout.addLayout(stats_grid)
        
        stats_grid = QHBoxLayout()
        stats_grid.addWidget(QLabel("Total Foci:"))
        self.total_foci_label = QLabel("0")
        stats_grid.addWidget(self.total_foci_label)
        stats_layout.addLayout(stats_grid)
        
        stats_grid = QHBoxLayout()
        stats_grid.addWidget(QLabel("Avg Foci per Nucleus:"))
        self.avg_foci_label = QLabel("0")
        stats_grid.addWidget(self.avg_foci_label)
        stats_layout.addLayout(stats_grid)
        
        stats_grid = QHBoxLayout()
        stats_grid.addWidget(QLabel("StdDev Foci per Nucleus:"))
        self.stddev_foci_label = QLabel("0")
        stats_grid.addWidget(self.stddev_foci_label)
        stats_layout.addLayout(stats_grid)
        
        results_layout.addWidget(stats_group)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Nucleus ID", "Foci Count", "Avg Intensity", "Total Intensity"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        
        # Histogram
        self.histogram_group = QGroupBox("Foci Distribution")
        histogram_layout = QVBoxLayout(self.histogram_group)
        self.histogram_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        histogram_layout.addWidget(self.histogram_canvas)
        results_layout.addWidget(self.histogram_group)
        
        splitter.addWidget(self.results_panel)
        
        # Set default sizes for splitter sections
        splitter.setSizes([200, 600, 400])
    
    def load_image(self):
        """Load and display an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.image_path = file_path
            # Use scikit-image's io.imread for better format handling
            self.original_image = io.imread(file_path)
            
            # Print image info for debugging
            print(f"Image loaded: shape={self.original_image.shape}, dtype={self.original_image.dtype}")
            
            # Handle different image formats
            if len(self.original_image.shape) == 2:  # Grayscale
                print("Converting grayscale to RGB")
                self.original_image = np.stack([self.original_image] * 3, axis=-1)
            elif len(self.original_image.shape) == 3:
                if self.original_image.shape[2] > 3:  # More than RGB
                    print(f"Image has {self.original_image.shape[2]} channels, truncating to RGB")
                    self.original_image = self.original_image[:, :, :3]  # Take first three channels
                elif self.original_image.shape[2] < 3:  # Less than RGB
                    print(f"Image has only {self.original_image.shape[2]} channels, expanding to RGB")
                    # Expand to 3 channels
                    channels = self.original_image.shape[2]
                    expanded = np.zeros((*self.original_image.shape[:2], 3), dtype=self.original_image.dtype)
                    for i in range(channels):
                        expanded[:, :, i] = self.original_image[:, :, i]
                    for i in range(channels, 3):
                        expanded[:, :, i] = self.original_image[:, :, channels-1]  # Duplicate last channel
                    self.original_image = expanded
            
            # Ensure image is normalized correctly for display
            if self.original_image.dtype != np.uint8:
                print(f"Converting image from {self.original_image.dtype} to uint8")
                self.original_image = (exposure.rescale_intensity(self.original_image, out_range=(0, 255))).astype(np.uint8)
            
            # Display the original image
            self.display_image(self.original_image)
            
            # Reset analysis results
            self.clear_results()
            
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_image(self, image):
        """Display an image on the image label"""
        if image is None:
            return
        
        # Convert the image for display
        if image.dtype != np.uint8:
            image = (exposure.rescale_intensity(image, out_range=(0, 255))).astype(np.uint8)
        
        height, width, channels = image.shape
        
        # Convert numpy array to QImage - use numpy's tobytes() instead of .data
        bytes_per_line = 3 * width
        q_img = QImage(image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert QImage to QPixmap and display
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        # Prepare analysis parameters
        params = {
            'threshold': self.threshold_slider.value(),
            'min_size': self.min_size_slider.value(),
            'sensitivity': self.sensitivity_slider.value() / 100.0,
            'min_sigma': self.min_sigma_slider.value() / 10.0,
            'max_sigma': self.max_sigma_slider.value() / 10.0,
            'detection_threshold': self.detection_threshold_slider.value() / 100.0
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
        
        # Store results
        self.nuclei_mask = results['nuclei_mask']
        self.foci_coordinates = results['foci']
        self.nuclei_data = results['nuclei_data']
        stats = results['stats']
        
        # Update statistics display
        self.total_nuclei_label.setText(str(stats['total_nuclei']))
        self.total_foci_label.setText(str(stats['total_foci']))
        self.avg_foci_label.setText(f"{stats['avg_foci']:.2f}")
        self.stddev_foci_label.setText(f"{stats['stddev_foci']:.2f}")
        
        # Update results table
        self.update_results_table()
        
        # Create and display histogram
        self.plot_foci_histogram()
        
        # Display results visually
        self.display_results()
        
        # Enable export button
        self.export_button.setEnabled(True)
        
        # Re-enable UI
        self.set_ui_enabled(True)
        self.status_label.setText("Analysis complete")
    
    def update_results_table(self):
        """Update the results table with nuclei data"""
        # Clear existing rows
        self.results_table.setRowCount(0)
        
        # Add new data
        for i, nucleus in enumerate(self.nuclei_data):
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(str(nucleus['nucleus_id'])))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(nucleus['foci_count'])))
            
            avg_intensity = nucleus['avg_intensity'] if nucleus['foci_count'] > 0 else 0
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{avg_intensity:.2f}"))
            
            total_intensity = nucleus['total_intensity'] if nucleus['foci_count'] > 0 else 0
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{total_intensity:.2f}"))
    
    def plot_foci_histogram(self):
        """Create a histogram of foci counts per nucleus"""
        if self.nuclei_data is None:
            return
        
        # Clear previous plot
        self.histogram_canvas.axes.clear()
        
        # Get foci counts
        foci_counts = [n['foci_count'] for n in self.nuclei_data]
        
        # Determine range for histogram
        max_foci = max(foci_counts) if foci_counts else 0
        bins = max(max_foci, 10)
        
        # Create histogram
        self.histogram_canvas.axes.hist(foci_counts, bins=range(0, bins+2), alpha=0.7, edgecolor='black')
        self.histogram_canvas.axes.set_xlabel('Foci Count per Nucleus')
        self.histogram_canvas.axes.set_ylabel('Frequency')
        self.histogram_canvas.axes.set_title('Distribution of Foci per Nucleus')
        self.histogram_canvas.axes.grid(axis='y', alpha=0.75)
        
        # Refresh canvas
        self.histogram_canvas.draw()
    
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
        
        # Add circles for foci
        canvas = blended.copy()
        
        # Draw circles for each focus (using a more efficient approach)
        for blob in self.foci_coordinates:
            y, x, r = blob
            radius = int(r * 1.5)  # Make circle a bit larger for visibility
            
            # Convert to integer coordinates
            x, y = int(x), int(y)
            
            # Draw circle using skimage's circle_perimeter
            rr, cc = circle_perimeter(y, x, radius, shape=canvas.shape)
            canvas[rr, cc, 1] = 255  # Green channel
        
        # Display the result
        self.display_image(canvas)
    
    def draw_circle(self, y, x, radius, max_y, max_x):
        """Basic circle drawing function (simple version)"""
        # This method is replaced with skimage.draw.circle_perimeter
        # Keeping for backward compatibility
        return circle_perimeter(y, x, radius, shape=(max_y, max_x))
    
    def clear_results(self):
        """Clear all analysis results"""
        self.nuclei_mask = None
        self.foci_coordinates = None
        self.nuclei_data = None
        
        # Clear statistics
        self.total_nuclei_label.setText("0")
        self.total_foci_label.setText("0")
        self.avg_foci_label.setText("0")
        self.stddev_foci_label.setText("0")
        
        # Clear table
        self.results_table.setRowCount(0)
        
        # Clear histogram
        if hasattr(self, 'histogram_canvas'):
            self.histogram_canvas.axes.clear()
            self.histogram_canvas.draw()
        
        # Disable export button
        self.export_button.setEnabled(False)
    
    def export_results(self):
        """Export analysis results to CSV and images"""
        if self.nuclei_data is None:
            QMessageBox.warning(self, "Warning", "No analysis results to export")
            return
        
        # Ask for export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
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
            
            # Export processed image if available
            if self.image_label.pixmap() is not None:
                image_path = os.path.join(export_dir, f"{base_name}_processed.png")
                self.image_label.pixmap().save(image_path)
            
            # Create colored labels for a separate segmentation image
            if self.nuclei_mask is not None:
                # Create segmentation overlay
                segmentation_path = os.path.join(export_dir, f"{base_name}_segmentation.png")
                overlay = label2rgb(self.nuclei_mask, bg_label=0, bg_color=(0, 0, 0))
                alpha = 0.3
                blended = (1-alpha) * self.original_image + alpha * overlay * 255
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                
                # Save using skimage's io
                io.imsave(segmentation_path, blended)
            
            # Export histogram
            histogram_path = os.path.join(export_dir, f"{base_name}_histogram.png")
            self.histogram_canvas.fig.savefig(histogram_path)
            
            QMessageBox.information(self, "Export Complete", f"Results exported to {export_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements during processing"""
        self.load_button.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        self.nucleus_channel_combo.setEnabled(enabled)
        self.threshold_slider.setEnabled(enabled)
        self.min_size_slider.setEnabled(enabled)
        self.sensitivity_slider.setEnabled(enabled)
        self.min_sigma_slider.setEnabled(enabled)
        self.max_sigma_slider.setEnabled(enabled)
        self.detection_threshold_slider.setEnabled(enabled)

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern cross-platform style
    window = NuclearFociAnalyzerQt()
    window.show()
    sys.exit(app.exec_()) 