# Nuclear Foci Analyzer

A Python application for analyzing fluorescence microscopy images to count green foci within cell nuclei.

## Features

- Load and process fluorescence microscopy images (TIFF, PNG, JPEG)
- Segment cell nuclei using watershed segmentation
- Detect green foci using Laplacian of Gaussian (LoG) blob detection
- Calculate statistics for each nucleus and the entire image
- Display results with visual overlays
- Export results to CSV and image files

## Requirements

- Python 3.7+
- NumPy
- pandas
- matplotlib
- scikit-image
- Pillow (PIL)
- PyQt5 (for the GUI version)

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

### PyQt5 GUI (Recommended)

The PyQt5 version provides a modern GUI that works on most platforms without Tkinter:

```
python nuclear_foci_analyzer_qt.py
```

### Command-line Interface (No GUI)

For batch processing or headless environments:

```
python nuclear_foci_analyzer_nogui.py input_image.tif output_directory
```

Optional parameters:
```
--nucleus-channel {red,green,blue}  Channel to use for nucleus segmentation (default: blue)
--threshold THRESHOLD               Threshold for nucleus segmentation (0-255, 0 for auto)
--min-nucleus-size SIZE             Minimum size for nucleus segmentation (pixels)
--watershed-sensitivity SENSITIVITY  Sensitivity for watershed segmentation (0-1)
--min-sigma MIN_SIGMA               Minimum sigma for blob detection
--max-sigma MAX_SIGMA               Maximum sigma for blob detection
--detection-threshold THRESHOLD     Threshold for blob detection
```

Example with custom parameters:
```
python nuclear_foci_analyzer_nogui.py fluorescence_image.tif results --nucleus-channel red --min-nucleus-size 800 --detection-threshold 0.05
```

### Simple Test Script

For quick analysis without parameters:

```
python test_analysis.py input_image.tif output_directory [nucleus_channel]
```

### Tkinter GUI (Legacy)

If you prefer the Tkinter version and have it properly installed:

```
python nuclear_foci_analyzer.py
```

## Output Files

When exporting results, the following files are created:

1. `*_results.csv`: Contains per-nucleus statistics
   - Nucleus ID
   - Nucleus Area (pixels)
   - Foci Count
   - Average Foci Intensity
   - Total Foci Intensity

2. `*_segmentation.png`: The original image with nuclei boundaries overlay

3. `*_visualization.png`: The original image with nuclei boundaries and foci highlighted

4. `*_histogram.png`: Distribution of foci counts per nucleus

## Troubleshooting

### GUI Issues

This application comes in multiple versions to accommodate different setups:

1. **PyQt5 Version** (`nuclear_foci_analyzer_qt.py`): Recommended for most users, works on all platforms.

2. **Command-line Version** (`nuclear_foci_analyzer_nogui.py`): Works in all environments, no GUI dependencies.

3. **Tkinter Version** (`nuclear_foci_analyzer.py`): Alternative GUI if you prefer Tkinter.

If you encounter Tkinter issues (`ModuleNotFoundError: No module named '_tkinter'`), use the PyQt5 version instead.

## Image Processing Pipeline

### Nucleus Segmentation
1. Apply Gaussian blur to reduce noise
2. Enhance contrast using CLAHE
3. Apply thresholding (manual or Otsu's method)
4. Remove small objects and border artifacts
5. Use watershed segmentation to separate touching nuclei

### Foci Detection
1. Apply background subtraction to the green channel
2. Use Laplacian of Gaussian blob detection to identify foci
3. For each detected focus, calculate position and intensity

### Analysis
1. Assign each focus to its containing nucleus
2. Calculate per-nucleus statistics
3. Generate summary statistics for the entire image
4. Create histogram of foci distribution

## License

This project is provided as open-source software. 