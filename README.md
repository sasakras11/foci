# Nuclear Foci Analyzer

A web-based tool for analyzing fluorescence microscopy images to count nuclear foci. This application helps researchers detect and count green foci within cell nuclei from microscopy images.

## Features

- Upload and analyze fluorescence microscopy images (.tif, .tiff, .png, .jpg, .jpeg)
- Automatically segment nuclei using watershed algorithm
- Detect and count green foci within nuclei
- Display results with visual markers for nuclei boundaries and detected foci
- Export results to CSV
- Simple, intuitive web interface

## Running Locally

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run nuclear_foci_analyzer_streamlit.py
   ```

## Deploying to Streamlit Cloud

1. Push this code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy the app by selecting the repository and the `nuclear_foci_analyzer_streamlit.py` file

## Usage

1. Upload an image using the file uploader
2. Select the channel that shows nuclei (usually blue)
3. Adjust the foci detection sensitivity if needed
4. Click "Count Foci" to run the analysis
5. View results and download CSV report

## Requirements

See `requirements.txt` for all dependencies 