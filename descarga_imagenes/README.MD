# Google Earth Engine API (GEE) Documentation

This guide explains how to use the `download_gee_api.py` script and run Earth Engine operations to download images from Sentinel-2 or Landsat datasets.

## Prerequisites

1. **Install the Earth Engine Python API**:
   ```bash
   pip install earthengine-api
2. **Authenticate: Ensure you have authenticated with your Google Earth Engine account by running:**
    ```bash
    earthengine authenticate
3. **Set up your project**:
 - Replace 'ee-proyect' with the correct project name in the code.
 - Ensure the required assets (e.g., elevation models, shapefiles) are uploaded to your Earth Engine account.

4. **Script**: `download_gee_api.py`

    **Description**
    
    This script processes and exports geospatial data using Google Earth Engine. It allows downloading merged datasets clipped to a specified region of interest (ROI).

    **Features**
    - Download Sentinel-2 or Landsat images.
    - Define regions using a custom shapefile (ROI).
    - Merge multiple layers (e.g., optical data, SAR, and elevation models).
    - Export datasets directly to Google Drive.

### Prerequisites

**Install the Earth Engine Python API**
```bash
pip install earthengine-api
```

### Authenticate
   Authenticate your Google Earth Engine account:
   earthengine authenticate

### Set Up Your Project
   - Replace `'ee-proyect'` in the script with your Google Earth Engine project name.
   - Ensure your GEE assets (e.g., shapefiles, elevation models) are available in your account.

### Customizations in the Script

- **Selecting Landsat or Sentinel data**:
  The script can be customized to download imagery from either Landsat or Sentinel. Edit the respective section in the script to specify the desired satellite dataset.

- **Quadrants (ROI)**:
  Quadrants are determined by assets in your GEE account. Replace the `cuadranteID` or region asset path in the script with your desired quadrant.

- **Masks**:
  You can apply water masks or other geospatial masks using your own assets. Ensure these assets are available in your GEE account and update the script accordingly.

---

## GEE Web Script (`google_earth.js`)

### Overview
This script is designed to run directly in the Google Earth Engine Code Editor. It creates tasks for downloading images from two different time periods, calculates NDVI for both, and generates a difference image between the two NDVI layers.

### Customizations

- **Region of Interest (ROI)**:
  The area is defined by the asset provided in the script. Update the asset path to specify your desired ROI.

- **Time Periods**:
  The script processes images for two specified time ranges. Modify these ranges to match your needs.

---
