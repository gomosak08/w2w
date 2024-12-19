import time
import sys
from datetime import datetime
import ee

# Authenticate and initialize Earth Engine
try:
    ee.Authenticate()
    ee.Initialize(project='ee-proyect')
except Exception as e:
    raise RuntimeError(f"Failed to authenticate or initialize Earth Engine: {e}")

# Define band names and elevation model
bandas = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
elevacion = ee.Image('projects/ee-proyect/assets/modelo_elevacion')

# Load the SAR image and region of interest
image = ee.Image('users/bjornburrnyberg/SARL')
cuadranteID = '032038'  # Example quadrant ID
cuad_selec = ee.FeatureCollection('projects/ee-proyect/assets/division_jalisco')

# Filter Sentinel-2 imagery
dataset = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate('2020-01-01', '2022-01-30')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .select(bandas)
    .median()
)

# Clip dataset and elevation model to the region of interest
dataset_cuad = dataset.clip(cuad_selec).toFloat()
elevacion_clipped = elevacion.clip(cuad_selec).toFloat()

# Prepare the SAR image
try:
    clipped_image = image.clip(cuad_selec).select('Y2020').toFloat()
except Exception as e:
    raise RuntimeError(f"Error processing SAR image: {e}")

# Merge all layers into a single image
try:
    merged_image = dataset_cuad.addBands([clipped_image, elevacion_clipped])
except Exception as e:
    raise RuntimeError(f"Error merging images: {e}")

# Define export region
region = cuad_selec.geometry()

# Export the merged image to Google Drive
try:
    export_task = ee.batch.Export.image.toDrive(
        image=merged_image,  # Image to export
        folder='l8_2020',  # Folder in Google Drive
        description="mascara_agua_4",  # Task description
        scale=20,  # Spatial resolution
        maxPixels=1e12,  # Maximum pixels to export
        crs='EPSG:6372',  # Projection system
        region=region.bounds().getInfo()['coordinates'],  # Export region
        fileFormat='GeoTIFF'  # Output file format
    )
except Exception as e:
    raise RuntimeError(f"Error setting up export task: {e}")

# Print band names in the merged image
try:
    band_names = merged_image.bandNames()
    print("Bands in the image:", band_names.getInfo())
except Exception as e:
    raise RuntimeError(f"Error retrieving band names: {e}")

# Start the export task
export_task.start()

# Monitor the export task status
start_time = datetime.now()
last_status = None

try:
    while export_task.active():
        # Check and print task status
        status = export_task.status().get('state')
        if status != last_status:
            sys.stdout.write(f"\rExport task status: {status}")
            sys.stdout.flush()
            last_status = status
        time.sleep(30)  # Wait before checking again
except KeyboardInterrupt:
    print("\nTask monitoring interrupted by user.")
except Exception as e:
    raise RuntimeError(f"Error monitoring export task: {e}")

# Final status of the task
if export_task.status().get('state') == 'COMPLETED':
    print("\nExport task completed successfully!")
else:
    print("Export task failed with error:", export_task.status().get('error_message'))

# Calculate and print total time taken
end_time = datetime.now()
time_taken = end_time - start_time
print(f"Total time taken for the export task: {time_taken}")
