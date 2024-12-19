////////////////////////////////////////////////////////////////////////////////////////
///            THIS CODE HAVE TO BE RUN ON GOOGLE EART ENGINE ONLY                   ///
////////////////////////////////////////////////////////////////////////////////////////


// Import the feature collection (Region of Interest)
var roi = ee.FeatureCollection("projects/ee-javila/assets/CARTOG/Grid_Albers");

// List of quadrants to process
var cuadrantes = ['MX_005011', 'MX_007011', 'MX_008011', 'MX_012011'];

// Iterate over the list of quadrants
cuadrantes.forEach(function(element) {

  // Split the quadrant string to extract relevant parts
  var arrayOfStrings = element.split("_");
  var año_inicio = 2016; // Start year
  var año_fin = 2022; // End year

  // Construct the export image name
  var name = "l8_" + arrayOfStrings[1] + "_" + año_inicio.toString() + "_" + año_fin.toString() + "_subtract";
  console.log(name);

  // Filter the region of interest by quadrant name
  var cuad_selec = roi.filter(ee.Filter.eq('NOM', element));

  // Bands of interest for export
  var b_interest = ['SR_B[2-7]'];

  // Function to scale and mask Landsat 8 surface reflectance images
  function prepSrL8(image) {
    // Mask unwanted pixels (fill, cloud, cloud shadow)
    var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
    var saturationMask = image.select('QA_RADSAT').eq(0);

    // Apply scaling factors
    var getFactorImg = function(factorNames) {
      var factorList = image.toDictionary().select(factorNames).values();
      return ee.Image.constant(factorList);
    };
    var scaleImg = getFactorImg(['REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B10']);
    var offsetImg = getFactorImg(['REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B10']);
    var scaled = image.select('SR_B.|ST_B10').multiply(scaleImg).add(offsetImg);

    // Replace original bands with scaled bands and apply masks
    return image.addBands(scaled, null, true)
                .updateMask(qaMask)
                .updateMask(saturationMask);
  }

  // Process images for the start year
  var img = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(cuad_selec)
    .filterDate(año_inicio.toString() + '-01-01', (año_inicio + 1).toString() + '-05-31')
    .map(prepSrL8)
    .select('SR.*')
    .median();

  // Visualization parameters
  var visualization = {
    bands: ['SR_B4', 'SR_B3', 'SR_B2'],
    min: 0.0,
    max: 0.3
  };

  // Add the processed image to the map
  Map.addLayer(img, visualization, 'True Color (432)');

  // Process images for the end year
  var img2 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(cuad_selec)
    .filterDate(año_fin.toString() + '-01-01', (año_fin + 1).toString() + '-05-31')
    .map(prepSrL8)
    .select('SR.*')
    .median();

  // Calculate the subtraction between the two images
  var subtraction = img.subtract(img2);

  // Calculate NDVI for both years and their difference
  var ndvi_2020 = img.normalizedDifference(['SR_B5', 'SR_B4']);
  var ndvi_2022 = img2.normalizedDifference(['SR_B5', 'SR_B4']);
  var subtraction_ndvi = ndvi_2020.subtract(ndvi_2022);

  // Export the first image
  Export.image.toDrive({
    image: img.select(b_interest),
    folder: 'l8_' + año_inicio.toString(),
    description: "l8_" + arrayOfStrings[1] + "_" + año_inicio.toString(),
    scale: 30,
    maxPixels: 1e12,
    crs: 'EPSG:6372',
    region: cuad_selec,
    fileFormat: 'GeoTIFF'
  });

  // Export the second image
  Export.image.toDrive({
    image: img2.select(b_interest),
    folder: 'l8_' + año_fin.toString(),
    description: "l8_" + arrayOfStrings[1] + "_" + año_fin.toString(),
    scale: 30,
    maxPixels: 1e12,
    crs: 'EPSG:6372',
    region: cuad_selec,
    fileFormat: 'GeoTIFF'
  });

  // Export the subtraction image
  Export.image.toDrive({
    image: subtraction.select(b_interest),
    folder: 'l8_substraction',
    description: "l8_" + arrayOfStrings[1] + "_" + año_inicio.toString() + "_" + año_fin.toString() + "_subtract",
    scale: 30,
    maxPixels: 1e12,
    crs: 'EPSG:6372',
    region: cuad_selec,
    fileFormat: 'GeoTIFF'
  });

  // Export the NDVI subtraction
  Export.image.toDrive({
    image: subtraction_ndvi,
    folder: 'l8_substraction_NDVI_1',
    description: "l8_" + arrayOfStrings[1] + "_" + año_inicio.toString() + "_" + año_fin.toString() + "_subtract_NDVI",
    scale: 30,
    maxPixels: 1e12,
    crs: 'EPSG:6372',
    region: cuad_selec,
    fileFormat: 'GeoTIFF'
  });

  // Add the selected quadrant to the map
  Map.addLayer(cuad_selec);

  // Log the current quadrant
  console.log(element);
});
