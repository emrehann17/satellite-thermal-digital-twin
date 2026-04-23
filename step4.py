import ee
import time

ee.Initialize(project='b7-thermal-digital-twin')
east_mediterranean = ee.Geometry.Rectangle([32.5, 35.7, 36.8, 38.0])

dataset = ee.ImageCollection('MODIS/061/MOD11A1') \
    .filterBounds(east_mediterranean) \
    .filterDate('2019-01-01', '2023-12-31')\
    .filter(ee.Filter.calendarRange(6, 9, 'month'))

mean_raw_data = dataset.select('LST_Day_1km').mean().multiply(0.02).subtract(273.15)

task = ee.batch.Export.image.toDrive(
    image=mean_raw_data,
    description='east_mediterranean_5yrs_mean_LST',
    folder = 'B7 - Thermal Digital Twin',
    fileNamePrefix='east_mediterranean_mean_LST',
    region=east_mediterranean,
    scale=1000, #1km resolution 
    crs = 'EPSG:4326',
    maxPixels=1e10
)

task.start()
print('successfully started the export task')
