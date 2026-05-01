GEE_PROJECT = "b7-thermal-digital-twin"

MODIS_COLLECTION = "MODIS/061/MOD11A1"
LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"

START_DATE = "2019-01-01"
END_DATE = "2023-12-31"

EXPORT_FOLDER = "B7_Thermal_Digital_Twin"

MODIS_EXPORT = {
    "description": "export_modis_lst_5y_summer_mean",
    "file_name_prefix": "modis_lst_dogu_akdeniz_5y_summer_mean",
    "scale": 1000,
}

LANDSAT_EXPORT = {
    "file_name_prefix": "landsat_lst_dogu_akdeniz",
    "scale": 30,
}

LANDSAT_SCALE = 0.00341802
LANDSAT_OFFSET = 149.0

REGION_NAME = "dogu_akdeniz"

MODIS_EXPORT_FOLDER = "B7_Thermal_Digital_Twin_MODIS"
LANDSAT_LST_EXPORT_FOLDER = "B7_Thermal_Digital_Twin_Landsat_Timeseries"
LANDSAT_QA_EXPORT_FOLDER = "B7_Thermal_Digital_Twin_Landsat_QA"

MODIS_FILE_PREFIX = "modis_lst_dogu_akdeniz_5y_summer_mean"

SUMMER_MONTH_START = 6
SUMMER_MONTH_END = 9

MAX_LANDSAT_DAILY_EXPORTS = 5