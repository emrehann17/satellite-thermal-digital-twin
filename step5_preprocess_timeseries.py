"""
step5_preprocess_timeseries.py

Yapılanlar:
    - GeoTIFF dosyalarını okumak
    - DN değerlerini Celsius'a çevirmek
    - Varsa QA_PIXEL bandı ile bulut maskeleme yapmak
    - Zaman serisi oluşturmak
    - Eksik değerleri zamansal interpolasyon ile doldurmak
    - Ortalama LST ve anomali rasteri üretmek
"""

from core.config import *
from core.io_utils import setup_logger

import re
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
import xarray as xr


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "landsat_timeseries"
QA_DIR = BASE_DIR / "data" / "landsat_qa"
OUTPUT_DIR = BASE_DIR / "outputs" / "step5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)


log, log_file = setup_logger("step5")



def extract_date_from_filename(path: Path) -> datetime.date:
    """
    Örnek dosya adları:
        - landsat_lst_dogu_akdeniz_2019-06-01_001.tif -> 2019-06-01
        - landsat_lst_dogu_akdeniz_20190601_001.tif   -> 2019-06-01
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{8})", path.name)
    
    if not match:
        raise ValueError(f"Dosya adında tarih bulunamadı: {path.name}")

    date_text = match.group(1)

    if "-" not in date_text:
        date_text = f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]}"

    return np.datetime64(date_text)


def dn_to_celsius(dn_array: np.ndarray) -> np.ndarray:
    """
    Landsat Collection 2 Level 2 ST_B10 DN değerini Celsius'a çevirir.

    Kelvin = DN * 0.00341802 + 149.0
    Celsius = Kelvin - 273.15
    """

    kelvin = dn_array * LANDSAT_SCALE + LANDSAT_OFFSET
    celcius = kelvin - 273.15
    return celcius

#yapilacaklar
#bulut maskeleme
#tek bant raster
#tek bant geotiff kayit etme
#landsat geotiff iccin data array
#main
