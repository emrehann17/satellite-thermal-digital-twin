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

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
import xarray as xr