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

def build_cloud_mask_from_qa(qa_array: np.ndarray) -> np.ndarray:
    """
        Landsat QA_PIXEL bandına göre bulut maskesi üretir.

        True  = temiz piksel
        False = maskelenecek piksel
    """

    fill = 1 << 0
    dilated_cloud = 1 << 1
    cirrus = 1 << 2
    cloud = 1 << 3
    cloud_shadow = 1 << 4
    snow = 1 << 5   

    bad_pixels = (fill | dilated_cloud | cirrus | cloud | cloud_shadow | snow) # 1111... olan bir sayı

    clean_mask = (qa_array.astype(np.uint16) & bad_pixels) == 0 
    return clean_mask

def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """tek bant raster okuyor"""

    with rasterio.open(path) as src:
        array = src.read(1).astype("float32")
        profile = src.profile.copy()

    return array, profile

def save_geotiff(array: np.ndarray, profile: dict, output_path: Path) -> None:
    """tek bant geotiff kaydeder"""
    output_profile = profile.copy()
    output_profile.update(
        dtype = "float32",
        count = 1,
        nodata = np.nan,
        compress = "lzw"
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(array.astype("float32"), 1)


def load_landsat_timeseries() -> tuple[xr.DataArray, dict]:
    """
    Landsat GeoTIFF dosyalarından zaman serisi DataArray oluşturur.
    """
    tif_files = sorted(INPUT_DIR.glob("*.tif"))

    if not tif_files:
        raise FileNotFoundError(
            f"GeoTIFF dosyası bulunamadı: {INPUT_DIR}\n"
            "Step5 zaman serisi için bu klasöre tarih içeren Landsat GeoTIFF dosyaları koymalısın.\n"
            "Örnek: landsat_lst_2019-06-15.tif"
        )

    arrays = []
    times = []
    base_profile = None

    for tif_path in tif_files:
        date = extract_date_from_filename(tif_path)
        dn_array, profile = read_raster(tif_path)

        if base_profile is None:
            base_profile = profile

        lst_celsius = dn_to_celsius(dn_array)

        qa_path = QA_DIR / tif_path.name.replace(".tif", "_qa.tif")
        if qa_path.exists():
            qa_array, _ = read_raster(qa_path)
            clean_mask = build_cloud_mask_from_qa(qa_array)
            lst_celsius = np.where(clean_mask, lst_celsius, np.nan)

        lst_celsius = np.where(
            (lst_celsius > -30) & (lst_celsius < 80),
            lst_celsius,
            np.nan
        )

        arrays.append(lst_celsius)
        times.append(date)

    stack = np.stack(arrays, axis=0)

    data = xr.DataArray(
        stack,
        dims=("time", "y", "x"),
        coords={"time": times},
        name="landsat_lst_celsius"
    )

    return data, base_profile

def main() -> None:
    log.info("=" * 60)
    log.info("STEP 5 BAŞLIYOR")
    log.info("=" * 60)

    lst_series, profile = load_landsat_timeseries()

    log.info(f"Zaman serisi yüklendi. Görüntü sayısı: {lst_series.sizes['time']}")

    lst_series = lst_series.sortby("time")

    log.info("Zamansal interpolasyon uygulanıyor.")
    interpolated = lst_series.interpolate_na(
        dim="time",
        method="linear",
        use_coordinate=True
    )

    mean_lst = interpolated.mean(dim="time", skipna=True)
    latest_lst = interpolated.isel(time=-1)
    anomaly = latest_lst - mean_lst

    save_geotiff(
        mean_lst.values,
        profile,
        OUTPUT_DIR / "landsat_lst_timeseries_mean_celsius.tif"
    )
    log.info("Ortalama LST GeoTIFF kaydedildi.")

    save_geotiff(
        anomaly.values,
        profile,
        OUTPUT_DIR / "landsat_lst_latest_anomaly_celsius.tif"
    )
    log.info("Son görüntü anomalisi GeoTIFF kaydedildi.")

    interpolated.to_netcdf(
        OUTPUT_DIR / "landsat_lst_timeseries_interpolated.nc"
    )
    log.info("İnterpole edilmiş zaman serisi NetCDF olarak kaydedildi.")

    metadata = {
        "step": "step5_preprocess_timeseries",
        "created_at": datetime.now().isoformat(),
        "input_dir": str(INPUT_DIR),
        "qa_dir": str(QA_DIR),
        "log_file": str(log_file),
        "time_count": int(interpolated.sizes["time"]),
        "outputs": {
            "mean_lst": "landsat_lst_timeseries_mean_celsius.tif",
            "latest_anomaly": "landsat_lst_latest_anomaly_celsius.tif",
            "interpolated_netcdf": "landsat_lst_timeseries_interpolated.nc"
        },
        "status": "processed"
    }

    metadata_path = OUTPUT_DIR / "step5_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    log.info(f"Metadata kaydedildi: {metadata_path}")
    log.info("=" * 60)
    log.info("STEP 5 TAMAMLANDI")
    log.info(f"Çıktı klasörü: {OUTPUT_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()