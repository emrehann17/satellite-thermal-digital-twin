"""
step5_preprocess_timeseries.py

Yeni Yaklaşım (Zaman Penceresi Bazlı Anomali):
    - Baseline zaman serisinden mean ve std hesaplamak
    - Current period median raster'ı okumak
    - Z-score bazlı anomali hesaplamak: (current - baseline_mean) / baseline_std
    - QA tabanlı bulut maskeleme yapmak
    - Zamansal interpolasyon uygulamak
    - Çıktıları kaydetmek

Eski Yaklaşım (Kaldırıldı):
    - "Son N sahne ortalaması" yaklaşımı kaldırıldı
    - Artık zaman penceresi ve median composite kullanılıyor
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
from rasterio.windows import Window

BASE_DIR = Path(__file__).resolve().parent
BASELINE_INPUT_DIR = BASE_DIR / "data" / "landsat_timeseries"
QA_DIR = BASE_DIR / "data" / "landsat_qa"
CURRENT_PERIOD_DIR = BASE_DIR / "data" / "current_period"
OUTPUT_DIR = BASE_DIR / "outputs" / "step5"

BASELINE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_PERIOD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEP5_WINDOW_SIZE = globals().get("STEP5_WINDOW_SIZE", 512)
STEP5_WRITE_INTERPOLATED_NETCDF = globals().get(
    "STEP5_WRITE_INTERPOLATED_NETCDF",
    False,
)

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
    kelvin = dn_array * LANDSAT_SCALE + LANDSAT_OFFSET
    return kelvin - 273.15

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

    bad_pixels = (fill | dilated_cloud | cirrus | cloud | cloud_shadow | snow)

    return (qa_array.astype(np.uint16) & bad_pixels) == 0


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Tek bant raster okur"""

    with rasterio.open(path) as src:
        array = src.read(1).astype("float32")
        profile = src.profile.copy()

    return array, profile


def save_geotiff(array: np.ndarray, profile: dict, output_path: Path) -> None:
    """Tek bant geotiff kaydeder"""
    output_profile = profile.copy()
    output_profile.update(
        dtype = "float32",
        count = 1,
        nodata = np.nan,
        compress = "lzw"
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(array.astype("float32"), 1)


def load_baseline_timeseries() -> tuple[xr.DataArray, dict]:
    """
    Baseline zaman serisini yükler (2019-2022 gibi geçmiş veri).
    Bu veriden mean ve std hesaplanacak.
    """
    tif_files = sorted(BASELINE_INPUT_DIR.glob("*.tif"))

    if not tif_files:
        raise FileNotFoundError(
            f"Baseline GeoTIFF dosyası bulunamadı: {BASELINE_INPUT_DIR}\n"
            "Step4'ten export edilen baseline zaman serisi dosyalarını buraya koymalısın."
        )

    log.info(f"Baseline zaman serisi yükleniyor: {len(tif_files)} dosya bulundu")

    arrays = []
    times = []
    base_profile = None

    for tif_path in tif_files:
        date = extract_date_from_filename(tif_path)
        dn_array, profile = read_raster(tif_path)

        if base_profile is None:
            base_profile = profile

        lst_celsius = dn_to_celsius(dn_array)

        # QA maskeleme
        qa_path = QA_DIR / tif_path.name.replace(".tif", "_qa.tif")
        if qa_path.exists():
            qa_array, _ = read_raster(qa_path)
            clean_mask = build_cloud_mask_from_qa(qa_array)
            lst_celsius = np.where(clean_mask, lst_celsius, np.nan)

        # Fiziksel sınırlar
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
        name="baseline_lst_celsius"
    )

    log.info(f"Baseline zaman serisi yüklendi: {data.sizes['time']} görüntü")

    return data, base_profile


def load_current_period_median(profile: dict) -> np.ndarray:
    """
    Current period median raster'ını yükler.
    Bu zaten Celsius cinsinden geliyor (GEE tarafında dönüştürülmüş).
    """
    current_files = list(CURRENT_PERIOD_DIR.glob("*.tif"))
    
    if not current_files:
        raise FileNotFoundError(
            f"Current period median dosyası bulunamadı: {CURRENT_PERIOD_DIR}\n"
            "Step4'ten export edilen 'landsat_current_period_XXdays.tif' dosyasını buraya koymalısın."
        )
    
    if len(current_files) > 1:
        log.warning(f"Birden fazla current period dosyası bulundu, ilki kullanılıyor: {current_files[0].name}")
    
    current_path = current_files[0]
    log.info(f"Current period median yükleniyor: {current_path.name}")
    
    current_celsius, _ = read_raster(current_path)
    
    # Fiziksel sınırlar
    current_celsius = np.where(
        (current_celsius > -30) & (current_celsius < 80),
        current_celsius,
        np.nan
    )
    
    log.info(f"Current period median yüklendi")
    
    return current_celsius


def main() -> None:
    log.info("=" * 60)
    log.info("STEP 5 BAŞLIYOR (Yeni Zaman Penceresi Yaklaşımı)")
    log.info("=" * 60)

    # 1. Baseline zaman serisi yükle
    log.info("1) Baseline zaman serisi yükleniyor...")
    baseline_series, profile = load_baseline_timeseries()
    
    # 2. Baseline'ı zamansal interpolasyon ile temizle
    log.info("2) Baseline zamansal interpolasyon...")
    baseline_series = baseline_series.sortby("time")
    baseline_interpolated = baseline_series.interpolate_na(
        dim="time",
        method="linear",
        use_coordinate=True
    )
    
    # 3. Baseline istatistikleri hesapla
    log.info("3) Baseline istatistikleri hesaplanıyor...")
    baseline_mean = baseline_interpolated.mean(dim="time", skipna=True)
    baseline_std = baseline_interpolated.std(dim="time", skipna=True)
    
    log.info(f"   Baseline mean: {float(baseline_mean.mean()):.2f}°C")
    log.info(f"   Baseline std:  {float(baseline_std.mean()):.2f}°C")
    
    # 4. Current period median yükle
    log.info("4) Current period median yükleniyor...")
    current_median = load_current_period_median(profile)
    
    # 5. Z-score bazlı anomali hesapla
    log.info("5) Z-score bazlı anomali hesaplanıyor...")
    log.info("   Formül: z_score = (current_median - baseline_mean) / baseline_std")
    
    anomaly_zscore = (current_median - baseline_mean.values) / baseline_std.values
    
    # Sonsuz değerleri temizle (std=0 olan pikseller için)
    anomaly_zscore = np.where(
        np.isfinite(anomaly_zscore),
        anomaly_zscore,
        np.nan
    )
    
    valid_pixels = np.sum(~np.isnan(anomaly_zscore))
    total_pixels = anomaly_zscore.size
    coverage_pct = 100 * valid_pixels / total_pixels
    
    log.info(f"   Geçerli piksel kapsama: {coverage_pct:.1f}%")
    log.info(f"   Anomali ortalaması: {np.nanmean(anomaly_zscore):.2f} σ")
    log.info(f"   Anomali aralığı: [{np.nanmin(anomaly_zscore):.2f}, {np.nanmax(anomaly_zscore):.2f}] σ")
    
    # 6. Çıktıları kaydet
    log.info("6) Çıktılar kaydediliyor...")
    
    # Baseline mean
    save_geotiff(
        baseline_mean.values,
        profile,
        OUTPUT_DIR / "baseline_lst_mean_celsius.tif"
    )
    log.info("   ✓ Baseline mean GeoTIFF kaydedildi")
    
    # Baseline std
    save_geotiff(
        baseline_std.values,
        profile,
        OUTPUT_DIR / "baseline_lst_std_celsius.tif"
    )
    log.info("   ✓ Baseline std GeoTIFF kaydedildi")
    
    # Current period median
    save_geotiff(
        current_median,
        profile,
        OUTPUT_DIR / "current_period_median_celsius.tif"
    )
    log.info("   ✓ Current period median GeoTIFF kaydedildi")
    
    # Anomali (z-score)
    save_geotiff(
        anomaly_zscore,
        profile,
        OUTPUT_DIR / "anomaly_zscore.tif"
    )
    log.info("   ✓ Anomali z-score GeoTIFF kaydedildi")
    
    # NetCDF (baseline zaman serisi)
    baseline_interpolated.to_netcdf(
        OUTPUT_DIR / "baseline_timeseries_interpolated.nc"
    )
    log.info("   ✓ Baseline zaman serisi NetCDF kaydedildi")
    
    # 7. Metadata kaydet
    metadata = {
        "step": "step5_preprocess_timeseries",
        "method": "window_based_zscore_anomaly",
        "created_at": datetime.now().isoformat(),
        "input_dirs": {
            "baseline_timeseries": str(BASELINE_INPUT_DIR),
            "qa_masks": str(QA_DIR),
            "current_period": str(CURRENT_PERIOD_DIR)
        },
        "log_file": str(log_file),
        "baseline": {
            "time_count": int(baseline_interpolated.sizes["time"]),
            "date_range": f"{baseline_interpolated.time.values[0]} to {baseline_interpolated.time.values[-1]}",
            "interpolation_method": "linear",
            "mean_celsius": float(baseline_mean.mean()),
            "std_celsius": float(baseline_std.mean())
        },
        "current_period": {
            "window_days": CURRENT_PERIOD_DAYS,
            "end_date": CURRENT_PERIOD_END_DATE,
            "mean_celsius": float(np.nanmean(current_median))
        },
        "anomaly": {
            "method": "z_score",
            "formula": "(current_median - baseline_mean) / baseline_std",
            "coverage_percent": float(coverage_pct),
            "mean_zscore": float(np.nanmean(anomaly_zscore)),
            "min_zscore": float(np.nanmin(anomaly_zscore)),
            "max_zscore": float(np.nanmax(anomaly_zscore))
        },
        "masking": {
            "qa_source": "QA_PIXEL",
            "physical_range": "[-30, 80] Celsius"
        },
        "outputs": {
            "baseline_mean": "baseline_lst_mean_celsius.tif",
            "baseline_std": "baseline_lst_std_celsius.tif",
            "current_median": "current_period_median_celsius.tif",
            "anomaly_zscore": "anomaly_zscore.tif",
            "baseline_netcdf": "baseline_timeseries_interpolated.nc"
        },
        "status": "processed"
    }

    metadata_path = OUTPUT_DIR / "step5_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    log.info(f"\n   ✓ Metadata kaydedildi: {metadata_path}")
    log.info("=" * 60)
    log.info("STEP 5 TAMAMLANDI")
    log.info(f"Çıktı klasörü: {OUTPUT_DIR}")
    log.info("\nÜretilen dosyalar:")
    log.info("  1. baseline_lst_mean_celsius.tif")
    log.info("  2. baseline_lst_std_celsius.tif")
    log.info("  3. current_period_median_celsius.tif")
    log.info("  4. anomaly_zscore.tif")
    log.info("  5. baseline_timeseries_interpolated.nc")
    log.info("=" * 60)


if __name__ == "__main__":
    main()