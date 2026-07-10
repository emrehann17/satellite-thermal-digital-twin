"""
step7d_predict_downscaled_lst.py

Egitilmis Step7C SAF MODIS->Landsat LST downscaling modelini TAM referans
raster gridine uygulayarak 30 m Landsat-benzeri downscaled LST GeoTIFF uretir.

ONEMLI:
    - Step7D bir DOWNSCALING raster tahmin adimidir; FIRE-RISK modeli DEGILDIR.
    - MCD64A1 veya FIRMS etiketleri KULLANILMAZ.
    - Step5 / Step5C / Step6 / Step7B / Step7C ciktilari DEGISTIRILMEZ.
    - Step7C leakage guard'ina uyulur: anomaly_zscore, current_tvdi,
      tvdi_difference, modis_context_zscore rasterlari OKUNMAZ/KULLANILMAZ
      (diskte mevcut olsalar bile).

Girdi:
    outputs/step7c/downscaling_model.joblib
    outputs/step7c/downscaling_model_metadata.json

Ciktilar:
    outputs/step7d/downscaled_lst_celsius.tif
    outputs/step7d/downscaled_lst_valid_mask.tif
    outputs/step7d/downscaling_prediction_metadata.json
    outputs/step7d/downscaling_prediction_stats.json
    outputs/step7d/downscaling_prediction_summary.md
    outputs/step7d/downscaling_residual_observed_minus_predicted.tif (varsa)
    outputs/step7d/downscaling_absolute_error.tif (varsa)
    outputs/step7d/predicted_lst_histogram.png (--plot)
    outputs/step7d/residual_histogram_on_observed_pixels.png (--plot)
    outputs/step7d/predicted_vs_observed_sample.png (--plot)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

from core.config import (
    STEP7D_TILE_SIZE,
    STEP7D_OUTPUT_DIR,
    STEP7D_MODEL_PATH,
    STEP7D_MODEL_METADATA_PATH,
    STEP7D_MIN_PREDICTED_CELSIUS,
    STEP7D_MAX_PREDICTED_CELSIUS,
    STEP7D_WRITE_RESIDUAL_PRODUCTS,
    STEP7D_PLOT_SAMPLE_SIZE,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.utils.tiling import iter_windows

BASE_DIR = PROJECT_ROOT

log, log_file = setup_logger("step7d")

LEAKAGE_FEATURES = [
    "anomaly_zscore",
    "current_tvdi",
    "tvdi_difference",
    "modis_context_zscore",
]

LEAKAGE_RASTER_PATHS = [
    "outputs/step5/anomaly_zscore.tif",
    "outputs/step5/modis_context_zscore.tif",
    "outputs/step5c/current_tvdi.tif",
    "outputs/step5c/tvdi_difference.tif",
    "outputs/step5c/tvdi_anomaly_zscore.tif",
]

DERIVED_COORD_FEATURES = {"lon", "lat", "row", "col", "row_norm", "col_norm"}

# Landcover kategoriktir; kaynak çözünürlüğü (ESA WorldCover ~10 m) referans
# Landsat gridinden (~30 m) farklı olduğu için TEK istisna olarak nearest-
# neighbor ile önceden hizalanmış bir kopyası kullanılır. Başka HİÇBİR raster
# Step7D içinde otomatik resample EDİLMEZ.
LANDCOVER_ALIGNED_RELPATH = "data/landcover/landcover_esa_worldcover_v200_aligned_to_landsat.tif"
LANDCOVER_SOURCE_RELPATH = "data/landcover/landcover_esa_worldcover_v200.tif"

FEATURE_RASTER_CANDIDATES: dict[str, list[str]] = {
    "modis_lst_mean_celsius": [
        "outputs/step5/modis_lst_mean_celsius_resampled.tif",
        "data/modis/modis_lst_dogu_akdeniz_4y_summer_mean.tif",
    ],
    "modis_lst_std_celsius": [
        "outputs/step5/modis_lst_std_celsius_resampled.tif",
    ],
    "ndvi": [
        "data/ndvi_current_period/current_ndvi_median.tif",
    ],
    "elevation": [
        "data/dem/elevation.tif",
    ],
    "slope": [
        "data/dem/slope.tif",
    ],
    # Belgeleme amaçlı: gerçek çözümleme resolve_feature_rasters() içinde
    # prepare_aligned_landcover() ile özel olarak ele alınır (bkz. aşağı).
    "landcover": [
        LANDCOVER_ALIGNED_RELPATH,
        LANDCOVER_SOURCE_RELPATH,
    ],
}


def load_model_and_metadata(model_path: Path, metadata_path: Path) -> tuple[dict, dict]:
    """joblib model bundle'ini ve Step7C metadata JSON'ini yukler."""
    if not model_path.exists():
        raise SystemExit(
            f"Step7C model dosyasi bulunamadi: {model_path}. Once Step7C'yi "
            "calistirin: python src/step7c_train_downscaling_model.py"
        )
    if not metadata_path.exists():
        raise SystemExit(f"Step7C metadata dosyasi bulunamadi: {metadata_path}")

    bundle = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    log.info("Model yuklendi: %s (type=%s)", model_path, metadata.get("model_type"))
    return bundle, metadata


def confirm_leakage_guard(metadata: dict, safe_features: list[str]) -> None:
    """Step7C leakage guard'inin etkin oldugunu ve leakage ozelligi kullanilmadigini dogrular."""
    guard_enabled = bool(
        metadata.get("leakage_guard_enabled") or metadata.get("leakage_guard")
    )
    if not guard_enabled:
        raise SystemExit(
            "Step7C metadata'sinda leakage_guard_enabled/leakage_guard True "
            "degil. Step7D guvenlik nedeniyle durduruluyor; once Step7C'yi "
            "leakage guard acik sekilde yeniden calistirin."
        )

    leaked = [f for f in safe_features if f in LEAKAGE_FEATURES]
    if leaked:
        raise SystemExit(
            f"KRITIK: safe_feature_columns icinde leakage ozelligi bulundu: "
            f"{leaked}. Step7D bu ozellikleri asla kullanmaz; islem durduruldu."
        )
    log.info(
        "Leakage guard dogrulandi: leakage_guard_enabled=True, "
        "excluded features (%s) ozellik setinde yok.", LEAKAGE_FEATURES
    )


def resolve_reference_grid(ctx: dict | None = None) -> tuple[Path, int]:
    """Step7B/Step7C ile ayni target rasterini (referans grid) cozer.

    ctx: None ise (varsayılan) legacy Kozan keşfi. Verilirse (Kozan-dışı)
        YALNIZCA ctx["step5_output_dir"] altına bakar.
    """
    if ctx is not None:
        p = ctx["step5_output_dir"] / "current_period_median_celsius.tif"
        if p.exists():
            return p, 1
        raise SystemExit(
            f"Referans grid (Landsat LST target) bulunamadi: {p}. Once "
            "Step5'i (namespaced) calistirin."
        )

    candidates = [
        (BASE_DIR / "outputs" / "step5" / "current_period_median_celsius.tif", 1),
        (BASE_DIR / "data" / "current_period" / "landsat_current_period_60days.tif", 1),
    ]
    for path, band in candidates:
        if path.exists():
            return path, band
    cp_dir = BASE_DIR / "data" / "current_period"
    if cp_dir.exists():
        for path in sorted(cp_dir.glob("landsat_current_period_*days.tif")):
            if "(" in path.name:
                continue
            return path, 1
    raise SystemExit(
        "Referans grid (Landsat LST target) bulunamadi. Beklenen: "
        "outputs/step5/current_period_median_celsius.tif veya "
        "data/current_period/landsat_current_period_*days.tif"
    )


def _grid_matches(path: Path, ref_width: int, ref_height: int, ref_crs, ref_transform) -> bool:
    """Bir rasterın verilen referans grid ile birebir eşleşip eşleşmediğini kontrol eder."""
    with rasterio.open(path) as src:
        return (
            src.width == ref_width
            and src.height == ref_height
            and src.crs == ref_crs
            and src.transform == ref_transform
        )


def prepare_aligned_landcover(
    reference_path: Path,
    source_path: Path | None,
    aligned_path: Path,
) -> dict:
    """
    ESA WorldCover (kategorik) landcover rasterini nearest-neighbor ile
    referans (Landsat) gridine hizalar.

    YALNIZCA landcover için geçerli bir istisnadır — başka hiçbir raster
    Step7D içinde otomatik/sessizce resample EDİLMEZ. Zaten hizalanmış ve
    referans gridle eşleşen bir dosya varsa yeniden kullanılır (reuse).
    Hizalama SONRASI dosya referans gridle tekrar doğrulanır; hâlâ
    eşleşmiyorsa net hata ile durulur.
    """
    with rasterio.open(reference_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    if aligned_path.exists():
        if _grid_matches(aligned_path, ref_width, ref_height, ref_crs, ref_transform):
            log.info("Önceden hizalanmış landcover yeniden kullanılıyor: %s", aligned_path)
            return {"created": False, "reused": True}
        log.warning(
            "Mevcut hizalanmış landcover (%s) referans gridle uyuşmuyor; "
            "yeniden oluşturulacak.", aligned_path,
        )

    if source_path is None or not source_path.exists():
        raise SystemExit(
            "Hizalanmış landcover oluşturulamıyor: kaynak dosya bulunamadı "
            f"({LANDCOVER_SOURCE_RELPATH}) ve mevcut hizalanmış dosya da yok/uyuşmuyor."
        )

    log.info("Preparing aligned categorical landcover using nearest-neighbor resampling.")

    with rasterio.open(source_path) as src:
        src_dtype = src.dtypes[0]
        src_nodata = src.nodata if src.nodata is not None else 0
        dst = np.full((ref_height, ref_width), src_nodata, dtype=src_dtype)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=src_nodata,
            resampling=Resampling.nearest,
        )
        out_profile = {
            "driver": "GTiff",
            "width": ref_width,
            "height": ref_height,
            "count": 1,
            "dtype": src_dtype,
            "crs": ref_crs,
            "transform": ref_transform,
            "nodata": src_nodata,
            "compress": "deflate",
            # 16'nın katı olmayan boyutlarda blok hatasını önlemek için tiled'ı
            # yalnızca boyutlar yeterince büyükse etkinleştir (GDAL varsayılan
            # 256x256 blok kullanır).
            "tiled": bool(ref_width >= 256 and ref_height >= 256),
        }

    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(aligned_path, "w", **out_profile) as dst_ds:
        dst_ds.write(dst, 1)
    log.info("Hizalanmış landcover yazıldı: %s", aligned_path)

    # Oluşturduktan SONRA referans gridle yeniden doğrula (madde 3).
    if not _grid_matches(aligned_path, ref_width, ref_height, ref_crs, ref_transform):
        raise SystemExit(
            f"Hizalanmış landcover ({aligned_path}) oluşturulduktan sonra bile "
            "referans gridle eşleşmiyor. İşlem durduruldu."
        )
    return {"created": True, "reused": False}


def resolve_feature_rasters(
    safe_features: list[str], reference_path: Path, ctx: dict | None = None,
) -> tuple[dict[str, Path], dict]:
    """
    Metadata'daki safe_feature_columns icin gercek raster yollarini cozer.

    Kozan-disi deneyler (ctx verilmis ve ctx["is_kozan"]=False): TÜM feature
    rasterlari YALNIZCA Step7B'nin ONCEDEN uretip
    outputs/experiments/<experiment_id>/step7b/aligned_inputs/<name>.tif'e
    yazdigi, referans gridle ZATEN eslesen dosyalardan okunur. Ham/coarse
    namespaced kaynaklara (data/modis, dem_input_dir, ndvi_current_dir, vb.)
    ASLA geri DUSULMEZ -- boylece Step7D kendi ic pencere-pencere reproject
    mantigina GUVENMEZ ("do not silently resample"). Bir aligned dosya
    eksikse, o ozellik icin HEMEN (diger ozellikleri kontrol etmeden) su
    net mesajla durur: "Aligned feature raster missing for Step7D: <feature>.
    Re-run Step7B."

    Kozan (ctx=None veya ctx["is_kozan"]=True): legacy davranis KORUNUR
    (FEATURE_RASTER_CANDIDATES, BASE_DIR relative, landcover icin
    LANDCOVER_ALIGNED_RELPATH/prepare_aligned_landcover). Ek olarak, eger
    outputs/step7b/aligned_inputs/<name>.tif zaten mevcutsa (Step7B bir
    deney icin degil de Kozan icin de calistirilip aligned_inputs
    uretmisse), bu ONCELIKLI olarak kullanilir -- ama mevcut degilse legacy
    kesif ile devam edilir (Kozan hicbir zaman bu yuzden BOZULMAZ).
    """
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    landcover_info: dict = {
        "original_landcover_path": None,
        "aligned_landcover_path": None,
        "landcover_alignment_method": None,
        "landcover_alignment_reason": None,
        "landcover_alignment_created": False,
    }

    use_aligned_inputs_only = ctx is not None and not ctx.get("is_kozan")
    aligned_inputs_dir = (
        ctx["step7b_output_dir"] / "aligned_inputs" if use_aligned_inputs_only else None
    )
    # Kozan icin de aligned_inputs varsa ONCELIKLI kullanilir (opsiyonel,
    # ZORUNLU degil -- bkz. docstring).
    kozan_aligned_inputs_dir = BASE_DIR / "outputs" / "step7b" / "aligned_inputs"

    for name in safe_features:
        if name in DERIVED_COORD_FEATURES:
            continue
        if name in LEAKAGE_FEATURES:
            raise SystemExit(f"KRITIK: leakage ozelligi '{name}' islenmeye calisildi.")

        if use_aligned_inputs_only:
            aligned_path = aligned_inputs_dir / f"{name}.tif"
            if not aligned_path.exists():
                raise SystemExit(
                    f"Aligned feature raster missing for Step7D: {name}. Re-run Step7B."
                )
            resolved[name] = aligned_path
            if name == "landcover":
                landcover_info.update({
                    "original_landcover_path": None,
                    "aligned_landcover_path": str(aligned_path),
                    "landcover_alignment_method": "step7b_aligned_inputs_reuse",
                    "landcover_alignment_reason": (
                        "reused Step7B's pre-aligned aligned_inputs/landcover.tif "
                        "(already matches the Step5 reference grid); no "
                        "resampling performed in Step7D"
                    ),
                    "landcover_alignment_created": False,
                })
            continue

        if name == "landcover":
            kozan_aligned_landcover = kozan_aligned_inputs_dir / "landcover.tif"
            if kozan_aligned_landcover.exists():
                resolved[name] = kozan_aligned_landcover
                landcover_info.update({
                    "original_landcover_path": None,
                    "aligned_landcover_path": str(kozan_aligned_landcover),
                    "landcover_alignment_method": "step7b_aligned_inputs_reuse",
                    "landcover_alignment_reason": (
                        "reused outputs/step7b/aligned_inputs/landcover.tif "
                        "(already matches reference grid); no resampling "
                        "performed here"
                    ),
                    "landcover_alignment_created": False,
                })
                continue

            aligned_path = BASE_DIR / LANDCOVER_ALIGNED_RELPATH
            source_path = BASE_DIR / LANDCOVER_SOURCE_RELPATH
            source_path_arg = source_path if source_path.exists() else None

            if not aligned_path.exists() and source_path_arg is None:
                missing.append(name)
                continue

            result = prepare_aligned_landcover(reference_path, source_path_arg, aligned_path)
            resolved[name] = aligned_path
            landcover_info.update({
                "original_landcover_path": (
                    str(source_path) if source_path.exists() else None
                ),
                "aligned_landcover_path": str(aligned_path),
                "landcover_alignment_method": "nearest_neighbor_to_reference_grid",
                "landcover_alignment_reason": (
                    "categorical raster; source resolution differs from "
                    "Landsat reference grid"
                ),
                "landcover_alignment_created": result["created"],
            })
            continue

        kozan_aligned_candidate = kozan_aligned_inputs_dir / f"{name}.tif"
        if kozan_aligned_candidate.exists():
            resolved[name] = kozan_aligned_candidate
            continue

        candidates = FEATURE_RASTER_CANDIDATES.get(name)
        if not candidates:
            raise SystemExit(
                f"Bilinmeyen/desteklenmeyen ozellik: '{name}'. Step7D icin "
                "raster kaynagi tanimli degil."
            )
        found = None
        for rel in candidates:
            p = BASE_DIR / rel
            if p.exists():
                found = p
                break
        if found is None:
            missing.append(name)
        else:
            resolved[name] = found

    if missing:
        raise SystemExit(
            "Gerekli feature raster(lar) bulunamadi: "
            f"{missing}. Bu ozellikler Step7C modelinde kullanildigindan, "
            "tum piksellerde tahmin uretmek icin raster dosyalari mevcut olmali."
        )
    return resolved, landcover_info


def validate_grid_alignment(reference_path: Path, feature_paths: dict[str, Path]) -> dict:
    """Tum feature rasterlarinin referans grid ile birebir eslestigini dogrular."""
    with rasterio.open(reference_path) as ref:
        ref_profile = {
            "width": ref.width, "height": ref.height,
            "crs": ref.crs, "transform": ref.transform,
        }

    mismatches = []
    for name, path in feature_paths.items():
        with rasterio.open(path) as src:
            if (
                src.width != ref_profile["width"]
                or src.height != ref_profile["height"]
                or src.crs != ref_profile["crs"]
                or src.transform != ref_profile["transform"]
            ):
                mismatches.append(
                    f"{name} ({path}): {src.width}x{src.height} {src.crs} "
                    f"vs reference {ref_profile['width']}x{ref_profile['height']} "
                    f"{ref_profile['crs']}"
                )

    if mismatches:
        raise SystemExit(
            "Feature raster(lar) referans grid ile eslesmiyor (Step7D sessizce "
            "resample ETMEZ):\n  - " + "\n  - ".join(mismatches)
        )
    log.info("Tum feature rasterlari referans grid ile hizali dogrulandi.")
    return ref_profile


def build_coord_arrays(
    write_win: Window, transform, raster_height: int, raster_width: int
) -> dict[str, np.ndarray]:
    """Pencere icin row, col, lon, lat, row_norm, col_norm 2B dizilerini uretir."""
    row0, col0 = int(write_win.row_off), int(write_win.col_off)
    h, w = int(write_win.height), int(write_win.width)

    rows_local, cols_local = np.meshgrid(
        np.arange(row0, row0 + h), np.arange(col0, col0 + w), indexing="ij"
    )
    rows_local = rows_local.astype("float64")
    cols_local = cols_local.astype("float64")

    a, b, c, d, e, f = (
        transform.a, transform.b, transform.c, transform.d, transform.e, transform.f,
    )
    lon = a * (cols_local + 0.5) + b * (rows_local + 0.5) + c
    lat = d * (cols_local + 0.5) + e * (rows_local + 0.5) + f

    row_denom = max(raster_height - 1, 1)
    col_denom = max(raster_width - 1, 1)

    return {
        "row": rows_local,
        "col": cols_local,
        "lon": lon,
        "lat": lat,
        "row_norm": rows_local / row_denom,
        "col_norm": cols_local / col_denom,
    }


def run_prediction(
    model,
    safe_features: list[str],
    reference_path: Path,
    reference_band: int,
    feature_paths: dict[str, Path],
    output_dir: Path,
    tile_size: int,
    write_residual_products: bool,
) -> dict:
    """Tum raster gridini pencere pencere gezerek tahmin uretir ve yazar."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / "downscaled_lst_celsius.tif"
    mask_path = output_dir / "downscaled_lst_valid_mask.tif"
    resid_path = output_dir / "downscaling_residual_observed_minus_predicted.tif"
    abs_err_path = output_dir / "downscaling_absolute_error.tif"

    counters = {
        "window_count": 0,
        "total_pixels": 0,
        "valid_prediction_pixels": 0,
        "invalid_prediction_pixels": 0,
        "out_of_range_prediction_count": 0,
        "observed_overlap_pixel_count": 0,
    }

    pred_sum = 0.0
    pred_sum_sq = 0.0
    pred_min = math.inf
    pred_max = -math.inf
    pred_values_sample: list[np.ndarray] = []

    resid_sum = 0.0
    resid_sum_sq = 0.0
    resid_abs_sum = 0.0
    obs_vals_for_metrics: list[np.ndarray] = []
    pred_vals_for_metrics: list[np.ndarray] = []

    feature_handles = {name: rasterio.open(p) for name, p in feature_paths.items()}

    with rasterio.open(reference_path) as ref_src:
        profile = ref_src.profile.copy()
        # Kaynak profilden gelebilecek blockxsize/blockysize (16'nın katı olmayabilir)
        # kaldırılır; tiled=True ile GDAL varsayılan (256) blok boyutunu kullanır.
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            count=1, dtype="float32", nodata=float("nan"),
            compress="deflate", tiled=True,
        )
        mask_profile = profile.copy()
        mask_profile.update(dtype="uint8", nodata=0)

        raster_height, raster_width = ref_src.height, ref_src.width
        transform = ref_src.transform

        pred_dst = rasterio.open(pred_path, "w", **profile)
        mask_dst = rasterio.open(mask_path, "w", **mask_profile)
        resid_dst = (
            rasterio.open(resid_path, "w", **profile) if write_residual_products else None
        )
        abs_err_dst = (
            rasterio.open(abs_err_path, "w", **profile) if write_residual_products else None
        )

        try:
            for write_win, _read_win, _core_off in iter_windows(
                ref_src, tile_size_pixels=tile_size, overlap_pixels=0
            ):
                counters["window_count"] += 1
                h, w = int(write_win.height), int(write_win.width)
                if h == 0 or w == 0:
                    continue
                counters["total_pixels"] += h * w

                coord_arrays = build_coord_arrays(
                    write_win, transform, raster_height, raster_width
                )

                feature_arrays: dict[str, np.ndarray] = {}
                for name in safe_features:
                    if name in coord_arrays:
                        feature_arrays[name] = coord_arrays[name]
                    else:
                        src = feature_handles[name]
                        arr = src.read(1, window=write_win, masked=True)
                        feature_arrays[name] = arr.astype("float64").filled(np.nan)

                valid = np.ones((h, w), dtype=bool)
                for name in safe_features:
                    valid &= np.isfinite(feature_arrays[name])

                n_valid = int(valid.sum())
                counters["valid_prediction_pixels"] += n_valid
                counters["invalid_prediction_pixels"] += int((~valid).sum())

                pred_window = np.full((h, w), np.nan, dtype="float32")
                if n_valid > 0:
                    X = pd.DataFrame(
                        {name: feature_arrays[name][valid] for name in safe_features},
                        columns=safe_features,
                    )
                    y_pred = model.predict(X).astype("float32")

                    out_of_range = (
                        (y_pred < STEP7D_MIN_PREDICTED_CELSIUS)
                        | (y_pred > STEP7D_MAX_PREDICTED_CELSIUS)
                    )
                    counters["out_of_range_prediction_count"] += int(out_of_range.sum())

                    pred_window[valid] = y_pred

                    pred_sum += float(y_pred.sum())
                    pred_sum_sq += float(np.square(y_pred, dtype="float64").sum())
                    pred_min = min(pred_min, float(y_pred.min()))
                    pred_max = max(pred_max, float(y_pred.max()))
                    if len(pred_values_sample) < 2_000_000:
                        step = max(1, n_valid // 2000 or 1)
                        pred_values_sample.append(y_pred[::step])

                mask_window = valid.astype("uint8")

                pred_dst.write(pred_window, 1, window=write_win)
                mask_dst.write(mask_window, 1, window=write_win)

                if write_residual_products:
                    observed = ref_src.read(
                        reference_band, window=write_win, masked=True
                    ).astype("float64").filled(np.nan)
                    both_finite = np.isfinite(observed) & np.isfinite(pred_window)
                    n_overlap = int(both_finite.sum())
                    counters["observed_overlap_pixel_count"] += n_overlap

                    resid_window = np.full((h, w), np.nan, dtype="float32")
                    abs_err_window = np.full((h, w), np.nan, dtype="float32")
                    if n_overlap > 0:
                        diff = (observed[both_finite] - pred_window[both_finite]).astype(
                            "float64"
                        )
                        resid_window[both_finite] = diff.astype("float32")
                        abs_err_window[both_finite] = np.abs(diff).astype("float32")

                        resid_sum += float(diff.sum())
                        resid_sum_sq += float(np.square(diff).sum())
                        resid_abs_sum += float(np.abs(diff).sum())
                        if len(obs_vals_for_metrics) < 2_000_000:
                            step = max(1, n_overlap // 2000 or 1)
                            obs_vals_for_metrics.append(observed[both_finite][::step])
                            pred_vals_for_metrics.append(pred_window[both_finite][::step])

                    resid_dst.write(resid_window, 1, window=write_win)
                    abs_err_dst.write(abs_err_window, 1, window=write_win)
        finally:
            pred_dst.close()
            mask_dst.close()
            if resid_dst is not None:
                resid_dst.close()
            if abs_err_dst is not None:
                abs_err_dst.close()
            for src in feature_handles.values():
                src.close()

    n_valid_total = counters["valid_prediction_pixels"]
    pred_mean = pred_sum / n_valid_total if n_valid_total else None
    pred_std = (
        math.sqrt(max(pred_sum_sq / n_valid_total - pred_mean ** 2, 0.0))
        if n_valid_total else None
    )
    pred_sample_concat = (
        np.concatenate(pred_values_sample) if pred_values_sample else np.array([])
    )

    n_overlap_total = counters["observed_overlap_pixel_count"]
    resid_mean = resid_sum / n_overlap_total if n_overlap_total else None
    resid_std = (
        math.sqrt(max(resid_sum_sq / n_overlap_total - resid_mean ** 2, 0.0))
        if n_overlap_total else None
    )
    resid_mae = resid_abs_sum / n_overlap_total if n_overlap_total else None
    resid_rmse = math.sqrt(resid_sum_sq / n_overlap_total) if n_overlap_total else None
    obs_concat = np.concatenate(obs_vals_for_metrics) if obs_vals_for_metrics else np.array([])
    pred_concat = (
        np.concatenate(pred_vals_for_metrics) if pred_vals_for_metrics else np.array([])
    )
    resid_concat = obs_concat - pred_concat if obs_concat.size else np.array([])

    r2 = None
    if len(obs_concat) > 1:
        try:
            from sklearn.metrics import r2_score
            r2 = float(r2_score(obs_concat, pred_concat))
        except Exception:  # noqa: BLE001
            r2 = None

    def _pctl(arr: np.ndarray, q: float) -> float | None:
        return float(np.percentile(arr, q)) if arr.size else None

    stats = {
        **counters,
        "prediction_coverage_pct": (
            round(100.0 * n_valid_total / counters["total_pixels"], 4)
            if counters["total_pixels"] else 0.0
        ),
        "predicted_lst_min": pred_min if n_valid_total else None,
        "predicted_lst_max": pred_max if n_valid_total else None,
        "predicted_lst_mean": pred_mean,
        "predicted_lst_std": pred_std,
        "predicted_lst_median": _pctl(pred_sample_concat, 50),
        "predicted_lst_p05": _pctl(pred_sample_concat, 5),
        "predicted_lst_p95": _pctl(pred_sample_concat, 95),
        "residual_mean": resid_mean,
        "residual_std": resid_std,
        "residual_median": _pctl(resid_concat, 50),
        "residual_p05": _pctl(resid_concat, 5),
        "residual_p95": _pctl(resid_concat, 95),
        "observed_vs_predicted_rmse": resid_rmse,
        "observed_vs_predicted_mae": resid_mae,
        "observed_vs_predicted_bias": (-resid_mean if resid_mean is not None else None),
        "observed_vs_predicted_r2": r2,
    }

    output_paths = {
        "predicted": str(pred_path),
        "valid_mask": str(mask_path),
        "residual": str(resid_path) if write_residual_products else None,
        "absolute_error": str(abs_err_path) if write_residual_products else None,
    }
    raster_info = {
        "raster_shape": [raster_height, raster_width],
        "crs": str(profile["crs"]),
        "transform": [
            transform.a, transform.b, transform.c, transform.d, transform.e, transform.f,
        ],
    }
    return {
        "stats": stats,
        "output_paths": output_paths,
        "raster_info": raster_info,
        "pred_sample": pred_sample_concat,
        "obs_sample": obs_concat,
        "pred_vs_obs_sample": pred_concat,
        "resid_sample": resid_concat,
    }


def write_plots(run_result: dict, output_dir: Path, sample_size: int) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    rng = np.random.default_rng(42)

    pred_sample = run_result["pred_sample"]
    if pred_sample.size:
        s = pred_sample
        if s.size > sample_size:
            idx = rng.choice(s.size, size=sample_size, replace=False)
            s = s[idx]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s, bins=60, color="#1f77b4", alpha=0.85)
        ax.set_xlabel("Predicted Landsat-like LST (C)")
        ax.set_ylabel("Count")
        ax.set_title("Step7D predicted LST distribution (sampled)")
        fig.tight_layout()
        path = output_dir / "predicted_lst_histogram.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    resid_sample = run_result["resid_sample"]
    if resid_sample.size:
        s = resid_sample
        if s.size > sample_size:
            idx = rng.choice(s.size, size=sample_size, replace=False)
            s = s[idx]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s, bins=60, color="#ff7f0e", alpha=0.85)
        ax.axvline(0, color="k", linewidth=1)
        ax.set_xlabel("Residual (observed - predicted), C")
        ax.set_ylabel("Count")
        ax.set_title(
            "Residual on observed-overlap pixels (in-sample diagnostic, "
            "not independent validation)"
        )
        fig.tight_layout()
        path = output_dir / "residual_histogram_on_observed_pixels.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    obs_s, pred_s = run_result["obs_sample"], run_result["pred_vs_obs_sample"]
    if obs_s.size and pred_s.size:
        n = min(obs_s.size, pred_s.size)
        o, p = obs_s[:n], pred_s[:n]
        if n > sample_size:
            idx = rng.choice(n, size=sample_size, replace=False)
            o, p = o[idx], p[idx]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(o, p, s=4, alpha=0.3, color="#2ca02c")
        lims = [min(o.min(), p.min()), max(o.max(), p.max())] if o.size else [0, 1]
        ax.plot(lims, lims, "r--", linewidth=1, label="1:1")
        ax.set_xlabel("Observed Landsat LST (C)")
        ax.set_ylabel("Predicted LST (C)")
        ax.set_title("Predicted vs observed (current-window overlap sample)")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "predicted_vs_observed_sample.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    return written


def load_step7c_metrics_summary() -> dict | None:
    """Varsa Step7C metrics.json'dan kompakt bir ozet dondurur (referans amacli)."""
    path = BASE_DIR / "outputs" / "step7c" / "downscaling_model_metrics.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "test": data.get("test"),
        "modis_baseline": data.get("modis_baseline"),
        "improvement_over_modis_baseline": data.get("improvement_over_modis_baseline"),
    }


def _modis_spatial_calibration_note(ctx: dict | None) -> str:
    """
    MODIS mean/std katmanlarinin ne oldugunu (ve ne OLMADIGINI) aciklayan
    metadata notu. Kozan (ctx=None veya ctx["is_kozan"]) icin legacy metin
    BIREBIR korunur (coklu-yil yaz-ortalamasi baseline). Kozan-disi bir
    deney icin (or. manavgat_2021), MODIS o deneyin PREDICTOR penceresi
    icin tek-sezonluk (single-season) export edildigi icin ("coklu-yil
    baseline" DEGIL -- bkz. scripts/prepare_modis_for_step7.py), metin
    bunu acikca yansitir.
    """
    if ctx is None or ctx.get("is_kozan"):
        return (
            "modis_lst_mean_celsius is a 4-year summer-mean MODIS context layer, "
            "not a current daily MODIS observation. Step7D is therefore a spatial "
            "downscaling/context calibration raster product, not yet daily MODIS "
            "gap-filling."
        )
    return (
        "modis_lst_mean_celsius and modis_lst_std_celsius are single-season "
        "MODIS predictor-window summary layers for "
        f"{ctx['predictor_start_date']} -> {ctx['predictor_end_date']}; they "
        "are not multi-year baselines and not daily MODIS products."
    )


def write_metadata(
    output_dir: Path,
    model_path: Path,
    metadata_path: Path,
    model_metadata: dict,
    safe_features: list[str],
    reference_path: Path,
    feature_paths: dict[str, Path],
    run_result: dict,
    tile_size: int,
    write_residual_products: bool,
    plots_written: list[str],
    warnings_list: list[str],
    landcover_info: dict | None = None,
    ctx: dict | None = None,
) -> Path:
    landcover_info = landcover_info or {}
    payload = {
        "created_at": datetime.now().isoformat(),
        "script": "step7d_predict_downscaled_lst.py",
        "experiment_id": ctx["experiment_id"] if ctx else None,
        "model_path": str(model_path),
        "model_metadata_path": str(metadata_path),
        "model_type": model_metadata.get("model_type"),
        "safe_feature_columns": safe_features,
        "feature_list": safe_features,
        "excluded_leakage_features": LEAKAGE_FEATURES,
        "leakage_guard_confirmed": True,
        "no_fire_risk_model_trained": True,
        "no_burned_area_labels_used": True,
        "no_firms_labels_used": True,
        "reference_grid_path": str(reference_path),
        "reference_raster": str(reference_path),
        "feature_raster_paths": {k: str(v) for k, v in feature_paths.items()},
        "feature_paths": {k: str(v) for k, v in feature_paths.items()},
        # validate_grid_alignment() bu noktaya ulaşılmadan ÖNCE her feature'ı
        # referans gridle birebir karşılaştırıp uyuşmazlıkta zaten fail-fast
        # yapar; bu yüzden buraya ulaşıldıysa ikisi de garantili True'dur.
        "all_features_match_reference_grid": True,
        "no_silent_resampling": True,
        "original_landcover_path": landcover_info.get("original_landcover_path"),
        "aligned_landcover_path": landcover_info.get("aligned_landcover_path"),
        "landcover_alignment_method": landcover_info.get("landcover_alignment_method"),
        "landcover_alignment_reason": landcover_info.get("landcover_alignment_reason"),
        "output_paths": run_result["output_paths"],
        "plots_written": plots_written,
        "tile_size": tile_size,
        "write_residual_products": write_residual_products,
        "raster_shape": run_result["raster_info"]["raster_shape"],
        "crs": run_result["raster_info"]["crs"],
        "transform": run_result["raster_info"]["transform"],
        "step7c_metrics_summary": load_step7c_metrics_summary(),
        "spatial_calibration_note": _modis_spatial_calibration_note(ctx),
        "warnings": warnings_list,
    }
    path = output_dir / "downscaling_prediction_metadata.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_stats(output_dir: Path, run_result: dict, tile_size: int) -> Path:
    payload = {
        "created_at": datetime.now().isoformat(),
        "raster_shape": run_result["raster_info"]["raster_shape"],
        "crs": run_result["raster_info"]["crs"],
        "transform": run_result["raster_info"]["transform"],
        "tile_size": tile_size,
        **run_result["stats"],
    }
    path = output_dir / "downscaling_prediction_stats.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_summary(
    output_dir: Path,
    model_metadata: dict,
    safe_features: list[str],
    run_result: dict,
    write_residual_products: bool,
    warnings_list: list[str],
    landcover_info: dict | None = None,
) -> Path:
    landcover_info = landcover_info or {}
    def fmt(v, digits=3):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    stats = run_result["stats"]
    step7c_summary = load_step7c_metrics_summary()

    lines = [
        "# Step7D: Downscaled Landsat-like LST Raster Prediction",
        "",
        "**Step7D applied the Step7C pure MODIS-to-Landsat LST downscaling "
        "model to the full reference raster grid. Output is a 30 m "
        "Landsat-like downscaled LST raster. This is not a fire-risk model. "
        "No burned-area or FIRMS labels were used.**",
        "",
        f"- Created at: `{datetime.now().isoformat()}`",
        f"- Model type: `{model_metadata.get('model_type')}`",
        f"- Reference raster shape (h, w): `{run_result['raster_info']['raster_shape']}`",
        f"- CRS: `{run_result['raster_info']['crs']}`",
        "",
        "## Leakage guard",
        "",
        "Target-derived TVDI/anomaly features were **excluded** and never read: "
        f"`{', '.join(LEAKAGE_FEATURES)}`.",
        f"- Features actually used for prediction: `{', '.join(safe_features)}`",
        "",
    ]

    if "landcover" in safe_features and landcover_info.get("aligned_landcover_path"):
        lines.extend([
            "## Landcover alignment",
            "",
            "**ESA WorldCover landcover was explicitly aligned to the Landsat "
            "reference grid using nearest-neighbor resampling before "
            "prediction.**",
            "",
            f"- Original (source) landcover: "
            f"`{landcover_info.get('original_landcover_path') or 'n/a'}`",
            f"- Aligned landcover (used for prediction): "
            f"`{landcover_info.get('aligned_landcover_path')}`",
            f"- Alignment method: "
            f"`{landcover_info.get('landcover_alignment_method')}`",
            f"- Reason: {landcover_info.get('landcover_alignment_reason')}",
            "",
        ])

    lines.extend([
        "## Prediction coverage",
        "",
        f"- Total pixels: `{stats['total_pixels']}`",
        f"- Valid prediction pixels: `{stats['valid_prediction_pixels']}`",
        f"- Invalid pixels (missing feature data): `{stats['invalid_prediction_pixels']}`",
        f"- Coverage: `{fmt(stats['prediction_coverage_pct'], 2)}%`",
        f"- Out-of-range predictions "
        f"(outside [{STEP7D_MIN_PREDICTED_CELSIUS}, {STEP7D_MAX_PREDICTED_CELSIUS}] C, "
        "NOT clamped, reported honestly): "
        f"`{stats['out_of_range_prediction_count']}`",
        "",
        "## Predicted LST distribution",
        "",
        f"- Min: `{fmt(stats['predicted_lst_min'])}` C",
        f"- Max: `{fmt(stats['predicted_lst_max'])}` C",
        f"- Mean: `{fmt(stats['predicted_lst_mean'])}` C",
        f"- Std: `{fmt(stats['predicted_lst_std'])}` C",
        f"- Median (sampled): `{fmt(stats['predicted_lst_median'])}` C",
        f"- P05 / P95 (sampled): `{fmt(stats['predicted_lst_p05'])}` / "
        f"`{fmt(stats['predicted_lst_p95'])}` C",
        "",
    ])

    if write_residual_products:
        lines.extend([
            "## Observed-overlap residual diagnostics (current-window, IN-SAMPLE)",
            "",
            "> **These residual metrics are in-sample/current-window diagnostics, "
            "NOT independent validation.** They compare predictions against the "
            "same current-period Landsat LST raster used elsewhere in this "
            "pipeline, over the same window. They do **not** replace Step7C's "
            "spatial_block test metrics as the actual model validation reference.",
            "",
            f"- Observed-overlap pixel count: `{stats['observed_overlap_pixel_count']}`",
            f"- Residual mean (observed - predicted): `{fmt(stats['residual_mean'])}` C",
            f"- Residual std: `{fmt(stats['residual_std'])}` C",
            f"- Residual median (sampled): `{fmt(stats['residual_median'])}` C",
            f"- RMSE: `{fmt(stats['observed_vs_predicted_rmse'])}` C",
            f"- MAE: `{fmt(stats['observed_vs_predicted_mae'])}` C",
            f"- Bias: `{fmt(stats['observed_vs_predicted_bias'])}` C",
            f"- R2: `{fmt(stats['observed_vs_predicted_r2'])}`",
            "",
        ])

    lines.extend(["## Step7C validation reference (actual model validation)", ""])
    if step7c_summary and step7c_summary.get("test"):
        t = step7c_summary["test"]
        b = step7c_summary.get("modis_baseline", {}) or {}
        lines.extend([
            "The Step7C **spatial_block** test split is the actual (more honest) "
            "validation reference for this model - it tests generalization to "
            "unseen spatial blocks, unlike the in-sample diagnostics above.",
            "",
            f"- Step7C test RMSE: `{fmt(t.get('rmse'))}` C, "
            f"MAE: `{fmt(t.get('mae'))}` C, R2: `{fmt(t.get('r2'))}`",
            f"- MODIS baseline test RMSE: `{fmt(b.get('rmse'))}` C",
        ])
    else:
        lines.append(
            "_Step7C metrics.json not found; cannot report spatial_block "
            "validation reference here. See outputs/step7c/ directly._"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- `modis_lst_mean_celsius` is currently a multi-year summer-mean MODIS "
        "context layer, not a current daily MODIS observation - Step7D is a "
        "**spatial downscaling/context calibration** raster product, not yet "
        "daily MODIS gap-filling.",
        "- Single AOI, single current pre-fire window; no claim of validated "
        "generalization beyond this AOI/window.",
        "- Step7C reported a train-test gap; broad generalization should not "
        "be overclaimed.",
    ])

    if warnings_list:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings_list)

    path = output_dir / "downscaling_prediction_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(
    model_path: str = STEP7D_MODEL_PATH,
    model_metadata_path: str = STEP7D_MODEL_METADATA_PATH,
    output_dir: str = STEP7D_OUTPUT_DIR,
    tile_size: int = STEP7D_TILE_SIZE,
    force: bool = False,
    write_residual_products: bool = STEP7D_WRITE_RESIDUAL_PRODUCTS,
    make_plots: bool = False,
    ctx: dict | None = None,
) -> dict:
    log.info("=" * 60)
    log.info(
        "STEP 7D BASLIYOR (Step7C modeli tam raster gridine uygulaniyor)%s",
        f" [experiment={ctx['experiment_id']}]" if ctx else "",
    )
    log.info("=" * 60)

    out_dir = BASE_DIR / output_dir
    required_outputs = [
        out_dir / "downscaled_lst_celsius.tif",
        out_dir / "downscaled_lst_valid_mask.tif",
        out_dir / "downscaling_prediction_metadata.json",
        out_dir / "downscaling_prediction_stats.json",
        out_dir / "downscaling_prediction_summary.md",
    ]
    if any(p.exists() for p in required_outputs) and not force:
        present = [p.name for p in required_outputs if p.exists()]
        raise SystemExit(
            "Step7D ciktilari zaten var (" + ", ".join(present)
            + "). Uzerine yazmak icin --force verin."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings_list: list[str] = []

    bundle, model_metadata = load_model_and_metadata(
        Path(model_path), Path(model_metadata_path)
    )
    model = bundle["model"]
    safe_features_meta = list(model_metadata.get("safe_feature_columns") or [])
    safe_features_bundle = list(bundle.get("feature_names") or [])

    if safe_features_bundle and safe_features_bundle != safe_features_meta:
        warnings_list.append(
            "Model bundle feature_names differs from metadata safe_feature_columns; "
            "using bundle feature order (what the model actually learned)."
        )
        log.warning(
            "feature_names uyusmazligi: bundle=%s metadata=%s -> bundle sirasi kullanilacak.",
            safe_features_bundle, safe_features_meta,
        )
        safe_features = safe_features_bundle
    else:
        safe_features = safe_features_meta or safe_features_bundle

    if not safe_features:
        raise SystemExit("safe_feature_columns / feature_names bos; tahmin uretilemez.")

    confirm_leakage_guard(model_metadata, safe_features)

    reference_path, reference_band = resolve_reference_grid(ctx)
    log.info("Referans grid: %s (band %s)", reference_path, reference_band)

    feature_paths, landcover_info = resolve_feature_rasters(safe_features, reference_path, ctx)
    log.info("Feature rasterlari cozuldu: %s", {k: str(v) for k, v in feature_paths.items()})

    validate_grid_alignment(reference_path, feature_paths)

    run_result = run_prediction(
        model, safe_features, reference_path, reference_band, feature_paths,
        out_dir, tile_size, write_residual_products,
    )

    plots_written: list[str] = []
    if make_plots:
        plots_written = write_plots(run_result, out_dir, STEP7D_PLOT_SAMPLE_SIZE)

    if run_result["stats"]["out_of_range_prediction_count"] > 0:
        warnings_list.append(
            f"{run_result['stats']['out_of_range_prediction_count']} predicted "
            f"pixels fall outside [{STEP7D_MIN_PREDICTED_CELSIUS}, "
            f"{STEP7D_MAX_PREDICTED_CELSIUS}] C. Values were NOT clamped and "
            "are written as-is in the output raster."
        )

    metadata_path_out = write_metadata(
        out_dir, Path(model_path), Path(model_metadata_path), model_metadata,
        safe_features, reference_path, feature_paths, run_result, tile_size,
        write_residual_products, plots_written, warnings_list, landcover_info,
        ctx=ctx,
    )
    stats_path_out = write_stats(out_dir, run_result, tile_size)
    summary_path_out = write_summary(
        out_dir, model_metadata, safe_features, run_result,
        write_residual_products, warnings_list, landcover_info,
    )

    log.info("Tahmin: %s", run_result["output_paths"]["predicted"])
    log.info("Gecerli maske: %s", run_result["output_paths"]["valid_mask"])
    log.info("Metadata: %s", metadata_path_out)
    log.info("Stats: %s", stats_path_out)
    log.info("Summary: %s", summary_path_out)
    log.info(
        "Kapsam: %.2f%% (%d/%d piksel)",
        run_result["stats"]["prediction_coverage_pct"],
        run_result["stats"]["valid_prediction_pixels"],
        run_result["stats"]["total_pixels"],
    )
    log.info("=" * 60)
    log.info("STEP 7D TAMAMLANDI (no fire-risk model, no burned-area/FIRMS labels used)")
    log.info("=" * 60)

    return {
        "predicted_path": run_result["output_paths"]["predicted"],
        "valid_mask_path": run_result["output_paths"]["valid_mask"],
        "metadata_path": str(metadata_path_out),
        "stats_path": str(stats_path_out),
        "summary_path": str(summary_path_out),
        "stats": run_result["stats"],
        "experiment_id": ctx["experiment_id"] if ctx else None,
    }


def run_step7d(ctx: dict | None = None, force: bool = False, **kwargs) -> dict:
    """
    Step7C modelini tam Manavgat/Kozan grid'ine uygular (windowed inference).

    ctx: None ise (varsayılan) legacy Kozan davranışı BİREBİR korunur.
        Verilirse (Kozan-dışı): model outputs/experiments/<id>/step7c/'den,
        referans grid ctx["step5_output_dir"]'den, feature rasterları
        namespaced Step5/Step5C + shared DEM + Step6A landcover'dan okunur;
        çıktı outputs/experiments/<id>/step7d/'ye yazılır.
    """
    use_ctx = ctx is not None and not ctx.get("is_kozan")
    model_path = None
    model_metadata_path = None
    output_dir = None
    if use_ctx:
        model_path = str(ctx["step7c_output_dir"] / "downscaling_model.joblib")
        model_metadata_path = str(ctx["step7c_output_dir"] / "downscaling_model_metadata.json")
        output_dir = str(ctx["step7d_output_dir"])
        log.info(
            "[experiment=%s] Step7D ctx override aktif. model_path=%s, "
            "output_dir=%s", ctx["experiment_id"], model_path, output_dir,
        )

    kwargs_final = dict(kwargs)
    if model_path is not None:
        kwargs_final["model_path"] = model_path
    if model_metadata_path is not None:
        kwargs_final["model_metadata_path"] = model_metadata_path
    if output_dir is not None:
        kwargs_final["output_dir"] = output_dir

    return main(force=force, ctx=ctx if use_ctx else None, **kwargs_final)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step7D: apply the trained Step7C downscaling model to the "
        "full raster grid (no fire-risk model, no burned-area/FIRMS labels)."
    )
    parser.add_argument("--model", type=str, default=STEP7D_MODEL_PATH)
    parser.add_argument("--model-metadata", type=str, default=STEP7D_MODEL_METADATA_PATH)
    parser.add_argument("--output-dir", type=str, default=STEP7D_OUTPUT_DIR)
    parser.add_argument("--tile-size", type=int, default=STEP7D_TILE_SIZE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-residual-products", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model,
        model_metadata_path=args.model_metadata,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        force=args.force,
        write_residual_products=not args.no_residual_products,
        make_plots=args.plot,
    )