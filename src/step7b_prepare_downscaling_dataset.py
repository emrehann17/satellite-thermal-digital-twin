"""
step7b_prepare_downscaling_dataset.py

MODIS -> Landsat LST downscaling için pencere/tile-bazlı EĞİTİM VERİSETİ hazırlar.

ÖNEMLİ:
    - Step7B model EĞİTMEZ (RF/XGBoost yok), fire-risk modeli ÜRETMEZ.
    - Step5/Step5C/Step6 bilimsel çıktılarını DEĞİŞTİRMEZ.
    - Yalnızca temiz, hizalanmış tabular örnekler üretir:
        target  = yüksek çözünürlüklü Landsat LST (Celsius)
        features= kaba MODIS context + NDVI + DEM (elevation/slope) + land cover
                  + koordinatlar (+ opsiyonel TVDI/anomaly).
    - Bellek dostu: rasterio windows/tiles; tüm rasterlar aynı anda RAM'e alınmaz.

Çıktılar:
    outputs/step7b/downscaling_training_samples.parquet  (pyarrow varsa)
    outputs/step7b/downscaling_training_samples.csv
    outputs/step7b/downscaling_dataset_stats.json
    outputs/step7b/downscaling_dataset_summary.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import transform as window_transform

from core.config import (
    STEP7B_TILE_SIZE,
    STEP7B_MAX_SAMPLES,
    STEP7B_RANDOM_SEED,
    STEP7B_SAMPLE_FRACTION,
    STEP7B_STRATIFY_BY_MODIS_PIXEL,
    STEP7B_MIN_TARGET_CELSIUS,
    STEP7B_MAX_TARGET_CELSIUS,
    STEP7B_REQUIRE_NDVI,
    STEP7B_REQUIRE_DEM,
    STEP7B_OUTPUT_FORMATS,
    STEP7B_INCLUDE_OPTIONAL_TVDI_FEATURES,
    STEP7B_INCLUDE_OPTIONAL_ANOMALY_FEATURES,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from utils.tiling import iter_windows

BASE_DIR = PROJECT_ROOT
OUTPUTS_DIR = BASE_DIR / "outputs" / "step7b"

log, log_file = setup_logger("step7b")

# Fiziksel makul aralıklar (clamp DEĞİL; bu aralık dışı satırlar DROP edilir).
NDVI_MIN, NDVI_MAX = -1.2, 1.2
SLOPE_MIN, SLOPE_MAX = 0.0, 90.0
ELEVATION_MIN, ELEVATION_MAX = -500.0, 9000.0


# =============================================================================
# Target + feature kaynak çözümleme
# =============================================================================
def resolve_target() -> tuple[Path | None, int]:
    """Landsat LST target rasterını çözer (öncelik: step5 celsius -> current_period)."""
    candidates = [
        (BASE_DIR / "outputs" / "step5" / "current_period_median_celsius.tif", 1),
        (BASE_DIR / "data" / "current_period" / "landsat_current_period_60days.tif", 1),
    ]
    for path, band in candidates:
        if path.exists():
            return path, band
    # 60days dosyası farklı gün sayısıyla olabilir; glob ile dene.
    cp_dir = BASE_DIR / "data" / "current_period"
    if cp_dir.exists():
        for path in sorted(cp_dir.glob("landsat_current_period_*days.tif")):
            if "(" in path.name:
                continue
            return path, 1
    return None, 1


def build_feature_registry(
    include_tvdi: bool = STEP7B_INCLUDE_OPTIONAL_TVDI_FEATURES,
    include_anomaly: bool = STEP7B_INCLUDE_OPTIONAL_ANOMALY_FEATURES,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Feature kaynaklarını çözer.

    Döner: (core_features, optional_features, missing_optional)
    Her giriş: {name, path, band, resampling}
    resampling: "bilinear" (sürekli) | "nearest" (kategorik/binary)
    """
    data = BASE_DIR / "data"
    s5 = BASE_DIR / "outputs" / "step5"
    s5c = BASE_DIR / "outputs" / "step5c"

    def first_existing(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    core: list[dict] = []

    # 1) MODIS LST context (mean zorunlu çekirdek; std/zscore varsa eklenir)
    modis_mean = first_existing([
        s5 / "modis_lst_mean_celsius_resampled.tif",
        data / "modis" / "modis_lst_dogu_akdeniz_4y_summer_mean.tif",
    ])
    if modis_mean is not None:
        core.append({"name": "modis_lst_mean_celsius", "path": modis_mean,
                     "band": 1, "resampling": "bilinear", "required": False})

    modis_std = first_existing([s5 / "modis_lst_std_celsius_resampled.tif"])
    if modis_std is not None:
        core.append({"name": "modis_lst_std_celsius", "path": modis_std,
                     "band": 1, "resampling": "bilinear", "required": False})

    modis_z = first_existing([s5 / "modis_context_zscore.tif"])
    if modis_z is not None:
        core.append({"name": "modis_context_zscore", "path": modis_z,
                     "band": 1, "resampling": "bilinear", "required": False})

    # 2) NDVI
    ndvi = first_existing([data / "ndvi_current_period" / "current_ndvi_median.tif"])
    if ndvi is not None:
        core.append({"name": "ndvi", "path": ndvi, "band": 1,
                     "resampling": "bilinear", "required": STEP7B_REQUIRE_NDVI})

    # 3) DEM elevation + slope
    elevation = first_existing([data / "dem" / "elevation.tif"])
    if elevation is not None:
        core.append({"name": "elevation", "path": elevation, "band": 1,
                     "resampling": "bilinear", "required": STEP7B_REQUIRE_DEM})
    slope = first_existing([data / "dem" / "slope.tif"])
    if slope is not None:
        core.append({"name": "slope", "path": slope, "band": 1,
                     "resampling": "bilinear", "required": STEP7B_REQUIRE_DEM})

    # 4) Land cover (kategorik -> nearest)
    landcover = first_existing([
        data / "landcover" / "landcover_esa_worldcover_v200.tif",
    ])
    if landcover is None and (data / "landcover").exists():
        for p in sorted((data / "landcover").glob("*.tif")):
            if "(" not in p.name:
                landcover = p
                break
    if landcover is not None:
        core.append({"name": "landcover", "path": landcover, "band": 1,
                     "resampling": "nearest", "required": False})

    # Opsiyonel context features
    optional: list[dict] = []
    missing: list[dict] = []

    def add_optional(name: str, candidates: list[Path], enabled: bool) -> None:
        if not enabled:
            return
        p = first_existing(candidates)
        if p is not None:
            optional.append({"name": name, "path": p, "band": 1,
                             "resampling": "bilinear", "required": False})
        else:
            missing.append({"name": name, "candidates": [str(c) for c in candidates]})

    add_optional("anomaly_zscore", [s5 / "anomaly_zscore.tif"], include_anomaly)
    add_optional("current_tvdi", [s5c / "current_tvdi.tif"], include_tvdi)
    add_optional("tvdi_difference", [s5c / "tvdi_difference.tif"], include_tvdi)

    return core, optional, missing


# =============================================================================
# Pencere bazlı feature okuma (target grid'ine resample)
# =============================================================================
def _resampling_enum(name: str) -> Resampling:
    return Resampling.nearest if name == "nearest" else Resampling.bilinear


def read_feature_into_target_window(
    feature_src,
    feature_resampling: str,
    target_window,
    target_transform,
    target_crs,
    win_h: int,
    win_w: int,
    same_grid: bool,
) -> np.ndarray:
    """
    Bir feature rasterını target pencere grid'ine okur/resample eder.

    same_grid=True ise doğrudan pencere okunur (resample yok). Aksi halde
    rasterio.warp.reproject ile target pencere transform/CRS'ine yansıtılır.
    """
    if same_grid:
        arr = feature_src.read(1, window=target_window, masked=True)
        return arr.astype("float32").filled(np.nan)

    dst = np.full((win_h, win_w), np.nan, dtype="float32")
    reproject(
        source=rasterio.band(feature_src, 1),
        destination=dst,
        src_transform=feature_src.transform,
        src_crs=feature_src.crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=_resampling_enum(feature_resampling),
        dst_nodata=np.nan,
    )
    return dst


# =============================================================================
# Ana veri seti üretimi
# =============================================================================
def build_dataset(
    target_path: Path,
    target_band: int,
    core_features: list[dict],
    optional_features: list[dict],
    tile_size: int,
    max_samples: int | None,
    sample_fraction: float | None,
    stratify: bool,
    seed: int,
    modis_source: Path | None,
) -> tuple["dict", dict]:
    """
    Pencere pencere geçerli örnekleri toplar (chunked) ve sayaçları döndürür.

    Tam rasterları RAM'e almaz; her pencere için target + feature pencereleri
    okunur, geçerli maske kurulur ve örnekler biriktirilir.
    """
    rng = np.random.default_rng(seed)

    # Tüm feature rasterlarını aç (dosya tanıtıcıları; veriyi yüklemez).
    feature_handles = []
    all_features = core_features + optional_features

    counters = {
        "total_candidate_pixels": 0,
        "total_valid_samples_before_cap": 0,
        "dropped_nan_target": 0,
        "dropped_invalid_target_range": 0,
        "dropped_nan_required_features": 0,
        "dropped_invalid_ndvi": 0,
        "dropped_invalid_slope": 0,
        "dropped_invalid_elevation": 0,
        "window_count": 0,
    }

    chunks: list[dict] = []

    with rasterio.open(target_path) as target_src:
        target_crs = target_src.crs
        feature_meta = []
        for feat in all_features:
            src = rasterio.open(feat["path"])
            same_grid = (
                src.crs == target_crs
                and src.width == target_src.width
                and src.height == target_src.height
                and src.transform == target_src.transform
            )
            feature_handles.append(src)
            feature_meta.append({**feat, "same_grid": same_grid})

        try:
            req_names = [f["name"] for f in core_features if f.get("required")]
            ndvi_present = any(f["name"] == "ndvi" for f in all_features)
            slope_present = any(f["name"] == "slope" for f in all_features)
            elev_present = any(f["name"] == "elevation" for f in all_features)

            for write_win, _read_win, _core_off in iter_windows(
                target_src, tile_size_pixels=tile_size, overlap_pixels=0
            ):
                counters["window_count"] += 1
                win_h = int(write_win.height)
                win_w = int(write_win.width)
                if win_h == 0 or win_w == 0:
                    continue

                target_transform = window_transform(write_win, target_src.transform)

                target_arr = target_src.read(
                    target_band, window=write_win, masked=True
                ).astype("float32").filled(np.nan)
                counters["total_candidate_pixels"] += int(target_arr.size)

                # Feature pencereleri (target grid'ine resample)
                feat_arrays: dict[str, np.ndarray] = {}
                for src, meta in zip(feature_handles, feature_meta):
                    feat_arrays[meta["name"]] = read_feature_into_target_window(
                        src, meta["resampling"], write_win,
                        target_transform, target_crs, win_h, win_w,
                        meta["same_grid"],
                    )

                # --- Geçerlilik maskesi (clamp YOK; satır DROP) ---
                valid = np.isfinite(target_arr)
                counters["dropped_nan_target"] += int((~valid).sum())

                in_range = (
                    (target_arr >= STEP7B_MIN_TARGET_CELSIUS)
                    & (target_arr <= STEP7B_MAX_TARGET_CELSIUS)
                )
                bad_range = valid & (~in_range)
                counters["dropped_invalid_target_range"] += int(bad_range.sum())
                valid &= in_range

                # Zorunlu feature'lar finite olmalı
                for rname in req_names:
                    if rname in feat_arrays:
                        finite = np.isfinite(feat_arrays[rname])
                        counters["dropped_nan_required_features"] += int(
                            (valid & (~finite)).sum()
                        )
                        valid &= finite

                # NDVI fiziksel aralık
                if ndvi_present:
                    nd = feat_arrays.get("ndvi")
                    nd_finite = np.isfinite(nd)
                    nd_in_range = nd_finite & (nd >= NDVI_MIN) & (nd <= NDVI_MAX)
                    if STEP7B_REQUIRE_NDVI:
                        # Zorunlu: NaN veya aralık-dışı -> drop.
                        counters["dropped_invalid_ndvi"] += int(
                            (valid & (~nd_in_range)).sum()
                        )
                        valid &= nd_in_range
                    else:
                        # Opsiyonel: yalnız finite-ama-aralık-dışı drop; NaN'a izin ver.
                        bad = valid & nd_finite & (~nd_in_range)
                        counters["dropped_invalid_ndvi"] += int(bad.sum())
                        valid &= (nd_in_range | ~nd_finite)

                # Slope fiziksel aralık (varsa)
                if slope_present:
                    sl = feat_arrays.get("slope")
                    sl_finite = np.isfinite(sl)
                    sl_ok = sl_finite & (sl >= SLOPE_MIN) & (sl <= SLOPE_MAX)
                    if STEP7B_REQUIRE_DEM:
                        counters["dropped_invalid_slope"] += int((valid & (~sl_ok)).sum())
                        valid &= sl_ok
                    else:
                        bad = valid & sl_finite & (~sl_ok)
                        counters["dropped_invalid_slope"] += int(bad.sum())
                        valid &= (sl_ok | ~sl_finite)

                # Elevation fiziksel aralık (varsa)
                if elev_present:
                    el = feat_arrays.get("elevation")
                    el_finite = np.isfinite(el)
                    el_ok = el_finite & (el >= ELEVATION_MIN) & (el <= ELEVATION_MAX)
                    if STEP7B_REQUIRE_DEM:
                        counters["dropped_invalid_elevation"] += int(
                            (valid & (~el_ok)).sum()
                        )
                        valid &= el_ok
                    else:
                        bad = valid & el_finite & (~el_ok)
                        counters["dropped_invalid_elevation"] += int(bad.sum())
                        valid &= (el_ok | ~el_finite)

                n_valid = int(valid.sum())
                if n_valid == 0:
                    continue
                counters["total_valid_samples_before_cap"] += n_valid

                rows_local, cols_local = np.where(valid)
                # Global piksel koordinatları
                global_rows = rows_local + int(write_win.row_off)
                global_cols = cols_local + int(write_win.col_off)

                # lon/lat (piksel merkezleri) target grid transform'undan
                xs, ys = rasterio.transform.xy(
                    target_src.transform,
                    global_rows.tolist(),
                    global_cols.tolist(),
                    offset="center",
                )
                lon = np.asarray(xs, dtype="float64")
                lat = np.asarray(ys, dtype="float64")

                chunk = {
                    "row": global_rows.astype("int32"),
                    "col": global_cols.astype("int32"),
                    "lon": lon,
                    "lat": lat,
                    "landsat_lst_celsius": target_arr[valid].astype("float32"),
                    "source_window_row": np.full(n_valid, int(write_win.row_off), "int32"),
                    "source_window_col": np.full(n_valid, int(write_win.col_off), "int32"),
                    "source_tile_id": np.full(
                        n_valid, counters["window_count"] - 1, "int32"
                    ),
                }
                for meta in feature_meta:
                    name = meta["name"]
                    chunk[name] = feat_arrays[name][valid].astype("float32")

                chunks.append(chunk)
        finally:
            for src in feature_handles:
                src.close()

    # Birleştir
    dataset = _concat_chunks(chunks)
    if dataset:
        # MODIS pixel id (stratifikasyon + grouped validation için) — örnekleme ÖNCESİ.
        mid = _modis_pixel_ids(dataset, modis_source)
        if mid is not None:
            dataset["_modis_pixel_id"] = mid  # geçici stratifikasyon anahtarı
            dataset["modis_pixel_id"] = mid.astype("int64")  # kalıcı kolon
    final = _sample_dataset(
        dataset, max_samples, sample_fraction, stratify, rng, counters
    )
    return final, counters


def _concat_chunks(chunks: list[dict]) -> dict:
    """Pencere chunk'larını tek sözlükte birleştirir."""
    if not chunks:
        return {}
    keys = chunks[0].keys()
    out = {}
    for k in keys:
        out[k] = np.concatenate([c[k] for c in chunks])
    return out


def _modis_pixel_ids(dataset: dict, modis_path: Path | None) -> np.ndarray | None:
    """
    Her örneğin lon/lat'ını MODIS kaynak grid'ine eşleyerek modis_pixel_id üretir.

    Stratifikasyon ve gelecekteki grouped validation (leakage önleme) için. MODIS
    kaynak rasterı yoksa None döner.
    """
    if modis_path is None or not modis_path.exists():
        return None
    try:
        with rasterio.open(modis_path) as msrc:
            inv = ~msrc.transform
            cols, rows = inv * (dataset["lon"], dataset["lat"])
            mrow = np.floor(rows).astype("int64")
            mcol = np.floor(cols).astype("int64")
            dataset["modis_pixel_row"] = mrow.astype("int32")
            dataset["modis_pixel_col"] = mcol.astype("int32")
            width = msrc.width
            return (mrow * width + mcol).astype("int64")
    except Exception:  # noqa: BLE001
        return None


def _sample_dataset(
    dataset: dict,
    max_samples: int | None,
    sample_fraction: float | None,
    stratify: bool,
    rng: np.random.Generator,
    counters: dict,
) -> dict:
    """Deterministik (opsiyonel stratified) örnekleme + cap uygular."""
    if not dataset:
        return dataset
    n = dataset["row"].size

    # sample_fraction önce uygulanır (varsa)
    target_n = n
    if sample_fraction is not None and 0 < sample_fraction < 1:
        target_n = int(n * sample_fraction)
    if max_samples is not None:
        target_n = min(target_n, max_samples)
    if target_n >= n:
        return dataset  # cap gereksiz

    modis_id = dataset.get("_modis_pixel_id")
    if stratify and modis_id is not None and modis_id.size == n:
        idx = _stratified_indices(modis_id, target_n, rng)
        counters["sampling_strategy"] = "stratified_by_modis_pixel"
    else:
        idx = rng.choice(n, size=target_n, replace=False)
        counters["sampling_strategy"] = (
            "uniform_random" if not stratify else "uniform_random_no_modis_grid"
        )

    idx.sort()
    return {k: v[idx] for k, v in dataset.items() if not k.startswith("_")}


def _stratified_indices(
    group_id: np.ndarray, target_n: int, rng: np.random.Generator
) -> np.ndarray:
    """
    MODIS pikseli başına dengeli örnekleme: tek bir kaba pikselden çok örnek alma.

    Her gruba round-robin ile kota dağıtılır; deterministik (seed'li rng).
    """
    order = rng.permutation(group_id.size)
    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(int(group_id[i]), []).append(int(i))

    selected: list[int] = []
    group_lists = list(groups.values())
    pos = 0
    while len(selected) < target_n and group_lists:
        advanced = False
        for g in group_lists:
            if pos < len(g):
                selected.append(g[pos])
                advanced = True
                if len(selected) >= target_n:
                    break
        if not advanced:
            break
        pos += 1
    return np.asarray(selected[:target_n], dtype="int64")


# =============================================================================
# Çıktı yazımı
# =============================================================================
def _column_order(dataset: dict) -> list[str]:
    """Required + opsiyonel kolonları istenen sırayla düzenler."""
    preferred = [
        "row", "col", "lon", "lat", "landsat_lst_celsius",
        "modis_lst_mean_celsius", "modis_lst_std_celsius", "modis_context_zscore",
        "ndvi", "elevation", "slope", "landcover",
        "anomaly_zscore", "current_tvdi", "tvdi_difference",
        "source_tile_id", "source_window_row", "source_window_col",
        "modis_pixel_row", "modis_pixel_col", "modis_pixel_id",
    ]
    cols = [c for c in preferred if c in dataset]
    # Beklenmeyen ekstra kolonları da sona ekle
    cols += [c for c in dataset if c not in cols and not c.startswith("_")]
    return cols


def write_outputs(
    dataset: dict, formats: list[str], force: bool
) -> dict:
    """Parquet/CSV yazar. pyarrow yoksa CSV yazılır, parquet_written=False."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = _column_order(dataset)
    n = dataset[cols[0]].size if cols else 0

    result = {
        "parquet_written": False,
        "csv_written": False,
        "parquet_available": False,
        "output_files": [],
        "columns": cols,
    }

    csv_path = OUTPUTS_DIR / "downscaling_training_samples.csv"
    parquet_path = OUTPUTS_DIR / "downscaling_training_samples.parquet"

    want_parquet = "parquet" in formats
    want_csv = "csv" in formats

    # Parquet (pyarrow varsa)
    if want_parquet:
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq
            result["parquet_available"] = True
            table = pa.table({c: dataset[c] for c in cols})
            pq.write_table(table, parquet_path)
            result["parquet_written"] = True
            result["output_files"].append(str(parquet_path))
        except Exception as exc:  # noqa: BLE001
            log.warning("Parquet yazılamadı (pyarrow yok/başarısız): %s", exc)

    # CSV (her zaman dene; en az bir format başarılı olmalı)
    if want_csv or not result["parquet_written"]:
        try:
            _write_csv(csv_path, dataset, cols)
            result["csv_written"] = True
            result["output_files"].append(str(csv_path))
        except Exception as exc:  # noqa: BLE001
            log.error("CSV yazılamadı: %s", exc)

    result["final_sample_count"] = int(n)
    return result


def _write_csv(path: Path, dataset: dict, cols: list[str]) -> None:
    """Bağımlılıksız CSV yazımı (numpy ile satır satır birleştirme)."""
    arrays = [dataset[c] for c in cols]
    n = arrays[0].size
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        # Vektörize string birleştirme (chunked, bellek dostu)
        chunk = 50000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            lines = []
            for i in range(start, end):
                vals = []
                for a in arrays:
                    v = a[i]
                    if isinstance(v, (np.floating, float)):
                        vals.append("" if not np.isfinite(v) else repr(float(v)))
                    else:
                        vals.append(str(int(v)))
                lines.append(",".join(vals))
            f.write("\n".join(lines) + "\n")


def write_stats_and_summary(
    target_path: Path,
    target_band: int,
    core_features: list[dict],
    optional_features: list[dict],
    missing_optional: list[dict],
    counters: dict,
    out_result: dict,
    grid_info: dict,
    tile_size: int,
    max_samples: int | None,
    sample_fraction: float | None,
    stratify: bool,
    seed: int,
    warnings_list: list[str],
) -> tuple[Path, Path]:
    """downscaling_dataset_stats.json + downscaling_dataset_summary.md yazar."""
    required_features = [f["name"] for f in core_features if f.get("required")]
    optional_used = [f["name"] for f in optional_features]
    optional_missing = [m["name"] for m in missing_optional]

    stats = {
        "step": "step7b_prepare_downscaling_dataset",
        "created_at": datetime.now().isoformat(),
        "no_model_trained": True,
        "target_path": str(target_path),
        "target_band": target_band,
        "target_name": "landsat_lst_celsius",
        "feature_paths": {f["name"]: str(f["path"]) for f in core_features + optional_features},
        "reference_grid": grid_info,
        "tile_size": tile_size,
        "window_count": counters.get("window_count"),
        "total_candidate_pixels": counters.get("total_candidate_pixels"),
        "total_valid_samples_before_cap": counters.get("total_valid_samples_before_cap"),
        "final_sample_count": out_result.get("final_sample_count"),
        "dropped_nan_target": counters.get("dropped_nan_target"),
        "dropped_invalid_target_range": counters.get("dropped_invalid_target_range"),
        "dropped_nan_required_features": counters.get("dropped_nan_required_features"),
        "dropped_invalid_ndvi": counters.get("dropped_invalid_ndvi"),
        "dropped_invalid_slope": counters.get("dropped_invalid_slope"),
        "dropped_invalid_elevation": counters.get("dropped_invalid_elevation"),
        "output_files": out_result.get("output_files"),
        "columns": out_result.get("columns"),
        "parquet_written": out_result.get("parquet_written"),
        "parquet_available": out_result.get("parquet_available"),
        "csv_written": out_result.get("csv_written"),
        "random_seed": seed,
        "max_samples": max_samples,
        "sample_fraction": sample_fraction,
        "stratify_by_modis_pixel": stratify,
        "sampling_strategy": counters.get("sampling_strategy", "no_cap_applied"),
        "required_features": required_features,
        "optional_features_used": optional_used,
        "optional_features_missing": optional_missing,
        "target_valid_range_celsius": [
            STEP7B_MIN_TARGET_CELSIUS, STEP7B_MAX_TARGET_CELSIUS,
        ],
        "warnings": warnings_list,
    }

    stats_path = OUTPUTS_DIR / "downscaling_dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Step7B: MODIS Downscaling Training Dataset",
        "",
        "**Step7B prepares a MODIS->Landsat LST downscaling training dataset. "
        "No model is trained here.** This is **not** fire-risk validation.",
        "",
        f"- Created at: `{stats['created_at']}`",
        f"- Target: `landsat_lst_celsius` from `{target_path.name}` (band {target_band})",
        "- Features include: MODIS LST context, NDVI, DEM elevation, slope, "
        "land cover, and spatial coordinates"
        + (" (+ optional TVDI/anomaly)" if optional_used else "") + ".",
        f"- Tile size (pixels): `{tile_size}`",
        f"- Windows processed: `{counters.get('window_count')}`",
        f"- Total candidate pixels: `{counters.get('total_candidate_pixels')}`",
        f"- Valid samples before cap: `{counters.get('total_valid_samples_before_cap')}`",
        f"- **Final sample count: `{out_result.get('final_sample_count')}`**",
        f"- Sampling strategy: `{stats['sampling_strategy']}`",
        "",
        "## Dropped (masked, not clamped)",
        "",
        f"- NaN target: `{counters.get('dropped_nan_target')}`",
        f"- Target out of Celsius range "
        f"[{STEP7B_MIN_TARGET_CELSIUS}, {STEP7B_MAX_TARGET_CELSIUS}]: "
        f"`{counters.get('dropped_invalid_target_range')}`",
        f"- NaN required features: `{counters.get('dropped_nan_required_features')}`",
        f"- Invalid NDVI: `{counters.get('dropped_invalid_ndvi')}`",
        f"- Invalid slope: `{counters.get('dropped_invalid_slope')}`",
        f"- Invalid elevation: `{counters.get('dropped_invalid_elevation')}`",
        "",
        "## Features",
        "",
        f"- Required: `{', '.join(required_features) or 'none'}`",
        f"- Optional used: `{', '.join(optional_used) or 'none'}`",
        f"- Optional missing: `{', '.join(optional_missing) or 'none'}`",
        "",
        "## Outputs",
        "",
    ]
    for f in out_result.get("output_files", []):
        md.append(f"- `{f}`")
    md.extend([
        f"- parquet_written: `{out_result.get('parquet_written')}` "
        f"(pyarrow available: `{out_result.get('parquet_available')}`)",
        f"- csv_written: `{out_result.get('csv_written')}`",
        "",
        "## Intended use",
        "",
        "This dataset is intended for **later** RF/XGBoost or similar MODIS "
        "downscaling (Step7C / Step8). No model is trained in Step7B, and this "
        "is kept separate from any fire-risk modeling.",
    ])
    if warnings_list:
        md.extend(["", "## Warnings", ""])
        md.extend(f"- {w}" for w in warnings_list)

    summary_path = OUTPUTS_DIR / "downscaling_dataset_summary.md"
    summary_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return stats_path, summary_path


# =============================================================================
# Ana akış
# =============================================================================
def main(
    max_samples: int | None = STEP7B_MAX_SAMPLES,
    tile_size: int = STEP7B_TILE_SIZE,
    output_format: str = "both",
    force: bool = False,
    include_optional_tvdi: bool = STEP7B_INCLUDE_OPTIONAL_TVDI_FEATURES,
    include_optional_anomaly: bool = STEP7B_INCLUDE_OPTIONAL_ANOMALY_FEATURES,
) -> dict:
    log.info("=" * 60)
    log.info("STEP 7B BAŞLIYOR (MODIS downscaling training dataset)")
    log.info("=" * 60)

    warnings_list: list[str] = []

    # Overwrite kontrolü
    existing = [
        OUTPUTS_DIR / "downscaling_training_samples.parquet",
        OUTPUTS_DIR / "downscaling_training_samples.csv",
        OUTPUTS_DIR / "downscaling_dataset_stats.json",
        OUTPUTS_DIR / "downscaling_dataset_summary.md",
    ]
    if any(p.exists() for p in existing) and not force:
        present = [p.name for p in existing if p.exists()]
        raise SystemExit(
            "Step7B çıktıları zaten var ("
            + ", ".join(present)
            + "). Üzerine yazmak için --force verin."
        )

    target_path, target_band = resolve_target()
    if target_path is None:
        raise SystemExit(
            "Landsat LST target rasterı bulunamadı. Beklenen: "
            "outputs/step5/current_period_median_celsius.tif veya "
            "data/current_period/landsat_current_period_*days.tif"
        )
    log.info("Target: %s (band %s)", target_path, target_band)

    core_features, optional_features, missing_optional = build_feature_registry(
        include_tvdi=include_optional_tvdi,
        include_anomaly=include_optional_anomaly,
    )
    log.info(
        "Core features: %s | Optional used: %s | Optional missing: %s",
        [f["name"] for f in core_features],
        [f["name"] for f in optional_features],
        [m["name"] for m in missing_optional],
    )

    # Zorunlu feature eksikse uyar (NDVI/DEM)
    core_names = {f["name"] for f in core_features}
    if STEP7B_REQUIRE_NDVI and "ndvi" not in core_names:
        warnings_list.append("Required NDVI feature not found; samples may be empty.")
    if STEP7B_REQUIRE_DEM and ("elevation" not in core_names or "slope" not in core_names):
        warnings_list.append("Required DEM (elevation/slope) feature missing.")

    # Grid info
    with rasterio.open(target_path) as tsrc:
        grid_info = {
            "path": str(target_path),
            "width": int(tsrc.width),
            "height": int(tsrc.height),
            "crs": str(tsrc.crs),
            "transform": [tsrc.transform.a, tsrc.transform.b, tsrc.transform.c,
                          tsrc.transform.d, tsrc.transform.e, tsrc.transform.f],
        }
        if not tsrc.crs:
            warnings_list.append("Target raster has no CRS.")
        if tsrc.width == 0 or tsrc.height == 0:
            raise SystemExit("Target raster has invalid dimensions.")

    modis_source = None
    for f in core_features:
        if f["name"] == "modis_lst_mean_celsius":
            modis_source = f["path"]
            break

    dataset, counters = build_dataset(
        target_path, target_band, core_features, optional_features,
        tile_size=tile_size,
        max_samples=max_samples,
        sample_fraction=STEP7B_SAMPLE_FRACTION,
        stratify=STEP7B_STRATIFY_BY_MODIS_PIXEL,
        seed=STEP7B_RANDOM_SEED,
        modis_source=modis_source,
    )

    if dataset and "modis_pixel_id" not in dataset:
        warnings_list.append(
            "MODIS source grid unavailable; modis_pixel_id not added."
        )

    if not dataset or dataset.get("row") is None or dataset["row"].size == 0:
        warnings_list.append("No valid samples produced.")
        out_result = {
            "final_sample_count": 0, "output_files": [], "columns": [],
            "parquet_written": False, "parquet_available": False, "csv_written": False,
        }
    else:
        formats = (
            ["parquet", "csv"] if output_format == "both" else [output_format]
        )
        out_result = write_outputs(dataset, formats, force)

    stats_path, summary_path = write_stats_and_summary(
        target_path, target_band, core_features, optional_features, missing_optional,
        counters, out_result, grid_info, tile_size, max_samples,
        STEP7B_SAMPLE_FRACTION, STEP7B_STRATIFY_BY_MODIS_PIXEL,
        STEP7B_RANDOM_SEED, warnings_list,
    )

    log.info("Final sample count: %s", out_result.get("final_sample_count"))
    log.info("Stats: %s", stats_path)
    log.info("Summary: %s", summary_path)
    log.info("=" * 60)
    log.info("STEP 7B TAMAMLANDI (no model trained)")
    log.info("=" * 60)

    return {
        "final_sample_count": out_result.get("final_sample_count"),
        "stats_path": str(stats_path),
        "summary_path": str(summary_path),
        "output_files": out_result.get("output_files"),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step7B: MODIS downscaling training dataset builder (no model trained)."
    )
    parser.add_argument("--max-samples", type=int, default=STEP7B_MAX_SAMPLES)
    parser.add_argument("--tile-size", type=int, default=STEP7B_TILE_SIZE)
    parser.add_argument(
        "--output-format", choices=["csv", "parquet", "both"], default="both"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-optional-tvdi", action="store_true")
    parser.add_argument("--no-optional-anomaly", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        max_samples=args.max_samples,
        tile_size=args.tile_size,
        output_format=args.output_format,
        force=args.force,
        include_optional_tvdi=not args.no_optional_tvdi and STEP7B_INCLUDE_OPTIONAL_TVDI_FEATURES,
        include_optional_anomaly=not args.no_optional_anomaly and STEP7B_INCLUDE_OPTIONAL_ANOMALY_FEATURES,
    )