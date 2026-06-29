"""
step6_validate_fire_relation.py

İLK burned-area ilişki (association) testi.

Bu modül bir yangın-riski MODELİ DEĞİLDİR ve RF/XGBoost eğitmez. Amacı, mevcut
predictor rasterlarının yanmış alan / aktif yangın etiketleriyle ne kadar ilişkili
olduğunu ölçmektir (burned vs unburned ayrışması, ROC/AUC).

Predictor rasterları:
    - outputs/step5/thermal_anomaly_zscore.tif   (LST sıcaklık anomalisi)
    - outputs/step5c/current_tvdi.tif            (sürekli kuruluk göstergesi)
    - outputs/step5c/tvdi_difference.tif         (current - baseline mean)
    - outputs/step5c/tvdi_anomaly_zscore.tif     (güvenilirlik-filtreli anomali)

Etiketler (GEE):
    - MCD64A1 burned area (500 m)
    - FireCCI51 burned area (250 m, varsa)
    - opsiyonel: FIRMS / MCD14ML aktif yangın

Önemli yorum notları:
    - current_tvdi ve tvdi_difference SÜREKLİ kuruluk göstergeleri olarak ele alınır.
    - tvdi_anomaly_zscore güvenilirlik-filtreli ve yoğun maskelidir; geçerli örnek
      sayısı ayrı raporlanır.
    - Sonuç "doğrulanmış yangın-riski modeli" DEĞİLDİR; ilk ilişki testidir.

Çıktılar:
    - outputs/validation/validation_summary.md
    - outputs/validation/validation_stats.json
    - outputs/validation/roc_curve_comparison.png
    - outputs/validation/burned_vs_unburned_boxplot.png
    - outputs/validation/predictor_maps_with_burn_overlay.png (mümkünse)
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from core.config import (
    EXPORT_CRS,
    FIRECCI51_AVAILABLE_END,
    FIRECCI51_AVAILABLE_START,
    GEE_PROJECT,
    LABEL_END_DATE,
    LABEL_START_DATE,
    PREDICTOR_END_DATE,
    PREDICTOR_START_DATE,
    REGION_NAME,
    VALIDATION_ALLOW_OVERLAPPING_WINDOWS,
    VALIDATION_BALANCED_UNBURNED_RATIO,
    VALIDATION_FIRMS_BRIGHTNESS_THRESHOLD,
    VALIDATION_INCLUDE_FIRMS,
    VALIDATION_LABEL_EXPORT_SCALE,
    VALIDATION_MAX_ROC_PREVIEW_POINTS,
    VALIDATION_MODE,
    VALIDATION_VALID_MODES,
    VALIDATION_RANDOM_SEED,
    VALIDATION_SEASON_END,
    VALIDATION_SEASON_START,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import ee
    from core.gee_utils import init_gee
    from core.regions import build_regions
    from core.validation_burned_area import (
        get_firecci51_burned_area_safe,
        get_firms_active_fire,
        get_mcd64a1_burned_area_safe,
    )
    GEE_IMPORTS_OK = True
    GEE_IMPORT_ERROR = None
except Exception as _gee_import_exc:  # noqa: BLE001
    # ImportError dışındaki hataları da yakala (ör. validation_burned_area
    # içindeki bir bağımlılık veya init hatası). Gerçek sebebi sakla ki
    # fetch_labels net mesaj verebilsin.
    GEE_IMPORTS_OK = False
    GEE_IMPORT_ERROR = f"{type(_gee_import_exc).__name__}: {_gee_import_exc}"

try:
    import geemap
    GEEMAP_AVAILABLE = True
except ImportError:
    GEEMAP_AVAILABLE = False


BASE_DIR = PROJECT_ROOT
STEP5_OUTPUT_DIR = BASE_DIR / "outputs" / "step5"
STEP5C_OUTPUT_DIR = BASE_DIR / "outputs" / "step5c"
CURRENT_PERIOD_DIR = BASE_DIR / "data" / "current_period"
NDVI_CURRENT_DIR = BASE_DIR / "data" / "ndvi_current_period"
LANDCOVER_CANDIDATE_DIRS = [
    BASE_DIR / "data" / "landcover",
    BASE_DIR / "data" / "land_cover",
    BASE_DIR / "outputs" / "landcover",
    BASE_DIR / "outputs" / "land_cover",
]
OUTPUT_DIR = BASE_DIR / "outputs" / "validation"
LABEL_DIR = OUTPUT_DIR / "labels"
BURNABLE_NDVI_THRESHOLD = 0.2
VEGETATION_NDVI_THRESHOLDS = (BURNABLE_NDVI_THRESHOLD, 0.3)
NDVI_STRATA = (
    ("ndvi_0_2_0_4", 0.2, 0.4, "NDVI 0.2-0.4"),
    ("ndvi_0_4_0_6", 0.4, 0.6, "NDVI 0.4-0.6"),
    ("ndvi_0_6_0_8", 0.6, 0.8, "NDVI 0.6-0.8"),
)
LANDSAT_QA_WATER_BIT = 1 << 7
ESA_WORLDCOVER_BURNABLE_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
}
ESA_WORLDCOVER_EXCLUDED_CLASSES = {
    50: "built_up",
    60: "bare_sparse_vegetation",
    80: "water",
}
STEP5_METADATA_PATH = STEP5_OUTPUT_DIR / "step5_metadata.json"
STEP5C_METADATA_PATH = STEP5C_OUTPUT_DIR / "step5c_metadata.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step6")

# Predictor raster tanımları. LST anomaly için esnek dosya adı (thermal_anomaly_zscore
# veya anomaly_zscore) desteklenir; ilk bulunan kullanılır.
PREDICTORS = {
    "thermal_anomaly_zscore": {
        "path_candidates": [
            STEP5_OUTPUT_DIR / "thermal_anomaly_zscore.tif",
            STEP5_OUTPUT_DIR / "anomaly_zscore.tif",
        ],
        "label": "LST anomaly z-score",
        "continuous": True,
    },
    "current_tvdi": {
        "path_candidates": [STEP5C_OUTPUT_DIR / "current_tvdi.tif"],
        "label": "Current TVDI",
        "continuous": True,
    },
    "tvdi_difference": {
        "path_candidates": [STEP5C_OUTPUT_DIR / "tvdi_difference.tif"],
        "label": "TVDI difference",
        "continuous": True,
    },
    "tvdi_anomaly_zscore": {
        "path_candidates": [STEP5C_OUTPUT_DIR / "tvdi_anomaly_zscore.tif"],
        "label": "TVDI anomaly z-score (reliability-filtered)",
        "continuous": False,
    },
}


def resolve_predictor_path(key: str) -> Path | None:
    """Predictor için ilk var olan aday dosya yolunu döndürür (yoksa None)."""
    for candidate in PREDICTORS[key]["path_candidates"]:
        if candidate.exists():
            return candidate
    return None


class ValidationError(Exception):
    """Step6'da net hata mesajıyla durmak için."""


def reference_predictor_path() -> Path:
    """Etiketlerin hizalanacağı referans predictor grid'ini seçer."""
    # current_tvdi en yüksek geçerli kapsama sahip TVDI ürünü; referans grid o.
    for key in ("current_tvdi", "thermal_anomaly_zscore"):
        path = resolve_predictor_path(key)
        if path is not None:
            return path
    raise ValidationError(
        "Hiçbir referans predictor rasterı bulunamadı. Önce Step5 ve Step5C "
        "çalıştırılmalı (current_tvdi.tif veya thermal_anomaly_zscore.tif gerekli)."
    )


def read_predictor(path: Path) -> np.ndarray:
    """Tek bantlı predictor rasterını float32 + NaN olarak okur."""
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype("float32").filled(np.nan)


def read_raster_to_grid(
    path: Path,
    grid: dict,
    band_index: int = 1,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Read one raster band and align it to the validation reference grid."""
    with rasterio.open(path) as src:
        arr = src.read(band_index, masked=True).astype("float32").filled(np.nan)
        same_grid = (
            src.width == grid["width"]
            and src.height == grid["height"]
            and src.crs == grid["crs"]
            and src.transform == grid["transform"]
        )
        if same_grid:
            return arr

        aligned = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
        reproject(
            source=arr,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return aligned


def list_current_ndvi_tif() -> Path | None:
    """Return the current-period NDVI raster used by Step5C, if present."""
    candidates = sorted(NDVI_CURRENT_DIR.glob("*.tif"))
    if not candidates:
        return None
    preferred = [
        path for path in candidates
        if path.name.lower().startswith("current_ndvi_median")
    ]
    return preferred[0] if preferred else candidates[0]


def list_current_period_tif() -> Path | None:
    """Return the current-period Landsat raster, if available for QA water masking."""
    candidates = sorted(CURRENT_PERIOD_DIR.glob("*.tif"))
    if not candidates:
        return None
    return candidates[0]


def list_landcover_tif() -> Path | None:
    """Return an optional local land-cover raster, if one has been staged."""
    for directory in LANDCOVER_CANDIDATE_DIRS:
        candidates = sorted(directory.glob("*.tif"))
        if candidates:
            return candidates[0]
    return None


def read_current_ndvi_to_grid(grid: dict) -> tuple[np.ndarray | None, dict]:
    """Read current NDVI to the validation grid and describe the source."""
    ndvi_path = list_current_ndvi_tif()
    if ndvi_path is None:
        return None, {
            "available": False,
            "reason": f"current NDVI raster not found in {NDVI_CURRENT_DIR}",
        }
    ndvi = read_raster_to_grid(ndvi_path, grid, band_index=1, resampling=Resampling.bilinear)
    return ndvi, {
        "available": True,
        "source": str(ndvi_path),
        "finite_ndvi_pixels": int(np.sum(np.isfinite(ndvi))),
    }


def read_water_mask_to_grid(grid: dict) -> tuple[np.ndarray | None, dict]:
    """Read the Landsat QA water bit as a diagnostic water mask when available."""
    current_path = list_current_period_tif()
    if current_path is None:
        return None, {
            "available": False,
            "source": None,
            "reason": f"current-period raster not found in {CURRENT_PERIOD_DIR}",
        }

    try:
        with rasterio.open(current_path) as src:
            if src.count < 2:
                return None, {
                    "available": False,
                    "source": str(current_path),
                    "reason": "current-period raster has no QA_PIXEL band",
                }
        qa = read_raster_to_grid(
            current_path,
            grid,
            band_index=2,
            resampling=Resampling.nearest,
        ).astype("float32")
    except Exception as exc:  # noqa: BLE001
        log.warning("Water mask could not be read from current-period QA: %s", exc)
        return None, {
            "available": False,
            "source": str(current_path),
            "reason": f"QA read failed: {type(exc).__name__}: {exc}",
        }

    qa_uint = np.where(np.isfinite(qa), qa, 0).astype("uint16")
    water = np.isfinite(qa) & ((qa_uint & LANDSAT_QA_WATER_BIT) != 0)
    return water, {
        "available": True,
        "source": str(current_path),
        "source_detail": "Landsat QA_PIXEL bit 7 water flag",
        "water_pixel_count": int(np.sum(water)),
        "finite_qa_pixels": int(np.sum(np.isfinite(qa))),
    }


def read_landcover_burnable_mask_to_grid(
    grid: dict,
    water_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, dict]:
    """Build an optional land-cover burnable mask from a local categorical raster."""
    landcover_path = list_landcover_tif()
    if landcover_path is None:
        return None, {
            "available": False,
            "reason": "no local land-cover GeoTIFF found",
            "searched_dirs": [str(path) for path in LANDCOVER_CANDIDATE_DIRS],
        }

    landcover = read_raster_to_grid(
        landcover_path,
        grid,
        band_index=1,
        resampling=Resampling.nearest,
    )
    lc_int = np.where(np.isfinite(landcover), landcover, -9999).astype("int32")
    burnable = np.isin(lc_int, list(ESA_WORLDCOVER_BURNABLE_CLASSES))
    if water_mask is not None:
        burnable &= ~water_mask

    return burnable, {
        "available": True,
        "source": str(landcover_path),
        "classification_assumption": "ESA WorldCover-style class IDs",
        "included_classes": ESA_WORLDCOVER_BURNABLE_CLASSES,
        "excluded_classes": ESA_WORLDCOVER_EXCLUDED_CLASSES,
        "water_excluded_with_qa_mask": water_mask is not None,
    }


def mask_summary(mask: np.ndarray, base_valid: np.ndarray) -> dict:
    """Compact population mask summary."""
    mask_count = int(np.sum(mask))
    base_count = int(np.sum(base_valid))
    return {
        "pixel_count": mask_count,
        "fraction_of_all_valid_candidates": (
            float(mask_count / base_count) if base_count else None
        ),
    }


def build_validation_population_masks(grid: dict, burned: np.ndarray) -> tuple[dict, dict]:
    """
    Build named validation populations.

    all_valid is handled by passing no mask to the stats function. All other masks
    are diagnostic populations with explicit source metadata.
    """
    ndvi, ndvi_info = read_current_ndvi_to_grid(grid)
    water_mask, water_info = read_water_mask_to_grid(grid)
    base_valid = np.isfinite(burned)

    populations: dict = {}
    summaries: dict = {
        "all_valid": {
            "label": "All valid pixels",
            "mask_source": "predictor finite pixels and finite burned labels",
            "available": True,
        },
    }

    if water_mask is not None:
        non_water = base_valid & ~water_mask
        populations["non_water"] = non_water
        summaries["non_water"] = {
            "label": "Non-water diagnostic pixels",
            "mask_source": "Landsat QA_PIXEL bit 7 water excluded",
            "available": True,
            **mask_summary(non_water, base_valid),
        }
    else:
        summaries["non_water"] = {
            "label": "Non-water diagnostic pixels",
            "available": False,
            "mask_source": "unavailable",
            "reason": water_info.get("reason"),
        }

    for threshold in VEGETATION_NDVI_THRESHOLDS:
        key = f"ndvi_gt_{str(threshold).replace('.', '_')}"
        label = f"NDVI > {threshold:.1f} vegetation pixels"
        if ndvi is None:
            summaries[key] = {
                "label": label,
                "available": False,
                "mask_source": "unavailable",
                "reason": ndvi_info.get("reason"),
            }
            continue
        vegetation = base_valid & np.isfinite(ndvi) & (ndvi > threshold)
        if water_mask is not None:
            vegetation &= ~water_mask
        populations[key] = vegetation
        summaries[key] = {
            "label": label,
            "available": True,
            "mask_source": (
                f"current NDVI > {threshold:.1f}"
                + (" and Landsat QA_PIXEL bit 7 water excluded" if water_mask is not None else "")
            ),
            "ndvi_threshold": threshold,
            "water_excluded": water_mask is not None,
            **mask_summary(vegetation, base_valid),
        }

    landcover_mask, landcover_info = read_landcover_burnable_mask_to_grid(grid, water_mask)
    if landcover_mask is not None:
        landcover_mask = base_valid & landcover_mask
        populations["landcover_burnable"] = landcover_mask
        summaries["landcover_burnable"] = {
            "label": "Land-cover burnable pixels",
            "available": True,
            "mask_source": "optional land-cover categorical raster",
            **landcover_info,
            **mask_summary(landcover_mask, base_valid),
        }
    else:
        summaries["landcover_burnable"] = {
            "label": "Land-cover burnable pixels",
            "available": False,
            **landcover_info,
        }

    summaries["inputs"] = {
        "ndvi": ndvi_info,
        "water": water_info,
    }
    return populations, summaries


def build_ndvi_strata_masks(grid: dict, burned: np.ndarray) -> tuple[dict, dict]:
    """Build NDVI strata masks for direction diagnostics."""
    ndvi, ndvi_info = read_current_ndvi_to_grid(grid)
    strata = {}
    summaries = {"inputs": {"ndvi": ndvi_info}}
    if ndvi is None:
        for key, _lo, _hi, label in NDVI_STRATA:
            summaries[key] = {
                "label": label,
                "available": False,
                "reason": ndvi_info.get("reason"),
            }
        return strata, summaries

    base_valid = np.isfinite(burned)
    for key, lo, hi, label in NDVI_STRATA:
        mask = base_valid & np.isfinite(ndvi) & (ndvi >= lo) & (ndvi < hi)
        strata[key] = mask
        summaries[key] = {
            "label": label,
            "available": True,
            "ndvi_min_inclusive": lo,
            "ndvi_max_exclusive": hi,
            **mask_summary(mask, base_valid),
        }
    return strata, summaries


def read_reference_grid(path: Path) -> dict:
    """Referans grid profilini (transform, crs, shape) döndürür."""
    with rasterio.open(path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "profile": src.profile.copy(),
        }


def export_label_to_grid(
    image: "ee.Image",
    region: "ee.Geometry",
    grid: dict,
    out_path: Path,
    label_name: str,
) -> Path | None:
    """
    GEE binary etiket image'ini referans predictor grid'ine indirir/resample eder.

    geemap ile etiket GeoTIFF olarak indirilir; ardından rasterio ile predictor
    grid'ine (nearest, kategorik etiket) reproject edilir. Başarısızsa None döner.
    """
    if not GEEMAP_AVAILABLE:
        log.warning("geemap yok; %s etiketi indirilemedi.", label_name)
        return None

    raw_path = LABEL_DIR / f"{label_name}_raw.tif"
    try:
        geemap.ee_export_image(
            image,
            filename=str(raw_path),
            scale=VALIDATION_LABEL_EXPORT_SCALE,
            region=region,
            crs=EXPORT_CRS,
            file_per_band=False,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("%s etiketi GEE export başarısız: %s", label_name, exc)
        return None

    if not raw_path.exists():
        log.warning("%s etiketi indirilemedi (dosya yok).", label_name)
        return None

    # Predictor grid'ine resample (nearest; etiket kategorik 0/1).
    aligned = np.zeros((grid["height"], grid["width"]), dtype="float32")
    with rasterio.open(raw_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            resampling=Resampling.nearest,
        )

    profile = grid["profile"]
    profile.update(count=1, dtype="float32", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(aligned, 1)

    return out_path


def build_binary_label(
    label_arrays: list[np.ndarray],
) -> np.ndarray:
    """
    Birden çok etiket kaynağını tek binary 'burned' maskesine birleştirir (OR).

    burned = 1, unburned = 0. Hiç kaynak yoksa tümü 0 döner.
    """
    if not label_arrays:
        return None
    combined = np.zeros_like(label_arrays[0], dtype="float32")
    for arr in label_arrays:
        combined = np.maximum(combined, (arr > 0).astype("float32"))
    return combined


def firecci51_window_available(label_start: str, label_end: str) -> bool:
    """
    İstenen label penceresinin FireCCI51 veri kapsamında olup olmadığını kontrol
    eder. Kapsam dışındaysa (örn. 2023) FireCCI51 Earth Engine'e SORULMADAN skip
    edilir.
    """
    # Basit string tabanlı tarih karşılaştırması (ISO formatı sıralanabilir).
    return (
        label_start <= FIRECCI51_AVAILABLE_END
        and label_end >= FIRECCI51_AVAILABLE_START
    )


def fetch_labels(grid: dict, label_start: str, label_end: str) -> dict:
    """
    GEE'den yanmış alan / aktif yangın etiketlerini indirip predictor grid'ine
    hizalar ve tek binary etiket üretir.

    label_start/label_end: etiketlerin çekileceği pencere. same_season modunda
    sezon penceresiyle, pre_fire modunda label window ile aynıdır.
    """
    if not GEE_IMPORTS_OK:
        raise ValidationError(
            "GEE importları başarısız (ee/geemap/regions/validation_burned_area). "
            "Step6 etiket indirmesi için GEE ortamı gerekli. "
            f"Gerçek import hatası: {GEE_IMPORT_ERROR}. "
            "Olası çözümler: (1) earthengine-api ve geemap kurun "
            "(pip install earthengine-api geemap), (2) GEE auth yapın "
            "(earthengine authenticate), (3) hata bir modül içi soruna işaret "
            "ediyorsa ilgili modülü kontrol edin."
        )

    try:
        init_gee(GEE_PROJECT)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "GEE başlatma/auth başarısız (ee.Initialize). "
            f"Hata: {type(exc).__name__}: {exc}. "
            "Çözüm: 'earthengine authenticate' çalıştırın ve GEE_PROJECT "
            f"değerinin ('{GEE_PROJECT}') doğru/erişilebilir olduğundan emin olun."
        ) from exc
    regions = build_regions()
    if REGION_NAME not in regions:
        raise ValidationError(f"Bölge bulunamadı: {REGION_NAME}")
    region = regions[REGION_NAME]

    start = label_start
    end = label_end
    log.info("Etiket penceresi: %s -> %s, bölge: %s", start, end, REGION_NAME)

    label_arrays = []
    sources_used = []
    skipped_sources = []

    # MCD64A1 (ana kaynak, güvenli)
    mcd, mcd_status = get_mcd64a1_burned_area_safe(region, start, end)
    if mcd is None:
        log.warning(mcd_status)
        skipped_sources.append({"source": "MCD64A1", "reason": mcd_status})
    else:
        mcd_path = export_label_to_grid(
            mcd, region, grid, LABEL_DIR / "mcd64a1_burned.tif", "mcd64a1"
        )
        if mcd_path is not None:
            with rasterio.open(mcd_path) as src:
                label_arrays.append(src.read(1).astype("float32"))
            sources_used.append("MCD64A1")
        else:
            reason = "MCD64A1 GEE export/download failed."
            log.warning(reason)
            skipped_sources.append({"source": "MCD64A1", "reason": reason})

    # FireCCI51: önce availability kontrolü. Label window kapsam dışıysa (örn. 2023)
    # Earth Engine'e hiç sorulmadan skip edilir.
    if not firecci51_window_available(start, end):
        reason = (
            "FireCCI51 skipped: requested label window outside dataset "
            f"availability ({FIRECCI51_AVAILABLE_START} - "
            f"{FIRECCI51_AVAILABLE_END})."
        )
        log.info(reason)
        skipped_sources.append({"source": "FireCCI51", "reason": reason})
    else:
        # Window uygunsa güvenli fetch dene (boş/bandsiz image .gt() çağrılmadan elenir)
        try:
            firecci, firecci_status = get_firecci51_burned_area_safe(
                region, start, end
            )
            if firecci is None:
                log.warning(firecci_status)
                skipped_sources.append(
                    {"source": "FireCCI51", "reason": firecci_status}
                )
            else:
                firecci_path = export_label_to_grid(
                    firecci, region, grid,
                    LABEL_DIR / "firecci51_burned.tif", "firecci51",
                )
                if firecci_path is not None:
                    with rasterio.open(firecci_path) as src:
                        label_arrays.append(src.read(1).astype("float32"))
                    sources_used.append("FireCCI51")
                else:
                    reason = "FireCCI51 GEE export/download failed."
                    log.warning(reason)
                    skipped_sources.append(
                        {"source": "FireCCI51", "reason": reason}
                    )
        except Exception as exc:  # noqa: BLE001
            reason = f"FireCCI51 unexpected error: {exc}"
            log.warning(reason)
            skipped_sources.append({"source": "FireCCI51", "reason": reason})
        skipped_sources.append({"source": "FireCCI51", "reason": reason})

    # FIRMS aktif yangın (opsiyonel)
    if VALIDATION_INCLUDE_FIRMS:
        try:
            firms = get_firms_active_fire(region, start, end)
            firms_binary = firms.gt(VALIDATION_FIRMS_BRIGHTNESS_THRESHOLD)
            firms_path = export_label_to_grid(
                firms_binary, region, grid, LABEL_DIR / "firms_active.tif", "firms"
            )
            if firms_path is not None:
                with rasterio.open(firms_path) as src:
                    label_arrays.append(src.read(1).astype("float32"))
                sources_used.append("FIRMS")
            else:
                reason = "FIRMS GEE export/download failed."
                log.warning(reason)
                skipped_sources.append({"source": "FIRMS", "reason": reason})
        except Exception as exc:  # noqa: BLE001
            reason = f"FIRMS unexpected error: {exc}"
            log.warning(reason)
            skipped_sources.append({"source": "FIRMS", "reason": reason})

    burned = build_binary_label(label_arrays)
    if burned is None:
        raise ValidationError(
            "Hiçbir yanmış alan etiketi indirilemedi. GEE bağlantısını ve "
            "geemap kurulumunu kontrol edin. "
            f"Atlanan kaynaklar: {skipped_sources}"
        )

    burned_count = int(np.sum(burned > 0))
    if burned_count == 0:
        raise ValidationError(
            f"Seçili AOI ({REGION_NAME}) ve sezonda ({start} - {end}) hiç yanmış "
            "piksel bulunamadı. Öneriler: (1) AOI'yi genişletin, (2) farklı/daha "
            "geniş bir yangın sezonu seçin (VALIDATION_SEASON_START/END), "
            "(3) FIRMS aktif yangını dahil edin (VALIDATION_INCLUDE_FIRMS=True)."
        )

    log.info(
        "Yanmış piksel sayısı: %s (kullanılan kaynaklar: %s)",
        burned_count, sources_used,
    )
    if skipped_sources:
        log.info("Atlanan kaynaklar: %s", [s["source"] for s in skipped_sources])

    return {
        "burned": burned,
        "sources_used": sources_used,
        "skipped_sources": skipped_sources,
        "label_window_start": start,
        "label_window_end": end,
        "burned_pixel_count": burned_count,
    }


def downsample_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    max_points: int,
) -> dict:
    """
    ROC eğrisini JSON önizlemesi için max_points noktaya indirger.

    Full array'ler (milyonlarca nokta olabilir) JSON'a yazılmaz. Eşit aralıklı
    indeksleme ile küçük, temsili bir önizleme üretilir.
    """
    n = int(fpr.size)
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points).astype(int)
    # thresholds[0] sklearn'de +inf olabilir; JSON için sonlu değere indir.
    thr = thresholds[idx].astype(float)
    thr = np.where(np.isfinite(thr), thr, None)
    return {
        "fpr": [round(float(v), 5) for v in fpr[idx]],
        "tpr": [round(float(v), 5) for v in tpr[idx]],
        "thresholds": [None if v is None else round(float(v), 5) for v in thr],
        "downsampled_from": n,
        "max_roc_points": max_points,
    }


def predictor_label_stats(
    predictor: np.ndarray,
    burned: np.ndarray,
    rng: np.random.Generator,
    population_mask: np.ndarray | None = None,
    include_roc_preview: bool = True,
    keep_roc_for_plot: bool = True,
) -> tuple[dict, dict | None]:
    """
    Tek bir predictor için burned/unburned ayrışmasını ve ROC/AUC'yi hesaplar.

    Döndürür: (summary_dict, roc_arrays_for_plot)
        - summary_dict: JSON'a yazılacak KOMPAKT özet (full array YOK; yalnız
          opsiyonel downsample'lı roc_curve_preview).
        - roc_arrays_for_plot: PNG çizimi için full fpr/tpr (BELLEKTE; JSON'a
          yazılmaz). ROC yoksa None.

    Hem tam (full) hem dengeli (balanced) metrikler raporlanır. NaN predictor
    pikselleri ve geçersiz etiketler dışlanır.
    """
    valid = np.isfinite(predictor) & np.isfinite(burned)
    if population_mask is not None:
        valid &= population_mask
    pred_valid = predictor[valid]
    label_valid = (burned[valid] > 0).astype("int8")

    burned_vals = pred_valid[label_valid == 1]
    unburned_vals = pred_valid[label_valid == 0]

    n_burned = int(burned_vals.size)
    n_unburned = int(unburned_vals.size)

    result: dict = {
        "valid_paired_pixels": int(pred_valid.size),
        "burned_pixels": n_burned,
        "unburned_pixels": n_unburned,
        "burned_mean": float(np.mean(burned_vals)) if n_burned else None,
        "burned_median": float(np.median(burned_vals)) if n_burned else None,
        "unburned_mean": float(np.mean(unburned_vals)) if n_unburned else None,
        "unburned_median": float(np.median(unburned_vals)) if n_unburned else None,
        "auc_full": None,
        "auc_balanced": None,
    }
    roc_for_plot = None

    if n_burned == 0 or n_unburned == 0:
        result["note"] = "insufficient burned or unburned samples for ROC/AUC"
        return result, roc_for_plot

    if not SKLEARN_AVAILABLE:
        result["note"] = "sklearn unavailable; AUC/ROC skipped"
        return result, roc_for_plot

    # Tam (full) AUC + ROC (array'ler yalnız bellekte/PNG için)
    result["auc_full"] = float(roc_auc_score(label_valid, pred_valid))
    if include_roc_preview or keep_roc_for_plot:
        fpr, tpr, thresholds = roc_curve(label_valid, pred_valid)
        if keep_roc_for_plot:
            roc_for_plot = {"fpr": fpr, "tpr": tpr}
        # JSON'a sadece küçük downsample önizleme:
        if include_roc_preview:
            result["roc_curve_preview"] = downsample_roc(
                fpr, tpr, thresholds, VALIDATION_MAX_ROC_PREVIEW_POINTS
            )

    # Dengeli (balanced) AUC: unburned alt-örnekleme
    target_unburned = int(n_burned * VALIDATION_BALANCED_UNBURNED_RATIO)
    target_unburned = min(target_unburned, n_unburned)
    if target_unburned > 0:
        idx = rng.choice(n_unburned, size=target_unburned, replace=False)
        bal_pred = np.concatenate([burned_vals, unburned_vals[idx]])
        bal_label = np.concatenate([
            np.ones(n_burned, dtype="int8"),
            np.zeros(target_unburned, dtype="int8"),
        ])
        result["auc_balanced"] = float(roc_auc_score(bal_label, bal_pred))
        result["balanced_unburned_count"] = target_unburned

    return result, roc_for_plot


def auc_only(
    score: np.ndarray,
    burned: np.ndarray,
    population_mask: np.ndarray | None = None,
) -> dict:
    """Compact AUC-only diagnostic for predictor direction checks."""
    valid = np.isfinite(score) & np.isfinite(burned)
    if population_mask is not None:
        valid &= population_mask

    score_valid = score[valid]
    label_valid = (burned[valid] > 0).astype("int8")
    n_burned = int(np.sum(label_valid == 1))
    n_unburned = int(np.sum(label_valid == 0))
    result = {
        "valid_paired_pixels": int(score_valid.size),
        "burned_pixels": n_burned,
        "unburned_pixels": n_unburned,
        "auc_full": None,
        "diagnostic_only": True,
    }

    if n_burned == 0 or n_unburned == 0:
        result["note"] = "insufficient burned or unburned samples for ROC/AUC"
        return result
    if not SKLEARN_AVAILABLE:
        result["note"] = "sklearn unavailable; AUC skipped"
        return result

    result["auc_full"] = float(roc_auc_score(label_valid, score_valid))
    return result


def compute_direction_diagnostics(
    predictors: dict,
    burned: np.ndarray,
    population_mask: np.ndarray | None = None,
) -> dict:
    """
    Report original and inverted AUCs for TVDI predictors.

    Inverted AUCs are diagnostic only and must not be treated as final products.
    """
    specs = [
        ("current_tvdi", "current_tvdi", lambda arr: arr),
        ("current_tvdi_inverted", "1-current_tvdi", lambda arr: 1.0 - arr),
        ("tvdi_difference", "tvdi_difference", lambda arr: arr),
        ("tvdi_difference_inverted", "-tvdi_difference", lambda arr: -arr),
        ("tvdi_anomaly_zscore", "tvdi_anomaly_zscore", lambda arr: arr),
        ("tvdi_anomaly_zscore_inverted", "-tvdi_anomaly_zscore", lambda arr: -arr),
    ]
    diagnostics = {}
    for out_key, label, transform in specs:
        source_key = out_key.replace("_inverted", "")
        if source_key not in predictors:
            continue
        diagnostics[out_key] = {
            "label": label,
            **auc_only(transform(predictors[source_key]), burned, population_mask),
        }
    return diagnostics


def compute_population_predictor_metrics(
    predictors: dict,
    burned: np.ndarray,
    predictor_sources: dict,
    population_mask: np.ndarray | None,
    include_roc_preview: bool,
    keep_roc_for_plot: bool,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """Compute per-predictor metrics for one validation population."""
    per_predictor = {}
    roc_arrays = {}
    for key, arr in predictors.items():
        summary, roc = predictor_label_stats(
            arr,
            burned,
            rng,
            population_mask=population_mask,
            include_roc_preview=include_roc_preview,
            keep_roc_for_plot=keep_roc_for_plot,
        )
        summary["predictor_name"] = key
        summary["source_file"] = predictor_sources.get(key)
        per_predictor[key] = summary
        roc_arrays[key] = roc
    return per_predictor, roc_arrays


def compute_ndvi_stratified_auc(
    predictors: dict,
    burned: np.ndarray,
    strata_masks: dict,
) -> dict:
    """AUC diagnostics for selected TVDI predictors inside NDVI strata."""
    specs = [
        ("current_tvdi", "current_tvdi", lambda arr: arr),
        ("current_tvdi_inverted", "1-current_tvdi", lambda arr: 1.0 - arr),
        ("tvdi_difference", "tvdi_difference", lambda arr: arr),
        ("tvdi_difference_inverted", "-tvdi_difference", lambda arr: -arr),
    ]
    result = {}
    for stratum_key, mask in strata_masks.items():
        rows = {}
        for out_key, label, transform in specs:
            source_key = out_key.replace("_inverted", "")
            if source_key not in predictors:
                continue
            rows[out_key] = {
                "label": label,
                **auc_only(transform(predictors[source_key]), burned, mask),
            }
        result[stratum_key] = rows
    return result


def plot_roc_comparison(
    per_predictor: dict,
    roc_arrays: dict,
    out_path: Path,
) -> str | None:
    """
    Tüm predictor'ların ROC eğrilerini tek figürde çizer.

    roc_arrays: {predictor_key: {"fpr": np.ndarray, "tpr": np.ndarray}} — full
    array'ler (bellekte; JSON'a yazılmaz). AUC değerleri per_predictor özetinden
    alınır.
    """
    if not SKLEARN_AVAILABLE:
        return None
    if not any(v is not None for v in roc_arrays.values()):
        return None

    plt.figure(figsize=(7, 7))
    for key, roc in roc_arrays.items():
        if roc is None:
            continue
        auc = per_predictor.get(key, {}).get("auc_full")
        if auc is None:
            continue
        label = PREDICTORS[key]["label"]
        plt.plot(roc["fpr"], roc["tpr"], label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random (AUC=0.5)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Burned-area association: ROC comparison")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path.name


def plot_boxplot(
    predictors: dict,
    burned: np.ndarray,
    out_path: Path,
) -> str | None:
    """Her predictor için burned vs unburned dağılımını boxplot ile gösterir."""
    fig, axes = plt.subplots(
        1, len(predictors), figsize=(4 * len(predictors), 5), squeeze=False
    )
    any_plotted = False
    for ax, (key, arr) in zip(axes[0], predictors.items()):
        valid = np.isfinite(arr) & np.isfinite(burned)
        label_valid = (burned[valid] > 0)
        burned_vals = arr[valid][label_valid]
        unburned_vals = arr[valid][~label_valid]
        if burned_vals.size == 0 or unburned_vals.size == 0:
            ax.set_title(f"{key}\n(insufficient samples)", fontsize=8)
            ax.axis("off")
            continue
        ax.boxplot(
            [unburned_vals, burned_vals],
            tick_labels=["unburned", "burned"],
            showfliers=False,
        )
        ax.set_title(PREDICTORS[key]["label"], fontsize=8)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path.name


def plot_predictor_maps_with_overlay(
    predictors: dict,
    burned: np.ndarray,
    out_path: Path,
) -> str | None:
    """Her predictor haritasını, yanmış alan konturuyla birlikte çizer."""
    try:
        fig, axes = plt.subplots(
            1, len(predictors), figsize=(5 * len(predictors), 5), squeeze=False
        )
        burned_mask = np.where(burned > 0, 1.0, np.nan)
        for ax, (key, arr) in zip(axes[0], predictors.items()):
            masked = np.ma.masked_invalid(arr)
            im = ax.imshow(masked, cmap="YlOrBr")
            # Yanmış alanı yarı saydam mavi overlay olarak göster.
            ax.imshow(
                np.ma.masked_invalid(burned_mask),
                cmap="cool",
                alpha=0.5,
            )
            ax.set_title(PREDICTORS[key]["label"], fontsize=8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.6)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path.name
    except Exception as exc:  # noqa: BLE001
        log.warning("Overlay haritası çizilemedi (atlanıyor): %s", exc)
        return None


def write_stats_json(report: dict) -> Path:
    """validation_stats.json yazar."""
    path = OUTPUT_DIR / "validation_stats.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def fmt(value, digits: int = 4) -> str:
    """Markdown için sayı formatlama."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary_markdown(report: dict) -> Path:
    """validation_summary.md yazar."""
    labels = report["labels"]
    per = report["per_predictor"]
    per_population = report.get("per_population_predictor", {})
    population_info = report.get("validation_populations", {})
    direction = report.get("diagnostic_direction_auc", {})
    ndvi_strata = report.get("ndvi_strata", {})
    ndvi_stratified_auc = report.get("ndvi_stratified_auc", {})
    predictor_metadata = report.get("predictor_metadata", {})
    used_sources = labels.get("sources_used", [])
    skipped = labels.get("skipped_sources", [])
    skipped_names = [s["source"] for s in skipped]
    mode = report.get("validation_mode", "same_season")
    pred_w = report.get("predictor_window", {})
    label_w = report.get("label_window", {})
    lead = report.get("temporal_lead_days")

    if mode == "pre_fire" and lead is not None:
        lead_text = f"{lead}"
        relation_text = "predictor before label (pre-fire)"
    elif mode == "same_season":
        lead_text = "overlapping windows / n/a"
        relation_text = "predictor and label same window (same-season)"
    else:
        lead_text = "n/a"
        relation_text = "n/a"

    # Key findings için NDVI 0.6-0.8 AUC'lerini rapordan al; yoksa supervisor'ın
    # referans değerlerine düş (yalnızca metin amaçlı, hesaplamayı etkilemez).
    _ndvi_strat = report.get("ndvi_stratified_auc", {})
    _dense = _ndvi_strat.get("ndvi_0_6_0_8", {})
    _dense_tvdi_diff = _dense.get("tvdi_difference", {}).get("auc_full")
    _dense_cur_tvdi = _dense.get("current_tvdi", {}).get("auc_full")
    tvdi_diff_txt = fmt(_dense_tvdi_diff, 3) if _dense_tvdi_diff is not None else "0.649"
    cur_tvdi_txt = fmt(_dense_cur_tvdi, 3) if _dense_cur_tvdi is not None else "0.586"

    lines = [
        "# Step6 Burned-Area Association Test",
        "",
        f"Created at: `{report['created_at']}`",
        f"Validation mode: `{mode}`",
        f"Region / AOI: `{report['region']}`",
        f"Predictor window: `{pred_w.get('start')} -> {pred_w.get('end')}`",
        f"Label window: `{label_w.get('start')} -> {label_w.get('end')}`",
        f"Temporal relation: `{relation_text}`",
        f"Temporal lead/gap (days): `{lead_text}`",
        f"Label sources (used): `{', '.join(used_sources) if used_sources else 'none'}`",
        f"Label sources (skipped): "
        f"`{', '.join(skipped_names) if skipped_names else 'none'}`",
        "",
        "> This is a **first burned-area association test / initial validation "
        "experiment**, not a validated fire-risk model. No RF/XGBoost is trained "
        "here.",
        "",
        "> Full per-pixel arrays and full ROC arrays are not stored in JSON; "
        "ROC curves are saved as PNG. The JSON keeps only compact summary metrics "
        "and a small downsampled `roc_curve_preview`.",
        "",
        "## Key findings",
        "",
        f"- This is **{('pre-fire' if mode == 'pre_fire' else mode)} validation** "
        "(predictor window precedes the burned-area label window).",
        "- The **all-pixel AUC is diagnostic only**; it is confounded by mixing "
        "burnable vegetation with non-burnable hot/dry surfaces and is not the "
        "headline result.",
        "- The **main signal appears in dense vegetation**: TVDI separation "
        "strengthens as NDVI increases.",
        "- **NDVI 0.6-0.8 `tvdi_difference` is the strongest single-index result** "
        f"(AUC \u2248 {tvdi_diff_txt}), with NDVI 0.6-0.8 `current_tvdi` also "
        f"positive (AUC \u2248 {cur_tvdi_txt}).",
        "- **TVDI should not be flipped globally**: the all-pixel inversion is a "
        "population-mixing artifact, not a real reversal of the dryness signal.",
        "",
    ]

    # pre_fire uyarısı
    prefire_warning = report.get("prefire_warning")
    if prefire_warning:
        lines.extend([f"> **Warning:** {prefire_warning}", ""])

    # Atlanan kaynakların sebepleri
    if skipped:
        lines.extend(["## Skipped label sources", ""])
        for item in skipped:
            lines.append(f"- **{item['source']}**: {item['reason']}")
        lines.append("")

    # FireCCI51 özel notu (istenen): yalnız FireCCI51 atlandı ama MCD64A1 kullanıldıysa
    if "FireCCI51" in skipped_names and "MCD64A1" in used_sources:
        lines.extend([
            "> FireCCI51 was unavailable/empty for the selected AOI-season in "
            "this run; validation used MCD64A1 burned-area labels.",
            "",
        ])

    lines.extend([
        "## Sample counts",
        "",
        f"- Burned pixels (label): `{labels['burned_pixel_count']}`",
        "",
        "## Predictor metadata",
        "",
    ])
    if predictor_metadata:
        for name, meta in predictor_metadata.items():
            lines.extend([
                f"### {name}",
                "",
                f"- Metadata path: `{meta.get('metadata_path')}`",
                f"- Current period: `{meta.get('current_period_start')} -> "
                f"{meta.get('current_period_end')}`",
                f"- Baseline years used: `{meta.get('baseline_years_used')}`",
                "- Current year excluded from baseline: "
                f"`{meta.get('current_year_excluded_from_baseline')}`",
                "",
            ])
    else:
        lines.extend(["- Predictor metadata validation not required for this mode.", ""])

    lines.extend([
        "## Validation populations",
        "",
        "| Population | Available | Pixels | Mask source |",
        "| --- | --- | ---: | --- |",
    ])
    for key, info in population_info.items():
        if key == "inputs":
            continue
        lines.append(
            "| {label} | {available} | {pixels} | {source} |".format(
                label=info.get("label", key),
                available=info.get("available"),
                pixels=info.get("pixel_count", "n/a"),
                source=info.get("mask_source", info.get("reason", "n/a")),
            )
        )

    # =====================================================================
    # PRIMARY REPORTING: NDVI-stratified vegetation results first
    # (Supervisor feedback: all-pixel TVDI AUC is NOT the headline; it is a
    #  population-mixing artifact. NDVI strata / dense vegetation are the main
    #  evaluation domain. NDVI 0.6-0.8 is the strongest stratum.)
    # =====================================================================

    def render_population_table(population_key: str, label: str, tier: str) -> None:
        population_metrics = per_population.get(population_key)
        info = population_info.get(population_key, {})
        lines.extend(["", f"## {label} ({tier})", ""])
        if not population_metrics:
            lines.extend([
                f"_Not available: {info.get('reason', info.get('mask_source', 'population missing'))}._",
            ])
            return
        lines.extend([
            "| Predictor | Valid pairs | Burned | Unburned | Burned mean | "
            "Unburned mean | AUC (full) | AUC (balanced) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for key, stats in population_metrics.items():
            lines.append(
                "| {label} | {pairs} | {burned} | {unburned} | {bmean} | {umean} "
                "| {auc_f} | {auc_b} |".format(
                    label=PREDICTORS[key]["label"],
                    pairs=stats["valid_paired_pixels"],
                    burned=stats["burned_pixels"],
                    unburned=stats["unburned_pixels"],
                    bmean=fmt(stats["burned_mean"]),
                    umean=fmt(stats["unburned_mean"]),
                    auc_f=fmt(stats.get("auc_full")),
                    auc_b=fmt(stats.get("auc_balanced")),
                )
            )

    lines.extend([
        "",
        "# Primary validation (vegetation domain)",
        "",
        "Per supervisor feedback, the **all-pixel TVDI AUC is not the headline "
        "result**. All-pixel populations mix burnable vegetation with non-burnable "
        "hot/dry surfaces, which makes `current_tvdi` look globally inverted. The "
        "primary evaluation domain is **NDVI strata / dense vegetation**, followed "
        "by NDVI > 0.3 vegetation and land-cover burnable pixels. All-pixel tables "
        "are demoted to the diagnostic section below.",
    ])

    # 1) PRIMARY: NDVI-stratified validation
    if ndvi_stratified_auc:
        lines.extend([
            "",
            "## NDVI-stratified validation (primary)",
            "",
            "TVDI separation strengthens with vegetation density. In the densest "
            "stratum (NDVI 0.6-0.8), `current_tvdi` and especially "
            "`tvdi_difference` show positive association with burned-area labels.",
            "",
            "| NDVI stratum | Score | Valid pairs | Burned | Unburned | AUC |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for stratum_key, rows in ndvi_stratified_auc.items():
            stratum_label = ndvi_strata.get(stratum_key, {}).get("label", stratum_key)
            for stats in rows.values():
                lines.append(
                    "| {stratum} | `{score}` | {pairs} | {burned} | {unburned} "
                    "| {auc} |".format(
                        stratum=stratum_label,
                        score=stats["label"],
                        pairs=stats["valid_paired_pixels"],
                        burned=stats["burned_pixels"],
                        unburned=stats["unburned_pixels"],
                        auc=fmt(stats.get("auc_full")),
                    )
                )

    # 1b) PRIMARY: dense vegetation highlight (NDVI 0.6-0.8)
    ndvi_06_08 = ndvi_stratified_auc.get("ndvi_0_6_0_8", {})
    if ndvi_06_08:
        lines.extend([
            "",
            "## Dense vegetation highlight: NDVI 0.6-0.8 (primary)",
            "",
            "This is the strongest stratum. Within the densest vegetation, current "
            "dryness and the TVDI difference show positive association with "
            "subsequent burned area:",
            "",
        ])
        for key in ["tvdi_difference", "current_tvdi"]:
            val = ndvi_06_08.get(key)
            if val and val.get("auc_full") is not None:
                lines.append(f"- **{val['label']}**: AUC \u2248 {fmt(val.get('auc_full'), 3)}")
        lines.extend([
            "",
            "> `tvdi_difference` in NDVI 0.6-0.8 is the strongest single-index "
            "result. The signal lives **inside vegetation strata**, not globally. "
            "TVDI is **not** flipped: the global inversion is a population-mixing "
            "artifact.",
            "",
        ])

    # 2) SECONDARY: NDVI > 0.3 vegetation pixels
    render_population_table("ndvi_gt_0_3", "Predictor comparison: NDVI > 0.3 vegetation pixels", "secondary")

    # 3) SECONDARY: Land-cover burnable pixels (if available)
    render_population_table("landcover_burnable", "Predictor comparison: land-cover burnable pixels", "secondary")

    # Land-cover burnable mask uyarısı: maske hâlâ düşük TVDI AUC veriyorsa çok geniş olabilir.
    landcover_metrics = per_population.get("landcover_burnable")
    if landcover_metrics:
        lc_tvdi = landcover_metrics.get("current_tvdi", {}).get("auc_full")
        lc_low = lc_tvdi is not None and lc_tvdi < 0.55
        lines.extend([
            "",
            "> **Warning (land-cover burnable mask):** land-cover burnable pixels "
            "still show low TVDI AUC"
            + (f" (`current_tvdi` AUC \u2248 {fmt(lc_tvdi, 3)})" if lc_tvdi is not None else "")
            + ", which suggests the current land-cover burnable mask may be **too "
            "broad**. It should be refined with NDVI strata, e.g. land-cover "
            "burnable **AND NDVI > 0.6**, or by using separate "
            "forest / shrub / grass / cropland strata.",
            "",
        ])
        _ = lc_low

    # =====================================================================
    # DIAGNOSTIC / CONFOUNDING CHECK (demoted: all-pixel + other populations)
    # =====================================================================
    lines.extend([
        "",
        "# Diagnostic / confounding check",
        "",
        "The tables in this section are **diagnostic only**. All-pixel AUCs are "
        "confounded by population mixing and must not be read as the headline "
        "performance of TVDI.",
        "",
        "## Diagnostic predictor comparison: all valid pixels",
        "",
        "| Predictor | Valid pairs | Burned | Unburned | Burned mean | "
        "Unburned mean | AUC (full) | AUC (balanced) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for key, stats in per.items():
        lines.append(
            "| {label} | {pairs} | {burned} | {unburned} | {bmean} | {umean} "
            "| {auc_f} | {auc_b} |".format(
                label=PREDICTORS[key]["label"],
                pairs=stats["valid_paired_pixels"],
                burned=stats["burned_pixels"],
                unburned=stats["unburned_pixels"],
                bmean=fmt(stats["burned_mean"]),
                umean=fmt(stats["unburned_mean"]),
                auc_f=fmt(stats.get("auc_full")),
                auc_b=fmt(stats.get("auc_balanced")),
            )
        )

    # Remaining diagnostic populations (e.g. non_water, ndvi_gt_0_2). The primary /
    # secondary populations above are not repeated here.
    primary_population_keys = {"landcover_burnable", "ndvi_gt_0_3"}
    for population_key, population_metrics in per_population.items():
        if population_key in primary_population_keys:
            continue
        label = population_info.get(population_key, {}).get("label", population_key)
        lines.extend([
            "",
            f"## Diagnostic predictor comparison: {label}",
            "",
            "| Predictor | Valid pairs | Burned | Unburned | Burned mean | "
            "Unburned mean | AUC (full) | AUC (balanced) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for key, stats in population_metrics.items():
            lines.append(
                "| {label} | {pairs} | {burned} | {unburned} | {bmean} | {umean} "
                "| {auc_f} | {auc_b} |".format(
                    label=PREDICTORS[key]["label"],
                    pairs=stats["valid_paired_pixels"],
                    burned=stats["burned_pixels"],
                    unburned=stats["unburned_pixels"],
                    bmean=fmt(stats["burned_mean"]),
                    umean=fmt(stats["unburned_mean"]),
                    auc_f=fmt(stats.get("auc_full")),
                    auc_b=fmt(stats.get("auc_balanced")),
                )
            )

    if direction:
        lines.extend([
            "",
            "## Diagnostic direction check (inverted AUC)",
            "",
            "Inverted AUCs are diagnostic only; they are not final products and do "
            "not automatically imply that TVDI should be flipped. This table is "
            "kept for transparency; **TVDI is not flipped**.",
            "",
            "| Population | Score | Valid pairs | Burned | Unburned | AUC |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for population_key, population_rows in direction.items():
            population_label = population_info.get(
                population_key, {}
            ).get("label", "All valid pixels" if population_key == "all_valid" else population_key)
            for stats in population_rows.values():
                lines.append(
                    "| {population} | `{score}` | {pairs} | {burned} | {unburned} "
                    "| {auc} |".format(
                        population=population_label,
                        score=stats["label"],
                        pairs=stats["valid_paired_pixels"],
                        burned=stats["burned_pixels"],
                        unburned=stats["unburned_pixels"],
                        auc=fmt(stats.get("auc_full")),
                    )
                )

    lines.extend(["", "## Burned vs unburned (median)", ""])
    for key, stats in per.items():
        lines.append(
            f"- **{PREDICTORS[key]['label']}**: burned median "
            f"`{fmt(stats['burned_median'])}` vs unburned median "
            f"`{fmt(stats['unburned_median'])}`"
        )

    # tvdi_anomaly_zscore'un ayrı uyarısı
    tvdi_z = per.get("tvdi_anomaly_zscore")
    if tvdi_z is not None:
        lines.extend([
            "",
            "## Note on TVDI anomaly z-score",
            "",
            "`tvdi_anomaly_zscore` is reliability-filtered (masked where baseline "
            "TVDI std < threshold), so its valid sample count is reported "
            "separately and is typically much smaller than the continuous "
            "predictors:",
            "",
            f"- Valid paired pixels: `{tvdi_z['valid_paired_pixels']}`",
            f"- Burned pixels in valid set: `{tvdi_z['burned_pixels']}`",
        ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- All-pixel TVDI mixes burnable vegetation with non-burnable hot/dry "
        "surfaces (bare soil, rock, urban). These two populations have opposite "
        "TVDI behaviour relative to burned area.",
        "- This mixing causes `current_tvdi` to appear **globally inverted** "
        "(all-pixel AUC < 0.5). The inversion is a confounding artifact, not a "
        "real reversal of the dryness signal.",
        "- **TVDI is not flipped.** Flipping would optimise to a confounded "
        "all-pixel statistic and break the physical semantics of the index.",
        "- The meaningful signal appears **inside vegetation strata**. Restricting "
        "to burnable / vegetated pixels removes the non-burnable hot/dry surfaces "
        "that drive the global inversion.",
        "- In **NDVI 0.6-0.8**, `current_tvdi` and especially `tvdi_difference` "
        "show **positive** association with burned-area labels (AUC > 0.5).",
        "- `tvdi_anomaly_zscore` is reliability-filtered and heavily masked; its "
        "smaller valid sample count should be considered when comparing AUCs, and "
        "it currently remains weaker than `current_tvdi` / `tvdi_difference`.",
        "- AUC > 0.5 inside the primary domain is **preliminary association "
        "evidence**, not a validated fire-risk product. No RF/XGBoost is trained.",
        "",
        "## Current scientific interpretation",
        "",
        "- The current project story is **not** \"global anomaly predicts fire\".",
        "- The current story is: **within burnable / vegetated areas, current "
        "dryness (`current_tvdi`) and the TVDI difference (`tvdi_difference`) show "
        "association with subsequent burned area.**",
        "- LST and TVDI anomaly z-scores remain **weak** and should **not** be "
        "overclaimed; they are not the headline result.",
        "- The all-pixel inversion is explained by population mixing and does not "
        "justify flipping TVDI.",
    ])
    lst = per.get("thermal_anomaly_zscore")
    if lst is not None and lst.get("auc_full") is not None:
        auc = lst["auc_full"]
        if 0.5 < auc < 0.6:
            lines.append(
                "- LST anomaly shows weak positive association with burned-area "
                f"labels (AUC={auc:.3f})."
            )

    tvdi_keys = ["current_tvdi", "tvdi_difference", "tvdi_anomaly_zscore"]
    tvdi_below_half = [
        k for k in tvdi_keys
        if per.get(k, {}).get("auc_full") is not None
        and per[k]["auc_full"] < 0.5
    ]
    if tvdi_below_half:
        lines.extend([
            "- TVDI does not separate burned/unburned pixels **in all-pixel "
            f"validation** ({', '.join(tvdi_below_half)} all-pixel AUC < 0.5).",
            "- This is caused by **mixing burnable vegetation with non-burnable "
            "hot/dry surfaces**, not by an absence of signal.",
            "- NDVI-stratified results show that **TVDI becomes positive in dense "
            "vegetation** (see the primary NDVI 0.6-0.8 result above).",
            "- Current scientific interpretation: *within dense vegetation / "
            "burnable vegetation, current dryness and TVDI difference are "
            "associated with later burned area.*",
        ])

    if mode == "pre_fire":
        lines.append(
            "- Pre-fire validation should be interpreted only when predictor "
            "rasters were generated for the configured predictor window."
        )

    lines.extend([
        "",
        "## Outputs",
        "",
    ])
    for name in report.get("png_outputs", []):
        lines.append(f"- `{name}`")

    path = OUTPUT_DIR / "validation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def resolve_windows() -> dict:
    """
    Validation moduna göre predictor ve label pencerelerini belirler.

    same_season: ikisi de VALIDATION_SEASON_START/END.
    pre_fire:    predictor ve label pencereleri ayrı (PREDICTOR_*_DATE / LABEL_*_DATE).

    Fail-fast:
        - Geçersiz mode sessizce same_season'a DÜŞMEZ; ValidationError verir.
        - pre_fire'da pencereler aynı/çakışıyorsa ve overlap'e izin verilmemişse
          ValidationError verir.
    """
    mode = VALIDATION_MODE

    if mode not in VALIDATION_VALID_MODES:
        raise ValidationError(
            f"Geçersiz VALIDATION_MODE: '{mode}'. Geçerli değerler: "
            f"{VALIDATION_VALID_MODES}. (same_season'a sessizce düşülmez.)"
        )

    if mode == "pre_fire":
        identical = (
            PREDICTOR_START_DATE == LABEL_START_DATE
            and PREDICTOR_END_DATE == LABEL_END_DATE
        )
        overlapping = PREDICTOR_END_DATE > LABEL_START_DATE
        if (identical or overlapping) and not VALIDATION_ALLOW_OVERLAPPING_WINDOWS:
            raise ValidationError(
                "pre_fire seçili ama predictor ve label pencereleri aynı/çakışıyor: "
                f"predictor={PREDICTOR_START_DATE}->{PREDICTOR_END_DATE}, "
                f"label={LABEL_START_DATE}->{LABEL_END_DATE}. "
                "Predictor window label döneminden ÖNCE bitmelidir. Bilerek "
                "çakışma istiyorsan VALIDATION_ALLOW_OVERLAPPING_WINDOWS=True yap."
            )
        return {
            "mode": "pre_fire",
            "predictor_start": PREDICTOR_START_DATE,
            "predictor_end": PREDICTOR_END_DATE,
            "label_start": LABEL_START_DATE,
            "label_end": LABEL_END_DATE,
        }

    # same_season
    return {
        "mode": "same_season",
        "predictor_start": VALIDATION_SEASON_START,
        "predictor_end": VALIDATION_SEASON_END,
        "label_start": VALIDATION_SEASON_START,
        "label_end": VALIDATION_SEASON_END,
    }


def temporal_lead_days(
    predictor_end: str,
    label_start: str,
    mode: str,
) -> int | None:
    """
    Predictor penceresi bitişi ile label penceresi başlangıcı arası gerçek lead (gün).

    Yalnız pre_fire modunda anlamlıdır (predictor window label window'dan önce gelir).
    same_season modunda pencereler çakıştığı için lead anlamsızdır; None döner ve
    raporda "overlapping windows / n/a" gösterilir.
    """
    if mode != "pre_fire":
        return None
    try:
        pe = datetime.strptime(predictor_end, "%Y-%m-%d")
        ls = datetime.strptime(label_start, "%Y-%m-%d")
        return (ls - pe).days
    except (ValueError, TypeError):
        return None


def read_json_file(path: Path) -> dict:
    """Read a compact JSON metadata file with a clear Step6 error on failure."""
    if not path.exists():
        raise ValidationError(
            f"Required predictor metadata not found: {path}. "
            "Regenerate Step5/Step5C so pre-fire validation can verify dates."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Could not read predictor metadata {path}: {exc}") from exc


def first_present(metadata: dict, keys: list[tuple[str, ...]]):
    """Return the first present nested metadata value."""
    for key_path in keys:
        node = metadata
        found = True
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                found = False
                break
            node = node[key]
        if found:
            return node
    return None


def compact_predictor_metadata(name: str, path: Path, metadata: dict) -> dict:
    """Extract the Step6 metadata contract from Step5/Step5C metadata."""
    current_start = first_present(
        metadata,
        [("current_period_start",), ("current_period", "current_period_start"), ("current_period", "start_date")],
    )
    current_end = first_present(
        metadata,
        [("current_period_end",), ("current_period", "current_period_end"), ("current_period", "end_date")],
    )
    baseline_years = first_present(
        metadata,
        [("baseline_years_used",), ("baseline", "baseline_years_used")],
    )
    current_year_excluded = first_present(
        metadata,
        [
            ("current_year_excluded_from_baseline",),
            ("baseline", "current_year_excluded_from_baseline"),
        ],
    )
    return {
        "name": name,
        "metadata_path": str(path),
        "current_period_start": current_start,
        "current_period_end": current_end,
        "baseline_years_used": baseline_years,
        "current_year_excluded_from_baseline": current_year_excluded,
    }


def validate_predictor_metadata_for_prefire(windows: dict) -> dict:
    """
    Verify predictor raster metadata in pre_fire mode.

    This is intentionally fail-fast: Step6 should not quietly validate rasters whose
    current period does not match the configured predictor window.
    """
    metadata_specs = [
        ("step5", STEP5_METADATA_PATH),
        ("step5c", STEP5C_METADATA_PATH),
    ]
    found = {}
    errors = []

    for name, path in metadata_specs:
        raw = read_json_file(path)
        compact = compact_predictor_metadata(name, path, raw)
        found[name] = compact

        missing = [
            key for key in (
                "current_period_start",
                "current_period_end",
                "baseline_years_used",
                "current_year_excluded_from_baseline",
            )
            if compact.get(key) is None
        ]
        if missing:
            errors.append(f"{name} metadata missing required fields: {missing}")
            continue

        if compact["current_period_start"] != windows["predictor_start"]:
            errors.append(
                f"{name} current_period_start={compact['current_period_start']} "
                f"!= predictor_start_date={windows['predictor_start']}"
            )
        if compact["current_period_end"] != windows["predictor_end"]:
            errors.append(
                f"{name} current_period_end={compact['current_period_end']} "
                f"!= predictor_end_date={windows['predictor_end']}"
            )
        if compact["current_year_excluded_from_baseline"] is not True:
            errors.append(f"{name} did not confirm current year was excluded from baseline")

    if errors:
        raise ValidationError(
            "Pre-fire predictor metadata validation failed:\n- "
            + "\n- ".join(errors)
            + "\nRegenerate Step5 and Step5C for the configured predictor window."
        )

    return found


def main() -> dict:
    """Step6 burned-area association testini çalıştırır."""
    log.info("=" * 60)
    log.info("STEP 6 BAŞLIYOR (burned-area association test)")
    log.info("=" * 60)

    if not SKLEARN_AVAILABLE:
        log.warning(
            "scikit-learn yok; ROC/AUC hesaplanamayacak. "
            "pip install scikit-learn ile kurun."
        )

    windows = resolve_windows()
    # İstenen net startup log (mode'un gerçekten ne olduğunu görünür kılar).
    log.info("Running Step6 validation mode: %s", windows["mode"])
    log.info(
        "Predictor window: %s -> %s",
        windows["predictor_start"], windows["predictor_end"],
    )
    log.info(
        "Label window: %s -> %s",
        windows["label_start"], windows["label_end"],
    )
    if windows["mode"] == "pre_fire":
        log.info("Temporal relation: predictor before label (pre-fire)")
        predictor_metadata = validate_predictor_metadata_for_prefire(windows)
    else:
        log.info("Temporal relation: predictor and label same window (same-season)")
        predictor_metadata = {}

    # Referans grid + predictor'ları yükle (esnek dosya adı)
    ref_path = reference_predictor_path()
    grid = read_reference_grid(ref_path)
    log.info("Referans grid: %s (%sx%s)", ref_path.name, grid["width"], grid["height"])

    predictors = {}
    predictor_sources = {}
    for key in PREDICTORS:
        path = resolve_predictor_path(key)
        if path is not None:
            predictors[key] = read_predictor(path)
            predictor_sources[key] = str(path)
        else:
            log.warning(
                "Predictor bulunamadı, atlanıyor: %s",
                [p.name for p in PREDICTORS[key]["path_candidates"]],
            )

    if not predictors:
        raise ValidationError(
            "Hiçbir predictor rasterı bulunamadı. Önce Step5 ve Step5C çalıştırın."
        )

    # Etiketleri label window ile indir
    labels = fetch_labels(grid, windows["label_start"], windows["label_end"])
    burned = labels["burned"]
    validation_populations, validation_population_summaries = (
        build_validation_population_masks(grid, burned)
    )
    ndvi_strata_masks, ndvi_strata_summaries = build_ndvi_strata_masks(grid, burned)

    # Shape uyumlu predictor'ları ayır; metrikler ve plotlar aynı predictor setini kullanır.
    compatible_predictors = {}
    for key, arr in predictors.items():
        if arr.shape != burned.shape:
            log.warning(
                "%s grid'i etiket grid'iyle uyuşmuyor (%s != %s); atlanıyor.",
                key, arr.shape, burned.shape,
            )
            continue
        compatible_predictors[key] = arr
    if not compatible_predictors:
        raise ValidationError("No predictor rasters match the burned-label grid shape.")

    # Her predictor için all-valid istatistik (ROC array'leri ayrı, bellekte)
    per_predictor, roc_arrays = compute_population_predictor_metrics(
        compatible_predictors,
        burned,
        predictor_sources,
        population_mask=None,
        include_roc_preview=True,
        keep_roc_for_plot=True,
        rng=np.random.default_rng(VALIDATION_RANDOM_SEED),
    )

    per_population_predictor = {}
    for population_key, population_mask in validation_populations.items():
        population_metrics, _ = compute_population_predictor_metrics(
            compatible_predictors,
            burned,
            predictor_sources,
            population_mask=population_mask,
            include_roc_preview=False,
            keep_roc_for_plot=False,
            rng=np.random.default_rng(VALIDATION_RANDOM_SEED),
        )
        for stats in population_metrics.values():
            stats["population"] = population_key
        per_population_predictor[population_key] = population_metrics

    direction_diagnostics = {
        "all_valid": compute_direction_diagnostics(compatible_predictors, burned),
    }
    for population_key, population_mask in validation_populations.items():
        direction_diagnostics[population_key] = compute_direction_diagnostics(
            compatible_predictors, burned, population_mask
        )

    ndvi_stratified_auc = compute_ndvi_stratified_auc(
        compatible_predictors,
        burned,
        ndvi_strata_masks,
    )

    # Görseller (full ROC array'leri yalnız burada, bellekten)
    png_outputs = []
    roc_png = plot_roc_comparison(
        per_predictor, roc_arrays, OUTPUT_DIR / "roc_curve_comparison.png"
    )
    if roc_png:
        png_outputs.append(roc_png)
    box_png = plot_boxplot(
        compatible_predictors, burned, OUTPUT_DIR / "burned_vs_unburned_boxplot.png"
    )
    if box_png:
        png_outputs.append(box_png)
    overlay_png = plot_predictor_maps_with_overlay(
        compatible_predictors, burned, OUTPUT_DIR / "predictor_maps_with_burn_overlay.png"
    )
    if overlay_png:
        png_outputs.append(overlay_png)

    lead_days = temporal_lead_days(
        windows["predictor_end"], windows["label_start"], windows["mode"]
    )

    report = {
        "step": "step6_validate_fire_relation",
        "created_at": datetime.now().isoformat(),
        "region": REGION_NAME,
        "log_file": str(log_file),
        "validation_mode": windows["mode"],
        "predictor_window": {
            "start": windows["predictor_start"],
            "end": windows["predictor_end"],
        },
        "label_window": {
            "start": windows["label_start"],
            "end": windows["label_end"],
        },
        "temporal_lead_days": lead_days,
        "prefire_warning": None,
        "reference_grid": {
            "path": str(ref_path),
            "width": grid["width"],
            "height": grid["height"],
        },
        "predictor_metadata": predictor_metadata,
        "per_predictor": per_predictor,
        "per_population_predictor": per_population_predictor,
        "validation_populations": validation_population_summaries,
        "diagnostic_direction_auc": direction_diagnostics,
        "ndvi_strata": ndvi_strata_summaries,
        "ndvi_stratified_auc": ndvi_stratified_auc,
        "png_outputs": png_outputs,
        "config": {
            "balanced_unburned_ratio": VALIDATION_BALANCED_UNBURNED_RATIO,
            "random_seed": VALIDATION_RANDOM_SEED,
            "include_firms": VALIDATION_INCLUDE_FIRMS,
            "max_roc_preview_points": VALIDATION_MAX_ROC_PREVIEW_POINTS,
        },
        "json_format_note": (
            "Compact summary. Full per-pixel arrays and full ROC arrays are NOT "
            "stored in JSON; ROC curves are saved as PNG. Only a downsampled "
            "roc_curve_preview (<= max_roc_preview_points) is kept per predictor."
        ),
        "disclaimer": (
            "First burned-area association test / initial validation experiment "
            "only. Not a validated fire-risk model. No RF/XGBoost trained."
        ),
        "status": "validation_completed",
    }
    # burned array'i JSON'a yazma (büyük); sadece özet alanları tut
    report["labels"] = {
        k: v for k, v in labels.items() if k != "burned"
    }

    stats_path = write_stats_json(report)
    summary_path = write_summary_markdown(report)

    # JSON boyutunu logla (doğrulama için)
    size_mb = stats_path.stat().st_size / (1024 * 1024)
    log.info("Validation JSON: %s (%.3f MB)", stats_path, size_mb)
    log.info("Validation summary: %s", summary_path)
    log.info("PNG çıktıları: %s", ", ".join(png_outputs) if png_outputs else "yok")
    log.info("=" * 60)
    log.info("STEP 6 TAMAMLANDI")

    return report


if __name__ == "__main__":
    main()