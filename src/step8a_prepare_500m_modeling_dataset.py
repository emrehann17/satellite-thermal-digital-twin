"""
step8a_prepare_500m_modeling_dataset.py

Native-MCD64A1-grid (500 m) burned-area MODELING DATASET preparation.

WHY (label-resolution honesty):
    MCD64A1 burned-area label is native ~500 m. Previous validation (Step6)
    downloaded/resampled MCD64A1 onto the 30 m predictor grid using
    nearest-neighbor. That duplicates every native 500 m burned cell into
    many (~250-300) 30 m pixels, which makes pixel counts and any downstream
    confidence/p-value style statistics misleading (pseudo-replication).
    Step8A fixes this by aggregating the 30 m predictor rasters up to the
    MCD64A1 native grid, so each row of the output table represents ONE
    native burned-area grid cell instead of one 30 m pixel.

IMPORTANT:
    - Step8A does NOT train any model.
    - Step8A does NOT run RF/XGBoost.
    - Step8A does NOT validate final fire risk.
    - Step8A only prepares a modeling dataset where each row is one native
      ~500 m MCD64A1 grid cell.
    - Step5, Step5C, Step6, Step7B, Step7C, Step7D, Step7E outputs are
      READ-ONLY inputs; none of them are modified.
    - TVDI formula is not touched. FIRMS semantics are not touched.
    - FIRMS is NOT used as target. MCD64A1 remains the primary burned-area
      label.

NATIVE 500 M GRID -- IMPLEMENTATION NOTE:
    No separately-georeferenced native-resolution MCD64A1 raster is stored
    locally by this repository: Step6 downloads MCD64A1 BurnDate directly
    from Earth Engine at VALIDATION_LABEL_EXPORT_SCALE (30 m), which already
    duplicates each native ~500 m pixel into a block of 30 m pixels sharing
    the same value. Step8A reconstructs an approximate native grid by
    grouping the existing 30 m reference grid into square pixel blocks of
    size round(STEP8A_MCD64A1_NATIVE_CELL_SIZE_M / STEP8A_REFERENCE_PIXEL_SIZE_M)
    and collapsing each block's burned-area sub-pixels back to a single
    representative value (majority/mode), instead of treating every 30 m
    sub-pixel as an independent observation. This is an approximation of the
    true MODIS sinusoidal grid (anchored to the Landsat/EPSG:4326 reference
    grid instead of the native MODIS tile grid), but it directly fixes the
    label-duplication/pseudo-replication problem the supervisor identified,
    using only data already present in this repository.

Input (read-only):
    outputs/step5/current_period_median_celsius.tif   (reference 30 m grid)
    outputs/step5/anomaly_zscore.tif
    outputs/step5c/current_tvdi.tif
    outputs/step5c/tvdi_difference.tif
    outputs/step7d/downscaled_lst_celsius.tif                 (optional)
    outputs/step7e/fused_lst_celsius.tif                      (optional)
    outputs/step7e/fused_lst_source_mask.tif                  (optional)
    data/ndvi_current_period/current_ndvi_median.tif
    data/dem/elevation.tif
    data/dem/slope.tif
    data/landcover/landcover_esa_worldcover_v200_aligned_to_landsat.tif (preferred)
    data/landcover/landcover_esa_worldcover_v200.tif                    (fallback)
    MCD64A1 label raster, auto-discovered (see resolve_label_raster()).
    RAW BurnDate is strongly preferred over the binary burned mask:
        outputs/validation/labels/mcd64a1_raw.tif      (preferred: raw BurnDate)
        outputs/step6/mcd64a1_raw.tif                  (preferred: raw BurnDate)
        outputs/**/*mcd64*raw*.tif                      (raw BurnDate, glob)
        outputs/validation/labels/mcd64a1_burned.tif   (LAST RESORT: binary mask)
        outputs/step6/mcd64a1_burned_label.tif         (LAST RESORT: binary mask)
    When only a binary mask is found, burn_date/burn_month are set to NaN and
    monthly lead-time stratification is marked unavailable.

Output:
    outputs/step8a/step8a_500m_modeling_dataset.parquet
    outputs/step8a/step8a_500m_modeling_dataset.csv
    outputs/step8a/step8a_dataset_stats.json
    outputs/step8a/step8a_dataset_summary.md
    outputs/step8a/step8a_500m_grid_burned_label.tif   (diagnostic)
    outputs/step8a/step8a_500m_grid_valid_mask.tif     (diagnostic)
    outputs/step8a/step8a_500m_cell_preview.geojson    (diagnostic)

CLI:
    python src/step8a_prepare_500m_modeling_dataset.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from core.config import (
    LABEL_START_DATE,
    LABEL_END_DATE,
    PREDICTOR_START_DATE,
    PREDICTOR_END_DATE,
    STEP8A_OUTPUT_DIR,
    STEP8A_MIN_30M_VALID_FRACTION,
    STEP8A_BURNABLE_FRACTION_THRESHOLD,
    STEP8A_WRITE_CSV,
    STEP8A_WRITE_PARQUET,
    STEP8A_RANDOM_SEED,
    STEP8A_MCD64A1_NATIVE_CELL_SIZE_M,
    STEP8A_REFERENCE_PIXEL_SIZE_M,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.utils.tiling import make_tile_grid

BASE_DIR = PROJECT_ROOT

log, log_file = setup_logger("step8a")

# -----------------------------------------------------------------------
# ESA WorldCover v200 class codes (standard).
# -----------------------------------------------------------------------
LC_TREE_COVER = 10
LC_SHRUBLAND = 20
LC_GRASSLAND = 30
LC_CROPLAND = 40
LC_BUILTUP = 50
LC_BARE_SPARSE = 60
LC_SNOW_ICE = 70
LC_PERMANENT_WATER = 80
LC_HERBACEOUS_WETLAND = 90
LC_MANGROVES = 95
LC_MOSS_LICHEN = 100

ESA_WORLDCOVER_CLASSES = {
    LC_TREE_COVER: "tree_cover",
    LC_SHRUBLAND: "shrubland",
    LC_GRASSLAND: "grassland",
    LC_CROPLAND: "cropland",
    LC_BUILTUP: "built_up",
    LC_BARE_SPARSE: "bare_sparse_vegetation",
    LC_SNOW_ICE: "snow_ice",
    LC_PERMANENT_WATER: "permanent_water",
    LC_HERBACEOUS_WETLAND: "herbaceous_wetland",
    LC_MANGROVES: "mangroves",
    LC_MOSS_LICHEN: "moss_lichen",
}

# IMPORTANT (supervisor requirement): cropland is explicitly EXCLUDED from
# both primary burnable masks. It is only ever reported as its own fraction.
BURNABLE_TREE_SHRUB_GRASS_CLASSES = (LC_TREE_COVER, LC_SHRUBLAND, LC_GRASSLAND)
BURNABLE_TREE_SHRUB_CLASSES = (LC_TREE_COVER, LC_SHRUBLAND)

LANDCOVER_ALIGNED_RELPATH = "data/landcover/landcover_esa_worldcover_v200_aligned_to_landsat.tif"
LANDCOVER_SOURCE_RELPATH = "data/landcover/landcover_esa_worldcover_v200.tif"

# Continuous predictor registry: (column_prefix, relpath, required)
# "required" candidates raise a clear error if none of them exist.
CONTINUOUS_PREDICTOR_CANDIDATES: dict[str, dict] = {
    "ndvi": {
        "candidates": ["data/ndvi_current_period/current_ndvi_median.tif"],
        "required": True,
    },
    "elevation": {
        "candidates": ["data/dem/elevation.tif"],
        "required": True,
    },
    "slope": {
        "candidates": ["data/dem/slope.tif"],
        "required": True,
    },
    "lst_anomaly": {
        "candidates": ["outputs/step5/anomaly_zscore.tif"],
        "required": False,
    },
    "current_lst": {
        "candidates": ["outputs/step5/current_period_median_celsius.tif"],
        "required": False,
    },
    "current_tvdi": {
        "candidates": ["outputs/step5c/current_tvdi.tif"],
        "required": False,
    },
    "tvdi_difference": {
        "candidates": ["outputs/step5c/tvdi_difference.tif"],
        "required": False,
    },
    "downscaled_lst": {
        "candidates": ["outputs/step7d/downscaled_lst_celsius.tif"],
        "required": False,
    },
    "fused_lst": {
        "candidates": ["outputs/step7e/fused_lst_celsius.tif"],
        "required": False,
    },
}

# "Baseline" (non-thermal) vs "thermal" grouping, used only for documentation
# in the summary/stats output (Step8B decides actual model feature sets).
BASELINE_PREDICTORS = ["ndvi", "elevation", "slope"]
THERMAL_PREDICTORS = [
    "lst_anomaly", "current_lst", "current_tvdi", "tvdi_difference",
    "downscaled_lst", "fused_lst",
]

FUSED_SOURCE_MASK_RELPATH = "outputs/step7e/fused_lst_source_mask.tif"

# fused_lst_source_mask codes, as written by Step7E (0=invalid,1=observed,2=gap-fill)
SOURCE_INVALID = 0
SOURCE_OBSERVED = 1
SOURCE_GAPFILL = 2


class Step8AError(SystemExit):
    """Fail-fast error for Step8A (extends SystemExit like other steps)."""


# Canonical filename for the Step6B gate's cell-level pre-label exclusion
# manifest (see src/step6b_burned_landcover_gate.py
# write_pre_label_exclusion_manifest()). Always lives alongside
# burned_landcover_gate.json in the same validation/labels output dir.
PRE_LABEL_EXCLUSION_MANIFEST_FILENAME = "pre_label_excluded_cells.parquet"


# =============================================================================
# Path resolution
# =============================================================================
def resolve_reference_30m(explicit: str | None) -> Path:
    """Resolves the 30 m reference predictor grid (default: Step5 current-period LST)."""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = BASE_DIR / p
        if not p.exists():
            raise Step8AError(f"Belirtilen referans 30 m grid bulunamadi: {p}")
        return p

    default = BASE_DIR / "outputs" / "step5" / "current_period_median_celsius.tif"
    if default.exists():
        return default
    raise Step8AError(
        "Referans 30 m grid bulunamadi. Beklenen: "
        "outputs/step5/current_period_median_celsius.tif "
        "(--reference-30m ile farkli bir yol verebilirsiniz)."
    )


# label kind sabitleri
LABEL_KIND_RAW = "raw_burndate"
LABEL_KIND_BINARY = "binary_fallback_no_months"


def _looks_like_raw(name: str) -> bool:
    return "raw" in name.lower()


def _looks_like_binary(name: str) -> bool:
    n = name.lower()
    return ("burned" in n) or ("binary" in n) or ("mask" in n)


def resolve_label_raster(explicit: str | None) -> tuple[Path, str]:
    """
    Discovers the MCD64A1 label raster exported by Step6.

    CRITICAL: Step8A needs the RAW MCD64A1 BurnDate raster (day-of-year
    values), NOT the binary burned mask. The binary mask stores 1 for
    "burned", and if that 1 is misread as BurnDate it maps to DOY 1
    (January 1), which is outside the Aug-Oct label window and floods the
    output with bogus "burned, out-of-window" cells. So raw BurnDate is
    strongly preferred; the binary mask is only a last-resort fallback and,
    when used, burn_date/burn_month are set to NaN and monthly lead-time
    stratification is marked unavailable.

    Returns (path, label_kind) where label_kind is one of:
        LABEL_KIND_RAW    -> raw BurnDate raster (day-of-year values)
        LABEL_KIND_BINARY -> binary burned mask fallback (no burn months)

    Search order (raw preferred at every level):
        1. Explicit --label-raster path (kind inferred from filename;
           a name that does not look binary is treated as raw).
        2. outputs/validation/labels/mcd64a1_raw.tif
        3. outputs/step6/mcd64a1_raw.tif
        4. outputs/validation/labels/*mcd64*raw*.tif
        5. outputs/step6/*mcd64*raw*.tif
        6. Any outputs/**/*mcd64*raw*.tif (glob)
        7. LAST RESORT binary mask fallback:
           outputs/validation/labels/mcd64a1_burned.tif,
           outputs/step6/mcd64a1_burned_label.tif, or any
           outputs/**/*mcd64*.tif that looks binary.

    Fails clearly if nothing is found.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = BASE_DIR / p
        if not p.exists():
            raise Step8AError(f"Belirtilen MCD64A1 etiket rasteri bulunamadi: {p}")
        kind = LABEL_KIND_BINARY if _looks_like_binary(p.name) and not _looks_like_raw(p.name) else LABEL_KIND_RAW
        log.info("MCD64A1 etiket rasteri (explicit): %s (kind=%s)", p, kind)
        if kind == LABEL_KIND_BINARY:
            log.warning(
                "Verilen etiket dosyasi ADI binary burned mask'e benziyor "
                "(%s); burn_date/burn_month NaN olacak. Raw BurnDate rasteri "
                "vermek istiyorsaniz dosya adinda 'raw' gecmeli.", p.name,
            )
        return p, kind

    val_labels = BASE_DIR / "outputs" / "validation" / "labels"
    step6 = BASE_DIR / "outputs" / "step6"
    outputs_dir = BASE_DIR / "outputs"

    # --- 1) Raw BurnDate: sabit isimler ---
    raw_fixed = [
        val_labels / "mcd64a1_raw.tif",
        step6 / "mcd64a1_raw.tif",
    ]
    for p in raw_fixed:
        if p.exists():
            log.info("Raw MCD64A1 BurnDate rasteri bulundu: %s", p)
            return p, LABEL_KIND_RAW

    # --- 2) Raw BurnDate: glob (klasor-oncelikli) ---
    for directory in (val_labels, step6):
        if directory.exists():
            raw_glob = sorted(directory.glob("*mcd64*raw*.tif"))
            raw_glob = [p for p in raw_glob if "(" not in p.name]
            if raw_glob:
                chosen = raw_glob[0]
                log.info("Raw MCD64A1 BurnDate rasteri (glob) bulundu: %s", chosen)
                return chosen, LABEL_KIND_RAW

    # --- 3) Raw BurnDate: genel arama ---
    if outputs_dir.exists():
        raw_any = sorted(
            p for p in outputs_dir.rglob("*mcd64*.tif")
            if _looks_like_raw(p.name) and "(" not in p.name
        )
        if raw_any:
            chosen = raw_any[0]
            log.warning(
                "Raw MCD64A1 BurnDate rasteri sabit konumlarda bulunamadi; "
                "genel aramayla bulundu: %s (alternatifler: %s)",
                chosen, [str(f) for f in raw_any[1:]],
            )
            return chosen, LABEL_KIND_RAW

    # --- 4) LAST RESORT: binary burned mask fallback ---
    binary_fixed = [
        val_labels / "mcd64a1_burned.tif",
        step6 / "mcd64a1_burned_label.tif",
        step6 / "mcd64a1_burned.tif",
    ]
    for p in binary_fixed:
        if p.exists():
            log.warning(
                "Raw MCD64A1 BurnDate rasteri bulunamadi; SON CARE olarak "
                "binary burned mask kullaniliyor: %s. burn_date/burn_month "
                "NaN olacak ve aylik lead-time stratifikasyonu YAPILAMAZ. "
                "Dogru sonuc icin Step6'nin raw BurnDate rasterini "
                "(mcd64a1_raw.tif) uretin.", p,
            )
            return p, LABEL_KIND_BINARY

    if outputs_dir.exists():
        binary_any = sorted(
            p for p in outputs_dir.rglob("*mcd64*.tif")
            if _looks_like_binary(p.name) and "(" not in p.name
        )
        if binary_any:
            chosen = binary_any[0]
            log.warning(
                "Raw MCD64A1 BurnDate rasteri bulunamadi; SON CARE olarak "
                "genel aramayla bulunan binary burned mask kullaniliyor: %s. "
                "burn_date/burn_month NaN olacak.", chosen,
            )
            return chosen, LABEL_KIND_BINARY

    raise Step8AError(
        "MCD64A1 etiket rasteri hicbir yerde bulunamadi. Once TERCIHEN raw "
        "BurnDate rasterini uretin: outputs/validation/labels/mcd64a1_raw.tif "
        "veya outputs/step6/mcd64a1_raw.tif. (Binary burned mask yalnizca son "
        "care fallback'tir ve aylik lead-time uretemez.) --label-raster ile "
        "acik yol da verebilirsiniz."
    )


def resolve_continuous_predictors(ctx: dict | None = None) -> tuple[dict[str, Path], list[str]]:
    """
    Resolves paths for the continuous predictor registry.

    ctx: None ise (varsayilan) legacy Kozan kesfi (CONTINUOUS_PREDICTOR_CANDIDATES,
        BASE_DIR relative). Verilirse (Kozan-disi, or. manavgat_2021): TUM
        predictor'lar YALNIZCA o deneyin namespaced Step5/Step5C/Step7
        dizinlerinden + DEM'den (ctx["dem_input_dir"], artik namespaced --
        bkz. scripts/prepare_dem_for_experiment.py) cozulur. Kozan'in legacy
        paylasilan yollarina (outputs/step5, outputs/step5c, outputs/step7d,
        outputs/step7e, data/dem) ASLA dusulmez.

    Returns (resolved_paths, missing_optional_names). Raises immediately if a
    required predictor has none of its candidate paths present.
    """
    resolved: dict[str, Path] = {}
    missing_optional: list[str] = []
    missing_required: list[str] = []

    if ctx is not None:
        candidates_by_name = {
            "ndvi": [ctx["ndvi_current_dir"] / "current_ndvi_median.tif"],
            "elevation": [ctx["dem_input_dir"] / "elevation.tif"],
            "slope": [ctx["dem_input_dir"] / "slope.tif"],
            "lst_anomaly": [ctx["step5_output_dir"] / "anomaly_zscore.tif"],
            "current_lst": [ctx["step5_output_dir"] / "current_period_median_celsius.tif"],
            "current_tvdi": [ctx["step5c_output_dir"] / "current_tvdi.tif"],
            "tvdi_difference": [ctx["step5c_output_dir"] / "tvdi_difference.tif"],
            "downscaled_lst": [ctx["step7d_output_dir"] / "downscaled_lst_celsius.tif"],
            "fused_lst": [ctx["step7e_output_dir"] / "fused_lst_celsius.tif"],
        }
        for name, info in CONTINUOUS_PREDICTOR_CANDIDATES.items():
            found = None
            for p in candidates_by_name.get(name, []):
                if p.exists():
                    found = p
                    break
            if found is not None:
                resolved[name] = found
            elif info["required"]:
                missing_required.append(name)
            else:
                missing_optional.append(name)
    else:
        for name, info in CONTINUOUS_PREDICTOR_CANDIDATES.items():
            found = None
            for rel in info["candidates"]:
                p = BASE_DIR / rel
                if p.exists():
                    found = p
                    break
            if found is not None:
                resolved[name] = found
            elif info["required"]:
                missing_required.append(name)
            else:
                missing_optional.append(name)

    if missing_required:
        raise Step8AError(
            "Zorunlu predictor raster(lar)i bulunamadi: "
            f"{missing_required}. Beklenen yollar: "
            + ", ".join(
                f"{n}={CONTINUOUS_PREDICTOR_CANDIDATES[n]['candidates']}"
                for n in missing_required
            )
        )
    if missing_optional:
        log.warning(
            "Opsiyonel/tamamlayici predictor raster(lar)i bulunamadi (NaN "
            "olarak birakilacak, satirlar bu yuzden DUSURULMEYECEK): %s",
            missing_optional,
        )
    return resolved, missing_optional


def _grid_matches(path: Path, ref_width: int, ref_height: int, ref_crs, ref_transform) -> bool:
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
    Aligns the categorical ESA WorldCover landcover raster to the reference
    30 m grid using nearest-neighbor resampling (same approach as Step7D).

    This is the ONLY raster Step8A is allowed to resample; every other
    predictor must already match the reference grid or the run fails
    clearly (no silent resampling elsewhere).
    """
    with rasterio.open(reference_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    if aligned_path.exists():
        if _grid_matches(aligned_path, ref_width, ref_height, ref_crs, ref_transform):
            log.info("Onceden hizalanmis landcover yeniden kullaniliyor: %s", aligned_path)
            return {"created": False, "reused": True, "path": aligned_path}
        log.warning(
            "Mevcut hizalanmis landcover (%s) referans gridle uyusmuyor; "
            "yeniden olusturulacak.", aligned_path,
        )

    if source_path is None or not source_path.exists():
        raise Step8AError(
            "Hizalanmis landcover olusturulamiyor: kaynak dosya bulunamadi "
            f"({LANDCOVER_SOURCE_RELPATH}) ve mevcut hizalanmis dosya da yok/uyusmuyor."
        )

    log.info("Landcover nearest-neighbor ile referans 30 m gride hizalaniyor.")
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
            "tiled": bool(ref_width >= 256 and ref_height >= 256),
        }

    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(aligned_path, "w", **out_profile) as dst_ds:
        dst_ds.write(dst, 1)
    log.info("Hizalanmis landcover yazildi: %s", aligned_path)

    if not _grid_matches(aligned_path, ref_width, ref_height, ref_crs, ref_transform):
        raise Step8AError(
            f"Hizalanmis landcover ({aligned_path}) olusturulduktan sonra "
            "bile referans gridle eslesmiyor. Islem durduruldu."
        )
    return {"created": True, "reused": False, "path": aligned_path}


def resolve_landcover(reference_path: Path, explicit: str | None = None) -> tuple[Path, dict]:
    """Resolves the landcover raster, aligning the fallback source if needed.

    explicit: verilirse (Kozan-disi deneyler icin, or.
        ctx["landcover_aligned_path"] -- Step6A gate-input asamasinda zaten
        referans gride hizalanmis) bu dosya KULLANILIR; grid'i referansla
        eslesmiyorsa net bir hata verilir (sessizce yeniden hizalanmaz).
        Kozan'in legacy data/landcover/ kesfine bu durumda HIC BAKILMAZ.
    """
    with rasterio.open(reference_path) as ref:
        ref_w, ref_h, ref_crs, ref_t = ref.width, ref.height, ref.crs, ref.transform

    if explicit:
        explicit_path = Path(explicit)
        if not explicit_path.exists():
            raise Step8AError(f"Belirtilen landcover rasteri bulunamadi: {explicit_path}")
        if not _grid_matches(explicit_path, ref_w, ref_h, ref_crs, ref_t):
            raise Step8AError(
                f"Belirtilen landcover rasteri ({explicit_path}) referans "
                "gridle eslesmiyor. Bu deney icin landcover zaten referans "
                "gride hizali olmalidir (bkz. Step6A gate-input hazirlama); "
                "sessizce yeniden hizalanmaz."
            )
        log.info("Hizalanmis landcover (explicit) kullaniliyor: %s", explicit_path)
        return explicit_path, {
            "original_landcover_path": None,
            "aligned_landcover_path": str(explicit_path),
            "landcover_alignment_method": "step6a_gate_input_reuse",
        }

    aligned = BASE_DIR / LANDCOVER_ALIGNED_RELPATH
    source = BASE_DIR / LANDCOVER_SOURCE_RELPATH

    with rasterio.open(reference_path) as ref:
        ref_w, ref_h, ref_crs, ref_t = ref.width, ref.height, ref.crs, ref.transform

    if aligned.exists() and _grid_matches(aligned, ref_w, ref_h, ref_crs, ref_t):
        log.info("Hizalanmis landcover kullaniliyor: %s", aligned)
        return aligned, {
            "original_landcover_path": str(source) if source.exists() else None,
            "aligned_landcover_path": str(aligned),
            "landcover_alignment_method": "preexisting_aligned_file",
        }

    if not source.exists() and not aligned.exists():
        raise Step8AError(
            "Landcover raster bulunamadi. Beklenen: "
            f"{LANDCOVER_ALIGNED_RELPATH} veya {LANDCOVER_SOURCE_RELPATH}."
        )

    result = prepare_aligned_landcover(reference_path, source if source.exists() else None, aligned)
    return result["path"], {
        "original_landcover_path": str(source) if source.exists() else None,
        "aligned_landcover_path": str(result["path"]),
        "landcover_alignment_method": "nearest_neighbor_to_reference_grid",
        "landcover_alignment_created": result["created"],
    }


def validate_grid_alignment(reference_path: Path, other_paths: dict[str, Path]) -> dict:
    """
    Confirms every non-landcover predictor raster matches the reference grid
    exactly. Step8A does NOT silently resample these; alignment mismatches
    fail clearly (landcover is the sole documented exception, handled
    separately in resolve_landcover()).
    """
    with rasterio.open(reference_path) as ref:
        ref_profile = {
            "width": ref.width, "height": ref.height,
            "crs": ref.crs, "transform": ref.transform,
        }

    mismatches = []
    for name, path in other_paths.items():
        with rasterio.open(path) as src:
            if (
                src.width != ref_profile["width"]
                or src.height != ref_profile["height"]
                or src.crs != ref_profile["crs"]
                or src.transform != ref_profile["transform"]
            ):
                mismatches.append(f"{name} ({path})")
    if mismatches:
        raise Step8AError(
            "Asagidaki raster(lar) referans 30 m grid ile eslesmiyor "
            "(Step8A sessizce resample ETMEZ, landcover haric): "
            + ", ".join(mismatches)
        )
    log.info("Tum zorunlu/opsiyonel predictor rasterlari referans gridle hizali dogrulandi.")
    return ref_profile


def label_window_doy_bounds(label_start: str, label_end: str) -> tuple[int, int, int]:
    """
    Returns (start_doy, end_doy, year) for the label window. Assumes the window
    lies within a single calendar year (true for the current 2023-08-01 ->
    2023-10-31 setup). For 2023 this yields (213, 304, 2023).
    """
    start_dt = datetime.strptime(label_start, "%Y-%m-%d")
    end_dt = datetime.strptime(label_end, "%Y-%m-%d")
    start_doy = int(start_dt.timetuple().tm_yday)
    end_doy = int(end_dt.timetuple().tm_yday)
    return start_doy, end_doy, start_dt.year


def inspect_label_raster(
    label_path: Path,
    label_kind: str,
    label_start: str,
    label_end: str,
) -> dict:
    """
    Inspects the selected MCD64A1 label raster BEFORE aggregation and returns a
    diagnostics dict (saved to stats JSON under 'label_raster_diagnostics').

    For a raster claimed to be raw BurnDate, this fails fast if the pixel
    values do not actually look like day-of-year BurnDate values -- e.g. a
    binary {0,1} mask mislabelled as raw. Catching this here prevents Step8A
    from silently emitting a zero-burned dataset that is useless for Step8B.
    """
    start_doy, end_doy, _year = label_window_doy_bounds(label_start, label_end)

    with rasterio.open(label_path) as src:
        arr = src.read(1, masked=True)
        nodata = src.nodata
        dtype = src.dtypes[0]

    masked_count = int(np.ma.count_masked(arr))
    finite = arr.compressed().astype("float64")
    finite = finite[np.isfinite(finite)]
    finite_count = int(finite.size)

    diag: dict = {
        "path": str(label_path),
        "label_kind": label_kind,
        "dtype": str(dtype),
        "nodata": None if nodata is None else float(nodata),
        "finite_pixel_count": finite_count,
        "masked_or_nodata_pixel_count": masked_count,
        "label_window_doy_range": [start_doy, end_doy],
    }

    if finite_count == 0:
        diag.update({
            "min": None, "max": None, "count_zero": 0, "count_one": 0,
            "count_gt_one": 0, "count_in_label_doy_range": 0,
            "unique_values": [], "unique_counts": [],
        })
        raise Step8AError(
            "Secilen MCD64A1 etiket rasterinde hic gecerli (finite) piksel yok: "
            f"{label_path}. Etiket rasterini kontrol edin."
        )

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    count_zero = int(np.sum(finite == 0))
    count_one = int(np.sum(finite == 1))
    count_gt_one = int(np.sum(finite > 1))
    count_in_range = int(np.sum((finite >= start_doy) & (finite <= end_doy)))
    count_positive = int(np.sum(finite > 0))

    # unique values only when the set is small (avoids huge dumps on real DOY rasters)
    uniq_vals, uniq_counts = np.unique(finite, return_counts=True)
    if uniq_vals.size <= 30:
        diag["unique_values"] = [float(v) for v in uniq_vals]
        diag["unique_counts"] = [int(c) for c in uniq_counts]
    else:
        diag["unique_values"] = "too_many_to_list(>30)"
        diag["unique_counts"] = int(uniq_vals.size)

    diag.update({
        "min": vmin,
        "max": vmax,
        "count_zero": count_zero,
        "count_one": count_one,
        "count_gt_one": count_gt_one,
        "count_positive": count_positive,
        "count_in_label_doy_range": count_in_range,
    })

    # ---- Fail-fast: does a "raw" raster actually contain BurnDate DOY values? ----
    if label_kind == LABEL_KIND_RAW:
        positive = finite[finite > 0]
        only_zero_one = bool(np.all(np.isin(uniq_vals, (0.0, 1.0))))
        mostly_zero_one = False
        if count_positive > 0:
            # "mostly {0,1}": essentially all positive values are exactly 1
            mostly_zero_one = (count_one / count_positive) >= 0.999
        all_positive_are_one = bool(positive.size > 0 and np.all(positive == 1.0))
        no_doy_in_range = (count_in_range == 0)

        looks_binary = only_zero_one or mostly_zero_one or all_positive_are_one
        diag["looks_binary"] = bool(looks_binary)
        diag["no_values_in_label_doy_range"] = bool(no_doy_in_range)

        if looks_binary or no_doy_in_range:
            diag["failed_validation"] = True
            raise Step8AError(
                "Selected MCD64A1 raw raster does not contain BurnDate DOY "
                "values. It appears to be binary. Re-export raw MCD64A1 "
                "BurnDate.\n"
                f"  path: {label_path}\n"
                f"  min={vmin}, max={vmax}, count_zero={count_zero}, "
                f"count_one={count_one}, count_gt_one={count_gt_one}, "
                f"count_in_DOY[{start_doy}-{end_doy}]={count_in_range}\n"
                "  Gerekli: MODIS/061/MCD64A1 'BurnDate' bandini DOY degerleriyle "
                "(BurnDate.gt(0) DEGIL) export edin. Yardimci script: "
                "scripts/export_mcd64a1_raw_burndate.py"
            )
        diag["failed_validation"] = False

    return diag


def align_label_to_reference(label_path: Path, ref_profile: dict, out_dir: Path) -> Path:
    """
    Ensures the MCD64A1 BurnDate raster matches the reference 30 m grid.

    If it already matches, it is used directly. If not, it is explicitly
    (never silently) reprojected with nearest-neighbor -- BurnDate is a
    categorical/day-index value and must not be interpolated -- and the
    resulting file is written under the Step8A output directory for
    traceability.
    """
    if _grid_matches(label_path, ref_profile["width"], ref_profile["height"],
                      ref_profile["crs"], ref_profile["transform"]):
        log.info("MCD64A1 etiket rasteri zaten referans gridle hizali: %s", label_path)
        return label_path

    log.warning(
        "MCD64A1 etiket rasteri (%s) referans gridle hizali degil; "
        "nearest-neighbor ile ACIKCA hizalaniyor (kategorik BurnDate "
        "degeri, interpolasyon YAPILMAZ).", label_path,
    )
    with rasterio.open(label_path) as src:
        src_nodata = src.nodata
        dst = np.full(
            (ref_profile["height"], ref_profile["width"]), np.nan, dtype="float32"
        )
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    out_path = out_dir / "step8a_mcd64a1_burndate_aligned_30m.tif"
    profile = {
        "driver": "GTiff", "width": ref_profile["width"], "height": ref_profile["height"],
        "count": 1, "dtype": "float32", "crs": ref_profile["crs"],
        "transform": ref_profile["transform"], "nodata": np.nan, "compress": "deflate",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst_ds:
        dst_ds.write(dst, 1)
    log.info("Hizalanmis MCD64A1 BurnDate yazildi: %s", out_path)
    return out_path


# =============================================================================
# Day-of-year -> calendar month (robust datetime conversion, no hardcoding)
# =============================================================================
def doy_to_month_and_date(doy: float, label_start: str, label_end: str) -> tuple[int | None, str | None]:
    """
    Converts an MCD64A1 BurnDate day-of-year value to (month, iso_date).

    Uses the label window's year(s) via a real datetime conversion (handles a
    label window spanning a year boundary by trying every year the window
    touches and keeping the result that actually falls inside the window).
    Returns (None, None) if the value cannot be mapped inside the window.
    """
    if not np.isfinite(doy) or doy <= 0:
        return None, None
    doy_int = int(round(doy))

    start_dt = datetime.strptime(label_start, "%Y-%m-%d")
    end_dt = datetime.strptime(label_end, "%Y-%m-%d")
    for year in range(start_dt.year, end_dt.year + 1):
        try:
            candidate = datetime(year, 1, 1) + timedelta(days=doy_int - 1)
        except (OverflowError, ValueError):
            continue
        if start_dt <= candidate <= end_dt:
            return candidate.month, candidate.strftime("%Y-%m-%d")
    return None, None


# =============================================================================
# Day-of-year -> position relative to the label window (leakage-safe gating)
# =============================================================================
def classify_burndate_relative_to_label(
    doy: float, label_start: str, label_end: str
) -> str:
    """
    Classifies an MCD64A1 BurnDate day-of-year relative to the label window.

    Returns one of:
        "in_window"  -> maps to a calendar date inside [label_start, label_end]
                        (a genuine label-window burn).
        "pre_label"  -> maps to a calendar date strictly BEFORE label_start
                        (an EARLIER burn; must be EXCLUDED for leakage safety,
                        never treated as an unburned negative).
        "post_label" -> maps to a calendar date strictly AFTER label_end
                        (a later burn; out-of-window but NOT a pre-label leak).
        "unmapped"   -> value is not finite/positive, or cannot be mapped
                        (treated as "no burn" -> unburned, same as before).

    This is a PURE function (datetime only). It does NOT change any existing
    behaviour: build_dataset() and Step6B keep using doy_to_month_and_date()
    for the burned/unburned decision. This helper is used ONLY where a
    leakage-safe pre-label exclusion is explicitly requested
    (experiment flag exclude_pre_label_burns=True, currently Muğla 2021).

    Single-year label windows (e.g. Muğla 2021-07-29 -> 2021-09-15) are the
    common case; a window spanning a year boundary is handled by trying every
    year the window touches for the in-window test, then comparing against the
    label_start year (pre) / label_end year (post) calendars.
    """
    if not np.isfinite(doy) or doy <= 0:
        return "unmapped"
    doy_int = int(round(doy))

    start_dt = datetime.strptime(label_start, "%Y-%m-%d")
    end_dt = datetime.strptime(label_end, "%Y-%m-%d")

    # 1) In-window? (reuse the exact same mapping logic as doy_to_month_and_date)
    for year in range(start_dt.year, end_dt.year + 1):
        try:
            candidate = datetime(year, 1, 1) + timedelta(days=doy_int - 1)
        except (OverflowError, ValueError):
            continue
        if start_dt <= candidate <= end_dt:
            return "in_window"

    # 2) Pre-label? (date in the label_start year is strictly before label_start)
    try:
        cand_start_year = datetime(start_dt.year, 1, 1) + timedelta(days=doy_int - 1)
        if cand_start_year < start_dt:
            return "pre_label"
    except (OverflowError, ValueError):
        pass

    # 3) Post-label? (date in the label_end year is strictly after label_end)
    try:
        cand_end_year = datetime(end_dt.year, 1, 1) + timedelta(days=doy_int - 1)
        if cand_end_year > end_dt:
            return "post_label"
    except (OverflowError, ValueError):
        pass

    return "unmapped"


# =============================================================================
# Native 500 m block grid
# =============================================================================
def compute_block_size_pixels() -> int:
    ratio = STEP8A_MCD64A1_NATIVE_CELL_SIZE_M / STEP8A_REFERENCE_PIXEL_SIZE_M
    block = int(round(ratio))
    return max(block, 1)


def compute_cell_identity(row_off: int, col_off: int, block_size: int) -> tuple[str, int, int]:
    """
    Canonical ~500 m reconstructed-cell identity from a tile's pixel-space
    origin (row_off, col_off) and the native block size (pixels).

    THE single source of truth for cell_id/row_500m/col_500m across the
    project: build_dataset() below AND Step6B's pre-label exclusion gate
    (src/step6b_burned_landcover_gate.py) both derive cell identity through
    this exact function -- never reimplemented independently -- so a given
    cell_id always refers to the identical physical 500 m block in both
    places (required for the Step6B exclusion manifest to join correctly
    onto Step8A's dataset).
    """
    row_500m = row_off // block_size
    col_500m = col_off // block_size
    cell_id = f"r{row_500m}_c{col_500m}"
    return cell_id, int(row_500m), int(col_500m)


def mode_and_agreement(values: np.ndarray) -> tuple[float, int, int]:
    """
    Returns (mode_value, mode_count, valid_count) for a 1-D array of finite
    values. Ties are broken deterministically in favor of the SMALLER value
    (numpy's ascending unique order + argmax keeps the first max), which
    means unburned (0) wins over burned on an exact tie -- a conservative
    choice that avoids inflating the burned-cell count from ambiguous blocks.
    """
    if values.size == 0:
        return np.nan, 0, 0
    uniq, counts = np.unique(values, return_counts=True)
    best_idx = int(np.argmax(counts))
    return float(uniq[best_idx]), int(counts[best_idx]), int(values.size)


def continuous_stats(values: np.ndarray, total_pixels: int) -> dict:
    valid = values[np.isfinite(values)]
    n = valid.size
    return {
        "mean": float(np.mean(valid)) if n else np.nan,
        "median": float(np.median(valid)) if n else np.nan,
        "std": float(np.std(valid)) if n > 1 else (0.0 if n == 1 else np.nan),
        "valid_count": int(n),
        "valid_fraction": float(n / total_pixels) if total_pixels else 0.0,
    }


def read_pre_label_exclusion_manifest(manifest_path: Path) -> frozenset[str]:
    """
    Reads the canonical Step6B gate pre-label exclusion manifest (parquet;
    see src/step6b_burned_landcover_gate.py write_pre_label_exclusion_manifest())
    and returns the set of excluded cell_id values.

    FAILS FAST (Step8AError) if the manifest is missing or internally
    inconsistent (no cell_id column, null cell_id, duplicate cell_id) --
    Step8A must never silently proceed with an empty/partial exclusion set
    when exclude_pre_label_burns=True is configured for the experiment.
    """
    if not manifest_path.exists():
        raise Step8AError(
            "Pre-label exclusion is enabled but the canonical gate exclusion "
            "manifest is missing.\n"
            "Re-run the label gate before Step8A.\n"
            f"Expected manifest path: {manifest_path}"
        )
    try:
        manifest_df = pd.read_parquet(manifest_path)
    except Exception as exc:  # noqa: BLE001
        raise Step8AError(
            f"Pre-label exclusion manifest ({manifest_path}) okunamadi: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if "cell_id" not in manifest_df.columns:
        raise Step8AError(
            f"Pre-label exclusion manifest ({manifest_path}) icinde 'cell_id' "
            "kolonu yok."
        )
    if manifest_df["cell_id"].isna().any():
        raise Step8AError(
            f"Pre-label exclusion manifest ({manifest_path}) null cell_id "
            "degeri iceriyor."
        )
    if not manifest_df["cell_id"].is_unique:
        dupes = manifest_df.loc[manifest_df["cell_id"].duplicated(), "cell_id"].tolist()
        raise Step8AError(
            f"Pre-label exclusion manifest ({manifest_path}) tekrarlanan "
            f"cell_id degerleri iceriyor: {dupes}."
        )
    return frozenset(manifest_df["cell_id"].astype(str))


def build_dataset(
    reference_path: Path,
    label_path: Path,
    label_kind: str,
    predictor_paths: dict[str, Path],
    landcover_path: Path,
    source_mask_path: Path | None,
    output_dir: Path,
    min_valid_fraction: float,
    burnable_threshold: float,
    label_start: str = LABEL_START_DATE,
    label_end: str = LABEL_END_DATE,
    pre_label_excluded_cell_ids: frozenset[str] | None = None,
) -> dict:
    block_size = compute_block_size_pixels()
    log.info(
        "Native ~500 m grid: referans 30 m piksel = %.1f m, "
        "block_size_pixels = %d (=> yaklasik hucre boyutu %.1f m).",
        STEP8A_REFERENCE_PIXEL_SIZE_M, block_size,
        block_size * STEP8A_REFERENCE_PIXEL_SIZE_M,
    )

    with rasterio.open(reference_path) as ref:
        ref_w, ref_h = ref.width, ref.height
        ref_transform = ref.transform
        ref_crs = ref.crs

    tile_grid = make_tile_grid({"width": ref_w, "height": ref_h}, tile_size_pixels=block_size)
    tiles = tile_grid["tiles"]
    log.info(
        "Toplam 500 m hucre sayisi (yaklasik): %d (%d satir x %d sutun, 30 m grid %dx%d)",
        len(tiles), tile_grid["n_tile_rows"], tile_grid["n_tile_cols"], ref_w, ref_h,
    )

    label_src = rasterio.open(label_path)
    predictor_srcs = {name: rasterio.open(p) for name, p in predictor_paths.items()}
    landcover_src = rasterio.open(landcover_path)
    landcover_nodata = landcover_src.nodata if landcover_src.nodata is not None else 0
    source_mask_src = rasterio.open(source_mask_path) if source_mask_path is not None else None

    rows: list[dict] = []
    feature_valid_counts = {name: 0 for name in CONTINUOUS_PREDICTOR_CANDIDATES}
    feature_missing_counts = {name: 0 for name in CONTINUOUS_PREDICTOR_CANDIDATES}
    burn_month_counts = {8: 0, 9: 0, 10: 0}
    landcover_class_counts_dominant: dict[str, int] = {}
    burned_count = 0
    unburned_count = 0
    valid_modeling_cells = 0
    invalid_cells = 0
    burnable_tsg_count = 0
    burnable_ts_count = 0
    burned_within_tsg = 0
    burned_within_ts = 0
    valid_30m_fraction_values: list[float] = []
    observed_fraction_values: list[float] = []
    gapfilled_fraction_values: list[float] = []
    warnings_list: list[str] = []
    out_of_window_burndate_cells = 0
    burn_month_available = (label_kind == LABEL_KIND_RAW)
    if not burn_month_available:
        warnings_list.append(
            "Binary burned mask (label_kind=binary_fallback_no_months) "
            "kullanildi; burn_date/burn_month NaN, aylik lead-time "
            "stratifikasyonu YAPILAMAZ."
        )

    # 500 m-resolution diagnostic raster buffers.
    n_rows_500m = tile_grid["n_tile_rows"]
    n_cols_500m = tile_grid["n_tile_cols"]
    burned_label_grid = np.full((n_rows_500m, n_cols_500m), np.nan, dtype="float32")
    valid_mask_grid = np.zeros((n_rows_500m, n_cols_500m), dtype="uint8")

    for tile in tiles:
        col_off, row_off, w, h = tile["write_window"]
        window = Window(col_off, row_off, w, h)
        total_pixels = w * h
        cell_id, row_500m, col_500m = compute_cell_identity(row_off, col_off, block_size)

        # --- Label ---
        label_arr = label_src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
        valid_label = label_arr[np.isfinite(label_arr)]
        record: dict = {
            "cell_id": cell_id,
            "row_500m": int(row_500m),
            "col_500m": int(col_500m),
        }

        cx, cy = rasterio.transform.xy(
            ref_transform, row_off + h / 2.0, col_off + w / 2.0, offset="center"
        )
        record["lon"] = float(cx)
        record["lat"] = float(cy)

        # --- Pre-label exclusion eligibility (Step6B gate manifest join) ---
        # A cell the gate excluded (burned BEFORE label_start; see
        # read_pre_label_exclusion_manifest()) is never eligible for the
        # analysis universe, regardless of predictor validity. This does NOT
        # touch record["burned"] below -- the raw label is preserved for
        # audit; only eligibility/valid_for_modeling are affected.
        pre_label_burn_excluded = bool(
            pre_label_excluded_cell_ids is not None and cell_id in pre_label_excluded_cell_ids
        )
        analysis_eligible = not pre_label_burn_excluded
        record["pre_label_burn_excluded"] = pre_label_burn_excluded
        record["analysis_eligible"] = analysis_eligible

        invalid_reasons: list[str] = []

        # --- Label ---
        # A cell is UNBURNED (burned=0) by default -- including when the whole
        # block is nodata/masked or all-zero. A cell is BURNED (burned=1) only
        # if it has positive BurnDate DOY value(s) that map inside the label
        # window. The label NEVER makes a cell invalid-for-modeling: unburned
        # cells are the negative class and must stay in the dataset.
        positive_doy = valid_label[valid_label > 0]

        if not burn_month_available:
            # Binary burned mask fallback: value 1 means "burned" but carries
            # NO day-of-year info. burn_date/burn_month stay NaN.
            burned_flag = 1 if positive_doy.size > 0 else 0
            agreement = (
                float(np.sum(valid_label == 1) / valid_label.size)
                if valid_label.size else np.nan
            )
            record.update({
                "burned": burned_flag,
                "burn_date": np.nan,
                "burn_month": np.nan,
                "burn_day_of_year": np.nan,
                "label_source": "MCD64A1_binary_fallback",
                "burn_date_pixel_agreement_fraction": agreement,
                "out_of_window_burndate": False,
            })
        else:
            # Raw BurnDate (day-of-year). Determine a representative positive
            # DOY via mode among POSITIVE values only (so a burned block is not
            # masked by its majority-zero background pixels). Zero/NaN-only or
            # all-nodata blocks are simply unburned.
            month = None
            out_of_window = False
            burned_flag = 0
            burn_date_out = 0.0
            burn_doy_out = 0.0
            burn_month_out = 0
            agreement = np.nan

            if positive_doy.size > 0:
                rep_doy, rep_count, _n = mode_and_agreement(positive_doy)
                agreement = float(rep_count / positive_doy.size)
                month, _iso = doy_to_month_and_date(
                    rep_doy, label_start, label_end
                )
                if month is not None:
                    burned_flag = 1
                    burn_date_out = float(rep_doy)
                    burn_doy_out = float(rep_doy)
                    burn_month_out = month
                    burn_month_counts[month] = burn_month_counts.get(month, 0) + 1
                else:
                    # positive BurnDate but outside label window -> unburned
                    out_of_window = True
                    out_of_window_burndate_cells += 1
                    if len([w for w in warnings_list if "label penceresi" in w]) < 20:
                        warnings_list.append(
                            f"{cell_id}: BurnDate={rep_doy} label penceresi "
                            f"({label_start} -> {label_end}) disinda; "
                            "bu label penceresi icin UNBURNED sayildi."
                        )

            record.update({
                "burned": burned_flag,
                "burn_date": burn_date_out if burned_flag else 0.0,
                "burn_month": burn_month_out if burned_flag else 0,
                "burn_day_of_year": burn_doy_out if burned_flag else 0.0,
                "label_source": "MCD64A1",
                "burn_date_pixel_agreement_fraction": agreement,
                "out_of_window_burndate": bool(out_of_window),
            })

        if record["burned"] == 1:
            burned_count += 1
        else:
            unburned_count += 1
        burned_label_grid[row_500m, col_500m] = float(record["burned"])

        # --- Continuous predictors ---
        required_valid_mask = None
        for name in CONTINUOUS_PREDICTOR_CANDIDATES:
            src = predictor_srcs.get(name)
            if src is None:
                # Raster entirely absent (missing optional input): keep the
                # column present with NaN/0 rather than omitting it, so the
                # dataset schema stays stable for Step8B.
                record[f"{name}_mean"] = np.nan
                record[f"{name}_median"] = np.nan
                record[f"{name}_std"] = np.nan
                record[f"{name}_valid_count"] = 0
                record[f"{name}_valid_fraction"] = 0.0
                feature_missing_counts[name] += 1
                if name in ("ndvi", "elevation", "slope"):
                    finite = np.zeros((h, w), dtype=bool)
                    required_valid_mask = finite if required_valid_mask is None else (required_valid_mask & finite)
                continue
            arr = src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
            stats = continuous_stats(arr, total_pixels)
            for stat_name, val in stats.items():
                record[f"{name}_{stat_name}"] = val
            if stats["valid_count"] > 0:
                feature_valid_counts[name] += 1
            else:
                feature_missing_counts[name] += 1
            if name in ("ndvi", "elevation", "slope"):
                finite = np.isfinite(arr)
                required_valid_mask = finite if required_valid_mask is None else (required_valid_mask & finite)

        if required_valid_mask is None:
            required_valid_mask = np.zeros((h, w), dtype=bool)

        # --- Landcover ---
        lc_arr = landcover_src.read(1, window=window, masked=True)
        lc_filled = lc_arr.filled(landcover_nodata).astype("float64")
        lc_valid = ~np.ma.getmaskarray(lc_arr)
        lc_valid = lc_valid & np.isfinite(lc_filled) & (lc_filled != landcover_nodata)
        lc_valid_count = int(lc_valid.sum())

        if lc_valid_count > 0:
            lc_vals = lc_filled[lc_valid].astype(int)
            uniq, counts = np.unique(lc_vals, return_counts=True)
            dominant_idx = int(np.argmax(counts))
            dominant_class = int(uniq[dominant_idx])
            dominant_name = ESA_WORLDCOVER_CLASSES.get(dominant_class, f"unknown_{dominant_class}")
            landcover_class_counts_dominant[dominant_name] = (
                landcover_class_counts_dominant.get(dominant_name, 0) + 1
            )

            def _frac(code: int) -> float:
                return float(np.sum(lc_vals == code) / lc_valid_count)

            tree_f = _frac(LC_TREE_COVER)
            shrub_f = _frac(LC_SHRUBLAND)
            grass_f = _frac(LC_GRASSLAND)
            crop_f = _frac(LC_CROPLAND)
            bare_f = _frac(LC_BARE_SPARSE)
            builtup_f = _frac(LC_BUILTUP)
            water_f = _frac(LC_PERMANENT_WATER)

            record["landcover_dominant"] = dominant_class
            record["landcover_tree_cover_fraction"] = tree_f
            record["landcover_shrubland_fraction"] = shrub_f
            record["landcover_grassland_fraction"] = grass_f
            record["landcover_cropland_fraction"] = crop_f
            record["landcover_bare_sparse_vegetation_fraction"] = bare_f
            record["landcover_built_up_fraction"] = builtup_f
            record["landcover_permanent_water_fraction"] = water_f

            burnable_tsg = (tree_f + shrub_f + grass_f) >= burnable_threshold
            burnable_ts = (tree_f + shrub_f) >= burnable_threshold
            record["burnable_tree_shrub_grass"] = bool(burnable_tsg)
            record["burnable_tree_shrub"] = bool(burnable_ts)
            if burnable_tsg:
                burnable_tsg_count += 1
                if record.get("burned") == 1:
                    burned_within_tsg += 1
            if burnable_ts:
                burnable_ts_count += 1
                if record.get("burned") == 1:
                    burned_within_ts += 1
        else:
            record["landcover_dominant"] = np.nan
            record["landcover_tree_cover_fraction"] = np.nan
            record["landcover_shrubland_fraction"] = np.nan
            record["landcover_grassland_fraction"] = np.nan
            record["landcover_cropland_fraction"] = np.nan
            record["landcover_bare_sparse_vegetation_fraction"] = np.nan
            record["landcover_built_up_fraction"] = np.nan
            record["landcover_permanent_water_fraction"] = np.nan
            record["burnable_tree_shrub_grass"] = False
            record["burnable_tree_shrub"] = False
            invalid_reasons.append("no_valid_landcover_pixels")

        # --- Coverage / provenance ---
        valid_30m_pixel_count = int(required_valid_mask.sum())
        valid_30m_fraction = float(valid_30m_pixel_count / total_pixels) if total_pixels else 0.0
        record["valid_30m_pixel_count"] = valid_30m_pixel_count
        record["total_30m_pixel_count"] = int(total_pixels)
        record["valid_30m_fraction"] = valid_30m_fraction
        valid_30m_fraction_values.append(valid_30m_fraction)

        if source_mask_src is not None:
            sm_arr = source_mask_src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
            sm_valid = np.isfinite(sm_arr)
            n_sm_valid = int(sm_valid.sum())
            observed_frac = float(np.sum(sm_arr == SOURCE_OBSERVED) / total_pixels) if total_pixels else 0.0
            gapfilled_frac = float(np.sum(sm_arr == SOURCE_GAPFILL) / total_pixels) if total_pixels else 0.0
            invalid_frac = float(
                (total_pixels - np.sum(sm_arr == SOURCE_OBSERVED) - np.sum(sm_arr == SOURCE_GAPFILL))
                / total_pixels
            ) if total_pixels else 1.0
            record["observed_fraction"] = observed_frac
            record["gapfilled_fraction"] = gapfilled_frac
            record["invalid_source_fraction"] = invalid_frac
            if n_sm_valid > 0:
                sm_uniq, sm_counts = np.unique(sm_arr[sm_valid].astype(int), return_counts=True)
                record["source_mask_majority"] = int(sm_uniq[int(np.argmax(sm_counts))])
            else:
                record["source_mask_majority"] = np.nan
            observed_fraction_values.append(observed_frac)
            gapfilled_fraction_values.append(gapfilled_frac)
        else:
            record["observed_fraction"] = np.nan
            record["gapfilled_fraction"] = np.nan
            record["invalid_source_fraction"] = np.nan
            record["source_mask_majority"] = np.nan

        thermal_means = [
            record.get(f"{name}_mean") for name in THERMAL_PREDICTORS if f"{name}_mean" in record
        ]
        record["thermal_any_missing"] = bool(
            any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in thermal_means)
        )

        # --- Validity for modeling ---
        # IMPORTANT: modeling validity depends ONLY on predictor validity,
        # landcover, and baseline feature availability -- NEVER on the label.
        # Unburned cells (burned=0), all-nodata-label blocks, and out-of-window
        # cells must all remain in the dataset as the negative class. Requiring
        # a positive/finite BurnDate here is exactly what collapsed the dataset
        # to only burned-like cells before.
        predictors_ok = valid_30m_fraction >= min_valid_fraction
        ndvi_ok = np.isfinite(record.get("ndvi_mean", np.nan))
        elev_ok = np.isfinite(record.get("elevation_mean", np.nan))
        slope_ok = np.isfinite(record.get("slope_mean", np.nan))
        lc_ok = lc_valid_count > 0

        if not predictors_ok:
            invalid_reasons.append("insufficient_valid_30m_predictor_fraction")
        if not ndvi_ok:
            invalid_reasons.append("ndvi_mean_not_finite")
        if not elev_ok:
            invalid_reasons.append("elevation_mean_not_finite")
        if not slope_ok:
            invalid_reasons.append("slope_mean_not_finite")

        predictor_valid = bool(predictors_ok and ndvi_ok and elev_ok and slope_ok and lc_ok)

        # valid_for_modeling = analysis_eligible AND predictor_valid. A
        # pre-label-excluded cell is NEVER valid_for_modeling, regardless of
        # predictor validity -- but its predictor QA reasons (above) are
        # still preserved in invalid_reason; pre_label_burn_excluded is only
        # PREPENDED as the primary reason, never replacing them.
        valid_for_modeling = bool(analysis_eligible and predictor_valid)
        record["valid_for_modeling"] = valid_for_modeling
        if pre_label_burn_excluded:
            invalid_reasons = ["pre_label_burn_excluded"] + invalid_reasons
        record["invalid_reason"] = ";".join(invalid_reasons) if invalid_reasons else None

        if valid_for_modeling:
            valid_modeling_cells += 1
            valid_mask_grid[row_500m, col_500m] = 1
        else:
            invalid_cells += 1

        rows.append(record)

    label_src.close()
    for src in predictor_srcs.values():
        src.close()
    landcover_src.close()
    if source_mask_src is not None:
        source_mask_src.close()

    df = pd.DataFrame(rows)

    # --- Pre-label eligibility breakdown (raw / eligible / final modeling) ---
    # pre_label_burn_excluded/analysis_eligible/valid_for_modeling are always
    # stamped per-row above, so these columns exist whenever df is non-empty.
    # raw_* reuses the loop-tallied burned_count/unburned_count directly (by
    # construction identical to a df-level count -- these are NEVER affected
    # by eligibility/pre-label exclusion, see "burned" column note above).
    if len(df):
        eligible_mask = df["analysis_eligible"] == True  # noqa: E712
        final_mask = df["valid_for_modeling"] == True  # noqa: E712
        eligible_burned_count = int((df.loc[eligible_mask, "burned"] == 1).sum())
        eligible_unburned_count = int((df.loc[eligible_mask, "burned"] == 0).sum())
        final_burned_count = int((df.loc[final_mask, "burned"] == 1).sum())
        final_unburned_count = int((df.loc[final_mask, "burned"] == 0).sum())
        pre_label_burn_excluded_count = int((df["pre_label_burn_excluded"] == True).sum())  # noqa: E712
        analysis_eligible_count = int(eligible_mask.sum())
        predictor_invalid_count_among_eligible = int(eligible_mask.sum() - final_mask.sum())
    else:
        eligible_burned_count = eligible_unburned_count = 0
        final_burned_count = final_unburned_count = 0
        pre_label_burn_excluded_count = analysis_eligible_count = 0
        predictor_invalid_count_among_eligible = 0

    # 500 m diagnostic transform (reference transform scaled by block_size).
    grid_500m_transform = ref_transform * rasterio.Affine.scale(block_size)

    burnable_landcover_diagnostics = compute_burnable_landcover_diagnostics(df, burn_month_available)

    return {
        "dataframe": df,
        "block_size_pixels": block_size,
        "n_rows_500m": n_rows_500m,
        "n_cols_500m": n_cols_500m,
        "grid_500m_transform": grid_500m_transform,
        "ref_crs": ref_crs,
        "burned_label_grid": burned_label_grid,
        "valid_mask_grid": valid_mask_grid,
        "label_kind": label_kind,
        "burn_month_available": burn_month_available,
        "burnable_landcover_diagnostics": burnable_landcover_diagnostics,
        "counters": {
            "total_500m_cells": len(rows),
            "valid_modeling_cells": valid_modeling_cells,
            "invalid_cells": invalid_cells,
            "burned_cell_count": burned_count,
            "unburned_cell_count": unburned_count,
            "burn_month_counts": burn_month_counts,
            "burn_month_available": burn_month_available,
            "label_kind": label_kind,
            "out_of_window_burndate_cells": out_of_window_burndate_cells,
            "burnable_tree_shrub_grass_count": burnable_tsg_count,
            "burnable_tree_shrub_count": burnable_ts_count,
            "burned_count_within_burnable_tree_shrub_grass": burned_within_tsg,
            "burned_count_within_burnable_tree_shrub": burned_within_ts,
            "feature_valid_counts": feature_valid_counts,
            "feature_missing_counts": feature_missing_counts,
            "landcover_class_counts_dominant": landcover_class_counts_dominant,
            "valid_30m_fraction_summary": _series_summary(valid_30m_fraction_values),
            "observed_fraction_summary": _series_summary(observed_fraction_values),
            "gapfilled_fraction_summary": _series_summary(gapfilled_fraction_values),
            # --- Pre-label exclusion eligibility (always present; all-zero /
            # eligible==total when exclude_pre_label_burns is not used) ---
            "pre_label_burn_excluded_count": pre_label_burn_excluded_count,
            "analysis_eligible_count": analysis_eligible_count,
            "predictor_invalid_count_among_eligible": predictor_invalid_count_among_eligible,
            "raw_label_counts_before_eligibility": {
                "burned": burned_count, "unburned": unburned_count,
            },
            "eligible_label_counts_after_pre_label_exclusion": {
                "burned": eligible_burned_count, "unburned": eligible_unburned_count,
            },
            "final_modeling_counts_after_predictor_validity": {
                "burned": final_burned_count, "unburned": final_unburned_count,
            },
        },
        "warnings": warnings_list,
    }


def _series_summary(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


# =============================================================================
# Burnable / landcover diagnostics (Step8B readiness checks)
# =============================================================================
# These are pure diagnostics computed AFTER the modeling dataset is built.
# They do NOT change burned/unburned labels, do NOT change any predictor
# value, and do NOT train or filter anything -- they only summarize how the
# burned/unburned classes are distributed across landcover so a modeler can
# see, before Step8B, whether the "burnable" strata actually contain enough
# burned positives.
PRIMARY_BURNABLE_MASK_COLUMN = "burnable_tree_shrub_grass"
MIN_BURNED_IN_PRIMARY_BURNABLE_MASK = 30

BURNED_LANDCOVER_FRACTION_COLUMNS = {
    "tree_cover_fraction": "landcover_tree_cover_fraction",
    "shrubland_fraction": "landcover_shrubland_fraction",
    "grassland_fraction": "landcover_grassland_fraction",
    "cropland_fraction": "landcover_cropland_fraction",
    "bare_sparse_vegetation_fraction": "landcover_bare_sparse_vegetation_fraction",
    "built_up_fraction": "landcover_built_up_fraction",
    "permanent_water_fraction": "landcover_permanent_water_fraction",
}


def _dominant_name(code) -> str:
    if code is None or (isinstance(code, float) and not np.isfinite(code)):
        return "unknown_nan"
    return ESA_WORLDCOVER_CLASSES.get(int(code), f"unknown_{int(code)}")


def _dominant_counts(sub: pd.DataFrame) -> dict:
    counts: dict[str, int] = {}
    for code in sub["landcover_dominant"]:
        name = _dominant_name(code)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _month_counts(sub: pd.DataFrame, burn_month_available: bool):
    if not burn_month_available:
        return "unavailable_binary_fallback"
    counts = {8: 0, 9: 0, 10: 0}
    for m in sub["burn_month"]:
        if m in (8, 9, 10):
            counts[int(m)] = counts.get(int(m), 0) + 1
    return counts


def compute_burnable_landcover_diagnostics(df: pd.DataFrame, burn_month_available: bool) -> dict:
    """
    Computes burnable-mask / landcover diagnostics restricted to the actual
    modeling population (`valid_for_modeling == True`), so the numbers reflect
    exactly what Step8B would see.

    Returns a dict with:
        burned_landcover_dominant_counts
        unburned_landcover_dominant_counts
        burned_landcover_fraction_summary
        burned_count_by_burnable_mask_and_month
        burned_count_by_dominant_landcover_and_month
        cropland_burned_count
        nonburnable_burned_count
        primary_burnable_mask
        burned_count_within_primary_burnable_mask
        diagnostics_population
    """
    if len(df) == 0:
        empty_summary = {k: _series_summary([]) for k in BURNED_LANDCOVER_FRACTION_COLUMNS}
        return {
            "burned_landcover_dominant_counts": {},
            "unburned_landcover_dominant_counts": {},
            "burned_landcover_fraction_summary": empty_summary,
            "burned_count_by_burnable_mask_and_month": {
                "burnable_tree_shrub_grass": {8: 0, 9: 0, 10: 0},
                "burnable_tree_shrub": {8: 0, 9: 0, 10: 0},
                "all_valid": {8: 0, 9: 0, 10: 0},
            },
            "burned_count_by_dominant_landcover_and_month": {},
            "cropland_burned_count": 0,
            "nonburnable_burned_count": 0,
            "primary_burnable_mask": PRIMARY_BURNABLE_MASK_COLUMN,
            "burned_count_within_primary_burnable_mask": 0,
            "diagnostics_population": "valid_for_modeling == True",
        }

    valid = df[df["valid_for_modeling"] == True]  # noqa: E712
    burned = valid[valid["burned"] == 1]
    unburned = valid[valid["burned"] == 0]

    # 1 & 2: dominant landcover counts, split by burned/unburned.
    burned_landcover_dominant_counts = _dominant_counts(burned)
    unburned_landcover_dominant_counts = _dominant_counts(unburned)

    # 3: fraction summaries (mean/median/min/max/n) over BURNED cells only.
    burned_landcover_fraction_summary = {}
    for out_name, col in BURNED_LANDCOVER_FRACTION_COLUMNS.items():
        vals = burned[col].dropna().tolist() if col in burned.columns else []
        burned_landcover_fraction_summary[out_name] = _series_summary(vals)

    # 4: burned count by burnable mask, split by burn_month.
    burned_count_by_burnable_mask_and_month = {
        "burnable_tree_shrub_grass": _month_counts(
            burned[burned["burnable_tree_shrub_grass"] == True], burn_month_available  # noqa: E712
        ),
        "burnable_tree_shrub": _month_counts(
            burned[burned["burnable_tree_shrub"] == True], burn_month_available  # noqa: E712
        ),
        "all_valid": _month_counts(burned, burn_month_available),
    }

    # 5: burned count by dominant landcover class, split by burn_month.
    burned_count_by_dominant_landcover_and_month: dict = {}
    if len(burned) > 0:
        names = burned["landcover_dominant"].map(_dominant_name)
        for name in sorted(names.unique()):
            grp = burned[names == name]
            burned_count_by_dominant_landcover_and_month[name] = _month_counts(grp, burn_month_available)

    # 6 & 7: cropland-dominant burned cells, and burned cells outside the
    # primary (tree+shrub+grass) burnable mask entirely.
    cropland_burned_count = int((burned["landcover_dominant"] == LC_CROPLAND).sum())
    nonburnable_burned_count = int((~burned["burnable_tree_shrub_grass"].astype(bool)).sum()) if len(burned) else 0

    burned_count_within_primary_burnable_mask = int(burned[PRIMARY_BURNABLE_MASK_COLUMN].astype(bool).sum()) if len(burned) else 0

    return {
        "burned_landcover_dominant_counts": burned_landcover_dominant_counts,
        "unburned_landcover_dominant_counts": unburned_landcover_dominant_counts,
        "burned_landcover_fraction_summary": burned_landcover_fraction_summary,
        "burned_count_by_burnable_mask_and_month": burned_count_by_burnable_mask_and_month,
        "burned_count_by_dominant_landcover_and_month": burned_count_by_dominant_landcover_and_month,
        "cropland_burned_count": cropland_burned_count,
        "nonburnable_burned_count": nonburnable_burned_count,
        "primary_burnable_mask": PRIMARY_BURNABLE_MASK_COLUMN,
        "burned_count_within_primary_burnable_mask": burned_count_within_primary_burnable_mask,
        "diagnostics_population": "valid_for_modeling == True",
    }


# =============================================================================
# Diagnostic outputs
# =============================================================================
def write_diagnostic_rasters(result: dict, output_dir: Path) -> dict:
    profile_common = {
        "driver": "GTiff",
        "width": result["n_cols_500m"],
        "height": result["n_rows_500m"],
        "count": 1,
        "crs": result["ref_crs"],
        "transform": result["grid_500m_transform"],
        "compress": "deflate",
    }

    label_path = output_dir / "step8a_500m_grid_burned_label.tif"
    label_profile = dict(profile_common, dtype="float32", nodata=np.nan)
    with rasterio.open(label_path, "w", **label_profile) as dst:
        dst.write(result["burned_label_grid"], 1)

    mask_path = output_dir / "step8a_500m_grid_valid_mask.tif"
    mask_profile = dict(profile_common, dtype="uint8", nodata=0)
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(result["valid_mask_grid"], 1)

    return {"label_raster": label_path, "valid_mask_raster": mask_path}


def write_cell_preview_geojson(df: pd.DataFrame, result: dict, output_dir: Path, max_features: int = 5000) -> Path:
    """Writes a lightweight GeoJSON preview of 500 m cell footprints (subset only)."""
    block_size = result["block_size_pixels"]
    transform = result["grid_500m_transform"]
    subset = df.head(max_features)

    features = []
    for _, r in subset.iterrows():
        row = int(r["row_500m"])
        col = int(r["col_500m"])
        x0, y0 = transform * (col, row)
        x1, y1 = transform * (col + 1, row + 1)
        coords = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {
                "cell_id": r["cell_id"],
                "burned": None if pd.isna(r["burned"]) else int(r["burned"]),
                "valid_for_modeling": bool(r["valid_for_modeling"]),
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    path = output_dir / "step8a_500m_cell_preview.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")
    return path


# =============================================================================
# Stats / summary writers
# =============================================================================
def write_stats(
    output_dir: Path,
    result: dict,
    reference_path: Path,
    label_path: Path,
    predictor_paths: dict[str, Path],
    landcover_path: Path,
    landcover_info: dict,
    source_mask_path: Path | None,
    min_valid_fraction: float,
    burnable_threshold: float,
    warnings_list: list[str],
    exclude_pre_label_burns: bool = False,
    pre_label_exclusion_manifest_path: str | None = None,
) -> Path:
    burnable_diag = result.get("burnable_landcover_diagnostics", {}) or {}
    with rasterio.open(reference_path) as ref:
        ref_info = {
            "path": str(reference_path), "width": ref.width, "height": ref.height,
            "crs": str(ref.crs), "transform": list(ref.transform)[:6],
        }
    with rasterio.open(label_path) as lbl:
        label_info = {
            "path": str(label_path), "width": lbl.width, "height": lbl.height,
            "crs": str(lbl.crs), "transform": list(lbl.transform)[:6],
        }

    counters = result["counters"]
    burned_rate = (
        counters["burned_cell_count"] / counters["valid_modeling_cells"]
        if counters["valid_modeling_cells"] else None
    )

    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": "step8a_prepare_500m_modeling_dataset",
        "predictor_start_date": None,
        "predictor_end_date": None,
        "label_start_date": LABEL_START_DATE,
        "label_end_date": LABEL_END_DATE,
        "reference_30m_grid": ref_info,
        "reference_500m_label_source": label_info,
        "label_kind": result.get("label_kind"),
        "label_source_description": result.get("label_source_description"),
        "label_raster_diagnostics": result.get("label_raster_diagnostics"),
        "burn_month_available": result.get("burn_month_available"),
        "block_size_pixels": result["block_size_pixels"],
        "min_30m_valid_fraction_threshold": min_valid_fraction,
        "burnable_fraction_threshold": burnable_threshold,
        "total_500m_cells": counters["total_500m_cells"],
        "valid_modeling_cells": counters["valid_modeling_cells"],
        "invalid_cells": counters["invalid_cells"],
        "burned_cell_count": counters["burned_cell_count"],
        "unburned_cell_count": counters["unburned_cell_count"],
        "burned_rate": burned_rate,
        "burn_month_counts": counters["burn_month_counts"],
        "out_of_window_burndate_cells": counters["out_of_window_burndate_cells"],
        "burnable_tree_shrub_grass_count": counters["burnable_tree_shrub_grass_count"],
        "burnable_tree_shrub_count": counters["burnable_tree_shrub_count"],
        "burned_count_within_each_burnable_mask": {
            "burnable_tree_shrub_grass": counters["burned_count_within_burnable_tree_shrub_grass"],
            "burnable_tree_shrub": counters["burned_count_within_burnable_tree_shrub"],
        },
        # --- Detailed burnable/landcover diagnostics (Step8B readiness) ---
        # All restricted to valid_for_modeling == True (see
        # burnable_landcover_diagnostics.diagnostics_population). Does not
        # change any label/predictor value; diagnostics only.
        "burned_landcover_dominant_counts": burnable_diag.get("burned_landcover_dominant_counts"),
        "unburned_landcover_dominant_counts": burnable_diag.get("unburned_landcover_dominant_counts"),
        "burned_landcover_fraction_summary": burnable_diag.get("burned_landcover_fraction_summary"),
        "burned_count_by_burnable_mask_and_month": burnable_diag.get("burned_count_by_burnable_mask_and_month"),
        "burned_count_by_dominant_landcover_and_month": burnable_diag.get("burned_count_by_dominant_landcover_and_month"),
        "cropland_burned_count": burnable_diag.get("cropland_burned_count"),
        "nonburnable_burned_count": burnable_diag.get("nonburnable_burned_count"),
        "primary_burnable_mask": burnable_diag.get("primary_burnable_mask"),
        "burned_count_within_primary_burnable_mask": burnable_diag.get("burned_count_within_primary_burnable_mask"),
        "burnable_diagnostics_population": burnable_diag.get("diagnostics_population"),
        "feature_missing_counts": counters["feature_missing_counts"],
        "feature_valid_counts": counters["feature_valid_counts"],
        "predictor_paths": {k: str(v) for k, v in predictor_paths.items()},
        "landcover_path": str(landcover_path),
        "landcover_info": landcover_info,
        "source_mask_path": str(source_mask_path) if source_mask_path else None,
        "valid_30m_fraction_summary": counters["valid_30m_fraction_summary"],
        "observed_fraction_summary": counters["observed_fraction_summary"],
        "gapfilled_fraction_summary": counters["gapfilled_fraction_summary"],
        "landcover_class_counts_dominant": counters["landcover_class_counts_dominant"],
        "warnings": warnings_list,
        "no_model_trained": True,
        "no_firms_label_used": True,
        "primary_label": "MCD64A1",
        "cropland_excluded_from_primary_burnable_mask": True,
        # --- Pre-label exclusion eligibility (leakage-safe; opt-in per
        # experiment, currently only mugla_2021). Always present with a
        # uniform schema; all-zero/eligible==total when not used. ---
        "exclude_pre_label_burns": bool(exclude_pre_label_burns),
        "pre_label_exclusion": {
            "pre_label_burn_excluded_count": counters.get("pre_label_burn_excluded_count", 0),
            "analysis_eligible_count": counters.get("analysis_eligible_count", counters["total_500m_cells"]),
            "predictor_invalid_count_among_eligible": counters.get(
                "predictor_invalid_count_among_eligible", counters["invalid_cells"]
            ),
            "raw_label_counts_before_eligibility": counters.get("raw_label_counts_before_eligibility"),
            "eligible_label_counts_after_pre_label_exclusion": counters.get(
                "eligible_label_counts_after_pre_label_exclusion"
            ),
            "final_modeling_counts_after_predictor_validity": counters.get(
                "final_modeling_counts_after_predictor_validity"
            ),
            "manifest_path": pre_label_exclusion_manifest_path,
        },
    }
    path = output_dir / "step8a_dataset_stats.json"
    path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return path


def write_summary(
    output_dir: Path,
    result: dict,
    stats_path: Path,
    label_start: str = LABEL_START_DATE,
    label_end: str = LABEL_END_DATE,
) -> Path:
    counters = result["counters"]
    burnable_diag = result.get("burnable_landcover_diagnostics", {}) or {}
    burned_rate = (
        counters["burned_cell_count"] / counters["valid_modeling_cells"] * 100.0
        if counters["valid_modeling_cells"] else float("nan")
    )
    lines = [
        "# Step8A: Native 500 m MCD64A1-Grid Modeling Dataset",
        "",
        "## What this step does",
        "",
        "- Step8A prepares a **500 m MCD64A1-grid** modeling dataset.",
        "- It fixes the previous 30 m label-duplication issue: MCD64A1 is "
        "native ~500 m, but earlier validation resampled it onto the 30 m "
        "predictor grid, duplicating every native burned cell into many "
        "30 m pixels and inflating pixel-based confidence/p-value style "
        "statistics.",
        "- **Each row is one native burned-area grid cell, not one 30 m "
        "pixel.**",
        "- **No model is trained** in Step8A (no RF/XGBoost, no fire-risk "
        "validation).",
        "- **No FIRMS label is used** anywhere in this step.",
        "- **MCD64A1 is the primary burned-area label.**",
        "- **Cropland is excluded from the primary burnable mask** (reported "
        "only as its own fraction).",
        "- Two burnable masks are provided: (1) tree + shrub + grass, "
        "(2) tree + shrub.",
        "",
        "## Label source",
        "",
        f"- Label source kind: `{result.get('label_source_description')}`.",
        (
            "- Label raster diagnostics: "
            f"min=`{(result.get('label_raster_diagnostics') or {}).get('min')}`, "
            f"max=`{(result.get('label_raster_diagnostics') or {}).get('max')}`, "
            f"count_in_DOY_range=`{(result.get('label_raster_diagnostics') or {}).get('count_in_label_doy_range')}`, "
            f"count_zero=`{(result.get('label_raster_diagnostics') or {}).get('count_zero')}`, "
            f"count_one=`{(result.get('label_raster_diagnostics') or {}).get('count_one')}` "
            "(full detail under `label_raster_diagnostics` in stats JSON)."
        ),
        "- If a future run aborts with 'Selected MCD64A1 raw raster does not "
        "contain BurnDate DOY values': **Raw MCD64A1 BurnDate raster is missing "
        "or invalid; re-export BurnDate band from MCD64A1 instead of binary "
        "burned mask.** Use `scripts/export_mcd64a1_raw_burndate.py` (exports "
        "MODIS/061/MCD64A1 `BurnDate` DOY values, not `BurnDate.gt(0)`).",
        (
            "- This dataset was built from the **raw MCD64A1 BurnDate** raster. "
            "Each 500 m cell keeps both burned and unburned cells; a cell is "
            "`burned=1` only if its native BurnDate falls inside the "
            f"{label_start} -> {label_end} label window."
            if result.get("burn_month_available")
            else "- **WARNING:** a **binary burned mask** was used as a "
            "last-resort fallback (raw BurnDate raster not found). `burn_date` "
            "and `burn_month` are therefore **NaN** and **monthly lead-time "
            "stratification is UNAVAILABLE**. Provide the raw MCD64A1 BurnDate "
            "raster (`mcd64a1_raw.tif`) to enable per-month analysis."
        ),
        (
            "- The 'each row is one native burned-area grid cell' framing is "
            "valid here because the native ~500 m grid was reconstructed from "
            "the raw 30 m BurnDate export (mode per block) and includes both "
            "burned and unburned cells."
            if result.get("label_source_description") == "reconstructed_from_30m_raw_burndate"
            else "- The 'each row is one native burned-area grid cell' framing "
            "is valid here because the label is a genuine native raw BurnDate "
            "grid with both burned and unburned cells."
            if result.get("burn_month_available")
            else "- NOTE: because only a binary mask was available, rows still "
            "represent aggregated 500 m cells, but burn timing (month) cannot "
            "be recovered."
        ),
        "",
        "## Native grid reconstruction note",
        "",
        f"- Reference 30 m pixel size assumed: `{STEP8A_REFERENCE_PIXEL_SIZE_M}` m.",
        f"- Target native MCD64A1 cell size: `{STEP8A_MCD64A1_NATIVE_CELL_SIZE_M}` m.",
        f"- Derived block size: `{result['block_size_pixels']}` x "
        f"`{result['block_size_pixels']}` 30 m pixels per 500 m cell.",
        "- No separately-georeferenced native 500 m MCD64A1 raster exists "
        "locally; Step6 exports MCD64A1 BurnDate directly at 30 m from "
        "Earth Engine, which already duplicates each native pixel. Step8A "
        "collapses each 30 m block back to a single representative "
        "(majority/mode) BurnDate value per block, instead of treating "
        "every 30 m sub-pixel as an independent sample.",
        "",
        "## Dataset size",
        "",
        f"- Total 500 m cells: `{counters['total_500m_cells']}`",
        f"- Valid modeling cells: `{counters['valid_modeling_cells']}`",
        f"- Invalid cells: `{counters['invalid_cells']}`",
        f"- Burned cells: `{counters['burned_cell_count']}`",
        f"- Unburned cells: `{counters['unburned_cell_count']}`",
        f"- Burned rate (of valid modeling cells): `{burned_rate:.3f}%`",
        (
            f"- Burn month counts (Aug/Sep/Oct): `{counters['burn_month_counts']}`"
            if result.get("burn_month_available")
            else "- Burn month counts: **unavailable** (binary label fallback)."
        ),
        f"- Out-of-window BurnDate cells (kept as unburned for this window): "
        f"`{counters['out_of_window_burndate_cells']}`",
        "",
        "## Burnable strata",
        "",
        f"- burnable_tree_shrub_grass cells: `{counters['burnable_tree_shrub_grass_count']}` "
        f"(burned within: `{counters['burned_count_within_burnable_tree_shrub_grass']}`)",
        f"- burnable_tree_shrub cells: `{counters['burnable_tree_shrub_count']}` "
        f"(burned within: `{counters['burned_count_within_burnable_tree_shrub']}`)",
        "",
        "## Burnable / landcover diagnostics (Step8B readiness)",
        "",
        f"- Population: `{burnable_diag.get('diagnostics_population')}` "
        "(diagnostics only -- no label, predictor, or filtering change).",
        f"- Primary burnable mask: `{burnable_diag.get('primary_burnable_mask')}`; "
        f"burned cells within it: "
        f"`{burnable_diag.get('burned_count_within_primary_burnable_mask')}` "
        f"(minimum recommended for stable Step8B modeling: "
        f"`{MIN_BURNED_IN_PRIMARY_BURNABLE_MASK}`).",
        f"- Cropland-dominant burned cells: `{burnable_diag.get('cropland_burned_count')}`.",
        f"- Burned cells outside the primary burnable mask entirely "
        f"(`nonburnable_burned_count`): `{burnable_diag.get('nonburnable_burned_count')}`.",
        f"- Burned cells by dominant landcover: "
        f"`{burnable_diag.get('burned_landcover_dominant_counts')}`.",
        f"- Unburned cells by dominant landcover: "
        f"`{burnable_diag.get('unburned_landcover_dominant_counts')}`.",
        f"- Burned count by burnable mask and month: "
        f"`{burnable_diag.get('burned_count_by_burnable_mask_and_month')}`.",
        f"- Burned count by dominant landcover and month: "
        f"`{burnable_diag.get('burned_count_by_dominant_landcover_and_month')}`.",
        "- Burned-cell landcover fraction summary (mean/median/min/max/n), "
        "full detail in stats JSON `burned_landcover_fraction_summary`:",
    ]
    for name, summ in (burnable_diag.get("burned_landcover_fraction_summary") or {}).items():
        lines.append(
            f"  - `{name}`: mean=`{summ.get('mean')}`, median=`{summ.get('median')}`, "
            f"min=`{summ.get('min')}`, max=`{summ.get('max')}`, n=`{summ.get('n')}`"
        )
    if (
        burnable_diag.get("burned_count_within_primary_burnable_mask") is not None
        and burnable_diag["burned_count_within_primary_burnable_mask"] < MIN_BURNED_IN_PRIMARY_BURNABLE_MASK
    ):
        lines.append(
            "- **WARNING: Primary burnable mask has too few burned positives "
            "for stable Step8 modeling.** Review the landcover/burnable-mask "
            "definition and the burned-cells-by-dominant-landcover breakdown "
            "above before proceeding to Step8B."
        )
    lines.extend([
        "",
        "## Next step",
        "",
        "- Step8B will use this dataset to compare a **baseline model** "
        "(elevation + slope + landcover + NDVI) against a **thermal model** "
        "(baseline + current TVDI + LST anomaly + TVDI difference + fused "
        "thermal).",
        "- Lead-time will be evaluated later by `burn_month` strata: "
        "August, September, October.",
        "- Class balancing/undersampling is **not** performed here; that "
        "belongs to Step8B.",
        "",
        f"Full stats: `{stats_path.name}`",
    ])
    if result["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in result["warnings"][:50])
        if len(result["warnings"]) > 50:
            lines.append(f"- ... ({len(result['warnings']) - 50} more, see stats JSON)")

    path = output_dir / "step8a_dataset_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =============================================================================
# Quality checks (fail-fast sanity assertions)
# =============================================================================
LEAKAGE_COLUMN_NAMES = {"anomaly_zscore", "modis_context_zscore"}


# =============================================================================
# Metadata date-consistency guards (fail-fast)
# =============================================================================
# Kozan'in legacy tarihleri. Kozan-disi HICBIR deneyin ciktisinda (ne JSON ne
# markdown) bu dizgiler GECMEMELIDIR. Sabit dizgi olarak yazmak yerine
# core.config'in GERCEK Kozan sabitlerinden turetiyoruz -- boylece Kozan'in
# config'i degisirse bu koruma otomatik olarak onunla birlikte guncellenir.
# (Prompt madde 7 acikca "2023-08-01" ve "2023-10-31"i istiyor; bunlar zaten
# LABEL_START_DATE / LABEL_END_DATE degerleridir.)
_KOZAN_STALE_DATE_STRINGS = tuple(
    d for d in (
        PREDICTOR_START_DATE, PREDICTOR_END_DATE,
        LABEL_START_DATE, LABEL_END_DATE,
    ) if d
)

# Deney bazli, ELLE dogrulanmis beklenen tarihler. Bir deney burada listeliyse,
# ciktisi bu degerlerle BIREBIR eslesmek ZORUNDADIR (prompt madde 6).
_EXPECTED_EXPERIMENT_DATES = {
    "manavgat_2021": {
        "predictor_start_date": "2021-06-01",
        "predictor_end_date": "2021-07-27",
        "label_start_date": "2021-07-28",
        "label_end_date": "2021-08-31",
    },
}


def assert_metadata_dates_consistent(stats_data: dict, ctx: dict | None) -> None:
    """
    Step8A ciktilari YAZILMADAN once calisan sert (hard) tutarlilik kontrolu.

    Kontroller (Kozan-disi deneyler icin):
      1. Top-level predictor_start_date/predictor_end_date/label_start_date/
         label_end_date, ctx'in degerleriyle BIREBIR ayni olmali.
      2. Top-level tarih alanlari, predictor_window / label_window listeleriyle
         BIREBIR ayni olmali (ikisi arasinda sessiz bir sapma olamaz).
      3. Deney _EXPECTED_EXPERIMENT_DATES'te listeliyse (or. manavgat_2021),
         tarihler o elle-dogrulanmis degerlerle BIREBIR ayni olmali.
      4. Hicbir top-level tarih alani Kozan'in legacy tarihlerinden biri olamaz.

    Kozan (ctx=None veya experiment_id == "kozan_2023") icin HICBIR SEY
    YAPMAZ -- legacy davranis BIREBIR korunur.
    """
    if ctx is None or ctx.get("experiment_id") == "kozan_2023":
        return

    experiment_id = ctx["experiment_id"]
    date_fields = (
        "predictor_start_date", "predictor_end_date",
        "label_start_date", "label_end_date",
    )

    # 1. ctx ile birebir eslesme
    mismatched = {
        f: (stats_data.get(f), ctx[f]) for f in date_fields
        if stats_data.get(f) != ctx[f]
    }
    if mismatched:
        raise Step8AError(
            f"METADATA TUTARSIZLIGI ('{experiment_id}'): top-level tarih "
            f"alanlari ctx ile eslesmiyor (alan: (yazilan, beklenen)): "
            f"{mismatched}. Islem DURDURULDU."
        )

    # 2. window listeleriyle birebir eslesme
    predictor_window = stats_data.get("predictor_window")
    label_window = stats_data.get("label_window")
    if predictor_window != [stats_data["predictor_start_date"], stats_data["predictor_end_date"]]:
        raise Step8AError(
            f"METADATA TUTARSIZLIGI ('{experiment_id}'): predictor_window "
            f"({predictor_window}) top-level predictor tarihleriyle "
            f"({stats_data['predictor_start_date']} -> "
            f"{stats_data['predictor_end_date']}) eslesmiyor. Islem DURDURULDU."
        )
    if label_window != [stats_data["label_start_date"], stats_data["label_end_date"]]:
        raise Step8AError(
            f"METADATA TUTARSIZLIGI ('{experiment_id}'): label_window "
            f"({label_window}) top-level label tarihleriyle "
            f"({stats_data['label_start_date']} -> "
            f"{stats_data['label_end_date']}) eslesmiyor. Islem DURDURULDU."
        )

    # 3. elle-dogrulanmis beklenen tarihler (varsa)
    expected = _EXPECTED_EXPERIMENT_DATES.get(experiment_id)
    if expected:
        wrong = {
            f: (stats_data.get(f), v) for f, v in expected.items()
            if stats_data.get(f) != v
        }
        if wrong:
            raise Step8AError(
                f"METADATA TUTARSIZLIGI ('{experiment_id}'): tarihler bu deney "
                f"icin elle-dogrulanmis beklenen degerlerle eslesmiyor "
                f"(alan: (yazilan, beklenen)): {wrong}. Islem DURDURULDU."
            )

    # 4. Kozan'in legacy tarihlerinden hicbiri gecmemeli
    stale = {
        f: stats_data.get(f) for f in date_fields
        if stats_data.get(f) in _KOZAN_STALE_DATE_STRINGS
    }
    if stale:
        raise Step8AError(
            f"METADATA TUTARSIZLIGI ('{experiment_id}'): su top-level tarih "
            f"alanlari hala Kozan'in legacy tarihlerini iceriyor: {stale}. "
            "Islem DURDURULDU."
        )


def assert_no_stale_kozan_dates_in_outputs(paths: list[Path], ctx: dict | None) -> None:
    """
    Kozan-disi bir deneyin YAZILMIS ciktilarini (JSON/markdown/vb.) tarayip
    Kozan'in legacy tarih dizgilerinden (2023-06-01/2023-07-31/2023-08-01/
    2023-10-31) herhangi birini iceriyorsa RuntimeError firlatir
    (prompt madde 7).

    Bu, madde 6'daki alan-bazli kontrolun YAKALAYAMADIGI sizintilari
    (or. serbest-metin markdown cumleleri, warnings[] listesindeki eski
    pencere metinleri) yakalar.

    Kozan icin HICBIR SEY YAPMAZ.
    """
    if ctx is None or ctx.get("experiment_id") == "kozan_2023":
        return

    experiment_id = ctx["experiment_id"]
    offenders: dict[str, list[str]] = {}
    for path in paths:
        if path is None or not Path(path).exists():
            continue
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binari (parquet/tif) dosyalari atla
        found = [s for s in _KOZAN_STALE_DATE_STRINGS if s in text]
        if found:
            offenders[str(path)] = found

    if offenders:
        raise RuntimeError(
            f"STALE KOZAN DATE SIZINTISI ('{experiment_id}'): asagidaki cikti "
            f"dosyalari Kozan'in legacy tarihlerini iceriyor: {offenders}. "
            "Kozan-disi bir deneyin ciktisinda bu tarihler ASLA gecmemelidir. "
            "Islem DURDURULDU."
        )


def _label_window_months(label_start: str, label_end: str) -> set[int]:
    """
    Label penceresinin (inclusive) kapsadigi takvim aylarinin kumesini
    dondurur. Kozan icin (2023-08-01 -> 2023-10-31) {8, 9, 10} -- BIREBIR
    eski davranis. Manavgat gibi farkli bir pencereye sahip deneyler icin
    (or. 2021-07-28 -> 2021-08-31) {7, 8} -- ARTIK DOGRU sekilde
    hesaplaniyor (eskiden HER ZAMAN sabit {8, 9, 10} varsayiliyordu, bkz.
    bug raporu).
    """
    start_dt = datetime.strptime(label_start, "%Y-%m-%d")
    end_dt = datetime.strptime(label_end, "%Y-%m-%d")
    months: set[int] = set()
    year, month = start_dt.year, start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        months.add(month)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def run_quality_checks(
    df: pd.DataFrame,
    result: dict,
    ref_pixel_total: int,
    allow_all_burned: bool,
    label_start: str = LABEL_START_DATE,
    label_end: str = LABEL_END_DATE,
) -> tuple[list[str], list[str]]:
    """
    Runs sanity checks. Returns (fatal_problems, soft_warnings).

    Fatal problems abort the run (unless a specific override applies).
    Burned-area labels must be sparse, so a dataset where every valid cell is
    "burned", or where there are no unburned cells, almost certainly indicates
    a label bug (e.g. a binary mask misread as BurnDate) and is treated as
    fatal.
    """
    fatal: list[str] = []
    warnings: list[str] = []
    counters = result["counters"]
    valid_months = _label_window_months(label_start, label_end)

    valid = int(counters["valid_modeling_cells"])
    burned = int(counters["burned_cell_count"])
    unburned = int(counters["unburned_cell_count"])
    total_cells = int(counters["total_500m_cells"])
    burned_rate = (burned / valid) if valid else 0.0

    burned_rows = df[df["burned"] == 1]

    # --- Point 6: zero burned cells is unusable for a classifier ---
    # (Only meaningful when a raw BurnDate label was used; with a binary
    # fallback the inspection step already fired, and here we still want a
    # non-empty positive class.)
    if burned == 0:
        fatal.append(
            "burned_cell_count == 0: no burned cells found. A classifier/AUC "
            "cannot be trained with zero positives. The label raster almost "
            "certainly does not contain valid in-window BurnDate DOY values "
            "(re-export raw MCD64A1 BurnDate; see label_raster_diagnostics)."
        )

    # --- Point 6: unburned must be > 0 ---
    if unburned == 0 and valid > 0 and not allow_all_burned:
        # (also covered below, but keep an explicit non-override-dependent
        # signal here for clarity)
        pass

    # --- Point 4: fail-fast label-sanity checks ---

    # 4a: all valid cells burned
    if valid > 0 and valid == burned and not allow_all_burned:
        fatal.append(
            f"valid_modeling_cells ({valid}) == burned_cell_count ({burned}): "
            "every valid cell is labelled burned. Burned-area labels should be "
            "sparse; this almost always means the label raster is wrong (e.g. "
            "a binary burned mask read as BurnDate). Use the raw MCD64A1 "
            "BurnDate raster, or pass --allow-all-burned to override."
        )

    # 4b: no unburned cells
    if valid > 0 and unburned == 0:
        msg = (
            "unburned_cell_count == 0: there are no unburned cells. Burned-area "
            "labels should be sparse; check the label raster (raw BurnDate vs "
            "binary mask)."
        )
        if allow_all_burned:
            warnings.append(msg + " (--allow-all-burned set; continuing.)")
        else:
            fatal.append(msg)

    # 4d: all burned burn_date values are 1.0 -> binary mask misread as BurnDate
    if len(burned_rows) > 0 and burned_rows["burn_date"].notna().any():
        bd = burned_rows["burn_date"].dropna()
        if len(bd) > 0 and bool((bd == 1.0).all()):
            fatal.append(
                "Binary burned mask was likely used as BurnDate. Use raw "
                "MCD64A1 BurnDate raster."
            )

    # 4c: burned_rate too high -> warn loudly (fatal only if extreme + not overridden)
    if valid > 0 and burned_rate > 0.5:
        msg = (
            f"burned_rate = {burned_rate:.3f} (> 0.5). Burned-area labels should "
            "be sparse; a majority-burned dataset is highly suspicious (likely "
            "a label-resolution or label-source bug)."
        )
        if allow_all_burned:
            warnings.append(msg + " (--allow-all-burned set; continuing.)")
        else:
            fatal.append(msg + " Pass --allow-all-burned to override if this is "
                         "genuinely expected.")

    # --- Aggregation sanity (soft) ---
    if len(df) >= ref_pixel_total:
        warnings.append(
            f"Row count ({len(df)}) is not smaller than the 30 m pixel count "
            f"({ref_pixel_total}); native-grid aggregation may have failed."
        )
    if burned >= ref_pixel_total:
        warnings.append(
            "Burned cell count is not far smaller than a plausible old 30 m "
            "burned pixel count; aggregation may not have collapsed pixels."
        )

    # --- burn_month must be inside the experiment's own label-window months
    # for burned rows (raw label). Kozan: {8,9,10}; other experiments (e.g.
    # Manavgat: {7,8}) use their OWN label window -- BIREBIR sabit {8,9,10}
    # varsayimi ARTIK YOK (bkz. bug raporu).
    if result.get("burn_month_available"):
        bad_months = burned_rows[~burned_rows["burn_month"].isin(valid_months)]
        bad_months_hard = bad_months[bad_months["burn_month"].notna()]
        if len(bad_months_hard) > 0:
            fatal.append(
                f"{len(bad_months_hard)} burned rows have burn_month outside "
                f"{sorted(valid_months)}; burned cells must map inside the "
                f"experiment's label window ({label_start} -> {label_end})."
            )

    # --- cropland must not by itself satisfy burnable masks (soft) ---
    if df["burnable_tree_shrub_grass"].any():
        crop_leak = df[(df["burnable_tree_shrub_grass"]) & (df["landcover_cropland_fraction"] > 0.99)]
        if len(crop_leak) > 0:
            warnings.append(
                f"{len(crop_leak)} burnable_tree_shrub_grass cells are almost "
                "entirely cropland; check burnable-mask logic."
            )

    # --- Point 8: too few burned positives within the primary burnable mask ---
    # (soft warning -- does not abort Step8A, but flags that Step8B may not be
    # able to fit a stable classifier restricted to the burnable stratum.)
    burnable_diag = result.get("burnable_landcover_diagnostics", {}) or {}
    burned_in_primary = burnable_diag.get("burned_count_within_primary_burnable_mask")
    if burned_in_primary is not None and burned_in_primary < MIN_BURNED_IN_PRIMARY_BURNABLE_MASK:
        warnings.append(
            "Primary burnable mask has too few burned positives for stable "
            f"Step8 modeling. (burned_count_within_primary_burnable_mask="
            f"{burned_in_primary}, threshold={MIN_BURNED_IN_PRIMARY_BURNABLE_MASK})"
        )

    # --- no leakage columns ---
    leaked_cols = LEAKAGE_COLUMN_NAMES.intersection(df.columns)
    if leaked_cols:
        fatal.append(f"Unexpected leakage-style columns present: {leaked_cols}")

    # --- source fractions approx sum to 1.0 (soft) ---
    if "observed_fraction" in df.columns and df["observed_fraction"].notna().any():
        total_frac = (
            df["observed_fraction"].fillna(0)
            + df["gapfilled_fraction"].fillna(0)
            + df["invalid_source_fraction"].fillna(0)
        )
        bad = total_frac[(total_frac - 1.0).abs() > 1e-3]
        if len(bad) > 0:
            warnings.append(
                f"{len(bad)} rows have observed+gapfilled+invalid source "
                "fractions not approximately summing to 1.0."
            )

    # --- Point 6: burn_month total must equal burned_cell_count (raw label) ---
    # Kozan: {8,9,10}; diger deneyler kendi label-penceresi aylarini kullanir
    # (bkz. yukaridaki valid_months / bug raporu).
    if result.get("burn_month_available"):
        bm = counters["burn_month_counts"]
        bm_total = sum(int(bm.get(m, 0)) for m in valid_months)
        if bm_total != burned:
            fatal.append(
                f"burn_month_counts total ({bm_total}) != burned_cell_count "
                f"({burned}); every burned cell must have a burn month in "
                f"{sorted(valid_months)} (experiment label window "
                f"{label_start} -> {label_end})."
            )

    # --- Point 6: valid_modeling_cells should be much larger than burned ---
    if burned > 0 and valid <= burned:
        fatal.append(
            f"valid_modeling_cells ({valid}) is not larger than "
            f"burned_cell_count ({burned}); the dataset must include the "
            "unburned negative class."
        )
    elif burned > 0 and valid < 5 * burned:
        warnings.append(
            f"valid_modeling_cells ({valid}) is only ~{valid / burned:.1f}x "
            f"burned_cell_count ({burned}); burned-area labels are usually much "
            "sparser -- double-check the label raster."
        )

    # --- Point 6: valid cells should not collapse far below total ---
    if total_cells > 0 and valid < 0.5 * total_cells:
        msg = (
            f"valid_modeling_cells ({valid}) < 50% of total_500m_cells "
            f"({total_cells}). Most cells were dropped -- this usually means "
            "predictor/landcover coverage is poor OR the label incorrectly "
            "gated modeling validity."
        )
        if allow_all_burned:
            warnings.append(msg + " (--allow-all-burned set; continuing.)")
        else:
            fatal.append(
                msg + " Investigate before Step8B, or pass --allow-all-burned "
                "if this coverage is genuinely expected."
            )

    return fatal, warnings


# =============================================================================
# Main
# =============================================================================
def main(
    output_dir_arg: str = STEP8A_OUTPUT_DIR,
    force: bool = False,
    write_csv: bool = STEP8A_WRITE_CSV,
    write_parquet: bool = STEP8A_WRITE_PARQUET,
    min_valid_fraction: float = STEP8A_MIN_30M_VALID_FRACTION,
    burnable_threshold: float = STEP8A_BURNABLE_FRACTION_THRESHOLD,
    label_raster_arg: str | None = None,
    reference_30m_arg: str | None = None,
    allow_all_burned: bool = False,
    ctx: dict | None = None,
) -> dict:
    log.info("=" * 60)
    log.info(
        "STEP 8A BASLIYOR (native ~500 m MCD64A1-grid modeling dataset)%s",
        f" [experiment={ctx['experiment_id']}]" if ctx else "",
    )
    log.info("=" * 60)

    out_dir = BASE_DIR / output_dir_arg
    required_outputs = [
        out_dir / "step8a_500m_modeling_dataset.parquet",
        out_dir / "step8a_500m_modeling_dataset.csv",
        out_dir / "step8a_dataset_stats.json",
        out_dir / "step8a_dataset_summary.md",
    ]
    if any(p.exists() for p in required_outputs) and not force:
        present = [p.name for p in required_outputs if p.exists()]
        raise Step8AError(
            "Step8A ciktilari zaten var (" + ", ".join(present)
            + "). Uzerine yazmak icin --force verin."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if ctx is not None and not ctx.get("is_kozan"):
        reference_30m_arg = reference_30m_arg or str(
            ctx["step5_output_dir"] / "current_period_median_celsius.tif"
        )
        label_raster_arg = label_raster_arg or str(ctx["gate_labels_dir"] / "mcd64a1_raw.tif")

    reference_path = resolve_reference_30m(reference_30m_arg)
    label_path_raw, label_kind = resolve_label_raster(label_raster_arg)
    predictor_paths, missing_optional = resolve_continuous_predictors(ctx=ctx)
    landcover_explicit = (
        str(ctx["landcover_aligned_path"])
        if (ctx is not None and not ctx.get("is_kozan") and ctx.get("landcover_aligned_path"))
        else None
    )
    landcover_path, landcover_info = resolve_landcover(reference_path, explicit=landcover_explicit)

    # BUG FIX: BurnDate -> ay/burned-pencere siniflandirmasi (asagida
    # inspect_label_raster + build_dataset icinde) daha once HER ZAMAN
    # modul-seviyesi LABEL_START_DATE/LABEL_END_DATE (Kozan'in 2023 legacy
    # sabitleri) kullaniyordu -- ctx'ten BAGIMSIZ. Bu, deney-farkinda
    # (Kozan-disi) calistirmalarda gercekte deneyin KENDI label penceresi
    # icinde olan BurnDate degerlerinin yanlislikla "pencere disi" sayilip
    # UNBURNED'a dusurulmesine yol aciyordu (bkz. bug raporu). Burada
    # kullanilacak GERCEK label penceresini ctx'ten (Kozan-disi) veya
    # legacy sabitlerden (Kozan, DEGISMEDEN) coziyoruz.
    if ctx is not None and not ctx.get("is_kozan"):
        effective_label_start = ctx["label_start_date"]
        effective_label_end = ctx["label_end_date"]
    else:
        effective_label_start = LABEL_START_DATE
        effective_label_end = LABEL_END_DATE

    with rasterio.open(reference_path) as ref:
        ref_profile = {
            "width": ref.width, "height": ref.height,
            "crs": ref.crs, "transform": ref.transform,
        }
        ref_pixel_total = ref.width * ref.height

    # Validate alignment for every non-landcover predictor (fail clearly, no
    # silent resample). "current_lst" is the reference raster itself so it is
    # trivially aligned; skip re-checking it explicitly.
    to_check = {k: v for k, v in predictor_paths.items() if k != "current_lst"}
    validate_grid_alignment(reference_path, to_check)

    # Determine how the label was represented, for honest reporting:
    #   - binary_fallback_no_months: binary burned mask (no BurnDate/months)
    #   - raw_burndate: raw BurnDate raster already at (approx) native 500 m
    #     resolution (a single cell per native pixel)
    #   - reconstructed_from_30m_raw_burndate: raw BurnDate raster stored at
    #     the 30 m reference grid (Step6's GEE export), which Step8A collapses
    #     back to native ~500 m blocks via mode.
    block_size = compute_block_size_pixels()
    if label_kind == LABEL_KIND_BINARY:
        label_source_description = LABEL_KIND_BINARY
    else:
        with rasterio.open(label_path_raw) as _lbl:
            lbl_px_x = abs(_lbl.transform.a)
            ref_px_x = abs(ref_profile["transform"].a)
        # If the label pixel size is already close to the native cell size
        # (i.e. much coarser than the 30 m reference), treat it as native raw
        # BurnDate; otherwise it is a 30 m export we reconstruct into blocks.
        ratio = (lbl_px_x / ref_px_x) if ref_px_x else 1.0
        if ratio >= block_size * 0.5:
            label_source_description = "raw_burndate"
        else:
            label_source_description = "reconstructed_from_30m_raw_burndate"
    log.info("Label source description: %s", label_source_description)

    # --- Label raster inspection (BEFORE aggregation) ---
    # Inspect the RAW source raster and fail fast if a "raw" raster is actually
    # binary / has no in-window DOY values. Diagnostics are always saved so the
    # user can see exactly what the label contained.
    try:
        label_diag = inspect_label_raster(
            label_path_raw, label_kind, effective_label_start, effective_label_end
        )
    except Step8AError as exc:
        # Persist whatever diagnostics we can before aborting.
        diag_path = out_dir / "step8a_label_raster_diagnostics.json"
        try:
            partial = {"error": str(exc), "label_path": str(label_path_raw)}
            diag_path.write_text(json.dumps(partial, indent=2, default=str), encoding="utf-8")
            log.error("Label raster diagnostics (partial) yazildi: %s", diag_path)
        except OSError:
            pass
        raise
    log.info(
        "Label raster diagnostics: kind=%s dtype=%s min=%s max=%s zero=%s "
        "one=%s gt_one=%s in_DOY_range=%s finite=%s masked=%s",
        label_diag.get("label_kind"), label_diag.get("dtype"),
        label_diag.get("min"), label_diag.get("max"), label_diag.get("count_zero"),
        label_diag.get("count_one"), label_diag.get("count_gt_one"),
        label_diag.get("count_in_label_doy_range"),
        label_diag.get("finite_pixel_count"),
        label_diag.get("masked_or_nodata_pixel_count"),
    )

    label_path = align_label_to_reference(label_path_raw, ref_profile, out_dir)

    source_mask_path = (
        ctx["step7e_output_dir"] / "fused_lst_source_mask.tif"
        if (ctx is not None and not ctx.get("is_kozan"))
        else (BASE_DIR / FUSED_SOURCE_MASK_RELPATH)
    )
    if not source_mask_path.exists():
        log.warning(
            "Fused LST source mask bulunamadi (%s); observed/gapfilled "
            "fraction sutunlari NaN birakilacak.", source_mask_path,
        )
        source_mask_path = None
    else:
        validate_grid_alignment(reference_path, {"fused_lst_source_mask": source_mask_path})

    log.info("Referans 30 m grid: %s (%dx%d)", reference_path, ref_profile["width"], ref_profile["height"])
    log.info("MCD64A1 etiket rasteri (hizali): %s (label_kind=%s)", label_path, label_kind)
    log.info("Landcover rasteri (hizali): %s", landcover_path)
    log.info("Cozulmus predictor sayisi: %d (eksik/opsiyonel: %s)", len(predictor_paths), missing_optional)

    # --- Leakage-safe pre-label exclusion (opt-in per experiment; currently
    # only mugla_2021). Config-driven via ctx["exclude_pre_label_burns"] --
    # never hard-coded to a specific experiment_id. If enabled, the Step6B
    # gate's canonical cell-level manifest is REQUIRED; missing it is a
    # fail-fast error (never a silent warning), since proceeding without it
    # would silently let already-burned cells leak back into the dataset. ---
    exclude_pre_label_burns = bool(ctx is not None and ctx.get("exclude_pre_label_burns", False))
    pre_label_excluded_cell_ids = None
    pre_label_exclusion_manifest_path: str | None = None
    if exclude_pre_label_burns:
        manifest_path = ctx["gate_labels_dir"] / PRE_LABEL_EXCLUSION_MANIFEST_FILENAME
        pre_label_exclusion_manifest_path = str(manifest_path)
        pre_label_excluded_cell_ids = read_pre_label_exclusion_manifest(manifest_path)
        log.info(
            "Pre-label exclusion AKTIF [%s]: %d hucre (manifest: %s) analiz "
            "evreninden dislanacak.", ctx["experiment_id"],
            len(pre_label_excluded_cell_ids), manifest_path,
        )

    result = build_dataset(
        reference_path=reference_path,
        label_path=label_path,
        label_kind=label_kind,
        predictor_paths=predictor_paths,
        landcover_path=landcover_path,
        source_mask_path=source_mask_path,
        output_dir=out_dir,
        min_valid_fraction=min_valid_fraction,
        burnable_threshold=burnable_threshold,
        label_start=effective_label_start,
        label_end=effective_label_end,
        pre_label_excluded_cell_ids=pre_label_excluded_cell_ids,
    )
    result["label_source_description"] = label_source_description
    result["label_raster_diagnostics"] = label_diag
    df = result["dataframe"]

    fatal, soft = run_quality_checks(
        df, result, ref_pixel_total, allow_all_burned,
        label_start=effective_label_start, label_end=effective_label_end,
    )
    for w in soft:
        log.warning("QUALITY CHECK (warning): %s", w)
    result["warnings"].extend(soft)

    if fatal:
        for p in fatal:
            log.error("QUALITY CHECK (FATAL): %s", p)
        raise Step8AError(
            "Step8A kalite kontrolleri BASARISIZ (cikti YAZILMADI):\n  - "
            + "\n  - ".join(fatal)
        )

    diag_rasters = write_diagnostic_rasters(result, out_dir)
    geojson_path = write_cell_preview_geojson(df, result, out_dir)

    csv_path = out_dir / "step8a_500m_modeling_dataset.csv"
    parquet_path = out_dir / "step8a_500m_modeling_dataset.parquet"
    parquet_written = False

    if write_csv:
        df.to_csv(csv_path, index=False)
        log.info("CSV yazildi: %s", csv_path)

    if write_parquet:
        try:
            df.to_parquet(parquet_path, index=False)
            parquet_written = True
            log.info("Parquet yazildi: %s", parquet_path)
        except (ImportError, ValueError) as exc:
            log.warning(
                "Parquet yazilamadi (pyarrow/fastparquet eksik olabilir): %s. "
                "CSV ile devam ediliyor.", exc,
            )

    stats_path = write_stats(
        out_dir, result, reference_path, label_path, predictor_paths,
        landcover_path, landcover_info, source_mask_path,
        min_valid_fraction, burnable_threshold, result["warnings"],
        exclude_pre_label_burns=exclude_pre_label_burns,
        pre_label_exclusion_manifest_path=pre_label_exclusion_manifest_path,
    )
    # parquet_written flag is added post-hoc so write_stats stays pure.
    stats_data = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_data["parquet_written"] = parquet_written
    stats_data["csv_written"] = bool(write_csv)
    stats_data["diagnostic_rasters"] = {k: str(v) for k, v in diag_rasters.items()}
    stats_data["cell_preview_geojson"] = str(geojson_path)
    stats_data["cell_level"] = "500m_reconstructed_mcd64a1_cell"
    stats_data["no_30m_label_claim"] = True
    if ctx is not None:
        stats_data["experiment_id"] = ctx["experiment_id"]
        stats_data["region_key"] = ctx["region_key"]
        stats_data["predictor_window"] = [ctx["predictor_start_date"], ctx["predictor_end_date"]]
        stats_data["label_window"] = [ctx["label_start_date"], ctx["label_end_date"]]
        stats_data["baseline_years"] = ctx["baseline_years"]
        # BUG FIX: write_stats() yukarida top-level predictor_start_date/
        # predictor_end_date/label_start_date/label_end_date alanlarini HER
        # ZAMAN None/None/LABEL_START_DATE/LABEL_END_DATE (Kozan'in legacy
        # core.config sabitleri) ile dolduruyordu -- ctx'ten BAGIMSIZ. Bu,
        # Manavgat gibi deney-farkinda calistirmalarda JSON'da Kozan'a ait
        # 2023 tarihlerinin (veya null) sessizce kalmasina yol aciyordu,
        # oysa dogru degerler zaten predictor_window/label_window'da mevcuttu.
        # Burada bu 4 alani da ctx'in GERCEK tarihleriyle DUZELTIYORUZ.
        stats_data["predictor_start_date"] = ctx["predictor_start_date"]
        stats_data["predictor_end_date"] = ctx["predictor_end_date"]
        stats_data["label_start_date"] = ctx["label_start_date"]
        stats_data["label_end_date"] = ctx["label_end_date"]

    # FAIL-FAST (madde 6): ciktilar DISKE YAZILMADAN once, tarih alanlarinin
    # ctx / predictor_window / label_window / elle-dogrulanmis beklenen
    # degerlerle birebir tutarli oldugunu ve Kozan'in legacy tarihlerini
    # icermedigini dogrula. Kozan icin hicbir sey yapmaz.
    assert_metadata_dates_consistent(stats_data, ctx)

    stats_path.write_text(json.dumps(stats_data, indent=2, default=str), encoding="utf-8")

    summary_path = write_summary(out_dir, result, stats_path, effective_label_start, effective_label_end)

    # FAIL-FAST (madde 7): YAZILMIS metin ciktilarini tarayip Kozan'in legacy
    # tarih dizgilerinden herhangi birinin sizip sizmadigini dogrula. Bu,
    # alan-bazli kontrolun yakalayamayacagi serbest-metin sizintilarini
    # (markdown cumleleri, warnings[] listesi, vb.) yakalar.
    assert_no_stale_kozan_dates_in_outputs([stats_path, summary_path], ctx)

    log.info("Dataset satir sayisi: %d (valid_for_modeling=%d)", len(df), result["counters"]["valid_modeling_cells"])
    log.info("Stats: %s", stats_path)
    log.info("Summary: %s", summary_path)
    if soft:
        log.warning("Toplam %d kalite kontrolu uyarisi (bkz. stats JSON 'warnings').", len(soft))
    log.info("=" * 60)
    log.info("STEP 8A TAMAMLANDI (no model trained, no FIRMS label used, MCD64A1 primary label)")
    log.info("=" * 60)

    return {
        "csv_path": str(csv_path) if write_csv else None,
        "parquet_path": str(parquet_path) if parquet_written else None,
        "stats_path": str(stats_path),
        "summary_path": str(summary_path),
        "row_count": len(df),
        "valid_modeling_cells": result["counters"]["valid_modeling_cells"],
    }


def run_step8a(ctx: dict | None = None, force: bool = False, **kwargs) -> dict:
    """
    Step8A: label-honest ~500 m MCD64A1-cell modeling dataset olusturur.

    ctx: None ise (varsayilan) legacy Kozan davranisi BIREBIR korunur --
        outputs/step8a'ya yazar, girdileri legacy Kozan yollarindan (data/,
        outputs/step5, outputs/step5c, outputs/step7d, outputs/step7e,
        data/landcover) kesfeder. Verilirse (Kozan-disi, or. manavgat_2021):
        outputs/experiments/<experiment_id>/step8a'ya yazar; TUM girdiler
        (label, referans grid, predictor'lar, landcover, fused source mask)
        o deneyin namespaced dizinlerinden cozulur -- Kozan'in legacy
        paylasilan dosyalarina ASLA dokunulmaz.

    30 m piksel HICBIR ZAMAN label olarak KULLANILMAZ; hucre seviyesi her
    zaman MCD64A1'in (yaklasik) native ~500 m gridine (block/tile
    reconstruction ile) sabittir (bkz. compute_block_size_pixels()).
    """
    use_ctx = ctx is not None and not ctx.get("is_kozan")
    output_dir_arg = str(ctx["step8a_output_dir"]) if use_ctx else STEP8A_OUTPUT_DIR
    result = main(output_dir_arg=output_dir_arg, force=force, ctx=ctx if use_ctx else None, **kwargs)
    if ctx is not None:
        result["experiment_id"] = ctx["experiment_id"]
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step8A: aggregate 30 m predictor rasters onto the native "
        "~500 m MCD64A1 grid to prepare a burned-area modeling dataset "
        "(no model trained, no FIRMS label, MCD64A1 primary label)."
    )
    parser.add_argument("--output-dir", type=str, default=STEP8A_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-csv", dest="write_csv", action="store_true", default=STEP8A_WRITE_CSV)
    parser.add_argument("--no-write-csv", dest="write_csv", action="store_false")
    parser.add_argument("--write-parquet", dest="write_parquet", action="store_true", default=STEP8A_WRITE_PARQUET)
    parser.add_argument("--no-write-parquet", dest="write_parquet", action="store_false")
    parser.add_argument("--min-valid-fraction", type=float, default=STEP8A_MIN_30M_VALID_FRACTION)
    parser.add_argument("--burnable-threshold", type=float, default=STEP8A_BURNABLE_FRACTION_THRESHOLD)
    parser.add_argument("--label-raster", type=str, default=None)
    parser.add_argument("--reference-30m", type=str, default=None)
    parser.add_argument(
        "--allow-all-burned", action="store_true",
        help="Override the fail-fast check that aborts when every valid cell "
        "is burned / burned_rate > 0.5. Only use if a majority-burned label "
        "window is genuinely expected.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir_arg=args.output_dir,
        force=args.force,
        write_csv=args.write_csv,
        write_parquet=args.write_parquet,
        min_valid_fraction=args.min_valid_fraction,
        burnable_threshold=args.burnable_threshold,
        label_raster_arg=args.label_raster,
        reference_30m_arg=args.reference_30m,
        allow_all_burned=args.allow_all_burned,
    )