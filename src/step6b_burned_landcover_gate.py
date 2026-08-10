"""
step6b_burned_landcover_gate.py

Burned-landcover diagnostic gate (Step6B).

WHY THIS EXISTS
---------------
Supervisor feedback: Kozan 2023 burned MCD64A1 cells are cropland/anız-burning
dominated, not natural-vegetation wildfire. The methodology (Step8A-8E) is
correct, but every NEW AOI/experiment must first pass a burned-landcover gate
BEFORE modeling, so we know up front whether an AOI is a genuine wildfire
candidate, a cropland/anız control, or has too few burned positives to say
anything useful.

This module answers exactly one question: "of the MCD64A1-burned ~500 m
cells, what landcover dominates them?" It does NOT train a model, does NOT
touch Step8 model science, and does NOT require Step7's thermal predictors
to exist -- only the raw MCD64A1 BurnDate raster (see Step6's
`export_raw_mcd64a1_labels()`) and a landcover raster aligned to the
predictor grid. This lets the gate run right after Step6's label export,
BEFORE Step7A-7E and Step8A.

REUSE, NOT REIMPLEMENTATION
----------------------------
This module deliberately imports its label/landcover/500m-cell-reconstruction
helpers directly from `src/step8a_prepare_500m_modeling_dataset.py`:
    resolve_reference_30m, resolve_label_raster, resolve_landcover,
    align_label_to_reference, inspect_label_raster, label_window_doy_bounds,
    doy_to_month_and_date, compute_block_size_pixels, mode_and_agreement,
    ESA_WORLDCOVER_CLASSES, LC_TREE_COVER, LC_SHRUBLAND, LC_GRASSLAND,
    LC_CROPLAND, LABEL_KIND_RAW, Step8AError.
This is the SAME 500 m block/tile reconstruction Step8A uses (same block
size, same tiling utility, same landcover class mapping), so gate numbers
are as comparable as possible to Step8A's own burned/landcover counts.
Step8A does NOT import this module, so there is no circular import.

Unlike Step8A's `build_dataset()`, this gate does NOT read any continuous
predictor raster (NDVI/DEM/thermal) -- only the label (BurnDate) and
landcover rasters -- so it can run before Step7/Step8A outputs exist.

GATE LEVEL
----------
    gate_level = "500m_reconstructed_mcd64a1_cell"
(an approximate native MCD64A1 grid reconstructed from the 30 m reference
grid, exactly like Step8A -- see STEP8A_MCD64A1_NATIVE_CELL_SIZE_M /
STEP8A_REFERENCE_PIXEL_SIZE_M in core/config.py).

DECISION RULES (supervisor-specified)
--------------------------------------
    burned_count < STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES
        -> "insufficient_burned_positives"
    burned_tree_shrub_grass_count / burned_count >= NATURAL_THRESHOLD (0.50)
        -> "wildfire_candidate_pass"
    burned_cropland_dominant_count / burned_count >= CROPLAND_THRESHOLD (0.50)
        -> "cropland_dominated_control"
    otherwise
        -> "mixed_or_uncertain"

This is DIAGNOSTIC ONLY. A "cropland_dominated_control" decision (the
expected Kozan 2023 outcome) does NOT stop the pipeline. This module only
raises (fails fast) if:
    - the raw BurnDate raster is missing or binary-looking (not real DOY),
    - required input rasters (reference grid / landcover) are missing,
    - the landcover class mapping cannot be resolved.

Inputs (read-only):
    outputs/validation/labels/mcd64a1_raw.tif   (preferred; see resolve_label_raster)
    outputs/step5/current_period_median_celsius.tif   (reference 30 m grid)
    data/landcover/landcover_esa_worldcover_v200_aligned_to_landsat.tif (preferred)
    data/landcover/landcover_esa_worldcover_v200.tif                    (fallback)

Outputs:
    outputs/validation/labels/burned_landcover_gate.json
    outputs/validation/labels/burned_landcover_gate.md
    outputs/validation/labels/burned_landcover_gate.csv

CLI:
    python src/step6b_burned_landcover_gate.py
    python src/step6b_burned_landcover_gate.py --force
    python src/step6b_burned_landcover_gate.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from core.config import (
    LABEL_END_DATE,
    LABEL_START_DATE,
    STEP6_BURNED_LANDCOVER_GATE_CROPLAND_THRESHOLD,
    STEP6_BURNED_LANDCOVER_GATE_LEVEL,
    STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES,
    STEP6_BURNED_LANDCOVER_GATE_NATURAL_THRESHOLD,
    STEP6_LABEL_OUTPUT_DIR,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.utils.tiling import make_tile_grid

# --- Reuse Step8A's already-validated label/landcover/500m-cell helpers. ---
# Step8A does NOT import this module -> no circular import.
from src.step8a_prepare_500m_modeling_dataset import (
    ESA_WORLDCOVER_CLASSES,
    LC_CROPLAND,
    LC_GRASSLAND,
    LC_SHRUBLAND,
    LC_TREE_COVER,
    LABEL_KIND_RAW,
    Step8AError,
    _grid_matches,
    align_label_to_reference,
    classify_burndate_relative_to_label,
    compute_block_size_pixels,
    compute_cell_identity,
    doy_to_month_and_date,
    inspect_label_raster,
    mode_and_agreement,
    prepare_aligned_landcover,
    resolve_label_raster,
    resolve_landcover,
    resolve_reference_30m,
)

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step6b_burned_landcover_gate")

GATE_LEVEL_500M = "500m_reconstructed_mcd64a1_cell"

# Canonical cell-level pre-label exclusion manifest filenames (see
# write_pre_label_exclusion_manifest() below). Parquet is the canonical
# machine-readable artifact Step8A reads; CSV is the same rows for human
# review only.
PRE_LABEL_EXCLUSION_MANIFEST_PARQUET = "pre_label_excluded_cells.parquet"
PRE_LABEL_EXCLUSION_MANIFEST_CSV = "pre_label_excluded_cells.csv"
PRE_LABEL_EXCLUSION_MANIFEST_METADATA = "pre_label_excluded_cells_metadata.json"

# --- Primary-population sample-size STOP state -------------------------------
# Emitted by the SEPARATE post-gate primary-population rule (see
# evaluate_primary_population_sample_size). This is NOT a burned-landcover
# composition outcome and must never be reported as one.
PRIMARY_POPULATION_STOP = "insufficient_primary_population_burned"
PRIMARY_POPULATION_GATE_PASS = "pass"
PRIMARY_POPULATION_GATE_STOP = "stop"


class Step6BError(SystemExit):
    """Fail-fast error for Step6B (extends SystemExit like other steps)."""


def _dominant_landcover_name(code: int) -> str:
    return ESA_WORLDCOVER_CLASSES.get(code, f"unknown_{code}")


def _safe_fraction(numerator: int, denominator: int):
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _date_to_doy(date_str: str) -> int:
    """'YYYY-MM-DD' -> day-of-year (1..366). Pure; for the Bördübet check."""
    from datetime import datetime

    return int(datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday)


def classify_gate_decision(
    burned_count: int,
    natural_fraction,
    cropland_fraction,
    min_positives: int,
    natural_threshold: float,
    cropland_threshold: float,
) -> tuple[str, str]:
    """
    Pure, testable implementation of the supervisor-specified gate decision
    rules. Returns (decision, reason). This is the SAME logic Step6B has
    always used; it is only extracted so the natural-vegetation gate decision
    can be unit-tested without building rasters.

    Note: the natural-vegetation numerator is tree+shrub+GRASS
    (burned_natural_vegetation_fraction), NOT tree+shrub. tree+shrub is
    reported separately and never conflated with the pass rule.
    """
    if burned_count < min_positives:
        return (
            "insufficient_burned_positives",
            f"burned_count={burned_count} < min_positives={min_positives}.",
        )
    if natural_fraction is not None and natural_fraction >= natural_threshold:
        return (
            "wildfire_candidate_pass",
            f"burned_tree_shrub_grass_count/burned_count="
            f"{natural_fraction:.3f} >= natural_threshold={natural_threshold}.",
        )
    if cropland_fraction is not None and cropland_fraction >= cropland_threshold:
        return (
            "cropland_dominated_control",
            f"burned_cropland_dominant_count/burned_count="
            f"{cropland_fraction:.3f} >= cropland_threshold={cropland_threshold}.",
        )
    return (
        "mixed_or_uncertain",
        "Ne natural (tree+shrub+grass dominant) ne de cropland-dominant "
        "esigi karsilandi (mixed/uncertain burned-landcover kompozisyonu).",
    )


def evaluate_primary_population_sample_size(
    primary_population_burned_count: int,
    population: str | None,
    min_burned_required: int | None,
) -> dict | None:
    """
    Pure, testable implementation of the SEPARATE post-gate primary-population
    sample-size rule.

    THIS IS NOT THE BURNED-LANDCOVER GATE. The composition gate above keeps
    its own generic total-burned feasibility threshold
    (`min_positives`, core/config.py STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES
    = 30), which is UNCHANGED and applies to `burned_count` -- every burned
    cell, any landcover. This rule is evaluated separately, AFTER all
    exclusions and landcover classification, on the PRIMARY POPULATION only
    (e.g. burned cells whose dominant landcover is tree/shrub/grass), and is
    configured per experiment in the registry -- it is not a global constant
    and it is not applied to experiments that do not declare it.

    A shortfall is a STOP for downstream authorization. It is deliberately
    reported in its own structure with its own decision vocabulary so it can
    never be mistaken for (or reinterpreted as) a failure of the
    natural/cropland composition gate.

    Returns None when the experiment declares no such rule (every existing
    experiment), so their gate output is unaffected.
    """
    if population is None and min_burned_required is None:
        return None
    if population is None or min_burned_required is None:
        raise Step6BError(
            "Primary-population sample-size rule is only half configured "
            f"(population={population!r}, min_burned_required="
            f"{min_burned_required!r}). Both must be declared together."
        )

    burned_count = int(primary_population_burned_count)
    passed = burned_count >= int(min_burned_required)
    return {
        "population": population,
        "burned_count": burned_count,
        "min_burned_required": int(min_burned_required),
        "decision": PRIMARY_POPULATION_GATE_PASS if passed else PRIMARY_POPULATION_GATE_STOP,
        "stop_state": None if passed else PRIMARY_POPULATION_STOP,
        "reason": (
            f"{population} burned_count={burned_count} >= "
            f"min_burned_required={min_burned_required}."
            if passed else
            f"{population} burned_count={burned_count} < "
            f"min_burned_required={min_burned_required}; downstream is NOT "
            "authorized. This is a PRIMARY-POPULATION SAMPLE-SIZE stop, NOT a "
            "natural/cropland composition gate failure."
        ),
        "evaluated_after": (
            "all analysis-universe exclusions (pre-label AND historical) and "
            "landcover classification"
        ),
        "independent_of": (
            "the burned-landcover composition gate and its generic "
            "min_positives total-burned feasibility threshold"
        ),
        "downstream_authorized": False,
    }


def _resolve_explicit_landcover(
    landcover_path_arg: str, reference_path: Path, out_dir: Path
) -> tuple[Path, dict]:
    """
    Acikca verilen (explicit) landcover rasterini cozer -- legacy
    data/landcover/ yollarina HIC dokunmadan.

    Verilen dosya referans gridle birebir uyusuyorsa dogrudan kullanilir.
    Uyusmuyorsa KAYNAK kabul edilir ve Step8A'nin prepare_aligned_landcover'i
    (nearest-neighbor; kategorik veri ASLA bilinear resample edilmez) ile
    out_dir altina (<stem>_aligned_to_reference.tif) hizalanir.
    """
    p = Path(landcover_path_arg)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        raise Step6BError(f"Belirtilen landcover rasteri bulunamadi: {p}")

    with rasterio.open(reference_path) as ref:
        ref_w, ref_h, ref_crs, ref_t = ref.width, ref.height, ref.crs, ref.transform

    if _grid_matches(p, ref_w, ref_h, ref_crs, ref_t):
        log.info("Explicit landcover referans gridle zaten hizali: %s", p)
        return p, {
            "original_landcover_path": str(p),
            "aligned_landcover_path": str(p),
            "landcover_alignment_method": "explicit_preexisting_aligned",
        }

    aligned_path = out_dir / f"{p.stem}_aligned_to_reference.tif"
    log.info(
        "Explicit landcover referans gridle uyusmuyor; nearest-neighbor ile "
        "hizalaniyor -> %s", aligned_path,
    )
    try:
        result = prepare_aligned_landcover(reference_path, p, aligned_path)
    except Step8AError as exc:
        raise Step6BError(str(exc)) from exc
    return result["path"], {
        "original_landcover_path": str(p),
        "aligned_landcover_path": str(result["path"]),
        "landcover_alignment_method": "nearest_neighbor_to_reference_grid",
        "landcover_alignment_created": result["created"],
    }


# =============================================================================
# Core gate computation (500 m-cell level)
# =============================================================================
def compute_gate(
    label_path: Path,
    label_kind: str,
    reference_path: Path,
    landcover_path: Path,
    label_start: str,
    label_end: str,
    output_dir: Path,
    min_positives: int,
    natural_threshold: float,
    cropland_threshold: float,
    exclude_pre_label_burns: bool = False,
    pre_label_label_path: Path | None = None,
    predictor_start: str | None = None,
    predictor_end: str | None = None,
    bordubet_check_window: tuple[str, str] | None = None,
    experiment_id: str | None = None,
    exclude_historical_burns: bool = False,
    historical_excluded_cell_ids: frozenset[str] | None = None,
    historical_exclusion_config: dict | None = None,
    primary_population: str | None = None,
    min_primary_population_burned: int | None = None,
) -> dict:
    """
    Reconstructs approximate native ~500 m MCD64A1 cells from the 30 m
    reference grid (SAME block size / tiling logic as Step8A) and, for each
    cell, determines (a) burned/unburned from the raw BurnDate raster and
    (b) the dominant landcover class -- WITHOUT reading any continuous
    predictor raster.

    LEAKAGE-SAFE PRE-LABEL EXCLUSION (opt-in; default OFF = legacy behaviour):
        When exclude_pre_label_burns is True, a SEPARATE pre-label BurnDate
        raster (pre_label_label_path, exported over [predictor_start,
        label_start)) is read. Any cell with a positive pre-label BurnDate is
        marked pre_label_burn_excluded and REMOVED from the analysis universe
        BEFORE the burned/unburned + natural-vegetation composition is
        evaluated: it is NOT counted as burned, NOT counted as unburned, and
        NOT included in any gate denominator. This is required because the
        canonical label raster is DOY-masked to the label window and therefore
        sets pre-label burns to 0 (indistinguishable from truly-unburned),
        which would otherwise leak already-burned cells into the negatives.
        With exclude_pre_label_burns=False the function behaves EXACTLY as
        before (Kozan/Manavgat/Bejís unchanged).

    HISTORICAL BURN EXCLUSION (opt-in; SEPARATE, INDEPENDENT axis):
        When exclude_historical_burns is True, a caller-supplied set of
        cell_id values (built and validated by
        src/historical_burn_exclusion.py from an EARLIER experiment's
        canonical Step8A burned cells) is also removed from the analysis
        universe. This is config-driven -- the caller resolves it from the
        registry; nothing here is keyed off an experiment_id.

    UNION SEMANTICS:
        Both flags are derived INDEPENDENTLY for every cell:

            pre_label_burn_excluded    (this experiment's predictor window)
            historical_burn_excluded   (an earlier event, same geography)
            analysis_excluded = pre_label_burn_excluded OR historical_burn_excluded

        A cell may carry BOTH flags; the overlap is counted and reported, and
        each reason keeps its own independent count. A union-excluded cell
        enters NEITHER the burned NOR the unburned gate counts, and appears in
        NO gate denominator.
    """
    with rasterio.open(reference_path) as ref:
        ref_w, ref_h = ref.width, ref.height
        ref_transform = ref.transform
        ref_crs = ref.crs
    ref_profile = {"width": ref_w, "height": ref_h, "crs": ref_crs, "transform": ref_transform}

    aligned_label_path = align_label_to_reference(label_path, ref_profile, output_dir)

    block_size = compute_block_size_pixels()
    tile_grid = make_tile_grid({"width": ref_w, "height": ref_h}, tile_size_pixels=block_size)
    tiles = tile_grid["tiles"]
    log.info(
        "500 m hucre rekonstruksiyonu (Step8A ile ayni block/tile mantigi): "
        "block_size_pixels=%d, toplam hucre (yaklasik)=%d",
        block_size, len(tiles),
    )

    label_src = rasterio.open(aligned_label_path)
    landcover_src = rasterio.open(landcover_path)
    landcover_nodata = landcover_src.nodata if landcover_src.nodata is not None else 0

    burn_month_available = (label_kind == LABEL_KIND_RAW)

    # --- Pre-label exclusion setup (opt-in) ---
    pre_label_src = None
    aligned_pre_label_path = None
    if exclude_pre_label_burns:
        if not burn_month_available:
            raise Step6BError(
                "exclude_pre_label_burns=True icin gercek raw BurnDate "
                "(DOY) etiketi gerekir; binary fallback ile pre-label "
                "burn tarihi belirlenemez."
            )
        if pre_label_label_path is None:
            raise Step6BError(
                "exclude_pre_label_burns=True verildi ama pre_label_label_path "
                "yok. Pre-label burn'leri dislamak icin [predictor_start, "
                "label_start) penceresi uzerinden AYRI bir MCD64A1 BurnDate "
                "rasteri gereklidir (canonical label rasteri label penceresine "
                "DOY-maskeli oldugu icin pre-label burn'leri 0'a esitler). "
                "Bkz. export_raw_mcd64a1_prelabel_labels()."
            )
        aligned_pre_label_path = align_label_to_reference(
            Path(pre_label_label_path), ref_profile, output_dir
        )
        pre_label_src = rasterio.open(aligned_pre_label_path)

    # --- Historical exclusion setup (opt-in; SEPARATE axis) ---
    # The set is built + provenance-validated by the caller
    # (src/historical_burn_exclusion.py); an enabled contract with no set is a
    # fail-closed error, never a silent empty exclusion.
    if exclude_historical_burns and historical_excluded_cell_ids is None:
        raise Step6BError(
            "exclude_historical_burns=True verildi ama historical_excluded_"
            "cell_id kumesi yok. Historical exclusion manifest'i gate'ten ONCE "
            "uretilip dogrulanmalidir (bkz. src/historical_burn_exclusion.py "
            "build_historical_burn_exclusion_manifest)."
        )
    historical_ids: frozenset[str] = (
        historical_excluded_cell_ids if exclude_historical_burns else frozenset()
    )

    # Bördübet predictor-window fire check DOY bounds (default 2021-06-21..25).
    bord_doy_lo = bord_doy_hi = None
    if bordubet_check_window is not None:
        bord_doy_lo = _date_to_doy(bordubet_check_window[0])
        bord_doy_hi = _date_to_doy(bordubet_check_window[1])
    pred_doy_lo = pred_doy_hi = None
    if predictor_start is not None and predictor_end is not None:
        pred_doy_lo = _date_to_doy(predictor_start)
        pred_doy_hi = _date_to_doy(predictor_end)

    total_cells = 0
    burned_count = 0
    unburned_count = 0
    burned_dominant_counts: dict = {}
    unburned_dominant_counts: dict = {}
    burned_cells_without_valid_landcover = 0
    unburned_cells_without_valid_landcover = 0
    warnings_list: list[str] = []

    # --- Pre-label exclusion + temporal QA accumulators ---
    pre_label_excluded_count = 0
    pre_label_excluded_dominant_counts: dict = {}
    pre_label_excluded_without_valid_landcover = 0
    # Cell-level rows for the canonical exclusion manifest (see
    # write_pre_label_exclusion_manifest()); populated ONLY when
    # exclude_pre_label_burns is True. Popped from the returned dict before
    # burned_landcover_gate.json/.md/.csv are written (main()) -- kept OUT
    # of those aggregate reports so existing experiments' gate outputs are
    # never affected by this feature.
    pre_label_excluded_rows: list[dict] = []
    # --- Historical exclusion accumulators (INDEPENDENT of the pre-label
    # ones above; a cell may be counted in both, and the overlap is tracked
    # explicitly so the union arithmetic can be verified). ---
    historical_excluded_count = 0
    historical_excluded_dominant_counts: dict = {}
    historical_excluded_without_valid_landcover = 0
    pre_label_historical_overlap_count = 0
    total_unique_excluded_count = 0
    temporal_qa = {
        "cell_level_note": (
            "Counts are per reconstructed ~500 m cell (representative BurnDate "
            "= mode of positive DOY within the cell), not per 30 m pixel."
        ),
        "count_before_predictor_start": 0,
        "count_within_predictor_window": 0,
        "count_between_predictor_end_and_label_start": 0,
        "count_within_label_window": 0,
        "count_after_label_end": 0,
        "count_missing_or_zero_burndate": 0,
        "count_pre_label_burn_excluded": 0,
        "bordubet_window": (
            list(bordubet_check_window) if bordubet_check_window else None
        ),
        "bordubet_window_burned_cell_count": 0,
        "min_nonzero_burndate_iso": None,
        "max_nonzero_burndate_iso": None,
        "out_of_span_note": (
            "count_before_predictor_start and count_after_label_end are 0 by "
            "export construction: the pre-label raster is DOY-masked to "
            "[predictor_start, label_start) and the label raster to "
            "[label_start, label_end]; burns outside that combined span are "
            "not exported. Detecting them would require a wider export."
        ),
    }
    _min_doy_seen = None
    _max_doy_seen = None

    def _iso_for_doy(doy_val: float, year: int) -> str:
        from datetime import datetime as _dt, timedelta as _td
        return (_dt(year, 1, 1) + _td(days=int(round(doy_val)) - 1)).strftime("%Y-%m-%d")

    _label_year = int(label_start[:4])

    for tile in tiles:
        col_off, row_off, w, h = tile["write_window"]
        window = Window(col_off, row_off, w, h)
        total_cells += 1
        cell_id, row_500m, col_500m = compute_cell_identity(row_off, col_off, block_size)

        # --- Landcover (dominant class within the same 500 m block) ---
        # Computed FIRST so that pre-label-EXCLUDED cells also get a dominant
        # landcover breakdown (task requirement: excluded-by-landcover).
        lc_arr = landcover_src.read(1, window=window, masked=True)
        lc_filled = lc_arr.filled(landcover_nodata).astype("float64")
        lc_valid = ~np.ma.getmaskarray(lc_arr)
        lc_valid = lc_valid & np.isfinite(lc_filled) & (lc_filled != landcover_nodata)
        lc_valid_count = int(lc_valid.sum())

        dominant_name = None
        if lc_valid_count > 0:
            lc_vals = lc_filled[lc_valid].astype(int)
            uniq, counts = np.unique(lc_vals, return_counts=True)
            dominant_class = int(uniq[int(np.argmax(counts))])
            dominant_name = _dominant_landcover_name(dominant_class)

        # --- Exclusion axis 1: pre-label burn (leakage-safe; opt-in) ---
        # A cell that already burned BEFORE label_start carries post-fire
        # predictor signatures for THIS experiment's own predictor window.
        pre_label_burn_excluded = False
        if pre_label_src is not None:
            pre_arr = pre_label_src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
            pre_valid = pre_arr[np.isfinite(pre_arr)]
            pre_positive = pre_valid[pre_valid > 0]
            if pre_positive.size > 0:
                pre_label_burn_excluded = True
                rep_pre_doy, _c, _n = mode_and_agreement(pre_positive)
                pre_label_excluded_count += 1
                if dominant_name is not None:
                    pre_label_excluded_dominant_counts[dominant_name] = (
                        pre_label_excluded_dominant_counts.get(dominant_name, 0) + 1
                    )
                else:
                    pre_label_excluded_without_valid_landcover += 1
                # Temporal QA for the excluded (pre-label) cell.
                temporal_qa["count_pre_label_burn_excluded"] += 1
                if pred_doy_lo is not None and pred_doy_lo <= rep_pre_doy <= pred_doy_hi:
                    temporal_qa["count_within_predictor_window"] += 1
                elif pred_doy_hi is not None and rep_pre_doy > pred_doy_hi:
                    # after predictor_end but before label_start (contiguous
                    # windows -> normally 0)
                    temporal_qa["count_between_predictor_end_and_label_start"] += 1
                if bord_doy_lo is not None and bord_doy_lo <= rep_pre_doy <= bord_doy_hi:
                    temporal_qa["bordubet_window_burned_cell_count"] += 1
                _iso = _iso_for_doy(rep_pre_doy, _label_year)
                if _min_doy_seen is None or rep_pre_doy < _min_doy_seen:
                    _min_doy_seen = rep_pre_doy
                    temporal_qa["min_nonzero_burndate_iso"] = _iso
                if _max_doy_seen is None or rep_pre_doy > _max_doy_seen:
                    _max_doy_seen = rep_pre_doy
                    temporal_qa["max_nonzero_burndate_iso"] = _iso
                pre_label_excluded_rows.append({
                    "experiment_id": experiment_id,
                    "cell_id": cell_id,
                    "row_500m": row_500m,
                    "col_500m": col_500m,
                    "pre_label_burn_date": _iso,
                    "exclusion_reason": "pre_label_burn_excluded",
                    # QA/audit only -- NOT the Step8 population definition
                    # (that stays owned by Step8A's burnable_threshold logic).
                    "landcover_dominant": dominant_name,
                    "burnable_tree_shrub": bool(dominant_name in ("tree_cover", "shrubland")),
                    "burnable_tree_shrub_grass": bool(
                        dominant_name in ("tree_cover", "shrubland", "grassland")
                    ),
                })

        # --- Exclusion axis 2: historical burn (earlier event, same AOI) ---
        # Derived INDEPENDENTLY of axis 1 above: the two reasons never
        # short-circuit each other, so a cell excluded for both is counted in
        # both breakdowns and in the overlap.
        historical_burn_excluded = cell_id in historical_ids
        if historical_burn_excluded:
            historical_excluded_count += 1
            if dominant_name is not None:
                historical_excluded_dominant_counts[dominant_name] = (
                    historical_excluded_dominant_counts.get(dominant_name, 0) + 1
                )
            else:
                historical_excluded_without_valid_landcover += 1

        if pre_label_burn_excluded and historical_burn_excluded:
            pre_label_historical_overlap_count += 1

        # --- Union: analysis_excluded = pre_label OR historical ---
        if pre_label_burn_excluded or historical_burn_excluded:
            total_unique_excluded_count += 1
            continue  # EXCLUDED: enters NEITHER burned NOR unburned counts

        # --- Label (BurnDate) within the label window ---
        label_arr = label_src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
        valid_label = label_arr[np.isfinite(label_arr)]
        positive_doy = valid_label[valid_label > 0]

        burned_flag = 0
        rep_label_doy = None
        if positive_doy.size > 0:
            if burn_month_available:
                rep_label_doy, _rep_count, _n = mode_and_agreement(positive_doy)
                month, _iso_lbl = doy_to_month_and_date(rep_label_doy, label_start, label_end)
                burned_flag = 1 if month is not None else 0
            else:
                # Binary fallback label: no DOY/month info; any positive
                # value (i.e. 1) means burned.
                burned_flag = 1

        if burned_flag == 1:
            burned_count += 1
            if dominant_name is not None:
                burned_dominant_counts[dominant_name] = burned_dominant_counts.get(dominant_name, 0) + 1
            else:
                burned_cells_without_valid_landcover += 1
            # Temporal QA (in-window burned cell).
            temporal_qa["count_within_label_window"] += 1
            if rep_label_doy is not None:
                _iso = _iso_for_doy(rep_label_doy, _label_year)
                if _min_doy_seen is None or rep_label_doy < _min_doy_seen:
                    _min_doy_seen = rep_label_doy
                    temporal_qa["min_nonzero_burndate_iso"] = _iso
                if _max_doy_seen is None or rep_label_doy > _max_doy_seen:
                    _max_doy_seen = rep_label_doy
                    temporal_qa["max_nonzero_burndate_iso"] = _iso
        else:
            unburned_count += 1
            if dominant_name is not None:
                unburned_dominant_counts[dominant_name] = unburned_dominant_counts.get(dominant_name, 0) + 1
            else:
                unburned_cells_without_valid_landcover += 1
            temporal_qa["count_missing_or_zero_burndate"] += 1

    label_src.close()
    landcover_src.close()
    if pre_label_src is not None:
        pre_label_src.close()

    if burned_cells_without_valid_landcover > 0:
        warnings_list.append(
            f"{burned_cells_without_valid_landcover} burned hucrede gecerli "
            "landcover pikseli yok; bu hucreler dominant-class sayimlarina "
            "DAHIL EDILMEDI (fraksiyonlarin paydasi yine de burned_count'tur, "
            "bu yuzden dominant-class sayimlarinin toplami burned_count'tan "
            "az olabilir)."
        )
    if not burn_month_available:
        warnings_list.append(
            "Binary burned mask (label_kind=binary_fallback_no_months) "
            "kullanildi; burned/unburned DOY penceresi kontrolu YAPILAMADI "
            "(her pozitif deger burned sayildi). Dogru sonuc icin gercek raw "
            "BurnDate rasterini kullanin."
        )

    burned_tree_cover_count = burned_dominant_counts.get("tree_cover", 0)
    burned_shrubland_count = burned_dominant_counts.get("shrubland", 0)
    burned_grassland_count = burned_dominant_counts.get("grassland", 0)
    burned_cropland_count = burned_dominant_counts.get("cropland", 0)
    burned_tree_shrub_grass_count = burned_tree_cover_count + burned_shrubland_count + burned_grassland_count
    burned_tree_shrub_count = burned_tree_cover_count + burned_shrubland_count
    burned_cropland_dominant_count = burned_cropland_count

    burned_natural_vegetation_fraction = _safe_fraction(burned_tree_shrub_grass_count, burned_count)
    burned_cropland_fraction = _safe_fraction(burned_cropland_dominant_count, burned_count)
    burned_tree_shrub_fraction = _safe_fraction(burned_tree_shrub_count, burned_count)

    # --- Pre-label excluded cells: landcover breakdown (tree/shrub/grass/...) ---
    excl_tree = pre_label_excluded_dominant_counts.get("tree_cover", 0)
    excl_shrub = pre_label_excluded_dominant_counts.get("shrubland", 0)
    excl_grass = pre_label_excluded_dominant_counts.get("grassland", 0)
    excl_crop = pre_label_excluded_dominant_counts.get("cropland", 0)
    excl_other = (
        pre_label_excluded_count
        - excl_tree - excl_shrub - excl_grass - excl_crop
        - pre_label_excluded_without_valid_landcover
    )
    pre_label_excluded_breakdown = {
        "total": pre_label_excluded_count,
        "tree_cover": excl_tree,
        "shrubland": excl_shrub,
        "grassland": excl_grass,
        "tree_shrub": excl_tree + excl_shrub,
        "tree_shrub_grass": excl_tree + excl_shrub + excl_grass,
        "cropland": excl_crop,
        "other": excl_other,
        "without_valid_landcover": pre_label_excluded_without_valid_landcover,
        "by_dominant_class": pre_label_excluded_dominant_counts,
    }

    # --- Historical excluded cells: landcover breakdown (same shape) ---
    hist_tree = historical_excluded_dominant_counts.get("tree_cover", 0)
    hist_shrub = historical_excluded_dominant_counts.get("shrubland", 0)
    hist_grass = historical_excluded_dominant_counts.get("grassland", 0)
    hist_crop = historical_excluded_dominant_counts.get("cropland", 0)
    hist_other = (
        historical_excluded_count
        - hist_tree - hist_shrub - hist_grass - hist_crop
        - historical_excluded_without_valid_landcover
    )
    historical_excluded_breakdown = {
        "total": historical_excluded_count,
        "tree_cover": hist_tree,
        "shrubland": hist_shrub,
        "grassland": hist_grass,
        "tree_shrub": hist_tree + hist_shrub,
        "tree_shrub_grass": hist_tree + hist_shrub + hist_grass,
        "cropland": hist_crop,
        "other": hist_other,
        "without_valid_landcover": historical_excluded_without_valid_landcover,
        "by_dominant_class": historical_excluded_dominant_counts,
    }

    # --- Leakage-safety assertions --------------------------------------
    # (1) Partition: every considered cell is exactly one of
    #     {analysis-excluded, burned, unburned}. The excluded term is the
    #     UNION count -- with no historical exclusion configured it equals
    #     pre_label_excluded_count, so this is the pre-existing assertion.
    analysis_universe_cells = burned_count + unburned_count
    partition_ok = (total_unique_excluded_count + analysis_universe_cells) == total_cells
    if not partition_ok:
        raise Step6BError(
            "LEAKAGE-SAFETY ASSERTION FAILED: total_unique_excluded "
            f"({total_unique_excluded_count}) + burned ({burned_count}) + "
            f"unburned ({unburned_count}) != total_cells ({total_cells}). "
            "An analysis-excluded cell must never enter burned/unburned counts."
        )
    # (2) Union arithmetic: |A u B| == |A| + |B| - |A n B|. A violation means
    #     the two reasons were not derived independently.
    expected_union = (
        pre_label_excluded_count
        + historical_excluded_count
        - pre_label_historical_overlap_count
    )
    if total_unique_excluded_count != expected_union:
        raise Step6BError(
            "EXCLUSION UNION ASSERTION FAILED: total_unique_excluded_count="
            f"{total_unique_excluded_count} != pre_label_burn_excluded_count "
            f"({pre_label_excluded_count}) + historical_burn_excluded_count "
            f"({historical_excluded_count}) - pre_label_historical_overlap_count "
            f"({pre_label_historical_overlap_count}) = {expected_union}."
        )
    if exclude_pre_label_burns and pre_label_excluded_count > 0:
        warnings_list.append(
            f"{pre_label_excluded_count} hucre label_start ({label_start}) "
            "ONCESI yandigi icin analiz evreninden DISLANDI (pre_label_burn_"
            "excluded); bu hucreler ne burned ne unburned sayildi, hicbir "
            "gate paydasinda YOK."
        )
    if exclude_historical_burns and historical_excluded_count > 0:
        warnings_list.append(
            f"{historical_excluded_count} hucre ONCEKI bir olayda "
            "(historical_burn_excluded) yandigi icin analiz evreninden "
            "DISLANDI; bu hucreler ne burned ne unburned sayildi, hicbir gate "
            f"paydasinda YOK. Bunlarin {pre_label_historical_overlap_count} "
            "tanesi ayni zamanda pre_label_burn_excluded'dir (iki neden "
            "BAGIMSIZ olarak turetilir ve ayri ayri raporlanir)."
        )

    # --- Decision rules (supervisor-specified; pure, extracted for testing) ---
    decision, reason = classify_gate_decision(
        burned_count=burned_count,
        natural_fraction=burned_natural_vegetation_fraction,
        cropland_fraction=burned_cropland_fraction,
        min_positives=min_positives,
        natural_threshold=natural_threshold,
        cropland_threshold=cropland_threshold,
    )

    # --- SEPARATE post-gate primary-population sample-size rule ---
    # Evaluated AFTER all exclusions and landcover classification, on the
    # primary population's burned count. None for every experiment that does
    # not declare the rule -- their gate output is byte-for-byte unaffected.
    primary_population_counts = {
        "burnable_tree_shrub_grass": burned_tree_shrub_grass_count,
        "burnable_tree_shrub": burned_tree_shrub_count,
    }
    if primary_population is not None and primary_population not in primary_population_counts:
        raise Step6BError(
            f"Bilinmeyen primary_population={primary_population!r}. Gecerli "
            f"degerler: {sorted(primary_population_counts)}."
        )
    primary_population_gate = evaluate_primary_population_sample_size(
        primary_population_burned_count=(
            primary_population_counts[primary_population]
            if primary_population is not None else 0
        ),
        population=primary_population,
        min_burned_required=min_primary_population_burned,
    )

    return {
        "gate_level": GATE_LEVEL_500M,
        "label_kind": label_kind,
        "burn_month_available": burn_month_available,
        "block_size_pixels": block_size,
        "total_valid_cells_or_pixels_considered": total_cells,
        "analysis_universe_cells_after_exclusions": analysis_universe_cells,
        "burned_count": burned_count,
        "unburned_count": unburned_count,
        # --- Pre-label leakage-safe exclusion ---
        "exclude_pre_label_burns": bool(exclude_pre_label_burns),
        "pre_label_burn_excluded_count": pre_label_excluded_count,
        "pre_label_burn_excluded_breakdown": pre_label_excluded_breakdown,
        "pre_label_burn_exclusion_rule": (
            f"valid nonzero BurnDate calendar date < label_start ({label_start})"
            if exclude_pre_label_burns else "not_applied"
        ),
        # --- Historical burn exclusion (SEPARATE, independent axis) ---
        "exclude_historical_burns": bool(exclude_historical_burns),
        "historical_burn_excluded_count": historical_excluded_count,
        "historical_burn_excluded_breakdown": historical_excluded_breakdown,
        "historical_burn_exclusion_rule": (
            (
                (historical_exclusion_config or {}).get("mask_definition")
                or "cell_id present in the validated historical burn exclusion manifest"
            )
            if exclude_historical_burns else "not_applied"
        ),
        "historical_burn_exclusion_config": historical_exclusion_config,
        # --- Union of the two independent exclusion axes ---
        "pre_label_historical_overlap_count": pre_label_historical_overlap_count,
        "total_unique_excluded_count": total_unique_excluded_count,
        "exclusion_union_rule": (
            "analysis_excluded = pre_label_burn_excluded OR "
            "historical_burn_excluded; total_unique_excluded_count = "
            "pre_label_burn_excluded_count + historical_burn_excluded_count - "
            "pre_label_historical_overlap_count; a union-excluded cell enters "
            "NEITHER burned NOR unburned counts and NO gate denominator."
        ),
        # --- SEPARATE primary-population sample-size rule (None when the
        # experiment declares none). NOT a composition-gate outcome. ---
        "primary_population_sample_size_gate": primary_population_gate,
        "temporal_label_qa": temporal_qa,
        # Cell-level rows for the manifest writer (main() pops this key
        # before writing burned_landcover_gate.json/.md/.csv -- never
        # persisted in the aggregate gate reports). Present only when
        # exclusion is enabled, so other experiments' gate dicts are
        # byte-for-byte unaffected by this key's mere existence.
        **({"pre_label_excluded_manifest_rows": pre_label_excluded_rows} if exclude_pre_label_burns else {}),
        # Gate is diagnostic + advisor-gated: passing NEVER auto-authorizes
        # predictor/Step7/Step8/Step9/Step10.
        "downstream_authorized": False,
        "burned_landcover_dominant_counts": burned_dominant_counts,
        "unburned_landcover_dominant_counts": unburned_dominant_counts,
        "burned_tree_cover_count": burned_tree_cover_count,
        "burned_shrubland_count": burned_shrubland_count,
        "burned_grassland_count": burned_grassland_count,
        "burned_cropland_count": burned_cropland_count,
        "burned_tree_shrub_grass_count": burned_tree_shrub_grass_count,
        "burned_tree_shrub_count": burned_tree_shrub_count,
        "burned_cropland_dominant_count": burned_cropland_dominant_count,
        "burned_natural_vegetation_fraction": burned_natural_vegetation_fraction,
        "burned_cropland_fraction": burned_cropland_fraction,
        "burned_tree_shrub_fraction": burned_tree_shrub_fraction,
        "burned_cells_without_valid_landcover": burned_cells_without_valid_landcover,
        "unburned_cells_without_valid_landcover": unburned_cells_without_valid_landcover,
        "decision": decision,
        "reason": reason,
        "warnings": warnings_list,
        "thresholds": {
            "min_positives": min_positives,
            "natural_threshold": natural_threshold,
            "cropland_threshold": cropland_threshold,
        },
        "inputs": {
            "label_path": str(label_path),
            "aligned_label_path": str(aligned_label_path),
            "reference_path": str(reference_path),
            "landcover_path": str(landcover_path),
            "label_start": label_start,
            "label_end": label_end,
            "predictor_start": predictor_start,
            "predictor_end": predictor_end,
            "pre_label_label_path": str(pre_label_label_path) if pre_label_label_path else None,
            "aligned_pre_label_path": str(aligned_pre_label_path) if aligned_pre_label_path else None,
        },
    }


# =============================================================================
# Output writers
# =============================================================================
def write_json(gate: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gate, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    log.info("JSON yazildi: %s", out_path)
    return out_path


def write_csv(gate: dict, out_path: Path) -> Path:
    scalar_fields = [
        "gate_level", "label_kind", "burn_month_available", "block_size_pixels",
        "total_valid_cells_or_pixels_considered",
        "analysis_universe_cells_after_exclusions",
        "exclude_pre_label_burns", "pre_label_burn_excluded_count",
        "exclude_historical_burns", "historical_burn_excluded_count",
        "pre_label_historical_overlap_count", "total_unique_excluded_count",
        "burned_count", "unburned_count",
        "burned_tree_cover_count", "burned_shrubland_count", "burned_grassland_count",
        "burned_cropland_count", "burned_tree_shrub_grass_count", "burned_tree_shrub_count",
        "burned_cropland_dominant_count", "burned_natural_vegetation_fraction",
        "burned_cropland_fraction", "burned_tree_shrub_fraction",
        "burned_cells_without_valid_landcover", "unburned_cells_without_valid_landcover",
        "decision", "reason", "downstream_authorized",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for field in scalar_fields:
            writer.writerow([field, gate.get(field)])
        for cls_name, count in sorted(gate.get("burned_landcover_dominant_counts", {}).items()):
            writer.writerow([f"burned_landcover_dominant_counts__{cls_name}", count])
        for cls_name, count in sorted(gate.get("unburned_landcover_dominant_counts", {}).items()):
            writer.writerow([f"unburned_landcover_dominant_counts__{cls_name}", count])
        breakdown = gate.get("pre_label_burn_excluded_breakdown", {}) or {}
        for key in ["total", "tree_cover", "shrubland", "grassland", "tree_shrub",
                    "tree_shrub_grass", "cropland", "other", "without_valid_landcover"]:
            if key in breakdown:
                writer.writerow([f"pre_label_burn_excluded__{key}", breakdown[key]])
        hist_breakdown = gate.get("historical_burn_excluded_breakdown", {}) or {}
        for key in ["total", "tree_cover", "shrubland", "grassland", "tree_shrub",
                    "tree_shrub_grass", "cropland", "other", "without_valid_landcover"]:
            if key in hist_breakdown:
                writer.writerow([f"historical_burn_excluded__{key}", hist_breakdown[key]])
        primary_gate = gate.get("primary_population_sample_size_gate") or {}
        for key in ["population", "burned_count", "min_burned_required",
                    "decision", "stop_state"]:
            if key in primary_gate:
                writer.writerow([f"primary_population_sample_size_gate__{key}",
                                 primary_gate[key]])
        qa = gate.get("temporal_label_qa", {}) or {}
        for key in ["count_within_predictor_window", "count_within_label_window",
                    "count_between_predictor_end_and_label_start",
                    "count_missing_or_zero_burndate", "bordubet_window_burned_cell_count",
                    "min_nonzero_burndate_iso", "max_nonzero_burndate_iso"]:
            if key in qa:
                writer.writerow([f"temporal_qa__{key}", qa[key]])
    log.info("CSV yazildi: %s", out_path)
    return out_path


def write_markdown(gate: dict, out_path: Path) -> Path:
    lines = []
    lines.append("# Burned-Landcover Gate (Step6B)")
    lines.append("")
    lines.append(
        "Diagnostic gate: summarizes the landcover composition of MCD64A1-burned "
        f"~500 m cells (`gate_level = {gate['gate_level']}`). This does NOT train "
        "a model and does NOT stop the pipeline on a cropland-dominated result."
    )
    lines.append("")
    lines.append(f"**Decision: `{gate['decision']}`**")
    lines.append("")
    lines.append(f"Reason: {gate['reason']}")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key in [
        "total_valid_cells_or_pixels_considered", "burned_count", "unburned_count",
        "burned_tree_cover_count", "burned_shrubland_count", "burned_grassland_count",
        "burned_cropland_count", "burned_tree_shrub_grass_count", "burned_tree_shrub_count",
        "burned_cropland_dominant_count",
    ]:
        lines.append(f"| {key} | {gate[key]} |")
    lines.append("")
    lines.append("## Fractions (denominator = burned_count)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key in ["burned_natural_vegetation_fraction", "burned_cropland_fraction", "burned_tree_shrub_fraction"]:
        val = gate[key]
        lines.append(f"| {key} | {val:.4f} |" if val is not None else f"| {key} | n/a |")
    lines.append("")
    lines.append("## Burned cells: dominant landcover breakdown")
    lines.append("")
    lines.append("| Dominant class | Burned cell count |")
    lines.append("|---|---|")
    for cls_name, count in sorted(gate["burned_landcover_dominant_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cls_name} | {count} |")
    lines.append("")

    # --- Pre-label leakage-safe exclusion ---
    if gate.get("exclude_pre_label_burns"):
        b = gate.get("pre_label_burn_excluded_breakdown", {}) or {}
        lines.append("## Pre-label burn exclusion (leakage safety)")
        lines.append("")
        lines.append(
            "Cells whose MCD64A1 BurnDate maps to a calendar date BEFORE "
            f"`label_start` ({gate['inputs']['label_start']}) were removed from "
            "the ENTIRE analysis universe BEFORE composition was evaluated. "
            "They are NOT counted as burned, NOT counted as unburned, and NOT "
            "in any gate denominator. Rule: "
            f"`{gate.get('pre_label_burn_exclusion_rule')}`."
        )
        lines.append("")
        lines.append(f"- total AOI cells considered: `{gate['total_valid_cells_or_pixels_considered']}`")
        lines.append(f"- pre_label_burn_excluded_count: `{gate['pre_label_burn_excluded_count']}`")
        lines.append(f"- analysis universe after exclusions: `{gate['analysis_universe_cells_after_exclusions']}`")
        manifest_info = gate.get("pre_label_exclusion_manifest")
        if manifest_info:
            lines.append(
                f"- cell-level exclusion manifest: `{manifest_info['parquet_path']}` "
                f"({manifest_info['excluded_cell_count']} unique cell_id; Step8A reads this "
                "verbatim to remove the same cells from its own analysis universe)."
            )
        lines.append("")
        lines.append("| Excluded cells by dominant landcover | Count |")
        lines.append("|---|---|")
        for key in ["tree_cover", "shrubland", "grassland", "tree_shrub",
                    "tree_shrub_grass", "cropland", "other", "without_valid_landcover"]:
            lines.append(f"| {key} | {b.get(key, 0)} |")
        lines.append("")

        qa = gate.get("temporal_label_qa", {}) or {}
        lines.append("## Temporal BurnDate QA (cell-level)")
        lines.append("")
        lines.append(f"_{qa.get('cell_level_note', '')}_")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key in [
            "min_nonzero_burndate_iso", "max_nonzero_burndate_iso",
            "count_before_predictor_start", "count_within_predictor_window",
            "count_between_predictor_end_and_label_start", "count_within_label_window",
            "count_after_label_end", "count_missing_or_zero_burndate",
            "count_pre_label_burn_excluded", "bordubet_window_burned_cell_count",
        ]:
            lines.append(f"| {key} | {qa.get(key)} |")
        lines.append("")
        if qa.get("bordubet_window"):
            lines.append(
                f"Bördübet predictor-window fire check "
                f"({qa['bordubet_window'][0]} .. {qa['bordubet_window'][1]}): "
                f"`{qa.get('bordubet_window_burned_cell_count', 0)}` cell(s) "
                "with BurnDate in that window intersect the AOI "
                "(all gate cells are within-AOI by construction). A zero here "
                "is reported honestly and may indicate an AOI/product-resolution issue."
            )
            lines.append("")
        lines.append(f"_{qa.get('out_of_span_note', '')}_")
        lines.append("")

    # --- Historical burn exclusion (SEPARATE axis) + union arithmetic ---
    if gate.get("exclude_historical_burns"):
        hb = gate.get("historical_burn_excluded_breakdown", {}) or {}
        cfg = gate.get("historical_burn_exclusion_config", {}) or {}
        lines.append("## Historical burn exclusion (earlier event, same geography)")
        lines.append("")
        lines.append(
            "Cells that burned in an EARLIER event over the SAME AOI were "
            "removed from the analysis universe because their current-year "
            "NDVI/LST/TVDI may reflect POST-FIRE RECOVERY rather than a normal "
            "vegetation state. This is an axis SEPARATE from the pre-label "
            "exclusion above: the two flags are derived independently, a cell "
            "may carry both, and the gate excludes their UNION."
        )
        lines.append("")
        lines.append(f"- source experiment: `{cfg.get('source_experiment_id')}`")
        lines.append(f"- source kind: `{cfg.get('source_kind')}`")
        lines.append(f"- mask definition: {gate.get('historical_burn_exclusion_rule')}")
        lines.append(f"- historical_burn_excluded_count: `{gate.get('historical_burn_excluded_count')}`")
        lines.append("")
        lines.append("| Excluded cells by dominant landcover | Count |")
        lines.append("|---|---|")
        for key in ["tree_cover", "shrubland", "grassland", "tree_shrub",
                    "tree_shrub_grass", "cropland", "other", "without_valid_landcover"]:
            lines.append(f"| {key} | {hb.get(key, 0)} |")
        lines.append("")
        lines.append("## Exclusion union")
        lines.append("")
        lines.append(f"_{gate.get('exclusion_union_rule')}_")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key in ["pre_label_burn_excluded_count", "historical_burn_excluded_count",
                    "pre_label_historical_overlap_count", "total_unique_excluded_count",
                    "analysis_universe_cells_after_exclusions", "burned_count",
                    "unburned_count", "total_valid_cells_or_pixels_considered"]:
            lines.append(f"| {key} | {gate.get(key)} |")
        lines.append("")

    # --- SEPARATE primary-population sample-size rule ---
    primary_gate = gate.get("primary_population_sample_size_gate")
    if primary_gate:
        lines.append("## Primary-population sample-size rule (SEPARATE from the composition gate)")
        lines.append("")
        lines.append(
            "This is NOT the burned-landcover composition gate and NOT its "
            f"generic total-burned feasibility threshold "
            f"(`min_positives = {gate.get('thresholds', {}).get('min_positives')}`, "
            "which applies to `burned_count` across every landcover and is "
            "unchanged). It is a separate, per-experiment sample-size rule "
            "evaluated after all exclusions and landcover classification, on "
            "the primary population only."
        )
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for key in ["population", "burned_count", "min_burned_required", "decision", "stop_state"]:
            lines.append(f"| {key} | {primary_gate.get(key)} |")
        lines.append("")
        lines.append(f"Reason: {primary_gate.get('reason')}")
        lines.append("")
        if primary_gate.get("decision") == PRIMARY_POPULATION_GATE_STOP:
            lines.append(
                f"**STOP: `{primary_gate.get('stop_state')}`** -- downstream is "
                "NOT authorized. This is a primary-population sample-size stop; "
                "it must NOT be reinterpreted as a natural/cropland composition "
                "gate failure."
            )
            lines.append("")

    lines.append(f"**downstream_authorized: `{gate.get('downstream_authorized')}`** "
                 "(passing the gate does NOT authorize predictor/Step7/Step8/"
                 "Step9/Step10; advisor review required).")
    lines.append("")

    if gate["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for w in gate["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Thresholds used")
    lines.append("")
    for k, v in gate["thresholds"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for k, v in gate["inputs"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown yazildi: %s", out_path)
    return out_path


# =============================================================================
# Cell-level pre-label exclusion manifest (the actual bug fix: Step8A must
# know WHICH cells the gate excluded, not just how many)
# =============================================================================
_MANIFEST_COLUMNS = [
    "experiment_id", "cell_id", "row_500m", "col_500m",
    "pre_label_burn_date", "exclusion_reason",
    "landcover_dominant", "burnable_tree_shrub", "burnable_tree_shrub_grass",
]


def write_pre_label_exclusion_manifest(
    rows: list[dict],
    experiment_id: str,
    output_dir: Path,
    exclusion_rule: str,
    predictor_start: str | None,
    label_start: str,
    source_raster: Path,
    gate_excluded_count: int,
) -> dict:
    """
    Writes the canonical CELL-LEVEL pre-label exclusion manifest (parquet +
    CSV + metadata JSON) Step8A reads to remove the SAME physical cells from
    its own analysis universe. This is the actual fix for the "gate excludes
    N cells in aggregate but Step8A doesn't know which N" bug -- Step8A must
    NEVER reimplement/re-derive this exclusion set independently; it reads
    this manifest verbatim (see src.step8a_prepare_500m_modeling_dataset
    .read_pre_label_exclusion_manifest()).

    Fail-fast integrity checks (never silently write an ambiguous manifest):
        - cell_id must be non-null and unique
        - (row_500m, col_500m) pairs must be unique (same cardinality as cell_id)
        - unique cell_id count MUST equal the gate's own aggregate
          pre_label_burn_excluded_count (single source of truth check)
    """
    manifest_df = pd.DataFrame(rows, columns=_MANIFEST_COLUMNS)
    manifest_df["experiment_id"] = experiment_id

    if not manifest_df.empty:
        if manifest_df["cell_id"].isna().any():
            raise Step6BError(
                "Pre-label exclusion manifest: cell_id null deger iceriyor."
            )
        if not manifest_df["cell_id"].is_unique:
            dupes = manifest_df.loc[manifest_df["cell_id"].duplicated(), "cell_id"].tolist()
            raise Step6BError(
                f"Pre-label exclusion manifest: tekrarlanan cell_id degerleri: {dupes}."
            )
        if manifest_df[["row_500m", "col_500m"]].drop_duplicates().shape[0] != len(manifest_df):
            raise Step6BError(
                "Pre-label exclusion manifest: (row_500m, col_500m) ciftleri "
                "cell_id ile ayni sayida degil -- hucre-kimligi semasi tutarsiz."
            )

    unique_count = int(manifest_df["cell_id"].nunique())
    if unique_count != gate_excluded_count:
        raise Step6BError(
            "Pre-label exclusion manifest sayisi gate aggregate sonucuyla "
            f"UYUSMUYOR: gate pre_label_burn_excluded_count={gate_excluded_count}, "
            f"manifest unique cell_id={unique_count}. Bunlar BIREBIR ayni olmali."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / PRE_LABEL_EXCLUSION_MANIFEST_PARQUET
    csv_path = output_dir / PRE_LABEL_EXCLUSION_MANIFEST_CSV
    metadata_path = output_dir / PRE_LABEL_EXCLUSION_MANIFEST_METADATA

    manifest_df.to_parquet(parquet_path, index=False)
    manifest_df.to_csv(csv_path, index=False)

    metadata = {
        "experiment_id": experiment_id,
        "exclude_pre_label_burns": True,
        "exclusion_rule": exclusion_rule,
        "predictor_start": predictor_start,
        "label_start": label_start,
        "excluded_cell_count": unique_count,
        "cell_id_scheme": (
            "r{row_500m}_c{col_500m}, native ~500m MCD64A1-grid block index "
            "(src.step8a_prepare_500m_modeling_dataset.compute_cell_identity) "
            "-- SAME scheme Step8A uses; never independently reimplemented."
        ),
        "source_raster": str(source_raster),
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        "Pre-label exclusion manifest yazildi: %s (%d unique cell_id; parquet+csv+metadata).",
        parquet_path, unique_count,
    )
    return {
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        "excluded_cell_count": unique_count,
    }


# =============================================================================
# Orchestration
# =============================================================================
def main(
    output_dir_arg: str = STEP6_LABEL_OUTPUT_DIR,
    force: bool = False,
    label_raster_arg: str | None = None,
    reference_30m_arg: str | None = None,
    landcover_path_arg: str | None = None,
    label_start: str = LABEL_START_DATE,
    label_end: str = LABEL_END_DATE,
    min_positives: int = STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES,
    natural_threshold: float = STEP6_BURNED_LANDCOVER_GATE_NATURAL_THRESHOLD,
    cropland_threshold: float = STEP6_BURNED_LANDCOVER_GATE_CROPLAND_THRESHOLD,
    exclude_pre_label_burns: bool = False,
    pre_label_raster_arg: str | None = None,
    predictor_start: str | None = None,
    predictor_end: str | None = None,
    bordubet_check_window: tuple[str, str] | None = None,
    experiment_id: str | None = None,
    exclude_historical_burns: bool = False,
    historical_exclusion_manifest_arg: str | None = None,
    historical_exclusion_config: dict | None = None,
    primary_population: str | None = None,
    min_primary_population_burned: int | None = None,
) -> dict:
    """
    Path-aware davranis (Step0C):
        Hicbir path argumani verilmezse legacy Kozan davranisi BIREBIR
        korunur (outputs/validation/labels + outputs/step5 referansi +
        data/landcover kaynagi).
        Acik path'ler verilirse (or. Manavgat namespaced dosyalari) yalnizca
        onlar kullanilir -- legacy Kozan dosyalarina NE OKUMA NE YAZMA
        yapilir. landcover_path_arg verilirse referans gridle birebir
        uyusmasi zorunludur; uyusmuyorsa kaynak kabul edilip nearest-neighbor
        ile output_dir altina hizalanir (kategorik veri, ASLA bilinear degil;
        Step8A'nin prepare_aligned_landcover'i reuse edilir).
    """
    log.info("=" * 60)
    log.info("STEP 6B BASLIYOR (burned-landcover diagnostic gate)")
    log.info("=" * 60)

    out_dir = Path(output_dir_arg)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    required_outputs = [
        out_dir / "burned_landcover_gate.json",
        out_dir / "burned_landcover_gate.md",
        out_dir / "burned_landcover_gate.csv",
    ]
    if any(p.exists() for p in required_outputs) and not force:
        present = [p.name for p in required_outputs if p.exists()]
        raise Step6BError(
            "Step6B ciktilari zaten var (" + ", ".join(present)
            + "). Uzerine yazmak icin --force verin."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_path = resolve_reference_30m(reference_30m_arg)

    # Step6B needs GENUINE raw BurnDate (day-of-year), not a binary fallback --
    # otherwise the burned/unburned-in-window determination is not honest.
    label_path, label_kind = resolve_label_raster(label_raster_arg)
    if label_kind != LABEL_KIND_RAW:
        raise Step6BError(
            f"Step6B icin gercek raw BurnDate rasteri gerekli, ama bulunan "
            f"etiket dosyasi binary gorunuyor (label_kind={label_kind!r}, "
            f"path={label_path}). Once Step6'nin canonical export'unu "
            "calistirin: src.step6_validate_fire_relation.export_raw_mcd64a1_labels() "
            "(veya: python scripts/export_mcd64a1_raw_burndate.py)."
        )

    # Fail-fast: raises if the "raw" raster actually looks binary or has no
    # values inside the label DOY window (same check Step8A relies on).
    try:
        label_diag = inspect_label_raster(label_path, label_kind, label_start, label_end)
    except Step8AError as exc:
        raise Step6BError(str(exc)) from exc
    log.info(
        "Label raster diagnostics: kind=%s min=%s max=%s count_one=%s "
        "count_gt_one=%s in_DOY_range=%s",
        label_diag.get("label_kind"), label_diag.get("min"), label_diag.get("max"),
        label_diag.get("count_one"), label_diag.get("count_gt_one"),
        label_diag.get("count_in_label_doy_range"),
    )

    if landcover_path_arg:
        landcover_path, landcover_info = _resolve_explicit_landcover(
            landcover_path_arg, reference_path, out_dir
        )
    else:
        landcover_path, landcover_info = resolve_landcover(reference_path)
    log.info("Landcover rasteri (hizali): %s", landcover_path)

    pre_label_path = None
    if exclude_pre_label_burns:
        if not pre_label_raster_arg:
            raise Step6BError(
                "exclude_pre_label_burns=True verildi ama --pre-label-raster "
                "(pre_label_raster_arg) yok. [predictor_start, label_start) "
                "penceresi uzerinden AYRI bir MCD64A1 BurnDate rasteri gerekli."
            )
        pre_label_path = Path(pre_label_raster_arg)
        if not pre_label_path.is_absolute():
            pre_label_path = BASE_DIR / pre_label_path
        if not pre_label_path.exists():
            raise Step6BError(f"Pre-label BurnDate rasteri bulunamadi: {pre_label_path}")
        if not experiment_id:
            raise Step6BError(
                "exclude_pre_label_burns=True verildi ama experiment_id yok. "
                "Cell-level pre-label exclusion manifest'i deney kimligiyle "
                "damgalamak icin experiment_id ZORUNLUDUR."
            )
        log.info("Pre-label exclusion AKTIF; pre-label raster: %s", pre_label_path)

    # --- Historical burn exclusion (SEPARATE axis; opt-in, config-driven) ---
    # The manifest is BUILT and provenance-validated by the caller
    # (scripts/run_label_gate_only.py) BEFORE the gate runs; here it is only
    # read back through its own dedicated reader, which keeps its own checks.
    historical_excluded_cell_ids = None
    historical_manifest_path = None
    if exclude_historical_burns:
        from src.historical_burn_exclusion import (
            HISTORICAL_BURN_EXCLUSION_MANIFEST_PARQUET,
            HistoricalBurnExclusionError,
            read_historical_burn_exclusion_manifest,
        )

        if not experiment_id:
            raise Step6BError(
                "exclude_historical_burns=True verildi ama experiment_id yok. "
                "Historical exclusion manifest'i deney kimligiyle dogrulamak "
                "icin experiment_id ZORUNLUDUR."
            )
        historical_manifest_path = Path(
            historical_exclusion_manifest_arg
            or (out_dir / HISTORICAL_BURN_EXCLUSION_MANIFEST_PARQUET)
        )
        if not historical_manifest_path.is_absolute():
            historical_manifest_path = BASE_DIR / historical_manifest_path
        try:
            historical_excluded_cell_ids = read_historical_burn_exclusion_manifest(
                historical_manifest_path, experiment_id=experiment_id,
            )
        except HistoricalBurnExclusionError as exc:
            raise Step6BError(str(exc)) from exc
        log.info(
            "Historical exclusion AKTIF [%s]: %d hucre (manifest: %s) analiz "
            "evreninden dislanacak.", experiment_id,
            len(historical_excluded_cell_ids), historical_manifest_path,
        )

    gate = compute_gate(
        label_path=label_path,
        label_kind=label_kind,
        reference_path=reference_path,
        landcover_path=landcover_path,
        label_start=label_start,
        label_end=label_end,
        output_dir=out_dir,
        min_positives=min_positives,
        natural_threshold=natural_threshold,
        cropland_threshold=cropland_threshold,
        exclude_pre_label_burns=exclude_pre_label_burns,
        pre_label_label_path=pre_label_path,
        predictor_start=predictor_start,
        predictor_end=predictor_end,
        bordubet_check_window=bordubet_check_window,
        experiment_id=experiment_id,
        exclude_historical_burns=exclude_historical_burns,
        historical_excluded_cell_ids=historical_excluded_cell_ids,
        historical_exclusion_config=historical_exclusion_config,
        primary_population=primary_population,
        min_primary_population_burned=min_primary_population_burned,
    )
    gate["label_raster_diagnostics"] = label_diag
    gate["landcover_info"] = landcover_info
    if exclude_historical_burns:
        gate["historical_burn_exclusion_manifest"] = {
            "parquet_path": str(historical_manifest_path),
            "excluded_cell_count": len(historical_excluded_cell_ids or ()),
        }

    # Manifest rows never go into the aggregate JSON/MD/CSV -- popped here,
    # written to their own dedicated files below.
    manifest_rows = gate.pop("pre_label_excluded_manifest_rows", None)

    manifest_result = None
    if exclude_pre_label_burns:
        manifest_result = write_pre_label_exclusion_manifest(
            rows=manifest_rows or [],
            experiment_id=experiment_id,
            output_dir=out_dir,
            exclusion_rule=gate.get("pre_label_burn_exclusion_rule", "not_applied"),
            predictor_start=predictor_start,
            label_start=label_start,
            source_raster=pre_label_path,
            gate_excluded_count=gate["pre_label_burn_excluded_count"],
        )
        # Only added to the gate dict (and therefore burned_landcover_gate.json)
        # for exclusion-enabled runs -- other experiments' gate JSON is
        # byte-for-byte unaffected by this feature.
        gate["pre_label_exclusion_manifest"] = manifest_result

    json_path = write_json(gate, out_dir / "burned_landcover_gate.json")
    md_path = write_markdown(gate, out_dir / "burned_landcover_gate.md")
    csv_path = write_csv(gate, out_dir / "burned_landcover_gate.csv")

    log.info(
        "Karar: %s (burned_count=%d, natural_fraction=%s, cropland_fraction=%s)",
        gate["decision"], gate["burned_count"],
        gate["burned_natural_vegetation_fraction"], gate["burned_cropland_fraction"],
    )
    primary_gate = gate.get("primary_population_sample_size_gate")
    if primary_gate:
        log.info(
            "Primary-population sample-size rule (SEPARATE from the composition "
            "gate): population=%s burned_count=%s min_burned_required=%s -> %s",
            primary_gate["population"], primary_gate["burned_count"],
            primary_gate["min_burned_required"], primary_gate["decision"],
        )
        if primary_gate["decision"] == PRIMARY_POPULATION_GATE_STOP:
            log.warning(
                "STOP: %s -- downstream NOT authorized. This is a "
                "primary-population sample-size stop, NOT a natural/cropland "
                "composition gate failure.", primary_gate["stop_state"],
            )

    log.info("=" * 60)
    log.info("STEP 6B TAMAMLANDI (diagnostic only; pipeline durdurulmadi)")
    log.info("=" * 60)

    return {
        "decision": gate["decision"],
        "json_path": str(json_path),
        "md_path": str(md_path),
        "csv_path": str(csv_path),
        "burned_count": gate["burned_count"],
        "manifest_result": manifest_result,
        # Exclusion accounting (uniform schema; zero/False when unused).
        "pre_label_burn_excluded_count": gate.get("pre_label_burn_excluded_count", 0),
        "historical_burn_excluded_count": gate.get("historical_burn_excluded_count", 0),
        "pre_label_historical_overlap_count": gate.get("pre_label_historical_overlap_count", 0),
        "total_unique_excluded_count": gate.get("total_unique_excluded_count", 0),
        "analysis_universe_cells_after_exclusions": gate.get(
            "analysis_universe_cells_after_exclusions"
        ),
        "historical_burn_exclusion_manifest": gate.get("historical_burn_exclusion_manifest"),
        # SEPARATE post-gate sample-size rule; None when undeclared.
        "primary_population_sample_size_gate": primary_gate,
        "downstream_authorized": False,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step6B: burned-landcover diagnostic gate. Summarizes the "
        "landcover composition of MCD64A1-burned ~500 m cells (same "
        "reconstruction level as Step8A) and classifies the AOI as a "
        "wildfire candidate, a cropland/anız control, or insufficient/mixed."
    )
    parser.add_argument("--output-dir", type=str, default=STEP6_LABEL_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--label-raster", "--label-path", dest="label_raster", type=str, default=None,
        help="Raw MCD64A1 BurnDate raster yolu (verilmezse legacy Kozan kesfi).",
    )
    parser.add_argument(
        "--reference-30m", "--reference-path", dest="reference_30m", type=str, default=None,
        help="30 m referans grid yolu (verilmezse legacy outputs/step5 dosyasi).",
    )
    parser.add_argument(
        "--landcover-path", type=str, default=None,
        help="Landcover raster yolu (verilmezse legacy data/landcover kesfi). "
        "Referans gridle uyusmuyorsa nearest-neighbor ile output-dir altina hizalanir.",
    )
    parser.add_argument("--label-start", type=str, default=LABEL_START_DATE)
    parser.add_argument("--label-end", type=str, default=LABEL_END_DATE)
    parser.add_argument("--min-positives", type=int, default=STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES)
    parser.add_argument("--natural-threshold", type=float, default=STEP6_BURNED_LANDCOVER_GATE_NATURAL_THRESHOLD)
    parser.add_argument("--cropland-threshold", type=float, default=STEP6_BURNED_LANDCOVER_GATE_CROPLAND_THRESHOLD)
    # --- Leakage-safe pre-label burn exclusion (opt-in; Muğla 2021) ---
    parser.add_argument(
        "--exclude-pre-label-burns", action="store_true",
        help="Cells that burned BEFORE label_start are excluded from the whole "
        "analysis universe (requires --pre-label-raster).",
    )
    parser.add_argument(
        "--pre-label-raster", type=str, default=None,
        help="AYRI MCD64A1 BurnDate rasteri, [predictor_start, label_start) "
        "penceresi uzerinden export edilmis (pre-label burn'leri belirlemek icin).",
    )
    parser.add_argument("--predictor-start", type=str, default=None,
                        help="Predictor penceresi baslangici (temporal QA icin).")
    parser.add_argument("--predictor-end", type=str, default=None,
                        help="Predictor penceresi bitisi (temporal QA icin).")
    parser.add_argument("--bordubet-start", type=str, default=None,
                        help="Bördübet predictor-window fire check baslangici (or. 2021-06-21).")
    parser.add_argument("--bordubet-end", type=str, default=None,
                        help="Bördübet predictor-window fire check bitisi (or. 2021-06-25).")
    parser.add_argument(
        "--experiment-id", type=str, default=None,
        help="Deney kimligi (--exclude-pre-label-burns / "
        "--exclude-historical-burns ile ZORUNLU; cell-level exclusion "
        "manifest'lerini damgalamak/dogrulamak icin kullanilir).",
    )
    # --- Historical burn exclusion (SEPARATE axis; opt-in) ---
    parser.add_argument(
        "--exclude-historical-burns", action="store_true",
        help="Onceki bir olayda (ayni AOI) yanan hucreleri de analiz "
        "evreninden disla. Gate'ten ONCE uretilmis historical exclusion "
        "manifest'i gerekir.",
    )
    parser.add_argument(
        "--historical-exclusion-manifest", type=str, default=None,
        help="historical_burn_excluded_cells.parquet yolu (verilmezse "
        "--output-dir altindaki kanonik ad kullanilir).",
    )
    # --- SEPARATE primary-population sample-size rule ---
    parser.add_argument(
        "--primary-population", type=str, default=None,
        help="Sample-size kuralinin uygulanacagi birincil populasyon "
        "(or. burnable_tree_shrub_grass). --min-primary-population-burned "
        "ile BIRLIKTE verilmelidir.",
    )
    parser.add_argument(
        "--min-primary-population-burned", type=int, default=None,
        help="Birincil populasyondaki minimum burned hucre sayisi. Bu, "
        "jenerik --min-positives (toplam burned) esiginden AYRIDIR ve onu "
        "DEGISTIRMEZ.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir_arg=args.output_dir,
        force=args.force,
        label_raster_arg=args.label_raster,
        reference_30m_arg=args.reference_30m,
        landcover_path_arg=args.landcover_path,
        label_start=args.label_start,
        label_end=args.label_end,
        min_positives=args.min_positives,
        natural_threshold=args.natural_threshold,
        cropland_threshold=args.cropland_threshold,
        exclude_pre_label_burns=args.exclude_pre_label_burns,
        pre_label_raster_arg=args.pre_label_raster,
        predictor_start=args.predictor_start,
        predictor_end=args.predictor_end,
        bordubet_check_window=(
            (args.bordubet_start, args.bordubet_end)
            if args.bordubet_start and args.bordubet_end else None
        ),
        experiment_id=args.experiment_id,
        exclude_historical_burns=args.exclude_historical_burns,
        historical_exclusion_manifest_arg=args.historical_exclusion_manifest,
        primary_population=args.primary_population,
        min_primary_population_burned=args.min_primary_population_burned,
    )