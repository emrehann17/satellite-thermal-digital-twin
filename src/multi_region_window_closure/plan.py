"""Static/temporal feature contract, export plan and download-request accounting.

Nothing in this module contacts Earth Engine. It plans only.

Two facts about this pipeline drive the whole module and are easy to get wrong:

1. Exports are **synchronous `getPixels` downloads**, not Earth Engine batch
   Drive tasks. `earth_engine_batch_task_count` is therefore 0, and what is
   counted here are download REQUESTS. Content provenance is per-raster
   `sha256`, not a task ID.
2. The tiled fallback escalates through
   `scripts/run_predictors_only._TILE_GRID_ESCALATION = [(2,2),(4,4),(6,6),(8,8)]`,
   trying each grid in turn and raising only after all four fail. The hard
   ceiling is therefore 8x8, not 4x4.

Design reference: docs/multi_region_window_closure_design/EXPORT_FEASIBILITY.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS,
    CANONICAL_VARIANT_ID,
    MultiRegionWindowClosureError,
    SHIFTED_VARIANTS,
    VARIANTS,
)

# =============================================================================
# Static vs temporal -- derived, never decided by feature name
# =============================================================================
#: Step8A predictor families whose canonical source lives in a stage this
#: analysis rebuilds. Mirrors
#: `window_closure_sensitivity.TIMING_DERIVED_SOURCE_DIRS`.
TEMPORAL_FAMILIES: tuple[str, ...] = (
    "current_lst", "current_tvdi", "downscaled_lst", "fused_lst",
    "lst_anomaly", "ndvi", "tvdi_difference",
)
#: Families whose canonical source is a frozen static input.
STATIC_FAMILIES: tuple[str, ...] = ("elevation", "slope")

#: Window-independent inputs that are reused read-only and never re-exported.
#: `landcover_dominant` is not in Step8A `predictor_paths` (it comes from the
#: aligned gate input), so it is listed explicitly rather than inferred.
STATIC_REUSE_ROLES: tuple[str, ...] = (
    "dem_elevation", "dem_slope", "landcover_aligned", "aoi_geometry",
    "reference_grid", "label_window", "label_raster",
    "model_feature_registry", "model_hyperparameters", "random_seed",
    "spatial_block_definition",
)

CLASSIFICATION_STATIC = "static"
CLASSIFICATION_TEMPORAL = "temporal"


def feature_classification(
    aoi: str, experiments_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Static/temporal classification for one AOI, DERIVED from Step8A lineage.

    Delegates to `window_closure_sensitivity.step8a_predictor_lineage`, which
    reads the frozen canonical `step8a_dataset_stats.json:predictor_paths` and
    raises on an unrecognised source directory. Nothing is guessed from names.
    """
    from src.window_closure_sensitivity import step8a_predictor_lineage

    lineage = step8a_predictor_lineage(aoi, experiments_root)
    rows: list[dict[str, Any]] = []
    for name, record in sorted(lineage["lineage"].items()):
        is_temporal = record["classification"] == "timing_derived"
        rows.append({
            "aoi": aoi,
            "feature": name,
            "family": name,
            "static_or_temporal": (
                CLASSIFICATION_TEMPORAL if is_temporal else CLASSIFICATION_STATIC
            ),
            "window_dependent": is_temporal,
            "source_product": record["canonical_source"],
            "canonical_source": record["canonical_source"],
            "reuse_or_recompute": "recompute" if is_temporal else "reuse",
            "export_required": is_temporal,
        })
    observed_temporal = tuple(sorted(lineage["timing_derived_predictors"]))
    observed_static = tuple(sorted(lineage["static_predictors"]))
    if observed_temporal != tuple(sorted(TEMPORAL_FAMILIES)):
        raise MultiRegionWindowClosureError(
            f"Step8A timing-derived families for '{aoi}' are {list(observed_temporal)}, "
            f"expected {list(sorted(TEMPORAL_FAMILIES))}. The static/temporal "
            "contract must be re-derived, never patched."
        )
    if observed_static != tuple(sorted(STATIC_FAMILIES)):
        raise MultiRegionWindowClosureError(
            f"Step8A static families for '{aoi}' are {list(observed_static)}, "
            f"expected {list(sorted(STATIC_FAMILIES))}."
        )
    return {
        "aoi": aoi,
        "rows": rows,
        "temporal_families": observed_temporal,
        "static_families": observed_static,
        "source": lineage["source"],
    }


# =============================================================================
# Raster artefact accounting
# =============================================================================
def expected_rasters_per_shifted_variant(baseline_years: Sequence[int]) -> int:
    """`(2 current + 2 per baseline year) x 2 products + 3 MODIS`.

    Delegates to the frozen per-AOI helper so the arithmetic cannot drift.
    """
    from src.window_closure_sensitivity import expected_raster_count

    return int(expected_raster_count(baseline_years))


def baseline_years_for(aoi: str) -> list[int]:
    from core.experiment_context import build_experiment_context

    return [int(y) for y in build_experiment_context(aoi)["baseline_years"]]


# =============================================================================
# Download-request accounting
# =============================================================================
def tile_grid_escalation() -> list[tuple[int, int]]:
    """The production escalation ladder, imported not re-declared."""
    from scripts.run_predictors_only import _TILE_GRID_ESCALATION

    return [tuple(int(v) for v in grid) for grid in _TILE_GRID_ESCALATION]


def hard_ceiling_requests_per_artifact() -> int:
    """Worst case the code permits for ONE artefact.

    One direct attempt, then every rung of the ladder in turn:
    `1 + 4 + 16 + 36 + 64 = 121`.
    """
    return 1 + sum(rows * cols for rows, cols in tile_grid_escalation())


def planning_upper_bound_requests_per_artifact() -> int:
    """Budgeting figure: a failed 2x2 followed by a successful 4x4 (`4 + 16 = 20`).

    This is NOT the maximum -- see `hard_ceiling_requests_per_artifact`.
    """
    ladder = tile_grid_escalation()
    if len(ladder) < 2:
        raise MultiRegionWindowClosureError(
            "Tile escalation ladder is too short to derive a planning bound."
        )
    return sum(rows * cols for rows, cols in ladder[:2])


#: Measured Manavgat profile: of 20 Landsat artefacts, 14 tiled at 2x2 and 6
#: downloaded directly; the 3 MODIS artefacts downloaded directly. Scaled to
#: other AOIs by Step8A row count, which is a measured artefact property.
#: Source: docs/multi_region_window_closure_design/EXPORT_FEASIBILITY.md 4.1-4.3.
EXPECTED_REQUESTS_PER_SHIFTED_VARIANT: dict[str, int] = {
    "bejis_2022": 50,
    "mugla_2021": 260,
    "evia_2021_extended": 65,
}
#: Pre-label BurnDate raster exports at 30 m (VALIDATION_LABEL_EXPORT_SCALE),
#: through the SAME tiled exporter -- verified, not assumed. Expected to tile
#: at 2x2.
EXPECTED_REQUESTS_PER_PRELABEL_RASTER = 4


def request_accounting(
    aois: Sequence[str] = ACTUAL_AOIS,
) -> dict[str, Any]:
    """Five distinct request-count fields plus the batch-task count.

    `earth_engine_batch_task_count` is reported explicitly as 0 so it can never
    be re-read as "tasks we forgot to count".
    """
    per_artifact_ceiling = hard_ceiling_requests_per_artifact()
    per_artifact_planning = planning_upper_bound_requests_per_artifact()

    predictor_rasters = 0
    expected_predictor_requests = 0
    per_aoi: dict[str, Any] = {}
    for aoi in aois:
        years = baseline_years_for(aoi)
        rasters_per_variant = expected_rasters_per_shifted_variant(years)
        aoi_rasters = rasters_per_variant * len(SHIFTED_VARIANTS)
        expected_per_variant = EXPECTED_REQUESTS_PER_SHIFTED_VARIANT.get(aoi)
        if expected_per_variant is None:
            raise MultiRegionWindowClosureError(
                f"No expected-request profile is registered for '{aoi}'."
            )
        aoi_expected = expected_per_variant * len(SHIFTED_VARIANTS)
        predictor_rasters += aoi_rasters
        expected_predictor_requests += aoi_expected
        per_aoi[aoi] = {
            "baseline_years": years,
            "rasters_per_shifted_variant": rasters_per_variant,
            "shifted_variants": len(SHIFTED_VARIANTS),
            "predictor_rasters": aoi_rasters,
            "expected_requests_per_shifted_variant": expected_per_variant,
            "expected_predictor_requests": aoi_expected,
            "prelabel_rasters": 1,
        }

    prelabel_rasters = len(aois)
    logical_rasters = predictor_rasters + prelabel_rasters
    expected_requests = (
        expected_predictor_requests
        + prelabel_rasters * EXPECTED_REQUESTS_PER_PRELABEL_RASTER
    )
    return {
        "earth_engine_batch_task_count": 0,
        "transport": "synchronous_getpixels_download",
        "logical_raster_artifact_count": logical_rasters,
        "predictor_raster_count": predictor_rasters,
        "prelabel_raster_count": prelabel_rasters,
        "minimum_download_request_count": logical_rasters,
        "expected_download_request_count": expected_requests,
        "planning_upper_bound_request_count": logical_rasters * per_artifact_planning,
        "hard_supported_request_ceiling": logical_rasters * per_artifact_ceiling,
        "tile_grid_escalation": [list(g) for g in tile_grid_escalation()],
        "requests_per_artifact_planning_upper_bound": per_artifact_planning,
        "requests_per_artifact_hard_ceiling": per_artifact_ceiling,
        "per_aoi": per_aoi,
        "note": (
            "No Earth Engine batch task is created on any code path this "
            "analysis reaches. Counts are synchronous download requests."
        ),
    }


# =============================================================================
# Export plan rows
# =============================================================================
EXPORT_PLAN_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "variant", "artifact_id", "role", "family",
    "static_or_temporal", "window_dependent", "reuse_or_recompute",
    "export_required", "grid_family", "export_scale_m", "start_date",
    "end_date", "expected_band_count", "is_count_product", "output_path",
    "producer", "estimated_request_count", "transport", "reason",
)


def export_plan_rows(
    aois: Sequence[str] = ACTUAL_AOIS,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Every artefact that will be produced or reused, and why.

    Pure planning: `predictor_artifact_jobs` is a pure function with no Earth
    Engine and no filesystem access. The canonical variant is never given an
    export job -- that helper RAISES for it, which is asserted here.
    """
    from src.window_closure_sensitivity import (
        nonzero_variants, predictor_artifact_jobs,
    )
    from src.multi_region_window_closure.dates import variant_windows_for

    rows: list[dict[str, Any]] = []
    for aoi in aois:
        years = baseline_years_for(aoi)
        variants = variant_windows_for(aoi)
        canonical = next(v for v in variants if v["is_canonical"])
        window_days = canonical["duration_days"]

        # --- canonical arm: reuse only, never export -----------------------
        for job in predictor_artifact_jobs(
            aoi, nonzero_variants(variants)[0], years, window_days, output_root,
        ):
            rows.append({
                "aoi": aoi,
                "variant": CANONICAL_VARIANT_ID,
                "artifact_id": job["artifact_id"],
                "role": job["role"],
                "family": job["family"],
                "static_or_temporal": CLASSIFICATION_TEMPORAL,
                "window_dependent": True,
                "reuse_or_recompute": "reuse",
                "export_required": False,
                "grid_family": job["grid_family"],
                "export_scale_m": job["export_scale_m"],
                "start_date": canonical["predictor_start_date"],
                "end_date": canonical["predictor_end_date"],
                "expected_band_count": job["expected_band_count"],
                "is_count_product": job["is_count_product"],
                "output_path": "",
                "producer": job["producer"],
                "estimated_request_count": 0,
                "transport": "reuse",
                "reason": (
                    "Canonical arm reads the frozen production Step8A outputs; "
                    "predictor_artifact_jobs refuses to plan an export for it."
                ),
            })

        # --- shifted arms: real export jobs --------------------------------
        for variant in nonzero_variants(variants):
            for job in predictor_artifact_jobs(
                aoi, variant, years, window_days, output_root,
            ):
                rows.append({
                    "aoi": aoi,
                    "variant": variant["variant_id"],
                    "artifact_id": job["artifact_id"],
                    "role": job["role"],
                    "family": job["family"],
                    "static_or_temporal": CLASSIFICATION_TEMPORAL,
                    "window_dependent": True,
                    "reuse_or_recompute": "recompute",
                    "export_required": True,
                    "grid_family": job["grid_family"],
                    "export_scale_m": job["export_scale_m"],
                    "start_date": job["start_date"],
                    "end_date": job["end_date"],
                    "expected_band_count": job["expected_band_count"],
                    "is_count_product": job["is_count_product"],
                    "output_path": job["output_path"],
                    "producer": job["producer"],
                    "estimated_request_count": 1,
                    "transport": "planned_direct_or_tiled",
                    "reason": "Predictor-window-dependent artefact; window moved.",
                })

        # --- pre-label censor raster ---------------------------------------
        censor_plan = _prelabel_plan(aoi, output_root)
        rows.append({
            "aoi": aoi,
            "variant": "shared",
            "artifact_id": "prelabel_burndate",
            "role": "prelabel_burndate",
            "family": "label",
            "static_or_temporal": CLASSIFICATION_TEMPORAL,
            "window_dependent": True,
            "reuse_or_recompute": "recompute",
            "export_required": True,
            "grid_family": "landsat_30m",
            "export_scale_m": _prelabel_scale(),
            "start_date": censor_plan["pre_label_start"],
            "end_date": censor_plan["pre_label_end"],
            "expected_band_count": 1,
            "is_count_product": False,
            "output_path": censor_plan["raster_path"],
            "producer": censor_plan["producer"],
            "estimated_request_count": 1,
            "transport": "planned_direct_or_tiled",
            "reason": (
                "One shared pre-label BurnDate raster per AOI, so the censoring "
                "cohort is identical across variants by construction."
            ),
        })

        # --- static reuse rows ---------------------------------------------
        for role in STATIC_REUSE_ROLES:
            rows.append({
                "aoi": aoi,
                "variant": "shared",
                "artifact_id": role,
                "role": role,
                "family": "static",
                "static_or_temporal": CLASSIFICATION_STATIC,
                "window_dependent": False,
                "reuse_or_recompute": "reuse",
                "export_required": False,
                "grid_family": "",
                "export_scale_m": 0,
                "start_date": "",
                "end_date": "",
                "expected_band_count": 0,
                "is_count_product": False,
                "output_path": "",
                "producer": "canonical (read-only)",
                "estimated_request_count": 0,
                "transport": "reuse",
                "reason": (
                    "Window-independent; deliberately held fixed so the closure "
                    "date is the only moving factor."
                ),
            })
    return rows


def _prelabel_scale() -> int:
    from core.config import VALIDATION_LABEL_EXPORT_SCALE

    return int(VALIDATION_LABEL_EXPORT_SCALE)


def _prelabel_plan(aoi: str, output_root: Optional[Path]) -> dict[str, Any]:
    from src.window_closure_sensitivity import prelabel_export_plan
    from src.multi_region_window_closure.dates import prelabel_censor_interval_for

    return prelabel_export_plan(aoi, prelabel_censor_interval_for(aoi), output_root)


def assert_export_plan(rows: Sequence[dict[str, Any]]) -> None:
    """Fail closed on a plan that would re-export a frozen artefact."""
    for row in rows:
        if row["variant"] == CANONICAL_VARIANT_ID and row["export_required"]:
            raise MultiRegionWindowClosureError(
                "BLOCKER: STATIC_ARTIFACT_REGENERATED -- canonical variant of "
                f"{row['aoi']} plans an export for {row['artifact_id']}. The "
                "canonical arm must reuse the frozen production outputs."
            )
        if row["static_or_temporal"] == CLASSIFICATION_STATIC:
            if row["export_required"] or row["reuse_or_recompute"] != "reuse":
                raise MultiRegionWindowClosureError(
                    "BLOCKER: STATIC_ARTIFACT_REGENERATED -- "
                    f"{row['aoi']}/{row['artifact_id']} is static but is "
                    "planned for recompute."
                )
        if row["window_dependent"] != (row["static_or_temporal"] == CLASSIFICATION_TEMPORAL):
            raise MultiRegionWindowClosureError(
                f"{row['aoi']}/{row['artifact_id']}: window_dependent "
                "disagrees with the static/temporal classification."
            )


def export_plan_row_count(aois: Sequence[str] = ACTUAL_AOIS) -> int:
    """Expected `export_plan.csv` row count, from the formula, per AOI."""
    total = 0
    for aoi in aois:
        rasters = expected_rasters_per_shifted_variant(baseline_years_for(aoi))
        total += rasters * len(VARIANTS)          # canonical reuse + 2 shifted
        total += 1                                # pre-label
        total += len(STATIC_REUSE_ROLES)          # static reuse
    return total
