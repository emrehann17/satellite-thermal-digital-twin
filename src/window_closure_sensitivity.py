"""
Generic, non-destructive, provenance-safe WINDOW-CLOSURE SENSITIVITY analysis.

Scientific question
-------------------
If the canonical predictor window is closed EARLIER -- shifted back by a fixed
number of days while keeping its LENGTH constant -- how sensitive are the
within-AOI baseline/thermal model results, and the thermal contribution, to the
predictor closure date?

What moves and what does not
----------------------------
Moving: the whole predictor window (start AND end shift by the same amount, so
duration is preserved), and therefore every predictor derived from it --
current Landsat LST, current Landsat NDVI, each baseline year's Landsat LST and
NDVI, the current-window MODIS mean/std/valid-observation support, and the
Step5/Step5C/Step7/Step8A products built on top of them.

Frozen: the label window, the DEM, slope, landcover, the AOI/grid, the model
feature registry, the model hyper-parameters, the random seed and the spatial
block definition. Changing only current LST while leaving NDVI, the baseline
years or MODIS frozen would be an incoherent half-shift and is refused.

Relationship to the existing counterfactual work
------------------------------------------------
`src/landsat_composite_counterfactual_audit.py` is a REDUCER counterfactual
(scene-weighted vs date-balanced compositing). This analysis changes the WINDOW,
not the reducer -- so it selects the production-equivalent scene-weighted
products only and never lets a date-balanced product into the primary plan.
Changing two factors at once would make neither attributable.

`src/landsat_composite_downstream_ab.py` changes LST only and keeps NDVI/MODIS
shared, which lets it enforce a baseline-invariance hard gate. A closure shift
moves current NDVI too, so baseline features legitimately change and that gate
is deliberately NOT reused here; both models are refit per variant instead.

Interpretation boundary
-----------------------
This is a predictor-timing sensitivity analysis. Closing the predictor window
earlier is NOT an operational forecasting validation, and a confidence interval
that includes zero is NOT evidence of equivalence -- no equivalence margin is
preregistered here.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.paths import PROJECT_ROOT

# --- Canonical date / EE-builder helpers (pure parts reused verbatim) --------
from src.landsat_composite_counterfactual_audit import (
    _baseline_year_window,
    _current_window,
    date_window_semantics,
    qa_mask_provenance,
)

# --- Canonical provenance helpers -------------------------------------------
from src.step8_large_block_robustness import (
    _git_commit,
    canonical_json,
    sha256_bytes,
    sha256_file,
)

SCHEMA_VERSION = "window_closure_sensitivity.v1"
DIAGNOSTIC_NAMESPACE = "window_closure_sensitivity"

PRIMARY_POPULATION = "burnable_tree_shrub_grass"
PRIMARY_MODEL = "random_forest"

DEFAULT_SHIFTS: tuple[int, ...] = (0, 7, 14)
CANONICAL_VARIANT_ID = "canonical"

# Production-equivalent compositing products ONLY. `date_balanced_*` belongs to
# the reducer counterfactual and must never enter this plan: window closure and
# compositing method would then move together and neither would be attributable.
PRODUCTION_LANDSAT_PRODUCTS: tuple[str, ...] = ("scene_weighted_median", "scene_valid_count")
FORBIDDEN_LANDSAT_PRODUCTS: tuple[str, ...] = (
    "date_balanced_median",
    "date_balanced_minus_scene_weighted",
)

# --- Label inputs -----------------------------------------------------------
# The label window is frozen, so the label rasters are frozen INPUTS: they are
# read read-only and their hashes enter the analysis identity. Both canonical
# production label products are pinned separately, because they are different
# artefacts with different roles: the raw BurnDate raster carries the burn day
# of year (and is what Step8A actually resolved), while the binary burned mask
# is the derived presence layer. Guessing a single `burned_labels.tif` would
# pin neither and would silently hash nothing.
LABEL_ROLE_RAW = "label_raw_burndate"
LABEL_ROLE_BINARY = "label_burned_binary"
REQUIRED_LABEL_ROLES: tuple[str, ...] = (LABEL_ROLE_RAW, LABEL_ROLE_BINARY)

# Canonical production file names, written by
# `src.step6_validate_fire_relation.export_raw_mcd64a1_prelabel_labels`
# (raw_out/binary_out defaults). Used only as the deterministic FALLBACK when
# no run metadata records the resolved path.
CANONICAL_LABEL_FILENAMES: dict[str, str] = {
    LABEL_ROLE_RAW: "mcd64a1_raw.tif",
    LABEL_ROLE_BINARY: "mcd64a1_burned.tif",
}
# Mirrors `src.step8a_prepare_500m_modeling_dataset.LABEL_KIND_RAW`; asserted
# against the production constant in the tests rather than imported, so this
# module stays free of Step8A's heavy raster imports.
LABEL_KIND_RAW_BURNDATE = "raw_burndate"
LABEL_METADATA_RELPATH = ("step8a", "step8a_dataset_stats.json")
LABEL_METADATA_KEYS: tuple[str, ...] = (
    "reference_500m_label_source", "label_raster_diagnostics",
)

# Frozen inputs that must exist AND hash before an actual plan may be written.
# The Step8A stats file is deliberately NOT required: it is a convenience
# resolver source, and the plan must not depend on a diagnostic side-file.
REQUIRED_FROZEN_INPUT_ROLES: tuple[str, ...] = (
    "canonical_step8a", "dem_elevation", "dem_slope", "landcover_aligned",
) + REQUIRED_LABEL_ROLES

LABEL_RESOLUTION_METADATA = "step8a_dataset_stats_metadata"
LABEL_RESOLUTION_CONTEXT = "experiment_context_gate_labels_dir"
LABEL_RESOLUTION_FALLBACK = "canonical_production_filename_fallback"

# --- Pre-label exclusion gate documents (frozen, read-only) ------------------
# An experiment may declare `exclude_pre_label_burns` in the registry. When it
# does, production Step8A REQUIRES the Step6B gate's cell-level exclusion
# manifest next to the label rasters (`ctx["gate_labels_dir"]`) and fails fast
# without it -- see
# `src.step8a_prepare_500m_modeling_dataset.read_pre_label_exclusion_manifest`.
#
# A variant's `gate_labels_dir` points INSIDE its own downstream input tree
# (`assert_local_downstream_context_safe` forbids it from pointing at the
# canonical namespace), so the gate documents have to be materialised there
# like every other frozen input. They are copied byte-verbatim and never
# regenerated: the exclusion set is a LABEL-side contract over the frozen label
# window, so it is identical for every predictor-timing variant.
#
# File names mirror `src.step6b_burned_landcover_gate`; they are asserted
# against those production constants in the tests rather than imported, so this
# module stays free of the gate's heavy raster imports (same pattern as
# `CANONICAL_LABEL_FILENAMES`).
PRELABEL_EXCLUSION_POLICY_FIELD = "exclude_pre_label_burns"
PRELABEL_EXCLUSION_ROLE_MANIFEST = "prelabel_exclusion_manifest"
PRELABEL_EXCLUSION_ROLE_METADATA = "prelabel_exclusion_manifest_metadata"
PRELABEL_EXCLUSION_ROLE_GATE_MANIFEST = "prelabel_exclusion_gate_manifest"
PRELABEL_EXCLUSION_FILENAMES: dict[str, str] = {
    PRELABEL_EXCLUSION_ROLE_MANIFEST: "pre_label_excluded_cells.parquet",
    PRELABEL_EXCLUSION_ROLE_METADATA: "pre_label_excluded_cells_metadata.json",
}
#: Written by `scripts/run_label_gate_only.py`; OPTIONAL for Step8A, but when
#: it exists production cross-validates it, so it is carried along.
PRELABEL_EXCLUSION_GATE_MANIFEST_TEMPLATE = "{experiment_id}_gate_manifest.json"
PRELABEL_EXCLUSION_REQUIRED_ROLES: tuple[str, ...] = (
    PRELABEL_EXCLUSION_ROLE_MANIFEST, PRELABEL_EXCLUSION_ROLE_METADATA,
)
#: The two Step8A audit columns the exclusion produces, and the stats-file
#: counters that account for them.
PRELABEL_EXCLUSION_AUDIT_COLUMNS: tuple[str, ...] = (
    "analysis_eligible", "pre_label_burn_excluded",
)
PRELABEL_EXCLUSION_STATS_COUNTERS: tuple[str, ...] = (
    "pre_label_burn_excluded_count", "analysis_eligible_count",
)

# --- Date-window semantics ---------------------------------------------------
# The upstream helper is a REDUCER counterfactual and says so in its note. That
# sentence is factually wrong for this analysis -- nothing about compositing
# moves here -- so the note is re-stated for the window-closure factor. The
# upstream module is NOT modified; only the record returned here is adapted.
WINDOW_CLOSURE_DATE_SEMANTICS_NOTE = (
    "Earth Engine filterDate end is exclusive. Reducer, QA masking and "
    "processing policy are held fixed; predictor-window timing is the "
    "intentionally changed factor."
)
# Wording inherited from the compositing counterfactual that must never appear
# in any window-closure record or report.
FOREIGN_FACTOR_PHRASES: tuple[str, ...] = (
    "compositing method is the only",
)

# --- MODIS current-window products ------------------------------------------
# File names are the production ones from
# `scripts/prepare_modis_for_step7.py:resolve_modis_output_paths`; only the
# namespace they are written into is the variant's.
MODIS_ROLE_FILENAMES: dict[str, str] = {
    "modis_lst_mean": "modis_lst_mean_celsius.tif",
    "modis_lst_std": "modis_lst_std_celsius.tif",
    "modis_valid_observation_count": "modis_valid_observation_count.tif",
}
MODIS_PRODUCER = "scripts/prepare_modis_for_step7.py:prepare_modis_for_step7"

# Inputs that MUST be identical across every variant.
# Context keys that deliberately keep pointing at the CANONICAL read-only
# artefacts: they are window-independent, so re-exporting them per variant
# would introduce a second moving part with no scientific justification.
READ_ONLY_SHARED_CONTEXT_KEYS: tuple[str, ...] = (
    "dem_input_dir", "landcover_aligned_path",
)

STATIC_SHARED_ROLES: tuple[str, ...] = (
    "dem_elevation", "dem_slope", "landcover_aligned", "aoi_geometry",
    "reference_grid", "label_window", "label_raster",
    "model_feature_registry", "model_hyperparameters", "random_seed",
    "spatial_block_definition",
)

STAGES: tuple[str, ...] = (
    "plan", "prelabel-export", "predictor-export", "local-downstream", "model", "compare",
)

# Every stage may now run for real. The build lock below is kept in place so a
# future stage cannot be executed before it is implemented and reviewed.
IMPLEMENTED_ACTUAL_STAGES: tuple[str, ...] = (
    "plan", "prelabel-export", "predictor-export", "local-downstream", "model",
    "compare",
)

STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "plan": (),
    "prelabel-export": ("plan",),
    "predictor-export": ("plan",),
    "local-downstream": ("predictor-export",),
    "model": ("prelabel-export", "local-downstream"),
    "compare": ("model",),
}

INTERVAL_SUPPORTED_INCREASE = "bootstrap_supported_increase"
INTERVAL_SUPPORTED_DECREASE = "bootstrap_supported_decrease"
INTERVAL_INCLUDES_ZERO = "interval_includes_zero"

BANNED_REPORT_PHRASES: tuple[str, ...] = (
    "statistically significant",
    "operational risk",
    "early-warning",
    "early warning",
    "prediction of future fires",
    "proves that",
    "equivalent performance",
) + FOREIGN_FACTOR_PHRASES

LIMITATIONS: tuple[str, ...] = (
    "Closing the predictor window earlier is a PREDICTOR-TIMING SENSITIVITY "
    "analysis. It is retrospective and is not an operational forecasting "
    "validation.",
    "A single AOI and a single season are not evidence of generalisability.",
    "A confidence interval that includes zero leaves directional uncertainty "
    "unresolved in this analysis.",
    "Any performance change is consistent with the predictor and "
    "observation-support changes that follow from the closure date. These "
    "results are descriptive and do not establish an underlying mechanism.",
    "Landsat and MODIS scene/observation support can itself change with the "
    "closure date, so support differences are part of what is being measured.",
    "PR-AUC depends on prevalence; every comparison here is made on the SAME "
    "common cohort so prevalence is held fixed across variants.",
    "The label window is frozen and identical across variants; only predictor "
    "timing moves.",
    "The calendar-month filter reported for the current-window Landsat roles "
    "is derived from the window itself, exactly as production Step3 derives "
    "it. When it widens to 1-12 it is REDUNDANT next to the exact filterDate "
    "range, which is the binding date contract; it does NOT mean that "
    "whole-year data is used.",
    "Results are descriptive. No deployment, alerting or future-fire "
    "forecasting claim is made or supported by this analysis.",
)


class WindowClosureError(SystemExit):
    """Fail-fast, contract-violating condition."""


# =============================================================================
# Shifts and variant identity
# =============================================================================
def variant_id(shift_days: int) -> str:
    return CANONICAL_VARIANT_ID if int(shift_days) == 0 else f"close_{int(shift_days)}d_earlier"


def normalize_shifts(shifts: Optional[Iterable[int]] = None) -> tuple[int, ...]:
    """Deterministically ordered, de-duplicated, validated closure shifts.

    Shifts are preregistered day counts by which the WHOLE window moves
    earlier. They are never window LENGTHS -- length is invariant.
    """
    if shifts is None:
        return tuple(DEFAULT_SHIFTS)
    values: list[int] = []
    for raw in shifts:
        if isinstance(raw, bool) or not isinstance(raw, (int,)):
            try:
                raw = int(str(raw).strip())
            except (TypeError, ValueError):
                raise WindowClosureError(f"Closure shift must be an integer number of days: {raw!r}.")
        if raw < 0:
            raise WindowClosureError(
                f"Negative closure shift is not allowed: {raw}. Shifts move the "
                "window EARLIER, so they must be >= 0."
            )
        values.append(int(raw))
    if not values:
        raise WindowClosureError("At least one closure shift must be given.")
    # Deterministic de-duplication: order of the caller's list cannot matter.
    return tuple(sorted(set(values)))


def _parse(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _fmt(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")


# =============================================================================
# Window construction
# =============================================================================
def canonical_window(ctx: dict) -> dict:
    """The canonical predictor and label windows, read from the registry ctx."""
    for key in ("predictor_start_date", "predictor_end_date", "label_start_date", "label_end_date"):
        if not ctx.get(key):
            raise WindowClosureError(f"Experiment context is missing '{key}'.")
    start, end = _parse(ctx["predictor_start_date"]), _parse(ctx["predictor_end_date"])
    label_start, label_end = _parse(ctx["label_start_date"]), _parse(ctx["label_end_date"])
    if end >= label_start:
        raise WindowClosureError(
            f"Canonical predictor_end ({_fmt(end)}) must precede label_start "
            f"({_fmt(label_start)})."
        )
    return {
        "predictor_start_date": _fmt(start),
        "predictor_end_date": _fmt(end),
        "duration_days": (end - start).days,
        "label_start_date": _fmt(label_start),
        "label_end_date": _fmt(label_end),
        "lead_days": (label_start - end).days,
        "baseline_years": list(ctx.get("baseline_years") or []),
        "current_period_days": int(ctx.get("current_period_days") or (end - start).days),
    }


def build_window_variants(ctx: dict, shifts: Optional[Iterable[int]] = None) -> list[dict]:
    """One variant per preregistered shift, with duration and label invariance.

    `variant_start = canonical_start - shift`, `variant_end = canonical_end -
    shift`: because BOTH ends move by the same amount, the window LENGTH is
    preserved exactly. The label window never moves.
    """
    canonical = canonical_window(ctx)
    shift_values = normalize_shifts(shifts)
    canonical_start, canonical_end = (
        _parse(canonical["predictor_start_date"]), _parse(canonical["predictor_end_date"])
    )
    label_start = _parse(canonical["label_start_date"])

    variants: list[dict] = []
    for shift in shift_values:
        start = canonical_start - timedelta(days=shift)
        end = canonical_end - timedelta(days=shift)
        duration = (end - start).days
        if duration != canonical["duration_days"]:
            raise WindowClosureError(
                f"Shift {shift} changed the window duration "
                f"({duration} != {canonical['duration_days']}); a closure shift "
                "must move both ends equally."
            )
        if end >= label_start:
            raise WindowClosureError(
                f"Shift {shift} yields predictor_end {_fmt(end)} which does not "
                f"precede label_start {_fmt(label_start)}."
            )
        variants.append({
            "variant_id": variant_id(shift),
            "shift_days": int(shift),
            "predictor_start_date": _fmt(start),
            "predictor_end_date": _fmt(end),
            "duration_days": duration,
            "duration_preserved": duration == canonical["duration_days"],
            "label_start_date": canonical["label_start_date"],
            "label_end_date": canonical["label_end_date"],
            "label_window_unchanged": True,
            "lead_days": (label_start - end).days,
            "is_canonical": int(shift) == 0,
        })
    return variants


def common_prelabel_interval(variants: Sequence[dict]) -> dict:
    """The censoring interval shared by EVERY variant.

    Moving the closure date earlier opens a gap between predictor_end and
    label_start. Cells that burn inside that gap are not label-window positives
    and leaving them as negatives would be wrong, so ONE interval -- from the
    earliest variant predictor_start to the day before label_start -- is
    preregistered and applied identically to all variants. This is generic: it
    does not depend on any experiment's `exclude_pre_label_burns` flag.
    """
    if not variants:
        raise WindowClosureError("Cannot derive a censoring interval without variants.")
    starts = [_parse(v["predictor_start_date"]) for v in variants]
    label_start = _parse(variants[0]["label_start_date"])
    if len({v["label_start_date"] for v in variants}) != 1:
        raise WindowClosureError("Variants disagree on label_start_date; label window must be frozen.")
    start = min(starts)
    end = label_start - timedelta(days=1)
    if end < start:
        raise WindowClosureError(
            f"Censoring interval is empty: {_fmt(start)} .. {_fmt(end)}."
        )
    return {
        "common_prelabel_start": _fmt(start),
        "common_prelabel_end": _fmt(end),
        "derivation": (
            "start = min(variant predictor_start) over all preregistered "
            "shifts; end = label_start - 1 day"
        ),
        "applies_to_all_variants": True,
        "independent_of_exclude_pre_label_burns_flag": True,
    }


# =============================================================================
# Variant context (never mutates the registry)
# =============================================================================
def diagnostics_root(output_root: Optional[Path] = None) -> Path:
    if output_root is not None:
        return Path(output_root)
    return PROJECT_ROOT / "outputs" / "diagnostics" / DIAGNOSTIC_NAMESPACE


def experiment_root(experiment_id: str, output_root: Optional[Path] = None) -> Path:
    return diagnostics_root(output_root) / experiment_id


def variant_root(experiment_id: str, variant: str, output_root: Optional[Path] = None) -> Path:
    return experiment_root(experiment_id, output_root) / "variants" / variant


def canonical_experiment_root(experiment_id: str, experiments_root: Optional[Path] = None) -> Path:
    """READ-ONLY canonical production namespace for this experiment."""
    if experiments_root is not None:
        from core.regions import get_experiment
        return Path(experiments_root) / get_experiment(experiment_id)["output_namespace"]
    from core.regions import get_experiment_output_root
    return get_experiment_output_root(experiment_id)


def build_window_variant_context(
    experiment_id: str,
    shift_days: int,
    base_context: Optional[dict] = None,
    output_root: Optional[Path] = None,
) -> dict:
    """A DEEP COPY of the registry context, re-pointed at the variant namespace.

    The global registry and the caller's context are never mutated: every path
    and date is rewritten on a copy. A zero shift still gets its own context so
    the canonical variant is described by the same code path, but the canonical
    variant reads frozen production outputs rather than re-exporting them.
    """
    if base_context is None:
        from core.experiment_context import build_experiment_context
        base_context = build_experiment_context(experiment_id)
    ctx = copy.deepcopy(base_context)

    canonical = canonical_window(base_context)
    shift = int(shift_days)
    if shift < 0:
        raise WindowClosureError(f"Negative closure shift is not allowed: {shift}.")

    start = _parse(canonical["predictor_start_date"]) - timedelta(days=shift)
    end = _parse(canonical["predictor_end_date"]) - timedelta(days=shift)
    if end >= _parse(canonical["label_start_date"]):
        raise WindowClosureError(
            f"Shift {shift} yields predictor_end {_fmt(end)} which does not "
            f"precede label_start {canonical['label_start_date']}."
        )

    variant = variant_id(shift)
    root = variant_root(experiment_id, variant, output_root)
    data_root = root / "data"

    ctx["experiment_id"] = experiment_id
    ctx["window_closure_variant_id"] = variant
    ctx["window_closure_shift_days"] = shift
    ctx["predictor_start_date"] = _fmt(start)
    ctx["predictor_end_date"] = _fmt(end)
    ctx["current_period_end_date"] = _fmt(end)
    # Window LENGTH is invariant; only its position moves.
    ctx["current_period_days"] = canonical["current_period_days"]
    # Label window is frozen -- copied through untouched, asserted below.
    ctx["label_start_date"] = canonical["label_start_date"]
    ctx["label_end_date"] = canonical["label_end_date"]

    ctx["output_root"] = root
    ctx["data_root"] = data_root
    ctx["baseline_input_dir"] = data_root / "landsat_timeseries"
    ctx["qa_dir"] = data_root / "landsat_qa"
    ctx["current_period_dir"] = data_root / "current_period"
    ctx["modis_input_dir"] = data_root / "modis"
    ctx["modis_dir"] = data_root / "modis"
    ctx["ndvi_baseline_dir"] = data_root / "ndvi_timeseries"
    ctx["ndvi_current_dir"] = data_root / "ndvi_current_period"
    # DEM/slope and the aligned landcover are STATIC shared inputs: they do not
    # depend on the predictor window, so they keep pointing at the canonical
    # read-only copies instead of being re-exported per variant. Re-exporting
    # them would add a second moving part for no scientific reason.
    for key in READ_ONLY_SHARED_CONTEXT_KEYS:
        if key in base_context:
            ctx[key] = base_context[key]
    for step in ("step5", "step5b", "step5c", "step7a", "step7b", "step7c",
                 "step7d", "step7e", "step8a", "step8b", "step8c", "step8d", "step8e"):
        ctx[f"{step}_output_dir"] = root / step
    ctx["output_dir"] = root / "step5"
    ctx["gate_labels_dir"] = root / "validation" / "labels"

    assert_variant_context_safe(ctx, experiment_id, base_context, output_root)
    return ctx


def assert_variant_context_safe(
    ctx: dict, experiment_id: str, base_context: dict, output_root: Optional[Path] = None,
) -> None:
    """No variant path may point into the canonical production namespace."""
    forbidden = Path(base_context["output_root"]).resolve()
    allowed = variant_root(
        experiment_id, ctx["window_closure_variant_id"], output_root
    ).resolve()
    for key, value in ctx.items():
        if not isinstance(value, Path):
            continue
        if key in READ_ONLY_SHARED_CONTEXT_KEYS:
            # Window-independent static input, intentionally still canonical.
            # It is READ here and never written; the write guard below is what
            # keeps the variant's own products inside its namespace.
            continue
        resolved = value.resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise WindowClosureError(
                f"Variant context key '{key}' points into the canonical "
                f"production namespace: {resolved}."
            )
        if allowed != resolved and allowed not in resolved.parents:
            raise WindowClosureError(
                f"Variant context key '{key}' escapes its variant namespace: {resolved}."
            )
    if ctx["label_start_date"] != base_context["label_start_date"] or \
            ctx["label_end_date"] != base_context["label_end_date"]:
        raise WindowClosureError("Label window must be identical to the canonical one.")


# =============================================================================
# Date-window semantics for THIS analysis
# =============================================================================
def window_closure_date_window_semantics(start_date: str, end_date: str) -> dict:
    """The frozen end-EXCLUSIVE filterDate record, re-stated for this factor.

    The arithmetic is the upstream counterfactual's, reused verbatim, so the
    off-by-one is documented identically. Only the `note` is adapted: the
    upstream sentence names the COMPOSITING METHOD as the changed factor, which
    is simply untrue here -- the reducer, the QA masking and the processing
    policy are all held fixed and predictor-window TIMING is what moves.
    `src/landsat_composite_counterfactual_audit.py` is left untouched.
    """
    semantics = dict(date_window_semantics(start_date, end_date))
    semantics["note"] = WINDOW_CLOSURE_DATE_SEMANTICS_NOTE
    semantics["changed_factor"] = "predictor_window_timing"
    semantics["held_fixed"] = ["reducer", "qa_masking", "processing_policy"]
    semantics["note_source"] = (
        "adapted in src/window_closure_sensitivity.py; the upstream reducer "
        "counterfactual wording does not apply to a window-closure shift"
    )
    assert_no_foreign_factor_wording(semantics, "date-window semantics")
    return semantics


def assert_no_foreign_factor_wording(payload: Any, where: str) -> None:
    """Refuse any record that still names a factor this analysis does not move."""
    blob = canonical_json(payload).lower() if not isinstance(payload, str) else payload.lower()
    found = sorted(p for p in FOREIGN_FACTOR_PHRASES if p in blob)
    if found:
        raise WindowClosureError(
            f"{where} carries wording inherited from another analysis: {found}. "
            "Window closure changes predictor-window timing, not the "
            "compositing method."
        )


# =============================================================================
# Calendar-month filter transparency
# =============================================================================
def calendar_month_filter_transparency(
    months_filter: Optional[str], start_date: str, end_date: str,
) -> dict:
    """Explain the production-derived calendar-month filter for one role.

    Production Step3 (`get_current_period_median` /
    `get_current_period_ndvi_median`) derives the month filter FROM the window
    -- `{(start.month + i) % 12 or 12 for i in range(window_days + 1)}` -- and
    the reused `_current_window` helper mirrors that formula exactly, so the
    value reported here is production-equivalent and is NOT changed here.

    A window whose arithmetic spans all twelve calendar months collapses the
    filter to `calendarRange(1, 12, "month")`, which admits every date the
    exact `filterDate` range already admits. The filter is then redundant and
    the exact start/end dates are the binding date contract.
    """
    if not months_filter:
        return {
            "calendar_month_filter": None,
            "calendar_month_filter_applied": False,
            "calendar_month_filter_redundant": False,
            "exact_filter_date_is_binding": True,
            "filter_date_start": start_date,
            "filter_date_end": end_date,
            "note": (
                "Production applies no calendar-month filter to this role; the "
                "exact filterDate range is the only date constraint."
            ),
        }
    first, _, last = str(months_filter).partition("-")
    try:
        month_start, month_end = int(first), int(last or first)
    except ValueError:
        raise WindowClosureError(f"Unparseable months_filter: {months_filter!r}.")
    redundant = (month_start, month_end) == (1, 12)
    note = (
        "Derived from the window by the production Step3 formula and left "
        "unchanged. It spans all twelve calendar months, so "
        "ee.Filter.calendarRange(1, 12, 'month') removes no scene that "
        f"filterDate({start_date}, {end_date}) already admits: the exact "
        "filterDate range is binding. A 1-12 month filter does NOT mean that "
        "whole-year data is used -- the predictor window is still "
        f"{start_date} .. {end_date}."
    ) if redundant else (
        "Derived from the window by the production Step3 formula and left "
        f"unchanged. It restricts scenes to calendar months {month_start}-"
        f"{month_end} INSIDE the exact filterDate range "
        f"({start_date} .. {end_date}), which remains binding."
    )
    return {
        "calendar_month_filter": str(months_filter),
        "calendar_month_filter_applied": True,
        "calendar_month_filter_redundant": redundant,
        "calendar_month_filter_production_equivalent": True,
        "calendar_month_filter_source": (
            "src.landsat_composite_counterfactual_audit._current_window, which "
            "mirrors src.step3_landsat_lst.get_current_period_median"
        ),
        "exact_filter_date_is_binding": True,
        "filter_date_start": start_date,
        "filter_date_end": end_date,
        "note": note,
    }


# =============================================================================
# Export plans (pure -- no Earth Engine call is made here)
# =============================================================================
def landsat_export_plan(variant: dict, baseline_years: Sequence[int], window_days: int) -> dict:
    """Landsat LST + NDVI roles for one variant, reusing the canonical windows.

    Uses `_current_window` / `_baseline_year_window` from the counterfactual
    audit so the month filter and the symmetric baseline windows are the exact
    production ones; only the closing date differs.
    """
    current = _current_window(variant["predictor_end_date"], window_days)
    current_transparency = calendar_month_filter_transparency(
        current["months_filter"], current["start_date"], current["end_date"],
    )
    roles: list[dict] = []
    for family in ("lst", "ndvi"):
        roles.append({
            "role": f"current_{family}",
            "family": family,
            "scope": "current_window",
            "start_date": current["start_date"],
            "end_date": current["end_date"],
            # Production value, reported unchanged, plus what it actually binds.
            "months_filter": current["months_filter"],
            **current_transparency,
            "products": list(PRODUCTION_LANDSAT_PRODUCTS),
            "date_window_semantics": window_closure_date_window_semantics(
                current["start_date"], current["end_date"]
            ),
        })
    for year in sorted(baseline_years):
        window = _baseline_year_window(variant["predictor_end_date"], window_days, int(year))
        transparency = calendar_month_filter_transparency(
            None, window["start_date"], window["end_date"],
        )
        for family in ("lst", "ndvi"):
            roles.append({
                "role": f"baseline_{family}_{year}",
                "family": family,
                "scope": "baseline_year",
                "baseline_year": int(year),
                "start_date": window["start_date"],
                "end_date": window["end_date"],
                "months_filter": None,
                **transparency,
                "products": list(PRODUCTION_LANDSAT_PRODUCTS),
                "date_window_semantics": window_closure_date_window_semantics(
                    window["start_date"], window["end_date"]
                ),
            })
    assert_no_forbidden_products(roles)
    return {
        "variant_id": variant["variant_id"],
        "window_days": int(window_days),
        "current_window": current,
        "calendar_month_filter_transparency": current_transparency,
        "roles": roles,
        "role_count": len(roles),
        "qa_mask_provenance": qa_mask_provenance(),
        "reducer": "scene_weighted",
        "reducer_note": (
            "Production-equivalent scene-weighted compositing only. The "
            "date_balanced reducer belongs to the separate compositing "
            "counterfactual; changing the window and the reducer together "
            "would make neither attributable."
        ),
    }


def assert_no_forbidden_products(roles: Sequence[dict]) -> None:
    for role in roles:
        leaked = sorted(set(role.get("products", [])) & set(FORBIDDEN_LANDSAT_PRODUCTS))
        if leaked:
            raise WindowClosureError(
                f"Role '{role.get('role')}' requests reducer-counterfactual "
                f"product(s) {leaked}; the window-closure primary plan is "
                "scene-weighted only."
            )


# Any product whose NAME starts with this prefix belongs to the reducer
# counterfactual, including future `date_balanced_*` variants that are not in
# the frozen FORBIDDEN_LANDSAT_PRODUCTS list yet.
FORBIDDEN_LANDSAT_PRODUCT_PREFIX = "date_balanced"


def is_forbidden_landsat_product(product: Any) -> bool:
    """Whether a REAL export product name is a reducer-counterfactual product.

    Applied to product NAMES only (`date_balanced_median`,
    `date_balanced_minus_scene_weighted`, any other `date_balanced_*`).
    Documentation fields that merely MENTION these products in order to ban
    them (`forbidden_products`, notes, limitations) are never routed through
    this predicate.
    """
    name = str(product)
    return (
        name in FORBIDDEN_LANDSAT_PRODUCTS
        or name.startswith(FORBIDDEN_LANDSAT_PRODUCT_PREFIX)
    )


def collect_actual_landsat_products(plan: dict) -> list[str]:
    """Every ACTUAL Landsat export product a plan/metadata document names.

    Walks only the SEMANTIC product fields -- the `products` list of each
    Landsat role and the `product` field of Landsat-family artefact/job
    records -- never the serialized JSON text. A raw substring check over the
    whole document is wrong by construction: valid plans legitimately mention
    `date_balanced_*` in `forbidden_products`, notes and limitations exactly
    in order to BAN it, and would all be flagged.
    """
    products: list[str] = []
    landsat = plan.get("landsat") if isinstance(plan.get("landsat"), dict) else {}
    for roles in (
        landsat.get("current_roles"), landsat.get("baseline_roles"),
        landsat.get("roles"), plan.get("roles"),
    ):
        for role in roles or []:
            if isinstance(role, dict):
                products.extend(str(item) for item in (role.get("products") or []))
    for records in (plan.get("expected_artifacts"), plan.get("artifact_inventory")):
        for record in records or []:
            if isinstance(record, dict) and record.get("family") in ("lst", "ndvi") \
                    and record.get("product") is not None:
                products.append(str(record["product"]))
    return products


def landsat_product_violations(plan: dict) -> list[str]:
    """Structural audit of the REAL Landsat export products of one document.

    Returns human-readable violations; an empty list means the document plans
    the production scene-weighted products only. Documentation fields naming
    the banned products are deliberately not inspected.
    """
    products = collect_actual_landsat_products(plan)
    violations: list[str] = []
    forbidden = sorted({p for p in products if is_forbidden_landsat_product(p)})
    if forbidden:
        violations.append(
            f"forbidden reducer-counterfactual product(s) {forbidden}"
        )
    unexpected = sorted(set(products) - set(PRODUCTION_LANDSAT_PRODUCTS) - set(forbidden))
    if unexpected:
        violations.append(
            f"non-production Landsat product(s) {unexpected}; only "
            f"{list(PRODUCTION_LANDSAT_PRODUCTS)} are allowed"
        )
    return violations


def modis_export_plan(
    variant: dict, experiment_id: str, output_root: Optional[Path] = None,
) -> dict:
    """Current-window MODIS roles, driven by the VARIANT dates.

    The numeric recipe is the production one in
    `scripts/prepare_modis_for_step7.py`; only the window it is evaluated over
    moves. No independent MODIS formula is defined here.

    Every role carries the EXACT predictor dates it will be evaluated over and
    the variant-namespaced path it will be written to, so a plan can be audited
    without re-deriving the binding from the variant context. The dates are the
    variant's own -- no AOI or calendar date is hard-coded in this module.
    """
    variant_data_root = (
        variant_root(experiment_id, variant["variant_id"], output_root) / "data" / "modis"
    )
    roles = [
        {
            "role": role,
            "scope": "current_window",
            "start_date": variant["predictor_start_date"],
            "end_date": variant["predictor_end_date"],
            "producer": MODIS_PRODUCER,
            "output_path": str(variant_data_root / filename),
            # The producer reads ctx["predictor_start_date"] /
            # ctx["predictor_end_date"] / ctx["modis_dir"] from the variant
            # context built by build_window_variant_context, never the registry.
            "uses_variant_context": True,
        }
        for role, filename in sorted(MODIS_ROLE_FILENAMES.items())
    ]
    return {
        "variant_id": variant["variant_id"],
        "start_date": variant["predictor_start_date"],
        "end_date": variant["predictor_end_date"],
        "output_dir": str(variant_data_root),
        "roles": roles,
        "producer": MODIS_PRODUCER,
        "producer_note": (
            "Reuses the production MODIS QC mask, reducers and export scale "
            "verbatim; only ctx predictor_start_date/predictor_end_date differ."
        ),
    }


def static_shared_plan() -> dict:
    return {
        "roles": list(STATIC_SHARED_ROLES),
        "mode": "shared_read_only",
        "note": (
            "Identical across every variant and never re-exported: these are "
            "the factors deliberately held fixed so the closure date is the "
            "only moving one."
        ),
    }


def prelabel_export_plan(
    experiment_id: str, censor: dict, output_root: Optional[Path] = None,
) -> dict:
    """Plan for the ONE shared pre-label BurnDate raster.

    Reuses `src.step6_validate_fire_relation.export_raw_mcd64a1_prelabel_labels`
    with an explicit window and an explicit output path inside this namespace.
    The canonical Step6/Step6B gate is never re-run and never overwritten.
    """
    root = experiment_root(experiment_id, output_root) / "prelabel_censor"
    return {
        "experiment_id": experiment_id,
        "pre_label_start": censor["common_prelabel_start"],
        "pre_label_end": censor["common_prelabel_end"],
        "output_dir": str(root),
        "raster_path": str(root / "prelabel_burndate.tif"),
        "producer": (
            "src.step6_validate_fire_relation.export_raw_mcd64a1_prelabel_labels"
        ),
        "writes_into_canonical_namespace": False,
        "canonical_gate_rerun": False,
        "note": (
            "One raster shared by every variant, so the censoring cohort is "
            "identical across variants by construction."
        ),
    }


# =============================================================================
# Frozen input inventory / analysis identity
# =============================================================================
def canonical_step8a_path(experiment_id: str, experiments_root: Optional[Path] = None) -> Path:
    return (
        canonical_experiment_root(experiment_id, experiments_root)
        / "step8a" / "step8a_500m_modeling_dataset.parquet"
    )


def _is_inside(path: Path, root: Path) -> bool:
    resolved, root = Path(path).resolve(), Path(root).resolve()
    return resolved == root or root in resolved.parents


def _metadata_raw_label_path(root: Path) -> Optional[Path]:
    """The raw BurnDate raster recorded by the frozen Step8A run, if any.

    This is the authoritative resolver source: it is the label the frozen
    canonical Step8A dataset was actually built from, not a guess. A recorded
    path is accepted only when it still lies inside this experiment's canonical
    namespace, so an injected `experiments_root` can never be escaped.
    """
    stats = root.joinpath(*LABEL_METADATA_RELPATH)
    if not stats.is_file():
        return None
    try:
        payload = json.loads(stats.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("label_kind") != LABEL_KIND_RAW_BURNDATE:
        return None
    for key in LABEL_METADATA_KEYS:
        entry = payload.get(key)
        if isinstance(entry, dict) and entry.get("path"):
            candidate = Path(str(entry["path"]))
            if _is_inside(candidate, root):
                return candidate
    return None


def resolve_label_inputs(
    experiment_id: str,
    experiments_root: Optional[Path] = None,
    base_context: Optional[dict] = None,
) -> dict[str, dict]:
    """Deterministically resolve BOTH canonical label rasters.

    No `burned_labels.tif` is ever guessed. Resolution order per role:

      1. the path recorded in the frozen Step8A run metadata (raw BurnDate
         only -- it is the only label Step8A records);
      2. the experiment context's `gate_labels_dir`, i.e. the production label
         directory contract, joined with the canonical production file name;
      3. the canonical `validation/labels/<name>` fallback.

    The returned entries are declarations only: nothing is read, hashed or
    required here, so a missing label is reported rather than raised.
    """
    root = canonical_experiment_root(experiment_id, experiments_root)
    fallback_dir = root / "validation" / "labels"

    context_dir: Optional[Path] = None
    if base_context is not None and base_context.get("gate_labels_dir"):
        candidate = Path(base_context["gate_labels_dir"])
        # Only trust the context directory when it belongs to the canonical
        # namespace this run is pinned to; an injected experiments_root wins.
        if _is_inside(candidate, root):
            context_dir = candidate

    metadata_raw = _metadata_raw_label_path(root)

    resolved: dict[str, dict] = {}
    for role in REQUIRED_LABEL_ROLES:
        filename = CANONICAL_LABEL_FILENAMES[role]
        if role == LABEL_ROLE_RAW and metadata_raw is not None:
            path, source = metadata_raw, LABEL_RESOLUTION_METADATA
        elif context_dir is not None:
            path, source = context_dir / filename, LABEL_RESOLUTION_CONTEXT
        else:
            path, source = fallback_dir / filename, LABEL_RESOLUTION_FALLBACK
        resolved[role] = {
            "role": role,
            "path": path,
            "resolved_from": source,
            "canonical_filename": filename,
            "required": True,
        }
    return resolved


def frozen_input_inventory(
    experiment_id: str,
    experiments_root: Optional[Path] = None,
    base_context: Optional[dict] = None,
) -> dict:
    """SHA-256 of the frozen canonical inputs this analysis reads read-only.

    The two label rasters are pinned as SEPARATE roles: they are distinct
    artefacts (raw BurnDate day-of-year vs derived binary mask) and both are
    frozen inputs of a frozen label window, so both hashes belong in the
    analysis identity.
    """
    root = canonical_experiment_root(experiment_id, experiments_root)
    candidates: dict[str, Path] = {
        "canonical_step8a": root / "step8a" / "step8a_500m_modeling_dataset.parquet",
        "canonical_step8a_stats": root / "step8a" / "step8a_dataset_stats.json",
        "dem_elevation": root / "data" / "dem" / "elevation.tif",
        "dem_slope": root / "data" / "dem" / "slope.tif",
        "landcover_aligned": (
            root / "gate_inputs"
            / "landcover_esa_worldcover_v200_aligned_to_reference.tif"
        ),
    }
    labels = resolve_label_inputs(experiment_id, experiments_root, base_context)
    inventory: dict[str, Any] = {}
    for name, path in candidates.items():
        inventory[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    for role, entry in labels.items():
        path = Path(entry["path"])
        inventory[role] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "role": role,
            "resolved_from": entry["resolved_from"],
            "canonical_filename": entry["canonical_filename"],
            "required": True,
        }
    return inventory


def label_prerequisites(inventory: dict) -> dict:
    """Whether every REQUIRED label input resolved to a real, hashable file."""
    missing: list[dict] = []
    for role in REQUIRED_LABEL_ROLES:
        entry = inventory.get(role) or {}
        if not entry.get("exists") or entry.get("sha256") is None:
            missing.append({
                "role": role,
                "path": entry.get("path"),
                "exists": bool(entry.get("exists")),
                "sha256": entry.get("sha256"),
                "reason": (
                    "required label input does not exist and therefore cannot "
                    "be hashed"
                ),
            })
    return {
        "required_label_roles": list(REQUIRED_LABEL_ROLES),
        "prerequisites_ready": not missing,
        "missing_required_inputs": missing,
    }


def assert_label_prerequisites(inventory: dict) -> None:
    """Fail fast BEFORE any preregistration is written.

    A null label hash would make the analysis identity unpinned while still
    looking pinned, so a missing required label stops the actual plan stage
    rather than being recorded as `sha256: null`.
    """
    status = label_prerequisites(inventory)
    if status["prerequisites_ready"]:
        return
    roles = ", ".join(entry["role"] for entry in status["missing_required_inputs"])
    paths = "; ".join(str(entry["path"]) for entry in status["missing_required_inputs"])
    raise WindowClosureError(
        f"Required label input(s) missing: {roles}. Expected at: {paths}. "
        "The analysis identity must pin every frozen label hash, so no "
        "preregistration is written with a null hash. Produce the canonical "
        "label rasters first (Step6 raw BurnDate export), or re-run with "
        "--dry-run to inspect the plan."
    )


# =============================================================================
# Pre-label exclusion binding
#
# GENERIC, registry-driven. No experiment id, date or count is hard-coded: the
# policy comes from `EXPERIMENTS[<id>][exclude_pre_label_burns]` through the
# built experiment context, and the documents come from that context's
# canonical `gate_labels_dir`.
# =============================================================================
def prelabel_exclusion_binding(
    experiment_id: str, base_context: dict,
    experiments_root: Optional[Path] = None,
) -> dict:
    """Resolve the experiment's pre-label censor policy and its gate documents.

    Read-only: it hashes what already exists and creates nothing. When the
    registry does not enable the policy the binding is inactive and carries no
    document, which is the correct contract for an experiment that has no
    pre-label exclusion (the analysis-wide censor in `common_prelabel_interval`
    is separate and always applies).
    """
    active = bool(base_context.get(PRELABEL_EXCLUSION_POLICY_FIELD, False))
    # Same resolution order as `resolve_label_inputs`: the context's gate
    # directory is trusted only when it belongs to the canonical namespace this
    # run is pinned to, so an injected `experiments_root` always wins.
    canonical_root = canonical_experiment_root(experiment_id, experiments_root)
    source_dir = canonical_root / "validation" / "labels"
    candidate = base_context.get("gate_labels_dir")
    if candidate and _is_inside(Path(candidate), canonical_root):
        source_dir = Path(candidate)
    documents: dict[str, dict] = {}
    if source_dir is not None:
        source_dir = Path(source_dir)
        wanted = dict(PRELABEL_EXCLUSION_FILENAMES)
        wanted[PRELABEL_EXCLUSION_ROLE_GATE_MANIFEST] = (
            PRELABEL_EXCLUSION_GATE_MANIFEST_TEMPLATE.format(experiment_id=experiment_id)
        )
        for role, filename in sorted(wanted.items()):
            path = source_dir / filename
            exists = path.is_file()
            documents[role] = {
                "role": role,
                "filename": filename,
                "path": str(path),
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
                "required": role in PRELABEL_EXCLUSION_REQUIRED_ROLES,
                "access": "read_only",
            }
    missing = sorted(
        role for role in PRELABEL_EXCLUSION_REQUIRED_ROLES
        if not (documents.get(role) or {}).get("exists")
    )
    return {
        "exclude_pre_label_burns": active,
        "policy_source": (
            f"core.regions.EXPERIMENTS[{experiment_id!r}]"
            f".{PRELABEL_EXCLUSION_POLICY_FIELD} via core.experiment_context"
        ),
        "binding_required": active,
        "canonical_gate_labels_dir": str(source_dir) if source_dir is not None else None,
        "documents": documents,
        "required_roles": list(PRELABEL_EXCLUSION_REQUIRED_ROLES),
        "missing_required_documents": missing if active else [],
        "expected_audit_columns": list(PRELABEL_EXCLUSION_AUDIT_COLUMNS),
        "expected_stats_counters": list(PRELABEL_EXCLUSION_STATS_COUNTERS),
        "consumer": (
            "src.step8a_prepare_500m_modeling_dataset."
            "read_pre_label_exclusion_manifest via ctx['gate_labels_dir']"
        ),
        "binding_ready": bool(not active or not missing),
    }


def assert_prelabel_exclusion_binding(binding: dict, when: str) -> dict:
    """Fail closed when the registry enables the policy but it cannot be bound.

    Run BEFORE the export stage so a missing gate document never costs an Earth
    Engine export: the policy, the document set and the expected variant
    contract are all statically resolvable at plan time.
    """
    if not binding["exclude_pre_label_burns"]:
        return binding
    if binding["canonical_gate_labels_dir"] is None:
        raise WindowClosureError(
            f"BLOCKER: PRELABEL_EXCLUSION_BINDING_MISSING ({when}) -- the "
            f"experiment declares {PRELABEL_EXCLUSION_POLICY_FIELD}=True but "
            "its context carries no 'gate_labels_dir', so the Step6B exclusion "
            "manifest cannot be bound to the variant namespace."
        )
    if binding["missing_required_documents"]:
        paths = "; ".join(
            str((binding["documents"].get(role) or {}).get("path"))
            for role in binding["missing_required_documents"]
        )
        raise WindowClosureError(
            f"BLOCKER: PRELABEL_EXCLUSION_BINDING_MISSING ({when}) -- the "
            f"experiment declares {PRELABEL_EXCLUSION_POLICY_FIELD}=True, so "
            "production Step8A requires the Step6B gate exclusion manifest in "
            "every variant's gate_labels_dir. Missing required document(s): "
            f"{binding['missing_required_documents']}. Expected at: {paths}. "
            "Re-run the label gate before this stage; nothing was exported, "
            "created or written."
        )
    return binding


def assert_prelabel_exclusion_accounting(
    variant_frame, stats_path: Path, binding: dict, variant_id: str,
) -> dict:
    """Reconcile the variant's censor audit columns with the bound manifest.

    Every rule here is a FAILURE, never a repair: the variant dataset is the
    production artefact and this function only decides whether it may be
    published.
    """
    import pandas as pd

    present = [
        column for column in PRELABEL_EXCLUSION_AUDIT_COLUMNS
        if column in variant_frame.columns
    ]
    if not binding["exclude_pre_label_burns"]:
        return {
            "exclude_pre_label_burns": False,
            "binding_active": False,
            "audit_columns_present": present,
            "pre_label_burn_excluded_count": None,
            "analysis_eligible_count": None,
            "manifest_cell_count": None,
            "manifest_cells_in_variant": None,
            "accounting_reconciled": True,
            "reconciliation": "policy inactive; no censor accounting is required",
        }

    if sorted(present) != sorted(PRELABEL_EXCLUSION_AUDIT_COLUMNS):
        raise WindowClosureError(
            f"BLOCKER: PRELABEL_EXCLUSION_AUDIT_MISSING -- variant "
            f"'{variant_id}' declares {PRELABEL_EXCLUSION_POLICY_FIELD}=True "
            f"but its Step8A dataset carries {present}; both of "
            f"{list(PRELABEL_EXCLUSION_AUDIT_COLUMNS)} are required together."
        )
    # Boolean semantics and the inverse relation are the production contract;
    # re-asserted here so a disagreement can never reach the model stage.
    validate_step8a_optional_audit_columns(variant_frame, frame_name=variant_id)

    excluded_mask = variant_frame["pre_label_burn_excluded"].astype(bool)
    eligible_mask = variant_frame["analysis_eligible"].astype(bool)
    if bool((eligible_mask == excluded_mask).any()):
        raise WindowClosureError(
            f"BLOCKER: PRELABEL_EXCLUSION_INVERSE_VIOLATION -- variant "
            f"'{variant_id}': analysis_eligible must equal NOT "
            "pre_label_burn_excluded on every row."
        )
    excluded_count = int(excluded_mask.sum())
    eligible_count = int(eligible_mask.sum())
    if excluded_count + eligible_count != len(variant_frame):
        raise WindowClosureError(
            f"BLOCKER: PRELABEL_EXCLUSION_ACCOUNTING_MISMATCH -- variant "
            f"'{variant_id}': excluded({excluded_count}) + "
            f"eligible({eligible_count}) != rows({len(variant_frame)})."
        )

    record = binding["documents"][PRELABEL_EXCLUSION_ROLE_MANIFEST]
    manifest_path = Path(record["path"])
    if not record["exists"] or not manifest_path.is_file():
        raise WindowClosureError(
            f"BLOCKER: PRELABEL_EXCLUSION_BINDING_MISSING -- variant "
            f"'{variant_id}' has no bound exclusion manifest at {manifest_path}."
        )
    manifest_ids = set(pd.read_parquet(manifest_path)["cell_id"].astype(str))
    variant_ids = set(variant_frame["cell_id"].astype(str))
    expected_excluded = manifest_ids & variant_ids
    observed_excluded = set(
        variant_frame.loc[excluded_mask, "cell_id"].astype(str)
    )
    if observed_excluded != expected_excluded:
        only_dataset = sorted(observed_excluded - expected_excluded)[:6]
        only_manifest = sorted(expected_excluded - observed_excluded)[:6]
        raise WindowClosureError(
            f"BLOCKER: PRELABEL_EXCLUSION_MANIFEST_DISAGREEMENT -- variant "
            f"'{variant_id}': the dataset's pre_label_burn_excluded cells do "
            "not match the bound gate manifest. Excluded in dataset only: "
            f"{only_dataset}; in manifest only: {only_manifest}."
        )

    # The production stats file keeps its own counters; when it is present they
    # must agree with the dataset, otherwise the published accounting would
    # depend on which artefact a reader opened.
    stats_counters: dict[str, Any] = {}
    if Path(stats_path).is_file():
        try:
            stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise WindowClosureError(
                f"BLOCKER: PRELABEL_EXCLUSION_ACCOUNTING_UNREADABLE -- variant "
                f"'{variant_id}': {stats_path} could not be read: {exc}."
            ) from exc
        block = stats.get("pre_label_exclusion")
        candidates = [stats, block if isinstance(block, dict) else {}]
        for counter, observed in (
            ("pre_label_burn_excluded_count", excluded_count),
            ("analysis_eligible_count", eligible_count),
        ):
            for candidate in candidates:
                if counter not in candidate:
                    continue
                stats_counters[counter] = candidate[counter]
                if int(candidate[counter]) != observed:
                    raise WindowClosureError(
                        f"BLOCKER: PRELABEL_EXCLUSION_ACCOUNTING_MISMATCH -- "
                        f"variant '{variant_id}': {stats_path.name} records "
                        f"{counter}={candidate[counter]}, dataset carries "
                        f"{observed}."
                    )
                break
    return {
        "exclude_pre_label_burns": True,
        "binding_active": True,
        "audit_columns_present": sorted(present),
        "pre_label_burn_excluded_count": excluded_count,
        "analysis_eligible_count": eligible_count,
        "variant_row_count": int(len(variant_frame)),
        "manifest_cell_count": int(len(manifest_ids)),
        "manifest_cells_in_variant": int(len(expected_excluded)),
        "manifest_path": str(manifest_path),
        "manifest_sha256": record["sha256"],
        "stats_counters": stats_counters,
        "accounting_reconciled": True,
        "reconciliation": (
            "pre_label_burn_excluded == (bound gate manifest cells INTERSECT "
            "variant cells); analysis_eligible == NOT pre_label_burn_excluded; "
            "excluded + eligible == variant rows"
        ),
    }


def scientific_configuration(
    experiment_id: str, ctx: dict, variants: Sequence[dict], censor: dict,
    inventory: dict, shifts: Sequence[int],
) -> dict:
    from core.config import (
        STEP8B_MIN_POSITIVES_PER_POPULATION, STEP8B_N_SPLITS, STEP8B_RANDOM_SEED,
        STEP8B_SPATIAL_BLOCK_SIZE_CELLS, STEP8C_N_BOOTSTRAP, STEP8C_RANDOM_SEED,
    )
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, CATEGORICAL_FEATURES, THERMAL_MODEL_FEATURES,
    )

    canonical = canonical_window(ctx)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "canonical_predictor_window": {
            "start_date": canonical["predictor_start_date"],
            "end_date": canonical["predictor_end_date"],
            "duration_days": canonical["duration_days"],
        },
        "label_window": {
            "start_date": canonical["label_start_date"],
            "end_date": canonical["label_end_date"],
            "frozen_across_variants": True,
        },
        "preregistered_shifts_days": list(shifts),
        "variants": [
            {
                "variant_id": v["variant_id"],
                "shift_days": v["shift_days"],
                "predictor_start_date": v["predictor_start_date"],
                "predictor_end_date": v["predictor_end_date"],
                "duration_days": v["duration_days"],
                "lead_days": v["lead_days"],
            }
            for v in variants
        ],
        "baseline_years": list(canonical["baseline_years"]),
        "feature_registry": {
            "baseline_features_in_order": list(BASELINE_FEATURES),
            "thermal_model_features_in_order": list(THERMAL_MODEL_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "source": "src.step8b_train_baseline_vs_thermal_model",
        },
        "primary_population": PRIMARY_POPULATION,
        "common_censor_interval": censor,
        "common_cohort_rule": (
            "exact cell_id intersection of the analysis-eligible, primary "
            "population, valid-grid rows of every variant, after removing the "
            "shared pre-label censored cells"
        ),
        "model_configuration": {
            "model": PRIMARY_MODEL,
            "n_splits": STEP8B_N_SPLITS,
            "random_seed": STEP8B_RANDOM_SEED,
            "spatial_block_size_cells": STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
            "min_positives": STEP8B_MIN_POSITIVES_PER_POPULATION,
            "fold_assignment": "single shared assignment reused by every variant",
        },
        "bootstrap_configuration": {
            "unit": "spatial_block_id",
            "n_bootstrap": STEP8C_N_BOOTSTRAP,
            "seed": STEP8C_RANDOM_SEED,
            "identical_block_draws_across_variants": True,
        },
        "frozen_input_sha256": {
            name: entry["sha256"] for name, entry in sorted(inventory.items())
        },
        # Both label hashes are pinned EXPLICITLY as well, so the analysis
        # identity visibly depends on the exact label rasters and not only on
        # whatever happens to be in the inventory. Only role names and hashes
        # enter here -- never paths, which would make the id host-dependent.
        "required_label_roles": list(REQUIRED_LABEL_ROLES),
        "label_input_sha256": {
            role: (inventory.get(role) or {}).get("sha256")
            for role in sorted(REQUIRED_LABEL_ROLES)
        },
        "reducer": "scene_weighted",
        "git_commit": _git_commit(),
    }


def compute_analysis_id(config: dict) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))


# =============================================================================
# Common cohort across N variants
# =============================================================================
def censored_cell_ids(censor_frame) -> set:
    """Cell IDs excluded by the shared pre-label censoring raster."""
    if censor_frame is None or len(censor_frame) == 0:
        return set()
    if "cell_id" not in getattr(censor_frame, "columns", []):
        raise WindowClosureError("Pre-label censor table must carry 'cell_id'.")
    return set(censor_frame["cell_id"].tolist())


def variant_eligible_rows(df, censored: Optional[set] = None):
    """Analysis-eligible, primary-population, valid-grid rows for one variant."""
    import pandas as pd  # noqa: F401

    required = {"cell_id", "row_500m", "col_500m", "burned", PRIMARY_POPULATION}
    missing = sorted(required - set(df.columns))
    if missing:
        raise WindowClosureError(f"Variant Step8A frame is missing column(s): {missing}.")

    mask = df[PRIMARY_POPULATION].astype(bool)
    if "valid_for_modeling" in df.columns:
        mask &= df["valid_for_modeling"].astype(bool)
    if "analysis_eligible" in df.columns:
        mask &= df["analysis_eligible"].astype(bool)
    mask &= df["row_500m"].notna() & df["col_500m"].notna()
    rows = df.loc[mask]
    if censored:
        rows = rows.loc[~rows["cell_id"].isin(censored)]
    return rows.sort_values("cell_id", kind="mergesort").reset_index(drop=True)


def build_common_cohort(
    frames_by_variant: dict, censored: Optional[set] = None,
) -> dict:
    """Exact N-way cell_id intersection with full cross-variant equality checks.

    `landsat_composite_downstream_ab.build_common_cohort` handles exactly two
    chains; this analysis compares three or more variants, so the same contract
    is applied N-way here rather than chaining pairwise intersections.
    """
    import numpy as np

    if len(frames_by_variant) < 2:
        raise WindowClosureError("A common cohort needs at least two variants.")

    native = {
        name: variant_eligible_rows(frame, censored)
        for name, frame in frames_by_variant.items()
    }
    for name, rows in native.items():
        if rows["cell_id"].duplicated().any():
            raise WindowClosureError(f"Variant '{name}' has duplicate cell_id values.")
        if rows.duplicated(subset=["row_500m", "col_500m"]).any():
            raise WindowClosureError(f"Variant '{name}' has duplicate (row_500m, col_500m) cells.")

    common_ids = None
    for rows in native.values():
        ids = set(rows["cell_id"].tolist())
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = sorted(common_ids or set())
    if not common_ids:
        raise WindowClosureError("The common cohort is empty; no comparison is possible.")

    common = {
        name: rows[rows["cell_id"].isin(common_ids)]
              .sort_values("cell_id", kind="mergesort").reset_index(drop=True)
        for name, rows in native.items()
    }

    order = sorted(common)
    anchor_name = order[0]
    anchor = common[anchor_name]
    for name in order[1:]:
        other = common[name]
        if not np.array_equal(anchor["cell_id"].to_numpy(), other["cell_id"].to_numpy()):
            raise WindowClosureError(
                f"Common cohort cell_id order differs between '{anchor_name}' and '{name}'."
            )
        if not np.array_equal(anchor["burned"].to_numpy(), other["burned"].to_numpy()):
            raise WindowClosureError(
                f"Labels differ on the common cohort between '{anchor_name}' and '{name}'; "
                "the label window must be frozen."
            )
        for column in ("row_500m", "col_500m"):
            if not np.array_equal(anchor[column].to_numpy(), other[column].to_numpy()):
                raise WindowClosureError(
                    f"'{column}' differs on the common cohort between "
                    f"'{anchor_name}' and '{name}'."
                )
        if not np.array_equal(
            anchor[PRIMARY_POPULATION].astype(bool).to_numpy(),
            other[PRIMARY_POPULATION].astype(bool).to_numpy(),
        ):
            raise WindowClosureError(
                f"Primary-population membership differs between '{anchor_name}' and '{name}'."
            )

    labels = anchor["burned"].astype(int).to_numpy()
    n_positive = int(labels.sum())
    if len(set(labels.tolist())) < 2:
        raise WindowClosureError(
            "The common cohort carries a single class; a model comparison is not possible."
        )
    from core.config import STEP8B_MIN_POSITIVES_PER_POPULATION
    if n_positive < STEP8B_MIN_POSITIVES_PER_POPULATION:
        raise WindowClosureError(
            f"The common cohort has {n_positive} positives, below the frozen "
            f"Step8B minimum of {STEP8B_MIN_POSITIVES_PER_POPULATION}."
        )

    summary = {
        "population": PRIMARY_POPULATION,
        "common_rows": len(anchor),
        "common_positives": n_positive,
        "pre_label_censored_cells": int(len(censored or set())),
        "per_variant": {},
    }
    for name in order:
        native_rows = len(native[name])
        native_positives = int(native[name]["burned"].astype(int).sum())
        summary["per_variant"][name] = {
            "native_eligible_rows": native_rows,
            "native_positive_rows": native_positives,
            "common_rows": len(common[name]),
            "common_positive_rows": n_positive,
            "common_row_retention": len(common[name]) / native_rows if native_rows else None,
            "common_positive_retention": (
                n_positive / native_positives if native_positives else None
            ),
            "dropped_rows": native_rows - len(common[name]),
        }
    return {"common": common, "native": native, "common_cell_ids": common_ids, "summary": summary}


# =============================================================================
# Metrics, changes and interval language
# =============================================================================
def thermal_contribution(result: dict) -> dict:
    """Signed thermal contribution: positive always favours the thermal model."""
    baseline, thermal = result["baseline"], result["thermal"]
    return {
        "baseline_roc_auc": baseline["roc_auc"],
        "baseline_pr_auc": baseline["pr_auc"],
        "baseline_brier": baseline["brier"],
        "thermal_roc_auc": thermal["roc_auc"],
        "thermal_pr_auc": thermal["pr_auc"],
        "thermal_brier": thermal["brier"],
        "delta_roc_auc": thermal["roc_auc"] - baseline["roc_auc"],
        "delta_pr_auc": thermal["pr_auc"] - baseline["pr_auc"],
        "brier_improvement": baseline["brier"] - thermal["brier"],
    }


PAIRED_CHANGE_METRICS: tuple[str, ...] = (
    "baseline_roc_auc", "baseline_pr_auc", "baseline_brier",
    "thermal_roc_auc", "thermal_pr_auc", "thermal_brier",
    "delta_roc_auc", "delta_pr_auc", "brier_improvement",
)


def paired_window_change(canonical_metrics: dict, variant_metrics: dict) -> dict:
    """change = earlier_closure - canonical, for every reported metric."""
    return {
        metric: variant_metrics[metric] - canonical_metrics[metric]
        for metric in PAIRED_CHANGE_METRICS
    }


def classify_change_interval(low: Optional[float], high: Optional[float]) -> str:
    """Interval language. Never 'significant', never 'equivalent'/'stable'.

    An interval containing zero is reported as exactly that -- it is NOT
    evidence of equivalence, because no equivalence margin is preregistered.
    """
    if low is None or high is None:
        return INTERVAL_INCLUDES_ZERO
    if low > 0.0:
        return INTERVAL_SUPPORTED_INCREASE
    if high < 0.0:
        return INTERVAL_SUPPORTED_DECREASE
    return INTERVAL_INCLUDES_ZERO


def multi_variant_block_bootstrap(
    cohort_df, labels, probabilities_by_variant: dict, *,
    n_bootstrap: Optional[int] = None, seed: Optional[int] = None,
    ci_lower: float = 2.5, ci_upper: float = 97.5,
) -> dict:
    """One set of spatial-block draws, scored for EVERY variant.

    `landsat_composite_downstream_ab.paired_block_bootstrap` is fixed at two
    chains, so the same canonical primitives (`build_block_index`,
    `compute_metrics` from Step8C) are reused here for N variants. Because a
    replicate draws blocks ONCE and scores all variants on exactly those rows,
    the earlier-minus-canonical changes are properly paired.
    """
    import numpy as np
    import pandas as pd

    from core.config import STEP8C_N_BOOTSTRAP, STEP8C_RANDOM_SEED
    from src.step8c_spatial_block_bootstrap_uncertainty import build_block_index, compute_metrics

    n_bootstrap = STEP8C_N_BOOTSTRAP if n_bootstrap is None else int(n_bootstrap)
    seed = STEP8C_RANDOM_SEED if seed is None else int(seed)

    frame = pd.DataFrame({
        "spatial_block_id": cohort_df["spatial_block_id"].to_numpy(),
        "burned": np.asarray(labels, dtype=int),
    })
    for variant, probs in sorted(probabilities_by_variant.items()):
        frame[f"p_baseline__{variant}"] = np.asarray(probs["baseline"], dtype="float64")
        frame[f"p_thermal__{variant}"] = np.asarray(probs["thermal"], dtype="float64")

    unique_blocks, block_to_idx, sub = build_block_index(frame)
    if len(unique_blocks) < 2:
        raise WindowClosureError(
            "The paired bootstrap needs at least two spatial blocks; a random "
            "row bootstrap is not an acceptable substitute."
        )

    rng = np.random.default_rng(seed)
    y_sub = sub["burned"].to_numpy()
    variants = sorted(probabilities_by_variant)
    records: list[dict] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_idx[b] for b in sampled])
        y_rep = y_sub[idx]
        if len(set(y_rep.tolist())) < 2:
            continue
        row: dict[str, Any] = {}
        ok = True
        for variant in variants:
            # Step8C's compute_metrics scores baseline and thermal together and
            # already returns the deltas, so the metric definitions here are
            # exactly the canonical ones.
            metrics = compute_metrics(
                y_rep,
                sub[f"p_baseline__{variant}"].to_numpy()[idx],
                sub[f"p_thermal__{variant}"].to_numpy()[idx],
            )
            if metrics is None:
                ok = False
                break
            row[f"{variant}__baseline_roc_auc"] = metrics["auc_baseline"]
            row[f"{variant}__baseline_pr_auc"] = metrics["pr_auc_baseline"]
            row[f"{variant}__baseline_brier"] = metrics["brier_baseline"]
            row[f"{variant}__thermal_roc_auc"] = metrics["auc_thermal"]
            row[f"{variant}__thermal_pr_auc"] = metrics["pr_auc_thermal"]
            row[f"{variant}__thermal_brier"] = metrics["brier_thermal"]
            row[f"{variant}__delta_roc_auc"] = metrics["delta_auc"]
            row[f"{variant}__delta_pr_auc"] = metrics["delta_pr_auc"]
            # BOTH conventions are recorded, neither silently:
            #   delta_brier        = thermal - baseline (RAW; negative is
            #                        better, because a lower Brier is better);
            #   brier_improvement  = -(delta_brier) (improvement-oriented,
            #                        positive favours thermal).
            # The model stage reports the RAW delta; the plan-stage summary
            # contract keeps the improvement-oriented field it froze.
            row[f"{variant}__delta_brier"] = metrics["delta_brier"]
            row[f"{variant}__brier_improvement"] = -metrics["delta_brier"]
        if ok:
            records.append(row)

    replicates = pd.DataFrame(records)
    return {
        "bootstrap_unit": "spatial_block_id",
        "n_blocks": int(len(unique_blocks)),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(len(replicates)),
        "seed": int(seed),
        "ci_lower_percentile": float(ci_lower),
        "ci_upper_percentile": float(ci_upper),
        "identical_block_draws_across_variants": True,
        "variants": variants,
        "replicates": replicates,
    }


def percentile_interval(values, ci_lower: float, ci_upper: float) -> dict:
    import numpy as np

    array = np.asarray([v for v in values if v is not None], dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"point": None, "ci_low": None, "ci_high": None, "n_replicates": 0}
    return {
        "point": float(np.mean(array)),
        "ci_low": float(np.percentile(array, ci_lower)),
        "ci_high": float(np.percentile(array, ci_upper)),
        "n_replicates": int(array.size),
    }


def validate_saved_bootstrap_replicate_counts(
    requested_replicates: Any,
    valid_replicates: Any,
    invalid_replicates: Any,
    replicate_row_count: int,
) -> tuple[int, int]:
    """Validate the model artifact's global shared-draw count contract.

    The replicate artifact has one row per globally valid shared draw. Invalid
    draws are omitted, so there is no separate identity, validity or reason
    column to group. Every comparison summary row therefore carries the same
    global counts.
    """
    try:
        requested = int(requested_replicates)
        valid = int(valid_replicates)
        invalid = int(invalid_replicates)
        rows = int(replicate_row_count)
    except (TypeError, ValueError) as exc:
        raise WindowClosureError(
            "Bootstrap replicate counts must be present integers."
        ) from exc
    if min(requested, valid, invalid, rows) < 0:
        raise WindowClosureError("Bootstrap replicate counts cannot be negative.")
    if valid != rows:
        raise WindowClosureError(
            "The recorded valid replicate count does not match the replicate table."
        )
    if invalid != requested - valid:
        raise WindowClosureError("The recorded invalid replicate count is not truthful.")
    return valid, invalid


def build_paired_change_rows(bootstrap: dict, point_metrics: dict) -> list[dict]:
    """Earlier-minus-canonical change rows with paired bootstrap intervals."""
    replicates = bootstrap["replicates"]
    rows: list[dict] = []
    for variant in bootstrap["variants"]:
        if variant == CANONICAL_VARIANT_ID:
            continue
        for metric in PAIRED_CHANGE_METRICS:
            canonical_column = f"{CANONICAL_VARIANT_ID}__{metric}"
            variant_column = f"{variant}__{metric}"
            if canonical_column in replicates.columns and variant_column in replicates.columns:
                differences = (replicates[variant_column] - replicates[canonical_column]).tolist()
            else:
                differences = []
            interval = percentile_interval(
                differences, bootstrap["ci_lower_percentile"], bootstrap["ci_upper_percentile"],
            )
            point = (
                point_metrics[variant][metric] - point_metrics[CANONICAL_VARIANT_ID][metric]
                if variant in point_metrics and CANONICAL_VARIANT_ID in point_metrics else None
            )
            rows.append({
                "variant_id": variant,
                "metric": metric,
                "change_definition": "earlier_closure_minus_canonical",
                "point_estimate": point,
                "bootstrap_mean": interval["point"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "valid_replicates": interval["n_replicates"],
                "interval_status": classify_change_interval(interval["ci_low"], interval["ci_high"]),
            })
    rows.sort(key=lambda r: (r["variant_id"], r["metric"]))
    return rows


# =============================================================================
# Rendering
# =============================================================================
def render_summary_markdown(summary: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Window-closure sensitivity — `{summary['experiment_id']}`")
    add("")
    add(f"- Schema: `{summary['schema_version']}`")
    add(f"- analysis_id: `{summary['analysis_id']}`")
    add(f"- Primary population: `{summary['primary_population']}`")
    add(f"- Primary model: `{summary['primary_model']}`")
    add("")
    add("## Predictor windows")
    add("")
    add("| Variant | Shift (days) | Predictor start | Predictor end | Duration | Lead to label start |")
    add("|---|---:|---|---|---:|---:|")
    for variant in summary["variants"]:
        add(
            f"| {variant['variant_id']} | {variant['shift_days']} | "
            f"{variant['predictor_start_date']} | {variant['predictor_end_date']} | "
            f"{variant['duration_days']} | {variant['lead_days']} |"
        )
    add("")
    add(f"The label window `{summary['label_window']['start_date']} .. "
        f"{summary['label_window']['end_date']}` is frozen and identical in every "
        "variant; only predictor timing moves, and every window keeps the same length.")
    add("")
    transparency = summary.get("calendar_month_filter_transparency")
    if isinstance(transparency, dict) and "calendar_month_filter" not in transparency:
        # Per-variant mapping: every variant shares the same window length, so
        # the filter behaviour is identical; report the canonical one.
        transparency = transparency.get(CANONICAL_VARIANT_ID) or next(
            iter(transparency.values()), None
        )
    if isinstance(transparency, dict) and transparency.get("calendar_month_filter_applied"):
        add("## Calendar-month filter")
        add("")
        add(f"- Filter: `{transparency['calendar_month_filter']}` "
            f"(production-derived, reported unchanged)")
        add(f"- Redundant: **{str(transparency['calendar_month_filter_redundant']).lower()}**")
        add(f"- Exact `filterDate` is binding: "
            f"**{str(transparency['exact_filter_date_is_binding']).lower()}**")
        add("")
        add(transparency["note"])
        add("")
    censor = summary["common_censor_interval"]
    add("## Shared pre-label censoring")
    add("")
    add(f"Cells burning in `{censor['common_prelabel_start']} .. "
        f"{censor['common_prelabel_end']}` are removed from the analysis cohort of "
        "EVERY variant. They are not label-window positives, and leaving them as "
        "negatives would misstate the label.")
    add("")
    cohort = summary.get("common_cohort_summary") or {}
    if cohort:
        add("## Common cohort")
        add("")
        add(f"- Rows: **{cohort.get('common_rows')}**")
        add(f"- Positives: **{cohort.get('common_positives')}**")
        add(f"- Pre-label censored cells: **{cohort.get('pre_label_censored_cells')}**")
        add("")
    if summary.get("variant_metrics"):
        add("## Metrics on the common cohort")
        add("")
        add("| Variant | baseline ROC-AUC | thermal ROC-AUC | Δ ROC-AUC | Δ PR-AUC | Brier improvement |")
        add("|---|---:|---:|---:|---:|---:|")
        for row in summary["variant_metrics"]:
            add(
                f"| {row['variant_id']} | {_num(row['baseline_roc_auc'])} | "
                f"{_num(row['thermal_roc_auc'])} | {_num(row['delta_roc_auc'])} | "
                f"{_num(row['delta_pr_auc'])} | {_num(row['brier_improvement'])} |"
            )
        add("")
    if summary.get("paired_changes"):
        add("## Change relative to the canonical window")
        add("")
        add("`change = earlier_closure - canonical`, with paired spatial-block "
            "bootstrap percentile intervals computed on identical block draws.")
        add("")
        add("| Variant | Metric | Change | CI low | CI high | Replicates | Interval |")
        add("|---|---|---:|---:|---:|---:|---|")
        for row in summary["paired_changes"]:
            add(
                f"| {row['variant_id']} | {row['metric']} | {_num(row['point_estimate'])} | "
                f"{_num(row['ci_low'])} | {_num(row['ci_high'])} | "
                f"{row['valid_replicates']} | {row['interval_status']} |"
            )
        add("")
    add("## Interpretation boundary")
    add("")
    for limitation in LIMITATIONS:
        add(f"- {limitation}")
    add("")
    add("An interval that includes zero means the data do not resolve the "
        "direction of the change; uncertainty remains about that direction.")
    add("")
    return "\n".join(lines)


def _num(value) -> str:
    return "—" if value is None else f"{value:.6f}"


def assert_report_wording(markdown: str) -> None:
    lowered = markdown.lower()
    found = sorted(p for p in BANNED_REPORT_PHRASES if p in lowered)
    if found:
        raise WindowClosureError(f"Report contains banned wording: {found}.")


# =============================================================================
# Stages
# =============================================================================
def validate_stage_range(from_stage: str, to_stage: str) -> list[str]:
    for name, value in (("--from-stage", from_stage), ("--to-stage", to_stage)):
        if value not in STAGES:
            # ONE consistent contract for a stage this build does not know:
            # the same "not enabled" wording an unimplemented-but-declared
            # stage would produce, so a caller never has to distinguish
            # "unknown" from "declared but not built". This runs first in
            # `run_analysis`, before any prerequisite, exporter, engine, mkdir
            # or write.
            raise WindowClosureError(
                f"{name}={value!r}: stage {value!r} is not enabled or "
                f"recognized in this build. Known stages are {list(STAGES)}. "
                "Nothing was created."
            )
    start, end = STAGES.index(from_stage), STAGES.index(to_stage)
    if start > end:
        raise WindowClosureError(
            f"--from-stage ({from_stage}) must not come after --to-stage ({to_stage})."
        )
    return list(STAGES[start:end + 1])


def assert_stage_prerequisites(stages: Sequence[str]) -> None:
    """A stage may only run once its inputs exist in this plan or already on disk."""
    planned = list(stages)
    for stage in planned:
        for required in STAGE_REQUIRES[stage]:
            if STAGES.index(required) >= STAGES.index(planned[0]):
                if required not in planned:
                    raise WindowClosureError(
                        f"Stage '{stage}' requires '{required}', which is neither "
                        "in the selected range nor already completed."
                    )


# =============================================================================
# Output layout
# =============================================================================
def plan_output_paths(
    experiment_id: str, variants: Sequence[dict], output_root: Optional[Path] = None,
) -> dict[str, str]:
    root = experiment_root(experiment_id, output_root)
    paths: dict[str, str] = {
        "config/preregistration.json": str(root / "config" / "preregistration.json"),
        "config/window_variants.csv": str(root / "config" / "window_variants.csv"),
        "config/frozen_input_inventory.json": str(root / "config" / "frozen_input_inventory.json"),
        "prelabel_censor/export_plan.json": str(root / "prelabel_censor" / "export_plan.json"),
        "prelabel_censor/prelabel_burndate.tif": str(root / "prelabel_censor" / "prelabel_burndate.tif"),
        "prelabel_censor/censoring_summary.json": str(root / "prelabel_censor" / "censoring_summary.json"),
    }
    for variant in variants:
        vroot = root / "variants" / variant["variant_id"]
        if variant["is_canonical"]:
            paths[f"variants/{variant['variant_id']}/frozen_reference.json"] = str(
                vroot / "frozen_reference.json"
            )
            continue
        paths[f"variants/{variant['variant_id']}/export_plan.json"] = str(vroot / "export_plan.json")
        paths[f"variants/{variant['variant_id']}/predictor_export_metadata.json"] = str(
            vroot / "predictor_export_metadata.json"
        )
        for step in ("data", "step5", "step5c", "step7b", "step7c", "step7d", "step7e", "step8a", "step8"):
            paths[f"variants/{variant['variant_id']}/{step}/"] = str(vroot / step)
    for name in (
        "common_cohort.parquet", "common_cohort_summary.json",
        "shared_fold_assignments.parquet", "variant_metrics.csv",
        "thermal_contribution.csv", "paired_window_changes.csv",
        "bootstrap_replicates.parquet", "window_closure_summary.json",
        "window_closure_summary.md", "manifest.json",
    ):
        paths[f"comparison/{name}"] = str(root / "comparison" / name)
    return dict(sorted(paths.items()))


# =============================================================================
# Actual PLAN stage: preregistration and export-plan documents
#
# The FIRST of the implemented actual stages (see IMPLEMENTED_ACTUAL_STAGES).
# It writes deterministic JSON and CSV documents into the dedicated
# diagnostics namespace and does nothing else: no Earth Engine
# import/query/export, no predictor, no raster, no parquet, no model, no
# bootstrap. Nothing is ever deleted.
# =============================================================================
PLAN_DOCUMENT_SUFFIXES: tuple[str, ...] = (".json", ".csv")

WINDOW_VARIANTS_CSV_COLUMNS: tuple[str, ...] = (
    "variant_id", "shift_days", "predictor_start_date", "predictor_end_date",
    "duration_days", "duration_preserved", "label_start_date", "label_end_date",
    "label_window_unchanged", "lead_days", "is_canonical",
)


def assert_actual_stages_supported(stages: Sequence[str]) -> None:
    """Only the implemented actual stages may run outside a dry run."""
    blocked = [stage for stage in stages if stage not in IMPLEMENTED_ACTUAL_STAGES]
    if not blocked:
        return
    raise WindowClosureError(
        "Live window-closure execution is not enabled in this build for "
        f"stage(s) {blocked}. Any stage outside "
        f"({[s for s in STAGES if s not in IMPLEMENTED_ACTUAL_STAGES]}) must be "
        "explicitly implemented and reviewed before any real run. Only "
        f"{list(IMPLEMENTED_ACTUAL_STAGES)} are implemented for an actual run: "
        "'plan' writes the preregistration and export-plan documents, "
        "'prelabel-export' exports the single shared pre-label BurnDate "
        "raster, 'predictor-export' rebuilds the Landsat/MODIS predictors of "
        "every non-canonical variant, 'local-downstream' runs the production "
        "Step5/Step5C/Step7/Step8A chain on those predictors inside the "
        "variant namespace, 'model' fits the production baseline/thermal "
        "fire-risk models on one exact common cohort, and 'compare' "
        "summarises the verified model outputs read-only. Run one stage at a "
        "time with matching --from-stage/--to-stage, or --dry-run to inspect "
        "the full plan."
    )


def assert_resume_force_exclusive(resume: bool, force: bool) -> None:
    """`--resume` reuses what exists; `--force` replaces it. Never both."""
    if resume and force:
        raise WindowClosureError(
            "--resume and --force are mutually exclusive: --resume reuses a "
            "valid existing output, --force replaces it. Choose one."
        )


def actual_plan_prerequisites(inventory: dict) -> dict:
    """Every frozen input the actual plan pins must exist and hash.

    Wider than `label_prerequisites`: an actual preregistration also pins the
    frozen canonical Step8A dataset and the static shared rasters, so a null
    hash for any of them would silently unpin the analysis identity.
    """
    missing: list[dict] = []
    for role in REQUIRED_FROZEN_INPUT_ROLES:
        entry = inventory.get(role) or {}
        if not entry.get("exists") or entry.get("sha256") is None:
            missing.append({
                "role": role,
                "path": entry.get("path"),
                "exists": bool(entry.get("exists")),
                "sha256": entry.get("sha256"),
                "reason": (
                    "required frozen input does not exist and therefore cannot "
                    "be hashed"
                ),
            })
    return {
        "required_frozen_input_roles": list(REQUIRED_FROZEN_INPUT_ROLES),
        "prerequisites_ready": not missing,
        "missing_required_inputs": missing,
    }


def assert_actual_plan_prerequisites(inventory: dict) -> dict:
    """Fail fast, before any directory or file is created."""
    status = actual_plan_prerequisites(inventory)
    if status["prerequisites_ready"]:
        return status
    roles = ", ".join(entry["role"] for entry in status["missing_required_inputs"])
    paths = "; ".join(str(entry["path"]) for entry in status["missing_required_inputs"])
    raise WindowClosureError(
        f"Required frozen input(s) missing: {roles}. Expected at: {paths}. "
        "The actual plan pins every frozen input hash, so no preregistration "
        "is written while any of them is absent."
    )


def frozen_hash_map(inventory: dict) -> dict[str, Optional[str]]:
    return {name: entry.get("sha256") for name, entry in sorted(inventory.items())}


def assert_frozen_hashes_unchanged(before: dict, after: dict, when: str) -> None:
    """A frozen input that moved under the plan invalidates the plan."""
    changed = sorted(
        name for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    )
    if changed:
        raise WindowClosureError(
            f"Frozen input hash(es) changed {when}: {changed}. The analysis "
            "identity pins these inputs, so the plan cannot be trusted; "
            "nothing further is written."
        )


# --- Document construction (pure) -------------------------------------------
def _json_document(payload: dict) -> str:
    """Deterministic, key-sorted JSON text."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"


def _csv_document(columns: Sequence[str], rows: Sequence[dict]) -> str:
    """Deterministic CSV text: fixed column order, caller-fixed row order."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return buffer.getvalue()


def _variant_planned_paths(variant_id: str, planned_paths: dict[str, str]) -> dict[str, str]:
    prefix = f"variants/{variant_id}/"
    return {key: value for key, value in sorted(planned_paths.items()) if key.startswith(prefix)}


def build_plan_documents(
    experiment_id: str,
    analysis_id: str,
    config: dict,
    canonical: dict,
    variants: Sequence[dict],
    censor: dict,
    inventory: dict,
    labels: dict,
    export_plans: dict,
    planned_paths: dict[str, str],
    output_root: Optional[Path] = None,
) -> dict[str, str]:
    """Every plan-owned document, keyed by its planned relative path.

    The canonical variant gets a frozen-reference record ONLY: it reads the
    frozen production outputs, so planning a predictor export for it would
    invent work that must not happen. Early variants are derived from the
    preregistered non-zero shifts, so no date and no AOI is hard-coded.
    """
    label_records = {
        role: {
            "role": role,
            "path": str(entry["path"]),
            "resolved_from": entry["resolved_from"],
            "canonical_filename": entry["canonical_filename"],
            "exists": inventory[role]["exists"],
            "sha256": inventory[role]["sha256"],
        }
        for role, entry in sorted(labels.items())
    }
    no_side_effects = {
        "gee_queries_run": False,
        "gee_exports_run": False,
        "model_fit": False,
        "bootstrap_run": False,
    }

    documents: dict[str, str] = {}

    documents["config/preregistration.json"] = _json_document({
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "stage": "plan",
        "diagnostic_namespace": DIAGNOSTIC_NAMESPACE,
        "written_by": "src/window_closure_sensitivity.py:run_analysis",
        "scientific_configuration": config,
        "canonical_window": {
            "predictor_start_date": canonical["predictor_start_date"],
            "predictor_end_date": canonical["predictor_end_date"],
            "duration_days": canonical["duration_days"],
            "lead_days": canonical["lead_days"],
        },
        "label_window": {
            "start_date": canonical["label_start_date"],
            "end_date": canonical["label_end_date"],
            "frozen_across_variants": True,
        },
        "variants": list(variants),
        "common_censor_interval": censor,
        "label_inputs": label_records,
        "frozen_input_sha256": frozen_hash_map(inventory),
        "prerequisites_ready": True,
        "planned_output_paths": dict(sorted(planned_paths.items())),
        "limitations": list(LIMITATIONS),
        **no_side_effects,
    })

    documents["config/window_variants.csv"] = _csv_document(
        WINDOW_VARIANTS_CSV_COLUMNS,
        sorted(variants, key=lambda v: int(v["shift_days"])),
    )

    documents["config/frozen_input_inventory.json"] = _json_document({
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "schema_version": SCHEMA_VERSION,
        "required_frozen_input_roles": list(REQUIRED_FROZEN_INPUT_ROLES),
        "required_label_roles": list(REQUIRED_LABEL_ROLES),
        "inventory": dict(sorted(inventory.items())),
        "frozen_input_sha256": frozen_hash_map(inventory),
        "label_inputs": label_records,
        "read_only": True,
    })

    prelabel = prelabel_export_plan(experiment_id, censor, output_root)
    documents["prelabel_censor/export_plan.json"] = _json_document({
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "schema_version": SCHEMA_VERSION,
        "common_prelabel_start": censor["common_prelabel_start"],
        "common_prelabel_end": censor["common_prelabel_end"],
        "derivation": censor["derivation"],
        "producer": prelabel["producer"],
        "planned_raster_path": prelabel["raster_path"],
        "output_dir": prelabel["output_dir"],
        "applies_to_all_variants": True,
        "independent_of_exclude_pre_label_burns_flag": True,
        "writes_into_canonical_namespace": False,
        "canonical_gate_rerun": False,
        "note": prelabel["note"],
        **no_side_effects,
    })

    for variant in variants:
        variant_id = variant["variant_id"]
        if variant["is_canonical"]:
            documents[f"variants/{variant_id}/frozen_reference.json"] = _json_document({
                "analysis_id": analysis_id,
                "experiment_id": experiment_id,
                "schema_version": SCHEMA_VERSION,
                "variant_id": variant_id,
                "shift_days": variant["shift_days"],
                "is_canonical": True,
                "predictor_start_date": variant["predictor_start_date"],
                "predictor_end_date": variant["predictor_end_date"],
                "duration_days": variant["duration_days"],
                "lead_days": variant["lead_days"],
                "label_start_date": variant["label_start_date"],
                "label_end_date": variant["label_end_date"],
                # The canonical variant is READ, never re-exported: it is the
                # frozen production result the early closures are compared to.
                "reads_frozen_production_outputs": True,
                "predictor_export_planned": False,
                "landsat_export_planned": False,
                "modis_export_planned": False,
                "frozen_canonical_step8a": {
                    "path": inventory["canonical_step8a"]["path"],
                    "sha256": inventory["canonical_step8a"]["sha256"],
                },
                "static_shared_roles": list(STATIC_SHARED_ROLES),
                **no_side_effects,
            })
            continue

        plan = export_plans[variant_id]
        landsat, modis = plan["landsat"], plan["modis"]
        documents[f"variants/{variant_id}/export_plan.json"] = _json_document({
            "analysis_id": analysis_id,
            "experiment_id": experiment_id,
            "schema_version": SCHEMA_VERSION,
            "variant_id": variant_id,
            "shift_days": variant["shift_days"],
            "is_canonical": False,
            "predictor_start_date": variant["predictor_start_date"],
            "predictor_end_date": variant["predictor_end_date"],
            "duration_days": variant["duration_days"],
            "duration_preserved": variant["duration_preserved"],
            "lead_days": variant["lead_days"],
            "label_start_date": variant["label_start_date"],
            "label_end_date": variant["label_end_date"],
            "label_window_unchanged": True,
            "landsat": {
                "window_days": landsat["window_days"],
                "current_window": landsat["current_window"],
                "calendar_month_filter_transparency":
                    landsat["calendar_month_filter_transparency"],
                "current_roles": [
                    role for role in landsat["roles"] if role["scope"] == "current_window"
                ],
                "baseline_roles": [
                    role for role in landsat["roles"] if role["scope"] == "baseline_year"
                ],
                "role_count": landsat["role_count"],
                "qa_mask_provenance": landsat["qa_mask_provenance"],
                "reducer": landsat["reducer"],
                "reducer_note": landsat["reducer_note"],
            },
            "modis": modis,
            "reducer": "scene_weighted",
            "static_shared": plan["static_shared"],
            "planned_output_paths": _variant_planned_paths(variant_id, planned_paths),
            **no_side_effects,
        })

    return documents


# --- Writing (the only side effect in this module) ---------------------------
def assert_plan_owned_targets(
    experiment_id: str, targets: dict[str, Path], output_root: Optional[Path] = None,
) -> None:
    """Every write target must be a plan-owned JSON/CSV inside this namespace."""
    root = experiment_root(experiment_id, output_root).resolve()
    for relative, path in targets.items():
        resolved = Path(path).resolve()
        if root not in resolved.parents:
            raise WindowClosureError(
                f"Plan document '{relative}' escapes the window-closure "
                f"namespace: {resolved}."
            )
        if resolved.suffix not in PLAN_DOCUMENT_SUFFIXES:
            raise WindowClosureError(
                f"Plan document '{relative}' is not a plan-owned "
                f"{list(PLAN_DOCUMENT_SUFFIXES)} file: {resolved}. The plan "
                "stage never writes rasters, tables or models."
            )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def existing_plan_analysis_id(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("analysis_id")
    except (OSError, ValueError, UnicodeDecodeError, AttributeError):
        return None


def write_plan_documents(
    experiment_id: str,
    analysis_id: str,
    documents: dict[str, str],
    planned_paths: dict[str, str],
    output_root: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Idempotently write the plan-owned documents. Nothing is ever deleted.

    * a fresh namespace is written normally;
    * a re-run under the SAME analysis_id verifies the documents and rewrites
      only those whose bytes differ, so `reused` is True when the namespace
      already matched;
    * a DIFFERENT analysis_id refuses unless `force`, which may overwrite the
      plan-owned JSON/CSV documents and nothing else. No raster, table, model,
      canonical output or foreign diagnostics file is ever touched.
    """
    missing_from_layout = sorted(set(documents) - set(planned_paths))
    if missing_from_layout:
        raise WindowClosureError(
            f"Plan document(s) {missing_from_layout} have no entry in the "
            "planned output layout; the layout is the single source of truth."
        )
    targets = {relative: Path(planned_paths[relative]) for relative in documents}
    assert_plan_owned_targets(experiment_id, targets, output_root)

    prereg = targets["config/preregistration.json"]
    previous_id = existing_plan_analysis_id(prereg)
    if previous_id is not None and previous_id != analysis_id and not force:
        raise WindowClosureError(
            f"{prereg} already holds a DIFFERENT analysis_id ({previous_id}) "
            f"than the one now planned ({analysis_id}). Refusing to overwrite a "
            "preregistration: re-run with force=True to replace the plan "
            "documents, or use a clean namespace."
        )

    written: list[str] = []
    rewritten: list[str] = []
    for relative in sorted(documents):
        path, text = targets[relative], documents[relative]
        already = path.is_file() and path.read_text(encoding="utf-8") == text
        if not already:
            _atomic_write_text(path, text)
            rewritten.append(str(path))
        written.append(str(path))

    # Plan-owned documents that this run does NOT own (e.g. a variant from an
    # earlier, differently preregistered shift set). They are reported, never
    # deleted or modified.
    owned = {str(path.resolve()) for path in targets.values()}
    root = experiment_root(experiment_id, output_root)
    unmanaged = sorted(
        str(path) for path in root.rglob("*")
        if path.is_file() and path.suffix in PLAN_DOCUMENT_SUFFIXES
        and str(path.resolve()) not in owned
    )
    return {
        "files_written": sorted(written),
        "files_written_count": len(written),
        "files_rewritten": sorted(rewritten),
        "reused": not rewritten and previous_id == analysis_id,
        "previous_analysis_id": previous_id,
        "forced": bool(force),
        "unmanaged_plan_documents": unmanaged,
    }


# =============================================================================
# Actual PRELABEL-EXPORT stage: the one shared censoring raster
#
# One Earth Engine export, into this namespace only. The canonical label
# rasters are never rewritten, the canonical Step6/Step6B gate is never re-run,
# and the resulting raster is NOT a predictor -- it exists solely so a later
# stage can censor cells that already burned before label_start.
# =============================================================================
PRELABEL_STAGE = "prelabel-export"
PRELABEL_SUMMARY_SCHEMA = "window_closure_prelabel_censor.v1"
PRELABEL_PRODUCER = "src.step6_validate_fire_relation.export_raw_mcd64a1_prelabel_labels"
PRELABEL_RASTER_NAME = "prelabel_burndate.tif"
PRELABEL_SUMMARY_NAME = "censoring_summary.json"
PRELABEL_CHECKPOINT_NAME = "prelabel_export_checkpoint.json"
PRELABEL_QUARANTINE_DIR = "_quarantine"

PRELABEL_EXPECTED_BAND_COUNT = 1
# The pre-label raster shares the canonical label grid: both are exported by
# the SAME Step6 helper, at the same scale/CRS, over the same AOI geometry.
PRELABEL_REFERENCE_ROLE = LABEL_ROLE_RAW

STATUS_PASS = "pass"

# Documents the plan stage owns and this stage may only READ.
PLAN_BINDING_DOCUMENTS: tuple[str, ...] = (
    "config/preregistration.json",
    "config/frozen_input_inventory.json",
    "prelabel_censor/export_plan.json",
)


# --- Date semantics (pure) ---------------------------------------------------
def prelabel_collection_query_bounds(start: str, end: str) -> tuple[str, str]:
    """Month-aligned MCD64A1 collection bounds -- mirrors production Step6.

    Reproduces `src.step6_validate_fire_relation._mcd64a1_collection_query_bounds`
    EXACTLY (asserted against it in the tests, and re-checked against the real
    helper at export time). It is mirrored rather than imported because
    importing Step6 pulls Earth Engine into the process, which the planning and
    dry-run paths must never do.
    """
    start_dt, end_dt = _parse(start), _parse(end)
    collection_start = start_dt.replace(day=1)
    if end_dt.month == 12:
        collection_end_exclusive = end_dt.replace(year=end_dt.year + 1, month=1, day=1)
    else:
        collection_end_exclusive = end_dt.replace(month=end_dt.month + 1, day=1)
    return _fmt(collection_start), _fmt(collection_end_exclusive)


def prelabel_date_semantics(start: str, end: str) -> dict:
    """The explicit, non-silent boundary contract of the pre-label export.

    There are TWO windows and they do NOT have the same semantics:

    * the REQUESTED censoring interval `[start, end]` is INCLUSIVE at both
      ends. Step6 turns it into per-image day-of-year bounds and keeps pixels
      with `BurnDate >= start_doy AND BurnDate <= end_doy`, so a burn on `end`
      itself IS included;
    * the Earth Engine `filterDate` call uses the MONTH-ALIGNED collection
      bounds, whose end is EXCLUSIVE. That window is deliberately WIDER than
      the requested interval (MCD64A1 is a monthly product whose images are
      stamped on the 1st, so a narrower filterDate would silently drop the
      month that actually contains the burns).

    Because the exclusive bound applies only to the wider collection query and
    the inclusive DOY mask applies to the values, the effective last included
    date equals the requested end date -- there is no off-by-one, and that is
    recorded here explicitly rather than left implicit.
    """
    start_dt, end_dt = _parse(start), _parse(end)
    if end_dt < start_dt:
        raise WindowClosureError(
            f"Pre-label interval is empty: {start} .. {end}."
        )
    ee_start, ee_end_exclusive = prelabel_collection_query_bounds(start, end)
    return {
        "requested_interval_start": start,
        "requested_interval_end": end,
        "interval_semantics": "inclusive_start_inclusive_end",
        "ee_filter_start": ee_start,
        "ee_filter_end": ee_end_exclusive,
        "ee_filter_end_semantics": "exclusive",
        "ee_filter_window_is_month_aligned": True,
        "effective_last_included_date": end,
        "effective_first_included_date": start,
        "burndate_doy_range_inclusive": [
            start_dt.timetuple().tm_yday, end_dt.timetuple().tm_yday,
        ],
        "crosses_year_boundary": start_dt.year != end_dt.year,
        "off_by_one_risk": "none",
        "note": (
            "The requested interval is inclusive at BOTH ends and is enforced "
            "per image by an inclusive BurnDate day-of-year mask "
            "(>= start_doy AND <= end_doy). The exclusive Earth Engine "
            "filterDate end applies only to the WIDER month-aligned MCD64A1 "
            "collection query, which exists so the monthly image carrying the "
            "burns is not dropped. The effective last included date therefore "
            "equals the requested end date."
        ),
        "source": (
            "src.step6_validate_fire_relation.build_raw_burndate_image / "
            "_mcd64a1_collection_query_bounds"
        ),
    }


def allowed_burndate_window(start: str, end: str) -> dict:
    start_dt, end_dt = _parse(start), _parse(end)
    return {
        "start_doy": start_dt.timetuple().tm_yday,
        "end_doy": end_dt.timetuple().tm_yday,
        "crosses_year": start_dt.year != end_dt.year,
    }


def burndate_value_is_allowed(value: float, window: dict) -> bool:
    """0 means unburned; any other value must fall inside the requested DOY window."""
    if value == 0:
        return True
    if value != int(value) or not (1 <= value <= 366):
        return False
    if window["crosses_year"]:
        return value >= window["start_doy"] or value <= window["end_doy"]
    return window["start_doy"] <= value <= window["end_doy"]


# --- Plan binding ------------------------------------------------------------
def _read_plan_document(path: Path, relative: str) -> dict:
    if not path.is_file():
        raise WindowClosureError(
            f"Plan document '{relative}' is missing at {path}. The "
            "prelabel-export stage binds to a completed plan; run "
            "--from-stage plan --to-stage plan first. No Earth Engine call "
            "was made."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise WindowClosureError(
            f"Plan document '{relative}' at {path} is unreadable: {exc}. "
            "No Earth Engine call was made."
        ) from exc
    if not isinstance(payload, dict):
        raise WindowClosureError(f"Plan document '{relative}' is not a JSON object.")
    return payload


def assert_plan_binding(
    experiment_id: str,
    analysis_id: str,
    shifts: Sequence[int],
    censor: dict,
    inventory: dict,
    planned_paths: dict[str, str],
) -> dict:
    """Bind this run to the plan documents already on disk.

    Every check below happens BEFORE Earth Engine is imported or called. A
    missing, unreadable or disagreeing plan document stops the stage.
    """
    documents = {
        relative: _read_plan_document(Path(planned_paths[relative]), relative)
        for relative in PLAN_BINDING_DOCUMENTS
    }
    prereg = documents["config/preregistration.json"]
    frozen = documents["config/frozen_input_inventory.json"]
    prelabel_plan = documents["prelabel_censor/export_plan.json"]

    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise WindowClosureError(
                f"Plan binding failed: {message} No Earth Engine call was made "
                "and nothing was written."
            )

    _require(
        prereg.get("schema_version") == SCHEMA_VERSION,
        f"preregistration schema is {prereg.get('schema_version')!r}, "
        f"expected {SCHEMA_VERSION!r}.",
    )
    _require(
        prereg.get("experiment_id") == experiment_id,
        f"preregistration experiment_id is {prereg.get('experiment_id')!r}, "
        f"expected {experiment_id!r}.",
    )
    for relative, document in (
        ("config/preregistration.json", prereg),
        ("config/frozen_input_inventory.json", frozen),
        ("prelabel_censor/export_plan.json", prelabel_plan),
    ):
        _require(
            document.get("analysis_id") == analysis_id,
            f"'{relative}' holds analysis_id {document.get('analysis_id')!r}, "
            f"but this run computed {analysis_id!r}.",
        )

    planned_shifts = (
        (prereg.get("scientific_configuration") or {}).get("preregistered_shifts_days")
    )
    _require(
        list(planned_shifts or []) == list(shifts),
        f"preregistered shifts {planned_shifts!r} differ from the requested "
        f"{list(shifts)!r}.",
    )
    _require(bool(prereg.get("prerequisites_ready")), "the plan is not marked ready.")

    for key in ("common_prelabel_start", "common_prelabel_end"):
        _require(
            prelabel_plan.get(key) == censor[key],
            f"pre-label export plan {key}={prelabel_plan.get(key)!r} differs "
            f"from the derived {censor[key]!r}.",
        )
        _require(
            (prereg.get("common_censor_interval") or {}).get(key) == censor[key],
            f"preregistration {key} differs from the derived {censor[key]!r}.",
        )

    current = frozen_hash_map(inventory)
    for relative, recorded in (
        ("config/preregistration.json", prereg.get("frozen_input_sha256") or {}),
        ("config/frozen_input_inventory.json", frozen.get("frozen_input_sha256") or {}),
    ):
        for role in REQUIRED_FROZEN_INPUT_ROLES:
            _require(
                recorded.get(role) is not None,
                f"'{relative}' carries no hash for required frozen input '{role}'.",
            )
            _require(
                recorded.get(role) == current.get(role),
                f"frozen input '{role}' hashes {current.get(role)!r} now but "
                f"'{relative}' pinned {recorded.get(role)!r}.",
            )

    return {
        "bound_to_plan": True,
        "plan_documents": {
            relative: str(planned_paths[relative]) for relative in PLAN_BINDING_DOCUMENTS
        },
        "analysis_id": analysis_id,
        "preregistered_shifts_days": list(planned_shifts or []),
        "common_prelabel_start": censor["common_prelabel_start"],
        "common_prelabel_end": censor["common_prelabel_end"],
        "frozen_input_sha256": current,
    }


# --- Raster contract ---------------------------------------------------------
def reference_grid_source(inventory: dict) -> Path:
    """The frozen canonical raster whose grid the pre-label raster must match.

    Deterministically the raw BurnDate label raster: it is a REQUIRED frozen
    input, it is produced by the same Step6 exporter at the same scale, CRS and
    AOI, and its hash is already pinned in the analysis identity.
    """
    entry = inventory.get(PRELABEL_REFERENCE_ROLE) or {}
    path = entry.get("path")
    if not path or not entry.get("exists"):
        raise WindowClosureError(
            f"Reference grid source '{PRELABEL_REFERENCE_ROLE}' is not "
            "available; the raster grid cannot be verified."
        )
    return Path(path)


def read_grid_signature(path: Path) -> dict:
    """CRS / transform / shape / band count of a raster, as plain JSON types."""
    import rasterio

    with rasterio.open(path) as dataset:
        return {
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": [float(value) for value in tuple(dataset.transform)[:6]],
            "width": int(dataset.width),
            "height": int(dataset.height),
            "band_count": int(dataset.count),
            "dtype": str(dataset.dtypes[0]),
            "nodata": None if dataset.nodata is None else float(dataset.nodata),
        }


def grid_signatures_match(actual: dict, reference: dict, tolerance: float = 1e-9) -> bool:
    if actual.get("crs") != reference.get("crs"):
        return False
    if (actual.get("width"), actual.get("height")) != \
            (reference.get("width"), reference.get("height")):
        return False
    left, right = actual.get("transform") or [], reference.get("transform") or []
    if len(left) != len(right):
        return False
    return all(abs(a - b) <= tolerance * max(1.0, abs(b)) for a, b in zip(left, right))


def inspect_prelabel_raster(path: Path, censor: dict, reference_path: Path) -> dict:
    """Full raster contract. Raises on anything that would poison the censoring.

    A zero pre-label burn count is a VALID scientific outcome and is NOT an
    error; a raster that cannot be read, does not carry a grid, or carries
    BurnDate values outside the requested window IS.
    """
    import numpy as np
    import rasterio

    if not path.is_file():
        raise WindowClosureError(f"Pre-label raster was not produced: {path}.")
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise WindowClosureError(f"Pre-label raster is empty (0 bytes): {path}.")

    try:
        with rasterio.open(path) as dataset:
            crs = dataset.crs
            transform = dataset.transform
            width, height, band_count = int(dataset.width), int(dataset.height), int(dataset.count)
            dtype = str(dataset.dtypes[0])
            nodata = dataset.nodata
            band = dataset.read(1, masked=True)
    except WindowClosureError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any reader failure is a contract failure
        raise WindowClosureError(
            f"Pre-label raster at {path} could not be read: "
            f"{type(exc).__name__}: {exc}."
        ) from exc

    if crs is None:
        raise WindowClosureError(f"Pre-label raster has no CRS: {path}.")
    if transform is None or not transform.is_rectilinear:
        raise WindowClosureError(f"Pre-label raster has no usable transform: {path}.")
    if width <= 0 or height <= 0:
        raise WindowClosureError(
            f"Pre-label raster has a non-positive shape ({width}x{height}): {path}."
        )
    if band_count != PRELABEL_EXPECTED_BAND_COUNT:
        raise WindowClosureError(
            f"Pre-label raster has {band_count} band(s), expected "
            f"{PRELABEL_EXPECTED_BAND_COUNT}: {path}."
        )

    signature = {
        "crs": str(crs),
        "transform": [float(value) for value in tuple(transform)[:6]],
        "width": width,
        "height": height,
        "band_count": band_count,
        "dtype": dtype,
        "nodata": None if nodata is None else float(nodata),
    }
    reference = read_grid_signature(reference_path)
    if not grid_signatures_match(signature, reference):
        raise WindowClosureError(
            "Pre-label raster grid does not match the canonical reference "
            f"grid from {reference_path}. raster={signature}, "
            f"reference={reference}. The censoring cohort would not align with "
            "the analysis grid, so the stage fails and no scientific summary "
            "is written."
        )

    values = band.compressed().astype("float64")
    values = values[np.isfinite(values)]
    finite_cell_count = int(values.size)

    window = allowed_burndate_window(
        censor["common_prelabel_start"], censor["common_prelabel_end"],
    )
    positive = values[values > 0]
    if finite_cell_count:
        unique = np.unique(values)
        illegal = sorted(
            float(value) for value in unique
            if not burndate_value_is_allowed(float(value), window)
        )
        if illegal:
            raise WindowClosureError(
                f"Pre-label raster carries BurnDate value(s) outside the "
                f"requested window {censor['common_prelabel_start']} .. "
                f"{censor['common_prelabel_end']} (DOY "
                f"{window['start_doy']}-{window['end_doy']}): {illegal[:10]}. "
                "Only 0 (unburned), in-window day-of-year values or nodata are "
                "allowed."
            )
        if positive.size and np.all(positive == 1.0):
            raise WindowClosureError(
                f"Pre-label raster looks BINARY (every positive value is 1.0): "
                f"{path}. A binary mask silently destroys the burn-date "
                "information the censoring relies on."
            )

    return {
        "raster_path": str(path),
        "raster_bytes": int(size_bytes),
        "raster_sha256": sha256_file(path),
        "grid_signature": signature,
        "reference_grid_path": str(reference_path),
        "reference_grid_signature": reference,
        "grid_matches_reference": True,
        "dtype": dtype,
        "nodata": signature["nodata"],
        "mask_semantics": (
            "rasterio masked read; nodata and unset pixels are excluded from "
            "every count. 0 means observed-and-unburned, a positive value is "
            "the MCD64A1 BurnDate day of year inside the requested window."
        ),
        "band_count": band_count,
        "finite_cell_count": finite_cell_count,
        "prelabel_burn_cell_count": int(positive.size),
        "zero_or_unburned_cell_count": int(finite_cell_count - positive.size),
        "min_finite_burndate": float(positive.min()) if positive.size else None,
        "max_finite_burndate": float(positive.max()) if positive.size else None,
        "allowed_burndate_doy_range": [window["start_doy"], window["end_doy"]],
        "zero_burn_is_a_valid_outcome": True,
    }


# --- The stage ---------------------------------------------------------------
def prelabel_output_paths(
    experiment_id: str, output_root: Optional[Path] = None,
) -> dict[str, Path]:
    root = experiment_root(experiment_id, output_root) / "prelabel_censor"
    return {
        "raster": root / PRELABEL_RASTER_NAME,
        "summary": root / PRELABEL_SUMMARY_NAME,
        "checkpoint": root / PRELABEL_CHECKPOINT_NAME,
    }


def assert_prelabel_owned_targets(
    experiment_id: str, targets: Iterable[Path], output_root: Optional[Path] = None,
) -> None:
    """This stage may only ever touch its own three outputs."""
    owned = {
        path.resolve() for path in prelabel_output_paths(experiment_id, output_root).values()
    }
    for path in targets:
        if Path(path).resolve() not in owned:
            raise WindowClosureError(
                f"'{path}' is not a prelabel-owned output. This stage writes "
                f"only {sorted(p.name for p in owned)} inside its own "
                "prelabel_censor/ directory."
            )


def _quarantine_raster(path: Path) -> Optional[str]:
    """Move an existing raster aside instead of deleting it. Never removes data."""
    if not path.is_file():
        return None
    digest = sha256_file(path)[:12]
    target = path.parent / PRELABEL_QUARANTINE_DIR / f"{path.stem}.{digest}{path.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, target)
    return str(target)


def _checkpoint_is_valid(checkpoint_path: Path, analysis_id: str, raster_path: Path) -> bool:
    if not checkpoint_path.is_file() or not raster_path.is_file():
        return False
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("analysis_id") != analysis_id:
        return False
    if payload.get("status") != STATUS_PASS:
        return False
    return payload.get("raster_sha256") == sha256_file(raster_path)


def run_prelabel_export(
    experiment_id: str,
    analysis_id: str,
    censor: dict,
    inventory: dict,
    binding: dict,
    output_root: Optional[Path] = None,
    force: bool = False,
    resume: bool = False,
    exporter: Optional[Any] = None,
) -> dict:
    """Export (or reuse) the ONE shared pre-label BurnDate raster.

    `exporter` is an explicit dependency-injection point with the production
    default `src.step6_validate_fire_relation.export_raw_mcd64a1_prelabel_labels`,
    imported LAZILY so that no planning or dry-run path ever pulls Earth Engine
    into the process. Tests inject a fake and never touch Earth Engine.
    """
    paths = prelabel_output_paths(experiment_id, output_root)
    raster_path, summary_path, checkpoint_path = (
        paths["raster"], paths["summary"], paths["checkpoint"]
    )
    assert_prelabel_owned_targets(experiment_id, paths.values(), output_root)

    semantics = prelabel_date_semantics(
        censor["common_prelabel_start"], censor["common_prelabel_end"],
    )
    reference_path = reference_grid_source(inventory)

    # --- Decide: reuse, refuse, or export ------------------------------------
    reused = False
    quarantined: Optional[str] = None
    if raster_path.is_file() and not force:
        if resume:
            if _checkpoint_is_valid(checkpoint_path, analysis_id, raster_path):
                # The contract is re-verified below regardless; a checkpoint is
                # never trusted on its own.
                reused = True
            else:
                quarantined = _quarantine_raster(raster_path)
        else:
            raise WindowClosureError(
                f"A pre-label raster already exists at {raster_path}. Refusing "
                "to overwrite it silently: re-run with resume=True to reuse a "
                "valid existing raster, or force=True to re-export it (the old "
                "raster is quarantined, never deleted)."
            )
    elif raster_path.is_file() and force:
        quarantined = _quarantine_raster(raster_path)

    gee_query_run = False
    gee_export_run = False
    export_record: Optional[dict] = None
    if not reused:
        if exporter is None:
            # Imported here and ONLY here: importing Step6 pulls Earth Engine.
            from src.step6_validate_fire_relation import (  # noqa: PLC0415
                _mcd64a1_collection_query_bounds,
                export_raw_mcd64a1_prelabel_labels,
            )
            # The recorded semantics must be the production ones, not a copy
            # that has drifted.
            production_bounds = _mcd64a1_collection_query_bounds(
                censor["common_prelabel_start"], censor["common_prelabel_end"],
            )
            if tuple(production_bounds) != (
                semantics["ee_filter_start"], semantics["ee_filter_end"],
            ):
                raise WindowClosureError(
                    "Recorded Earth Engine filter bounds "
                    f"{(semantics['ee_filter_start'], semantics['ee_filter_end'])} "
                    f"disagree with production {tuple(production_bounds)}; the "
                    "date-semantics record has drifted from Step6."
                )
            exporter = export_raw_mcd64a1_prelabel_labels

        raster_path.parent.mkdir(parents=True, exist_ok=True)
        gee_query_run = True
        gee_export_run = True
        # Dates come from the PLAN, never from the registry: this is the one
        # interval every variant shares.
        export_record = exporter(
            experiment_id=experiment_id,
            pre_label_start=censor["common_prelabel_start"],
            pre_label_end=censor["common_prelabel_end"],
            raw_out=raster_path,
        )

    # --- Raster contract (always, reused or freshly exported) ----------------
    inspection = inspect_prelabel_raster(raster_path, censor, reference_path)

    tiles_dir = raster_path.parent / "_tiles"
    summary = {
        "schema_version": PRELABEL_SUMMARY_SCHEMA,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "stage": PRELABEL_STAGE,
        "common_prelabel_start": censor["common_prelabel_start"],
        "common_prelabel_end": censor["common_prelabel_end"],
        "applies_to_all_variants": True,
        "date_semantics": semantics,
        "producer": PRELABEL_PRODUCER,
        "raster_path": inspection["raster_path"],
        "raster_sha256": inspection["raster_sha256"],
        "raster_bytes": inspection["raster_bytes"],
        "grid_signature": inspection["grid_signature"],
        "grid_matches_reference": inspection["grid_matches_reference"],
        "reference_grid_path": inspection["reference_grid_path"],
        "reference_grid_role": PRELABEL_REFERENCE_ROLE,
        "dtype": inspection["dtype"],
        "nodata": inspection["nodata"],
        "mask_semantics": inspection["mask_semantics"],
        "band_count": inspection["band_count"],
        "finite_cell_count": inspection["finite_cell_count"],
        "prelabel_burn_cell_count": inspection["prelabel_burn_cell_count"],
        "zero_or_unburned_cell_count": inspection["zero_or_unburned_cell_count"],
        "min_finite_burndate": inspection["min_finite_burndate"],
        "max_finite_burndate": inspection["max_finite_burndate"],
        "allowed_burndate_doy_range": inspection["allowed_burndate_doy_range"],
        "zero_burn_is_a_valid_outcome": True,
        "gee_query_run": gee_query_run,
        "gee_export_run": gee_export_run,
        "model_fit": False,
        "bootstrap_run": False,
        "reused_existing_raster": reused,
        "quarantined_previous_raster": quarantined,
        "tiles_directory_present": tiles_dir.exists(),
        "canonical_outputs_modified": False,
        "canonical_label_raster_modified": False,
        "canonical_gate_rerun": False,
        "used_as_predictor": False,
        "purpose": (
            "Shared pre-label censoring only: it identifies cells that already "
            "burned before label_start so a later stage can remove them from "
            "the common cohort of EVERY variant. It is never a predictor."
        ),
        "bound_to_plan": binding["bound_to_plan"],
        "frozen_hashes_unchanged": True,
        "status": STATUS_PASS,
    }
    checkpoint = {
        "schema_version": PRELABEL_SUMMARY_SCHEMA,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "stage": PRELABEL_STAGE,
        "raster_path": inspection["raster_path"],
        "raster_sha256": inspection["raster_sha256"],
        "raster_bytes": inspection["raster_bytes"],
        "grid_signature": inspection["grid_signature"],
        "date_semantics": semantics,
        "frozen_input_sha256": frozen_hash_map(inventory),
        "status": STATUS_PASS,
    }

    written: list[str] = [str(raster_path)]
    rewritten: list[str] = [] if reused else [str(raster_path)]
    for path, payload in ((summary_path, summary), (checkpoint_path, checkpoint)):
        text = _json_document(payload)
        if not (path.is_file() and path.read_text(encoding="utf-8") == text):
            _atomic_write_text(path, text)
            rewritten.append(str(path))
        written.append(str(path))

    return {
        "files_written": sorted(written),
        "files_rewritten": sorted(rewritten),
        "reused": reused,
        "gee_query_run": gee_query_run,
        "gee_export_run": gee_export_run,
        "quarantined_previous_raster": quarantined,
        "summary": summary,
        "export_record": None if export_record is None else {
            key: str(value) for key, value in sorted(export_record.items())
        },
    }


# =============================================================================
# Actual PREDICTOR-EXPORT stage
#
# Rebuilds, for every NON-CANONICAL variant, exactly the predictors that follow
# from the shifted window: current Landsat LST/NDVI, each preregistered
# baseline year's Landsat LST/NDVI, and the current-window MODIS
# mean/std/valid-count. The canonical variant is never exported -- it IS the
# frozen production result the early closures are compared against.
#
# No compositing, QA, reducer or MODIS formula is defined here: every image
# comes from the production Step3 builders and the production
# prepare_modis_for_step7, driven by a variant context whose only difference is
# the predictor window timing.
# =============================================================================
PREDICTOR_STAGE = "predictor-export"
PREDICTOR_METADATA_SCHEMA = "window_closure_predictor_export.v1"
PREDICTOR_METADATA_NAME = "predictor_export_metadata.json"
PREDICTOR_QUARANTINE_DIR = "_quarantine"

# Two production products per Landsat logical role, taken as separate rasters
# from the SAME production image: they are already two named bands there, so
# nothing is recomputed.
LANDSAT_PRODUCTS_PER_ROLE: tuple[str, ...] = PRODUCTION_LANDSAT_PRODUCTS
PRODUCT_SCENE_WEIGHTED_MEDIAN = "scene_weighted_median"
PRODUCT_SCENE_VALID_COUNT = "scene_valid_count"

# Production band names, verified against src/step3_landsat_lst.py.
LANDSAT_ROLE_BANDS: dict[str, dict[str, str]] = {
    "current_lst": {
        PRODUCT_SCENE_WEIGHTED_MEDIAN: "Current_Period_LST_Celsius",
        PRODUCT_SCENE_VALID_COUNT: "Current_Period_Valid_Count",
    },
    "current_ndvi": {
        PRODUCT_SCENE_WEIGHTED_MEDIAN: "Current_Period_NDVI",
        PRODUCT_SCENE_VALID_COUNT: "Current_Period_NDVI_Valid_Count",
    },
    "baseline_lst": {
        PRODUCT_SCENE_WEIGHTED_MEDIAN: "ST_B10",
        PRODUCT_SCENE_VALID_COUNT: "Baseline_Window_Valid_Count",
    },
    "baseline_ndvi": {
        PRODUCT_SCENE_WEIGHTED_MEDIAN: "NDVI",
        PRODUCT_SCENE_VALID_COUNT: "Baseline_Window_NDVI_Valid_Count",
    },
}

MODIS_ROLE_ORDER: tuple[str, ...] = (
    "modis_lst_mean", "modis_lst_std", "modis_valid_observation_count",
)
MODIS_COUNT_ROLE = "modis_valid_observation_count"

# Products whose values are OBSERVATION COUNTS: non-negative integers, and an
# all-zero raster is a legitimate (if notable) result.
COUNT_PRODUCTS: frozenset = frozenset({PRODUCT_SCENE_VALID_COUNT, MODIS_COUNT_ROLE})

# Grid families. Landsat and MODIS legitimately live on DIFFERENT grids, so a
# single reference signature would be wrong; each family is checked against the
# grid its production export scale implies.
GRID_FAMILY_LANDSAT = "landsat_30m"
GRID_FAMILY_MODIS = "modis_1km"
LANDSAT_EXPORT_SCALE_M = 30
# Mirrors scripts.prepare_modis_for_step7.MODIS_EXPORT_SCALE (asserted in tests).
MODIS_EXPORT_SCALE_M = 1000
GRID_FAMILY_SCALES: dict[str, int] = {
    GRID_FAMILY_LANDSAT: LANDSAT_EXPORT_SCALE_M,
    GRID_FAMILY_MODIS: MODIS_EXPORT_SCALE_M,
}
# Relative tolerance on the family pixel size. The Landsat family is exported at
# exactly the reference grid's scale, so it matches to floating-point precision;
# the MODIS expectation is DERIVED (reference pixel * 1000/30) via Earth
# Engine's metres-to-degrees conversion, which may round differently. 1e-3 still
# catches every scale error that matters -- a Landsat/MODIS mix-up is a factor
# of ~33 -- while not failing a correct export on the last decimal.
GRID_PIXEL_SIZE_RELATIVE_TOLERANCE = 1e-3

# Variant data sub-directories, taken from the variant context keys so the
# layout can never drift from the context the exporters are actually given.
LANDSAT_ROLE_CONTEXT_DIR: dict[str, str] = {
    "current_lst": "current_period_dir",
    "current_ndvi": "ndvi_current_dir",
    "baseline_lst": "baseline_input_dir",
    "baseline_ndvi": "ndvi_baseline_dir",
}

# Key the MODIS production guard reads to allow a dedicated diagnostics root.
MODIS_NAMESPACE_ALLOWED_ROOTS_KEY = "namespace_allowed_roots"

PREDICTOR_EXPORT_LIMITATIONS: tuple[str, ...] = (
    "Production MODIS preparation applies a FIXED calendar-month filter "
    "(core.config SUMMER_MONTH_START..SUMMER_MONTH_END) on top of the "
    "predictor window. A variant window that reaches outside those months is "
    "CLIPPED by that filter, so its effective MODIS window is shorter than "
    "its requested window. This is production behaviour, is left unchanged, "
    "and is reported per variant -- but it means part of the MODIS support "
    "difference between variants comes from the fixed month filter and not "
    "from the closure shift alone.",
    "Landsat and MODIS observation support legitimately differs between "
    "variants; identical support is NEVER forced, because support change is "
    "part of what this sensitivity analysis measures.",
    "The pre-label censoring raster is not a predictor and is never read by "
    "this stage.",
)


# --- Date semantics (pure) ---------------------------------------------------
def landsat_job_date_semantics(start_date: str, end_date: str) -> dict:
    """Explicit boundary contract for one Landsat export window.

    Production calls `filterDate(start_date, end_date)`, whose END is
    EXCLUSIVE. That behaviour is preserved verbatim -- this stage never adds a
    silent +1 day -- and the effective last included date is recorded so the
    off-by-one is visible rather than implicit.
    """
    semantics = window_closure_date_window_semantics(start_date, end_date)
    return {
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "ee_filter_start": semantics["filter_date_start"],
        "ee_filter_end": semantics["filter_date_end"],
        "ee_filter_end_semantics": semantics["end_semantics"],
        "effective_last_included_date": semantics["effective_last_included_date"],
        "duration_days": (_parse(end_date) - _parse(start_date)).days,
        "changed_factor": semantics["changed_factor"],
        "held_fixed": semantics["held_fixed"],
        "note": semantics["note"],
    }


def modis_month_filter_transparency(start_date: str, end_date: str) -> dict:
    """How the production MODIS summer-month filter interacts with a window.

    `scripts/prepare_modis_for_step7._build_qc_masked_modis_stack` applies a
    FIXED `ee.Filter.calendarRange(SUMMER_MONTH_START, SUMMER_MONTH_END)` on
    top of `filterDate`. Unlike the Landsat month filter -- which production
    DERIVES from the window and which therefore never clips it -- this one is a
    constant, so an earlier-closing window can legitimately lose its earliest
    days. Production is not changed; the effect is measured and reported.
    """
    from core.config import SUMMER_MONTH_END, SUMMER_MONTH_START

    start_dt, end_dt = _parse(start_date), _parse(end_date)
    months = list(range(int(SUMMER_MONTH_START), int(SUMMER_MONTH_END) + 1))
    included = [
        start_dt + timedelta(days=offset)
        for offset in range((end_dt - start_dt).days)  # filterDate end is exclusive
        if (start_dt + timedelta(days=offset)).month in months
    ]
    clipped = (end_dt - start_dt).days - len(included)
    return {
        "calendar_month_filter": f"{int(SUMMER_MONTH_START)}-{int(SUMMER_MONTH_END)}",
        "calendar_month_filter_is_fixed": True,
        "calendar_month_filter_source": (
            "core.config SUMMER_MONTH_START / SUMMER_MONTH_END, applied by "
            "scripts.prepare_modis_for_step7._build_qc_masked_modis_stack"
        ),
        "calendar_month_filter_clips_window": clipped > 0,
        "clipped_day_count": int(clipped),
        "effective_first_included_date": _fmt(included[0]) if included else None,
        "effective_last_included_date": _fmt(included[-1]) if included else None,
        "effective_included_day_count": len(included),
        "note": (
            "A FIXED production month filter is applied on top of filterDate. "
            "Days of the requested window that fall outside those calendar "
            "months are removed by production, so the effective MODIS window "
            "can be shorter than the requested one. Production behaviour is "
            "left unchanged and the difference is reported, not corrected."
        ),
    }


def modis_job_date_semantics(start_date: str, end_date: str) -> dict:
    """Boundary contract for the MODIS current-window export."""
    end_dt = _parse(end_date)
    month_filter = modis_month_filter_transparency(start_date, end_date)
    return {
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "ee_filter_start": start_date,
        "ee_filter_end": end_date,
        "ee_filter_end_semantics": "exclusive",
        "effective_last_included_date": (
            month_filter["effective_last_included_date"]
            or _fmt(end_dt - timedelta(days=1))
        ),
        "duration_days": (end_dt - _parse(start_date)).days,
        "calendar_month_filter_transparency": month_filter,
        "note": (
            "Production filterDate end is exclusive and is preserved verbatim; "
            "this stage never adds a silent +1 day. The effective last "
            "included date additionally accounts for the fixed production "
            "calendar-month filter."
        ),
    }


# --- Fixed month-filter clipping, read back from export provenance -----------
# The clipping is DERIVED at export time by `modis_month_filter_transparency`
# and persisted per MODIS artefact inside `predictor_export_metadata.json`.
# Downstream stages read it back from there instead of recomputing it, so a
# published number can never disagree with what production actually exported.
MODIS_CLIPPING_TRANSPARENCY_KEY = "calendar_month_filter_transparency"


def modis_clipping_from_predictor_metadata(metadata: dict, variant_id: str, source: str) -> dict:
    """The variant's fixed-month-filter clipping, read from export provenance.

    Fail-closed in every direction: a missing MODIS record, a missing
    transparency block, a non-integer or negative count, two MODIS roles that
    disagree, or a clipped/effective/requested triple that does not add up are
    all failures. An unknown clipping is NEVER reported as zero.
    """
    if not isinstance(metadata, dict):
        raise WindowClosureError(
            f"BLOCKER: MODIS_CLIPPING_PROVENANCE_MISSING -- variant "
            f"'{variant_id}': {source} is not a JSON object."
        )
    records: dict[str, dict] = {}
    for record in (metadata.get("artifact_inventory") or []):
        if not isinstance(record, dict):
            continue
        role = str(record.get("role") or record.get("artifact_id") or "")
        if role not in MODIS_ROLE_FILENAMES:
            continue
        transparency = (
            (record.get("date_semantics") or {}).get(MODIS_CLIPPING_TRANSPARENCY_KEY)
        )
        if not isinstance(transparency, dict):
            raise WindowClosureError(
                f"BLOCKER: MODIS_CLIPPING_PROVENANCE_MISSING -- variant "
                f"'{variant_id}': MODIS role '{role}' in {source} carries no "
                f"'{MODIS_CLIPPING_TRANSPARENCY_KEY}' block."
            )
        clipped = transparency.get("clipped_day_count")
        effective = transparency.get("effective_included_day_count")
        requested = (record.get("date_semantics") or {}).get("duration_days")
        for name, value in (
            ("clipped_day_count", clipped),
            ("effective_included_day_count", effective),
            ("duration_days", requested),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise WindowClosureError(
                    f"BLOCKER: MODIS_CLIPPING_PROVENANCE_INVALID -- variant "
                    f"'{variant_id}' role '{role}': {name}={value!r} is not an "
                    "integer."
                )
            if value < 0:
                raise WindowClosureError(
                    f"BLOCKER: MODIS_CLIPPING_PROVENANCE_INVALID -- variant "
                    f"'{variant_id}' role '{role}': {name}={value} is negative."
                )
        if int(clipped) + int(effective) != int(requested):
            raise WindowClosureError(
                f"BLOCKER: MODIS_CLIPPING_PROVENANCE_INCONSISTENT -- variant "
                f"'{variant_id}' role '{role}': clipped({clipped}) + "
                f"effective({effective}) != requested({requested})."
            )
        records[role] = {
            "clipped_day_count": int(clipped),
            "effective_included_day_count": int(effective),
            "requested_day_count": int(requested),
            "calendar_month_filter": transparency.get("calendar_month_filter"),
        }
    if not records:
        raise WindowClosureError(
            f"BLOCKER: MODIS_CLIPPING_PROVENANCE_MISSING -- variant "
            f"'{variant_id}': {source} records no MODIS current-window "
            f"artefact, so the fixed month-filter clipping is unknown. An "
            "unknown clipping is never reported as zero."
        )
    distinct = sorted({record["clipped_day_count"] for record in records.values()})
    if len(distinct) != 1:
        raise WindowClosureError(
            f"BLOCKER: MODIS_CLIPPING_PROVENANCE_INCONSISTENT -- variant "
            f"'{variant_id}': MODIS roles disagree on clipped_day_count "
            f"{distinct}; they share one requested window."
        )
    anchor = records[sorted(records)[0]]
    return {
        "variant_id": variant_id,
        "clipped_day_count": distinct[0],
        "effective_included_day_count": anchor["effective_included_day_count"],
        "requested_day_count": anchor["requested_day_count"],
        "calendar_month_filter": anchor["calendar_month_filter"],
        "roles": sorted(records),
        "source": source,
        "derived_at": "predictor-export",
    }


# --- Job set (pure) ----------------------------------------------------------
def nonzero_variants(variants: Sequence[dict]) -> list[dict]:
    """Non-canonical variants, deterministically ordered by increasing shift."""
    return sorted(
        (variant for variant in variants if not variant["is_canonical"]),
        key=lambda variant: int(variant["shift_days"]),
    )


def expected_logical_role_count(baseline_years: Sequence[int]) -> int:
    """2 current Landsat + 2 per baseline year + 3 MODIS. Never hard-coded."""
    return 2 + 2 * len(baseline_years) + len(MODIS_ROLE_ORDER)


def expected_raster_count(baseline_years: Sequence[int]) -> int:
    landsat_roles = 2 + 2 * len(baseline_years)
    return landsat_roles * len(LANDSAT_PRODUCTS_PER_ROLE) + len(MODIS_ROLE_ORDER)


def _variant_data_dirs(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> dict[str, Path]:
    data_root = variant_root(experiment_id, variant_id, output_root) / "data"
    return {
        "current_period_dir": data_root / "current_period",
        "ndvi_current_dir": data_root / "ndvi_current_period",
        "baseline_input_dir": data_root / "landsat_timeseries",
        "ndvi_baseline_dir": data_root / "ndvi_timeseries",
        "modis_input_dir": data_root / "modis",
    }


def predictor_artifact_jobs(
    experiment_id: str,
    variant: dict,
    baseline_years: Sequence[int],
    window_days: int,
    output_root: Optional[Path] = None,
) -> list[dict]:
    """Every raster this stage will produce for ONE non-canonical variant.

    Pure: no Earth Engine, no file system. Dates come from the same
    `landsat_export_plan` / `modis_export_plan` the preregistration was built
    from, so a job can never disagree with the frozen plan.
    """
    if variant["is_canonical"]:
        raise WindowClosureError(
            f"The canonical variant '{variant['variant_id']}' has no predictor "
            "export: it reads the frozen production outputs."
        )
    variant_id = variant["variant_id"]
    dirs = _variant_data_dirs(experiment_id, variant_id, output_root)
    landsat = landsat_export_plan(variant, baseline_years, window_days)
    modis = modis_export_plan(variant, experiment_id, output_root)
    assert_no_forbidden_products(landsat["roles"])

    jobs: list[dict] = []
    for role in landsat["roles"]:
        role_name = role["role"]
        family_key = (
            role_name if role["scope"] == "current_window"
            else f"baseline_{role['family']}"
        )
        bands = LANDSAT_ROLE_BANDS[family_key]
        directory = dirs[LANDSAT_ROLE_CONTEXT_DIR[family_key]]
        semantics = landsat_job_date_semantics(role["start_date"], role["end_date"])
        for product in LANDSAT_PRODUCTS_PER_ROLE:
            jobs.append({
                "artifact_id": f"{role_name}__{product}",
                "role": role_name,
                "family": role["family"],
                "scope": role["scope"],
                "baseline_year": role.get("baseline_year"),
                "product": product,
                "band": bands[product],
                "grid_family": GRID_FAMILY_LANDSAT,
                "export_scale_m": LANDSAT_EXPORT_SCALE_M,
                "expected_band_count": 1,
                "is_count_product": product in COUNT_PRODUCTS,
                "start_date": role["start_date"],
                "end_date": role["end_date"],
                "date_semantics": semantics,
                "calendar_month_filter": role.get("calendar_month_filter"),
                "calendar_month_filter_redundant": role.get(
                    "calendar_month_filter_redundant"
                ),
                "output_path": str(directory / f"{role_name}__{product}.tif"),
                "producer": (
                    "src.step3_landsat_lst.get_current_period_median / "
                    "get_current_period_ndvi_median / "
                    "get_landsat_baseline_window_median_collection / "
                    "get_landsat_baseline_window_ndvi_collection"
                ),
                "uses_variant_context": True,
            })

    modis_semantics = modis_job_date_semantics(modis["start_date"], modis["end_date"])
    for role in modis["roles"]:
        jobs.append({
            "artifact_id": role["role"],
            "role": role["role"],
            "family": "modis",
            "scope": role["scope"],
            "baseline_year": None,
            "product": role["role"],
            "band": None,
            "grid_family": GRID_FAMILY_MODIS,
            "export_scale_m": MODIS_EXPORT_SCALE_M,
            "expected_band_count": 1,
            "is_count_product": role["role"] in COUNT_PRODUCTS,
            "start_date": role["start_date"],
            "end_date": role["end_date"],
            "date_semantics": modis_semantics,
            "output_path": str(
                dirs["modis_input_dir"] / MODIS_ROLE_FILENAMES[role["role"]]
            ),
            "producer": role["producer"],
            "uses_variant_context": True,
        })

    jobs.sort(key=lambda job: job["artifact_id"])
    assert_predictor_job_set(jobs, baseline_years, variant)
    return jobs


def assert_predictor_job_set(
    jobs: Sequence[dict], baseline_years: Sequence[int], variant: dict,
) -> None:
    """Refuse a duplicate, missing, extra or forbidden artefact."""
    expected_roles = (
        {"current_lst", "current_ndvi"}
        | {f"baseline_lst_{year}" for year in baseline_years}
        | {f"baseline_ndvi_{year}" for year in baseline_years}
        | set(MODIS_ROLE_ORDER)
    )
    roles = {job["role"] for job in jobs}
    missing = sorted(expected_roles - roles)
    extra = sorted(roles - expected_roles)
    if missing:
        raise WindowClosureError(
            f"Variant '{variant['variant_id']}' is missing predictor role(s): "
            f"{missing}."
        )
    if extra:
        raise WindowClosureError(
            f"Variant '{variant['variant_id']}' plans forbidden/unexpected "
            f"predictor role(s): {extra}."
        )

    ids = [job["artifact_id"] for job in jobs]
    if len(set(ids)) != len(ids):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        raise WindowClosureError(
            f"Variant '{variant['variant_id']}' plans duplicate artefact "
            f"id(s): {duplicates}."
        )
    paths = [job["output_path"] for job in jobs]
    if len(set(paths)) != len(paths):
        duplicates = sorted({value for value in paths if paths.count(value) > 1})
        raise WindowClosureError(
            f"Variant '{variant['variant_id']}' plans duplicate output "
            f"path(s): {duplicates}."
        )
    for job in jobs:
        if job["product"] in FORBIDDEN_LANDSAT_PRODUCTS or "date_balanced" in job["product"]:
            raise WindowClosureError(
                f"Variant '{variant['variant_id']}' plans reducer-"
                f"counterfactual product '{job['product']}'; the window-closure "
                "predictor export is scene-weighted only."
            )
    if len(roles) != expected_logical_role_count(baseline_years):
        raise WindowClosureError(
            f"Variant '{variant['variant_id']}' plans {len(roles)} logical "
            f"roles, expected {expected_logical_role_count(baseline_years)}."
        )
    if len(jobs) != expected_raster_count(baseline_years):
        raise WindowClosureError(
            f"Variant '{variant['variant_id']}' plans {len(jobs)} rasters, "
            f"expected {expected_raster_count(baseline_years)}."
        )


def assert_jobs_inside_variant_namespace(
    experiment_id: str, variant_id: str, jobs: Sequence[dict],
    output_root: Optional[Path] = None,
) -> None:
    """Every mutable output path must sit inside this variant's namespace.

    Checked BEFORE Earth Engine is imported, so a mis-pointed context can never
    reach a live export.
    """
    allowed = variant_root(experiment_id, variant_id, output_root).resolve()
    canonical_variant = variant_root(
        experiment_id, CANONICAL_VARIANT_ID, output_root,
    ).resolve()
    for job in jobs:
        resolved = Path(job["output_path"]).resolve()
        if resolved == canonical_variant or canonical_variant in resolved.parents:
            raise WindowClosureError(
                f"Predictor artefact '{job['artifact_id']}' would be written "
                f"into the CANONICAL variant namespace: {resolved}."
            )
        if allowed not in resolved.parents:
            raise WindowClosureError(
                f"Predictor artefact '{job['artifact_id']}' escapes the "
                f"variant namespace {allowed}: {resolved}."
            )
        if resolved.suffix != ".tif":
            raise WindowClosureError(
                f"Predictor artefact '{job['artifact_id']}' is not a GeoTIFF: "
                f"{resolved}."
            )


def predictor_variant_plan(
    experiment_id: str,
    variant: dict,
    baseline_years: Sequence[int],
    window_days: int,
    output_root: Optional[Path] = None,
) -> dict:
    """Pure per-variant plan, shared by the dry run and the actual stage."""
    if variant["is_canonical"]:
        return {
            "variant_id": variant["variant_id"],
            "shift_days": variant["shift_days"],
            "predictor_start_date": variant["predictor_start_date"],
            "predictor_end_date": variant["predictor_end_date"],
            "lead_days": variant["lead_days"],
            "baseline_years": list(baseline_years),
            "export_enabled": False,
            "frozen_reference_only": True,
            "expected_logical_roles": [],
            "expected_logical_role_count": 0,
            "expected_artifacts": [],
            "expected_raster_count": 0,
            "reason": (
                "The canonical variant reads the frozen production outputs; "
                "re-exporting it would replace the very reference the early "
                "closures are compared against."
            ),
        }
    jobs = predictor_artifact_jobs(
        experiment_id, variant, baseline_years, window_days, output_root,
    )
    assert_jobs_inside_variant_namespace(
        experiment_id, variant["variant_id"], jobs, output_root,
    )
    return {
        "variant_id": variant["variant_id"],
        "shift_days": variant["shift_days"],
        "predictor_start_date": variant["predictor_start_date"],
        "predictor_end_date": variant["predictor_end_date"],
        "lead_days": variant["lead_days"],
        "baseline_years": list(baseline_years),
        "export_enabled": True,
        "frozen_reference_only": False,
        "expected_logical_roles": sorted({job["role"] for job in jobs}),
        "expected_logical_role_count": len({job["role"] for job in jobs}),
        "expected_artifacts": jobs,
        "expected_raster_count": len(jobs),
        "reducer": "scene_weighted",
        "metadata_path": str(
            variant_root(experiment_id, variant["variant_id"], output_root)
            / PREDICTOR_METADATA_NAME
        ),
        "landsat_date_semantics": {
            job["role"]: job["date_semantics"]
            for job in jobs if job["family"] in ("lst", "ndvi")
        },
        "modis_date_semantics": next(
            (job["date_semantics"] for job in jobs if job["family"] == "modis"), None
        ),
        "output_paths": [job["output_path"] for job in jobs],
    }


def predictor_export_summary(
    experiment_id: str,
    variants: Sequence[dict],
    baseline_years: Sequence[int],
    window_days: int,
    output_root: Optional[Path] = None,
) -> dict:
    """The whole-analysis predictor plan, as reported by a dry run."""
    plans = {
        variant["variant_id"]: predictor_variant_plan(
            experiment_id, variant, baseline_years, window_days, output_root,
        )
        for variant in variants
    }
    early = nonzero_variants(variants)
    total = sum(plans[variant["variant_id"]]["expected_raster_count"] for variant in early)
    every_job = [
        job for variant in early
        for job in plans[variant["variant_id"]]["expected_artifacts"]
    ]
    root = experiment_root(experiment_id, output_root).resolve()
    contained = all(
        root in Path(job["output_path"]).resolve().parents for job in every_job
    )
    forbidden = any(
        "date_balanced" in job["product"] for job in every_job
    )
    return {
        "canonical_export_enabled": False,
        "nonzero_variant_ids": [variant["variant_id"] for variant in early],
        "logical_roles_per_variant": expected_logical_role_count(baseline_years),
        "rasters_per_variant": expected_raster_count(baseline_years),
        "total_planned_rasters": total,
        "reducer": "scene_weighted",
        "forbidden_products_present": forbidden,
        "all_paths_inside_dedicated_namespace": contained,
        "baseline_years": list(baseline_years),
        "variant_plans": plans,
        "limitations": list(PREDICTOR_EXPORT_LIMITATIONS),
    }


# --- Binding -----------------------------------------------------------------
PREDICTOR_BINDING_DOCUMENTS: tuple[str, ...] = (
    "config/preregistration.json",
    "config/frozen_input_inventory.json",
    "config/window_variants.csv",
    "prelabel_censor/export_plan.json",
    "prelabel_censor/censoring_summary.json",
    "prelabel_censor/prelabel_export_checkpoint.json",
)


def prelabel_raster_path(
    experiment_id: str, output_root: Optional[Path] = None,
) -> Path:
    return prelabel_output_paths(experiment_id, output_root)["raster"]


# The shared pre-label censoring raster joins the frozen set for THIS stage: it
# is produced by the previous stage and must not move underneath this one.
PRELABEL_FROZEN_ROLE = "prelabel_burndate"

# Frozen inputs whose hash is REQUIRED by the predictor stage: the analysis
# identity roles plus the pre-label raster.
#
# Everything else the inventory happens to carry is CONVENIENCE metadata.
# `canonical_step8a_stats` is the clearest case: it is a resolver side-file
# (it can name the raw BurnDate label Step8A actually used), it is deliberately
# absent from REQUIRED_FROZEN_INPUT_ROLES, and the analysis identity does not
# depend on it existing. Treating "every inventory entry must hash" as the
# contract would report such a side-file as if it bound the frozen scientific
# identity -- which is exactly what it does not do.
REQUIRED_PREDICTOR_FROZEN_HASH_ROLES: tuple[str, ...] = (
    REQUIRED_FROZEN_INPUT_ROLES + (PRELABEL_FROZEN_ROLE,)
)


def missing_required_frozen_hashes(
    inventory: dict,
    required: Sequence[str] = REQUIRED_PREDICTOR_FROZEN_HASH_ROLES,
) -> list[str]:
    """Required roles that are absent from `inventory` or carry no hash.

    Only the roles named in `required` are identity-bearing, so an optional
    convenience role without a hash is never returned.
    """
    return sorted(
        role for role in required
        if not isinstance(inventory.get(role), dict)
        or inventory[role].get("sha256") is None
    )


def optional_frozen_roles(
    inventory: dict,
    required: Sequence[str] = REQUIRED_PREDICTOR_FROZEN_HASH_ROLES,
) -> list[str]:
    """Inventory roles that are recorded but do NOT bind the analysis identity."""
    known = set(required)
    return sorted(role for role in inventory if role not in known)


def predictor_frozen_inputs(
    experiment_id: str, inventory: dict, output_root: Optional[Path] = None,
) -> dict:
    """Frozen inputs this stage must not disturb, including the prelabel raster."""
    extended = dict(inventory)
    raster = prelabel_raster_path(experiment_id, output_root)
    extended[PRELABEL_FROZEN_ROLE] = {
        "path": str(raster),
        "exists": raster.is_file(),
        "sha256": sha256_file(raster) if raster.is_file() else None,
    }
    return extended


def assert_predictor_binding(
    experiment_id: str,
    analysis_id: str,
    shifts: Sequence[int],
    canonical: dict,
    variants: Sequence[dict],
    censor: dict,
    inventory: dict,
    planned_paths: dict[str, str],
    output_root: Optional[Path] = None,
) -> dict:
    """Bind to the completed plan AND prelabel stages -- read only.

    Runs entirely before Earth Engine is imported: a disagreeing document, a
    failed prelabel stage or a moved frozen input stops the stage with nothing
    created.
    """
    binding = assert_plan_binding(
        experiment_id, analysis_id, shifts, censor, inventory, planned_paths,
    )
    root = experiment_root(experiment_id, output_root)

    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise WindowClosureError(
                f"Predictor binding failed: {message} No Earth Engine call was "
                "made and nothing was written."
            )

    for relative in PREDICTOR_BINDING_DOCUMENTS:
        _require((root / relative).is_file(), f"'{relative}' is missing.")

    # --- Variant CSV: the frozen per-variant dates -------------------------
    rows = list(csv.DictReader(
        (root / "config" / "window_variants.csv").read_text(encoding="utf-8").splitlines()
    ))
    csv_by_id = {row["variant_id"]: row for row in rows}
    _require(
        set(csv_by_id) == {variant["variant_id"] for variant in variants},
        f"window_variants.csv lists {sorted(csv_by_id)}, but this run derived "
        f"{sorted(v['variant_id'] for v in variants)}.",
    )
    for variant in variants:
        row = csv_by_id[variant["variant_id"]]
        for key in ("predictor_start_date", "predictor_end_date"):
            _require(
                row[key] == variant[key],
                f"window_variants.csv {variant['variant_id']}.{key}="
                f"{row[key]!r} differs from the derived {variant[key]!r}.",
            )
        _require(
            int(row["shift_days"]) == int(variant["shift_days"])
            and int(row["lead_days"]) == int(variant["lead_days"]),
            f"window_variants.csv {variant['variant_id']} shift/lead differ "
            "from the derived variant.",
        )

    # --- Canonical frozen reference ----------------------------------------
    reference = _read_plan_document(
        root / "variants" / CANONICAL_VARIANT_ID / "frozen_reference.json",
        f"variants/{CANONICAL_VARIANT_ID}/frozen_reference.json",
    )
    _require(reference.get("analysis_id") == analysis_id,
             "the canonical frozen reference holds a different analysis_id.")
    for key in ("predictor_export_planned", "landsat_export_planned",
                "modis_export_planned"):
        _require(reference.get(key) is False,
                 f"the canonical frozen reference has {key}={reference.get(key)!r}; "
                 "the canonical variant must never be exported.")
    _require(reference.get("is_canonical") is True,
             "the canonical frozen reference is not marked canonical.")

    # --- Non-zero variant export plans -------------------------------------
    preregistered_baselines = list(canonical["baseline_years"])
    for variant in nonzero_variants(variants):
        relative = f"variants/{variant['variant_id']}/export_plan.json"
        plan = _read_plan_document(root / relative, relative)
        _require(plan.get("analysis_id") == analysis_id,
                 f"'{relative}' holds a different analysis_id.")
        for key in ("predictor_start_date", "predictor_end_date", "lead_days"):
            _require(
                plan.get(key) == variant[key],
                f"'{relative}' {key}={plan.get(key)!r} differs from the derived "
                f"{variant[key]!r}.",
            )
        _require(plan.get("reducer") == "scene_weighted",
                 f"'{relative}' reducer is {plan.get('reducer')!r}.")
        landsat = plan.get("landsat") or {}
        current_roles = landsat.get("current_roles") or []
        baseline_roles = landsat.get("baseline_roles") or []
        _require(len(current_roles) == 2,
                 f"'{relative}' has {len(current_roles)} current Landsat roles.")
        _require(
            len(baseline_roles) == 2 * len(preregistered_baselines),
            f"'{relative}' has {len(baseline_roles)} baseline Landsat roles, "
            f"expected {2 * len(preregistered_baselines)}.",
        )
        plan_years = sorted({role.get("baseline_year") for role in baseline_roles})
        _require(
            plan_years == sorted(preregistered_baselines),
            f"'{relative}' baseline years {plan_years} differ from the "
            f"preregistered {sorted(preregistered_baselines)}.",
        )
        # Structural product audit -- NEVER a raw substring over the document.
        # Valid plans legitimately mention date_balanced in forbidden_products/
        # reducer_note/limitations in order to BAN it; only the real product
        # fields (role `products` lists, artefact `product` fields) count.
        violations = landsat_product_violations(plan)
        _require(
            not violations,
            f"'{relative}' plans {'; '.join(violations)}.",
        )
        modis_plan = plan.get("modis") or {}
        _require(
            modis_plan.get("start_date") == variant["predictor_start_date"]
            and modis_plan.get("end_date") == variant["predictor_end_date"],
            f"'{relative}' MODIS dates do not use the variant window.",
        )

    # --- Pre-label stage ----------------------------------------------------
    summary = _read_plan_document(
        root / "prelabel_censor" / "censoring_summary.json",
        "prelabel_censor/censoring_summary.json",
    )
    checkpoint = _read_plan_document(
        root / "prelabel_censor" / "prelabel_export_checkpoint.json",
        "prelabel_censor/prelabel_export_checkpoint.json",
    )
    _require(summary.get("analysis_id") == analysis_id,
             "the pre-label summary holds a different analysis_id.")
    _require(checkpoint.get("analysis_id") == analysis_id,
             "the pre-label checkpoint holds a different analysis_id.")
    _require(summary.get("status") == STATUS_PASS,
             f"the pre-label stage status is {summary.get('status')!r}, not "
             f"{STATUS_PASS!r}.")
    _require(summary.get("grid_matches_reference") is True,
             "the pre-label raster did not match the reference grid.")
    _require(summary.get("frozen_hashes_unchanged") is True,
             "the pre-label stage did not confirm unchanged frozen hashes.")

    raster = prelabel_raster_path(experiment_id, output_root)
    _require(raster.is_file(), f"the pre-label raster is missing at {raster}.")
    digest = sha256_file(raster)
    _require(summary.get("raster_sha256") == digest,
             "the pre-label raster hash differs from the censoring summary.")
    _require(checkpoint.get("raster_sha256") == digest,
             "the pre-label raster hash differs from the export checkpoint.")

    return {
        **binding,
        "bound_to_prelabel": True,
        "prelabel_raster_sha256": digest,
        "prelabel_status": summary.get("status"),
        "prelabel_burn_cell_count": summary.get("prelabel_burn_cell_count"),
        "canonical_frozen_reference_verified": True,
        "nonzero_variant_ids": [v["variant_id"] for v in nonzero_variants(variants)],
        "baseline_years": list(preregistered_baselines),
    }


# --- Raster contract ---------------------------------------------------------
def expected_family_pixel_size(reference: dict, grid_family: str) -> float:
    """Pixel size a family's production export scale implies, in reference units.

    The reference grid is the frozen 30 m canonical label raster, so a family
    exported at S metres must have `reference_pixel_size * S / 30`.
    """
    reference_pixel = abs(float((reference.get("transform") or [0.0])[0]))
    scale = GRID_FAMILY_SCALES[grid_family]
    return reference_pixel * (scale / LANDSAT_EXPORT_SCALE_M)


def inspect_predictor_raster(path: Path, job: dict, reference: dict) -> dict:
    """Full per-raster contract. Raises on anything that would poison a variant."""
    import numpy as np
    import rasterio

    artifact = job["artifact_id"]
    if not path.is_file():
        raise WindowClosureError(f"Predictor artefact '{artifact}' was not produced: {path}.")
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise WindowClosureError(f"Predictor artefact '{artifact}' is empty (0 bytes): {path}.")

    try:
        with rasterio.open(path) as dataset:
            crs = dataset.crs
            transform = dataset.transform
            width, height, band_count = int(dataset.width), int(dataset.height), int(dataset.count)
            dtype = str(dataset.dtypes[0])
            nodata = dataset.nodata
            band = dataset.read(1, masked=True)
    except WindowClosureError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any reader failure is a contract failure
        raise WindowClosureError(
            f"Predictor artefact '{artifact}' at {path} could not be read: "
            f"{type(exc).__name__}: {exc}."
        ) from exc

    if crs is None:
        raise WindowClosureError(f"Predictor artefact '{artifact}' has no CRS: {path}.")
    if transform is None or not transform.is_rectilinear:
        raise WindowClosureError(
            f"Predictor artefact '{artifact}' has no usable transform: {path}."
        )
    if width <= 0 or height <= 0:
        raise WindowClosureError(
            f"Predictor artefact '{artifact}' has a non-positive shape "
            f"({width}x{height}): {path}."
        )
    if band_count != job["expected_band_count"]:
        raise WindowClosureError(
            f"Predictor artefact '{artifact}' has {band_count} band(s), expected "
            f"{job['expected_band_count']}: {path}."
        )

    signature = {
        "crs": str(crs),
        "transform": [float(value) for value in tuple(transform)[:6]],
        "width": width,
        "height": height,
        "band_count": band_count,
        "dtype": dtype,
        "nodata": None if nodata is None else float(nodata),
    }
    if signature["crs"] != reference.get("crs"):
        raise WindowClosureError(
            f"Predictor artefact '{artifact}' CRS {signature['crs']!r} differs "
            f"from the production CRS {reference.get('crs')!r}: {path}."
        )
    expected_pixel = expected_family_pixel_size(reference, job["grid_family"])
    actual_pixel = abs(signature["transform"][0])
    if abs(actual_pixel - expected_pixel) > GRID_PIXEL_SIZE_RELATIVE_TOLERANCE * expected_pixel:
        raise WindowClosureError(
            f"Predictor artefact '{artifact}' pixel size {actual_pixel} does not "
            f"match the {job['grid_family']} production grid contract "
            f"({expected_pixel}, i.e. {GRID_FAMILY_SCALES[job['grid_family']]} m): "
            f"{path}."
        )

    values = band.compressed().astype("float64")
    if values.size and not np.all(np.isfinite(values)):
        raise WindowClosureError(
            f"Predictor artefact '{artifact}' carries non-finite (+/-inf or NaN) "
            f"values outside its nodata mask: {path}."
        )
    finite_cell_count = int(values.size)

    if job["is_count_product"]:
        if finite_cell_count and float(values.min()) < 0.0:
            raise WindowClosureError(
                f"Observation-count artefact '{artifact}' carries negative "
                f"value(s) (min={float(values.min())}): {path}."
            )
        if finite_cell_count and not np.all(np.equal(np.mod(values, 1.0), 0.0)):
            raise WindowClosureError(
                f"Observation-count artefact '{artifact}' carries fractional "
                f"value(s); counts must be whole numbers: {path}."
            )
    elif finite_cell_count == 0:
        # A count raster may legitimately be all zero, but a scientific value
        # raster that is entirely nodata carries no predictor at all.
        raise WindowClosureError(
            f"Predictor artefact '{artifact}' is entirely nodata; it carries no "
            f"usable predictor values: {path}."
        )

    return {
        "artifact_id": artifact,
        "role": job["role"],
        "family": job["family"],
        "scope": job["scope"],
        "baseline_year": job["baseline_year"],
        "product": job["product"],
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(size_bytes),
        "band_count": band_count,
        "dtype": dtype,
        "nodata": signature["nodata"],
        "width": width,
        "height": height,
        "crs": signature["crs"],
        "transform": signature["transform"],
        "grid_signature": signature,
        "grid_family": job["grid_family"],
        "grid_contract_passed": True,
        "finite_cell_count": finite_cell_count,
        "min_finite": float(values.min()) if finite_cell_count else None,
        "max_finite": float(values.max()) if finite_cell_count else None,
        "mask_semantics": (
            "rasterio masked read; nodata and unset pixels are excluded from "
            "every count and statistic reported here."
        ),
        "is_count_product": job["is_count_product"],
        "start_date": job["start_date"],
        "end_date": job["end_date"],
        "date_semantics": job["date_semantics"],
        "export_transport": job.get("export_transport"),
        "producer": job["producer"],
        "uses_variant_context": True,
    }


# --- The production engine (Earth Engine; imported lazily) -------------------
def production_predictor_engine(variant_context: dict, variant: dict, jobs: Sequence[dict]) -> dict:
    """Export every artefact of ONE variant using the production builders.

    Nothing scientific is defined here: the Landsat images come from
    `src.step3_landsat_lst`, the MODIS products from
    `scripts.prepare_modis_for_step7.prepare_modis_for_step7`, and every raster
    goes through the production `export_image_direct_or_tiled`.
    """
    import ee  # noqa: PLC0415  -- Earth Engine enters the process only here

    from core.config import EXPORT_CRS, GEE_PROJECT
    from core.experiment_context import get_region
    from core.gee_utils import init_gee
    from scripts.prepare_modis_for_step7 import prepare_modis_for_step7
    from scripts.run_predictors_only import export_image_direct_or_tiled
    import src.step3_landsat_lst as step3

    init_gee(GEE_PROJECT)
    region = get_region(variant_context)
    region_name = variant_context["region_key"]
    end_date = variant_context["current_period_end_date"]
    window_days = variant_context["current_period_days"]
    data_root = Path(variant_context["data_root"])

    landsat_jobs = [job for job in jobs if job["family"] in ("lst", "ndvi")]
    modis_jobs = [job for job in jobs if job["family"] == "modis"]

    # --- Production images, built ONCE per variant -------------------------
    images: dict[str, Any] = {}
    if any(job["role"] == "current_lst" for job in landsat_jobs):
        images["current_lst"], _ = step3.get_current_period_median(
            region, region_name, end_date, window_days,
        )
    if any(job["role"] == "current_ndvi" for job in landsat_jobs):
        images["current_ndvi"], _ = step3.get_current_period_ndvi_median(
            region, region_name, end_date, window_days,
        )

    baseline_years = sorted({
        int(job["baseline_year"]) for job in landsat_jobs
        if job["baseline_year"] is not None
    })
    if baseline_years:
        baseline_start = variant_context["baseline_start_date"]
        baseline_end = variant_context["baseline_end_date"]
        lst_collection, lst_meta = step3.get_landsat_baseline_window_median_collection(
            region, region_name, end_date, window_days,
            baseline_start=baseline_start, baseline_end=baseline_end,
        )
        ndvi_collection, ndvi_meta = step3.get_landsat_baseline_window_ndvi_collection(
            region, region_name, end_date, window_days,
            baseline_start=baseline_start, baseline_end=baseline_end,
        )
        for name, meta in (("LST", lst_meta), ("NDVI", ndvi_meta)):
            produced = sorted(int(year) for year in meta["baseline_years"])
            if produced != baseline_years:
                raise WindowClosureError(
                    f"Production baseline {name} years {produced} differ from "
                    f"the preregistered {baseline_years} for variant "
                    f"'{variant['variant_id']}'."
                )
            for record in meta["windows"]:
                expected = _baseline_year_window(end_date, window_days, int(record["year"]))
                if (record["window_start"], record["window_end"]) != (
                    expected["start_date"], expected["end_date"],
                ):
                    raise WindowClosureError(
                        f"Production baseline {name} window for "
                        f"{record['year']} is "
                        f"{record['window_start']}..{record['window_end']}, "
                        f"expected {expected['start_date']}..{expected['end_date']}."
                    )
        for year in baseline_years:
            images[f"baseline_lst_{year}"] = ee.Image(
                lst_collection.filter(ee.Filter.eq("baseline_year", year)).first()
            )
            images[f"baseline_ndvi_{year}"] = ee.Image(
                ndvi_collection.filter(ee.Filter.eq("baseline_year", year)).first()
            )

    results: dict[str, dict] = {}
    for job in sorted(landsat_jobs, key=lambda item: item["artifact_id"]):
        out_path = Path(job["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        band_image = images[job["role"]].select(job["band"])
        outcome = export_image_direct_or_tiled(
            band_image, out_path, region,
            scale=LANDSAT_EXPORT_SCALE_M, crs=EXPORT_CRS,
            label=f"{variant['variant_id']}__{job['artifact_id']}",
            force=True,
            tiles_dir=data_root / "_tiles" / job["artifact_id"],
            band_count=1,
        )
        results[job["artifact_id"]] = {
            "path": Path(outcome["path"]),
            "transport": outcome["transport"],
        }

    if modis_jobs:
        modis_result = prepare_modis_for_step7(variant_context, force=True)
        produced = {
            "modis_lst_mean": Path(modis_result["mean_path"]),
            "modis_lst_std": Path(modis_result["std_path"]),
            "modis_valid_observation_count": Path(modis_result["valid_count_path"]),
        }
        for job in modis_jobs:
            actual = produced[job["role"]].resolve()
            if actual != Path(job["output_path"]).resolve():
                raise WindowClosureError(
                    f"Production MODIS wrote '{job['role']}' to {actual}, but "
                    f"the plan expects {job['output_path']}."
                )
            results[job["artifact_id"]] = {
                "path": actual,
                "transport": modis_result.get("status", "production_modis"),
            }
    return results


# --- The stage ---------------------------------------------------------------
def predictor_metadata_path(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> Path:
    return (
        variant_root(experiment_id, variant_id, output_root) / PREDICTOR_METADATA_NAME
    )


def build_predictor_variant_context(
    experiment_id: str,
    variant: dict,
    base_context: dict,
    output_root: Optional[Path] = None,
) -> dict:
    """A variant context whose MUTABLE paths all live in the variant namespace.

    Built on `build_window_variant_context`, which already deep-copies the
    registry context, rewrites every output path and asserts containment. The
    only addition is the explicit opt-in that lets the production MODIS guard
    accept this dedicated diagnostics root -- the guard still refuses anything
    outside `outputs/` and anything inside `outputs/experiments/`.
    """
    ctx = build_window_variant_context(
        experiment_id, variant["shift_days"], base_context, output_root,
    )
    if ctx["window_closure_variant_id"] != variant["variant_id"]:
        raise WindowClosureError(
            f"Variant context is for '{ctx['window_closure_variant_id']}', "
            f"expected '{variant['variant_id']}'."
        )
    if ctx["predictor_start_date"] != variant["predictor_start_date"] or \
            ctx["predictor_end_date"] != variant["predictor_end_date"]:
        raise WindowClosureError(
            f"Variant context dates {ctx['predictor_start_date']}.."
            f"{ctx['predictor_end_date']} differ from the preregistered "
            f"{variant['predictor_start_date']}..{variant['predictor_end_date']}."
        )
    ctx[MODIS_NAMESPACE_ALLOWED_ROOTS_KEY] = [
        variant_root(experiment_id, variant["variant_id"], output_root)
    ]
    return ctx


def _quarantine_predictor_artifact(path: Path, variant_dir: Path) -> Optional[str]:
    """Move an artefact aside instead of deleting it. Never removes data."""
    if not path.is_file():
        return None
    digest = sha256_file(path)[:12]
    target = variant_dir / PREDICTOR_QUARANTINE_DIR / f"{path.stem}.{digest}{path.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, target)
    return str(target)


def predictor_variant_is_reusable(
    experiment_id: str, analysis_id: str, variant: dict, jobs: Sequence[dict],
    reference: dict, output_root: Optional[Path] = None,
) -> tuple[bool, Optional[dict], str]:
    """Whether a previously exported variant may be reused untouched.

    Requires the metadata to exist with the SAME analysis_id and status=pass,
    every planned artefact to exist, every hash to match the metadata, and the
    full raster contract to pass again. Anything else is not reusable -- a
    partial variant is never silently accepted.
    """
    metadata_path = predictor_metadata_path(experiment_id, variant["variant_id"], output_root)
    if not metadata_path.is_file():
        return False, None, "no predictor metadata"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False, None, "unreadable predictor metadata"
    if not isinstance(metadata, dict):
        return False, None, "predictor metadata is not a JSON object"
    if metadata.get("analysis_id") != analysis_id:
        return False, None, "analysis_id mismatch"
    if metadata.get("status") != STATUS_PASS:
        return False, None, f"previous status is {metadata.get('status')!r}"

    recorded = metadata.get("artifact_sha256") or {}
    for job in jobs:
        path = Path(job["output_path"])
        if not path.is_file():
            return False, metadata, f"missing artefact {job['artifact_id']}"
        digest = recorded.get(job["artifact_id"])
        if digest is None:
            return False, metadata, f"no recorded hash for {job['artifact_id']}"
        if sha256_file(path) != digest:
            return False, metadata, f"hash mismatch for {job['artifact_id']}"
        try:
            inspect_predictor_raster(path, job, reference)
        except WindowClosureError as exc:
            return False, metadata, f"raster contract failed for {job['artifact_id']}: {exc}"
    if int(metadata.get("produced_raster_count") or 0) != len(jobs):
        return False, metadata, "recorded raster count differs from the plan"
    return True, metadata, "complete and verified"


def run_predictor_export(
    experiment_id: str,
    analysis_id: str,
    base_context: dict,
    canonical: dict,
    variants: Sequence[dict],
    inventory: dict,
    binding: dict,
    output_root: Optional[Path] = None,
    force: bool = False,
    resume: bool = False,
    engine: Optional[Any] = None,
) -> dict:
    """Export (or reuse) the predictors of every non-canonical variant.

    Variants are processed in increasing shift order and are fully independent:
    a later failure never invalidates or deletes an earlier variant's verified
    outputs, and a variant only gets `status=pass` metadata once every one of
    its rasters passed the contract.

    `engine` is the dependency-injection point; the default is the Earth Engine
    production engine, imported lazily so no dry run or test pulls `ee` in.
    """
    baseline_years = list(canonical["baseline_years"])
    window_days = canonical["current_period_days"]
    reference = read_grid_signature(reference_grid_source(inventory))
    early = nonzero_variants(variants)
    if not early:
        raise WindowClosureError(
            "No non-canonical variant is preregistered, so there is no "
            "predictor to export."
        )
    # The binding is the authority on WHICH variants and baseline years this
    # analysis identity covers; a disagreement here means the plan on disk and
    # the run have drifted apart.
    if list(binding.get("nonzero_variant_ids") or []) != [v["variant_id"] for v in early]:
        raise WindowClosureError(
            f"Plan binding covers {binding.get('nonzero_variant_ids')!r} but "
            f"this run derived {[v['variant_id'] for v in early]!r}."
        )
    if sorted(int(y) for y in (binding.get("baseline_years") or [])) != \
            sorted(int(y) for y in baseline_years):
        raise WindowClosureError(
            f"Plan binding baseline years {binding.get('baseline_years')!r} "
            f"differ from the derived {baseline_years!r}."
        )

    frozen_before = frozen_hash_map(
        predictor_frozen_inputs(experiment_id, inventory, output_root)
    )

    files_written: list[str] = []
    files_rewritten: list[str] = []
    processed: list[str] = []
    reused_variants: list[str] = []
    exported_variants: list[str] = []
    quarantined: list[str] = []
    variant_reports: dict[str, dict] = {}
    logical_roles_produced = 0
    rasters_produced = 0
    gee_query_run = False
    gee_export_run = False

    for variant in early:
        variant_id = variant["variant_id"]
        vroot = variant_root(experiment_id, variant_id, output_root)
        jobs = predictor_artifact_jobs(
            experiment_id, variant, baseline_years, window_days, output_root,
        )
        assert_jobs_inside_variant_namespace(experiment_id, variant_id, jobs, output_root)
        metadata_path = predictor_metadata_path(experiment_id, variant_id, output_root)

        reusable, previous, reason = predictor_variant_is_reusable(
            experiment_id, analysis_id, variant, jobs, reference, output_root,
        )
        if reusable and not force:
            if not resume:
                raise WindowClosureError(
                    f"Variant '{variant_id}' already has a complete, verified "
                    f"predictor export at {metadata_path}. Refusing to "
                    "overwrite it silently: re-run with resume=True to reuse "
                    "it, or force=True to re-export it (old rasters are "
                    "quarantined, never deleted)."
                )
            processed.append(variant_id)
            reused_variants.append(variant_id)
            files_written += [job["output_path"] for job in jobs] + [str(metadata_path)]
            logical_roles_produced += len({job["role"] for job in jobs})
            rasters_produced += len(jobs)
            variant_reports[variant_id] = {
                "variant_id": variant_id, "reused": True, "reason": reason,
                "raster_count": len(jobs), "status": STATUS_PASS,
            }
            continue
        if not resume and not force and previous is not None:
            raise WindowClosureError(
                f"Variant '{variant_id}' has an existing but NOT reusable "
                f"predictor export ({reason}) at {metadata_path}. Refusing to "
                "overwrite it silently: inspect it, then re-run with "
                "force=True (old rasters are quarantined, never deleted)."
            )

        # Re-exporting: quarantine whatever is there, never delete it.
        for job in jobs:
            moved = _quarantine_predictor_artifact(Path(job["output_path"]), vroot)
            if moved:
                quarantined.append(moved)
        if metadata_path.is_file():
            # A stale status=pass metadata must not survive a partial re-export.
            _atomic_write_text(metadata_path, _json_document({
                "schema_version": PREDICTOR_METADATA_SCHEMA,
                "analysis_id": analysis_id,
                "experiment_id": experiment_id,
                "variant_id": variant_id,
                "status": "superseded_export_in_progress",
            }))

        variant_context = build_predictor_variant_context(
            experiment_id, variant, base_context, output_root,
        )
        active_engine = engine if engine is not None else production_predictor_engine
        gee_query_run = True
        gee_export_run = True
        outcomes = active_engine(variant_context, variant, jobs)

        inventory_records: list[dict] = []
        for job in sorted(jobs, key=lambda item: item["artifact_id"]):
            outcome = outcomes.get(job["artifact_id"]) or {}
            job = dict(job, export_transport=outcome.get("transport"))
            produced_path = Path(outcome.get("path") or job["output_path"])
            if produced_path.resolve() != Path(job["output_path"]).resolve():
                raise WindowClosureError(
                    f"Variant '{variant_id}' artefact '{job['artifact_id']}' was "
                    f"written to {produced_path}, not to the planned "
                    f"{job['output_path']}."
                )
            inventory_records.append(
                inspect_predictor_raster(produced_path, job, reference)
            )

        produced_roles = sorted({record["role"] for record in inventory_records})
        metadata = build_predictor_metadata(
            experiment_id, analysis_id, variant, baseline_years,
            inventory_records, produced_roles, frozen_before,
            frozen_hash_map(predictor_frozen_inputs(experiment_id, inventory, output_root)),
        )
        assert_frozen_hashes_unchanged(
            frozen_before, metadata["frozen_input_sha256_after"],
            f"while exporting variant '{variant_id}'",
        )
        _atomic_write_text(metadata_path, _json_document(metadata))

        processed.append(variant_id)
        exported_variants.append(variant_id)
        files_written += [record["path"] for record in inventory_records] + [str(metadata_path)]
        files_rewritten += [record["path"] for record in inventory_records] + [str(metadata_path)]
        logical_roles_produced += len(produced_roles)
        rasters_produced += len(inventory_records)
        variant_reports[variant_id] = {
            "variant_id": variant_id, "reused": False,
            "raster_count": len(inventory_records),
            "logical_role_count": len(produced_roles),
            "metadata_path": str(metadata_path),
            "status": STATUS_PASS,
        }

    frozen_after = frozen_hash_map(
        predictor_frozen_inputs(experiment_id, inventory, output_root)
    )
    assert_frozen_hashes_unchanged(frozen_before, frozen_after, "while exporting predictors")

    return {
        "files_written": sorted(set(files_written)),
        "files_rewritten": sorted(set(files_rewritten)),
        "processed_variants": processed,
        "reused_variants": reused_variants,
        "exported_variants": exported_variants,
        "quarantined_artifacts": sorted(quarantined),
        "logical_roles_produced": logical_roles_produced,
        "predictor_rasters_produced": rasters_produced,
        "gee_query_run": gee_query_run,
        "gee_export_run": gee_export_run,
        "reused": bool(processed) and not exported_variants,
        "variant_reports": variant_reports,
        "frozen_input_sha256_before": frozen_before,
        "frozen_input_sha256_after": frozen_after,
        "canonical_export_attempted": False,
    }


def build_predictor_metadata(
    experiment_id: str,
    analysis_id: str,
    variant: dict,
    baseline_years: Sequence[int],
    inventory_records: Sequence[dict],
    produced_roles: Sequence[str],
    frozen_before: dict,
    frozen_after: dict,
) -> dict:
    """The per-variant predictor-export record. Deterministic and self-describing."""
    from core.config import EXPORT_CRS

    current = [r for r in inventory_records if r["scope"] == "current_window"]
    baselines = [r for r in inventory_records if r["scope"] == "baseline_year"]
    modis = [r for r in inventory_records if r["family"] == "modis"]
    return {
        "schema_version": PREDICTOR_METADATA_SCHEMA,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "variant_id": variant["variant_id"],
        "shift_days": int(variant["shift_days"]),
        "predictor_start_date": variant["predictor_start_date"],
        "predictor_end_date": variant["predictor_end_date"],
        "lead_days": int(variant["lead_days"]),
        "duration_days": int(variant["duration_days"]),
        "label_start_date": variant["label_start_date"],
        "label_end_date": variant["label_end_date"],
        "baseline_years": [int(year) for year in baseline_years],
        "reducer": "scene_weighted",
        "production_policy": {
            "landsat_builders": (
                "src.step3_landsat_lst.get_current_period_median / "
                "get_current_period_ndvi_median / "
                "get_landsat_baseline_window_median_collection / "
                "get_landsat_baseline_window_ndvi_collection"
            ),
            "landsat_qa_mask": "src.step3_landsat_lst.apply_qa_mask",
            "landsat_reducer": "median (scene-weighted) + ImageCollection.count()",
            "landsat_export_scale_m": LANDSAT_EXPORT_SCALE_M,
            "modis_producer": MODIS_PRODUCER,
            "modis_export_scale_m": MODIS_EXPORT_SCALE_M,
            "export_crs": EXPORT_CRS,
            "exporter": (
                "scripts.run_predictors_only.export_image_direct_or_tiled"
            ),
            "allowed_products": list(LANDSAT_PRODUCTS_PER_ROLE),
            "forbidden_products": list(FORBIDDEN_LANDSAT_PRODUCTS),
            "new_formula_introduced": False,
        },
        "expected_logical_role_count": expected_logical_role_count(baseline_years),
        "produced_logical_role_count": len(produced_roles),
        "expected_raster_count": expected_raster_count(baseline_years),
        "produced_raster_count": len(inventory_records),
        "landsat_current": sorted(r["artifact_id"] for r in current),
        "landsat_baselines": sorted(r["artifact_id"] for r in baselines),
        "modis": sorted(r["artifact_id"] for r in modis),
        "artifact_inventory": sorted(inventory_records, key=lambda r: r["artifact_id"]),
        "artifact_sha256": {
            record["artifact_id"]: record["sha256"] for record in inventory_records
        },
        "raster_contract_passed": True,
        "all_paths_inside_variant_namespace": True,
        "canonical_export_attempted": False,
        "prelabel_used_as_predictor": False,
        "gee_queries_run": True,
        "gee_exports_run": True,
        "model_fit": False,
        "bootstrap_run": False,
        "frozen_input_sha256_before": dict(frozen_before),
        "frozen_input_sha256_after": dict(frozen_after),
        "frozen_hashes_unchanged": frozen_before == frozen_after,
        "canonical_outputs_modified": False,
        "limitations": list(PREDICTOR_EXPORT_LIMITATIONS),
        "status": STATUS_PASS,
    }


# =============================================================================
# Actual LOCAL-DOWNSTREAM stage
#
# Runs the PRODUCTION downstream chain -- Step5, Step5C, Step7A-E, Step8A -- on
# every NON-CANONICAL variant's exported predictors, entirely locally, so each
# variant gets its own Step8A modelling dataset built by exactly the production
# code that built the frozen canonical one.
#
# Nothing scientific is defined here. No TVDI, downscaling, fusion, aggregation,
# alignment, missing-value or feature formula is written or altered: every
# number comes out of `src.step5_preprocess_timeseries`, `src.step5c_tvdi`,
# `src.step7a..step7e` and `src.step8a_prepare_500m_modeling_dataset`, driven by
# a variant context whose only differences from the canonical one are the
# predictor-window dates and the namespace the products are written into.
#
# The canonical variant is NEVER re-run: its frozen Step8A dataset is opened
# read-only, as the reference the early closures are compared against.
# =============================================================================
LOCAL_DOWNSTREAM_STAGE = "local-downstream"
LOCAL_DOWNSTREAM_METADATA_SCHEMA = "window_closure_local_downstream.v1"
LOCAL_DOWNSTREAM_METADATA_NAME = "local_downstream_metadata.json"
LOCAL_DOWNSTREAM_ROOT_DIR = "downstream"
# Production-named copies of the variant predictors and of the frozen static
# inputs. The production steps discover their inputs BY FILE NAME (Step5 globs
# a date-stamped baseline prefix, Step5C globs `current_ndvi_median*`, Step8A
# joins `mcd64a1_raw.tif`), so the window-closure artefact names
# (`<role>__<product>.tif`) have to be re-laid-out under the production names
# before the chain can run. This is a CONTAINER operation only -- see
# `materialize_local_downstream_inputs`.
LOCAL_DOWNSTREAM_INPUT_DIR = "inputs"
LOCAL_DOWNSTREAM_QUARANTINE_DIR = "_quarantine"
LOCAL_DOWNSTREAM_QUARANTINE_KIND = "local_downstream"

# The REAL production stages this chain reuses, in the order the canonical
# Manavgat pipeline runs them. Derived from the canonical runners
# (`scripts/run_step7_downscaling_only.py`, `scripts/run_step8_modeling.py`,
# `core/pipeline_orchestrator.py`) and from the existing local-downstream
# precedent `scripts/run_landsat_composite_downstream_ab.py`. There is no
# separate production "step7"/"step8" computation: Step7 is A-E and the Step8
# dataset stage is Step8A, so no empty `step7/`/`step8/` directory is created.
PRODUCTION_STAGE_SEQUENCE: tuple[str, ...] = (
    "step5", "step5c", "step7a", "step7b", "step7c", "step7d", "step7e", "step8a",
)

# stage -> (production module, production entry point, context key it writes to)
PRODUCTION_STAGE_HELPERS: dict[str, dict[str, str]] = {
    "step5": {
        "module": "src.step5_preprocess_timeseries",
        "function": "run_step5",
        "output_context_key": "step5_output_dir",
    },
    "step5c": {
        "module": "src.step5c_tvdi",
        "function": "run_step5c",
        "output_context_key": "step5c_output_dir",
    },
    "step7a": {
        "module": "src.step7a_tiling_infrastructure",
        "function": "run_step7a",
        "output_context_key": "step7a_output_dir",
    },
    "step7b": {
        "module": "src.step7b_prepare_downscaling_dataset",
        "function": "run_step7b",
        "output_context_key": "step7b_output_dir",
    },
    "step7c": {
        "module": "src.step7c_train_downscaling_model",
        "function": "run_step7c",
        "output_context_key": "step7c_output_dir",
    },
    "step7d": {
        "module": "src.step7d_predict_downscaled_lst",
        "function": "run_step7d",
        "output_context_key": "step7d_output_dir",
    },
    "step7e": {
        "module": "src.step7e_fuse_landsat_downscaled_lst",
        "function": "run_step7e",
        "output_context_key": "step7e_output_dir",
    },
    "step8a": {
        "module": "src.step8a_prepare_500m_modeling_dataset",
        "function": "run_step8a",
        "output_context_key": "step8a_output_dir",
    },
}
# Production entry points that take no `force` keyword (they overwrite in place).
PRODUCTION_STAGES_WITHOUT_FORCE: frozenset = frozenset({"step5", "step5c"})

STEP8A_DATASET_NAME = "step8a_500m_modeling_dataset.parquet"
STEP8A_STATS_NAME = "step8a_dataset_stats.json"

# --- Production input layout --------------------------------------------------
# File names taken verbatim from `scripts/run_predictors_only.py`
# (`_export_predictors_direct`), which is the production producer of these very
# inputs. Only the ROOT they are written under is the variant's.
INPUT_ROLE_CURRENT_LST = "current_lst"
INPUT_ROLE_CURRENT_NDVI = "current_ndvi"

PRODUCTION_INPUT_CONTEXT_DIRS: dict[str, str] = {
    "current_period": "current_period_dir",
    "ndvi_current_period": "ndvi_current_dir",
    "landsat_timeseries": "baseline_input_dir",
    "ndvi_timeseries": "ndvi_baseline_dir",
    "modis": "modis_input_dir",
    "landsat_qa": "qa_dir",
    "dem": "dem_input_dir",
    "gate_inputs": None,
    "labels": "gate_labels_dir",
}

# Production Step3 exports the current Landsat products as ONE two-band image
# (value + valid count); the window-closure predictor stage took the same two
# bands out as two separate single-band rasters so each could be hashed and
# contract-checked on its own. Re-assembling them is a pure band concatenation
# of already-computed production values -- see `_stack_single_band_rasters`.
CURRENT_ROLE_BAND_ORDER: tuple[str, ...] = (
    PRODUCT_SCENE_WEIGHTED_MEDIAN, PRODUCT_SCENE_VALID_COUNT,
)

# --- What "model_fit" means for THIS stage -----------------------------------
# The production chain TRAINS one model: the Step7C MODIS->Landsat downscaling
# random forest. It is an integral part of the production PREDICTOR chain and
# is exactly what the canonical Step7D downscaled LST was built from -- it is
# NOT the fire-risk baseline/thermal model, which lives in the still-locked
# `model` stage. Reporting `model_fit=false` here would have been factually
# wrong, so the two are reported separately instead.
DOWNSCALING_MODEL_STAGE = "step7c"
LOCAL_DOWNSTREAM_MODEL_SEMANTICS: dict[str, Any] = {
    "model_fit": True,
    "downscaling_model_fit": True,
    "downscaling_model_stage": DOWNSCALING_MODEL_STAGE,
    "downscaling_model_producer": (
        "src.step7c_train_downscaling_model.run_step7c "
        "(production hyper-parameters, unchanged)"
    ),
    "fire_risk_model_fit": False,
    "fire_risk_model_stage_run": False,
    "bootstrap_run": False,
}
LOCAL_DOWNSTREAM_DRY_RUN_MODEL_SEMANTICS: dict[str, Any] = {
    # A dry run fits nothing; it only declares what an actual run WOULD fit.
    "model_fit": False,
    "downscaling_model_fit": False,
    "downscaling_model_fit_planned": True,
    "downscaling_model_stage": DOWNSCALING_MODEL_STAGE,
    "fire_risk_model_fit": False,
    "fire_risk_model_stage_run": False,
    "bootstrap_run": False,
}

# --- Baseline binding ---------------------------------------------------------
# Production Step5 discovers its baseline stack by scanning the baseline
# directory when no Step4 export manifest exists. That fallback is not
# acceptable here: an unmanaged or stale GeoTIFF in the variant namespace would
# silently enter the baseline composite. The window-closure chain therefore
# hands Step5 an EXPLICIT, hash-pinned baseline list via its opt-in context key.
BASELINE_BINDING_SOURCE = "predictor_export_metadata"
STEP5_EXPLICIT_BASELINE_PATHS_KEY = "explicit_baseline_lst_paths"

LOCAL_DOWNSTREAM_LIMITATIONS: tuple[str, ...] = (
    "Every downstream number is produced by the production Step5/Step5C/"
    "Step7A-E/Step8A helpers. This stage introduces no TVDI, downscaling, "
    "fusion, aggregation, alignment, imputation or feature formula of its own "
    "and adds no feature to the Step8A contract.",
    "The variant predictors are re-laid-out under the production input file "
    "names before the chain runs. That step copies bytes and concatenates "
    "already-exported bands; it never resamples, reprojects or recomputes a "
    "value.",
    "Landsat and MODIS observation support legitimately differs between "
    "variants, so the number of Step8A rows that pass the production validity "
    "policy may differ too. Row counts are reported, never forced to match.",
    "No common cohort and no shared fold assignment is built here; both belong "
    "to the model stage.",
    "The pre-label censoring raster is not a predictor and is never read as "
    "one; it is recorded only as a censoring/provenance input.",
)


# --- Paths --------------------------------------------------------------------
def local_downstream_root(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> Path:
    return variant_root(experiment_id, variant_id, output_root) / LOCAL_DOWNSTREAM_ROOT_DIR


def local_downstream_input_root(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> Path:
    return local_downstream_root(experiment_id, variant_id, output_root) / LOCAL_DOWNSTREAM_INPUT_DIR


def local_downstream_stage_dir(
    experiment_id: str, variant_id: str, stage: str, output_root: Optional[Path] = None,
) -> Path:
    if stage not in PRODUCTION_STAGE_SEQUENCE:
        raise WindowClosureError(
            f"'{stage}' is not one of the production downstream stages "
            f"{list(PRODUCTION_STAGE_SEQUENCE)}."
        )
    return local_downstream_root(experiment_id, variant_id, output_root) / stage


def local_downstream_metadata_path(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> Path:
    return (
        variant_root(experiment_id, variant_id, output_root)
        / LOCAL_DOWNSTREAM_METADATA_NAME
    )


def variant_step8a_dataset_path(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> Path:
    return (
        local_downstream_stage_dir(experiment_id, variant_id, "step8a", output_root)
        / STEP8A_DATASET_NAME
    )


def variant_step8a_stats_path(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> Path:
    return (
        local_downstream_stage_dir(experiment_id, variant_id, "step8a", output_root)
        / STEP8A_STATS_NAME
    )


def canonical_step8a_stats_path(
    experiment_id: str, experiments_root: Optional[Path] = None,
) -> Path:
    return (
        canonical_experiment_root(experiment_id, experiments_root)
        / "step8a" / STEP8A_STATS_NAME
    )


# --- Which production stage consumes which logical role ----------------------
def production_stage_input_roles(
    baseline_years: Sequence[int], censor_roles: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Logical inputs of each production downstream stage.

    Read off the production helpers themselves (the `ctx` keys and file names
    each `run_stepX` resolves), not guessed: Step5 consumes the current Landsat
    LST plus every baseline-year LST; Step5C additionally consumes the current
    and baseline NDVI; Step7B additionally consumes the current-window MODIS
    mean/std, the DEM and the aligned landcover; Step8A consumes the Step5/
    Step5C/Step7D/Step7E products, the current NDVI, the DEM, the landcover and
    the frozen raw BurnDate label.
    """
    years = [int(year) for year in sorted(baseline_years)]
    baseline_lst = [f"baseline_lst_{year}" for year in years]
    baseline_ndvi = [f"baseline_ndvi_{year}" for year in years]
    return {
        "step5": [INPUT_ROLE_CURRENT_LST, *baseline_lst],
        "step5c": ["step5", INPUT_ROLE_CURRENT_NDVI, *baseline_ndvi],
        "step7a": ["step5"],
        "step7b": [
            "step5", "step5c", INPUT_ROLE_CURRENT_NDVI,
            "modis_lst_mean", "modis_lst_std",
            "dem_elevation", "dem_slope", "landcover_aligned",
        ],
        "step7c": ["step7b"],
        "step7d": ["step5", "step7b", "step7c"],
        "step7e": ["step5", "step7d"],
        # Step8A additionally consumes the bound pre-label exclusion gate
        # documents, but ONLY for an experiment whose registry record enables
        # the policy -- so the declaration is driven by the caller's resolved
        # binding, never by an experiment name.
        "step8a": [
            "step5", "step5c", "step7d", "step7e", INPUT_ROLE_CURRENT_NDVI,
            "dem_elevation", "dem_slope", "landcover_aligned", LABEL_ROLE_RAW,
            *sorted(censor_roles),
        ],
    }


# --- Predictor binding --------------------------------------------------------
PREDICTOR_METADATA_REQUIRED_FLAGS: dict[str, bool] = {
    "canonical_export_attempted": False,
    "prelabel_used_as_predictor": False,
    "raster_contract_passed": True,
    "all_paths_inside_variant_namespace": True,
    "frozen_hashes_unchanged": True,
    "canonical_outputs_modified": False,
}


def read_predictor_metadata(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> dict:
    path = predictor_metadata_path(experiment_id, variant_id, output_root)
    if not path.is_file():
        raise WindowClosureError(
            f"Predictor export metadata for variant '{variant_id}' is missing "
            f"at {path}. The local-downstream stage binds to a completed "
            "predictor export; run --from-stage predictor-export --to-stage "
            "predictor-export first. Nothing was created."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise WindowClosureError(
            f"Predictor export metadata for variant '{variant_id}' at {path} "
            f"is unreadable: {exc}. Nothing was created."
        ) from exc
    if not isinstance(payload, dict):
        raise WindowClosureError(
            f"Predictor export metadata for variant '{variant_id}' is not a "
            "JSON object."
        )
    return payload


def assert_predictor_metadata_contract(
    experiment_id: str,
    analysis_id: str,
    variant: dict,
    metadata: dict,
    baseline_years: Sequence[int],
    output_root: Optional[Path] = None,
    prelabel_sha256: Optional[str] = None,
) -> dict:
    """Full predictor-export binding for ONE variant, before anything is made.

    Runs entirely on documents and file hashes: no directory is created, no
    raster or parquet is written and no production helper is imported until
    every check below has passed.
    """
    variant_id = variant["variant_id"]

    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise WindowClosureError(
                f"Local-downstream predictor binding failed for variant "
                f"'{variant_id}': {message} No downstream directory was "
                "created, no raster or parquet was written and no production "
                "stage was called."
            )

    _require(
        metadata.get("schema_version") == PREDICTOR_METADATA_SCHEMA,
        f"predictor metadata schema is {metadata.get('schema_version')!r}, "
        f"expected {PREDICTOR_METADATA_SCHEMA!r}.",
    )
    _require(
        metadata.get("analysis_id") == analysis_id,
        f"predictor metadata analysis_id is {metadata.get('analysis_id')!r}, "
        f"but this run computed {analysis_id!r}.",
    )
    _require(
        metadata.get("experiment_id") == experiment_id,
        f"predictor metadata experiment_id is {metadata.get('experiment_id')!r}.",
    )
    _require(
        metadata.get("variant_id") == variant_id,
        f"predictor metadata variant_id is {metadata.get('variant_id')!r}.",
    )
    _require(
        metadata.get("status") == STATUS_PASS,
        f"predictor metadata status is {metadata.get('status')!r}, not "
        f"{STATUS_PASS!r}.",
    )
    _require(
        int(metadata.get("shift_days", -1)) == int(variant["shift_days"]),
        f"predictor metadata shift_days is {metadata.get('shift_days')!r}, "
        f"expected {variant['shift_days']!r}.",
    )
    for key in ("predictor_start_date", "predictor_end_date",
                "label_start_date", "label_end_date"):
        _require(
            metadata.get(key) == variant[key],
            f"predictor metadata {key}={metadata.get(key)!r} differs from the "
            f"preregistered {variant[key]!r}.",
        )
    recorded_years = sorted(int(year) for year in (metadata.get("baseline_years") or []))
    _require(
        recorded_years == sorted(int(year) for year in baseline_years),
        f"predictor metadata baseline years {recorded_years} differ from the "
        f"preregistered {sorted(int(y) for y in baseline_years)}.",
    )

    expected_roles = expected_logical_role_count(baseline_years)
    expected_rasters = expected_raster_count(baseline_years)
    for key, expected in (
        ("expected_logical_role_count", expected_roles),
        ("produced_logical_role_count", expected_roles),
        ("expected_raster_count", expected_rasters),
        ("produced_raster_count", expected_rasters),
    ):
        _require(
            int(metadata.get(key, -1)) == int(expected),
            f"predictor metadata {key}={metadata.get(key)!r}, expected {expected}.",
        )

    for flag, expected in PREDICTOR_METADATA_REQUIRED_FLAGS.items():
        _require(
            metadata.get(flag) is expected,
            f"predictor metadata {flag}={metadata.get(flag)!r}, expected {expected!r}.",
        )

    artifacts = metadata.get("artifact_inventory") or []
    recorded_hashes = metadata.get("artifact_sha256") or {}
    _require(
        len(artifacts) == expected_rasters,
        f"predictor metadata lists {len(artifacts)} artefacts, expected "
        f"{expected_rasters}.",
    )

    allowed_root = variant_root(experiment_id, variant_id, output_root).resolve()
    canonical_variant = variant_root(
        experiment_id, CANONICAL_VARIANT_ID, output_root,
    ).resolve()
    prelabel = prelabel_raster_path(experiment_id, output_root).resolve()

    ids: list[str] = []
    paths: list[str] = []
    roles: dict[str, set] = {}
    resolved: list[dict] = []
    for record in artifacts:
        if not isinstance(record, dict):
            _require(False, "the artefact inventory carries a non-object entry.")
        artifact_id = str(record.get("artifact_id"))
        role = str(record.get("role"))
        product = str(record.get("product"))
        path = Path(str(record.get("path") or ""))
        _require(
            not is_forbidden_landsat_product(product),
            f"artefact '{artifact_id}' carries the reducer-counterfactual "
            f"product '{product}'; the window-closure chain is scene-weighted only.",
        )
        _require(path.is_file(), f"artefact '{artifact_id}' is missing at {path}.")
        digest = sha256_file(path)
        _require(
            digest == record.get("sha256"),
            f"artefact '{artifact_id}' hashes {digest} but the metadata "
            f"recorded {record.get('sha256')!r}.",
        )
        _require(
            digest == recorded_hashes.get(artifact_id),
            f"artefact '{artifact_id}' hash disagrees with the metadata "
            "artifact_sha256 map.",
        )
        target = path.resolve()
        # The pre-label censoring raster is checked FIRST and by BOTH identity
        # and content, so it can never enter the predictor inventory -- neither
        # by path nor as a copy placed inside the variant namespace.
        _require(
            target != prelabel,
            f"artefact '{artifact_id}' IS the pre-label censoring raster; the "
            "pre-label raster is never a predictor.",
        )
        if prelabel_sha256 is not None:
            _require(
                digest != prelabel_sha256,
                f"artefact '{artifact_id}' has the pre-label censoring "
                "raster's hash; the pre-label raster is never a predictor.",
            )
        _require(
            allowed_root in target.parents,
            f"artefact '{artifact_id}' lies outside the variant namespace "
            f"{allowed_root}: {target}.",
        )
        _require(
            not (target == canonical_variant or canonical_variant in target.parents),
            f"artefact '{artifact_id}' lies inside the canonical variant "
            f"namespace: {target}.",
        )
        ids.append(artifact_id)
        paths.append(str(target))
        roles.setdefault(role, set()).add(product)
        resolved.append({
            "artifact_id": artifact_id,
            "role": role,
            "family": record.get("family"),
            "scope": record.get("scope"),
            "baseline_year": record.get("baseline_year"),
            "product": product,
            # Plain strings only: this record is carried in the run result, so
            # it must stay JSON-serialisable.
            "path": str(path),
            "sha256": digest,
            "start_date": record.get("start_date"),
            "end_date": record.get("end_date"),
        })

    duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
    _require(not duplicate_ids, f"duplicate artefact id(s) {duplicate_ids}.")
    duplicate_paths = sorted({value for value in paths if paths.count(value) > 1})
    _require(not duplicate_paths, f"duplicate artefact path(s) {duplicate_paths}.")
    _require(
        len(roles) == expected_roles,
        f"the inventory carries {len(roles)} logical roles, expected {expected_roles}.",
    )

    required_roles = (
        {INPUT_ROLE_CURRENT_LST, INPUT_ROLE_CURRENT_NDVI}
        | {f"baseline_lst_{int(year)}" for year in baseline_years}
        | {f"baseline_ndvi_{int(year)}" for year in baseline_years}
        | set(MODIS_ROLE_ORDER)
    )
    missing_roles = sorted(required_roles - set(roles))
    extra_roles = sorted(set(roles) - required_roles)
    _require(not missing_roles, f"missing logical role(s) {missing_roles}.")
    _require(not extra_roles, f"unexpected logical role(s) {extra_roles}.")
    for role, products in sorted(roles.items()):
        if role in MODIS_ROLE_ORDER:
            continue
        _require(
            sorted(products) == sorted(LANDSAT_PRODUCTS_PER_ROLE),
            f"Landsat role '{role}' carries products {sorted(products)}, "
            f"expected {sorted(LANDSAT_PRODUCTS_PER_ROLE)}.",
        )

    metadata_path = predictor_metadata_path(experiment_id, variant_id, output_root)
    return {
        "variant_id": variant_id,
        "predictor_metadata_path": str(metadata_path),
        "predictor_metadata_sha256": sha256_file(metadata_path),
        "predictor_artifact_count": len(resolved),
        "predictor_logical_role_count": len(roles),
        "predictor_artifact_sha256": {
            record["artifact_id"]: record["sha256"] for record in resolved
        },
        "artifacts": resolved,
        "baseline_years": [int(year) for year in sorted(baseline_years)],
    }


def assert_local_downstream_binding(
    experiment_id: str,
    analysis_id: str,
    shifts: Sequence[int],
    canonical: dict,
    variants: Sequence[dict],
    censor: dict,
    inventory: dict,
    planned_paths: dict[str, str],
    output_root: Optional[Path] = None,
) -> dict:
    """Bind to the completed plan, prelabel AND predictor-export stages.

    Read-only. Every gate runs before any downstream directory is created, any
    intermediate raster/parquet is written or any production helper is imported.
    """
    binding = assert_predictor_binding(
        experiment_id, analysis_id, shifts, canonical, variants, censor,
        inventory, planned_paths, output_root,
    )
    baseline_years = list(canonical["baseline_years"])
    prelabel_sha256 = binding.get("prelabel_raster_sha256")

    per_variant: dict[str, dict] = {}
    for variant in nonzero_variants(variants):
        metadata = read_predictor_metadata(
            experiment_id, variant["variant_id"], output_root,
        )
        per_variant[variant["variant_id"]] = assert_predictor_metadata_contract(
            experiment_id, analysis_id, variant, metadata, baseline_years,
            output_root, prelabel_sha256,
        )
    return {
        **binding,
        "bound_to_predictor_export": True,
        "predictor_binding": per_variant,
        "predictor_binding_ready": True,
    }


def inspect_predictor_binding_readiness(
    experiment_id: str,
    analysis_id: str,
    canonical: dict,
    variants: Sequence[dict],
    output_root: Optional[Path] = None,
) -> dict:
    """Non-raising readiness report used by the DRY RUN.

    A dry run must describe what an actual run WOULD do without failing on an
    incomplete namespace and without creating anything, so the same contract is
    evaluated here into booleans instead of exceptions.
    """
    baseline_years = list(canonical["baseline_years"])
    per_variant: dict[str, dict] = {}
    ready = True
    all_present = True
    all_hashes = True
    for variant in nonzero_variants(variants):
        variant_id = variant["variant_id"]
        path = predictor_metadata_path(experiment_id, variant_id, output_root)
        entry: dict[str, Any] = {
            "variant_id": variant_id,
            "predictor_metadata_path": str(path),
            "predictor_metadata_present": path.is_file(),
            "predictor_metadata_sha256": sha256_file(path) if path.is_file() else None,
            "predictor_artifact_count": 0,
            "expected_predictor_artifact_count": expected_raster_count(baseline_years),
            "all_predictor_artifacts_present": False,
            "all_predictor_hashes_match": False,
            "binding_ready": False,
            "reason": "predictor export metadata is missing",
        }
        if path.is_file():
            try:
                metadata = read_predictor_metadata(experiment_id, variant_id, output_root)
                assert_predictor_metadata_contract(
                    experiment_id, analysis_id, variant, metadata, baseline_years,
                    output_root,
                )
            except WindowClosureError as exc:
                entry["reason"] = str(exc)
            else:
                artifacts = metadata.get("artifact_inventory") or []
                entry.update({
                    "predictor_artifact_count": len(artifacts),
                    "all_predictor_artifacts_present": True,
                    "all_predictor_hashes_match": True,
                    "binding_ready": True,
                    "reason": "complete and verified",
                })
        ready = ready and entry["binding_ready"]
        all_present = all_present and entry["all_predictor_artifacts_present"]
        all_hashes = all_hashes and entry["all_predictor_hashes_match"]
        per_variant[variant_id] = entry
    return {
        "predictor_binding_ready": ready,
        "all_predictor_artifacts_present": all_present,
        "all_predictor_hashes_match": all_hashes,
        "per_variant": per_variant,
    }


# --- Production input layout (pure) ------------------------------------------
def production_input_bindings(
    experiment_id: str,
    variant: dict,
    base_context: dict,
    artifacts: Sequence[dict],
    inventory: dict,
    output_root: Optional[Path] = None,
    censor_binding: Optional[dict] = None,
) -> list[dict]:
    """Map every predictor / frozen static input onto its production file name.

    Resolution is driven ONLY by the predictor metadata inventory (logical role
    + product) and by the frozen-input inventory. No file name is guessed from
    disk and no canonical path is read for a predictor.

    The production names come from the production producer of these very
    inputs, `scripts/run_predictors_only.py:_export_predictors_direct`:

      * current LST  -> ``landsat_current_period_<window_days>days.tif`` (2 bands)
      * current NDVI -> ``current_ndvi_median.tif``                      (2 bands)
      * baseline LST -> ``<landsat_file_prefix>_baseline_<window_end>.tif``
      * baseline NDVI-> ``ndvi_baseline_<window_end>.tif``
      * MODIS        -> the production names in ``MODIS_ROLE_FILENAMES``

    ``window_end`` is the artefact's own recorded end date, i.e. the exact
    production ``_baseline_year_window`` end the predictor was exported over --
    never a re-derived or hard-coded calendar value.
    """
    variant_id = variant["variant_id"]
    root = local_downstream_input_root(experiment_id, variant_id, output_root)
    window_days = int(base_context["current_period_days"])
    file_prefix = str(base_context["landsat_file_prefix"])

    by_role: dict[str, dict[str, dict]] = {}
    for record in artifacts:
        by_role.setdefault(record["role"], {})[record["product"]] = record

    def _sources(role: str, products: Sequence[str]) -> list[dict]:
        available = by_role.get(role) or {}
        missing = [product for product in products if product not in available]
        if missing:
            raise WindowClosureError(
                f"Variant '{variant_id}' predictor inventory has no "
                f"{missing} product for logical role '{role}'; the production "
                "input cannot be bound."
            )
        return [available[product] for product in products]

    bindings: list[dict] = []

    # --- Current Landsat: two production bands, re-assembled ----------------
    for role, subdir, filename in (
        (INPUT_ROLE_CURRENT_LST, "current_period",
         f"landsat_current_period_{window_days}days.tif"),
        (INPUT_ROLE_CURRENT_NDVI, "ndvi_current_period", "current_ndvi_median.tif"),
    ):
        sources = _sources(role, CURRENT_ROLE_BAND_ORDER)
        bindings.append({
            "input_role": role,
            "mode": "stack_bands",
            "sources": [record["path"] for record in sources],
            "source_artifact_ids": [record["artifact_id"] for record in sources],
            "source_sha256": [record["sha256"] for record in sources],
            "band_order": list(CURRENT_ROLE_BAND_ORDER),
            "target": root / subdir / filename,
            "context_dir_key": PRODUCTION_INPUT_CONTEXT_DIRS[subdir],
            "variant_derived": True,
            "consumed_by_production": True,
            "producer": "scripts/run_predictors_only.py:_export_predictors_direct",
        })

    # --- Baseline Landsat: one production band per year ---------------------
    for role in sorted(by_role):
        if not role.startswith(("baseline_lst_", "baseline_ndvi_")):
            continue
        median = _sources(role, (PRODUCT_SCENE_WEIGHTED_MEDIAN,))[0]
        window_end = str(median.get("end_date") or "")
        if not window_end:
            raise WindowClosureError(
                f"Variant '{variant_id}' baseline artefact "
                f"'{median['artifact_id']}' records no end date, so the "
                "production baseline file name cannot be resolved."
            )
        if role.startswith("baseline_lst_"):
            subdir, filename = "landsat_timeseries", f"{file_prefix}_baseline_{window_end}.tif"
        else:
            subdir, filename = "ndvi_timeseries", f"ndvi_baseline_{window_end}.tif"
        bindings.append({
            "input_role": role,
            "mode": "copy",
            "sources": [median["path"]],
            "source_artifact_ids": [median["artifact_id"]],
            "source_sha256": [median["sha256"]],
            "band_order": [PRODUCT_SCENE_WEIGHTED_MEDIAN],
            "target": root / subdir / filename,
            "context_dir_key": PRODUCTION_INPUT_CONTEXT_DIRS[subdir],
            "variant_derived": True,
            "consumed_by_production": True,
            "producer": "scripts/run_predictors_only.py:_export_predictors_direct",
        })
        # Production Step5/Step5C read only the median band of a baseline year,
        # so the exported support raster is RETAINED in the predictor namespace
        # and recorded, not silently dropped and not invented into an input.
        count = (by_role.get(role) or {}).get(PRODUCT_SCENE_VALID_COUNT)
        if count is not None:
            bindings.append({
                "input_role": f"{role}__{PRODUCT_SCENE_VALID_COUNT}",
                "mode": "retained_not_consumed",
                "sources": [count["path"]],
                "source_artifact_ids": [count["artifact_id"]],
                "source_sha256": [count["sha256"]],
                "band_order": [PRODUCT_SCENE_VALID_COUNT],
                "target": None,
                "context_dir_key": None,
                "variant_derived": True,
                "consumed_by_production": False,
                "producer": "scripts/run_predictors_only.py:_export_predictors_direct",
                "note": (
                    "Production Step5/Step5C consume only the baseline median "
                    "band; this support raster is preserved read-only in the "
                    "predictor namespace and never re-derived."
                ),
            })

    # --- MODIS current-window products --------------------------------------
    for role in MODIS_ROLE_ORDER:
        record = _sources(role, (role,))[0]
        bindings.append({
            "input_role": role,
            "mode": "copy",
            "sources": [record["path"]],
            "source_artifact_ids": [record["artifact_id"]],
            "source_sha256": [record["sha256"]],
            "band_order": [role],
            "target": root / "modis" / MODIS_ROLE_FILENAMES[role],
            "context_dir_key": PRODUCTION_INPUT_CONTEXT_DIRS["modis"],
            "variant_derived": True,
            # Step7B resolves the MODIS mean/std from `modis_input_dir`; the
            # valid-observation count is exported support that production does
            # not read, so it is carried but flagged honestly.
            "consumed_by_production": role != MODIS_COUNT_ROLE,
            "producer": MODIS_PRODUCER,
        })

    # --- Frozen STATIC inputs, copied read-only ------------------------------
    static_targets: tuple[tuple[str, Path], ...] = (
        ("dem_elevation", root / "dem" / "elevation.tif"),
        ("dem_slope", root / "dem" / "slope.tif"),
        ("landcover_aligned", root / "gate_inputs" / (
            "landcover_esa_worldcover_v200_aligned_to_reference.tif"
        )),
        (LABEL_ROLE_RAW, root / "labels" / CANONICAL_LABEL_FILENAMES[LABEL_ROLE_RAW]),
        (LABEL_ROLE_BINARY, root / "labels" / CANONICAL_LABEL_FILENAMES[LABEL_ROLE_BINARY]),
    )
    for role, target in static_targets:
        entry = inventory.get(role) or {}
        source = Path(str(entry.get("path") or ""))
        if not entry.get("exists") or entry.get("sha256") is None:
            raise WindowClosureError(
                f"Frozen static input '{role}' is missing or unhashed at "
                f"{source}; the local-downstream chain cannot be bound."
            )
        bindings.append({
            "input_role": role,
            "mode": "copy",
            "sources": [source],
            "source_artifact_ids": [role],
            "source_sha256": [entry["sha256"]],
            "band_order": [role],
            "target": target,
            "context_dir_key": None,
            "variant_derived": False,
            "consumed_by_production": role != LABEL_ROLE_BINARY,
            "producer": "frozen canonical production output (read-only)",
        })

    # --- Frozen pre-label EXCLUSION gate documents ---------------------------
    # Only when the registry enables the policy. Production Step8A resolves
    # them from ctx["gate_labels_dir"], which for a variant is this materialised
    # `labels/` directory, so they must be laid out under their production file
    # names exactly like the label rasters above. They are documents, not
    # rasters, so they use the verbatim-document mode.
    censor_binding = assert_prelabel_exclusion_binding(
        censor_binding if censor_binding is not None
        else prelabel_exclusion_binding(experiment_id, base_context),
        f"local-downstream input binding for variant '{variant_id}'",
    )
    if censor_binding["exclude_pre_label_burns"]:
        for role in sorted(censor_binding["documents"]):
            record = censor_binding["documents"][role]
            if not record["exists"]:
                # Only the OPTIONAL gate provenance manifest can be absent
                # here; the required ones already failed the assertion above.
                continue
            bindings.append({
                "input_role": role,
                "mode": "copy_document",
                "sources": [Path(record["path"])],
                "source_artifact_ids": [role],
                "source_sha256": [record["sha256"]],
                "band_order": [role],
                "target": root / "labels" / record["filename"],
                "context_dir_key": PRODUCTION_INPUT_CONTEXT_DIRS["labels"],
                "variant_derived": False,
                "consumed_by_production": (
                    role in PRELABEL_EXCLUSION_REQUIRED_ROLES
                ),
                "producer": (
                    "frozen Step6B burned-landcover gate output (read-only)"
                ),
            })
    bindings.sort(key=lambda item: item["input_role"])
    return bindings


def resolve_baseline_lst_binding(
    variant_id: str, bindings: Sequence[dict], baseline_years: Sequence[int],
) -> dict:
    """The EXPLICIT baseline LST list Step5 will be pinned to.

    Resolved from the predictor-export metadata inventory only, in increasing
    baseline-year order. Refuses a missing year, an extra year, a duplicate and
    -- explicitly -- any `scene_valid_count` support raster: production Step5
    composites the baseline MEDIAN band, and a count raster in that list would
    silently corrupt the baseline stack.
    """
    years = sorted(int(year) for year in baseline_years)
    by_year: dict[int, dict] = {}
    for binding in bindings:
        role = str(binding["input_role"])
        if not role.startswith("baseline_lst_"):
            continue
        if binding["mode"] != "copy":
            # `retained_not_consumed` entries are support rasters; they are
            # deliberately never part of the baseline stack.
            continue
        suffix = role[len("baseline_lst_"):]
        if not suffix.isdigit():
            raise WindowClosureError(
                f"Variant '{variant_id}' baseline binding role '{role}' is not "
                "a plain baseline year; the baseline stack must be pinned by "
                "year only."
            )
        year = int(suffix)
        if year in by_year:
            raise WindowClosureError(
                f"Variant '{variant_id}' binds baseline year {year} twice."
            )
        products = list(binding.get("band_order") or [])
        if products != [PRODUCT_SCENE_WEIGHTED_MEDIAN]:
            raise WindowClosureError(
                f"Variant '{variant_id}' baseline year {year} is bound to "
                f"product(s) {products}; production Step5 composites the "
                f"'{PRODUCT_SCENE_WEIGHTED_MEDIAN}' band only."
            )
        by_year[year] = binding

    missing = sorted(set(years) - set(by_year))
    extra = sorted(set(by_year) - set(years))
    if missing:
        raise WindowClosureError(
            f"Variant '{variant_id}' has no baseline LST binding for "
            f"preregistered year(s) {missing}; the local downstream refuses to "
            "fall back to a directory scan."
        )
    if extra:
        raise WindowClosureError(
            f"Variant '{variant_id}' binds non-preregistered baseline year(s) "
            f"{extra}."
        )

    ordered = [by_year[year] for year in years]
    paths = [Path(binding["target"]) for binding in ordered]
    if len({str(path) for path in paths}) != len(paths):
        raise WindowClosureError(
            f"Variant '{variant_id}' binds duplicate baseline LST path(s)."
        )
    return {
        "baseline_years": years,
        "paths": paths,
        "baseline_binding_source": BASELINE_BINDING_SOURCE,
        "baseline_directory_scan_used": False,
        "records": [
            {
                "baseline_year": year,
                "input_role": binding["input_role"],
                "source_artifact_id": binding["source_artifact_ids"][0],
                "source_sha256": binding["source_sha256"][0],
                "source_path": str(binding["sources"][0]),
                "bound_path": str(binding["target"]),
                "product": PRODUCT_SCENE_WEIGHTED_MEDIAN,
            }
            for year, binding in zip(years, ordered)
        ],
    }


def _raster_profile(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as dataset:
        return {
            "crs": dataset.crs,
            "transform": dataset.transform,
            "width": int(dataset.width),
            "height": int(dataset.height),
            "count": int(dataset.count),
            "dtype": str(dataset.dtypes[0]),
            "nodata": dataset.nodata,
        }


def _copy_verbatim(source: Path, target: Path) -> None:
    """Byte-for-byte copy. The copy must hash exactly like its source."""
    import shutil

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    if sha256_file(target) != sha256_file(source):
        raise WindowClosureError(
            f"Materialised input {target} does not hash like its source {source}."
        )


def _stack_single_band_rasters(sources: Sequence[Path], target: Path) -> dict:
    """Concatenate single-band rasters into the production multi-band raster.

    This is a CONTAINER operation and nothing else: every band is written back
    with the values it was read with, on the grid it was exported on. The
    sources must already agree on CRS, transform, shape, dtype and nodata --
    they are two bands of ONE production image that the predictor stage split
    apart -- and a disagreement fails the stage instead of being resampled or
    cast away.
    """
    import numpy as np
    import rasterio

    if not sources:
        raise WindowClosureError("Cannot stack an empty list of rasters.")
    profiles = [_raster_profile(Path(path)) for path in sources]
    reference = profiles[0]
    for path, profile in zip(sources, profiles):
        if profile["count"] != 1:
            raise WindowClosureError(
                f"{path} carries {profile['count']} bands; the production "
                "current-window inputs are assembled from single-band exports."
            )
        for key in ("crs", "transform", "width", "height", "dtype"):
            if profile[key] != reference[key]:
                raise WindowClosureError(
                    f"{path} disagrees with {sources[0]} on '{key}' "
                    f"({profile[key]!r} != {reference[key]!r}); the two bands "
                    "of a production current-window export must share a grid."
                )
        if (profile["nodata"] is None) != (reference["nodata"] is None) or (
            profile["nodata"] is not None
            and not np.array_equal(
                np.asarray([profile["nodata"]]), np.asarray([reference["nodata"]])
            )
        ):
            raise WindowClosureError(
                f"{path} disagrees with {sources[0]} on nodata "
                f"({profile['nodata']!r} != {reference['nodata']!r})."
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        with rasterio.open(
            temporary, "w", driver="GTiff",
            width=reference["width"], height=reference["height"],
            count=len(sources), dtype=reference["dtype"],
            crs=reference["crs"], transform=reference["transform"],
            nodata=reference["nodata"],
        ) as destination:
            for index, path in enumerate(sources, start=1):
                with rasterio.open(path) as band_source:
                    destination.write(band_source.read(1), index)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "band_count": len(sources),
        "dtype": reference["dtype"],
        "nodata": None if reference["nodata"] is None else float(reference["nodata"]),
        "width": reference["width"],
        "height": reference["height"],
        "crs": str(reference["crs"]),
    }


def materialize_local_downstream_inputs(bindings: Sequence[dict]) -> list[dict]:
    """Write the production-named inputs. Values are never touched."""
    records: list[dict] = []
    for binding in bindings:
        if binding["mode"] == "retained_not_consumed":
            records.append({
                **{k: v for k, v in binding.items() if k not in ("sources", "target")},
                "sources": [str(path) for path in binding["sources"]],
                "target": None,
                "materialized": False,
                "sha256": None,
            })
            continue
        target = Path(binding["target"])
        sources = [Path(path) for path in binding["sources"]]
        if binding["mode"] == "copy":
            _copy_verbatim(sources[0], target)
            profile = _raster_profile(target)
            detail = {
                "band_count": profile["count"], "dtype": profile["dtype"],
                "width": profile["width"], "height": profile["height"],
                "crs": str(profile["crs"]),
                "nodata": None if profile["nodata"] is None else float(profile["nodata"]),
            }
        elif binding["mode"] == "copy_document":
            # A parquet/JSON gate document: the same byte-verbatim copy, but it
            # carries no raster profile, so none is invented for it.
            _copy_verbatim(sources[0], target)
            detail = {
                "band_count": None, "dtype": None, "width": None,
                "height": None, "crs": None, "nodata": None,
                "document": True,
            }
        elif binding["mode"] == "stack_bands":
            detail = _stack_single_band_rasters(sources, target)
        else:
            raise WindowClosureError(
                f"Unknown input materialisation mode {binding['mode']!r}."
            )
        records.append({
            **{k: v for k, v in binding.items() if k not in ("sources", "target")},
            "sources": [str(path) for path in sources],
            "target": str(target),
            "materialized": True,
            "sha256": sha256_file(target),
            "bytes": int(target.stat().st_size),
            **detail,
        })
    records.sort(key=lambda record: record["input_role"])
    return records


def assert_materialized_values_unchanged(records: Sequence[dict]) -> None:
    """A copied input must hash like its source; a stacked one must not lose a value.

    Copies are compared by hash. Stacks cannot be compared by hash (the
    container changed), so every band is compared elementwise against the
    single-band export it came from, NaN-for-NaN.
    """
    import numpy as np
    import rasterio

    for record in records:
        if not record.get("materialized"):
            continue
        target = Path(record["target"])
        sources = [Path(path) for path in record["sources"]]
        if record["mode"] in ("copy", "copy_document"):
            if sha256_file(target) != sha256_file(sources[0]):
                raise WindowClosureError(
                    f"Materialised input '{record['input_role']}' does not hash "
                    f"like its source: {target}."
                )
            continue
        with rasterio.open(target) as destination:
            for index, source in enumerate(sources, start=1):
                with rasterio.open(source) as band_source:
                    left = destination.read(index)
                    right = band_source.read(1)
                if not np.array_equal(left, right, equal_nan=True):
                    raise WindowClosureError(
                        f"Band {index} of materialised input "
                        f"'{record['input_role']}' differs from its source "
                        f"{source}; the band assembly must copy values verbatim."
                    )


# --- Variant context ----------------------------------------------------------
# Every mutable path the production chain may touch. Checked for containment
# BEFORE any production helper is imported, because the production steps create
# their output directories eagerly.
LOCAL_DOWNSTREAM_CONTEXT_PATH_KEYS: tuple[str, ...] = (
    "output_root", "data_root",
    "baseline_input_dir", "current_period_dir", "qa_dir",
    "ndvi_baseline_dir", "ndvi_current_dir", "modis_input_dir", "modis_dir",
    "dem_input_dir", "landcover_aligned_path", "gate_labels_dir",
    "step5_output_dir", "step5b_output_dir", "step5c_output_dir", "output_dir",
    "step7a_output_dir", "step7b_output_dir", "step7c_output_dir",
    "step7d_output_dir", "step7e_output_dir",
    "step8a_output_dir", "step8b_output_dir", "step8c_output_dir",
    "step8d_output_dir", "step8e_output_dir",
)


def build_local_downstream_variant_context(
    experiment_id: str,
    variant: dict,
    base_context: dict,
    analysis_id: Optional[str] = None,
    baseline_binding: Optional[dict] = None,
    output_root: Optional[Path] = None,
) -> dict:
    """A production ExperimentContext re-rooted into the variant's downstream tree.

    Built on `build_window_variant_context`, which already deep-copies the
    registry context (the global registry is never mutated), carries the variant
    predictor window and freezes the label window. Every remaining path is then
    re-pointed at `variants/<variant_id>/downstream/`, so:

      * every INPUT the production chain reads is the materialised, production-
        named copy inside the variant namespace -- the canonical experiment
        namespace and the predictor-export `data/` tree are never written to,
        and the production steps (which create their input directories eagerly)
        cannot reach them;
      * every OUTPUT lands in the variant's own stage directory.

    No window, baseline year, seed, threshold or feature definition is altered.
    """
    ctx = build_window_variant_context(
        experiment_id, variant["shift_days"], base_context, output_root,
    )
    if ctx["window_closure_variant_id"] != variant["variant_id"]:
        raise WindowClosureError(
            f"Variant context is for '{ctx['window_closure_variant_id']}', "
            f"expected '{variant['variant_id']}'."
        )
    for key in ("predictor_start_date", "predictor_end_date"):
        if ctx[key] != variant[key]:
            raise WindowClosureError(
                f"Variant context {key}={ctx[key]!r} differs from the "
                f"preregistered {variant[key]!r}."
            )

    variant_id = variant["variant_id"]
    downstream = local_downstream_root(experiment_id, variant_id, output_root)
    inputs = local_downstream_input_root(experiment_id, variant_id, output_root)

    ctx["is_kozan"] = False
    ctx["window_closure_stage"] = LOCAL_DOWNSTREAM_STAGE
    ctx["output_root"] = downstream
    ctx["data_root"] = inputs
    ctx["baseline_input_dir"] = inputs / "landsat_timeseries"
    ctx["current_period_dir"] = inputs / "current_period"
    ctx["qa_dir"] = inputs / "landsat_qa"
    ctx["ndvi_baseline_dir"] = inputs / "ndvi_timeseries"
    ctx["ndvi_current_dir"] = inputs / "ndvi_current_period"
    ctx["modis_input_dir"] = inputs / "modis"
    ctx["modis_dir"] = inputs / "modis"
    # DEM, landcover and the frozen label are window-INDEPENDENT, so they are
    # not re-derived; they are materialised read-only copies of the canonical
    # artefacts, and the copy is hash-verified against its source.
    ctx["dem_input_dir"] = inputs / "dem"
    ctx["dem_is_shared_read_only"] = False
    ctx["landcover_aligned_path"] = (
        inputs / "gate_inputs"
        / "landcover_esa_worldcover_v200_aligned_to_reference.tif"
    )
    ctx["gate_labels_dir"] = inputs / "labels"
    # Manavgat-style experiments have no Step4 Drive-export metadata; Step5 then
    # falls back to scanning the baseline directory, exactly as production does.
    ctx["step4_metadata_path"] = None
    for stage in PRODUCTION_STAGE_SEQUENCE:
        ctx[f"{stage}_output_dir"] = downstream / stage
    ctx["step5b_output_dir"] = downstream / "step5b"
    for stage in ("step8b", "step8c", "step8d", "step8e"):
        ctx[f"{stage}_output_dir"] = downstream / stage
    ctx["output_dir"] = downstream / "step5"
    ctx[MODIS_NAMESPACE_ALLOWED_ROOTS_KEY] = [
        variant_root(experiment_id, variant_id, output_root)
    ]

    # --- OPT-IN: pin Step5's baseline stack to the hash-verified inventory ---
    # Without this key production Step5 falls back to scanning the baseline
    # directory, which would let an unmanaged or stale GeoTIFF into the
    # composite. With it, neither the Step4 manifest lookup nor the directory
    # scan runs.
    if baseline_binding is not None:
        ctx[STEP5_EXPLICIT_BASELINE_PATHS_KEY] = [
            Path(path) for path in baseline_binding["paths"]
        ]
        ctx["baseline_binding_source"] = baseline_binding["baseline_binding_source"]
        ctx["baseline_directory_scan_used"] = False

    # --- OPT-IN: variant-aware Step8A date validation -----------------------
    # Step8A pins the CANONICAL predictor window of a manually verified
    # experiment. A closure variant runs the same chain over a preregistered
    # SHIFTED window, so the expectation source -- never the strictness -- has
    # to move to the frozen preregistration. Step8A re-validates every field
    # below against that document itself and refuses anything that is not a
    # preregistered non-canonical variant of this exact analysis_id.
    if analysis_id is not None:
        ctx["window_closure_variant_mode"] = True
        ctx["base_experiment_id"] = experiment_id
        ctx["analysis_id"] = analysis_id
        ctx["variant_id"] = variant_id
        ctx["shift_days"] = int(variant["shift_days"])
        ctx["expected_predictor_start_date"] = variant["predictor_start_date"]
        ctx["expected_predictor_end_date"] = variant["predictor_end_date"]
        # Read-only document reference and the namespace Step8A may write into.
        # Stored as strings: they are not mutable output paths of this context.
        ctx["window_closure_preregistration_path"] = str(
            experiment_root(experiment_id, output_root) / "config" / "preregistration.json"
        )
        ctx["window_closure_allowed_output_root"] = str(downstream)

    assert_local_downstream_context_safe(ctx, experiment_id, variant_id, base_context, output_root)
    return ctx


def assert_local_downstream_context_safe(
    ctx: dict,
    experiment_id: str,
    variant_id: str,
    base_context: dict,
    output_root: Optional[Path] = None,
) -> None:
    """No mutable downstream path may leave the variant's downstream tree."""
    downstream = local_downstream_root(experiment_id, variant_id, output_root).resolve()
    canonical_production = Path(base_context["output_root"]).resolve()
    canonical_variant = variant_root(
        experiment_id, CANONICAL_VARIANT_ID, output_root,
    ).resolve()
    predictor_data = (variant_root(experiment_id, variant_id, output_root) / "data").resolve()
    prelabel_dir = (
        experiment_root(experiment_id, output_root) / "prelabel_censor"
    ).resolve()

    for key in LOCAL_DOWNSTREAM_CONTEXT_PATH_KEYS:
        value = ctx.get(key)
        if value is None:
            continue
        resolved = Path(value).resolve()
        for forbidden, label in (
            (canonical_production, "the canonical production namespace"),
            (canonical_variant, "the canonical variant namespace"),
            (predictor_data, "the predictor-export data namespace"),
            (prelabel_dir, "the pre-label censoring namespace"),
        ):
            if resolved == forbidden or forbidden in resolved.parents:
                raise WindowClosureError(
                    f"Local-downstream context key '{key}' points into "
                    f"{label}: {resolved}."
                )
        if not (resolved == downstream or downstream in resolved.parents):
            raise WindowClosureError(
                f"Local-downstream context key '{key}' escapes the variant "
                f"downstream namespace {downstream}: {resolved}."
            )
    if ctx["label_start_date"] != base_context["label_start_date"] or \
            ctx["label_end_date"] != base_context["label_end_date"]:
        raise WindowClosureError("Label window must be identical to the canonical one.")


def assert_local_downstream_owned_targets(
    experiment_id: str, variant_id: str, targets: Iterable[Path],
    output_root: Optional[Path] = None,
) -> None:
    """This stage may only ever write inside its own downstream tree."""
    downstream = local_downstream_root(experiment_id, variant_id, output_root).resolve()
    metadata = local_downstream_metadata_path(
        experiment_id, variant_id, output_root,
    ).resolve()
    quarantine = (
        variant_root(experiment_id, variant_id, output_root)
        / LOCAL_DOWNSTREAM_QUARANTINE_DIR / LOCAL_DOWNSTREAM_QUARANTINE_KIND
    ).resolve()
    for path in targets:
        resolved = Path(path).resolve()
        if resolved == metadata:
            continue
        if downstream in resolved.parents or resolved == downstream:
            continue
        if quarantine in resolved.parents or resolved == quarantine:
            continue
        raise WindowClosureError(
            f"'{resolved}' is not a local-downstream-owned target. This stage "
            f"writes only inside {downstream} and its own "
            f"{LOCAL_DOWNSTREAM_METADATA_NAME}."
        )


# --- The production engine (no Earth Engine anywhere) -------------------------
def production_local_downstream_engine(
    variant_context: dict, variant: dict, plan: dict,
) -> dict:
    """Run Step5 -> Step5C -> Step7A-E -> Step8A through the production helpers.

    Every helper is the production one and receives the variant context, so the
    scientific calculation is byte-for-byte the canonical implementation; only
    the dates in the context and the directories it points at differ. Nothing
    is imported until this function is reached, and none of these modules
    touches Earth Engine.
    """
    import src.step5_preprocess_timeseries as step5
    import src.step5c_tvdi as step5c
    import src.step7a_tiling_infrastructure as step7a
    import src.step7b_prepare_downscaling_dataset as step7b
    import src.step7c_train_downscaling_model as step7c
    import src.step7d_predict_downscaled_lst as step7d
    import src.step7e_fuse_landsat_downscaled_lst as step7e
    import src.step8a_prepare_500m_modeling_dataset as step8a

    runners = {
        "step5": step5.run_step5,
        "step5c": step5c.run_step5c,
        "step7a": step7a.run_step7a,
        "step7b": step7b.run_step7b,
        "step7c": step7c.run_step7c,
        "step7d": step7d.run_step7d,
        "step7e": step7e.run_step7e,
        "step8a": step8a.run_step8a,
    }
    stage_results: dict[str, Any] = {}
    stages_run: list[str] = []
    for stage in PRODUCTION_STAGE_SEQUENCE:
        runner = runners[stage]
        if stage in PRODUCTION_STAGES_WITHOUT_FORCE:
            stage_results[stage] = runner(ctx=variant_context)
        else:
            # The downstream tree is empty (fresh) or quarantined before this
            # engine is reached, so `force` only prevents a spurious refusal on
            # a leftover from a crashed run inside THIS namespace.
            stage_results[stage] = runner(ctx=variant_context, force=True)
        stages_run.append(stage)
    return {
        "stages_run": stages_run,
        "stage_results": {
            stage: type(result).__name__ for stage, result in stage_results.items()
        },
    }


# --- Artifact inventory -------------------------------------------------------
MEDIA_TYPES: dict[str, str] = {
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".parquet": "application/vnd.apache.parquet",
    ".json": "application/json",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".geojson": "application/geo+json",
    ".pkl": "application/octet-stream",
    ".joblib": "application/octet-stream",
    ".npy": "application/octet-stream",
    ".npz": "application/octet-stream",
}


def _media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _raster_artifact_detail(path: Path) -> dict:
    import numpy as np
    import rasterio

    with rasterio.open(path) as dataset:
        transform = [float(value) for value in tuple(dataset.transform)[:6]]
        detail = {
            "band_count": int(dataset.count),
            "dtype": str(dataset.dtypes[0]),
            "nodata": None if dataset.nodata is None else float(dataset.nodata),
            "width": int(dataset.width),
            "height": int(dataset.height),
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": transform,
        }
        band = dataset.read(1, masked=True)
    values = band.compressed().astype("float64")
    values = values[np.isfinite(values)]
    detail["grid_signature"] = {
        "crs": detail["crs"], "transform": transform,
        "width": detail["width"], "height": detail["height"],
        "band_count": detail["band_count"], "dtype": detail["dtype"],
        "nodata": detail["nodata"],
    }
    detail["finite_cell_count"] = int(values.size)
    detail["min_finite"] = float(values.min()) if values.size else None
    detail["max_finite"] = float(values.max()) if values.size else None
    return detail


def _parquet_artifact_detail(path: Path) -> dict:
    import pandas as pd

    frame = pd.read_parquet(path)
    key_column = step8a_key_column(frame)
    duplicate_key_count = (
        int(frame[key_column].duplicated().sum()) if key_column else None
    )
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "key_column": key_column,
        "duplicate_key_count": duplicate_key_count,
        "primary_population_row_count": (
            int(frame[PRIMARY_POPULATION].astype(bool).sum())
            if PRIMARY_POPULATION in frame.columns else None
        ),
        "burned_count": (
            int((frame["burned"].astype(int) == 1).sum())
            if "burned" in frame.columns else None
        ),
        "unburned_count": (
            int((frame["burned"].astype(int) == 0).sum())
            if "burned" in frame.columns else None
        ),
    }


def _json_artifact_detail(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {"schema_version": None, "deterministic_sha256": None, "readable": False}
    version = None
    if isinstance(payload, dict):
        for key in ("schema_version", "schema", "version", "step"):
            if payload.get(key) is not None:
                version = str(payload[key])
                break
    return {
        "schema_version": version,
        "deterministic_sha256": sha256_bytes(canonical_json(payload).encode("utf-8")),
        "readable": True,
    }


def inspect_local_downstream_artifact(
    path: Path, stage: str, root: Path, input_roles: Sequence[str],
) -> dict:
    """One stage-owned artefact, fully described. Raises on an unusable file."""
    if not path.is_file():
        raise WindowClosureError(f"Local-downstream artefact is missing: {path}.")
    size_bytes = int(path.stat().st_size)
    if size_bytes == 0:
        raise WindowClosureError(f"Local-downstream artefact is empty (0 bytes): {path}.")
    relative = path.relative_to(root).as_posix()
    record: dict[str, Any] = {
        "artifact_id": f"{stage}/{relative}",
        "stage": stage,
        "role": relative,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": size_bytes,
        "media_type": _media_type(path),
        "producer": (
            f"{PRODUCTION_STAGE_HELPERS[stage]['module']}."
            f"{PRODUCTION_STAGE_HELPERS[stage]['function']}"
        ),
        "input_roles": list(input_roles),
        "variant_derived": True,
        "status": STATUS_PASS,
    }
    suffix = path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            record.update(_raster_artifact_detail(path))
        elif suffix == ".parquet":
            record.update(_parquet_artifact_detail(path))
        elif suffix in (".json", ".geojson"):
            record.update(_json_artifact_detail(path))
    except WindowClosureError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any reader failure is a contract failure
        raise WindowClosureError(
            f"Local-downstream artefact {path} could not be inspected: "
            f"{type(exc).__name__}: {exc}."
        ) from exc
    return record


def build_local_downstream_artifact_inventory(
    experiment_id: str, variant_id: str, stages: Sequence[str],
    baseline_years: Sequence[int], output_root: Optional[Path] = None,
) -> list[dict]:
    """Deterministic inventory of everything the stages actually produced.

    The stage-owned outputs are discovered from the stage directories rather
    than assumed, so no raster/parquet/json count is hard-coded anywhere.
    """
    stage_inputs = production_stage_input_roles(baseline_years)
    root = local_downstream_root(experiment_id, variant_id, output_root)
    inventory: list[dict] = []
    for stage in stages:
        stage_dir = local_downstream_stage_dir(experiment_id, variant_id, stage, output_root)
        if not stage_dir.is_dir():
            raise WindowClosureError(
                f"Production stage '{stage}' produced no output directory at "
                f"{stage_dir}."
            )
        files = sorted(
            path for path in stage_dir.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )
        if not files:
            raise WindowClosureError(
                f"Production stage '{stage}' produced no artefact in {stage_dir}."
            )
        for path in files:
            inventory.append(
                inspect_local_downstream_artifact(
                    path, stage, root, stage_inputs.get(stage, []),
                )
            )
    inventory.sort(key=lambda record: record["artifact_id"])
    return inventory


# =============================================================================
# Step8A feature contract and static/label invariance
# =============================================================================
# Column groups of the production Step8A record, taken from
# `src.step8a_prepare_500m_modeling_dataset` (the per-cell record it builds) and
# from the Step8B feature registry. The PREDICTOR-derived groups are NOT listed
# here: they are derived per experiment from the frozen Step8A lineage, see
# `step8a_predictor_lineage`.
STEP8A_KEY_COLUMNS: tuple[str, ...] = ("cell_id", "row_500m", "col_500m", "lon", "lat")
STEP8A_LABEL_COLUMNS: tuple[str, ...] = (
    "burned", "burn_date", "burn_month", "burn_day_of_year", "label_source",
    "burn_date_pixel_agreement_fraction", "out_of_window_burndate",
)
STEP8A_OPTIONAL_AUDIT_COLUMNS: frozenset[str] = frozenset({
    "analysis_eligible", "pre_label_burn_excluded",
})
STEP8A_POPULATION_COLUMNS: tuple[str, ...] = (
    "landcover_dominant", "landcover_tree_cover_fraction",
    "landcover_shrubland_fraction", "landcover_grassland_fraction",
    "landcover_cropland_fraction", "landcover_bare_sparse_vegetation_fraction",
    "landcover_built_up_fraction", "landcover_permanent_water_fraction",
    "burnable_tree_shrub_grass", "burnable_tree_shrub",
)
# Block geometry: fixed by the reference grid and the frozen block size, so it
# is invariant AS LONG AS the variant reference grid equals the canonical one --
# which this stage verifies explicitly rather than assuming.
STEP8A_GRID_SUPPORT_COLUMNS: tuple[str, ...] = ("total_30m_pixel_count",)
# Support/validity columns that follow from PREDICTOR availability, so they move
# with the closure date exactly as the predictor features do.
STEP8A_TIMING_SUPPORT_COLUMNS: tuple[str, ...] = (
    "valid_30m_pixel_count", "valid_30m_fraction", "observed_fraction",
    "gapfilled_fraction", "invalid_source_fraction", "source_mask_majority",
    "thermal_any_missing", "valid_for_modeling", "invalid_reason",
)
STEP8A_PREDICTOR_COLUMN_SUFFIXES: tuple[str, ...] = (
    "_mean", "_median", "_std", "_valid_count", "_valid_fraction",
)
# Canonical directories a Step8A predictor source may live in, and whether a
# product from that directory is rebuilt by this analysis (and therefore may
# legitimately change) or is a frozen static input (and therefore must not).
TIMING_DERIVED_SOURCE_DIRS: tuple[str, ...] = (
    "step5", "step5c", "step7d", "step7e",
    "data/current_period", "data/ndvi_current_period",
    "data/landsat_timeseries", "data/ndvi_timeseries", "data/modis",
)
STATIC_SOURCE_DIRS: tuple[str, ...] = ("data/dem",)


def step8a_key_column(frame) -> Optional[str]:
    """The stable production key of a Step8A dataset.

    `cell_id` is the production key: Step8A's `compute_cell_identity` is the
    single source of truth for it and derives it from the 500 m block offsets,
    so the SAME id always names the same physical block. No floating-point
    fuzzy matching is invented; when `cell_id` is absent the production
    (row_500m, col_500m) grid key is reused instead.
    """
    columns = set(getattr(frame, "columns", []))
    if "cell_id" in columns:
        return "cell_id"
    if {"row_500m", "col_500m"} <= columns:
        return "row_500m__col_500m"
    return None


def _key_series(frame, key_column: str):
    if key_column == "row_500m__col_500m":
        return (
            frame["row_500m"].astype("int64").astype(str)
            + "_" + frame["col_500m"].astype("int64").astype(str)
        )
    return frame[key_column].astype(str)


def step8a_predictor_lineage(
    experiment_id: str, experiments_root: Optional[Path] = None,
) -> dict:
    """Which Step8A predictor prefixes move with the closure date, and which do not.

    Derived from the FROZEN canonical Step8A run metadata
    (`step8a_dataset_stats.json:predictor_paths`), which records the exact
    raster each predictor family was built from. A predictor whose source lives
    in a stage this analysis rebuilds (Step5, Step5C, Step7D, Step7E, or a
    re-exported predictor input) is timing-derived; a predictor whose source is
    a frozen static input (the DEM) is not. Nothing is guessed and no whitelist
    is hard-coded.
    """
    root = canonical_experiment_root(experiment_id, experiments_root)
    stats_path = canonical_step8a_stats_path(experiment_id, experiments_root)
    if not stats_path.is_file():
        raise WindowClosureError(
            f"Canonical Step8A stats are missing at {stats_path}; the "
            "timing-derived feature lineage cannot be derived and must never "
            "be guessed."
        )
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise WindowClosureError(
            f"Canonical Step8A stats at {stats_path} are unreadable: {exc}."
        ) from exc
    predictor_paths = (stats or {}).get("predictor_paths")
    if not isinstance(predictor_paths, dict) or not predictor_paths:
        raise WindowClosureError(
            f"Canonical Step8A stats at {stats_path} record no predictor_paths; "
            "the timing-derived feature lineage cannot be derived."
        )

    timing: list[str] = []
    static: list[str] = []
    lineage: dict[str, dict] = {}
    for name, raw in sorted(predictor_paths.items()):
        source = Path(str(raw))
        try:
            relative = source.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            raise WindowClosureError(
                f"Canonical Step8A predictor '{name}' points outside the "
                f"canonical experiment namespace: {source}."
            )
        if any(relative.startswith(f"{prefix}/") for prefix in TIMING_DERIVED_SOURCE_DIRS):
            timing.append(name)
            kind = "timing_derived"
        elif any(relative.startswith(f"{prefix}/") for prefix in STATIC_SOURCE_DIRS):
            static.append(name)
            kind = "static"
        else:
            raise WindowClosureError(
                f"Canonical Step8A predictor '{name}' has an unrecognised "
                f"source directory '{relative}'. The timing-derived whitelist "
                "must be re-derived from the production lineage rather than "
                "guessed."
            )
        lineage[name] = {
            "predictor": name,
            "canonical_source": relative,
            "classification": kind,
        }
    return {
        "source": str(stats_path),
        "timing_derived_predictors": sorted(timing),
        "static_predictors": sorted(static),
        "lineage": lineage,
        "column_suffixes": list(STEP8A_PREDICTOR_COLUMN_SUFFIXES),
    }


def classify_step8a_columns(columns: Sequence[str], lineage: dict) -> dict:
    """Split the canonical Step8A columns into invariant and timing-derived.

    Any column that cannot be classified fails: an unrecognised column means the
    production Step8A schema moved and the invariance whitelist has to be
    re-derived, which is exactly the situation that must never pass silently.
    """
    available = list(dict.fromkeys(str(column) for column in columns))
    remaining = set(available)

    def _take(names: Iterable[str]) -> list[str]:
        taken = [name for name in names if name in remaining]
        remaining.difference_update(taken)
        return taken

    def _predictor_columns(prefixes: Sequence[str]) -> list[str]:
        names = [
            f"{prefix}{suffix}"
            for prefix in prefixes
            for suffix in STEP8A_PREDICTOR_COLUMN_SUFFIXES
        ]
        return _take(names)

    key = _take(STEP8A_KEY_COLUMNS)
    label = _take(STEP8A_LABEL_COLUMNS)
    audit = _take(sorted(STEP8A_OPTIONAL_AUDIT_COLUMNS))
    population = _take(STEP8A_POPULATION_COLUMNS)
    static_predictor = _predictor_columns(lineage["static_predictors"])
    grid_support = _take(STEP8A_GRID_SUPPORT_COLUMNS)
    timing_predictor = _predictor_columns(lineage["timing_derived_predictors"])
    timing_support = _take(STEP8A_TIMING_SUPPORT_COLUMNS)

    unclassified = sorted(remaining)
    if unclassified:
        raise WindowClosureError(
            f"Step8A column(s) {unclassified} could not be classified as key, "
            "label, population, static-predictor, grid-support, timing-derived "
            "predictor or timing-derived support. The production Step8A schema "
            "has changed, so the static/timing split must be re-derived from "
            "the production lineage before this stage may run."
        )

    invariant = key + label + population + static_predictor + grid_support
    timing_derived = timing_predictor + timing_support
    return {
        "key_columns": key,
        "label_columns": label,
        "audit_columns": audit,
        "population_columns": population,
        "static_predictor_columns": static_predictor,
        "grid_support_columns": grid_support,
        "timing_derived_predictor_columns": timing_predictor,
        "timing_derived_support_columns": timing_support,
        "invariant_columns": invariant,
        "timing_derived_columns": timing_derived,
        "all_columns": available,
    }


# --- Semantic dtype compatibility (deliberately NARROW) -----------------------
# Step8A writes `source_mask_majority` per cell as either `int(code)` -- a
# discrete Step7E fused-source code -- or `np.nan` when the cell has no valid
# source pixel (see `build_dataset`). pandas therefore infers `float64` when a
# dataset happens to contain at least one null cell and `int64` when it does
# not. Observation support legitimately moves with the closure date, so two
# scientifically identical columns can end up with different pandas dtypes for
# no reason other than whether a NaN occurred.
#
# A literal dtype equality check calls that a contract violation. Relaxing the
# check GLOBALLY would be far worse: it would also excuse a real int/float
# change in a continuous feature. So the exemption is restricted to columns
# that are declared DISCRETE PRODUCTION CODES, and even for those every value
# must still be finite, integral and inside the production codebook.
SEMANTIC_TYPE_NULLABLE_INTEGER_CODE = "nullable_integer_categorical_code"


def step8a_discrete_code_domains() -> dict[str, tuple[int, ...]]:
    """Step8A columns that carry discrete production codes, with their domain.

    The domain is READ FROM the production module that defines it -- the
    Step7E `fused_lst_source_mask` codebook exposed as
    `src.step8a_prepare_500m_modeling_dataset.SOURCE_INVALID / SOURCE_OBSERVED
    / SOURCE_GAPFILL` -- so it can never drift from production and is never
    guessed or re-listed here.
    """
    from src.step8a_prepare_500m_modeling_dataset import (
        SOURCE_GAPFILL, SOURCE_INVALID, SOURCE_OBSERVED,
    )

    return {
        "source_mask_majority": tuple(sorted({
            int(SOURCE_INVALID), int(SOURCE_OBSERVED), int(SOURCE_GAPFILL),
        })),
    }


def _discrete_code_values(series, label: str, domain: Sequence[int]) -> tuple[list, Optional[str]]:
    """Non-null values of a discrete-code column, or the reason it is not one.

    Read-only: it never fills, rounds, casts in place or otherwise touches the
    caller's frame -- `dropna()` returns a copy and nulls are counted, not
    removed from the data.
    """
    import numpy as np
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return [], f"{label} is boolean; a discrete code must be numeric"
    if not pd.api.types.is_numeric_dtype(series):
        return [], (
            f"{label} has non-numeric dtype {series.dtype}; object/string/"
            "datetime representations of a code are not accepted"
        )
    values = series.dropna().to_numpy(dtype="float64")
    if values.size and not np.all(np.isfinite(values)):
        return [], f"{label} carries +/-inf values"
    if values.size and not np.all(np.equal(np.mod(values, 1.0), 0.0)):
        fractional = sorted({float(v) for v in values[np.mod(values, 1.0) != 0.0]})
        return [], f"{label} carries fractional code(s) {fractional[:4]}"
    codes = sorted({int(v) for v in values})
    outside = [code for code in codes if code not in set(domain)]
    if outside:
        return [], (
            f"{label} carries code(s) {outside} outside the production "
            f"codebook {list(domain)}"
        )
    return codes, None


def semantic_dtype_compatibility(
    column: str, variant_series, canonical_series, domain: Sequence[int],
) -> dict:
    """Whether two literal dtypes are the SAME discrete production code column.

    Compatible only when both sides are numeric, finite, integral and inside
    the production codebook. Nulls are preserved and reported, never filled and
    never turned into zero.
    """
    variant_codes, variant_reason = _discrete_code_values(
        variant_series, "variant", domain,
    )
    canonical_codes, canonical_reason = _discrete_code_values(
        canonical_series, "canonical", domain,
    )
    reason = variant_reason or canonical_reason
    return {
        "column": column,
        "canonical_dtype": str(canonical_series.dtype),
        "variant_dtype": str(variant_series.dtype),
        "semantic_type": SEMANTIC_TYPE_NULLABLE_INTEGER_CODE,
        "production_code_domain": list(domain),
        "canonical_codes_present": canonical_codes,
        "variant_codes_present": variant_codes,
        "canonical_null_count": int(canonical_series.isna().sum()),
        "variant_null_count": int(variant_series.isna().sum()),
        "nulls_preserved": True,
        "compatibility": "fail" if reason else "pass",
        "reason": reason or (
            "same discrete production code column; the literal dtype differs "
            "only because pandas infers float64 when a null is present and "
            "int64 when none is"
        ),
    }


def step8a_feature_contract(frame, lineage: dict) -> dict:
    """The production feature contract of ONE Step8A dataset."""
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, CATEGORICAL_FEATURES, THERMAL_MODEL_FEATURES,
    )

    columns = [str(column) for column in frame.columns]
    classification = classify_step8a_columns(columns, lineage)
    return {
        "columns": columns,
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "baseline_features_in_order": list(BASELINE_FEATURES),
        "model_feature_columns_in_order": list(THERMAL_MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "label_columns": classification["label_columns"],
        "audit_columns": classification["audit_columns"],
        "population_columns": classification["population_columns"],
        "key_columns": classification["key_columns"],
        "invariant_columns": classification["invariant_columns"],
        "timing_derived_columns": classification["timing_derived_columns"],
        "classification": classification,
        "key_column": step8a_key_column(frame),
    }


def validate_step8a_optional_audit_columns(frame, *, frame_name: str) -> dict:
    """Validate the exact optional pre-label censor audit pair, read-only."""
    import numpy as np
    import pandas as pd

    present = sorted(STEP8A_OPTIONAL_AUDIT_COLUMNS & set(frame.columns))
    if present and set(present) != set(STEP8A_OPTIONAL_AUDIT_COLUMNS):
        raise WindowClosureError(
            f"Step8A optional audit contract failed for {frame_name}: both "
            f"columns are required together; present={present}."
        )
    normalized: dict[str, Any] = {}
    for column in present:
        series = frame[column]
        if series.isna().any():
            raise WindowClosureError(
                f"Step8A optional audit contract failed for {frame_name}: "
                f"{column} contains null values."
            )
        if pd.api.types.is_bool_dtype(series.dtype):
            normalized[column] = series.astype(bool).to_numpy()
        elif pd.api.types.is_numeric_dtype(series.dtype):
            values = series.to_numpy(dtype="float64")
            if not np.all(np.isfinite(values)) or not np.all(np.isin(values, [0.0, 1.0])):
                raise WindowClosureError(
                    f"Step8A optional audit contract failed for {frame_name}: "
                    f"{column} is not boolean-semantic."
                )
            normalized[column] = values.astype(bool)
        else:
            raise WindowClosureError(
                f"Step8A optional audit contract failed for {frame_name}: "
                f"{column} dtype {series.dtype} is not safely boolean-semantic."
            )
    if present and not np.array_equal(
        normalized["analysis_eligible"],
        ~normalized["pre_label_burn_excluded"],
    ):
        raise WindowClosureError(
            f"Step8A optional audit contract failed for {frame_name}: "
            "analysis_eligible must equal NOT pre_label_burn_excluded."
        )
    return {
        "present": present,
        "contract_passed": True,
        "classification": "eligibility/audit metadata; never a model feature",
    }


def assert_step8a_feature_contract(variant_frame, canonical_frame, lineage: dict) -> dict:
    """The variant Step8A dataset must carry the CANONICAL production contract.

    No feature is added, none is dropped and no column is re-ordered to make a
    check pass: the column order is whatever the production Step8A helper
    produced, and it is required to be the canonical one.
    """
    canonical_audit = validate_step8a_optional_audit_columns(
        canonical_frame, frame_name="canonical",
    )
    variant_audit = validate_step8a_optional_audit_columns(
        variant_frame, frame_name="variant",
    )
    canonical_contract = step8a_feature_contract(canonical_frame, lineage)
    variant_contract = step8a_feature_contract(variant_frame, lineage)

    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise WindowClosureError(f"Step8A feature contract failed: {message}")

    _require(len(variant_frame) > 0, "the variant Step8A dataset is empty.")

    canonical_columns = set(canonical_contract["columns"])
    variant_columns = set(variant_contract["columns"])
    legacy_canonical = not canonical_audit["present"]
    _require(
        legacy_canonical or bool(variant_audit["present"]),
        "the canonical audit pair is present but the variant audit pair is absent.",
    )
    added = sorted(variant_columns - canonical_columns)
    dropped = sorted(canonical_columns - variant_columns)
    allowed_added = set(STEP8A_OPTIONAL_AUDIT_COLUMNS) if legacy_canonical else set()
    _require(not (set(added) - allowed_added), f"the variant carries new column(s) {added}.")
    _require(not dropped, f"the variant is missing canonical column(s) {dropped}.")
    canonical_non_audit = [c for c in canonical_contract["columns"] if c not in STEP8A_OPTIONAL_AUDIT_COLUMNS]
    variant_non_audit = [c for c in variant_contract["columns"] if c not in STEP8A_OPTIONAL_AUDIT_COLUMNS]
    _require(
        variant_non_audit == canonical_non_audit,
        "the variant column ORDER differs from the canonical one "
        f"({variant_contract['columns'][:6]} vs {canonical_contract['columns'][:6]}); "
        "the production helper must produce it deterministically and it is "
        "never re-ordered here.",
    )

    model_features = list(canonical_contract["model_feature_columns_in_order"])
    missing_features = [name for name in model_features if name not in variant_columns]
    _require(
        not missing_features,
        f"model feature column(s) {missing_features} are absent from the variant.",
    )
    _require(
        [name for name in variant_contract["columns"] if name in set(model_features)]
        == [name for name in canonical_contract["columns"] if name in set(model_features)],
        "the model feature column set differs from the canonical one.",
    )
    _require(
        variant_contract["label_columns"] == canonical_contract["label_columns"],
        "the label column set differs from the canonical one.",
    )
    _require(
        variant_contract["population_columns"] == canonical_contract["population_columns"],
        "the population/landcover column set differs from the canonical one.",
    )
    _require(
        variant_contract["key_columns"] == canonical_contract["key_columns"],
        "the key/coordinate column set differs from the canonical one.",
    )

    # --- dtype contract --------------------------------------------------
    # Exact by default. The ONLY exemption is a column declared as a discrete
    # production code, whose pandas dtype depends on whether a null happened to
    # occur; every other int/float mismatch is still a hard failure.
    domains = step8a_discrete_code_domains()
    literal_differences: list[dict] = []
    accepted: list[dict] = []
    rejected: list[str] = []
    for column in canonical_non_audit:
        variant_dtype = variant_contract["dtypes"].get(column)
        canonical_dtype = canonical_contract["dtypes"].get(column)
        if variant_dtype == canonical_dtype:
            continue
        literal_differences.append({
            "column": column,
            "variant_dtype": variant_dtype,
            "canonical_dtype": canonical_dtype,
        })
        domain = domains.get(column)
        if domain is None:
            rejected.append(f"{column}: {variant_dtype} != {canonical_dtype}")
            continue
        record = semantic_dtype_compatibility(
            column, variant_frame[column], canonical_frame[column], domain,
        )
        if record["compatibility"] == "pass":
            accepted.append(record)
        else:
            rejected.append(
                f"{column}: {variant_dtype} != {canonical_dtype} "
                f"({record['reason']})"
            )
    _require(not rejected, f"dtype contract broken: {sorted(rejected)[:6]}.")

    key_column = variant_contract["key_column"]
    _require(key_column is not None, "no stable production key column is available.")
    duplicates = int(_key_series(variant_frame, key_column).duplicated().sum())
    _require(duplicates == 0, f"the variant carries {duplicates} duplicate key(s).")

    labels = set(variant_frame["burned"].astype(int).tolist())
    _require(
        {0, 1} <= labels,
        f"the variant label carries only class(es) {sorted(labels)}; the "
        "production Step8A contract requires both burned and unburned cells.",
    )

    return {
        "feature_contract_passed": True,
        "key_column": key_column,
        "key_uniqueness_passed": True,
        "legacy_canonical_audit_columns_absent": legacy_canonical,
        "optional_audit_columns_present_in_variant": variant_audit["present"],
        "optional_audit_contract_passed": True,
        "model_feature_registry_unchanged": True,
        "canonical_bytes_unchanged": True,
        # Computed in the VALIDATION layer only: no parquet, CSV, raster or
        # value is read-modified-written to reconcile a dtype.
        "semantic_dtype_contract": {
            "discrete_code_columns": sorted(domains),
            "production_code_domains": {
                column: list(domain) for column, domain in sorted(domains.items())
            },
            "source": (
                "src.step8a_prepare_500m_modeling_dataset SOURCE_INVALID / "
                "SOURCE_OBSERVED / SOURCE_GAPFILL (Step7E fused_lst_source_mask "
                "codebook)"
            ),
            "exact_dtype_required_elsewhere": True,
        },
        "literal_dtype_differences": literal_differences,
        "accepted_semantic_dtype_compatibilities": accepted,
        "canonical_feature_contract_sha256": sha256_bytes(
            canonical_json({
                "columns": canonical_contract["columns"],
                "dtypes": canonical_contract["dtypes"],
                "model_feature_columns_in_order":
                    canonical_contract["model_feature_columns_in_order"],
                "label_columns": canonical_contract["label_columns"],
                "population_columns": canonical_contract["population_columns"],
                "key_columns": canonical_contract["key_columns"],
                "invariant_columns": canonical_contract["invariant_columns"],
                "timing_derived_columns": canonical_contract["timing_derived_columns"],
            }).encode("utf-8")
        ),
        "canonical_contract": canonical_contract,
        "variant_contract": variant_contract,
    }


# Absolute tolerance for the invariant FLOAT columns. Coordinates, DEM and
# landcover statistics are aggregated by the same production code from the same
# frozen rasters, so they must agree to floating-point noise -- not "roughly".
STATIC_INVARIANCE_ABS_TOLERANCE = 1e-9


def compare_step8a_invariance(
    variant_frame, canonical_frame, contract: dict, key_column: str,
) -> dict:
    """Static and label invariance on the COMMON keys. No cohort is built.

    The variant and the canonical dataset may legitimately contain different
    numbers of rows -- missingness and observation support move with the closure
    date -- so only the overlap is compared, and the row-count differences are
    reported rather than treated as failures.
    """
    import numpy as np
    import pandas as pd

    classification = contract["canonical_contract"]["classification"]
    invariant_columns = [
        column for column in classification["invariant_columns"]
        if column in variant_frame.columns and column in canonical_frame.columns
    ]

    variant_keys = _key_series(variant_frame, key_column)
    canonical_keys = _key_series(canonical_frame, key_column)
    for label, keys in (("variant", variant_keys), ("canonical", canonical_keys)):
        duplicates = int(keys.duplicated().sum())
        if duplicates:
            raise WindowClosureError(
                f"The {label} Step8A dataset carries {duplicates} duplicate "
                f"'{key_column}' value(s); the overlap comparison needs a "
                "unique production key."
            )
    variant_indexed = variant_frame.assign(_wcs_key=variant_keys.to_numpy()).set_index("_wcs_key")
    canonical_indexed = canonical_frame.assign(
        _wcs_key=canonical_keys.to_numpy()
    ).set_index("_wcs_key")

    overlap = sorted(set(variant_indexed.index) & set(canonical_indexed.index))
    variant_only = sorted(set(variant_indexed.index) - set(canonical_indexed.index))
    canonical_only = sorted(set(canonical_indexed.index) - set(variant_indexed.index))
    if not overlap:
        raise WindowClosureError(
            "The variant and canonical Step8A datasets share no key; static and "
            "label invariance cannot be verified."
        )

    left = variant_indexed.loc[overlap]
    right = canonical_indexed.loc[overlap]
    mismatches: list[dict] = []
    for column in invariant_columns:
        a, b = left[column], right[column]
        if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
            av, bv = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
            equal = np.isclose(
                av, bv, rtol=0.0, atol=STATIC_INVARIANCE_ABS_TOLERANCE, equal_nan=True,
            )
        else:
            equal = (a.to_numpy() == b.to_numpy()) | (a.isna().to_numpy() & b.isna().to_numpy())
        differing = int((~equal).sum())
        if differing:
            index = int(np.argmax(~equal))
            mismatches.append({
                "column": column,
                "differing_rows": differing,
                "example_key": str(overlap[index]),
                "example_variant_value": str(a.iloc[index]),
                "example_canonical_value": str(b.iloc[index]),
            })
    if mismatches:
        raise WindowClosureError(
            "Static/label invariance failed on the common keys: "
            f"{json.dumps(mismatches[:6], sort_keys=True)}. Only "
            "predictor-timing-derived features may change between variants; "
            "labels, DEM, slope, landcover, population, coordinates and the "
            "spatial block geometry are frozen."
        )

    label_columns = [
        column for column in classification["label_columns"]
        if column in invariant_columns
    ]
    static_columns = [
        column for column in invariant_columns if column not in label_columns
    ]
    return {
        "static_invariance_passed": True,
        "label_invariance_passed": True,
        "compared_invariant_columns": invariant_columns,
        "compared_label_columns": label_columns,
        "compared_static_columns": static_columns,
        "timing_derived_columns_allowed_to_change": list(
            classification["timing_derived_columns"]
        ),
        "key_column": key_column,
        "variant_row_count": int(len(variant_frame)),
        "canonical_row_count": int(len(canonical_frame)),
        "overlap_row_count": int(len(overlap)),
        "variant_only_row_count": int(len(variant_only)),
        "canonical_only_row_count": int(len(canonical_only)),
        "row_count_difference_is_not_a_failure": True,
        "common_cohort_created": False,
    }


def assert_reference_grid_matches_canonical(
    variant_stats_path: Path, canonical_stats_path: Path,
) -> dict:
    """The variant Step8A must sit on the canonical 500 m/30 m reference grid.

    Step8A derives its cell identities from the reference 30 m grid, so a grid
    difference would silently make `cell_id` mean something else in the two
    datasets. Comparing the grids explicitly is what lets the block geometry be
    treated as invariant instead of assumed.
    """
    def _grid(path: Path, label: str) -> dict:
        if not path.is_file():
            raise WindowClosureError(f"{label} Step8A stats are missing at {path}.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise WindowClosureError(f"{label} Step8A stats at {path} are unreadable: {exc}.") from exc
        grid = (payload or {}).get("reference_30m_grid")
        if not isinstance(grid, dict):
            raise WindowClosureError(f"{label} Step8A stats record no reference_30m_grid.")
        return grid

    variant_grid = _grid(variant_stats_path, "Variant")
    canonical_grid = _grid(canonical_stats_path, "Canonical")
    signature = ("width", "height", "crs", "transform")
    differences = {
        key: [variant_grid.get(key), canonical_grid.get(key)]
        for key in signature
        if variant_grid.get(key) != canonical_grid.get(key)
    }
    if differences:
        raise WindowClosureError(
            "The variant Step8A reference grid differs from the canonical one: "
            f"{json.dumps(differences, sort_keys=True, default=str)}. Cell "
            "identities would not describe the same physical blocks."
        )
    return {
        "reference_grid_matches_canonical": True,
        "reference_grid": {key: canonical_grid.get(key) for key in signature},
    }


# =============================================================================
# The LOCAL-DOWNSTREAM stage
# =============================================================================
def local_downstream_planned_stage_outputs(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> dict[str, str]:
    return {
        stage: str(local_downstream_stage_dir(experiment_id, variant_id, stage, output_root))
        for stage in PRODUCTION_STAGE_SEQUENCE
    }


def local_downstream_variant_plan(
    experiment_id: str, variant: dict, baseline_years: Sequence[int],
    output_root: Optional[Path] = None, experiments_root: Optional[Path] = None,
) -> dict:
    """Pure per-variant plan, shared by the dry run and the actual stage."""
    variant_id = variant["variant_id"]
    if variant["is_canonical"]:
        return {
            "variant_id": variant_id,
            "shift_days": int(variant["shift_days"]),
            "predictor_start_date": variant["predictor_start_date"],
            "predictor_end_date": variant["predictor_end_date"],
            "lead_days": int(variant["lead_days"]),
            "export_enabled": False,
            "frozen_reference_only": True,
            "planned_output_count": 0,
            "planned_stage_outputs": {},
            "planned_step8a_path": None,
            "reason": (
                "The canonical variant reads the frozen production Step8A "
                "dataset; re-running its downstream chain would replace the "
                "very reference the early closures are compared against."
            ),
        }
    return {
        "variant_id": variant_id,
        "shift_days": int(variant["shift_days"]),
        "predictor_start_date": variant["predictor_start_date"],
        "predictor_end_date": variant["predictor_end_date"],
        "lead_days": int(variant["lead_days"]),
        "baseline_years": [int(year) for year in baseline_years],
        "export_enabled": True,
        "frozen_reference_only": False,
        "predictor_metadata_path": str(
            predictor_metadata_path(experiment_id, variant_id, output_root)
        ),
        "predictor_artifact_count": expected_raster_count(baseline_years),
        "production_stage_sequence": list(PRODUCTION_STAGE_SEQUENCE),
        "planned_stage_outputs": local_downstream_planned_stage_outputs(
            experiment_id, variant_id, output_root,
        ),
        "planned_input_root": str(
            local_downstream_input_root(experiment_id, variant_id, output_root)
        ),
        "planned_step8a_path": str(
            variant_step8a_dataset_path(experiment_id, variant_id, output_root)
        ),
        "planned_step8a_stats_path": str(
            variant_step8a_stats_path(experiment_id, variant_id, output_root)
        ),
        "planned_metadata_path": str(
            local_downstream_metadata_path(experiment_id, variant_id, output_root)
        ),
        "feature_contract_source": str(
            canonical_step8a_path(experiment_id, experiments_root)
        ),
        "static_invariance_check_planned": True,
        "label_invariance_check_planned": True,
        "downscaling_model_fit_planned": True,
        "fire_risk_model_fit": False,
        "baseline_binding_source": BASELINE_BINDING_SOURCE,
        "baseline_directory_scan_used": False,
    }


def local_downstream_summary(
    experiment_id: str,
    analysis_id: str,
    canonical: dict,
    variants: Sequence[dict],
    inventory: dict,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
) -> dict:
    """The whole-analysis local-downstream plan, as reported by a dry run.

    Read-only: nothing is created, no production helper is imported and no
    Earth Engine module is touched.
    """
    baseline_years = list(canonical["baseline_years"])
    early = nonzero_variants(variants)
    readiness = inspect_predictor_binding_readiness(
        experiment_id, analysis_id, canonical, variants, output_root,
    )
    plans = {
        variant["variant_id"]: local_downstream_variant_plan(
            experiment_id, variant, baseline_years, output_root, experiments_root,
        )
        for variant in variants
    }
    for variant_id, entry in readiness["per_variant"].items():
        plans[variant_id].update({
            "predictor_metadata_present": entry["predictor_metadata_present"],
            "predictor_metadata_sha256": entry["predictor_metadata_sha256"],
            "predictor_binding_ready": entry["binding_ready"],
            "predictor_binding_reason": entry["reason"],
        })
    root = experiment_root(experiment_id, output_root).resolve()
    contained = True
    for variant in early:
        plan = plans[variant["variant_id"]]
        for path in [plan["planned_step8a_path"], plan["planned_step8a_stats_path"],
                     plan["planned_input_root"], plan["planned_metadata_path"],
                     *plan["planned_stage_outputs"].values()]:
            if root not in Path(path).resolve().parents:
                contained = False
    canonical_stats = canonical_step8a_stats_path(experiment_id, experiments_root)
    return {
        "canonical_processing_enabled": False,
        "canonical_frozen_reference_only": True,
        "canonical_step8a_path": (inventory.get("canonical_step8a") or {}).get("path"),
        "canonical_step8a_sha256": (inventory.get("canonical_step8a") or {}).get("sha256"),
        "canonical_step8a_stats_path": str(canonical_stats),
        "canonical_step8a_stats_sha256": (
            sha256_file(canonical_stats) if canonical_stats.is_file() else None
        ),
        "nonzero_variant_ids": [variant["variant_id"] for variant in early],
        "production_stage_sequence": list(PRODUCTION_STAGE_SEQUENCE),
        "production_helpers": {
            stage: f"{spec['module']}.{spec['function']}"
            for stage, spec in sorted(PRODUCTION_STAGE_HELPERS.items())
        },
        "predictor_binding_ready": readiness["predictor_binding_ready"],
        "all_predictor_artifacts_present": readiness["all_predictor_artifacts_present"],
        "all_predictor_hashes_match": readiness["all_predictor_hashes_match"],
        "all_paths_inside_dedicated_namespace": contained,
        "predictor_artifacts_per_variant": expected_raster_count(baseline_years),
        "baseline_years": [int(year) for year in baseline_years],
        "primary_population": PRIMARY_POPULATION,
        "primary_population_filter_applied": False,
        "common_cohort_created": False,
        "gee_queries_run": False,
        "gee_exports_run": False,
        # A dry run fits nothing. It only declares that an actual run WOULD
        # train the production Step7C downscaling model -- and that it would
        # still not train the fire-risk model, whose stage stays locked.
        **LOCAL_DOWNSTREAM_DRY_RUN_MODEL_SEMANTICS,
        "variant_plans": plans,
        "limitations": list(LOCAL_DOWNSTREAM_LIMITATIONS),
    }


# =============================================================================
# Stage-owned state snapshot (READ-ONLY)
#
# A dry run may find stage-owned paths that a PREVIOUS actual run left behind
# -- a partial downstream tree, a Step7C downscaling model, an interrupted
# Step8A. Their mere existence says nothing about the dry run; what matters is
# that the dry run does not create, modify or delete anything. So instead of
# demanding an empty tree, the dry run records the stage-owned state before and
# after planning and reports the difference.
#
# Nothing here creates a directory, writes a file, moves anything or rewrites
# any timestamp: it only stats and reads.
# =============================================================================
LOCAL_DOWNSTREAM_STAGE_OWNED_NAMES: tuple[str, ...] = (
    LOCAL_DOWNSTREAM_ROOT_DIR, LOCAL_DOWNSTREAM_METADATA_NAME,
)


def local_downstream_stage_owned_paths(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> list[Path]:
    """The two paths this stage owns for one variant."""
    root = variant_root(experiment_id, variant_id, output_root)
    return [root / name for name in LOCAL_DOWNSTREAM_STAGE_OWNED_NAMES]


def _relative_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def snapshot_local_downstream_state(
    experiment_id: str, variants: Sequence[dict], output_root: Optional[Path] = None,
) -> dict:
    """Read-only inventory of the stage-owned paths that ALREADY exist.

    Files carry their relative path, size and SHA-256; directories carry a
    deterministic relative-path inventory only. The aggregate digest covers the
    CONTENT view (relative path + size + hash), so it is stable across hosts
    and is what before/after equality is decided on.

    This function never creates, writes, moves or deletes anything, and it
    never touches a file's mtime -- it opens files read-only to hash them.
    """
    experiment = experiment_root(experiment_id, output_root)
    roots: dict[str, dict[str, str]] = {}
    directories: set[str] = set()
    files: dict[str, dict] = {}

    for variant in nonzero_variants(variants):
        variant_id = variant["variant_id"]
        downstream, metadata = local_downstream_stage_owned_paths(
            experiment_id, variant_id, output_root,
        )
        roots[variant_id] = {
            LOCAL_DOWNSTREAM_ROOT_DIR: str(downstream),
            LOCAL_DOWNSTREAM_METADATA_NAME: str(metadata),
        }
        for target in (downstream, metadata):
            if not target.exists():
                continue
            if target.is_dir():
                directories.add(_relative_label(target, experiment))
                for path in sorted(target.rglob("*")):
                    label = _relative_label(path, experiment)
                    if path.is_dir():
                        directories.add(label)
                    elif path.is_file():
                        files[label] = {
                            "relative_path": label,
                            "path": str(path),
                            "bytes": int(path.stat().st_size),
                            "sha256": sha256_file(path),
                        }
            elif target.is_file():
                label = _relative_label(target, experiment)
                files[label] = {
                    "relative_path": label,
                    "path": str(target),
                    "bytes": int(target.stat().st_size),
                    "sha256": sha256_file(target),
                }

    content_view = {
        "directories": sorted(directories),
        "files": {
            label: {"bytes": record["bytes"], "sha256": record["sha256"]}
            for label, record in sorted(files.items())
        },
    }
    return {
        "experiment_root": str(experiment),
        "stage_owned_names": list(LOCAL_DOWNSTREAM_STAGE_OWNED_NAMES),
        "stage_owned_roots": roots,
        "directories": sorted(directories),
        "directory_count": len(directories),
        "files": dict(sorted(files.items())),
        "file_count": len(files),
        "digest": sha256_bytes(canonical_json(content_view).encode("utf-8")),
    }


def local_downstream_state_diff(before: dict, after: dict) -> dict:
    """What changed between two stage-owned snapshots. Empty means untouched."""
    before_files, after_files = before["files"], after["files"]
    before_dirs = set(before["directories"])
    after_dirs = set(after["directories"])

    created = sorted(
        (set(after_files) - set(before_files)) | (after_dirs - before_dirs)
    )
    deleted = sorted(
        (set(before_files) - set(after_files)) | (before_dirs - after_dirs)
    )
    modified = sorted(
        label for label in set(before_files) & set(after_files)
        if (before_files[label]["sha256"], before_files[label]["bytes"])
        != (after_files[label]["sha256"], after_files[label]["bytes"])
    )
    return {
        "preexisting_stage_owned_paths": sorted(
            set(before_files) | before_dirs
        ),
        "stage_owned_snapshot_before": before,
        "stage_owned_snapshot_after": after,
        "stage_owned_snapshot_before_sha256": before["digest"],
        "stage_owned_snapshot_after_sha256": after["digest"],
        "stage_owned_snapshot_unchanged": (
            before["digest"] == after["digest"]
            and not created and not deleted and not modified
        ),
        "dry_run_created_paths": created,
        "dry_run_modified_paths": modified,
        "dry_run_deleted_paths": deleted,
    }


def local_downstream_frozen_inputs(
    experiment_id: str, inventory: dict, variants: Sequence[dict],
    output_root: Optional[Path] = None, experiments_root: Optional[Path] = None,
    censor_binding: Optional[dict] = None,
) -> dict:
    """Everything this stage must not disturb, hashed.

    The frozen canonical Step8A dataset and its stats, the DEM, the slope, the
    aligned landcover, both label rasters, the shared pre-label raster, the
    bound pre-label exclusion gate documents, every variant's predictor
    metadata and every one of its predictor rasters.
    """
    extended = dict(predictor_frozen_inputs(experiment_id, inventory, output_root))
    for role, record in sorted((censor_binding or {}).get("documents", {}).items()):
        if not record.get("exists"):
            continue
        extended[f"prelabel_exclusion__{role}"] = {
            "path": record["path"],
            "exists": True,
            "sha256": record["sha256"],
        }
    stats = canonical_step8a_stats_path(experiment_id, experiments_root)
    extended["canonical_step8a_stats"] = {
        "path": str(stats),
        "exists": stats.is_file(),
        "sha256": sha256_file(stats) if stats.is_file() else None,
    }
    for variant in nonzero_variants(variants):
        variant_id = variant["variant_id"]
        metadata_path = predictor_metadata_path(experiment_id, variant_id, output_root)
        extended[f"predictor_export_metadata__{variant_id}"] = {
            "path": str(metadata_path),
            "exists": metadata_path.is_file(),
            "sha256": sha256_file(metadata_path) if metadata_path.is_file() else None,
        }
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        for record in (metadata.get("artifact_inventory") or []):
            if not isinstance(record, dict):
                continue
            path = Path(str(record.get("path") or ""))
            extended[f"predictor_raster__{variant_id}__{record.get('artifact_id')}"] = {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
    return extended


def _quarantine_local_downstream(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
    *, reason: str = "explicit force rebuild",
) -> list[str]:
    """Move the stage-owned outputs aside instead of deleting them.

    Only `downstream/` and `local_downstream_metadata.json` are moved. Predictor
    data, predictor metadata, plan documents, the pre-label raster, the
    canonical outputs and any unmanaged file are never touched, and nothing is
    ever removed.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_root = (
        variant_root(experiment_id, variant_id, output_root)
        / LOCAL_DOWNSTREAM_QUARANTINE_DIR / LOCAL_DOWNSTREAM_QUARANTINE_KIND / stamp
    )
    moved: list[str] = []
    records: list[dict[str, str]] = []
    for source in (
        local_downstream_root(experiment_id, variant_id, output_root),
        local_downstream_metadata_path(experiment_id, variant_id, output_root),
    ):
        if not source.exists():
            continue
        assert_local_downstream_owned_targets(
            experiment_id, variant_id, [source], output_root,
        )
        target_root.mkdir(parents=True, exist_ok=True)
        destination = target_root / source.name
        os.replace(source, destination)
        moved.append(str(destination))
        records.append({"source": str(source), "target": str(destination)})
    if records:
        recovery_record = target_root / "quarantine_metadata.json"
        _atomic_write_text(recovery_record, _json_document({
            "timestamp_utc": stamp,
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "reason": reason,
            "moves": records,
            "deleted_files": [],
        }))
        moved.append(str(recovery_record))
    return sorted(moved)


def local_downstream_variant_is_reusable(
    experiment_id: str,
    analysis_id: str,
    variant: dict,
    predictor_binding: dict,
    canonical_frame,
    lineage: dict,
    experiments_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    censor_binding: Optional[dict] = None,
) -> tuple[bool, Optional[dict], str]:
    """Whether a previously produced variant downstream may be reused untouched.

    Requires: the metadata to exist with the right schema, the SAME analysis_id
    and status=pass; the predictor metadata hash to be unchanged; every recorded
    artefact to exist with a matching hash; the Step8A feature contract to pass
    again; the static/label invariance to hold again; and -- when the registry
    enables it -- the pre-label exclusion binding to still be present, bound to
    the same gate manifest and reconciled. Anything else is not reusable: a
    partial or drifted downstream is never silently accepted.
    """
    import pandas as pd

    variant_id = variant["variant_id"]
    metadata_path = local_downstream_metadata_path(experiment_id, variant_id, output_root)
    if not metadata_path.is_file():
        return False, None, "no local-downstream metadata"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False, None, "unreadable local-downstream metadata"
    if not isinstance(metadata, dict):
        return False, None, "local-downstream metadata is not a JSON object"
    if metadata.get("schema_version") != LOCAL_DOWNSTREAM_METADATA_SCHEMA:
        return False, metadata, (
            f"metadata schema is {metadata.get('schema_version')!r}"
        )
    if metadata.get("analysis_id") != analysis_id:
        return False, metadata, "analysis_id mismatch"
    if metadata.get("status") != STATUS_PASS:
        return False, metadata, f"previous status is {metadata.get('status')!r}"
    if metadata.get("predictor_metadata_sha256") != predictor_binding["predictor_metadata_sha256"]:
        return False, metadata, "the predictor export metadata hash has changed"
    if metadata.get("predictor_artifact_sha256") != predictor_binding["predictor_artifact_sha256"]:
        return False, metadata, "a predictor raster hash has changed"

    artifacts = metadata.get("artifact_inventory") or []
    if not artifacts:
        return False, metadata, "the recorded artifact inventory is empty"
    for record in artifacts:
        path = Path(str((record or {}).get("path") or ""))
        if not path.is_file():
            return False, metadata, f"missing artefact {record.get('artifact_id')}"
        if sha256_file(path) != record.get("sha256"):
            return False, metadata, f"hash mismatch for {record.get('artifact_id')}"

    dataset = variant_step8a_dataset_path(experiment_id, variant_id, output_root)
    stats = variant_step8a_stats_path(experiment_id, variant_id, output_root)
    if not dataset.is_file() or not stats.is_file():
        return False, metadata, "the variant Step8A dataset or stats file is missing"
    try:
        frame = pd.read_parquet(dataset)
        assert_reference_grid_matches_canonical(
            stats, canonical_step8a_stats_path(experiment_id, experiments_root),
        )
        contract = assert_step8a_feature_contract(frame, canonical_frame, lineage)
        compare_step8a_invariance(
            frame, canonical_frame, contract, contract["key_column"],
        )
    except WindowClosureError as exc:
        return False, metadata, f"Step8A contract/invariance failed again: {exc}"
    except Exception as exc:  # noqa: BLE001 -- an unreadable dataset is not reusable
        return False, metadata, f"the variant Step8A dataset is unusable: {exc}"

    if (censor_binding or {}).get("exclude_pre_label_burns"):
        if not metadata.get("prelabel_exclusion_applied"):
            return False, metadata, (
                "the registry enables exclude_pre_label_burns but the recorded "
                "downstream declares no pre-label exclusion binding"
            )
        if not metadata.get("prelabel_exclusion_accounting_reconciled"):
            return False, metadata, "the recorded pre-label exclusion accounting is not reconciled"
        recorded = (
            (metadata.get("prelabel_exclusion_binding") or {}).get("documents") or {}
        )
        for role in PRELABEL_EXCLUSION_REQUIRED_ROLES:
            expected = (censor_binding["documents"].get(role) or {}).get("sha256")
            if (recorded.get(role) or {}).get("sha256") != expected:
                return False, metadata, (
                    f"the bound pre-label exclusion document '{role}' has changed"
                )
        try:
            assert_prelabel_exclusion_accounting(
                frame, stats, censor_binding, variant_id,
            )
        except WindowClosureError as exc:
            return False, metadata, f"pre-label exclusion accounting failed again: {exc}"
    return True, metadata, "complete and verified"


def run_local_downstream(
    experiment_id: str,
    analysis_id: str,
    base_context: dict,
    canonical: dict,
    variants: Sequence[dict],
    inventory: dict,
    binding: dict,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
    force: bool = False,
    resume: bool = False,
    recover_partial: bool = False,
    engine: Optional[Any] = None,
) -> dict:
    """Run (or reuse) the production downstream chain of every non-canonical variant.

    Variants are processed in increasing shift order and are fully independent:
    a later failure never invalidates, deletes or downgrades an earlier
    variant's verified outputs, and a variant only gets `status=pass` metadata
    once every stage, every artefact, the Step8A feature contract and the
    static/label invariance have all passed.

    The three modes are strictly separated (`resume` and `force` are mutually
    exclusive, see `assert_resume_force_exclusive`):

    * plain  -- builds a variant that has NO downstream yet; refuses to
      overwrite an existing one, reusable or not;
    * resume -- FAIL-CLOSED: reuses a complete, verified status=pass variant
      and nothing else. A partial, missing, failed, drifted or
      contract-breaking downstream stops the run without quarantining,
      moving, deleting or writing anything;
    * force  -- the ONLY mode allowed to quarantine an existing downstream and
      re-produce it. Nothing is ever deleted.

    `engine` is the dependency-injection point; the default is the production
    chain, imported lazily so no dry run or test pulls the heavy raster modules
    in.
    """
    import pandas as pd

    baseline_years = list(canonical["baseline_years"])
    early = nonzero_variants(variants)
    if not early:
        raise WindowClosureError(
            "No non-canonical variant is preregistered, so there is no local "
            "downstream chain to run."
        )
    if list(binding.get("nonzero_variant_ids") or []) != [v["variant_id"] for v in early]:
        raise WindowClosureError(
            f"Plan binding covers {binding.get('nonzero_variant_ids')!r} but "
            f"this run derived {[v['variant_id'] for v in early]!r}."
        )

    # --- Frozen canonical reference, opened READ-ONLY ------------------------
    canonical_dataset = canonical_step8a_path(experiment_id, experiments_root)
    canonical_stats = canonical_step8a_stats_path(experiment_id, experiments_root)
    if not canonical_dataset.is_file():
        raise WindowClosureError(
            f"The frozen canonical Step8A dataset is missing at "
            f"{canonical_dataset}; there is no reference to compare against."
        )
    lineage = step8a_predictor_lineage(experiment_id, experiments_root)
    canonical_frame = pd.read_parquet(canonical_dataset)
    canonical_sha256 = sha256_file(canonical_dataset)
    canonical_stats_sha256 = sha256_file(canonical_stats) if canonical_stats.is_file() else None

    prelabel_summary_path = (
        experiment_root(experiment_id, output_root) / "prelabel_censor" / PRELABEL_SUMMARY_NAME
    )
    prelabel_positive_cell_count = 0
    if prelabel_summary_path.is_file():
        try:
            prelabel_positive_cell_count = int(
                (json.loads(prelabel_summary_path.read_text(encoding="utf-8")) or {})
                .get("prelabel_burn_cell_count") or 0
            )
        except (OSError, ValueError, UnicodeDecodeError):
            prelabel_positive_cell_count = 0

    # The registry-driven pre-label censor policy is resolved and asserted
    # ONCE, before any variant is touched: a missing gate document is a plan
    # error, not a per-variant surprise after the production chain has started.
    censor_binding = assert_prelabel_exclusion_binding(
        prelabel_exclusion_binding(experiment_id, base_context, experiments_root),
        "local-downstream stage entry",
    )

    frozen_before = frozen_hash_map(
        local_downstream_frozen_inputs(
            experiment_id, inventory, variants, output_root, experiments_root,
            censor_binding,
        )
    )

    files_written: list[str] = []
    files_rewritten: list[str] = []
    processed: list[str] = []
    reused_variants: list[str] = []
    completed_variants: list[str] = []
    quarantined: list[str] = []
    variant_reports: dict[str, dict] = {}
    artifacts_produced = 0
    datasets_produced = 0

    for variant in early:
        variant_id = variant["variant_id"]
        predictor_binding = (binding.get("predictor_binding") or {}).get(variant_id)
        if not predictor_binding:
            raise WindowClosureError(
                f"Variant '{variant_id}' has no verified predictor binding; the "
                "local-downstream stage refuses to start."
            )
        metadata_path = local_downstream_metadata_path(experiment_id, variant_id, output_root)
        downstream = local_downstream_root(experiment_id, variant_id, output_root)

        reusable, previous, reason = local_downstream_variant_is_reusable(
            experiment_id, analysis_id, variant, predictor_binding, canonical_frame,
            lineage, experiments_root, output_root, censor_binding,
        )
        if reusable and not force:
            if not resume and not recover_partial:
                raise WindowClosureError(
                    f"Variant '{variant_id}' already has a complete, verified "
                    f"local downstream at {metadata_path}. Refusing to "
                    "overwrite it silently: re-run with resume=True to reuse "
                    "it, or force=True to rebuild it (old outputs are "
                    "quarantined, never deleted)."
                )
            processed.append(variant_id)
            reused_variants.append(variant_id)
            completed_variants.append(variant_id)
            recorded = previous or {}
            files_written += [
                str(record["path"]) for record in (recorded.get("artifact_inventory") or [])
            ] + [str(metadata_path)]
            artifacts_produced += len(recorded.get("artifact_inventory") or [])
            datasets_produced += 1
            variant_reports[variant_id] = {
                "variant_id": variant_id, "reused": True, "reason": reason,
                "artifact_count": len(recorded.get("artifact_inventory") or []),
                "status": STATUS_PASS,
            }
            continue

        # --- `--resume` is FAIL-CLOSED ---------------------------------------
        # It may only ever REUSE a complete, verified, status=pass variant.
        # A partial downstream, a missing/failed metadata, a missing artefact,
        # a hash mismatch or a contract failure stops the run here: nothing is
        # quarantined, moved, deleted, rebuilt or written, and no production
        # helper is reached. Quarantining and re-producing is exclusively
        # `--force`'s authority.
        if resume:
            raise WindowClosureError(
                f"--resume cannot reuse variant '{variant_id}': {reason}. "
                "--resume only ever REUSES a complete, verified status=pass "
                "local downstream; it never rebuilds one. Nothing was "
                f"quarantined, moved, deleted or written at {downstream}. "
                "Inspect the outputs, then either run the stage without "
                "--resume on a variant that has none, or re-run with "
                "force=True to rebuild it (old outputs are quarantined, never "
                "deleted)."
            )

        if not force and not recover_partial and (previous is not None or downstream.exists()):
            raise WindowClosureError(
                f"Variant '{variant_id}' has an existing but NOT reusable local "
                f"downstream ({reason}) at {downstream}. Refusing to overwrite "
                "it silently: inspect it, then re-run with force=True (old "
                "outputs are quarantined, never deleted)."
            )

        # Rebuilding (force, or a clean namespace): quarantine whatever is
        # there, never delete it.
        quarantined += _quarantine_local_downstream(
            experiment_id, variant_id, output_root,
            reason=(
                f"outer stage resume recovery: local-downstream is not reusable ({reason})"
                if recover_partial else "explicit force rebuild"
            ),
        )

        # --- Production inputs, laid out under the production names ---------
        bindings = production_input_bindings(
            experiment_id, variant, base_context, predictor_binding["artifacts"],
            inventory, output_root, censor_binding,
        )
        assert_local_downstream_owned_targets(
            experiment_id, variant_id,
            [Path(entry["target"]) for entry in bindings if entry["target"] is not None],
            output_root,
        )
        # Resolved BEFORE anything is materialised and before any production
        # helper is imported, so a missing baseline role or a leaked support
        # raster fails with nothing created.
        baseline_binding = resolve_baseline_lst_binding(
            variant_id, bindings, baseline_years,
        )
        materialized = materialize_local_downstream_inputs(bindings)
        assert_materialized_values_unchanged(materialized)

        # --- Variant context and the production chain -----------------------
        variant_context = build_local_downstream_variant_context(
            experiment_id, variant, base_context, analysis_id,
            baseline_binding, output_root,
        )
        if variant_context.get("window_closure_variant_mode") is not True:
            raise WindowClosureError(
                f"Variant '{variant_id}' context does not declare the "
                "window-closure variant mode, so Step8A would validate its "
                "dates against the CANONICAL window. Refusing to run."
            )
        if not variant_context.get(STEP5_EXPLICIT_BASELINE_PATHS_KEY):
            raise WindowClosureError(
                f"Variant '{variant_id}' context carries no explicit baseline "
                "LST binding, so production Step5 would fall back to a "
                "directory scan. Refusing to run."
            )
        active_engine = engine if engine is not None else production_local_downstream_engine
        plan = local_downstream_variant_plan(
            experiment_id, variant, baseline_years, output_root, experiments_root,
        )
        outcome = active_engine(variant_context, variant, plan) or {}
        stages_run = list(outcome.get("stages_run") or PRODUCTION_STAGE_SEQUENCE)
        if stages_run != list(PRODUCTION_STAGE_SEQUENCE):
            raise WindowClosureError(
                f"Variant '{variant_id}' ran production stages {stages_run}, "
                f"expected the deterministic sequence "
                f"{list(PRODUCTION_STAGE_SEQUENCE)}."
            )

        # --- Artefacts -------------------------------------------------------
        artifact_inventory = build_local_downstream_artifact_inventory(
            experiment_id, variant_id, stages_run, baseline_years, output_root,
        )
        dataset_path = variant_step8a_dataset_path(experiment_id, variant_id, output_root)
        stats_path = variant_step8a_stats_path(experiment_id, variant_id, output_root)
        if not dataset_path.is_file():
            raise WindowClosureError(
                f"Variant '{variant_id}' produced no Step8A modelling dataset "
                f"at {dataset_path}."
            )
        grid_check = assert_reference_grid_matches_canonical(stats_path, canonical_stats)

        # --- Step8A contract and invariance ----------------------------------
        variant_frame = pd.read_parquet(dataset_path)
        contract = assert_step8a_feature_contract(variant_frame, canonical_frame, lineage)
        invariance = compare_step8a_invariance(
            variant_frame, canonical_frame, contract, contract["key_column"],
        )
        # The censor accounting is reconciled against the BOUND manifest and
        # the production stats counters before the variant may be published.
        censor_accounting = assert_prelabel_exclusion_accounting(
            variant_frame, stats_path, censor_binding, variant_id,
        )
        # The fixed month-filter clipping is read back from this variant's own
        # export provenance; it is never recomputed or defaulted here.
        modis_clipping = modis_clipping_from_predictor_metadata(
            read_predictor_metadata(experiment_id, variant_id, output_root),
            variant_id,
            str(predictor_metadata_path(experiment_id, variant_id, output_root)),
        )

        frozen_after = frozen_hash_map(
            local_downstream_frozen_inputs(
                experiment_id, inventory, variants, output_root, experiments_root,
                censor_binding,
            )
        )
        assert_frozen_hashes_unchanged(
            frozen_before, frozen_after,
            f"while running the local downstream of variant '{variant_id}'",
        )

        metadata = build_local_downstream_metadata(
            experiment_id, analysis_id, variant, baseline_years, variant_context,
            predictor_binding, materialized, baseline_binding, stages_run,
            artifact_inventory, contract, invariance, grid_check, variant_frame,
            canonical_dataset, canonical_sha256, canonical_stats, canonical_stats_sha256,
            lineage, prelabel_positive_cell_count, frozen_before, frozen_after,
            output_root, censor_binding, censor_accounting, modis_clipping,
        )
        _atomic_write_text(metadata_path, _json_document(metadata))

        processed.append(variant_id)
        completed_variants.append(variant_id)
        produced = [record["path"] for record in artifact_inventory]
        produced += [
            record["target"] for record in materialized if record.get("materialized")
        ]
        files_written += produced + [str(metadata_path)]
        files_rewritten += produced + [str(metadata_path)]
        artifacts_produced += len(artifact_inventory)
        datasets_produced += 1
        variant_reports[variant_id] = {
            "variant_id": variant_id,
            "reused": False,
            "artifact_count": len(artifact_inventory),
            "production_stage_sequence": list(stages_run),
            "step8a_dataset_path": str(dataset_path),
            "step8a_dataset_sha256": sha256_file(dataset_path),
            "variant_row_count": invariance["variant_row_count"],
            "overlap_row_count": invariance["overlap_row_count"],
            "metadata_path": str(metadata_path),
            "status": STATUS_PASS,
        }

    frozen_after = frozen_hash_map(
        local_downstream_frozen_inputs(
            experiment_id, inventory, variants, output_root, experiments_root,
            censor_binding,
        )
    )
    assert_frozen_hashes_unchanged(
        frozen_before, frozen_after, "while running the local downstream chain",
    )
    return {
        "files_written": sorted(set(files_written)),
        "files_rewritten": sorted(set(files_rewritten)),
        "processed_variants": processed,
        "reused_variants": reused_variants,
        "completed_variants": completed_variants,
        "quarantined_artifacts": sorted(quarantined),
        "downstream_artifacts_produced": artifacts_produced,
        "step8a_datasets_produced": datasets_produced,
        "reused": bool(processed) and processed == reused_variants,
        "variant_reports": variant_reports,
        "frozen_input_sha256_before": frozen_before,
        "frozen_input_sha256_after": frozen_after,
        "canonical_downstream_attempted": False,
        "common_cohort_created": False,
        "gee_query_run": False,
        "gee_export_run": False,
        **LOCAL_DOWNSTREAM_MODEL_SEMANTICS,
    }


def build_local_downstream_metadata(
    experiment_id: str,
    analysis_id: str,
    variant: dict,
    baseline_years: Sequence[int],
    variant_context: dict,
    predictor_binding: dict,
    materialized: Sequence[dict],
    baseline_binding: dict,
    stages_run: Sequence[str],
    artifact_inventory: Sequence[dict],
    contract: dict,
    invariance: dict,
    grid_check: dict,
    variant_frame,
    canonical_dataset: Path,
    canonical_sha256: str,
    canonical_stats: Path,
    canonical_stats_sha256: Optional[str],
    lineage: dict,
    prelabel_positive_cell_count: int,
    frozen_before: dict,
    frozen_after: dict,
    output_root: Optional[Path] = None,
    censor_binding: Optional[dict] = None,
    censor_accounting: Optional[dict] = None,
    modis_clipping: Optional[dict] = None,
) -> dict:
    """The per-variant local-downstream record. Deterministic and self-describing."""
    variant_id = variant["variant_id"]
    dataset_path = variant_step8a_dataset_path(experiment_id, variant_id, output_root)
    stats_path = variant_step8a_stats_path(experiment_id, variant_id, output_root)
    burned = int((variant_frame["burned"].astype(int) == 1).sum())
    unburned = int((variant_frame["burned"].astype(int) == 0).sum())
    return {
        "schema_version": LOCAL_DOWNSTREAM_METADATA_SCHEMA,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "variant_id": variant_id,
        "shift_days": int(variant["shift_days"]),
        "predictor_start_date": variant["predictor_start_date"],
        "predictor_end_date": variant["predictor_end_date"],
        "lead_days": int(variant["lead_days"]),
        "label_start_date": variant["label_start_date"],
        "label_end_date": variant["label_end_date"],
        "baseline_years": [int(year) for year in baseline_years],

        "predictor_metadata_path": predictor_binding["predictor_metadata_path"],
        "predictor_metadata_sha256": predictor_binding["predictor_metadata_sha256"],
        "predictor_artifact_count": predictor_binding["predictor_artifact_count"],
        "predictor_logical_role_count": predictor_binding["predictor_logical_role_count"],
        "predictor_artifact_sha256": dict(predictor_binding["predictor_artifact_sha256"]),

        "production_stage_sequence": list(stages_run),
        "production_helpers": {
            stage: f"{PRODUCTION_STAGE_HELPERS[stage]['module']}."
                   f"{PRODUCTION_STAGE_HELPERS[stage]['function']}"
            for stage in stages_run
        },
        "production_stage_input_roles": production_stage_input_roles(
            baseline_years,
            [
                role for role in PRELABEL_EXCLUSION_REQUIRED_ROLES
                if (censor_binding or {}).get("exclude_pre_label_burns")
            ],
        ),
        "production_policy": {
            "scientific_calculation_unchanged": True,
            "new_formula_introduced": False,
            "new_feature_introduced": False,
            "new_reducer_introduced": False,
            "new_imputation_introduced": False,
            "input_materialisation": (
                "byte-verbatim copy, plus band concatenation of the two "
                "single-band exports that production writes as one two-band "
                "current-window image; no resampling, reprojection or "
                "arithmetic"
            ),
            "step8a_feature_registry": "src.step8b_train_baseline_vs_thermal_model",
            "step8a_predictor_lineage_source": lineage["source"],
            "timing_derived_predictors": list(lineage["timing_derived_predictors"]),
            "static_predictors": list(lineage["static_predictors"]),
        },
        "variant_context_summary": {
            key: str(variant_context[key])
            for key in LOCAL_DOWNSTREAM_CONTEXT_PATH_KEYS
            if variant_context.get(key) is not None
        },
        "materialized_inputs": list(materialized),
        "materialized_input_count": sum(
            1 for record in materialized if record.get("materialized")
        ),

        # Step5's baseline stack is pinned to the hash-verified predictor
        # inventory; the production directory-scan fallback never runs here.
        "baseline_binding_source": baseline_binding["baseline_binding_source"],
        "baseline_directory_scan_used": baseline_binding["baseline_directory_scan_used"],
        "baseline_lst_binding": list(baseline_binding["records"]),
        "baseline_lst_paths": [str(path) for path in baseline_binding["paths"]],

        "artifact_inventory": list(artifact_inventory),
        "artifact_count": len(artifact_inventory),
        "artifact_sha256": {
            record["artifact_id"]: record["sha256"] for record in artifact_inventory
        },

        "step8a_dataset_path": str(dataset_path),
        "step8a_dataset_sha256": sha256_file(dataset_path),
        "step8a_stats_path": str(stats_path),
        "step8a_stats_sha256": sha256_file(stats_path) if stats_path.is_file() else None,

        "canonical_step8a_path": str(canonical_dataset),
        "canonical_step8a_sha256": canonical_sha256,
        "canonical_step8a_stats_path": str(canonical_stats),
        "canonical_step8a_stats_sha256": canonical_stats_sha256,
        "canonical_feature_contract_sha256": contract["canonical_feature_contract_sha256"],

        "feature_contract_passed": contract["feature_contract_passed"],
        "legacy_canonical_audit_columns_absent":
            contract["legacy_canonical_audit_columns_absent"],
        "optional_audit_columns_present_in_variant":
            contract["optional_audit_columns_present_in_variant"],
        "optional_audit_contract_passed": contract["optional_audit_contract_passed"],
        "model_feature_registry_unchanged": contract["model_feature_registry_unchanged"],
        "canonical_bytes_unchanged": contract["canonical_bytes_unchanged"],
        "key_uniqueness_passed": contract["key_uniqueness_passed"],
        "key_column": contract["key_column"],
        # Which literal dtype differences occurred and which were accepted as
        # the SAME discrete production code under a different pandas
        # representation. Empty lists mean the dtypes matched exactly.
        "semantic_dtype_contract": contract["semantic_dtype_contract"],
        "literal_dtype_differences": contract["literal_dtype_differences"],
        "accepted_semantic_dtype_compatibilities":
            contract["accepted_semantic_dtype_compatibilities"],
        "model_feature_columns_in_order": list(
            contract["canonical_contract"]["model_feature_columns_in_order"]
        ),
        "label_columns": list(contract["canonical_contract"]["label_columns"]),
        "population_columns": list(contract["canonical_contract"]["population_columns"]),
        "invariant_columns": list(invariance["compared_invariant_columns"]),
        "timing_derived_columns": list(
            invariance["timing_derived_columns_allowed_to_change"]
        ),
        "static_invariance_passed": invariance["static_invariance_passed"],
        "label_invariance_passed": invariance["label_invariance_passed"],
        **grid_check,

        "variant_row_count": invariance["variant_row_count"],
        "canonical_row_count": invariance["canonical_row_count"],
        "overlap_row_count": invariance["overlap_row_count"],
        "variant_only_row_count": invariance["variant_only_row_count"],
        "canonical_only_row_count": invariance["canonical_only_row_count"],
        "row_count_difference_is_not_a_failure": True,
        "primary_population": PRIMARY_POPULATION,
        "primary_population_row_count": (
            int(variant_frame[PRIMARY_POPULATION].astype(bool).sum())
            if PRIMARY_POPULATION in variant_frame.columns else None
        ),
        "primary_population_filter_applied": False,
        "burned_count": burned,
        "unburned_count": unburned,

        "prelabel_used_as_predictor": False,
        "prelabel_positive_cell_count": int(prelabel_positive_cell_count),
        "prelabel_role": "censoring_provenance_only",
        "common_cohort_created": False,

        # --- Registry-driven pre-label EXCLUSION contract -------------------
        # The policy, the bound gate documents and the reconciled per-variant
        # counts, so a reader never has to infer whether the exclusion was
        # applied or how many cells it removed.
        "prelabel_exclusion_binding": dict(censor_binding or {}),
        "prelabel_exclusion_accounting": dict(censor_accounting or {}),
        "prelabel_exclusion_applied": bool(
            (censor_binding or {}).get("exclude_pre_label_burns", False)
        ),
        "prelabel_exclusion_binding_ready": bool(
            (censor_binding or {}).get("binding_ready", False)
        ),
        "prelabel_exclusion_accounting_reconciled": bool(
            (censor_accounting or {}).get("accounting_reconciled", False)
        ),

        # --- Fixed MODIS month-filter clipping, read from export provenance --
        "modis_month_filter_clipping": dict(modis_clipping or {}),
        "modis_clipped_day_count": (
            None if not modis_clipping else int(modis_clipping["clipped_day_count"])
        ),
        "modis_clipping_provenance_source": (
            None if not modis_clipping else modis_clipping["source"]
        ),

        "all_paths_inside_variant_namespace": True,
        "canonical_downstream_attempted": False,
        "canonical_outputs_modified": False,

        "gee_queries_run": False,
        "gee_exports_run": False,
        # The Step7C downscaling random forest IS trained here; the fire-risk
        # baseline/thermal model is NOT (that stage is still locked).
        **LOCAL_DOWNSTREAM_MODEL_SEMANTICS,

        "frozen_input_sha256_before": dict(frozen_before),
        "frozen_input_sha256_after": dict(frozen_after),
        "frozen_hashes_unchanged": frozen_before == frozen_after,
        "limitations": list(LOCAL_DOWNSTREAM_LIMITATIONS),
        "status": STATUS_PASS,
    }


# =============================================================================
# Actual MODEL stage
#
# Fits the PRODUCTION baseline / baseline+thermal fire-risk models of every
# variant on ONE exact common cohort, with ONE shared spatial-fold assignment,
# and quantifies the closure effect with ONE paired spatial-block bootstrap.
#
# Nothing scientific is defined here. The feature registry, the pipeline, the
# model family, the hyper-parameters, the seeds, the fold construction, the
# metric definitions and the bootstrap primitives all come from
# `src.step8b_train_baseline_vs_thermal_model` and
# `src.step8c_spatial_block_bootstrap_uncertainty`, driven by frozen
# `core.config` values. No calibration, transfer adjustment, or alternative
# model procedure is applied in this stage; the machine-readable statement of
# that is `model_configuration.calibration` / `.adaptation`, both null.
#
# The canonical variant is READ (its frozen Step8A dataset); no Step5-Step8A
# product and no Step7C downscaling model is re-produced here.
# =============================================================================
MODEL_STAGE = "model"
MODEL_METADATA_SCHEMA = "window_closure_model.v1"
MODEL_ROOT_DIR = "model"
MODEL_METADATA_NAME = "model_stage_metadata.json"
MODEL_STAGING_DIR = "_model_staging"
MODEL_QUARANTINE_KIND = "model"

MODEL_FAMILIES: tuple[str, ...] = ("baseline", "thermal")
#: Metrics reported for every variant/model family. Names and definitions come
#: from `step8b.compute_binary_metrics` / `step8c.compute_metrics`.
MODEL_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "brier")
#: Thermal contribution, as a RAW `thermal - baseline` delta per metric.
MODEL_CONTRIBUTION_METRICS: tuple[str, ...] = (
    "delta_roc_auc", "delta_pr_auc", "delta_brier",
)
#: Brier is a loss: a NEGATIVE raw delta means the thermal model is better.
#: Stated explicitly instead of silently re-orienting the sign.
BRIER_SIGN_CONVENTION = (
    "delta_brier = thermal - baseline (RAW). Brier is a LOSS, so a NEGATIVE "
    "delta_brier means the thermal model scored BETTER. The sign is never "
    "flipped to an improvement orientation in this stage."
)
METRIC_SIGN_CONVENTIONS: dict[str, str] = {
    "roc_auc": "higher is better; delta_roc_auc = thermal - baseline",
    "pr_auc": "higher is better; delta_pr_auc = thermal - baseline",
    "brier": BRIER_SIGN_CONVENTION,
}

COMPARISON_THERMAL_CONTRIBUTION = "thermal_contribution_within_variant"
COMPARISON_CLOSURE_CHANGE = "closure_change_within_model_family"
COMPARISON_CONTRIBUTION_CHANGE = "thermal_contribution_change"

MODEL_STAGE_SEMANTICS: dict[str, Any] = {
    "model_fit": True,
    "fire_risk_model_fit": True,
    "fire_risk_model_stage_run": True,
    # The Step7C downscaling model is an UPSTREAM artefact of the
    # local-downstream stage. This stage references it and never refits it.
    "downscaling_model_fit": False,
    "downscaling_model_refit": False,
    "gee_queries_run": False,
    "gee_exports_run": False,
    "common_cohort_created": True,
    "bootstrap_run": True,
    "compare_run": False,
}
MODEL_DRY_RUN_SEMANTICS: dict[str, Any] = {
    "model_fit": False,
    "fire_risk_model_fit": False,
    "fire_risk_model_stage_run": False,
    "downscaling_model_fit": False,
    "downscaling_model_refit": False,
    "gee_queries_run": False,
    "gee_exports_run": False,
    "common_cohort_created": False,
    "bootstrap_run": False,
    "compare_run": False,
    "fire_risk_model_fit_planned": True,
    "common_cohort_creation_planned": True,
    "shared_folds_planned": True,
    "paired_spatial_block_bootstrap_planned": True,
    "compare_planned": False,
}

MODEL_LIMITATIONS: tuple[str, ...] = (
    "Every model, hyper-parameter, seed, fold rule and metric is the "
    "production one (src.step8b_train_baseline_vs_thermal_model, "
    "src.step8c_spatial_block_bootstrap_uncertainty, core.config). No "
    "calibration, transfer adjustment, or alternative model procedure was "
    "applied. The machine-readable statement of this is "
    "model_configuration.calibration = null and "
    "model_configuration.adaptation = null; the prose deliberately does not "
    "enumerate method names, so a text scan of this record cannot be "
    "misread as evidence that such a method was used.",
    "All six evaluations (3 variants x 2 model families) run on ONE exact "
    "common cohort with ONE shared spatial-fold assignment, so prevalence, "
    "cell set and fold membership are held fixed and every change is "
    "attributable to predictor timing alone.",
    "The paired spatial-block bootstrap resamples BLOCKS and re-scores the "
    "already-computed out-of-fold predictions; it never refits a model, so "
    "the intervals describe sampling uncertainty of the metric, not of the "
    "training procedure.",
    "A confidence interval that includes zero is reported as exactly that and "
    "leaves directional uncertainty unresolved.",
    "delta_brier is a RAW thermal-minus-baseline difference of a LOSS, so a "
    "negative value favours the thermal model.",
)


# --- Paths --------------------------------------------------------------------
def model_root(experiment_id: str, output_root: Optional[Path] = None) -> Path:
    return experiment_root(experiment_id, output_root) / MODEL_ROOT_DIR


def model_staging_root(experiment_id: str, output_root: Optional[Path] = None) -> Path:
    """Transient build directory, on the SAME filesystem as `model/`.

    Everything is produced here and promoted with a single `os.replace`, so a
    failure can never leave a half-written `model/` tree behind.
    """
    return experiment_root(experiment_id, output_root) / MODEL_STAGING_DIR


def model_metadata_path(experiment_id: str, output_root: Optional[Path] = None) -> Path:
    return model_root(experiment_id, output_root) / MODEL_METADATA_NAME


MODEL_STAGE_OWNED_NAMES: tuple[str, ...] = (MODEL_ROOT_DIR,)


def model_stage_owned_paths(
    experiment_id: str, output_root: Optional[Path] = None,
) -> list[Path]:
    root = experiment_root(experiment_id, output_root)
    return [root / name for name in MODEL_STAGE_OWNED_NAMES]


def model_relative_layout() -> dict[str, str]:
    """Every file the stage owns, relative to `model/`. Single source of truth."""
    layout = {
        "common_cohort": "common_cohort/common_cohort.parquet",
        "common_cohort_metadata": "common_cohort/common_cohort_metadata.json",
        "shared_folds": "shared_folds/shared_spatial_folds.parquet",
        "shared_folds_metadata": "shared_folds/shared_spatial_folds_metadata.json",
        "point_metrics_csv": "metrics/point_metrics.csv",
        "point_metrics_json": "metrics/point_metrics.json",
        "thermal_contributions_csv": "metrics/thermal_contributions.csv",
        "bootstrap_replicates": "bootstrap/paired_bootstrap_replicates.parquet",
        "bootstrap_summary_csv": "bootstrap/paired_bootstrap_summary.csv",
        "bootstrap_summary_json": "bootstrap/paired_bootstrap_summary.json",
        "metadata": MODEL_METADATA_NAME,
    }
    return dict(sorted(layout.items()))


def model_variant_oof_relpath(variant_id: str, family: str) -> str:
    return f"variants/{variant_id}/{family}/oof_predictions.parquet"


def model_variant_metrics_relpath(variant_id: str, family: str) -> str:
    return f"variants/{variant_id}/{family}/fold_metrics.csv"


def snapshot_model_state(
    experiment_id: str, output_root: Optional[Path] = None,
) -> dict:
    """Read-only inventory of the model stage-owned paths that already exist.

    Same shape and guarantees as the local-downstream snapshot: no mkdir, no
    write, no move, no mtime change.
    """
    experiment = experiment_root(experiment_id, output_root)
    directories: set[str] = set()
    files: dict[str, dict] = {}
    for target in model_stage_owned_paths(experiment_id, output_root):
        if not target.exists():
            continue
        if target.is_dir():
            directories.add(_relative_label(target, experiment))
            for path in sorted(target.rglob("*")):
                label = _relative_label(path, experiment)
                if path.is_dir():
                    directories.add(label)
                elif path.is_file():
                    files[label] = {
                        "relative_path": label, "path": str(path),
                        "bytes": int(path.stat().st_size),
                        "sha256": sha256_file(path),
                    }
        elif target.is_file():
            label = _relative_label(target, experiment)
            files[label] = {
                "relative_path": label, "path": str(target),
                "bytes": int(target.stat().st_size), "sha256": sha256_file(target),
            }
    content_view = {
        "directories": sorted(directories),
        "files": {
            label: {"bytes": record["bytes"], "sha256": record["sha256"]}
            for label, record in sorted(files.items())
        },
    }
    return {
        "experiment_root": str(experiment),
        "stage_owned_names": list(MODEL_STAGE_OWNED_NAMES),
        "stage_owned_roots": {
            MODEL_ROOT_DIR: str(model_root(experiment_id, output_root)),
        },
        "directories": sorted(directories),
        "directory_count": len(directories),
        "files": dict(sorted(files.items())),
        "file_count": len(files),
        "digest": sha256_bytes(canonical_json(content_view).encode("utf-8")),
    }


# --- Frozen configuration (never re-chosen here) ------------------------------
def model_frozen_configuration() -> dict:
    """Every frozen knob this stage runs on, read from production config.

    Fails closed if a required bootstrap/fold parameter is absent instead of
    substituting a silent default.
    """
    import core.config as config

    required = (
        "STEP8B_N_SPLITS", "STEP8B_RANDOM_SEED", "STEP8B_SPATIAL_BLOCK_SIZE_CELLS",
        "STEP8B_MIN_POSITIVES_PER_POPULATION",
        "STEP8C_N_BOOTSTRAP", "STEP8C_RANDOM_SEED",
        "STEP8C_CI_LOWER", "STEP8C_CI_UPPER",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise WindowClosureError(
            f"Frozen model/bootstrap configuration is incomplete: {missing} is "
            "not defined in core.config. This stage refuses to substitute a "
            "default for a preregistered parameter."
        )
    return {
        "model": PRIMARY_MODEL,
        "primary_population": PRIMARY_POPULATION,
        "n_splits": int(config.STEP8B_N_SPLITS),
        "fold_random_seed": int(config.STEP8B_RANDOM_SEED),
        "spatial_block_size_cells": int(config.STEP8B_SPATIAL_BLOCK_SIZE_CELLS),
        "min_positives": int(config.STEP8B_MIN_POSITIVES_PER_POPULATION),
        "n_bootstrap": int(config.STEP8C_N_BOOTSTRAP),
        "bootstrap_seed": int(config.STEP8C_RANDOM_SEED),
        "ci_lower_percentile": float(config.STEP8C_CI_LOWER),
        "ci_upper_percentile": float(config.STEP8C_CI_UPPER),
        "confidence_level": float(config.STEP8C_CI_UPPER) - float(config.STEP8C_CI_LOWER),
        # The project's existing minimum-valid-replicate guard is
        # `step8c.summarize_bootstrap`, which declares a bootstrap unavailable
        # when no replicate succeeded. It is reused rather than replaced by a
        # newly invented threshold.
        "minimum_valid_replicates": 1,
        "minimum_valid_replicates_source": (
            "src.step8c_spatial_block_bootstrap_uncertainty.summarize_bootstrap "
            "(available = n_bootstrap_successful > 0)"
        ),
        "source": "core.config (frozen)",
        "strict_folds": True,
        "calibration": None,
        "adaptation": None,
    }


def model_feature_registry() -> dict:
    """The canonical fire-risk feature contract, imported, never re-declared."""
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
        THERMAL_FEATURES, THERMAL_MODEL_FEATURES,
    )

    return {
        "baseline_features_in_order": list(BASELINE_FEATURES),
        "thermal_features_in_order": list(THERMAL_FEATURES),
        "thermal_model_features_in_order": list(THERMAL_MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "feature_union": sorted(set(BASELINE_FEATURES) | set(THERMAL_MODEL_FEATURES)),
        "source": "src.step8b_train_baseline_vs_thermal_model",
    }


# --- Frozen input binding -----------------------------------------------------
LOCAL_DOWNSTREAM_REQUIRED_FLAGS: dict[str, bool] = {
    "feature_contract_passed": True,
    "static_invariance_passed": True,
    "label_invariance_passed": True,
    "key_uniqueness_passed": True,
    "frozen_hashes_unchanged": True,
    "canonical_outputs_modified": False,
    "canonical_downstream_attempted": False,
    "prelabel_used_as_predictor": False,
}


def read_local_downstream_metadata(
    experiment_id: str, variant_id: str, output_root: Optional[Path] = None,
) -> dict:
    path = local_downstream_metadata_path(experiment_id, variant_id, output_root)
    if not path.is_file():
        raise WindowClosureError(
            f"Local-downstream metadata for variant '{variant_id}' is missing "
            f"at {path}. The model stage binds to a completed local downstream; "
            "run --from-stage local-downstream --to-stage local-downstream "
            "first. Nothing was created."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise WindowClosureError(
            f"Local-downstream metadata for variant '{variant_id}' at {path} "
            f"is unreadable: {exc}. Nothing was created."
        ) from exc
    if not isinstance(payload, dict):
        raise WindowClosureError(
            f"Local-downstream metadata for variant '{variant_id}' is not a "
            "JSON object."
        )
    return payload


def assert_model_binding(
    experiment_id: str,
    analysis_id: str,
    shifts: Sequence[int],
    canonical: dict,
    variants: Sequence[dict],
    censor: dict,
    inventory: dict,
    planned_paths: dict[str, str],
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
) -> dict:
    """Bind to every completed upstream stage. Read-only, before any write.

    Resolves the three Step8A datasets the models will be fitted on -- the
    frozen canonical one and the two shifted ones -- and verifies each against
    the hash the producing stage recorded. No path and no hash is guessed.
    """
    binding = assert_local_downstream_binding(
        experiment_id, analysis_id, shifts, canonical, variants, censor,
        inventory, planned_paths, output_root,
    )

    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise WindowClosureError(
                f"Model binding failed: {message} No directory was created and "
                "nothing was written."
            )

    canonical_dataset = canonical_step8a_path(experiment_id, experiments_root)
    canonical_stats = canonical_step8a_stats_path(experiment_id, experiments_root)
    _require(canonical_dataset.is_file(),
             f"the frozen canonical Step8A dataset is missing at {canonical_dataset}.")
    _require(canonical_stats.is_file(),
             f"the canonical Step8A stats are missing at {canonical_stats}.")
    canonical_sha256 = sha256_file(canonical_dataset)

    datasets: dict[str, dict] = {
        CANONICAL_VARIANT_ID: {
            "variant_id": CANONICAL_VARIANT_ID,
            "is_canonical": True,
            "dataset_path": str(canonical_dataset),
            "dataset_sha256": canonical_sha256,
            "stats_path": str(canonical_stats),
            "stats_sha256": sha256_file(canonical_stats),
            "source": "frozen canonical production Step8A (read-only)",
        },
    }
    pinned_canonical: set = set()

    for variant in nonzero_variants(variants):
        variant_id = variant["variant_id"]
        metadata = read_local_downstream_metadata(experiment_id, variant_id, output_root)
        metadata_path = local_downstream_metadata_path(
            experiment_id, variant_id, output_root,
        )
        _require(
            metadata.get("schema_version") == LOCAL_DOWNSTREAM_METADATA_SCHEMA,
            f"'{variant_id}' local-downstream schema is "
            f"{metadata.get('schema_version')!r}.",
        )
        _require(metadata.get("status") == STATUS_PASS,
                 f"'{variant_id}' local-downstream status is "
                 f"{metadata.get('status')!r}, not {STATUS_PASS!r}.")
        _require(metadata.get("analysis_id") == analysis_id,
                 f"'{variant_id}' local-downstream analysis_id is "
                 f"{metadata.get('analysis_id')!r}, expected {analysis_id!r}.")
        _require(metadata.get("experiment_id") == experiment_id,
                 f"'{variant_id}' local-downstream experiment_id is "
                 f"{metadata.get('experiment_id')!r}.")
        _require(metadata.get("variant_id") == variant_id,
                 f"'{variant_id}' local-downstream metadata names variant "
                 f"{metadata.get('variant_id')!r}.")
        for flag, expected in LOCAL_DOWNSTREAM_REQUIRED_FLAGS.items():
            _require(
                metadata.get(flag) is expected,
                f"'{variant_id}' local-downstream {flag}={metadata.get(flag)!r}, "
                f"expected {expected!r}.",
            )

        dataset_path = Path(str(metadata.get("step8a_dataset_path") or ""))
        expected_path = variant_step8a_dataset_path(experiment_id, variant_id, output_root)
        _require(
            dataset_path.resolve() == expected_path.resolve(),
            f"'{variant_id}' records Step8A at {dataset_path}, expected "
            f"{expected_path}.",
        )
        _require(dataset_path.is_file(),
                 f"'{variant_id}' Step8A dataset is missing at {dataset_path}.")
        digest = sha256_file(dataset_path)
        _require(
            digest == metadata.get("step8a_dataset_sha256"),
            f"'{variant_id}' Step8A dataset hashes {digest} but its "
            f"local-downstream metadata recorded "
            f"{metadata.get('step8a_dataset_sha256')!r}.",
        )
        stats_path = Path(str(metadata.get("step8a_stats_path") or ""))
        _require(stats_path.is_file(),
                 f"'{variant_id}' Step8A stats are missing at {stats_path}.")
        _require(
            sha256_file(stats_path) == metadata.get("step8a_stats_sha256"),
            f"'{variant_id}' Step8A stats hash differs from its metadata.",
        )
        pinned_canonical.add(metadata.get("canonical_step8a_sha256"))

        datasets[variant_id] = {
            "variant_id": variant_id,
            "is_canonical": False,
            "shift_days": int(variant["shift_days"]),
            "dataset_path": str(dataset_path),
            "dataset_sha256": digest,
            "stats_path": str(stats_path),
            "stats_sha256": sha256_file(stats_path),
            "local_downstream_metadata_path": str(metadata_path),
            "local_downstream_metadata_sha256": sha256_file(metadata_path),
            "source": "window-closure local downstream (status=pass)",
        }

    _require(
        pinned_canonical == {canonical_sha256},
        f"the local-downstream metadata pinned canonical Step8A hash(es) "
        f"{sorted(str(v) for v in pinned_canonical)}, but the frozen dataset "
        f"hashes {canonical_sha256}.",
    )
    _require(
        len(datasets) == 1 + len(nonzero_variants(variants)),
        f"expected {1 + len(nonzero_variants(variants))} bound Step8A datasets, "
        f"got {len(datasets)}.",
    )
    return {
        **binding,
        "bound_to_local_downstream": True,
        "model_datasets": datasets,
        "canonical_step8a_sha256": canonical_sha256,
    }


def model_frozen_inputs(
    experiment_id: str, inventory: dict, variants: Sequence[dict],
    output_root: Optional[Path] = None, experiments_root: Optional[Path] = None,
) -> dict:
    """Everything the model stage must not disturb, hashed.

    Extends the local-downstream frozen set with the shifted Step8A datasets,
    their stats and the local-downstream metadata documents themselves.
    """
    extended = dict(local_downstream_frozen_inputs(
        experiment_id, inventory, variants, output_root, experiments_root,
    ))
    for variant in nonzero_variants(variants):
        variant_id = variant["variant_id"]
        for role, path in (
            (f"local_downstream_metadata__{variant_id}",
             local_downstream_metadata_path(experiment_id, variant_id, output_root)),
            (f"variant_step8a__{variant_id}",
             variant_step8a_dataset_path(experiment_id, variant_id, output_root)),
            (f"variant_step8a_stats__{variant_id}",
             variant_step8a_stats_path(experiment_id, variant_id, output_root)),
        ):
            extended[role] = {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
    return extended


# --- Shared pre-label censor --------------------------------------------------
def prelabel_censored_cells(raster_path: Path) -> dict:
    """500 m cells that carry ANY positive pre-label BurnDate pixel.

    Reuses the production Step8A 500 m grid contract verbatim --
    `compute_block_size_pixels`, `make_tile_grid` and `compute_cell_identity`,
    the single source of truth for cell identity across the project -- so a
    censored `cell_id` names the same physical block Step8A named. A cell is
    censored when ANY constituent source pixel is positive: no majority rule
    and no prevalence threshold is applied. A zero-positive raster is a valid
    outcome and the censor is still recorded as applied.
    """
    import numpy as np
    import rasterio

    from core.utils.tiling import make_tile_grid
    from src.step8a_prepare_500m_modeling_dataset import (
        compute_block_size_pixels, compute_cell_identity,
    )
    from rasterio.windows import Window

    if not raster_path.is_file():
        raise WindowClosureError(
            f"The shared pre-label censoring raster is missing at {raster_path}."
        )
    block_size = compute_block_size_pixels()
    censored: set[str] = set()
    positive_pixels = 0
    with rasterio.open(raster_path) as dataset:
        # `make_tile_grid` returns a GRID DICT, not an iterable of tiles: the
        # per-tile records live under its "tiles" key. Production Step8A
        # traverses it exactly this way (`tiles = tile_grid["tiles"]`), and so
        # does this censor -- including partial edge tiles, whose smaller
        # `write_window` still maps to the correct 500 m cell because
        # `compute_cell_identity` divides the pixel offsets by the block size.
        grid = make_tile_grid(
            {"width": dataset.width, "height": dataset.height},
            tile_size_pixels=block_size,
        )
        tiles = grid["tiles"]
        for tile in tiles:
            col_off, row_off, width, height = tile["write_window"]
            window = Window(col_off, row_off, width, height)
            # Same masked-read idiom as Step8A: nodata and unset pixels become
            # NaN and are then excluded, so a masked pixel is never positive.
            block = dataset.read(1, window=window, masked=True).astype("float64").filled(np.nan)
            values = block[np.isfinite(block)]
            positives = int((values > 0).sum())
            if positives:
                positive_pixels += positives
                cell_id, _, _ = compute_cell_identity(row_off, col_off, block_size)
                censored.add(cell_id)
    return {
        "censor_applied": True,
        "raster_path": str(raster_path),
        "raster_sha256": sha256_file(raster_path),
        "block_size_pixels": int(block_size),
        "tile_count": int(grid["n_tiles"]),
        "tile_grid_shape": [int(grid["n_tile_rows"]), int(grid["n_tile_cols"])],
        "grid_contract_source": (
            "src.step8a_prepare_500m_modeling_dataset.compute_block_size_pixels "
            "/ compute_cell_identity + core.utils.tiling.make_tile_grid"
        ),
        "rule": "any positive constituent source pixel censors the whole cell",
        "majority_or_threshold_used": False,
        "positive_source_pixel_count": positive_pixels,
        "censored_cell_ids": sorted(censored),
        "censored_cell_count": len(censored),
        "zero_censored_cells_is_a_valid_outcome": True,
    }


# --- Exact common cohort ------------------------------------------------------
def build_model_common_cohort(
    frames_by_variant: dict,
    censor: dict,
    registry: dict,
    lineage: dict,
) -> dict:
    """ONE exact common cohort shared by all six evaluations.

    Reuses the production cell key (`variant_eligible_rows`, which already
    enforces `valid_for_modeling`, `analysis_eligible` and the primary
    population) and the N-way exact intersection with its cross-variant label,
    coordinate and population equality gates (`build_common_cohort`). The extra
    gate applied here is feature-union availability: a cell must carry every
    baseline and thermal model feature in EVERY variant, otherwise the six
    evaluations would not be scored on the same rows.
    """
    import numpy as np
    import pandas as pd

    order = sorted(frames_by_variant)
    censored = set(censor["censored_cell_ids"])
    features = list(registry["feature_union"])

    # The two shifted arms are produced from the same frozen pre-label censor.
    # Compare their decisions before eligibility filtering so a disagreement
    # cannot be hidden merely because one arm drops the affected cell.
    shifted_with_audit = [
        name for name in order
        if name != CANONICAL_VARIANT_ID
        and set(STEP8A_OPTIONAL_AUDIT_COLUMNS) <= set(frames_by_variant[name].columns)
    ]
    for name in shifted_with_audit:
        validate_step8a_optional_audit_columns(
            frames_by_variant[name], frame_name=name,
        )
    if len(shifted_with_audit) >= 2:
        anchor_name = shifted_with_audit[0]
        anchor_audit = frames_by_variant[anchor_name][
            ["cell_id", "analysis_eligible", "pre_label_burn_excluded"]
        ].set_index("cell_id")
        for name in shifted_with_audit[1:]:
            other = frames_by_variant[name][
                ["cell_id", "analysis_eligible", "pre_label_burn_excluded"]
            ].set_index("cell_id")
            common_audit_ids = anchor_audit.index.intersection(other.index)
            mismatch = (
                anchor_audit.loc[common_audit_ids].astype(bool).to_numpy()
                != other.loc[common_audit_ids].astype(bool).to_numpy()
            ).any(axis=1)
            if mismatch.any():
                examples = list(common_audit_ids[mismatch][:6])
                raise WindowClosureError(
                    "Shifted Step8A censor audit mismatch for common cell(s) "
                    f"{examples} between '{anchor_name}' and '{name}'."
                )

    initial_rows_by_variant: dict[str, int] = {}
    after_valid: dict[str, int] = {}
    after_population: dict[str, int] = {}
    after_censor: dict[str, int] = {}
    after_features: dict[str, int] = {}
    eligible: dict[str, Any] = {}

    for name in order:
        frame = frames_by_variant[name]
        initial_rows_by_variant[name] = int(len(frame))
        valid = frame
        if "valid_for_modeling" in frame.columns:
            valid = valid.loc[valid["valid_for_modeling"].astype(bool)]
        if "analysis_eligible" in frame.columns:
            valid = valid.loc[valid["analysis_eligible"].astype(bool)]
        after_valid[name] = int(len(valid))
        population = valid.loc[valid[PRIMARY_POPULATION].astype(bool)]
        after_population[name] = int(len(population))
        uncensored = population.loc[~population["cell_id"].isin(censored)]
        after_censor[name] = int(len(uncensored))
        missing_feature = uncensored[features].isna().any(axis=1)
        usable = uncensored.loc[~missing_feature]
        after_features[name] = int(len(usable))
        eligible[name] = usable.sort_values("cell_id", kind="mergesort").reset_index(drop=True)

    key_sets = [set(frame["cell_id"]) for frame in eligible.values()]
    common_ids = sorted(set.intersection(*key_sets)) if key_sets else []
    if not common_ids:
        raise WindowClosureError(
            "The exact common cohort is empty; no comparison is possible."
        )
    common = {
        name: frame.loc[frame["cell_id"].isin(common_ids)]
                   .sort_values("cell_id", kind="mergesort").reset_index(drop=True)
        for name, frame in eligible.items()
    }

    anchor_name = order[0]
    anchor = common[anchor_name]
    if anchor["cell_id"].duplicated().any():
        raise WindowClosureError("The common cohort carries duplicate cell_id values.")

    classification = classify_step8a_columns(list(anchor.columns), lineage)
    invariant_columns = [
        column for column in classification["invariant_columns"]
        if all(column in common[name].columns for name in order)
    ]
    label_columns = set(classification["label_columns"])

    label_mismatches: list[str] = []
    static_mismatches: list[str] = []
    for name in order[1:]:
        other = common[name]
        if not np.array_equal(anchor["cell_id"].to_numpy(), other["cell_id"].to_numpy()):
            raise WindowClosureError(
                f"Common cohort cell_id order differs between '{anchor_name}' "
                f"and '{name}'."
            )
        for column in invariant_columns:
            left, right = anchor[column], other[column]
            if pd.api.types.is_float_dtype(left) and pd.api.types.is_float_dtype(right):
                equal = np.isclose(
                    left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64"),
                    rtol=0.0, atol=STATIC_INVARIANCE_ABS_TOLERANCE, equal_nan=True,
                )
            else:
                equal = (left.to_numpy() == right.to_numpy()) | (
                    left.isna().to_numpy() & right.isna().to_numpy()
                )
            differing = int((~equal).sum())
            if not differing:
                continue
            target = label_mismatches if column in label_columns else static_mismatches
            target.append(f"{name}.{column}: {differing} row(s)")

    if label_mismatches:
        raise WindowClosureError(
            f"Label mismatch on the common cohort: {label_mismatches[:6]}. The "
            "label window is frozen, so the same cell must carry the same "
            "label in every variant."
        )
    if static_mismatches:
        raise WindowClosureError(
            f"Static invariance failure on the common cohort: "
            f"{static_mismatches[:6]}. Only predictor-timing-derived features "
            "may differ between variants."
        )

    labels = anchor["burned"].astype(int).to_numpy()
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise WindowClosureError(
            f"The common cohort carries a single class (positives={positives}, "
            f"negatives={negatives}); a model comparison is not possible."
        )

    metadata = {
        "stable_cell_key_columns": list(classification["key_columns"]),
        "primary_population": PRIMARY_POPULATION,
        "initial_rows_by_variant": initial_rows_by_variant,
        "rows_present_in_all_variants": int(len(common_ids)),
        "removed_not_valid_for_modeling": {
            name: initial_rows_by_variant[name] - after_valid[name] for name in order
        },
        "removed_outside_primary_population": {
            name: after_valid[name] - after_population[name] for name in order
        },
        "removed_prelabel_censor": {
            name: after_population[name] - after_censor[name] for name in order
        },
        "removed_missing_required_feature_union": {
            name: after_censor[name] - after_features[name] for name in order
        },
        "removed_variant_only_keys": {
            name: after_features[name] - int(len(common_ids)) for name in order
        },
        # Label / static disagreements are FAILURES, never silent removals, so
        # these counters are zero on every run that completes.
        "removed_label_mismatch": 0,
        "removed_static_invariance_failure": 0,
        "final_common_cohort_rows": int(len(anchor)),
        "final_positive_rows": positives,
        "final_negative_rows": negatives,
        "prevalence": float(positives / len(anchor)) if len(anchor) else None,
        "required_feature_union": features,
        "compared_invariant_columns": invariant_columns,
        "censor": {
            key: value for key, value in censor.items()
            if key != "censored_cell_ids"
        },
        "censored_cell_ids_applied": sorted(censored & set(
            pd.concat([frames_by_variant[name]["cell_id"] for name in order]).unique()
        )),
    }
    return {"common": common, "cell_ids": common_ids, "metadata": metadata}


# --- Shared spatial folds -----------------------------------------------------
def build_shared_spatial_folds(cohort_frame, configuration: dict) -> dict:
    """ONE fold assignment, produced by the production fold helper.

    `step8b.add_spatial_block_id` and `step8b.make_spatial_folds` are the
    production primitives; the block size, fold count and seed come from frozen
    config. Folds are derived from the cell geometry and the label only through
    the production stratification -- no predictor value participates.
    """
    import numpy as np

    from src.step8b_train_baseline_vs_thermal_model import (
        add_spatial_block_id, make_spatial_folds,
    )

    blocked = add_spatial_block_id(
        cohort_frame, configuration["spatial_block_size_cells"],
    )
    labels = blocked["burned"].astype(int).to_numpy()
    groups = blocked["spatial_block_id"].to_numpy()
    folds, n_splits_used = make_spatial_folds(
        labels, groups, configuration["n_splits"], configuration["fold_random_seed"],
        strict=configuration["strict_folds"],
    )
    fold_id = np.full(len(blocked), -1, dtype=int)
    for index, (_, test_idx) in enumerate(folds):
        if np.any(fold_id[test_idx] != -1):
            raise WindowClosureError(
                "A cohort row was assigned to more than one validation fold."
            )
        fold_id[test_idx] = index
    if np.any(fold_id < 0):
        raise WindowClosureError(
            "Some cohort rows received no validation fold; every row must "
            "receive exactly one out-of-fold prediction."
        )

    assignment = {
        str(cell): int(fold)
        for cell, fold in zip(blocked["cell_id"].tolist(), fold_id.tolist())
    }
    block_of = dict(zip(blocked["cell_id"].tolist(), groups.tolist()))
    blocks_per_fold: dict[int, set] = {}
    for cell, fold in assignment.items():
        blocks_per_fold.setdefault(fold, set()).add(block_of[cell])
    split_blocks = sorted({
        block
        for left in blocks_per_fold
        for right in blocks_per_fold
        if left < right
        for block in blocks_per_fold[left] & blocks_per_fold[right]
    })
    if split_blocks:
        raise WindowClosureError(
            f"Spatial block(s) {split_blocks[:6]} are split across validation "
            "folds; blocks must stay whole."
        )

    rows_per_fold = {
        int(fold): int(np.sum(fold_id == fold)) for fold in sorted(set(fold_id.tolist()))
    }
    positives_per_fold = {
        int(fold): int(np.sum(labels[fold_id == fold] == 1)) for fold in rows_per_fold
    }
    negatives_per_fold = {
        int(fold): int(np.sum(labels[fold_id == fold] == 0)) for fold in rows_per_fold
    }
    return {
        "frame": blocked.assign(fold_id=fold_id),
        "fold_id": fold_id,
        "assignment": assignment,
        "fold_count": int(n_splits_used),
        "requested_fold_count": int(configuration["n_splits"]),
        "random_seed": int(configuration["fold_random_seed"]),
        "spatial_block_definition": {
            "block_size_cells": int(configuration["spatial_block_size_cells"]),
            "source": (
                "src.step8b_train_baseline_vs_thermal_model.add_spatial_block_id "
                "(block_row = row_500m // block_size, block_col = col_500m // "
                "block_size, fixed origin)"
            ),
        },
        "unique_block_count": int(len(set(groups.tolist()))),
        "rows_per_fold": rows_per_fold,
        "positives_per_fold": positives_per_fold,
        "negatives_per_fold": negatives_per_fold,
        "block_disjointness_pass": True,
        "every_row_assigned_once": True,
        "label_or_predictor_used_in_assignment": (
            "label used only by the production stratified grouped split; no "
            "predictor value participates"
        ),
        "assignment_sha256": sha256_bytes(
            canonical_json(dict(sorted(assignment.items()))).encode("utf-8")
        ),
    }


# --- Model fitting ------------------------------------------------------------
def fit_variant_models(
    variant_id: str, cohort_frame, shared: dict, configuration: dict,
) -> dict:
    """Fit BOTH production fire-risk models of one variant on the shared folds.

    `step8b.train_population` is the production routine: it fits Model A
    (baseline) and Model B (baseline+thermal) on the SAME spatial-block folds
    with the production pipeline, hyper-parameters and seed. It is called, not
    reimplemented, and the fold assignment it derives is asserted to be the
    shared one.
    """
    import numpy as np

    from src.step8b_train_baseline_vs_thermal_model import train_population

    result = train_population(
        cohort_frame,
        population_name=PRIMARY_POPULATION,
        n_splits=configuration["n_splits"],
        random_state=configuration["fold_random_seed"],
        model_name=configuration["model"],
        min_positives=configuration["min_positives"],
        group_column="spatial_block_id",
        strict_folds=configuration["strict_folds"],
    )
    if result is None or result.get("skipped"):
        raise WindowClosureError(
            f"Variant '{variant_id}' could not be modelled: "
            f"{(result or {}).get('reason', 'population skipped')}."
        )
    if not np.array_equal(np.asarray(result["fold_id"]), shared["fold_id"]):
        raise WindowClosureError(
            f"Variant '{variant_id}' produced a different fold assignment than "
            "the shared one; all six evaluations must run on identical folds."
        )
    for family, probabilities in (
        ("baseline", result["oof_prob_baseline"]),
        ("thermal", result["oof_prob_thermal"]),
    ):
        array = np.asarray(probabilities, dtype="float64")
        if array.shape[0] != len(cohort_frame) or not np.all(np.isfinite(array)):
            raise WindowClosureError(
                f"Variant '{variant_id}' {family} out-of-fold predictions are "
                "incomplete; every cohort row must receive exactly one."
            )
    return result


def build_oof_table(variant_id: str, family: str, cohort_frame, shared: dict, probabilities):
    """The per-evaluation out-of-fold prediction table."""
    import numpy as np
    import pandas as pd

    return pd.DataFrame({
        "cell_id": cohort_frame["cell_id"].to_numpy(),
        "spatial_block_id": cohort_frame["spatial_block_id"].to_numpy(),
        "fold_id": np.asarray(shared["fold_id"], dtype=int),
        "y_true": cohort_frame["burned"].astype(int).to_numpy(),
        "y_score": np.asarray(probabilities, dtype="float64"),
        "variant_id": variant_id,
        "model_family": family,
    })


def build_point_metrics(results_by_variant: dict, shared: dict) -> list[dict]:
    """Pooled out-of-fold point metrics per variant and model family."""
    rows: list[dict] = []
    for variant_id in sorted(results_by_variant):
        result = results_by_variant[variant_id]
        for family, key in (("baseline", "overall_baseline"), ("thermal", "overall_thermal")):
            metrics = result[key]
            rows.append({
                "variant_id": variant_id,
                "model_family": family,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier": metrics["brier_score"],
                "row_count": int(metrics["positive_count"] + metrics["negative_count"]),
                "positive_count": int(metrics["positive_count"]),
                "negative_count": int(metrics["negative_count"]),
                "prevalence": (
                    metrics["positive_count"]
                    / (metrics["positive_count"] + metrics["negative_count"])
                    if (metrics["positive_count"] + metrics["negative_count"]) else None
                ),
                "fold_count": int(shared["fold_count"]),
                "metric_source": (
                    "src.step8b_train_baseline_vs_thermal_model.compute_binary_metrics"
                ),
            })
    rows.sort(key=lambda row: (row["variant_id"], row["model_family"]))
    return rows


def build_thermal_contribution_rows(results_by_variant: dict) -> list[dict]:
    """RAW thermal-minus-baseline contribution per variant and metric."""
    rows: list[dict] = []
    for variant_id in sorted(results_by_variant):
        result = results_by_variant[variant_id]
        baseline, thermal = result["overall_baseline"], result["overall_thermal"]
        for metric, baseline_key, thermal_key in (
            ("roc_auc", "roc_auc", "roc_auc"),
            ("pr_auc", "pr_auc", "pr_auc"),
            ("brier", "brier_score", "brier_score"),
        ):
            rows.append({
                "variant_id": variant_id,
                "metric": metric,
                "baseline": baseline[baseline_key],
                "thermal": thermal[thermal_key],
                "contribution_delta": (
                    thermal[thermal_key] - baseline[baseline_key]
                    if baseline[baseline_key] is not None
                    and thermal[thermal_key] is not None else None
                ),
                "delta_definition": "thermal - baseline (raw)",
                "sign_convention": METRIC_SIGN_CONVENTIONS[metric],
            })
    rows.sort(key=lambda row: (row["variant_id"], row["metric"]))
    return rows


# --- Paired spatial-block bootstrap comparisons -------------------------------
#: Replicate column suffix for each reported metric, per model family.
_FAMILY_METRIC_COLUMN = {
    ("baseline", "roc_auc"): "baseline_roc_auc",
    ("baseline", "pr_auc"): "baseline_pr_auc",
    ("baseline", "brier"): "baseline_brier",
    ("thermal", "roc_auc"): "thermal_roc_auc",
    ("thermal", "pr_auc"): "thermal_pr_auc",
    ("thermal", "brier"): "thermal_brier",
}
_CONTRIBUTION_COLUMN = {
    "roc_auc": "delta_roc_auc",
    "pr_auc": "delta_pr_auc",
    "brier": "delta_brier",
}


def _interval_row(
    replicates, expression, point_estimate, bootstrap: dict, configuration: dict,
    **fields,
) -> dict:
    """One comparison row with its paired percentile interval and status."""
    values = expression(replicates).tolist() if len(replicates) else []
    interval = percentile_interval(
        values, bootstrap["ci_lower_percentile"], bootstrap["ci_upper_percentile"],
    )
    valid = int(interval["n_replicates"])
    invalid = int(bootstrap["n_bootstrap_requested"]) - valid
    return {
        **fields,
        "point_delta": point_estimate,
        "bootstrap_mean": interval["point"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "confidence_level": configuration["confidence_level"],
        "ci_lower_percentile": bootstrap["ci_lower_percentile"],
        "ci_upper_percentile": bootstrap["ci_upper_percentile"],
        "requested_replicates": int(bootstrap["n_bootstrap_requested"]),
        "valid_replicates": valid,
        "invalid_replicates": invalid,
        "bootstrap_seed": int(bootstrap["seed"]),
        "block_count": int(bootstrap["n_blocks"]),
        "status": classify_change_interval(interval["ci_low"], interval["ci_high"]),
    }


def build_paired_bootstrap_rows(
    bootstrap: dict, point_metrics: Sequence[dict], configuration: dict,
) -> list[dict]:
    """The three preregistered comparison families, on ONE set of block draws.

    A: thermal contribution within each variant (thermal - baseline);
    B: closure change within each model family (earlier - canonical);
    C: change in thermal contribution (earlier - canonical).

    Every comparison is read off the SAME replicate table, so the draws are
    shared and the differences are properly paired.
    """
    replicates = bootstrap["replicates"]
    point = {
        (row["variant_id"], row["model_family"]): row for row in point_metrics
    }
    variants = list(bootstrap["variants"])
    rows: list[dict] = []

    def _column(variant_id: str, suffix: str) -> str:
        return f"{variant_id}__{suffix}"

    for variant_id in variants:
        for metric in MODEL_METRICS:
            baseline_point = point[(variant_id, "baseline")][metric]
            thermal_point = point[(variant_id, "thermal")][metric]
            column = _column(variant_id, _CONTRIBUTION_COLUMN[metric])
            rows.append(_interval_row(
                replicates, lambda df, c=column: df[c], (
                    thermal_point - baseline_point
                    if baseline_point is not None and thermal_point is not None else None
                ),
                bootstrap, configuration,
                comparison=COMPARISON_THERMAL_CONTRIBUTION,
                variant_id=variant_id, model_family="thermal_minus_baseline",
                metric=metric,
                delta_definition="thermal - baseline (raw)",
                sign_convention=METRIC_SIGN_CONVENTIONS[metric],
            ))

    for variant_id in variants:
        if variant_id == CANONICAL_VARIANT_ID:
            continue
        for family in MODEL_FAMILIES:
            for metric in MODEL_METRICS:
                suffix = _FAMILY_METRIC_COLUMN[(family, metric)]
                left = _column(variant_id, suffix)
                right = _column(CANONICAL_VARIANT_ID, suffix)
                variant_point = point[(variant_id, family)][metric]
                canonical_point = point[(CANONICAL_VARIANT_ID, family)][metric]
                rows.append(_interval_row(
                    replicates, lambda df, a=left, b=right: df[a] - df[b], (
                        variant_point - canonical_point
                        if variant_point is not None and canonical_point is not None
                        else None
                    ),
                    bootstrap, configuration,
                    comparison=COMPARISON_CLOSURE_CHANGE,
                    variant_id=variant_id, model_family=family, metric=metric,
                    delta_definition="earlier_closure - canonical (raw)",
                    sign_convention=METRIC_SIGN_CONVENTIONS[metric],
                ))

    for variant_id in variants:
        if variant_id == CANONICAL_VARIANT_ID:
            continue
        for metric in MODEL_METRICS:
            suffix = _CONTRIBUTION_COLUMN[metric]
            left = _column(variant_id, suffix)
            right = _column(CANONICAL_VARIANT_ID, suffix)
            variant_contribution = (
                point[(variant_id, "thermal")][metric]
                - point[(variant_id, "baseline")][metric]
            )
            canonical_contribution = (
                point[(CANONICAL_VARIANT_ID, "thermal")][metric]
                - point[(CANONICAL_VARIANT_ID, "baseline")][metric]
            )
            rows.append(_interval_row(
                replicates, lambda df, a=left, b=right: df[a] - df[b],
                variant_contribution - canonical_contribution,
                bootstrap, configuration,
                comparison=COMPARISON_CONTRIBUTION_CHANGE,
                variant_id=variant_id, model_family="thermal_minus_baseline",
                metric=metric,
                delta_definition=(
                    "(thermal - baseline)_earlier - (thermal - baseline)_canonical"
                ),
                sign_convention=METRIC_SIGN_CONVENTIONS[metric],
            ))

    rows.sort(key=lambda row: (row["comparison"], row["variant_id"],
                               row["model_family"], row["metric"]))
    return rows


def assert_bootstrap_sufficient(bootstrap: dict, configuration: dict) -> None:
    """Fail closed when too few replicates survived. No silent substitution."""
    valid = int(bootstrap["n_bootstrap_valid"])
    minimum = int(configuration["minimum_valid_replicates"])
    if valid < minimum:
        raise WindowClosureError(
            f"The paired spatial-block bootstrap produced {valid} valid "
            f"replicate(s), below the project minimum of {minimum} "
            f"({configuration['minimum_valid_replicates_source']}). A replicate "
            "whose resampled blocks carry a single class leaves every metric "
            "UNDEFINED and is counted invalid -- it is never silently scored as "
            "zero. No summary is published."
        )


# --- The stage ----------------------------------------------------------------
def model_variant_is_reusable(
    experiment_id: str, analysis_id: str, binding: dict,
    output_root: Optional[Path] = None,
) -> tuple[bool, Optional[dict], str]:
    """Whether an existing model output may be reused untouched."""
    metadata_path = model_metadata_path(experiment_id, output_root)
    if not metadata_path.is_file():
        return False, None, "no model stage metadata"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False, None, "unreadable model stage metadata"
    if not isinstance(metadata, dict):
        return False, None, "model stage metadata is not a JSON object"
    if metadata.get("schema_version") != MODEL_METADATA_SCHEMA:
        return False, metadata, f"metadata schema is {metadata.get('schema_version')!r}"
    if metadata.get("analysis_id") != analysis_id:
        return False, metadata, "analysis_id mismatch"
    if metadata.get("status") != STATUS_PASS:
        return False, metadata, f"previous status is {metadata.get('status')!r}"

    recorded_inputs = metadata.get("input_dataset_sha256") or {}
    current_inputs = {
        variant_id: record["dataset_sha256"]
        for variant_id, record in (binding.get("model_datasets") or {}).items()
    }
    if recorded_inputs != current_inputs:
        return False, metadata, "a bound Step8A dataset hash has changed"

    inventory = metadata.get("artifact_inventory") or []
    if not inventory:
        return False, metadata, "the recorded artifact inventory is empty"
    for record in inventory:
        path = Path(str((record or {}).get("path") or ""))
        if not path.is_file():
            return False, metadata, f"missing artefact {record.get('relative_path')}"
        if sha256_file(path) != record.get("sha256"):
            return False, metadata, f"hash mismatch for {record.get('relative_path')}"
    return True, metadata, "complete and verified"


def _quarantine_model_outputs(
    experiment_id: str, output_root: Optional[Path] = None,
) -> dict:
    """Move ONLY `model/` aside. Never touches an upstream or config path."""
    root = model_root(experiment_id, output_root)
    if not root.exists():
        return {"quarantined": False, "entries": []}
    before = snapshot_model_state(experiment_id, output_root)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    target_root = (
        experiment_root(experiment_id, output_root)
        / LOCAL_DOWNSTREAM_QUARANTINE_DIR / MODEL_QUARANTINE_KIND / stamp
    )
    target_root.mkdir(parents=True, exist_ok=True)
    destination = target_root / MODEL_ROOT_DIR
    os.replace(root, destination)
    return {
        "quarantined": True,
        "entries": [{
            "original_path": str(root),
            "quarantined_path": str(destination),
            "reason": "force rebuild of the model stage",
            "timestamp_utc": stamp,
            "pre_quarantine_inventory_sha256": before["digest"],
            "pre_quarantine_file_count": before["file_count"],
        }],
    }


def run_model_stage(
    experiment_id: str,
    analysis_id: str,
    canonical: dict,
    variants: Sequence[dict],
    binding: dict,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
    force: bool = False,
    resume: bool = False,
    configuration_overrides: Optional[dict] = None,
) -> dict:
    """Fit the six production evaluations and publish the model stage atomically.

    `configuration_overrides` is a TEST-ONLY dependency-injection point; a
    production run passes None and therefore uses the frozen `core.config`
    values verbatim.
    """
    import shutil

    import pandas as pd

    metadata_path = model_metadata_path(experiment_id, output_root)
    root = model_root(experiment_id, output_root)

    reusable, previous, reason = model_variant_is_reusable(
        experiment_id, analysis_id, binding, output_root,
    )
    if reusable and not force:
        if not resume:
            raise WindowClosureError(
                f"The model stage already has a complete, verified output at "
                f"{metadata_path}. Refusing to overwrite it silently: re-run "
                "with resume=True to reuse it, or force=True to rebuild it "
                "(the old model/ tree is quarantined, never deleted)."
            )
        return {
            "reused": True, "reason": reason, "files_written": [],
            "quarantine": {"quarantined": False, "entries": []},
            "metadata": previous or {},
            **MODEL_STAGE_SEMANTICS,
        }
    if resume:
        raise WindowClosureError(
            f"--resume cannot reuse the model stage: {reason}. --resume only "
            "ever REUSES a complete, verified status=pass output; it never "
            f"rebuilds one. Nothing was quarantined, moved, deleted or written "
            f"at {root}. Re-run with force=True to rebuild it."
        )
    if not force and root.exists():
        raise WindowClosureError(
            f"The model stage has an existing but NOT reusable output "
            f"({reason}) at {root}. Refusing to overwrite it silently: inspect "
            "it, then re-run with force=True (the old model/ tree is "
            "quarantined, never deleted)."
        )

    configuration = dict(model_frozen_configuration())
    if configuration_overrides:
        configuration.update(configuration_overrides)
        configuration["source"] = "core.config (frozen) + explicit test override"
        configuration["overridden_keys"] = sorted(configuration_overrides)
    registry = model_feature_registry()
    lineage = step8a_predictor_lineage(experiment_id, experiments_root)

    frozen_before = frozen_hash_map(model_frozen_inputs(
        experiment_id,
        frozen_input_inventory(experiment_id, experiments_root),
        variants, output_root, experiments_root,
    ))

    # --- Inputs (read-only) --------------------------------------------------
    datasets = binding["model_datasets"]
    frames = {
        variant_id: pd.read_parquet(record["dataset_path"])
        for variant_id, record in sorted(datasets.items())
    }
    censor = prelabel_censored_cells(prelabel_raster_path(experiment_id, output_root))
    cohort = build_model_common_cohort(frames, censor, registry, lineage)
    cohort_metadata = dict(cohort["metadata"])
    cohort_metadata["input_dataset_paths"] = {
        variant_id: record["dataset_path"] for variant_id, record in sorted(datasets.items())
    }
    cohort_metadata["input_dataset_sha256"] = {
        variant_id: record["dataset_sha256"] for variant_id, record in sorted(datasets.items())
    }

    shared = build_shared_spatial_folds(
        cohort["common"][CANONICAL_VARIANT_ID], configuration,
    )
    block_ids = shared["frame"]["spatial_block_id"].to_numpy()

    # --- Six evaluations on identical rows and identical folds ---------------
    results: dict[str, dict] = {}
    oof_tables: dict[tuple[str, str], Any] = {}
    probabilities_by_variant: dict[str, dict] = {}
    for variant_id in sorted(cohort["common"]):
        frame = cohort["common"][variant_id].assign(
            spatial_block_id=block_ids, fold_id=shared["fold_id"],
        )
        result = fit_variant_models(variant_id, frame, shared, configuration)
        results[variant_id] = result
        probabilities_by_variant[variant_id] = {
            "baseline": result["oof_prob_baseline"],
            "thermal": result["oof_prob_thermal"],
        }
        for family, probabilities in (
            ("baseline", result["oof_prob_baseline"]),
            ("thermal", result["oof_prob_thermal"]),
        ):
            oof_tables[(variant_id, family)] = build_oof_table(
                variant_id, family, frame, shared, probabilities,
            )

    point_metrics = build_point_metrics(results, shared)
    contributions = build_thermal_contribution_rows(results)

    # --- ONE paired spatial-block bootstrap ----------------------------------
    labels = cohort["common"][CANONICAL_VARIANT_ID]["burned"].astype(int).to_numpy()
    bootstrap = multi_variant_block_bootstrap(
        shared["frame"], labels, probabilities_by_variant,
        n_bootstrap=configuration["n_bootstrap"], seed=configuration["bootstrap_seed"],
        ci_lower=configuration["ci_lower_percentile"],
        ci_upper=configuration["ci_upper_percentile"],
    )
    assert_bootstrap_sufficient(bootstrap, configuration)
    comparison_rows = build_paired_bootstrap_rows(bootstrap, point_metrics, configuration)

    # --- Staged writes, promoted atomically ----------------------------------
    staging = model_staging_root(experiment_id, output_root)
    if staging.exists():
        shutil.rmtree(staging)
    layout = model_relative_layout()
    written: list[dict] = []

    def _stage(relative: str, writer) -> None:
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)
        written.append({"relative_path": relative, "path": path})

    try:
        cohort_frame = shared["frame"]
        _stage(layout["common_cohort"],
               lambda p: cohort_frame.to_parquet(p, index=False))
        _stage(layout["common_cohort_metadata"],
               lambda p: p.write_text(_json_document(cohort_metadata), encoding="utf-8"))
        _stage(layout["shared_folds"], lambda p: pd.DataFrame({
            "cell_id": cohort_frame["cell_id"].to_numpy(),
            "spatial_block_id": block_ids,
            "fold_id": shared["fold_id"],
        }).to_parquet(p, index=False))
        _stage(layout["shared_folds_metadata"], lambda p: p.write_text(
            _json_document({
                key: value for key, value in shared.items()
                if key not in ("frame", "fold_id", "assignment")
            }), encoding="utf-8",
        ))
        for (variant_id, family), table in sorted(oof_tables.items()):
            _stage(model_variant_oof_relpath(variant_id, family),
                   lambda p, t=table: t.to_parquet(p, index=False))
        for variant_id in sorted(results):
            fold_rows = results[variant_id]["fold_rows"]
            for family in MODEL_FAMILIES:
                _stage(
                    model_variant_metrics_relpath(variant_id, family),
                    lambda p, rows=fold_rows, f=family: p.write_text(
                        _csv_document(
                            sorted({key for row in rows for key in row}),
                            [dict(row, model_family=f) for row in rows],
                        ),
                        encoding="utf-8",
                    ),
                )
        _stage(layout["point_metrics_csv"], lambda p: p.write_text(
            _csv_document(sorted({k for row in point_metrics for k in row}), point_metrics),
            encoding="utf-8",
        ))
        _stage(layout["point_metrics_json"], lambda p: p.write_text(
            _json_document({
                "schema_version": MODEL_METADATA_SCHEMA,
                "analysis_id": analysis_id,
                "metric_sign_conventions": METRIC_SIGN_CONVENTIONS,
                "point_metrics": point_metrics,
            }), encoding="utf-8",
        ))
        _stage(layout["thermal_contributions_csv"], lambda p: p.write_text(
            _csv_document(sorted({k for row in contributions for k in row}), contributions),
            encoding="utf-8",
        ))
        _stage(layout["bootstrap_replicates"],
               lambda p: bootstrap["replicates"].to_parquet(p, index=False))
        _stage(layout["bootstrap_summary_csv"], lambda p: p.write_text(
            _csv_document(
                sorted({k for row in comparison_rows for k in row}), comparison_rows,
            ),
            encoding="utf-8",
        ))
        _stage(layout["bootstrap_summary_json"], lambda p: p.write_text(
            _json_document({
                "schema_version": MODEL_METADATA_SCHEMA,
                "analysis_id": analysis_id,
                "bootstrap_configuration": {
                    key: bootstrap[key] for key in (
                        "bootstrap_unit", "n_blocks", "n_bootstrap_requested",
                        "n_bootstrap_valid", "seed", "ci_lower_percentile",
                        "ci_upper_percentile",
                        "identical_block_draws_across_variants",
                    )
                },
                "models_refit_per_replicate": False,
                "allowed_statuses": [
                    INTERVAL_SUPPORTED_INCREASE, INTERVAL_SUPPORTED_DECREASE,
                    INTERVAL_INCLUDES_ZERO,
                ],
                "comparisons": comparison_rows,
            }), encoding="utf-8",
        ))

        inventory_records = [
            {
                "relative_path": record["relative_path"],
                "path": str(root / record["relative_path"]),
                "sha256": sha256_file(record["path"]),
                "bytes": int(record["path"].stat().st_size),
                "media_type": _media_type(record["path"]),
            }
            for record in sorted(written, key=lambda item: item["relative_path"])
        ]
        metadata = build_model_stage_metadata(
            experiment_id, analysis_id, canonical, variants, binding, configuration,
            registry, censor, cohort_metadata, shared, point_metrics, contributions,
            bootstrap, comparison_rows, inventory_records, frozen_before,
            output_root, experiments_root,
        )
        metadata_text = _json_document(metadata)
        assert_report_wording(metadata_text)
        (staging / layout["metadata"]).write_text(metadata_text, encoding="utf-8")
        inventory_records.append({
            "relative_path": layout["metadata"],
            "path": str(root / layout["metadata"]),
            "sha256": sha256_bytes(metadata_text.encode("utf-8")),
            "bytes": len(metadata_text.encode("utf-8")),
            "media_type": "application/json",
        })

        assert_frozen_hashes_unchanged(
            frozen_before,
            frozen_hash_map(model_frozen_inputs(
                experiment_id,
                frozen_input_inventory(experiment_id, experiments_root),
                variants, output_root, experiments_root,
            )),
            "while running the model stage",
        )

        quarantine = _quarantine_model_outputs(experiment_id, output_root)
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "reused": False,
        "reason": "produced",
        "files_written": sorted(str(root / record["relative_path"]) for record in inventory_records),
        "quarantine": quarantine,
        "metadata": metadata,
        "artifact_inventory": inventory_records,
        **MODEL_STAGE_SEMANTICS,
    }


def build_model_stage_metadata(
    experiment_id: str,
    analysis_id: str,
    canonical: dict,
    variants: Sequence[dict],
    binding: dict,
    configuration: dict,
    registry: dict,
    censor: dict,
    cohort_metadata: dict,
    shared: dict,
    point_metrics: Sequence[dict],
    contributions: Sequence[dict],
    bootstrap: dict,
    comparison_rows: Sequence[dict],
    inventory_records: Sequence[dict],
    frozen_before: dict,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
) -> dict:
    """The model stage record. Deterministic and self-describing."""
    frozen_after = frozen_hash_map(model_frozen_inputs(
        experiment_id, frozen_input_inventory(experiment_id, experiments_root),
        variants, output_root, experiments_root,
    ))
    datasets = binding["model_datasets"]
    return {
        "schema_version": MODEL_METADATA_SCHEMA,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "stage": MODEL_STAGE,
        "variant_ids": sorted(datasets),
        "nonzero_variant_ids": [v["variant_id"] for v in nonzero_variants(variants)],
        "canonical_variant_id": CANONICAL_VARIANT_ID,

        "input_dataset_paths": {
            variant_id: record["dataset_path"] for variant_id, record in sorted(datasets.items())
        },
        "input_dataset_sha256": {
            variant_id: record["dataset_sha256"] for variant_id, record in sorted(datasets.items())
        },
        "input_binding": {
            variant_id: {
                key: value for key, value in record.items() if key != "dataset_path"
            }
            for variant_id, record in sorted(datasets.items())
        },
        "canonical_step8a_sha256": binding["canonical_step8a_sha256"],

        "model_configuration": configuration,
        "feature_registry": registry,
        "prelabel_censor": {
            key: value for key, value in censor.items() if key != "censored_cell_ids"
        },
        "prelabel_censored_cell_count": censor["censored_cell_count"],
        "common_cohort": cohort_metadata,
        "shared_folds": {
            key: value for key, value in shared.items()
            if key not in ("frame", "fold_id", "assignment")
        },

        "model_evaluations": [
            {"variant_id": variant_id, "model_family": family}
            for variant_id in sorted(datasets) for family in MODEL_FAMILIES
        ],
        "model_evaluation_count": len(datasets) * len(MODEL_FAMILIES),
        "point_metrics": list(point_metrics),
        "thermal_contributions": list(contributions),
        "metric_sign_conventions": METRIC_SIGN_CONVENTIONS,
        "brier_sign_convention": BRIER_SIGN_CONVENTION,

        "bootstrap": {
            key: bootstrap[key] for key in (
                "bootstrap_unit", "n_blocks", "n_bootstrap_requested",
                "n_bootstrap_valid", "seed", "ci_lower_percentile",
                "ci_upper_percentile", "identical_block_draws_across_variants",
                "variants",
            )
        },
        "bootstrap_invalid_replicates": int(
            bootstrap["n_bootstrap_requested"] - bootstrap["n_bootstrap_valid"]
        ),
        "bootstrap_models_refit_per_replicate": False,
        "comparisons": list(comparison_rows),
        "allowed_statuses": [
            INTERVAL_SUPPORTED_INCREASE, INTERVAL_SUPPORTED_DECREASE,
            INTERVAL_INCLUDES_ZERO,
        ],

        "artifact_inventory": list(inventory_records),
        "artifact_count": len(inventory_records),
        "all_paths_inside_model_namespace": True,

        **MODEL_STAGE_SEMANTICS,
        "canonical_outputs_modified": False,
        "upstream_outputs_modified": False,
        "frozen_input_sha256_before": dict(frozen_before),
        "frozen_input_sha256_after": dict(frozen_after),
        "frozen_hashes_unchanged": frozen_before == frozen_after,
        "limitations": list(MODEL_LIMITATIONS),
        "status": STATUS_PASS,
    }


def model_stage_summary(
    experiment_id: str,
    analysis_id: str,
    canonical: dict,
    variants: Sequence[dict],
    inventory: dict,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
) -> dict:
    """The whole-analysis model plan, as reported by a dry run. Read-only."""
    datasets: dict[str, dict] = {}
    canonical_dataset = canonical_step8a_path(experiment_id, experiments_root)
    datasets[CANONICAL_VARIANT_ID] = {
        "variant_id": CANONICAL_VARIANT_ID,
        "dataset_path": str(canonical_dataset),
        "dataset_present": canonical_dataset.is_file(),
        "dataset_sha256": sha256_file(canonical_dataset) if canonical_dataset.is_file() else None,
        "source": "frozen canonical production Step8A (read-only)",
    }
    binding_ready = canonical_dataset.is_file()
    for variant in nonzero_variants(variants):
        variant_id = variant["variant_id"]
        metadata_path = local_downstream_metadata_path(experiment_id, variant_id, output_root)
        dataset = variant_step8a_dataset_path(experiment_id, variant_id, output_root)
        recorded = None
        status = None
        if metadata_path.is_file():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                recorded = payload.get("step8a_dataset_sha256")
                status = payload.get("status")
            except (OSError, ValueError, UnicodeDecodeError):
                recorded = None
        present = dataset.is_file()
        digest = sha256_file(dataset) if present else None
        ready = bool(present and status == STATUS_PASS and digest == recorded)
        binding_ready = binding_ready and ready
        datasets[variant_id] = {
            "variant_id": variant_id,
            "dataset_path": str(dataset),
            "dataset_present": present,
            "dataset_sha256": digest,
            "recorded_sha256": recorded,
            "local_downstream_status": status,
            "binding_ready": ready,
            "source": "window-closure local downstream",
        }

    root = model_root(experiment_id, output_root)
    layout = model_relative_layout()
    planned = sorted(
        [str(root / relative) for relative in layout.values()]
        + [
            str(root / model_variant_oof_relpath(variant_id, family))
            for variant_id in sorted(datasets) for family in MODEL_FAMILIES
        ]
    )
    experiment = experiment_root(experiment_id, output_root).resolve()
    contained = all(experiment in Path(path).resolve().parents for path in planned)
    try:
        configuration = model_frozen_configuration()
        configuration_error = None
    except WindowClosureError as exc:
        configuration, configuration_error = None, str(exc)

    prelabel = prelabel_raster_path(experiment_id, output_root)
    return {
        "stage_root": str(root),
        "variant_ids": sorted(datasets),
        "input_datasets": datasets,
        "input_binding_ready": binding_ready,
        "expected_input_dataset_count": len(datasets),
        "primary_population": PRIMARY_POPULATION,
        "stable_cell_key_columns": list(STEP8A_KEY_COLUMNS),
        "model_evaluations_planned": [
            {"variant_id": variant_id, "model_family": family}
            for variant_id in sorted(datasets) for family in MODEL_FAMILIES
        ],
        "model_evaluation_count_planned": len(datasets) * len(MODEL_FAMILIES),
        "feature_registry": model_feature_registry(),
        "model_configuration": configuration,
        "model_configuration_error": configuration_error,
        "prelabel_censor": {
            "raster_path": str(prelabel),
            "raster_present": prelabel.is_file(),
            "raster_sha256": sha256_file(prelabel) if prelabel.is_file() else None,
            "censor_applied_planned": True,
            "rule": "any positive constituent source pixel censors the whole cell",
            "majority_or_threshold_used": False,
        },
        "point_metrics_planned": list(MODEL_METRICS),
        "contribution_metrics_planned": list(MODEL_CONTRIBUTION_METRICS),
        "metric_sign_conventions": METRIC_SIGN_CONVENTIONS,
        "comparison_families_planned": [
            COMPARISON_THERMAL_CONTRIBUTION, COMPARISON_CLOSURE_CHANGE,
            COMPARISON_CONTRIBUTION_CHANGE,
        ],
        "allowed_statuses": [
            INTERVAL_SUPPORTED_INCREASE, INTERVAL_SUPPORTED_DECREASE,
            INTERVAL_INCLUDES_ZERO,
        ],
        "planned_output_paths": planned,
        "all_paths_inside_model_namespace": contained,
        **MODEL_DRY_RUN_SEMANTICS,
        "limitations": list(MODEL_LIMITATIONS),
    }


# =============================================================================
# Actual COMPARE stage
#
# READ-ONLY. It fits nothing, draws nothing and recomputes no bootstrap
# replicate: it opens the VERIFIED model-stage artefacts, re-derives every
# number from them so a mis-stated value cannot survive, and publishes
# deterministic tables plus a scientific synthesis.
#
# The synthesis deliberately does NOT collapse to one overall scientific
# verdict: technical artefact validation is reported separately from the
# evidence, and the evidence is reported per metric and per comparison family.
# No majority vote across metrics is taken.
# =============================================================================
COMPARE_STAGE = "compare"
COMPARE_METADATA_SCHEMA = "window_closure_compare.v1"
COMPARE_ROOT_DIR = "compare"
COMPARE_METADATA_NAME = "compare_stage_metadata.json"
COMPARE_STAGING_DIR = "_compare_staging"
COMPARE_QUARANTINE_KIND = "compare"

#: Deterministic orderings. Nothing in this stage depends on glob order.
COMPARE_FAMILY_ORDER: tuple[str, ...] = (
    COMPARISON_THERMAL_CONTRIBUTION, COMPARISON_CLOSURE_CHANGE,
    COMPARISON_CONTRIBUTION_CHANGE,
)
COMPARE_DISPLAY_DECIMALS = 3

#: Prose wording that must never appear in a compare artefact. These are
#: significance-, equivalence- and stability-style claims this analysis cannot
#: support. They are checked against PROSE fields only -- machine-readable key
#: names such as `frozen_hashes_unchanged` are structural identifiers, not
#: claims, exactly as `landsat_product_violations` inspects product fields
#: rather than the serialized document text.
FORBIDDEN_COMPARE_PHRASES: tuple[str, ...] = (
    "statistically significant", "significant difference", "non-significant",
    "insignificant", "p-value", "p value", "hypothesis test",
    "equivalent", "equivalence", "unchanged", "stable", "robust",
)

METRIC_DIRECTION_NOTES: dict[str, str] = {
    "roc_auc": (
        "ROC-AUC: a positive raw delta indicates a higher ROC-AUC value."
    ),
    "pr_auc": (
        "PR-AUC: a positive raw delta indicates a higher PR-AUC value."
    ),
    "brier": (
        "Brier score is a loss: a negative raw delta indicates a lower Brier "
        "score. The sign is reported raw and is never re-oriented."
    ),
}

COMPARE_LIMITATION_STATEMENT = (
    "Descriptive evidence from one AOI, one season and one common cohort. It "
    "is an observational predictive comparison, not causal inference and not "
    "an operational forecasting claim."
)

COMPARE_ANALYSIS_CONTRACT: tuple[str, ...] = (
    "Only the predictor closure timing was changed; both window ends moved "
    "together so the window LENGTH is preserved.",
    "The label window was held fixed and is identical in every variant.",
    "One exact common cohort was held fixed across all six evaluations.",
    "One shared spatial-fold assignment was used by all six evaluations.",
    "The model family, feature registry, preprocessing, hyper-parameters and "
    "seeds were held fixed.",
    "The existing production MODIS seasonal policy was preserved.",
    "Reducer, QA and processing policy were held fixed; the closure date's "
    "interaction with the fixed production policy could change observation "
    "support and effective MODIS coverage.",
    "The analysis therefore measures the closure date together with its "
    "interaction with that fixed production policy, not the closure date in "
    "isolation.",
)

COMPARE_INTERPRETATION_LIMITS: tuple[str, ...] = (
    "Results apply to the Manavgat-2021-style common cohort of this analysis "
    "only; they are not automatically transferable to another AOI or fire "
    "season.",
    "These results are descriptive and do not establish an underlying "
    "mechanism.",
    "It is not an operational forecasting validation and supports no "
    "deployment or alerting claim.",
    "Where a bootstrap interval includes zero, the direction of the change is "
    "not resolved by these data; uncertainty remains.",
    "The compare stage produces no new model and no new uncertainty estimate. "
    "It summarises verified model-stage outputs.",
    "No global comparison-evidence criterion is preregistered in this analysis, "
    "so no such overall claim is made.",
)


# --- Paths --------------------------------------------------------------------
def compare_root(experiment_id: str, output_root: Optional[Path] = None) -> Path:
    return experiment_root(experiment_id, output_root) / COMPARE_ROOT_DIR


def compare_staging_root(experiment_id: str, output_root: Optional[Path] = None) -> Path:
    return experiment_root(experiment_id, output_root) / COMPARE_STAGING_DIR


def compare_metadata_path(experiment_id: str, output_root: Optional[Path] = None) -> Path:
    return compare_root(experiment_id, output_root) / COMPARE_METADATA_NAME


COMPARE_STAGE_OWNED_NAMES: tuple[str, ...] = (COMPARE_ROOT_DIR,)


def compare_relative_layout() -> dict[str, str]:
    """Every file the compare stage owns, relative to `compare/`."""
    return dict(sorted({
        "point_metrics_long": "tables/point_metrics_long.csv",
        "point_metrics_wide": "tables/point_metrics_wide.csv",
        "thermal_contributions": "tables/thermal_contributions.csv",
        "closure_changes": "tables/closure_changes.csv",
        "thermal_contribution_changes": "tables/thermal_contribution_changes.csv",
        "bootstrap_evidence_matrix": "tables/bootstrap_evidence_matrix.csv",
        "comparison_summary": "summaries/comparison_summary.json",
        "scientific_conclusions": "summaries/scientific_conclusions.json",
        "provenance_summary": "summaries/provenance_summary.json",
        "report": "report/window_closure_comparison.md",
        "metadata": COMPARE_METADATA_NAME,
    }.items()))


def snapshot_compare_state(
    experiment_id: str, output_root: Optional[Path] = None,
) -> dict:
    """Read-only inventory of the compare stage-owned tree. No writes, no mkdir."""
    experiment = experiment_root(experiment_id, output_root)
    directories: set[str] = set()
    files: dict[str, dict] = {}
    root = compare_root(experiment_id, output_root)
    if root.exists():
        directories.add(_relative_label(root, experiment))
        for path in sorted(root.rglob("*")):
            label = _relative_label(path, experiment)
            if path.is_dir():
                directories.add(label)
            elif path.is_file():
                files[label] = {
                    "relative_path": label, "path": str(path),
                    "bytes": int(path.stat().st_size), "sha256": sha256_file(path),
                }
    content_view = {
        "directories": sorted(directories),
        "files": {
            label: {"bytes": record["bytes"], "sha256": record["sha256"]}
            for label, record in sorted(files.items())
        },
    }
    return {
        "experiment_root": str(experiment),
        "stage_owned_names": list(COMPARE_STAGE_OWNED_NAMES),
        "stage_owned_roots": {COMPARE_ROOT_DIR: str(root)},
        "directories": sorted(directories),
        "directory_count": len(directories),
        "files": dict(sorted(files.items())),
        "file_count": len(files),
        "digest": sha256_bytes(canonical_json(content_view).encode("utf-8")),
    }


# --- Expected cardinalities (derived, never hard-coded) -----------------------
def compare_expected_cardinalities(variant_ids: Sequence[str]) -> dict[str, int]:
    variants = len(variant_ids)
    earlier = len([v for v in variant_ids if v != CANONICAL_VARIANT_ID])
    families, metrics = len(MODEL_FAMILIES), len(MODEL_METRICS)
    contribution = variants * metrics
    closure = earlier * families * metrics
    contribution_change = earlier * metrics
    return {
        "point_metrics": variants * families * metrics,
        "thermal_contributions": contribution,
        "closure_changes": closure,
        "thermal_contribution_changes": contribution_change,
        "bootstrap_evidence_matrix": contribution + closure + contribution_change,
    }


def compare_variant_order(variant_ids: Sequence[str]) -> list[str]:
    """canonical first, then earlier closures by increasing shift."""
    earlier = sorted(
        (v for v in variant_ids if v != CANONICAL_VARIANT_ID),
        key=lambda name: int(str(name).split("_")[1].rstrip("d")),
    )
    return [CANONICAL_VARIANT_ID, *earlier]


# --- Wording ------------------------------------------------------------------
def assert_compare_wording(payload: Any, where: str) -> None:
    """Refuse significance-, equivalence- or stability-style PROSE.

    Only prose is inspected. Machine-readable key names (e.g.
    `frozen_hashes_unchanged`) are structural identifiers, not claims, and are
    never routed through this check -- the same distinction the reducer audit
    already makes between product fields and documentation.
    """
    text = payload if isinstance(payload, str) else " ".join(
        _collect_prose(payload)
    )
    lowered = text.lower()
    found = sorted(phrase for phrase in FORBIDDEN_COMPARE_PHRASES if phrase in lowered)
    if found:
        raise WindowClosureError(
            f"{where} carries forbidden wording {found}. A bootstrap interval "
            "that includes zero means the direction is unresolved; it is not "
            "evidence of no effect, of identical performance or of a stable "
            "result, and no significance test was performed."
        )


_PROSE_FIELD_TOKENS: tuple[str, ...] = (
    "prose", "statement", "note", "limitation", "interpretation", "contract",
)


def _collect_prose(payload: Any, prose_context: bool = False) -> list[str]:
    """Collect user-facing prose without treating paths/IDs as claims."""
    prose: list[str] = []
    if isinstance(payload, str):
        if prose_context:
            prose.append(payload)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            child_context = prose_context or any(
                token in str(key).lower() for token in _PROSE_FIELD_TOKENS
            )
            prose.extend(_collect_prose(value, child_context))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            prose.extend(_collect_prose(value, prose_context))
    return prose


def evidence_statement(metric: str, status: str) -> str:
    """Directional evidence wording. Never 'better'/'worse', never significance."""
    if metric == "brier":
        if status == INTERVAL_SUPPORTED_INCREASE:
            return "The interval supports a higher Brier score."
        if status == INTERVAL_SUPPORTED_DECREASE:
            return "The interval supports a lower Brier score."
        return (
            "The interval includes zero; uncertainty remains about the "
            "direction of the metric change."
        )
    if status == INTERVAL_SUPPORTED_INCREASE:
        return "The interval supports a higher metric value."
    if status == INTERVAL_SUPPORTED_DECREASE:
        return "The interval supports a lower metric value."
    return (
        "The interval includes zero; uncertainty remains about the direction "
        "of the metric change."
    )


# --- Binding ------------------------------------------------------------------
MODEL_STAGE_REQUIRED_FLAGS: dict[str, bool] = {
    "model_fit": True,
    "fire_risk_model_fit": True,
    "bootstrap_run": True,
    "common_cohort_created": True,
    "compare_run": False,
    "downscaling_model_refit": False,
    "gee_queries_run": False,
    "gee_exports_run": False,
    "canonical_outputs_modified": False,
    "upstream_outputs_modified": False,
    "frozen_hashes_unchanged": True,
}


def read_model_stage_metadata(
    experiment_id: str, output_root: Optional[Path] = None,
) -> dict:
    path = model_metadata_path(experiment_id, output_root)
    if not path.is_file():
        raise WindowClosureError(
            f"Model stage metadata is missing at {path}. The compare stage "
            "binds to a completed model stage; run --from-stage model "
            "--to-stage model first. Nothing was created."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise WindowClosureError(
            f"Model stage metadata at {path} is unreadable: {exc}."
        ) from exc
    if not isinstance(payload, dict):
        raise WindowClosureError("Model stage metadata is not a JSON object.")
    return payload


def assert_compare_binding(
    experiment_id: str,
    analysis_id: str,
    shifts: Sequence[int],
    canonical: dict,
    variants: Sequence[dict],
    censor: dict,
    inventory: dict,
    planned_paths: dict[str, str],
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
) -> dict:
    """Bind to the verified model stage. Read-only, before any write."""
    binding = assert_model_binding(
        experiment_id, analysis_id, shifts, canonical, variants, censor,
        inventory, planned_paths, output_root, experiments_root,
    )

    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise WindowClosureError(
                f"Compare binding failed: {message} No directory was created "
                "and nothing was written."
            )

    metadata = read_model_stage_metadata(experiment_id, output_root)
    metadata_path = model_metadata_path(experiment_id, output_root)
    _require(metadata.get("schema_version") == MODEL_METADATA_SCHEMA,
             f"model schema is {metadata.get('schema_version')!r}, expected "
             f"{MODEL_METADATA_SCHEMA!r}.")
    _require(metadata.get("status") == STATUS_PASS,
             f"model status is {metadata.get('status')!r}, not {STATUS_PASS!r}.")
    _require(metadata.get("analysis_id") == analysis_id,
             f"model analysis_id is {metadata.get('analysis_id')!r}.")
    _require(metadata.get("experiment_id") == experiment_id,
             f"model experiment_id is {metadata.get('experiment_id')!r}.")
    for flag, expected in MODEL_STAGE_REQUIRED_FLAGS.items():
        _require(metadata.get(flag) is expected,
                 f"model metadata {flag}={metadata.get(flag)!r}, expected {expected!r}.")

    root = model_root(experiment_id, output_root)
    layout = model_relative_layout()
    required = [layout[key] for key in sorted(layout) if key != "metadata"]
    recorded = {
        str(record.get("relative_path")): record
        for record in (metadata.get("artifact_inventory") or [])
    }
    for relative in required:
        _require(relative in recorded,
                 f"model metadata records no artefact for '{relative}'.")
    variant_ids = compare_variant_order(sorted(metadata.get("variant_ids") or []))
    for variant_id in variant_ids:
        for family in MODEL_FAMILIES:
            relative = model_variant_oof_relpath(variant_id, family)
            _require(relative in recorded,
                     f"model metadata records no artefact for '{relative}'.")

    artifacts: dict[str, dict] = {}
    for relative, record in sorted(recorded.items()):
        path = Path(str(record.get("path") or ""))
        _require(path.is_file(), f"model artefact '{relative}' is missing at {path}.")
        digest = sha256_file(path)
        _require(digest == record.get("sha256"),
                 f"model artefact '{relative}' hashes {digest} but the model "
                 f"metadata recorded {record.get('sha256')!r}.")
        _require(root in path.resolve().parents,
                 f"model artefact '{relative}' lies outside {root}.")
        artifacts[relative] = {
            "relative_path": relative, "path": str(path), "sha256": digest,
            "bytes": int(path.stat().st_size),
        }

    return {
        **binding,
        "bound_to_model": True,
        "model_stage_metadata": metadata,
        "model_stage_metadata_path": str(metadata_path),
        "model_stage_metadata_sha256": sha256_file(metadata_path),
        "model_artifacts": artifacts,
        "model_variant_ids": variant_ids,
    }


def compare_frozen_inputs(
    experiment_id: str, inventory: dict, variants: Sequence[dict],
    output_root: Optional[Path] = None, experiments_root: Optional[Path] = None,
) -> dict:
    """Everything compare must not disturb, including the whole model tree."""
    extended = dict(model_frozen_inputs(
        experiment_id, inventory, variants, output_root, experiments_root,
    ))
    metadata_path = model_metadata_path(experiment_id, output_root)
    extended["model_stage_metadata"] = {
        "path": str(metadata_path),
        "exists": metadata_path.is_file(),
        "sha256": sha256_file(metadata_path) if metadata_path.is_file() else None,
    }
    root = model_root(experiment_id, output_root)
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                extended[f"model_artifact__{path.relative_to(root).as_posix()}"] = {
                    "path": str(path), "exists": True, "sha256": sha256_file(path),
                }
    return extended


# --- Read-only re-derivation --------------------------------------------------
def recompute_compare_evidence(
    experiment_id: str, binding: dict, output_root: Optional[Path] = None,
) -> dict:
    """Re-derive every reported number from the saved model artefacts.

    Nothing is fitted, drawn or re-sampled: the out-of-fold predictions and the
    bootstrap replicate table are READ and the published values are checked
    against them. A mismatch fails the stage.
    """
    import numpy as np
    import pandas as pd

    from src.step8b_train_baseline_vs_thermal_model import compute_binary_metrics

    metadata = binding["model_stage_metadata"]
    root = model_root(experiment_id, output_root)
    variant_ids = binding["model_variant_ids"]
    expected = compare_expected_cardinalities(variant_ids)

    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise WindowClosureError(f"Compare re-derivation failed: {message}")

    # --- Point metrics ------------------------------------------------------
    point_rows: list[dict] = []
    recorded_points = {
        (row["variant_id"], row["model_family"]): row
        for row in (metadata.get("point_metrics") or [])
    }
    for variant_id in variant_ids:
        for family in MODEL_FAMILIES:
            recorded = recorded_points.get((variant_id, family))
            _require(recorded is not None,
                     f"the model stage reports no point metric for "
                     f"{variant_id}/{family}.")
            table = pd.read_parquet(
                root / model_variant_oof_relpath(variant_id, family)
            ).sort_values("cell_id", kind="mergesort")
            recomputed = compute_binary_metrics(
                table["y_true"].astype(int).to_numpy(),
                table["y_score"].to_numpy(dtype="float64"),
            )
            for metric, key in (("roc_auc", "roc_auc"), ("pr_auc", "pr_auc"),
                                ("brier", "brier_score")):
                value, reference = recorded.get(metric), recomputed[key]
                _require(
                    value is not None and reference is not None
                    and abs(float(value) - float(reference)) <= 1e-9,
                    f"{variant_id}/{family}/{metric} does not recompute from "
                    f"the saved out-of-fold predictions "
                    f"({value!r} vs {reference!r}).",
                )
                point_rows.append({
                    "variant_id": variant_id, "model_family": family,
                    "metric": metric, "value": float(reference),
                    "row_count": int(len(table)),
                    "positive_count": int(recorded["positive_count"]),
                    "negative_count": int(recorded["negative_count"]),
                    "prevalence": float(recorded["prevalence"]),
                    "fold_count": int(recorded["fold_count"]),
                    "metric_direction_note": METRIC_DIRECTION_NOTES[metric],
                })
    _require(len(point_rows) == expected["point_metrics"],
             f"expected {expected['point_metrics']} point-metric rows, derived "
             f"{len(point_rows)}.")

    point_of = {
        (row["variant_id"], row["model_family"], row["metric"]): row["value"]
        for row in point_rows
    }

    # --- Thermal contribution: thermal - baseline ---------------------------
    contribution_rows: list[dict] = []
    for variant_id in variant_ids:
        for metric in MODEL_METRICS:
            baseline = point_of[(variant_id, "baseline", metric)]
            thermal = point_of[(variant_id, "thermal", metric)]
            contribution_rows.append({
                "variant_id": variant_id, "metric": metric,
                "baseline": baseline, "thermal": thermal,
                "contribution_delta": thermal - baseline,
                "raw_delta_definition": "thermal - baseline (raw)",
                "metric_direction_note": METRIC_DIRECTION_NOTES[metric],
            })
    _require(len(contribution_rows) == expected["thermal_contributions"],
             f"expected {expected['thermal_contributions']} contribution rows.")
    contribution_of = {
        (row["variant_id"], row["metric"]): row["contribution_delta"]
        for row in contribution_rows
    }

    # --- Closure change: earlier - canonical --------------------------------
    closure_rows: list[dict] = []
    for variant_id in variant_ids:
        if variant_id == CANONICAL_VARIANT_ID:
            continue
        for family in MODEL_FAMILIES:
            for metric in MODEL_METRICS:
                earlier = point_of[(variant_id, family, metric)]
                reference = point_of[(CANONICAL_VARIANT_ID, family, metric)]
                closure_rows.append({
                    "variant_id": variant_id,
                    "reference_variant_id": CANONICAL_VARIANT_ID,
                    "model_family": family, "metric": metric,
                    "variant_value": earlier, "reference_value": reference,
                    "closure_delta": earlier - reference,
                    "raw_delta_definition": "earlier_closure - canonical (raw)",
                    "metric_direction_note": METRIC_DIRECTION_NOTES[metric],
                })
    _require(len(closure_rows) == expected["closure_changes"],
             f"expected {expected['closure_changes']} closure-change rows.")

    # --- Contribution change ------------------------------------------------
    contribution_change_rows: list[dict] = []
    for variant_id in variant_ids:
        if variant_id == CANONICAL_VARIANT_ID:
            continue
        for metric in MODEL_METRICS:
            earlier = contribution_of[(variant_id, metric)]
            reference = contribution_of[(CANONICAL_VARIANT_ID, metric)]
            contribution_change_rows.append({
                "variant_id": variant_id,
                "reference_variant_id": CANONICAL_VARIANT_ID,
                "metric": metric,
                "variant_contribution": earlier,
                "reference_contribution": reference,
                "contribution_change_delta": earlier - reference,
                "raw_delta_definition": (
                    "(thermal - baseline)_earlier - (thermal - baseline)_canonical"
                ),
                "metric_direction_note": METRIC_DIRECTION_NOTES[metric],
            })
    _require(
        len(contribution_change_rows) == expected["thermal_contribution_changes"],
        f"expected {expected['thermal_contribution_changes']} contribution-change rows.",
    )

    # --- Bootstrap evidence matrix ------------------------------------------
    replicates = pd.read_parquet(
        root / model_relative_layout()["bootstrap_replicates"]
    )
    bootstrap_meta = metadata.get("bootstrap") or {}
    requested = bootstrap_meta.get("n_bootstrap_requested")
    try:
        global_valid, global_invalid = validate_saved_bootstrap_replicate_counts(
            requested,
            bootstrap_meta.get("n_bootstrap_valid"),
            metadata.get("bootstrap_invalid_replicates"),
            len(replicates),
        )
    except WindowClosureError as exc:
        raise WindowClosureError(f"Compare re-derivation failed: {exc}") from exc
    requested = int(requested)
    _require(len(replicates) >= 1,
             "no valid bootstrap replicate is available; the compare stage "
             "publishes no interval without one.")

    point_delta_of = {
        (COMPARISON_THERMAL_CONTRIBUTION, row["variant_id"],
         "thermal_minus_baseline", row["metric"]): row["contribution_delta"]
        for row in contribution_rows
    }
    point_delta_of.update({
        (COMPARISON_CLOSURE_CHANGE, row["variant_id"], row["model_family"],
         row["metric"]): row["closure_delta"]
        for row in closure_rows
    })

    def _replicate_interval(row: dict) -> dict:
        """Reuse the model-stage expressions and finite-value count semantics."""
        variant_id = row["variant_id"]
        metric = row["metric"]
        family = row["model_family"]
        comparison = row["comparison"]
        if comparison == COMPARISON_THERMAL_CONTRIBUTION:
            values = replicates[f"{variant_id}__{_CONTRIBUTION_COLUMN[metric]}"]
        elif comparison == COMPARISON_CLOSURE_CHANGE:
            suffix = _FAMILY_METRIC_COLUMN[(family, metric)]
            values = (
                replicates[f"{variant_id}__{suffix}"]
                - replicates[f"{CANONICAL_VARIANT_ID}__{suffix}"]
            )
        else:
            suffix = _CONTRIBUTION_COLUMN[metric]
            values = (
                replicates[f"{variant_id}__{suffix}"]
                - replicates[f"{CANONICAL_VARIANT_ID}__{suffix}"]
            )
        return percentile_interval(
            values.tolist(),
            float(bootstrap_meta["ci_lower_percentile"]),
            float(bootstrap_meta["ci_upper_percentile"]),
        )
    point_delta_of.update({
        (COMPARISON_CONTRIBUTION_CHANGE, row["variant_id"],
         "thermal_minus_baseline", row["metric"]): row["contribution_change_delta"]
        for row in contribution_change_rows
    })

    evidence_rows: list[dict] = []
    seen: set = set()
    for row in (metadata.get("comparisons") or []):
        key = (row.get("comparison"), row.get("variant_id"),
               row.get("model_family"), row.get("metric"))
        _require(key not in seen, f"duplicate bootstrap comparison row {key}.")
        seen.add(key)
        _require(row.get("comparison") in COMPARE_FAMILY_ORDER,
                 f"unknown comparison family {row.get('comparison')!r}.")
        _require(row.get("metric") in MODEL_METRICS,
                 f"unknown metric {row.get('metric')!r}.")
        _require(row.get("variant_id") in variant_ids,
                 f"unknown variant {row.get('variant_id')!r}.")
        _require(
            row.get("model_family") in set(MODEL_FAMILIES) | {"thermal_minus_baseline"},
            f"unknown model family {row.get('model_family')!r}.",
        )
        expected_delta = point_delta_of.get(key)
        _require(expected_delta is not None,
                 f"the model stage reports a comparison {key} the compare "
                 "stage cannot re-derive.")
        _require(
            row.get("point_delta") is not None
            and abs(float(row["point_delta"]) - float(expected_delta)) <= 1e-9,
            f"comparison {key} point delta {row.get('point_delta')!r} does not "
            f"re-derive ({expected_delta!r}).",
        )
        interval = _replicate_interval(row)
        for field in ("bootstrap_mean", "ci_low", "ci_high"):
            recorded_value = row.get(field)
            recomputed_value = interval["point" if field == "bootstrap_mean" else field]
            _require(
                (recorded_value is None and recomputed_value is None)
                or (
                    recorded_value is not None and recomputed_value is not None
                    and abs(float(recorded_value) - float(recomputed_value)) <= 1e-9
                ),
                f"comparison {key} {field} does not recompute from its saved "
                "replicate columns.",
            )
        status = classify_change_interval(interval["ci_low"], interval["ci_high"])
        _require(row.get("status") == status,
                 f"comparison {key} status {row.get('status')!r} does not "
                 f"follow from its interval ({status!r}).")
        try:
            valid, invalid = validate_saved_bootstrap_replicate_counts(
                row.get("requested_replicates"), row.get("valid_replicates"),
                row.get("invalid_replicates"), len(replicates),
            )
        except WindowClosureError as exc:
            raise WindowClosureError(
                f"Compare re-derivation failed: comparison {key}: {exc}"
            ) from exc
        _require(
            requested == int(row["requested_replicates"])
            and valid == global_valid and invalid == global_invalid,
            f"comparison {key} replicate counts differ from the shared "
            "bootstrap counts.",
        )
        _require(
            int(interval["n_replicates"]) == global_valid,
            f"comparison {key} does not contain one finite metric value per "
            "saved shared draw.",
        )
        reference = (
            CANONICAL_VARIANT_ID
            if row["comparison"] != COMPARISON_THERMAL_CONTRIBUTION else None
        )
        evidence_rows.append({
            "comparison_family": row["comparison"],
            "variant_id": row["variant_id"],
            "reference_variant_id": reference,
            "model_family": row["model_family"],
            "metric": row["metric"],
            "point_delta": float(row["point_delta"]),
            "ci_low": row.get("ci_low"),
            "ci_high": row.get("ci_high"),
            "confidence_level": row.get("confidence_level"),
            "requested_replicates": requested,
            "valid_replicates": valid,
            "invalid_replicates": invalid,
            "status": status,
            "raw_delta_definition": row.get("delta_definition"),
            "metric_direction_note": METRIC_DIRECTION_NOTES[row["metric"]],
            "evidence_statement": evidence_statement(row["metric"], status),
            "limitation_statement": COMPARE_LIMITATION_STATEMENT,
        })

    _require(
        len(evidence_rows) == expected["bootstrap_evidence_matrix"],
        f"expected {expected['bootstrap_evidence_matrix']} bootstrap evidence "
        f"rows, derived {len(evidence_rows)}.",
    )
    _require(
        seen == set(point_delta_of),
        "bootstrap comparison groups do not exactly match the expected "
        "comparison keys.",
    )
    families = {row["comparison_family"] for row in evidence_rows}
    _require(families == set(COMPARE_FAMILY_ORDER),
             f"the evidence matrix is missing comparison family/families "
             f"{sorted(set(COMPARE_FAMILY_ORDER) - families)}.")

    order = {name: index for index, name in enumerate(COMPARE_FAMILY_ORDER)}
    variant_rank = {name: index for index, name in enumerate(variant_ids)}
    family_rank = {name: index for index, name in enumerate(MODEL_FAMILIES)}
    family_rank["thermal_minus_baseline"] = len(MODEL_FAMILIES)
    metric_rank = {name: index for index, name in enumerate(MODEL_METRICS)}
    evidence_rows.sort(key=lambda row: (
        order[row["comparison_family"]], variant_rank[row["variant_id"]],
        family_rank[row["model_family"]], metric_rank[row["metric"]],
    ))
    point_rows.sort(key=lambda row: (
        variant_rank[row["variant_id"]], family_rank[row["model_family"]],
        metric_rank[row["metric"]],
    ))
    for rows in (contribution_rows, contribution_change_rows):
        rows.sort(key=lambda row: (
            variant_rank[row["variant_id"]], metric_rank[row["metric"]],
        ))
    closure_rows.sort(key=lambda row: (
        variant_rank[row["variant_id"]], family_rank[row["model_family"]],
        metric_rank[row["metric"]],
    ))

    wide_rows = [
        {
            "variant_id": variant_id, "metric": metric,
            "baseline": point_of[(variant_id, "baseline", metric)],
            "thermal": point_of[(variant_id, "thermal", metric)],
            "thermal_contribution_raw_delta": contribution_of[(variant_id, metric)],
            "raw_delta_definition": "thermal - baseline (raw)",
            "metric_direction_note": METRIC_DIRECTION_NOTES[metric],
        }
        for variant_id in variant_ids for metric in MODEL_METRICS
    ]
    return {
        "variant_ids": variant_ids,
        "expected_cardinalities": expected,
        "point_metrics_long": point_rows,
        "point_metrics_wide": wide_rows,
        "thermal_contributions": contribution_rows,
        "closure_changes": closure_rows,
        "thermal_contribution_changes": contribution_change_rows,
        "bootstrap_evidence_matrix": evidence_rows,
        "bootstrap_recomputed": False,
        "models_refit": False,
        "replicates_regenerated": False,
    }


# --- Scientific synthesis (no single overall verdict) -------------------------
def build_compare_conclusions(evidence_rows: Sequence[dict]) -> dict:
    """Evidence grouped BY METRIC and BY COMPARISON FAMILY. No majority vote."""
    statuses = [INTERVAL_SUPPORTED_INCREASE, INTERVAL_SUPPORTED_DECREASE,
                INTERVAL_INCLUDES_ZERO]

    def _counts(rows: Sequence[dict]) -> dict:
        return {status: sum(1 for row in rows if row["status"] == status)
                for status in statuses}

    by_metric = {
        metric: {
            "metric": metric,
            "metric_direction_note": METRIC_DIRECTION_NOTES[metric],
            "row_count": len([r for r in evidence_rows if r["metric"] == metric]),
            "evidence_counts": _counts([r for r in evidence_rows if r["metric"] == metric]),
            "rows": [
                {
                    "comparison_family": r["comparison_family"],
                    "variant_id": r["variant_id"],
                    "model_family": r["model_family"],
                    "status": r["status"],
                    "evidence_statement": r["evidence_statement"],
                }
                for r in evidence_rows if r["metric"] == metric
            ],
        }
        for metric in MODEL_METRICS
    }
    by_family = {
        family: {
            "comparison_family": family,
            "row_count": len([r for r in evidence_rows if r["comparison_family"] == family]),
            "evidence_counts": _counts(
                [r for r in evidence_rows if r["comparison_family"] == family]
            ),
            "rows": [
                {
                    "variant_id": r["variant_id"], "model_family": r["model_family"],
                    "metric": r["metric"], "status": r["status"],
                    "evidence_statement": r["evidence_statement"],
                }
                for r in evidence_rows if r["comparison_family"] == family
            ],
        }
        for family in COMPARE_FAMILY_ORDER
    }
    return {
        "schema_version": COMPARE_METADATA_SCHEMA,
        # Technical artefact validation is SEPARATE from the evidence; it is
        # never combined into one scientific verdict.
        "technical_validation_status": STATUS_PASS,
        "single_overall_scientific_verdict_produced": False,
        "majority_vote_across_metrics_taken": False,
        "evidence_counts": _counts(evidence_rows),
        "conclusions_by_metric": by_metric,
        "conclusions_by_comparison_family": by_family,
        "allowed_statuses": list(statuses),
        "interpretation_limits": list(COMPARE_INTERPRETATION_LIMITS),
        "analysis_contract": list(COMPARE_ANALYSIS_CONTRACT),
    }


def _display(value, decimals: int = COMPARE_DISPLAY_DECIMALS) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{decimals}f}"


def render_compare_markdown(
    experiment_id: str, analysis_id: str, derived: dict, conclusions: dict,
    cohort: dict, folds: dict, provenance: dict,
) -> str:
    """Human-readable synthesis. Display rounding here only."""
    lines: list[str] = []
    add = lines.append
    add(f"# Window-closure comparison — `{experiment_id}`")
    add("")
    add(f"- Schema: `{COMPARE_METADATA_SCHEMA}`")
    add(f"- analysis_id: `{analysis_id}`")
    add(f"- Primary population: `{PRIMARY_POPULATION}`")
    add(f"- Display rounding: {COMPARE_DISPLAY_DECIMALS} decimals. The CSV and "
        "JSON artefacts carry full precision.")
    add("")

    add("## 1. Analysis contract")
    add("")
    for item in COMPARE_ANALYSIS_CONTRACT:
        add(f"- {item}")
    add("")

    add("## 2. Common cohort and shared folds")
    add("")
    add(f"- Common cohort rows: **{cohort.get('final_common_cohort_rows')}**")
    add(f"- Positives / negatives: **{cohort.get('final_positive_rows')}** / "
        f"**{cohort.get('final_negative_rows')}**")
    add(f"- Prevalence: **{_display(cohort.get('prevalence'))}**")
    add(f"- Shared folds: **{folds.get('fold_count')}**, spatial blocks: "
        f"**{folds.get('unique_block_count')}**")
    add("- All six evaluations used these exact cells and this exact fold "
        "assignment.")
    add("")

    add("## 3. Point metrics")
    add("")
    add("| Variant | Metric | Baseline | Thermal | Thermal contribution (raw) |")
    add("|---|---|---:|---:|---:|")
    for row in derived["point_metrics_wide"]:
        add(f"| {row['variant_id']} | {row['metric']} | "
            f"{_display(row['baseline'])} | {_display(row['thermal'])} | "
            f"{_display(row['thermal_contribution_raw_delta'])} |")
    add("")

    add("## 4. Thermal contributions")
    add("")
    add("Raw `thermal - baseline` per variant and metric. Negative raw deltas "
        "indicate lower Brier scores.")
    add("")
    add("| Variant | Metric | Raw delta |")
    add("|---|---|---:|")
    for row in derived["thermal_contributions"]:
        add(f"| {row['variant_id']} | {row['metric']} | "
            f"{_display(row['contribution_delta'])} |")
    add("")

    add("## 5. Predictor-closure changes")
    add("")
    add("Raw `earlier_closure - canonical` per model family and metric. "
        "Negative raw deltas indicate lower Brier scores.")
    add("")
    add("| Variant | Model family | Metric | Raw delta |")
    add("|---|---|---|---:|")
    for row in derived["closure_changes"]:
        add(f"| {row['variant_id']} | {row['model_family']} | {row['metric']} | "
            f"{_display(row['closure_delta'])} |")
    add("")

    add("## 6. Thermal-contribution changes")
    add("")
    add("Raw `(thermal - baseline)_earlier - (thermal - baseline)_canonical`.")
    add("")
    add("| Variant | Metric | Raw delta |")
    add("|---|---|---:|")
    for row in derived["thermal_contribution_changes"]:
        add(f"| {row['variant_id']} | {row['metric']} | "
            f"{_display(row['contribution_change_delta'])} |")
    add("")

    add("## 7. Bootstrap evidence")
    add("")
    add("Paired spatial-block bootstrap on the model stage's own replicate "
        "draws. The compare stage recomputes no replicate.")
    add("")
    add("| Comparison | Variant | Model family | Metric | Delta | CI low | "
        "CI high | Valid | Status |")
    add("|---|---|---|---|---:|---:|---:|---:|---|")
    for row in derived["bootstrap_evidence_matrix"]:
        add(f"| {row['comparison_family']} | {row['variant_id']} | "
            f"{row['model_family']} | {row['metric']} | "
            f"{_display(row['point_delta'])} | {_display(row['ci_low'])} | "
            f"{_display(row['ci_high'])} | {row['valid_replicates']} | "
            f"{row['status']} |")
    add("")
    counts = conclusions["evidence_counts"]
    add(f"- bootstrap-supported increase: **{counts[INTERVAL_SUPPORTED_INCREASE]}**")
    add(f"- bootstrap-supported decrease: **{counts[INTERVAL_SUPPORTED_DECREASE]}**")
    add(f"- interval includes zero; uncertainty remains: "
        f"**{counts[INTERVAL_INCLUDES_ZERO]}**")
    add("")
    add("### Evidence by metric")
    add("")
    for metric in MODEL_METRICS:
        entry = conclusions["conclusions_by_metric"][metric]
        add(f"- `{metric}` — {entry['metric_direction_note']} "
            f"increase: {entry['evidence_counts'][INTERVAL_SUPPORTED_INCREASE]}, "
            f"decrease: {entry['evidence_counts'][INTERVAL_SUPPORTED_DECREASE]}, "
            f"interval includes zero: "
            f"{entry['evidence_counts'][INTERVAL_INCLUDES_ZERO]}.")
    add("")
    add("### Evidence by comparison family")
    add("")
    for family in COMPARE_FAMILY_ORDER:
        entry = conclusions["conclusions_by_comparison_family"][family]
        add(f"- `{family}` — "
            f"increase: {entry['evidence_counts'][INTERVAL_SUPPORTED_INCREASE]}, "
            f"decrease: {entry['evidence_counts'][INTERVAL_SUPPORTED_DECREASE]}, "
            f"interval includes zero: "
            f"{entry['evidence_counts'][INTERVAL_INCLUDES_ZERO]}.")
    add("")
    add("No single overall scientific verdict is produced and no majority vote "
        "is taken across metrics: each metric and each comparison family is "
        "reported on its own evidence.")
    add("")

    add("## 8. Interpretation limits")
    add("")
    for item in COMPARE_INTERPRETATION_LIMITS:
        add(f"- {item}")
    add("")

    add("## 9. Provenance and integrity")
    add("")
    add(f"- Model stage metadata: `{provenance['source_model_metadata_sha256']}`")
    add(f"- Bound model artefacts: **{provenance['input_artifact_count']}**")
    add("- The compare stage fitted no model, generated no out-of-fold "
        "prediction and drew no bootstrap replicate.")
    add("- No canonical, predictor, pre-label, local-downstream or model "
        "artefact was written by this stage.")
    add("")
    return "\n".join(lines)


# --- The stage ----------------------------------------------------------------
def compare_is_reusable(
    experiment_id: str, analysis_id: str, binding: dict,
    output_root: Optional[Path] = None,
) -> tuple[bool, Optional[dict], str]:
    path = compare_metadata_path(experiment_id, output_root)
    if not path.is_file():
        return False, None, "no compare stage metadata"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False, None, "unreadable compare stage metadata"
    if not isinstance(metadata, dict):
        return False, None, "compare stage metadata is not a JSON object"
    if metadata.get("schema_version") != COMPARE_METADATA_SCHEMA:
        return False, metadata, f"metadata schema is {metadata.get('schema_version')!r}"
    if metadata.get("analysis_id") != analysis_id:
        return False, metadata, "analysis_id mismatch"
    if metadata.get("status") != STATUS_PASS:
        return False, metadata, f"previous status is {metadata.get('status')!r}"
    if metadata.get("source_model_metadata_sha256") != binding["model_stage_metadata_sha256"]:
        return False, metadata, "the model stage metadata hash has changed"
    outputs = metadata.get("output_artifacts") or []
    if not outputs:
        return False, metadata, "the recorded output inventory is empty"
    for record in outputs:
        target = Path(str((record or {}).get("path") or ""))
        if not target.is_file():
            return False, metadata, f"missing artefact {record.get('relative_path')}"
        if sha256_file(target) != record.get("sha256"):
            return False, metadata, f"hash mismatch for {record.get('relative_path')}"
    return True, metadata, "complete and verified"


def _quarantine_compare_outputs(
    experiment_id: str, output_root: Optional[Path] = None,
) -> dict:
    """Move ONLY `compare/` aside. Nothing is ever deleted."""
    root = compare_root(experiment_id, output_root)
    if not root.exists():
        return {"quarantined": False, "entries": []}
    before = snapshot_compare_state(experiment_id, output_root)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    target_root = (
        experiment_root(experiment_id, output_root)
        / LOCAL_DOWNSTREAM_QUARANTINE_DIR / COMPARE_QUARANTINE_KIND / stamp
    )
    target_root.mkdir(parents=True, exist_ok=True)
    destination = target_root / COMPARE_ROOT_DIR
    os.replace(root, destination)
    return {
        "quarantined": True,
        "entries": [{
            "original_path": str(root),
            "quarantined_path": str(destination),
            "reason": "force rebuild of the compare stage",
            "timestamp_utc": stamp,
            "pre_quarantine_inventory_sha256": before["digest"],
            "pre_quarantine_file_count": before["file_count"],
        }],
    }


COMPARE_STAGE_SEMANTICS: dict[str, Any] = {
    # These flags describe the COMPARE stage only. The model-stage metadata
    # separately records that a model was fitted and a bootstrap was run.
    "model_fit": False,
    "fire_risk_model_fit": False,
    "downscaling_model_fit": False,
    "bootstrap_run": False,
    "bootstrap_recomputed": False,
    "compare_run": True,
    "gee_queries_run": False,
    "gee_exports_run": False,
    "canonical_outputs_modified": False,
    "upstream_outputs_modified": False,
}
COMPARE_DRY_RUN_SEMANTICS: dict[str, Any] = {
    "model_fit": False,
    "fire_risk_model_fit": False,
    "downscaling_model_fit": False,
    "bootstrap_run": False,
    "bootstrap_recomputed": False,
    "compare_run": False,
    "gee_queries_run": False,
    "gee_exports_run": False,
    "compare_run_planned": True,
    "model_refit_planned": False,
    "bootstrap_recompute_planned": False,
    "report_generation_planned": True,
    "tables_generation_planned": True,
}


def run_compare_stage(
    experiment_id: str,
    analysis_id: str,
    variants: Sequence[dict],
    binding: dict,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
    force: bool = False,
    resume: bool = False,
) -> dict:
    """Publish the compare tables and synthesis atomically. Read-only inputs."""
    import shutil

    root = compare_root(experiment_id, output_root)
    metadata_path = compare_metadata_path(experiment_id, output_root)

    reusable, previous, reason = compare_is_reusable(
        experiment_id, analysis_id, binding, output_root,
    )
    if reusable and not force:
        if not resume:
            raise WindowClosureError(
                f"The compare stage already has a complete, verified output at "
                f"{metadata_path}. Refusing to overwrite it silently: re-run "
                "with resume=True to reuse it, or force=True to rebuild it "
                "(the old compare/ tree is quarantined, never deleted)."
            )
        return {
            "reused": True, "reason": reason, "files_written": [],
            "quarantine": {"quarantined": False, "entries": []},
            "metadata": previous or {}, **COMPARE_STAGE_SEMANTICS,
        }
    if resume:
        raise WindowClosureError(
            f"--resume cannot reuse the compare stage: {reason}. --resume only "
            "ever REUSES a complete, verified status=pass output; it never "
            f"rebuilds one. Nothing was quarantined, moved, deleted or written "
            f"at {root}. Re-run with force=True to rebuild it."
        )
    if not force and root.exists():
        raise WindowClosureError(
            f"The compare stage has an existing but NOT reusable output "
            f"({reason}) at {root}. Refusing to overwrite it silently: inspect "
            "it, then re-run with force=True (the old compare/ tree is "
            "quarantined, never deleted)."
        )

    frozen_before = frozen_hash_map(compare_frozen_inputs(
        experiment_id, frozen_input_inventory(experiment_id, experiments_root),
        variants, output_root, experiments_root,
    ))

    model_metadata = binding["model_stage_metadata"]
    derived = recompute_compare_evidence(experiment_id, binding, output_root)
    conclusions = build_compare_conclusions(derived["bootstrap_evidence_matrix"])
    provenance = {
        "schema_version": COMPARE_METADATA_SCHEMA,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "source_model_schema": model_metadata.get("schema_version"),
        "source_model_metadata_path": binding["model_stage_metadata_path"],
        "source_model_metadata_sha256": binding["model_stage_metadata_sha256"],
        "input_artifacts": list(binding["model_artifacts"].values()),
        "input_artifact_count": len(binding["model_artifacts"]),
        "input_dataset_sha256": model_metadata.get("input_dataset_sha256"),
        "canonical_step8a_sha256": model_metadata.get("canonical_step8a_sha256"),
        **COMPARE_STAGE_SEMANTICS,
    }
    cohort = model_metadata.get("common_cohort") or {}
    folds = model_metadata.get("shared_folds") or {}

    layout = compare_relative_layout()
    staging = compare_staging_root(experiment_id, output_root)
    if staging.exists():
        shutil.rmtree(staging)
    written: list[dict] = []

    def _stage(relative: str, text: str) -> None:
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append({"relative_path": relative, "path": path})

    def _csv(relative: str, rows: Sequence[dict]) -> None:
        columns = sorted({key for row in rows for key in row})
        _stage(relative, _csv_document(columns, rows))

    try:
        summary = {
            "schema_version": COMPARE_METADATA_SCHEMA,
            "analysis_id": analysis_id,
            "experiment_id": experiment_id,
            "primary_population": PRIMARY_POPULATION,
            "variant_order": derived["variant_ids"],
            "model_family_order": list(MODEL_FAMILIES),
            "metric_order": list(MODEL_METRICS),
            "comparison_family_order": list(COMPARE_FAMILY_ORDER),
            "expected_cardinalities": derived["expected_cardinalities"],
            "metric_sign_conventions": METRIC_SIGN_CONVENTIONS,
            "metric_direction_notes": METRIC_DIRECTION_NOTES,
            "point_metrics_long": derived["point_metrics_long"],
            "point_metrics_wide": derived["point_metrics_wide"],
            "thermal_contributions": derived["thermal_contributions"],
            "closure_changes": derived["closure_changes"],
            "thermal_contribution_changes": derived["thermal_contribution_changes"],
            "bootstrap_evidence_matrix": derived["bootstrap_evidence_matrix"],
            "common_cohort": cohort,
            "shared_folds": folds,
            **COMPARE_STAGE_SEMANTICS,
        }
        for payload, where in (
            (summary, "compare comparison summary"),
            (conclusions, "compare scientific conclusions"),
            (provenance, "compare provenance summary"),
        ):
            assert_compare_wording(payload, where)

        _csv(layout["point_metrics_long"], derived["point_metrics_long"])
        _csv(layout["point_metrics_wide"], derived["point_metrics_wide"])
        _csv(layout["thermal_contributions"], derived["thermal_contributions"])
        _csv(layout["closure_changes"], derived["closure_changes"])
        _csv(layout["thermal_contribution_changes"],
             derived["thermal_contribution_changes"])
        _csv(layout["bootstrap_evidence_matrix"], derived["bootstrap_evidence_matrix"])
        _stage(layout["comparison_summary"], _json_document(summary))
        _stage(layout["scientific_conclusions"], _json_document(conclusions))
        _stage(layout["provenance_summary"], _json_document(provenance))

        report = render_compare_markdown(
            experiment_id, analysis_id, derived, conclusions, cohort, folds, provenance,
        )
        assert_compare_wording(report, "compare markdown report")
        assert_report_wording(report)
        assert_no_foreign_factor_wording(report, "compare markdown report")
        _stage(layout["report"], report)

        output_artifacts = [
            {
                "relative_path": record["relative_path"],
                "path": str(root / record["relative_path"]),
                "sha256": sha256_file(record["path"]),
                "bytes": int(record["path"].stat().st_size),
                "media_type": _media_type(record["path"]),
            }
            for record in sorted(written, key=lambda item: item["relative_path"])
        ]
        metadata = build_compare_stage_metadata(
            experiment_id, analysis_id, binding, derived, conclusions,
            provenance, output_artifacts, frozen_before, variants,
            output_root, experiments_root,
        )
        metadata_text = _json_document(metadata)
        assert_compare_wording(metadata, "compare stage metadata")
        (staging / layout["metadata"]).write_text(metadata_text, encoding="utf-8")
        output_artifacts.append({
            "relative_path": layout["metadata"],
            "path": str(root / layout["metadata"]),
            "sha256": sha256_bytes(metadata_text.encode("utf-8")),
            "bytes": len(metadata_text.encode("utf-8")),
            "media_type": "application/json",
        })

        assert_frozen_hashes_unchanged(
            frozen_before,
            frozen_hash_map(compare_frozen_inputs(
                experiment_id,
                frozen_input_inventory(experiment_id, experiments_root),
                variants, output_root, experiments_root,
            )),
            "while running the compare stage",
        )
        quarantine = _quarantine_compare_outputs(experiment_id, output_root)
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "reused": False,
        "reason": "produced",
        "files_written": sorted(record["path"] for record in output_artifacts),
        "quarantine": quarantine,
        "metadata": metadata,
        "output_artifacts": output_artifacts,
        **COMPARE_STAGE_SEMANTICS,
    }


def build_compare_stage_metadata(
    experiment_id: str,
    analysis_id: str,
    binding: dict,
    derived: dict,
    conclusions: dict,
    provenance: dict,
    output_artifacts: Sequence[dict],
    frozen_before: dict,
    variants: Sequence[dict],
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
) -> dict:
    frozen_after = frozen_hash_map(compare_frozen_inputs(
        experiment_id, frozen_input_inventory(experiment_id, experiments_root),
        variants, output_root, experiments_root,
    ))
    return {
        "schema_version": COMPARE_METADATA_SCHEMA,
        "status": STATUS_PASS,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "stage": COMPARE_STAGE,
        "primary_population": PRIMARY_POPULATION,
        "source_model_schema": provenance["source_model_schema"],
        "source_model_metadata_path": provenance["source_model_metadata_path"],
        "source_model_metadata_sha256": provenance["source_model_metadata_sha256"],
        "input_artifacts": list(binding["model_artifacts"].values()),
        "input_artifact_count": len(binding["model_artifacts"]),
        "output_artifacts": list(output_artifacts),
        "output_artifact_count": len(output_artifacts),
        "variant_order": derived["variant_ids"],
        "model_family_order": list(MODEL_FAMILIES),
        "metric_order": list(MODEL_METRICS),
        "comparison_family_order": list(COMPARE_FAMILY_ORDER),
        "expected_cardinalities": derived["expected_cardinalities"],
        "point_metric_row_count": len(derived["point_metrics_long"]),
        "thermal_contribution_row_count": len(derived["thermal_contributions"]),
        "closure_change_row_count": len(derived["closure_changes"]),
        "thermal_contribution_change_row_count": len(
            derived["thermal_contribution_changes"]
        ),
        "bootstrap_summary_row_count": len(derived["bootstrap_evidence_matrix"]),
        "evidence_status_counts": conclusions["evidence_counts"],
        "technical_validation_status": conclusions["technical_validation_status"],
        "single_overall_scientific_verdict_produced": False,
        "majority_vote_across_metrics_taken": False,
        "display_decimals": COMPARE_DISPLAY_DECIMALS,
        "machine_readable_values_rounded": False,
        "analysis_contract": list(COMPARE_ANALYSIS_CONTRACT),
        "interpretation_limits": list(COMPARE_INTERPRETATION_LIMITS),
        "allowed_statuses": [
            INTERVAL_SUPPORTED_INCREASE, INTERVAL_SUPPORTED_DECREASE,
            INTERVAL_INCLUDES_ZERO,
        ],
        **COMPARE_STAGE_SEMANTICS,
        "frozen_input_sha256_before": dict(frozen_before),
        "frozen_input_sha256_after": dict(frozen_after),
        "frozen_hashes_unchanged": frozen_before == frozen_after,
    }


def compare_stage_summary(
    experiment_id: str,
    analysis_id: str,
    variants: Sequence[dict],
    output_root: Optional[Path] = None,
) -> dict:
    """The compare plan, as reported by a dry run. Read-only."""
    metadata_path = model_metadata_path(experiment_id, output_root)
    metadata: dict = {}
    present = metadata_path.is_file()
    if present:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            metadata = {}
    variant_ids = compare_variant_order(
        sorted(metadata.get("variant_ids") or [
            v["variant_id"] for v in variants
        ])
    )
    expected = compare_expected_cardinalities(variant_ids)

    root = model_root(experiment_id, output_root)
    inputs: dict[str, dict] = {}
    ready = present and metadata.get("status") == STATUS_PASS
    for record in (metadata.get("artifact_inventory") or []):
        relative = str(record.get("relative_path"))
        path = Path(str(record.get("path") or ""))
        matches = path.is_file() and sha256_file(path) == record.get("sha256")
        ready = ready and matches
        inputs[relative] = {
            "relative_path": relative, "path": str(path),
            "present": path.is_file(), "sha256_matches": matches,
            "recorded_sha256": record.get("sha256"),
        }
    for flag, expected_value in MODEL_STAGE_REQUIRED_FLAGS.items():
        ready = ready and metadata.get(flag) is expected_value

    compare_dir = compare_root(experiment_id, output_root)
    planned = sorted(
        str(compare_dir / relative) for relative in compare_relative_layout().values()
    )
    experiment = experiment_root(experiment_id, output_root).resolve()
    contained = all(experiment in Path(path).resolve().parents for path in planned)
    return {
        "stage_root": str(compare_dir),
        "source_model_metadata_path": str(metadata_path),
        "source_model_metadata_present": present,
        "source_model_metadata_sha256": (
            sha256_file(metadata_path) if present else None
        ),
        "source_model_schema": metadata.get("schema_version"),
        "source_model_status": metadata.get("status"),
        "model_root": str(root),
        "input_artifacts": inputs,
        "input_artifact_count": len(inputs),
        "input_binding_ready": bool(ready),
        "variant_order": variant_ids,
        "model_family_order": list(MODEL_FAMILIES),
        "metric_order": list(MODEL_METRICS),
        "comparison_family_order": list(COMPARE_FAMILY_ORDER),
        "expected_cardinalities": expected,
        "allowed_statuses": [
            INTERVAL_SUPPORTED_INCREASE, INTERVAL_SUPPORTED_DECREASE,
            INTERVAL_INCLUDES_ZERO,
        ],
        "metric_direction_notes": METRIC_DIRECTION_NOTES,
        "raw_delta_definitions": {
            COMPARISON_THERMAL_CONTRIBUTION: "thermal - baseline (raw)",
            COMPARISON_CLOSURE_CHANGE: "earlier_closure - canonical (raw)",
            COMPARISON_CONTRIBUTION_CHANGE: (
                "(thermal - baseline)_earlier - (thermal - baseline)_canonical"
            ),
        },
        "planned_output_paths": planned,
        "all_paths_inside_compare_namespace": contained,
        "display_decimals": COMPARE_DISPLAY_DECIMALS,
        "machine_readable_values_rounded": False,
        **COMPARE_DRY_RUN_SEMANTICS,
        "interpretation_limits": list(COMPARE_INTERPRETATION_LIMITS),
    }


# =============================================================================
# Public entry point
# =============================================================================
def run_analysis(
    experiment_id: str,
    shifts: Optional[Iterable[int]] = DEFAULT_SHIFTS,
    from_stage: str = "plan",
    to_stage: str = "compare",
    dry_run: bool = False,
    force: bool = False,
    resume: bool = False,
    recover_partial_local_downstream: bool = False,
    output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None,
    prelabel_exporter: Optional[Any] = None,
    predictor_engine: Optional[Any] = None,
    local_downstream_engine: Optional[Any] = None,
    model_configuration_overrides: Optional[dict] = None,
) -> dict:
    """Plan (and, when explicitly staged, execute) the window-closure analysis.

    `output_root` / `experiments_root` are explicit dependency-injection points:
    None means the canonical diagnostics / outputs-experiments roots. Tests pass
    tmp_path rather than monkeypatching another module's globals.

    Live Earth Engine work happens ONLY when `prelabel-export` /
    `predictor-export` are explicitly inside the selected stage range and
    `dry_run` is False. `prelabel_exporter` and `predictor_engine` override the
    production exporters for tests, so no test ever reaches Earth Engine.
    """
    from core.experiment_context import build_experiment_context

    shift_values = normalize_shifts(shifts)
    stages = validate_stage_range(from_stage, to_stage)
    assert_stage_prerequisites(stages)
    assert_resume_force_exclusive(resume, force)

    base_context = build_experiment_context(experiment_id)
    canonical = canonical_window(base_context)
    variants = build_window_variants(base_context, shift_values)
    censor = common_prelabel_interval(variants)
    inventory = frozen_input_inventory(experiment_id, experiments_root, base_context)
    labels = resolve_label_inputs(experiment_id, experiments_root, base_context)
    prerequisites = label_prerequisites(inventory)
    actual_prerequisites = actual_plan_prerequisites(inventory)
    config = scientific_configuration(
        experiment_id, base_context, variants, censor, inventory, shift_values,
    )
    analysis_id = compute_analysis_id(config)

    window_days = canonical["current_period_days"]
    export_plans = {
        variant["variant_id"]: {
            "landsat": landsat_export_plan(variant, canonical["baseline_years"], window_days),
            "modis": modis_export_plan(variant, experiment_id, output_root),
            "static_shared": static_shared_plan(),
        }
        for variant in variants
    }
    planned_paths = plan_output_paths(experiment_id, variants, output_root)

    if dry_run:
        # Read-only: what stage-owned state ALREADY exists (e.g. a partial
        # downstream from an earlier, failed actual run). Bracketing the plan
        # with two snapshots is what proves the dry run touched nothing --
        # demanding an empty tree would be a false positive on any namespace
        # that has been run before.
        stage_owned_before = snapshot_local_downstream_state(
            experiment_id, variants, output_root,
        )
        model_state_before = snapshot_model_state(experiment_id, output_root)
        compare_state_before = snapshot_compare_state(experiment_id, output_root)
        result = {
            "ran": False,
            "dry_run": True,
            "experiment_id": experiment_id,
            "schema_version": SCHEMA_VERSION,
            "analysis_id": analysis_id,
            "canonical_window": {
                "predictor_start_date": canonical["predictor_start_date"],
                "predictor_end_date": canonical["predictor_end_date"],
                "duration_days": canonical["duration_days"],
                "lead_days": canonical["lead_days"],
            },
            "canonical_duration_days": canonical["duration_days"],
            "label_window": {
                "start_date": canonical["label_start_date"],
                "end_date": canonical["label_end_date"],
                "frozen_across_variants": True,
            },
            "shift_days": list(shift_values),
            "variants": variants,
            "duration_preserved": all(v["duration_preserved"] for v in variants),
            "label_window_unchanged": all(v["label_window_unchanged"] for v in variants),
            "common_prelabel_censor": censor,
            "prelabel_export_plan": prelabel_export_plan(experiment_id, censor, output_root),
            "baseline_windows_per_year": {
                variant["variant_id"]: [
                    role for role in export_plans[variant["variant_id"]]["landsat"]["roles"]
                    if role["scope"] == "baseline_year"
                ]
                for variant in variants
            },
            "landsat_export_roles": {
                vid: plan["landsat"]["roles"] for vid, plan in export_plans.items()
            },
            "modis_export_roles": {
                vid: plan["modis"]["roles"] for vid, plan in export_plans.items()
            },
            "static_shared_roles": list(STATIC_SHARED_ROLES),
            "calendar_month_filter_transparency": {
                vid: plan["landsat"]["calendar_month_filter_transparency"]
                for vid, plan in export_plans.items()
            },
            "frozen_canonical_step8a": {
                "path": str(canonical_step8a_path(experiment_id, experiments_root)),
                "sha256": inventory.get("canonical_step8a", {}).get("sha256"),
            },
            "label_inputs": {
                role: {
                    "role": role,
                    "path": str(entry["path"]),
                    "resolved_from": entry["resolved_from"],
                    "canonical_filename": entry["canonical_filename"],
                    "exists": inventory[role]["exists"],
                    "sha256": inventory[role]["sha256"],
                }
                for role, entry in sorted(labels.items())
            },
            "required_label_roles": list(REQUIRED_LABEL_ROLES),
            "prerequisites_ready": prerequisites["prerequisites_ready"],
            "missing_required_inputs": prerequisites["missing_required_inputs"],
            # The WIDER gate an actual run has to pass: the label-scoped fields
            # above are kept unchanged for byte-stability of the dry-run
            # contract, so the full requirement is reported alongside them.
            "actual_plan_prerequisites": {
                "ready": actual_prerequisites["prerequisites_ready"],
                "required_roles": list(REQUIRED_FROZEN_INPUT_ROLES),
                "missing_required_inputs": actual_prerequisites["missing_required_inputs"],
            },
            "frozen_input_inventory": inventory,
            "predictor_export_summary": predictor_export_summary(
                experiment_id, variants, canonical["baseline_years"],
                window_days, output_root,
            ),
            # Read-only: it hashes documents that already exist and creates
            # nothing. No production downstream helper is imported here.
            "local_downstream_summary": local_downstream_summary(
                experiment_id, analysis_id, canonical, variants, inventory,
                output_root, experiments_root,
            ),
            # Read-only as well: it hashes existing documents, imports no model
            # library and creates nothing.
            "model_stage_summary": model_stage_summary(
                experiment_id, analysis_id, canonical, variants, inventory,
                output_root, experiments_root,
            ),
            # Read-only too: it hashes existing model artefacts and creates
            # nothing. No model library and no bootstrap is touched.
            "compare_stage_summary": compare_stage_summary(
                experiment_id, analysis_id, variants, output_root,
            ),
            "planned_output_paths": planned_paths,
            "planned_stages": stages,
            "scientific_configuration": config,
            "files_written": False,
            "gee_queries_run": False,
            "gee_exports_run": False,
            "model_fit": False,
            "bootstrap_run": False,
            "limitations": list(LIMITATIONS),
        }
        # ...and again once the whole plan has been built. Any created,
        # modified or deleted stage-owned path is reported here rather than
        # inferred from an emptiness assumption.
        result.update(local_downstream_state_diff(
            stage_owned_before,
            snapshot_local_downstream_state(experiment_id, variants, output_root),
        ))
        # The SAME read-only before/after mechanism, applied to the model
        # stage-owned tree and reported under its own explicit key names.
        model_diff = local_downstream_state_diff(
            model_state_before, snapshot_model_state(experiment_id, output_root),
        )
        result.update({
            "preexisting_model_stage_owned_paths":
                model_diff["preexisting_stage_owned_paths"],
            "model_stage_owned_snapshot_before":
                model_diff["stage_owned_snapshot_before"],
            "model_stage_owned_snapshot_after":
                model_diff["stage_owned_snapshot_after"],
            "model_stage_owned_snapshot_before_sha256":
                model_diff["stage_owned_snapshot_before_sha256"],
            "model_stage_owned_snapshot_after_sha256":
                model_diff["stage_owned_snapshot_after_sha256"],
            "model_stage_owned_snapshot_unchanged":
                model_diff["stage_owned_snapshot_unchanged"],
            "model_dry_run_created_paths": model_diff["dry_run_created_paths"],
            "model_dry_run_modified_paths": model_diff["dry_run_modified_paths"],
            "model_dry_run_deleted_paths": model_diff["dry_run_deleted_paths"],
        })
        compare_diff = local_downstream_state_diff(
            compare_state_before, snapshot_compare_state(experiment_id, output_root),
        )
        result.update({
            "preexisting_compare_stage_owned_paths":
                compare_diff["preexisting_stage_owned_paths"],
            "compare_stage_owned_snapshot_before":
                compare_diff["stage_owned_snapshot_before"],
            "compare_stage_owned_snapshot_after":
                compare_diff["stage_owned_snapshot_after"],
            "compare_stage_owned_snapshot_before_sha256":
                compare_diff["stage_owned_snapshot_before_sha256"],
            "compare_stage_owned_snapshot_after_sha256":
                compare_diff["stage_owned_snapshot_after_sha256"],
            "compare_stage_owned_snapshot_unchanged":
                compare_diff["stage_owned_snapshot_unchanged"],
            "compare_dry_run_created_paths": compare_diff["dry_run_created_paths"],
            "compare_dry_run_modified_paths": compare_diff["dry_run_modified_paths"],
            "compare_dry_run_deleted_paths": compare_diff["dry_run_deleted_paths"],
        })
        # A dry run must never carry another analysis's changed-factor wording.
        assert_no_foreign_factor_wording(result, "window-closure dry-run plan")
        return result

    # --- Actual (non-dry-run) work starts here -------------------------------
    # Every gate below runs BEFORE any directory or file is created.
    # 1. Only the planning stage is implemented for a real run.
    assert_actual_stages_supported(stages)
    # 2. A missing required label would be frozen into the analysis identity as
    #    a null hash, so it stops the plan with the most specific message.
    assert_label_prerequisites(inventory)
    # 3. ...and so would a missing frozen Step8A or static shared raster.
    prerequisite_status = assert_actual_plan_prerequisites(inventory)
    # 4. If the registry enables the pre-label exclusion policy, production
    #    Step8A will REQUIRE the Step6B gate exclusion manifest inside every
    #    variant's own gate_labels_dir. That is statically resolvable now, so
    #    it is asserted here -- on EVERY actual invocation, including the plan
    #    stage that precedes the export stage -- rather than after an Earth
    #    Engine export has already been paid for.
    censor_binding = assert_prelabel_exclusion_binding(
        prelabel_exclusion_binding(experiment_id, base_context, experiments_root),
        f"actual run preflight (stages {list(stages)})",
    )

    # Re-hash immediately before any write: the inventory that produced the
    # analysis_id must still describe what is on disk right now.
    hashes_before = frozen_hash_map(inventory)
    assert_frozen_hashes_unchanged(
        hashes_before,
        frozen_hash_map(frozen_input_inventory(experiment_id, experiments_root, base_context)),
        "between planning and writing",
    )

    files_written: list[str] = []
    files_rewritten: list[str] = []
    reused_flags: list[bool] = []
    plan_outcome: Optional[dict] = None
    prelabel_outcome: Optional[dict] = None

    if "plan" in stages:
        documents = build_plan_documents(
            experiment_id, analysis_id, config, canonical, variants, censor,
            inventory, labels, export_plans, planned_paths, output_root,
        )
        for relative, text in sorted(documents.items()):
            assert_no_foreign_factor_wording(text, f"plan document '{relative}'")
        plan_outcome = write_plan_documents(
            experiment_id, analysis_id, documents, planned_paths, output_root, force,
        )
        files_written += plan_outcome["files_written"]
        files_rewritten += plan_outcome["files_rewritten"]
        reused_flags.append(plan_outcome["reused"])

    binding: Optional[dict] = None
    if PRELABEL_STAGE in stages:
        # Bind to the plan documents ON DISK -- including the ones a 'plan'
        # stage in this same call has just verified or written. Every check
        # happens before Earth Engine is imported.
        binding = assert_plan_binding(
            experiment_id, analysis_id, shift_values, censor, inventory, planned_paths,
        )
        prelabel_outcome = run_prelabel_export(
            experiment_id, analysis_id, censor, inventory, binding,
            output_root=output_root, force=force, resume=resume,
            exporter=prelabel_exporter,
        )
        files_written += prelabel_outcome["files_written"]
        files_rewritten += prelabel_outcome["files_rewritten"]
        reused_flags.append(prelabel_outcome["reused"])

    predictor_outcome: Optional[dict] = None
    if PREDICTOR_STAGE in stages:
        # Binds to the completed plan AND prelabel stages, and verifies every
        # variant/namespace contract, before Earth Engine is imported.
        binding = assert_predictor_binding(
            experiment_id, analysis_id, shift_values, canonical, variants,
            censor, inventory, planned_paths, output_root,
        )
        predictor_outcome = run_predictor_export(
            experiment_id, analysis_id, base_context, canonical, variants,
            inventory, binding, output_root=output_root, force=force,
            resume=resume, engine=predictor_engine,
        )
        files_written += predictor_outcome["files_written"]
        files_rewritten += predictor_outcome["files_rewritten"]
        reused_flags.append(predictor_outcome["reused"])

    local_downstream_outcome: Optional[dict] = None
    if LOCAL_DOWNSTREAM_STAGE in stages:
        # Binds to the completed plan, prelabel AND predictor-export stages,
        # and verifies every variant/namespace contract, before any downstream
        # directory is created or any production helper is imported.
        binding = assert_local_downstream_binding(
            experiment_id, analysis_id, shift_values, canonical, variants,
            censor, inventory, planned_paths, output_root,
        )
        local_downstream_outcome = run_local_downstream(
            experiment_id, analysis_id, base_context, canonical, variants,
            inventory, binding, output_root=output_root,
            experiments_root=experiments_root, force=force, resume=resume,
            recover_partial=recover_partial_local_downstream,
            engine=local_downstream_engine,
        )
        files_written += local_downstream_outcome["files_written"]
        files_rewritten += local_downstream_outcome["files_rewritten"]
        reused_flags.append(local_downstream_outcome["reused"])

    model_outcome: Optional[dict] = None
    if MODEL_STAGE in stages:
        # Binds to the completed plan, prelabel, predictor-export AND
        # local-downstream stages, and resolves the three Step8A datasets by
        # their recorded hashes, before any directory is created.
        binding = assert_model_binding(
            experiment_id, analysis_id, shift_values, canonical, variants,
            censor, inventory, planned_paths, output_root, experiments_root,
        )
        model_outcome = run_model_stage(
            experiment_id, analysis_id, canonical, variants, binding,
            output_root=output_root, experiments_root=experiments_root,
            force=force, resume=resume,
            configuration_overrides=model_configuration_overrides,
        )
        files_written += model_outcome["files_written"]
        files_rewritten += model_outcome["files_written"]
        reused_flags.append(model_outcome["reused"])

    compare_outcome: Optional[dict] = None
    if COMPARE_STAGE in stages:
        # Binds to every completed upstream stage AND to the verified model
        # stage, hashing each recorded model artefact, before any write.
        binding = assert_compare_binding(
            experiment_id, analysis_id, shift_values, canonical, variants,
            censor, inventory, planned_paths, output_root, experiments_root,
        )
        compare_outcome = run_compare_stage(
            experiment_id, analysis_id, variants, binding,
            output_root=output_root, experiments_root=experiments_root,
            force=force, resume=resume,
        )
        files_written += compare_outcome["files_written"]
        files_rewritten += compare_outcome["files_written"]
        reused_flags.append(compare_outcome["reused"])

    # ...and again afterwards, so nothing can be published against inputs that
    # moved while the stage was running.
    hashes_after = frozen_hash_map(
        frozen_input_inventory(experiment_id, experiments_root, base_context)
    )
    assert_frozen_hashes_unchanged(hashes_before, hashes_after, "while writing the plan")

    result = {
        "ran": True,
        "dry_run": False,
        "experiment_id": experiment_id,
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "stages_run": list(stages),
        "shift_days": list(shift_values),
        "variants": variants,
        "output_root": str(experiment_root(experiment_id, output_root)),
        "files_written": sorted(files_written),
        "files_written_count": len(files_written),
        "files_rewritten": sorted(files_rewritten),
        "reused": bool(reused_flags) and all(reused_flags),
        "forced": bool(force),
        "resume_requested": bool(resume),
        "prerequisites_ready": prerequisite_status["prerequisites_ready"],
        "missing_required_inputs": prerequisite_status["missing_required_inputs"],
        "prelabel_exclusion_binding": censor_binding,
        "label_inputs": {
            role: {
                "path": str(entry["path"]),
                "resolved_from": entry["resolved_from"],
                "exists": inventory[role]["exists"],
                "sha256": inventory[role]["sha256"],
            }
            for role, entry in sorted(labels.items())
        },
        "frozen_input_sha256": hashes_after,
        "frozen_hashes_unchanged": True,
        "gee_queries_run": bool(
            (prelabel_outcome and prelabel_outcome["gee_query_run"])
            or (predictor_outcome and predictor_outcome["gee_query_run"])
        ),
        "gee_exports_run": bool(
            (prelabel_outcome and prelabel_outcome["gee_export_run"])
            or (predictor_outcome and predictor_outcome["gee_export_run"])
        ),
        "model_fit": False,
        "bootstrap_run": False,
        "canonical_export_attempted": False,
        "status": STATUS_PASS,
        "limitations": list(LIMITATIONS),
    }
    if plan_outcome is not None:
        result["plan"] = {
            "files_written": plan_outcome["files_written"],
            "files_written_count": plan_outcome["files_written_count"],
            "files_rewritten": plan_outcome["files_rewritten"],
            "reused": plan_outcome["reused"],
            "previous_analysis_id": plan_outcome["previous_analysis_id"],
            "unmanaged_plan_documents": plan_outcome["unmanaged_plan_documents"],
        }
        # Backwards-compatible top-level plan fields for a plan-only run.
        result["previous_analysis_id"] = plan_outcome["previous_analysis_id"]
        result["unmanaged_plan_documents"] = plan_outcome["unmanaged_plan_documents"]
    if prelabel_outcome is not None:
        summary = prelabel_outcome["summary"]
        result["prelabel_censor"] = summary
        result["plan_binding"] = binding
        result["raster_sha256"] = summary["raster_sha256"]
        result["prelabel_burn_cell_count"] = summary["prelabel_burn_cell_count"]
        result["date_semantics"] = summary["date_semantics"]
        result["canonical_outputs_modified"] = False
    if predictor_outcome is not None:
        result["predictor_export"] = predictor_outcome
        result["plan_binding"] = binding
        result["processed_variants"] = predictor_outcome["processed_variants"]
        result["reused_variants"] = predictor_outcome["reused_variants"]
        result["exported_variants"] = predictor_outcome["exported_variants"]
        result["logical_roles_produced"] = predictor_outcome["logical_roles_produced"]
        result["predictor_rasters_produced"] = predictor_outcome["predictor_rasters_produced"]
        result["quarantined_artifacts"] = predictor_outcome["quarantined_artifacts"]
        result["canonical_outputs_modified"] = False
    if local_downstream_outcome is not None:
        result["local_downstream"] = local_downstream_outcome
        result["plan_binding"] = binding
        result["processed_variants"] = local_downstream_outcome["processed_variants"]
        result["reused_variants"] = local_downstream_outcome["reused_variants"]
        result["completed_variants"] = local_downstream_outcome["completed_variants"]
        result["downstream_artifacts_produced"] = (
            local_downstream_outcome["downstream_artifacts_produced"]
        )
        result["step8a_datasets_produced"] = (
            local_downstream_outcome["step8a_datasets_produced"]
        )
        result["quarantined_artifacts"] = local_downstream_outcome["quarantined_artifacts"]
        result["canonical_downstream_attempted"] = False
        result["common_cohort_created"] = False
        result["canonical_outputs_modified"] = False
        # The production chain really did train the Step7C downscaling model,
        # so the result says so; the fire-risk model stage remains locked.
        result.update(LOCAL_DOWNSTREAM_MODEL_SEMANTICS)
    if model_outcome is not None:
        result["model_stage"] = {
            key: value for key, value in model_outcome.items() if key != "metadata"
        }
        result["plan_binding"] = binding
        result["model_stage_metadata"] = model_outcome["metadata"]
        result["model_reused"] = model_outcome["reused"]
        result["quarantine_manifest"] = model_outcome["quarantine"]
        result["canonical_outputs_modified"] = False
        result["upstream_outputs_modified"] = False
        # This stage DOES fit the fire-risk models and DOES bootstrap; the
        # Step7C downscaling model is referenced, never refit.
        result.update(MODEL_STAGE_SEMANTICS)
    if compare_outcome is not None:
        result["compare_stage"] = {
            key: value for key, value in compare_outcome.items() if key != "metadata"
        }
        result["plan_binding"] = binding
        result["compare_stage_metadata"] = compare_outcome["metadata"]
        result["compare_reused"] = compare_outcome["reused"]
        result["quarantine_manifest"] = compare_outcome["quarantine"]
        # Compare fits nothing and bootstraps nothing; these flags describe the
        # COMPARE stage only, whatever an earlier stage recorded about itself.
        result.update(COMPARE_STAGE_SEMANTICS)
    return result
