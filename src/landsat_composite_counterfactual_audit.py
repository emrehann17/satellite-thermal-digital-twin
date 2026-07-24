"""landsat_composite_counterfactual_audit.py

Strictly diagnostic, NON-DESTRUCTIVE Landsat composite counterfactual audit.

SCIENTIFIC QUESTION
-------------------
The canonical current-period and yearly-baseline Landsat functions reduce all
QA-clean *source scenes* directly with ``ImageCollection.median()`` and
``ImageCollection.count()`` (see :mod:`src.step3_landsat_lst`). When several
overlapping scenes are acquired on the SAME date, each of those same-date
scenes receives its own temporal weight in that reducer. This audit tests
whether first collapsing same-date overlapping scenes into ONE daily composite
(``date_balanced``) reduces observed Landsat scene/support discontinuities,
relative to the existing per-scene reducer (``scene_weighted``).

CLAIM BOUNDARY
--------------
This module does NOT assert that same-date duplication is the sole root cause
of any seam. Daily compositing removes duplicate same-date weighting, but
different WRS path/row regions can still carry different acquisition-date
support. The audit only measures whether the change reduces jumps, with a
paired bootstrap interval, and reports an honest status.

NON-DESTRUCTIVE GUARANTEES
--------------------------
* No canonical predictor / Step5 / Step5C / Step7 / Step8 / Step9 / Step10
  behaviour is modified. This module imports canonical QA/NDVI helpers from
  :mod:`src.step3_landsat_lst` and Step5 raster math from
  :mod:`src.step5_preprocess_timeseries` READ-ONLY, and reproduces the
  ``scene_weighted`` chain from them verbatim.
* Every writable path is asserted to live under
  ``outputs/diagnostics/landsat_composite_counterfactual/<experiment_id>/``
  (see :func:`assert_diagnostic_namespace_safe`). Nothing is ever written under
  ``outputs/experiments/<id>/{data,step5*,step7*,step8*}`` or the legacy shared
  ``data/`` / ``outputs/`` trees.

QA MASK HONESTY
---------------
The canonical ``apply_qa_mask`` uses QA_PIXEL ONLY. The existing source-scene
provenance config text nevertheless claims ``QA_PIXEL ... + QA_RADSAT``. This
audit PRESERVES QA_PIXEL-only behaviour, and only corrects the *diagnostic*
description of what was actually applied (see :func:`qa_mask_provenance`). It
does NOT add QA_RADSAT masking to any canonical or diagnostic product, because
that would change two methodological factors at once.

DATE SEMANTICS
--------------
Earth Engine ``filterDate`` uses an EXCLUSIVE end date. The counterfactual
preserves the currently frozen date-window behaviour so that the compositing
method is the only changed factor; the effective end-exclusive semantics are
recorded in metadata (:func:`date_window_semantics`) rather than silently
"fixed" here.

The GEE-touching functions import ``ee`` lazily so this module (and its pure
helpers) import cleanly without Earth Engine and without live credentials.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.paths import PROJECT_ROOT

# Canonical read-only imports. These are NEVER mutated here.
from core.config import (  # noqa: E402
    LANDSAT_COLLECTION,
    LANDSAT_OFFSET,
    LANDSAT_SCALE,
    STEP5_MIN_BASELINE_STD_CELSIUS,
    STEP5_MIN_BASELINE_VALID_COUNT,
    STEP5_MIN_CURRENT_VALID_COUNT,
)

DIAGNOSTIC_NAMESPACE = "landsat_composite_counterfactual"

# Canonical QA_PIXEL bits masked by src.step3_landsat_lst.apply_qa_mask. Kept
# here only for honest *description*; the actual masking is performed by the
# imported canonical helper, never re-implemented.
QA_PIXEL_MASKED_BITS = [
    "fill",
    "dilated_cloud",
    "cirrus",
    "cloud",
    "cloud_shadow",
    "snow",
    "medium_high_cloud_confidence",
    "medium_high_cloud_shadow_confidence",
    "medium_high_snow_ice_confidence",
    "medium_high_cirrus_confidence",
]

# Counterfactual chains. "A" reproduces the canonical per-scene reducer; "B"
# collapses same-date overlapping scenes to one daily composite first.
CHAIN_SCENE_WEIGHTED = "scene_weighted"
CHAIN_DATE_BALANCED = "date_balanced"
CHAINS = (CHAIN_SCENE_WEIGHTED, CHAIN_DATE_BALANCED)

# Explicit nodata sentinel stamped on every GEE-exported diagnostic raster so
# masked / AOI-exterior pixels carry a verifiable nodata tag and are NEVER read
# as physical numeric zero. Value/count/difference/multiplicity products are all
# >= this sentinel in practice, so it can be distinguished unambiguously.
NODATA_SENTINEL = -9999.0

# Predeclared canonical-reproduction tolerances (fixed BEFORE inspecting any
# result). DN/count products must match near-exactly; physical float32 products
# (Celsius / NDVI / z-score) allow a small fixed float32 export tolerance.
REPRODUCTION_TOLERANCES = {
    "dn_count": 1e-6,
    "physical_float32": 1e-5,
}

# Boundary types the live audit reports separately.
BOUNDARY_TYPES = (
    "source_scene_path_row",      # verified provenance boundaries (when available)
    "scene_count_edge",           # scene-observation-count edges
    "unique_date_count_edge",     # unique acquisition-date support edges
    "same_day_multiplicity_edge", # same-day multiplicity edges
    "export_tile_boundary",       # negative control (from recorded export grid)
)


# =============================================================================
# NAMESPACE SAFETY (pure)
# =============================================================================
def diagnostic_output_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Return the ONLY directory tree this audit is permitted to write under."""
    return (
        Path(base_dir)
        / "outputs"
        / "diagnostics"
        / DIAGNOSTIC_NAMESPACE
        / experiment_id
    )


def canonical_forbidden_roots(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> list[Path]:
    """Directories that this diagnostic must NEVER write to (defense in depth).

    Covers the experiment's canonical predictor/Step5/Step7/Step8 outputs and
    the legacy shared ``data/`` and ``outputs/`` trees.
    """
    base = Path(base_dir)
    exp_root = base / "outputs" / "experiments" / experiment_id
    forbidden = [
        exp_root / "data",
        exp_root / "step5",
        exp_root / "step5b",
        exp_root / "step5c",
        exp_root / "step7a",
        exp_root / "step7b",
        exp_root / "step7c",
        exp_root / "step7d",
        exp_root / "step7e",
        exp_root / "step8a",
        exp_root / "step8b",
        exp_root / "step8c",
        exp_root / "step8d",
        exp_root / "step8e",
        # legacy shared predictor inputs / step outputs
        base / "data" / "landsat_timeseries",
        base / "data" / "landsat_qa",
        base / "data" / "current_period",
        base / "data" / "ndvi_timeseries",
        base / "data" / "ndvi_current_period",
        base / "data" / "modis",
        base / "outputs" / "step3",
        base / "outputs" / "step4",
        base / "outputs" / "step5",
        base / "outputs" / "step5b",
        base / "outputs" / "step5c",
    ]
    return [p.resolve() for p in forbidden]


class NamespaceSafetyError(RuntimeError):
    """Raised when a planned diagnostic output escapes the safe namespace."""


def assert_diagnostic_namespace_safe(
    paths,
    experiment_id: str,
    base_dir: Path = PROJECT_ROOT,
) -> None:
    """Fail-fast if ANY path is outside the diagnostic root or inside a
    canonical/legacy output directory.

    A path is safe iff it is equal to, or a descendant of,
    :func:`diagnostic_output_root`, AND is not equal to / inside any
    :func:`canonical_forbidden_roots` entry.
    """
    root = diagnostic_output_root(experiment_id, base_dir).resolve()
    forbidden = canonical_forbidden_roots(experiment_id, base_dir)
    for raw in paths:
        resolved = Path(raw).resolve()
        inside_root = resolved == root or root in resolved.parents
        if not inside_root:
            raise NamespaceSafetyError(
                f"Diagnostic output path escapes the safe namespace: {resolved} "
                f"is not under {root}"
            )
        for bad in forbidden:
            if resolved == bad or bad in resolved.parents:
                raise NamespaceSafetyError(
                    f"Diagnostic output path lands in a canonical/legacy directory: "
                    f"{resolved} is under forbidden {bad}"
                )


# =============================================================================
# ACQUISITION-DATE GROUPING (pure)
# =============================================================================
def acquisition_date_of(record: dict) -> str | None:
    """Return the ``YYYY-MM-DD`` acquisition date for a scene record.

    Accepts either an explicit ``acquisition_date`` or a full
    ``acquisition_datetime`` and truncates to the calendar date. Returns None
    when neither is present (the record cannot be grouped by date).
    """
    value = record.get("acquisition_date")
    if not value:
        value = record.get("acquisition_datetime")
    if not value:
        return None
    return str(value)[:10]


def group_records_by_date(records) -> "OrderedDict[str, list]":
    """Group scene records by acquisition date, deterministically.

    Returns an ``OrderedDict`` keyed by ascending date; within each date the
    records preserve stable ``scene_id`` ordering. Records without a resolvable
    date are dropped (they cannot participate in a daily composite).
    """
    buckets: dict[str, list] = {}
    for record in records:
        date = acquisition_date_of(record)
        if date is None:
            continue
        buckets.setdefault(date, []).append(record)
    ordered = OrderedDict()
    for date in sorted(buckets):
        ordered[date] = sorted(
            buckets[date], key=lambda r: str(r.get("scene_id"))
        )
    return ordered


def unique_acquisition_dates(records) -> list[str]:
    """Sorted list of distinct acquisition dates present in ``records``."""
    return list(group_records_by_date(records).keys())


def plan_daily_composites(records) -> list[dict]:
    """Plan EXACTLY ONE daily composite per unique acquisition date.

    This is the pure contract behind the ``date_balanced`` reducer: the number
    of planned daily images equals the number of unique acquisition dates, and
    every source scene for a date is listed under that single composite.
    """
    grouped = group_records_by_date(records)
    plan = []
    for date, day_records in grouped.items():
        scene_ids = [str(r.get("scene_id")) for r in day_records]
        plan.append(
            {
                "acquisition_date": date,
                "scene_ids": scene_ids,
                "source_scene_count": len(day_records),
            }
        )
    return plan


def count_semantics(records) -> dict:
    """Scene-count vs unique-date-count semantics for a scene set.

    ``same_day_multiplicity`` is ``scene_count - unique_date_count`` -- the
    number of "extra" same-date scenes that receive a duplicate temporal weight
    under the canonical per-scene reducer but a single weight under daily
    compositing.
    """
    datable = [r for r in records if acquisition_date_of(r) is not None]
    scene_count = len(datable)
    unique_dates = len(unique_acquisition_dates(datable))
    return {
        "scene_count": scene_count,
        "unique_date_count": unique_dates,
        "same_day_multiplicity": scene_count - unique_dates,
        "undatable_scene_count": len(records) - scene_count,
    }


# =============================================================================
# QA MASK PROVENANCE HONESTY (pure)
# =============================================================================
def qa_mask_provenance() -> dict:
    """Honest description of the QA masking actually applied by the canonical
    ``apply_qa_mask`` (QA_PIXEL ONLY), plus the recorded mismatch with the
    existing provenance config text that falsely mentions QA_RADSAT.
    """
    return {
        "qa_source": "QA_PIXEL",
        "qa_pixel_only": True,
        "qa_radsat_applied": False,
        "masked_bits": list(QA_PIXEL_MASKED_BITS),
        "qa_water_bit_preserved": True,
        "canonical_helper": "src.step3_landsat_lst.apply_qa_mask",
        "reused_verbatim": True,
        "metadata_mismatch": {
            "source": "core.source_scene_provenance_config composites text",
            "claimed": "QA_PIXEL cloud/shadow/snow/fill + QA_RADSAT",
            "actual": "QA_PIXEL cloud/shadow/snow/fill ONLY (no QA_RADSAT)",
            "diagnostic_action": (
                "This audit reports the mismatch and corrects only the "
                "diagnostic/provenance DESCRIPTION. It does NOT add QA_RADSAT "
                "to any canonical or diagnostic product; doing so would change "
                "two factors at once."
            ),
        },
    }


# =============================================================================
# DATE-WINDOW SEMANTICS (pure)
# =============================================================================
def date_window_semantics(start_date: str, end_date: str) -> dict:
    """Record the frozen, end-EXCLUSIVE ``filterDate`` window semantics.

    The counterfactual preserves this behaviour exactly so the compositing
    method is the only changed factor; the off-by-one is documented, not fixed.
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    last_inclusive = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "filter_date_start": start_date,
        "filter_date_end": end_date,
        "end_semantics": "exclusive",
        "effective_last_included_date": last_inclusive,
        "note": (
            "Earth Engine filterDate end is exclusive. This frozen behaviour is "
            "preserved unchanged in the counterfactual; the compositing method "
            "is the only intentionally changed factor."
        ),
    }


# =============================================================================
# PAIRED BOOTSTRAP STATISTICS (pure)
# =============================================================================
def classify_paired_interval(low: float | None, high: float | None) -> str:
    """Classify a reduction percentile interval.

    ``reduction = absolute_jump_scene_weighted - absolute_jump_date_balanced``,
    so a positive reduction means date-balanced compositing lowered the jump.

    * ``supported_reduction``  -- interval entirely above zero
    * ``supported_increase``   -- interval entirely below zero
    * ``uncertain``            -- interval includes zero
    * ``insufficient_evidence``-- interval undefined (None)
    """
    if low is None or high is None:
        return "insufficient_evidence"
    if low > 0.0:
        return "supported_reduction"
    if high < 0.0:
        return "supported_increase"
    return "uncertain"


def bootstrap_reduction_interval(
    reductions,
    *,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 20240717,
) -> dict:
    """Paired bootstrap percentile interval of the mean per-segment reduction.

    ``reductions`` is one reduction value per independent boundary segment. The
    interval is a two-sided ``ci`` percentile interval of the bootstrap
    distribution of the mean. Deterministic given ``seed``.
    """
    import numpy as np

    values = np.asarray(
        [v for v in reductions if v is not None and np.isfinite(v)],
        dtype="float64",
    )
    n = int(values.size)
    result = {
        "n_segments": n,
        "n_boot": int(n_boot),
        "ci": float(ci),
        "seed": int(seed),
        "point_estimate": float(np.mean(values)) if n else None,
        "interval_low": None,
        "interval_high": None,
    }
    if n < 2:
        return result
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = values[idx].mean(axis=1)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    result["interval_low"] = float(np.quantile(boot_means, lo_q))
    result["interval_high"] = float(np.quantile(boot_means, hi_q))
    return result


def classify_reduction(
    reductions,
    *,
    min_segments: int = 8,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 20240717,
) -> dict:
    """Full paired-comparison verdict.

    When fewer than ``min_segments`` independent segments exist the status is
    ``insufficient_evidence`` regardless of the point estimate. Otherwise the
    bootstrap interval decides. Thresholds here are fixed a-priori (segment
    count and CI); no favourable jump-ratio threshold is introduced or tuned
    after examining results.
    """
    interval = bootstrap_reduction_interval(
        reductions, n_boot=n_boot, ci=ci, seed=seed
    )
    if interval["n_segments"] < int(min_segments):
        status = "insufficient_evidence"
    else:
        status = classify_paired_interval(
            interval["interval_low"], interval["interval_high"]
        )
    return {
        "status": status,
        "min_segments_required": int(min_segments),
        **interval,
        "reduction_definition": (
            "absolute_jump_scene_weighted - absolute_jump_date_balanced "
            "(positive => date_balanced reduces the jump)"
        ),
    }


# =============================================================================
# STEP5 POLICY SNAPSHOT (pure, read-only mirror of canonical constants)
# =============================================================================
def step5_policy_snapshot() -> dict:
    """The Step5 guard policies preserved by the diagnostic derived comparison.

    These mirror the canonical ``core.config`` constants read-only; the audit
    applies the SAME thresholds to both chains so the compositing method is the
    only changed factor.
    """
    return {
        "baseline_min_valid_year_count": int(STEP5_MIN_BASELINE_VALID_COUNT),
        "minimum_baseline_std_celsius": float(STEP5_MIN_BASELINE_STD_CELSIUS),
        "minimum_current_valid_support": int(STEP5_MIN_CURRENT_VALID_COUNT),
        "current_support_semantics": {
            CHAIN_SCENE_WEIGHTED: "valid QA-clean source-scene count",
            CHAIN_DATE_BALANCED: "valid UNIQUE acquisition-date count",
        },
        "baseline_valid_count_semantics": (
            "count of valid YEARLY composites (matches Step5 baseline_valid_count "
            "for both chains); NOT the same as the date-balanced current "
            "unique-acquisition-date support count"
        ),
        "landsat_scale": float(LANDSAT_SCALE),
        "landsat_offset": float(LANDSAT_OFFSET),
    }


# =============================================================================
# AUDIT CONFIG + FILE MANIFEST (pure, deterministic)
# =============================================================================
def build_audit_config(
    ctx: dict,
    *,
    seed: int = 20240717,
    n_boot: int = 10000,
    ci: float = 0.95,
    min_segments: int = 8,
    base_dir: Path = PROJECT_ROOT,
) -> "OrderedDict":
    """Deterministic audit configuration document (stable key ordering)."""
    experiment_id = ctx["experiment_id"]
    baseline_years = sorted(int(y) for y in ctx["baseline_years"])
    current_year = int(str(ctx["current_period_end_date"])[:4])
    baseline_years = [y for y in baseline_years if y < current_year]
    current = _current_window(
        ctx["current_period_end_date"], ctx["current_period_days"]
    )
    config = OrderedDict()
    config["audit"] = DIAGNOSTIC_NAMESPACE
    config["audit_kind"] = "diagnostic_non_destructive_counterfactual"
    config["experiment_id"] = experiment_id
    config["region_key"] = ctx.get("region_key")
    config["source_collection"] = LANDSAT_COLLECTION
    config["chains"] = list(CHAINS)
    config["current_period"] = OrderedDict(
        [
            ("end_date", ctx["current_period_end_date"]),
            ("window_days", int(ctx["current_period_days"])),
            ("start_date", current["start_date"]),
            ("months_filter", current["months_filter"]),
            ("date_window_semantics", date_window_semantics(
                current["start_date"], ctx["current_period_end_date"]
            )),
        ]
    )
    config["baseline_years"] = baseline_years
    config["step5_policy"] = step5_policy_snapshot()
    config["qa_mask"] = qa_mask_provenance()
    config["paired_comparison"] = OrderedDict(
        [
            ("seed", int(seed)),
            ("n_boot", int(n_boot)),
            ("ci", float(ci)),
            ("min_segments", int(min_segments)),
            ("reduction", (
                "absolute_jump_scene_weighted - absolute_jump_date_balanced"
            )),
            ("statuses", [
                "supported_reduction", "uncertain",
                "supported_increase", "insufficient_evidence",
            ]),
        ]
    )
    config["output_root"] = str(diagnostic_output_root(experiment_id, base_dir))
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    config["claim_boundary"] = (
        "Daily compositing removes duplicate same-date weighting but does NOT "
        "by itself prove same-date duplication is the sole root cause; "
        "different path/row regions can still carry different date support."
    )
    return config


def sha256_and_size(path: Path) -> dict:
    """SHA-256 hex digest and byte size for a single file."""
    data = Path(path).read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def build_file_manifest(paths, *, output_dir: Path) -> "OrderedDict":
    """Deterministic manifest of produced files, sorted by relative name.

    Each entry carries the relative path, SHA-256, and byte size. Ordering is
    stable so the manifest (and its own hash) are reproducible.
    """
    output_dir = Path(output_dir)
    entries = []
    for path in paths:
        path = Path(path)
        try:
            rel = str(path.resolve().relative_to(output_dir.resolve()))
        except ValueError:
            rel = path.name
        entries.append((rel, path))
    entries.sort(key=lambda item: item[0])
    files = []
    for rel, path in entries:
        info = sha256_and_size(path)
        files.append(
            OrderedDict(
                [("path", rel), ("sha256", info["sha256"]), ("bytes", info["bytes"])]
            )
        )
    manifest = OrderedDict()
    manifest["audit"] = DIAGNOSTIC_NAMESPACE
    manifest["output_dir"] = str(output_dir)
    manifest["file_count"] = len(files)
    manifest["files"] = files
    return manifest


# =============================================================================
# EE WINDOW HELPERS (pure date arithmetic, mirrors canonical step3 exactly)
# =============================================================================
def _current_window(end_date: str, window_days: int) -> dict:
    """Reproduce the canonical current-period window + month filter EXACTLY.

    Mirrors ``step3_landsat_lst.get_current_period_median`` so ``scene_weighted``
    and ``date_balanced`` share an identical source-scene population and the
    only changed factor is the reducer.
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=window_days)
    start_date = start_dt.strftime("%Y-%m-%d")
    window_months = sorted(
        {(start_dt.month + i) % 12 or 12 for i in range((end_dt - start_dt).days + 1)}
    )
    return {
        "start_date": start_date,
        "end_date": end_date,
        "month_start": min(window_months),
        "month_end": max(window_months),
        "months_filter": f"{min(window_months)}-{max(window_months)}",
    }


def _baseline_year_window(end_date: str, window_days: int, year: int) -> dict:
    """Reproduce the canonical symmetric baseline-year window EXACTLY.

    Mirrors ``step3_landsat_lst.get_landsat_baseline_window_median_collection``:
    no calendar-month filter, only the shifted symmetric date window.
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=window_days)
    window_start = start_dt.replace(year=year)
    window_end = end_dt.replace(year=year)
    return {
        "start_date": window_start.strftime("%Y-%m-%d"),
        "end_date": window_end.strftime("%Y-%m-%d"),
        "baseline_year": year,
    }


# =============================================================================
# RASTER OUTPUT PLAN (pure)
# =============================================================================
def _lst_pair_names(prefix: str) -> "OrderedDict[str, str]":
    """Diagnostic raster names for one LST role (current or one baseline year).

    Produces the six required diagnostic counterparts per role.
    """
    return OrderedDict(
        [
            (f"{prefix}_scene_weighted_median", f"rasters/{prefix}_scene_weighted_median.tif"),
            (f"{prefix}_date_balanced_median", f"rasters/{prefix}_date_balanced_median.tif"),
            (f"{prefix}_scene_valid_count", f"rasters/{prefix}_scene_valid_count.tif"),
            (f"{prefix}_unique_date_valid_count", f"rasters/{prefix}_unique_date_valid_count.tif"),
            (f"{prefix}_same_day_multiplicity", f"rasters/{prefix}_same_day_multiplicity.tif"),
            (f"{prefix}_date_balanced_minus_scene_weighted", f"rasters/{prefix}_date_balanced_minus_scene_weighted.tif"),
        ]
    )


def plan_raster_outputs(ctx: dict) -> "OrderedDict[str, str]":
    """Ordered {name: relative_path} of every diagnostic raster this audit
    would export. Used identically by ``--dry-run`` (list only) and a live run.
    """
    baseline_years = sorted(int(y) for y in ctx["baseline_years"])
    current_year = int(str(ctx["current_period_end_date"])[:4])
    baseline_years = [y for y in baseline_years if y < current_year]

    outputs: "OrderedDict[str, str]" = OrderedDict()
    # --- CURRENT LST ---
    outputs.update(_lst_pair_names("current_lst"))
    # --- BASELINE LST per year ---
    for year in baseline_years:
        outputs.update(_lst_pair_names(f"baseline_lst_{year}"))
    # --- CURRENT NDVI ---
    outputs.update(_lst_pair_names("current_ndvi"))
    # --- BASELINE NDVI per year ---
    for year in baseline_years:
        outputs.update(_lst_pair_names(f"baseline_ndvi_{year}"))
    # --- DERIVED comparison per chain (Step5-policy diagnostic-only) ---
    for chain in CHAINS:
        outputs[f"{chain}_baseline_mean"] = f"derived/{chain}/baseline_lst_mean_celsius.tif"
        outputs[f"{chain}_baseline_std"] = f"derived/{chain}/baseline_lst_std_celsius.tif"
        outputs[f"{chain}_baseline_valid_year_count"] = f"derived/{chain}/baseline_valid_year_count.tif"
        outputs[f"{chain}_current_minus_baseline"] = f"derived/{chain}/current_minus_baseline_celsius.tif"
        outputs[f"{chain}_anomaly_zscore"] = f"derived/{chain}/anomaly_zscore.tif"
    return outputs


def plan_document_outputs() -> "OrderedDict[str, str]":
    """Ordered {name: relative_path} of the non-raster deliverables."""
    return OrderedDict(
        [
            ("audit_config", "audit_config.json"),
            ("source_scene_metadata", "source_scene_metadata.json"),
            ("scene_manifest_csv", "scene_manifest.csv"),
            ("scene_footprints", "scene_footprints.geojson"),
            ("scene_boundaries", "scene_boundaries.geojson"),
            ("raster_inventory", "raster_inventory.csv"),
            ("boundary_metrics", "boundary_metrics.csv"),
            ("paired_boundary_comparison", "paired_boundary_comparison.csv"),
            ("counterfactual_summary_json", "counterfactual_summary.json"),
            ("counterfactual_summary_md", "counterfactual_summary.md"),
            ("manifest", "manifest.json"),
            ("run_log", "run_log.txt"),
        ]
    )


def plan_all_output_paths(ctx: dict, base_dir: Path = PROJECT_ROOT) -> list[Path]:
    """Absolute paths of every file the audit could write (rasters + docs +
    PNG maps). Used for the up-front namespace-safety assertion.
    """
    root = diagnostic_output_root(ctx["experiment_id"], base_dir)
    paths = [root / rel for rel in plan_raster_outputs(ctx).values()]
    paths += [root / rel for rel in plan_document_outputs().values()]
    # one PNG per compared pair (shared robust stretch per pair)
    for name, rel in plan_raster_outputs(ctx).items():
        if name.endswith("_scene_weighted_median") or name.endswith("_date_balanced_median"):
            stem = Path(rel).stem
            paths.append(root / "maps" / f"{stem}.png")
    return paths


# =============================================================================
# EE IMAGE BUILDERS (lazy ee import; reducer is the ONLY changed factor)
# =============================================================================
def _set_export_date(image):
    """Attach a ``YYYY-MM-dd`` ``export_date`` property (canonical pattern)."""
    import ee

    export_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
    return image.set("export_date", export_date)


def _base_filtered_collection(region, start_date, end_date, month_range, bands):
    """Canonical-identical filtered collection for a role BEFORE any reducer.

    ``month_range`` is ``(month_start, month_end)`` for the current period, or
    ``None`` for baseline-year windows (which use no month filter, matching
    canonical behaviour). Both chains consume this exact collection.
    """
    import ee

    collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )
    if month_range is not None:
        collection = collection.filter(
            ee.Filter.calendarRange(month_range[0], month_range[1], "month")
        )
    collection = collection.select(bands).map(lambda img: img.clip(region))
    return collection


def _daily_composite_collection(base_collection, value_band, region):
    """Generic date-grouping reducer honouring the collection's EXACT window.

    For each distinct acquisition date: (1) select that date's scenes, (2) apply
    the canonical QA_PIXEL mask, (3) take the same-date SPATIAL median of the
    value band, (4) emit exactly ONE daily image. No hard-coded month filter --
    the window is whatever ``base_collection`` was already filtered to.
    """
    import ee

    from src.step3_landsat_lst import apply_qa_mask

    dated = base_collection.map(_set_export_date)
    unique_dates = dated.aggregate_array("export_date").distinct().sort()

    def build_daily(date_value):
        date_value = ee.String(date_value)
        day = dated.filter(ee.Filter.eq("export_date", date_value))
        masked = day.map(apply_qa_mask)
        daily_median = masked.select(value_band).median().rename(value_band)
        return daily_median.clip(region).set("export_date", date_value)

    return ee.ImageCollection(unique_dates.map(build_daily))


def build_lst_role_images(
    region,
    start_date,
    end_date,
    month_range,
    *,
    to_celsius: bool,
) -> dict:
    """Build the six diagnostic LST images for one role (current or one year).

    ``scene_weighted`` reproduces the canonical per-scene reducer verbatim;
    ``date_balanced`` collapses same-date scenes to one daily composite first.
    Both share the identical filtered source-scene population.
    """
    from src.step3_landsat_lst import apply_qa_mask

    base = _base_filtered_collection(
        region, start_date, end_date, month_range, ["ST_B10", "QA_PIXEL"]
    )

    # --- A. scene_weighted: canonical per-scene reducer ---
    masked = base.map(apply_qa_mask).select("ST_B10")
    sw_median = masked.median().rename("ST_B10")
    sw_count = masked.count().rename("scene_valid_count").toFloat()

    # --- B. date_balanced: one daily composite per date, then temporal median ---
    daily = _daily_composite_collection(base, "ST_B10", region)
    db_median = daily.median().rename("ST_B10")
    db_count = daily.count().rename("unique_date_valid_count").toFloat()

    if to_celsius:
        def _celsius(img):
            return (
                img.multiply(LANDSAT_SCALE).add(LANDSAT_OFFSET).subtract(273.15)
                .rename("LST_Celsius").toFloat()
            )
        sw_value = _celsius(sw_median)
        db_value = _celsius(db_median)
    else:
        sw_value = sw_median.toFloat()
        db_value = db_median.toFloat()

    multiplicity = sw_count.subtract(db_count).rename("same_day_multiplicity")
    difference = db_value.subtract(sw_value).rename(
        "date_balanced_minus_scene_weighted"
    )
    return {
        "scene_weighted_median": sw_value.clip(region),
        "date_balanced_median": db_value.clip(region),
        "scene_valid_count": sw_count.clip(region),
        "unique_date_valid_count": db_count.clip(region),
        "same_day_multiplicity": multiplicity.clip(region),
        "date_balanced_minus_scene_weighted": difference.clip(region),
    }


def build_ndvi_role_images(region, start_date, end_date, month_range) -> dict:
    """Build the six diagnostic NDVI images for one role.

    NDVI is derived per scene with the canonical ``add_ndvi_band`` (which itself
    applies the physical-range mask), then reduced by each chain.
    """
    import ee

    from src.step3_landsat_lst import (
        add_ndvi_band,
        apply_qa_mask,
        mask_ndvi_physical_range,
    )

    base = _base_filtered_collection(
        region, start_date, end_date, month_range,
        ["SR_B4", "SR_B5", "QA_PIXEL"],
    )
    # scene_weighted: canonical per-scene NDVI reducer
    masked = base.map(apply_qa_mask).map(add_ndvi_band).select("NDVI")
    sw_median = mask_ndvi_physical_range(masked.median()).rename("NDVI").toFloat()
    sw_count = masked.count().rename("scene_valid_count").toFloat()

    # date_balanced: per-date NDVI daily composite, then temporal median
    dated = base.map(_set_export_date)
    unique_dates = dated.aggregate_array("export_date").distinct().sort()

    def build_daily(date_value):
        date_value = ee.String(date_value)
        day = dated.filter(ee.Filter.eq("export_date", date_value))
        ndvi_day = day.map(apply_qa_mask).map(add_ndvi_band).select("NDVI")
        daily = mask_ndvi_physical_range(ndvi_day.median()).rename("NDVI")
        return daily.clip(region).set("export_date", date_value)

    daily_collection = ee.ImageCollection(unique_dates.map(build_daily))
    db_median = mask_ndvi_physical_range(daily_collection.median()).rename("NDVI").toFloat()
    db_count = daily_collection.count().rename("unique_date_valid_count").toFloat()

    multiplicity = sw_count.subtract(db_count).rename("same_day_multiplicity")
    difference = db_median.subtract(sw_median).rename(
        "date_balanced_minus_scene_weighted"
    )
    return {
        "scene_weighted_median": sw_median.clip(region),
        "date_balanced_median": db_median.clip(region),
        "scene_valid_count": sw_count.clip(region),
        "unique_date_valid_count": db_count.clip(region),
        "same_day_multiplicity": multiplicity.clip(region),
        "date_balanced_minus_scene_weighted": difference.clip(region),
    }


def build_ee_images(ctx: dict, region) -> "OrderedDict":
    """Build every diagnostic EE image, keyed to match :func:`plan_raster_outputs`.

    Returns an ``OrderedDict`` of ``{output_name: ee.Image}``. Derived-comparison
    rasters are NOT included here -- they are computed on exported GeoTIFFs by
    :func:`compute_derived_comparison` using Step5 policy.
    """
    baseline_years = sorted(int(y) for y in ctx["baseline_years"])
    current_year = int(str(ctx["current_period_end_date"])[:4])
    baseline_years = [y for y in baseline_years if y < current_year]
    current = _current_window(ctx["current_period_end_date"], ctx["current_period_days"])

    images: "OrderedDict" = OrderedDict()

    def _register(prefix, role_images):
        for suffix, image in role_images.items():
            images[f"{prefix}_{suffix}"] = image

    # CURRENT LST (Celsius)
    _register(
        "current_lst",
        build_lst_role_images(
            region, current["start_date"], current["end_date"],
            (current["month_start"], current["month_end"]), to_celsius=True,
        ),
    )
    # BASELINE LST per year (DN, Step5 converts)
    for year in baseline_years:
        win = _baseline_year_window(
            ctx["current_period_end_date"], ctx["current_period_days"], year
        )
        _register(
            f"baseline_lst_{year}",
            build_lst_role_images(
                region, win["start_date"], win["end_date"], None, to_celsius=False,
            ),
        )
    # CURRENT NDVI
    _register(
        "current_ndvi",
        build_ndvi_role_images(
            region, current["start_date"], current["end_date"],
            (current["month_start"], current["month_end"]),
        ),
    )
    # BASELINE NDVI per year
    for year in baseline_years:
        win = _baseline_year_window(
            ctx["current_period_end_date"], ctx["current_period_days"], year
        )
        _register(
            f"baseline_ndvi_{year}",
            build_ndvi_role_images(
                region, win["start_date"], win["end_date"], None,
            ),
        )
    return images


# =============================================================================
# SOURCE-SCENE METADATA (lazy ee getInfo of METADATA ONLY -- no pixels)
# =============================================================================
def _scene_records_from_feature_collection(collection, role, baseline_year=None) -> list[dict]:
    """Turn an EE ImageCollection's per-scene metadata into explicit records.

    Only a cheap ``getInfo()`` on properties/geometry is performed -- never a
    pixel download. Scene IDs and footprints come straight from Earth Engine;
    none are invented.
    """
    import ee

    def _feature(image):
        image = ee.Image(image)
        return ee.Feature(image.geometry(), {
            "scene_id": image.get("system:index"),
            "landsat_product_id": image.get("LANDSAT_PRODUCT_ID"),
            "spacecraft_id": image.get("SPACECRAFT_ID"),
            "sensor_id": image.get("SENSOR_ID"),
            "wrs_path": image.get("WRS_PATH"),
            "wrs_row": image.get("WRS_ROW"),
            "acquisition_datetime": ee.Date(image.get("system:time_start")).format(
                "YYYY-MM-dd'T'HH:mm:ss"
            ),
            "cloud_cover": image.get("CLOUD_COVER"),
            "cloud_cover_land": image.get("CLOUD_COVER_LAND"),
            "processing_level": image.get("PROCESSING_LEVEL"),
            "collection_category": image.get("COLLECTION_CATEGORY"),
            "collection_number": image.get("COLLECTION_NUMBER"),
        })

    features = ee.FeatureCollection(collection.map(_feature)).getInfo()
    records = []
    for feature in features.get("features", []):
        props = dict(feature.get("properties", {}))
        acquired = props.get("acquisition_datetime")
        record = {
            "scene_id": props.get("scene_id"),
            "source_collection": LANDSAT_COLLECTION,
            "landsat_product_id": props.get("landsat_product_id"),
            "spacecraft_id": props.get("spacecraft_id"),
            "sensor_id": props.get("sensor_id"),
            "wrs_path": props.get("wrs_path"),
            "wrs_row": props.get("wrs_row"),
            "acquisition_datetime": acquired,
            "acquisition_date": str(acquired)[:10] if acquired else None,
            "cloud_cover": props.get("cloud_cover"),
            "cloud_cover_land": props.get("cloud_cover_land"),
            "processing_level": props.get("processing_level"),
            "collection_category": props.get("collection_category"),
            "collection_number": props.get("collection_number"),
            "geometry": feature.get("geometry"),
            "footprint": feature.get("geometry"),
            "input_role": role,
            "role": role,
            "baseline_year": baseline_year,
        }
        records.append(record)
    return records


def build_source_scene_metadata(ctx: dict, region) -> dict:
    """Assemble the explicit source-scene manifest by querying EE metadata.

    Produces a ``source_scene_metadata.json``-shaped dict whose ``collections``
    are consumable by the existing read-only ``source_scene_provenance`` schema.
    """
    baseline_years = sorted(int(y) for y in ctx["baseline_years"])
    current_year = int(str(ctx["current_period_end_date"])[:4])
    baseline_years = [y for y in baseline_years if y < current_year]
    current = _current_window(ctx["current_period_end_date"], ctx["current_period_days"])

    collections: "OrderedDict[str, list]" = OrderedDict()

    def _fc(start, end, month_range, bands):
        return _base_filtered_collection(region, start, end, month_range, bands)

    collections["current_lst"] = _scene_records_from_feature_collection(
        _fc(current["start_date"], current["end_date"],
            (current["month_start"], current["month_end"]), ["ST_B10", "QA_PIXEL"]),
        "current_lst",
    )
    collections["current_ndvi"] = _scene_records_from_feature_collection(
        _fc(current["start_date"], current["end_date"],
            (current["month_start"], current["month_end"]),
            ["SR_B4", "SR_B5", "QA_PIXEL"]),
        "current_ndvi",
    )
    baseline_lst = []
    baseline_ndvi = []
    for year in baseline_years:
        win = _baseline_year_window(
            ctx["current_period_end_date"], ctx["current_period_days"], year
        )
        baseline_lst.extend(_scene_records_from_feature_collection(
            _fc(win["start_date"], win["end_date"], None, ["ST_B10", "QA_PIXEL"]),
            "baseline_lst", baseline_year=year,
        ))
        baseline_ndvi.extend(_scene_records_from_feature_collection(
            _fc(win["start_date"], win["end_date"], None, ["SR_B4", "SR_B5", "QA_PIXEL"]),
            "baseline_ndvi", baseline_year=year,
        ))
    collections["baseline_lst"] = baseline_lst
    collections["baseline_ndvi"] = baseline_ndvi

    per_role_counts = OrderedDict(
        (role, count_semantics(records)) for role, records in collections.items()
    )
    return OrderedDict(
        [
            ("audit", DIAGNOSTIC_NAMESPACE),
            ("experiment_id", ctx["experiment_id"]),
            ("source_collection", LANDSAT_COLLECTION),
            ("qa_mask", qa_mask_provenance()),
            ("scene_id_source", "earth_engine system:index (not invented)"),
            ("footprint_source", "earth_engine image.geometry() (not invented)"),
            ("collections", collections),
            ("count_semantics", per_role_counts),
            ("created_at", datetime.now(timezone.utc).isoformat()),
        ]
    )


# =============================================================================
# NEIGHBOUR-JUMP + SUPPORT-EDGE SEAM METRICS (raster; no smoothing/blending)
# =============================================================================
def neighbour_jump_metrics(array) -> dict:
    """Distribution of absolute/signed adjacent-pixel jumps over valid pairs.

    Operates on RAW values only: adjacent horizontal and vertical pixel pairs
    where BOTH are finite. No smoothing, blending, or normalization is applied.
    """
    import numpy as np

    a = np.asarray(array, dtype="float64")
    diffs = []
    # horizontal neighbours
    left, right = a[:, :-1], a[:, 1:]
    hmask = np.isfinite(left) & np.isfinite(right)
    diffs.append((right - left)[hmask])
    # vertical neighbours
    top, bottom = a[:-1, :], a[1:, :]
    vmask = np.isfinite(top) & np.isfinite(bottom)
    diffs.append((bottom - top)[vmask])
    signed = np.concatenate(diffs) if diffs else np.array([])
    absolute = np.abs(signed)
    valid = int(signed.size)
    if valid == 0:
        return {
            "valid_pair_count": 0,
            "absolute_jump_median": None,
            "absolute_jump_p90": None,
            "absolute_jump_p95": None,
            "absolute_jump_p99": None,
            "signed_jump_median": None,
        }
    return {
        "valid_pair_count": valid,
        "absolute_jump_median": float(np.median(absolute)),
        "absolute_jump_p90": float(np.percentile(absolute, 90)),
        "absolute_jump_p95": float(np.percentile(absolute, 95)),
        "absolute_jump_p99": float(np.percentile(absolute, 99)),
        "signed_jump_median": float(np.median(signed)),
    }


def valid_coverage_fraction(array) -> float:
    """Fraction of finite pixels in a raster array."""
    import numpy as np

    a = np.asarray(array, dtype="float64")
    return float(np.isfinite(a).mean()) if a.size else 0.0


# -----------------------------------------------------------------------------
# Explicit adjacency-edge representation (BLOCKING ISSUE 1 fix)
#
# An "adjacency edge" is the interface between two orthogonally-adjacent pixels.
# For an (H, W) raster there are two adjacency lattices:
#   * horizontal, shape (H, W-1): edge between (r, c) and (r, c+1)
#   * vertical,   shape (H-1, W): edge between (r, c) and (r+1, c)
# Every boundary definition is a boolean mask over these lattices, and the SAME
# mask is applied to BOTH chains, so scene_weighted and date_balanced jumps are
# always measured at IDENTICAL row/column adjacency indices. A pair is retained
# only when BOTH chains are finite at BOTH endpoints; otherwise it is dropped
# with an explicit skipped reason. Zero is NEVER substituted.
# -----------------------------------------------------------------------------
ORIENTATIONS = ("horizontal", "vertical")


def _edge_pairs(arr, orientation):
    """Return the (a, b) endpoint views for one adjacency lattice."""
    if orientation == "horizontal":
        return arr[:, :-1], arr[:, 1:]
    return arr[:-1, :], arr[1:, :]


def _anchor_indices(shape, orientation):
    """Row/col index of the 'a' (left/top) endpoint of every adjacency edge."""
    import numpy as np

    height, width = shape
    if orientation == "horizontal":
        return np.meshgrid(np.arange(height), np.arange(width - 1), indexing="ij")
    return np.meshgrid(np.arange(height - 1), np.arange(width), indexing="ij")


def _count_change_mask(count_array, orientation):
    """Adjacency edges where an integer support count differs across the edge."""
    import numpy as np

    a, b = _edge_pairs(np.asarray(count_array, dtype="float64"), orientation)
    return np.isfinite(a) & np.isfinite(b) & (a != b)


def build_edge_masks(scene_count, unique_date_count, multiplicity) -> dict:
    """Predeclared common adjacency-edge masks (identical for both chains).

    Returns ``{edge_name: {"horizontal": mask, "vertical": mask}}`` for:
      * ``scene_count_edge``          -- scene-observation-count changes
      * ``unique_date_count_edge``    -- unique acquisition-date support changes
      * ``same_day_multiplicity_edge``-- same-day multiplicity changes
      * ``union_support_edge``        -- union of scene-count and date-count edges
    """
    masks: dict[str, dict] = {}
    for name, arr in (
        ("scene_count_edge", scene_count),
        ("unique_date_count_edge", unique_date_count),
        ("same_day_multiplicity_edge", multiplicity),
    ):
        masks[name] = {o: _count_change_mask(arr, o) for o in ORIENTATIONS}
    masks["union_support_edge"] = {
        o: masks["scene_count_edge"][o] | masks["unique_date_count_edge"][o]
        for o in ORIENTATIONS
    }
    return masks


def sample_paired_edges(sw_array, db_array, edge_mask) -> dict:
    """Sample BOTH chains at the SAME adjacency edges selected by ``edge_mask``.

    ``edge_mask`` is ``{"horizontal": mask, "vertical": mask}``. A pair is
    retained iff the edge is selected AND all four endpoints (sw_a, sw_b, db_a,
    db_b) are finite. For every retained pair::

        sw_abs_jump    = abs(sw_b - sw_a)
        db_abs_jump    = abs(db_b - db_a)
        paired_reduction = sw_abs_jump - db_abs_jump

    Both chains are guaranteed to be sampled at identical (orientation, row,
    col) adjacency indices. Pairs missing an observation in either chain are
    dropped and counted under ``skipped_missing_observation`` -- never imputed.
    """
    import numpy as np

    sw = np.asarray(sw_array, dtype="float64")
    db = np.asarray(db_array, dtype="float64")
    parts: dict[str, list] = {
        k: [] for k in (
            "orientation", "row", "col", "sw_signed", "db_signed",
            "sw_abs", "db_abs", "reduction",
        )
    }
    skipped = 0
    for orientation in ORIENTATIONS:
        sa, sb = _edge_pairs(sw, orientation)
        da, db_b = _edge_pairs(db, orientation)
        selected = np.asarray(edge_mask[orientation], dtype=bool)
        both = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(da) & np.isfinite(db_b)
        retained = selected & both
        skipped += int(np.sum(selected & ~both))
        rr, cc = _anchor_indices(sw.shape, orientation)
        sw_signed = (sb - sa)[retained]
        db_signed = (db_b - da)[retained]
        n = int(retained.sum())
        parts["orientation"].append(np.full(n, orientation, dtype=object))
        parts["row"].append(rr[retained])
        parts["col"].append(cc[retained])
        parts["sw_signed"].append(sw_signed)
        parts["db_signed"].append(db_signed)
        parts["sw_abs"].append(np.abs(sw_signed))
        parts["db_abs"].append(np.abs(db_signed))
        parts["reduction"].append(np.abs(sw_signed) - np.abs(db_signed))

    def _cat(key, dtype="float64"):
        arrays = parts[key]
        if not arrays:
            return np.array([], dtype=dtype)
        return np.concatenate(arrays)

    return {
        "orientation": _cat("orientation", dtype=object),
        "row": _cat("row", dtype="int64").astype("int64"),
        "col": _cat("col", dtype="int64").astype("int64"),
        "sw_signed": _cat("sw_signed"),
        "db_signed": _cat("db_signed"),
        "sw_abs": _cat("sw_abs"),
        "db_abs": _cat("db_abs"),
        "reduction": _cat("reduction"),
        "pair_count": int(_cat("reduction").size),
        "skipped_missing_observation": skipped,
    }


def _percentile_block(abs_values, signed_values) -> dict:
    """valid pair count + signed/absolute jump median and p90/p95/p99."""
    import numpy as np

    absolute = np.asarray(abs_values, dtype="float64")
    signed = np.asarray(signed_values, dtype="float64")
    if absolute.size == 0:
        return {
            "valid_pair_count": 0,
            "signed_jump_median": None,
            "absolute_jump_median": None,
            "absolute_jump_p90": None,
            "absolute_jump_p95": None,
            "absolute_jump_p99": None,
        }
    return {
        "valid_pair_count": int(absolute.size),
        "signed_jump_median": float(np.median(signed)),
        "absolute_jump_median": float(np.median(absolute)),
        "absolute_jump_p90": float(np.percentile(absolute, 90)),
        "absolute_jump_p95": float(np.percentile(absolute, 95)),
        "absolute_jump_p99": float(np.percentile(absolute, 99)),
    }


def _ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    if abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def sample_matched_control(sw_array, db_array, exclude_mask, n_per_orientation, *, seed):
    """Interior/offset control adjacency pairs, matched by orientation and count.

    Controls use the SAME orientation as the boundary and a comparable segment
    length (matched pair count per orientation). Any adjacency in
    ``exclude_mask`` (e.g. the union support edge) is excluded so controls are
    genuine interior/offset pairs. Deterministic given ``seed``. Returns per-
    chain absolute jump arrays plus the per-orientation counts actually drawn.
    """
    import numpy as np

    sw = np.asarray(sw_array, dtype="float64")
    db = np.asarray(db_array, dtype="float64")
    rng = np.random.default_rng(seed)
    out = {"sw_abs": [], "db_abs": [], "sw_signed": [], "db_signed": [], "n_per_orientation": {}}
    for orientation in ORIENTATIONS:
        sa, sb = _edge_pairs(sw, orientation)
        da, db_b = _edge_pairs(db, orientation)
        excluded = np.asarray(exclude_mask[orientation], dtype=bool)
        eligible = (
            np.isfinite(sa) & np.isfinite(sb) & np.isfinite(da) & np.isfinite(db_b)
            & ~excluded
        )
        idx = np.flatnonzero(eligible.ravel())
        want = int(n_per_orientation.get(orientation, 0))
        if want <= 0 or idx.size == 0:
            out["n_per_orientation"][orientation] = 0
            continue
        take = rng.choice(idx, size=min(want, idx.size), replace=False)
        sw_signed = (sb - sa).ravel()[take]
        db_signed = (db_b - da).ravel()[take]
        out["sw_signed"].append(sw_signed)
        out["db_signed"].append(db_signed)
        out["sw_abs"].append(np.abs(sw_signed))
        out["db_abs"].append(np.abs(db_signed))
        out["n_per_orientation"][orientation] = int(take.size)

    def _cat(key):
        return np.concatenate(out[key]) if out[key] else np.array([], dtype="float64")

    return {
        "sw_abs": _cat("sw_abs"),
        "db_abs": _cat("db_abs"),
        "sw_signed": _cat("sw_signed"),
        "db_signed": _cat("db_signed"),
        "n_per_orientation": out["n_per_orientation"],
    }


def boundary_chain_metrics(sample, control, chain: str) -> dict:
    """Per (chain, boundary) jump distribution + matched-control comparison.

    ``sample`` is a :func:`sample_paired_edges` result; ``control`` a
    :func:`sample_matched_control` result. Reports valid pair count, signed and
    absolute medians, p90/p95/p99, the matched-control median/p95, and the
    boundary/control median and p95 ratios for ``chain`` in {sw, db}.
    """
    key = "sw" if chain == CHAIN_SCENE_WEIGHTED else "db"
    boundary = _percentile_block(sample[f"{key}_abs"], sample[f"{key}_signed"])
    control_block = _percentile_block(control[f"{key}_abs"], control[f"{key}_signed"])
    return {
        "chain": chain,
        **boundary,
        "control_absolute_jump_median": control_block["absolute_jump_median"],
        "control_absolute_jump_p95": control_block["absolute_jump_p95"],
        "control_pair_count": control_block["valid_pair_count"],
        "boundary_control_median_ratio": _ratio(
            boundary["absolute_jump_median"], control_block["absolute_jump_median"]
        ),
        "boundary_control_p95_ratio": _ratio(
            boundary["absolute_jump_p95"], control_block["absolute_jump_p95"]
        ),
    }


# -----------------------------------------------------------------------------
# Bootstrap over PREDECLARED units (BLOCKING ISSUE 2 fix)
#
# Row strips are no longer used. Raster-derived support edges are grouped into
# predeclared 2-D spatial blocks; each block intersecting the edge mask that
# contains >=1 retained paired pair is one bootstrap unit. Export-tile and
# provenance boundary units are passed in with explicit unit ids. The cluster
# bootstrap resamples UNITS with replacement and pools their paired reductions.
# -----------------------------------------------------------------------------
def assign_spatial_block_units(sample, *, block_size: int = 128) -> list[str]:
    """Deterministic 2-D spatial-block unit id for each retained paired pair."""
    import numpy as np

    rows = np.asarray(sample["row"], dtype="int64")
    cols = np.asarray(sample["col"], dtype="int64")
    br = rows // int(block_size)
    bc = cols // int(block_size)
    return [f"block_r{int(r)}_c{int(c)}" for r, c in zip(br, bc)]


def cluster_bootstrap_interval(unit_reductions, *, n_boot=10000, ci=0.95, seed=20240717):
    """Cluster (unit) percentile bootstrap of the pooled mean paired reduction.

    ``unit_reductions`` maps ``unit_id -> 1-D array of paired reductions``. Units
    (not pixels, not row strips) are resampled with replacement; their pooled
    reductions form each bootstrap mean.
    """
    import numpy as np

    unit_ids = sorted(unit_reductions)
    unit_arrays = [np.asarray(unit_reductions[u], dtype="float64") for u in unit_ids]
    unit_arrays = [a[np.isfinite(a)] for a in unit_arrays]
    unit_ids = [u for u, a in zip(unit_ids, unit_arrays) if a.size]
    unit_arrays = [a for a in unit_arrays if a.size]
    pooled = np.concatenate(unit_arrays) if unit_arrays else np.array([])
    n_units = len(unit_ids)
    result = {
        "n_units": n_units,
        "n_pairs": int(pooled.size),
        "n_boot": int(n_boot),
        "ci": float(ci),
        "seed": int(seed),
        "point_estimate": float(pooled.mean()) if pooled.size else None,
        "interval_low": None,
        "interval_high": None,
    }
    if n_units < 2:
        return result
    rng = np.random.default_rng(seed)
    boot_means = np.empty(int(n_boot), dtype="float64")
    index = np.arange(n_units)
    for b in range(int(n_boot)):
        chosen = rng.choice(index, size=n_units, replace=True)
        boot_means[b] = np.concatenate([unit_arrays[i] for i in chosen]).mean()
    lo_q = (1.0 - ci) / 2.0
    result["interval_low"] = float(np.quantile(boot_means, lo_q))
    result["interval_high"] = float(np.quantile(boot_means, 1.0 - lo_q))
    return result


def classify_paired_units(
    unit_reductions,
    *,
    unit_type: str,
    min_units: int = 8,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 20240717,
) -> dict:
    """Paired-comparison verdict over predeclared bootstrap units.

    ``insufficient_evidence`` when fewer than ``min_units`` units exist;
    otherwise the cluster-bootstrap interval decides via
    :func:`classify_paired_interval`. Reports unit type and the unit ids.
    """
    interval = cluster_bootstrap_interval(
        unit_reductions, n_boot=n_boot, ci=ci, seed=seed
    )
    if interval["n_units"] < int(min_units):
        status = "insufficient_evidence"
    else:
        status = classify_paired_interval(
            interval["interval_low"], interval["interval_high"]
        )
    return {
        "status": status,
        "unit_type": unit_type,
        "unit_ids": sorted(u for u, a in unit_reductions.items() if len(a)),
        "min_units_required": int(min_units),
        **interval,
        "reduction_definition": (
            "absolute_jump_scene_weighted - absolute_jump_date_balanced "
            "(positive => date_balanced reduces the jump)"
        ),
    }


def paired_reduction_by_blocks(sample, *, unit_type="raster_support_block", block_size=128, **kwargs) -> dict:
    """Group a paired sample into spatial-block units and classify the paired
    reduction. Never uses row strips; never imputes zero for dropped pairs.
    """
    import numpy as np

    unit_ids = assign_spatial_block_units(sample, block_size=block_size)
    reductions = np.asarray(sample["reduction"], dtype="float64")
    grouped: dict[str, list] = {}
    for uid, value in zip(unit_ids, reductions):
        grouped.setdefault(uid, []).append(value)
    grouped = {uid: np.asarray(vals, dtype="float64") for uid, vals in grouped.items()}
    return classify_paired_units(grouped, unit_type=unit_type, **kwargs)


def _orientation_counts(sample) -> dict:
    """Per-orientation retained pair counts for a paired sample."""
    import numpy as np

    orientation = np.asarray(sample["orientation"], dtype=object)
    return {o: int(np.sum(orientation == o)) for o in ORIENTATIONS}


# =============================================================================
# PER-PRODUCT BOUNDARY AUDIT (BLOCKING ISSUE 3) -- all boundary types together
# =============================================================================
def audit_product_boundaries(
    product: str,
    sw_values,
    db_values,
    edge_masks: dict,
    *,
    tile_units: dict | None = None,
    provenance_status: str = "insufficient_boundary_metadata",
    provenance_edge_mask: dict | None = None,
    control_seed: int = 20240717,
    block_size: int = 128,
    min_units: int = 8,
    n_boot: int = 10000,
    ci: float = 0.95,
) -> dict:
    """Compute per (product, chain, boundary_type) metrics + paired verdicts.

    Support edges (scene-count / unique-date / same-day-multiplicity) come from
    ``edge_masks``; matched controls exclude the union support edge and use the
    SAME orientation and count-matched length. Export-tile boundaries (if
    ``tile_units`` given) are the negative control, one bootstrap unit per real
    seam. Verified source-scene/path-row boundaries are only evaluated when
    ``provenance_status == 'provenance_available'`` and a rasterized
    ``provenance_edge_mask`` is supplied; otherwise they are reported as
    non-evidence, never fabricated.
    """
    metric_rows: list[dict] = []
    verdicts: "OrderedDict" = OrderedDict()
    union_exclude = edge_masks["union_support_edge"]

    def _emit(boundary_type, sample):
        control = sample_matched_control(
            sw_values, db_values, union_exclude,
            _orientation_counts(sample), seed=control_seed,
        )
        for chain in CHAINS:
            metric_rows.append({
                "product": product,
                "boundary_type": boundary_type,
                "skipped_missing_observation": sample["skipped_missing_observation"],
                **boundary_chain_metrics(sample, control, chain),
            })

    # --- support-edge boundary types ---
    for boundary_type in ("scene_count_edge", "unique_date_count_edge", "same_day_multiplicity_edge"):
        sample = sample_paired_edges(sw_values, db_values, edge_masks[boundary_type])
        _emit(boundary_type, sample)
        verdicts[boundary_type] = paired_reduction_by_blocks(
            sample, unit_type="raster_support_block", block_size=block_size,
            min_units=min_units, n_boot=n_boot, ci=ci, seed=control_seed,
        )

    # --- export-tile negative control ---
    if tile_units:
        unit_reductions: dict[str, list] = {}
        pooled = sample_paired_edges(
            sw_values, db_values,
            {o: _union_tile_mask(tile_units, o) for o in ORIENTATIONS},
        )
        _emit("export_tile_boundary", pooled)
        for uid, mask in tile_units.items():
            seg = sample_paired_edges(sw_values, db_values, mask)
            unit_reductions[uid] = seg["reduction"]
        verdicts["export_tile_boundary"] = classify_paired_units(
            unit_reductions, unit_type="export_tile_segment",
            min_units=min_units, n_boot=n_boot, ci=ci, seed=control_seed,
        )
    else:
        verdicts["export_tile_boundary"] = {
            "status": "unavailable",
            "unit_type": "export_tile_segment",
            "reason": "product exported directly or paired grids differ",
        }

    # --- verified source-scene / path-row boundary ---
    if provenance_status == "provenance_available" and provenance_edge_mask:
        sample = sample_paired_edges(sw_values, db_values, provenance_edge_mask)
        _emit("source_scene_path_row", sample)
        verdicts["source_scene_path_row"] = paired_reduction_by_blocks(
            sample, unit_type="provenance_boundary", block_size=block_size,
            min_units=min_units, n_boot=n_boot, ci=ci, seed=control_seed,
        )
    else:
        verdicts["source_scene_path_row"] = {
            "status": provenance_status,
            "unit_type": "provenance_boundary",
            "is_verified_source_boundary_evidence": False,
            "reason": (
                "verified source-scene/path-row boundaries require "
                "provenance_available and a rasterized provenance edge mask"
            ),
        }

    return {"product": product, "metric_rows": metric_rows, "verdicts": verdicts}


def _union_tile_mask(tile_units: dict, orientation: str):
    import numpy as np

    masks = [m[orientation] for m in tile_units.values()]
    if not masks:
        return None
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out |= m
    return out


# =============================================================================
# GRID CONTRACT (BLOCKING ISSUE 5) -- no resampling; fail fast on mismatch
# =============================================================================
class GridMismatchError(RuntimeError):
    """Raised when required rasters do not share an exact grid signature."""


def grid_signature(path) -> dict:
    """Exact grid signature: CRS, width, height, affine transform, bounds."""
    import rasterio

    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "crs": str(src.crs),
            "width": int(src.width),
            "height": int(src.height),
            "transform": [float(v) for v in tuple(src.transform)[:6]],
            "bounds": [float(v) for v in tuple(src.bounds)],
        }


def _signatures_match(a: dict, b: dict, *, atol: float = 1e-9) -> bool:
    if (a["crs"], a["width"], a["height"]) != (b["crs"], b["width"], b["height"]):
        return False
    if any(abs(x - y) > atol for x, y in zip(a["transform"], b["transform"])):
        return False
    return all(abs(x - y) <= atol for x, y in zip(a["bounds"], b["bounds"]))


def assert_same_grid(paths) -> dict:
    """Assert every raster shares the exact grid signature; return it.

    Raises :class:`GridMismatchError` (never resamples) if CRS, width, height,
    transform, or bounds differ.
    """
    paths = list(paths)
    if not paths:
        raise GridMismatchError("assert_same_grid requires at least one raster")
    reference = grid_signature(paths[0])
    for path in paths[1:]:
        candidate = grid_signature(path)
        if not _signatures_match(reference, candidate):
            raise GridMismatchError(
                "grid_mismatch: "
                f"{candidate['path']} (crs={candidate['crs']}, "
                f"{candidate['width']}x{candidate['height']}, "
                f"transform={candidate['transform']}, bounds={candidate['bounds']}) "
                f"!= reference {reference['path']} (crs={reference['crs']}, "
                f"{reference['width']}x{reference['height']}, "
                f"transform={reference['transform']}, bounds={reference['bounds']})"
            )
    return reference


# =============================================================================
# NODATA / AOI-EDGE SAFETY
# =============================================================================
def read_masked_array(path, band: int = 1):
    """Read a band as float64 with masked/nodata pixels set to NaN.

    Honours both the GeoTIFF nodata tag and the ``NODATA_SENTINEL``, so masked /
    AOI-exterior pixels become NaN and are excluded from every metric -- they
    are never read as physical numeric zero.
    """
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(band, masked=True).astype("float64")
    filled = arr.filled(np.nan)
    filled = np.where(filled == NODATA_SENTINEL, np.nan, filled)
    return filled


def validate_nodata_mask(path, *, expected_nodata=NODATA_SENTINEL, band: int = 1) -> dict:
    """Post-export check that a raster carries a verifiable nodata mask and that
    masked pixels equal the sentinel (not silently zero).
    """
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        nodata = src.nodata
        arr = src.read(band, masked=True)
        mask = np.ma.getmaskarray(arr)
        raw = np.asarray(arr.filled(expected_nodata))
    masked_count = int(mask.sum())
    # Masked cells must not read as physical 0 through the nodata tag.
    masked_as_zero = bool(masked_count and np.any(raw[mask] == 0.0) and expected_nodata != 0.0)
    has_tag = nodata is not None
    status = "ok"
    if not has_tag:
        status = "missing_nodata_tag"
    elif masked_as_zero:
        status = "masked_read_as_zero"
    return {
        "path": str(path),
        "nodata_tag": None if nodata is None else float(nodata),
        "expected_nodata": float(expected_nodata),
        "masked_pixel_count": masked_count,
        "masked_read_as_zero": masked_as_zero,
        "status": status,
    }


# =============================================================================
# EXPORT-TILE NEGATIVE CONTROL (from the ACTUAL recorded export grid)
# =============================================================================
def export_tile_boundary_edge_masks(width: int, height: int, tile_grid) -> dict:
    """Adjacency-edge masks along the recorded export tile-grid seams.

    ``tile_grid`` is ``[rows, cols]`` as recorded by
    ``export_image_direct_or_tiled``. Returns a per-boundary-segment dict
    ``{unit_id: {"horizontal": mask, "vertical": mask}}`` so each real tile seam
    is one bootstrap unit. Returns ``{}`` when there is no interior seam.
    """
    import numpy as np

    if not (isinstance(tile_grid, (list, tuple)) and len(tile_grid) == 2):
        return {}
    rows, cols = int(tile_grid[0]), int(tile_grid[1])
    units: dict[str, dict] = {}
    # vertical seams (between tile columns) -> a vertical column edge index
    for col in range(1, cols):
        idx = round(width * col / cols)
        if not (0 < idx < width):
            continue
        h = np.zeros((height, width - 1), dtype=bool)
        v = np.zeros((height - 1, width), dtype=bool)
        h[:, idx - 1] = True  # edge between col idx-1 and idx
        units[f"tile_v{col}"] = {"horizontal": h, "vertical": v}
    # horizontal seams (between tile rows) -> a horizontal row edge index
    for row in range(1, rows):
        idx = round(height * row / rows)
        if not (0 < idx < height):
            continue
        h = np.zeros((height, width - 1), dtype=bool)
        v = np.zeros((height - 1, width), dtype=bool)
        v[idx - 1, :] = True  # edge between row idx-1 and idx
        units[f"tile_h{row}"] = {"horizontal": h, "vertical": v}
    return units


def export_tile_control_availability(sw_inventory: dict, db_inventory: dict) -> dict:
    """Whether an export-tile negative control is valid for a compared pair.

    Uses ONLY the recorded transport/tile_grid. If either product was exported
    directly (no interior seam), the control is unavailable. If the two products
    used different tile grids, their tile boundaries are NOT paired -- reported
    as insufficient rather than pretending they align.
    """
    sw_grid = sw_inventory.get("tile_grid")
    db_grid = db_inventory.get("tile_grid")
    sw_tiled = sw_inventory.get("transport", "").startswith("tiled") and sw_grid
    db_tiled = db_inventory.get("transport", "").startswith("tiled") and db_grid
    if not (sw_tiled and db_tiled):
        return {
            "available": False,
            "reason": "at_least_one_product_exported_directly",
            "scene_weighted_transport": sw_inventory.get("transport"),
            "date_balanced_transport": db_inventory.get("transport"),
        }
    if list(sw_grid) != list(db_grid):
        return {
            "available": False,
            "reason": "paired_products_used_different_tile_grids",
            "scene_weighted_tile_grid": sw_grid,
            "date_balanced_tile_grid": db_grid,
        }
    return {"available": True, "tile_grid": list(sw_grid)}


# =============================================================================
# PROVENANCE STATUS MAPPING (kept separate from support-count evidence)
# =============================================================================
def map_provenance_status(provenance_summary: dict | None) -> str:
    """Map a source_scene_provenance summary status onto the three explicit
    diagnostic states, never inventing a completed source-boundary audit.
    """
    if not provenance_summary:
        return "insufficient_boundary_metadata"
    status = provenance_summary.get("status")
    if status == "available":
        return "provenance_available"
    if status == "metadata_available_boundary_incomplete":
        return "provenance_incomplete"
    return "insufficient_boundary_metadata"


def assert_scene_list_nonempty(metadata: dict) -> int:
    """Preflight: fail before the raster-export loop if metadata has no scenes."""
    collections = metadata.get("collections", {})
    total = sum(len(records) for records in collections.values())
    if total == 0:
        raise RuntimeError(
            "source_scene_metadata extraction returned an EMPTY scene list; "
            "aborting before the raster-export loop."
        )
    return total


# =============================================================================
# CANONICAL REPRODUCTION GATE (BLOCKING ISSUE 4)
# =============================================================================
def resolve_canonical_paths(ctx: dict) -> "OrderedDict":
    """Resolve frozen canonical paths from ExperimentContext + predictor export
    metadata (NOT filename guessing).

    Prefers exact paths recorded in ``predictor_export_metadata.json``; falls
    back to the ExperimentContext canonical filenames. Every entry records how
    it was resolved.
    """
    output_root = Path(ctx["output_root"])
    export_meta_path = output_root / "predictor_export_metadata.json"
    export_meta = {}
    if export_meta_path.exists():
        try:
            export_meta = json.loads(export_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            export_meta = {}
    exports = export_meta.get("exports", {})

    baseline_years = sorted(int(y) for y in ctx["baseline_years"])
    current_year = int(str(ctx["current_period_end_date"])[:4])
    baseline_years = [y for y in baseline_years if y < current_year]

    resolved: "OrderedDict" = OrderedDict()

    def _resolve(label, ctx_fallback: Path):
        entry = exports.get(label)
        if entry and entry.get("path"):
            return {"path": entry["path"], "resolution": f"predictor_export_metadata:{label}"}
        return {"path": str(ctx_fallback), "resolution": "experiment_context_filename"}

    days = ctx["current_period_days"]
    resolved["current_lst"] = _resolve(
        "current_lst",
        Path(ctx["current_period_dir"]) / f"landsat_current_period_{days}days.tif",
    )
    for year in baseline_years:
        win = _baseline_year_window(ctx["current_period_end_date"], days, year)
        resolved[f"baseline_lst_{year}"] = _resolve(
            f"baseline_lst_{year}",
            Path(ctx["baseline_input_dir"])
            / f"{ctx['landsat_file_prefix']}_baseline_{win['end_date']}.tif",
        )
    step5_dir = Path(ctx["step5_output_dir"])
    resolved["step5_baseline_mean"] = {
        "path": str(step5_dir / "baseline_lst_mean_celsius.tif"),
        "resolution": "step5_canonical_filename",
    }
    resolved["step5_baseline_std"] = {
        "path": str(step5_dir / "baseline_lst_std_celsius.tif"),
        "resolution": "step5_canonical_filename",
    }
    resolved["step5_anomaly_zscore"] = {
        "path": str(step5_dir / "anomaly_zscore.tif"),
        "resolution": "step5_canonical_filename",
    }
    return resolved


def compare_raster_to_canonical(
    diagnostic_path,
    canonical_path,
    *,
    tolerance: float,
    diagnostic_band: int = 1,
    canonical_band: int = 1,
) -> dict:
    """Compare a diagnostic raster to a frozen canonical raster.

    Requires EXACT grid equality (CRS, width, height, transform). Reports
    valid-mask agreement, compared pixel count, max/mean absolute difference,
    the tolerance, and a reproduction status of ``pass`` / ``fail`` /
    ``grid_mismatch`` / ``not_available``.
    """
    import numpy as np

    if not Path(canonical_path).exists():
        return {"reproduction_status": "not_available", "canonical_path": str(canonical_path)}
    if not Path(diagnostic_path).exists():
        return {"reproduction_status": "not_available", "diagnostic_path": str(diagnostic_path)}
    diag_sig = grid_signature(diagnostic_path)
    canon_sig = grid_signature(canonical_path)
    if (diag_sig["crs"], diag_sig["width"], diag_sig["height"]) != (
        canon_sig["crs"], canon_sig["width"], canon_sig["height"]
    ) or any(abs(a - b) > 1e-9 for a, b in zip(diag_sig["transform"], canon_sig["transform"])):
        return {
            "reproduction_status": "grid_mismatch",
            "diagnostic_grid": diag_sig,
            "canonical_grid": canon_sig,
            "tolerance": tolerance,
        }
    diag = read_masked_array(diagnostic_path, diagnostic_band)
    canon = read_masked_array(canonical_path, canonical_band)
    diag_valid = np.isfinite(diag)
    canon_valid = np.isfinite(canon)
    mask_agreement = float(np.mean(diag_valid == canon_valid))
    both = diag_valid & canon_valid
    compared = int(both.sum())
    if compared == 0:
        return {
            "reproduction_status": "fail",
            "reason": "no_common_valid_pixels",
            "valid_mask_agreement": mask_agreement,
            "compared_pixel_count": 0,
            "tolerance": tolerance,
        }
    abs_diff = np.abs(diag[both] - canon[both])
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    status = "pass" if max_abs <= tolerance else "fail"
    return {
        "reproduction_status": status,
        "valid_mask_agreement": mask_agreement,
        "compared_pixel_count": compared,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "tolerance": tolerance,
        "canonical_path": str(canonical_path),
        "diagnostic_path": str(diagnostic_path),
    }


def run_canonical_reproduction_gate(ctx: dict, root: Path, raster_plan: dict) -> dict:
    """Compare scene_weighted diagnostic outputs to frozen canonical products.

    Returns ``{"status": ..., "checks": {...}}`` where status is ``pass``,
    ``canonical_reproduction_failed``, or ``not_available``. For manavgat_2021
    the frozen files are REQUIRED, so a missing canonical file fails the gate.
    """
    canonical = resolve_canonical_paths(ctx)
    root = Path(root)
    required = ctx["experiment_id"] == "manavgat_2021"
    tol_dn = REPRODUCTION_TOLERANCES["dn_count"]
    tol_phys = REPRODUCTION_TOLERANCES["physical_float32"]

    checks: "OrderedDict" = OrderedDict()
    # current LST median (band 1, Celsius) + valid count (band 2)
    checks["current_lst_median"] = compare_raster_to_canonical(
        root / raster_plan["current_lst_scene_weighted_median"],
        canonical["current_lst"]["path"], tolerance=tol_phys,
        canonical_band=1,
    )
    checks["current_lst_valid_count"] = compare_raster_to_canonical(
        root / raster_plan["current_lst_scene_valid_count"],
        canonical["current_lst"]["path"], tolerance=tol_dn,
        canonical_band=2,
    )
    # baseline yearly scene_weighted DN medians
    for key, entry in canonical.items():
        if not key.startswith("baseline_lst_"):
            continue
        diag_name = f"{key}_scene_weighted_median"
        if diag_name in raster_plan:
            checks[key] = compare_raster_to_canonical(
                root / raster_plan[diag_name], entry["path"], tolerance=tol_dn,
            )
    # derived Step5 baseline mean/std + anomaly z-score (scene_weighted chain)
    derived_dir = root / "derived" / CHAIN_SCENE_WEIGHTED
    checks["derived_baseline_mean"] = compare_raster_to_canonical(
        derived_dir / "baseline_lst_mean_celsius.tif",
        canonical["step5_baseline_mean"]["path"], tolerance=tol_phys,
    )
    checks["derived_baseline_std"] = compare_raster_to_canonical(
        derived_dir / "baseline_lst_std_celsius.tif",
        canonical["step5_baseline_std"]["path"], tolerance=tol_phys,
    )
    checks["derived_anomaly_zscore"] = compare_raster_to_canonical(
        derived_dir / "anomaly_zscore.tif",
        canonical["step5_anomaly_zscore"]["path"], tolerance=tol_phys,
    )

    statuses = [c.get("reproduction_status") for c in checks.values()]
    if "fail" in statuses or "grid_mismatch" in statuses:
        gate = "canonical_reproduction_failed"
    elif "not_available" in statuses:
        gate = "canonical_reproduction_failed" if required else "not_available"
    else:
        gate = "pass"
    return {
        "status": gate,
        "required_frozen_files": required,
        "canonical_paths": canonical,
        "checks": checks,
    }


# =============================================================================
# FORCE SEMANTICS -- clear ONLY the diagnostic namespace
# =============================================================================
def clear_diagnostic_namespace(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> str | None:
    """Delete ONLY ``outputs/diagnostics/.../<experiment_id>/`` (incl. tiles).

    Namespace-safety is asserted against the resolved path before any removal,
    so a computed path outside the diagnostic root can never be deleted.
    """
    import shutil

    root = diagnostic_output_root(experiment_id, base_dir)
    # Assert the target IS exactly the diagnostic root before deleting anything.
    assert_diagnostic_namespace_safe([root], experiment_id, base_dir)
    if root.exists():
        shutil.rmtree(root)
        return str(root)
    return None


# =============================================================================
# DERIVED COMPARISON (Step5 policy, diagnostic-only, reuses canonical math)
# =============================================================================
def compute_derived_comparison(
    chain: str,
    *,
    baseline_year_median_paths: list[Path],
    current_median_celsius_path: Path,
    current_count_path: Path,
    out_dir: Path,
) -> dict:
    """Diagnostic-only baseline mean/std, raw difference, and z-score for one
    chain, applying the SAME Step5 policies (:func:`step5_policy_snapshot`).

    Reuses the canonical Step5 math (``dn_to_celsius``, physical masking,
    nan-mean/std) READ-ONLY. Writes five rasters under ``out_dir`` (which must
    already be namespace-checked). Returns summary statistics.
    """
    import numpy as np
    import rasterio

    from src.step5_preprocess_timeseries import (
        dn_to_celsius,
        mask_physical_celsius,
        nanmean_float32,
        nanstd_float32,
        output_profile,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GRID CONTRACT: every input raster must share the exact grid (no resampling).
    assert_same_grid(
        list(baseline_year_median_paths)
        + [current_median_celsius_path, current_count_path]
    )

    with rasterio.open(baseline_year_median_paths[0]) as ref:
        profile = ref.profile.copy()
    profile = output_profile(profile)

    # Baseline yearly stack (DN -> Celsius, physical mask), one layer per year.
    # read_masked_array honours the NODATA_SENTINEL so exterior pixels are NaN,
    # never physical zero.
    layers = []
    for path in baseline_year_median_paths:
        dn = read_masked_array(path, 1)
        layers.append(mask_physical_celsius(dn_to_celsius(dn)))
    stack = np.stack(layers, axis=0)

    valid_year_count = np.sum(np.isfinite(stack), axis=0).astype("float32")
    baseline_mean = nanmean_float32(stack)
    baseline_std = nanstd_float32(stack)
    enough_years = valid_year_count >= STEP5_MIN_BASELINE_VALID_COUNT
    baseline_mean = np.where(enough_years, baseline_mean, np.nan).astype("float32")
    baseline_std = np.where(enough_years, baseline_std, np.nan).astype("float32")

    current = mask_physical_celsius(read_masked_array(current_median_celsius_path, 1))
    current_count = read_masked_array(current_count_path, 1)
    enough_current = current_count >= STEP5_MIN_CURRENT_VALID_COUNT
    current = np.where(enough_current, current, np.nan).astype("float32")

    difference = np.where(
        enough_current, current - baseline_mean, np.nan
    ).astype("float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = np.where(
            enough_current & (baseline_std >= STEP5_MIN_BASELINE_STD_CELSIUS),
            (current - baseline_mean) / baseline_std,
            np.nan,
        ).astype("float32")

    written = {}
    for name, array in [
        ("baseline_lst_mean_celsius.tif", baseline_mean),
        ("baseline_lst_std_celsius.tif", baseline_std),
        ("baseline_valid_year_count.tif", valid_year_count),
        ("current_minus_baseline_celsius.tif", difference),
        ("anomaly_zscore.tif", zscore),
    ]:
        path = out_dir / name
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array, 1)
        written[name] = str(path)

    def _stat(a):
        finite = a[np.isfinite(a)]
        return {
            "valid_pixels": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else None,
        }

    return {
        "chain": chain,
        "policy": step5_policy_snapshot(),
        "written": written,
        "baseline_mean": _stat(baseline_mean),
        "baseline_std": _stat(baseline_std),
        "difference": _stat(difference),
        "anomaly_zscore": _stat(zscore),
    }


# =============================================================================
# DIAGNOSTIC PNG MAPS (one shared robust stretch per compared pair)
# =============================================================================
def render_pair_maps(
    scene_weighted_path: Path,
    date_balanced_path: Path,
    out_dir: Path,
    *,
    pair_name: str,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> list[str]:
    """Render both members of a compared pair with ONE shared robust stretch.

    The colour range is a single robust percentile stretch computed over BOTH
    rasters together, so the maps are directly comparable. No per-tile
    normalization, smoothing, blending, or Gaussian blur is applied
    (``interpolation="nearest"``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for label, path in (("scene_weighted", scene_weighted_path), ("date_balanced", date_balanced_path)):
        with rasterio.open(path) as src:
            arrays[label] = src.read(1, masked=True).astype("float32").filled(np.nan)
    combined = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays.values()])
    if combined.size:
        vmin = float(np.percentile(combined, percentile_low))
        vmax = float(np.percentile(combined, percentile_high))
    else:
        vmin, vmax = 0.0, 1.0

    written = []
    for label, array in arrays.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(array, vmin=vmin, vmax=vmax, cmap="magma", interpolation="nearest")
        ax.set_title(f"{pair_name}\n{label} (shared stretch [{vmin:.3g}, {vmax:.3g}])")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        stem = Path(scene_weighted_path if label == "scene_weighted" else date_balanced_path).stem
        path = out_dir / f"{stem}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written
