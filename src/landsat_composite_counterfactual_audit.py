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

# -----------------------------------------------------------------------------
# CANONICAL REPRODUCTION GATE VERSIONING + SEMANTIC TOLERANCES
# -----------------------------------------------------------------------------
# The decisive canonical-reproduction gate has a schema/version identifier so
# checkpointed results computed under an older gate implementation are
# invalidated and recomputed WITHOUT rerunning Earth Engine (valid raster
# exports and boundary metrics are still reused). Bump this string whenever the
# gate SEMANTICS change.
#
# v1  -- raw-raster equality gate (compare_raster_to_canonical only). It failed
#        with mask_mismatch whenever the canonical annual baseline TIFFs
#        serialized GEE-masked pixels as raw DN 0 (no nodata tag) while the
#        diagnostic exports used an explicit -9999 nodata sentinel -- a
#        demonstrated FALSE NEGATIVE, because after the canonical Step5
#        DN->Celsius->physical-mask the two are pixel-identical.
# v2  -- two-layer gate: a diagnostic raw-encoding comparison (never fails on
#        its own) plus a decisive pipeline-semantic reproduction that applies the
#        exact production Step5 semantics before comparing valid masks/values,
#        classifies anomaly threshold-boundary mask disagreements explicitly, and
#        bounds anomaly value differences by a propagated numerical tolerance.
CANONICAL_GATE_VERSION = "2.0-pipeline-semantic"

# Decisive semantic gate name recorded in the JSON output.
DECISIVE_SEMANTIC_GATE = "pipeline_semantic_reproduction"

# Per-pixel semantic tolerance for the annual baseline LST comparison AFTER the
# production Step5 DN->Celsius->physical-mask has been applied to raw values.
# This is effectively exact (values are pixel-identical in practice); a tiny
# float allowance covers only DN->Celsius float rounding.
BASELINE_SEMANTIC_MAX_ABS_DIFF = 1e-6

# Predeclared float32 REDUCTION reproducibility tolerance for the scene-weighted
# derived baseline mean/std. This is a NUMERICAL reproducibility tolerance for
# order-of-operations / float32 rounding differences between the diagnostic
# windowed Step5 reduction and the canonical Step5 reduction -- it is NOT a
# scientific effect-size threshold. (Recorded as such in JSON + Markdown.)
DERIVED_REDUCTION_TOLERANCE = 5e-5

# Accepted upstream reproduction tolerances (all the same float32 reduction
# tolerance) used to (a) decide the anomaly std threshold neighbourhood and
# (b) propagate a per-pixel bound on the anomaly z-score value difference.
ANOMALY_CURRENT_TOL = DERIVED_REDUCTION_TOLERANCE
ANOMALY_MEAN_TOL = DERIVED_REDUCTION_TOLERANCE
ANOMALY_STD_TOL = DERIVED_REDUCTION_TOLERANCE

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
# REPORT/METADATA CONSISTENCY (pure)
#
# Report-schema version. Bump when the shape/wording of the generated summary
# (limitations, QA masking-rule text, provenance qualification) changes so a
# --finalize-only refresh regenerates ONLY the affected report/provenance stages
# and reuses every valid boundary/raster/derived calculation unchanged.
# =============================================================================
REPORT_SCHEMA_VERSION = "1.1-limitations-qa-provenance"

# The QA masking that the canonical ``apply_qa_mask`` ACTUALLY performs. Every
# generated provenance/report field must state this, never "QA_PIXEL + QA_RADSAT".
QA_MASKING_RULE_TEXT = "QA_PIXEL cloud/shadow/snow/fill masking; QA_RADSAT not applied"


def correct_composite_masking_rules(config: dict) -> dict:
    """Return a deep copy of a provenance config whose every composite
    ``masking_rules`` states the ACTUAL canonical behaviour (QA_PIXEL only, no
    QA_RADSAT).

    This is a METADATA correction only: it never changes raster computation and
    never adds QA_RADSAT. The canonical ``core.source_scene_provenance_config``
    text falsely claims "QA_PIXEL ... + QA_RADSAT"; this audit rewrites only the
    generated description so no downstream provenance/report field repeats the
    false claim.
    """
    import copy

    corrected = copy.deepcopy(config)
    composites = corrected.get("composites")
    if isinstance(composites, dict):
        for spec in composites.values():
            if isinstance(spec, dict) and "masking_rules" in spec:
                spec["masking_rules"] = QA_MASKING_RULE_TEXT
                spec["qa_radsat_applied"] = False
    return corrected


def provenance_evidence_qualification(mode: str) -> dict:
    """Qualify EXACTLY what the source path/row boundary evidence establishes.

    For ``metadata_only`` provenance the verified boundaries are real geometric
    shared-footprint edges: valid path/row boundary evidence, but NOT pixel-level
    selected-scene provenance. A median reducer has no semantically valid single
    selected-scene ID, so the absence of pixel-level provenance is a property of
    the reducer, not a defect -- it does NOT invalidate the geometric evidence.
    """
    metadata_derived = mode == "metadata_only"
    return {
        "provenance_mode": mode,
        "evidence_kind": (
            "metadata-derived source-footprint/path-row boundary evidence"
            if metadata_derived
            else "pixel-level selected-scene provenance"
        ),
        "is_valid_geometric_boundary_evidence": True,
        "is_pixel_level_selected_scene_provenance": not metadata_derived,
        "median_reducer_has_selected_scene_id": False,
        "qualification": (
            "Verified boundaries are metadata-derived source-footprint/path-row "
            "boundary evidence: valid GEOMETRIC boundary evidence, but NOT "
            "pixel-level selected-scene provenance. A median reducer has no "
            "semantically valid single selected-scene ID, so pixel-level "
            "provenance was not requested; this qualifies -- and does not "
            "invalidate -- the geometric boundary evidence."
        ),
    }


def _verdict_status(evidence: dict | None) -> str | None:
    return (evidence or {}).get("status")


def build_report_limitations(
    *,
    experiment_id: str,
    verified_source_evidence: dict | None = None,
    export_tile_negative_control: dict | None = None,
    reproduction_status: str = "pass",
    verified_source_present: bool = True,
    provenance_mode: str = "metadata_only",
) -> list[dict]:
    """Concise, machine-readable limitations for the counterfactual report.

    Each entry is ``{"code": <stable slug>, "statement": <text>, ...}``. The
    fixed methodological caveats are always present; data-dependent entries also
    attach the actual per-product verdict statuses so the entry is grounded in
    this run's results (never an unverifiable claim).
    """
    verified_source_evidence = verified_source_evidence or {}
    export_tile_negative_control = export_tile_negative_control or {}

    pathrow_status = {
        product: _verdict_status(verified_source_evidence.get(product))
        for product in ("current_lst", "current_minus_baseline", "anomaly_zscore")
    }
    tile_status = {
        product: _verdict_status(export_tile_negative_control.get(product))
        for product in export_tile_negative_control
    }

    limitations = [
        {
            "code": "single_aoi_manavgat",
            "statement": (
                f"This is a single-AOI ({experiment_id}) Manavgat composite "
                "counterfactual; conclusions are specific to this AOI and window."
            ),
        },
        {
            "code": "no_generalization",
            "statement": (
                "The result does NOT establish generalization to other AOIs, "
                "regions, sensors, or time windows."
            ),
        },
        {
            "code": "step8_model_impact_not_evaluated",
            "statement": (
                "Downstream Step8 / model impact of the compositing change has "
                "NOT yet been evaluated; only seam/boundary behaviour was audited."
            ),
        },
        {
            "code": "export_tile_control_unavailable",
            "statement": (
                "The export-tile negative-control comparison is unavailable "
                "because the relevant products were direct exports or did not "
                "provide a comparable paired tile partition."
            ),
            "export_tile_negative_control_status": tile_status,
        },
        {
            "code": "provenance_metadata_derived_not_pixel_level",
            "statement": (
                "Provenance is metadata-derived source-footprint/path-row "
                "boundary evidence (valid geometric boundary evidence), NOT "
                "pixel-level selected-scene provenance; a median reducer has no "
                "semantically valid single selected-scene ID."
            ),
            "provenance_mode": provenance_mode,
        },
        {
            "code": "pathrow_reduction_small_or_uncertain",
            "statement": (
                "Verified path/row-boundary reduction is very small for "
                "current_lst and current_minus_baseline and uncertain for "
                "anomaly_zscore."
            ),
            "verified_path_row_status": pathrow_status,
        },
        {
            "code": "supported_reduction_not_causal_proof",
            "statement": (
                "A supported_reduction does NOT prove same-date duplication is "
                "the sole seam cause and does NOT automatically mandate a "
                "production reducer change."
            ),
        },
    ]

    # Preserve the pre-existing conditional caveats when they actually apply.
    if not verified_source_present:
        limitations.append({
            "code": "no_verified_source_boundary_sampled",
            "statement": (
                "No verified source-scene/path-row boundary evidence was sampled; "
                "support-count-edge results are a proxy only."
            ),
        })
    if reproduction_status != "pass":
        limitations.append({
            "code": "canonical_reproduction_not_passed",
            "statement": (
                "Canonical reproduction did not pass; date_balanced results are "
                "not interpretable and no supported_reduction is emitted."
            ),
        })
    return limitations


def limitations_markdown(limitations: list[dict]) -> list[str]:
    """Compact Markdown bullet lines for the limitations list."""
    lines = ["## Limitations", ""]
    for item in limitations:
        code = item.get("code", "limitation")
        lines.append(f"- `{code}`: {item.get('statement', '')}")
    return lines


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
            ("canonical_reproduction", "canonical_reproduction.json"),
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
    provenance_units: dict | None = None,
    control_seed: int = 20240717,
    block_size: int = 128,
    min_units: int = 8,
    n_boot: int = 10000,
    ci: float = 0.95,
) -> dict:
    """Compute per (product, chain, boundary_type) metrics + paired verdicts.

    Support edges (scene-count / unique-date / same-day-multiplicity) come from
    ``edge_masks``; matched controls exclude the union support edge and use the
    SAME orientation and count-matched length.

    ``export_tile_boundary`` is a NEGATIVE control derived from an APPROXIMATE
    grid partition (see :func:`export_tile_boundary_edge_masks`); it is flagged
    ``can_affect_final_status=False`` and can never create positive evidence.

    Verified ``source_scene_path_row`` boundaries are only reported as evidence
    when ``provenance_status == 'provenance_available'`` AND actual rasterized
    boundary adjacency masks (``provenance_units``: ``{boundary_id: edge_mask}``)
    are supplied and sampled. Each ``boundary_id`` is preserved as its own
    bootstrap unit. If masks are not sampled the verdict is explicitly
    ``insufficient_boundary_metadata`` -- never fabricated, and never reported as
    ``provenance_available`` without sampling.
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

    # --- predeclared support-edge boundary types ---
    for boundary_type in ("scene_count_edge", "unique_date_count_edge", "same_day_multiplicity_edge"):
        sample = sample_paired_edges(sw_values, db_values, edge_masks[boundary_type])
        _emit(boundary_type, sample)
        verdicts[boundary_type] = paired_reduction_by_blocks(
            sample, unit_type="raster_support_block", block_size=block_size,
            min_units=min_units, n_boot=n_boot, ci=ci, seed=control_seed,
        )

    # --- export-tile NEGATIVE control (approximate partition; never evidence) ---
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
        tile_verdict = classify_paired_units(
            unit_reductions, unit_type="export_tile_partition_segment",
            min_units=min_units, n_boot=n_boot, ci=ci, seed=control_seed,
        )
    else:
        tile_verdict = {
            "status": "unavailable",
            "unit_type": "export_tile_partition_segment",
            "reason": "product exported directly or paired grids differ",
        }
    tile_verdict["is_negative_control"] = True
    tile_verdict["control_kind"] = "approximate_grid_partition"
    tile_verdict["can_affect_final_status"] = False
    verdicts["export_tile_boundary"] = tile_verdict

    # --- verified source-scene / path-row boundary ---
    sampled = (
        provenance_status == "provenance_available"
        and isinstance(provenance_units, dict)
        and len(provenance_units) > 0
    )
    if sampled:
        union_prov = {o: _union_tile_mask(provenance_units, o) for o in ORIENTATIONS}
        pooled = sample_paired_edges(sw_values, db_values, union_prov)
        _emit("source_scene_path_row", pooled)
        unit_reductions = {}
        for boundary_id, mask in provenance_units.items():
            seg = sample_paired_edges(sw_values, db_values, mask)
            unit_reductions[boundary_id] = seg["reduction"]
        verdict = classify_paired_units(
            unit_reductions, unit_type="provenance_boundary_id",
            min_units=min_units, n_boot=n_boot, ci=ci, seed=control_seed,
        )
        # Evidence only when at least one boundary_id actually produced pairs.
        verdict["is_verified_source_boundary_evidence"] = verdict["n_units"] > 0
        verdicts["source_scene_path_row"] = verdict
    else:
        verdicts["source_scene_path_row"] = {
            "status": "insufficient_boundary_metadata",
            "unit_type": "provenance_boundary_id",
            "is_verified_source_boundary_evidence": False,
            "provenance_status": provenance_status,
            "reason": (
                "verified source-scene/path-row boundaries require "
                "provenance_available AND rasterized boundary adjacency masks "
                "actually sampled; not reported as provenance_available without "
                "sampling"
            ),
        }

    return {"product": product, "metric_rows": metric_rows, "verdicts": verdicts}


def rasterize_provenance_boundaries(geojson: dict, transform, width: int, height: int) -> dict:
    """Rasterize verified ``scene_boundaries.geojson`` onto the EXACT raster grid.

    Each verified boundary feature (a shared source-footprint edge LineString in
    EPSG:4326) is burned onto the (``transform``, ``height`` x ``width``) grid
    and converted to adjacency-edge masks, preserving its ``boundary_id`` as the
    unit key: ``{boundary_id: {"horizontal": mask, "vertical": mask}}``. Only
    ``verification_status == 'verified'`` features are used. No resampling is
    performed; geometry is burned directly onto the product grid.
    """
    import numpy as np
    from rasterio.features import rasterize

    units: dict[str, dict] = {}
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        if props.get("verification_status") not in (None, "verified"):
            continue
        boundary_id = props.get("boundary_id") or props.get("source_boundary_id")
        geometry = feature.get("geometry")
        if not boundary_id or geometry is None:
            continue
        burned = rasterize(
            [(geometry, 1)], out_shape=(height, width), transform=transform,
            fill=0, all_touched=True, dtype="uint8",
        ).astype(bool)
        if not burned.any():
            continue
        # An adjacency edge is 'on the boundary' if either incident pixel burned.
        h = burned[:, :-1] | burned[:, 1:]
        v = burned[:-1, :] | burned[1:, :]
        units[str(boundary_id)] = {"horizontal": h, "vertical": v}
    return units


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


def read_raw_array(path, band: int = 1):
    """Read a band as float64 WITHOUT inventing or applying any nodata mask.

    Returns ``(raw_float64, nodata_tag_or_None)``. Unlike
    :func:`read_masked_array`, this NEVER substitutes NaN for a sentinel: the raw
    serialized value survives (so a canonical raster that stores GEE-masked
    pixels as raw DN 0 is seen as 0, and a diagnostic raster that stores them as
    -9999 is seen as -9999). The semantic gate applies the production Step5
    conversion to these raw values itself, rather than depending on a nodata
    encoding that differs between canonical and diagnostic products.
    """
    import rasterio

    with rasterio.open(path) as src:
        raw = src.read(band).astype("float64")
        nodata = src.nodata
    return raw, (None if nodata is None else float(nodata))


def validate_nodata_mask(path, *, expected_nodata=NODATA_SENTINEL, band: int = 1) -> dict:
    """Post-export check that a raster carries a verifiable nodata mask.

    Requirements for ``status == 'ok'`` (all must hold):
      * the GeoTIFF ``nodata`` tag is present AND equals ``expected_nodata``;
      * the RAW (unmasked) band is inspected separately -- every pixel whose raw
        value equals the sentinel is actually masked (so the sentinel is never
        read back as physical data);
      * no masked pixel reads as physical numeric zero.

    Otherwise the status is ``missing_nodata_tag`` / ``wrong_nodata_tag`` /
    ``sentinel_not_masked`` / ``masked_read_as_zero``.
    """
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        nodata = src.nodata
        raw = np.asarray(src.read(band), dtype="float64")      # RAW, unmasked
        mask = src.read_masks(band) == 0                       # True where masked
    masked_count = int(mask.sum())
    sentinel_pixels = raw == float(expected_nodata)
    sentinel_count = int(sentinel_pixels.sum())
    # Every raw-sentinel pixel must be masked (tag actually took effect).
    sentinel_not_masked = bool(np.any(sentinel_pixels & ~mask))
    # No masked pixel may read as physical 0 in the raw band.
    masked_as_zero = bool(masked_count and np.any((raw == 0.0) & mask) and expected_nodata != 0.0)

    if nodata is None:
        status = "missing_nodata_tag"
    elif float(nodata) != float(expected_nodata):
        status = "wrong_nodata_tag"
    elif sentinel_not_masked:
        status = "sentinel_not_masked"
    elif masked_as_zero:
        status = "masked_read_as_zero"
    else:
        status = "ok"
    return {
        "path": str(path),
        "nodata_tag": None if nodata is None else float(nodata),
        "expected_nodata": float(expected_nodata),
        "masked_pixel_count": masked_count,
        "sentinel_pixel_count": sentinel_count,
        "sentinel_not_masked": sentinel_not_masked,
        "masked_read_as_zero": masked_as_zero,
        "status": status,
    }


# =============================================================================
# EXPORT-TILE NEGATIVE CONTROL (APPROXIMATE grid partition)
# =============================================================================
def export_tile_boundary_edge_masks(width: int, height: int, tile_grid) -> dict:
    """APPROXIMATE tile-partition adjacency-edge masks (negative control only).

    ``tile_grid`` is ``[rows, cols]`` as recorded by
    ``export_image_direct_or_tiled``. The exact per-tile pixel offsets are NOT
    recoverable from the merged raster, so seam positions are APPROXIMATED as an
    even ``round(width * col / cols)`` / ``round(height * row / rows)`` partition
    -- this is an approximate partition control, NOT the exact retained tile
    transforms. It is used strictly as a negative control and is flagged so it
    can never affect ``final_status`` (see
    :func:`audit_product_boundaries`). Returns ``{unit_id: {"horizontal": mask,
    "vertical": mask}}`` (one unit per approximate seam) or ``{}`` when there is
    no interior seam.
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

    ``pass`` requires ALL of: exact grid equality (CRS, width, height,
    transform, AND bounds -- compared explicitly), EXACT valid-mask equality
    (every pixel valid in one is valid in the other), and max absolute
    difference over the common valid pixels <= ``tolerance``. Reports valid-mask
    agreement, compared pixel count, max/mean absolute difference, tolerance,
    and a reproduction status of ``pass`` / ``fail`` / ``mask_mismatch`` /
    ``grid_mismatch`` / ``not_available``.
    """
    import numpy as np

    if not Path(canonical_path).exists():
        return {"reproduction_status": "not_available", "canonical_path": str(canonical_path)}
    if not Path(diagnostic_path).exists():
        return {"reproduction_status": "not_available", "diagnostic_path": str(diagnostic_path)}
    diag_sig = grid_signature(diagnostic_path)
    canon_sig = grid_signature(canonical_path)
    mismatch_fields = []
    if diag_sig["crs"] != canon_sig["crs"]:
        mismatch_fields.append("crs")
    if diag_sig["width"] != canon_sig["width"]:
        mismatch_fields.append("width")
    if diag_sig["height"] != canon_sig["height"]:
        mismatch_fields.append("height")
    if any(abs(a - b) > 1e-9 for a, b in zip(diag_sig["transform"], canon_sig["transform"])):
        mismatch_fields.append("transform")
    # Bounds are compared EXPLICITLY (not only via transform+dims).
    if any(abs(a - b) > 1e-9 for a, b in zip(diag_sig["bounds"], canon_sig["bounds"])):
        mismatch_fields.append("bounds")
    if mismatch_fields:
        return {
            "reproduction_status": "grid_mismatch",
            "mismatch_fields": mismatch_fields,
            "diagnostic_grid": diag_sig,
            "canonical_grid": canon_sig,
            "tolerance": tolerance,
        }
    diag = read_masked_array(diagnostic_path, diagnostic_band)
    canon = read_masked_array(canonical_path, canonical_band)
    diag_valid = np.isfinite(diag)
    canon_valid = np.isfinite(canon)
    mask_agreement = float(np.mean(diag_valid == canon_valid))
    valid_mask_exact = bool(np.array_equal(diag_valid, canon_valid))
    both = diag_valid & canon_valid
    compared = int(both.sum())
    abs_diff = np.abs(diag[both] - canon[both]) if compared else np.array([0.0])
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    # Exact valid-mask equality is REQUIRED for pass.
    if not valid_mask_exact:
        status = "mask_mismatch"
    elif compared == 0:
        status = "fail"
    elif max_abs <= tolerance:
        status = "pass"
    else:
        status = "fail"
    return {
        "reproduction_status": status,
        "valid_mask_agreement": mask_agreement,
        "valid_mask_exact": valid_mask_exact,
        "compared_pixel_count": compared,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "tolerance": tolerance,
        "canonical_path": str(canonical_path),
        "diagnostic_path": str(diagnostic_path),
    }


# =============================================================================
# PIPELINE-SEMANTIC REPRODUCTION (decisive gate) + RAW-ENCODING DIAGNOSTIC
# =============================================================================
def _grid_mismatch_fields(diag_sig: dict, canon_sig: dict) -> list[str]:
    """Return the list of grid fields that differ (empty when grids are equal).

    Compares CRS, width, height, transform, AND bounds explicitly. Never
    resamples -- a non-empty result means the two rasters are NOT on the same
    grid and must fail reproduction.
    """
    fields = []
    if diag_sig["crs"] != canon_sig["crs"]:
        fields.append("crs")
    if diag_sig["width"] != canon_sig["width"]:
        fields.append("width")
    if diag_sig["height"] != canon_sig["height"]:
        fields.append("height")
    if any(abs(a - b) > 1e-9 for a, b in zip(diag_sig["transform"], canon_sig["transform"])):
        fields.append("transform")
    if any(abs(a - b) > 1e-9 for a, b in zip(diag_sig["bounds"], canon_sig["bounds"])):
        fields.append("bounds")
    return fields


def _raw_valid_mask(raw, nodata):
    """Raw valid mask honouring ONLY the raster's own declared nodata tag.

    A canonical annual baseline TIFF has no nodata tag, so its raw-valid mask is
    "every finite pixel" (including the DN 0 zero-fills). A diagnostic export
    tags -9999, so its raw-valid mask excludes the sentinel. The DIFFERENCE
    between these two masks is exactly the raw-encoding difference the diagnostic
    layer records; it never touches the decisive semantic comparison.
    """
    import numpy as np

    valid = np.isfinite(raw)
    if nodata is not None:
        valid = valid & (raw != nodata)
    return valid


def raw_encoding_comparison(raw_diag, diag_nodata, raw_canon, canon_nodata) -> dict:
    """Diagnostic (NON-failing) raw byte-encoding comparison of two rasters.

    Records the raw nodata tags, raw valid-mask agreement, the count of
    canonical-only raw-valid pixels (present in the canonical raw mask but masked
    in the diagnostic export), and whether every such canonical-only pixel is a
    zero-fill. This layer NEVER decides scientific reproduction: it exists so the
    zero-vs--9999 representation stays visible in the JSON and is never labelled
    "bitwise identical".
    """
    import numpy as np

    diag_valid = _raw_valid_mask(raw_diag, diag_nodata)
    canon_valid = _raw_valid_mask(raw_canon, canon_nodata)
    agreement = float(np.mean(diag_valid == canon_valid))
    exact = bool(np.array_equal(diag_valid, canon_valid))
    canonical_only = canon_valid & ~diag_valid
    count = int(canonical_only.sum())
    all_zero = bool(np.all(raw_canon[canonical_only] == 0.0)) if count else True
    return {
        "diagnostic_nodata_tag": None if diag_nodata is None else float(diag_nodata),
        "canonical_nodata_tag": None if canon_nodata is None else float(canon_nodata),
        "raw_valid_mask_agreement": agreement,
        "raw_valid_mask_exact": exact,
        "canonical_only_valid_pixel_count": count,
        "canonical_only_all_zero_filled": all_zero,
        "bitwise_identical": False,
        "note": (
            "Raw byte-level encoding differs: the canonical annual baseline TIFF "
            "serializes GEE-masked pixels as raw DN 0 with NO nodata tag, while "
            "the diagnostic export uses an explicit -9999 nodata sentinel. This "
            "is NOT a bitwise-identical file; equivalence is established by the "
            "pipeline-semantic comparison below, not by raw bytes."
        ),
    }


def _semantic_value_verdict(sem_diag, sem_canon, max_abs_diff: float) -> dict:
    """Compare two SEMANTICALLY-processed arrays (masked pixels are NaN).

    Requires EXACT valid-mask equality and max abs difference over the common
    valid pixels <= ``max_abs_diff``. Returns a ``reproduction_status`` of
    ``pass`` / ``fail`` / ``mask_mismatch``.
    """
    import numpy as np

    diag_valid = np.isfinite(sem_diag)
    canon_valid = np.isfinite(sem_canon)
    mask_agreement = float(np.mean(diag_valid == canon_valid))
    valid_mask_exact = bool(np.array_equal(diag_valid, canon_valid))
    both = diag_valid & canon_valid
    compared = int(both.sum())
    abs_diff = np.abs(sem_diag[both] - sem_canon[both]) if compared else np.array([0.0])
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    if not valid_mask_exact:
        status = "mask_mismatch"
    elif compared == 0:
        status = "fail"
    elif max_abs <= max_abs_diff:
        status = "pass"
    else:
        status = "fail"
    return {
        "reproduction_status": status,
        "valid_mask_agreement": mask_agreement,
        "valid_mask_exact": valid_mask_exact,
        "compared_pixel_count": compared,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "tolerance": max_abs_diff,
    }


def _semantic_transform(raw, kind: str):
    """Apply the production Step5 semantics for a given raster ``kind``.

    * ``baseline_dn`` -- annual baseline DN raster: dn_to_celsius then the
      physical-temperature mask (exact production helpers, no re-derived
      constants). DN 0 zero-fills AND -9999 sentinels both fall outside the
      physical range and become NaN, so the canonical/diagnostic encodings
      converge to the SAME valid mask.
    * ``celsius``     -- an already-Celsius raster (e.g. current-period median):
      only the production physical-temperature mask.
    * ``count``       -- an observation-count raster: a count of 0 observations
      carries no reduction information and is semantically equivalent to masked,
      so 0 (and any sentinel) maps to NaN.
    """
    import numpy as np

    from src.step5_preprocess_timeseries import dn_to_celsius, mask_physical_celsius

    if kind == "baseline_dn":
        return mask_physical_celsius(dn_to_celsius(raw))
    if kind == "celsius":
        return mask_physical_celsius(raw.astype("float32"))
    if kind == "count":
        finite = np.isfinite(raw) & (raw != NODATA_SENTINEL) & (raw > 0)
        return np.where(finite, raw, np.nan)
    raise ValueError(f"unknown semantic kind: {kind}")


def compare_semantic_reproduction(
    diagnostic_path,
    canonical_path,
    *,
    kind: str,
    max_abs_diff: float = BASELINE_SEMANTIC_MAX_ABS_DIFF,
    diagnostic_band: int = 1,
    canonical_band: int = 1,
) -> dict:
    """Decisive pipeline-semantic reproduction of one raster pair.

    Reads RAW values from both rasters (never inventing nodata), applies the
    EXACT production Step5 semantics for ``kind`` (:func:`_semantic_transform`),
    then requires exact valid-mask equality and max abs difference
    <= ``max_abs_diff``. Also records the diagnostic raw-encoding comparison and,
    when the raw encodings differ but the semantics agree, emits the
    ``raw_encoding_difference_semantically_equivalent`` warning. Grid equality
    (CRS/width/height/transform/bounds) is required and never relaxed.
    """
    if not Path(canonical_path).exists():
        return {"reproduction_status": "not_available", "canonical_path": str(canonical_path)}
    if not Path(diagnostic_path).exists():
        return {"reproduction_status": "not_available", "diagnostic_path": str(diagnostic_path)}

    diag_sig = grid_signature(diagnostic_path)
    canon_sig = grid_signature(canonical_path)
    mismatch_fields = _grid_mismatch_fields(diag_sig, canon_sig)
    if mismatch_fields:
        return {
            "reproduction_status": "grid_mismatch",
            "mismatch_fields": mismatch_fields,
            "diagnostic_grid": diag_sig,
            "canonical_grid": canon_sig,
            "semantic_kind": kind,
        }

    raw_diag, diag_nodata = read_raw_array(diagnostic_path, diagnostic_band)
    raw_canon, canon_nodata = read_raw_array(canonical_path, canonical_band)
    raw_encoding = raw_encoding_comparison(raw_diag, diag_nodata, raw_canon, canon_nodata)

    sem_diag = _semantic_transform(raw_diag, kind)
    sem_canon = _semantic_transform(raw_canon, kind)
    verdict = _semantic_value_verdict(sem_diag, sem_canon, max_abs_diff)

    warnings = []
    if not raw_encoding["raw_valid_mask_exact"]:
        if verdict["reproduction_status"] == "pass":
            warnings.append("raw_encoding_difference_semantically_equivalent")
        else:
            warnings.append("raw_encoding_difference_semantic_mismatch")

    return {
        **verdict,
        "semantic_kind": kind,
        "raw_encoding": raw_encoding,
        "warnings": warnings,
        "canonical_path": str(canonical_path),
        "diagnostic_path": str(diagnostic_path),
    }


def compare_derived_reduction(
    diagnostic_path,
    canonical_path,
    *,
    kind: str,
    tolerance: float = DERIVED_REDUCTION_TOLERANCE,
) -> dict:
    """Decisive semantic reproduction of a scene-weighted DERIVED Step5 product.

    Both inputs are the production Step5 float32 outputs (baseline mean / std),
    reproduced through the SAME production Step5 windowed reduction (no parallel
    re-implementation). Requires exact valid-mask equality and max abs difference
    <= ``tolerance`` (a float32 REDUCTION reproducibility tolerance, NOT a
    scientific effect threshold). Records the mean, p95, p99, p99.9, and maximum
    absolute difference plus the count of pixels above tolerance.
    """
    import numpy as np

    if not Path(canonical_path).exists():
        return {"reproduction_status": "not_available", "canonical_path": str(canonical_path)}
    if not Path(diagnostic_path).exists():
        return {"reproduction_status": "not_available", "diagnostic_path": str(diagnostic_path)}

    diag_sig = grid_signature(diagnostic_path)
    canon_sig = grid_signature(canonical_path)
    mismatch_fields = _grid_mismatch_fields(diag_sig, canon_sig)
    if mismatch_fields:
        return {
            "reproduction_status": "grid_mismatch",
            "mismatch_fields": mismatch_fields,
            "diagnostic_grid": diag_sig,
            "canonical_grid": canon_sig,
            "derived_kind": kind,
        }

    diag = read_masked_array(diagnostic_path, 1)
    canon = read_masked_array(canonical_path, 1)
    diag_valid = np.isfinite(diag)
    canon_valid = np.isfinite(canon)
    mask_agreement = float(np.mean(diag_valid == canon_valid))
    valid_mask_exact = bool(np.array_equal(diag_valid, canon_valid))
    both = diag_valid & canon_valid
    compared = int(both.sum())
    abs_diff = np.abs(diag[both] - canon[both]) if compared else np.array([0.0])
    max_abs = float(abs_diff.max())
    count_above = int(np.count_nonzero(abs_diff > tolerance))

    def _p(q):
        return float(np.percentile(abs_diff, q)) if compared else 0.0

    if not valid_mask_exact:
        status = "mask_mismatch"
    elif compared == 0:
        status = "fail"
    elif max_abs <= tolerance:
        status = "pass"
    else:
        status = "fail"
    return {
        "reproduction_status": status,
        "derived_kind": kind,
        "valid_mask_agreement": mask_agreement,
        "valid_mask_exact": valid_mask_exact,
        "compared_pixel_count": compared,
        "abs_diff_mean": float(abs_diff.mean()) if compared else 0.0,
        "abs_diff_p95": _p(95),
        "abs_diff_p99": _p(99),
        "abs_diff_p99_9": _p(99.9),
        "max_abs_diff": max_abs,
        "count_above_tolerance": count_above,
        "tolerance": tolerance,
        "tolerance_kind": "float32_reduction_reproducibility_not_scientific_effect",
        "canonical_path": str(canonical_path),
        "diagnostic_path": str(diagnostic_path),
    }


def compare_anomaly_threshold_boundary(
    *,
    diagnostic_zscore_path,
    canonical_zscore_path,
    diagnostic_std_path,
    canonical_std_path,
    diagnostic_mean_path,
    canonical_mean_path,
    diagnostic_current_path,
    canonical_current_path,
    std_threshold: float = None,
    std_tolerance: float = ANOMALY_STD_TOL,
    current_tolerance: float = ANOMALY_CURRENT_TOL,
    mean_tolerance: float = ANOMALY_MEAN_TOL,
    max_coord_samples: int = 64,
) -> dict:
    """Decisive semantic reproduction of the scene-weighted anomaly z-score.

    Anomaly z-score mask disagreements are NOT generally ignored. A disagreement
    is accepted as ``threshold_boundary_equivalent`` ONLY when EVERY mismatching
    pixel satisfies ALL of:

      * upstream current-LST masks agree (canonical vs diagnostic);
      * upstream baseline-mean masks agree (canonical vs diagnostic);
      * both baseline-std values are finite;
      * both baseline-std values are within ``std_tolerance`` (the accepted
        upstream std reproduction tolerance) of ``STEP5_MIN_BASELINE_STD_CELSIUS``;
      * NO mismatch occurs away from that threshold neighbourhood.

    Any mask mismatch outside the threshold neighbourhood FAILS reproduction.

    The z-score VALUE difference over common-valid pixels is bounded per pixel by
    a documented propagation of the accepted upstream tolerances rather than a
    tolerance chosen from the observed maximum::

        z = (current - mean) / std
        |dz| <= (current_tol + mean_tol)/std + |z| * std_tol / std

    (``std >= STEP5_MIN_BASELINE_STD_CELSIUS`` wherever z is defined, so
    ``1/std <= 1``). Ordinary descriptive value differences (mean/p95/p99/p99.9/
    max) are recorded SEPARATELY from this pass/fail bound.
    """
    import numpy as np

    from src.step5_preprocess_timeseries import mask_physical_celsius

    if std_threshold is None:
        std_threshold = float(STEP5_MIN_BASELINE_STD_CELSIUS)

    paths = {
        "diagnostic_zscore": diagnostic_zscore_path,
        "canonical_zscore": canonical_zscore_path,
        "diagnostic_std": diagnostic_std_path,
        "canonical_std": canonical_std_path,
        "diagnostic_mean": diagnostic_mean_path,
        "canonical_mean": canonical_mean_path,
        "diagnostic_current": diagnostic_current_path,
        "canonical_current": canonical_current_path,
    }
    missing = [name for name, p in paths.items() if not Path(p).exists()]
    if missing:
        return {"reproduction_status": "not_available", "missing_inputs": missing}

    # Grid contract: every input must share the exact grid (never resample).
    try:
        assert_same_grid(list(paths.values()))
    except GridMismatchError as exc:
        return {"reproduction_status": "grid_mismatch", "detail": str(exc)}

    z_diag = read_masked_array(diagnostic_zscore_path, 1)
    z_canon = read_masked_array(canonical_zscore_path, 1)
    zd_valid = np.isfinite(z_diag)
    zc_valid = np.isfinite(z_canon)
    valid_mask_exact = bool(np.array_equal(zd_valid, zc_valid))
    mismatch = zd_valid != zc_valid
    n_mismatch = int(mismatch.sum())

    std_d = read_masked_array(diagnostic_std_path, 1)
    std_c = read_masked_array(canonical_std_path, 1)

    # --- threshold-boundary classification of any mask disagreement ---
    if n_mismatch == 0:
        classification = "exact_mask_equality"
        threshold_boundary = {
            "classification": classification,
            "mismatch_pixel_count": 0,
        }
    else:
        cur_d_valid = np.isfinite(mask_physical_celsius(
            read_raw_array(diagnostic_current_path, 1)[0].astype("float32")))
        cur_c_valid = np.isfinite(mask_physical_celsius(
            read_raw_array(canonical_current_path, 1)[0].astype("float32")))
        mean_d_valid = np.isfinite(read_masked_array(diagnostic_mean_path, 1))
        mean_c_valid = np.isfinite(read_masked_array(canonical_mean_path, 1))

        current_masks_agree = cur_d_valid == cur_c_valid
        mean_masks_agree = mean_d_valid == mean_c_valid
        std_finite = np.isfinite(std_c) & np.isfinite(std_d)
        std_near_c = np.abs(std_c - std_threshold) <= std_tolerance
        std_near_d = np.abs(std_d - std_threshold) <= std_tolerance
        per_pixel_ok = (
            current_masks_agree & mean_masks_agree
            & std_finite & std_near_c & std_near_d
        )
        all_in_neighbourhood = bool(np.all(per_pixel_ok[mismatch]))
        classification = (
            "threshold_boundary_equivalent" if all_in_neighbourhood
            else "mask_mismatch_outside_threshold_neighbourhood"
        )

        rows, cols = np.nonzero(mismatch)
        sample_n = min(int(max_coord_samples), rows.size)
        coord_sample = [[int(rows[i]), int(cols[i])] for i in range(sample_n)]
        std_c_vals = std_c[mismatch]
        std_d_vals = std_d[mismatch]
        finite_c = std_c_vals[np.isfinite(std_c_vals)]
        finite_d = std_d_vals[np.isfinite(std_d_vals)]
        dist = np.abs(
            np.concatenate([finite_c, finite_d]) - std_threshold
        ) if (finite_c.size or finite_d.size) else np.array([])
        threshold_boundary = {
            "classification": classification,
            "mismatch_pixel_count": n_mismatch,
            "coordinate_sample": coord_sample,
            "coordinate_sample_truncated": bool(sample_n < rows.size),
            "std_threshold": std_threshold,
            "std_tolerance": std_tolerance,
            "canonical_std_range": (
                [float(finite_c.min()), float(finite_c.max())] if finite_c.size else None
            ),
            "diagnostic_std_range": (
                [float(finite_d.min()), float(finite_d.max())] if finite_d.size else None
            ),
            "max_distance_from_threshold": float(dist.max()) if dist.size else None,
        }

    # --- per-pixel propagated value bound over common-valid pixels ---
    both = zd_valid & zc_valid
    compared = int(both.sum())
    if compared:
        zc_both = z_canon[both]
        s = std_c[both]
        # std >= threshold wherever z is defined; guard non-finite defensively.
        s = np.where(np.isfinite(s) & (s >= std_threshold), s, std_threshold)
        abs_diff = np.abs(z_diag[both] - zc_both)
        bound = (current_tolerance + mean_tolerance) / s + np.abs(zc_both) * std_tolerance / s
        # small float32 rounding allowance on the bound itself
        exceed = abs_diff > (bound + 1e-7)
        value_within_bound = bool(not np.any(exceed))
        n_value_exceed = int(np.count_nonzero(exceed))

        def _p(q):
            return float(np.percentile(abs_diff, q))

        value_stats = {
            "compared_pixel_count": compared,
            "abs_diff_mean": float(abs_diff.mean()),
            "abs_diff_p95": _p(95),
            "abs_diff_p99": _p(99),
            "abs_diff_p99_9": _p(99.9),
            "max_abs_diff": float(abs_diff.max()),
            "max_propagated_bound": float(bound.max()),
            "pixels_exceeding_propagated_bound": n_value_exceed,
        }
    else:
        value_within_bound = False
        value_stats = {"compared_pixel_count": 0}

    mask_ok = classification in ("exact_mask_equality", "threshold_boundary_equivalent")
    if not mask_ok:
        status = "mask_mismatch"
    elif compared == 0:
        status = "fail"
    elif not value_within_bound:
        status = "fail"
    else:
        status = "pass"

    return {
        "reproduction_status": status,
        "valid_mask_exact": valid_mask_exact,
        "threshold_boundary": threshold_boundary,
        "value_comparison": value_stats,
        "value_bound_definition": (
            "per-pixel |dz| <= (current_tol + mean_tol)/std + |z|*std_tol/std, "
            "std >= STEP5_MIN_BASELINE_STD_CELSIUS; a propagated NUMERICAL bound "
            "from the accepted upstream reproduction tolerances, not a tolerance "
            "chosen from the observed maximum"
        ),
        "canonical_path": str(canonical_zscore_path),
        "diagnostic_path": str(diagnostic_zscore_path),
    }


def run_canonical_reproduction_gate(ctx: dict, root: Path, raster_plan: dict) -> dict:
    """Two-layer canonical reproduction gate (see ``CANONICAL_GATE_VERSION``).

    Layer A -- ``raw_encoding``: a DIAGNOSTIC comparison of the raw GeoTIFF
    encodings (nodata tags, raw valid-mask agreement, zero-filled canonical-only
    pixel counts). It never fails scientific reproduction on its own and never
    labels the raw files bitwise identical.

    Layer B -- ``semantic_checks`` (the decisive ``pipeline_semantic_reproduction``
    gate): each diagnostic scene_weighted product is compared to its frozen
    canonical counterpart AFTER applying the exact production Step5 semantics.
    Annual baseline LST rasters are compared DN->Celsius->physical-mask
    (max abs diff <= ``BASELINE_SEMANTIC_MAX_ABS_DIFF``); derived baseline
    mean/std use the float32 reduction tolerance ``DERIVED_REDUCTION_TOLERANCE``;
    the anomaly z-score classifies threshold-boundary mask disagreements and
    bounds value differences by a propagated numerical tolerance.

    ``status`` is ``pass`` only when every decisive semantic check passes. Grid,
    CRS, shape, transform, physical masking, and mismatches outside the std
    threshold neighbourhood are never relaxed. For manavgat_2021 the frozen files
    are REQUIRED, so a missing canonical file fails the gate.
    """
    canonical = resolve_canonical_paths(ctx)
    root = Path(root)
    required = ctx["experiment_id"] == "manavgat_2021"
    derived_dir = root / "derived" / CHAIN_SCENE_WEIGHTED

    # --- Layer B: decisive pipeline-semantic checks -------------------------
    checks: "OrderedDict" = OrderedDict()
    # current LST median (Celsius semantic) + valid count (count semantic).
    checks["current_lst_median"] = compare_semantic_reproduction(
        root / raster_plan["current_lst_scene_weighted_median"],
        canonical["current_lst"]["path"],
        kind="celsius", max_abs_diff=REPRODUCTION_TOLERANCES["physical_float32"],
        canonical_band=1,
    )
    checks["current_lst_valid_count"] = compare_semantic_reproduction(
        root / raster_plan["current_lst_scene_valid_count"],
        canonical["current_lst"]["path"],
        kind="count", max_abs_diff=REPRODUCTION_TOLERANCES["dn_count"],
        canonical_band=2,
    )
    # baseline yearly scene_weighted rasters: DN -> Celsius -> physical mask.
    for key, entry in canonical.items():
        if not key.startswith("baseline_lst_"):
            continue
        diag_name = f"{key}_scene_weighted_median"
        if diag_name in raster_plan:
            checks[key] = compare_semantic_reproduction(
                root / raster_plan[diag_name], entry["path"],
                kind="baseline_dn", max_abs_diff=BASELINE_SEMANTIC_MAX_ABS_DIFF,
            )
    # derived Step5 baseline mean/std: float32 reduction reproducibility.
    checks["derived_baseline_mean"] = compare_derived_reduction(
        derived_dir / "baseline_lst_mean_celsius.tif",
        canonical["step5_baseline_mean"]["path"], kind="baseline_mean",
    )
    checks["derived_baseline_std"] = compare_derived_reduction(
        derived_dir / "baseline_lst_std_celsius.tif",
        canonical["step5_baseline_std"]["path"], kind="baseline_std",
    )
    # anomaly z-score: threshold-boundary classification + propagated value bound.
    checks["derived_anomaly_zscore"] = compare_anomaly_threshold_boundary(
        diagnostic_zscore_path=derived_dir / "anomaly_zscore.tif",
        canonical_zscore_path=canonical["step5_anomaly_zscore"]["path"],
        diagnostic_std_path=derived_dir / "baseline_lst_std_celsius.tif",
        canonical_std_path=canonical["step5_baseline_std"]["path"],
        diagnostic_mean_path=derived_dir / "baseline_lst_mean_celsius.tif",
        canonical_mean_path=canonical["step5_baseline_mean"]["path"],
        diagnostic_current_path=root / raster_plan["current_lst_scene_weighted_median"],
        canonical_current_path=canonical["current_lst"]["path"],
    )

    # --- Layer A: diagnostic raw-encoding summary (never fails on its own) ---
    raw_encoding_years = OrderedDict()
    for key, check in checks.items():
        enc = check.get("raw_encoding") if isinstance(check, dict) else None
        if enc is not None:
            raw_encoding_years[key] = enc
    any_raw_diff = any(
        not enc.get("raw_valid_mask_exact", True) for enc in raw_encoding_years.values()
    )
    raw_encoding = {
        "status": (
            "difference_semantically_equivalent" if any_raw_diff else "raw_masks_agree"
        ),
        "bitwise_identical": False,
        "per_product": raw_encoding_years,
        "note": (
            "Raw encoding is a DIAGNOSTIC layer: canonical annual baseline TIFFs "
            "serialize masked pixels as raw DN 0 with no nodata tag, diagnostic "
            "exports use an explicit -9999 nodata sentinel. This difference never "
            "fails the decisive pipeline-semantic gate."
        ),
    }

    # --- decisive gate decision (semantic checks only) ----------------------
    statuses = [c.get("reproduction_status") for c in checks.values()]
    failing = {"fail", "grid_mismatch", "mask_mismatch"}
    if any(s in failing for s in statuses):
        gate = "canonical_reproduction_failed"
    elif "not_available" in statuses:
        gate = "canonical_reproduction_failed" if required else "not_available"
    else:
        gate = "pass"

    warnings = []
    if any_raw_diff:
        warnings.append("raw_encoding_difference_semantically_equivalent")
    for check in checks.values():
        for w in (check.get("warnings") or []):
            if w not in warnings:
                warnings.append(w)

    return {
        "status": gate,
        "canonical_gate_version": CANONICAL_GATE_VERSION,
        "decisive_gate": DECISIVE_SEMANTIC_GATE,
        "required_frozen_files": required,
        "raw_encoding": raw_encoding,
        "semantic_checks": checks,
        "threshold_boundary": (
            checks["derived_anomaly_zscore"].get("threshold_boundary")
            if isinstance(checks.get("derived_anomaly_zscore"), dict) else None
        ),
        "reduction_tolerance_note": (
            f"{DERIVED_REDUCTION_TOLERANCE} Celsius is a float32 numerical "
            "reproducibility tolerance for the derived baseline mean/std, NOT a "
            "scientific effect-size threshold."
        ),
        "warnings": warnings,
        "canonical_paths": canonical,
        # Back-compat: downstream summary/gate logic consumes ``checks`` and
        # ``status`` unchanged; ``checks`` mirrors the decisive semantic checks.
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
# RESUME / CHECKPOINT SUPPORT (non-destructive; validate-before-skip)
# =============================================================================
QUARANTINE_DIRNAME = "_interrupted_or_invalid"


def validate_existing_diagnostic_raster(
    path,
    *,
    expected_crs: str,
    expected_nodata: float = NODATA_SENTINEL,
    reference_signature: dict | None = None,
    band: int = 1,
) -> dict:
    """Validate an already-present diagnostic raster before it may be reused.

    A raster is reusable ONLY when every check passes:
      * it opens as a readable GeoTIFF;
      * width, height and band count are all non-zero;
      * CRS equals ``expected_crs``;
      * the GeoTIFF nodata tag equals ``expected_nodata`` (NODATA_SENTINEL);
      * at least one valid / non-nodata pixel exists;
      * when ``reference_signature`` is provided, its grid signature (CRS,
        width, height, transform, bounds) matches EXACTLY (no resampling).

    Returns ``{"valid": bool, "reason": str, "signature": dict|None,
    "nodata_status": str}``. Never raises on a corrupt file -- an unreadable
    raster is reported as ``valid=False`` so the caller can quarantine it.
    """
    import numpy as np
    import rasterio

    path = Path(path)
    if not path.exists():
        return {"valid": False, "reason": "missing", "signature": None, "nodata_status": None}
    try:
        with rasterio.open(path) as src:
            width, height, count = int(src.width), int(src.height), int(src.count)
            crs = str(src.crs)
            nodata = src.nodata
            if width <= 0 or height <= 0 or count <= 0:
                return {"valid": False, "reason": "empty_dimensions_or_bands",
                        "signature": None, "nodata_status": None}
    except Exception as exc:  # noqa: BLE001  (corrupt/unreadable file)
        return {"valid": False, "reason": f"unreadable_geotiff: {type(exc).__name__}",
                "signature": None, "nodata_status": None}

    signature = grid_signature(path)
    if crs.upper() != str(expected_crs).upper():
        return {"valid": False, "reason": f"crs_mismatch: {crs} != {expected_crs}",
                "signature": signature, "nodata_status": None}
    nodata_check = validate_nodata_mask(path, expected_nodata=expected_nodata, band=band)
    if nodata_check["status"] != "ok":
        return {"valid": False, "reason": f"nodata_{nodata_check['status']}",
                "signature": signature, "nodata_status": nodata_check["status"]}
    values = read_masked_array(path, band)
    if int(np.isfinite(values).sum()) == 0:
        return {"valid": False, "reason": "no_valid_pixels",
                "signature": signature, "nodata_status": nodata_check["status"]}
    if reference_signature is not None and not _signatures_match(reference_signature, signature):
        return {"valid": False, "reason": "grid_mismatch_vs_reference",
                "signature": signature, "nodata_status": nodata_check["status"]}
    return {"valid": True, "reason": "ok", "signature": signature,
            "nodata_status": nodata_check["status"]}


def quarantine_invalid_raster(
    experiment_id: str,
    path,
    *,
    base_dir: Path = PROJECT_ROOT,
    timestamp: str | None = None,
) -> str:
    """Move an invalid diagnostic raster into ``<root>/_interrupted_or_invalid/``.

    The file is MOVED (never deleted) under a timestamped name inside the exact
    diagnostic namespace. Namespace-safety is asserted for both the source and
    the destination, so nothing outside the diagnostic root is ever touched.
    """
    import shutil

    path = Path(path)
    root = diagnostic_output_root(experiment_id, base_dir)
    quarantine_dir = root / QUARANTINE_DIRNAME
    # Both the source raster and the quarantine target must stay in-namespace.
    assert_diagnostic_namespace_safe([path, quarantine_dir], experiment_id, base_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = quarantine_dir / f"{path.stem}.{stamp}{path.suffix}"
    shutil.move(str(path), str(dest))
    return str(dest)


def write_json_atomic(path, payload) -> Path:
    """Atomically write JSON: write to a temp file then os.replace into place."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(str(tmp), str(path))
    return path


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


# =============================================================================
# BOUNDED-MEMORY LOCAL FINALIZATION (windowed boundary audit, no full-RAM edge
# materialization, unit-summary bootstrap, disk-backed exact quantiles)
#
# WHY THIS EXISTS
# ---------------
# The original in-memory boundary audit loaded whole 2970x2338 rasters as
# float64, held several full adjacency masks at once, materialized per-edge
# orientation/row/col/signed/abs/reduction arrays for millions of edges, and --
# worst of all -- called np.concatenate inside every one of 10,000 cluster
# bootstrap iterations AND (when provenance was available) rasterized every one
# of ~177k boundary_ids into its own full-size boolean mask. That peak-RAM
# amplification is what repeatedly crashed WSL after export completed.
#
# The functions below reproduce the SAME scientific quantities with bounded
# memory:
#   * windows are streamed with a one-pixel right/bottom halo so every adjacency
#     edge (including edges that fall exactly on a window boundary) is owned by
#     exactly one window;
#   * the paired reduction definition is unchanged
#     (abs_jump_scene_weighted - abs_jump_date_balanced);
#   * the cluster bootstrap resamples UNITS using only per-unit
#     (finite_pair_count, reduction_sum) summaries -- never per-pixel arrays and
#     never np.concatenate;
#   * descriptive quantiles are computed EXACTLY from disk-backed temporary
#     float streams processed one product/boundary/chain at a time;
#   * provenance boundary_ids are represented once as a disk-backed int32 code
#     raster (bounded), not as ~177k separate full masks, and each boundary_id
#     remains its own bootstrap unit.
# None of the canonical Step3/Step5/... code or the pure reference helpers above
# are modified; the windowed path is an additive, equivalent finalization route.
# =============================================================================

ANALYSIS_TMP_DIRNAME = "_analysis_tmp"
ANALYSIS_CHECKPOINT_NAME = "analysis_checkpoint.json"
DEFAULT_WINDOW_SIZE = 512

# Predeclared negative-control sampling cap (recorded in outputs whenever it is
# hit). Matched controls are a descriptive negative control and can NEVER create
# positive final-status evidence, so bounding their sample size is scientifically
# safe; the cap is reported so a reader can see whether it was reached.
CONTROL_MAX_SAMPLES_PER_ORIENTATION = 2_000_000


# -----------------------------------------------------------------------------
# Resource logging (lightweight; psutil if present, else resource.getrusage)
# -----------------------------------------------------------------------------
def process_rss_mib():
    """Current process RSS in MiB (psutil), or peak RSS (resource), or None.

    Uses only libraries that may already be installed; never adds a dependency.
    """
    try:
        import psutil  # already installed in this environment

        return round(psutil.Process().memory_info().rss / 1048576.0, 1)
    except Exception:  # noqa: BLE001
        try:
            import resource

            # ru_maxrss is KiB on Linux; this is PEAK, not current, RSS.
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:  # noqa: BLE001
            return None


def dir_disk_usage_mib(path) -> float:
    """Total size (MiB) of all files under ``path`` (0.0 if it does not exist)."""
    total = 0
    root = Path(path)
    if not root.exists():
        return 0.0
    for p in root.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return round(total / 1048576.0, 1)


def stage_resource_record(stage: str, *, phase: str, tmp_dir=None, **extra) -> "OrderedDict":
    """One structured resource line for the run log (RSS/product/boundary/etc.).

    ``phase`` is ``begin`` or ``end``. Extra keys (product, boundary_type,
    n_paired_edges, n_units, ...) are recorded verbatim so a post-crash log makes
    the running stage unambiguous.
    """
    rec = OrderedDict()
    rec["ts"] = datetime.now(timezone.utc).isoformat()
    rec["stage"] = stage
    rec["phase"] = phase
    rec["rss_mib"] = process_rss_mib()
    if tmp_dir is not None:
        rec["tmp_disk_mib"] = dir_disk_usage_mib(tmp_dir)
    for key, value in extra.items():
        rec[key] = value
    return rec


# -----------------------------------------------------------------------------
# Windowed IO helpers
# -----------------------------------------------------------------------------
def iter_analysis_windows(height: int, width: int, window_size: int = DEFAULT_WINDOW_SIZE):
    """Yield ``(r0, r1, c0, c1)`` owned pixel tiles covering the whole grid.

    Windows partition the pixel grid exactly (no overlap). Each window is later
    read WITH a one-pixel right/bottom halo so boundary-crossing adjacency edges
    can be evaluated, while remaining owned by exactly one window.
    """
    for r0 in range(0, height, window_size):
        r1 = min(r0 + window_size, height)
        for c0 in range(0, width, window_size):
            c1 = min(c0 + window_size, width)
            yield r0, r1, c0, c1


def _read_block_nan(src, r0, r1, c0, c1, *, band: int = 1):
    """Read a float64 block for owned rows/cols [r0,r1)x[c0,c1) PLUS a 1px
    right/bottom halo, with nodata and NODATA_SENTINEL mapped to NaN.
    """
    import numpy as np
    import rasterio.windows as rwin

    height, width = src.height, src.width
    rr1 = min(r1 + 1, height)
    cc1 = min(c1 + 1, width)
    window = rwin.Window(c0, r0, cc1 - c0, rr1 - r0)
    arr = src.read(band, window=window, masked=True).astype("float64").filled(np.nan)
    arr = np.where(arr == NODATA_SENTINEL, np.nan, arr)
    return arr


def _read_code_block(mm, r0, r1, c0, c1, height, width):
    """Read an int32 code block (owned + 1px right/bottom halo) from a memmap."""
    rr1 = min(r1 + 1, height)
    cc1 = min(c1 + 1, width)
    return mm[r0:rr1, c0:cc1]


def _edge_ab(block, rr, cc, orientation):
    """Return the (a, b) endpoint arrays for OWNED adjacency edges of one
    orientation within a haloed block.

    ``rr``/``cc`` are the OWNED row/col counts (r1-r0 / c1-c0). Ownership rule:
    an edge is owned by the window that contains its top/left (anchor) pixel; the
    halo supplies the opposite endpoint. Anchor pixel of ``a[i, j]`` maps to
    global ``(r0 + i, c0 + j)``. Returns ``(None, None)`` when the orientation
    has no owned edge in this block.
    """
    bh, bw = block.shape
    if orientation == "horizontal":
        if rr < 1 or bw < 2:
            return None, None
        return block[0:rr, 0:bw - 1], block[0:rr, 1:bw]
    if cc < 1 or bh < 2:
        return None, None
    return block[0:bh - 1, 0:cc], block[1:bh, 0:cc]


# -----------------------------------------------------------------------------
# Disk-backed float stream + EXACT bounded-memory percentiles
# -----------------------------------------------------------------------------
class DiskFloatSink:
    """Append-only float64 stream backed by a temporary file under _analysis_tmp.

    Used to hold one boundary distribution (signed jumps for one chain) on disk
    so its EXACT median/p90/p95/p99 can be computed by loading a single array at
    a time -- never all products/boundaries at once.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")
        self.count = 0
        self._closed = False

    def append(self, arr) -> None:
        import numpy as np

        a = np.asarray(arr, dtype="float64")
        if a.size:
            self._fh.write(np.ascontiguousarray(a).tobytes())
            self.count += int(a.size)

    def close(self) -> None:
        if not self._closed:
            self._fh.flush()
            self._fh.close()
            self._closed = True

    def load(self):
        """Materialize the whole stream as ONE float64 array (bounded: one at a
        time). Returns an empty array when nothing was written."""
        import numpy as np

        self.close()
        if self.count == 0:
            return np.array([], dtype="float64")
        return np.fromfile(self.path, dtype="float64")

    def unlink(self) -> None:
        self.close()
        try:
            self.path.unlink()
        except OSError:
            pass


def percentile_block_from_signed(signed_values) -> dict:
    """valid pair count + signed/absolute jump median and p90/p95/p99.

    EXACT (numpy default linear interpolation), identical in definition to the
    in-memory :func:`_percentile_block` but taking a single signed array and
    deriving the absolute jumps as ``abs(signed)``.
    """
    import numpy as np

    signed = np.asarray(signed_values, dtype="float64")
    if signed.size == 0:
        return {
            "valid_pair_count": 0,
            "signed_jump_median": None,
            "absolute_jump_median": None,
            "absolute_jump_p90": None,
            "absolute_jump_p95": None,
            "absolute_jump_p99": None,
        }
    absolute = np.abs(signed)
    return {
        "valid_pair_count": int(signed.size),
        "signed_jump_median": float(np.median(signed)),
        "absolute_jump_median": float(np.median(absolute)),
        "absolute_jump_p90": float(np.percentile(absolute, 90)),
        "absolute_jump_p95": float(np.percentile(absolute, 95)),
        "absolute_jump_p99": float(np.percentile(absolute, 99)),
    }


# -----------------------------------------------------------------------------
# Unit-summary cluster bootstrap (mathematically equivalent to the pooled-mean
# per-pixel cluster bootstrap, but never concatenates per-pixel arrays)
# -----------------------------------------------------------------------------
def summaries_from_unit_reductions(unit_reductions) -> "OrderedDict":
    """Collapse ``{unit_id: reductions}`` to ``{unit_id: {count, reduction_sum}}``.

    Only FINITE reductions are counted, mirroring the in-memory cluster bootstrap
    which drops non-finite values and empty units. Provided so the equivalence
    test can derive summaries from the same dict the old bootstrap consumes.
    """
    import numpy as np

    out: "OrderedDict" = OrderedDict()
    for uid in sorted(unit_reductions):
        arr = np.asarray(unit_reductions[uid], dtype="float64")
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[uid] = {"count": int(arr.size), "reduction_sum": float(arr.sum())}
    return out


def cluster_bootstrap_interval_from_summaries(
    unit_summaries, *, n_boot=10000, ci=0.95, seed=20240717
):
    """Cluster (unit) percentile bootstrap of the pooled mean paired reduction,
    computed from per-unit ``(count, reduction_sum)`` summaries only.

    The pooled point estimate is ``sum(reduction_sum) / sum(count)``. Each
    replicate resamples units with replacement and forms
    ``sum(selected reduction_sum) / sum(selected count)``. This is identical to
    concatenating the selected units' per-pixel reductions and taking the mean,
    but WITHOUT np.concatenate and WITHOUT holding per-pixel arrays. Given the
    same sorted unit order and seed it reproduces the per-pixel cluster bootstrap
    exactly (same rng.choice draw sequence).
    """
    import numpy as np

    unit_ids = [u for u in sorted(unit_summaries) if int(unit_summaries[u]["count"]) > 0]
    counts = np.array(
        [float(unit_summaries[u]["count"]) for u in unit_ids], dtype="float64"
    )
    sums = np.array(
        [float(unit_summaries[u]["reduction_sum"]) for u in unit_ids], dtype="float64"
    )
    n_units = len(unit_ids)
    total_count = float(counts.sum()) if n_units else 0.0
    total_sum = float(sums.sum()) if n_units else 0.0
    result = {
        "n_units": n_units,
        "n_pairs": int(total_count),
        "n_boot": int(n_boot),
        "ci": float(ci),
        "seed": int(seed),
        "point_estimate": (total_sum / total_count) if total_count else None,
        "interval_low": None,
        "interval_high": None,
    }
    if n_units < 2:
        return result
    rng = np.random.default_rng(seed)
    index = np.arange(n_units)
    boot_means = np.empty(int(n_boot), dtype="float64")
    for b in range(int(n_boot)):
        chosen = rng.choice(index, size=n_units, replace=True)
        boot_means[b] = sums[chosen].sum() / counts[chosen].sum()
    lo_q = (1.0 - ci) / 2.0
    result["interval_low"] = float(np.quantile(boot_means, lo_q))
    result["interval_high"] = float(np.quantile(boot_means, 1.0 - lo_q))
    return result


def classify_paired_units_from_summaries(
    unit_summaries,
    *,
    unit_type: str,
    min_units: int = 8,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 20240717,
    max_unit_ids_listed: int = 10000,
) -> dict:
    """Paired-comparison verdict over predeclared bootstrap units, from summaries.

    Mirrors :func:`classify_paired_units` but consumes ``{unit_id: {count,
    reduction_sum}}``. ``unit_ids`` is truncated to ``max_unit_ids_listed`` (with
    a ``unit_ids_truncated`` flag) so a huge provenance verdict (~177k units)
    does not bloat the summary JSON; ``n_units`` is always exact.
    """
    interval = cluster_bootstrap_interval_from_summaries(
        unit_summaries, n_boot=n_boot, ci=ci, seed=seed
    )
    if interval["n_units"] < int(min_units):
        status = "insufficient_evidence"
    else:
        status = classify_paired_interval(
            interval["interval_low"], interval["interval_high"]
        )
    listed = [u for u in sorted(unit_summaries) if int(unit_summaries[u]["count"]) > 0]
    truncated = len(listed) > int(max_unit_ids_listed)
    return {
        "status": status,
        "unit_type": unit_type,
        "unit_ids": listed[: int(max_unit_ids_listed)],
        "unit_ids_truncated": truncated,
        "min_units_required": int(min_units),
        **interval,
        "reduction_definition": (
            "absolute_jump_scene_weighted - absolute_jump_date_balanced "
            "(positive => date_balanced reduces the jump)"
        ),
    }


# -----------------------------------------------------------------------------
# Windowed edge streaming (support-count boundaries) with per-block units
# -----------------------------------------------------------------------------
def _merge_block_summaries(unit_summ, grow, gcol, reduction, block_size) -> None:
    """Aggregate retained edges into 2-D spatial-block units, vectorized."""
    import numpy as np

    if grow.size == 0:
        return
    br = (grow // int(block_size)).astype(np.int64)
    bc = (gcol // int(block_size)).astype(np.int64)
    keys = (br << 20) | bc
    uniq, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv)
    ssum = np.bincount(inv, weights=np.asarray(reduction, dtype="float64"))
    for k, c, s in zip(uniq.tolist(), cnt.tolist(), ssum.tolist()):
        br_ = k >> 20
        bc_ = k & ((1 << 20) - 1)
        uid = f"block_r{int(br_)}_c{int(bc_)}"
        entry = unit_summ.get(uid)
        if entry is None:
            unit_summ[uid] = [int(c), float(s)]
        else:
            entry[0] += int(c)
            entry[1] += float(s)


def stream_support_boundary(
    sw_path,
    db_path,
    support_path,
    *,
    tmp_dir,
    key_prefix,
    block_size: int = 128,
    window_size: int = DEFAULT_WINDOW_SIZE,
    collect_edges: bool = False,
) -> dict:
    """Windowed pass for ONE support-count boundary type.

    A support edge is an adjacency edge where the integer support count differs
    across it (``finite(a) & finite(b) & (a != b)``), identical to
    :func:`_count_change_mask`. Only edges finite in BOTH chains at BOTH
    endpoints are retained; the rest are counted under ``skipped``. Per-chain
    signed jumps are streamed to disk (for exact quantiles) and per-2-D-block
    unit summaries are accumulated for the cluster bootstrap.

    Returns ``{unit_summaries, sw_sink, db_sink, orient_counts, skipped,
    pair_count}`` (and, when ``collect_edges``, the retained
    ``orientation/row/col`` arrays for equivalence testing on small rasters).
    """
    import numpy as np
    import rasterio

    tmp_dir = Path(tmp_dir)
    sw_sink = DiskFloatSink(tmp_dir / f"{key_prefix}.sw_signed.f64")
    db_sink = DiskFloatSink(tmp_dir / f"{key_prefix}.db_signed.f64")
    unit_summ: dict[str, list] = {}
    orient_counts = {o: 0 for o in ORIENTATIONS}
    skipped = 0
    edges = {"orientation": [], "row": [], "col": []} if collect_edges else None

    with rasterio.open(sw_path) as sw_src, rasterio.open(db_path) as db_src, \
            rasterio.open(support_path) as sup_src:
        height, width = sw_src.height, sw_src.width
        for (r0, r1, c0, c1) in iter_analysis_windows(height, width, window_size):
            rr, cc = r1 - r0, c1 - c0
            sw_b = _read_block_nan(sw_src, r0, r1, c0, c1)
            db_b = _read_block_nan(db_src, r0, r1, c0, c1)
            sup_b = _read_block_nan(sup_src, r0, r1, c0, c1)
            for orientation in ORIENTATIONS:
                sa, sb = _edge_ab(sw_b, rr, cc, orientation)
                if sa is None:
                    continue
                da, dbb = _edge_ab(db_b, rr, cc, orientation)
                ca, cb = _edge_ab(sup_b, rr, cc, orientation)
                selected = np.isfinite(ca) & np.isfinite(cb) & (ca != cb)
                valid = (
                    np.isfinite(sa) & np.isfinite(sb)
                    & np.isfinite(da) & np.isfinite(dbb)
                )
                retained = selected & valid
                skipped += int(np.sum(selected & ~valid))
                if not retained.any():
                    continue
                sw_signed = (sb - sa)[retained]
                db_signed = (dbb - da)[retained]
                reduction = np.abs(sw_signed) - np.abs(db_signed)
                li, lj = np.nonzero(retained)
                grow = (r0 + li).astype(np.int64)
                gcol = (c0 + lj).astype(np.int64)
                sw_sink.append(sw_signed)
                db_sink.append(db_signed)
                _merge_block_summaries(unit_summ, grow, gcol, reduction, block_size)
                orient_counts[orientation] += int(retained.sum())
                if edges is not None:
                    edges["orientation"].append(
                        np.full(grow.size, orientation, dtype=object)
                    )
                    edges["row"].append(grow)
                    edges["col"].append(gcol)

    sw_sink.close()
    db_sink.close()
    result = {
        "unit_summaries": {u: {"count": e[0], "reduction_sum": e[1]}
                           for u, e in unit_summ.items()},
        "sw_sink": sw_sink,
        "db_sink": db_sink,
        "orient_counts": orient_counts,
        "skipped": skipped,
        "pair_count": int(sw_sink.count),
    }
    if edges is not None:
        result["edges"] = {
            "orientation": (np.concatenate(edges["orientation"]) if edges["orientation"]
                            else np.array([], dtype=object)),
            "row": (np.concatenate(edges["row"]) if edges["row"]
                    else np.array([], dtype="int64")),
            "col": (np.concatenate(edges["col"]) if edges["col"]
                    else np.array([], dtype="int64")),
        }
    return result


# -----------------------------------------------------------------------------
# Windowed matched control (reservoir sampler; no eligible-index array)
# -----------------------------------------------------------------------------
def _reservoir_add(res_sw, res_db, cap, seen, batch_sw, batch_db, rng):
    """Vectorized reservoir (Algorithm R) update for one orientation batch.

    Returns the new ``seen`` count. ``res_sw``/``res_db`` are preallocated arrays
    of length ``cap``. Deterministic given ``rng``. Bounded: only ``cap`` samples
    are ever held, never the full eligible-index array.
    """
    import numpy as np

    m = int(batch_sw.size)
    if m == 0 or cap <= 0:
        return seen
    idxs = np.arange(seen, seen + m)
    fill_mask = idxs < cap
    if fill_mask.any():
        fpos = idxs[fill_mask]
        res_sw[fpos] = batch_sw[fill_mask]
        res_db[fpos] = batch_db[fill_mask]
    repl_mask = ~fill_mask
    if repl_mask.any():
        ridx = idxs[repl_mask].astype("float64")
        u = rng.random(ridx.size)
        accept = u < (cap / (ridx + 1.0))
        if accept.any():
            slots = rng.integers(0, cap, size=int(accept.sum()))
            res_sw[slots] = batch_sw[repl_mask][accept]
            res_db[slots] = batch_db[repl_mask][accept]
    return seen + m


def stream_matched_control(
    sw_path,
    db_path,
    scene_count_path,
    unique_date_path,
    want,
    *,
    seed,
    window_size: int = DEFAULT_WINDOW_SIZE,
    cap: int = CONTROL_MAX_SAMPLES_PER_ORIENTATION,
) -> dict:
    """Windowed reservoir sampler for orientation-matched interior controls.

    Eligible edges are finite in both chains at both endpoints AND NOT on the
    union support edge (scene-count change OR unique-date change) -- the same
    exclusion as the in-memory control. Up to ``want[orientation]`` samples are
    kept per orientation via a reservoir (capped at ``cap``); the full eligible
    index array is never built. Records eligible/requested/actual counts, seed,
    and whether the predeclared cap was reached.
    """
    import numpy as np
    import rasterio

    rng = np.random.default_rng(seed)
    reservoirs = {}
    capacities = {}
    seen = {o: 0 for o in ORIENTATIONS}
    eligible = {o: 0 for o in ORIENTATIONS}
    for orientation in ORIENTATIONS:
        capacity = min(int(want.get(orientation, 0)), int(cap))
        capacities[orientation] = capacity
        reservoirs[orientation] = (
            np.empty(capacity, dtype="float64"),
            np.empty(capacity, dtype="float64"),
        )

    with rasterio.open(sw_path) as sw_src, rasterio.open(db_path) as db_src, \
            rasterio.open(scene_count_path) as sc_src, \
            rasterio.open(unique_date_path) as uc_src:
        height, width = sw_src.height, sw_src.width
        for (r0, r1, c0, c1) in iter_analysis_windows(height, width, window_size):
            rr, cc = r1 - r0, c1 - c0
            sw_b = _read_block_nan(sw_src, r0, r1, c0, c1)
            db_b = _read_block_nan(db_src, r0, r1, c0, c1)
            sc_b = _read_block_nan(sc_src, r0, r1, c0, c1)
            uc_b = _read_block_nan(uc_src, r0, r1, c0, c1)
            for orientation in ORIENTATIONS:
                capacity = capacities[orientation]
                if capacity <= 0:
                    continue
                sa, sb = _edge_ab(sw_b, rr, cc, orientation)
                if sa is None:
                    continue
                da, dbb = _edge_ab(db_b, rr, cc, orientation)
                sca, scb = _edge_ab(sc_b, rr, cc, orientation)
                uca, ucb = _edge_ab(uc_b, rr, cc, orientation)
                scene_change = np.isfinite(sca) & np.isfinite(scb) & (sca != scb)
                date_change = np.isfinite(uca) & np.isfinite(ucb) & (uca != ucb)
                excluded = scene_change | date_change
                elig = (
                    np.isfinite(sa) & np.isfinite(sb)
                    & np.isfinite(da) & np.isfinite(dbb)
                    & ~excluded
                )
                if not elig.any():
                    continue
                eligible[orientation] += int(elig.sum())
                batch_sw = (sb - sa)[elig]
                batch_db = (dbb - da)[elig]
                res_sw, res_db = reservoirs[orientation]
                seen[orientation] = _reservoir_add(
                    res_sw, res_db, capacity, seen[orientation],
                    batch_sw, batch_db, rng,
                )

    sw_parts, db_parts = [], []
    actual = {}
    for orientation in ORIENTATIONS:
        capacity = capacities[orientation]
        filled = min(seen[orientation], capacity)
        actual[orientation] = int(filled)
        if filled:
            res_sw, res_db = reservoirs[orientation]
            sw_parts.append(res_sw[:filled])
            db_parts.append(res_db[:filled])
    sw_signed = np.concatenate(sw_parts) if sw_parts else np.array([], dtype="float64")
    db_signed = np.concatenate(db_parts) if db_parts else np.array([], dtype="float64")
    cap_used = any(eligible[o] > capacities[o] for o in ORIENTATIONS)
    return {
        "sw_signed": sw_signed,
        "db_signed": db_signed,
        "eligible_per_orientation": eligible,
        "requested_per_orientation": {o: int(want.get(o, 0)) for o in ORIENTATIONS},
        "actual_per_orientation": actual,
        "seed": int(seed),
        "cap_per_orientation": int(cap),
        "cap_used": bool(cap_used),
    }


# -----------------------------------------------------------------------------
# Windowed export-tile negative control (approximate grid partition)
# -----------------------------------------------------------------------------
def export_tile_seam_specs(width: int, height: int, tile_grid) -> list[dict]:
    """Approximate interior seam specs (no full masks) for the negative control.

    Mirrors the seam positions of :func:`export_tile_boundary_edge_masks` but as
    lightweight ``{unit_id, orientation, anchor}`` specs: a vertical tile seam is
    a HORIZONTAL adjacency edge at anchor column ``idx-1``; a horizontal tile
    seam is a VERTICAL adjacency edge at anchor row ``idx-1``.
    """
    if not (isinstance(tile_grid, (list, tuple)) and len(tile_grid) == 2):
        return []
    rows, cols = int(tile_grid[0]), int(tile_grid[1])
    specs = []
    for col in range(1, cols):
        idx = round(width * col / cols)
        if 0 < idx < width:
            specs.append({"unit_id": f"tile_v{col}", "orientation": "horizontal",
                          "anchor": idx - 1})
    for row in range(1, rows):
        idx = round(height * row / rows)
        if 0 < idx < height:
            specs.append({"unit_id": f"tile_h{row}", "orientation": "vertical",
                          "anchor": idx - 1})
    return specs


def stream_tile_control(
    sw_path, db_path, seam_specs, *, tmp_dir, key_prefix,
    window_size: int = DEFAULT_WINDOW_SIZE,
):
    """Windowed pass over approximate export-tile seams (negative control).

    Each seam is one bootstrap unit; a pooled disk sink over all seams feeds the
    descriptive metrics. Returns ``{unit_summaries, sw_sink, db_sink, skipped,
    orient_counts}``.
    """
    import numpy as np
    import rasterio

    tmp_dir = Path(tmp_dir)
    sw_sink = DiskFloatSink(tmp_dir / f"{key_prefix}.sw_signed.f64")
    db_sink = DiskFloatSink(tmp_dir / f"{key_prefix}.db_signed.f64")
    unit_summ = {s["unit_id"]: [0, 0.0] for s in seam_specs}
    orient_counts = {o: 0 for o in ORIENTATIONS}
    skipped = 0
    by_orient = {o: [s for s in seam_specs if s["orientation"] == o] for o in ORIENTATIONS}

    with rasterio.open(sw_path) as sw_src, rasterio.open(db_path) as db_src:
        height, width = sw_src.height, sw_src.width
        for (r0, r1, c0, c1) in iter_analysis_windows(height, width, window_size):
            rr, cc = r1 - r0, c1 - c0
            if not any(by_orient[o] for o in ORIENTATIONS):
                break
            sw_b = _read_block_nan(sw_src, r0, r1, c0, c1)
            db_b = _read_block_nan(db_src, r0, r1, c0, c1)
            for orientation in ORIENTATIONS:
                specs = by_orient[orientation]
                if not specs:
                    continue
                sa, sb = _edge_ab(sw_b, rr, cc, orientation)
                if sa is None:
                    continue
                da, dbb = _edge_ab(db_b, rr, cc, orientation)
                valid = (
                    np.isfinite(sa) & np.isfinite(sb)
                    & np.isfinite(da) & np.isfinite(dbb)
                )
                for spec in specs:
                    anchor = spec["anchor"]
                    if orientation == "horizontal":
                        if not (c0 <= anchor < c0 + sa.shape[1]):
                            continue
                        local = anchor - c0
                        col_valid = valid[:, local]
                        if not col_valid.any():
                            continue
                        sw_signed = (sb[:, local] - sa[:, local])[col_valid]
                        db_signed = (dbb[:, local] - da[:, local])[col_valid]
                    else:
                        if not (r0 <= anchor < r0 + sa.shape[0]):
                            continue
                        local = anchor - r0
                        row_valid = valid[local, :]
                        if not row_valid.any():
                            continue
                        sw_signed = (sb[local, :] - sa[local, :])[row_valid]
                        db_signed = (dbb[local, :] - da[local, :])[row_valid]
                    reduction = np.abs(sw_signed) - np.abs(db_signed)
                    entry = unit_summ[spec["unit_id"]]
                    entry[0] += int(sw_signed.size)
                    entry[1] += float(reduction.sum())
                    sw_sink.append(sw_signed)
                    db_sink.append(db_signed)
                    orient_counts[orientation] += int(sw_signed.size)

    sw_sink.close()
    db_sink.close()
    return {
        "unit_summaries": {u: {"count": e[0], "reduction_sum": e[1]}
                           for u, e in unit_summ.items() if e[0] > 0},
        "sw_sink": sw_sink,
        "db_sink": db_sink,
        "skipped": skipped,
        "orient_counts": orient_counts,
    }


# -----------------------------------------------------------------------------
# Windowed provenance boundaries: ONE disk-backed int32 code raster (bounded),
# not ~177k full masks; each boundary_id stays its own bootstrap unit.
# -----------------------------------------------------------------------------
def _feature_pixel_bbox(geometry, transform, width, height):
    """Inclusive pixel bbox ``(rmin, rmax, cmin, cmax)`` of a geometry, or None.

    Uses the affine ``transform`` directly (no shapely). Clamps to the grid.
    """
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f

    def _coords(geom):
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "LineString":
            return list(coords)
        if gtype in ("MultiLineString", "Polygon"):
            pts = []
            for part in coords:
                pts.extend(part)
            return pts
        if gtype == "MultiPolygon":
            pts = []
            for poly in coords:
                for ring in poly:
                    pts.extend(ring)
            return pts
        if gtype == "Point":
            return [coords]
        return []

    pts = _coords(geometry)
    if not pts:
        return None
    # Only rectilinear north-up transforms are used here (b == d == 0).
    cols = [((x - c) / a) for x, y in pts]
    rows = [((y - f) / e) for x, y in pts]
    cmin = max(0, int(min(cols)) - 1)
    cmax = min(width - 1, int(max(cols)) + 1)
    rmin = max(0, int(min(rows)) - 1)
    rmax = min(height - 1, int(max(rows)) + 1)
    if cmin > cmax or rmin > rmax:
        return None
    return rmin, rmax, cmin, cmax


def build_provenance_code_memmap(
    geojson, transform, width, height, tmp_dir,
    *, window_size: int = DEFAULT_WINDOW_SIZE,
):
    """Rasterize verified boundaries ONCE into a disk-backed int32 code raster.

    Only ``verification_status in (None, 'verified')`` features with a
    ``boundary_id`` are used. Code ``k`` (1-based) marks pixels of
    ``boundary_ids[k-1]``; 0 is background. Rasterization is windowed (bounded
    memory) and written straight to a memmap under ``_analysis_tmp`` -- there is
    never a per-boundary_id full mask, and never all boundaries in RAM at once.

    Returns ``{codes_path, boundary_ids, n_codes}`` or None when no verified
    boundary is present.
    """
    import numpy as np
    import rasterio.windows as rwin
    from rasterio.features import rasterize

    boundary_ids: list[str] = []
    geometries: list = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        if props.get("verification_status") not in (None, "verified"):
            continue
        boundary_id = props.get("boundary_id") or props.get("source_boundary_id")
        geometry = feature.get("geometry")
        if not boundary_id or geometry is None:
            continue
        boundary_ids.append(str(boundary_id))
        geometries.append(geometry)
    if not boundary_ids:
        return None

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    codes_path = tmp_dir / "provenance_codes.int32"
    mm = np.memmap(codes_path, dtype="int32", mode="w+", shape=(height, width))
    mm[:] = 0

    # Precompute per-feature pixel bbox for cheap window prefiltering.
    bboxes = [_feature_pixel_bbox(g, transform, width, height) for g in geometries]

    for (r0, r1, c0, c1) in iter_analysis_windows(height, width, window_size):
        feats = []
        for code0, bbox in enumerate(bboxes):
            if bbox is None:
                continue
            rmin, rmax, cmin, cmax = bbox
            if rmax < r0 or rmin >= r1 or cmax < c0 or cmin >= c1:
                continue
            feats.append((geometries[code0], code0 + 1))
        if not feats:
            continue
        win = rwin.Window(c0, r0, c1 - c0, r1 - r0)
        win_transform = rwin.transform(win, transform)
        sub = rasterize(
            feats, out_shape=(r1 - r0, c1 - c0), transform=win_transform,
            fill=0, all_touched=True, dtype="int32",
        )
        mm[r0:r1, c0:c1] = sub
    mm.flush()
    return {"codes_path": str(codes_path), "boundary_ids": boundary_ids,
            "n_codes": len(boundary_ids)}


def stream_provenance_boundaries(
    sw_path, db_path, code_spec, *, tmp_dir, key_prefix,
    window_size: int = DEFAULT_WINDOW_SIZE,
):
    """Windowed pass assigning retained edges to provenance boundary_id units.

    An adjacency edge is on boundary code ``k`` if EITHER incident pixel carries
    code ``k`` (the same "either incident pixel burned" rule as
    :func:`rasterize_provenance_boundaries`). With a single collapsed code raster
    an edge belongs to at most its two pixels' codes; each ``boundary_id`` stays
    its own bootstrap unit, and a boundary is evidence only when it produces at
    least one valid paired edge. A pooled disk sink over all boundary edges feeds
    the descriptive metrics.

    Returns ``{unit_summaries, sw_sink, db_sink, skipped, orient_counts,
    n_boundary_units}``.
    """
    import numpy as np
    import rasterio

    boundary_ids = code_spec["boundary_ids"]
    n_codes = int(code_spec["n_codes"])
    codes_path = code_spec["codes_path"]

    tmp_dir = Path(tmp_dir)
    sw_sink = DiskFloatSink(tmp_dir / f"{key_prefix}.sw_signed.f64")
    db_sink = DiskFloatSink(tmp_dir / f"{key_prefix}.db_signed.f64")
    counts_by_code = np.zeros(n_codes + 1, dtype="int64")
    sums_by_code = np.zeros(n_codes + 1, dtype="float64")
    orient_counts = {o: 0 for o in ORIENTATIONS}
    skipped = 0

    with rasterio.open(sw_path) as sw_src, rasterio.open(db_path) as db_src:
        height, width = sw_src.height, sw_src.width
        mm = np.memmap(codes_path, dtype="int32", mode="r", shape=(height, width))
        for (r0, r1, c0, c1) in iter_analysis_windows(height, width, window_size):
            rr, cc = r1 - r0, c1 - c0
            code_full = _read_code_block(mm, r0, r1, c0, c1, height, width)
            # Cheap skip: window (with halo) has no boundary pixel at all.
            if not code_full.any():
                continue
            sw_b = _read_block_nan(sw_src, r0, r1, c0, c1)
            db_b = _read_block_nan(db_src, r0, r1, c0, c1)
            for orientation in ORIENTATIONS:
                sa, sb = _edge_ab(sw_b, rr, cc, orientation)
                if sa is None:
                    continue
                da, dbb = _edge_ab(db_b, rr, cc, orientation)
                ca, cb = _edge_ab(code_full, rr, cc, orientation)
                on_boundary = (ca > 0) | (cb > 0)
                if not on_boundary.any():
                    continue
                valid = (
                    np.isfinite(sa) & np.isfinite(sb)
                    & np.isfinite(da) & np.isfinite(dbb)
                )
                retained = on_boundary & valid
                skipped += int(np.sum(on_boundary & ~valid))
                if not retained.any():
                    continue
                sw_signed = (sb - sa)[retained]
                db_signed = (dbb - da)[retained]
                reduction = np.abs(sw_signed) - np.abs(db_signed)
                sw_sink.append(sw_signed)
                db_sink.append(db_signed)
                orient_counts[orientation] += int(retained.sum())
                ra = ca[retained]
                rb = cb[retained]
                mask_a = ra > 0
                if mask_a.any():
                    counts_by_code += np.bincount(
                        ra[mask_a], minlength=n_codes + 1
                    )
                    sums_by_code += np.bincount(
                        ra[mask_a], weights=reduction[mask_a], minlength=n_codes + 1
                    )
                mask_b = (rb > 0) & (rb != ra)
                if mask_b.any():
                    counts_by_code += np.bincount(
                        rb[mask_b], minlength=n_codes + 1
                    )
                    sums_by_code += np.bincount(
                        rb[mask_b], weights=reduction[mask_b], minlength=n_codes + 1
                    )
        del mm

    sw_sink.close()
    db_sink.close()
    unit_summaries = {}
    nonzero = np.nonzero(counts_by_code[1:])[0]
    for idx in nonzero.tolist():
        unit_summaries[boundary_ids[idx]] = {
            "count": int(counts_by_code[idx + 1]),
            "reduction_sum": float(sums_by_code[idx + 1]),
        }
    return {
        "unit_summaries": unit_summaries,
        "sw_sink": sw_sink,
        "db_sink": db_sink,
        "skipped": skipped,
        "orient_counts": orient_counts,
        "n_boundary_units": len(unit_summaries),
    }


# -----------------------------------------------------------------------------
# Assemble metric rows (identical shape to boundary_chain_metrics output)
# -----------------------------------------------------------------------------
def _metric_rows_from_blocks(product, boundary_type, skipped,
                             boundary_sw, boundary_db, control_sw, control_db):
    """Build the two per-chain metric rows for one (product, boundary_type).

    ``boundary_*``/``control_*`` are percentile blocks from
    :func:`percentile_block_from_signed`. The row shape matches the in-memory
    :func:`boundary_chain_metrics` exactly.
    """
    rows = []
    for chain, boundary, control in (
        (CHAIN_SCENE_WEIGHTED, boundary_sw, control_sw),
        (CHAIN_DATE_BALANCED, boundary_db, control_db),
    ):
        rows.append({
            "product": product,
            "boundary_type": boundary_type,
            "skipped_missing_observation": skipped,
            "chain": chain,
            **boundary,
            "control_absolute_jump_median": control["absolute_jump_median"],
            "control_absolute_jump_p95": control["absolute_jump_p95"],
            "control_pair_count": control["valid_pair_count"],
            "boundary_control_median_ratio": _ratio(
                boundary["absolute_jump_median"], control["absolute_jump_median"]
            ),
            "boundary_control_p95_ratio": _ratio(
                boundary["absolute_jump_p95"], control["absolute_jump_p95"]
            ),
        })
    return rows


def audit_product_boundaries_windowed(
    product: str,
    sw_path,
    db_path,
    support_paths: dict,
    *,
    tmp_dir,
    tile_seam_specs: list | None = None,
    provenance_status: str = "insufficient_boundary_metadata",
    provenance_code_spec: dict | None = None,
    control_seed: int = 20240717,
    block_size: int = 128,
    min_units: int = 8,
    n_boot: int = 10000,
    ci: float = 0.95,
    window_size: int = DEFAULT_WINDOW_SIZE,
    resource_log: list | None = None,
) -> dict:
    """Bounded-memory equivalent of :func:`audit_product_boundaries`.

    ``support_paths`` maps ``scene_count_edge`` / ``unique_date_count_edge`` /
    ``same_day_multiplicity_edge`` to the recorded support-count raster paths.
    Every boundary type is streamed one at a time so peak RAM stays near a single
    boundary distribution. Preserves boundary definitions, the paired reduction
    definition, the export-tile negative-control flags, and the verified-source
    evidence gating. Returns ``{product, metric_rows, verdicts,
    control_provenance}``.
    """
    import numpy as np

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    verdicts: "OrderedDict" = OrderedDict()
    control_provenance: "OrderedDict" = OrderedDict()

    def _log(stage, phase, **extra):
        if resource_log is not None:
            resource_log.append(
                stage_resource_record(stage, phase=phase, tmp_dir=tmp_dir, **extra)
            )

    def _finish_boundary(boundary_type, sample, want):
        nonlocal metric_rows
        control = stream_matched_control(
            sw_path, db_path, support_paths["scene_count_edge"],
            support_paths["unique_date_count_edge"], want,
            seed=control_seed, window_size=window_size,
        )
        control_provenance[boundary_type] = {
            k: control[k] for k in (
                "eligible_per_orientation", "requested_per_orientation",
                "actual_per_orientation", "seed", "cap_per_orientation", "cap_used",
            )
        }
        boundary_sw = percentile_block_from_signed(sample["sw_sink"].load())
        boundary_db = percentile_block_from_signed(sample["db_sink"].load())
        control_sw = percentile_block_from_signed(control["sw_signed"])
        control_db = percentile_block_from_signed(control["db_signed"])
        metric_rows.extend(_metric_rows_from_blocks(
            product, boundary_type, sample["skipped"],
            boundary_sw, boundary_db, control_sw, control_db,
        ))
        sample["sw_sink"].unlink()
        sample["db_sink"].unlink()

    # --- predeclared support-edge boundary types ---
    for boundary_type in (
        "scene_count_edge", "unique_date_count_edge", "same_day_multiplicity_edge",
    ):
        _log("boundary", "begin", product=product, boundary_type=boundary_type)
        sample = stream_support_boundary(
            sw_path, db_path, support_paths[boundary_type],
            tmp_dir=tmp_dir, key_prefix=f"{product}.{boundary_type}",
            block_size=block_size, window_size=window_size,
        )
        verdicts[boundary_type] = classify_paired_units_from_summaries(
            sample["unit_summaries"], unit_type="raster_support_block",
            min_units=min_units, n_boot=n_boot, ci=ci, seed=control_seed,
        )
        _finish_boundary(boundary_type, sample, sample["orient_counts"])
        _log("boundary", "end", product=product, boundary_type=boundary_type,
             n_paired_edges=sample["pair_count"], n_units=verdicts[boundary_type]["n_units"])

    # --- export-tile NEGATIVE control (approximate partition; never evidence) ---
    if tile_seam_specs:
        _log("boundary", "begin", product=product, boundary_type="export_tile_boundary")
        tile = stream_tile_control(
            sw_path, db_path, tile_seam_specs, tmp_dir=tmp_dir,
            key_prefix=f"{product}.export_tile_boundary", window_size=window_size,
        )
        tile_verdict = classify_paired_units_from_summaries(
            tile["unit_summaries"], unit_type="export_tile_partition_segment",
            min_units=min_units, n_boot=n_boot, ci=ci, seed=control_seed,
        )
        # Pooled control uses interior offset edges (union support excluded).
        want_pool = tile["orient_counts"]
        _finish_boundary("export_tile_boundary", tile, want_pool)
        _log("boundary", "end", product=product, boundary_type="export_tile_boundary",
             n_units=tile_verdict["n_units"])
    else:
        tile_verdict = {
            "status": "unavailable",
            "unit_type": "export_tile_partition_segment",
            "reason": "product exported directly or paired grids differ",
        }
    tile_verdict["is_negative_control"] = True
    tile_verdict["control_kind"] = "approximate_grid_partition"
    tile_verdict["can_affect_final_status"] = False
    verdicts["export_tile_boundary"] = tile_verdict

    # --- verified source-scene / path-row boundary (windowed, per boundary_id) ---
    sampled = (
        provenance_status == "provenance_available"
        and isinstance(provenance_code_spec, dict)
        and int(provenance_code_spec.get("n_codes", 0)) > 0
    )
    if sampled:
        _log("boundary", "begin", product=product, boundary_type="source_scene_path_row")
        prov = stream_provenance_boundaries(
            sw_path, db_path, provenance_code_spec, tmp_dir=tmp_dir,
            key_prefix=f"{product}.source_scene_path_row", window_size=window_size,
        )
        verdict = classify_paired_units_from_summaries(
            prov["unit_summaries"], unit_type="provenance_boundary_id",
            min_units=min_units, n_boot=n_boot, ci=ci, seed=control_seed,
        )
        verdict["is_verified_source_boundary_evidence"] = verdict["n_units"] > 0
        _finish_boundary("source_scene_path_row", prov, prov["orient_counts"])
        verdicts["source_scene_path_row"] = verdict
        _log("boundary", "end", product=product, boundary_type="source_scene_path_row",
             n_units=verdict["n_units"], n_paired_edges=prov["sw_sink"].count
             if hasattr(prov["sw_sink"], "count") else None)
    else:
        verdicts["source_scene_path_row"] = {
            "status": "insufficient_boundary_metadata",
            "unit_type": "provenance_boundary_id",
            "is_verified_source_boundary_evidence": False,
            "provenance_status": provenance_status,
            "reason": (
                "verified source-scene/path-row boundaries require "
                "provenance_available AND a rasterized boundary code raster "
                "actually sampled; not reported as provenance_available without "
                "sampling"
            ),
        }

    return {
        "product": product,
        "metric_rows": metric_rows,
        "verdicts": verdicts,
        "control_provenance": control_provenance,
    }


# -----------------------------------------------------------------------------
# Local analysis checkpoint (SEPARATE from raster_inventory.partial.json)
# -----------------------------------------------------------------------------
def analysis_checkpoint_path(root) -> Path:
    return Path(root) / ANALYSIS_CHECKPOINT_NAME


def read_analysis_checkpoint(root) -> dict:
    """Read the local analysis checkpoint, or ``{}`` if absent/unreadable."""
    path = analysis_checkpoint_path(root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_analysis_checkpoint(root, payload) -> Path:
    """Atomically write the local analysis checkpoint."""
    return write_json_atomic(analysis_checkpoint_path(root), payload)


def files_present_and_signed(entries) -> bool:
    """Validate checkpoint file references: each listed file must still exist and
    match its recorded byte size. Checkpoint TEXT is never trusted on its own --
    a completed stage is only reused when its referenced files still validate.
    """
    for entry in entries or []:
        path = Path(entry.get("path", ""))
        if not path.exists():
            return False
        try:
            if int(entry.get("bytes", -1)) != int(path.stat().st_size):
                return False
        except OSError:
            return False
    return True


def clean_analysis_tmp(root) -> str | None:
    """Remove the temporary analysis scratch dir after a successful finalization.

    Namespace-safety is asserted (the scratch dir must live under the diagnostic
    root) before any deletion, so nothing outside ``_analysis_tmp`` is touched.
    """
    import shutil

    root = Path(root)
    tmp = root / ANALYSIS_TMP_DIRNAME
    if not tmp.exists():
        return None
    resolved = tmp.resolve()
    if resolved.name != ANALYSIS_TMP_DIRNAME or root.resolve() not in resolved.parents:
        raise NamespaceSafetyError(
            f"refusing to clean a path outside the analysis tmp namespace: {resolved}"
        )
    shutil.rmtree(tmp)
    return str(tmp)
