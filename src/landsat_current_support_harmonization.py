"""
landsat_current_support_harmonization.py

DIAGNOSTIC-ONLY current-period Landsat ACQUISITION-DATE OFFSET HARMONIZATION
counterfactual for the Manavgat date-balanced candidate, run after the completed
residual seam attribution audit whose final status was

    current_support_dominant

QUESTION
--------
The residual audit showed that the remaining seam is carried at CURRENT
observation-support boundaries, and that the current unique-acquisition-date
count boundary carries ~0.894 C excess absolute jump in current-minus-baseline
and ~0.538 z excess absolute jump in the anomaly, numerator-dominated. That is
consistent with -- but does not prove -- a mechanism in which different pixels
composite DIFFERENT MIXTURES OF ACQUISITION DATES, and the dates themselves sit
at different absolute temperature levels (different weather, different overpass
conditions). Where the date mixture changes across an adjacency edge, the
composite jumps even though the surface did not.

This experiment tests exactly that interaction:

    acquisition-date identity  x  spatially varying date support

by estimating one ADDITIVE offset per acquisition date from the SPATIAL OVERLAPS
between daily mosaics, subtracting it from each daily mosaic, and recomposing the
current-period median over the SAME per-pixel set of dates.

WHAT IS HELD FIXED (the intervention is one factor only)
--------------------------------------------------------
    same source scenes, same current-period date window, same Landsat scaling,
    same QA mask, same same-day mosaicking rule, same per-pixel valid date
    support, same valid pixel mask, same frozen four-year baseline climatology,
    same Step5 count and baseline-std masks, Celsius units throughout.

The ONLY changed factor is a per-date additive constant estimated from overlap
evidence.

WHAT THIS MODULE IS NOT
-----------------------
    - It is NOT a fix and NOT a production change. The production reducer and
      the frozen baseline climatology are never touched, never recomputed and
      never re-exported.
    - It performs NO spatial operation on any raster: nothing is smoothed,
      blended, feathered, interpolated, in-painted or cosmetically altered.
      `Y'_d(p) = Y_d(p) - alpha_d` is a pure per-date scalar subtraction.
    - It is NOT a common-date-subset experiment and NOT a minimum-count masking
      experiment. Raising the support threshold or dropping difficult pixels
      would be a different (and much weaker) claim, so an EXACT support
      invariance gate makes it impossible to pass by changing the pixel
      population.
    - It NEVER uses labels, burned-area data, Step8 metrics or any model
      performance number, either to construct the candidate or to select it.
    - It NEVER emits `seam_fixed`, `production_approved` or `production_ready`.
    - It contains NO Earth Engine code. The daily current-period mosaics it
      consumes are prepared by the runner's isolated export stage; every
      analysis callable in this module is local-only.

MODEL
-----
For acquisition date d and pixel p, the daily mosaic is modelled as

    Y_d(p) = T(p) + alpha_d + error_d(p)

with T(p) the (unknown) date-invariant surface field and alpha_d a scene-wide
additive date offset. Overlaps identify the DIFFERENCES alpha_j - alpha_i only,
so the level is fixed by the predeclared identifying constraint

    sum_d n_d * alpha_d = 0,     n_d = valid pixel count of daily mosaic d

which anchors the physical scale to the whole observed population rather than to
an arbitrarily chosen single date.

ISOLATION CONTRACT
------------------
Everything this module writes lives under

    outputs/diagnostics/landsat_current_support_harmonization/<experiment_id>/

The frozen counterfactual, downstream A/B, residual seam attribution and
canonical experiment namespaces are READ-ONLY inputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path

import src.landsat_composite_counterfactual_audit as audit
import src.landsat_composite_downstream_ab as ab
import src.landsat_residual_seam_attribution as rs
from core.paths import PROJECT_ROOT

# -----------------------------------------------------------------------------
# Shared primitives -- ONE implementation, ONE contract. Every one of these is
# reused verbatim so this experiment is measured on exactly the same lattice,
# the same spatial blocks and the same bootstrap machinery as the audit whose
# result it is trying to move.
# -----------------------------------------------------------------------------
NamespaceSafetyError = audit.NamespaceSafetyError
GridMismatchError = audit.GridMismatchError
write_json_atomic = audit.write_json_atomic
sha256_and_size = audit.sha256_and_size
grid_signature = audit.grid_signature
assert_same_grid = audit.assert_same_grid
process_rss_mib = audit.process_rss_mib
NODATA_SENTINEL = audit.NODATA_SENTINEL
ORIENTATIONS = audit.ORIENTATIONS

# Reused from the residual seam attribution audit (identical semantics).
_edge_pairs = rs._edge_pairs
edge_valid_mask = rs.edge_valid_mask
edge_difference = rs.edge_difference
edge_anchor_rows_cols = rs.edge_anchor_rows_cols
spatial_block_ids = rs.spatial_block_ids
block_id_to_label = rs.block_id_to_label
gradient_bin = rs.gradient_bin
stratum_keys = rs.stratum_keys
build_edge_flags = rs.build_edge_flags
stratified_class_codes = rs.stratified_class_codes
control_pair_mask = rs.control_pair_mask
StratumAccumulator = rs.StratumAccumulator
MeanAccumulator = rs.MeanAccumulator
HistogramAccumulator = rs.HistogramAccumulator
matched_block_accumulators = rs.matched_block_accumulators
draw_bootstrap_indices = rs.draw_bootstrap_indices
iter_row_windows = rs.iter_row_windows
read_window = rs.read_window
rasterize_pathrow_boundaries = rs.rasterize_pathrow_boundaries
resolve_pathrow_availability = rs.resolve_pathrow_availability
STRATUM_SPACE = rs.STRATUM_SPACE


class HarmonizationError(RuntimeError):
    """Fail-fast error for the date-offset harmonization counterfactual."""


class PrerequisiteError(HarmonizationError):
    """A required frozen input or upstream prerequisite is missing/invalid."""


class SupportInvarianceError(HarmonizationError):
    """The candidate changed the pixel population -- the experiment is void."""


# =============================================================================
# Identity / versions
# =============================================================================
DIAGNOSTIC_NAMESPACE = "landsat_current_support_harmonization"
COUNTERFACTUAL_NAMESPACE = audit.DIAGNOSTIC_NAMESPACE
DOWNSTREAM_AB_NAMESPACE = ab.DIAGNOSTIC_NAMESPACE
RESIDUAL_SEAM_NAMESPACE = rs.DIAGNOSTIC_NAMESPACE

REPORT_SCHEMA_VERSION = "1.0-overlap-date-harmonization"
DECISION_RULE_VERSION = "1.0-harmonization-ordered"

#: One AOI only. A second AOI needs its own frozen inputs and predeclaration.
SUPPORTED_EXPERIMENT_IDS = ("manavgat_2021",)

#: The two composites under comparison.
REFERENCE_COMPOSITE = "date_balanced_reference"
CANDIDATE_COMPOSITE = "overlap_harmonized_date_balanced"

#: The chain the reference composite belongs to (frozen downstream A/B side).
CANDIDATE_CHAIN = ab.CHAIN_CANDIDATE          # date_balanced_lst_only
CANDIDATE_SIDE = ab.CHAIN_SIDE[CANDIDATE_CHAIN]  # "candidate"

#: Required upstream statuses -- all four are checked before anything runs.
REQUIRED_COUNTERFACTUAL_FINAL_STATUS = "supported_reduction"
REQUIRED_DOWNSTREAM_AB_FINAL_STATUS = "eligible_for_second_aoi_validation"
REQUIRED_RESIDUAL_SEAM_FINAL_STATUS = "current_support_dominant"
REQUIRED_AB_REFERENCE_REPRODUCTION = "pass"

#: Hard invariants asserted in the config snapshot and every report.
USES_LABELS = False
USES_STEP8_METRICS = False
USES_MODEL_PERFORMANCE = False
SMOOTHING_APPLIED = False
SPATIAL_INTERPOLATION_APPLIED = False
RECOMPUTES_BASELINE = False
CHANGES_PRODUCTION_REDUCER = False


# =============================================================================
# Target products
# =============================================================================
TARGET_LST = "current_lst_celsius"
TARGET_CMB = "current_minus_baseline_celsius"
TARGET_ANOMALY = "anomaly_zscore"
TARGET_PRODUCTS = (TARGET_LST, TARGET_CMB, TARGET_ANOMALY)

PRODUCT_UNITS = OrderedDict((
    (TARGET_LST, "celsius"),
    (TARGET_CMB, "celsius"),
    (TARGET_ANOMALY, "zscore"),
))

#: The two products whose support-boundary reduction the decision rule requires.
DECISION_PRODUCTS = (TARGET_CMB, TARGET_ANOMALY)


# =============================================================================
# Predeclared final statuses (ORDERED; see decide_final_status)
# =============================================================================
STATUS_INVALID_INPUTS = "invalid_inputs"
STATUS_INVALID_REFERENCE = "invalid_reference_reproduction"
STATUS_INSUFFICIENT_GRAPH = "insufficient_date_overlap_graph"
STATUS_SUPPORT_INVARIANCE_FAILED = "support_invariance_failed"
STATUS_NOT_SUPPORTED = "seam_reduction_not_supported"
STATUS_NONBOUNDARY_TRADEOFF = "seam_reduced_with_nonboundary_tradeoff"
STATUS_VALUE_SCALE_TRADEOFF = "seam_reduced_with_value_scale_tradeoff"
STATUS_ELIGIBLE = "eligible_for_downstream_ab"

FINAL_STATUSES = (
    STATUS_INVALID_INPUTS,
    STATUS_INVALID_REFERENCE,
    STATUS_INSUFFICIENT_GRAPH,
    STATUS_SUPPORT_INVARIANCE_FAILED,
    STATUS_NOT_SUPPORTED,
    STATUS_NONBOUNDARY_TRADEOFF,
    STATUS_VALUE_SCALE_TRADEOFF,
    STATUS_ELIGIBLE,
)

#: Conclusions this experiment can NEVER reach, in any field of any report.
FORBIDDEN_CONCLUSIONS = ("seam_fixed", "production_approved", "production_ready")

FINAL_STATUS_MEANINGS = OrderedDict((
    (STATUS_INVALID_INPUTS,
     "A required frozen input, upstream status or grid contract failed. No "
     "scientific claim is made."),
    (STATUS_INVALID_REFERENCE,
     "The frozen date-balanced current composite could not be reproduced from "
     "the daily mosaics, so the candidate has no valid reference to be "
     "compared against. No scientific claim is made."),
    (STATUS_INSUFFICIENT_GRAPH,
     "The primary date-overlap graph is not one connected component, so the "
     "per-date offsets are not jointly identified. No harmonized candidate "
     "raster is presented as valid."),
    (STATUS_SUPPORT_INVARIANCE_FAILED,
     "The candidate did not preserve the exact per-pixel date support / valid "
     "mask. Any apparent seam change could be masking rather than "
     "harmonization, so the experiment is void."),
    (STATUS_NOT_SUPPORTED,
     "Support-boundary excess jump reduction is not supported: the paired "
     "block-bootstrap interval crosses zero, or the predeclared minimum 10% "
     "point relative reduction was not reached."),
    (STATUS_NONBOUNDARY_TRADEOFF,
     "Support-boundary reduction is supported, but it is accompanied by a "
     "supported INCREASE away from the targeted support boundaries "
     "(non-boundary terrain or path/row-only pairs), which is consistent with "
     "over-correction rather than with removing a support artefact."),
    (STATUS_VALUE_SCALE_TRADEOFF,
     "Support-boundary reduction is supported, but the fitted offsets move the "
     "absolute value scale further than predeclared (global median shift or a "
     "single date offset exceeds its bound), or the graph residual "
     "diagnostics indicate unstable offset estimation."),
    (STATUS_ELIGIBLE,
     "Eligible to be carried into a controlled downstream A/B ONLY. This is "
     "NOT a fix, NOT production acceptance and NOT a production reducer "
     "change, and it does not prove that current support is the only seam "
     "mechanism."),
))


# =============================================================================
# Step5 policy (frozen; read from the canonical configuration, never redefined)
# =============================================================================
def step5_thresholds() -> dict:
    """The frozen Step5 guard thresholds, read from the canonical config."""
    return rs.step5_thresholds()


#: Physical LST validity window applied by canonical Step5 to the current
#: composite. Reused verbatim so the reference reproduction and the candidate
#: pass through EXACTLY the same guard.
PHYSICAL_CELSIUS_MIN = -30.0
PHYSICAL_CELSIUS_MAX = 80.0

#: Landsat scaling recorded in the frozen counterfactual provenance. Held here
#: only to be asserted against that provenance -- never redefined.
LANDSAT_SCALE = audit.LANDSAT_SCALE
LANDSAT_OFFSET = audit.LANDSAT_OFFSET


# =============================================================================
# Reference reproduction tolerances (PREDECLARED, before any run)
# =============================================================================
#: The gating tolerance for the physical float32 reproduction of each product is
#: the EXISTING project tolerance for that product in the frozen downstream A/B
#: reproduction policy. It is a failure boundary, not an expectation.
REPRODUCTION_TOLERANCES = OrderedDict((
    (TARGET_LST, ab.REPRODUCTION_TOLERANCES["current_lst_celsius"]),
    (TARGET_CMB, ab.REPRODUCTION_TOLERANCES["current_minus_baseline_celsius"]),
    (TARGET_ANOMALY, ab.REPRODUCTION_TOLERANCES["anomaly_zscore"]),
))

#: Additionally REPORTED (never gating): the much tighter float32 round-off
#: tolerance used by the counterfactual audit's canonical gate.
REPRODUCTION_TIGHT_REFERENCE_TOL = audit.REPRODUCTION_TOLERANCES["physical_float32"]

#: Grid, valid mask and valid-date count reproduction are EXACT -- no tolerance.
REPRODUCTION_EXACT_CHECKS = (
    "grid_signature_equality",
    "valid_mask_equality",
    "valid_date_count_equality",
)


# =============================================================================
# Overlap graph: predeclared eligibility and estimation constants
# =============================================================================
#: Spatial blocks are the SAME predeclared robust-diagnostic blocks used by the
#: residual seam attribution audit, so "independent block" means the same thing
#: in the graph and in the bootstrap.
GRAPH_BLOCK_SIZE_CELLS = rs.BOOTSTRAP_BLOCK_SIZE_CELLS      # 128

#: A block contributes ONE median only when it carries at least this many pixels
#: valid on BOTH dates. Predeclared; never tuned after seeing results.
MIN_BLOCK_COMMON_PIXELS = 100

#: PRIMARY edge eligibility. The primary candidate uses ONLY this graph.
PRIMARY_MIN_COMMON_PIXELS = 10000
PRIMARY_MIN_INDEPENDENT_BLOCKS = 8

#: Unconditional sensitivity graphs. Reported ALWAYS, never selected from.
SENSITIVITY_THRESHOLDS = (
    OrderedDict((("label", "primary"),
                 ("min_common_pixels", PRIMARY_MIN_COMMON_PIXELS),
                 ("min_independent_blocks", PRIMARY_MIN_INDEPENDENT_BLOCKS))),
    OrderedDict((("label", "loose_5000_5"),
                 ("min_common_pixels", 5000), ("min_independent_blocks", 5))),
    OrderedDict((("label", "strict_25000_12"),
                 ("min_common_pixels", 25000), ("min_independent_blocks", 12))),
)

THRESHOLD_SELECTION_POLICY = (
    "The PRIMARY candidate is built from the 10000-pixel / 8-block graph and "
    "from nothing else. The 5000/5 and 25000/12 graphs are solved and reported "
    "UNCONDITIONALLY as sensitivity evidence. No threshold set may be chosen "
    "on the basis of which one produces the better seam result; the primary "
    "set was fixed before the graph was ever built."
)

#: Robust dispersion of an edge: scaled MAD of its block medians. Floored so a
#: pathologically small dispersion cannot make one edge dominate the fit.
MAD_TO_SIGMA = 1.4826
MIN_EDGE_SIGMA_CELSIUS = 0.05

#: Weight cap: no single edge may carry more than this multiple of the MEDIAN
#: edge weight, so one date pair cannot dominate the least-squares solution.
WEIGHT_CAP_MULTIPLE = 10.0

EDGE_WEIGHT_FORMULA = (
    "w_ij = n_blocks_ij / sigma_ij^2 with sigma_ij = max(1.4826 * MAD of the "
    "eligible block medians, 0.05 C), then capped at 10x the median raw edge "
    "weight. Weights use ONLY overlap evidence (independent block count and "
    "robust edge dispersion); no target, label, model result or seam metric "
    "enters the weighting."
)

IDENTIFYING_CONSTRAINT = (
    "weighted mean of alpha_d is zero, with date weights equal to the number "
    "of valid pixels in that date's daily mosaic. The physical scale is NOT "
    "anchored to an arbitrarily chosen single date."
)

#: Solver diagnostics: an offset solution is flagged UNSTABLE when either bound
#: is exceeded. Predeclared.
MAX_GRAPH_CONDITION_NUMBER = 1e8
MAX_EDGE_RESIDUAL_RMS_CELSIUS = 1.0


# =============================================================================
# Predeclared decision bounds
# =============================================================================
#: An individual fitted date offset larger than this is a value-scale trade-off.
MAX_ABS_DATE_OFFSET_CELSIUS = 5.0

#: A global candidate-minus-reference median current-LST shift larger than this
#: is a value-scale trade-off.
MAX_ABS_GLOBAL_MEDIAN_SHIFT_CELSIUS = 0.5

#: Minimum POINT relative reduction of the support-boundary excess jump.
MIN_RELATIVE_REDUCTION = 0.10


# =============================================================================
# Bootstrap configuration (identical to the residual seam attribution audit)
# =============================================================================
BOOTSTRAP_REPLICATES = rs.BOOTSTRAP_REPLICATES      # 1000
BOOTSTRAP_SEED = rs.BOOTSTRAP_SEED                  # 42
BOOTSTRAP_CI = rs.BOOTSTRAP_CI                      # 0.95
BOOTSTRAP_CI_LOWER_PCT = rs.BOOTSTRAP_CI_LOWER_PCT  # 2.5
BOOTSTRAP_CI_UPPER_PCT = rs.BOOTSTRAP_CI_UPPER_PCT  # 97.5
BOOTSTRAP_BLOCK_SIZE_CELLS = rs.BOOTSTRAP_BLOCK_SIZE_CELLS   # 128
MIN_BOOTSTRAP_UNITS = rs.MIN_BOOTSTRAP_UNITS        # 8

BOOTSTRAP_UNIT_POLICY = (
    "Spatial blocks -- never individual pixel pairs -- are resampled with "
    "replacement. Reference and candidate are evaluated on the SAME pair "
    "population and the SAME single index matrix, so every comparison is "
    "genuinely paired."
)


# =============================================================================
# Boundary definitions (reused verbatim from the residual seam attribution)
# =============================================================================
#: Every boundary the experiment must report, in report order. The first twelve
#: are evaluated as boundary-vs-matched-control EXCESS; `none_of_known_boundaries`
#: is the non-boundary control population itself and is evaluated as a plain
#: mean absolute jump (an "excess" against itself would be identically zero).
EVAL_MODE_EXCESS = "excess_vs_matched_control"
EVAL_MODE_MEAN = "mean_absolute_jump"

EVALUATED_BOUNDARIES = OrderedDict((
    ("current_support_change", EVAL_MODE_EXCESS),
    ("current_unique_date_count_change", EVAL_MODE_EXCESS),
    ("current_scene_count_change", EVAL_MODE_EXCESS),
    ("current_valid_count_change", EVAL_MODE_EXCESS),
    ("same_day_multiplicity_change", EVAL_MODE_EXCESS),
    ("baseline_valid_year_change", EVAL_MODE_EXCESS),
    ("baseline_annual_date_support_change", EVAL_MODE_EXCESS),
    ("near_std_threshold_boundary", EVAL_MODE_EXCESS),
    ("source_path_row_boundary", EVAL_MODE_EXCESS),
    (rs.CLASS_PATHROW_ONLY, EVAL_MODE_EXCESS),
    (rs.CLASS_SUPPORT_AND_PATHROW, EVAL_MODE_EXCESS),
    (rs.CLASS_SUPPORT_ONLY, EVAL_MODE_EXCESS),
    (rs.CLASS_NONE, EVAL_MODE_MEAN),
))

#: Boundaries whose reduction the strongest status REQUIRES.
REQUIRED_REDUCTION_BOUNDARIES = (
    "current_support_change",
    "current_unique_date_count_change",
)

#: The primary non-boundary control for the trade-off check.
NONBOUNDARY_CONTROL = rs.CLASS_NONE

#: Boundaries at which a supported INCREASE blocks the strongest status.
NO_SUPPORTED_INCREASE_BOUNDARIES = (rs.CLASS_PATHROW_ONLY, rs.CLASS_NONE)

#: Predeclared mapping of a supported increase at `pathrow_only` onto a status.
#: A path/row-only increase is a trade-off at a boundary the intervention did
#: not target, so it is reported under the non-boundary trade-off status with
#: its own explicit reason string. Fixed before any result was inspected.
PATHROW_INCREASE_STATUS = STATUS_NONBOUNDARY_TRADEOFF

#: Verdict vocabulary for a paired reduction interval.
VERDICT_SUPPORTED_REDUCTION = "supported_reduction"
VERDICT_SUPPORTED_INCREASE = "supported_increase"
VERDICT_UNCERTAIN = "uncertain"
VERDICT_INSUFFICIENT = "insufficient_evidence"


# =============================================================================
# Raster-change reporting thresholds (DESCRIPTIVE ONLY -- never gate anything)
# =============================================================================
CELSIUS_CHANGE_THRESHOLDS = (0.05, 0.25, 0.5, 1.0, 2.0)

#: Predeclared anomaly-equivalent thresholds. z units, chosen as the Celsius
#: thresholds divided by the Step5 minimum baseline std (1.0 C) rounded to the
#: reporting grid already used by the frozen A/B changed-pixel policy.
ANOMALY_CHANGE_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50)

CHANGE_THRESHOLDS = OrderedDict((
    (TARGET_LST, CELSIUS_CHANGE_THRESHOLDS),
    (TARGET_CMB, CELSIUS_CHANGE_THRESHOLDS),
    (TARGET_ANOMALY, ANOMALY_CHANGE_THRESHOLDS),
))

#: Fixed-edge histogram ranges for deterministic bounded-memory quantiles.
HISTOGRAM_BINS = 200000
HISTOGRAM_MAX = OrderedDict((
    (TARGET_LST, 50.0),
    (TARGET_CMB, 50.0),
    (TARGET_ANOMALY, 20.0),
))

#: Windowed streaming: rows per read window. Aligned to the spatial-block size
#: so a graph block never straddles two windows.
WINDOW_ROWS = GRAPH_BLOCK_SIZE_CELLS
EDGE_WINDOW_ROWS = rs.WINDOW_ROWS


# =============================================================================
# Namespace resolution and safety
# =============================================================================
def diagnostic_output_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """The ONE directory this experiment may write beneath."""
    return Path(base_dir) / "outputs" / "diagnostics" / DIAGNOSTIC_NAMESPACE / experiment_id


def counterfactual_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen composite-counterfactual root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "diagnostics" / COUNTERFACTUAL_NAMESPACE / experiment_id


def downstream_ab_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen downstream A/B root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "diagnostics" / DOWNSTREAM_AB_NAMESPACE / experiment_id


def residual_seam_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen residual-seam attribution root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "diagnostics" / RESIDUAL_SEAM_NAMESPACE / experiment_id


def canonical_experiment_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen canonical experiment root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "experiments" / experiment_id


def forbidden_write_roots(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> list[Path]:
    """Roots that must never be written, overwritten or deleted."""
    return [
        counterfactual_root(experiment_id, base_dir),
        downstream_ab_root(experiment_id, base_dir),
        residual_seam_root(experiment_id, base_dir),
        canonical_experiment_root(experiment_id, base_dir),
        Path(base_dir) / "data",
        Path(base_dir) / "config",
        Path(base_dir) / "outputs" / "step5",
        Path(base_dir) / "outputs" / "step5c",
        Path(base_dir) / "outputs" / "step3",
    ]


def assert_namespace_safe(paths, experiment_id: str, base_dir: Path = PROJECT_ROOT) -> None:
    """Every supplied write path must resolve strictly under this root."""
    root = diagnostic_output_root(experiment_id, base_dir).resolve()
    forbidden = [p.resolve() for p in forbidden_write_roots(experiment_id, base_dir)]

    for raw in paths:
        candidate = Path(raw).resolve()
        for bad in forbidden:
            if candidate == bad or bad in candidate.parents:
                raise NamespaceSafetyError(
                    f"refusing to write inside a frozen namespace: {candidate} "
                    f"(forbidden root: {bad})"
                )
        if candidate != root and root not in candidate.parents:
            raise NamespaceSafetyError(
                f"refusing to write outside the dedicated harmonization root: "
                f"{candidate} (allowed root: {root})"
            )


def clear_diagnostic_namespace(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> str | None:
    """`--force` deletion of ONLY the dedicated harmonization namespace."""
    root = diagnostic_output_root(experiment_id, base_dir)
    if not root.exists():
        return None
    resolved = root.resolve()
    assert_namespace_safe([resolved], experiment_id, base_dir)
    expected = diagnostic_output_root(experiment_id, base_dir).resolve()
    if resolved != expected:
        raise NamespaceSafetyError(
            f"refusing to delete {resolved}: it is not the dedicated root {expected}"
        )
    if DIAGNOSTIC_NAMESPACE not in resolved.parts or experiment_id not in resolved.parts:
        raise NamespaceSafetyError(
            f"refusing to delete a path that is not namespaced to this "
            f"experiment: {resolved}"
        )
    shutil.rmtree(resolved)
    return str(resolved)


def assert_supported_experiment(experiment_id: str) -> None:
    if experiment_id not in SUPPORTED_EXPERIMENT_IDS:
        raise HarmonizationError(
            f"unsupported --experiment {experiment_id!r}. This experiment supports "
            f"only {list(SUPPORTED_EXPERIMENT_IDS)}; another AOI needs its own "
            "frozen inputs and its own predeclaration."
        )


# =============================================================================
# Output layout
# =============================================================================
def plan_output_layout(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, Path]":
    """The full planned directory layout (informational; creates nothing)."""
    root = diagnostic_output_root(experiment_id, base_dir)
    return OrderedDict((
        ("root", root),
        ("config", root / "config"),
        ("checkpoints", root / "checkpoints"),
        ("daily", root / "daily"),
        ("daily_reference", root / "daily" / "reference"),
        ("daily_harmonized", root / "daily" / "harmonized"),
        ("graph", root / "graph"),
        ("rasters", root / "rasters"),
        ("tables", root / "tables"),
        ("maps", root / "maps"),
    ))


RASTER_FILES = (
    "reference_current_lst_celsius.tif",
    "harmonized_current_lst_celsius.tif",
    "reference_current_minus_baseline_celsius.tif",
    "harmonized_current_minus_baseline_celsius.tif",
    "reference_anomaly_zscore.tif",
    "harmonized_anomaly_zscore.tif",
    "candidate_minus_reference_current_lst.tif",
    "candidate_minus_reference_current_minus_baseline.tif",
    "candidate_minus_reference_anomaly.tif",
    # Support-invariance evidence rasters (integer-valued, float32 encoded).
    "reference_unique_date_valid_count.tif",
    "harmonized_unique_date_valid_count.tif",
    "reference_date_membership_bitmask.tif",
    "harmonized_date_membership_bitmask.tif",
)

GRAPH_FILES = (
    "date_nodes.csv",
    "date_edges.csv",
    "date_offsets.csv",
    "graph_components.json",
    "graph_diagnostics.json",
)

TABLE_FILES = (
    "raster_change_summary.csv",
    "boundary_jump_comparison.csv",
    "paired_bootstrap_summary.csv",
    "nonboundary_tradeoff.csv",
    "date_offset_sensitivity.csv",
)

DOCUMENT_FILES = (
    "reference_reproduction.json",
    "support_invariance.json",
    "harmonization_summary.json",
    "harmonization_summary.md",
    "harmonization_manifest.json",
    "input_provenance.json",
)

MAP_FILES = (
    "reference_current_lst.png",
    "harmonized_current_lst.png",
    "difference_current_lst.png",
    "reference_current_minus_baseline.png",
    "harmonized_current_minus_baseline.png",
    "difference_current_minus_baseline.png",
    "reference_anomaly.png",
    "harmonized_anomaly.png",
    "difference_anomaly.png",
    "current_support_boundaries_over_reference.png",
    "current_support_boundaries_over_candidate.png",
    "top_1_percent_residual_jump_pairs.png",
    "date_offset_graph.png",
    "per_date_offset_magnitude.png",
)


def plan_expected_files(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, Path]":
    """Every file the live run is expected to produce (creates nothing)."""
    layout = plan_output_layout(experiment_id, base_dir)
    root = layout["root"]
    expected: "OrderedDict[str, Path]" = OrderedDict()
    expected["config/harmonization_config.json"] = layout["config"] / "harmonization_config.json"
    expected["checkpoints/harmonization_checkpoint.json"] = (
        layout["checkpoints"] / CHECKPOINT_FILENAME
    )
    expected["daily/daily_inventory.json"] = layout["daily"] / "daily_inventory.json"
    for name in GRAPH_FILES:
        expected[f"graph/{name}"] = layout["graph"] / name
    for name in RASTER_FILES:
        expected[f"rasters/{name}"] = layout["rasters"] / name
    for name in TABLE_FILES:
        expected[f"tables/{name}"] = layout["tables"] / name
    for name in MAP_FILES:
        expected[f"maps/{name}"] = layout["maps"] / name
    for name in DOCUMENT_FILES:
        expected[name] = root / name
    return expected


# =============================================================================
# Frozen inputs
# =============================================================================
def baseline_years(experiment_id: str) -> list[int]:
    return rs.baseline_years(experiment_id)


def candidate_step5_dir(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    return downstream_ab_root(experiment_id, base_dir) / CANDIDATE_SIDE / "step5"


def candidate_derived_dir(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    return downstream_ab_root(experiment_id, base_dir) / CANDIDATE_SIDE / ab.DERIVED_SUBDIR


def build_input_plan(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, dict]":
    """Logical input role -> {path, source, required, family, purpose}.

    Filenames are never assumed blind: every path is resolved against the frozen
    namespaces and reported as missing rather than invented. The frozen baseline
    rasters are inputs ONLY -- they are read, never recomputed and never
    re-exported.
    """
    step5 = candidate_step5_dir(experiment_id, base_dir)
    derived = candidate_derived_dir(experiment_id, base_dir)
    cf = counterfactual_root(experiment_id, base_dir) / "rasters"
    shared = downstream_ab_root(experiment_id, base_dir) / "inputs" / "shared"

    plan: "OrderedDict[str, dict]" = OrderedDict()

    def add(role, path, *, source, required, family, purpose):
        plan[role] = OrderedDict((
            ("role", role),
            ("path", Path(path)),
            ("source", source),
            ("required", bool(required)),
            ("family", family),
            ("purpose", purpose),
        ))

    # --- the frozen reference composite and its derived products -------------
    add("frozen_reference_current_lst_celsius",
        step5 / "current_period_median_celsius.tif",
        source="downstream_ab_candidate", required=True, family="reference",
        purpose="frozen date-balanced current LST; the reproduction target")
    add("frozen_reference_current_minus_baseline_celsius",
        derived / "current_minus_baseline_celsius.tif",
        source="downstream_ab_candidate", required=True, family="reference",
        purpose="frozen D = C - M; reproduction target")
    add("frozen_reference_anomaly_zscore", step5 / "anomaly_zscore.tif",
        source="downstream_ab_candidate", required=True, family="reference",
        purpose="frozen Z = D / S; reproduction target and anomaly valid mask")

    # --- the FROZEN baseline climatology (held fixed; never recomputed) ------
    add("baseline_lst_mean_celsius", step5 / "baseline_lst_mean_celsius.tif",
        source="downstream_ab_candidate", required=True, family="frozen_baseline",
        purpose="M; frozen four-year baseline mean, held fixed")
    add("baseline_lst_std_celsius", step5 / "baseline_lst_std_celsius.tif",
        source="downstream_ab_candidate", required=True, family="frozen_baseline",
        purpose="S; frozen four-year baseline std, held fixed")
    add("baseline_valid_count", step5 / "baseline_valid_count.tif",
        source="downstream_ab_candidate", required=True, family="frozen_baseline",
        purpose="frozen baseline valid-YEAR count, held fixed")
    add("low_baseline_std_mask", step5 / "low_baseline_std_mask.tif",
        source="downstream_ab_candidate", required=True, family="frozen_mask",
        purpose="frozen Step5 low-baseline-std guard flag")
    add("low_baseline_count_mask", step5 / "low_baseline_count_mask.tif",
        source="downstream_ab_candidate", required=True, family="frozen_mask",
        purpose="frozen Step5 low-baseline-count guard flag")

    # --- current-period support and masks ------------------------------------
    add("current_period_valid_count", step5 / "current_period_valid_count.tif",
        source="downstream_ab_candidate", required=True, family="support",
        purpose="Step5 current support (unique-acquisition-date semantics)")
    add("low_current_count_mask", step5 / "low_current_count_mask.tif",
        source="downstream_ab_candidate", required=True, family="mask",
        purpose="Step5 low-current-count guard flag")
    add("current_unique_date_valid_count", cf / "current_lst_unique_date_valid_count.tif",
        source="counterfactual", required=True, family="support",
        purpose="current unique acquisition-date support")
    add("current_scene_valid_count", cf / "current_lst_scene_valid_count.tif",
        source="counterfactual", required=True, family="support",
        purpose="current raw scene-observation support")
    add("current_same_day_multiplicity", cf / "current_lst_same_day_multiplicity.tif",
        source="counterfactual", required=True, family="support",
        purpose="same-day multiplicity support")

    # --- per-year baseline support (OPTIONAL; reported when absent) ----------
    for year in baseline_years(experiment_id):
        add(f"baseline_{year}_unique_date_valid_count",
            cf / f"baseline_lst_{year}_unique_date_valid_count.tif",
            source="counterfactual", required=False, family="support",
            purpose=f"per-year baseline unique-date support ({year})")

    # --- matched-control covariates (OPTIONAL) -------------------------------
    add("elevation", shared / "dem" / "elevation.tif",
        source="downstream_ab_shared", required=False, family="covariate",
        purpose="elevation-gradient matching bin")
    add("slope", shared / "dem" / "slope.tif",
        source="downstream_ab_shared", required=False, family="covariate",
        purpose="slope-gradient matching bin")
    add("ndvi_current", shared / "ndvi_current_period" / "current_ndvi_median.tif",
        source="downstream_ab_shared", required=False, family="covariate",
        purpose="NDVI-gradient matching bin (no new data dependency)")

    # --- frozen provenance the daily inventory is derived FROM ---------------
    add("scene_manifest", counterfactual_root(experiment_id, base_dir) / "scene_manifest.csv",
        source="counterfactual", required=True, family="provenance",
        purpose="frozen source-scene inventory; the ONLY scene list used")
    add("audit_config", counterfactual_root(experiment_id, base_dir) / "audit_config.json",
        source="counterfactual", required=True, family="provenance",
        purpose="frozen current-period date window, QA mask and Landsat scaling")
    add("source_scene_metadata",
        counterfactual_root(experiment_id, base_dir) / "source_scene_metadata.json",
        source="counterfactual", required=False, family="provenance",
        purpose="frozen per-scene metadata (path/row, acquisition datetime)")

    return plan


def pathrow_boundary_sources(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Frozen metadata-derived path/row artefacts (READ-ONLY, all optional)."""
    return rs.pathrow_boundary_sources(experiment_id, base_dir)


def upstream_report_paths(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Frozen upstream reports whose prerequisites this experiment checks."""
    cf_root = counterfactual_root(experiment_id, base_dir)
    ab_root = downstream_ab_root(experiment_id, base_dir)
    rs_root = residual_seam_root(experiment_id, base_dir)
    return OrderedDict((
        ("counterfactual_summary", cf_root / "counterfactual_summary.json"),
        ("counterfactual_manifest", cf_root / "manifest.json"),
        ("downstream_ab_summary", ab_root / "downstream_ab_summary.json"),
        ("downstream_ab_manifest", ab_root / "downstream_ab_manifest.json"),
        ("downstream_ab_reference_reproduction", ab_root / "reference_reproduction.json"),
        ("downstream_ab_input_provenance", ab_root / "input_provenance.json"),
        ("residual_seam_summary", rs_root / "residual_seam_summary.json"),
        ("residual_seam_manifest", rs_root / "residual_seam_manifest.json"),
    ))


def missing_required_inputs(plan: "OrderedDict[str, dict]") -> list[str]:
    return [
        f"{entry['role']}={entry['path']}"
        for entry in plan.values()
        if entry["required"] and not Path(entry["path"]).exists()
    ]


def missing_optional_inputs(plan: "OrderedDict[str, dict]") -> list[str]:
    return [
        f"{entry['role']}={entry['path']}"
        for entry in plan.values()
        if not entry["required"] and not Path(entry["path"]).exists()
    ]


def assert_required_inputs(plan: "OrderedDict[str, dict]", experiment_id: str) -> None:
    missing = missing_required_inputs(plan)
    if missing:
        raise PrerequisiteError(
            f"experiment {experiment_id!r} is missing required frozen inputs for the "
            "date-offset harmonization counterfactual; this experiment never "
            "regenerates them and never recomputes the baseline. Missing:\n  "
            + "\n  ".join(missing)
        )


# =============================================================================
# Upstream prerequisite validation
# =============================================================================
def _read_json(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_upstream_state(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Read the four frozen upstream statuses this experiment depends on."""
    paths = upstream_report_paths(experiment_id, base_dir)
    cf = _read_json(paths["counterfactual_summary"]) or {}
    ab_summary = _read_json(paths["downstream_ab_summary"]) or {}
    ab_repro = _read_json(paths["downstream_ab_reference_reproduction"]) or {}
    seam = _read_json(paths["residual_seam_summary"]) or {}

    cf_repro = (cf.get("canonical_reproduction") or {})
    baseline_invariance = _extract_baseline_invariance(ab_summary, cf)

    state = OrderedDict((
        ("counterfactual_final_status", cf.get("final_status")),
        ("counterfactual_final_status_required", REQUIRED_COUNTERFACTUAL_FINAL_STATUS),
        ("downstream_ab_final_status", ab_summary.get("final_status")),
        ("downstream_ab_final_status_required", REQUIRED_DOWNSTREAM_AB_FINAL_STATUS),
        ("residual_seam_final_status", seam.get("final_status")),
        ("residual_seam_final_status_required", REQUIRED_RESIDUAL_SEAM_FINAL_STATUS),
        ("downstream_ab_reference_reproduction",
         ab_repro.get("status") or ab_repro.get("overall_status")
         or (ab_summary.get("technical_validity") or {}).get("reference_reproduction")),
        ("downstream_ab_reference_reproduction_required", REQUIRED_AB_REFERENCE_REPRODUCTION),
        ("counterfactual_canonical_reproduction",
         cf_repro.get("status") or cf_repro.get("final_status")),
        ("baseline_invariance", baseline_invariance["status"]),
        ("baseline_invariance_source", baseline_invariance["source"]),
        ("baseline_invariance_required", "pass"),
        ("production_approved", bool(ab_summary.get("production_approved"))),
        ("changes_production_reducer", bool(ab_summary.get("changes_production_reducer"))),
        ("reports_present", OrderedDict(
            (key, Path(value).exists()) for key, value in paths.items()
        )),
    ))
    state["prerequisites_met"] = upstream_prerequisites_met(state)
    state["failures"] = _prerequisite_failures(state)
    return state


def _extract_baseline_invariance(ab_summary: dict, cf_summary: dict) -> dict:
    """Locate the frozen baseline-invariance verdict without inventing one.

    The downstream A/B experiment held the baseline fixed between chains and
    recorded that fact; the counterfactual audit recorded the same invariance in
    its canonical gate. Whichever is present is reported with its source. When
    neither is present the status is `unknown`, which fails the gate rather than
    being optimistically treated as a pass.
    """
    technical = ab_summary.get("technical_validity") or {}
    for key in ("baseline_invariance", "shared_baseline_invariance",
                "baseline_invariance_status"):
        if key in technical and technical[key] is not None:
            value = technical[key]
            status = value.get("status") if isinstance(value, dict) else value
            if isinstance(status, bool):
                status = "pass" if status else "fail"
            return {"status": status, "source": f"downstream_ab_summary.technical_validity.{key}"}

    invariance = ab_summary.get("shared_modis_invariance")
    if isinstance(invariance, dict) and invariance.get("baseline_invariance") is not None:
        value = invariance["baseline_invariance"]
        status = value.get("status") if isinstance(value, dict) else value
        if isinstance(status, bool):
            status = "pass" if status else "fail"
        return {"status": status, "source": "downstream_ab_summary.shared_modis_invariance"}

    gate = (cf_summary.get("canonical_reproduction") or {}).get("baseline_semantic")
    if isinstance(gate, dict) and gate.get("status") is not None:
        return {"status": gate["status"],
                "source": "counterfactual_summary.canonical_reproduction.baseline_semantic"}

    return {"status": "unknown", "source": None}


def _prerequisite_failures(state: dict) -> list[str]:
    failures: list[str] = []
    checks = (
        ("counterfactual_final_status", REQUIRED_COUNTERFACTUAL_FINAL_STATUS),
        ("downstream_ab_final_status", REQUIRED_DOWNSTREAM_AB_FINAL_STATUS),
        ("residual_seam_final_status", REQUIRED_RESIDUAL_SEAM_FINAL_STATUS),
        ("downstream_ab_reference_reproduction", REQUIRED_AB_REFERENCE_REPRODUCTION),
        ("baseline_invariance", "pass"),
    )
    for key, required in checks:
        actual = state.get(key)
        if actual != required:
            failures.append(f"{key}={actual!r} (required {required!r})")
    return failures


def upstream_prerequisites_met(state: dict) -> bool:
    """Every required upstream status must match EXACTLY."""
    return not _prerequisite_failures(state)


def validate_upstream_state(state: dict) -> None:
    if not upstream_prerequisites_met(state):
        raise PrerequisiteError(
            "upstream prerequisites are not met; this experiment only follows a "
            "completed and valid counterfactual -> downstream A/B -> residual "
            "seam attribution chain. Failures:\n  "
            + "\n  ".join(_prerequisite_failures(state))
        )


# =============================================================================
# Input provenance
# =============================================================================
def raster_provenance_record(role: str, entry: dict) -> dict:
    """Path + hash + grid signature for one frozen raster input."""
    path = Path(entry["path"])
    record = OrderedDict((
        ("role", role),
        ("path", str(path)),
        ("source", entry["source"]),
        ("family", entry["family"]),
        ("required", bool(entry["required"])),
        ("present", path.exists()),
        ("purpose", entry["purpose"]),
        ("sha256", None),
        ("bytes", None),
        ("grid", None),
    ))
    if not path.exists():
        return record
    signed = sha256_and_size(path)
    record["sha256"] = signed["sha256"]
    record["bytes"] = signed["bytes"]
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            record["grid"] = grid_signature(path)
        except Exception:                                   # noqa: BLE001
            record["grid"] = None
    return record


def json_provenance_record(role: str, path: Path) -> dict:
    path = Path(path)
    record = OrderedDict((
        ("role", role), ("path", str(path)), ("present", path.exists()),
        ("sha256", None), ("bytes", None),
    ))
    if path.exists():
        signed = sha256_and_size(path)
        record["sha256"] = signed["sha256"]
        record["bytes"] = signed["bytes"]
    return record


def build_input_provenance(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Every frozen input, hashed, with the upstream statuses that gate it."""
    plan = build_input_plan(experiment_id, base_dir)
    upstream = upstream_report_paths(experiment_id, base_dir)
    pathrow = pathrow_boundary_sources(experiment_id, base_dir)
    state = load_upstream_state(experiment_id, base_dir)

    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("reference_composite", REFERENCE_COMPOSITE),
        ("candidate_composite", CANDIDATE_COMPOSITE),
        ("frozen_inputs", [raster_provenance_record(role, entry)
                           for role, entry in plan.items()]),
        ("upstream_reports", [json_provenance_record(role, path)
                              for role, path in upstream.items()]),
        ("pathrow_sources", [json_provenance_record(role, path)
                             for role, path in pathrow.items()]),
        ("missing_required_inputs", missing_required_inputs(plan)),
        ("missing_optional_inputs", missing_optional_inputs(plan)),
        ("upstream_state", state),
        ("baseline_recomputed", RECOMPUTES_BASELINE),
        ("baseline_re_exported", False),
        ("labels_used", USES_LABELS),
        ("step8_metrics_used", USES_STEP8_METRICS),
        ("model_performance_used", USES_MODEL_PERFORMANCE),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def assert_grid_contract(plan: "OrderedDict[str, dict]") -> dict:
    """Every present raster input must share ONE exact grid."""
    paths = [Path(entry["path"]) for entry in plan.values()
             if Path(entry["path"]).suffix.lower() in (".tif", ".tiff")
             and Path(entry["path"]).exists()]
    if not paths:
        raise PrerequisiteError("no raster inputs are present; cannot establish a grid")
    signature = assert_same_grid(paths)
    return OrderedDict((
        ("status", "pass"),
        ("raster_count", len(paths)),
        ("signature", signature),
    ))


# =============================================================================
# Frozen current-period window and source-scene inventory
# =============================================================================
def load_frozen_audit_config(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """The frozen counterfactual provenance: window, QA mask, Landsat scaling."""
    path = counterfactual_root(experiment_id, base_dir) / "audit_config.json"
    payload = _read_json(path)
    if payload is None:
        raise PrerequisiteError(
            f"frozen counterfactual provenance is missing or unreadable: {path}. "
            "The current-period window and scene inventory are NEVER re-derived "
            "from scratch by this experiment."
        )
    return payload


def frozen_current_window(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """The EXACT frozen current-period date window, semantics included.

    Earth Engine's `filterDate` end is exclusive; the frozen provenance records
    that behaviour and it is preserved unchanged. Nothing about the window is
    recomputed here.
    """
    config = load_frozen_audit_config(experiment_id, base_dir)
    current = config.get("current_period") or {}
    semantics = current.get("date_window_semantics") or {}
    window = OrderedDict((
        ("start_date", current.get("start_date")),
        ("end_date", current.get("end_date")),
        ("end_semantics", semantics.get("end_semantics", "exclusive")),
        ("effective_last_included_date", semantics.get("effective_last_included_date")),
        ("window_days", current.get("window_days")),
        ("months_filter", current.get("months_filter")),
        ("source_collection", config.get("source_collection")),
        ("landsat_scale", (config.get("step5_policy") or {}).get("landsat_scale")),
        ("landsat_offset", (config.get("step5_policy") or {}).get("landsat_offset")),
        ("qa_mask", config.get("qa_mask")),
        ("provenance_path",
         str(counterfactual_root(experiment_id, base_dir) / "audit_config.json")),
    ))
    if not window["start_date"] or not window["end_date"]:
        raise PrerequisiteError(
            "frozen counterfactual provenance does not record a current-period "
            "start/end date; refusing to invent one."
        )
    return window


def assert_frozen_scaling(window: dict) -> None:
    """The Landsat scaling MUST be the frozen one -- never silently re-derived."""
    scale, offset = window.get("landsat_scale"), window.get("landsat_offset")
    if scale is None or offset is None:
        raise PrerequisiteError(
            "frozen provenance does not record the Landsat scale/offset; "
            "refusing to assume one."
        )
    if not (math.isclose(float(scale), float(LANDSAT_SCALE), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(offset), float(LANDSAT_OFFSET), rel_tol=0.0, abs_tol=1e-12)):
        raise PrerequisiteError(
            f"frozen Landsat scaling ({scale}, {offset}) differs from the canonical "
            f"constants ({LANDSAT_SCALE}, {LANDSAT_OFFSET}); the experiment requires "
            "the EXACT same scaling as the reference composite."
        )


def read_frozen_scene_manifest(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> list[dict]:
    """Every row of the frozen scene manifest (READ-ONLY)."""
    path = counterfactual_root(experiment_id, base_dir) / "scene_manifest.csv"
    if not Path(path).exists():
        raise PrerequisiteError(
            f"frozen scene manifest is missing: {path}. The source-scene inventory "
            "is taken from the frozen counterfactual provenance and from nowhere else."
        )
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


CURRENT_LST_ROLE = "current_lst"


def current_scene_records(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> list[dict]:
    """The frozen current-period LST scenes ONLY, deterministically ordered."""
    rows = [r for r in read_frozen_scene_manifest(experiment_id, base_dir)
            if (r.get("input_role") or "").strip() == CURRENT_LST_ROLE]
    rows.sort(key=lambda r: (str(r.get("acquisition_date")), str(r.get("scene_id"))))
    return rows


def daily_date_inventory(records) -> "OrderedDict[str, dict]":
    """Group frozen current scenes into ONE entry per unique acquisition date.

    Same-day scenes are collapsed into a single temporal observation: they
    populate one date entry and are mosaicked by the frozen same-day rule. They
    are NEVER counted as separate temporal observations. Ordering is fully
    deterministic (dates sorted, scene ids sorted within a date).
    """
    grouped: "OrderedDict[str, dict]" = OrderedDict()
    for record in sorted(records, key=lambda r: (str(r.get("acquisition_date")),
                                                 str(r.get("scene_id")))):
        date = str(record.get("acquisition_date") or "").strip()
        if not date:
            raise HarmonizationError(
                f"frozen scene record without an acquisition date: {record.get('scene_id')!r}"
            )
        entry = grouped.setdefault(date, OrderedDict((
            ("acquisition_date", date),
            ("scene_ids", []),
            ("landsat_product_ids", []),
            ("path_rows", []),
            ("wrs_paths", []),
            ("acquisition_datetimes", []),
            ("scene_count", 0),
        )))
        entry["scene_ids"].append(str(record.get("scene_id")))
        entry["landsat_product_ids"].append(str(record.get("landsat_product_id")))
        path, row = str(record.get("wrs_path")), str(record.get("wrs_row"))
        entry["path_rows"].append(f"{path}_{row}")
        if path not in entry["wrs_paths"]:
            entry["wrs_paths"].append(path)
        entry["acquisition_datetimes"].append(str(record.get("acquisition_datetime")))
        entry["scene_count"] += 1

    for entry in grouped.values():
        entry["scene_ids"] = sorted(entry["scene_ids"])
        entry["landsat_product_ids"] = sorted(entry["landsat_product_ids"])
        entry["path_rows"] = sorted(set(entry["path_rows"]))
        entry["wrs_paths"] = sorted(set(entry["wrs_paths"]))
        entry["same_day_scene_count"] = entry["scene_count"]
        entry["temporal_observations"] = 1
    return grouped


def assert_dates_within_frozen_window(dates, window: dict) -> None:
    """Every retained date must lie inside the EXACT frozen window."""
    start = str(window["start_date"])
    end = str(window["end_date"])
    exclusive = str(window.get("end_semantics", "exclusive")) == "exclusive"
    for date in dates:
        if str(date) < start:
            raise HarmonizationError(
                f"acquisition date {date} precedes the frozen window start {start}"
            )
        if exclusive and str(date) >= end:
            raise HarmonizationError(
                f"acquisition date {date} is not inside the frozen half-open window "
                f"[{start}, {end}) recorded in the counterfactual provenance"
            )
        if not exclusive and str(date) > end:
            raise HarmonizationError(
                f"acquisition date {date} is after the frozen window end {end}"
            )


def daily_raster_filename(date: str, *, kind: str) -> str:
    """`kind` is `reference` (Y_d) or `harmonized` (Y'_d)."""
    if kind not in ("reference", "harmonized"):
        raise HarmonizationError(f"unknown daily raster kind: {kind!r}")
    return f"current_lst_daily_{date}_{kind}.tif"


def daily_raster_path(root: Path, date: str, *, kind: str) -> Path:
    return Path(root) / "daily" / kind / daily_raster_filename(date, kind=kind)


#: The runner path, kept next to the command text so the two cannot drift.
RUNNER_SCRIPT = "scripts/run_landsat_current_support_harmonization.py"


def daily_fetch_command(experiment_id: str) -> str:
    """The exact command the USER runs later to fetch the daily mosaics.

    Fetching is the isolated first stage of the live run, so there is exactly
    ONE command rather than a separate download entry point that could drift
    from the experiment's own frozen scene inventory.
    """
    return (
        f"python {RUNNER_SCRIPT} \\\n"
        f"    --experiment {experiment_id} \\\n"
        f"    --run"
    )


def build_daily_export_plan(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """A pure-data plan for the daily current-period mosaics.

    This function performs NO Earth Engine work: it neither imports, initialises,
    authenticates nor calls Earth Engine. It states exactly which daily mosaics
    are required, which already exist locally, and what the export contract is.
    The runner's isolated export stage is the only place a live Earth Engine
    operation may ever occur, and only under `--run`.
    """
    root = diagnostic_output_root(experiment_id, base_dir)
    window = frozen_current_window(experiment_id, base_dir)
    assert_frozen_scaling(window)
    inventory = daily_date_inventory(current_scene_records(experiment_id, base_dir))
    assert_dates_within_frozen_window(inventory.keys(), window)

    items = []
    for date, entry in inventory.items():
        path = daily_raster_path(root, date, kind="reference")
        present = path.exists()
        items.append(OrderedDict((
            ("acquisition_date", date),
            ("output_path", str(path)),
            ("planned_download_path", str(path)),
            ("present_locally", present),
            ("verified_sha256", sha256_and_size(path)["sha256"] if present else None),
            ("scene_count", entry["scene_count"]),
            ("scene_ids", list(entry["scene_ids"])),
            ("landsat_product_ids", list(entry["landsat_product_ids"])),
            ("path_rows", list(entry["path_rows"])),
            ("acquisition_datetimes", list(entry["acquisition_datetimes"])),
            ("temporal_observations", 1),
        )))

    missing = [i["acquisition_date"] for i in items if not i["present_locally"]]
    return OrderedDict((
        ("experiment_id", experiment_id),
        ("date_count", len(inventory)),
        ("scene_count", sum(e["scene_count"] for e in inventory.values())),
        ("current_window", window),
        ("items", items),
        ("missing_locally", missing),
        #: The single question the dry-run must answer out loud.
        ("complete_daily_mosaics_present", bool(items) and not missing),
        ("daily_mosaic_status",
         "complete" if (items and not missing)
         else ("none_present" if len(missing) == len(items) else "partial")),
        ("required_dates", [i["acquisition_date"] for i in items]),
        ("planned_download_root", str(root / "daily" / "reference")),
        ("planned_download_paths", [i["planned_download_path"] for i in items]),
        #: The command the USER runs later to fetch them. The agent never runs it.
        ("fetch_command", daily_fetch_command(experiment_id)),
        ("fetch_command_note",
         "The daily mosaics are fetched by the isolated FIRST stage of the live "
         "run: it exports only the missing diagnostic daily current-period "
         "rasters listed above, then every remaining stage executes under an "
         "Earth Engine guard. This command must be executed by the user; the "
         "agent never runs it. Add --no-earth-engine to assert the mosaics are "
         "already present and forbid the export stage outright."),
        ("export_contract", OrderedDict((
            ("source_collection", window["source_collection"]),
            ("date_window", f"[{window['start_date']}, {window['end_date']})"),
            ("qa_mask", "frozen counterfactual QA_PIXEL mask, reused verbatim"),
            ("same_day_rule",
             "same-date scenes are mosaicked by the frozen same-day SPATIAL "
             "median reducer into ONE daily image; they are never separate "
             "temporal observations"),
            ("scaling",
             f"ST_B10 * {window['landsat_scale']} + {window['landsat_offset']} "
             "- 273.15 -> Celsius, applied per daily mosaic"),
            ("units", "celsius"),
            ("exports_only", "current-period diagnostic daily rasters"),
            ("never_exports", "any baseline raster, any production raster"),
            ("transport", "existing direct/tiled safe download infrastructure"),
            ("per_file_provenance_required", ["scene_ids", "acquisition_date", "sha256"]),
        ))),
        ("earth_engine_touched_by_this_function", False),
    ))


# =============================================================================
# Raster IO helpers (bounded memory; nodata is NEVER replaced with zero)
# =============================================================================
def reference_grid_path(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """The raster whose grid every product must match exactly."""
    return build_input_plan(experiment_id, base_dir)["frozen_reference_current_lst_celsius"]["path"]


def raster_shape(path: Path) -> tuple[int, int]:
    import rasterio

    with rasterio.open(path) as src:
        return int(src.height), int(src.width)


def output_profile_from(path: Path) -> dict:
    """The canonical Step5 float32 output profile, taken from the frozen grid."""
    import rasterio

    from src.step5_preprocess_timeseries import output_profile

    with rasterio.open(path) as src:
        return output_profile(src.profile.copy())


class WindowedWriter:
    """Atomic windowed float32 GeoTIFF writer (temp file + os.replace)."""

    def __init__(self, path: Path, profile: dict) -> None:
        self.path = Path(path)
        self.tmp = self.path.parent / f".{self.path.name}.tmp"
        self.profile = profile
        self._dst = None

    def __enter__(self) -> "WindowedWriter":
        import rasterio

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._dst = rasterio.open(self.tmp, "w", **self.profile)
        return self

    def write(self, array, start: int) -> None:
        import numpy as np
        from rasterio.windows import Window

        array = np.asarray(array, dtype="float32")
        window = Window(0, int(start), int(array.shape[1]), int(array.shape[0]))
        self._dst.write(array, 1, window=window)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._dst is not None:
            self._dst.close()
            self._dst = None
        if exc_type is None:
            os.replace(str(self.tmp), str(self.path))
        elif self.tmp.exists():
            self.tmp.unlink()
        return False


def mask_physical_celsius(array):
    """Canonical Step5 physical LST guard, reused verbatim (never widened)."""
    import numpy as np

    from src.step5_preprocess_timeseries import mask_physical_celsius as canonical

    return canonical(np.asarray(array, dtype="float32")).astype("float64")


def apply_step5_current_policy(median_values, valid_count, thresholds: dict):
    """Exactly the canonical Step5 current-composite guard.

    `mask_physical_celsius` first, then the minimum-current-support mask. This
    is the SAME order Step5 applies, and it is applied identically to the
    reference reproduction and to the candidate so the two differ by the
    intervention alone.
    """
    import numpy as np

    guarded = mask_physical_celsius(median_values)
    enough = np.asarray(valid_count, dtype="float64") >= float(
        thresholds["min_current_valid_count"]
    )
    return np.where(enough, guarded, np.nan)


def build_current_minus_baseline(current, baseline_mean):
    """D = C - M, with the FROZEN baseline mean. No baseline is recomputed."""
    import numpy as np

    return np.asarray(current, dtype="float64") - np.asarray(baseline_mean, dtype="float64")


def build_anomaly_zscore(current, baseline_mean, baseline_std, valid_count, thresholds: dict):
    """Z = (C - M) / S under the EXACT canonical Step5 anomaly mask.

    The mask is `enough_current AND baseline_std >= min_baseline_std`. Both
    operands come from frozen rasters that this experiment never modifies, and
    the current support is invariant by construction, so the anomaly valid mask
    is identical between reference and candidate -- which the support invariance
    gate then verifies rather than assumes.
    """
    import numpy as np

    current = np.asarray(current, dtype="float64")
    mean = np.asarray(baseline_mean, dtype="float64")
    std = np.asarray(baseline_std, dtype="float64")
    enough = np.asarray(valid_count, dtype="float64") >= float(
        thresholds["min_current_valid_count"]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(enough & (std >= float(thresholds["min_baseline_std_celsius"])),
                     (current - mean) / std, np.nan)
    return np.where(np.isfinite(z), z, np.nan)


def nanmedian_over_dates(stack):
    """Per-pixel median over the valid dates of a (n_dates, rows, cols) stack.

    Pixels with no valid date stay NaN. Nothing is zero-filled, and no spatial
    neighbourhood is ever consulted -- this is a purely temporal reduction.
    """
    import numpy as np

    import warnings

    stack = np.asarray(stack, dtype="float64")
    valid = np.isfinite(stack)
    count = valid.sum(axis=0).astype("float64")
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # A pixel with no valid date is EXPECTED and stays NaN; the all-NaN
        # slice warning would be pure noise on every window.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
    return np.where(count > 0, median, np.nan), count


def date_membership_bitmask(stack):
    """Per-pixel bitmask of WHICH dates are valid (date order = stack order).

    Equality of this raster is a strictly stronger statement than equality of
    the unique-date COUNT: it proves the candidate composited exactly the same
    acquisition dates at every pixel, not merely the same number of them.
    """
    import numpy as np

    stack = np.asarray(stack, dtype="float64")
    if stack.shape[0] > 53:
        raise HarmonizationError(
            f"{stack.shape[0]} dates exceed the exact float64 bitmask capacity"
        )
    mask = np.zeros(stack.shape[1:], dtype="int64")
    for index in range(stack.shape[0]):
        mask |= (np.isfinite(stack[index]).astype("int64") << index)
    return mask.astype("float64")


def read_daily_stack(paths, start: int, stop: int):
    """Read one row window of every daily mosaic as a float64 stack."""
    import numpy as np

    arrays = [read_window(Path(p), start, stop) for p in paths]
    return np.stack(arrays, axis=0)


# =============================================================================
# B. Date-overlap graph
# =============================================================================
def block_grid_ids(rows: int, cols: int, row_offset: int, *,
                   block_size: int = GRAPH_BLOCK_SIZE_CELLS):
    """Block id of every cell of a window, on the SAME lattice as the bootstrap."""
    import numpy as np

    r = (np.arange(rows, dtype="int64") + int(row_offset)) // int(block_size)
    c = np.arange(cols, dtype="int64") // int(block_size)
    return r[:, None] * 100000 + c[None, :]


def accumulate_pair_block_medians(stack, block_ids, store, *,
                                  min_block_pixels: int = MIN_BLOCK_COMMON_PIXELS):
    """Add this window's per-block medians of `Y_j - Y_i` for every date pair.

    Only pixels valid on BOTH dates contribute -- an edge is never computed from
    a pixel that one of the two dates does not see, and a missing value is never
    replaced by zero. Because windows are aligned to the block lattice, a block
    is completed inside exactly one window, so a block median is computed once
    from all of its pixels.
    """
    import numpy as np

    n_dates = stack.shape[0]
    flat_blocks = block_ids.ravel()
    unique_blocks, inverse = np.unique(flat_blocks, return_inverse=True)

    for i in range(n_dates):
        for j in range(i + 1, n_dates):
            diff = (stack[j] - stack[i]).ravel()
            finite = np.isfinite(diff)
            if not finite.any():
                continue
            key = (i, j)
            entry = store.setdefault(key, OrderedDict((
                ("date_i_index", i), ("date_j_index", j),
                ("block_medians", []), ("block_labels", []),
                ("block_pixel_counts", []),
                ("common_valid_pixels", 0),
                ("blocks_seen", 0),
                ("blocks_below_min_pixels", 0),
            )))
            entry["common_valid_pixels"] += int(finite.sum())

            order = np.argsort(inverse[finite], kind="stable")
            groups = inverse[finite][order]
            values = diff[finite][order]
            boundaries = np.flatnonzero(np.diff(groups)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [groups.size]))
            for s, e in zip(starts, ends):
                count = int(e - s)
                entry["blocks_seen"] += 1
                if count < int(min_block_pixels):
                    entry["blocks_below_min_pixels"] += 1
                    continue
                entry["block_medians"].append(float(np.median(values[s:e])))
                entry["block_labels"].append(
                    block_id_to_label(int(unique_blocks[groups[s]]))
                )
                entry["block_pixel_counts"].append(count)
    return store


def robust_edge_estimate(block_medians) -> dict:
    """Edge difference = MEDIAN of the eligible block medians, plus dispersion.

    Reporting the median of block medians (rather than the pooled pixel median)
    is what makes the edge estimate robust to a single large, spatially
    concentrated feature: every eligible block contributes exactly one vote.
    """
    import numpy as np

    values = np.asarray(block_medians, dtype="float64")
    values = values[np.isfinite(values)]
    result = OrderedDict((
        ("n_blocks", int(values.size)),
        ("median_difference_celsius", None),
        ("mad_celsius", None),
        ("sigma_celsius", None),
        ("standard_error_celsius", None),
        ("standardised_residual_measure", None),
        ("block_median_min", None),
        ("block_median_max", None),
    ))
    if not values.size:
        return result
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = max(MAD_TO_SIGMA * mad, MIN_EDGE_SIGMA_CELSIUS)
    result["median_difference_celsius"] = median
    result["mad_celsius"] = mad
    result["sigma_celsius"] = float(sigma)
    result["standard_error_celsius"] = float(sigma / math.sqrt(values.size))
    # Scale-free spread of the block votes about the edge estimate: how many
    # robust sigmas the typical block sits from the edge median.
    result["standardised_residual_measure"] = (
        float(np.median(np.abs(values - median)) / sigma) if sigma > 0 else None
    )
    result["block_median_min"] = float(values.min())
    result["block_median_max"] = float(values.max())
    return result


def build_overlap_graph(dates, store, *, min_common_pixels: int,
                        min_independent_blocks: int,
                        date_entries=None, grid_cells: int | None = None) -> dict:
    """Assemble the date-overlap graph at ONE predeclared eligibility setting.

    `store` holds every candidate pair's block medians and is computed ONCE, so
    the primary and sensitivity graphs are different eligibility *filters* over
    identical overlap evidence rather than different measurements.
    """
    edges = []
    rejected = []
    for (i, j), entry in sorted(store.items()):
        estimate = robust_edge_estimate(entry["block_medians"])
        eligible = (
            entry["common_valid_pixels"] >= int(min_common_pixels)
            and estimate["n_blocks"] >= int(min_independent_blocks)
        )
        coverage = (
            float(entry["common_valid_pixels"]) / float(grid_cells)
            if grid_cells else None
        )
        row = OrderedDict((
            ("date_i", dates[i]),
            ("date_j", dates[j]),
            ("date_i_index", i),
            ("date_j_index", j),
            ("common_valid_pixels", int(entry["common_valid_pixels"])),
            ("independent_blocks", estimate["n_blocks"]),
            ("blocks_seen", int(entry["blocks_seen"])),
            ("blocks_below_min_pixels", int(entry["blocks_below_min_pixels"])),
            ("edge_median_difference_celsius", estimate["median_difference_celsius"]),
            ("edge_mad_celsius", estimate["mad_celsius"]),
            ("edge_sigma_celsius", estimate["sigma_celsius"]),
            ("edge_standard_error_celsius", estimate["standard_error_celsius"]),
            ("edge_standardised_residual_measure",
             estimate["standardised_residual_measure"]),
            ("block_median_min", estimate["block_median_min"]),
            ("block_median_max", estimate["block_median_max"]),
            ("spatial_coverage_fraction", coverage),
            ("eligible", bool(eligible)),
        ))
        if date_entries is not None:
            row["date_i_path_rows"] = ",".join(date_entries[dates[i]]["path_rows"])
            row["date_j_path_rows"] = ",".join(date_entries[dates[j]]["path_rows"])
            row["shares_wrs_path"] = bool(
                set(date_entries[dates[i]]["wrs_paths"])
                & set(date_entries[dates[j]]["wrs_paths"])
            )
        else:
            row["date_i_path_rows"] = None
            row["date_j_path_rows"] = None
            row["shares_wrs_path"] = None
        (edges if eligible else rejected).append(row)

    return OrderedDict((
        ("min_common_pixels", int(min_common_pixels)),
        ("min_independent_blocks", int(min_independent_blocks)),
        ("min_block_common_pixels", int(MIN_BLOCK_COMMON_PIXELS)),
        ("block_size_cells", int(GRAPH_BLOCK_SIZE_CELLS)),
        ("date_count", len(dates)),
        ("candidate_pair_count", len(store)),
        ("edge_count", len(edges)),
        ("rejected_edge_count", len(rejected)),
        ("edges", edges),
        ("rejected_edges", rejected),
    ))


def graph_adjacency(dates, edges) -> "OrderedDict[str, list]":
    adjacency: "OrderedDict[str, list]" = OrderedDict((d, []) for d in dates)
    for edge in edges:
        adjacency[edge["date_i"]].append(edge["date_j"])
        adjacency[edge["date_j"]].append(edge["date_i"])
    for key in adjacency:
        adjacency[key] = sorted(adjacency[key])
    return adjacency


def connected_components(dates, edges) -> list[list[str]]:
    """Deterministic connected components (sorted nodes, sorted components)."""
    adjacency = graph_adjacency(dates, edges)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in dates:
        if node in seen:
            continue
        queue, component = deque([node]), []
        seen.add(node)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(sorted(component))
    components.sort(key=lambda c: (-len(c), c[0]))
    return components


def graph_is_connected(dates, edges) -> bool:
    """One component containing EVERY retained date -- no date is dropped."""
    if not dates:
        return False
    return len(connected_components(dates, edges)) == 1


def articulation_nodes(dates, edges) -> list[str]:
    """Dates whose removal would disconnect the graph (iterative Hopcroft-Tarjan)."""
    adjacency = graph_adjacency(dates, edges)
    index_of = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    discovery = [-1] * n
    low = [0] * n
    parent = [-1] * n
    is_articulation = [False] * n
    timer = 0

    for start in range(n):
        if discovery[start] != -1:
            continue
        root_children = 0
        stack = [(start, iter(adjacency[dates[start]]))]
        discovery[start] = low[start] = timer
        timer += 1
        while stack:
            node, iterator = stack[-1]
            advanced = False
            for neighbour_name in iterator:
                neighbour = index_of[neighbour_name]
                if discovery[neighbour] == -1:
                    parent[neighbour] = node
                    discovery[neighbour] = low[neighbour] = timer
                    timer += 1
                    if node == start:
                        root_children += 1
                    stack.append((neighbour, iter(adjacency[neighbour_name])))
                    advanced = True
                    break
                if neighbour != parent[node]:
                    low[node] = min(low[node], discovery[neighbour])
            if advanced:
                continue
            stack.pop()
            if stack:
                up = stack[-1][0]
                low[up] = min(low[up], low[node])
                if parent[node] != -1 and up != start and low[node] >= discovery[up]:
                    is_articulation[up] = True
        if root_children > 1:
            is_articulation[start] = True
    return [dates[i] for i in range(n) if is_articulation[i]]


def spanning_tree_edges(dates, edges):
    """A deterministic BFS spanning forest; returns (tree_edges, non_tree_edges)."""
    by_pair = {(e["date_i"], e["date_j"]): e for e in edges}
    adjacency = graph_adjacency(dates, edges)
    seen: set[str] = set()
    tree, tree_keys = [], set()
    for root in dates:
        if root in seen:
            continue
        seen.add(root)
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                key = (node, neighbour) if (node, neighbour) in by_pair else (neighbour, node)
                tree.append(by_pair[key])
                tree_keys.add(key)
                queue.append(neighbour)
    non_tree = [e for e in edges if (e["date_i"], e["date_j"]) not in tree_keys]
    return tree, non_tree


def cycle_consistency(dates, edges) -> dict:
    """Fundamental-cycle closure of the raw overlap differences.

    For a perfectly additive date-offset world every cycle of edge differences
    sums to zero. A large closure error means the additive model is not
    sufficient -- it is reported, never silently absorbed.
    """
    import numpy as np

    tree, non_tree = spanning_tree_edges(dates, edges)
    tree_adjacency: dict[str, list[tuple[str, float]]] = {d: [] for d in dates}
    for edge in tree:
        delta = float(edge["edge_median_difference_celsius"])
        tree_adjacency[edge["date_i"]].append((edge["date_j"], delta))
        tree_adjacency[edge["date_j"]].append((edge["date_i"], -delta))

    def tree_path_sum(source: str, target: str):
        """Sum of `alpha_next - alpha_current` along the unique tree path."""
        seen = {source}
        queue = deque([(source, 0.0)])
        while queue:
            node, total = queue.popleft()
            if node == target:
                return total
            for neighbour, delta in tree_adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append((neighbour, total + delta))
        return None

    closures = []
    for edge in non_tree:
        path = tree_path_sum(edge["date_i"], edge["date_j"])
        if path is None:
            continue
        closures.append(OrderedDict((
            ("date_i", edge["date_i"]), ("date_j", edge["date_j"]),
            ("edge_difference_celsius", float(edge["edge_median_difference_celsius"])),
            ("tree_path_difference_celsius", float(path)),
            ("closure_error_celsius",
             float(edge["edge_median_difference_celsius"]) - float(path)),
        )))
    values = np.array([abs(c["closure_error_celsius"]) for c in closures], dtype="float64")
    return OrderedDict((
        ("independent_cycle_count", len(closures)),
        ("tree_edge_count", len(tree)),
        ("non_tree_edge_count", len(non_tree)),
        ("max_abs_closure_error_celsius", float(values.max()) if values.size else None),
        ("median_abs_closure_error_celsius", float(np.median(values)) if values.size else None),
        ("cycles", closures),
    ))


def build_graph_diagnostics(dates, graph: dict, date_entries=None) -> dict:
    """Everything the connectivity gate must report BEFORE offsets are solved."""
    edges = graph["edges"]
    components = connected_components(dates, edges)
    adjacency = graph_adjacency(dates, edges)
    observation_counts = {
        d: float((date_entries or {}).get(d, {}).get("valid_pixel_count") or 0.0)
        for d in dates
    }
    total_observations = sum(observation_counts.values())

    component_rows = []
    for index, component in enumerate(components):
        represented = sum(observation_counts[d] for d in component)
        component_rows.append(OrderedDict((
            ("component_index", index),
            ("date_count", len(component)),
            ("dates", list(component)),
            ("valid_observation_count", represented),
            ("valid_observation_fraction",
             (represented / total_observations) if total_observations else None),
        )))

    return OrderedDict((
        ("date_count", len(dates)),
        ("dates", list(dates)),
        ("edge_count", len(edges)),
        ("rejected_edge_count", graph["rejected_edge_count"]),
        ("connected_component_count", len(components)),
        ("connected", len(components) == 1 and bool(dates)),
        ("components", component_rows),
        ("degree_per_date", OrderedDict((d, len(adjacency[d])) for d in dates)),
        ("isolated_dates", [d for d in dates if not adjacency[d]]),
        ("articulation_nodes", articulation_nodes(dates, edges)),
        ("cycle_consistency", cycle_consistency(dates, edges)),
        ("dates_dropped", []),
        ("drop_policy",
         "No date is ever silently dropped. Every current-period date retained "
         "by the frozen inventory must belong to the single connected "
         "component, otherwise the experiment ends at "
         "insufficient_date_overlap_graph."),
    ))


# =============================================================================
# D. Offset solution
# =============================================================================
def edge_weights(edges) -> dict:
    """Overlap-only edge weights, capped so no single pair can dominate."""
    import numpy as np

    raw = np.array([
        float(e["independent_blocks"]) / float(e["edge_sigma_celsius"]) ** 2
        for e in edges
    ], dtype="float64") if edges else np.empty(0, dtype="float64")
    if not raw.size:
        return OrderedDict((("raw", []), ("capped", []), ("cap", None),
                            ("median_raw", None), ("capped_edge_count", 0)))
    median_raw = float(np.median(raw))
    cap = float(WEIGHT_CAP_MULTIPLE * median_raw)
    capped = np.minimum(raw, cap)
    return OrderedDict((
        ("raw", [float(v) for v in raw]),
        ("capped", [float(v) for v in capped]),
        ("cap", cap),
        ("median_raw", median_raw),
        ("capped_edge_count", int((raw > cap).sum())),
        ("formula", EDGE_WEIGHT_FORMULA),
    ))


def solve_date_offsets(dates, edges, date_observation_counts) -> dict:
    """Deterministic weighted least squares for the additive date offsets.

    Minimises `sum_ij w_ij ((alpha_j - alpha_i) - delta_ij)^2` subject to the
    predeclared identifying constraint `sum_d n_d alpha_d = 0`. The normal
    equations are the weighted graph Laplacian, which is singular by exactly one
    dimension on a connected graph; the constraint is imposed exactly with a
    Lagrange multiplier rather than by anchoring an arbitrary date to zero.
    """
    import numpy as np

    n = len(dates)
    index_of = {d: i for i, d in enumerate(dates)}
    weights = edge_weights(edges)

    laplacian = np.zeros((n, n), dtype="float64")
    rhs = np.zeros(n, dtype="float64")
    for edge, weight in zip(edges, weights["capped"]):
        i, j = index_of[edge["date_i"]], index_of[edge["date_j"]]
        delta = float(edge["edge_median_difference_celsius"])
        laplacian[i, i] += weight
        laplacian[j, j] += weight
        laplacian[i, j] -= weight
        laplacian[j, i] -= weight
        rhs[i] -= weight * delta
        rhs[j] += weight * delta

    constraint = np.array([float(date_observation_counts.get(d, 0.0)) for d in dates],
                          dtype="float64")
    if not np.any(constraint > 0):
        raise HarmonizationError(
            "every daily mosaic reports zero valid observations; the identifying "
            "constraint is undefined"
        )

    augmented = np.zeros((n + 1, n + 1), dtype="float64")
    augmented[:n, :n] = laplacian
    augmented[:n, n] = constraint
    augmented[n, :n] = constraint
    target = np.zeros(n + 1, dtype="float64")
    target[:n] = rhs

    condition_number = float(np.linalg.cond(augmented))
    solver_status = "solved"
    try:
        solution = np.linalg.solve(augmented, target)
    except np.linalg.LinAlgError as error:                  # pragma: no cover
        raise HarmonizationError(
            f"the weighted least-squares system is singular ({error}); the "
            "date-overlap graph does not identify the offsets"
        ) from error
    alpha = solution[:n]

    residuals = []
    weighted_sse = 0.0
    for edge, weight in zip(edges, weights["capped"]):
        i, j = index_of[edge["date_i"]], index_of[edge["date_j"]]
        delta = float(edge["edge_median_difference_celsius"])
        residual = float((alpha[j] - alpha[i]) - delta)
        weighted_sse += weight * residual * residual
        residuals.append(OrderedDict((
            ("date_i", edge["date_i"]), ("date_j", edge["date_j"]),
            ("observed_difference_celsius", delta),
            ("fitted_difference_celsius", float(alpha[j] - alpha[i])),
            ("residual_celsius", residual),
            ("weight", float(weight)),
            ("independent_blocks", int(edge["independent_blocks"])),
        )))

    dof = len(edges) - (n - 1)
    residual_variance = (weighted_sse / dof) if dof > 0 else None
    try:
        inverse = np.linalg.inv(augmented)[:n, :n]
        if residual_variance is not None:
            variances = np.clip(np.diag(inverse) * residual_variance, 0.0, None)
            standard_errors = [float(math.sqrt(v)) for v in variances]
        else:
            standard_errors = [None] * n
    except np.linalg.LinAlgError:                           # pragma: no cover
        standard_errors = [None] * n

    residual_values = np.array([r["residual_celsius"] for r in residuals], dtype="float64")
    residual_rms = float(math.sqrt(float((residual_values ** 2).mean()))) if residual_values.size else 0.0

    weighted_mean = float((constraint * alpha).sum() / constraint.sum())
    unstable_reasons = []
    if condition_number > MAX_GRAPH_CONDITION_NUMBER:
        unstable_reasons.append(
            f"graph condition number {condition_number:.3e} > {MAX_GRAPH_CONDITION_NUMBER:.0e}"
        )
    if residual_rms > MAX_EDGE_RESIDUAL_RMS_CELSIUS:
        unstable_reasons.append(
            f"edge residual RMS {residual_rms:.4f} C > {MAX_EDGE_RESIDUAL_RMS_CELSIUS} C"
        )

    offsets = []
    for i, date in enumerate(dates):
        offsets.append(OrderedDict((
            ("acquisition_date", date),
            ("alpha_celsius", float(alpha[i])),
            ("standard_error_celsius", standard_errors[i]),
            ("valid_observation_count", float(constraint[i])),
            ("graph_degree", sum(1 for e in edges
                                 if date in (e["date_i"], e["date_j"]))),
        )))

    absolute = np.abs(alpha)
    return OrderedDict((
        ("solver", "deterministic weighted least squares (LAPACK gesv) on the "
                   "constraint-augmented weighted graph Laplacian"),
        ("solver_status", solver_status),
        ("identifying_constraint", IDENTIFYING_CONSTRAINT),
        ("offsets", offsets),
        ("alpha_by_date", OrderedDict((d, float(alpha[i])) for i, d in enumerate(dates))),
        ("max_abs_offset_celsius", float(absolute.max()) if absolute.size else None),
        ("median_abs_offset_celsius", float(np.median(absolute)) if absolute.size else None),
        ("weighted_mean_offset_celsius", weighted_mean),
        ("weighted_mean_offset_is_zero", bool(abs(weighted_mean) <= 1e-9)),
        ("edge_residuals", residuals),
        ("edge_residual_rms_celsius", residual_rms),
        ("edge_residual_max_abs_celsius",
         float(np.abs(residual_values).max()) if residual_values.size else None),
        ("weighted_residual_sum_of_squares", float(weighted_sse)),
        ("degrees_of_freedom", int(dof)),
        ("residual_variance", residual_variance),
        ("graph_condition_number", condition_number),
        ("edge_weights", weights),
        ("estimation_stable", not unstable_reasons),
        ("instability_reasons", unstable_reasons),
    ))


# =============================================================================
# Support invariance gate
# =============================================================================
SUPPORT_INVARIANCE_CHECKS = (
    "unique_date_valid_count",
    "current_valid_count",
    "valid_mask",
    "daily_membership_per_pixel",
    "anomaly_valid_mask",
    "low_current_count_mask",
)


class ExactComparisonAccumulator:
    """Streaming exact equality evidence for one support raster.

    Deliberately reports COUNTS rather than a boolean: the gate needs to say how
    many pixels disagreed and by how much, not merely that something did.
    """

    __slots__ = ("name", "equal", "unequal", "max_abs_difference",
                 "reference_valid", "candidate_valid", "mask_equal",
                 "mask_unequal", "changed_valid_pixels", "total")

    def __init__(self, name: str) -> None:
        self.name = name
        self.equal = 0
        self.unequal = 0
        self.max_abs_difference = 0.0
        self.reference_valid = 0
        self.candidate_valid = 0
        self.mask_equal = 0
        self.mask_unequal = 0
        self.changed_valid_pixels = 0
        self.total = 0

    def add(self, reference, candidate) -> None:
        import numpy as np

        reference = np.asarray(reference, dtype="float64")
        candidate = np.asarray(candidate, dtype="float64")
        ref_valid = np.isfinite(reference)
        cand_valid = np.isfinite(candidate)

        self.total += int(reference.size)
        self.reference_valid += int(ref_valid.sum())
        self.candidate_valid += int(cand_valid.sum())
        same_mask = ref_valid == cand_valid
        self.mask_equal += int(same_mask.sum())
        self.mask_unequal += int((~same_mask).sum())
        self.changed_valid_pixels += int((ref_valid != cand_valid).sum())

        both = ref_valid & cand_valid
        # NaN == NaN is False, so pixels invalid on BOTH sides are counted as
        # equal explicitly rather than being silently treated as a difference.
        neither = (~ref_valid) & (~cand_valid)
        if both.any():
            difference = np.abs(reference[both] - candidate[both])
            equal = difference == 0.0
            self.equal += int(equal.sum()) + int(neither.sum())
            self.unequal += int((~equal).sum())
            if difference.size:
                self.max_abs_difference = max(self.max_abs_difference,
                                              float(difference.max()))
        else:
            self.equal += int(neither.sum())
        self.unequal += int((~same_mask).sum())

    def report(self) -> dict:
        agreement = (self.mask_equal / self.total) if self.total else None
        return OrderedDict((
            ("check", self.name),
            ("total_pixels", self.total),
            ("exact_equal_pixel_count", self.equal),
            ("unequal_pixel_count", self.unequal),
            ("max_count_difference", self.max_abs_difference),
            ("reference_valid_pixels", self.reference_valid),
            ("candidate_valid_pixels", self.candidate_valid),
            ("changed_valid_pixel_count", self.changed_valid_pixels),
            ("mask_agreement", agreement),
            ("passes", bool(self.unequal == 0 and self.changed_valid_pixels == 0
                            and agreement == 1.0)),
        ))


def support_invariance_verdict(reports) -> dict:
    """The gate: every check must be EXACTLY invariant."""
    rows = list(reports)
    failures = [r["check"] for r in rows if not r["passes"]]
    return OrderedDict((
        ("checks", rows),
        ("required", OrderedDict((
            ("unequal_pixel_count", 0),
            ("changed_valid_pixel_count", 0),
            ("mask_agreement", 1.0),
        ))),
        ("failed_checks", failures),
        ("passes", not failures),
        ("purpose",
         "This gate prevents an apparent seam improvement produced by masking, "
         "dropping or re-selecting difficult pixels. The candidate must "
         "composite exactly the same acquisition dates at exactly the same "
         "pixels as the reference."),
    ))


# =============================================================================
# Raster-level evaluation
# =============================================================================
class SignedHistogramAccumulator:
    """Symmetric fixed-edge histogram for deterministic bounded-memory medians."""

    __slots__ = ("edges", "counts", "under", "over", "total")

    def __init__(self, maximum: float, bins: int = HISTOGRAM_BINS) -> None:
        import numpy as np

        self.edges = np.linspace(-float(maximum), float(maximum), int(bins) + 1)
        self.counts = np.zeros(int(bins), dtype="int64")
        self.under = 0
        self.over = 0
        self.total = 0

    def add(self, values) -> None:
        import numpy as np

        values = np.asarray(values, dtype="float64")
        values = values[np.isfinite(values)]
        if not values.size:
            return
        self.total += int(values.size)
        under = values < self.edges[0]
        over = values > self.edges[-1]
        self.under += int(under.sum())
        self.over += int(over.sum())
        inside = values[~(under | over)]
        if inside.size:
            self.counts += np.histogram(inside, bins=self.edges)[0]

    def quantile(self, percentile: float) -> float | None:
        import numpy as np

        if not self.total:
            return None
        target = (float(percentile) / 100.0) * self.total
        cumulative = np.cumsum(self.counts) + self.under
        if target <= self.under:
            return float(self.edges[0])
        if target > cumulative[-1]:
            return float(self.edges[-1])
        index = int(np.searchsorted(cumulative, target, side="left"))
        index = min(index, self.counts.size - 1)
        return float(0.5 * (self.edges[index] + self.edges[index + 1]))

    def describe(self) -> dict:
        return OrderedDict((
            ("bins", int(self.counts.size)),
            ("range", [float(self.edges[0]), float(self.edges[-1])]),
            ("bin_width", float(self.edges[1] - self.edges[0])),
            ("total_values", int(self.total)),
            ("below_range", int(self.under)),
            ("above_range", int(self.over)),
        ))


class RasterChangeAccumulator:
    """Streaming candidate-minus-reference statistics on the EXACT common mask."""

    def __init__(self, product: str) -> None:
        import numpy as np

        self.product = product
        self.thresholds = CHANGE_THRESHOLDS[product]
        self.n = 0
        self.sum_signed = 0.0
        self.sum_abs = 0.0
        self.sum_squared = 0.0
        self.above = {float(t): 0 for t in self.thresholds}
        self.abs_hist = HistogramAccumulator(HISTOGRAM_MAX[product], bins=HISTOGRAM_BINS)
        self.signed_hist = SignedHistogramAccumulator(HISTOGRAM_MAX[product],
                                                      bins=HISTOGRAM_BINS)
        self.reference_valid = 0
        self.candidate_valid = 0
        self.mask_equal = 0
        self.total = 0
        self._np = np

    def add(self, reference, candidate) -> None:
        np = self._np

        reference = np.asarray(reference, dtype="float64")
        candidate = np.asarray(candidate, dtype="float64")
        ref_valid = np.isfinite(reference)
        cand_valid = np.isfinite(candidate)
        self.total += int(reference.size)
        self.reference_valid += int(ref_valid.sum())
        self.candidate_valid += int(cand_valid.sum())
        self.mask_equal += int((ref_valid == cand_valid).sum())

        common = ref_valid & cand_valid
        if not common.any():
            return
        difference = candidate[common] - reference[common]
        absolute = np.abs(difference)
        self.n += int(difference.size)
        self.sum_signed += float(difference.sum())
        self.sum_abs += float(absolute.sum())
        self.sum_squared += float((difference ** 2).sum())
        for threshold in self.thresholds:
            self.above[float(threshold)] += int((absolute > float(threshold)).sum())
        self.abs_hist.add(absolute)
        self.signed_hist.add(difference)

    def report(self) -> dict:
        n = self.n
        return OrderedDict((
            ("product", self.product),
            ("units", PRODUCT_UNITS[self.product]),
            ("common_valid_pixels", n),
            ("mean_signed_difference", (self.sum_signed / n) if n else None),
            ("median_signed_difference", self.signed_hist.quantile(50.0)),
            ("global_median_shift", self.signed_hist.quantile(50.0)),
            ("mae", (self.sum_abs / n) if n else None),
            ("rmse", math.sqrt(self.sum_squared / n) if n else None),
            ("p95_absolute_difference", self.abs_hist.quantile(95.0)),
            ("p99_absolute_difference", self.abs_hist.quantile(99.0)),
            ("fraction_above", OrderedDict(
                (f"{threshold:g}", (self.above[float(threshold)] / n) if n else None)
                for threshold in self.thresholds
            )),
            ("count_above", OrderedDict(
                (f"{threshold:g}", self.above[float(threshold)])
                for threshold in self.thresholds
            )),
            ("reference_valid_pixels", self.reference_valid),
            ("candidate_valid_pixels", self.candidate_valid),
            ("valid_mask_agreement", (self.mask_equal / self.total) if self.total else None),
            ("absolute_histogram", self.abs_hist.describe()),
            ("signed_histogram", self.signed_hist.describe()),
            ("interpretation",
             "These are CHANGES, not errors. The reference is a frozen "
             "diagnostic composite, not ground truth."),
            # A row counts as COMPUTED only when the exact common mask actually
            # carried pixels. An empty common mask measured nothing, and saying
            # so is more honest than reporting None metrics as a computed result.
            ("computed", bool(n)),
            ("not_computed_reason", None if n else (
                "no pixel is valid in BOTH the reference and the candidate, so "
                "the common mask is empty and nothing was measured")),
        ))


# -----------------------------------------------------------------------------
# ONE canonical raster-change schema for the JSON summary, the CSV table and the
# Markdown renderer.
#
# `RasterChangeAccumulator.report()` is the PRODUCER and therefore the authority:
# the field list below is asserted against it at import time, so the schema can
# never silently drift from the calculation that fills it.
#
# `mean_signed_difference` is the producer's own name for
# `mean(candidate - reference)` over the exact common mask. It is NOT renamed,
# aliased or duplicated -- `mean_difference` and `candidate_minus_reference_mean`
# do not exist anywhere in this experiment and must not be introduced.
# -----------------------------------------------------------------------------
RASTER_CHANGE_FIELDS = (
    "product",
    "units",
    "common_valid_pixels",
    "mean_signed_difference",
    "median_signed_difference",
    "global_median_shift",
    "mae",
    "rmse",
    "p95_absolute_difference",
    "p99_absolute_difference",
    "fraction_above",
    "count_above",
    "reference_valid_pixels",
    "candidate_valid_pixels",
    "valid_mask_agreement",
    "absolute_histogram",
    "signed_histogram",
    "interpretation",
    "computed",
    "not_computed_reason",
)

#: Scalar metrics that a COMPUTED row must carry as a real number. A computed
#: row is never allowed to leave one of these as None.
RASTER_CHANGE_NUMERIC_FIELDS = (
    "mean_signed_difference",
    "median_signed_difference",
    "global_median_shift",
    "mae",
    "rmse",
)


class ReportSchemaError(HarmonizationError):
    """A report row does not carry the canonical schema its renderer requires."""


def empty_raster_change_report(product: str, *, reason: str) -> dict:
    """A NOT-COMPUTED raster-change row that still carries the full schema.

    Used when an ordered gate stopped the run before the candidate composite
    existed (failed reference reproduction, disconnected graph). Every metric is
    explicitly `None` and `computed` is `False`, so a reader can never mistake a
    missing measurement for a measured zero. Nothing is substituted, defaulted
    or back-filled.
    """
    if product not in PRODUCT_UNITS:
        raise ReportSchemaError(f"unknown product for a raster-change row: {product!r}")
    row = OrderedDict((field, None) for field in RASTER_CHANGE_FIELDS)
    row["product"] = product
    row["units"] = PRODUCT_UNITS[product]
    row["fraction_above"] = OrderedDict(
        (f"{threshold:g}", None) for threshold in CHANGE_THRESHOLDS[product])
    row["count_above"] = OrderedDict(
        (f"{threshold:g}", None) for threshold in CHANGE_THRESHOLDS[product])
    row["interpretation"] = (
        "NOT COMPUTED. An ordered gate stopped the experiment before a candidate "
        "composite existed, so there is no change to report. This is an absence "
        "of measurement, not a measured zero."
    )
    row["computed"] = False
    row["not_computed_reason"] = str(reason)
    return row


def validate_raster_change_rows(changes, *, section: str,
                                products=TARGET_PRODUCTS) -> dict:
    """Fail loudly, and usefully, when a raster-change row is off-schema.

    Raised BEFORE report generation so a schema drift surfaces as an explicit
    contract failure naming the section, the product, the missing keys and the
    keys that were actually present -- instead of a bare `KeyError` from deep
    inside a Markdown f-string.
    """
    if not isinstance(changes, dict):
        raise ReportSchemaError(
            f"[{section}] expected a mapping of product -> raster-change row, "
            f"got {type(changes).__name__}"
        )
    missing_products = [p for p in products if p not in changes]
    if missing_products:
        raise ReportSchemaError(
            f"[{section}] raster-change rows are missing for product(s) "
            f"{missing_products}; available products: {sorted(changes)}"
        )
    for product in products:
        row = changes[product]
        if not isinstance(row, dict):
            raise ReportSchemaError(
                f"[{section}] product {product!r}: expected a raster-change row "
                f"mapping, got {type(row).__name__}"
            )
        missing = [field for field in RASTER_CHANGE_FIELDS if field not in row]
        if missing:
            raise ReportSchemaError(
                f"[{section}] product {product!r}: raster-change row is missing "
                f"required key(s) {missing}. Available keys: {sorted(row)}. "
                f"The canonical schema is produced by "
                f"RasterChangeAccumulator.report() (computed rows) or "
                f"empty_raster_change_report() (gate-stopped rows); build rows "
                f"with one of those rather than assembling them by hand."
            )
        if row.get("computed"):
            blank = [field for field in RASTER_CHANGE_NUMERIC_FIELDS
                     if row.get(field) is None]
            if blank:
                raise ReportSchemaError(
                    f"[{section}] product {product!r}: row is marked computed but "
                    f"metric(s) {blank} are None. A computed metric is never "
                    f"replaced by None, zero or NaN. Available keys: {sorted(row)}."
                )
    return OrderedDict((
        ("section", section),
        ("products", list(products)),
        ("fields", list(RASTER_CHANGE_FIELDS)),
        ("computed", {p: bool(changes[p].get("computed")) for p in products}),
        ("status", "pass"),
    ))


def normalise_raster_changes(changes, *, section: str, reason: str | None = None,
                             products=TARGET_PRODUCTS) -> "OrderedDict[str, dict]":
    """Return validated canonical rows for every product, ONCE.

    `None` (the gate-stopped case) becomes a full set of explicitly
    not-computed rows. Anything else is validated as-is -- an already-computed
    row is passed through UNTOUCHED, so no scientific value is ever rewritten,
    rounded, defaulted or recomputed by report preparation.
    """
    if changes is None:
        if not reason:
            raise ReportSchemaError(
                f"[{section}] raster changes are absent but no reason was given; "
                "a not-computed row must always say why it was not computed"
            )
        changes = OrderedDict(
            (product, empty_raster_change_report(product, reason=reason))
            for product in products
        )
    normalised = OrderedDict((product, changes[product]) for product in products
                            if product in changes)
    for product, row in changes.items():
        normalised.setdefault(product, row)
    validate_raster_change_rows(normalised, section=section, products=products)
    return normalised


def _assert_producer_matches_schema() -> None:
    """The producer defines the schema; drift is a hard import-time failure."""
    produced = tuple(RasterChangeAccumulator(TARGET_PRODUCTS[0]).report())
    if produced != RASTER_CHANGE_FIELDS:
        raise ReportSchemaError(
            "RASTER_CHANGE_FIELDS has drifted from "
            "RasterChangeAccumulator.report(). Producer emits "
            f"{list(produced)}; schema declares {list(RASTER_CHANGE_FIELDS)}."
        )
    unknown = [c for c in RASTER_CHANGE_COLUMNS if c not in RASTER_CHANGE_FIELDS]
    if unknown:
        raise ReportSchemaError(
            f"raster-change CSV columns {unknown} are not in the canonical schema"
        )


# =============================================================================
# Paired boundary-jump evaluation
# =============================================================================
def blocks_from_stratum(accumulator: StratumAccumulator,
                        *, stratum_space: int = STRATUM_SPACE) -> MeanAccumulator:
    """Aggregate a (block, stratum) accumulator to block-level sums/counts."""
    import numpy as np

    out = MeanAccumulator()
    if accumulator.n_cells == 0:
        return out
    blocks = rs._block_from_cell_key(accumulator.keys, stratum_space)
    for block, total, count in zip(blocks, accumulator.sums, accumulator.counts):
        block = int(block)
        out.sums[block] = out.sums.get(block, 0.0) + float(total)
        out.counts[block] = out.counts.get(block, 0) + int(count)
    return out


def _pooled_mean(accumulator: MeanAccumulator, units) -> float | None:
    total = sum(accumulator.sums.get(u, 0.0) for u in units)
    count = sum(accumulator.counts.get(u, 0) for u in units)
    return (total / count) if count else None


def bootstrap_paired_reduction(reference_boundary: MeanAccumulator,
                               reference_control: MeanAccumulator | None,
                               candidate_boundary: MeanAccumulator,
                               candidate_control: MeanAccumulator | None,
                               *, mode: str = EVAL_MODE_EXCESS,
                               replicates: int = BOOTSTRAP_REPLICATES,
                               seed: int = BOOTSTRAP_SEED,
                               min_units: int = MIN_BOOTSTRAP_UNITS,
                               ci: float = BOOTSTRAP_CI) -> dict:
    """Paired spatial-block bootstrap of `reference excess - candidate excess`.

    Reference and candidate are measured on the SAME pair population (the pair
    population is a function of the support rasters, which are invariant), and
    ONE index matrix is drawn and applied to every arm, so a block always
    contributes its reference and candidate pairs together. Individual pixel
    pairs are never resampled.
    """
    import numpy as np

    excess_mode = mode == EVAL_MODE_EXCESS
    arms = [reference_boundary, candidate_boundary]
    if excess_mode:
        if reference_control is None or candidate_control is None:
            raise HarmonizationError("excess mode requires both control arms")
        arms += [reference_control, candidate_control]

    units = sorted(set.intersection(*[
        {u for u, c in arm.counts.items() if c > 0} for arm in arms
    ])) if arms else []
    n_units = len(units)

    result = OrderedDict((
        ("mode", mode),
        ("n_units", n_units),
        ("unit_type", "spatial_block"),
        ("block_size_cells", BOOTSTRAP_BLOCK_SIZE_CELLS),
        ("n_reference_boundary_pairs",
         int(sum(reference_boundary.counts.get(u, 0) for u in units))),
        ("n_candidate_boundary_pairs",
         int(sum(candidate_boundary.counts.get(u, 0) for u in units))),
        ("n_control_pairs",
         int(sum(reference_control.counts.get(u, 0) for u in units))
         if excess_mode else None),
        ("n_bootstrap_requested", int(replicates)),
        ("n_bootstrap_used", 0),
        ("n_bootstrap_skipped", 0),
        ("skipped_reason", None),
        ("seed", int(seed)),
        ("ci", float(ci)),
        ("min_units_required", int(min_units)),
        ("identical_draws_for_reference_and_candidate", True),
        ("resamples_individual_pairs", False),
        ("reference_boundary_mean_abs_jump", None),
        ("candidate_boundary_mean_abs_jump", None),
        ("reference_control_mean_abs_jump", None),
        ("candidate_control_mean_abs_jump", None),
        ("reference_excess_absolute_jump", None),
        ("candidate_excess_absolute_jump", None),
        ("paired_reduction", None),
        ("relative_paired_reduction", None),
        ("interval_low", None),
        ("interval_high", None),
        ("status", "insufficient_units"),
        ("verdict", VERDICT_INSUFFICIENT),
    ))
    if n_units == 0:
        result["skipped_reason"] = "no spatial block carries every required arm"
        return result

    def arrays(arm):
        return (np.array([arm.sums.get(u, 0.0) for u in units], dtype="float64"),
                np.array([arm.counts.get(u, 0) for u in units], dtype="float64"))

    rb_sum, rb_count = arrays(reference_boundary)
    cb_sum, cb_count = arrays(candidate_boundary)
    result["reference_boundary_mean_abs_jump"] = _pooled_mean(reference_boundary, units)
    result["candidate_boundary_mean_abs_jump"] = _pooled_mean(candidate_boundary, units)

    if excess_mode:
        rc_sum, rc_count = arrays(reference_control)
        cc_sum, cc_count = arrays(candidate_control)
        result["reference_control_mean_abs_jump"] = _pooled_mean(reference_control, units)
        result["candidate_control_mean_abs_jump"] = _pooled_mean(candidate_control, units)
        reference_excess = (result["reference_boundary_mean_abs_jump"]
                            - result["reference_control_mean_abs_jump"])
        candidate_excess = (result["candidate_boundary_mean_abs_jump"]
                            - result["candidate_control_mean_abs_jump"])
    else:
        reference_excess = result["reference_boundary_mean_abs_jump"]
        candidate_excess = result["candidate_boundary_mean_abs_jump"]

    result["reference_excess_absolute_jump"] = reference_excess
    result["candidate_excess_absolute_jump"] = candidate_excess
    reduction = reference_excess - candidate_excess
    result["paired_reduction"] = float(reduction)
    result["relative_paired_reduction"] = (
        float(reduction / reference_excess) if reference_excess and reference_excess > 0
        else None
    )

    if n_units < int(min_units):
        result["skipped_reason"] = (
            f"only {n_units} independent spatial blocks (< {min_units} required)"
        )
        return result

    indices = draw_bootstrap_indices(n_units, replicates=replicates, seed=seed)

    def pooled(sums, counts):
        return sums[indices].sum(axis=1), counts[indices].sum(axis=1)

    rb_s, rb_c = pooled(rb_sum, rb_count)
    cb_s, cb_c = pooled(cb_sum, cb_count)
    usable = (rb_c > 0) & (cb_c > 0)
    if excess_mode:
        rc_s, rc_c = pooled(rc_sum, rc_count)
        cc_s, cc_c = pooled(cc_sum, cc_count)
        usable &= (rc_c > 0) & (cc_c > 0)

    result["n_bootstrap_used"] = int(usable.sum())
    result["n_bootstrap_skipped"] = int((~usable).sum())
    if result["n_bootstrap_skipped"]:
        result["skipped_reason"] = "resampled blocks contained an empty arm"
    if not usable.any():
        result["status"] = "no_usable_replicates"
        return result

    reference_replicate = rb_s[usable] / rb_c[usable]
    candidate_replicate = cb_s[usable] / cb_c[usable]
    if excess_mode:
        reference_replicate = reference_replicate - rc_s[usable] / rc_c[usable]
        candidate_replicate = candidate_replicate - cc_s[usable] / cc_c[usable]
    draws = reference_replicate - candidate_replicate

    lo_q = (1.0 - float(ci)) / 2.0
    result["interval_low"] = float(np.quantile(draws, lo_q))
    result["interval_high"] = float(np.quantile(draws, 1.0 - lo_q))
    result["status"] = "estimated"
    result["verdict"] = classify_reduction_interval(result)
    return result


def classify_reduction_interval(interval: dict) -> str:
    """`supported_reduction` / `supported_increase` / `uncertain`.

    The interval is on `reference excess - candidate excess`, so a wholly
    positive interval means the candidate REDUCED the jump and a wholly negative
    interval means it INCREASED it.
    """
    if interval.get("status") != "estimated":
        return VERDICT_INSUFFICIENT
    low, high = interval.get("interval_low"), interval.get("interval_high")
    if low is not None and low > 0.0:
        return VERDICT_SUPPORTED_REDUCTION
    if high is not None and high < 0.0:
        return VERDICT_SUPPORTED_INCREASE
    return VERDICT_UNCERTAIN


# =============================================================================
# Predeclared, ORDERED decision rule (never relaxed after inspecting results)
# =============================================================================
DECISION_RULE_TEXT = (
    "Ordered gates. 1) invalid_inputs. 2) invalid_reference_reproduction. "
    "3) insufficient_date_overlap_graph. 4) support_invariance_failed. "
    "5) seam_reduction_not_supported unless the paired block-bootstrap interval "
    "is wholly above zero AND the point relative reduction is at least 10% for "
    "current_support_change and current_unique_date_count_change, on BOTH "
    "current_minus_baseline_celsius and anomaly_zscore. "
    "6) seam_reduced_with_nonboundary_tradeoff when a supported INCREASE appears "
    "at none_of_known_boundaries or at pathrow_only. "
    "7) seam_reduced_with_value_scale_tradeoff when |global median current-LST "
    "shift| > 0.5 C, any |alpha_d| > 5 C, or the graph residual diagnostics are "
    "unstable. 8) otherwise eligible_for_downstream_ab. No criterion may be "
    "relaxed after inspecting results, and seam_fixed / production_approved / "
    "production_ready are unreachable."
)


def _status(status: str, reasons, evidence: dict, checks: dict) -> dict:
    if status not in FINAL_STATUSES:
        raise HarmonizationError(f"attempted to emit an unknown status: {status!r}")
    if status in FORBIDDEN_CONCLUSIONS:                     # pragma: no cover
        raise HarmonizationError(f"forbidden conclusion: {status!r}")
    return OrderedDict((
        ("final_status", status),
        ("final_status_meaning", FINAL_STATUS_MEANINGS[status]),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("decision_rule", DECISION_RULE_TEXT),
        ("reasons", list(reasons)),
        ("checks", checks),
        ("allowed_final_statuses", list(FINAL_STATUSES)),
        ("forbidden_conclusions", list(FORBIDDEN_CONCLUSIONS)),
        ("seam_fixed", False),
        ("production_approved", False),
        ("changes_production_reducer", CHANGES_PRODUCTION_REDUCER),
        ("labels_used", USES_LABELS),
        ("step8_metrics_used", USES_STEP8_METRICS),
        ("model_performance_used", USES_MODEL_PERFORMANCE),
    ))


def decide_final_status(evidence: dict) -> dict:
    """Apply the ordered predeclared rule to the assembled evidence."""
    checks: "OrderedDict[str, object]" = OrderedDict()

    # --- 1. technical validity -------------------------------------------
    checks["inputs_valid"] = bool(evidence.get("inputs_valid"))
    if not checks["inputs_valid"]:
        return _status(STATUS_INVALID_INPUTS,
                       evidence.get("invalid_input_reasons") or ["inputs are not valid"],
                       evidence, checks)

    # --- 2. frozen reference reproduction ---------------------------------
    checks["reference_reproduction_passes"] = bool(
        evidence.get("reference_reproduction_passes")
    )
    if not checks["reference_reproduction_passes"]:
        return _status(STATUS_INVALID_REFERENCE,
                       evidence.get("reference_reproduction_failures")
                       or ["the frozen reference composite was not reproduced"],
                       evidence, checks)

    # --- 3. graph connectivity --------------------------------------------
    checks["primary_graph_connected"] = bool(evidence.get("primary_graph_connected"))
    if not checks["primary_graph_connected"]:
        return _status(STATUS_INSUFFICIENT_GRAPH,
                       evidence.get("graph_failure_reasons")
                       or ["the primary date-overlap graph is not connected"],
                       evidence, checks)

    # --- 4. exact support invariance --------------------------------------
    checks["support_invariance_passes"] = bool(evidence.get("support_invariance_passes"))
    if not checks["support_invariance_passes"]:
        return _status(STATUS_SUPPORT_INVARIANCE_FAILED,
                       evidence.get("support_invariance_failures")
                       or ["the candidate changed the pixel population"],
                       evidence, checks)

    # --- 5. primary support-boundary reduction ----------------------------
    reductions = evidence.get("boundary_reductions") or {}
    unmet: list[str] = []
    supported_map: "OrderedDict[str, bool]" = OrderedDict()
    relative_map: "OrderedDict[str, float | None]" = OrderedDict()
    for product in DECISION_PRODUCTS:
        for boundary in REQUIRED_REDUCTION_BOUNDARIES:
            row = (reductions.get(product) or {}).get(boundary) or {}
            key = f"{product}|{boundary}"
            supported = row.get("verdict") == VERDICT_SUPPORTED_REDUCTION
            relative = row.get("relative_paired_reduction")
            supported_map[key] = bool(supported)
            relative_map[key] = relative
            if not supported:
                unmet.append(
                    f"{key}: verdict={row.get('verdict')!r} "
                    f"interval=[{row.get('interval_low')}, {row.get('interval_high')}]"
                )
            elif relative is None or relative < MIN_RELATIVE_REDUCTION:
                unmet.append(
                    f"{key}: point relative reduction {relative} < {MIN_RELATIVE_REDUCTION}"
                )
    checks["required_boundary_reductions_supported"] = dict(supported_map)
    checks["required_boundary_relative_reductions"] = dict(relative_map)
    checks["minimum_relative_reduction"] = MIN_RELATIVE_REDUCTION
    if unmet:
        return _status(STATUS_NOT_SUPPORTED, unmet, evidence, checks)

    # --- 6. trade-off away from the targeted boundaries -------------------
    tradeoffs: list[str] = []
    for boundary in NO_SUPPORTED_INCREASE_BOUNDARIES:
        for product in TARGET_PRODUCTS:
            row = (reductions.get(product) or {}).get(boundary) or {}
            if row.get("verdict") == VERDICT_SUPPORTED_INCREASE:
                tradeoffs.append(
                    f"supported INCREASE at {boundary} for {product}: "
                    f"interval=[{row.get('interval_low')}, {row.get('interval_high')}]"
                )
    checks["no_supported_increase_at_nonboundary"] = not any(
        (reductions.get(p) or {}).get(NONBOUNDARY_CONTROL, {}).get("verdict")
        == VERDICT_SUPPORTED_INCREASE for p in TARGET_PRODUCTS
    )
    checks["no_supported_increase_at_pathrow_only"] = not any(
        (reductions.get(p) or {}).get(rs.CLASS_PATHROW_ONLY, {}).get("verdict")
        == VERDICT_SUPPORTED_INCREASE for p in TARGET_PRODUCTS
    )
    if tradeoffs:
        return _status(PATHROW_INCREASE_STATUS, tradeoffs, evidence, checks)

    # --- 7. value-scale trade-off ------------------------------------------
    scale_issues: list[str] = []
    shift = evidence.get("global_median_current_lst_shift")
    checks["global_median_current_lst_shift"] = shift
    checks["global_median_shift_bound"] = MAX_ABS_GLOBAL_MEDIAN_SHIFT_CELSIUS
    if shift is None or abs(float(shift)) > MAX_ABS_GLOBAL_MEDIAN_SHIFT_CELSIUS:
        scale_issues.append(
            f"global candidate-minus-reference median current-LST shift {shift} C "
            f"exceeds +/-{MAX_ABS_GLOBAL_MEDIAN_SHIFT_CELSIUS} C"
        )
    max_offset = evidence.get("max_abs_date_offset")
    checks["max_abs_date_offset"] = max_offset
    checks["max_abs_date_offset_bound"] = MAX_ABS_DATE_OFFSET_CELSIUS
    if max_offset is None or abs(float(max_offset)) > MAX_ABS_DATE_OFFSET_CELSIUS:
        scale_issues.append(
            f"a fitted date offset of {max_offset} C exceeds the predeclared "
            f"{MAX_ABS_DATE_OFFSET_CELSIUS} C bound"
        )
    checks["offset_estimation_stable"] = bool(evidence.get("offset_estimation_stable"))
    if not evidence.get("offset_estimation_stable"):
        scale_issues.extend(evidence.get("offset_instability_reasons")
                            or ["graph residual diagnostics indicate unstable estimation"])
    if scale_issues:
        return _status(STATUS_VALUE_SCALE_TRADEOFF, scale_issues, evidence, checks)

    # --- 8. eligible --------------------------------------------------------
    checks["labels_or_model_metrics_used"] = bool(
        USES_LABELS or USES_STEP8_METRICS or USES_MODEL_PERFORMANCE
    )
    return _status(
        STATUS_ELIGIBLE,
        ["every predeclared gate passed; the candidate may be carried into a "
         "controlled downstream A/B and into an independent second AOI, and "
         "into nothing else"],
        evidence, checks,
    )


def next_experiment_text(final_status: str) -> str:
    if final_status == STATUS_ELIGIBLE:
        return (
            "Run the same controlled downstream A/B used for the date-balanced "
            "candidate, with the overlap-harmonized current composite as the "
            "candidate chain and the date-balanced composite as the reference, "
            "then repeat this harmonization in an independent AOI (bejis_2022). "
            "Do NOT change the production reducer on single-AOI evidence."
        )
    if final_status == STATUS_INSUFFICIENT_GRAPH:
        return (
            "The current-period dates do not overlap enough to identify joint "
            "additive offsets. Either widen the overlap evidence (a second AOI "
            "with more sidelap, or per-path/row sub-graphs reported separately) "
            "or test a mechanism that does not require a connected date graph."
        )
    if final_status in (STATUS_NONBOUNDARY_TRADEOFF, STATUS_VALUE_SCALE_TRADEOFF):
        return (
            "Constrain the offset model before re-testing: the additive "
            "scene-wide offset is evidently absorbing structure it should not. "
            "A per-path/row or per-date-pair-restricted variant, still with the "
            "exact support-invariance gate, is the next diagnostic step."
        )
    if final_status == STATUS_NOT_SUPPORTED:
        return (
            "Acquisition-date offsets alone do not explain the residual "
            "current-support seam in this AOI. Test the remaining predeclared "
            "secondary mechanisms (baseline support, anomaly-threshold "
            "boundaries) or a per-pixel selected-scene provenance experiment."
        )
    return (
        "Repair the failing technical gate before any scientific claim is made. "
        "No result from this run may be interpreted."
    )


def required_limitations() -> list[str]:
    """The limitations every report MUST carry, verbatim."""
    return [
        "Manavgat only. This is one AOI and generalises to nothing.",
        "The correction model is a single ADDITIVE offset per acquisition date; "
        "no multiplicative, emissivity-dependent or land-cover-dependent term is "
        "estimated.",
        "No spatially varying weather correction is applied. Real weather varies "
        "across the AOI within a single overpass; a scene-wide constant cannot "
        "represent that.",
        "Overlap estimates can retain land-cover or terrain confounding: two "
        "dates are compared only where both see the surface, and that common "
        "area is not a random sample of the AOI.",
        "Scene and path/row evidence is METADATA-derived, not pixel-level "
        "selected-scene provenance.",
        "There is no pixel-level record of which scene supplied which composite "
        "value, so date attribution at a pixel is inferred from validity, not "
        "observed.",
        "No production decision is made or implied.",
        "No model-performance result was computed, consulted or used.",
        "No generalisation beyond this AOI, this window and these seven "
        "acquisition dates is supported.",
        "The baseline remains the frozen four-year climatology; it was neither "
        "recomputed nor re-exported.",
        "Secondary baseline-support and anomaly-threshold effects are NOT "
        "corrected here and remain in the residual.",
        "A successful harmonisation would NOT prove that current support is the "
        "only seam mechanism; it would only show that date-offset structure is "
        "sufficient to move this particular boundary statistic.",
    ]


def inherited_limitations(state: dict) -> list[str]:
    return [
        f"Inherited: the composite counterfactual finished at "
        f"{state.get('counterfactual_final_status')!r}.",
        f"Inherited: the downstream A/B finished at "
        f"{state.get('downstream_ab_final_status')!r}, which is eligibility for "
        "a second-AOI validation only.",
        f"Inherited: the residual seam attribution finished at "
        f"{state.get('residual_seam_final_status')!r}.",
        "Inherited: every frozen input carries the QA-mask provenance mismatch "
        "recorded by the counterfactual audit (QA_PIXEL only; QA_RADSAT not "
        "applied).",
    ]


# =============================================================================
# Checkpoints
# =============================================================================
CHECKPOINT_FILENAME = "harmonization_checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = "1.0-current-support-harmonization"

PLANNED_STAGES = (
    "input_validation",
    "daily_mosaic_inventory",
    "reference_reproduction",
    "overlap_graph_construction",
    "graph_solution",
    "daily_harmonisation",
    "candidate_composite",
    "support_invariance",
    "derived_products",
    "boundary_analysis",
    "bootstrap",
    "maps",
    "reports",
)


def checkpoint_path(root: Path) -> Path:
    return Path(root) / "checkpoints" / CHECKPOINT_FILENAME


def read_checkpoint(root: Path) -> dict:
    path = checkpoint_path(root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def file_reference(path: Path) -> dict:
    """Size AND sha256, so resume validates content rather than presence."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "bytes": -1, "sha256": None}
    signed = sha256_and_size(path)
    return {"path": str(path), "bytes": signed["bytes"], "sha256": signed["sha256"]}


def write_checkpoint_stage(root: Path, stage: str, outputs, extra: dict | None = None) -> dict:
    """Atomically record a completed stage with its output hashes."""
    if stage not in PLANNED_STAGES:
        raise HarmonizationError(f"unknown checkpoint stage: {stage!r}")
    root = Path(root)
    payload = read_checkpoint(root)
    payload.setdefault("experiment", DIAGNOSTIC_NAMESPACE)
    payload["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    payload.setdefault("stages", {})
    payload["stages"][stage] = OrderedDict((
        ("completed_at", datetime.now(timezone.utc).isoformat()),
        ("outputs", [file_reference(p) for p in outputs]),
        ("rss_mib", process_rss_mib()),
        *tuple((extra or {}).items()),
    ))
    payload["last_stage"] = stage
    write_json_atomic(checkpoint_path(root), payload)
    return payload


def outputs_still_valid(entries) -> bool:
    """Every recorded output must still exist with its recorded size AND hash."""
    for entry in entries or []:
        path = Path(entry.get("path", ""))
        if not path.exists():
            return False
        try:
            if int(entry.get("bytes", -1)) != int(path.stat().st_size):
                return False
        except OSError:
            return False
        recorded = entry.get("sha256")
        if recorded and sha256_and_size(path)["sha256"] != recorded:
            return False
    return True


def stage_is_reusable(root: Path, stage: str) -> bool:
    """A stage may be reused ONLY when its recorded outputs still validate."""
    checkpoint = read_checkpoint(root)
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return False
    entry = (checkpoint.get("stages") or {}).get(stage)
    if not entry:
        return False
    return outputs_still_valid(entry.get("outputs") or [])


# =============================================================================
# Configuration snapshot (everything predeclared, in one frozen record)
# =============================================================================
def build_config_snapshot(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    window = frozen_current_window(experiment_id, base_dir)
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("experiment_kind", "diagnostic_non_destructive_counterfactual"),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("reference_composite", REFERENCE_COMPOSITE),
        ("candidate_composite", CANDIDATE_COMPOSITE),
        ("intervention", OrderedDict((
            ("changed_factor",
             "additive per-acquisition-date offsets estimated from spatial "
             "overlaps between daily mosaics"),
            ("model", "Y_d(p) = T(p) + alpha_d + error_d(p)"),
            ("applied_as", "Y'_d(p) = Y_d(p) - alpha_d"),
            ("spatial_operation_performed", False),
            ("held_fixed", [
                "source scenes", "current-period date window", "Landsat scaling",
                "QA mask", "same-day mosaicking rule", "per-pixel valid date support",
                "valid pixel mask", "frozen baseline climatology",
                "Step5 count and baseline-std masks", "Celsius units",
            ]),
            ("is_common_date_subset_experiment", False),
            ("is_minimum_count_masking_experiment", False),
        ))),
        ("current_window", window),
        ("step5_policy", step5_thresholds()),
        ("physical_celsius_range", [PHYSICAL_CELSIUS_MIN, PHYSICAL_CELSIUS_MAX]),
        ("target_products", list(TARGET_PRODUCTS)),
        ("decision_products", list(DECISION_PRODUCTS)),
        ("reference_reproduction", OrderedDict((
            ("exact_checks", list(REPRODUCTION_EXACT_CHECKS)),
            ("gating_tolerances", dict(REPRODUCTION_TOLERANCES)),
            ("gating_tolerance_source",
             "src.landsat_composite_downstream_ab.REPRODUCTION_TOLERANCES"),
            ("reported_tight_tolerance", REPRODUCTION_TIGHT_REFERENCE_TOL),
            ("reported_tight_tolerance_source",
             "src.landsat_composite_counterfactual_audit."
             "REPRODUCTION_TOLERANCES['physical_float32']"),
        ))),
        ("overlap_graph", OrderedDict((
            ("block_size_cells", GRAPH_BLOCK_SIZE_CELLS),
            ("min_block_common_pixels", MIN_BLOCK_COMMON_PIXELS),
            ("edge_statistic",
             "median over eligible spatial blocks of the within-block median of "
             "Y_j - Y_i on pixels valid on BOTH dates"),
            ("primary_min_common_pixels", PRIMARY_MIN_COMMON_PIXELS),
            ("primary_min_independent_blocks", PRIMARY_MIN_INDEPENDENT_BLOCKS),
            ("sensitivity_thresholds", [dict(t) for t in SENSITIVITY_THRESHOLDS]),
            ("threshold_selection_policy", THRESHOLD_SELECTION_POLICY),
            ("connectivity_gate",
             "all retained current-period dates must form ONE connected "
             "component in the PRIMARY graph"),
        ))),
        ("offset_solution", OrderedDict((
            ("objective", "minimise sum_ij w_ij ((alpha_j - alpha_i) - delta_ij)^2"),
            ("weights", EDGE_WEIGHT_FORMULA),
            ("weight_cap_multiple", WEIGHT_CAP_MULTIPLE),
            ("min_edge_sigma_celsius", MIN_EDGE_SIGMA_CELSIUS),
            ("identifying_constraint", IDENTIFYING_CONSTRAINT),
            ("solver", "deterministic weighted least squares (LAPACK) on the "
                       "constraint-augmented weighted graph Laplacian"),
            ("max_graph_condition_number", MAX_GRAPH_CONDITION_NUMBER),
            ("max_edge_residual_rms_celsius", MAX_EDGE_RESIDUAL_RMS_CELSIUS),
        ))),
        ("support_invariance", OrderedDict((
            ("checks", list(SUPPORT_INVARIANCE_CHECKS)),
            ("required_unequal_pixel_count", 0),
            ("required_changed_valid_pixel_count", 0),
            ("required_mask_agreement", 1.0),
        ))),
        ("boundary_evaluation", OrderedDict((
            ("boundaries", dict(EVALUATED_BOUNDARIES)),
            ("required_reduction_boundaries", list(REQUIRED_REDUCTION_BOUNDARIES)),
            ("nonboundary_control", NONBOUNDARY_CONTROL),
            ("no_supported_increase_boundaries", list(NO_SUPPORTED_INCREASE_BOUNDARIES)),
            ("reused_from", RESIDUAL_SEAM_NAMESPACE),
            ("reused_semantics", [
                "horizontal and vertical adjacency pairs",
                "no zero filling; an edge touching a NaN endpoint is dropped",
                "128-cell spatial block definition",
                "within-block matched controls stratified by predeclared "
                "elevation-, slope- and NDVI-gradient bins",
                "boundary metadata and path/row rasterization",
                "bootstrap seed and replicate count",
                "current-support definitions",
                "pathrow-only definition (excludes support and threshold pairs)",
                "threshold definitions and the predeclared near-std epsilon",
            ]),
            ("matched_control_strategy", rs.MATCHED_CONTROL_STRATEGY),
            ("near_std_threshold_epsilon", rs.STD_THRESHOLD_EPSILON_PRIMARY),
            ("elevation_gradient_bins_m", list(rs.ELEVATION_GRADIENT_BINS)),
            ("slope_gradient_bins_deg", list(rs.SLOPE_GRADIENT_BINS)),
            ("ndvi_gradient_bins", list(rs.NDVI_GRADIENT_BINS)),
        ))),
        ("bootstrap", OrderedDict((
            ("unit", "spatial_block"),
            ("block_size_cells", BOOTSTRAP_BLOCK_SIZE_CELLS),
            ("replicates", BOOTSTRAP_REPLICATES),
            ("seed", BOOTSTRAP_SEED),
            ("ci", BOOTSTRAP_CI),
            ("ci_lower_percentile", BOOTSTRAP_CI_LOWER_PCT),
            ("ci_upper_percentile", BOOTSTRAP_CI_UPPER_PCT),
            ("min_units", MIN_BOOTSTRAP_UNITS),
            ("resamples_individual_pairs", False),
            ("identical_draws_for_reference_and_candidate", True),
            ("policy", BOOTSTRAP_UNIT_POLICY),
        ))),
        ("decision_bounds", OrderedDict((
            ("min_relative_reduction", MIN_RELATIVE_REDUCTION),
            ("max_abs_date_offset_celsius", MAX_ABS_DATE_OFFSET_CELSIUS),
            ("max_abs_global_median_shift_celsius",
             MAX_ABS_GLOBAL_MEDIAN_SHIFT_CELSIUS),
        ))),
        ("change_reporting_thresholds", OrderedDict(
            (product, list(values)) for product, values in CHANGE_THRESHOLDS.items()
        )),
        ("allowed_final_statuses", list(FINAL_STATUSES)),
        ("forbidden_conclusions", list(FORBIDDEN_CONCLUSIONS)),
        ("decision_rule", DECISION_RULE_TEXT),
        ("planned_stages", list(PLANNED_STAGES)),
        ("smoothing_applied", SMOOTHING_APPLIED),
        ("spatial_interpolation_applied", SPATIAL_INTERPOLATION_APPLIED),
        ("baseline_recomputed", RECOMPUTES_BASELINE),
        ("modifies_production_reducer", CHANGES_PRODUCTION_REDUCER),
        ("labels_used", USES_LABELS),
        ("step8_metrics_used", USES_STEP8_METRICS),
        ("model_performance_used", USES_MODEL_PERFORMANCE),
        ("reruns_step6_step7_step8", False),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


# =============================================================================
# Dry-run plan (ZERO writes, ZERO Earth Engine)
# =============================================================================
def build_dry_run_plan(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Everything the live run would do, without doing any of it.

    This function creates no directory, writes no file and performs no Earth
    Engine operation of any kind -- it does not import, initialise, authenticate
    or call Earth Engine.
    """
    assert_supported_experiment(experiment_id)
    plan = build_input_plan(experiment_id, base_dir)
    layout = plan_output_layout(experiment_id, base_dir)
    expected = plan_expected_files(experiment_id, base_dir)
    state = load_upstream_state(experiment_id, base_dir)

    try:
        export_plan = build_daily_export_plan(experiment_id, base_dir)
        export_error = None
    except (PrerequisiteError, HarmonizationError) as error:
        export_plan = None
        export_error = str(error)

    try:
        pathrow = resolve_pathrow_availability(experiment_id, base_dir)
    except Exception as error:                              # noqa: BLE001
        pathrow = {"availability": "unavailable", "reason": str(error),
                   "interface_count": 0, "interfaces": []}

    resolved = OrderedDict()
    for role, entry in plan.items():
        resolved[role] = OrderedDict((
            ("path", str(entry["path"])),
            ("present", Path(entry["path"]).exists()),
            ("required", bool(entry["required"])),
            ("source", entry["source"]),
            ("family", entry["family"]),
        ))

    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("reference_composite", REFERENCE_COMPOSITE),
        ("candidate_composite", CANDIDATE_COMPOSITE),
        ("target_products", list(TARGET_PRODUCTS)),
        ("output_root", str(layout["root"])),
        ("output_layout", OrderedDict((k, str(v)) for k, v in layout.items())),
        ("resolved_inputs", resolved),
        ("missing_required_inputs", missing_required_inputs(plan)),
        ("missing_optional_inputs", missing_optional_inputs(plan)),
        ("upstream_prerequisites", state),
        ("pathrow_evidence", pathrow),
        ("daily_export_plan", export_plan),
        ("daily_export_plan_error", export_error),
        ("configuration", build_config_snapshot(experiment_id, base_dir)),
        ("decision_rule", DECISION_RULE_TEXT),
        ("allowed_final_statuses", list(FINAL_STATUSES)),
        ("forbidden_conclusions", list(FORBIDDEN_CONCLUSIONS)),
        ("planned_stages", list(PLANNED_STAGES)),
        ("expected_files", OrderedDict((k, str(v)) for k, v in expected.items())),
        ("limitations", required_limitations()),
        ("writes_performed", False),
        ("directories_created", 0),
        ("earth_engine_calls", 0),
        ("earth_engine_initialised", False),
        ("rasters_modified", 0),
        ("frozen_namespaces_touched", 0),
        ("smoothing_applied", SMOOTHING_APPLIED),
        ("spatial_interpolation_applied", SPATIAL_INTERPOLATION_APPLIED),
        ("baseline_recomputed", RECOMPUTES_BASELINE),
        ("labels_used", USES_LABELS),
        ("step8_metrics_used", USES_STEP8_METRICS),
    ))


# =============================================================================
# Daily-raster validity contract
# =============================================================================
#: Bumped whenever the MEANING of an exported daily raster changes (reducer,
#: QA policy, scaling, masking or fill encoding). Daily files recorded under an
#: older semantic version are refused, not silently reused.
DAILY_SEMANTIC_VERSION = "2.0-sentinel-verified-edge"

#: Bumped whenever the nodata/mask decoding policy changes.
NODATA_POLICY_VERSION = "2.0-single-sentinel-no-constant-fill"

#: Bumped whenever the reference reconstruction implementation changes.
RECONSTRUCTION_VERSION = "2.1-roundoff-aware-reporting"

DAILY_CONTRACT_VERSIONS = OrderedDict((
    ("daily_semantic_version", DAILY_SEMANTIC_VERSION),
    ("nodata_policy_version", NODATA_POLICY_VERSION),
    ("reconstruction_version", RECONSTRUCTION_VERSION),
))

#: A pixel where at least this many dates carry the BIT-IDENTICAL finite value
#: is a constant-fill fingerprint, not a physical coincidence: seven independent
#: overpasses cannot all read exactly the same float32 temperature.
CONSTANT_FILL_MIN_DATES = 3

#: Root-cause classifications this contract can emit.
ROOT_CAUSE_OK = "daily_rasters_conform"
ROOT_CAUSE_FLOAT32_ROUNDOFF = "float32_roundoff_only"
ROOT_CAUSE_CONSTANT_FILL = "daily_export_constant_fill_artefact"
ROOT_CAUSE_NODATA_TAG = "daily_nodata_tag_mismatch"
ROOT_CAUSE_GRID = "daily_grid_mismatch"


class DailyRasterContractError(HarmonizationError):
    """An exported daily raster violates the daily-raster validity contract."""


def _edge_band_summary(rows, cols, height: int, width: int) -> dict:
    """Describe whether flagged pixels sit in complete edge rows/columns.

    A download that pads the AOI margin writes WHOLE edge rows or columns. That
    geometry is what separates an export artefact from scattered bad pixels.
    """
    import numpy as np

    rows = np.asarray(rows, dtype="int64")
    cols = np.asarray(cols, dtype="int64")
    if not rows.size:
        return OrderedDict((("rows_touched", []), ("cols_touched", []),
                            ("confined_to_edge_band", False), ("edge_band_rows", 0)))
    unique_rows = np.unique(rows)
    unique_cols = np.unique(cols)
    top = int((unique_rows < 2).sum())
    bottom = int((unique_rows >= height - 2).sum())
    left = int((unique_cols < 2).sum())
    right = int((unique_cols >= width - 2).sum())
    confined = bool(
        unique_rows.size <= 4 and (top + bottom) == unique_rows.size
    ) or bool(
        unique_cols.size <= 4 and (left + right) == unique_cols.size
    )
    return OrderedDict((
        ("rows_touched", [int(r) for r in unique_rows[:8]]),
        ("row_count_touched", int(unique_rows.size)),
        ("cols_touched_count", int(unique_cols.size)),
        ("confined_to_edge_band", confined),
        ("edge_band_rows", top + bottom),
        ("edge_band_cols", left + right),
        ("interpretation",
         "Whole edge rows/columns carrying one identical value across every "
         "date is the fingerprint of a download that padded the AOI margin "
         "instead of honouring the declared nodata sentinel."
         if confined else
         "Flagged pixels are not confined to the grid margin."),
    ))


def validate_daily_raster_contract(paths, dates, *, height: int, width: int,
                                   window_rows: int = WINDOW_ROWS,
                                   logger=None) -> dict:
    """The strict validity contract every daily mosaic must satisfy.

    Enforced BEFORE the reference reproduction, so a corrupt export is reported
    as a corrupt export rather than as a failed scientific gate.

    Checks, in order:

    1. every file declares the single expected nodata sentinel;
    2. every file shares the one exact grid;
    3. no pixel carries a BIT-IDENTICAL finite value across `CONSTANT_FILL_MIN_DATES`
       or more dates -- the constant-fill fingerprint.

    Zero is NOT treated as nodata. A genuine 0.0 C reading on ONE date stays
    valid; only a value repeated identically across many dates is flagged, and
    the flagged pixels are reported with their value, count and geometry rather
    than being quietly dropped.
    """
    import numpy as np
    import rasterio

    paths = [Path(p) for p in paths]
    per_date = []
    nodata_failures = []
    for date, path in zip(dates, paths):
        with rasterio.open(path) as src:
            nodata = src.nodata
        entry = OrderedDict((
            ("acquisition_date", date),
            ("path", str(path)),
            ("declared_nodata", None if nodata is None else float(nodata)),
            ("expected_nodata", float(NODATA_SENTINEL)),
            ("nodata_tag_ok", nodata is not None
             and float(nodata) == float(NODATA_SENTINEL)),
            ("sentinel_pixels", 0),
            ("finite_pixels", 0),
            ("constant_fill_pixels", 0),
            ("min_finite", None),
            ("max_finite", None),
        ))
        if not entry["nodata_tag_ok"]:
            nodata_failures.append(
                f"{date}: declares nodata={nodata!r}, expected {NODATA_SENTINEL}")
        per_date.append(entry)

    grid = assert_same_grid(paths)

    fill_values: dict[float, int] = {}
    fill_rows: list = []
    fill_cols: list = []
    total_fill = 0

    for start in range(0, int(height), int(window_rows)):
        stop = min(start + int(window_rows), int(height))
        stack = read_daily_stack(paths, start, stop)
        valid = np.isfinite(stack)

        for index, entry in enumerate(per_date):
            layer = stack[index]
            finite = valid[index]
            entry["sentinel_pixels"] += int((~finite).sum())
            entry["finite_pixels"] += int(finite.sum())
            if finite.any():
                low, high = float(layer[finite].min()), float(layer[finite].max())
                entry["min_finite"] = low if entry["min_finite"] is None \
                    else min(entry["min_finite"], low)
                entry["max_finite"] = high if entry["max_finite"] is None \
                    else max(entry["max_finite"], high)

        # Bit-identical repetition across dates, counted per pixel.
        import warnings

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # A pixel invalid on every date is expected and contributes nothing.
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            first = np.where(valid, stack, np.nan)
            reference_layer = np.nanmax(first, axis=0)
        agree = valid & (stack == reference_layer[None, :, :])
        agree_count = agree.sum(axis=0)
        flagged = (agree_count >= int(CONSTANT_FILL_MIN_DATES)) & \
            (valid.sum(axis=0) == agree_count)
        if flagged.any():
            rr, cc = np.where(flagged)
            total_fill += int(rr.size)
            fill_rows.append(rr + start)
            fill_cols.append(cc)
            for value, count in zip(*np.unique(reference_layer[flagged],
                                               return_counts=True)):
                fill_values[float(value)] = fill_values.get(float(value), 0) + int(count)
            for index, entry in enumerate(per_date):
                entry["constant_fill_pixels"] += int((agree[index] & flagged).sum())
        if logger is not None:
            logger("daily_raster_contract", start, stop, int(height))

    rows = np.concatenate(fill_rows) if fill_rows else np.empty(0, dtype="int64")
    cols = np.concatenate(fill_cols) if fill_cols else np.empty(0, dtype="int64")

    failures: list[str] = list(nodata_failures)
    root_cause = ROOT_CAUSE_OK
    if nodata_failures:
        root_cause = ROOT_CAUSE_NODATA_TAG
    if total_fill:
        root_cause = ROOT_CAUSE_CONSTANT_FILL
        top = sorted(fill_values.items(), key=lambda kv: -kv[1])[:5]
        failures.append(
            f"{total_fill} pixel(s) carry one bit-identical value across "
            f">= {CONSTANT_FILL_MIN_DATES} dates; most frequent fill value(s): "
            + ", ".join(f"{v:g} x{n}" for v, n in top)
        )

    report = OrderedDict((
        ("contract", "daily_raster_validity"),
        ("versions", dict(DAILY_CONTRACT_VERSIONS)),
        ("date_count", len(paths)),
        ("grid", grid),
        ("per_date", per_date),
        ("constant_fill", OrderedDict((
            ("min_dates_for_flag", int(CONSTANT_FILL_MIN_DATES)),
            ("flagged_pixels", int(total_fill)),
            ("values", OrderedDict(
                (f"{value:g}", count) for value, count in
                sorted(fill_values.items(), key=lambda kv: -kv[1])[:10])),
            ("geometry", _edge_band_summary(rows, cols, int(height), int(width))),
            ("policy",
             "Zero is NOT assumed to be nodata. A single date reading 0.0 C stays "
             "valid; only a value repeated bit-identically across several dates "
             "is flagged, and it is reported rather than silently masked."),
        ))),
        ("root_cause", root_cause),
        ("failures", failures),
        ("passes", not failures),
    ))
    return report


def assert_daily_raster_contract(report: dict) -> None:
    """Refuse to proceed on a corrupt daily export, with the exact remedy."""
    if report["passes"]:
        return
    fill = report["constant_fill"]
    raise DailyRasterContractError(
        "the exported daily current-period mosaics violate the daily-raster "
        f"validity contract (root cause: {report['root_cause']}).\n  "
        + "\n  ".join(report["failures"])
        + f"\n  flagged pixels: {fill['flagged_pixels']}; geometry: "
          f"{fill['geometry'].get('interpretation')}"
        + "\n  These files cannot be repaired locally: only Earth Engine holds "
          "the true per-date values at the affected pixels, and masking them "
          "would drop pixels the frozen composite legitimately carries. "
          "Re-export ONLY the seven diagnostic daily current-period rasters "
          "with --force-daily-export. No frozen, baseline or production output "
          "is involved."
    )


# =============================================================================
# Reproduction forensics (bounded memory; sparse extremes are never hidden)
# =============================================================================
#: How many worst-value discrepancies to keep, with full per-pixel provenance.
TOP_DISCREPANCY_COUNT = 100


class ReproductionForensics:
    """Localises WHERE and on WHICH DATE a reproduction diverges.

    Percentile summaries hide a defect that touches a few thousand pixels out of
    seven million, which is exactly how an edge-fill artefact survives a p99
    check. This accumulator therefore keeps: the per-date membership of every
    mismatching pixel, a histogram of count differences, and the single worst
    discrepancies with their row/col and date-membership bitmask.
    """

    def __init__(self, dates) -> None:
        import numpy as np

        self.dates = list(dates)
        self.n_dates = len(self.dates)
        self.count_mismatch = 0
        self.false_valid = 0            # reconstruction sees MORE valid dates
        self.false_invalid = 0          # reconstruction sees FEWER valid dates
        self.count_diff_histogram: dict[int, int] = {}
        #: per-date: how often that date is valid at a mismatching pixel
        self.per_date_present = [0] * self.n_dates
        self.per_date_absent = [0] * self.n_dates
        self.mismatch_rows: list[int] = []
        self._top: list[tuple] = []
        self._np = np

    def add_counts(self, frozen_count, reconstructed_count, stack, membership,
                   row_offset: int) -> None:
        np = self._np

        frozen = np.asarray(frozen_count, dtype="float64")
        ours = np.asarray(reconstructed_count, dtype="float64")
        both = np.isfinite(frozen) & np.isfinite(ours)
        mismatch = both & (frozen != ours)
        if not mismatch.any():
            return
        self.count_mismatch += int(mismatch.sum())
        self.false_valid += int((both & (ours > frozen)).sum())
        self.false_invalid += int((both & (ours < frozen)).sum())

        difference = (ours - frozen)[mismatch].astype("int64")
        for value, count in zip(*np.unique(difference, return_counts=True)):
            self.count_diff_histogram[int(value)] = \
                self.count_diff_histogram.get(int(value), 0) + int(count)

        valid = np.isfinite(np.asarray(stack, dtype="float64"))
        for index in range(self.n_dates):
            layer = valid[index][mismatch]
            self.per_date_present[index] += int(layer.sum())
            self.per_date_absent[index] += int((~layer).sum())

        rows = np.where(mismatch)[0] + int(row_offset)
        self.mismatch_rows.extend(int(r) for r in np.unique(rows)[:64])

    def add_values(self, frozen_values, reconstructed_values, stack, membership,
                   row_offset: int) -> None:
        """Keep the worst absolute value discrepancies, with provenance."""
        np = self._np

        frozen = np.asarray(frozen_values, dtype="float64")
        ours = np.asarray(reconstructed_values, dtype="float64")
        both = np.isfinite(frozen) & np.isfinite(ours)
        if not both.any():
            return
        difference = np.where(both, np.abs(frozen - ours), 0.0)
        keep = min(TOP_DISCREPANCY_COUNT, int((difference > 0).sum()))
        if keep <= 0:
            return
        flat = np.argpartition(difference.ravel(), -keep)[-keep:]
        rows, cols = np.unravel_index(flat, difference.shape)
        stack = np.asarray(stack, dtype="float64")
        membership = np.asarray(membership, dtype="float64")
        for r, c in zip(rows, cols):
            magnitude = float(difference[r, c])
            if magnitude <= 0.0:
                continue
            self._top.append((
                magnitude, int(r) + int(row_offset), int(c),
                float(frozen[r, c]), float(ours[r, c]),
                int(membership[r, c]),
                [None if not np.isfinite(stack[k, r, c]) else float(stack[k, r, c])
                 for k in range(self.n_dates)],
            ))
        self._top.sort(key=lambda item: -item[0])
        del self._top[TOP_DISCREPANCY_COUNT:]

    def _membership_labels(self, bitmask: int) -> list[str]:
        return [date for index, date in enumerate(self.dates)
                if bitmask & (1 << index)]

    def classify(self) -> dict:
        """Name the mechanism, rather than reporting an anonymous failure."""
        """if not self.count_mismatch and not self._top:
            return OrderedDict((("root_cause", ROOT_CAUSE_OK),
                                ("explanation", "no discrepancy was observed")))"""

        if not self.count_mismatch:
            if not self._top:
                return OrderedDict((
                    ("root_cause", ROOT_CAUSE_OK),
                    ("explanation", "no discrepancy was observed"),
                ))

            max_abs_difference = max(item[0] for item in self._top)

            if max_abs_difference <= float(REPRODUCTION_TIGHT_REFERENCE_TOL):
                return OrderedDict((
                    ("root_cause", ROOT_CAUSE_FLOAT32_ROUNDOFF),
                    ("explanation",
                    "The frozen and reconstructed composites have identical "
                    "date membership, valid counts and masks. The remaining "
                    f"maximum value difference ({max_abs_difference:.10g}) is "
                    "within the predeclared tight float32 reproduction tolerance "
                    f"({REPRODUCTION_TIGHT_REFERENCE_TOL:.10g}) and is classified "
                    "as numerical round-off, not as a membership or reducer "
                    "mismatch."),
                ))

            return OrderedDict((
                ("root_cause", "value_only_reproduction_mismatch"),
                ("explanation",
                "Date membership, valid counts and masks match exactly, but the "
                "remaining value discrepancy exceeds the tight float32 reference "
                "tolerance. This is a value-only reproduction mismatch, not a "
                "date-membership mismatch."),
            ))
        

        one_sided = self.false_invalid == 0 and self.false_valid > 0
        every_date = all(count == self.count_mismatch
                         for count in self.per_date_present) and self.count_mismatch
        if one_sided and every_date:
            return OrderedDict((
                ("root_cause", ROOT_CAUSE_CONSTANT_FILL),
                ("explanation",
                 "Every mismatching pixel is valid on EVERY date in the "
                 "reconstruction while the frozen composite sees fewer dates, "
                 "and the discrepancy is entirely one-sided. That is a constant "
                 "fill written by the export, not a compositing difference: a "
                 "real QA mask never validates all dates at exactly the pixels "
                 "the canonical chain masks."),
            ))
        if one_sided:
            return OrderedDict((
                ("root_cause", ROOT_CAUSE_CONSTANT_FILL),
                ("explanation",
                 "The reconstruction sees strictly MORE valid dates than the "
                 "frozen composite at every mismatching pixel, which points at "
                 "fill/nodata decoding rather than at the reducer."),
            ))
        return OrderedDict((
            ("root_cause", "daily_membership_or_reducer_mismatch"),
            ("explanation",
             "Mismatches run in both directions, so the daily membership or the "
             "same-day reducer differs from the canonical chain."),
        ))

    def report(self) -> dict:
        return OrderedDict((
            ("count_mismatch_pixels", self.count_mismatch),
            ("false_valid_pixels", self.false_valid),
            ("false_invalid_pixels", self.false_invalid),
            ("count_difference_histogram", OrderedDict(
                (str(key), self.count_diff_histogram[key])
                for key in sorted(self.count_diff_histogram))),
            ("mismatch_by_date", [
                OrderedDict((
                    ("acquisition_date", date),
                    ("valid_in_reconstruction_at_mismatch", self.per_date_present[i]),
                    ("invalid_in_reconstruction_at_mismatch", self.per_date_absent[i]),
                ))
                for i, date in enumerate(self.dates)
            ]),
            ("mismatch_rows_sample", sorted(set(self.mismatch_rows))[:32]),
            ("top_value_discrepancies", [
                OrderedDict((
                    ("abs_difference", magnitude),
                    ("row", row), ("col", col),
                    ("frozen_value", frozen), ("reconstructed_value", ours),
                    ("date_membership_bitmask", bitmask),
                    ("dates_valid_in_reconstruction", self._membership_labels(bitmask)),
                    ("daily_values", values),
                ))
                for magnitude, row, col, frozen, ours, bitmask, values in self._top
            ]),
            ("sparse_extreme_policy",
             "The worst discrepancies are reported in full with their row/col, "
             "per-date values and membership bitmask. A percentile summary alone "
             "would hide a defect confined to a few thousand pixels."),
            ("classification", self.classify()),
        ))


# =============================================================================
# Streaming stage A: reference reproduction from the daily mosaics
# =============================================================================
def _compare_arrays(reference, candidate, accumulator: dict) -> None:
    """Accumulate exact-mask and tolerance evidence for one product window."""
    import numpy as np

    reference = np.asarray(reference, dtype="float64")
    candidate = np.asarray(candidate, dtype="float64")
    ref_valid = np.isfinite(reference)
    cand_valid = np.isfinite(candidate)

    accumulator["total_pixels"] += int(reference.size)
    accumulator["frozen_valid_pixels"] += int(ref_valid.sum())
    accumulator["reproduced_valid_pixels"] += int(cand_valid.sum())
    accumulator["mask_equal_pixels"] += int((ref_valid == cand_valid).sum())
    common = ref_valid & cand_valid
    if not common.any():
        return
    difference = np.abs(reference[common] - candidate[common])
    accumulator["common_pixels"] += int(difference.size)
    accumulator["max_abs_difference"] = max(accumulator["max_abs_difference"],
                                            float(difference.max()))
    accumulator["sum_abs_difference"] += float(difference.sum())
    accumulator["histogram"].add(difference)


def _new_reproduction_accumulator(product: str) -> dict:
    return {
        "product": product,
        "total_pixels": 0,
        "frozen_valid_pixels": 0,
        "reproduced_valid_pixels": 0,
        "mask_equal_pixels": 0,
        "common_pixels": 0,
        "max_abs_difference": 0.0,
        "sum_abs_difference": 0.0,
        "histogram": HistogramAccumulator(HISTOGRAM_MAX[product], bins=HISTOGRAM_BINS),
    }


def _reproduction_report(accumulator: dict) -> dict:
    product = accumulator["product"]
    tolerance = float(REPRODUCTION_TOLERANCES[product])
    common = accumulator["common_pixels"]
    mask_agreement = (
        accumulator["mask_equal_pixels"] / accumulator["total_pixels"]
        if accumulator["total_pixels"] else None
    )
    within = accumulator["max_abs_difference"] <= tolerance
    return OrderedDict((
        ("product", product),
        ("units", PRODUCT_UNITS[product]),
        ("gating_tolerance", tolerance),
        ("gating_tolerance_source",
         "src.landsat_composite_downstream_ab.REPRODUCTION_TOLERANCES"),
        ("reported_tight_tolerance", REPRODUCTION_TIGHT_REFERENCE_TOL),
        ("max_abs_difference", accumulator["max_abs_difference"]),
        ("mean_abs_difference",
         (accumulator["sum_abs_difference"] / common) if common else None),
        ("p99_abs_difference", accumulator["histogram"].quantile(99.0)),
        ("p999_abs_difference", accumulator["histogram"].quantile(99.9)),
        ("common_valid_pixels", common),
        ("frozen_valid_pixels", accumulator["frozen_valid_pixels"]),
        ("reproduced_valid_pixels", accumulator["reproduced_valid_pixels"]),
        ("valid_mask_exactly_equal",
         accumulator["mask_equal_pixels"] == accumulator["total_pixels"]),
        ("valid_mask_agreement", mask_agreement),
        ("within_gating_tolerance", bool(within)),
        ("within_reported_tight_tolerance",
         bool(accumulator["max_abs_difference"] <= REPRODUCTION_TIGHT_REFERENCE_TOL)),
        ("passes", bool(within and accumulator["mask_equal_pixels"]
                        == accumulator["total_pixels"])),
    ))


def run_reference_reproduction(experiment_id: str, root: Path, daily_paths,
                               dates, *, base_dir: Path = PROJECT_ROOT,
                               logger=None) -> dict:
    """Rebuild the frozen date-balanced products from the daily mosaics.

    Everything downstream is compared against THIS reproduction, not against the
    frozen rasters directly, so the reference and the candidate travel through
    one identical numerical pathway and their difference isolates the offsets.
    The gate below proves that pathway lands on the frozen product.
    """
    import numpy as np

    plan = build_input_plan(experiment_id, base_dir)
    thresholds = step5_thresholds()
    grid = reference_grid_path(experiment_id, base_dir)
    height, width = raster_shape(grid)
    profile = output_profile_from(grid)
    rasters = Path(root) / "rasters"

    frozen = {
        TARGET_LST: plan["frozen_reference_current_lst_celsius"]["path"],
        TARGET_CMB: plan["frozen_reference_current_minus_baseline_celsius"]["path"],
        TARGET_ANOMALY: plan["frozen_reference_anomaly_zscore"]["path"],
    }
    accumulators = {p: _new_reproduction_accumulator(p) for p in TARGET_PRODUCTS}
    count_check = ExactComparisonAccumulator("unique_date_valid_count_vs_frozen")
    step5_count_check = ExactComparisonAccumulator("current_valid_count_vs_frozen")
    forensics = ReproductionForensics(dates)

    outputs = OrderedDict((
        (TARGET_LST, rasters / "reference_current_lst_celsius.tif"),
        (TARGET_CMB, rasters / "reference_current_minus_baseline_celsius.tif"),
        (TARGET_ANOMALY, rasters / "reference_anomaly_zscore.tif"),
        ("unique_date_valid_count", rasters / "reference_unique_date_valid_count.tif"),
        ("date_membership_bitmask", rasters / "reference_date_membership_bitmask.tif"),
    ))
    assert_namespace_safe(outputs.values(), experiment_id, base_dir)

    writers = {key: WindowedWriter(path, profile) for key, path in outputs.items()}
    for writer in writers.values():
        writer.__enter__()
    try:
        for start in range(0, height, WINDOW_ROWS):
            stop = min(start + WINDOW_ROWS, height)
            stack = read_daily_stack(daily_paths, start, stop)
            median, valid_count = nanmedian_over_dates(stack)
            membership = date_membership_bitmask(stack)

            frozen_count = read_window(plan["current_unique_date_valid_count"]["path"],
                                       start, stop)
            step5_count = read_window(plan["current_period_valid_count"]["path"],
                                      start, stop)
            # Support is measured on the daily mosaics; the frozen counts are the
            # comparison target, never the source.
            counted = np.where(valid_count > 0, valid_count, np.nan)
            count_check.add(frozen_count, counted)
            step5_count_check.add(step5_count, counted)
            forensics.add_counts(frozen_count, counted, stack, membership, start)

            baseline_mean = read_window(plan["baseline_lst_mean_celsius"]["path"],
                                        start, stop)
            baseline_std = read_window(plan["baseline_lst_std_celsius"]["path"],
                                       start, stop)
            current = apply_step5_current_policy(median, valid_count, thresholds)
            cmb = build_current_minus_baseline(current, baseline_mean)
            anomaly = build_anomaly_zscore(current, baseline_mean, baseline_std,
                                           valid_count, thresholds)

            reproduced = {TARGET_LST: current, TARGET_CMB: cmb, TARGET_ANOMALY: anomaly}
            for product, values in reproduced.items():
                frozen_window = read_window(frozen[product], start, stop)
                _compare_arrays(frozen_window, values, accumulators[product])
                if product == TARGET_LST:
                    forensics.add_values(frozen_window, values, stack, membership,
                                         start)
                writers[product].write(values, start)
            writers["unique_date_valid_count"].write(counted, start)
            writers["date_membership_bitmask"].write(membership, start)

            if logger is not None:
                logger("reference_reproduction", start, stop, height)
    finally:
        for writer in writers.values():
            writer.__exit__(None, None, None)

    products = OrderedDict(
        (product, _reproduction_report(accumulators[product])) for product in TARGET_PRODUCTS
    )

    grid_paths = [grid , *[Path(p) for p in daily_paths]]

    grid_check = OrderedDict((
        ("status", "pass"),
        ("raster_count", len(grid_paths)),
        ("daily_and_frozen_grids_identical", True),
        ("signature", assert_same_grid([grid, *[Path(p) for p in daily_paths]])),
    ))
    count_report = count_check.report()
    step5_report = step5_count_check.report()

    failures = [f"{name}: not reproduced" for name, report in products.items()
                if not report["passes"]]
    if not count_report["passes"]:
        failures.append("unique-date valid count does not reproduce the frozen raster")
    if not step5_report["passes"]:
        failures.append("current valid count does not reproduce the frozen Step5 raster")

    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("reference_composite", REFERENCE_COMPOSITE),
        ("dates", list(dates)),
        ("date_count", len(dates)),
        ("grid_contract", grid_check),
        ("exact_checks", OrderedDict((
            ("grid_signature_equality", True),
            ("valid_mask_equality", all(p["valid_mask_exactly_equal"]
                                        for p in products.values())),
            ("valid_date_count_equality", count_report["passes"]),
        ))),
        ("unique_date_valid_count", count_report),
        ("current_valid_count", step5_report),
        ("products", products),
        ("outputs", OrderedDict((k, str(v)) for k, v in outputs.items())),
        ("forensics", forensics.report()),
        ("versions", dict(DAILY_CONTRACT_VERSIONS)),
        ("daily_raster_hashes", OrderedDict(
            (date, sha256_and_size(Path(path))["sha256"])
            for date, path in zip(dates, daily_paths))),
        ("passes", not failures),
        ("failures", failures),
        ("policy",
         "Reproduction is EXACT for the grid, the valid mask and the valid-date "
         "count, and within the existing project tolerance for the physical "
         "float32 values. A failure ends the experiment at "
         "invalid_reference_reproduction; scientific evaluation is not reached."),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


# =============================================================================
# Streaming stage B: overlap evidence
# =============================================================================
def run_overlap_evidence(daily_paths, height: int, width: int, *, logger=None) -> tuple:
    """Collect every date pair's per-block overlap medians in ONE pass.

    Row windows are aligned to the spatial-block lattice, so each block is
    completed inside exactly one window and its median is computed once over all
    of its common-valid pixels.
    """
    store: dict = {}
    date_valid_counts = [0] * len(daily_paths)

    import numpy as np

    for start in range(0, height, WINDOW_ROWS):
        stop = min(start + WINDOW_ROWS, height)
        stack = read_daily_stack(daily_paths, start, stop)
        for index in range(stack.shape[0]):
            date_valid_counts[index] += int(np.isfinite(stack[index]).sum())
        blocks = block_grid_ids(stop - start, width, start)
        accumulate_pair_block_medians(stack, blocks, store)
        if logger is not None:
            logger("overlap_graph_construction", start, stop, height)
    return store, date_valid_counts


# =============================================================================
# Streaming stage C: harmonised dailies and the candidate composite
# =============================================================================
def run_harmonisation(experiment_id: str, root: Path, daily_paths, dates,
                      alpha_by_date, *, base_dir: Path = PROJECT_ROOT,
                      logger=None) -> dict:
    """Write `Y'_d = Y_d - alpha_d` and the candidate composite + derived products.

    The subtraction is a pure per-date scalar; no neighbourhood, kernel, filter,
    resampling or interpolation is involved anywhere in this function.
    """
    import numpy as np

    plan = build_input_plan(experiment_id, base_dir)
    thresholds = step5_thresholds()
    grid = reference_grid_path(experiment_id, base_dir)
    height, width = raster_shape(grid)
    profile = output_profile_from(grid)
    rasters = Path(root) / "rasters"

    offsets = np.array([float(alpha_by_date[d]) for d in dates], dtype="float64")
    harmonized_paths = [daily_raster_path(root, d, kind="harmonized") for d in dates]
    outputs = OrderedDict((
        (TARGET_LST, rasters / "harmonized_current_lst_celsius.tif"),
        (TARGET_CMB, rasters / "harmonized_current_minus_baseline_celsius.tif"),
        (TARGET_ANOMALY, rasters / "harmonized_anomaly_zscore.tif"),
        ("unique_date_valid_count", rasters / "harmonized_unique_date_valid_count.tif"),
        ("date_membership_bitmask", rasters / "harmonized_date_membership_bitmask.tif"),
        ("difference_lst", rasters / "candidate_minus_reference_current_lst.tif"),
        ("difference_cmb",
         rasters / "candidate_minus_reference_current_minus_baseline.tif"),
        ("difference_anomaly", rasters / "candidate_minus_reference_anomaly.tif"),
    ))
    assert_namespace_safe([*outputs.values(), *harmonized_paths], experiment_id, base_dir)

    reference_products = {
        TARGET_LST: rasters / "reference_current_lst_celsius.tif",
        TARGET_CMB: rasters / "reference_current_minus_baseline_celsius.tif",
        TARGET_ANOMALY: rasters / "reference_anomaly_zscore.tif",
    }
    invariance = {
        "unique_date_valid_count": ExactComparisonAccumulator("unique_date_valid_count"),
        "daily_membership_per_pixel": ExactComparisonAccumulator("daily_membership_per_pixel"),
        "current_valid_count": ExactComparisonAccumulator("current_valid_count"),
        "valid_mask": ExactComparisonAccumulator("valid_mask"),
        "anomaly_valid_mask": ExactComparisonAccumulator("anomaly_valid_mask"),
        "low_current_count_mask": ExactComparisonAccumulator("low_current_count_mask"),
    }
    changes = {product: RasterChangeAccumulator(product) for product in TARGET_PRODUCTS}

    writers = {key: WindowedWriter(path, profile) for key, path in outputs.items()}
    daily_writers = [WindowedWriter(path, profile) for path in harmonized_paths]
    for writer in (*writers.values(), *daily_writers):
        writer.__enter__()
    try:
        for start in range(0, height, WINDOW_ROWS):
            stop = min(start + WINDOW_ROWS, height)
            stack = read_daily_stack(daily_paths, start, stop)
            # THE ONLY intervention: one additive scalar per acquisition date.
            harmonized = stack - offsets[:, None, None]
            for index, writer in enumerate(daily_writers):
                writer.write(harmonized[index], start)

            median, valid_count = nanmedian_over_dates(harmonized)
            membership = date_membership_bitmask(harmonized)
            counted = np.where(valid_count > 0, valid_count, np.nan)

            baseline_mean = read_window(plan["baseline_lst_mean_celsius"]["path"],
                                        start, stop)
            baseline_std = read_window(plan["baseline_lst_std_celsius"]["path"],
                                       start, stop)
            current = apply_step5_current_policy(median, valid_count, thresholds)
            cmb = build_current_minus_baseline(current, baseline_mean)
            anomaly = build_anomaly_zscore(current, baseline_mean, baseline_std,
                                           valid_count, thresholds)
            candidate = {TARGET_LST: current, TARGET_CMB: cmb, TARGET_ANOMALY: anomaly}

            reference = {product: read_window(path, start, stop)
                         for product, path in reference_products.items()}
            reference_membership = read_window(
                rasters / "reference_date_membership_bitmask.tif", start, stop)
            reference_count = read_window(
                rasters / "reference_unique_date_valid_count.tif", start, stop)

            invariance["unique_date_valid_count"].add(reference_count, counted)
            invariance["daily_membership_per_pixel"].add(reference_membership, membership)
            invariance["current_valid_count"].add(
                read_window(plan["current_period_valid_count"]["path"], start, stop),
                counted)
            invariance["valid_mask"].add(
                np.where(np.isfinite(reference[TARGET_LST]), 1.0, np.nan),
                np.where(np.isfinite(current), 1.0, np.nan))
            invariance["anomaly_valid_mask"].add(
                np.where(np.isfinite(reference[TARGET_ANOMALY]), 1.0, np.nan),
                np.where(np.isfinite(anomaly), 1.0, np.nan))
            frozen_low_current = read_window(plan["low_current_count_mask"]["path"],
                                             start, stop)
            candidate_low_current = (
                np.isfinite(counted) &
                (counted < float(thresholds["min_current_valid_count"]))
            ).astype("float64")
            invariance["low_current_count_mask"].add(frozen_low_current,
                                                     candidate_low_current)

            for product in TARGET_PRODUCTS:
                changes[product].add(reference[product], candidate[product])
                writers[product].write(candidate[product], start)
            writers["unique_date_valid_count"].write(counted, start)
            writers["date_membership_bitmask"].write(membership, start)
            writers["difference_lst"].write(current - reference[TARGET_LST], start)
            writers["difference_cmb"].write(cmb - reference[TARGET_CMB], start)
            writers["difference_anomaly"].write(anomaly - reference[TARGET_ANOMALY], start)

            if logger is not None:
                logger("daily_harmonisation", start, stop, height)
    finally:
        for writer in (*writers.values(), *daily_writers):
            writer.__exit__(None, None, None)

    invariance_reports = [invariance[name].report() for name in SUPPORT_INVARIANCE_CHECKS]
    return OrderedDict((
        ("outputs", OrderedDict((k, str(v)) for k, v in outputs.items())),
        ("harmonized_daily_paths", [str(p) for p in harmonized_paths]),
        ("support_invariance", support_invariance_verdict(invariance_reports)),
        ("raster_changes", OrderedDict(
            (product, changes[product].report()) for product in TARGET_PRODUCTS
        )),
        ("spatial_operation_performed", False),
    ))


# =============================================================================
# Streaming stage D: paired boundary-jump analysis
# =============================================================================
#: Window roles required by `rs.build_edge_flags`, resolved from frozen inputs.
FLAG_WINDOW_ROLES = (
    "current_unique_date_valid_count",
    "current_scene_valid_count",
    "current_period_valid_count",
    "baseline_valid_count",
    "current_same_day_multiplicity",
    "low_baseline_std_mask",
    "baseline_lst_std_celsius",
    "low_current_count_mask",
    "low_baseline_count_mask",
)

COVARIATE_ROLES = ("elevation", "slope", "ndvi_current")


def _boundary_masks(flags: dict, codes) -> "OrderedDict[str, object]":
    """Boolean pair mask for every evaluated boundary, on one orientation."""
    masks: "OrderedDict[str, object]" = OrderedDict()
    for boundary in EVALUATED_BOUNDARIES:
        if boundary in flags:
            masks[boundary] = flags[boundary]
        elif boundary in rs.OVERLAP_CODES:
            masks[boundary] = codes == rs.OVERLAP_CODES[boundary]
        else:                                               # pragma: no cover
            raise HarmonizationError(f"unresolvable boundary definition: {boundary!r}")
    return masks


def run_boundary_analysis(experiment_id: str, root: Path, *,
                          base_dir: Path = PROJECT_ROOT, logger=None) -> dict:
    """Accumulate |jump| for reference and candidate on the SAME pair population.

    Adjacency semantics, spatial blocks, matched-control stratification, boundary
    definitions, the near-std epsilon and the path/row rasterization are reused
    verbatim from the residual seam attribution audit. Because the support
    rasters are invariant, the boundary and control pair populations are
    identical between reference and candidate by construction -- the two sides
    differ only in the VALUES they carry.
    """
    import numpy as np

    plan = build_input_plan(experiment_id, base_dir)
    thresholds = step5_thresholds()
    grid = reference_grid_path(experiment_id, base_dir)
    height, width = raster_shape(grid)
    rasters = Path(root) / "rasters"

    sides = OrderedDict((
        ("reference", OrderedDict((
            (TARGET_LST, rasters / "reference_current_lst_celsius.tif"),
            (TARGET_CMB, rasters / "reference_current_minus_baseline_celsius.tif"),
            (TARGET_ANOMALY, rasters / "reference_anomaly_zscore.tif"),
        ))),
        ("candidate", OrderedDict((
            (TARGET_LST, rasters / "harmonized_current_lst_celsius.tif"),
            (TARGET_CMB, rasters / "harmonized_current_minus_baseline_celsius.tif"),
            (TARGET_ANOMALY, rasters / "harmonized_anomaly_zscore.tif"),
        ))),
    ))

    pathrow = resolve_pathrow_availability(experiment_id, base_dir)
    pathrow_masks = None
    if pathrow.get("availability") == "available":
        try:
            import rasterio

            geojson = json.loads(
                Path(pathrow["boundaries_path"]).read_text(encoding="utf-8")
            )
            with rasterio.open(grid) as src:
                transform = src.transform
            pathrow_masks = rasterize_pathrow_boundaries(
                geojson, transform, width, height,
            )
        except Exception as error:                          # noqa: BLE001
            pathrow = dict(pathrow)
            pathrow["availability"] = "unavailable"
            pathrow["reason"] = f"path/row rasterization failed: {error}"
            pathrow_masks = None

    accumulators = {
        side: {
            product: {boundary: StratumAccumulator() for boundary in EVALUATED_BOUNDARIES}
            | {"control": StratumAccumulator()}
            for product in TARGET_PRODUCTS
        }
        for side in sides
    }
    anomaly_jump_histograms = {
        side: HistogramAccumulator(HISTOGRAM_MAX[TARGET_ANOMALY], bins=HISTOGRAM_BINS)
        for side in sides
    }
    pair_counts = {"total": 0, "dropped_invalid_endpoint": 0}
    boundary_pair_counts = {b: 0 for b in EVALUATED_BOUNDARIES}

    for start, stop, horizontal_rows, vertical_rows in iter_row_windows(height,
                                                                       EDGE_WINDOW_ROWS):
        window = {role: read_window(plan[role]["path"], start, stop)
                  for role in FLAG_WINDOW_ROLES}
        for role, entry in plan.items():
            if role.endswith("_unique_date_valid_count") and role.startswith("baseline_"):
                if Path(entry["path"]).exists():
                    window[role] = read_window(entry["path"], start, stop)
        covariates = {
            role: (read_window(plan[role]["path"], start, stop)
                   if Path(plan[role]["path"]).exists() else None)
            for role in COVARIATE_ROLES
        }
        values = {
            side: {product: read_window(path, start, stop)
                   for product, path in products.items()}
            for side, products in sides.items()
        }
        local_pathrow = None
        if pathrow_masks is not None:
            local_pathrow = {"union": pathrow_masks["union"][start:stop, :]}

        for orientation_index, orientation in enumerate(ORIENTATIONS):
            limit = horizontal_rows if orientation == "horizontal" else vertical_rows
            if limit <= 0:
                continue
            flags = build_edge_flags(window, orientation,
                                     epsilon=rs.STD_THRESHOLD_EPSILON_PRIMARY,
                                     thresholds=thresholds, pathrow_masks=local_pathrow)
            flags = {name: array[:limit, :] for name, array in flags.items()}
            codes, support, threshold, pathrow_flag = stratified_class_codes(flags)
            control = control_pair_mask(support, threshold, pathrow_flag)
            masks = _boundary_masks(flags, codes)

            rows, cols = rs._anchor_indices(window[FLAG_WINDOW_ROLES[0]].shape,
                                            orientation)
            rows, cols = rows[:limit, :], cols[:limit, :]
            blocks = spatial_block_ids(rows + start, cols,
                                       block_size=BOOTSTRAP_BLOCK_SIZE_CELLS)

            gradient_bins = []
            for role, edges in (("elevation", rs.ELEVATION_GRADIENT_BINS),
                                ("slope", rs.SLOPE_GRADIENT_BINS),
                                ("ndvi_current", rs.NDVI_GRADIENT_BINS)):
                array = covariates[role]
                if array is None:
                    gradient_bins.append(np.full(rows.shape, -1, dtype="int16"))
                else:
                    gradient_bins.append(
                        gradient_bin(edge_difference(array, orientation)[:limit, :], edges)
                    )
            keys = stratum_keys(blocks, orientation_index, *gradient_bins)

            # A pair is kept only when BOTH sides are finite at BOTH endpoints,
            # so reference and candidate are always compared on the exact same
            # pairs. Nothing is zero-filled.
            valid = None
            jumps = {}
            for side, products in values.items():
                for product, array in products.items():
                    finite = edge_valid_mask(array, orientation=orientation)[:limit, :]
                    valid = finite if valid is None else (valid & finite)
            for side, products in values.items():
                jumps[side] = {
                    product: np.abs(edge_difference(array, orientation)[:limit, :])
                    for product, array in products.items()
                }
            pair_counts["total"] += int(valid.size)
            pair_counts["dropped_invalid_endpoint"] += int((~valid).sum())

            for side in sides:
                for product in TARGET_PRODUCTS:
                    magnitude = jumps[side][product]
                    store = accumulators[side][product]
                    selection = valid & control
                    if selection.any():
                        store["control"].add(keys[selection], magnitude[selection])
                    for boundary, mask in masks.items():
                        selection = valid & mask
                        if selection.any():
                            store[boundary].add(keys[selection], magnitude[selection])
                anomaly_jump_histograms[side].add(
                    jumps[side][TARGET_ANOMALY][valid]
                )
            for boundary, mask in masks.items():
                boundary_pair_counts[boundary] += int((valid & mask).sum())

        if logger is not None:
            logger("boundary_analysis", start, stop, height)

    return OrderedDict((
        ("accumulators", accumulators),
        ("pair_counts", pair_counts),
        ("boundary_pair_counts", boundary_pair_counts),
        ("anomaly_jump_histograms", anomaly_jump_histograms),
        ("pathrow_evidence", pathrow),
        ("pathrow_rasterized", pathrow_masks is not None),
        ("adjacency_semantics", OrderedDict((
            ("orientations", list(ORIENTATIONS)),
            ("zero_filling", False),
            ("dropped_pair_policy",
             "an adjacency pair whose reference OR candidate value is missing at "
             "either endpoint is dropped from BOTH sides, never zero-filled"),
            ("block_size_cells", BOOTSTRAP_BLOCK_SIZE_CELLS),
            ("reused_from", RESIDUAL_SEAM_NAMESPACE),
        ))),
    ))


def evaluate_boundaries(analysis: dict) -> dict:
    """Turn the streamed accumulators into paired reductions and verdicts."""
    accumulators = analysis["accumulators"]
    results: "OrderedDict[str, OrderedDict]" = OrderedDict()
    matching: "OrderedDict[str, OrderedDict]" = OrderedDict()

    for product in TARGET_PRODUCTS:
        results[product] = OrderedDict()
        matching[product] = OrderedDict()
        for boundary, mode in EVALUATED_BOUNDARIES.items():
            if mode == EVAL_MODE_EXCESS:
                reference_boundary, reference_control, reference_diagnostics = (
                    matched_block_accumulators(
                        accumulators["reference"][product][boundary],
                        accumulators["reference"][product]["control"],
                        stratum_space=STRATUM_SPACE)
                )
                candidate_boundary, candidate_control, candidate_diagnostics = (
                    matched_block_accumulators(
                        accumulators["candidate"][product][boundary],
                        accumulators["candidate"][product]["control"],
                        stratum_space=STRATUM_SPACE)
                )
                row = bootstrap_paired_reduction(
                    reference_boundary, reference_control,
                    candidate_boundary, candidate_control, mode=mode)
                matching[product][boundary] = OrderedDict((
                    ("reference", reference_diagnostics),
                    ("candidate", candidate_diagnostics),
                ))
            else:
                reference_boundary = blocks_from_stratum(
                    accumulators["reference"][product][boundary])
                candidate_boundary = blocks_from_stratum(
                    accumulators["candidate"][product][boundary])
                row = bootstrap_paired_reduction(
                    reference_boundary, None, candidate_boundary, None, mode=mode)
                matching[product][boundary] = None
            row["product"] = product
            row["boundary"] = boundary
            row["units"] = PRODUCT_UNITS[product]
            results[product][boundary] = row

    return OrderedDict((
        ("boundary_reductions", results),
        ("matching_diagnostics", matching),
        ("evaluation_modes", dict(EVALUATED_BOUNDARIES)),
    ))


def nonboundary_tradeoff(evaluation: dict) -> dict:
    """Candidate-minus-reference mean absolute jump at non-boundary pairs."""
    rows = []
    for product in TARGET_PRODUCTS:
        row = (evaluation["boundary_reductions"].get(product) or {}).get(
            NONBOUNDARY_CONTROL) or {}
        reference = row.get("reference_boundary_mean_abs_jump")
        candidate = row.get("candidate_boundary_mean_abs_jump")
        rows.append(OrderedDict((
            ("product", product),
            ("units", PRODUCT_UNITS[product]),
            ("control_definition", NONBOUNDARY_CONTROL),
            ("reference_mean_absolute_jump", reference),
            ("candidate_mean_absolute_jump", candidate),
            ("candidate_minus_reference",
             (candidate - reference) if (reference is not None
                                         and candidate is not None) else None),
            ("paired_reduction", row.get("paired_reduction")),
            ("relative_paired_reduction", row.get("relative_paired_reduction")),
            ("interval_low", row.get("interval_low")),
            ("interval_high", row.get("interval_high")),
            ("n_pairs", row.get("n_reference_boundary_pairs")),
            ("n_units", row.get("n_units")),
            ("verdict", row.get("verdict")),
        )))
    return OrderedDict((
        ("rows", rows),
        ("interpretation",
         "A supported reduction at support boundaries is the target. A large "
         "supported reduction HERE too -- on terrain that carries none of the "
         "known boundary mechanisms -- would indicate over-correction rather "
         "than removal of a support artefact. A supported INCREASE here is also "
         "a trade-off. Lower global spatial variability is NOT automatically "
         "better."),
    ))


# =============================================================================
# Tables
# =============================================================================
def write_csv(path: Path, rows, columns) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(str(tmp), str(path))
    return path


RASTER_CHANGE_COLUMNS = (
    "product", "units", "common_valid_pixels", "mean_signed_difference",
    "median_signed_difference", "global_median_shift", "mae", "rmse",
    "p95_absolute_difference", "p99_absolute_difference",
    "reference_valid_pixels", "candidate_valid_pixels", "valid_mask_agreement",
)

BOUNDARY_COLUMNS = (
    "product", "units", "boundary", "mode",
    "reference_boundary_mean_abs_jump", "candidate_boundary_mean_abs_jump",
    "reference_control_mean_abs_jump", "candidate_control_mean_abs_jump",
    "reference_excess_absolute_jump", "candidate_excess_absolute_jump",
    "paired_reduction", "relative_paired_reduction",
    "n_reference_boundary_pairs", "n_candidate_boundary_pairs", "n_control_pairs",
    "n_units", "interval_low", "interval_high", "verdict", "status",
)

BOOTSTRAP_COLUMNS = (
    "product", "boundary", "mode", "unit_type", "block_size_cells",
    "n_units", "n_bootstrap_requested", "n_bootstrap_used", "n_bootstrap_skipped",
    "seed", "ci", "interval_low", "interval_high",
    "identical_draws_for_reference_and_candidate", "resamples_individual_pairs",
    "status", "verdict", "skipped_reason",
)

NONBOUNDARY_COLUMNS = (
    "product", "units", "control_definition", "reference_mean_absolute_jump",
    "candidate_mean_absolute_jump", "candidate_minus_reference",
    "paired_reduction", "relative_paired_reduction", "interval_low",
    "interval_high", "n_pairs", "n_units", "verdict",
)

DATE_NODE_COLUMNS = (
    "acquisition_date", "scene_count", "temporal_observations", "scene_ids",
    "path_rows", "wrs_paths", "valid_pixel_count", "graph_degree",
    "is_articulation_node", "component_index",
)

DATE_EDGE_COLUMNS = (
    "date_i", "date_j", "eligible", "common_valid_pixels", "independent_blocks",
    "blocks_seen", "blocks_below_min_pixels", "edge_median_difference_celsius",
    "edge_mad_celsius", "edge_sigma_celsius", "edge_standard_error_celsius",
    "edge_standardised_residual_measure", "block_median_min", "block_median_max",
    "spatial_coverage_fraction", "date_i_path_rows", "date_j_path_rows",
    "shares_wrs_path", "weight", "fitted_difference_celsius", "residual_celsius",
)

DATE_OFFSET_COLUMNS = (
    "acquisition_date", "alpha_celsius", "standard_error_celsius",
    "valid_observation_count", "graph_degree",
)

SENSITIVITY_COLUMNS = (
    "threshold_label", "min_common_pixels", "min_independent_blocks",
    "edge_count", "rejected_edge_count", "connected_component_count",
    "connected", "acquisition_date", "alpha_celsius", "max_abs_offset_celsius",
    "median_abs_offset_celsius", "weighted_mean_offset_celsius",
    "edge_residual_rms_celsius", "graph_condition_number", "solver_status",
    "used_for_primary_candidate",
)


def raster_change_rows(changes: dict) -> list[dict]:
    rows = []
    for product in TARGET_PRODUCTS:
        report = changes.get(product) or {}
        row = {key: report.get(key) for key in RASTER_CHANGE_COLUMNS}
        for threshold, value in (report.get("fraction_above") or {}).items():
            row[f"fraction_above_{threshold}"] = value
        rows.append(row)
    return rows


def raster_change_columns(changes: dict) -> list[str]:
    extra: list[str] = []
    for product in TARGET_PRODUCTS:
        for threshold in (changes.get(product, {}).get("fraction_above") or {}):
            name = f"fraction_above_{threshold}"
            if name not in extra:
                extra.append(name)
    return [*RASTER_CHANGE_COLUMNS, *extra]


def boundary_rows(evaluation: dict) -> list[dict]:
    rows = []
    for product in TARGET_PRODUCTS:
        for boundary, mode in EVALUATED_BOUNDARIES.items():
            row = dict((evaluation["boundary_reductions"][product][boundary]))
            row["mode"] = mode
            rows.append(row)
    return rows


def date_edge_rows(graph: dict, solution: dict | None) -> list[dict]:
    residuals = {}
    weights = {}
    if solution:
        for residual in solution["edge_residuals"]:
            residuals[(residual["date_i"], residual["date_j"])] = residual
        for edge, weight in zip(graph["edges"], solution["edge_weights"]["capped"]):
            weights[(edge["date_i"], edge["date_j"])] = weight
    rows = []
    for edge in [*graph["edges"], *graph["rejected_edges"]]:
        row = dict(edge)
        key = (edge["date_i"], edge["date_j"])
        row["weight"] = weights.get(key)
        residual = residuals.get(key)
        row["fitted_difference_celsius"] = (
            residual["fitted_difference_celsius"] if residual else None
        )
        row["residual_celsius"] = residual["residual_celsius"] if residual else None
        rows.append(row)
    rows.sort(key=lambda r: (not r["eligible"], r["date_i"], r["date_j"]))
    return rows


def date_node_rows(dates, inventory, diagnostics: dict,
                   date_valid_counts) -> list[dict]:
    component_of = {}
    for component in diagnostics["components"]:
        for date in component["dates"]:
            component_of[date] = component["component_index"]
    articulation = set(diagnostics["articulation_nodes"])
    rows = []
    for index, date in enumerate(dates):
        entry = inventory[date]
        rows.append({
            "acquisition_date": date,
            "scene_count": entry["scene_count"],
            "temporal_observations": entry["temporal_observations"],
            "scene_ids": ";".join(entry["scene_ids"]),
            "path_rows": ";".join(entry["path_rows"]),
            "wrs_paths": ";".join(entry["wrs_paths"]),
            "valid_pixel_count": date_valid_counts[index],
            "graph_degree": diagnostics["degree_per_date"].get(date),
            "is_articulation_node": date in articulation,
            "component_index": component_of.get(date),
        })
    return rows


def sensitivity_rows(sensitivity) -> list[dict]:
    rows = []
    for entry in sensitivity:
        graph = entry["graph"]
        solution = entry.get("solution")
        diagnostics = entry["diagnostics"]
        base = {
            "threshold_label": entry["label"],
            "min_common_pixels": graph["min_common_pixels"],
            "min_independent_blocks": graph["min_independent_blocks"],
            "edge_count": graph["edge_count"],
            "rejected_edge_count": graph["rejected_edge_count"],
            "connected_component_count": diagnostics["connected_component_count"],
            "connected": diagnostics["connected"],
            "used_for_primary_candidate": entry["label"] == "primary",
        }
        if not solution:
            rows.append({**base, "acquisition_date": None, "alpha_celsius": None,
                         "solver_status": "not_solved_graph_disconnected"})
            continue
        for offset in solution["offsets"]:
            rows.append({
                **base,
                "acquisition_date": offset["acquisition_date"],
                "alpha_celsius": offset["alpha_celsius"],
                "max_abs_offset_celsius": solution["max_abs_offset_celsius"],
                "median_abs_offset_celsius": solution["median_abs_offset_celsius"],
                "weighted_mean_offset_celsius": solution["weighted_mean_offset_celsius"],
                "edge_residual_rms_celsius": solution["edge_residual_rms_celsius"],
                "graph_condition_number": solution["graph_condition_number"],
                "solver_status": solution["solver_status"],
            })
    return rows


# =============================================================================
# Summary assembly
# =============================================================================
def build_evidence(reproduction: dict, diagnostics: dict, solution: dict | None,
                   invariance: dict, changes: dict, evaluation: dict,
                   *, inputs_valid: bool, invalid_reasons=None) -> dict:
    """Everything the ordered decision rule reads, and nothing else."""
    return OrderedDict((
        ("inputs_valid", bool(inputs_valid)),
        ("invalid_input_reasons", list(invalid_reasons or [])),
        ("reference_reproduction_passes", bool(reproduction.get("passes"))),
        ("reference_reproduction_failures", list(reproduction.get("failures") or [])),
        ("primary_graph_connected", bool(diagnostics.get("connected"))),
        ("graph_failure_reasons", (
            [] if diagnostics.get("connected") else [
                f"{diagnostics.get('connected_component_count')} connected components "
                f"over {diagnostics.get('date_count')} dates; isolated dates: "
                f"{diagnostics.get('isolated_dates')}"
            ]
        )),
        ("support_invariance_passes", bool(invariance.get("passes"))),
        ("support_invariance_failures", list(invariance.get("failed_checks") or [])),
        ("boundary_reductions", evaluation.get("boundary_reductions") or {}),
        ("global_median_current_lst_shift",
         (changes.get(TARGET_LST) or {}).get("global_median_shift")),
        ("max_abs_date_offset",
         solution.get("max_abs_offset_celsius") if solution else None),
        ("offset_estimation_stable",
         bool(solution.get("estimation_stable")) if solution else False),
        ("offset_instability_reasons",
         list(solution.get("instability_reasons") or []) if solution
         else ["no offset solution was produced"]),
    ))


def build_summary(experiment_id: str, *, state: dict, config: dict,
                  provenance: dict, inventory, reproduction: dict,
                  graph: dict, diagnostics: dict, solution: dict | None,
                  sensitivity, invariance: dict, changes: dict,
                  evaluation: dict, tradeoff: dict, decision: dict,
                  resources: dict) -> dict:
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("reference_composite", REFERENCE_COMPOSITE),
        ("candidate_composite", CANDIDATE_COMPOSITE),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("final_status", decision["final_status"]),
        ("final_status_meaning", decision["final_status_meaning"]),
        ("seam_fixed", False),
        ("production_approved", False),
        ("production_ready", False),
        ("changes_production_reducer", CHANGES_PRODUCTION_REDUCER),
        ("smoothing_applied", SMOOTHING_APPLIED),
        ("spatial_interpolation_applied", SPATIAL_INTERPOLATION_APPLIED),
        ("baseline_recomputed", RECOMPUTES_BASELINE),
        ("labels_used", USES_LABELS),
        ("step8_metrics_used", USES_STEP8_METRICS),
        ("model_performance_used", USES_MODEL_PERFORMANCE),
        ("decision", decision),
        ("configuration", config),
        ("technical_validity", OrderedDict((
            ("upstream_prerequisites_met", bool(state.get("prerequisites_met"))),
            ("upstream_state", state),
            ("missing_required_inputs", provenance["missing_required_inputs"]),
            ("missing_optional_inputs", provenance["missing_optional_inputs"]),
            ("grid_contract", reproduction["grid_contract"]),
            ("earth_engine_used_in_analysis", False),
            ("step6_step7_step8_rerun", False),
        ))),
        ("frozen_reference_reproduction", reproduction),
        ("date_overlap_graph", OrderedDict((
            ("date_count", diagnostics["date_count"]),
            ("dates", diagnostics["dates"]),
            ("edge_count", diagnostics["edge_count"]),
            ("rejected_edge_count", diagnostics["rejected_edge_count"]),
            ("connected_component_count", diagnostics["connected_component_count"]),
            ("connected", diagnostics["connected"]),
            ("components", diagnostics["components"]),
            ("degree_per_date", diagnostics["degree_per_date"]),
            ("isolated_dates", diagnostics["isolated_dates"]),
            ("articulation_nodes", diagnostics["articulation_nodes"]),
            ("cycle_consistency", OrderedDict(
                (k, v) for k, v in diagnostics["cycle_consistency"].items()
                if k != "cycles")),
            ("eligibility", OrderedDict((
                ("min_common_pixels", graph["min_common_pixels"]),
                ("min_independent_blocks", graph["min_independent_blocks"]),
                ("min_block_common_pixels", graph["min_block_common_pixels"]),
                ("selection_policy", THRESHOLD_SELECTION_POLICY),
            ))),
            ("drop_policy", diagnostics["drop_policy"]),
        ))),
        ("fitted_date_offsets", (
            OrderedDict((k, v) for k, v in solution.items()
                        if k not in ("edge_residuals",))
            if solution else None
        )),
        ("date_offset_sensitivity", [
            OrderedDict((
                ("label", entry["label"]),
                ("min_common_pixels", entry["graph"]["min_common_pixels"]),
                ("min_independent_blocks", entry["graph"]["min_independent_blocks"]),
                ("edge_count", entry["graph"]["edge_count"]),
                ("connected", entry["diagnostics"]["connected"]),
                ("connected_component_count",
                 entry["diagnostics"]["connected_component_count"]),
                ("max_abs_offset_celsius",
                 (entry.get("solution") or {}).get("max_abs_offset_celsius")),
                ("weighted_mean_offset_celsius",
                 (entry.get("solution") or {}).get("weighted_mean_offset_celsius")),
                ("edge_residual_rms_celsius",
                 (entry.get("solution") or {}).get("edge_residual_rms_celsius")),
                ("used_for_primary_candidate", entry["label"] == "primary"),
            ))
            for entry in sensitivity
        ]),
        ("support_invariance", invariance),
        ("raster_changes", changes),
        ("support_boundary_reductions", evaluation["boundary_reductions"]),
        ("matching_diagnostics", evaluation["matching_diagnostics"]),
        ("nonboundary_tradeoff", tradeoff),
        ("pathrow_check", OrderedDict((
            ("evidence", evaluation.get("pathrow_evidence")),
            ("pathrow_only_reductions", OrderedDict(
                (product, evaluation["boundary_reductions"][product].get(
                    rs.CLASS_PATHROW_ONLY))
                for product in TARGET_PRODUCTS)),
            ("qualification",
             "Path/row evidence is METADATA-derived. `pathrow_only` strictly "
             "excludes support and threshold pairs, so it is independent of the "
             "support overlap this experiment targets."),
        ))),
        ("daily_mosaic_inventory", [
            OrderedDict((
                ("acquisition_date", date),
                ("scene_count", entry["scene_count"]),
                ("temporal_observations", entry["temporal_observations"]),
                ("scene_ids", entry["scene_ids"]),
                ("path_rows", entry["path_rows"]),
            ))
            for date, entry in inventory.items()
        ]),
        ("resources", resources),
        ("limitations", required_limitations()),
        ("inherited_limitations", inherited_limitations(state)),
        ("next_experiment", next_experiment_text(decision["final_status"])),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


# =============================================================================
# Markdown report
# =============================================================================
def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _interval_text(row: dict) -> str:
    low, high = row.get("interval_low"), row.get("interval_high")
    if low is None or high is None:
        return "n/a"
    return f"[{low:.4f}, {high:.4f}]"


def render_summary_markdown(summary: dict) -> str:
    """The twelve required sections, generated from the summary payload ONLY.

    Report generation is a pure read of the already-computed metrics: it never
    recomputes, rounds into, or otherwise alters a scientific value.
    """
    decision = summary["decision"]
    lines: list[str] = []
    add = lines.append

    add(f"# Current-period date-offset harmonization counterfactual "
        f"({summary['experiment_id']})")
    add("")
    add(f"- reference composite: `{summary['reference_composite']}`")
    add(f"- candidate composite: `{summary['candidate_composite']}`")
    add(f"- final status: **`{summary['final_status']}`**")
    add(f"- report schema: `{summary['report_schema_version']}`; "
        f"decision rule: `{summary['decision_rule_version']}`")
    add("")
    add("> This is a DIAGNOSTIC counterfactual. It is not a fix, it does not "
        "change the production reducer, it does not recompute the baseline, and "
        "it never smooths, blends or interpolates any raster.")
    add("")

    # 1 -----------------------------------------------------------------
    technical = summary["technical_validity"]
    add("## 1. Technical validity")
    add("")
    add(f"- upstream prerequisites met: {_fmt(technical['upstream_prerequisites_met'])}")
    for key in ("counterfactual_final_status", "downstream_ab_final_status",
                "residual_seam_final_status", "downstream_ab_reference_reproduction",
                "baseline_invariance"):
        add(f"  - `{key}`: `{technical['upstream_state'].get(key)}`")
    add(f"- missing required inputs: {technical['missing_required_inputs'] or 'none'}")
    add(f"- missing optional inputs: {len(technical['missing_optional_inputs'])}")
    add(f"- grid contract: `{technical['grid_contract']['status']}` over "
        f"{technical['grid_contract'].get('raster_count', 'n/a')} rasters")
    add(f"- Earth Engine used in analysis: {_fmt(technical['earth_engine_used_in_analysis'])}")
    add(f"- Step6/Step7/Step8 re-run: {_fmt(technical['step6_step7_step8_rerun'])}")
    add("")

    # 2 -----------------------------------------------------------------
    reproduction = summary["frozen_reference_reproduction"]
    add("## 2. Frozen reference reproduction")
    add("")
    add(f"- daily mosaics: {reproduction['date_count']} unique acquisition dates")
    add(f"- exact checks: " + ", ".join(
        f"`{k}`={_fmt(v)}" for k, v in reproduction["exact_checks"].items()))
    add("")
    add("| product | max abs diff | gating tol | mask exactly equal | passes |")
    add("| --- | --- | --- | --- | --- |")
    for product, report in reproduction["products"].items():
        add(f"| `{product}` | {_fmt(report['max_abs_difference'], 8)} | "
            f"{_fmt(report['gating_tolerance'], 8)} | "
            f"{_fmt(report['valid_mask_exactly_equal'])} | "
            f"{_fmt(report['passes'])} |")
    add("")
    add(f"- unique-date valid count reproduces exactly: "
        f"{_fmt(reproduction['unique_date_valid_count']['passes'])} "
        f"(unequal pixels: {reproduction['unique_date_valid_count']['unequal_pixel_count']})")
    add(f"- overall: **{_fmt(reproduction['passes'])}**")
    add("")

    # 3 -----------------------------------------------------------------
    graph = summary["date_overlap_graph"]
    add("## 3. Date-overlap graph")
    add("")
    add(f"- dates: {graph['date_count']}; eligible edges: {graph['edge_count']}; "
        f"rejected: {graph['rejected_edge_count']}")
    add(f"- connected components: {graph['connected_component_count']}; "
        f"connected: **{_fmt(graph['connected'])}**")
    add(f"- eligibility (PRIMARY): >= {graph['eligibility']['min_common_pixels']} "
        f"common valid pixels and >= "
        f"{graph['eligibility']['min_independent_blocks']} independent "
        f"{BOOTSTRAP_BLOCK_SIZE_CELLS}-cell blocks")
    add(f"- articulation nodes: {graph['articulation_nodes'] or 'none'}")
    add(f"- isolated dates: {graph['isolated_dates'] or 'none'}")
    cycles = graph["cycle_consistency"]
    add(f"- independent cycles: {cycles['independent_cycle_count']}; "
        f"max |closure error|: {_fmt(cycles['max_abs_closure_error_celsius'])} C; "
        f"median: {_fmt(cycles['median_abs_closure_error_celsius'])} C")
    add("")
    add("| component | dates | valid-observation fraction |")
    add("| --- | --- | --- |")
    for component in graph["components"]:
        add(f"| {component['component_index']} | "
            f"{', '.join(component['dates'])} | "
            f"{_fmt(component['valid_observation_fraction'])} |")
    add("")
    add(f"_{graph['drop_policy']}_")
    add("")

    # 4 -----------------------------------------------------------------
    add("## 4. Fitted date offsets")
    add("")
    solution = summary["fitted_date_offsets"]
    if not solution:
        add("No offset solution was produced: the primary graph did not pass the "
            "connectivity gate, so no harmonized candidate raster is presented "
            "as valid.")
    else:
        add(f"- solver: {solution['solver']} (`{solution['solver_status']}`)")
        add(f"- identifying constraint: {solution['identifying_constraint']}")
        add(f"- weighted mean offset: {_fmt(solution['weighted_mean_offset_celsius'], 10)} C "
            f"(is zero: {_fmt(solution['weighted_mean_offset_is_zero'])})")
        add(f"- max |alpha|: {_fmt(solution['max_abs_offset_celsius'])} C; "
            f"median |alpha|: {_fmt(solution['median_abs_offset_celsius'])} C")
        add(f"- edge residual RMS: {_fmt(solution['edge_residual_rms_celsius'])} C; "
            f"max: {_fmt(solution['edge_residual_max_abs_celsius'])} C")
        add(f"- graph condition number: {_fmt(solution['graph_condition_number'], 3)}; "
            f"degrees of freedom: {solution['degrees_of_freedom']}")
        add(f"- estimation stable: **{_fmt(solution['estimation_stable'])}** "
            f"{solution['instability_reasons'] or ''}")
        add(f"- edge weights capped at {_fmt(solution['edge_weights'].get('cap'), 3)} "
            f"({solution['edge_weights'].get('capped_edge_count')} edge(s) capped)")
        add("")
        add("| acquisition date | alpha (C) | standard error (C) | degree | valid pixels |")
        add("| --- | --- | --- | --- | --- |")
        for offset in solution["offsets"]:
            add(f"| {offset['acquisition_date']} | "
                f"{_fmt(offset['alpha_celsius'])} | "
                f"{_fmt(offset['standard_error_celsius'])} | "
                f"{offset['graph_degree']} | "
                f"{_fmt(offset['valid_observation_count'], 0)} |")
    add("")
    add("### Threshold sensitivity (reported unconditionally)")
    add("")
    add("| threshold set | edges | components | connected | max abs alpha (C) | primary |")
    add("| --- | --- | --- | --- | --- | --- |")
    for entry in summary["date_offset_sensitivity"]:
        add(f"| `{entry['label']}` ({entry['min_common_pixels']}/"
            f"{entry['min_independent_blocks']}) | {entry['edge_count']} | "
            f"{entry['connected_component_count']} | {_fmt(entry['connected'])} | "
            f"{_fmt(entry['max_abs_offset_celsius'])} | "
            f"{_fmt(entry['used_for_primary_candidate'])} |")
    add("")

    # 5 -----------------------------------------------------------------
    invariance = summary["support_invariance"]
    add("## 5. Support invariance")
    add("")
    add("| check | unequal pixels | changed valid pixels | mask agreement | passes |")
    add("| --- | --- | --- | --- | --- |")
    for check in invariance["checks"]:
        add(f"| `{check['check']}` | {check['unequal_pixel_count']} | "
            f"{check['changed_valid_pixel_count']} | "
            f"{_fmt(check['mask_agreement'], 6)} | {_fmt(check['passes'])} |")
    add("")
    add(f"- overall: **{_fmt(invariance['passes'])}**; failed checks: "
        f"{invariance['failed_checks'] or 'none'}")
    add(f"- _{invariance['purpose']}_")
    add("")

    # 6 -----------------------------------------------------------------
    add("## 6. Raster changes (candidate minus reference)")
    add("")
    add("| product | mean signed | median signed | MAE | RMSE | p95 abs | p99 abs | mask agreement |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for product in TARGET_PRODUCTS:
        row = summary["raster_changes"][product]
        add(f"| `{product}` | {_fmt(row['mean_signed_difference'])} | "
            f"{_fmt(row['median_signed_difference'])} | {_fmt(row['mae'])} | "
            f"{_fmt(row['rmse'])} | {_fmt(row['p95_absolute_difference'])} | "
            f"{_fmt(row['p99_absolute_difference'])} | "
            f"{_fmt(row['valid_mask_agreement'], 6)} |")
    add("")
    for product in TARGET_PRODUCTS:
        row = summary["raster_changes"][product]
        fractions = ", ".join(
            f"{threshold}: {_fmt(value, 6)}"
            for threshold, value in (row["fraction_above"] or {}).items())
        add(f"- `{product}` fraction above ({PRODUCT_UNITS[product]}): {fractions}")
    add("")
    add("_These are CHANGES, not errors: the reference is a frozen diagnostic "
        "composite, not ground truth._")
    add("")

    # 7 -----------------------------------------------------------------
    add("## 7. Support-boundary reductions")
    add("")
    for product in TARGET_PRODUCTS:
        add(f"### `{product}` ({PRODUCT_UNITS[product]})")
        add("")
        add("| boundary | ref excess | cand excess | reduction | rel. | 95% interval | pairs | blocks | verdict |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for boundary in EVALUATED_BOUNDARIES:
            row = summary["support_boundary_reductions"][product][boundary]
            add(f"| `{boundary}` | "
                f"{_fmt(row['reference_excess_absolute_jump'])} | "
                f"{_fmt(row['candidate_excess_absolute_jump'])} | "
                f"{_fmt(row['paired_reduction'])} | "
                f"{_fmt(row['relative_paired_reduction'], 3)} | "
                f"{_interval_text(row)} | "
                f"{row['n_reference_boundary_pairs']} | {row['n_units']} | "
                f"`{row['verdict']}` |")
        add("")
    add(f"_{BOOTSTRAP_UNIT_POLICY}_")
    add("")

    # 8 -----------------------------------------------------------------
    tradeoff = summary["nonboundary_tradeoff"]
    add("## 8. Non-boundary trade-offs")
    add("")
    add("| product | reference mean abs jump | candidate mean abs jump | candidate - reference | 95% interval | verdict |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in tradeoff["rows"]:
        add(f"| `{row['product']}` | {_fmt(row['reference_mean_absolute_jump'])} | "
            f"{_fmt(row['candidate_mean_absolute_jump'])} | "
            f"{_fmt(row['candidate_minus_reference'])} | "
            f"{_interval_text(row)} | `{row['verdict']}` |")
    add("")
    add(f"_{tradeoff['interpretation']}_")
    add("")

    # 9 -----------------------------------------------------------------
    pathrow = summary["pathrow_check"]
    add("## 9. Path/row check")
    add("")
    evidence = pathrow["evidence"] or {}
    add(f"- availability: `{evidence.get('availability')}` "
        f"({evidence.get('reason')})")
    add(f"- distinct metadata interfaces: {evidence.get('interface_count')}")
    add("")
    add("| product | ref excess | cand excess | reduction | 95% interval | verdict |")
    add("| --- | --- | --- | --- | --- | --- |")
    for product, row in pathrow["pathrow_only_reductions"].items():
        row = row or {}
        add(f"| `{product}` | {_fmt(row.get('reference_excess_absolute_jump'))} | "
            f"{_fmt(row.get('candidate_excess_absolute_jump'))} | "
            f"{_fmt(row.get('paired_reduction'))} | {_interval_text(row)} | "
            f"`{row.get('verdict')}` |")
    add("")
    add(f"_{pathrow['qualification']}_")
    add("")

    # 10 ----------------------------------------------------------------
    add("## 10. Decision")
    add("")
    add(f"- **`{summary['final_status']}`** -- {summary['final_status_meaning']}")
    add("")
    add("Reasons:")
    for reason in decision["reasons"]:
        add(f"- {reason}")
    add("")
    add(f"- seam_fixed: {_fmt(summary['seam_fixed'])}; production_approved: "
        f"{_fmt(summary['production_approved'])}; changes production reducer: "
        f"{_fmt(summary['changes_production_reducer'])}")
    add(f"- labels used: {_fmt(summary['labels_used'])}; Step8 metrics used: "
        f"{_fmt(summary['step8_metrics_used'])}; model performance used: "
        f"{_fmt(summary['model_performance_used'])}")
    add(f"- smoothing applied: {_fmt(summary['smoothing_applied'])}; spatial "
        f"interpolation applied: {_fmt(summary['spatial_interpolation_applied'])}")
    add("")
    add(f"_Decision rule ({summary['decision_rule_version']}): "
        f"{decision['decision_rule']}_")
    add("")

    # 11 ----------------------------------------------------------------
    add("## 11. Limitations")
    add("")
    for limitation in summary["limitations"]:
        add(f"- {limitation}")
    add("")
    for limitation in summary["inherited_limitations"]:
        add(f"- {limitation}")
    add("")

    # 12 ----------------------------------------------------------------
    add("## 12. Next experiment")
    add("")
    add(summary["next_experiment"])
    add("")
    return "\n".join(lines)


def report_generation_preserves_metrics(before: dict, after: dict) -> bool:
    """Report generation must not alter a single scientific value."""
    return json.dumps(before, sort_keys=True, default=str) == json.dumps(
        after, sort_keys=True, default=str)


def _scrub_declared_prohibitions(payload):
    """Drop the keys that legitimately NAME a forbidden conclusion to forbid it."""
    declared = {"forbidden_conclusions", "allowed_final_statuses", "seam_fixed",
                "production_approved", "production_ready", "decision_rule",
                "final_status_rule", "claim_boundary"}
    if isinstance(payload, dict):
        return {k: _scrub_declared_prohibitions(v) for k, v in payload.items()
                if k not in declared}
    if isinstance(payload, list):
        return [_scrub_declared_prohibitions(v) for v in payload]
    return payload


def summary_forbids_banned_conclusions(payload) -> bool:
    """No report may CLAIM a forbidden conclusion anywhere."""
    text = json.dumps(_scrub_declared_prohibitions(payload), default=str)
    return not any(banned in text for banned in FORBIDDEN_CONCLUSIONS)


# =============================================================================
# Manifest
# =============================================================================
MANIFEST_EXCLUDED_SUBTREES = ("_tiles", "_tiles_resume")


def manifest_candidate_files(root: Path) -> list[Path]:
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in MANIFEST_EXCLUDED_SUBTREES for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def build_manifest(experiment_id: str, root: Path, summary: dict) -> dict:
    entries = []
    for path in manifest_candidate_files(root):
        signed = sha256_and_size(path)
        entries.append(OrderedDict((
            ("path", str(path.relative_to(root))),
            ("bytes", signed["bytes"]),
            ("sha256", signed["sha256"]),
        )))
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("final_status", summary["final_status"]),
        ("output_root", str(root)),
        ("file_count", len(entries)),
        ("files", entries),
        ("frozen_namespaces_written", 0),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


# =============================================================================
# Maps (fixed scale, directly comparable, never smoothed before plotting)
# =============================================================================
#: Display rasters are plotted with nearest-neighbour interpolation ONLY. No
#: filter, resample or antialiasing pass may touch a scientific raster before it
#: is drawn -- a smoothed display would hide exactly the seam under study.
MAP_INTERPOLATION = "nearest"
MAP_DPI = 140


def _read_full(path: Path):
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        array = src.read(1, masked=True).astype("float64").filled(np.nan)
    return np.where(array == NODATA_SENTINEL, np.nan, array)


def _shared_limits(*arrays, percentile: float = 2.0):
    import numpy as np

    values = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays
                             if np.isfinite(a).any()]) if arrays else None
    if values is None or not values.size:
        return None, None
    return (float(np.percentile(values, percentile)),
            float(np.percentile(values, 100.0 - percentile)))


def _symmetric_limit(array, percentile: float = 99.0):
    import numpy as np

    finite = array[np.isfinite(array)]
    if not finite.size:
        return 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    return limit if limit > 0 else 1.0


def _save_single(path: Path, array, *, title: str, cmap: str,
                 vmin=None, vmax=None, label: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8.0, 6.4))
    image = axes.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation=MAP_INTERPOLATION)
    axes.set_title(title, fontsize=10)
    axes.set_xticks([])
    axes.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.82, label=label)
    figure.tight_layout()
    tmp = path.parent / f".{path.name}.tmp.png"
    figure.savefig(tmp, dpi=MAP_DPI)
    plt.close(figure)
    os.replace(str(tmp), str(path))
    return path


def render_product_maps(root: Path, *, logger=None) -> list[Path]:
    """Reference / candidate / difference triples on IDENTICAL colour limits."""
    rasters = Path(root) / "rasters"
    maps = Path(root) / "maps"
    written: list[Path] = []

    triples = (
        (TARGET_LST, "reference_current_lst_celsius.tif",
         "harmonized_current_lst_celsius.tif",
         "candidate_minus_reference_current_lst.tif",
         ("reference_current_lst.png", "harmonized_current_lst.png",
          "difference_current_lst.png"), "current LST", "C", "inferno"),
        (TARGET_CMB, "reference_current_minus_baseline_celsius.tif",
         "harmonized_current_minus_baseline_celsius.tif",
         "candidate_minus_reference_current_minus_baseline.tif",
         ("reference_current_minus_baseline.png",
          "harmonized_current_minus_baseline.png",
          "difference_current_minus_baseline.png"),
         "current minus baseline", "C", "coolwarm"),
        (TARGET_ANOMALY, "reference_anomaly_zscore.tif",
         "harmonized_anomaly_zscore.tif",
         "candidate_minus_reference_anomaly.tif",
         ("reference_anomaly.png", "harmonized_anomaly.png",
          "difference_anomaly.png"), "anomaly z-score", "z", "coolwarm"),
    )
    for _product, ref_name, cand_name, diff_name, out_names, title, unit, cmap in triples:
        reference = _read_full(rasters / ref_name)
        candidate = _read_full(rasters / cand_name)
        difference = _read_full(rasters / diff_name)
        vmin, vmax = _shared_limits(reference, candidate)
        written.append(_save_single(
            maps / out_names[0], reference,
            title=f"{title} -- {REFERENCE_COMPOSITE}", cmap=cmap,
            vmin=vmin, vmax=vmax, label=unit))
        written.append(_save_single(
            maps / out_names[1], candidate,
            title=f"{title} -- {CANDIDATE_COMPOSITE}", cmap=cmap,
            vmin=vmin, vmax=vmax, label=unit))
        limit = _symmetric_limit(difference)
        written.append(_save_single(
            maps / out_names[2], difference,
            title=f"{title} -- candidate minus reference", cmap="RdBu_r",
            vmin=-limit, vmax=limit, label=unit))
        if logger is not None:
            logger("maps", 0, 1, 1)
    return written


def render_support_boundary_maps(experiment_id: str, root: Path,
                                 *, base_dir: Path = PROJECT_ROOT) -> list[Path]:
    """Current-support boundaries drawn over the reference and the candidate."""
    import numpy as np

    plan = build_input_plan(experiment_id, base_dir)
    rasters = Path(root) / "rasters"
    maps = Path(root) / "maps"

    unique = _read_full(plan["current_unique_date_valid_count"]["path"])
    scene = _read_full(plan["current_scene_valid_count"]["path"])
    step5 = _read_full(plan["current_period_valid_count"]["path"])
    boundary = np.zeros(unique.shape, dtype=bool)
    for array in (unique, scene, step5):
        horizontal = rs.edge_change_flag(array, "horizontal")
        vertical = rs.edge_change_flag(array, "vertical")
        boundary[:, :-1] |= horizontal
        boundary[:-1, :] |= vertical

    reference = _read_full(rasters / "reference_current_lst_celsius.tif")
    candidate = _read_full(rasters / "harmonized_current_lst_celsius.tif")
    vmin, vmax = _shared_limits(reference, candidate)

    written = []
    for name, array, label in (
        ("current_support_boundaries_over_reference.png", reference, REFERENCE_COMPOSITE),
        ("current_support_boundaries_over_candidate.png", candidate, CANDIDATE_COMPOSITE),
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = maps / name
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(figsize=(8.0, 6.4))
        image = axes.imshow(array, cmap="inferno", vmin=vmin, vmax=vmax,
                            interpolation=MAP_INTERPOLATION)
        overlay = np.where(boundary, 1.0, np.nan)
        axes.imshow(overlay, cmap="cool", vmin=0.0, vmax=1.0, alpha=0.85,
                    interpolation=MAP_INTERPOLATION)
        axes.set_title(f"current-support boundaries over {label}", fontsize=10)
        axes.set_xticks([])
        axes.set_yticks([])
        figure.colorbar(image, ax=axes, shrink=0.82, label="C")
        figure.tight_layout()
        tmp = path.parent / f".{path.name}.tmp.png"
        figure.savefig(tmp, dpi=MAP_DPI)
        plt.close(figure)
        os.replace(str(tmp), str(path))
        written.append(path)
    return written


def render_top_residual_jump_map(root: Path, histograms: dict) -> Path:
    """Anchors of the top 1% |anomaly jump| pairs, reference vs candidate.

    The threshold comes from the streamed jump histogram, so the map is
    deterministic and needs no unbounded in-memory pair list.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rasters = Path(root) / "rasters"
    path = Path(root) / "maps" / "top_1_percent_residual_jump_pairs.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    for index, (side, name) in enumerate((
        ("reference", "reference_anomaly_zscore.tif"),
        ("candidate", "harmonized_anomaly_zscore.tif"),
    )):
        array = _read_full(rasters / name)
        threshold = histograms[side].quantile(99.0) or 0.0
        marks = np.zeros(array.shape, dtype=bool)
        horizontal = np.abs(rs.edge_difference(array, "horizontal"))
        vertical = np.abs(rs.edge_difference(array, "vertical"))
        marks[:, :-1] |= np.isfinite(horizontal) & (horizontal >= threshold)
        marks[:-1, :] |= np.isfinite(vertical) & (vertical >= threshold)
        axes[index].imshow(np.where(np.isfinite(array), 0.0, np.nan), cmap="Greys",
                           vmin=0.0, vmax=1.0, interpolation=MAP_INTERPOLATION)
        axes[index].imshow(np.where(marks, 1.0, np.nan), cmap="autumn",
                           vmin=0.0, vmax=1.0, interpolation=MAP_INTERPOLATION)
        axes[index].set_title(
            f"top 1% |anomaly jump| pairs -- {side}\n(threshold {threshold:.4f} z, "
            f"{int(marks.sum())} anchors)", fontsize=9)
        axes[index].set_xticks([])
        axes[index].set_yticks([])
    figure.tight_layout()
    tmp = path.parent / f".{path.name}.tmp.png"
    figure.savefig(tmp, dpi=MAP_DPI)
    plt.close(figure)
    os.replace(str(tmp), str(path))
    return path


def render_graph_maps(root: Path, dates, graph: dict, diagnostics: dict,
                      solution: dict | None) -> list[Path]:
    """The date-offset graph and the per-date offset magnitudes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    maps = Path(root) / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    written = []

    # --- graph layout: dates on a circle, edge width ~ independent blocks ---
    angles = np.linspace(0.0, 2.0 * np.pi, len(dates), endpoint=False)
    positions = {date: (float(np.cos(a)), float(np.sin(a)))
                 for date, a in zip(dates, angles)}
    figure, axes = plt.subplots(figsize=(7.6, 7.2))
    max_blocks = max([e["independent_blocks"] for e in graph["edges"]], default=1)
    for edge in graph["edges"]:
        x0, y0 = positions[edge["date_i"]]
        x1, y1 = positions[edge["date_j"]]
        width = 0.6 + 3.4 * (edge["independent_blocks"] / max(max_blocks, 1))
        axes.plot([x0, x1], [y0, y1], color="#4a7ebb", linewidth=width, zorder=1)
        axes.text((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                  f"{edge['edge_median_difference_celsius']:+.2f}",
                  fontsize=7, ha="center", va="center", color="#1f3b57",
                  bbox={"facecolor": "white", "alpha": 0.75, "pad": 1.0}, zorder=3)
    for edge in graph["rejected_edges"]:
        x0, y0 = positions[edge["date_i"]]
        x1, y1 = positions[edge["date_j"]]
        axes.plot([x0, x1], [y0, y1], color="#bbbbbb", linewidth=0.6,
                  linestyle=":", zorder=0)
    alpha_by_date = (solution or {}).get("alpha_by_date") or {}
    for date, (x, y) in positions.items():
        axes.scatter([x], [y], s=520, color="#f5b041", edgecolor="#7d5109", zorder=2)
        label = date if date not in alpha_by_date else \
            f"{date}\n{alpha_by_date[date]:+.2f} C"
        axes.text(x, y, label, fontsize=7, ha="center", va="center", zorder=4)
    axes.set_title(
        f"date-overlap graph (primary {graph['min_common_pixels']} px / "
        f"{graph['min_independent_blocks']} blocks)\n"
        f"{graph['edge_count']} eligible edges, "
        f"{diagnostics['connected_component_count']} component(s); "
        f"dotted = rejected", fontsize=10)
    axes.set_axis_off()
    axes.set_xlim(-1.35, 1.35)
    axes.set_ylim(-1.35, 1.35)
    figure.tight_layout()
    path = maps / "date_offset_graph.png"
    tmp = path.parent / f".{path.name}.tmp.png"
    figure.savefig(tmp, dpi=MAP_DPI)
    plt.close(figure)
    os.replace(str(tmp), str(path))
    written.append(path)

    # --- per-date offset magnitude -----------------------------------------
    figure, axes = plt.subplots(figsize=(8.4, 4.6))
    values = [float(alpha_by_date.get(d, 0.0)) for d in dates]
    errors = [
        (o.get("standard_error_celsius") or 0.0)
        for o in ((solution or {}).get("offsets") or [{} for _ in dates])
    ]
    axes.bar(range(len(dates)), values, yerr=errors, capsize=3,
             color=["#c0392b" if v < 0 else "#2471a3" for v in values])
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.axhline(MAX_ABS_DATE_OFFSET_CELSIUS, color="#7b241c", linewidth=0.8,
                 linestyle="--", label=f"+/-{MAX_ABS_DATE_OFFSET_CELSIUS} C bound")
    axes.axhline(-MAX_ABS_DATE_OFFSET_CELSIUS, color="#7b241c", linewidth=0.8,
                 linestyle="--")
    axes.set_xticks(range(len(dates)))
    axes.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    axes.set_ylabel("alpha (C)")
    axes.set_title("fitted additive acquisition-date offsets "
                   "(weighted mean constrained to zero)", fontsize=10)
    axes.legend(fontsize=8)
    figure.tight_layout()
    path = maps / "per_date_offset_magnitude.png"
    tmp = path.parent / f".{path.name}.tmp.png"
    figure.savefig(tmp, dpi=MAP_DPI)
    plt.close(figure)
    os.replace(str(tmp), str(path))
    written.append(path)
    return written


# =============================================================================
# Import-time schema contract
# =============================================================================
# Declared LAST so every participant (producer, CSV columns, canonical field
# list) is defined. If the scientific producer ever gains, loses or renames a
# field, importing this module fails immediately with an explicit message
# instead of a KeyError surfacing later inside Markdown rendering.
_assert_producer_matches_schema()
