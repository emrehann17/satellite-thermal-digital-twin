"""
landsat_residual_seam_attribution.py

Local-only DIAGNOSTIC ATTRIBUTION of the residual seam that is still visible in
the Manavgat date-balanced candidate after the completed downstream A/B
experiment, specifically in

    current_minus_baseline_celsius
    anomaly_zscore

The audit does not try to remove the seam. It separates each residual boundary
jump into the mechanisms that can produce it, using EXACT algebraic
decompositions rather than approximations.

WHAT THIS MODULE IS NOT
-----------------------
    - It is NOT a fix. Nothing is smoothed, blended, interpolated, in-painted,
      or cosmetically altered. Every raster this module writes is a diagnostic
      overlay on the exact input grid.
    - It NEVER changes the production reducer and never returns a production
      decision. `seam_fixed` and `production_approved` are not reachable
      outcomes.
    - It NEVER runs Earth Engine and never re-runs Step5-Step8. Every input is
      a frozen local file produced by an already-completed run.
    - It re-implements no scientific computation: the adjacency lattice, the
      spatial-block units, the cluster bootstrap, the provenance-boundary
      rasterization, the grid contract and the atomic-write helpers are reused
      from `src/landsat_composite_counterfactual_audit.py`.

ISOLATION CONTRACT
------------------
Everything this module writes lives under

    outputs/diagnostics/landsat_residual_seam_attribution/<experiment_id>/

The frozen downstream A/B namespace, the frozen counterfactual namespace and
the frozen canonical experiment namespace are READ-ONLY inputs and are never
written, deleted, or used as an output root.

DECOMPOSITIONS
--------------
Q1  current_minus_baseline, for an adjacency pair (A, B):

        target_jump        = D_B - D_A                       (D = C - M)
        current_component  = C_B - C_A
        baseline_component = -(M_B - M_A)
        target_jump       == current_component + baseline_component

Q2  anomaly_zscore, with D = current_minus_baseline, S = baseline_std,
    Z = D / S. The EXACT symmetric decomposition (no Taylor expansion):

        numerator_contribution   = 0.5 * (1/S_A + 1/S_B) * (D_B - D_A)
        denominator_contribution = 0.5 * (D_A + D_B)     * (1/S_B - 1/S_A)
        Z_B - Z_A               == numerator_contribution
                                 + denominator_contribution

    The identity is algebraically exact:
        0.5*(a+b)*(y-x) + 0.5*(x+y)*(b-a) = b*y - a*x
    with a = 1/S_A, b = 1/S_B, x = D_A, y = D_B.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import src.landsat_composite_counterfactual_audit as audit
import src.landsat_composite_downstream_ab as ab
from core.paths import PROJECT_ROOT

# Shared primitives -- ONE implementation, ONE contract.
NamespaceSafetyError = audit.NamespaceSafetyError
GridMismatchError = audit.GridMismatchError
write_json_atomic = audit.write_json_atomic
sha256_and_size = audit.sha256_and_size
grid_signature = audit.grid_signature
assert_same_grid = audit.assert_same_grid
files_present_and_signed = audit.files_present_and_signed
process_rss_mib = audit.process_rss_mib
ORIENTATIONS = audit.ORIENTATIONS
_edge_pairs = audit._edge_pairs
_anchor_indices = audit._anchor_indices


class ResidualSeamError(RuntimeError):
    """Fail-fast error for the residual-seam attribution audit."""


class PrerequisiteError(ResidualSeamError):
    """A required frozen input or upstream prerequisite is missing/invalid."""


# =============================================================================
# Identity / versions
# =============================================================================
DIAGNOSTIC_NAMESPACE = "landsat_residual_seam_attribution"
DOWNSTREAM_AB_NAMESPACE = ab.DIAGNOSTIC_NAMESPACE
COUNTERFACTUAL_NAMESPACE = audit.DIAGNOSTIC_NAMESPACE

#: Bumped when the anomaly reconstruction check was split into an ALGEBRAIC
#: IDENTITY check (gating, float64) and a STORED-RASTER REPRODUCTION check
#: (descriptive, float32 serialization). A report written under 1.0 carries
#: the older single-tolerance `stored_vs_recomputed_z` block instead.
REPORT_SCHEMA_VERSION = "1.1-separated-anomaly-checks"
DECISION_RULE_VERSION = "1.0-residual-attribution-ordered"

#: This task supports exactly one AOI. A second AOI needs its own frozen inputs
#: and its own predeclaration; it is deliberately NOT reachable here.
SUPPORTED_EXPERIMENT_IDS = ("manavgat_2021",)

#: The chain under attribution. The reference chain is not re-audited here.
CANDIDATE_CHAIN = ab.CHAIN_CANDIDATE  # date_balanced_lst_only
CANDIDATE_SIDE = ab.CHAIN_SIDE[CANDIDATE_CHAIN]  # candidate

#: Upstream prerequisites (the A/B run must have been technically valid).
REQUIRED_AB_REFERENCE_REPRODUCTION = "pass"
REQUIRED_COUNTERFACTUAL_FINAL_STATUS = "supported_reduction"

#: Production approval is explicitly NOT required and is never checked for.
PRODUCTION_APPROVAL_REQUIRED = False


# =============================================================================
# Target products
# =============================================================================
TARGET_CMB = "current_minus_baseline_celsius"
TARGET_ANOMALY = "anomaly_zscore"
TARGET_PRODUCTS = (TARGET_CMB, TARGET_ANOMALY)


# =============================================================================
# Predeclared final statuses (ORDERED; see decide_final_status)
# =============================================================================
STATUS_INVALID_INPUTS = "invalid_inputs"
STATUS_RESIDUAL_NOT_DETECTED = "residual_not_detected"
STATUS_CURRENT_SUPPORT = "current_support_dominant"
STATUS_BASELINE_SUPPORT = "baseline_support_dominant"
STATUS_BASELINE_VARIANCE = "baseline_variance_amplification_dominant"
STATUS_PATHROW = "pathrow_bias_supported"
STATUS_MIXED = "mixed_mechanisms"
STATUS_INCONCLUSIVE = "residual_mechanism_inconclusive"

FINAL_STATUSES = (
    STATUS_INVALID_INPUTS,
    STATUS_RESIDUAL_NOT_DETECTED,
    STATUS_CURRENT_SUPPORT,
    STATUS_BASELINE_SUPPORT,
    STATUS_BASELINE_VARIANCE,
    STATUS_PATHROW,
    STATUS_MIXED,
    STATUS_INCONCLUSIVE,
)

#: Outcomes this audit can never produce, in any field, in any report.
FORBIDDEN_CONCLUSIONS = (
    "seam_fixed", "seam fixed", "production_approved", "production approved",
    "production_ready", "approved_for_production",
)

FINAL_STATUS_MEANINGS = {
    STATUS_INVALID_INPUTS:
        "A required frozen input, grid contract or upstream prerequisite did not "
        "hold. No attribution claim is made.",
    STATUS_RESIDUAL_NOT_DETECTED:
        "The candidate target products show no excess boundary jump at any tested "
        "known boundary relative to matched within-block controls. This is NOT a "
        "statement that the seam is fixed; it means this audit's boundary "
        "definitions do not localise the residual structure.",
    STATUS_CURRENT_SUPPORT:
        "The residual jump is predominantly carried by the current-period LST "
        "component at current observation-support boundaries.",
    STATUS_BASELINE_SUPPORT:
        "The residual jump is predominantly carried by the baseline-mean component "
        "at baseline observation-support boundaries.",
    STATUS_BASELINE_VARIANCE:
        "The residual anomaly jump is predominantly carried by the baseline "
        "standard-deviation denominator, concentrated near low or threshold-"
        "adjacent baseline variance.",
    STATUS_PATHROW:
        "A residual metadata-derived path/row effect is supported on pairs that "
        "carry no observation-support or threshold boundary.",
    STATUS_MIXED:
        "At least two mechanisms have independently supported evidence and no "
        "single mechanism satisfies its predeclared dominance rule.",
    STATUS_INCONCLUSIVE:
        "A technically valid run whose residual mechanism meets none of the "
        "stronger predeclared categories.",
}


# =============================================================================
# Predeclared thresholds -- fixed BEFORE any result is inspected
# =============================================================================
def step5_thresholds() -> dict:
    """The production Step5 guard values; never redefined here."""
    from core.config import (
        STEP5_MIN_BASELINE_STD_CELSIUS,
        STEP5_MIN_BASELINE_VALID_COUNT,
        STEP5_MIN_CURRENT_VALID_COUNT,
    )

    return OrderedDict((
        ("min_baseline_std_celsius", float(STEP5_MIN_BASELINE_STD_CELSIUS)),
        ("min_baseline_valid_count", int(STEP5_MIN_BASELINE_VALID_COUNT)),
        ("min_current_valid_count", int(STEP5_MIN_CURRENT_VALID_COUNT)),
    ))


#: `near_std_threshold_boundary` epsilon. PREDECLARED: the primary value is the
#: one used by every headline number; the sensitivity values are ALWAYS reported
#: alongside it so no epsilon can be chosen post hoc.
STD_THRESHOLD_EPSILON_PRIMARY = 0.10
STD_THRESHOLD_EPSILON_SENSITIVITY = (0.05, 0.20)
STD_THRESHOLD_EPSILONS = (
    STD_THRESHOLD_EPSILON_PRIMARY, *STD_THRESHOLD_EPSILON_SENSITIVITY,
)

#: Descriptive-only hotspot percentiles. These are MAP thresholds; they never
#: gate a scientific claim and are never called a significance threshold.
HOTSPOT_PERCENTILES = (99.0, 95.0)
HOTSPOT_LABELS = OrderedDict(((99.0, "top_1_percent"), (95.0, "top_5_percent")))

#: Reconstruction tolerances.
#:
#: current_minus_baseline: the target is the STORED float32 derived raster while
#: the components are the STORED float32 Step5 rasters. float32 has a 24-bit
#: mantissa (eps = 2**-24 ~ 5.96e-8), so a stored magnitude below 1e3 Celsius
#: carries at most ~3e-5 of storage error; two endpoints bound the residual at
#: ~1.2e-4. 1e-3 leaves an order of magnitude of margin. The OBSERVED maximum
#: residual is always reported next to the tolerance.
CMB_RECONSTRUCTION_ABS_TOL = 1e-3

# -----------------------------------------------------------------------------
# The anomaly has TWO INDEPENDENT checks. They answer different questions, they
# have different tolerances, and only the first one can invalidate the audit.
#
#   1. ALGEBRAIC IDENTITY CHECK (gating).
#      Z_A and Z_B are recomputed in float64 as D/S from the stored components,
#      and the symmetric decomposition must reproduce Z_B - Z_A. The identity is
#      exact in real arithmetic, so the only admissible residual is float64
#      round-off. A failure here means the DECOMPOSITION is wrong.
#
#   2. STORED-RASTER REPRODUCTION CHECK (descriptive).
#      The recomputed float64 D/S is compared against the STORED float32
#      `anomaly_zscore` raster. Step5 divided its own internal float32 difference
#      and then serialised the quotient to float32, so a small disagreement is
#      EXPECTED serialization error, not a decomposition defect. This check can
#      never invalidate the audit and never contributes a failure reason.
# -----------------------------------------------------------------------------
#: Check 1 -- scale-aware float64 tolerance. Step5 masks S < 1.0 Celsius, so
#: 1/S <= 1 wherever the anomaly is valid and the expression is well conditioned;
#: the relative term only matters for the largest |Z| in the scene.
ANOMALY_IDENTITY_ABS_TOL = 1e-10
ANOMALY_IDENTITY_REL_TOL = 1e-12
ANOMALY_IDENTITY_TOLERANCE_POLICY = (
    "scale-aware float64: tolerance = max(ANOMALY_IDENTITY_ABS_TOL, "
    "ANOMALY_IDENTITY_REL_TOL * max(|Z_A|, |Z_B|, |numerator|, |denominator|))"
)


def anomaly_identity_tolerance(scale):
    """Per-pair float64 tolerance for the algebraic identity check."""
    import numpy as np

    return np.maximum(
        ANOMALY_IDENTITY_ABS_TOL,
        ANOMALY_IDENTITY_REL_TOL * np.abs(np.asarray(scale, dtype="float64")),
    )


#: Check 2 -- predeclared float32-compatible tolerance. NOT invented here: it is
#: the counterfactual audit's own float32 physical-quantity reproduction
#: tolerance, reused verbatim.
ANOMALY_STORED_REPRODUCTION_TOL = audit.REPRODUCTION_TOLERANCES["physical_float32"]
ANOMALY_STORED_REPRODUCTION_TOL_SOURCE = (
    "src.landsat_composite_counterfactual_audit."
    "REPRODUCTION_TOLERANCES['physical_float32']"
)

#: The outer bound of the EXISTING Step5-family reproduction policy, inherited
#: from the downstream A/B chain gate for this exact product. An error between
#: the predeclared tolerance and this bound is still within established policy
#: and is reported as such -- never as a decomposition failure.
ANOMALY_STORED_REPRODUCTION_POLICY_TOL = ab.REPRODUCTION_TOLERANCES["anomaly_zscore"]
ANOMALY_STORED_REPRODUCTION_POLICY_SOURCE = (
    "src.landsat_composite_downstream_ab.REPRODUCTION_TOLERANCES['anomaly_zscore']"
)

#: Percentiles of the absolute stored-vs-recomputed error that are always
#: reported alongside the maximum.
ANOMALY_STORED_REPRODUCTION_PERCENTILES = (50.0, 95.0, 99.0, 99.9)

#: Histogram range/resolution for those percentiles: 1e-8 per bin over [0, 1e-3].
STORED_REPRODUCTION_HISTOGRAM_MAX = 1e-3
STORED_REPRODUCTION_HISTOGRAM_BINS = 100000

#: A signed component pair "cancels" when the two contributions have opposite
#: signs (the jump is smaller than either component) and "reinforces" when they
#: share a sign. Exact zeros are counted separately and never forced into either.
CANCELLATION_ZERO_EPS = 0.0


# =============================================================================
# Bootstrap configuration (predeclared)
# =============================================================================
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI = 0.95
BOOTSTRAP_CI_LOWER_PCT = 2.5
BOOTSTRAP_CI_UPPER_PCT = 97.5

#: Spatial-block edge length in 30 m cells. 128 cells ~ 3.8 km, which is far
#: wider than the adjacency lattice and wider than the local terrain
#: autocorrelation the matched controls are meant to absorb.
BOOTSTRAP_BLOCK_SIZE_CELLS = 128

#: Below this many independent units an interval is not reported as evidence.
MIN_BOOTSTRAP_UNITS = 8

#: The path/row test needs both enough spatial blocks AND more than one distinct
#: metadata-derived interface, otherwise a single footprint edge could carry the
#: whole effect.
MIN_PATHROW_ONLY_UNITS = 8
MIN_PATHROW_INTERFACES = 2

#: Dominance thresholds. A share interval must exclude 0.50 from ABOVE.
DOMINANCE_SHARE_LOWER_BOUND = 0.50

#: `current_support_dominant` requires the effect in at least this many distinct
#: current-support boundary definitions.
MIN_CURRENT_SUPPORT_DEFINITIONS = 2


# =============================================================================
# Windowed processing
# =============================================================================
#: Row-block height for the streaming passes. Each window carries a one-row halo
#: so vertical adjacency pairs that straddle a window edge are still built once
#: and only once.
WINDOW_ROWS = 256

#: Absolute-jump histogram used for the descriptive hotspot percentiles. Fixed
#: edges keep the quantile deterministic and bounded in memory.
HISTOGRAM_BINS = 5000
HISTOGRAM_MAX = OrderedDict(((TARGET_CMB, 50.0), (TARGET_ANOMALY, 20.0)))

#: Deterministic pair sample written to `tables/pair_sample.csv`. Reservoir
#: sampling with a fixed seed; the full pair set is NEVER written by default.
PAIR_SAMPLE_SIZE = 20000
PAIR_SAMPLE_SEED = 42


# =============================================================================
# Boundary flags and stratified classes
# =============================================================================
#: Raw, non-exclusive mechanism flags. Every pair keeps all of them.
CURRENT_SUPPORT_FLAGS = (
    "current_unique_date_count_change",
    "current_scene_count_change",
    "current_valid_count_change",
)
BASELINE_SUPPORT_FLAGS = (
    "baseline_valid_year_change",
    "baseline_annual_date_support_change",
)
SUPPORT_FLAGS = (
    *CURRENT_SUPPORT_FLAGS,
    *BASELINE_SUPPORT_FLAGS,
    "same_day_multiplicity_change",
)
#: The support union used by the stratified classes. It includes the COMPOSITE
#: `current_support_change` as well as its primitives, so a caller that sets only
#: the composite is never silently treated as a non-boundary pair.
SUPPORT_UNION_FLAGS = ("current_support_change", *SUPPORT_FLAGS)

THRESHOLD_FLAGS = (
    "low_baseline_std_boundary",
    "near_std_threshold_boundary",
    "current_count_threshold_boundary",
    "baseline_count_threshold_boundary",
)
PATHROW_FLAGS = ("source_path_row_boundary",)

BOUNDARY_FLAGS = (
    "current_support_change",
    *SUPPORT_FLAGS,
    *THRESHOLD_FLAGS,
    *PATHROW_FLAGS,
)

#: Stratified classes. The five required classes are always present. Overlaps
#: that do not fit them get their OWN label instead of being forced into one --
#: the raw flags above remain available for every pair.
CLASS_SUPPORT_ONLY = "support_only"
CLASS_PATHROW_ONLY = "pathrow_only"
CLASS_SUPPORT_AND_PATHROW = "support_and_pathrow"
CLASS_THRESHOLD_ONLY = "threshold_only"
CLASS_PATHROW_AND_THRESHOLD = "pathrow_and_threshold"
CLASS_NONE = "none_of_known_boundaries"

STRATIFIED_CLASSES = (
    CLASS_SUPPORT_ONLY,
    CLASS_PATHROW_ONLY,
    CLASS_SUPPORT_AND_PATHROW,
    CLASS_THRESHOLD_ONLY,
    CLASS_PATHROW_AND_THRESHOLD,
    CLASS_NONE,
)

#: Boundary definitions carried into the excess-jump / matched-control test.
EXCESS_JUMP_BOUNDARIES = (
    "current_support_change",
    "current_unique_date_count_change",
    "current_scene_count_change",
    "current_valid_count_change",
    "baseline_valid_year_change",
    "baseline_annual_date_support_change",
    "same_day_multiplicity_change",
    "low_baseline_std_boundary",
    "near_std_threshold_boundary",
    "current_count_threshold_boundary",
    "baseline_count_threshold_boundary",
    "source_path_row_boundary",
    CLASS_PATHROW_ONLY,
    CLASS_SUPPORT_AND_PATHROW,
    CLASS_SUPPORT_ONLY,
    # Derived: the baseline-support test with every current-support pair removed,
    # so rule B cannot be satisfied by current-support contamination.
    "baseline_support_excluding_current",
)


# =============================================================================
# Matched-control stratification (predeclared bin edges)
# =============================================================================
#: Controls are matched WITHIN a spatial block and orientation, and additionally
#: stratified by the local terrain / greenness gradient across the SAME pair, so
#: a boundary pair on a steep hillside is compared with non-boundary pairs on
#: comparably steep ground. Bin edges are fixed in advance; they are physical
#: round numbers, not quantiles of this dataset.
ELEVATION_GRADIENT_BINS = (0.0, 1.0, 5.0, 20.0, 50.0)     # metres per 30 m step
SLOPE_GRADIENT_BINS = (0.0, 0.5, 2.0, 5.0)                # degrees per 30 m step
NDVI_GRADIENT_BINS = (0.0, 0.01, 0.05, 0.15)              # unitless per 30 m step

MATCHED_CONTROL_STRATEGY = (
    "Within-block matched controls. For every boundary definition, boundary "
    "pairs are compared with pairs in the SAME spatial block and the SAME "
    "orientation that carry none of the known boundary flags, additionally "
    "stratified by predeclared elevation-gradient, slope-gradient and (when "
    "available) NDVI-gradient bins of the same pair. Only strata containing "
    "both boundary and control pairs contribute. The excess is aggregated to "
    "block level and the interval comes from a cluster bootstrap over blocks."
)


# =============================================================================
# Namespace resolution and safety
# =============================================================================
def diagnostic_output_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """The ONE directory this audit may write beneath."""
    return Path(base_dir) / "outputs" / "diagnostics" / DIAGNOSTIC_NAMESPACE / experiment_id


def downstream_ab_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen downstream A/B root (READ-ONLY)."""
    return ab.diagnostic_output_root(experiment_id, base_dir)


def counterfactual_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen counterfactual audit root (READ-ONLY)."""
    return ab.counterfactual_source_root(experiment_id, base_dir)


def canonical_experiment_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen canonical experiment root (READ-ONLY)."""
    return ab.canonical_experiment_root(experiment_id, base_dir)


def forbidden_write_roots(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> list[Path]:
    """Roots that must never be written, overwritten, or deleted."""
    return [
        downstream_ab_root(experiment_id, base_dir),
        counterfactual_root(experiment_id, base_dir),
        canonical_experiment_root(experiment_id, base_dir),
        Path(base_dir) / "data",
        Path(base_dir) / "outputs" / "step5",
        Path(base_dir) / "outputs" / "step5c",
        Path(base_dir) / "outputs" / "step3",
        Path(base_dir) / "config",
    ]


def assert_namespace_safe(paths, experiment_id: str, base_dir: Path = PROJECT_ROOT) -> None:
    """Every supplied write path must resolve strictly under the audit root."""
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
                f"refusing to write outside the dedicated residual-seam root: "
                f"{candidate} (allowed root: {root})"
            )


def clear_diagnostic_namespace(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> str | None:
    """`--force` deletion of ONLY the dedicated residual-seam namespace."""
    root = diagnostic_output_root(experiment_id, base_dir)
    if not root.exists():
        return None
    resolved = root.resolve()
    assert_namespace_safe([resolved], experiment_id, base_dir)
    expected = diagnostic_output_root(experiment_id, base_dir).resolve()
    if resolved != expected:
        raise NamespaceSafetyError(
            f"refusing to delete {resolved}: it is not the dedicated audit root {expected}"
        )
    if DIAGNOSTIC_NAMESPACE not in resolved.parts or experiment_id not in resolved.parts:
        raise NamespaceSafetyError(
            f"refusing to delete a path that is not namespaced to this audit: {resolved}"
        )
    shutil.rmtree(resolved)
    return str(resolved)


def plan_output_layout(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, Path]":
    """The full planned directory layout (informational; creates nothing)."""
    root = diagnostic_output_root(experiment_id, base_dir)
    return OrderedDict((
        ("root", root),
        ("config", root / "config"),
        ("checkpoints", root / "checkpoints"),
        ("tables", root / "tables"),
        ("maps", root / "maps"),
        ("maps_current_minus_baseline", root / "maps" / "current_minus_baseline"),
        ("maps_anomaly_zscore", root / "maps" / "anomaly_zscore"),
        ("maps_support", root / "maps" / "support"),
        ("maps_baseline_std", root / "maps" / "baseline_std"),
        ("maps_attribution", root / "maps" / "attribution"),
    ))


TABLE_FILES = (
    "current_minus_baseline_decomposition.csv",
    "anomaly_decomposition.csv",
    "mask_discontinuity_summary.csv",
    "boundary_excess_jump.csv",
    "hotspot_mechanism_overlap.csv",
    "pathrow_stratified_test.csv",
    "bootstrap_summary.csv",
    "pair_sample.csv",
)

DOCUMENT_FILES = (
    "residual_seam_summary.json",
    "residual_seam_summary.md",
    "residual_seam_manifest.json",
    "input_provenance.json",
)


def plan_expected_files(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, Path]":
    """Every document/table this audit is expected to produce."""
    layout = plan_output_layout(experiment_id, base_dir)
    expected: "OrderedDict[str, Path]" = OrderedDict()
    for name in DOCUMENT_FILES:
        expected[name] = layout["root"] / name
    for name in TABLE_FILES:
        expected[f"tables/{name}"] = layout["tables"] / name
    expected["config/residual_seam_config.json"] = layout["config"] / "residual_seam_config.json"
    expected["checkpoints/residual_seam_checkpoint.json"] = (
        layout["checkpoints"] / CHECKPOINT_FILENAME
    )
    for key, path in MAP_OUTPUTS.items():
        expected[f"maps/{key}"] = layout["root"] / "maps" / path
    return expected


#: Diagnostic overlays. Every one is a categorical or magnitude raster written
#: on the EXACT input grid at the 'a' endpoint of each adjacency edge. No
#: resampling, no smoothing, no interpolation anywhere.
MAP_OUTPUTS = OrderedDict((
    ("residual_cmb_abs_jump", "current_minus_baseline/residual_abs_jump.tif"),
    ("residual_cmb_signed_jump", "current_minus_baseline/residual_signed_jump.tif"),
    ("cmb_hotspot_class", "current_minus_baseline/hotspot_class.tif"),
    ("residual_anomaly_abs_jump", "anomaly_zscore/residual_abs_jump.tif"),
    ("residual_anomaly_signed_jump", "anomaly_zscore/residual_signed_jump.tif"),
    ("anomaly_hotspot_class", "anomaly_zscore/hotspot_class.tif"),
    ("anomaly_mask_discontinuity", "anomaly_zscore/mask_discontinuity.tif"),
    ("current_support_change", "support/current_support_change.tif"),
    ("baseline_support_change", "support/baseline_support_change.tif"),
    ("support_pathrow_overlap", "support/support_pathrow_overlap.tif"),
    ("baseline_std", "baseline_std/baseline_std_min_of_pair.tif"),
    ("near_std_threshold", "baseline_std/near_std_threshold_boundary.tif"),
    ("cmb_attribution", "attribution/current_minus_baseline_attribution.tif"),
    ("anomaly_attribution", "attribution/anomaly_attribution.tif"),
))

#: Categorical codes written into the attribution overlays.
ATTRIBUTION_CODES = OrderedDict((
    ("no_pair", 0),
    ("current_dominant", 1),
    ("baseline_mean_dominant", 2),
    ("components_tied", 3),
))
ANOMALY_ATTRIBUTION_CODES = OrderedDict((
    ("no_pair", 0),
    ("numerator_dominant", 1),
    ("denominator_dominant", 2),
    ("components_tied", 3),
))
OVERLAP_CODES = OrderedDict(
    (name, index) for index, name in enumerate(("no_pair", *STRATIFIED_CLASSES))
)
HOTSPOT_CODES = OrderedDict((
    ("no_pair", 0), ("below_top_5_percent", 1),
    ("top_5_percent", 2), ("top_1_percent", 3),
))


# =============================================================================
# Input resolution
# =============================================================================
#: Baseline years used by the frozen candidate chain. Resolved from the
#: experiment context at run time; this is only the ordering fallback.
def baseline_years(experiment_id: str) -> list[int]:
    from core.experiment_context import build_experiment_context

    return [int(y) for y in build_experiment_context(experiment_id)["baseline_years"]]


def candidate_step5_dir(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    return downstream_ab_root(experiment_id, base_dir) / CANDIDATE_SIDE / "step5"


def candidate_derived_dir(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    return downstream_ab_root(experiment_id, base_dir) / CANDIDATE_SIDE / ab.DERIVED_SUBDIR


def build_input_plan(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, dict]":
    """Logical input role -> {path, chain, required, family}.

    Filenames are never assumed blind: every optional counterfactual raster is
    resolved against the frozen directory and reported as missing rather than
    invented. The per-year baseline support rasters are enumerated from the
    experiment's own baseline-year list.
    """
    step5 = candidate_step5_dir(experiment_id, base_dir)
    derived = candidate_derived_dir(experiment_id, base_dir)
    cf = counterfactual_root(experiment_id, base_dir) / "rasters"
    shared = downstream_ab_root(experiment_id, base_dir) / "inputs" / "shared"

    plan: "OrderedDict[str, dict]" = OrderedDict()

    def add(role, path, *, chain, required, family, purpose):
        plan[role] = OrderedDict((
            ("role", role),
            ("path", Path(path)),
            ("source_chain", chain),
            ("required", bool(required)),
            ("family", family),
            ("purpose", purpose),
        ))

    # --- decomposition components (all REQUIRED, all 30 m, exact grid) -------
    add("current_lst_celsius", step5 / "current_period_median_celsius.tif",
        chain="downstream_ab_candidate", required=True, family="target_component",
        purpose="C in D = C - M; current LST jump")
    add("baseline_lst_mean_celsius", step5 / "baseline_lst_mean_celsius.tif",
        chain="downstream_ab_candidate", required=True, family="target_component",
        purpose="M in D = C - M; baseline mean jump")
    add("baseline_lst_std_celsius", step5 / "baseline_lst_std_celsius.tif",
        chain="downstream_ab_candidate", required=True, family="target_component",
        purpose="S in Z = D / S; denominator contribution")
    add(TARGET_CMB, derived / "current_minus_baseline_celsius.tif",
        chain="downstream_ab_candidate", required=True, family="target",
        purpose="target product 1; stored D")
    add(TARGET_ANOMALY, step5 / "anomaly_zscore.tif",
        chain="downstream_ab_candidate", required=True, family="target",
        purpose="target product 2; stored Z and its valid mask")

    # --- Step5 support / mask rasters (REQUIRED) -----------------------------
    add("current_period_valid_count", step5 / "current_period_valid_count.tif",
        chain="downstream_ab_candidate", required=True, family="support",
        purpose="Step5 current observation support (unique-date semantics)")
    add("baseline_valid_count", step5 / "baseline_valid_count.tif",
        chain="downstream_ab_candidate", required=True, family="support",
        purpose="Step5 baseline valid-YEAR count")
    add("low_baseline_std_mask", step5 / "low_baseline_std_mask.tif",
        chain="downstream_ab_candidate", required=True, family="mask",
        purpose="Step5 low-baseline-std guard flag")
    add("low_baseline_count_mask", step5 / "low_baseline_count_mask.tif",
        chain="downstream_ab_candidate", required=True, family="mask",
        purpose="Step5 low-baseline-count guard flag")
    add("low_current_count_mask", step5 / "low_current_count_mask.tif",
        chain="downstream_ab_candidate", required=True, family="mask",
        purpose="Step5 low-current-count guard flag")

    # --- counterfactual support rasters (REQUIRED for support boundaries) ----
    add("current_unique_date_valid_count", cf / "current_lst_unique_date_valid_count.tif",
        chain="counterfactual", required=True, family="support",
        purpose="current unique acquisition-date support")
    add("current_scene_valid_count", cf / "current_lst_scene_valid_count.tif",
        chain="counterfactual", required=True, family="support",
        purpose="current raw scene-observation support")
    add("current_same_day_multiplicity", cf / "current_lst_same_day_multiplicity.tif",
        chain="counterfactual", required=True, family="support",
        purpose="same-day multiplicity support")

    # --- per-year baseline support (OPTIONAL; reported when absent) ----------
    for year in baseline_years(experiment_id):
        add(f"baseline_{year}_unique_date_valid_count",
            cf / f"baseline_lst_{year}_unique_date_valid_count.tif",
            chain="counterfactual", required=False, family="support",
            purpose=f"per-year baseline unique-date support ({year})")
        add(f"baseline_{year}_scene_valid_count",
            cf / f"baseline_lst_{year}_scene_valid_count.tif",
            chain="counterfactual", required=False, family="support",
            purpose=f"per-year baseline scene support ({year})")

    # --- matched-control covariates (OPTIONAL) -------------------------------
    add("elevation", shared / "dem" / "elevation.tif",
        chain="downstream_ab_shared", required=False, family="covariate",
        purpose="elevation-gradient matching bin")
    add("slope", shared / "dem" / "slope.tif",
        chain="downstream_ab_shared", required=False, family="covariate",
        purpose="slope-gradient matching bin")
    add("ndvi_current", shared / "ndvi_current_period" / "current_ndvi_median.tif",
        chain="downstream_ab_shared", required=False, family="covariate",
        purpose="NDVI-gradient matching bin (no new data dependency)")

    return plan


def pathrow_boundary_sources(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Frozen metadata-derived path/row artefacts (READ-ONLY, all optional)."""
    root = counterfactual_root(experiment_id, base_dir)
    return OrderedDict((
        ("scene_boundaries_geojson", root / "scene_boundaries.geojson"),
        ("scene_footprints_geojson", root / "scene_footprints.geojson"),
        ("source_scene_metadata", root / "source_scene_metadata.json"),
        ("counterfactual_summary", root / "counterfactual_summary.json"),
        ("counterfactual_manifest", root / "manifest.json"),
    ))


def upstream_report_paths(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Frozen upstream reports whose prerequisites this audit checks."""
    ab_root = downstream_ab_root(experiment_id, base_dir)
    cf_root = counterfactual_root(experiment_id, base_dir)
    return OrderedDict((
        ("downstream_ab_summary", ab_root / "downstream_ab_summary.json"),
        ("downstream_ab_manifest", ab_root / "downstream_ab_manifest.json"),
        ("downstream_ab_input_provenance", ab_root / "input_provenance.json"),
        ("downstream_ab_reference_reproduction", ab_root / "reference_reproduction.json"),
        ("counterfactual_summary", cf_root / "counterfactual_summary.json"),
        ("counterfactual_manifest", cf_root / "manifest.json"),
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
    """Fail clearly when a required support raster is absent."""
    missing = missing_required_inputs(plan)
    if missing:
        raise PrerequisiteError(
            f"experiment {experiment_id!r} is missing required frozen inputs for the "
            "residual-seam attribution audit; this audit never regenerates them and "
            "never calls Earth Engine. Missing:\n  " + "\n  ".join(missing)
        )


# =============================================================================
# Upstream prerequisites
# =============================================================================
def load_upstream_state(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Read the frozen A/B and counterfactual verdicts this audit depends on.

    Production approval is deliberately NOT part of the contract: the A/B
    experiment can never grant it, and this audit does not need it.
    """
    paths = upstream_report_paths(experiment_id, base_dir)
    state: "OrderedDict[str, object]" = OrderedDict((
        ("downstream_ab_root", str(downstream_ab_root(experiment_id, base_dir))),
        ("counterfactual_root", str(counterfactual_root(experiment_id, base_dir))),
        ("present", OrderedDict((k, Path(v).exists()) for k, v in paths.items())),
        ("ab_final_status", None),
        ("ab_final_status_present", False),
        ("ab_reference_reproduction_status", None),
        ("ab_baseline_invariance_status", None),
        ("ab_candidate_chain", None),
        ("ab_candidate_audit_prerequisites_met", None),
        ("ab_production_approved", None),
        ("ab_modis_compatibility_mode", None),
        ("counterfactual_final_status", None),
        ("counterfactual_canonical_reproduction", None),
        ("report_file_hashes", OrderedDict()),
    ))

    summary_path = Path(paths["downstream_ab_summary"])
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validity = summary.get("technical_validity") or {}
        state["ab_final_status"] = summary.get("final_status")
        state["ab_final_status_present"] = bool(summary.get("final_status"))
        state["ab_reference_reproduction_status"] = validity.get("reference_reproduction_status")
        state["ab_baseline_invariance_status"] = validity.get("baseline_invariance_status")
        state["ab_candidate_chain"] = summary.get("candidate_chain")
        state["ab_candidate_audit_prerequisites_met"] = validity.get(
            "candidate_audit_prerequisites_met"
        )
        state["ab_production_approved"] = summary.get("production_approved")
        state["ab_modis_compatibility_mode"] = validity.get("modis_compatibility_mode")
        state["ab_warnings"] = summary.get("warnings") or []

    cf_summary_path = Path(paths["counterfactual_summary"])
    if cf_summary_path.exists():
        cf_summary = json.loads(cf_summary_path.read_text(encoding="utf-8"))
        state["counterfactual_final_status"] = cf_summary.get("final_status")
        state["counterfactual_canonical_reproduction"] = (
            (cf_summary.get("canonical_reproduction") or {}).get("status")
        )

    hashes: "OrderedDict[str, dict]" = OrderedDict()
    for name, path in paths.items():
        if Path(path).exists():
            hashes[name] = sha256_and_size(Path(path))
    state["report_file_hashes"] = hashes
    state["prerequisites_met"] = upstream_prerequisites_met(state)
    return state


def upstream_prerequisites_met(state: dict) -> bool:
    """The four required upstream conditions; production approval is NOT one."""
    return bool(
        state.get("ab_final_status_present")
        and state.get("ab_candidate_chain") == CANDIDATE_CHAIN
        and state.get("ab_reference_reproduction_status") == REQUIRED_AB_REFERENCE_REPRODUCTION
        and state.get("ab_candidate_audit_prerequisites_met") is True
    )


def validate_upstream_state(state: dict) -> None:
    if not upstream_prerequisites_met(state):
        raise PrerequisiteError(
            "the frozen downstream A/B prerequisites are not met for the residual-seam "
            "audit. Required: a recorded final status, candidate chain "
            f"{CANDIDATE_CHAIN!r}, reference reproduction "
            f"{REQUIRED_AB_REFERENCE_REPRODUCTION!r}, and candidate audit "
            f"prerequisites met. Found: final_status={state.get('ab_final_status')!r}, "
            f"candidate_chain={state.get('ab_candidate_chain')!r}, "
            f"reference_reproduction={state.get('ab_reference_reproduction_status')!r}, "
            f"candidate_audit_prerequisites_met="
            f"{state.get('ab_candidate_audit_prerequisites_met')!r}. "
            "Production approval is NOT required and is never checked."
        )


# =============================================================================
# Input provenance
# =============================================================================
def raster_provenance_record(role: str, entry: dict) -> dict:
    """Full read-only provenance for one input raster."""
    import numpy as np
    import rasterio

    path = Path(entry["path"])
    record = OrderedDict((
        ("role", role),
        ("absolute_path", str(path.resolve()) if path.exists() else str(path)),
        ("source_diagnostic_chain", entry["source_chain"]),
        ("family", entry["family"]),
        ("purpose", entry["purpose"]),
        ("required", entry["required"]),
        ("present", path.exists()),
    ))
    if not path.exists():
        return record

    signed = sha256_and_size(path)
    record["sha256"] = signed["sha256"]
    record["bytes"] = signed["bytes"]
    with rasterio.open(path) as src:
        nodata = src.nodata
        record.update(OrderedDict((
            ("crs", str(src.crs)),
            ("transform", [float(v) for v in tuple(src.transform)[:6]]),
            ("width", int(src.width)),
            ("height", int(src.height)),
            ("dtype", str(src.dtypes[0])),
            ("nodata", None if nodata is None else float(nodata)),
        )))
        array = src.read(1, masked=True).astype("float64").filled(np.nan)
    array = np.where(array == audit.NODATA_SENTINEL, np.nan, array)
    record["valid_pixel_count"] = int(np.isfinite(array).sum())
    return record


def json_provenance_record(role: str, path: Path) -> dict:
    path = Path(path)
    record = OrderedDict((
        ("role", role),
        ("absolute_path", str(path.resolve()) if path.exists() else str(path)),
        ("present", path.exists()),
    ))
    if path.exists():
        signed = sha256_and_size(path)
        record["sha256"] = signed["sha256"]
        record["bytes"] = signed["bytes"]
    return record


def build_input_provenance(
    experiment_id: str, plan: "OrderedDict[str, dict]", *, state: dict,
    grid: dict, pathrow: dict, base_dir: Path = PROJECT_ROOT,
) -> dict:
    """Assemble `input_provenance.json`."""
    rasters = [raster_provenance_record(role, entry) for role, entry in plan.items()]
    metadata = [
        json_provenance_record(name, path)
        for name, path in pathrow_boundary_sources(experiment_id, base_dir).items()
    ]
    upstream = [
        json_provenance_record(name, path)
        for name, path in upstream_report_paths(experiment_id, base_dir).items()
    ]
    return OrderedDict((
        ("audit", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("chain_under_attribution", CANDIDATE_CHAIN),
        ("raster_inputs", rasters),
        ("boundary_metadata_inputs", metadata),
        ("upstream_report_inputs", upstream),
        ("grid_contract", grid),
        ("missing_required_inputs", missing_required_inputs(plan)),
        ("missing_optional_inputs", missing_optional_inputs(plan)),
        ("pathrow_evidence", OrderedDict((
            ("availability", pathrow.get("availability")),
            ("reason", pathrow.get("reason")),
            ("interface_count", pathrow.get("interface_count")),
            ("interfaces", pathrow.get("interfaces")),
            ("evidence_qualification",
             "metadata-derived source footprint boundaries; NOT pixel-level "
             "selected-scene provenance"),
        ))),
        ("upstream_prerequisites", OrderedDict((
            ("downstream_ab_final_status", state.get("ab_final_status")),
            ("downstream_ab_candidate_chain", state.get("ab_candidate_chain")),
            ("reference_reproduction_status", state.get("ab_reference_reproduction_status")),
            ("baseline_invariance_status", state.get("ab_baseline_invariance_status")),
            ("candidate_audit_prerequisites_met",
             state.get("ab_candidate_audit_prerequisites_met")),
            ("downstream_ab_production_approved", state.get("ab_production_approved")),
            ("production_approval_required", PRODUCTION_APPROVAL_REQUIRED),
            ("counterfactual_final_status", state.get("counterfactual_final_status")),
            ("prerequisites_met", upstream_prerequisites_met(state)),
            ("upstream_report_hashes", state.get("report_file_hashes")),
        ))),
        ("inherited_warnings", state.get("ab_warnings") or []),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def assert_grid_contract(plan: "OrderedDict[str, dict]") -> dict:
    """Exact grid equality for every 30 m raster used in pairwise decomposition.

    No resampling is ever performed; a mismatch aborts the audit.
    """
    paths = [
        Path(entry["path"]) for entry in plan.values()
        if Path(entry["path"]).exists()
        and entry["family"] in ("target", "target_component", "support", "mask", "covariate")
    ]
    if not paths:
        raise GridMismatchError("no input rasters resolved for the grid contract")
    reference = assert_same_grid(paths)
    return OrderedDict((
        ("required", "exact grid equality (CRS, width, height, transform, bounds); "
                     "no resampling is ever performed"),
        ("reference_grid", reference),
        ("checked_raster_count", len(paths)),
        ("checked_rasters", [str(p) for p in paths]),
        ("passed", True),
    ))


# =============================================================================
# Path/row boundary resolution (metadata-derived; never invented)
# =============================================================================
#: Only LST source roles matter for an LST product's seam. NDVI-only boundaries
#: are excluded so the flag means what its name says.
PATHROW_LST_ROLES = ("current_lst", "baseline_lst")


def resolve_pathrow_availability(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Decide whether metadata-derived path/row evidence exists at all.

    Returns an `availability` of `available`, `incomplete` or `unavailable`.
    When it is not `available` the path/row mechanism is reported as UNAVAILABLE;
    no provenance is invented and no positive evidence can arise from it.
    """
    sources = pathrow_boundary_sources(experiment_id, base_dir)
    summary_path = Path(sources["counterfactual_summary"])
    geojson_path = Path(sources["scene_boundaries_geojson"])

    result: "OrderedDict[str, object]" = OrderedDict((
        ("availability", "unavailable"),
        ("reason", None),
        ("provenance_state", None),
        ("boundaries_path", str(geojson_path)),
        ("footprints_path", str(sources["scene_footprints_geojson"])),
        ("interface_count", 0),
        ("interfaces", []),
        ("feature_count", 0),
        ("lst_feature_count", 0),
    ))

    if not summary_path.exists():
        result["reason"] = f"counterfactual summary not found: {summary_path}"
        return result

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = summary.get("provenance") or {}
    # The frozen summary records the state under `state`, and the nested source
    # summary under `summary.status`. Read BOTH rather than assuming one shape.
    state = provenance.get("state")
    if state is None:
        state = audit.map_provenance_status(provenance.get("summary") or provenance)
    result["provenance_state"] = state

    if state != "provenance_available":
        result["availability"] = (
            "incomplete" if state == "provenance_incomplete" else "unavailable"
        )
        result["reason"] = f"frozen provenance state is {state!r}"
        return result

    if not geojson_path.exists():
        result["reason"] = f"scene_boundaries.geojson not found: {geojson_path}"
        return result

    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = geojson.get("features") or []
    result["feature_count"] = len(features)

    interfaces = summarize_pathrow_interfaces(features)
    result["interfaces"] = [
        OrderedDict((("interface_id", k), ("feature_count", v)))
        for k, v in interfaces.items()
    ]
    result["interface_count"] = len(interfaces)
    result["lst_feature_count"] = int(sum(interfaces.values()))

    if not interfaces:
        result["availability"] = "incomplete"
        result["reason"] = (
            "no verified path_row_boundary features carry an LST source role; the "
            "path/row mechanism is reported as unavailable rather than invented"
        )
        return result

    result["availability"] = "available"
    result["reason"] = (
        f"{result['lst_feature_count']} verified path_row_boundary features across "
        f"{len(interfaces)} distinct metadata-derived interfaces"
    )
    return result


def _feature_is_lst_pathrow_boundary(feature: dict) -> bool:
    props = feature.get("properties") or {}
    if props.get("boundary_type") != "path_row_boundary":
        return False
    if props.get("verification_status") not in (None, "verified"):
        return False
    roles = props.get("source_product_role") or []
    if isinstance(roles, str):
        roles = [roles]
    return bool(set(roles) & set(PATHROW_LST_ROLES))


def pathrow_interface_id(feature: dict) -> str | None:
    """Unordered `<path_row>|<path_row>` interface id for one boundary feature."""
    props = feature.get("properties") or {}
    left = (props.get("left_support") or {}).get("path_row")
    right = (props.get("right_support") or {}).get("path_row")
    if not left or not right or left == right:
        return None
    return "|".join(sorted((str(left), str(right))))


def summarize_pathrow_interfaces(features) -> "OrderedDict[str, int]":
    """Feature count per distinct metadata-derived path/row interface."""
    counts: dict[str, int] = {}
    for feature in features:
        if not _feature_is_lst_pathrow_boundary(feature):
            continue
        interface = pathrow_interface_id(feature)
        if interface is None:
            continue
        counts[interface] = counts.get(interface, 0) + 1
    return OrderedDict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def rasterize_pathrow_boundaries(
    geojson: dict, transform, width: int, height: int,
) -> dict:
    """Burn the verified LST path/row boundary lines onto the EXACT grid.

    All features of one interface are burned in a SINGLE rasterize call, so the
    cost is one pass per interface rather than one per line. Nothing is
    resampled and no geometry is buffered or smoothed.

    Returns ``{"union": bool_mask, "interfaces": {interface_id: bool_mask}}`` on
    the pixel grid (not the edge lattice).
    """
    import numpy as np
    from rasterio.features import rasterize

    grouped: dict[str, list] = {}
    for feature in geojson.get("features") or []:
        if not _feature_is_lst_pathrow_boundary(feature):
            continue
        interface = pathrow_interface_id(feature)
        geometry = feature.get("geometry")
        if interface is None or geometry is None:
            continue
        grouped.setdefault(interface, []).append((geometry, 1))

    union = np.zeros((height, width), dtype=bool)
    interfaces: "OrderedDict[str, object]" = OrderedDict()
    for interface in sorted(grouped):
        burned = rasterize(
            grouped[interface], out_shape=(height, width), transform=transform,
            fill=0, all_touched=True, dtype="uint8",
        ).astype(bool)
        if not burned.any():
            continue
        interfaces[interface] = burned
        union |= burned
    return {"union": union, "interfaces": interfaces}


# =============================================================================
# Pair construction (bounded memory, exact adjacency lattice)
# =============================================================================
def edge_valid_mask(*arrays, orientation: str):
    """Edges where EVERY supplied array is finite at BOTH endpoints.

    Invalid pixels are never zero-filled: an edge touching a NaN endpoint is
    dropped, and the drop is counted.
    """
    import numpy as np

    mask = None
    for array in arrays:
        a, b = _edge_pairs(np.asarray(array, dtype="float64"), orientation)
        valid = np.isfinite(a) & np.isfinite(b)
        mask = valid if mask is None else (mask & valid)
    return mask


def edge_difference(array, orientation: str):
    """`b - a` across every adjacency edge (NaN wherever an endpoint is NaN)."""
    import numpy as np

    a, b = _edge_pairs(np.asarray(array, dtype="float64"), orientation)
    return b - a


def edge_change_flag(array, orientation: str):
    """Edges where an integer support count differs across the edge.

    Both endpoints must be finite; an edge with a missing count is NOT called a
    change (absence of evidence is not a boundary).
    """
    import numpy as np

    a, b = _edge_pairs(np.asarray(array, dtype="float64"), orientation)
    return np.isfinite(a) & np.isfinite(b) & (a != b)


def edge_threshold_straddle(array, orientation: str, threshold: float):
    """Edges whose two endpoints fall on opposite sides of a Step5 threshold."""
    import numpy as np

    a, b = _edge_pairs(np.asarray(array, dtype="float64"), orientation)
    finite = np.isfinite(a) & np.isfinite(b)
    return finite & ((a < threshold) != (b < threshold))


def edge_near_value(array, orientation: str, target: float, epsilon: float):
    """Edges where EITHER endpoint lies within `epsilon` of `target`."""
    import numpy as np

    a, b = _edge_pairs(np.asarray(array, dtype="float64"), orientation)
    near_a = np.isfinite(a) & (np.abs(a - target) <= epsilon)
    near_b = np.isfinite(b) & (np.abs(b - target) <= epsilon)
    return near_a | near_b


def edge_mask_from_pixel_mask(pixel_mask, orientation: str):
    """Edges where either incident pixel is set in a pixel-level mask."""
    a, b = _edge_pairs(pixel_mask, orientation)
    return a | b


def edge_anchor_rows_cols(shape, orientation: str):
    """Flattened row/col of the 'a' endpoint of every adjacency edge."""
    rows, cols = _anchor_indices(shape, orientation)
    return rows.ravel(), cols.ravel()


def endpoint_b_rows_cols(rows, cols, orientation: str):
    """Row/col of the 'b' endpoint given the 'a' endpoint indices."""
    if orientation == "horizontal":
        return rows, cols + 1
    return rows + 1, cols


def spatial_block_ids(rows, cols, *, block_size: int = BOOTSTRAP_BLOCK_SIZE_CELLS):
    """Deterministic spatial-block unit id per pair, from its 'a' endpoint.

    Same convention as the counterfactual audit's `assign_spatial_block_units`,
    kept identical so units are comparable across the two audits.
    """
    import numpy as np

    br = np.asarray(rows, dtype="int64") // int(block_size)
    bc = np.asarray(cols, dtype="int64") // int(block_size)
    return br.astype("int64") * 100000 + bc.astype("int64")


def block_id_to_label(block_id: int) -> str:
    return f"block_r{int(block_id) // 100000}_c{int(block_id) % 100000}"


def gradient_bin(values, edges):
    """Right-open bin index of |gradient| against predeclared edges.

    NaN maps to -1 ('unmatched'), which is treated as its own stratum so a pair
    with a missing covariate is never silently matched against one that has it.
    """
    import numpy as np

    values = np.asarray(values, dtype="float64")
    out = np.full(values.shape, -1, dtype="int16")
    finite = np.isfinite(values)
    if finite.any():
        out[finite] = np.digitize(np.abs(values[finite]), np.asarray(edges, dtype="float64"))
    return out


# =============================================================================
# Exact decompositions
# =============================================================================
def decompose_current_minus_baseline(current_a, current_b, mean_a, mean_b):
    """Q1: exact additive split of the current-minus-baseline jump.

        target_jump        = D_B - D_A     with D = C - M
        current_component  = C_B - C_A
        baseline_component = -(M_B - M_A)

    Returns ``(target_jump, current_component, baseline_component)`` computed in
    float64 from the stored float32 endpoints.
    """
    import numpy as np

    current_a = np.asarray(current_a, dtype="float64")
    current_b = np.asarray(current_b, dtype="float64")
    mean_a = np.asarray(mean_a, dtype="float64")
    mean_b = np.asarray(mean_b, dtype="float64")

    current_component = current_b - current_a
    baseline_component = -(mean_b - mean_a)
    target_jump = current_component + baseline_component
    return target_jump, current_component, baseline_component


def decompose_anomaly(d_a, d_b, s_a, s_b):
    """Q2: EXACT symmetric numerator/denominator split of the anomaly jump.

        numerator_contribution   = 0.5 * (1/S_A + 1/S_B) * (D_B - D_A)
        denominator_contribution = 0.5 * (D_A + D_B)     * (1/S_B - 1/S_A)

    The identity `Z_B - Z_A == numerator + denominator` is algebraically exact
    (not a Taylor expansion): with a = 1/S_A, b = 1/S_B, x = D_A, y = D_B,

        0.5*(a+b)*(y-x) + 0.5*(x+y)*(b-a)
          = 0.5*(ay - ax + by - bx + xb - xa + yb - ya)
          = by - ax

    Returns ``(z_a, z_b, numerator_contribution, denominator_contribution)``.
    """
    import numpy as np

    d_a = np.asarray(d_a, dtype="float64")
    d_b = np.asarray(d_b, dtype="float64")
    s_a = np.asarray(s_a, dtype="float64")
    s_b = np.asarray(s_b, dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_a = 1.0 / s_a
        inv_b = 1.0 / s_b
        z_a = d_a * inv_a
        z_b = d_b * inv_b
        numerator = 0.5 * (inv_a + inv_b) * (d_b - d_a)
        denominator = 0.5 * (d_a + d_b) * (inv_b - inv_a)
    return z_a, z_b, numerator, denominator


def reconstruction_residual(target, *components):
    """`target - sum(components)` -- the residual a tolerance gate checks."""
    import numpy as np

    total = np.zeros_like(np.asarray(target, dtype="float64"))
    for component in components:
        total = total + np.asarray(component, dtype="float64")
    return np.asarray(target, dtype="float64") - total


def classify_signed_interaction(component_a, component_b):
    """Cancellation vs reinforcement for a two-component split.

    Returns ``(cancelling, reinforcing, degenerate)`` boolean arrays. A pair
    where either component is exactly zero is DEGENERATE and is never forced
    into either category.
    """
    import numpy as np

    a = np.asarray(component_a, dtype="float64")
    b = np.asarray(component_b, dtype="float64")
    finite = np.isfinite(a) & np.isfinite(b)
    zero = (np.abs(a) <= CANCELLATION_ZERO_EPS) | (np.abs(b) <= CANCELLATION_ZERO_EPS)
    degenerate = finite & zero
    cancelling = finite & ~zero & ((a > 0) != (b > 0))
    reinforcing = finite & ~zero & ((a > 0) == (b > 0))
    return cancelling, reinforcing, degenerate


def component_share(component_a, component_b):
    """`|a| / (|a| + |b|)` -- NaN where both components are exactly zero."""
    import numpy as np

    a = np.abs(np.asarray(component_a, dtype="float64"))
    b = np.abs(np.asarray(component_b, dtype="float64"))
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0.0, a / total, np.nan)
    return share


# =============================================================================
# Streaming accumulators (bounded memory)
# =============================================================================
class MeanAccumulator:
    """Per-unit sum/count for a mean statistic.

    Cluster bootstrap of a pooled mean over resampled units only needs each
    unit's sum and count, so an EXACT bootstrap runs in O(units) memory instead
    of holding every pair.
    """

    __slots__ = ("sums", "counts")

    def __init__(self) -> None:
        self.sums: dict[int, float] = {}
        self.counts: dict[int, int] = {}

    def add(self, unit_ids, values) -> None:
        import numpy as np

        values = np.asarray(values, dtype="float64")
        finite = np.isfinite(values)
        if not finite.any():
            return
        units = np.asarray(unit_ids, dtype="int64")[finite]
        values = values[finite]
        for unit, total, count in zip(*_group_sums(units, values)):
            self.sums[int(unit)] = self.sums.get(int(unit), 0.0) + float(total)
            self.counts[int(unit)] = self.counts.get(int(unit), 0) + int(count)

    @property
    def n_units(self) -> int:
        return len([u for u, c in self.counts.items() if c > 0])

    @property
    def n_pairs(self) -> int:
        return int(sum(self.counts.values()))

    def point_estimate(self) -> float | None:
        total = sum(self.sums.values())
        count = self.n_pairs
        return (total / count) if count else None

    def unit_arrays(self):
        import numpy as np

        units = sorted(u for u, c in self.counts.items() if c > 0)
        sums = np.array([self.sums[u] for u in units], dtype="float64")
        counts = np.array([self.counts[u] for u in units], dtype="float64")
        return units, sums, counts


def _group_sums(units, values):
    """Grouped (unit, sum, count) for one batch -- vectorised, no Python loop."""
    import numpy as np

    order = np.argsort(units, kind="stable")
    units_sorted = units[order]
    values_sorted = values[order]
    boundaries = np.flatnonzero(np.diff(units_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [units_sorted.size]))
    cumulative = np.concatenate(([0.0], np.cumsum(values_sorted)))
    sums = cumulative[ends] - cumulative[starts]
    counts = ends - starts
    return units_sorted[starts], sums, counts


class HistogramAccumulator:
    """Fixed-edge histogram of |jump| for deterministic descriptive quantiles."""

    __slots__ = ("edges", "counts", "overflow", "total")

    def __init__(self, maximum: float, bins: int = HISTOGRAM_BINS) -> None:
        import numpy as np

        self.edges = np.linspace(0.0, float(maximum), int(bins) + 1)
        self.counts = np.zeros(int(bins), dtype="int64")
        self.overflow = 0
        self.total = 0

    def add(self, values) -> None:
        import numpy as np

        values = np.asarray(values, dtype="float64")
        values = values[np.isfinite(values)]
        if not values.size:
            return
        self.total += int(values.size)
        over = values > self.edges[-1]
        self.overflow += int(over.sum())
        inside = values[~over]
        if inside.size:
            self.counts += np.histogram(inside, bins=self.edges)[0]

    def quantile(self, percentile: float) -> float | None:
        """Descriptive quantile from the histogram (bin upper edge)."""
        import numpy as np

        if not self.total:
            return None
        target = (float(percentile) / 100.0) * self.total
        cumulative = np.cumsum(self.counts)
        if target > cumulative[-1]:
            # Falls inside the overflow tail; report the histogram ceiling.
            return float(self.edges[-1])
        index = int(np.searchsorted(cumulative, target, side="left"))
        index = min(index, self.counts.size - 1)
        return float(self.edges[index + 1])

    def describe(self) -> dict:
        return OrderedDict((
            ("bins", int(self.counts.size)),
            ("range_max", float(self.edges[-1])),
            ("bin_width", float(self.edges[1] - self.edges[0])),
            ("total_values", int(self.total)),
            ("overflow_values", int(self.overflow)),
            ("quantile_resolution_note",
             "Descriptive percentiles are read from fixed-edge histogram bins, so "
             "each is accurate to one bin width. They are MAP thresholds only."),
        ))


class ReservoirSampler:
    """Deterministic reservoir sample of pair records (bounded memory)."""

    __slots__ = ("size", "rng", "seen", "rows")

    def __init__(self, size: int = PAIR_SAMPLE_SIZE, seed: int = PAIR_SAMPLE_SEED) -> None:
        import numpy as np

        self.size = int(size)
        self.rng = np.random.default_rng(int(seed))
        self.seen = 0
        self.rows: list[dict] = []

    def offer_batch(self, rows: list[dict]) -> None:
        for row in rows:
            self.seen += 1
            if len(self.rows) < self.size:
                self.rows.append(row)
            else:
                index = int(self.rng.integers(0, self.seen))
                if index < self.size:
                    self.rows[index] = row


# =============================================================================
# Cluster bootstrap over spatial units (identical draws by construction)
# =============================================================================
def draw_bootstrap_indices(n_units: int, *, replicates: int = BOOTSTRAP_REPLICATES,
                           seed: int = BOOTSTRAP_SEED):
    """The ONE set of unit draws reused by every statistic in a comparison.

    Units -- never individual pixel pairs -- are resampled with replacement. The
    same index matrix is handed to every statistic computed on the same pair
    population, so component shares are always compared on identical draws.
    """
    import numpy as np

    rng = np.random.default_rng(int(seed))
    if n_units <= 0:
        return np.empty((0, 0), dtype="int64")
    return rng.integers(0, n_units, size=(int(replicates), int(n_units)), dtype="int64")


def bootstrap_mean_interval(
    accumulator: MeanAccumulator, indices=None, *,
    replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
    min_units: int = MIN_BOOTSTRAP_UNITS, ci: float = BOOTSTRAP_CI,
) -> dict:
    """Cluster percentile bootstrap of a pooled mean over spatial units.

    A replicate whose resampled units carry ZERO pairs is skipped and counted
    rather than silently contributing a NaN.
    """
    import numpy as np

    units, sums, counts = accumulator.unit_arrays()
    n_units = len(units)
    result = OrderedDict((
        ("n_pairs", accumulator.n_pairs),
        ("n_units", n_units),
        ("unit_type", "spatial_block"),
        ("block_size_cells", BOOTSTRAP_BLOCK_SIZE_CELLS),
        ("n_bootstrap_requested", int(replicates)),
        ("n_bootstrap_used", 0),
        ("n_bootstrap_skipped", 0),
        ("skipped_reason", None),
        ("seed", int(seed)),
        ("ci", float(ci)),
        ("min_units_required", int(min_units)),
        ("point_estimate", accumulator.point_estimate()),
        ("interval_low", None),
        ("interval_high", None),
        ("status", "insufficient_units"),
    ))
    if n_units < int(min_units):
        result["skipped_reason"] = (
            f"only {n_units} independent spatial units (< {min_units} required)"
        )
        return result

    if indices is None:
        indices = draw_bootstrap_indices(n_units, replicates=replicates, seed=seed)
    if indices.shape[1] != n_units:
        raise ResidualSeamError(
            f"bootstrap index matrix has {indices.shape[1]} columns but the "
            f"accumulator has {n_units} units; identical draws require identical "
            "unit ordering"
        )

    replicate_sums = sums[indices].sum(axis=1)
    replicate_counts = counts[indices].sum(axis=1)
    usable = replicate_counts > 0
    result["n_bootstrap_used"] = int(usable.sum())
    result["n_bootstrap_skipped"] = int((~usable).sum())
    if result["n_bootstrap_skipped"]:
        result["skipped_reason"] = "resampled units contained zero retained pairs"
    if not usable.any():
        result["status"] = "no_usable_replicates"
        return result

    means = replicate_sums[usable] / replicate_counts[usable]
    lo_q = (1.0 - float(ci)) / 2.0
    result["interval_low"] = float(np.quantile(means, lo_q))
    result["interval_high"] = float(np.quantile(means, 1.0 - lo_q))
    result["status"] = "estimated"
    return result


def bootstrap_difference_interval(
    boundary: MeanAccumulator, control: MeanAccumulator, *,
    replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
    min_units: int = MIN_BOOTSTRAP_UNITS, ci: float = BOOTSTRAP_CI,
) -> dict:
    """Excess = pooled boundary mean - pooled matched-control mean.

    Both arms are resampled on the SAME unit draw, so a block contributes its
    boundary and control pairs together and the difference is genuinely paired.
    Only units carrying BOTH arms are eligible.
    """
    import numpy as np

    shared_units = sorted(
        u for u in boundary.counts
        if boundary.counts.get(u, 0) > 0 and control.counts.get(u, 0) > 0
    )
    n_units = len(shared_units)
    result = OrderedDict((
        ("n_boundary_pairs", int(sum(boundary.counts.get(u, 0) for u in shared_units))),
        ("n_control_pairs", int(sum(control.counts.get(u, 0) for u in shared_units))),
        ("n_units", n_units),
        ("unit_type", "spatial_block"),
        ("block_size_cells", BOOTSTRAP_BLOCK_SIZE_CELLS),
        ("n_bootstrap_requested", int(replicates)),
        ("n_bootstrap_used", 0),
        ("n_bootstrap_skipped", 0),
        ("skipped_reason", None),
        ("seed", int(seed)),
        ("ci", float(ci)),
        ("min_units_required", int(min_units)),
        ("boundary_mean_abs_jump", None),
        ("control_mean_abs_jump", None),
        ("excess_absolute_jump", None),
        ("interval_low", None),
        ("interval_high", None),
        ("status", "insufficient_units"),
    ))
    if n_units == 0:
        result["skipped_reason"] = "no spatial block carries both boundary and control pairs"
        return result

    b_sums = np.array([boundary.sums[u] for u in shared_units], dtype="float64")
    b_counts = np.array([boundary.counts[u] for u in shared_units], dtype="float64")
    c_sums = np.array([control.sums[u] for u in shared_units], dtype="float64")
    c_counts = np.array([control.counts[u] for u in shared_units], dtype="float64")

    result["boundary_mean_abs_jump"] = float(b_sums.sum() / b_counts.sum())
    result["control_mean_abs_jump"] = float(c_sums.sum() / c_counts.sum())
    result["excess_absolute_jump"] = (
        result["boundary_mean_abs_jump"] - result["control_mean_abs_jump"]
    )
    if n_units < int(min_units):
        result["skipped_reason"] = (
            f"only {n_units} matched spatial units (< {min_units} required)"
        )
        return result

    indices = draw_bootstrap_indices(n_units, replicates=replicates, seed=seed)
    bs = b_sums[indices].sum(axis=1)
    bc = b_counts[indices].sum(axis=1)
    cs = c_sums[indices].sum(axis=1)
    cc = c_counts[indices].sum(axis=1)
    usable = (bc > 0) & (cc > 0)
    result["n_bootstrap_used"] = int(usable.sum())
    result["n_bootstrap_skipped"] = int((~usable).sum())
    if result["n_bootstrap_skipped"]:
        result["skipped_reason"] = "resampled units contained an empty arm"
    if not usable.any():
        result["status"] = "no_usable_replicates"
        return result

    excess = bs[usable] / bc[usable] - cs[usable] / cc[usable]
    lo_q = (1.0 - float(ci)) / 2.0
    result["interval_low"] = float(np.quantile(excess, lo_q))
    result["interval_high"] = float(np.quantile(excess, 1.0 - lo_q))
    result["status"] = "estimated"
    return result


def interval_wholly_above(interval: dict, value: float = 0.0) -> bool:
    low = interval.get("interval_low")
    return bool(low is not None and low > value)


def interval_wholly_below(interval: dict, value: float = 0.0) -> bool:
    high = interval.get("interval_high")
    return bool(high is not None and high < value)


def classify_excess_interval(interval: dict) -> str:
    """`supported_excess` / `supported_deficit` / `uncertain` / `insufficient_evidence`."""
    if interval.get("status") != "estimated":
        return "insufficient_evidence"
    if interval_wholly_above(interval):
        return "supported_excess"
    if interval_wholly_below(interval):
        return "supported_deficit"
    return "uncertain"


# =============================================================================
# Predeclared attribution logic (ORDERED -- never relaxed after seeing results)
# =============================================================================
def decide_final_status(evidence: dict) -> dict:
    """Apply the predeclared, ordered attribution rule.

    Order: invalid_inputs -> residual_not_detected -> current_support_dominant ->
    baseline_support_dominant -> baseline_variance_amplification_dominant ->
    pathrow_bias_supported -> mixed_mechanisms -> residual_mechanism_inconclusive.

    Every rule is evaluated from bootstrap INTERVALS, never from a point
    estimate and never from a signed mean alone.
    """
    reasons: list[str] = []
    supported: "OrderedDict[str, bool]" = OrderedDict()

    # --- 1. invalid inputs -------------------------------------------------
    if not evidence.get("inputs_valid", False):
        return _status(STATUS_INVALID_INPUTS, [
            "a required frozen input, the grid contract or an upstream "
            "prerequisite did not hold: "
            f"{evidence.get('invalid_input_reasons')}",
        ], evidence, supported)

    excess = evidence.get("excess_by_boundary") or {}

    def _supported(name: str) -> bool:
        return classify_excess_interval(excess.get(name) or {}) == "supported_excess"

    current_definitions = [
        name for name in CURRENT_SUPPORT_FLAGS + ("current_support_change",)
        if _supported(name)
    ]
    baseline_definitions = [name for name in BASELINE_SUPPORT_FLAGS if _supported(name)]
    threshold_definitions = [name for name in THRESHOLD_FLAGS if _supported(name)]
    pathrow_only = evidence.get("pathrow_only") or {}

    supported["current_support_boundary_excess"] = bool(current_definitions)
    supported["baseline_support_boundary_excess"] = bool(baseline_definitions)
    supported["threshold_boundary_excess"] = bool(threshold_definitions)
    supported["pathrow_only_excess"] = bool(pathrow_only.get("supported"))

    # --- 2. residual not detected ------------------------------------------
    any_excess = any((
        supported["current_support_boundary_excess"],
        supported["baseline_support_boundary_excess"],
        supported["threshold_boundary_excess"],
        supported["pathrow_only_excess"],
        _supported(CLASS_SUPPORT_ONLY),
        _supported(CLASS_SUPPORT_AND_PATHROW),
        _supported("source_path_row_boundary"),
    ))
    if not any_excess:
        return _status(STATUS_RESIDUAL_NOT_DETECTED, [
            "no tested boundary definition shows an excess absolute jump whose "
            "bootstrap interval lies wholly above zero relative to matched "
            "within-block controls",
        ], evidence, supported)

    cmb_current = evidence.get("cmb_current_share") or {}
    cmb_baseline = evidence.get("cmb_baseline_share") or {}
    anomaly_numerator = evidence.get("anomaly_numerator_share") or {}
    anomaly_denominator = evidence.get("anomaly_denominator_share") or {}

    current_share_dominant = interval_wholly_above(cmb_current, DOMINANCE_SHARE_LOWER_BOUND)
    baseline_share_dominant = interval_wholly_above(cmb_baseline, DOMINANCE_SHARE_LOWER_BOUND)
    numerator_dominant = interval_wholly_above(anomaly_numerator, DOMINANCE_SHARE_LOWER_BOUND)
    denominator_dominant = interval_wholly_above(
        anomaly_denominator, DOMINANCE_SHARE_LOWER_BOUND
    )
    supported["cmb_current_share_dominant"] = current_share_dominant
    supported["cmb_baseline_share_dominant"] = baseline_share_dominant
    supported["anomaly_numerator_share_dominant"] = numerator_dominant
    supported["anomaly_denominator_share_dominant"] = denominator_dominant

    # --- 3. current_support_dominant ---------------------------------------
    if (
        supported["current_support_boundary_excess"]
        and current_share_dominant
        and len(current_definitions) >= MIN_CURRENT_SUPPORT_DEFINITIONS
        and not baseline_share_dominant
    ):
        reasons.append(
            "current-support boundaries show a supported excess jump in "
            f"{len(current_definitions)} definitions ({current_definitions}); the "
            "current-component share interval lies wholly above "
            f"{DOMINANCE_SHARE_LOWER_BOUND}; no contradictory baseline dominance"
        )
        return _status(STATUS_CURRENT_SUPPORT, reasons, evidence, supported)

    # --- 4. baseline_support_dominant --------------------------------------
    baseline_excluding_current = evidence.get("baseline_excess_excluding_current_only") or {}
    if (
        baseline_share_dominant
        and supported["baseline_support_boundary_excess"]
        and classify_excess_interval(baseline_excluding_current) == "supported_excess"
    ):
        reasons.append(
            "the baseline-mean component share interval lies wholly above "
            f"{DOMINANCE_SHARE_LOWER_BOUND}; baseline support boundaries "
            f"({baseline_definitions}) show a supported excess jump that survives "
            "excluding current-support-only pairs"
        )
        return _status(STATUS_BASELINE_SUPPORT, reasons, evidence, supported)

    # --- 5. baseline_variance_amplification_dominant -----------------------
    std_concentration = evidence.get("anomaly_std_concentration") or {}
    mask_near_threshold = evidence.get("mask_discontinuity_near_std_threshold") or {}
    epsilon_support = evidence.get("near_std_epsilon_support") or {}
    multi_epsilon = sum(1 for v in epsilon_support.values() if v) >= 2
    supported["anomaly_low_std_concentration"] = bool(std_concentration.get("supported"))
    supported["mask_discontinuity_elevated_near_threshold"] = bool(
        mask_near_threshold.get("elevated")
    )
    supported["near_std_effect_robust_across_epsilons"] = multi_epsilon
    if (
        denominator_dominant
        and supported["anomaly_low_std_concentration"]
        and supported["mask_discontinuity_elevated_near_threshold"]
        and multi_epsilon
    ):
        reasons.append(
            "the anomaly denominator-contribution share interval lies wholly above "
            f"{DOMINANCE_SHARE_LOWER_BOUND}; residual anomaly jumps concentrate near "
            "low or threshold-adjacent baseline standard deviation; mask "
            "discontinuities are elevated in the same zone; the effect holds at more "
            "than one predeclared epsilon"
        )
        return _status(STATUS_BASELINE_VARIANCE, reasons, evidence, supported)

    # --- 6. pathrow_bias_supported -----------------------------------------
    if pathrow_only.get("supported"):
        reasons.append(
            "path/row-only pairs (carrying no observation-support and no threshold "
            "boundary) show an excess absolute jump whose interval lies wholly above "
            f"zero across {pathrow_only.get('n_units')} spatial units and "
            f"{pathrow_only.get('n_interfaces')} distinct metadata-derived interfaces"
        )
        return _status(STATUS_PATHROW, reasons, evidence, supported)

    # --- 7. mixed_mechanisms ------------------------------------------------
    independent = [name for name, ok in supported.items() if ok and name.endswith("_excess")]
    if len(independent) >= 2:
        reasons.append(
            f"at least two mechanisms have independently supported evidence "
            f"({independent}) and none satisfies its predeclared dominance rule"
        )
        return _status(STATUS_MIXED, reasons, evidence, supported)

    # --- 8. inconclusive ----------------------------------------------------
    reasons.append(
        "the run is technically valid but no predeclared dominance rule is "
        f"satisfied; supported evidence: {[n for n, ok in supported.items() if ok]}"
    )
    return _status(STATUS_INCONCLUSIVE, reasons, evidence, supported)


def _status(status: str, reasons: list[str], evidence: dict, supported: dict) -> dict:
    if status not in FINAL_STATUSES:
        raise ResidualSeamError(f"undeclared final status: {status!r}")
    return OrderedDict((
        ("final_status", status),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("decision_rule_order", list(FINAL_STATUSES)),
        ("meaning", FINAL_STATUS_MEANINGS[status]),
        ("reasons", reasons),
        ("supported_mechanisms", OrderedDict(supported)),
        ("secondary_supported_mechanisms",
         [name for name, ok in supported.items() if ok]),
        ("seam_fixed", False),
        ("production_approved", False),
        ("changes_production_reducer", False),
        ("evidence_snapshot", evidence),
    ))


# =============================================================================
# Limitations (required, verbatim scope statements)
# =============================================================================
def required_limitations() -> list[str]:
    return [
        "Single AOI: manavgat_2021 only; nothing here generalises to another region.",
        "Path/row evidence is metadata-derived from frozen source-scene footprints.",
        "No pixel-level selected-scene provenance exists, so a path/row flag says a "
        "metadata boundary crosses the pair, not which scene supplied the pixel.",
        "No causal identification: every association is descriptive attribution, not "
        "a causal effect.",
        "No smoothing or visual correction is applied anywhere; the seam is measured, "
        "never removed.",
        "Matched controls cannot remove every natural spatial confounder; terrain and "
        "land-cover structure can align with acquisition geometry.",
        "The baseline climatology contains only four years, so baseline-mean and "
        "baseline-standard-deviation estimates are themselves noisy.",
        "The result is conditional on the frozen candidate inputs of the completed "
        "downstream A/B experiment.",
        "A dominant mechanism does not prove it is the only mechanism.",
        "No production reducer decision follows from this audit.",
    ]


def inherited_limitations(state: dict) -> list[str]:
    """Limitations carried forward from the frozen upstream A/B run."""
    items: list[str] = []
    for warning in state.get("ab_warnings") or []:
        code = warning.get("code")
        effect = warning.get("scientific_effect")
        if code and effect:
            items.append(f"Inherited from the downstream A/B run [{code}]: {effect}")
    return items


# =============================================================================
# Checkpointing (atomic; checkpoint TEXT never bypasses file validation)
# =============================================================================
CHECKPOINT_FILENAME = "residual_seam_checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = "1.0-residual-seam"

PLANNED_STAGES = (
    "input_validation",
    "pair_mask_construction",
    "current_minus_baseline_decomposition",
    "anomaly_decomposition",
    "mask_boundary_analysis",
    "matched_control_analysis",
    "pathrow_analysis",
    "bootstrap",
    "map_generation",
    "report_generation",
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
    """Size AND sha256, so resume validates content, not just presence."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "bytes": -1, "sha256": None}
    signed = sha256_and_size(path)
    return {"path": str(path), "bytes": signed["bytes"], "sha256": signed["sha256"]}


def write_checkpoint_stage(root: Path, stage: str, outputs, extra: dict | None = None) -> dict:
    """Atomically record a completed stage with its output hashes."""
    if stage not in PLANNED_STAGES:
        raise ResidualSeamError(f"unknown checkpoint stage: {stage!r}")
    root = Path(root)
    payload = read_checkpoint(root)
    payload.setdefault("audit", DIAGNOSTIC_NAMESPACE)
    payload["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    payload.setdefault("stages", {})
    payload["stages"][stage] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "outputs": [file_reference(p) for p in outputs],
        **(extra or {}),
    }
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
# Configuration snapshot
# =============================================================================
def build_config_snapshot(experiment_id: str) -> dict:
    """Everything predeclared, in one frozen record."""
    return OrderedDict((
        ("audit", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("chain_under_attribution", CANDIDATE_CHAIN),
        ("target_products", list(TARGET_PRODUCTS)),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("decompositions", decomposition_formulas()),
        ("boundary_flags", list(BOUNDARY_FLAGS)),
        ("stratified_classes", list(STRATIFIED_CLASSES)),
        ("step5_thresholds", step5_thresholds()),
        ("near_std_threshold_epsilon", OrderedDict((
            ("primary", STD_THRESHOLD_EPSILON_PRIMARY),
            ("sensitivity", list(STD_THRESHOLD_EPSILON_SENSITIVITY)),
            ("all_reported", list(STD_THRESHOLD_EPSILONS)),
            ("predeclared_before_inspecting_results", True),
        ))),
        ("reconstruction_checks", OrderedDict((
            ("current_minus_baseline", OrderedDict((
                ("check", "algebraic_identity"),
                ("tolerance_absolute", CMB_RECONSTRUCTION_ABS_TOL),
                ("gates_the_audit", True),
            ))),
            ("anomaly_algebraic_identity", OrderedDict((
                ("check", "algebraic_identity"),
                ("computed_in", "float64"),
                ("tolerance_absolute", ANOMALY_IDENTITY_ABS_TOL),
                ("tolerance_relative", ANOMALY_IDENTITY_REL_TOL),
                ("tolerance_policy", ANOMALY_IDENTITY_TOLERANCE_POLICY),
                ("gates_the_audit", True),
            ))),
            ("anomaly_stored_raster_reproduction", OrderedDict((
                ("check", "stored_raster_reproduction"),
                ("predeclared_tolerance", ANOMALY_STORED_REPRODUCTION_TOL),
                ("predeclared_tolerance_source", ANOMALY_STORED_REPRODUCTION_TOL_SOURCE),
                ("step5_reproduction_policy_tolerance",
                 ANOMALY_STORED_REPRODUCTION_POLICY_TOL),
                ("step5_reproduction_policy_source",
                 ANOMALY_STORED_REPRODUCTION_POLICY_SOURCE),
                ("reported_percentiles", list(ANOMALY_STORED_REPRODUCTION_PERCENTILES)),
                ("gates_the_audit", False),
                ("is_decomposition_failure", False),
                ("float32_serialization_error_is_expected", True),
            ))),
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
            ("identical_draws_for_component_comparison", True),
        ))),
        ("matched_controls", OrderedDict((
            ("strategy", MATCHED_CONTROL_STRATEGY),
            ("elevation_gradient_bins_m", list(ELEVATION_GRADIENT_BINS)),
            ("slope_gradient_bins_deg", list(SLOPE_GRADIENT_BINS)),
            ("ndvi_gradient_bins", list(NDVI_GRADIENT_BINS)),
        ))),
        ("hotspots", OrderedDict((
            ("percentiles", list(HOTSPOT_PERCENTILES)),
            ("labels", {str(k): v for k, v in HOTSPOT_LABELS.items()}),
            ("descriptive_only", True),
            ("is_significance_threshold", False),
            ("note", "Hotspot percentiles select pairs for DESCRIPTIVE MAPS only. "
                     "Primary inference uses continuous pairwise jumps."),
        ))),
        ("pathrow_test", OrderedDict((
            ("min_units", MIN_PATHROW_ONLY_UNITS),
            ("min_interfaces", MIN_PATHROW_INTERFACES),
            ("insufficient_status", "insufficient_pathrow_only_support"),
            ("evidence_type", "metadata_derived_source_footprint_boundaries"),
        ))),
        ("dominance", OrderedDict((
            ("share_interval_lower_bound", DOMINANCE_SHARE_LOWER_BOUND),
            ("min_current_support_definitions", MIN_CURRENT_SUPPORT_DEFINITIONS),
        ))),
        ("windowing", OrderedDict((
            ("window_rows", WINDOW_ROWS),
            ("halo_rows", 1),
            ("histogram_bins", HISTOGRAM_BINS),
            ("pair_sample_size", PAIR_SAMPLE_SIZE),
            ("pair_sample_seed", PAIR_SAMPLE_SEED),
            ("writes_every_pair", False),
        ))),
        ("allowed_final_statuses", list(FINAL_STATUSES)),
        ("forbidden_conclusions", list(FORBIDDEN_CONCLUSIONS)),
        ("smoothing_applied", False),
        ("earth_engine_used", False),
        ("reruns_step5_to_step8", False),
        ("modifies_production_reducer", False),
    ))


def decomposition_formulas() -> dict:
    """The exact formulas, printed by the dry-run and stored in every report."""
    return OrderedDict((
        ("current_minus_baseline", OrderedDict((
            ("definition", "D = C - M  (C = current LST, M = baseline mean)"),
            ("identity", "delta(D) = delta(C) - delta(M)"),
            ("target_jump", "D_B - D_A"),
            ("current_component", "C_B - C_A"),
            ("baseline_component", "-(M_B - M_A)"),
            ("exactness", "additive identity; exact in real arithmetic"),
            ("tolerance", CMB_RECONSTRUCTION_ABS_TOL),
        ))),
        ("anomaly_zscore", OrderedDict((
            ("definition", "D = current_minus_baseline, S = baseline_std, Z = D / S"),
            ("numerator_contribution", "0.5 * (1/S_A + 1/S_B) * (D_B - D_A)"),
            ("denominator_contribution", "0.5 * (D_A + D_B) * (1/S_B - 1/S_A)"),
            ("identity", "Z_B - Z_A == numerator_contribution + denominator_contribution"),
            ("exactness",
             "EXACT symmetric decomposition, not a Taylor expansion: with "
             "a=1/S_A, b=1/S_B, x=D_A, y=D_B the two terms sum to b*y - a*x"),
            ("check_1_algebraic_identity", OrderedDict((
                ("recomputes", "Z_A = D_A/S_A and Z_B = D_B/S_B in float64"),
                ("verifies",
                 "Z_B - Z_A == numerator_contribution + denominator_contribution"),
                ("tolerance_absolute", ANOMALY_IDENTITY_ABS_TOL),
                ("tolerance_relative", ANOMALY_IDENTITY_REL_TOL),
                ("tolerance_policy", ANOMALY_IDENTITY_TOLERANCE_POLICY),
                ("gates_the_audit", True),
            ))),
            ("check_2_stored_raster_reproduction", OrderedDict((
                ("compares",
                 "recomputed float64 D/S against the stored float32 "
                 "anomaly_zscore raster"),
                ("predeclared_tolerance", ANOMALY_STORED_REPRODUCTION_TOL),
                ("predeclared_tolerance_source", ANOMALY_STORED_REPRODUCTION_TOL_SOURCE),
                ("step5_reproduction_policy_tolerance",
                 ANOMALY_STORED_REPRODUCTION_POLICY_TOL),
                ("step5_reproduction_policy_source",
                 ANOMALY_STORED_REPRODUCTION_POLICY_SOURCE),
                ("reported_percentiles", list(ANOMALY_STORED_REPRODUCTION_PERCENTILES)),
                ("gates_the_audit", False),
                ("is_decomposition_failure", False),
            ))),
        ))),
    ))


# =============================================================================
# Dry-run plan
# =============================================================================
def anomaly_reconstruction_check_declaration() -> dict:
    """The TWO anomaly checks as predeclared policy, with no results attached.

    Printed by the dry-run so the separation, the tolerances and their sources
    are visible before anything is computed.
    """
    return OrderedDict((
        ("separation_rationale",
         "The two checks answer different questions. The algebraic identity "
         "check tests the DECOMPOSITION and gates the audit. The stored-raster "
         "reproduction check tests float32 SERIALIZATION and never does."),
        ("algebraic_identity_check", OrderedDict((
            ("check", "algebraic_identity"),
            ("question",
             "Recomputing Z_A = D_A/S_A and Z_B = D_B/S_B in float64, does "
             "Z_B - Z_A equal numerator_contribution + denominator_contribution?"),
            ("computed_in", "float64"),
            ("tolerance_absolute", ANOMALY_IDENTITY_ABS_TOL),
            ("tolerance_relative", ANOMALY_IDENTITY_REL_TOL),
            ("tolerance_policy", ANOMALY_IDENTITY_TOLERANCE_POLICY),
            ("gates_the_audit", True),
            ("failure_meaning",
             "a failure here means the decomposition itself is wrong, and the "
             "audit terminates as invalid_inputs"),
            ("results_available", False),
        ))),
        ("stored_raster_reproduction_check", OrderedDict((
            ("check", "stored_raster_reproduction"),
            ("question",
             "How closely does the recomputed float64 D/S reproduce the stored "
             "float32 anomaly_zscore raster?"),
            ("compared", "float64 D/S (recomputed) vs stored float32 anomaly_zscore"),
            ("predeclared_tolerance", ANOMALY_STORED_REPRODUCTION_TOL),
            ("predeclared_tolerance_source", ANOMALY_STORED_REPRODUCTION_TOL_SOURCE),
            ("step5_reproduction_policy_tolerance",
             ANOMALY_STORED_REPRODUCTION_POLICY_TOL),
            ("step5_reproduction_policy_source",
             ANOMALY_STORED_REPRODUCTION_POLICY_SOURCE),
            ("reported_percentiles", list(ANOMALY_STORED_REPRODUCTION_PERCENTILES)),
            ("reported_statistics", ["max_abs_error", "fraction_within_tolerance"]),
            ("gates_the_audit", False),
            ("is_decomposition_failure", False),
            ("interpretation",
             "Step5 divided its own internal float32 difference and serialised "
             "the quotient to float32. A residual of this size is EXPECTED "
             "serialization error and is never treated as a decomposition "
             "failure."),
            ("results_available", False),
        ))),
    ))


def build_dry_run_plan(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Resolve everything a dry-run must print. Creates and writes NOTHING."""
    assert_supported_experiment(experiment_id)

    plan = build_input_plan(experiment_id, base_dir)
    layout = plan_output_layout(experiment_id, base_dir)
    expected = plan_expected_files(experiment_id, base_dir)
    state = load_upstream_state(experiment_id, base_dir)
    pathrow = resolve_pathrow_availability(experiment_id, base_dir)

    resolved: "OrderedDict[str, dict]" = OrderedDict()
    for role, entry in plan.items():
        resolved[role] = OrderedDict((
            ("path", str(entry["path"])),
            ("present", Path(entry["path"]).exists()),
            ("required", entry["required"]),
            ("source_chain", entry["source_chain"]),
            ("purpose", entry["purpose"]),
        ))

    return OrderedDict((
        ("audit", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("chain_under_attribution", CANDIDATE_CHAIN),
        ("target_products", list(TARGET_PRODUCTS)),
        ("resolved_inputs", resolved),
        ("missing_required_inputs", missing_required_inputs(plan)),
        ("missing_optional_provenance_inputs", missing_optional_inputs(plan)),
        ("pathrow_evidence", pathrow),
        ("upstream_prerequisites", OrderedDict((
            ("downstream_ab_final_status", state.get("ab_final_status")),
            ("downstream_ab_final_status_present", state.get("ab_final_status_present")),
            ("candidate_chain", state.get("ab_candidate_chain")),
            ("reference_reproduction_status", state.get("ab_reference_reproduction_status")),
            ("candidate_audit_prerequisites_met",
             state.get("ab_candidate_audit_prerequisites_met")),
            ("production_approval_required", PRODUCTION_APPROVAL_REQUIRED),
            ("counterfactual_final_status", state.get("counterfactual_final_status")),
            ("prerequisites_met", state.get("prerequisites_met")),
        ))),
        ("output_root", str(layout["root"])),
        ("output_layout", OrderedDict((k, str(v)) for k, v in layout.items())),
        ("decomposition_formulas", decomposition_formulas()),
        ("anomaly_reconstruction_checks", anomaly_reconstruction_check_declaration()),
        ("boundary_classes", OrderedDict((
            ("raw_flags", list(BOUNDARY_FLAGS)),
            ("stratified_classes", list(STRATIFIED_CLASSES)),
            ("excess_jump_boundaries", list(EXCESS_JUMP_BOUNDARIES)),
        ))),
        ("thresholds", OrderedDict((
            ("step5", step5_thresholds()),
            ("near_std_threshold_epsilon_primary", STD_THRESHOLD_EPSILON_PRIMARY),
            ("near_std_threshold_epsilon_sensitivity",
             list(STD_THRESHOLD_EPSILON_SENSITIVITY)),
            ("hotspot_percentiles_descriptive_only", list(HOTSPOT_PERCENTILES)),
            ("dominance_share_lower_bound", DOMINANCE_SHARE_LOWER_BOUND),
        ))),
        ("bootstrap_configuration", build_config_snapshot(experiment_id)["bootstrap"]),
        ("matched_control_strategy", build_config_snapshot(experiment_id)["matched_controls"]),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("allowed_final_statuses", list(FINAL_STATUSES)),
        ("planned_stages", list(PLANNED_STAGES)),
        ("expected_files", OrderedDict((k, str(v)) for k, v in expected.items())),
        ("writes_performed", False),
        ("directories_created", 0),
        ("earth_engine_calls", 0),
        ("rasters_modified", 0),
        ("smoothing_applied", False),
    ))


def assert_supported_experiment(experiment_id: str) -> None:
    if experiment_id not in SUPPORTED_EXPERIMENT_IDS:
        raise ResidualSeamError(
            f"unsupported --experiment {experiment_id!r}. This task supports only "
            f"{list(SUPPORTED_EXPERIMENT_IDS)}; another AOI needs its own frozen "
            "inputs and its own predeclaration."
        )


# =============================================================================
# Reporting
# =============================================================================
def build_summary(
    experiment_id: str, *, config: dict, provenance: dict, state: dict,
    detection: dict, cmb: dict, anomaly: dict, mask_analysis: dict,
    excess: dict, pathrow: dict, bootstrap_summary: dict, hotspots: dict,
    decision: dict, resources: dict,
) -> dict:
    """Assemble `residual_seam_summary.json`."""
    return OrderedDict((
        ("audit", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("chain_under_attribution", CANDIDATE_CHAIN),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("final_status", decision["final_status"]),
        ("final_status_meaning", decision["meaning"]),
        ("seam_fixed", False),
        ("production_approved", False),
        ("changes_production_reducer", False),
        ("smoothing_applied", False),
        ("decision", decision),
        ("configuration", config),
        ("technical_validity", OrderedDict((
            ("grid_contract_passed", provenance["grid_contract"]["passed"]),
            ("downstream_ab_final_status", state.get("ab_final_status")),
            ("downstream_ab_reference_reproduction",
             state.get("ab_reference_reproduction_status")),
            ("downstream_ab_baseline_invariance",
             state.get("ab_baseline_invariance_status")),
            ("candidate_audit_prerequisites_met",
             state.get("ab_candidate_audit_prerequisites_met")),
            ("production_approval_required", PRODUCTION_APPROVAL_REQUIRED),
            ("missing_required_inputs", provenance["missing_required_inputs"]),
            ("missing_optional_inputs", provenance["missing_optional_inputs"]),
            ("pathrow_evidence_availability", pathrow.get("availability")),
            ("earth_engine_used", False),
            ("step5_to_step8_rerun", False),
        ))),
        ("residual_seam_detection", detection),
        ("current_minus_baseline_decomposition", cmb),
        ("anomaly_decomposition", anomaly),
        ("mask_and_threshold_effects", mask_analysis),
        ("support_boundary_effects", excess),
        ("pathrow_evidence", pathrow),
        ("hotspot_mechanism_overlap", hotspots),
        ("bootstrap", bootstrap_summary),
        ("resources", resources),
        ("limitations", required_limitations()),
        ("inherited_limitations", inherited_limitations(state)),
        ("next_experiment", next_experiment_text(decision["final_status"])),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def next_experiment_text(final_status: str) -> str:
    if final_status == STATUS_CURRENT_SUPPORT:
        return (
            "Design a current-period support experiment: hold the baseline fixed and "
            "vary only the current-period compositing support (for example a minimum "
            "unique-date requirement), then re-measure the same boundary jumps. Do "
            "not change the production reducer on this single-AOI evidence."
        )
    if final_status == STATUS_BASELINE_SUPPORT:
        return (
            "Design a baseline-support experiment: extend or rebalance the baseline "
            "climatology and re-measure the baseline-mean component at the same "
            "boundaries. Four baseline years is the binding constraint to test first."
        )
    if final_status == STATUS_BASELINE_VARIANCE:
        return (
            "Design a baseline-variance experiment: vary STEP5_MIN_BASELINE_STD_CELSIUS "
            "and the variance estimator over a predeclared grid and re-measure the "
            "denominator contribution and the mask-discontinuity rate. Treat the "
            "current 1.0 Celsius guard as the hypothesis under test, not as fixed."
        )
    if final_status == STATUS_PATHROW:
        return (
            "Obtain pixel-level selected-scene provenance before acting. The present "
            "evidence is metadata-derived and cannot say which scene supplied a pixel; "
            "a per-pixel scene index would turn this association into a testable claim."
        )
    if final_status == STATUS_MIXED:
        return (
            "Run the mechanism experiments separately rather than jointly: a combined "
            "intervention would confound the mechanisms this audit has just separated."
        )
    if final_status == STATUS_RESIDUAL_NOT_DETECTED:
        return (
            "The residual structure is not localised by any boundary definition tested "
            "here. Widen the boundary vocabulary (for example atmospheric correction "
            "or emissivity strata) before designing a compositing intervention."
        )
    if final_status == STATUS_INVALID_INPUTS:
        return (
            "Repair the inputs before drawing any attribution conclusion: the technical "
            "preconditions of this audit did not hold."
        )
    return (
        "No mechanism reached its predeclared dominance rule. Increase the independent "
        "unit count (a second AOI or a wider window) before re-running the same "
        "predeclared analysis; do not relax the rules on this evidence."
    )


def render_summary_markdown(summary: dict) -> str:
    """`residual_seam_summary.md` with the ten required sections."""
    lines: list[str] = []
    add = lines.append

    add(f"# Residual seam attribution -- {summary['experiment_id']}")
    add("")
    add(f"- Chain under attribution: `{summary['chain_under_attribution']}`")
    add(f"- Target products: `{', '.join(TARGET_PRODUCTS)}`")
    add(f"- Report schema: `{summary['report_schema_version']}`")
    add(f"- Decision rule: `{summary['decision_rule_version']}`")
    add(f"- **Final status: `{summary['final_status']}`**")
    add("")
    add(f"> {summary['final_status_meaning']}")
    add("")
    add("This is a diagnostic attribution audit. It does NOT change the production "
        "reducer, it does NOT smooth or otherwise correct any raster, and it can "
        "never report that the seam is fixed or that anything is approved.")
    add("")

    add("## 1. Technical validity")
    add("")
    for key, value in summary["technical_validity"].items():
        add(f"- `{key}`: `{value}`")
    add("")

    add("## 2. Residual-seam detection")
    add("")
    detection = summary["residual_seam_detection"]
    add("| product | pairs | mean abs jump | p95 abs jump | p99 abs jump | max abs jump |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for product, row in (detection.get("per_product") or {}).items():
        add("| {p} | {n} | {m} | {p95} | {p99} | {mx} |".format(
            p=product, n=row.get("n_pairs"), m=_fmt(row.get("mean_abs_jump")),
            p95=_fmt(row.get("p95_abs_jump")), p99=_fmt(row.get("p99_abs_jump")),
            mx=_fmt(row.get("max_abs_jump")),
        ))
    add("")
    add("Percentiles come from fixed-edge histograms and are DESCRIPTIVE. Primary "
        "inference uses the continuous pairwise jumps below, never a hotspot cut.")
    add("")

    add("## 3. Current-minus-baseline decomposition")
    add("")
    cmb = summary["current_minus_baseline_decomposition"]
    add("Exact identity: `delta(D) = delta(C) - delta(M)`. "
        f"Reconstruction tolerance `{CMB_RECONSTRUCTION_ABS_TOL}`; observed maximum "
        f"residual `{_fmt(cmb.get('max_reconstruction_residual'))}`; "
        f"exact reconstruction `{cmb.get('reconstruction_exact')}`.")
    add("")
    add("| class | pairs | blocks | mean signed | mean abs | mean abs current | "
        "mean abs baseline | current share [95%] | baseline share [95%] | cancel | reinforce |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |")
    for row in cmb.get("by_class") or []:
        add("| {c} | {n} | {u} | {ms} | {ma} | {mc} | {mb} | {cs} | {bs} | {cf} | {rf} |".format(
            c=row.get("boundary_class"), n=row.get("n_pairs"), u=row.get("n_units"),
            ms=_fmt(row.get("mean_signed_target_jump")),
            ma=_fmt(row.get("mean_abs_target_jump")),
            mc=_fmt(row.get("mean_abs_current_component")),
            mb=_fmt(row.get("mean_abs_baseline_component")),
            cs=_interval(row.get("current_share")), bs=_interval(row.get("baseline_share")),
            cf=_fmt(row.get("cancellation_fraction")),
            rf=_fmt(row.get("reinforcement_fraction")),
        ))
    add("")
    add("Dominance is read from the SHARE intervals, never from a signed mean.")
    add("")

    add("## 4. Anomaly numerator/denominator decomposition")
    add("")
    anomaly = summary["anomaly_decomposition"]
    add("Exact symmetric decomposition (no Taylor expansion): "
        "`Z_B - Z_A = 0.5*(1/S_A + 1/S_B)*(D_B - D_A) + 0.5*(D_A + D_B)*(1/S_B - 1/S_A)`.")
    add("")
    lines.extend(_render_anomaly_checks(anomaly))
    add("| class | pairs | blocks | mean abs Z jump | mean abs numerator | "
        "mean abs denominator | numerator share [95%] | denominator share [95%] | "
        "cancel | reinforce |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |")
    for row in anomaly.get("by_class") or []:
        add("| {c} | {n} | {u} | {mz} | {mn} | {md} | {ns} | {ds} | {cf} | {rf} |".format(
            c=row.get("boundary_class"), n=row.get("n_pairs"), u=row.get("n_units"),
            mz=_fmt(row.get("mean_abs_anomaly_jump")),
            mn=_fmt(row.get("mean_abs_numerator_contribution")),
            md=_fmt(row.get("mean_abs_denominator_contribution")),
            ns=_interval(row.get("numerator_share")),
            ds=_interval(row.get("denominator_share")),
            cf=_fmt(row.get("cancellation_fraction")),
            rf=_fmt(row.get("reinforcement_fraction")),
        ))
    add("")
    std = anomaly.get("baseline_std_distribution") or {}
    add(f"Baseline std on valid anomaly pairs: min `{_fmt(std.get('min'))}`, "
        f"p05 `{_fmt(std.get('p05'))}`, median `{_fmt(std.get('median'))}`, "
        f"max `{_fmt(std.get('max'))}`. Step5 masks S below "
        f"`{step5_thresholds()['min_baseline_std_celsius']}` Celsius, so 1/S <= 1 "
        "wherever the anomaly is valid.")
    add("")

    add("## 5. Mask and threshold effects")
    add("")
    mask = summary["mask_and_threshold_effects"]
    add("A pair valid on only one side is a MASK DISCONTINUITY EVENT. It is never "
        "assigned a numerical anomaly jump.")
    add("")
    add("| stratum | both valid | A only | B only | neither | discontinuity rate |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in mask.get("by_stratum") or []:
        add("| {s} | {bv} | {a} | {b} | {n} | {r} |".format(
            s=row.get("stratum"), bv=row.get("both_valid"), a=row.get("a_only_valid"),
            b=row.get("b_only_valid"), n=row.get("neither_valid"),
            r=_fmt(row.get("mask_discontinuity_rate")),
        ))
    add("")
    add("### Near-std-threshold epsilon sensitivity")
    add("")
    add(f"Primary epsilon `{STD_THRESHOLD_EPSILON_PRIMARY}` Celsius, predeclared "
        "before any result was inspected. All epsilons are reported; none is "
        "selected post hoc.")
    add("")
    add("| epsilon | boundary pairs | excess abs jump | interval | verdict |")
    add("| ---: | ---: | ---: | --- | --- |")
    for row in mask.get("epsilon_sensitivity") or []:
        add("| {e} | {n} | {x} | {i} | {v} |".format(
            e=row.get("epsilon"), n=row.get("n_boundary_pairs"),
            x=_fmt(row.get("excess_absolute_jump")), i=_interval(row),
            v=row.get("verdict"),
        ))
    add("")

    add("## 6. Support-boundary effects")
    add("")
    add("Excess absolute jump = boundary mean |jump| minus matched within-block "
        "control mean |jump|, with a cluster bootstrap over spatial blocks.")
    add("")
    add("| product | boundary | pairs | controls | blocks | excess | interval | verdict |")
    add("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in summary["support_boundary_effects"].get("rows") or []:
        add("| {p} | {b} | {n} | {c} | {u} | {x} | {i} | {v} |".format(
            p=row.get("product"), b=row.get("boundary"),
            n=row.get("n_boundary_pairs"), c=row.get("n_control_pairs"),
            u=row.get("n_units"), x=_fmt(row.get("excess_absolute_jump")),
            i=_interval(row), v=row.get("verdict"),
        ))
    add("")

    add("## 7. Path/row evidence")
    add("")
    pathrow = summary["pathrow_evidence"]
    add(f"Availability: `{pathrow.get('availability')}` -- {pathrow.get('reason')}")
    add("")
    add("Path/row evidence is METADATA-DERIVED from frozen source-scene footprints. "
        "It is NOT pixel-level selected-scene provenance.")
    add("")
    if pathrow.get("availability") == "available":
        add("| stratum | pairs | blocks | interfaces | excess | interval | verdict |")
        add("| --- | ---: | ---: | ---: | ---: | --- | --- |")
        for row in pathrow.get("stratified_rows") or []:
            add("| {s} | {n} | {u} | {i} | {x} | {iv} | {v} |".format(
                s=row.get("stratum"), n=row.get("n_boundary_pairs"),
                u=row.get("n_units"), i=row.get("n_interfaces"),
                x=_fmt(row.get("excess_absolute_jump")), iv=_interval(row),
                v=row.get("verdict"),
            ))
        add("")
        add(f"Path/row-only verdict: `{pathrow.get('verdict')}`. "
            f"Supported: `{pathrow.get('supported')}`.")
    else:
        add("The path/row mechanism is reported as UNAVAILABLE. No provenance was "
            "invented and no positive path/row evidence can arise from this run.")
    add("")

    add("## 8. Primary attribution")
    add("")
    add(f"**`{summary['final_status']}`**")
    add("")
    for reason in summary["decision"].get("reasons") or []:
        add(f"- {reason}")
    add("")
    secondary = summary["decision"].get("secondary_supported_mechanisms") or []
    add(f"Supported secondary mechanisms: `{secondary}`" if secondary
        else "No secondary mechanism reached supported evidence.")
    add("")
    add("This audit can never conclude that the seam is fixed and never issues a "
        "production decision.")
    add("")

    add("## 9. Limitations")
    add("")
    for item in summary["limitations"]:
        add(f"- {item}")
    for item in summary.get("inherited_limitations") or []:
        add(f"- {item}")
    add("")

    add("## 10. Next experiment")
    add("")
    add(summary["next_experiment"])
    add("")
    return "\n".join(lines)


def _render_anomaly_checks(anomaly: dict) -> list[str]:
    """The two anomaly reconstruction checks, reported SEPARATELY.

    They answer different questions and only the first can invalidate the audit,
    so they never share a row, a tolerance or a verdict.
    """
    lines: list[str] = ["### 4a. Algebraic identity check (gates the audit)", ""]
    identity = anomaly.get("algebraic_identity_check") or {}
    lines.append(
        "Z_A and Z_B are recomputed in float64 as `D/S`; the decomposition must "
        "reproduce `Z_B - Z_A` exactly up to float64 round-off."
    )
    lines.append("")
    lines.append(f"- tolerance policy: `{identity.get('tolerance_policy')}`")
    lines.append(f"- absolute tolerance: `{identity.get('tolerance_absolute')}`, "
                 f"relative tolerance: `{identity.get('tolerance_relative')}`")
    lines.append(f"- pairs checked: `{identity.get('n_pairs_checked')}`, "
                 f"pairs exceeding tolerance: `{identity.get('n_pairs_exceeding_tolerance')}`")
    lines.append(f"- max absolute residual: `{_fmt(identity.get('max_absolute_residual'))}`, "
                 f"max residual / tolerance: "
                 f"`{_fmt(identity.get('max_residual_over_tolerance'))}`")
    lines.append(f"- **passed: `{identity.get('passed')}`** "
                 f"(gates the audit: `{identity.get('gates_the_audit')}`)")
    lines.append("")

    lines.append("### 4b. Stored-raster reproduction check (descriptive only)")
    lines.append("")
    stored = anomaly.get("stored_raster_reproduction_check") or {}
    lines.append(
        "The recomputed float64 `D/S` is compared against the STORED float32 "
        "`anomaly_zscore` raster. Step5 divided its own internal float32 "
        "difference and serialised the quotient to float32, so a residual of "
        "this size is EXPECTED serialization error."
    )
    lines.append("")
    lines.append(f"- predeclared tolerance: `{stored.get('predeclared_tolerance')}` "
                 f"(source `{stored.get('predeclared_tolerance_source')}`)")
    lines.append(f"- existing Step5 reproduction policy bound: "
                 f"`{stored.get('step5_reproduction_policy_tolerance')}` "
                 f"(source `{stored.get('step5_reproduction_policy_source')}`)")
    lines.append(f"- pixels checked: `{stored.get('n_pixels_checked')}`")
    lines.append("")
    lines.append("| statistic | absolute error |")
    lines.append("| --- | ---: |")
    for percentile in ANOMALY_STORED_REPRODUCTION_PERCENTILES:
        key = f"p{str(percentile).replace('.', '_')}_abs_error"
        lines.append(f"| p{percentile} | {_fmt(stored.get(key))} |")
    lines.append(f"| max | {_fmt(stored.get('max_abs_error'))} |")
    lines.append("")
    lines.append(
        f"- within predeclared tolerance: "
        f"`{_fmt(stored.get('fraction_within_predeclared_tolerance'))}`; "
        f"within Step5 policy bound: "
        f"`{_fmt(stored.get('fraction_within_policy_tolerance'))}`"
    )
    lines.append(f"- status: `{stored.get('status')}`")
    lines.append(f"- **is a decomposition failure: "
                 f"`{stored.get('is_decomposition_failure')}`** "
                 f"(gates the audit: `{stored.get('gates_the_audit')}`)")
    lines.append("")
    lines.append(
        "Expected float32 serialization error is NEVER treated as a "
        "decomposition failure; only check 4a can invalidate the decomposition."
    )
    lines.append("")
    return lines


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _interval(row) -> str:
    if not isinstance(row, dict):
        return "n/a"
    low, high = row.get("interval_low"), row.get("interval_high")
    if low is None or high is None:
        return "n/a"
    return f"[{_fmt(low)}, {_fmt(high)}]"


#: `maps/` PNG quicklooks and GeoTIFF overlays are hashed like everything else.
MANIFEST_EXCLUDED_SUBTREES = ("_analysis_tmp",)


def manifest_candidate_files(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and not set(p.relative_to(root).parts) & set(MANIFEST_EXCLUDED_SUBTREES)
    )


def build_manifest(experiment_id: str, root: Path, summary: dict) -> dict:
    """`residual_seam_manifest.json`: every produced file with size + sha256."""
    root = Path(root)
    files = audit.build_file_manifest(
        manifest_candidate_files(root), output_dir=root,
    )["files"]
    return OrderedDict((
        ("audit", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("output_root", str(root)),
        ("chain_under_attribution", CANDIDATE_CHAIN),
        ("final_status", summary["final_status"]),
        ("seam_fixed", False),
        ("production_approved", False),
        ("changes_production_reducer", False),
        ("smoothing_applied", False),
        ("file_count", len(files)),
        ("files", files),
        ("excluded_subtrees", list(MANIFEST_EXCLUDED_SUBTREES)),
        ("frozen_inputs_hashed_in", "input_provenance.json"),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def report_generation_preserves_metrics(before: dict, after: dict) -> bool:
    """Report generation must never alter a scientific number."""
    return json.dumps(before, sort_keys=True, default=str) == json.dumps(
        after, sort_keys=True, default=str
    )


def _scrub_declared_prohibitions(payload):
    """Drop the audit's own `forbidden_conclusions` declaration before scanning.

    The config snapshot lists the banned phrases on purpose; matching against
    that list would make the guard fire on its own predeclaration.
    """
    if isinstance(payload, dict):
        return {
            key: _scrub_declared_prohibitions(value)
            for key, value in payload.items()
            if key != "forbidden_conclusions"
        }
    if isinstance(payload, (list, tuple)):
        return [_scrub_declared_prohibitions(item) for item in payload]
    return payload


def summary_forbids_banned_conclusions(payload) -> bool:
    """No report may contain a 'seam fixed' or production-approval claim."""
    if isinstance(payload, str):
        text = payload.lower()
    else:
        text = json.dumps(_scrub_declared_prohibitions(payload), default=str).lower()
    # `"seam_fixed": false` and `"production_approved": false` are explicit
    # NEGATIVE assertions and are the only permitted occurrences.
    for token in ("seam_fixed", "production_approved"):
        text = text.replace(f'"{token}": false', "").replace(f"{token}: false", "")
        text = text.replace(f"`{token}`", "")
    return not any(bad in text for bad in ("seam fixed", "production approved",
                                           "production_ready", "approved_for_production"))


# =============================================================================
# Stratum-matched accumulation (numpy-backed; bounded by POPULATED cells only)
# =============================================================================
class StratumAccumulator:
    """Sum/count per (spatial block, matching stratum) cell.

    Backed by parallel numpy arrays rather than a Python dict, so memory scales
    with the number of cells that actually contain pairs (~24 bytes each) rather
    than with the full block x stratum product.
    """

    __slots__ = ("keys", "sums", "counts")

    def __init__(self) -> None:
        import numpy as np

        self.keys = np.empty(0, dtype="int64")
        self.sums = np.empty(0, dtype="float64")
        self.counts = np.empty(0, dtype="int64")

    def add(self, keys, values) -> None:
        import numpy as np

        keys = np.asarray(keys, dtype="int64")
        values = np.asarray(values, dtype="float64")
        finite = np.isfinite(values)
        if not finite.any():
            return
        keys, values = keys[finite], values[finite]

        merged_keys = np.concatenate((self.keys, keys))
        merged_sums = np.concatenate((self.sums, values))
        merged_counts = np.concatenate((self.counts, np.ones(values.size, dtype="int64")))

        unique, inverse = np.unique(merged_keys, return_inverse=True)
        self.keys = unique
        self.sums = np.bincount(inverse, weights=merged_sums, minlength=unique.size)
        self.counts = np.bincount(
            inverse, weights=merged_counts, minlength=unique.size,
        ).astype("int64")

    @property
    def n_cells(self) -> int:
        return int(self.keys.size)

    @property
    def n_pairs(self) -> int:
        return int(self.counts.sum())


def _block_from_cell_key(cell_keys, stratum_space: int):
    """Recover the spatial-block id from a packed (block, stratum) cell key."""
    import numpy as np

    return np.asarray(cell_keys, dtype="int64") // int(stratum_space)


def matched_block_accumulators(
    boundary: StratumAccumulator, control: StratumAccumulator, *,
    stratum_space: int,
) -> tuple[MeanAccumulator, MeanAccumulator, dict]:
    """Direct-standardised matched contrast, aggregated to block level.

    For every (block, stratum) cell that contains BOTH boundary and control
    pairs, the control cell mean is re-weighted by the BOUNDARY pair count. Both
    arms therefore end up with the same per-block count and the difference is a
    genuine matched contrast rather than two differently-composed populations.
    Cells with no control counterpart are dropped and reported.
    """
    import numpy as np

    boundary_acc = MeanAccumulator()
    control_acc = MeanAccumulator()
    diagnostics = OrderedDict((
        ("boundary_cells", boundary.n_cells),
        ("control_cells", control.n_cells),
        ("matched_cells", 0),
        ("unmatched_boundary_cells", 0),
        ("boundary_pairs_total", boundary.n_pairs),
        ("boundary_pairs_matched", 0),
        ("boundary_pairs_dropped_unmatched", 0),
    ))
    if boundary.n_cells == 0 or control.n_cells == 0:
        diagnostics["unmatched_boundary_cells"] = boundary.n_cells
        diagnostics["boundary_pairs_dropped_unmatched"] = boundary.n_pairs
        return boundary_acc, control_acc, diagnostics

    position = np.searchsorted(control.keys, boundary.keys)
    position = np.clip(position, 0, control.keys.size - 1)
    matched = control.keys[position] == boundary.keys

    diagnostics["matched_cells"] = int(matched.sum())
    diagnostics["unmatched_boundary_cells"] = int((~matched).sum())
    diagnostics["boundary_pairs_matched"] = int(boundary.counts[matched].sum())
    diagnostics["boundary_pairs_dropped_unmatched"] = int(boundary.counts[~matched].sum())
    if not matched.any():
        return boundary_acc, control_acc, diagnostics

    cell_keys = boundary.keys[matched]
    b_sums = boundary.sums[matched]
    b_counts = boundary.counts[matched]
    c_sums = control.sums[position[matched]]
    c_counts = control.counts[position[matched]]

    control_cell_means = c_sums / c_counts
    standardised_control_sums = b_counts * control_cell_means

    blocks = _block_from_cell_key(cell_keys, stratum_space)
    for block, b_sum, b_count, c_sum in zip(
        blocks, b_sums, b_counts, standardised_control_sums,
    ):
        block = int(block)
        boundary_acc.sums[block] = boundary_acc.sums.get(block, 0.0) + float(b_sum)
        boundary_acc.counts[block] = boundary_acc.counts.get(block, 0) + int(b_count)
        control_acc.sums[block] = control_acc.sums.get(block, 0.0) + float(c_sum)
        control_acc.counts[block] = control_acc.counts.get(block, 0) + int(b_count)
    return boundary_acc, control_acc, diagnostics


# =============================================================================
# Windowed reading
# =============================================================================
def iter_row_windows(height: int, window_rows: int = WINDOW_ROWS):
    """Yield `(read_start, read_stop, horizontal_rows, vertical_rows)`.

    Each window reads ONE halo row past its own last row so vertical adjacency
    pairs that straddle a window edge are still built -- once, and only once.
    `horizontal_rows` is the number of leading local rows that emit horizontal
    edges; `vertical_rows` is the number of leading local vertical edges.
    """
    height = int(height)
    step = int(window_rows)
    for start in range(0, height, step):
        stop = min(start + step, height)
        read_stop = min(stop + 1, height)
        n_local = read_stop - start
        horizontal_rows = stop - start
        # Vertical anchors within this window: local rows 0 .. n_local-2, but
        # never beyond this window's own last row.
        vertical_rows = max(0, min(n_local - 1, stop - start))
        yield start, read_stop, horizontal_rows, vertical_rows


def read_window(path: Path, start: int, stop: int, *, band: int = 1):
    """Read one row window as float64 with nodata/sentinel mapped to NaN.

    Masked, sentinel and AOI-exterior pixels become NaN. Zero is NEVER
    substituted for missing data anywhere in this module.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        window = Window(0, int(start), int(src.width), int(stop) - int(start))
        arr = src.read(band, window=window, masked=True).astype("float64")
    filled = arr.filled(np.nan)
    return np.where(filled == audit.NODATA_SENTINEL, np.nan, filled)


def _slice_h(array, rows: int):
    """Horizontal-edge sub-array: leading `rows` rows only (drop the halo)."""
    return array[:rows, :]


# =============================================================================
# Boundary flag construction
# =============================================================================
def build_edge_flags(window: dict, orientation: str, *, epsilon: float,
                     thresholds: dict, pathrow_masks: dict | None) -> dict:
    """Every raw mechanism flag for one orientation of one window.

    Flags are NON-EXCLUSIVE by design: a pair that carries a support change AND
    a path/row boundary keeps both. Nothing is forced into a single mechanism.
    """
    import numpy as np

    flags: "OrderedDict[str, object]" = OrderedDict()

    flags["current_unique_date_count_change"] = edge_change_flag(
        window["current_unique_date_valid_count"], orientation,
    )
    flags["current_scene_count_change"] = edge_change_flag(
        window["current_scene_valid_count"], orientation,
    )
    flags["current_valid_count_change"] = edge_change_flag(
        window["current_period_valid_count"], orientation,
    )
    flags["current_support_change"] = (
        flags["current_unique_date_count_change"]
        | flags["current_scene_count_change"]
        | flags["current_valid_count_change"]
    )
    flags["baseline_valid_year_change"] = edge_change_flag(
        window["baseline_valid_count"], orientation,
    )

    annual = None
    for key, array in window.items():
        if key.startswith("baseline_") and key.endswith("_unique_date_valid_count"):
            change = edge_change_flag(array, orientation)
            annual = change if annual is None else (annual | change)
    if annual is None:
        shape = _edge_pairs(window["baseline_valid_count"], orientation)[0].shape
        annual = np.zeros(shape, dtype=bool)
    flags["baseline_annual_date_support_change"] = annual

    flags["same_day_multiplicity_change"] = edge_change_flag(
        window["current_same_day_multiplicity"], orientation,
    )

    flags["low_baseline_std_boundary"] = edge_change_flag(
        window["low_baseline_std_mask"], orientation,
    )
    flags["near_std_threshold_boundary"] = edge_near_value(
        window["baseline_lst_std_celsius"], orientation,
        thresholds["min_baseline_std_celsius"], epsilon,
    )
    flags["current_count_threshold_boundary"] = (
        edge_change_flag(window["low_current_count_mask"], orientation)
        | edge_threshold_straddle(
            window["current_period_valid_count"], orientation,
            thresholds["min_current_valid_count"],
        )
    )
    flags["baseline_count_threshold_boundary"] = (
        edge_change_flag(window["low_baseline_count_mask"], orientation)
        | edge_threshold_straddle(
            window["baseline_valid_count"], orientation,
            thresholds["min_baseline_valid_count"],
        )
    )

    shape = flags["current_support_change"].shape
    if pathrow_masks and pathrow_masks.get("union") is not None:
        flags["source_path_row_boundary"] = edge_mask_from_pixel_mask(
            pathrow_masks["union"], orientation,
        )
    else:
        flags["source_path_row_boundary"] = np.zeros(shape, dtype=bool)
    return flags


def stratified_class_codes(flags: dict):
    """Assign every edge to exactly one stratified class CODE.

    The five required classes are always produced. Overlaps that none of them
    describe get their own label (`pathrow_and_threshold`) instead of being
    forced into one of the five -- the raw flags stay available for every pair.
    `pathrow_only` strictly excludes BOTH support and threshold pairs, which is
    what makes the path/row test independent of support overlap.
    """
    import numpy as np

    support = np.zeros(flags["current_support_change"].shape, dtype=bool)
    for name in SUPPORT_UNION_FLAGS:
        support |= flags[name]
    threshold = np.zeros_like(support)
    for name in THRESHOLD_FLAGS:
        threshold |= flags[name]
    pathrow = flags["source_path_row_boundary"]

    codes = np.full(support.shape, OVERLAP_CODES[CLASS_NONE], dtype="int8")
    codes[threshold & ~support & ~pathrow] = OVERLAP_CODES[CLASS_THRESHOLD_ONLY]
    codes[pathrow & threshold & ~support] = OVERLAP_CODES[CLASS_PATHROW_AND_THRESHOLD]
    codes[pathrow & ~support & ~threshold] = OVERLAP_CODES[CLASS_PATHROW_ONLY]
    codes[support & ~pathrow] = OVERLAP_CODES[CLASS_SUPPORT_ONLY]
    codes[support & pathrow] = OVERLAP_CODES[CLASS_SUPPORT_AND_PATHROW]
    return codes, support, threshold, pathrow


def control_pair_mask(support, threshold, pathrow):
    """Non-boundary control pairs: NONE of the known mechanisms is present."""
    return ~support & ~threshold & ~pathrow


def stratum_keys(block_ids, orientation_index: int, elevation_bin, slope_bin, ndvi_bin):
    """Pack (block, orientation, elevation, slope, NDVI) into one int64 cell key.

    The packing is injective for the predeclared bin counts, so two pairs share a
    key exactly when they share a spatial block AND a matching stratum.
    """
    import numpy as np

    e = np.asarray(elevation_bin, dtype="int64") + 1     # -1 (unmatched) -> 0
    s = np.asarray(slope_bin, dtype="int64") + 1
    n = np.asarray(ndvi_bin, dtype="int64") + 1
    stratum = (
        int(orientation_index) * ELEVATION_STRATA * SLOPE_STRATA * NDVI_STRATA
        + e * SLOPE_STRATA * NDVI_STRATA
        + s * NDVI_STRATA
        + n
    )
    return np.asarray(block_ids, dtype="int64") * STRATUM_SPACE + stratum


#: Bin cardinalities: len(edges) + 1 possible digitize outputs, plus one slot
#: for the 'unmatched / covariate missing' bin.
ELEVATION_STRATA = len(ELEVATION_GRADIENT_BINS) + 2
SLOPE_STRATA = len(SLOPE_GRADIENT_BINS) + 2
NDVI_STRATA = len(NDVI_GRADIENT_BINS) + 2
ORIENTATION_STRATA = len(ORIENTATIONS)
STRATUM_SPACE = ORIENTATION_STRATA * ELEVATION_STRATA * SLOPE_STRATA * NDVI_STRATA


# =============================================================================
# Streaming analysis state
# =============================================================================
CMB_STATISTICS = (
    "signed_target_jump", "abs_target_jump",
    "abs_current_component", "abs_baseline_component",
    "current_share", "baseline_share",
    "cancellation", "reinforcement", "degenerate",
)
ANOMALY_STATISTICS = (
    "signed_anomaly_jump", "abs_anomaly_jump",
    "abs_numerator_contribution", "abs_denominator_contribution",
    "numerator_share", "denominator_share",
    "cancellation", "reinforcement", "degenerate",
    "min_baseline_std", "max_inverse_std", "min_distance_to_std_threshold",
)

MASK_STRATA = (
    "all_pairs",
    "low_baseline_std_boundary",
    "near_std_threshold_boundary",
    "baseline_count_threshold_boundary",
    "current_count_threshold_boundary",
    "current_support_change",
    "baseline_valid_year_change",
    "source_path_row_boundary",
    CLASS_PATHROW_ONLY,
    CLASS_SUPPORT_ONLY,
    CLASS_NONE,
)


class AnalysisState:
    """All bounded-memory accumulators for one streaming pass.

    Nothing here stores a per-pair record: every statistic is kept as per-unit
    sufficient statistics (sum/count), a fixed-edge histogram, or a fixed-size
    reservoir sample. The full pair set is never materialised.
    """

    def __init__(self) -> None:
        self.cmb: dict[tuple[str, str], MeanAccumulator] = {}
        self.anomaly: dict[tuple[str, str], MeanAccumulator] = {}
        self.boundary_strata: dict[tuple[str, str], StratumAccumulator] = {}
        self.control_strata: dict[str, StratumAccumulator] = {}
        self.histograms: dict[str, HistogramAccumulator] = {
            product: HistogramAccumulator(HISTOGRAM_MAX[product])
            for product in TARGET_PRODUCTS
        }
        self.std_histogram = HistogramAccumulator(20.0)
        self.sample = ReservoirSampler()
        self.mask_counts: dict[str, dict[str, int]] = {}
        self.epsilon_strata: dict[float, StratumAccumulator] = {}
        self.pathrow_only_pairs: dict[str, int] = {}
        self.pathrow_only_blocks: dict[str, set] = {}
        self.pair_counts: dict[str, int] = {}
        self.dropped: dict[str, int] = {}
        self.max_residual: dict[str, float] = {TARGET_CMB: 0.0, TARGET_ANOMALY: 0.0}
        # --- check 1: algebraic identity (gating) -------------------------
        self.identity_max_tolerance_ratio = 0.0
        self.identity_pairs_checked = 0
        self.identity_pairs_exceeding = 0
        # --- check 2: stored-raster reproduction (descriptive) ------------
        self.stored_reproduction = HistogramAccumulator(
            STORED_REPRODUCTION_HISTOGRAM_MAX, STORED_REPRODUCTION_HISTOGRAM_BINS,
        )
        self.stored_reproduction_max = 0.0
        self.stored_reproduction_pixels = 0
        self.stored_reproduction_within_predeclared = 0
        self.stored_reproduction_within_policy = 0
        self.windows_processed = 0
        self.resource_log: list[dict] = []

    # -- accessors ---------------------------------------------------------
    def cmb_acc(self, boundary_class: str, statistic: str) -> MeanAccumulator:
        return self.cmb.setdefault((boundary_class, statistic), MeanAccumulator())

    def anomaly_acc(self, boundary_class: str, statistic: str) -> MeanAccumulator:
        return self.anomaly.setdefault((boundary_class, statistic), MeanAccumulator())

    def boundary_stratum(self, product: str, boundary: str) -> StratumAccumulator:
        return self.boundary_strata.setdefault((product, boundary), StratumAccumulator())

    def control_stratum(self, product: str) -> StratumAccumulator:
        return self.control_strata.setdefault(product, StratumAccumulator())

    def epsilon_stratum(self, epsilon: float) -> StratumAccumulator:
        return self.epsilon_strata.setdefault(float(epsilon), StratumAccumulator())

    def bump(self, counter: dict, key: str, amount: int) -> None:
        counter[key] = counter.get(key, 0) + int(amount)

    def mask_bump(self, stratum: str, key: str, amount: int) -> None:
        bucket = self.mask_counts.setdefault(stratum, {
            "both_valid": 0, "a_only_valid": 0, "b_only_valid": 0, "neither_valid": 0,
        })
        bucket[key] = bucket.get(key, 0) + int(amount)


def _accumulate_by_class(accessor, class_codes, unit_ids, statistics: dict) -> None:
    """Feed one window's statistics into the per-class accumulators.

    `all_pairs` receives every retained pair; each stratified class receives its
    own subset. A pair therefore contributes to exactly one stratified class and
    to the pooled total, never twice within the same stratum.
    """
    import numpy as np

    for name, values in statistics.items():
        accessor("all_pairs", name).add(unit_ids, values)
    for label, code in OVERLAP_CODES.items():
        if label == "no_pair":
            continue
        selected = class_codes == code
        if not selected.any():
            continue
        units = unit_ids[selected]
        for name, values in statistics.items():
            accessor(label, name).add(units, np.asarray(values)[selected])


def analyse_window(
    state: AnalysisState, window: dict, orientation: str, orientation_index: int, *,
    thresholds: dict, pathrow_masks: dict | None, row_offset: int, sample_stride: int,
) -> None:
    """Process ONE orientation of ONE window end to end.

    Everything is computed on the adjacency lattice of this window; nothing is
    written and no array outlives the call.
    """
    import numpy as np

    flags = build_edge_flags(
        window, orientation, epsilon=STD_THRESHOLD_EPSILON_PRIMARY,
        thresholds=thresholds, pathrow_masks=pathrow_masks,
    )
    class_codes, support, threshold, pathrow = stratified_class_codes(flags)
    controls = control_pair_mask(support, threshold, pathrow)

    rows, cols = edge_anchor_rows_cols(
        window["current_lst_celsius"].shape, orientation,
    )
    rows = rows + int(row_offset)
    shape = class_codes.shape

    def flat(array):
        return np.asarray(array).reshape(-1)

    block_ids = spatial_block_ids(rows, cols)

    elevation_bin = gradient_bin(
        flat(edge_difference(window.get("elevation", _nan_like(window)), orientation)),
        ELEVATION_GRADIENT_BINS,
    )
    slope_bin = gradient_bin(
        flat(edge_difference(window.get("slope", _nan_like(window)), orientation)),
        SLOPE_GRADIENT_BINS,
    )
    ndvi_bin = gradient_bin(
        flat(edge_difference(window.get("ndvi_current", _nan_like(window)), orientation)),
        NDVI_GRADIENT_BINS,
    )
    cells = stratum_keys(block_ids, orientation_index, elevation_bin, slope_bin, ndvi_bin)

    # ---- Q1: current minus baseline --------------------------------------
    c_a, c_b = _edge_pairs(window["current_lst_celsius"], orientation)
    m_a, m_b = _edge_pairs(window["baseline_lst_mean_celsius"], orientation)
    d_a, d_b = _edge_pairs(window[TARGET_CMB], orientation)

    cmb_valid = flat(
        np.isfinite(c_a) & np.isfinite(c_b) & np.isfinite(m_a) & np.isfinite(m_b)
        & np.isfinite(d_a) & np.isfinite(d_b)
    )
    state.bump(state.pair_counts, f"{TARGET_CMB}_{orientation}", int(cmb_valid.sum()))
    state.bump(
        state.dropped, f"{TARGET_CMB}_{orientation}_invalid_endpoint",
        int(cmb_valid.size - cmb_valid.sum()),
    )

    if cmb_valid.any():
        target_jump, current_component, baseline_component = decompose_current_minus_baseline(
            flat(c_a)[cmb_valid], flat(c_b)[cmb_valid],
            flat(m_a)[cmb_valid], flat(m_b)[cmb_valid],
        )
        stored_jump = flat(d_b)[cmb_valid] - flat(d_a)[cmb_valid]
        residual = reconstruction_residual(
            stored_jump, current_component, baseline_component,
        )
        finite_residual = residual[np.isfinite(residual)]
        if finite_residual.size:
            state.max_residual[TARGET_CMB] = max(
                state.max_residual[TARGET_CMB], float(np.abs(finite_residual).max()),
            )

        cancelling, reinforcing, degenerate = classify_signed_interaction(
            current_component, baseline_component,
        )
        statistics = {
            "signed_target_jump": stored_jump,
            "abs_target_jump": np.abs(stored_jump),
            "abs_current_component": np.abs(current_component),
            "abs_baseline_component": np.abs(baseline_component),
            "current_share": component_share(current_component, baseline_component),
            "baseline_share": component_share(baseline_component, current_component),
            "cancellation": cancelling.astype("float64"),
            "reinforcement": reinforcing.astype("float64"),
            "degenerate": degenerate.astype("float64"),
        }
        _accumulate_by_class(
            state.cmb_acc, class_codes.reshape(-1)[cmb_valid],
            block_ids[cmb_valid], statistics,
        )
        state.histograms[TARGET_CMB].add(np.abs(stored_jump))
        _accumulate_excess_arms(
            state, TARGET_CMB, np.abs(stored_jump), cells[cmb_valid],
            flags, class_codes, controls, cmb_valid,
        )

    # ---- Q2: anomaly ------------------------------------------------------
    s_a, s_b = _edge_pairs(window["baseline_lst_std_celsius"], orientation)
    z_a_stored, z_b_stored = _edge_pairs(window[TARGET_ANOMALY], orientation)

    z_valid_a = flat(np.isfinite(z_a_stored))
    z_valid_b = flat(np.isfinite(z_b_stored))
    both_valid = z_valid_a & z_valid_b
    a_only = z_valid_a & ~z_valid_b
    b_only = ~z_valid_a & z_valid_b
    neither = ~z_valid_a & ~z_valid_b

    _accumulate_mask_events(
        state, flags, class_codes, both_valid, a_only, b_only, neither,
    )

    numeric = both_valid & flat(
        np.isfinite(d_a) & np.isfinite(d_b) & np.isfinite(s_a) & np.isfinite(s_b)
        & (s_a != 0.0) & (s_b != 0.0)
    )
    state.bump(state.pair_counts, f"{TARGET_ANOMALY}_{orientation}", int(numeric.sum()))
    state.bump(
        state.dropped, f"{TARGET_ANOMALY}_{orientation}_one_sided_mask",
        int(a_only.sum() + b_only.sum()),
    )

    if numeric.any():
        z_a, z_b, numerator, denominator = decompose_anomaly(
            flat(d_a)[numeric], flat(d_b)[numeric],
            flat(s_a)[numeric], flat(s_b)[numeric],
        )
        recomputed_jump = z_b - z_a
        _accumulate_identity_check(
            state, recomputed_jump, numerator, denominator, z_a, z_b,
        )

        cancelling, reinforcing, degenerate = classify_signed_interaction(
            numerator, denominator,
        )
        std_a, std_b = flat(s_a)[numeric], flat(s_b)[numeric]
        min_std = np.minimum(std_a, std_b)
        statistics = {
            "signed_anomaly_jump": recomputed_jump,
            "abs_anomaly_jump": np.abs(recomputed_jump),
            "abs_numerator_contribution": np.abs(numerator),
            "abs_denominator_contribution": np.abs(denominator),
            "numerator_share": component_share(numerator, denominator),
            "denominator_share": component_share(denominator, numerator),
            "cancellation": cancelling.astype("float64"),
            "reinforcement": reinforcing.astype("float64"),
            "degenerate": degenerate.astype("float64"),
            "min_baseline_std": min_std,
            "max_inverse_std": np.maximum(1.0 / std_a, 1.0 / std_b),
            "min_distance_to_std_threshold": np.abs(
                min_std - thresholds["min_baseline_std_celsius"]
            ),
        }
        _accumulate_by_class(
            state.anomaly_acc, class_codes.reshape(-1)[numeric],
            block_ids[numeric], statistics,
        )
        state.histograms[TARGET_ANOMALY].add(np.abs(recomputed_jump))
        state.std_histogram.add(min_std)
        _accumulate_excess_arms(
            state, TARGET_ANOMALY, np.abs(recomputed_jump), cells[numeric],
            flags, class_codes, controls, numeric,
        )
        _accumulate_epsilon_sensitivity(
            state, window, orientation, thresholds, np.abs(recomputed_jump),
            cells[numeric], numeric,
        )
        _accumulate_pathrow_units(
            state, class_codes.reshape(-1)[numeric], block_ids[numeric],
            pathrow_masks, orientation, numeric, shape,
        )

        # Check 2 runs on the PIXEL grid, and only on the horizontal pass whose
        # window is sliced to its own rows -- so every valid pixel is compared
        # exactly once and no window halo is double counted.
        if orientation == "horizontal":
            _accumulate_stored_reproduction(state, window)

        state.sample.offer_batch(_build_sample_rows(
            rows[numeric], cols[numeric], orientation, block_ids[numeric],
            class_codes.reshape(-1)[numeric], flags, numeric, window,
            recomputed_jump, numerator, denominator, std_a, std_b, stride=sample_stride,
        ))


def _accumulate_identity_check(state, recomputed_jump, numerator, denominator, z_a, z_b) -> None:
    """CHECK 1 -- the algebraic identity, in float64, with a scale-aware tolerance.

    `Z_B - Z_A == numerator + denominator` is exact in real arithmetic, so the
    only admissible residual is float64 round-off. Both the raw residual and its
    ratio to the per-pair tolerance are tracked, because a raw maximum alone
    cannot say whether a large-|Z| pair was actually out of tolerance.
    """
    import numpy as np

    residual = np.abs(reconstruction_residual(recomputed_jump, numerator, denominator))
    finite = np.isfinite(residual)
    if not finite.any():
        return

    scale = np.maximum(
        np.maximum(np.abs(z_a), np.abs(z_b)),
        np.maximum(np.abs(numerator), np.abs(denominator)),
    )
    tolerance = anomaly_identity_tolerance(scale)
    ratio = residual[finite] / tolerance[finite]

    state.identity_pairs_checked += int(finite.sum())
    state.identity_pairs_exceeding += int(np.count_nonzero(ratio > 1.0))
    state.max_residual[TARGET_ANOMALY] = max(
        state.max_residual[TARGET_ANOMALY], float(residual[finite].max()),
    )
    state.identity_max_tolerance_ratio = max(
        state.identity_max_tolerance_ratio, float(ratio.max()),
    )


def _accumulate_stored_reproduction(state, window: dict) -> None:
    """CHECK 2 -- recomputed float64 D/S versus the STORED float32 raster.

    Descriptive only. Step5 divided its own internal float32 difference and then
    serialised the quotient to float32, so a small disagreement is expected
    serialization error. This never gates the audit and never contributes a
    failure reason.
    """
    import numpy as np

    difference = window[TARGET_CMB]
    std = window["baseline_lst_std_celsius"]
    stored = window[TARGET_ANOMALY]
    valid = (
        np.isfinite(difference) & np.isfinite(std) & np.isfinite(stored) & (std != 0.0)
    )
    if not valid.any():
        return

    with np.errstate(divide="ignore", invalid="ignore"):
        recomputed = difference[valid] / std[valid]
    error = np.abs(recomputed - stored[valid])
    error = error[np.isfinite(error)]
    if not error.size:
        return

    state.stored_reproduction.add(error)
    state.stored_reproduction_max = max(
        state.stored_reproduction_max, float(error.max()),
    )
    state.stored_reproduction_pixels += int(error.size)
    state.stored_reproduction_within_predeclared += int(
        np.count_nonzero(error <= ANOMALY_STORED_REPRODUCTION_TOL)
    )
    state.stored_reproduction_within_policy += int(
        np.count_nonzero(error <= ANOMALY_STORED_REPRODUCTION_POLICY_TOL)
    )


def _nan_like(window: dict):
    """An all-NaN stand-in for an absent optional covariate raster."""
    import numpy as np

    return np.full_like(window["current_lst_celsius"], np.nan)


def _accumulate_excess_arms(
    state: AnalysisState, product: str, abs_jump, cells, flags, class_codes,
    controls, selection,
) -> None:
    """Boundary and shared-control stratum arms for the matched-control test."""
    import numpy as np

    flat_codes = class_codes.reshape(-1)[selection]
    control_selected = controls.reshape(-1)[selection]
    state.control_stratum(product).add(cells[control_selected], abs_jump[control_selected])

    for boundary in EXCESS_JUMP_BOUNDARIES:
        if boundary in flags:
            chosen = flags[boundary].reshape(-1)[selection]
        elif boundary in OVERLAP_CODES:
            chosen = flat_codes == OVERLAP_CODES[boundary]
        elif boundary == "baseline_support_excluding_current":
            baseline = (
                flags["baseline_valid_year_change"]
                | flags["baseline_annual_date_support_change"]
            ).reshape(-1)[selection]
            chosen = baseline & ~flags["current_support_change"].reshape(-1)[selection]
        else:
            continue
        if not chosen.any():
            continue
        state.boundary_stratum(product, boundary).add(cells[chosen], abs_jump[chosen])


def _accumulate_epsilon_sensitivity(
    state: AnalysisState, window: dict, orientation: str, thresholds: dict,
    abs_jump, cells, selection,
) -> None:
    """Near-std-threshold boundary arms at EVERY predeclared epsilon.

    All epsilons are always accumulated, so the report can never present only
    the most favourable one.
    """
    for epsilon in STD_THRESHOLD_EPSILONS:
        near = edge_near_value(
            window["baseline_lst_std_celsius"], orientation,
            thresholds["min_baseline_std_celsius"], epsilon,
        ).reshape(-1)[selection]
        if near.any():
            state.epsilon_stratum(epsilon).add(cells[near], abs_jump[near])


def _accumulate_mask_events(
    state: AnalysisState, flags, class_codes, both_valid, a_only, b_only, neither,
) -> None:
    """Mask-discontinuity counts by stratum.

    A one-sided valid pair is counted here as a DISCONTINUITY EVENT. It is never
    given a numerical anomaly jump anywhere in this module.
    """
    import numpy as np

    def _count(mask, selector):
        return int(np.count_nonzero(mask & selector))

    for stratum in MASK_STRATA:
        if stratum == "all_pairs":
            selector = np.ones(both_valid.shape, dtype=bool)
        elif stratum in flags:
            selector = flags[stratum].reshape(-1)
        elif stratum in OVERLAP_CODES:
            selector = class_codes.reshape(-1) == OVERLAP_CODES[stratum]
        else:
            continue
        state.mask_bump(stratum, "both_valid", _count(both_valid, selector))
        state.mask_bump(stratum, "a_only_valid", _count(a_only, selector))
        state.mask_bump(stratum, "b_only_valid", _count(b_only, selector))
        state.mask_bump(stratum, "neither_valid", _count(neither, selector))


def _accumulate_pathrow_units(
    state: AnalysisState, class_codes_flat, block_ids, pathrow_masks,
    orientation, selection, shape,
) -> None:
    """Per-interface pair and block counts for `pathrow_only` pairs only."""
    import numpy as np

    if not pathrow_masks or not pathrow_masks.get("interfaces"):
        return
    pathrow_only = class_codes_flat == OVERLAP_CODES[CLASS_PATHROW_ONLY]
    if not pathrow_only.any():
        return
    for interface, mask in pathrow_masks["interfaces"].items():
        edge_mask = edge_mask_from_pixel_mask(mask, orientation).reshape(-1)[selection]
        chosen = edge_mask & pathrow_only
        if not chosen.any():
            continue
        state.pathrow_only_pairs[interface] = (
            state.pathrow_only_pairs.get(interface, 0) + int(chosen.sum())
        )
        blocks = state.pathrow_only_blocks.setdefault(interface, set())
        blocks.update(int(b) for b in np.unique(block_ids[chosen]))


def _build_sample_rows(
    rows, cols, orientation, block_ids, class_codes_flat, flags, selection, window,
    anomaly_jump, numerator, denominator, std_a, std_b, *, stride: int,
) -> list[dict]:
    """A strided slice of fully-populated pair records for the reservoir sample."""
    import numpy as np

    stride = max(1, int(stride))
    index = np.arange(rows.size)[::stride]
    if not index.size:
        return []

    code_to_label = {code: label for label, code in OVERLAP_CODES.items()}
    b_rows, b_cols = endpoint_b_rows_cols(rows[index], cols[index], orientation)
    flag_values = {
        name: flags[name].reshape(-1)[selection][index] for name in BOUNDARY_FLAGS
    }

    records = []
    for position, i in enumerate(index):
        record = OrderedDict((
            ("row_a", int(rows[i])), ("col_a", int(cols[i])),
            ("row_b", int(b_rows[position])), ("col_b", int(b_cols[position])),
            ("orientation", orientation),
            ("spatial_block", block_id_to_label(int(block_ids[i]))),
            ("boundary_class", code_to_label.get(int(class_codes_flat[i]))),
            ("anomaly_jump", float(anomaly_jump[i])),
            ("abs_anomaly_jump", float(abs(anomaly_jump[i]))),
            ("numerator_contribution", float(numerator[i])),
            ("denominator_contribution", float(denominator[i])),
            ("baseline_std_a", float(std_a[i])),
            ("baseline_std_b", float(std_b[i])),
        ))
        for name, values in flag_values.items():
            record[name] = bool(values[position])
        records.append(record)
    return records


# =============================================================================
# Streaming driver
# =============================================================================
#: Every pair is analysed; only the CSV sample is strided, so the sample stays
#: bounded without touching a single scientific number.
PAIR_SAMPLE_STRIDE = 199


def window_raster_roles(plan: "OrderedDict[str, dict]") -> list[str]:
    """Input roles read per window (present rasters only)."""
    return [
        role for role, entry in plan.items()
        if Path(entry["path"]).exists()
        and entry["family"] in ("target", "target_component", "support", "mask", "covariate")
    ]


def run_streaming_pass(
    plan: "OrderedDict[str, dict]", *, height: int, width: int,
    pathrow_masks: dict | None, thresholds: dict | None = None,
    window_rows: int = WINDOW_ROWS, log=None,
) -> AnalysisState:
    """One bounded-memory pass over the whole adjacency lattice.

    Row windows carry a one-row halo so every vertical pair is built exactly
    once. Nothing is written; no array outlives its window.
    """
    import time

    thresholds = thresholds or step5_thresholds()
    roles = window_raster_roles(plan)
    state = AnalysisState()
    started = time.time()

    for start, read_stop, horizontal_rows, vertical_rows in iter_row_windows(
        height, window_rows,
    ):
        window_started = time.time()
        block = {
            role: read_window(Path(plan[role]["path"]), start, read_stop)
            for role in roles
        }
        pathrow_block = None
        if pathrow_masks:
            pathrow_block = {
                "union": pathrow_masks["union"][start:read_stop, :],
                "interfaces": {
                    name: mask[start:read_stop, :]
                    for name, mask in (pathrow_masks.get("interfaces") or {}).items()
                },
            }

        if horizontal_rows > 0:
            horizontal = {k: _slice_h(v, horizontal_rows) for k, v in block.items()}
            horizontal_pathrow = None
            if pathrow_block:
                horizontal_pathrow = {
                    "union": _slice_h(pathrow_block["union"], horizontal_rows),
                    "interfaces": {
                        name: _slice_h(mask, horizontal_rows)
                        for name, mask in pathrow_block["interfaces"].items()
                    },
                }
            analyse_window(
                state, horizontal, "horizontal", 0, thresholds=thresholds,
                pathrow_masks=horizontal_pathrow, row_offset=start,
                sample_stride=PAIR_SAMPLE_STRIDE,
            )

        if vertical_rows > 0:
            analyse_window(
                state, block, "vertical", 1, thresholds=thresholds,
                pathrow_masks=pathrow_block, row_offset=start,
                sample_stride=PAIR_SAMPLE_STRIDE,
            )

        state.windows_processed += 1
        record = OrderedDict((
            ("window_start_row", int(start)),
            ("window_stop_row", int(read_stop)),
            ("horizontal_rows", int(horizontal_rows)),
            ("vertical_rows", int(vertical_rows)),
            ("rss_mib", process_rss_mib()),
            ("elapsed_s", round(time.time() - window_started, 3)),
            ("cumulative_pairs", int(sum(state.pair_counts.values()))),
        ))
        state.resource_log.append(record)
        if log is not None:
            log.info(
                "[window %d-%d] rss=%s MiB pairs=%d elapsed=%.2fs",
                start, read_stop, record["rss_mib"], record["cumulative_pairs"],
                record["elapsed_s"],
            )

    state.resource_log.append(OrderedDict((
        ("stage", "streaming_pass"),
        ("windows_processed", state.windows_processed),
        ("total_elapsed_s", round(time.time() - started, 3)),
        ("peak_rss_mib", process_rss_mib()),
    )))
    return state


# =============================================================================
# Aggregation into report structures
# =============================================================================
def _mean_of(accumulator: MeanAccumulator | None):
    return accumulator.point_estimate() if accumulator is not None else None


def build_detection_report(state: AnalysisState) -> dict:
    """Section 2: is there a residual to attribute at all?"""
    per_product: "OrderedDict[str, dict]" = OrderedDict()
    for product, statistic, accessor in (
        (TARGET_CMB, "abs_target_jump", state.cmb),
        (TARGET_ANOMALY, "abs_anomaly_jump", state.anomaly),
    ):
        accumulator = accessor.get(("all_pairs", statistic))
        histogram = state.histograms[product]
        per_product[product] = OrderedDict((
            ("n_pairs", accumulator.n_pairs if accumulator else 0),
            ("n_units", accumulator.n_units if accumulator else 0),
            ("mean_abs_jump", _mean_of(accumulator)),
            ("p95_abs_jump", histogram.quantile(95.0)),
            ("p99_abs_jump", histogram.quantile(99.0)),
            ("max_abs_jump", histogram.edges[-1] if histogram.overflow else
             histogram.quantile(100.0)),
            ("histogram", histogram.describe()),
        ))
    return OrderedDict((
        ("per_product", per_product),
        ("pair_counts_by_orientation", OrderedDict(sorted(state.pair_counts.items()))),
        ("dropped_pairs", OrderedDict(sorted(state.dropped.items()))),
        ("drop_policy",
         "A pair is retained only when every component raster is finite at BOTH "
         "endpoints. Invalid pixels are dropped and counted; zero is never "
         "substituted for missing data."),
        ("inference_basis",
         "Continuous pairwise jumps. Hotspot percentiles are descriptive map "
         "thresholds and never gate a claim."),
    ))


def build_cmb_report(state: AnalysisState, intervals: dict) -> dict:
    """Section 3: current-minus-baseline decomposition by boundary class."""
    rows = []
    for label in ("all_pairs", *STRATIFIED_CLASSES):
        target = state.cmb.get((label, "abs_target_jump"))
        if target is None or target.n_pairs == 0:
            continue
        rows.append(OrderedDict((
            ("boundary_class", label),
            ("n_pairs", target.n_pairs),
            ("n_units", target.n_units),
            ("mean_signed_target_jump", _mean_of(state.cmb.get((label, "signed_target_jump")))),
            ("mean_abs_target_jump", _mean_of(target)),
            ("mean_abs_current_component",
             _mean_of(state.cmb.get((label, "abs_current_component")))),
            ("mean_abs_baseline_component",
             _mean_of(state.cmb.get((label, "abs_baseline_component")))),
            ("current_share", intervals.get(("cmb", label, "current_share"))),
            ("baseline_share", intervals.get(("cmb", label, "baseline_share"))),
            ("cancellation_fraction", _mean_of(state.cmb.get((label, "cancellation")))),
            ("reinforcement_fraction", _mean_of(state.cmb.get((label, "reinforcement")))),
            ("degenerate_fraction", _mean_of(state.cmb.get((label, "degenerate")))),
        )))
    exact = state.max_residual[TARGET_CMB] <= CMB_RECONSTRUCTION_ABS_TOL
    return OrderedDict((
        ("identity", decomposition_formulas()["current_minus_baseline"]),
        ("reconstruction_tolerance", CMB_RECONSTRUCTION_ABS_TOL),
        ("max_reconstruction_residual", state.max_residual[TARGET_CMB]),
        ("reconstruction_exact", bool(exact)),
        ("by_class", rows),
        ("dominance_note",
         "Dominance is decided from the bootstrap SHARE interval, never from a "
         "signed mean: opposite-signed components can cancel to a small signed "
         "mean while both remain large."),
    ))


def build_anomaly_report(state: AnalysisState, intervals: dict) -> dict:
    """Section 4: anomaly numerator/denominator decomposition."""
    rows = []
    for label in ("all_pairs", *STRATIFIED_CLASSES):
        target = state.anomaly.get((label, "abs_anomaly_jump"))
        if target is None or target.n_pairs == 0:
            continue
        rows.append(OrderedDict((
            ("boundary_class", label),
            ("n_pairs", target.n_pairs),
            ("n_units", target.n_units),
            ("mean_signed_anomaly_jump",
             _mean_of(state.anomaly.get((label, "signed_anomaly_jump")))),
            ("mean_abs_anomaly_jump", _mean_of(target)),
            ("mean_abs_numerator_contribution",
             _mean_of(state.anomaly.get((label, "abs_numerator_contribution")))),
            ("mean_abs_denominator_contribution",
             _mean_of(state.anomaly.get((label, "abs_denominator_contribution")))),
            ("numerator_share", intervals.get(("anomaly", label, "numerator_share"))),
            ("denominator_share", intervals.get(("anomaly", label, "denominator_share"))),
            ("cancellation_fraction", _mean_of(state.anomaly.get((label, "cancellation")))),
            ("reinforcement_fraction", _mean_of(state.anomaly.get((label, "reinforcement")))),
            ("degenerate_fraction", _mean_of(state.anomaly.get((label, "degenerate")))),
            ("mean_min_baseline_std", _mean_of(state.anomaly.get((label, "min_baseline_std")))),
            ("mean_max_inverse_std", _mean_of(state.anomaly.get((label, "max_inverse_std")))),
            ("mean_distance_to_std_threshold",
             _mean_of(state.anomaly.get((label, "min_distance_to_std_threshold")))),
        )))
    identity = build_anomaly_identity_check(state)
    stored = build_stored_reproduction_check(state)
    histogram = state.std_histogram
    return OrderedDict((
        ("identity", decomposition_formulas()["anomaly_zscore"]),
        ("algebraic_identity_check", identity),
        ("stored_raster_reproduction_check", stored),
        ("checks_are_independent",
         "The algebraic identity check gates the audit; the stored-raster "
         "reproduction check never does. Expected float32 serialization error is "
         "NEVER treated as a decomposition failure."),
        # Back-compatible headline fields, both sourced from CHECK 1 only.
        ("reconstruction_tolerance", ANOMALY_IDENTITY_ABS_TOL),
        ("max_reconstruction_residual", state.max_residual[TARGET_ANOMALY]),
        ("reconstruction_exact", identity["passed"]),
        ("by_class", rows),
        ("baseline_std_distribution", OrderedDict((
            ("min", histogram.quantile(0.0)),
            ("p05", histogram.quantile(5.0)),
            ("median", histogram.quantile(50.0)),
            ("p95", histogram.quantile(95.0)),
            ("max", histogram.quantile(100.0)),
            ("histogram", histogram.describe()),
        ))),
    ))


def build_anomaly_identity_check(state: AnalysisState) -> dict:
    """CHECK 1: does the symmetric decomposition reproduce Z_B - Z_A in float64?

    This is a statement about the DECOMPOSITION and it gates the audit.
    """
    # Every pair is judged against its OWN scale-aware tolerance, so the verdict
    # is the exceedance count, not a comparison of the raw maximum.
    return OrderedDict((
        ("check", "algebraic_identity"),
        ("question",
         "Recomputing Z_A = D_A/S_A and Z_B = D_B/S_B in float64, does "
         "Z_B - Z_A equal numerator_contribution + denominator_contribution?"),
        ("computed_in", "float64"),
        ("identity", "Z_B - Z_A == numerator_contribution + denominator_contribution"),
        ("exactness",
         "exact in real arithmetic; the only admissible residual is float64 "
         "round-off"),
        ("tolerance_absolute", ANOMALY_IDENTITY_ABS_TOL),
        ("tolerance_relative", ANOMALY_IDENTITY_REL_TOL),
        ("tolerance_policy", ANOMALY_IDENTITY_TOLERANCE_POLICY),
        ("n_pairs_checked", state.identity_pairs_checked),
        ("n_pairs_exceeding_tolerance", state.identity_pairs_exceeding),
        ("max_absolute_residual", state.max_residual[TARGET_ANOMALY]),
        ("max_residual_over_tolerance", state.identity_max_tolerance_ratio),
        ("passed", bool(state.identity_pairs_exceeding == 0)),
        ("gates_the_audit", True),
        ("failure_meaning",
         "a failure here means the decomposition itself is wrong, and the audit "
         "terminates as invalid_inputs"),
    ))


def build_stored_reproduction_check(state: AnalysisState) -> dict:
    """CHECK 2: how closely does float64 D/S reproduce the STORED float32 raster?

    This is a statement about float32 SERIALIZATION, not about the
    decomposition. It is descriptive and never gates the audit.
    """
    histogram = state.stored_reproduction
    pixels = state.stored_reproduction_pixels
    percentiles = OrderedDict(
        (f"p{str(p).replace('.', '_')}_abs_error", histogram.quantile(p))
        for p in ANOMALY_STORED_REPRODUCTION_PERCENTILES
    )
    if state.stored_reproduction_max <= ANOMALY_STORED_REPRODUCTION_TOL:
        status = "within_predeclared_tolerance"
    elif state.stored_reproduction_max <= ANOMALY_STORED_REPRODUCTION_POLICY_TOL:
        status = "within_step5_reproduction_policy"
    else:
        status = "exceeds_step5_reproduction_policy"

    return OrderedDict((
        ("check", "stored_raster_reproduction"),
        ("question",
         "How closely does the recomputed float64 D/S reproduce the stored "
         "float32 anomaly_zscore raster?"),
        ("compared", "float64 D/S (recomputed) vs stored float32 anomaly_zscore"),
        ("unit", "z-score"),
        ("predeclared_tolerance", ANOMALY_STORED_REPRODUCTION_TOL),
        ("predeclared_tolerance_source", ANOMALY_STORED_REPRODUCTION_TOL_SOURCE),
        ("step5_reproduction_policy_tolerance", ANOMALY_STORED_REPRODUCTION_POLICY_TOL),
        ("step5_reproduction_policy_source", ANOMALY_STORED_REPRODUCTION_POLICY_SOURCE),
        ("n_pixels_checked", pixels),
        ("max_abs_error", state.stored_reproduction_max),
        *percentiles.items(),
        ("fraction_within_predeclared_tolerance",
         (state.stored_reproduction_within_predeclared / pixels) if pixels else None),
        ("fraction_within_policy_tolerance",
         (state.stored_reproduction_within_policy / pixels) if pixels else None),
        ("histogram", histogram.describe()),
        ("status", status),
        ("gates_the_audit", False),
        ("is_decomposition_failure", False),
        ("interpretation",
         "Step5 divided its own internal float32 difference and serialised the "
         "quotient to float32. A residual of this size is EXPECTED serialization "
         "error and is never treated as a decomposition failure; only the "
         "algebraic identity check can invalidate the decomposition."),
    ))


def build_mask_report(state: AnalysisState, epsilon_rows: list[dict]) -> dict:
    """Section 5: mask discontinuities and threshold effects."""
    rows = []
    for stratum in MASK_STRATA:
        counts = state.mask_counts.get(stratum)
        if not counts:
            continue
        total = sum(counts.values())
        one_sided = counts["a_only_valid"] + counts["b_only_valid"]
        rows.append(OrderedDict((
            ("stratum", stratum),
            ("both_valid", counts["both_valid"]),
            ("a_only_valid", counts["a_only_valid"]),
            ("b_only_valid", counts["b_only_valid"]),
            ("neither_valid", counts["neither_valid"]),
            ("total_pairs", total),
            ("mask_discontinuity_events", one_sided),
            ("mask_discontinuity_rate", (one_sided / total) if total else None),
        )))
    baseline = next((r for r in rows if r["stratum"] == "all_pairs"), None)
    baseline_rate = baseline["mask_discontinuity_rate"] if baseline else None
    for row in rows:
        rate = row["mask_discontinuity_rate"]
        row["rate_relative_to_all_pairs"] = (
            (rate / baseline_rate) if (rate is not None and baseline_rate) else None
        )
    return OrderedDict((
        ("one_sided_policy",
         "A pair valid on only ONE side is a mask discontinuity EVENT. It is "
         "never assigned a numerical anomaly jump and never enters a "
         "decomposition."),
        ("by_stratum", rows),
        ("epsilon_predeclared", STD_THRESHOLD_EPSILON_PRIMARY),
        ("epsilon_sensitivity", epsilon_rows),
        ("epsilon_selection_policy",
         "The primary epsilon was predeclared before any result was inspected; "
         "every sensitivity epsilon is reported unconditionally."),
    ))


def build_excess_report(rows: list[dict]) -> dict:
    """Section 6: matched-control excess absolute jump by boundary definition."""
    return OrderedDict((
        ("definition",
         "excess_absolute_jump = boundary mean |jump| - matched-control mean "
         "|jump|, where controls are within-block, same-orientation pairs "
         "carrying no known boundary flag, direct-standardised to the boundary "
         "stratum distribution."),
        ("matched_control_strategy", MATCHED_CONTROL_STRATEGY),
        ("rows", rows),
    ))


def build_pathrow_report(
    availability: dict, state: AnalysisState, stratified_rows: list[dict],
) -> dict:
    """Section 7: the metadata-derived path/row test.

    A positive result requires `pathrow_only` pairs -- which exclude every
    support and threshold overlap by construction -- to carry an excess interval
    wholly above zero across enough independent spatial units AND enough
    distinct metadata interfaces.
    """
    report = OrderedDict(availability)
    report["evidence_qualification"] = (
        "Metadata-derived source-footprint boundaries. This is NOT pixel-level "
        "selected-scene provenance: a flag means a metadata boundary crosses the "
        "pair, not which scene supplied either pixel."
    )
    report["stratified_rows"] = stratified_rows
    report["generalises_beyond_manavgat"] = False

    if availability.get("availability") != "available":
        report["verdict"] = "unavailable"
        report["supported"] = False
        report["reason_not_supported"] = availability.get("reason")
        return report

    # The verdict is read from the CELSIUS product: current_minus_baseline is the
    # raw residual and cannot be inflated or deflated by the anomaly denominator.
    only_row = next(
        (r for r in stratified_rows
         if r["stratum"] == CLASS_PATHROW_ONLY and r.get("product") == TARGET_CMB),
        None,
    )
    contributing = {
        interface: count for interface, count in state.pathrow_only_pairs.items() if count
    }
    n_interfaces = len(contributing)
    n_units = len(set().union(*state.pathrow_only_blocks.values())) \
        if state.pathrow_only_blocks else 0

    report["pathrow_only_pairs_by_interface"] = OrderedDict(sorted(contributing.items()))
    report["n_interfaces"] = n_interfaces
    report["n_units"] = n_units
    report["min_units_required"] = MIN_PATHROW_ONLY_UNITS
    report["min_interfaces_required"] = MIN_PATHROW_INTERFACES

    if only_row is None or n_units < MIN_PATHROW_ONLY_UNITS or n_interfaces < MIN_PATHROW_INTERFACES:
        report["verdict"] = "insufficient_pathrow_only_support"
        report["supported"] = False
        report["reason_not_supported"] = (
            f"pathrow_only carries {n_units} independent spatial units "
            f"(>= {MIN_PATHROW_ONLY_UNITS} required) across {n_interfaces} distinct "
            f"metadata interfaces (>= {MIN_PATHROW_INTERFACES} required)"
        )
        return report

    verdict = only_row.get("verdict")
    report["verdict"] = verdict
    report["supported"] = verdict == "supported_excess"
    if not report["supported"]:
        report["reason_not_supported"] = (
            "the pathrow_only excess interval is not wholly above zero"
        )
    report["not_explained_by_support_overlap"] = True
    report["support_overlap_note"] = (
        "pathrow_only excludes every pair that also carries an observation-"
        "support or threshold boundary, so the estimate cannot be produced by "
        "support overlap."
    )
    return report


# =============================================================================
# Bootstrap orchestration (identical draws inside every comparison)
# =============================================================================
SHARE_COMPARISONS = OrderedDict((
    ("cmb", ("current_share", "baseline_share")),
    ("anomaly", ("numerator_share", "denominator_share")),
))


def compute_share_intervals(state: AnalysisState) -> dict:
    """Bootstrap every component share, reusing ONE draw per comparison.

    The two shares of a comparison are defined on the SAME pairs and therefore
    the same spatial units in the same order, so a single index matrix is drawn
    per (product, class) and handed to both. A mismatch is an error, never a
    silent re-draw.
    """
    intervals: dict[tuple[str, str, str], dict] = {}
    for product, (first, second) in SHARE_COMPARISONS.items():
        source = state.cmb if product == "cmb" else state.anomaly
        for label in ("all_pairs", *STRATIFIED_CLASSES):
            a = source.get((label, first))
            b = source.get((label, second))
            if a is None or b is None or a.n_pairs == 0:
                continue
            units_a = a.unit_arrays()[0]
            units_b = b.unit_arrays()[0]
            if units_a != units_b:
                raise ResidualSeamError(
                    f"{product}/{label}: component shares do not share a unit set; "
                    "identical bootstrap draws would be meaningless"
                )
            indices = draw_bootstrap_indices(len(units_a))
            for name, accumulator in ((first, a), (second, b)):
                interval = bootstrap_mean_interval(accumulator, indices)
                interval["identical_draws_with"] = second if name == first else first
                interval["comparison"] = f"{product}:{label}"
                intervals[(product, label, name)] = interval
    return intervals


def compute_excess_rows(state: AnalysisState) -> list[dict]:
    """Matched-control excess absolute jump for every boundary definition."""
    rows: list[dict] = []
    for product in TARGET_PRODUCTS:
        control = state.control_stratum(product)
        for boundary in EXCESS_JUMP_BOUNDARIES:
            accumulator = state.boundary_strata.get((product, boundary))
            if accumulator is None or accumulator.n_pairs == 0:
                continue
            boundary_blocks, control_blocks, diagnostics = matched_block_accumulators(
                accumulator, control, stratum_space=STRATUM_SPACE,
            )
            interval = bootstrap_difference_interval(boundary_blocks, control_blocks)
            row = OrderedDict((
                ("product", product),
                ("boundary", boundary),
                ("n_boundary_pairs", interval["n_boundary_pairs"]),
                ("n_control_pairs", interval["n_control_pairs"]),
                ("n_units", interval["n_units"]),
                ("boundary_mean_abs_jump", interval["boundary_mean_abs_jump"]),
                ("control_mean_abs_jump", interval["control_mean_abs_jump"]),
                ("excess_absolute_jump", interval["excess_absolute_jump"]),
                ("interval_low", interval["interval_low"]),
                ("interval_high", interval["interval_high"]),
                ("verdict", classify_excess_interval(interval)),
                ("n_bootstrap_used", interval["n_bootstrap_used"]),
                ("n_bootstrap_skipped", interval["n_bootstrap_skipped"]),
                ("skipped_reason", interval["skipped_reason"]),
                ("status", interval["status"]),
            ))
            row.update({f"matching_{k}": v for k, v in diagnostics.items()})
            rows.append(row)
    return rows


def compute_epsilon_rows(state: AnalysisState) -> list[dict]:
    """Near-std-threshold excess at EVERY predeclared epsilon, always reported."""
    control = state.control_stratum(TARGET_ANOMALY)
    rows: list[dict] = []
    for epsilon in STD_THRESHOLD_EPSILONS:
        accumulator = state.epsilon_strata.get(float(epsilon))
        if accumulator is None:
            rows.append(OrderedDict((
                ("epsilon", float(epsilon)),
                ("is_primary", float(epsilon) == STD_THRESHOLD_EPSILON_PRIMARY),
                ("n_boundary_pairs", 0), ("n_units", 0),
                ("excess_absolute_jump", None),
                ("interval_low", None), ("interval_high", None),
                ("verdict", "insufficient_evidence"),
            )))
            continue
        boundary_blocks, control_blocks, _ = matched_block_accumulators(
            accumulator, control, stratum_space=STRATUM_SPACE,
        )
        interval = bootstrap_difference_interval(boundary_blocks, control_blocks)
        rows.append(OrderedDict((
            ("epsilon", float(epsilon)),
            ("is_primary", float(epsilon) == STD_THRESHOLD_EPSILON_PRIMARY),
            ("n_boundary_pairs", interval["n_boundary_pairs"]),
            ("n_control_pairs", interval["n_control_pairs"]),
            ("n_units", interval["n_units"]),
            ("boundary_mean_abs_jump", interval["boundary_mean_abs_jump"]),
            ("control_mean_abs_jump", interval["control_mean_abs_jump"]),
            ("excess_absolute_jump", interval["excess_absolute_jump"]),
            ("interval_low", interval["interval_low"]),
            ("interval_high", interval["interval_high"]),
            ("verdict", classify_excess_interval(interval)),
            ("n_bootstrap_used", interval["n_bootstrap_used"]),
            ("n_bootstrap_skipped", interval["n_bootstrap_skipped"]),
        )))
    return rows


def compute_pathrow_rows(state: AnalysisState, excess_rows: list[dict]) -> list[dict]:
    """The three predeclared path/row strata, in order."""
    wanted = OrderedDict((
        ("source_path_row_boundary", "all_pathrow_boundary_pairs"),
        (CLASS_PATHROW_ONLY, CLASS_PATHROW_ONLY),
        (CLASS_SUPPORT_AND_PATHROW, CLASS_SUPPORT_AND_PATHROW),
    ))
    rows: list[dict] = []
    for boundary, label in wanted.items():
        for source in excess_rows:
            if source["boundary"] != boundary:
                continue
            row = OrderedDict(source)
            row["stratum"] = label
            row["n_interfaces"] = (
                len([c for c in state.pathrow_only_pairs.values() if c])
                if boundary == CLASS_PATHROW_ONLY else None
            )
            rows.append(row)
    return rows


def build_bootstrap_summary(
    state: AnalysisState, share_intervals: dict, excess_rows: list[dict],
    epsilon_rows: list[dict],
) -> dict:
    """Section: the bookkeeping every bootstrap must expose."""
    rows: list[dict] = []
    for (product, label, name), interval in sorted(share_intervals.items()):
        rows.append(OrderedDict((
            ("statistic", f"{product}:{label}:{name}"),
            ("n_pairs", interval["n_pairs"]),
            ("n_units", interval["n_units"]),
            ("unit_type", interval["unit_type"]),
            ("n_bootstrap_requested", interval["n_bootstrap_requested"]),
            ("n_bootstrap_used", interval["n_bootstrap_used"]),
            ("n_bootstrap_skipped", interval["n_bootstrap_skipped"]),
            ("skipped_reason", interval["skipped_reason"]),
            ("point_estimate", interval["point_estimate"]),
            ("interval_low", interval["interval_low"]),
            ("interval_high", interval["interval_high"]),
            ("identical_draws_with", interval.get("identical_draws_with")),
            ("status", interval["status"]),
        )))
    for source, prefix in ((excess_rows, "excess"), (epsilon_rows, "epsilon")):
        for row in source:
            name = row.get("boundary") or f"eps_{row.get('epsilon')}"
            rows.append(OrderedDict((
                ("statistic", f"{prefix}:{row.get('product', TARGET_ANOMALY)}:{name}"),
                ("n_pairs", row.get("n_boundary_pairs")),
                ("n_units", row.get("n_units")),
                ("unit_type", "spatial_block"),
                ("n_bootstrap_requested", BOOTSTRAP_REPLICATES),
                ("n_bootstrap_used", row.get("n_bootstrap_used")),
                ("n_bootstrap_skipped", row.get("n_bootstrap_skipped")),
                ("skipped_reason", row.get("skipped_reason")),
                ("point_estimate", row.get("excess_absolute_jump")),
                ("interval_low", row.get("interval_low")),
                ("interval_high", row.get("interval_high")),
                ("identical_draws_with", None),
                ("status", row.get("status")),
            )))
    return OrderedDict((
        ("configuration", OrderedDict((
            ("unit", "spatial_block"),
            ("block_size_cells", BOOTSTRAP_BLOCK_SIZE_CELLS),
            ("replicates", BOOTSTRAP_REPLICATES),
            ("seed", BOOTSTRAP_SEED),
            ("ci", BOOTSTRAP_CI),
            ("resamples_individual_pairs", False),
            ("identical_draws_within_a_comparison", True),
        ))),
        ("rows", rows),
    ))


# =============================================================================
# Hotspot / map pass (descriptive only)
# =============================================================================
MAP_DTYPES = {
    "residual_cmb_abs_jump": "float32",
    "residual_cmb_signed_jump": "float32",
    "cmb_hotspot_class": "uint8",
    "residual_anomaly_abs_jump": "float32",
    "residual_anomaly_signed_jump": "float32",
    "anomaly_hotspot_class": "uint8",
    "anomaly_mask_discontinuity": "uint8",
    "current_support_change": "uint8",
    "baseline_support_change": "uint8",
    "support_pathrow_overlap": "uint8",
    "baseline_std": "float32",
    "near_std_threshold": "uint8",
    "cmb_attribution": "uint8",
    "anomaly_attribution": "uint8",
}


def hotspot_thresholds(state: AnalysisState) -> dict:
    """Descriptive-only |jump| cuts, read from the fixed-edge histograms."""
    thresholds: "OrderedDict[str, dict]" = OrderedDict()
    for product in TARGET_PRODUCTS:
        histogram = state.histograms[product]
        thresholds[product] = OrderedDict(
            (HOTSPOT_LABELS[p], histogram.quantile(p)) for p in HOTSPOT_PERCENTILES
        )
        thresholds[product]["histogram"] = histogram.describe()
        thresholds[product]["descriptive_only"] = True
        thresholds[product]["is_significance_threshold"] = False
    return thresholds


def build_hotspot_report(overlap_counts: dict, thresholds: dict) -> dict:
    """Section: what fraction of hotspot pairs touches each mechanism."""
    rows: list[dict] = []
    for (product, label, mechanism), count in sorted(overlap_counts.items()):
        if mechanism == "__total__":
            continue
        total = overlap_counts.get((product, label, "__total__"), 0)
        rows.append(OrderedDict((
            ("product", product),
            ("hotspot_class", label),
            ("mechanism", mechanism),
            ("hotspot_pairs", total),
            ("hotspot_pairs_intersecting_mechanism", count),
            ("fraction", (count / total) if total else None),
        )))
    return OrderedDict((
        ("thresholds", thresholds),
        ("interpretation",
         "Hotspot percentiles are DESCRIPTIVE map thresholds. They are not a "
         "statistical significance threshold and no final status depends on them."),
        ("rows", rows),
    ))


# =============================================================================
# CSV table construction
# =============================================================================
def csv_rows_current_minus_baseline(cmb: dict) -> tuple[list[dict], list[str]]:
    rows = [
        OrderedDict(
            (k, v) for k, v in row.items() if not isinstance(v, dict)
        ) | OrderedDict((
            ("current_share_point", (row.get("current_share") or {}).get("point_estimate")),
            ("current_share_low", (row.get("current_share") or {}).get("interval_low")),
            ("current_share_high", (row.get("current_share") or {}).get("interval_high")),
            ("baseline_share_point", (row.get("baseline_share") or {}).get("point_estimate")),
            ("baseline_share_low", (row.get("baseline_share") or {}).get("interval_low")),
            ("baseline_share_high", (row.get("baseline_share") or {}).get("interval_high")),
        ))
        for row in cmb.get("by_class") or []
    ]
    columns = list(rows[0].keys()) if rows else ["boundary_class"]
    return rows, columns


def csv_rows_anomaly(anomaly: dict) -> tuple[list[dict], list[str]]:
    rows = [
        OrderedDict(
            (k, v) for k, v in row.items() if not isinstance(v, dict)
        ) | OrderedDict((
            ("numerator_share_point", (row.get("numerator_share") or {}).get("point_estimate")),
            ("numerator_share_low", (row.get("numerator_share") or {}).get("interval_low")),
            ("numerator_share_high", (row.get("numerator_share") or {}).get("interval_high")),
            ("denominator_share_point",
             (row.get("denominator_share") or {}).get("point_estimate")),
            ("denominator_share_low", (row.get("denominator_share") or {}).get("interval_low")),
            ("denominator_share_high",
             (row.get("denominator_share") or {}).get("interval_high")),
        ))
        for row in anomaly.get("by_class") or []
    ]
    columns = list(rows[0].keys()) if rows else ["boundary_class"]
    return rows, columns


def csv_rows_simple(rows: list[dict], fallback: list[str]) -> tuple[list[dict], list[str]]:
    flattened = [
        OrderedDict((k, v) for k, v in row.items() if not isinstance(v, (dict, list)))
        for row in rows
    ]
    columns = list(flattened[0].keys()) if flattened else list(fallback)
    return flattened, columns


# =============================================================================
# Hotspot + map pass (descriptive overlays; NOTHING is smoothed)
# =============================================================================
def open_map_writers(root: Path, grid_profile: dict, experiment_id: str):
    """Open every diagnostic overlay for windowed writing.

    Each overlay uses the EXACT input grid. No resampling, no interpolation, no
    smoothing kernel is applied anywhere -- the overlays ARE the lattice.
    """
    import rasterio

    writers: "OrderedDict[str, object]" = OrderedDict()
    for key, relative in MAP_OUTPUTS.items():
        path = Path(root) / "maps" / relative
        assert_namespace_safe([path], experiment_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        dtype = MAP_DTYPES[key]
        profile = dict(grid_profile)
        profile.update(
            driver="GTiff", count=1, dtype=dtype, compress="lzw", BIGTIFF="IF_SAFER",
            nodata=(float("nan") if dtype == "float32" else 0),
        )
        writers[key] = rasterio.open(path, "w", **profile)
    return writers


def run_hotspot_and_map_pass(
    plan: "OrderedDict[str, dict]", *, root: Path, experiment_id: str,
    height: int, width: int, grid_profile: dict, pathrow_masks: dict | None,
    thresholds: dict, hotspot_cuts: dict, window_rows: int = WINDOW_ROWS, log=None,
) -> dict:
    """Second bounded pass: descriptive hotspot overlap + diagnostic overlays.

    Runs only after the first pass has fixed the hotspot cuts, so the cuts are a
    property of the whole distribution rather than of one window.
    """
    import numpy as np

    step5 = thresholds
    overlap: dict[tuple[str, str, str], int] = {}
    roles = window_raster_roles(plan)
    writers = open_map_writers(root, grid_profile, experiment_id)
    written: list[str] = []

    def bump(product, label, mechanism, count):
        key = (product, label, mechanism)
        overlap[key] = overlap.get(key, 0) + int(count)

    try:
        for start, read_stop, horizontal_rows, vertical_rows in iter_row_windows(
            height, window_rows,
        ):
            block = {
                role: read_window(Path(plan[role]["path"]), start, read_stop)
                for role in roles
            }
            pathrow_block = None
            if pathrow_masks:
                pathrow_block = {
                    "union": pathrow_masks["union"][start:read_stop, :],
                    "interfaces": {},
                }

            emit_rows = min(start + window_rows, height) - start
            panels = {
                key: np.zeros((emit_rows, width), dtype=MAP_DTYPES[key])
                for key in MAP_OUTPUTS
            }
            for key, dtype in MAP_DTYPES.items():
                if dtype == "float32":
                    panels[key][:] = np.nan

            for orientation, rows_available in (
                ("horizontal", horizontal_rows), ("vertical", vertical_rows),
            ):
                if rows_available <= 0:
                    continue
                if orientation == "horizontal":
                    sub = {k: _slice_h(v, horizontal_rows) for k, v in block.items()}
                    sub_pathrow = (
                        {"union": _slice_h(pathrow_block["union"], horizontal_rows),
                         "interfaces": {}}
                        if pathrow_block else None
                    )
                else:
                    sub, sub_pathrow = block, pathrow_block

                _paint_window(
                    panels, sub, orientation, step5, sub_pathrow, hotspot_cuts,
                    emit_rows, width, bump,
                )

            window = _rio_window(0, start, width, emit_rows)
            for key, writer in writers.items():
                writer.write(panels[key].astype(MAP_DTYPES[key]), 1, window=window)
            if log is not None:
                log.info(
                    "[map-window %d-%d] rss=%s MiB", start, start + emit_rows,
                    process_rss_mib(),
                )
    finally:
        for writer in writers.values():
            path = writer.name
            writer.close()
            written.append(str(path))

    return {"overlap_counts": overlap, "written": written}


def _rio_window(col_off, row_off, width, height):
    from rasterio.windows import Window

    return Window(int(col_off), int(row_off), int(width), int(height))


#: Mechanisms reported in the hotspot-overlap table.
HOTSPOT_MECHANISMS = (
    "current_support_change",
    "baseline_valid_year_change",
    "baseline_annual_date_support_change",
    "same_day_multiplicity_change",
    "low_baseline_std_boundary",
    "near_std_threshold_boundary",
    "current_count_threshold_boundary",
    "baseline_count_threshold_boundary",
    "source_path_row_boundary",
    CLASS_PATHROW_ONLY,
    CLASS_SUPPORT_ONLY,
    CLASS_SUPPORT_AND_PATHROW,
    CLASS_NONE,
)


def _paint_window(
    panels, window, orientation, thresholds, pathrow_masks, hotspot_cuts,
    emit_rows, width, bump,
) -> None:
    """Paint one orientation's overlays and accumulate hotspot overlap."""
    import numpy as np

    flags = build_edge_flags(
        window, orientation, epsilon=STD_THRESHOLD_EPSILON_PRIMARY,
        thresholds=thresholds, pathrow_masks=pathrow_masks,
    )
    class_codes, support, threshold, pathrow = stratified_class_codes(flags)

    c_a, c_b = _edge_pairs(window["current_lst_celsius"], orientation)
    m_a, m_b = _edge_pairs(window["baseline_lst_mean_celsius"], orientation)
    d_a, d_b = _edge_pairs(window[TARGET_CMB], orientation)
    s_a, s_b = _edge_pairs(window["baseline_lst_std_celsius"], orientation)
    z_a, z_b = _edge_pairs(window[TARGET_ANOMALY], orientation)

    cmb_valid = (
        np.isfinite(c_a) & np.isfinite(c_b) & np.isfinite(m_a) & np.isfinite(m_b)
        & np.isfinite(d_a) & np.isfinite(d_b)
    )
    _, current_component, baseline_component = decompose_current_minus_baseline(
        c_a, c_b, m_a, m_b,
    )
    cmb_jump = np.where(cmb_valid, d_b - d_a, np.nan)

    anomaly_valid = (
        np.isfinite(z_a) & np.isfinite(z_b) & np.isfinite(d_a) & np.isfinite(d_b)
        & np.isfinite(s_a) & np.isfinite(s_b) & (s_a != 0.0) & (s_b != 0.0)
    )
    za, zb, numerator, denominator = decompose_anomaly(d_a, d_b, s_a, s_b)
    anomaly_jump = np.where(anomaly_valid, zb - za, np.nan)
    one_sided = (np.isfinite(z_a) != np.isfinite(z_b))

    def place(key, values, mask=None):
        """Write edge values at their 'a' anchor, horizontal-first on collision."""
        target = panels[key]
        rows = values.shape[0]
        cols = values.shape[1]
        view = target[:rows, :cols]
        if MAP_DTYPES[key] == "float32":
            candidate = np.abs(values) if key.endswith("abs_jump") else values
            take = np.isfinite(candidate) & (
                ~np.isfinite(view) | (np.abs(candidate) > np.abs(np.nan_to_num(view)))
            )
            view[take] = candidate[take]
        else:
            selected = values.astype("uint8")
            take = (view == 0) & (selected != 0) if mask is None else (mask & (view == 0))
            view[take] = selected[take]

    place("residual_cmb_abs_jump", np.abs(cmb_jump))
    place("residual_cmb_signed_jump", cmb_jump)
    place("residual_anomaly_abs_jump", np.abs(anomaly_jump))
    place("residual_anomaly_signed_jump", anomaly_jump)
    place("baseline_std", np.minimum(s_a, s_b))

    place("current_support_change", flags["current_support_change"].astype("uint8"))
    place("baseline_support_change", (
        flags["baseline_valid_year_change"] | flags["baseline_annual_date_support_change"]
    ).astype("uint8"))
    place("near_std_threshold", flags["near_std_threshold_boundary"].astype("uint8"))
    place("support_pathrow_overlap", class_codes.astype("uint8"))
    place("anomaly_mask_discontinuity", one_sided.astype("uint8"))

    cmb_attr = np.zeros(cmb_valid.shape, dtype="uint8")
    with np.errstate(invalid="ignore"):
        cur, base = np.abs(current_component), np.abs(baseline_component)
    cmb_attr[cmb_valid & (cur > base)] = ATTRIBUTION_CODES["current_dominant"]
    cmb_attr[cmb_valid & (base > cur)] = ATTRIBUTION_CODES["baseline_mean_dominant"]
    cmb_attr[cmb_valid & (cur == base)] = ATTRIBUTION_CODES["components_tied"]
    place("cmb_attribution", cmb_attr)

    anomaly_attr = np.zeros(anomaly_valid.shape, dtype="uint8")
    num, den = np.abs(numerator), np.abs(denominator)
    anomaly_attr[anomaly_valid & (num > den)] = ANOMALY_ATTRIBUTION_CODES["numerator_dominant"]
    anomaly_attr[anomaly_valid & (den > num)] = ANOMALY_ATTRIBUTION_CODES["denominator_dominant"]
    anomaly_attr[anomaly_valid & (num == den)] = ANOMALY_ATTRIBUTION_CODES["components_tied"]
    place("anomaly_attribution", anomaly_attr)

    for product, jump, valid, key in (
        (TARGET_CMB, cmb_jump, cmb_valid, "cmb_hotspot_class"),
        (TARGET_ANOMALY, anomaly_jump, anomaly_valid, "anomaly_hotspot_class"),
    ):
        cuts = hotspot_cuts[product]
        abs_jump = np.abs(jump)
        top5 = valid & np.isfinite(abs_jump) & (abs_jump >= (cuts["top_5_percent"] or np.inf))
        top1 = valid & np.isfinite(abs_jump) & (abs_jump >= (cuts["top_1_percent"] or np.inf))
        codes = np.zeros(valid.shape, dtype="uint8")
        codes[valid] = HOTSPOT_CODES["below_top_5_percent"]
        codes[top5] = HOTSPOT_CODES["top_5_percent"]
        codes[top1] = HOTSPOT_CODES["top_1_percent"]
        place(key, codes)

        for label, selected in (("top_5_percent", top5), ("top_1_percent", top1)):
            bump(product, label, "__total__", int(selected.sum()))
            for mechanism in HOTSPOT_MECHANISMS:
                if mechanism in flags:
                    mask = flags[mechanism]
                elif mechanism in OVERLAP_CODES:
                    mask = class_codes == OVERLAP_CODES[mechanism]
                else:
                    continue
                bump(product, label, mechanism, int(np.count_nonzero(selected & mask)))


# =============================================================================
# Decision evidence assembly
# =============================================================================
def build_decision_evidence(
    *, inputs_valid: bool, invalid_reasons: list[str], share_intervals: dict,
    excess_rows: list[dict], epsilon_rows: list[dict], mask_report: dict,
    pathrow_report: dict,
) -> dict:
    """Translate the computed tables into the predeclared decision inputs."""
    excess_by_boundary: "OrderedDict[str, dict]" = OrderedDict()
    for row in excess_rows:
        if row["product"] != TARGET_CMB:
            continue
        excess_by_boundary[row["boundary"]] = row
    anomaly_excess = OrderedDict(
        (row["boundary"], row) for row in excess_rows if row["product"] == TARGET_ANOMALY
    )

    std_supported = any(
        classify_excess_interval(anomaly_excess.get(name) or {}) == "supported_excess"
        for name in ("low_baseline_std_boundary", "near_std_threshold_boundary")
    )
    all_pairs_rate = next(
        (r["mask_discontinuity_rate"] for r in mask_report.get("by_stratum") or []
         if r["stratum"] == "all_pairs"), None,
    )
    near_rate = next(
        (r["mask_discontinuity_rate"] for r in mask_report.get("by_stratum") or []
         if r["stratum"] == "near_std_threshold_boundary"), None,
    )

    return OrderedDict((
        ("inputs_valid", bool(inputs_valid)),
        ("invalid_input_reasons", list(invalid_reasons)),
        ("excess_by_boundary", excess_by_boundary),
        ("anomaly_excess_by_boundary", anomaly_excess),
        ("baseline_excess_excluding_current_only",
         excess_by_boundary.get("baseline_support_excluding_current") or {}),
        ("cmb_current_share", share_intervals.get(("cmb", "all_pairs", "current_share")) or {}),
        ("cmb_baseline_share", share_intervals.get(("cmb", "all_pairs", "baseline_share")) or {}),
        ("anomaly_numerator_share",
         share_intervals.get(("anomaly", "all_pairs", "numerator_share")) or {}),
        ("anomaly_denominator_share",
         share_intervals.get(("anomaly", "all_pairs", "denominator_share")) or {}),
        ("anomaly_std_concentration", OrderedDict((
            ("supported", std_supported),
            ("definitions", ["low_baseline_std_boundary", "near_std_threshold_boundary"]),
        ))),
        ("mask_discontinuity_near_std_threshold", OrderedDict((
            ("all_pairs_rate", all_pairs_rate),
            ("near_threshold_rate", near_rate),
            ("elevated", bool(
                all_pairs_rate is not None and near_rate is not None
                and near_rate > all_pairs_rate
            )),
        ))),
        ("near_std_epsilon_support", OrderedDict(
            (str(row["epsilon"]), row.get("verdict") == "supported_excess")
            for row in epsilon_rows
        )),
        ("pathrow_only", OrderedDict((
            ("availability", pathrow_report.get("availability")),
            ("verdict", pathrow_report.get("verdict")),
            ("supported", bool(pathrow_report.get("supported"))),
            ("n_units", pathrow_report.get("n_units")),
            ("n_interfaces", pathrow_report.get("n_interfaces")),
        ))),
    ))
