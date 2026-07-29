"""
landsat_composite_downstream_ab.py

Scientifically controlled DOWNSTREAM A/B experiment for the validated Landsat
compositing counterfactual.

    A. scene_weighted_reference   -- the current canonical compositing behaviour
    B. date_balanced_lst_only     -- date-balanced daily compositing for Landsat
                                     LST ONLY (canonical scene-weighted NDVI is
                                     kept byte-identical between chains)

The question is whether the seam reduction established by the completed
`landsat_composite_counterfactual` audit for manavgat_2021 propagates through
Step5, Step7 and Step8 WITHOUT weakening the supported within-region thermal
contribution.

WHAT THIS MODULE IS NOT
-----------------------
    - It is NOT a production reducer change. The canonical Step3 reducer default
      is untouched; this is a diagnostic candidate experiment only.
    - It NEVER returns "production approved". The strongest allowed outcome is
      `eligible_for_second_aoi_validation`, meaning ONLY that independent
      validation in a second AOI is warranted.
    - It NEVER runs Earth Engine. Every raw Landsat product is taken from
      already-frozen local files (canonical experiment outputs for the
      reference chain, the frozen counterfactual audit for the candidate chain).
    - It re-implements NO scientific computation. Step5/Step5C/Step7A-E/Step8A-C
      are executed through their OWN production callables with a namespaced
      ExperimentContext; the boundary audit, paired bootstrap, grid contract and
      atomic-checkpoint helpers are reused from
      `src/landsat_composite_counterfactual_audit.py`.

ISOLATION CONTRACT
------------------
Everything this module writes lives under

    outputs/diagnostics/landsat_composite_downstream_ab/<experiment_id>/

The frozen canonical experiment namespace
(`outputs/experiments/<experiment_id>/`) and the frozen counterfactual namespace
(`outputs/diagnostics/landsat_composite_counterfactual/<experiment_id>/`) are
READ-ONLY inputs and are never written, deleted, or used as an output root.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import src.landsat_composite_counterfactual_audit as audit
from core.paths import PROJECT_ROOT

# The namespace-safety exception and the atomic/checkpoint primitives are shared
# with the counterfactual audit on purpose: one implementation, one contract.
NamespaceSafetyError = audit.NamespaceSafetyError
write_json_atomic = audit.write_json_atomic
files_present_and_signed = audit.files_present_and_signed
grid_signature = audit.grid_signature
assert_same_grid = audit.assert_same_grid
GridMismatchError = audit.GridMismatchError
classify_paired_interval = audit.classify_paired_interval


class DownstreamABError(RuntimeError):
    """Fail-fast error for the downstream A/B experiment."""


class PrerequisiteError(DownstreamABError):
    """A required frozen input or audit prerequisite is missing/invalid."""


# =============================================================================
# Identity / versions
# =============================================================================
DIAGNOSTIC_NAMESPACE = "landsat_composite_downstream_ab"
SOURCE_AUDIT_NAMESPACE = audit.DIAGNOSTIC_NAMESPACE  # landsat_composite_counterfactual

CHAIN_REFERENCE = "scene_weighted_reference"
CHAIN_CANDIDATE = "date_balanced_lst_only"
CHAINS = (CHAIN_REFERENCE, CHAIN_CANDIDATE)

#: Chain -> output sub-namespace under the A/B root.
CHAIN_SIDE = OrderedDict((
    (CHAIN_REFERENCE, "reference"),
    (CHAIN_CANDIDATE, "candidate"),
))

REPORT_SCHEMA_VERSION = "1.0-downstream-ab"
DECISION_RULE_VERSION = "1.0-downstream-ab-ordered"

#: The ONLY candidate supported by this task. A second candidate (e.g.
#: `date_balanced_all_landsat`) is deliberately NOT implemented here.
SUPPORTED_CANDIDATES = (CHAIN_CANDIDATE,)

#: Prerequisites the source counterfactual audit must satisfy.
REQUIRED_SOURCE_FINAL_STATUS = "supported_reduction"
REQUIRED_SOURCE_CANONICAL_REPRODUCTION = "pass"


# =============================================================================
# Predeclared final statuses (ordered; see decide_final_status)
# =============================================================================
STATUS_INVALID_REFERENCE = "invalid_reference_reproduction"
STATUS_BASELINE_INVARIANCE_FAILED = "baseline_invariance_failed"
STATUS_POPULATION_REVIEW = "population_alignment_requires_review"
STATUS_SEAM_REDUCED_TRADEOFF = "seam_reduced_performance_tradeoff"
STATUS_ELIGIBLE_SECOND_AOI = "eligible_for_second_aoi_validation"
STATUS_INCONCLUSIVE = "downstream_effect_inconclusive"

FINAL_STATUSES = (
    STATUS_INVALID_REFERENCE,
    STATUS_BASELINE_INVARIANCE_FAILED,
    STATUS_POPULATION_REVIEW,
    STATUS_SEAM_REDUCED_TRADEOFF,
    STATUS_ELIGIBLE_SECOND_AOI,
    STATUS_INCONCLUSIVE,
)

#: This experiment can never conclude that the candidate is production-ready.
FORBIDDEN_CONCLUSIONS = ("production_approved", "production_ready", "approved_for_production")


# =============================================================================
# Modelling contract (frozen to the canonical Step8 configuration)
# =============================================================================
#: Primary population for the A/B comparison. Declared BEFORE the run; never
#: re-selected after seeing results.
PRIMARY_POPULATION = "burnable_tree_shrub_grass"

#: Paired candidate-minus-reference bootstrap configuration. The bootstrap UNIT
#: is the same 500 m spatial block used for Step8B's CV groups and Step8C's
#: uncertainty analysis.
PAIRED_BOOTSTRAP_REPLICATES = 1000
PAIRED_BOOTSTRAP_CI_LOWER = 2.5
PAIRED_BOOTSTRAP_CI_UPPER = 97.5

#: Deterministic tolerance for the baseline-invariance gate. Baseline features
#: are built from inputs that are byte-identical between chains, so the baseline
#: OOF predictions must agree to within float round-off only.
BASELINE_OOF_MAX_ABS_DIFF = 1e-12
BASELINE_FEATURE_MAX_ABS_DIFF = 0.0

#: Population-alignment review thresholds (predeclared, descriptive).
MIN_COMMON_ROW_RETENTION = 0.98
MIN_COMMON_POSITIVE_RETENTION = 0.98


# =============================================================================
# Reference-reproduction tolerances (predeclared)
#
# Float32 pipelines with a legitimately different operation order are compared
# semantically, not bitwise -- the same policy the counterfactual audit's
# canonical gate uses. The reference chain re-runs the CANONICAL code on the
# CANONICAL raw inputs, so agreement should be at round-off level; the
# tolerances below are the predeclared failure boundary, not an expectation.
# =============================================================================
REPRODUCTION_TOLERANCES = OrderedDict((
    # Step5 (degrees Celsius / dimensionless z-score)
    ("current_lst_celsius", 1e-4),
    ("baseline_lst_mean_celsius", 1e-4),
    ("baseline_lst_std_celsius", 1e-4),
    ("current_minus_baseline_celsius", 1e-4),
    ("anomaly_zscore", 1e-4),
    ("baseline_valid_count", 0.0),
    # Step5C / Step7 (dimensionless TVDI, Celsius LST)
    ("current_tvdi", 1e-4),
    ("tvdi_difference", 1e-4),
    ("downscaled_lst_celsius", 1e-3),
    ("fused_lst_celsius", 1e-3),
))

#: Minimum valid-mask agreement required for a reproduced product.
REPRODUCTION_MIN_MASK_AGREEMENT = 0.9999

#: Step8 reproduction tolerances against the frozen canonical Step8 run.
REPRODUCTION_STEP8_METRIC_TOL = 1e-6
REPRODUCTION_STEP8_ROWCOUNT_EXACT = True


# =============================================================================
# Product registry
# =============================================================================
#: Descriptive "changed pixel" thresholds. These are reporting conveniences for
#: the changed-pixel fraction column ONLY; they never gate a scientific claim
#: and are never tuned after seeing results.
CHANGED_PIXEL_THRESHOLDS = OrderedDict((
    ("current_lst_celsius", 0.05),
    ("baseline_lst_mean_celsius", 0.05),
    ("baseline_lst_std_celsius", 0.05),
    ("current_minus_baseline_celsius", 0.05),
    ("anomaly_zscore", 0.01),
    ("current_tvdi", 0.005),
    ("tvdi_difference", 0.005),
    ("downscaled_lst_celsius", 0.05),
    ("fused_lst_celsius", 0.05),
))


def compared_raster_products() -> "OrderedDict[str, dict]":
    """Registry of the Step5/Step5C/Step7 rasters compared between chains.

    Each entry maps a product key to ``{stage_dir, filename, units, map}``
    where ``stage_dir`` is the ExperimentContext output-dir key.
    """
    return OrderedDict((
        ("current_lst_celsius", {
            "stage_dir": "step5_output_dir",
            "filename": "current_period_median_celsius.tif",
            "units": "celsius", "map": True,
        }),
        ("baseline_lst_mean_celsius", {
            "stage_dir": "step5_output_dir",
            "filename": "baseline_lst_mean_celsius.tif",
            "units": "celsius", "map": False,
        }),
        ("baseline_lst_std_celsius", {
            "stage_dir": "step5_output_dir",
            "filename": "baseline_lst_std_celsius.tif",
            "units": "celsius", "map": False,
        }),
        # Derived per chain from that chain's own Step5 outputs -- canonical
        # Step5 writes the two components and the z-score, not their raw
        # difference. See DERIVED_PRODUCTS / build_current_minus_baseline.
        ("current_minus_baseline_celsius", {
            "stage_dir": None,
            "filename": "current_minus_baseline_celsius.tif",
            "units": "celsius", "map": True,
        }),
        ("anomaly_zscore", {
            "stage_dir": "step5_output_dir",
            "filename": "anomaly_zscore.tif",
            "units": "zscore", "map": True,
        }),
        ("current_tvdi", {
            "stage_dir": "step5c_output_dir",
            "filename": "current_tvdi.tif",
            "units": "tvdi", "map": True,
        }),
        ("tvdi_difference", {
            "stage_dir": "step5c_output_dir",
            "filename": "tvdi_difference.tif",
            "units": "tvdi", "map": True,
        }),
        ("downscaled_lst_celsius", {
            "stage_dir": "step7d_output_dir",
            "filename": "downscaled_lst_celsius.tif",
            "units": "celsius", "map": True,
        }),
        ("fused_lst_celsius", {
            "stage_dir": "step7e_output_dir",
            "filename": "fused_lst_celsius.tif",
            "units": "celsius", "map": True,
        }),
    ))


#: `current_minus_baseline_celsius` is not a Step5 output file; canonical Step5
#: writes the z-score and the two components. It is derived per chain from the
#: chain's OWN Step5 outputs with the SAME Step5 policy the canonical pipeline
#: applies (see build_current_minus_baseline).
DERIVED_PRODUCTS = ("current_minus_baseline_celsius",)
DERIVED_SUBDIR = "derived"


def product_path(ctx: dict, product: str, root_for_derived: Path | None = None) -> Path:
    """Resolve one compared product's raster path for a chain context."""
    registry = compared_raster_products()
    if product not in registry:
        raise DownstreamABError(f"unknown compared product: {product!r}")
    if product in DERIVED_PRODUCTS:
        base = Path(root_for_derived) if root_for_derived is not None else Path(ctx["output_root"])
        return base / DERIVED_SUBDIR / "current_minus_baseline_celsius.tif"
    entry = registry[product]
    return Path(ctx[entry["stage_dir"]]) / entry["filename"]


#: Products carried into the boundary-propagation audit. `current_lst_celsius`
#: anchors the "key Step5 seam reduction remains supported" check; the rest are
#: the downstream propagation targets required by the experiment design.
BOUNDARY_PROPAGATION_PRODUCTS = (
    "current_lst_celsius",
    "current_minus_baseline_celsius",
    "anomaly_zscore",
    "current_tvdi",
    "tvdi_difference",
    "downscaled_lst_celsius",
    "fused_lst_celsius",
)

#: The Step5 product whose seam reduction must remain supported for the
#: strongest final status.
KEY_STEP5_SEAM_PRODUCT = "current_lst_celsius"

#: The boundary type that carries the predeclared seam evidence (identical to
#: the source counterfactual's decisive boundary).
KEY_BOUNDARY_TYPE = "scene_count_edge"

#: Boundary types reported for every product. `export_tile_boundary` is a
#: negative control and can never create positive evidence.
BOUNDARY_TYPES = (
    "scene_count_edge",
    "unique_date_count_edge",
    "same_day_multiplicity_edge",
    "export_tile_boundary",
    "source_scene_path_row",
)


# =============================================================================
# Namespace resolution and safety
# =============================================================================
def diagnostic_output_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """The ONE directory this experiment may write beneath."""
    return Path(base_dir) / "outputs" / "diagnostics" / DIAGNOSTIC_NAMESPACE / experiment_id


def counterfactual_source_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen source counterfactual audit root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "diagnostics" / SOURCE_AUDIT_NAMESPACE / experiment_id


def canonical_experiment_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen canonical experiment root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "experiments" / experiment_id


def forbidden_write_roots(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> list[Path]:
    """Roots that must never be written, overwritten, or deleted."""
    return [
        canonical_experiment_root(experiment_id, base_dir),
        counterfactual_source_root(experiment_id, base_dir),
        Path(base_dir) / "data",
        Path(base_dir) / "outputs" / "step5",
        Path(base_dir) / "outputs" / "step5c",
        Path(base_dir) / "outputs" / "step3",
        Path(base_dir) / "outputs" / "cross_region",
        Path(base_dir) / "outputs" / "robustness",
    ]


def assert_downstream_namespace_safe(
    paths, experiment_id: str, base_dir: Path = PROJECT_ROOT,
) -> None:
    """Every supplied write path must resolve strictly under the A/B root.

    Raises :class:`NamespaceSafetyError` for anything outside it, and
    explicitly for anything under a frozen canonical/counterfactual root.
    """
    root = diagnostic_output_root(experiment_id, base_dir).resolve()
    forbidden = [p.resolve() for p in forbidden_write_roots(experiment_id, base_dir)]

    for raw in paths:
        candidate = Path(raw).resolve()
        for bad in forbidden:
            if candidate == bad or bad in candidate.parents:
                raise NamespaceSafetyError(
                    f"refusing to write inside a frozen/canonical namespace: {candidate} "
                    f"(forbidden root: {bad})"
                )
        if candidate != root and root not in candidate.parents:
            raise NamespaceSafetyError(
                f"refusing to write outside the dedicated A/B diagnostic root: "
                f"{candidate} (allowed root: {root})"
            )


def clear_diagnostic_namespace(
    experiment_id: str, base_dir: Path = PROJECT_ROOT,
) -> str | None:
    """`--force` deletion of ONLY the dedicated A/B diagnostic namespace.

    The path is namespace-checked before a single byte is removed; the frozen
    experiment and counterfactual roots can never be reached from here.
    """
    root = diagnostic_output_root(experiment_id, base_dir)
    if not root.exists():
        return None
    resolved = root.resolve()
    # Belt and braces: the resolved path must still be the A/B root itself.
    assert_downstream_namespace_safe([resolved], experiment_id, base_dir)
    expected = diagnostic_output_root(experiment_id, base_dir).resolve()
    if resolved != expected:
        raise NamespaceSafetyError(
            f"refusing to delete {resolved}: it is not the dedicated A/B root {expected}"
        )
    if DIAGNOSTIC_NAMESPACE not in resolved.parts or experiment_id not in resolved.parts:
        raise NamespaceSafetyError(
            f"refusing to delete a path that is not namespaced to this experiment: {resolved}"
        )
    shutil.rmtree(resolved)
    return str(resolved)


def plan_output_layout(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, Path]":
    """The full planned directory layout (informational; creates nothing)."""
    root = diagnostic_output_root(experiment_id, base_dir)
    layout: "OrderedDict[str, Path]" = OrderedDict()
    layout["root"] = root
    layout["config"] = root / "config"
    layout["inputs"] = root / "inputs"
    layout["inputs_shared"] = root / "inputs" / "shared"
    for chain in CHAINS:
        layout[f"inputs_{chain}"] = root / "inputs" / chain
    for chain, side in CHAIN_SIDE.items():
        layout[side] = root / side
        for stage in ("step5", "step5c", "step7a", "step7b", "step7c", "step7d",
                      "step7e", "step8"):
            layout[f"{side}_{stage}"] = root / side / stage
        layout[f"{side}_derived"] = root / side / DERIVED_SUBDIR
    layout["comparison"] = root / "comparison"
    layout["comparison_maps"] = root / "comparison" / "maps"
    layout["comparison_tables"] = root / "comparison" / "tables"
    layout["checkpoints"] = root / "checkpoints"
    return layout


def plan_expected_files(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, Path]":
    """Report/table artefacts the live run is expected to produce."""
    root = diagnostic_output_root(experiment_id, base_dir)
    return OrderedDict((
        ("downstream_ab_summary.json", root / "downstream_ab_summary.json"),
        ("downstream_ab_summary.md", root / "downstream_ab_summary.md"),
        ("downstream_ab_manifest.json", root / "downstream_ab_manifest.json"),
        ("input_provenance.json", root / "input_provenance.json"),
        ("reference_reproduction.json", root / "reference_reproduction.json"),
        ("population_alignment.json", root / "population_alignment.json"),
        ("fold_assignment.csv", root / "fold_assignment.csv"),
        ("raster_change_summary.csv", root / "comparison" / "tables" / "raster_change_summary.csv"),
        ("boundary_propagation.csv", root / "comparison" / "tables" / "boundary_propagation.csv"),
        ("step8_metrics.csv", root / "comparison" / "tables" / "step8_metrics.csv"),
        ("step8_paired_bootstrap.csv", root / "comparison" / "tables" / "step8_paired_bootstrap.csv"),
        ("oof_predictions.csv", root / "comparison" / "oof_predictions.csv"),
    ))


# =============================================================================
# Source counterfactual prerequisites
# =============================================================================
def load_source_audit_state(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Read the frozen counterfactual audit's status documents (read-only).

    Returns a dict describing the manifest/summary/reproduction paths, the
    reported statuses, the report schema version, and the audit-file hashes.
    Never raises for a *failing* status -- validation is a separate step so the
    failure can be reported precisely.
    """
    root = counterfactual_source_root(experiment_id, base_dir)
    manifest_path = root / "manifest.json"
    summary_path = root / "counterfactual_summary.json"
    reproduction_path = root / "canonical_reproduction.json"

    state: dict = {
        "source_audit": SOURCE_AUDIT_NAMESPACE,
        "experiment_id": experiment_id,
        "source_root": str(root),
        "source_manifest_path": str(manifest_path),
        "source_summary_path": str(summary_path),
        "source_canonical_reproduction_path": str(reproduction_path),
        "present": root.exists(),
        "final_status": None,
        "canonical_reproduction_status": None,
        "canonical_gate_version": None,
        "report_schema_version": None,
        "audit_file_hashes": OrderedDict(),
        "missing_files": [],
    }

    for label, path in (
        ("manifest.json", manifest_path),
        ("counterfactual_summary.json", summary_path),
        ("canonical_reproduction.json", reproduction_path),
    ):
        if not path.exists():
            state["missing_files"].append(str(path))
            continue
        state["audit_file_hashes"][label] = audit.sha256_and_size(path)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        state["final_status"] = summary.get("final_status")
        state["final_status_rule"] = summary.get("final_status_rule")
        reproduction = summary.get("canonical_reproduction") or {}
        state["canonical_reproduction_status"] = reproduction.get("status")
        state["canonical_gate_version"] = reproduction.get("canonical_gate_version")
        limitations = summary.get("limitations")
        if isinstance(limitations, dict):
            state["report_schema_version"] = limitations.get("report_schema_version")
        state["report_schema_version"] = (
            state["report_schema_version"] or audit.REPORT_SCHEMA_VERSION
        )
    if reproduction_path.exists() and state["canonical_reproduction_status"] is None:
        reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
        state["canonical_reproduction_status"] = reproduction.get("status")
        state["canonical_gate_version"] = reproduction.get("canonical_gate_version")

    state["prerequisites_met"] = bool(
        state["present"]
        and not state["missing_files"]
        and state["final_status"] == REQUIRED_SOURCE_FINAL_STATUS
        and state["canonical_reproduction_status"] == REQUIRED_SOURCE_CANONICAL_REPRODUCTION
    )
    return state


def validate_source_audit_state(state: dict) -> None:
    """Fail loudly unless the source counterfactual satisfies both gates.

    A missing or failing source audit must NEVER be silently repaired by
    re-exporting from Earth Engine -- this runner has no such code path.
    """
    if not state.get("present"):
        raise PrerequisiteError(
            f"source counterfactual audit not found: {state.get('source_root')}. "
            "This runner never falls back to a GEE export; run the counterfactual "
            "audit first."
        )
    if state.get("missing_files"):
        raise PrerequisiteError(
            "source counterfactual audit is incomplete; missing: "
            f"{state['missing_files']}"
        )
    if state.get("final_status") != REQUIRED_SOURCE_FINAL_STATUS:
        raise PrerequisiteError(
            "source counterfactual final_status must be "
            f"{REQUIRED_SOURCE_FINAL_STATUS!r}, got {state.get('final_status')!r}."
        )
    if state.get("canonical_reproduction_status") != REQUIRED_SOURCE_CANONICAL_REPRODUCTION:
        raise PrerequisiteError(
            "source counterfactual canonical reproduction must be "
            f"{REQUIRED_SOURCE_CANONICAL_REPRODUCTION!r}, got "
            f"{state.get('canonical_reproduction_status')!r}."
        )


# =============================================================================
# Raw input plan (reference vs candidate vs shared)
# =============================================================================
#: Under date-balanced compositing the reducer-consistent observation support is
#: the UNIQUE-DATE count, not the raw scene count: the composite median is taken
#: over one image per acquisition date. The candidate current-period raster
#: therefore carries `current_lst_unique_date_valid_count` in band 2, which is
#: what Step5's STEP5_MIN_CURRENT_VALID_COUNT guard then evaluates. The
#: reference carries the canonical scene count (band 2 of the frozen canonical
#: current-period export, byte-identical to the audit's
#: `current_lst_scene_valid_count`). This is a DECLARED design decision, not an
#: incidental one, and its population consequence is reported in
#: population_alignment.json.
CANDIDATE_CURRENT_COUNT_ROLE = "current_lst_unique_date_valid_count"
REFERENCE_CURRENT_COUNT_SEMANTICS = "scene_valid_count"
CANDIDATE_CURRENT_COUNT_SEMANTICS = "unique_date_valid_count"


def build_input_plan(ctx: dict, experiment_id: str, base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, dict]":
    """Logical role -> {source paths per chain, materialized path, shared flag}.

    Only the current-period and annual-baseline Landsat **LST** roles differ
    between chains. Everything else (NDVI, MODIS, DEM, slope, land cover,
    MCD64A1 labels) is one shared materialized copy referenced by both chains.
    """
    root = diagnostic_output_root(experiment_id, base_dir)
    cf = counterfactual_source_root(experiment_id, base_dir)
    canonical = canonical_experiment_root(experiment_id, base_dir)
    inputs = root / "inputs"

    baseline_years = list(ctx["baseline_years"])
    current_days = ctx["current_period_days"]
    prefix = ctx["landsat_file_prefix"]

    plan: "OrderedDict[str, dict]" = OrderedDict()

    # --- current-period Landsat LST (differs) --------------------------------
    current_name = f"landsat_current_period_{current_days}days.tif"
    plan["current_lst"] = {
        "role": "current_lst",
        "family": "landsat_lst",
        "shared": False,
        "differs_between_chains": True,
        "reference_source": canonical / "data" / "current_period" / current_name,
        "candidate_source": cf / "rasters" / "current_lst_date_balanced_median.tif",
        "candidate_count_source": cf / "rasters" / f"{CANDIDATE_CURRENT_COUNT_ROLE}.tif",
        "materialized": OrderedDict((
            (CHAIN_REFERENCE, inputs / CHAIN_REFERENCE / "current_period" / current_name),
            (CHAIN_CANDIDATE, inputs / CHAIN_CANDIDATE / "current_period" / current_name),
        )),
        "materialization": "reference=verbatim_copy; candidate=2-band compose "
                          "(band1 date-balanced Celsius median, band2 "
                          f"{CANDIDATE_CURRENT_COUNT_SEMANTICS})",
    }

    # --- annual baseline Landsat LST (differs) -------------------------------
    canonical_baseline = sorted(
        (canonical / "data" / "landsat_timeseries").glob(f"{prefix}_baseline_*.tif")
    )
    baseline_by_year = {}
    for path in canonical_baseline:
        for year in baseline_years:
            if f"_baseline_{year}-" in path.name:
                baseline_by_year[year] = path
    for year in baseline_years:
        source = baseline_by_year.get(year)
        name = source.name if source is not None else f"{prefix}_baseline_{year}.tif"
        plan[f"baseline_lst_{year}"] = {
            "role": f"baseline_lst_{year}",
            "family": "landsat_lst",
            "shared": False,
            "differs_between_chains": True,
            "reference_source": source,
            "candidate_source": cf / "rasters" / f"baseline_lst_{year}_date_balanced_median.tif",
            "materialized": OrderedDict((
                (CHAIN_REFERENCE, inputs / CHAIN_REFERENCE / "landsat_timeseries" / name),
                (CHAIN_CANDIDATE, inputs / CHAIN_CANDIDATE / "landsat_timeseries" / name),
            )),
            "materialization": "verbatim_copy",
        }

    # --- shared inputs (identical for both chains) ---------------------------
    shared_specs = [
        ("ndvi_current", canonical / "data" / "ndvi_current_period" / "current_ndvi_median.tif",
         inputs / "shared" / "ndvi_current_period" / "current_ndvi_median.tif", "ndvi"),
        ("modis_lst_mean", canonical / "data" / "modis" / "modis_lst_mean_celsius.tif",
         inputs / "shared" / "modis" / "modis_lst_mean_celsius.tif", "modis"),
        ("modis_lst_std", canonical / "data" / "modis" / "modis_lst_std_celsius.tif",
         inputs / "shared" / "modis" / "modis_lst_std_celsius.tif", "modis"),
        ("dem_elevation", canonical / "data" / "dem" / "elevation.tif",
         inputs / "shared" / "dem" / "elevation.tif", "dem"),
        ("dem_slope", canonical / "data" / "dem" / "slope.tif",
         inputs / "shared" / "dem" / "slope.tif", "slope"),
        ("landcover_aligned",
         canonical / "gate_inputs" / "landcover_esa_worldcover_v200_aligned_to_reference.tif",
         inputs / "shared" / "gate_inputs" / "landcover_esa_worldcover_v200_aligned_to_reference.tif",
         "landcover"),
        ("mcd64a1_raw_burndate", canonical / "validation" / "labels" / "mcd64a1_raw.tif",
         inputs / "shared" / "labels" / "mcd64a1_raw.tif", "label"),
        ("mcd64a1_burned", canonical / "validation" / "labels" / "mcd64a1_burned.tif",
         inputs / "shared" / "labels" / "mcd64a1_burned.tif", "label"),
    ]
    ndvi_baseline_dir = canonical / "data" / "ndvi_timeseries"
    for year in baseline_years:
        matches = sorted(ndvi_baseline_dir.glob(f"ndvi_baseline_{year}-*.tif"))
        src = matches[0] if matches else ndvi_baseline_dir / f"ndvi_baseline_{year}.tif"
        shared_specs.append((
            f"ndvi_baseline_{year}", src,
            inputs / "shared" / "ndvi_timeseries" / src.name, "ndvi",
        ))

    for role, source, materialized, family in shared_specs:
        plan[role] = {
            "role": role,
            "family": family,
            "shared": True,
            "differs_between_chains": False,
            "reference_source": source,
            "candidate_source": source,
            "materialized": OrderedDict((
                (CHAIN_REFERENCE, materialized),
                (CHAIN_CANDIDATE, materialized),
            )),
            "materialization": "verbatim_copy_shared_by_both_chains",
        }

    return plan


def missing_plan_sources(plan: "OrderedDict[str, dict]") -> list[str]:
    """Every source file every chain needs, that does not exist."""
    missing: list[str] = []
    for entry in plan.values():
        for key in ("reference_source", "candidate_source", "candidate_count_source"):
            source = entry.get(key)
            if source is None:
                if key in ("reference_source", "candidate_source"):
                    missing.append(f"{entry['role']}:{key}=<unresolved>")
                continue
            if not Path(source).exists():
                missing.append(f"{entry['role']}:{key}={source}")
    return missing


def assert_required_frozen_inputs(plan: "OrderedDict[str, dict]", experiment_id: str) -> None:
    """The runner must fail clearly for an experiment lacking frozen inputs."""
    missing = missing_plan_sources(plan)
    if missing:
        raise PrerequisiteError(
            f"experiment {experiment_id!r} is missing required frozen inputs for the "
            "downstream A/B experiment; this runner never falls back to an Earth "
            f"Engine export. Missing:\n  " + "\n  ".join(missing)
        )


# =============================================================================
# Grid gate + provenance records
# =============================================================================
def raster_signature(path: Path) -> dict:
    """Grid + dtype/nodata signature of one raster (read-only)."""
    import rasterio

    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "transform": [float(v) for v in tuple(src.transform)[:6]],
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "dtype": str(src.dtypes[0]),
            "nodata": None if src.nodata is None else float(src.nodata),
        }


def grids_equal(a: dict, b: dict, *, atol: float = 1e-9) -> bool:
    """Exact grid equality (CRS, transform, width, height)."""
    if a["crs"] != b["crs"]:
        return False
    if int(a["width"]) != int(b["width"]) or int(a["height"]) != int(b["height"]):
        return False
    return all(abs(x - y) <= atol for x, y in zip(a["transform"], b["transform"]))


def assert_reference_candidate_grid_equality(plan: "OrderedDict[str, dict]") -> "OrderedDict[str, dict]":
    """HARD GATE: corresponding reference/candidate raw LST products must share
    the exact same grid. Any mismatch aborts the experiment.
    """
    checked: "OrderedDict[str, dict]" = OrderedDict()
    for role, entry in plan.items():
        if not entry.get("differs_between_chains"):
            continue
        ref_sig = raster_signature(Path(entry["reference_source"]))
        cand_sig = raster_signature(Path(entry["candidate_source"]))
        equal = grids_equal(ref_sig, cand_sig)
        checked[role] = {
            "reference": ref_sig, "candidate": cand_sig, "grid_equal": equal,
        }
        if not equal:
            raise GridMismatchError(
                f"raw LST grid mismatch for role {role!r}: "
                f"reference={ref_sig} candidate={cand_sig}"
            )
    return checked


def provenance_record(
    *, role: str, chain: str, source: Path, materialized: Path,
    shared: bool, family: str, materialization: str,
) -> dict:
    """One `input_provenance.json` row for one role in one chain."""
    materialized = Path(materialized)
    record = OrderedDict((
        ("logical_role", role),
        ("family", family),
        ("source_chain", chain),
        ("shared_between_chains", bool(shared)),
        ("source_path", str(source)),
        ("materialized_path", str(materialized)),
        ("materialization", materialization),
    ))
    if materialized.exists():
        signed = audit.sha256_and_size(materialized)
        record["file_size_bytes"] = signed["bytes"]
        record["sha256"] = signed["sha256"]
        sig = raster_signature(materialized)
        record.update(OrderedDict((
            ("crs", sig["crs"]),
            ("transform", sig["transform"]),
            ("width", sig["width"]),
            ("height", sig["height"]),
            ("band_count", sig["count"]),
            ("dtype", sig["dtype"]),
            ("nodata", sig["nodata"]),
        )))
    else:
        record["file_size_bytes"] = None
        record["sha256"] = None
    return record


# =============================================================================
# Input materialization
# =============================================================================
def _copy_safe(source: Path, destination: Path) -> None:
    """Copy a frozen source into the A/B namespace, atomically.

    The SOURCE is only ever read. The destination is written via a temp file and
    os.replace so an interrupted copy can never leave a half-written input that
    a later --resume would trust.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.parent / f".{destination.name}.tmp"
    shutil.copy2(str(source), str(tmp))
    os.replace(str(tmp), str(destination))


def compose_candidate_current_period(
    lst_source: Path, count_source: Path, destination: Path,
) -> dict:
    """Build the candidate 2-band current-period raster.

    Band 1 = date-balanced Celsius median, band 2 = the reducer-consistent
    observation support (unique acquisition dates). The canonical current-period
    export has exactly this layout, so Step5 reads the candidate through the
    SAME code path with no special-casing.
    """
    import numpy as np
    import rasterio

    assert_same_grid([lst_source, count_source])

    with rasterio.open(lst_source) as lst_src, rasterio.open(count_source) as cnt_src:
        profile = lst_src.profile.copy()
        lst = lst_src.read(1, masked=True).astype("float32")
        cnt = cnt_src.read(1, masked=True).astype("float32")
        lst_nodata = lst_src.nodata

    fill = float(lst_nodata) if lst_nodata is not None else audit.NODATA_SENTINEL
    profile.update(count=2, dtype="float32", nodata=fill, compress="lzw", BIGTIFF="IF_SAFER")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.parent / f".{destination.name}.tmp"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(lst.filled(fill).astype("float32"), 1)
        dst.write(cnt.filled(fill).astype("float32"), 2)
    os.replace(str(tmp), str(destination))

    return {
        "band_1": "date_balanced_lst_celsius_median",
        "band_2": CANDIDATE_CURRENT_COUNT_SEMANTICS,
        "nodata": fill,
        "valid_lst_pixels": int(np.sum(~np.ma.getmaskarray(lst))),
    }


def materialize_inputs(
    plan: "OrderedDict[str, dict]", experiment_id: str, base_dir: Path = PROJECT_ROOT,
) -> dict:
    """Build both isolated raw input bundles and return the provenance payload.

    Every destination is namespace-checked BEFORE it is written; frozen sources
    are opened read-only and are never modified.
    """
    destinations = []
    for entry in plan.values():
        destinations.extend(entry["materialized"].values())
    assert_downstream_namespace_safe(destinations, experiment_id, base_dir)

    grid_gate = assert_reference_candidate_grid_equality(plan)

    compose_notes: "OrderedDict[str, dict]" = OrderedDict()
    for role, entry in plan.items():
        if entry["shared"]:
            destination = entry["materialized"][CHAIN_REFERENCE]
            _copy_safe(Path(entry["reference_source"]), destination)
            continue
        _copy_safe(Path(entry["reference_source"]), entry["materialized"][CHAIN_REFERENCE])
        if role == "current_lst":
            compose_notes[role] = compose_candidate_current_period(
                Path(entry["candidate_source"]),
                Path(entry["candidate_count_source"]),
                entry["materialized"][CHAIN_CANDIDATE],
            )
        else:
            _copy_safe(Path(entry["candidate_source"]), entry["materialized"][CHAIN_CANDIDATE])

    return build_input_provenance(
        plan, experiment_id, grid_gate=grid_gate, compose_notes=compose_notes,
        base_dir=base_dir,
    )


def build_input_provenance(
    plan: "OrderedDict[str, dict]", experiment_id: str, *,
    grid_gate: dict, compose_notes: dict, base_dir: Path = PROJECT_ROOT,
    source_audit_state: dict | None = None,
) -> dict:
    """Assemble `input_provenance.json` for both chains."""
    records: list[dict] = []
    for role, entry in plan.items():
        for chain in CHAINS:
            source = entry["reference_source"] if chain == CHAIN_REFERENCE else entry["candidate_source"]
            records.append(provenance_record(
                role=role, chain=chain, source=Path(source),
                materialized=entry["materialized"][chain],
                shared=entry["shared"], family=entry["family"],
                materialization=entry["materialization"],
            ))

    state = source_audit_state or load_source_audit_state(experiment_id, base_dir)
    detection = modis_compatibility_required(experiment_id, base_dir)
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", CHAIN_CANDIDATE),
        ("inputs", records),
        ("legacy_modis_compatibility", OrderedDict((
            ("historical_compatibility_required", detection["required"]),
            ("mode", detection["mode_if_required"] if detection["required"]
                     else detection["mode_otherwise"]),
            ("zero_fill_guard_threshold", detection["zero_fill_guard_threshold"]),
            ("frozen_modis_evidence", detection["rasters"]),
            ("attestation_declaration_path",
             str(legacy_modis_attestation_config_path(base_dir))),
            ("warning", legacy_modis_compatibility_warning()
                        if detection["required"] else None),
            ("limitations", legacy_modis_compatibility_limitations()),
        ))),
        ("raw_lst_grid_equality_gate", OrderedDict((
            ("required", "exact grid equality between corresponding reference and "
                         "candidate raw LST products"),
            ("per_role", grid_gate),
            ("passed", all(v["grid_equal"] for v in grid_gate.values())),
        ))),
        ("candidate_current_period_composition", compose_notes),
        ("observation_support_semantics", OrderedDict((
            ("reference", REFERENCE_CURRENT_COUNT_SEMANTICS),
            ("candidate", CANDIDATE_CURRENT_COUNT_SEMANTICS),
            ("declared_before_run", True),
            ("rationale",
             "Under date-balanced compositing the median is taken over one image "
             "per acquisition date, so the reducer-consistent observation support "
             "is the unique-date count. Using the raw scene count would overstate "
             "the candidate's independent observations at the "
             "STEP5_MIN_CURRENT_VALID_COUNT guard."),
        ))),
        ("candidate_audit_provenance", OrderedDict((
            ("source_counterfactual_manifest_path", state.get("source_manifest_path")),
            ("source_counterfactual_summary_path", state.get("source_summary_path")),
            ("source_final_status", state.get("final_status")),
            ("source_canonical_reproduction_status", state.get("canonical_reproduction_status")),
            ("source_canonical_gate_version", state.get("canonical_gate_version")),
            ("report_schema_version", state.get("report_schema_version")),
            ("audit_file_hashes", state.get("audit_file_hashes")),
            ("required_final_status", REQUIRED_SOURCE_FINAL_STATUS),
            ("required_canonical_reproduction", REQUIRED_SOURCE_CANONICAL_REPRODUCTION),
            ("prerequisites_met", state.get("prerequisites_met")),
        ))),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def ndvi_inputs_identical(provenance: dict) -> bool:
    """Canonical NDVI must be the SAME materialized file for both chains."""
    by_role: dict[str, set] = {}
    for record in provenance["inputs"]:
        if record["family"] != "ndvi":
            continue
        by_role.setdefault(record["logical_role"], set()).add(record["materialized_path"])
    return bool(by_role) and all(len(paths) == 1 for paths in by_role.values())


def candidate_modifies_lst_only(provenance: dict) -> bool:
    """Only `landsat_lst` roles may differ between the two chains."""
    differing = {
        r["logical_role"] for r in provenance["inputs"]
        if not r["shared_between_chains"]
    }
    families = {
        r["family"] for r in provenance["inputs"] if r["logical_role"] in differing
    }
    return families == {"landsat_lst"} or not differing


# =============================================================================
# Legacy frozen-MODIS historical-compatibility attestation
#
# WHY THIS EXISTS
# ---------------
# The frozen canonical Manavgat Step7 run consumed a MODIS mean/std pair that
# declares NO nodata and encodes sea / no-observation cells as exact 0.0. The
# CURRENT Step7B guard correctly rejects that signature. Re-exporting MODIS,
# masking zero locally, or disabling the guard globally would each change the
# frozen reference or add a SECOND intervention on top of the Landsat one, and
# would destroy the LST-only A/B contract.
#
# So this module reproduces -- for this diagnostic A/B experiment ONLY, for
# manavgat_2021 ONLY, and only behind an exact path+hash attestation -- the
# historical Step7 behaviour. It does NOT declare zero a valid MODIS LST value,
# it rewrites nothing, and it leaves the default Step7B guard untouched.
# =============================================================================
#: Internal mode name. Must match `step7b.LEGACY_FROZEN_MODIS_COMPATIBILITY_MODE`.
LEGACY_MODIS_COMPATIBILITY_MODE = "legacy_frozen_modis_compatibility"

#: The mode every other caller (including the Step7B CLI) keeps.
MODIS_STRICT_MODE = "strict_default_guard"

#: The compatibility path is reachable for this experiment and no other.
LEGACY_MODIS_COMPATIBILITY_EXPERIMENT_IDS = ("manavgat_2021",)

LEGACY_MODIS_ATTESTATION_SCHEMA_VERSION = "1.0-legacy-frozen-modis"

#: Repo-committed declaration, derived from the frozen inputs by
#: `write_legacy_modis_attestation_declaration` -- never hand-written.
LEGACY_MODIS_ATTESTATION_CONFIG_RELPATH = (
    "config/legacy_modis_compatibility_attestation.json"
)

#: Step7B feature name -> the frozen canonical file and its provenance role.
MODIS_COMPATIBILITY_RASTERS = OrderedDict((
    ("modis_lst_mean_celsius", OrderedDict((
        ("filename", "modis_lst_mean_celsius.tif"),
        ("provenance_role", "modis_lst_mean"),
    ))),
    ("modis_lst_std_celsius", OrderedDict((
        ("filename", "modis_lst_std_celsius.tif"),
        ("provenance_role", "modis_lst_std"),
    ))),
))

#: Technical failure recorded when the shared-MODIS gates do not hold. It is a
#: dedicated field, NOT a new scientific final status: `FINAL_STATUSES` is
#: unchanged and such a run terminates as `baseline_invariance_failed`.
TECHNICAL_FAILURE_SHARED_MODIS = "shared_modis_invariance_failed"


class LegacyModisCompatibilityError(DownstreamABError):
    """The historical-MODIS compatibility gate refused to authorize the run."""


def legacy_modis_attestation_config_path(base_dir: Path = PROJECT_ROOT) -> Path:
    return Path(base_dir) / LEGACY_MODIS_ATTESTATION_CONFIG_RELPATH


def frozen_modis_dir(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """The frozen canonical MODIS namespace (READ-ONLY)."""
    return canonical_experiment_root(experiment_id, base_dir) / "data" / "modis"


def modis_zero_fill_threshold() -> float:
    """The production Step7B threshold; never redefined here."""
    from core.config import STEP7B_MODIS_SUSPICIOUS_ZERO_FRACTION

    return float(STEP7B_MODIS_SUSPICIOUS_ZERO_FRACTION)


def describe_modis_raster(path: Path) -> dict:
    """Full read-only evidence record for one MODIS raster.

    Nothing is written, nothing is masked and no value is altered: the file is
    opened read-only and closed again.
    """
    import numpy as np
    import rasterio

    path = Path(path)
    signed = audit.sha256_and_size(path)
    with rasterio.open(path) as src:
        nodata = src.nodata
        array = src.read(1)
        record = OrderedDict((
            ("path", str(path.resolve())),
            ("sha256", signed["sha256"]),
            ("bytes", signed["bytes"]),
            ("crs", str(src.crs)),
            ("transform", [float(v) for v in tuple(src.transform)[:6]]),
            ("shape_hw", [int(src.height), int(src.width)]),
            ("dtype", str(src.dtypes[0])),
            ("nodata", None if nodata is None else float(nodata)),
        ))

    finite = np.isfinite(array)
    total = int(array.size)
    zero_count = int(np.count_nonzero(array[finite] == 0.0)) if total else 0
    record.update(OrderedDict((
        ("total_pixel_count", total),
        ("finite_count", int(finite.sum())),
        ("exact_zero_count", zero_count),
        ("exact_zero_fraction", (zero_count / total) if total else 0.0),
        ("min", float(array[finite].min()) if finite.any() else None),
        ("max", float(array[finite].max()) if finite.any() else None),
        ("modified_at", datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc).isoformat()),
    )))
    return record


def zero_fill_guard_would_reject(record: dict) -> bool:
    """Does the DEFAULT Step7B guard reject this raster? (pure predicate)"""
    return (
        record.get("nodata") is None
        and float(record.get("exact_zero_fraction") or 0.0) > modis_zero_fill_threshold()
    )


def frozen_step7b_historical_evidence(
    experiment_id: str, base_dir: Path = PROJECT_ROOT,
) -> dict:
    """Historical MODIS evidence read from the FROZEN canonical Step7B run.

    Read-only: `downscaling_dataset_stats.json` and the two frozen aligned MODIS
    rasters are inspected, never rewritten.
    """
    step7b_root = canonical_experiment_root(experiment_id, base_dir) / "step7b"
    stats_path = step7b_root / "downscaling_dataset_stats.json"
    evidence = OrderedDict((
        ("stats_path", str(stats_path)),
        ("present", stats_path.exists()),
        ("canonical_step7b_created_at", None),
        ("rasters", OrderedDict()),
        ("aligned_inputs", OrderedDict()),
    ))
    if not stats_path.exists():
        return evidence

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    evidence["canonical_step7b_created_at"] = stats.get("created_at")
    diagnostics = {
        d.get("name"): d for d in (stats.get("alignment_diagnostics") or [])
    }
    for feature in MODIS_COMPATIBILITY_RASTERS:
        diag = diagnostics.get(feature)
        if diag is None:
            continue
        evidence["rasters"][feature] = OrderedDict((
            ("source_nodata", diag.get("source_nodata")),
            ("source_valid_pixel_count", diag.get("source_valid_pixel_count")),
            ("source_total_pixel_count", diag.get("source_total_pixel_count")),
            ("aligned_valid_pixel_count", diag.get("aligned_valid_pixel_count")),
            ("aligned_valid_fraction", diag.get("aligned_valid_fraction")),
            ("original_source_path", diag.get("source_path")),
            ("aligned_path", diag.get("aligned_path")),
            # Absent in the frozen run: it predates the guard, which is itself
            # part of the historical evidence.
            ("modis_source_validation_recorded", "modis_source_validation" in diag),
        ))
        aligned = step7b_root / "aligned_inputs" / f"{feature}.tif"
        if aligned.exists():
            signed = audit.sha256_and_size(aligned)
            evidence["aligned_inputs"][feature] = OrderedDict((
                ("path", str(aligned)),
                ("sha256", signed["sha256"]),
                ("bytes", signed["bytes"]),
            ))
    return evidence


def historical_evidence_confirms_no_nodata_source(evidence: dict) -> bool:
    """The frozen Step7B metadata must EXPLICITLY show the no-nodata source.

    That means, for BOTH MODIS rasters: `source_nodata` recorded as null AND
    every source pixel counted as valid (valid == total).
    """
    rasters = (evidence or {}).get("rasters") or {}
    if set(rasters) != set(MODIS_COMPATIBILITY_RASTERS):
        return False
    for record in rasters.values():
        if record.get("source_nodata") is not None:
            return False
        valid = record.get("source_valid_pixel_count")
        total = record.get("source_total_pixel_count")
        if valid is None or total is None or int(valid) != int(total) or int(total) <= 0:
            return False
    return True


def legacy_modis_compatibility_warning() -> dict:
    """The prominent machine-readable warning carried by every report."""
    return OrderedDict((
        ("code", "legacy_zero_filled_modis_compatibility"),
        ("statement",
         "This downstream A/B run reproduced the frozen historical MODIS "
         "representation: a mean/std pair that declares no nodata and encodes "
         "sea / no-observation cells as exact 0.0. The default Step7B zero-fill "
         "guard is unchanged and still rejects this signature for every other "
         "caller; it was waived only here, only for "
         f"{LEGACY_MODIS_COMPATIBILITY_EXPERIMENT_IDS[0]}, and only after an "
         "exact path and SHA-256 attestation against the frozen inputs."),
        ("scientific_effect",
         "The historical zero-filled MODIS representation was preserved "
         "identically in both chains to isolate the Landsat compositing "
         "intervention. This does not validate zero as physical MODIS LST and "
         "does not replace a future MODIS nodata repair experiment."),
    ))


def legacy_modis_compatibility_limitations() -> list[str]:
    """Required limitation statements tied to the compatibility path."""
    return [
        "The downstream A/B result is conditional on the frozen historical MODIS "
        "representation used by the original Manavgat Step7 run.",
        "A future MODIS nodata repair must be evaluated separately and may not be "
        "combined with the Landsat candidate decision.",
    ]


def build_legacy_modis_attestation_declaration(
    experiment_id: str, base_dir: Path = PROJECT_ROOT,
) -> dict:
    """Derive the repo-committed declaration from the FROZEN inputs.

    Every hash here is computed from the frozen file at derivation time -- no
    hash is ever hand-written. The runtime gate re-derives and re-checks all of
    them, so a stale declaration fails closed.
    """
    modis_dir = frozen_modis_dir(experiment_id, base_dir)
    rasters: "OrderedDict[str, dict]" = OrderedDict()
    for feature, spec in MODIS_COMPATIBILITY_RASTERS.items():
        path = modis_dir / spec["filename"]
        if not path.exists():
            raise LegacyModisCompatibilityError(
                f"frozen MODIS raster not found for {experiment_id!r}: {path}"
            )
        record = describe_modis_raster(path)
        record["provenance_role"] = spec["provenance_role"]
        record["default_guard_would_reject"] = zero_fill_guard_would_reject(record)
        rasters[feature] = record

    metadata_path = modis_dir / "modis_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else None
    )
    historical = frozen_step7b_historical_evidence(experiment_id, base_dir)

    return OrderedDict((
        ("schema_version", LEGACY_MODIS_ATTESTATION_SCHEMA_VERSION),
        ("mode", LEGACY_MODIS_COMPATIBILITY_MODE),
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("scope",
         "Diagnostic downstream A/B experiment only. This declaration authorizes "
         "NOTHING outside src/landsat_composite_downstream_ab.py, and the default "
         "Step7B guard is unchanged for every other caller."),
        ("frozen_namespace", str(modis_dir)),
        ("zero_fill_guard_threshold", modis_zero_fill_threshold()),
        ("rasters", rasters),
        ("frozen_modis_metadata", metadata),
        ("frozen_step7b_historical_evidence", historical),
        ("historical_step7b_evidence_confirmed",
         historical_evidence_confirms_no_nodata_source(historical)),
        ("declares_zero_scientifically_valid", False),
        ("warning", legacy_modis_compatibility_warning()),
        ("limitations", legacy_modis_compatibility_limitations()),
        ("derived_at", datetime.now(timezone.utc).isoformat()),
        ("derivation",
         "src.landsat_composite_downstream_ab."
         "build_legacy_modis_attestation_declaration"),
    ))


def write_legacy_modis_attestation_declaration(
    experiment_id: str, base_dir: Path = PROJECT_ROOT,
) -> Path:
    """Controlled derivation stage: write `config/...attestation.json`.

    This is the ONLY writer of that file and it writes nothing else. Frozen
    inputs are read-only throughout.
    """
    payload = build_legacy_modis_attestation_declaration(experiment_id, base_dir)
    return write_json_atomic(legacy_modis_attestation_config_path(base_dir), payload)


def load_legacy_modis_attestation_declaration(base_dir: Path = PROJECT_ROOT) -> dict:
    path = legacy_modis_attestation_config_path(base_dir)
    if not path.exists():
        raise LegacyModisCompatibilityError(
            f"the historical MODIS compatibility declaration is missing: {path}. "
            "Derive it from the frozen inputs with "
            "`write_legacy_modis_attestation_declaration(<experiment_id>)`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def modis_provenance_records(provenance: dict) -> "OrderedDict[str, list[dict]]":
    """Provenance rows for the two MODIS roles, keyed by Step7B feature name."""
    by_role: "OrderedDict[str, list[dict]]" = OrderedDict(
        (feature, []) for feature in MODIS_COMPATIBILITY_RASTERS
    )
    role_to_feature = {
        spec["provenance_role"]: feature
        for feature, spec in MODIS_COMPATIBILITY_RASTERS.items()
    }
    for record in (provenance or {}).get("inputs") or []:
        feature = role_to_feature.get(record.get("logical_role"))
        if feature is not None:
            by_role[feature].append(record)
    return by_role


def modis_compatibility_required(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Would the DEFAULT Step7B guard stop this experiment? (read-only)

    Used by the dry-run to report, without writing anything, whether the
    historical-compatibility path will be needed.
    """
    modis_dir = frozen_modis_dir(experiment_id, base_dir)
    rasters: "OrderedDict[str, dict]" = OrderedDict()
    required = False
    for feature, spec in MODIS_COMPATIBILITY_RASTERS.items():
        path = modis_dir / spec["filename"]
        if not path.exists():
            rasters[feature] = OrderedDict((("path", str(path)), ("present", False)))
            continue
        record = describe_modis_raster(path)
        record["present"] = True
        record["default_guard_would_reject"] = zero_fill_guard_would_reject(record)
        # Only the mean raster is subject to rule 1 in production Step7B.
        if feature == "modis_lst_mean_celsius" and record["default_guard_would_reject"]:
            required = True
        rasters[feature] = record
    return OrderedDict((
        ("required", required),
        ("mode_if_required", LEGACY_MODIS_COMPATIBILITY_MODE),
        ("mode_otherwise", MODIS_STRICT_MODE),
        ("zero_fill_guard_threshold", modis_zero_fill_threshold()),
        ("rasters", rasters),
    ))


def _iso_to_datetime(value):
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def validate_legacy_modis_compatibility(
    experiment_id: str,
    provenance: dict,
    chain_contexts: "OrderedDict[str, dict] | dict",
    base_dir: Path = PROJECT_ROOT,
    declaration: dict | None = None,
) -> dict:
    """The controlled compatibility stage. Runs BEFORE any Step7B call.

    Returns a runtime attestation when the compatibility path is authorized, or
    a `required=False` record when the default strict guard already suffices.
    Raises :class:`LegacyModisCompatibilityError` when compatibility is needed
    but any gate fails -- the experiment then stops with NO scientific status.

    Read-only: it hashes and describes frozen files and writes nothing.
    """
    detection = modis_compatibility_required(experiment_id, base_dir)
    chains = OrderedDict(
        (chain, MODIS_STRICT_MODE) for chain in (chain_contexts or CHAINS)
    )
    if not detection["required"]:
        return OrderedDict((
            ("experiment", DIAGNOSTIC_NAMESPACE),
            ("experiment_id", experiment_id),
            ("schema_version", LEGACY_MODIS_ATTESTATION_SCHEMA_VERSION),
            ("required", False),
            ("status", "not_required"),
            ("mode", MODIS_STRICT_MODE),
            ("reason", "the default Step7B zero-fill guard accepts the frozen "
                       "MODIS inputs; no compatibility path is used"),
            ("chains", chains),
            ("detection", detection),
            ("created_at", datetime.now(timezone.utc).isoformat()),
        ))

    declaration = declaration or load_legacy_modis_attestation_declaration(base_dir)
    modis_dir = frozen_modis_dir(experiment_id, base_dir).resolve()
    provenance_rows = modis_provenance_records(provenance)
    historical = frozen_step7b_historical_evidence(experiment_id, base_dir)
    provenance_created = _iso_to_datetime((provenance or {}).get("created_at"))

    checks: "OrderedDict[str, bool]" = OrderedDict()
    failures: list[str] = []

    def _require(name: str, ok: bool, message: str) -> bool:
        checks[name] = bool(ok)
        if not ok:
            failures.append(f"{name}: {message}")
        return bool(ok)

    _require(
        "experiment_id_is_authorized",
        experiment_id in LEGACY_MODIS_COMPATIBILITY_EXPERIMENT_IDS,
        f"{experiment_id!r} is not one of "
        f"{list(LEGACY_MODIS_COMPATIBILITY_EXPERIMENT_IDS)}",
    )
    _require(
        "declaration_matches_experiment",
        declaration.get("experiment_id") == experiment_id
        and declaration.get("mode") == LEGACY_MODIS_COMPATIBILITY_MODE
        and declaration.get("schema_version") == LEGACY_MODIS_ATTESTATION_SCHEMA_VERSION,
        "the declaration is bound to a different experiment, mode or schema "
        f"({declaration.get('experiment_id')!r}/{declaration.get('mode')!r}/"
        f"{declaration.get('schema_version')!r})",
    )

    rasters: "OrderedDict[str, dict]" = OrderedDict()
    declared = declaration.get("rasters") or {}
    for feature, spec in MODIS_COMPATIBILITY_RASTERS.items():
        source = modis_dir / spec["filename"]
        entry = OrderedDict((("feature", feature),))
        if not source.exists():
            _require(f"{feature}__frozen_source_present", False, f"missing {source}")
            rasters[feature] = entry
            continue

        current = describe_modis_raster(source)
        entry.update(current)
        entry["provenance_role"] = spec["provenance_role"]
        entry["default_guard_would_reject"] = zero_fill_guard_would_reject(current)

        _require(
            f"{feature}__inside_frozen_namespace",
            Path(current["path"]).parent == modis_dir,
            f"{current['path']} is not inside the frozen canonical namespace {modis_dir}",
        )
        declared_entry = declared.get(feature) or {}
        _require(
            f"{feature}__declared_path_exact",
            str(declared_entry.get("path") or "") == current["path"],
            f"declaration path {declared_entry.get('path')!r} != {current['path']!r}",
        )
        _require(
            f"{feature}__declared_hash_matches",
            declared_entry.get("sha256") == current["sha256"]
            and int(declared_entry.get("bytes", -1)) == current["bytes"],
            "the frozen file no longer matches the declared SHA-256/byte size",
        )

        rows = provenance_rows.get(feature) or []
        paths = {r.get("materialized_path") for r in rows}
        hashes = {r.get("sha256") for r in rows}
        _require(
            f"{feature}__shared_identically_by_both_chains",
            len(rows) == len(CHAINS)
            and all(r.get("shared_between_chains") for r in rows)
            and len(paths) == 1 and len(hashes) == 1,
            "reference and candidate do not reference ONE identical materialized "
            f"MODIS file (paths={paths}, hashes={hashes})",
        )
        _require(
            f"{feature}__provenance_hash_matches_frozen_source",
            bool(hashes) and hashes == {current["sha256"]},
            f"the A/B input-provenance hash {hashes} differs from the frozen "
            f"source hash {current['sha256']!r}",
        )

        materialized = next(iter(paths), None)
        authorized = [current["path"]]
        if materialized:
            materialized_path = Path(materialized)
            entry["materialized_path"] = str(materialized_path)
            if materialized_path.exists():
                materialized_now = audit.sha256_and_size(materialized_path)
                entry["materialized_sha256"] = materialized_now["sha256"]
                _require(
                    f"{feature}__materialized_copy_is_byte_identical",
                    materialized_now["sha256"] == current["sha256"],
                    "the materialized A/B copy is not byte-identical to the "
                    "frozen source",
                )
                authorized.append(str(materialized_path.resolve()))
            else:
                _require(
                    f"{feature}__materialized_copy_present", False,
                    f"the materialized copy {materialized_path} is missing",
                )
        entry["authorized_paths"] = authorized

        modified_at = _iso_to_datetime(current["modified_at"])
        _require(
            f"{feature}__not_modified_after_materialization",
            provenance_created is None or modified_at is None
            or modified_at <= provenance_created,
            f"the frozen source was modified at {current['modified_at']} which is "
            f"after the A/B materialization at {provenance.get('created_at')}",
        )
        rasters[feature] = entry

    _require(
        "historical_step7b_metadata_confirms_no_nodata_source",
        historical_evidence_confirms_no_nodata_source(historical),
        "the frozen Step7B metadata does not explicitly record a no-nodata MODIS "
        "source with every source pixel counted as valid",
    )
    _require(
        "reference_and_candidate_use_the_same_mode",
        len(set(chains.values())) == 1,
        f"the two chains disagree on the compatibility mode: {dict(chains)}",
    )

    if failures:
        raise LegacyModisCompatibilityError(
            "the historical MODIS compatibility attestation FAILED; the Landsat "
            "A/B experiment is refused and no scientific status is issued.\n  "
            + "\n  ".join(failures)
        )

    for chain in chains:
        chains[chain] = LEGACY_MODIS_COMPATIBILITY_MODE

    attestation = OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("schema_version", LEGACY_MODIS_ATTESTATION_SCHEMA_VERSION),
        ("required", True),
        ("status", "pass"),
        ("mode", LEGACY_MODIS_COMPATIBILITY_MODE),
        ("declaration_path", str(legacy_modis_attestation_config_path(base_dir))),
        ("declaration_sha256",
         audit.sha256_and_size(legacy_modis_attestation_config_path(base_dir))["sha256"]),
        ("rasters", rasters),
        ("frozen_step7b_historical_evidence", historical),
        ("historical_step7b_evidence_confirmed", True),
        ("gate_checks", checks),
        ("chains", chains),
        ("declares_zero_scientifically_valid", False),
        ("rasters_rewritten", False),
        ("nodata_assigned", False),
        ("zero_converted_to_nan", False),
        ("values_or_mask_changed", False),
        ("default_step7b_guard_changed", False),
        ("warning", legacy_modis_compatibility_warning()),
        ("limitations", legacy_modis_compatibility_limitations()),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))
    attestation["attestation_id"] = attestation_binding(attestation)["binding_sha256"]
    return attestation


def attestation_binding(attestation: dict) -> dict:
    """The hash binding a checkpoint stage to THIS attestation.

    Resume revalidates it, so a Step7B-and-later stage produced under a
    different MODIS attestation can never be silently reused.
    """
    import hashlib

    rasters = OrderedDict(
        (feature, (entry or {}).get("sha256"))
        for feature, entry in ((attestation or {}).get("rasters") or {}).items()
    )
    payload = OrderedDict((
        ("mode", (attestation or {}).get("mode", MODIS_STRICT_MODE)),
        ("required", bool((attestation or {}).get("required", False))),
        ("experiment_id", (attestation or {}).get("experiment_id")),
        ("raster_sha256", rasters),
    ))
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    payload["binding_sha256"] = digest
    return payload


def step7b_compatibility_attestation(attestation: dict):
    """Build the typed Step7B attestation, or None for the strict default.

    The Step7B callable re-verifies every path and hash itself; this only hands
    it the validated expectations.
    """
    import src.step7b_prepare_downscaling_dataset as step7b

    if not attestation or not attestation.get("required"):
        return None
    if attestation.get("status") != "pass":
        raise LegacyModisCompatibilityError(
            "refusing to build a Step7B compatibility attestation from a "
            f"non-passing gate result (status={attestation.get('status')!r})."
        )
    rasters = OrderedDict()
    for feature, entry in (attestation.get("rasters") or {}).items():
        rasters[feature] = {
            "sha256": entry.get("sha256"),
            "bytes": entry.get("bytes"),
            "authorized_paths": list(entry.get("authorized_paths") or []),
        }
    return step7b.LegacyModisCompatibilityAttestation.from_mapping({
        "mode": attestation["mode"],
        "experiment_id": attestation["experiment_id"],
        "rasters": rasters,
        "historical_step7b_evidence_confirmed":
            attestation.get("historical_step7b_evidence_confirmed") is True,
        "issued_by": DIAGNOSTIC_NAMESPACE,
        "attestation_id": attestation.get("attestation_id", ""),
        "notes": tuple(attestation.get("limitations") or ()),
    })


# =============================================================================
# Shared-MODIS baseline / chain invariance
# =============================================================================
def chain_step7b_compatibility_mode(ctx: dict) -> str | None:
    """The MODIS compatibility mode recorded by ONE chain's own Step7B run."""
    stats_path = Path(ctx["step7b_output_dir"]) / "downscaling_dataset_stats.json"
    if not stats_path.exists():
        return None
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    modes = set()
    for diag in stats.get("alignment_diagnostics") or []:
        validation = diag.get("modis_source_validation") or {}
        guard = validation.get("zero_fill_guard") or {}
        if guard.get("mode"):
            modes.add(guard["mode"])
    if not modes:
        return None
    return modes.pop() if len(modes) == 1 else "inconsistent"


def chain_aligned_modis_signatures(ctx: dict) -> "OrderedDict[str, dict]":
    """SHA-256 of one chain's aligned MODIS rasters (read-only)."""
    aligned_dir = Path(ctx["step7b_output_dir"]) / "aligned_inputs"
    out: "OrderedDict[str, dict]" = OrderedDict()
    for feature, spec in MODIS_COMPATIBILITY_RASTERS.items():
        path = aligned_dir / spec["filename"]
        out[feature] = (
            OrderedDict((("path", str(path)), ("present", True),
                         *audit.sha256_and_size(path).items()))
            if path.exists() else
            OrderedDict((("path", str(path)), ("present", False)))
        )
    return out


def check_shared_modis_invariance(
    provenance: dict, reference_ctx: dict, candidate_ctx: dict,
    attestation: dict, base_dir: Path = PROJECT_ROOT,
) -> dict:
    """HARD GATE: MODIS must be identical in both chains, before and after Step7B.

    Fails with the dedicated technical field `shared_modis_invariance_failed`;
    the run then terminates as `baseline_invariance_failed` and no scientific
    conclusion is issued.
    """
    checks: "OrderedDict[str, bool]" = OrderedDict()
    reasons: list[str] = []

    def _require(name: str, ok: bool, message: str) -> None:
        checks[name] = bool(ok)
        if not ok:
            reasons.append(f"{name}: {message}")

    rows = modis_provenance_records(provenance)
    for feature, records in rows.items():
        paths = {r.get("materialized_path") for r in records}
        hashes = {r.get("sha256") for r in records}
        _require(
            f"{feature}__byte_identical_input_between_chains",
            len(records) == len(CHAINS) and len(paths) == 1 and len(hashes) == 1
            and all(r.get("shared_between_chains") for r in records),
            f"reference/candidate MODIS inputs differ (paths={paths}, hashes={hashes})",
        )

    reference_aligned = chain_aligned_modis_signatures(reference_ctx)
    candidate_aligned = chain_aligned_modis_signatures(candidate_ctx)
    for feature in MODIS_COMPATIBILITY_RASTERS:
        ref = reference_aligned[feature]
        cand = candidate_aligned[feature]
        _require(
            f"{feature}__identical_aligned_array_between_chains",
            ref.get("present") and cand.get("present")
            and ref.get("sha256") == cand.get("sha256"),
            "the aligned MODIS rasters are not byte-identical between chains "
            f"(reference={ref.get('sha256')}, candidate={cand.get('sha256')})",
        )

    reference_mode = chain_step7b_compatibility_mode(reference_ctx)
    candidate_mode = chain_step7b_compatibility_mode(candidate_ctx)
    expected_mode = (attestation or {}).get("mode", MODIS_STRICT_MODE)
    _require(
        "identical_compatibility_mode_between_chains",
        reference_mode == candidate_mode,
        f"reference mode {reference_mode!r} != candidate mode {candidate_mode!r}",
    )
    _require(
        "compatibility_mode_matches_the_attestation",
        reference_mode in (None, expected_mode),
        f"chains ran under {reference_mode!r} but the attestation authorized "
        f"{expected_mode!r}",
    )

    unchanged = True
    for feature, entry in ((attestation or {}).get("rasters") or {}).items():
        path = Path(entry.get("path", ""))
        if not path.exists() or audit.sha256_and_size(path)["sha256"] != entry.get("sha256"):
            unchanged = False
    _require(
        "no_modis_value_or_mask_changed_by_the_run",
        unchanged,
        "a frozen MODIS source no longer matches its attested SHA-256; the run "
        "must not modify MODIS in any way",
    )

    status = "pass" if not reasons else "fail"
    return OrderedDict((
        ("status", status),
        ("technical_failure", None if status == "pass" else TECHNICAL_FAILURE_SHARED_MODIS),
        ("checks", checks),
        ("reasons", reasons),
        ("modis_compatibility_mode", expected_mode),
        ("reference_mode", reference_mode),
        ("candidate_mode", candidate_mode),
        ("reference_aligned_modis", reference_aligned),
        ("candidate_aligned_modis", candidate_aligned),
        ("baseline_feature_invariance_still_required", True),
        ("reference_reproduction_still_required", True),
    ))


# =============================================================================
# Chain ExperimentContexts
# =============================================================================
def build_chain_context(
    experiment_id: str, chain: str, base_dir: Path = PROJECT_ROOT,
) -> dict:
    """A production ExperimentContext re-rooted into the A/B namespace.

    The canonical context is built by the production helper and then every
    input/output directory is remapped; no window, date, baseline-year, seed or
    threshold is altered. Chain-specific raw LST comes from the chain's own
    input bundle; every other input comes from the ONE shared bundle.
    """
    from core.experiment_context import build_experiment_context

    if chain not in CHAINS:
        raise DownstreamABError(f"unknown chain: {chain!r}. Expected one of {CHAINS}.")

    ctx = dict(build_experiment_context(experiment_id))
    root = diagnostic_output_root(experiment_id, base_dir)
    side = CHAIN_SIDE[chain]
    outputs_root = root / side
    chain_inputs = root / "inputs" / chain
    shared_inputs = root / "inputs" / "shared"

    ctx.update({
        "experiment_id": experiment_id,
        "is_kozan": False,
        "ab_chain": chain,
        "ab_side": side,
        "ab_root": root,
        "output_root": outputs_root,
        # --- inputs ---
        "data_root": chain_inputs,
        "baseline_input_dir": chain_inputs / "landsat_timeseries",
        "current_period_dir": chain_inputs / "current_period",
        "qa_dir": chain_inputs / "landsat_qa",
        "ndvi_baseline_dir": shared_inputs / "ndvi_timeseries",
        "ndvi_current_dir": shared_inputs / "ndvi_current_period",
        "modis_input_dir": shared_inputs / "modis",
        "dem_input_dir": shared_inputs / "dem",
        "dem_is_shared_read_only": False,
        "landcover_aligned_path":
            shared_inputs / "gate_inputs"
            / "landcover_esa_worldcover_v200_aligned_to_reference.tif",
        "gate_labels_dir": shared_inputs / "labels",
        "step4_metadata_path": None,
        # --- outputs ---
        "step5_output_dir": outputs_root / "step5",
        "step5b_output_dir": outputs_root / "step5b",
        "step5c_output_dir": outputs_root / "step5c",
        "output_dir": outputs_root / "step5",
        "step7a_output_dir": outputs_root / "step7a",
        "step7b_output_dir": outputs_root / "step7b",
        "step7c_output_dir": outputs_root / "step7c",
        "step7d_output_dir": outputs_root / "step7d",
        "step7e_output_dir": outputs_root / "step7e",
        "step8a_output_dir": outputs_root / "step8" / "step8a",
        "step8b_output_dir": outputs_root / "step8" / "step8b",
        "step8c_output_dir": outputs_root / "step8" / "step8c",
        "step8d_output_dir": outputs_root / "step8" / "step8d",
        "step8e_output_dir": outputs_root / "step8" / "step8e",
    })
    assert_chain_context_namespaced(ctx, experiment_id, base_dir)
    return ctx


CONTEXT_PATH_KEYS = (
    "output_root", "data_root", "baseline_input_dir", "current_period_dir", "qa_dir",
    "ndvi_baseline_dir", "ndvi_current_dir", "modis_input_dir", "dem_input_dir",
    "landcover_aligned_path", "gate_labels_dir",
    "step5_output_dir", "step5b_output_dir", "step5c_output_dir", "output_dir",
    "step7a_output_dir", "step7b_output_dir", "step7c_output_dir",
    "step7d_output_dir", "step7e_output_dir",
    "step8a_output_dir", "step8b_output_dir", "step8c_output_dir",
    "step8d_output_dir", "step8e_output_dir",
)


def assert_chain_context_namespaced(
    ctx: dict, experiment_id: str, base_dir: Path = PROJECT_ROOT,
) -> None:
    """No chain path may escape the A/B root -- inputs included.

    Input directories are checked too, because the production steps create
    output directories eagerly and a leaked input path would mean production
    code writing next to a frozen canonical file.
    """
    paths = [ctx[key] for key in CONTEXT_PATH_KEYS if ctx.get(key) is not None]
    assert_downstream_namespace_safe(paths, experiment_id, base_dir)


# =============================================================================
# Earth Engine guard
# =============================================================================
#: Production symbols that would submit an Earth Engine query/export. This
#: runner must never reach any of them.
FORBIDDEN_EE_CALLABLES = (
    ("core.gee_utils", "init_gee"),
    ("src.landsat_composite_counterfactual_audit", "build_ee_images"),
    ("src.landsat_composite_counterfactual_audit", "build_source_scene_metadata"),
    ("scripts.run_predictors_only", "export_image_direct_or_tiled"),
)


class EarthEngineGuard:
    """Context manager that makes every Earth Engine entry point raise.

    Installed for the whole live run, so "no Earth Engine code path is
    reachable" is enforced at runtime rather than merely asserted in prose.
    """

    def __init__(self) -> None:
        self._patched: list[tuple[object, str, object]] = []

    def _fail(self, name: str):
        def _raise(*args, **kwargs):
            raise DownstreamABError(
                f"Earth Engine entry point {name!r} was reached from the downstream "
                "A/B experiment. This runner is local-only by contract."
            )
        return _raise

    def __enter__(self) -> "EarthEngineGuard":
        import importlib
        import sys

        ee = sys.modules.get("ee")
        if ee is None:
            try:
                ee = importlib.import_module("ee")
            except ImportError:
                ee = None
        if ee is not None:
            for attr in ("Initialize", "Authenticate"):
                if hasattr(ee, attr):
                    self._patched.append((ee, attr, getattr(ee, attr)))
                    setattr(ee, attr, self._fail(f"ee.{attr}"))
            data = getattr(ee, "data", None)
            if data is not None:
                for attr in ("getInfo", "computeValue", "startProcessing"):
                    if hasattr(data, attr):
                        self._patched.append((data, attr, getattr(data, attr)))
                        setattr(data, attr, self._fail(f"ee.data.{attr}"))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for target, attr, original in reversed(self._patched):
            setattr(target, attr, original)
        self._patched.clear()
        return False


# =============================================================================
# Derived current-minus-baseline (per chain, canonical Step5 policy)
# =============================================================================
def build_current_minus_baseline(ctx: dict, out_path: Path) -> dict:
    """Write `current_minus_baseline_celsius.tif` for one chain.

    Canonical Step5 writes the z-score and both components but not their raw
    difference. It is rebuilt here from the chain's OWN Step5 outputs using the
    SAME guard policy Step5 applies (valid-count and baseline-count masks are
    already baked into those outputs), so no Step5 semantics are re-derived.
    """
    import numpy as np
    import rasterio

    from src.step5_preprocess_timeseries import output_profile

    step5 = Path(ctx["step5_output_dir"])
    current_path = step5 / "current_period_median_celsius.tif"
    baseline_path = step5 / "baseline_lst_mean_celsius.tif"
    assert_same_grid([current_path, baseline_path])

    with rasterio.open(current_path) as src:
        profile = output_profile(src.profile.copy())
        current = src.read(1, masked=True).astype("float32").filled(np.nan)
    with rasterio.open(baseline_path) as src:
        baseline = src.read(1, masked=True).astype("float32").filled(np.nan)

    difference = (current - baseline).astype("float32")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f".{out_path.name}.tmp"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(difference, 1)
    os.replace(str(tmp), str(out_path))

    finite = difference[np.isfinite(difference)]
    return {
        "path": str(out_path),
        "valid_pixels": int(finite.size),
        "mean": float(finite.mean()) if finite.size else None,
    }


# =============================================================================
# Reference reproduction against the frozen canonical pipeline
# =============================================================================
def canonical_product_path(experiment_id: str, product: str, base_dir: Path = PROJECT_ROOT) -> Path | None:
    """Frozen canonical counterpart of a compared product, when one exists."""
    root = canonical_experiment_root(experiment_id, base_dir)
    registry = compared_raster_products()
    mapping = {
        "current_lst_celsius": root / "step5" / "current_period_median_celsius.tif",
        "baseline_lst_mean_celsius": root / "step5" / "baseline_lst_mean_celsius.tif",
        "baseline_lst_std_celsius": root / "step5" / "baseline_lst_std_celsius.tif",
        "baseline_valid_count": root / "step5" / "baseline_valid_count.tif",
        "anomaly_zscore": root / "step5" / "anomaly_zscore.tif",
        "current_tvdi": root / "step5c" / "current_tvdi.tif",
        "tvdi_difference": root / "step5c" / "tvdi_difference.tif",
        "downscaled_lst_celsius": root / "step7d" / "downscaled_lst_celsius.tif",
        "fused_lst_celsius": root / "step7e" / "fused_lst_celsius.tif",
    }
    if product in mapping:
        return mapping[product]
    if product in registry and product not in DERIVED_PRODUCTS:
        return None
    return None


def compare_raster_semantic(
    produced: Path, canonical: Path, *, tolerance: float,
    min_mask_agreement: float = REPRODUCTION_MIN_MASK_AGREEMENT,
) -> dict:
    """Pipeline-semantic comparison of a produced raster against a frozen one.

    Grid equality is exact. Values are compared on the COMMON valid mask with a
    float32-appropriate tolerance -- never bitwise, because production operation
    order may legitimately differ. Mask agreement is reported separately so a
    coverage change can never hide inside a value tolerance.
    """
    import numpy as np
    import rasterio

    result: dict = {
        "produced_path": str(produced), "canonical_path": str(canonical),
        "tolerance": float(tolerance),
        "min_mask_agreement_required": float(min_mask_agreement),
    }
    if not Path(produced).exists() or not Path(canonical).exists():
        result.update({"status": "missing_input", "passed": False})
        return result

    prod_sig, canon_sig = raster_signature(produced), raster_signature(canonical)
    result["grid_equal"] = grids_equal(prod_sig, canon_sig)
    if not result["grid_equal"]:
        result.update({"status": "grid_mismatch", "passed": False,
                       "produced_grid": prod_sig, "canonical_grid": canon_sig})
        return result

    with rasterio.open(produced) as src:
        a = src.read(1, masked=True).astype("float64").filled(np.nan)
    with rasterio.open(canonical) as src:
        b = src.read(1, masked=True).astype("float64").filled(np.nan)

    mask_a, mask_b = np.isfinite(a), np.isfinite(b)
    total = int(mask_a.size)
    agreement = float(np.sum(mask_a == mask_b) / total) if total else 0.0
    common = mask_a & mask_b
    n_common = int(common.sum())
    if n_common:
        diff = np.abs(a[common] - b[common])
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
    else:
        max_abs, mean_abs = None, None

    result.update({
        "valid_mask_agreement": agreement,
        "produced_valid_pixels": int(mask_a.sum()),
        "canonical_valid_pixels": int(mask_b.sum()),
        "common_valid_pixels": n_common,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
    })
    passed = (
        agreement >= min_mask_agreement
        and n_common > 0
        and max_abs is not None
        and max_abs <= tolerance
    )
    result["passed"] = bool(passed)
    result["status"] = "reproduced" if passed else "not_reproduced"
    return result


def compare_reference_step8_to_canonical(
    reference_dataset, canonical_dataset, reference_metrics: dict | None,
    canonical_metrics: dict | None,
) -> dict:
    """Compare the isolated reference Step8A/8B run to the frozen canonical run.

    Population membership, labels, block ids and the canonical Step8B metrics
    are all checked; folds are compared when the frozen run exposed them.
    """
    import numpy as np
    import pandas as pd

    out: dict = {}
    ref = reference_dataset.sort_values("cell_id").reset_index(drop=True)
    canon = canonical_dataset.sort_values("cell_id").reset_index(drop=True)

    out["reference_rows"] = int(len(ref))
    out["canonical_rows"] = int(len(canon))
    out["row_count_equal"] = bool(len(ref) == len(canon))
    out["cell_ids_equal"] = bool(
        len(ref) == len(canon) and np.array_equal(
            ref["cell_id"].to_numpy(), canon["cell_id"].to_numpy()
        )
    )
    for column in ("burned", "valid_for_modeling", PRIMARY_POPULATION, "row_500m", "col_500m"):
        if column in ref.columns and column in canon.columns and out["cell_ids_equal"]:
            out[f"{column}_equal"] = bool(
                pd.Series(ref[column]).equals(pd.Series(canon[column]))
            )
        else:
            out[f"{column}_equal"] = None

    metric_checks: dict = {}
    if reference_metrics and canonical_metrics:
        for population in (PRIMARY_POPULATION, "all_valid"):
            r = (reference_metrics.get("populations") or {}).get(population) or {}
            c = (canonical_metrics.get("populations") or {}).get(population) or {}
            for key in ("delta_auc", "delta_pr_auc", "delta_brier"):
                rv, cv = r.get(key), c.get(key)
                if rv is None or cv is None:
                    metric_checks[f"{population}.{key}"] = {
                        "reference": rv, "canonical": cv, "within_tolerance": None,
                    }
                    continue
                metric_checks[f"{population}.{key}"] = {
                    "reference": float(rv), "canonical": float(cv),
                    "abs_diff": abs(float(rv) - float(cv)),
                    "within_tolerance": abs(float(rv) - float(cv)) <= REPRODUCTION_STEP8_METRIC_TOL,
                }
    out["step8b_metric_checks"] = metric_checks

    checks = [
        out["row_count_equal"], out["cell_ids_equal"],
        out.get("burned_equal"), out.get(f"{PRIMARY_POPULATION}_equal"),
    ]
    metric_ok = [
        v["within_tolerance"] for v in metric_checks.values()
        if v.get("within_tolerance") is not None
    ]
    out["passed"] = bool(
        all(c for c in checks if c is not None) and all(metric_ok)
    )
    return out


def build_reference_reproduction_report(
    experiment_id: str, raster_checks: "OrderedDict[str, dict]", step8_check: dict,
) -> dict:
    """Assemble `reference_reproduction.json` and its pass/fail verdict."""
    raster_passed = all(v.get("passed") for v in raster_checks.values())
    passed = bool(raster_passed and step8_check.get("passed"))
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("gate", "reference_chain_reproduces_frozen_canonical_pipeline"),
        ("comparison_policy",
         "exact grid equality; valid-mask agreement reported separately; values "
         "compared on the common valid mask with predeclared float32 tolerances. "
         "No bitwise requirement is imposed where production operation order can "
         "legitimately differ."),
        ("predeclared_tolerances", dict(REPRODUCTION_TOLERANCES)),
        ("rasters", raster_checks),
        ("step8", step8_check),
        ("raster_checks_passed", raster_passed),
        ("step8_checks_passed", bool(step8_check.get("passed"))),
        ("status", "pass" if passed else "fail"),
        ("failure_status_if_not_reproduced", STATUS_INVALID_REFERENCE),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


# =============================================================================
# Step5/Step7 raster change summary
# =============================================================================
def compare_raster_change(
    reference_path: Path, candidate_path: Path, *, product: str,
    changed_threshold: float,
) -> dict:
    """Descriptive candidate-minus-reference difference statistics.

    Tiny float32 differences are NOT treated as scientific change: the
    changed-pixel fraction is reported against a documented descriptive
    threshold, and the raw distribution statistics are reported alongside it.
    """
    import numpy as np
    import rasterio

    row: "OrderedDict[str, object]" = OrderedDict((
        ("product", product),
        ("reference_path", str(reference_path)),
        ("candidate_path", str(candidate_path)),
        ("changed_pixel_threshold", float(changed_threshold)),
    ))

    ref_sig = raster_signature(reference_path)
    cand_sig = raster_signature(candidate_path)
    row["grid_equal"] = grids_equal(ref_sig, cand_sig)
    if not row["grid_equal"]:
        row["status"] = "grid_mismatch"
        return row

    with rasterio.open(reference_path) as src:
        a = src.read(1, masked=True).astype("float64").filled(np.nan)
    with rasterio.open(candidate_path) as src:
        b = src.read(1, masked=True).astype("float64").filled(np.nan)

    mask_a, mask_b = np.isfinite(a), np.isfinite(b)
    total = int(mask_a.size)
    common = mask_a & mask_b
    n_common = int(common.sum())

    row["reference_valid_pixels"] = int(mask_a.sum())
    row["candidate_valid_pixels"] = int(mask_b.sum())
    row["valid_mask_agreement"] = float(np.sum(mask_a == mask_b) / total) if total else 0.0
    row["reference_only_valid_pixels"] = int(np.sum(mask_a & ~mask_b))
    row["candidate_only_valid_pixels"] = int(np.sum(mask_b & ~mask_a))
    row["common_valid_pixels"] = n_common

    if n_common == 0:
        row["status"] = "no_common_valid_pixels"
        return row

    d = b[common] - a[common]  # candidate minus reference
    abs_d = np.abs(d)
    row.update(OrderedDict((
        ("mean", float(d.mean())),
        ("median", float(np.median(d))),
        ("std", float(d.std(ddof=0))),
        ("mae", float(abs_d.mean())),
        ("rmse", float(np.sqrt(np.mean(d ** 2)))),
        ("p01", float(np.percentile(d, 1))),
        ("p05", float(np.percentile(d, 5))),
        ("p50", float(np.percentile(d, 50))),
        ("p95", float(np.percentile(d, 95))),
        ("p99", float(np.percentile(d, 99))),
        ("max_abs_diff", float(abs_d.max())),
        ("changed_pixel_fraction", float(np.mean(abs_d > changed_threshold))),
        ("status", "compared"),
    )))
    return row


RASTER_CHANGE_COLUMNS = (
    "product", "status", "grid_equal", "reference_valid_pixels",
    "candidate_valid_pixels", "valid_mask_agreement", "reference_only_valid_pixels",
    "candidate_only_valid_pixels", "common_valid_pixels", "mean", "median", "std",
    "mae", "rmse", "p01", "p05", "p50", "p95", "p99", "max_abs_diff",
    "changed_pixel_threshold", "changed_pixel_fraction",
    "reference_path", "candidate_path",
)


# =============================================================================
# Common modelling cohort
# =============================================================================
def eligible_rows(df, population: str = PRIMARY_POPULATION):
    """Rows a chain considers eligible: valid for modelling AND in the primary
    population. Uses the frozen Step8A/Step8B semantics, not a new rule."""
    mask = (df["valid_for_modeling"] == True)  # noqa: E712
    if population != "all_valid":
        mask = mask & df[population].astype(bool)
    return df.loc[mask].sort_values("cell_id").reset_index(drop=True)


def build_common_cohort(reference_df, candidate_df, *, population: str = PRIMARY_POPULATION) -> dict:
    """Exact intersection of eligible reference and candidate cells.

    Returns the two per-chain frames restricted to the identical, identically
    ordered cohort plus the alignment bookkeeping. Labels, grid indices and
    population membership must match on the intersection or the caller must
    treat the comparison as un-credible.
    """
    import numpy as np

    ref = eligible_rows(reference_df, population)
    cand = eligible_rows(candidate_df, population)

    ref_ids = set(ref["cell_id"].tolist())
    cand_ids = set(cand["cell_id"].tolist())
    common_ids = sorted(ref_ids & cand_ids)

    ref_common = ref[ref["cell_id"].isin(common_ids)].sort_values("cell_id").reset_index(drop=True)
    cand_common = cand[cand["cell_id"].isin(common_ids)].sort_values("cell_id").reset_index(drop=True)

    labels_match = bool(np.array_equal(
        ref_common["burned"].to_numpy(), cand_common["burned"].to_numpy()
    ))
    rowcol_match = bool(
        np.array_equal(ref_common["row_500m"].to_numpy(), cand_common["row_500m"].to_numpy())
        and np.array_equal(ref_common["col_500m"].to_numpy(), cand_common["col_500m"].to_numpy())
    )
    population_match = bool(np.array_equal(
        ref_common[population].astype(bool).to_numpy(),
        cand_common[population].astype(bool).to_numpy(),
    )) if population != "all_valid" else True

    return {
        "population": population,
        "reference": ref_common,
        "candidate": cand_common,
        "reference_native": ref,
        "candidate_native": cand,
        "common_cell_ids": common_ids,
        "reference_only_cell_ids": sorted(ref_ids - cand_ids),
        "candidate_only_cell_ids": sorted(cand_ids - ref_ids),
        "labels_match": labels_match,
        "row_col_match": rowcol_match,
        "population_match": population_match,
    }


def feature_exclusion_reasons(df, chain_label: str) -> dict:
    """Why rows were excluded, by Step8A's own `invalid_reason` bookkeeping."""
    invalid = df.loc[df["valid_for_modeling"] != True]  # noqa: E712
    counts: dict[str, int] = {}
    for raw in invalid.get("invalid_reason", []):
        if raw is None or (isinstance(raw, float)):
            key = "unspecified"
            counts[key] = counts.get(key, 0) + 1
            continue
        for reason in str(raw).split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return {"chain": chain_label, "invalid_rows": int(len(invalid)), "reasons": counts}


def build_population_alignment(
    experiment_id: str, cohort: dict, reference_df, candidate_df,
) -> dict:
    """Assemble `population_alignment.json`.

    Every difference is recorded even when the common-cohort analysis remains
    possible; the review verdict uses predeclared retention thresholds.
    """
    ref_native, cand_native = cohort["reference_native"], cohort["candidate_native"]
    n_ref, n_cand = int(len(ref_native)), int(len(cand_native))
    n_common = int(len(cohort["common_cell_ids"]))

    def _positives(frame, ids=None):
        if ids is not None:
            frame = frame[frame["cell_id"].isin(ids)]
        return int((frame["burned"].astype(int) == 1).sum())

    ref_pos, cand_pos = _positives(ref_native), _positives(cand_native)
    common_pos_ref = _positives(ref_native, cohort["common_cell_ids"])
    common_pos_cand = _positives(cand_native, cohort["common_cell_ids"])

    denominator = max(n_ref, n_cand)
    retention = float(n_common / denominator) if denominator else 0.0
    pos_denominator = max(ref_pos, cand_pos)
    positive_retention = float(min(common_pos_ref, common_pos_cand) / pos_denominator) if pos_denominator else 0.0

    consistent = bool(
        cohort["labels_match"] and cohort["row_col_match"] and cohort["population_match"]
    )
    review_reasons: list[str] = []
    if not cohort["labels_match"]:
        review_reasons.append("labels_differ_on_common_cells")
    if not cohort["row_col_match"]:
        review_reasons.append("grid_indices_differ_on_common_cells")
    if not cohort["population_match"]:
        review_reasons.append("primary_population_membership_differs_on_common_cells")
    if retention < MIN_COMMON_ROW_RETENTION:
        review_reasons.append(
            f"common_row_retention {retention:.6f} < {MIN_COMMON_ROW_RETENTION}"
        )
    if positive_retention < MIN_COMMON_POSITIVE_RETENTION:
        review_reasons.append(
            f"common_positive_retention {positive_retention:.6f} < {MIN_COMMON_POSITIVE_RETENTION}"
        )
    if n_common == 0:
        review_reasons.append("empty_common_cohort")

    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("primary_population", cohort["population"]),
        ("primary_population_declared_before_run", True),
        ("total_reference_rows", n_ref),
        ("total_candidate_rows", n_cand),
        ("common_rows", n_common),
        ("reference_only_rows", int(len(cohort["reference_only_cell_ids"]))),
        ("candidate_only_rows", int(len(cohort["candidate_only_cell_ids"]))),
        ("reference_only_cell_ids_sample", cohort["reference_only_cell_ids"][:100]),
        ("candidate_only_cell_ids_sample", cohort["candidate_only_cell_ids"][:100]),
        ("positive_cells", OrderedDict((
            ("reference_eligible", ref_pos),
            ("candidate_eligible", cand_pos),
            ("common_reference_view", common_pos_ref),
            ("common_candidate_view", common_pos_cand),
            ("reference_only", ref_pos - common_pos_ref),
            ("candidate_only", cand_pos - common_pos_cand),
        ))),
        ("common_row_retention_fraction", retention),
        ("common_positive_retention_fraction", positive_retention),
        ("labels_match_on_common_cells", cohort["labels_match"]),
        ("row_col_match_on_common_cells", cohort["row_col_match"]),
        ("population_match_on_common_cells", cohort["population_match"]),
        ("row_exclusion_reasons", [
            feature_exclusion_reasons(reference_df, CHAIN_REFERENCE),
            feature_exclusion_reasons(candidate_df, CHAIN_CANDIDATE),
        ]),
        ("predeclared_thresholds", OrderedDict((
            ("min_common_row_retention", MIN_COMMON_ROW_RETENTION),
            ("min_common_positive_retention", MIN_COMMON_POSITIVE_RETENTION),
        ))),
        ("cohort_consistent", consistent),
        ("review_reasons", review_reasons),
        ("status", "ok" if not review_reasons else "requires_review"),
        ("note",
         "The primary A/B model comparison uses ONLY the frozen common cohort. "
         "Variant-native populations are reported as sensitivity diagnostics and "
         "never as the primary result."),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


# =============================================================================
# Spatial blocks and folds (one deterministic assignment shared by both chains)
# =============================================================================
def build_fold_assignment(cohort_df, *, population: str = PRIMARY_POPULATION):
    """ONE deterministic spatial-block / CV-fold assignment for the cohort.

    Uses the production Step8B helpers (`add_spatial_block_id`,
    `make_spatial_folds`) with the frozen canonical Step8 configuration --
    same block size, fold count, seed and StratifiedGroupKFold algorithm. A
    random row split is structurally impossible here: folds always come from
    grouped, block-stratified splitting.
    """
    import numpy as np
    import pandas as pd

    from core.config import STEP8B_N_SPLITS, STEP8B_RANDOM_SEED, STEP8B_SPATIAL_BLOCK_SIZE_CELLS
    from src.step8b_train_baseline_vs_thermal_model import add_spatial_block_id, make_spatial_folds

    df = add_spatial_block_id(cohort_df, STEP8B_SPATIAL_BLOCK_SIZE_CELLS)
    y = df["burned"].astype(int).to_numpy()
    groups = df["spatial_block_id"].to_numpy()
    folds, n_splits_used = make_spatial_folds(
        y, groups, STEP8B_N_SPLITS, STEP8B_RANDOM_SEED,
    )

    fold_id = np.full(len(df), -1, dtype=int)
    for index, (_, test_idx) in enumerate(folds):
        fold_id[test_idx] = index
    if int((fold_id < 0).sum()):
        raise DownstreamABError(
            "fold assignment left rows without a test fold; refusing to continue."
        )

    assignment = pd.DataFrame({
        "cell_id": df["cell_id"].to_numpy(),
        "grid_row": df["row_500m"].to_numpy(),
        "grid_col": df["col_500m"].to_numpy(),
        "label": y,
        "population": population,
        "spatial_block_id": groups,
        "cv_fold": fold_id,
        "seed": STEP8B_RANDOM_SEED,
        "block_size_cells": STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
        "n_splits": n_splits_used,
    })
    return assignment, df, folds


def assert_identical_fold_assignment(reference_fold_id, candidate_fold_id, canonical_fold_id) -> dict:
    """Both chains must land on the SAME fold assignment as the shared manifest."""
    import numpy as np

    ref = np.asarray(reference_fold_id)
    cand = np.asarray(candidate_fold_id)
    canon = np.asarray(canonical_fold_id)
    reference_matches = bool(np.array_equal(ref, canon))
    candidate_matches = bool(np.array_equal(cand, canon))
    if not (reference_matches and candidate_matches):
        raise DownstreamABError(
            "fold assignment differs between chains or from the shared manifest; "
            "the paired comparison would not be like-for-like."
        )
    return {
        "reference_matches_manifest": reference_matches,
        "candidate_matches_manifest": candidate_matches,
        "chains_identical": bool(np.array_equal(ref, cand)),
    }


# =============================================================================
# Step8 model comparison on the common cohort
# =============================================================================
def run_chain_model(cohort_df, *, population: str = PRIMARY_POPULATION) -> dict:
    """Train baseline and thermal models for ONE chain on the common cohort.

    Delegates to the production `train_population` so the feature sets, model
    class, hyper-parameters, CV construction and metric definitions are exactly
    the canonical Step8B ones. Nothing is tuned here.
    """
    from core.config import (
        STEP8B_MIN_POSITIVES_PER_POPULATION, STEP8B_N_SPLITS, STEP8B_RANDOM_SEED,
    )
    from src.step8b_train_baseline_vs_thermal_model import train_population

    result = train_population(
        cohort_df, population,
        n_splits=STEP8B_N_SPLITS,
        random_state=STEP8B_RANDOM_SEED,
        model_name="random_forest",
        min_positives=STEP8B_MIN_POSITIVES_PER_POPULATION,
    )
    if result is None or result.get("skipped"):
        raise DownstreamABError(
            "Step8B refused to model the common cohort: "
            f"{(result or {}).get('reason', 'no result')}"
        )
    return result


def check_baseline_invariance(reference_df, candidate_df, reference_result, candidate_result) -> dict:
    """HARD GATE: the baseline chain must be identical despite the LST-only change.

    Feature names and order, labels, folds, baseline feature values and the
    baseline OOF predictions are all compared. Any mismatch ends the experiment
    with `baseline_invariance_failed`.
    """
    import numpy as np

    from src.step8b_train_baseline_vs_thermal_model import BASELINE_FEATURES, THERMAL_MODEL_FEATURES

    checks: "OrderedDict[str, object]" = OrderedDict()
    checks["baseline_feature_names"] = list(BASELINE_FEATURES)
    checks["thermal_feature_names"] = list(THERMAL_MODEL_FEATURES)
    checks["feature_names_and_order_equal"] = True  # single frozen registry, shared by both chains

    ref_features = reference_df[list(BASELINE_FEATURES)]
    cand_features = candidate_df[list(BASELINE_FEATURES)]
    checks["baseline_columns_equal"] = bool(
        list(ref_features.columns) == list(cand_features.columns)
    )

    per_feature: "OrderedDict[str, object]" = OrderedDict()
    features_equal = True
    for name in BASELINE_FEATURES:
        a, b = ref_features[name], cand_features[name]
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            av = a.to_numpy(dtype="float64")
            bv = b.to_numpy(dtype="float64")
            both_nan = np.isnan(av) & np.isnan(bv)
            diff = np.abs(np.where(both_nan, 0.0, av - bv))
            max_abs = float(np.nanmax(diff)) if diff.size else 0.0
            equal = bool(max_abs <= BASELINE_FEATURE_MAX_ABS_DIFF)
            per_feature[name] = {"max_abs_diff": max_abs, "equal": equal}
        else:
            equal = bool(a.astype(str).equals(b.astype(str)))
            per_feature[name] = {"max_abs_diff": None, "equal": equal}
        features_equal = features_equal and equal
    checks["baseline_feature_values"] = per_feature
    checks["baseline_feature_values_equal"] = features_equal

    checks["labels_equal"] = bool(np.array_equal(
        reference_df["burned"].astype(int).to_numpy(),
        candidate_df["burned"].astype(int).to_numpy(),
    ))
    checks["folds_equal"] = bool(np.array_equal(
        np.asarray(reference_result["fold_id"]), np.asarray(candidate_result["fold_id"])
    ))
    checks["model_configuration_equal"] = True  # both call the same frozen builder

    ref_oof = np.asarray(reference_result["oof_prob_baseline"], dtype="float64")
    cand_oof = np.asarray(candidate_result["oof_prob_baseline"], dtype="float64")
    if ref_oof.shape != cand_oof.shape:
        max_oof_diff = None
        oof_equal = False
    else:
        max_oof_diff = float(np.nanmax(np.abs(ref_oof - cand_oof))) if ref_oof.size else 0.0
        oof_equal = bool(max_oof_diff <= BASELINE_OOF_MAX_ABS_DIFF)
    checks["baseline_oof_max_abs_diff"] = max_oof_diff
    checks["baseline_oof_tolerance"] = BASELINE_OOF_MAX_ABS_DIFF
    checks["baseline_oof_predictions_equal"] = oof_equal

    passed = bool(
        checks["baseline_columns_equal"] and features_equal and checks["labels_equal"]
        and checks["folds_equal"] and oof_equal
    )
    checks["passed"] = passed
    checks["status"] = "pass" if passed else "fail"
    checks["failure_status_if_not_invariant"] = STATUS_BASELINE_INVARIANCE_FAILED
    checks["documented_harmless_serialization_differences"] = [
        "Reference annual baseline LST is float64 with GEE-masked pixels serialized "
        "as raw DN 0 and no nodata tag; candidate annual baseline LST is float32 "
        "with an explicit -9999 nodata sentinel. Both encodings are removed by the "
        "identical Step5 physical-range mask before any value is used, so they "
        "cannot reach a feature value.",
    ]
    return checks


def paired_block_bootstrap(
    cohort_df, y, prob_baseline, prob_thermal_reference, prob_thermal_candidate, *,
    n_bootstrap: int = PAIRED_BOOTSTRAP_REPLICATES,
    seed: int | None = None,
    ci_lower: float = PAIRED_BOOTSTRAP_CI_LOWER,
    ci_upper: float = PAIRED_BOOTSTRAP_CI_UPPER,
) -> dict:
    """Spatial-block bootstrap evaluated on IDENTICAL block draws for both chains.

    One replicate draws spatial blocks with replacement ONCE and then scores the
    baseline, reference-thermal and candidate-thermal predictions on exactly the
    same resampled rows. This makes the candidate-minus-reference deltas properly
    paired: replicate-to-replicate sampling noise is shared, not independent.

    Reuses Step8C's `build_block_index` and `compute_metrics` so the resampling
    unit and the metric definitions are the canonical ones.
    """
    import numpy as np
    import pandas as pd

    from core.config import STEP8C_RANDOM_SEED
    from src.step8c_spatial_block_bootstrap_uncertainty import build_block_index, compute_metrics

    seed = STEP8C_RANDOM_SEED if seed is None else seed

    frame = pd.DataFrame({
        "spatial_block_id": cohort_df["spatial_block_id"].to_numpy(),
        "burned": np.asarray(y, dtype=int),
        "p_baseline": np.asarray(prob_baseline, dtype="float64"),
        "p_thermal_reference": np.asarray(prob_thermal_reference, dtype="float64"),
        "p_thermal_candidate": np.asarray(prob_thermal_candidate, dtype="float64"),
    })
    unique_blocks, block_to_idx, sub = build_block_index(frame)
    n_blocks = len(unique_blocks)
    if n_blocks < 2:
        raise DownstreamABError(
            "paired bootstrap needs at least two spatial blocks; refusing a random "
            "row bootstrap."
        )

    rng = np.random.default_rng(seed)
    y_sub = sub["burned"].to_numpy()
    p_base = sub["p_baseline"].to_numpy()
    p_ref = sub["p_thermal_reference"].to_numpy()
    p_cand = sub["p_thermal_candidate"].to_numpy()

    replicates: list[dict] = []
    block_draw_digest: list[str] = []
    for iteration in range(int(n_bootstrap)):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_to_idx[b] for b in sampled_blocks])
        if iteration < 3:
            block_draw_digest.append(",".join(map(str, sampled_blocks[:8])))

        reference = compute_metrics(y_sub[idx], p_base[idx], p_ref[idx])
        candidate = compute_metrics(y_sub[idx], p_base[idx], p_cand[idx])
        if reference is None or candidate is None:
            continue
        replicates.append({
            "iteration": iteration,
            "n_rows": int(idx.size),
            "n_blocks_sampled": int(n_blocks),
            # thermal-minus-baseline, per chain (same rows, same draw)
            "reference_delta_roc_auc": reference["delta_auc"],
            "reference_delta_pr_auc": reference["delta_pr_auc"],
            "reference_delta_brier": reference["delta_brier"],
            "candidate_delta_roc_auc": candidate["delta_auc"],
            "candidate_delta_pr_auc": candidate["delta_pr_auc"],
            "candidate_delta_brier": candidate["delta_brier"],
            # direct paired candidate-minus-reference thermal deltas
            "paired_delta_roc_auc": candidate["auc_thermal"] - reference["auc_thermal"],
            "paired_delta_pr_auc": candidate["pr_auc_thermal"] - reference["pr_auc_thermal"],
            "paired_delta_brier": candidate["brier_thermal"] - reference["brier_thermal"],
        })

    if len(replicates) < 2:
        raise DownstreamABError("paired bootstrap produced fewer than two usable replicates.")

    replicate_frame = pd.DataFrame(replicates)

    def _interval(column: str) -> dict:
        values = replicate_frame[column].to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        low = float(np.percentile(values, ci_lower))
        high = float(np.percentile(values, ci_upper))
        return OrderedDict((
            ("point_estimate_bootstrap_mean", float(values.mean())),
            ("interval_low", low),
            ("interval_high", high),
            ("interval_excludes_zero", bool(low > 0.0 or high < 0.0)),
            ("interval_wholly_above_zero", bool(low > 0.0)),
            ("interval_wholly_below_zero", bool(high < 0.0)),
            ("n_replicates", int(values.size)),
        ))

    intervals = OrderedDict(
        (column, _interval(column))
        for column in (
            "reference_delta_roc_auc", "reference_delta_pr_auc", "reference_delta_brier",
            "candidate_delta_roc_auc", "candidate_delta_pr_auc", "candidate_delta_brier",
            "paired_delta_roc_auc", "paired_delta_pr_auc", "paired_delta_brier",
        )
    )

    return {
        "bootstrap_unit": "spatial_block_id",
        "n_blocks": int(n_blocks),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_used": int(len(replicates)),
        "seed": int(seed),
        "ci_lower_percentile": float(ci_lower),
        "ci_upper_percentile": float(ci_upper),
        "identical_block_draws_for_both_chains": True,
        "block_draw_digest_first_replicates": block_draw_digest,
        "intervals": intervals,
        "replicates": replicate_frame,
        "interval_language":
            "Report intervals as excluding/including zero. Never as "
            "'statistically significant'.",
        "direction": OrderedDict((
            ("roc_auc", "candidate_minus_reference > 0 means improvement"),
            ("pr_auc", "candidate_minus_reference > 0 means improvement"),
            ("brier", "candidate_minus_reference < 0 means improvement"),
        )),
    }


def metric_improved(metric: str, value: float | None) -> bool | None:
    """Sign convention for candidate-minus-reference deltas."""
    if value is None:
        return None
    if metric in ("roc_auc", "pr_auc"):
        return bool(value > 0.0)
    if metric == "brier":
        return bool(value < 0.0)
    raise DownstreamABError(f"unknown metric for direction check: {metric!r}")


def build_step8_metric_rows(reference_result: dict, candidate_result: dict, intervals: dict) -> list[dict]:
    """`step8_metrics.csv` rows: per-chain point metrics on the common cohort."""
    rows: list[dict] = []
    for chain, result, prefix in (
        (CHAIN_REFERENCE, reference_result, "reference"),
        (CHAIN_CANDIDATE, candidate_result, "candidate"),
    ):
        rows.append(OrderedDict((
            ("chain", chain),
            ("population", PRIMARY_POPULATION),
            ("cohort", "common_cohort"),
            ("n_rows", int(result["n_positives"] + result["n_negatives"])),
            ("n_positives", int(result["n_positives"])),
            ("n_negatives", int(result["n_negatives"])),
            ("baseline_roc_auc", result["overall_baseline"]["roc_auc"]),
            ("baseline_pr_auc", result["overall_baseline"]["pr_auc"]),
            ("baseline_brier", result["overall_baseline"]["brier_score"]),
            ("thermal_roc_auc", result["overall_thermal"]["roc_auc"]),
            ("thermal_pr_auc", result["overall_thermal"]["pr_auc"]),
            ("thermal_brier", result["overall_thermal"]["brier_score"]),
            ("delta_roc_auc_thermal_minus_baseline", result["delta_auc"]),
            ("delta_pr_auc_thermal_minus_baseline", result["delta_pr_auc"]),
            ("delta_brier_thermal_minus_baseline", result["delta_brier"]),
            ("delta_roc_auc_interval_low", intervals[f"{prefix}_delta_roc_auc"]["interval_low"]),
            ("delta_roc_auc_interval_high", intervals[f"{prefix}_delta_roc_auc"]["interval_high"]),
            ("delta_pr_auc_interval_low", intervals[f"{prefix}_delta_pr_auc"]["interval_low"]),
            ("delta_pr_auc_interval_high", intervals[f"{prefix}_delta_pr_auc"]["interval_high"]),
            ("delta_brier_interval_low", intervals[f"{prefix}_delta_brier"]["interval_low"]),
            ("delta_brier_interval_high", intervals[f"{prefix}_delta_brier"]["interval_high"]),
        )))
    return rows


def build_paired_bootstrap_rows(reference_result: dict, candidate_result: dict, bootstrap: dict) -> list[dict]:
    """`step8_paired_bootstrap.csv` rows: direct candidate-minus-reference deltas."""
    intervals = bootstrap["intervals"]
    point = {
        "roc_auc": (
            candidate_result["overall_thermal"]["roc_auc"]
            - reference_result["overall_thermal"]["roc_auc"]
        ),
        "pr_auc": (
            candidate_result["overall_thermal"]["pr_auc"]
            - reference_result["overall_thermal"]["pr_auc"]
        ),
        "brier": (
            candidate_result["overall_thermal"]["brier_score"]
            - reference_result["overall_thermal"]["brier_score"]
        ),
    }
    rows: list[dict] = []
    for metric, column in (
        ("roc_auc", "paired_delta_roc_auc"),
        ("pr_auc", "paired_delta_pr_auc"),
        ("brier", "paired_delta_brier"),
    ):
        interval = intervals[column]
        rows.append(OrderedDict((
            ("metric", metric),
            ("comparison", "candidate_minus_reference_thermal"),
            ("population", PRIMARY_POPULATION),
            ("cohort", "common_cohort"),
            ("point_estimate", point[metric]),
            ("bootstrap_mean", interval["point_estimate_bootstrap_mean"]),
            ("interval_low", interval["interval_low"]),
            ("interval_high", interval["interval_high"]),
            ("interval_excludes_zero", interval["interval_excludes_zero"]),
            ("interval_wholly_above_zero", interval["interval_wholly_above_zero"]),
            ("interval_wholly_below_zero", interval["interval_wholly_below_zero"]),
            ("improvement_direction",
             "positive_is_improvement" if metric != "brier" else "negative_is_improvement"),
            ("point_estimate_indicates_improvement", metric_improved(metric, point[metric])),
            ("bootstrap_unit", bootstrap["bootstrap_unit"]),
            ("n_blocks", bootstrap["n_blocks"]),
            ("n_bootstrap_used", bootstrap["n_bootstrap_used"]),
            ("seed", bootstrap["seed"]),
            ("identical_block_draws_for_both_chains",
             bootstrap["identical_block_draws_for_both_chains"]),
        )))
    return rows


def build_oof_predictions(cohort_df, assignment, reference_result: dict, candidate_result: dict):
    """`oof_predictions.csv`: one row per common-cohort cell."""
    import numpy as np
    import pandas as pd

    baseline = np.asarray(reference_result["oof_prob_baseline"], dtype="float64")
    thermal_reference = np.asarray(reference_result["oof_prob_thermal"], dtype="float64")
    thermal_candidate = np.asarray(candidate_result["oof_prob_thermal"], dtype="float64")

    return pd.DataFrame({
        "cell_id": cohort_df["cell_id"].to_numpy(),
        "label": cohort_df["burned"].astype(int).to_numpy(),
        "spatial_block_id": cohort_df["spatial_block_id"].to_numpy(),
        "cv_fold": assignment["cv_fold"].to_numpy(),
        "baseline_probability": baseline,
        "thermal_reference_probability": thermal_reference,
        "thermal_candidate_probability": thermal_candidate,
        "candidate_minus_reference_probability": thermal_candidate - thermal_reference,
    })


# =============================================================================
# Boundary propagation
# =============================================================================
def boundary_support_paths(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Frozen support-count rasters that DEFINE the boundary adjacency pairs.

    These come from the completed counterfactual audit and are identical for
    both chains by construction, so reference and candidate are always sampled
    at the same adjacency indices.
    """
    root = counterfactual_source_root(experiment_id, base_dir)
    return {
        "scene_count_edge": root / "rasters" / "current_lst_scene_valid_count.tif",
        "unique_date_count_edge": root / "rasters" / "current_lst_unique_date_valid_count.tif",
        "same_day_multiplicity_edge": root / "rasters" / "current_lst_same_day_multiplicity.tif",
    }


def frozen_provenance_state(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> tuple[str, Path | None]:
    """Reuse the frozen boundary definitions; never regenerate GEE provenance."""
    root = counterfactual_source_root(experiment_id, base_dir)
    summary_path = root / "counterfactual_summary.json"
    geojson_path = root / "scene_boundaries.geojson"
    state = "insufficient_boundary_metadata"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        provenance = summary.get("provenance") or {}
        state = provenance.get("status") or audit.map_provenance_status(provenance) or state
    if state != "provenance_available" or not geojson_path.exists():
        return state, None
    return state, geojson_path


def run_boundary_propagation(
    experiment_id: str, reference_ctx: dict, candidate_ctx: dict, *,
    tmp_dir: Path, base_dir: Path = PROJECT_ROOT, products=BOUNDARY_PROPAGATION_PRODUCTS,
    resource_log: list | None = None,
) -> dict:
    """Bounded-memory boundary audit of every downstream product, both chains.

    Delegates to the counterfactual audit's windowed implementation with the
    FROZEN boundary definitions. `sw_path` is the reference chain and `db_path`
    is the candidate chain, so the audit's reduction definition
    (``absolute_jump_reference - absolute_jump_candidate``; positive means the
    candidate has the smaller boundary jump) carries over unchanged.
    """
    support_paths = boundary_support_paths(experiment_id, base_dir)
    missing = [str(p) for p in support_paths.values() if not Path(p).exists()]
    if missing:
        raise PrerequisiteError(
            f"frozen boundary support rasters are missing: {missing}"
        )

    provenance_state, geojson_path = frozen_provenance_state(experiment_id, base_dir)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    provenance_code_spec = None
    if geojson_path is not None:
        import rasterio

        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        with rasterio.open(support_paths["scene_count_edge"]) as src:
            transform, width, height = src.transform, src.width, src.height
        provenance_code_spec = audit.build_provenance_code_memmap(
            geojson, transform, width, height, tmp_dir,
        )

    rows: list[dict] = []
    verdicts: "OrderedDict[str, dict]" = OrderedDict()
    adjacency_identity: "OrderedDict[str, dict]" = OrderedDict()

    for product in products:
        ref_path = product_path(reference_ctx, product, Path(reference_ctx["output_root"]))
        cand_path = product_path(candidate_ctx, product, Path(candidate_ctx["output_root"]))
        if not Path(ref_path).exists() or not Path(cand_path).exists():
            verdicts[product] = {"status": "unavailable", "reason": "product raster missing"}
            continue

        # Identical adjacency pairs require an identical grid for both chains
        # AND for the support rasters that define the edges.
        assert_same_grid([support_paths["scene_count_edge"], ref_path, cand_path])
        adjacency_identity[product] = {
            "support_raster": str(support_paths["scene_count_edge"]),
            "reference_path": str(ref_path),
            "candidate_path": str(cand_path),
            "grid_shared_with_support": True,
        }

        result = audit.audit_product_boundaries_windowed(
            product, ref_path, cand_path, support_paths,
            tmp_dir=tmp_dir / product,
            tile_seam_specs=None,  # no genuinely comparable paired tile partition exists
            provenance_status=provenance_state,
            provenance_code_spec=provenance_code_spec,
            resource_log=resource_log,
        )
        verdicts[product] = result["verdicts"]
        for boundary_type, verdict in result["verdicts"].items():
            rows.append(OrderedDict((
                ("product", product),
                ("boundary_type", boundary_type),
                ("status", verdict.get("status")),
                ("unit_type", verdict.get("unit_type")),
                ("paired_edge_count", verdict.get("n_pairs")),
                ("bootstrap_unit_count", verdict.get("n_units")),
                ("paired_reduction_point_estimate", verdict.get("point_estimate")),
                ("interval_low", verdict.get("interval_low")),
                ("interval_high", verdict.get("interval_high")),
                ("is_negative_control", verdict.get("is_negative_control", False)),
                ("can_affect_final_status", verdict.get("can_affect_final_status", True)),
                ("reduction_definition",
                 "absolute_jump_reference - absolute_jump_candidate "
                 "(positive => candidate has the smaller boundary jump)"),
            )))
        rows.extend(
            OrderedDict((("product", product), ("boundary_type", r.get("boundary_type")),
                         ("chain_metric_row", True), ("chain", r.get("chain")),
                         ("valid_pair_count", r.get("valid_pair_count")),
                         ("absolute_jump_median", r.get("absolute_jump_median")),
                         ("absolute_jump_p95", r.get("absolute_jump_p95")),
                         ("absolute_jump_p99", r.get("absolute_jump_p99")),
                         ("signed_jump_median", r.get("signed_jump_median"))))
            for r in result["metric_rows"]
        )

    return {
        "provenance_status": provenance_state,
        "provenance_evidence_is_metadata_derived": True,
        "export_tile_control": "unavailable_no_comparable_paired_partition",
        "adjacency_identity": adjacency_identity,
        "verdicts": verdicts,
        "rows": rows,
    }


BOUNDARY_PROPAGATION_COLUMNS = (
    "product", "boundary_type", "status", "unit_type", "paired_edge_count",
    "bootstrap_unit_count", "paired_reduction_point_estimate", "interval_low",
    "interval_high", "is_negative_control", "can_affect_final_status",
    "reduction_definition", "chain_metric_row", "chain", "valid_pair_count",
    "absolute_jump_median", "absolute_jump_p95", "absolute_jump_p99",
    "signed_jump_median",
)


def summarize_boundary_propagation(verdicts: dict) -> dict:
    """Which downstream products show a supported reduction / an increase.

    The export-tile negative control can never contribute positive evidence and
    is excluded from both lists.
    """
    supported: list[str] = []
    increased: list[str] = []
    per_product: "OrderedDict[str, dict]" = OrderedDict()

    for product, product_verdicts in verdicts.items():
        if not isinstance(product_verdicts, dict) or "status" in product_verdicts:
            per_product[product] = {"status": product_verdicts.get("status")
                                    if isinstance(product_verdicts, dict) else None}
            continue
        entry: "OrderedDict[str, str]" = OrderedDict()
        for boundary_type, verdict in product_verdicts.items():
            if verdict.get("can_affect_final_status") is False:
                entry[boundary_type] = f"{verdict.get('status')} (negative control; excluded)"
                continue
            entry[boundary_type] = verdict.get("status")
        per_product[product] = entry
        evidential = [
            v.get("status") for bt, v in product_verdicts.items()
            if v.get("can_affect_final_status") is not False
        ]
        if "supported_reduction" in evidential:
            supported.append(product)
        if "supported_increase" in evidential:
            increased.append(product)

    key = verdicts.get(KEY_STEP5_SEAM_PRODUCT) or {}
    key_verdict = key.get(KEY_BOUNDARY_TYPE, {}) if isinstance(key, dict) else {}
    return {
        "per_product": per_product,
        "supported_reduction_products": supported,
        "supported_increase_products": increased,
        "key_step5_product": KEY_STEP5_SEAM_PRODUCT,
        "key_boundary_type": KEY_BOUNDARY_TYPE,
        "key_step5_seam_status": key_verdict.get("status"),
        "key_step5_seam_reduction_supported":
            key_verdict.get("status") == "supported_reduction",
        "downstream_products_considered": [
            p for p in BOUNDARY_PROPAGATION_PRODUCTS if p != KEY_STEP5_SEAM_PRODUCT
        ],
        "downstream_supported_reduction_products": [
            p for p in supported if p != KEY_STEP5_SEAM_PRODUCT
        ],
        "downstream_supported_increase_products": [
            p for p in increased if p != KEY_STEP5_SEAM_PRODUCT
        ],
    }


# =============================================================================
# Predeclared decision logic (ORDERED -- never relaxed after seeing results)
# =============================================================================
def decide_final_status(evidence: dict) -> dict:
    """Apply the predeclared, ordered decision rule.

    Order: A invalid_reference_reproduction -> B baseline_invariance_failed ->
    C population_alignment_requires_review -> D seam_reduced_performance_tradeoff
    -> E eligible_for_second_aoi_validation -> F downstream_effect_inconclusive.

    `eligible_for_second_aoi_validation` means ONLY that independent validation
    in Bejis is warranted. It is never production acceptance and never a
    non-inferiority proof.
    """
    reasons: list[str] = []

    # --- A ---
    if evidence.get("reference_reproduction_status") != "pass":
        return _status(STATUS_INVALID_REFERENCE, [
            "the isolated reference chain did not reproduce the frozen canonical "
            "pipeline within the predeclared tolerances",
        ], evidence)

    # --- A2: shared-MODIS technical gate ---------------------------------
    # Ahead of every model-comparison status. `FINAL_STATUSES` is unchanged:
    # the failure surfaces as `baseline_invariance_failed` carrying the
    # dedicated technical field `shared_modis_invariance_failed`.
    modis_status = evidence.get("shared_modis_invariance_status")
    if modis_status is not None and modis_status != "pass":
        return _status(
            STATUS_BASELINE_INVARIANCE_FAILED,
            ["the two chains did not use identical MODIS inputs, identical "
             "aligned MODIS arrays and an identical MODIS compatibility mode: "
             f"{evidence.get('shared_modis_invariance_reasons')}"],
            evidence, technical_failure=TECHNICAL_FAILURE_SHARED_MODIS,
        )
    attestation_status = evidence.get("modis_compatibility_attestation_status")
    if evidence.get("modis_compatibility_required") and attestation_status != "pass":
        return _status(
            STATUS_BASELINE_INVARIANCE_FAILED,
            ["the historical MODIS compatibility attestation did not pass "
             f"(status={attestation_status!r}); no scientific conclusion may be "
             "issued for this run"],
            evidence, technical_failure=TECHNICAL_FAILURE_SHARED_MODIS,
        )

    # --- B ---
    if evidence.get("baseline_invariance_status") != "pass":
        return _status(STATUS_BASELINE_INVARIANCE_FAILED, [
            "the baseline chain differed despite the intended LST-only intervention",
        ], evidence)

    # --- C ---
    if evidence.get("population_alignment_status") != "ok":
        return _status(STATUS_POPULATION_REVIEW, [
            "row-set or positive-cell differences are large enough to prevent a "
            "credible common-cohort comparison: "
            f"{evidence.get('population_review_reasons')}",
        ], evidence)

    paired = evidence.get("paired_intervals") or {}
    roc = paired.get("roc_auc") or {}
    pr = paired.get("pr_auc") or {}
    brier = paired.get("brier") or {}
    candidate_support = evidence.get("candidate_thermal_support") or {}
    reference_support = evidence.get("reference_thermal_support") or {}

    seam_supported = bool(evidence.get("key_step5_seam_reduction_supported"))
    lost_support = bool(
        (reference_support.get("roc_auc_interval_above_zero")
         and not candidate_support.get("roc_auc_interval_above_zero"))
        or (reference_support.get("pr_auc_interval_above_zero")
            and not candidate_support.get("pr_auc_interval_above_zero"))
    )

    # --- D ---
    tradeoff_reasons: list[str] = []
    if roc.get("interval_wholly_below_zero"):
        tradeoff_reasons.append("candidate-minus-reference ROC-AUC interval is wholly below zero")
    if pr.get("interval_wholly_below_zero"):
        tradeoff_reasons.append("candidate-minus-reference PR-AUC interval is wholly below zero")
    if brier.get("interval_wholly_above_zero"):
        tradeoff_reasons.append("candidate-minus-reference Brier interval is wholly above zero")
    if lost_support:
        tradeoff_reasons.append(
            "thermal-minus-baseline support present in the reference is lost in the candidate"
        )
    if seam_supported and tradeoff_reasons:
        return _status(STATUS_SEAM_REDUCED_TRADEOFF, tradeoff_reasons, evidence)

    # --- E ---
    propagates = bool(evidence.get("downstream_supported_reduction_products"))
    no_contradiction = not bool(evidence.get("downstream_supported_increase_products"))
    eligible_checks = OrderedDict((
        ("reference_reproduction_passes", True),
        ("baseline_invariance_passes", True),
        ("common_cohort_valid", True),
        ("key_step5_seam_reduction_supported", seam_supported),
        ("propagates_to_at_least_one_downstream_thermal_product", propagates),
        ("no_contradictory_increase_across_key_product_chain", no_contradiction),
        ("candidate_thermal_roc_auc_interval_above_zero",
         bool(candidate_support.get("roc_auc_interval_above_zero"))),
        ("candidate_thermal_pr_auc_interval_above_zero",
         bool(candidate_support.get("pr_auc_interval_above_zero"))),
        ("paired_roc_auc_not_wholly_below_zero", not bool(roc.get("interval_wholly_below_zero"))),
        ("paired_pr_auc_not_wholly_below_zero", not bool(pr.get("interval_wholly_below_zero"))),
        ("paired_brier_not_wholly_above_zero", not bool(brier.get("interval_wholly_above_zero"))),
    ))
    if all(eligible_checks.values()):
        result = _status(STATUS_ELIGIBLE_SECOND_AOI, [
            "every predeclared eligibility condition is met",
        ], evidence)
        result["eligibility_checks"] = eligible_checks
        return result

    # --- F ---
    failed = [name for name, ok in eligible_checks.items() if not ok]
    reasons.append(
        "the run is valid but satisfies none of the stronger categories; unmet "
        f"eligibility conditions: {failed}"
    )
    result = _status(STATUS_INCONCLUSIVE, reasons, evidence)
    result["eligibility_checks"] = eligible_checks
    return result


def _status(
    status: str, reasons: list[str], evidence: dict, *,
    technical_failure: str | None = None,
) -> dict:
    if status not in FINAL_STATUSES:
        raise DownstreamABError(f"undeclared final status: {status!r}")
    return {
        "final_status": status,
        "decision_rule_version": DECISION_RULE_VERSION,
        "decision_rule_order": list(FINAL_STATUSES),
        "reasons": reasons,
        "evidence_snapshot": evidence,
        "production_approved": False,
        "technical_failure": technical_failure,
        "meaning": FINAL_STATUS_MEANINGS[status],
    }


FINAL_STATUS_MEANINGS = {
    STATUS_INVALID_REFERENCE:
        "The isolated reference chain does not reproduce the frozen canonical "
        "pipeline. No candidate scientific conclusion may be issued.",
    STATUS_BASELINE_INVARIANCE_FAILED:
        "The baseline chain changed although only Landsat LST was intervened on. "
        "The comparison is not a controlled A/B and no candidate conclusion is issued.",
    STATUS_POPULATION_REVIEW:
        "Row-set or positive-cell differences prevent a credible common-cohort "
        "comparison; the alignment needs review before any model claim.",
    STATUS_SEAM_REDUCED_TRADEOFF:
        "The seam reduction is supported but the candidate's within-region model "
        "performance is worse on at least one predeclared paired criterion.",
    STATUS_ELIGIBLE_SECOND_AOI:
        "Eligible for independent validation in Bejis ONLY. This is NOT production "
        "acceptance, NOT a production reducer change, and NOT a non-inferiority proof.",
    STATUS_INCONCLUSIVE:
        "A technically valid run whose downstream effect meets none of the stronger "
        "predeclared categories.",
}


# =============================================================================
# Limitations (required, verbatim scope statements)
# =============================================================================
def required_limitations() -> list[str]:
    return [
        "Single AOI: manavgat_2021 only.",
        "Single event window: one predictor/label window; no temporal replication.",
        "No cross-region validation is performed in this task.",
        "No production reducer change: the canonical Step3 compositing default is untouched.",
        "No causal claim: the comparison is a controlled substitution, not a causal identification.",
        "Metadata-derived path/row evidence is NOT pixel-level selected-scene provenance.",
        "Common-cohort evaluation may exclude cells that are valid in only one variant.",
        "Absence of a supported degradation is NOT proof of non-inferiority.",
        "Step8 metric preservation does NOT establish improved transfer to another region.",
        *legacy_modis_compatibility_limitations(),
    ]


# =============================================================================
# Reports
# =============================================================================
def summary_warnings(modis_compatibility: dict | None) -> list[dict]:
    """Machine-readable warnings for every report artefact.

    The legacy-MODIS warning is emitted whenever the compatibility path was
    used; it is never downgraded to a footnote.
    """
    warnings: list[dict] = []
    if (modis_compatibility or {}).get("required"):
        warnings.append(legacy_modis_compatibility_warning())
    return warnings


def build_modis_compatibility_report(
    modis_compatibility: dict | None, shared_modis_invariance: dict | None,
) -> dict:
    """The MODIS block carried by the summary and the manifest."""
    modis_compatibility = modis_compatibility or {}
    shared_modis_invariance = shared_modis_invariance or {}
    required = bool(modis_compatibility.get("required"))
    return OrderedDict((
        ("mode", modis_compatibility.get("mode", MODIS_STRICT_MODE)),
        ("historical_compatibility_required", required),
        ("attestation_status", modis_compatibility.get("status")),
        ("attestation_id", modis_compatibility.get("attestation_id")),
        ("declaration_path", modis_compatibility.get("declaration_path")),
        ("declaration_sha256", modis_compatibility.get("declaration_sha256")),
        ("attested_raster_sha256", OrderedDict(
            (feature, (entry or {}).get("sha256"))
            for feature, entry in (modis_compatibility.get("rasters") or {}).items()
        )),
        ("frozen_step7b_historical_evidence",
         modis_compatibility.get("frozen_step7b_historical_evidence")),
        ("applied_identically_to_both_chains", modis_compatibility.get("chains")),
        ("shared_modis_invariance_status", shared_modis_invariance.get("status")),
        ("shared_modis_invariance_checks", shared_modis_invariance.get("checks")),
        ("shared_modis_invariance_reasons", shared_modis_invariance.get("reasons")),
        ("default_step7b_guard_changed", False),
        ("rasters_rewritten", False),
        ("nodata_assigned", False),
        ("zero_converted_to_nan", False),
        ("values_or_mask_changed", False),
        ("declares_zero_scientifically_valid", False),
        ("modis_nodata_issue_resolved", False),
        ("warning", legacy_modis_compatibility_warning() if required else None),
        ("limitations", legacy_modis_compatibility_limitations()),
    ))


def build_summary(
    experiment_id: str, *, candidate: str, config: dict, provenance: dict,
    reproduction: dict, alignment: dict, fold_manifest: dict, baseline_invariance: dict,
    raster_change_rows: list[dict], boundary_summary: dict, boundary_result: dict,
    step8_metric_rows: list[dict], paired_rows: list[dict], bootstrap: dict,
    decision: dict, modis_compatibility: dict | None = None,
    shared_modis_invariance: dict | None = None,
) -> dict:
    """Assemble `downstream_ab_summary.json`."""
    modis_compatibility = modis_compatibility or {}
    shared_modis_invariance = shared_modis_invariance or {}
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", candidate),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("final_status", decision["final_status"]),
        ("final_status_meaning", decision["meaning"]),
        ("production_approved", False),
        ("changes_production_reducer", False),
        ("technical_failure", decision.get("technical_failure")),
        # Prominent, machine-readable: first-class key, not buried in a note.
        ("warnings", summary_warnings(modis_compatibility)),
        ("modis_compatibility", build_modis_compatibility_report(
            modis_compatibility, shared_modis_invariance,
        )),
        ("decision", decision),
        ("configuration", config),
        ("technical_validity", OrderedDict((
            ("reference_reproduction_status", reproduction["status"]),
            ("baseline_invariance_status", baseline_invariance["status"]),
            ("shared_modis_invariance_status", shared_modis_invariance.get("status")),
            ("shared_modis_technical_failure",
             shared_modis_invariance.get("technical_failure")),
            ("modis_compatibility_mode",
             modis_compatibility.get("mode", MODIS_STRICT_MODE)),
            ("modis_compatibility_attestation_status",
             modis_compatibility.get("status")),
            ("population_alignment_status", alignment["status"]),
            ("raw_lst_grid_equality_passed",
             provenance["raw_lst_grid_equality_gate"]["passed"]),
            ("candidate_audit_prerequisites_met",
             provenance["candidate_audit_provenance"]["prerequisites_met"]),
            ("candidate_modifies_lst_only", candidate_modifies_lst_only(provenance)),
            ("canonical_ndvi_identical_between_chains", ndvi_inputs_identical(provenance)),
            ("fold_assignment", fold_manifest),
        ))),
        ("raster_downstream_propagation", OrderedDict((
            ("raster_change_summary", raster_change_rows),
            ("boundary_propagation", boundary_summary),
            ("boundary_provenance_status", boundary_result["provenance_status"]),
            ("export_tile_control", boundary_result["export_tile_control"]),
        ))),
        ("within_region_model_impact", OrderedDict((
            ("primary_population", PRIMARY_POPULATION),
            ("cohort", "common_cohort"),
            ("per_chain_metrics", step8_metric_rows),
        ))),
        ("candidate_versus_reference_paired_comparison", OrderedDict((
            ("paired_rows", paired_rows),
            ("bootstrap_unit", bootstrap["bootstrap_unit"]),
            ("n_blocks", bootstrap["n_blocks"]),
            ("n_bootstrap_used", bootstrap["n_bootstrap_used"]),
            ("seed", bootstrap["seed"]),
            ("identical_block_draws_for_both_chains",
             bootstrap["identical_block_draws_for_both_chains"]),
            ("interval_language", bootstrap["interval_language"]),
            ("direction", bootstrap["direction"]),
        ))),
        ("limitations", required_limitations()),
        ("next_decision", next_decision_text(decision["final_status"])),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def next_decision_text(final_status: str) -> str:
    if final_status == STATUS_ELIGIBLE_SECOND_AOI:
        return (
            "Run the same controlled A/B in an independent AOI (bejis_2022) before "
            "any further consideration. Do NOT change the production reducer on the "
            "strength of a single AOI."
        )
    if final_status == STATUS_SEAM_REDUCED_TRADEOFF:
        return (
            "Do not pursue the candidate as a production reducer. Record the seam "
            "reduction alongside the measured within-region cost and, if the seam "
            "matters for a specific product, investigate a targeted remedy."
        )
    if final_status in (STATUS_INVALID_REFERENCE, STATUS_BASELINE_INVARIANCE_FAILED):
        return (
            "Repair the experiment before drawing any candidate conclusion: the "
            "control condition itself did not hold."
        )
    if final_status == STATUS_POPULATION_REVIEW:
        return (
            "Review the row-set differences reported in population_alignment.json "
            "before any model-level claim is made."
        )
    return (
        "No stronger category is supported. Treat the downstream effect as "
        "unresolved; a second AOI is not yet warranted on this evidence alone."
    )


def render_warning_block(summary: dict) -> list[str]:
    """The prominent warning banner, rendered directly under the final status."""
    warnings = summary.get("warnings") or []
    if not warnings:
        return []
    lines = ["## WARNINGS", ""]
    for warning in warnings:
        lines.append(f"> **`{warning['code']}`**")
        lines.append(">")
        lines.append(f"> {warning['statement']}")
        lines.append(">")
        lines.append(f"> **Scientific effect.** {warning['scientific_effect']}")
        lines.append("")
    modis = summary.get("modis_compatibility") or {}
    if modis.get("historical_compatibility_required"):
        lines.append(
            f"MODIS compatibility mode: `{modis.get('mode')}` "
            f"(attestation `{modis.get('attestation_status')}`, applied identically "
            f"to both chains: `{modis.get('applied_identically_to_both_chains')}`). "
            "No MODIS raster value, mask, dtype or grid was changed and the default "
            "Step7B zero-fill guard is untouched for every other caller."
        )
        lines.append("")
        lines.append(
            "This run does NOT resolve the MODIS nodata issue "
            f"(`modis_nodata_issue_resolved: {modis.get('modis_nodata_issue_resolved')}`)."
        )
        lines.append("")
    return lines


def render_summary_markdown(summary: dict) -> str:
    """`downstream_ab_summary.md` with the six required sections."""
    lines: list[str] = []
    add = lines.append

    add(f"# Landsat compositing downstream A/B -- {summary['experiment_id']}")
    add("")
    add(f"- Reference chain: `{summary['reference_chain']}`")
    add(f"- Candidate chain: `{summary['candidate_chain']}`")
    add(f"- Report schema: `{summary['report_schema_version']}`")
    add(f"- Decision rule: `{summary['decision_rule_version']}`")
    add(f"- **Final status: `{summary['final_status']}`**")
    add("")
    add(f"> {summary['final_status_meaning']}")
    add("")
    add("This is a diagnostic candidate experiment. It does NOT change the default "
        "production reducer and it can never return a production approval.")
    add("")
    lines.extend(render_warning_block(summary))

    add("## 1. Technical validity")
    add("")
    validity = summary["technical_validity"]
    for key, value in validity.items():
        if key == "fold_assignment":
            continue
        add(f"- `{key}`: `{value}`")
    folds = validity["fold_assignment"]
    add(f"- fold assignment: seed `{folds.get('seed')}`, "
        f"{folds.get('n_splits')} folds, block size `{folds.get('block_size_cells')}` cells, "
        f"grouping `{folds.get('grouping')}`")
    add(f"- fold assignment identical across chains: `{folds.get('chains_identical')}`")
    add("")

    add("## 2. Raster-level downstream propagation")
    add("")
    add("Candidate minus reference, common valid pixels only. Tiny float32 "
        "differences are reported descriptively and are not treated as scientific change.")
    add("")
    add("| product | common valid px | mean | MAE | RMSE | max abs | changed frac (thr) |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["raster_downstream_propagation"]["raster_change_summary"]:
        add("| {p} | {n} | {mean} | {mae} | {rmse} | {mx} | {cf} ({thr}) |".format(
            p=row.get("product"), n=row.get("common_valid_pixels"),
            mean=_fmt(row.get("mean")), mae=_fmt(row.get("mae")),
            rmse=_fmt(row.get("rmse")), mx=_fmt(row.get("max_abs_diff")),
            cf=_fmt(row.get("changed_pixel_fraction")),
            thr=row.get("changed_pixel_threshold"),
        ))
    add("")
    boundary = summary["raster_downstream_propagation"]["boundary_propagation"]
    add(f"Key Step5 seam product `{boundary['key_step5_product']}` at "
        f"`{boundary['key_boundary_type']}`: `{boundary['key_step5_seam_status']}`.")
    add("")
    add("Downstream products with a supported reduction: "
        f"`{boundary['downstream_supported_reduction_products']}`.")
    add("Downstream products with a supported increase (contradictory): "
        f"`{boundary['downstream_supported_increase_products']}`.")
    add("")
    add("A positive paired reduction means the CANDIDATE has the smaller boundary jump. "
        "The export-tile partition is a negative control and can never create positive "
        f"evidence; here it is `{summary['raster_downstream_propagation']['export_tile_control']}`.")
    add("")

    add("## 3. Within-region model impact")
    add("")
    add(f"Primary population: `{summary['within_region_model_impact']['primary_population']}` "
        "on the frozen common cohort.")
    add("")
    add("| chain | n | pos | thermal ROC-AUC | thermal PR-AUC | thermal Brier | "
        "d ROC-AUC (thermal-baseline) [95% interval] |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in summary["within_region_model_impact"]["per_chain_metrics"]:
        add("| {c} | {n} | {p} | {ra} | {pa} | {br} | {d} [{lo}, {hi}] |".format(
            c=row["chain"], n=row["n_rows"], p=row["n_positives"],
            ra=_fmt(row["thermal_roc_auc"]), pa=_fmt(row["thermal_pr_auc"]),
            br=_fmt(row["thermal_brier"]),
            d=_fmt(row["delta_roc_auc_thermal_minus_baseline"]),
            lo=_fmt(row["delta_roc_auc_interval_low"]),
            hi=_fmt(row["delta_roc_auc_interval_high"]),
        ))
    add("")

    add("## 4. Candidate-versus-reference paired comparison")
    add("")
    paired = summary["candidate_versus_reference_paired_comparison"]
    add(f"Spatial-block bootstrap over `{paired['bootstrap_unit']}` "
        f"({paired['n_blocks']} blocks, {paired['n_bootstrap_used']} replicates, "
        f"seed {paired['seed']}). Both chains are scored on IDENTICAL block draws: "
        f"`{paired['identical_block_draws_for_both_chains']}`.")
    add("")
    add("| metric | point | interval | excludes zero | improvement direction |")
    add("| --- | ---: | --- | --- | --- |")
    for row in paired["paired_rows"]:
        add("| {m} | {pt} | [{lo}, {hi}] | {ex} | {d} |".format(
            m=row["metric"], pt=_fmt(row["point_estimate"]),
            lo=_fmt(row["interval_low"]), hi=_fmt(row["interval_high"]),
            ex=row["interval_excludes_zero"], d=row["improvement_direction"],
        ))
    add("")
    add("Sign convention: for ROC-AUC and PR-AUC `candidate_minus_reference > 0` means "
        "improvement; for Brier `candidate_minus_reference < 0` means improvement. "
        "Intervals are described as excluding or including zero -- never as "
        "'statistically significant'.")
    add("")

    add("## 5. Limitations")
    add("")
    for item in summary["limitations"]:
        add(f"- {item}")
    add("")

    add("## 6. Next decision")
    add("")
    add(summary["next_decision"])
    add("")
    if summary["final_status"] == STATUS_ELIGIBLE_SECOND_AOI:
        add("`eligible_for_second_aoi_validation` means ONLY that independent validation "
            "in Bejis is warranted. It is NOT production approval, NOT a production "
            "reducer change, and NOT a non-inferiority proof.")
        add("")
        if (summary.get("modis_compatibility") or {}).get(
            "historical_compatibility_required"
        ):
            add("It also does NOT mean the MODIS nodata issue is resolved: this result "
                "is conditional on the frozen historical zero-filled MODIS "
                "representation, and a MODIS nodata repair must be evaluated as a "
                "separate experiment.")
            add("")
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


#: Sub-trees excluded from the manifest hash sweep. `inputs/` is already hashed
#: file-by-file in input_provenance.json (re-hashing ~1 GB of copied raw inputs
#: would add nothing), and `_analysis_tmp` is scratch that is deleted on success.
MANIFEST_EXCLUDED_SUBTREES = ("inputs", "_analysis_tmp")


def manifest_candidate_files(root: Path) -> list[Path]:
    """Produced files eligible for the manifest hash sweep, deterministically ordered."""
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and not set(p.relative_to(root).parts) & set(MANIFEST_EXCLUDED_SUBTREES)
    )


def build_manifest(experiment_id: str, root: Path, summary: dict) -> dict:
    """`downstream_ab_manifest.json`: every produced file with size + sha256."""
    root = Path(root)
    manifest_files = audit.build_file_manifest(
        manifest_candidate_files(root), output_dir=root,
    )["files"]
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("output_root", str(root)),
        ("final_status", summary["final_status"]),
        ("production_approved", False),
        ("changes_production_reducer", False),
        ("technical_failure", summary.get("technical_failure")),
        ("warnings", summary.get("warnings") or []),
        ("modis_compatibility", summary.get("modis_compatibility")),
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", summary["candidate_chain"]),
        ("file_count", len(manifest_files)),
        ("files", manifest_files),
        ("excluded_subtrees", list(MANIFEST_EXCLUDED_SUBTREES)),
        ("raw_input_hashes_recorded_in", "input_provenance.json"),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def render_pair_maps_for_product(
    reference_path: Path, candidate_path: Path, out_dir: Path, *, product: str,
) -> list[str]:
    """Reference and candidate side by side under ONE shared display range.

    Delegates to the counterfactual audit's `render_pair_maps` so both chains
    share a single robust stretch and neither raster is smoothed
    (nearest-neighbour rendering only).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `render_pair_maps` names each PNG after its source raster's stem. The two
    # chains produce IDENTICALLY named rasters in different directories, so they
    # are first exposed under chain-distinct names (symlinks; a copy only if the
    # filesystem refuses one) to keep the two panels from colliding. The rasters
    # themselves are never modified.
    staging = out_dir / "_pair_inputs"
    staging.mkdir(parents=True, exist_ok=True)
    staged: "OrderedDict[str, Path]" = OrderedDict()
    for chain, source in ((CHAIN_REFERENCE, reference_path), (CHAIN_CANDIDATE, candidate_path)):
        target = staging / f"{product}__{chain}.tif"
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(Path(source).resolve())
        except OSError:
            shutil.copy2(str(source), str(target))
        staged[chain] = target

    written = audit.render_pair_maps(
        staged[CHAIN_REFERENCE], staged[CHAIN_CANDIDATE], out_dir, pair_name=product,
    )
    shutil.rmtree(staging, ignore_errors=True)
    return written


def report_generation_preserves_metrics(before: dict, after: dict) -> bool:
    """Report generation must never alter a scientific number."""
    return json.dumps(before, sort_keys=True, default=str) == json.dumps(
        after, sort_keys=True, default=str
    )


# =============================================================================
# Configuration snapshot (dry-run and live)
# =============================================================================
def build_config_snapshot(experiment_id: str, candidate: str, ctx: dict) -> dict:
    """Frozen model/block/bootstrap configuration used by both chains."""
    from core.config import (
        STEP5_MIN_BASELINE_STD_CELSIUS, STEP5_MIN_BASELINE_VALID_COUNT,
        STEP5_MIN_CURRENT_VALID_COUNT, STEP8A_MIN_30M_VALID_FRACTION,
        STEP8B_MIN_POSITIVES_PER_POPULATION, STEP8B_N_SPLITS, STEP8B_RANDOM_SEED,
        STEP8B_SPATIAL_BLOCK_SIZE_CELLS, STEP8C_RANDOM_SEED,
    )
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, THERMAL_MODEL_FEATURES,
    )

    return OrderedDict((
        ("experiment_id", experiment_id),
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", candidate),
        ("primary_population", PRIMARY_POPULATION),
        ("predictor_window", [ctx["predictor_start_date"], ctx["predictor_end_date"]]),
        ("label_window", [ctx["label_start_date"], ctx["label_end_date"]]),
        ("baseline_years", list(ctx["baseline_years"])),
        ("current_period_days", ctx["current_period_days"]),
        ("model", OrderedDict((
            ("name", "random_forest"),
            ("source", "src.step8b_train_baseline_vs_thermal_model.build_classifier"),
            ("baseline_features", list(BASELINE_FEATURES)),
            ("thermal_features", list(THERMAL_MODEL_FEATURES)),
            ("random_seed", STEP8B_RANDOM_SEED),
            ("tuned_for_this_experiment", False),
        ))),
        ("spatial_blocks", OrderedDict((
            ("block_size_cells", STEP8B_SPATIAL_BLOCK_SIZE_CELLS),
            ("n_splits", STEP8B_N_SPLITS),
            ("seed", STEP8B_RANDOM_SEED),
            ("splitter", "StratifiedGroupKFold(shuffle=True)"),
            ("grouping", "spatial_block_id"),
            ("random_row_split_possible", False),
            ("min_positives_per_population", STEP8B_MIN_POSITIVES_PER_POPULATION),
            ("large_block_robustness_included", False),
        ))),
        ("bootstrap", OrderedDict((
            ("unit", "spatial_block_id"),
            ("replicates", PAIRED_BOOTSTRAP_REPLICATES),
            ("seed", STEP8C_RANDOM_SEED),
            ("ci_lower_percentile", PAIRED_BOOTSTRAP_CI_LOWER),
            ("ci_upper_percentile", PAIRED_BOOTSTRAP_CI_UPPER),
            ("identical_block_draws_for_both_chains", True),
        ))),
        ("step5_policy", OrderedDict((
            ("min_baseline_valid_count", STEP5_MIN_BASELINE_VALID_COUNT),
            ("min_baseline_std_celsius", STEP5_MIN_BASELINE_STD_CELSIUS),
            ("min_current_valid_count", STEP5_MIN_CURRENT_VALID_COUNT),
        ))),
        ("step8a_policy", OrderedDict((
            ("min_30m_valid_fraction", STEP8A_MIN_30M_VALID_FRACTION),
        ))),
        ("observation_support_semantics", OrderedDict((
            ("reference", REFERENCE_CURRENT_COUNT_SEMANTICS),
            ("candidate", CANDIDATE_CURRENT_COUNT_SEMANTICS),
        ))),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
    ))


# =============================================================================
# Planned stages
# =============================================================================
PLANNED_STAGES = (
    "validate_inputs",
    "materialize_inputs",
    "reference_step5",
    "reference_step5c",
    "candidate_step5",
    "candidate_step5c",
    # Inserted BEFORE any Step7B call: the historical-MODIS attestation must
    # pass before either chain builds a downscaling dataset.
    "modis_compatibility_attestation",
    "reference_step7a", "reference_step7b", "reference_step7c",
    "reference_step7d", "reference_step7e",
    "candidate_step7a", "candidate_step7b", "candidate_step7c",
    "candidate_step7d", "candidate_step7e",
    "reference_step8a",
    "candidate_step8a",
    "reference_reproduction",
    "population_alignment",
    "fold_assignment",
    "reference_step8_model",
    "candidate_step8_model",
    "paired_bootstrap",
    "raster_comparison",
    "boundary_propagation",
    "report_generation",
)

#: Every stage from Step7B onward. These consume MODIS through Step7B, so they
#: are invalidated by a checkpoint-schema bump or by a different MODIS
#: attestation. Input validation, materialization, Step5, Step5C and Step7A are
#: MODIS-independent and stay reusable after their own file validation.
MODIS_DEPENDENT_STAGES = tuple(
    stage for stage in PLANNED_STAGES
    if stage not in (
        "validate_inputs", "materialize_inputs",
        "reference_step5", "reference_step5c",
        "candidate_step5", "candidate_step5c",
        "modis_compatibility_attestation",
        "reference_step7a", "candidate_step7a",
    )
)


# =============================================================================
# Checkpointing (atomic; checkpoint TEXT never bypasses file validation)
# =============================================================================
CHECKPOINT_FILENAME = "downstream_ab_checkpoint.json"

#: Bumped when the MODIS compatibility attestation stage was introduced. A
#: checkpoint written by an older schema keeps its Step5/Step5C/Step7A stages
#: but can no longer reuse Step7B or anything after it.
CHECKPOINT_SCHEMA_VERSION = "2.0-modis-compatibility"


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
    path = Path(path)
    return {"path": str(path), "bytes": int(path.stat().st_size) if path.exists() else -1}


def write_checkpoint_stage(
    root: Path, stage: str, outputs, extra: dict | None = None,
    attestation: dict | None = None,
) -> dict:
    """Atomically record a completed stage together with its output signatures.

    The recorded byte sizes are what `--resume` re-validates; the stage name
    alone is never sufficient to reuse a stage. Step7B-and-later stages
    additionally record the MODIS attestation binding they were produced under,
    which resume re-checks against the freshly derived attestation.
    """
    if stage not in PLANNED_STAGES:
        raise DownstreamABError(f"unknown checkpoint stage: {stage!r}")
    root = Path(root)
    payload = read_checkpoint(root)
    payload.setdefault("experiment", DIAGNOSTIC_NAMESPACE)
    payload["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    payload.setdefault("stages", {})
    entry = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "outputs": [file_reference(p) for p in outputs],
        **(extra or {}),
    }
    if attestation is not None:
        entry["modis_attestation_binding"] = attestation_binding(attestation)
    payload["stages"][stage] = entry
    payload["last_stage"] = stage
    write_json_atomic(checkpoint_path(root), payload)
    return payload


def checkpoint_schema_version(root: Path) -> str | None:
    return read_checkpoint(root).get("checkpoint_schema_version")


def stage_is_reusable(
    root: Path, stage: str, attestation: dict | None = None,
) -> bool:
    """A stage may be reused ONLY when everything about it still validates.

    Three independent conditions, all required:
      1. the stage's recorded output files still exist at their recorded sizes;
      2. for Step7B-and-later stages, the checkpoint was written by the current
         schema -- an older checkpoint invalidates Step7B onward and nothing
         earlier, so no `--force` is ever needed;
      3. for Step7B-and-later stages, the recorded MODIS attestation binding
         equals the binding derived in THIS run.
    """
    checkpoint = read_checkpoint(root)
    entry = (checkpoint.get("stages") or {}).get(stage)
    if not entry:
        return False
    if not files_present_and_signed(entry.get("outputs") or []):
        return False
    if stage not in MODIS_DEPENDENT_STAGES:
        return True
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return False
    recorded = (entry.get("modis_attestation_binding") or {}).get("binding_sha256")
    expected = attestation_binding(attestation or {})["binding_sha256"]
    return recorded == expected


# =============================================================================
# Dry-run plan
# =============================================================================
def build_dry_run_modis_compatibility(
    experiment_id: str, base_dir: Path = PROJECT_ROOT,
) -> dict:
    """Report whether the historical-MODIS path will be required. Writes NOTHING."""
    detection = modis_compatibility_required(experiment_id, base_dir)
    config_path = legacy_modis_attestation_config_path(base_dir)
    declaration_present = config_path.exists()
    historical = frozen_step7b_historical_evidence(experiment_id, base_dir)
    mean = (detection["rasters"] or {}).get("modis_lst_mean_celsius") or {}

    return OrderedDict((
        ("mode_name", LEGACY_MODIS_COMPATIBILITY_MODE),
        ("default_mode", MODIS_STRICT_MODE),
        ("historical_compatibility_required", detection["required"]),
        ("reason",
         "the frozen MODIS mean declares no nodata and "
         f"{float(mean.get('exact_zero_fraction') or 0.0) * 100:.1f}% of its pixels "
         "are exactly 0.0, which the default Step7B guard rejects"
         if detection["required"] else
         "the default Step7B zero-fill guard accepts the frozen MODIS inputs"),
        ("authorized_experiment_ids", list(LEGACY_MODIS_COMPATIBILITY_EXPERIMENT_IDS)),
        ("experiment_is_authorized",
         experiment_id in LEGACY_MODIS_COMPATIBILITY_EXPERIMENT_IDS),
        ("zero_fill_guard_threshold", detection["zero_fill_guard_threshold"]),
        ("frozen_modis_namespace", str(frozen_modis_dir(experiment_id, base_dir))),
        ("frozen_modis_evidence", detection["rasters"]),
        ("attestation_declaration_path", str(config_path)),
        ("attestation_declaration_present", declaration_present),
        ("frozen_step7b_historical_evidence", historical),
        ("historical_step7b_evidence_confirmed",
         historical_evidence_confirms_no_nodata_source(historical)),
        ("default_step7b_guard_changed", False),
        ("warning", legacy_modis_compatibility_warning() if detection["required"] else None),
        ("limitations", legacy_modis_compatibility_limitations()),
        ("writes_performed", False),
    ))


def build_dry_run_plan(
    experiment_id: str, candidate: str, base_dir: Path = PROJECT_ROOT,
) -> dict:
    """Resolve everything a dry-run must print. Creates and writes NOTHING."""
    from core.experiment_context import build_experiment_context

    if candidate not in SUPPORTED_CANDIDATES:
        raise DownstreamABError(
            f"unsupported candidate {candidate!r}. Supported: {SUPPORTED_CANDIDATES}. "
            "A `date_balanced_all_landsat` variant is deliberately NOT implemented "
            "in this task."
        )

    ctx = build_experiment_context(experiment_id)
    state = load_source_audit_state(experiment_id, base_dir)
    plan = build_input_plan(ctx, experiment_id, base_dir)
    layout = plan_output_layout(experiment_id, base_dir)
    expected = plan_expected_files(experiment_id, base_dir)

    reference_sources: "OrderedDict[str, str]" = OrderedDict()
    candidate_sources: "OrderedDict[str, str]" = OrderedDict()
    for role, entry in plan.items():
        reference_sources[role] = str(entry["reference_source"])
        candidate_sources[role] = str(entry["candidate_source"])
        if entry.get("candidate_count_source"):
            candidate_sources[f"{role}__observation_count"] = str(entry["candidate_count_source"])

    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", candidate),
        ("reference_source_paths", reference_sources),
        ("candidate_source_paths", candidate_sources),
        ("audit_prerequisite_status", OrderedDict((
            ("source_root", state["source_root"]),
            ("present", state["present"]),
            ("final_status", state["final_status"]),
            ("required_final_status", REQUIRED_SOURCE_FINAL_STATUS),
            ("canonical_reproduction_status", state["canonical_reproduction_status"]),
            ("required_canonical_reproduction", REQUIRED_SOURCE_CANONICAL_REPRODUCTION),
            ("report_schema_version", state["report_schema_version"]),
            ("audit_file_hashes", state["audit_file_hashes"]),
            ("prerequisites_met", state["prerequisites_met"]),
        ))),
        ("missing_sources", missing_plan_sources(plan)),
        ("modis_compatibility", build_dry_run_modis_compatibility(experiment_id, base_dir)),
        ("output_namespace", str(layout["root"])),
        ("output_layout", OrderedDict((k, str(v)) for k, v in layout.items())),
        ("planned_stages", list(PLANNED_STAGES)),
        ("expected_files", OrderedDict((k, str(v)) for k, v in expected.items())),
        ("configuration", build_config_snapshot(experiment_id, candidate, ctx)),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("allowed_final_statuses", list(FINAL_STATUSES)),
        ("writes_performed", False),
        ("earth_engine_calls", 0),
        ("models_run", 0),
        ("directories_created", 0),
    ))
