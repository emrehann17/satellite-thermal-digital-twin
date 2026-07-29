"""
landsat_harmonization_downstream_ab.py

Isolated DOWNSTREAM A/B experiment for the current-period acquisition-date
offset harmonization candidate.

    reference: date_balanced_reference
    candidate: overlap_harmonized_date_balanced

QUESTION
--------
Does current-period date-offset harmonization propagate through Step5, Step5C,
Step7 and Step8 without weakening within-region thermal performance?

WHAT DIFFERS BETWEEN THE TWO CHAINS
-----------------------------------
Exactly one thing: **band 1 of the current-period Landsat LST raster**. Band 2
(the unique-acquisition-date support count) is required to be BITWISE identical
between the chains, and that requirement is a hard gate, not an expectation --
the harmonization candidate was built by subtracting one additive scalar per
acquisition date, so per-pixel date support cannot have moved. If it did move,
this experiment stops at `support_invariance_failed`.

Everything else is ONE materialized copy shared by both chains: the frozen
date-balanced annual Landsat baselines (taken from the previous downstream A/B
candidate bundle, so the baseline climatology is byte-identical to the run that
produced the reference), NDVI, MODIS, DEM/slope, land cover, MCD64A1 labels,
thresholds, features, model parameters and seeds. Both chain contexts share
`baseline_input_dir` and differ only in `current_period_dir`.

WHAT THIS MODULE IS NOT
-----------------------
    - It is NOT a production decision and never returns one. `seam_fixed`,
      `production_approved`, `production_ready` and `approved_for_production`
      are unreachable.
    - It NEVER claims non-inferiority, transfer improvement, cross-region
      generalization or causality.
    - It NEVER runs Earth Engine. Every input is a frozen local file, and the
      live run installs the shared `EarthEngineGuard` for its whole duration.
    - It NEVER changes production Step5/Step7/Step8 code, `core.config`, the
      production reducer, or the Step7B MODIS guard default.
    - The old scene-weighted canonical chain is NOT the reference here. The
      reference is the previous A/B's CANDIDATE chain, reproduced from the
      harmonization experiment's own frozen reference raster.

RELATIONSHIP TO THE EXISTING A/B
--------------------------------
`src/landsat_composite_downstream_ab.py` is the implementation template and is
imported, not copied: the common cohort, shared spatial folds, Step8
baseline/thermal models, baseline invariance, paired spatial-block bootstrap,
raster comparisons, maps, checkpointing/resume, MODIS compatibility machinery
and report rendering are all reused from it verbatim. Only the identity, the
input contract, the prerequisites, the reference-reproduction target, the
support-invariance gate and the key boundary type are redefined here.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import src.landsat_composite_counterfactual_audit as audit
import src.landsat_composite_downstream_ab as ab
import src.landsat_current_support_harmonization as hz
from core.paths import PROJECT_ROOT

# -----------------------------------------------------------------------------
# Shared primitives -- ONE implementation, ONE contract. Everything below is the
# EXACT machinery the previous downstream A/B used, so the two experiments stay
# comparable and no scientific computation is re-implemented here.
# -----------------------------------------------------------------------------
write_json_atomic = audit.write_json_atomic
sha256_and_size = audit.sha256_and_size
assert_same_grid = audit.assert_same_grid
grid_signature = audit.grid_signature

EarthEngineGuard = ab.EarthEngineGuard
FORBIDDEN_EE_CALLABLES = ab.FORBIDDEN_EE_CALLABLES

raster_signature = ab.raster_signature
grids_equal = ab.grids_equal
compare_raster_semantic = ab.compare_raster_semantic
compare_raster_change = ab.compare_raster_change
compare_reference_step8_to_canonical = ab.compare_reference_step8_to_canonical
compared_raster_products = ab.compared_raster_products
product_path = ab.product_path
build_current_minus_baseline = ab.build_current_minus_baseline

build_common_cohort = ab.build_common_cohort
build_fold_assignment = ab.build_fold_assignment
assert_identical_fold_assignment = ab.assert_identical_fold_assignment
run_chain_model = ab.run_chain_model
check_baseline_invariance = ab.check_baseline_invariance
paired_block_bootstrap = ab.paired_block_bootstrap

build_paired_bootstrap_rows = ab.build_paired_bootstrap_rows
build_oof_predictions = ab.build_oof_predictions
metric_improved = ab.metric_improved
eligible_rows = ab.eligible_rows

boundary_support_paths = ab.boundary_support_paths
frozen_provenance_state = ab.frozen_provenance_state

############################################################################################################################
def build_step8_metric_rows(
    reference_result: dict,
    candidate_result: dict,
    intervals: dict,
) -> list[dict]:
    rows = ab.build_step8_metric_rows(
        reference_result,
        candidate_result,
        intervals,
    )
    for row in rows:
        old = row.get("chain")
        row["chain"] = _OLD_TO_NEW_CHAIN.get(old, old)
    return rows


def run_boundary_propagation(*args, **kwargs) -> dict:
    result = ab.run_boundary_propagation(*args, **kwargs)

    for row in result.get("rows") or []:
        old = row.get("chain")
        row["chain"] = _OLD_TO_NEW_CHAIN.get(old, old)

    return result
############################################################################################################################


# --- MODIS compatibility: reused wholesale, ISSUER DELIBERATELY UNCHANGED -----
# The historical attestation was issued by, and remains scoped to, the previous
# experiment. Re-badging it under this namespace would misrepresent which run
# performed the historical verification, so the issuer stays
# `landsat_composite_downstream_ab`.
MODIS_ATTESTATION_ISSUER = ab.DIAGNOSTIC_NAMESPACE
modis_compatibility_required = ab.modis_compatibility_required

# MODIS compatibility wrappers are defined after build_input_provenance.

step7b_compatibility_attestation = ab.step7b_compatibility_attestation
build_modis_compatibility_report = ab.build_modis_compatibility_report
legacy_modis_compatibility_limitations = ab.legacy_modis_compatibility_limitations
summary_warnings = ab.summary_warnings
MODIS_STRICT_MODE = ab.MODIS_STRICT_MODE
LEGACY_MODIS_COMPATIBILITY_MODE = ab.LEGACY_MODIS_COMPATIBILITY_MODE
TECHNICAL_FAILURE_SHARED_MODIS = ab.TECHNICAL_FAILURE_SHARED_MODIS

# --- Comparison / reporting constants reused verbatim ------------------------
REPRODUCTION_TOLERANCES = ab.REPRODUCTION_TOLERANCES
REPRODUCTION_MIN_MASK_AGREEMENT = ab.REPRODUCTION_MIN_MASK_AGREEMENT
REPRODUCTION_STEP8_METRIC_TOL = ab.REPRODUCTION_STEP8_METRIC_TOL
CHANGED_PIXEL_THRESHOLDS = ab.CHANGED_PIXEL_THRESHOLDS
RASTER_CHANGE_COLUMNS = ab.RASTER_CHANGE_COLUMNS
BOUNDARY_PROPAGATION_COLUMNS = ab.BOUNDARY_PROPAGATION_COLUMNS
BOUNDARY_PROPAGATION_PRODUCTS = ab.BOUNDARY_PROPAGATION_PRODUCTS
BOUNDARY_TYPES = ab.BOUNDARY_TYPES
DERIVED_SUBDIR = ab.DERIVED_SUBDIR
DERIVED_PRODUCTS = ab.DERIVED_PRODUCTS

PRIMARY_POPULATION = ab.PRIMARY_POPULATION
PAIRED_BOOTSTRAP_REPLICATES = ab.PAIRED_BOOTSTRAP_REPLICATES
PAIRED_BOOTSTRAP_CI_LOWER = ab.PAIRED_BOOTSTRAP_CI_LOWER
PAIRED_BOOTSTRAP_CI_UPPER = ab.PAIRED_BOOTSTRAP_CI_UPPER
BASELINE_OOF_MAX_ABS_DIFF = ab.BASELINE_OOF_MAX_ABS_DIFF
BASELINE_FEATURE_MAX_ABS_DIFF = ab.BASELINE_FEATURE_MAX_ABS_DIFF
MIN_COMMON_ROW_RETENTION = ab.MIN_COMMON_ROW_RETENTION
MIN_COMMON_POSITIVE_RETENTION = ab.MIN_COMMON_POSITIVE_RETENTION


class HarmonizationDownstreamABError(RuntimeError):
    """Fail-fast error for the harmonization downstream A/B experiment."""


class PrerequisiteError(HarmonizationDownstreamABError):
    """A required frozen input or upstream prerequisite is missing/invalid."""


class SupportInvarianceError(HarmonizationDownstreamABError):
    """The two chains do not share an identical per-pixel date support."""


# =============================================================================
# Identity
# =============================================================================
DIAGNOSTIC_NAMESPACE = "landsat_harmonization_downstream_ab"

#: Upstream namespaces. All THREE are read-only inputs.
HARMONIZATION_NAMESPACE = hz.DIAGNOSTIC_NAMESPACE
PREVIOUS_AB_NAMESPACE = ab.DIAGNOSTIC_NAMESPACE
COUNTERFACTUAL_NAMESPACE = audit.DIAGNOSTIC_NAMESPACE

CHAIN_REFERENCE = "date_balanced_reference"
CHAIN_CANDIDATE = "overlap_harmonized_date_balanced"
CHAINS = (CHAIN_REFERENCE, CHAIN_CANDIDATE)

CHAIN_SIDE = OrderedDict((
    (CHAIN_REFERENCE, "reference"),
    (CHAIN_CANDIDATE, "candidate"),
))

#: The chain of the PREVIOUS A/B that this experiment's reference must
#: reproduce. It is the previous CANDIDATE, never the canonical scene-weighted
#: chain.
PREVIOUS_AB_REFERENCE_SIDE = ab.CHAIN_SIDE[ab.CHAIN_CANDIDATE]      # "candidate"
PREVIOUS_AB_REFERENCE_CHAIN = ab.CHAIN_CANDIDATE                    # date_balanced_lst_only

REPORT_SCHEMA_VERSION = "1.0-harmonization-downstream-ab"
DECISION_RULE_VERSION = "1.0-harmonization-downstream-ab-ordered"

#: Only one candidate is reachable; another needs its own predeclaration.
SUPPORTED_CANDIDATES = (CHAIN_CANDIDATE,)
SUPPORTED_EXPERIMENT_IDS = ("manavgat_2021",)

_OLD_TO_NEW_CHAIN = {
    ab.CHAIN_REFERENCE: CHAIN_REFERENCE,
    ab.CHAIN_CANDIDATE: CHAIN_CANDIDATE,
    "scene_weighted": CHAIN_REFERENCE,
    "date_balanced": CHAIN_CANDIDATE,
}

_OLD_TO_NEW_CHAIN = {
    ab.CHAIN_REFERENCE: CHAIN_REFERENCE,
    ab.CHAIN_CANDIDATE: CHAIN_CANDIDATE,
    "scene_weighted": CHAIN_REFERENCE,
    "date_balanced": CHAIN_CANDIDATE,
}


# =============================================================================
# Chain label mapping (this experiment's names, never the previous experiment's)
# =============================================================================
#: The previous A/B's chain names are structurally meaningful here -- its
#: reference slot is this experiment's reference slot, and its candidate slot is
#: this experiment's candidate slot -- but its LABELS belong to that experiment.
#: Every artefact this experiment writes carries THIS experiment's names.
#: Built from the module's own `_OLD_TO_NEW_CHAIN` so there is ONE mapping, and
#: ordered longest-key-first so `scene_weighted_reference` is rewritten before
#: the shorter `scene_weighted` can corrupt it.
CHAIN_LABEL_MAP = OrderedDict(
    sorted(_OLD_TO_NEW_CHAIN.items(), key=lambda kv: -len(kv[0]))
)


#: The FULL previous-experiment chain names. Only these are rewritten inside
#: free text. The short forms (`scene_weighted`, `date_balanced`) are exact-match
#: keys only: they are substrings of this experiment's OWN names
#: (`date_balanced_reference`), so a substring rewrite would corrupt them.
STALE_CHAIN_NAMES = (ab.CHAIN_REFERENCE, ab.CHAIN_CANDIDATE)


def relabel_chain(value):
    """Map a previous-experiment chain name onto this experiment's name.

    Exact match only, so a value that is already one of this experiment's names
    is returned untouched.
    """
    if isinstance(value, str):
        return CHAIN_LABEL_MAP.get(value, value)
    return value


def relabel_text(text):
    """Rewrite every FULL previous-experiment chain name inside free text.

    One left-to-right pass (`re.sub`), longest name first, so a replacement is
    never re-scanned and `date_balanced_reference` cannot be mangled into
    `overlap_harmonized_date_balanced_reference`.
    """
    import re

    if not isinstance(text, str):
        return text
    names = sorted(STALE_CHAIN_NAMES, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(name) for name in names))
    return pattern.sub(lambda m: CHAIN_LABEL_MAP[m.group(0)], text)


def render_pair_maps_for_product(reference_path, candidate_path, out_dir, *,
                                 product: str) -> list:
    """Reference/candidate panels under ONE shared stretch, THIS experiment's labels.

    Same behaviour as the shared helper -- one robust stretch across both
    panels, nearest-neighbour rendering, source rasters never modified -- but
    the staged panel names carry this experiment's chain names, so no output
    file is labelled with the previous experiment's chains.
    """
    import shutil as _shutil

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staging = out_dir / "_pair_inputs"
    staging.mkdir(parents=True, exist_ok=True)
    staged: "OrderedDict[str, Path]" = OrderedDict()
    for chain, source in ((CHAIN_REFERENCE, reference_path),
                          (CHAIN_CANDIDATE, candidate_path)):
        target = staging / f"{product}__{chain}.tif"
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(Path(source).resolve())
        except OSError:
            _shutil.copy2(str(source), str(target))
        staged[chain] = target

    written = audit.render_pair_maps(
        staged[CHAIN_REFERENCE], staged[CHAIN_CANDIDATE], out_dir, pair_name=product)
    _shutil.rmtree(staging, ignore_errors=True)
    return written


def stale_chain_labels_in(text) -> list:
    """Previous-experiment chain names still present in a rendered artefact."""
    haystack = text if isinstance(text, str) else json.dumps(text, default=str)
    return [name for name in STALE_CHAIN_NAMES if name in haystack]


def build_population_alignment(
    experiment_id: str,
    cohort: dict,
    reference_df,
    candidate_df,
) -> dict:
    report = ab.build_population_alignment(
        experiment_id,
        cohort,
        reference_df,
        candidate_df,
    )

    report["experiment"] = DIAGNOSTIC_NAMESPACE
    report["experiment_id"] = experiment_id
    report["report_schema_version"] = REPORT_SCHEMA_VERSION

    for item in report.get("row_exclusion_reasons") or []:
        old = item.get("chain")
        item["chain"] = _OLD_TO_NEW_CHAIN.get(old, old)

    return report

def build_population_alignment(
    experiment_id: str,
    cohort: dict,
    reference_df,
    candidate_df,
) -> dict:
    report = ab.build_population_alignment(
        experiment_id,
        cohort,
        reference_df,
        candidate_df,
    )

    report["experiment"] = DIAGNOSTIC_NAMESPACE
    report["experiment_id"] = experiment_id
    report["report_schema_version"] = REPORT_SCHEMA_VERSION

    for item in report.get("row_exclusion_reasons") or []:
        old = item.get("chain")
        item["chain"] = _OLD_TO_NEW_CHAIN.get(old, old)

    return report


# =============================================================================
# Required upstream statuses
# =============================================================================
#: Harmonization experiment (the source of the candidate).
REQUIRED_HARMONIZATION_FINAL_STATUS = "eligible_for_downstream_ab"

#: Previous compositing downstream A/B (the source of the reference chain).
REQUIRED_PREVIOUS_AB_FINAL_STATUS = "eligible_for_second_aoi_validation"
REQUIRED_PREVIOUS_AB_REFERENCE_REPRODUCTION = "pass"
REQUIRED_PREVIOUS_AB_BASELINE_INVARIANCE = "pass"
REQUIRED_PREVIOUS_AB_SHARED_MODIS = "pass"
REQUIRED_PREVIOUS_AB_POPULATION_ALIGNMENT = "ok"


# =============================================================================
# Predeclared final statuses (ORDERED -- see decide_final_status)
# =============================================================================
STATUS_INVALID_REFERENCE = "invalid_reference_reproduction"
STATUS_SUPPORT_INVARIANCE_FAILED = "support_invariance_failed"
STATUS_BASELINE_INVARIANCE_FAILED = "baseline_invariance_failed"
STATUS_POPULATION_REVIEW = "population_alignment_requires_review"
STATUS_SEAM_REDUCED_TRADEOFF = "seam_reduced_performance_tradeoff"
STATUS_ELIGIBLE_SECOND_AOI = "eligible_for_second_aoi_validation"
STATUS_INCONCLUSIVE = "downstream_effect_inconclusive"

FINAL_STATUSES = (
    STATUS_INVALID_REFERENCE,
    STATUS_SUPPORT_INVARIANCE_FAILED,
    STATUS_BASELINE_INVARIANCE_FAILED,
    STATUS_POPULATION_REVIEW,
    STATUS_SEAM_REDUCED_TRADEOFF,
    STATUS_ELIGIBLE_SECOND_AOI,
    STATUS_INCONCLUSIVE,
)

#: Claims this experiment may never make, in any field of any report.
FORBIDDEN_CONCLUSIONS = (
    "seam_fixed",
    "production_approved",
    "production_ready",
    "approved_for_production",
    "non_inferior",
    "non_inferiority",
    "transfer_improvement",
    "generalizes",
    "causal",
)

FINAL_STATUS_MEANINGS = OrderedDict((
    (STATUS_INVALID_REFERENCE,
     "The isolated reference chain did not reproduce the frozen date-balanced "
     "candidate chain of the previous downstream A/B. No scientific claim is "
     "made."),
    (STATUS_SUPPORT_INVARIANCE_FAILED,
     "The two chains did not share a bitwise-identical per-pixel unique-date "
     "support raster, so any downstream difference could be a change of pixel "
     "population rather than of date-offset harmonization. The experiment is "
     "void."),
    (STATUS_BASELINE_INVARIANCE_FAILED,
     "The baseline chain differed despite the intended current-LST-only "
     "intervention, or the two chains did not share identical MODIS inputs."),
    (STATUS_POPULATION_REVIEW,
     "Row-set or positive-cell differences are large enough to prevent a "
     "credible common-cohort comparison."),
    (STATUS_SEAM_REDUCED_TRADEOFF,
     "The seam evidence propagates, but within-region thermal performance is "
     "weakened: candidate thermal support is lost, or a paired ROC/PR interval "
     "lies wholly below zero, or the paired Brier interval lies wholly above "
     "zero."),
    (STATUS_ELIGIBLE_SECOND_AOI,
     "Eligible to repeat the SAME controlled A/B in bejis_2022, and nothing "
     "else. This is NOT production acceptance, NOT a non-inferiority proof, "
     "NOT evidence of transfer improvement or cross-region generalization, and "
     "NOT a causal claim."),
    (STATUS_INCONCLUSIVE,
     "The run is technically valid but satisfies none of the stronger "
     "categories."),
))


# =============================================================================
# Boundary evidence
# =============================================================================
#: The Step5 product whose seam reduction anchors the propagation question.
KEY_STEP5_SEAM_PRODUCT = ab.KEY_STEP5_SEAM_PRODUCT       # current_lst_celsius

#: The boundary that carries this experiment's predeclared seam evidence. The
#: harmonization intervention targets the CURRENT UNIQUE-ACQUISITION-DATE
#: support boundary, so that -- not the scene-count edge of the previous A/B --
#: is the key boundary here.
KEY_BOUNDARY_TYPE = "unique_date_count_edge"


# =============================================================================
# Namespace resolution and safety
# =============================================================================
def diagnostic_output_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """The ONE directory this experiment may write beneath."""
    return Path(base_dir) / "outputs" / "diagnostics" / DIAGNOSTIC_NAMESPACE / experiment_id


def harmonization_source_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen current-support harmonization root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "diagnostics" / HARMONIZATION_NAMESPACE / experiment_id


def previous_ab_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen compositing downstream A/B root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "diagnostics" / PREVIOUS_AB_NAMESPACE / experiment_id


def counterfactual_source_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen composite-counterfactual root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "diagnostics" / COUNTERFACTUAL_NAMESPACE / experiment_id


def canonical_experiment_root(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Frozen canonical experiment root (READ-ONLY)."""
    return Path(base_dir) / "outputs" / "experiments" / experiment_id


def previous_ab_reference_dir(experiment_id: str, stage: str,
                              base_dir: Path = PROJECT_ROOT) -> Path:
    """A stage directory of the PREVIOUS A/B candidate chain (READ-ONLY).

    This is the reproduction target. The canonical scene-weighted chain is
    deliberately NOT consulted anywhere in this experiment.
    """
    return previous_ab_root(experiment_id, base_dir) / PREVIOUS_AB_REFERENCE_SIDE / stage


def forbidden_write_roots(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> list[Path]:
    """Roots that must never be written, overwritten or deleted."""
    return [
        harmonization_source_root(experiment_id, base_dir),
        previous_ab_root(experiment_id, base_dir),
        counterfactual_source_root(experiment_id, base_dir),
        canonical_experiment_root(experiment_id, base_dir),
        Path(base_dir) / "data",
        Path(base_dir) / "config",
        Path(base_dir) / "core",
        Path(base_dir) / "src",
        Path(base_dir) / "scripts",
        Path(base_dir) / "outputs" / "step5",
        Path(base_dir) / "outputs" / "step5c",
        Path(base_dir) / "outputs" / "step3",
    ]


def assert_namespace_safe(paths, experiment_id: str,
                          base_dir: Path = PROJECT_ROOT) -> None:
    """Every supplied write path must resolve strictly under this root."""
    root = diagnostic_output_root(experiment_id, base_dir).resolve()
    forbidden = [p.resolve() for p in forbidden_write_roots(experiment_id, base_dir)]

    for raw in paths:
        candidate = Path(raw).resolve()
        for bad in forbidden:
            if candidate == bad or bad in candidate.parents:
                raise ab.NamespaceSafetyError(
                    f"refusing to write inside a frozen/read-only namespace: "
                    f"{candidate} (forbidden root: {bad})"
                )
        if candidate != root and root not in candidate.parents:
            raise ab.NamespaceSafetyError(
                f"refusing to write outside the dedicated harmonization "
                f"downstream-A/B root: {candidate} (allowed root: {root})"
            )


def clear_diagnostic_namespace(experiment_id: str,
                               base_dir: Path = PROJECT_ROOT) -> str | None:
    """`--force` deletion of ONLY the new harmonization downstream-A/B root."""
    root = diagnostic_output_root(experiment_id, base_dir)
    if not root.exists():
        return None
    resolved = root.resolve()
    assert_namespace_safe([resolved], experiment_id, base_dir)
    expected = diagnostic_output_root(experiment_id, base_dir).resolve()
    if resolved != expected:
        raise ab.NamespaceSafetyError(
            f"refusing to delete {resolved}: it is not the dedicated root {expected}"
        )
    if DIAGNOSTIC_NAMESPACE not in resolved.parts or experiment_id not in resolved.parts:
        raise ab.NamespaceSafetyError(
            f"refusing to delete a path that is not namespaced to this "
            f"experiment: {resolved}"
        )
    shutil.rmtree(resolved)
    return str(resolved)


def assert_supported_experiment(experiment_id: str) -> None:
    if experiment_id not in SUPPORTED_EXPERIMENT_IDS:
        raise HarmonizationDownstreamABError(
            f"unsupported --experiment {experiment_id!r}. This experiment supports "
            f"only {list(SUPPORTED_EXPERIMENT_IDS)}; another AOI needs its own "
            "frozen inputs and its own predeclaration."
        )


def assert_supported_candidate(candidate: str) -> None:
    if candidate not in SUPPORTED_CANDIDATES:
        raise HarmonizationDownstreamABError(
            f"unsupported --candidate {candidate!r}. This experiment supports only "
            f"{list(SUPPORTED_CANDIDATES)}."
        )


# =============================================================================
# Output layout
# =============================================================================
def plan_output_layout(experiment_id: str,
                       base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, Path]":
    """The full planned directory layout (informational; creates nothing)."""
    root = diagnostic_output_root(experiment_id, base_dir)
    layout: "OrderedDict[str, Path]" = OrderedDict()
    layout["root"] = root
    layout["config"] = root / "config"
    layout["inputs"] = root / "inputs"
    layout["inputs_shared"] = root / "inputs" / "shared"
    layout["inputs_shared_landsat_timeseries"] = (
        root / "inputs" / "shared" / "landsat_timeseries")
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


def plan_expected_files(experiment_id: str,
                        base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, Path]":
    """Every report/table artefact the live run is expected to produce."""
    root = diagnostic_output_root(experiment_id, base_dir)
    tables = root / "comparison" / "tables"
    return OrderedDict((
        ("harmonization_downstream_ab_summary.json",
         root / "harmonization_downstream_ab_summary.json"),
        ("harmonization_downstream_ab_summary.md",
         root / "harmonization_downstream_ab_summary.md"),
        ("harmonization_downstream_ab_manifest.json",
         root / "harmonization_downstream_ab_manifest.json"),
        ("input_provenance.json", root / "input_provenance.json"),
        ("current_support_invariance.json", root / "current_support_invariance.json"),
        ("reference_reproduction.json", root / "reference_reproduction.json"),
        ("population_alignment.json", root / "population_alignment.json"),
        ("fold_assignment.csv", root / "fold_assignment.csv"),
        ("oof_predictions.csv", root / "comparison" / "oof_predictions.csv"),
        ("raster_change_summary.csv", tables / "raster_change_summary.csv"),
        ("boundary_propagation.csv", tables / "boundary_propagation.csv"),
        ("step8_metrics.csv", tables / "step8_metrics.csv"),
        ("step8_paired_bootstrap.csv", tables / "step8_paired_bootstrap.csv"),
        ("step8_paired_bootstrap_replicates.csv",
         tables / "step8_paired_bootstrap_replicates.csv"),
    ))


# =============================================================================
# Upstream prerequisites
# =============================================================================
def _read_json(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def upstream_report_paths(experiment_id: str,
                          base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, Path]":
    """Frozen upstream reports whose prerequisites this experiment checks."""
    hz_root = harmonization_source_root(experiment_id, base_dir)
    ab_root = previous_ab_root(experiment_id, base_dir)
    return OrderedDict((
        ("harmonization_summary", hz_root / "harmonization_summary.json"),
        ("harmonization_manifest", hz_root / "harmonization_manifest.json"),
        ("harmonization_support_invariance", hz_root / "support_invariance.json"),
        ("harmonization_reference_reproduction", hz_root / "reference_reproduction.json"),
        ("previous_ab_summary", ab_root / "downstream_ab_summary.json"),
        ("previous_ab_manifest", ab_root / "downstream_ab_manifest.json"),
        ("previous_ab_reference_reproduction", ab_root / "reference_reproduction.json"),
        ("previous_ab_population_alignment", ab_root / "population_alignment.json"),
        ("previous_ab_input_provenance", ab_root / "input_provenance.json"),
    ))


def load_harmonization_state(experiment_id: str,
                             base_dir: Path = PROJECT_ROOT) -> dict:
    """The six required facts from the frozen harmonization summary."""
    summary = _read_json(
        upstream_report_paths(experiment_id, base_dir)["harmonization_summary"]) or {}
    reproduction = summary.get("frozen_reference_reproduction") or {}
    invariance = summary.get("support_invariance") or {}
    offsets = summary.get("fitted_date_offsets") or {}

    state = OrderedDict((
        ("final_status", summary.get("final_status")),
        ("final_status_required", REQUIRED_HARMONIZATION_FINAL_STATUS),
        ("frozen_reference_reproduction_passes", reproduction.get("passes")),
        ("support_invariance_passes", invariance.get("passes")),
        ("estimation_stable", offsets.get("estimation_stable")),
        ("production_approved", summary.get("production_approved")),
        ("changes_production_reducer", summary.get("changes_production_reducer")),
        ("seam_fixed", summary.get("seam_fixed")),
        ("max_abs_offset_celsius", offsets.get("max_abs_offset_celsius")),
        ("weighted_mean_offset_celsius", offsets.get("weighted_mean_offset_celsius")),
    ))
    state["failures"] = harmonization_prerequisite_failures(state)
    state["prerequisites_met"] = not state["failures"]
    return state


def harmonization_prerequisite_failures(state: dict) -> list[str]:
    """Every harmonization prerequisite, checked EXACTLY."""
    failures: list[str] = []
    if state.get("final_status") != REQUIRED_HARMONIZATION_FINAL_STATUS:
        failures.append(
            f"harmonization final_status={state.get('final_status')!r} "
            f"(required {REQUIRED_HARMONIZATION_FINAL_STATUS!r})")
    for key in ("frozen_reference_reproduction_passes", "support_invariance_passes",
                "estimation_stable"):
        if state.get(key) is not True:
            failures.append(f"harmonization {key}={state.get(key)!r} (required True)")
    for key in ("production_approved", "changes_production_reducer"):
        if state.get(key) is not False:
            failures.append(f"harmonization {key}={state.get(key)!r} (required False)")
    return failures


def load_previous_ab_state(experiment_id: str,
                           base_dir: Path = PROJECT_ROOT) -> dict:
    """The six required facts from the frozen compositing downstream A/B."""
    paths = upstream_report_paths(experiment_id, base_dir)
    summary = _read_json(paths["previous_ab_summary"]) or {}
    technical = summary.get("technical_validity") or {}
    alignment = _read_json(paths["previous_ab_population_alignment"]) or {}

    state = OrderedDict((
        ("final_status", summary.get("final_status")),
        ("final_status_required", REQUIRED_PREVIOUS_AB_FINAL_STATUS),
        ("reference_reproduction_status", technical.get("reference_reproduction_status")),
        ("baseline_invariance_status", technical.get("baseline_invariance_status")),
        ("shared_modis_invariance_status", technical.get("shared_modis_invariance_status")),
        ("population_alignment_status",
         technical.get("population_alignment_status") or alignment.get("status")),
        ("production_approved", summary.get("production_approved")),
        ("changes_production_reducer", summary.get("changes_production_reducer")),
        ("candidate_chain", summary.get("candidate_chain")),
        ("modis_compatibility_mode", technical.get("modis_compatibility_mode")),
    ))
    state["failures"] = previous_ab_prerequisite_failures(state)
    state["prerequisites_met"] = not state["failures"]
    return state


def previous_ab_prerequisite_failures(state: dict) -> list[str]:
    """Every previous-A/B prerequisite, checked EXACTLY."""
    failures: list[str] = []
    checks = (
        ("final_status", REQUIRED_PREVIOUS_AB_FINAL_STATUS),
        ("reference_reproduction_status", REQUIRED_PREVIOUS_AB_REFERENCE_REPRODUCTION),
        ("baseline_invariance_status", REQUIRED_PREVIOUS_AB_BASELINE_INVARIANCE),
        ("shared_modis_invariance_status", REQUIRED_PREVIOUS_AB_SHARED_MODIS),
        ("population_alignment_status", REQUIRED_PREVIOUS_AB_POPULATION_ALIGNMENT),
    )
    for key, required in checks:
        if state.get(key) != required:
            failures.append(
                f"previous A/B {key}={state.get(key)!r} (required {required!r})")
    if state.get("production_approved") is not False:
        failures.append(
            f"previous A/B production_approved={state.get('production_approved')!r} "
            "(required False)")
    if state.get("candidate_chain") != PREVIOUS_AB_REFERENCE_CHAIN:
        failures.append(
            f"previous A/B candidate_chain={state.get('candidate_chain')!r} "
            f"(required {PREVIOUS_AB_REFERENCE_CHAIN!r}); this experiment's "
            "reference must be that chain")
    return failures


def load_upstream_state(experiment_id: str, base_dir: Path = PROJECT_ROOT) -> dict:
    """Both prerequisite blocks, plus the combined verdict."""
    harmonization = load_harmonization_state(experiment_id, base_dir)
    previous = load_previous_ab_state(experiment_id, base_dir)
    paths = upstream_report_paths(experiment_id, base_dir)
    state = OrderedDict((
        ("harmonization", harmonization),
        ("previous_downstream_ab", previous),
        ("reports_present", OrderedDict(
            (key, Path(path).exists()) for key, path in paths.items())),
    ))
    state["failures"] = list(harmonization["failures"]) + list(previous["failures"])
    state["prerequisites_met"] = not state["failures"]
    return state


def validate_upstream_state(state: dict) -> None:
    if not state.get("prerequisites_met"):
        raise PrerequisiteError(
            "upstream prerequisites are not met; this experiment only follows a "
            "completed and valid harmonization run and a completed and valid "
            "compositing downstream A/B. Failures:\n  "
            + "\n  ".join(state.get("failures") or [])
        )


# =============================================================================
# Input contract
# =============================================================================
#: Band semantics of the composed two-band Step5 current-period input. Both
#: chains use this EXACT layout, so Step5 reads them through one code path.
CURRENT_BAND_1 = "current_lst_celsius"
CURRENT_BAND_2 = "unique_date_valid_count"

#: The frozen harmonization rasters that are the ONLY per-chain difference.
HARMONIZATION_CURRENT_SOURCES = OrderedDict((
    (CHAIN_REFERENCE, OrderedDict((
        ("lst", "reference_current_lst_celsius.tif"),
        ("count", "reference_unique_date_valid_count.tif"),
    ))),
    (CHAIN_CANDIDATE, OrderedDict((
        ("lst", "harmonized_current_lst_celsius.tif"),
        ("count", "harmonized_unique_date_valid_count.tif"),
    ))),
))


def harmonization_current_raster(experiment_id: str, chain: str, role: str,
                                 base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve one frozen harmonization current-period raster (READ-ONLY)."""
    if chain not in CHAINS:
        raise HarmonizationDownstreamABError(f"unknown chain: {chain!r}")
    if role not in ("lst", "count"):
        raise HarmonizationDownstreamABError(f"unknown role: {role!r}")
    return (harmonization_source_root(experiment_id, base_dir) / "rasters"
            / HARMONIZATION_CURRENT_SOURCES[chain][role])


def shared_baseline_source_dir(experiment_id: str,
                               base_dir: Path = PROJECT_ROOT) -> Path:
    """The date-balanced annual Landsat baselines both chains must share.

    They come from the PREVIOUS A/B candidate input bundle, so the baseline
    climatology entering this experiment is byte-identical to the one that
    produced the reference composite.
    """
    return (previous_ab_root(experiment_id, base_dir) / "inputs"
            / PREVIOUS_AB_REFERENCE_CHAIN / "landsat_timeseries")


def build_input_plan(ctx: dict, experiment_id: str,
                     base_dir: Path = PROJECT_ROOT) -> "OrderedDict[str, dict]":
    """Logical role -> {per-chain sources, materialized path, shared flag}.

    ONLY `current_lst` differs between chains. The annual Landsat baselines are
    a SHARED materialized copy here -- unlike the previous A/B, where they were
    the intervention -- because the harmonization candidate changed nothing
    outside the current period.
    """
    root = diagnostic_output_root(experiment_id, base_dir)
    canonical = canonical_experiment_root(experiment_id, base_dir)
    previous = previous_ab_root(experiment_id, base_dir)
    inputs = root / "inputs"
    shared = inputs / "shared"

    baseline_years = list(ctx["baseline_years"])
    current_days = ctx["current_period_days"]

    plan: "OrderedDict[str, dict]" = OrderedDict()

    # --- current-period Landsat LST: THE ONLY DIFFERENCE ---------------------
    current_name = f"landsat_current_period_{current_days}days.tif"
    previous_reference_current = (
        previous
        / "inputs"
        / PREVIOUS_AB_REFERENCE_CHAIN
        / "current_period"
        / current_name
    )
    plan["current_lst"] = {
        "role": "current_lst",
        "family": "landsat_lst",
        "shared": False,
        "differs_between_chains": True,
        "reference_source": previous_reference_current,
        "reference_count_source": previous_reference_current,
        "candidate_source": harmonization_current_raster(
            experiment_id, CHAIN_CANDIDATE, "lst", base_dir),
        "candidate_count_source": harmonization_current_raster(
            experiment_id, CHAIN_CANDIDATE, "count", base_dir),
        "materialized": OrderedDict((
            (CHAIN_REFERENCE, inputs / CHAIN_REFERENCE / "current_period" / current_name),
            (CHAIN_CANDIDATE, inputs / CHAIN_CANDIDATE / "current_period" / current_name),
        )),
        "materialization":
            "reference = verbatim copy of the previous A/B date-balanced "
            "two-band current-period input; candidate = two-band compose "
            f"(band1 {CURRENT_BAND_1}, band2 {CURRENT_BAND_2})",
    }

    # --- shared date-balanced annual Landsat baselines -----------------------
    baseline_dir = shared_baseline_source_dir(experiment_id, base_dir)
    baseline_sources = sorted(baseline_dir.glob("*.tif")) if baseline_dir.exists() else []
    for source in baseline_sources:
        plan[f"baseline_lst::{source.name}"] = {
            "role": f"baseline_lst::{source.name}",
            "family": "landsat_lst",
            "shared": True,
            "differs_between_chains": False,
            "reference_source": source,
            "candidate_source": source,
            "materialized": OrderedDict(
                (chain, shared / "landsat_timeseries" / source.name)
                for chain in CHAINS),
            "materialization": "verbatim_copy_shared_by_both_chains",
        }
    plan["_baseline_inventory"] = {
        "role": "_baseline_inventory",
        "family": "meta",
        "shared": True,
        "differs_between_chains": False,
        "reference_source": baseline_dir,
        "candidate_source": baseline_dir,
        "materialized": OrderedDict(
            (chain, shared / "landsat_timeseries") for chain in CHAINS),
        "materialization": "directory_marker",
        "baseline_years": baseline_years,
        "file_count": len(baseline_sources),
    }

    # --- every other input: ONE shared materialized copy ---------------------
    shared_specs = [
        ("ndvi_current", canonical / "data" / "ndvi_current_period" / "current_ndvi_median.tif",
         shared / "ndvi_current_period" / "current_ndvi_median.tif", "ndvi"),
        ("modis_lst_mean", canonical / "data" / "modis" / "modis_lst_mean_celsius.tif",
         shared / "modis" / "modis_lst_mean_celsius.tif", "modis"),
        ("modis_lst_std", canonical / "data" / "modis" / "modis_lst_std_celsius.tif",
         shared / "modis" / "modis_lst_std_celsius.tif", "modis"),
        ("dem_elevation", canonical / "data" / "dem" / "elevation.tif",
         shared / "dem" / "elevation.tif", "dem"),
        ("dem_slope", canonical / "data" / "dem" / "slope.tif",
         shared / "dem" / "slope.tif", "slope"),
        ("landcover_aligned",
         canonical / "gate_inputs" / "landcover_esa_worldcover_v200_aligned_to_reference.tif",
         shared / "gate_inputs" / "landcover_esa_worldcover_v200_aligned_to_reference.tif",
         "landcover"),
        ("mcd64a1_raw_burndate", canonical / "validation" / "labels" / "mcd64a1_raw.tif",
         shared / "labels" / "mcd64a1_raw.tif", "label"),
        ("mcd64a1_burned", canonical / "validation" / "labels" / "mcd64a1_burned.tif",
         shared / "labels" / "mcd64a1_burned.tif", "label"),
    ]
    ndvi_baseline_dir = canonical / "data" / "ndvi_timeseries"
    for year in baseline_years:
        matches = sorted(ndvi_baseline_dir.glob(f"ndvi_baseline_{year}-*.tif"))
        src = matches[0] if matches else ndvi_baseline_dir / f"ndvi_baseline_{year}.tif"
        shared_specs.append((
            f"ndvi_baseline_{year}", src,
            shared / "ndvi_timeseries" / src.name, "ndvi",
        ))

    for role, source, materialized, family in shared_specs:
        plan[role] = {
            "role": role,
            "family": family,
            "shared": True,
            "differs_between_chains": False,
            "reference_source": source,
            "candidate_source": source,
            "materialized": OrderedDict((chain, materialized) for chain in CHAINS),
            "materialization": "verbatim_copy_shared_by_both_chains",
        }

    plan["_previous_ab_reference_bundle"] = {
        "role": "_previous_ab_reference_bundle",
        "family": "meta",
        "shared": True,
        "differs_between_chains": False,
        "reference_source": previous / PREVIOUS_AB_REFERENCE_SIDE,
        "candidate_source": previous / PREVIOUS_AB_REFERENCE_SIDE,
        "materialized": OrderedDict((chain, None) for chain in CHAINS),
        "materialization": "read_only_reproduction_target_never_copied",
    }
    return plan


def differing_roles(plan: "OrderedDict[str, dict]") -> list[str]:
    return [role for role, entry in plan.items() if entry["differs_between_chains"]]


def only_current_lst_differs(plan: "OrderedDict[str, dict]") -> bool:
    """The whole experiment rests on this being true."""
    return differing_roles(plan) == ["current_lst"]


def missing_plan_sources(plan: "OrderedDict[str, dict]") -> list[str]:
    """Every source every chain needs, that does not exist."""
    missing: list[str] = []
    for entry in plan.values():
        if entry["family"] == "meta":
            continue
        for key in ("reference_source", "reference_count_source",
                    "candidate_source", "candidate_count_source"):
            source = entry.get(key)
            if source is None:
                continue
            if not Path(source).exists():
                missing.append(f"{entry['role']}:{key}={source}")
    return missing


def assert_required_frozen_inputs(plan: "OrderedDict[str, dict]",
                                  experiment_id: str) -> None:
    missing = missing_plan_sources(plan)
    if missing:
        raise PrerequisiteError(
            f"experiment {experiment_id!r} is missing required frozen inputs for "
            "the harmonization downstream A/B; this runner never falls back to an "
            "Earth Engine export. Missing:\n  " + "\n  ".join(missing)
        )
    inventory = plan.get("_baseline_inventory") or {}
    if not inventory.get("file_count"):
        raise PrerequisiteError(
            "the shared date-balanced annual Landsat baseline bundle is empty at "
            f"{inventory.get('reference_source')}. Both chains must share the "
            "frozen baselines from the previous A/B candidate bundle."
        )


# =============================================================================
# Current-support invariance gate
# =============================================================================
def check_current_support_invariance(reference_count: Path, candidate_count: Path,
                                     *, experiment_id: str) -> dict:
    """EXACT equality of the two unique-date valid-count rasters.

    Requires same grid, same valid mask, zero unequal pixels, zero changed valid
    pixels, a maximum difference of zero and a mask agreement of exactly 1.0.
    Nothing is tolerated: the candidate was produced by subtracting one scalar
    per acquisition date, which cannot move per-pixel date support. Any movement
    means the two chains no longer share a pixel population, and a downstream
    difference could then be a change of population rather than of harmonization.
    """
    import numpy as np
    import rasterio

    reference_count = Path(reference_count)
    candidate_count = Path(candidate_count)
    grid = assert_same_grid([reference_count, candidate_count])

    with rasterio.open(reference_count) as src:
        reference_band = 2 if src.count >= 2 else 1
        reference = src.read(reference_band, masked=True)

    with rasterio.open(candidate_count) as src:
        candidate_band = 1
        candidate = src.read(candidate_band, masked=True)

    reference_values = reference.filled(np.nan).astype("float64")
    candidate_values = candidate.filled(np.nan).astype("float64")
    reference_valid = np.isfinite(reference_values)
    candidate_valid = np.isfinite(candidate_values)

    total = int(reference_values.size)
    mask_equal = int((reference_valid == candidate_valid).sum())
    changed_valid = int((reference_valid != candidate_valid).sum())
    both = reference_valid & candidate_valid
    difference = np.abs(reference_values[both] - candidate_values[both]) if both.any() \
        else np.zeros(0, dtype="float64")
    unequal = int((difference != 0.0).sum()) + changed_valid
    max_difference = float(difference.max()) if difference.size else 0.0
    mask_agreement = (mask_equal / total) if total else None

    passes = bool(
        unequal == 0 and changed_valid == 0 and max_difference == 0.0
        and mask_agreement == 1.0
    )
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("gate", "current_unique_date_support_is_bitwise_invariant"),
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", CHAIN_CANDIDATE),
        ("reference_count_band", reference_band),
        ("candidate_count_band", candidate_band),
        ("reference_count_raster", str(reference_count)),
        ("candidate_count_raster", str(candidate_count)),
        ("grid", grid),
        ("total_pixels", total),
        ("reference_valid_pixels", int(reference_valid.sum())),
        ("candidate_valid_pixels", int(candidate_valid.sum())),
        ("unequal_pixels", unequal),
        ("changed_valid_pixels", changed_valid),
        ("max_difference", max_difference),
        ("mask_agreement", mask_agreement),
        ("required", OrderedDict((
            ("same_grid", True), ("same_valid_mask", True),
            ("unequal_pixels", 0), ("changed_valid_pixels", 0),
            ("max_difference", 0.0), ("mask_agreement", 1.0),
        ))),
        ("passes", passes),
        ("failure_status_if_not_invariant", STATUS_SUPPORT_INVARIANCE_FAILED),
        ("purpose",
         "Prevents a downstream difference that is really a change of pixel "
         "population. The intervention is one additive scalar per acquisition "
         "date, so per-pixel date support must be bitwise unchanged."),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def assert_current_support_invariance(report: dict) -> None:
    if not report.get("passes"):
        raise SupportInvarianceError(
            "the reference and candidate current-period unique-date support "
            "rasters are not identical: "
            f"unequal_pixels={report.get('unequal_pixels')}, "
            f"changed_valid_pixels={report.get('changed_valid_pixels')}, "
            f"max_difference={report.get('max_difference')}, "
            f"mask_agreement={report.get('mask_agreement')}. "
            f"The experiment stops at {STATUS_SUPPORT_INVARIANCE_FAILED}."
        )


# =============================================================================
# Materialization
# =============================================================================
def _copy_safe(source: Path, destination: Path) -> None:
    """Copy a frozen source into this namespace, atomically. Source is read-only."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.parent / f".{destination.name}.tmp"
    shutil.copy2(str(source), str(tmp))
    os.replace(str(tmp), str(destination))


def compose_current_period(lst_source: Path, count_source: Path,
                           destination: Path, *, chain: str) -> dict:
    """Build one chain's two-band current-period raster.

    Band 1 = current LST Celsius, band 2 = unique-date valid count, float32,
    nodata preserved from the LST source. Masked pixels are written as the
    declared nodata value and are NEVER zero-filled.
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
    if fill == 0.0:
        raise HarmonizationDownstreamABError(
            f"{lst_source} declares nodata=0.0; composing on a zero fill would "
            "make a masked pixel indistinguishable from a 0 C reading"
        )
    profile.update(count=2, dtype="float32", nodata=fill,
                   compress="lzw", BIGTIFF="IF_SAFER")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.parent / f".{destination.name}.tmp"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(lst.filled(fill).astype("float32"), 1)
        dst.write(cnt.filled(fill).astype("float32"), 2)
    os.replace(str(tmp), str(destination))

    return OrderedDict((
        ("chain", chain),
        ("band_1", CURRENT_BAND_1),
        ("band_2", CURRENT_BAND_2),
        ("dtype", "float32"),
        ("nodata", fill),
        ("zero_filled", False),
        ("lst_source", str(lst_source)),
        ("count_source", str(count_source)),
        ("valid_lst_pixels", int(np.sum(~np.ma.getmaskarray(lst)))),
        ("valid_count_pixels", int(np.sum(~np.ma.getmaskarray(cnt)))),
    ))


def assert_reference_candidate_grid_equality(
        plan: "OrderedDict[str, dict]") -> "OrderedDict[str, dict]":
    """Both chains' raw current-period LST rasters must share ONE exact grid."""
    entry = plan["current_lst"]
    paths = [Path(entry["reference_source"]), Path(entry["reference_count_source"]),
             Path(entry["candidate_source"]), Path(entry["candidate_count_source"])]
    signatures = OrderedDict((str(p), raster_signature(p)) for p in paths)
    first = next(iter(signatures.values()))
    passed = all(grids_equal(first, s) for s in signatures.values())
    return OrderedDict((
        ("gate", "raw_current_lst_grid_equality"),
        ("rasters", signatures),
        ("passed", bool(passed)),
        ("policy",
         "The reference and candidate current-period rasters, and both support "
         "rasters, must share one exact grid before anything is composed."),
    ))


def materialize_inputs(plan: "OrderedDict[str, dict]", experiment_id: str,
                       base_dir: Path = PROJECT_ROOT) -> dict:
    """Build both isolated input bundles; returns the provenance payload.

    Every destination is namespace-checked BEFORE it is written; frozen sources
    are opened read-only and are never modified.
    """
    destinations = []
    for entry in plan.values():
        for value in entry["materialized"].values():
            if value is not None:
                destinations.append(value)
    assert_namespace_safe(destinations, experiment_id, base_dir)

    grid_gate = assert_reference_candidate_grid_equality(plan)
    if not grid_gate["passed"]:
        raise PrerequisiteError(
            "the frozen harmonization current-period rasters do not share one "
            "exact grid; refusing to compose Step5 inputs.")

    compose_notes: "OrderedDict[str, dict]" = OrderedDict()
    for role, entry in plan.items():
        if entry["family"] == "meta":
            continue
        if entry["shared"]:
            destination = entry["materialized"][CHAIN_REFERENCE]
            _copy_safe(Path(entry["reference_source"]), destination)
            continue

        _copy_safe(
            Path(entry["reference_source"]),
            entry["materialized"][CHAIN_REFERENCE],
        )

        compose_notes[f"{role}::{CHAIN_REFERENCE}"] = OrderedDict((
            ("chain", CHAIN_REFERENCE),
            ("materialization", "verbatim_copy"),
            ("source", str(entry["reference_source"])),
            ("band_1", CURRENT_BAND_1),
            ("band_2", CURRENT_BAND_2),
        ))

        compose_notes[f"{role}::{CHAIN_CANDIDATE}"] = compose_current_period(
            Path(entry["candidate_source"]),
            Path(entry["candidate_count_source"]),
            entry["materialized"][CHAIN_CANDIDATE],
            chain=CHAIN_CANDIDATE,
        )

    return build_input_provenance(plan, experiment_id, grid_gate=grid_gate,
                                  compose_notes=compose_notes, base_dir=base_dir)


def _provenance_record(role: str, entry: dict, base_dir: Path) -> dict:
    record = OrderedDict((
        ("role", role),
        ("family", entry["family"]),
        ("shared_between_chains", bool(entry["shared"])),
        ("differs_between_chains", bool(entry["differs_between_chains"])),
        ("materialization", entry["materialization"]),
        ("sources", OrderedDict()),
        ("materialized", OrderedDict()),
    ))
    for key in ("reference_source", "reference_count_source",
                "candidate_source", "candidate_count_source"):
        source = entry.get(key)
        if source is None:
            continue
        path = Path(source)
        signed = sha256_and_size(path) if path.is_file() else {"sha256": None, "bytes": None}
        record["sources"][key] = OrderedDict((
            ("path", str(path)), ("exists", path.exists()),
            ("sha256", signed["sha256"]), ("bytes", signed["bytes"]),
        ))
    for chain, path in entry["materialized"].items():
        if path is None:
            continue
        path = Path(path)
        signed = sha256_and_size(path) if path.is_file() else {"sha256": None, "bytes": None}
        record["materialized"][chain] = OrderedDict((
            ("path", str(path)), ("exists", path.exists()),
            ("sha256", signed["sha256"]), ("bytes", signed["bytes"]),
        ))
    return record


def build_input_provenance(plan: "OrderedDict[str, dict]", experiment_id: str, *,
                           grid_gate: dict, compose_notes: dict,
                           base_dir: Path = PROJECT_ROOT) -> dict:
    """Assemble `input_provenance.json`, hashing every source and report."""
    records = [_provenance_record(role, entry, base_dir)
               for role, entry in plan.items()]
    upstream = OrderedDict()
    for key, path in upstream_report_paths(experiment_id, base_dir).items():
        path = Path(path)
        signed = sha256_and_size(path) if path.is_file() else {"sha256": None, "bytes": None}
        upstream[key] = OrderedDict((
            ("path", str(path)), ("exists", path.exists()),
            ("sha256", signed["sha256"]), ("bytes", signed["bytes"]),
        ))

    shared_roles = [r["role"] for r in records if r["shared_between_chains"]]
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", CHAIN_CANDIDATE),
        ("inputs", records),
        ("compose_notes", compose_notes),
        ("raw_current_lst_grid_equality_gate", grid_gate),
        ("roles_that_differ_between_chains", differing_roles(plan)),
        ("only_current_lst_differs", only_current_lst_differs(plan)),
        ("shared_role_count", len(shared_roles)),
        ("shared_baseline_source",
         str(shared_baseline_source_dir(experiment_id, base_dir))),
        ("baseline_shared_between_chains", True),
        ("baseline_recomputed", False),
        ("upstream_reports", upstream),
        ("upstream_state", load_upstream_state(experiment_id, base_dir)),
        ("earth_engine_used", False),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def _legacy_modis_provenance_view(provenance: dict) -> dict:
    """Adapt nested provenance to the flat schema expected by the old validator."""
    records: list[dict] = []

    for record in provenance.get("inputs") or []:
        role = record.get("role")
        if role not in ("modis_lst_mean", "modis_lst_std"):
            continue

        materialized = record.get("materialized") or {}
        source = (
            (record.get("sources") or {}).get("reference_source") or {}
        )

        for chain in CHAINS:
            item = materialized.get(chain) or {}

            records.append(OrderedDict((
                ("logical_role", role),
                ("family", "modis"),
                ("source_chain", chain),
                ("shared_between_chains", True),
                ("source_path", source.get("path")),
                ("materialized_path", item.get("path")),
                ("file_size_bytes", item.get("bytes")),
                ("sha256", item.get("sha256")),
            )))

    return {
        "experiment_id": provenance.get("experiment_id"),
        "created_at": provenance.get("created_at"),
        "inputs": records,
    }


def validate_legacy_modis_compatibility(
    experiment_id: str,
    provenance: dict,
    chain_contexts,
    base_dir: Path = PROJECT_ROOT,
    declaration: dict | None = None,
) -> dict:
    return ab.validate_legacy_modis_compatibility(
        experiment_id,
        _legacy_modis_provenance_view(provenance),
        chain_contexts,
        base_dir=base_dir,
        declaration=declaration,
    )


def check_shared_modis_invariance(
    provenance: dict,
    reference_ctx: dict,
    candidate_ctx: dict,
    attestation: dict,
    base_dir: Path = PROJECT_ROOT,
) -> dict:
    return ab.check_shared_modis_invariance(
        _legacy_modis_provenance_view(provenance),
        reference_ctx,
        candidate_ctx,
        attestation,
        base_dir=base_dir,
    )


def candidate_modifies_current_lst_only(provenance: dict) -> bool:
    return bool(provenance.get("only_current_lst_differs"))


def baselines_shared_between_chains(provenance: dict) -> bool:
    for record in provenance.get("inputs") or []:
        if record["family"] != "landsat_lst" or record["role"] == "current_lst":
            continue
        if not record["shared_between_chains"]:
            return False
        paths = {v["path"] for v in record["materialized"].values()}
        if len(paths) != 1:
            return False
    return True


# =============================================================================
# Chain contexts
# =============================================================================
def build_chain_context(experiment_id: str, chain: str,
                        base_dir: Path = PROJECT_ROOT) -> dict:
    """A production ExperimentContext re-rooted into this namespace.

    Both chains share `baseline_input_dir` (and every other input directory);
    they differ ONLY in `current_period_dir`. No window, date, baseline year,
    seed or threshold is altered.
    """
    from core.experiment_context import build_experiment_context

    if chain not in CHAINS:
        raise HarmonizationDownstreamABError(
            f"unknown chain: {chain!r}. Expected one of {CHAINS}.")

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
        # --- inputs: everything shared EXCEPT the current period -------------
        "data_root": chain_inputs,
        "baseline_input_dir": shared_inputs / "landsat_timeseries",
        "current_period_dir": chain_inputs / "current_period",
        "qa_dir": shared_inputs / "landsat_qa",
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
        # --- outputs ---------------------------------------------------------
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


CONTEXT_PATH_KEYS = ab.CONTEXT_PATH_KEYS


def assert_chain_context_namespaced(ctx: dict, experiment_id: str,
                                    base_dir: Path = PROJECT_ROOT) -> None:
    """No chain path may escape this root -- inputs included."""
    paths = [ctx[key] for key in CONTEXT_PATH_KEYS if ctx.get(key) is not None]
    assert_namespace_safe(paths, experiment_id, base_dir)


def contexts_share_all_inputs_except_current_period(reference_ctx: dict,
                                                    candidate_ctx: dict) -> dict:
    """Verify the contexts differ in `current_period_dir` and nothing else."""
    input_keys = ("baseline_input_dir", "qa_dir", "ndvi_baseline_dir",
                  "ndvi_current_dir", "modis_input_dir", "dem_input_dir",
                  "landcover_aligned_path", "gate_labels_dir")
    shared = OrderedDict(
        (key, str(reference_ctx.get(key)) == str(candidate_ctx.get(key)))
        for key in input_keys)
    return OrderedDict((
        ("shared_input_keys", shared),
        ("all_shared", all(shared.values())),
        ("current_period_dir_differs",
         str(reference_ctx.get("current_period_dir"))
         != str(candidate_ctx.get("current_period_dir"))),
        ("baseline_input_dir_identical", shared["baseline_input_dir"]),
    ))


# =============================================================================
# Reference reproduction: target is the PREVIOUS A/B CANDIDATE chain
# =============================================================================
#: Where each compared product lives inside the previous A/B candidate side.
PREVIOUS_AB_PRODUCT_STAGE = OrderedDict((
    ("current_lst_celsius", ("step5", "current_period_median_celsius.tif")),
    ("baseline_lst_mean_celsius", ("step5", "baseline_lst_mean_celsius.tif")),
    ("baseline_lst_std_celsius", ("step5", "baseline_lst_std_celsius.tif")),
    ("baseline_valid_count", ("step5", "baseline_valid_count.tif")),
    ("anomaly_zscore", ("step5", "anomaly_zscore.tif")),
    ("current_minus_baseline_celsius",
     (DERIVED_SUBDIR, "current_minus_baseline_celsius.tif")),
    ("current_tvdi", ("step5c", "current_tvdi.tif")),
    ("tvdi_difference", ("step5c", "tvdi_difference.tif")),
    ("downscaled_lst_celsius", ("step7d", "downscaled_lst_celsius.tif")),
    ("fused_lst_celsius", ("step7e", "fused_lst_celsius.tif")),
))


def previous_ab_product_path(experiment_id: str, product: str,
                             base_dir: Path = PROJECT_ROOT) -> Path | None:
    """The reproduction target for one product (READ-ONLY).

    Deliberately points at the previous A/B CANDIDATE side. The canonical
    scene-weighted chain is never consulted.
    """
    entry = PREVIOUS_AB_PRODUCT_STAGE.get(product)
    if entry is None:
        return None
    stage, filename = entry
    return previous_ab_reference_dir(experiment_id, stage, base_dir) / filename


def previous_ab_step8_dataset_path(experiment_id: str,
                                   base_dir: Path = PROJECT_ROOT) -> Path:
    return (previous_ab_reference_dir(experiment_id, "step8", base_dir)
            / "step8a" / "step8a_500m_modeling_dataset.parquet")


def build_reference_reproduction_report(experiment_id: str,
                                        raster_checks: "OrderedDict[str, dict]",
                                        step8_check: dict) -> dict:
    """Assemble `reference_reproduction.json` and its pass/fail verdict."""
    raster_passed = all(v.get("passed") for v in raster_checks.values())
    passed = bool(raster_passed and step8_check.get("passed"))
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("gate", "reference_chain_reproduces_previous_ab_candidate_chain"),
        ("reproduction_target", OrderedDict((
            ("namespace", PREVIOUS_AB_NAMESPACE),
            ("chain", PREVIOUS_AB_REFERENCE_CHAIN),
            ("side", PREVIOUS_AB_REFERENCE_SIDE),
            ("note",
             "The canonical scene-weighted chain is NOT the reproduction target "
             "for this experiment and is never compared against."),
        ))),
        ("comparison_policy",
         "exact grid equality; valid-mask agreement reported separately; values "
         "compared on the common valid mask with the EXISTING predeclared "
         "float32 tolerances and the existing semantic comparison helpers."),
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
# Boundary propagation summary (key boundary = unique_date_count_edge)
# =============================================================================
def summarize_boundary_propagation(verdicts: dict) -> dict:
    """Reuse the shared summariser, then re-key it to THIS experiment's boundary.

    The shared helper keys its verdict on the previous experiment's
    `scene_count_edge`. The harmonization intervention targets the current
    unique-acquisition-date support boundary, so the key fields are recomputed
    against `unique_date_count_edge` here. Every other field -- per-product
    verdicts, supported/increase lists, negative-control exclusion -- is the
    shared helper's output, unmodified.
    """
    summary = dict(ab.summarize_boundary_propagation(verdicts))
    key = verdicts.get(KEY_STEP5_SEAM_PRODUCT) or {}
    key_verdict = key.get(KEY_BOUNDARY_TYPE, {}) if isinstance(key, dict) else {}
    summary["key_step5_product"] = KEY_STEP5_SEAM_PRODUCT
    summary["key_boundary_type"] = KEY_BOUNDARY_TYPE
    summary["key_step5_seam_status"] = key_verdict.get("status")
    summary["key_step5_seam_reduction_supported"] = (
        key_verdict.get("status") == "supported_reduction")
    summary["key_boundary_rationale"] = (
        "The candidate changes only per-acquisition-date offsets, so the "
        "predeclared seam evidence is carried at the current unique-date "
        "support boundary, not at the scene-count boundary of the previous "
        "experiment.")
    return summary


def frozen_source_boundary_evidence(experiment_id: str,
                                    base_dir: Path = PROJECT_ROOT) -> dict:
    """The harmonization run's OWN boundary result, carried in as context.

    Kept strictly separate from the downstream propagation and Step8 evidence
    computed by THIS experiment: it is prior evidence about the Step5 seam, not
    a downstream result, and it may never be presented as one.
    """
    summary = _read_json(
        upstream_report_paths(experiment_id, base_dir)["harmonization_summary"]) or {}
    reductions = summary.get("support_boundary_reductions") or {}
    carried: "OrderedDict[str, dict]" = OrderedDict()
    for product, rows in reductions.items():
        if not isinstance(rows, dict):
            continue
        entry = rows.get("current_unique_date_count_change") or {}
        carried[product] = OrderedDict((
            ("boundary", "current_unique_date_count_change"),
            ("reference_excess_absolute_jump",
             entry.get("reference_excess_absolute_jump")),
            ("candidate_excess_absolute_jump",
             entry.get("candidate_excess_absolute_jump")),
            ("paired_reduction", entry.get("paired_reduction")),
            ("relative_paired_reduction", entry.get("relative_paired_reduction")),
            ("interval_low", entry.get("interval_low")),
            ("interval_high", entry.get("interval_high")),
            ("verdict", entry.get("verdict")),
        ))
    return OrderedDict((
        ("source_experiment", HARMONIZATION_NAMESPACE),
        ("source_final_status", summary.get("final_status")),
        ("evidence_kind", "frozen_step5_seam_evidence_from_the_source_experiment"),
        ("is_downstream_propagation_evidence", False),
        ("is_step8_evidence", False),
        ("separation_note",
         "These numbers were computed by the harmonization experiment on Step5 "
         "products. They are carried here as context only and are NEVER mixed "
         "with the downstream propagation or Step8 evidence computed by this "
         "experiment."),
        ("per_product", carried),
    ))


# =============================================================================
# Predeclared, ORDERED decision rule
# =============================================================================
DECISION_RULE_TEXT = (
    "Ordered gates. 1) invalid_reference_reproduction. 2) support_invariance_failed. "
    "3) baseline_invariance_failed (including the shared-MODIS technical gate). "
    "4) population_alignment_requires_review. 5) seam_reduced_performance_tradeoff "
    "when the key seam evidence is supported AND candidate thermal support is "
    "lost, or the paired ROC/PR interval is wholly below zero, or the paired "
    "Brier interval is wholly above zero. 6) eligible_for_second_aoi_validation "
    "when every predeclared eligibility condition holds. 7) otherwise "
    "downstream_effect_inconclusive. The strongest status means ONLY: repeat "
    "the same controlled A/B in bejis_2022."
)


def _status(status: str, reasons, evidence: dict, *,
            technical_failure: str | None = None,
            eligibility_checks: dict | None = None) -> dict:
    if status not in FINAL_STATUSES:
        raise HarmonizationDownstreamABError(f"undeclared final status: {status!r}")
    result = OrderedDict((
        ("final_status", status),
        ("meaning", FINAL_STATUS_MEANINGS[status]),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("decision_rule", DECISION_RULE_TEXT),
        ("reasons", list(reasons)),
        ("technical_failure", technical_failure),
        ("allowed_final_statuses", list(FINAL_STATUSES)),
        ("forbidden_conclusions", list(FORBIDDEN_CONCLUSIONS)),
        ("seam_fixed", False),
        ("production_approved", False),
        ("production_ready", False),
        ("claims_non_inferiority", False),
        ("claims_transfer_improvement", False),
        ("claims_cross_region_generalization", False),
        ("claims_causality", False),
    ))
    if eligibility_checks is not None:
        result["eligibility_checks"] = eligibility_checks
    return result


def decide_final_status(evidence: dict) -> dict:
    """Apply the predeclared, ordered decision rule."""
    # --- 1 ---
    if evidence.get("reference_reproduction_status") != "pass":
        return _status(STATUS_INVALID_REFERENCE, [
            "the isolated reference chain did not reproduce the frozen "
            f"{PREVIOUS_AB_REFERENCE_CHAIN} chain of the previous downstream A/B",
        ], evidence)

    # --- 2 ---
    if evidence.get("current_support_invariance_status") != "pass":
        return _status(STATUS_SUPPORT_INVARIANCE_FAILED, [
            "the reference and candidate current-period unique-date support "
            "rasters are not bitwise identical, so a downstream difference "
            "could be a change of pixel population rather than of date-offset "
            "harmonization",
        ], evidence)

    # --- 3 (including the shared-MODIS technical gate) ---
    modis_status = evidence.get("shared_modis_invariance_status")
    if modis_status is not None and modis_status != "pass":
        return _status(
            STATUS_BASELINE_INVARIANCE_FAILED,
            ["the two chains did not use identical MODIS inputs, identical "
             "aligned MODIS arrays and an identical MODIS compatibility mode: "
             f"{evidence.get('shared_modis_invariance_reasons')}"],
            evidence, technical_failure=TECHNICAL_FAILURE_SHARED_MODIS)
    attestation_status = evidence.get("modis_compatibility_attestation_status")
    if evidence.get("modis_compatibility_required") and attestation_status != "pass":
        return _status(
            STATUS_BASELINE_INVARIANCE_FAILED,
            ["the historical MODIS compatibility attestation did not pass "
             f"(status={attestation_status!r}); no scientific conclusion may be "
             "issued for this run"],
            evidence, technical_failure=TECHNICAL_FAILURE_SHARED_MODIS)
    if evidence.get("baseline_invariance_status") != "pass":
        return _status(STATUS_BASELINE_INVARIANCE_FAILED, [
            "the baseline chain differed despite the intended current-LST-only "
            "intervention",
        ], evidence)

    # --- 4 ---
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

    # --- 5 ---
    tradeoff_reasons: list[str] = []
    if roc.get("interval_wholly_below_zero"):
        tradeoff_reasons.append(
            "candidate-minus-reference ROC-AUC interval is wholly below zero")
    if pr.get("interval_wholly_below_zero"):
        tradeoff_reasons.append(
            "candidate-minus-reference PR-AUC interval is wholly below zero")
    if brier.get("interval_wholly_above_zero"):
        tradeoff_reasons.append(
            "candidate-minus-reference Brier interval is wholly above zero")
    if lost_support:
        tradeoff_reasons.append(
            "thermal-minus-baseline support present in the reference is lost in "
            "the candidate")
    if seam_supported and tradeoff_reasons:
        return _status(STATUS_SEAM_REDUCED_TRADEOFF, tradeoff_reasons, evidence)

    # --- 6 ---
    propagates = bool(evidence.get("downstream_supported_reduction_products"))
    no_contradiction = not bool(evidence.get("downstream_supported_increase_products"))
    eligible_checks = OrderedDict((
        ("reference_reproduction_passes", True),
        ("current_support_invariance_passes", True),
        ("baseline_invariance_passes", True),
        ("common_cohort_valid", True),
        ("key_seam_reduction_supported_at_unique_date_count_edge", seam_supported),
        ("propagates_to_at_least_one_downstream_thermal_product", propagates),
        ("no_contradictory_increase_across_key_product_chain", no_contradiction),
        ("candidate_thermal_roc_auc_interval_above_zero",
         bool(candidate_support.get("roc_auc_interval_above_zero"))),
        ("candidate_thermal_pr_auc_interval_above_zero",
         bool(candidate_support.get("pr_auc_interval_above_zero"))),
        ("paired_roc_auc_not_wholly_below_zero",
         not bool(roc.get("interval_wholly_below_zero"))),
        ("paired_pr_auc_not_wholly_below_zero",
         not bool(pr.get("interval_wholly_below_zero"))),
        ("paired_brier_not_wholly_above_zero",
         not bool(brier.get("interval_wholly_above_zero"))),
    ))
    if all(eligible_checks.values()):
        return _status(STATUS_ELIGIBLE_SECOND_AOI,
                       ["every predeclared eligibility condition is met"],
                       evidence, eligibility_checks=eligible_checks)

    # --- 7 ---
    failed = [name for name, ok in eligible_checks.items() if not ok]
    return _status(STATUS_INCONCLUSIVE, [
        "the run is valid but satisfies none of the stronger categories; unmet "
        f"eligibility conditions: {failed}",
    ], evidence, eligibility_checks=eligible_checks)


def next_decision_text(final_status: str) -> str:
    if final_status == STATUS_ELIGIBLE_SECOND_AOI:
        return (
            "Repeat the SAME controlled A/B in an independent AOI (bejis_2022). "
            "That is the only next step this status licenses: it is not "
            "production acceptance, not a non-inferiority proof, and not "
            "evidence of transfer improvement or cross-region generalization."
        )
    if final_status == STATUS_SEAM_REDUCED_TRADEOFF:
        return (
            "Do not carry the harmonized candidate further. Record the "
            "trade-off: the Step5 seam evidence propagates but within-region "
            "thermal performance is weakened."
        )
    if final_status == STATUS_SUPPORT_INVARIANCE_FAILED:
        return (
            "Repair the support invariance before any comparison is meaningful: "
            "the candidate must composite exactly the same acquisition dates at "
            "exactly the same pixels as the reference."
        )
    if final_status in (STATUS_INVALID_REFERENCE, STATUS_BASELINE_INVARIANCE_FAILED,
                        STATUS_POPULATION_REVIEW):
        return (
            "Repair the failing technical gate before any scientific claim is "
            "made. No result from this run may be interpreted."
        )
    return (
        "Treat the downstream effect as unresolved in this AOI. Do not escalate "
        "on inconclusive evidence."
    )


def required_limitations() -> list[str]:
    """The limitations every report MUST carry, verbatim."""
    return [
        "Manavgat only. One AOI; this generalises to nothing.",
        "The candidate differs from the reference ONLY in band 1 of the "
        "current-period Landsat LST raster; nothing about the baseline, NDVI, "
        "MODIS, DEM, land cover, labels, thresholds, features, model parameters "
        "or seeds differs.",
        "The intervention is a single ADDITIVE offset per acquisition date; no "
        "multiplicative, land-cover-dependent or spatially varying correction "
        "was estimated or applied.",
        "This is a within-region comparison on a common cohort with shared "
        "spatial folds. It says nothing about cross-region transfer.",
        "No non-inferiority claim is made or supported. An interval that "
        "includes zero is not evidence of equivalence.",
        "No causal claim is made: the comparison is observational and the "
        "chains differ by construction, not by randomisation.",
        "No production decision, production approval or production readiness is "
        "implied by any status this experiment can emit.",
        "The strongest reachable status licenses exactly one next action: "
        "repeat the same controlled A/B in bejis_2022.",
        "Interval language only. Nothing here is described as statistically "
        "significant.",
        "The frozen source-experiment boundary evidence carried into this "
        "report is Step5 evidence from the harmonization run; it is NOT "
        "downstream propagation evidence and NOT Step8 evidence.",
        "The historical MODIS compatibility attestation is reused unchanged and "
        "remains issued by, and scoped to, "
        f"{MODIS_ATTESTATION_ISSUER}.",
    ]


# =============================================================================
# Checkpointing (same stage set and validation policy as the previous A/B)
# =============================================================================
PLANNED_STAGES = ab.PLANNED_STAGES
MODIS_DEPENDENT_STAGES = ab.MODIS_DEPENDENT_STAGES
CHECKPOINT_FILENAME = "harmonization_downstream_ab_checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = "1.0-harmonization-downstream-ab"


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


def write_checkpoint_stage(root: Path, stage: str, outputs,
                           extra: dict | None = None,
                           attestation: dict | None = None) -> dict:
    """Atomically record a completed stage together with its output signatures."""
    if stage not in PLANNED_STAGES:
        raise HarmonizationDownstreamABError(f"unknown checkpoint stage: {stage!r}")
    root = Path(root)
    payload = read_checkpoint(root)
    payload.setdefault("experiment", DIAGNOSTIC_NAMESPACE)
    payload["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    payload.setdefault("stages", {})
    entry = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "outputs": [ab.file_reference(p) for p in outputs],
        **(extra or {}),
    }
    if attestation is not None:
        entry["modis_attestation_binding"] = ab.attestation_binding(attestation)
    payload["stages"][stage] = entry
    payload["last_stage"] = stage
    write_json_atomic(checkpoint_path(root), payload)
    return payload


def stage_is_reusable(root: Path, stage: str,
                      attestation: dict | None = None) -> bool:
    """Identical reuse policy to the previous A/B, on this experiment's file."""
    checkpoint = read_checkpoint(root)
    entry = (checkpoint.get("stages") or {}).get(stage)
    if not entry:
        return False
    if not ab.files_present_and_signed(entry.get("outputs") or []):
        return False
    if stage not in MODIS_DEPENDENT_STAGES:
        return True
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return False
    recorded = (entry.get("modis_attestation_binding") or {}).get("binding_sha256")
    expected = ab.attestation_binding(attestation or {})["binding_sha256"]
    return recorded == expected


# =============================================================================
# Configuration snapshot
# =============================================================================
def build_config_snapshot(experiment_id: str, candidate: str, ctx: dict) -> dict:
    """Everything predeclared, in one frozen record."""
    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("experiment_kind", "isolated_downstream_ab"),
        ("report_schema_version", REPORT_SCHEMA_VERSION),
        ("decision_rule_version", DECISION_RULE_VERSION),
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", candidate),
        ("question",
         "Does current-period date-offset harmonization propagate through "
         "Step5, Step5C, Step7 and Step8 without weakening within-region "
         "thermal performance?"),
        ("intervention", OrderedDict((
            ("changed_input", "current-period Landsat LST band 1 only"),
            ("band_1", CURRENT_BAND_1),
            ("band_2", CURRENT_BAND_2),
            ("band_2_must_be_bitwise_identical", True),
            ("shared_between_chains", [
                "date-balanced annual Landsat baselines", "NDVI", "MODIS",
                "DEM/slope", "land cover", "labels", "thresholds", "features",
                "model parameters", "seeds",
            ]),
            ("contexts_share_baseline_input_dir", True),
            ("contexts_differ_only_in_current_period_dir", True),
        ))),
        ("reproduction_target", OrderedDict((
            ("namespace", PREVIOUS_AB_NAMESPACE),
            ("chain", PREVIOUS_AB_REFERENCE_CHAIN),
            ("side", PREVIOUS_AB_REFERENCE_SIDE),
            ("canonical_scene_weighted_used", False),
        ))),
        ("prerequisites", OrderedDict((
            ("harmonization_final_status", REQUIRED_HARMONIZATION_FINAL_STATUS),
            ("harmonization_frozen_reference_reproduction_passes", True),
            ("harmonization_support_invariance_passes", True),
            ("harmonization_estimation_stable", True),
            ("harmonization_production_approved", False),
            ("harmonization_changes_production_reducer", False),
            ("previous_ab_final_status", REQUIRED_PREVIOUS_AB_FINAL_STATUS),
            ("previous_ab_reference_reproduction",
             REQUIRED_PREVIOUS_AB_REFERENCE_REPRODUCTION),
            ("previous_ab_baseline_invariance",
             REQUIRED_PREVIOUS_AB_BASELINE_INVARIANCE),
            ("previous_ab_shared_modis_invariance",
             REQUIRED_PREVIOUS_AB_SHARED_MODIS),
            ("previous_ab_population_alignment",
             REQUIRED_PREVIOUS_AB_POPULATION_ALIGNMENT),
            ("previous_ab_production_approved", False),
        ))),
        ("current_support_invariance_gate", OrderedDict((
            ("unequal_pixels", 0), ("changed_valid_pixels", 0),
            ("max_difference", 0.0), ("mask_agreement", 1.0),
            ("failure_status", STATUS_SUPPORT_INVARIANCE_FAILED),
        ))),
        ("reproduction_tolerances", dict(REPRODUCTION_TOLERANCES)),
        ("comparison", OrderedDict((
            ("primary_population", PRIMARY_POPULATION),
            ("key_step5_seam_product", KEY_STEP5_SEAM_PRODUCT),
            ("key_boundary_type", KEY_BOUNDARY_TYPE),
            ("boundary_types", list(BOUNDARY_TYPES)),
            ("boundary_propagation_products", list(BOUNDARY_PROPAGATION_PRODUCTS)),
            ("compared_raster_products", list(compared_raster_products())),
            ("paired_bootstrap_replicates", PAIRED_BOOTSTRAP_REPLICATES),
            ("bootstrap_unit", "spatial_block"),
            ("ci_lower_percentile", PAIRED_BOOTSTRAP_CI_LOWER),
            ("ci_upper_percentile", PAIRED_BOOTSTRAP_CI_UPPER),
            ("identical_block_draws_for_both_chains", True),
            ("row_split", "shared spatial folds; never a random row split"),
            ("direction",
             "positive ROC-AUC / PR-AUC difference is improvement; negative "
             "Brier difference is improvement"),
            ("interval_language",
             "interval includes zero / excludes zero -- never 'statistically "
             "significant'"),
        ))),
        ("modis", OrderedDict((
            ("machinery", "reused from src.landsat_composite_downstream_ab"),
            ("attestation_issuer", MODIS_ATTESTATION_ISSUER),
            ("issuer_unchanged", True),
            ("step7b_guard_default_changed", False),
            ("modis_values_modified", False),
            ("nodata_assigned", False),
            ("zeros_converted_to_nan", False),
            ("identical_inputs_required_for_both_chains", True),
        ))),
        ("allowed_final_statuses", list(FINAL_STATUSES)),
        ("forbidden_conclusions", list(FORBIDDEN_CONCLUSIONS)),
        ("decision_rule", DECISION_RULE_TEXT),
        ("planned_stages", list(PLANNED_STAGES)),
        ("baseline_years", list(ctx.get("baseline_years") or [])),
        ("current_period_days", ctx.get("current_period_days")),
        ("earth_engine_used", False),
        ("modifies_production_code", False),
        ("modifies_core_configuration", False),
        ("adds_new_model_or_feature_tuning", False),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


# =============================================================================
# Summary
# =============================================================================
def _limitations_for(modis_compatibility: dict) -> list[str]:
    """Required limitations, plus the legacy-MODIS ones when that path is used."""
    limitations = list(required_limitations())
    if (modis_compatibility or {}).get("mode") == LEGACY_MODIS_COMPATIBILITY_MODE:
        limitations.extend(legacy_modis_compatibility_limitations())
    return limitations


def build_summary(experiment_id: str, *, candidate: str, config: dict,
                  provenance: dict, support_invariance: dict, reproduction: dict,
                  alignment: dict, fold_manifest: dict, baseline_invariance: dict,
                  raster_change_rows: list, boundary_summary: dict,
                  boundary_result: dict, step8_metric_rows: list,
                  paired_rows: list, bootstrap: dict, decision: dict,
                  source_boundary_evidence: dict,
                  modis_compatibility: dict | None = None,
                  shared_modis_invariance: dict | None = None) -> dict:
    """Assemble `harmonization_downstream_ab_summary.json`."""
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
        ("seam_fixed", False),
        ("production_approved", False),
        ("production_ready", False),
        ("changes_production_reducer", False),
        ("claims_non_inferiority", False),
        ("claims_transfer_improvement", False),
        ("claims_cross_region_generalization", False),
        ("claims_causality", False),
        ("technical_failure", decision.get("technical_failure")),
        ("warnings", summary_warnings(modis_compatibility)),
        ("modis_compatibility", build_modis_compatibility_report(
            modis_compatibility, shared_modis_invariance)),
        ("modis_attestation_issuer", MODIS_ATTESTATION_ISSUER),
        ("decision", decision),
        ("configuration", config),
        ("technical_validity", OrderedDict((
            ("reference_reproduction_status", reproduction["status"]),
            ("reference_reproduction_target", PREVIOUS_AB_REFERENCE_CHAIN),
            ("current_support_invariance_status",
             "pass" if support_invariance.get("passes") else "fail"),
            ("current_support_unequal_pixels",
             support_invariance.get("unequal_pixels")),
            ("current_support_changed_valid_pixels",
             support_invariance.get("changed_valid_pixels")),
            ("current_support_mask_agreement",
             support_invariance.get("mask_agreement")),
            ("baseline_invariance_status", baseline_invariance["status"]),
            ("shared_modis_invariance_status", shared_modis_invariance.get("status")),
            ("shared_modis_technical_failure",
             shared_modis_invariance.get("technical_failure")),
            ("modis_compatibility_mode",
             modis_compatibility.get("mode", MODIS_STRICT_MODE)),
            ("modis_compatibility_attestation_status",
             modis_compatibility.get("status")),
            ("population_alignment_status", alignment["status"]),
            ("raw_current_lst_grid_equality_passed",
             provenance["raw_current_lst_grid_equality_gate"]["passed"]),
            ("only_current_lst_differs",
             candidate_modifies_current_lst_only(provenance)),
            ("baselines_shared_between_chains",
             baselines_shared_between_chains(provenance)),
            ("upstream_prerequisites_met",
             provenance["upstream_state"]["prerequisites_met"]),
            ("fold_assignment", fold_manifest),
            ("earth_engine_used", False),
        ))),
        ("raster_downstream_propagation", OrderedDict((
            ("raster_change_summary", raster_change_rows),
            ("boundary_propagation", boundary_summary),
            ("boundary_provenance_status", boundary_result.get("provenance_status")),
            ("export_tile_control", boundary_result.get("export_tile_control")),
            ("evidence_kind", "computed_by_this_experiment"),
        ))),
        ("frozen_source_boundary_evidence", source_boundary_evidence),
        ("within_region_model_impact", OrderedDict((
            ("primary_population", PRIMARY_POPULATION),
            ("cohort", "common_cohort"),
            ("per_chain_metrics", step8_metric_rows),
            ("evidence_kind", "computed_by_this_experiment"),
        ))),
        ("candidate_versus_reference_paired_comparison", OrderedDict((
            ("paired_rows", paired_rows),
            ("bootstrap_unit", bootstrap.get("bootstrap_unit")),
            ("n_blocks", bootstrap.get("n_blocks")),
            ("n_bootstrap_used", bootstrap.get("n_bootstrap_used")),
            ("seed", bootstrap.get("seed")),
            ("identical_block_draws_for_both_chains",
             bootstrap.get("identical_block_draws_for_both_chains")),
            ("interval_language", bootstrap.get("interval_language")),
            ("direction", bootstrap.get("direction")),
        ))),
        ("limitations", _limitations_for(modis_compatibility)),
        ("next_decision", next_decision_text(decision["final_status"])),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


def _scrub_declared_prohibitions(payload):
    """Drop keys that legitimately NAME a forbidden claim in order to deny it."""
    declared = {
        "forbidden_conclusions", "allowed_final_statuses", "seam_fixed",
        "production_approved", "production_ready", "changes_production_reducer",
        "claims_non_inferiority", "claims_transfer_improvement",
        "claims_cross_region_generalization", "claims_causality",
        "decision_rule", "limitations", "final_status_meaning", "meaning",
        "next_decision", "question",
    }
    if isinstance(payload, dict):
        return {k: _scrub_declared_prohibitions(v) for k, v in payload.items()
                if k not in declared}
    if isinstance(payload, list):
        return [_scrub_declared_prohibitions(v) for v in payload]
    return payload


def _iter_string_values(payload):
    """Yield string VALUES only; dictionary keys are metadata, not claims."""
    if isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_string_values(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_string_values(value)
    elif isinstance(payload, str):
        yield payload


def summary_forbids_banned_conclusions(payload) -> bool:
    """Reject forbidden conclusions in semantic string values, not key names."""
    scrubbed = _scrub_declared_prohibitions(payload)
    text = "\n".join(_iter_string_values(scrubbed)).lower()
    return not any(
        banned.lower() in text
        for banned in FORBIDDEN_CONCLUSIONS
    )


def report_generation_preserves_metrics(before: dict, after: dict) -> bool:
    return json.dumps(before, sort_keys=True, default=str) == json.dumps(
        after, sort_keys=True, default=str)


# =============================================================================
# Markdown
# =============================================================================
def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _bounds(row: dict, prefix: str) -> str:
    """`[low, high]` from the producer's `<prefix>_interval_low/high` fields."""
    low = row.get(f"{prefix}_interval_low")
    high = row.get(f"{prefix}_interval_high")
    if low is None or high is None:
        return "n/a"
    return f"[{low:.6g}, {high:.6g}]"


def _interval(row: dict) -> str:
    low, high = row.get("interval_low"), row.get("interval_high")
    if low is None or high is None:
        return "n/a"
    return f"[{low:.6g}, {high:.6g}]"


def render_summary_markdown(summary: dict) -> str:
    """Render the report from the summary payload ONLY (never recomputes)."""
    technical = summary["technical_validity"]
    decision = summary["decision"]
    lines: list[str] = []
    add = lines.append

    add(f"# Harmonization downstream A/B ({summary['experiment_id']})")
    add("")
    add(f"- reference chain: `{summary['reference_chain']}`")
    add(f"- candidate chain: `{summary['candidate_chain']}`")
    add(f"- final status: **`{summary['final_status']}`**")
    add(f"- {summary['final_status_meaning']}")
    add("")
    add("> Interval language only: an interval either includes or excludes "
        "zero. Nothing here is described as statistically significant, "
        "non-inferior, causal, or as evidence of transfer improvement or "
        "cross-region generalization.")
    add("")

    add("## 1. Technical validity")
    add("")
    for key in ("reference_reproduction_status",
                "current_support_invariance_status", "baseline_invariance_status",
                "shared_modis_invariance_status",
                "modis_compatibility_attestation_status",
                "population_alignment_status",
                "raw_current_lst_grid_equality_passed",
                "only_current_lst_differs", "baselines_shared_between_chains",
                "upstream_prerequisites_met", "earth_engine_used"):
        add(f"- `{key}`: {_fmt(technical.get(key))}")
    # Stated semantically, never as a chain identifier: the target is the
    # PREVIOUS A/B's candidate chain, so relabelling it into this experiment's
    # candidate name (`CHAIN_CANDIDATE`) would name the wrong chain. The JSON
    # keeps the identifier; only the prose is corrected here.
    add(f"- reference reproduction target: frozen date-balanced candidate "
        f"chain of the previous downstream A/B "
        f"(frozen source: the `{PREVIOUS_AB_NAMESPACE}` "
        f"`{PREVIOUS_AB_REFERENCE_SIDE}` side)")
    add(f"- MODIS attestation issuer: `{summary['modis_attestation_issuer']}` "
        "(deliberately unchanged)")
    add("")

    add("## 2. Current-support invariance")
    add("")
    add(f"- unequal pixels: {_fmt(technical.get('current_support_unequal_pixels'))} "
        "(required 0)")
    add(f"- changed valid pixels: "
        f"{_fmt(technical.get('current_support_changed_valid_pixels'))} (required 0)")
    add(f"- mask agreement: {_fmt(technical.get('current_support_mask_agreement'))} "
        "(required 1.0)")
    add("")

    add("## 3. Raster changes (candidate minus reference)")
    add("")
    rows = summary["raster_downstream_propagation"]["raster_change_summary"]
    if rows:
        # Field names are the PRODUCER's (`compare_raster_change`): mean,
        # max_abs_diff, changed_pixel_fraction. Never invented aliases.
        add("| product | mean | median | MAE | RMSE | p95 | max abs diff | "
            "changed fraction (thr) | mask agreement |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            add(f"| `{row.get('product')}` | {_fmt(row.get('mean'))} | "
                f"{_fmt(row.get('median'))} | {_fmt(row.get('mae'))} | "
                f"{_fmt(row.get('rmse'))} | {_fmt(row.get('p95'))} | "
                f"{_fmt(row.get('max_abs_diff'))} | "
                f"{_fmt(row.get('changed_pixel_fraction'))} "
                f"({_fmt(row.get('changed_pixel_threshold'))}) | "
                f"{_fmt(row.get('valid_mask_agreement'))} |")
    else:
        add("_no raster comparison was reached._")
    add("")

    add("## 4. Boundary propagation (computed by this experiment)")
    add("")
    boundary = summary["raster_downstream_propagation"]["boundary_propagation"]
    add(f"- key product: `{boundary.get('key_step5_product')}`")
    add(f"- key boundary: `{boundary.get('key_boundary_type')}`")
    add(f"- key seam status: `{boundary.get('key_step5_seam_status')}`")
    add(f"- downstream supported reductions: "
        f"{boundary.get('downstream_supported_reduction_products')}")
    add(f"- downstream supported increases: "
        f"{boundary.get('downstream_supported_increase_products')}")
    add(f"- _{boundary.get('key_boundary_rationale', '')}_")
    add("")

    add("## 5. Frozen source-experiment boundary evidence (context only)")
    add("")
    source = summary["frozen_source_boundary_evidence"]
    add(f"- source: `{source.get('source_experiment')}` "
        f"(status `{source.get('source_final_status')}`)")
    add(f"- _{source.get('separation_note', '')}_")
    add("")
    per_product = source.get("per_product") or {}
    if per_product:
        add("| product | reference excess | candidate excess | reduction | interval | verdict |")
        add("| --- | --- | --- | --- | --- | --- |")
        for product, row in per_product.items():
            add(f"| `{product}` | "
                f"{_fmt(row.get('reference_excess_absolute_jump'))} | "
                f"{_fmt(row.get('candidate_excess_absolute_jump'))} | "
                f"{_fmt(row.get('paired_reduction'))} | {_interval(row)} | "
                f"`{row.get('verdict')}` |")
        add("")

    add("## 6. Within-region model impact (Step8, common cohort)")
    add("")
    metrics = summary["within_region_model_impact"]["per_chain_metrics"]
    if metrics:
        add("Per-chain point metrics on the common cohort "
            f"(population `{summary['within_region_model_impact']['primary_population']}`):")
        add("")
        add("| chain | rows | positives | baseline ROC-AUC | thermal ROC-AUC | "
            "baseline PR-AUC | thermal PR-AUC | baseline Brier | thermal Brier |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in metrics:
            add(f"| `{relabel_chain(row.get('chain'))}` | {_fmt(row.get('n_rows'))} | "
                f"{_fmt(row.get('n_positives'))} | "
                f"{_fmt(row.get('baseline_roc_auc'))} | "
                f"{_fmt(row.get('thermal_roc_auc'))} | "
                f"{_fmt(row.get('baseline_pr_auc'))} | "
                f"{_fmt(row.get('thermal_pr_auc'))} | "
                f"{_fmt(row.get('baseline_brier'))} | "
                f"{_fmt(row.get('thermal_brier'))} |")
        add("")
        add("Thermal-minus-baseline support within each chain "
            "(95% spatial-block bootstrap interval):")
        add("")
        add("| chain | delta ROC-AUC | interval | delta PR-AUC | interval | "
            "delta Brier | interval |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for row in metrics:
            add(f"| `{relabel_chain(row.get('chain'))}` | "
                f"{_fmt(row.get('delta_roc_auc_thermal_minus_baseline'))} | "
                f"{_bounds(row, 'delta_roc_auc')} | "
                f"{_fmt(row.get('delta_pr_auc_thermal_minus_baseline'))} | "
                f"{_bounds(row, 'delta_pr_auc')} | "
                f"{_fmt(row.get('delta_brier_thermal_minus_baseline'))} | "
                f"{_bounds(row, 'delta_brier')} |")
    else:
        add("_no Step8 comparison was reached._")
    add("")

    add("## 7. Paired candidate-minus-reference comparison")
    add("")
    paired = summary["candidate_versus_reference_paired_comparison"]
    add(f"- bootstrap unit: `{paired.get('bootstrap_unit')}`; blocks: "
        f"{_fmt(paired.get('n_blocks'))}; replicates used: "
        f"{_fmt(paired.get('n_bootstrap_used'))}; seed: {_fmt(paired.get('seed'))}")
    add(f"- identical block draws for both chains: "
        f"{_fmt(paired.get('identical_block_draws_for_both_chains'))}")
    add(f"- direction: {paired.get('direction')}")
    add("")
    rows = paired.get("paired_rows") or []
    if rows:
        # `point_estimate` is the PRODUCER's field name for the
        # candidate-minus-reference difference.
        add("| metric | candidate - reference (point estimate) | bootstrap mean | "
            "95% interval | excludes zero | direction | point favours candidate |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            add(f"| `{row.get('metric')}` | {_fmt(row.get('point_estimate'))} | "
                f"{_fmt(row.get('bootstrap_mean'))} | {_interval(row)} | "
                f"{_fmt(row.get('interval_excludes_zero'))} | "
                f"{row.get('improvement_direction')} | "
                f"{_fmt(row.get('point_estimate_indicates_improvement'))} |")
    else:
        add("_no paired comparison was reached._")
    add("")

    add("## 8. Decision")
    add("")
    add(f"- **`{summary['final_status']}`**")
    for reason in decision.get("reasons") or []:
        add(f"- {reason}")
    add("")
    add(f"- seam_fixed: {_fmt(summary['seam_fixed'])}; production_approved: "
        f"{_fmt(summary['production_approved'])}; production_ready: "
        f"{_fmt(summary['production_ready'])}")
    add(f"- claims non-inferiority: {_fmt(summary['claims_non_inferiority'])}; "
        f"transfer improvement: {_fmt(summary['claims_transfer_improvement'])}; "
        f"cross-region generalization: "
        f"{_fmt(summary['claims_cross_region_generalization'])}; "
        f"causality: {_fmt(summary['claims_causality'])}")
    add("")
    add(f"_Decision rule ({summary['decision_rule_version']}): "
        f"{decision.get('decision_rule')}_")
    add("")

    add("## 9. Limitations")
    add("")
    for limitation in summary["limitations"]:
        add(f"- {limitation}")
    add("")

    add("## 10. Next decision")
    add("")
    add(summary["next_decision"])
    add("")
    # Final safety net: no rendered line may carry a previous-experiment chain
    # name. This relabels TEXT only; no metric is touched.
    return relabel_text("\n".join(lines))


# =============================================================================
# Manifest
# =============================================================================
MANIFEST_EXCLUDED_SUBTREES = ("inputs", "_analysis_tmp")

#: The manifest itself is never one of its own entries: its hash is written
#: after the entries are signed, so a self-entry could only ever record the
#: PREVIOUS manifest's digest.
MANIFEST_FILENAME = "harmonization_downstream_ab_manifest.json"


def manifest_candidate_files(root: Path) -> list[Path]:
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in MANIFEST_EXCLUDED_SUBTREES for part in relative.parts):
            continue
        if relative == Path(MANIFEST_FILENAME):
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
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", summary.get("candidate_chain")),
        ("final_status", summary["final_status"]),
        ("output_root", str(root)),
        ("file_count", len(entries)),
        ("files", entries),
        ("frozen_namespaces_written", 0),
        ("created_at", datetime.now(timezone.utc).isoformat()),
    ))


# =============================================================================
# Report-only regeneration (NO model, NO raster, NO Step5-Step8)
# =============================================================================
def relabel_map_outputs(root: Path, experiment_id: str) -> dict:
    """Rename map PNGs that still carry a previous-experiment chain name.

    Pure file rename inside this namespace: no raster is read, no figure is
    re-rendered and no pixel is recomputed. A destination that already exists is
    left alone and reported rather than overwritten.
    """
    root = Path(root)
    maps_dir = root / "comparison" / "maps"
    renamed: list[dict] = []
    skipped: list[dict] = []
    if not maps_dir.exists():
        return OrderedDict((("renamed", renamed), ("skipped", skipped),
                            ("maps_dir_present", False)))

    for path in sorted(maps_dir.rglob("*.png")):
        stale = stale_chain_labels_in(path.name)
        if not stale:
            continue
        new_name = relabel_text(path.name)
        destination = path.with_name(new_name)
        assert_namespace_safe([destination], experiment_id)
        if destination.exists():
            skipped.append({"path": str(path), "reason": "destination exists"})
            continue
        path.replace(destination)
        renamed.append({"from": str(path), "to": str(destination)})
    return OrderedDict((("renamed", renamed), ("skipped", skipped),
                        ("maps_dir_present", True)))


#: Scientific sections that a report-only regeneration must never alter.
SCIENTIFIC_SUMMARY_SECTIONS = (
    "final_status",
    "decision",
    "technical_validity",
    "raster_downstream_propagation",
    "frozen_source_boundary_evidence",
    "within_region_model_impact",
    "candidate_versus_reference_paired_comparison",
    "modis_compatibility",
)


def scientific_fingerprint(summary: dict) -> dict:
    """A stable hash of every scientific section, for before/after comparison."""
    import hashlib

    fingerprint: "OrderedDict[str, str]" = OrderedDict()
    for section in SCIENTIFIC_SUMMARY_SECTIONS:
        payload = json.dumps(summary.get(section), sort_keys=True, default=str)
        fingerprint[section] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return fingerprint


def rebuild_reports_from_summary(experiment_id: str,
                                 base_dir: Path = PROJECT_ROOT, *,
                                 relabel_maps: bool = True) -> dict:
    """Re-render the Markdown and manifest from the EXISTING summary JSON.

    Reads `harmonization_downstream_ab_summary.json`, re-renders the Markdown
    and rebuilds the manifest. It never trains a model, never reads or writes a
    raster, never runs Step5-Step8 and never touches Earth Engine. The summary
    JSON's scientific sections are fingerprinted before and after and must be
    byte-identical, so a report fix can never move a number.
    """
    root = diagnostic_output_root(experiment_id, base_dir)
    summary_path = root / "harmonization_downstream_ab_summary.json"
    markdown_path = root / "harmonization_downstream_ab_summary.md"
    manifest_path = root / "harmonization_downstream_ab_manifest.json"

    summary = _read_json(summary_path)
    if summary is None:
        raise PrerequisiteError(
            f"no completed summary to re-render at {summary_path}. The "
            "report-only path never runs the experiment; run the experiment "
            "first."
        )
    before = scientific_fingerprint(summary)
    final_status_before = summary.get("final_status")

    assert_namespace_safe([markdown_path, manifest_path], experiment_id)
    markdown = render_summary_markdown(summary)
    stale = stale_chain_labels_in(markdown)
    if stale:
        raise HarmonizationDownstreamABError(
            f"the re-rendered Markdown still carries previous-experiment chain "
            f"name(s) {stale}; refusing to write."
        )
    if not summary_forbids_banned_conclusions(summary):
        raise HarmonizationDownstreamABError(
            "the summary claims a forbidden conclusion; refusing to write.")

    tmp = markdown_path.parent / f".{markdown_path.name}.tmp"
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(markdown_path)

    map_relabel = (relabel_map_outputs(root, experiment_id) if relabel_maps
                   else OrderedDict((("renamed", []), ("skipped", []))))

    manifest = build_manifest(experiment_id, root, summary)
    write_json_atomic(manifest_path, manifest)

    after = scientific_fingerprint(_read_json(summary_path) or {})
    if after != before:
        raise HarmonizationDownstreamABError(
            "report-only regeneration altered a scientific section of the "
            f"summary JSON: {[k for k in before if before[k] != after.get(k)]}"
        )
    return OrderedDict((
        ("experiment_id", experiment_id),
        ("mode", "report_only"),
        ("final_status", final_status_before),
        ("final_status_unchanged",
         (_read_json(summary_path) or {}).get("final_status") == final_status_before),
        ("scientific_sections_unchanged", True),
        ("scientific_fingerprint", dict(after)),
        ("summary_json_rewritten", False),
        ("markdown_path", str(markdown_path)),
        ("manifest_path", str(manifest_path)),
        ("map_relabel", map_relabel),
        ("models_trained", 0),
        ("rasters_written", 0),
        ("pipeline_steps_run", 0),
        ("earth_engine_calls", 0),
    ))


# =============================================================================
# Dry-run plan (ZERO writes, ZERO Earth Engine)
# =============================================================================
def build_dry_run_plan(experiment_id: str, candidate: str,
                       base_dir: Path = PROJECT_ROOT) -> dict:
    """Everything the live run would do, without doing any of it."""
    assert_supported_experiment(experiment_id)
    assert_supported_candidate(candidate)

    from core.experiment_context import build_experiment_context

    ctx = build_experiment_context(experiment_id)
    plan = build_input_plan(ctx, experiment_id, base_dir)
    layout = plan_output_layout(experiment_id, base_dir)
    expected = plan_expected_files(experiment_id, base_dir)
    state = load_upstream_state(experiment_id, base_dir)

    try:
        modis = ab.build_dry_run_modis_compatibility(experiment_id, base_dir)
    except Exception as error:                              # noqa: BLE001
        modis = {"status": "unavailable", "reason": str(error)}

    resolved = OrderedDict()
    for role, entry in plan.items():
        if entry["family"] == "meta":
            continue
        resolved[role] = OrderedDict((
            ("shared", bool(entry["shared"])),
            ("differs_between_chains", bool(entry["differs_between_chains"])),
            ("reference_source", str(entry.get("reference_source"))),
            ("candidate_source", str(entry.get("candidate_source"))),
            ("reference_present", Path(entry["reference_source"]).exists()
             if entry.get("reference_source") else False),
            ("candidate_present", Path(entry["candidate_source"]).exists()
             if entry.get("candidate_source") else False),
        ))

    reference_ctx_preview = OrderedDict((
        ("baseline_input_dir", str(layout["inputs_shared"] / "landsat_timeseries")),
        ("current_period_dir", str(layout[f"inputs_{CHAIN_REFERENCE}"] / "current_period")),
    ))
    candidate_ctx_preview = OrderedDict((
        ("baseline_input_dir", str(layout["inputs_shared"] / "landsat_timeseries")),
        ("current_period_dir", str(layout[f"inputs_{CHAIN_CANDIDATE}"] / "current_period")),
    ))

    return OrderedDict((
        ("experiment", DIAGNOSTIC_NAMESPACE),
        ("experiment_id", experiment_id),
        ("reference_chain", CHAIN_REFERENCE),
        ("candidate_chain", candidate),
        ("output_root", str(layout["root"])),
        ("output_layout", OrderedDict((k, str(v)) for k, v in layout.items())),
        ("resolved_inputs", resolved),
        ("roles_that_differ_between_chains", differing_roles(plan)),
        ("only_current_lst_differs", only_current_lst_differs(plan)),
        ("shared_baseline_source",
         str(shared_baseline_source_dir(experiment_id, base_dir))),
        ("shared_baseline_file_count",
         (plan.get("_baseline_inventory") or {}).get("file_count")),
        ("missing_sources", missing_plan_sources(plan)),
        ("upstream_prerequisites", state),
        ("reproduction_target", OrderedDict((
            ("namespace", PREVIOUS_AB_NAMESPACE),
            ("chain", PREVIOUS_AB_REFERENCE_CHAIN),
            ("root", str(previous_ab_root(experiment_id, base_dir)
                         / PREVIOUS_AB_REFERENCE_SIDE)),
            ("canonical_scene_weighted_used", False),
        ))),
        ("chain_context_preview", OrderedDict((
            (CHAIN_REFERENCE, reference_ctx_preview),
            (CHAIN_CANDIDATE, candidate_ctx_preview),
        ))),
        ("current_support_invariance_gate", OrderedDict((
            ("reference_count_raster", str(harmonization_current_raster(
                experiment_id, CHAIN_REFERENCE, "count", base_dir))),
            ("candidate_count_raster", str(harmonization_current_raster(
                experiment_id, CHAIN_CANDIDATE, "count", base_dir))),
            ("required_unequal_pixels", 0),
            ("required_changed_valid_pixels", 0),
            ("required_max_difference", 0.0),
            ("required_mask_agreement", 1.0),
            ("failure_status", STATUS_SUPPORT_INVARIANCE_FAILED),
        ))),
        ("modis_compatibility", modis),
        ("modis_attestation_issuer", MODIS_ATTESTATION_ISSUER),
        ("configuration", build_config_snapshot(experiment_id, candidate, ctx)),
        ("planned_stages", list(PLANNED_STAGES)),
        ("expected_files", OrderedDict((k, str(v)) for k, v in expected.items())),
        ("allowed_final_statuses", list(FINAL_STATUSES)),
        ("forbidden_conclusions", list(FORBIDDEN_CONCLUSIONS)),
        ("decision_rule", DECISION_RULE_TEXT),
        ("limitations", required_limitations()),
        ("writes_performed", False),
        ("directories_created", 0),
        ("earth_engine_calls", 0),
        ("rasters_modified", 0),
        ("frozen_namespaces_touched", 0),
    ))
