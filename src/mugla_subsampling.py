"""Muğla sample-size subsampling sensitivity — `mugla_subsampling.v1`.

Implements the frozen design in `docs/mugla_subsampling_design/`. That design
is binding: this module makes no scientific decision of its own.

The question: when the Muğla modeling population (41,730 primary cells) is
reduced to exactly Manavgat's cell count (20,511), how do these three
quantities move relative to their full-Muğla references?

  A. within-Muğla 5-fold spatial OOF performance,
  B. Muğla -> Manavgat / Bejís raw transfer (Muğla as SOURCE),
  C. Manavgat / Bejís -> Muğla target evaluation (Muğla as TARGET).

DIAGNOSTIC CLASS: `population_size_matched_subsampling_sensitivity`.
This is a TOTAL-SAMPLE-SIZE sensitivity analysis and nothing else. Muğla's
prevalence is preserved (within integer rounding limits); Muğla's positive
count is NOT equalised to Manavgat's; no predictor or label structure is
altered. It is NOT a causal decomposition, NOT proof of a regional effect and
NOT an operational deployment claim.

Everything scientific is reused unchanged from the canonical Step8/Step9
contract: `build_pipeline`, `check_no_forbidden_features`,
`compute_binary_metrics`, `assign_large_blocks`, `population_subset`, the
feature constants and the seeds. No new model family, no hyperparameter
tuning, no threshold selection, no bootstrap, no p-value.

The 5-fold spatial fold mapping is NOT rebuilt: it is inherited, unchanged, per
`cell_id`, from the frozen full-Muğla 10-cell artifact, and every repeat shares
it.

Writes exclusively under
`outputs/diagnostics/mugla_subsampling/<analysis_id>/`.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import (
    STEP8B_MIN_POSITIVES_PER_POPULATION,
    STEP8B_N_SPLITS,
    STEP8B_RANDOM_SEED,
    STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
)
from core.paths import PROJECT_ROOT
from src.step8_large_block_robustness import NOMINAL_SCALES, assign_large_blocks
from src.step8b_train_baseline_vs_thermal_model import (
    BASELINE_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURE_COLUMNS,
    TARGET_COLUMN,
    THERMAL_FEATURES,
    THERMAL_MODEL_FEATURES,
    build_classifier,
    build_pipeline,
    check_no_forbidden_features,
    compute_binary_metrics,
)
from src.step9a_audit_cross_region_inputs import (
    PRIMARY_POPULATIONS,
    resolve_step8a_dataset_path,
    resolve_step8a_stats_path,
)
from src.step9b_run_cross_region_transfer import population_subset


class MuglaSubsamplingError(SystemExit):
    """Fail-fast error (same convention as every other step in this repo)."""


# =============================================================================
# Frozen contract -- docs/mugla_subsampling_design/SCIENTIFIC_CONTRACT.md
# =============================================================================
SCHEMA_VERSION = "mugla_subsampling.v1"
DIAGNOSTIC_NAMESPACE = "mugla_subsampling"
DIAGNOSTIC_CLASS = "population_size_matched_subsampling_sensitivity"

PRIMARY_EXPERIMENTS: tuple[str, ...] = ("manavgat_2021", "bejis_2022", "mugla_2021")
SUBSAMPLED_EXPERIMENT = "mugla_2021"
SIZE_REFERENCE_EXPERIMENT = "manavgat_2021"

EXCLUDED_EXPERIMENTS: dict[str, str] = {
    "evia_2021": "out_of_scope_for_this_frozen_analysis",
    "evia_2021_extended": (
        "high_prevalence_different_regime_sensitivity_control_not_part_of_this_"
        "size_matched_subsampling_contract"
    ),
    "kozan_2023": "legacy_shared_namespace_aoi_out_of_scope",
}
EXCLUDED_TOKENS: tuple[str, ...] = ("evia", "kozan")

POPULATION = PRIMARY_POPULATIONS[0]  # "burnable_tree_shrub_grass"
VALID_UNIVERSE = "valid_for_modeling == True"

ARM_ID = "size_matched_to_manavgat"
TARGET_SAMPLE_SIZE = 20511
N_REPEATS = 20

BLOCK_SIZE_CELLS = 10
BLOCK_NOMINAL_SCALE = NOMINAL_SCALES[BLOCK_SIZE_CELLS]  # "approximately_5_km"
BLOCK_COLUMN = "large_block_id"
CANONICAL_SMALL_BLOCK_SIZE_CELLS = STEP8B_SPATIAL_BLOCK_SIZE_CELLS  # 2, context only

FOLD_COUNT = STEP8B_N_SPLITS  # 5
FOLD_RANDOM_STATE = STEP8B_RANDOM_SEED  # 42
ESTIMATOR_SEED = STEP8B_RANDOM_SEED  # 42
MODEL_NAME = "random_forest"

MODEL_FAMILIES: tuple[str, ...] = ("baseline", "thermal")
FEATURE_LISTS: dict[str, list[str]] = {
    "baseline": list(BASELINE_FEATURES),
    "thermal": list(THERMAL_MODEL_FEATURES),
}

METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "brier_score")
PRIMARY_METRIC = "roc_auc"
LOWER_IS_BETTER: frozenset[str] = frozenset({"brier_score"})
ORIENTATION_HIGHER = "higher_is_better"
ORIENTATION_NEGATED = "lower_is_better_oriented_by_negation"

# Arms and their direction tokens. Direction tokens are NEVER sorted.
ARM_WITHIN = "within_mugla"
ARM_SOURCE = "mugla_as_source"
ARM_TARGET = "mugla_as_target"
ARMS: tuple[str, ...] = (ARM_WITHIN, ARM_SOURCE, ARM_TARGET)

WITHIN_DIRECTION = SUBSAMPLED_EXPERIMENT  # within-region: not a pair
SOURCE_PAIRS: tuple[tuple[str, str], ...] = (
    ("mugla_2021", "manavgat_2021"),
    ("mugla_2021", "bejis_2022"),
)
TARGET_PAIRS: tuple[tuple[str, str], ...] = (
    ("manavgat_2021", "mugla_2021"),
    ("bejis_2022", "mugla_2021"),
)

INTERVAL_PCT_LOW = 2.5
INTERVAL_PCT_HIGH = 97.5
PERCENTILE_METHOD = "linear"

POSITION_BELOW = "below_subsampling_interval"
POSITION_INSIDE = "inside_subsampling_interval"
POSITION_ABOVE = "above_subsampling_interval"
POSITION_TOKENS: tuple[str, ...] = (POSITION_BELOW, POSITION_INSIDE, POSITION_ABOVE)

SENTENCE_INSIDE = (
    "Observed performance is compatible with sample-size-matched Mugla subsets "
    "under this selection design."
)
SENTENCE_OUTSIDE = (
    "The full-population point estimate differs from the range observed across "
    "the deterministic size-matched subsets."
)

# Significance vocabulary is forbidden anywhere in the emitted outputs.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "confidence interval", "95% ci", "ci_2_5", "ci_97_5", "ci_lower", "ci_upper",
    "significant", "significance", "p-value", "p_value", "pvalue",
    "istatistiksel olarak anlaml", "anlaml",
    "sample size causes", "regional effect is proven", "difference is eliminated",
)

LIMITATIONS: tuple[str, ...] = (
    "The 20 repeats vary only in WHICH cells fill each stratum. The stratum "
    "allocation, the positive count, the per-fold row counts and the fold "
    "mapping are identical across repeats. The reported range therefore "
    "describes within-stratum selection variability alone; it is narrower than "
    "the variability of an unconstrained random subsample and much narrower "
    "than any sampling distribution of the estimator.",
    "The subsample retains every 10-cell block of the full population. Spatial "
    "extent is NOT reduced, only density within each block. This design "
    "isolates count from geographic coverage and cannot speak to the effect of "
    "a smaller AOI.",
    "Prevalence is preserved, not equalised. Mugla's positive count remains far "
    "above Manavgat's. Any residual difference between regions may still be a "
    "positive-count difference, which this analysis does not separate.",
    "The Mugla-as-target arm reuses frozen full-source models. It measures "
    "target cohort sensitivity under a fixed source and says nothing about "
    "source-side size effects.",
    "The final reading must be made from all three arms jointly: does "
    "within-region performance move; does transfer move when the Mugla source "
    "shrinks; is the metric ordering preserved across Mugla target subsets. No "
    "single arm supports a conclusion on its own.",
)

# --- Canonical frozen input digests ------------------------------------------
# Tests monkeypatch this mapping to redirect the hash gate onto synthetic
# fixtures. FROZEN_MUGLA_STEP8A_SHA256 below is the literal production digest
# and is deliberately NOT patched -- it is what decides whether the observed
# frame is the real one, and therefore whether the production inventory
# literals of docs/mugla_subsampling_design/SAMPLING_FEASIBILITY.md apply.
CANONICAL_STEP8A_SHA256: dict[str, str] = {
    "manavgat_2021": "054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439",
    "bejis_2022": "3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393",
    "mugla_2021": "c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e",
}
FROZEN_MUGLA_STEP8A_SHA256 = (
    "c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e"
)

# Read-only expectations that hold ONLY for the real frozen Muğla frame.
# Verified 2026-08-03; see docs/mugla_subsampling_design/SAMPLING_FEASIBILITY.md.
PRODUCTION_INVENTORY: dict[str, Any] = {
    "population_rows": 41730,
    "population_positives": 2911,
    "population_negatives": 38819,
    "target_sample_size": 20511,
    "n_blocks": 576,
    "n_strata": 636,
    "n_positive_strata": 70,
    "floor_total": 20211,
    "remainder_units": 300,
    "strata_above_cut": 295,
    "strata_tied_at_cut": 12,
    "tie_units_awarded": 5,
    "sampled_positives": 1438,
    "sampled_negatives": 19073,
    "fold_rows": (4111, 4096, 4107, 4096, 4101),
    "fold_positives": (293, 280, 295, 281, 289),
}
PRODUCTION_TARGET_POPULATIONS: dict[str, int] = {
    "manavgat_2021": 20511,
    "bejis_2022": 15190,
}

# --- Frozen reference artifacts ----------------------------------------------
WITHIN_REFERENCE_RELATIVE = Path(
    "mugla_2021/robustness/step8_big_blocks/block_10_cells"
)
WITHIN_REFERENCE_METRICS_NAME = "step8b_metrics.json"
WITHIN_REFERENCE_OOF_NAME = "oof_predictions.parquet"
WITHIN_REFERENCE_FOLDS_NAME = "fold_assignments.parquet"
WITHIN_REFERENCE_MANIFEST_NAME = "block_manifest.json"
FOLD_ARTIFACT_SCHEMA = "step8.big_block_robustness.v2"

TRANSFER_METRICS_NAME = "cross_region_transfer_metrics.json"
TRANSFER_PREDICTIONS_NAME = "cross_region_transfer_predictions.parquet"

# --- Stages -------------------------------------------------------------------
STAGES: tuple[str, ...] = ("plan", "fit", "summarize")
STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "plan": (),
    "fit": ("plan",),
    "summarize": ("plan", "fit"),
}

OOF_PREDICTIONS_DIRNAME = "oof_predictions.parquet"
QUARANTINE_DIRNAME = "_quarantine"

STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "plan": (
        "config.json",
        "input_hashes.json",
        "sampling_inventory.csv",
        "stratum_allocation.csv",
        "selected_cells.parquet",
        "fold_mapping.parquet",
        "reference_metrics.csv",
    ),
    "fit": (
        OOF_PREDICTIONS_DIRNAME,
        "repeat_metrics.csv",
    ),
    "summarize": (
        "subsampling_summary.csv",
        "summary.json",
        "report.md",
        "manifest.json",
    ),
}


# =============================================================================
# Contract helpers
# =============================================================================
def validate_stage_range(from_stage: str, to_stage: str) -> list[str]:
    if from_stage not in STAGES:
        raise MuglaSubsamplingError(
            f"Unknown from_stage {from_stage!r}; stage order is {list(STAGES)}."
        )
    if to_stage not in STAGES:
        raise MuglaSubsamplingError(
            f"Unknown to_stage {to_stage!r}; stage order is {list(STAGES)}."
        )
    start, end = STAGES.index(from_stage), STAGES.index(to_stage)
    if start > end:
        raise MuglaSubsamplingError(
            f"from_stage {from_stage!r} comes after to_stage {to_stage!r}; "
            f"stage order is {list(STAGES)}."
        )
    return list(STAGES[start:end + 1])


def resolve_experiments(experiments: Optional[Sequence[str]] = None) -> list[str]:
    """The three primary AOIs, with the Evia/Kozan exclusion enforced."""
    resolved = list(experiments) if experiments else list(PRIMARY_EXPERIMENTS)
    if len(set(resolved)) != len(resolved):
        raise MuglaSubsamplingError(
            f"Duplicate experiment id in {resolved}; each AOI may appear once."
        )
    if len(resolved) != 3:
        raise MuglaSubsamplingError(
            "This frozen analysis requires exactly three experiments "
            f"(two full cohorts plus the subsampled one); got {resolved}."
        )
    for experiment in resolved:
        assert_not_excluded(experiment)
    return resolved


def assert_not_excluded(experiment_id: str) -> None:
    lowered = str(experiment_id).lower()
    if experiment_id in EXCLUDED_EXPERIMENTS:
        raise MuglaSubsamplingError(
            f"{experiment_id!r} is excluded from this frozen analysis: "
            f"{EXCLUDED_EXPERIMENTS[experiment_id]}. Including it requires a new "
            "preregistration and a new analysis_id."
        )
    for token in EXCLUDED_TOKENS:
        if token in lowered:
            raise MuglaSubsamplingError(
                f"{experiment_id!r} matches the excluded token {token!r}. This "
                f"analysis is frozen to {list(PRIMARY_EXPERIMENTS)}."
            )


def direction_token(source_id: str, target_id: str) -> str:
    if source_id == target_id:
        raise MuglaSubsamplingError(f"Self-pair is forbidden: {source_id} -> {target_id}.")
    return f"{source_id}_to_{target_id}"


def source_directions() -> list[str]:
    return [direction_token(s, t) for s, t in SOURCE_PAIRS]


def target_directions() -> list[str]:
    return [direction_token(s, t) for s, t in TARGET_PAIRS]


def all_direction_rows() -> list[tuple[str, str]]:
    """(arm, direction) for every reported evaluation, in a frozen order."""
    rows: list[tuple[str, str]] = [(ARM_WITHIN, WITHIN_DIRECTION)]
    rows += [(ARM_SOURCE, d) for d in source_directions()]
    rows += [(ARM_TARGET, d) for d in target_directions()]
    return rows


def metric_orientation(metric: str) -> str:
    return ORIENTATION_NEGATED if metric in LOWER_IS_BETTER else ORIENTATION_HIGHER


def natural_delta(subsample_value: Optional[float],
                  full_reference_value: Optional[float]) -> Optional[float]:
    """Always `subsample - full`, on the metric's natural scale."""
    if subsample_value is None or full_reference_value is None:
        return None
    return float(subsample_value) - float(full_reference_value)


def oriented_delta(metric: str, subsample_value: Optional[float],
                   full_reference_value: Optional[float]) -> Optional[float]:
    """Positive ALWAYS means the subsample result is better.

    ROC-AUC / PR-AUC: subsample - full.  Brier: full - subsample.
    """
    if subsample_value is None or full_reference_value is None:
        return None
    if metric in LOWER_IS_BETTER:
        return float(full_reference_value) - float(subsample_value)
    return float(subsample_value) - float(full_reference_value)


def oriented_value(metric: str, value: Optional[float]) -> Optional[float]:
    """The metric on a higher-is-better scale, for position comparison only."""
    if value is None:
        return None
    return -float(value) if metric in LOWER_IS_BETTER else float(value)


def expected_unique_fit_count(n_repeats: Optional[int] = None) -> dict[str, int]:
    repeats = int(N_REPEATS if n_repeats is None else n_repeats)
    families = len(MODEL_FAMILIES)
    within = repeats * families * FOLD_COUNT
    source = repeats * families
    return {
        "within_fits": within,
        "source_fits": source,
        "target_fits": 0,
        "unique_fits": within + source,
        "reuse_events": source,  # each source fit serves both targets
    }


def planned_output_layout() -> dict[str, str]:
    """Every file this analysis may write. Used by --dry-run."""
    return {
        "config.json": "Frozen scientific configuration; the object hashed into analysis_id.",
        "input_hashes.json": "Canonical Step8A digests plus every frozen reference artifact digest.",
        "sampling_inventory.csv": "Per-experiment population accounting and the subsample summary.",
        "stratum_allocation.csv": "Per-stratum Hamilton allocation; repeat-invariant.",
        "selected_cells.parquet": "The frozen per-repeat cell selection, written before any fit.",
        "fold_mapping.parquet": "The inherited full-Mugla block->fold mapping.",
        "reference_metrics.csv": "Full-population reference values with independent recomputation.",
        f"{OOF_PREDICTIONS_DIRNAME}/part-<arm>.parquet":
            "Predictions, one logical dataset partitioned by arm.",
        "repeat_metrics.csv": "Per-repeat metric values with natural and oriented deltas.",
        "subsampling_summary.csv": "The primary result: median / interval / min / max per direction x family x metric.",
        "summary.json": "Headline table, accounting, limitations.",
        "report.md": "Human-readable report.",
        "stages/plan.json": "Stage marker with per-file hashes.",
        "stages/fit.json": "Stage marker with per-file hashes.",
        "stages/summarize.json": "Stage marker with per-file hashes.",
        "manifest.json": "Every produced file with size and sha256; the citable record.",
    }


# =============================================================================
# Serialisation + atomic writes
# =============================================================================
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_document(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                      default=_json_default) + "\n"


def canonical_json(payload: Any) -> str:
    """Stable serialisation used for the analysis identity."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_json_default)


def compute_analysis_id(scientific_config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(scientific_config).encode("utf-8")).hexdigest()


def _csv_cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _csv_document(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _csv_cell(row.get(column)) for column in columns})
    return buffer.getvalue()


def assert_inside_namespace(path: Path, root: Path) -> None:
    resolved, root_resolved = Path(path).resolve(), Path(root).resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise MuglaSubsamplingError(
            f"Refusing to write outside the analysis namespace: {resolved} is not "
            f"under {root_resolved}."
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_namespaced(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    assert_inside_namespace(path, root)
    _atomic_write_text(path, text)
    return path


def _write_namespaced_parquet(root: Path, relative: str, frame: pd.DataFrame) -> Path:
    path = root / relative
    assert_inside_namespace(path, root)
    _atomic_write_parquet(path, frame)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash of a file, or of a directory's contents (the OOF dataset)."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(sha256_file(child).encode("utf-8"))
        return digest.hexdigest()
    raise MuglaSubsamplingError(f"Cannot hash a path that does not exist: {path}.")


def _git_commit() -> Optional[str]:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None
    return None


def _package_versions() -> dict[str, str]:
    import sklearn
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


# =============================================================================
# Paths
# =============================================================================
def outputs_root(output_root: Optional[Path] = None) -> Path:
    return Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"


def diagnostics_root(output_root: Optional[Path] = None) -> Path:
    return outputs_root(output_root) / "diagnostics" / DIAGNOSTIC_NAMESPACE


def analysis_root(analysis_id: str, output_root: Optional[Path] = None) -> Path:
    return diagnostics_root(output_root) / analysis_id


def stage_marker_path(analysis_id: str, stage: str,
                      output_root: Optional[Path] = None) -> Path:
    return analysis_root(analysis_id, output_root) / "stages" / f"{stage}.json"


def canonical_step8a_path(experiment_id: str,
                          experiments_root: Optional[Path] = None) -> Path:
    if experiments_root is None:
        return resolve_step8a_dataset_path(experiment_id)
    return (Path(experiments_root) / experiment_id / "step8a"
            / "step8a_500m_modeling_dataset.parquet")


def canonical_step8a_manifest_path(experiment_id: str,
                                   experiments_root: Optional[Path] = None) -> Path:
    if experiments_root is None:
        return resolve_step8a_stats_path(experiment_id)
    return Path(experiments_root) / experiment_id / "step8a" / "step8a_dataset_stats.json"


def within_reference_dir(experiments_root: Optional[Path] = None) -> Path:
    """The frozen full-Muğla 10-cell condition directory.

    Resolved from the canonical Step8A location of the subsampled experiment,
    so an injected `experiments_root` redirects it too.
    """
    step8a = canonical_step8a_path(SUBSAMPLED_EXPERIMENT, experiments_root)
    experiment_dir = step8a.parent.parent
    return experiment_dir / "robustness" / "step8_big_blocks" / "block_10_cells"


def transfer_reference_dir(source_id: str, target_id: str,
                           output_root: Optional[Path] = None) -> Path:
    """Direction S->T is ALWAYS read from outputs/cross_region/{S}__{T}/step9b/.

    Both pair directories of a bidirectional Step9B run contain both
    directions, and they are byte-different (separate runs, different row
    order) while being metric-identical. This resolution rule removes the
    ambiguity; the chosen file's digest is frozen into input_hashes.json.
    """
    return outputs_root(output_root) / "cross_region" / f"{source_id}__{target_id}" / "step9b"


# =============================================================================
# Inputs and the hash gate
# =============================================================================
def build_frozen_input_inventory(
    experiment_ids: Sequence[str], experiments_root: Optional[Path] = None,
) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for experiment_id in experiment_ids:
        dataset_path = canonical_step8a_path(experiment_id, experiments_root)
        if not dataset_path.is_file():
            raise MuglaSubsamplingError(
                f"Canonical Step8A dataset missing for {experiment_id!r}: {dataset_path}."
            )
        manifest_path = canonical_step8a_manifest_path(experiment_id, experiments_root)
        expected = CANONICAL_STEP8A_SHA256.get(experiment_id)
        observed = sha256_file(dataset_path)
        inventory[experiment_id] = {
            "path": str(dataset_path),
            "sha256": observed,
            "expected_sha256": expected,
            "match": (expected is not None) and (observed == expected),
            "expectation_registered": expected is not None,
            "step8a_manifest_path": str(manifest_path),
            "step8a_manifest_sha256": (
                sha256_file(manifest_path) if manifest_path.is_file() else None
            ),
        }
    return inventory


def assert_canonical_step8a_hashes(inventory: dict[str, Any], *, strict: bool = True
                                   ) -> dict[str, Any]:
    """Hard hash gate. Runs before any computation, in every stage."""
    mismatches = [
        experiment_id for experiment_id, entry in inventory.items()
        if entry.get("expectation_registered") and not entry.get("match")
    ]
    unregistered = [
        experiment_id for experiment_id, entry in inventory.items()
        if not entry.get("expectation_registered")
    ]
    if strict and mismatches:
        detail = "; ".join(
            f"{experiment_id}: expected {inventory[experiment_id]['expected_sha256']}, "
            f"observed {inventory[experiment_id]['sha256']}"
            for experiment_id in mismatches
        )
        raise MuglaSubsamplingError(
            "Canonical Step8A hash mismatch -- the frozen inputs of this analysis "
            f"have changed. {detail}. Refusing to run."
        )
    if strict and unregistered:
        raise MuglaSubsamplingError(
            f"No registered canonical hash for {unregistered}. This frozen analysis "
            f"only accepts {sorted(CANONICAL_STEP8A_SHA256)}."
        )
    return {"mismatches": mismatches, "unregistered": unregistered,
            "all_match": not mismatches and not unregistered}


def is_production_mugla_frame(inventory: dict[str, Any]) -> bool:
    """Is the resolved Muğla Step8A the real frozen production artifact?

    Only then do the SAMPLING_FEASIBILITY.md inventory literals apply. Synthetic
    test fixtures never match this digest, so they are never held to them.
    """
    entry = inventory.get(SUBSAMPLED_EXPERIMENT) or {}
    return entry.get("sha256") == FROZEN_MUGLA_STEP8A_SHA256


# =============================================================================
# Population loading and strata
# =============================================================================
def load_primary_population(experiment_id: str,
                            experiments_root: Optional[Path] = None,
                            frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Canonical Step8A dataset -> 10-cell blocks -> primary population.

    Blocks are assigned BEFORE the valid/population filter, exactly as the
    canonical `assign_large_blocks` contract requires.
    """
    if frame is None:
        frame = pd.read_parquet(canonical_step8a_path(experiment_id, experiments_root))
    assigned = assign_large_blocks(frame, BLOCK_SIZE_CELLS)
    population = population_subset(assigned, POPULATION)
    if population.empty:
        raise MuglaSubsamplingError(
            f"{experiment_id!r}: primary population {POPULATION!r} is empty."
        )
    population = population.copy()
    population["label"] = population[TARGET_COLUMN].astype(int)
    n_positive = int((population["label"] == 1).sum())
    n_negative = int((population["label"] == 0).sum())
    if min(n_positive, n_negative) < STEP8B_MIN_POSITIVES_PER_POPULATION:
        raise MuglaSubsamplingError(
            f"{experiment_id!r}: population has {n_positive} positive / {n_negative} "
            f"negative rows; the canonical minimum is "
            f"{STEP8B_MIN_POSITIVES_PER_POPULATION} of each."
        )
    return population.reset_index(drop=True)


def stratum_id_of(large_block_id: Any, label: Any) -> str:
    return f"{large_block_id}|L{int(label)}"


def stratum_capacity_table(population: pd.DataFrame) -> pd.DataFrame:
    """One row per non-empty (large_block_id x label) stratum, sorted by id.

    Order-independent: `groupby(..., sort=True)` over a value pair, then a
    stable sort on `stratum_id`.
    """
    grouped = (
        population.groupby([BLOCK_COLUMN, "label"], sort=True)
        .size().rename("capacity").reset_index()
    )
    grouped["stratum_id"] = [
        stratum_id_of(block, label)
        for block, label in zip(grouped[BLOCK_COLUMN], grouped["label"])
    ]
    grouped = grouped.sort_values("stratum_id", kind="mergesort").reset_index(drop=True)
    return grouped[["stratum_id", BLOCK_COLUMN, "label", "capacity"]]


def hamilton_allocation(capacity_table: pd.DataFrame, target_total: int) -> pd.DataFrame:
    """Integer-exact proportional (largest-remainder) allocation.

        floor_s     = (c_s * N) // N_total
        remainder_s = (c_s * N) %  N_total

    The `N - sum(floor)` remaining units go to the largest remainders, ties
    broken by `stratum_id` ascending. Integer arithmetic is used so the tie set
    is exact rather than a floating-point artefact.
    """
    table = capacity_table.copy()
    capacity = table["capacity"].to_numpy(dtype=np.int64)
    population_total = int(capacity.sum())
    target_total = int(target_total)
    if target_total <= 0:
        raise MuglaSubsamplingError(f"target_sample_size must be positive; got {target_total}.")
    if target_total > population_total:
        raise MuglaSubsamplingError(
            f"Cannot draw {target_total} rows without replacement from a population "
            f"of {population_total}."
        )

    quota_numerator = capacity * target_total
    floor_allocation = quota_numerator // population_total
    remainder_numerator = quota_numerator % population_total
    shortfall = target_total - int(floor_allocation.sum())
    if shortfall < 0:
        raise MuglaSubsamplingError(
            f"Hamilton floor allocation overshot the target ({floor_allocation.sum()} "
            f"> {target_total}); the arithmetic contract is violated."
        )

    stratum_ids = table["stratum_id"].to_numpy()
    # lexsort keys are applied last-first: primary -remainder, secondary id.
    order = np.lexsort((stratum_ids, -remainder_numerator))
    rank = np.empty(len(table), dtype=np.int64)
    rank[order] = np.arange(len(table), dtype=np.int64)
    received = rank < shortfall

    table["quota_numerator"] = quota_numerator
    table["floor_allocation"] = floor_allocation
    table["remainder_numerator"] = remainder_numerator
    table["remainder_rank"] = rank
    table["received_remainder_unit"] = received
    table["allocation_count"] = floor_allocation + received.astype(np.int64)
    table["capacity_headroom"] = table["capacity"] - table["allocation_count"]
    table.attrs["population_total"] = population_total
    table.attrs["target_total"] = target_total
    table.attrs["floor_total"] = int(floor_allocation.sum())
    table.attrs["remainder_units"] = int(shortfall)
    if shortfall > 0:
        cut_value = int(np.sort(remainder_numerator)[::-1][shortfall - 1])
        tied = remainder_numerator == cut_value
        table.attrs["cut_remainder_numerator"] = cut_value
        table.attrs["strata_above_cut"] = int((remainder_numerator > cut_value).sum())
        table.attrs["strata_tied_at_cut"] = int(tied.sum())
        table.attrs["tie_units_awarded"] = int((tied & received).sum())
    else:
        table.attrs["cut_remainder_numerator"] = None
        table.attrs["strata_above_cut"] = 0
        table.attrs["strata_tied_at_cut"] = 0
        table.attrs["tie_units_awarded"] = 0
    return table


def assert_allocation_valid(table: pd.DataFrame, target_total: int) -> None:
    allocation = table["allocation_count"].to_numpy(dtype=np.int64)
    capacity = table["capacity"].to_numpy(dtype=np.int64)
    if int(allocation.sum()) != int(target_total):
        raise MuglaSubsamplingError(
            f"Allocation sums to {int(allocation.sum())}, not the required "
            f"{int(target_total)}."
        )
    over = table.loc[allocation > capacity, "stratum_id"].tolist()
    if over:
        raise MuglaSubsamplingError(
            f"Allocation exceeds stratum capacity for {over[:5]} "
            f"({len(over)} stratum/strata); no stratum may be over-drawn."
        )
    dropped = table.loc[allocation < 1, "stratum_id"].tolist()
    if dropped:
        raise MuglaSubsamplingError(
            f"Allocation dropped {len(dropped)} stratum/strata entirely (e.g. "
            f"{dropped[:5]}); every stratum must retain at least one cell."
        )


def prevalence_accounting(capacity_table: pd.DataFrame,
                          allocation_table: pd.DataFrame,
                          target_total: int) -> dict[str, Any]:
    """Prevalence drift and the rounding bound it must respect.

    The bound is `n_label1_strata / target_total`: largest-remainder rounding
    can move each positive stratum by at most one unit.
    """
    population_total = int(capacity_table["capacity"].sum())
    full_positives = int(capacity_table.loc[capacity_table["label"] == 1, "capacity"].sum())
    positive_mask = allocation_table["label"] == 1
    sampled_positives = int(allocation_table.loc[positive_mask, "allocation_count"].sum())
    n_positive_strata = int(positive_mask.sum())

    full_prevalence = full_positives / population_total
    sampled_prevalence = sampled_positives / int(target_total)
    drift = abs(sampled_prevalence - full_prevalence)
    bound = n_positive_strata / int(target_total)
    accounting = {
        "population_total": population_total,
        "population_positives": full_positives,
        "population_negatives": population_total - full_positives,
        "prevalence_full": full_prevalence,
        "sampled_positives": sampled_positives,
        "sampled_negatives": int(target_total) - sampled_positives,
        "prevalence_subsample": sampled_prevalence,
        "prevalence_absolute_drift": drift,
        "prevalence_relative_drift": (
            (sampled_prevalence - full_prevalence) / full_prevalence
            if full_prevalence else None
        ),
        "exact_proportional_positives": full_positives * int(target_total) / population_total,
        "n_positive_strata": n_positive_strata,
        "prevalence_drift_bound": bound,
        "prevalence_within_bound": drift <= bound,
    }
    if not accounting["prevalence_within_bound"]:
        raise MuglaSubsamplingError(
            f"Prevalence drift {drift:.8f} exceeds the rounding bound {bound:.8f}. "
            "The allocation does not preserve prevalence within rounding limits."
        )
    return accounting


def assert_production_inventory(capacity_table: pd.DataFrame,
                                allocation_table: pd.DataFrame,
                                prevalence: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if the real Muğla frame does not reproduce the frozen design.

    Only ever called when the observed Step8A digest is the production one.
    """
    expected = PRODUCTION_INVENTORY
    observed = {
        "population_rows": int(capacity_table["capacity"].sum()),
        "population_positives": prevalence["population_positives"],
        "population_negatives": prevalence["population_negatives"],
        "n_blocks": int(capacity_table[BLOCK_COLUMN].nunique()),
        "n_strata": int(len(capacity_table)),
        "n_positive_strata": int((capacity_table["label"] == 1).sum()),
        "floor_total": int(allocation_table.attrs["floor_total"]),
        "remainder_units": int(allocation_table.attrs["remainder_units"]),
        "strata_above_cut": int(allocation_table.attrs["strata_above_cut"]),
        "strata_tied_at_cut": int(allocation_table.attrs["strata_tied_at_cut"]),
        "tie_units_awarded": int(allocation_table.attrs["tie_units_awarded"]),
        "sampled_positives": prevalence["sampled_positives"],
        "sampled_negatives": prevalence["sampled_negatives"],
    }
    drift = {
        key: (expected[key], value)
        for key, value in observed.items() if expected[key] != value
    }
    if drift:
        detail = "; ".join(
            f"{key}: expected {exp}, observed {obs}" for key, (exp, obs) in drift.items()
        )
        raise MuglaSubsamplingError(
            "The frozen production Muğla frame no longer reproduces the design "
            f"inventory of docs/mugla_subsampling_design/. {detail}."
        )
    return observed


# =============================================================================
# Deterministic selection
# =============================================================================
def _blake_seed(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 32)


def repeat_seed(repeat_id: int) -> int:
    """Deterministic from the schema, the arm and the repeat, and nothing else."""
    return _blake_seed(f"{SCHEMA_VERSION}|{ARM_ID}|{int(repeat_id)}")


def stratum_seed(repeat_id: int, stratum_id: str) -> int:
    """Deterministic from the repeat and the stratum's own identity.

    Deliberately independent of the order in which strata are visited, so the
    selection cannot depend on iteration order or on input row order.
    """
    return _blake_seed(f"{SCHEMA_VERSION}|{int(repeat_id)}|{stratum_id}")


def select_repeat(population: pd.DataFrame, allocation_table: pd.DataFrame,
                  repeat_id: int) -> pd.DataFrame:
    """The frozen per-repeat selection.

    Within each stratum: sort by `cell_id` (stable identity), permute with the
    stratum's own deterministic seed, take the first `allocation_count`. No
    replacement, and row order of `population` cannot change the result.
    """
    if "stratum_id" not in population.columns:
        population = population.assign(stratum_id=[
            stratum_id_of(block, label)
            for block, label in zip(population[BLOCK_COLUMN], population["label"])
        ])
    ordered = population.sort_values("cell_id", kind="mergesort")
    positions = ordered.groupby("stratum_id", sort=True).indices

    allocation = dict(zip(allocation_table["stratum_id"],
                          allocation_table["allocation_count"]))
    missing = set(allocation) - set(positions)
    if missing:
        raise MuglaSubsamplingError(
            f"Repeat {repeat_id}: {len(missing)} allocated stratum/strata are absent "
            f"from the population frame (e.g. {sorted(missing)[:5]})."
        )

    chosen: list[np.ndarray] = []
    seeds: list[np.ndarray] = []
    for stratum, count in sorted(allocation.items()):
        count = int(count)
        if count <= 0:
            continue
        candidates = positions[stratum]
        if count > len(candidates):
            raise MuglaSubsamplingError(
                f"Repeat {repeat_id}, stratum {stratum!r}: allocation {count} exceeds "
                f"capacity {len(candidates)}."
            )
        seed = stratum_seed(repeat_id, stratum)
        permutation = np.random.default_rng(seed).permutation(len(candidates))
        picked = candidates[permutation[:count]]
        chosen.append(picked)
        seeds.append(np.full(count, seed, dtype=np.int64))

    if not chosen:
        raise MuglaSubsamplingError(f"Repeat {repeat_id}: selection is empty.")
    selected_positions = np.concatenate(chosen)
    selection = ordered.iloc[selected_positions].copy()
    selection["sampling_seed"] = np.concatenate(seeds)
    selection["repeat_id"] = int(repeat_id)
    selection["repeat_seed"] = repeat_seed(repeat_id)
    selection = selection.sort_values("cell_id", kind="mergesort").reset_index(drop=True)

    if selection["cell_id"].duplicated().any():
        raise MuglaSubsamplingError(
            f"Repeat {repeat_id}: duplicate cell_id in the selection -- sampling must "
            "be without replacement."
        )
    return selection


def sample_hash(cell_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for cell_id in sorted(str(value) for value in cell_ids):
        digest.update(cell_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


SELECTED_CELL_COLUMNS: list[str] = [
    "repeat_id", "cell_id", "large_block_id", "label", "fold_id", "sampling_seed",
    "stratum_id", "allocation_count", "mugla_step8a_sha256", "repeat_seed",
    "row_500m", "col_500m",
]


def build_selected_cells(population: pd.DataFrame, allocation_table: pd.DataFrame,
                         fold_mapping: pd.DataFrame, mugla_sha256: str,
                         n_repeats: int, target_total: int) -> pd.DataFrame:
    """Every repeat's frozen selection, with block, fold and seed provenance."""
    allocation_lookup = dict(zip(allocation_table["stratum_id"],
                                 allocation_table["allocation_count"]))
    fold_lookup = dict(zip(fold_mapping["cell_id"], fold_mapping["fold_id"]))

    enriched = population.copy()
    enriched["stratum_id"] = [
        stratum_id_of(block, label)
        for block, label in zip(enriched[BLOCK_COLUMN], enriched["label"])
    ]

    frames: list[pd.DataFrame] = []
    for repeat_id in range(int(n_repeats)):
        selection = select_repeat(enriched, allocation_table, repeat_id)
        if len(selection) != int(target_total):
            raise MuglaSubsamplingError(
                f"Repeat {repeat_id}: selected {len(selection)} rows, not the required "
                f"{int(target_total)}."
            )
        selection["allocation_count"] = selection["stratum_id"].map(allocation_lookup).astype("int64")
        selection["fold_id"] = selection["cell_id"].map(fold_lookup)
        if selection["fold_id"].isna().any():
            unmapped = int(selection["fold_id"].isna().sum())
            raise MuglaSubsamplingError(
                f"Repeat {repeat_id}: {unmapped} selected cell(s) have no inherited "
                "fold assignment."
            )
        selection["fold_id"] = selection["fold_id"].astype("int64")
        selection["mugla_step8a_sha256"] = mugla_sha256
        frames.append(selection[SELECTED_CELL_COLUMNS])

    selected = pd.concat(frames, ignore_index=True)
    return selected


# =============================================================================
# Inherited fold mapping
# =============================================================================
FOLD_MAPPING_COLUMNS: list[str] = [
    "cell_id", "large_block_id", "frozen_block_id", "fold_id", "label",
    "source_artifact_path", "source_artifact_sha256", "fold_source",
]


def load_frozen_fold_mapping(population: pd.DataFrame,
                             experiments_root: Optional[Path] = None,
                             ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The full-Muğla block->fold mapping, loaded per `cell_id`.

    Folds are NEVER re-optimised here: this is the frozen 5-fold strict
    StratifiedGroupKFold mapping produced by the canonical 10-cell condition,
    and every repeat inherits it unchanged.
    """
    directory = within_reference_dir(experiments_root)
    artifact = directory / WITHIN_REFERENCE_OOF_NAME
    if not artifact.is_file():
        raise MuglaSubsamplingError(
            f"Frozen full-Mugla fold artifact not found: {artifact}. This analysis "
            "inherits the canonical Step8 fold mapping and never rebuilds it silently."
        )
    frozen = pd.read_parquet(artifact)
    required = {"cell_id", "fold_id", "spatial_block_id"}
    missing = required - set(frozen.columns)
    if missing:
        raise MuglaSubsamplingError(
            f"Frozen fold artifact {artifact} lacks required column(s) {sorted(missing)}."
        )
    if frozen["cell_id"].duplicated().any():
        raise MuglaSubsamplingError(
            f"Frozen fold artifact {artifact} contains duplicate cell_id values."
        )

    population_cells = set(population["cell_id"].astype(str))
    frozen_cells = set(frozen["cell_id"].astype(str))
    if population_cells - frozen_cells:
        raise MuglaSubsamplingError(
            f"{len(population_cells - frozen_cells)} primary-population cell(s) have no "
            f"fold in {artifact}; the mapping must cover the whole population."
        )
    if frozen_cells - population_cells:
        raise MuglaSubsamplingError(
            f"{len(frozen_cells - population_cells)} fold-artifact cell(s) are outside the "
            "primary population; the frozen mapping is not the one this analysis expects."
        )

    mapping = population[["cell_id", BLOCK_COLUMN, "label"]].copy()
    frozen_slim = frozen[["cell_id", "fold_id", "spatial_block_id"]].rename(
        columns={"spatial_block_id": "frozen_block_id"})
    mapping = mapping.merge(frozen_slim, on="cell_id", how="left", validate="one_to_one")
    mapping["fold_id"] = mapping["fold_id"].astype("int64")
    mapping["source_artifact_path"] = str(artifact)
    digest = sha256_file(artifact)
    mapping["source_artifact_sha256"] = digest
    mapping["fold_source"] = "persisted_artifact"
    mapping = mapping.sort_values("cell_id", kind="mergesort").reset_index(drop=True)

    provenance = assert_fold_contract(mapping)
    provenance.update({
        "artifact_path": str(artifact),
        "artifact_sha256": digest,
        "artifact_schema": FOLD_ARTIFACT_SCHEMA,
        "fold_source": "persisted_artifact",
        "reoptimised_per_repeat": False,
        "splitter": "StratifiedGroupKFold",
        "shuffle": True,
        "random_state": FOLD_RANDOM_STATE,
        "strict_folds": True,
        "fold_count": FOLD_COUNT,
        "block_size_cells": BLOCK_SIZE_CELLS,
        "nominal_scale": BLOCK_NOMINAL_SCALE,
    })
    return mapping[FOLD_MAPPING_COLUMNS], provenance


def assert_fold_contract(mapping: pd.DataFrame) -> dict[str, Any]:
    """No block spans two folds; the fold count is the frozen one."""
    spans = mapping.groupby(BLOCK_COLUMN)["fold_id"].nunique()
    leaking = spans[spans > 1]
    if len(leaking):
        raise MuglaSubsamplingError(
            f"{len(leaking)} spatial block(s) span more than one fold (e.g. "
            f"{list(leaking.index[:5])}); the inherited mapping is not block-consistent."
        )
    bijection = mapping.groupby("frozen_block_id")[BLOCK_COLUMN].nunique()
    if len(bijection) and int(bijection.max()) > 1:
        raise MuglaSubsamplingError(
            "The frozen block id and the large_block_id do not describe the same "
            "partition; the block-scale contract is violated."
        )
    folds = sorted(int(value) for value in mapping["fold_id"].unique())
    if len(folds) != FOLD_COUNT:
        raise MuglaSubsamplingError(
            f"The inherited mapping has {len(folds)} folds; the frozen contract is "
            f"{FOLD_COUNT}. Fold counts are never reduced."
        )
    return {
        "n_cells": int(len(mapping)),
        "n_blocks": int(mapping[BLOCK_COLUMN].nunique()),
        "folds": folds,
        "blocks_spanning_folds": 0,
    }


def fold_composition(selected: pd.DataFrame) -> pd.DataFrame:
    """Per (repeat, fold) row and positive counts of the selection."""
    return (
        selected.groupby(["repeat_id", "fold_id"], sort=True)
        .agg(rows=("cell_id", "size"), positives=("label", "sum"))
        .reset_index()
    )


def assert_selection_fold_contract(selected: pd.DataFrame, n_repeats: int) -> dict[str, Any]:
    """Every fold usable on both sides, in every repeat; coverage exactly once."""
    composition = fold_composition(selected)
    per_repeat_rows: dict[int, list[int]] = {}
    per_repeat_positives: dict[int, list[int]] = {}
    for repeat_id, group in composition.groupby("repeat_id", sort=True):
        folds = sorted(int(value) for value in group["fold_id"])
        if len(folds) != FOLD_COUNT:
            raise MuglaSubsamplingError(
                f"Repeat {repeat_id}: the selection covers {len(folds)} folds, not "
                f"{FOLD_COUNT}."
            )
        rows = [int(value) for value in group.sort_values("fold_id")["rows"]]
        positives = [int(value) for value in group.sort_values("fold_id")["positives"]]
        total_positives = sum(positives)
        total_rows = sum(rows)
        for fold_id, fold_rows, fold_positives in zip(folds, rows, positives):
            if fold_positives < 1:
                raise MuglaSubsamplingError(
                    f"Repeat {repeat_id}, fold {fold_id}: evaluation side has no "
                    "positive rows."
                )
            if fold_rows - fold_positives < 1:
                raise MuglaSubsamplingError(
                    f"Repeat {repeat_id}, fold {fold_id}: evaluation side has no "
                    "negative rows."
                )
            if total_positives - fold_positives < 1:
                raise MuglaSubsamplingError(
                    f"Repeat {repeat_id}, fold {fold_id}: training side has no "
                    "positive rows."
                )
            if (total_rows - fold_rows - (total_positives - fold_positives)) < 1:
                raise MuglaSubsamplingError(
                    f"Repeat {repeat_id}, fold {fold_id}: training side has no "
                    "negative rows."
                )
        per_repeat_rows[int(repeat_id)] = rows
        per_repeat_positives[int(repeat_id)] = positives

    distinct_rows = {tuple(value) for value in per_repeat_rows.values()}
    distinct_positives = {tuple(value) for value in per_repeat_positives.values()}
    if len(distinct_rows) != 1 or len(distinct_positives) != 1:
        raise MuglaSubsamplingError(
            "Per-fold composition differs between repeats; the allocation is "
            "repeat-invariant by contract, so this cannot happen."
        )
    return {
        "per_fold_rows": list(next(iter(distinct_rows))),
        "per_fold_positives": list(next(iter(distinct_positives))),
        "identical_across_repeats": True,
        "n_repeats": int(n_repeats),
    }


def assert_full_oof_coverage(coverage: np.ndarray, context: str) -> None:
    coverage = np.asarray(coverage)
    if not np.all(coverage == 1):
        unpredicted = int((coverage == 0).sum())
        duplicated = int((coverage > 1).sum())
        raise MuglaSubsamplingError(
            f"{context}: OOF coverage violated -- {unpredicted} row(s) predicted zero "
            f"times and {duplicated} row(s) predicted more than once."
        )


# =============================================================================
# Fit registry
# =============================================================================
class FitRegistry:
    """Memoises fit RESULTS by fit identity.

    Results (prediction vectors), not fitted estimators, are cached: memory
    stays flat while the sharing contract stays structural. A second request
    for the same identity can never trigger a second fit -- which is exactly
    what makes one Muğla-as-source fit serve both targets.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self.fit_count = 0
        self.reuse_count = 0

    def get_or_fit(self, fit_id: str, arm: str, compute: Callable[[], Any]) -> Any:
        entry = self._entries.get(fit_id)
        if entry is not None:
            entry["reference_count"] += 1
            self.reuse_count += 1
            if entry["result"] is None:
                raise MuglaSubsamplingError(
                    f"Fit identity {fit_id!r} was released before its last reuse; "
                    "the registry would have to refit, which the contract forbids."
                )
            return entry["result"]
        result = compute()
        self._entries[fit_id] = {"arm": arm, "result": result, "reference_count": 1}
        self.fit_count += 1
        return result

    def release(self, prefix: str) -> int:
        """Drop cached results whose identity starts with `prefix`.

        Accounting is kept, so releasing memory never changes reported totals.
        """
        released = [key for key in self._entries if key.startswith(prefix)]
        for key in released:
            self._entries[key]["result"] = None
        return len(released)

    def identities(self) -> dict[str, dict[str, int]]:
        return {
            fit_id: {"arm": entry["arm"], "reference_count": entry["reference_count"]}
            for fit_id, entry in self._entries.items()
        }

    def accounting(self) -> dict[str, Any]:
        by_arm = {arm: 0 for arm in ARMS}
        references = {arm: 0 for arm in ARMS}
        for entry in self._entries.values():
            by_arm[entry["arm"]] += 1
            references[entry["arm"]] += entry["reference_count"]
        return {
            "unique_fits": self.fit_count,
            "within_fits": by_arm[ARM_WITHIN],
            "source_fits": by_arm[ARM_SOURCE],
            "target_fits": by_arm[ARM_TARGET],
            "reuse_events": self.reuse_count,
            "references_by_arm": references,
            "source_reuse_per_fit": (
                references[ARM_SOURCE] / by_arm[ARM_SOURCE] if by_arm[ARM_SOURCE] else None
            ),
        }


def within_fit_identity(repeat_id: int, fold_id: int, family: str) -> str:
    return f"{ARM_WITHIN}|{int(repeat_id)}|{int(fold_id)}|{family}"


def source_fit_identity(repeat_id: int, family: str) -> str:
    """Deliberately excludes the target: a source-only fit never sees the target,
    so one fit serves both directions and refitting per direction is forbidden."""
    return f"{ARM_SOURCE}|{int(repeat_id)}|{family}"


def fit_and_predict(train_frame: pd.DataFrame, eval_frames: Sequence[pd.DataFrame],
                    feature_list: Sequence[str]) -> list[np.ndarray]:
    """The ONLY place a model is fitted.

    Canonical `build_pipeline` -> plain `fit` -> `predict_proba`, once per call,
    with one prediction vector per evaluation frame. No `sample_weight`, no
    oversampling, no tuning, no threshold selection. Preprocessing lives inside
    the Pipeline, so imputers and the encoder are fitted on the training frame
    only and never see an evaluation row.
    """
    features = list(feature_list)
    check_no_forbidden_features(features)
    pipeline = build_pipeline(features, MODEL_NAME, ESTIMATOR_SEED)
    pipeline.fit(train_frame[features], train_frame[TARGET_COLUMN].astype(int))
    return [
        pipeline.predict_proba(frame[features])[:, 1].astype(np.float64)
        for frame in eval_frames
    ]


# =============================================================================
# References
# =============================================================================
REFERENCE_COLUMNS: list[str] = [
    "arm", "direction", "model_family", "metric", "full_reference_value",
    "reference_artifact_path", "reference_artifact_sha256", "reference_n_rows",
    "reference_n_positives", "recomputed_from_predictions", "recomputation_matches",
]
RECOMPUTATION_TOLERANCE = 1e-9


def _metric_from_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Optional[float]]:
    metrics = compute_binary_metrics(np.asarray(y_true), np.asarray(y_prob))
    return {metric: metrics.get(metric) for metric in METRICS}


def load_within_reference(experiments_root: Optional[Path] = None) -> dict[str, Any]:
    """Frozen full-Muğla 10-cell within-region OOF reference.

    Values are read from the artifact; nothing is hard-coded. They are then
    independently recomputed from the artifact's own probability vectors.
    """
    directory = within_reference_dir(experiments_root)
    metrics_path = directory / WITHIN_REFERENCE_METRICS_NAME
    oof_path = directory / WITHIN_REFERENCE_OOF_NAME
    if not metrics_path.is_file():
        raise MuglaSubsamplingError(
            f"Frozen within-region reference metrics not found: {metrics_path}."
        )
    if not oof_path.is_file():
        raise MuglaSubsamplingError(
            f"Frozen within-region reference predictions not found: {oof_path}."
        )
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    frozen = pd.read_parquet(oof_path)
    y_true = frozen[TARGET_COLUMN].astype(int).to_numpy()

    values: dict[str, dict[str, Optional[float]]] = {}
    recomputed: dict[str, dict[str, Optional[float]]] = {}
    for family in MODEL_FAMILIES:
        values[family] = {
            "roc_auc": document.get(f"{family}_roc_auc"),
            "pr_auc": document.get(f"{family}_pr_auc"),
            "brier_score": document.get(f"{family}_brier"),
        }
        recomputed[family] = _metric_from_predictions(
            y_true, frozen[f"{family}_probability"].to_numpy())
    return {
        "arm": ARM_WITHIN,
        "direction": WITHIN_DIRECTION,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_path": str(oof_path),
        "predictions_sha256": sha256_file(oof_path),
        "folds_sha256": (
            sha256_file(directory / WITHIN_REFERENCE_FOLDS_NAME)
            if (directory / WITHIN_REFERENCE_FOLDS_NAME).is_file() else None
        ),
        "block_manifest_sha256": (
            sha256_file(directory / WITHIN_REFERENCE_MANIFEST_NAME)
            if (directory / WITHIN_REFERENCE_MANIFEST_NAME).is_file() else None
        ),
        "n_rows": int(len(frozen)),
        "n_positives": int(y_true.sum()),
        "values": values,
        "recomputed": recomputed,
    }


def load_transfer_reference(source_id: str, target_id: str,
                            output_root: Optional[Path] = None) -> dict[str, Any]:
    """Frozen canonical raw-transfer reference for one direction."""
    direction = direction_token(source_id, target_id)
    directory = transfer_reference_dir(source_id, target_id, output_root)
    metrics_path = directory / TRANSFER_METRICS_NAME
    predictions_path = directory / TRANSFER_PREDICTIONS_NAME
    if not metrics_path.is_file():
        raise MuglaSubsamplingError(
            f"Frozen raw-transfer metrics not found for {direction}: {metrics_path}."
        )
    if not predictions_path.is_file():
        raise MuglaSubsamplingError(
            f"Frozen raw-transfer predictions not found for {direction}: {predictions_path}."
        )
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    entries = [
        entry for entry in document.get("results", [])
        if entry.get("transfer_direction") == direction
        and entry.get("population") == POPULATION
    ]
    if len(entries) != 1:
        raise MuglaSubsamplingError(
            f"Expected exactly one {direction} / {POPULATION} result in {metrics_path}; "
            f"found {len(entries)}."
        )
    entry = entries[0]
    if entry.get("skipped"):
        raise MuglaSubsamplingError(
            f"The frozen reference for {direction} is marked skipped: {entry.get('reason')}."
        )

    predictions = pd.read_parquet(predictions_path)
    subset = predictions[
        (predictions["transfer_direction"] == direction)
        & (predictions["population"] == POPULATION)
    ].copy()
    if subset.empty:
        raise MuglaSubsamplingError(
            f"No {direction} / {POPULATION} rows in {predictions_path}."
        )
    if subset["target_cell_id"].duplicated().any():
        raise MuglaSubsamplingError(
            f"Duplicate target_cell_id in the frozen predictions for {direction}."
        )
    y_true = subset[TARGET_COLUMN].astype(int).to_numpy()

    values: dict[str, dict[str, Optional[float]]] = {}
    recomputed: dict[str, dict[str, Optional[float]]] = {}
    for family in MODEL_FAMILIES:
        family_metrics = entry.get(f"{family}_metrics") or {}
        values[family] = {metric: family_metrics.get(metric) for metric in METRICS}
        recomputed[family] = _metric_from_predictions(
            y_true, subset[f"{family}_probability"].to_numpy())
    return {
        "direction": direction,
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "resolved_inputs": document.get("resolved_inputs"),
        "n_rows": int(len(subset)),
        "n_positives": int(y_true.sum()),
        "values": values,
        "recomputed": recomputed,
        "predictions": subset,
    }


def build_reference_inventory(experiments_root: Optional[Path] = None,
                              output_root: Optional[Path] = None) -> dict[str, Any]:
    """Every full-population reference this analysis compares against."""
    inventory: dict[str, Any] = {
        "within": load_within_reference(experiments_root),
        "source": {}, "target": {},
    }
    for source_id, target_id in SOURCE_PAIRS:
        inventory["source"][direction_token(source_id, target_id)] = (
            load_transfer_reference(source_id, target_id, output_root))
    for source_id, target_id in TARGET_PAIRS:
        inventory["target"][direction_token(source_id, target_id)] = (
            load_transfer_reference(source_id, target_id, output_root))
    return inventory


def _reference_rows(entry: dict[str, Any], arm: str, direction: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        for metric in METRICS:
            stored = entry["values"][family].get(metric)
            recomputed = entry["recomputed"][family].get(metric)
            matches = (
                stored is not None and recomputed is not None
                and abs(float(stored) - float(recomputed)) <= RECOMPUTATION_TOLERANCE
            )
            rows.append({
                "arm": arm,
                "direction": direction,
                "model_family": family,
                "metric": metric,
                "full_reference_value": stored,
                "reference_artifact_path": entry["metrics_path"],
                "reference_artifact_sha256": entry["metrics_sha256"],
                "reference_n_rows": entry["n_rows"],
                "reference_n_positives": entry["n_positives"],
                "recomputed_from_predictions": recomputed,
                "recomputation_matches": matches,
            })
    return rows


def build_reference_metrics(reference_inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _reference_rows(reference_inventory["within"], ARM_WITHIN, WITHIN_DIRECTION)
    for direction, entry in reference_inventory["source"].items():
        rows += _reference_rows(entry, ARM_SOURCE, direction)
    for direction, entry in reference_inventory["target"].items():
        rows += _reference_rows(entry, ARM_TARGET, direction)
    failures = [
        f"{row['arm']}/{row['direction']}/{row['model_family']}/{row['metric']}"
        for row in rows if not row["recomputation_matches"]
    ]
    if failures:
        raise MuglaSubsamplingError(
            "Frozen reference values do not reproduce from their own probability "
            f"vectors: {failures[:6]} ({len(failures)} of {len(rows)}). Refusing to "
            "compare against a reference that cannot be recomputed."
        )
    return rows


def reference_lookup(reference_rows: Sequence[dict[str, Any]]
                     ) -> dict[tuple[str, str, str, str], Optional[float]]:
    return {
        (row["arm"], row["direction"], row["model_family"], row["metric"]):
            row["full_reference_value"]
        for row in reference_rows
    }


# =============================================================================
# Arms
# =============================================================================
WITHIN_PREDICTION_COLUMNS = [
    "repeat_id", "cell_id", "fold_id", "large_block_id", TARGET_COLUMN,
    "baseline_probability", "thermal_probability",
]
SOURCE_PREDICTION_COLUMNS = [
    "repeat_id", "direction", "target_experiment_id", "target_cell_id",
    "target_spatial_block_id", TARGET_COLUMN,
    "baseline_probability", "thermal_probability",
]
TARGET_PREDICTION_COLUMNS = [
    "repeat_id", "direction", "source_experiment_id", "target_cell_id", "fold_id",
    TARGET_COLUMN, "baseline_probability", "thermal_probability",
    "reused_from_artifact", "source_artifact_sha256",
]


def run_within_arm(repeat_frame: pd.DataFrame, repeat_id: int,
                   registry: FitRegistry) -> pd.DataFrame:
    """5-fold spatial OOF on the sampled Muğla frame, using inherited folds."""
    frame = repeat_frame.reset_index(drop=True)
    fold_ids = frame["fold_id"].to_numpy()
    coverage = np.zeros(len(frame), dtype=int)
    predictions = {family: np.full(len(frame), np.nan) for family in MODEL_FAMILIES}

    for fold_id in sorted(int(value) for value in np.unique(fold_ids)):
        test_mask = fold_ids == fold_id
        train_mask = ~test_mask
        train_frame = frame.loc[train_mask]
        eval_frame = frame.loc[test_mask]
        train_blocks = set(train_frame[BLOCK_COLUMN])
        eval_blocks = set(eval_frame[BLOCK_COLUMN])
        if train_blocks & eval_blocks:
            raise MuglaSubsamplingError(
                f"Repeat {repeat_id}, fold {fold_id}: a spatial block appears on both "
                "sides of the split."
            )
        for family in MODEL_FAMILIES:
            identity = within_fit_identity(repeat_id, fold_id, family)
            vector = registry.get_or_fit(
                identity, ARM_WITHIN,
                lambda f=family, tr=train_frame, ev=eval_frame: fit_and_predict(
                    tr, [ev], FEATURE_LISTS[f])[0],
            )
            predictions[family][test_mask] = vector
        coverage[test_mask] += 1

    assert_full_oof_coverage(coverage, f"within-Mugla repeat {repeat_id}")
    output = frame[["repeat_id", "cell_id", "fold_id", BLOCK_COLUMN]].copy()
    output[TARGET_COLUMN] = frame["label"].astype(int)
    for family in MODEL_FAMILIES:
        output[f"{family}_probability"] = predictions[family]
    registry.release(f"{ARM_WITHIN}|{int(repeat_id)}|")
    return output[WITHIN_PREDICTION_COLUMNS]


def run_source_arm(repeat_frame: pd.DataFrame, repeat_id: int,
                   target_frames: dict[str, pd.DataFrame],
                   registry: FitRegistry) -> pd.DataFrame:
    """Sampled Muğla as source; both targets served by ONE fit per family."""
    ordered_targets = [target_id for _, target_id in SOURCE_PAIRS]
    frames: list[pd.DataFrame] = []
    per_direction: dict[str, dict[str, np.ndarray]] = {}

    for family in MODEL_FAMILIES:
        identity = source_fit_identity(repeat_id, family)

        def _compute(f=family) -> dict[str, np.ndarray]:
            vectors = fit_and_predict(
                repeat_frame,
                [target_frames[target_id] for target_id in ordered_targets],
                FEATURE_LISTS[f],
            )
            return dict(zip(ordered_targets, vectors))

        for source_id, target_id in SOURCE_PAIRS:
            # One `get_or_fit` call per direction: the first fits, the second
            # is a registry reuse. Refitting per direction is forbidden.
            vectors = registry.get_or_fit(identity, ARM_SOURCE, _compute)
            direction = direction_token(source_id, target_id)
            per_direction.setdefault(direction, {})[family] = vectors[target_id]

    for source_id, target_id in SOURCE_PAIRS:
        direction = direction_token(source_id, target_id)
        target_frame = target_frames[target_id]
        block_column = (
            BLOCK_COLUMN if BLOCK_COLUMN in target_frame.columns else None
        )
        rows = pd.DataFrame({
            "repeat_id": int(repeat_id),
            "direction": direction,
            "target_experiment_id": target_id,
            "target_cell_id": target_frame["cell_id"].to_numpy(),
            "target_spatial_block_id": (
                target_frame[block_column].to_numpy() if block_column else None
            ),
            TARGET_COLUMN: target_frame["label"].astype(int).to_numpy(),
        })
        for family in MODEL_FAMILIES:
            rows[f"{family}_probability"] = per_direction[direction][family]
        frames.append(rows[SOURCE_PREDICTION_COLUMNS])

    registry.release(f"{ARM_SOURCE}|{int(repeat_id)}|")
    return pd.concat(frames, ignore_index=True)


def run_target_arm(repeat_frame: pd.DataFrame, repeat_id: int,
                   reference_inventory: dict[str, Any]) -> pd.DataFrame:
    """Full-source models evaluated on the repeat's Muğla cells. ZERO fits.

    The frozen raw-transfer artifacts hold a per-cell probability for every
    Muğla primary cell, so restricting to a repeat's selection and recomputing
    is exact rather than approximate. No model is constructed here.
    """
    selected = repeat_frame[["cell_id", "fold_id", "label"]].copy()
    frames: list[pd.DataFrame] = []
    for source_id, target_id in TARGET_PAIRS:
        direction = direction_token(source_id, target_id)
        entry = reference_inventory["target"][direction]
        frozen = entry["predictions"]
        merged = selected.merge(
            frozen[["target_cell_id", TARGET_COLUMN,
                    "baseline_probability", "thermal_probability"]],
            left_on="cell_id", right_on="target_cell_id", how="left",
            validate="one_to_one",
        )
        if merged["target_cell_id"].isna().any():
            missing = int(merged["target_cell_id"].isna().sum())
            raise MuglaSubsamplingError(
                f"Repeat {repeat_id}, {direction}: {missing} selected cell(s) are absent "
                "from the frozen raw-transfer predictions; the artifact does not cover "
                "the Mugla primary population."
            )
        if not np.array_equal(merged["label"].astype(int).to_numpy(),
                              merged[TARGET_COLUMN].astype(int).to_numpy()):
            raise MuglaSubsamplingError(
                f"Repeat {repeat_id}, {direction}: labels in the frozen predictions "
                "disagree with the canonical Step8A labels."
            )
        rows = pd.DataFrame({
            "repeat_id": int(repeat_id),
            "direction": direction,
            "source_experiment_id": source_id,
            "target_cell_id": merged["cell_id"].to_numpy(),
            "fold_id": merged["fold_id"].to_numpy(),
            TARGET_COLUMN: merged[TARGET_COLUMN].astype(int).to_numpy(),
            "baseline_probability": merged["baseline_probability"].to_numpy(),
            "thermal_probability": merged["thermal_probability"].to_numpy(),
            "reused_from_artifact": True,
            "source_artifact_sha256": entry["predictions_sha256"],
        })
        frames.append(rows[TARGET_PREDICTION_COLUMNS])
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Repeat metrics
# =============================================================================
REPEAT_METRIC_COLUMNS: list[str] = [
    "arm", "direction", "model_family", "metric", "repeat_id",
    "full_reference_value", "subsample_value", "natural_delta", "oriented_delta",
    "metric_orientation", "n_eval_rows", "n_eval_positives", "n_fits_consumed",
    "sample_hash",
]

FITS_CONSUMED = {ARM_WITHIN: FOLD_COUNT, ARM_SOURCE: 1, ARM_TARGET: 0}


def _repeat_metric_rows(arm: str, direction: str, repeat_id: int,
                        y_true: np.ndarray, probabilities: dict[str, np.ndarray],
                        references: dict[tuple[str, str, str, str], Optional[float]],
                        repeat_sample_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_rows = int(len(y_true))
    n_positives = int(np.asarray(y_true).sum())
    for family in MODEL_FAMILIES:
        computed = _metric_from_predictions(y_true, probabilities[family])
        for metric in METRICS:
            value = computed.get(metric)
            if value is None:
                raise MuglaSubsamplingError(
                    f"{arm}/{direction}/{family}/{metric} repeat {repeat_id}: metric is "
                    "undefined (a class is absent). No repeat or direction may be "
                    "silently skipped."
                )
            reference = references.get((arm, direction, family, metric))
            rows.append({
                "arm": arm,
                "direction": direction,
                "model_family": family,
                "metric": metric,
                "repeat_id": int(repeat_id),
                "full_reference_value": reference,
                "subsample_value": float(value),
                "natural_delta": natural_delta(value, reference),
                "oriented_delta": oriented_delta(metric, value, reference),
                "metric_orientation": metric_orientation(metric),
                "n_eval_rows": n_rows,
                "n_eval_positives": n_positives,
                "n_fits_consumed": FITS_CONSUMED[arm],
                "sample_hash": repeat_sample_hash,
            })
    return rows


# =============================================================================
# Summarisation
# =============================================================================
def subsampling_interval(values: Sequence[Optional[float]]) -> dict[str, Optional[float]]:
    """Median and 2.5/97.5 percentiles over repeats.

    This describes variability across WHICH CELLS WERE SELECTED and nothing
    else. It is not a confidence interval and supports no significance claim.
    """
    clean = [
        float(value) for value in values
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    ]
    if not clean:
        return {
            "median": None, "interval_lower": None, "interval_upper": None,
            "minimum": None, "maximum": None, "n_repeats_observed": 0,
        }
    array = np.asarray(clean, dtype=float)
    return {
        "median": float(np.median(array)),
        "interval_lower": float(np.percentile(array, INTERVAL_PCT_LOW,
                                              method=PERCENTILE_METHOD)),
        "interval_upper": float(np.percentile(array, INTERVAL_PCT_HIGH,
                                              method=PERCENTILE_METHOD)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "n_repeats_observed": int(array.size),
    }


def reference_position(metric: str, full_reference_value: Optional[float],
                       interval_lower: Optional[float],
                       interval_upper: Optional[float]) -> Optional[str]:
    """Where the full reference sits relative to the subsampling range.

    Compared on the ORIENTED scale so the token means the same thing for Brier
    as for the AUCs. Purely descriptive: it carries no evidential weight.
    """
    if full_reference_value is None or interval_lower is None or interval_upper is None:
        return None
    reference = oriented_value(metric, full_reference_value)
    bounds = sorted(
        value for value in (oriented_value(metric, interval_lower),
                            oriented_value(metric, interval_upper))
    )
    if reference < bounds[0]:
        return POSITION_BELOW
    if reference > bounds[1]:
        return POSITION_ABOVE
    return POSITION_INSIDE


def interpretation_sentence(position_token: Optional[str]) -> Optional[str]:
    if position_token is None:
        return None
    if position_token == POSITION_INSIDE:
        return SENTENCE_INSIDE
    if position_token in (POSITION_BELOW, POSITION_ABOVE):
        return SENTENCE_OUTSIDE
    raise MuglaSubsamplingError(f"Unknown position token {position_token!r}.")


SUMMARY_COLUMNS: list[str] = [
    "arm", "direction", "model_family", "metric", "full_reference_value",
    "n_repeats_observed", "subsample_median",
    "subsampling_interval_lower", "subsampling_interval_upper",
    "subsample_minimum", "subsample_maximum",
    "oriented_delta_median", "oriented_delta_interval_lower",
    "oriented_delta_interval_upper", "oriented_delta_minimum",
    "oriented_delta_maximum", "reference_position", "interpretation_sentence",
]


def build_subsampling_summary(repeat_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, direction in all_direction_rows():
        for family in MODEL_FAMILIES:
            for metric in METRICS:
                slice_ = repeat_metrics[
                    (repeat_metrics["arm"] == arm)
                    & (repeat_metrics["direction"] == direction)
                    & (repeat_metrics["model_family"] == family)
                    & (repeat_metrics["metric"] == metric)
                ]
                if slice_.empty:
                    raise MuglaSubsamplingError(
                        f"No repeat rows for {arm}/{direction}/{family}/{metric}; no "
                        "direction may be silently skipped."
                    )
                values = subsampling_interval(slice_["subsample_value"].tolist())
                deltas = subsampling_interval(slice_["oriented_delta"].tolist())
                references = slice_["full_reference_value"].dropna().unique()
                reference = float(references[0]) if len(references) else None
                position = reference_position(
                    metric, reference, values["interval_lower"], values["interval_upper"])
                rows.append({
                    "arm": arm,
                    "direction": direction,
                    "model_family": family,
                    "metric": metric,
                    "full_reference_value": reference,
                    "n_repeats_observed": values["n_repeats_observed"],
                    "subsample_median": values["median"],
                    "subsampling_interval_lower": values["interval_lower"],
                    "subsampling_interval_upper": values["interval_upper"],
                    "subsample_minimum": values["minimum"],
                    "subsample_maximum": values["maximum"],
                    "oriented_delta_median": deltas["median"],
                    "oriented_delta_interval_lower": deltas["interval_lower"],
                    "oriented_delta_interval_upper": deltas["interval_upper"],
                    "oriented_delta_minimum": deltas["minimum"],
                    "oriented_delta_maximum": deltas["maximum"],
                    "reference_position": position,
                    "interpretation_sentence": interpretation_sentence(position),
                })
    return rows


def scan_forbidden_tokens(root: Path) -> list[dict[str, str]]:
    """Literal scan of every emitted text artifact for significance vocabulary."""
    hits: list[dict[str, str]] = []
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        if path.suffix.lower() not in (".json", ".csv", ".md", ".txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeDecodeError):
            continue
        for token in FORBIDDEN_TOKENS:
            if token in text:
                hits.append({"path": str(path.relative_to(root)), "token": token})
    return hits


# =============================================================================
# Scientific configuration + identity
# =============================================================================
def build_scientific_config(experiment_ids: Sequence[str]) -> dict[str, Any]:
    """The object hashed into `analysis_id`.

    Deliberately excludes wall-clock time, git commit and package versions, so
    re-running the same frozen contract on the same frozen inputs lands in the
    same namespace.
    """
    classifier = build_classifier(MODEL_NAME, ESTIMATOR_SEED)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "experiments": list(experiment_ids),
        "excluded_experiments": dict(EXCLUDED_EXPERIMENTS),
        "subsampled_experiment": SUBSAMPLED_EXPERIMENT,
        "size_reference_experiment": SIZE_REFERENCE_EXPERIMENT,
        "population": POPULATION,
        "valid_universe": VALID_UNIVERSE,
        "arm_id": ARM_ID,
        "target_sample_size": int(TARGET_SAMPLE_SIZE),
        "n_repeats": int(N_REPEATS),
        "sampling": {
            "with_replacement": False,
            "block_size_cells": BLOCK_SIZE_CELLS,
            "nominal_scale": BLOCK_NOMINAL_SCALE,
            "block_column": BLOCK_COLUMN,
            "block_utility": "src.step8_large_block_robustness.assign_large_blocks",
            "stratum_definition": "large_block_id x label",
            "allocation_method": "hamilton_largest_remainder_integer_exact",
            "allocation_tie_break": "sort by (-remainder_numerator, stratum_id) ascending",
            "within_stratum_order": "cell_id ascending, then deterministic permutation",
            "seed_derivation": "blake2b(schema_version|repeat_id|stratum_id, 8) % 2**32",
            "allocation_is_repeat_invariant": True,
        },
        "folds": {
            "source": "inherited_from_frozen_full_mugla_artifact",
            "artifact_relative": str(
                WITHIN_REFERENCE_RELATIVE / WITHIN_REFERENCE_OOF_NAME),
            "artifact_schema": FOLD_ARTIFACT_SCHEMA,
            "fold_column": "fold_id",
            "fold_count": FOLD_COUNT,
            "splitter": "StratifiedGroupKFold",
            "shuffle": True,
            "random_state": FOLD_RANDOM_STATE,
            "strict_folds": True,
            "reoptimised_per_repeat": False,
            "canonical_small_block_size_cells_context_only":
                CANONICAL_SMALL_BLOCK_SIZE_CELLS,
        },
        "model": {
            "name": MODEL_NAME,
            "class": type(classifier).__name__,
            "hyperparameters": classifier.get_params(deep=False),
            "source": "src.step8b_train_baseline_vs_thermal_model.build_pipeline",
            "estimator_seed": ESTIMATOR_SEED,
            "families": list(MODEL_FAMILIES),
            "tuning_performed": False,
            "threshold_selection_performed": False,
            "sample_weight_argument_used": False,
            "oversampling_performed": False,
        },
        "feature_lists": {family: list(features)
                          for family, features in FEATURE_LISTS.items()},
        "forbidden_feature_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "preprocessing": {
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot_handle_unknown_ignore",
            "categorical_features": list(CATEGORICAL_FEATURES),
            "fit_scope": "training_frame_of_each_condition_only",
        },
        "metrics": {
            "primary": PRIMARY_METRIC,
            "secondary": [metric for metric in METRICS if metric != PRIMARY_METRIC],
            "orientation": {metric: metric_orientation(metric) for metric in METRICS},
            "helper": "src.step8b_train_baseline_vs_thermal_model.compute_binary_metrics",
        },
        "summarisation": {
            "statistics": ["median", "p2_5", "p97_5", "minimum", "maximum"],
            "percentile_method": PERCENTILE_METHOD,
            "interval_names": ["subsampling_interval_lower", "subsampling_interval_upper"],
            "position_tokens": list(POSITION_TOKENS),
            "bootstrap_performed": False,
            "inferential_statistics_produced": False,
        },
        "arms": list(ARMS),
        "directions": {
            ARM_WITHIN: [WITHIN_DIRECTION],
            ARM_SOURCE: source_directions(),
            ARM_TARGET: target_directions(),
        },
        "expected_unique_fits": expected_unique_fit_count(),
        "thermal_features_context": list(THERMAL_FEATURES),
    }


# =============================================================================
# Stage markers and resume
# =============================================================================
def write_stage_marker(analysis_id: str, stage: str, output_root: Optional[Path] = None,
                       extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    root = analysis_root(analysis_id, output_root)
    files: dict[str, str] = {}
    for relative in STAGE_OUTPUTS[stage]:
        path = root / relative
        if not path.exists():
            raise MuglaSubsamplingError(
                f"Stage {stage!r} claims completion but did not produce {relative}."
            )
        files[relative] = sha256_path(path)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "stage": stage,
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "requires": list(STAGE_REQUIRES[stage]),
        "files": files,
        **(extra or {}),
    }
    _atomic_write_text(stage_marker_path(analysis_id, stage, output_root),
                       _json_document(marker))
    return marker


def read_stage_marker(analysis_id: str, stage: str, output_root: Optional[Path] = None
                      ) -> Optional[dict[str, Any]]:
    path = stage_marker_path(analysis_id, stage, output_root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_stage_complete(analysis_id: str, stage: str, output_root: Optional[Path] = None
                          ) -> dict[str, Any]:
    """A stage is reusable only when its marker says PASS, the identity and
    schema match, and every recorded file is still present with its hash.

    A partially written stage therefore never resumes as complete.
    """
    marker = read_stage_marker(analysis_id, stage, output_root)
    if marker is None:
        return {"complete": False, "reason": "no stage marker", "stage": stage}
    if marker.get("status") != "pass":
        return {"complete": False, "reason": f"status={marker.get('status')!r}", "stage": stage}
    if marker.get("analysis_id") != analysis_id:
        return {"complete": False, "reason": "analysis_id mismatch", "stage": stage}
    if marker.get("schema_version") != SCHEMA_VERSION:
        return {"complete": False, "reason": "schema_version mismatch", "stage": stage}

    root = analysis_root(analysis_id, output_root)
    recorded = marker.get("files") or {}
    missing = [relative for relative in STAGE_OUTPUTS[stage] if relative not in recorded]
    if missing:
        return {"complete": False, "reason": f"marker omits {missing}", "stage": stage}
    for relative, expected in recorded.items():
        path = root / relative
        if not path.exists():
            return {"complete": False, "reason": f"missing artifact {relative}", "stage": stage}
        if sha256_path(path) != expected:
            return {"complete": False, "reason": f"hash drift in {relative}", "stage": stage}
    return {"complete": True, "stage": stage, "marker": marker}


def verify_arm_partition(analysis_id: str, arm: str,
                         output_root: Optional[Path] = None) -> bool:
    """Is this arm's partition present AND hash-bound by a passing fit marker?

    A partition that exists but is not recorded counts as partial and is never
    accepted, so a half-written arm can never be silently reused.
    """
    marker = read_stage_marker(analysis_id, "fit", output_root)
    if marker is None or marker.get("status") != "pass":
        return False
    recorded = (marker.get("arm_partitions") or {}).get(arm)
    if not recorded:
        return False
    path = (analysis_root(analysis_id, output_root) / OOF_PREDICTIONS_DIRNAME
            / f"part-{arm}.parquet")
    if not path.is_file():
        return False
    return sha256_file(path) == recorded


def quarantine_namespace(analysis_id: str, output_root: Optional[Path] = None
                         ) -> Optional[str]:
    """--force moves an existing namespace aside. It never deletes."""
    root = analysis_root(analysis_id, output_root)
    if not root.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = diagnostics_root(output_root) / QUARANTINE_DIRNAME / analysis_id / stamp
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root), str(destination))
    return str(destination)


# =============================================================================
# Context
# =============================================================================
def build_context(experiment_ids: Sequence[str], input_inventory: dict[str, Any],
                  experiments_root: Optional[Path] = None,
                  output_root: Optional[Path] = None,
                  frames: Optional[dict[str, pd.DataFrame]] = None) -> dict[str, Any]:
    """Populations, strata, allocation, fold mapping and references.

    Everything here is read-only with respect to production paths.
    """
    frames = frames or {}
    populations = {
        experiment_id: load_primary_population(
            experiment_id, experiments_root, frames.get(experiment_id))
        for experiment_id in experiment_ids
    }
    mugla = populations[SUBSAMPLED_EXPERIMENT]

    target_total = int(TARGET_SAMPLE_SIZE)
    size_reference_rows = int(len(populations[SIZE_REFERENCE_EXPERIMENT]))
    production = is_production_mugla_frame(input_inventory)
    if production and size_reference_rows != target_total:
        raise MuglaSubsamplingError(
            f"{SIZE_REFERENCE_EXPERIMENT} primary population is {size_reference_rows}, "
            f"but the frozen target sample size is {target_total}. The size-matching "
            "contract is broken; refusing to silently re-target."
        )

    capacity_table = stratum_capacity_table(mugla)
    allocation_table = hamilton_allocation(capacity_table, target_total)
    assert_allocation_valid(allocation_table, target_total)
    prevalence = prevalence_accounting(capacity_table, allocation_table, target_total)

    fold_mapping, fold_provenance = load_frozen_fold_mapping(mugla, experiments_root)
    # `merge` does not carry `.attrs`; the allocation accounting must survive it.
    allocation_attrs = dict(allocation_table.attrs)
    allocation_table = allocation_table.merge(
        fold_mapping.groupby(BLOCK_COLUMN, as_index=False)["fold_id"].first(),
        on=BLOCK_COLUMN, how="left", validate="many_to_one",
    )
    allocation_table.attrs.update(allocation_attrs)

    production_inventory = None
    if production:
        production_inventory = assert_production_inventory(
            capacity_table, allocation_table, prevalence)

    selected = build_selected_cells(
        mugla, allocation_table, fold_mapping,
        input_inventory[SUBSAMPLED_EXPERIMENT]["sha256"],
        int(N_REPEATS), target_total,
    )
    fold_summary = assert_selection_fold_contract(selected, int(N_REPEATS))
    if production:
        expected_rows = list(PRODUCTION_INVENTORY["fold_rows"])
        expected_positives = list(PRODUCTION_INVENTORY["fold_positives"])
        if (fold_summary["per_fold_rows"] != expected_rows
                or fold_summary["per_fold_positives"] != expected_positives):
            raise MuglaSubsamplingError(
                "Per-fold subsample composition does not reproduce the frozen design: "
                f"rows {fold_summary['per_fold_rows']} (expected {expected_rows}), "
                f"positives {fold_summary['per_fold_positives']} "
                f"(expected {expected_positives})."
            )

    canonical_cells = set(mugla["cell_id"].astype(str))
    foreign = set(selected["cell_id"].astype(str)) - canonical_cells
    if foreign:
        raise MuglaSubsamplingError(
            f"{len(foreign)} selected cell(s) are not in the canonical Mugla primary "
            "population."
        )

    references = build_reference_inventory(experiments_root, output_root)
    reference_rows = build_reference_metrics(references)

    if production:
        for experiment_id, expected_rows_count in PRODUCTION_TARGET_POPULATIONS.items():
            observed = int(len(populations[experiment_id]))
            if observed != expected_rows_count:
                raise MuglaSubsamplingError(
                    f"{experiment_id} primary population is {observed}, expected "
                    f"{expected_rows_count}; the frozen target cohort has changed."
                )

    return {
        "populations": populations,
        "mugla": mugla,
        "capacity_table": capacity_table,
        "allocation_table": allocation_table,
        "prevalence": prevalence,
        "fold_mapping": fold_mapping,
        "fold_provenance": fold_provenance,
        "fold_summary": fold_summary,
        "selected_cells": selected,
        "references": references,
        "reference_rows": reference_rows,
        "target_total": target_total,
        "production_frame": production,
        "production_inventory": production_inventory,
    }


SAMPLING_INVENTORY_COLUMNS: list[str] = [
    "experiment_id", "role", "subsampled", "rows_primary", "positives", "negatives",
    "prevalence", "n_blocks", "n_strata", "target_sample_size", "sampled_positives",
    "sampled_negatives", "sampled_prevalence", "prevalence_absolute_drift",
    "prevalence_drift_bound", "sampling_fraction", "step8a_sha256",
]


def build_sampling_inventory(context: dict[str, Any],
                             input_inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prevalence = context["prevalence"]
    for experiment_id, population in context["populations"].items():
        subsampled = experiment_id == SUBSAMPLED_EXPERIMENT
        positives = int((population["label"] == 1).sum())
        rows.append({
            "experiment_id": experiment_id,
            "role": (
                "subsampled_source_and_target" if subsampled
                else "full_source_and_target_cohort"
            ),
            "subsampled": subsampled,
            "rows_primary": int(len(population)),
            "positives": positives,
            "negatives": int(len(population)) - positives,
            "prevalence": positives / len(population),
            "n_blocks": int(population[BLOCK_COLUMN].nunique()) if subsampled else None,
            "n_strata": int(len(context["capacity_table"])) if subsampled else None,
            "target_sample_size": context["target_total"] if subsampled else None,
            "sampled_positives": prevalence["sampled_positives"] if subsampled else None,
            "sampled_negatives": prevalence["sampled_negatives"] if subsampled else None,
            "sampled_prevalence": prevalence["prevalence_subsample"] if subsampled else None,
            "prevalence_absolute_drift": (
                prevalence["prevalence_absolute_drift"] if subsampled else None),
            "prevalence_drift_bound": (
                prevalence["prevalence_drift_bound"] if subsampled else None),
            "sampling_fraction": (
                context["target_total"] / len(population) if subsampled else None),
            "step8a_sha256": input_inventory[experiment_id]["sha256"],
        })
    return rows


STRATUM_ALLOCATION_COLUMNS: list[str] = [
    "stratum_id", "large_block_id", "label", "capacity", "quota_numerator",
    "floor_allocation", "remainder_numerator", "remainder_rank",
    "received_remainder_unit", "allocation_count", "capacity_headroom", "fold_id",
]


# =============================================================================
# PLAN stage
# =============================================================================
def run_plan_stage(analysis_id: str, experiment_ids: Sequence[str],
                   context: dict[str, Any], input_inventory: dict[str, Any],
                   scientific_config: dict[str, Any],
                   output_root: Optional[Path] = None) -> dict[str, Any]:
    """Freeze the selection. No model is fitted in this stage."""
    root = analysis_root(analysis_id, output_root)
    root.mkdir(parents=True, exist_ok=True)

    _write_namespaced(root, "config.json", _json_document({
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "scientific_config": scientific_config,
        "planned_outputs": planned_output_layout(),
        "production_frame": context["production_frame"],
    }))
    _write_namespaced(root, "input_hashes.json", _json_document({
        "step8a": input_inventory,
        "hash_gate": "strict",
        "within_reference": {
            "metrics_path": context["references"]["within"]["metrics_path"],
            "metrics_sha256": context["references"]["within"]["metrics_sha256"],
            "predictions_path": context["references"]["within"]["predictions_path"],
            "predictions_sha256": context["references"]["within"]["predictions_sha256"],
            "fold_assignments_sha256": context["references"]["within"]["folds_sha256"],
            "block_manifest_sha256": context["references"]["within"]["block_manifest_sha256"],
        },
        "fold_artifact": {
            "path": context["fold_provenance"]["artifact_path"],
            "sha256": context["fold_provenance"]["artifact_sha256"],
            "schema": context["fold_provenance"]["artifact_schema"],
            "fold_source": context["fold_provenance"]["fold_source"],
        },
        "transfer_references": {
            direction: {
                "metrics_path": entry["metrics_path"],
                "metrics_sha256": entry["metrics_sha256"],
                "predictions_path": entry["predictions_path"],
                "predictions_sha256": entry["predictions_sha256"],
            }
            for group in ("source", "target")
            for direction, entry in context["references"][group].items()
        },
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
    }))
    _write_namespaced(root, "sampling_inventory.csv", _csv_document(
        SAMPLING_INVENTORY_COLUMNS,
        build_sampling_inventory(context, input_inventory)))
    _write_namespaced(root, "stratum_allocation.csv", _csv_document(
        STRATUM_ALLOCATION_COLUMNS,
        context["allocation_table"].to_dict(orient="records")))
    _write_namespaced_parquet(root, "selected_cells.parquet", context["selected_cells"])
    _write_namespaced_parquet(root, "fold_mapping.parquet", context["fold_mapping"])
    _write_namespaced(root, "reference_metrics.csv", _csv_document(
        REFERENCE_COLUMNS, context["reference_rows"]))

    marker = write_stage_marker(analysis_id, "plan", output_root, extra={
        "n_repeats": int(N_REPEATS),
        "target_sample_size": context["target_total"],
        "n_strata": int(len(context["capacity_table"])),
        "n_blocks": int(context["mugla"][BLOCK_COLUMN].nunique()),
        "allocation_accounting": {
            key: context["allocation_table"].attrs.get(key)
            for key in ("population_total", "target_total", "floor_total",
                        "remainder_units", "cut_remainder_numerator",
                        "strata_above_cut", "strata_tied_at_cut", "tie_units_awarded")
        },
        "prevalence_accounting": context["prevalence"],
        "fold_accounting": {**context["fold_provenance"], **context["fold_summary"]},
        "fit_performed": False,
    })
    return {
        "stage": "plan",
        "selected_rows": int(len(context["selected_cells"])),
        "n_strata": int(len(context["capacity_table"])),
        "marker": marker,
    }


# =============================================================================
# FIT stage
# =============================================================================
def run_fit_stage(analysis_id: str, context: dict[str, Any],
                  output_root: Optional[Path] = None,
                  resume: bool = False) -> dict[str, Any]:
    root = analysis_root(analysis_id, output_root)
    selected = pd.read_parquet(root / "selected_cells.parquet")
    reference_rows = context["reference_rows"]
    references = reference_lookup(reference_rows)

    mugla = context["mugla"].copy()
    mugla_by_cell = mugla.set_index("cell_id", drop=False)
    target_frames = {
        target_id: context["populations"][target_id]
        for _, target_id in SOURCE_PAIRS
    }

    registry = FitRegistry()
    within_frames: list[pd.DataFrame] = []
    source_frames: list[pd.DataFrame] = []
    target_frames_out: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for repeat_id in sorted(int(value) for value in selected["repeat_id"].unique()):
        repeat_cells = selected[selected["repeat_id"] == repeat_id]
        repeat_frame = mugla_by_cell.loc[repeat_cells["cell_id"].to_numpy()].copy()
        repeat_frame["fold_id"] = repeat_cells["fold_id"].to_numpy()
        repeat_frame["repeat_id"] = repeat_id
        repeat_frame = repeat_frame.reset_index(drop=True)
        repeat_sample_hash = sample_hash(repeat_frame["cell_id"].tolist())

        within = run_within_arm(repeat_frame, repeat_id, registry)
        within_frames.append(within)
        metric_rows += _repeat_metric_rows(
            ARM_WITHIN, WITHIN_DIRECTION, repeat_id,
            within[TARGET_COLUMN].to_numpy(),
            {family: within[f"{family}_probability"].to_numpy()
             for family in MODEL_FAMILIES},
            references, repeat_sample_hash,
        )

        source = run_source_arm(repeat_frame, repeat_id, target_frames, registry)
        source_frames.append(source)
        for direction in source_directions():
            slice_ = source[source["direction"] == direction]
            metric_rows += _repeat_metric_rows(
                ARM_SOURCE, direction, repeat_id,
                slice_[TARGET_COLUMN].to_numpy(),
                {family: slice_[f"{family}_probability"].to_numpy()
                 for family in MODEL_FAMILIES},
                references, repeat_sample_hash,
            )

        target = run_target_arm(repeat_frame, repeat_id, context["references"])
        target_frames_out.append(target)
        for direction in target_directions():
            slice_ = target[target["direction"] == direction]
            metric_rows += _repeat_metric_rows(
                ARM_TARGET, direction, repeat_id,
                slice_[TARGET_COLUMN].to_numpy(),
                {family: slice_[f"{family}_probability"].to_numpy()
                 for family in MODEL_FAMILIES},
                references, repeat_sample_hash,
            )

    partitions = {
        ARM_WITHIN: pd.concat(within_frames, ignore_index=True),
        ARM_SOURCE: pd.concat(source_frames, ignore_index=True),
        ARM_TARGET: pd.concat(target_frames_out, ignore_index=True),
    }
    arm_partitions: dict[str, str] = {}
    for arm, frame in partitions.items():
        relative = f"{OOF_PREDICTIONS_DIRNAME}/part-{arm}.parquet"
        path = _write_namespaced_parquet(root, relative, frame)
        arm_partitions[arm] = sha256_file(path)

    _write_namespaced(root, "repeat_metrics.csv",
                      _csv_document(REPEAT_METRIC_COLUMNS, metric_rows))

    accounting = registry.accounting()
    expected = expected_unique_fit_count()
    observed = {
        "within_fits": accounting["within_fits"],
        "source_fits": accounting["source_fits"],
        "target_fits": accounting["target_fits"],
        "unique_fits": accounting["unique_fits"],
        "reuse_events": accounting["reuse_events"],
    }
    if observed != expected:
        raise MuglaSubsamplingError(
            f"Fit accounting mismatch: expected {expected}, observed {observed}. The "
            "fit-sharing contract of the design was not honoured."
        )

    marker = write_stage_marker(analysis_id, "fit", output_root, extra={
        "arm_partitions": arm_partitions,
        "fit_accounting": {
            **accounting,
            "expected": expected,
            "contract_upper_bound": (
                int(N_REPEATS) * len(MODEL_FAMILIES) * FOLD_COUNT
                + int(N_REPEATS) * len(SOURCE_PAIRS) * len(MODEL_FAMILIES)
                + len(TARGET_PAIRS) * len(MODEL_FAMILIES)
            ),
            "reductions": [
                "source arm: fit identity excludes target_id (a source-only model is "
                "target-independent), so one fit serves both directions",
                "target arm: frozen per-cell raw-transfer predictions are reused, so "
                "no model is fitted at all",
            ],
        },
        "repeat_metric_rows": len(metric_rows),
        "resume_requested": bool(resume),
    })
    return {
        "stage": "fit",
        "repeat_metric_rows": len(metric_rows),
        "fit_accounting": accounting,
        "marker": marker,
    }


# =============================================================================
# SUMMARIZE stage
# =============================================================================
def build_manifest(analysis_id: str, output_root: Optional[Path] = None) -> dict[str, Any]:
    root = analysis_root(analysis_id, output_root)
    predictions_dir = root / OOF_PREDICTIONS_DIRNAME
    # The summarize stage marker is written AFTER the manifest, because the
    # marker hash-binds manifest.json. It therefore cannot appear in the
    # manifest it certifies; the dependency is declared rather than hidden.
    deferred = [str(Path("stages") / "summarize.json")]
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "manifest.json":
            continue
        if str(path.relative_to(root)) in deferred:
            continue
        if predictions_dir in path.parents:
            # Exposed as ONE logical dataset below, never as loose parts.
            continue
        files.append({
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    logical = {
        OOF_PREDICTIONS_DIRNAME: {
            "kind": "partitioned_parquet_dataset",
            "partition_scheme": "one part per arm",
            "parts": sorted(p.name for p in predictions_dir.glob("*.parquet"))
            if predictions_dir.is_dir() else [],
            "dataset_sha256": (
                sha256_path(predictions_dir) if predictions_dir.is_dir() else None),
            "size_bytes": sum(p.stat().st_size for p in predictions_dir.glob("*.parquet"))
            if predictions_dir.is_dir() else 0,
            "read_as": f"pandas.read_parquet(<analysis_root>/{OOF_PREDICTIONS_DIRNAME})",
        }
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "namespace": str(root),
        "files": files,
        "deferred_files": {
            "paths": deferred,
            "reason": (
                "written after this manifest because the summarize stage marker "
                "hash-binds manifest.json; hashed by the marker instead"
            ),
        },
        "logical_datasets": logical,
        "stages": {
            stage: read_stage_marker(analysis_id, stage, output_root) is not None
            for stage in STAGES
        },
        "earth_engine_used": False,
        "bootstrap_performed": False,
    }


def render_report(analysis_id: str, summary: dict[str, Any],
                  summary_rows: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = [
        f"# Mugla subsampling sensitivity — `{SCHEMA_VERSION}`",
        "",
        f"Analysis ID: `{analysis_id}`",
        f"Diagnostic class: `{DIAGNOSTIC_CLASS}`",
        "",
        "## Question",
        "",
        "When the Mugla modeling population is reduced to exactly Manavgat's cell "
        "count, how do within-region performance, Mugla-as-source transfer and "
        "Mugla-as-target evaluation move relative to their full-population "
        "references?",
        "",
        "This is a total-sample-size sensitivity analysis. Prevalence is preserved "
        "within integer rounding limits; the positive count is NOT equalised to "
        "Manavgat's; no predictor or label structure is altered.",
        "",
        "## Sampling contract",
        "",
        f"- Arm: `{ARM_ID}`",
        f"- Target sample size: {summary['target_sample_size']} rows, "
        f"{summary['n_repeats']} deterministic repeats, without replacement",
        f"- Strata: {summary['allocation_accounting']['n_strata']} "
        f"(large_block_id x label) over "
        f"{summary['allocation_accounting']['n_blocks']} blocks at "
        f"{BLOCK_SIZE_CELLS} cells ({BLOCK_NOMINAL_SCALE})",
        f"- Allocation: integer-exact Hamilton largest remainder; floor total "
        f"{summary['allocation_accounting']['floor_total']}, "
        f"{summary['allocation_accounting']['remainder_units']} remainder units, "
        f"{summary['allocation_accounting']['tie_units_awarded']} awarded by the "
        f"stratum_id tie-break among "
        f"{summary['allocation_accounting']['strata_tied_at_cut']} tied strata",
        f"- Prevalence: full "
        f"{summary['prevalence_accounting']['prevalence_full']:.8f} vs subsample "
        f"{summary['prevalence_accounting']['prevalence_subsample']:.8f} "
        f"(drift {summary['prevalence_accounting']['prevalence_absolute_drift']:.8f}, "
        f"bound {summary['prevalence_accounting']['prevalence_drift_bound']:.8f})",
        "",
        "## Fold contract",
        "",
        f"- The full-Mugla block-to-fold mapping is inherited unchanged from "
        f"`{summary['fold_accounting']['fold_source']}` and is never re-optimised "
        f"per repeat.",
        f"- {summary['fold_accounting']['fold_count']} folds, "
        f"{summary['fold_accounting']['blocks_spanning_folds']} blocks spanning "
        f"more than one fold.",
        f"- Per-fold subsample rows: {summary['fold_accounting']['per_fold_rows']}; "
        f"positives: {summary['fold_accounting']['per_fold_positives']} — identical "
        f"in every repeat.",
        "",
        "## Fit accounting",
        "",
        f"- Within-Mugla fits: {summary['fit_accounting']['within_fits']}",
        f"- Mugla-as-source fits: {summary['fit_accounting']['source_fits']} "
        f"(each serving both targets; "
        f"{summary['fit_accounting']['reuse_events']} reuse events)",
        f"- Mugla-as-target fits: {summary['fit_accounting']['target_fits']} "
        f"(frozen raw-transfer predictions reused)",
        f"- Total unique fits: {summary['fit_accounting']['unique_fits']}",
        "",
        "## Results",
        "",
        "`oriented_delta` is positive when the subsample result is better, for every "
        "metric. The interval is a subsampling range over repeats, not an "
        "uncertainty estimate.",
        "",
        "| Arm | Direction | Family | Metric | Full reference | Subsample median | "
        "Interval lower | Interval upper | Oriented delta median | Reference position |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['arm']} | {row['direction']} | {row['model_family']} | "
            f"{row['metric']} | {row['full_reference_value']:.6f} | "
            f"{row['subsample_median']:.6f} | "
            f"{row['subsampling_interval_lower']:.6f} | "
            f"{row['subsampling_interval_upper']:.6f} | "
            f"{row['oriented_delta_median']:+.6f} | {row['reference_position']} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "The two permitted readings are:",
        "",
        f"- Reference inside the range: {SENTENCE_INSIDE}",
        f"- Reference outside the range: {SENTENCE_OUTSIDE}",
        "",
        "The final scientific reading must be made from all three arms jointly.",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"{index}. {text}" for index, text in enumerate(LIMITATIONS, start=1)]
    lines += [
        "",
        "No bootstrap was run and no probability statement of any kind is made. "
        "Contacts no Earth Engine; writes only inside its own namespace.",
        "",
    ]
    return "\n".join(lines)


def build_summary(analysis_id: str, experiment_ids: Sequence[str],
                  summary_rows: Sequence[dict[str, Any]],
                  repeat_metrics: pd.DataFrame, context: dict[str, Any],
                  fit_accounting: dict[str, Any]) -> dict[str, Any]:
    allocation = context["allocation_table"]
    headline = [
        row for row in summary_rows
        if row["metric"] == PRIMARY_METRIC
    ] + [row for row in summary_rows if row["metric"] != PRIMARY_METRIC]
    positions = {
        f"{row['arm']}|{row['direction']}|{row['model_family']}|{row['metric']}":
            row["reference_position"]
        for row in summary_rows
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "experiments": list(experiment_ids),
        "arm_id": ARM_ID,
        "target_sample_size": context["target_total"],
        "n_repeats": int(N_REPEATS),
        "population_accounting": {
            experiment_id: {
                "rows_primary": int(len(population)),
                "positives": int((population["label"] == 1).sum()),
                "negatives": int((population["label"] == 0).sum()),
            }
            for experiment_id, population in context["populations"].items()
        },
        "allocation_accounting": {
            "n_blocks": int(context["mugla"][BLOCK_COLUMN].nunique()),
            "n_strata": int(len(context["capacity_table"])),
            "n_positive_strata": context["prevalence"]["n_positive_strata"],
            "floor_total": allocation.attrs.get("floor_total"),
            "remainder_units": allocation.attrs.get("remainder_units"),
            "cut_remainder_numerator": allocation.attrs.get("cut_remainder_numerator"),
            "strata_above_cut": allocation.attrs.get("strata_above_cut"),
            "strata_tied_at_cut": allocation.attrs.get("strata_tied_at_cut"),
            "tie_units_awarded": allocation.attrs.get("tie_units_awarded"),
            "allocation_sum": int(allocation["allocation_count"].sum()),
            "strata_over_capacity": int(
                (allocation["allocation_count"] > allocation["capacity"]).sum()),
            "strata_zero_allocation": int((allocation["allocation_count"] < 1).sum()),
            "allocation_is_repeat_invariant": True,
        },
        "prevalence_accounting": context["prevalence"],
        "fold_accounting": {**context["fold_provenance"], **context["fold_summary"]},
        "fit_accounting": fit_accounting,
        "headline": headline,
        "reference_positions": positions,
        "three_arm_reading": {
            "within_region_moves": (
                "See the within_mugla rows: the oriented_delta interval and the "
                "reference_position token together describe whether within-region "
                "performance moves when the population is size-matched."
            ),
            "source_transfer_moves": (
                "See the mugla_as_source rows: target cohorts are full and unchanged, "
                "so any movement is attributable to the source training set size "
                "under this selection design."
            ),
            "target_ordering_preserved": (
                "See the mugla_as_target rows: source models are full and frozen, so "
                "these rows describe target cohort size and composition sensitivity "
                "only."
            ),
            "joint_reading_required": True,
        },
        "limitations": list(LIMITATIONS),
        "repeat_metric_rows": int(len(repeat_metrics)),
        "summary_rows": len(summary_rows),
        "earth_engine_used": False,
        "bootstrap_performed": False,
        "inferential_statistics_produced": False,
        "canonical_outputs_written": False,
    }


def run_summarize_stage(analysis_id: str, experiment_ids: Sequence[str],
                        context: dict[str, Any],
                        output_root: Optional[Path] = None) -> dict[str, Any]:
    root = analysis_root(analysis_id, output_root)
    metrics_path = root / "repeat_metrics.csv"
    if not metrics_path.is_file():
        raise MuglaSubsamplingError(
            f"Stage 'summarize' requires {metrics_path} from stage 'fit'."
        )
    repeat_metrics = pd.read_csv(metrics_path)
    summary_rows = build_subsampling_summary(repeat_metrics)
    _write_namespaced(root, "subsampling_summary.csv",
                      _csv_document(SUMMARY_COLUMNS, summary_rows))

    fit_marker = read_stage_marker(analysis_id, "fit", output_root) or {}
    fit_accounting = fit_marker.get("fit_accounting", {})
    summary = build_summary(analysis_id, experiment_ids, summary_rows,
                            repeat_metrics, context, fit_accounting)
    _write_namespaced(root, "summary.json", _json_document(summary))
    _write_namespaced(root, "report.md",
                      render_report(analysis_id, summary, summary_rows))

    hits = scan_forbidden_tokens(root)
    if hits:
        raise MuglaSubsamplingError(
            f"Forbidden significance vocabulary found in the emitted outputs: "
            f"{hits[:5]} ({len(hits)} hit(s)). This analysis reports no probability "
            "statement of any kind."
        )

    manifest = build_manifest(analysis_id, output_root)
    _write_namespaced(root, "manifest.json", _json_document(manifest))
    marker = write_stage_marker(analysis_id, "summarize", output_root, extra={
        "summary_rows": len(summary_rows),
        "forbidden_token_hits": 0,
    })
    return {
        "stage": "summarize",
        "summary_rows": len(summary_rows),
        "manifest_files": len(manifest["files"]),
        "marker": marker,
    }


# =============================================================================
# Orchestration
# =============================================================================
def run_analysis(experiments: Optional[Sequence[str]] = None, from_stage: str = "plan",
                 to_stage: str = "summarize", dry_run: bool = False,
                 resume: bool = False, force: bool = False,
                 output_root: Optional[Path] = None,
                 experiments_root: Optional[Path] = None,
                 frames: Optional[dict[str, pd.DataFrame]] = None) -> dict[str, Any]:
    """Run the requested stage range.

    `dry_run` is strictly read-only: it resolves inputs, verifies hashes and
    reports the plan, but creates no directory, writes no file and fits no
    model. `force` quarantines an existing namespace (never deletes it).
    `resume` reuses only complete, hash-bound stages.
    """
    stages = validate_stage_range(from_stage, to_stage)
    experiment_ids = resolve_experiments(experiments)
    if SUBSAMPLED_EXPERIMENT not in experiment_ids:
        raise MuglaSubsamplingError(
            f"{SUBSAMPLED_EXPERIMENT!r} is the subsampled experiment and must be "
            f"present; got {experiment_ids}."
        )

    input_inventory = build_frozen_input_inventory(experiment_ids, experiments_root)
    hash_gate = assert_canonical_step8a_hashes(input_inventory, strict=True)
    scientific_config = build_scientific_config(experiment_ids)
    analysis_id = compute_analysis_id(scientific_config)
    root = analysis_root(analysis_id, output_root)

    if dry_run:
        return {
            "ran": False,
            "dry_run": True,
            "schema_version": SCHEMA_VERSION,
            "analysis_id": analysis_id,
            "diagnostic_class": DIAGNOSTIC_CLASS,
            "experiments": experiment_ids,
            "subsampled_experiment": SUBSAMPLED_EXPERIMENT,
            "arm_id": ARM_ID,
            "target_sample_size": int(TARGET_SAMPLE_SIZE),
            "n_repeats": int(N_REPEATS),
            "population": POPULATION,
            "arms": list(ARMS),
            "directions": {
                ARM_WITHIN: [WITHIN_DIRECTION],
                ARM_SOURCE: source_directions(),
                ARM_TARGET: target_directions(),
            },
            "stages_requested": stages,
            "stages_executed": [],
            "expected_fit_accounting": expected_unique_fit_count(),
            "input_hashes": input_inventory,
            "hash_gate": hash_gate,
            "output_namespace": str(root),
            "planned_outputs": planned_output_layout(),
            "namespace_exists": root.exists(),
            "files_written": [],
            "fit_performed": False,
            "earth_engine_used": False,
        }

    quarantined: Optional[str] = None
    if force:
        quarantined = quarantine_namespace(analysis_id, output_root)

    if root.exists() and not (resume or force):
        raise MuglaSubsamplingError(
            f"Analysis namespace already exists: {root}. Pass --resume to verify and "
            "reuse it, or --force to quarantine it (nothing is ever deleted)."
        )

    for stage in stages:
        for prerequisite in STAGE_REQUIRES[stage]:
            if prerequisite in stages:
                continue
            state = verify_stage_complete(analysis_id, prerequisite, output_root)
            if not state["complete"]:
                raise MuglaSubsamplingError(
                    f"Stage {stage!r} requires a complete {prerequisite!r} stage: "
                    f"{state['reason']}."
                )

    context = build_context(experiment_ids, input_inventory, experiments_root,
                            output_root, frames)

    executed: list[dict[str, Any]] = []
    for stage in stages:
        if resume:
            state = verify_stage_complete(analysis_id, stage, output_root)
            if state["complete"]:
                executed.append({"stage": stage, "reused": True})
                continue
        if stage == "plan":
            executed.append(run_plan_stage(analysis_id, experiment_ids, context,
                                           input_inventory, scientific_config,
                                           output_root))
        elif stage == "fit":
            executed.append(run_fit_stage(analysis_id, context, output_root,
                                          resume=resume))
        elif stage == "summarize":
            executed.append(run_summarize_stage(analysis_id, experiment_ids, context,
                                                output_root))

    return {
        "ran": True,
        "dry_run": False,
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "experiments": experiment_ids,
        "arm_id": ARM_ID,
        "target_sample_size": context["target_total"],
        "n_repeats": int(N_REPEATS),
        "stages_executed": [entry["stage"] for entry in executed],
        "stage_results": executed,
        "output_namespace": str(root),
        "quarantined_previous_namespace": quarantined,
        "earth_engine_used": False,
    }
