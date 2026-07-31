"""Few-shot recovery curve — `few_shot_recovery.v1`.

Implements the frozen design in `docs/few_shot_recovery_design/`. That design
is binding: this module makes no scientific decision of its own.

The question: when a limited number of labeled spatial blocks from the target
region is supplied, how much of the gap between zero-shot raw transfer and the
target-only within-region ceiling is recovered?

DIAGNOSTIC CLASS: `target_label_supervised_few_shot_adaptation_sensitivity`.
Target labels ARE used, deliberately, for adaptation and evaluation. This is
NOT an operational deployment claim, NOT active learning, NOT a causal
decomposition and NOT target-label-free adaptation.

Everything scientific is reused unchanged from the canonical Step8 contract:
`build_pipeline`, `check_no_forbidden_features`, `make_spatial_folds`,
`compute_binary_metrics`, `assign_large_blocks`, the feature constants and the
seeds. No new model family, no hyperparameter tuning, no `sample_weight`, no
oversampling. The estimator's pre-existing `class_weight="balanced"` is
documented, never modified.

Writes exclusively under
`outputs/diagnostics/few_shot_recovery/<analysis_id>/`.
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
from typing import Any, Callable, Iterable, Optional, Sequence

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
    make_spatial_folds,
)
from src.step9a_audit_cross_region_inputs import (
    PRIMARY_POPULATIONS,
    resolve_step8a_dataset_path,
    resolve_step8a_stats_path,
)
from src.step9b_run_cross_region_transfer import population_subset


class FewShotRecoveryError(SystemExit):
    """Fail-fast error (same convention as every other step in this repo)."""


# =============================================================================
# Frozen contract  --  docs/few_shot_recovery_design/SCIENTIFIC_CONTRACT.md
# =============================================================================
SCHEMA_VERSION = "few_shot_recovery.v1"
DIAGNOSTIC_NAMESPACE = "few_shot_recovery"
DIAGNOSTIC_CLASS = "target_label_supervised_few_shot_adaptation_sensitivity"

PRIMARY_EXPERIMENTS: tuple[str, ...] = ("manavgat_2021", "bejis_2022", "mugla_2021")
EXPECTED_DIRECTED_PAIRS = 6

# Excluded by design. evia_2021_extended is a high-prevalence,
# different-regime sensitivity/control AOI -- not an equal-prevalence primary
# transfer validation AOI. Adding it needs a new preregistration.
EXCLUDED_EXPERIMENTS: dict[str, str] = {
    "evia_2021_extended": (
        "high_prevalence_different_regime_sensitivity_control_not_equal_"
        "prevalence_primary_transfer_validation_aoi"
    ),
    "evia_2021": "out_of_scope_for_this_frozen_analysis",
}
EXCLUDED_TOKENS: tuple[str, ...] = ("evia",)

POPULATION = PRIMARY_POPULATIONS[0]  # "burnable_tree_shrub_grass"
VALID_UNIVERSE = "valid_for_modeling == True"

BLOCK_SIZE_CELLS = 10
BLOCK_NOMINAL_SCALE = NOMINAL_SCALES[BLOCK_SIZE_CELLS]  # "approximately_5_km"
BLOCK_COLUMN = "large_block_id"
CANONICAL_SMALL_BLOCK_SIZE_CELLS = STEP8B_SPATIAL_BLOCK_SIZE_CELLS  # 2, departed from

N_OUTER_FOLDS = STEP8B_N_SPLITS  # 5
FOLD_RANDOM_STATE = STEP8B_RANDOM_SEED  # 42
ESTIMATOR_SEED = STEP8B_RANDOM_SEED  # 42
MODEL_NAME = "random_forest"

MODEL_FAMILIES: tuple[str, ...] = ("baseline", "thermal")
MODEL_ROLES = {"thermal": "primary", "baseline": "secondary"}
FEATURE_LISTS: dict[str, list[str]] = {
    "baseline": list(BASELINE_FEATURES),
    "thermal": list(THERMAL_MODEL_FEATURES),
}

BUDGETS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
NONZERO_BUDGETS: tuple[int, ...] = tuple(k for k in BUDGETS if k > 0)
N_REPEATS = 10
N_REPEATS_SINGLE_REALISATION = 1  # raw (k=0) and ceiling carry no selection randomness

TIER_BOTH = "both_classes"
TIER_POSITIVE = "positives_only"
TIER_NEGATIVE = "negatives_only"
TIER_ORDER: tuple[str, ...] = (TIER_BOTH, TIER_POSITIVE, TIER_NEGATIVE)

CONDITION_RAW = "raw"
CONDITION_FEWSHOT = "few_shot"
CONDITION_CEILING = "ceiling"
CONDITIONS: tuple[str, ...] = (CONDITION_RAW, CONDITION_FEWSHOT, CONDITION_CEILING)

BUDGET_CEILING_SENTINEL = -1  # ceiling rows carry no budget

METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "brier_score")
PRIMARY_METRIC = "roc_auc"
PRIMARY_FAMILY = "thermal"
LOWER_IS_BETTER: tuple[str, ...] = ("brier_score",)
ORIENTATION_HIGHER = "higher_is_better"
ORIENTATION_NEGATED = "lower_is_better_oriented_by_negation"

# Identical to transfer_decomposition.RATIO_DEGENERATE_THRESHOLD.
DEGENERATE_DENOMINATOR_THRESHOLD = 1e-6
SELECTION_PCT_LOW = 2.5
SELECTION_PCT_HIGH = 97.5
PERCENTILE_METHOD = "linear"

STATUS_INTERPRETABLE = "interpretable"
STATUS_DEGENERATE = "undefined_degenerate_denominator"
STATUS_CEILING_NOT_ABOVE_RAW = "ceiling_not_above_raw"

EVALUATION_LEVEL_OOF = "oof"
EVALUATION_LEVEL_FOLD = "fold"
FOLD_SENTINEL_OOF = -1

# Frozen canonical Step8A digests. Verified 2026-08-02.
CANONICAL_STEP8A_SHA256: dict[str, str] = {
    "manavgat_2021": "054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439",
    "bejis_2022": "3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393",
    "mugla_2021": "c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e",
}

# Frozen 10-cell within-region ceiling references. The `validate` stage checks
# the produced ceiling against these. mugla_2021 has no counterpart -- that
# robustness run covered only the manavgat_2021__bejis_2022 pair.
FROZEN_CEILING_REFERENCE: dict[str, dict[str, Any]] = {
    "manavgat_2021": {
        "metrics_path": (
            "outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/"
            "manavgat_2021/block_10_cells/step8b_large_block_metrics.json"
        ),
        "bootstrap_path": (
            "outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/"
            "manavgat_2021/block_10_cells/step8c_large_block_bootstrap_summary.json"
        ),
        "roc_auc": {"baseline": 0.7475502988238435, "thermal": 0.7974298472620660},
    },
    "bejis_2022": {
        "metrics_path": (
            "outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/"
            "bejis_2022/block_10_cells/step8b_large_block_metrics.json"
        ),
        "bootstrap_path": (
            "outputs/robustness/step8_large_block/manavgat_2021__bejis_2022/"
            "bejis_2022/block_10_cells/step8c_large_block_bootstrap_summary.json"
        ),
        "roc_auc": {"baseline": 0.7793700238725079, "thermal": 0.8244685786179753},
    },
    "mugla_2021": None,
}
CEILING_REPRODUCTION_TOLERANCE = 1e-9

# Wording firewall. A selection interval is not a confidence interval.
FORBIDDEN_UNCERTAINTY_TERMS: tuple[str, ...] = (
    "confidence interval", "95% ci", "ci_2_5", "ci_97_5",
    "significant", "significance", "p-value", "p_value", "pvalue",
    "istatistiksel olarak anlaml", "anlaml",
)

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
        "target_block_inventory.csv",
        "direction_budget_feasibility.csv",
        "selected_blocks.parquet",
    ),
    "fit": (
        OOF_PREDICTIONS_DIRNAME,
        "repeat_metrics.csv",
    ),
    "summarize": (
        "recovery_curve.csv",
        "summary.json",
        "report.md",
        "manifest.json",
    ),
}


# =============================================================================
# Contract helpers
# =============================================================================
def validate_stage_range(from_stage: str, to_stage: str) -> list[str]:
    """The ordered stage slice, or a hard failure."""
    if from_stage not in STAGES:
        raise FewShotRecoveryError(
            f"Unknown from_stage {from_stage!r}. Valid stages: {list(STAGES)}."
        )
    if to_stage not in STAGES:
        raise FewShotRecoveryError(
            f"Unknown to_stage {to_stage!r}. Valid stages: {list(STAGES)}."
        )
    start, end = STAGES.index(from_stage), STAGES.index(to_stage)
    if start > end:
        raise FewShotRecoveryError(
            f"Stage range is reversed: {from_stage!r} comes after {to_stage!r}. "
            f"Stage order: {list(STAGES)}."
        )
    return list(STAGES[start:end + 1])


def resolve_experiments(experiments: Optional[Sequence[str]] = None) -> list[str]:
    """The three primary AOIs, with the Evia exclusion enforced."""
    resolved = list(experiments) if experiments else list(PRIMARY_EXPERIMENTS)
    if len(set(resolved)) != len(resolved):
        raise FewShotRecoveryError(
            f"Duplicate experiment id in {resolved}; each AOI may appear once."
        )
    if len(resolved) < 2:
        raise FewShotRecoveryError(
            "At least two experiments are required to form a directed pair; "
            f"got {resolved}."
        )
    for experiment in resolved:
        assert_not_excluded(experiment)
    return resolved


def assert_not_excluded(experiment_id: str) -> None:
    lowered = str(experiment_id).lower()
    if experiment_id in EXCLUDED_EXPERIMENTS:
        raise FewShotRecoveryError(
            f"{experiment_id!r} is excluded from this frozen analysis: "
            f"{EXCLUDED_EXPERIMENTS[experiment_id]}. Including it requires a new "
            "preregistration and a new analysis_id."
        )
    for token in EXCLUDED_TOKENS:
        if token in lowered:
            raise FewShotRecoveryError(
                f"{experiment_id!r} matches the excluded token {token!r}. "
                "This analysis is frozen to the three equal-prevalence primary "
                f"AOIs {list(PRIMARY_EXPERIMENTS)}."
            )


def directed_pairs(experiment_ids: Sequence[str]) -> list[tuple[str, str]]:
    """Every ordered (source, target) pair. Never sorted, never self-paired."""
    pairs = [
        (source, target)
        for source in experiment_ids
        for target in experiment_ids
        if source != target
    ]
    expected = len(experiment_ids) * (len(experiment_ids) - 1)
    if len(pairs) != expected:
        raise FewShotRecoveryError(
            f"Expected {expected} directed pairs from {list(experiment_ids)}; built {len(pairs)}."
        )
    return pairs


def direction_token(source_id: str, target_id: str) -> str:
    if source_id == target_id:
        raise FewShotRecoveryError(f"Self-pair is forbidden: {source_id} -> {target_id}.")
    return f"{source_id}_to_{target_id}"


def selection_key(source_id: str, target_id: str, outer_fold: int, repeat_id: int,
                  budget: int) -> str:
    return f"{direction_token(source_id, target_id)}|{outer_fold}|{repeat_id}|{budget}"


def selection_seed(source_id: str, target_id: str, outer_fold: int, repeat_id: int) -> int:
    """Deterministic from direction + fold + repeat, and from nothing else.

    Deliberately independent of the budget (one ordering serves every budget,
    which is what makes nesting exact) and of the model family (so baseline and
    thermal see identical adaptation blocks and are paired by construction).
    """
    key = f"{SCHEMA_VERSION}|{source_id}|{target_id}|{outer_fold}|{repeat_id}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 32)


def fit_identity(condition: str, *, family: str, source_id: Optional[str] = None,
                 target_id: Optional[str] = None, outer_fold: Optional[int] = None,
                 budget: Optional[int] = None, repeat_id: Optional[int] = None) -> str:
    """The identity that decides whether two fits are the SAME fit.

    raw      -- (source, target, family): the source model never sees the
                target, so it is fold- and repeat-independent.
    ceiling  -- (target, fold, family): source-independent, so the two
                directions sharing a target share the fit.
    few_shot -- (source, target, fold, budget, repeat, family): never shared.
    """
    if condition == CONDITION_RAW:
        parts = (CONDITION_RAW, source_id, target_id, family)
    elif condition == CONDITION_CEILING:
        parts = (CONDITION_CEILING, target_id, outer_fold, family)
    elif condition == CONDITION_FEWSHOT:
        parts = (CONDITION_FEWSHOT, source_id, target_id, outer_fold, budget,
                 repeat_id, family)
    else:
        raise FewShotRecoveryError(f"Unknown condition {condition!r}.")
    return "|".join(str(part) for part in parts)


def expected_unique_fit_count(n_directions: int, n_targets: int) -> dict[str, int]:
    raw = n_directions * len(MODEL_FAMILIES)
    few_shot = (
        n_directions * len(MODEL_FAMILIES) * N_OUTER_FOLDS
        * len(NONZERO_BUDGETS) * N_REPEATS
    )
    ceiling = n_targets * N_OUTER_FOLDS * len(MODEL_FAMILIES)
    return {
        "raw_fits": raw,
        "few_shot_fits": few_shot,
        "ceiling_fits": ceiling,
        "unique_fits": raw + few_shot + ceiling,
    }


# =============================================================================
# Paths
# =============================================================================
def diagnostics_root(output_root: Optional[Path] = None) -> Path:
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    return root / "diagnostics" / DIAGNOSTIC_NAMESPACE


def analysis_root(analysis_id: str, output_root: Optional[Path] = None) -> Path:
    return diagnostics_root(output_root) / analysis_id


def stage_marker_path(analysis_id: str, stage: str, output_root: Optional[Path] = None) -> Path:
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
    return Path(experiments_root) / experiment_id / "step8a" / "step8a_stats.json"


def planned_output_layout() -> dict[str, str]:
    """Every file this analysis may write. Used by --dry-run."""
    return {
        "config.json": "Frozen scientific configuration; the object hashed into analysis_id.",
        "input_hashes.json": "Canonical Step8A digests and external ceiling references.",
        "target_block_inventory.csv": "Per-target and per-fold 10-cell block inventory.",
        "direction_budget_feasibility.csv": "Per direction x fold x budget feasibility and tier composition.",
        "selected_blocks.parquet": "Adaptation block provenance, frozen before any fit.",
        f"{OOF_PREDICTIONS_DIRNAME}/part-<direction>.parquet": "Full target OOF predictions, one logical dataset partitioned by direction.",
        "repeat_metrics.csv": "Per-repeat metric values at oof and fold level.",
        "recovery_curve.csv": "The primary result: signed unclipped recovery per direction x family x metric x budget.",
        "summary.json": "Headline curve, ceiling reproduction, limitations, fit accounting.",
        "report.md": "Human-readable report.",
        "stages/plan.json": "Stage marker with per-file hashes.",
        "stages/fit.json": "Stage marker with per-file hashes.",
        "stages/summarize.json": "Stage marker with per-file hashes.",
        "manifest.json": "Every produced file with size and sha256; the citable record.",
    }


# =============================================================================
# Serialisation + atomic writes
# =============================================================================
def _json_document(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                      default=_json_default) + "\n"


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


def canonical_json(payload: Any) -> str:
    """Stable serialisation used for the analysis identity."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_json_default)


def compute_analysis_id(scientific_config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(scientific_config).encode("utf-8")).hexdigest()


def _csv_document(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _csv_cell(row.get(column)) for column in columns})
    return buffer.getvalue()


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


def assert_inside_namespace(path: Path, root: Path) -> None:
    resolved, root_resolved = Path(path).resolve(), Path(root).resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise FewShotRecoveryError(
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
    raise FewShotRecoveryError(f"Cannot hash a path that does not exist: {path}.")


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
# Inputs
# =============================================================================
def build_frozen_input_inventory(
    experiment_ids: Sequence[str], experiments_root: Optional[Path] = None,
) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for experiment_id in experiment_ids:
        dataset_path = canonical_step8a_path(experiment_id, experiments_root)
        if not dataset_path.is_file():
            raise FewShotRecoveryError(
                f"Canonical Step8A dataset missing for {experiment_id!r}: {dataset_path}."
            )
        manifest_path = canonical_step8a_manifest_path(experiment_id, experiments_root)
        expected = CANONICAL_STEP8A_SHA256.get(experiment_id)
        observed = sha256_file(dataset_path)
        inventory[experiment_id] = {
            "path": str(dataset_path),
            "sha256": observed,
            "expected_sha256": expected,
            "match": (expected is None) or (observed == expected),
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
        raise FewShotRecoveryError(
            "Canonical Step8A hash mismatch -- the frozen inputs of this analysis "
            f"have changed. {detail}. Refusing to run."
        )
    if strict and unregistered:
        raise FewShotRecoveryError(
            f"No registered canonical hash for {unregistered}. This frozen analysis "
            f"only accepts {list(CANONICAL_STEP8A_SHA256)}."
        )
    return {"mismatches": mismatches, "unregistered": unregistered,
            "all_match": not mismatches and not unregistered}


def external_ceiling_reference_inventory(
    experiment_ids: Sequence[str], output_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Frozen 10-cell ceiling artifacts, where they exist.

    These are READ-ONLY cross-checks living outside this namespace. mugla_2021
    has none; that is a stated limitation, not an error.
    """
    base = Path(output_root).parent if output_root is not None else PROJECT_ROOT
    inventory: dict[str, Any] = {}
    for experiment_id in experiment_ids:
        reference = FROZEN_CEILING_REFERENCE.get(experiment_id)
        if reference is None:
            inventory[experiment_id] = {
                "available": False, "reason": "no_frozen_block_10_artifact",
            }
            continue
        entry: dict[str, Any] = {"available": True}
        for key in ("metrics_path", "bootstrap_path"):
            path = base / reference[key]
            entry[key] = str(path)
            entry[f"{key}_sha256"] = sha256_file(path) if path.is_file() else None
            entry[f"{key}_present"] = path.is_file()
        entry["expected_roc_auc"] = dict(reference["roc_auc"])
        if not entry.get("metrics_path_present"):
            entry["available"] = False
            entry["reason"] = "frozen_block_10_artifact_not_found_on_disk"
        inventory[experiment_id] = entry
    return inventory


def load_target_frame(experiment_id: str, experiments_root: Optional[Path] = None,
                      frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Canonical Step8A dataset -> 10-cell blocks -> primary population.

    Blocks are assigned BEFORE population filtering, exactly as the frozen
    large-block robustness analysis does, so block identity does not depend on
    which population is being modelled.
    """
    assert_not_excluded(experiment_id)
    if frame is None:
        path = canonical_step8a_path(experiment_id, experiments_root)
        if not path.is_file():
            raise FewShotRecoveryError(
                f"Canonical Step8A dataset missing for {experiment_id!r}: {path}."
            )
        frame = pd.read_parquet(path)

    required = set(THERMAL_MODEL_FEATURES) | {
        TARGET_COLUMN, "valid_for_modeling", POPULATION, "row_500m", "col_500m", "cell_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FewShotRecoveryError(
            f"{experiment_id!r} Step8A dataset is missing required columns: {missing}."
        )

    assigned = assign_large_blocks(frame, BLOCK_SIZE_CELLS)
    population = population_subset(assigned, POPULATION).copy().reset_index(drop=True)

    n_positive = int((population[TARGET_COLUMN] == 1).sum())
    n_negative = int((population[TARGET_COLUMN] == 0).sum())
    if min(n_positive, n_negative) < STEP8B_MIN_POSITIVES_PER_POPULATION:
        raise FewShotRecoveryError(
            f"{experiment_id!r} population {POPULATION!r} has positives={n_positive}, "
            f"negatives={n_negative}; both must be >= "
            f"{STEP8B_MIN_POSITIVES_PER_POPULATION}."
        )
    population["experiment_id"] = experiment_id
    return population


# =============================================================================
# Outer folds  --  target-only, canonical strict spatial CV
# =============================================================================
def build_outer_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Strict 5-fold StratifiedGroupKFold over 10-cell blocks.

    strict=True guarantees, and hard-fails otherwise: the fold count is never
    silently reduced, no block appears on both sides, both classes are present
    on both sides, and every row lands in exactly one test fold.
    """
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    groups = frame[BLOCK_COLUMN].to_numpy()
    folds, n_splits_used = make_spatial_folds(
        y, groups, N_OUTER_FOLDS, FOLD_RANDOM_STATE, strict=True,
    )
    if n_splits_used != N_OUTER_FOLDS:
        raise FewShotRecoveryError(
            f"Strict spatial CV returned {n_splits_used} folds; this analysis "
            f"requires exactly {N_OUTER_FOLDS}."
        )
    return folds


def block_tier_table(frame: pd.DataFrame, block_ids: Iterable[str]) -> pd.DataFrame:
    """Per-block row/positive counts and tier, sorted by block id."""
    wanted = set(block_ids)
    subset = frame[frame[BLOCK_COLUMN].isin(wanted)]
    grouped = subset.groupby(BLOCK_COLUMN)[TARGET_COLUMN].agg(["size", "sum"])
    grouped = grouped.rename(columns={"size": "block_row_count", "sum": "block_positive_count"})
    grouped["block_row_count"] = grouped["block_row_count"].astype(int)
    grouped["block_positive_count"] = grouped["block_positive_count"].astype(int)
    grouped["block_tier"] = np.where(
        grouped["block_positive_count"] == 0, TIER_NEGATIVE,
        np.where(grouped["block_positive_count"] == grouped["block_row_count"],
                 TIER_POSITIVE, TIER_BOTH),
    )
    return grouped.reset_index().sort_values(BLOCK_COLUMN, kind="mergesort").reset_index(drop=True)


def nested_block_ordering(tier_table: pd.DataFrame, seed: int) -> list[str]:
    """One deterministic ordering; every budget is a prefix of it.

    Tiers are exhausted in the frozen order both_classes -> positives_only ->
    negatives_only, each permuted independently. Blocks are sorted by id before
    shuffling, so the ordering does not depend on input row order.

    Because the both-class tier is never empty in any fold of this analysis,
    every k >= 1 prefix contains at least one burned-containing block.
    """
    rng = np.random.default_rng(seed)
    ordering: list[str] = []
    for tier in TIER_ORDER:
        members = tier_table.loc[tier_table["block_tier"] == tier, BLOCK_COLUMN].tolist()
        members = sorted(members)
        if members:
            permuted = rng.permutation(len(members))
            ordering.extend(members[index] for index in permuted)
    return ordering


# =============================================================================
# Fit registry  --  enforces the fit identities, never merely reports them
# =============================================================================
class FitRegistry:
    """Memoises fit RESULTS by fit identity.

    Results (prediction vectors), not fitted estimators, are cached: that keeps
    memory flat while making the sharing contract structural. A second request
    for the same identity can never trigger a second fit.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self.fit_count = 0
        self.reuse_count = 0

    def get_or_fit(self, fit_id: str, condition: str, compute: Callable[[], Any]) -> Any:
        entry = self._entries.get(fit_id)
        if entry is not None:
            entry["reference_count"] += 1
            self.reuse_count += 1
            return entry["result"]
        result = compute()
        self._entries[fit_id] = {
            "condition": condition, "result": result, "reference_count": 1,
        }
        self.fit_count += 1
        return result

    def release(self, prefix: str) -> int:
        """Drop cached results whose identity starts with `prefix`.

        Used to release a direction's raw fit once that direction is written.
        Accounting (`fit_count`, `identities`) is kept, so releasing memory
        never changes the reported fit totals.
        """
        released = [key for key in self._entries if key.startswith(prefix)]
        for key in released:
            self._entries[key]["result"] = None
        return len(released)

    def identities(self) -> dict[str, dict[str, Any]]:
        return {
            fit_id: {"condition": entry["condition"],
                     "reference_count": entry["reference_count"]}
            for fit_id, entry in self._entries.items()
        }

    def accounting(self) -> dict[str, Any]:
        by_condition: dict[str, int] = {condition: 0 for condition in CONDITIONS}
        references: dict[str, int] = {condition: 0 for condition in CONDITIONS}
        for entry in self._entries.values():
            by_condition[entry["condition"]] += 1
            references[entry["condition"]] += entry["reference_count"]
        return {
            "unique_fits": self.fit_count,
            "raw_fits": by_condition[CONDITION_RAW],
            "few_shot_fits": by_condition[CONDITION_FEWSHOT],
            "ceiling_fits": by_condition[CONDITION_CEILING],
            "reuse_events": self.reuse_count,
            "references_by_condition": references,
            "raw_reuse_per_fit": (
                references[CONDITION_RAW] / by_condition[CONDITION_RAW]
                if by_condition[CONDITION_RAW] else None
            ),
            "ceiling_reuse_per_fit": (
                references[CONDITION_CEILING] / by_condition[CONDITION_CEILING]
                if by_condition[CONDITION_CEILING] else None
            ),
        }


def fit_and_predict(train_frame: pd.DataFrame, eval_frame: pd.DataFrame,
                    feature_list: Sequence[str]) -> np.ndarray:
    """The ONLY place a model is fitted.

    Canonical `build_pipeline` -> plain `fit` -> `predict_proba`. No
    `sample_weight`, no oversampling, no tuning. Preprocessing lives inside the
    Pipeline, so imputers and the encoder are fitted on the training frame
    only and never see an evaluation row.
    """
    features = list(feature_list)
    check_no_forbidden_features(features)
    pipeline = build_pipeline(features, MODEL_NAME, ESTIMATOR_SEED)
    pipeline.fit(train_frame[features], train_frame[TARGET_COLUMN].astype(int))
    return pipeline.predict_proba(eval_frame[features])[:, 1].astype(np.float64)


# =============================================================================
# Metrics and recovery
# =============================================================================
def metric_orientation(metric: str) -> str:
    return ORIENTATION_NEGATED if metric in LOWER_IS_BETTER else ORIENTATION_HIGHER


def oriented(metric: str, value: Optional[float]) -> Optional[float]:
    """Higher-is-better view. Brier is negated; ROC/PR are unchanged."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return -float(value) if metric in LOWER_IS_BETTER else float(value)


def selection_interval(values: Sequence[Optional[float]]) -> dict[str, Optional[float]]:
    """Median and 2.5/97.5 percentiles over repeats.

    This describes variability across WHICH BLOCKS WERE SELECTED and nothing
    else. It is never a confidence interval and supports no significance claim.
    """
    clean = [float(v) for v in values
             if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return {"selection_median": None, "selection_interval_lower": None,
                "selection_interval_upper": None, "selection_min": None,
                "selection_max": None, "n_repeats_observed": 0}
    array = np.asarray(clean, dtype=float)
    return {
        "selection_median": float(np.median(array)),
        "selection_interval_lower": float(
            np.percentile(array, SELECTION_PCT_LOW, method=PERCENTILE_METHOD)),
        "selection_interval_upper": float(
            np.percentile(array, SELECTION_PCT_HIGH, method=PERCENTILE_METHOD)),
        "selection_min": float(array.min()),
        "selection_max": float(array.max()),
        "n_repeats_observed": int(array.size),
    }


def recovery_quantities(raw_oriented: Optional[float], fewshot_oriented: Optional[float],
                        ceiling_oriented: Optional[float]) -> dict[str, Any]:
    """Signed, unclipped recovery. Values below 0 and above 1 both survive."""
    if raw_oriented is None or fewshot_oriented is None or ceiling_oriented is None:
        return {
            "absolute_recovery": None, "ceiling_gap": None, "recovery_fraction": None,
            "recovery_fraction_status": STATUS_DEGENERATE,
            "denominator_near_zero": None, "ceiling_not_above_raw": None,
            "recovery_negative": None, "recovery_above_ceiling": None,
        }

    absolute_recovery = fewshot_oriented - raw_oriented
    ceiling_gap = ceiling_oriented - raw_oriented
    denominator_near_zero = bool(abs(ceiling_gap) < DEGENERATE_DENOMINATOR_THRESHOLD)
    ceiling_not_above_raw = bool(ceiling_gap <= 0.0)

    if denominator_near_zero:
        fraction: Optional[float] = None
        status = STATUS_DEGENERATE
    else:
        fraction = float(absolute_recovery / ceiling_gap)
        status = STATUS_CEILING_NOT_ABOVE_RAW if ceiling_not_above_raw else STATUS_INTERPRETABLE

    return {
        "absolute_recovery": float(absolute_recovery),
        "ceiling_gap": float(ceiling_gap),
        "recovery_fraction": fraction,
        "recovery_fraction_status": status,
        "denominator_near_zero": denominator_near_zero,
        "ceiling_not_above_raw": ceiling_not_above_raw,
        "recovery_negative": bool(absolute_recovery < 0.0),
        "recovery_above_ceiling": (
            None if fraction is None else bool(fraction > 1.0)
        ),
    }


def assert_full_oof_coverage(coverage: np.ndarray, context: str) -> None:
    """Every target population row predicted exactly once, with no gap.

    `coverage` counts how many evaluation folds produced a prediction for each
    row. Anything other than exactly 1 everywhere is a contract violation.
    """
    coverage = np.asarray(coverage)
    if not np.all(coverage == 1):
        unpredicted = int((coverage == 0).sum())
        duplicated = int((coverage > 1).sum())
        raise FewShotRecoveryError(
            f"{context}: OOF coverage violated -- {unpredicted} row(s) predicted zero "
            f"times and {duplicated} row(s) predicted more than once."
        )


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
            raise FewShotRecoveryError(
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


def verify_direction_partition(analysis_id: str, direction: str,
                               output_root: Optional[Path] = None) -> bool:
    """Is this direction's OOF partition present AND hash-bound by the marker?

    A direction whose partition exists but is not recorded in a passing `fit`
    marker counts as partial and is never accepted.
    """
    marker = read_stage_marker(analysis_id, "fit", output_root)
    if marker is None or marker.get("status") != "pass":
        return False
    recorded = (marker.get("direction_partitions") or {}).get(direction)
    if not recorded:
        return False
    path = (analysis_root(analysis_id, output_root) / OOF_PREDICTIONS_DIRNAME
            / f"part-{direction}.parquet")
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
# Scientific configuration + identity
# =============================================================================
def build_scientific_config(experiment_ids: Sequence[str],
                            input_inventory: dict[str, Any]) -> dict[str, Any]:
    """The object hashed into `analysis_id`.

    Deliberately excludes wall-clock time, git commit and package versions, so
    that re-running the same frozen contract on the same frozen inputs lands in
    the same namespace.
    """
    classifier = build_classifier(MODEL_NAME, ESTIMATOR_SEED)
    pairs = directed_pairs(experiment_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "primary_experiments": list(experiment_ids),
        "excluded_experiments": dict(EXCLUDED_EXPERIMENTS),
        "directed_pairs": [list(pair) for pair in pairs],
        "expected_directed_pairs": len(pairs),
        "population": POPULATION,
        "valid_universe": VALID_UNIVERSE,
        "model": {
            "name": MODEL_NAME,
            "class": type(classifier).__name__,
            "hyperparameters": classifier.get_params(deep=False),
            "source": "src.step8b_train_baseline_vs_thermal_model.build_classifier",
            "tuning_performed": False,
            "sample_weight_argument_used": False,
            "oversampling_performed": False,
            "pre_existing_class_weighting": {
                "present": True,
                "mechanism": "class_weight='balanced'",
                "note": (
                    "Canonical Step8B/9B/10 behaviour, recomputed by scikit-learn on "
                    "whichever training frame is fitted. Because the few-shot frame is "
                    "source union k target blocks, effective per-row weights shift "
                    "slightly as k grows. Not introduced by this analysis; no new "
                    "weighting rule added."
                ),
            },
        },
        "families": {"primary": PRIMARY_FAMILY, "secondary": "baseline"},
        "feature_lists": {family: list(features) for family, features in FEATURE_LISTS.items()},
        "forbidden_feature_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "preprocessing": {
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot_handle_unknown_ignore",
            "categorical_features": list(CATEGORICAL_FEATURES),
            "fit_scope": "training_frame_of_each_condition_only",
        },
        "spatial_blocks": {
            "block_size_cells": BLOCK_SIZE_CELLS,
            "nominal_scale": BLOCK_NOMINAL_SCALE,
            "utility": "src.step8_large_block_robustness.assign_large_blocks",
            "id_format": "b10_r{block_row}_c{block_col}",
            "origin": [0, 0],
            "assigned_before_population_filtering": True,
            "canonical_small_block_size_cells": CANONICAL_SMALL_BLOCK_SIZE_CELLS,
            "departure_reason": (
                "2-cell blocks hold a median of 4 cells; not a unit of labeling effort "
                "and adjacent to evaluation blocks. The ~5 km fallback is "
                "pre-authorised and already canonical in this repository."
            ),
        },
        "outer_folds": {
            "utility": "src.step8b_train_baseline_vs_thermal_model.make_spatial_folds",
            "splitter": "StratifiedGroupKFold",
            "n_splits": N_OUTER_FOLDS,
            "shuffle": True,
            "random_state": FOLD_RANDOM_STATE,
            "strict": True,
            "grouping_column": BLOCK_COLUMN,
            "depends_on": "target_only",
        },
        "budgets": list(BUDGETS),
        "budgets_dropped": [],
        "n_repeats": N_REPEATS,
        "n_repeats_raw": N_REPEATS_SINGLE_REALISATION,
        "n_repeats_ceiling": N_REPEATS_SINGLE_REALISATION,
        "selection": {
            "unit": "spatial_block",
            "nested": True,
            "tier_order": list(TIER_ORDER),
            "within_tier": "shuffle_with_derived_rng_after_sorting_by_block_id",
            "seed_derivation": (
                "blake2b(schema|source|target|outer_fold|repeat, digest_size=8) mod 2**32"
            ),
            "seed_depends_on_budget": False,
            "seed_depends_on_model_family": False,
            "result_dependent_branching": False,
        },
        "metrics": {
            "primary": PRIMARY_METRIC,
            "secondary": [metric for metric in METRICS if metric != PRIMARY_METRIC],
            "helper": "src.step8b_train_baseline_vs_thermal_model.compute_binary_metrics",
            "brier_orientation": "oriented_value = -brier_score",
            "threshold_selection_performed": False,
        },
        "recovery": {
            "clipped": False,
            "absolute_valued": False,
            "degenerate_denominator_threshold": DEGENERATE_DENOMINATOR_THRESHOLD,
            "statuses": [STATUS_INTERPRETABLE, STATUS_DEGENERATE,
                         STATUS_CEILING_NOT_ABOVE_RAW],
            "flags": ["denominator_near_zero", "ceiling_not_above_raw",
                      "recovery_negative", "recovery_above_ceiling"],
        },
        "uncertainty": {
            "interval_name": "selection_interval",
            "percentiles": [SELECTION_PCT_LOW, SELECTION_PCT_HIGH],
            "percentile_method": PERCENTILE_METHOD,
            "basis": "variability_across_block_selection_repeats_only",
            "bootstrap_performed": False,
            "p_values_produced": False,
            # The firewall vocabulary lives in code, not in the artifact:
            # writing the literal terms here would make config.json trip the
            # very scan it declares.
            "forbidden_terms_source": (
                "src.few_shot_recovery.FORBIDDEN_UNCERTAINTY_TERMS"
            ),
            "forbidden_terms_count": len(FORBIDDEN_UNCERTAINTY_TERMS),
        },
        "earth_engine": {"used": False, "importable": False},
        "input_hashes": {
            experiment_id: entry["sha256"]
            for experiment_id, entry in sorted(input_inventory.items())
        },
    }


# =============================================================================
# Shared provenance columns
# =============================================================================
def _provenance(analysis_id: str, source_id: str, target_id: str,
                input_inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "source_experiment": source_id,
        "target_experiment": target_id,
        "direction": direction_token(source_id, target_id),
        "population": POPULATION,
        "source_step8a_sha256": input_inventory[source_id]["sha256"],
        "target_step8a_sha256": input_inventory[target_id]["sha256"],
        "block_size_cells": BLOCK_SIZE_CELLS,
        "block_nominal_scale": BLOCK_NOMINAL_SCALE,
        "n_outer_folds": N_OUTER_FOLDS,
        "estimator_random_state": ESTIMATOR_SEED,
    }


PROVENANCE_COLUMNS = [
    "schema_version", "analysis_id", "source_experiment", "target_experiment",
    "direction", "population", "source_step8a_sha256", "target_step8a_sha256",
    "block_size_cells", "block_nominal_scale", "n_outer_folds",
    "estimator_random_state",
]


# =============================================================================
# PLAN stage
# =============================================================================
def build_target_context(experiment_ids: Sequence[str],
                         experiments_root: Optional[Path] = None,
                         frames: Optional[dict[str, pd.DataFrame]] = None
                         ) -> dict[str, dict[str, Any]]:
    """Per-target frame, folds, pools and tier tables. Target-only, so the two
    directions sharing a target share all of it."""
    context: dict[str, dict[str, Any]] = {}
    for experiment_id in experiment_ids:
        frame = load_target_frame(
            experiment_id, experiments_root,
            frame=None if frames is None else frames.get(experiment_id),
        )
        folds = build_outer_folds(frame)
        blocks = frame[BLOCK_COLUMN].to_numpy()
        fold_entries = []
        for fold_index, (train_idx, test_idx) in enumerate(folds):
            train_blocks = set(blocks[train_idx])
            eval_blocks = set(blocks[test_idx])
            overlap = train_blocks & eval_blocks
            if overlap:
                raise FewShotRecoveryError(
                    f"{experiment_id} fold {fold_index}: {len(overlap)} block(s) on "
                    "both sides of the split."
                )
            tier_table = block_tier_table(frame, train_blocks)
            fold_entries.append({
                "fold": fold_index,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_blocks": sorted(train_blocks),
                "eval_blocks": sorted(eval_blocks),
                "tier_table": tier_table,
            })
        context[experiment_id] = {"frame": frame, "folds": fold_entries}
    return context


def build_block_inventory(analysis_id: str, context: dict[str, dict[str, Any]]
                          ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_id, entry in context.items():
        frame = entry["frame"]
        whole = block_tier_table(frame, frame[BLOCK_COLUMN].unique())
        rows.append({
            "schema_version": SCHEMA_VERSION, "analysis_id": analysis_id,
            "target_experiment": experiment_id, "population": POPULATION,
            "outer_fold": FOLD_SENTINEL_OOF,
            "block_size_cells": BLOCK_SIZE_CELLS,
            "block_nominal_scale": BLOCK_NOMINAL_SCALE,
            "population_rows": int(len(frame)),
            "population_positives": int((frame[TARGET_COLUMN] == 1).sum()),
            "population_negatives": int((frame[TARGET_COLUMN] == 0).sum()),
            "total_blocks": int(len(whole)),
            "blocks_with_burned": int((whole["block_positive_count"] > 0).sum()),
            "blocks_unburned_only": int((whole["block_tier"] == TIER_NEGATIVE).sum()),
            "blocks_both_classes": int((whole["block_tier"] == TIER_BOTH).sum()),
            "blocks_burned_only": int((whole["block_tier"] == TIER_POSITIVE).sum()),
            "median_rows_per_block": float(whole["block_row_count"].median()),
            "pool_blocks": None, "pool_tier_a": None, "pool_tier_b": None,
            "pool_tier_c": None, "eval_blocks": None, "eval_rows": None,
            "eval_positives": None,
        })
        for fold_entry in entry["folds"]:
            tier_table = fold_entry["tier_table"]
            test_idx = fold_entry["test_idx"]
            y_test = frame[TARGET_COLUMN].astype(int).to_numpy()[test_idx]
            rows.append({
                "schema_version": SCHEMA_VERSION, "analysis_id": analysis_id,
                "target_experiment": experiment_id, "population": POPULATION,
                "outer_fold": fold_entry["fold"],
                "block_size_cells": BLOCK_SIZE_CELLS,
                "block_nominal_scale": BLOCK_NOMINAL_SCALE,
                "population_rows": int(len(frame)),
                "population_positives": int((frame[TARGET_COLUMN] == 1).sum()),
                "population_negatives": int((frame[TARGET_COLUMN] == 0).sum()),
                "total_blocks": int(len(whole)),
                "blocks_with_burned": int((whole["block_positive_count"] > 0).sum()),
                "blocks_unburned_only": int((whole["block_tier"] == TIER_NEGATIVE).sum()),
                "blocks_both_classes": int((whole["block_tier"] == TIER_BOTH).sum()),
                "blocks_burned_only": int((whole["block_tier"] == TIER_POSITIVE).sum()),
                "median_rows_per_block": float(whole["block_row_count"].median()),
                "pool_blocks": int(len(tier_table)),
                "pool_tier_a": int((tier_table["block_tier"] == TIER_BOTH).sum()),
                "pool_tier_b": int((tier_table["block_tier"] == TIER_POSITIVE).sum()),
                "pool_tier_c": int((tier_table["block_tier"] == TIER_NEGATIVE).sum()),
                "eval_blocks": int(len(fold_entry["eval_blocks"])),
                "eval_rows": int(len(test_idx)),
                "eval_positives": int(y_test.sum()),
            })
    return rows


BLOCK_INVENTORY_COLUMNS = [
    "schema_version", "analysis_id", "target_experiment", "population", "outer_fold",
    "block_size_cells", "block_nominal_scale", "population_rows",
    "population_positives", "population_negatives", "total_blocks",
    "blocks_with_burned", "blocks_unburned_only", "blocks_both_classes",
    "blocks_burned_only", "median_rows_per_block", "pool_blocks", "pool_tier_a",
    "pool_tier_b", "pool_tier_c", "eval_blocks", "eval_rows", "eval_positives",
]


def build_selection_plan(analysis_id: str, experiment_ids: Sequence[str],
                         context: dict[str, dict[str, Any]],
                         input_inventory: dict[str, Any]
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze every adaptation-block selection BEFORE any model is fitted.

    Returns (selected_block_rows, feasibility_rows). A budget larger than the
    fold's training pool is a hard failure -- k is never silently reduced.
    """
    selected_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []

    for source_id, target_id in directed_pairs(experiment_ids):
        provenance = _provenance(analysis_id, source_id, target_id, input_inventory)
        eval_block_lookup = {
            fold_entry["fold"]: set(fold_entry["eval_blocks"])
            for fold_entry in context[target_id]["folds"]
        }
        for fold_entry in context[target_id]["folds"]:
            fold = fold_entry["fold"]
            tier_table = fold_entry["tier_table"]
            pool_size = int(len(tier_table))
            tier_counts = {
                TIER_BOTH: int((tier_table["block_tier"] == TIER_BOTH).sum()),
                TIER_POSITIVE: int((tier_table["block_tier"] == TIER_POSITIVE).sum()),
                TIER_NEGATIVE: int((tier_table["block_tier"] == TIER_NEGATIVE).sum()),
            }
            positive_containing = tier_counts[TIER_BOTH] + tier_counts[TIER_POSITIVE]
            block_lookup = tier_table.set_index(BLOCK_COLUMN)

            orderings: dict[int, list[str]] = {}
            for repeat_id in range(N_REPEATS):
                seed = selection_seed(source_id, target_id, fold, repeat_id)
                ordering = nested_block_ordering(tier_table, seed)
                if len(ordering) != pool_size:
                    raise FewShotRecoveryError(
                        f"{provenance['direction']} fold {fold} repeat {repeat_id}: "
                        f"ordering covers {len(ordering)} of {pool_size} pool blocks."
                    )
                orderings[repeat_id] = ordering

            for budget in BUDGETS:
                feasible = budget <= pool_size
                if not feasible:
                    # k is never silently reduced: record it and fail.
                    feasibility_rows.append({
                        **provenance, "outer_fold": fold, "budget_blocks": budget,
                        "target_pool_blocks": pool_size,
                        "pool_tier_a": tier_counts[TIER_BOTH],
                        "pool_tier_b": tier_counts[TIER_POSITIVE],
                        "pool_tier_c": tier_counts[TIER_NEGATIVE],
                        "feasible": False,
                        "infeasibility_reason": (
                            f"budget {budget} exceeds the fold training pool of {pool_size} blocks"
                        ),
                        "fills_from_positive_containing_only": False,
                        "requires_tier_c": None,
                        "min_selected_positive_blocks": None,
                        "n_repeats_planned": (
                            N_REPEATS_SINGLE_REALISATION if budget == 0 else N_REPEATS
                        ),
                    })
                    raise FewShotRecoveryError(
                        f"{provenance['direction']} fold {fold}: budget {budget} exceeds "
                        f"the training pool of {pool_size} blocks. This frozen analysis "
                        "does not reduce k silently; the budget set would have to change."
                    )

                n_repeats = N_REPEATS_SINGLE_REALISATION if budget == 0 else N_REPEATS
                min_positive_blocks = None
                requires_tier_c = False

                for repeat_id in range(n_repeats):
                    if budget == 0:
                        continue
                    ordering = orderings[repeat_id]
                    chosen = ordering[:budget]
                    if len(set(chosen)) != budget:
                        raise FewShotRecoveryError(
                            f"{provenance['direction']} fold {fold} repeat {repeat_id} "
                            f"budget {budget}: selection is not {budget} distinct blocks."
                        )
                    overlap = set(chosen) & eval_block_lookup[fold]
                    if overlap:
                        raise FewShotRecoveryError(
                            f"{provenance['direction']} fold {fold} repeat {repeat_id} "
                            f"budget {budget}: {len(overlap)} adaptation block(s) are "
                            "evaluation blocks."
                        )
                    positives_here = 0
                    for rank, block_id in enumerate(chosen):
                        record = block_lookup.loc[block_id]
                        tier = str(record["block_tier"])
                        if tier == TIER_NEGATIVE:
                            requires_tier_c = True
                        else:
                            positives_here += 1
                        selected_rows.append({
                            **provenance,
                            "selection_key": selection_key(
                                source_id, target_id, fold, repeat_id, budget),
                            "outer_fold": fold,
                            "repeat_id": repeat_id,
                            "budget_blocks": budget,
                            "selection_seed": int(
                                selection_seed(source_id, target_id, fold, repeat_id)),
                            "adaptation_block_id": block_id,
                            "selection_rank": rank,
                            "block_tier": tier,
                            "block_row_count": int(record["block_row_count"]),
                            "block_positive_count": int(record["block_positive_count"]),
                        })
                    if positives_here == 0:
                        raise FewShotRecoveryError(
                            f"{provenance['direction']} fold {fold} repeat {repeat_id} "
                            f"budget {budget}: no burned-containing block selected. The "
                            "tier rule guarantees at least one; the pool must be malformed."
                        )
                    min_positive_blocks = (
                        positives_here if min_positive_blocks is None
                        else min(min_positive_blocks, positives_here)
                    )

                feasibility_rows.append({
                    **provenance, "outer_fold": fold, "budget_blocks": budget,
                    "target_pool_blocks": pool_size,
                    "pool_tier_a": tier_counts[TIER_BOTH],
                    "pool_tier_b": tier_counts[TIER_POSITIVE],
                    "pool_tier_c": tier_counts[TIER_NEGATIVE],
                    "feasible": True,
                    "infeasibility_reason": None,
                    "fills_from_positive_containing_only": bool(budget <= positive_containing),
                    "requires_tier_c": bool(requires_tier_c),
                    "min_selected_positive_blocks": min_positive_blocks,
                    "n_repeats_planned": n_repeats,
                })

    return selected_rows, feasibility_rows


SELECTED_BLOCK_COLUMNS = PROVENANCE_COLUMNS + [
    "selection_key", "outer_fold", "repeat_id", "budget_blocks", "selection_seed",
    "adaptation_block_id", "selection_rank", "block_tier", "block_row_count",
    "block_positive_count",
]

FEASIBILITY_COLUMNS = PROVENANCE_COLUMNS + [
    "outer_fold", "budget_blocks", "target_pool_blocks", "pool_tier_a", "pool_tier_b",
    "pool_tier_c", "feasible", "infeasibility_reason",
    "fills_from_positive_containing_only", "requires_tier_c",
    "min_selected_positive_blocks", "n_repeats_planned",
]


def assert_nested_budgets(selected: pd.DataFrame) -> None:
    """Budget k's blocks must be a prefix of budget k+'s blocks."""
    keys = ["direction", "outer_fold", "repeat_id"]
    for (direction, fold, repeat_id), group in selected.groupby(keys, sort=True):
        by_budget = {
            int(budget): set(chunk["adaptation_block_id"])
            for budget, chunk in group.groupby("budget_blocks")
        }
        ordered = sorted(by_budget)
        for smaller, larger in zip(ordered, ordered[1:]):
            if not by_budget[smaller] <= by_budget[larger]:
                raise FewShotRecoveryError(
                    f"{direction} fold {fold} repeat {repeat_id}: budget {smaller} is "
                    f"not nested inside budget {larger}."
                )
        # Ranks must agree with the shared ordering.
        for budget, chunk in group.groupby("budget_blocks"):
            ranks = sorted(int(rank) for rank in chunk["selection_rank"])
            if ranks != list(range(int(budget))):
                raise FewShotRecoveryError(
                    f"{direction} fold {fold} repeat {repeat_id} budget {budget}: "
                    f"selection ranks are {ranks}, expected 0..{int(budget) - 1}."
                )


def run_plan_stage(analysis_id: str, experiment_ids: Sequence[str],
                   context: dict[str, dict[str, Any]], input_inventory: dict[str, Any],
                   scientific_config: dict[str, Any], output_root: Optional[Path] = None,
                   ) -> dict[str, Any]:
    root = analysis_root(analysis_id, output_root)
    root.mkdir(parents=True, exist_ok=True)

    config_document = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "analysis_id": analysis_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "scientific_configuration": scientific_config,
    }
    _write_namespaced(root, "config.json", _json_document(config_document))

    hashes_document = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "canonical_step8a": input_inventory,
        "external_references": external_ceiling_reference_inventory(
            experiment_ids, output_root),
        "read_only_assertion": (
            "No path outside outputs/diagnostics/few_shot_recovery/<analysis_id>/ was "
            "opened for writing."
        ),
    }
    _write_namespaced(root, "input_hashes.json", _json_document(hashes_document))

    inventory_rows = build_block_inventory(analysis_id, context)
    _write_namespaced(root, "target_block_inventory.csv",
                      _csv_document(BLOCK_INVENTORY_COLUMNS, inventory_rows))

    selected_rows, feasibility_rows = build_selection_plan(
        analysis_id, experiment_ids, context, input_inventory)
    _write_namespaced(root, "direction_budget_feasibility.csv",
                      _csv_document(FEASIBILITY_COLUMNS, feasibility_rows))

    selected = pd.DataFrame(selected_rows, columns=SELECTED_BLOCK_COLUMNS)
    assert_nested_budgets(selected)
    selected_path = root / "selected_blocks.parquet"
    assert_inside_namespace(selected_path, root)
    _atomic_write_parquet(selected_path, selected)

    marker = write_stage_marker(analysis_id, "plan", output_root, extra={
        "directed_pairs": len(directed_pairs(experiment_ids)),
        "selected_block_rows": int(len(selected)),
        "feasibility_rows": len(feasibility_rows),
        "all_budgets_feasible": all(row["feasible"] for row in feasibility_rows),
    })
    return {
        "stage": "plan",
        "selected_block_rows": int(len(selected)),
        "feasibility_rows": len(feasibility_rows),
        "block_inventory_rows": len(inventory_rows),
        "marker": marker,
    }


def _write_namespaced(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    assert_inside_namespace(path, root)
    _atomic_write_text(path, text)
    return path


# =============================================================================
# FIT stage
# =============================================================================
REPEAT_METRIC_COLUMNS = PROVENANCE_COLUMNS + [
    "model_family", "model_role", "metric", "metric_orientation", "condition",
    "budget_blocks", "repeat_id", "evaluation_level", "outer_fold", "metric_value",
    "oriented_value", "n_evaluation_rows", "n_evaluation_positives",
    "n_evaluation_blocks", "n_train_rows", "n_train_source_rows",
    "n_train_target_rows", "adaptation_row_count", "adaptation_positive_count",
    "n_blocks_tier_a", "n_blocks_tier_b", "n_blocks_tier_c", "selection_key", "fit_id",
]

OOF_PREDICTION_COLUMNS = PROVENANCE_COLUMNS + [
    "condition", "budget_blocks", "repeat_id", "outer_fold", "evaluation_block_id",
    "cell_id", "burned", "baseline_probability", "thermal_probability", "selection_key",
]


def _series_plan() -> list[dict[str, Any]]:
    """Every (condition, budget, repeat) series a direction must produce."""
    plan: list[dict[str, Any]] = [
        {"condition": CONDITION_RAW, "budget_blocks": 0, "repeat_id": 0},
        {"condition": CONDITION_CEILING, "budget_blocks": BUDGET_CEILING_SENTINEL,
         "repeat_id": 0},
    ]
    for budget in NONZERO_BUDGETS:
        for repeat_id in range(N_REPEATS):
            plan.append({"condition": CONDITION_FEWSHOT, "budget_blocks": budget,
                         "repeat_id": repeat_id})
    return plan


def run_direction(analysis_id: str, source_id: str, target_id: str,
                  context: dict[str, dict[str, Any]], selected: pd.DataFrame,
                  input_inventory: dict[str, Any], registry: FitRegistry,
                  ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit and evaluate one direction. Returns (oof_frame, repeat_metric_rows).

    Memory discipline: only this direction's predictions are held; the caller
    writes and releases before starting the next direction.
    """
    provenance = _provenance(analysis_id, source_id, target_id, input_inventory)
    direction = provenance["direction"]

    source_frame = context[source_id]["frame"]
    target_frame = context[target_id]["frame"]
    target_folds = context[target_id]["folds"]

    target_cells = target_frame["cell_id"].to_numpy()
    target_labels = target_frame[TARGET_COLUMN].astype(int).to_numpy()
    target_blocks = target_frame[BLOCK_COLUMN].to_numpy()
    n_target = len(target_frame)

    fold_of_row = np.full(n_target, -1, dtype=int)
    for fold_entry in target_folds:
        fold_of_row[fold_entry["test_idx"]] = fold_entry["fold"]
    if (fold_of_row < 0).any():
        raise FewShotRecoveryError(
            f"{direction}: {int((fold_of_row < 0).sum())} target rows are in no "
            "evaluation fold."
        )

    selection_lookup: dict[tuple[int, int, int], list[str]] = {}
    direction_selection = selected[selected["direction"] == direction]
    for (fold, repeat_id, budget), group in direction_selection.groupby(
        ["outer_fold", "repeat_id", "budget_blocks"]
    ):
        ordered = group.sort_values("selection_rank")
        selection_lookup[(int(fold), int(repeat_id), int(budget))] = list(
            ordered["adaptation_block_id"])

    target_block_count = int(len(set(target_blocks.tolist())))
    fold_block_counts = {
        fold_entry["fold"]: len(fold_entry["eval_blocks"]) for fold_entry in target_folds
    }

    series_plan = _series_plan()
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for series in series_plan:
        condition = series["condition"]
        budget = int(series["budget_blocks"])
        repeat_id = int(series["repeat_id"])

        oof_probability = {family: np.full(n_target, np.nan) for family in MODEL_FAMILIES}
        coverage = np.zeros(n_target, dtype=int)
        per_fold_stats: dict[int, dict[str, Any]] = {}

        for fold_entry in target_folds:
            fold = fold_entry["fold"]
            test_idx = fold_entry["test_idx"]
            train_idx = fold_entry["train_idx"]
            eval_frame = target_frame.iloc[test_idx]

            if condition == CONDITION_RAW:
                train_frame = source_frame
                target_train_frame = target_frame.iloc[0:0]
                adaptation_blocks: list[str] = []
            elif condition == CONDITION_CEILING:
                train_frame = target_frame.iloc[train_idx]
                target_train_frame = train_frame
                adaptation_blocks = []
            else:
                adaptation_blocks = selection_lookup[(fold, repeat_id, budget)]
                adaptation_rows = target_frame.iloc[train_idx]
                adaptation_rows = adaptation_rows[
                    adaptation_rows[BLOCK_COLUMN].isin(adaptation_blocks)]
                target_train_frame = adaptation_rows
                train_frame = pd.concat([source_frame, adaptation_rows], ignore_index=True)

            # --- firewall: no evaluation cell or block may appear in the
            # TARGET-derived part of a training frame.
            #
            # Scoped to the target on purpose: cell_id is "r{row}_c{col}" over
            # each AOI's own grid, so it is NOT unique across regions -- a
            # source cell can legitimately carry the same id as a target cell.
            # Comparing across regions would raise on a coincidence rather than
            # a leak. Source rows can never leak a target evaluation label
            # because they carry the source's labels, not the target's.
            if len(target_train_frame):
                train_cells = set(target_train_frame["cell_id"].to_numpy().tolist())
                eval_cells = set(eval_frame["cell_id"].to_numpy().tolist())
                leaked_cells = train_cells & eval_cells
                train_blocks = set(target_train_frame[BLOCK_COLUMN].to_numpy().tolist())
                leaked_blocks = train_blocks & set(fold_entry["eval_blocks"])
                if leaked_cells or leaked_blocks:
                    raise FewShotRecoveryError(
                        f"{direction} fold {fold} {condition}: "
                        f"{len(leaked_cells)} evaluation cell(s) and "
                        f"{len(leaked_blocks)} evaluation block(s) are in the target "
                        "part of the training frame."
                    )

            adaptation_row_count = 0
            adaptation_positive_count = 0
            tier_counts = {TIER_BOTH: 0, TIER_POSITIVE: 0, TIER_NEGATIVE: 0}
            if condition == CONDITION_FEWSHOT:
                tier_table = fold_entry["tier_table"].set_index(BLOCK_COLUMN)
                for block_id in adaptation_blocks:
                    record = tier_table.loc[block_id]
                    adaptation_row_count += int(record["block_row_count"])
                    adaptation_positive_count += int(record["block_positive_count"])
                    tier_counts[str(record["block_tier"])] += 1

            for family in MODEL_FAMILIES:
                features = FEATURE_LISTS[family]
                fit_id = fit_identity(
                    condition, family=family, source_id=source_id, target_id=target_id,
                    outer_fold=fold, budget=budget, repeat_id=repeat_id,
                )

                if condition == CONDITION_RAW:
                    # Fold-independent: fit once, predict the whole target, slice.
                    def _compute_raw(features=features):
                        return fit_and_predict(source_frame, target_frame, features)

                    full = registry.get_or_fit(fit_id, condition, _compute_raw)
                    probability = np.asarray(full)[test_idx]
                else:
                    def _compute(train_frame=train_frame, eval_frame=eval_frame,
                                 features=features):
                        return fit_and_predict(train_frame, eval_frame, features)

                    probability = np.asarray(registry.get_or_fit(fit_id, condition, _compute))

                oof_probability[family][test_idx] = probability

            coverage[test_idx] += 1
            per_fold_stats[fold] = {
                "n_train_rows": int(len(train_frame)),
                "n_train_source_rows": (
                    0 if condition == CONDITION_CEILING else int(len(source_frame))),
                "n_train_target_rows": (
                    int(len(train_frame)) if condition == CONDITION_CEILING
                    else adaptation_row_count),
                "adaptation_row_count": adaptation_row_count,
                "adaptation_positive_count": adaptation_positive_count,
                "n_blocks_tier_a": tier_counts[TIER_BOTH],
                "n_blocks_tier_b": tier_counts[TIER_POSITIVE],
                "n_blocks_tier_c": tier_counts[TIER_NEGATIVE],
                "test_idx": test_idx,
            }

        context_label = (
            f"{direction} {condition} budget={budget} repeat={repeat_id}"
        )
        assert_full_oof_coverage(coverage, context_label)
        for family in MODEL_FAMILIES:
            if np.isnan(oof_probability[family]).any():
                raise FewShotRecoveryError(
                    f"{context_label} {family}: OOF vector has "
                    f"{int(np.isnan(oof_probability[family]).sum())} unpredicted rows."
                )

        # One string object per fold, referenced n_target times -- not n_target
        # distinct Python strings.
        if condition == CONDITION_FEWSHOT:
            fold_keys = np.array(
                [selection_key(source_id, target_id, fold, repeat_id, budget)
                 for fold in range(N_OUTER_FOLDS)], dtype=object)
            selection_keys = fold_keys[fold_of_row]
        else:
            selection_keys = np.full(n_target, None, dtype=object)

        prediction_frames.append(pd.DataFrame({
            "condition": condition,
            "budget_blocks": np.full(n_target, budget, dtype=np.int16),
            "repeat_id": np.full(n_target, repeat_id, dtype=np.int8),
            "outer_fold": fold_of_row.astype(np.int8),
            "evaluation_block_id": target_blocks,
            "cell_id": target_cells,
            "burned": target_labels.astype(np.int8),
            "baseline_probability": oof_probability["baseline"].astype(np.float32),
            "thermal_probability": oof_probability["thermal"].astype(np.float32),
            "selection_key": selection_keys,
        }))

        metric_rows.extend(_metric_rows_for_series(
            provenance, condition, budget, repeat_id, target_labels,
            target_block_count, fold_block_counts, oof_probability, per_fold_stats,
            source_id, target_id,
        ))

        del oof_probability

    oof_frame = pd.concat(prediction_frames, ignore_index=True)
    # Constant provenance is attached once, after the concat, so the six string
    # columns are pointer arrays rather than per-series copies.
    for column in PROVENANCE_COLUMNS:
        oof_frame[column] = provenance[column]
    return oof_frame[OOF_PREDICTION_COLUMNS], metric_rows


def _metric_rows_for_series(provenance: dict[str, Any], condition: str, budget: int,
                            repeat_id: int, labels: np.ndarray,
                            target_block_count: int,
                            fold_block_counts: dict[int, int],
                            oof_probability: dict[str, np.ndarray],
                            per_fold_stats: dict[int, dict[str, Any]],
                            source_id: str, target_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_train = sum(stats["n_train_rows"] for stats in per_fold_stats.values())
    n_folds = max(len(per_fold_stats), 1)

    for family in MODEL_FAMILIES:
        probability = oof_probability[family]
        overall = compute_binary_metrics(labels, probability)
        for metric in METRICS:
            value = overall.get(metric)
            rows.append({
                **provenance,
                "model_family": family, "model_role": MODEL_ROLES[family],
                "metric": metric, "metric_orientation": metric_orientation(metric),
                "condition": condition, "budget_blocks": budget, "repeat_id": repeat_id,
                "evaluation_level": EVALUATION_LEVEL_OOF, "outer_fold": FOLD_SENTINEL_OOF,
                "metric_value": value, "oriented_value": oriented(metric, value),
                "n_evaluation_rows": int(len(labels)),
                "n_evaluation_positives": int(labels.sum()),
                "n_evaluation_blocks": target_block_count,
                "n_train_rows": int(round(total_train / n_folds)),
                "n_train_source_rows": int(round(
                    sum(s["n_train_source_rows"] for s in per_fold_stats.values()) / n_folds)),
                "n_train_target_rows": int(round(
                    sum(s["n_train_target_rows"] for s in per_fold_stats.values()) / n_folds)),
                "adaptation_row_count": int(round(
                    sum(s["adaptation_row_count"] for s in per_fold_stats.values()) / n_folds)),
                "adaptation_positive_count": int(round(
                    sum(s["adaptation_positive_count"] for s in per_fold_stats.values()) / n_folds)),
                "n_blocks_tier_a": int(round(
                    sum(s["n_blocks_tier_a"] for s in per_fold_stats.values()) / n_folds)),
                "n_blocks_tier_b": int(round(
                    sum(s["n_blocks_tier_b"] for s in per_fold_stats.values()) / n_folds)),
                "n_blocks_tier_c": int(round(
                    sum(s["n_blocks_tier_c"] for s in per_fold_stats.values()) / n_folds)),
                # An OOF row spans five evaluation folds, so it has no single
                # selection and -- except for the fold-independent raw fit -- no
                # single fit identity.
                "selection_key": None,
                "fit_id": (
                    fit_identity(condition, family=family, source_id=source_id,
                                 target_id=target_id, outer_fold=None,
                                 budget=budget, repeat_id=repeat_id)
                    if condition == CONDITION_RAW else None
                ),
            })

        for fold, stats in sorted(per_fold_stats.items()):
            test_idx = stats["test_idx"]
            fold_metrics = compute_binary_metrics(labels[test_idx], probability[test_idx])
            fold_key = (selection_key(source_id, target_id, fold, repeat_id, budget)
                        if condition == CONDITION_FEWSHOT else None)
            for metric in METRICS:
                value = fold_metrics.get(metric)
                rows.append({
                    **provenance,
                    "model_family": family, "model_role": MODEL_ROLES[family],
                    "metric": metric, "metric_orientation": metric_orientation(metric),
                    "condition": condition, "budget_blocks": budget,
                    "repeat_id": repeat_id, "evaluation_level": EVALUATION_LEVEL_FOLD,
                    "outer_fold": fold,
                    "metric_value": value, "oriented_value": oriented(metric, value),
                    "n_evaluation_rows": int(len(test_idx)),
                    "n_evaluation_positives": int(labels[test_idx].sum()),
                    "n_evaluation_blocks": fold_block_counts[fold],
                    "n_train_rows": stats["n_train_rows"],
                    "n_train_source_rows": stats["n_train_source_rows"],
                    "n_train_target_rows": stats["n_train_target_rows"],
                    "adaptation_row_count": stats["adaptation_row_count"],
                    "adaptation_positive_count": stats["adaptation_positive_count"],
                    "n_blocks_tier_a": stats["n_blocks_tier_a"],
                    "n_blocks_tier_b": stats["n_blocks_tier_b"],
                    "n_blocks_tier_c": stats["n_blocks_tier_c"],
                    "selection_key": fold_key,
                    "fit_id": fit_identity(
                        condition, family=family, source_id=source_id,
                        target_id=target_id, outer_fold=fold, budget=budget,
                        repeat_id=repeat_id),
                })
    return rows


def run_fit_stage(analysis_id: str, experiment_ids: Sequence[str],
                  context: dict[str, dict[str, Any]], input_inventory: dict[str, Any],
                  output_root: Optional[Path] = None, resume: bool = False,
                  ) -> dict[str, Any]:
    root = analysis_root(analysis_id, output_root)
    selected_path = root / "selected_blocks.parquet"
    if not selected_path.is_file():
        raise FewShotRecoveryError(
            "Stage 'fit' requires the frozen selection from 'plan'; "
            f"{selected_path} is missing."
        )
    selected = pd.read_parquet(selected_path)
    assert_nested_budgets(selected)

    predictions_dir = root / OOF_PREDICTIONS_DIRNAME
    predictions_dir.mkdir(parents=True, exist_ok=True)

    registry = FitRegistry()
    partitions: dict[str, str] = {}
    metric_rows: list[dict[str, Any]] = []
    reused_directions: list[str] = []

    previous_marker = read_stage_marker(analysis_id, "fit", output_root) if resume else None
    previous_metrics: Optional[pd.DataFrame] = None
    if previous_marker is not None and (root / "repeat_metrics.csv").is_file():
        previous_metrics = pd.read_csv(root / "repeat_metrics.csv")

    for source_id, target_id in directed_pairs(experiment_ids):
        direction = direction_token(source_id, target_id)

        if resume and verify_direction_partition(analysis_id, direction, output_root):
            # Complete AND hash-bound: reuse. A partition that merely exists on
            # disk without a passing marker entry is never accepted.
            partitions[direction] = sha256_file(
                predictions_dir / f"part-{direction}.parquet")
            if previous_metrics is not None:
                kept = previous_metrics[previous_metrics["direction"] == direction]
                metric_rows.extend(kept.to_dict("records"))
            reused_directions.append(direction)
            continue

        oof_frame, rows = run_direction(
            analysis_id, source_id, target_id, context, selected, input_inventory,
            registry,
        )
        partition_path = predictions_dir / f"part-{direction}.parquet"
        assert_inside_namespace(partition_path, root)
        _atomic_write_parquet(partition_path, oof_frame)
        partitions[direction] = sha256_file(partition_path)
        metric_rows.extend(rows)

        # Release this direction's predictions and its raw fit before moving on.
        del oof_frame
        registry.release(f"{CONDITION_RAW}|{source_id}|{target_id}|")

    _write_namespaced(root, "repeat_metrics.csv",
                      _csv_document(REPEAT_METRIC_COLUMNS, metric_rows))

    accounting = registry.accounting()
    expected = expected_unique_fit_count(
        len(directed_pairs(experiment_ids)), len(experiment_ids))
    accounting["expected"] = expected
    accounting["matches_expected"] = (
        not reused_directions and accounting["unique_fits"] == expected["unique_fits"]
    )
    accounting["reused_directions"] = reused_directions

    marker = write_stage_marker(analysis_id, "fit", output_root, extra={
        "direction_partitions": partitions,
        "fit_accounting": accounting,
        "fit_identities": len(registry.identities()),
        "repeat_metric_rows": len(metric_rows),
    })
    return {
        "stage": "fit", "fit_accounting": accounting,
        "direction_partitions": list(partitions),
        "repeat_metric_rows": len(metric_rows), "marker": marker,
    }


# =============================================================================
# SUMMARIZE stage
# =============================================================================
RECOVERY_CURVE_COLUMNS = PROVENANCE_COLUMNS + [
    "model_family", "model_role", "metric", "metric_orientation", "budget_blocks",
    "n_repeats", "raw_value", "fewshot_value", "ceiling_value", "raw_oriented",
    "fewshot_oriented", "ceiling_oriented", "absolute_recovery", "ceiling_gap",
    "recovery_fraction", "recovery_fraction_status", "denominator_near_zero",
    "ceiling_not_above_raw", "recovery_negative", "recovery_above_ceiling",
    "selection_median", "selection_interval_lower", "selection_interval_upper",
    "selection_min", "selection_max", "n_repeats_observed",
    "recovery_fraction_selection_median", "recovery_fraction_selection_lower",
    "recovery_fraction_selection_upper", "mean_adaptation_row_count",
    "mean_adaptation_positive_count", "mean_n_blocks_tier_a", "mean_n_blocks_tier_b",
    "mean_n_blocks_tier_c",
]


def build_recovery_curve(repeat_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    oof = repeat_metrics[repeat_metrics["evaluation_level"] == EVALUATION_LEVEL_OOF]
    rows: list[dict[str, Any]] = []

    for (direction, family, metric), group in oof.groupby(
        ["direction", "model_family", "metric"], sort=True
    ):
        base = group.iloc[0]
        provenance = {column: base[column] for column in PROVENANCE_COLUMNS}

        raw_rows = group[group["condition"] == CONDITION_RAW]
        ceiling_rows = group[group["condition"] == CONDITION_CEILING]
        if len(raw_rows) != 1 or len(ceiling_rows) != 1:
            raise FewShotRecoveryError(
                f"{direction}/{family}/{metric}: expected exactly one raw and one "
                f"ceiling OOF row; got {len(raw_rows)} and {len(ceiling_rows)}."
            )
        raw_value = _as_float(raw_rows.iloc[0]["metric_value"])
        ceiling_value = _as_float(ceiling_rows.iloc[0]["metric_value"])
        raw_oriented = oriented(metric, raw_value)
        ceiling_oriented = oriented(metric, ceiling_value)

        for budget in BUDGETS:
            if budget == 0:
                values = [raw_value]
                subset = raw_rows
            else:
                subset = group[(group["condition"] == CONDITION_FEWSHOT)
                               & (group["budget_blocks"] == budget)]
                if len(subset) != N_REPEATS:
                    raise FewShotRecoveryError(
                        f"{direction}/{family}/{metric} budget {budget}: expected "
                        f"{N_REPEATS} repeats, found {len(subset)}."
                    )
                values = [_as_float(v) for v in subset["metric_value"]]

            interval = selection_interval(values)
            fewshot_value = interval["selection_median"]
            fewshot_oriented = oriented(metric, fewshot_value)
            quantities = recovery_quantities(raw_oriented, fewshot_oriented, ceiling_oriented)

            per_repeat_fraction = [
                recovery_quantities(raw_oriented, oriented(metric, value),
                                    ceiling_oriented)["recovery_fraction"]
                for value in values
            ]
            fraction_interval = selection_interval(per_repeat_fraction)

            rows.append({
                **provenance,
                "model_family": family, "model_role": MODEL_ROLES[family],
                "metric": metric, "metric_orientation": metric_orientation(metric),
                "budget_blocks": budget,
                "n_repeats": (N_REPEATS_SINGLE_REALISATION if budget == 0 else N_REPEATS),
                "raw_value": raw_value, "fewshot_value": fewshot_value,
                "ceiling_value": ceiling_value,
                "raw_oriented": raw_oriented, "fewshot_oriented": fewshot_oriented,
                "ceiling_oriented": ceiling_oriented,
                **quantities,
                **interval,
                "recovery_fraction_selection_median": fraction_interval["selection_median"],
                "recovery_fraction_selection_lower": fraction_interval["selection_interval_lower"],
                "recovery_fraction_selection_upper": fraction_interval["selection_interval_upper"],
                "mean_adaptation_row_count": float(subset["adaptation_row_count"].mean()),
                "mean_adaptation_positive_count": float(
                    subset["adaptation_positive_count"].mean()),
                "mean_n_blocks_tier_a": float(subset["n_blocks_tier_a"].mean()),
                "mean_n_blocks_tier_b": float(subset["n_blocks_tier_b"].mean()),
                "mean_n_blocks_tier_c": float(subset["n_blocks_tier_c"].mean()),
            })
    return rows


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


LIMITATIONS: tuple[str, ...] = (
    "Outer evaluation blocks are 10-cell (~5 km), not Step8B's canonical 2-cell "
    "blocks. Values are not directly comparable to 2-cell Step8B/Step9B/Step10 numbers.",
    "No bootstrap interval exists for the raw endpoint at this block scale; existing "
    "Step9C/Step10 replicates resample 2-cell blocks and are not comparable. No new "
    "bootstrap was designed.",
    "The reported interval is a selection interval over 10 repeats. It describes "
    "block-selection variability only, is not a confidence interval, and supports no "
    "claim about statistical support.",
    "mugla_2021 has no frozen 10-cell ceiling artifact; its ceiling carries no external "
    "reproduction anchor.",
    "At k=16 and k=32 some folds must include unburned-only adaptation blocks; the tier "
    "composition columns record where.",
    "evia_2021_extended is excluded by design; nothing here describes high-prevalence "
    "different-regime transfer.",
)


def build_summary(analysis_id: str, experiment_ids: Sequence[str],
                  curve: pd.DataFrame, repeat_metrics: pd.DataFrame,
                  fit_accounting: dict[str, Any], output_root: Optional[Path] = None,
                  ) -> dict[str, Any]:
    headline_rows = curve[(curve["metric"] == PRIMARY_METRIC)
                          & (curve["model_family"] == PRIMARY_FAMILY)]
    per_direction: list[dict[str, Any]] = []
    for direction, group in headline_rows.groupby("direction", sort=True):
        ordered = group.sort_values("budget_blocks")
        first = ordered.iloc[0]
        per_direction.append({
            "direction": direction,
            "source_experiment": first["source_experiment"],
            "target_experiment": first["target_experiment"],
            "raw_value": _as_float(first["raw_value"]),
            "ceiling_value": _as_float(first["ceiling_value"]),
            "ceiling_gap": _as_float(first["ceiling_gap"]),
            "budget_curve": [
                {
                    "budget_blocks": int(row["budget_blocks"]),
                    "fewshot_value": _as_float(row["fewshot_value"]),
                    "selection_interval_lower": _as_float(row["selection_interval_lower"]),
                    "selection_interval_upper": _as_float(row["selection_interval_upper"]),
                    "absolute_recovery": _as_float(row["absolute_recovery"]),
                    "recovery_fraction": _as_float(row["recovery_fraction"]),
                    "recovery_fraction_status": row["recovery_fraction_status"],
                }
                for _, row in ordered.iterrows()
            ],
        })

    ceiling_reproduction = _ceiling_reproduction(curve, experiment_ids, output_root)
    external = _external_ceiling_reference(experiment_ids, output_root)

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "headline": {
            "primary_metric": PRIMARY_METRIC,
            "primary_family": PRIMARY_FAMILY,
            "interval_name": "selection_interval",
            "per_direction": per_direction,
        },
        "external_ceiling_reference": external,
        "ceiling_reproduction": ceiling_reproduction,
        "limitations": list(LIMITATIONS),
        "fit_accounting": fit_accounting,
        "recovery_curve_rows": int(len(curve)),
        "repeat_metric_rows": int(len(repeat_metrics)),
        "p_values_produced": False,
        "bootstrap_performed": False,
        "earth_engine_used": False,
    }


def _ceiling_reproduction(curve: pd.DataFrame, experiment_ids: Sequence[str],
                          output_root: Optional[Path] = None) -> dict[str, Any]:
    """Check the produced ceiling against the frozen 10-cell artifacts."""
    result: dict[str, Any] = {}
    for experiment_id in experiment_ids:
        reference = FROZEN_CEILING_REFERENCE.get(experiment_id)
        if reference is None:
            result[experiment_id] = {
                "available": False, "reason": "no_frozen_block_10_artifact",
            }
            continue
        families: dict[str, Any] = {}
        for family in MODEL_FAMILIES:
            expected = reference["roc_auc"][family]
            rows = curve[(curve["target_experiment"] == experiment_id)
                         & (curve["model_family"] == family)
                         & (curve["metric"] == PRIMARY_METRIC)]
            observed_values = {
                _as_float(value) for value in rows["ceiling_value"] if value is not None
            }
            observed = next(iter(observed_values)) if len(observed_values) == 1 else None
            abs_diff = None if observed is None else abs(observed - expected)
            families[family] = {
                "expected": expected, "observed": observed, "abs_diff": abs_diff,
                "tolerance": CEILING_REPRODUCTION_TOLERANCE,
                "match": None if abs_diff is None else bool(
                    abs_diff <= CEILING_REPRODUCTION_TOLERANCE),
                "distinct_observed_values": len(observed_values),
            }
        result[experiment_id] = {"available": True, "family": families}
    return result


def _external_ceiling_reference(experiment_ids: Sequence[str],
                                output_root: Optional[Path] = None) -> dict[str, Any]:
    """Copy the frozen 10-cell ceiling bootstrap in verbatim, correctly labelled.

    It is reproduced for context only. It is NOT a selection interval, and there
    is no comparable interval for the raw endpoint.
    """
    base = Path(output_root).parent if output_root is not None else PROJECT_ROOT
    result: dict[str, Any] = {}
    for experiment_id in experiment_ids:
        reference = FROZEN_CEILING_REFERENCE.get(experiment_id)
        if reference is None:
            result[experiment_id] = {"available": False,
                                     "reason": "no_frozen_block_10_artifact"}
            continue
        path = base / reference["bootstrap_path"]
        if not path.is_file():
            result[experiment_id] = {
                "available": False,
                "reason": "frozen_block_10_bootstrap_not_found_on_disk",
                "source_path": str(path),
            }
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        series = payload.get("series", {})
        entry = {
            "available": True,
            "source_path": str(path),
            "interval_name_in_source": "spatial_block_bootstrap_2_5 / spatial_block_bootstrap_97_5",
            "n_replicates": payload.get("valid_replicates"),
            "random_seed": payload.get("random_seed"),
            "block_size_cells": payload.get("block_size_cells"),
            "is_selection_interval": False,
            "comparable_to_raw_endpoint": False,
            "note": (
                "Frozen 10-cell paired spatial-block bootstrap of the target-only "
                "ceiling. Reproduced verbatim for context. It is NOT a selection "
                "interval and there is no comparable interval for the raw endpoint."
            ),
        }
        for family in MODEL_FAMILIES:
            key = f"auc_{family}"
            if key in series:
                entry[key] = {
                    "lower": series[key].get("ci_2_5"),
                    "upper": series[key].get("ci_97_5"),
                }
        result[experiment_id] = entry
    return result


def render_report(analysis_id: str, curve: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Few-Shot Recovery Curve",
        "",
        f"- schema: `{SCHEMA_VERSION}`",
        f"- analysis_id: `{analysis_id}`",
        f"- diagnostic class: `{DIAGNOSTIC_CLASS}`",
        f"- population: `{POPULATION}`",
        f"- blocks: {BLOCK_SIZE_CELLS} cells ({BLOCK_NOMINAL_SCALE}), "
        f"{N_OUTER_FOLDS} strict spatial folds",
        f"- budgets: {list(BUDGETS)}; {N_REPEATS} repeats for every k > 0",
        "",
        "## Claim boundary",
        "",
        "This is a supervised adaptation sensitivity analysis in which target labels",
        "are deliberately used. It is **not** an operational deployment claim, **not**",
        "active learning, **not** a causal decomposition and **not** target-label-free",
        "adaptation.",
        "",
        "## Forced decisions",
        "",
        "1. Outer evaluation blocks are 10-cell (~5 km), not Step8B's canonical 2-cell",
        "   blocks: a 2-cell block holds a median of 4 cells, so it is not a unit of",
        "   labeling effort and would sit adjacent to the evaluation blocks.",
        "2. No new bootstrap was designed. The frozen 10-cell ceiling artifacts are",
        "   reused as a reproduction anchor; the 2-cell raw-transfer replicates are not",
        "   comparable and are not reused.",
        "3. k=0 and the ceiling carry one deterministic realisation, not ten duplicates,",
        "   because neither has any block-selection randomness.",
        "",
        "## Uncertainty wording",
        "",
        "Every interval reported here is a **selection interval**: the 2.5th and 97.5th",
        f"percentiles across the {N_REPEATS} block-selection repeats. It describes which",
        "blocks were selected and nothing else.",
        "No hypothesis test was performed and no p-value is reported.",
        "",
        f"## Primary curve — {PRIMARY_METRIC}, {PRIMARY_FAMILY} model",
        "",
    ]

    for entry in summary["headline"]["per_direction"]:
        lines.extend([
            f"### {entry['direction']}",
            "",
            f"- raw (k=0): {_fmt(entry['raw_value'])}",
            f"- ceiling (target-only): {_fmt(entry['ceiling_value'])}",
            f"- ceiling gap: {_fmt(entry['ceiling_gap'])}",
            "",
            "| budget | few-shot | selection interval | absolute recovery | recovery fraction | status |",
            "|---:|---:|---|---:|---:|---|",
        ])
        for point in entry["budget_curve"]:
            lines.append(
                f"| {point['budget_blocks']} | {_fmt(point['fewshot_value'])} | "
                f"[{_fmt(point['selection_interval_lower'])}, "
                f"{_fmt(point['selection_interval_upper'])}] | "
                f"{_fmt(point['absolute_recovery'])} | "
                f"{_fmt(point['recovery_fraction'])} | "
                f"{point['recovery_fraction_status']} |"
            )
        lines.append("")

    lines.extend([
        "## Secondary metrics",
        "",
        "`pr_auc` and `brier_score` are in `recovery_curve.csv` for every direction,",
        "family and budget. Brier is lower-is-better, so recovery arithmetic uses",
        "`oriented_value = -brier_score`; the natural-sign Brier is preserved in",
        "`metric_value` and `raw_value`/`fewshot_value`/`ceiling_value`.",
        "",
        "## Recovery-fraction rules",
        "",
        "The fraction is signed and unclipped. Values below 0 (few-shot worse than raw)",
        "and above 1 (few-shot above the ceiling) are preserved. A denominator within",
        f"{DEGENERATE_DENOMINATOR_THRESHOLD} of zero yields an undefined fraction, and a",
        "ceiling at or below raw is flagged rather than suppressed.",
        "",
        "## Ceiling reproduction",
        "",
    ])
    for experiment_id, entry in sorted(summary["ceiling_reproduction"].items()):
        if not entry.get("available"):
            lines.append(f"- `{experiment_id}`: no frozen 10-cell anchor ({entry.get('reason')}).")
            continue
        for family, detail in sorted(entry["family"].items()):
            lines.append(
                f"- `{experiment_id}` / {family}: expected {_fmt(detail['expected'])}, "
                f"observed {_fmt(detail['observed'])}, match={detail['match']}"
            )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.extend([
        "",
        "## Fit accounting",
        "",
        f"- unique fits: {summary['fit_accounting'].get('unique_fits')}",
        f"- raw fits: {summary['fit_accounting'].get('raw_fits')} "
        f"(reused across folds; mean references per fit "
        f"{summary['fit_accounting'].get('raw_reuse_per_fit')})",
        f"- ceiling fits: {summary['fit_accounting'].get('ceiling_fits')} "
        f"(shared across the directions of a target; mean references per fit "
        f"{summary['fit_accounting'].get('ceiling_reuse_per_fit')})",
        f"- few-shot fits: {summary['fit_accounting'].get('few_shot_fits')}",
        "",
    ])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{float(value):.6f}"


def build_manifest(analysis_id: str, output_root: Optional[Path] = None) -> dict[str, Any]:
    root = analysis_root(analysis_id, output_root)
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "manifest.json":
            continue
        relative = str(path.relative_to(root))
        files.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    predictions_dir = root / OOF_PREDICTIONS_DIRNAME
    logical = {
        OOF_PREDICTIONS_DIRNAME: {
            "kind": "partitioned_parquet_dataset",
            "partition_scheme": "one part per direction",
            "parts": sorted(p.name for p in predictions_dir.glob("*.parquet"))
            if predictions_dir.is_dir() else [],
            "dataset_sha256": sha256_path(predictions_dir) if predictions_dir.is_dir() else None,
            "read_as": "pandas.read_parquet(<analysis_root>/oof_predictions.parquet)",
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
        "logical_datasets": logical,
        "stages": {
            stage: read_stage_marker(analysis_id, stage, output_root) is not None
            for stage in STAGES
        },
        "earth_engine_used": False,
        "p_values_produced": False,
    }


def run_summarize_stage(analysis_id: str, experiment_ids: Sequence[str],
                        output_root: Optional[Path] = None) -> dict[str, Any]:
    root = analysis_root(analysis_id, output_root)
    metrics_path = root / "repeat_metrics.csv"
    if not metrics_path.is_file():
        raise FewShotRecoveryError(
            f"Stage 'summarize' requires {metrics_path} from stage 'fit'."
        )
    repeat_metrics = pd.read_csv(metrics_path)

    curve_rows = build_recovery_curve(repeat_metrics)
    _write_namespaced(root, "recovery_curve.csv",
                      _csv_document(RECOVERY_CURVE_COLUMNS, curve_rows))
    curve = pd.DataFrame(curve_rows, columns=RECOVERY_CURVE_COLUMNS)

    fit_marker = read_stage_marker(analysis_id, "fit", output_root) or {}
    fit_accounting = fit_marker.get("fit_accounting", {})

    summary = build_summary(analysis_id, experiment_ids, curve, repeat_metrics,
                            fit_accounting, output_root)
    _write_namespaced(root, "summary.json", _json_document(summary))
    _write_namespaced(root, "report.md", render_report(analysis_id, curve, summary))

    manifest = build_manifest(analysis_id, output_root)
    _write_namespaced(root, "manifest.json", _json_document(manifest))

    marker = write_stage_marker(analysis_id, "summarize", output_root, extra={
        "recovery_curve_rows": len(curve_rows),
    })
    return {
        "stage": "summarize", "recovery_curve_rows": len(curve_rows),
        "manifest_files": len(manifest["files"]), "marker": marker,
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
    `resume` reuses only complete, hash-bound stages and direction partitions.
    """
    stages = validate_stage_range(from_stage, to_stage)
    experiment_ids = resolve_experiments(experiments)

    input_inventory = build_frozen_input_inventory(experiment_ids, experiments_root)
    hash_gate = assert_canonical_step8a_hashes(input_inventory, strict=True)
    scientific_config = build_scientific_config(experiment_ids, input_inventory)
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
            "directed_pairs": [list(pair) for pair in directed_pairs(experiment_ids)],
            "directed_pair_count": len(directed_pairs(experiment_ids)),
            "stages_requested": stages,
            "stages_executed": [],
            "population": POPULATION,
            "budgets": list(BUDGETS),
            "n_repeats": N_REPEATS,
            "expected_fit_accounting": expected_unique_fit_count(
                len(directed_pairs(experiment_ids)), len(experiment_ids)),
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
        raise FewShotRecoveryError(
            f"Analysis namespace already exists: {root}. Pass --resume to verify and "
            "reuse it, or --force to quarantine it (nothing is ever deleted)."
        )

    for stage in stages:
        for prerequisite in STAGE_REQUIRES[stage]:
            if prerequisite in stages:
                continue
            state = verify_stage_complete(analysis_id, prerequisite, output_root)
            if not state["complete"]:
                raise FewShotRecoveryError(
                    f"Stage {stage!r} requires a complete {prerequisite!r} stage: "
                    f"{state['reason']}."
                )

    context = build_target_context(experiment_ids, experiments_root, frames)

    executed: list[dict[str, Any]] = []
    for stage in stages:
        if resume and stage != "fit":
            state = verify_stage_complete(analysis_id, stage, output_root)
            if state["complete"]:
                executed.append({"stage": stage, "reused": True})
                continue
        if stage == "plan":
            executed.append(run_plan_stage(
                analysis_id, experiment_ids, context, input_inventory,
                scientific_config, output_root))
        elif stage == "fit":
            executed.append(run_fit_stage(
                analysis_id, experiment_ids, context, input_inventory, output_root,
                resume=resume))
        elif stage == "summarize":
            executed.append(run_summarize_stage(analysis_id, experiment_ids, output_root))

    return {
        "ran": True,
        "dry_run": False,
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "experiments": experiment_ids,
        "directed_pair_count": len(directed_pairs(experiment_ids)),
        "stages_executed": [entry["stage"] for entry in executed],
        "stage_results": executed,
        "output_namespace": str(root),
        "quarantined_previous_namespace": quarantined,
        "earth_engine_used": False,
    }
