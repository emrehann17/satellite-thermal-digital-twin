"""Frozen Step8 robustness analysis at predefined 10-cell and 20-cell blocks.

This module reuses the Step8B feature/model pipeline and Step8C metric helper,
but writes only below outputs/robustness/step8_large_block/.  It never writes
to the original Step8A-E namespaces.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from core.config import (
    STEP8B_N_SPLITS, STEP8B_RANDOM_SEED, STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
    STEP8C_CI_LOWER, STEP8C_CI_UPPER, STEP8C_N_BOOTSTRAP, STEP8C_RANDOM_SEED,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from src.step8b_train_baseline_vs_thermal_model import (
    BASELINE_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, THERMAL_FEATURES,
    THERMAL_MODEL_FEATURES, build_classifier, build_pipeline, compute_binary_metrics,
)
from src.step8c_spatial_block_bootstrap_uncertainty import compute_metrics as compute_paired_metrics

log, log_file = setup_logger("step8_large_block_robustness")

ANALYSIS_SCHEMA_VERSION = "step8.large_block_robustness.v1"
EXPECTED_EXPERIMENTS = ("manavgat_2021", "bejis_2022")
EXPECTED_BLOCK_SIZES = (10, 20)
NOMINAL_SCALES = {10: "approximately_5_km", 20: "approximately_10_km"}
PRIMARY_POPULATION = "burnable_tree_shrub_grass"
MODEL_NAME = "random_forest"
MIN_VALID_BOOTSTRAP = 900
OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "robustness" / "step8_large_block"
    / "manavgat_2021__bejis_2022"
)
PROTECTED_RELATIVE_PATHS = (
    "step8a/step8a_500m_modeling_dataset.parquet",
    "step8b/step8b_predictions.parquet",
    "step8b/step8b_model_comparison_metrics.json",
    "step8c/step8c_bootstrap_metrics.json",
    "step8e/final_step8_report.json",
)


class Step8RobustnessError(SystemExit):
    """Fail-fast error for the frozen large-block robustness analysis."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_analysis_request(experiments: list[str] | tuple[str, ...], block_sizes: list[int] | tuple[int, ...]) -> None:
    if tuple(experiments) != EXPECTED_EXPERIMENTS:
        raise Step8RobustnessError(
            f"This frozen analysis requires experiments in exact order {list(EXPECTED_EXPERIMENTS)}; got {list(experiments)}."
        )
    if tuple(int(value) for value in block_sizes) != EXPECTED_BLOCK_SIZES:
        raise Step8RobustnessError(
            f"This analysis version requires block sizes in exact order {list(EXPECTED_BLOCK_SIZES)}; got {list(block_sizes)}."
        )


def experiment_step8_root(experiment_id: str) -> Path:
    return PROJECT_ROOT / "outputs" / "experiments" / experiment_id


def protected_paths(experiments: tuple[str, ...] = EXPECTED_EXPERIMENTS) -> dict[str, Path]:
    return {
        f"{experiment_id}/{relative}": experiment_step8_root(experiment_id) / relative
        for experiment_id in experiments for relative in PROTECTED_RELATIVE_PATHS
    }


def hash_protected_inputs(paths: dict[str, Path] | None = None) -> dict[str, dict[str, Any]]:
    paths = paths or protected_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise Step8RobustnessError("Missing frozen original Step8 input/reference file(s): " + ", ".join(missing))
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def assert_protected_hashes_unchanged(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> None:
    changed = sorted(name for name, value in before.items() if after.get(name, {}).get("sha256") != value["sha256"])
    if changed:
        raise Step8RobustnessError("Original Step8 protection failed; SHA-256 changed: " + ", ".join(changed))


def _package_versions() -> dict[str, str]:
    names = {"numpy": "numpy", "pandas": "pandas", "scikit_learn": "scikit-learn"}
    return {key: importlib.metadata.version(package) for key, package in names.items()}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def original_reference_payload(experiment_id: str) -> dict[str, Any]:
    root = experiment_step8_root(experiment_id)
    metrics = json.loads((root / "step8b" / "step8b_model_comparison_metrics.json").read_text())
    bootstrap = json.loads((root / "step8c" / "step8c_bootstrap_metrics.json").read_text())
    spatial = metrics.get("spatial_cv_config", {})
    if spatial.get("spatial_block_size_cells") != STEP8B_SPATIAL_BLOCK_SIZE_CELLS or spatial.get("spatial_block_size_cells") != 2:
        raise Step8RobustnessError(f"{experiment_id}: frozen original Step8 block size is not proven to be 2 cells.")
    if (
        spatial.get("method") != "StratifiedGroupKFold"
        or spatial.get("n_splits_requested") != STEP8B_N_SPLITS
        or spatial.get("random_state") != STEP8B_RANDOM_SEED
        or spatial.get("random_split_used") is not False
    ):
        raise Step8RobustnessError(f"{experiment_id}: frozen original Step8 CV provenance disagrees with runtime configuration.")
    feature_sets = metrics.get("feature_sets", {})
    if (
        metrics.get("model") != MODEL_NAME
        or feature_sets.get("baseline") != BASELINE_FEATURES
        or feature_sets.get("thermal_additional") != THERMAL_FEATURES
        or feature_sets.get("thermal_model_full") != THERMAL_MODEL_FEATURES
    ):
        raise Step8RobustnessError(f"{experiment_id}: frozen Step8B model/feature provenance disagrees with runtime configuration.")
    if (
        bootstrap.get("n_bootstrap_requested") != STEP8C_N_BOOTSTRAP
        or bootstrap.get("random_seed") != STEP8C_RANDOM_SEED
    ):
        raise Step8RobustnessError(f"{experiment_id}: frozen Step8C bootstrap provenance disagrees with runtime configuration.")
    point = metrics.get("population_metrics", {}).get(PRIMARY_POPULATION)
    ci = bootstrap.get("bootstrap_ci_by_population", {}).get(PRIMARY_POPULATION)
    if not point or not ci:
        raise Step8RobustnessError(f"{experiment_id}: primary-population Step8B/Step8C reference is missing.")
    return {"metrics": metrics, "bootstrap": bootstrap, "point": point, "ci": ci}


def scientific_configuration(protected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    references = {experiment: original_reference_payload(experiment) for experiment in EXPECTED_EXPERIMENTS}
    classifier = build_classifier(MODEL_NAME, STEP8B_RANDOM_SEED)
    model_params = classifier.get_params(deep=False)
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment_ids": list(EXPECTED_EXPERIMENTS),
        "primary_population": PRIMARY_POPULATION,
        "predefined_block_sizes_cells": list(EXPECTED_BLOCK_SIZES),
        "nominal_scales": [NOMINAL_SCALES[size] for size in EXPECTED_BLOCK_SIZES],
        "fixed_grid_origin": {"row_500m": 0, "col_500m": 0},
        "block_construction": "block_row=floor(row_500m/block_size_cells); block_col=floor(col_500m/block_size_cells); large_block_id=b{size}_r{block_row}_c{block_col}",
        "original_small_block_reference_size_cells": STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
        "baseline_features": list(BASELINE_FEATURES),
        "thermal_additional_features": list(THERMAL_FEATURES),
        "thermal_model_features": list(THERMAL_MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "model_class": type(classifier).__name__,
        "model_name": MODEL_NAME,
        "model_hyperparameters": model_params,
        "preprocessing": {
            "numeric": "SimpleImputer(strategy=median)",
            "categorical": "SimpleImputer(strategy=most_frequent) + OneHotEncoder(handle_unknown=ignore)",
            "fit_scope": "inside each training fold only",
        },
        "cv": {"class": "StratifiedGroupKFold", "n_splits": STEP8B_N_SPLITS, "shuffle": True, "random_state": STEP8B_RANDOM_SEED, "dynamic_fold_reduction": False},
        "bootstrap": {"requested_replicates": STEP8C_N_BOOTSTRAP, "random_state": STEP8C_RANDOM_SEED, "ci_method": f"{STEP8C_CI_LOWER}/{STEP8C_CI_UPPER} percentile", "unit": "corresponding large spatial block", "paired": True, "minimum_valid_replicates": MIN_VALID_BOOTSTRAP},
        "primary_estimands": ["delta_roc_auc=thermal_oof_roc_auc-baseline_oof_roc_auc", "delta_pr_auc=thermal_oof_pr_auc-baseline_oof_pr_auc"],
        "interpretation_rules": {
            "positive": "bootstrap_supported_positive iff percentile CI entirely above zero",
            "uncertain": "uncertain_interval_includes_zero iff percentile CI includes zero",
            "negative": "bootstrap_supported_negative iff percentile CI entirely below zero",
            "joint": "supported_on_both_metrics iff both delta intervals entirely above zero; partial_metric_support iff exactly one is above zero; otherwise not_supported",
            "overall": "claim robustness across both predefined scales and both regions only if all four conditions support both metrics",
        },
        "protected_original_inputs": protected,
        "package_versions": _package_versions(),
        "verified_original_provenance": {
            experiment: {
                "spatial_block_size_cells": references[experiment]["metrics"]["spatial_cv_config"]["spatial_block_size_cells"],
                "n_splits_requested": references[experiment]["metrics"]["spatial_cv_config"]["n_splits_requested"],
                "model": references[experiment]["metrics"]["model"],
                "bootstrap_requested": references[experiment]["bootstrap"]["n_bootstrap_requested"],
                "bootstrap_seed": references[experiment]["bootstrap"]["random_seed"],
            } for experiment in EXPECTED_EXPERIMENTS
        },
        "prohibited_actions": ["post_hoc_block_selection", "feature_selection", "hyperparameter_tuning", "random_row_split", "probability_calibration", "prediction_inversion"],
    }


def build_manifest(protected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    content = scientific_configuration(protected)
    analysis_id = sha256_bytes(canonical_json(content).encode("utf-8"))
    return {"analysis_id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(), "scientific_configuration": content}


def _preregistration_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Step8 Large-Spatial-Block Robustness Preregistration (IMMUTABLE)",
        "",
        f"- analysis_id: {manifest['analysis_id']}",
        f"- created_at: {manifest['created_at']}",
        f"- experiments: {list(EXPECTED_EXPERIMENTS)}",
        f"- predefined block sizes: {list(EXPECTED_BLOCK_SIZES)}",
        f"- primary population: {PRIMARY_POPULATION}",
        "",
        "Only spatial grouping scale changes. No favorable scale will be selected post hoc.",
    ]
    return "\n".join(lines) + "\n"


def validate_or_write_manifest(output_root: Path, protected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = output_root / "step8_large_block_preregistration.json"
    md_path = output_root / "step8_large_block_preregistration.md"
    expected_content = scientific_configuration(protected)
    expected_id = sha256_bytes(canonical_json(expected_content).encode("utf-8"))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("analysis_id") != expected_id or existing.get("scientific_configuration") != expected_content:
            raise Step8RobustnessError("Existing immutable large-block preregistration disagrees with runtime scientific configuration.")
        if not md_path.is_file() or md_path.read_text(encoding="utf-8") != _preregistration_markdown(existing):
            raise Step8RobustnessError("Existing immutable Markdown preregistration is missing or changed.")
        return existing
    if md_path.exists():
        raise Step8RobustnessError("Immutable Markdown preregistration exists without its JSON manifest.")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(protected)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(_preregistration_markdown(manifest), encoding="utf-8")
    return manifest


def assert_downstream_outputs_writable(output_root: Path, force: bool) -> None:
    """Fail before fitting if any non-preregistration robustness output exists."""
    if force or not output_root.exists():
        return
    immutable = {
        output_root / "step8_large_block_preregistration.json",
        output_root / "step8_large_block_preregistration.md",
    }
    present = sorted(
        str(candidate) for candidate in output_root.rglob("*")
        if candidate.is_file() and candidate not in immutable
    )
    if present:
        raise Step8RobustnessError(
            "Downstream robustness outputs already exist; use --force to overwrite only those outputs: "
            + ", ".join(present)
        )


def validate_canonical_grid(df: pd.DataFrame) -> None:
    required = {"row_500m", "col_500m", "cell_id"}
    if not required.issubset(df.columns):
        raise Step8RobustnessError(f"Canonical large-block reconstruction is impossible; missing columns: {sorted(required-set(df.columns))}")
    if df[["row_500m", "col_500m"]].isna().any().any():
        raise Step8RobustnessError("Canonical row/column indices contain missing values.")
    row = df["row_500m"].astype(int)
    col = df["col_500m"].astype(int)
    if (row < 0).any() or (col < 0).any():
        raise Step8RobustnessError("Canonical row/column indices must use the non-negative fixed Step8A origin.")
    expected_ids = "r" + row.astype(str) + "_c" + col.astype(str)
    if not np.array_equal(expected_ids.to_numpy(), df["cell_id"].astype(str).to_numpy()):
        raise Step8RobustnessError("cell_id does not match canonical row_500m/col_500m indices.")


def assign_large_blocks(df: pd.DataFrame, block_size_cells: int) -> pd.DataFrame:
    """Label-independent fixed-origin block assignment, before population filtering."""
    if int(block_size_cells) not in EXPECTED_BLOCK_SIZES:
        raise Step8RobustnessError(f"Unsupported frozen block size: {block_size_cells}")
    validate_canonical_grid(df)
    result = df.copy()
    block_row = result["row_500m"].astype(np.int64) // int(block_size_cells)
    block_col = result["col_500m"].astype(np.int64) // int(block_size_cells)
    result["large_block_row"] = block_row
    result["large_block_col"] = block_col
    result["large_block_id"] = "b" + str(int(block_size_cells)) + "_r" + block_row.astype(str) + "_c" + block_col.astype(str)
    return result


def make_strict_spatial_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = STEP8B_N_SPLITS, random_state: int = STEP8B_RANDOM_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    y, groups = np.asarray(y), np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        raise Step8RobustnessError(f"Cannot construct frozen {n_splits}-fold CV: too few large blocks.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(np.zeros(len(y)), y, groups=groups))
    coverage = np.zeros(len(y), dtype=int)
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train_blocks, test_blocks = set(groups[train_idx]), set(groups[test_idx])
        if train_blocks.intersection(test_blocks):
            raise Step8RobustnessError(f"Fold {fold_id}: a large block appears in both train and test.")
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            raise Step8RobustnessError(f"Fold {fold_id}: train or test is single-class; frozen fold count will not be reduced.")
        coverage[test_idx] += 1
    if not np.all(coverage == 1):
        raise Step8RobustnessError("Every modeling row must occur in exactly one OOF test fold.")
    return folds


def block_and_fold_qa(df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped = df.groupby("large_block_id", sort=True)[TARGET_COLUMN].agg(["size", "sum"])
    positives = grouped["sum"]
    fold_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        y_train = df.iloc[train_idx][TARGET_COLUMN].astype(int)
        y_test = df.iloc[test_idx][TARGET_COLUMN].astype(int)
        train_blocks = set(df.iloc[train_idx]["large_block_id"])
        test_blocks = set(df.iloc[test_idx]["large_block_id"])
        fold_rows.append({
            "fold_id": fold_id, "train_rows": len(train_idx), "test_rows": len(test_idx),
            "train_positives": int(y_train.sum()), "test_positives": int(y_test.sum()),
            "train_blocks": len(train_blocks), "test_blocks": len(test_blocks),
            "block_overlap": len(train_blocks.intersection(test_blocks)),
        })
    audit = {
        "total_modeling_rows": int(len(df)), "positive_rows": int(df[TARGET_COLUMN].sum()),
        "negative_rows": int((df[TARGET_COLUMN] == 0).sum()), "total_large_blocks": int(len(grouped)),
        "blocks_containing_positives": int((positives > 0).sum()),
        "blocks_containing_negatives": int((positives < grouped["size"]).sum()),
        "mixed_class_blocks": int(((positives > 0) & (positives < grouped["size"])).sum()),
        "rows_per_block": {"min": int(grouped["size"].min()), "median": float(grouped["size"].median()), "max": int(grouped["size"].max())},
        "positives_per_block": {"min": int(positives.min()), "median": float(positives.median()), "max": int(positives.max())},
        "number_of_cv_folds": len(folds), "folds": fold_rows,
    }
    assignments = []
    for fold_id, (_, test_idx) in enumerate(folds):
        test = df.iloc[test_idx]
        for block_id, subset in test.groupby("large_block_id", sort=True):
            assignments.append({"fold_id": fold_id, "large_block_id": block_id, "test_rows": len(subset), "test_positives": int(subset[TARGET_COLUMN].sum())})
    return audit, assignments


def run_oof_condition(df_all: pd.DataFrame, experiment_id: str, block_size_cells: int, analysis_id: str, pipeline_builder=build_pipeline) -> dict[str, Any]:
    assigned = assign_large_blocks(df_all, block_size_cells)  # before valid/population filtering
    required_features = set(THERMAL_MODEL_FEATURES) | {TARGET_COLUMN, "valid_for_modeling", PRIMARY_POPULATION}
    missing = sorted(required_features - set(assigned.columns))
    if missing:
        raise Step8RobustnessError(f"{experiment_id}: feature/input schema differs from frozen Step8B: {missing}")
    modeling = assigned.loc[assigned["valid_for_modeling"].astype(bool) & assigned[PRIMARY_POPULATION].astype(bool)].copy().reset_index(drop=True)
    if modeling.empty:
        raise Step8RobustnessError(f"{experiment_id}: primary population is empty.")
    y = modeling[TARGET_COLUMN].astype(int).to_numpy()
    groups = modeling["large_block_id"].astype(str).to_numpy()
    folds = make_strict_spatial_folds(y, groups)
    baseline = np.full(len(modeling), np.nan)
    thermal = np.full(len(modeling), np.nan)
    fold_ids = np.full(len(modeling), -1, dtype=int)
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train, test = modeling.iloc[train_idx], modeling.iloc[test_idx]
        baseline_pipeline = pipeline_builder(BASELINE_FEATURES, MODEL_NAME, STEP8B_RANDOM_SEED)
        thermal_pipeline = pipeline_builder(THERMAL_MODEL_FEATURES, MODEL_NAME, STEP8B_RANDOM_SEED)
        baseline_pipeline.fit(train[BASELINE_FEATURES], y[train_idx])
        thermal_pipeline.fit(train[THERMAL_MODEL_FEATURES], y[train_idx])
        baseline[test_idx] = baseline_pipeline.predict_proba(test[BASELINE_FEATURES])[:, 1]
        thermal[test_idx] = thermal_pipeline.predict_proba(test[THERMAL_MODEL_FEATURES])[:, 1]
        fold_ids[test_idx] = fold_id
    if np.isnan(baseline).any() or np.isnan(thermal).any() or (fold_ids < 0).any():
        raise Step8RobustnessError("OOF completeness check failed.")
    base_metrics, thermal_metrics = compute_binary_metrics(y, baseline), compute_binary_metrics(y, thermal)
    metrics = {
        "baseline_roc_auc": base_metrics["roc_auc"], "thermal_roc_auc": thermal_metrics["roc_auc"],
        "delta_roc_auc": thermal_metrics["roc_auc"] - base_metrics["roc_auc"],
        "baseline_pr_auc": base_metrics["pr_auc"], "thermal_pr_auc": thermal_metrics["pr_auc"],
        "delta_pr_auc": thermal_metrics["pr_auc"] - base_metrics["pr_auc"],
    }
    predictions = pd.DataFrame({
        "analysis_id": analysis_id, "experiment_id": experiment_id,
        "block_size_cells": int(block_size_cells), "nominal_scale": NOMINAL_SCALES[int(block_size_cells)],
        "cell_id": modeling["cell_id"].to_numpy(), "row_500m": modeling["row_500m"].to_numpy(),
        "col_500m": modeling["col_500m"].to_numpy(), "large_block_id": groups,
        "spatial_block_id": groups, "burned": y, "fold_id": fold_ids,
        "y_prob_baseline": baseline, "y_prob_thermal": thermal,
    })
    audit, assignments = block_and_fold_qa(modeling, folds)
    return {"metrics": metrics, "predictions": predictions, "block_audit": audit, "fold_assignments": assignments}


def paired_large_block_bootstrap(predictions: pd.DataFrame, n_replicates: int = STEP8C_N_BOOTSTRAP, seed: int = STEP8C_RANDOM_SEED) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"large_block_id", "burned", "y_prob_baseline", "y_prob_thermal"}
    if not required.issubset(predictions.columns):
        raise Step8RobustnessError(f"Bootstrap predictions missing: {sorted(required-set(predictions.columns))}")
    blocks = predictions["large_block_id"].drop_duplicates().to_numpy()
    indices = {block: np.flatnonzero(predictions["large_block_id"].to_numpy() == block) for block in blocks}
    rng = np.random.default_rng(seed)
    rows, successful = [], []
    for replicate in range(n_replicates):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        row_idx = np.concatenate([indices[block] for block in sampled])
        y = predictions.iloc[row_idx]["burned"].to_numpy(dtype=int)
        metrics = compute_paired_metrics(y, predictions.iloc[row_idx]["y_prob_baseline"].to_numpy(), predictions.iloc[row_idx]["y_prob_thermal"].to_numpy())
        base = {"replicate": replicate, "sampled_block_ids_sha256": sha256_bytes("|".join(map(str, sampled)).encode())}
        if metrics is None:
            rows.append({**base, "valid": False, "invalid_reason": "single_class"})
        else:
            kept = {name: metrics[name] for name in ("auc_baseline", "auc_thermal", "delta_auc", "pr_auc_baseline", "pr_auc_thermal", "delta_pr_auc")}
            successful.append(kept)
            rows.append({**base, "valid": True, "invalid_reason": None, **kept})
    valid = len(successful)
    summary = {"requested_replicates": n_replicates, "valid_replicates": valid, "invalid_single_class_replicates": n_replicates-valid, "bootstrap_stability": "stable" if valid >= MIN_VALID_BOOTSTRAP else "bootstrap_unstable", "random_seed": seed, "ci_method": f"{STEP8C_CI_LOWER}/{STEP8C_CI_UPPER} percentile", "series": {}}
    for name in ("auc_baseline", "auc_thermal", "delta_auc", "pr_auc_baseline", "pr_auc_thermal", "delta_pr_auc"):
        values = np.array([item[name] for item in successful], dtype=float)
        summary["series"][name] = {"mean": float(np.mean(values)) if valid else None, "median": float(np.median(values)) if valid else None, "ci_2_5": float(np.percentile(values, STEP8C_CI_LOWER)) if valid else None, "ci_97_5": float(np.percentile(values, STEP8C_CI_UPPER)) if valid else None}
    return summary, pd.DataFrame(rows)


def classify_interval(ci: list[float | None] | tuple[float | None, float | None]) -> str:
    low, high = ci
    if low is None or high is None:
        return "unavailable"
    if low > 0:
        return "bootstrap_supported_positive"
    if high < 0:
        return "bootstrap_supported_negative"
    return "uncertain_interval_includes_zero"


def classify_joint(roc_status: str, pr_status: str) -> str:
    supported = [roc_status == "bootstrap_supported_positive", pr_status == "bootstrap_supported_positive"]
    if all(supported):
        return "supported_on_both_metrics"
    if any(supported):
        return "partial_metric_support"
    return "not_supported"


def _condition_output_dir(experiment_id: str, block_size_cells: int, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / experiment_id / f"block_{block_size_cells}_cells"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_condition_outputs(output_dir: Path, analysis_id: str, experiment_id: str, block_size: int, result: dict[str, Any], bootstrap: dict[str, Any], replicates: pd.DataFrame, force: bool) -> dict[str, Any]:
    expected = ["block_assignment_audit.json", "fold_assignment.csv", "step8b_large_block_metrics.json", "step8b_large_block_metrics.csv", "step8b_large_block_predictions.parquet", "step8c_large_block_bootstrap_summary.json", "step8c_large_block_bootstrap_summary.csv", "step8c_large_block_bootstrap_replicates.parquet"]
    present = [name for name in expected if (output_dir / name).exists()]
    if present and not force:
        raise Step8RobustnessError(f"Downstream robustness outputs exist in {output_dir}; use --force: {present}")
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {"analysis_id": analysis_id, "experiment_id": experiment_id, "block_size_cells": block_size, "nominal_scale": NOMINAL_SCALES[block_size]}
    audit = {**common, **result["block_audit"]}
    (output_dir / "block_assignment_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    _write_csv(output_dir / "fold_assignment.csv", [{**common, **row} for row in result["fold_assignments"]])
    classifier = build_classifier(MODEL_NAME, STEP8B_RANDOM_SEED)
    metrics = {**common, **result["metrics"], "primary_population": PRIMARY_POPULATION, "feature_sets": {"baseline": BASELINE_FEATURES, "thermal_additional": THERMAL_FEATURES, "thermal_model_full": THERMAL_MODEL_FEATURES}, "model_class": type(classifier).__name__, "model_hyperparameters": classifier.get_params(deep=False), "cv": {"class": "StratifiedGroupKFold", "n_splits": STEP8B_N_SPLITS, "random_state": STEP8B_RANDOM_SEED}}
    (output_dir / "step8b_large_block_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    _write_csv(output_dir / "step8b_large_block_metrics.csv", [{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))}])
    result["predictions"].to_parquet(output_dir / "step8b_large_block_predictions.parquet", index=False)
    boot_payload = {**common, **bootstrap}
    (output_dir / "step8c_large_block_bootstrap_summary.json").write_text(json.dumps(boot_payload, indent=2) + "\n")
    boot_rows = [{**common, "series": name, **values, "requested_replicates": bootstrap["requested_replicates"], "valid_replicates": bootstrap["valid_replicates"], "invalid_single_class_replicates": bootstrap["invalid_single_class_replicates"], "bootstrap_stability": bootstrap["bootstrap_stability"]} for name, values in bootstrap["series"].items()]
    _write_csv(output_dir / "step8c_large_block_bootstrap_summary.csv", boot_rows)
    for key, value in common.items(): replicates[key] = value
    ordered = list(common) + [column for column in replicates.columns if column not in common]
    replicates[ordered].to_parquet(output_dir / "step8c_large_block_bootstrap_replicates.parquet", index=False)
    return {"metrics": metrics, "bootstrap": boot_payload, "audit": audit}


def reference_comparison_row(experiment_id: str, analysis_id: str) -> dict[str, Any]:
    reference = original_reference_payload(experiment_id)
    point, ci = reference["point"], reference["ci"]
    roc_ci, pr_ci = ci["delta_auc_ci95"], ci["delta_pr_auc_ci95"]
    roc_status, pr_status = classify_interval(roc_ci), classify_interval(pr_ci)
    valid = ci["n_bootstrap_successful"]
    stability = "stable" if valid >= MIN_VALID_BOOTSTRAP else "bootstrap_unstable"
    status = classify_joint(roc_status, pr_status) if stability == "stable" else "bootstrap_unstable"
    return {"analysis_id": analysis_id, "experiment": experiment_id, "nominal_block_scale": "approximately_1_km", "block_size_cells": 2, "source_type": "frozen_original_small_block", "baseline_roc_auc": point["overall_baseline"]["roc_auc"], "thermal_roc_auc": point["overall_thermal"]["roc_auc"], "delta_roc_auc": point["delta_auc"], "delta_roc_auc_ci_low": roc_ci[0], "delta_roc_auc_ci_high": roc_ci[1], "baseline_pr_auc": point["overall_baseline"]["pr_auc"], "thermal_pr_auc": point["overall_thermal"]["pr_auc"], "delta_pr_auc": point["delta_pr_auc"], "delta_pr_auc_ci_low": pr_ci[0], "delta_pr_auc_ci_high": pr_ci[1], "valid_bootstrap_replicates": valid, "bootstrap_stability": stability, "roc_support": roc_status, "pr_support": pr_status, "robustness_status": status}


def large_comparison_row(condition: dict[str, Any], analysis_id: str) -> dict[str, Any]:
    metrics, bootstrap = condition["metrics"], condition["bootstrap"]
    roc = bootstrap["series"]["delta_auc"]; pr = bootstrap["series"]["delta_pr_auc"]
    roc_status = classify_interval([roc["ci_2_5"], roc["ci_97_5"]]); pr_status = classify_interval([pr["ci_2_5"], pr["ci_97_5"]])
    stability = bootstrap["bootstrap_stability"]
    status = classify_joint(roc_status, pr_status) if stability == "stable" else "bootstrap_unstable"
    return {"analysis_id": analysis_id, "experiment": metrics["experiment_id"], "nominal_block_scale": metrics["nominal_scale"], "block_size_cells": metrics["block_size_cells"], "source_type": "new_large_block_robustness", "baseline_roc_auc": metrics["baseline_roc_auc"], "thermal_roc_auc": metrics["thermal_roc_auc"], "delta_roc_auc": metrics["delta_roc_auc"], "delta_roc_auc_ci_low": roc["ci_2_5"], "delta_roc_auc_ci_high": roc["ci_97_5"], "baseline_pr_auc": metrics["baseline_pr_auc"], "thermal_pr_auc": metrics["thermal_pr_auc"], "delta_pr_auc": metrics["delta_pr_auc"], "delta_pr_auc_ci_low": pr["ci_2_5"], "delta_pr_auc_ci_high": pr["ci_97_5"], "valid_bootstrap_replicates": bootstrap["valid_replicates"], "bootstrap_stability": stability, "roc_support": roc_status, "pr_support": pr_status, "robustness_status": status}


def write_final_report(output_root: Path, analysis_id: str, comparison: list[dict[str, Any]], conditions: list[dict[str, Any]], protected_status: str) -> dict[str, Any]:
    large = [row for row in comparison if row["source_type"] == "new_large_block_robustness"]
    all_supported = len(large) == 4 and all(row["robustness_status"] == "supported_on_both_metrics" for row in large) and all(c["bootstrap"]["bootstrap_stability"] == "stable" for c in conditions)
    failures = [{"experiment": row["experiment"], "block_size_cells": row["block_size_cells"], "roc_support": row["roc_support"], "pr_support": row["pr_support"], "bootstrap_stability": row["bootstrap_stability"]} for row in large if row["robustness_status"] != "supported_on_both_metrics"]
    statement = "thermal contribution remained bootstrap-supported across both predefined large-block scales in both wildfire regions" if all_supported else "The result was sensitive to spatial block scale or metric support; unsupported region-scale-metric conditions are listed explicitly."
    report = {"analysis_id": analysis_id, "report_schema_version": ANALYSIS_SCHEMA_VERSION, "original_small_block_reference": [row for row in comparison if row["source_type"] == "frozen_original_small_block"], "new_large_block_conditions": large, "block_fold_qa": [c["audit"] for c in conditions], "overall_predefined_scale_robustness": {"all_four_conditions_supported_on_both_metrics": all_supported, "statement": statement, "conditions_losing_support": failures}, "protected_original_step8_hash_check": protected_status, "claim_boundaries": ["spatial autocorrelation was not claimed eliminated", "no causal thermal effects", "not operational wildfire prediction", "no statistical significance or p-values", "no proof that residual spatial dependence is absent", "not successful cross-region transfer", "no best block size was selected"]}
    (output_root / "step8_large_block_final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    primary_rows = [[r["experiment"], r["block_size_cells"], r["nominal_block_scale"], f"{r['delta_roc_auc']:.6f}", f"[{r['delta_roc_auc_ci_low']:.6f}, {r['delta_roc_auc_ci_high']:.6f}]", r["roc_support"], f"{r['delta_pr_auc']:.6f}", f"[{r['delta_pr_auc_ci_low']:.6f}, {r['delta_pr_auc_ci_high']:.6f}]", r["pr_support"], r["robustness_status"]] for r in large]
    qa_rows = [[a["experiment_id"], a["block_size_cells"], a["total_large_blocks"], a["blocks_containing_positives"], a["number_of_cv_folds"], min(f["test_positives"] for f in a["folds"]), next(c["bootstrap"]["valid_replicates"] for c in conditions if c["metrics"]["experiment_id"] == a["experiment_id"] and c["metrics"]["block_size_cells"] == a["block_size_cells"])] for a in report["block_fold_qa"]]
    reference_rows = [[r["experiment"], r["block_size_cells"], r["delta_roc_auc"], f"[{r['delta_roc_auc_ci_low']:.6f}, {r['delta_roc_auc_ci_high']:.6f}]", r["delta_pr_auc"], f"[{r['delta_pr_auc_ci_low']:.6f}, {r['delta_pr_auc_ci_high']:.6f}]", r["valid_bootstrap_replicates"], r["robustness_status"]] for r in report["original_small_block_reference"]]
    def table(headers, rows): return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    lines = ["# Step8 Large-Spatial-Block Robustness Report", "", f"- analysis_id: `{analysis_id}`", f"- protected original Step8 hash check: **{protected_status}**", "", "## New predefined large-block conditions", ""] + table(["experiment", "block cells", "nominal scale", "delta ROC-AUC", "ROC CI", "ROC support", "delta PR-AUC", "PR CI", "PR support", "joint status"], primary_rows) + ["", "## Block and fold feasibility", ""] + table(["experiment", "block cells", "total blocks", "positive blocks", "folds", "min test positives", "valid bootstrap"], qa_rows) + ["", "## Frozen original 2-cell reference", ""] + table(["experiment", "block cells", "delta ROC-AUC", "ROC CI", "delta PR-AUC", "PR CI", "valid bootstrap", "status"], reference_rows) + ["", "## Overall predefined-scale interpretation", "", statement]
    if failures: lines += ["", "Conditions losing support:", ""] + [f"- {item['experiment']} block={item['block_size_cells']}: ROC={item['roc_support']}; PR={item['pr_support']}; bootstrap={item['bootstrap_stability']}" for item in failures]
    lines += ["", "## Claim boundaries", ""] + [f"- {item}" for item in report["claim_boundaries"]]
    (output_root / "step8_large_block_final_report.md").write_text("\n".join(lines) + "\n")
    return report


def dry_run_plan(experiments: list[str], block_sizes: list[int], output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    validate_analysis_request(experiments, block_sizes)
    paths = protected_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise Step8RobustnessError("Dry-run missing frozen input(s): " + ", ".join(missing))
    references = {experiment: original_reference_payload(experiment) for experiment in EXPECTED_EXPERIMENTS}
    classifier = build_classifier(MODEL_NAME, STEP8B_RANDOM_SEED)
    return {"ran": False, "dry_run": True, "experiments": list(EXPECTED_EXPERIMENTS), "block_sizes_cells": list(EXPECTED_BLOCK_SIZES), "nominal_scales": [NOMINAL_SCALES[size] for size in EXPECTED_BLOCK_SIZES], "resolved_step8a_inputs": [str(experiment_step8_root(e) / PROTECTED_RELATIVE_PATHS[0]) for e in EXPECTED_EXPERIMENTS], "protected_original_paths": [str(path) for path in paths.values()], "output_namespace": str(output_root), "features": {"baseline": BASELINE_FEATURES, "thermal": THERMAL_MODEL_FEATURES}, "model": {"class": type(classifier).__name__, "hyperparameters": classifier.get_params(deep=False)}, "cv": {"class": "StratifiedGroupKFold", "folds": STEP8B_N_SPLITS, "seed": STEP8B_RANDOM_SEED, "dynamic_fold_reduction": False}, "block_reconstruction_source": "Step8A canonical row_500m/col_500m; fixed origin (0,0); assigned before population filtering", "primary_population": PRIMARY_POPULATION, "bootstrap": {"replicates": STEP8C_N_BOOTSTRAP, "seed": STEP8C_RANDOM_SEED, "ci": [STEP8C_CI_LOWER, STEP8C_CI_UPPER], "paired_large_blocks": True}, "original_block_size_verified": {e: references[e]["metrics"]["spatial_cv_config"]["spatial_block_size_cells"] for e in EXPECTED_EXPERIMENTS}, "preregistration_status": "present_will_validate" if (output_root / "step8_large_block_preregistration.json").exists() else "absent_will_create_before_fit", "fit_or_bootstrap_performed": False, "files_written": False}


def run_analysis(experiments: list[str], block_sizes: list[int], dry_run: bool = False, force: bool = False, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    validate_analysis_request(experiments, block_sizes)
    if dry_run:
        return dry_run_plan(experiments, block_sizes, output_root)
    before = hash_protected_inputs()
    manifest_path = output_root / "step8_large_block_preregistration.json"
    if manifest_path.exists():
        manifest = validate_or_write_manifest(output_root, before)
        assert_downstream_outputs_writable(output_root, force)
    else:
        assert_downstream_outputs_writable(output_root, force)
        manifest = validate_or_write_manifest(output_root, before)
    analysis_id = manifest["analysis_id"]
    audit = {"analysis_id": analysis_id, "created_at": manifest["created_at"], "protected_original_inputs": before, "package_versions": _package_versions(), "schemas": {}}
    data = {}
    for experiment in EXPECTED_EXPERIMENTS:
        path = experiment_step8_root(experiment) / PROTECTED_RELATIVE_PATHS[0]
        df = pd.read_parquet(path)
        validate_canonical_grid(df)
        audit["schemas"][experiment] = {"step8a_path": str(path), "row_count": len(df), "columns": list(df.columns), "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()}}
        data[experiment] = df
    (output_root / "step8_large_block_input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    conditions = []
    for experiment in EXPECTED_EXPERIMENTS:
        for block_size in EXPECTED_BLOCK_SIZES:
            result = run_oof_condition(data[experiment], experiment, block_size, analysis_id)
            bootstrap, replicates = paired_large_block_bootstrap(result["predictions"])
            condition = write_condition_outputs(_condition_output_dir(experiment, block_size, output_root), analysis_id, experiment, block_size, result, bootstrap, replicates, force)
            conditions.append(condition)
    comparison = [reference_comparison_row(experiment, analysis_id) for experiment in EXPECTED_EXPERIMENTS] + [large_comparison_row(condition, analysis_id) for condition in conditions]
    _write_csv(output_root / "step8_large_block_comparison.csv", comparison)
    after = hash_protected_inputs()
    assert_protected_hashes_unchanged(before, after)
    report = write_final_report(output_root, analysis_id, comparison, conditions, "passed")
    return {"ran": True, "analysis_id": analysis_id, "conditions": conditions, "report": report, "protected_hash_check": "passed"}
