"""Regression tests for the single-experiment Step8 big-spatial-block
robustness analysis (src/step8_big_block_robustness.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.step8_big_block_robustness as robust
from scripts import run_step8_big_block_robustness as direct_runner
from src.step8b_train_baseline_vs_thermal_model import (
    BASELINE_FEATURES,
    THERMAL_MODEL_FEATURES,
    Step8BError,
    build_classifier,
)


def grid_frame(rows: list[int], cols: list[int], labels: list[int] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({
        "row_500m": rows,
        "col_500m": cols,
        "cell_id": [f"r{row}_c{col}" for row, col in zip(rows, cols)],
    })
    if labels is not None:
        frame["burned"] = labels
    return frame


def modeling_frame() -> pd.DataFrame:
    """10 distinct big-blocks (at block_size=10) x 2 rows, alternating
    label -- the same shape used by the frozen v1 large-block test suite,
    known to satisfy StratifiedGroupKFold(n_splits=5, strict) cleanly."""
    rows, cols, labels = [], [], []
    for group in range(10):
        rows.extend([group * 10, group * 10])
        cols.extend([0, 1])
        labels.extend([0, 1])
    frame = grid_frame(rows, cols, labels)
    for feature in THERMAL_MODEL_FEATURES:
        if feature not in frame:
            frame[feature] = np.arange(len(frame), dtype=float)
    frame["landcover_dominant"] = "tree"
    frame["valid_for_modeling"] = True
    frame["burnable_tree_shrub_grass"] = True
    frame["burnable_tree_shrub"] = True
    frame["burn_month"] = 8
    return frame


# ---------------------------------------------------------------------------
# 1-2. Deterministic block IDs at 10 and 20 cells
# ---------------------------------------------------------------------------
def test_deterministic_10_cell_block_id():
    frame = grid_frame([0, 9, 10, 19], [0, 9, 0, 19])
    assigned = robust.add_spatial_block_id(frame, 10, column_name="big_block_id", id_prefix="block10", include_row_col=True)
    assert assigned["big_block_id"].tolist() == ["block10_0_0", "block10_0_0", "block10_1_0", "block10_1_1"]
    # same row/col always yields the same block id (determinism)
    again = robust.add_spatial_block_id(frame, 10, column_name="big_block_id", id_prefix="block10", include_row_col=True)
    assert assigned["big_block_id"].tolist() == again["big_block_id"].tolist()


def test_deterministic_20_cell_block_id():
    frame = grid_frame([0, 19, 20, 39], [0, 19, 20, 39])
    assigned = robust.add_spatial_block_id(frame, 20, column_name="big_block_id", id_prefix="block20", include_row_col=True)
    assert assigned["big_block_id"].tolist() == ["block20_0_0", "block20_0_0", "block20_1_1", "block20_1_1"]


# ---------------------------------------------------------------------------
# 3. Neighboring cells fall inside the same block
# ---------------------------------------------------------------------------
def test_neighboring_cells_same_block():
    frame = grid_frame([0, 1, 2], [0, 1, 2])
    assigned = robust.add_spatial_block_id(frame, 10, column_name="big_block_id", id_prefix="block10")
    assert assigned["big_block_id"].nunique() == 1


# ---------------------------------------------------------------------------
# 4. Boundary cells enter adjacent blocks
# ---------------------------------------------------------------------------
def test_boundary_cells_enter_adjacent_blocks():
    frame = grid_frame([9, 10], [9, 10])
    assigned = robust.add_spatial_block_id(frame, 10, column_name="big_block_id", id_prefix="block10")
    assert assigned["big_block_id"].nunique() == 2


# ---------------------------------------------------------------------------
# 5-6. No train/test leakage; baseline and thermal share the same folds
# ---------------------------------------------------------------------------
def test_no_train_test_block_leakage_and_shared_folds():
    frame = modeling_frame()
    with patch.object(robust, "STEP8B_MIN_POSITIVES_PER_POPULATION", 5):
        result = robust.run_big_block_condition(frame, "test_experiment", 10, "analysis-test")
    assert result["status"] == "fitted"
    predictions = result["predictions"]
    # Both models scored from the SAME fold partition (single fold_id column
    # shared by baseline_probability and thermal_probability).
    assert predictions["fold_id"].isin(range(5)).all()
    for fold in result["block_audit"]["folds"]:
        assert fold["block_overlap"] == 0
    assert result["block_audit"]["train_test_block_leakage_free"] is True


# ---------------------------------------------------------------------------
# 7. Paired bootstrap uses identical samples for baseline and thermal
# ---------------------------------------------------------------------------
def test_paired_bootstrap_uses_identical_samples():
    predictions = pd.DataFrame({
        "spatial_block_id": ["a", "a", "b", "b"],
        "burned": [0, 1, 0, 1],
        "baseline_probability": [0.1, 0.8, 0.2, 0.7],
        "thermal_probability": [0.2, 0.9, 0.3, 0.8],
    })
    calls = []

    def fake_metrics(y, baseline, thermal):
        calls.append((y.copy(), baseline.copy(), thermal.copy()))
        assert np.allclose(thermal - baseline, 0.1)
        return {
            "auc_baseline": 0.5, "auc_thermal": 0.6, "delta_auc": 0.1,
            "pr_auc_baseline": 0.5, "pr_auc_thermal": 0.6, "delta_pr_auc": 0.1,
            "brier_baseline": 0.2, "brier_thermal": 0.15, "delta_brier": -0.05,
        }

    with patch.object(robust, "compute_paired_metrics", side_effect=fake_metrics):
        summary, replicates = robust.paired_big_block_bootstrap(predictions, n_replicates=12, seed=42)
    assert len(calls) == 12
    assert summary["valid_replicates"] == 12
    assert replicates["valid"].all()


# ---------------------------------------------------------------------------
# 8. Bootstrap unit is the spatial block, not the row
# ---------------------------------------------------------------------------
def test_bootstrap_unit_is_spatial_block_not_row():
    predictions = pd.DataFrame({
        "spatial_block_id": ["a", "a", "a", "b"],
        "burned": [0, 1, 1, 0],
        "baseline_probability": [0.1, 0.8, 0.7, 0.2],
        "thermal_probability": [0.2, 0.9, 0.8, 0.3],
    })
    seen_sizes = []

    def fake_metrics(y, baseline, thermal):
        seen_sizes.append(len(y))
        return None  # force "invalid" so we don't need real AUC validity

    with patch.object(robust, "compute_paired_metrics", side_effect=fake_metrics):
        robust.paired_big_block_bootstrap(predictions, n_replicates=20, seed=1)
    # block "a" has 3 rows, block "b" has 1 row; each replicate draws 2
    # block-picks with replacement, so the resampled row count must always
    # be a sum of two values from {1, 3} -- i.e. in {2, 4, 6} -- never an
    # arbitrary row-level count such as 3 or 5 that would only arise from
    # resampling individual rows instead of whole spatial blocks.
    assert set(seen_sizes).issubset({2, 4, 6})
    assert len(seen_sizes) == 20


# ---------------------------------------------------------------------------
# 9. Invalid single-class bootstrap replicates are counted, not silently dropped
# ---------------------------------------------------------------------------
def test_invalid_single_class_bootstrap_replicate_counted():
    predictions = pd.DataFrame({
        "spatial_block_id": ["a", "a", "b"],
        "burned": [0, 0, 0],
        "baseline_probability": [0.1, 0.2, 0.3],
        "thermal_probability": [0.2, 0.3, 0.4],
    })
    summary, replicates = robust.paired_big_block_bootstrap(predictions, n_replicates=7, seed=42)
    assert summary["valid_replicates"] == 0
    assert summary["invalid_single_class_replicates"] == 7
    assert (replicates["invalid_reason"] == "single_class").all()


# ---------------------------------------------------------------------------
# 10-11. Existing model parameters / feature lists unchanged
# ---------------------------------------------------------------------------
def test_existing_model_parameters_unchanged():
    classifier = build_classifier(robust.MODEL_NAME, robust.STEP8B_RANDOM_SEED)
    params = classifier.get_params(deep=False)
    assert params["n_estimators"] == 300
    assert params["min_samples_leaf"] == 3
    assert params["class_weight"] == "balanced"
    assert params["random_state"] == robust.STEP8B_RANDOM_SEED


def test_existing_feature_lists_unchanged():
    assert robust.BASELINE_FEATURES == list(BASELINE_FEATURES)
    assert robust.THERMAL_MODEL_FEATURES == list(THERMAL_MODEL_FEATURES)


# ---------------------------------------------------------------------------
# 12. Old (Step8A/B/C/E) outputs are not overwritten -- protected-hash check
# ---------------------------------------------------------------------------
def test_old_outputs_not_overwritten(tmp_path):
    original = tmp_path / "step8b_predictions.parquet"
    original.write_bytes(b"frozen-step8-output")
    before = robust.hash_protected_inputs({"step8b/step8b_predictions.parquet": original})
    after = robust.hash_protected_inputs({"step8b/step8b_predictions.parquet": original})
    robust.assert_all_protected_unchanged(before, after)
    original.write_bytes(b"mutated")
    changed = robust.hash_protected_inputs({"step8b/step8b_predictions.parquet": original})
    with pytest.raises(robust.Step8BigBlockRobustnessError):
        robust.assert_all_protected_unchanged(before, changed)


# ---------------------------------------------------------------------------
# 13. block_10_cells / block_20_cells namespaces are isolated
# ---------------------------------------------------------------------------
def test_two_block_namespaces_isolated():
    root = robust.experiment_output_root("test_experiment")
    ten = robust._condition_output_dir("test_experiment", 10, root)
    twenty = robust._condition_output_dir("test_experiment", 20, root)
    assert ten != twenty
    assert not ten.is_relative_to(twenty)
    assert not twenty.is_relative_to(ten)
    assert ten.name == "block_10_cells"
    assert twenty.name == "block_20_cells"


# ---------------------------------------------------------------------------
# 14. Original small-block reference metrics are loaded from the artifact,
#     never hard-coded
# ---------------------------------------------------------------------------
def test_original_reference_metrics_loaded_from_artifact(tmp_path):
    root = tmp_path / "outputs" / "experiments" / "test_experiment"
    (root / "step8b").mkdir(parents=True)
    (root / "step8c").mkdir(parents=True)
    metrics = {
        "spatial_cv_config": {
            "spatial_block_size_cells": 2, "method": "StratifiedGroupKFold",
            "n_splits_requested": 5, "random_state": 42, "random_split_used": False,
        },
        "model": "random_forest",
        "feature_sets": {
            "baseline": list(BASELINE_FEATURES), "thermal_additional": list(robust.THERMAL_FEATURES),
            "thermal_model_full": list(THERMAL_MODEL_FEATURES),
        },
        "population_metrics": {
            "burnable_tree_shrub_grass": {
                "overall_baseline": {"roc_auc": 0.1234, "pr_auc": 0.5},
                "overall_thermal": {"roc_auc": 0.9, "pr_auc": 0.6},
                "delta_auc": 0.7766, "delta_pr_auc": 0.1,
            },
        },
    }
    bootstrap = {
        "n_bootstrap_requested": 1000, "random_seed": 42,
        "bootstrap_ci_by_population": {
            "burnable_tree_shrub_grass": {"delta_auc_ci95": [0.7, 0.85], "delta_pr_auc_ci95": [0.05, 0.15]},
        },
    }
    (root / "step8b" / "step8b_model_comparison_metrics.json").write_text(json.dumps(metrics))
    (root / "step8c" / "step8c_bootstrap_metrics.json").write_text(json.dumps(bootstrap))
    with patch.object(robust, "experiment_step8_root", return_value=root):
        reference = robust.original_small_block_reference("test_experiment")
    # the exact, unusual value 0.1234 could only have come from the file.
    assert reference["point"]["overall_baseline"]["roc_auc"] == 0.1234
    assert reference["reference_check"]["reference_metric_mismatch"] is True


# ---------------------------------------------------------------------------
# 15. Delta sign convention: ROC/PR = thermal - baseline; Brier = thermal -
#     baseline but NEGATIVE means thermal is better (documented, not silently
#     flipped anywhere else in the module).
# ---------------------------------------------------------------------------
def test_delta_sign_convention_correct():
    metrics = robust.compute_paired_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.2, 0.3, 0.4, 0.5]),
        np.array([0.1, 0.9, 0.1, 0.9]),
    )
    assert metrics["delta_auc"] == pytest.approx(metrics["auc_thermal"] - metrics["auc_baseline"])
    assert metrics["delta_pr_auc"] == pytest.approx(metrics["pr_auc_thermal"] - metrics["pr_auc_baseline"])
    assert metrics["delta_brier"] == pytest.approx(metrics["brier_thermal"] - metrics["brier_baseline"])
    # thermal is clearly better-calibrated here (predictions closer to 0/1
    # matching truth), so delta_brier must be negative under this convention.
    assert metrics["delta_brier"] < 0


# ---------------------------------------------------------------------------
# 16. CI support / robustness classification is deterministic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(0.01, 0.2, "supported_positive"), (-0.01, 0.2, "uncertain"), (-0.2, -0.01, "supported_negative")],
)
def test_classify_metric_support_deterministic(low, high, expected):
    assert robust.classify_metric_support(low, high) == expected


def test_classify_brier_support_sign_flip():
    # delta_brier CI entirely negative (thermal better) -> supported_positive
    assert robust.classify_brier_support(-0.2, -0.01) == "supported_positive"
    assert robust.classify_brier_support(0.01, 0.2) == "supported_negative"
    assert robust.classify_brier_support(-0.1, 0.1) == "uncertain"


@pytest.mark.parametrize(
    ("s10", "s20", "expected"),
    [
        ("retained", "retained", "strongly_robust"),
        ("retained", "partially_retained", "moderately_robust"),
        ("partially_retained", "retained", "moderately_robust"),
        ("partially_retained", "partially_retained", "moderately_robust"),
        ("retained", "not_retained", "scale_sensitive"),
        ("not_retained", "retained", "scale_sensitive"),
        ("partially_retained", "not_retained", "scale_sensitive"),
        ("not_retained", "partially_retained", "scale_sensitive"),
        ("not_retained", "not_retained", "not_robust"),
    ],
)
def test_classify_final_robustness_full_table(s10, s20, expected):
    assert robust.classify_final_robustness(s10, s20) == expected


def test_classify_overall_support():
    assert robust.classify_overall_support("supported_positive", "supported_positive") == "retained"
    assert robust.classify_overall_support("supported_positive", "uncertain") == "partially_retained"
    assert robust.classify_overall_support("uncertain", "uncertain") == "not_retained"


# ---------------------------------------------------------------------------
# 17. Infeasible fold partition fails honestly (no silent fallback)
# ---------------------------------------------------------------------------
def test_infeasible_fold_partition_fails_honestly():
    # Only 2 unique big-blocks -- cannot build 5 strict spatial-group folds.
    frame = grid_frame([0, 0, 10, 10], [0, 1, 0, 1], [0, 1, 0, 1])
    for feature in THERMAL_MODEL_FEATURES:
        if feature not in frame:
            frame[feature] = np.arange(len(frame), dtype=float)
    frame["landcover_dominant"] = "tree"
    frame["valid_for_modeling"] = True
    frame["burnable_tree_shrub_grass"] = True
    frame["burnable_tree_shrub"] = True
    frame["burn_month"] = 8
    with patch.object(robust, "STEP8B_MIN_POSITIVES_PER_POPULATION", 1):
        result = robust.run_big_block_condition(frame, "test_experiment", 10, "analysis-test")
    assert result["status"] == "infeasible_with_existing_cv_protocol"
    assert "reason" in result


# ---------------------------------------------------------------------------
# 18. Future AOI works without any experiment-specific branch
# ---------------------------------------------------------------------------
def test_no_aoi_specific_branching_in_source():
    source = Path(robust.__file__).read_text(encoding="utf-8")
    assert not re.search(r'experiment_id\s*==\s*["\']', source)
    assert "mugla_2021" not in source


def test_run_big_block_condition_works_for_an_arbitrary_experiment_id():
    frame = modeling_frame()
    with patch.object(robust, "STEP8B_MIN_POSITIVES_PER_POPULATION", 5):
        result_a = robust.run_big_block_condition(frame, "some_future_experiment_2099", 10, "analysis-test")
        result_b = robust.run_big_block_condition(frame, "mugla_2021", 10, "analysis-test")
    assert result_a["status"] == "fitted"
    assert result_b["status"] == "fitted"
    assert result_a["predictions"]["experiment_id"].unique().tolist() == ["some_future_experiment_2099"]


# ---------------------------------------------------------------------------
# Namespace / dispatch sanity
# ---------------------------------------------------------------------------
def test_condition_output_namespace_is_under_experiment_root():
    path = robust._condition_output_dir("mugla_2021", 10)
    assert path.as_posix().endswith("outputs/experiments/mugla_2021/robustness/step8_big_blocks/block_10_cells")


def test_direct_runner_dispatches_exact_values():
    with patch.object(direct_runner, "run_analysis", return_value={"ran": False}) as mocked:
        result = direct_runner.main(experiment="mugla_2021", block_sizes=[10, 20], dry_run=True, force=False)
    assert result == {"ran": False}
    # `output_root` selects a VERSIONED namespace; None keeps the default one.
    mocked.assert_called_once_with(
        experiment_id="mugla_2021", block_sizes=[10, 20], dry_run=True, force=False,
        output_root=None,
    )


def test_direct_runner_regenerate_reports_only_dispatches_exclusively():
    """--regenerate-reports-only must call ONLY
    regenerate_reports_from_frozen_artifacts (reads the frozen
    preregistration + per-condition JSON/Parquet artifacts) and must never
    call run_analysis (which can create a preregistration, build folds, fit
    models, or sample bootstrap replicates). --block-sizes must NOT be
    forwarded -- block sizes come from the frozen preregistration."""
    with (
        patch.object(direct_runner, "regenerate_reports_from_frozen_artifacts", return_value={"ran": True}) as mocked_regen,
        patch.object(direct_runner, "run_analysis", side_effect=AssertionError("run_analysis must not be called")) as mocked_run,
    ):
        result = direct_runner.main(
            experiment="mugla_2021", block_sizes=[10, 20], dry_run=False, force=False,
            regenerate_reports_only=True,
        )
    assert result == {"ran": True}
    mocked_regen.assert_called_once_with(
        experiment_id="mugla_2021", dry_run=False, output_root=None,
    )
    mocked_run.assert_not_called()


def test_validate_block_sizes_rejects_small_or_invalid_sizes():
    robust.validate_block_sizes([10, 20])
    with pytest.raises(robust.Step8BigBlockRobustnessError):
        robust.validate_block_sizes([2])
    with pytest.raises(robust.Step8BigBlockRobustnessError):
        robust.validate_block_sizes([0])
    with pytest.raises(robust.Step8BigBlockRobustnessError):
        robust.validate_block_sizes([])


# ---------------------------------------------------------------------------
# regenerate_reports_from_frozen_artifacts control-flow tests
# ---------------------------------------------------------------------------
def _write_frozen_original_step8_artifacts(step8_root: Path) -> None:
    for sub in ("step8a", "step8b", "step8c", "step8e"):
        (step8_root / sub).mkdir(parents=True, exist_ok=True)
    (step8_root / "step8a" / "step8a_500m_modeling_dataset.parquet").write_bytes(b"dummy")
    (step8_root / "step8b" / "step8b_predictions.parquet").write_bytes(b"dummy")
    (step8_root / "step8e" / "final_step8_report.json").write_text("{}")
    metrics = {
        "spatial_cv_config": {
            "spatial_block_size_cells": 2, "method": "StratifiedGroupKFold",
            "n_splits_requested": 5, "random_state": 42, "random_split_used": False,
        },
        "model": "random_forest",
        "feature_sets": {
            "baseline": list(BASELINE_FEATURES), "thermal_additional": list(robust.THERMAL_FEATURES),
            "thermal_model_full": list(THERMAL_MODEL_FEATURES),
        },
        "population_metrics": {
            "burnable_tree_shrub_grass": {
                "overall_baseline": {"roc_auc": 0.74, "pr_auc": 0.22, "brier_score": 0.11},
                "overall_thermal": {"roc_auc": 0.86, "pr_auc": 0.44, "brier_score": 0.07},
                "delta_auc": 0.12, "delta_pr_auc": 0.22, "delta_brier": -0.04,
            },
        },
    }
    bootstrap = {
        "n_bootstrap_requested": 1000, "random_seed": 42,
        "bootstrap_ci_by_population": {
            "burnable_tree_shrub_grass": {
                "delta_auc_ci95": [0.10, 0.13], "delta_pr_auc_ci95": [0.19, 0.24],
                "delta_brier_ci95": [-0.045, -0.035],
            },
        },
    }
    (step8_root / "step8b" / "step8b_model_comparison_metrics.json").write_text(json.dumps(metrics))
    (step8_root / "step8c" / "step8c_bootstrap_metrics.json").write_text(json.dumps(bootstrap))


def _write_frozen_big_block_manifest(output_root: Path, analysis_id: str, block_sizes=(10, 20)) -> None:
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "analysis_id": analysis_id,
        "created_at": "2020-01-01T00:00:00+00:00",
        "git_commit": "abc123",
        "scientific_configuration": {
            "analysis_schema_version": robust.ANALYSIS_SCHEMA_VERSION,
            "block_sizes_cells": list(block_sizes),
        },
    }
    (comparison_dir / "manifest.json").write_text(json.dumps(manifest))
    for size in block_sizes:
        (output_root / f"block_{size}_cells").mkdir(parents=True, exist_ok=True)


def _canned_condition(analysis_id: str, experiment_id: str, block_size: int) -> dict:
    common = {
        "analysis_id": analysis_id, "experiment_id": experiment_id,
        "block_size_cells": block_size, "nominal_scale": robust.nominal_scale_label(block_size),
        "primary_population": "burnable_tree_shrub_grass",
    }
    metrics = {
        **common,
        "baseline_roc_auc": 0.6, "thermal_roc_auc": 0.75, "delta_roc_auc": 0.15,
        "baseline_pr_auc": 0.2, "thermal_pr_auc": 0.35, "delta_pr_auc": 0.15,
        "baseline_brier": 0.2, "thermal_brier": 0.15, "delta_brier": -0.05,
    }
    series_default = {"mean": 0.1, "median": 0.1, "ci_2_5": 0.05, "ci_97_5": 0.15}
    # Mirrors the frozen bootstrap_summary.json schema on disk: the replicate
    # accounting fields live FLAT at the top level (not nested under a
    # scientific_configuration block), and the report-only Markdown renderer
    # reads them verbatim. A fixture missing them is not a lighter fixture --
    # it is a different schema from the one report-only mode consumes.
    bootstrap = {
        **common, "bootstrap_stability": "stable",
        "requested_replicates": 1000,
        "valid_replicates": 1000,
        "invalid_single_class_replicates": 0,
        "series": {name: dict(series_default) for name in (
            "auc_baseline", "auc_thermal", "delta_auc",
            "pr_auc_baseline", "pr_auc_thermal", "delta_pr_auc",
            "brier_baseline", "brier_thermal", "delta_brier",
        )},
    }
    block_audit = {
        "total_rows": 100, "eligible_rows": 100, "positive_rows": 20, "negative_rows": 80,
        "unique_spatial_blocks": 10, "positive_containing_blocks": 5,
        "negative_containing_blocks": 10, "mixed_class_blocks": 3, "fold_count": 5,
        "train_test_block_leakage_total": 0, "train_test_block_leakage_free": True,
        "folds": [],
    }
    return {"metrics": metrics, "bootstrap": bootstrap, "block_audit": block_audit}


def _fake_load_condition_artifacts(analysis_id: str, experiment_id: str):
    def _loader(output_dir: Path):
        block_size = int(output_dir.name.split("_")[1])
        return _canned_condition(analysis_id, experiment_id, block_size)
    return _loader


def test_regenerate_reports_bypasses_runtime_scientific_config_comparison(tmp_path):
    experiment_id = "test_experiment"
    analysis_id = "frozen-analysis-id-abc"
    step8_root = tmp_path / "outputs" / "experiments" / experiment_id
    _write_frozen_original_step8_artifacts(step8_root)
    output_root = tmp_path / "big_blocks"
    _write_frozen_big_block_manifest(output_root, analysis_id, block_sizes=(10, 20))

    def _boom(*_args, **_kwargs):
        raise AssertionError("runtime scientific-config comparison must not run in report-only mode")

    with (
        patch.object(robust, "experiment_step8_root", return_value=step8_root),
        patch.object(robust, "scientific_configuration", side_effect=_boom),
        patch.object(robust, "validate_or_write_manifest", side_effect=_boom),
        patch.object(robust, "load_condition_artifacts", side_effect=_fake_load_condition_artifacts(analysis_id, experiment_id)),
    ):
        result = robust.regenerate_reports_from_frozen_artifacts(experiment_id, output_root=output_root)

    assert result["ran"] is True


def test_regenerate_reports_preserves_frozen_analysis_id(tmp_path):
    experiment_id = "test_experiment"
    analysis_id = "frozen-analysis-id-xyz"
    step8_root = tmp_path / "outputs" / "experiments" / experiment_id
    _write_frozen_original_step8_artifacts(step8_root)
    output_root = tmp_path / "big_blocks"
    _write_frozen_big_block_manifest(output_root, analysis_id, block_sizes=(10, 20))

    with (
        patch.object(robust, "experiment_step8_root", return_value=step8_root),
        patch.object(robust, "load_condition_artifacts", side_effect=_fake_load_condition_artifacts(analysis_id, experiment_id)),
    ):
        result = robust.regenerate_reports_from_frozen_artifacts(experiment_id, output_root=output_root)

    # The frozen analysis_id is carried through, never recomputed.
    assert result["analysis_id"] == analysis_id
    assert result["report"]["analysis_id"] == analysis_id
    written_manifest = json.loads((output_root / "comparison" / "manifest.json").read_text())
    assert written_manifest["analysis_id"] == analysis_id


def test_regenerate_reports_does_not_create_preregistration_or_execute_analysis(tmp_path):
    experiment_id = "test_experiment"
    analysis_id = "frozen-analysis-id-noexec"
    step8_root = tmp_path / "outputs" / "experiments" / experiment_id
    _write_frozen_original_step8_artifacts(step8_root)
    output_root = tmp_path / "big_blocks"
    _write_frozen_big_block_manifest(output_root, analysis_id, block_sizes=(10, 20))

    def _boom(*_args, **_kwargs):
        raise AssertionError("preregistration creation / analysis execution must not run in report-only mode")

    with (
        patch.object(robust, "experiment_step8_root", return_value=step8_root),
        patch.object(robust, "load_condition_artifacts", side_effect=_fake_load_condition_artifacts(analysis_id, experiment_id)),
        patch.object(robust, "build_manifest", side_effect=_boom),
        patch.object(robust, "validate_or_write_manifest", side_effect=_boom),
        patch.object(robust, "run_big_block_condition", side_effect=_boom),
        patch.object(robust, "paired_big_block_bootstrap", side_effect=_boom),
        patch.object(robust, "write_condition_outputs", side_effect=_boom),
    ):
        result = robust.regenerate_reports_from_frozen_artifacts(experiment_id, output_root=output_root)

    assert result["ran"] is True
    assert result["models_refit"] is False
    assert result["bootstrap_rerun"] is False


def test_normal_mode_rejects_incompatible_immutable_preregistration(tmp_path):
    """run_analysis's immutable preregistration validation must remain
    unchanged: an existing manifest whose scientific_configuration disagrees
    with the runtime one still fails fast."""
    output_root = tmp_path / "big_blocks"
    (output_root / "comparison").mkdir(parents=True)
    existing = {
        "analysis_id": "old-analysis-id",
        "created_at": "2020-01-01T00:00:00+00:00",
        "git_commit": None,
        "scientific_configuration": {"analysis_schema_version": "step8.big_block_robustness.v1", "block_sizes_cells": [10, 20]},
    }
    (output_root / "comparison" / "manifest.json").write_text(json.dumps(existing))

    with patch.object(
        robust, "scientific_configuration",
        return_value={"analysis_schema_version": robust.ANALYSIS_SCHEMA_VERSION, "block_sizes_cells": [10, 20]},
    ):
        with pytest.raises(robust.Step8BigBlockRobustnessError, match="disagrees with runtime scientific configuration"):
            robust.validate_or_write_manifest(output_root, "test_experiment", [10, 20], {})


# =============================================================================
# Report-only regeneration: frozen bootstrap accounting + no model/bootstrap
# =============================================================================
def _real_bootstrap_summary_keys() -> set[str]:
    """The replicate-accounting keys the frozen artefacts actually carry."""
    return {"requested_replicates", "valid_replicates", "invalid_single_class_replicates"}


def test_report_only_reads_the_frozen_bootstrap_replicate_accounting(tmp_path):
    """The regenerated bootstrap_summary must carry the frozen replicate
    accounting through verbatim, read from the FLAT top-level keys the frozen
    artefacts use -- report-only mode never re-derives or re-samples them."""
    experiment_id = "test_experiment"
    analysis_id = "frozen-analysis-id-boot"
    step8_root = tmp_path / "outputs" / "experiments" / experiment_id
    _write_frozen_original_step8_artifacts(step8_root)
    output_root = tmp_path / "big_blocks"
    _write_frozen_big_block_manifest(output_root, analysis_id, block_sizes=(10, 20))

    with (
        patch.object(robust, "experiment_step8_root", return_value=step8_root),
        patch.object(robust, "load_condition_artifacts", side_effect=_fake_load_condition_artifacts(analysis_id, experiment_id)),
    ):
        robust.regenerate_reports_from_frozen_artifacts(experiment_id, output_root=output_root)

    canned = _canned_condition(analysis_id, experiment_id, 10)["bootstrap"]
    for block_size in (10, 20):
        block_dir = output_root / f"block_{block_size}_cells"
        written = json.loads((block_dir / "bootstrap_summary.json").read_text())
        for key in _real_bootstrap_summary_keys():
            assert key in written, f"{key} missing from regenerated bootstrap_summary.json"
            assert written[key] == canned[key]
        markdown = (block_dir / "bootstrap_summary.md").read_text(encoding="utf-8")
        assert f"requested replicates: {canned['requested_replicates']}" in markdown
        assert f"valid replicates: {canned['valid_replicates']}" in markdown


def test_report_only_matches_the_frozen_on_disk_bootstrap_schema():
    """Guard the fixture against drifting away from the real artefacts: every
    replicate-accounting key the renderer needs must exist, flat, on disk."""
    frozen = sorted(
        (_PROJECT_ROOT / "outputs" / "experiments").glob(
            "*/robustness/step8_big_blocks*/block_*_cells/bootstrap_summary.json"
        )
    )
    if not frozen:
        pytest.skip("no frozen big-block bootstrap summaries in this checkout")
    for path in frozen:
        payload = json.loads(path.read_text())
        missing = _real_bootstrap_summary_keys() - set(payload)
        assert not missing, f"{path} is missing {sorted(missing)}"


def test_report_only_never_writes_replicate_or_prediction_parquets(tmp_path):
    """Report-only regeneration must not produce any of the artefacts that
    only a real fit/bootstrap can produce."""
    experiment_id = "test_experiment"
    analysis_id = "frozen-analysis-id-noparquet"
    step8_root = tmp_path / "outputs" / "experiments" / experiment_id
    _write_frozen_original_step8_artifacts(step8_root)
    output_root = tmp_path / "big_blocks"
    _write_frozen_big_block_manifest(output_root, analysis_id, block_sizes=(10, 20))

    with (
        patch.object(robust, "experiment_step8_root", return_value=step8_root),
        patch.object(robust, "load_condition_artifacts", side_effect=_fake_load_condition_artifacts(analysis_id, experiment_id)),
    ):
        result = robust.regenerate_reports_from_frozen_artifacts(experiment_id, output_root=output_root)

    assert result["models_refit"] is False
    assert result["bootstrap_rerun"] is False
    assert list(output_root.rglob("*.parquet")) == []
    # The frozen analysis identity is preserved, never recomputed.
    for block_size in (10, 20):
        block_dir = output_root / f"block_{block_size}_cells"
        for name in ("step8b_metrics.json", "bootstrap_summary.json", "block_manifest.json"):
            assert json.loads((block_dir / name).read_text())["analysis_id"] == analysis_id
