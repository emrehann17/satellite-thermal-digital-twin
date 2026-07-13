"""Focused tests for the frozen Step8 large-spatial-block robustness analysis."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.step8_large_block_robustness as robust
from core.config import STEP8B_SPATIAL_BLOCK_SIZE_CELLS
from scripts import run_step8_large_block_robustness as direct_runner
from src.step8b_train_baseline_vs_thermal_model import (
    BASELINE_FEATURES,
    THERMAL_MODEL_FEATURES,
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
    frame[robust.PRIMARY_POPULATION] = True
    return frame


def test_original_step8_default_remains_two_cells():
    assert STEP8B_SPATIAL_BLOCK_SIZE_CELLS == 2


def test_frozen_request_accepts_exact_plan_only():
    robust.validate_analysis_request(["manavgat_2021", "bejis_2022"], [10, 20])
    with pytest.raises(robust.Step8RobustnessError):
        robust.validate_analysis_request(["manavgat_2021", "bejis_2022"], [5, 20])
    with pytest.raises(robust.Step8RobustnessError):
        robust.validate_analysis_request(["bejis_2022", "manavgat_2021"], [10, 20])
    with pytest.raises(robust.Step8RobustnessError):
        robust.validate_analysis_request(["manavgat_2021"], [10, 20])


def test_large_block_assignment_is_deterministic_and_label_independent():
    frame = grid_frame([0, 9, 10, 19, 20], [0, 9, 0, 19, 20], [0, 1, 1, 0, 1])
    first = robust.assign_large_blocks(frame, 10)
    second = robust.assign_large_blocks(frame.drop(columns="burned"), 10)
    assert first["large_block_id"].tolist() == second["large_block_id"].tolist()
    assert first["large_block_id"].tolist() == [
        "b10_r0_c0", "b10_r0_c0", "b10_r1_c0", "b10_r1_c1", "b10_r2_c2"
    ]


def test_fixed_origin_makes_twenty_cell_partition_nested_over_ten_cell_partition():
    frame = grid_frame([0, 9, 10, 19, 20, 39], [0, 19, 20, 39, 40, 59])
    ten = robust.assign_large_blocks(frame, 10)
    twenty = robust.assign_large_blocks(frame, 20)
    assert (twenty["large_block_row"].to_numpy() == ten["large_block_row"].to_numpy() // 2).all()
    assert (twenty["large_block_col"].to_numpy() == ten["large_block_col"].to_numpy() // 2).all()


def test_canonical_grid_mismatch_fails_clearly():
    frame = grid_frame([0], [0])
    frame.loc[0, "cell_id"] = "r9_c9"
    with pytest.raises(robust.Step8RobustnessError):
        robust.assign_large_blocks(frame, 10)


def test_strict_folds_have_no_block_overlap_and_exact_oof_coverage():
    frame = modeling_frame()
    assigned = robust.assign_large_blocks(frame, 10)
    folds = robust.make_strict_spatial_folds(
        assigned["burned"].to_numpy(), assigned["large_block_id"].to_numpy()
    )
    coverage = np.zeros(len(assigned), dtype=int)
    for train_idx, test_idx in folds:
        assert set(assigned.iloc[train_idx]["large_block_id"]).isdisjoint(
            set(assigned.iloc[test_idx]["large_block_id"])
        )
        coverage[test_idx] += 1
    assert np.array_equal(coverage, np.ones(len(assigned), dtype=int))
    assert len(folds) == 5


def test_strict_folds_never_reduce_frozen_fold_count():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    with pytest.raises(robust.Step8RobustnessError, match="too few large blocks"):
        robust.make_strict_spatial_folds(y, groups)


def test_oof_fits_preprocessing_only_on_training_rows_and_predicts_each_row_once():
    fit_predict_pairs = []

    class SpyPipeline:
        def __init__(self):
            self.train_index = None

        def fit(self, x, y):
            self.train_index = set(x.index)
            return self

        def predict_proba(self, x):
            assert self.train_index.isdisjoint(set(x.index))
            fit_predict_pairs.append((self.train_index, set(x.index)))
            return np.column_stack([np.full(len(x), 0.5), np.full(len(x), 0.5)])

    result = robust.run_oof_condition(
        modeling_frame(), "manavgat_2021", 10, "analysis-test",
        pipeline_builder=lambda *_: SpyPipeline(),
    )
    predictions = result["predictions"]
    assert len(predictions) == len(modeling_frame())
    assert predictions["cell_id"].is_unique
    assert sorted(predictions["fold_id"].unique()) == [0, 1, 2, 3, 4]
    assert len(fit_predict_pairs) == 10


def test_model_features_and_hyperparameters_are_reused_from_step8b():
    reference = {
        "metrics": {
            "spatial_cv_config": {"spatial_block_size_cells": 2, "n_splits_requested": 5},
            "model": robust.MODEL_NAME,
        },
        "bootstrap": {"n_bootstrap_requested": 1000, "random_seed": 42},
    }
    with (
        patch.object(robust, "original_reference_payload", return_value=reference),
        patch.object(robust, "_package_versions", return_value={}),
    ):
        config = robust.scientific_configuration({})
    assert config["baseline_features"] == list(BASELINE_FEATURES)
    assert config["thermal_model_features"] == list(THERMAL_MODEL_FEATURES)
    assert config["model_hyperparameters"] == build_classifier(
        robust.MODEL_NAME, robust.STEP8B_RANDOM_SEED
    ).get_params(deep=False)


def test_paired_bootstrap_passes_same_sampled_rows_to_both_models():
    predictions = pd.DataFrame({
        "large_block_id": ["a", "a", "b", "b"],
        "burned": [0, 1, 0, 1],
        "y_prob_baseline": [0.1, 0.8, 0.2, 0.7],
        "y_prob_thermal": [0.2, 0.9, 0.3, 0.8],
    })
    calls = []

    def fake_metrics(y, baseline, thermal):
        calls.append((y.copy(), baseline.copy(), thermal.copy()))
        assert np.allclose(thermal - baseline, 0.1)
        return {
            "auc_baseline": 0.5, "auc_thermal": 0.6, "delta_auc": 0.1,
            "pr_auc_baseline": 0.5, "pr_auc_thermal": 0.6, "delta_pr_auc": 0.1,
        }

    with patch.object(robust, "compute_paired_metrics", side_effect=fake_metrics):
        summary, replicates = robust.paired_large_block_bootstrap(predictions, n_replicates=12, seed=42)
    assert len(calls) == 12
    assert summary["valid_replicates"] == 12
    assert replicates["valid"].all()


def test_one_class_bootstrap_replicates_are_invalidated_jointly():
    predictions = pd.DataFrame({
        "large_block_id": ["a", "a", "b"],
        "burned": [0, 0, 0],
        "y_prob_baseline": [0.1, 0.2, 0.3],
        "y_prob_thermal": [0.2, 0.3, 0.4],
    })
    summary, replicates = robust.paired_large_block_bootstrap(predictions, n_replicates=7, seed=42)
    assert summary["valid_replicates"] == 0
    assert summary["invalid_single_class_replicates"] == 7
    assert (replicates["invalid_reason"] == "single_class").all()


def test_protected_hashes_detect_change_without_modifying_inputs(tmp_path):
    original = tmp_path / "frozen.bin"
    original.write_bytes(b"frozen")
    before = robust.hash_protected_inputs({"frozen": original})
    after = robust.hash_protected_inputs({"frozen": original})
    robust.assert_protected_hashes_unchanged(before, after)
    original.write_bytes(b"changed")
    changed = robust.hash_protected_inputs({"frozen": original})
    with pytest.raises(robust.Step8RobustnessError):
        robust.assert_protected_hashes_unchanged(before, changed)


def test_condition_output_namespace_never_points_into_original_step8():
    path = robust._condition_output_dir("manavgat_2021", 10)
    assert path.is_relative_to(robust.OUTPUT_ROOT)
    assert "outputs/experiments" not in path.as_posix()
    assert path.as_posix().endswith("manavgat_2021/block_10_cells")


def test_manifest_is_immutable_even_when_downstream_force_would_be_requested(tmp_path):
    with (
        patch.object(robust, "scientific_configuration", return_value={"frozen": 1}),
        patch.object(robust, "_git_commit", return_value="abc"),
    ):
        manifest = robust.validate_or_write_manifest(tmp_path, {})
    manifest_path = tmp_path / "step8_large_block_preregistration.json"
    original = manifest_path.read_bytes()
    assert json.loads(original)["analysis_id"] == manifest["analysis_id"]
    with patch.object(robust, "scientific_configuration", return_value={"frozen": 2}):
        with pytest.raises(robust.Step8RobustnessError):
            robust.validate_or_write_manifest(tmp_path, {})
    assert manifest_path.read_bytes() == original


def test_existing_downstream_outputs_fail_before_fit_without_force(tmp_path):
    output = tmp_path / "robustness"
    output.mkdir()
    (output / "step8_large_block_input_audit.json").write_text("{}")
    with pytest.raises(robust.Step8RobustnessError, match="Downstream robustness outputs"):
        robust.assert_downstream_outputs_writable(output, force=False)
    robust.assert_downstream_outputs_writable(output, force=True)


def test_unstable_bootstrap_cannot_receive_strong_joint_support():
    condition = {
        "metrics": {
            "experiment_id": "manavgat_2021",
            "nominal_scale": robust.NOMINAL_SCALES[10],
            "block_size_cells": 10,
            "baseline_roc_auc": 0.5,
            "thermal_roc_auc": 0.6,
            "delta_roc_auc": 0.1,
            "baseline_pr_auc": 0.2,
            "thermal_pr_auc": 0.3,
            "delta_pr_auc": 0.1,
        },
        "bootstrap": {
            "valid_replicates": 899,
            "bootstrap_stability": "bootstrap_unstable",
            "series": {
                "delta_auc": {"ci_2_5": 0.01, "ci_97_5": 0.2},
                "delta_pr_auc": {"ci_2_5": 0.01, "ci_97_5": 0.2},
            },
        },
    }
    row = robust.large_comparison_row(condition, "analysis-test")
    assert row["roc_support"] == "bootstrap_supported_positive"
    assert row["pr_support"] == "bootstrap_supported_positive"
    assert row["robustness_status"] == "bootstrap_unstable"


def test_dry_run_performs_no_fit_bootstrap_or_scientific_write(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.write_bytes(b"x")
    reference = {"metrics": {"spatial_cv_config": {"spatial_block_size_cells": 2}}}
    output = tmp_path / "robustness-output"
    with (
        patch.object(robust, "protected_paths", return_value={"frozen": frozen}),
        patch.object(robust, "original_reference_payload", return_value=reference),
        patch.object(robust, "run_oof_condition", side_effect=AssertionError("fit called")),
        patch.object(robust, "paired_large_block_bootstrap", side_effect=AssertionError("bootstrap called")),
    ):
        plan = robust.run_analysis(
            ["manavgat_2021", "bejis_2022"], [10, 20], dry_run=True, output_root=output
        )
    assert plan["fit_or_bootstrap_performed"] is False
    assert plan["files_written"] is False
    assert not output.exists()


def test_direct_runner_dispatches_exact_values():
    with patch.object(direct_runner, "run_analysis", return_value={"ran": False}) as mocked:
        result = direct_runner.main(
            experiments=["manavgat_2021", "bejis_2022"],
            block_sizes_cells=[10, 20],
            dry_run=True,
            force=False,
        )
    assert result == {"ran": False}
    mocked.assert_called_once_with(
        experiments=["manavgat_2021", "bejis_2022"],
        block_sizes=[10, 20],
        dry_run=True,
        force=False,
    )


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ((0.01, 0.2), "bootstrap_supported_positive"),
        ((-0.01, 0.2), "uncertain_interval_includes_zero"),
        ((-0.2, -0.01), "bootstrap_supported_negative"),
    ],
)
def test_interval_interpretation_rules(interval, expected):
    assert robust.classify_interval(interval) == expected


def test_joint_interpretation_rules():
    positive = "bootstrap_supported_positive"
    uncertain = "uncertain_interval_includes_zero"
    negative = "bootstrap_supported_negative"
    assert robust.classify_joint(positive, positive) == "supported_on_both_metrics"
    assert robust.classify_joint(positive, uncertain) == "partial_metric_support"
    assert robust.classify_joint(uncertain, negative) == "not_supported"
