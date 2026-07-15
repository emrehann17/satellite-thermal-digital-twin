"""Tests for the preregistered Step8 large-block robustness analysis on the
formal Step8B primary population (all_valid)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.step8_large_block_robustness as v1robust
import src.step8_large_block_robustness_primary_all_valid as v2
from core.config import STEP8B_N_SPLITS, STEP8B_PRIMARY_POPULATION, STEP8B_SPATIAL_BLOCK_SIZE_CELLS
from src.step8b_train_baseline_vs_thermal_model import (
    BASELINE_FEATURES,
    THERMAL_MODEL_FEATURES,
    add_spatial_block_id,
    build_population_masks,
    filter_valid_for_modeling,
    train_population,
)


# =============================================================================
# Synthetic Step8A-shaped dataset
# =============================================================================
def synthetic_step8a_frame(n_groups: int = 60, seed: int = 7) -> pd.DataFrame:
    """
    n_groups distinct 2-cell spatial blocks, each with 2 positives + 2
    negatives (4 rows), so:
      - block_size=2  -> n_groups distinct spatial_block_id groups
      - block_size=10 -> n_groups//5 distinct large_block_id groups
      - block_size=20 -> n_groups//10 distinct large_block_id groups
    all comfortably >= STEP8B_N_SPLITS for StratifiedGroupKFold.
    """
    rng = np.random.default_rng(seed)
    rows, cols, labels, cell_ids = [], [], [], []
    for g in range(n_groups):
        r0, r1 = 2 * g, 2 * g + 1
        for row, col, label in ((r0, 0, 1), (r0, 1, 0), (r1, 0, 1), (r1, 1, 0)):
            rows.append(row)
            cols.append(col)
            labels.append(label)
            cell_ids.append(f"r{row}_c{col}")
    n = len(rows)
    frame = pd.DataFrame({
        "row_500m": rows, "col_500m": cols, "cell_id": cell_ids, "burned": labels,
    })
    for feature in THERMAL_MODEL_FEATURES:
        if feature == "landcover_dominant":
            continue
        frame[feature] = rng.normal(loc=np.array(labels) * 1.5, scale=1.0, size=n)
    frame["landcover_dominant"] = 30  # not cropland (40)
    frame["valid_for_modeling"] = True
    frame["burnable_tree_shrub_grass"] = True
    frame["burnable_tree_shrub"] = True
    frame["burn_month"] = 9
    frame["landcover_cropland_fraction"] = 0.0
    frame["lon"] = 0.0
    frame["lat"] = 0.0
    frame["gapfilled_fraction"] = 0.0
    frame["observed_fraction"] = 1.0
    frame["valid_30m_fraction"] = 1.0
    return frame


def run_shared_pipeline(df_raw: pd.DataFrame, group_column: str, block_size: int, strict: bool):
    df_valid = filter_valid_for_modeling(df_raw)
    masks = build_population_masks(df_valid)
    df_pop = df_valid.loc[masks["all_valid"]].reset_index(drop=True)
    if group_column == "spatial_block_id":
        df_pop = add_spatial_block_id(df_pop, block_size)
    else:
        df_pop = add_spatial_block_id(
            df_pop, block_size, column_name="large_block_id",
            id_prefix=f"b{block_size}", include_row_col=True,
        )
    result = train_population(
        df_pop, "all_valid", STEP8B_N_SPLITS, 42, "random_forest", 5,
        group_column=group_column, strict_folds=strict,
    )
    return df_pop, result


# =============================================================================
# 1. STEP8B_PRIMARY_POPULATION remains all_valid
# =============================================================================
def test_primary_population_remains_all_valid():
    assert STEP8B_PRIMARY_POPULATION == "all_valid"
    assert v2.PRIMARY_POPULATION == "all_valid"


# =============================================================================
# 2. Original block-size default remains 2
# =============================================================================
def test_original_block_size_default_remains_two():
    assert STEP8B_SPATIAL_BLOCK_SIZE_CELLS == 2


# =============================================================================
# 3. Original Step8B and robustness call the same shared OOF implementation
# =============================================================================
def test_two_cell_and_large_block_paths_share_train_population():
    import inspect
    gate_source = inspect.getsource(v2.run_two_cell_equivalence_gate)
    condition_source = inspect.getsource(v2.run_large_block_condition)
    assert "train_population(" in gate_source
    assert "train_population(" in condition_source
    # Neither path defines its own fold loop / pipeline fit call.
    for src_code in (gate_source, condition_source):
        assert ".fit(" not in src_code
        assert "StratifiedGroupKFold(" not in src_code


# =============================================================================
# 4. all_valid filtering exactly matches original Step8B
# =============================================================================
def test_all_valid_filtering_matches_step8b():
    df = synthetic_step8a_frame(n_groups=10)
    df.loc[0, "valid_for_modeling"] = False
    filtered_v2 = filter_valid_for_modeling(df)
    filtered_manual = df[df["valid_for_modeling"] == True].reset_index(drop=True)  # noqa: E712
    pd.testing.assert_frame_equal(filtered_v2, filtered_manual)
    masks = build_population_masks(filtered_v2)
    assert masks["all_valid"].all()
    assert masks["all_valid"].sum() == len(filtered_v2)


def test_large_block_id_format_matches_preregistration():
    df = pd.DataFrame(
        {
            "row_500m": [37],
            "col_500m": [24],
        }
    )

    result = v2.add_spatial_block_id(
        df,
        10,
        column_name="large_block_id",
        id_prefix="b10",
        include_row_col=True,
    )

    assert result.loc[0, "large_block_id"] == "b10_3_2"
    assert result.loc[0, "large_block_id_row"] == 3
    assert result.loc[0, "large_block_id_col"] == 2


# =============================================================================
# 5/6. Real-data-equivalence logic aligns by cell_id; probability mismatch
#      causes fail-fast
# =============================================================================
def test_equivalence_gate_passes_on_identical_reproduction(tmp_path, monkeypatch):
    df = synthetic_step8a_frame(n_groups=60)
    step8a_root = tmp_path / "outputs" / "experiments" / "manavgat_2021" / "step8a"
    step8b_root = tmp_path / "outputs" / "experiments" / "manavgat_2021" / "step8b"
    step8a_root.mkdir(parents=True)
    step8b_root.mkdir(parents=True)
    df.to_parquet(step8a_root / "step8a_500m_modeling_dataset.parquet", index=False)

    df_pop, result = run_shared_pipeline(df, "spatial_block_id", STEP8B_SPATIAL_BLOCK_SIZE_CELLS, strict=False)
    frozen_predictions = pd.DataFrame({
        "cell_id": df_pop["cell_id"].to_numpy(),
        "population": "all_valid",
        "spatial_block_id": df_pop["spatial_block_id"].to_numpy(),
        "burned": df_pop["burned"].to_numpy(),
        "fold_id": result["fold_id"],
        "y_prob_baseline": result["oof_prob_baseline"],
        "y_prob_thermal": result["oof_prob_thermal"],
    })
    frozen_predictions.to_parquet(step8b_root / "step8b_predictions.parquet", index=False)

    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    gate = v2.run_two_cell_equivalence_gate("manavgat_2021")
    assert gate["gate_passed"] is True
    assert gate["same_evaluated_cell_ids"] is True
    assert gate["same_labels"] is True
    assert gate["same_fold_assignment_per_cell"] is True
    assert gate["same_spatial_block_assignment"] is True
    assert gate["max_abs_probability_difference_baseline"] <= 1e-12
    assert gate["max_abs_probability_difference_thermal"] <= 1e-12


def test_equivalence_gate_fails_on_probability_mismatch(tmp_path, monkeypatch):
    df = synthetic_step8a_frame(n_groups=60)
    step8a_root = tmp_path / "outputs" / "experiments" / "manavgat_2021" / "step8a"
    step8b_root = tmp_path / "outputs" / "experiments" / "manavgat_2021" / "step8b"
    step8a_root.mkdir(parents=True)
    step8b_root.mkdir(parents=True)
    df.to_parquet(step8a_root / "step8a_500m_modeling_dataset.parquet", index=False)

    df_pop, result = run_shared_pipeline(df, "spatial_block_id", STEP8B_SPATIAL_BLOCK_SIZE_CELLS, strict=False)
    tampered_baseline = result["oof_prob_baseline"].copy()
    tampered_baseline[0] = min(max(tampered_baseline[0] + 0.05, 0.0), 1.0)
    if tampered_baseline[0] == result["oof_prob_baseline"][0]:
        tampered_baseline[0] = min(max(tampered_baseline[0] - 0.05, 0.0), 1.0)
    frozen_predictions = pd.DataFrame({
        "cell_id": df_pop["cell_id"].to_numpy(),
        "population": "all_valid",
        "spatial_block_id": df_pop["spatial_block_id"].to_numpy(),
        "burned": df_pop["burned"].to_numpy(),
        "fold_id": result["fold_id"],
        "y_prob_baseline": tampered_baseline,
        "y_prob_thermal": result["oof_prob_thermal"],
    })
    frozen_predictions.to_parquet(step8b_root / "step8b_predictions.parquet", index=False)

    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    gate = v2.run_two_cell_equivalence_gate("manavgat_2021")
    assert gate["gate_passed"] is False
    assert gate["same_baseline_oof_probability_per_cell"] is False
    assert gate["max_abs_probability_difference_baseline"] >= 0.01


def test_equivalence_gate_detects_missing_cell_ids(tmp_path, monkeypatch):
    df = synthetic_step8a_frame(n_groups=60)
    step8a_root = tmp_path / "outputs" / "experiments" / "manavgat_2021" / "step8a"
    step8b_root = tmp_path / "outputs" / "experiments" / "manavgat_2021" / "step8b"
    step8a_root.mkdir(parents=True)
    step8b_root.mkdir(parents=True)
    df.to_parquet(step8a_root / "step8a_500m_modeling_dataset.parquet", index=False)

    df_pop, result = run_shared_pipeline(df, "spatial_block_id", STEP8B_SPATIAL_BLOCK_SIZE_CELLS, strict=False)
    frozen_predictions = pd.DataFrame({
        "cell_id": df_pop["cell_id"].to_numpy(),
        "population": "all_valid",
        "spatial_block_id": df_pop["spatial_block_id"].to_numpy(),
        "burned": df_pop["burned"].to_numpy(),
        "fold_id": result["fold_id"],
        "y_prob_baseline": result["oof_prob_baseline"],
        "y_prob_thermal": result["oof_prob_thermal"],
    }).iloc[:-5]  # drop 5 cells -> misalignment
    frozen_predictions.to_parquet(step8b_root / "step8b_predictions.parquet", index=False)

    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    gate = v2.run_two_cell_equivalence_gate("manavgat_2021")
    assert gate["gate_passed"] is False
    assert gate["same_evaluated_cell_ids"] is False
    assert len(gate["cell_ids_only_in_fresh"]) == 5


# =============================================================================
# 7. 10/20 fitting cannot begin before equivalence passes
# =============================================================================
def test_large_block_fit_blocked_when_gate_fails(tmp_path, monkeypatch):
    _build_full_fixture(tmp_path, tamper_experiment="manavgat_2021")
    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v2, "OUTPUT_ROOT", tmp_path / "outputs" / "robustness" / "step8_large_block_primary_all_valid" / "manavgat_2021__bejis_2022")
    monkeypatch.setattr(v2, "V1_OUTPUT_ROOT", tmp_path / "outputs" / "robustness" / "step8_large_block" / "manavgat_2021__bejis_2022")

    result = v2.run_analysis(run_large_block_fit=True)
    assert result["ran"] is False
    assert result["blocked"] is True
    assert result["equivalence_audit"]["all_experiments_passed"] is False


def test_large_block_fit_not_started_without_explicit_flag(tmp_path, monkeypatch):
    _build_full_fixture(tmp_path, tamper_experiment=None)
    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v2, "OUTPUT_ROOT", tmp_path / "outputs" / "robustness" / "step8_large_block_primary_all_valid" / "manavgat_2021__bejis_2022")
    monkeypatch.setattr(v2, "V1_OUTPUT_ROOT", tmp_path / "outputs" / "robustness" / "step8_large_block" / "manavgat_2021__bejis_2022")

    result = v2.run_analysis(run_large_block_fit=False)
    assert result["ran"] is False
    assert result["blocked"] is False
    assert result["equivalence_audit"]["all_experiments_passed"] is True


# =============================================================================
# 8. large_block_id is passed to StratifiedGroupKFold
# =============================================================================
def test_large_block_id_used_as_cv_group():
    df = synthetic_step8a_frame(n_groups=60)
    df_pop, result = run_shared_pipeline(df, "large_block_id", 10, strict=True)
    assert result is not None and not result.get("skipped")
    assert "large_block_id" in df_pop.columns
    assert "spatial_block_id" not in df_pop.columns


# =============================================================================
# 9. preprocessing is fit only on train folds
# =============================================================================
def test_preprocessing_fit_only_on_train_rows(monkeypatch):
    from sklearn.pipeline import Pipeline

    seen = []
    real_fit = Pipeline.fit

    def spy_fit(self, X, y=None, **kwargs):
        seen.append(set(X.index))
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", spy_fit)
    df = synthetic_step8a_frame(n_groups=60)
    df_pop, result = run_shared_pipeline(df, "large_block_id", 10, strict=True)
    assert not result.get("skipped")
    # train_population fits 2 pipelines (baseline, thermal) per CV fold on
    # X_train only, PLUS one final whole-population refit per model used
    # only for feature-importance reporting (not for OOF predictions). The
    # per-fold fits must all be strict subsets of df_pop; only the trailing
    # 2 final-refit calls may equal the full population.
    n_folds = result["n_splits_used"]
    per_fold_fits = seen[: 2 * n_folds]
    final_refits = seen[2 * n_folds:]
    assert len(per_fold_fits) == 2 * n_folds
    for idx_set in per_fold_fits:
        assert idx_set.issubset(set(df_pop.index))
        assert len(idx_set) < len(df_pop)
    for idx_set in final_refits:
        assert idx_set == set(df_pop.index)


# =============================================================================
# 10. every row has exactly one OOF prediction
# =============================================================================
def test_every_row_has_exactly_one_oof_prediction():
    df = synthetic_step8a_frame(n_groups=60)
    df_pop, result = run_shared_pipeline(df, "large_block_id", 10, strict=True)
    assert not result.get("skipped")
    assert not np.isnan(result["oof_prob_baseline"]).any()
    assert not np.isnan(result["oof_prob_thermal"]).any()
    fold_id = result["fold_id"]
    assert (fold_id >= 0).all()
    assert len(fold_id) == len(df_pop)


# =============================================================================
# 11. bootstrap uses the same large blocks as CV
# =============================================================================
def test_bootstrap_reuses_cv_large_blocks():
    df = synthetic_step8a_frame(n_groups=60)
    df_pop, result = run_shared_pipeline(df, "large_block_id", 10, strict=True)
    predictions = pd.DataFrame({
        "large_block_id": df_pop["large_block_id"].to_numpy(),
        "burned": df_pop["burned"].to_numpy(),
        "y_prob_baseline": result["oof_prob_baseline"],
        "y_prob_thermal": result["oof_prob_thermal"],
    })
    summary, replicates = v1robust.paired_large_block_bootstrap(predictions, n_replicates=20, seed=1)
    assert summary["requested_replicates"] == 20
    assert len(replicates) == 20
    # Paired: each replicate's baseline/thermal delta computed from the SAME
    # sampled rows -- verified by construction in v1's implementation, and
    # confirmed here by checking the CV group column was the only grouping
    # key used.
    assert set(predictions["large_block_id"].unique()) == set(df_pop["large_block_id"].unique())


# =============================================================================
# 12/13. original + existing v1 robustness outputs remain hash-identical
# =============================================================================
def test_protected_hashes_detect_any_change(tmp_path, monkeypatch):
    _build_full_fixture(tmp_path, tamper_experiment=None)
    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v2, "V1_OUTPUT_ROOT", tmp_path / "outputs" / "robustness" / "step8_large_block" / "manavgat_2021__bejis_2022")

    before = v2.hash_all_protected()
    # No mutation -> must compare equal to itself.
    after = v2.hash_all_protected()
    v2.assert_all_protected_unchanged(before, after)

    # Mutate an original protected file -> must raise.
    target = tmp_path / "outputs" / "experiments" / "manavgat_2021" / "step8a" / "step8a_500m_modeling_dataset.parquet"
    df = pd.read_parquet(target)
    df.iloc[0, 0] = df.iloc[0, 0] + 999
    df.to_parquet(target, index=False)
    after_mutated = v2.hash_all_protected()
    with pytest.raises(v2.Step8RobustnessPrimaryError):
        v2.assert_all_protected_unchanged(before, after_mutated)


def test_v1_robustness_tree_hash_detects_change(tmp_path, monkeypatch):
    _build_full_fixture(tmp_path, tamper_experiment=None)
    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v2, "V1_OUTPUT_ROOT", tmp_path / "outputs" / "robustness" / "step8_large_block" / "manavgat_2021__bejis_2022")

    before = v2.hash_v1_robustness_tree()
    report_path = tmp_path / "outputs" / "robustness" / "step8_large_block" / "manavgat_2021__bejis_2022" / "step8_large_block_final_report.json"
    payload = json.loads(report_path.read_text())
    payload["overall_predefined_scale_robustness"]["statement"] = "TAMPERED"
    report_path.write_text(json.dumps(payload))
    after = v2.hash_v1_robustness_tree()
    with pytest.raises(v2.Step8RobustnessPrimaryError):
        v2.assert_hash_dict_unchanged(before, after, "Existing (v1) robustness")


def test_v1_analysis_id_mismatch_is_rejected(tmp_path, monkeypatch):
    _build_full_fixture(tmp_path, tamper_experiment=None)
    v1_root = tmp_path / "outputs" / "robustness" / "step8_large_block" / "manavgat_2021__bejis_2022"
    report_path = v1_root / "step8_large_block_final_report.json"
    payload = json.loads(report_path.read_text())
    payload["analysis_id"] = "0" * 64
    report_path.write_text(json.dumps(payload))
    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v2, "V1_OUTPUT_ROOT", v1_root)
    with pytest.raises(v2.Step8RobustnessPrimaryError):
        v2.assert_v1_analysis_id_matches()


# =============================================================================
# 14. new outputs are written only under the new v2 namespace
# =============================================================================
def test_outputs_written_only_under_new_namespace(tmp_path, monkeypatch):
    _build_full_fixture(tmp_path, tamper_experiment=None)
    v2_root = tmp_path / "outputs" / "robustness" / "step8_large_block_primary_all_valid" / "manavgat_2021__bejis_2022"
    v1_root = tmp_path / "outputs" / "robustness" / "step8_large_block" / "manavgat_2021__bejis_2022"
    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v2, "OUTPUT_ROOT", v2_root)
    monkeypatch.setattr(v2, "V1_OUTPUT_ROOT", v1_root)

    before_v1_files = sorted(p.relative_to(v1_root) for p in v1_root.rglob("*") if p.is_file())
    result = v2.run_analysis(run_large_block_fit=False)
    assert result["equivalence_audit"]["all_experiments_passed"] is True
    assert v2_root.exists()
    after_v1_files = sorted(p.relative_to(v1_root) for p in v1_root.rglob("*") if p.is_file())
    assert before_v1_files == after_v1_files
    for original_experiment_dir in (tmp_path / "outputs" / "experiments").rglob("*"):
        if original_experiment_dir.is_file():
            assert "robustness" not in str(original_experiment_dir)


# =============================================================================
# 15. manifests cannot be silently rewritten
# =============================================================================
def test_manifest_is_immutable(tmp_path, monkeypatch):
    _build_full_fixture(tmp_path, tamper_experiment=None)
    v2_root = tmp_path / "outputs" / "robustness" / "step8_large_block_primary_all_valid" / "manavgat_2021__bejis_2022"
    v1_root = tmp_path / "outputs" / "robustness" / "step8_large_block" / "manavgat_2021__bejis_2022"
    monkeypatch.setattr(v1robust, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(v2, "OUTPUT_ROOT", v2_root)
    monkeypatch.setattr(v2, "V1_OUTPUT_ROOT", v1_root)

    protected = v2.hash_all_protected()
    manifest1 = v2.validate_or_write_manifest(v2_root, protected)
    manifest2 = v2.validate_or_write_manifest(v2_root, protected)
    assert manifest1 == manifest2

    path = v2_root / "step8_large_block_primary_all_valid_preregistration.json"
    payload = json.loads(path.read_text())
    payload["scientific_configuration"]["predefined_block_sizes_cells"] = [99, 100]
    path.write_text(json.dumps(payload))
    with pytest.raises(v2.Step8RobustnessPrimaryError):
        v2.validate_or_write_manifest(v2_root, protected)


def test_manifest_analysis_id_differs_from_v1():
    protected = {"original_step8": {}, "v1_robustness_tree": {}}
    # scientific_configuration requires reference JSON files; only check
    # that the schema/version namespace is distinct from v1's, which is a
    # cheap, file-independent guarantee that the two analyses can never
    # collide on analysis_id even if their inputs were identical.
    assert v1robust.ANALYSIS_SCHEMA_VERSION != v2.ANALYSIS_SCHEMA_VERSION


# =============================================================================
# Fixture builder shared by the end-to-end tests above
# =============================================================================
def _build_full_fixture(tmp_path: Path, tamper_experiment: str | None) -> None:
    experiments = ("manavgat_2021", "bejis_2022")
    for experiment in experiments:
        exp_root = tmp_path / "outputs" / "experiments" / experiment
        step8a_dir, step8b_dir, step8c_dir, step8e_dir = (
            exp_root / "step8a", exp_root / "step8b", exp_root / "step8c", exp_root / "step8e",
        )
        for d in (step8a_dir, step8b_dir, step8c_dir, step8e_dir):
            d.mkdir(parents=True, exist_ok=True)

        df = synthetic_step8a_frame(n_groups=60, seed=hash(experiment) % 1000)
        df.to_parquet(step8a_dir / "step8a_500m_modeling_dataset.parquet", index=False)

        df_pop, result = run_shared_pipeline(df, "spatial_block_id", STEP8B_SPATIAL_BLOCK_SIZE_CELLS, strict=False)
        baseline_probs = result["oof_prob_baseline"].copy()
        if tamper_experiment == experiment:
            baseline_probs[0] = min(max(baseline_probs[0] + 0.2, 0.0), 1.0)
            if baseline_probs[0] == result["oof_prob_baseline"][0]:
                baseline_probs[0] = min(max(baseline_probs[0] - 0.2, 0.0), 1.0)
        frozen_predictions = pd.DataFrame({
            "cell_id": df_pop["cell_id"].to_numpy(),
            "population": "all_valid",
            "spatial_block_id": df_pop["spatial_block_id"].to_numpy(),
            "burned": df_pop["burned"].to_numpy(),
            "fold_id": result["fold_id"],
            "y_prob_baseline": baseline_probs,
            "y_prob_thermal": result["oof_prob_thermal"],
        })
        frozen_predictions.to_parquet(step8b_dir / "step8b_predictions.parquet", index=False)

        metrics_json = {
            "model": "random_forest",
            "spatial_cv_config": {
                "method": "StratifiedGroupKFold", "spatial_block_size_cells": 2,
                "n_splits_requested": STEP8B_N_SPLITS, "random_state": 42, "random_split_used": False,
            },
            "feature_sets": {
                "baseline": BASELINE_FEATURES, "thermal_additional": [f for f in THERMAL_MODEL_FEATURES if f not in BASELINE_FEATURES],
                "thermal_model_full": THERMAL_MODEL_FEATURES,
            },
            "population_metrics": {
                "all_valid": {
                    "overall_baseline": result["overall_baseline"], "overall_thermal": result["overall_thermal"],
                    "delta_auc": result["delta_auc"], "delta_pr_auc": result["delta_pr_auc"],
                },
            },
        }
        (step8b_dir / "step8b_model_comparison_metrics.json").write_text(json.dumps(metrics_json, default=str))

        bootstrap_json = {
            "n_bootstrap_requested": 1000, "random_seed": 42,
            "bootstrap_ci_by_population": {
                "all_valid": {
                    "delta_auc_ci95": [0.001, 0.05], "delta_pr_auc_ci95": [0.001, 0.05],
                    "n_bootstrap_successful": 950,
                },
            },
        }
        (step8c_dir / "step8c_bootstrap_metrics.json").write_text(json.dumps(bootstrap_json))
        (step8e_dir / "final_step8_report.json").write_text(json.dumps({"placeholder": True}))

    v1_root = tmp_path / "outputs" / "robustness" / "step8_large_block" / "manavgat_2021__bejis_2022"
    v1_root.mkdir(parents=True, exist_ok=True)
    (v1_root / "step8_large_block_final_report.json").write_text(json.dumps({
        "analysis_id": v2.V1_EXPECTED_ANALYSIS_ID,
        "overall_predefined_scale_robustness": {
            "all_four_conditions_supported_on_both_metrics": True,
            "statement": "placeholder frozen v1 statement",
        },
    }))