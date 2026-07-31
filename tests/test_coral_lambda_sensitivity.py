from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.coral_lambda_sensitivity as cls
from scripts import validate_coral_lambda_sensitivity as validator


def test_deterministic_analysis_id_and_grid():
    assert cls.compute_analysis_id() == cls.compute_analysis_id()
    assert tuple(cls.lambda_grid_frame().lambda_value) == cls.LAMBDA_GRID
    assert tuple(cls.lambda_grid_frame().lambda_token) == cls.LAMBDA_TOKEN_SEQUENCE
    assert cls.lambda_grid_frame().iloc[4].is_canonical
    assert cls.lambda_token(1e-5) == "lambda_1e_m5"


def test_hash_gate_is_fail_closed(tmp_path):
    paths = {}
    for name in cls.PRIMARY_EXPERIMENTS:
        path = tmp_path / name; path.write_bytes(b"wrong"); paths[name] = path
    with pytest.raises(cls.CoralLambdaSensitivityError):
        cls.assert_canonical_step8a_hashes(paths)


def test_directional_resolver_rejects_reverse(tmp_path):
    selected = tmp_path / "cross_region/a__b/step10"; selected.mkdir(parents=True)
    reverse = tmp_path / "cross_region/b__a/step10"; reverse.mkdir(parents=True)
    (selected / "x").write_text("selected"); (reverse / "x").write_text("reverse")
    monkey_directions = cls.DIRECTIONS
    try:
        cls.DIRECTIONS = monkey_directions + ("a_to_b",)
        result = cls.resolve_step10_reference("a", "b", tmp_path)
    finally:
        cls.DIRECTIONS = monkey_directions
    assert result["selected"]["path"].endswith("a__b/step10")
    assert result["rejected_duplicate"]["reason"] == "rejected_duplicate"


def test_exact_coral_ddof_zero_source_only_transform():
    xs = np.array([[0., 1.], [1., -1.], [-1., 0.], [2., 2.]])
    xt = np.array([[2., 0.], [-2., 1.], [0., -1.], [1., 2.]])
    result = cls.coral_cell(xs, xt, 1e-4)
    expected_cs = np.cov(xs, rowvar=False, ddof=0) + 1e-4 * np.eye(2)
    expected_ct = np.cov(xt, rowvar=False, ddof=0) + 1e-4 * np.eye(2)
    expected_a = cls._sym_matrix_power(expected_cs, -.5) @ cls._sym_matrix_power(expected_ct, .5)
    np.testing.assert_allclose(result["Cs"], expected_cs)
    np.testing.assert_allclose(result["Ct"], expected_ct)
    np.testing.assert_allclose(result["A"], expected_a)
    np.testing.assert_array_equal(result["Xt_model"], xt)
    np.testing.assert_allclose(result["Xs_coral"], xs @ expected_a)


def test_label_firewall_and_label_independence():
    features = cls.FEATURE_LISTS["baseline"]
    source = pd.DataFrame({features[0]: [1., 2., 3.], features[1]: [4., 6., 9.],
                           features[2]: [0., 2., 1.], features[3]: [1, 2, 1]})
    target = source.iloc[::-1].reset_index(drop=True)
    a = cls.zscore_pair(source, target, "baseline")
    target_with_label = target.assign(burned=[0, 1, 0])
    with pytest.raises(SystemExit): cls.zscore_pair(source, target_with_label, "baseline")
    np.testing.assert_array_equal(a[1][features].to_numpy(), cls.zscore_pair(source, target, "baseline")[1][features].to_numpy())


def test_zero_finite_and_floor_required_cases():
    rng = np.random.default_rng(4)
    finite = cls.coral_cell(rng.normal(size=(30, 3)), rng.normal(size=(25, 3)), 0)
    assert finite["diagnostics"]["numerical_status"] == "pass"
    singular = cls.coral_cell(np.ones((8, 3)), np.ones((9, 3)), 0)
    assert singular["diagnostics"]["numerical_status"] == "singular_unregularised_covariance"
    assert singular["Xs_coral"] is None
    assert not singular["diagnostics"]["eigenvalue_floor_applied"]


def test_positive_lambda_diagnostics_and_nonfinite_retention():
    singular = cls.coral_cell(np.ones((8, 3)), np.ones((9, 3)), 1e-4)
    assert singular["diagnostics"]["numerical_status"] == "pass"
    assert singular["diagnostics"]["source_covariance_shape"] == [3, 3]
    bad = cls.coral_cell(np.array([[np.nan, 1.], [2., 3.]]), np.ones((3, 2)), 1e-4)
    assert bad["diagnostics"]["numerical_status"] in {"eigenvalue_floor_required", "nonfinite_matrix_transform"}


def test_metric_gates_brier_provenance_and_tolerances():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .3, .7, .9])
    metrics = cls.compute_all_metrics(y, p)
    tier1 = cls.run_tier1_gate(y, p, metrics)
    assert all(r["gate_status"] == "pass" for r in tier1)
    brier = next(r for r in tier1 if r["metric"] == "brier_score")
    assert brier["reference_origin"] == "recomputed_from_persisted_probabilities"
    assert cls.tier2_metric_tolerance("pr_auc", y) == 1e-6
    assert cls.tier2_metric_tolerance("roc_auc", y) == max(1e-6, 8 / 4)
    assert cls.tier2_metric_tolerance("brier_score", y) == 1e-12


def test_fit_accounting_and_no_cross_target_identity_reuse():
    identities = cls.expected_fit_identities()
    assert len(identities) == 72
    assert cls.verify_complete_fit_identities(list(identities))
    targets = {d.split("_to_")[1] for d, _, _ in identities}
    assert targets == set(cls.PRIMARY_EXPERIMENTS)


def test_paired_bootstrap_1000_and_brier_orientation():
    frame = pd.DataFrame({"spatial_block_id": np.repeat(np.arange(10), 2),
                          "burned": np.tile([0, 1], 10),
                          "p1": np.tile([.2, .8], 10), "p2": np.tile([.3, .7], 10)})
    result = cls.paired_bootstrap(frame, {"lambda_0": "p1", "lambda_1e_m5": "p2"})
    assert result["n_requested"] == 1000 and len(result["replicates_df"]) == 1000
    assert result["replicates_df"].draw_hash.notna().all()
    assert cls.oriented_delta("brier_score", .1, .2) == pytest.approx(.1)
    assert cls.natural_delta("brier_score", .1, .2) == pytest.approx(-.1)


@pytest.mark.parametrize("metric,value,token", [
    ("roc_auc", .005, "insensitive_over_grid"), ("pr_auc", .006, "modest_lambda_sensitivity"),
    ("roc_auc", .021, "material_lambda_sensitivity"), ("brier_score", .001, "insensitive_over_grid"),
    ("brier_score", .002, "modest_lambda_sensitivity"), ("brier_score", .006, "material_lambda_sensitivity")])
def test_magnitude_tokens(metric, value, token):
    assert cls.sensitivity_token(metric, value) == token


def test_support_tokens():
    assert cls.bootstrap_delta_summary([2, 3], [1, 1], "roc_auc")["support_token"] == "bootstrap_supported_positive"
    assert cls.bootstrap_delta_summary([1, 1], [2, 3], "roc_auc")["support_token"] == "bootstrap_supported_negative"
    assert cls.bootstrap_delta_summary([0, 2], [1, 1], "roc_auc")["support_token"] == "interval_includes_zero"
    assert cls.bootstrap_delta_summary([], [], "roc_auc", numerical_failure=True)["support_token"] == "unavailable_due_to_numerical_failure"


def test_dry_run_no_writes_and_partial_resume_rejected(tmp_path):
    result = cls.run_analysis(from_stage="plan", to_stage="summarize", dry_run=True, output_root=tmp_path)
    assert not Path(result["analysis_root"]).exists()
    root = Path(result["analysis_root"]); root.mkdir(parents=True)
    cls.write_stage_marker(root, "fit", {"analysis_id": root.name, "fit_identities": []})
    assert not cls.stage_is_reusable(root, "fit")


def test_force_quarantine_preserves_files(tmp_path):
    root = tmp_path / "diagnostics/coral_lambda_sensitivity/id"; root.mkdir(parents=True)
    (root / "evidence").write_text("x")
    destination = cls.quarantine_namespace(root)
    assert destination is not None and (destination / "evidence").read_text() == "x" and not root.exists()


def test_source_contains_no_gee_or_excluded_result_scope():
    text = Path(cls.__file__).read_text().lower()
    assert "import ee" not in text and "gee_utils" not in text and "earthengine" not in text
    assert all("evia" not in d for d in cls.DIRECTIONS)
    assert all("manavgat_2021_to_bejis_2022" != d for d in cls.DIRECTIONS)


def test_validator_dry_run_nonexistent_root_passes_without_writes(tmp_path, capsys):
    root = tmp_path / cls.compute_analysis_id()
    assert validator.main([str(root), "--dry-run"]) == 0
    stdout = capsys.readouterr().out
    assert stdout.strip()
    assert '"status": "PASS"' in stdout
    assert "OVERALL STATUS: PASS" in stdout
    assert not root.exists()


def test_validator_dry_run_wrong_analysis_id_fails_without_writes(tmp_path, capsys):
    root = tmp_path / "wrong-analysis-id"
    assert validator.main([str(root), "--dry-run"]) == 1
    stdout = capsys.readouterr().out
    assert '"status": "FAIL"' in stdout
    assert "OVERALL STATUS: FAIL" in stdout
    assert not root.exists()


def test_validator_dry_run_does_not_require_config(tmp_path):
    root = tmp_path / cls.compute_analysis_id()
    report = validator.validate_dry_run(root)
    assert report["status"] == "PASS"
    assert not root.exists() and not (root / "config.json").exists()


def test_validator_actual_missing_config_is_visible_failure(tmp_path, capsys):
    root = tmp_path / "missing"
    assert validator.main([str(root)]) == 1
    stdout = capsys.readouterr().out
    assert '"check_id": "config_readable"' in stdout
    assert "OVERALL STATUS: FAIL" in stdout
    assert not root.exists()


def test_validator_actual_pass_fixture_prints_json_and_status(tmp_path, capsys):
    sci = cls.scientific_config()
    root = tmp_path / cls.compute_analysis_id(sci)
    root.mkdir()
    (root / "config.json").write_text(__import__("json").dumps({
        "schema_version": cls.SCHEMA_VERSION,
        "diagnostic_class": cls.DIAGNOSTIC_CLASS,
        "scientific_config": sci,
    }))
    assert validator.main([str(root)]) == 0
    stdout = capsys.readouterr().out
    assert '"status": "PASS"' in stdout
    assert "OVERALL STATUS: PASS" in stdout
    assert (root / "validation_report.json").is_file()


def _synthetic_stage_inputs():
    frames = {}
    thermal = cls.FEATURE_LISTS["thermal"]
    for offset, experiment in enumerate(cls.PRIMARY_EXPERIMENTS):
        n = 8
        data = {feature: np.arange(n, dtype=float) + offset + index / 10
                for index, feature in enumerate(thermal) if feature not in cls.CATEGORICAL_FEATURES}
        for feature in cls.CATEGORICAL_FEATURES: data[feature] = np.tile([1, 2], n // 2)
        data.update({"cell_id": [f"{experiment}_{i}" for i in range(n)],
                     "burned": np.tile([0, 1], n // 2),
                     "valid_for_modeling": True, cls.PRIMARY_POPULATION: True})
        frames[experiment] = pd.DataFrame(data)
    references = {}
    for direction in cls.DIRECTIONS:
        _, target_id = cls.split_direction(direction); target = frames[target_id]
        probability = np.where(target.burned.to_numpy() == 1, .8, .2)
        prediction_rows, metric_rows = [], []
        for family in cls.MODEL_FAMILIES:
            for method in ("raw_source_only", "regionwise_zscore", "coral_after_regionwise_zscore"):
                prediction_rows.extend({"direction": direction, "population": cls.PRIMARY_POPULATION,
                    "model_family": family,
                    "adaptation_method": method, "target_cell_id": cell,
                    "target_spatial_block_id": f"block_{i}", "prediction_probability": prob}
                    for i, (cell, prob) in enumerate(zip(target.cell_id, probability)))
                scores = cls.compute_all_metrics(target.burned, probability)
                metric_rows.append({"direction": direction, "model_family": family, "method": method,
                                    "roc_auc": scores["roc_auc"], "pr_auc": scores["pr_auc"]})
        references[direction] = {"predictions": pd.DataFrame(prediction_rows),
                                 "metrics": pd.DataFrame(metric_rows), "inventory": {}}
    return {"frames": frames, "references": references}


def _fake_fit_predictor(call_log):
    def run(source_z, target_z, y_source, family, lambda_):
        call_log.append(lambda_)
        probability = np.tile([.2, .8], len(target_z) // 2)
        return {"probabilities": probability, "diagnostics": {"numerical_status": "pass",
                "probabilities_finite": True}}
    return run


def test_concrete_fit_runner_gate_first_and_exact_accounting(tmp_path):
    plan = cls.run_analysis(from_stage="plan", to_stage="plan", output_root=tmp_path)
    root = Path(plan["analysis_root"]); calls = []
    result = cls.run_fit_stage(root, _synthetic_stage_inputs(), fit_predictor=_fake_fit_predictor(calls))
    assert calls[:8] == [cls.CANONICAL_LAMBDA] * 8
    assert len(calls) == 72
    assert result["scientific_fits"] == 72 and result["audit_fit_executions"] == 8
    marker = __import__("json").loads((root / "stages/fit.json").read_text())
    assert cls.verify_complete_fit_identities([tuple(x) for x in marker["fit_identities"]])
    partitions = pd.read_parquet(root / "predictions.parquet")
    for _, group in partitions.groupby(["direction", "model_family"]):
        mappings = [partitions[(partitions.direction == group.direction.iloc[0])
                    & (partitions.model_family == group.model_family.iloc[0])
                    & (partitions.lambda_token == token)][["target_cell_id", "target_spatial_block_id"]]
                    .sort_values("target_cell_id").reset_index(drop=True)
                    for token in cls.LAMBDA_TOKEN_SEQUENCE]
        assert all(mapping.equals(mappings[0]) for mapping in mappings[1:])


def test_fit_without_passing_plan_is_rejected(tmp_path):
    root = tmp_path / "diagnostics" / cls.DIAGNOSTIC_NAMESPACE / cls.compute_analysis_id()
    with pytest.raises(cls.CoralLambdaSensitivityError):
        cls.run_fit_stage(root, _synthetic_stage_inputs(), fit_predictor=_fake_fit_predictor([]))
    assert not root.exists()


def test_gate_failure_writes_no_partial_scientific_result(tmp_path):
    plan = cls.run_analysis(from_stage="plan", to_stage="plan", output_root=tmp_path)
    root = Path(plan["analysis_root"]); inputs = _synthetic_stage_inputs()
    inputs["references"][cls.DIRECTIONS[0]]["metrics"].loc[:, "roc_auc"] = 0.0
    with pytest.raises(cls.CoralLambdaSensitivityError):
        cls.run_fit_stage(root, inputs, fit_predictor=_fake_fit_predictor([]))
    assert not (root / "stages/fit.json").exists()
    assert not (root / "metrics.csv").exists()
    assert not (root / "predictions.parquet").exists()


def test_run_analysis_actual_fit_uses_default_concrete_wiring(tmp_path, monkeypatch):
    cls.run_analysis(from_stage="plan", to_stage="plan", output_root=tmp_path)
    seen = {}
    monkeypatch.setattr(cls, "_load_production_inputs", lambda **kwargs: {"resolved": True})
    monkeypatch.setattr(cls, "run_fit_stage", lambda root, inputs: seen.update(inputs) or {"scientific_fits": 72})
    result = cls.run_analysis(from_stage="fit", to_stage="fit", resume=True, output_root=tmp_path)
    assert seen == {"resolved": True}
    assert result["stage_results"]["fit"]["scientific_fits"] == 72


def test_runner_exception_leaves_no_passing_marker(tmp_path):
    cls.run_analysis(from_stage="plan", to_stage="plan", output_root=tmp_path)
    def fail(root, inputs): raise RuntimeError("synthetic runner failure")
    with pytest.raises(RuntimeError):
        cls.run_analysis(from_stage="fit", to_stage="fit", resume=True, output_root=tmp_path,
                         input_loader=lambda **kwargs: {}, stage_runners={"fit": fail})
    root = cls.analysis_root(tmp_path)
    assert not (root / "stages/fit.json").exists()


def test_bootstrap_and_summarize_dispatch_report_zero_fits(tmp_path):
    root = cls.analysis_root(tmp_path); root.mkdir(parents=True)
    cls.atomic_write_json(root / "config.json", {"analysis_id": root.name,
        "scientific_config": cls.scientific_config()}, root)
    cls.atomic_write_csv(root / "lambda_grid.csv", cls.lambda_grid_frame(), root)
    cls.write_stage_marker(root, "plan", {"analysis_id": root.name})
    partition_hashes = {}
    for direction, family, token in cls.expected_fit_identities():
        path = root / "predictions.parquet" / f"direction={direction}" / f"model_family={family}" / f"lambda_token={token}" / "part.parquet"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"synthetic")
        partition_hashes[str(path.relative_to(root))] = cls.sha256_file(path)
    cls.write_stage_marker(root, "fit", {"analysis_id": root.name,
        "fit_identities": [list(x) for x in cls.expected_fit_identities()],
        "prediction_partition_hashes": partition_hashes})
    bootstrap_seen = {}
    def bootstrap(root, inputs): bootstrap_seen.update(inputs); cls.write_stage_marker(root, "bootstrap", {"analysis_id": root.name}); return {"model_fits": 0}
    result = cls.run_analysis(from_stage="bootstrap", to_stage="summarize", resume=True,
        output_root=tmp_path, input_loader=lambda **kwargs: {"persisted_predictions": True},
        stage_runners={"bootstrap": bootstrap, "summarize": lambda root: {"model_fits": 0, "bootstrap_runs": 0}})
    assert bootstrap_seen == {"persisted_predictions": True}
    assert result["stage_results"]["bootstrap"]["model_fits"] == 0
    assert result["stage_results"]["summarize"] == {"model_fits": 0, "bootstrap_runs": 0}


def test_source_model_population_does_not_require_spatial_block():
    frame = _synthetic_stage_inputs()["frames"][cls.PRIMARY_EXPERIMENTS[0]]
    assert "spatial_block_id" not in frame
    source = cls._source_model_population(frame)
    assert len(source) == len(frame) and "burned" in source


def test_persisted_target_block_mapping_exact_cell_join_without_step8a_block():
    inputs = _synthetic_stage_inputs(); direction = cls.DIRECTIONS[0]
    _, target_id = cls.split_direction(direction); frame = inputs["frames"][target_id]
    assert "spatial_block_id" not in frame
    mapping = cls.resolve_target_block_mapping(frame, inputs["references"][direction]["predictions"], direction)
    assert list(mapping.cell_id) == list(frame.cell_id)
    assert mapping.spatial_block_id.notna().all()


@pytest.mark.parametrize("mutation", ["missing", "extra", "null", "conflict"])
def test_target_block_mapping_rejects_bad_coverage_or_mapping(mutation):
    inputs = _synthetic_stage_inputs(); direction = cls.DIRECTIONS[0]
    _, target_id = cls.split_direction(direction); frame = inputs["frames"][target_id]
    predictions = inputs["references"][direction]["predictions"].copy()
    if mutation == "missing": predictions = predictions[predictions.target_cell_id != frame.cell_id.iloc[0]]
    elif mutation == "extra":
        row = predictions.iloc[[0]].copy(); row["target_cell_id"] = "extra_cell"; predictions = pd.concat([predictions, row])
    elif mutation == "null": predictions.loc[predictions.index[0], "target_spatial_block_id"] = None
    else:
        row = predictions.iloc[[0]].copy(); row["target_spatial_block_id"] = "conflicting"; predictions = pd.concat([predictions, row])
    with pytest.raises(cls.CoralLambdaSensitivityError):
        cls.resolve_target_block_mapping(frame, predictions, direction)


def test_same_target_directions_reuse_exact_mapping():
    inputs = _synthetic_stage_inputs()
    directions = [d for d in cls.DIRECTIONS if d.endswith("_to_mugla_2021")]
    mappings = []
    for direction in directions:
        _, target_id = cls.split_direction(direction)
        mappings.append(cls.resolve_target_block_mapping(inputs["frames"][target_id],
            inputs["references"][direction]["predictions"], direction))
    cls._assert_same_target_mapping(mappings[0], mappings[1], "mugla_2021")


def test_target_label_is_absent_from_adaptation_frame():
    frame = _synthetic_stage_inputs()["frames"][cls.PRIMARY_EXPERIMENTS[0]]
    target = cls._target_model_population(frame)
    assert "burned" not in target.columns
    cls.assert_label_blind(target, "test")


def test_no_unrelated_large_block_helper_is_reused():
    source = Path(cls.__file__).read_text()
    assert "assign_large_blocks" not in source
    assert "add_spatial_block_id(" not in source


def _evia_validator_fixture(tmp_path, *, excluded=True):
    sci = cls.scientific_config()
    if excluded: sci["excluded_experiments"] = {"evia_2021_extended": "excluded from scientific scope"}
    root = tmp_path / cls.compute_analysis_id(sci); root.mkdir()
    (root / "config.json").write_text(__import__("json").dumps({
        "schema_version": cls.SCHEMA_VERSION, "diagnostic_class": cls.DIAGNOSTIC_CLASS,
        "scientific_config": sci}))
    return root, sci


def test_evia_exclusion_metadata_and_report_text_are_allowed(tmp_path):
    root, sci = _evia_validator_fixture(tmp_path)
    (root / "report.md").write_text("Evia was excluded from the scientific cohort.\n")
    (root / "repository_inventory.json").write_text(__import__("json").dumps({
        "context_only": ["evia_2021_extended"]}))
    passed, evidence = validator.check_no_evia_result(root, sci)
    assert passed and evidence["violations"] == []


def test_evia_direction_in_metrics_is_rejected(tmp_path):
    root, sci = _evia_validator_fixture(tmp_path)
    pd.DataFrame({"direction": ["evia_2021_extended_to_mugla_2021"], "metric": ["roc_auc"]}).to_csv(root / "metrics.csv", index=False)
    passed, evidence = validator.check_no_evia_result(root, sci)
    assert not passed and evidence["violations"][0]["column"] == "direction"


def test_evia_source_or_target_in_prediction_partition_is_rejected(tmp_path):
    root, sci = _evia_validator_fixture(tmp_path)
    leaf = root / "predictions.parquet/direction=bejis_2022_to_mugla_2021/model_family=baseline/lambda_token=lambda_0"
    leaf.mkdir(parents=True)
    pd.DataFrame({"source_experiment": ["evia_2021"], "target_experiment": ["mugla_2021"],
                  "prediction_probability": [.5]}).to_parquet(leaf / "part.parquet", index=False)
    passed, evidence = validator.check_no_evia_result(root, sci)
    assert not passed
    assert any(item.get("column") == "source_experiment" for item in evidence["violations"])


def test_evia_direction_in_sensitivity_summary_is_rejected(tmp_path):
    root, sci = _evia_validator_fixture(tmp_path)
    pd.DataFrame({"direction": ["mugla_2021_to_evia_2021_extended"]}).to_csv(root / "sensitivity_summary.csv", index=False)
    passed, _ = validator.check_no_evia_result(root, sci)
    assert not passed


def test_current_artifact_structural_evia_check_passes():
    root = Path("outputs/diagnostics/coral_lambda_sensitivity/b74d643edc359e62213f4b8fc26f128512de60ed3f2127c317722d6b2d27d17a")
    assert root.exists(), "canonical local artifact required by this targeted validator test"
    sci = __import__("json").loads((root / "config.json").read_text())["scientific_config"]
    passed, evidence = validator.check_no_evia_result(root, sci)
    assert passed, evidence


def test_partial_fit_output_without_marker_is_not_resumed(tmp_path):
    result = cls.run_analysis(from_stage="plan", to_stage="plan", output_root=tmp_path)
    root = Path(result["analysis_root"]); (root / "metrics.csv").write_text("partial")
    with pytest.raises(cls.CoralLambdaSensitivityError):
        cls.run_analysis(from_stage="fit", to_stage="fit", resume=True, output_root=tmp_path,
                         input_loader=lambda **kwargs: {})
    assert not (root / "stages/fit.json").exists()
