"""Regression tests for the generic multi-experiment Step9G univariate-AUC
comparison (src/step9g_multi_aoi_comparison/, scripts/main.py
`concept-shift-compare`).

Uses entirely synthetic/placeholder experiment IDs and hand-built JSON
fixtures (matching this repo's existing convention), redirecting
PROJECT_ROOT so nothing here touches the real repo output tree."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.step9g_univariate_feature_auc_direction_reversal as step9g
from src.step9g_multi_aoi_comparison import build as build_mod
from src.step9g_multi_aoi_comparison.consistency import ConsistencyError
from src.step9g_multi_aoi_comparison.parse import ScientificContractError
from scripts.main import build_parser, cmd_concept_shift_compare

FAKE_A = "aoi_alpha_2099"
FAKE_B = "aoi_beta_2099"
FAKE_C = "aoi_gamma_2099"
FAKE_FUTURE = "aoi_future_2099"

NUMERIC_FEATURES = step9g.NUMERIC_FEATURES


def _prereg(**overrides) -> dict:
    config = {
        "block_size_cells": 10,
        "nominal_block_scale": "approximately_5_km",
        "bootstrap": {"replicates": 1000, "seed": 42},
        "primary_population": "burnable_tree_shrub_grass",
    }
    config.update(overrides)
    return {"analysis_id": "prereg_id", "scientific_configuration": config}


def make_row(
    feature: str, a_id: str, b_id: str,
    a_auc: float = 0.55, a_ci=(0.5, 0.6), a_dir="higher_values_rank_burned", a_support="bootstrap_supported_higher_values_rank_burned",
    b_auc: float = 0.56, b_ci=(0.51, 0.61), b_dir="higher_values_rank_burned", b_support="bootstrap_supported_higher_values_rank_burned",
    reversal_status="no_direction_reversal", point_reversal=False, step9e_flag=False,
    source_id=None, target_id=None,
) -> dict:
    source_id = source_id or a_id
    target_id = target_id or b_id
    return {
        "feature": feature,
        "source_experiment_id": source_id, "target_experiment_id": target_id,
        f"{a_id}_auc": a_auc, f"{a_id}_ci_low": a_ci[0], f"{a_id}_ci_high": a_ci[1],
        f"{a_id}_direction": a_dir, f"{a_id}_support_status": a_support,
        f"{b_id}_auc": b_auc, f"{b_id}_ci_low": b_ci[0], f"{b_id}_ci_high": b_ci[1],
        f"{b_id}_direction": b_dir, f"{b_id}_support_status": b_support,
        "auc_difference_target_minus_source": (b_auc - a_auc) if target_id == b_id else (a_auc - b_auc),
        "auc_difference_ci_low": 0.0, "auc_difference_ci_high": 0.1,
        "point_direction_reversal": point_reversal,
        "reversal_status": reversal_status,
        "step9e_relationship_direction_flag": step9e_flag,
        "integrated_interpretation": "placeholder",
    }


REGION_AUC = {FAKE_A: 0.55, FAKE_B: 0.56, FAKE_C: 0.57, FAKE_FUTURE: 0.58}


def default_rows(a_id: str, b_id: str, overrides: dict[str, dict] | None = None) -> list[dict]:
    """Region-identity-consistent fixture rows: each region's AUC/CI is
    fixed regardless of whether it appears as `a_id` or `b_id`, so the same
    region compared across different pairs never spuriously conflicts."""
    overrides = overrides or {}
    rows = []
    for feature in NUMERIC_FEATURES:
        kwargs = overrides.get(feature, {})
        kwargs.setdefault("a_auc", REGION_AUC[a_id])
        kwargs.setdefault("a_ci", (REGION_AUC[a_id] - 0.05, REGION_AUC[a_id] + 0.05))
        kwargs.setdefault("b_auc", REGION_AUC[b_id])
        kwargs.setdefault("b_ci", (REGION_AUC[b_id] - 0.05, REGION_AUC[b_id] + 0.05))
        rows.append(make_row(feature, a_id, b_id, **kwargs))
    return rows


def write_pair(tmp_path: Path, a_id: str, b_id: str, rows: list[dict], prereg_overrides: dict | None = None) -> Path:
    pair_id = f"{a_id}__{b_id}"
    pair_dir = tmp_path / "outputs" / "diagnostics" / "step9g_univariate_feature_auc_direction_reversal" / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "analysis_id": f"analysis_id_{pair_id}",
        "schema_version": "step9g.univariate_feature_auc_direction_reversal.v1",
        "source_experiment_id": a_id, "target_experiment_id": b_id,
        "primary_population": "burnable_tree_shrub_grass",
        "direction_reversal_table": rows,
        "answers": {},
    }
    (pair_dir / "step9g_final_report.json").write_text(json.dumps(report, indent=2))
    (pair_dir / "step9g_preregistration.json").write_text(json.dumps(_prereg(**(prereg_overrides or {}))))
    return pair_dir / "step9g_final_report.json"


@pytest.fixture(autouse=True)
def _redirect_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(step9g, "PROJECT_ROOT", tmp_path)


# ---------------------------------------------------------------------------
# 1. Arbitrary future experiment IDs accepted through the resolver
# ---------------------------------------------------------------------------
def test_resolve_experiments_accepts_arbitrary_future_ids():
    with patch.object(build_mod, "get_experiment", return_value={"experiment_id": FAKE_FUTURE}):
        resolved = build_mod.resolve_experiments([FAKE_FUTURE, FAKE_A])
    assert resolved == (FAKE_FUTURE, FAKE_A)


def test_cli_parses_arbitrary_future_experiment_ids():
    parser = build_parser()
    args = parser.parse_args([
        "concept-shift-compare", "--experiments", "totally_new_future_id_2099", "another_future_id_2099", "--dry-run",
    ])
    assert args.experiments == ["totally_new_future_id_2099", "another_future_id_2099"]


def test_cli_dispatches_through_orchestrator():
    parser = build_parser()
    args = parser.parse_args(["concept-shift-compare", "--experiments", FAKE_A, FAKE_B, "--dry-run"])
    with patch.object(sys.modules["scripts.main"].orch, "run_concept_shift_compare_stage", return_value={"ran": False}) as mocked:
        assert cmd_concept_shift_compare(args) == 0
    mocked.assert_called_once_with(experiments=[FAKE_A, FAKE_B], dry_run=True, force=False)


# ---------------------------------------------------------------------------
# 2. Experiment argument order does not change the synthesis analysis ID
# ---------------------------------------------------------------------------
def test_analysis_id_is_order_invariant(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    with patch.object(build_mod, "get_experiment", return_value={}):
        result_1 = build_mod.build_comparison([FAKE_A, FAKE_B], dry_run=False)
        result_2 = build_mod.build_comparison([FAKE_B, FAKE_A], dry_run=False)
    assert result_1["manifest"]["analysis_id"] == result_2["manifest"]["analysis_id"]


# ---------------------------------------------------------------------------
# 3. Duplicate region-feature results across pair reports are deduplicated
# ---------------------------------------------------------------------------
def test_duplicate_region_feature_results_deduplicated(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    write_pair(tmp_path, FAKE_A, FAKE_C, default_rows(FAKE_A, FAKE_C))
    with patch.object(build_mod, "get_experiment", return_value={}):
        result = build_mod.build_comparison([FAKE_A, FAKE_B, FAKE_C], dry_run=False)
    long_rows = [r for r in result["long_rows"] if r["experiment_id"] == FAKE_A and r["feature"] == "ndvi_mean"]
    assert len(long_rows) == 1  # FAKE_A/ndvi_mean appears in 2 reports, deduped to 1 row
    assert set(long_rows[0]["source_pair_reports"].split(";")) == {f"{FAKE_A}__{FAKE_B}", f"{FAKE_A}__{FAKE_C}"}


# ---------------------------------------------------------------------------
# 4-5. Conflicting duplicate AUC / CI values fail clearly
# ---------------------------------------------------------------------------
def test_conflicting_duplicate_auc_fails_clearly(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    conflicting = default_rows(FAKE_A, FAKE_C, overrides={"ndvi_mean": {"a_auc": 0.999}})
    write_pair(tmp_path, FAKE_A, FAKE_C, conflicting)
    with patch.object(build_mod, "get_experiment", return_value={}):
        with pytest.raises(ConsistencyError, match="auc"):
            build_mod.build_comparison([FAKE_A, FAKE_B, FAKE_C], dry_run=False)


def test_conflicting_duplicate_ci_fails_clearly(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    conflicting = default_rows(FAKE_A, FAKE_C, overrides={"ndvi_mean": {"a_ci": (0.1, 0.2)}})
    write_pair(tmp_path, FAKE_A, FAKE_C, conflicting)
    with patch.object(build_mod, "get_experiment", return_value={}):
        with pytest.raises(ConsistencyError, match="ci_low|ci_high"):
            build_mod.build_comparison([FAKE_A, FAKE_B, FAKE_C], dry_run=False)


# ---------------------------------------------------------------------------
# 6-7. Different primary populations / block-size configs fail clearly
# ---------------------------------------------------------------------------
def test_different_primary_population_fails_clearly(tmp_path):
    path = write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    report = json.loads(path.read_text())
    report["primary_population"] = "all_valid"
    path.write_text(json.dumps(report))
    with pytest.raises(ScientificContractError, match="primary_population"):
        build_mod.parse_pair_report(path)


def test_different_block_size_config_fails_clearly(tmp_path):
    path = write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B), prereg_overrides={"block_size_cells": 20})
    with pytest.raises(ScientificContractError, match="block_size_cells"):
        build_mod.parse_pair_report(path)


def test_different_bootstrap_config_fails_clearly(tmp_path):
    path = write_pair(
        tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B),
        prereg_overrides={"bootstrap": {"replicates": 500, "seed": 42}},
    )
    with pytest.raises(ScientificContractError, match="bootstrap"):
        build_mod.parse_pair_report(path)


# ---------------------------------------------------------------------------
# 8. Missing pair reports are recorded rather than fabricated
# ---------------------------------------------------------------------------
def test_missing_pair_reports_recorded_not_fabricated(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    write_pair(tmp_path, FAKE_A, FAKE_C, default_rows(FAKE_A, FAKE_C))
    # No FAKE_B / FAKE_C report exists.
    with patch.object(build_mod, "get_experiment", return_value={}):
        result = build_mod.build_comparison([FAKE_A, FAKE_B, FAKE_C], dry_run=False)
    assert result["missing_pairs"] == [f"{FAKE_B}__{FAKE_C}"]
    assert result["complete_pairwise_matrix"] is False
    pairwise_pairs = {(r["experiment_a"], r["experiment_b"]) for r in result["pairwise_rows"]}
    assert (FAKE_B, FAKE_C) not in pairwise_pairs


def test_experiment_with_zero_reports_fails_clearly(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    # FAKE_C appears in no report at all.
    with patch.object(build_mod, "get_experiment", return_value={}):
        with pytest.raises(build_mod.ComparisonError):
            build_mod.build_comparison([FAKE_A, FAKE_B, FAKE_C], dry_run=False)


# ---------------------------------------------------------------------------
# 9. Wide output contains all selected regions
# ---------------------------------------------------------------------------
def test_wide_output_contains_all_selected_regions(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    write_pair(tmp_path, FAKE_A, FAKE_C, default_rows(FAKE_A, FAKE_C))
    write_pair(tmp_path, FAKE_B, FAKE_C, default_rows(FAKE_B, FAKE_C))
    with patch.object(build_mod, "get_experiment", return_value={}):
        result = build_mod.build_comparison([FAKE_A, FAKE_B, FAKE_C], dry_run=False)
    wide_row = result["wide_rows"][0]
    for eid in (FAKE_A, FAKE_B, FAKE_C):
        assert f"{eid}_auc" in wide_row
        assert f"{eid}_ci_low" in wide_row
        assert f"{eid}_ci_high" in wide_row
        assert f"{eid}_direction" in wide_row
        assert f"{eid}_support_status" in wide_row
    assert len(result["wide_rows"]) == len(NUMERIC_FEATURES)


# ---------------------------------------------------------------------------
# 10. Landcover is excluded from scalar AUC
# ---------------------------------------------------------------------------
def test_landcover_excluded_from_scalar_auc(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    with patch.object(build_mod, "get_experiment", return_value={}):
        result = build_mod.build_comparison([FAKE_A, FAKE_B], dry_run=False)
    long_features = {r["feature"] for r in result["long_rows"]}
    assert "landcover_dominant" not in long_features
    assert "landcover_excluded_reason" in result["manifest"]["scientific_contract"]


# ---------------------------------------------------------------------------
# 16. Step8A/Step9E/Step9F/Step10 artifacts are not modified
# ---------------------------------------------------------------------------
def test_step8a_step9_artifacts_not_touched(tmp_path):
    sentinel_dirs = ["outputs/experiments", "outputs/cross_region", "outputs/diagnostics/step10_self_calibrated_transfer"]
    for d in sentinel_dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / d / "sentinel.txt").write_text("do not touch")

    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    with patch.object(build_mod, "get_experiment", return_value={}):
        build_mod.run_comparison([FAKE_A, FAKE_B], dry_run=False)

    for d in sentinel_dirs:
        assert (tmp_path / d / "sentinel.txt").read_text() == "do not touch"


# ---------------------------------------------------------------------------
# 17. Dry-run writes no files
# ---------------------------------------------------------------------------
def test_dry_run_writes_no_files(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    with patch.object(build_mod, "get_experiment", return_value={}):
        result = build_mod.run_comparison([FAKE_A, FAKE_B], dry_run=True)
    assert result["ran"] is False
    comparison_root = build_mod.comparison_output_root()
    assert not comparison_root.exists()


# ---------------------------------------------------------------------------
# Full real-run smoke test: outputs written, force guard works.
# ---------------------------------------------------------------------------
def test_full_run_writes_expected_outputs_and_force_guard(tmp_path):
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B))
    with patch.object(build_mod, "get_experiment", return_value={}):
        first = build_mod.run_comparison([FAKE_A, FAKE_B], dry_run=False)
        output_dir = Path(first["output_dir"])
        for name in (
            "multi_aoi_univariate_auc_comparison.json", "multi_aoi_univariate_auc_long.csv",
            "multi_aoi_univariate_auc_wide.csv", "pairwise_direction_reversal_summary.csv",
            "multi_aoi_univariate_auc_comparison.md", "manifest.json",
        ):
            assert (output_dir / name).is_file()

        second = build_mod.run_comparison([FAKE_A, FAKE_B], dry_run=False)
        assert second["analysis_id"] == first["analysis_id"]  # idempotent rerun

    # Now change the underlying report -> different analysis_id -> must fail without --force.
    write_pair(tmp_path, FAKE_A, FAKE_B, default_rows(FAKE_A, FAKE_B, overrides={"ndvi_mean": {"a_auc": 0.71}}))
    with patch.object(build_mod, "get_experiment", return_value={}):
        with pytest.raises(build_mod.ComparisonError):
            build_mod.run_comparison([FAKE_A, FAKE_B], dry_run=False, force=False)
        forced = build_mod.run_comparison([FAKE_A, FAKE_B], dry_run=False, force=True)
        assert forced["ran"] is True
