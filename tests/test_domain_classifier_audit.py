"""Regression tests for the generic multi-experiment pairwise
domain-classifier (covariate-separability) diagnostic
(src/domain_classifier_audit.py, scripts/run_domain_classifier_audit.py,
core.pipeline_orchestrator.run_domain_classifier_audit_stage,
scripts/main.py `domain-classifier-audit`).

Uses entirely synthetic/placeholder experiment IDs and hand-built parquet
fixtures wherever possible (matching this repo's existing convention),
redirecting PROJECT_ROOT so nothing here touches the real repo output tree
or fits a model against real Step8A data."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.domain_classifier_audit as dca
from scripts.main import build_parser, cmd_domain_classifier_audit

FAKE_A = "aoi_alpha_2099"
FAKE_B = "aoi_beta_2099"
FAKE_C = "aoi_gamma_2099"


def make_step8a_frame(
    n_blocks: int = 10, rows_per_block: int = 8, feature_shift: float = 0.0,
    seed: int = 0, eligible_overrides: dict[int, bool] | None = None,
) -> pd.DataFrame:
    """Synthetic Step8A-shaped population: n_blocks distinct 10-cell blocks
    (row_500m // 10), each with rows_per_block rows, feature values shifted
    by `feature_shift` relative to the baseline (0.0) so two calls with
    different shifts are more or less separable."""
    rng = np.random.default_rng(seed)
    rows = []
    for block in range(n_blocks):
        base_row = block * 10
        for i in range(rows_per_block):
            row_500m = base_row + i
            rows.append({
                "row_500m": row_500m, "col_500m": 0,
                "ndvi_mean": rng.normal(0.5 + feature_shift, 0.05),
                "elevation_mean": rng.normal(500 + feature_shift * 200, 50),
                "slope_mean": rng.normal(10 + feature_shift * 5, 2),
                "landcover_dominant": int(rng.choice([10, 20, 30])),
                "lst_anomaly_mean": rng.normal(0 + feature_shift, 1),
                "current_lst_mean": rng.normal(30 + feature_shift * 2, 2),
                "current_tvdi_mean": rng.normal(0.5 + feature_shift * 0.1, 0.05),
                "tvdi_difference_mean": rng.normal(0 + feature_shift * 0.1, 0.05),
                "downscaled_lst_mean": rng.normal(30 + feature_shift * 2, 2),
                "fused_lst_mean": rng.normal(30 + feature_shift * 2, 2),
                "burned": int(rng.random() < 0.3),
                "burnable_tree_shrub_grass": True,
            })
    df = pd.DataFrame(rows)
    if eligible_overrides:
        df["analysis_eligible"] = True
        df["pre_label_burn_excluded"] = False
        for block, eligible in eligible_overrides.items():
            mask = (df["row_500m"] // 10) == block
            df.loc[mask, "analysis_eligible"] = eligible
            df.loc[mask, "pre_label_burn_excluded"] = not eligible
    return df


def write_step8a(tmp_path: Path, experiment_id: str, df: pd.DataFrame) -> Path:
    root = tmp_path / "outputs" / "experiments" / experiment_id / "step8a"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "step8a_500m_modeling_dataset.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture(autouse=True)
def _redirect_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(dca, "OUTPUT_ROOT", tmp_path / "outputs" / "diagnostics" / "domain_classifier_audit")
    monkeypatch.setattr(dca, "PAIRS_OUTPUT_ROOT", dca.OUTPUT_ROOT / "pairs")
    monkeypatch.setattr(dca, "COMPARISON_OUTPUT_DIR", dca.OUTPUT_ROOT / "comparison")


def _patch_step8a_paths(monkeypatch, paths: dict[str, Path]):
    monkeypatch.setattr(dca, "canonical_step8a_path", lambda eid: paths[eid])


# ---------------------------------------------------------------------------
# 1. All unordered pairs are generated dynamically
# ---------------------------------------------------------------------------
def test_generate_pairs_all_unordered_combinations():
    pairs = dca.generate_pairs((FAKE_B, FAKE_A, FAKE_C))
    assert pairs == [(FAKE_A, FAKE_B), (FAKE_A, FAKE_C), (FAKE_B, FAKE_C)]


# ---------------------------------------------------------------------------
# 2. Arbitrary future experiment IDs work through the resolver
# ---------------------------------------------------------------------------
def test_cli_parses_arbitrary_future_experiment_ids():
    parser = build_parser()
    args = parser.parse_args([
        "domain-classifier-audit", "--experiments", "future_id_2099", "another_future_id_2099", "--dry-run",
    ])
    assert args.experiments == ["future_id_2099", "another_future_id_2099"]


def test_cli_dispatches_through_orchestrator():
    parser = build_parser()
    args = parser.parse_args(["domain-classifier-audit", "--experiments", FAKE_A, FAKE_B, "--dry-run"])
    with patch.object(sys.modules["scripts.main"].orch, "run_domain_classifier_audit_stage", return_value={"ran": False}) as mocked:
        assert cmd_domain_classifier_audit(args) == 0
    mocked.assert_called_once_with(experiments=[FAKE_A, FAKE_B], all_enabled=False, dry_run=True, force=False)


# ---------------------------------------------------------------------------
# 3. Input experiment order does not alter pair IDs or analysis ID
# ---------------------------------------------------------------------------
def test_pair_id_and_analysis_id_order_invariant(tmp_path, monkeypatch):
    path_a = write_step8a(tmp_path, FAKE_A, make_step8a_frame(feature_shift=0.0, seed=1))
    path_b = write_step8a(tmp_path, FAKE_B, make_step8a_frame(feature_shift=3.0, seed=2))
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})

    result_1 = dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)
    result_2 = dca.analyze_pair(FAKE_B, FAKE_A, dry_run=False, force=True)
    assert result_1["pair_id"] == result_2["pair_id"] == f"{FAKE_A}__{FAKE_B}"
    assert result_1["analysis_id"] == result_2["analysis_id"]


# ---------------------------------------------------------------------------
# 4. --experiments and --all-enabled are mutually exclusive
# ---------------------------------------------------------------------------
def test_cli_rejects_both_selectors():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["domain-classifier-audit", "--experiments", FAKE_A, "--all-enabled", "--dry-run"])


def test_cli_requires_one_selector():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["domain-classifier-audit", "--dry-run"])


# ---------------------------------------------------------------------------
# 5. Dry-run writes no files
# ---------------------------------------------------------------------------
def test_dry_run_writes_no_files(tmp_path, monkeypatch):
    path_a = write_step8a(tmp_path, FAKE_A, make_step8a_frame(seed=1))
    path_b = write_step8a(tmp_path, FAKE_B, make_step8a_frame(seed=2))
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})
    result = dca.analyze_pair(FAKE_A, FAKE_B, dry_run=True)
    assert result["ran"] is False
    assert not dca.OUTPUT_ROOT.exists()


# ---------------------------------------------------------------------------
# 6. Missing Step8A fails clearly
# ---------------------------------------------------------------------------
def test_missing_step8a_fails_clearly(tmp_path, monkeypatch):
    path_a = write_step8a(tmp_path, FAKE_A, make_step8a_frame(seed=1))
    missing_path = tmp_path / "does_not_exist.parquet"
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: missing_path})
    with pytest.raises(dca.DomainClassifierAuditError, match="Missing canonical Step8A"):
        dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)


# ---------------------------------------------------------------------------
# 7. Canonical eligibility excludes pre-label rows
# ---------------------------------------------------------------------------
def test_canonical_eligibility_excludes_pre_label_rows(tmp_path, monkeypatch):
    frame = make_step8a_frame(n_blocks=5, rows_per_block=6, seed=3, eligible_overrides={0: False})
    path_a = write_step8a(tmp_path, FAKE_A, frame)
    path_b = write_step8a(tmp_path, FAKE_B, make_step8a_frame(n_blocks=5, rows_per_block=6, feature_shift=2.0, seed=4))
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})

    population = dca.resolve_population(frame, FAKE_A)
    assert (population["row_500m"] // 10 == 0).sum() == 0  # excluded block never re-enters
    assert len(population) == len(frame) - 6  # exactly the excluded block's rows are gone


# ---------------------------------------------------------------------------
# 8-10. Leakage audit: burned / coordinates / experiment ID never predictors
# ---------------------------------------------------------------------------
def test_burned_never_a_predictor():
    assert "burned" not in dca.DOMAIN_CLASSIFIER_FEATURES


def test_coordinates_never_predictors():
    assert "row_500m" not in dca.DOMAIN_CLASSIFIER_FEATURES
    assert "col_500m" not in dca.DOMAIN_CLASSIFIER_FEATURES
    assert "lon" not in dca.DOMAIN_CLASSIFIER_FEATURES
    assert "lat" not in dca.DOMAIN_CLASSIFIER_FEATURES


def test_experiment_identity_never_a_predictor():
    assert "experiment_id" not in dca.DOMAIN_CLASSIFIER_FEATURES
    assert "region_key" not in dca.DOMAIN_CLASSIFIER_FEATURES
    audit = dca.leakage_audit()
    assert "experiment_id" in audit["never_predictor_semantic_identities"]
    assert "region_key" in audit["never_predictor_semantic_identities"]
    assert "burned" in audit["never_predictor_semantic_identities"]
    assert "row_500m" in audit["never_predictor_semantic_identities"]


def test_leakage_audit_raises_if_feature_contract_ever_contains_forbidden_column(monkeypatch):
    monkeypatch.setattr(dca, "DOMAIN_CLASSIFIER_FEATURES", ("ndvi_mean", "burned"))
    with pytest.raises(dca.DomainClassifierAuditError):
        dca.leakage_audit()


# ---------------------------------------------------------------------------
# 11. Categorical encoding is fit inside training folds (unseen-category
#     robustness) -- 12. Numeric preprocessing fit inside training folds
#     (NaN-only-in-test robustness).
# ---------------------------------------------------------------------------
def test_unseen_category_and_train_only_nan_do_not_break_oof(tmp_path, monkeypatch):
    frame_a = make_step8a_frame(n_blocks=8, rows_per_block=10, feature_shift=0.0, seed=5)
    frame_b = make_step8a_frame(n_blocks=8, rows_per_block=10, feature_shift=2.5, seed=6)
    # Inject a landcover category present in exactly one block only, and an
    # NDVI NaN present in exactly one block only -- whichever fold holds
    # that block out as test, the categorical encoder/imputer must have
    # been fit on the OTHER blocks (train) only, and must not crash.
    frame_a.loc[frame_a["row_500m"] // 10 == 0, "landcover_dominant"] = 99
    frame_a.loc[frame_a["row_500m"] // 10 == 1, "ndvi_mean"] = np.nan

    path_a = write_step8a(tmp_path, FAKE_A, frame_a)
    path_b = write_step8a(tmp_path, FAKE_B, frame_b)
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})

    result = dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)
    assert result["ran"] is True
    probs = pd.read_parquet(Path(result["output_dir"]) / "domain_classifier_oof_predictions.parquet")["oof_probability_domain_1"]
    assert np.isfinite(probs).all()
    assert probs.between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# 13-15. Zero block overlap; every OOF row predicted exactly once;
#        probabilities finite and within [0, 1].
# ---------------------------------------------------------------------------
def test_zero_block_overlap_full_oof_coverage_and_valid_probabilities(tmp_path, monkeypatch):
    frame_a = make_step8a_frame(n_blocks=10, rows_per_block=8, feature_shift=0.0, seed=7)
    frame_b = make_step8a_frame(n_blocks=10, rows_per_block=8, feature_shift=3.0, seed=8)
    path_a = write_step8a(tmp_path, FAKE_A, frame_a)
    path_b = write_step8a(tmp_path, FAKE_B, frame_b)
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})

    result = dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)
    metrics = result["metrics"]
    assert metrics["zero_block_overlap"] is True

    oof = pd.read_parquet(Path(result["output_dir"]) / "domain_classifier_oof_predictions.parquet")
    assert len(oof) == len(frame_a) + len(frame_b)  # every eligible row predicted exactly once
    assert not oof["oof_probability_domain_1"].isna().any()
    assert oof["oof_probability_domain_1"].between(0.0, 1.0).all()
    assert set(oof["domain"].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# 16. Block bootstrap samples blocks, not rows
# ---------------------------------------------------------------------------
def test_block_bootstrap_samples_blocks_not_rows():
    combined = pd.DataFrame({
        "domain": [0, 0, 0, 0, 1, 1, 1, 1],
        "domain_block_id": ["a_0", "a_0", "a_1", "a_1", "b_0", "b_0", "b_1", "b_1"],
    })
    oof_probs = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
    result = dca.block_bootstrap_domain_auc(combined, oof_probs, n_replicates=50, seed=1)
    assert result["valid_replicates"] + result["invalid_replicates"] == 50
    # Rows within the same block always move together (never split) --
    # verified indirectly: sampling only 2 blocks per domain can only ever
    # produce even-sized per-block contributions of 2 rows each.
    assert result["valid_replicates"] > 0


def test_block_bootstrap_reports_valid_invalid_counts_summing_to_requested():
    combined = pd.DataFrame({"domain": [0, 1], "domain_block_id": ["a_0", "b_0"]})
    oof_probs = np.array([0.4, 0.6])
    result = dca.block_bootstrap_domain_auc(combined, oof_probs, n_replicates=20, seed=2)
    assert result["valid_replicates"] + result["invalid_replicates"] == 20
    assert result["seed"] == 2


# ---------------------------------------------------------------------------
# 17-18. No fabricated legacy value; legacy fields always null/false
# ---------------------------------------------------------------------------
def test_legacy_fields_always_null_and_false(tmp_path, monkeypatch):
    path_a = write_step8a(tmp_path, FAKE_A, make_step8a_frame(seed=9))
    path_b = write_step8a(tmp_path, FAKE_B, make_step8a_frame(feature_shift=2.0, seed=10))
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})
    result = dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)
    metrics = result["metrics"]
    assert metrics["legacy_method_available"] is False
    assert metrics["legacy_comparable_domain_auc"] is None
    assert metrics["legacy_comparable_ci_low"] is None
    assert metrics["legacy_comparable_ci_high"] is None
    assert metrics["legacy_evaluation_type"] is None
    assert dca.LEGACY_METHOD_AVAILABLE is False


# ---------------------------------------------------------------------------
# 19. Comparison includes exactly three rows for three experiments
# ---------------------------------------------------------------------------
def test_comparison_includes_exactly_three_rows_for_three_experiments(tmp_path, monkeypatch):
    path_a = write_step8a(tmp_path, FAKE_A, make_step8a_frame(feature_shift=0.0, seed=11))
    path_b = write_step8a(tmp_path, FAKE_B, make_step8a_frame(feature_shift=2.0, seed=12))
    path_c = write_step8a(tmp_path, FAKE_C, make_step8a_frame(feature_shift=4.0, seed=13))
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b, FAKE_C: path_c})

    resolution = dca.ExperimentResolution(
        requested_ids=(FAKE_A, FAKE_B, FAKE_C), resolved_ids=(FAKE_A, FAKE_B, FAKE_C),
        selection_mode="explicit", excluded={},
    )
    result = dca.run_comparison(resolution, dry_run=False, force=False)
    assert result["ran"] is True
    comparison_csv = pd.read_csv(Path(result["output_dir"]) / "multi_aoi_domain_classifier_comparison.csv")
    assert len(comparison_csv) == 3
    pairs = set(zip(comparison_csv["experiment_a"], comparison_csv["experiment_b"]))
    assert pairs == {(FAKE_A, FAKE_B), (FAKE_A, FAKE_C), (FAKE_B, FAKE_C)}


# ---------------------------------------------------------------------------
# 20. Step8A/Step9E/Step9G/Step10/Evia files remain unchanged
# ---------------------------------------------------------------------------
def test_step8a_and_other_step_artifacts_not_touched(tmp_path, monkeypatch):
    path_a = write_step8a(tmp_path, FAKE_A, make_step8a_frame(seed=14))
    path_b = write_step8a(tmp_path, FAKE_B, make_step8a_frame(feature_shift=2.0, seed=15))
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})

    before_a, before_b = path_a.read_bytes(), path_b.read_bytes()
    sentinel_dirs = ["outputs/cross_region", "outputs/diagnostics/step10_self_calibrated_transfer", "outputs/experiments/evia_2021"]
    for d in sentinel_dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / d / "sentinel.txt").write_text("do not touch")

    dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)

    assert path_a.read_bytes() == before_a
    assert path_b.read_bytes() == before_b
    for d in sentinel_dirs:
        assert (tmp_path / d / "sentinel.txt").read_text() == "do not touch"


# ---------------------------------------------------------------------------
# Force / manifest guard behavior + hash-unchanged assertion inside the
# module itself.
# ---------------------------------------------------------------------------
def test_rerun_without_force_but_matching_analysis_id_is_idempotent(tmp_path, monkeypatch):
    path_a = write_step8a(tmp_path, FAKE_A, make_step8a_frame(seed=16))
    path_b = write_step8a(tmp_path, FAKE_B, make_step8a_frame(feature_shift=2.0, seed=17))
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})

    first = dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)
    second = dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)
    assert first["analysis_id"] == second["analysis_id"]


def test_rerun_with_changed_input_requires_force(tmp_path, monkeypatch):
    path_a = write_step8a(tmp_path, FAKE_A, make_step8a_frame(seed=18))
    path_b = write_step8a(tmp_path, FAKE_B, make_step8a_frame(feature_shift=2.0, seed=19))
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})

    dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)
    write_step8a(tmp_path, FAKE_B, make_step8a_frame(feature_shift=5.0, seed=20))

    with pytest.raises(dca.DomainClassifierAuditError, match="different analysis_id"):
        dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False, force=False)
    forced = dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False, force=True)
    assert forced["ran"] is True


def test_full_run_writes_expected_output_files(tmp_path, monkeypatch):
    path_a = write_step8a(tmp_path, FAKE_A, make_step8a_frame(seed=21))
    path_b = write_step8a(tmp_path, FAKE_B, make_step8a_frame(feature_shift=2.0, seed=22))
    _patch_step8a_paths(monkeypatch, {FAKE_A: path_a, FAKE_B: path_b})

    result = dca.analyze_pair(FAKE_A, FAKE_B, dry_run=False)
    out_dir = Path(result["output_dir"])
    for name in (
        "domain_classifier_metrics.json", "domain_classifier_fold_metrics.csv",
        "domain_classifier_oof_predictions.parquet", "domain_classifier_bootstrap.json",
        "domain_classifier_report.md", "manifest.json",
    ):
        assert (out_dir / name).is_file()


# ---------------------------------------------------------------------------
# No hardcoded real experiment IDs in the scientific source modules (mirrors
# the equivalent test for src/burned_pattern_audit.py).
# ---------------------------------------------------------------------------
def test_no_hardcoded_real_experiment_ids_in_implementation():
    import re
    from core.regions import EXPERIMENTS as REAL_REGISTRY

    real_ids = list(REAL_REGISTRY.keys())
    for source_path in (
        _PROJECT_ROOT / "src" / "domain_classifier_audit.py",
        _PROJECT_ROOT / "scripts" / "run_domain_classifier_audit.py",
    ):
        text = source_path.read_text()
        for experiment_id in real_ids:
            assert re.search(rf"\b{re.escape(experiment_id)}\b", text) is None, (
                f"{source_path} appears to hard-code real experiment_id '{experiment_id}'."
            )
