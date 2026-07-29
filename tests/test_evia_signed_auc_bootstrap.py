"""Contract tests for the Evia signed-AUC spatial-block bootstrap."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.paths import PROJECT_ROOT
from src import evia_signed_auc_bootstrap as mod
from src.step9a_audit_cross_region_inputs import TARGET_COLUMN

OUT_DIR = mod.OUTPUT_ROOT / mod.CANONICAL_EXPERIMENT
SUMMARY_CSV = OUT_DIR / "evia_signed_auc_summary.csv"
REPLICATES = OUT_DIR / "evia_signed_auc_bootstrap_replicates.parquet"
REPORT_JSON = OUT_DIR / "evia_signed_auc_final_report.json"

requires_outputs = pytest.mark.skipif(
    not SUMMARY_CSV.is_file(), reason="signed-AUC analysis has not been run yet"
)


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_CSV)


@pytest.fixture(scope="module")
def replicates() -> pd.DataFrame:
    return pd.read_parquet(REPLICATES)


# --------------------------------------------------------------------- 1, 2
def test_signed_auc_definition_is_two_auc_minus_one():
    for raw in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert mod.signed_auc(raw) == pytest.approx(2.0 * raw - 1.0)


def test_auc_of_one_half_maps_to_signed_zero():
    assert mod.signed_auc(0.5) == pytest.approx(0.0, abs=1e-15)


@requires_outputs
def test_summary_signed_auc_matches_definition(summary):
    valid = summary.dropna(subset=["raw_auc", "signed_auc"])
    assert not valid.empty
    expected = 2.0 * valid["raw_auc"] - 1.0
    assert np.allclose(valid["signed_auc"], expected, atol=1e-12)


# ------------------------------------------------------------------------ 3
@requires_outputs
def test_auc_is_never_inverted(summary):
    """A sub-chance feature must keep raw_auc < 0.5 and a NEGATIVE signed value."""
    below = summary[summary["raw_auc"] < 0.5]
    assert not below.empty, "expected at least one sub-chance feature in Evia"
    assert (below["signed_auc"] < 0).all()
    assert (below["direction"] == "lower_values_rank_burned").all()

    above = summary[summary["raw_auc"] > 0.5]
    assert (above["signed_auc"] > 0).all()
    assert (above["direction"] == "higher_values_rank_burned").all()


# ------------------------------------------------------------------------ 4
def test_bootstrap_unit_is_five_km_spatial_block():
    assert mod.BLOCK_SIZE_CELLS == 10
    assert mod.NOMINAL_BLOCK_SCALE == "approximately_5_km"
    prereg = mod.build_preregistration(mod.CANONICAL_EXPERIMENT)
    assert prereg["bootstrap"]["unit"] == "spatial_block"
    assert prereg["bootstrap"]["block_size_cells"] == 10
    assert prereg["bootstrap"]["cell_size_m"] == 500


def test_bootstrap_resamples_whole_blocks():
    source = inspect.getsource(mod._block_bootstrap_auc)
    assert "large_block_id" in source
    assert "replace=True" in source


# ------------------------------------------------------------------------ 5
def test_row_bootstrap_is_forbidden():
    prereg = mod.build_preregistration(mod.CANONICAL_EXPERIMENT)
    assert prereg["bootstrap"]["row_bootstrap"] == "forbidden"
    source = inspect.getsource(mod._block_bootstrap_auc)
    # Resampling must index blocks, never raw rows.
    assert "rng.choice(unique_blocks" in source


# ------------------------------------------------------------------------ 6
@requires_outputs
def test_replicate_signed_distribution_matches_transformed_raw(summary, replicates):
    """The signed CI must equal the monotone transform of the raw CI."""
    for feature, group in replicates.groupby("feature"):
        assert np.allclose(group["signed_auc"], 2.0 * group["raw_auc"] - 1.0, atol=1e-12)

        row = summary[summary["feature"] == feature].iloc[0]
        lo = np.percentile(group["signed_auc"], mod.CI_LOWER_PCT)
        hi = np.percentile(group["signed_auc"], mod.CI_UPPER_PCT)
        assert row["signed_auc_ci_low"] == pytest.approx(lo, abs=1e-12)
        assert row["signed_auc_ci_high"] == pytest.approx(hi, abs=1e-12)

        # Equivalence with transforming the raw CI.
        assert row["signed_auc_ci_low"] == pytest.approx(2.0 * row["raw_auc_ci_low"] - 1.0, abs=1e-12)
        assert row["signed_auc_ci_high"] == pytest.approx(2.0 * row["raw_auc_ci_high"] - 1.0, abs=1e-12)


@requires_outputs
def test_support_status_follows_signed_ci(summary):
    for _, r in summary.iterrows():
        if pd.isna(r["signed_auc_ci_low"]):
            continue
        if r["signed_auc_ci_low"] > 0:
            assert r["support_status"] == mod.SUPPORT_POSITIVE
        elif r["signed_auc_ci_high"] < 0:
            assert r["support_status"] == mod.SUPPORT_NEGATIVE
        else:
            assert r["support_status"] == mod.SUPPORT_ZERO


# ------------------------------------------------------------------------ 7
def test_feature_contract():
    expected = (
        "ndvi_mean", "elevation_mean", "slope_mean", "lst_anomaly_mean",
        "current_lst_mean", "current_tvdi_mean", "tvdi_difference_mean",
        "downscaled_lst_mean", "fused_lst_mean",
    )
    assert tuple(mod.NUMERIC_FEATURES) == expected


@requires_outputs
def test_summary_covers_exactly_the_contract_features(summary):
    assert sorted(summary["feature"]) == sorted(mod.NUMERIC_FEATURES)


@requires_outputs
def test_summary_has_required_columns(summary):
    required = {
        "experiment_id", "feature", "n_rows", "n_burned", "n_unburned",
        "n_spatial_blocks", "raw_auc", "raw_auc_ci_low", "raw_auc_ci_high",
        "signed_auc", "signed_auc_ci_low", "signed_auc_ci_high", "direction",
        "support_status", "successful_replicates", "requested_replicates",
    }
    assert required.issubset(set(summary.columns))


# ------------------------------------------------------------------------ 8
def test_target_label_is_used_only_for_diagnostic_auc():
    prereg = mod.build_preregistration(mod.CANONICAL_EXPERIMENT)
    assert prereg["target_label_use"] == "diagnostic AUC computation only"
    # The module must never fit or adapt anything.
    source = inspect.getsource(mod)
    for forbidden in ("fit(", "predict_proba", "CORAL", "train_test_split"):
        assert forbidden not in source, f"unexpected modeling call: {forbidden}"


def test_population_is_the_canonical_primary_population():
    assert mod.PRIMARY_POPULATION == "burnable_tree_shrub_grass"


# ------------------------------------------------------------------------ 9
@requires_outputs
def test_deterministic_rerun_reproduces_summary(summary):
    recomputed = mod.compute_experiment(mod.CANONICAL_EXPERIMENT)["summary"]
    merged = summary.merge(recomputed, on="feature", suffixes=("_frozen", "_new"))
    assert len(merged) == len(summary)
    assert np.allclose(
        merged["signed_auc_frozen"].astype(float),
        merged["signed_auc_new"].astype(float),
        atol=1e-12, equal_nan=True,
    )
    assert np.allclose(
        merged["signed_auc_ci_low_frozen"].astype(float),
        merged["signed_auc_ci_low_new"].astype(float),
        atol=1e-12, equal_nan=True,
    )


def test_off_contract_seed_or_replicates_are_refused():
    with pytest.raises(SystemExit):
        mod.run(seed=7, dry_run=True)
    with pytest.raises(SystemExit):
        mod.run(bootstrap_replicates=500, dry_run=True)


# ----------------------------------------------------------------------- 10
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@requires_outputs
def test_frozen_step9g_inputs_unchanged():
    """Every frozen input recorded in the audit must still hash identically."""
    audit = json.loads((OUT_DIR / "evia_signed_auc_input_audit.json").read_text())
    checked = 0
    for entry in audit["inputs"]:
        path = PROJECT_ROOT / entry["path"]
        assert path.is_file(), f"frozen input disappeared: {entry['path']}"
        assert _sha256(path) == entry["sha256"], f"frozen input MUTATED: {entry['path']}"
        checked += 1
    assert checked >= 10


@requires_outputs
def test_module_reproduces_frozen_step9g_point_estimates():
    payload = json.loads(REPORT_JSON.read_text())
    xv = payload["step9g_crossvalidation"]
    assert xv["available"] is True
    assert xv["reproduces_frozen_step9g"] is True
    assert xv["max_abs_diff_raw_auc"] <= 1e-9


@requires_outputs
def test_analysis_writes_only_inside_its_namespace():
    prereg = json.loads((OUT_DIR / "evia_signed_auc_preregistration.json").read_text())
    assert prereg["writes_only_under"].endswith("evia_signed_auc_bootstrap")
    for path in OUT_DIR.glob("*"):
        assert mod.OUTPUT_ROOT in path.parents


@requires_outputs
def test_step10c_is_explicitly_rejected():
    prereg = json.loads((OUT_DIR / "evia_signed_auc_preregistration.json").read_text())
    assert "step10c" in prereg["rejected_module"]["module"]
    assert "estimand" in prereg["rejected_module"]["reason"]


@requires_outputs
def test_replicate_counts_meet_the_minimum(summary):
    available = summary.dropna(subset=["raw_auc"])
    assert (available["successful_replicates"] >= mod.MIN_VALID_REPLICATES).all()
    assert (available["requested_replicates"] == 1000).all()


@requires_outputs
def test_population_counts_match_canonical_step8a(summary):
    """n_rows/n_burned must equal the canonical primary population."""
    df = mod.load_step8a(mod.CANONICAL_EXPERIMENT)
    pop = mod.assign_blocks_then_filter(df, mod.CANONICAL_EXPERIMENT)
    y = pd.to_numeric(pop[TARGET_COLUMN], errors="coerce")
    assert int(summary["n_rows"].iloc[0]) == len(pop)
    assert int(summary["n_burned"].iloc[0]) == int((y == 1).sum())


def test_step9g_support_vocabulary_maps_onto_signed_vocabulary():
    assert mod.normalize_step9g_support(
        "bootstrap_supported_higher_values_rank_burned"
    ) == mod.SUPPORT_POSITIVE
    assert mod.normalize_step9g_support(
        "bootstrap_supported_lower_values_rank_burned"
    ) == mod.SUPPORT_NEGATIVE
    assert mod.normalize_step9g_support("interval_includes_chance") == mod.SUPPORT_ZERO
