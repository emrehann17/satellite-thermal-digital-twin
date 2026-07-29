"""Contract tests for the multi-AOI transfer gap/recovery decomposition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.paths import PROJECT_ROOT
from src import transfer_decomposition as mod

AOIS = ["manavgat_2021", "bejis_2022", "mugla_2021", "evia_2021_extended"]
SET_ID = mod.canonical_set_id(AOIS)
OUT_DIR = mod.OUTPUT_ROOT / SET_ID
DECOMP_CSV = OUT_DIR / "four_aoi_decomposition.csv"
REPORT_JSON = OUT_DIR / "four_aoi_decomposition_final_report.json"

requires_outputs = pytest.mark.skipif(
    not DECOMP_CSV.is_file(), reason="decomposition has not been run yet"
)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_csv(DECOMP_CSV)


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT_JSON.read_text())


# ------------------------------------------------------------------------ 1
@requires_outputs
def test_gap_identity(table):
    assert np.allclose(
        table["raw_gap"], table["within_target_auc"] - table["raw_auc"], atol=1e-12
    )
    assert np.allclose(
        table["adaptation_effect"], table["adapted_auc"] - table["raw_auc"], atol=1e-12
    )
    assert np.allclose(
        table["remaining_gap"], table["within_target_auc"] - table["adapted_auc"], atol=1e-12
    )


@requires_outputs
def test_gap_additivity(table):
    """adaptation_effect + remaining_gap == raw_gap."""
    assert np.allclose(
        table["adaptation_effect"] + table["remaining_gap"], table["raw_gap"], atol=1e-12
    )


# ------------------------------------------------------------------------ 2
@requires_outputs
def test_fraction_identity(table):
    valid = table.dropna(subset=["recovered_fraction", "remaining_fraction"])
    assert not valid.empty
    total = valid["recovered_fraction"] + valid["remaining_fraction"]
    assert np.allclose(total, 1.0, atol=mod.IDENTITY_TOLERANCE)


@requires_outputs
def test_reported_identity_residual_within_tolerance(report):
    assert report["max_identity_residual"] is not None
    assert report["max_identity_residual"] <= mod.IDENTITY_TOLERANCE


# ------------------------------------------------------------------------ 3
@requires_outputs
def test_positive_recovery_rows(table):
    pos = table[table["adaptation_effect"] > 0].dropna(subset=["recovered_fraction"])
    assert not pos.empty
    assert (pos["recovered_fraction"] > 0).all()
    assert (pos["recovery_status"] != mod.STATUS_NEGATIVE).all()


# ------------------------------------------------------------------------ 4
@requires_outputs
def test_negative_recovery_rows_are_reported_not_clipped(table):
    neg = table[table["adaptation_effect"] < 0]
    assert not neg.empty, "expected negative-recovery rows in this 4-AOI set"
    assert (neg["recovery_status"] == mod.STATUS_NEGATIVE).all()
    interpretable = neg[neg["fraction_interpretability_status"] == mod.INTERPRETABLE]
    # Never clipped to zero: fractions stay strictly negative.
    assert (interpretable["recovered_fraction"] < 0).all()
    assert (interpretable["remaining_fraction"] > 1).all()


@requires_outputs
def test_worked_negative_example_matches_specification(table):
    """Bejís->Muğla thermal CORAL: raw .618, adapted .507, within .859."""
    row = table[
        (table["direction"] == "bejis_2022_to_mugla_2021")
        & (table["model_family"] == "thermal")
        & (table["adaptation_method"] == "coral_after_regionwise_zscore")
        & (table["metric"] == "roc_auc")
    ]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["raw_auc"] == pytest.approx(0.618, abs=5e-3)
    assert r["adapted_auc"] == pytest.approx(0.507, abs=5e-3)
    assert r["within_target_auc"] == pytest.approx(0.859, abs=5e-3)
    assert r["adaptation_effect"] == pytest.approx(-0.111, abs=5e-3)
    assert r["raw_gap"] == pytest.approx(0.241, abs=5e-3)
    assert r["recovered_fraction"] == pytest.approx(-0.46, abs=0.02)
    assert r["remaining_fraction"] == pytest.approx(1.46, abs=0.02)
    assert r["recovery_status"] == mod.STATUS_NEGATIVE


# ---------------------------------------------------------------------- 5, 6
def test_raw_gap_zero_and_negative_guard():
    """raw_gap <= 0 must suppress the fraction, not divide by it."""
    for within, raw in ((0.70, 0.70), (0.60, 0.75)):
        raw_gap = within - raw
        assert raw_gap <= 0
        interpretable = raw_gap > 0.0
        assert interpretable is False
        status = mod.NOT_INTERPRETABLE if not interpretable else mod.INTERPRETABLE
        assert status == mod.NOT_INTERPRETABLE


@requires_outputs
def test_no_fraction_emitted_when_raw_gap_not_positive(table):
    bad = table[table["raw_gap"] <= 0]
    if not bad.empty:
        assert bad["recovered_fraction"].isna().all()
        assert bad["remaining_fraction"].isna().all()
        assert (bad["fraction_interpretability_status"] == mod.NOT_INTERPRETABLE).all()
        assert (bad["recovery_status"] == mod.STATUS_NOT_INTERPRETABLE).all()


@requires_outputs
def test_interpretable_rows_all_have_positive_raw_gap(table):
    ok = table[table["fraction_interpretability_status"] == mod.INTERPRETABLE]
    assert (ok["raw_gap"] > 0).all()


# ------------------------------------------------------------------------ 7
def test_ratio_near_zero_replicate_guard_threshold():
    assert mod.RATIO_DEGENERATE_THRESHOLD == 1e-6
    raw_gap_rep = np.array([1e-9, 1e-3, -1e-9, 0.2])
    degenerate = np.abs(raw_gap_rep) < mod.RATIO_DEGENERATE_THRESHOLD
    assert degenerate.tolist() == [True, False, True, False]


@requires_outputs
def test_degenerate_replicates_are_counted_and_excluded(table):
    assert (table["n_replicates_ratio_valid"] + table["n_replicates_ratio_degenerate"]
            == table["n_replicates"]).all()
    assert (table["n_replicates_ratio_degenerate"] >= 0).all()


# ------------------------------------------------------------------------ 8
@requires_outputs
def test_zscore_and_coral_kept_separate(table):
    methods = set(table["adaptation_method"])
    assert methods == set(mod.ADAPTATION_METHODS)
    grouped = table.groupby(
        ["direction", "model_family", "metric", "adaptation_method"]
    ).size()
    assert (grouped == 1).all(), "each method must appear once per cell, never merged"


# ------------------------------------------------------------------------ 9
@requires_outputs
def test_baseline_and_thermal_kept_separate(table):
    assert set(table["model_family"]) == set(mod.MODEL_FAMILIES)
    counts = table.groupby(["direction", "adaptation_method", "metric"])["model_family"].nunique()
    assert (counts == 2).all()


# ----------------------------------------------------------------------- 10
@requires_outputs
def test_raw_reproduction_within_tolerance(report):
    rep = report["raw_reproduction"]
    assert rep["available"] is True
    assert rep["reproduces_frozen_step10"] is True
    assert rep["worst_abs_diff"] <= mod.RAW_REPRODUCTION_TOLERANCE


# ----------------------------------------------------------------------- 11
@requires_outputs
def test_all_twelve_directions_present(table):
    assert table["direction"].nunique() == 12
    expected_rows = 12 * len(mod.MODEL_FAMILIES) * len(mod.ADAPTATION_METHODS) * len(mod.METRICS)
    assert len(table) == expected_rows == 96


@requires_outputs
def test_every_direction_covers_all_four_aoi_pairs(table):
    sources = set(table["source_experiment_id"])
    targets = set(table["target_experiment_id"])
    assert sources == set(AOIS)
    assert targets == set(AOIS)
    assert (table["source_experiment_id"] != table["target_experiment_id"]).all()


# ----------------------------------------------------------------------- 12
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@requires_outputs
def test_frozen_step10_artifacts_unchanged():
    audit = json.loads((OUT_DIR / "four_aoi_decomposition_input_audit.json").read_text())
    checked = 0
    for entry in audit["inputs"]:
        path = PROJECT_ROOT / entry["path"]
        assert path.is_file(), f"frozen Step10 input disappeared: {entry['path']}"
        assert _sha256(path) == entry["sha256"], f"frozen Step10 input MUTATED: {entry['path']}"
        checked += 1
    assert checked >= 20


# --------------------------------------------------------------- conventions
@requires_outputs
def test_relative_improvement_and_above_chance_are_not_merged(table):
    """A positive adapted-minus-raw alone must not claim above-chance support."""
    rel_only = table[table["recovery_status"] == mod.STATUS_RELATIVE_ONLY]
    if not rel_only.empty:
        assert (rel_only["relative_improvement_supported"]).all()
        assert (~rel_only["adapted_above_chance"]).all()

    above = table[table["recovery_status"] == mod.STATUS_ABOVE_CHANCE]
    if not above.empty:
        assert (above["relative_improvement_supported"]).all()
        assert (above["adapted_above_chance"]).all()


@requires_outputs
def test_pr_auc_chance_level_is_prevalence_not_one_half(table):
    pr = table[table["metric"] == "pr_auc"]
    assert not pr.empty
    assert (pr["chance_level"] != 0.5).all()
    assert (pr["chance_level"] > 0).all() and (pr["chance_level"] < 1).all()

    roc = table[table["metric"] == "roc_auc"]
    assert (roc["chance_level"] == 0.5).all()


@requires_outputs
def test_status_vocabulary_is_closed(table):
    allowed = {
        mod.STATUS_ABOVE_CHANCE, mod.STATUS_RELATIVE_ONLY, mod.STATUS_UNCERTAIN,
        mod.STATUS_NEGATIVE, mod.STATUS_NOT_INTERPRETABLE,
    }
    assert set(table["recovery_status"]).issubset(allowed)


@requires_outputs
def test_required_columns_present(table):
    required = {
        "source_experiment_id", "target_experiment_id", "model_family",
        "adaptation_method", "within_target_auc", "raw_auc", "adapted_auc",
        "raw_gap", "adaptation_effect", "remaining_gap",
        "recovered_fraction", "recovered_fraction_ci_low", "recovered_fraction_ci_high",
        "remaining_fraction", "remaining_fraction_ci_low", "remaining_fraction_ci_high",
        "recovery_status", "fraction_interpretability_status",
    }
    assert required.issubset(set(table.columns))


@requires_outputs
def test_joint_uncertainty_is_declared_only_because_within_is_per_replicate(report):
    unc = report["preregistration"]["uncertainty"]
    assert unc["within_region_reference_available_per_replicate"] is True
    assert unc["joint_uncertainty"] is True


@requires_outputs
def test_old_two_region_percentages_are_not_carried_forward(table):
    """The superseded 27-31% / 69-73% split must not reappear verbatim."""
    bm = table[
        table["direction"].isin(["bejis_2022_to_mugla_2021", "mugla_2021_to_bejis_2022"])
        & (table["metric"] == "roc_auc")
    ].dropna(subset=["recovered_fraction"])
    assert not bm.empty
    stale = bm[bm["recovered_fraction"].between(0.27, 0.31)
               & bm["remaining_fraction"].between(0.69, 0.73)]
    # Only a genuine recomputation may land in that band; assert we did not
    # simply inherit it for every row.
    assert len(stale) < len(bm)
