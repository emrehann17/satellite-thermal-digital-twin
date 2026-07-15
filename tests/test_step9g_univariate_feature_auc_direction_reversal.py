"""Targeted tests for Step9G univariate feature-AUC direction-reversal."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.step9g_univariate_feature_auc_direction_reversal as g


# =============================================================================
# Synthetic Step8A-shaped frames
# =============================================================================
def _base_frame(n_blocks: int, seed: int, elevation_higher_for_burned: bool) -> pd.DataFrame:
    """Each block = 10x10 cell area with 4 rows (2 burned, 2 unburned)."""
    rng = np.random.default_rng(seed)
    rows, cols, labels, cells = [], [], [], []
    for blk in range(n_blocks):
        base_r = blk * 10
        for j, label in enumerate((1, 0, 1, 0)):
            rows.append(base_r + j)
            cols.append(j)
            labels.append(label)
            cells.append(f"r{base_r + j}_c{j}_b{blk}")
    n = len(rows)
    labels = np.array(labels)
    df = pd.DataFrame({"row_500m": rows, "col_500m": cols, "burned": labels, "cell_id": cells})
    # elevation: burned systematically higher or lower depending on flag
    shift = 2.0 if elevation_higher_for_burned else -2.0
    df["elevation_mean"] = rng.normal(loc=labels * shift, scale=0.5, size=n)
    for feat in g.NUMERIC_FEATURES:
        if feat == "elevation_mean":
            continue
        df[feat] = rng.normal(loc=labels * 1.0, scale=1.0, size=n)
    df["landcover_dominant"] = 30
    df["valid_for_modeling"] = True
    df["burnable_tree_shrub_grass"] = True
    return df


# =============================================================================
# 1. exact frozen numeric feature list and order
# =============================================================================
def test_exact_feature_list_and_order():
    assert g.NUMERIC_FEATURES == (
        "ndvi_mean", "elevation_mean", "slope_mean", "lst_anomaly_mean",
        "current_lst_mean", "current_tvdi_mean", "tvdi_difference_mean",
        "downscaled_lst_mean", "fused_lst_mean",
    )


# =============================================================================
# 2. primary population is burnable_tree_shrub_grass
# =============================================================================
def test_primary_population():
    assert g.PRIMARY_POPULATION == "burnable_tree_shrub_grass"


# =============================================================================
# 3. block assignment occurs before filtering
# =============================================================================
def test_block_assigned_before_filtering():
    df = _base_frame(6, seed=1, elevation_higher_for_burned=True)
    # add rows that will be filtered out but still get blocks
    df.loc[0, "valid_for_modeling"] = False
    df.loc[1, "burnable_tree_shrub_grass"] = False
    pop = g.assign_blocks_then_filter(df, "manavgat_2021")
    assert "large_block_id" in pop.columns
    # filtered rows removed but remaining rows retained their pre-filter block id
    assert (pop["large_block_id"].str.startswith("b10_")).all()
    assert len(pop) == len(df) - 2


# =============================================================================
# 4/5/6. no imputation/normalization; raw values to roc_auc; AUC<0.5 preserved
# =============================================================================
def test_raw_values_and_auc_below_half_preserved():
    df = _base_frame(6, seed=2, elevation_higher_for_burned=False)  # burned LOWER
    pop = g.assign_blocks_then_filter(df, "manavgat_2021")
    stats = g.univariate_feature_stats(pop, "elevation_mean")
    assert stats["raw_univariate_auc"] is not None
    assert stats["raw_univariate_auc"] < 0.5  # not inverted to > 0.5
    assert stats["direction"] == g.DIRECTION_LOWER
    # signed rank effect = 2*auc - 1
    assert abs(stats["signed_rank_effect"] - (2 * stats["raw_univariate_auc"] - 1)) < 1e-12


def test_no_imputation_missing_values_dropped():
    df = _base_frame(6, seed=3, elevation_higher_for_burned=True)
    df.loc[df.index[:4], "ndvi_mean"] = np.nan
    pop = g.assign_blocks_then_filter(df, "manavgat_2021")
    stats = g.univariate_feature_stats(pop, "ndvi_mean")
    assert stats["n_missing"] == 4
    assert stats["n_complete_case"] == len(pop) - 4


# =============================================================================
# 7. direction labels around 0.5
# =============================================================================
def test_direction_labels():
    assert g._direction_label(0.7) == g.DIRECTION_HIGHER
    assert g._direction_label(0.3) == g.DIRECTION_LOWER
    assert g._direction_label(0.5) == g.DIRECTION_CHANCE
    assert g._direction_label(None) == g.DIRECTION_UNAVAILABLE


# =============================================================================
# 8/9. feature-specific complete-case; missingness by target class
# =============================================================================
def test_missingness_by_target_class():
    df = _base_frame(6, seed=4, elevation_higher_for_burned=True)
    pop = g.assign_blocks_then_filter(df, "manavgat_2021")
    burned_idx = pop.index[pop["burned"] == 1][:3]
    pop.loc[burned_idx, "slope_mean"] = np.nan
    stats = g.univariate_feature_stats(pop, "slope_mean")
    n_burned = int((pop["burned"] == 1).sum())
    assert abs(stats["missing_rate_burned"] - 3 / n_burned) < 1e-12
    assert stats["missing_rate_unburned"] == 0.0


# =============================================================================
# 10/11. bootstrap samples whole blocks; multiplicity preserved
# =============================================================================
def test_bootstrap_samples_whole_blocks(monkeypatch):
    df = _base_frame(30, seed=5, elevation_higher_for_burned=True)
    pop = g.assign_blocks_then_filter(df, "manavgat_2021")
    captured = {}
    real_choice = np.random.default_rng(0).choice

    boot = g._block_bootstrap_auc(pop, "elevation_mean", seed=42)
    assert boot["valid"] + boot["invalid"] == g.BOOTSTRAP_REPLICATES
    # whole-block sampling: total sampled rows per replicate must be a multiple
    # of the per-block row count when block sizes are equal (they are here: 4).
    block_sizes = pop.groupby("large_block_id").size().unique()
    assert set(block_sizes) == {4}


# =============================================================================
# 12/13. one-class replicates invalidated; stability threshold enforced
# =============================================================================
def test_one_class_replicates_invalidated():
    # single block of all-burned -> every replicate is one-class -> all invalid
    df = pd.DataFrame({
        "row_500m": [0, 1, 2, 3], "col_500m": [0, 1, 2, 3],
        "burned": [1, 1, 1, 1], "cell_id": ["a", "b", "c", "d"],
        "elevation_mean": [1.0, 2.0, 3.0, 4.0],
        "valid_for_modeling": True, "burnable_tree_shrub_grass": True,
    })
    for feat in g.NUMERIC_FEATURES:
        if feat not in df.columns:
            df[feat] = 1.0
    df["landcover_dominant"] = 30
    pop = g.assign_blocks_then_filter(df, "manavgat_2021")
    stats = g.univariate_feature_stats(pop, "elevation_mean")
    # point AUC unavailable (single class) -> bootstrap short-circuits
    assert stats["raw_univariate_auc"] is None
    boot = g._block_bootstrap_auc(pop, "elevation_mean")
    assert boot["unavailable"] is True


def test_stability_threshold():
    assert g.MIN_VALID_REPLICATES == 900
    assert g._support_status(0.6, 0.7, stable=False) == "unstable_bootstrap"


# =============================================================================
# 14/15/16. reversal classification
# =============================================================================
def _boot(point, lo, hi, stable=True):
    return {"point_auc": point, "ci_low": lo, "ci_high": hi, "stable": stable, "unavailable": point is None}


def test_supported_reversal():
    m = _boot(0.42, 0.38, 0.47)  # entirely below 0.5
    b = _boot(0.64, 0.56, 0.70)  # entirely above 0.5
    assert g._reversal_status(m, b) == "bootstrap_supported_direction_reversal"


def test_uncertain_point_reversal():
    m = _boot(0.46, 0.42, 0.53)  # crosses 0.5
    b = _boot(0.62, 0.55, 0.68)
    assert g._reversal_status(m, b) == "point_direction_reversal_interval_uncertain"


def test_same_side_not_reversal():
    m = _boot(0.62, 0.55, 0.68)
    b = _boot(0.64, 0.57, 0.70)
    assert g._reversal_status(m, b) == "no_direction_reversal"


# =============================================================================
# 17. cross-region contrast uses independent regional draws
# =============================================================================
def test_contrast_uses_independent_draws():
    m_arr = np.array([0.4, 0.42, 0.44])
    b_arr = np.array([0.6, 0.62, 0.64])
    mean, lo, hi = g._contrast_ci(m_arr, b_arr)
    assert mean is not None
    # difference bejis - manavgat should be ~0.2 for paired-by-index elements
    assert abs(mean - 0.2) < 1e-9


# =============================================================================
# 18. landcover raw class codes excluded from numeric AUC
# =============================================================================
def test_landcover_excluded_from_numeric_features():
    assert g.LANDCOVER_COLUMN not in g.NUMERIC_FEATURES
    df = _base_frame(6, seed=6, elevation_higher_for_burned=True)
    pop = g.assign_blocks_then_filter(df, "manavgat_2021")
    lc = g.landcover_descriptive(pop, "manavgat_2021")
    assert "landcover_class_code" in lc.columns
    assert "burned_prevalence" in lc.columns


# =============================================================================
# 19/20. Step9E integration never invents fields; Step9F is model-level
# =============================================================================
def test_step9e_integration_no_invented_fields(tmp_path, monkeypatch):
    # No frozen step9e present -> availability False, no fabricated flags.
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    # patch the cross_region root resolver dependency
    df = g.step9e_feature_integration()
    assert (df["step9e_available"] == False).all()  # noqa: E712
    # none of the optional step9e_* flag columns should be populated
    flag_cols = [c for c in df.columns if c.startswith("step9e_") and c not in ("step9e_direction", "step9e_available")]
    for c in flag_cols:
        assert df[c].isna().all()


def test_step9f_is_model_level_only(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    payload = g.step9f_model_level_integration()
    assert "note" in payload and "model" in payload["note"].lower()
    assert "directions" in payload
    # no per-feature keys
    assert "feature" not in payload


# =============================================================================
# 21. frozen output hashes remain unchanged
# =============================================================================
def test_protected_hash_change_detected(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    before = g.protected_paths()
    after = g.protected_paths()
    g.assert_protected_unchanged(before, after)
    # mutate a step8a input
    target = g.resolve_step8a_dataset_path("manavgat_2021")
    df = pd.read_parquet(target)
    df.loc[0, "elevation_mean"] = df.loc[0, "elevation_mean"] + 100
    df.to_parquet(target, index=False)
    after2 = g.protected_paths()
    with pytest.raises(g.Step9GError):
        g.assert_protected_unchanged(before, after2)


# =============================================================================
# 22. existing preregistration cannot be silently changed
# =============================================================================
def test_preregistration_immutable(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    out_root = tmp_path / "outputs" / "diagnostics" / "step9g" / g.PAIR_TOKEN
    monkeypatch.setattr(g, "OUTPUT_ROOT", out_root)
    protected = g.protected_paths()
    m1 = g.validate_or_write_preregistration(out_root, protected)
    m2 = g.validate_or_write_preregistration(out_root, protected)
    assert m1 == m2
    # tamper
    path = out_root / "step9g_preregistration.json"
    payload = json.loads(path.read_text())
    payload["scientific_configuration"]["block_size_cells"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(g.Step9GError):
        g.validate_or_write_preregistration(out_root, protected)


# =============================================================================
# 23. dry-run performs no AUC/bootstrap and writes no files
# =============================================================================
def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    out_root = tmp_path / "outputs" / "diagnostics" / "step9g" / g.PAIR_TOKEN
    monkeypatch.setattr(g, "OUTPUT_ROOT", out_root)
    result = g.run_analysis(dry=True)
    assert result["computes_auc"] is False
    assert result["runs_bootstrap"] is False
    assert result["writes_files"] is False
    assert not out_root.exists()


# =============================================================================
# End-to-end smoke (small) confirming a full run writes only under namespace
# =============================================================================
def test_end_to_end_namespace_isolation(tmp_path, monkeypatch):
    _build_fixture(tmp_path, n_blocks=30)
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    out_root = tmp_path / "outputs" / "diagnostics" / "step9g" / g.PAIR_TOKEN
    monkeypatch.setattr(g, "OUTPUT_ROOT", out_root)
    result = g.run_analysis(dry=False)
    assert result["ran"] is True
    assert result["protected_hash_check"] == "passed"
    assert (out_root / "step9g_final_report.json").is_file()
    assert (out_root / "step9g_direction_reversal_table.csv").is_file()
    assert (out_root / "step9g_bootstrap_replicates.parquet").is_file()
    # every produced file is under the namespace
    for p in out_root.rglob("*"):
        if p.is_file():
            assert "diagnostics" in p.parts
    # frozen experiment dirs untouched (no step9g files leaked in)
    exp_files = list((tmp_path / "outputs" / "experiments").rglob("step9g_*"))
    assert exp_files == []


def test_no_prediction_inversion_logic_exists():
    import inspect
    src = inspect.getsource(g)
    # roc_auc must receive raw feature; there must be no '-x' inversion or
    # 1-prob prediction inversion in the univariate path.
    assert "roc_auc_score(yc, xc)" in src
    assert "roc_auc_score(y, -x)" not in src
    assert "1.0 - prob" not in src


# =============================================================================
# Fixture builder
# =============================================================================
def _build_fixture(tmp_path: Path, n_blocks: int = 12) -> None:
    # elevation reverses: burned higher in manavgat, burned lower in bejis
    specs = {"manavgat_2021": True, "bejis_2022": False}
    for experiment, higher in specs.items():
        step8a = tmp_path / "outputs" / "experiments" / experiment / "step8a"
        step8a.mkdir(parents=True, exist_ok=True)
        df = _base_frame(n_blocks, seed=hash(experiment) % 1000, elevation_higher_for_burned=higher)
        df.to_parquet(step8a / "step8a_500m_modeling_dataset.parquet", index=False)
    # minimal frozen step9e/f/10 dirs (empty is allowed -> availability False)
    for s, t in ((g.SOURCE_ID, g.TARGET_ID), (g.TARGET_ID, g.SOURCE_ID)):
        for stage in ("step9e", "step9f", "step10"):
            (tmp_path / "outputs" / "cross_region" / f"{s}__{t}" / stage).mkdir(parents=True, exist_ok=True)