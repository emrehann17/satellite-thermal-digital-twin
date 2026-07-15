"""Targeted tests for the Step9G report-integration correction (v2)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.step9g_integration_correction_v2 as v2


# =============================================================================
# Fixture builders: reproduce the ACTUAL shared pair-level schemas
# =============================================================================
FEATURES = v2.NUMERIC_FEATURES
FWD = v2.FORWARD_DIRECTION
REV = v2.REVERSE_DIRECTION


def _frozen_reversal_table() -> pd.DataFrame:
    """One supported reversal (elevation_mean); four uncertain LST/TVDI point
    reversals; the rest same-direction. Matches requirement (10)/(11)."""
    supported = {"elevation_mean"}
    uncertain = {"current_lst_mean", "tvdi_difference_mean", "downscaled_lst_mean", "fused_lst_mean"}
    rows = []
    for f in FEATURES:
        if f in supported:
            status = "bootstrap_supported_direction_reversal"
            m_auc, b_auc, point = 0.45, 0.64, True
            m_lo, m_hi, b_lo, b_hi = 0.40, 0.48, 0.57, 0.70
        elif f in uncertain:
            status = "point_direction_reversal_interval_uncertain"
            m_auc, b_auc, point = 0.47, 0.55, True
            m_lo, m_hi, b_lo, b_hi = 0.43, 0.52, 0.49, 0.61  # intervals include 0.5
        else:
            status = "no_direction_reversal"
            m_auc, b_auc, point = 0.62, 0.64, False
            m_lo, m_hi, b_lo, b_hi = 0.56, 0.68, 0.58, 0.70
        rows.append({
            "feature": f,
            "manavgat_auc": m_auc, "manavgat_ci_low": m_lo, "manavgat_ci_high": m_hi,
            "manavgat_direction": "lower_values_rank_burned" if m_auc < 0.5 else "higher_values_rank_burned",
            "bejis_auc": b_auc, "bejis_ci_low": b_lo, "bejis_ci_high": b_hi,
            "bejis_direction": "lower_values_rank_burned" if b_auc < 0.5 else "higher_values_rank_burned",
            "reversal_status": status,
            "point_direction_reversal": point,
        })
    return pd.DataFrame(rows)


def _step9e_flips() -> pd.DataFrame:
    """Pair-global relationship flips. The five direction-reversing features
    also carry a rank-effect direction flip. Booleans stored as real bools so
    that CSV round-trip yields the string form the parser must coerce."""
    reversing = {"elevation_mean", "current_lst_mean", "tvdi_difference_mean", "downscaled_lst_mean", "fused_lst_mean"}
    rows = []
    for f in FEATURES:
        flip = f in reversing
        rows.append({
            "feature": f, "population": v2.step9g.PRIMARY_POPULATION,
            "source_experiment_id": v2.SOURCE_ID, "target_experiment_id": v2.TARGET_ID,
            "mean_direction_flip": flip, "median_direction_flip": flip,
            "rank_effect_direction_flip": flip,
            "raw_auc_below_0_5_in_one_region_only": flip,
            "relationship_flip_score": 3 if flip else 0,
        })
    return pd.DataFrame(rows)


def _step9f_screening() -> pd.DataFrame:
    """Shared artifact with per-direction columns '{direction}__...' for BOTH
    directions."""
    return pd.DataFrame({
        "variant": ["original_thermal", "stable_core"],
        f"{FWD}__ranking_reversal_suspected": [True, False],
        f"{FWD}__delta_roc_auc": [-0.2, -0.1],
        f"{REV}__ranking_reversal_suspected": [True, True],
        f"{REV}__delta_roc_auc": [-0.25, -0.15],
    })


def _step9f_bootstrap() -> dict:
    return {"groups": [
        {"transfer_direction": FWD, "variant": "original_thermal"},
        {"transfer_direction": REV, "variant": "original_thermal"},
    ]}


def _step10_combined_report() -> dict:
    """Single combined report keyed by direction (both present)."""
    return {
        "analysis_id": "step10frozenid",
        "target_performance": [
            {"direction": FWD, "model": "rf", "roc_auc": 0.41},
            {"direction": REV, "model": "rf", "roc_auc": 0.43},
        ],
        "integrated_interpretation": {
            FWD: {"raw_below_chance": True, "adapted_recovered": True},
            REV: {"raw_below_chance": True, "adapted_recovered": True},
        },
    }


def _build_fixture(tmp_path: Path, frozen_id: str | None = None) -> None:
    frozen_id = frozen_id or v2.EXPECTED_FROZEN_STEP9G_ANALYSIS_ID
    # frozen Step9G numeric outputs
    g_root = tmp_path / "outputs" / "diagnostics" / "step9g_univariate_feature_auc_direction_reversal" / v2.PAIR_TOKEN
    g_root.mkdir(parents=True, exist_ok=True)
    _frozen_reversal_table().to_csv(g_root / "step9g_direction_reversal_table.csv", index=False)
    pd.DataFrame({"experiment_id": [v2.SOURCE_ID], "feature": ["ndvi_mean"], "raw_univariate_auc": [0.7]}).to_csv(
        g_root / "step9g_univariate_auc_by_region.csv", index=False)
    pd.DataFrame({"experiment_id": [v2.SOURCE_ID], "feature": ["ndvi_mean"], "replicate": [0], "auc": [0.7]}).to_parquet(
        g_root / "step9g_bootstrap_replicates.parquet", index=False)
    (g_root / "step9g_final_report.json").write_text(json.dumps({"analysis_id": frozen_id}))
    (g_root / "step9g_preregistration.json").write_text(json.dumps({"analysis_id": frozen_id}))

    # shared pair-level references
    pair = tmp_path / "outputs" / "cross_region" / v2.PAIR_TOKEN
    (pair / "step9e").mkdir(parents=True, exist_ok=True)
    (pair / "step9f").mkdir(parents=True, exist_ok=True)
    (pair / "step10").mkdir(parents=True, exist_ok=True)
    _step9e_flips().to_csv(pair / "step9e" / "relationship_direction_flips.csv", index=False)
    _step9f_screening().to_csv(pair / "step9f" / "exploratory_candidate_screening.csv", index=False)
    (pair / "step9f" / "spatial_block_bootstrap_deltas.json").write_text(json.dumps(_step9f_bootstrap()))
    (pair / "step9f" / "step9f_experiment_manifest.json").write_text(json.dumps({"analysis_id": "step9fid"}))
    (pair / "step10" / "step10_final_report.json").write_text(json.dumps(_step10_combined_report()))


def _patch_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(v2, "PROJECT_ROOT", tmp_path)
    # module-level cached roots must be recomputed against the patched PROJECT_ROOT
    monkeypatch.setattr(v2, "STEP9G_FROZEN_ROOT", v2._step9g_frozen_root())
    monkeypatch.setattr(v2, "OUTPUT_ROOT", v2._output_root())


# =============================================================================
# 1. parse two logical directions from one combined Step10 report
# =============================================================================
def test_step10_both_directions_from_combined_report(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    step10 = v2.parse_step10()
    assert step10["available"] is True
    assert step10["directions"][FWD]["available"] is True
    assert step10["directions"][REV]["available"] is True


# =============================================================================
# 2. parse shared pair-level Step9E artifacts
# =============================================================================
def test_step9e_pair_global(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    step9e = v2.parse_step9e()
    assert step9e["available"] is True
    assert step9e["schema"] == "pair_global"
    assert step9e["per_feature"]["elevation_mean"]["rank_effect_direction_flip"] is True
    assert step9e["per_feature"]["ndvi_mean"]["rank_effect_direction_flip"] is False


# =============================================================================
# 3. parse both Step9F directions when stored in a shared artifact
# =============================================================================
def test_step9f_both_directions_shared_artifact(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    step9f = v2.parse_step9f()
    assert step9f["level"] == "model_representation_level_only"
    assert step9f["directions"][FWD]["available"] is True
    assert step9f["directions"][REV]["available"] is True
    assert step9f["directions"][REV]["any_ranking_reversal_suspected"] is True


# =============================================================================
# 4. never mark an existing logical direction unavailable merely because a
#    direction-specific directory is absent
# =============================================================================
def test_reverse_direction_not_marked_unavailable(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    # There is NO cross_region/bejis_2022__manavgat_2021 directory at all.
    assert not (tmp_path / "outputs" / "cross_region" / f"{v2.TARGET_ID}__{v2.SOURCE_ID}").exists()
    corrected = v2.build_corrected_integration()
    avail = v2.availability_table(corrected)
    step9f_rev = next(r for r in avail if r["stage"] == "step9f" and r["direction"] == REV)
    step10_rev = next(r for r in avail if r["stage"] == "step10" and r["direction"] == REV)
    assert step9f_rev["before_v1"] == "unavailable"
    assert step9f_rev["after_v2"] == "available"
    assert step10_rev["after_v2"] == "available"


# =============================================================================
# 5. preserve all frozen Step9G numeric values exactly
# =============================================================================
def test_frozen_numeric_values_preserved(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    original = _frozen_reversal_table()
    corrected = v2.build_corrected_integration()
    per_feature = {r["feature"]: r for r in corrected["per_feature"]}
    for _, row in original.iterrows():
        pf = per_feature[row["feature"]]
        assert pf["manavgat_auc"] == row["manavgat_auc"]
        assert pf["bejis_auc"] == row["bejis_auc"]
        assert pf["reversal_status"] == row["reversal_status"]
    # feature ordering preserved
    assert [r["feature"] for r in corrected["per_feature"]] == list(FEATURES)


# =============================================================================
# 6. reject string booleans in corrected JSON
# =============================================================================
def test_no_string_booleans_in_output(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    result = v2.run_correction(dry=False)
    report_path = Path(result["report_paths"]["final_report_json"])
    raw = report_path.read_text()
    # crude but effective: no quoted python-style booleans anywhere
    assert '"True"' not in raw and '"False"' not in raw
    # and the per-feature booleans are real bools
    payload = json.loads(raw)
    for row in payload["per_feature_integration"]:
        assert isinstance(row["point_direction_reversal"], bool)
        assert isinstance(row["step9e_consistent"], bool)
        assert row["step9e_rank_effect_direction_flip"] in (True, False, None)


# =============================================================================
# 7. do not classify elevation as a thermal feature
# =============================================================================
def test_elevation_not_thermal(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    corrected = v2.build_corrected_integration()
    assert "elevation_mean" in corrected["features_consistent_with_step9e"]
    assert "elevation_mean" in corrected["baseline_features_consistent_with_step9e"]
    assert "elevation_mean" not in corrected["thermal_features_consistent_with_step9e"]
    # the four LST/TVDI reversing features ARE thermal-consistent
    for f in ("current_lst_mean", "tvdi_difference_mean", "downscaled_lst_mean", "fused_lst_mean"):
        assert f in corrected["thermal_features_consistent_with_step9e"]


# =============================================================================
# 8. do not overwrite the original Step9G outputs
# =============================================================================
def test_original_step9g_outputs_untouched(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    g_root = v2._step9g_frozen_root()
    before = {p.name: v2.sha256_file(p) for p in g_root.iterdir() if p.is_file()}
    result = v2.run_correction(dry=False)
    after = {p.name: v2.sha256_file(p) for p in g_root.iterdir() if p.is_file()}
    assert before == after
    assert result["frozen_step9g_hash_check"] == "passed"
    # outputs went only under the v2 namespace
    out_root = v2._output_root()
    assert out_root.exists()
    for p in out_root.rglob("*"):
        if p.is_file():
            assert "integration_v2" in str(p)


# =============================================================================
# Supporting: required-findings guard + uncertain != supported (req 11)
# =============================================================================
def test_uncertain_not_labeled_supported(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    result = v2.run_correction(dry=False)
    supported = set(result["bootstrap_supported_direction_reversal"])
    uncertain = set(result["point_reversals_interval_uncertain"])
    assert supported == {"elevation_mean"}
    assert uncertain == {"current_lst_mean", "tvdi_difference_mean", "downscaled_lst_mean", "fused_lst_mean"}
    assert not (supported & uncertain)


def test_wrong_frozen_analysis_id_rejected(tmp_path, monkeypatch):
    _build_fixture(tmp_path, frozen_id="0" * 64)
    _patch_roots(monkeypatch, tmp_path)
    with pytest.raises(v2.Step9GIntegrationError):
        v2.assert_frozen_step9g_analysis_id()


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    result = v2.run_correction(dry=True)
    assert result["writes_files"] is False
    assert result["recomputes_step9g_numeric"] is False
    assert not v2._output_root().exists()


def test_force_required_to_overwrite(tmp_path, monkeypatch):
    _build_fixture(tmp_path)
    _patch_roots(monkeypatch, tmp_path)
    v2.run_correction(dry=False)
    with pytest.raises(v2.Step9GIntegrationError):
        v2.run_correction(dry=False, force=False)
    # force succeeds
    result = v2.run_correction(dry=False, force=True)
    assert result["ran"] is True