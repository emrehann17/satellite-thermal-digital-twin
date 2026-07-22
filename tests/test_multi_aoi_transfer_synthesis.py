"""Tests for src/multi_aoi_transfer_synthesis.

Pure-logic tests (AOI-set validation, direction/pair generation, status
derivation, adapters, end-to-end build/render) use ONLY synthetic
placeholder experiment IDs (aoi_alpha_2021, aoi_beta_2022, aoi_gamma_2021)
and small hand-built fixture files under tmp_path -- never real project
outputs. One integration-style test points the resolvers at the REAL frozen
outputs already on disk for the pre-existing experiment IDs to confirm the
resolvers work against real schemas; its assertions are about resolution
succeeding and about generic/derived status labels, never hardcoded prose.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from src.multi_aoi_transfer_synthesis import status_derivation
from src.multi_aoi_transfer_synthesis.aoi_set import (
    AoiSetError, canonicalize_pair, validate_aoi_set,
)
from src.multi_aoi_transfer_synthesis import schema_adapters
from src.multi_aoi_transfer_synthesis.resolvers import InputResolutionError, PRIMARY_POPULATION

AOI_A = "aoi_alpha_2021"
AOI_B = "aoi_beta_2022"
AOI_C = "aoi_gamma_2021"
AOI_D = "aoi_delta_2023"
AOI_E = "aoi_epsilon_2020"


# =============================================================================
# aoi_set.py
# =============================================================================
def test_validate_aoi_set_accepts_2_to_5():
    for n in (2, 3, 4, 5):
        aois = [AOI_A, AOI_B, AOI_C, AOI_D, AOI_E][:n]
        aoi_set = validate_aoi_set(aois)
        assert len(aoi_set.canonical_order) == n


def test_validate_aoi_set_rejects_too_few():
    with pytest.raises(AoiSetError):
        validate_aoi_set([AOI_A])


def test_validate_aoi_set_rejects_too_many():
    with pytest.raises(AoiSetError):
        validate_aoi_set([AOI_A, AOI_B, AOI_C, AOI_D, AOI_E, "aoi_zeta_2024"])


def test_validate_aoi_set_rejects_duplicates():
    with pytest.raises(AoiSetError):
        validate_aoi_set([AOI_A, AOI_B, AOI_A])


def test_display_order_preserved_canonical_order_sorted():
    aoi_set = validate_aoi_set([AOI_C, AOI_A, AOI_B])
    assert aoi_set.display_order == (AOI_C, AOI_A, AOI_B)
    assert aoi_set.canonical_order == tuple(sorted([AOI_A, AOI_B, AOI_C]))


def test_canonical_set_id_deterministic_and_order_independent():
    id1 = validate_aoi_set([AOI_A, AOI_B, AOI_C]).canonical_set_id
    id2 = validate_aoi_set([AOI_C, AOI_B, AOI_A]).canonical_set_id
    assert id1 == id2
    assert id1  # non-empty
    assert " " not in id1


def test_canonical_set_id_hash_fallback_for_long_slug():
    long_ids = [f"aoi_{'x' * 60}_{i}_2021" for i in range(3)]
    aoi_set = validate_aoi_set(long_ids)
    assert len(aoi_set.canonical_set_id) <= 80


def test_ordered_directions_count_and_no_self_pairs():
    aoi_set = validate_aoi_set([AOI_A, AOI_B, AOI_C])
    directions = aoi_set.ordered_directions()
    assert len(directions) == 3 * 2  # N*(N-1)
    for src, tgt in directions:
        assert src != tgt
    assert len(set(directions)) == len(directions)


def test_unordered_pairs_count_and_canonical_order():
    aoi_set = validate_aoi_set([AOI_A, AOI_B, AOI_C])
    pairs = aoi_set.unordered_pairs()
    assert len(pairs) == 3  # N choose 2
    for a, b in pairs:
        assert a < b


def test_canonicalize_pair_order_independent():
    assert canonicalize_pair(AOI_B, AOI_A) == canonicalize_pair(AOI_A, AOI_B)


# =============================================================================
# status_derivation.py
# =============================================================================
def test_derive_raw_transfer_status_not_supported_when_ci_crosses_chance():
    assert status_derivation.derive_raw_transfer_status(0.48, 0.1) == status_derivation.RAW_DISCRIMINATION_NOT_SUPPORTED


def test_derive_raw_transfer_status_supported_both():
    assert status_derivation.derive_raw_transfer_status(0.55, 0.1) == status_derivation.RAW_ROC_AND_PR_SUPPORTED


def test_derive_raw_transfer_status_missing_ci_is_conservative():
    assert status_derivation.derive_raw_transfer_status(None, 0.1) == status_derivation.RAW_DISCRIMINATION_NOT_SUPPORTED


def test_derive_adaptation_support_status_positive_negative_uncertain():
    assert status_derivation.derive_adaptation_support_status(0.01, 0.05) == status_derivation.ADAPTATION_SUPPORTED
    assert status_derivation.derive_adaptation_support_status(-0.05, -0.01) == status_derivation.ADAPTATION_DEGRADED_TRANSFER
    assert status_derivation.derive_adaptation_support_status(-0.02, 0.02) == status_derivation.ADAPTATION_EFFECT_UNCERTAIN
    assert status_derivation.derive_adaptation_support_status(None, None) == status_derivation.ADAPTATION_EFFECT_UNCERTAIN


def test_derive_residual_gap_status():
    assert status_derivation.derive_residual_gap_status(0.1, 0.2) == status_derivation.RESIDUAL_GAP_REMAINS
    assert status_derivation.derive_residual_gap_status(-0.2, -0.1) == status_derivation.RESIDUAL_GAP_NOT_SUPPORTED
    assert status_derivation.derive_residual_gap_status(-0.1, 0.1) == status_derivation.RESIDUAL_GAP_UNCERTAIN


def test_derive_recovery_pattern_status():
    assert status_derivation.derive_recovery_pattern_status(
        [status_derivation.ADAPTATION_SUPPORTED, status_derivation.ADAPTATION_SUPPORTED]
    ) == status_derivation.BIDIRECTIONAL_RECOVERY
    assert status_derivation.derive_recovery_pattern_status(
        [status_derivation.ADAPTATION_SUPPORTED, status_derivation.ADAPTATION_EFFECT_UNCERTAIN]
    ) == status_derivation.DIRECTION_DEPENDENT_RECOVERY
    assert status_derivation.derive_recovery_pattern_status(
        [status_derivation.ADAPTATION_DEGRADED_TRANSFER, status_derivation.ADAPTATION_EFFECT_UNCERTAIN]
    ) == status_derivation.NO_RECOVERY_SUPPORTED


def test_derive_shift_categories_no_rows_is_unavailable():
    assert status_derivation.derive_shift_categories([]) == [status_derivation.SHIFT_AUDIT_UNAVAILABLE]


def test_derive_shift_categories_large_shift():
    rows = [{"abs_smd_ge_0_5": True, "abs_smd_ge_0_2": True, "psi_ge_0_25": False, "psi_ge_0_10": False,
             "outside_source_support_ge_0_10": False}]
    assert status_derivation.LARGE_COVARIATE_SHIFT_DETECTED in status_derivation.derive_shift_categories(rows)


def test_derive_shift_categories_no_material_shift():
    rows = [{"abs_smd_ge_0_5": False, "abs_smd_ge_0_2": False, "psi_ge_0_25": False, "psi_ge_0_10": False,
             "outside_source_support_ge_0_10": False}]
    assert status_derivation.derive_shift_categories(rows) == [status_derivation.NO_MATERIAL_SHIFT_DETECTED]


def test_derive_ranking_reversal_suspected():
    rows = [{"abs_smd_ge_0_5": True, "psi_ge_0_25": False, "outside_source_support_ge_0_10": True}]
    assert status_derivation.derive_ranking_reversal_suspected(rows) is True
    rows2 = [{"abs_smd_ge_0_5": True, "psi_ge_0_25": False, "outside_source_support_ge_0_10": False}]
    assert status_derivation.derive_ranking_reversal_suspected(rows2) is False


def test_bucket_reversal_status_unknown_is_unavailable():
    assert status_derivation.bucket_reversal_status("something_unexpected") == status_derivation.FEATURE_UNAVAILABLE
    assert status_derivation.bucket_reversal_status(status_derivation.BOOTSTRAP_SUPPORTED_DIRECTION_REVERSAL) == (
        status_derivation.BOOTSTRAP_SUPPORTED_DIRECTION_REVERSAL
    )


def test_no_banned_phrases_in_status_vocabulary():
    banned = ["partial_transfer_supported", "operational", "universal", "statistical significance", "causal concept shift"]
    all_labels = " ".join(
        status_derivation.RAW_TRANSFER_STATUSES
        + status_derivation.ADAPTATION_SUPPORT_STATUSES
        + status_derivation.RESIDUAL_GAP_STATUSES
        + status_derivation.RECOVERY_PATTERN_STATUSES
        + status_derivation.SHIFT_CATEGORY_VOCABULARY
        + status_derivation.REVERSAL_STATUS_VOCABULARY
    ).lower()
    for phrase in banned:
        assert phrase not in all_labels


# =============================================================================
# schema_adapters.py (small hand-built fixtures)
# =============================================================================
def test_adapt_step8_within_region():
    raw = {
        "population_metrics": {
            PRIMARY_POPULATION: {
                "overall_baseline": {"roc_auc": 0.7, "pr_auc": 0.2},
                "overall_thermal": {"roc_auc": 0.85, "pr_auc": 0.4},
            }
        }
    }
    normalized = schema_adapters.adapt_step8_within_region(raw, AOI_A)
    assert normalized["baseline"]["roc_auc"] == 0.7
    assert normalized["thermal"]["roc_auc"] == 0.85


def test_adapt_step8_within_region_missing_population_raises():
    with pytest.raises(InputResolutionError):
        schema_adapters.adapt_step8_within_region({"population_metrics": {}}, AOI_A)


def test_adapt_step9g_pair_resolves_variable_prefix():
    row = {
        "feature": "ndvi_mean",
        f"{AOI_A}_auc": 0.6, f"{AOI_A}_ci_low": 0.5, f"{AOI_A}_ci_high": 0.7, f"{AOI_A}_direction": "higher_values_rank_burned",
        f"{AOI_B}_auc": 0.4, f"{AOI_B}_ci_low": 0.3, f"{AOI_B}_ci_high": 0.5, f"{AOI_B}_direction": "lower_values_rank_burned",
        "point_direction_reversal": True,
        "reversal_status": status_derivation.BOOTSTRAP_SUPPORTED_DIRECTION_REVERSAL,
        "step9e_relationship_direction_flag": "True",
    }
    payload = {"schema": "v1", "report": {"primary_population": PRIMARY_POPULATION, "direction_reversal_table": [row]}}
    normalized = schema_adapters.adapt_step9g_pair(payload, AOI_A, AOI_B)
    feat = normalized["features"][0]
    assert feat["a"]["auc"] == 0.6
    assert feat["b"]["auc"] == 0.4
    assert feat["step9e_relationship_direction_flag"] is True


def test_adapt_step9g_pair_short_prefix_convention():
    # Some frozen files truncate the experiment id (e.g. "manavgat_2021" ->
    # "manavgat"); the adapter must find whichever prefix the row actually uses.
    short_a = AOI_A.rsplit("_", 1)[0]
    short_b = AOI_B.rsplit("_", 1)[0]
    row = {
        "feature": "ndvi_mean",
        f"{short_a}_auc": 0.6, f"{short_a}_ci_low": 0.5, f"{short_a}_ci_high": 0.7, f"{short_a}_direction": "d1",
        f"{short_b}_auc": 0.4, f"{short_b}_ci_low": 0.3, f"{short_b}_ci_high": 0.5, f"{short_b}_direction": "d2",
        "point_direction_reversal": False,
        "reversal_status": status_derivation.NO_DIRECTION_REVERSAL,
        "step9e_relationship_direction_flag": "False",
    }
    payload = {"schema": "v1", "report": {"primary_population": PRIMARY_POPULATION, "direction_reversal_table": [row]}}
    normalized = schema_adapters.adapt_step9g_pair(payload, AOI_A, AOI_B)
    feat = normalized["features"][0]
    assert feat["a"]["auc"] == 0.6
    assert feat["b"]["auc"] == 0.4


def test_adapt_step9g_pair_wrong_primary_population_raises():
    payload = {"schema": "v1", "report": {"primary_population": "all_valid", "direction_reversal_table": []}}
    with pytest.raises(InputResolutionError):
        schema_adapters.adapt_step9g_pair(payload, AOI_A, AOI_B)


def test_adapt_step10_pair_omits_brier_when_unavailable():
    raw = {
        "target_performance": [{
            "direction": f"{AOI_A}_to_{AOI_B}", "model_family": "baseline", "adaptation_method": "raw_source_only",
            "target_row_count": 100, "target_burned_count": 10, "roc_auc": 0.6,
            "roc_auc_bootstrap_95_percentile_ci": [0.55, 0.65], "pr_auc": 0.2,
            "pr_auc_bootstrap_95_percentile_ci": [0.15, 0.25],
            "brier": None, "brier_availability": "unavailable_in_frozen_outputs",
        }],
        "paired_adaptation_differences": [],
        "within_transfer_decomposition": [],
    }
    normalized = schema_adapters.adapt_step10_pair(raw, AOI_A, AOI_B)
    assert normalized["rows"][0]["brier_available"] is False
    assert normalized["rows"][0]["brier"] is None


# =============================================================================
# End-to-end build (small synthetic fixture tree under tmp_path)
# =============================================================================
def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


@pytest.fixture()
def synthetic_repo(tmp_path, monkeypatch):
    """Build a minimal, fully-synthetic 2-AOI frozen-output tree and point
    core.paths.PROJECT_ROOT / core.regions.get_experiment_output_root at it,
    so build_synthesis can be exercised end-to-end without touching any real
    project output."""
    root = tmp_path / "repo"
    a, b = AOI_A, AOI_B

    # Step8 within-region.
    for exp, base, thermal in ((a, 0.70, 0.85), (b, 0.65, 0.80)):
        _write_json(
            root / "outputs" / "experiments" / exp / "step8b" / "step8b_model_comparison_metrics.json",
            {"population_metrics": {PRIMARY_POPULATION: {
                "overall_baseline": {"roc_auc": base, "pr_auc": 0.2},
                "overall_thermal": {"roc_auc": thermal, "pr_auc": 0.4},
            }}},
        )

    # Step9D merged transfer report (single pair dir covers both directions).
    def transfer_entry(src, tgt, roc_ci, pr_ci):
        return {
            "transfer_direction": f"{src}_to_{tgt}",
            "source_experiment_id": src, "target_experiment_id": tgt,
            "primary_population": PRIMARY_POPULATION,
            "population_results": {PRIMARY_POPULATION: {
                "transfer": {"target_cell_count": 500, "target_positive_count": 40},
                "bootstrap": {"confidence_intervals": {
                    "baseline_roc_auc": {"ci_2_5": roc_ci[0], "ci_97_5": roc_ci[1]},
                    "thermal_roc_auc": {"ci_2_5": roc_ci[0] + 0.1, "ci_97_5": roc_ci[1] + 0.1},
                    "baseline_pr_auc": {"ci_2_5": pr_ci[0], "ci_97_5": pr_ci[1]},
                    "thermal_pr_auc": {"ci_2_5": pr_ci[0] + 0.1, "ci_97_5": pr_ci[1] + 0.1},
                }},
            }},
        }

    _write_json(
        root / "outputs" / "cross_region" / f"{a}__{b}" / "step9d" / "final_cross_region_report.json",
        {"primary_population": PRIMARY_POPULATION, "direction_summaries": [
            transfer_entry(a, b, (0.55, 0.65), (0.1, 0.2)),
            transfer_entry(b, a, (0.45, 0.50), (0.05, 0.1)),
        ]},
    )

    # Step9E shift audit -- only forward direction available (mirrors real
    # repo's occasional missing-reverse-direction case); build.py should
    # surface it as "shift_audit_unavailable" for the missing direction in
    # dry-run mode, not crash.
    _write_json(
        root / "outputs" / "cross_region" / f"{a}__{b}" / "step9e" / "distribution_shift_audit.json",
        {"source_experiment_id": a, "target_experiment_id": b, "primary_populations": [PRIMARY_POPULATION],
         "part_a_numeric_feature_shift": [
             {"feature": "ndvi_mean", "population": PRIMARY_POPULATION,
              "source_experiment_id": a, "target_experiment_id": b,
              "abs_smd_ge_0_5": False, "abs_smd_ge_0_2": True, "psi_ge_0_25": False, "psi_ge_0_10": True,
              "outside_source_support_ge_0_10": False},
         ]},
    )
    _write_json(
        root / "outputs" / "cross_region" / f"{a}__{b}" / "step9e_reverse_marker" / "unused.json", {"x": 1},
    )
    # Reverse direction step9e (same pair dir convention as forward; needed
    # for the non-dry-run full build to succeed).
    _write_json(
        root / "outputs" / "cross_region" / f"{b}__{a}" / "step9e" / "distribution_shift_audit.json",
        {"source_experiment_id": b, "target_experiment_id": a, "primary_populations": [PRIMARY_POPULATION],
         "part_a_numeric_feature_shift": [
             {"feature": "ndvi_mean", "population": PRIMARY_POPULATION,
              "source_experiment_id": b, "target_experiment_id": a,
              "abs_smd_ge_0_5": False, "abs_smd_ge_0_2": False, "psi_ge_0_25": False, "psi_ge_0_10": False,
              "outside_source_support_ge_0_10": False},
         ]},
    )

    # Step9G v1 pair report.
    _write_json(
        root / "outputs" / "diagnostics" / "step9g_univariate_feature_auc_direction_reversal" / f"{a}__{b}" / "step9g_final_report.json",
        {"schema_version": "step9g.univariate_feature_auc_direction_reversal.v1",
         "primary_population": PRIMARY_POPULATION,
         "direction_reversal_table": [{
             "feature": "ndvi_mean",
             f"{a}_auc": 0.6, f"{a}_ci_low": 0.5, f"{a}_ci_high": 0.7, f"{a}_direction": "higher_values_rank_burned",
             f"{b}_auc": 0.4, f"{b}_ci_low": 0.3, f"{b}_ci_high": 0.5, f"{b}_direction": "lower_values_rank_burned",
             "point_direction_reversal": True,
             "reversal_status": status_derivation.BOOTSTRAP_SUPPORTED_DIRECTION_REVERSAL,
             "step9e_relationship_direction_flag": "True",
         }]},
    )

    # Step10 pair report (both directions embedded).
    def perf_row(src, tgt, family, method, roc, roc_ci):
        return {
            "direction": f"{src}_to_{tgt}", "model_family": family, "adaptation_method": method,
            "target_row_count": 500, "target_burned_count": 40, "roc_auc": roc,
            "roc_auc_bootstrap_95_percentile_ci": roc_ci, "pr_auc": 0.2,
            "pr_auc_bootstrap_95_percentile_ci": [0.15, 0.25],
            "brier": None, "brier_availability": "unavailable_in_frozen_outputs",
        }

    target_performance = []
    adaptation_differences = []
    decomposition = []
    for src, tgt in ((a, b), (b, a)):
        for family in ("baseline", "thermal"):
            target_performance.append(perf_row(src, tgt, family, "raw_source_only", 0.55, [0.5, 0.6]))
            target_performance.append(perf_row(src, tgt, family, "regionwise_zscore", 0.62, [0.57, 0.67]))
            adaptation_differences.append({
                "direction": f"{src}_to_{tgt}", "model_family": family,
                "comparison": "regionwise_zscore_minus_raw_source_only",
                "point_estimate_difference": 0.07, "paired_bootstrap_95_percentile_ci": [0.02, 0.12],
            })
            decomposition.append({
                "direction": f"{src}_to_{tgt}", "model_family": family, "adapted_method": "regionwise_zscore",
                "metric": "roc_auc", "target_within_region_step8b_oof_metric": 0.85,
                "raw_transfer_metric": 0.55, "adapted_transfer_metric": 0.62,
                "remaining_transfer_gap": 0.23, "remaining_gap_95_paired_bootstrap_ci": [0.18, 0.28],
            })
    _write_json(
        root / "outputs" / "cross_region" / f"{a}__{b}" / "step10" / "step10_final_report.json",
        {"report_schema_version": "step10.final_report.v2", "analysis_id": "synthetic",
         "directions": [f"{a}_to_{b}", f"{b}_to_{a}"],
         "target_performance": target_performance,
         "paired_adaptation_differences": adaptation_differences,
         "within_transfer_decomposition": decomposition},
    )

    monkeypatch.setattr("src.multi_aoi_transfer_synthesis.resolvers.PROJECT_ROOT", root)
    monkeypatch.setattr(
        "src.multi_aoi_transfer_synthesis.resolvers.get_experiment_output_root",
        lambda experiment_id: root / "outputs" / "experiments" / experiment_id,
    )
    return root


def test_build_synthesis_end_to_end(synthetic_repo):
    from src.multi_aoi_transfer_synthesis.build import build_synthesis

    synthesis = build_synthesis([AOI_A, AOI_B], dry_run=False)

    assert synthesis["canonical_set_id"]
    assert len(synthesis["within_region"]) == 2 * 2  # 2 AOIs x 2 model families
    assert len(synthesis["transfer_matrix"]) > 0
    assert len(synthesis["feature_stability"]) == 1  # one feature, one pair

    for row in synthesis["transfer_matrix"]:
        assert row["primary_population"] == PRIMARY_POPULATION
        assert "brier" not in row or row.get("brier") is None

    feat_row = synthesis["feature_stability"][0]
    assert feat_row["reversal_status"] == status_derivation.BOOTSTRAP_SUPPORTED_DIRECTION_REVERSAL

    # No AOI-specific banned phrases anywhere in the rendered markdown.
    from src.multi_aoi_transfer_synthesis.render import render_markdown
    md = render_markdown(synthesis)
    lowered = md.lower()
    for phrase in ("partial_transfer_supported", "successful operational transfer",
                   "universal adaptation success", "statistical significance", "causal concept shift"):
        assert phrase not in lowered


def test_build_synthesis_dry_run_reports_missing_inputs_without_raising(synthetic_repo, tmp_path):
    from src.multi_aoi_transfer_synthesis.build import build_synthesis

    # A third AOI with no frozen outputs at all -- dry-run must surface this
    # as a resolution error, not raise.
    missing_aoi = AOI_C
    synthesis = build_synthesis([AOI_A, AOI_B, missing_aoi], dry_run=True)
    assert synthesis["dry_run"] is True
    assert len(synthesis["resolution_errors"]) > 0
    families_with_errors = {e["family"] for e in synthesis["resolution_errors"]}
    assert "step8_within_region" in families_with_errors


def test_build_synthesis_non_dry_run_raises_on_missing_input(synthetic_repo):
    from src.multi_aoi_transfer_synthesis.build import build_synthesis, SynthesisBuildError

    with pytest.raises(SynthesisBuildError):
        build_synthesis([AOI_A, AOI_B, AOI_C], dry_run=False)


def test_render_all_writes_expected_files(synthetic_repo, tmp_path):
    from src.multi_aoi_transfer_synthesis.build import build_synthesis
    from src.multi_aoi_transfer_synthesis.render import render_all
    from src.multi_aoi_transfer_synthesis.manifest import render_manifest

    synthesis = build_synthesis([AOI_A, AOI_B], dry_run=False)
    out_dir = tmp_path / "out"
    paths = render_all(synthesis, out_dir)
    for name in (
        "multi_aoi_transfer_synthesis.json", "multi_aoi_transfer_synthesis.md",
        "multi_aoi_transfer_matrix.csv", "multi_aoi_feature_stability.csv",
        "multi_aoi_within_region.csv",
    ):
        assert paths[name].is_file()

    manifest = render_manifest(synthesis, paths, out_dir / "multi_aoi_manifest.json")
    assert (out_dir / "multi_aoi_manifest.json").is_file()
    assert len(manifest["output_file_hashes"]) == 5
    assert len(manifest["resolved_inputs"]) > 0


# =============================================================================
# Integration-style test against REAL frozen outputs already on disk.
# Assertions are generic (resolution succeeds / status is in the known
# vocabulary), never hardcoded expected prose.
# =============================================================================
REAL_EXPERIMENTS_PRESENT = all(
    os.path.isdir(os.path.join("outputs", "experiments", exp))
    for exp in ("manavgat_2021", "bejis_2022", "mugla_2021")
)


@pytest.mark.skipif(not REAL_EXPERIMENTS_PRESENT, reason="Real frozen experiment outputs not present in this checkout.")
def test_resolvers_against_real_frozen_outputs():
    from src.multi_aoi_transfer_synthesis import resolvers

    exp_a, exp_b = "bejis_2022", "mugla_2021"

    record, raw = resolvers.resolve_step8_within_region(exp_a)
    assert record["primary_population"] == PRIMARY_POPULATION
    normalized = schema_adapters.adapt_step8_within_region(raw, exp_a)
    assert isinstance(normalized["baseline"]["roc_auc"], float)

    record, raw = resolvers.resolve_step9_transfer(exp_a, exp_b)
    normalized = schema_adapters.adapt_step9_transfer_step9d(raw, exp_a, exp_b) if \
        record["schema_version"] != "step9b_step9c_fallback" else \
        schema_adapters.adapt_step9_transfer_fallback(raw, exp_a, exp_b)
    for family in ("baseline", "thermal"):
        ci = normalized["model_family_metrics"][family]
        status = status_derivation.derive_raw_transfer_status(ci["roc_auc_ci_low"], ci["pr_auc_ci_low"])
        assert status in status_derivation.RAW_TRANSFER_STATUSES

    record, raw = resolvers.resolve_step9g_pair(exp_a, exp_b)
    normalized = schema_adapters.adapt_step9g_pair(raw, exp_a, exp_b)
    for feat in normalized["features"]:
        bucketed = status_derivation.bucket_reversal_status(feat["reversal_status"])
        assert bucketed in status_derivation.REVERSAL_STATUS_VOCABULARY

    record, raw = resolvers.resolve_step10_pair(exp_a, exp_b)
    normalized = schema_adapters.adapt_step10_pair(raw, exp_a, exp_b)
    assert len(normalized["rows"]) > 0
