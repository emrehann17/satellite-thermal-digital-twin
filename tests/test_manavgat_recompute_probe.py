"""Targeted tests for the Manavgat local recomputation determinism probe.

Side-effect free: nothing here fits a model, reads the real frozen namespace,
touches Earth Engine, or writes outside `tmp_path`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.multi_region_window_closure.contract import REFERENCE_AOI
from src.multi_region_window_closure import recompute_probe as rp
from src.multi_region_window_closure import reference_replay as rr


def _write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _synthetic_frozen_tree(root: Path) -> Path:
    """Covers every stage of the partition, including the two recomputed ones."""
    _write(root / "config" / "preregistration.json", '{"analysis_id": "frozen"}')
    _write(root / "variants" / "canonical" / "frozen_reference.json", "{}")
    _write(root / "variants" / "close_7d_earlier" / "export_plan.json", "{}")
    _write(root / "prelabel_censor" / "censoring_summary.json", "{}")
    _write(root / "variants" / "close_7d_earlier" / "predictor_export_metadata.json", "{}")
    _write(root / "variants" / "close_7d_earlier" / "data" / "modis" / "m.tif", "raster")
    _write(root / "variants" / "close_7d_earlier" / "local_downstream_metadata.json", "{}")
    _write(
        root / "variants" / "close_7d_earlier" / "downstream" / "step8a" /
        "step8a_500m_modeling_dataset.parquet", "parquet",
    )
    _write(root / "model" / "metrics" / "point_metrics.csv", "roc_auc\n0.8\n")
    _write(root / "model" / "model_stage_metadata.json", "{}")
    _write(root / "compare" / "tables" / "closure_changes.csv", "delta\n0.1\n")
    _write(root / "_quarantine" / "old" / "stale.json", "{}")
    return root


# =============================================================================
# The probe must never copy what it claims to recompute
# =============================================================================
def test_materialize_copies_upstream_and_excludes_model_and_compare(tmp_path):
    frozen = _synthetic_frozen_tree(tmp_path / "frozen")
    probe = tmp_path / "probe"
    before = rr.tree_snapshot(frozen)

    result = rp.materialize_upstream(frozen, probe)
    experiment_root = probe / REFERENCE_AOI

    assert (experiment_root / "config" / "preregistration.json").is_file()
    assert (experiment_root / "variants" / "close_7d_earlier" / "data" / "modis" / "m.tif").is_file()
    assert (
        experiment_root / "variants" / "close_7d_earlier" / "downstream" / "step8a" /
        "step8a_500m_modeling_dataset.parquet"
    ).is_file()
    # The whole point of the probe: these are NOT copied.
    assert not (experiment_root / "model").exists()
    assert not (experiment_root / "compare").exists()
    assert not (experiment_root / "_quarantine").exists()

    assert result["reused_artifact_count"] == 8
    assert result["excluded_frozen_artifact_count"] == 3
    assert sorted(result["excluded_frozen_artifacts"]) == [
        "compare/tables/closure_changes.csv",
        "model/metrics/point_metrics.csv",
        "model/model_stage_metadata.json",
    ]
    assert result["gee_queries_run"] is False
    assert rr.tree_snapshot(frozen) == before


def test_materialize_verifies_every_copy(tmp_path, monkeypatch):
    frozen = _synthetic_frozen_tree(tmp_path / "frozen")
    digests = iter(["aaa", "bbb"] * 40)
    monkeypatch.setattr(rp, "sha256_path", lambda path: next(digests))
    with pytest.raises(rp.RecomputeProbeError, match="UPSTREAM_REUSE_HASH_MISMATCH"):
        rp.materialize_upstream(frozen, tmp_path / "probe")


def test_materialize_refuses_a_preseeded_model_tree(tmp_path):
    frozen = _synthetic_frozen_tree(tmp_path / "frozen")
    probe = tmp_path / "probe"
    _write(probe / REFERENCE_AOI / "model" / "sneaked.csv", "x")
    with pytest.raises(rp.RecomputeProbeError, match="RECOMPUTED_TREE_PRESEEDED"):
        rp.materialize_upstream(frozen, probe)


def test_recomputed_and_reused_stages_partition_the_frozen_tree():
    assert tuple(rp.REUSED_FROZEN_STAGES) + tuple(rp.RECOMPUTED_STAGES) == rr.REPLAYED_STAGES
    assert set(rp.REUSED_FROZEN_STAGES) & set(rp.RECOMPUTED_STAGES) == set()


def test_probe_namespace_is_distinct_from_every_other_root():
    assert rp.PROBE_DIAGNOSTIC_NAMESPACE == "window_closure_region_recompute_probe"
    assert rp.PROBE_DIAGNOSTIC_NAMESPACE not in {
        "window_closure_sensitivity", "window_closure_region",
        "window_closure_region_replay", "window_closure_synthesis",
    }
    assert rp.probe_output_root().name == rp.PROBE_DIAGNOSTIC_NAMESPACE


# =============================================================================
# CLI guards
# =============================================================================
def test_probe_cli_requires_the_explicit_execution_guard():
    from scripts.run_manavgat_recompute_probe import build_parser, main

    assert build_parser().parse_args([]).execute_probe is False
    with pytest.raises(rp.RecomputeProbeError):
        main(["--preflight-only", "--execute-probe"])


def test_probe_cli_exposes_no_experiment_argument():
    from scripts.run_manavgat_recompute_probe import build_parser

    assert "experiment" not in {a.dest for a in build_parser()._actions}


# =============================================================================
# Binding construction is hash-verified, not assumed
# =============================================================================
def _variant_records() -> list[dict]:
    """The three variant records `nonzero_variants` expects."""
    return [
        {"variant_id": "canonical", "shift_days": 0, "is_canonical": True},
        {"variant_id": "close_7d_earlier", "shift_days": 7, "is_canonical": False},
        {"variant_id": "close_14d_earlier", "shift_days": 14, "is_canonical": False},
    ]


def _binding_fixture(tmp_path, *, dataset_hash_ok=True, canonical_pin_ok=True):
    import pandas as pd

    probe = tmp_path / "probe"
    experiment_root = probe / REFERENCE_AOI
    canonical_bytes = b"canonical-step8a"
    experiments = tmp_path / "experiments" / REFERENCE_AOI / "step8a"
    experiments.mkdir(parents=True)
    (experiments / "step8a_500m_modeling_dataset.parquet").write_bytes(canonical_bytes)
    (experiments / "step8a_dataset_stats.json").write_text("{}")
    canonical_hash = rp.sha256_path(experiments / "step8a_500m_modeling_dataset.parquet")

    for variant in ("close_7d_earlier", "close_14d_earlier"):
        step8a = experiment_root / "variants" / variant / "downstream" / "step8a"
        step8a.mkdir(parents=True)
        (step8a / "step8a_500m_modeling_dataset.parquet").write_bytes(f"{variant}".encode())
        (step8a / "step8a_dataset_stats.json").write_text("{}")
        digest = rp.sha256_path(step8a / "step8a_500m_modeling_dataset.parquet")
        _write(
            experiment_root / "variants" / variant / "local_downstream_metadata.json",
            json.dumps({
                "status": "pass", "experiment_id": REFERENCE_AOI, "variant_id": variant,
                "step8a_dataset_sha256": digest if dataset_hash_ok else "0" * 64,
                "step8a_stats_sha256": rp.sha256_path(
                    step8a / "step8a_dataset_stats.json"
                ),
                "canonical_step8a_sha256": canonical_hash if canonical_pin_ok else "0" * 64,
            }),
        )
    return probe, tmp_path / "experiments", canonical_hash


def test_model_binding_fails_closed_on_a_step8a_hash_mismatch(tmp_path, monkeypatch):
    probe, experiments, canonical_hash = _binding_fixture(tmp_path, dataset_hash_ok=False)
    from src.multi_region_window_closure import inputs

    monkeypatch.setitem(inputs.CANONICAL_STEP8A_SHA256, REFERENCE_AOI, canonical_hash)
    variants = _variant_records()
    with pytest.raises(rp.RecomputeProbeError, match="PROBE_MODEL_BINDING"):
        rp.build_model_binding(
            experiment_id=REFERENCE_AOI, variants=variants, probe_root=probe,
            experiments_root=experiments,
        )


def test_model_binding_fails_closed_on_a_canonical_pin_mismatch(tmp_path, monkeypatch):
    probe, experiments, canonical_hash = _binding_fixture(tmp_path, canonical_pin_ok=False)
    from src.multi_region_window_closure import inputs

    monkeypatch.setitem(inputs.CANONICAL_STEP8A_SHA256, REFERENCE_AOI, canonical_hash)
    variants = _variant_records()
    with pytest.raises(rp.RecomputeProbeError, match="pinned canonical hash"):
        rp.build_model_binding(
            experiment_id=REFERENCE_AOI, variants=variants, probe_root=probe,
            experiments_root=experiments,
        )


def test_model_binding_resolves_three_verified_datasets(tmp_path, monkeypatch):
    probe, experiments, canonical_hash = _binding_fixture(tmp_path)
    from src.multi_region_window_closure import inputs

    monkeypatch.setitem(inputs.CANONICAL_STEP8A_SHA256, REFERENCE_AOI, canonical_hash)
    variants = _variant_records()
    binding = rp.build_model_binding(
        experiment_id=REFERENCE_AOI, variants=variants, probe_root=probe,
        experiments_root=experiments,
    )
    assert set(binding["model_datasets"]) == {
        "canonical", "close_7d_earlier", "close_14d_earlier",
    }
    assert binding["canonical_step8a_sha256"] == canonical_hash
    # Shifted datasets must resolve INSIDE the probe, never in the frozen tree.
    for variant in ("close_7d_earlier", "close_14d_earlier"):
        assert str(probe) in binding["model_datasets"][variant]["dataset_path"]


# =============================================================================
# Comparators: a real scientific change must FAIL
# =============================================================================
def _tables(*, y_score=0.4, roc=0.87, point=0.01, draw=0.5):
    import pandas as pd

    return {
        "point_metrics": pd.DataFrame([{
            "variant_id": "canonical", "model_family": "thermal",
            "roc_auc": roc, "pr_auc": 0.22, "brier": 0.05,
        }]),
        "thermal_contributions": pd.DataFrame([{
            "variant_id": "canonical", "metric": "roc_auc",
            "baseline": 0.80, "thermal": roc, "contribution_delta": roc - 0.80,
            "delta_definition": "thermal - baseline (raw)",
            "sign_convention": "higher is better",
        }]),
        "bootstrap_summary": pd.DataFrame([{
            "comparison": "thermal_contribution_within_variant",
            "variant_id": "canonical", "model_family": "thermal", "metric": "roc_auc",
            "point_delta": point, "ci_low": -0.01, "ci_high": 0.03,
            "bootstrap_mean": 0.011, "valid_replicates": 1000, "invalid_replicates": 0,
            "block_count": 5350, "bootstrap_seed": 42, "requested_replicates": 1000,
        }]),
        "bootstrap_replicates": pd.DataFrame({"canonical__thermal_roc_auc": [draw, 0.6]}),
        "oof": pd.DataFrame([{
            "variant_id": "canonical", "model_family": "thermal", "cell_id": 1,
            "y_score": y_score, "y_true": 1, "fold_id": 0, "spatial_block_id": 3,
        }]),
        "folds": pd.DataFrame([{"cell_id": 1, "fold_id": 0, "spatial_block_id": 3}]),
        "cohort_metadata": {
            "final_common_cohort_rows": 1, "final_positive_rows": 1,
            "final_negative_rows": 0, "removed_label_mismatch": 0,
            "removed_static_invariance_failure": 0,
            "primary_population": "burnable_tree_shrub_grass",
            "initial_rows_by_variant": {"canonical": 1},
            "removed_not_valid_for_modeling": {"canonical": 0},
            "removed_outside_primary_population": {"canonical": 0},
            "removed_prelabel_censor": {"canonical": 0},
            "removed_missing_required_feature_union": {"canonical": 0},
            "removed_variant_only_keys": {"canonical": 0},
            "input_dataset_sha256": {"canonical": "a" * 64},
            "required_feature_union": ["ndvi_mean"],
        },
        "cohort": pd.DataFrame([{"cell_id": 1, "burned": 1}]),
    }


def test_identical_recomputation_is_scientifically_identical():
    old, new = _tables(), _tables()
    assert compare_all_equivalent(old, new)


def compare_all_equivalent(old, new) -> bool:
    return all([
        rp.compare_cohort_and_folds(old, new)["equivalent"],
        rp.compare_oof(old, new)["equivalent"],
        rp.compare_metrics(old, new)["equivalent"],
        rp.compare_bootstrap(old, new)["equivalent"],
    ])


def test_a_changed_y_score_is_a_material_failure():
    result = rp.compare_oof(_tables(), _tables(y_score=0.41))
    assert result["equivalent"] is False
    assert result["classification"] == rp.CLASS_MATERIAL
    assert result["parts"][0]["max_abs_diff"] == pytest.approx(0.01)


def test_a_changed_metric_is_a_material_failure():
    result = rp.compare_metrics(_tables(), _tables(roc=0.86))
    assert result["equivalent"] is False
    assert result["classification"] == rp.CLASS_MATERIAL


def test_a_changed_ci_or_point_estimate_is_a_material_failure():
    result = rp.compare_bootstrap(_tables(), _tables(point=0.02))
    assert result["equivalent"] is False
    assert result["summary"]["classification"] == rp.CLASS_MATERIAL


def test_a_changed_replicate_sequence_is_a_material_failure():
    result = rp.compare_bootstrap(_tables(), _tables(draw=0.55))
    assert result["equivalent"] is False
    assert result["replicates"]["classification"] == rp.CLASS_MATERIAL
    assert result["replicates"]["max_abs_diff"] == pytest.approx(0.05)


def test_a_changed_cohort_is_a_material_failure():
    old = _tables()
    new = _tables()
    new["cohort_metadata"]["final_positive_rows"] = 2
    result = rp.compare_cohort_and_folds(old, new)
    assert result["equivalent"] is False
    assert [r["field"] for r in result["unequal_fields"]] == ["cohort.final_positive_rows"]


def test_serialization_level_float_noise_stays_equivalent_but_is_not_exact():
    result = rp.compare_oof(_tables(), _tables(y_score=0.4 + 1e-15))
    assert result["equivalent"] is True
    assert result["classification"] == rp.CLASS_FLOAT
    assert 0 < result["parts"][0]["max_abs_diff"] < rp.FLOAT_SERIALIZATION_TOLERANCE


# =============================================================================
# Model binaries are reported, never a PASS criterion
# =============================================================================
def test_model_artifact_byte_comparison_is_reported_not_gating(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write(old / "model" / "a.json", "same")
    _write(new / "model" / "a.json", "same")
    _write(old / "model" / "b.json", "one")
    _write(new / "model" / "b.json", "two")
    result = rp.compare_model_artifact_hashes(old, new)
    assert result["is_pass_criterion"] is False
    assert result["artifact_count"] == 2
    assert result["byte_identical"] == 1
    assert result["not_byte_identical"] == ["model/b.json"]
    assert "equivalent" not in result


# =============================================================================
# Three-way chain
# =============================================================================
def _point_metrics_tree(root: Path, roc: float) -> Path:
    _write(
        root / "model" / "metrics" / "point_metrics.csv",
        "variant_id,model_family,roc_auc,pr_auc,brier\n"
        f"canonical,thermal,{roc},0.22,0.05\n",
    )
    return root


def test_three_way_headline_detects_a_divergent_leg(tmp_path):
    old = _point_metrics_tree(tmp_path / "old", 0.87)
    fresh = _point_metrics_tree(tmp_path / "fresh", 0.87)
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "metrics.csv").write_text(
        "variant,model,metric,estimate\ncanonical,thermal,roc_auc,0.87\n"
        "canonical,thermal,pr_auc,0.22\ncanonical,thermal,brier,0.05\n"
    )
    agreeing = rp.three_way_headline(old, replay, fresh)
    assert agreeing["all_three_equal"] is True
    assert agreeing["disagreements"] == []

    diverged = _point_metrics_tree(tmp_path / "diverged", 0.86)
    result = rp.three_way_headline(old, replay, diverged)
    assert result["all_three_equal"] is False
    assert result["disagreements"][0]["fresh_recomputation"] == 0.86


def test_three_way_reports_when_the_replay_leg_is_absent(tmp_path):
    old = _point_metrics_tree(tmp_path / "old", 0.87)
    fresh = _point_metrics_tree(tmp_path / "fresh", 0.87)
    result = rp.three_way_headline(old, None, fresh)
    assert result["replay_available"] is False
    assert result["all_three_equal"] is False


# =============================================================================
# Mutation guard covers the replay tree too
# =============================================================================
def test_probe_guard_covers_every_live_window_closure_namespace():
    """Post-migration the reference is guarded under the regional root too."""
    from src.multi_region_window_closure.contract import ACTUAL_AOIS

    names = set(rp.probe_guarded_trees())
    regional = {n for n in names if n.startswith("regional/")}
    # One namespace per actual AOI, plus the read-only reference.
    for aoi in (*ACTUAL_AOIS, REFERENCE_AOI):
        assert any(n.startswith(f"regional/{aoi}/") for n in regional), aoi
    assert len(regional) == len(ACTUAL_AOIS) + 1
    # The retired frozen path is guarded only while it still exists, and the
    # replay root only while an un-promoted replay is present.
    from src.multi_region_window_closure.reference_replay import frozen_manavgat_root
    assert any(n.startswith("frozen/") for n in names) == frozen_manavgat_root().exists()
    assert any(n.startswith("replay/") for n in names) == (
        rp.replay_namespace_root() is not None
    )


# =============================================================================
# Compare-table comparison must separate ULP noise from a changed word
# =============================================================================
def _compare_tree(root: Path, *, value: str, label: str) -> Path:
    _write(
        root / "compare" / "tables" / "closure_changes.csv",
        f"variant,interval_status,closure_delta\ncanonical,{label},{value}\n",
    )
    _write(root / "compare" / "report" / "window_closure_comparison.md", "report\n")
    return root


def test_compare_tables_treats_ulp_noise_as_float_equivalent(tmp_path):
    old = _compare_tree(tmp_path / "old", value="0.123456789012345", label="interval_includes_zero")
    new = _compare_tree(
        tmp_path / "new", value="0.1234567890123451", label="interval_includes_zero",
    )
    result = rp.compare_compare_tables(old, new)
    assert result["equivalent"] is True
    assert result["classification"] == rp.CLASS_FLOAT
    assert result["material_difference_tables"] == []
    assert 0 < result["max_abs_diff"] <= rp.FLOAT_SERIALIZATION_TOLERANCE
    assert result["byte_identical_tables"] == 0


def test_compare_tables_flags_a_changed_interval_status_as_material(tmp_path):
    old = _compare_tree(tmp_path / "old", value="0.1", label="interval_includes_zero")
    new = _compare_tree(tmp_path / "new", value="0.1", label="bootstrap_supported_increase")
    result = rp.compare_compare_tables(old, new)
    assert result["equivalent"] is False
    assert result["classification"] == rp.CLASS_MATERIAL
    assert result["tables"][0]["non_numeric_mismatched_columns"] == ["interval_status"]


def test_compare_tables_flags_a_substantive_numeric_change_as_material(tmp_path):
    old = _compare_tree(tmp_path / "old", value="0.1", label="interval_includes_zero")
    new = _compare_tree(tmp_path / "new", value="0.2", label="interval_includes_zero")
    result = rp.compare_compare_tables(old, new)
    assert result["equivalent"] is False
    assert result["classification"] == rp.CLASS_MATERIAL
    assert result["max_abs_diff"] == pytest.approx(0.1)


def test_compare_tables_flags_a_missing_table(tmp_path):
    old = _compare_tree(tmp_path / "old", value="0.1", label="interval_includes_zero")
    new = tmp_path / "new"
    (new / "compare" / "tables").mkdir(parents=True)
    result = rp.compare_compare_tables(old, new)
    assert result["equivalent"] is False
    assert result["tables"][0]["present"] is False


def test_identical_compare_trees_are_scientifically_identical(tmp_path):
    old = _compare_tree(tmp_path / "old", value="0.1", label="interval_includes_zero")
    new = _compare_tree(tmp_path / "new", value="0.1", label="interval_includes_zero")
    result = rp.compare_compare_tables(old, new)
    assert result["classification"] == rp.CLASS_SCIENTIFIC
    assert result["exactly_equal_tables"] == 1
    assert result["byte_identical_tables"] == 1
    assert result["report_byte_identical"] is True
