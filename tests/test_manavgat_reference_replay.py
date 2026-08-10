"""Targeted tests for the Manavgat reference replay.

Everything here is side-effect free: no test touches the frozen Manavgat
namespace, the four production regional namespaces, Earth Engine, or a model
library. The replay engine is exercised against a synthetic frozen tree in
`tmp_path`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS, MultiRegionWindowClosureError, REFERENCE_AOI,
    active_reference_replay_aoi, aoi_role, assert_regional_aoi,
    reference_replay_scope,
)
from src.multi_region_window_closure import reference_replay as rr


# =============================================================================
# Scope guards -- the replay must not widen the production contract
# =============================================================================
def test_normal_regional_gate_still_rejects_manavgat():
    with pytest.raises(MultiRegionWindowClosureError, match="read-only synthesis reference"):
        assert_regional_aoi(REFERENCE_AOI)
    assert active_reference_replay_aoi() is None


def test_production_regional_cli_cannot_select_manavgat():
    from scripts.run_window_closure_region import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--experiment", REFERENCE_AOI, "--dry-run"])


def test_explicit_replay_scope_admits_only_manavgat():
    with reference_replay_scope(REFERENCE_AOI):
        assert assert_regional_aoi(REFERENCE_AOI) == REFERENCE_AOI
        # Every other AOI keeps its normal treatment inside the scope.
        assert assert_regional_aoi("bejis_2022") == "bejis_2022"
        with pytest.raises(MultiRegionWindowClosureError):
            assert_regional_aoi("evia_2021")
        with pytest.raises(MultiRegionWindowClosureError):
            assert_regional_aoi("some_new_aoi")
    assert active_reference_replay_aoi() is None


@pytest.mark.parametrize("aoi", ["bejis_2022", "evia_2021", "mugla_2021", "manavgat_2022", ""])
def test_replay_scope_refuses_any_other_aoi(aoi):
    with pytest.raises(MultiRegionWindowClosureError, match="REFERENCE_REPLAY_SCOPE_INVALID"):
        with reference_replay_scope(aoi):
            pass


def test_replay_scope_is_released_on_exception():
    with pytest.raises(ValueError):
        with reference_replay_scope(REFERENCE_AOI):
            raise ValueError("boom")
    assert active_reference_replay_aoi() is None
    with pytest.raises(MultiRegionWindowClosureError):
        assert_regional_aoi(REFERENCE_AOI)


def test_replay_never_changes_the_aoi_scope_or_roles():
    assert REFERENCE_AOI not in ACTUAL_AOIS
    assert ACTUAL_AOIS == ("bejis_2022", "evia_2021_extended", "montiferru_2021", "mugla_2021")
    with reference_replay_scope(REFERENCE_AOI):
        assert aoi_role(REFERENCE_AOI) == "read_only_reference"
        assert aoi_role("evia_2021_extended") == "different_regime_control"
        assert aoi_role("bejis_2022") == "new_actual"
    assert aoi_role(REFERENCE_AOI) == "read_only_reference"


def test_replay_cli_requires_the_explicit_execution_guard():
    from scripts.run_manavgat_reference_replay import build_parser, main

    parsed = build_parser().parse_args([])
    assert parsed.execute_replay is False
    with pytest.raises(MultiRegionWindowClosureError):
        main(["--preflight-only", "--execute-replay"])
    with pytest.raises(MultiRegionWindowClosureError):
        main(["--force"])
    with pytest.raises(MultiRegionWindowClosureError):
        main(["--execute-replay", "--resume", "--force"])


def test_replay_cli_exposes_no_experiment_argument():
    from scripts.run_manavgat_reference_replay import build_parser

    options = {action.dest for action in build_parser()._actions}
    assert "experiment" not in options


# =============================================================================
# Stage partition -- every reused artefact is attributed to a stage
# =============================================================================
@pytest.mark.parametrize("relative,stage", [
    ("config/preregistration.json", "plan"),
    ("variants/canonical/frozen_reference.json", "plan"),
    ("variants/close_7d_earlier/export_plan.json", "plan"),
    ("prelabel_censor/prelabel_burndate.tif", "export"),
    ("variants/close_7d_earlier/predictor_export_metadata.json", "export"),
    ("variants/close_7d_earlier/data/modis/modis_lst_mean_celsius.tif", "export"),
    ("variants/close_7d_earlier/local_downstream_metadata.json", "local-downstream"),
    ("variants/close_7d_earlier/downstream/step8a/step8a_500m_modeling_dataset.parquet",
     "local-downstream"),
    ("model/metrics/point_metrics.csv", "fit"),
    ("compare/tables/point_metrics_long.csv", "compare"),
])
def test_stage_partition_attributes_known_artifacts(relative, stage):
    assert rr.stage_of_frozen_artifact(Path(relative)) == stage


def test_stage_partition_fails_closed_on_an_unknown_artifact():
    with pytest.raises(rr.ReferenceReplayError, match="UNCLASSIFIED_FROZEN_ARTIFACT"):
        rr.stage_of_frozen_artifact(Path("something_new/file.tif"))


def _write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _synthetic_frozen_tree(root: Path) -> Path:
    _write(root / "config" / "preregistration.json", '{"analysis_id": "frozen"}')
    _write(root / "model" / "metrics" / "point_metrics.csv", "roc_auc\n0.8\n")
    _write(root / "compare" / "tables" / "closure_changes.csv", "delta\n0.1\n")
    _write(root / "prelabel_censor" / "censoring_summary.json", "{}")
    _write(root / "variants" / "canonical" / "frozen_reference.json", "{}")
    _write(root / "variants" / "close_7d_earlier" / "predictor_export_metadata.json", "{}")
    _write(root / "variants" / "close_7d_earlier" / "local_downstream_metadata.json", "{}")
    _write(root / "_quarantine" / "old" / "stale.json", "{}")
    return root


def test_partition_covers_every_non_quarantine_file(tmp_path):
    root = _synthetic_frozen_tree(tmp_path / "frozen")
    partition = rr.partition_frozen_source(root)
    assert sum(len(v) for v in partition.values()) == 7
    flat = {str(p) for values in partition.values() for p in values}
    assert not any("_quarantine" in name for name in flat)
    assert set(partition) == set(rr.REPLAYED_STAGES)


def test_materialize_verifies_every_copy_and_never_writes_the_source(tmp_path):
    frozen = _synthetic_frozen_tree(tmp_path / "frozen")
    before = rr.tree_snapshot(frozen)
    with reference_replay_scope(REFERENCE_AOI):
        engine = rr.ManavgatReferenceReplayEngine(frozen_source=frozen)
        root = tmp_path / "replay"
        root.mkdir()
        context = {"aoi": REFERENCE_AOI, "analysis_id": "replay-id"}
        for stage in rr.REPLAYED_STAGES:
            detail = engine.run_stage(stage, root, context)
            assert detail["recomputed"] is False
            assert detail["gee_queries_run"] is False
    production = root / "_production" / REFERENCE_AOI
    for relative in (p.relative_to(frozen) for p in rr.frozen_source_files(frozen)):
        assert (production / relative).is_file()
        assert rr.sha256_path(production / relative) == rr.sha256_path(frozen / relative)
    assert not (production / "_quarantine").exists()
    assert rr.tree_snapshot(frozen) == before


def test_materialize_fails_closed_when_a_copy_does_not_verify(tmp_path, monkeypatch):
    frozen = _synthetic_frozen_tree(tmp_path / "frozen")
    with reference_replay_scope(REFERENCE_AOI):
        engine = rr.ManavgatReferenceReplayEngine(frozen_source=frozen)
        root = tmp_path / "replay"
        root.mkdir()
        digests = iter(["aaa", "bbb"] * 20)
        monkeypatch.setattr(rr, "sha256_path", lambda path: next(digests))
        with pytest.raises(rr.ReferenceReplayError, match="FROZEN_REUSE_HASH_MISMATCH"):
            engine.run_stage("plan", root, {"aoi": REFERENCE_AOI, "analysis_id": "x"})


def test_replay_write_target_can_never_be_a_canonical_namespace(tmp_path):
    frozen = _synthetic_frozen_tree(tmp_path / "frozen")
    with reference_replay_scope(REFERENCE_AOI):
        engine = rr.ManavgatReferenceReplayEngine(frozen_source=frozen)
        ok = tmp_path / "window_closure_region_replay" / REFERENCE_AOI / "id"
        ok.mkdir(parents=True)
        assert engine._production_root(ok) == ok / "_production"
        for forbidden in (
            tmp_path / "outputs" / "diagnostics" / "window_closure_sensitivity" / REFERENCE_AOI,
            tmp_path / "outputs" / "diagnostics" / "window_closure_region" / "bejis_2022" / "id",
        ):
            forbidden.mkdir(parents=True)
            with pytest.raises(rr.ReferenceReplayError, match="REPLAY_WRITE_TARGET_FORBIDDEN"):
                engine._production_root(forbidden)


def test_replay_namespace_is_distinct_from_both_canonical_roots():
    assert rr.REPLAY_DIAGNOSTIC_NAMESPACE == "window_closure_region_replay"
    assert rr.REPLAY_DIAGNOSTIC_NAMESPACE not in {
        "window_closure_sensitivity", "window_closure_region", "window_closure_synthesis",
    }
    assert "replay" in rr.REPLAY_DIAGNOSTIC_NAMESPACE
    root = rr.replay_output_root()
    assert root.name == rr.REPLAY_DIAGNOSTIC_NAMESPACE
    assert rr.frozen_manavgat_root().parent.name == "window_closure_sensitivity"


def test_summarize_stage_delegates_to_the_real_regional_normalizer(tmp_path, monkeypatch):
    """The wrapper stage must be executed, not reused, or the replay proves nothing."""
    frozen = _synthetic_frozen_tree(tmp_path / "frozen")
    calls = []
    monkeypatch.setattr(
        rr, "normalize_production_regional_outputs",
        lambda root, context: calls.append((Path(root), dict(context))) or {"normalized": True},
    )
    with reference_replay_scope(REFERENCE_AOI):
        engine = rr.ManavgatReferenceReplayEngine(frozen_source=frozen)
        detail = engine.run_stage(
            "summarize", tmp_path / "replay", {"aoi": REFERENCE_AOI, "analysis_id": "x"},
        )
    assert len(calls) == 1
    assert detail["recomputed"] is True
    assert detail["replay_mode"] == rr.REPLAY_MODE_ID


def test_replay_engine_refuses_a_foreign_stage_context(tmp_path):
    frozen = _synthetic_frozen_tree(tmp_path / "frozen")
    with reference_replay_scope(REFERENCE_AOI):
        engine = rr.ManavgatReferenceReplayEngine(frozen_source=frozen)
        with pytest.raises(rr.ReferenceReplayError):
            engine.run_stage("plan", tmp_path, {"aoi": "bejis_2022"})
        with pytest.raises(rr.ReferenceReplayError):
            engine.run_stage("nonsense", tmp_path, {"aoi": REFERENCE_AOI})


# =============================================================================
# Contract preflight -- fail-closed on drift
# =============================================================================
def test_preflight_stops_on_a_frozen_step8a_hash_mismatch(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    canonical = tmp_path / "step8a.parquet"
    canonical.write_bytes(b"canonical-bytes")
    real_hash = rr.sha256_path(canonical)
    _write(frozen / "config" / "frozen_input_inventory.json", json.dumps(
        {"inventory": {"canonical_step8a": {"path": str(canonical)}}}
    ))
    _write(frozen / "config" / "preregistration.json", json.dumps({
        "analysis_id": "frozen",
        "scientific_configuration": _minimal_config(sha256="0" * 64),
    }))
    monkeypatch.setattr(rr, "run_analysis_for_preflight", None, raising=False)
    monkeypatch.setitem(rr.CANONICAL_STEP8A_SHA256, REFERENCE_AOI, real_hash)
    monkeypatch.setattr(
        "src.window_closure_sensitivity.run_analysis",
        lambda **kw: _minimal_plan(str(canonical), real_hash),
    )
    result = rr.replay_contract_preflight(frozen)
    assert result["status"] == "STOP"
    assert result["verdict"].startswith("STOP")
    fields = {m["field"] for m in result["mismatches"]}
    assert "step8a.sha256_frozen_vs_central" in fields
    assert "step8a.sha256_frozen_vs_actual_bytes" in fields


def test_preflight_stops_on_a_predictor_window_mismatch(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    canonical = tmp_path / "step8a.parquet"
    canonical.write_bytes(b"canonical-bytes")
    real_hash = rr.sha256_path(canonical)
    _write(frozen / "config" / "frozen_input_inventory.json", json.dumps(
        {"inventory": {"canonical_step8a": {"path": str(canonical)}}}
    ))
    drifted = _minimal_config(sha256=real_hash)
    drifted["variants"][0]["predictor_start_date"] = "1999-01-01"
    _write(frozen / "config" / "preregistration.json", json.dumps(
        {"analysis_id": "frozen", "scientific_configuration": drifted}
    ))
    monkeypatch.setitem(rr.CANONICAL_STEP8A_SHA256, REFERENCE_AOI, real_hash)
    monkeypatch.setattr(
        "src.window_closure_sensitivity.run_analysis",
        lambda **kw: _minimal_plan(str(canonical), real_hash),
    )
    result = rr.replay_contract_preflight(frozen)
    assert result["status"] == "STOP"
    assert any("predictor_start" in m["field"] for m in result["mismatches"])


def _minimal_config(*, sha256: str) -> dict:
    """A `scientific_configuration` shaped like the frozen Manavgat one."""
    from src.multi_region_window_closure.contract import (
        frozen_bootstrap_configuration, frozen_model_configuration,
    )
    from src.multi_region_window_closure.dates import window_date_rows

    rows = {row["variant"]: row for row in window_date_rows((REFERENCE_AOI,), (0, 7, 14))}
    model = frozen_model_configuration()
    bootstrap = frozen_bootstrap_configuration()
    return {
        "git_commit": "deadbeef",
        "primary_population": model["primary_population"],
        "frozen_input_sha256": {"canonical_step8a": sha256},
        "label_window": {
            "start_date": rows["canonical"]["label_start"],
            "end_date": rows["canonical"]["label_end"],
        },
        "variants": [
            {
                "variant_id": variant,
                "predictor_start_date": rows[variant]["predictor_start"],
                "predictor_end_date": rows[variant]["predictor_end"],
                "duration_days": rows[variant]["calendar_duration_days"],
                "lead_days": rows[variant]["lead_days"],
                "shift_days": rows[variant]["shift_days"],
            }
            for variant in ("canonical", "close_7d_earlier", "close_14d_earlier")
        ],
        "model_configuration": {
            "model": model["model"], "n_splits": model["n_splits"],
            "random_seed": model["fold_random_seed"],
            "spatial_block_size_cells": model["spatial_block_size_cells"],
            "min_positives": model["min_positives"],
        },
        "bootstrap_configuration": {
            key: bootstrap[key] for key in
            ("unit", "n_bootstrap", "seed", "identical_block_draws_across_variants")
        },
        "feature_registry": {"registry": "frozen"},
        "common_cohort_rule": {"rule": "frozen"},
    }


def _minimal_plan(step8a_path: str, sha256: str) -> dict:
    config = _minimal_config(sha256=sha256)
    config["git_commit"] = "cafebabe"
    return {
        "analysis_id": "derived",
        "scientific_configuration": config,
        "frozen_canonical_step8a": {"path": step8a_path, "sha256": sha256},
    }


def test_preflight_passes_when_only_git_commit_differs(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    canonical = tmp_path / "step8a.parquet"
    canonical.write_bytes(b"canonical-bytes")
    real_hash = rr.sha256_path(canonical)
    _write(frozen / "config" / "frozen_input_inventory.json", json.dumps(
        {"inventory": {"canonical_step8a": {"path": str(canonical)}}}
    ))
    _write(frozen / "config" / "preregistration.json", json.dumps(
        {"analysis_id": "frozen", "scientific_configuration": _minimal_config(sha256=real_hash)}
    ))
    monkeypatch.setitem(rr.CANONICAL_STEP8A_SHA256, REFERENCE_AOI, real_hash)
    monkeypatch.setattr(
        "src.window_closure_sensitivity.run_analysis",
        lambda **kw: _minimal_plan(str(canonical), real_hash),
    )
    result = rr.replay_contract_preflight(frozen)
    assert result["status"] == "PASS"
    assert [c["field"] for c in result["non_scientific_differences"]] == [
        "scientific_configuration.git_commit"
    ]


# =============================================================================
# Comparator -- material vs metadata-only differences
# =============================================================================
def test_numeric_comparator_reports_a_material_metric_difference():
    old = {("canonical", "thermal", "roc_auc"): 0.870828}
    new = {("canonical", "thermal", "roc_auc"): 0.860000}
    result = rr._numeric_diffs("point_metrics", old, new)
    assert result["classification"] == rr.CLASS_MATERIAL
    assert result["equivalent"] is False
    assert result["max_abs_diff"] == pytest.approx(0.010828)


def test_numeric_comparator_absorbs_only_serialization_level_float_noise():
    old = {"a": 0.5}
    new = {"a": 0.5 + 1e-15}
    result = rr._numeric_diffs("point_metrics", old, new)
    assert result["classification"] == rr.CLASS_FLOAT
    assert result["equivalent"] is True

    louder = rr._numeric_diffs("point_metrics", {"a": 0.5}, {"a": 0.5 + 1e-9})
    assert louder["classification"] == rr.CLASS_MATERIAL
    assert louder["equivalent"] is False


def test_numeric_comparator_flags_a_row_key_change():
    result = rr._numeric_diffs("point_metrics", {"a": 1.0}, {"b": 1.0})
    assert result["row_key_equal"] is False
    assert result["classification"] == rr.CLASS_MATERIAL


def test_metadata_only_difference_is_not_a_scientific_failure(tmp_path):
    """Path/analysis-id metadata differs by construction; that is not a FAIL."""
    replay = tmp_path / "replay"
    replay.mkdir()
    for name in rr.classify_replay_only_artifacts.__doc__ and []:  # pragma: no cover
        pass
    for name in (
        "config.json", "input_hashes.json", "repository_inventory.json",
        "window_dates.csv", "export_plan.csv", "cohort_inventory.csv",
        "fold_mapping.parquet", "variant_artifact_index.csv", "metrics.csv",
        "oof_predictions.parquet", "bootstrap_replicates.parquet",
        "bootstrap_summary.csv", "regional_summary.csv", "summary.json",
        "report.md", "manifest.json", "manifest.sha256",
        "validator_results.json", "validator_summary.json",
        "equivalence_report.json", "equivalence_report.md",
    ):
        _write(replay / name, "x")
    _write(replay / "stages" / "plan.json", "{}")
    _write(replay / "_production" / REFERENCE_AOI / "model" / "m.csv", "x")
    result = rr.classify_replay_only_artifacts(replay)
    assert result["unclassified"] == []
    assert result["equivalent"] is True
    assert result["new_scientific_output"] == []
    assert "_production/manavgat_2021/model/m.csv" not in result["classified"]


def test_unknown_replay_artifact_is_reported_rather_than_ignored(tmp_path):
    replay = tmp_path / "replay"
    _write(replay / "surprise.csv", "x")
    result = rr.classify_replay_only_artifacts(replay)
    assert result["unclassified"] == ["surprise.csv"]
    assert result["equivalent"] is False


def test_frozen_reuse_comparator_detects_a_dropped_or_altered_artifact(tmp_path):
    frozen = _synthetic_frozen_tree(tmp_path / "frozen")
    replay = tmp_path / "replay"
    production = replay / "_production" / REFERENCE_AOI
    for path in rr.frozen_source_files(frozen):
        target = production / path.relative_to(frozen)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    assert rr.compare_frozen_reuse(frozen, replay)["classification"] == rr.CLASS_EXACT

    (production / "model" / "metrics" / "point_metrics.csv").write_text("roc_auc\n0.1\n")
    altered = rr.compare_frozen_reuse(frozen, replay)
    assert altered["classification"] == rr.CLASS_MATERIAL
    assert altered["equivalent"] is False
    assert len(altered["hash_mismatches"]) == 1

    (production / "model" / "metrics" / "point_metrics.csv").unlink()
    dropped = rr.compare_frozen_reuse(frozen, replay)
    assert dropped["missing_in_replay"] == ["model/metrics/point_metrics.csv"]
    assert dropped["equivalent"] is False


# =============================================================================
# Source mutation guard
# =============================================================================
def test_mutation_guard_detects_any_content_or_structure_change(tmp_path):
    tree = _synthetic_frozen_tree(tmp_path / "frozen")
    before = {"t": rr.tree_snapshot(tree, ("config/preregistration.json",))}
    assert rr.compare_guarded_snapshots(before, {"t": rr.tree_snapshot(
        tree, ("config/preregistration.json",))})["all_unchanged"] is True

    (tree / "config" / "preregistration.json").write_text('{"analysis_id": "tampered"}')
    changed = rr.compare_guarded_snapshots(
        before, {"t": rr.tree_snapshot(tree, ("config/preregistration.json",))}
    )
    assert changed["all_unchanged"] is False
    assert changed["changed_trees"][0]["tree"] == "t"


def test_mutation_guard_detects_a_new_file(tmp_path):
    tree = _synthetic_frozen_tree(tmp_path / "frozen")
    before = {"t": rr.tree_snapshot(tree)}
    _write(tree / "extra.json", "{}")
    assert rr.compare_guarded_snapshots(before, {"t": rr.tree_snapshot(tree)})["all_unchanged"] is False


# =============================================================================
# Four-AOI production behaviour is unchanged
# =============================================================================
def test_regional_summary_role_matches_the_frozen_contract_for_every_aoi():
    """`aoi_role` replaced an inline evia test; the four AOIs must be unaffected."""
    expected = {
        "bejis_2022": "new_actual", "mugla_2021": "new_actual",
        "montiferru_2021": "new_actual",
        "evia_2021_extended": "different_regime_control",
    }
    for aoi, role in expected.items():
        assert aoi_role(aoi) == role
    with reference_replay_scope(REFERENCE_AOI):
        assert aoi_role(REFERENCE_AOI) == "read_only_reference"


def test_production_stage_map_and_regional_stages_are_untouched():
    from src.multi_region_window_closure.driver import REGIONAL_STAGES
    from src.multi_region_window_closure.production import PRODUCTION_STAGE_MAP

    assert REGIONAL_STAGES == (
        "plan", "export", "local-downstream", "fit", "compare", "summarize", "validate",
    )
    assert tuple(PRODUCTION_STAGE_MAP) == ("plan", "export", "local-downstream", "fit", "compare")
    assert tuple(rr.REPLAYED_STAGES) + tuple(rr.EXECUTED_STAGES) == REGIONAL_STAGES


def test_regional_validator_registry_is_unchanged():
    from src.multi_region_window_closure.validators import (
        REGIONAL_CHECK_COUNTS, REGIONAL_CHECKS,
    )

    assert REGIONAL_CHECK_COUNTS == {"total": 32, "required": 31, "advisory": 1}
    assert REGIONAL_CHECKS[0].check_id == "REG-SCOPE-ONE-AOI"


# =============================================================================
# Post-migration: the reference now shares the regional root but not the role
# =============================================================================
def test_canonical_target_guard_protects_the_migrated_reference_namespace():
    """Unifying the physical root must not make the reference writable."""
    from core.paths import PROJECT_ROOT
    from src.multi_region_window_closure.contract import REGIONAL_DIAGNOSTIC_NAMESPACE
    from src.multi_region_window_closure.stages import assert_not_canonical_target

    diagnostics = Path(PROJECT_ROOT) / "outputs" / "diagnostics"
    reference = diagnostics / REGIONAL_DIAGNOSTIC_NAMESPACE / REFERENCE_AOI

    for blocked in (
        reference,
        reference / "some_analysis_id" / "model",
        diagnostics / "window_closure_sensitivity",
        Path(PROJECT_ROOT) / "outputs" / "experiments" / REFERENCE_AOI,
    ):
        with pytest.raises(MultiRegionWindowClosureError, match="CANONICAL_OVERWRITE"):
            assert_not_canonical_target(blocked)

    # The four actual AOIs must stay writable in their own namespaces.
    for aoi in ACTUAL_AOIS:
        assert_not_canonical_target(
            diagnostics / REGIONAL_DIAGNOSTIC_NAMESPACE / aoi / "an_analysis_id"
        )


def test_reference_keeps_its_role_after_sharing_the_regional_root():
    from src.multi_region_window_closure.contract import REGIONAL_DIAGNOSTIC_NAMESPACE

    assert REGIONAL_DIAGNOSTIC_NAMESPACE == "window_closure_region"
    assert REFERENCE_AOI == "manavgat_2021"
    assert REFERENCE_AOI not in ACTUAL_AOIS
    assert aoi_role(REFERENCE_AOI) == "read_only_reference"
    with pytest.raises(MultiRegionWindowClosureError):
        assert_regional_aoi(REFERENCE_AOI)


def test_window_closure_sensitivity_module_is_still_the_regional_backend():
    """Only the OUTPUT namespace was retired; the implementation must survive."""
    import src.window_closure_sensitivity as wcs
    from src.multi_region_window_closure.production import ProductionRegionalEngine

    assert callable(wcs.run_analysis)
    assert wcs.SCHEMA_VERSION == "window_closure_sensitivity.v1"
    assert ProductionRegionalEngine(aoi="bejis_2022").runner is wcs.run_analysis


# =============================================================================
# Retired entry points must never recreate the retired output namespace
# =============================================================================
def _retired_root() -> Path:
    from core.paths import PROJECT_ROOT
    from src.multi_region_window_closure.contract import RETIRED_DIAGNOSTIC_NAMESPACE

    return Path(PROJECT_ROOT) / "outputs" / "diagnostics" / RETIRED_DIAGNOSTIC_NAMESPACE


def test_retired_output_namespace_does_not_exist():
    assert not _retired_root().exists()


def test_main_cli_retired_command_is_non_zero_and_writes_nothing(capsys):
    from scripts.main import build_parser, cmd_window_closure_sensitivity

    before = _retired_root().exists()
    args = build_parser().parse_args(
        ["window-closure-sensitivity", "--experiment", REFERENCE_AOI, "--dry-run"]
    )
    assert cmd_window_closure_sensitivity(args) == 2
    message = capsys.readouterr().err
    assert "window-closure-sensitivity is retired" in message
    assert "window-closure-region" in message
    assert _retired_root().exists() == before is False


def test_standalone_runner_cli_is_blocked_and_writes_nothing(capsys):
    from scripts.run_window_closure_sensitivity import blocked_cli_entry

    assert blocked_cli_entry() == 2
    message = capsys.readouterr().err
    assert "window-closure-sensitivity is retired" in message
    assert "--output-root" in message
    assert not _retired_root().exists()


def test_standalone_runner_subprocess_invocation_creates_no_namespace():
    """The real command line, not just the helper, must refuse."""
    import subprocess
    import sys as _sys

    from core.paths import PROJECT_ROOT

    result = subprocess.run(
        [_sys.executable, "scripts/run_window_closure_sensitivity.py",
         "--experiment", REFERENCE_AOI, "--dry-run"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode != 0
    assert "is retired" in result.stderr
    assert not _retired_root().exists()


def test_main_cli_subprocess_invocation_creates_no_namespace():
    import subprocess
    import sys as _sys

    from core.paths import PROJECT_ROOT

    result = subprocess.run(
        [_sys.executable, "scripts/main.py", "window-closure-sensitivity",
         "--experiment", REFERENCE_AOI, "--dry-run"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode != 0
    assert "is retired" in result.stderr
    assert not _retired_root().exists()


def test_regional_cli_still_works_for_the_four_actual_aois():
    """Blocking the retired path must not disturb the current one."""
    from scripts.run_window_closure_region import build_parser

    for aoi in ACTUAL_AOIS:
        args = build_parser().parse_args(["--experiment", aoi, "--dry-run"])
        assert args.experiment == aoi
        assert args.dry_run is True
        assert args.execute_actual is False


def test_backend_binding_survives_the_entry_point_block():
    import src.window_closure_sensitivity as wcs
    from scripts.run_window_closure_sensitivity import main as programmatic_main
    from src.multi_region_window_closure.production import ProductionRegionalEngine

    assert callable(wcs.run_analysis)
    assert callable(programmatic_main)
    assert ProductionRegionalEngine(aoi="bejis_2022").runner is wcs.run_analysis
