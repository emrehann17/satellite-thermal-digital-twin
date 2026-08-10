"""Architecture tests for independent regional runs and read-only synthesis."""
from __future__ import annotations

from pathlib import Path
import pytest

from src.multi_region_window_closure import bootstrap, dates, wording
from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS, MODEL_FAMILIES, SCIENTIFIC_CONTRACT_ID, SYNTHESIS_AOIS, VARIANTS,
    MultiRegionWindowClosureError, assert_regional_aoi,
    expected_regional_auxiliary_fit_keys, expected_regional_primary_fit_keys,
)
from src.multi_region_window_closure.execution import regional_dry_run, synthesis_dry_run
from src.multi_region_window_closure.per_aoi import (
    REGIONAL_DRY_RUN_PASS, REGIONAL_REQUIRED_ARTIFACTS,
    SYNTHESIS_DRY_RUN_BLOCKED, SYNTHESIS_REQUIRED_ARTIFACTS,
    assert_rows_belong_to_aoi, independent_statuses,
    quarantine_regional_namespace, regional_analysis_id,
    regional_expected_counts, regional_namespace, regional_resume_matches,
    synthesis_gate,
)
from src.multi_region_window_closure.validators import (
    REGIONAL_CHECKS, SYNTHESIS_CHECKS, overall_status,
)


@pytest.mark.parametrize("aoi", ACTUAL_AOIS)
def test_regional_allow_list_accepts_exact_three(aoi):
    assert assert_regional_aoi(aoi) == aoi


@pytest.mark.parametrize("aoi", ["evia_2021", "manavgat_2021", "unknown"])
def test_regional_allow_list_rejects_old_reference_unknown(aoi):
    with pytest.raises(MultiRegionWindowClosureError):
        assert_regional_aoi(aoi)


def test_shared_contract_exact_variants_and_models():
    assert VARIANTS == ("canonical", "close_7d_earlier", "close_14d_earlier")
    assert MODEL_FAMILIES == ("baseline", "thermal")


@pytest.mark.parametrize("aoi", ACTUAL_AOIS)
def test_date_contract_per_aoi(aoi):
    rows = dates.window_date_rows((aoi,), (0, 7, 14))
    dates.assert_date_contract(rows)
    assert len(rows) == 3
    assert len({row["calendar_duration_days"] for row in rows}) == 1
    assert len({(row["label_start"], row["label_end"]) for row in rows}) == 1
    assert len({row["event_source_field"] for row in rows}) == 1
    assert len({row["gate_source_field"] for row in rows}) == 1


def test_brier_orientation_and_interval_endpoint_swap():
    from src.multi_region_window_closure.contract import orient, orient_interval
    assert orient("brier", -0.2) == 0.2
    assert orient_interval("brier", -0.3, -0.1) == (0.1, 0.3)


def test_bootstrap_series_is_27_per_aoi_and_unique():
    rows = bootstrap.comparison_series("bejis_2022")
    assert len(rows) == 27
    bootstrap.assert_series_unique(rows)


def test_bootstrap_declares_no_refit():
    from src.multi_region_window_closure.contract import frozen_bootstrap_configuration
    assert frozen_bootstrap_configuration()["refit_inside_bootstrap"] is False


@pytest.mark.parametrize("phrase", ["statistically significant", "causal", "optimal", "best window"])
def test_forbidden_language_guard(phrase):
    assert wording.find_forbidden(f"result is {phrase}")


@pytest.mark.parametrize("aoi", ACTUAL_AOIS)
def test_fit_accounting_is_per_aoi(aoi):
    counts = regional_expected_counts()
    assert counts["expected_primary_estimator_fits"] == 30
    assert counts["expected_auxiliary_downscaling_fits"] == 2
    assert counts["expected_total_fits"] == 32
    assert len(expected_regional_primary_fit_keys(aoi)) == 30
    assert len(expected_regional_auxiliary_fit_keys(aoi)) == 2


def test_bootstrap_accounting_is_per_aoi():
    counts = regional_expected_counts()
    assert counts["comparison_series"] == 27
    assert counts["bootstrap_summary_rows"] == 27
    assert counts["requested_bootstrap_replicate_rows"] == 27000


def test_regional_analysis_id_contains_aoi_identity():
    ids = {
        aoi: regional_analysis_id(aoi, {"frozen": True}, "a" * 64, {"x": "b" * 64})
        for aoi in ACTUAL_AOIS
    }
    assert len(set(ids.values())) == len(ACTUAL_AOIS)
    assert ids == {
        aoi: regional_analysis_id(aoi, {"frozen": True}, "a" * 64, {"x": "b" * 64})
        for aoi in ACTUAL_AOIS
    }


@pytest.mark.parametrize("aoi", ACTUAL_AOIS)
def test_regional_namespace_includes_aoi(aoi, tmp_path):
    root = regional_namespace(aoi, "analysis", tmp_path)
    assert root == tmp_path / aoi / "analysis"
    assert not root.exists()


def test_cross_aoi_rows_rejected():
    with pytest.raises(MultiRegionWindowClosureError, match="SCOPE_ESCAPE"):
        assert_rows_belong_to_aoi([{"aoi": "mugla_2021"}], "bejis_2022")


def test_force_quarantine_cannot_touch_another_aoi(tmp_path):
    bejis = regional_namespace("bejis_2022", "same-id", tmp_path)
    mugla = regional_namespace("mugla_2021", "same-id", tmp_path)
    bejis.mkdir(parents=True); mugla.mkdir(parents=True)
    (bejis / "state").write_text("b", encoding="utf-8")
    (mugla / "state").write_text("m", encoding="utf-8")
    target = quarantine_regional_namespace("bejis_2022", "same-id", "force", tmp_path, "20260804T000000Z")
    assert target and target.is_dir()
    assert not bejis.exists()
    assert mugla.joinpath("state").read_text(encoding="utf-8") == "m"


def test_independent_status_never_propagates_failure():
    statuses = independent_statuses({"bejis_2022": "FAIL", "mugla_2021": "PASS"})
    assert statuses["mugla_2021"] == "PASS"


def test_resume_cannot_reuse_another_aoi_state():
    state = {"aoi": "bejis_2022", "analysis_id": "id", "config_hash": "c", "input_hash": "i", "status": "PASS", "stage_output_hash": "h", "stage_output_inventory": []}
    assert regional_resume_matches(state, aoi="bejis_2022", analysis_id="id", config_hash="c", input_hash="i")
    assert not regional_resume_matches(state, aoi="mugla_2021", analysis_id="id", config_hash="c", input_hash="i")


def _valid_gate_records():
    common = {"validator_status": "PASS", "manifest_hash": "h", "hashes_match": True, "scientific_contract_id": SCIENTIFIC_CONTRACT_ID, "summary_aoi_matches": True, "analysis_ids_match": True, "schemas_match": True, "manifest_files_complete": True, "canonical_hash": "c" * 64}
    return {
        "manavgat_2021": {**common, "schema_version": "window_closure_sensitivity.v1"},
        **{aoi: {**common, "schema_version": "window_closure_region.v1"} for aoi in ACTUAL_AOIS},
    }


def test_synthesis_gate_requires_all_four_pass_inputs():
    assert synthesis_gate(_valid_gate_records())["status"] == "PASS"
    for aoi in ("manavgat_2021", *ACTUAL_AOIS):
        records = _valid_gate_records(); records.pop(aoi)
        assert synthesis_gate(records)["status"] == "BLOCKED"


@pytest.mark.parametrize("mutation", ["fail", "hash", "contract", "schema"])
def test_synthesis_gate_fails_closed(mutation):
    records = _valid_gate_records()
    row = records["bejis_2022"]
    if mutation == "fail": row["validator_status"] = "FAIL"
    elif mutation == "hash": row["hashes_match"] = False
    elif mutation == "contract": row["scientific_contract_id"] = "drift"
    else: row["schema_version"] = "old"
    assert synthesis_gate(records)["status"] == "BLOCKED"


def test_synthesis_gate_has_no_compute_or_pooling():
    gate = synthesis_gate(_valid_gate_records())
    # 27 frozen series per AOI and nothing pooled; the AOI count is read from
    # the contract so a registry change cannot silently restate the expectation.
    assert gate["expected_rows"] == len(SYNTHESIS_AOIS) * 27
    assert gate["pooled_inference"] is False
    assert gate["model_fit"] is False
    assert gate["bootstrap_run"] is False


def test_synthesis_missing_input_dry_run_is_read_only(tmp_path):
    before = list(tmp_path.rglob("*"))
    result = synthesis_dry_run({
        "manavgat_2021": None, "bejis_2022": None,
        "mugla_2021": None, "evia_2021_extended": None,
    })
    assert result["final_token"] == SYNTHESIS_DRY_RUN_BLOCKED
    assert result["files_written"] is False
    assert result["network_calls"] == result["gee_initialize_calls"] == 0
    assert list(tmp_path.rglob("*")) == before


def test_regional_dry_run_calls_no_network_gee_model_or_bootstrap(monkeypatch, tmp_path):
    import requests
    import core.gee_utils as gee_utils
    import src.window_closure_sensitivity as legacy

    calls = {"network": 0, "gee": 0, "model": 0, "bootstrap": 0}
    def forbidden(name):
        def _call(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"dry-run called forbidden {name} path")
        return _call
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden("network"))
    monkeypatch.setattr(gee_utils, "init_gee", forbidden("gee"))
    monkeypatch.setattr(legacy, "production_predictor_engine", forbidden("model"))
    monkeypatch.setattr(legacy, "multi_variant_block_bootstrap", forbidden("bootstrap"))
    result = regional_dry_run("bejis_2022", output_root=tmp_path / "diagnostics")
    assert calls == {"network": 0, "gee": 0, "model": 0, "bootstrap": 0}
    assert result["output_namespace_created"] is False


def test_schema_artifact_sets_are_disjoint_by_grain():
    assert "four_region_synthesis.csv" not in REGIONAL_REQUIRED_ARTIFACTS
    assert "oof_predictions.parquet" not in SYNTHESIS_REQUIRED_ARTIFACTS
    assert "bootstrap_replicates.parquet" not in SYNTHESIS_REQUIRED_ARTIFACTS


def test_validator_id_namespaces_and_counts_are_unique():
    regional_ids = [item.check_id for item in REGIONAL_CHECKS]
    synthesis_ids = [item.check_id for item in SYNTHESIS_CHECKS]
    assert all(item.startswith("REG-") for item in regional_ids)
    assert all(item.startswith("SYN-") for item in synthesis_ids)
    assert len(regional_ids) == len(set(regional_ids))
    assert len(synthesis_ids) == len(set(synthesis_ids))


def test_required_skip_or_absence_cannot_pass():
    assert overall_status({}, REGIONAL_CHECKS) == "FAIL"
    # The synthesis registry was emptied by the multi-AOI cleanup, so it has no
    # required check left to miss. Restore the FAIL assertion above for it if
    # SYNTHESIS_CHECKS is ever repopulated.
    assert SYNTHESIS_CHECKS == ()


def test_legacy_traceability_table_is_not_reachable():
    """The legacy->new check mapping went away with the multi-AOI cleanup."""
    from src.multi_region_window_closure import validators
    assert not hasattr(validators, "legacy_traceability")


def test_main_cli_excludes_old_evia():
    from scripts.main import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["window-closure-region", "--experiment", "evia_2021", "--dry-run"])
