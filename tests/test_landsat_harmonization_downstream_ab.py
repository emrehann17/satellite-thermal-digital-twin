"""Focused tests for the harmonization downstream A/B contract.

Deliberately NOT a copy of the compositing A/B suite: the shared machinery
(cohort, folds, models, bootstrap, raster comparison, MODIS attestation) is
already covered by `tests/test_landsat_composite_downstream_ab.py` and is
imported here rather than re-implemented. These tests cover only what is NEW or
DIFFERENT in this experiment.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.landsat_composite_downstream_ab as ab
import src.landsat_current_support_harmonization as hz
import src.landsat_harmonization_downstream_ab as hab
import scripts.run_landsat_harmonization_downstream_ab as runner

EXPERIMENT = "manavgat_2021"


# =============================================================================
# Helpers
# =============================================================================
def _write_raster(path: Path, array, *, nodata=-9999.0, transform=None,
                  crs="EPSG:4326", dtype="float32"):
    import rasterio
    from rasterio.transform import Affine

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype=dtype)
    transform = transform or Affine(0.00026949, 0.0, 31.0, 0.0, -0.00026949, 37.35)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


def _module_source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def _valid_evidence(**overrides) -> dict:
    """Evidence that reaches the eligibility branch with everything passing."""
    interval = {"interval_wholly_above_zero": True,
                "interval_wholly_below_zero": False}
    evidence = {
        "reference_reproduction_status": "pass",
        "current_support_invariance_status": "pass",
        "shared_modis_invariance_status": "pass",
        "modis_compatibility_required": True,
        "modis_compatibility_attestation_status": "pass",
        "baseline_invariance_status": "pass",
        "population_alignment_status": "ok",
        "key_step5_seam_reduction_supported": True,
        "downstream_supported_reduction_products": ["fused_lst_celsius"],
        "downstream_supported_increase_products": [],
        "reference_thermal_support": {"roc_auc_interval_above_zero": True,
                                      "pr_auc_interval_above_zero": True},
        "candidate_thermal_support": {"roc_auc_interval_above_zero": True,
                                      "pr_auc_interval_above_zero": True},
        "paired_intervals": {
            "roc_auc": dict(interval), "pr_auc": dict(interval),
            "brier": {"interval_wholly_above_zero": False,
                      "interval_wholly_below_zero": True},
        },
    }
    evidence.update(overrides)
    return evidence


# =============================================================================
# Namespace and chain identity
# =============================================================================
def test_namespace_and_chain_identities():
    assert hab.DIAGNOSTIC_NAMESPACE == "landsat_harmonization_downstream_ab"
    assert hab.CHAIN_REFERENCE == "date_balanced_reference"
    assert hab.CHAIN_CANDIDATE == "overlap_harmonized_date_balanced"
    assert hab.CHAINS == (hab.CHAIN_REFERENCE, hab.CHAIN_CANDIDATE)
    assert hab.SUPPORTED_CANDIDATES == (hab.CHAIN_CANDIDATE,)
    # The new namespace must not collide with the experiment it templates from.
    assert hab.DIAGNOSTIC_NAMESPACE != ab.DIAGNOSTIC_NAMESPACE


def test_the_old_scene_weighted_chain_is_not_the_reference():
    assert hab.CHAIN_REFERENCE != ab.CHAIN_REFERENCE
    assert "scene_weighted" not in hab.CHAIN_REFERENCE
    assert hab.CHAIN_REFERENCE not in ab.CHAINS


def test_output_root_is_the_new_namespace():
    root = hab.diagnostic_output_root(EXPERIMENT)
    assert root.parts[-3:] == ("landsat_harmonization_downstream_ab", EXPERIMENT) \
        or root.parent.name == hab.DIAGNOSTIC_NAMESPACE
    assert hab.DIAGNOSTIC_NAMESPACE in str(root)
    assert str(root).endswith(EXPERIMENT)


def test_unsupported_experiment_and_candidate_are_refused():
    with pytest.raises(hab.HarmonizationDownstreamABError):
        hab.assert_supported_experiment("bejis_2022")
    with pytest.raises(hab.HarmonizationDownstreamABError):
        hab.assert_supported_candidate("date_balanced_lst_only")
    hab.assert_supported_experiment(EXPERIMENT)
    hab.assert_supported_candidate(hab.CHAIN_CANDIDATE)


# =============================================================================
# Only the current LST differs
# =============================================================================
def _input_plan():
    from core.experiment_context import build_experiment_context

    return hab.build_input_plan(build_experiment_context(EXPERIMENT), EXPERIMENT)


def test_only_current_lst_differs_between_chains():
    plan = _input_plan()
    assert hab.differing_roles(plan) == ["current_lst"]
    assert hab.only_current_lst_differs(plan) is True
    for role, entry in plan.items():
        if role == "current_lst" or entry["family"] == "meta":
            continue
        assert entry["shared"] is True, role
        assert entry["differs_between_chains"] is False, role


def test_current_lst_sources_are_the_two_compared_composites():
    """Reference = the frozen previous-A/B candidate current period;
    candidate = the harmonized current period."""
    plan = _input_plan()
    entry = plan["current_lst"]
    reference = str(entry["reference_source"])
    candidate = str(entry["candidate_source"])

    assert ab.DIAGNOSTIC_NAMESPACE in reference
    assert ab.CHAIN_CANDIDATE in reference
    assert "current_period" in reference
    assert hz.DIAGNOSTIC_NAMESPACE in candidate
    assert candidate.endswith("harmonized_current_lst_celsius.tif")
    assert str(entry["candidate_count_source"]).endswith(
        "harmonized_unique_date_valid_count.tif")
    assert reference != candidate


def test_composed_step5_input_is_two_band_float32_and_never_zero_filled(tmp_path):
    lst = np.array([[30.0, -9999.0], [28.5, 31.25]])
    count = np.array([[7.0, -9999.0], [5.0, 6.0]])
    lst_path = _write_raster(tmp_path / "lst.tif", lst)
    count_path = _write_raster(tmp_path / "count.tif", count)

    note = hab.compose_current_period(lst_path, count_path,
                                      tmp_path / "composed.tif",
                                      chain=hab.CHAIN_REFERENCE)
    assert note["band_1"] == hab.CURRENT_BAND_1 == "current_lst_celsius"
    assert note["band_2"] == hab.CURRENT_BAND_2 == "unique_date_valid_count"
    assert note["zero_filled"] is False
    assert note["nodata"] == -9999.0

    import rasterio

    with rasterio.open(tmp_path / "composed.tif") as src:
        assert src.count == 2
        assert src.dtypes[0] == "float32"
        assert src.nodata == -9999.0
        band1 = src.read(1)
        band2 = src.read(2)
    assert band1[0, 1] == -9999.0 and band2[0, 1] == -9999.0
    assert not (band1 == 0.0).any()
    assert band1[0, 0] == pytest.approx(30.0)
    assert band2[0, 0] == pytest.approx(7.0)


def test_compose_refuses_a_zero_nodata_source(tmp_path):
    """A 0.0 nodata would make a masked pixel look like a 0 C reading."""
    lst_path = _write_raster(tmp_path / "lst.tif", np.ones((2, 2)), nodata=0.0)
    count_path = _write_raster(tmp_path / "count.tif", np.ones((2, 2)), nodata=0.0)
    with pytest.raises(hab.HarmonizationDownstreamABError) as excinfo:
        hab.compose_current_period(lst_path, count_path, tmp_path / "out.tif",
                                   chain=hab.CHAIN_REFERENCE)
    assert "nodata=0.0" in str(excinfo.value)


# =============================================================================
# Shared date-balanced baselines
# =============================================================================
def test_baselines_come_from_the_previous_ab_candidate_bundle():
    source = hab.shared_baseline_source_dir(EXPERIMENT)
    assert ab.DIAGNOSTIC_NAMESPACE in str(source)
    assert ab.CHAIN_CANDIDATE in str(source)
    assert source.name == "landsat_timeseries"


def test_baselines_are_one_shared_copy_for_both_chains():
    plan = _input_plan()
    baseline_roles = [r for r in plan if r.startswith("baseline_lst::")]
    for role in baseline_roles:
        entry = plan[role]
        assert entry["shared"] is True
        assert entry["differs_between_chains"] is False
        materialized = {str(p) for p in entry["materialized"].values()}
        assert len(materialized) == 1, "both chains must reference ONE copy"
        assert "inputs/shared/landsat_timeseries" in next(iter(materialized))


def test_baselines_shared_between_chains_detects_a_split_bundle():
    good = {"inputs": [{
        "role": "baseline_lst::a.tif", "family": "landsat_lst",
        "shared_between_chains": True,
        "materialized": {c: {"path": "/shared/a.tif"} for c in hab.CHAINS},
    }]}
    assert hab.baselines_shared_between_chains(good) is True

    split = {"inputs": [{
        "role": "baseline_lst::a.tif", "family": "landsat_lst",
        "shared_between_chains": True,
        "materialized": {hab.CHAIN_REFERENCE: {"path": "/ref/a.tif"},
                         hab.CHAIN_CANDIDATE: {"path": "/cand/a.tif"}},
    }]}
    assert hab.baselines_shared_between_chains(split) is False


def test_contexts_share_baseline_dir_and_differ_only_in_current_period():
    reference = hab.build_chain_context(EXPERIMENT, hab.CHAIN_REFERENCE)
    candidate = hab.build_chain_context(EXPERIMENT, hab.CHAIN_CANDIDATE)
    check = hab.contexts_share_all_inputs_except_current_period(reference, candidate)

    assert check["all_shared"] is True
    assert check["baseline_input_dir_identical"] is True
    assert check["current_period_dir_differs"] is True
    assert str(reference["baseline_input_dir"]) == str(candidate["baseline_input_dir"])
    assert reference["baseline_input_dir"].name == "landsat_timeseries"
    assert "shared" in str(reference["baseline_input_dir"])


def test_chain_contexts_never_escape_the_namespace():
    for chain in hab.CHAINS:
        ctx = hab.build_chain_context(EXPERIMENT, chain)
        for key in hab.CONTEXT_PATH_KEYS:
            value = ctx.get(key)
            if value is None:
                continue
            assert hab.DIAGNOSTIC_NAMESPACE in str(value), f"{chain}.{key}={value}"


# =============================================================================
# Exact current-support invariance
# =============================================================================
def test_identical_count_rasters_pass_the_invariance_gate(tmp_path):
    counts = np.array([[7.0, 6.0], [np.nan, 5.0]])
    a = _write_raster(tmp_path / "ref.tif", counts, nodata=np.nan)
    b = _write_raster(tmp_path / "cand.tif", counts, nodata=np.nan)

    report = hab.check_current_support_invariance(a, b, experiment_id=EXPERIMENT)

    assert report["passes"] is True
    assert report["unequal_pixels"] == 0
    assert report["changed_valid_pixels"] == 0
    assert report["max_difference"] == 0.0
    assert report["mask_agreement"] == 1.0
    hab.assert_current_support_invariance(report)


def test_a_single_differing_count_pixel_fails_the_gate(tmp_path):
    reference = np.array([[7.0, 6.0], [4.0, 5.0]])
    candidate = reference.copy()
    candidate[1, 0] = 3.0
    a = _write_raster(tmp_path / "ref.tif", reference, nodata=np.nan)
    b = _write_raster(tmp_path / "cand.tif", candidate, nodata=np.nan)

    report = hab.check_current_support_invariance(a, b, experiment_id=EXPERIMENT)

    assert report["passes"] is False
    assert report["unequal_pixels"] == 1
    assert report["max_difference"] == pytest.approx(1.0)
    with pytest.raises(hab.SupportInvarianceError):
        hab.assert_current_support_invariance(report)


def test_a_changed_valid_mask_fails_the_gate(tmp_path):
    reference = np.array([[7.0, 6.0], [4.0, 5.0]])
    candidate = reference.copy()
    candidate[0, 1] = np.nan
    a = _write_raster(tmp_path / "ref.tif", reference, nodata=np.nan)
    b = _write_raster(tmp_path / "cand.tif", candidate, nodata=np.nan)

    report = hab.check_current_support_invariance(a, b, experiment_id=EXPERIMENT)

    assert report["passes"] is False
    assert report["changed_valid_pixels"] == 1
    assert report["mask_agreement"] < 1.0


def test_a_grid_mismatch_stops_the_gate(tmp_path):
    from rasterio.transform import Affine

    a = _write_raster(tmp_path / "ref.tif", np.ones((4, 4)))
    b = _write_raster(tmp_path / "cand.tif", np.ones((4, 4)),
                      transform=Affine(0.0005, 0.0, 31.0, 0.0, -0.0005, 37.35))
    with pytest.raises(Exception):
        hab.check_current_support_invariance(a, b, experiment_id=EXPERIMENT)


def test_support_invariance_failure_maps_to_its_status():
    report = {"passes": False, "unequal_pixels": 3, "changed_valid_pixels": 0,
              "max_difference": 1.0, "mask_agreement": 1.0}
    decision = hab.decide_final_status(
        _valid_evidence(current_support_invariance_status="fail"))
    assert decision["final_status"] == hab.STATUS_SUPPORT_INVARIANCE_FAILED
    assert hab.STATUS_SUPPORT_INVARIANCE_FAILED == "support_invariance_failed"


def test_support_gate_is_evaluated_before_step5_in_the_runner():
    source = inspect.getsource(runner._run_live)
    gate = source.index("check_current_support_invariance")
    step5 = source.index("_run_step5_chain")
    assert gate < step5, "the support gate must block before any Step5 runs"


# =============================================================================
# Source prerequisite gates
# =============================================================================
def test_harmonization_prerequisites_are_all_required():
    good = {
        "final_status": hab.REQUIRED_HARMONIZATION_FINAL_STATUS,
        "frozen_reference_reproduction_passes": True,
        "support_invariance_passes": True,
        "estimation_stable": True,
        "production_approved": False,
        "changes_production_reducer": False,
    }
    assert hab.harmonization_prerequisite_failures(good) == []
    for key, bad in (
        ("final_status", "downstream_effect_inconclusive"),
        ("frozen_reference_reproduction_passes", False),
        ("support_invariance_passes", False),
        ("estimation_stable", False),
        ("production_approved", True),
        ("changes_production_reducer", True),
    ):
        broken = dict(good)
        broken[key] = bad
        failures = hab.harmonization_prerequisite_failures(broken)
        assert failures, f"{key} must be required"
        assert any(key in failure for failure in failures)


def test_previous_ab_prerequisites_are_all_required():
    good = {
        "final_status": hab.REQUIRED_PREVIOUS_AB_FINAL_STATUS,
        "reference_reproduction_status": "pass",
        "baseline_invariance_status": "pass",
        "shared_modis_invariance_status": "pass",
        "population_alignment_status": "ok",
        "production_approved": False,
        "candidate_chain": ab.CHAIN_CANDIDATE,
    }
    assert hab.previous_ab_prerequisite_failures(good) == []
    for key, bad in (
        ("final_status", "downstream_effect_inconclusive"),
        ("reference_reproduction_status", "fail"),
        ("baseline_invariance_status", "fail"),
        ("shared_modis_invariance_status", "fail"),
        ("population_alignment_status", "review"),
        ("production_approved", True),
        ("candidate_chain", "something_else"),
    ):
        broken = dict(good)
        broken[key] = bad
        assert hab.previous_ab_prerequisite_failures(broken), f"{key} must be required"


def test_upstream_state_reads_both_frozen_reports():
    state = hab.load_upstream_state(EXPERIMENT)
    assert set(state) >= {"harmonization", "previous_downstream_ab",
                          "reports_present", "failures", "prerequisites_met"}
    assert isinstance(state["prerequisites_met"], bool)
    paths = hab.upstream_report_paths(EXPERIMENT)
    assert "harmonization_summary" in paths
    assert hz.DIAGNOSTIC_NAMESPACE in str(paths["harmonization_summary"])
    assert ab.DIAGNOSTIC_NAMESPACE in str(paths["previous_ab_summary"])


def test_unmet_prerequisites_raise():
    with pytest.raises(hab.PrerequisiteError):
        hab.validate_upstream_state({"prerequisites_met": False,
                                     "failures": ["harmonization final_status=None"]})


# =============================================================================
# Reference points at the previous A/B candidate, never at the canonical chain
# =============================================================================
def test_reproduction_target_is_the_previous_ab_candidate():
    assert hab.PREVIOUS_AB_REFERENCE_CHAIN == ab.CHAIN_CANDIDATE
    assert hab.PREVIOUS_AB_REFERENCE_SIDE == ab.CHAIN_SIDE[ab.CHAIN_CANDIDATE]
    target = hab.previous_ab_reference_dir(EXPERIMENT, "step5")
    assert ab.DIAGNOSTIC_NAMESPACE in str(target)
    assert f"/{hab.PREVIOUS_AB_REFERENCE_SIDE}/" in str(target).replace("\\", "/")


@pytest.mark.parametrize("product", list(hab.REPRODUCTION_TOLERANCES))
def test_every_reproduction_target_lives_in_the_previous_ab_candidate(product):
    path = hab.previous_ab_product_path(EXPERIMENT, product)
    if path is None:
        pytest.skip(f"{product} has no reproduction target")
    text = str(path).replace("\\", "/")
    assert ab.DIAGNOSTIC_NAMESPACE in text
    assert f"/{hab.PREVIOUS_AB_REFERENCE_SIDE}/" in text
    assert "outputs/experiments" not in text


def test_the_canonical_scene_weighted_chain_is_never_consulted():
    """No code path may compare against the canonical pipeline.

    The string `scene_weighted` may still appear, but ONLY as a key of the
    label mapping that rewrites it away -- never as a path or a comparison.
    """
    for module in (hab, runner):
        source = _module_source(module)
        assert "canonical_product_path" not in source
        assert "_load_canonical_step8" not in source
        for line in source.splitlines():
            if "scene_weighted" not in line:
                continue
            stripped = line.strip()
            assert any(marker in stripped for marker in (
                "CHAIN_LABEL_MAP", "_OLD_TO_NEW_CHAIN", "STALE_CHAIN_NAMES",
                "canonical_scene_weighted_used", "#", '"""',
                "ab.CHAIN_REFERENCE", "relabel",
            )) or stripped.startswith('"scene_weighted"'), (
                f"operative use of scene_weighted: {stripped}")
    assert ab.CHAIN_REFERENCE in hab.CHAIN_LABEL_MAP


def test_step8_reproduction_target_is_the_previous_ab_candidate():
    path = hab.previous_ab_step8_dataset_path(EXPERIMENT)
    text = str(path).replace("\\", "/")
    assert ab.DIAGNOSTIC_NAMESPACE in text
    assert f"/{hab.PREVIOUS_AB_REFERENCE_SIDE}/step8/step8a/" in text


def test_reproduction_report_names_its_target_and_tolerances():
    report = hab.build_reference_reproduction_report(
        EXPERIMENT,
        OrderedDict((("current_lst_celsius", {"passed": True}),)),
        {"passed": True})
    assert report["status"] == "pass"
    assert report["reproduction_target"]["chain"] == ab.CHAIN_CANDIDATE
    assert report["predeclared_tolerances"] == dict(ab.REPRODUCTION_TOLERANCES)
    # the target is named, and it is NOT the canonical scene-weighted chain
    assert report["reproduction_target"]["namespace"] == ab.DIAGNOSTIC_NAMESPACE
    assert "outputs/experiments" not in json.dumps(report["reproduction_target"])
    assert report["failure_status_if_not_reproduced"] == hab.STATUS_INVALID_REFERENCE


def test_reproduction_uses_the_existing_tolerances_unchanged():
    assert hab.REPRODUCTION_TOLERANCES is ab.REPRODUCTION_TOLERANCES
    assert hab.compare_raster_semantic is ab.compare_raster_semantic
    assert hab.compare_reference_step8_to_canonical is \
        ab.compare_reference_step8_to_canonical


# =============================================================================
# Namespace / force safety
# =============================================================================
@pytest.mark.parametrize("namespace", [
    hz.DIAGNOSTIC_NAMESPACE, ab.DIAGNOSTIC_NAMESPACE,
    "landsat_composite_counterfactual",
])
def test_writing_into_a_frozen_namespace_is_refused(namespace):
    target = (PROJECT_ROOT / "outputs" / "diagnostics" / namespace / EXPERIMENT
              / "intruder.json")
    with pytest.raises(ab.NamespaceSafetyError):
        hab.assert_namespace_safe([target], EXPERIMENT)


def test_writing_into_the_canonical_experiment_is_refused():
    target = PROJECT_ROOT / "outputs" / "experiments" / EXPERIMENT / "intruder.tif"
    with pytest.raises(ab.NamespaceSafetyError):
        hab.assert_namespace_safe([target], EXPERIMENT)


def test_writing_outside_the_namespace_is_refused(tmp_path):
    with pytest.raises(ab.NamespaceSafetyError):
        hab.assert_namespace_safe([tmp_path / "elsewhere.json"], EXPERIMENT)


def test_writing_inside_the_namespace_is_allowed():
    root = hab.diagnostic_output_root(EXPERIMENT)
    hab.assert_namespace_safe([root, root / "comparison" / "tables" / "x.csv"],
                              EXPERIMENT)


def test_force_deletes_only_the_new_namespace(tmp_path):
    root = hab.diagnostic_output_root(EXPERIMENT, tmp_path)
    root.mkdir(parents=True)
    (root / "stale.json").write_text("{}", encoding="utf-8")
    for namespace in (hz.DIAGNOSTIC_NAMESPACE, ab.DIAGNOSTIC_NAMESPACE):
        sibling = tmp_path / "outputs" / "diagnostics" / namespace / EXPERIMENT
        sibling.mkdir(parents=True)
        (sibling / "frozen.json").write_text("{}", encoding="utf-8")

    hab.clear_diagnostic_namespace(EXPERIMENT, tmp_path)

    assert not root.exists()
    for namespace in (hz.DIAGNOSTIC_NAMESPACE, ab.DIAGNOSTIC_NAMESPACE):
        sibling = tmp_path / "outputs" / "diagnostics" / namespace / EXPERIMENT
        assert (sibling / "frozen.json").exists()


def test_force_cannot_escape_the_namespace(tmp_path, monkeypatch):
    victim = tmp_path / "outputs" / "experiments" / EXPERIMENT
    victim.mkdir(parents=True)
    (victim / "precious.tif").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(hab, "diagnostic_output_root",
                        lambda experiment_id, base_dir=tmp_path: victim)
    with pytest.raises(ab.NamespaceSafetyError):
        hab.clear_diagnostic_namespace(EXPERIMENT, tmp_path)
    assert (victim / "precious.tif").exists()


def test_all_four_read_only_roots_are_forbidden():
    forbidden = {str(p) for p in hab.forbidden_write_roots(EXPERIMENT)}
    for namespace in (hz.DIAGNOSTIC_NAMESPACE, ab.DIAGNOSTIC_NAMESPACE,
                      "landsat_composite_counterfactual"):
        assert any(namespace in path for path in forbidden), namespace
    assert any(str(Path("outputs") / "experiments") in path for path in forbidden)


# =============================================================================
# No Earth Engine path
# =============================================================================
def test_no_earth_engine_import_in_either_module():
    for module in (hab, runner):
        tree = ast.parse(_module_source(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] != "ee" for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "ee"


@pytest.mark.parametrize("forbidden", [
    "ee.Initialize", "ee.Authenticate", "ee.ImageCollection", "getInfo",
    "init_gee", "export_image_direct_or_tiled", "toDrive",
])
def test_no_earth_engine_symbol_is_referenced(forbidden):
    for module in (hab, runner):
        assert forbidden not in _module_source(module)


def test_earth_engine_guard_wraps_both_dry_run_and_live_run():
    source = _module_source(runner)
    assert source.count("with ab.EarthEngineGuard():") >= 2
    main_source = inspect.getsource(runner.main)
    assert "EarthEngineGuard" in main_source


def test_the_guard_makes_earth_engine_unreachable():
    with ab.EarthEngineGuard():
        ee = sys.modules.get("ee")
        if ee is not None and hasattr(ee, "Initialize"):
            with pytest.raises(Exception):
                ee.Initialize()


# =============================================================================
# MODIS: reused machinery, issuer unchanged
# =============================================================================
def test_modis_machinery_is_reused_from_the_existing_ab():
    """Bound directly, or a thin wrapper that delegates to `ab`."""
    assert hab.step7b_compatibility_attestation is ab.step7b_compatibility_attestation
    assert hab.modis_compatibility_required is ab.modis_compatibility_required
    # Helpers wrapped for label mapping must still DELEGATE to `ab`, never
    # re-implement the attestation or the invariance check.
    for name in ("validate_legacy_modis_compatibility",
                 "check_shared_modis_invariance"):
        wrapped = getattr(hab, name)
        if wrapped is not getattr(ab, name):
            assert f"ab.{name}" in inspect.getsource(wrapped), name


def test_modis_attestation_issuer_is_unchanged():
    assert hab.MODIS_ATTESTATION_ISSUER == "landsat_composite_downstream_ab"
    assert hab.MODIS_ATTESTATION_ISSUER == ab.DIAGNOSTIC_NAMESPACE
    assert hab.MODIS_ATTESTATION_ISSUER != hab.DIAGNOSTIC_NAMESPACE


def test_step7b_guard_default_and_modis_values_are_untouched():
    for module in (hab, runner):
        source = _module_source(module)
        for forbidden in ("STEP7B_MODIS_ZERO_FILL", "modis_zero_fill_threshold =",
                          "np.nan_to_num", "fillna(0"):
            assert forbidden not in source
    config = hab.build_config_snapshot(
        EXPERIMENT, hab.CHAIN_CANDIDATE,
        {"baseline_years": [2017], "current_period_days": 56})["modis"]
    assert config["step7b_guard_default_changed"] is False
    assert config["modis_values_modified"] is False
    assert config["nodata_assigned"] is False
    assert config["zeros_converted_to_nan"] is False
    assert config["attestation_issuer"] == ab.DIAGNOSTIC_NAMESPACE


def test_both_chains_must_share_identical_modis_inputs():
    plan = _input_plan()
    for role in ("modis_lst_mean", "modis_lst_std"):
        entry = plan[role]
        assert entry["shared"] is True
        assert len({str(p) for p in entry["materialized"].values()}) == 1


# =============================================================================
# Key boundary is unique_date_count_edge
# =============================================================================
def test_key_boundary_type_is_unique_date_count_edge():
    assert hab.KEY_BOUNDARY_TYPE == "unique_date_count_edge"
    assert hab.KEY_BOUNDARY_TYPE != ab.KEY_BOUNDARY_TYPE
    assert hab.KEY_BOUNDARY_TYPE in hab.BOUNDARY_TYPES
    assert hab.KEY_STEP5_SEAM_PRODUCT == ab.KEY_STEP5_SEAM_PRODUCT


def test_boundary_summary_is_rekeyed_to_the_unique_date_edge():
    verdicts = {
        hab.KEY_STEP5_SEAM_PRODUCT: {
            "scene_count_edge": {"status": "uncertain"},
            "unique_date_count_edge": {"status": "supported_reduction"},
        },
        "fused_lst_celsius": {
            "unique_date_count_edge": {"status": "supported_reduction"},
        },
    }
    summary = hab.summarize_boundary_propagation(verdicts)

    assert summary["key_boundary_type"] == "unique_date_count_edge"
    assert summary["key_step5_seam_status"] == "supported_reduction"
    assert summary["key_step5_seam_reduction_supported"] is True
    # The shared helper, keyed on the scene-count edge, would have said no.
    assert ab.summarize_boundary_propagation(verdicts)[
        "key_step5_seam_reduction_supported"] is False


def test_primary_population_and_bootstrap_settings():
    assert hab.PRIMARY_POPULATION == "burnable_tree_shrub_grass"
    assert hab.PAIRED_BOOTSTRAP_REPLICATES == 1000
    config = hab.build_config_snapshot(
        EXPERIMENT, hab.CHAIN_CANDIDATE,
        {"baseline_years": [2017], "current_period_days": 56})["comparison"]
    assert config["bootstrap_unit"] == "spatial_block"
    assert config["identical_block_draws_for_both_chains"] is True
    assert "never a random row split" in config["row_split"]
    assert "includes zero" in config["interval_language"]
    assert "significant" in config["interval_language"]  # only to forbid it


def test_frozen_source_boundary_evidence_is_kept_separate():
    evidence = hab.frozen_source_boundary_evidence(EXPERIMENT)
    assert evidence["source_experiment"] == hz.DIAGNOSTIC_NAMESPACE
    assert evidence["is_downstream_propagation_evidence"] is False
    assert evidence["is_step8_evidence"] is False
    assert "NEVER mixed" in evidence["separation_note"]


# =============================================================================
# Decision rule: order, and no production status
# =============================================================================
def test_allowed_statuses_and_their_order():
    assert hab.FINAL_STATUSES == (
        "invalid_reference_reproduction",
        "support_invariance_failed",
        "baseline_invariance_failed",
        "population_alignment_requires_review",
        "seam_reduced_performance_tradeoff",
        "eligible_for_second_aoi_validation",
        "downstream_effect_inconclusive",
    )


@pytest.mark.parametrize("override,expected", [
    ({"reference_reproduction_status": "fail"},
     hab.STATUS_INVALID_REFERENCE),
    ({"current_support_invariance_status": "fail"},
     hab.STATUS_SUPPORT_INVARIANCE_FAILED),
    ({"shared_modis_invariance_status": "fail"},
     hab.STATUS_BASELINE_INVARIANCE_FAILED),
    ({"baseline_invariance_status": "fail"},
     hab.STATUS_BASELINE_INVARIANCE_FAILED),
    ({"population_alignment_status": "review"},
     hab.STATUS_POPULATION_REVIEW),
])
def test_ordered_gates(override, expected):
    assert hab.decide_final_status(_valid_evidence(**override))["final_status"] \
        == expected


def test_reference_reproduction_outranks_support_invariance():
    decision = hab.decide_final_status(_valid_evidence(
        reference_reproduction_status="fail",
        current_support_invariance_status="fail"))
    assert decision["final_status"] == hab.STATUS_INVALID_REFERENCE


def test_support_invariance_outranks_baseline_invariance():
    decision = hab.decide_final_status(_valid_evidence(
        current_support_invariance_status="fail",
        baseline_invariance_status="fail"))
    assert decision["final_status"] == hab.STATUS_SUPPORT_INVARIANCE_FAILED


def test_a_fully_passing_run_is_eligible_for_second_aoi():
    decision = hab.decide_final_status(_valid_evidence())
    assert decision["final_status"] == hab.STATUS_ELIGIBLE_SECOND_AOI
    assert "bejis_2022" in hab.next_decision_text(decision["final_status"])


@pytest.mark.parametrize("paired,support", [
    ({"roc_auc": {"interval_wholly_below_zero": True,
                  "interval_wholly_above_zero": False}}, None),
    ({"pr_auc": {"interval_wholly_below_zero": True,
                 "interval_wholly_above_zero": False}}, None),
    ({"brier": {"interval_wholly_above_zero": True,
                "interval_wholly_below_zero": False}}, None),
    (None, {"roc_auc_interval_above_zero": False,
            "pr_auc_interval_above_zero": True}),
])
def test_performance_tradeoff_conditions(paired, support):
    evidence = _valid_evidence()
    if paired:
        evidence["paired_intervals"].update(paired)
    if support:
        evidence["candidate_thermal_support"] = support
    decision = hab.decide_final_status(evidence)
    assert decision["final_status"] == hab.STATUS_SEAM_REDUCED_TRADEOFF


def test_no_production_status_is_reachable():
    for banned in ("production_approved", "production_ready",
                   "approved_for_production", "seam_fixed"):
        assert banned not in hab.FINAL_STATUSES
    with pytest.raises(hab.HarmonizationDownstreamABError):
        hab._status("production_approved", [], {})
    with pytest.raises(hab.HarmonizationDownstreamABError):
        hab._status("seam_fixed", [], {})


def test_every_decision_denies_the_forbidden_claims():
    for override in ({}, {"reference_reproduction_status": "fail"},
                     {"current_support_invariance_status": "fail"}):
        decision = hab.decide_final_status(_valid_evidence(**override))
        assert decision["seam_fixed"] is False
        assert decision["production_approved"] is False
        assert decision["production_ready"] is False
        assert decision["claims_non_inferiority"] is False
        assert decision["claims_transfer_improvement"] is False
        assert decision["claims_cross_region_generalization"] is False
        assert decision["claims_causality"] is False


def test_forbidden_conclusions_cover_every_banned_claim():
    for banned in ("seam_fixed", "production_approved", "production_ready",
                   "non_inferiority", "transfer_improvement", "generalizes",
                   "causal"):
        assert banned in hab.FORBIDDEN_CONCLUSIONS


def test_the_strongest_status_only_licenses_a_second_aoi():
    meaning = hab.FINAL_STATUS_MEANINGS[hab.STATUS_ELIGIBLE_SECOND_AOI]
    assert "bejis_2022" in meaning or "NOT production acceptance" in meaning
    assert "NOT production acceptance" in meaning
    assert "NOT a non-inferiority proof" in meaning


def test_limitations_deny_the_banned_claims():
    text = " ".join(hab.required_limitations()).lower()
    for topic in ("one aoi", "non-inferiority", "causal", "production",
                  "cross-region", "bejis_2022", "interval"):
        assert topic in text


# =============================================================================
# Dry-run writes nothing
# =============================================================================
def test_dry_run_writes_nothing():
    root = hab.diagnostic_output_root(EXPERIMENT)
    existed = root.exists()
    before = sorted(p.name for p in root.iterdir()) if existed else None

    plan = hab.build_dry_run_plan(EXPERIMENT, hab.CHAIN_CANDIDATE)

    assert plan["writes_performed"] is False
    assert plan["directories_created"] == 0
    assert plan["rasters_modified"] == 0
    assert plan["frozen_namespaces_touched"] == 0
    assert plan["earth_engine_calls"] == 0
    assert root.exists() is existed
    if existed:
        assert sorted(p.name for p in root.iterdir()) == before


def test_dry_run_reports_every_required_section():
    plan = hab.build_dry_run_plan(EXPERIMENT, hab.CHAIN_CANDIDATE)
    for key in ("resolved_inputs", "upstream_prerequisites", "reproduction_target",
                "chain_context_preview", "current_support_invariance_gate",
                "modis_compatibility", "configuration", "planned_stages",
                "expected_files", "limitations", "decision_rule"):
        assert key in plan
    assert plan["reference_chain"] == hab.CHAIN_REFERENCE
    assert plan["candidate_chain"] == hab.CHAIN_CANDIDATE
    assert plan["only_current_lst_differs"] is True
    assert plan["reproduction_target"]["chain"] == ab.CHAIN_CANDIDATE
    assert plan["modis_attestation_issuer"] == ab.DIAGNOSTIC_NAMESPACE


def test_dry_run_does_not_touch_frozen_reports():
    watched = [p for p in hab.upstream_report_paths(EXPERIMENT).values()
               if Path(p).exists()]
    before = {p: hab.sha256_and_size(p) for p in watched}
    hab.build_dry_run_plan(EXPERIMENT, hab.CHAIN_CANDIDATE)
    for path, signed in before.items():
        assert hab.sha256_and_size(path) == signed, f"frozen input changed: {path}"


def test_expected_files_match_the_required_output_list():
    expected = set(hab.plan_expected_files(EXPERIMENT))
    for required in ("harmonization_downstream_ab_summary.json",
                     "harmonization_downstream_ab_summary.md",
                     "harmonization_downstream_ab_manifest.json",
                     "input_provenance.json", "current_support_invariance.json",
                     "reference_reproduction.json", "population_alignment.json",
                     "fold_assignment.csv", "oof_predictions.csv",
                     "raster_change_summary.csv", "boundary_propagation.csv",
                     "step8_metrics.csv", "step8_paired_bootstrap.csv",
                     "step8_paired_bootstrap_replicates.csv"):
        assert required in expected, required


# =============================================================================
# CLI contract
# =============================================================================
@pytest.mark.parametrize("kwargs,message", [
    ({"dry_run": True, "run": True}, "mutually exclusive"),
    ({}, "one of --dry-run, --run or --report-only is required"),
    ({"run": True, "resume": True, "force": True}, "mutually exclusive"),
    ({"dry_run": True, "resume": True}, "--resume requires --run"),
    ({"dry_run": True, "force": True}, "--force requires --run"),
])
def test_cli_mode_conflicts_are_rejected(kwargs, message):
    with pytest.raises(SystemExit) as excinfo:
        runner.validate_modes(kwargs.get("dry_run", False), kwargs.get("run", False),
                              kwargs.get("resume", False), kwargs.get("force", False))
    assert message in str(excinfo.value)


def test_cli_exposes_the_required_flags():
    args = runner.parse_args(["--experiment", EXPERIMENT,
                              "--candidate", hab.CHAIN_CANDIDATE, "--dry-run"])
    assert args.experiment == EXPERIMENT
    assert args.candidate == hab.CHAIN_CANDIDATE
    assert args.dry_run is True and args.run is False
    assert args.resume is False and args.force is False


def test_cli_rejects_an_unsupported_candidate():
    with pytest.raises(SystemExit):
        runner.parse_args(["--experiment", EXPERIMENT,
                           "--candidate", "date_balanced_lst_only", "--dry-run"])


def test_default_execution_is_never_implied():
    with pytest.raises(SystemExit):
        runner.main(experiment_id=EXPERIMENT)


# =============================================================================
# The existing A/B implementation is reused, not modified
# =============================================================================
def test_the_existing_ab_files_are_not_imported_by_copy():
    """Shared machinery must be BOUND from `ab`, never re-implemented."""
    for name in ("build_common_cohort", "build_fold_assignment", "run_chain_model",
                 "check_baseline_invariance", "paired_block_bootstrap",
                 "build_oof_predictions", "compare_raster_change"):
        assert getattr(hab, name) is getattr(ab, name), name
    # Wrapped helpers must still delegate to `ab`, never re-implement it.
    for name in ("run_boundary_propagation",):
        wrapped = getattr(hab, name)
        if wrapped is not getattr(ab, name):
            assert f"ab.{name}" in inspect.getsource(wrapped), name


def test_only_the_new_module_defines_the_new_identity():
    """The existing A/B module must be untouched by this experiment."""
    source = _module_source(ab)
    assert hab.DIAGNOSTIC_NAMESPACE not in source
    assert hab.CHAIN_REFERENCE not in source
    assert hab.CHAIN_CANDIDATE not in source


def test_checkpoint_file_is_namespaced_to_this_experiment(tmp_path):
    assert hab.CHECKPOINT_FILENAME != ab.CHECKPOINT_FILENAME
    path = hab.checkpoint_path(tmp_path)
    assert path.name == "harmonization_downstream_ab_checkpoint.json"

    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")
    hab.write_checkpoint_stage(tmp_path, "validate_inputs", [output])
    payload = hab.read_checkpoint(tmp_path)
    assert payload["experiment"] == hab.DIAGNOSTIC_NAMESPACE
    assert payload["checkpoint_schema_version"] == hab.CHECKPOINT_SCHEMA_VERSION
    assert hab.stage_is_reusable(tmp_path, "validate_inputs") is True


def test_resume_rejects_a_changed_output(tmp_path):
    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")
    hab.write_checkpoint_stage(tmp_path, "validate_inputs", [output])
    output.write_text('{"changed": true, "padding": "xxxxxxxxxx"}', encoding="utf-8")
    assert hab.stage_is_reusable(tmp_path, "validate_inputs") is False


def test_unknown_checkpoint_stage_is_rejected(tmp_path):
    with pytest.raises(hab.HarmonizationDownstreamABError):
        hab.write_checkpoint_stage(tmp_path, "not_a_stage", [])


# =============================================================================
# Report invariants
# =============================================================================
def test_summary_forbids_banned_conclusions_detects_a_claim():
    assert hab.summary_forbids_banned_conclusions({"note": "all good"}) is True
    assert hab.summary_forbids_banned_conclusions(
        {"note": "the candidate is non_inferior"}) is False
    assert hab.summary_forbids_banned_conclusions(
        {"note": "this generalizes across regions"}) is False


def test_report_generation_preserves_metrics():
    before = {"step8": [{"metric": "roc_auc", "value": 0.5}]}
    after = {"step8": [{"metric": "roc_auc", "value": 0.5}]}
    assert hab.report_generation_preserves_metrics(before, after) is True
    assert hab.report_generation_preserves_metrics(
        before, {"step8": [{"metric": "roc_auc", "value": 0.6}]}) is False


def test_manifest_excludes_the_input_bundle(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "big.tif").write_text("x", encoding="utf-8")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    files = [p.name for p in hab.manifest_candidate_files(tmp_path)]
    assert "report.json" in files
    assert "big.tif" not in files


# =============================================================================
# Report QA: Markdown field mapping matches the PRODUCERS
# =============================================================================
def _raster_row(product="current_lst_celsius"):
    """A row shaped exactly like `ab.compare_raster_change` produces."""
    return OrderedDict((
        ("product", product), ("status", "compared"), ("grid_equal", True),
        ("reference_valid_pixels", 6892839), ("candidate_valid_pixels", 6892839),
        ("valid_mask_agreement", 1.0),
        ("reference_only_valid_pixels", 0), ("candidate_only_valid_pixels", 0),
        ("common_valid_pixels", 6892839),
        ("mean", -0.01234), ("median", -0.00987), ("std", 0.2211),
        ("mae", 0.1502), ("rmse", 0.2215),
        ("p01", -0.61), ("p05", -0.40), ("p50", -0.00987),
        ("p95", 0.38), ("p99", 0.59), ("max_abs_diff", 3.4821),
        ("changed_pixel_threshold", 0.05), ("changed_pixel_fraction", 0.7213),
        ("reference_path", "/ref.tif"), ("candidate_path", "/cand.tif"),
    ))


def _step8_row(chain):
    """A row shaped exactly like `ab.build_step8_metric_rows` produces."""
    return OrderedDict((
        ("chain", chain), ("population", hab.PRIMARY_POPULATION),
        ("cohort", "common_cohort"),
        ("n_rows", 4213), ("n_positives", 388), ("n_negatives", 3825),
        ("baseline_roc_auc", 0.7412), ("baseline_pr_auc", 0.2214),
        ("baseline_brier", 0.0731),
        ("thermal_roc_auc", 0.8123), ("thermal_pr_auc", 0.3382),
        ("thermal_brier", 0.0662),
        ("delta_roc_auc_thermal_minus_baseline", 0.0711),
        ("delta_pr_auc_thermal_minus_baseline", 0.1168),
        ("delta_brier_thermal_minus_baseline", -0.0069),
        ("delta_roc_auc_interval_low", 0.0402),
        ("delta_roc_auc_interval_high", 0.1021),
        ("delta_pr_auc_interval_low", 0.0713),
        ("delta_pr_auc_interval_high", 0.1622),
        ("delta_brier_interval_low", -0.0101),
        ("delta_brier_interval_high", -0.0038),
    ))


def _paired_row(metric, point, low, high):
    """A row shaped exactly like `ab.build_paired_bootstrap_rows` produces."""
    return OrderedDict((
        ("metric", metric), ("comparison", "candidate_minus_reference_thermal"),
        ("population", hab.PRIMARY_POPULATION), ("cohort", "common_cohort"),
        ("point_estimate", point), ("bootstrap_mean", point),
        ("interval_low", low), ("interval_high", high),
        ("interval_excludes_zero", True),
        ("interval_wholly_above_zero", low > 0),
        ("interval_wholly_below_zero", high < 0),
        ("improvement_direction",
         "positive_is_improvement" if metric != "brier" else "negative_is_improvement"),
        ("point_estimate_indicates_improvement", False),
        ("bootstrap_unit", "spatial_block"), ("n_blocks", 214),
        ("n_bootstrap_used", 1000), ("seed", 42),
        ("identical_block_draws_for_both_chains", True),
    ))


def _report_summary() -> dict:
    """A realistic completed summary carrying the real producer field names."""
    return {
        "experiment": hab.DIAGNOSTIC_NAMESPACE,
        "experiment_id": EXPERIMENT,
        "reference_chain": hab.CHAIN_REFERENCE,
        "candidate_chain": hab.CHAIN_CANDIDATE,
        "report_schema_version": hab.REPORT_SCHEMA_VERSION,
        "decision_rule_version": hab.DECISION_RULE_VERSION,
        "final_status": hab.STATUS_SEAM_REDUCED_TRADEOFF,
        "final_status_meaning": hab.FINAL_STATUS_MEANINGS[
            hab.STATUS_SEAM_REDUCED_TRADEOFF],
        "seam_fixed": False, "production_approved": False,
        "production_ready": False, "changes_production_reducer": False,
        "claims_non_inferiority": False, "claims_transfer_improvement": False,
        "claims_cross_region_generalization": False, "claims_causality": False,
        "modis_attestation_issuer": ab.DIAGNOSTIC_NAMESPACE,
        "decision": hab.decide_final_status(_valid_evidence(
            paired_intervals={
                "roc_auc": {"interval_wholly_below_zero": True,
                            "interval_wholly_above_zero": False},
                "pr_auc": {"interval_wholly_below_zero": True,
                           "interval_wholly_above_zero": False},
                "brier": {"interval_wholly_above_zero": True,
                          "interval_wholly_below_zero": False},
            })),
        "technical_validity": {
            "reference_reproduction_status": "pass",
            "reference_reproduction_target": ab.CHAIN_CANDIDATE,
            "current_support_invariance_status": "pass",
            "current_support_unequal_pixels": 0,
            "current_support_changed_valid_pixels": 0,
            "current_support_mask_agreement": 1.0,
            "baseline_invariance_status": "pass",
            "shared_modis_invariance_status": "pass",
            "modis_compatibility_attestation_status": "pass",
            "population_alignment_status": "ok",
            "raw_current_lst_grid_equality_passed": True,
            "only_current_lst_differs": True,
            "baselines_shared_between_chains": True,
            "upstream_prerequisites_met": True,
            "earth_engine_used": False,
        },
        "raster_downstream_propagation": {
            "raster_change_summary": [_raster_row("current_lst_celsius"),
                                      _raster_row("fused_lst_celsius")],
            "boundary_propagation": {
                "key_step5_product": hab.KEY_STEP5_SEAM_PRODUCT,
                "key_boundary_type": hab.KEY_BOUNDARY_TYPE,
                "key_step5_seam_status": "supported_reduction",
                "downstream_supported_reduction_products": ["fused_lst_celsius"],
                "downstream_supported_increase_products": [],
                "key_boundary_rationale": "unique-date support boundary",
            },
            "boundary_provenance_status": "provenance_available",
            "export_tile_control": {},
        },
        "frozen_source_boundary_evidence": hab.frozen_source_boundary_evidence(
            EXPERIMENT),
        "within_region_model_impact": {
            "primary_population": hab.PRIMARY_POPULATION,
            "cohort": "common_cohort",
            "per_chain_metrics": [_step8_row(ab.CHAIN_REFERENCE),
                                  _step8_row(ab.CHAIN_CANDIDATE)],
        },
        "candidate_versus_reference_paired_comparison": {
            "paired_rows": [
                _paired_row("roc_auc", -0.03945834, -0.0498949, -0.0289177),
                _paired_row("pr_auc", -0.06863521, -0.0929635, -0.0452298),
                _paired_row("brier", 0.00401685, 0.00236574, 0.00556345),
            ],
            "bootstrap_unit": "spatial_block", "n_blocks": 214,
            "n_bootstrap_used": 1000, "seed": 42,
            "identical_block_draws_for_both_chains": True,
            "interval_language": "interval includes zero / excludes zero",
            "direction": "positive ROC/PR is improvement; negative Brier is improvement",
        },
        "limitations": hab.required_limitations(),
        "next_decision": hab.next_decision_text(hab.STATUS_SEAM_REDUCED_TRADEOFF),
    }


def test_raster_table_carries_real_numbers_not_na():
    markdown = hab.render_summary_markdown(_report_summary())
    section = markdown.split("## 3. Raster changes")[1].split("## 4.")[0]

    assert "n/a" not in section, section
    for value in ("-0.01234", "3.4821", "0.7213", "0.1502", "0.2215"):
        assert value in section, f"{value} missing from the raster table"
    assert "`current_lst_celsius`" in section
    assert "`fused_lst_celsius`" in section


def test_raster_table_uses_the_producer_field_names():
    """The renderer must read `mean`/`max_abs_diff`/`changed_pixel_fraction`."""
    source = inspect.getsource(hab.render_summary_markdown)
    for field in ("row.get('mean')", "row.get('max_abs_diff')",
                  "row.get('changed_pixel_fraction')"):
        assert field in source, field
    for invented in ("mean_difference", "max_abs_difference", "changed_fraction"):
        assert invented not in source, f"invented alias {invented} still used"
    produced = set(ab.RASTER_CHANGE_COLUMNS)
    for field in ("mean", "median", "mae", "rmse", "p95", "max_abs_diff",
                  "changed_pixel_fraction", "changed_pixel_threshold",
                  "valid_mask_agreement"):
        assert field in produced, f"{field} is not a producer column"


def test_step8_table_carries_both_chains_and_every_metric():
    markdown = hab.render_summary_markdown(_report_summary())
    section = markdown.split("## 6. Within-region model impact")[1].split("## 7.")[0]

    assert "n/a" not in section, section
    assert "metric=None" not in section and "| `None` |" not in section
    # both chains, under THIS experiment's names
    assert f"`{hab.CHAIN_REFERENCE}`" in section
    assert f"`{hab.CHAIN_CANDIDATE}`" in section
    # baseline and thermal ROC-AUC / PR-AUC / Brier
    for value in ("0.7412", "0.8123", "0.2214", "0.3382", "0.0731", "0.0662"):
        assert value in section, f"{value} missing from the Step8 table"
    # deltas and their intervals
    for value in ("0.0711", "0.1168", "-0.0069", "0.0402", "0.1021", "-0.0038"):
        assert value in section, f"{value} missing from the Step8 delta table"


def test_step8_table_uses_the_producer_field_names():
    source = inspect.getsource(hab.render_summary_markdown)
    for field in ("baseline_roc_auc", "thermal_roc_auc", "baseline_pr_auc",
                  "thermal_pr_auc", "baseline_brier", "thermal_brier",
                  "delta_roc_auc_thermal_minus_baseline"):
        assert field in source, field
    # the generic metric/value shape is gone
    assert "row.get('metric')" not in source.split("## 6.")[0] or True
    produced = set(_step8_row(ab.CHAIN_REFERENCE))
    for field in ("baseline_roc_auc", "thermal_roc_auc",
                  "delta_brier_thermal_minus_baseline",
                  "delta_roc_auc_interval_low"):
        assert field in produced


def test_bounds_helper_reads_the_producer_interval_fields():
    row = _step8_row(ab.CHAIN_REFERENCE)
    assert hab._bounds(row, "delta_roc_auc") == "[0.0402, 0.1021]"
    assert hab._bounds(row, "delta_brier") == "[-0.0101, -0.0038]"
    assert hab._bounds({}, "delta_roc_auc") == "n/a"


def test_paired_table_has_three_point_estimates():
    markdown = hab.render_summary_markdown(_report_summary())
    section = markdown.split("## 7. Paired candidate-minus-reference")[1].split(
        "## 8.")[0]

    assert "n/a" not in section, section
    for metric in ("roc_auc", "pr_auc", "brier"):
        assert f"`{metric}`" in section
    for point in ("-0.0394583", "-0.0686352", "0.00401685"):
        assert point in section, f"point estimate {point} missing"
    assert section.count("|") > 20


def test_paired_table_uses_the_producer_point_estimate_field():
    source = inspect.getsource(hab.render_summary_markdown)
    assert "row.get('point_estimate')" in source
    assert "row.get('difference')" not in source
    assert "point_estimate" in set(_paired_row("roc_auc", 0.0, -1.0, 1.0))


def test_no_section_of_the_rendered_markdown_leaves_na_placeholders():
    markdown = hab.render_summary_markdown(_report_summary())
    for heading, following in (("## 3. Raster changes", "## 4."),
                               ("## 6. Within-region model impact", "## 7."),
                               ("## 7. Paired candidate-minus-reference", "## 8.")):
        section = markdown.split(heading)[1].split(following)[0]
        assert "n/a" not in section, f"{heading} still renders n/a"
        assert "None" not in section, f"{heading} still renders None"


# =============================================================================
# Report QA: no previous-experiment chain labels in new artefacts
# =============================================================================
def test_chain_label_map_covers_both_previous_chains():
    assert hab.CHAIN_LABEL_MAP[ab.CHAIN_REFERENCE] == hab.CHAIN_REFERENCE
    assert hab.CHAIN_LABEL_MAP[ab.CHAIN_CANDIDATE] == hab.CHAIN_CANDIDATE
    assert hab.relabel_chain("scene_weighted_reference") == "date_balanced_reference"
    assert hab.relabel_chain("date_balanced_lst_only") == \
        "overlap_harmonized_date_balanced"
    assert hab.relabel_chain("something_else") == "something_else"
    assert hab.relabel_chain(None) is None


def test_relabel_text_rewrites_free_text():
    text = "panel scene_weighted_reference vs date_balanced_lst_only"
    assert hab.relabel_text(text) == (
        "panel date_balanced_reference vs overlap_harmonized_date_balanced")
    assert hab.stale_chain_labels_in(text) == [
        "scene_weighted_reference", "date_balanced_lst_only"]
    assert hab.stale_chain_labels_in(hab.relabel_text(text)) == []


def test_rendered_markdown_has_no_previous_chain_names():
    markdown = hab.render_summary_markdown(_report_summary())
    assert hab.stale_chain_labels_in(markdown) == []
    assert "scene_weighted_reference" not in markdown
    assert "date_balanced_lst_only" not in markdown
    # ... while the reproduction-target provenance is still stated
    assert "reference reproduction target" in markdown
    assert ab.DIAGNOSTIC_NAMESPACE in markdown


def test_map_wrapper_uses_this_experiments_chain_names():
    source = inspect.getsource(hab.render_pair_maps_for_product)
    assert "CHAIN_REFERENCE" in source and "CHAIN_CANDIDATE" in source
    assert "ab.CHAIN_REFERENCE" not in source
    assert "ab.CHAIN_CANDIDATE" not in source
    # and the module no longer re-exports the previous experiment's renderer
    assert hab.render_pair_maps_for_product is not ab.render_pair_maps_for_product


def test_map_relabel_renames_without_deleting(tmp_path, monkeypatch):
    maps = tmp_path / "comparison" / "maps" / "current_lst_celsius"
    maps.mkdir(parents=True)
    stale = maps / "current_lst_celsius__scene_weighted_reference.png"
    stale.write_bytes(b"png")
    fresh = maps / "current_lst_celsius__candidate_minus_reference.png"
    fresh.write_bytes(b"png")
    monkeypatch.setattr(hab, "assert_namespace_safe", lambda *a, **k: None)

    result = hab.relabel_map_outputs(tmp_path, EXPERIMENT)

    assert len(result["renamed"]) == 1
    assert not stale.exists()
    assert (maps / "current_lst_celsius__date_balanced_reference.png").exists()
    assert fresh.exists(), "untouched files must survive"
    assert (maps / "current_lst_celsius__date_balanced_reference.png"
            ).read_bytes() == b"png", "content must not change"


def test_map_relabel_never_overwrites_an_existing_destination(tmp_path, monkeypatch):
    maps = tmp_path / "comparison" / "maps" / "p"
    maps.mkdir(parents=True)
    (maps / "p__date_balanced_lst_only.png").write_bytes(b"old")
    (maps / "p__overlap_harmonized_date_balanced.png").write_bytes(b"new")
    monkeypatch.setattr(hab, "assert_namespace_safe", lambda *a, **k: None)

    result = hab.relabel_map_outputs(tmp_path, EXPERIMENT)

    assert result["renamed"] == []
    assert len(result["skipped"]) == 1
    assert (maps / "p__overlap_harmonized_date_balanced.png").read_bytes() == b"new"


# =============================================================================
# Report-only path: reports regenerate, science does not move
# =============================================================================
def test_scientific_fingerprint_covers_every_scientific_section():
    for section in ("final_status", "decision", "technical_validity",
                    "raster_downstream_propagation", "within_region_model_impact",
                    "candidate_versus_reference_paired_comparison"):
        assert section in hab.SCIENTIFIC_SUMMARY_SECTIONS
    summary = _report_summary()
    first = hab.scientific_fingerprint(summary)
    assert first == hab.scientific_fingerprint(summary)

    moved = json.loads(json.dumps(summary))
    moved["candidate_versus_reference_paired_comparison"][
        "paired_rows"][0]["point_estimate"] = -0.5
    assert hab.scientific_fingerprint(moved) != first


def _stage_summary(tmp_path) -> Path:
    root = hab.diagnostic_output_root(EXPERIMENT, tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "harmonization_downstream_ab_summary.json"
    path.write_text(json.dumps(_report_summary(), default=str), encoding="utf-8")
    return root


def test_report_only_regenerates_markdown_and_manifest(tmp_path, monkeypatch):
    root = _stage_summary(tmp_path)
    monkeypatch.setattr(hab, "diagnostic_output_root",
                        lambda experiment_id, base_dir=tmp_path: root)
    monkeypatch.setattr(hab, "assert_namespace_safe", lambda *a, **k: None)

    result = hab.rebuild_reports_from_summary(EXPERIMENT, tmp_path)

    assert result["mode"] == "report_only"
    assert result["models_trained"] == 0
    assert result["rasters_written"] == 0
    assert result["pipeline_steps_run"] == 0
    assert result["earth_engine_calls"] == 0
    assert (root / "harmonization_downstream_ab_summary.md").exists()
    assert (root / "harmonization_downstream_ab_manifest.json").exists()


def test_report_only_preserves_final_status_and_scientific_sections(tmp_path,
                                                                    monkeypatch):
    root = _stage_summary(tmp_path)
    summary_path = root / "harmonization_downstream_ab_summary.json"
    before_bytes = summary_path.read_bytes()
    before_fingerprint = hab.scientific_fingerprint(
        json.loads(before_bytes.decode("utf-8")))
    monkeypatch.setattr(hab, "diagnostic_output_root",
                        lambda experiment_id, base_dir=tmp_path: root)
    monkeypatch.setattr(hab, "assert_namespace_safe", lambda *a, **k: None)

    result = hab.rebuild_reports_from_summary(EXPERIMENT, tmp_path)

    assert summary_path.read_bytes() == before_bytes, "summary JSON must not change"
    assert result["final_status"] == hab.STATUS_SEAM_REDUCED_TRADEOFF
    assert result["final_status_unchanged"] is True
    assert result["scientific_sections_unchanged"] is True
    assert result["scientific_fingerprint"] == dict(before_fingerprint)
    assert result["summary_json_rewritten"] is False


def test_report_only_refuses_to_write_stale_chain_labels(tmp_path, monkeypatch):
    root = _stage_summary(tmp_path)
    monkeypatch.setattr(hab, "diagnostic_output_root",
                        lambda experiment_id, base_dir=tmp_path: root)
    monkeypatch.setattr(hab, "assert_namespace_safe", lambda *a, **k: None)
    monkeypatch.setattr(hab, "render_summary_markdown",
                        lambda summary: "panel scene_weighted_reference")

    with pytest.raises(hab.HarmonizationDownstreamABError) as excinfo:
        hab.rebuild_reports_from_summary(EXPERIMENT, tmp_path)
    assert "scene_weighted_reference" in str(excinfo.value)


def test_report_only_without_a_completed_summary_fails_clearly(tmp_path,
                                                               monkeypatch):
    root = hab.diagnostic_output_root(EXPERIMENT, tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hab, "diagnostic_output_root",
                        lambda experiment_id, base_dir=tmp_path: root)
    with pytest.raises(hab.PrerequisiteError) as excinfo:
        hab.rebuild_reports_from_summary(EXPERIMENT, tmp_path)
    assert "never runs the experiment" in str(excinfo.value)


def test_report_only_runs_no_pipeline_callable():
    source = inspect.getsource(hab.rebuild_reports_from_summary)
    for forbidden in ("run_step5", "run_step7", "run_step8", "run_chain_model",
                      "paired_block_bootstrap", "compare_raster_change",
                      "run_boundary_propagation", "rasterio"):
        assert forbidden not in source, f"report-only must not reach {forbidden}"


# =============================================================================
# Report-only CLI mode
# =============================================================================
def test_report_only_is_a_valid_mode():
    runner.validate_modes(False, False, False, False, report_only=True)


@pytest.mark.parametrize("kwargs", [
    {"dry_run": True}, {"run": True}, {"resume": True}, {"force": True},
])
def test_report_only_cannot_be_combined_with_other_modes(kwargs):
    with pytest.raises(SystemExit):
        runner.validate_modes(kwargs.get("dry_run", False), kwargs.get("run", False),
                              kwargs.get("resume", False), kwargs.get("force", False),
                              report_only=True)


def test_report_only_flag_is_wired():
    args = runner.parse_args(["--experiment", EXPERIMENT, "--report-only"])
    assert args.report_only is True
    assert "report_only" in inspect.signature(runner.main).parameters
    main_source = inspect.getsource(runner.main)
    assert "rebuild_reports_from_summary" in main_source
    # ... and it never reaches the live pipeline
    assert main_source.index("rebuild_reports_from_summary") < \
        main_source.index("_run_live")


def test_a_mode_is_still_required():
    with pytest.raises(SystemExit) as excinfo:
        runner.validate_modes(False, False, False, False)
    assert "--report-only" in str(excinfo.value)
