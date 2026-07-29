"""Focused tests for the current-support date-offset harmonization counterfactual.

No Earth Engine, no Step5-Step8 rerun and no live experiment are required. The
tests exercise the CLI mode contract, the no-op dry-run, namespace/force safety,
the deterministic daily inventory, the overlap-graph construction and
connectivity gate, the weighted least-squares offset solution, the EXACT support
invariance gate, the reference reproduction, the paired spatial-block bootstrap,
the ordered decision rule, atomic checkpointing/resume, and the report
invariants.
"""

from __future__ import annotations

import ast
import csv
import inspect
import json
import shutil
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.landsat_composite_downstream_ab as ab
import src.landsat_current_support_harmonization as hz
import src.landsat_residual_seam_attribution as rs
import scripts.run_landsat_current_support_harmonization as runner

EXPERIMENT = "manavgat_2021"


# =============================================================================
# Helpers
# =============================================================================
def _write_raster(path: Path, array, *, nodata=np.nan, transform=None,
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


def _mean_accumulator(unit_values):
    accumulator = hz.MeanAccumulator()
    for unit, values in unit_values.items():
        accumulator.add(np.full(len(values), unit, dtype="int64"),
                        np.asarray(values, dtype="float64"))
    return accumulator


def _edge(date_i, date_j, delta, *, blocks=20, sigma=0.2, pixels=50000):
    return OrderedDict((
        ("date_i", date_i), ("date_j", date_j),
        ("edge_median_difference_celsius", float(delta)),
        ("independent_blocks", int(blocks)),
        ("edge_sigma_celsius", float(sigma)),
        ("common_valid_pixels", int(pixels)),
        ("eligible", True),
    ))


def _reduction_row(verdict, relative=0.5, low=0.1, high=0.9):
    return {
        "verdict": verdict, "relative_paired_reduction": relative,
        "interval_low": low, "interval_high": high, "status": "estimated",
    }


def _valid_evidence(**overrides):
    """Evidence that reaches the eligibility branch with everything passing."""
    reductions = {}
    for product in hz.TARGET_PRODUCTS:
        reductions[product] = {
            boundary: _reduction_row(hz.VERDICT_UNCERTAIN, relative=None,
                                     low=-0.1, high=0.1)
            for boundary in hz.EVALUATED_BOUNDARIES
        }
        for boundary in hz.REQUIRED_REDUCTION_BOUNDARIES:
            reductions[product][boundary] = _reduction_row(
                hz.VERDICT_SUPPORTED_REDUCTION, relative=0.42, low=0.2, high=0.7)
    evidence = {
        "inputs_valid": True,
        "invalid_input_reasons": [],
        "reference_reproduction_passes": True,
        "reference_reproduction_failures": [],
        "primary_graph_connected": True,
        "graph_failure_reasons": [],
        "support_invariance_passes": True,
        "support_invariance_failures": [],
        "boundary_reductions": reductions,
        "global_median_current_lst_shift": 0.02,
        "max_abs_date_offset": 1.4,
        "offset_estimation_stable": True,
        "offset_instability_reasons": [],
    }
    evidence.update(overrides)
    return evidence


# =============================================================================
# Synthetic frozen fixture (no Earth Engine, no pipeline rerun)
# =============================================================================
GRID = 384          # 3 x 3 spatial blocks of 128 cells
DATES = ("2021-06-04", "2021-06-13", "2021-06-20")
TRUE_ALPHA = (-1.0, 0.25, 0.75)


def _synthetic_dailies():
    """Three daily mosaics that share a surface field plus a known date offset."""
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    surface = 28.0 + 0.01 * rows + 0.004 * cols
    stack = []
    for index, alpha in enumerate(TRUE_ALPHA):
        daily = surface + alpha
        # Each date loses a different corner, so the per-pixel date support
        # genuinely varies -- which is the mechanism under test.
        daily = daily.astype("float64").copy()
        if index == 0:
            daily[:40, :40] = np.nan
        elif index == 1:
            daily[-40:, -40:] = np.nan
        else:
            daily[:30, -30:] = np.nan
        stack.append(daily)
    return np.stack(stack, axis=0)


def _build_fixture(base_dir: Path) -> dict:
    """A minimal but COMPLETE frozen namespace, built from numpy only."""
    stack = _synthetic_dailies()
    valid_count = np.isfinite(stack).sum(axis=0).astype("float64")
    median = np.where(valid_count > 0,
                      np.nanmedian(np.where(np.isfinite(stack), stack, np.nan), axis=0),
                      np.nan)
    current = np.where(valid_count >= 2, median, np.nan)
    current = np.where((current > -30) & (current < 80), current, np.nan)
    baseline_mean = np.full((GRID, GRID), 26.0)
    baseline_std = np.full((GRID, GRID), 2.0)
    baseline_std[:20, :] = 0.5                       # low-std guard region
    baseline_count = np.full((GRID, GRID), 4.0)
    cmb = current - baseline_mean
    anomaly = np.where((valid_count >= 2) & (baseline_std >= 1.0),
                       cmb / baseline_std, np.nan)

    ab_root = (base_dir / "outputs" / "diagnostics"
               / hz.DOWNSTREAM_AB_NAMESPACE / EXPERIMENT)
    step5 = ab_root / "candidate" / "step5"
    derived = ab_root / "candidate" / "derived"
    cf_root = (base_dir / "outputs" / "diagnostics"
               / hz.COUNTERFACTUAL_NAMESPACE / EXPERIMENT)
    rs_root = (base_dir / "outputs" / "diagnostics"
               / hz.RESIDUAL_SEAM_NAMESPACE / EXPERIMENT)

    _write_raster(step5 / "current_period_median_celsius.tif", current)
    _write_raster(step5 / "baseline_lst_mean_celsius.tif", baseline_mean)
    _write_raster(step5 / "baseline_lst_std_celsius.tif", baseline_std)
    _write_raster(step5 / "baseline_valid_count.tif", baseline_count)
    _write_raster(step5 / "current_period_valid_count.tif",
                  np.where(valid_count > 0, valid_count, np.nan))
    _write_raster(step5 / "anomaly_zscore.tif", anomaly)
    _write_raster(step5 / "low_baseline_std_mask.tif",
                  (baseline_std < 1.0).astype("float32"))
    _write_raster(step5 / "low_baseline_count_mask.tif", np.zeros((GRID, GRID)))
    _write_raster(step5 / "low_current_count_mask.tif",
                  np.where(valid_count > 0, (valid_count < 2).astype("float64"), np.nan))
    _write_raster(derived / "current_minus_baseline_celsius.tif", cmb)

    _write_raster(cf_root / "rasters" / "current_lst_unique_date_valid_count.tif",
                  np.where(valid_count > 0, valid_count, np.nan))
    _write_raster(cf_root / "rasters" / "current_lst_scene_valid_count.tif",
                  np.where(valid_count > 0, valid_count, np.nan))
    _write_raster(cf_root / "rasters" / "current_lst_same_day_multiplicity.tif",
                  np.where(valid_count > 0, 0.0, np.nan))

    columns = ["input_role", "baseline_year", "scene_id", "landsat_product_id",
               "spacecraft_id", "sensor_id", "wrs_path", "wrs_row",
               "acquisition_datetime", "acquisition_date", "cloud_cover",
               "cloud_cover_land", "processing_level", "collection_category",
               "collection_number", "source_collection"]
    rows_out = []
    for index, date in enumerate(DATES):
        for row in (34, 35):
            rows_out.append({
                "input_role": "current_lst", "baseline_year": "",
                "scene_id": f"LC08_17{7 + index % 2}0{row}_{date.replace('-', '')}",
                "landsat_product_id": "LC08_L2SP", "spacecraft_id": "LANDSAT_8",
                "sensor_id": "OLI_TIRS", "wrs_path": str(177 + index % 2),
                "wrs_row": str(row), "acquisition_datetime": f"{date}T08:27:00",
                "acquisition_date": date, "cloud_cover": "1.0",
                "cloud_cover_land": "1.0", "processing_level": "L2SP",
                "collection_category": "T1", "collection_number": "2",
                "source_collection": "LANDSAT/LC08/C02/T1_L2",
            })
    cf_root.mkdir(parents=True, exist_ok=True)
    with open(cf_root / "scene_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows_out)

    (cf_root / "audit_config.json").write_text(json.dumps({
        "source_collection": "LANDSAT/LC08/C02/T1_L2",
        "current_period": {
            "start_date": "2021-06-01", "end_date": "2021-07-27",
            "window_days": 56, "months_filter": "1-12",
            "date_window_semantics": {"end_semantics": "exclusive",
                                      "effective_last_included_date": "2021-07-26"},
        },
        "baseline_years": [2017, 2018, 2019, 2020],
        "step5_policy": {"landsat_scale": 0.00341802, "landsat_offset": 149.0},
        "qa_mask": {"qa_source": "QA_PIXEL"},
    }), encoding="utf-8")
    (cf_root / "counterfactual_summary.json").write_text(json.dumps({
        "final_status": "supported_reduction",
        "canonical_reproduction": {"status": "pass"},
        "provenance": {"state": "provenance_unavailable"},
    }), encoding="utf-8")
    (cf_root / "manifest.json").write_text("{}", encoding="utf-8")

    ab_root.mkdir(parents=True, exist_ok=True)
    (ab_root / "downstream_ab_summary.json").write_text(json.dumps({
        "final_status": "eligible_for_second_aoi_validation",
        "production_approved": False, "changes_production_reducer": False,
        "technical_validity": {"baseline_invariance_status": "pass"},
    }), encoding="utf-8")
    (ab_root / "reference_reproduction.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8")
    (ab_root / "downstream_ab_manifest.json").write_text("{}", encoding="utf-8")
    (ab_root / "input_provenance.json").write_text("{}", encoding="utf-8")

    rs_root.mkdir(parents=True, exist_ok=True)
    (rs_root / "residual_seam_summary.json").write_text(
        json.dumps({"final_status": "current_support_dominant"}), encoding="utf-8")
    (rs_root / "residual_seam_manifest.json").write_text("{}", encoding="utf-8")

    root = hz.diagnostic_output_root(EXPERIMENT, base_dir)
    daily_paths = []
    for index, date in enumerate(DATES):
        path = hz.daily_raster_path(root, date, kind="reference")
        _write_raster(path, stack[index])
        daily_paths.append(path)

    return {
        "base_dir": base_dir, "root": root, "daily_paths": daily_paths,
        "stack": stack, "current": current, "cmb": cmb, "anomaly": anomaly,
        "valid_count": valid_count,
    }


@pytest.fixture()
def fixture(tmp_path):
    return _build_fixture(tmp_path)


# =============================================================================
# CLI contract
# =============================================================================
@pytest.mark.parametrize("kwargs,message", [
    ({"dry_run": True, "run": True}, "mutually exclusive"),
    ({}, "one of --dry-run or --run is required"),
    ({"run": True, "resume": True, "force": True}, "mutually exclusive"),
    ({"dry_run": True, "resume": True}, "--resume requires --run"),
    ({"dry_run": True, "force": True}, "--force requires --run"),
])
def test_cli_mode_conflicts_are_rejected(kwargs, message):
    with pytest.raises(SystemExit) as excinfo:
        runner.validate_modes(
            kwargs.get("dry_run", False), kwargs.get("run", False),
            kwargs.get("resume", False), kwargs.get("force", False),
        )
    assert message in str(excinfo.value)


def test_default_execution_is_never_implied():
    with pytest.raises(SystemExit):
        runner.main(experiment_id=EXPERIMENT)


def test_argparse_requires_a_mode():
    parser = runner.build_parser()
    args = parser.parse_args(["--experiment", EXPERIMENT])
    with pytest.raises(SystemExit):
        runner.validate_modes(args.dry_run, args.run, args.resume, args.force)


def test_unsupported_experiment_is_refused():
    with pytest.raises(hz.HarmonizationError):
        hz.assert_supported_experiment("bejis_2022")


# =============================================================================
# Dry-run writes nothing
# =============================================================================
def test_dry_run_writes_nothing(tmp_path):
    root = hz.diagnostic_output_root(EXPERIMENT)
    existed = root.exists()
    before = sorted(p.name for p in root.iterdir()) if existed else None

    plan = hz.build_dry_run_plan(EXPERIMENT)

    assert plan["writes_performed"] is False
    assert plan["directories_created"] == 0
    assert plan["rasters_modified"] == 0
    assert plan["frozen_namespaces_touched"] == 0
    assert root.exists() is existed
    if existed:
        assert sorted(p.name for p in root.iterdir()) == before


def test_dry_run_reports_every_required_section():
    plan = hz.build_dry_run_plan(EXPERIMENT)
    for key in ("resolved_inputs", "upstream_prerequisites", "daily_export_plan",
                "configuration", "decision_rule", "expected_files",
                "planned_stages", "limitations", "pathrow_evidence"):
        assert key in plan
    assert plan["planned_stages"] == list(hz.PLANNED_STAGES)
    assert plan["allowed_final_statuses"] == list(hz.FINAL_STATUSES)
    assert plan["smoothing_applied"] is False
    assert plan["baseline_recomputed"] is False


def test_dry_run_resolves_the_frozen_inputs_and_upstream_gate():
    plan = hz.build_dry_run_plan(EXPERIMENT)
    assert plan["missing_required_inputs"] == []
    state = plan["upstream_prerequisites"]
    assert state["counterfactual_final_status"] == "supported_reduction"
    assert state["downstream_ab_final_status"] == "eligible_for_second_aoi_validation"
    assert state["residual_seam_final_status"] == "current_support_dominant"
    assert state["downstream_ab_reference_reproduction"] == "pass"
    assert state["baseline_invariance"] == "pass"
    assert state["prerequisites_met"] is True


@pytest.mark.parametrize("key", [
    "counterfactual_final_status", "downstream_ab_final_status",
    "residual_seam_final_status", "downstream_ab_reference_reproduction",
    "baseline_invariance",
])
def test_every_upstream_status_is_required(key):
    state = hz.load_upstream_state(EXPERIMENT)
    broken = dict(state)
    broken[key] = "something_else"
    assert hz.upstream_prerequisites_met(broken) is False
    with pytest.raises(hz.PrerequisiteError):
        hz.validate_upstream_state(broken)


# =============================================================================
# Earth Engine is unreachable from the analysis
# =============================================================================
def _module_source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def test_analysis_module_never_imports_earth_engine():
    tree = ast.parse(_module_source(hz))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "ee" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "ee"


@pytest.mark.parametrize("forbidden", [
    "ee.Initialize", "ee.Authenticate", "ee.ImageCollection", "getInfo",
    "init_gee", "toDrive",
])
def test_analysis_module_references_no_earth_engine_symbol(forbidden):
    assert forbidden not in _module_source(hz)


def test_analysis_callables_never_touch_earth_engine():
    for callable_ in (hz.build_input_plan, hz.build_daily_export_plan,
                      hz.daily_date_inventory, hz.run_reference_reproduction,
                      hz.run_overlap_evidence, hz.run_harmonisation,
                      hz.run_boundary_analysis, hz.solve_date_offsets,
                      hz.decide_final_status, hz.build_dry_run_plan):
        source = inspect.getsource(callable_)
        for forbidden in ("import ee", "ee.", "gee_utils", "getInfo"):
            assert forbidden not in source, f"{callable_.__name__} touches {forbidden}"


def test_earth_engine_guard_wraps_the_dry_run_and_the_analysis():
    source = _module_source(runner)
    assert source.count("with ab.EarthEngineGuard():") >= 2


def test_tests_cannot_initialise_earth_engine():
    """Under the guard, any EE entry point raises instead of contacting Google."""
    import src.landsat_composite_downstream_ab as ab

    with ab.EarthEngineGuard():
        plan = hz.build_dry_run_plan(EXPERIMENT)
        assert plan["earth_engine_calls"] == 0
        assert plan["earth_engine_initialised"] is False
        ee = sys.modules.get("ee")
        if ee is not None and hasattr(ee, "Initialize"):
            with pytest.raises(Exception):
                ee.Initialize()


def test_export_stage_is_the_only_earth_engine_entry_point():
    tree = ast.parse(_module_source(runner))
    ee_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = ast.dump(node)
            if "'ee'" in body and "Import" in body:
                ee_functions.add(node.name)
    assert ee_functions <= {"_build_daily_ee_image"}


# =============================================================================
# Frozen outputs are never modified
# =============================================================================
def test_frozen_outputs_are_never_modified():
    plan = hz.build_input_plan(EXPERIMENT)
    watched = [Path(e["path"]) for e in plan.values() if Path(e["path"]).exists()]
    watched += [p for p in hz.upstream_report_paths(EXPERIMENT).values() if p.exists()]
    before = {p: hz.sha256_and_size(p) for p in watched}

    hz.build_dry_run_plan(EXPERIMENT)
    hz.build_input_provenance(EXPERIMENT)
    hz.build_config_snapshot(EXPERIMENT)

    for path, signed in before.items():
        assert hz.sha256_and_size(path) == signed, f"frozen input changed: {path}"


@pytest.mark.parametrize("namespace", [
    hz.COUNTERFACTUAL_NAMESPACE, hz.DOWNSTREAM_AB_NAMESPACE,
    hz.RESIDUAL_SEAM_NAMESPACE,
])
def test_writing_into_a_frozen_namespace_is_refused(namespace):
    target = (hz.PROJECT_ROOT / "outputs" / "diagnostics" / namespace
              / EXPERIMENT / "intruder.json")
    with pytest.raises(hz.NamespaceSafetyError):
        hz.assert_namespace_safe([target], EXPERIMENT)


def test_writing_outside_the_diagnostic_root_is_refused(tmp_path):
    with pytest.raises(hz.NamespaceSafetyError):
        hz.assert_namespace_safe([tmp_path / "elsewhere.json"], EXPERIMENT)


def test_force_cannot_escape_the_diagnostic_root(tmp_path, monkeypatch):
    victim = tmp_path / "outputs" / "experiments" / EXPERIMENT
    victim.mkdir(parents=True)
    (victim / "precious.tif").write_text("do not delete", encoding="utf-8")

    monkeypatch.setattr(
        hz, "diagnostic_output_root",
        lambda experiment_id, base_dir=tmp_path: victim,
    )
    with pytest.raises(hz.NamespaceSafetyError):
        hz.clear_diagnostic_namespace(EXPERIMENT, tmp_path)
    assert (victim / "precious.tif").exists()


def test_force_deletes_only_the_dedicated_namespace(tmp_path):
    root = hz.diagnostic_output_root(EXPERIMENT, tmp_path)
    root.mkdir(parents=True)
    (root / "stale.json").write_text("{}", encoding="utf-8")
    sibling = (tmp_path / "outputs" / "diagnostics"
               / hz.RESIDUAL_SEAM_NAMESPACE / EXPERIMENT)
    sibling.mkdir(parents=True)
    (sibling / "frozen.json").write_text("{}", encoding="utf-8")

    hz.clear_diagnostic_namespace(EXPERIMENT, tmp_path)

    assert not root.exists()
    assert (sibling / "frozen.json").exists()


# =============================================================================
# Grid contract
# =============================================================================
def test_exact_grid_mismatch_fails(tmp_path):
    from rasterio.transform import Affine

    a = _write_raster(tmp_path / "a.tif", np.zeros((8, 8)))
    b = _write_raster(tmp_path / "b.tif", np.zeros((8, 8)),
                      transform=Affine(0.0005, 0.0, 31.0, 0.0, -0.0005, 37.35))
    with pytest.raises(hz.GridMismatchError):
        hz.assert_same_grid([a, b])


def test_shape_mismatch_fails(tmp_path):
    a = _write_raster(tmp_path / "a.tif", np.zeros((8, 8)))
    b = _write_raster(tmp_path / "b.tif", np.zeros((9, 8)))
    with pytest.raises(hz.GridMismatchError):
        hz.assert_same_grid([a, b])


# =============================================================================
# Nodata is never replaced with zero
# =============================================================================
def test_nodata_is_never_replaced_with_zero(tmp_path):
    array = np.array([[1.0, np.nan], [3.0, 4.0]])
    path = _write_raster(tmp_path / "nan.tif", array)
    window = hz.read_window(path, 0, 2)
    assert np.isnan(window[0, 1])
    assert not (window == 0.0).any()


def test_sentinel_nodata_becomes_nan_not_zero(tmp_path):
    array = np.array([[1.0, hz.NODATA_SENTINEL], [3.0, 4.0]])
    path = _write_raster(tmp_path / "sentinel.tif", array, nodata=hz.NODATA_SENTINEL)
    window = hz.read_window(path, 0, 2)
    assert np.isnan(window[0, 1])
    assert not (window == 0.0).any()


def test_missing_dates_never_contribute_zero_to_the_median():
    stack = np.array([[[10.0]], [[np.nan]], [[20.0]]])
    median, count = hz.nanmedian_over_dates(stack)
    assert median[0, 0] == pytest.approx(15.0)
    assert count[0, 0] == 2


def test_a_pixel_with_no_valid_date_stays_nan():
    stack = np.full((3, 1, 1), np.nan)
    median, count = hz.nanmedian_over_dates(stack)
    assert np.isnan(median[0, 0])
    assert count[0, 0] == 0


# =============================================================================
# Daily date inventory
# =============================================================================
def test_daily_date_inventory_is_deterministic():
    records = hz.current_scene_records(EXPERIMENT)
    first = hz.daily_date_inventory(records)
    shuffled = list(reversed(records))
    second = hz.daily_date_inventory(shuffled)
    assert list(first) == list(second) == sorted(first)
    for date in first:
        assert first[date]["scene_ids"] == second[date]["scene_ids"]


def test_same_day_scenes_are_one_temporal_observation():
    inventory = hz.daily_date_inventory(hz.current_scene_records(EXPERIMENT))
    assert inventory, "the frozen manifest must carry current-period scenes"
    for date, entry in inventory.items():
        assert entry["temporal_observations"] == 1
        assert entry["scene_count"] >= 1
    total_scenes = sum(e["scene_count"] for e in inventory.values())
    assert total_scenes > len(inventory), (
        "this AOI has same-day scene pairs; collapsing them must reduce the count"
    )


def test_daily_inventory_rejects_a_date_outside_the_frozen_window():
    window = hz.frozen_current_window(EXPERIMENT)
    with pytest.raises(hz.HarmonizationError):
        hz.assert_dates_within_frozen_window(["2021-09-01"], window)
    with pytest.raises(hz.HarmonizationError):
        hz.assert_dates_within_frozen_window(["2021-05-01"], window)


def test_frozen_window_end_is_exclusive_and_preserved():
    window = hz.frozen_current_window(EXPERIMENT)
    assert window["end_semantics"] == "exclusive"
    hz.assert_dates_within_frozen_window(
        [window["effective_last_included_date"]], window)
    with pytest.raises(hz.HarmonizationError):
        hz.assert_dates_within_frozen_window([window["end_date"]], window)


def test_export_plan_never_touches_earth_engine_and_exports_no_baseline():
    plan = hz.build_daily_export_plan(EXPERIMENT)
    assert plan["earth_engine_touched_by_this_function"] is False
    assert plan["date_count"] == len(plan["items"])
    assert "baseline" in plan["export_contract"]["never_exports"]
    for item in plan["items"]:
        assert "daily" in item["output_path"]
        assert hz.DIAGNOSTIC_NAMESPACE in item["output_path"]


# =============================================================================
# Overlap graph construction
# =============================================================================
def test_overlap_edge_uses_common_valid_pixels_only():
    """A pixel one date cannot see must not enter that pair's difference."""
    stack = np.stack([
        np.full((128, 128), 10.0),
        np.full((128, 128), 12.0),
    ])
    stack[1, :64, :] = np.nan
    # Where date 1 is blind, date 0 carries an extreme value that would wreck a
    # naive difference if it were included.
    stack[0, :64, :] = 1000.0

    blocks = hz.block_grid_ids(128, 128, 0)
    store = hz.accumulate_pair_block_medians(stack, blocks, {})
    entry = store[(0, 1)]
    assert entry["common_valid_pixels"] == 64 * 128
    estimate = hz.robust_edge_estimate(entry["block_medians"])
    assert estimate["median_difference_celsius"] == pytest.approx(2.0)


def test_overlap_edge_uses_spatial_block_medians_not_a_pooled_mean():
    """One spatially concentrated block cannot drag the edge estimate."""
    stack = np.stack([np.zeros((384, 384)), np.full((384, 384), 1.0)])
    # One whole block is wildly different; block-median voting must ignore it.
    stack[1, :128, :128] = 500.0

    blocks = hz.block_grid_ids(384, 384, 0)
    store = hz.accumulate_pair_block_medians(stack, blocks, {})
    estimate = hz.robust_edge_estimate(store[(0, 1)]["block_medians"])
    assert estimate["n_blocks"] == 9
    assert estimate["median_difference_celsius"] == pytest.approx(1.0)
    pooled_mean = float(np.nanmean(stack[1] - stack[0]))
    assert pooled_mean > 50.0, "the contaminated pooled mean is the thing we avoid"


def test_blocks_below_the_minimum_pixel_count_are_not_counted():
    stack = np.stack([np.zeros((128, 128)), np.ones((128, 128))])
    stack[:, :, 50:] = np.nan          # only 50 common columns per row
    stack[:, 5:, :] = np.nan           # ... and only 5 rows -> 250 pixels
    blocks = hz.block_grid_ids(128, 128, 0)
    store = hz.accumulate_pair_block_medians(stack, blocks, {},
                                             min_block_pixels=1000)
    entry = store[(0, 1)]
    assert entry["blocks_below_min_pixels"] == 1
    assert entry["block_medians"] == []


def test_edge_eligibility_uses_the_predeclared_primary_thresholds():
    store = {(0, 1): {"date_i_index": 0, "date_j_index": 1,
                      "block_medians": [1.0] * 4, "block_labels": [],
                      "block_pixel_counts": [], "common_valid_pixels": 999999,
                      "blocks_seen": 4, "blocks_below_min_pixels": 0}}
    graph = hz.build_overlap_graph(
        ["a", "b"], store,
        min_common_pixels=hz.PRIMARY_MIN_COMMON_PIXELS,
        min_independent_blocks=hz.PRIMARY_MIN_INDEPENDENT_BLOCKS)
    assert graph["edge_count"] == 0, "4 blocks is below the 8-block minimum"

    loose = hz.build_overlap_graph(["a", "b"], store, min_common_pixels=5000,
                                   min_independent_blocks=4)
    assert loose["edge_count"] == 1


def test_primary_thresholds_are_the_predeclared_ones():
    assert hz.PRIMARY_MIN_COMMON_PIXELS == 10000
    assert hz.PRIMARY_MIN_INDEPENDENT_BLOCKS == 8
    labels = [t["label"] for t in hz.SENSITIVITY_THRESHOLDS]
    assert labels[0] == "primary"
    assert ("loose_5000_5", "strict_25000_12") == tuple(labels[1:])


# =============================================================================
# Graph connectivity gate
# =============================================================================
def test_graph_connectivity_gate():
    dates = ["d1", "d2", "d3"]
    connected = [_edge("d1", "d2", 0.5), _edge("d2", "d3", 0.5)]
    assert hz.graph_is_connected(dates, connected) is True
    assert len(hz.connected_components(dates, connected)) == 1

    split = [_edge("d1", "d2", 0.5)]
    assert hz.graph_is_connected(dates, split) is False
    components = hz.connected_components(dates, split)
    assert [len(c) for c in components] == [2, 1]


def test_no_date_is_ever_silently_dropped():
    dates = ["d1", "d2", "d3"]
    diagnostics = hz.build_graph_diagnostics(
        dates, hz.build_overlap_graph(dates, {}, min_common_pixels=1,
                                      min_independent_blocks=1))
    assert diagnostics["dates_dropped"] == []
    assert set(diagnostics["isolated_dates"]) == set(dates)
    assert diagnostics["connected"] is False


def test_articulation_nodes_are_identified():
    dates = ["a", "b", "c"]
    edges = [_edge("a", "b", 0.1), _edge("b", "c", 0.1)]
    assert hz.articulation_nodes(dates, edges) == ["b"]
    edges.append(_edge("a", "c", 0.2))
    assert hz.articulation_nodes(dates, edges) == []


def test_cycle_consistency_detects_an_inconsistent_triangle():
    dates = ["a", "b", "c"]
    consistent = [_edge("a", "b", 1.0), _edge("b", "c", 1.0), _edge("a", "c", 2.0)]
    assert hz.cycle_consistency(dates, consistent)["max_abs_closure_error_celsius"] \
        == pytest.approx(0.0)
    inconsistent = [_edge("a", "b", 1.0), _edge("b", "c", 1.0), _edge("a", "c", 5.0)]
    assert hz.cycle_consistency(dates, inconsistent)["max_abs_closure_error_celsius"] \
        == pytest.approx(3.0)


def test_disconnected_graph_cannot_produce_an_eligible_status():
    decision = hz.decide_final_status(_valid_evidence(primary_graph_connected=False))
    assert decision["final_status"] == hz.STATUS_INSUFFICIENT_GRAPH
    assert decision["final_status"] != hz.STATUS_ELIGIBLE


# =============================================================================
# Offset solution
# =============================================================================
def test_pairwise_offset_signs_are_correct():
    """delta_ij is median(Y_j - Y_i) = alpha_j - alpha_i."""
    dates = ["d1", "d2"]
    edges = [_edge("d1", "d2", 2.0)]
    solution = hz.solve_date_offsets(dates, edges, {"d1": 100.0, "d2": 100.0})
    alpha = solution["alpha_by_date"]
    assert alpha["d2"] - alpha["d1"] == pytest.approx(2.0)
    assert alpha["d1"] < 0 < alpha["d2"]


def test_weighted_mean_offset_constraint_equals_zero():
    dates = ["d1", "d2", "d3"]
    edges = [_edge("d1", "d2", 1.0), _edge("d2", "d3", 2.0), _edge("d1", "d3", 3.0)]
    counts = {"d1": 100.0, "d2": 300.0, "d3": 50.0}
    solution = hz.solve_date_offsets(dates, edges, counts)
    weighted = sum(counts[d] * solution["alpha_by_date"][d] for d in dates)
    assert weighted == pytest.approx(0.0, abs=1e-9)
    assert solution["weighted_mean_offset_is_zero"] is True


def test_offsets_are_not_anchored_to_a_single_date():
    """The level comes from the weighted constraint, not from pinning a date.

    Anchoring would make one date's offset identically zero for EVERY choice of
    observation weights. Here the whole vector slides when the weights change,
    which is only possible if no date is pinned.
    """
    dates = ["d1", "d2", "d3"]
    edges = [_edge("d1", "d2", 1.0), _edge("d2", "d3", 1.0)]
    balanced = hz.solve_date_offsets(dates, edges, dict.fromkeys(dates, 100.0))
    skewed = hz.solve_date_offsets(dates, edges,
                                   {"d1": 1000.0, "d2": 10.0, "d3": 10.0})

    # Same differences (the graph identifies only differences) ...
    for a, b in (("d1", "d2"), ("d2", "d3")):
        assert (balanced["alpha_by_date"][b] - balanced["alpha_by_date"][a]) == \
            pytest.approx(skewed["alpha_by_date"][b] - skewed["alpha_by_date"][a])
    # ... but a different level, so no single date is pinned to zero.
    for date in dates:
        assert balanced["alpha_by_date"][date] != pytest.approx(
            skewed["alpha_by_date"][date])


def test_offsets_recover_a_known_consistent_solution():
    dates = ["d1", "d2", "d3"]
    truth = {"d1": -1.0, "d2": 0.25, "d3": 0.75}
    edges = [
        _edge("d1", "d2", truth["d2"] - truth["d1"]),
        _edge("d2", "d3", truth["d3"] - truth["d2"]),
        _edge("d1", "d3", truth["d3"] - truth["d1"]),
    ]
    solution = hz.solve_date_offsets(dates, edges, dict.fromkeys(dates, 100.0))
    centre = sum(truth.values()) / 3.0
    for date in dates:
        assert solution["alpha_by_date"][date] == pytest.approx(truth[date] - centre)
    assert solution["edge_residual_rms_celsius"] == pytest.approx(0.0, abs=1e-9)


def test_graph_solution_is_deterministic():
    dates = ["d1", "d2", "d3", "d4"]
    edges = [_edge("d1", "d2", 0.4, blocks=11), _edge("d2", "d3", -0.7, blocks=31),
             _edge("d3", "d4", 1.1, blocks=9), _edge("d1", "d4", 0.9, blocks=17)]
    counts = {"d1": 10.0, "d2": 20.0, "d3": 30.0, "d4": 40.0}
    first = hz.solve_date_offsets(dates, edges, counts)
    second = hz.solve_date_offsets(dates, edges, counts)
    assert first["alpha_by_date"] == second["alpha_by_date"]
    assert first["graph_condition_number"] == second["graph_condition_number"]


def test_edge_weights_are_capped_so_one_pair_cannot_dominate():
    edges = [_edge("a", "b", 0.1, blocks=10, sigma=0.2),
             _edge("b", "c", 0.1, blocks=10, sigma=0.2),
             _edge("a", "c", 0.1, blocks=100000, sigma=0.05)]
    weights = hz.edge_weights(edges)
    assert weights["capped_edge_count"] == 1
    assert max(weights["capped"]) == pytest.approx(
        hz.WEIGHT_CAP_MULTIPLE * weights["median_raw"])
    assert max(weights["capped"]) < max(weights["raw"])


def test_edge_weights_use_only_overlap_evidence():
    source = inspect.getsource(hz.edge_weights)
    for forbidden in ("label", "step8", "auc", "model", "jump", "seam"):
        assert forbidden not in source.lower()


def test_edge_sigma_is_floored():
    estimate = hz.robust_edge_estimate([1.0, 1.0, 1.0, 1.0])
    assert estimate["mad_celsius"] == pytest.approx(0.0)
    assert estimate["sigma_celsius"] == pytest.approx(hz.MIN_EDGE_SIGMA_CELSIUS)


# =============================================================================
# Reference reproduction and support invariance (synthetic frozen fixture)
# =============================================================================
def test_reference_daily_median_reproduces_the_frozen_reference(fixture):
    report = hz.run_reference_reproduction(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES),
        base_dir=fixture["base_dir"])

    assert report["passes"] is True, report["failures"]
    assert report["exact_checks"]["valid_mask_equality"] is True
    assert report["exact_checks"]["valid_date_count_equality"] is True
    for product in hz.TARGET_PRODUCTS:
        entry = report["products"][product]
        assert entry["valid_mask_exactly_equal"] is True
        assert entry["max_abs_difference"] <= entry["gating_tolerance"]
    assert report["unique_date_valid_count"]["unequal_pixel_count"] == 0


def test_reference_reproduction_failure_blocks_scientific_evaluation(fixture):
    """A corrupted frozen reference must end the run, not be worked around."""
    step5 = (fixture["base_dir"] / "outputs" / "diagnostics"
             / hz.DOWNSTREAM_AB_NAMESPACE / EXPERIMENT / "candidate" / "step5")
    corrupted = fixture["current"] + 5.0
    _write_raster(step5 / "current_period_median_celsius.tif", corrupted)

    report = hz.run_reference_reproduction(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES),
        base_dir=fixture["base_dir"])
    assert report["passes"] is False

    decision = hz.decide_final_status(
        _valid_evidence(reference_reproduction_passes=False,
                        reference_reproduction_failures=report["failures"]))
    assert decision["final_status"] == hz.STATUS_INVALID_REFERENCE


def test_support_count_and_masks_are_exactly_invariant(fixture):
    hz.run_reference_reproduction(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES),
        base_dir=fixture["base_dir"])
    alpha = {date: value for date, value in zip(DATES, (-0.4, 0.1, 0.3))}
    result = hz.run_harmonisation(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES), alpha,
        base_dir=fixture["base_dir"])

    invariance = result["support_invariance"]
    assert invariance["passes"] is True, invariance["failed_checks"]
    by_name = {c["check"]: c for c in invariance["checks"]}
    for name in ("unique_date_valid_count", "valid_mask", "anomaly_valid_mask",
                 "daily_membership_per_pixel", "current_valid_count",
                 "low_current_count_mask"):
        assert by_name[name]["unequal_pixel_count"] == 0
        assert by_name[name]["changed_valid_pixel_count"] == 0
        assert by_name[name]["mask_agreement"] == 1.0


def test_harmonisation_actually_changes_values_while_preserving_support(fixture):
    hz.run_reference_reproduction(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES),
        base_dir=fixture["base_dir"])
    alpha = {date: value for date, value in zip(DATES, TRUE_ALPHA)}
    centre = float(np.mean(TRUE_ALPHA))
    alpha = {date: value - centre for date, value in alpha.items()}
    result = hz.run_harmonisation(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES), alpha,
        base_dir=fixture["base_dir"])

    assert result["support_invariance"]["passes"] is True
    change = result["raster_changes"][hz.TARGET_LST]
    assert change["mae"] is not None and change["mae"] > 0.0, (
        "removing a real date offset must change the composite values"
    )
    assert change["valid_mask_agreement"] == 1.0


def test_support_invariance_failure_is_detected_and_blocks_the_status():
    accumulator = hz.ExactComparisonAccumulator("valid_mask")
    reference = np.array([[1.0, 1.0], [1.0, np.nan]])
    candidate = np.array([[1.0, np.nan], [1.0, np.nan]])
    accumulator.add(reference, candidate)
    report = accumulator.report()
    assert report["changed_valid_pixel_count"] == 1
    assert report["mask_agreement"] < 1.0
    assert report["passes"] is False

    verdict = hz.support_invariance_verdict([report])
    assert verdict["passes"] is False
    decision = hz.decide_final_status(
        _valid_evidence(support_invariance_passes=False,
                        support_invariance_failures=verdict["failed_checks"]))
    assert decision["final_status"] == hz.STATUS_SUPPORT_INVARIANCE_FAILED


def test_exact_comparison_counts_both_sides_invalid_as_equal():
    accumulator = hz.ExactComparisonAccumulator("count")
    accumulator.add(np.array([[np.nan, 2.0]]), np.array([[np.nan, 2.0]]))
    report = accumulator.report()
    assert report["unequal_pixel_count"] == 0
    assert report["exact_equal_pixel_count"] == 2
    assert report["passes"] is True


# =============================================================================
# No smoothing / no spatial interpolation
# =============================================================================
FORBIDDEN_SPATIAL_OPERATIONS = frozenset({
    "gaussian_filter", "uniform_filter", "median_filter", "convolve",
    "convolve2d", "griddata", "interp2d", "interpn", "RectBivariateSpline",
    "fillnodata", "focal_mean", "boxcar", "smooth", "blur", "feather",
    "inpaint", "reproject", "zoom", "resample", "gaussian_blur",
})


def _called_names(module) -> set[str]:
    """Every function/method name that is actually CALLED in a module."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(_module_source(module))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


@pytest.mark.parametrize("module", [hz, runner])
def test_no_smoothing_or_spatial_interpolation_function_is_reachable(module):
    called = _called_names(module)
    offenders = sorted(called & FORBIDDEN_SPATIAL_OPERATIONS)
    assert offenders == [], (
        f"{offenders} must not be reachable: this experiment never alters a "
        "raster spatially"
    )


@pytest.mark.parametrize("symbol", sorted(FORBIDDEN_SPATIAL_OPERATIONS))
def test_no_smoothing_symbol_is_imported(symbol):
    for module in (hz, runner):
        for node in ast.walk(ast.parse(_module_source(module))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    assert alias.name.split(".")[-1] != symbol
                    assert (alias.asname or "") != symbol


def test_scipy_ndimage_is_never_imported():
    for module in (hz, runner):
        source = _module_source(module)
        assert "scipy.ndimage" not in source
        assert "from scipy" not in source


def test_harmonisation_is_a_pure_per_date_scalar_subtraction():
    source = inspect.getsource(hz.run_harmonisation)
    assert "stack - offsets[:, None, None]" in source
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else
              getattr(node.func, "id", "")
              for node in ast.walk(ast.parse(inspect.getsource(hz.run_harmonisation)
                                             .lstrip()))
              if isinstance(node, ast.Call)}
    assert not (called & FORBIDDEN_SPATIAL_OPERATIONS)


def test_maps_are_never_smoothed_before_plotting():
    assert hz.MAP_INTERPOLATION == "nearest"
    sources = "\n".join(
        inspect.getsource(getattr(hz, name)) for name in
        ("_save_single", "render_product_maps", "render_support_boundary_maps",
         "render_top_residual_jump_map"))
    assert sources.count("interpolation=MAP_INTERPOLATION") >= 4
    assert "bilinear" not in sources and "bicubic" not in sources
    assert "antialiased=True" not in sources


# =============================================================================
# Paired bootstrap
# =============================================================================
def test_paired_bootstrap_samples_blocks_not_pairs():
    """Doubling every block's pair count must not shrink the interval."""
    blocks = {i: [1.0 + 0.1 * i] * 4 for i in range(12)}
    doubled = {i: values * 2 for i, values in blocks.items()}

    def run(mapping):
        boundary = _mean_accumulator({i: [v + 0.5 for v in values]
                                      for i, values in mapping.items()})
        control = _mean_accumulator(mapping)
        return hz.bootstrap_paired_reduction(
            boundary, control,
            _mean_accumulator({i: [v + 0.2 for v in values]
                               for i, values in mapping.items()}),
            control)

    single, double = run(blocks), run(doubled)
    assert single["n_units"] == double["n_units"] == 12
    width_single = single["interval_high"] - single["interval_low"]
    width_double = double["interval_high"] - double["interval_low"]
    assert width_double == pytest.approx(width_single, rel=1e-9)
    assert single["resamples_individual_pairs"] is False


def test_reference_and_candidate_reuse_identical_bootstrap_draws():
    blocks = {i: [float(i)] for i in range(16)}
    boundary_ref = _mean_accumulator({i: [v + 1.0 for v in values]
                                      for i, values in blocks.items()})
    boundary_cand = _mean_accumulator({i: [v + 0.6 for v in values]
                                       for i, values in blocks.items()})
    control = _mean_accumulator(blocks)

    first = hz.bootstrap_paired_reduction(boundary_ref, control,
                                          boundary_cand, control)
    second = hz.bootstrap_paired_reduction(boundary_ref, control,
                                           boundary_cand, control)
    assert first["interval_low"] == second["interval_low"]
    assert first["interval_high"] == second["interval_high"]
    assert first["identical_draws_for_reference_and_candidate"] is True
    # The paired reduction of two constant shifts is exactly their difference,
    # which is only possible when both arms share every draw.
    assert first["paired_reduction"] == pytest.approx(0.4)
    assert first["interval_low"] == pytest.approx(0.4)
    assert first["interval_high"] == pytest.approx(0.4)


def test_bootstrap_uses_the_declared_seed_replicates_and_block_size():
    blocks = {i: [float(i)] for i in range(10)}
    row = hz.bootstrap_paired_reduction(
        _mean_accumulator(blocks), _mean_accumulator(blocks),
        _mean_accumulator(blocks), _mean_accumulator(blocks))
    assert row["seed"] == 42
    assert row["n_bootstrap_requested"] == 1000
    assert row["block_size_cells"] == 128
    assert row["ci"] == 0.95
    assert row["unit_type"] == "spatial_block"


def test_bootstrap_refuses_too_few_blocks():
    blocks = {i: [float(i)] for i in range(3)}
    row = hz.bootstrap_paired_reduction(
        _mean_accumulator(blocks), _mean_accumulator(blocks),
        _mean_accumulator(blocks), _mean_accumulator(blocks))
    assert row["status"] == "insufficient_units"
    assert row["verdict"] == hz.VERDICT_INSUFFICIENT
    assert row["interval_low"] is None


@pytest.mark.parametrize("low,high,expected", [
    (0.1, 0.9, hz.VERDICT_SUPPORTED_REDUCTION),
    (-0.9, -0.1, hz.VERDICT_SUPPORTED_INCREASE),
    (-0.2, 0.4, hz.VERDICT_UNCERTAIN),
])
def test_reduction_interval_classification(low, high, expected):
    row = {"status": "estimated", "interval_low": low, "interval_high": high}
    assert hz.classify_reduction_interval(row) == expected


# =============================================================================
# Decision rule
# =============================================================================
def test_a_fully_passing_run_is_eligible_for_downstream_ab():
    decision = hz.decide_final_status(_valid_evidence())
    assert decision["final_status"] == hz.STATUS_ELIGIBLE
    assert decision["seam_fixed"] is False
    assert decision["production_approved"] is False


def test_invalid_inputs_short_circuit_everything():
    decision = hz.decide_final_status(
        _valid_evidence(inputs_valid=False, invalid_input_reasons=["missing raster"]))
    assert decision["final_status"] == hz.STATUS_INVALID_INPUTS


def test_minimum_10_percent_reduction_rule():
    evidence = _valid_evidence()
    evidence["boundary_reductions"][hz.TARGET_ANOMALY]["current_unique_date_count_change"] = \
        _reduction_row(hz.VERDICT_SUPPORTED_REDUCTION, relative=0.09, low=0.01, high=0.2)
    decision = hz.decide_final_status(evidence)
    assert decision["final_status"] == hz.STATUS_NOT_SUPPORTED
    assert any("relative reduction" in reason for reason in decision["reasons"])

    evidence["boundary_reductions"][hz.TARGET_ANOMALY]["current_unique_date_count_change"] = \
        _reduction_row(hz.VERDICT_SUPPORTED_REDUCTION, relative=0.10, low=0.01, high=0.2)
    assert hz.decide_final_status(evidence)["final_status"] == hz.STATUS_ELIGIBLE


def test_an_interval_crossing_zero_is_not_a_supported_reduction():
    evidence = _valid_evidence()
    evidence["boundary_reductions"][hz.TARGET_CMB]["current_support_change"] = \
        _reduction_row(hz.VERDICT_UNCERTAIN, relative=0.4, low=-0.1, high=0.9)
    assert hz.decide_final_status(evidence)["final_status"] == hz.STATUS_NOT_SUPPORTED


def test_both_decision_products_are_required():
    for product in hz.DECISION_PRODUCTS:
        evidence = _valid_evidence()
        evidence["boundary_reductions"][product]["current_support_change"] = \
            _reduction_row(hz.VERDICT_UNCERTAIN, relative=None, low=-0.2, high=0.2)
        assert hz.decide_final_status(evidence)["final_status"] == \
            hz.STATUS_NOT_SUPPORTED


def test_nonboundary_tradeoff_rule():
    evidence = _valid_evidence()
    evidence["boundary_reductions"][hz.TARGET_CMB][rs.CLASS_NONE] = \
        _reduction_row(hz.VERDICT_SUPPORTED_INCREASE, relative=-0.3,
                       low=-0.9, high=-0.2)
    decision = hz.decide_final_status(evidence)
    assert decision["final_status"] == hz.STATUS_NONBOUNDARY_TRADEOFF
    assert decision["checks"]["no_supported_increase_at_nonboundary"] is False


def test_pathrow_only_increase_blocks_eligibility():
    evidence = _valid_evidence()
    evidence["boundary_reductions"][hz.TARGET_ANOMALY][rs.CLASS_PATHROW_ONLY] = \
        _reduction_row(hz.VERDICT_SUPPORTED_INCREASE, relative=-0.2,
                       low=-0.5, high=-0.1)
    decision = hz.decide_final_status(evidence)
    assert decision["final_status"] != hz.STATUS_ELIGIBLE
    assert decision["final_status"] == hz.PATHROW_INCREASE_STATUS
    assert decision["checks"]["no_supported_increase_at_pathrow_only"] is False


@pytest.mark.parametrize("shift", [0.51, -0.51, 3.0, None])
def test_global_median_shift_rule(shift):
    decision = hz.decide_final_status(
        _valid_evidence(global_median_current_lst_shift=shift))
    assert decision["final_status"] == hz.STATUS_VALUE_SCALE_TRADEOFF


def test_global_median_shift_at_the_bound_is_allowed():
    decision = hz.decide_final_status(
        _valid_evidence(global_median_current_lst_shift=0.5))
    assert decision["final_status"] == hz.STATUS_ELIGIBLE


@pytest.mark.parametrize("offset", [5.01, -6.0, None])
def test_five_celsius_maximum_offset_rule(offset):
    decision = hz.decide_final_status(_valid_evidence(max_abs_date_offset=offset))
    assert decision["final_status"] == hz.STATUS_VALUE_SCALE_TRADEOFF


def test_offset_at_the_bound_is_allowed():
    decision = hz.decide_final_status(_valid_evidence(max_abs_date_offset=5.0))
    assert decision["final_status"] == hz.STATUS_ELIGIBLE


def test_unstable_offset_estimation_is_a_value_scale_tradeoff():
    decision = hz.decide_final_status(
        _valid_evidence(offset_estimation_stable=False,
                        offset_instability_reasons=["edge residual RMS too large"]))
    assert decision["final_status"] == hz.STATUS_VALUE_SCALE_TRADEOFF


def test_decision_bounds_are_the_predeclared_ones():
    assert hz.MIN_RELATIVE_REDUCTION == 0.10
    assert hz.MAX_ABS_DATE_OFFSET_CELSIUS == 5.0
    assert hz.MAX_ABS_GLOBAL_MEDIAN_SHIFT_CELSIUS == 0.5


def test_forbidden_final_statuses_cannot_be_emitted():
    for banned in hz.FORBIDDEN_CONCLUSIONS:
        assert banned not in hz.FINAL_STATUSES
        with pytest.raises(hz.HarmonizationError):
            hz._status(banned, [], {}, {})
    with pytest.raises(hz.HarmonizationError):
        hz._status("definitely_not_a_status", [], {}, {})


def test_every_reachable_status_is_declared():
    variants = [
        _valid_evidence(inputs_valid=False),
        _valid_evidence(reference_reproduction_passes=False),
        _valid_evidence(primary_graph_connected=False),
        _valid_evidence(support_invariance_passes=False),
        _valid_evidence(global_median_current_lst_shift=9.0),
        _valid_evidence(max_abs_date_offset=99.0),
        _valid_evidence(),
    ]
    for evidence in variants:
        assert hz.decide_final_status(evidence)["final_status"] in hz.FINAL_STATUSES


# =============================================================================
# No labels, no Step8, no model metrics
# =============================================================================
#: Modules whose import would mean a label, burned-area or model-performance
#: artefact had become reachable.
FORBIDDEN_MODULES = (
    "src.step6_validate_fire_relation", "src.step6a_prepare_gate_inputs",
    "src.step6b_burned_landcover_gate", "src.step7b_prepare_downscaling_dataset",
    "src.step7c_train_downscaling_model", "src.step8a_prepare_500m_modeling_dataset",
    "src.step8b_train_baseline_vs_thermal_model", "src.step8e_final_report",
    "src.burned_pattern_audit", "core.validation_burned_area", "sklearn",
    "lightgbm", "xgboost",
)


@pytest.mark.parametrize("forbidden", FORBIDDEN_MODULES)
def test_no_label_or_step8_module_is_importable_from_here(forbidden):
    head = forbidden.split(".")[0]
    for module in (hz, runner):
        for node in ast.walk(ast.parse(_module_source(module))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != forbidden
                    assert alias.name.split(".")[0] != head or head not in (
                        "sklearn", "lightgbm", "xgboost")
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "") != forbidden


def _code_identifiers(module) -> set[str]:
    """Every identifier that appears in EXECUTABLE code (no prose, no strings).

    Comments, docstrings and string literals are excluded on purpose: a module
    is allowed -- and required -- to state in prose that it never touches labels
    or Step8 metrics. What it must not do is NAME one in running code.
    """
    identifiers: set[str] = set()
    for node in ast.walk(ast.parse(_module_source(module))):
        if isinstance(node, ast.Name):
            identifiers.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr.lower())
        elif isinstance(node, ast.FunctionDef):
            identifiers.add(node.name.lower())
            identifiers.update(a.arg.lower() for a in node.args.args)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                identifiers.add((alias.asname or alias.name).lower())
    return identifiers


#: Substrings that may never appear in an executable identifier. Declarative
#: invariants such as `USES_LABELS = False` are allowlisted by exact name.
LABEL_TOKENS = ("burned", "burn_date", "mcd64", "auc", "r2_score", "oof",
                "y_true", "y_pred", "landcover", "fire")

DECLARATIVE_IDENTIFIERS = frozenset({
    "uses_labels", "uses_step8_metrics", "uses_model_performance",
    "labels_used", "step8_metrics_used", "model_performance_used",
})


@pytest.mark.parametrize("token", LABEL_TOKENS)
def test_no_label_or_model_metric_identifier_is_reachable(token):
    for module in (hz, runner):
        offenders = sorted(
            name for name in _code_identifiers(module)
            if token in name and name not in DECLARATIVE_IDENTIFIERS
        )
        assert offenders == [], (
            f"{offenders} name a label/burned-area/model artefact in executable "
            "code; no such input may construct or select this candidate"
        )


@pytest.mark.parametrize("token", ["step6", "step7", "step8", "label"])
def test_pipeline_and_label_words_are_never_executable_identifiers(token):
    """The words may appear in prose prohibitions, never in running code."""
    allowed = DECLARATIVE_IDENTIFIERS | {
        "step6_step7_step8_rerun", "reruns_step6_step7_step8",
        # `block_id_to_label` / `block_labels` / `threshold_label` are graph and
        # plotting labels, not target labels.
        "block_id_to_label", "block_labels", "block_label", "label",
        "threshold_label", "labels", "set_xticklabels", "set_ylabel",
        "set_xlabel", "xlabel", "ylabel", "_membership_labels",
        "membership_labels", "block_labels",
    }
    for module in (hz, runner):
        offenders = sorted(
            name for name in _code_identifiers(module)
            if token in name and name not in allowed
        )
        assert offenders == [], f"{offenders} appear in executable code"


def test_the_only_pipeline_stage_imported_is_step5_policy():
    """Step5 output-profile helpers are the ONLY pipeline import permitted."""
    imported = set()
    for module in (hz, runner):
        for node in ast.walk(ast.parse(_module_source(module))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                imported.add(node.module)
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names if a.name.startswith("src."))
    assert imported <= {
        "src.step5_preprocess_timeseries",
        "src.landsat_composite_counterfactual_audit",
        "src.landsat_composite_downstream_ab",
        "src.landsat_residual_seam_attribution",
        "src.landsat_current_support_harmonization",
    }, f"unexpected pipeline import: {imported}"


def test_input_plan_reads_no_label_or_step8_artefact():
    plan = hz.build_input_plan(EXPERIMENT)
    for entry in plan.values():
        path = str(entry["path"]).lower()
        for token in ("step8", "step7", "step6", "label", "burn", "validation"):
            assert token not in path, f"{entry['role']} points at {path}"


def test_declared_invariants_are_false():
    assert hz.USES_LABELS is False
    assert hz.USES_STEP8_METRICS is False
    assert hz.USES_MODEL_PERFORMANCE is False
    assert hz.SMOOTHING_APPLIED is False
    assert hz.SPATIAL_INTERPOLATION_APPLIED is False
    assert hz.RECOMPUTES_BASELINE is False
    assert hz.CHANGES_PRODUCTION_REDUCER is False


def test_the_baseline_is_read_only_and_never_recomputed():
    plan = hz.build_input_plan(EXPERIMENT)
    frozen = [role for role, entry in plan.items()
              if entry["family"] in ("frozen_baseline", "frozen_mask")]
    assert "baseline_lst_mean_celsius" in frozen
    assert "baseline_lst_std_celsius" in frozen
    assert "baseline_valid_count" in frozen
    source = _module_source(hz)
    assert "nanstd" not in source, "a baseline std must never be recomputed here"
    provenance = hz.build_input_provenance(EXPERIMENT)
    assert provenance["baseline_recomputed"] is False
    assert provenance["baseline_re_exported"] is False


# =============================================================================
# Reports
# =============================================================================
def _minimal_summary() -> dict:
    evidence = _valid_evidence()
    decision = hz.decide_final_status(evidence)
    reproduction = OrderedDict((
        ("date_count", 3), ("dates", list(DATES)),
        ("grid_contract", {"status": "pass", "raster_count": 4}),
        ("exact_checks", OrderedDict((("grid_signature_equality", True),
                                      ("valid_mask_equality", True),
                                      ("valid_date_count_equality", True)))),
        ("unique_date_valid_count", {"passes": True, "unequal_pixel_count": 0}),
        ("current_valid_count", {"passes": True, "unequal_pixel_count": 0}),
        ("products", OrderedDict(
            (product, {"max_abs_difference": 1e-7,
                       "gating_tolerance": hz.REPRODUCTION_TOLERANCES[product],
                       "valid_mask_exactly_equal": True, "passes": True})
            for product in hz.TARGET_PRODUCTS)),
        ("passes", True), ("failures", []),
    ))
    graph = hz.build_overlap_graph(
        list(DATES),
        {(0, 1): {"block_medians": [1.0] * 10, "common_valid_pixels": 50000,
                  "blocks_seen": 10, "blocks_below_min_pixels": 0},
         (1, 2): {"block_medians": [0.5] * 10, "common_valid_pixels": 50000,
                  "blocks_seen": 10, "blocks_below_min_pixels": 0},
         (0, 2): {"block_medians": [1.5] * 10, "common_valid_pixels": 50000,
                  "blocks_seen": 10, "blocks_below_min_pixels": 0}},
        min_common_pixels=hz.PRIMARY_MIN_COMMON_PIXELS,
        min_independent_blocks=hz.PRIMARY_MIN_INDEPENDENT_BLOCKS,
        grid_cells=GRID * GRID)
    date_entries = {d: {"valid_pixel_count": 1000.0} for d in DATES}
    diagnostics = hz.build_graph_diagnostics(list(DATES), graph, date_entries)
    solution = hz.solve_date_offsets(list(DATES), graph["edges"],
                                     dict.fromkeys(DATES, 1000.0))
    invariance = hz.support_invariance_verdict([
        hz.ExactComparisonAccumulator(name).report()
        for name in hz.SUPPORT_INVARIANCE_CHECKS])
    changes = OrderedDict(
        (product, hz.RasterChangeAccumulator(product).report())
        for product in hz.TARGET_PRODUCTS)
    evaluation = runner._empty_evaluation()
    tradeoff = hz.nonboundary_tradeoff(evaluation)
    return hz.build_summary(
        EXPERIMENT, state=hz.load_upstream_state(EXPERIMENT),
        config=hz.build_config_snapshot(EXPERIMENT),
        provenance={"missing_required_inputs": [], "missing_optional_inputs": []},
        inventory=hz.daily_date_inventory(hz.current_scene_records(EXPERIMENT)),
        reproduction=reproduction, graph=graph, diagnostics=diagnostics,
        solution=solution, sensitivity=[], invariance=invariance,
        changes=changes, evaluation=evaluation, tradeoff=tradeoff,
        decision=decision, resources={"elapsed_seconds": 1.0})


def test_report_generation_does_not_alter_scientific_metrics():
    summary = _minimal_summary()
    before = json.loads(json.dumps(summary, default=str))
    hz.render_summary_markdown(summary)
    after = json.loads(json.dumps(summary, default=str))
    assert hz.report_generation_preserves_metrics(before, after)


def test_markdown_carries_the_twelve_required_sections():
    markdown = hz.render_summary_markdown(_minimal_summary())
    for heading in (
        "## 1. Technical validity",
        "## 2. Frozen reference reproduction",
        "## 3. Date-overlap graph",
        "## 4. Fitted date offsets",
        "## 5. Support invariance",
        "## 6. Raster changes",
        "## 7. Support-boundary reductions",
        "## 8. Non-boundary trade-offs",
        "## 9. Path/row check",
        "## 10. Decision",
        "## 11. Limitations",
        "## 12. Next experiment",
    ):
        assert heading in markdown


def test_markdown_carries_every_required_limitation():
    markdown = hz.render_summary_markdown(_minimal_summary())
    for limitation in hz.required_limitations():
        assert limitation in markdown


def test_summary_never_claims_a_forbidden_conclusion():
    summary = _minimal_summary()
    assert hz.summary_forbids_banned_conclusions(summary)
    assert summary["seam_fixed"] is False
    assert summary["production_approved"] is False
    assert summary["production_ready"] is False


def test_a_summary_that_claims_a_forbidden_conclusion_is_rejected():
    payload = {"conclusion": "the seam_fixed result is final"}
    assert hz.summary_forbids_banned_conclusions(payload) is False


def test_required_limitations_cover_the_declared_topics():
    text = " ".join(hz.required_limitations()).lower()
    for topic in ("manavgat", "additive", "weather", "terrain", "metadata",
                  "pixel-level", "production", "model-performance",
                  "generalisation", "climatology", "anomaly-threshold",
                  "only seam mechanism"):
        assert topic in text


# =============================================================================
# Tables
# =============================================================================
def test_every_declared_table_has_columns():
    assert set(hz.TABLE_FILES) == {
        "raster_change_summary.csv", "boundary_jump_comparison.csv",
        "paired_bootstrap_summary.csv", "nonboundary_tradeoff.csv",
        "date_offset_sensitivity.csv",
    }
    for columns in (hz.RASTER_CHANGE_COLUMNS, hz.BOUNDARY_COLUMNS,
                    hz.BOOTSTRAP_COLUMNS, hz.NONBOUNDARY_COLUMNS,
                    hz.DATE_NODE_COLUMNS, hz.DATE_EDGE_COLUMNS,
                    hz.DATE_OFFSET_COLUMNS, hz.SENSITIVITY_COLUMNS):
        assert len(columns) == len(set(columns)) and columns


def test_boundary_table_covers_every_declared_boundary_and_product():
    rows = hz.boundary_rows(runner._empty_evaluation())
    pairs = {(row["product"], row["boundary"]) for row in rows}
    expected = {(p, b) for p in hz.TARGET_PRODUCTS for b in hz.EVALUATED_BOUNDARIES}
    assert pairs == expected


def test_sensitivity_table_marks_only_the_primary_graph():
    entries = []
    for thresholds in hz.SENSITIVITY_THRESHOLDS:
        graph = hz.build_overlap_graph(
            list(DATES), {}, min_common_pixels=thresholds["min_common_pixels"],
            min_independent_blocks=thresholds["min_independent_blocks"])
        entries.append({"label": thresholds["label"], "graph": graph,
                        "diagnostics": hz.build_graph_diagnostics(list(DATES), graph),
                        "solution": None})
    rows = hz.sensitivity_rows(entries)
    primary = [r for r in rows if r["used_for_primary_candidate"]]
    assert len(primary) == 1
    assert primary[0]["threshold_label"] == "primary"
    assert primary[0]["min_common_pixels"] == hz.PRIMARY_MIN_COMMON_PIXELS


def test_csv_writer_is_atomic(tmp_path):
    path = tmp_path / "table.csv"
    hz.write_csv(path, [{"a": 1, "b": 2}], ("a", "b"))
    assert path.read_text(encoding="utf-8").splitlines()[0] == "a,b"
    assert not list(tmp_path.glob(".*.tmp"))


# =============================================================================
# Checkpoints and resume
# =============================================================================
def test_checkpoint_stage_names_are_validated(tmp_path):
    with pytest.raises(hz.HarmonizationError):
        hz.write_checkpoint_stage(tmp_path, "not_a_stage", [])


def test_checkpoint_is_written_atomically_and_records_hashes(tmp_path):
    output = tmp_path / "artifact.json"
    output.write_text('{"value": 1}', encoding="utf-8")
    hz.write_checkpoint_stage(tmp_path, "input_validation", [output])

    payload = hz.read_checkpoint(tmp_path)
    entry = payload["stages"]["input_validation"]
    assert entry["outputs"][0]["sha256"] == hz.sha256_and_size(output)["sha256"]
    assert payload["last_stage"] == "input_validation"
    assert not list((tmp_path / "checkpoints").glob(".*.tmp"))


def test_resume_validates_hashes_and_rejects_a_changed_output(tmp_path):
    output = tmp_path / "artifact.json"
    output.write_text('{"value": 1}', encoding="utf-8")
    hz.write_checkpoint_stage(tmp_path, "input_validation", [output])
    assert hz.stage_is_reusable(tmp_path, "input_validation") is True

    output.write_text('{"value": 2}', encoding="utf-8")
    assert hz.stage_is_reusable(tmp_path, "input_validation") is False


def test_resume_rejects_a_deleted_output(tmp_path):
    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")
    hz.write_checkpoint_stage(tmp_path, "reports", [output])
    output.unlink()
    assert hz.stage_is_reusable(tmp_path, "reports") is False


def test_resume_rejects_a_foreign_checkpoint_schema(tmp_path):
    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")
    hz.write_checkpoint_stage(tmp_path, "maps", [output])
    path = hz.checkpoint_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint_schema_version"] = "0.0-old"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert hz.stage_is_reusable(tmp_path, "maps") is False


def test_all_planned_stages_are_checkpointable(tmp_path):
    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")
    for stage in hz.PLANNED_STAGES:
        hz.write_checkpoint_stage(tmp_path, stage, [output])
    payload = hz.read_checkpoint(tmp_path)
    assert set(payload["stages"]) == set(hz.PLANNED_STAGES)


def test_checkpoint_records_memory_usage(tmp_path):
    output = tmp_path / "artifact.json"
    output.write_text("{}", encoding="utf-8")
    hz.write_checkpoint_stage(tmp_path, "bootstrap", [output])
    entry = hz.read_checkpoint(tmp_path)["stages"]["bootstrap"]
    assert entry["rss_mib"] is None or entry["rss_mib"] >= 0.0


# =============================================================================
# Reused semantics (identical to the residual seam attribution audit)
# =============================================================================
def test_boundary_and_bootstrap_semantics_are_reused_not_reimplemented():
    assert hz.BOOTSTRAP_BLOCK_SIZE_CELLS == rs.BOOTSTRAP_BLOCK_SIZE_CELLS == 128
    assert hz.BOOTSTRAP_SEED == rs.BOOTSTRAP_SEED == 42
    assert hz.BOOTSTRAP_REPLICATES == rs.BOOTSTRAP_REPLICATES == 1000
    assert hz.BOOTSTRAP_CI == rs.BOOTSTRAP_CI == 0.95
    assert hz.MIN_BOOTSTRAP_UNITS == rs.MIN_BOOTSTRAP_UNITS
    assert hz.build_edge_flags is rs.build_edge_flags
    assert hz.stratified_class_codes is rs.stratified_class_codes
    assert hz.matched_block_accumulators is rs.matched_block_accumulators
    assert hz.draw_bootstrap_indices is rs.draw_bootstrap_indices
    assert hz.spatial_block_ids is rs.spatial_block_ids
    assert hz.GRAPH_BLOCK_SIZE_CELLS == rs.BOOTSTRAP_BLOCK_SIZE_CELLS


def test_every_required_boundary_is_evaluated():
    required = {
        "current_support_change", "current_unique_date_count_change",
        "current_scene_count_change", "current_valid_count_change",
        "same_day_multiplicity_change", "baseline_valid_year_change",
        "baseline_annual_date_support_change", "near_std_threshold_boundary",
        "source_path_row_boundary", rs.CLASS_PATHROW_ONLY,
        rs.CLASS_SUPPORT_AND_PATHROW, rs.CLASS_SUPPORT_ONLY, rs.CLASS_NONE,
    }
    assert set(hz.EVALUATED_BOUNDARIES) == required
    assert hz.EVALUATED_BOUNDARIES[rs.CLASS_NONE] == hz.EVAL_MODE_MEAN


def test_target_products_are_the_three_declared_ones():
    assert hz.TARGET_PRODUCTS == ("current_lst_celsius",
                                  "current_minus_baseline_celsius",
                                  "anomaly_zscore")
    assert hz.DECISION_PRODUCTS == ("current_minus_baseline_celsius",
                                    "anomaly_zscore")


def test_step5_policy_is_read_from_the_canonical_configuration():
    thresholds = hz.step5_thresholds()
    assert thresholds == rs.step5_thresholds()
    assert thresholds["min_current_valid_count"] == 2
    assert thresholds["min_baseline_std_celsius"] == 1.0


def test_anomaly_mask_follows_the_canonical_step5_rule():
    thresholds = hz.step5_thresholds()
    current = np.array([[30.0, 30.0, 30.0]])
    mean = np.array([[26.0, 26.0, 26.0]])
    std = np.array([[2.0, 0.5, 2.0]])
    count = np.array([[3.0, 3.0, 1.0]])
    anomaly = hz.build_anomaly_zscore(current, mean, std, count, thresholds)
    assert anomaly[0, 0] == pytest.approx(2.0)
    assert np.isnan(anomaly[0, 1])       # baseline std below the guard
    assert np.isnan(anomaly[0, 2])       # current support below the guard


# =============================================================================
# End-to-end integration on the synthetic frozen fixture
# =============================================================================
def test_end_to_end_recovers_an_injected_date_offset(fixture):
    """The whole chain, on data with a KNOWN injected acquisition-date offset.

    This is the positive control: the daily mosaics share one surface field and
    differ only by `TRUE_ALPHA`, and the per-pixel date support varies. The
    experiment must (a) reproduce the frozen reference, (b) build a connected
    graph, (c) recover the injected offsets up to the weighted-mean constraint,
    (d) preserve support EXACTLY, and (e) shrink the support-boundary jump it
    was designed to target.
    """
    reproduction = hz.run_reference_reproduction(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES),
        base_dir=fixture["base_dir"])
    assert reproduction["passes"] is True

    store, counts = hz.run_overlap_evidence(fixture["daily_paths"], GRID, GRID)
    graph = hz.build_overlap_graph(
        list(DATES), store,
        min_common_pixels=hz.PRIMARY_MIN_COMMON_PIXELS,
        min_independent_blocks=hz.PRIMARY_MIN_INDEPENDENT_BLOCKS,
        grid_cells=GRID * GRID)
    diagnostics = hz.build_graph_diagnostics(
        list(DATES), graph,
        {d: {"valid_pixel_count": counts[i]} for i, d in enumerate(DATES)})
    assert diagnostics["connected"] is True
    assert graph["edge_count"] == 3

    solution = hz.solve_date_offsets(
        list(DATES), graph["edges"],
        {d: float(counts[i]) for i, d in enumerate(DATES)})
    centre = float(np.mean(TRUE_ALPHA))
    for date, truth in zip(DATES, TRUE_ALPHA):
        assert solution["alpha_by_date"][date] == pytest.approx(truth - centre,
                                                                abs=0.02)
    assert solution["weighted_mean_offset_is_zero"] is True
    assert solution["estimation_stable"] is True
    assert abs(solution["max_abs_offset_celsius"]) <= hz.MAX_ABS_DATE_OFFSET_CELSIUS

    result = hz.run_harmonisation(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES),
        solution["alpha_by_date"], base_dir=fixture["base_dir"])
    assert result["support_invariance"]["passes"] is True
    shift = result["raster_changes"][hz.TARGET_LST]["global_median_shift"]
    assert abs(shift) <= hz.MAX_ABS_GLOBAL_MEDIAN_SHIFT_CELSIUS

    analysis = hz.run_boundary_analysis(EXPERIMENT, fixture["root"],
                                        base_dir=fixture["base_dir"])
    assert analysis["pair_counts"]["total"] > 0
    assert analysis["pair_counts"]["dropped_invalid_endpoint"] > 0, (
        "pairs touching a missing endpoint must be dropped, never zero-filled"
    )
    evaluation = hz.evaluate_boundaries(analysis)

    for product in hz.DECISION_PRODUCTS:
        row = evaluation["boundary_reductions"][product]["current_unique_date_count_change"]
        assert row["reference_excess_absolute_jump"] > 0.0
        assert row["paired_reduction"] > 0.0
        assert row["relative_paired_reduction"] > hz.MIN_RELATIVE_REDUCTION
        # The non-boundary control must NOT move: the intervention is targeted.
        control = evaluation["boundary_reductions"][product][hz.NONBOUNDARY_CONTROL]
        assert control["paired_reduction"] == pytest.approx(0.0, abs=1e-9)


def test_end_to_end_writes_only_inside_the_diagnostic_root(fixture):
    hz.run_reference_reproduction(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES),
        base_dir=fixture["base_dir"])
    hz.run_harmonisation(
        EXPERIMENT, fixture["root"], fixture["daily_paths"], list(DATES),
        dict.fromkeys(DATES, 0.1), base_dir=fixture["base_dir"])

    root = fixture["root"].resolve()
    frozen_roots = [
        hz.counterfactual_root(EXPERIMENT, fixture["base_dir"]).resolve(),
        hz.downstream_ab_root(EXPERIMENT, fixture["base_dir"]).resolve(),
        hz.residual_seam_root(EXPERIMENT, fixture["base_dir"]).resolve(),
    ]
    written = [p for p in fixture["base_dir"].rglob("*") if p.is_file()]
    new = [p for p in written if p.stat().st_mtime > 0 and root in p.resolve().parents]
    assert new, "the experiment must write something"
    for path in written:
        resolved = path.resolve()
        if any(frozen in resolved.parents for frozen in frozen_roots):
            assert "harmonization" not in resolved.name, (
                f"an output leaked into a frozen namespace: {resolved}"
            )


# =============================================================================
# Dry-run must answer the daily-mosaic question explicitly
# =============================================================================
def test_export_plan_states_whether_daily_mosaics_already_exist():
    plan = hz.build_daily_export_plan(EXPERIMENT)
    assert isinstance(plan["complete_daily_mosaics_present"], bool)
    assert plan["daily_mosaic_status"] in ("complete", "partial", "none_present")
    expected = bool(plan["items"]) and not plan["missing_locally"]
    assert plan["complete_daily_mosaics_present"] is expected
    if plan["daily_mosaic_status"] == "complete":
        assert plan["missing_locally"] == []
    if plan["daily_mosaic_status"] == "none_present":
        assert len(plan["missing_locally"]) == plan["date_count"]


def test_export_plan_lists_exact_dates_scenes_and_download_paths():
    plan = hz.build_daily_export_plan(EXPERIMENT)
    inventory = hz.daily_date_inventory(hz.current_scene_records(EXPERIMENT))

    assert plan["required_dates"] == list(inventory)
    assert len(plan["planned_download_paths"]) == plan["date_count"]
    for item in plan["items"]:
        entry = inventory[item["acquisition_date"]]
        assert item["scene_ids"] == entry["scene_ids"]
        assert item["landsat_product_ids"] == entry["landsat_product_ids"]
        assert item["path_rows"] == entry["path_rows"]
        assert item["temporal_observations"] == 1
        assert item["planned_download_path"] == item["output_path"]
        assert item["planned_download_path"].startswith(
            plan["planned_download_root"])
        assert hz.DIAGNOSTIC_NAMESPACE in item["planned_download_path"]
    total = sum(len(i["scene_ids"]) for i in plan["items"])
    assert total == plan["scene_count"]


def test_export_plan_carries_the_user_executed_fetch_command():
    plan = hz.build_daily_export_plan(EXPERIMENT)
    command = plan["fetch_command"]
    assert hz.RUNNER_SCRIPT in command
    assert f"--experiment {EXPERIMENT}" in command
    assert "--run" in command
    assert "--dry-run" not in command
    assert (PROJECT_ROOT / hz.RUNNER_SCRIPT).exists()
    assert "user" in plan["fetch_command_note"].lower()


def test_present_daily_mosaics_are_reported_with_a_hash(fixture):
    """A locally present mosaic must be identified by content, not by name."""
    base = fixture["base_dir"]

    complete = hz.build_daily_export_plan(EXPERIMENT, base)
    assert complete["complete_daily_mosaics_present"] is True
    assert complete["daily_mosaic_status"] == "complete"
    assert complete["missing_locally"] == []
    for item in complete["items"]:
        assert item["present_locally"] is True
        assert item["verified_sha256"] == hz.sha256_and_size(
            Path(item["planned_download_path"]))["sha256"]

    # Remove one date: the verdict must flip to partial, never stay "complete".
    removed = complete["items"][1]["acquisition_date"]
    Path(complete["items"][1]["planned_download_path"]).unlink()

    partial = hz.build_daily_export_plan(EXPERIMENT, base)
    assert partial["complete_daily_mosaics_present"] is False
    assert partial["daily_mosaic_status"] == "partial"
    assert partial["missing_locally"] == [removed]
    by_date = {i["acquisition_date"]: i for i in partial["items"]}
    assert by_date[removed]["verified_sha256"] is None
    assert by_date[removed]["present_locally"] is False


def test_dry_run_printer_emits_the_daily_mosaic_verdict(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger=runner.log.name):
        runner._print_dry_run(hz.build_dry_run_plan(EXPERIMENT))
    text = caplog.text
    assert "COMPLETE DAILY CURRENT-PERIOD MOSAICS ALREADY EXIST LOCALLY:" in text
    assert "exact required dates" in text
    assert "exact source-scene inventory" in text
    assert "planned diagnostic download paths" in text


def test_dry_run_printer_creates_no_file(tmp_path, caplog):
    import logging

    root = hz.diagnostic_output_root(EXPERIMENT)
    existed = root.exists()
    with caplog.at_level(logging.INFO, logger=runner.log.name):
        runner._print_dry_run(hz.build_dry_run_plan(EXPERIMENT))
    assert root.exists() is existed


def test_fetch_command_is_printed_only_when_mosaics_are_missing(caplog):
    import logging

    plan = hz.build_dry_run_plan(EXPERIMENT)
    export_plan = plan["daily_export_plan"]
    with caplog.at_level(logging.INFO, logger=runner.log.name):
        runner._print_dry_run(plan)
    text = caplog.text
    if export_plan["complete_daily_mosaics_present"]:
        assert "later USER-EXECUTED command" not in text
    else:
        assert "later USER-EXECUTED command" in text
        assert "--run" in text
        assert "has performed no Earth Engine operation" in text


# =============================================================================
# Earth Engine initialisation is lazy, isolated and happens exactly once
# =============================================================================
class _InitRecorder:
    """Records the order of `init_gee` / `get_region` / export calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def init_gee(self, *args, **kwargs) -> None:
        self.calls.append("init_gee")

    def get_region(self, ctx):
        self.calls.append("get_region")
        return "REGION"

    def build_image(self, region, window, date):
        self.calls.append(f"build_image:{date}")
        return _FakeImage()

    def export(self, image, out_path, *args, **kwargs):
        self.calls.append(f"export:{Path(out_path).name}")
        _write_raster(Path(out_path), np.zeros((4, 4)), nodata=hz.NODATA_SENTINEL)
        return {"path": str(out_path), "transport": "fake"}

    @property
    def init_count(self) -> int:
        return self.calls.count("init_gee")


class _FakeImage:
    def unmask(self, *args, **kwargs):
        return self


@pytest.fixture()
def ee_recorder(monkeypatch):
    """Patch every Earth Engine touchpoint of the isolated export stage."""
    import core.experiment_context as experiment_context
    import core.gee_utils as gee_utils
    import scripts.run_predictors_only as predictors

    recorder = _InitRecorder()
    monkeypatch.setattr(gee_utils, "init_gee", recorder.init_gee)
    # `build_experiment_context` is pure local configuration and is left real;
    # `get_region` is the call that constructs ee.Geometry objects.
    monkeypatch.setattr(experiment_context, "get_region", recorder.get_region)
    monkeypatch.setattr(predictors, "export_image_direct_or_tiled", recorder.export)
    monkeypatch.setattr(runner, "_build_daily_ee_image", recorder.build_image)
    monkeypatch.setattr(runner.audit, "validate_nodata_mask",
                        lambda path: {"status": "ok"})
    # `grid_signature` is deliberately NOT faked: the recorder writes real
    # GeoTIFFs, so the grid-consistency check runs for real.
    return recorder


@pytest.fixture()
def ee_export_env(fixture, ee_recorder, monkeypatch):
    """`ee_recorder` plus the namespace guard rebased onto the temp tree.

    The REAL `assert_namespace_safe` still runs -- it is simply rooted at the
    fixture's base directory instead of the repository, so the safety check is
    exercised rather than disabled.
    """
    base_dir = fixture["base_dir"]
    real = hz.assert_namespace_safe
    monkeypatch.setattr(
        hz, "assert_namespace_safe",
        lambda paths, experiment_id, base_dir=base_dir: real(
            paths, experiment_id, base_dir),
    )
    return ee_recorder


def test_dry_run_never_calls_init_gee(ee_recorder):
    runner.main(experiment_id=EXPERIMENT, dry_run=True)
    assert ee_recorder.calls == []
    assert ee_recorder.init_count == 0


def test_building_the_export_plan_never_calls_init_gee(ee_recorder):
    hz.build_daily_export_plan(EXPERIMENT)
    hz.build_dry_run_plan(EXPERIMENT)
    hz.build_input_provenance(EXPERIMENT)
    assert ee_recorder.init_count == 0


def test_locally_complete_inventory_never_calls_init_gee(fixture, ee_export_env):
    """All mosaics present -> Earth Engine is not initialised at all."""
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    assert plan["complete_daily_mosaics_present"] is True

    records = runner._export_daily_mosaics(
        EXPERIMENT, fixture["root"], plan, force=False)

    assert ee_export_env.calls == []
    assert ee_export_env.init_count == 0
    assert len(records) == plan["date_count"]
    assert {r["status"] for r in records} == {"reused_existing"}


def test_missing_daily_live_export_calls_init_gee_before_get_region(
        fixture, ee_export_env):
    """The regression: get_region builds ee.Geometry and needs an initialised client."""
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    for item in plan["items"]:
        Path(item["output_path"]).unlink()
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    assert len(plan["missing_locally"]) == plan["date_count"]

    runner._export_daily_mosaics(EXPERIMENT, fixture["root"], plan, force=False)

    assert "init_gee" in ee_export_env.calls
    assert "get_region" in ee_export_env.calls
    assert ee_export_env.calls.index("init_gee") < \
        ee_export_env.calls.index("get_region"), ee_export_env.calls
    first_export = next(i for i, c in enumerate(ee_export_env.calls)
                        if c.startswith("export:"))
    assert ee_export_env.calls.index("init_gee") < first_export


def test_init_gee_is_called_exactly_once_for_many_missing_dates(
        fixture, ee_export_env):
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    for item in plan["items"]:
        Path(item["output_path"]).unlink()
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])

    runner._export_daily_mosaics(EXPERIMENT, fixture["root"], plan, force=False)

    assert ee_export_env.init_count == 1
    assert len([c for c in ee_export_env.calls if c.startswith("export:")]) == \
        plan["date_count"]


def test_a_single_missing_date_still_initialises_once(fixture, ee_export_env):
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    Path(plan["items"][1]["output_path"]).unlink()
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])

    runner._export_daily_mosaics(EXPERIMENT, fixture["root"], plan, force=False)

    assert ee_export_env.init_count == 1


def test_initialisation_failure_stops_before_any_export_or_download(
        fixture, ee_export_env, monkeypatch):
    import core.gee_utils as gee_utils

    def _boom(*args, **kwargs):
        ee_export_env.calls.append("init_gee")
        raise RuntimeError("Earth Engine client library not initialized")

    monkeypatch.setattr(gee_utils, "init_gee", _boom)

    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    for item in plan["items"]:
        Path(item["output_path"]).unlink()
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])

    with pytest.raises(runner.HarmonizationRunnerError) as excinfo:
        runner._export_daily_mosaics(EXPERIMENT, fixture["root"], plan, force=False)

    message = str(excinfo.value)
    assert "GEE initialization failed" in message
    assert "GEE_PROJECT" in message
    assert "NO daily mosaic export occurred" in message
    assert "Earth Engine client library not initialized" in message
    assert excinfo.value.__cause__ is not None

    # Nothing downloaded, nothing written, and get_region never reached.
    assert "get_region" not in ee_export_env.calls
    assert not any(c.startswith("export:") for c in ee_export_env.calls)
    for item in plan["items"]:
        assert not Path(item["output_path"]).exists()


def test_the_runner_never_calls_ee_authenticate():
    """The word may appear in the prohibition; the CALL may never appear."""
    assert "Authenticate" not in _called_names(runner)
    for node in ast.walk(ast.parse(_module_source(runner))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "Authenticate"


def test_init_gee_is_imported_lazily_not_at_module_import():
    """`core.gee_utils` imports `ee` at module scope, so it must stay local."""
    for node in ast.walk(ast.parse(_module_source(runner))):
        if isinstance(node, ast.ImportFrom) and node.col_offset == 0:
            assert node.module != "core.gee_utils"
        if isinstance(node, ast.Import) and node.col_offset == 0:
            assert all(a.name != "core.gee_utils" for a in node.names)


def test_init_gee_has_exactly_one_call_site():
    tree = ast.parse(_module_source(runner))
    callers = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(inner, ast.Call)
                and getattr(inner.func, "id", "") == "init_gee"
                for inner in ast.walk(node))
    ]
    assert callers == ["_initialise_earth_engine"]

    invokers = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(inner, ast.Call)
                and getattr(inner.func, "id", "") == "_initialise_earth_engine"
                for inner in ast.walk(node))
    ]
    assert invokers == ["_export_daily_mosaics"]


def test_local_only_stages_never_initialise_earth_engine():
    """No analysis stage may reach the initialiser."""
    tree = ast.parse(_module_source(runner))
    local_stages = {"_analyse", "_finalise", "_print_dry_run", "main",
                    "validate_modes", "_verify_daily_inventory"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in local_stages:
            names = {getattr(inner.func, "id", "")
                     for inner in ast.walk(node) if isinstance(inner, ast.Call)}
            assert "init_gee" not in names
            assert "_initialise_earth_engine" not in names


def test_pending_export_items_drives_initialisation(fixture):
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    assert runner._pending_export_items(plan, force=False) == []
    # --force re-exports everything, so it must initialise.
    assert len(runner._pending_export_items(plan, force=True)) == plan["date_count"]

    Path(plan["items"][0]["output_path"]).unlink()
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    pending = runner._pending_export_items(plan, force=False)
    assert [p["acquisition_date"] for p in pending] == plan["missing_locally"]


def test_tests_never_reach_the_real_ee_initialize(fixture, ee_export_env, monkeypatch):
    """Belt and braces: a real `ee.Initialize` would fail the test outright."""
    import ee

    def _forbidden(*args, **kwargs):
        raise AssertionError("ee.Initialize was reached from the test suite")

    monkeypatch.setattr(ee, "Initialize", _forbidden)

    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    for item in plan["items"]:
        Path(item["output_path"]).unlink()
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])

    runner._export_daily_mosaics(EXPERIMENT, fixture["root"], plan, force=False)
    assert ee_export_env.init_count == 1


# =============================================================================
# Canonical raster-change schema (JSON / CSV / Markdown must agree)
# =============================================================================
def test_producer_defines_the_canonical_raster_change_schema():
    """`RasterChangeAccumulator.report()` is the authority; the rest follows it."""
    produced = tuple(hz.RasterChangeAccumulator(hz.TARGET_LST).report())
    assert produced == hz.RASTER_CHANGE_FIELDS
    hz._assert_producer_matches_schema()          # import-time contract


def test_the_canonical_mean_key_is_mean_signed_difference():
    """The producer's own name is used; no alias is introduced."""
    row = hz.RasterChangeAccumulator(hz.TARGET_LST).report()
    assert "mean_signed_difference" in row
    for alias in ("mean_difference", "candidate_minus_reference_mean",
                  "mean_signed_diff", "signed_mean"):
        assert alias not in row
        assert alias not in hz.RASTER_CHANGE_FIELDS
        assert alias not in hz.RASTER_CHANGE_COLUMNS
        assert alias not in hz.empty_raster_change_report(
            hz.TARGET_LST, reason="x")
    # No alias is read by any renderer or table writer either.
    for name in ("render_summary_markdown", "raster_change_rows",
                 "raster_change_columns"):
        source = inspect.getsource(getattr(hz, name))
        assert "candidate_minus_reference_mean" not in source
        assert "mean_difference" not in source.replace(
            "mean_signed_difference", "")


def test_mean_signed_difference_is_candidate_minus_reference():
    """The semantic meaning is preserved, not re-derived."""
    accumulator = hz.RasterChangeAccumulator(hz.TARGET_LST)
    reference = np.array([[10.0, 20.0]])
    candidate = np.array([[11.0, 23.0]])
    accumulator.add(reference, candidate)
    row = accumulator.report()
    assert row["mean_signed_difference"] == pytest.approx(2.0)
    assert row["computed"] is True
    assert row["not_computed_reason"] is None


def test_json_csv_and_markdown_use_one_schema():
    changes = OrderedDict()
    for product in hz.TARGET_PRODUCTS:
        accumulator = hz.RasterChangeAccumulator(product)
        accumulator.add(np.array([[1.0, 2.0]]), np.array([[1.5, 2.5]]))
        changes[product] = accumulator.report()
    for column in hz.RASTER_CHANGE_COLUMNS:
        assert column in hz.RASTER_CHANGE_FIELDS
    rows = hz.raster_change_rows(changes)
    assert len(rows) == len(hz.TARGET_PRODUCTS)
    for row in rows:
        for column in hz.RASTER_CHANGE_COLUMNS:
            assert column in row
    # The Markdown renderer indexes these directly; all must be present.
    for renderer_key in ("mean_signed_difference", "median_signed_difference",
                         "mae", "rmse", "p95_absolute_difference",
                         "p99_absolute_difference", "valid_mask_agreement",
                         "fraction_above"):
        assert renderer_key in hz.RASTER_CHANGE_FIELDS


def test_csv_columns_survive_a_not_computed_row(tmp_path):
    changes = OrderedDict(
        (product, hz.empty_raster_change_report(product, reason="gate stopped"))
        for product in hz.TARGET_PRODUCTS)
    path = hz.write_csv(tmp_path / "raster_change_summary.csv",
                        hz.raster_change_rows(changes),
                        hz.raster_change_columns(changes))
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in hz.RASTER_CHANGE_COLUMNS:
        assert column in header


# =============================================================================
# Not-computed rows carry the schema without inventing values
# =============================================================================
def test_empty_raster_change_report_carries_the_full_schema():
    row = hz.empty_raster_change_report(hz.TARGET_ANOMALY, reason="graph split")
    assert tuple(row) == hz.RASTER_CHANGE_FIELDS
    assert row["product"] == hz.TARGET_ANOMALY
    assert row["units"] == hz.PRODUCT_UNITS[hz.TARGET_ANOMALY]
    assert row["computed"] is False
    assert "graph split" in row["not_computed_reason"]


@pytest.mark.parametrize("field", hz.RASTER_CHANGE_NUMERIC_FIELDS)
def test_not_computed_metrics_are_none_never_zero_or_nan(field):
    row = hz.empty_raster_change_report(hz.TARGET_CMB, reason="gate stopped")
    assert row[field] is None, "an absent measurement must not become a number"
    assert row[field] is not np.nan
    assert row[field] != 0


def test_not_computed_row_says_it_is_an_absence_of_measurement():
    row = hz.empty_raster_change_report(hz.TARGET_LST, reason="reproduction failed")
    assert "NOT COMPUTED" in row["interpretation"]
    assert "not a measured zero" in row["interpretation"]


def test_empty_raster_change_report_rejects_an_unknown_product():
    with pytest.raises(hz.ReportSchemaError):
        hz.empty_raster_change_report("not_a_product", reason="x")


# =============================================================================
# Schema validator error messages
# =============================================================================
def test_missing_key_error_names_section_product_missing_and_available():
    broken = OrderedDict()
    for product in hz.TARGET_PRODUCTS:
        accumulator = hz.RasterChangeAccumulator(product)
        accumulator.add(np.array([[1.0, 2.0]]), np.array([[1.5, 2.5]]))
        broken[product] = accumulator.report()
    del broken[hz.TARGET_CMB]["mean_signed_difference"]
    del broken[hz.TARGET_CMB]["rmse"]

    with pytest.raises(hz.ReportSchemaError) as excinfo:
        hz.validate_raster_change_rows(broken, section="raster_changes")

    message = str(excinfo.value)
    assert "raster_changes" in message                       # section
    assert hz.TARGET_CMB in message                          # product
    assert "mean_signed_difference" in message               # missing key
    assert "rmse" in message                                 # missing key
    assert "mae" in message                                  # available keys
    assert "RasterChangeAccumulator.report()" in message     # how to fix


def test_validator_reports_a_missing_product():
    accumulator = hz.RasterChangeAccumulator(hz.TARGET_LST)
    accumulator.add(np.array([[1.0]]), np.array([[1.5]]))
    changes = {hz.TARGET_LST: accumulator.report()}
    with pytest.raises(hz.ReportSchemaError) as excinfo:
        hz.validate_raster_change_rows(changes, section="raster_changes")
    message = str(excinfo.value)
    assert hz.TARGET_CMB in message and hz.TARGET_ANOMALY in message
    assert hz.TARGET_LST in message                           # available


def test_validator_rejects_a_computed_row_with_a_blank_metric():
    changes = OrderedDict()
    for product in hz.TARGET_PRODUCTS:
        accumulator = hz.RasterChangeAccumulator(product)
        accumulator.add(np.array([[1.0, 2.0]]), np.array([[1.5, 2.5]]))
        changes[product] = accumulator.report()
    changes[hz.TARGET_LST]["mae"] = None
    changes[hz.TARGET_LST]["computed"] = True
    with pytest.raises(hz.ReportSchemaError) as excinfo:
        hz.validate_raster_change_rows(changes, section="raster_changes")
    assert "marked computed" in str(excinfo.value)
    assert "never" in str(excinfo.value)


@pytest.mark.parametrize("bad", [None, [], "rows", 42])
def test_validator_rejects_a_non_mapping(bad):
    with pytest.raises(hz.ReportSchemaError):
        hz.validate_raster_change_rows(bad, section="raster_changes")


def test_validator_accepts_computed_and_not_computed_rows():
    computed = OrderedDict()
    for product in hz.TARGET_PRODUCTS:
        accumulator = hz.RasterChangeAccumulator(product)
        accumulator.add(np.array([[1.0, 2.0]]), np.array([[1.5, 2.5]]))
        computed[product] = accumulator.report()
    assert hz.validate_raster_change_rows(
        computed, section="raster_changes")["status"] == "pass"

    stopped = OrderedDict(
        (product, hz.empty_raster_change_report(product, reason="gate"))
        for product in hz.TARGET_PRODUCTS)
    report = hz.validate_raster_change_rows(stopped, section="raster_changes")
    assert report["status"] == "pass"
    assert set(report["computed"].values()) == {False}


# =============================================================================
# Normalisation happens once, and never rewrites a computed value
# =============================================================================
def test_normalise_builds_not_computed_rows_from_none():
    changes = hz.normalise_raster_changes(
        None, section="raster_changes", reason="reference reproduction failed")
    assert set(changes) == set(hz.TARGET_PRODUCTS)
    for product, row in changes.items():
        assert tuple(row) == hz.RASTER_CHANGE_FIELDS
        assert row["computed"] is False
        assert "reference reproduction failed" in row["not_computed_reason"]


def test_normalise_requires_a_reason_for_absent_changes():
    with pytest.raises(hz.ReportSchemaError) as excinfo:
        hz.normalise_raster_changes(None, section="raster_changes")
    assert "reason" in str(excinfo.value)


def test_normalise_passes_computed_rows_through_untouched():
    original = OrderedDict()
    for product in hz.TARGET_PRODUCTS:
        accumulator = hz.RasterChangeAccumulator(product)
        accumulator.add(np.array([[10.0, 20.0]]), np.array([[11.0, 23.0]]))
        original[product] = accumulator.report()
    snapshot = json.loads(json.dumps(original, default=str))

    normalised = hz.normalise_raster_changes(original, section="raster_changes")

    for product in hz.TARGET_PRODUCTS:
        assert normalised[product] is original[product]     # identity, not a copy
        assert normalised[product]["mean_signed_difference"] == pytest.approx(2.0)
    assert json.loads(json.dumps(original, default=str)) == snapshot


def test_the_runner_normalises_once_rather_than_scattering_get_fallbacks():
    renderer = inspect.getsource(hz.render_summary_markdown)
    assert "row['mean_signed_difference']" in renderer or \
        'row["mean_signed_difference"]' in renderer
    assert ".get('mean_signed_difference'" not in renderer
    assert '.get("mean_signed_difference"' not in renderer
    finalise = inspect.getsource(runner._finalise)
    assert "normalise_raster_changes" in finalise


# =============================================================================
# Markdown renders from a realistic COMPLETED and gate-stopped summary
# =============================================================================
def _summary_with_changes(changes, decision_evidence=None) -> dict:
    summary = _minimal_summary()
    summary["raster_changes"] = changes
    if decision_evidence is not None:
        summary["decision"] = hz.decide_final_status(decision_evidence)
        summary["final_status"] = summary["decision"]["final_status"]
        summary["final_status_meaning"] = summary["decision"]["final_status_meaning"]
    return summary


def test_markdown_renders_from_a_realistic_completed_summary():
    changes = OrderedDict()
    for product in hz.TARGET_PRODUCTS:
        accumulator = hz.RasterChangeAccumulator(product)
        rng = np.random.default_rng(7)
        reference = rng.normal(28.0, 3.0, size=(64, 64))
        accumulator.add(reference, reference + 0.15)
        changes[product] = accumulator.report()
    summary = _summary_with_changes(changes)

    hz.validate_raster_change_rows(summary["raster_changes"],
                                   section="harmonization_summary.raster_changes")
    markdown = hz.render_summary_markdown(summary)

    assert "## 6. Raster changes" in markdown
    assert "0.1500" in markdown
    for product in hz.TARGET_PRODUCTS:
        assert f"`{product}`" in markdown


def test_markdown_renders_when_a_gate_stopped_the_run():
    """The exact failure mode: reproduction failed, so no candidate exists."""
    changes = hz.normalise_raster_changes(
        None, section="raster_changes",
        reason="an ordered gate stopped the experiment before a candidate "
               "composite existed: reference reproduction failed")
    summary = _summary_with_changes(
        changes,
        decision_evidence=_valid_evidence(
            reference_reproduction_passes=False,
            reference_reproduction_failures=["current_lst_celsius: not reproduced"]))

    markdown = hz.render_summary_markdown(summary)

    assert summary["final_status"] == hz.STATUS_INVALID_REFERENCE
    assert "## 6. Raster changes" in markdown
    assert "n/a" in markdown
    assert "## 12. Next experiment" in markdown


def test_markdown_rendering_never_raises_a_bare_keyerror():
    broken = OrderedDict()
    for product in hz.TARGET_PRODUCTS:
        accumulator = hz.RasterChangeAccumulator(product)
        accumulator.add(np.array([[1.0, 2.0]]), np.array([[1.5, 2.5]]))
        broken[product] = accumulator.report()
    del broken[hz.TARGET_LST]["mean_signed_difference"]
    summary = _summary_with_changes(broken)

    with pytest.raises(hz.ReportSchemaError):
        hz.validate_raster_change_rows(
            summary["raster_changes"], section="harmonization_summary.raster_changes")
    with pytest.raises(KeyError):
        hz.render_summary_markdown(summary)     # validator is the guard, not luck


def test_report_generation_still_preserves_metrics_with_not_computed_rows():
    changes = hz.normalise_raster_changes(
        None, section="raster_changes", reason="graph disconnected")
    summary = _summary_with_changes(changes)
    before = json.loads(json.dumps(summary, default=str))
    hz.render_summary_markdown(summary)
    after = json.loads(json.dumps(summary, default=str))
    assert hz.report_generation_preserves_metrics(before, after)


# =============================================================================
# Resume after a report-only failure
# =============================================================================
def test_resume_reuses_the_hash_validated_reproduction_verdict(tmp_path):
    root = tmp_path
    reproduction_path = root / "reference_reproduction.json"
    raster = _write_raster(root / "rasters" / "reference_current_lst_celsius.tif",
                           np.zeros((4, 4)))
    daily = _write_raster(root / "daily" / "reference" / "d.tif", np.zeros((4, 4)))
    payload = {"passes": False, "failures": ["current_lst_celsius: not reproduced"],
               "outputs": {"current_lst_celsius": str(raster)},
               "versions": dict(hz.DAILY_CONTRACT_VERSIONS),
               "daily_raster_hashes": {
                   DATES[0]: hz.sha256_and_size(daily)["sha256"]}}
    hz.write_json_atomic(reproduction_path, payload)
    hz.write_checkpoint_stage(root, "reference_reproduction", [reproduction_path],
                              {"passes": False})

    reused = runner._reuse_reference_reproduction(
        root, reproduction_path, resume=True,
        daily_paths=[daily], dates=[DATES[0]])
    assert reused is not None
    assert reused["passes"] is False
    assert reused["failures"] == payload["failures"]


def test_resume_recomputes_when_the_checkpoint_hash_no_longer_matches(tmp_path):
    root = tmp_path
    reproduction_path = root / "reference_reproduction.json"
    hz.write_json_atomic(reproduction_path, {"passes": True, "outputs": {}})
    hz.write_checkpoint_stage(root, "reference_reproduction", [reproduction_path],
                              {"passes": True})
    hz.write_json_atomic(reproduction_path, {"passes": False, "outputs": {}})

    assert runner._reuse_reference_reproduction(root, reproduction_path,
                                                resume=True) is None


def test_resume_recomputes_when_a_reference_raster_disappeared(tmp_path):
    root = tmp_path
    reproduction_path = root / "reference_reproduction.json"
    payload = {"passes": True, "failures": [],
               "outputs": {"current_lst_celsius": str(root / "gone.tif")}}
    hz.write_json_atomic(reproduction_path, payload)
    hz.write_checkpoint_stage(root, "reference_reproduction", [reproduction_path],
                              {"passes": True})

    assert runner._reuse_reference_reproduction(root, reproduction_path,
                                                resume=True) is None


def test_without_resume_the_reproduction_is_always_recomputed(tmp_path):
    root = tmp_path
    reproduction_path = root / "reference_reproduction.json"
    hz.write_json_atomic(reproduction_path, {"passes": True, "outputs": {}})
    hz.write_checkpoint_stage(root, "reference_reproduction", [reproduction_path],
                              {"passes": True})

    assert runner._reuse_reference_reproduction(root, reproduction_path,
                                                resume=False) is None


def test_resume_after_a_report_only_failure_reruns_no_export_and_no_science(
        fixture, ee_export_env, monkeypatch):
    """The whole point: a Markdown crash must not cost seven exports and a re-read."""
    root = fixture["root"]
    reproduction_path = root / "reference_reproduction.json"

    # Stage 3 completed on the earlier run and is checkpointed.
    real = hz.run_reference_reproduction(
        EXPERIMENT, root, fixture["daily_paths"], list(DATES),
        base_dir=fixture["base_dir"])
    hz.write_json_atomic(reproduction_path, real)
    hz.write_checkpoint_stage(root, "reference_reproduction", [reproduction_path],
                              {"passes": real["passes"]})

    # Any re-run of the scientific stages is a hard failure.
    def _forbidden(*args, **kwargs):
        raise AssertionError("scientific analysis was recomputed on resume")

    monkeypatch.setattr(hz, "run_reference_reproduction", _forbidden)
    monkeypatch.setattr(hz, "run_overlap_evidence", _forbidden)
    monkeypatch.setattr(hz, "run_harmonisation", _forbidden)
    monkeypatch.setattr(hz, "run_boundary_analysis", _forbidden)

    reused = runner._reuse_reference_reproduction(root, reproduction_path,
                                                  resume=True)
    assert reused is not None
    assert reused["passes"] == real["passes"]

    # And no export ran: every daily mosaic is present, so EE is never touched.
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    records = runner._export_daily_mosaics(EXPERIMENT, root, plan, force=False)
    assert ee_export_env.init_count == 0
    assert {r["status"] for r in records} == {"reused_existing"}


def test_an_empty_common_mask_is_reported_as_not_computed():
    """Zero common pixels measured nothing; it must not read as a computed row."""
    row = hz.RasterChangeAccumulator(hz.TARGET_LST).report()
    assert row["common_valid_pixels"] == 0
    assert row["computed"] is False
    assert "common mask is empty" in row["not_computed_reason"]
    assert row["mean_signed_difference"] is None
    # ... and the validator accepts it rather than demanding invented numbers.
    changes = OrderedDict(
        (product, hz.RasterChangeAccumulator(product).report())
        for product in hz.TARGET_PRODUCTS)
    assert hz.validate_raster_change_rows(
        changes, section="raster_changes")["status"] == "pass"


def test_failed_reference_gate_payload_renders(tmp_path):
    """Regression: a FAILED reference-reproduction gate must still render.

    When the gate fails, the experiment stops before any candidate composite
    exists, so `raster_changes` is None. Markdown rendering used to crash on
    exactly that shape. The failure list is constructed here rather than read
    from the frozen run: the gate's verdict on disk is a moving scientific
    result (it currently PASSES, after the reference composite was
    reproduced), whereas the rendering contract this guards is permanent.
    """
    failures = [
        "grid_contract mismatch: transform differs from the reference composite",
        "valid_mask mismatch: 12 cells differ from the reference composite",
    ]
    changes = hz.normalise_raster_changes(
        None, section="raster_changes",
        reason="an ordered gate stopped the experiment before a candidate "
               "composite existed: reference reproduction failed")
    summary = _summary_with_changes(
        changes,
        decision_evidence=_valid_evidence(
            reference_reproduction_passes=False,
            reference_reproduction_failures=failures))

    hz.validate_raster_change_rows(
        summary["raster_changes"], section="harmonization_summary.raster_changes")
    markdown = hz.render_summary_markdown(summary)

    assert summary["final_status"] == hz.STATUS_INVALID_REFERENCE
    assert "## 6. Raster changes" in markdown
    for failure in failures:
        assert failure in markdown


def test_frozen_reference_reproduction_verdict_is_self_consistent():
    """Whatever the frozen gate currently says, it must be internally coherent.

    This is the part that legitimately depends on the real run: a passing
    gate must list no failures, and a failing one must say why. It asserts
    consistency, not a particular verdict, so a later re-run cannot turn a
    scientific result into a red test.
    """
    frozen = (PROJECT_ROOT / "outputs" / "diagnostics"
              / hz.DIAGNOSTIC_NAMESPACE / EXPERIMENT / "reference_reproduction.json")
    if not frozen.exists():
        pytest.skip("no completed run present in this checkout")
    reproduction = json.loads(frozen.read_text(encoding="utf-8"))
    assert isinstance(reproduction["passes"], bool)
    assert isinstance(reproduction["failures"], list)
    if reproduction["passes"]:
        assert reproduction["failures"] == []
    else:
        assert reproduction["failures"], "a failed gate must record its failures"


# =============================================================================
# Daily-raster validity contract
# =============================================================================
def _daily_set(tmp_path, stacks, dates=DATES, nodata=hz.NODATA_SENTINEL):
    paths = []
    for date, layer in zip(dates, stacks):
        paths.append(_write_raster(tmp_path / f"daily_{date}.tif", layer,
                                   nodata=nodata))
    return paths


def test_nodata_sentinel_is_excluded_from_the_median(tmp_path):
    layers = [np.full((8, 8), 30.0), np.full((8, 8), hz.NODATA_SENTINEL),
              np.full((8, 8), 32.0)]
    paths = _daily_set(tmp_path, layers, dates=DATES)
    stack = hz.read_daily_stack(paths, 0, 8)
    assert np.isnan(stack[1]).all()
    median, count = hz.nanmedian_over_dates(stack)
    assert count[0, 0] == 2
    assert median[0, 0] == pytest.approx(31.0)
    assert not (stack == hz.NODATA_SENTINEL).any()


def test_a_legitimate_zero_is_not_treated_as_nodata(tmp_path):
    """0.0 C on ONE date is real data and must survive."""
    layers = [np.zeros((8, 8)), np.full((8, 8), 30.0), np.full((8, 8), 32.0)]
    paths = _daily_set(tmp_path, layers)
    stack = hz.read_daily_stack(paths, 0, 8)
    median, count = hz.nanmedian_over_dates(stack)
    assert count[0, 0] == 3
    assert median[0, 0] == pytest.approx(30.0)

    report = hz.validate_daily_raster_contract(paths, list(DATES), height=8, width=8)
    assert report["passes"] is True, report["failures"]
    assert report["constant_fill"]["flagged_pixels"] == 0


def test_constant_fill_across_dates_is_detected(tmp_path):
    """The real defect: every date bit-identical at the same pixels."""
    layers = [np.full((8, 8), 30.0 + i) for i in range(3)]
    for layer in layers:
        layer[-1, :] = 0.0                      # zero-padded bottom edge
    paths = _daily_set(tmp_path, layers)

    report = hz.validate_daily_raster_contract(paths, list(DATES), height=8, width=8)

    assert report["passes"] is False
    assert report["root_cause"] == hz.ROOT_CAUSE_CONSTANT_FILL
    assert report["constant_fill"]["flagged_pixels"] == 8
    assert report["constant_fill"]["values"]["0"] == 8
    assert report["constant_fill"]["geometry"]["confined_to_edge_band"] is True
    with pytest.raises(hz.DailyRasterContractError) as excinfo:
        hz.assert_daily_raster_contract(report)
    assert "re-export" in str(excinfo.value).lower()
    assert "cannot be repaired locally" in str(excinfo.value)


def test_constant_fill_detection_is_not_zero_specific(tmp_path):
    """Any repeated fill value is caught, not just zero."""
    layers = [np.full((8, 8), 30.0 + i) for i in range(3)]
    for layer in layers:
        layer[0, :] = -3.5
    paths = _daily_set(tmp_path, layers)
    report = hz.validate_daily_raster_contract(paths, list(DATES), height=8, width=8)
    assert report["passes"] is False
    assert report["constant_fill"]["values"]["-3.5"] == 8


def test_a_nodata_tag_mismatch_is_reported(tmp_path):
    layers = [np.full((8, 8), 30.0 + i) for i in range(3)]
    paths = _daily_set(tmp_path, layers, nodata=-32768.0)
    report = hz.validate_daily_raster_contract(paths, list(DATES), height=8, width=8)
    assert report["passes"] is False
    assert report["root_cause"] == hz.ROOT_CAUSE_NODATA_TAG
    assert any("-32768" in f for f in report["failures"])


def test_contract_passes_on_healthy_dailies(tmp_path):
    rng = np.random.default_rng(3)
    layers = [rng.normal(30.0, 4.0, size=(16, 16)) for _ in range(3)]
    for layer in layers:
        layer[0, 0] = hz.NODATA_SENTINEL
    paths = _daily_set(tmp_path, layers)
    report = hz.validate_daily_raster_contract(paths, list(DATES), height=16, width=16)
    assert report["passes"] is True
    assert report["root_cause"] == hz.ROOT_CAUSE_OK
    assert report["versions"] == dict(hz.DAILY_CONTRACT_VERSIONS)


def test_same_day_union_semantics_one_valid_one_invalid():
    """A date is valid where ANY eligible same-day scene is valid."""
    scene_a = np.array([[30.0, np.nan], [np.nan, np.nan]])
    scene_b = np.array([[32.0, 28.0], [np.nan, np.nan]])
    union = np.isfinite(scene_a) | np.isfinite(scene_b)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        daily = np.nanmedian(np.stack([scene_a, scene_b]), axis=0)

    assert union.tolist() == [[True, True], [False, False]]
    assert daily[0, 0] == pytest.approx(31.0)      # both valid -> same-day median
    assert daily[0, 1] == pytest.approx(28.0)      # only one valid -> that one
    assert np.isnan(daily[1, 0])                   # neither valid -> stays NaN
    assert int(np.isfinite(daily).sum()) == int(union.sum())
    # One temporal observation regardless of how many same-day scenes existed.
    _, count = hz.nanmedian_over_dates(daily[None, :, :])
    assert count.max() == 1


def test_canonical_daily_reducer_is_reused_not_reimplemented():
    """The daily mosaic is built by the frozen counterfactual helpers."""
    source = inspect.getsource(runner._build_daily_ee_image)
    assert "audit._base_filtered_collection" in source
    assert "audit._daily_composite_collection" in source
    assert "audit.LANDSAT_SCALE" in source and "audit.LANDSAT_OFFSET" in source
    assert "apply_qa_mask" not in source          # comes from the canonical helper


# =============================================================================
# Reproduction forensics
# =============================================================================
def _forensics_case(frozen_counts, daily_stack, dates=DATES):
    forensics = hz.ReproductionForensics(list(dates))
    ours = np.isfinite(daily_stack).sum(axis=0).astype("float64")
    ours = np.where(ours > 0, ours, np.nan)
    membership = hz.date_membership_bitmask(daily_stack)
    forensics.add_counts(np.asarray(frozen_counts, dtype="float64"), ours,
                         daily_stack, membership, 0)
    return forensics


def test_per_date_membership_mismatch_is_localised():
    stack = np.full((3, 2, 2), 30.0)
    stack[1, 0, 0] = np.nan                     # date 2 genuinely absent here
    frozen = np.array([[3.0, 3.0], [3.0, 3.0]])
    forensics = _forensics_case(frozen, stack)
    report = forensics.report()

    assert report["count_mismatch_pixels"] == 1
    assert report["false_invalid_pixels"] == 1
    assert report["false_valid_pixels"] == 0
    by_date = {r["acquisition_date"]: r for r in report["mismatch_by_date"]}
    assert by_date[DATES[1]]["invalid_in_reconstruction_at_mismatch"] == 1
    assert by_date[DATES[0]]["valid_in_reconstruction_at_mismatch"] == 1


def test_count_difference_histogram_is_reported():
    stack = np.full((3, 1, 3), 30.0)
    frozen = np.array([[2.0, 1.0, 3.0]])
    report = _forensics_case(frozen, stack).report()
    assert report["count_difference_histogram"] == {"1": 1, "2": 1}


def test_one_sided_all_date_mismatch_is_classified_as_constant_fill():
    """The exact signature of the observed defect."""
    stack = np.full((3, 2, 2), 0.0)             # every date "valid" everywhere
    frozen = np.array([[2.0, 1.0], [2.0, 2.0]])
    report = _forensics_case(frozen, stack).report()
    assert report["false_invalid_pixels"] == 0
    assert report["false_valid_pixels"] == 4
    assert report["classification"]["root_cause"] == hz.ROOT_CAUSE_CONSTANT_FILL
    assert "constant fill" in report["classification"]["explanation"]


def test_two_sided_mismatch_is_classified_as_membership_or_reducer():
    stack = np.full((3, 1, 2), 30.0)
    stack[0, 0, 0] = np.nan
    frozen = np.array([[3.0, 2.0]])
    report = _forensics_case(frozen, stack).report()
    assert report["false_valid_pixels"] and report["false_invalid_pixels"]
    assert report["classification"]["root_cause"] == \
        "daily_membership_or_reducer_mismatch"


def test_sparse_extreme_values_are_reported_not_hidden():
    """A defect on 1 pixel in 10,000 must still appear in the report."""
    forensics = hz.ReproductionForensics(list(DATES))
    frozen = np.full((100, 100), 30.0)
    ours = frozen.copy()
    ours[7, 9] = 0.0                            # one catastrophic pixel
    stack = np.full((3, 100, 100), 30.0)
    stack[:, 7, 9] = 0.0
    forensics.add_values(frozen, ours, stack, hz.date_membership_bitmask(stack), 0)
    report = forensics.report()

    top = report["top_value_discrepancies"]
    assert top, "a sparse extreme must not be summarised away"
    assert top[0]["abs_difference"] == pytest.approx(30.0)
    assert (top[0]["row"], top[0]["col"]) == (7, 9)
    assert top[0]["frozen_value"] == pytest.approx(30.0)
    assert top[0]["reconstructed_value"] == pytest.approx(0.0)
    assert top[0]["dates_valid_in_reconstruction"] == list(DATES)
    assert top[0]["daily_values"] == [0.0, 0.0, 0.0]
    assert "percentile summary alone would hide" in report["sparse_extreme_policy"]


def test_top_discrepancy_list_is_bounded():
    forensics = hz.ReproductionForensics(list(DATES))
    frozen = np.zeros((60, 60))
    ours = np.arange(3600, dtype="float64").reshape(60, 60)
    stack = np.zeros((3, 60, 60))
    for _ in range(3):
        forensics.add_values(frozen, ours, stack,
                             hz.date_membership_bitmask(stack), 0)
    assert len(forensics.report()["top_value_discrepancies"]) == \
        hz.TOP_DISCREPANCY_COUNT


# =============================================================================
# Tolerances are never relaxed
# =============================================================================
def test_reproduction_tolerance_is_never_relaxed():
    assert hz.REPRODUCTION_TOLERANCES[hz.TARGET_LST] == \
        ab.REPRODUCTION_TOLERANCES["current_lst_celsius"] == 1e-4
    assert hz.REPRODUCTION_TOLERANCES[hz.TARGET_CMB] == 1e-4
    assert hz.REPRODUCTION_TOLERANCES[hz.TARGET_ANOMALY] == 1e-4
    assert hz.REPRODUCTION_EXACT_CHECKS == (
        "grid_signature_equality", "valid_mask_equality", "valid_date_count_equality")


def test_no_tolerance_is_widened_anywhere_in_the_fix():
    source = _module_source(hz)
    for widened in ("1e-2", "1e-1", "0.01,", "atol=1", "rtol=0.1"):
        assert f"REPRODUCTION_TOLERANCES = OrderedDict((\n    ({widened}" not in source
    # The exact checks stay exact: no tolerance is applied to counts or masks.
    reproduction = inspect.getsource(hz.run_reference_reproduction)
    assert "isclose" not in reproduction
    assert "atol" not in reproduction


# =============================================================================
# Stale checkpoint invalidation and re-export policy
# =============================================================================
def _stale_payload(daily_paths, dates, **overrides):
    payload = {
        "passes": False,
        "versions": dict(hz.DAILY_CONTRACT_VERSIONS),
        "daily_raster_hashes": {
            date: hz.sha256_and_size(Path(p))["sha256"]
            for date, p in zip(dates, daily_paths)},
        "outputs": {},
    }
    payload.update(overrides)
    return payload


def test_a_fresh_verdict_is_not_considered_stale(fixture):
    payload = _stale_payload(fixture["daily_paths"], list(DATES))
    assert runner._stale_reproduction_reasons(
        payload, fixture["daily_paths"], list(DATES)) == []


@pytest.mark.parametrize("key", list(hz.DAILY_CONTRACT_VERSIONS))
def test_a_version_change_invalidates_the_verdict(fixture, key):
    payload = _stale_payload(fixture["daily_paths"], list(DATES))
    payload["versions"][key] = "0.0-old"
    reasons = runner._stale_reproduction_reasons(
        payload, fixture["daily_paths"], list(DATES))
    assert any(key in reason for reason in reasons)


def test_a_changed_daily_raster_invalidates_the_verdict(fixture):
    payload = _stale_payload(fixture["daily_paths"], list(DATES))
    _write_raster(fixture["daily_paths"][1], np.full((GRID, GRID), 21.0))
    reasons = runner._stale_reproduction_reasons(
        payload, fixture["daily_paths"], list(DATES))
    assert any(DATES[1] in reason and "changed" in reason for reason in reasons)


def test_a_verdict_without_recorded_hashes_is_stale(fixture):
    payload = _stale_payload(fixture["daily_paths"], list(DATES))
    payload.pop("daily_raster_hashes")
    reasons = runner._stale_reproduction_reasons(
        payload, fixture["daily_paths"], list(DATES))
    assert any("hashes" in reason for reason in reasons)


def test_the_stale_failed_checkpoint_is_not_blindly_reused(fixture, tmp_path):
    """The failed verdict from before this fix must never be resumed into."""
    root = fixture["root"]
    reproduction_path = root / "reference_reproduction.json"
    legacy = {"passes": False,
              "failures": ["current_lst_celsius: not reproduced"],
              "outputs": {}}                    # pre-fix: no versions, no hashes
    hz.write_json_atomic(reproduction_path, legacy)
    hz.write_checkpoint_stage(root, "reference_reproduction", [reproduction_path],
                              {"passes": False})

    assert runner._reuse_reference_reproduction(
        root, reproduction_path, resume=True,
        daily_paths=fixture["daily_paths"], dates=list(DATES)) is None


def test_compatible_local_dailies_avoid_earth_engine(fixture, ee_export_env):
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    runner._export_daily_mosaics(EXPERIMENT, fixture["root"], plan, force=False)
    assert ee_export_env.init_count == 0


def test_incompatible_export_semantics_require_re_export(fixture, ee_export_env):
    """--force-daily-export re-exports the seven rasters even when present."""
    plan = hz.build_daily_export_plan(EXPERIMENT, fixture["base_dir"])
    assert plan["complete_daily_mosaics_present"] is True
    assert runner._pending_export_items(plan, force=False) == []
    pending = runner._pending_export_items(plan, force=True)
    assert len(pending) == plan["date_count"]

    runner._export_daily_mosaics(EXPERIMENT, fixture["root"], plan, force=True)
    assert ee_export_env.init_count == 1


def test_force_daily_export_flag_is_wired(tmp_path):
    parser = runner.build_parser()
    args = parser.parse_args(["--experiment", EXPERIMENT, "--run",
                              "--force-daily-export"])
    assert args.force_daily_export is True
    assert "force_daily_export" in inspect.signature(runner.main).parameters
    assert "force_daily_export" in inspect.getsource(runner._run_live)


def test_quarantine_moves_incompatible_dailies_without_deleting(fixture):
    runner._EXPERIMENT_FOR_SAFETY[0] = EXPERIMENT
    root = fixture["root"]
    contract = {"root_cause": hz.ROOT_CAUSE_CONSTANT_FILL,
                "failures": ["3706 pixels"], "versions": dict(hz.DAILY_CONTRACT_VERSIONS),
                "constant_fill": {"flagged_pixels": 3706}}
    real = hz.assert_namespace_safe
    import unittest.mock as mock
    with mock.patch.object(hz, "assert_namespace_safe",
                           lambda paths, exp, base_dir=fixture["base_dir"]:
                           real(paths, exp, base_dir)):
        quarantine = runner._quarantine_incompatible_dailies(
            root, fixture["daily_paths"], contract)

    assert quarantine.exists()
    moved = sorted(quarantine.glob("*.tif"))
    assert len(moved) == len(DATES), "files must be MOVED, never deleted"
    for path in fixture["daily_paths"]:
        assert not Path(path).exists()
    why = json.loads((quarantine / "why_incompatible.json").read_text(encoding="utf-8"))
    assert why["root_cause"] == hz.ROOT_CAUSE_CONSTANT_FILL
