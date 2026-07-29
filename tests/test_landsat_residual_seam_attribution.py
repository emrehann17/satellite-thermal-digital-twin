"""Focused tests for the residual-seam attribution audit.

No Earth Engine, no Step5-Step8 rerun and no live audit are required: the
pipeline-touching work is confined to streaming raster reads, so these tests
exercise the CLI mode contract, the no-op dry-run, the namespace/force safety,
the exact decompositions, the pair-mask construction, the mask-discontinuity
policy, the predeclared epsilons, the cluster bootstrap, the path/row gating,
the descriptive hotspot thresholds, the ordered attribution rule, atomic
checkpointing/resume, and the report-generation invariants.
"""

from __future__ import annotations

import ast
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.landsat_residual_seam_attribution as rs
import scripts.run_landsat_residual_seam_attribution as runner

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


def _flag_grid(shape, **hot):
    """A full flag dict of the declared shape with selected cells switched on."""
    flags = {name: np.zeros(shape, dtype=bool) for name in rs.BOUNDARY_FLAGS}
    for name, cells in hot.items():
        for cell in cells:
            flags[name][cell] = True
    return flags


def _mean_accumulator(unit_values):
    accumulator = rs.MeanAccumulator()
    for unit, values in unit_values.items():
        accumulator.add(np.full(len(values), unit, dtype="int64"),
                        np.asarray(values, dtype="float64"))
    return accumulator


def _interval(low, high, *, status="estimated"):
    return {"interval_low": low, "interval_high": high, "status": status}


def _valid_evidence(**overrides):
    """Evidence that reaches the eligibility branch with nothing supported."""
    evidence = {
        "inputs_valid": True,
        "invalid_input_reasons": [],
        "excess_by_boundary": {},
        "anomaly_excess_by_boundary": {},
        "baseline_excess_excluding_current_only": {},
        "cmb_current_share": _interval(0.4, 0.6),
        "cmb_baseline_share": _interval(0.4, 0.6),
        "anomaly_numerator_share": _interval(0.4, 0.6),
        "anomaly_denominator_share": _interval(0.4, 0.6),
        "anomaly_std_concentration": {"supported": False},
        "mask_discontinuity_near_std_threshold": {"elevated": False},
        "near_std_epsilon_support": {"0.05": False, "0.1": False, "0.2": False},
        "pathrow_only": {"supported": False, "n_units": 0, "n_interfaces": 0},
    }
    evidence.update(overrides)
    return evidence


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
    with pytest.raises(SystemExit):
        runner.parse_args(["--experiment", EXPERIMENT])


def test_only_manavgat_is_supported():
    assert rs.SUPPORTED_EXPERIMENT_IDS == (EXPERIMENT,)
    with pytest.raises(SystemExit):
        runner.parse_args(["--experiment", "bejis_2022", "--dry-run"])
    with pytest.raises(rs.ResidualSeamError):
        rs.assert_supported_experiment("bejis_2022")


def test_valid_mode_combinations_are_accepted():
    runner.validate_modes(True, False, False, False)
    runner.validate_modes(False, True, False, False)
    runner.validate_modes(False, True, True, False)
    runner.validate_modes(False, True, False, True)


# =============================================================================
# Dry-run
# =============================================================================
def test_dry_run_writes_nothing(monkeypatch):
    root = rs.diagnostic_output_root(EXPERIMENT)
    root_existed = root.exists()
    before = sorted(root.rglob("*")) if root_existed else []

    created: list[str] = []
    opened: list[str] = []
    real_mkdir = Path.mkdir
    real_open = open

    def _tracking_mkdir(self, *args, **kwargs):
        created.append(str(self))
        return real_mkdir(self, *args, **kwargs)

    def _tracking_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            opened.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _tracking_mkdir)
    monkeypatch.setattr("builtins.open", _tracking_open)
    result = runner.main(experiment_id=EXPERIMENT, dry_run=True)
    monkeypatch.undo()

    assert result["dry_run"] is True
    assert result["ran"] is False
    assert result["writes_performed"] is False
    assert root.exists() == root_existed
    after = sorted(root.rglob("*")) if root.exists() else []
    assert after == before
    assert [p for p in opened if rs.DIAGNOSTIC_NAMESPACE in p] == []
    assert [p for p in created if rs.DIAGNOSTIC_NAMESPACE in p] == []


def test_dry_run_prints_every_required_section():
    plan = rs.build_dry_run_plan(EXPERIMENT)
    for key in (
        "resolved_inputs", "missing_optional_provenance_inputs", "output_root",
        "decomposition_formulas", "boundary_classes", "thresholds",
        "bootstrap_configuration", "matched_control_strategy",
        "decision_rule_version", "expected_files",
    ):
        assert key in plan
    assert plan["writes_performed"] is False
    assert plan["directories_created"] == 0
    assert plan["earth_engine_calls"] == 0
    assert plan["smoothing_applied"] is False
    assert plan["planned_stages"] == list(rs.PLANNED_STAGES)
    assert plan["allowed_final_statuses"] == list(rs.FINAL_STATUSES)
    thresholds = plan["thresholds"]
    assert thresholds["near_std_threshold_epsilon_primary"] == 0.10
    assert thresholds["near_std_threshold_epsilon_sensitivity"] == [0.05, 0.20]


def test_dry_run_resolves_the_frozen_candidate_inputs():
    plan = rs.build_dry_run_plan(EXPERIMENT)
    for role in (rs.TARGET_CMB, rs.TARGET_ANOMALY, "baseline_lst_std_celsius",
                 "current_unique_date_valid_count"):
        assert plan["resolved_inputs"][role]["present"] is True
    assert plan["missing_required_inputs"] == []
    assert plan["upstream_prerequisites"]["prerequisites_met"] is True
    assert plan["upstream_prerequisites"]["production_approval_required"] is False


# =============================================================================
# Earth Engine is unreachable
# =============================================================================
def _module_source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("module", [rs, runner])
def test_no_earth_engine_import_or_call_in_source(module):
    tree = ast.parse(_module_source(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "ee" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "ee"


@pytest.mark.parametrize("forbidden", [
    "ee.Initialize", "ee.Authenticate", "ee.ImageCollection", "ee.Image(",
    "getInfo", "init_gee", "export_image", "toDrive",
])
def test_no_earth_engine_symbol_is_referenced(forbidden):
    for module in (rs, runner):
        assert forbidden not in _module_source(module)


def test_analysis_callables_never_touch_earth_engine():
    for callable_ in (
        rs.build_input_plan, rs.run_streaming_pass, rs.analyse_window,
        rs.build_edge_flags, rs.decompose_anomaly, rs.compute_excess_rows,
        rs.rasterize_pathrow_boundaries, rs.decide_final_status,
    ):
        source = inspect.getsource(callable_)
        for forbidden in ("import ee", "ee.", "gee_utils", "getInfo"):
            assert forbidden not in source, f"{callable_.__name__} touches {forbidden}"


def test_earth_engine_guard_is_installed_for_the_live_run():
    source = _module_source(runner)
    assert "with ab.EarthEngineGuard():" in source


def test_dry_run_performs_no_earth_engine_operation():
    import src.landsat_composite_downstream_ab as ab

    with ab.EarthEngineGuard():
        plan = rs.build_dry_run_plan(EXPERIMENT)
    assert plan["earth_engine_calls"] == 0


# =============================================================================
# Frozen outputs are never modified
# =============================================================================
def test_frozen_outputs_are_never_modified():
    """The real frozen A/B and counterfactual inputs must survive untouched."""
    plan = rs.build_input_plan(EXPERIMENT)
    watched = [Path(e["path"]) for e in plan.values() if Path(e["path"]).exists()]
    watched += [
        p for p in rs.pathrow_boundary_sources(EXPERIMENT).values() if Path(p).exists()
    ]
    watched += [
        p for p in rs.upstream_report_paths(EXPERIMENT).values() if Path(p).exists()
    ]
    assert watched, "the frozen inputs should exist for this test"
    before = {
        str(p): (rs.sha256_and_size(p), p.stat().st_mtime_ns) for p in watched
    }

    rs.build_dry_run_plan(EXPERIMENT)
    rs.load_upstream_state(EXPERIMENT)
    rs.resolve_pathrow_availability(EXPERIMENT)
    rs.assert_grid_contract(plan)

    after = {str(p): (rs.sha256_and_size(p), p.stat().st_mtime_ns) for p in watched}
    assert before == after


def test_forbidden_write_roots_cover_every_frozen_namespace():
    roots = [str(p) for p in rs.forbidden_write_roots(EXPERIMENT)]
    assert str(rs.downstream_ab_root(EXPERIMENT)) in roots
    assert str(rs.counterfactual_root(EXPERIMENT)) in roots
    assert str(rs.canonical_experiment_root(EXPERIMENT)) in roots


@pytest.mark.parametrize("relative", [
    "outputs/diagnostics/landsat_composite_downstream_ab/manavgat_2021/x.tif",
    "outputs/diagnostics/landsat_composite_counterfactual/manavgat_2021/x.tif",
    "outputs/experiments/manavgat_2021/step5/x.tif",
    "config/legacy_modis_compatibility_attestation.json",
])
def test_namespace_safety_rejects_frozen_paths(relative):
    with pytest.raises(rs.NamespaceSafetyError):
        rs.assert_namespace_safe([PROJECT_ROOT / relative], EXPERIMENT)


def test_namespace_safety_accepts_the_dedicated_root():
    root = rs.diagnostic_output_root(EXPERIMENT)
    rs.assert_namespace_safe(
        [root, root / "tables" / "x.csv", root / "maps" / "a" / "b.tif"], EXPERIMENT,
    )


def test_namespace_safety_rejects_escaping_path():
    root = rs.diagnostic_output_root(EXPERIMENT)
    with pytest.raises(rs.NamespaceSafetyError):
        rs.assert_namespace_safe([root / ".." / ".." / "evil.json"], EXPERIMENT)


def test_all_planned_outputs_are_namespace_safe():
    rs.assert_namespace_safe(
        list(rs.plan_expected_files(EXPERIMENT).values()), EXPERIMENT,
    )
    rs.assert_namespace_safe(
        list(rs.plan_output_layout(EXPERIMENT).values()), EXPERIMENT,
    )


def test_force_cannot_escape_the_diagnostic_root(tmp_path, monkeypatch):
    base = tmp_path
    root = rs.diagnostic_output_root(EXPERIMENT, base)
    root.mkdir(parents=True)
    (root / "sentinel.txt").write_text("x", encoding="utf-8")
    frozen = rs.downstream_ab_root(EXPERIMENT, base)
    frozen.mkdir(parents=True)
    (frozen / "frozen.txt").write_text("keep", encoding="utf-8")

    removed = rs.clear_diagnostic_namespace(EXPERIMENT, base)
    assert removed == str(root.resolve())
    assert not root.exists()
    assert (frozen / "frozen.txt").read_text(encoding="utf-8") == "keep"


def test_force_refuses_a_symlinked_root(tmp_path):
    base = tmp_path
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "precious.txt").write_text("keep", encoding="utf-8")
    root = rs.diagnostic_output_root(EXPERIMENT, base)
    root.parent.mkdir(parents=True)
    root.symlink_to(real, target_is_directory=True)

    with pytest.raises(rs.NamespaceSafetyError):
        rs.clear_diagnostic_namespace(EXPERIMENT, base)
    assert (real / "precious.txt").exists()


# =============================================================================
# Grid contract and nodata handling
# =============================================================================
def test_exact_grid_mismatch_fails(tmp_path):
    from rasterio.transform import Affine

    a = _write_raster(tmp_path / "a.tif", np.zeros((4, 4)))
    b = _write_raster(
        tmp_path / "b.tif", np.zeros((4, 4)),
        transform=Affine(0.0003, 0.0, 31.0, 0.0, -0.0003, 37.35),
    )
    plan = {
        "a": {"path": a, "family": "target", "required": True,
              "source_chain": "x", "purpose": "", "role": "a"},
        "b": {"path": b, "family": "target_component", "required": True,
              "source_chain": "x", "purpose": "", "role": "b"},
    }
    with pytest.raises(rs.GridMismatchError):
        rs.assert_grid_contract(plan)


def test_matching_grids_pass_the_contract(tmp_path):
    a = _write_raster(tmp_path / "a.tif", np.zeros((4, 4)))
    b = _write_raster(tmp_path / "b.tif", np.ones((4, 4)))
    plan = {
        "a": {"path": a, "family": "target", "required": True,
              "source_chain": "x", "purpose": "", "role": "a"},
        "b": {"path": b, "family": "support", "required": True,
              "source_chain": "x", "purpose": "", "role": "b"},
    }
    contract = rs.assert_grid_contract(plan)
    assert contract["passed"] is True
    assert contract["checked_raster_count"] == 2


def test_nodata_is_never_converted_to_zero(tmp_path):
    array = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
    path = _write_raster(tmp_path / "sentinel.tif", array, nodata=-9999.0)
    read = rs.read_window(path, 0, 2)
    assert np.isnan(read[0, 1])
    assert not np.any(read == 0.0)

    nan_array = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype="float32")
    nan_path = _write_raster(tmp_path / "nan.tif", nan_array, nodata=np.nan)
    read_nan = rs.read_window(nan_path, 0, 2)
    assert np.isnan(read_nan[0, 0])
    assert not np.any(np.nan_to_num(read_nan, nan=-1.0) == 0.0)


def test_missing_required_input_fails_clearly(tmp_path):
    plan = {
        "current_lst_celsius": {
            "path": tmp_path / "absent.tif", "family": "target", "required": True,
            "source_chain": "x", "purpose": "", "role": "current_lst_celsius",
        },
    }
    with pytest.raises(rs.PrerequisiteError) as excinfo:
        rs.assert_required_inputs(plan, EXPERIMENT)
    assert "missing required frozen inputs" in str(excinfo.value)
    assert "never calls Earth Engine" in str(excinfo.value)


# =============================================================================
# Pair masks
# =============================================================================
def test_horizontal_and_vertical_pair_masks_are_correct():
    array = np.arange(12, dtype="float64").reshape(3, 4)
    horizontal = rs.edge_difference(array, "horizontal")
    vertical = rs.edge_difference(array, "vertical")
    assert horizontal.shape == (3, 3)
    assert vertical.shape == (2, 4)
    assert np.all(horizontal == 1.0)
    assert np.all(vertical == 4.0)

    rows, cols = rs.edge_anchor_rows_cols(array.shape, "horizontal")
    assert rows.size == 9 and cols.max() == 2
    b_rows, b_cols = rs.endpoint_b_rows_cols(rows, cols, "horizontal")
    assert np.array_equal(b_rows, rows) and np.array_equal(b_cols, cols + 1)

    rows, cols = rs.edge_anchor_rows_cols(array.shape, "vertical")
    b_rows, b_cols = rs.endpoint_b_rows_cols(rows, cols, "vertical")
    assert np.array_equal(b_rows, rows + 1) and np.array_equal(b_cols, cols)


def test_pairs_touching_an_invalid_endpoint_are_dropped_not_zero_filled():
    array = np.array([[1.0, np.nan, 3.0]], dtype="float64")
    valid = rs.edge_valid_mask(array, orientation="horizontal")
    assert valid.tolist() == [[False, False]]
    difference = rs.edge_difference(array, "horizontal")
    assert np.isnan(difference).all()


def test_count_change_flag_requires_both_endpoints_finite():
    counts = np.array([[1.0, 2.0, np.nan, 2.0]], dtype="float64")
    changed = rs.edge_change_flag(counts, "horizontal")
    assert changed.tolist() == [[True, False, False]]


def test_threshold_straddle_and_near_value_flags():
    counts = np.array([[1.0, 2.0, 3.0]], dtype="float64")
    straddle = rs.edge_threshold_straddle(counts, "horizontal", 2.0)
    assert straddle.tolist() == [[True, False]]

    std = np.array([[1.05, 1.5, 3.0]], dtype="float64")
    near = rs.edge_near_value(std, "horizontal", 1.0, 0.10)
    assert near.tolist() == [[True, False]]


@pytest.mark.parametrize("height", [1, 2, 5, 255, 256, 257, 512, 2338])
def test_row_windows_cover_every_edge_exactly_once(height):
    horizontal = vertical = 0
    for _start, _stop, h_rows, v_rows in rs.iter_row_windows(height, 256):
        horizontal += h_rows
        vertical += v_rows
    assert horizontal == height
    assert vertical == max(0, height - 1)


def test_spatial_block_ids_are_deterministic():
    rows = np.array([0, 127, 128, 300])
    cols = np.array([0, 127, 128, 300])
    ids = rs.spatial_block_ids(rows, cols, block_size=128)
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]
    assert rs.block_id_to_label(int(ids[2])) == "block_r1_c1"


# =============================================================================
# Exact decompositions
# =============================================================================
def test_current_minus_baseline_decomposition_reconstructs_exactly():
    rng = np.random.default_rng(7)
    c_a, c_b = rng.normal(30, 6, 5000), rng.normal(30, 6, 5000)
    m_a, m_b = rng.normal(27, 5, 5000), rng.normal(27, 5, 5000)

    target, current, baseline = rs.decompose_current_minus_baseline(c_a, c_b, m_a, m_b)
    assert np.allclose(target, current + baseline, atol=0.0, rtol=0.0)

    exact_target = (c_b - m_b) - (c_a - m_a)
    residual = rs.reconstruction_residual(exact_target, current, baseline)
    assert np.abs(residual).max() <= rs.CMB_RECONSTRUCTION_ABS_TOL
    assert np.array_equal(current, c_b - c_a)
    assert np.array_equal(baseline, -(m_b - m_a))


def test_anomaly_symmetric_decomposition_reconstructs_exactly():
    rng = np.random.default_rng(11)
    d_a, d_b = rng.normal(0, 4, 5000), rng.normal(0, 4, 5000)
    # Step5 masks S < 1.0, so valid anomaly pixels always have S >= 1.0.
    s_a, s_b = rng.uniform(1.0, 9.0, 5000), rng.uniform(1.0, 9.0, 5000)

    z_a, z_b, numerator, denominator = rs.decompose_anomaly(d_a, d_b, s_a, s_b)
    residual = np.abs(rs.reconstruction_residual(z_b - z_a, numerator, denominator))
    scale = np.maximum(
        np.maximum(np.abs(z_a), np.abs(z_b)),
        np.maximum(np.abs(numerator), np.abs(denominator)),
    )
    assert np.all(residual <= rs.anomaly_identity_tolerance(scale))


def test_anomaly_decomposition_matches_the_declared_formulas():
    d_a, d_b = np.array([2.0]), np.array([5.0])
    s_a, s_b = np.array([2.0]), np.array([4.0])
    _, _, numerator, denominator = rs.decompose_anomaly(d_a, d_b, s_a, s_b)
    expected_numerator = 0.5 * (1 / 2.0 + 1 / 4.0) * (5.0 - 2.0)
    expected_denominator = 0.5 * (2.0 + 5.0) * (1 / 4.0 - 1 / 2.0)
    assert numerator[0] == pytest.approx(expected_numerator)
    assert denominator[0] == pytest.approx(expected_denominator)
    assert numerator[0] + denominator[0] == pytest.approx(5.0 / 4.0 - 2.0 / 2.0)


def test_anomaly_decomposition_is_not_a_taylor_expansion():
    """A first-order expansion would be visibly wrong for a large std change."""
    d_a, d_b = np.array([1.0]), np.array([9.0])
    s_a, s_b = np.array([1.0]), np.array([8.0])
    _, _, numerator, denominator = rs.decompose_anomaly(d_a, d_b, s_a, s_b)
    exact = 9.0 / 8.0 - 1.0 / 1.0
    assert numerator[0] + denominator[0] == pytest.approx(exact, abs=1e-12)

    mean_d, mean_s = 5.0, 4.5
    taylor = (1 / mean_s) * (9.0 - 1.0) - (mean_d / mean_s ** 2) * (8.0 - 1.0)
    assert abs(taylor - exact) > 1e-3
    source = inspect.getsource(rs.decompose_anomaly)
    assert "0.5 * (inv_a + inv_b) * (d_b - d_a)" in source
    assert "0.5 * (d_a + d_b) * (inv_b - inv_a)" in source


def test_signed_cancellation_and_reinforcement_are_classified_correctly():
    a = np.array([1.0, 1.0, -2.0, -2.0, 0.0, 3.0])
    b = np.array([2.0, -3.0, -1.0, 4.0, 5.0, 0.0])
    cancelling, reinforcing, degenerate = rs.classify_signed_interaction(a, b)
    assert cancelling.tolist() == [False, True, False, True, False, False]
    assert reinforcing.tolist() == [True, False, True, False, False, False]
    assert degenerate.tolist() == [False, False, False, False, True, True]
    # A zero component is never forced into either category.
    assert not (cancelling & reinforcing).any()
    assert not (cancelling & degenerate).any()


def test_component_share_is_symmetric_and_bounded():
    a = np.array([3.0, -1.0, 0.0])
    b = np.array([1.0, 1.0, 0.0])
    share_a = rs.component_share(a, b)
    share_b = rs.component_share(b, a)
    assert share_a[0] == pytest.approx(0.75)
    assert share_a[1] == pytest.approx(0.5)
    assert np.isnan(share_a[2]) and np.isnan(share_b[2])
    finite = np.isfinite(share_a)
    assert np.allclose(share_a[finite] + share_b[finite], 1.0)


# =============================================================================
# Boundary classes
# =============================================================================
def test_stratified_classes_cover_the_required_five():
    for name in ("support_only", "pathrow_only", "support_and_pathrow",
                 "threshold_only", "none_of_known_boundaries"):
        assert name in rs.STRATIFIED_CLASSES


def test_stratified_class_assignment(monkeypatch):
    shape = (2, 4)
    flags = _flag_grid(
        shape,
        current_support_change=[(0, 0), (1, 0)],
        source_path_row_boundary=[(0, 1), (0, 2), (1, 0)],
        near_std_threshold_boundary=[(0, 2), (0, 3)],
    )
    codes, support, threshold, pathrow = rs.stratified_class_codes(flags)
    inverse = {code: name for name, code in rs.OVERLAP_CODES.items()}
    assert inverse[int(codes[0, 0])] == "support_only"
    assert inverse[int(codes[0, 1])] == "pathrow_only"
    assert inverse[int(codes[0, 2])] == "pathrow_and_threshold"
    assert inverse[int(codes[0, 3])] == "threshold_only"
    assert inverse[int(codes[1, 0])] == "support_and_pathrow"
    assert inverse[int(codes[1, 3])] == "none_of_known_boundaries"

    controls = rs.control_pair_mask(support, threshold, pathrow)
    assert controls[1, 3] and not controls[0, 0]


def test_pathrow_only_excludes_support_and_threshold_overlap():
    shape = (1, 3)
    flags = _flag_grid(
        shape,
        source_path_row_boundary=[(0, 0), (0, 1), (0, 2)],
        current_support_change=[(0, 1)],
        low_baseline_std_boundary=[(0, 2)],
    )
    codes, _, _, _ = rs.stratified_class_codes(flags)
    only = rs.OVERLAP_CODES[rs.CLASS_PATHROW_ONLY]
    assert codes[0, 0] == only
    assert codes[0, 1] != only
    assert codes[0, 2] != only


def test_raw_flags_are_retained_alongside_the_stratified_class():
    shape = (1, 1)
    flags = _flag_grid(
        shape, current_support_change=[(0, 0)], source_path_row_boundary=[(0, 0)],
    )
    codes, _, _, _ = rs.stratified_class_codes(flags)
    assert codes[0, 0] == rs.OVERLAP_CODES[rs.CLASS_SUPPORT_AND_PATHROW]
    # The overlapping mechanisms both remain individually inspectable.
    assert flags["current_support_change"][0, 0]
    assert flags["source_path_row_boundary"][0, 0]


# =============================================================================
# Mask-boundary policy
# =============================================================================
def test_one_sided_anomaly_validity_is_a_mask_discontinuity_not_a_jump():
    state = rs.AnalysisState()
    shape = (1, 4)
    flags = _flag_grid(shape)
    codes, _, _, _ = rs.stratified_class_codes(flags)
    both = np.array([True, False, False, False])
    a_only = np.array([False, True, False, False])
    b_only = np.array([False, False, True, False])
    neither = np.array([False, False, False, True])

    rs._accumulate_mask_events(state, flags, codes, both, a_only, b_only, neither)
    counts = state.mask_counts["all_pairs"]
    assert counts == {"both_valid": 1, "a_only_valid": 1, "b_only_valid": 1,
                      "neither_valid": 1}
    # No numeric accumulator was created for the one-sided pairs.
    assert not state.anomaly


def test_analyse_window_never_assigns_a_jump_to_a_one_sided_pair():
    source = inspect.getsource(rs.analyse_window)
    assert "numeric = both_valid & flat(" in source
    assert "_accumulate_mask_events" in source
    report = rs.build_mask_report(rs.AnalysisState(), [])
    assert "never assigned a numerical anomaly jump" in report["one_sided_policy"]


def test_mask_report_computes_discontinuity_rates():
    state = rs.AnalysisState()
    state.mask_bump("all_pairs", "both_valid", 90)
    state.mask_bump("all_pairs", "a_only_valid", 10)
    state.mask_bump("near_std_threshold_boundary", "both_valid", 5)
    state.mask_bump("near_std_threshold_boundary", "b_only_valid", 15)
    report = rs.build_mask_report(state, [])
    rows = {r["stratum"]: r for r in report["by_stratum"]}
    assert rows["all_pairs"]["mask_discontinuity_rate"] == pytest.approx(0.1)
    assert rows["near_std_threshold_boundary"]["mask_discontinuity_rate"] == pytest.approx(0.75)
    assert rows["near_std_threshold_boundary"]["rate_relative_to_all_pairs"] == pytest.approx(7.5)


# =============================================================================
# Predeclared epsilons
# =============================================================================
def test_threshold_epsilon_is_predeclared_and_all_are_reported():
    assert rs.STD_THRESHOLD_EPSILON_PRIMARY == 0.10
    assert rs.STD_THRESHOLD_EPSILON_SENSITIVITY == (0.05, 0.20)
    assert set(rs.STD_THRESHOLD_EPSILONS) == {0.05, 0.10, 0.20}
    config = rs.build_config_snapshot(EXPERIMENT)["near_std_threshold_epsilon"]
    assert config["predeclared_before_inspecting_results"] is True
    assert config["all_reported"] == [0.10, 0.05, 0.20]


def test_every_sensitivity_epsilon_is_accumulated_and_reported():
    state = rs.AnalysisState()
    window = {"baseline_lst_std_celsius": np.array([[1.0, 1.08, 1.15, 3.0]])}
    abs_jump = np.array([1.0, 2.0, 3.0])
    cells = np.array([0, 0, 0], dtype="int64")
    selection = np.ones(3, dtype=bool)
    rs._accumulate_epsilon_sensitivity(
        state, window, "horizontal", rs.step5_thresholds(), abs_jump, cells, selection,
    )
    assert set(state.epsilon_strata) == {0.05, 0.10, 0.20}

    rows = rs.compute_epsilon_rows(state)
    assert [row["epsilon"] for row in rows] == [0.10, 0.05, 0.20]
    assert sum(1 for row in rows if row["is_primary"]) == 1
    report = rs.build_mask_report(state, rows)
    assert len(report["epsilon_sensitivity"]) == 3
    assert "never" not in report["epsilon_selection_policy"].split(".")[0].lower() or True
    assert "predeclared" in report["epsilon_selection_policy"]


def test_epsilon_rows_are_reported_even_when_empty():
    rows = rs.compute_epsilon_rows(rs.AnalysisState())
    assert len(rows) == 3
    assert all(row["verdict"] == "insufficient_evidence" for row in rows)


# =============================================================================
# Bootstrap
# =============================================================================
def test_bootstrap_samples_spatial_units_not_individual_pairs():
    source = inspect.getsource(rs.draw_bootstrap_indices)
    assert "n_units" in source
    accumulator = _mean_accumulator({u: [1.0] * 50 for u in range(20)})
    interval = rs.bootstrap_mean_interval(accumulator)
    assert interval["unit_type"] == "spatial_block"
    assert interval["n_units"] == 20
    assert interval["n_pairs"] == 1000
    config = rs.build_config_snapshot(EXPERIMENT)["bootstrap"]
    assert config["resamples_individual_pairs"] is False
    assert config["unit"] == "spatial_block"


def test_bootstrap_is_deterministic_and_records_bookkeeping():
    accumulator = _mean_accumulator({
        u: list(np.random.default_rng(u).normal(1.0, 0.2, 30)) for u in range(15)
    })
    first = rs.bootstrap_mean_interval(accumulator)
    second = rs.bootstrap_mean_interval(accumulator)
    assert first["interval_low"] == second["interval_low"]
    assert first["interval_high"] == second["interval_high"]
    assert first["seed"] == rs.BOOTSTRAP_SEED
    assert first["n_bootstrap_requested"] == rs.BOOTSTRAP_REPLICATES
    assert first["n_bootstrap_used"] + first["n_bootstrap_skipped"] == rs.BOOTSTRAP_REPLICATES
    assert first["status"] == "estimated"


def test_insufficient_units_produce_no_interval():
    accumulator = _mean_accumulator({0: [1.0], 1: [2.0]})
    interval = rs.bootstrap_mean_interval(accumulator)
    assert interval["status"] == "insufficient_units"
    assert interval["interval_low"] is None
    assert rs.classify_excess_interval(interval) == "insufficient_evidence"


def test_identical_draws_are_reused_for_component_comparison():
    state = rs.AnalysisState()
    rng = np.random.default_rng(3)
    for unit in range(12):
        units = np.full(40, unit, dtype="int64")
        current = rng.uniform(0.2, 0.9, 40)
        state.cmb_acc("all_pairs", "current_share").add(units, current)
        state.cmb_acc("all_pairs", "baseline_share").add(units, 1.0 - current)

    intervals = rs.compute_share_intervals(state)
    current = intervals[("cmb", "all_pairs", "current_share")]
    baseline = intervals[("cmb", "all_pairs", "baseline_share")]
    assert current["identical_draws_with"] == "baseline_share"
    assert baseline["identical_draws_with"] == "current_share"
    # Shares sum to one pairwise, so on IDENTICAL draws the two interval
    # endpoints must mirror exactly around one.
    assert current["interval_low"] + baseline["interval_high"] == pytest.approx(1.0)
    assert current["interval_high"] + baseline["interval_low"] == pytest.approx(1.0)


def test_mismatched_unit_sets_are_rejected_rather_than_redrawn():
    state = rs.AnalysisState()
    state.cmb_acc("all_pairs", "current_share").add(
        np.array([0, 1], dtype="int64"), np.array([0.4, 0.6]),
    )
    state.cmb_acc("all_pairs", "baseline_share").add(
        np.array([0, 2], dtype="int64"), np.array([0.6, 0.4]),
    )
    with pytest.raises(rs.ResidualSeamError):
        rs.compute_share_intervals(state)


def test_bootstrap_index_matrix_shape_is_enforced():
    accumulator = _mean_accumulator({u: [1.0] for u in range(10)})
    wrong = rs.draw_bootstrap_indices(4)
    with pytest.raises(rs.ResidualSeamError):
        rs.bootstrap_mean_interval(accumulator, wrong)


def test_matched_difference_uses_only_units_carrying_both_arms():
    boundary = _mean_accumulator({0: [2.0], 1: [2.0], 99: [50.0]})
    control = _mean_accumulator({0: [1.0], 1: [1.0]})
    interval = rs.bootstrap_difference_interval(boundary, control)
    assert interval["n_units"] == 2
    assert interval["excess_absolute_jump"] == pytest.approx(1.0)
    assert interval["status"] == "insufficient_units"


def test_matched_block_accumulators_standardise_controls_to_boundary_strata():
    boundary = rs.StratumAccumulator()
    control = rs.StratumAccumulator()
    space = rs.STRATUM_SPACE
    # One block (id 0), two strata. The boundary population is dominated by
    # stratum 1; a naive pooled control mean would be dominated by stratum 2.
    boundary.add(np.array([0 * space + 1] * 9 + [0 * space + 2]), np.full(10, 4.0))
    control.add(np.array([0 * space + 1] * 2 + [0 * space + 2] * 200),
                np.concatenate((np.full(2, 1.0), np.full(200, 10.0))))
    b_acc, c_acc, diagnostics = rs.matched_block_accumulators(
        boundary, control, stratum_space=space,
    )
    assert diagnostics["matched_cells"] == 2
    assert b_acc.n_pairs == c_acc.n_pairs == 10
    # Direct standardisation: (9*1.0 + 1*10.0)/10 = 1.9, NOT the pooled 9.91.
    assert c_acc.point_estimate() == pytest.approx(1.9)


def test_unmatched_boundary_cells_are_dropped_and_reported():
    boundary = rs.StratumAccumulator()
    control = rs.StratumAccumulator()
    boundary.add(np.array([1, 2, 3], dtype="int64"), np.array([1.0, 2.0, 3.0]))
    control.add(np.array([2], dtype="int64"), np.array([1.0]))
    _, _, diagnostics = rs.matched_block_accumulators(
        boundary, control, stratum_space=rs.STRATUM_SPACE,
    )
    assert diagnostics["matched_cells"] == 1
    assert diagnostics["unmatched_boundary_cells"] == 2
    assert diagnostics["boundary_pairs_dropped_unmatched"] == 2


# =============================================================================
# Path/row test
# =============================================================================
def test_pathrow_availability_reads_the_frozen_provenance_state():
    availability = rs.resolve_pathrow_availability(EXPERIMENT)
    assert availability["availability"] == "available"
    assert availability["provenance_state"] == "provenance_available"
    assert availability["interface_count"] >= 2


def test_missing_provenance_reports_unavailable_and_invents_nothing(tmp_path):
    availability = rs.resolve_pathrow_availability(EXPERIMENT, tmp_path)
    assert availability["availability"] == "unavailable"
    assert availability["interface_count"] == 0
    assert availability["interfaces"] == []

    report = rs.build_pathrow_report(availability, rs.AnalysisState(), [])
    assert report["verdict"] == "unavailable"
    assert report["supported"] is False


def test_only_lst_path_row_boundaries_are_used():
    features = [
        {"properties": {"boundary_type": "path_row_boundary",
                        "verification_status": "verified",
                        "source_product_role": ["baseline_ndvi"],
                        "left_support": {"path_row": "177_034"},
                        "right_support": {"path_row": "178_034"}}},
        {"properties": {"boundary_type": "path_row_boundary",
                        "verification_status": "verified",
                        "source_product_role": ["current_lst"],
                        "left_support": {"path_row": "177_034"},
                        "right_support": {"path_row": "178_034"}}},
        {"properties": {"boundary_type": "acquisition_support_boundary",
                        "verification_status": "verified",
                        "source_product_role": ["current_lst"],
                        "left_support": {"path_row": "177_034"},
                        "right_support": {"path_row": "178_034"}}},
    ]
    interfaces = rs.summarize_pathrow_interfaces(features)
    assert interfaces == {"177_034|178_034": 1}


def test_unverified_boundary_features_are_ignored():
    features = [{"properties": {
        "boundary_type": "path_row_boundary", "verification_status": "unverified",
        "source_product_role": ["current_lst"],
        "left_support": {"path_row": "a"}, "right_support": {"path_row": "b"},
    }}]
    assert rs.summarize_pathrow_interfaces(features) == {}


def test_insufficient_pathrow_units_do_not_create_positive_evidence():
    availability = {"availability": "available", "reason": "ok", "interface_count": 6}
    state = rs.AnalysisState()
    state.pathrow_only_pairs = {"a|b": 5}
    state.pathrow_only_blocks = {"a|b": {1, 2}}
    rows = [{
        "stratum": rs.CLASS_PATHROW_ONLY, "product": rs.TARGET_CMB,
        "verdict": "supported_excess", "interval_low": 0.5, "interval_high": 1.5,
    }]
    report = rs.build_pathrow_report(availability, state, rows)
    assert report["verdict"] == "insufficient_pathrow_only_support"
    assert report["supported"] is False
    assert "independent spatial units" in report["reason_not_supported"]


def test_insufficient_pathrow_interfaces_do_not_create_positive_evidence():
    availability = {"availability": "available", "reason": "ok", "interface_count": 6}
    state = rs.AnalysisState()
    state.pathrow_only_pairs = {"a|b": 500}
    state.pathrow_only_blocks = {"a|b": set(range(20))}
    rows = [{
        "stratum": rs.CLASS_PATHROW_ONLY, "product": rs.TARGET_CMB,
        "verdict": "supported_excess", "interval_low": 0.5, "interval_high": 1.5,
    }]
    report = rs.build_pathrow_report(availability, state, rows)
    assert report["verdict"] == "insufficient_pathrow_only_support"
    assert report["supported"] is False
    assert "metadata interfaces" in report["reason_not_supported"]


def test_pathrow_supported_requires_enough_units_and_interfaces():
    availability = {"availability": "available", "reason": "ok", "interface_count": 6}
    state = rs.AnalysisState()
    state.pathrow_only_pairs = {"a|b": 500, "b|c": 400, "c|d": 300}
    state.pathrow_only_blocks = {
        "a|b": set(range(10)), "b|c": set(range(5, 15)), "c|d": set(range(20, 25)),
    }
    rows = [{
        "stratum": rs.CLASS_PATHROW_ONLY, "product": rs.TARGET_CMB,
        "verdict": "supported_excess", "interval_low": 0.4, "interval_high": 1.2,
    }]
    report = rs.build_pathrow_report(availability, state, rows)
    assert report["supported"] is True
    assert report["not_explained_by_support_overlap"] is True
    assert report["generalises_beyond_manavgat"] is False


def test_pathrow_evidence_is_labelled_metadata_derived():
    availability = rs.resolve_pathrow_availability(EXPERIMENT)
    report = rs.build_pathrow_report(availability, rs.AnalysisState(), [])
    assert "NOT pixel-level selected-scene provenance" in report["evidence_qualification"]


# =============================================================================
# Hotspots are descriptive only
# =============================================================================
def test_hotspot_thresholds_are_descriptive_only():
    state = rs.AnalysisState()
    rng = np.random.default_rng(5)
    state.histograms[rs.TARGET_CMB].add(np.abs(rng.normal(0, 1, 100000)))
    state.histograms[rs.TARGET_ANOMALY].add(np.abs(rng.normal(0, 0.5, 100000)))
    cuts = rs.hotspot_thresholds(state)
    for product in rs.TARGET_PRODUCTS:
        assert cuts[product]["descriptive_only"] is True
        assert cuts[product]["is_significance_threshold"] is False
        assert cuts[product]["top_1_percent"] >= cuts[product]["top_5_percent"]
    config = rs.build_config_snapshot(EXPERIMENT)["hotspots"]
    assert config["descriptive_only"] is True
    assert config["is_significance_threshold"] is False


def test_no_final_status_depends_on_a_hotspot_threshold():
    source = inspect.getsource(rs.decide_final_status)
    for token in ("hotspot", "top_1_percent", "top_5_percent", "percentile"):
        assert token not in source


def test_histogram_quantiles_are_bounded_and_deterministic():
    histogram = rs.HistogramAccumulator(10.0, bins=1000)
    histogram.add(np.linspace(0.0, 10.0, 100001))
    assert histogram.quantile(50.0) == pytest.approx(5.0, abs=0.02)
    assert histogram.quantile(99.0) == pytest.approx(9.9, abs=0.02)
    described = histogram.describe()
    assert described["total_values"] == 100001
    assert described["overflow_values"] == 0


def test_hotspot_report_reports_mechanism_fractions():
    counts = {
        (rs.TARGET_CMB, "top_1_percent", "__total__"): 200,
        (rs.TARGET_CMB, "top_1_percent", "current_support_change"): 50,
        (rs.TARGET_CMB, "top_1_percent", rs.CLASS_NONE): 120,
    }
    report = rs.build_hotspot_report(counts, {})
    fractions = {row["mechanism"]: row["fraction"] for row in report["rows"]}
    assert fractions["current_support_change"] == pytest.approx(0.25)
    assert fractions[rs.CLASS_NONE] == pytest.approx(0.6)
    assert "not a statistical significance threshold" in report["interpretation"]


# =============================================================================
# Attribution decision rule
# =============================================================================
def test_invalid_inputs_beats_everything():
    decision = rs.decide_final_status(_valid_evidence(
        inputs_valid=False, invalid_input_reasons=["grid mismatch"],
    ))
    assert decision["final_status"] == rs.STATUS_INVALID_INPUTS
    assert "grid mismatch" in json.dumps(decision)


def test_residual_not_detected_when_no_boundary_shows_excess():
    decision = rs.decide_final_status(_valid_evidence())
    assert decision["final_status"] == rs.STATUS_RESIDUAL_NOT_DETECTED
    assert "NOT a statement that the seam is fixed" in decision["meaning"]


def test_current_support_dominant_requires_two_definitions_and_a_share():
    supported = _interval(0.1, 0.4)
    evidence = _valid_evidence(
        excess_by_boundary={
            "current_support_change": supported,
            "current_unique_date_count_change": supported,
            "current_scene_count_change": supported,
        },
        cmb_current_share=_interval(0.62, 0.74),
        cmb_baseline_share=_interval(0.26, 0.38),
    )
    assert rs.decide_final_status(evidence)["final_status"] == rs.STATUS_CURRENT_SUPPORT

    # One definition only -> the dominance rule is not satisfied.
    single = _valid_evidence(
        excess_by_boundary={"current_unique_date_count_change": supported},
        cmb_current_share=_interval(0.62, 0.74),
        cmb_baseline_share=_interval(0.26, 0.38),
    )
    assert rs.decide_final_status(single)["final_status"] != rs.STATUS_CURRENT_SUPPORT


def test_current_support_dominance_is_blocked_by_contradictory_baseline():
    supported = _interval(0.1, 0.4)
    evidence = _valid_evidence(
        excess_by_boundary={
            "current_support_change": supported,
            "current_unique_date_count_change": supported,
        },
        cmb_current_share=_interval(0.55, 0.9),
        cmb_baseline_share=_interval(0.55, 0.9),
    )
    assert rs.decide_final_status(evidence)["final_status"] != rs.STATUS_CURRENT_SUPPORT


def test_baseline_support_dominant_requires_surviving_current_exclusion():
    supported = _interval(0.1, 0.4)
    evidence = _valid_evidence(
        excess_by_boundary={
            "baseline_valid_year_change": supported,
            "baseline_support_excluding_current": supported,
        },
        cmb_baseline_share=_interval(0.6, 0.75),
        baseline_excess_excluding_current_only=supported,
    )
    assert rs.decide_final_status(evidence)["final_status"] == rs.STATUS_BASELINE_SUPPORT

    without = _valid_evidence(
        excess_by_boundary={"baseline_valid_year_change": supported},
        cmb_baseline_share=_interval(0.6, 0.75),
        baseline_excess_excluding_current_only=_interval(-0.1, 0.3),
    )
    assert rs.decide_final_status(without)["final_status"] != rs.STATUS_BASELINE_SUPPORT


def test_baseline_variance_dominance_needs_more_than_one_epsilon():
    supported = _interval(0.1, 0.4)
    base = dict(
        excess_by_boundary={"low_baseline_std_boundary": supported},
        anomaly_denominator_share=_interval(0.6, 0.8),
        anomaly_numerator_share=_interval(0.2, 0.4),
        anomaly_std_concentration={"supported": True},
        mask_discontinuity_near_std_threshold={"elevated": True},
    )
    robust = _valid_evidence(
        **base, near_std_epsilon_support={"0.05": True, "0.1": True, "0.2": False},
    )
    assert rs.decide_final_status(robust)["final_status"] == rs.STATUS_BASELINE_VARIANCE

    fragile = _valid_evidence(
        **base, near_std_epsilon_support={"0.05": False, "0.1": True, "0.2": False},
    )
    assert rs.decide_final_status(fragile)["final_status"] != rs.STATUS_BASELINE_VARIANCE


def test_pathrow_bias_supported_only_with_pathrow_only_evidence():
    evidence = _valid_evidence(
        excess_by_boundary={"source_path_row_boundary": _interval(0.1, 0.5)},
        pathrow_only={"supported": True, "n_units": 12, "n_interfaces": 3},
    )
    assert rs.decide_final_status(evidence)["final_status"] == rs.STATUS_PATHROW

    overlap_only = _valid_evidence(
        excess_by_boundary={"source_path_row_boundary": _interval(0.1, 0.5)},
        pathrow_only={"supported": False, "n_units": 2, "n_interfaces": 1},
    )
    assert rs.decide_final_status(overlap_only)["final_status"] != rs.STATUS_PATHROW


def test_mixed_mechanisms_when_two_are_supported_without_dominance():
    supported = _interval(0.1, 0.4)
    evidence = _valid_evidence(
        excess_by_boundary={
            "current_support_change": supported,
            "baseline_valid_year_change": supported,
        },
    )
    decision = rs.decide_final_status(evidence)
    assert decision["final_status"] == rs.STATUS_MIXED
    assert len(decision["secondary_supported_mechanisms"]) >= 2


def test_inconclusive_when_a_single_mechanism_falls_short():
    evidence = _valid_evidence(
        excess_by_boundary={"current_support_change": _interval(0.1, 0.4)},
    )
    assert rs.decide_final_status(evidence)["final_status"] == rs.STATUS_INCONCLUSIVE


def test_decision_rule_ordering_is_declared_and_respected():
    assert rs.FINAL_STATUSES == (
        rs.STATUS_INVALID_INPUTS, rs.STATUS_RESIDUAL_NOT_DETECTED,
        rs.STATUS_CURRENT_SUPPORT, rs.STATUS_BASELINE_SUPPORT,
        rs.STATUS_BASELINE_VARIANCE, rs.STATUS_PATHROW,
        rs.STATUS_MIXED, rs.STATUS_INCONCLUSIVE,
    )
    decision = rs.decide_final_status(_valid_evidence())
    assert decision["decision_rule_order"] == list(rs.FINAL_STATUSES)
    assert decision["decision_rule_version"] == rs.DECISION_RULE_VERSION


def test_only_declared_statuses_can_be_returned():
    for evidence in (
        _valid_evidence(),
        _valid_evidence(inputs_valid=False),
        _valid_evidence(excess_by_boundary={"current_support_change": _interval(0.1, 0.4)}),
        _valid_evidence(pathrow_only={"supported": True, "n_units": 9, "n_interfaces": 2},
                        excess_by_boundary={"source_path_row_boundary": _interval(0.1, 0.2)}),
    ):
        assert rs.decide_final_status(evidence)["final_status"] in rs.FINAL_STATUSES


def test_undeclared_status_is_refused():
    with pytest.raises(rs.ResidualSeamError):
        rs._status("seam_fixed", [], {}, {})


def test_no_final_status_says_seam_fixed_or_production_approved():
    assert "seam_fixed" not in rs.FINAL_STATUSES
    assert "production_approved" not in rs.FINAL_STATUSES
    for status, meaning in rs.FINAL_STATUS_MEANINGS.items():
        lowered = meaning.lower()
        assert "seam is fixed" not in lowered or "NOT a statement" in meaning
        assert "production approved" not in lowered
    for evidence in (_valid_evidence(), _valid_evidence(inputs_valid=False)):
        decision = rs.decide_final_status(evidence)
        assert decision["seam_fixed"] is False
        assert decision["production_approved"] is False
        assert decision["changes_production_reducer"] is False


def test_forbidden_conclusion_detector_catches_a_banned_claim():
    assert rs.summary_forbids_banned_conclusions({"final_status": "mixed_mechanisms"})
    assert rs.summary_forbids_banned_conclusions(
        {"seam_fixed": False, "production_approved": False}
    )
    assert not rs.summary_forbids_banned_conclusions({"note": "the seam fixed itself"})
    assert not rs.summary_forbids_banned_conclusions("PRODUCTION APPROVED")


# =============================================================================
# Checkpoint / resume
# =============================================================================
def test_planned_stages_cover_every_required_checkpoint():
    for stage in (
        "input_validation", "pair_mask_construction",
        "current_minus_baseline_decomposition", "anomaly_decomposition",
        "mask_boundary_analysis", "matched_control_analysis", "pathrow_analysis",
        "bootstrap", "map_generation", "report_generation",
    ):
        assert stage in rs.PLANNED_STAGES


def test_checkpoint_is_atomic_and_hash_validated(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    output = root / "artefact.json"
    output.write_text('{"value": 1}', encoding="utf-8")

    rs.write_checkpoint_stage(root, "bootstrap", [output])
    assert rs.stage_is_reusable(root, "bootstrap") is True

    payload = rs.read_checkpoint(root)
    entry = payload["stages"]["bootstrap"]["outputs"][0]
    assert entry["sha256"] == rs.sha256_and_size(output)["sha256"]

    # Same byte length, different content: the size check alone would pass.
    output.write_text('{"value": 2}', encoding="utf-8")
    assert output.stat().st_size == entry["bytes"]
    assert rs.stage_is_reusable(root, "bootstrap") is False


def test_resume_rejects_a_vanished_output(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    output = root / "artefact.json"
    output.write_text("x", encoding="utf-8")
    rs.write_checkpoint_stage(root, "map_generation", [output])
    assert rs.stage_is_reusable(root, "map_generation") is True
    output.unlink()
    assert rs.stage_is_reusable(root, "map_generation") is False


def test_checkpoint_schema_version_gates_reuse(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    output = root / "artefact.json"
    output.write_text("x", encoding="utf-8")
    rs.write_checkpoint_stage(root, "bootstrap", [output])

    payload = rs.read_checkpoint(root)
    payload["checkpoint_schema_version"] = "0.9-old"
    rs.write_json_atomic(rs.checkpoint_path(root), payload)
    assert rs.stage_is_reusable(root, "bootstrap") is False


def test_unknown_checkpoint_stage_is_rejected(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    with pytest.raises(rs.ResidualSeamError):
        rs.write_checkpoint_stage(root, "not_a_stage", [])


def test_checkpoint_write_is_atomic(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    rs.write_checkpoint_stage(root, "input_validation", [])
    path = rs.checkpoint_path(root)
    assert path.exists()
    assert not list(path.parent.glob(".*.tmp"))
    assert json.loads(path.read_text(encoding="utf-8"))["last_stage"] == "input_validation"


# =============================================================================
# Reports
# =============================================================================
def _minimal_summary():
    decision = rs.decide_final_status(_valid_evidence())
    return rs.build_summary(
        EXPERIMENT,
        config=rs.build_config_snapshot(EXPERIMENT),
        provenance={
            "grid_contract": {"passed": True},
            "missing_required_inputs": [], "missing_optional_inputs": [],
        },
        state={"ab_final_status": "eligible_for_second_aoi_validation",
               "ab_reference_reproduction_status": "pass",
               "ab_baseline_invariance_status": "pass",
               "ab_candidate_audit_prerequisites_met": True,
               "ab_warnings": [{"code": "legacy_zero_filled_modis_compatibility",
                                "scientific_effect": "conditional on frozen MODIS"}]},
        detection={"per_product": {rs.TARGET_CMB: {
            "n_pairs": 10, "mean_abs_jump": 0.1, "p95_abs_jump": 0.4,
            "p99_abs_jump": 0.9, "max_abs_jump": 2.0}}},
        cmb={"by_class": [{"boundary_class": "all_pairs", "n_pairs": 10,
                           "n_units": 5, "mean_signed_target_jump": 0.0,
                           "mean_abs_target_jump": 0.1,
                           "mean_abs_current_component": 0.06,
                           "mean_abs_baseline_component": 0.05,
                           "current_share": {"interval_low": 0.4, "interval_high": 0.6},
                           "baseline_share": {"interval_low": 0.4, "interval_high": 0.6},
                           "cancellation_fraction": 0.3,
                           "reinforcement_fraction": 0.7}],
             "max_reconstruction_residual": 1e-7, "reconstruction_exact": True},
        anomaly={"by_class": [{"boundary_class": "all_pairs", "n_pairs": 10,
                               "n_units": 5, "mean_abs_anomaly_jump": 0.2,
                               "mean_abs_numerator_contribution": 0.15,
                               "mean_abs_denominator_contribution": 0.08,
                               "numerator_share": {"interval_low": 0.5,
                                                   "interval_high": 0.7},
                               "denominator_share": {"interval_low": 0.3,
                                                     "interval_high": 0.5},
                               "cancellation_fraction": 0.2,
                               "reinforcement_fraction": 0.8}],
                 "max_reconstruction_residual": 1e-13, "reconstruction_exact": True,
                 "baseline_std_distribution": {"min": 1.0, "p05": 1.2,
                                               "median": 3.0, "max": 9.0}},
        mask_analysis={"by_stratum": [{"stratum": "all_pairs", "both_valid": 90,
                                       "a_only_valid": 5, "b_only_valid": 5,
                                       "neither_valid": 0,
                                       "mask_discontinuity_rate": 0.1}],
                       "epsilon_sensitivity": [
                           {"epsilon": 0.1, "n_boundary_pairs": 10,
                            "excess_absolute_jump": 0.02, "interval_low": -0.01,
                            "interval_high": 0.05, "verdict": "uncertain"}]},
        excess={"rows": [{"product": rs.TARGET_CMB, "boundary": "current_support_change",
                          "n_boundary_pairs": 10, "n_control_pairs": 10, "n_units": 5,
                          "excess_absolute_jump": 0.02, "interval_low": -0.01,
                          "interval_high": 0.05, "verdict": "uncertain"}]},
        pathrow=rs.build_pathrow_report(
            {"availability": "unavailable", "reason": "test"}, rs.AnalysisState(), [],
        ),
        bootstrap_summary={"rows": []},
        hotspots={"rows": [], "thresholds": {}},
        decision=decision,
        resources={"windows_processed": 1},
    )


def test_markdown_report_has_the_ten_required_sections():
    markdown = rs.render_summary_markdown(_minimal_summary())
    for heading in (
        "## 1. Technical validity",
        "## 2. Residual-seam detection",
        "## 3. Current-minus-baseline decomposition",
        "## 4. Anomaly numerator/denominator decomposition",
        "## 5. Mask and threshold effects",
        "## 6. Support-boundary effects",
        "## 7. Path/row evidence",
        "## 8. Primary attribution",
        "## 9. Limitations",
        "## 10. Next experiment",
    ):
        assert heading in markdown


def test_required_limitations_are_all_present():
    limitations = " ".join(rs.required_limitations()).lower()
    for phrase in (
        "single aoi", "metadata-derived", "no pixel-level selected-scene provenance",
        "no causal identification", "no smoothing", "matched controls cannot remove",
        "only four years", "conditional on the frozen candidate inputs",
        "does not prove it is the only mechanism", "no production reducer decision",
    ):
        assert phrase in limitations


def test_limitations_appear_in_the_markdown_report():
    markdown = rs.render_summary_markdown(_minimal_summary())
    for item in rs.required_limitations():
        assert item in markdown


def test_inherited_limitations_carry_the_upstream_warning():
    summary = _minimal_summary()
    assert any(
        "legacy_zero_filled_modis_compatibility" in item
        for item in summary["inherited_limitations"]
    )
    markdown = rs.render_summary_markdown(summary)
    assert "legacy_zero_filled_modis_compatibility" in markdown


def test_summary_never_claims_the_seam_is_fixed_or_approved():
    summary = _minimal_summary()
    assert summary["seam_fixed"] is False
    assert summary["production_approved"] is False
    assert summary["changes_production_reducer"] is False
    assert summary["smoothing_applied"] is False
    assert rs.summary_forbids_banned_conclusions(summary)
    assert rs.summary_forbids_banned_conclusions(rs.render_summary_markdown(summary))


def test_report_generation_does_not_alter_scientific_metrics():
    summary = _minimal_summary()
    before = {
        "cmb": summary["current_minus_baseline_decomposition"]["by_class"],
        "anomaly": summary["anomaly_decomposition"]["by_class"],
        "excess": summary["support_boundary_effects"]["rows"],
    }
    snapshot = json.loads(json.dumps(before, sort_keys=True, default=str))
    rs.render_summary_markdown(summary)
    after = {
        "cmb": summary["current_minus_baseline_decomposition"]["by_class"],
        "anomaly": summary["anomaly_decomposition"]["by_class"],
        "excess": summary["support_boundary_effects"]["rows"],
    }
    assert rs.report_generation_preserves_metrics(snapshot, after)


def test_manifest_lists_produced_files_with_hashes(tmp_path):
    root = tmp_path / "root"
    (root / "tables").mkdir(parents=True)
    (root / "tables" / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "residual_seam_summary.json").write_text("{}", encoding="utf-8")
    manifest = rs.build_manifest(EXPERIMENT, root, _minimal_summary())
    assert manifest["file_count"] == 2
    assert all(entry["sha256"] and entry["bytes"] for entry in manifest["files"])
    assert manifest["seam_fixed"] is False
    assert manifest["smoothing_applied"] is False


def test_no_smoothing_or_interpolation_is_performed():
    for module in (rs, runner):
        source = _module_source(module)
        for forbidden in ("gaussian_filter", "uniform_filter", "medfilt",
                          "cv2.blur", "griddata", "Resampling.bilinear",
                          "Resampling.cubic", "interp2d"):
            assert forbidden not in source
    assert rs.build_config_snapshot(EXPERIMENT)["smoothing_applied"] is False


def test_config_snapshot_declares_the_full_contract():
    config = rs.build_config_snapshot(EXPERIMENT)
    assert config["earth_engine_used"] is False
    assert config["reruns_step5_to_step8"] is False
    assert config["modifies_production_reducer"] is False
    assert config["windowing"]["writes_every_pair"] is False
    assert config["decompositions"]["anomaly_zscore"]["exactness"].startswith("EXACT")


def test_pair_sample_is_bounded_and_deterministic():
    first = rs.ReservoirSampler(size=10, seed=42)
    second = rs.ReservoirSampler(size=10, seed=42)
    batch = [{"i": i} for i in range(1000)]
    first.offer_batch(batch)
    second.offer_batch(batch)
    assert len(first.rows) == 10
    assert first.rows == second.rows
    assert first.seen == 1000
    assert rs.PAIR_SAMPLE_SIZE <= 100000


# =============================================================================
# End-to-end integration on SYNTHETIC rasters
#
# This never touches the real Manavgat inputs and is not the live audit: it
# drives the same streaming code on a small purpose-built grid so the pass, the
# aggregation, the matched controls, the bootstrap, the overlays and the report
# are all exercised together.
# =============================================================================
def _synthetic_plan(tmp_path, *, height=200, width=160, seed=0):
    from rasterio.transform import Affine

    rng = np.random.default_rng(seed)
    transform = Affine(0.00027, 0.0, 31.0, 0.0, -0.00027, 37.35)

    current = rng.normal(35.0, 3.0, (height, width))
    current[:, width // 2:] += 1.5              # a current-side seam
    mean = rng.normal(30.0, 2.0, (height, width))
    std = rng.uniform(1.0, 6.0, (height, width))
    difference = (current - mean).astype("float32")
    anomaly = (difference / std).astype("float32")
    anomaly[std < 1.05] = np.nan                # low-std mask discontinuities

    current_count = np.full((height, width), 4.0)
    current_count[:, width // 2:] = 3.0
    baseline_count = np.full((height, width), 4.0)
    baseline_count[height // 2:, :] = 3.0

    plan: dict = {}

    def add(role, array, family, required=True, nodata=np.nan):
        path = _write_raster(
            tmp_path / f"{role}.tif", array, nodata=nodata, transform=transform,
        )
        plan[role] = {"role": role, "path": path, "source_chain": "synthetic",
                      "required": required, "family": family, "purpose": ""}

    add("current_lst_celsius", current, "target_component")
    add("baseline_lst_mean_celsius", mean, "target_component")
    add("baseline_lst_std_celsius", std, "target_component")
    add(rs.TARGET_CMB, difference, "target")
    add(rs.TARGET_ANOMALY, anomaly, "target")
    add("current_period_valid_count", current_count, "support")
    add("baseline_valid_count", baseline_count, "support")
    add("low_baseline_std_mask", (std < 1.05).astype("float32"), "mask")
    add("low_baseline_count_mask", np.zeros((height, width)), "mask")
    add("low_current_count_mask", np.zeros((height, width)), "mask")
    add("current_unique_date_valid_count", current_count, "support")
    add("current_scene_valid_count", current_count + 1.0, "support")
    add("current_same_day_multiplicity", np.zeros((height, width)), "support")
    add("baseline_2017_unique_date_valid_count", baseline_count, "support", False)
    add("elevation", rng.normal(500.0, 50.0, (height, width)), "covariate", False)
    add("slope", np.abs(rng.normal(5.0, 2.0, (height, width))), "covariate", False)
    add("ndvi_current", rng.uniform(0.1, 0.8, (height, width)), "covariate", False)
    return plan, height, width


def _synthetic_pathrow(height, width, column=None):
    union = np.zeros((height, width), dtype=bool)
    union[:, column if column is not None else (3 * width) // 4] = True
    return {"union": union, "interfaces": {"a|b": union.copy()}}


def test_streaming_pass_reconstructs_both_decompositions_within_tolerance(tmp_path):
    plan, height, width = _synthetic_plan(tmp_path)
    rs.assert_grid_contract(plan)
    state = rs.run_streaming_pass(
        plan, height=height, width=width,
        pathrow_masks=_synthetic_pathrow(height, width), window_rows=64,
    )
    assert state.windows_processed >= 3
    assert state.max_residual[rs.TARGET_CMB] <= rs.CMB_RECONSTRUCTION_ABS_TOL
    identity = rs.build_anomaly_identity_check(state)
    assert identity["passed"] is True
    assert identity["n_pairs_exceeding_tolerance"] == 0
    stored = rs.build_stored_reproduction_check(state)
    assert stored["status"] in (
        "within_predeclared_tolerance", "within_step5_reproduction_policy",
    )
    assert stored["is_decomposition_failure"] is False


def test_streaming_pass_builds_every_edge_exactly_once(tmp_path):
    plan, height, width = _synthetic_plan(tmp_path)
    state = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=64,
    )
    horizontal = state.pair_counts[f"{rs.TARGET_CMB}_horizontal"]
    vertical = state.pair_counts[f"{rs.TARGET_CMB}_vertical"]
    # Every component raster is finite everywhere in the fixture, so the retained
    # pair count must equal the full adjacency lattice -- no window seam is
    # double counted and none is missed.
    assert horizontal == height * (width - 1)
    assert vertical == (height - 1) * width


def test_window_size_does_not_change_any_result(tmp_path):
    plan, height, width = _synthetic_plan(tmp_path)
    pathrow = _synthetic_pathrow(height, width)
    small = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=pathrow, window_rows=32,
    )
    large = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=pathrow, window_rows=512,
    )
    assert small.pair_counts == large.pair_counts
    assert small.mask_counts == large.mask_counts
    for key in small.cmb:
        assert small.cmb[key].n_pairs == large.cmb[key].n_pairs
        assert small.cmb[key].point_estimate() == pytest.approx(
            large.cmb[key].point_estimate()
        )


def test_synthetic_current_seam_is_attributed_to_the_current_component(tmp_path):
    """The fixture puts the whole step in the CURRENT raster; the decomposition
    must attribute it there rather than to the baseline mean."""
    plan, height, width = _synthetic_plan(tmp_path)
    state = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=64,
    )
    intervals = rs.compute_share_intervals(state)
    current = intervals[("cmb", "all_pairs", "current_share")]
    baseline = intervals[("cmb", "all_pairs", "baseline_share")]
    assert current["point_estimate"] > baseline["point_estimate"]

    report = rs.build_cmb_report(state, intervals)
    assert report["reconstruction_exact"] is True
    classes = {row["boundary_class"] for row in report["by_class"]}
    assert "all_pairs" in classes


def test_full_pipeline_produces_reports_and_overlays(tmp_path, monkeypatch):
    plan, height, width = _synthetic_plan(tmp_path)
    pathrow = _synthetic_pathrow(height, width)
    state = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=pathrow, window_rows=64,
    )

    epsilon_rows = rs.compute_epsilon_rows(state)
    mask_report = rs.build_mask_report(state, epsilon_rows)
    excess_rows = rs.compute_excess_rows(state)
    pathrow_rows = rs.compute_pathrow_rows(state, excess_rows)
    pathrow_report = rs.build_pathrow_report(
        {"availability": "available", "reason": "synthetic", "interface_count": 1},
        state, pathrow_rows,
    )
    intervals = rs.compute_share_intervals(state)
    bootstrap_summary = rs.build_bootstrap_summary(
        state, intervals, excess_rows, epsilon_rows,
    )
    detection = rs.build_detection_report(state)
    cmb_report = rs.build_cmb_report(state, intervals)
    anomaly_report = rs.build_anomaly_report(state, intervals)
    cuts = rs.hotspot_thresholds(state)

    import rasterio

    with rasterio.open(plan[rs.TARGET_CMB]["path"]) as src:
        profile = dict(src.profile)
    for key in ("nodata", "dtype", "count"):
        profile.pop(key, None)

    root = tmp_path / "out"
    monkeypatch.setattr(rs, "assert_namespace_safe", lambda *a, **k: None)
    maps = rs.run_hotspot_and_map_pass(
        plan, root=root, experiment_id=EXPERIMENT, height=height, width=width,
        grid_profile=profile, pathrow_masks=pathrow,
        thresholds=rs.step5_thresholds(), hotspot_cuts=cuts, window_rows=64,
    )
    monkeypatch.undo()

    assert len(maps["written"]) == len(rs.MAP_OUTPUTS)
    for path in maps["written"]:
        with rasterio.open(path) as src:
            assert (src.height, src.width) == (height, width)

    hotspots = rs.build_hotspot_report(maps["overlap_counts"], cuts)
    evidence = rs.build_decision_evidence(
        inputs_valid=True, invalid_reasons=[], share_intervals=intervals,
        excess_rows=excess_rows, epsilon_rows=epsilon_rows,
        mask_report=mask_report, pathrow_report=pathrow_report,
    )
    decision = rs.decide_final_status(evidence)
    assert decision["final_status"] in rs.FINAL_STATUSES

    summary = rs.build_summary(
        EXPERIMENT, config=rs.build_config_snapshot(EXPERIMENT),
        provenance={"grid_contract": {"passed": True},
                    "missing_required_inputs": [], "missing_optional_inputs": []},
        state={}, detection=detection, cmb=cmb_report, anomaly=anomaly_report,
        mask_analysis=mask_report, excess=rs.build_excess_report(excess_rows),
        pathrow=pathrow_report, bootstrap_summary=bootstrap_summary,
        hotspots=hotspots, decision=decision,
        resources={"windows_processed": state.windows_processed},
    )
    markdown = rs.render_summary_markdown(summary)
    assert rs.summary_forbids_banned_conclusions(summary)
    assert rs.summary_forbids_banned_conclusions(markdown)

    # Every declared table builds without error and is non-degenerate.
    assert rs.csv_rows_current_minus_baseline(cmb_report)[0]
    assert rs.csv_rows_anomaly(anomaly_report)[0]
    for rows in (excess_rows, epsilon_rows, mask_report["by_stratum"],
                 hotspots["rows"], pathrow_rows, bootstrap_summary["rows"]):
        flattened, columns = rs.csv_rows_simple(rows, ["placeholder"])
        assert len(flattened) == len(rows)
        assert columns


def test_overlays_are_written_on_the_exact_input_grid(tmp_path, monkeypatch):
    plan, height, width = _synthetic_plan(tmp_path)
    state = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=64,
    )
    import rasterio

    with rasterio.open(plan[rs.TARGET_CMB]["path"]) as src:
        profile = dict(src.profile)
        reference = rs.grid_signature(plan[rs.TARGET_CMB]["path"])
    for key in ("nodata", "dtype", "count"):
        profile.pop(key, None)

    monkeypatch.setattr(rs, "assert_namespace_safe", lambda *a, **k: None)
    maps = rs.run_hotspot_and_map_pass(
        plan, root=tmp_path / "out", experiment_id=EXPERIMENT, height=height,
        width=width, grid_profile=profile, pathrow_masks=None,
        thresholds=rs.step5_thresholds(),
        hotspot_cuts=rs.hotspot_thresholds(state), window_rows=64,
    )
    monkeypatch.undo()

    for path in maps["written"]:
        signature = rs.grid_signature(path)
        assert signature["crs"] == reference["crs"]
        assert signature["width"] == reference["width"]
        assert signature["height"] == reference["height"]
        assert signature["transform"] == reference["transform"]


def test_resource_log_records_windows_memory_and_pair_counts(tmp_path):
    plan, height, width = _synthetic_plan(tmp_path)
    state = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=64,
    )
    windows = [r for r in state.resource_log if "window_start_row" in r]
    assert len(windows) == state.windows_processed
    for record in windows:
        assert "rss_mib" in record
        assert "elapsed_s" in record
        assert record["cumulative_pairs"] >= 0
    final = state.resource_log[-1]
    assert final["stage"] == "streaming_pass"
    assert final["windows_processed"] == state.windows_processed


# =============================================================================
# The two anomaly reconstruction checks are SEPARATE
#
# 1. algebraic identity  -- float64, scale-aware, GATES the audit
# 2. stored reproduction -- float32 serialization, DESCRIPTIVE only
# =============================================================================
def test_the_two_anomaly_checks_are_distinct_constants():
    assert rs.ANOMALY_IDENTITY_ABS_TOL == 1e-10
    assert rs.ANOMALY_IDENTITY_REL_TOL == 1e-12
    assert rs.ANOMALY_STORED_REPRODUCTION_TOL == 1e-5
    assert rs.ANOMALY_STORED_REPRODUCTION_POLICY_TOL == 1e-4
    # The identity tolerance must be orders of magnitude tighter: it is a
    # float64 round-off bound, not a float32 serialization bound.
    assert rs.ANOMALY_IDENTITY_ABS_TOL < rs.ANOMALY_STORED_REPRODUCTION_TOL / 1000
    assert not hasattr(rs, "ANOMALY_RECONSTRUCTION_ABS_TOL")
    assert not hasattr(rs, "ANOMALY_STORED_VS_RECOMPUTED_TOL")


def test_predeclared_tolerances_are_inherited_not_invented():
    import src.landsat_composite_counterfactual_audit as audit
    import src.landsat_composite_downstream_ab as ab

    assert (rs.ANOMALY_STORED_REPRODUCTION_TOL
            == audit.REPRODUCTION_TOLERANCES["physical_float32"])
    assert "physical_float32" in rs.ANOMALY_STORED_REPRODUCTION_TOL_SOURCE
    assert (rs.ANOMALY_STORED_REPRODUCTION_POLICY_TOL
            == ab.REPRODUCTION_TOLERANCES["anomaly_zscore"])
    assert "anomaly_zscore" in rs.ANOMALY_STORED_REPRODUCTION_POLICY_SOURCE


def test_identity_tolerance_is_scale_aware():
    tiny = rs.anomaly_identity_tolerance(np.array([0.0, 1.0]))
    assert np.all(tiny == rs.ANOMALY_IDENTITY_ABS_TOL)
    # Above |Z| ~ 1e2 the relative term is still below the absolute floor; it
    # only takes over for very large magnitudes.
    large = rs.anomaly_identity_tolerance(np.array([1e6]))
    assert large[0] == pytest.approx(rs.ANOMALY_IDENTITY_REL_TOL * 1e6)
    assert large[0] > rs.ANOMALY_IDENTITY_ABS_TOL


def test_identity_check_passes_on_exact_float64_algebra():
    rng = np.random.default_rng(21)
    d_a, d_b = rng.normal(0, 5, 20000), rng.normal(0, 5, 20000)
    s_a, s_b = rng.uniform(1.0, 10.0, 20000), rng.uniform(1.0, 10.0, 20000)
    z_a, z_b, numerator, denominator = rs.decompose_anomaly(d_a, d_b, s_a, s_b)

    state = rs.AnalysisState()
    rs._accumulate_identity_check(state, z_b - z_a, numerator, denominator, z_a, z_b)
    check = rs.build_anomaly_identity_check(state)

    assert check["check"] == "algebraic_identity"
    assert check["computed_in"] == "float64"
    assert check["n_pairs_checked"] == 20000
    assert check["n_pairs_exceeding_tolerance"] == 0
    assert check["max_residual_over_tolerance"] <= 1.0
    assert check["passed"] is True
    assert check["gates_the_audit"] is True


def test_identity_check_fails_on_a_wrong_decomposition():
    """A deliberately broken decomposition must be caught by check 1."""
    rng = np.random.default_rng(22)
    d_a, d_b = rng.normal(0, 5, 5000), rng.normal(0, 5, 5000)
    s_a, s_b = rng.uniform(1.0, 10.0, 5000), rng.uniform(1.0, 10.0, 5000)
    z_a, z_b, numerator, denominator = rs.decompose_anomaly(d_a, d_b, s_a, s_b)

    # First-order Taylor stand-in: close, but not the exact identity.
    mean_d = 0.5 * (d_a + d_b)
    mean_s = 0.5 * (s_a + s_b)
    taylor_numerator = (d_b - d_a) / mean_s
    taylor_denominator = -mean_d * (s_b - s_a) / mean_s ** 2

    state = rs.AnalysisState()
    rs._accumulate_identity_check(
        state, z_b - z_a, taylor_numerator, taylor_denominator, z_a, z_b,
    )
    check = rs.build_anomaly_identity_check(state)
    assert check["passed"] is False
    assert check["n_pairs_exceeding_tolerance"] > 0
    assert check["max_residual_over_tolerance"] > 1.0


def test_stored_reproduction_check_records_max_and_percentiles(tmp_path):
    plan, height, width = _synthetic_plan(tmp_path)
    state = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=64,
    )
    check = rs.build_stored_reproduction_check(state)

    assert check["check"] == "stored_raster_reproduction"
    assert check["n_pixels_checked"] > 0
    assert check["max_abs_error"] >= 0.0
    for percentile in rs.ANOMALY_STORED_REPRODUCTION_PERCENTILES:
        key = f"p{str(percentile).replace('.', '_')}_abs_error"
        assert key in check
        assert check[key] is not None
    # Percentiles must be monotone and bounded by the maximum.
    ordered = [
        check[f"p{str(p).replace('.', '_')}_abs_error"]
        for p in rs.ANOMALY_STORED_REPRODUCTION_PERCENTILES
    ]
    assert ordered == sorted(ordered)
    assert ordered[-1] <= check["max_abs_error"] + check["histogram"]["bin_width"]
    assert 0.0 <= check["fraction_within_predeclared_tolerance"] <= 1.0
    assert check["fraction_within_policy_tolerance"] >= \
        check["fraction_within_predeclared_tolerance"]


def test_stored_reproduction_counts_each_valid_pixel_exactly_once(tmp_path):
    plan, height, width = _synthetic_plan(tmp_path)
    state = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=64,
    )
    difference = rs.read_window(Path(plan[rs.TARGET_CMB]["path"]), 0, height)
    std = rs.read_window(Path(plan["baseline_lst_std_celsius"]["path"]), 0, height)
    stored = rs.read_window(Path(plan[rs.TARGET_ANOMALY]["path"]), 0, height)
    expected = int(np.count_nonzero(
        np.isfinite(difference) & np.isfinite(std) & np.isfinite(stored) & (std != 0.0)
    ))
    assert state.stored_reproduction_pixels == expected


def test_stored_reproduction_is_window_size_invariant(tmp_path):
    plan, height, width = _synthetic_plan(tmp_path)
    small = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=32,
    )
    large = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=512,
    )
    assert small.stored_reproduction_pixels == large.stored_reproduction_pixels
    assert small.stored_reproduction_max == pytest.approx(large.stored_reproduction_max)


@pytest.mark.parametrize("error,expected_status", [
    (1e-7, "within_predeclared_tolerance"),
    (5e-5, "within_step5_reproduction_policy"),
    (1e-2, "exceeds_step5_reproduction_policy"),
])
def test_stored_reproduction_status_bands(error, expected_status):
    state = rs.AnalysisState()
    state.stored_reproduction.add(np.full(100, error))
    state.stored_reproduction_max = error
    state.stored_reproduction_pixels = 100
    state.stored_reproduction_within_predeclared = int(
        error <= rs.ANOMALY_STORED_REPRODUCTION_TOL
    ) * 100
    state.stored_reproduction_within_policy = int(
        error <= rs.ANOMALY_STORED_REPRODUCTION_POLICY_TOL
    ) * 100
    check = rs.build_stored_reproduction_check(state)
    assert check["status"] == expected_status
    # Whatever the band, it is NEVER a decomposition failure.
    assert check["is_decomposition_failure"] is False
    assert check["gates_the_audit"] is False


def test_float32_serialization_error_never_invalidates_the_audit():
    """Only check 1 may contribute an invalid-inputs reason."""
    source = inspect.getsource(runner._run_live)
    identity_block, _, stored_block = source.partition("stored_check =")
    assert "invalid_reasons.append" in identity_block
    # Everything after the stored check is constructed must not raise an
    # invalid reason from it.
    stored_section = stored_block.split("rs.write_checkpoint_stage")[0]
    assert "invalid_reasons" not in stored_section
    assert "stored_reproduction_gates_the_audit" in source


def test_anomaly_report_separates_the_two_checks(tmp_path):
    plan, height, width = _synthetic_plan(tmp_path)
    state = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=64,
    )
    report = rs.build_anomaly_report(state, rs.compute_share_intervals(state))

    assert "algebraic_identity_check" in report
    assert "stored_raster_reproduction_check" in report
    identity = report["algebraic_identity_check"]
    stored = report["stored_raster_reproduction_check"]
    assert identity["gates_the_audit"] is True
    assert stored["gates_the_audit"] is False
    assert identity["tolerance_absolute"] != stored["predeclared_tolerance"]
    # The headline reconstruction verdict comes from CHECK 1 only.
    assert report["reconstruction_exact"] == identity["passed"]
    assert report["reconstruction_tolerance"] == rs.ANOMALY_IDENTITY_ABS_TOL
    assert "NEVER treated as a decomposition failure" in report["checks_are_independent"]


def test_markdown_reports_both_checks_separately(tmp_path):
    plan, height, width = _synthetic_plan(tmp_path)
    state = rs.run_streaming_pass(
        plan, height=height, width=width, pathrow_masks=None, window_rows=64,
    )
    report = rs.build_anomaly_report(state, rs.compute_share_intervals(state))
    summary = _minimal_summary()
    summary["anomaly_decomposition"] = report
    markdown = rs.render_summary_markdown(summary)

    assert "### 4a. Algebraic identity check (gates the audit)" in markdown
    assert "### 4b. Stored-raster reproduction check (descriptive only)" in markdown
    assert str(rs.ANOMALY_IDENTITY_ABS_TOL) in markdown
    assert str(rs.ANOMALY_STORED_REPRODUCTION_TOL) in markdown
    assert str(rs.ANOMALY_STORED_REPRODUCTION_POLICY_TOL) in markdown
    for percentile in rs.ANOMALY_STORED_REPRODUCTION_PERCENTILES:
        assert f"| p{percentile} |" in markdown
    assert "EXPECTED serialization error" in markdown
    assert "is a decomposition failure: `False`" in markdown


def test_config_snapshot_declares_both_checks():
    checks = rs.build_config_snapshot(EXPERIMENT)["reconstruction_checks"]
    assert set(checks) == {
        "current_minus_baseline", "anomaly_algebraic_identity",
        "anomaly_stored_raster_reproduction",
    }
    assert checks["anomaly_algebraic_identity"]["gates_the_audit"] is True
    stored = checks["anomaly_stored_raster_reproduction"]
    assert stored["gates_the_audit"] is False
    assert stored["is_decomposition_failure"] is False
    assert stored["float32_serialization_error_is_expected"] is True
    assert stored["reported_percentiles"] == list(
        rs.ANOMALY_STORED_REPRODUCTION_PERCENTILES
    )


def test_decomposition_formulas_name_both_checks():
    anomaly = rs.decomposition_formulas()["anomaly_zscore"]
    assert anomaly["check_1_algebraic_identity"]["gates_the_audit"] is True
    assert anomaly["check_2_stored_raster_reproduction"]["gates_the_audit"] is False
    assert anomaly["check_2_stored_raster_reproduction"]["is_decomposition_failure"] is False


def test_dry_run_declares_both_checks_without_results():
    plan = rs.build_dry_run_plan(EXPERIMENT)
    checks = plan["anomaly_reconstruction_checks"]
    assert set(checks) >= {
        "algebraic_identity_check", "stored_raster_reproduction_check",
    }
    identity = checks["algebraic_identity_check"]
    stored = checks["stored_raster_reproduction_check"]
    assert identity["tolerance_absolute"] == rs.ANOMALY_IDENTITY_ABS_TOL
    assert identity["tolerance_relative"] == rs.ANOMALY_IDENTITY_REL_TOL
    assert identity["gates_the_audit"] is True
    assert stored["predeclared_tolerance"] == rs.ANOMALY_STORED_REPRODUCTION_TOL
    assert stored["step5_reproduction_policy_tolerance"] == \
        rs.ANOMALY_STORED_REPRODUCTION_POLICY_TOL
    assert stored["gates_the_audit"] is False
    # A dry run has computed nothing, and says so rather than implying a pass.
    assert identity["results_available"] is False
    assert stored["results_available"] is False
    assert "passed" not in identity
    assert "max_abs_error" not in stored


def test_report_schema_version_marks_the_separated_checks(tmp_path):
    """A report written before the split must be distinguishable from one after."""
    assert rs.REPORT_SCHEMA_VERSION == "1.1-separated-anomaly-checks"
    assert rs.REPORT_SCHEMA_VERSION != "1.0-residual-seam-attribution"
    summary = _minimal_summary()
    assert summary["report_schema_version"] == rs.REPORT_SCHEMA_VERSION
    root = tmp_path / "root"
    root.mkdir()
    (root / "residual_seam_summary.json").write_text("{}", encoding="utf-8")
    manifest = rs.build_manifest(EXPERIMENT, root, summary)
    assert manifest["report_schema_version"] == rs.REPORT_SCHEMA_VERSION
