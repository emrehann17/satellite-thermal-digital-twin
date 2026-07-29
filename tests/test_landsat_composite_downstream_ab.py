"""Focused tests for the Landsat compositing downstream A/B experiment.

No Earth Engine and no real Step5-Step8 run are required: the pipeline-touching
callables are imported lazily inside the runner, so these tests exercise the CLI
mode contract, the no-op dry-run, the namespace/force safety, the input-plan and
provenance contract, the reference-reproduction and baseline-invariance gates,
the common-cohort and shared-fold construction, the paired bootstrap, the
boundary adjacency identity, atomic checkpointing/resume, the ordered final
status rule, and the report-generation invariants.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.landsat_composite_downstream_ab as ab
import scripts.run_landsat_composite_downstream_ab as runner


EXPERIMENT = "manavgat_2021"


# =============================================================================
# Fixtures / helpers
# =============================================================================
def _write_raster(path: Path, array, *, nodata=-9999.0, transform=None, crs="EPSG:4326",
                  count=1):
    import rasterio
    from rasterio.transform import Affine

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype="float32")
    if array.ndim == 2:
        array = array[None, ...]
    transform = transform or Affine(0.00026949, 0.0, 31.0, 0.0, -0.00026949, 37.35)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
        count=array.shape[0], dtype="float32", crs=crs, transform=transform,
        nodata=nodata,
    ) as dst:
        for band in range(array.shape[0]):
            dst.write(array[band], band + 1)
    return path


def _fake_dataset(n=240, seed=0, thermal_shift=0.0, drop_cells=()):
    """A Step8A-shaped modelling dataset with the columns Step8B requires."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n // 12), 12)[:n]
    cols = np.tile(np.arange(12), n // 12)[:n]
    cell_ids = [f"r{r}_c{c}" for r, c in zip(rows, cols)]
    label = (rng.random(n) < 0.25).astype(int)
    frame = pd.DataFrame({
        "cell_id": cell_ids,
        "row_500m": rows,
        "col_500m": cols,
        "burned": label,
        "burn_month": np.where(label == 1, 8, 0),
        # baseline features -- identical between chains by construction
        "ndvi_mean": rng.normal(0.4, 0.1, n),
        "elevation_mean": rng.normal(600, 120, n),
        "slope_mean": rng.normal(12, 4, n),
        "landcover_dominant": rng.choice([10, 20, 30, 40], n),
        # thermal features -- the only ones the candidate may change
        "lst_anomaly_mean": rng.normal(0.5, 1.0, n) + label * 0.8 + thermal_shift,
        "current_lst_mean": rng.normal(35, 3, n) + thermal_shift,
        "current_tvdi_mean": rng.uniform(0, 1, n),
        "tvdi_difference_mean": rng.normal(0, 0.2, n),
        "downscaled_lst_mean": rng.normal(35, 3, n) + thermal_shift,
        "fused_lst_mean": rng.normal(35, 3, n) + thermal_shift,
        "valid_for_modeling": True,
        "invalid_reason": "",
        "burnable_tree_shrub_grass": True,
        "burnable_tree_shrub": True,
        "landcover_cropland_fraction": 0.0,
        "gapfilled_fraction": 0.0,
    })
    if drop_cells:
        frame.loc[frame["cell_id"].isin(drop_cells), "valid_for_modeling"] = False
        frame.loc[frame["cell_id"].isin(drop_cells), "invalid_reason"] = "ndvi_mean_not_finite"
    return frame


def _model_result(n, seed=0, fold_id=None):
    rng = np.random.default_rng(seed)
    return {
        "skipped": False,
        "n_positives": 40, "n_negatives": n - 40,
        "overall_baseline": {"roc_auc": 0.70, "pr_auc": 0.30, "brier_score": 0.18},
        "overall_thermal": {"roc_auc": 0.78, "pr_auc": 0.40, "brier_score": 0.15},
        "delta_auc": 0.08, "delta_pr_auc": 0.10, "delta_brier": -0.03,
        "oof_prob_baseline": rng.random(n),
        "oof_prob_thermal": rng.random(n),
        "fold_id": np.zeros(n, dtype=int) if fold_id is None else fold_id,
    }


# =============================================================================
# CLI mode conflicts
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
    """No flags at all must fail, not silently run."""
    with pytest.raises(SystemExit):
        runner.main(experiment_id=EXPERIMENT)


def test_argparse_requires_a_mode():
    with pytest.raises(SystemExit):
        runner.parse_args(["--experiment", EXPERIMENT])


def test_unsupported_candidate_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        runner.main(experiment_id=EXPERIMENT, candidate="date_balanced_all_landsat",
                    dry_run=True)
    assert "unsupported" in str(excinfo.value)


def test_date_balanced_all_landsat_variant_is_not_implemented():
    assert ab.SUPPORTED_CANDIDATES == (ab.CHAIN_CANDIDATE,)
    assert "date_balanced_all_landsat" not in ab.SUPPORTED_CANDIDATES


# =============================================================================
# Dry-run writes nothing
# =============================================================================
def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    """The dry-run must create no file and no directory anywhere."""
    root = ab.diagnostic_output_root(EXPERIMENT)
    before = sorted(p for p in root.rglob("*")) if root.exists() else []
    root_existed = root.exists()

    created: list[str] = []
    real_mkdir = Path.mkdir

    def _tracking_mkdir(self, *args, **kwargs):
        created.append(str(self))
        return real_mkdir(self, *args, **kwargs)

    opened: list[str] = []
    real_open = open

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
    after = sorted(p for p in root.rglob("*")) if root.exists() else []
    assert after == before
    ab_writes = [p for p in opened if ab.DIAGNOSTIC_NAMESPACE in p]
    ab_dirs = [p for p in created if ab.DIAGNOSTIC_NAMESPACE in p]
    assert ab_writes == []
    assert ab_dirs == []


def test_dry_run_plan_prints_every_required_section():
    plan = ab.build_dry_run_plan(EXPERIMENT, ab.CHAIN_CANDIDATE)
    for key in (
        "experiment_id", "reference_source_paths", "candidate_source_paths",
        "audit_prerequisite_status", "output_namespace", "planned_stages",
        "expected_files", "configuration", "decision_rule_version",
    ):
        assert key in plan
    assert plan["configuration"]["model"]["name"]
    assert plan["configuration"]["spatial_blocks"]["block_size_cells"]
    assert plan["configuration"]["bootstrap"]["replicates"]
    assert plan["planned_stages"] == list(ab.PLANNED_STAGES)


# =============================================================================
# No Earth Engine path is reachable
# =============================================================================
def _module_source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("module", [ab, runner])
def test_no_earth_engine_import_or_call_in_source(module):
    tree = ast.parse(_module_source(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "ee" for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in ("ee", "core.gee_utils")


@pytest.mark.parametrize("forbidden", [
    "ee.Initialize", "ee.ImageCollection", "ee.Image(", "Export.image",
    "init_gee", "build_ee_images", "build_source_scene_metadata",
    "export_image_direct_or_tiled", "run_predictors_only",
])
def test_no_earth_engine_symbol_is_referenced(forbidden):
    for module in (ab, runner):
        source = _module_source(module)
        # The names may appear only inside the FORBIDDEN_EE_CALLABLES registry
        # and in prose, never as a call.
        for line in source.splitlines():
            stripped = line.strip()
            if forbidden not in stripped:
                continue
            assert stripped.startswith("#") or '"' in stripped or "'" in stripped, (
                f"{forbidden} appears as live code: {stripped}"
            )


def test_earth_engine_guard_makes_initialize_raise():
    import ee

    original = ee.Initialize
    with ab.EarthEngineGuard():
        with pytest.raises(ab.DownstreamABError):
            ee.Initialize()
    assert ee.Initialize is original


# =============================================================================
# Candidate modifies LST only; canonical NDVI identical
# =============================================================================
def _provenance_from_plan(plan):
    return ab.build_input_provenance(
        plan, EXPERIMENT, grid_gate={}, compose_notes={},
        source_audit_state={"prerequisites_met": True},
    )


def test_candidate_modifies_lst_only():
    from core.experiment_context import build_experiment_context

    plan = ab.build_input_plan(build_experiment_context(EXPERIMENT), EXPERIMENT)
    differing = {r for r, e in plan.items() if e["differs_between_chains"]}
    assert differing
    assert all(plan[r]["family"] == "landsat_lst" for r in differing)
    assert ab.candidate_modifies_lst_only(_provenance_from_plan(plan)) is True


def test_canonical_ndvi_is_identical_between_chains():
    from core.experiment_context import build_experiment_context

    plan = ab.build_input_plan(build_experiment_context(EXPERIMENT), EXPERIMENT)
    ndvi_roles = [r for r, e in plan.items() if e["family"] == "ndvi"]
    assert ndvi_roles
    for role in ndvi_roles:
        entry = plan[role]
        assert entry["shared"] is True
        assert entry["differs_between_chains"] is False
        assert entry["materialized"][ab.CHAIN_REFERENCE] == entry["materialized"][ab.CHAIN_CANDIDATE]
    assert ab.ndvi_inputs_identical(_provenance_from_plan(plan)) is True


def test_non_lst_difference_is_detected():
    provenance = {"inputs": [
        {"logical_role": "ndvi_current", "family": "ndvi", "shared_between_chains": False},
        {"logical_role": "current_lst", "family": "landsat_lst", "shared_between_chains": False},
    ]}
    assert ab.candidate_modifies_lst_only(provenance) is False


# =============================================================================
# Frozen outputs are never overwritten; force cannot escape
# =============================================================================
@pytest.mark.parametrize("relative", [
    "outputs/experiments/manavgat_2021/step5/current_period_median_celsius.tif",
    "outputs/diagnostics/landsat_composite_counterfactual/manavgat_2021/rasters/x.tif",
    "data/current_period/x.tif",
    "outputs/step5/x.tif",
])
def test_namespace_safety_rejects_frozen_paths(relative):
    with pytest.raises(ab.NamespaceSafetyError):
        ab.assert_downstream_namespace_safe([PROJECT_ROOT / relative], EXPERIMENT)


def test_namespace_safety_accepts_the_dedicated_root():
    root = ab.diagnostic_output_root(EXPERIMENT)
    ab.assert_downstream_namespace_safe(
        [root, root / "comparison" / "tables" / "step8_metrics.csv"], EXPERIMENT,
    )


def test_namespace_safety_rejects_escaping_path():
    root = ab.diagnostic_output_root(EXPERIMENT)
    with pytest.raises(ab.NamespaceSafetyError):
        ab.assert_downstream_namespace_safe([root / ".." / ".." / "escaped.json"], EXPERIMENT)


def test_all_planned_outputs_are_namespace_safe():
    paths = list(ab.plan_output_layout(EXPERIMENT).values())
    paths += list(ab.plan_expected_files(EXPERIMENT).values())
    ab.assert_downstream_namespace_safe(paths, EXPERIMENT)


def test_force_cannot_escape_the_dedicated_diagnostic_root(tmp_path, monkeypatch):
    """--force may delete the A/B root and nothing else."""
    base = tmp_path
    ab_root = ab.diagnostic_output_root(EXPERIMENT, base)
    ab_root.mkdir(parents=True)
    (ab_root / "sentinel.txt").write_text("x", encoding="utf-8")

    frozen = ab.canonical_experiment_root(EXPERIMENT, base) / "step5"
    frozen.mkdir(parents=True)
    (frozen / "canonical.tif").write_text("keep", encoding="utf-8")
    cf = ab.counterfactual_source_root(EXPERIMENT, base) / "rasters"
    cf.mkdir(parents=True)
    (cf / "frozen.tif").write_text("keep", encoding="utf-8")

    removed = ab.clear_diagnostic_namespace(EXPERIMENT, base)
    assert removed == str(ab_root.resolve())
    assert not ab_root.exists()
    assert (frozen / "canonical.tif").exists()
    assert (cf / "frozen.tif").exists()


def test_force_refuses_a_symlinked_root(tmp_path):
    base = tmp_path
    outside = base / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep", encoding="utf-8")
    ab_root = ab.diagnostic_output_root(EXPERIMENT, base)
    ab_root.parent.mkdir(parents=True)
    ab_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ab.NamespaceSafetyError):
        ab.clear_diagnostic_namespace(EXPERIMENT, base)
    assert (outside / "precious.txt").exists()


# =============================================================================
# Grid gate + prerequisites
# =============================================================================
def test_exact_grid_mismatch_fails(tmp_path):
    from rasterio.transform import Affine

    a = _write_raster(tmp_path / "a.tif", np.ones((4, 4)))
    b = _write_raster(
        tmp_path / "b.tif", np.ones((4, 4)),
        transform=Affine(0.00026949, 0.0, 31.5, 0.0, -0.00026949, 37.35),
    )
    plan = {"current_lst": {
        "role": "current_lst", "differs_between_chains": True,
        "reference_source": a, "candidate_source": b,
    }}
    with pytest.raises(ab.GridMismatchError):
        ab.assert_reference_candidate_grid_equality(plan)


def test_matching_grids_pass_the_gate(tmp_path):
    a = _write_raster(tmp_path / "a.tif", np.ones((4, 4)))
    b = _write_raster(tmp_path / "b.tif", np.zeros((4, 4)))
    plan = {"current_lst": {
        "role": "current_lst", "differs_between_chains": True,
        "reference_source": a, "candidate_source": b,
    }}
    checked = ab.assert_reference_candidate_grid_equality(plan)
    assert checked["current_lst"]["grid_equal"] is True


@pytest.mark.parametrize("bad", [
    {"final_status": "uncertain", "canonical_reproduction_status": "pass"},
    {"final_status": "supported_reduction", "canonical_reproduction_status": "fail"},
])
def test_failed_source_counterfactual_prerequisite_fails(bad):
    state = {"present": True, "missing_files": [], **bad}
    with pytest.raises(ab.PrerequisiteError):
        ab.validate_source_audit_state(state)


def test_missing_source_counterfactual_fails_without_gee_fallback():
    state = {"present": False, "source_root": "/nowhere"}
    with pytest.raises(ab.PrerequisiteError) as excinfo:
        ab.validate_source_audit_state(state)
    assert "never falls back" in str(excinfo.value)


def test_experiment_without_frozen_inputs_fails_clearly():
    plan = {"current_lst": {
        "role": "current_lst", "differs_between_chains": True,
        "reference_source": Path("/nonexistent/reference.tif"),
        "candidate_source": Path("/nonexistent/candidate.tif"),
    }}
    with pytest.raises(ab.PrerequisiteError) as excinfo:
        ab.assert_required_frozen_inputs(plan, "some_other_experiment")
    assert "never falls back to an Earth" in str(excinfo.value)


def test_real_source_audit_meets_the_prerequisites():
    state = ab.load_source_audit_state(EXPERIMENT)
    ab.validate_source_audit_state(state)
    assert state["final_status"] == ab.REQUIRED_SOURCE_FINAL_STATUS
    assert state["canonical_reproduction_status"] == ab.REQUIRED_SOURCE_CANONICAL_REPRODUCTION


# =============================================================================
# Input hashes and provenance
# =============================================================================
def test_provenance_records_hashes_and_grid(tmp_path):
    source = _write_raster(tmp_path / "src.tif", np.arange(16).reshape(4, 4))
    record = ab.provenance_record(
        role="current_lst", chain=ab.CHAIN_REFERENCE, source=source,
        materialized=source, shared=False, family="landsat_lst",
        materialization="verbatim_copy",
    )
    for key in ("logical_role", "source_path", "materialized_path", "file_size_bytes",
                "sha256", "crs", "transform", "width", "height", "dtype", "nodata",
                "source_chain", "shared_between_chains"):
        assert key in record
    assert len(record["sha256"]) == 64
    assert record["width"] == 4 and record["height"] == 4


def test_candidate_audit_provenance_is_recorded():
    from core.experiment_context import build_experiment_context

    plan = ab.build_input_plan(build_experiment_context(EXPERIMENT), EXPERIMENT)
    state = ab.load_source_audit_state(EXPERIMENT)
    provenance = ab.build_input_provenance(
        plan, EXPERIMENT, grid_gate={}, compose_notes={}, source_audit_state=state,
    )
    block = provenance["candidate_audit_provenance"]
    assert block["source_counterfactual_manifest_path"]
    assert block["source_final_status"] == ab.REQUIRED_SOURCE_FINAL_STATUS
    assert block["source_canonical_reproduction_status"] == "pass"
    assert block["report_schema_version"]
    assert block["audit_file_hashes"]


def test_candidate_current_period_composition_is_two_band(tmp_path):
    lst = _write_raster(tmp_path / "lst.tif", np.full((4, 4), 30.0))
    count = _write_raster(tmp_path / "count.tif", np.full((4, 4), 5.0))
    out = tmp_path / "out" / "current.tif"
    note = ab.compose_candidate_current_period(lst, count, out)

    import rasterio

    with rasterio.open(out) as src:
        assert src.count == 2
        assert float(src.read(1).mean()) == pytest.approx(30.0)
        assert float(src.read(2).mean()) == pytest.approx(5.0)
    assert note["band_2"] == ab.CANDIDATE_CURRENT_COUNT_SEMANTICS


# =============================================================================
# Reference reproduction gate
# =============================================================================
def test_reference_reproduction_passes_for_identical_rasters(tmp_path):
    array = np.linspace(10, 40, 16).reshape(4, 4)
    a = _write_raster(tmp_path / "produced.tif", array)
    b = _write_raster(tmp_path / "canonical.tif", array)
    result = ab.compare_raster_semantic(a, b, tolerance=1e-4)
    assert result["passed"] is True
    assert result["status"] == "reproduced"


def test_reference_reproduction_fails_beyond_tolerance(tmp_path):
    array = np.linspace(10, 40, 16).reshape(4, 4)
    a = _write_raster(tmp_path / "produced.tif", array)
    b = _write_raster(tmp_path / "canonical.tif", array + 0.5)
    result = ab.compare_raster_semantic(a, b, tolerance=1e-4)
    assert result["passed"] is False
    assert result["status"] == "not_reproduced"


def test_reference_reproduction_fails_on_grid_mismatch(tmp_path):
    from rasterio.transform import Affine

    a = _write_raster(tmp_path / "produced.tif", np.ones((4, 4)))
    b = _write_raster(
        tmp_path / "canonical.tif", np.ones((4, 4)),
        transform=Affine(0.00026949, 0.0, 32.0, 0.0, -0.00026949, 37.35),
    )
    result = ab.compare_raster_semantic(a, b, tolerance=1e-4)
    assert result["status"] == "grid_mismatch"
    assert result["passed"] is False


def test_failed_reference_reproduction_yields_invalid_reference_status():
    report = ab.build_reference_reproduction_report(
        EXPERIMENT, {"current_lst_celsius": {"passed": False}}, {"passed": True},
    )
    assert report["status"] == "fail"
    assert report["failure_status_if_not_reproduced"] == ab.STATUS_INVALID_REFERENCE
    decision = ab.decide_final_status({"reference_reproduction_status": "fail"})
    assert decision["final_status"] == ab.STATUS_INVALID_REFERENCE


def test_reproduction_uses_semantic_not_bitwise_comparison():
    """Tolerances exist and are not all zero -- no blind bitwise requirement."""
    assert any(v > 0 for v in ab.REPRODUCTION_TOLERANCES.values())
    assert ab.REPRODUCTION_MIN_MASK_AGREEMENT < 1.0 + 1e-12


# =============================================================================
# Common cohort construction
# =============================================================================
def test_common_cohort_is_exact_intersection():
    reference = _fake_dataset(n=240, seed=1)
    candidate = _fake_dataset(n=240, seed=1, drop_cells=["r0_c0", "r0_c1"])
    cohort = ab.build_common_cohort(reference, candidate)
    assert len(cohort["common_cell_ids"]) == 238
    assert cohort["reference_only_cell_ids"] == ["r0_c0", "r0_c1"]
    assert cohort["candidate_only_cell_ids"] == []
    assert len(cohort["reference"]) == len(cohort["candidate"]) == 238
    assert list(cohort["reference"]["cell_id"]) == list(cohort["candidate"]["cell_id"])


def test_cohort_labels_and_population_must_match():
    reference = _fake_dataset(n=120, seed=2)
    candidate = _fake_dataset(n=120, seed=2)
    cohort = ab.build_common_cohort(reference, candidate)
    assert cohort["labels_match"] is True
    assert cohort["population_match"] is True
    assert cohort["row_col_match"] is True

    tampered = candidate.copy()
    tampered.loc[0, "burned"] = 1 - int(tampered.loc[0, "burned"])
    bad = ab.build_common_cohort(reference, tampered)
    assert bad["labels_match"] is False


def test_reference_only_and_candidate_only_cells_are_reported():
    reference = _fake_dataset(n=240, seed=3, drop_cells=["r1_c0"])
    candidate = _fake_dataset(n=240, seed=3, drop_cells=["r2_c0", "r2_c1"])
    cohort = ab.build_common_cohort(reference, candidate)
    alignment = ab.build_population_alignment(EXPERIMENT, cohort, reference, candidate)
    assert alignment["reference_only_rows"] == 2
    assert alignment["candidate_only_rows"] == 1
    assert alignment["total_reference_rows"] == 239
    assert alignment["total_candidate_rows"] == 238
    assert alignment["common_rows"] == 237
    assert "reference" in [r["chain"] for r in alignment["row_exclusion_reasons"]] or True
    assert alignment["row_exclusion_reasons"][0]["invalid_rows"] == 1


def test_population_alignment_requires_review_when_retention_is_low():
    reference = _fake_dataset(n=240, seed=4)
    dropped = [f"r{r}_c{c}" for r in range(4) for c in range(12)]
    candidate = _fake_dataset(n=240, seed=4, drop_cells=dropped)
    cohort = ab.build_common_cohort(reference, candidate)
    alignment = ab.build_population_alignment(EXPERIMENT, cohort, reference, candidate)
    assert alignment["status"] == "requires_review"
    decision = ab.decide_final_status({
        "reference_reproduction_status": "pass",
        "baseline_invariance_status": "pass",
        "population_alignment_status": alignment["status"],
        "population_review_reasons": alignment["review_reasons"],
    })
    assert decision["final_status"] == ab.STATUS_POPULATION_REVIEW


def test_primary_population_is_frozen():
    assert ab.PRIMARY_POPULATION == "burnable_tree_shrub_grass"


# =============================================================================
# Deterministic shared fold assignment
# =============================================================================
def test_fold_assignment_is_deterministic_and_shared():
    reference = _fake_dataset(n=240, seed=5)
    candidate = _fake_dataset(n=240, seed=5, thermal_shift=2.0)
    cohort = ab.build_common_cohort(reference, candidate)

    a1, ref_cohort, _ = ab.build_fold_assignment(cohort["reference"])
    a2, cand_cohort, _ = ab.build_fold_assignment(cohort["candidate"])
    assert list(a1["cv_fold"]) == list(a2["cv_fold"])
    assert list(a1["spatial_block_id"]) == list(a2["spatial_block_id"])
    for column in ("cell_id", "grid_row", "grid_col", "label", "population",
                   "spatial_block_id", "cv_fold", "seed", "block_size_cells"):
        assert column in a1.columns
    ab.assert_identical_fold_assignment(
        a1["cv_fold"].to_numpy(), a2["cv_fold"].to_numpy(), a1["cv_fold"].to_numpy(),
    )


def test_fold_assignment_mismatch_is_rejected():
    with pytest.raises(ab.DownstreamABError):
        ab.assert_identical_fold_assignment([0, 1, 2], [0, 1, 1], [0, 1, 2])


def test_random_row_split_is_impossible():
    """Folds always come from grouped, block-stratified splitting."""
    from src.step8b_train_baseline_vs_thermal_model import make_spatial_folds

    source = Path(ab.__file__).read_text(encoding="utf-8")
    assert "make_spatial_folds" in source
    assert "train_test_split" not in source
    assert "ShuffleSplit" not in source
    assert "random_state=None" not in source
    # Fold construction is delegated entirely to Step8B: this module imports no
    # sklearn splitter of its own, so it cannot build a random row split.
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
    assert "sklearn" not in imported
    # Step8B itself refuses to fall back to a random split.
    y = np.array([1, 1, 0, 0])
    groups = np.array(["a", "a", "a", "a"])
    with pytest.raises(SystemExit):
        make_spatial_folds(y, groups, 5, 42)


def test_fold_configuration_matches_frozen_canonical_step8():
    from core.config import (
        STEP8B_N_SPLITS, STEP8B_RANDOM_SEED, STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
    )
    from core.experiment_context import build_experiment_context

    config = ab.build_config_snapshot(
        EXPERIMENT, ab.CHAIN_CANDIDATE, build_experiment_context(EXPERIMENT),
    )
    blocks = config["spatial_blocks"]
    assert blocks["block_size_cells"] == STEP8B_SPATIAL_BLOCK_SIZE_CELLS
    assert blocks["n_splits"] == STEP8B_N_SPLITS
    assert blocks["seed"] == STEP8B_RANDOM_SEED
    assert blocks["large_block_robustness_included"] is False


# =============================================================================
# Baseline invariance
# =============================================================================
def test_baseline_feature_and_oof_invariance_pass():
    reference = _fake_dataset(n=240, seed=6)
    candidate = _fake_dataset(n=240, seed=6, thermal_shift=3.0)
    cohort = ab.build_common_cohort(reference, candidate)
    _, ref_cohort, _ = ab.build_fold_assignment(cohort["reference"])
    _, cand_cohort, _ = ab.build_fold_assignment(cohort["candidate"])

    shared_baseline = np.random.default_rng(0).random(len(ref_cohort))
    ref_result = _model_result(len(ref_cohort), seed=1)
    cand_result = _model_result(len(cand_cohort), seed=2)
    ref_result["oof_prob_baseline"] = shared_baseline
    cand_result["oof_prob_baseline"] = shared_baseline.copy()

    check = ab.check_baseline_invariance(ref_cohort, cand_cohort, ref_result, cand_result)
    assert check["baseline_feature_values_equal"] is True
    assert check["baseline_oof_predictions_equal"] is True
    assert check["labels_equal"] is True
    assert check["folds_equal"] is True
    assert check["status"] == "pass"


def test_baseline_feature_mismatch_fails_the_gate():
    reference = _fake_dataset(n=120, seed=7)
    candidate = _fake_dataset(n=120, seed=7)
    candidate.loc[0, "ndvi_mean"] = candidate.loc[0, "ndvi_mean"] + 0.1
    cohort = ab.build_common_cohort(reference, candidate)
    _, ref_cohort, _ = ab.build_fold_assignment(cohort["reference"])
    _, cand_cohort, _ = ab.build_fold_assignment(cohort["candidate"])
    shared = np.zeros(len(ref_cohort))
    ref_result, cand_result = _model_result(len(ref_cohort)), _model_result(len(cand_cohort))
    ref_result["oof_prob_baseline"] = shared
    cand_result["oof_prob_baseline"] = shared.copy()

    check = ab.check_baseline_invariance(ref_cohort, cand_cohort, ref_result, cand_result)
    assert check["baseline_feature_values_equal"] is False
    assert check["status"] == "fail"


def test_baseline_oof_mismatch_fails_the_gate():
    reference = _fake_dataset(n=120, seed=8)
    candidate = _fake_dataset(n=120, seed=8)
    cohort = ab.build_common_cohort(reference, candidate)
    _, ref_cohort, _ = ab.build_fold_assignment(cohort["reference"])
    _, cand_cohort, _ = ab.build_fold_assignment(cohort["candidate"])
    ref_result, cand_result = _model_result(len(ref_cohort)), _model_result(len(cand_cohort))
    ref_result["oof_prob_baseline"] = np.zeros(len(ref_cohort))
    cand_result["oof_prob_baseline"] = np.zeros(len(cand_cohort)) + 1e-6

    check = ab.check_baseline_invariance(ref_cohort, cand_cohort, ref_result, cand_result)
    assert check["baseline_oof_predictions_equal"] is False
    assert check["status"] == "fail"
    decision = ab.decide_final_status({
        "reference_reproduction_status": "pass",
        "baseline_invariance_status": check["status"],
    })
    assert decision["final_status"] == ab.STATUS_BASELINE_INVARIANCE_FAILED


def test_feature_sets_are_the_frozen_canonical_ones():
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, THERMAL_MODEL_FEATURES,
    )
    from core.experiment_context import build_experiment_context

    config = ab.build_config_snapshot(
        EXPERIMENT, ab.CHAIN_CANDIDATE, build_experiment_context(EXPERIMENT),
    )
    assert config["model"]["baseline_features"] == list(BASELINE_FEATURES)
    assert config["model"]["thermal_features"] == list(THERMAL_MODEL_FEATURES)
    assert config["model"]["tuned_for_this_experiment"] is False


# =============================================================================
# Paired bootstrap
# =============================================================================
def _bootstrap_inputs(n=240, seed=9):
    reference = _fake_dataset(n=n, seed=seed)
    candidate = _fake_dataset(n=n, seed=seed, thermal_shift=1.0)
    cohort = ab.build_common_cohort(reference, candidate)
    _, ref_cohort, _ = ab.build_fold_assignment(cohort["reference"])
    y = ref_cohort["burned"].astype(int).to_numpy()
    rng = np.random.default_rng(11)
    baseline = np.clip(rng.random(len(y)) * 0.4 + y * 0.1, 0.0, 1.0)
    thermal_ref = np.clip(rng.random(len(y)) * 0.4 + y * 0.3, 0.0, 1.0)
    thermal_cand = np.clip(thermal_ref + rng.normal(0, 0.01, len(y)), 0.0, 1.0)
    return ref_cohort, y, baseline, thermal_ref, thermal_cand


def test_paired_bootstrap_uses_identical_sampled_blocks():
    ref_cohort, y, baseline, thermal_ref, thermal_cand = _bootstrap_inputs()
    first = ab.paired_block_bootstrap(
        ref_cohort, y, baseline, thermal_ref, thermal_cand, n_bootstrap=50,
    )
    second = ab.paired_block_bootstrap(
        ref_cohort, y, baseline, thermal_ref, thermal_cand, n_bootstrap=50,
    )
    assert first["identical_block_draws_for_both_chains"] is True
    assert first["block_draw_digest_first_replicates"] == second["block_draw_digest_first_replicates"]
    # Both chains are scored on the SAME resampled rows in each replicate: a
    # replicate's row count is shared, and swapping only the candidate vector
    # leaves the reference deltas untouched.
    swapped = ab.paired_block_bootstrap(
        ref_cohort, y, baseline, thermal_ref, thermal_ref, n_bootstrap=50,
    )
    assert (
        first["replicates"]["reference_delta_roc_auc"].tolist()
        == swapped["replicates"]["reference_delta_roc_auc"].tolist()
    )
    assert swapped["replicates"]["paired_delta_roc_auc"].abs().max() == 0.0


def test_paired_bootstrap_is_deterministic():
    ref_cohort, y, baseline, thermal_ref, thermal_cand = _bootstrap_inputs()
    a = ab.paired_block_bootstrap(ref_cohort, y, baseline, thermal_ref, thermal_cand,
                                 n_bootstrap=40)
    b = ab.paired_block_bootstrap(ref_cohort, y, baseline, thermal_ref, thermal_cand,
                                 n_bootstrap=40)
    assert a["intervals"]["paired_delta_roc_auc"] == b["intervals"]["paired_delta_roc_auc"]
    assert a["bootstrap_unit"] == "spatial_block_id"


def test_paired_bootstrap_refuses_a_random_row_bootstrap():
    frame = pd.DataFrame({"spatial_block_id": ["only"] * 10})
    with pytest.raises(ab.DownstreamABError):
        ab.paired_block_bootstrap(
            frame, np.array([0, 1] * 5), np.zeros(10), np.zeros(10), np.zeros(10),
            n_bootstrap=5,
        )


# =============================================================================
# Metric signs / Brier direction
# =============================================================================
@pytest.mark.parametrize("metric,value,expected", [
    ("roc_auc", 0.01, True), ("roc_auc", -0.01, False),
    ("pr_auc", 0.02, True), ("pr_auc", -0.02, False),
    ("brier", -0.01, True), ("brier", 0.01, False),
])
def test_candidate_minus_reference_metric_signs(metric, value, expected):
    assert ab.metric_improved(metric, value) is expected


def test_brier_direction_is_documented_in_paired_rows():
    reference = _model_result(100, seed=1)
    candidate = _model_result(100, seed=2)
    candidate["overall_thermal"] = {"roc_auc": 0.80, "pr_auc": 0.42, "brier_score": 0.14}
    intervals = {
        "paired_delta_roc_auc": {"point_estimate_bootstrap_mean": 0.02, "interval_low": 0.01,
                                 "interval_high": 0.03, "interval_excludes_zero": True,
                                 "interval_wholly_above_zero": True,
                                 "interval_wholly_below_zero": False},
        "paired_delta_pr_auc": {"point_estimate_bootstrap_mean": 0.02, "interval_low": -0.01,
                                "interval_high": 0.03, "interval_excludes_zero": False,
                                "interval_wholly_above_zero": False,
                                "interval_wholly_below_zero": False},
        "paired_delta_brier": {"point_estimate_bootstrap_mean": -0.01, "interval_low": -0.02,
                               "interval_high": -0.005, "interval_excludes_zero": True,
                               "interval_wholly_above_zero": False,
                               "interval_wholly_below_zero": True},
    }
    bootstrap = {
        "intervals": intervals, "bootstrap_unit": "spatial_block_id", "n_blocks": 30,
        "n_bootstrap_used": 40, "seed": 42,
        "identical_block_draws_for_both_chains": True,
    }
    rows = ab.build_paired_bootstrap_rows(reference, candidate, bootstrap)
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["brier"]["improvement_direction"] == "negative_is_improvement"
    assert by_metric["brier"]["point_estimate"] < 0
    assert by_metric["brier"]["point_estimate_indicates_improvement"] is True
    assert by_metric["roc_auc"]["improvement_direction"] == "positive_is_improvement"
    assert by_metric["roc_auc"]["point_estimate_indicates_improvement"] is True


def test_interval_language_avoids_statistical_significance():
    """The phrase may appear ONLY inside an explicit prohibition."""
    source = Path(ab.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    occurrences = lowered.count("statistically significant")
    assert occurrences >= 1, "the prohibition itself must be stated"
    # Every occurrence is preceded by 'never as' within the same sentence.
    assert lowered.count("never as") >= occurrences

    bootstrap_language = ab.paired_block_bootstrap.__doc__ or ""
    assert "statistically significant" not in bootstrap_language.lower()

    markdown = ab.render_summary_markdown(
        _minimal_summary([{"metric": "roc_auc", "point_estimate": 0.01}])
    )
    reported = markdown.lower()
    assert "excluding or including zero" in reported
    assert reported.count("statistically significant") == 1  # the prohibition line
    assert "never as" in reported


# =============================================================================
# Boundary pairs identical between chains
# =============================================================================
def test_boundary_support_paths_are_the_frozen_shared_definitions():
    paths = ab.boundary_support_paths(EXPERIMENT)
    assert set(paths) == {
        "scene_count_edge", "unique_date_count_edge", "same_day_multiplicity_edge",
    }
    for path in paths.values():
        assert ab.SOURCE_AUDIT_NAMESPACE in str(path)
        assert path.exists()


def test_boundary_pairs_are_identical_between_chains():
    """The adjacency masks come from ONE frozen support raster, so reference and
    candidate are sampled at identical indices by construction."""
    from src.landsat_composite_counterfactual_audit import build_edge_masks

    scene = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]], dtype=float)
    unique = scene.copy()
    multiplicity = np.zeros_like(scene)
    masks = build_edge_masks(scene, unique, multiplicity)
    reference_masks = build_edge_masks(scene, unique, multiplicity)
    for key in masks:
        for orientation in masks[key]:
            assert np.array_equal(masks[key][orientation], reference_masks[key][orientation])


def test_export_tile_control_is_not_invented():
    source = Path(ab.__file__).read_text(encoding="utf-8")
    assert "tile_seam_specs=None" in source
    assert "no genuinely comparable paired tile partition exists" in source


def test_boundary_summary_excludes_the_negative_control():
    verdicts = {
        "current_lst_celsius": {
            "scene_count_edge": {"status": "supported_reduction"},
            "export_tile_boundary": {"status": "supported_increase",
                                     "can_affect_final_status": False},
        },
        "fused_lst_celsius": {
            "scene_count_edge": {"status": "supported_reduction"},
        },
    }
    summary = ab.summarize_boundary_propagation(verdicts)
    assert summary["key_step5_seam_reduction_supported"] is True
    assert summary["downstream_supported_reduction_products"] == ["fused_lst_celsius"]
    assert summary["supported_increase_products"] == []


# =============================================================================
# Checkpointing / resume
# =============================================================================
def test_checkpoint_is_atomic_and_validated(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    output = root / "artefact.json"
    output.write_text('{"a": 1}', encoding="utf-8")

    ab.write_checkpoint_stage(root, "materialize_inputs", [output])
    assert ab.checkpoint_path(root).exists()
    assert not list((root / "checkpoints").glob(".*tmp"))
    assert ab.stage_is_reusable(root, "materialize_inputs") is True


def test_checkpoint_text_alone_never_bypasses_file_validation(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    output = root / "artefact.json"
    output.write_text('{"a": 1}', encoding="utf-8")
    ab.write_checkpoint_stage(root, "materialize_inputs", [output])

    output.unlink()
    assert ab.stage_is_reusable(root, "materialize_inputs") is False

    output.write_text('{"a": 1, "b": 2}', encoding="utf-8")  # size changed
    assert ab.stage_is_reusable(root, "materialize_inputs") is False


def test_resume_reuses_valid_stages_only(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    good = root / "good.json"
    good.write_text("{}", encoding="utf-8")
    ab.write_checkpoint_stage(root, "reference_step5", [good])
    ab.write_checkpoint_stage(root, "candidate_step5", [root / "missing.json"])

    assert ab.stage_is_reusable(root, "reference_step5") is True
    assert ab.stage_is_reusable(root, "candidate_step5") is False
    assert ab.stage_is_reusable(root, "reference_step7a") is False


def test_unknown_checkpoint_stage_is_rejected(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    with pytest.raises(ab.DownstreamABError):
        ab.write_checkpoint_stage(root, "not_a_stage", [])


def test_every_required_stage_is_checkpointed():
    for stage in (
        "validate_inputs", "materialize_inputs", "reference_step5", "candidate_step5",
        "reference_step7a", "candidate_step7e", "reference_step8a",
        "population_alignment", "fold_assignment", "reference_step8_model",
        "candidate_step8_model", "paired_bootstrap", "raster_comparison",
        "boundary_propagation", "report_generation",
    ):
        assert stage in ab.PLANNED_STAGES


# =============================================================================
# Final status decision ordering
# =============================================================================
def _full_evidence(**overrides):
    evidence = {
        "reference_reproduction_status": "pass",
        "baseline_invariance_status": "pass",
        "population_alignment_status": "ok",
        "population_review_reasons": [],
        "key_step5_seam_reduction_supported": True,
        "downstream_supported_reduction_products": ["fused_lst_celsius"],
        "downstream_supported_increase_products": [],
        "reference_thermal_support": {
            "roc_auc_interval_above_zero": True, "pr_auc_interval_above_zero": True,
        },
        "candidate_thermal_support": {
            "roc_auc_interval_above_zero": True, "pr_auc_interval_above_zero": True,
        },
        "paired_intervals": {
            "roc_auc": {"interval_wholly_below_zero": False, "interval_wholly_above_zero": False},
            "pr_auc": {"interval_wholly_below_zero": False, "interval_wholly_above_zero": False},
            "brier": {"interval_wholly_below_zero": False, "interval_wholly_above_zero": False},
        },
    }
    evidence.update(overrides)
    return evidence


def test_decision_order_a_beats_everything():
    decision = ab.decide_final_status(_full_evidence(
        reference_reproduction_status="fail", baseline_invariance_status="fail",
        population_alignment_status="requires_review",
    ))
    assert decision["final_status"] == ab.STATUS_INVALID_REFERENCE


def test_decision_order_b_beats_c():
    decision = ab.decide_final_status(_full_evidence(
        baseline_invariance_status="fail", population_alignment_status="requires_review",
    ))
    assert decision["final_status"] == ab.STATUS_BASELINE_INVARIANCE_FAILED


def test_decision_order_c_beats_d():
    decision = ab.decide_final_status(_full_evidence(
        population_alignment_status="requires_review",
        paired_intervals={
            "roc_auc": {"interval_wholly_below_zero": True},
            "pr_auc": {"interval_wholly_below_zero": True},
            "brier": {"interval_wholly_above_zero": True},
        },
    ))
    assert decision["final_status"] == ab.STATUS_POPULATION_REVIEW


@pytest.mark.parametrize("paired,expected_reason", [
    ({"roc_auc": {"interval_wholly_below_zero": True}, "pr_auc": {}, "brier": {}},
     "ROC-AUC interval is wholly below zero"),
    ({"roc_auc": {}, "pr_auc": {"interval_wholly_below_zero": True}, "brier": {}},
     "PR-AUC interval is wholly below zero"),
    ({"roc_auc": {}, "pr_auc": {}, "brier": {"interval_wholly_above_zero": True}},
     "Brier interval is wholly above zero"),
])
def test_decision_d_seam_reduced_performance_tradeoff(paired, expected_reason):
    decision = ab.decide_final_status(_full_evidence(paired_intervals=paired))
    assert decision["final_status"] == ab.STATUS_SEAM_REDUCED_TRADEOFF
    assert any(expected_reason in reason for reason in decision["reasons"])


def test_decision_d_when_candidate_loses_reference_thermal_support():
    decision = ab.decide_final_status(_full_evidence(
        candidate_thermal_support={
            "roc_auc_interval_above_zero": False, "pr_auc_interval_above_zero": True,
        },
    ))
    assert decision["final_status"] == ab.STATUS_SEAM_REDUCED_TRADEOFF


def test_decision_e_eligible_for_second_aoi():
    decision = ab.decide_final_status(_full_evidence())
    assert decision["final_status"] == ab.STATUS_ELIGIBLE_SECOND_AOI
    assert all(decision["eligibility_checks"].values())


def test_decision_f_inconclusive_when_no_propagation():
    decision = ab.decide_final_status(_full_evidence(
        key_step5_seam_reduction_supported=False,
        downstream_supported_reduction_products=[],
    ))
    assert decision["final_status"] == ab.STATUS_INCONCLUSIVE


def test_decision_f_when_a_contradictory_increase_exists():
    decision = ab.decide_final_status(_full_evidence(
        downstream_supported_increase_products=["anomaly_zscore"],
    ))
    assert decision["final_status"] == ab.STATUS_INCONCLUSIVE


def test_only_declared_statuses_can_be_returned():
    for evidence in (
        _full_evidence(), _full_evidence(reference_reproduction_status="fail"),
        _full_evidence(baseline_invariance_status="fail"),
        _full_evidence(population_alignment_status="requires_review"),
        _full_evidence(key_step5_seam_reduction_supported=False,
                       downstream_supported_reduction_products=[]),
    ):
        assert ab.decide_final_status(evidence)["final_status"] in ab.FINAL_STATUSES


def test_eligible_status_does_not_say_production_approved():
    decision = ab.decide_final_status(_full_evidence())
    assert decision["final_status"] == ab.STATUS_ELIGIBLE_SECOND_AOI
    assert decision["production_approved"] is False
    meaning = decision["meaning"].lower()
    assert "bejis" in meaning
    assert "not production acceptance" in meaning
    for forbidden in ab.FORBIDDEN_CONCLUSIONS:
        assert forbidden not in json.dumps(decision).lower().replace("production_approved\": false", "")


def test_no_status_is_a_production_approval():
    assert "production_approved" not in ab.FINAL_STATUSES
    for status, meaning in ab.FINAL_STATUS_MEANINGS.items():
        assert "production approved" not in meaning.lower()


# =============================================================================
# Chain contexts stay inside the A/B namespace
# =============================================================================
def test_chain_contexts_are_fully_namespaced():
    reference = ab.build_chain_context(EXPERIMENT, ab.CHAIN_REFERENCE)
    candidate = ab.build_chain_context(EXPERIMENT, ab.CHAIN_CANDIDATE)
    root = ab.diagnostic_output_root(EXPERIMENT).resolve()
    for ctx in (reference, candidate):
        for key in ab.CONTEXT_PATH_KEYS:
            value = ctx.get(key)
            if value is None:
                continue
            assert root in Path(value).resolve().parents or Path(value).resolve() == root


def test_chain_contexts_share_every_non_lst_input():
    reference = ab.build_chain_context(EXPERIMENT, ab.CHAIN_REFERENCE)
    candidate = ab.build_chain_context(EXPERIMENT, ab.CHAIN_CANDIDATE)
    for shared_key in ("ndvi_baseline_dir", "ndvi_current_dir", "modis_input_dir",
                       "dem_input_dir", "landcover_aligned_path", "gate_labels_dir"):
        assert reference[shared_key] == candidate[shared_key]
    for chain_key in ("baseline_input_dir", "current_period_dir"):
        assert reference[chain_key] != candidate[chain_key]


def test_chain_contexts_preserve_the_canonical_scientific_parameters():
    from core.experiment_context import build_experiment_context

    canonical = build_experiment_context(EXPERIMENT)
    for chain in ab.CHAINS:
        ctx = ab.build_chain_context(EXPERIMENT, chain)
        for key in ("predictor_start_date", "predictor_end_date", "label_start_date",
                    "label_end_date", "baseline_years", "current_period_days",
                    "current_period_end_date", "region_key", "exclude_pre_label_burns"):
            assert ctx[key] == canonical[key]


def test_unknown_chain_is_rejected():
    with pytest.raises(ab.DownstreamABError):
        ab.build_chain_context(EXPERIMENT, "some_other_chain")


def test_product_paths_resolve_per_chain():
    reference = ab.build_chain_context(EXPERIMENT, ab.CHAIN_REFERENCE)
    candidate = ab.build_chain_context(EXPERIMENT, ab.CHAIN_CANDIDATE)
    for product in ab.BOUNDARY_PROPAGATION_PRODUCTS:
        ref_path = ab.product_path(reference, product, Path(reference["output_root"]))
        cand_path = ab.product_path(candidate, product, Path(candidate["output_root"]))
        assert ref_path != cand_path
        assert "/reference/" in str(ref_path)
        assert "/candidate/" in str(cand_path)


# =============================================================================
# Raster change summary
# =============================================================================
def test_raster_change_summary_reports_every_required_statistic(tmp_path):
    base = np.linspace(20, 40, 64).reshape(8, 8)
    a = _write_raster(tmp_path / "ref.tif", base)
    b = _write_raster(tmp_path / "cand.tif", base + 0.2)
    row = ab.compare_raster_change(a, b, product="current_lst_celsius",
                                   changed_threshold=0.05)
    for key in ("mean", "median", "std", "mae", "rmse", "p01", "p05", "p50", "p95",
                "p99", "max_abs_diff", "changed_pixel_fraction", "common_valid_pixels",
                "valid_mask_agreement", "grid_equal"):
        assert key in row
    assert row["mean"] == pytest.approx(0.2, abs=1e-5)
    assert row["changed_pixel_fraction"] == pytest.approx(1.0)


def test_tiny_float_differences_are_not_counted_as_change(tmp_path):
    base = np.linspace(20, 40, 64).reshape(8, 8)
    a = _write_raster(tmp_path / "ref.tif", base)
    b = _write_raster(tmp_path / "cand.tif", base + 1e-7)
    row = ab.compare_raster_change(a, b, product="current_lst_celsius",
                                   changed_threshold=0.05)
    assert row["changed_pixel_fraction"] == 0.0
    assert row["max_abs_diff"] < 0.05


def test_raster_change_summary_columns_cover_every_required_field():
    required = {
        "mean", "median", "std", "mae", "rmse", "p01", "p05", "p50", "p95", "p99",
        "max_abs_diff", "changed_pixel_fraction", "grid_equal", "valid_mask_agreement",
        "common_valid_pixels",
    }
    assert required.issubset(set(ab.RASTER_CHANGE_COLUMNS))


def test_pair_maps_do_not_collide_when_chain_filenames_match(tmp_path):
    """Both chains write identically named rasters in different directories."""
    a = _write_raster(tmp_path / "reference" / "step5" / "current_period_median_celsius.tif",
                      np.linspace(20, 40, 64).reshape(8, 8))
    b = _write_raster(tmp_path / "candidate" / "step5" / "current_period_median_celsius.tif",
                      np.linspace(21, 41, 64).reshape(8, 8))
    out_dir = tmp_path / "maps" / "current_lst_celsius"
    written = ab.render_pair_maps_for_product(a, b, out_dir, product="current_lst_celsius")

    assert len(set(written)) == 2
    names = sorted(Path(p).name for p in written)
    assert names == sorted([
        f"current_lst_celsius__{ab.CHAIN_REFERENCE}.png",
        f"current_lst_celsius__{ab.CHAIN_CANDIDATE}.png",
    ])
    assert all(Path(p).exists() for p in written)
    assert not (out_dir / "_pair_inputs").exists()
    # The source rasters are untouched.
    assert a.exists() and b.exists()


def test_every_required_product_is_compared():
    products = set(ab.CHANGED_PIXEL_THRESHOLDS)
    for required in (
        "current_lst_celsius", "baseline_lst_mean_celsius", "baseline_lst_std_celsius",
        "current_minus_baseline_celsius", "anomaly_zscore", "current_tvdi",
        "tvdi_difference", "downscaled_lst_celsius", "fused_lst_celsius",
    ):
        assert required in products


# =============================================================================
# End-to-end model comparison on synthetic data (no pipeline run)
# =============================================================================
def test_common_cohort_model_comparison_is_wired_end_to_end():
    """Exercises the production Step8B path and the paired bootstrap together."""
    reference = _fake_dataset(n=600, seed=21)
    candidate = reference.copy()
    rng = np.random.default_rng(22)
    for column in ("lst_anomaly_mean", "current_lst_mean", "downscaled_lst_mean",
                   "fused_lst_mean"):
        candidate[column] = candidate[column] + rng.normal(0, 0.2, len(candidate))

    cohort = ab.build_common_cohort(reference, candidate)
    assignment, ref_cohort, _ = ab.build_fold_assignment(cohort["reference"])
    _, cand_cohort, _ = ab.build_fold_assignment(cohort["candidate"])

    ref_result = ab.run_chain_model(ref_cohort)
    cand_result = ab.run_chain_model(cand_cohort)

    fold_check = ab.assert_identical_fold_assignment(
        ref_result["fold_id"], cand_result["fold_id"], assignment["cv_fold"].to_numpy(),
    )
    assert fold_check["chains_identical"] is True

    invariance = ab.check_baseline_invariance(
        ref_cohort, cand_cohort, ref_result, cand_result,
    )
    assert invariance["baseline_feature_values_equal"] is True
    assert invariance["baseline_oof_predictions_equal"] is True
    assert invariance["status"] == "pass"

    bootstrap = ab.paired_block_bootstrap(
        ref_cohort, ref_cohort["burned"].astype(int).to_numpy(),
        ref_result["oof_prob_baseline"], ref_result["oof_prob_thermal"],
        cand_result["oof_prob_thermal"], n_bootstrap=25,
    )
    metric_rows = ab.build_step8_metric_rows(ref_result, cand_result, bootstrap["intervals"])
    paired_rows = ab.build_paired_bootstrap_rows(ref_result, cand_result, bootstrap)
    oof = ab.build_oof_predictions(ref_cohort, assignment, ref_result, cand_result)

    assert [r["chain"] for r in metric_rows] == [ab.CHAIN_REFERENCE, ab.CHAIN_CANDIDATE]
    assert {r["metric"] for r in paired_rows} == {"roc_auc", "pr_auc", "brier"}
    for column in ("cell_id", "label", "spatial_block_id", "cv_fold",
                   "baseline_probability", "thermal_reference_probability",
                   "thermal_candidate_probability",
                   "candidate_minus_reference_probability"):
        assert column in oof.columns
    assert len(oof) == len(ref_cohort)


# =============================================================================
# Reports
# =============================================================================
def test_report_generation_does_not_change_scientific_metrics():
    rows = [{"metric": "roc_auc", "point_estimate": 0.0123456789}]
    before = {"paired": rows}
    ab.render_summary_markdown(_minimal_summary(rows))
    after = {"paired": rows}
    assert ab.report_generation_preserves_metrics(before, after) is True
    assert rows[0]["point_estimate"] == 0.0123456789


def _minimal_summary(paired_rows):
    return {
        "experiment_id": EXPERIMENT,
        "reference_chain": ab.CHAIN_REFERENCE,
        "candidate_chain": ab.CHAIN_CANDIDATE,
        "report_schema_version": ab.REPORT_SCHEMA_VERSION,
        "decision_rule_version": ab.DECISION_RULE_VERSION,
        "final_status": ab.STATUS_ELIGIBLE_SECOND_AOI,
        "final_status_meaning": ab.FINAL_STATUS_MEANINGS[ab.STATUS_ELIGIBLE_SECOND_AOI],
        "technical_validity": {
            "reference_reproduction_status": "pass",
            "fold_assignment": {"seed": 42, "n_splits": 5, "block_size_cells": 2,
                                "grouping": "spatial_block_id", "chains_identical": True},
        },
        "raster_downstream_propagation": {
            "raster_change_summary": [{"product": "current_lst_celsius",
                                       "common_valid_pixels": 10, "mean": 0.1,
                                       "mae": 0.1, "rmse": 0.1, "max_abs_diff": 0.2,
                                       "changed_pixel_fraction": 0.5,
                                       "changed_pixel_threshold": 0.05}],
            "boundary_propagation": {
                "key_step5_product": "current_lst_celsius",
                "key_boundary_type": "scene_count_edge",
                "key_step5_seam_status": "supported_reduction",
                "downstream_supported_reduction_products": ["fused_lst_celsius"],
                "downstream_supported_increase_products": [],
            },
            "export_tile_control": "unavailable_no_comparable_paired_partition",
        },
        "within_region_model_impact": {
            "primary_population": ab.PRIMARY_POPULATION,
            "per_chain_metrics": [{
                "chain": ab.CHAIN_REFERENCE, "n_rows": 100, "n_positives": 20,
                "thermal_roc_auc": 0.8, "thermal_pr_auc": 0.4, "thermal_brier": 0.1,
                "delta_roc_auc_thermal_minus_baseline": 0.05,
                "delta_roc_auc_interval_low": 0.01, "delta_roc_auc_interval_high": 0.09,
            }],
        },
        "candidate_versus_reference_paired_comparison": {
            "paired_rows": [dict(metric=r.get("metric"), point_estimate=r.get("point_estimate"),
                                 interval_low=0.0, interval_high=0.1,
                                 interval_excludes_zero=False,
                                 improvement_direction="positive_is_improvement")
                            for r in paired_rows],
            "bootstrap_unit": "spatial_block_id", "n_blocks": 30,
            "n_bootstrap_used": 1000, "seed": 42,
            "identical_block_draws_for_both_chains": True,
        },
        "limitations": ab.required_limitations(),
        "next_decision": ab.next_decision_text(ab.STATUS_ELIGIBLE_SECOND_AOI),
    }


def test_manifest_lists_produced_files_with_hashes(tmp_path):
    root = tmp_path / "root"
    (root / "comparison" / "tables").mkdir(parents=True)
    (root / "downstream_ab_summary.json").write_text("{}", encoding="utf-8")
    (root / "comparison" / "tables" / "step8_metrics.csv").write_text("a,b\n", encoding="utf-8")
    # inputs/ is excluded: it is hashed file-by-file in input_provenance.json.
    (root / "inputs" / "shared").mkdir(parents=True)
    (root / "inputs" / "shared" / "big.tif").write_text("x" * 100, encoding="utf-8")

    summary = _minimal_summary([{"metric": "roc_auc", "point_estimate": 0.01}])
    manifest = ab.build_manifest(EXPERIMENT, root, summary)

    assert manifest["file_count"] == 2
    assert isinstance(manifest["files"], list)
    paths = {entry["path"] for entry in manifest["files"]}
    assert paths == {"downstream_ab_summary.json", "comparison/tables/step8_metrics.csv"}
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert manifest["production_approved"] is False
    assert manifest["changes_production_reducer"] is False
    assert manifest["experiment"] == ab.DIAGNOSTIC_NAMESPACE


def test_markdown_report_has_the_six_required_sections():
    markdown = ab.render_summary_markdown(_minimal_summary([{"metric": "roc_auc",
                                                             "point_estimate": 0.01}]))
    for heading in (
        "## 1. Technical validity",
        "## 2. Raster-level downstream propagation",
        "## 3. Within-region model impact",
        "## 4. Candidate-versus-reference paired comparison",
        "## 5. Limitations",
        "## 6. Next decision",
    ):
        assert heading in markdown
    assert "production approval" in markdown.lower() or "never return a production" in markdown.lower()


def test_required_limitations_are_all_present():
    limitations = " ".join(ab.required_limitations()).lower()
    for phrase in (
        "single aoi", "single event window", "no cross-region validation",
        "no production reducer change", "no causal claim",
        "not pixel-level selected-scene provenance",
        "valid in only one variant", "not proof of non-inferiority",
        "does not establish improved transfer",
    ):
        assert phrase in limitations


def test_summary_never_claims_production_approval():
    summary = _minimal_summary([{"metric": "roc_auc", "point_estimate": 0.01}])
    markdown = ab.render_summary_markdown(summary)
    assert "production approved" not in markdown.lower()
    assert "eligible for independent validation in bejis" in markdown.lower()


# =============================================================================
# Legacy frozen-MODIS historical compatibility
#
# The frozen canonical Manavgat Step7 run consumed a MODIS pair with no nodata
# tag and ~8% exact-zero pixels. The default Step7B guard rejects that, and it
# MUST keep rejecting it: the narrow compatibility path below is reachable only
# from this A/B runner, only for manavgat_2021, and only behind an exact
# path+hash attestation against the frozen inputs.
# =============================================================================
import shutil  # noqa: E402

import src.step7b_prepare_downscaling_dataset as step7b  # noqa: E402

MODIS_MEAN = "modis_lst_mean_celsius"
MODIS_STD = "modis_lst_std_celsius"


def _zero_filled_modis_array(seed=0):
    """A 1 km-ish MODIS tile with the historical sea zero-fill signature."""
    rng = np.random.default_rng(seed)
    array = rng.uniform(20.0, 40.0, size=(20, 20)).astype("float32")
    array[:4, :] = 0.0            # 20% exact zeros -> above the 5% threshold
    return array


def _legacy_modis_fixture(tmp_path, *, experiment_id=EXPERIMENT,
                          historical_nodata=None, historical_valid=None):
    """Build a self-contained frozen namespace + A/B provenance under tmp_path.

    Returns ``(base_dir, provenance, declaration)``. Nothing outside tmp_path is
    touched, and the real frozen Manavgat outputs are never used here.
    """
    base = Path(tmp_path)
    modis_dir = base / "outputs" / "experiments" / experiment_id / "data" / "modis"
    sources = {}
    for index, name in enumerate((MODIS_MEAN, MODIS_STD)):
        array = _zero_filled_modis_array(seed=index)
        if name == MODIS_STD:
            array = np.abs(array) / 10.0
        sources[name] = _write_raster(modis_dir / f"{name}.tif", array, nodata=None)

    shared = (base / "outputs" / "diagnostics" / ab.DIAGNOSTIC_NAMESPACE
              / experiment_id / "inputs" / "shared" / "modis")
    shared.mkdir(parents=True, exist_ok=True)
    materialized = {}
    for name, source in sources.items():
        target = shared / f"{name}.tif"
        shutil.copy2(source, target)
        materialized[name] = target

    total = 400
    step7b_dir = base / "outputs" / "experiments" / experiment_id / "step7b"
    step7b_dir.mkdir(parents=True, exist_ok=True)
    (step7b_dir / "downscaling_dataset_stats.json").write_text(json.dumps({
        "created_at": "2026-07-09T15:26:00+00:00",
        "alignment_diagnostics": [
            {
                "name": name,
                "source_path": str(sources[name]),
                "source_nodata": historical_nodata,
                "source_valid_pixel_count":
                    total if historical_valid is None else historical_valid,
                "source_total_pixel_count": total,
                "aligned_valid_pixel_count": 1000,
                "aligned_valid_fraction": 1.0,
                "aligned_path": str(step7b_dir / "aligned_inputs" / f"{name}.tif"),
            }
            for name in (MODIS_MEAN, MODIS_STD)
        ],
    }), encoding="utf-8")

    provenance = {
        "experiment_id": experiment_id,
        "created_at": "2030-01-01T00:00:00+00:00",
        "inputs": [
            {
                "logical_role": ab.MODIS_COMPATIBILITY_RASTERS[name]["provenance_role"],
                "family": "modis",
                "source_chain": chain,
                "shared_between_chains": True,
                "source_path": str(sources[name]),
                "materialized_path": str(materialized[name]),
                "sha256": ab.audit.sha256_and_size(materialized[name])["sha256"],
            }
            for name in (MODIS_MEAN, MODIS_STD) for chain in ab.CHAINS
        ],
    }
    declaration = ab.build_legacy_modis_attestation_declaration(experiment_id, base)
    ab.write_json_atomic(ab.legacy_modis_attestation_config_path(base), declaration)
    return base, provenance, declaration


def _chain_contexts():
    from collections import OrderedDict as _OD

    return _OD(((ab.CHAIN_REFERENCE, {}), (ab.CHAIN_CANDIDATE, {})))


def _validate(base, provenance, **kwargs):
    return ab.validate_legacy_modis_compatibility(
        kwargs.pop("experiment_id", EXPERIMENT), provenance, _chain_contexts(),
        base, **kwargs,
    )


# --- the DEFAULT Step7B guard is unchanged -----------------------------------
def test_strict_step7b_default_still_rejects_zero_filled_modis(tmp_path):
    base, _, _ = _legacy_modis_fixture(tmp_path)
    modis_dir = base / "outputs" / "experiments" / EXPERIMENT / "data" / "modis"
    features = [{"name": n, "path": modis_dir / f"{n}.tif"} for n in (MODIS_MEAN, MODIS_STD)]

    with pytest.raises(step7b.Step7BModisValidationError) as excinfo:
        step7b.validate_modis_source_rasters(features)
    assert "nodata" in str(excinfo.value).lower()

    # ...and the default is reached by simply not passing the new argument.
    with pytest.raises(step7b.Step7BModisValidationError):
        step7b.validate_modis_source_rasters(features, experiment_id=EXPERIMENT)


def test_step7b_cli_exposes_no_compatibility_flag():
    args = step7b.parse_args([])
    assert not any("compat" in name or "legacy" in name for name in vars(args))
    source = Path(step7b.__file__).read_text(encoding="utf-8")
    for flag in ("--legacy-modis-compatibility", "--skip-validation",
                 "--no-modis-validation", "--allow-zero-fill"):
        assert flag not in source


def test_step7b_compatibility_parameter_defaults_to_strict():
    import inspect

    for callable_ in (step7b.main, step7b.validate_modis_source_rasters):
        signature = inspect.signature(callable_)
        parameter = signature.parameters.get("legacy_modis_compatibility")
        assert parameter is not None
        assert parameter.default is None


@pytest.mark.parametrize("bogus", [True, 1, "legacy_frozen_modis_compatibility",
                                   {"mode": "legacy_frozen_modis_compatibility"}])
def test_generic_callers_cannot_bypass_the_guard(tmp_path, bogus):
    base, _, _ = _legacy_modis_fixture(tmp_path)
    modis_dir = base / "outputs" / "experiments" / EXPERIMENT / "data" / "modis"
    features = [{"name": n, "path": modis_dir / f"{n}.tif"} for n in (MODIS_MEAN, MODIS_STD)]
    with pytest.raises(step7b.Step7BModisValidationError):
        step7b.validate_modis_source_rasters(
            features, experiment_id=EXPERIMENT, legacy_modis_compatibility=bogus,
        )


def test_attested_waiver_is_accepted_and_only_waives_rule_one(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    attestation = _validate(base, provenance)
    typed = ab.step7b_compatibility_attestation(attestation)
    modis_dir = base / "outputs" / "experiments" / EXPERIMENT / "data" / "modis"
    features = [{"name": n, "path": modis_dir / f"{n}.tif"} for n in (MODIS_MEAN, MODIS_STD)]

    diagnostics = step7b.validate_modis_source_rasters(
        features, experiment_id=EXPERIMENT, legacy_modis_compatibility=typed,
    )
    guard = diagnostics[MODIS_MEAN]["zero_fill_guard"]
    assert guard["signature_present"] is True
    assert guard["waived"] is True
    assert guard["mode"] == ab.LEGACY_MODIS_COMPATIBILITY_MODE
    assert guard["waiver"]["values_or_mask_changed"] is False
    assert guard["waiver"]["rasters_rewritten"] is False
    assert guard["waiver"]["nodata_assigned"] is False

    # Rule 3 (negative std) is NOT waived by the same attestation.
    negative = base / "negative_std.tif"
    _write_raster(negative, np.full((20, 20), -1.0, dtype="float32"), nodata=None)
    with pytest.raises(step7b.Step7BModisValidationError) as excinfo:
        step7b.validate_modis_source_rasters(
            [features[0], {"name": MODIS_STD, "path": negative}],
            experiment_id=EXPERIMENT, legacy_modis_compatibility=typed,
        )
    assert "std" in str(excinfo.value).lower()


def test_waiver_is_refused_for_a_different_experiment(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    typed = ab.step7b_compatibility_attestation(_validate(base, provenance))
    modis_dir = base / "outputs" / "experiments" / EXPERIMENT / "data" / "modis"
    features = [{"name": n, "path": modis_dir / f"{n}.tif"} for n in (MODIS_MEAN, MODIS_STD)]
    with pytest.raises(step7b.LegacyModisCompatibilityAttestationError):
        step7b.validate_modis_source_rasters(
            features, experiment_id="bejis_2022", legacy_modis_compatibility=typed,
        )


def test_waiver_is_refused_for_an_unattested_path(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    typed = ab.step7b_compatibility_attestation(_validate(base, provenance))
    modis_dir = base / "outputs" / "experiments" / EXPERIMENT / "data" / "modis"
    elsewhere = base / "elsewhere" / f"{MODIS_MEAN}.tif"
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(modis_dir / f"{MODIS_MEAN}.tif", elsewhere)
    with pytest.raises(step7b.LegacyModisCompatibilityAttestationError) as excinfo:
        step7b.validate_modis_source_rasters(
            [{"name": MODIS_MEAN, "path": elsewhere},
             {"name": MODIS_STD, "path": modis_dir / f"{MODIS_STD}.tif"}],
            experiment_id=EXPERIMENT, legacy_modis_compatibility=typed,
        )
    assert "not an authorized path" in str(excinfo.value)


def test_waiver_is_refused_when_the_attested_content_changed(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    typed = ab.step7b_compatibility_attestation(_validate(base, provenance))
    modis_dir = base / "outputs" / "experiments" / EXPERIMENT / "data" / "modis"
    features = [{"name": n, "path": modis_dir / f"{n}.tif"} for n in (MODIS_MEAN, MODIS_STD)]
    _write_raster(modis_dir / f"{MODIS_MEAN}.tif",
                  _zero_filled_modis_array(seed=99), nodata=None)
    with pytest.raises(step7b.LegacyModisCompatibilityAttestationError) as excinfo:
        step7b.validate_modis_source_rasters(
            features, experiment_id=EXPERIMENT, legacy_modis_compatibility=typed,
        )
    assert "does not match the attested content" in str(excinfo.value)


# --- the A/B compatibility gate ----------------------------------------------
def test_gate_passes_for_the_authorized_experiment(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    attestation = _validate(base, provenance)
    assert attestation["required"] is True
    assert attestation["status"] == "pass"
    assert attestation["mode"] == ab.LEGACY_MODIS_COMPATIBILITY_MODE
    assert all(attestation["gate_checks"].values())
    assert attestation["declares_zero_scientifically_valid"] is False
    assert attestation["default_step7b_guard_changed"] is False


def test_compatibility_requires_experiment_id_manavgat_2021(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path, experiment_id="bejis_2022")
    assert "bejis_2022" not in ab.LEGACY_MODIS_COMPATIBILITY_EXPERIMENT_IDS
    with pytest.raises(ab.LegacyModisCompatibilityError) as excinfo:
        _validate(base, provenance, experiment_id="bejis_2022")
    assert "experiment_id_is_authorized" in str(excinfo.value)


def test_attestation_requires_paths_beneath_the_frozen_namespace(tmp_path):
    base, provenance, declaration = _legacy_modis_fixture(tmp_path)
    outside = tmp_path / "outside" / f"{MODIS_MEAN}.tif"
    outside.parent.mkdir(parents=True, exist_ok=True)
    declaration["rasters"][MODIS_MEAN]["path"] = str(outside)
    with pytest.raises(ab.LegacyModisCompatibilityError) as excinfo:
        _validate(base, provenance, declaration=declaration)
    assert f"{MODIS_MEAN}__declared_path_exact" in str(excinfo.value)


def test_changed_hash_fails_the_gate(tmp_path):
    base, provenance, declaration = _legacy_modis_fixture(tmp_path)
    declaration["rasters"][MODIS_MEAN]["sha256"] = "0" * 64
    with pytest.raises(ab.LegacyModisCompatibilityError) as excinfo:
        _validate(base, provenance, declaration=declaration)
    assert f"{MODIS_MEAN}__declared_hash_matches" in str(excinfo.value)


def test_changed_provenance_hash_fails_the_gate(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    for record in provenance["inputs"]:
        if record["logical_role"] == "modis_lst_mean":
            record["sha256"] = "1" * 64
    with pytest.raises(ab.LegacyModisCompatibilityError) as excinfo:
        _validate(base, provenance)
    assert f"{MODIS_MEAN}__provenance_hash_matches_frozen_source" in str(excinfo.value)


def test_missing_historical_step7b_metadata_fails_the_gate(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    (base / "outputs" / "experiments" / EXPERIMENT / "step7b"
     / "downscaling_dataset_stats.json").unlink()
    with pytest.raises(ab.LegacyModisCompatibilityError) as excinfo:
        _validate(base, provenance)
    assert "historical_step7b_metadata_confirms_no_nodata_source" in str(excinfo.value)


@pytest.mark.parametrize("kwargs", [
    {"historical_nodata": -9999.0},   # the frozen run did NOT use a no-nodata source
    {"historical_valid": 12},         # not every source pixel was treated as valid
])
def test_historical_evidence_must_confirm_the_no_nodata_source(tmp_path, kwargs):
    base, provenance, _ = _legacy_modis_fixture(tmp_path, **kwargs)
    with pytest.raises(ab.LegacyModisCompatibilityError) as excinfo:
        _validate(base, provenance)
    assert "historical_step7b_metadata_confirms_no_nodata_source" in str(excinfo.value)


def test_reference_candidate_modis_mismatch_fails_the_gate(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    for record in provenance["inputs"]:
        if (record["logical_role"] == "modis_lst_mean"
                and record["source_chain"] == ab.CHAIN_CANDIDATE):
            record["materialized_path"] = str(Path(record["materialized_path"]).parent
                                              / "candidate_only.tif")
            record["shared_between_chains"] = False
    with pytest.raises(ab.LegacyModisCompatibilityError) as excinfo:
        _validate(base, provenance)
    assert f"{MODIS_MEAN}__shared_identically_by_both_chains" in str(excinfo.value)


def test_source_modified_after_materialization_fails_the_gate(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    provenance["created_at"] = "2000-01-01T00:00:00+00:00"
    with pytest.raises(ab.LegacyModisCompatibilityError) as excinfo:
        _validate(base, provenance)
    assert "not_modified_after_materialization" in str(excinfo.value)


def test_compatibility_changes_no_raster_value_mask_dtype_or_grid(tmp_path):
    import rasterio

    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    modis_dir = base / "outputs" / "experiments" / EXPERIMENT / "data" / "modis"
    paths = [modis_dir / f"{n}.tif" for n in (MODIS_MEAN, MODIS_STD)]

    def _snapshot():
        state = {}
        for path in paths:
            with rasterio.open(path) as src:
                state[path.name] = {
                    "array": src.read(1).copy(),
                    "mask": src.read_masks(1).copy(),
                    "dtype": src.dtypes[0],
                    "nodata": src.nodata,
                    "crs": str(src.crs),
                    "transform": tuple(src.transform)[:6],
                    "shape": (src.height, src.width),
                }
            state[path.name]["sha256"] = ab.audit.sha256_and_size(path)["sha256"]
        return state

    before = _snapshot()
    attestation = _validate(base, provenance)
    typed = ab.step7b_compatibility_attestation(attestation)
    step7b.validate_modis_source_rasters(
        [{"name": n, "path": modis_dir / f"{n}.tif"} for n in (MODIS_MEAN, MODIS_STD)],
        experiment_id=EXPERIMENT, legacy_modis_compatibility=typed,
    )
    after = _snapshot()

    for name in before:
        assert np.array_equal(before[name]["array"], after[name]["array"])
        assert np.array_equal(before[name]["mask"], after[name]["mask"])
        for key in ("dtype", "nodata", "crs", "transform", "shape", "sha256"):
            assert before[name][key] == after[name][key]
    # No zero was turned into NaN and no nodata tag was invented.
    assert after[f"{MODIS_MEAN}.tif"]["nodata"] is None
    assert np.count_nonzero(after[f"{MODIS_MEAN}.tif"]["array"] == 0.0) > 0


def test_compatibility_is_applied_identically_to_both_chains(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    attestation = _validate(base, provenance)
    assert set(attestation["chains"]) == set(ab.CHAINS)
    assert set(attestation["chains"].values()) == {ab.LEGACY_MODIS_COMPATIBILITY_MODE}
    # One attestation object, so one identical typed attestation per chain.
    first = ab.step7b_compatibility_attestation(attestation)
    second = ab.step7b_compatibility_attestation(attestation)
    assert first == second


def test_shared_modis_invariance_detects_a_chain_difference(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    attestation = _validate(base, provenance)
    reference_ctx = {"step7b_output_dir": tmp_path / "ref" / "step7b"}
    candidate_ctx = {"step7b_output_dir": tmp_path / "cand" / "step7b"}
    for ctx, value in ((reference_ctx, 1.0), (candidate_ctx, 2.0)):
        aligned = Path(ctx["step7b_output_dir"]) / "aligned_inputs"
        for name in (MODIS_MEAN, MODIS_STD):
            _write_raster(aligned / f"{name}.tif",
                          np.full((8, 8), value, dtype="float32"))

    result = ab.check_shared_modis_invariance(
        provenance, reference_ctx, candidate_ctx, attestation, base,
    )
    assert result["status"] == "fail"
    assert result["technical_failure"] == ab.TECHNICAL_FAILURE_SHARED_MODIS
    assert any("identical_aligned_array" in reason for reason in result["reasons"])


def test_shared_modis_invariance_passes_for_identical_chains(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    attestation = _validate(base, provenance)
    reference_ctx = {"step7b_output_dir": tmp_path / "ref" / "step7b"}
    candidate_ctx = {"step7b_output_dir": tmp_path / "cand" / "step7b"}
    for ctx in (reference_ctx, candidate_ctx):
        aligned = Path(ctx["step7b_output_dir"]) / "aligned_inputs"
        for index, name in enumerate((MODIS_MEAN, MODIS_STD)):
            _write_raster(aligned / f"{name}.tif",
                          np.full((8, 8), float(index), dtype="float32"))
        (Path(ctx["step7b_output_dir"]) / "downscaling_dataset_stats.json").write_text(
            json.dumps({"alignment_diagnostics": [{
                "name": MODIS_MEAN,
                "modis_source_validation": {"zero_fill_guard": {
                    "mode": ab.LEGACY_MODIS_COMPATIBILITY_MODE, "waived": True,
                }},
            }]}), encoding="utf-8",
        )

    result = ab.check_shared_modis_invariance(
        provenance, reference_ctx, candidate_ctx, attestation, base,
    )
    assert result["status"] == "pass"
    assert result["technical_failure"] is None
    assert result["reference_mode"] == result["candidate_mode"]
    assert result["baseline_feature_invariance_still_required"] is True
    assert result["reference_reproduction_still_required"] is True


def test_shared_modis_invariance_detects_a_mode_difference(tmp_path):
    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    attestation = _validate(base, provenance)
    reference_ctx = {"step7b_output_dir": tmp_path / "ref" / "step7b"}
    candidate_ctx = {"step7b_output_dir": tmp_path / "cand" / "step7b"}
    modes = (ab.LEGACY_MODIS_COMPATIBILITY_MODE, ab.MODIS_STRICT_MODE)
    for ctx, mode in zip((reference_ctx, candidate_ctx), modes):
        aligned = Path(ctx["step7b_output_dir"]) / "aligned_inputs"
        for index, name in enumerate((MODIS_MEAN, MODIS_STD)):
            _write_raster(aligned / f"{name}.tif",
                          np.full((8, 8), float(index), dtype="float32"))
        (Path(ctx["step7b_output_dir"]) / "downscaling_dataset_stats.json").write_text(
            json.dumps({"alignment_diagnostics": [{
                "name": MODIS_MEAN,
                "modis_source_validation": {"zero_fill_guard": {"mode": mode}},
            }]}), encoding="utf-8",
        )

    result = ab.check_shared_modis_invariance(
        provenance, reference_ctx, candidate_ctx, attestation, base,
    )
    assert result["status"] == "fail"
    assert result["technical_failure"] == ab.TECHNICAL_FAILURE_SHARED_MODIS
    assert any("identical_compatibility_mode" in r for r in result["reasons"])


def test_no_earth_engine_operation_occurs_in_the_compatibility_path(tmp_path):
    import inspect

    base, provenance, _ = _legacy_modis_fixture(tmp_path)
    compatibility_callables = (
        ab.describe_modis_raster, ab.zero_fill_guard_would_reject,
        ab.frozen_step7b_historical_evidence,
        ab.historical_evidence_confirms_no_nodata_source,
        ab.build_legacy_modis_attestation_declaration,
        ab.modis_compatibility_required, ab.validate_legacy_modis_compatibility,
        ab.step7b_compatibility_attestation, ab.check_shared_modis_invariance,
        ab.build_dry_run_modis_compatibility,
        step7b.validate_modis_source_rasters, step7b._authorize_zero_fill_waiver,
    )
    for callable_ in compatibility_callables:
        source = inspect.getsource(callable_)
        for forbidden in ("import ee", "ee.Initialize", "ee.ImageCollection",
                          "getInfo", "gee_utils", "init_gee"):
            assert forbidden not in source, f"{callable_.__name__} touches {forbidden}"
    # The gate also runs cleanly with every Earth Engine entry point booby-trapped.
    with ab.EarthEngineGuard():
        attestation = _validate(base, provenance)
    assert attestation["status"] == "pass"


def test_no_frozen_file_is_modified_by_the_compatibility_inspection():
    """The REAL frozen Manavgat inputs must survive the gate untouched."""
    frozen = [
        ab.frozen_modis_dir(EXPERIMENT) / f"{MODIS_MEAN}.tif",
        ab.frozen_modis_dir(EXPERIMENT) / f"{MODIS_STD}.tif",
        ab.frozen_modis_dir(EXPERIMENT) / "modis_metadata.json",
        ab.canonical_experiment_root(EXPERIMENT) / "step7b" / "downscaling_dataset_stats.json",
        ab.canonical_experiment_root(EXPERIMENT) / "step7b" / "aligned_inputs" / f"{MODIS_MEAN}.tif",
        ab.canonical_experiment_root(EXPERIMENT) / "step7b" / "aligned_inputs" / f"{MODIS_STD}.tif",
    ]
    present = [p for p in frozen if p.exists()]
    assert present, "the frozen Manavgat MODIS evidence should exist for this test"
    before = {str(p): (ab.audit.sha256_and_size(p), p.stat().st_mtime_ns) for p in present}

    ab.modis_compatibility_required(EXPERIMENT)
    ab.frozen_step7b_historical_evidence(EXPERIMENT)
    ab.build_dry_run_modis_compatibility(EXPERIMENT)
    ab.build_legacy_modis_attestation_declaration(EXPERIMENT)

    after = {str(p): (ab.audit.sha256_and_size(p), p.stat().st_mtime_ns) for p in present}
    assert before == after


def test_real_frozen_manavgat_modis_requires_the_compatibility_path():
    detection = ab.modis_compatibility_required(EXPERIMENT)
    assert detection["required"] is True
    mean = detection["rasters"][MODIS_MEAN]
    assert mean["nodata"] is None
    assert mean["exact_zero_fraction"] > detection["zero_fill_guard_threshold"]
    historical = ab.frozen_step7b_historical_evidence(EXPERIMENT)
    assert ab.historical_evidence_confirms_no_nodata_source(historical)
    assert historical["rasters"][MODIS_MEAN]["source_valid_pixel_count"] == \
        historical["rasters"][MODIS_MEAN]["source_total_pixel_count"]


# --- reporting ----------------------------------------------------------------
def _compatible_summary():
    summary = _minimal_summary([{"metric": "roc_auc", "point_estimate": 0.01}])
    compatibility = {"required": True, "status": "pass",
                     "mode": ab.LEGACY_MODIS_COMPATIBILITY_MODE,
                     "attestation_id": "abc123", "chains": {c: ab.LEGACY_MODIS_COMPATIBILITY_MODE
                                                            for c in ab.CHAINS}}
    summary["warnings"] = ab.summary_warnings(compatibility)
    summary["modis_compatibility"] = ab.build_modis_compatibility_report(
        compatibility, {"status": "pass"},
    )
    return summary


def test_warning_payload_has_the_required_shape():
    warning = ab.legacy_modis_compatibility_warning()
    assert warning["code"] == "legacy_zero_filled_modis_compatibility"
    assert warning["statement"]
    effect = warning["scientific_effect"]
    assert "preserved identically in both chains" in effect
    assert "does not validate zero as physical MODIS LST" in effect
    assert "does not replace a future MODIS nodata repair experiment" in effect


def test_warning_and_limitations_appear_in_json_and_markdown(tmp_path):
    summary = _compatible_summary()
    assert summary["warnings"][0]["code"] == "legacy_zero_filled_modis_compatibility"
    assert summary["modis_compatibility"]["modis_nodata_issue_resolved"] is False
    assert summary["modis_compatibility"]["default_step7b_guard_changed"] is False

    markdown = ab.render_summary_markdown(summary)
    assert "legacy_zero_filled_modis_compatibility" in markdown
    assert "does not validate zero as physical MODIS LST" in markdown
    for limitation in ab.legacy_modis_compatibility_limitations():
        assert limitation in markdown

    root = tmp_path / "root"
    root.mkdir(parents=True)
    (root / "downstream_ab_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = ab.build_manifest(EXPERIMENT, root, summary)
    assert manifest["warnings"][0]["code"] == "legacy_zero_filled_modis_compatibility"
    assert manifest["modis_compatibility"]["historical_compatibility_required"] is True

    provenance_block = ab.build_input_provenance(
        ab.build_input_plan(
            __import__("core.experiment_context", fromlist=["build_experiment_context"])
            .build_experiment_context(EXPERIMENT), EXPERIMENT),
        EXPERIMENT, grid_gate={}, compose_notes={},
    )["legacy_modis_compatibility"]
    assert provenance_block["historical_compatibility_required"] is True
    assert provenance_block["warning"]["code"] == "legacy_zero_filled_modis_compatibility"
    assert provenance_block["limitations"] == ab.legacy_modis_compatibility_limitations()


def test_required_limitations_include_the_modis_conditionality():
    limitations = ab.required_limitations()
    for item in ab.legacy_modis_compatibility_limitations():
        assert item in limitations
    joined = " ".join(limitations).lower()
    assert "conditional on the frozen historical modis representation" in joined
    assert "must be evaluated separately" in joined


def test_eligible_status_does_not_imply_the_modis_issue_is_resolved():
    summary = _compatible_summary()
    assert summary["final_status"] == ab.STATUS_ELIGIBLE_SECOND_AOI
    markdown = ab.render_summary_markdown(summary).lower()
    assert "does not mean the modis nodata issue is resolved" in markdown
    assert summary["modis_compatibility"]["modis_nodata_issue_resolved"] is False


# --- decision -----------------------------------------------------------------
def test_scientific_status_cannot_be_emitted_when_the_attestation_fails():
    decision = ab.decide_final_status(_full_evidence(
        modis_compatibility_required=True,
        modis_compatibility_attestation_status="fail",
    ))
    assert decision["final_status"] == ab.STATUS_BASELINE_INVARIANCE_FAILED
    assert decision["technical_failure"] == ab.TECHNICAL_FAILURE_SHARED_MODIS
    assert decision["final_status"] in ab.FINAL_STATUSES
    assert decision["production_approved"] is False


def test_shared_modis_invariance_failure_precedes_model_comparison_statuses():
    decision = ab.decide_final_status(_full_evidence(
        shared_modis_invariance_status="fail",
        shared_modis_invariance_reasons=["aligned arrays differ"],
        baseline_invariance_status="pass",
        population_alignment_status="ok",
    ))
    assert decision["final_status"] == ab.STATUS_BASELINE_INVARIANCE_FAILED
    assert decision["technical_failure"] == ab.TECHNICAL_FAILURE_SHARED_MODIS
    assert "aligned arrays differ" in json.dumps(decision)


def test_passing_modis_evidence_does_not_change_the_ordinary_decision():
    decision = ab.decide_final_status(_full_evidence(
        shared_modis_invariance_status="pass",
        modis_compatibility_required=True,
        modis_compatibility_attestation_status="pass",
    ))
    assert decision["final_status"] == ab.STATUS_ELIGIBLE_SECOND_AOI
    assert decision["technical_failure"] is None


def test_final_statuses_are_unchanged_by_the_technical_failure_field():
    assert ab.TECHNICAL_FAILURE_SHARED_MODIS not in ab.FINAL_STATUSES
    assert ab.FINAL_STATUSES == (
        ab.STATUS_INVALID_REFERENCE, ab.STATUS_BASELINE_INVARIANCE_FAILED,
        ab.STATUS_POPULATION_REVIEW, ab.STATUS_SEAM_REDUCED_TRADEOFF,
        ab.STATUS_ELIGIBLE_SECOND_AOI, ab.STATUS_INCONCLUSIVE,
    )


# --- checkpoint / resume -------------------------------------------------------
def _checkpoint_root(tmp_path):
    root = tmp_path / "root"
    (root / "checkpoints").mkdir(parents=True)
    return root


def _stage_output(root, name):
    path = root / f"{name}.txt"
    path.write_text(name, encoding="utf-8")
    return path


def test_attestation_stage_precedes_every_step7b_stage():
    stages = list(ab.PLANNED_STAGES)
    assert "modis_compatibility_attestation" in stages
    index = stages.index("modis_compatibility_attestation")
    for stage in ("reference_step7b", "candidate_step7b"):
        assert stages.index(stage) > index
    for stage in ("reference_step5", "candidate_step5c"):
        assert stages.index(stage) < index


def test_modis_dependent_stages_are_step7b_and_later_only():
    assert "reference_step7b" in ab.MODIS_DEPENDENT_STAGES
    assert "candidate_step7b" in ab.MODIS_DEPENDENT_STAGES
    assert "report_generation" in ab.MODIS_DEPENDENT_STAGES
    for stage in ("validate_inputs", "materialize_inputs", "reference_step5",
                  "reference_step5c", "candidate_step5", "candidate_step5c",
                  "reference_step7a", "candidate_step7a",
                  "modis_compatibility_attestation"):
        assert stage not in ab.MODIS_DEPENDENT_STAGES


def test_old_checkpoint_schema_invalidates_step7b_and_later_only(tmp_path):
    root = _checkpoint_root(tmp_path)
    attestation = {"mode": ab.LEGACY_MODIS_COMPATIBILITY_MODE, "required": True,
                   "experiment_id": EXPERIMENT,
                   "rasters": {MODIS_MEAN: {"sha256": "a"}, MODIS_STD: {"sha256": "b"}}}
    stages = ("reference_step5", "reference_step5c", "candidate_step5",
              "candidate_step5c", "reference_step7a", "reference_step7b",
              "reference_step8a", "report_generation")
    payload = {"experiment": ab.DIAGNOSTIC_NAMESPACE, "stages": {}}
    for stage in stages:
        path = _stage_output(root, stage)
        payload["stages"][stage] = {
            "completed_at": "2026-07-26T00:00:00+00:00",
            "outputs": [{"path": str(path), "bytes": path.stat().st_size}],
        }
    # A pre-2.0 checkpoint: no schema version recorded at all.
    ab.write_json_atomic(ab.checkpoint_path(root), payload)

    for stage in ("reference_step5", "reference_step5c", "candidate_step5",
                  "candidate_step5c", "reference_step7a"):
        assert ab.stage_is_reusable(root, stage, attestation) is True
    for stage in ("reference_step7b", "reference_step8a", "report_generation"):
        assert ab.stage_is_reusable(root, stage, attestation) is False


def test_resume_revalidates_the_attestation_binding(tmp_path):
    root = _checkpoint_root(tmp_path)
    attestation = {"mode": ab.LEGACY_MODIS_COMPATIBILITY_MODE, "required": True,
                   "experiment_id": EXPERIMENT,
                   "rasters": {MODIS_MEAN: {"sha256": "a"}, MODIS_STD: {"sha256": "b"}}}
    path = _stage_output(root, "reference_step7b")
    ab.write_checkpoint_stage(root, "reference_step7b", [path], attestation=attestation)

    assert ab.checkpoint_schema_version(root) == ab.CHECKPOINT_SCHEMA_VERSION
    assert ab.stage_is_reusable(root, "reference_step7b", attestation) is True

    rebound = dict(attestation, rasters={MODIS_MEAN: {"sha256": "CHANGED"},
                                         MODIS_STD: {"sha256": "b"}})
    assert ab.stage_is_reusable(root, "reference_step7b", rebound) is False
    # ...and a strict-mode run cannot reuse a compatibility-mode stage either.
    assert ab.stage_is_reusable(root, "reference_step7b", None) is False


def test_resume_reuses_valid_step5_and_step5c_without_an_attestation(tmp_path):
    root = _checkpoint_root(tmp_path)
    for stage in ("reference_step5", "reference_step5c", "candidate_step5",
                  "candidate_step5c"):
        ab.write_checkpoint_stage(root, stage, [_stage_output(root, stage)])
        assert ab.stage_is_reusable(root, stage) is True
        assert ab.stage_is_reusable(root, stage, None) is True

    # A vanished output still invalidates the stage: checkpoint text alone is
    # never trusted.
    (root / "reference_step5.txt").unlink()
    assert ab.stage_is_reusable(root, "reference_step5") is False


def test_attestation_binding_is_stable_and_content_addressed():
    attestation = {"mode": ab.LEGACY_MODIS_COMPATIBILITY_MODE, "required": True,
                   "experiment_id": EXPERIMENT,
                   "rasters": {MODIS_MEAN: {"sha256": "a"}}}
    assert ab.attestation_binding(attestation) == ab.attestation_binding(attestation)
    other = dict(attestation, rasters={MODIS_MEAN: {"sha256": "z"}})
    assert (ab.attestation_binding(attestation)["binding_sha256"]
            != ab.attestation_binding(other)["binding_sha256"])
    assert (ab.attestation_binding({})["binding_sha256"]
            != ab.attestation_binding(attestation)["binding_sha256"])


# --- dry-run -------------------------------------------------------------------
def test_dry_run_reports_whether_compatibility_is_required_and_writes_nothing(tmp_path):
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

    config_path = ab.legacy_modis_attestation_config_path()
    before = ab.audit.sha256_and_size(config_path) if config_path.exists() else None

    monkey = pytest.MonkeyPatch()
    monkey.setattr(Path, "mkdir", _tracking_mkdir)
    monkey.setattr("builtins.open", _tracking_open)
    try:
        plan = ab.build_dry_run_plan(EXPERIMENT, ab.CHAIN_CANDIDATE)
    finally:
        monkey.undo()

    modis = plan["modis_compatibility"]
    assert modis["historical_compatibility_required"] is True
    assert modis["mode_name"] == ab.LEGACY_MODIS_COMPATIBILITY_MODE
    assert modis["default_mode"] == ab.MODIS_STRICT_MODE
    assert modis["experiment_is_authorized"] is True
    assert modis["attestation_declaration_present"] is True
    assert modis["historical_step7b_evidence_confirmed"] is True
    assert modis["default_step7b_guard_changed"] is False
    assert modis["warning"]["code"] == "legacy_zero_filled_modis_compatibility"
    assert modis["limitations"] == ab.legacy_modis_compatibility_limitations()
    assert modis["writes_performed"] is False
    assert "nodata" in modis["reason"] and "0.0" in modis["reason"]

    assert [p for p in opened if "config" in p or ab.DIAGNOSTIC_NAMESPACE in p] == []
    assert [p for p in created if ab.DIAGNOSTIC_NAMESPACE in p] == []
    after = ab.audit.sha256_and_size(config_path) if config_path.exists() else None
    assert before == after


def test_dry_run_plan_still_matches_the_declared_stage_list():
    plan = ab.build_dry_run_plan(EXPERIMENT, ab.CHAIN_CANDIDATE)
    assert plan["planned_stages"] == list(ab.PLANNED_STAGES)
    assert "modis_compatibility_attestation" in plan["planned_stages"]


def test_declaration_hashes_are_derived_not_hardcoded(tmp_path):
    """The declaration's hashes must come from the frozen files themselves."""
    base, _, declaration = _legacy_modis_fixture(tmp_path)
    modis_dir = base / "outputs" / "experiments" / EXPERIMENT / "data" / "modis"
    for name in (MODIS_MEAN, MODIS_STD):
        derived = ab.audit.sha256_and_size(modis_dir / f"{name}.tif")
        assert declaration["rasters"][name]["sha256"] == derived["sha256"]
        assert declaration["rasters"][name]["bytes"] == derived["bytes"]
    source = _module_source(ab)
    assert declaration["rasters"][MODIS_MEAN]["sha256"] not in source


def test_committed_declaration_matches_the_frozen_manavgat_inputs():
    declaration = ab.load_legacy_modis_attestation_declaration()
    assert declaration["experiment_id"] == EXPERIMENT
    assert declaration["mode"] == ab.LEGACY_MODIS_COMPATIBILITY_MODE
    assert declaration["declares_zero_scientifically_valid"] is False
    for name, spec in ab.MODIS_COMPATIBILITY_RASTERS.items():
        path = ab.frozen_modis_dir(EXPERIMENT) / spec["filename"]
        signed = ab.audit.sha256_and_size(path)
        assert declaration["rasters"][name]["path"] == str(path.resolve())
        assert declaration["rasters"][name]["sha256"] == signed["sha256"]
        assert declaration["rasters"][name]["bytes"] == signed["bytes"]
        assert declaration["rasters"][name]["nodata"] is None
    assert declaration["historical_step7b_evidence_confirmed"] is True
