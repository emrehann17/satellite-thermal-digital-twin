"""Tests for the window-closure MODEL stage
(src/window_closure_sensitivity.py + scripts/validate_window_closure_model.py).

Everything is synthetic and runs under tmp_path through the module's public
`output_root` / `experiments_root` injection points. The PRODUCTION fire-risk
pipeline, fold helper, metric helpers and bootstrap primitives really do run --
only the replicate count is reduced through an explicit test-only override, so
the metric and interval logic is genuinely exercised. No test touches the real
canonical outputs, Earth Engine, or the Step7C downscaling model.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.validate_window_closure_model as validator  # noqa: E402
import src.window_closure_sensitivity as wcs  # noqa: E402

# The local-downstream test module owns the synthetic upstream fixtures; they
# are reused verbatim so the two stages cannot drift apart. `tests/` is not a
# package, so it is imported by file location rather than as `tests.<name>`.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import test_window_closure_local_downstream as ld  # noqa: E402

_NONZERO = ld._NONZERO
_SHIFTS = ld._SHIFTS
_canonical_frame = ld._canonical_frame
_fake_downstream_engine = ld._fake_downstream_engine
_namespace_snapshot = ld._namespace_snapshot
_predictor_env = ld._predictor_env
_variant_frame = ld._variant_frame
any_experiment = ld.any_experiment
ctx_for = ld.ctx_for

#: Small but production-legal: the frozen Step8B minimum is 30 positives AND
#: 30 negatives per population, and 5 spatial folds must each carry positives.
_COHORT_ROWS = 420
#: Test-only bootstrap override. Production uses the frozen core.config values;
#: a dedicated test below asserts that.
_FAST_BOOTSTRAP = {"n_bootstrap": 24}


# =============================================================================
# Fail-closed guard
# =============================================================================
@pytest.fixture(autouse=True)
def _no_production_side_effects(monkeypatch):
    def _blocked(name):
        def _fail(*_args, **_kwargs):
            raise AssertionError(f"fail-closed guard: production {name} was invoked")
        return _fail

    import core.gee_utils as gee_utils
    import scripts.prepare_modis_for_step7 as prepare_modis
    import scripts.run_predictors_only as run_predictors_only
    import src.step6_validate_fire_relation as step6
    import src.step7c_train_downscaling_model as step7c

    monkeypatch.setattr(
        step6, "export_raw_mcd64a1_prelabel_labels", _blocked("Step6 prelabel exporter"),
    )
    monkeypatch.setattr(gee_utils, "init_gee", _blocked("init_gee"))
    monkeypatch.setattr(
        run_predictors_only, "export_image_direct_or_tiled", _blocked("GEE exporter"),
    )
    monkeypatch.setattr(
        prepare_modis, "prepare_modis_for_step7", _blocked("production MODIS exporter"),
    )
    # The Step7C downscaling model must NEVER be refit by the model stage.
    monkeypatch.setattr(step7c, "run_step7c", _blocked("Step7C downscaling refit"))
    monkeypatch.setattr(
        wcs, "production_local_downstream_engine",
        _blocked("production local-downstream engine"),
    )
    if "geemap" not in sys.modules:
        guard = types.ModuleType("geemap")
        guard.ee_export_image = _blocked("geemap download")
        monkeypatch.setitem(sys.modules, "geemap", guard)


# =============================================================================
# Synthetic environment
# =============================================================================
def _model_canonical_frame(rows: int = _COHORT_ROWS) -> pd.DataFrame:
    """A Step8A frame big enough for the frozen fold/positive requirements.

    Signal is deliberately built in: the thermal features carry more label
    information than the baseline ones, so the six evaluations produce
    non-degenerate, distinguishable metrics.
    """
    frame = _canonical_frame(rows=rows)
    index = np.arange(rows)
    rng = np.random.default_rng(20260731)

    burned = (index % 7 == 0).astype("int64")
    frame["burned"] = burned
    frame["burn_date"] = np.where(burned == 1, 210.0, np.nan)
    frame["burn_month"] = np.where(burned == 1, 8, 0).astype("int64")
    frame["burn_day_of_year"] = frame["burn_date"]
    # Spread cells over a real 2-D grid so spatial blocks are meaningful.
    frame["row_500m"] = (index // 20).astype("int64")
    frame["col_500m"] = (index % 20).astype("int64")
    frame["cell_id"] = [
        f"r{r}_c{c}" for r, c in zip(frame["row_500m"], frame["col_500m"])
    ]
    frame["lon"] = 31.0 + frame["col_500m"] * 0.01
    frame["lat"] = 37.0 + frame["row_500m"] * 0.01
    frame["burnable_tree_shrub_grass"] = True
    frame["burnable_tree_shrub"] = True
    frame["valid_for_modeling"] = True
    frame["analysis_eligible"] = True
    frame["pre_label_burn_excluded"] = False

    for name in ("ndvi_mean", "elevation_mean", "slope_mean"):
        frame[name] = rng.normal(0.0, 1.0, rows) + burned * 0.25
    frame["landcover_dominant"] = np.where(index % 3 == 0, 10, 20).astype("int64")
    for name in ("lst_anomaly_mean", "current_lst_mean", "current_tvdi_mean",
                 "tvdi_difference_mean", "downscaled_lst_mean", "fused_lst_mean"):
        frame[name] = rng.normal(0.0, 1.0, rows) + burned * 1.6
    return frame


def _model_variant_frame(canonical: pd.DataFrame, shift: float) -> pd.DataFrame:
    """A variant: identical cells/labels/statics, shifted timing features."""
    variant = canonical.copy()
    rng = np.random.default_rng(1000 + int(shift * 100))
    for name in ("ndvi_mean", "lst_anomaly_mean", "current_lst_mean",
                 "current_tvdi_mean", "tvdi_difference_mean",
                 "downscaled_lst_mean", "fused_lst_mean"):
        variant[name] = variant[name] + shift + rng.normal(0.0, 0.05, len(variant))
    return variant


def _model_env(tmp_path: Path, *, canonical: Optional[pd.DataFrame] = None,
               variant_frames: Optional[dict] = None):
    """plan -> prelabel -> predictor-export -> local-downstream, all completed."""
    canonical = _model_canonical_frame() if canonical is None else canonical
    experiment_id, out, experiments = _predictor_env(tmp_path, canonical=canonical)
    frames = variant_frames or {
        "close_7d_earlier": _model_variant_frame(canonical, 0.30),
        "close_14d_earlier": _model_variant_frame(canonical, 0.60),
    }
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical, frames=frames),
    )
    return experiment_id, out, experiments


def _run_model(tmp_path: Path, *, env=None, overrides=None, **kwargs):
    experiment_id, out, experiments = env or _model_env(tmp_path)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="model", to_stage="model",
        output_root=out, experiments_root=experiments,
        model_configuration_overrides=(
            _FAST_BOOTSTRAP if overrides is None else overrides
        ),
        **kwargs,
    )
    return result, experiment_id, out, experiments


def _dry_run_model(tmp_path: Path, env=None) -> tuple[dict, str, Path, Path]:
    experiment_id, out, experiments = env or _model_env(tmp_path)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=True,
        from_stage="model", to_stage="model",
        output_root=out, experiments_root=experiments,
    )
    return result, experiment_id, out, experiments


def _metadata(out: Path, experiment_id: str) -> dict:
    return json.loads(
        wcs.model_metadata_path(experiment_id, out).read_text(encoding="utf-8")
    )


def _write_log(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "INFO before\n" + json.dumps(payload, indent=2, default=str) + "\nINFO after\n",
        encoding="utf-8",
    )
    return path


def _validate(mode: str, experiment_id: str, out: Path, *,
              log: Optional[Path] = None, experiments: Optional[Path] = None) -> int:
    argv = ["--experiment", experiment_id, "--mode", mode,
            "--shifts", *[str(s) for s in _SHIFTS], "--output-root", str(out)]
    if log is not None:
        argv += ["--log", str(log)]
    if experiments is not None:
        argv += ["--experiments-root", str(experiments)]
    return validator.main(argv)


def _upstream_snapshot(experiment_id: str, out: Path, experiments: Path) -> dict:
    return _namespace_snapshot(
        out / experiment_id / "config",
        out / experiment_id / "prelabel_censor",
        *[out / experiment_id / "variants" / v for v in _NONZERO],
        wcs.canonical_experiment_root(experiment_id, experiments),
    )


# =============================================================================
# 1. Stage lock
# =============================================================================
def test_model_is_implemented():
    assert wcs.MODEL_STAGE == "model"
    assert wcs.MODEL_STAGE in wcs.IMPLEMENTED_ACTUAL_STAGES
    assert wcs.IMPLEMENTED_ACTUAL_STAGES == wcs.STAGES
    wcs.assert_actual_stages_supported(wcs.validate_stage_range("model", "model"))


def test_an_unimplemented_stage_still_fails_fast(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match="not enabled"):
        wcs.assert_actual_stages_supported(["model", "some-future-stage"])
    assert _namespace_snapshot(out) == before
    assert not wcs.model_root(experiment_id, out).exists()


# =============================================================================
# 2-6. Frozen input binding, all before any write
# =============================================================================
def _expect_binding_failure(experiment_id, out, experiments, match=""):
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match=match):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="model", to_stage="model",
            output_root=out, experiments_root=experiments,
            model_configuration_overrides=_FAST_BOOTSTRAP,
        )
    assert not wcs.model_root(experiment_id, out).exists()
    assert not wcs.model_staging_root(experiment_id, out).exists()
    assert _namespace_snapshot(out) == before


def test_a_wrong_analysis_id_fails_before_any_write(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match="analysis_id|shift"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
            from_stage="model", to_stage="model",
            output_root=out, experiments_root=experiments,
            model_configuration_overrides=_FAST_BOOTSTRAP,
        )
    assert not wcs.model_root(experiment_id, out).exists()
    assert _namespace_snapshot(out) == before


def test_a_missing_canonical_step8a_fails_before_any_write(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    wcs.canonical_step8a_path(experiment_id, experiments).unlink()
    _expect_binding_failure(experiment_id, out, experiments)


def test_a_canonical_hash_mismatch_fails_before_any_write(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    path = wcs.canonical_step8a_path(experiment_id, experiments)
    frame = pd.read_parquet(path)
    frame.loc[0, "ndvi_mean"] = float(frame.loc[0, "ndvi_mean"]) + 1.0
    frame.to_parquet(path, index=False)
    _expect_binding_failure(experiment_id, out, experiments)


def test_a_non_pass_local_downstream_metadata_fails(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    path = wcs.local_downstream_metadata_path(experiment_id, "close_7d_earlier", out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "fail"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _expect_binding_failure(experiment_id, out, experiments, "status")


def test_a_shifted_step8a_hash_mismatch_fails(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    dataset = wcs.variant_step8a_dataset_path(experiment_id, "close_14d_earlier", out)
    frame = pd.read_parquet(dataset)
    frame.loc[0, "current_lst_mean"] = float(frame.loc[0, "current_lst_mean"]) + 5.0
    frame.to_parquet(dataset, index=False)
    _expect_binding_failure(experiment_id, out, experiments, "hashes")


@pytest.mark.parametrize("flag", sorted(wcs.LOCAL_DOWNSTREAM_REQUIRED_FLAGS))
def test_every_required_local_downstream_flag_is_enforced(tmp_path, flag):
    experiment_id, out, experiments = _model_env(tmp_path)
    path = wcs.local_downstream_metadata_path(experiment_id, "close_7d_earlier", out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[flag] = not wcs.LOCAL_DOWNSTREAM_REQUIRED_FLAGS[flag]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _expect_binding_failure(experiment_id, out, experiments, flag)


# =============================================================================
# 7-13. Exact common cohort
# =============================================================================
def test_one_exact_common_cohort_is_created(tmp_path):
    result, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    cohort_meta = metadata["common_cohort"]
    assert result["common_cohort_created"] is True
    assert cohort_meta["final_common_cohort_rows"] > 0
    assert cohort_meta["final_positive_rows"] > 0
    assert cohort_meta["final_negative_rows"] > 0
    assert cohort_meta["prevalence"] == pytest.approx(
        cohort_meta["final_positive_rows"] / cohort_meta["final_common_cohort_rows"]
    )
    assert cohort_meta["primary_population"] == wcs.PRIMARY_POPULATION
    assert "cell_id" in cohort_meta["stable_cell_key_columns"]
    for field in ("initial_rows_by_variant", "rows_present_in_all_variants",
                  "removed_not_valid_for_modeling", "removed_outside_primary_population",
                  "removed_missing_required_feature_union", "removed_label_mismatch",
                  "removed_static_invariance_failure", "removed_prelabel_censor",
                  "input_dataset_paths", "input_dataset_sha256"):
        assert field in cohort_meta, field


def test_the_stable_cell_key_is_the_production_one(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    layout = wcs.model_relative_layout()
    cohort = pd.read_parquet(
        wcs.model_root(experiment_id, out) / layout["common_cohort"]
    )
    assert not cohort["cell_id"].duplicated().any()
    assert (
        cohort["cell_id"]
        == "r" + cohort["row_500m"].astype(str) + "_c" + cohort["col_500m"].astype(str)
    ).all()


def test_a_variant_only_invalidity_shrinks_the_common_cohort(tmp_path):
    """`valid_for_modeling` is support-driven, so it may differ per variant."""
    canonical = _model_canonical_frame()
    variant = _model_variant_frame(canonical, 0.30)
    variant.loc[0:9, "valid_for_modeling"] = False
    experiment_id, out, experiments = _model_env(tmp_path, canonical=canonical, variant_frames={
        "close_7d_earlier": variant,
        "close_14d_earlier": _model_variant_frame(canonical, 0.60),
    })
    _run_model(tmp_path, env=(experiment_id, out, experiments))
    cohort_meta = _metadata(out, experiment_id)["common_cohort"]
    assert cohort_meta["removed_not_valid_for_modeling"]["close_7d_earlier"] == 10
    assert cohort_meta["final_common_cohort_rows"] == len(canonical) - 10


# --- Cohort gates, exercised directly on the builder -------------------------
def _cohort_inputs(tmp_path: Path):
    _, _, lineage = ld._lineage(tmp_path)
    canonical = _model_canonical_frame()
    frames = {
        "canonical": canonical,
        "close_7d_earlier": _model_variant_frame(canonical, 0.30),
        "close_14d_earlier": _model_variant_frame(canonical, 0.60),
    }
    censor = {
        "censor_applied": True, "censored_cell_ids": [],
        "censored_cell_count": 0, "positive_source_pixel_count": 0,
    }
    return frames, censor, wcs.model_feature_registry(), lineage


def test_the_cohort_builder_produces_one_exact_intersection(tmp_path):
    frames, censor, registry, lineage = _cohort_inputs(tmp_path)
    cohort = wcs.build_model_common_cohort(frames, censor, registry, lineage)
    keys = {name: list(frame["cell_id"]) for name, frame in cohort["common"].items()}
    assert len(set(map(tuple, keys.values()))) == 1
    assert cohort["metadata"]["final_common_cohort_rows"] == len(cohort["cell_ids"])


def test_the_cohort_builder_enforces_the_primary_population(tmp_path):
    frames, censor, registry, lineage = _cohort_inputs(tmp_path)
    frames["close_7d_earlier"] = frames["close_7d_earlier"].copy()
    frames["close_7d_earlier"].loc[0:9, wcs.PRIMARY_POPULATION] = False
    cohort = wcs.build_model_common_cohort(frames, censor, registry, lineage)
    metadata = cohort["metadata"]
    assert metadata["removed_outside_primary_population"]["close_7d_earlier"] == 10
    assert metadata["final_common_cohort_rows"] == len(frames["canonical"]) - 10


def test_the_cohort_builder_applies_the_prelabel_censor(tmp_path):
    frames, censor, registry, lineage = _cohort_inputs(tmp_path)
    victims = sorted(frames["canonical"]["cell_id"])[:3]
    censor = dict(censor, censored_cell_ids=victims, censored_cell_count=len(victims))
    cohort = wcs.build_model_common_cohort(frames, censor, registry, lineage)
    metadata = cohort["metadata"]
    assert metadata["final_common_cohort_rows"] == len(frames["canonical"]) - 3
    for name in metadata["removed_prelabel_censor"]:
        assert metadata["removed_prelabel_censor"][name] == 3
    for frame in cohort["common"].values():
        assert not set(frame["cell_id"]) & set(victims)


def test_the_cohort_builder_fails_on_a_label_mismatch(tmp_path):
    frames, censor, registry, lineage = _cohort_inputs(tmp_path)
    frames["close_7d_earlier"] = frames["close_7d_earlier"].copy()
    frames["close_7d_earlier"].loc[0, "burned"] = (
        1 - int(frames["close_7d_earlier"].loc[0, "burned"])
    )
    with pytest.raises(wcs.WindowClosureError, match="Label mismatch"):
        wcs.build_model_common_cohort(frames, censor, registry, lineage)


@pytest.mark.parametrize("column,value", [
    ("elevation_mean", 9999.0), ("slope_mean", 9999.0),
    ("landcover_dominant", 99), ("lon", 99.0), ("row_500m", 9999),
])
def test_the_cohort_builder_fails_on_a_static_mismatch(tmp_path, column, value):
    frames, censor, registry, lineage = _cohort_inputs(tmp_path)
    frames["close_14d_earlier"] = frames["close_14d_earlier"].copy()
    frames["close_14d_earlier"].loc[0, column] = value
    with pytest.raises(
        wcs.WindowClosureError, match="Static invariance failure|cell_id order",
    ):
        wcs.build_model_common_cohort(frames, censor, registry, lineage)


def test_the_cohort_builder_fails_on_a_single_class_cohort(tmp_path):
    frames, censor, registry, lineage = _cohort_inputs(tmp_path)
    for name in frames:
        frames[name] = frames[name].copy()
        frames[name]["burned"] = 0
    with pytest.raises(wcs.WindowClosureError, match="single class"):
        wcs.build_model_common_cohort(frames, censor, registry, lineage)


def test_a_missing_required_feature_is_excluded_from_every_variant(tmp_path):
    canonical = _model_canonical_frame()
    variant = _model_variant_frame(canonical, 0.30)
    variant.loc[0:4, "fused_lst_mean"] = np.nan
    experiment_id, out, experiments = _model_env(tmp_path, canonical=canonical, variant_frames={
        "close_7d_earlier": variant,
        "close_14d_earlier": _model_variant_frame(canonical, 0.60),
    })
    _run_model(tmp_path, env=(experiment_id, out, experiments))
    cohort_meta = _metadata(out, experiment_id)["common_cohort"]
    assert cohort_meta["removed_missing_required_feature_union"]["close_7d_earlier"] == 5
    assert cohort_meta["final_common_cohort_rows"] == len(canonical) - 5


# =============================================================================
# 14, 15. Shared pre-label censor
# =============================================================================
def test_a_zero_positive_prelabel_raster_still_records_an_applied_censor(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    censor = _metadata(out, experiment_id)["prelabel_censor"]
    assert censor["censor_applied"] is True
    assert censor["majority_or_threshold_used"] is False
    assert censor["positive_source_pixel_count"] == 0
    assert censor["censored_cell_count"] == 0
    assert censor["zero_censored_cells_is_a_valid_outcome"] is True


def test_a_positive_prelabel_cell_is_censored_from_every_variant(tmp_path):
    import rasterio
    from src.step8a_prepare_500m_modeling_dataset import compute_block_size_pixels

    experiment_id, out, experiments = _model_env(tmp_path)
    raster = wcs.prelabel_raster_path(experiment_id, out)
    with rasterio.open(raster) as src:
        profile, data = src.profile, src.read(1)
    data[0, 0] = 200
    with rasterio.open(raster, "w", **profile) as dst:
        dst.write(data, 1)

    censored = wcs.prelabel_censored_cells(raster)
    block = compute_block_size_pixels()
    assert censored["censor_applied"] is True
    assert censored["positive_source_pixel_count"] == 1
    assert censored["censored_cell_ids"] == ["r0_c0"]
    assert censored["block_size_pixels"] == block


def test_the_censor_uses_the_production_grid_contract(tmp_path):
    experiment_id, out, _ = _model_env(tmp_path)
    censored = wcs.prelabel_censored_cells(wcs.prelabel_raster_path(experiment_id, out))
    assert "compute_cell_identity" in censored["grid_contract_source"]
    assert censored["rule"].startswith("any positive")


# =============================================================================
# Regression: the production tile-grid traversal contract
#
# `core.utils.tiling.make_tile_grid` returns a GRID DICT whose per-tile records
# live under "tiles". Iterating the dict itself yields its string keys, which is
# what previously broke the censor. These tests pin the production traversal.
# =============================================================================
def _block_size() -> int:
    from src.step8a_prepare_500m_modeling_dataset import compute_block_size_pixels

    return compute_block_size_pixels()


def _write_prelabel_raster(path: Path, shape, positives=(), *, nodata=-9999.0,
                           masked_cells=()) -> Path:
    """A synthetic BurnDate raster: 0 everywhere, given cells positive."""
    import rasterio

    array = np.zeros(shape, dtype="float32")
    for (row, col), value in positives:
        array[row, col] = value
    for row, col in masked_cells:
        array[row, col] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
        dtype="float32", crs="EPSG:4326",
        transform=ld._TEST_TRANSFORM, nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


def test_make_tile_grid_returns_a_grid_dict_not_an_iterable_of_tiles():
    """The contract the censor must honour, asserted against production."""
    from core.utils.tiling import make_tile_grid

    grid = make_tile_grid({"width": 40, "height": 40}, tile_size_pixels=_block_size())
    assert isinstance(grid, dict)
    assert isinstance(grid["tiles"], list)
    assert all(isinstance(tile, dict) for tile in grid["tiles"])
    # Iterating the grid itself yields STRING keys -- never a tile record.
    assert all(isinstance(key, str) for key in grid)
    assert "write_window" not in grid
    assert set(grid["tiles"][0]) >= {"index", "write_window", "read_window"}


def test_the_censor_traverses_the_tiles_list_and_never_a_string(tmp_path, monkeypatch):
    """Every object the censor treats as a tile must be a real tile record."""
    import core.utils.tiling as tiling

    seen: list = []
    real = tiling.make_tile_grid

    def _recording(*args, **kwargs):
        grid = real(*args, **kwargs)
        seen.append(grid)
        return grid

    monkeypatch.setattr(tiling, "make_tile_grid", _recording)

    raster = _write_prelabel_raster(tmp_path / "prelabel.tif", (40, 40))
    result = wcs.prelabel_censored_cells(raster)

    assert seen, "the production tile grid helper was not called"
    grid = seen[0]
    assert result["tile_count"] == grid["n_tiles"] == len(grid["tiles"])
    assert result["tile_grid_shape"] == [grid["n_tile_rows"], grid["n_tile_cols"]]


def test_the_censor_calls_the_production_identity_helpers(tmp_path, monkeypatch):
    import src.step8a_prepare_500m_modeling_dataset as step8a

    calls = {"block_size": 0, "identity": 0}
    real_block, real_identity = step8a.compute_block_size_pixels, step8a.compute_cell_identity

    def _block(*args, **kwargs):
        calls["block_size"] += 1
        return real_block(*args, **kwargs)

    def _identity(*args, **kwargs):
        calls["identity"] += 1
        return real_identity(*args, **kwargs)

    monkeypatch.setattr(step8a, "compute_block_size_pixels", _block)
    monkeypatch.setattr(step8a, "compute_cell_identity", _identity)

    raster = _write_prelabel_raster(
        tmp_path / "prelabel.tif", (40, 40), positives=[((0, 0), 200.0)],
    )
    result = wcs.prelabel_censored_cells(raster)
    assert calls["block_size"] == 1
    assert calls["identity"] >= 1
    assert result["censored_cell_ids"] == ["r0_c0"]


def test_a_zero_positive_raster_records_an_applied_censor(tmp_path):
    raster = _write_prelabel_raster(tmp_path / "prelabel.tif", (40, 40))
    result = wcs.prelabel_censored_cells(raster)
    assert result["censor_applied"] is True
    assert result["positive_source_pixel_count"] == 0
    assert result["censored_cell_count"] == 0
    assert result["censored_cell_ids"] == []
    assert result["majority_or_threshold_used"] is False


def test_the_first_source_pixel_censors_the_first_production_cell(tmp_path):
    raster = _write_prelabel_raster(
        tmp_path / "prelabel.tif", (40, 40), positives=[((0, 0), 210.0)],
    )
    result = wcs.prelabel_censored_cells(raster)
    assert result["censored_cell_ids"] == ["r0_c0"]
    assert result["positive_source_pixel_count"] == 1


def test_a_positive_pixel_in_an_edge_tile_censors_the_edge_cell(tmp_path):
    """The last tile row/column is PARTIAL; its cell id must still be right."""
    block = _block_size()
    size = 2 * block + 3          # a full tile, then a 3-pixel partial edge tile
    raster = _write_prelabel_raster(
        tmp_path / "prelabel.tif", (size, size),
        positives=[((size - 1, size - 1), 205.0)],
    )
    result = wcs.prelabel_censored_cells(raster)
    from src.step8a_prepare_500m_modeling_dataset import compute_cell_identity

    expected, _, _ = compute_cell_identity(2 * block, 2 * block, block)
    assert result["censored_cell_ids"] == [expected]
    assert result["tile_grid_shape"] == [3, 3]


def test_several_positives_in_one_tile_produce_one_exclusion(tmp_path):
    raster = _write_prelabel_raster(
        tmp_path / "prelabel.tif", (40, 40),
        positives=[((0, 0), 200.0), ((1, 2), 201.0), ((3, 4), 202.0)],
    )
    result = wcs.prelabel_censored_cells(raster)
    assert result["censored_cell_ids"] == ["r0_c0"]
    assert result["censored_cell_count"] == 1
    assert result["positive_source_pixel_count"] == 3


def test_positives_in_different_tiles_produce_distinct_cells(tmp_path):
    from src.step8a_prepare_500m_modeling_dataset import compute_cell_identity

    block = _block_size()
    raster = _write_prelabel_raster(
        tmp_path / "prelabel.tif", (2 * block, 2 * block),
        positives=[((0, 0), 200.0), ((block, block), 201.0)],
    )
    result = wcs.prelabel_censored_cells(raster)
    first, _, _ = compute_cell_identity(0, 0, block)
    second, _, _ = compute_cell_identity(block, block, block)
    assert result["censored_cell_ids"] == sorted([first, second])
    assert result["censored_cell_count"] == 2


def test_a_nodata_pixel_is_never_positive(tmp_path):
    """The nodata value is POSITIVE here, so only real masking can pass this."""
    raster = _write_prelabel_raster(
        tmp_path / "prelabel.tif", (40, 40), nodata=250.0,
        masked_cells=[(0, 0), (5, 5)],
    )
    result = wcs.prelabel_censored_cells(raster)
    assert result["positive_source_pixel_count"] == 0
    assert result["censored_cell_ids"] == []


def test_the_censor_leaves_the_raster_read_only(tmp_path):
    raster = _write_prelabel_raster(
        tmp_path / "prelabel.tif", (40, 40), positives=[((0, 0), 200.0)],
    )
    before_hash = wcs.sha256_file(raster)
    before_mtime = raster.stat().st_mtime_ns

    wcs.prelabel_censored_cells(raster)

    assert wcs.sha256_file(raster) == before_hash
    assert raster.stat().st_mtime_ns == before_mtime


def test_the_censor_is_deterministic(tmp_path):
    raster = _write_prelabel_raster(
        tmp_path / "prelabel.tif", (40, 40),
        positives=[((0, 0), 200.0), ((_block_size(), 1), 201.0)],
    )
    first = wcs.prelabel_censored_cells(raster)
    second = wcs.prelabel_censored_cells(raster)
    assert first == second
    assert first["censored_cell_ids"] == sorted(first["censored_cell_ids"])


def test_a_missing_prelabel_raster_creates_no_model_output(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    wcs.prelabel_raster_path(experiment_id, out).unlink()
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="model", to_stage="model",
            output_root=out, experiments_root=experiments,
            model_configuration_overrides=_FAST_BOOTSTRAP,
        )
    assert not wcs.model_root(experiment_id, out).exists()
    assert not wcs.model_staging_root(experiment_id, out).exists()
    assert _namespace_snapshot(out) == before


# =============================================================================
# 16-19. Shared folds and identical evaluations
# =============================================================================
def test_all_six_evaluations_use_identical_cells_and_folds(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    root = wcs.model_root(experiment_id, out)
    layout = wcs.model_relative_layout()
    folds = pd.read_parquet(root / layout["shared_folds"])
    expected = dict(zip(folds["cell_id"], folds["fold_id"]))

    tables = {}
    for variant_id in ["canonical", *_NONZERO]:
        for family in wcs.MODEL_FAMILIES:
            table = pd.read_parquet(
                root / wcs.model_variant_oof_relpath(variant_id, family)
            )
            tables[(variant_id, family)] = table
            assert sorted(table["cell_id"]) == sorted(folds["cell_id"])
            assert not table["cell_id"].duplicated().any()
            assert dict(zip(table["cell_id"], table["fold_id"])) == expected
            assert set(table["model_family"]) == {family}
            assert set(table["variant_id"]) == {variant_id}
    assert len(tables) == 6


def test_spatial_blocks_are_fold_disjoint(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    layout = wcs.model_relative_layout()
    folds = pd.read_parquet(wcs.model_root(experiment_id, out) / layout["shared_folds"])
    per_fold: dict[int, set] = {}
    for block, fold in zip(folds["spatial_block_id"], folds["fold_id"]):
        per_fold.setdefault(int(fold), set()).add(block)
    for a in per_fold:
        for b in per_fold:
            if a < b:
                assert not (per_fold[a] & per_fold[b])
    shared = _metadata(out, experiment_id)["shared_folds"]
    assert shared["block_disjointness_pass"] is True
    assert shared["every_row_assigned_once"] is True
    assert shared["assignment_sha256"]
    for field in ("fold_count", "random_seed", "spatial_block_definition",
                  "unique_block_count", "rows_per_fold", "positives_per_fold",
                  "negatives_per_fold"):
        assert field in shared, field


def test_every_row_gets_exactly_one_oof_prediction_per_evaluation(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    root = wcs.model_root(experiment_id, out)
    layout = wcs.model_relative_layout()
    cohort = pd.read_parquet(root / layout["common_cohort"])
    for variant_id in ["canonical", *_NONZERO]:
        for family in wcs.MODEL_FAMILIES:
            table = pd.read_parquet(
                root / wcs.model_variant_oof_relpath(variant_id, family)
            )
            assert len(table) == len(cohort)
            assert table["y_score"].notna().all()
            assert np.isfinite(table["y_score"].to_numpy()).all()


# =============================================================================
# 20-22. Registry, hyper-parameters, no adaptation
# =============================================================================
def test_the_feature_registry_is_the_production_one(tmp_path):
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, CATEGORICAL_FEATURES, THERMAL_MODEL_FEATURES,
    )

    _, experiment_id, out, _ = _run_model(tmp_path)
    registry = _metadata(out, experiment_id)["feature_registry"]
    assert registry["baseline_features_in_order"] == list(BASELINE_FEATURES)
    assert registry["thermal_model_features_in_order"] == list(THERMAL_MODEL_FEATURES)
    assert registry["categorical_features"] == list(CATEGORICAL_FEATURES)
    assert registry["source"] == "src.step8b_train_baseline_vs_thermal_model"


def test_hyperparameters_and_seeds_come_from_frozen_config():
    import core.config as config

    configuration = wcs.model_frozen_configuration()
    assert configuration["model"] == wcs.PRIMARY_MODEL
    assert configuration["n_splits"] == config.STEP8B_N_SPLITS
    assert configuration["fold_random_seed"] == config.STEP8B_RANDOM_SEED
    assert configuration["spatial_block_size_cells"] == config.STEP8B_SPATIAL_BLOCK_SIZE_CELLS
    assert configuration["min_positives"] == config.STEP8B_MIN_POSITIVES_PER_POPULATION
    assert configuration["n_bootstrap"] == config.STEP8C_N_BOOTSTRAP
    assert configuration["bootstrap_seed"] == config.STEP8C_RANDOM_SEED
    assert configuration["ci_lower_percentile"] == config.STEP8C_CI_LOWER
    assert configuration["ci_upper_percentile"] == config.STEP8C_CI_UPPER
    assert configuration["source"] == "core.config (frozen)"


def test_a_missing_frozen_bootstrap_parameter_fails_closed(monkeypatch):
    import core.config as config

    monkeypatch.setattr(config, "STEP8C_N_BOOTSTRAP", None, raising=False)
    with pytest.raises(wcs.WindowClosureError, match="STEP8C_N_BOOTSTRAP"):
        wcs.model_frozen_configuration()


#: Method names that must not appear anywhere in the serialized model-stage
#: record. The machine-readable statement is `calibration`/`adaptation` = null;
#: the prose must not enumerate method names, because a text scan cannot tell a
#: denial ("no CORAL") from a use ("CORAL applied").
BANNED_METHOD_TOKENS: tuple[str, ...] = (
    "coral", "z-score transfer", "recalibrat", "domain adapt",
)


def test_no_calibration_or_adaptation_runs(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    assert metadata["model_configuration"]["calibration"] is None
    assert metadata["model_configuration"]["adaptation"] is None
    blob = json.dumps(metadata, default=str).lower()
    for token in BANNED_METHOD_TOKENS:
        assert token not in blob, token


def test_the_serialized_metadata_file_contains_no_banned_method_token(tmp_path):
    """The bytes actually written to disk, not just the in-memory record."""
    _, experiment_id, out, _ = _run_model(tmp_path)
    raw = wcs.model_metadata_path(experiment_id, out).read_text(encoding="utf-8")
    lowered = raw.lower()
    for token in BANNED_METHOD_TOKENS:
        assert token not in lowered, token
    assert "no calibration, transfer adjustment, or alternative model procedure" in lowered


def test_the_limitations_prose_names_no_forbidden_method():
    """The prose source itself, independent of any run."""
    blob = " ".join(wcs.MODEL_LIMITATIONS).lower()
    for token in BANNED_METHOD_TOKENS:
        assert token not in blob, token
    assert "no calibration, transfer adjustment, or alternative model procedure" in blob


def test_calibration_and_adaptation_remain_the_machine_readable_statement(tmp_path):
    """Prose is descriptive; the contract is the two null configuration keys."""
    configuration = wcs.model_frozen_configuration()
    assert configuration["calibration"] is None
    assert configuration["adaptation"] is None

    _, experiment_id, out, _ = _run_model(tmp_path)
    recorded = _metadata(out, experiment_id)["model_configuration"]
    assert recorded["calibration"] is None
    assert recorded["adaptation"] is None


def test_the_prose_wording_does_not_affect_the_model_results(tmp_path):
    """Every reported number still recomputes from the saved artefacts.

    The limitations text is metadata only: it is not read by the cohort, the
    folds, the fit, the metrics or the bootstrap, so editing it cannot move a
    result. This asserts that end to end rather than assuming it.
    """
    from src.step8b_train_baseline_vs_thermal_model import compute_binary_metrics

    _, experiment_id, out, _ = _run_model(tmp_path)
    root = wcs.model_root(experiment_id, out)
    metadata = _metadata(out, experiment_id)

    # The prose is carried, and is not part of any computed payload.
    assert metadata["limitations"] == list(wcs.MODEL_LIMITATIONS)
    for section in ("model_configuration", "feature_registry", "common_cohort",
                    "shared_folds", "point_metrics", "thermal_contributions",
                    "comparisons", "bootstrap"):
        assert "limitations" not in json.dumps(metadata[section], default=str)

    # ...and every reported metric still recomputes from the saved predictions.
    for row in metadata["point_metrics"]:
        table = pd.read_parquet(
            root / wcs.model_variant_oof_relpath(row["variant_id"], row["model_family"])
        ).sort_values("cell_id", kind="mergesort")
        recomputed = compute_binary_metrics(
            table["y_true"].astype(int).to_numpy(),
            table["y_score"].to_numpy(dtype="float64"),
        )
        assert row["roc_auc"] == pytest.approx(recomputed["roc_auc"])
        assert row["pr_auc"] == pytest.approx(recomputed["pr_auc"])
        assert row["brier"] == pytest.approx(recomputed["brier_score"])

    # ...and every interval status still follows from its own interval.
    for row in metadata["comparisons"]:
        assert row["status"] == wcs.classify_change_interval(row["ci_low"], row["ci_high"])


# =============================================================================
# 23-28. Point metrics and sign conventions
# =============================================================================
def test_point_metrics_recompute_from_the_saved_predictions(tmp_path):
    from src.step8b_train_baseline_vs_thermal_model import compute_binary_metrics

    _, experiment_id, out, _ = _run_model(tmp_path)
    root = wcs.model_root(experiment_id, out)
    metadata = _metadata(out, experiment_id)
    recorded = {
        (row["variant_id"], row["model_family"]): row
        for row in metadata["point_metrics"]
    }
    assert len(recorded) == 6
    for (variant_id, family), row in recorded.items():
        table = pd.read_parquet(
            root / wcs.model_variant_oof_relpath(variant_id, family)
        ).sort_values("cell_id", kind="mergesort")
        recomputed = compute_binary_metrics(
            table["y_true"].astype(int).to_numpy(),
            table["y_score"].to_numpy(dtype="float64"),
        )
        assert row["roc_auc"] == pytest.approx(recomputed["roc_auc"])
        assert row["pr_auc"] == pytest.approx(recomputed["pr_auc"])
        assert row["brier"] == pytest.approx(recomputed["brier_score"])
        assert row["row_count"] == len(table)
        assert row["fold_count"] == metadata["shared_folds"]["fold_count"]
        assert row["prevalence"] == pytest.approx(
            row["positive_count"] / row["row_count"]
        )


def test_thermal_contribution_is_the_raw_thermal_minus_baseline_delta(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    points = {
        (row["variant_id"], row["model_family"]): row for row in metadata["point_metrics"]
    }
    for row in metadata["thermal_contributions"]:
        expected = (
            points[(row["variant_id"], "thermal")][row["metric"]]
            - points[(row["variant_id"], "baseline")][row["metric"]]
        )
        assert row["contribution_delta"] == pytest.approx(expected)
        assert row["delta_definition"] == "thermal - baseline (raw)"


def test_the_brier_sign_convention_is_documented(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    convention = metadata["brier_sign_convention"].lower()
    assert "raw" in convention
    assert "negative" in convention and "better" in convention
    assert metadata["metric_sign_conventions"]["brier"] == metadata["brier_sign_convention"]


def test_closure_deltas_are_earlier_minus_canonical(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    points = {
        (row["variant_id"], row["model_family"]): row for row in metadata["point_metrics"]
    }
    closure = [
        row for row in metadata["comparisons"]
        if row["comparison"] == wcs.COMPARISON_CLOSURE_CHANGE
    ]
    assert closure
    assert {row["variant_id"] for row in closure} == set(_NONZERO)
    for row in closure:
        assert row["delta_definition"] == "earlier_closure - canonical (raw)"
        expected = (
            points[(row["variant_id"], row["model_family"])][row["metric"]]
            - points[("canonical", row["model_family"])][row["metric"]]
        )
        assert row["point_delta"] == pytest.approx(expected)


def test_all_three_comparison_families_are_produced(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    families = {row["comparison"] for row in metadata["comparisons"]}
    assert families == {
        wcs.COMPARISON_THERMAL_CONTRIBUTION, wcs.COMPARISON_CLOSURE_CHANGE,
        wcs.COMPARISON_CONTRIBUTION_CHANGE,
    }
    for row in metadata["comparisons"]:
        for field in ("point_delta", "ci_low", "ci_high", "confidence_level",
                      "requested_replicates", "valid_replicates",
                      "invalid_replicates", "bootstrap_seed", "block_count",
                      "status"):
            assert field in row, field


# =============================================================================
# 29-36. Paired bootstrap
# =============================================================================
def test_the_bootstrap_draws_are_paired_and_refit_nothing(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    assert metadata["bootstrap"]["identical_block_draws_across_variants"] is True
    assert metadata["bootstrap_models_refit_per_replicate"] is False
    root = wcs.model_root(experiment_id, out)
    replicates = pd.read_parquet(
        root / wcs.model_relative_layout()["bootstrap_replicates"]
    )
    # One row per replicate, carrying every variant's metrics -> shared draws.
    for variant_id in ["canonical", *_NONZERO]:
        for suffix in ("baseline_roc_auc", "thermal_roc_auc", "delta_brier"):
            assert f"{variant_id}__{suffix}" in replicates.columns
    assert len(replicates) == metadata["bootstrap"]["n_bootstrap_valid"]


def test_replicate_counts_are_truthful(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    requested = metadata["bootstrap"]["n_bootstrap_requested"]
    valid = metadata["bootstrap"]["n_bootstrap_valid"]
    assert requested == _FAST_BOOTSTRAP["n_bootstrap"]
    assert metadata["bootstrap_invalid_replicates"] == requested - valid
    for row in metadata["comparisons"]:
        assert row["requested_replicates"] == requested
        assert row["valid_replicates"] + row["invalid_replicates"] == requested


def test_a_single_class_replicate_is_counted_invalid_not_zero():
    """A replicate whose draw carries one class leaves the metric undefined."""
    from src.step8c_spatial_block_bootstrap_uncertainty import compute_metrics

    labels = np.zeros(8, dtype=int)
    assert compute_metrics(labels, np.full(8, 0.5), np.full(8, 0.5)) is None


def test_too_few_valid_replicates_fails_closed():
    configuration = dict(wcs.model_frozen_configuration())
    bootstrap = {"n_bootstrap_valid": 0, "n_bootstrap_requested": 100}
    with pytest.raises(wcs.WindowClosureError, match="valid replicate"):
        wcs.assert_bootstrap_sufficient(bootstrap, configuration)
    wcs.assert_bootstrap_sufficient(
        {"n_bootstrap_valid": 1, "n_bootstrap_requested": 100}, configuration,
    )


@pytest.mark.parametrize("low,high,expected", [
    (0.01, 0.05, "bootstrap_supported_increase"),
    (-0.05, -0.01, "bootstrap_supported_decrease"),
    (-0.02, 0.03, "interval_includes_zero"),
    (None, None, "interval_includes_zero"),
])
def test_interval_status_wording(low, high, expected):
    assert wcs.classify_change_interval(low, high) == expected


def test_only_allowed_status_wording_is_used(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    allowed = {
        wcs.INTERVAL_SUPPORTED_INCREASE, wcs.INTERVAL_SUPPORTED_DECREASE,
        wcs.INTERVAL_INCLUDES_ZERO,
    }
    assert set(metadata["allowed_statuses"]) == allowed
    assert {row["status"] for row in metadata["comparisons"]} <= allowed
    blob = json.dumps(metadata, default=str).lower()
    for banned in ("statistically significant", "statistical significance",
                   "p-value", "equivalent performance"):
        assert banned not in blob


def test_every_status_follows_from_its_own_interval(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    for row in _metadata(out, experiment_id)["comparisons"]:
        assert row["status"] == wcs.classify_change_interval(row["ci_low"], row["ci_high"])


# =============================================================================
# 37, 38. Dry run
# =============================================================================
def test_the_dry_run_writes_nothing_and_fits_nothing(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    before = _namespace_snapshot(out)
    result, _, _, _ = _dry_run_model(tmp_path, env=(experiment_id, out, experiments))

    assert result["ran"] is False
    assert result["dry_run"] is True
    assert result["planned_stages"] == ["model"]
    assert result["files_written"] is False
    summary = result["model_stage_summary"]
    for flag in ("model_fit", "fire_risk_model_fit", "bootstrap_run",
                 "common_cohort_created", "compare_run", "compare_planned",
                 "downscaling_model_fit", "gee_queries_run", "gee_exports_run"):
        assert summary[flag] is False, flag
    for flag in ("fire_risk_model_fit_planned", "common_cohort_creation_planned",
                 "shared_folds_planned", "paired_spatial_block_bootstrap_planned"):
        assert summary[flag] is True, flag
    assert not wcs.model_root(experiment_id, out).exists()
    assert not wcs.model_staging_root(experiment_id, out).exists()
    assert _namespace_snapshot(out) == before


def test_the_dry_run_binds_exactly_three_datasets(tmp_path):
    result, experiment_id, out, experiments = _dry_run_model(tmp_path)
    summary = result["model_stage_summary"]
    assert sorted(summary["input_datasets"]) == sorted(["canonical", *_NONZERO])
    assert summary["expected_input_dataset_count"] == 3
    assert summary["input_binding_ready"] is True
    assert summary["model_evaluation_count_planned"] == 6
    assert summary["primary_population"] == wcs.PRIMARY_POPULATION
    assert summary["prelabel_censor"]["raster_present"] is True
    assert summary["all_paths_inside_model_namespace"] is True
    for variant_id, record in summary["input_datasets"].items():
        assert record["dataset_sha256"] == wcs.sha256_file(Path(record["dataset_path"]))


def test_the_dry_run_snapshot_of_an_existing_model_tree_is_unchanged(tmp_path):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    before = _namespace_snapshot(out)
    result, _, _, _ = _dry_run_model(tmp_path, env=(experiment_id, out, experiments))
    assert result["model_stage_owned_snapshot_unchanged"] is True
    assert result["model_dry_run_created_paths"] == []
    assert result["model_dry_run_modified_paths"] == []
    assert result["model_dry_run_deleted_paths"] == []
    assert result["preexisting_model_stage_owned_paths"]
    assert _namespace_snapshot(out) == before


# =============================================================================
# 39-45. Normal / resume / force and atomicity
# =============================================================================
def test_a_plain_rerun_rejects_an_existing_pass_output(tmp_path):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match="Refusing to overwrite"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="model", to_stage="model",
            output_root=out, experiments_root=experiments,
            model_configuration_overrides=_FAST_BOOTSTRAP,
        )
    assert _namespace_snapshot(out) == before


def test_resume_reuses_a_valid_pass_output_without_mutation(tmp_path):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    before = _namespace_snapshot(out)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="model", to_stage="model", resume=True,
        output_root=out, experiments_root=experiments,
        model_configuration_overrides=_FAST_BOOTSTRAP,
    )
    assert result["model_reused"] is True
    assert _namespace_snapshot(out) == before


def test_resume_rejects_a_partial_output_without_mutation(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    partial = wcs.model_root(experiment_id, out) / "metrics"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "point_metrics.csv").write_bytes(b"partial")
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match="cannot reuse the model stage"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="model", to_stage="model", resume=True,
            output_root=out, experiments_root=experiments,
            model_configuration_overrides=_FAST_BOOTSTRAP,
        )
    assert _namespace_snapshot(out) == before
    assert (partial / "point_metrics.csv").read_bytes() == b"partial"


def test_a_plain_rerun_rejects_a_partial_output_without_mutation(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    partial = wcs.model_root(experiment_id, out) / "metrics"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "point_metrics.csv").write_bytes(b"partial")
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match="NOT reusable"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="model", to_stage="model",
            output_root=out, experiments_root=experiments,
            model_configuration_overrides=_FAST_BOOTSTRAP,
        )
    assert _namespace_snapshot(out) == before


def test_resume_and_force_stay_mutually_exclusive(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="mutually exclusive"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="model", to_stage="model", resume=True, force=True,
            output_root=out, experiments_root=experiments,
        )


def test_force_quarantines_only_the_model_tree_and_preserves_upstream(tmp_path):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    upstream_before = _upstream_snapshot(experiment_id, out, experiments)
    old_metadata = wcs.model_metadata_path(experiment_id, out).read_bytes()

    result, _, _, _ = _run_model(
        tmp_path, env=(experiment_id, out, experiments), force=True,
    )
    manifest = result["quarantine_manifest"]
    assert manifest["quarantined"] is True
    entry = manifest["entries"][0]
    for field in ("original_path", "quarantined_path", "reason", "timestamp_utc",
                  "pre_quarantine_inventory_sha256"):
        assert entry[field], field
    quarantined = Path(entry["quarantined_path"]) / wcs.MODEL_METADATA_NAME
    assert quarantined.read_bytes() == old_metadata
    assert wcs.model_metadata_path(experiment_id, out).is_file()
    assert _upstream_snapshot(experiment_id, out, experiments) == upstream_before


def test_a_failure_writes_no_pass_metadata_and_leaves_no_staging(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    before = _namespace_snapshot(out)
    # A cohort that cannot satisfy the frozen minimum-positives requirement.
    with pytest.raises(wcs.WindowClosureError):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="model", to_stage="model",
            output_root=out, experiments_root=experiments,
            model_configuration_overrides={
                "n_bootstrap": 8, "min_positives": 10 ** 9,
            },
        )
    assert not wcs.model_root(experiment_id, out).exists()
    assert not wcs.model_staging_root(experiment_id, out).exists()
    assert _namespace_snapshot(out) == before


# =============================================================================
# 46-50. Namespace, provenance and truthful metadata
# =============================================================================
def test_every_output_stays_inside_the_model_root(tmp_path):
    result, experiment_id, out, _ = _run_model(tmp_path)
    root = wcs.model_root(experiment_id, out).resolve()
    for path in result["files_written"]:
        assert root in Path(path).resolve().parents or Path(path).resolve() == root
    metadata = _metadata(out, experiment_id)
    for record in metadata["artifact_inventory"]:
        assert root in Path(record["path"]).resolve().parents
    assert metadata["all_paths_inside_model_namespace"] is True
    assert not (out / experiment_id / "comparison").exists()


def test_upstream_and_canonical_outputs_are_untouched(tmp_path):
    experiment_id, out, experiments = _model_env(tmp_path)
    before = _upstream_snapshot(experiment_id, out, experiments)
    _run_model(tmp_path, env=(experiment_id, out, experiments))
    assert _upstream_snapshot(experiment_id, out, experiments) == before


def test_no_compare_artifact_is_produced(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    root = wcs.model_root(experiment_id, out)
    offending = [
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and "compare" in p.name.lower()
    ]
    assert offending == []
    assert _metadata(out, experiment_id)["compare_run"] is False


def test_the_metadata_distinguishes_fire_risk_from_downscaling_models(tmp_path):
    result, experiment_id, out, _ = _run_model(tmp_path)
    metadata = _metadata(out, experiment_id)
    for payload in (metadata, result):
        assert payload["model_fit"] is True
        assert payload["fire_risk_model_fit"] is True
        assert payload["fire_risk_model_stage_run"] is True
        assert payload["downscaling_model_fit"] is False
        assert payload["downscaling_model_refit"] is False
        assert payload["bootstrap_run"] is True
        assert payload["common_cohort_created"] is True
        assert payload["compare_run"] is False
        assert payload["gee_queries_run"] is False
        assert payload["gee_exports_run"] is False


def test_no_step7c_downscaling_model_is_refit(tmp_path):
    """The autouse guard fails the test if Step7C is ever invoked."""
    import src.step7c_train_downscaling_model as step7c

    _run_model(tmp_path)
    with pytest.raises(AssertionError, match="fail-closed guard"):
        step7c.run_step7c()


def test_the_full_artifact_layout_is_produced(tmp_path):
    _, experiment_id, out, _ = _run_model(tmp_path)
    root = wcs.model_root(experiment_id, out)
    for relative in wcs.model_relative_layout().values():
        assert (root / relative).is_file(), relative
    for variant_id in ["canonical", *_NONZERO]:
        for family in wcs.MODEL_FAMILIES:
            assert (root / wcs.model_variant_oof_relpath(variant_id, family)).is_file()
            assert (root / wcs.model_variant_metrics_relpath(variant_id, family)).is_file()


# =============================================================================
# Validator
# =============================================================================
def test_validator_reports_the_stage_lock(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run_model(tmp_path)
    _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "l.log", result))
    captured = capsys.readouterr().out
    assert "[PASS] model is an implemented actual stage" in captured
    assert "[PASS] no unimplemented stage is reachable" in captured


def test_validator_accepts_a_valid_dry_run(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run_model(tmp_path)
    code = _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result))
    output = capsys.readouterr().out
    assert code == 0, output
    assert "OVERALL STATUS: PASS" in output


def test_validator_rejects_a_dry_run_that_planned_a_fit(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run_model(tmp_path)
    result["model_stage_summary"]["fire_risk_model_fit"] = True
    assert _validate(
        "dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result),
    ) == 1
    assert "[FAIL] fire_risk_model_fit is false in the dry run" in capsys.readouterr().out


def test_validator_rejects_a_dry_run_with_a_wrong_dataset_count(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run_model(tmp_path)
    result["model_stage_summary"]["input_datasets"].pop("close_14d_earlier")
    assert _validate(
        "dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result),
    ) == 1
    assert "[FAIL] exactly the three Step8A datasets are bound" in capsys.readouterr().out


def test_validator_accepts_a_valid_actual_run(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    code = _validate("actual", experiment_id, out, experiments=experiments)
    output = capsys.readouterr().out
    assert code == 0, output
    assert "OVERALL STATUS: PASS" in output
    assert "every recorded point metric recomputes" in output


def test_validator_actual_rejects_a_tampered_point_metric(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    path = wcs.model_metadata_path(experiment_id, out)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["point_metrics"][0]["roc_auc"] = 0.999999
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "every recorded point metric recomputes" in capsys.readouterr().out


def test_validator_actual_rejects_a_status_wording_violation(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    path = wcs.model_metadata_path(experiment_id, out)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["comparisons"][0]["status"] = "statistically significant increase"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    output = capsys.readouterr().out
    assert "no unsupported inferential wording is used" in output


def test_validator_actual_rejects_a_drifted_input_dataset(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    dataset = wcs.variant_step8a_dataset_path(experiment_id, "close_7d_earlier", out)
    dataset.write_bytes(dataset.read_bytes() + b"drift")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "every contributing Step8A dataset is unchanged" in capsys.readouterr().out


def test_validator_actual_rejects_a_compare_artifact(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    stray = wcs.model_root(experiment_id, out) / "metrics" / "compare_summary.csv"
    stray.write_bytes(b"stray")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "no compare-stage artefact exists" in capsys.readouterr().out


def test_validator_actual_rejects_a_broken_fold_sharing(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    path = (
        wcs.model_root(experiment_id, out)
        / wcs.model_variant_oof_relpath("close_7d_earlier", "thermal")
    )
    table = pd.read_parquet(path)
    table.loc[0, "fold_id"] = int(table.loc[0, "fold_id"]) + 1
    table.to_parquet(path, index=False)
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "share the cohort" in capsys.readouterr().out


def test_validator_never_writes(tmp_path):
    _, experiment_id, out, experiments = _run_model(tmp_path)
    before = _namespace_snapshot(out)
    _validate("actual", experiment_id, out, experiments=experiments)
    assert _namespace_snapshot(out) == before


def test_no_aoi_or_date_is_hard_coded_in_the_model_code():
    import re

    REGISTRY_IDS = ld.REGISTRY_IDS
    _executable_string_literals = ld._executable_string_literals

    for module_path in (
        _PROJECT_ROOT / "src" / "window_closure_sensitivity.py",
        _PROJECT_ROOT / "scripts" / "validate_window_closure_model.py",
    ):
        literals = _executable_string_literals(module_path)
        for experiment_id in REGISTRY_IDS:
            assert not [s for s in literals if experiment_id in s]
        assert [s for s in literals if re.search(r"(19|20)\d\d-\d\d-\d\d", s)] == []
