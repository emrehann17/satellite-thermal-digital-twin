"""Tests for the window-closure LOCAL-DOWNSTREAM stage
(src/window_closure_sensitivity.py + scripts/validate_window_closure_local_downstream.py).

Everything is synthetic and runs under tmp_path, injected through the module's
public `output_root` / `experiments_root` parameters -- never by monkeypatching
another module's globals. The production downstream chain is replaced by an
injected fake engine, so no test touches the real Step5/Step5C/Step7/Step8A
code, the real canonical outputs or Earth Engine. Experiment IDs come from the
registry dynamically, so no AOI name is hard-coded here either.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import Affine

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.validate_window_closure_local_downstream as validator  # noqa: E402
import src.window_closure_sensitivity as wcs  # noqa: E402
from core.experiment_context import build_experiment_context  # noqa: E402
from core.pipeline_orchestrator import LEGACY_EXPERIMENT_ID  # noqa: E402
from core.regions import get_experiment, list_experiments  # noqa: E402

REGISTRY_IDS = tuple(
    sorted(e for e in list_experiments(include_disabled=False) if e != LEGACY_EXPERIMENT_ID)
)

_TEST_TRANSFORM = Affine(0.00026949458523585647, 0.0, 31.05,
                         0.0, -0.00026949458523585647, 37.35)
_TEST_SHAPE = (6, 5)
_MODIS_PIXEL_FACTOR = wcs.MODIS_EXPORT_SCALE_M / wcs.LANDSAT_EXPORT_SCALE_M
_MODIS_TRANSFORM = Affine(
    0.00026949458523585647 * _MODIS_PIXEL_FACTOR, 0.0, 31.05,
    0.0, -0.00026949458523585647 * _MODIS_PIXEL_FACTOR, 37.35,
)
_PREDICTOR_NODATA = -9999.0
_SHIFTS = (0, 7, 14)
_NONZERO = ("close_7d_earlier", "close_14d_earlier")


def any_experiment() -> str:
    for experiment_id in REGISTRY_IDS:
        experiment = get_experiment(experiment_id)
        if experiment.get("predictor_start_date") and experiment.get("label_start_date"):
            return experiment_id
    pytest.skip("no registry experiment with a defined predictor window")


def ctx_for(experiment_id: str) -> dict:
    return build_experiment_context(experiment_id)


def _baseline_years(experiment_id: str) -> list[int]:
    return [
        int(year)
        for year in wcs.canonical_window(ctx_for(experiment_id))["baseline_years"]
    ]


# =============================================================================
# Fail-closed guard: no test in this module may reach production Earth Engine
# or the real production downstream chain.
# =============================================================================
@pytest.fixture(autouse=True)
def _no_production_side_effects(monkeypatch):
    def _blocked(name):
        def _fail(*_args, **_kwargs):
            raise AssertionError(
                f"fail-closed guard: production {name} was invoked from a unit "
                "test; inject a fake exporter / engine instead"
            )
        return _fail

    import core.gee_utils as gee_utils
    import scripts.prepare_modis_for_step7 as prepare_modis
    import scripts.run_predictors_only as run_predictors_only
    import src.step6_validate_fire_relation as step6

    monkeypatch.setattr(
        step6, "export_raw_mcd64a1_prelabel_labels",
        _blocked("Step6 prelabel exporter"),
    )
    monkeypatch.setattr(gee_utils, "init_gee", _blocked("init_gee"))
    monkeypatch.setattr(
        run_predictors_only, "export_image_direct_or_tiled", _blocked("GEE exporter"),
    )
    monkeypatch.setattr(
        prepare_modis, "prepare_modis_for_step7", _blocked("production MODIS exporter"),
    )
    monkeypatch.setattr(
        wcs, "production_predictor_engine", _blocked("production predictor engine"),
    )
    monkeypatch.setattr(
        wcs, "production_local_downstream_engine",
        _blocked("production local-downstream engine"),
    )
    if "geemap" in sys.modules:
        monkeypatch.setattr(
            sys.modules["geemap"], "ee_export_image", _blocked("geemap download"),
            raising=False,
        )
    else:
        guard = types.ModuleType("geemap")
        guard.ee_export_image = _blocked("geemap download")
        monkeypatch.setitem(sys.modules, "geemap", guard)


# =============================================================================
# Synthetic environment
# =============================================================================
def _write_raster(path: Path, values=None, *, transform=None, crs="EPSG:4326",
                  dtype="float32", nodata=_PREDICTOR_NODATA, bands: int = 1) -> Path:
    import rasterio

    array = np.asarray(
        np.full(_TEST_SHAPE, 3.0) if values is None else values, dtype=dtype,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=bands, dtype=dtype, crs=crs,
        transform=transform if transform is not None else _TEST_TRANSFORM,
        nodata=nodata,
    ) as dst:
        for band in range(1, bands + 1):
            dst.write(array, band)
    return path


def _canonical_reference_grid() -> dict:
    return {
        "path": "synthetic-reference",
        "width": _TEST_SHAPE[1],
        "height": _TEST_SHAPE[0],
        "crs": "EPSG:4326",
        "transform": [float(v) for v in tuple(_TEST_TRANSFORM)[:6]],
    }


def _canonical_predictor_paths(root: Path) -> dict[str, str]:
    """The production Step8A lineage, mirroring the canonical stats layout."""
    return {
        "ndvi": str(root / "data" / "ndvi_current_period" / "current_ndvi_median.tif"),
        "elevation": str(root / "data" / "dem" / "elevation.tif"),
        "slope": str(root / "data" / "dem" / "slope.tif"),
        "lst_anomaly": str(root / "step5" / "anomaly_zscore.tif"),
        "current_lst": str(root / "step5" / "current_period_median_celsius.tif"),
        "current_tvdi": str(root / "step5c" / "current_tvdi.tif"),
        "tvdi_difference": str(root / "step5c" / "tvdi_difference.tif"),
        "downscaled_lst": str(root / "step7d" / "downscaled_lst_celsius.tif"),
        "fused_lst": str(root / "step7e" / "fused_lst_celsius.tif"),
    }


TIMING_PREFIXES = (
    "ndvi", "lst_anomaly", "current_lst", "current_tvdi", "tvdi_difference",
    "downscaled_lst", "fused_lst",
)
STATIC_PREFIXES = ("elevation", "slope")


def _canonical_frame(rows: int = 12) -> pd.DataFrame:
    """A synthetic Step8A dataset carrying exactly the production column groups.

    The column list is built from the module's own classification constants and
    the lineage prefixes, so it can never drift from what the implementation
    classifies.
    """
    data: dict[str, np.ndarray] = {}
    index = np.arange(rows)
    data["cell_id"] = np.array([f"r{i}_c{i * 2}" for i in index], dtype=object)
    data["row_500m"] = index.astype("int64")
    data["col_500m"] = (index * 2).astype("int64")
    data["lon"] = (31.0 + index * 0.01).astype("float64")
    data["lat"] = (37.0 + index * 0.01).astype("float64")

    data["pre_label_burn_excluded"] = np.zeros(rows, dtype=bool)
    data["analysis_eligible"] = np.ones(rows, dtype=bool)
    data["burned"] = (index % 3 == 0).astype("int64")
    data["burn_date"] = np.where(index % 3 == 0, 210.0, np.nan).astype("float64")
    data["burn_month"] = np.where(index % 3 == 0, 8, 0).astype("int64")
    data["burn_day_of_year"] = data["burn_date"]
    data["label_source"] = np.array(["mcd64a1_raw"] * rows, dtype=object)
    data["burn_date_pixel_agreement_fraction"] = np.full(rows, 0.9, dtype="float64")
    data["out_of_window_burndate"] = np.zeros(rows, dtype=bool)

    for prefix in TIMING_PREFIXES + STATIC_PREFIXES:
        for suffix in wcs.STEP8A_PREDICTOR_COLUMN_SUFFIXES:
            column = f"{prefix}{suffix}"
            if suffix == "_valid_count":
                data[column] = np.full(rows, 17, dtype="int64")
            else:
                data[column] = (index * 0.5 + len(prefix)).astype("float64")

    data["landcover_dominant"] = np.full(rows, 10, dtype="int64")
    for name in (
        "tree_cover", "shrubland", "grassland", "cropland",
        "bare_sparse_vegetation", "built_up", "permanent_water",
    ):
        data[f"landcover_{name}_fraction"] = np.full(rows, 1.0 / 7.0, dtype="float64")
    data["burnable_tree_shrub_grass"] = (index % 4 != 3)
    data["burnable_tree_shrub"] = (index % 5 != 4)

    data["valid_30m_pixel_count"] = np.full(rows, 280, dtype="int64")
    data["total_30m_pixel_count"] = np.full(rows, 289, dtype="int64")
    data["valid_30m_fraction"] = np.full(rows, 0.97, dtype="float64")
    data["observed_fraction"] = np.full(rows, 0.8, dtype="float64")
    data["gapfilled_fraction"] = np.full(rows, 0.2, dtype="float64")
    data["invalid_source_fraction"] = np.zeros(rows, dtype="float64")
    # Production writes this as `int(code)` or NaN, so a canonical dataset that
    # has at least one cell without a valid source pixel comes back as float64.
    # The fixture mirrors that: it is what makes the int64/float64 difference a
    # representation difference rather than a semantic one.
    data["source_mask_majority"] = np.where(
        index % 7 == 6, np.nan, 1.0,
    ).astype("float64")
    data["thermal_any_missing"] = np.zeros(rows, dtype=bool)
    data["valid_for_modeling"] = np.ones(rows, dtype=bool)
    data["invalid_reason"] = np.array([""] * rows, dtype=object)
    return pd.DataFrame(data)


def _variant_frame(canonical: pd.DataFrame, *, drop_rows: int = 2) -> pd.DataFrame:
    """A legitimate variant dataset: timing-derived features moved, rows dropped."""
    frame = canonical.iloc[: len(canonical) - drop_rows].copy()
    for prefix in TIMING_PREFIXES:
        for suffix in ("_mean", "_median", "_std", "_valid_fraction"):
            frame[f"{prefix}{suffix}"] = frame[f"{prefix}{suffix}"] + 1.25
        frame[f"{prefix}_valid_count"] = frame[f"{prefix}_valid_count"] - 1
    frame["valid_30m_pixel_count"] = frame["valid_30m_pixel_count"] - 3
    frame["valid_30m_fraction"] = frame["valid_30m_fraction"] - 0.01
    frame["observed_fraction"] = frame["observed_fraction"] - 0.05
    return frame.reset_index(drop=True)


def _seed_frozen_inputs(
    experiments: Path, experiment_id: str, canonical: Optional[pd.DataFrame] = None,
) -> dict[str, Path]:
    """Every REQUIRED frozen input, as a REAL file of the right kind.

    `canonical` lets a caller (e.g. the model-stage tests, which need a cohort
    large enough for the frozen minimum-positives and 5-fold requirements) seed
    its own canonical Step8A dataset.
    """
    inventory = wcs.frozen_input_inventory(experiment_id, experiments)
    written: dict[str, Path] = {}
    canonical = _canonical_frame() if canonical is None else canonical
    for role in wcs.REQUIRED_FROZEN_INPUT_ROLES:
        path = Path(inventory[role]["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == "canonical_step8a":
            canonical.to_parquet(path, index=False)
        else:
            _write_raster(path, np.zeros(_TEST_SHAPE), dtype="float32")
        written[role] = path

    root = wcs.canonical_experiment_root(experiment_id, experiments)
    stats = wcs.canonical_step8a_stats_path(experiment_id, experiments)
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text(json.dumps({
        "step": "step8a",
        "label_kind": wcs.LABEL_KIND_RAW_BURNDATE,
        "experiment_id": experiment_id,
        "reference_30m_grid": _canonical_reference_grid(),
        "predictor_paths": _canonical_predictor_paths(root),
    }, indent=2, sort_keys=True), encoding="utf-8")
    written["canonical_step8a_stats"] = stats
    return written


def _fake_prelabel_exporter(experiment_id, pre_label_start, pre_label_end, raw_out):
    _write_raster(Path(raw_out), np.zeros(_TEST_SHAPE), dtype="float32")
    return {"raw_path": Path(raw_out), "experiment_id": experiment_id}


def _fake_predictor_engine(variant_context, variant, jobs):
    """Writes contract-valid predictor rasters. Never touches Earth Engine."""
    results = {}
    for job in jobs:
        path = Path(job["output_path"])
        transform = (
            _MODIS_TRANSFORM if job["grid_family"] == wcs.GRID_FAMILY_MODIS
            else _TEST_TRANSFORM
        )
        values = (
            np.full(_TEST_SHAPE, 3.0) if job["is_count_product"]
            else np.full(_TEST_SHAPE, 21.5)
        )
        _write_raster(path, values, transform=transform, dtype="float32")
        results[job["artifact_id"]] = {"path": path, "transport": "direct"}
    return results


# --- The fake production downstream chain ------------------------------------
FAKE_STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "step5": ("current_period_median_celsius.tif", "anomaly_zscore.tif", "step5_metadata.json"),
    "step5c": ("current_tvdi.tif", "tvdi_difference.tif", "step5c_metadata.json"),
    "step7a": ("tiling_test_summary.json",),
    "step7b": ("downscaling_dataset_stats.json",),
    "step7c": ("downscaling_model_metrics.json",),
    "step7d": ("downscaled_lst_celsius.tif",),
    "step7e": ("fused_lst_celsius.tif", "fused_lst_source_mask.tif"),
    "step8a": (wcs.STEP8A_DATASET_NAME, wcs.STEP8A_STATS_NAME),
}


def _fake_downstream_engine(
    canonical: pd.DataFrame, *, calls: Optional[list] = None,
    fail_variants: Sequence[str] = (), frames: Optional[dict] = None,
    stats_override: Optional[dict] = None, stages: Optional[Sequence[str]] = None,
    skip_step8a_dataset: bool = False,
):
    """Stand-in for the production Step5/Step5C/Step7/Step8A chain."""
    def engine(variant_context, variant, plan):
        variant_id = variant["variant_id"]
        if calls is not None:
            calls.append({
                "variant_id": variant_id,
                "predictor_start_date": variant_context["predictor_start_date"],
                "predictor_end_date": variant_context["predictor_end_date"],
                "context": dict(variant_context),
                "plan": plan,
            })
        if variant_id in fail_variants:
            raise wcs.WindowClosureError(f"synthetic engine failure for {variant_id}")

        ran = list(stages) if stages is not None else list(wcs.PRODUCTION_STAGE_SEQUENCE)
        for stage in ran:
            stage_dir = Path(variant_context[f"{stage}_output_dir"])
            stage_dir.mkdir(parents=True, exist_ok=True)
            for name in FAKE_STAGE_OUTPUTS[stage]:
                target = stage_dir / name
                if name.endswith(".tif"):
                    _write_raster(target, np.full(_TEST_SHAPE, 12.5), dtype="float32")
                elif name == wcs.STEP8A_DATASET_NAME:
                    if skip_step8a_dataset:
                        continue
                    frame = (frames or {}).get(variant_id)
                    if frame is None:
                        frame = _variant_frame(canonical)
                    frame.to_parquet(target, index=False)
                elif name == wcs.STEP8A_STATS_NAME:
                    payload = {
                        "step": "step8a",
                        "experiment_id": variant_context["experiment_id"],
                        "reference_30m_grid": (
                            stats_override or _canonical_reference_grid()
                        ),
                    }
                    target.write_text(
                        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
                    )
                else:
                    target.write_text(
                        json.dumps({"stage": stage, "schema_version": f"{stage}.v1"},
                                   indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
        return {"stages_run": ran}
    return engine


# --- Environments -------------------------------------------------------------
def _predictor_env(
    tmp_path: Path, shifts=_SHIFTS, canonical: Optional[pd.DataFrame] = None,
) -> tuple[str, Path, Path]:
    """A namespace with plan, prelabel-export and predictor-export completed."""
    experiment_id = any_experiment()
    experiments = tmp_path / "experiments"
    out = tmp_path / "diagnostics"
    _seed_frozen_inputs(experiments, experiment_id, canonical)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(shifts), dry_run=False,
        from_stage="plan", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_prelabel_exporter,
        predictor_engine=_fake_predictor_engine,
    )
    return experiment_id, out, experiments


def _run_local_downstream(tmp_path: Path, *, shifts=_SHIFTS, engine=None,
                          env=None, **kwargs):
    experiment_id, out, experiments = env or _predictor_env(tmp_path, shifts)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(shifts), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=(
            engine if engine is not None else _fake_downstream_engine(canonical)
        ),
        **kwargs,
    )
    return result, experiment_id, out, experiments


def _exploding_engine(*_args, **_kwargs):
    raise AssertionError("the local-downstream engine must not be reached")


def _relative_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _metadata(out: Path, experiment_id: str, variant_id: str) -> dict:
    return json.loads(
        (out / experiment_id / "variants" / variant_id
         / wcs.LOCAL_DOWNSTREAM_METADATA_NAME).read_text(encoding="utf-8")
    )


# =============================================================================
# 1-5. Stage lock
# =============================================================================
def test_local_downstream_is_an_implemented_actual_stage():
    assert wcs.IMPLEMENTED_ACTUAL_STAGES == wcs.STAGES
    assert wcs.LOCAL_DOWNSTREAM_STAGE == "local-downstream"


def test_local_downstream_single_stage_range_is_supported():
    wcs.assert_actual_stages_supported(
        wcs.validate_stage_range("local-downstream", "local-downstream")
    )


@pytest.mark.parametrize("from_stage,to_stage", [
    ("local-downstream", "some-future-stage"),
])
def test_an_unimplemented_stage_stays_locked(tmp_path, from_stage, to_stage):
    experiment_id = any_experiment()
    experiments = tmp_path / "experiments"
    out = tmp_path / "diagnostics"
    _seed_frozen_inputs(experiments, experiment_id)
    with pytest.raises(wcs.WindowClosureError, match="not enabled"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage=from_stage, to_stage=to_stage,
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )
    assert not out.exists(), "a locked stage created a directory"


def test_an_unknown_stage_creates_no_downstream_directory(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="not enabled"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="some-future-stage",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )
    for variant_id in _NONZERO:
        assert not wcs.local_downstream_root(experiment_id, variant_id, out).exists()


def test_local_downstream_is_dispatched_as_a_single_stage(tmp_path):
    calls: list = []
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical, calls=calls),
    )
    assert result["stages_run"] == ["local-downstream"]
    assert [call["variant_id"] for call in calls] == list(_NONZERO)


def test_the_stage_never_fits_the_fire_risk_model_or_bootstraps(tmp_path):
    """The production Step7C downscaling model IS trained; the fire-risk one is not.

    `model_fit=false` would have been a false statement about this chain, so
    the two are reported separately instead of collapsing them.
    """
    result, _, _, _ = _run_local_downstream(tmp_path)
    for payload in (result, result["local_downstream"]):
        assert payload["model_fit"] is True
        assert payload["downscaling_model_fit"] is True
        assert payload["downscaling_model_stage"] == "step7c"
        assert payload["fire_risk_model_fit"] is False
        assert payload["fire_risk_model_stage_run"] is False
        assert payload["bootstrap_run"] is False


def test_resume_and_force_together_are_refused(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="mutually exclusive"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            resume=True, force=True,
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )


# =============================================================================
# 6-17. Binding gates -- every one of them runs BEFORE the engine
# =============================================================================
def _expect_binding_failure(experiment_id: str, out: Path, experiments: Path,
                            match: str = "") -> None:
    with pytest.raises(wcs.WindowClosureError, match=match):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )
    for variant_id in _NONZERO:
        assert not wcs.local_downstream_root(experiment_id, variant_id, out).exists(), (
            "a failed binding created a downstream directory"
        )
        assert not wcs.local_downstream_metadata_path(
            experiment_id, variant_id, out
        ).exists()


def test_a_missing_preregistration_fails_before_the_engine(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    (out / experiment_id / "config" / "preregistration.json").unlink()
    _expect_binding_failure(experiment_id, out, experiments)


def test_an_analysis_id_mismatch_fails_before_the_engine(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="analysis_id|shift"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )
    assert not wcs.local_downstream_root(experiment_id, "close_7d_earlier", out).exists()


def test_missing_predictor_metadata_fails_before_the_engine(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    wcs.predictor_metadata_path(experiment_id, "close_14d_earlier", out).unlink()
    _expect_binding_failure(experiment_id, out, experiments, "metadata")


def _mutate_predictor_metadata(experiment_id, out, variant_id, mutate) -> None:
    path = wcs.predictor_metadata_path(experiment_id, variant_id, out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_a_non_pass_predictor_status_fails(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _mutate_predictor_metadata(
        experiment_id, out, "close_7d_earlier",
        lambda p: p.__setitem__("status", "fail"),
    )
    _expect_binding_failure(experiment_id, out, experiments, "status")


def test_a_wrong_predictor_artifact_count_fails(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _mutate_predictor_metadata(
        experiment_id, out, "close_7d_earlier",
        lambda p: p.__setitem__("produced_raster_count", 22),
    )
    _expect_binding_failure(experiment_id, out, experiments, "produced_raster_count")


def test_a_missing_predictor_raster_fails(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    metadata = json.loads(
        wcs.predictor_metadata_path(experiment_id, "close_7d_earlier", out)
        .read_text(encoding="utf-8")
    )
    Path(metadata["artifact_inventory"][0]["path"]).unlink()
    _expect_binding_failure(experiment_id, out, experiments, "missing")


def test_a_predictor_hash_mismatch_fails(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    metadata = json.loads(
        wcs.predictor_metadata_path(experiment_id, "close_7d_earlier", out)
        .read_text(encoding="utf-8")
    )
    target = Path(metadata["artifact_inventory"][0]["path"])
    target.write_bytes(target.read_bytes() + b"tampered")
    _expect_binding_failure(experiment_id, out, experiments, "hashes")


def test_a_duplicate_predictor_path_fails(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)

    def _duplicate(payload):
        inventory = payload["artifact_inventory"]
        inventory[1]["path"] = inventory[0]["path"]
        inventory[1]["sha256"] = inventory[0]["sha256"]
        payload["artifact_sha256"][inventory[1]["artifact_id"]] = inventory[0]["sha256"]

    _mutate_predictor_metadata(experiment_id, out, "close_7d_earlier", _duplicate)
    _expect_binding_failure(experiment_id, out, experiments, "duplicate")


def test_a_date_balanced_predictor_product_fails(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _mutate_predictor_metadata(
        experiment_id, out, "close_7d_earlier",
        lambda p: p["artifact_inventory"][0].__setitem__("product", "date_balanced_median"),
    )
    _expect_binding_failure(experiment_id, out, experiments, "reducer-counterfactual")


def test_the_prelabel_raster_may_not_be_a_predictor_by_path(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    prelabel = wcs.prelabel_raster_path(experiment_id, out)

    def _inject(payload):
        record = payload["artifact_inventory"][0]
        record["path"] = str(prelabel)
        record["sha256"] = wcs.sha256_file(prelabel)
        payload["artifact_sha256"][record["artifact_id"]] = record["sha256"]

    _mutate_predictor_metadata(experiment_id, out, "close_7d_earlier", _inject)
    _expect_binding_failure(experiment_id, out, experiments, "pre-label censoring raster")


def test_a_copy_of_the_prelabel_raster_may_not_be_a_predictor(tmp_path):
    """Even inside the variant namespace, the censoring raster is not a predictor."""
    import shutil

    experiment_id, out, experiments = _predictor_env(tmp_path)
    prelabel = wcs.prelabel_raster_path(experiment_id, out)
    metadata_path = wcs.predictor_metadata_path(experiment_id, "close_7d_earlier", out)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    record = payload["artifact_inventory"][0]
    shutil.copyfile(prelabel, Path(record["path"]))
    record["sha256"] = wcs.sha256_file(prelabel)
    payload["artifact_sha256"][record["artifact_id"]] = record["sha256"]
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _expect_binding_failure(experiment_id, out, experiments, "raster's hash")


def test_a_canonical_export_attempt_flag_fails(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _mutate_predictor_metadata(
        experiment_id, out, "close_7d_earlier",
        lambda p: p.__setitem__("canonical_export_attempted", True),
    )
    _expect_binding_failure(experiment_id, out, experiments, "canonical_export_attempted")


@pytest.mark.parametrize("flag", sorted(wcs.PREDICTOR_METADATA_REQUIRED_FLAGS))
def test_every_required_predictor_flag_is_enforced(tmp_path, flag):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    expected = wcs.PREDICTOR_METADATA_REQUIRED_FLAGS[flag]
    _mutate_predictor_metadata(
        experiment_id, out, "close_7d_earlier",
        lambda p: p.__setitem__(flag, not expected),
    )
    _expect_binding_failure(experiment_id, out, experiments, flag)


def test_binding_completes_before_any_production_helper_is_called(tmp_path):
    """A broken binding must reach neither the engine nor a materialised input."""
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _mutate_predictor_metadata(
        experiment_id, out, "close_7d_earlier",
        lambda p: p.__setitem__("status", "fail"),
    )
    _expect_binding_failure(experiment_id, out, experiments)
    for variant_id in _NONZERO:
        assert not wcs.local_downstream_input_root(experiment_id, variant_id, out).exists()


# =============================================================================
# 18-22. Canonical isolation
# =============================================================================
def test_the_canonical_variant_never_enters_the_work_queue(tmp_path):
    calls: list = []
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical, calls=calls),
    )
    assert wcs.CANONICAL_VARIANT_ID not in {call["variant_id"] for call in calls}
    assert result["processed_variants"] == list(_NONZERO)
    assert result["canonical_downstream_attempted"] is False


def test_no_canonical_downstream_directory_is_created(tmp_path):
    _, experiment_id, out, _ = _run_local_downstream(tmp_path)
    canonical_dir = out / experiment_id / "variants" / wcs.CANONICAL_VARIANT_ID
    assert _relative_files(canonical_dir) == ["frozen_reference.json"]
    assert not wcs.local_downstream_root(
        experiment_id, wcs.CANONICAL_VARIANT_ID, out,
    ).exists()


def test_the_canonical_experiment_namespace_is_untouched(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical_root = wcs.canonical_experiment_root(experiment_id, experiments)
    before = {p: p.read_bytes() for p in canonical_root.rglob("*") if p.is_file()}
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    after = {p: p.read_bytes() for p in canonical_root.rglob("*") if p.is_file()}
    assert after == before


def test_plan_prelabel_and_predictor_documents_are_untouched(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    root = out / experiment_id
    watched = sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
    )
    before = {name: (root / name).read_bytes() for name in watched}
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    for name, payload in before.items():
        assert (root / name).read_bytes() == payload, f"{name} was modified"


def test_the_frozen_canonical_step8a_hash_is_unchanged(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    dataset = wcs.canonical_step8a_path(experiment_id, experiments)
    before = wcs.sha256_file(dataset)
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    assert wcs.sha256_file(dataset) == before
    metadata = _metadata(out, experiment_id, "close_7d_earlier")
    assert metadata["canonical_step8a_sha256"] == before


# =============================================================================
# 23-32. Production reuse
# =============================================================================
def test_the_production_stage_sequence_is_the_canonical_one():
    assert wcs.PRODUCTION_STAGE_SEQUENCE == (
        "step5", "step5c", "step7a", "step7b", "step7c", "step7d", "step7e", "step8a",
    )
    assert sorted(wcs.PRODUCTION_STAGE_HELPERS) == sorted(wcs.PRODUCTION_STAGE_SEQUENCE)


@pytest.mark.parametrize("stage", list(wcs.PRODUCTION_STAGE_SEQUENCE))
def test_every_reused_production_helper_exists(stage):
    import importlib

    spec = wcs.PRODUCTION_STAGE_HELPERS[stage]
    module = importlib.import_module(spec["module"])
    assert callable(getattr(module, spec["function"]))


def test_the_engine_runs_the_stages_in_a_deterministic_order(tmp_path):
    result, experiment_id, out, _ = _run_local_downstream(tmp_path)
    for variant_id in _NONZERO:
        metadata = _metadata(out, experiment_id, variant_id)
        assert metadata["production_stage_sequence"] == list(wcs.PRODUCTION_STAGE_SEQUENCE)
        assert sorted(metadata["production_helpers"]) == sorted(wcs.PRODUCTION_STAGE_SEQUENCE)


def test_a_wrong_stage_sequence_is_refused(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    engine = _fake_downstream_engine(
        canonical, stages=("step5", "step5c", "step7a", "step7b", "step7c", "step7d", "step7e"),
    )
    with pytest.raises(wcs.WindowClosureError, match="deterministic sequence"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=engine,
        )


def test_the_variant_context_carries_the_materialised_production_inputs(tmp_path):
    calls: list = []
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical, calls=calls),
    )
    for call in calls:
        ctx = call["context"]
        variant_id = call["variant_id"]
        inputs = wcs.local_downstream_input_root(experiment_id, variant_id, out)
        assert Path(ctx["baseline_input_dir"]) == inputs / "landsat_timeseries"
        assert Path(ctx["current_period_dir"]) == inputs / "current_period"
        assert Path(ctx["ndvi_current_dir"]) == inputs / "ndvi_current_period"
        assert Path(ctx["modis_input_dir"]) == inputs / "modis"
        assert Path(ctx["dem_input_dir"]) == inputs / "dem"
        assert Path(ctx["gate_labels_dir"]) == inputs / "labels"
        assert ctx["step4_metadata_path"] is None
        assert ctx["is_kozan"] is False
        # ...and every one of them really exists before the chain runs.
        assert (inputs / "current_period").is_dir()
        assert (inputs / "labels" / "mcd64a1_raw.tif").is_file()


def test_the_variant_context_carries_the_variant_dates_and_the_frozen_label_window(tmp_path):
    calls: list = []
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical, calls=calls),
    )
    base = ctx_for(experiment_id)
    variants = {
        v["variant_id"]: v for v in wcs.build_window_variants(base, list(_SHIFTS))
    }
    for call in calls:
        variant = variants[call["variant_id"]]
        assert call["predictor_start_date"] == variant["predictor_start_date"]
        assert call["predictor_end_date"] == variant["predictor_end_date"]
        assert call["context"]["label_start_date"] == base["label_start_date"]
        assert call["context"]["label_end_date"] == base["label_end_date"]
        assert call["context"]["current_period_days"] == base["current_period_days"]


def test_the_global_experiment_registry_is_never_mutated(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    before = json.dumps(ctx_for(experiment_id), sort_keys=True, default=str)
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    assert json.dumps(ctx_for(experiment_id), sort_keys=True, default=str) == before


def _guarded_earth_engine_import():
    """Record every attempt to import Earth Engine without blocking it."""
    import builtins

    real_import = builtins.__import__
    touched: list[str] = []

    def guarded(name, *args, **kwargs):
        if name == "ee" or name.startswith("ee."):
            touched.append(name)
        return real_import(name, *args, **kwargs)

    return guarded, touched


def test_the_stage_imports_no_earth_engine_module(tmp_path):
    import builtins
    from unittest.mock import patch

    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    guarded, touched = _guarded_earth_engine_import()
    with patch.object(builtins, "__import__", side_effect=guarded):
        result = wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_fake_downstream_engine(canonical),
        )
    assert touched == [], f"the local-downstream stage imported Earth Engine: {touched}"
    assert result["local_downstream"]["gee_query_run"] is False
    assert result["local_downstream"]["gee_export_run"] is False
    assert result["gee_queries_run"] is False
    assert result["gee_exports_run"] is False


def test_no_new_formula_module_is_introduced():
    """Every downstream number must come from a production module."""
    modules = {spec["module"] for spec in wcs.PRODUCTION_STAGE_HELPERS.values()}
    assert modules == {
        "src.step5_preprocess_timeseries", "src.step5c_tvdi",
        "src.step7a_tiling_infrastructure", "src.step7b_prepare_downscaling_dataset",
        "src.step7c_train_downscaling_model", "src.step7d_predict_downscaled_lst",
        "src.step7e_fuse_landsat_downscaled_lst",
        "src.step8a_prepare_500m_modeling_dataset",
    }


# =============================================================================
# Production input materialisation
# =============================================================================
def test_production_input_names_are_the_production_ones(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    base = ctx_for(experiment_id)
    for variant_id in _NONZERO:
        inputs = wcs.local_downstream_input_root(experiment_id, variant_id, out)
        current = inputs / "current_period" / (
            f"landsat_current_period_{base['current_period_days']}days.tif"
        )
        assert current.is_file()
        assert (inputs / "ndvi_current_period" / "current_ndvi_median.tif").is_file()
        baselines = sorted((inputs / "landsat_timeseries").glob("*.tif"))
        assert len(baselines) == len(_baseline_years(experiment_id))
        for path in baselines:
            assert path.name.startswith(f"{base['landsat_file_prefix']}_baseline_")
        ndvi_baselines = sorted((inputs / "ndvi_timeseries").glob("*.tif"))
        assert len(ndvi_baselines) == len(_baseline_years(experiment_id))
        for path in ndvi_baselines:
            assert path.name.startswith("ndvi_baseline_")
        for name in wcs.MODIS_ROLE_FILENAMES.values():
            assert (inputs / "modis" / name).is_file()


def test_the_current_window_inputs_are_reassembled_as_two_band_rasters(tmp_path):
    import rasterio

    experiment_id, out, experiments = _predictor_env(tmp_path)
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    base = ctx_for(experiment_id)
    inputs = wcs.local_downstream_input_root(experiment_id, "close_7d_earlier", out)
    for path in (
        inputs / "current_period" / f"landsat_current_period_{base['current_period_days']}days.tif",
        inputs / "ndvi_current_period" / "current_ndvi_median.tif",
    ):
        with rasterio.open(path) as dataset:
            assert dataset.count == 2


def test_band_assembly_copies_values_verbatim(tmp_path):
    import rasterio

    median = _write_raster(tmp_path / "median.tif", np.full(_TEST_SHAPE, 21.5))
    count = _write_raster(tmp_path / "count.tif", np.full(_TEST_SHAPE, 3.0))
    target = tmp_path / "stacked.tif"
    wcs._stack_single_band_rasters([median, count], target)
    with rasterio.open(target) as stacked, rasterio.open(median) as a, \
            rasterio.open(count) as b:
        assert stacked.count == 2
        assert np.array_equal(stacked.read(1), a.read(1))
        assert np.array_equal(stacked.read(2), b.read(1))
        assert stacked.crs == a.crs
        assert stacked.transform == a.transform


def test_band_assembly_refuses_a_grid_mismatch(tmp_path):
    median = _write_raster(tmp_path / "median.tif")
    count = _write_raster(tmp_path / "count.tif", transform=_MODIS_TRANSFORM)
    with pytest.raises(wcs.WindowClosureError, match="transform"):
        wcs._stack_single_band_rasters([median, count], tmp_path / "stacked.tif")


def test_copied_inputs_hash_like_their_source(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    metadata = _metadata(out, experiment_id, "close_7d_earlier")
    copies = [
        record for record in metadata["materialized_inputs"]
        if record["mode"] == "copy" and record["materialized"]
    ]
    assert copies
    for record in copies:
        assert record["sha256"] == wcs.sha256_file(Path(record["sources"][0]))


def test_baseline_support_rasters_are_retained_not_consumed(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    metadata = _metadata(out, experiment_id, "close_7d_earlier")
    retained = [
        record for record in metadata["materialized_inputs"]
        if record["mode"] == "retained_not_consumed"
    ]
    assert len(retained) == 2 * len(_baseline_years(experiment_id))
    for record in retained:
        assert record["consumed_by_production"] is False
        assert record["target"] is None
        assert record["materialized"] is False


def test_the_baseline_file_dates_come_from_the_predictor_metadata(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    predictor = json.loads(
        wcs.predictor_metadata_path(experiment_id, "close_7d_earlier", out)
        .read_text(encoding="utf-8")
    )
    ends = {
        record["end_date"] for record in predictor["artifact_inventory"]
        if str(record["role"]).startswith("baseline_lst_")
    }
    inputs = wcs.local_downstream_input_root(experiment_id, "close_7d_earlier", out)
    names = {p.name for p in (inputs / "landsat_timeseries").glob("*.tif")}
    for end in ends:
        assert any(end in name for name in names), f"no baseline input for {end}"


# =============================================================================
# 33-37. Namespace
# =============================================================================
def test_every_mutable_output_lives_under_the_variant_downstream_tree(tmp_path):
    result, experiment_id, out, _ = _run_local_downstream(tmp_path)
    for variant_id in _NONZERO:
        downstream = wcs.local_downstream_root(experiment_id, variant_id, out)
        metadata = _metadata(out, experiment_id, variant_id)
        for record in metadata["artifact_inventory"]:
            assert downstream in Path(record["path"]).parents
        for record in metadata["materialized_inputs"]:
            if record["materialized"]:
                assert downstream in Path(record["target"]).parents


def test_nothing_is_written_into_the_predictor_data_tree(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    predictor_data = {
        p: p.read_bytes()
        for variant_id in _NONZERO
        for p in (out / experiment_id / "variants" / variant_id / "data").rglob("*")
        if p.is_file()
    }
    assert predictor_data
    _run_local_downstream(tmp_path, env=(experiment_id, out, experiments))
    for path, payload in predictor_data.items():
        assert path.read_bytes() == payload


def test_a_context_pointing_outside_the_downstream_tree_is_refused(tmp_path):
    experiment_id = any_experiment()
    base = ctx_for(experiment_id)
    variant = next(
        v for v in wcs.build_window_variants(base, [0, 7]) if not v["is_canonical"]
    )
    ctx = wcs.build_local_downstream_variant_context(
        experiment_id, variant, base, output_root=tmp_path / "out",
    )
    ctx["step8a_output_dir"] = tmp_path / "elsewhere" / "step8a"
    with pytest.raises(wcs.WindowClosureError, match="escapes"):
        wcs.assert_local_downstream_context_safe(
            ctx, experiment_id, variant["variant_id"], base, tmp_path / "out",
        )


@pytest.mark.parametrize("key", ["step5_output_dir", "step8a_output_dir", "data_root"])
def test_a_context_pointing_at_the_canonical_namespace_is_refused(tmp_path, key):
    experiment_id = any_experiment()
    base = ctx_for(experiment_id)
    variant = next(
        v for v in wcs.build_window_variants(base, [0, 7]) if not v["is_canonical"]
    )
    ctx = wcs.build_local_downstream_variant_context(
        experiment_id, variant, base, output_root=tmp_path / "out",
    )
    ctx[key] = Path(base["output_root"]) / "step5"
    with pytest.raises(wcs.WindowClosureError, match="canonical production namespace"):
        wcs.assert_local_downstream_context_safe(
            ctx, experiment_id, variant["variant_id"], base, tmp_path / "out",
        )


def test_a_context_pointing_at_the_predictor_data_tree_is_refused(tmp_path):
    experiment_id = any_experiment()
    base = ctx_for(experiment_id)
    variant = next(
        v for v in wcs.build_window_variants(base, [0, 7]) if not v["is_canonical"]
    )
    out = tmp_path / "out"
    ctx = wcs.build_local_downstream_variant_context(
        experiment_id, variant, base, output_root=out,
    )
    ctx["current_period_dir"] = (
        wcs.variant_root(experiment_id, variant["variant_id"], out) / "data" / "current_period"
    )
    with pytest.raises(wcs.WindowClosureError, match="predictor-export data namespace"):
        wcs.assert_local_downstream_context_safe(
            ctx, experiment_id, variant["variant_id"], base, out,
        )


def test_a_context_pointing_at_the_prelabel_namespace_is_refused(tmp_path):
    experiment_id = any_experiment()
    base = ctx_for(experiment_id)
    variant = next(
        v for v in wcs.build_window_variants(base, [0, 7]) if not v["is_canonical"]
    )
    out = tmp_path / "out"
    ctx = wcs.build_local_downstream_variant_context(
        experiment_id, variant, base, output_root=out,
    )
    ctx["gate_labels_dir"] = wcs.experiment_root(experiment_id, out) / "prelabel_censor"
    with pytest.raises(wcs.WindowClosureError, match="pre-label censoring namespace"):
        wcs.assert_local_downstream_context_safe(
            ctx, experiment_id, variant["variant_id"], base, out,
        )


def test_a_write_target_outside_the_stage_namespace_is_refused(tmp_path):
    experiment_id = any_experiment()
    with pytest.raises(wcs.WindowClosureError, match="not a local-downstream-owned target"):
        wcs.assert_local_downstream_owned_targets(
            experiment_id, "close_7d_earlier",
            [tmp_path / "somewhere" / "step8a.parquet"], tmp_path / "out",
        )


# =============================================================================
# 38-47. Step8A feature contract
# =============================================================================
def _lineage(tmp_path: Path) -> tuple[str, Path, dict]:
    experiment_id = any_experiment()
    experiments = tmp_path / "experiments"
    _seed_frozen_inputs(experiments, experiment_id)
    return experiment_id, experiments, wcs.step8a_predictor_lineage(experiment_id, experiments)


def test_the_lineage_is_derived_from_the_frozen_step8a_stats(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    assert lineage["timing_derived_predictors"] == sorted(TIMING_PREFIXES)
    assert lineage["static_predictors"] == sorted(STATIC_PREFIXES)


def test_an_unknown_predictor_source_refuses_to_guess(tmp_path):
    experiment_id, experiments, _ = _lineage(tmp_path)
    stats = wcs.canonical_step8a_stats_path(experiment_id, experiments)
    payload = json.loads(stats.read_text(encoding="utf-8"))
    root = wcs.canonical_experiment_root(experiment_id, experiments)
    payload["predictor_paths"]["mystery"] = str(root / "step99" / "mystery.tif")
    stats.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(wcs.WindowClosureError, match="unrecognised source directory"):
        wcs.step8a_predictor_lineage(experiment_id, experiments)


def test_a_valid_variant_dataset_passes_the_contract(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    outcome = wcs.assert_step8a_feature_contract(
        _variant_frame(canonical), canonical, lineage,
    )
    assert outcome["feature_contract_passed"] is True
    assert outcome["key_column"] == "cell_id"
    assert outcome["key_uniqueness_passed"] is True


def test_a_missing_required_column_fails(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical).drop(columns=["slope_mean"])
    with pytest.raises(wcs.WindowClosureError, match="missing canonical column"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


def test_an_extra_model_feature_fails(tmp_path):
    """An unknown feature column is refused BY NAME, with the reason spelled out.

    The refusal comes from the column classification rather than from a plain
    set difference: a column the production lineage cannot place is a signal
    that the Step8A schema moved, so the message must name the offending
    column AND say that the static/timing split has to be re-derived. The
    assertions below are structural on purpose -- they pin the semantics, not
    one particular sentence.
    """
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical)
    variant["extra_thermal_feature_mean"] = 1.0

    with pytest.raises(wcs.WindowClosureError) as exc_info:
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)

    message = str(exc_info.value)
    assert "extra_thermal_feature_mean" in message
    assert "could not be classified" in message
    assert "schema has changed" in message


def test_a_dtype_mismatch_fails(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical)
    variant["elevation_valid_count"] = variant["elevation_valid_count"].astype("float64")
    with pytest.raises(wcs.WindowClosureError, match="dtype contract broken"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


def test_an_empty_variant_dataset_fails(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    with pytest.raises(wcs.WindowClosureError, match="empty"):
        wcs.assert_step8a_feature_contract(canonical.iloc[0:0], canonical, lineage)


def test_a_duplicate_key_fails(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical)
    variant.loc[1, "cell_id"] = variant.loc[0, "cell_id"]
    with pytest.raises(wcs.WindowClosureError, match="duplicate key"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


def test_a_single_class_label_fails(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical)
    variant["burned"] = 0
    with pytest.raises(wcs.WindowClosureError, match="only class"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


def test_a_reordered_column_set_fails(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical)
    variant = variant[list(reversed(variant.columns))]
    with pytest.raises(wcs.WindowClosureError, match="column ORDER"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


def test_an_unclassifiable_column_refuses_to_guess(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="could not be classified"):
        wcs.classify_step8a_columns(
            list(_canonical_frame().columns) + ["brand_new_production_column"], lineage,
        )


def test_the_model_feature_registry_is_the_production_one(tmp_path):
    from src.step8b_train_baseline_vs_thermal_model import THERMAL_MODEL_FEATURES

    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    contract = wcs.step8a_feature_contract(canonical, lineage)
    assert contract["model_feature_columns_in_order"] == list(THERMAL_MODEL_FEATURES)


def test_the_primary_population_count_is_reported_and_not_filtered(tmp_path):
    _, experiment_id, out, _ = _run_local_downstream(tmp_path)
    metadata = _metadata(out, experiment_id, "close_7d_earlier")
    assert metadata["primary_population"] == wcs.PRIMARY_POPULATION
    assert metadata["primary_population_filter_applied"] is False
    assert isinstance(metadata["primary_population_row_count"], int)
    assert metadata["primary_population_row_count"] < metadata["variant_row_count"]


# =============================================================================
# Semantic dtype contract for discrete production codes
#
# Step8A writes `source_mask_majority` as `int(code)` or `np.nan`, so pandas
# infers float64 when a dataset happens to contain a null cell and int64 when
# it does not. Two scientifically identical columns can therefore differ in
# literal dtype for no reason other than observation support.
# =============================================================================
SOURCE_MASK_COLUMN = "source_mask_majority"


def _set_codes(frame: pd.DataFrame, values, dtype) -> pd.DataFrame:
    """Replace the source-mask column with a repeating code pattern."""
    out = frame.copy()
    sequence = [values[i % len(values)] for i in range(len(out))]
    out[SOURCE_MASK_COLUMN] = pd.Series(sequence, index=out.index, dtype=dtype)
    return out


def _code_case(tmp_path: Path, canonical_values, canonical_dtype,
               variant_values, variant_dtype):
    _, _, lineage = _lineage(tmp_path)
    canonical = _set_codes(_canonical_frame(), canonical_values, canonical_dtype)
    variant = _set_codes(
        _variant_frame(_canonical_frame()), variant_values, variant_dtype,
    )
    return canonical, variant, lineage


def test_the_discrete_code_domain_comes_from_production():
    """The codebook is read from Step8A/Step7E, never re-listed here."""
    from src.step8a_prepare_500m_modeling_dataset import (
        SOURCE_GAPFILL, SOURCE_INVALID, SOURCE_OBSERVED,
    )

    domains = wcs.step8a_discrete_code_domains()
    assert set(domains) == {SOURCE_MASK_COLUMN}
    assert domains[SOURCE_MASK_COLUMN] == tuple(sorted(
        {int(SOURCE_INVALID), int(SOURCE_OBSERVED), int(SOURCE_GAPFILL)}
    ))


# --- 1, 2. Matching representations need no exemption ------------------------
@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_matching_source_mask_dtypes_pass_without_an_exemption(tmp_path, dtype):
    canonical, variant, lineage = _code_case(tmp_path, [1, 2], dtype, [1, 2], dtype)
    outcome = wcs.assert_step8a_feature_contract(variant, canonical, lineage)
    assert outcome["feature_contract_passed"] is True
    assert outcome["literal_dtype_differences"] == []
    assert outcome["accepted_semantic_dtype_compatibilities"] == []


# --- 3, 4. Missingness-driven representation differences are compatible ------
def test_a_nullable_canonical_and_an_integer_variant_are_compatible(tmp_path):
    canonical, variant, lineage = _code_case(
        tmp_path, [1.0, 2.0, np.nan], "float64", [1, 2, 1], "int64",
    )
    outcome = wcs.assert_step8a_feature_contract(variant, canonical, lineage)
    assert outcome["feature_contract_passed"] is True
    assert [d["column"] for d in outcome["literal_dtype_differences"]] == \
        [SOURCE_MASK_COLUMN]
    accepted = outcome["accepted_semantic_dtype_compatibilities"]
    assert len(accepted) == 1
    record = accepted[0]
    assert record["column"] == SOURCE_MASK_COLUMN
    assert record["canonical_dtype"] == "float64"
    assert record["variant_dtype"] == "int64"
    assert record["semantic_type"] == "nullable_integer_categorical_code"
    assert record["compatibility"] == "pass"
    assert record["canonical_null_count"] > 0
    assert record["variant_null_count"] == 0
    assert record["nulls_preserved"] is True


def test_an_integer_canonical_and_a_nullable_variant_are_compatible(tmp_path):
    canonical, variant, lineage = _code_case(
        tmp_path, [1, 2], "int64", [1.0, 2.0, np.nan], "float64",
    )
    outcome = wcs.assert_step8a_feature_contract(variant, canonical, lineage)
    assert outcome["feature_contract_passed"] is True
    record = outcome["accepted_semantic_dtype_compatibilities"][0]
    assert record["canonical_dtype"] == "int64"
    assert record["variant_dtype"] == "float64"
    assert record["variant_null_count"] > 0


# --- 5, 6, 7, 8. The exemption never becomes a blanket relaxation ------------
def test_a_fractional_source_mask_code_fails(tmp_path):
    canonical, variant, lineage = _code_case(
        tmp_path, [1, 2], "int64", [1.0, 1.5], "float64",
    )
    with pytest.raises(wcs.WindowClosureError, match="fractional code"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_a_non_finite_source_mask_code_fails(tmp_path, bad):
    canonical, variant, lineage = _code_case(
        tmp_path, [1, 2], "int64", [1.0, bad], "float64",
    )
    with pytest.raises(wcs.WindowClosureError, match=r"inf"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


def test_a_string_source_mask_code_fails(tmp_path):
    canonical, variant, lineage = _code_case(
        tmp_path, [1.0, 2.0], "float64", ["1", "2"], "object",
    )
    with pytest.raises(wcs.WindowClosureError, match="non-numeric dtype"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


def test_a_boolean_source_mask_code_fails(tmp_path):
    canonical, variant, lineage = _code_case(
        tmp_path, [1.0, 0.0], "float64", [True, False], "bool",
    )
    with pytest.raises(wcs.WindowClosureError, match="boolean"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


def test_a_code_outside_the_production_codebook_fails(tmp_path):
    canonical, variant, lineage = _code_case(
        tmp_path, [1.0, 2.0], "float64", [1, 7], "int64",
    )
    with pytest.raises(wcs.WindowClosureError, match="outside the production"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


# --- 9. Nulls survive the check ---------------------------------------------
def test_the_check_preserves_nulls_and_values(tmp_path):
    canonical, variant, lineage = _code_case(
        tmp_path, [1.0, 2.0, np.nan], "float64", [1, 2, 1], "int64",
    )
    canonical_nulls = int(canonical[SOURCE_MASK_COLUMN].isna().sum())
    canonical_values = canonical[SOURCE_MASK_COLUMN].to_numpy(copy=True)
    variant_values = variant[SOURCE_MASK_COLUMN].to_numpy(copy=True)
    assert canonical_nulls > 0

    wcs.assert_step8a_feature_contract(variant, canonical, lineage)

    assert int(canonical[SOURCE_MASK_COLUMN].isna().sum()) == canonical_nulls
    assert np.array_equal(
        canonical[SOURCE_MASK_COLUMN].to_numpy(), canonical_values, equal_nan=True,
    )
    assert np.array_equal(variant[SOURCE_MASK_COLUMN].to_numpy(), variant_values)
    assert str(canonical[SOURCE_MASK_COLUMN].dtype) == "float64"
    assert str(variant[SOURCE_MASK_COLUMN].dtype) == "int64"


# --- 10. Other features are NOT excused --------------------------------------
@pytest.mark.parametrize("column", [
    "elevation_valid_count", "ndvi_valid_count", "valid_30m_pixel_count",
    "total_30m_pixel_count", "landcover_dominant",
])
def test_another_int_float_mismatch_is_not_excused(tmp_path, column):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical)
    variant[column] = variant[column].astype("float64")
    with pytest.raises(wcs.WindowClosureError, match="dtype contract broken"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


def test_a_reordered_column_set_still_fails_with_a_compatible_code(tmp_path):
    canonical, variant, lineage = _code_case(
        tmp_path, [1.0, 2.0, np.nan], "float64", [1, 2, 1], "int64",
    )
    variant = variant[list(reversed(variant.columns))]
    with pytest.raises(wcs.WindowClosureError, match="column ORDER"):
        wcs.assert_step8a_feature_contract(variant, canonical, lineage)


# --- 12. The check touches no file -------------------------------------------
def test_the_contract_check_modifies_no_parquet_or_csv(tmp_path):
    canonical, variant, lineage = _code_case(
        tmp_path, [1.0, 2.0, np.nan], "float64", [1, 2, 1], "int64",
    )
    canonical_path = tmp_path / "canonical.parquet"
    variant_path = tmp_path / "variant.parquet"
    canonical_csv = tmp_path / "canonical.csv"
    canonical.to_parquet(canonical_path, index=False)
    variant.to_parquet(variant_path, index=False)
    canonical.to_csv(canonical_csv, index=False)
    before = _namespace_snapshot(tmp_path)

    wcs.assert_step8a_feature_contract(
        pd.read_parquet(variant_path), pd.read_parquet(canonical_path), lineage,
    )
    assert _namespace_snapshot(tmp_path) == before


# --- 13, 14, 15. End to end through the stage --------------------------------
def _int_coded_variant(canonical: pd.DataFrame) -> pd.DataFrame:
    """A variant whose support left no null source-mask cell -> int64."""
    variant = _variant_frame(canonical)
    variant[SOURCE_MASK_COLUMN] = pd.Series(
        [1 if i % 2 else 2 for i in range(len(variant))],
        index=variant.index, dtype="int64",
    )
    return variant


def test_a_variant_with_an_integer_source_mask_completes(tmp_path):
    """The frozen canonical is NEVER rewritten; only the variant differs."""
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical_path = wcs.canonical_step8a_path(experiment_id, experiments)
    canonical_sha256_before = wcs.sha256_file(canonical_path)
    canonical = pd.read_parquet(canonical_path)
    assert str(canonical[SOURCE_MASK_COLUMN].dtype) == "float64"
    assert int(canonical[SOURCE_MASK_COLUMN].isna().sum()) > 0

    engine = _fake_downstream_engine(canonical, frames={
        variant_id: _int_coded_variant(canonical) for variant_id in _NONZERO
    })
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=engine,
    )
    assert result["completed_variants"] == list(_NONZERO)
    assert wcs.sha256_file(canonical_path) == canonical_sha256_before

    for variant_id in _NONZERO:
        metadata = _metadata(out, experiment_id, variant_id)
        assert metadata["status"] == "pass"
        assert metadata["feature_contract_passed"] is True
        assert [d["column"] for d in metadata["literal_dtype_differences"]] == \
            [SOURCE_MASK_COLUMN]
        accepted = metadata["accepted_semantic_dtype_compatibilities"]
        assert len(accepted) == 1
        assert accepted[0]["compatibility"] == "pass"
        assert accepted[0]["semantic_type"] == "nullable_integer_categorical_code"
        assert metadata["semantic_dtype_contract"]["exact_dtype_required_elsewhere"] is True
        assert metadata["semantic_dtype_contract"]["production_code_domains"][
            SOURCE_MASK_COLUMN
        ] == [0, 1, 2]
        # ...and the static/label invariance checks still ran.
        assert metadata["static_invariance_passed"] is True
        assert metadata["label_invariance_passed"] is True


def test_an_incompatible_code_writes_no_pass_metadata(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    broken = _variant_frame(canonical)
    broken[SOURCE_MASK_COLUMN] = pd.Series(
        [7] * len(broken), index=broken.index, dtype="int64",
    )
    engine = _fake_downstream_engine(canonical, frames={"close_7d_earlier": broken})
    with pytest.raises(wcs.WindowClosureError, match="outside the production"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=engine,
        )
    assert not wcs.local_downstream_metadata_path(
        experiment_id, "close_7d_earlier", out,
    ).exists()
    assert not wcs.local_downstream_metadata_path(
        experiment_id, "close_14d_earlier", out,
    ).exists()


# =============================================================================
# 48-58. Static / label invariance
# =============================================================================
def _invariance(canonical: pd.DataFrame, variant: pd.DataFrame, lineage: dict) -> dict:
    contract = wcs.assert_step8a_feature_contract(variant, canonical, lineage)
    return wcs.compare_step8a_invariance(
        variant, canonical, contract, contract["key_column"],
    )


def test_matching_static_and_label_columns_pass(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    outcome = _invariance(canonical, _variant_frame(canonical), lineage)
    assert outcome["static_invariance_passed"] is True
    assert outcome["label_invariance_passed"] is True


@pytest.mark.parametrize("column,value", [
    ("burned", 0),
    ("burn_month", 9),
    ("elevation_mean", 999.0),
    ("slope_mean", 999.0),
    ("landcover_dominant", 40),
    ("lon", 99.0),
    ("lat", 99.0),
    ("row_500m", 999),
    ("col_500m", 999),
    ("burnable_tree_shrub_grass", False),
    ("total_30m_pixel_count", 1),
])
def test_an_invariant_column_mismatch_fails(tmp_path, column, value):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical)
    variant.loc[0, column] = value
    with pytest.raises(wcs.WindowClosureError, match="Static/label invariance failed"):
        _invariance(canonical, variant, lineage)


def test_timing_derived_changes_are_allowed(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical)
    for prefix in TIMING_PREFIXES:
        variant[f"{prefix}_mean"] = variant[f"{prefix}_mean"] + 42.0
    variant["valid_for_modeling"] = False
    variant["thermal_any_missing"] = True
    outcome = _invariance(canonical, variant, lineage)
    assert outcome["static_invariance_passed"] is True


def test_a_different_row_count_is_not_a_failure(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    outcome = _invariance(canonical, _variant_frame(canonical, drop_rows=5), lineage)
    assert outcome["variant_row_count"] == len(canonical) - 5
    assert outcome["canonical_row_count"] == len(canonical)
    assert outcome["row_count_difference_is_not_a_failure"] is True


def test_overlap_and_only_counts_are_correct(tmp_path):
    _, _, lineage = _lineage(tmp_path)
    canonical = _canonical_frame()
    variant = _variant_frame(canonical, drop_rows=3)
    variant.loc[0, "cell_id"] = "r999_c999"
    variant.loc[0, "row_500m"] = 999
    variant.loc[0, "col_500m"] = 999
    variant.loc[0, "lon"] = 40.0
    variant.loc[0, "lat"] = 40.0
    outcome = _invariance(canonical, variant, lineage)
    assert outcome["variant_row_count"] == len(canonical) - 3
    assert outcome["canonical_row_count"] == len(canonical)
    assert outcome["overlap_row_count"] == len(canonical) - 4
    assert outcome["variant_only_row_count"] == 1
    assert outcome["canonical_only_row_count"] == 4


def test_no_common_cohort_is_created(tmp_path):
    result, experiment_id, out, _ = _run_local_downstream(tmp_path)
    assert result["common_cohort_created"] is False
    for variant_id in _NONZERO:
        metadata = _metadata(out, experiment_id, variant_id)
        assert metadata["common_cohort_created"] is False
        names = [
            record["role"].lower() for record in metadata["artifact_inventory"]
        ]
        assert not [name for name in names if "cohort" in name or "fold" in name]


def test_the_key_column_is_the_production_one(tmp_path):
    canonical = _canonical_frame()
    assert wcs.step8a_key_column(canonical) == "cell_id"
    assert wcs.step8a_key_column(canonical.drop(columns=["cell_id"])) == "row_500m__col_500m"


def test_a_reference_grid_mismatch_fails(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    drifted = dict(_canonical_reference_grid(), width=_TEST_SHAPE[1] + 1)
    with pytest.raises(wcs.WindowClosureError, match="reference grid differs"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_fake_downstream_engine(
                canonical, stats_override=drifted,
            ),
        )


# =============================================================================
# 59-68. Artefacts and metadata
# =============================================================================
def test_the_artifact_inventory_is_complete_and_hashed(tmp_path):
    _, experiment_id, out, _ = _run_local_downstream(tmp_path)
    for variant_id in _NONZERO:
        metadata = _metadata(out, experiment_id, variant_id)
        assert metadata["artifact_count"] == len(metadata["artifact_inventory"])
        assert metadata["artifact_count"] > 0
        for record in metadata["artifact_inventory"]:
            for field in ("artifact_id", "stage", "role", "path", "sha256", "bytes",
                          "media_type", "producer", "input_roles", "variant_derived",
                          "status"):
                assert record[field] is not None, field
            assert wcs.sha256_file(Path(record["path"])) == record["sha256"]
            assert metadata["artifact_sha256"][record["artifact_id"]] == record["sha256"]


def test_raster_artifacts_carry_the_raster_contract(tmp_path):
    _, experiment_id, out, _ = _run_local_downstream(tmp_path)
    metadata = _metadata(out, experiment_id, "close_7d_earlier")
    rasters = [r for r in metadata["artifact_inventory"] if r["media_type"] == "image/tiff"]
    assert rasters
    for record in rasters:
        for field in ("band_count", "dtype", "nodata", "width", "height", "crs",
                      "transform", "grid_signature", "finite_cell_count",
                      "min_finite", "max_finite"):
            assert field in record, field


def test_the_parquet_artifact_carries_the_dataset_contract(tmp_path):
    _, experiment_id, out, _ = _run_local_downstream(tmp_path)
    metadata = _metadata(out, experiment_id, "close_7d_earlier")
    parquet = [
        r for r in metadata["artifact_inventory"]
        if r["media_type"] == "application/vnd.apache.parquet"
    ]
    assert len(parquet) == 1
    record = parquet[0]
    for field in ("row_count", "column_count", "columns", "dtypes", "key_column",
                  "duplicate_key_count", "primary_population_row_count",
                  "burned_count", "unburned_count"):
        assert field in record, field
    assert record["duplicate_key_count"] == 0
    assert record["key_column"] == "cell_id"


def test_json_artifacts_carry_a_deterministic_hash(tmp_path):
    _, experiment_id, out, _ = _run_local_downstream(tmp_path)
    metadata = _metadata(out, experiment_id, "close_7d_earlier")
    documents = [
        r for r in metadata["artifact_inventory"] if r["media_type"] == "application/json"
    ]
    assert documents
    for record in documents:
        assert record["deterministic_sha256"]
        assert "schema_version" in record


def test_the_metadata_schema_and_required_fields(tmp_path):
    _, experiment_id, out, _ = _run_local_downstream(tmp_path)
    required = (
        "schema_version", "analysis_id", "experiment_id", "variant_id",
        "shift_days", "predictor_start_date", "predictor_end_date", "lead_days",
        "baseline_years", "predictor_metadata_path", "predictor_metadata_sha256",
        "predictor_artifact_count", "predictor_artifact_sha256",
        "production_stage_sequence", "production_helpers", "production_policy",
        "variant_context_summary", "artifact_inventory", "artifact_count",
        "artifact_sha256", "step8a_dataset_path", "step8a_dataset_sha256",
        "step8a_stats_path", "step8a_stats_sha256", "canonical_step8a_path",
        "canonical_step8a_sha256", "canonical_feature_contract_sha256",
        "feature_contract_passed", "static_invariance_passed",
        "label_invariance_passed", "key_uniqueness_passed",
        "semantic_dtype_contract", "literal_dtype_differences",
        "accepted_semantic_dtype_compatibilities", "variant_row_count",
        "canonical_row_count", "overlap_row_count", "variant_only_row_count",
        "canonical_only_row_count", "primary_population_row_count",
        "burned_count", "unburned_count", "prelabel_used_as_predictor",
        "prelabel_positive_cell_count", "common_cohort_created",
        "all_paths_inside_variant_namespace", "canonical_downstream_attempted",
        "canonical_outputs_modified", "gee_queries_run", "gee_exports_run",
        "model_fit", "downscaling_model_fit", "fire_risk_model_fit",
        "fire_risk_model_stage_run", "bootstrap_run",
        "baseline_binding_source", "baseline_directory_scan_used",
        "baseline_lst_binding", "frozen_input_sha256_before",
        "frozen_input_sha256_after", "frozen_hashes_unchanged", "status",
    )
    for variant_id in _NONZERO:
        metadata = _metadata(out, experiment_id, variant_id)
        assert metadata["schema_version"] == "window_closure_local_downstream.v1"
        for field in required:
            assert field in metadata, f"{variant_id} metadata is missing {field}"
        assert metadata["status"] == "pass"
        assert metadata["model_fit"] is True
        assert metadata["downscaling_model_fit"] is True
        assert metadata["fire_risk_model_fit"] is False
        assert metadata["fire_risk_model_stage_run"] is False
        assert metadata["bootstrap_run"] is False
        assert metadata["baseline_binding_source"] == "predictor_export_metadata"
        assert metadata["baseline_directory_scan_used"] is False
        assert metadata["prelabel_used_as_predictor"] is False
        assert metadata["prelabel_positive_cell_count"] == 0
        assert metadata["common_cohort_created"] is False
        assert metadata["canonical_downstream_attempted"] is False
        assert metadata["canonical_outputs_modified"] is False
        assert metadata["frozen_hashes_unchanged"] is True


def test_the_metadata_pins_the_predictor_and_canonical_hashes(tmp_path):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    for variant_id in _NONZERO:
        metadata = _metadata(out, experiment_id, variant_id)
        predictor_path = wcs.predictor_metadata_path(experiment_id, variant_id, out)
        assert metadata["predictor_metadata_sha256"] == wcs.sha256_file(predictor_path)
        predictor = json.loads(predictor_path.read_text(encoding="utf-8"))
        assert metadata["predictor_artifact_sha256"] == predictor["artifact_sha256"]
        assert metadata["canonical_step8a_sha256"] == wcs.sha256_file(
            wcs.canonical_step8a_path(experiment_id, experiments)
        )


def test_the_metadata_is_deterministically_sorted(tmp_path):
    _, experiment_id, out, _ = _run_local_downstream(tmp_path)
    text = wcs.local_downstream_metadata_path(
        experiment_id, "close_7d_earlier", out,
    ).read_text(encoding="utf-8")
    assert text == json.dumps(
        json.loads(text), indent=2, sort_keys=True, ensure_ascii=False, default=str,
    ) + "\n"


def test_no_temporary_metadata_file_is_left_behind(tmp_path):
    _, experiment_id, out, _ = _run_local_downstream(tmp_path)
    leftovers = [
        p for p in (out / experiment_id).rglob("*")
        if p.is_file() and (p.name.startswith(".") or p.suffix == ".tmp")
    ]
    assert leftovers == []


def test_a_failing_variant_writes_no_pass_metadata(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    engine = _fake_downstream_engine(canonical, fail_variants=("close_7d_earlier",))
    with pytest.raises(wcs.WindowClosureError, match="synthetic engine failure"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=engine,
        )
    assert not wcs.local_downstream_metadata_path(
        experiment_id, "close_7d_earlier", out,
    ).exists()


def test_a_variant_that_fails_the_contract_writes_no_pass_metadata(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    broken = _variant_frame(canonical)
    broken.loc[0, "elevation_mean"] = 12345.0
    engine = _fake_downstream_engine(canonical, frames={"close_7d_earlier": broken})
    with pytest.raises(wcs.WindowClosureError, match="Static/label invariance failed"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=engine,
        )
    assert not wcs.local_downstream_metadata_path(
        experiment_id, "close_7d_earlier", out,
    ).exists()


def test_a_missing_step8a_dataset_fails(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    engine = _fake_downstream_engine(canonical, skip_step8a_dataset=True)
    with pytest.raises(wcs.WindowClosureError, match="no Step8A modelling dataset"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=engine,
        )


# =============================================================================
# 69-75. Resume / force / failure isolation
# =============================================================================
def test_a_plain_rerun_refuses_to_overwrite(tmp_path):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="Refusing to overwrite"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )


def test_resume_reuses_a_complete_variant_without_running_the_chain(tmp_path):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    before = {
        p: p.read_bytes()
        for variant_id in _NONZERO
        for p in wcs.local_downstream_root(experiment_id, variant_id, out).rglob("*")
        if p.is_file()
    }
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        resume=True, output_root=out, experiments_root=experiments,
        local_downstream_engine=_exploding_engine,
    )
    assert result["reused_variants"] == list(_NONZERO)
    assert result["completed_variants"] == list(_NONZERO)
    for path, payload in before.items():
        assert path.read_bytes() == payload


def _namespace_snapshot(*roots: Path) -> dict[str, str]:
    """path -> sha256 for every file under the given roots."""
    snapshot: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                snapshot[str(path)] = wcs.sha256_file(path)
    return snapshot


def _assert_resume_is_fail_closed(experiment_id, out, experiments, *, match):
    """--resume must refuse and leave the namespace bit-for-bit unchanged.

    It may not quarantine, move, delete, rebuild or write anything, and it may
    not reach the production engine.
    """
    canonical_root = wcs.canonical_experiment_root(experiment_id, experiments)
    before = _namespace_snapshot(out, canonical_root)

    with pytest.raises(wcs.WindowClosureError, match=match):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            resume=True, output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )

    assert _namespace_snapshot(out, canonical_root) == before, (
        "a fail-closed --resume changed the namespace"
    )
    for variant_id in _NONZERO:
        quarantine = (
            out / experiment_id / "variants" / variant_id
            / wcs.LOCAL_DOWNSTREAM_QUARANTINE_DIR
        )
        assert not quarantine.exists(), "--resume quarantined an output"


def test_resume_refuses_a_changed_predictor_metadata_hash(tmp_path):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    path = wcs.predictor_metadata_path(experiment_id, "close_7d_earlier", out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reducer_note_for_test"] = "changed"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _assert_resume_is_fail_closed(
        experiment_id, out, experiments,
        match="cannot reuse variant 'close_7d_earlier'",
    )


def test_resume_refuses_a_missing_artifact(tmp_path):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    metadata = _metadata(out, experiment_id, "close_14d_earlier")
    Path(metadata["artifact_inventory"][0]["path"]).unlink()

    _assert_resume_is_fail_closed(
        experiment_id, out, experiments,
        match="cannot reuse variant 'close_14d_earlier'",
    )


def test_resume_refuses_a_broken_step8a_contract(tmp_path):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    dataset = wcs.variant_step8a_dataset_path(experiment_id, "close_7d_earlier", out)
    frame = pd.read_parquet(dataset)
    frame = frame.drop(columns=["slope_mean"])
    frame.to_parquet(dataset, index=False)

    _assert_resume_is_fail_closed(
        experiment_id, out, experiments,
        match="cannot reuse variant 'close_7d_earlier'",
    )


def test_resume_refuses_a_failed_metadata_status(tmp_path):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    path = wcs.local_downstream_metadata_path(experiment_id, "close_7d_earlier", out)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["status"] = "fail"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    _assert_resume_is_fail_closed(
        experiment_id, out, experiments, match="previous status is 'fail'",
    )


def test_resume_refuses_a_missing_metadata(tmp_path):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    wcs.local_downstream_metadata_path(experiment_id, "close_14d_earlier", out).unlink()

    _assert_resume_is_fail_closed(
        experiment_id, out, experiments, match="no local-downstream metadata",
    )


def test_resume_never_rebuilds_after_reusing_an_earlier_variant(tmp_path):
    """A valid first variant is reused; a broken second one stops the run."""
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    dataset = wcs.variant_step8a_dataset_path(experiment_id, "close_14d_earlier", out)
    dataset.write_bytes(b"corrupted")

    _assert_resume_is_fail_closed(
        experiment_id, out, experiments,
        match="cannot reuse variant 'close_14d_earlier'",
    )
    # ...and the first, still-valid variant was left exactly as it was.
    assert _metadata(out, experiment_id, "close_7d_earlier")["status"] == "pass"


def test_force_quarantines_only_stage_owned_files(tmp_path):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    predictor_before = {
        p: p.read_bytes()
        for variant_id in _NONZERO
        for p in (out / experiment_id / "variants" / variant_id / "data").rglob("*")
        if p.is_file()
    }
    prelabel = wcs.prelabel_raster_path(experiment_id, out)
    prelabel_before = prelabel.read_bytes()
    canonical_before = {
        p: p.read_bytes()
        for p in wcs.canonical_experiment_root(experiment_id, experiments).rglob("*")
        if p.is_file()
    }
    old_dataset = wcs.variant_step8a_dataset_path(
        experiment_id, "close_7d_earlier", out,
    ).read_bytes()

    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        force=True, output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical),
    )
    assert result["quarantined_artifacts"], "force deleted instead of quarantining"
    quarantine = (
        out / experiment_id / "variants" / "close_7d_earlier"
        / wcs.LOCAL_DOWNSTREAM_QUARANTINE_DIR / wcs.LOCAL_DOWNSTREAM_QUARANTINE_KIND
    )
    kept = [
        p for p in quarantine.rglob(wcs.STEP8A_DATASET_NAME) if p.is_file()
    ]
    assert kept and kept[0].read_bytes() == old_dataset

    for path, payload in predictor_before.items():
        assert path.read_bytes() == payload
    assert prelabel.read_bytes() == prelabel_before
    for path, payload in canonical_before.items():
        assert path.read_bytes() == payload


def test_a_second_variant_failure_preserves_the_first(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    engine = _fake_downstream_engine(canonical, fail_variants=("close_14d_earlier",))
    with pytest.raises(wcs.WindowClosureError, match="synthetic engine failure"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=engine,
        )
    first = _metadata(out, experiment_id, "close_7d_earlier")
    assert first["status"] == "pass"
    assert wcs.variant_step8a_dataset_path(experiment_id, "close_7d_earlier", out).is_file()
    assert not wcs.local_downstream_metadata_path(
        experiment_id, "close_14d_earlier", out,
    ).exists()
    for variant_id in _NONZERO:
        assert wcs.predictor_metadata_path(experiment_id, variant_id, out).is_file()


def test_variants_are_processed_in_increasing_shift_order(tmp_path):
    calls: list = []
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[14, 0, 7], dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical, calls=calls),
    )
    assert [call["variant_id"] for call in calls] == list(_NONZERO)


# =============================================================================
# 76-83. Dry run
# =============================================================================
def _dry_run(tmp_path: Path, env=None, shifts=_SHIFTS) -> tuple[dict, str, Path, Path]:
    experiment_id, out, experiments = env or _predictor_env(tmp_path, shifts)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(shifts), dry_run=True,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=_exploding_engine,
    )
    return result, experiment_id, out, experiments


def test_the_dry_run_writes_nothing(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    before = _relative_files(out / experiment_id)
    result, _, _, _ = _dry_run(tmp_path, env=(experiment_id, out, experiments))
    assert result["files_written"] is False
    assert _relative_files(out / experiment_id) == before
    for variant_id in _NONZERO:
        assert not wcs.local_downstream_root(experiment_id, variant_id, out).exists()


def test_the_dry_run_reports_the_local_downstream_contract(tmp_path):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    assert result["ran"] is False
    assert result["dry_run"] is True
    assert result["planned_stages"] == ["local-downstream"]
    summary = result["local_downstream_summary"]
    assert summary["canonical_processing_enabled"] is False
    assert summary["canonical_frozen_reference_only"] is True
    assert summary["nonzero_variant_ids"] == list(_NONZERO)
    assert summary["production_stage_sequence"] == list(wcs.PRODUCTION_STAGE_SEQUENCE)
    assert summary["predictor_binding_ready"] is True
    assert summary["all_predictor_artifacts_present"] is True
    assert summary["all_predictor_hashes_match"] is True
    assert summary["all_paths_inside_dedicated_namespace"] is True
    assert summary["common_cohort_created"] is False
    for flag in ("gee_queries_run", "gee_exports_run", "model_fit", "bootstrap_run"):
        assert summary[flag] is False
        assert result[flag] is False
    # A dry run fits nothing but must declare what an actual run would fit.
    assert summary["downscaling_model_fit_planned"] is True
    assert summary["downscaling_model_fit"] is False
    assert summary["fire_risk_model_fit"] is False
    assert summary["fire_risk_model_stage_run"] is False


def test_the_dry_run_plans_both_nonzero_variants(tmp_path):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    plans = result["local_downstream_summary"]["variant_plans"]
    base = ctx_for(experiment_id)
    variants = {
        v["variant_id"]: v for v in wcs.build_window_variants(base, list(_SHIFTS))
    }
    for variant_id in _NONZERO:
        plan = plans[variant_id]
        assert plan["export_enabled"] is True
        assert plan["predictor_artifact_count"] == wcs.expected_raster_count(
            _baseline_years(experiment_id)
        )
        assert plan["shift_days"] == variants[variant_id]["shift_days"]
        assert plan["predictor_start_date"] == variants[variant_id]["predictor_start_date"]
        assert plan["predictor_end_date"] == variants[variant_id]["predictor_end_date"]
        assert plan["lead_days"] == variants[variant_id]["lead_days"]
        assert plan["static_invariance_check_planned"] is True
        assert plan["label_invariance_check_planned"] is True
        assert plan["downscaling_model_fit_planned"] is True
        assert plan["fire_risk_model_fit"] is False
        assert plan["baseline_binding_source"] == "predictor_export_metadata"
        assert plan["baseline_directory_scan_used"] is False
        assert sorted(plan["planned_stage_outputs"]) == sorted(wcs.PRODUCTION_STAGE_SEQUENCE)
        assert Path(plan["planned_step8a_path"]) == wcs.variant_step8a_dataset_path(
            experiment_id, variant_id, out
        )


def test_the_dry_run_disables_the_canonical_variant(tmp_path):
    result, _, _, _ = _dry_run(tmp_path)
    plan = result["local_downstream_summary"]["variant_plans"][wcs.CANONICAL_VARIANT_ID]
    assert plan["export_enabled"] is False
    assert plan["frozen_reference_only"] is True
    assert plan["planned_output_count"] == 0


def test_the_dry_run_never_imports_earth_engine(tmp_path):
    import builtins
    from unittest.mock import patch

    experiment_id, out, experiments = _predictor_env(tmp_path)
    guarded, touched = _guarded_earth_engine_import()
    with patch.object(builtins, "__import__", side_effect=guarded):
        result = wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=True,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )
    assert touched == [], f"the dry run imported Earth Engine: {touched}"
    assert result["planned_stages"] == ["local-downstream"]


def test_the_dry_run_reports_predictor_binding_gaps_without_raising(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    wcs.predictor_metadata_path(experiment_id, "close_7d_earlier", out).unlink()
    result, _, _, _ = _dry_run(tmp_path, env=(experiment_id, out, experiments))
    summary = result["local_downstream_summary"]
    assert summary["predictor_binding_ready"] is False
    assert summary["variant_plans"]["close_7d_earlier"]["predictor_binding_ready"] is False


# =============================================================================
# Result contract
# =============================================================================
def test_the_result_contract(tmp_path):
    result, experiment_id, out, _ = _run_local_downstream(tmp_path)
    assert result["ran"] is True
    assert result["dry_run"] is False
    assert result["experiment_id"] == experiment_id
    assert len(result["analysis_id"]) == 64
    assert result["stages_run"] == ["local-downstream"]
    assert result["processed_variants"] == list(_NONZERO)
    assert result["reused_variants"] == []
    assert result["completed_variants"] == list(_NONZERO)
    assert result["files_written"]
    assert result["downstream_artifacts_produced"] > 0
    assert result["step8a_datasets_produced"] == 2
    assert result["gee_queries_run"] is False
    assert result["gee_exports_run"] is False
    assert result["model_fit"] is True
    assert result["downscaling_model_fit"] is True
    assert result["fire_risk_model_fit"] is False
    assert result["fire_risk_model_stage_run"] is False
    assert result["bootstrap_run"] is False
    assert result["canonical_downstream_attempted"] is False
    assert result["common_cohort_created"] is False
    assert result["frozen_hashes_unchanged"] is True
    assert result["canonical_outputs_modified"] is False
    assert result["status"] == "pass"


# =============================================================================
# 84-96. Validator
# =============================================================================
def _write_log(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "INFO noise before\n"
        + json.dumps(payload, indent=2, default=str)
        + "\nINFO noise after\n",
        encoding="utf-8",
    )
    return path


def _validate(mode: str, experiment_id: str, out: Path, *,
              log: Optional[Path] = None, experiments: Optional[Path] = None,
              shifts=_SHIFTS) -> int:
    argv = [
        "--experiment", experiment_id, "--mode", mode,
        "--shifts", *[str(s) for s in shifts],
        "--output-root", str(out),
    ]
    if log is not None:
        argv += ["--log", str(log)]
    if experiments is not None:
        argv += ["--experiments-root", str(experiments)]
    return validator.main(argv)


def test_validator_reports_the_stage_lock(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "l.log", result))
    captured = capsys.readouterr().out
    assert "[PASS] local-downstream is an implemented actual stage" in captured
    assert "[PASS] no unimplemented stage is reachable" in captured


def test_validator_accepts_a_valid_dry_run_log(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    code = _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result))
    output = capsys.readouterr().out
    assert code == 0, output
    assert "OVERALL STATUS: PASS" in output


def test_validator_rejects_a_dry_run_that_wrote_files(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    result["files_written"] = True
    assert _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result)) == 1
    assert "[FAIL] no dry-run file writes detected" in capsys.readouterr().out


def test_validator_rejects_a_canonical_enabled_dry_run(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    result["local_downstream_summary"]["canonical_processing_enabled"] = True
    assert _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result)) == 1
    assert "[FAIL] canonical variant is frozen-reference-only" in capsys.readouterr().out


def test_validator_rejects_a_missing_variant_in_the_dry_run(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    summary = result["local_downstream_summary"]
    summary["nonzero_variant_ids"] = ["close_7d_earlier"]
    summary["variant_plans"].pop("close_14d_earlier")
    assert _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result)) == 1
    assert "[FAIL] every preregistered non-zero variant is planned" in capsys.readouterr().out


def test_validator_rejects_a_step8a_path_outside_the_namespace(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    plan = result["local_downstream_summary"]["variant_plans"]["close_7d_earlier"]
    plan["planned_step8a_path"] = str(tmp_path / "elsewhere" / "step8a.parquet")
    assert _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result)) == 1
    output = capsys.readouterr().out
    assert "[FAIL] close_7d_earlier planned Step8A path" in output


def test_validator_rejects_a_common_cohort_in_the_dry_run(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    result["local_downstream_summary"]["common_cohort_created"] = True
    assert _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result)) == 1
    assert "[FAIL] no common cohort is planned in this stage" in capsys.readouterr().out


def test_validator_rejects_a_missing_log(tmp_path, capsys):
    experiment_id, out, _ = _predictor_env(tmp_path)
    assert _validate("dry-run", experiment_id, out, log=tmp_path / "absent.log") == 1
    assert "[FAIL] dry-run log exists" in capsys.readouterr().out


# --- Actual mode --------------------------------------------------------------
def test_validator_accepts_a_valid_actual_run(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    code = _validate("actual", experiment_id, out, experiments=experiments)
    output = capsys.readouterr().out
    assert code == 0, output
    assert "OVERALL STATUS: PASS" in output


def test_validator_actual_rejects_missing_metadata(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    wcs.local_downstream_metadata_path(experiment_id, "close_14d_earlier", out).unlink()
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] close_14d_earlier local-downstream metadata exists" in capsys.readouterr().out


def test_validator_actual_rejects_an_artifact_hash_mismatch(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    metadata = _metadata(out, experiment_id, "close_7d_earlier")
    target = Path(metadata["artifact_inventory"][0]["path"])
    target.write_bytes(target.read_bytes() + b"tampered")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "every artefact hash matches the metadata" in capsys.readouterr().out


def test_validator_actual_rejects_a_step8a_feature_mismatch(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    dataset = wcs.variant_step8a_dataset_path(experiment_id, "close_7d_earlier", out)
    frame = pd.read_parquet(dataset)
    frame["unexpected_feature_mean"] = 1.0
    frame.to_parquet(dataset, index=False)
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "re-verify against the frozen canonical dataset" in capsys.readouterr().out


def test_validator_actual_rejects_a_static_or_label_mismatch(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    dataset = wcs.variant_step8a_dataset_path(experiment_id, "close_7d_earlier", out)
    frame = pd.read_parquet(dataset)
    frame.loc[0, "burned"] = 1 - int(frame.loc[0, "burned"])
    frame.to_parquet(dataset, index=False)
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "re-verify against the frozen canonical dataset" in capsys.readouterr().out


def test_validator_actual_rejects_a_canonical_downstream_file(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    stray = wcs.local_downstream_root(
        experiment_id, wcs.CANONICAL_VARIANT_ID, out,
    ) / "step8a" / wcs.STEP8A_DATASET_NAME
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"stray")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    output = capsys.readouterr().out
    assert "[FAIL] no downstream file exists under the canonical variant" in output


def test_validator_actual_rejects_the_prelabel_raster_as_an_artifact(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    prelabel = wcs.prelabel_raster_path(experiment_id, out)
    path = wcs.local_downstream_metadata_path(experiment_id, "close_7d_earlier", out)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    record = dict(metadata["artifact_inventory"][0])
    record["artifact_id"] = "step5/prelabel_burndate.tif"
    record["path"] = str(prelabel)
    record["sha256"] = wcs.sha256_file(prelabel)
    metadata["artifact_inventory"].append(record)
    metadata["artifact_count"] += 1
    metadata["artifact_sha256"][record["artifact_id"]] = record["sha256"]
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "does not carry the pre-label raster as an artefact" in capsys.readouterr().out


def test_validator_actual_rejects_a_common_cohort_artifact(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    stray = (
        wcs.local_downstream_root(experiment_id, "close_7d_earlier", out)
        / "step8a" / "common_cohort.parquet"
    )
    stray.write_bytes(b"stray")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "no compare/bootstrap/common-cohort artefact exists" in capsys.readouterr().out


def test_validator_actual_rejects_a_fire_risk_model_artifact(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    stray = (
        wcs.local_downstream_root(experiment_id, "close_7d_earlier", out)
        / "step8a" / "random_forest.joblib"
    )
    stray.write_bytes(b"stray")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "no fire-risk model artefact exists" in capsys.readouterr().out


def test_validator_actual_allows_the_production_downscaling_model(tmp_path, capsys):
    """Step7C persists a production downscaling model; that is not a violation."""
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    for variant_id in _NONZERO:
        model = (
            wcs.local_downstream_root(experiment_id, variant_id, out)
            / "step7c" / "downscaling_model.joblib"
        )
        model.write_bytes(b"synthetic-downscaling-model")
    code = _validate("actual", experiment_id, out, experiments=experiments)
    output = capsys.readouterr().out
    assert code == 0, output
    assert "pre-existing Step7C downscaling model is allowed and unchanged" in output


def test_validator_actual_rejects_a_drifted_predictor_raster(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    predictor = json.loads(
        wcs.predictor_metadata_path(experiment_id, "close_7d_earlier", out)
        .read_text(encoding="utf-8")
    )
    target = Path(predictor["artifact_inventory"][0]["path"])
    target.write_bytes(target.read_bytes() + b"drift")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "every bound predictor raster is unchanged" in capsys.readouterr().out


def test_validator_actual_accepts_a_recorded_semantic_dtype_compatibility(tmp_path, capsys):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    engine = _fake_downstream_engine(canonical, frames={
        variant_id: _int_coded_variant(canonical) for variant_id in _NONZERO
    })
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=engine,
    )
    code = _validate("actual", experiment_id, out, experiments=experiments)
    output = capsys.readouterr().out
    assert code == 0, output
    assert "records the semantic dtype contract" in output
    assert "every literal dtype difference is explicitly accepted" in output
    assert "accepted dtype compatibilities are production codes only" in output


def test_validator_actual_rejects_an_unexplained_dtype_difference(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    path = wcs.local_downstream_metadata_path(experiment_id, "close_7d_earlier", out)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["literal_dtype_differences"] = [{
        "column": "current_lst_mean",
        "variant_dtype": "int64",
        "canonical_dtype": "float64",
    }]
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] close_7d_earlier every literal dtype difference is explicitly accepted" \
        in capsys.readouterr().out


def test_validator_actual_rejects_an_undeclared_accepted_compatibility(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    path = wcs.local_downstream_metadata_path(experiment_id, "close_7d_earlier", out)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "column": "current_lst_mean",
        "canonical_dtype": "float64",
        "variant_dtype": "int64",
        "semantic_type": "nullable_integer_categorical_code",
        "production_code_domain": [0, 1, 2],
        "canonical_codes_present": [1],
        "variant_codes_present": [1],
        "nulls_preserved": True,
        "compatibility": "pass",
    }
    metadata["literal_dtype_differences"] = [{
        "column": "current_lst_mean", "variant_dtype": "int64",
        "canonical_dtype": "float64",
    }]
    metadata["accepted_semantic_dtype_compatibilities"] = [record]
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "accepted dtype compatibilities stay inside the declared" \
        in capsys.readouterr().out


def test_validator_actual_rejects_a_missing_semantic_dtype_record(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    path = wcs.local_downstream_metadata_path(experiment_id, "close_14d_earlier", out)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.pop("semantic_dtype_contract")
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] close_14d_earlier records the semantic dtype contract" \
        in capsys.readouterr().out


def test_validator_actual_rejects_a_bad_analysis_id(tmp_path, capsys):
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    path = wcs.local_downstream_metadata_path(experiment_id, "close_7d_earlier", out)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["analysis_id"] = "0" * 64
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "metadata analysis_id matches preregistration" in capsys.readouterr().out


# =============================================================================
# Regression: dry-run state snapshot vs. a pre-existing partial downstream
# =============================================================================
def _seed_partial_downstream(experiment_id: str, out: Path, *,
                             variant_id: str = "close_7d_earlier",
                             extra: Optional[dict] = None) -> dict[str, bytes]:
    """Leftovers a previous, FAILED actual run would have left behind."""
    downstream = wcs.local_downstream_root(experiment_id, variant_id, out)
    written: dict[str, bytes] = {}
    seeds = {
        "step5/current_period_median_celsius.tif": b"partial-step5",
        "step7c/downscaling_model.joblib": b"production-downscaling-model",
        "step7c/downscaling_model_metrics.json": b"{}",
    }
    seeds.update(extra or {})
    for relative, payload in seeds.items():
        path = downstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        written[str(path)] = payload
    return written


def _dry_run_on(experiment_id: str, out: Path, experiments: Path) -> dict:
    return wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=True,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=_exploding_engine,
    )


# --- 1. Clean tree ------------------------------------------------------------
def test_a_clean_tree_dry_run_snapshot_is_empty_and_unchanged(tmp_path, capsys):
    result, experiment_id, out, experiments = _dry_run(tmp_path)
    assert result["preexisting_stage_owned_paths"] == []
    assert result["stage_owned_snapshot_unchanged"] is True
    assert result["stage_owned_snapshot_before_sha256"] == \
        result["stage_owned_snapshot_after_sha256"]
    assert result["dry_run_created_paths"] == []
    assert result["dry_run_modified_paths"] == []
    assert result["dry_run_deleted_paths"] == []

    code = _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "c.log", result))
    output = capsys.readouterr().out
    assert code == 0, output
    assert "pre-existing partial downstream state was unchanged" in output


# --- 2, 3. A pre-existing partial tree is legitimate --------------------------
def test_a_preexisting_partial_downstream_passes_the_dry_run(tmp_path, capsys):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    seeded = _seed_partial_downstream(experiment_id, out)

    result = _dry_run_on(experiment_id, out, experiments)
    assert result["preexisting_stage_owned_paths"], "the partial tree was not snapshotted"
    assert result["stage_owned_snapshot_unchanged"] is True
    assert result["dry_run_created_paths"] == []
    assert result["dry_run_modified_paths"] == []
    assert result["dry_run_deleted_paths"] == []
    assert result["files_written"] is False

    code = _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "p.log", result))
    output = capsys.readouterr().out
    assert code == 0, output
    assert "pre-existing partial downstream state was unchanged" in output
    assert "pre-existing Step7C downscaling model is allowed and unchanged" in output
    assert "no fire-risk model artefact exists" in output
    assert "no compare/bootstrap/common-cohort artefact exists" in output

    for path, payload in seeded.items():
        assert Path(path).read_bytes() == payload, "the dry run touched the partial tree"


def test_a_preexisting_downscaling_model_is_recorded_in_the_snapshot(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _seed_partial_downstream(experiment_id, out)
    result = _dry_run_on(experiment_id, out, experiments)
    recorded = result["stage_owned_snapshot_after"]["files"]
    model = (
        "variants/close_7d_earlier/downstream/step7c/downscaling_model.joblib"
    )
    assert model in recorded
    assert recorded[model]["sha256"] == wcs.sha256_bytes(b"production-downscaling-model")
    assert model in result["preexisting_stage_owned_paths"]


# --- 4, 5, 6. Forbidden classes still fail -----------------------------------
@pytest.mark.parametrize("relative,expected", [
    ("step8b/baseline_vs_thermal_model.joblib", "no fire-risk model artefact exists"),
    ("step8a/common_cohort.parquet", "no compare/bootstrap/common-cohort artefact exists"),
    ("step8a/bootstrap_replicates.parquet", "no compare/bootstrap/common-cohort artefact exists"),
    ("step8a/shared_fold_assignments.parquet", "no compare/bootstrap/common-cohort artefact exists"),
])
def test_a_preexisting_forbidden_artifact_fails_the_dry_run(
    tmp_path, capsys, relative, expected,
):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _seed_partial_downstream(experiment_id, out, extra={relative: b"forbidden"})
    result = _dry_run_on(experiment_id, out, experiments)
    assert _validate(
        "dry-run", experiment_id, out, log=_write_log(tmp_path / "f.log", result),
    ) == 1
    assert f"[FAIL] dry run: {expected}" in capsys.readouterr().out


# --- 7, 8, 9, 10, 12. A dry run that touched anything fails -------------------
@pytest.mark.parametrize("field", [
    "dry_run_created_paths", "dry_run_modified_paths", "dry_run_deleted_paths",
])
def test_a_dry_run_that_touched_a_path_fails(tmp_path, capsys, field):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    result[field] = ["variants/close_7d_earlier/downstream/step5/new.tif"]
    result["stage_owned_snapshot_unchanged"] = False
    assert _validate(
        "dry-run", experiment_id, out, log=_write_log(tmp_path / "t.log", result),
    ) == 1
    assert "[FAIL] the dry run created, modified and deleted no stage-owned path" \
        in capsys.readouterr().out


def test_a_snapshot_digest_mismatch_fails(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    result["stage_owned_snapshot_after_sha256"] = "0" * 64
    assert _validate(
        "dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result),
    ) == 1
    assert "[FAIL] pre-existing partial downstream state was unchanged" \
        in capsys.readouterr().out


def test_a_missing_snapshot_fails(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    result.pop("stage_owned_snapshot_before")
    assert _validate(
        "dry-run", experiment_id, out, log=_write_log(tmp_path / "m.log", result),
    ) == 1
    assert "[FAIL] the dry run recorded a stage-owned state snapshot" \
        in capsys.readouterr().out


def test_a_dry_run_that_published_metadata_fails(tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run(tmp_path)
    result["dry_run_created_paths"] = [
        f"variants/close_7d_earlier/{wcs.LOCAL_DOWNSTREAM_METADATA_NAME}"
    ]
    result["stage_owned_snapshot_unchanged"] = False
    assert _validate(
        "dry-run", experiment_id, out, log=_write_log(tmp_path / "pm.log", result),
    ) == 1
    assert "[FAIL] the dry run published no local-downstream metadata" \
        in capsys.readouterr().out


def test_a_stage_owned_file_that_drifted_after_the_dry_run_fails(tmp_path, capsys):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _seed_partial_downstream(experiment_id, out)
    result = _dry_run_on(experiment_id, out, experiments)
    drifted = (
        wcs.local_downstream_root(experiment_id, "close_7d_earlier", out)
        / "step5" / "current_period_median_celsius.tif"
    )
    drifted.write_bytes(b"changed-after-the-dry-run")
    assert _validate(
        "dry-run", experiment_id, out, log=_write_log(tmp_path / "s.log", result),
    ) == 1
    assert "[FAIL] every recorded stage-owned file still hashes as the dry run saw it" \
        in capsys.readouterr().out


# --- 11. A non-pass metadata is not, on its own, a dry-run failure -----------
def test_a_preexisting_non_pass_metadata_does_not_fail_the_dry_run(tmp_path, capsys):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _seed_partial_downstream(experiment_id, out)
    metadata_path = wcs.local_downstream_metadata_path(
        experiment_id, "close_7d_earlier", out,
    )
    metadata_path.write_text(
        json.dumps({
            "schema_version": wcs.LOCAL_DOWNSTREAM_METADATA_SCHEMA,
            "status": "superseded_run_in_progress",
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result = _dry_run_on(experiment_id, out, experiments)
    code = _validate(
        "dry-run", experiment_id, out, log=_write_log(tmp_path / "np.log", result),
    )
    output = capsys.readouterr().out
    assert code == 0, output
    assert f"variants/close_7d_earlier/{wcs.LOCAL_DOWNSTREAM_METADATA_NAME}" in \
        result["preexisting_stage_owned_paths"]


# --- 13. The snapshot itself writes nothing ----------------------------------
def test_the_snapshot_never_creates_or_writes_anything(tmp_path, monkeypatch):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _seed_partial_downstream(experiment_id, out)
    base = ctx_for(experiment_id)
    variants = wcs.build_window_variants(base, list(_SHIFTS))

    def _blocked(name):
        def _fail(*_args, **_kwargs):
            raise AssertionError(f"the snapshot called {name}")
        return _fail

    monkeypatch.setattr(Path, "mkdir", _blocked("Path.mkdir"))
    monkeypatch.setattr(Path, "write_text", _blocked("Path.write_text"))
    monkeypatch.setattr(Path, "write_bytes", _blocked("Path.write_bytes"))
    monkeypatch.setattr(Path, "touch", _blocked("Path.touch"))
    monkeypatch.setattr(Path, "unlink", _blocked("Path.unlink"))
    monkeypatch.setattr(Path, "rename", _blocked("Path.rename"))
    monkeypatch.setattr(Path, "replace", _blocked("Path.replace"))
    monkeypatch.setattr(os, "replace", _blocked("os.replace"))

    snapshot = wcs.snapshot_local_downstream_state(experiment_id, variants, out)
    assert snapshot["file_count"] > 0
    assert snapshot["digest"]


def test_the_snapshot_does_not_change_mtimes(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    seeded = _seed_partial_downstream(experiment_id, out)
    before = {path: Path(path).stat().st_mtime_ns for path in seeded}

    base = ctx_for(experiment_id)
    wcs.snapshot_local_downstream_state(
        experiment_id, wcs.build_window_variants(base, list(_SHIFTS)), out,
    )
    for path, mtime in before.items():
        assert Path(path).stat().st_mtime_ns == mtime


# --- 14. Frozen inputs are untouched by the dry run --------------------------
def test_the_dry_run_leaves_frozen_inputs_untouched(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    _seed_partial_downstream(experiment_id, out)
    canonical_root = wcs.canonical_experiment_root(experiment_id, experiments)
    watched = [
        out / experiment_id / "config",
        out / experiment_id / "prelabel_censor",
        *[out / experiment_id / "variants" / v / "data" for v in _NONZERO],
        *[
            out / experiment_id / "variants" / v / wcs.PREDICTOR_METADATA_NAME
            for v in _NONZERO
        ],
        canonical_root,
    ]
    before = _namespace_snapshot(*watched)
    _dry_run_on(experiment_id, out, experiments)
    assert _namespace_snapshot(*watched) == before


# --- 15. Actual / resume / force semantics are unchanged ---------------------
def test_a_dry_run_does_not_change_actual_resume_or_force_semantics(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    seeded = _seed_partial_downstream(experiment_id, out)
    _dry_run_on(experiment_id, out, experiments)

    # plain actual still refuses the partial without overwriting it
    with pytest.raises(wcs.WindowClosureError, match="NOT reusable"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )
    # --resume stays fail-closed and touches nothing
    _assert_resume_is_fail_closed(
        experiment_id, out, experiments,
        match="cannot reuse variant 'close_7d_earlier'",
    )
    for path, payload in seeded.items():
        assert Path(path).read_bytes() == payload

    # --force may still quarantine and rebuild
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        force=True, output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical),
    )
    assert result["completed_variants"] == list(_NONZERO)
    quarantine = (
        out / experiment_id / "variants" / "close_7d_earlier"
        / wcs.LOCAL_DOWNSTREAM_QUARANTINE_DIR / wcs.LOCAL_DOWNSTREAM_QUARANTINE_KIND
    )
    kept = [
        p for p in quarantine.rglob("downscaling_model.joblib")
        if p.read_bytes() == b"production-downscaling-model"
    ]
    assert kept, "the pre-existing downscaling model was not quarantined"


def test_validator_never_runs_a_stage(tmp_path):
    """The validator only reads; it may not create anything."""
    _, experiment_id, out, experiments = _run_local_downstream(tmp_path)
    before = _relative_files(out / experiment_id)
    _validate("actual", experiment_id, out, experiments=experiments)
    assert _relative_files(out / experiment_id) == before


def _executable_string_literals(module_path: Path) -> list[str]:
    """Every string literal that is real code, i.e. not a docstring."""
    import ast

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


# =============================================================================
# Regression: Step8A variant-aware date validation (opt-in, fail-fast)
# =============================================================================
import src.step5_preprocess_timeseries as step5  # noqa: E402
import src.step8a_prepare_500m_modeling_dataset as step8a  # noqa: E402


def _pinned_experiment() -> str:
    """A registry experiment whose canonical dates Step8A pins by hand."""
    for experiment_id in REGISTRY_IDS:
        if experiment_id in step8a._EXPECTED_EXPERIMENT_DATES:
            return experiment_id
    pytest.skip("no registry experiment has hand-verified Step8A dates")


def _variant_ctx(tmp_path: Path, variant_id: str, env=None):
    """A real local-downstream context for one preregistered variant."""
    experiment_id, out, experiments = env or _predictor_env(tmp_path)
    base = ctx_for(experiment_id)
    analysis_id = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=True,
        output_root=out, experiments_root=experiments,
    )["analysis_id"]
    variant = next(
        v for v in wcs.build_window_variants(base, list(_SHIFTS))
        if v["variant_id"] == variant_id
    )
    ctx = wcs.build_local_downstream_variant_context(
        experiment_id, variant, base, analysis_id, output_root=out,
    )
    return ctx, experiment_id, out, experiments, variant


# --- 1, 2. Canonical behaviour is untouched ----------------------------------
def test_canonical_context_still_accepts_the_canonical_dates():
    experiment_id = _pinned_experiment()
    ctx = ctx_for(experiment_id)
    assert step8a.window_closure_variant_mode(ctx) is False
    assert step8a.resolve_expected_experiment_dates(ctx) == \
        step8a._EXPECTED_EXPERIMENT_DATES[experiment_id]
    step8a.assert_step8a_preflight(ctx, Path("/nonexistent/canonical/step8a"))


def test_canonical_context_still_rejects_shifted_dates(tmp_path):
    experiment_id = _pinned_experiment()
    ctx = dict(ctx_for(experiment_id))
    ctx["predictor_start_date"] = "1999-01-01"
    out_dir = tmp_path / "canonical_step8a"
    with pytest.raises(step8a.Step8AError, match="beklenen degerlerle eslesmiyor"):
        step8a.assert_step8a_preflight(ctx, out_dir)
    assert not out_dir.exists(), "a rejected canonical run created its output dir"


# --- 3, 4. A preregistered variant is accepted -------------------------------
@pytest.mark.parametrize("variant_id", list(_NONZERO))
def test_a_preregistered_variant_context_accepts_its_shifted_dates(tmp_path, variant_id):
    ctx, experiment_id, out, _, variant = _variant_ctx(tmp_path, variant_id)
    assert step8a.window_closure_variant_mode(ctx) is True

    expected = step8a.resolve_expected_experiment_dates(ctx)
    assert expected["predictor_start_date"] == variant["predictor_start_date"]
    assert expected["predictor_end_date"] == variant["predictor_end_date"]
    assert expected["predictor_start_date"] != ctx_for(experiment_id)["predictor_start_date"]
    assert expected["label_start_date"] == ctx_for(experiment_id)["label_start_date"]

    step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


# --- 5, 6, 7. The opt-in never becomes a bypass ------------------------------
def test_an_arbitrary_shifted_window_is_refused(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx["predictor_start_date"] = "2020-01-01"
    ctx["expected_predictor_start_date"] = "2020-01-01"
    with pytest.raises(step8a.Step8AError, match="bildirilen"):
        step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


def test_a_context_date_that_leaves_the_preregistration_is_refused(tmp_path):
    """Declared expectation stays preregistered, but the run window drifts."""
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx["predictor_end_date"] = "2020-12-31"
    with pytest.raises(step8a.Step8AError, match="calisma baglami"):
        step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


def test_an_analysis_id_mismatch_is_refused_by_step8a(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx["analysis_id"] = "0" * 64
    with pytest.raises(step8a.Step8AError, match="analysis_id uyusmuyor"):
        step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


def test_a_non_preregistered_variant_id_is_refused(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx["variant_id"] = "close_21d_earlier"
    with pytest.raises(step8a.Step8AError, match="preregistration'da"):
        step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


def test_the_canonical_variant_cannot_use_the_opt_in_path(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx["variant_id"] = wcs.CANONICAL_VARIANT_ID
    ctx["shift_days"] = 0
    with pytest.raises(step8a.Step8AError, match="canonical varyant"):
        step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


def test_a_changed_label_window_is_refused(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx["label_end_date"] = "2021-09-30"
    with pytest.raises(step8a.Step8AError, match="calisma baglami|label penceresi"):
        step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


def test_a_shift_days_mismatch_is_refused(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx["shift_days"] = 14
    with pytest.raises(step8a.Step8AError, match="shift_days uyusmuyor"):
        step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


def test_variant_mode_without_the_full_block_is_refused(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx.pop("analysis_id")
    with pytest.raises(step8a.Step8AError, match="eksik"):
        step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


# --- 8, 9. The guard runs before ANY artefact is written ---------------------
def test_a_variant_date_mismatch_writes_no_csv_parquet_or_stats(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx["predictor_end_date"] = "2020-12-31"
    out_dir = tmp_path / "step8a_out"

    with pytest.raises(step8a.Step8AError):
        step8a.main(output_dir_arg=str(out_dir), ctx=ctx, force=True)

    assert not out_dir.exists(), (
        "Step8A created its output directory before the date guard ran"
    )


def test_a_step8a_context_failure_leaves_no_pass_metadata(tmp_path):
    ctx, experiment_id, out, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx["analysis_id"] = "0" * 64
    out_dir = Path(ctx["step8a_output_dir"])

    with pytest.raises(step8a.Step8AError):
        step8a.main(output_dir_arg=str(out_dir), ctx=ctx, force=True)

    assert not out_dir.exists()
    assert not wcs.local_downstream_metadata_path(
        experiment_id, "close_7d_earlier", out,
    ).exists()


# --- 10. Output containment --------------------------------------------------
def test_a_step8a_output_path_outside_the_namespace_is_refused(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    stray = tmp_path / "elsewhere" / "step8a"
    with pytest.raises(step8a.Step8AError, match="ayrilmis namespace"):
        step8a.assert_step8a_preflight(ctx, stray)
    assert not stray.exists()


def test_variant_mode_requires_a_declared_allowed_output_root(tmp_path):
    ctx, _, _, _, _ = _variant_ctx(tmp_path, "close_7d_earlier")
    ctx.pop("window_closure_allowed_output_root")
    with pytest.raises(step8a.Step8AError, match="allowed_output_root"):
        step8a.assert_step8a_preflight(ctx, Path(ctx["step8a_output_dir"]))


# =============================================================================
# Regression: explicit Step5 baseline binding (no directory scan)
# =============================================================================
def _bindings_for(tmp_path: Path, variant_id: str = "close_7d_earlier"):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    base = ctx_for(experiment_id)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=True,
        output_root=out, experiments_root=experiments,
    )
    analysis_id = result["analysis_id"]
    canonical = wcs.canonical_window(base)
    variants = wcs.build_window_variants(base, list(_SHIFTS))
    variant = next(v for v in variants if v["variant_id"] == variant_id)
    inventory = wcs.frozen_input_inventory(experiment_id, experiments, base)
    metadata = wcs.read_predictor_metadata(experiment_id, variant_id, out)
    predictor_binding = wcs.assert_predictor_metadata_contract(
        experiment_id, analysis_id, variant, metadata,
        canonical["baseline_years"], out,
    )
    bindings = wcs.production_input_bindings(
        experiment_id, variant, base, predictor_binding["artifacts"], inventory, out,
    )
    return bindings, canonical["baseline_years"], experiment_id, out, experiments, variant


def test_the_baseline_list_comes_only_from_the_predictor_metadata(tmp_path):
    bindings, years, experiment_id, out, _, _ = _bindings_for(tmp_path)
    binding = wcs.resolve_baseline_lst_binding("close_7d_earlier", bindings, years)

    assert binding["baseline_binding_source"] == "predictor_export_metadata"
    assert binding["baseline_directory_scan_used"] is False
    assert binding["baseline_years"] == sorted(int(y) for y in years)
    assert len(binding["paths"]) == len(years)

    predictor = json.loads(
        wcs.predictor_metadata_path(experiment_id, "close_7d_earlier", out)
        .read_text(encoding="utf-8")
    )
    recorded = predictor["artifact_sha256"]
    for record in binding["records"]:
        assert record["product"] == "scene_weighted_median"
        assert recorded[record["source_artifact_id"]] == record["source_sha256"]


def test_count_products_never_enter_the_baseline_list(tmp_path):
    bindings, years, _, _, _, _ = _bindings_for(tmp_path)
    binding = wcs.resolve_baseline_lst_binding("close_7d_earlier", bindings, years)
    names = [Path(path).name for path in binding["paths"]]
    assert not [name for name in names if "valid_count" in name]
    assert all(
        record["product"] == "scene_weighted_median" for record in binding["records"]
    )


def test_a_missing_baseline_role_fails_before_any_production_helper(tmp_path):
    bindings, years, _, _, _, _ = _bindings_for(tmp_path)
    dropped = f"baseline_lst_{sorted(int(y) for y in years)[0]}"
    reduced = [b for b in bindings if b["input_role"] != dropped]
    with pytest.raises(wcs.WindowClosureError, match="no baseline LST binding"):
        wcs.resolve_baseline_lst_binding("close_7d_earlier", reduced, years)


def test_a_baseline_hash_mismatch_never_reaches_the_engine(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    predictor = json.loads(
        wcs.predictor_metadata_path(experiment_id, "close_7d_earlier", out)
        .read_text(encoding="utf-8")
    )
    baseline = next(
        record for record in predictor["artifact_inventory"]
        if str(record["role"]).startswith("baseline_lst_")
        and record["product"] == "scene_weighted_median"
    )
    target = Path(baseline["path"])
    target.write_bytes(target.read_bytes() + b"drifted")

    with pytest.raises(wcs.WindowClosureError, match="hashes"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )
    assert not wcs.local_downstream_root(experiment_id, "close_7d_earlier", out).exists()


def test_the_variant_context_pins_the_explicit_baseline_paths(tmp_path):
    calls: list = []
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical, calls=calls),
    )
    for call in calls:
        ctx = call["context"]
        pinned = ctx[step5.EXPLICIT_BASELINE_PATHS_KEY]
        assert len(pinned) == len(_baseline_years(experiment_id))
        assert all(Path(path).is_file() for path in pinned)
        assert ctx["baseline_directory_scan_used"] is False
        assert ctx["baseline_binding_source"] == "predictor_export_metadata"


def test_step5_never_scans_the_directory_when_the_list_is_explicit(tmp_path, monkeypatch):
    """The opt-in short-circuits BOTH the Step4 lookup and the directory glob."""
    explicit = [
        _write_raster(tmp_path / "baseline" / f"pinned_{year}.tif")
        for year in (2017, 2018)
    ]

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("the Step4 metadata lookup must not run")

    monkeypatch.setattr(step5, "list_baseline_tifs_from_step4_metadata", _must_not_run)

    ctx = {
        "baseline_input_dir": tmp_path / "does_not_exist",
        "baseline_start_date": "2017-01-01",
        "baseline_end_date": "2020-12-31",
        "landsat_file_prefix": "unused_prefix",
        step5.EXPLICIT_BASELINE_PATHS_KEY: explicit,
    }
    assert step5.list_baseline_tifs(ctx) == explicit


def test_step5_directory_scan_still_runs_without_the_opt_in(tmp_path, monkeypatch):
    """Production behaviour is untouched when the opt-in key is absent."""
    baseline_dir = tmp_path / "baseline"
    _write_raster(baseline_dir / "pfx_2017-07-20.tif")
    monkeypatch.setattr(
        step5, "list_baseline_tifs_from_step4_metadata", lambda ctx=None: [],
    )
    ctx = {
        "baseline_input_dir": baseline_dir,
        "baseline_start_date": "2017-01-01",
        "baseline_end_date": "2020-12-31",
        "landsat_file_prefix": "pfx",
        "step4_metadata_path": None,
    }
    assert [p.name for p in step5.list_baseline_tifs(ctx)] == ["pfx_2017-07-20.tif"]


def test_step5_refuses_a_missing_explicit_baseline_raster(tmp_path):
    ctx = {step5.EXPLICIT_BASELINE_PATHS_KEY: [tmp_path / "absent.tif"]}
    with pytest.raises(FileNotFoundError, match="Explicit baseline"):
        step5.list_baseline_tifs(ctx)


def test_step5_refuses_a_qa_raster_in_the_explicit_list(tmp_path):
    qa = _write_raster(tmp_path / "scene_2017-07-20_qa.tif")
    ctx = {step5.EXPLICIT_BASELINE_PATHS_KEY: [qa]}
    with pytest.raises(ValueError, match="QA raster"):
        step5.list_baseline_tifs(ctx)


# =============================================================================
# Regression: partial-output recovery, locks and isolation
# =============================================================================
def test_a_partial_downstream_is_not_overwritten_without_force(tmp_path):
    """A crashed run leaves a downstream tree with no pass metadata."""
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    partial = wcs.local_downstream_root(experiment_id, "close_7d_earlier", out) / "step5"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "current_period_median_celsius.tif").write_bytes(b"partial")

    with pytest.raises(wcs.WindowClosureError, match="NOT reusable"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_fake_downstream_engine(canonical),
        )
    assert (partial / "current_period_median_celsius.tif").read_bytes() == b"partial"


def test_resume_refuses_a_partial_downstream_without_touching_it(tmp_path):
    """--resume is fail-closed: it never quarantines or rebuilds a partial."""
    experiment_id, out, experiments = _predictor_env(tmp_path)
    partial = wcs.local_downstream_root(experiment_id, "close_7d_earlier", out) / "step5"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "current_period_median_celsius.tif").write_bytes(b"partial")

    _assert_resume_is_fail_closed(
        experiment_id, out, experiments,
        match="cannot reuse variant 'close_7d_earlier'",
    )
    assert (partial / "current_period_median_celsius.tif").read_bytes() == b"partial"


def test_resume_and_force_stay_mutually_exclusive_for_a_partial(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    partial = wcs.local_downstream_root(experiment_id, "close_7d_earlier", out) / "step5"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "current_period_median_celsius.tif").write_bytes(b"partial")

    with pytest.raises(wcs.WindowClosureError, match="mutually exclusive"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            resume=True, force=True,
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )
    assert (partial / "current_period_median_celsius.tif").read_bytes() == b"partial"


def test_force_recovers_a_partial_downstream_by_quarantining_it(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    partial = wcs.local_downstream_root(experiment_id, "close_7d_earlier", out) / "step8a"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / wcs.STEP8A_DATASET_NAME).write_bytes(b"half-written")

    watched: dict[Path, bytes] = {}
    for variant_id in _NONZERO:
        variant_dir = out / experiment_id / "variants" / variant_id
        watched[variant_dir / wcs.PREDICTOR_METADATA_NAME] = (
            (variant_dir / wcs.PREDICTOR_METADATA_NAME).read_bytes()
        )
        for path in (variant_dir / "data").rglob("*"):
            if path.is_file():
                watched[path] = path.read_bytes()
    for path in (out / experiment_id / "prelabel_censor").rglob("*"):
        if path.is_file():
            watched[path] = path.read_bytes()
    for path in (out / experiment_id / "config").rglob("*"):
        if path.is_file():
            watched[path] = path.read_bytes()
    canonical_root = wcs.canonical_experiment_root(experiment_id, experiments)
    for path in canonical_root.rglob("*"):
        if path.is_file():
            watched[path] = path.read_bytes()

    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
        from_stage="local-downstream", to_stage="local-downstream",
        force=True, output_root=out, experiments_root=experiments,
        local_downstream_engine=_fake_downstream_engine(canonical),
    )
    assert result["completed_variants"] == list(_NONZERO)
    assert result["quarantined_artifacts"]

    quarantine = (
        out / experiment_id / "variants" / "close_7d_earlier"
        / wcs.LOCAL_DOWNSTREAM_QUARANTINE_DIR / wcs.LOCAL_DOWNSTREAM_QUARANTINE_KIND
    )
    kept = [
        p for p in quarantine.rglob(wcs.STEP8A_DATASET_NAME)
        if p.read_bytes() == b"half-written"
    ]
    assert kept, "the partial dataset was not quarantined"

    for path, payload in watched.items():
        assert path.read_bytes() == payload, f"{path} was modified by --force"


def test_a_first_variant_failure_stops_the_second_variant(tmp_path):
    calls: list = []
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical = pd.read_parquet(wcs.canonical_step8a_path(experiment_id, experiments))
    engine = _fake_downstream_engine(
        canonical, calls=calls, fail_variants=("close_7d_earlier",),
    )
    with pytest.raises(wcs.WindowClosureError, match="synthetic engine failure"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="local-downstream", to_stage="local-downstream",
            output_root=out, experiments_root=experiments,
            local_downstream_engine=engine,
        )
    assert [call["variant_id"] for call in calls] == ["close_7d_earlier"]
    assert not wcs.local_downstream_root(
        experiment_id, "close_14d_earlier", out,
    ).exists()


@pytest.mark.parametrize("stage", ["some-future-stage"])
def test_the_later_stages_are_still_fail_fast_locked(tmp_path, stage):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    before = _relative_files(out / experiment_id)
    with pytest.raises(wcs.WindowClosureError, match="not enabled"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage=stage, to_stage=stage,
            output_root=out, experiments_root=experiments,
            local_downstream_engine=_exploding_engine,
        )
    assert _relative_files(out / experiment_id) == before


def test_no_real_gee_entry_point_is_reachable(tmp_path):
    """The fail-closed guard would raise if any production GEE symbol ran."""
    import core.gee_utils as gee_utils
    import scripts.prepare_modis_for_step7 as prepare_modis
    import scripts.run_predictors_only as run_predictors_only

    result, _, _, _ = _run_local_downstream(tmp_path)
    assert result["gee_queries_run"] is False
    assert result["gee_exports_run"] is False
    for module, name in (
        (gee_utils, "init_gee"),
        (run_predictors_only, "export_image_direct_or_tiled"),
        (prepare_modis, "prepare_modis_for_step7"),
    ):
        with pytest.raises(AssertionError, match="fail-closed guard"):
            getattr(module, name)()


def test_no_aoi_or_date_is_hard_coded_in_the_local_downstream_code():
    import re

    for module_path in (
        _PROJECT_ROOT / "src" / "window_closure_sensitivity.py",
        _PROJECT_ROOT / "scripts" / "validate_window_closure_local_downstream.py",
    ):
        literals = _executable_string_literals(module_path)
        for experiment_id in REGISTRY_IDS:
            assert not [s for s in literals if experiment_id in s]
        assert [s for s in literals if re.search(r"(19|20)\d\d-\d\d-\d\d", s)] == []
