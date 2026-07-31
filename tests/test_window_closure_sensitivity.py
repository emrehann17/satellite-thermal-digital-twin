"""Tests for the window-closure sensitivity analysis
(src/window_closure_sensitivity.py).

Everything is synthetic and runs under tmp_path, injected through the module's
public `output_root` / `experiments_root` parameters -- never by monkeypatching
another module's globals. Experiment IDs come from the registry dynamically so
no AOI name is hard-coded here either.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.window_closure_sensitivity as wcs
from core.pipeline_orchestrator import LEGACY_EXPERIMENT_ID
from core.experiment_context import build_experiment_context
from core.regions import get_experiment, list_experiments

REGISTRY_IDS = tuple(
    sorted(e for e in list_experiments(include_disabled=False) if e != LEGACY_EXPERIMENT_ID)
)


def any_experiment() -> str:
    """An arbitrary registry experiment -- never a hard-coded AOI name."""
    for experiment_id in REGISTRY_IDS:
        experiment = get_experiment(experiment_id)
        if experiment.get("predictor_start_date") and experiment.get("label_start_date"):
            return experiment_id
    pytest.skip("no registry experiment with a defined predictor window")


def ctx_for(experiment_id: str) -> dict:
    return build_experiment_context(experiment_id)


def _parse(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# =============================================================================
# Fail-closed guard: no test in this module may reach production Earth Engine.
#
# `plan -> predictor-export`, `prelabel-export -> predictor-export` and
# `predictor-export -> predictor-export` are SUPPORTED actual stage ranges, so
# a test that runs them without injecting a fake exporter/engine would
# otherwise reach the real Step6 prelabel exporter, GEE init/query/export or a
# geemap download. Every production entry point is therefore replaced, for
# EVERY test in this module, with a function that fails the test explicitly.
# =============================================================================
@pytest.fixture(autouse=True)
def _no_production_earth_engine(monkeypatch):
    def _blocked(name):
        def _fail(*_args, **_kwargs):
            raise AssertionError(
                f"fail-closed guard: production {name} was invoked from a "
                "unit test; inject a fake prelabel exporter / predictor "
                "engine instead"
            )
        return _fail

    import core.gee_utils as gee_utils
    import scripts.prepare_modis_for_step7 as prepare_modis
    import scripts.run_predictors_only as run_predictors_only
    import src.step6_validate_fire_relation as step6

    monkeypatch.setattr(
        step6, "export_raw_mcd64a1_prelabel_labels",
        _blocked("Step6 prelabel exporter (export_raw_mcd64a1_prelabel_labels)"),
    )
    monkeypatch.setattr(
        gee_utils, "init_gee", _blocked("Earth Engine initialisation (init_gee)"),
    )
    monkeypatch.setattr(
        run_predictors_only, "export_image_direct_or_tiled",
        _blocked("GEE exporter (export_image_direct_or_tiled)"),
    )
    monkeypatch.setattr(
        prepare_modis, "prepare_modis_for_step7",
        _blocked("production MODIS exporter (prepare_modis_for_step7)"),
    )
    monkeypatch.setattr(
        wcs, "production_predictor_engine",
        _blocked("production predictor engine (production_predictor_engine)"),
    )
    if "geemap" in sys.modules:
        monkeypatch.setattr(
            sys.modules["geemap"], "ee_export_image",
            _blocked("geemap download (ee_export_image)"), raising=False,
        )
    else:
        guard_geemap = types.ModuleType("geemap")
        guard_geemap.ee_export_image = _blocked("geemap download (ee_export_image)")
        monkeypatch.setitem(sys.modules, "geemap", guard_geemap)


# =============================================================================
# 1-3, 10-13. Shifts, identity and registry sourcing
# =============================================================================
def test_shifts_are_deterministically_ordered():
    assert wcs.normalize_shifts([14, 0, 7]) == (0, 7, 14)
    assert wcs.normalize_shifts([7, 14, 0]) == (0, 7, 14)
    assert wcs.normalize_shifts(None) == (0, 7, 14)


def test_input_shift_order_does_not_change_the_analysis_id():
    experiment_id = any_experiment()
    ctx = ctx_for(experiment_id)
    censor_a = wcs.common_prelabel_interval(wcs.build_window_variants(ctx, [0, 7, 14]))
    censor_b = wcs.common_prelabel_interval(wcs.build_window_variants(ctx, [14, 7, 0]))
    inventory: dict = {}
    config_a = wcs.scientific_configuration(
        experiment_id, ctx, wcs.build_window_variants(ctx, [0, 7, 14]),
        censor_a, inventory, wcs.normalize_shifts([0, 7, 14]),
    )
    config_b = wcs.scientific_configuration(
        experiment_id, ctx, wcs.build_window_variants(ctx, [14, 7, 0]),
        censor_b, inventory, wcs.normalize_shifts([14, 7, 0]),
    )
    assert wcs.compute_analysis_id(config_a) == wcs.compute_analysis_id(config_b)


def test_duplicate_shifts_are_deterministically_deduplicated():
    assert wcs.normalize_shifts([7, 7, 0, 14, 14]) == (0, 7, 14)


def test_negative_shift_fails_fast():
    with pytest.raises(wcs.WindowClosureError, match="Negative closure shift"):
        wcs.normalize_shifts([0, -7])


def test_canonical_window_comes_from_the_registry():
    experiment_id = any_experiment()
    registry = get_experiment(experiment_id)
    canonical = wcs.canonical_window(ctx_for(experiment_id))
    assert canonical["predictor_start_date"] == registry["predictor_start_date"]
    assert canonical["predictor_end_date"] == registry["predictor_end_date"]
    assert canonical["label_start_date"] == registry["label_start_date"]


def test_arbitrary_future_experiment_id_is_supported_by_the_pure_layer():
    """The window contract depends only on dates, never on an AOI name."""
    synthetic_ctx = {
        "predictor_start_date": "2030-06-01",
        "predictor_end_date": "2030-07-27",
        "label_start_date": "2030-07-28",
        "label_end_date": "2030-08-31",
        "baseline_years": [2026, 2027],
        "current_period_days": 56,
    }
    variants = wcs.build_window_variants(synthetic_ctx, [0, 7, 14])
    assert [v["variant_id"] for v in variants] == [
        "canonical", "close_7d_earlier", "close_14d_earlier",
    ]


def test_no_registry_aoi_name_is_hard_coded_in_the_implementation():
    for module_path in (
        _PROJECT_ROOT / "src" / "window_closure_sensitivity.py",
        _PROJECT_ROOT / "scripts" / "run_window_closure_sensitivity.py",
    ):
        source = module_path.read_text(encoding="utf-8")
        for experiment_id in REGISTRY_IDS:
            assert experiment_id not in source, f"{experiment_id} hard-coded in {module_path.name}"


def test_building_a_variant_context_does_not_mutate_the_registry_context(tmp_path):
    experiment_id = any_experiment()
    base = ctx_for(experiment_id)
    snapshot = {k: v for k, v in base.items()}
    wcs.build_window_variant_context(
        experiment_id, 7, base_context=base, output_root=tmp_path / "out",
    )
    assert base["predictor_start_date"] == snapshot["predictor_start_date"]
    assert base["predictor_end_date"] == snapshot["predictor_end_date"]
    assert base["output_root"] == snapshot["output_root"]
    # ...and the registry entry itself is untouched.
    assert get_experiment(experiment_id)["predictor_start_date"] == snapshot["predictor_start_date"]


# =============================================================================
# 4-9. Window arithmetic
# =============================================================================
@pytest.mark.parametrize("shift", [7, 14])
def test_shift_moves_both_ends_by_exactly_that_many_days(shift):
    experiment_id = any_experiment()
    ctx = ctx_for(experiment_id)
    canonical = wcs.canonical_window(ctx)
    variant = next(
        v for v in wcs.build_window_variants(ctx, [0, shift]) if v["shift_days"] == shift
    )
    assert _parse(variant["predictor_start_date"]) == \
           _parse(canonical["predictor_start_date"]) - timedelta(days=shift)
    assert _parse(variant["predictor_end_date"]) == \
           _parse(canonical["predictor_end_date"]) - timedelta(days=shift)


def test_every_variant_preserves_the_canonical_duration():
    ctx = ctx_for(any_experiment())
    canonical = wcs.canonical_window(ctx)
    for variant in wcs.build_window_variants(ctx, [0, 7, 14]):
        assert variant["duration_days"] == canonical["duration_days"]
        assert variant["duration_preserved"] is True


def test_label_window_is_identical_in_every_variant():
    ctx = ctx_for(any_experiment())
    canonical = wcs.canonical_window(ctx)
    for variant in wcs.build_window_variants(ctx, [0, 7, 14]):
        assert variant["label_start_date"] == canonical["label_start_date"]
        assert variant["label_end_date"] == canonical["label_end_date"]
        assert variant["label_window_unchanged"] is True


def test_lead_days_are_computed_correctly():
    ctx = ctx_for(any_experiment())
    canonical = wcs.canonical_window(ctx)
    for variant in wcs.build_window_variants(ctx, [0, 7, 14]):
        expected = (
            _parse(variant["label_start_date"]) - _parse(variant["predictor_end_date"])
        ).days
        assert variant["lead_days"] == expected
        # Each earlier closure adds exactly its shift to the lead.
        assert variant["lead_days"] == canonical["lead_days"] + variant["shift_days"]


def test_predictor_end_always_precedes_label_start():
    ctx = ctx_for(any_experiment())
    for variant in wcs.build_window_variants(ctx, [0, 7, 14]):
        assert _parse(variant["predictor_end_date"]) < _parse(variant["label_start_date"])


def test_window_length_and_closure_shift_are_not_conflated():
    """A 7-day shift must not become a 7-day window."""
    ctx = ctx_for(any_experiment())
    canonical = wcs.canonical_window(ctx)
    variant = next(v for v in wcs.build_window_variants(ctx, [0, 7]) if v["shift_days"] == 7)
    assert variant["duration_days"] == canonical["duration_days"] != 7


def test_a_shift_that_would_cross_label_start_fails_fast():
    synthetic_ctx = {
        "predictor_start_date": "2030-06-01",
        "predictor_end_date": "2030-07-27",
        "label_start_date": "2030-07-28",
        "label_end_date": "2030-08-31",
        "baseline_years": [2027],
        "current_period_days": 56,
    }
    forward = dict(synthetic_ctx, predictor_end_date="2030-07-28")
    with pytest.raises(wcs.WindowClosureError, match="precede label_start"):
        wcs.canonical_window(forward)


# =============================================================================
# 15-23. Export plans
# =============================================================================
def _landsat_plan(shift: int = 7):
    ctx = ctx_for(any_experiment())
    canonical = wcs.canonical_window(ctx)
    variant = next(v for v in wcs.build_window_variants(ctx, [0, shift]) if v["shift_days"] == shift)
    return variant, canonical, wcs.landsat_export_plan(
        variant, canonical["baseline_years"], canonical["current_period_days"],
    )


def test_current_lst_and_ndvi_use_the_variant_window():
    variant, canonical, plan = _landsat_plan(7)
    current = [r for r in plan["roles"] if r["scope"] == "current_window"]
    assert {r["role"] for r in current} == {"current_lst", "current_ndvi"}
    for role in current:
        assert role["end_date"] == variant["predictor_end_date"]
        expected_start = _parse(variant["predictor_end_date"]) - timedelta(
            days=canonical["current_period_days"]
        )
        assert role["start_date"] == expected_start.strftime("%Y-%m-%d")


def test_baseline_lst_and_ndvi_years_use_the_same_shifted_calendar_window():
    variant, canonical, plan = _landsat_plan(14)
    baseline = [r for r in plan["roles"] if r["scope"] == "baseline_year"]
    assert baseline, "baseline roles must be planned"
    by_year: dict[int, list[dict]] = {}
    for role in baseline:
        by_year.setdefault(role["baseline_year"], []).append(role)
    for year, roles in by_year.items():
        assert {r["family"] for r in roles} == {"lst", "ndvi"}
        # Both families share one identical shifted window for that year.
        assert len({(r["start_date"], r["end_date"]) for r in roles}) == 1
        # ...and that window is the canonical symmetric one, shifted.
        expected = wcs._baseline_year_window(
            variant["predictor_end_date"], canonical["current_period_days"], year,
        )
        assert roles[0]["start_date"] == expected["start_date"]
        assert roles[0]["end_date"] == expected["end_date"]


def test_modis_current_window_uses_the_variant_dates():
    experiment_id = any_experiment()
    ctx = ctx_for(experiment_id)
    variant = next(v for v in wcs.build_window_variants(ctx, [0, 7]) if v["shift_days"] == 7)
    plan = wcs.modis_export_plan(variant, experiment_id)
    assert plan["start_date"] == variant["predictor_start_date"]
    assert plan["end_date"] == variant["predictor_end_date"]
    assert {r["role"] for r in plan["roles"]} == {
        "modis_lst_mean", "modis_lst_std", "modis_valid_observation_count",
    }
    assert "prepare_modis_for_step7" in plan["producer"]


def test_production_scene_weighted_products_are_selected():
    _, _, plan = _landsat_plan(7)
    for role in plan["roles"]:
        assert role["products"] == ["scene_weighted_median", "scene_valid_count"]
    assert plan["reducer"] == "scene_weighted"


def test_date_balanced_products_cannot_enter_the_primary_plan():
    _, _, plan = _landsat_plan(7)
    for role in plan["roles"]:
        for product in role["products"]:
            assert "date_balanced" not in product
    leaked = [dict(plan["roles"][0], products=["date_balanced_median"])]
    with pytest.raises(wcs.WindowClosureError, match="reducer-counterfactual"):
        wcs.assert_no_forbidden_products(leaked)


def test_static_inputs_are_shared_and_read_only():
    plan = wcs.static_shared_plan()
    assert plan["mode"] == "shared_read_only"
    for role in ("dem_elevation", "dem_slope", "landcover_aligned", "label_window",
                 "model_feature_registry", "random_seed", "spatial_block_definition"):
        assert role in plan["roles"]


def test_predictor_export_plan_carries_no_label_or_burned_role():
    _, _, landsat = _landsat_plan(7)
    experiment_id = any_experiment()
    ctx = ctx_for(experiment_id)
    variant = next(v for v in wcs.build_window_variants(ctx, [0, 7]) if v["shift_days"] == 7)
    modis = wcs.modis_export_plan(variant, experiment_id)
    blob = json.dumps({"landsat": landsat, "modis": modis}).lower()
    for banned in ("burned", "burn_date", "burndate", "label_raster"):
        assert banned not in blob, f"{banned!r} leaked into the predictor export plan"


# =============================================================================
# 25-27. Common pre-label censoring
# =============================================================================
def test_common_prelabel_interval_spans_earliest_start_to_label_start_minus_one():
    ctx = ctx_for(any_experiment())
    variants = wcs.build_window_variants(ctx, [0, 7, 14])
    censor = wcs.common_prelabel_interval(variants)
    earliest = min(v["predictor_start_date"] for v in variants)
    label_start = _parse(variants[0]["label_start_date"])
    assert censor["common_prelabel_start"] == earliest
    assert censor["common_prelabel_end"] == (label_start - timedelta(days=1)).strftime("%Y-%m-%d")


def test_censor_interval_is_independent_of_the_exclude_flag():
    ctx = ctx_for(any_experiment())
    censor = wcs.common_prelabel_interval(wcs.build_window_variants(ctx, [0, 7, 14]))
    assert censor["applies_to_all_variants"] is True
    assert censor["independent_of_exclude_pre_label_burns_flag"] is True


def test_prelabel_export_plan_stays_inside_the_diagnostics_namespace(tmp_path):
    experiment_id = any_experiment()
    ctx = ctx_for(experiment_id)
    censor = wcs.common_prelabel_interval(wcs.build_window_variants(ctx, [0, 7, 14]))
    plan = wcs.prelabel_export_plan(experiment_id, censor, output_root=tmp_path / "out")
    assert str(tmp_path / "out") in plan["raster_path"]
    assert plan["writes_into_canonical_namespace"] is False
    assert plan["canonical_gate_rerun"] is False


def test_zero_prelabel_burns_is_a_valid_outcome():
    """No censored cell must be reported as count 0, never as an error."""
    assert wcs.censored_cell_ids(None) == set()
    assert wcs.censored_cell_ids(pd.DataFrame({"cell_id": []})) == set()


# =============================================================================
# Synthetic Step8A frames
# =============================================================================
def synthetic_step8a(n: int = 80, *, seed: int = 0, offset: float = 0.0,
                     drop_ids: list[str] | None = None,
                     burned_override: list[int] | None = None,
                     rowcol_override: bool = False,
                     duplicate: bool = False) -> pd.DataFrame:
    from src.step8b_train_baseline_vs_thermal_model import THERMAL_MODEL_FEATURES

    rng = np.random.default_rng(seed)
    rows = list(range(n))
    cols = [i % 4 for i in range(n)]
    cell_ids = [f"r{r}_c{c}" for r, c in zip(rows, cols)]
    burned = burned_override if burned_override is not None else [1 if i % 2 == 0 else 0 for i in range(n)]
    frame = pd.DataFrame({
        "cell_id": cell_ids,
        "row_500m": rows,
        "col_500m": cols,
        "burned": burned,
        "valid_for_modeling": True,
        "analysis_eligible": True,
        wcs.PRIMARY_POPULATION: True,
    })
    for index, feature in enumerate(THERMAL_MODEL_FEATURES):
        if feature == "landcover_dominant":
            frame[feature] = 10
            continue
        signal = np.asarray(burned, dtype="float64") * (0.6 + 0.1 * index)
        frame[feature] = offset + signal + rng.normal(0, 0.4, size=n)
    frame["landcover_dominant"] = 10
    if rowcol_override:
        frame.loc[0, "row_500m"] = 9999
    if drop_ids:
        frame = frame[~frame["cell_id"].isin(drop_ids)].reset_index(drop=True)
    if duplicate:
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    return frame


# =============================================================================
# 28-34. Common cohort
# =============================================================================
def test_common_cohort_is_the_exact_three_way_intersection():
    frames = {
        "canonical": synthetic_step8a(seed=1),
        "close_7d_earlier": synthetic_step8a(seed=2, drop_ids=["r0_c0"]),
        "close_14d_earlier": synthetic_step8a(seed=3, drop_ids=["r1_c1"]),
    }
    cohort = wcs.build_common_cohort(frames)
    ids = set(cohort["common_cell_ids"])
    assert "r0_c0" not in ids and "r1_c1" not in ids
    assert len(ids) == 78
    for name in frames:
        assert len(cohort["common"][name]) == 78


def test_common_cohort_fails_when_labels_differ():
    base = synthetic_step8a(seed=1)
    flipped = base.copy()
    flipped["burned"] = 1 - flipped["burned"]
    with pytest.raises(wcs.WindowClosureError, match="Labels differ"):
        wcs.build_common_cohort({"canonical": base, "close_7d_earlier": flipped})


def test_common_cohort_fails_when_row_col_differ():
    base = synthetic_step8a(seed=1)
    moved = synthetic_step8a(seed=1, rowcol_override=True)
    with pytest.raises(wcs.WindowClosureError, match="differs on the common cohort"):
        wcs.build_common_cohort({"canonical": base, "close_7d_earlier": moved})


def test_common_cohort_fails_when_population_membership_differs():
    base = synthetic_step8a(seed=1)
    other = synthetic_step8a(seed=1)
    # Same eligibility filter passes, but the recorded membership flag differs.
    other.loc[:, wcs.PRIMARY_POPULATION] = other[wcs.PRIMARY_POPULATION].astype(object)
    other.loc[0, wcs.PRIMARY_POPULATION] = True
    cohort = wcs.build_common_cohort({"canonical": base, "close_7d_earlier": other})
    assert cohort["summary"]["common_rows"] == len(base)


def test_duplicate_cell_id_fails_fast():
    with pytest.raises(wcs.WindowClosureError, match="duplicate cell_id"):
        wcs.build_common_cohort({
            "canonical": synthetic_step8a(seed=1, duplicate=True),
            "close_7d_earlier": synthetic_step8a(seed=2),
        })


def test_empty_common_cohort_fails_fast():
    left = synthetic_step8a(seed=1)
    right = synthetic_step8a(seed=1)
    right["cell_id"] = right["cell_id"] + "_shifted"
    with pytest.raises(wcs.WindowClosureError, match="common cohort is empty"):
        wcs.build_common_cohort({"canonical": left, "close_7d_earlier": right})


def test_single_class_common_cohort_fails_fast():
    frames = {
        "canonical": synthetic_step8a(seed=1, burned_override=[1] * 80),
        "close_7d_earlier": synthetic_step8a(seed=2, burned_override=[1] * 80),
    }
    with pytest.raises(wcs.WindowClosureError, match="single class"):
        wcs.build_common_cohort(frames)


def test_censored_cells_are_removed_from_every_variant_identically():
    frames = {
        "canonical": synthetic_step8a(seed=1),
        "close_7d_earlier": synthetic_step8a(seed=2),
        "close_14d_earlier": synthetic_step8a(seed=3),
    }
    censored = {"r0_c0", "r2_c2", "r4_c0"}
    cohort = wcs.build_common_cohort(frames, censored=censored)
    for name in frames:
        remaining = set(cohort["common"][name]["cell_id"])
        assert not (remaining & censored)
    assert cohort["summary"]["pre_label_censored_cells"] == 3


def test_cohort_summary_reports_native_and_retention_figures():
    frames = {
        "canonical": synthetic_step8a(seed=1),
        "close_7d_earlier": synthetic_step8a(seed=2, drop_ids=["r0_c0", "r1_c1"]),
    }
    summary = wcs.build_common_cohort(frames)["summary"]
    assert summary["common_rows"] == 78
    for name in frames:
        entry = summary["per_variant"][name]
        assert entry["native_eligible_rows"] >= entry["common_rows"]
        assert 0.0 < entry["common_row_retention"] <= 1.0
        assert entry["common_positive_retention"] is not None


def test_analysis_eligible_false_rows_are_excluded():
    frame = synthetic_step8a(seed=1)
    frame.loc[:9, "analysis_eligible"] = False
    assert len(wcs.variant_eligible_rows(frame)) == 70


# =============================================================================
# 35-38. Shared folds and feature contract
# =============================================================================
def test_one_fold_assignment_is_reused_by_every_variant():
    from src.landsat_composite_downstream_ab import build_fold_assignment

    frames = {
        "canonical": synthetic_step8a(seed=1),
        "close_7d_earlier": synthetic_step8a(seed=2),
        "close_14d_earlier": synthetic_step8a(seed=3),
    }
    cohort = wcs.build_common_cohort(frames)
    assignment, _, _ = build_fold_assignment(cohort["common"]["canonical"])
    fold_id = assignment["cv_fold"].to_numpy()
    # The SAME assignment object is applied to every variant, so identity is
    # structural rather than a coincidence of re-derivation.
    for name in frames:
        assert np.array_equal(fold_id, assignment["cv_fold"].to_numpy())
        assert len(cohort["common"][name]) == len(fold_id)


def test_fold_assignment_is_grouped_not_a_random_row_split():
    from src.landsat_composite_downstream_ab import build_fold_assignment

    cohort = wcs.build_common_cohort({
        "canonical": synthetic_step8a(seed=1),
        "close_7d_earlier": synthetic_step8a(seed=2),
    })
    assignment, _, _ = build_fold_assignment(cohort["common"]["canonical"])
    # Every row of a spatial block lands in the same fold -- impossible under a
    # random row split.
    grouped = assignment.groupby("spatial_block_id")["cv_fold"].nunique()
    assert (grouped == 1).all()
    assert assignment["seed"].nunique() == 1
    assert assignment["block_size_cells"].nunique() == 1


def test_feature_order_matches_the_production_step8_registry():
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, THERMAL_MODEL_FEATURES,
    )
    config = wcs.scientific_configuration(
        any_experiment(), ctx_for(any_experiment()),
        wcs.build_window_variants(ctx_for(any_experiment()), [0, 7]),
        wcs.common_prelabel_interval(wcs.build_window_variants(ctx_for(any_experiment()), [0, 7])),
        {}, (0, 7),
    )
    registry = config["feature_registry"]
    assert registry["baseline_features_in_order"] == list(BASELINE_FEATURES)
    assert registry["thermal_model_features_in_order"] == list(THERMAL_MODEL_FEATURES)


def test_no_baseline_invariance_gate_is_applied():
    """A closure shift moves current NDVI, so baseline features legitimately
    change; the LST-only downstream A/B gate must not be reused."""
    source = (_PROJECT_ROOT / "src" / "window_closure_sensitivity.py").read_text(encoding="utf-8")
    assert "check_baseline_invariance" not in source
    assert "baseline" in source.lower()


# =============================================================================
# 39-41. Metric directions
# =============================================================================
def test_thermal_contribution_directions():
    result = {
        "baseline": {"roc_auc": 0.70, "pr_auc": 0.20, "brier": 0.20},
        "thermal": {"roc_auc": 0.80, "pr_auc": 0.30, "brier": 0.15},
    }
    contribution = wcs.thermal_contribution(result)
    assert contribution["delta_roc_auc"] == pytest.approx(0.10)
    assert contribution["delta_pr_auc"] == pytest.approx(0.10)
    # Positive brier_improvement favours the thermal model.
    assert contribution["brier_improvement"] == pytest.approx(0.05)


def test_brier_improvement_sign_favours_the_thermal_model():
    worse = wcs.thermal_contribution({
        "baseline": {"roc_auc": 0.7, "pr_auc": 0.2, "brier": 0.10},
        "thermal": {"roc_auc": 0.7, "pr_auc": 0.2, "brier": 0.18},
    })
    assert worse["brier_improvement"] < 0


def test_paired_change_is_earlier_minus_canonical():
    canonical = {m: 0.0 for m in wcs.PAIRED_CHANGE_METRICS}
    canonical["delta_pr_auc"] = 0.10
    variant = {m: 0.0 for m in wcs.PAIRED_CHANGE_METRICS}
    variant["delta_pr_auc"] = 0.08
    change = wcs.paired_window_change(canonical, variant)
    assert change["delta_pr_auc"] == pytest.approx(-0.02)


def test_scientific_regression_delta_pr_auc_changes():
    """canonical 0.10, close_7d 0.08 (-0.02), close_14d 0.04 (-0.06)."""
    canonical = {m: 0.0 for m in wcs.PAIRED_CHANGE_METRICS} | {"delta_pr_auc": 0.10}
    close_7 = {m: 0.0 for m in wcs.PAIRED_CHANGE_METRICS} | {"delta_pr_auc": 0.08}
    close_14 = {m: 0.0 for m in wcs.PAIRED_CHANGE_METRICS} | {"delta_pr_auc": 0.04}
    assert wcs.paired_window_change(canonical, close_7)["delta_pr_auc"] == pytest.approx(-0.02)
    assert wcs.paired_window_change(canonical, close_14)["delta_pr_auc"] == pytest.approx(-0.06)
    # Both earlier closures reduce the thermal contribution.
    assert wcs.paired_window_change(canonical, close_7)["delta_pr_auc"] < 0
    assert wcs.paired_window_change(canonical, close_14)["delta_pr_auc"] < \
           wcs.paired_window_change(canonical, close_7)["delta_pr_auc"]


# =============================================================================
# 42-45. Bootstrap and interval language
# =============================================================================
def test_bootstrap_resamples_spatial_blocks_and_shares_draws():
    frames = {
        "canonical": synthetic_step8a(seed=1),
        "close_7d_earlier": synthetic_step8a(seed=2),
    }
    cohort = wcs.build_common_cohort(frames)
    from src.landsat_composite_downstream_ab import build_fold_assignment
    _, blocked, _ = build_fold_assignment(cohort["common"]["canonical"])

    labels = blocked["burned"].astype(int).to_numpy()
    rng = np.random.default_rng(0)
    probabilities = {
        "canonical": {
            "baseline": rng.uniform(0, 1, len(labels)),
            "thermal": rng.uniform(0, 1, len(labels)),
        },
        "close_7d_earlier": {
            "baseline": rng.uniform(0, 1, len(labels)),
            "thermal": rng.uniform(0, 1, len(labels)),
        },
    }
    bootstrap = wcs.multi_variant_block_bootstrap(
        blocked, labels, probabilities, n_bootstrap=40, seed=7,
    )
    assert bootstrap["bootstrap_unit"] == "spatial_block_id"
    assert bootstrap["identical_block_draws_across_variants"] is True
    assert bootstrap["n_bootstrap_valid"] > 0
    # One replicate row carries every variant's metrics -> the draws are shared.
    for variant in ("canonical", "close_7d_earlier"):
        assert f"{variant}__delta_pr_auc" in bootstrap["replicates"].columns


def test_bootstrap_refuses_a_single_block():
    frame = pd.DataFrame({"spatial_block_id": ["b0"] * 10})
    with pytest.raises(wcs.WindowClosureError, match="at least two spatial blocks"):
        wcs.multi_variant_block_bootstrap(
            frame, [0, 1] * 5,
            {"canonical": {"baseline": [0.5] * 10, "thermal": [0.5] * 10}},
            n_bootstrap=5,
        )


def test_interval_classification_language():
    assert wcs.classify_change_interval(0.01, 0.2) == wcs.INTERVAL_SUPPORTED_INCREASE
    assert wcs.classify_change_interval(-0.2, -0.01) == wcs.INTERVAL_SUPPORTED_DECREASE
    assert wcs.classify_change_interval(-0.01, 0.05) == wcs.INTERVAL_INCLUDES_ZERO
    assert wcs.classify_change_interval(None, None) == wcs.INTERVAL_INCLUDES_ZERO


def test_interval_including_zero_never_claims_equivalence():
    for value in (wcs.INTERVAL_SUPPORTED_INCREASE, wcs.INTERVAL_SUPPORTED_DECREASE,
                  wcs.INTERVAL_INCLUDES_ZERO):
        lowered = value.lower()
        for banned in ("stable", "equivalent", "significant", "no_difference"):
            assert banned not in lowered


def test_paired_change_rows_are_deterministically_ordered():
    replicates = pd.DataFrame({
        "canonical__delta_pr_auc": [0.10, 0.11, 0.09],
        "close_7d_earlier__delta_pr_auc": [0.08, 0.09, 0.07],
        "canonical__delta_roc_auc": [0.05, 0.05, 0.05],
        "close_7d_earlier__delta_roc_auc": [0.04, 0.04, 0.04],
    })
    bootstrap = {
        "replicates": replicates, "variants": ["canonical", "close_7d_earlier"],
        "ci_lower_percentile": 2.5, "ci_upper_percentile": 97.5,
    }
    point_metrics = {
        "canonical": {m: 0.0 for m in wcs.PAIRED_CHANGE_METRICS} | {"delta_pr_auc": 0.10},
        "close_7d_earlier": {m: 0.0 for m in wcs.PAIRED_CHANGE_METRICS} | {"delta_pr_auc": 0.08},
    }
    rows = wcs.build_paired_change_rows(bootstrap, point_metrics)
    assert rows == sorted(rows, key=lambda r: (r["variant_id"], r["metric"]))
    assert all(r["variant_id"] != "canonical" for r in rows)
    pr = next(r for r in rows if r["metric"] == "delta_pr_auc")
    assert pr["change_definition"] == "earlier_closure_minus_canonical"
    assert pr["point_estimate"] == pytest.approx(-0.02)
    assert pr["interval_status"] == wcs.INTERVAL_SUPPORTED_DECREASE


# =============================================================================
# 46-49, 52-57. Dry run, namespace safety, provenance
# =============================================================================
def test_dry_run_writes_no_file_or_directory(tmp_path):
    out = tmp_path / "out"
    result = wcs.run_analysis(
        experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True, output_root=out,
    )
    assert result["ran"] is False
    assert result["files_written"] is False
    assert not out.exists()


def test_dry_run_reports_every_required_flag(tmp_path):
    result = wcs.run_analysis(
        experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
        output_root=tmp_path / "out",
    )
    for key in ("experiment_id", "schema_version", "canonical_window", "label_window",
                "variants", "shift_days", "duration_preserved", "common_prelabel_censor",
                "baseline_windows_per_year", "landsat_export_roles", "modis_export_roles",
                "static_shared_roles", "frozen_canonical_step8a", "planned_output_paths",
                "planned_stages"):
        assert key in result, f"dry-run is missing '{key}'"
    for flag in ("files_written", "gee_queries_run", "gee_exports_run", "model_fit", "bootstrap_run"):
        assert result[flag] is False, f"{flag} must be False in a dry run"


def test_dry_run_never_imports_or_calls_earth_engine(tmp_path):
    """`ee` must not be touched: no initialise, no query, no export."""
    import builtins

    real_import = builtins.__import__
    touched: list[str] = []

    def guarded(name, *args, **kwargs):
        if name == "ee" or name.startswith("ee."):
            touched.append(name)
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=guarded):
        wcs.run_analysis(
            experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
            output_root=tmp_path / "out",
        )
    assert touched == [], f"dry-run imported Earth Engine: {touched}"


def test_dry_run_never_fits_a_model_or_runs_a_bootstrap(tmp_path):
    from sklearn.base import BaseEstimator

    def _boom(*_args, **_kwargs):
        raise AssertionError("dry-run must not fit a model")

    def _boom_bootstrap(*_args, **_kwargs):
        raise AssertionError("dry-run must not run a bootstrap")

    with patch.object(BaseEstimator, "fit", _boom, create=True), \
            patch.object(wcs, "multi_variant_block_bootstrap", _boom_bootstrap):
        result = wcs.run_analysis(
            experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
            output_root=tmp_path / "out",
        )
    assert result["model_fit"] is False and result["bootstrap_run"] is False


def test_output_root_injection_moves_every_planned_path(tmp_path):
    out = tmp_path / "injected"
    result = wcs.run_analysis(
        experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True, output_root=out,
    )
    for path in result["planned_output_paths"].values():
        assert str(out) in path, f"planned path escaped the injected root: {path}"


def test_experiments_root_injection_moves_the_frozen_reference(tmp_path):
    experiment_id = any_experiment()
    fake_root = tmp_path / "experiments"
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7], dry_run=True,
        output_root=tmp_path / "out", experiments_root=fake_root,
    )
    assert str(fake_root) in result["frozen_canonical_step8a"]["path"]
    # Nothing exists there, so the hash is honestly reported as None.
    assert result["frozen_canonical_step8a"]["sha256"] is None


def test_variant_namespace_never_points_into_canonical_outputs(tmp_path):
    experiment_id = any_experiment()
    base = ctx_for(experiment_id)
    ctx = wcs.build_window_variant_context(
        experiment_id, 7, base_context=base, output_root=tmp_path / "out",
    )
    canonical_root = Path(base["output_root"]).resolve()
    for key, value in ctx.items():
        if not isinstance(value, Path) or key in wcs.READ_ONLY_SHARED_CONTEXT_KEYS:
            continue
        resolved = value.resolve()
        assert canonical_root not in resolved.parents and resolved != canonical_root, \
            f"{key} points into the canonical namespace"


def test_static_shared_inputs_stay_on_the_canonical_read_only_copies(tmp_path):
    """DEM/slope and the aligned landcover do not depend on the predictor
    window, so every variant READS the same canonical artefact rather than
    re-exporting its own -- one less moving part."""
    experiment_id = any_experiment()
    base = ctx_for(experiment_id)
    for shift in (7, 14):
        ctx = wcs.build_window_variant_context(
            experiment_id, shift, base_context=base, output_root=tmp_path / "out",
        )
        for key in wcs.READ_ONLY_SHARED_CONTEXT_KEYS:
            if key in base:
                assert ctx[key] == base[key], f"{key} must stay canonical/read-only"


def test_planned_paths_stay_inside_the_dedicated_namespace(tmp_path):
    experiment_id = any_experiment()
    ctx = ctx_for(experiment_id)
    variants = wcs.build_window_variants(ctx, [0, 7, 14])
    out = tmp_path / "out"
    root = str(wcs.experiment_root(experiment_id, out))
    for path in wcs.plan_output_paths(experiment_id, variants, out).values():
        assert path.startswith(root)


def test_frozen_sentinels_are_untouched_by_a_dry_run(tmp_path):
    sentinels = {}
    for relative in ("experiments/step8a_sentinel.parquet",
                     "diagnostics/other_namespace/sentinel.json"):
        path = tmp_path / "frozen" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"frozen": True}))
        sentinels[path] = _sha256(path)

    wcs.run_analysis(
        experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
        output_root=tmp_path / "out",
    )
    for path, digest in sentinels.items():
        assert _sha256(path) == digest


def test_canonical_step8a_hash_is_reported_and_unchanged(tmp_path):
    experiment_id = any_experiment()
    path = wcs.canonical_step8a_path(experiment_id)
    if not path.is_file():
        pytest.skip("canonical Step8A not present in this checkout")
    before = _sha256(path)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=True,
        output_root=tmp_path / "out",
    )
    assert result["frozen_canonical_step8a"]["sha256"] == before
    assert _sha256(path) == before


def test_an_unrecognized_stage_is_gated_behind_an_explicit_error(tmp_path):
    """A stage this build does not know refuses loudly and creates nothing.

    Every DECLARED stage is implemented now, so the build lock's remaining job
    is to catch a stage name that was never built. The refusal happens in
    `validate_stage_range`, i.e. before any prerequisite, exporter, engine,
    mkdir or write.
    """
    out = tmp_path / "out"
    with pytest.raises(wcs.WindowClosureError, match="not enabled or recognized"):
        wcs.run_analysis(
            experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="some-future-stage",
            output_root=out,
        )
    assert not out.exists(), "an unrecognized stage created the namespace"
    with pytest.raises(wcs.WindowClosureError, match="not enabled"):
        wcs.assert_actual_stages_supported(["some-future-stage"])


# =============================================================================
# Stage range
# =============================================================================
def test_stage_order_and_range_validation():
    assert wcs.STAGES == (
        "plan", "prelabel-export", "predictor-export", "local-downstream", "model", "compare",
    )
    assert wcs.validate_stage_range("plan", "compare") == list(wcs.STAGES)
    assert wcs.validate_stage_range("model", "compare") == ["model", "compare"]
    with pytest.raises(wcs.WindowClosureError):
        wcs.validate_stage_range("compare", "plan")
    with pytest.raises(wcs.WindowClosureError, match="not enabled or recognized"):
        wcs.validate_stage_range("plan", "not_a_stage")


def test_model_stage_requires_its_inputs():
    with pytest.raises(wcs.WindowClosureError, match="requires"):
        wcs.assert_stage_prerequisites(["prelabel-export", "predictor-export", "model"])


def test_compare_stage_requires_the_model_stage():
    with pytest.raises(wcs.WindowClosureError, match="requires 'model'"):
        wcs.assert_stage_prerequisites(["local-downstream", "compare"])


# =============================================================================
# 62-65. Reporting
# =============================================================================
def _summary_fixture() -> dict:
    ctx = ctx_for(any_experiment())
    canonical = wcs.canonical_window(ctx)
    variants = wcs.build_window_variants(ctx, [0, 7, 14])
    return {
        "schema_version": wcs.SCHEMA_VERSION,
        "experiment_id": any_experiment(),
        "analysis_id": "a" * 64,
        "primary_population": wcs.PRIMARY_POPULATION,
        "primary_model": wcs.PRIMARY_MODEL,
        "variants": variants,
        "label_window": {
            "start_date": canonical["label_start_date"],
            "end_date": canonical["label_end_date"],
        },
        "common_censor_interval": wcs.common_prelabel_interval(variants),
        "common_cohort_summary": {
            "common_rows": 100, "common_positives": 40, "pre_label_censored_cells": 3,
        },
        "variant_metrics": [
            {"variant_id": "canonical", "baseline_roc_auc": 0.70, "thermal_roc_auc": 0.80,
             "delta_roc_auc": 0.10, "delta_pr_auc": 0.10, "brier_improvement": 0.05},
            {"variant_id": "close_7d_earlier", "baseline_roc_auc": 0.70, "thermal_roc_auc": 0.78,
             "delta_roc_auc": 0.08, "delta_pr_auc": 0.08, "brier_improvement": 0.04},
        ],
        "paired_changes": [
            {"variant_id": "close_7d_earlier", "metric": "delta_pr_auc",
             "point_estimate": -0.02, "ci_low": -0.05, "ci_high": -0.005,
             "valid_replicates": 1000, "interval_status": wcs.INTERVAL_SUPPORTED_DECREASE},
        ],
    }


def test_markdown_carries_no_banned_operational_or_causal_wording():
    markdown = wcs.render_summary_markdown(_summary_fixture())
    wcs.assert_report_wording(markdown)
    lowered = markdown.lower()
    for banned in ("statistically significant", "early warning", "operational risk"):
        assert banned not in lowered


def test_markdown_states_the_interpretation_boundary():
    lowered = wcs.render_summary_markdown(_summary_fixture()).lower()
    assert "not an operational forecasting" in lowered
    # Neutral uncertainty language only: an interval including zero is
    # reported as unresolved direction, never re-framed as any other finding.
    assert "interval that includes zero" in lowered
    assert "do not resolve the direction of the change" in lowered
    assert "uncertainty remains" in lowered
    assert "does not establish a causal mechanism" in lowered
    assert "prevalence" in lowered
    assert "single aoi and a single season" in lowered
    # The scientific wording contract: none of these substrings may appear
    # anywhere in the generated prose. ("stable" is checked as the word, not a
    # stem: the required causal-boundary sentence contains "establish".)
    for forbidden in ("equivalent", "equivalence", "stable", "stability",
                      "robust", "robustness", "significant", "significance"):
        assert forbidden not in lowered, forbidden


def test_report_wording_guard_rejects_banned_phrases():
    with pytest.raises(wcs.WindowClosureError, match="banned wording"):
        wcs.assert_report_wording("The change was statistically significant.")


def test_markdown_lists_canonical_and_both_earlier_variants():
    markdown = wcs.render_summary_markdown(_summary_fixture())
    for variant in ("canonical", "close_7d_earlier", "close_14d_earlier"):
        assert variant in markdown


def test_planned_outputs_contain_only_preregistered_variants(tmp_path):
    result = wcs.run_analysis(
        experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
        output_root=tmp_path / "out",
    )
    allowed = {"canonical", "close_7d_earlier", "close_14d_earlier"}
    for key in result["planned_output_paths"]:
        if key.startswith("variants/"):
            assert key.split("/")[1] in allowed, f"unexpected variant namespace: {key}"


def test_planned_output_paths_are_deterministically_sorted(tmp_path):
    paths = wcs.run_analysis(
        experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
        output_root=tmp_path / "out",
    )["planned_output_paths"]
    assert list(paths) == sorted(paths)


# =============================================================================
# 66-78. Pre-actual provenance: label resolution, date semantics, month filter,
#        MODIS window binding
# =============================================================================
def _synthetic_window_context() -> dict:
    """A fully synthetic experiment context.

    No registry AOI and no production date is involved: the expected MODIS /
    Landsat windows below are derived from THESE numbers, so the module can
    never satisfy them by hard-coding a real experiment.
    """
    return {
        "experiment_id": "synthetic_window_ctx",
        "predictor_start_date": "2021-06-01",
        "predictor_end_date": "2021-07-27",
        "label_start_date": "2021-07-28",
        "label_end_date": "2021-08-31",
        "baseline_years": [2019, 2020],
        "current_period_days": 56,
    }


def _label_roles() -> tuple[str, str]:
    return wcs.LABEL_ROLE_RAW, wcs.LABEL_ROLE_BINARY


def _seed_canonical_labels(root: Path, experiment_id: str) -> dict[str, Path]:
    """Write synthetic canonical label rasters into an injected namespace."""
    labels_dir = wcs.canonical_experiment_root(experiment_id, root) / "validation" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for role, name in wcs.CANONICAL_LABEL_FILENAMES.items():
        path = labels_dir / name
        path.write_bytes(f"synthetic-{role}".encode("utf-8"))
        written[role] = path
    return written


# --- Label resolution --------------------------------------------------------
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


def test_guessed_burned_labels_filename_is_never_used(tmp_path):
    """The guessed `burned_labels.tif` must not be resolvable code, nor in the plan.

    Prose may still explain WHY it was wrong; only executable literals and the
    emitted plan are checked.
    """
    literals = _executable_string_literals(
        _PROJECT_ROOT / "src" / "window_closure_sensitivity.py"
    )
    assert not [s for s in literals if "burned_labels" in s]

    experiment_id = any_experiment()
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7], dry_run=True,
        output_root=tmp_path / "out",
    )
    blob = json.dumps(result, default=str)
    assert "burned_labels.tif" not in blob
    assert "label_raster" not in result["frozen_input_inventory"]


def test_canonical_raw_and_binary_label_paths_are_resolved(tmp_path):
    experiment_id = any_experiment()
    seeded = _seed_canonical_labels(tmp_path, experiment_id)
    resolved = wcs.resolve_label_inputs(experiment_id, tmp_path)

    raw_role, binary_role = _label_roles()
    assert set(resolved) == {raw_role, binary_role}
    assert Path(resolved[raw_role]["path"]) == seeded[raw_role]
    assert Path(resolved[binary_role]["path"]) == seeded[binary_role]
    assert resolved[raw_role]["canonical_filename"] == "mcd64a1_raw.tif"
    assert resolved[binary_role]["canonical_filename"] == "mcd64a1_burned.tif"


def test_label_kind_constant_matches_the_production_step8a_contract():
    from src.step8a_prepare_500m_modeling_dataset import LABEL_KIND_RAW

    assert wcs.LABEL_KIND_RAW_BURNDATE == LABEL_KIND_RAW


def test_step8a_run_metadata_is_preferred_over_the_filename_fallback(tmp_path):
    """The resolver prefers the label the frozen Step8A actually recorded."""
    experiment_id = any_experiment()
    _seed_canonical_labels(tmp_path, experiment_id)
    root = wcs.canonical_experiment_root(experiment_id, tmp_path)
    recorded = root / "validation" / "labels" / "mcd64a1_raw.tif"
    stats = root / "step8a" / "step8a_dataset_stats.json"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text(json.dumps({
        "label_kind": wcs.LABEL_KIND_RAW_BURNDATE,
        "reference_500m_label_source": {"path": str(recorded)},
    }), encoding="utf-8")

    resolved = wcs.resolve_label_inputs(experiment_id, tmp_path)
    raw_role, _ = _label_roles()
    assert resolved[raw_role]["resolved_from"] == wcs.LABEL_RESOLUTION_METADATA
    assert Path(resolved[raw_role]["path"]) == recorded


def test_recorded_label_path_outside_the_pinned_namespace_is_refused(tmp_path):
    """Injected roots are never escaped by a stale recorded absolute path."""
    experiment_id = any_experiment()
    _seed_canonical_labels(tmp_path, experiment_id)
    root = wcs.canonical_experiment_root(experiment_id, tmp_path)
    stats = root / "step8a" / "step8a_dataset_stats.json"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text(json.dumps({
        "label_kind": wcs.LABEL_KIND_RAW_BURNDATE,
        "reference_500m_label_source": {"path": "/somewhere/else/mcd64a1_raw.tif"},
    }), encoding="utf-8")

    resolved = wcs.resolve_label_inputs(experiment_id, tmp_path)
    raw_role, _ = _label_roles()
    assert Path(resolved[raw_role]["path"]).is_relative_to(root)
    assert resolved[raw_role]["resolved_from"] != wcs.LABEL_RESOLUTION_METADATA


def test_both_label_roles_are_hashed_in_the_frozen_input_inventory(tmp_path):
    experiment_id = any_experiment()
    seeded = _seed_canonical_labels(tmp_path, experiment_id)
    inventory = wcs.frozen_input_inventory(experiment_id, tmp_path)

    for role, path in seeded.items():
        entry = inventory[role]
        assert entry["exists"] is True
        assert entry["sha256"] == _sha256(path)
        assert entry["path"] == str(path)
        assert entry["role"] == role


def test_both_label_hashes_enter_the_analysis_id(tmp_path):
    """Changing EITHER label raster must change the analysis identity."""
    experiment_id = any_experiment()
    ctx = ctx_for(experiment_id)
    variants = wcs.build_window_variants(ctx, [0, 7])
    censor = wcs.common_prelabel_interval(variants)

    ids: dict[str, str] = {}
    configs: dict[str, dict] = {}
    for tag, mutated in (("base", None), ("raw", wcs.LABEL_ROLE_RAW),
                         ("binary", wcs.LABEL_ROLE_BINARY)):
        root = tmp_path / tag
        seeded = _seed_canonical_labels(root, experiment_id)
        if mutated is not None:
            seeded[mutated].write_bytes(b"mutated-label-content")
        inventory = wcs.frozen_input_inventory(experiment_id, root)
        config = wcs.scientific_configuration(
            experiment_id, ctx, variants, censor, inventory, [0, 7],
        )
        configs[tag] = config
        ids[tag] = wcs.compute_analysis_id(config)

    raw_role, binary_role = _label_roles()
    for role in (raw_role, binary_role):
        assert configs["base"]["frozen_input_sha256"][role] is not None
        assert configs["base"]["label_input_sha256"][role] is not None
    assert ids["raw"] != ids["base"], "raw BurnDate hash does not reach the analysis id"
    assert ids["binary"] != ids["base"], "binary label hash does not reach the analysis id"


def test_analysis_id_does_not_depend_on_the_injected_root(tmp_path):
    """Only hashes, never host paths, may enter the analysis identity."""
    experiment_id = any_experiment()
    ctx = ctx_for(experiment_id)
    variants = wcs.build_window_variants(ctx, [0, 7])
    censor = wcs.common_prelabel_interval(variants)

    ids = []
    for tag in ("root_a", "root_b"):
        root = tmp_path / tag
        _seed_canonical_labels(root, experiment_id)
        inventory = wcs.frozen_input_inventory(experiment_id, root)
        ids.append(wcs.compute_analysis_id(wcs.scientific_configuration(
            experiment_id, ctx, variants, censor, inventory, [0, 7],
        )))
    assert ids[0] == ids[1]


def test_input_shift_order_still_does_not_change_the_analysis_id(tmp_path):
    """Label hashes were added without making the id order-dependent."""
    experiment_id = any_experiment()
    root = tmp_path / "experiments"
    _seed_canonical_labels(root, experiment_id)
    ids = {
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=order, dry_run=True,
            output_root=tmp_path / "out", experiments_root=root,
        )["analysis_id"]
        for order in ([0, 7, 14], [14, 0, 7], [7, 14, 0, 7])
    }
    assert len(ids) == 1


# --- Missing-prerequisite behaviour ------------------------------------------
def test_missing_required_label_stops_the_actual_plan(tmp_path):
    """No preregistration may be written with a null label hash."""
    experiment_id = any_experiment()
    empty_root = tmp_path / "experiments"  # nothing seeded: both labels missing
    with pytest.raises(wcs.WindowClosureError, match="Required label input"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
            from_stage="plan", to_stage="plan",
            output_root=tmp_path / "out", experiments_root=empty_root,
        )
    assert not (tmp_path / "out").exists()


def test_one_missing_label_role_is_enough_to_stop_the_actual_plan(tmp_path):
    experiment_id = any_experiment()
    root = tmp_path / "experiments"
    seeded = _seed_canonical_labels(root, experiment_id)
    seeded[wcs.LABEL_ROLE_BINARY].unlink()
    with pytest.raises(wcs.WindowClosureError, match=wcs.LABEL_ROLE_BINARY):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
            from_stage="plan", to_stage="plan",
            output_root=tmp_path / "out", experiments_root=root,
        )


def test_dry_run_reports_missing_prerequisites_explicitly(tmp_path):
    experiment_id = any_experiment()
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7], dry_run=True,
        output_root=tmp_path / "out", experiments_root=tmp_path / "experiments",
    )
    assert result["prerequisites_ready"] is False
    missing = {entry["role"] for entry in result["missing_required_inputs"]}
    assert missing == set(wcs.REQUIRED_LABEL_ROLES)
    for entry in result["missing_required_inputs"]:
        assert entry["exists"] is False and entry["sha256"] is None
        assert entry["path"], "a missing input must still report where it was expected"
    assert not (tmp_path / "out").exists()


def test_dry_run_reports_ready_prerequisites_when_both_labels_exist(tmp_path):
    experiment_id = any_experiment()
    root = tmp_path / "experiments"
    _seed_canonical_labels(root, experiment_id)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7], dry_run=True,
        output_root=tmp_path / "out", experiments_root=root,
    )
    assert result["prerequisites_ready"] is True
    assert result["missing_required_inputs"] == []
    for role in wcs.REQUIRED_LABEL_ROLES:
        assert result["label_inputs"][role]["exists"] is True
        assert result["label_inputs"][role]["sha256"] is not None


# --- Date-semantics wording --------------------------------------------------
def test_window_closure_records_carry_no_compositing_only_wording(tmp_path):
    result = wcs.run_analysis(
        experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
        output_root=tmp_path / "out",
    )
    blob = json.dumps(result, default=str).lower()
    assert "compositing method is the only" not in blob
    assert "the compositing method is the only intentionally changed factor" not in blob


def test_window_closure_uses_the_correct_timing_wording():
    semantics = wcs.window_closure_date_window_semantics("2021-06-01", "2021-07-27")
    assert semantics["note"] == (
        "Earth Engine filterDate end is exclusive. Reducer, QA masking and "
        "processing policy are held fixed; predictor-window timing is the "
        "intentionally changed factor."
    )
    assert semantics["changed_factor"] == "predictor_window_timing"
    assert set(semantics["held_fixed"]) == {"reducer", "qa_masking", "processing_policy"}


def test_end_exclusive_arithmetic_is_still_the_upstream_one():
    """Only the wording is adapted; the off-by-one contract is unchanged."""
    from src.landsat_composite_counterfactual_audit import date_window_semantics

    upstream = date_window_semantics("2021-06-01", "2021-07-27")
    adapted = wcs.window_closure_date_window_semantics("2021-06-01", "2021-07-27")
    for key in ("filter_date_start", "filter_date_end", "end_semantics",
                "effective_last_included_date"):
        assert adapted[key] == upstream[key]
    assert adapted["effective_last_included_date"] == "2021-07-26"


def test_upstream_counterfactual_module_is_not_modified():
    """The reducer counterfactual keeps its own (correct, for it) wording."""
    from src.landsat_composite_counterfactual_audit import date_window_semantics

    assert "compositing method is the only" in date_window_semantics(
        "2021-06-01", "2021-07-27"
    )["note"].lower()


def test_foreign_factor_wording_guard_rejects_the_inherited_sentence():
    payload = {"note": "the compositing method is the only intentionally changed factor"}
    with pytest.raises(wcs.WindowClosureError, match="another analysis"):
        wcs.assert_no_foreign_factor_wording(payload, "test payload")


def test_report_guard_also_rejects_the_inherited_sentence():
    with pytest.raises(wcs.WindowClosureError, match="banned wording"):
        wcs.assert_report_wording(
            "the compositing method is the only intentionally changed factor"
        )


# --- Calendar-month filter transparency --------------------------------------
def test_current_window_month_filter_matches_the_production_step3_formula():
    """1-12 is production behaviour, not something this module invented."""
    ctx = _synthetic_window_context()
    current = wcs._current_window(ctx["predictor_end_date"], ctx["current_period_days"])

    start_dt = _parse(current["start_date"])
    end_dt = _parse(current["end_date"])
    production_months = sorted(
        {(start_dt.month + i) % 12 or 12 for i in range((end_dt - start_dt).days + 1)}
    )
    expected = f"{min(production_months)}-{max(production_months)}"
    assert current["months_filter"] == expected == "1-12"


def test_exact_filter_date_binding_is_reported_for_current_roles():
    ctx = _synthetic_window_context()
    variant = next(v for v in wcs.build_window_variants(ctx, [0, 7]) if v["shift_days"] == 7)
    plan = wcs.landsat_export_plan(
        variant, ctx["baseline_years"], ctx["current_period_days"],
    )
    current = [r for r in plan["roles"] if r["scope"] == "current_window"]
    assert current
    for role in current:
        assert role["calendar_month_filter"] == role["months_filter"] == "1-12"
        assert role["calendar_month_filter_redundant"] is True
        assert role["exact_filter_date_is_binding"] is True
        assert role["filter_date_start"] == role["start_date"]
        assert role["filter_date_end"] == role["end_date"]
        assert "does not mean that whole-year data is used" in role["note"].lower()


def test_production_month_filter_value_is_not_silently_changed():
    ctx = _synthetic_window_context()
    variant = wcs.build_window_variants(ctx, [0])[0]
    plan = wcs.landsat_export_plan(
        variant, ctx["baseline_years"], ctx["current_period_days"],
    )
    current = next(r for r in plan["roles"] if r["scope"] == "current_window")
    upstream = wcs._current_window(
        variant["predictor_end_date"], ctx["current_period_days"],
    )
    assert current["months_filter"] == upstream["months_filter"]


def test_a_narrow_window_is_not_reported_as_redundant():
    transparency = wcs.calendar_month_filter_transparency("6-7", "2021-06-01", "2021-07-27")
    assert transparency["calendar_month_filter_redundant"] is False
    assert transparency["exact_filter_date_is_binding"] is True


def test_dry_run_carries_the_month_filter_transparency_block(tmp_path):
    result = wcs.run_analysis(
        experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
        output_root=tmp_path / "out",
    )
    block = result["calendar_month_filter_transparency"]
    assert set(block) == {v["variant_id"] for v in result["variants"]}
    for entry in block.values():
        assert entry["exact_filter_date_is_binding"] is True
        assert "calendar_month_filter_redundant" in entry


def test_report_states_that_a_full_month_filter_is_not_whole_year_data():
    summary = _summary_fixture()
    summary["calendar_month_filter_transparency"] = wcs.calendar_month_filter_transparency(
        "1-12", "2021-06-01", "2021-07-27",
    )
    markdown = wcs.render_summary_markdown(summary)
    lowered = markdown.lower()
    assert "does not mean that whole-year data is used" in lowered
    assert "exact `filterdate` is binding" in lowered
    wcs.assert_report_wording(markdown)


def test_limitations_always_state_the_month_filter_caveat():
    markdown = wcs.render_summary_markdown(_summary_fixture())
    assert "does not mean that whole-year data is used" in markdown.lower()


# --- MODIS window binding ----------------------------------------------------
def test_modis_roles_carry_the_exact_predictor_window_of_every_variant(tmp_path):
    """Expected dates come from the SYNTHETIC context, never from the module."""
    ctx = _synthetic_window_context()
    variants = wcs.build_window_variants(ctx, [0, 7, 14])
    expected = {
        "canonical": ("2021-06-01", "2021-07-27"),
        "close_7d_earlier": ("2021-05-25", "2021-07-20"),
        "close_14d_earlier": ("2021-05-18", "2021-07-13"),
    }
    assert {v["variant_id"] for v in variants} == set(expected)

    experiment_id = any_experiment()
    for variant in variants:
        plan = wcs.modis_export_plan(variant, experiment_id, tmp_path / "out")
        start, end = expected[variant["variant_id"]]
        assert (plan["start_date"], plan["end_date"]) == (start, end)
        assert {r["role"] for r in plan["roles"]} == set(wcs.MODIS_ROLE_FILENAMES)
        for role in plan["roles"]:
            assert role["start_date"] == start
            assert role["end_date"] == end
            assert role["scope"] == "current_window"
            assert role["uses_variant_context"] is True
            assert "prepare_modis_for_step7" in role["producer"]
            assert role["output_path"].endswith(
                wcs.MODIS_ROLE_FILENAMES[role["role"]]
            )


def test_modis_role_output_paths_stay_in_the_variant_namespace(tmp_path):
    ctx = _synthetic_window_context()
    experiment_id = any_experiment()
    out = tmp_path / "out"
    for variant in wcs.build_window_variants(ctx, [0, 7, 14]):
        vroot = wcs.variant_root(experiment_id, variant["variant_id"], out).resolve()
        for role in wcs.modis_export_plan(variant, experiment_id, out)["roles"]:
            assert Path(role["output_path"]).resolve().is_relative_to(vroot)


def test_modis_role_output_filenames_are_the_production_ones():
    from scripts.prepare_modis_for_step7 import resolve_modis_output_paths

    paths = resolve_modis_output_paths(
        {"is_kozan": False, "modis_input_dir": Path("/tmp/x/data/modis")}
    )
    production = {
        "modis_lst_mean": paths["mean_path"].name,
        "modis_lst_std": paths["std_path"].name,
        "modis_valid_observation_count": paths["valid_count_path"].name,
    }
    assert wcs.MODIS_ROLE_FILENAMES == production


def test_dry_run_modis_roles_carry_every_required_field(tmp_path):
    result = wcs.run_analysis(
        experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
        output_root=tmp_path / "out",
    )
    required = {"role", "scope", "start_date", "end_date", "producer",
                "output_path", "uses_variant_context"}
    variant_windows = {v["variant_id"]: (v["predictor_start_date"], v["predictor_end_date"])
                       for v in result["variants"]}
    for variant_id, roles in result["modis_export_roles"].items():
        start, end = variant_windows[variant_id]
        for role in roles:
            assert required <= set(role)
            assert (role["start_date"], role["end_date"]) == (start, end)
            assert role["uses_variant_context"] is True


# --- Dry-run side-effect freedom (with the new provenance layer) --------------
def test_dry_run_still_writes_nothing_and_calls_nothing(tmp_path):
    """Label hashing must read files, never write, fit or query."""
    import builtins

    from sklearn.base import BaseEstimator

    out = tmp_path / "out"
    experiments = tmp_path / "experiments"
    _seed_canonical_labels(experiments, any_experiment())
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    real_import = builtins.__import__
    touched: list[str] = []

    def guarded(name, *args, **kwargs):
        if name == "ee" or name.startswith("ee."):
            touched.append(name)
        return real_import(name, *args, **kwargs)

    def _boom(*_args, **_kwargs):
        raise AssertionError("dry-run must not fit a model")

    with patch.object(builtins, "__import__", side_effect=guarded), \
            patch.object(BaseEstimator, "fit", _boom, create=True), \
            patch.object(wcs, "multi_variant_block_bootstrap", _boom):
        result = wcs.run_analysis(
            experiment_id=any_experiment(), shifts=[0, 7, 14], dry_run=True,
            output_root=out, experiments_root=experiments,
        )

    assert touched == [], f"dry-run imported Earth Engine: {touched}"
    assert not out.exists()
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    for flag in ("files_written", "gee_queries_run", "gee_exports_run",
                 "model_fit", "bootstrap_run"):
        assert result[flag] is False


# =============================================================================
# 79-98. Actual PLAN stage (the first implemented actual stage)
# =============================================================================
PLAN_DOCUMENT_KEYS: tuple[str, ...] = (
    "config/preregistration.json",
    "config/window_variants.csv",
    "config/frozen_input_inventory.json",
    "prelabel_censor/export_plan.json",
    "variants/canonical/frozen_reference.json",
    "variants/close_7d_earlier/export_plan.json",
    "variants/close_14d_earlier/export_plan.json",
)

FORBIDDEN_PLAN_SUFFIXES: tuple[str, ...] = (
    ".tif", ".tiff", ".parquet", ".pkl", ".joblib", ".npy", ".npz", ".nc", ".h5",
)


def _seed_frozen_inputs(root: Path, experiment_id: str) -> dict[str, Path]:
    """Create every REQUIRED frozen input inside an injected experiments root.

    The layout comes from the module's own inventory, so the fixture can never
    drift from the paths the plan actually pins.
    """
    inventory = wcs.frozen_input_inventory(experiment_id, root)
    written: dict[str, Path] = {}
    for role in wcs.REQUIRED_FROZEN_INPUT_ROLES:
        path = Path(inventory[role]["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-frozen-{role}".encode("utf-8"))
        written[role] = path
    return written


def _plan_env(tmp_path: Path) -> tuple[str, Path, Path]:
    experiment_id = any_experiment()
    experiments = tmp_path / "experiments"
    out = tmp_path / "diagnostics"
    _seed_frozen_inputs(experiments, experiment_id)
    return experiment_id, out, experiments


def _run_plan(tmp_path: Path, shifts=(0, 7, 14), **kwargs) -> dict:
    experiment_id, out, experiments = _plan_env(tmp_path)
    return wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(shifts), dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments, **kwargs,
    )


def _relative_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


# --- 1, 20. The seven plan documents ----------------------------------------
def test_actual_plan_writes_exactly_the_seven_plan_documents(tmp_path):
    result = _run_plan(tmp_path)
    out = tmp_path / "diagnostics"
    experiment_root = out / result["experiment_id"]

    assert result["files_written_count"] == 7
    assert len(result["files_written"]) == 7
    written = {str(Path(p).relative_to(experiment_root)) for p in result["files_written"]}
    assert written == set(PLAN_DOCUMENT_KEYS)
    # ...and nothing else exists on disk.
    assert _relative_files(experiment_root) == sorted(PLAN_DOCUMENT_KEYS)


def test_actual_plan_result_flags(tmp_path):
    result = _run_plan(tmp_path)
    assert result["ran"] is True
    assert result["dry_run"] is False
    assert result["stages_run"] == ["plan"]
    assert result["prerequisites_ready"] is True
    assert result["missing_required_inputs"] == []
    assert result["frozen_hashes_unchanged"] is True
    assert result["analysis_id"] and len(result["analysis_id"]) == 64
    for flag in ("gee_queries_run", "gee_exports_run", "model_fit", "bootstrap_run"):
        assert result[flag] is False, f"{flag} must be False in the plan stage"


# --- 2. No scientific artefact ----------------------------------------------
def test_plan_stage_produces_no_raster_table_or_model(tmp_path):
    result = _run_plan(tmp_path)
    out = tmp_path / "diagnostics"
    for path in out.rglob("*"):
        if path.is_file():
            assert path.suffix in wcs.PLAN_DOCUMENT_SUFFIXES, f"plan wrote {path}"
            assert path.suffix not in FORBIDDEN_PLAN_SUFFIXES
    assert result["files_written_count"] == 7


# --- 3, 4. No Earth Engine, no model, no bootstrap --------------------------
def test_plan_stage_never_imports_or_calls_earth_engine(tmp_path):
    import builtins

    experiment_id, out, experiments = _plan_env(tmp_path)
    real_import = builtins.__import__
    touched: list[str] = []

    def guarded(name, *args, **kwargs):
        if name == "ee" or name.startswith("ee."):
            touched.append(name)
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=guarded):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="plan",
            output_root=out, experiments_root=experiments,
        )
    assert touched == [], f"plan stage imported Earth Engine: {touched}"


def test_plan_stage_never_fits_a_model_or_runs_a_bootstrap(tmp_path):
    from sklearn.base import BaseEstimator

    experiment_id, out, experiments = _plan_env(tmp_path)

    def _boom(*_args, **_kwargs):
        raise AssertionError("the plan stage must not fit or bootstrap")

    with patch.object(BaseEstimator, "fit", _boom, create=True), \
            patch.object(wcs, "multi_variant_block_bootstrap", _boom), \
            patch.object(wcs, "build_common_cohort", _boom):
        result = wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="plan",
            output_root=out, experiments_root=experiments,
        )
    assert result["model_fit"] is False and result["bootstrap_run"] is False


# --- 5, 6. Canonical vs early variants --------------------------------------
def test_canonical_variant_gets_no_export_plan(tmp_path):
    result = _run_plan(tmp_path)
    root = tmp_path / "diagnostics" / result["experiment_id"]
    canonical_dir = root / "variants" / wcs.CANONICAL_VARIANT_ID

    assert _relative_files(canonical_dir) == ["frozen_reference.json"]
    assert not (canonical_dir / "export_plan.json").exists()

    document = json.loads((canonical_dir / "frozen_reference.json").read_text())
    assert document["is_canonical"] is True
    assert document["predictor_export_planned"] is False
    assert document["landsat_export_planned"] is False
    assert document["modis_export_planned"] is False
    assert document["reads_frozen_production_outputs"] is True
    blob = json.dumps(document).lower()
    for token in ("current_lst", "current_ndvi", "baseline_lst", "modis_lst_mean"):
        assert token not in blob, f"canonical frozen reference plans {token}"


@pytest.mark.parametrize("shifts", [(0, 7, 14), (0, 3, 21), (0, 5)])
def test_every_non_zero_shift_gets_an_export_plan(tmp_path, shifts):
    """Variant documents follow the preregistered shifts, not a fixed list."""
    result = _run_plan(tmp_path, shifts=shifts)
    root = tmp_path / "diagnostics" / result["experiment_id"]
    non_zero = [s for s in sorted(set(shifts)) if s != 0]

    for shift in non_zero:
        plan_path = root / "variants" / wcs.variant_id(shift) / "export_plan.json"
        assert plan_path.is_file(), f"missing export plan for shift {shift}"
        assert json.loads(plan_path.read_text())["shift_days"] == shift
    assert result["files_written_count"] == 4 + 1 + len(non_zero)


# --- 7, 8. Exact variant dates ----------------------------------------------
def test_plan_documents_carry_the_exact_variant_dates(tmp_path):
    result = _run_plan(tmp_path)
    root = tmp_path / "diagnostics" / result["experiment_id"]
    expected = {
        v["variant_id"]: (v["predictor_start_date"], v["predictor_end_date"])
        for v in result["variants"]
    }

    rows = list(csv.DictReader(
        (root / "config" / "window_variants.csv").read_text().splitlines()
    ))
    assert [r["variant_id"] for r in rows] == [
        v["variant_id"] for v in sorted(result["variants"], key=lambda v: v["shift_days"])
    ]
    for row in rows:
        assert (row["predictor_start_date"], row["predictor_end_date"]) == \
            expected[row["variant_id"]]

    for variant in result["variants"]:
        if variant["is_canonical"]:
            continue
        document = json.loads(
            (root / "variants" / variant["variant_id"] / "export_plan.json").read_text()
        )
        start, end = expected[variant["variant_id"]]
        assert document["predictor_start_date"] == start
        assert document["predictor_end_date"] == end
        assert document["lead_days"] == variant["lead_days"]
        assert document["reducer"] == "scene_weighted"
        for role in document["landsat"]["current_roles"]:
            assert role["end_date"] == end
        assert document["landsat"]["baseline_roles"], "baseline roles must be planned"
        assert {r["role"] for r in document["landsat"]["current_roles"]} == \
            {"current_lst", "current_ndvi"}
        assert document["gee_queries_run"] is False
        assert document["gee_exports_run"] is False
        assert document["planned_output_paths"]


def test_plan_modis_records_use_the_exact_variant_dates(tmp_path):
    result = _run_plan(tmp_path)
    root = tmp_path / "diagnostics" / result["experiment_id"]
    for variant in result["variants"]:
        if variant["is_canonical"]:
            continue
        modis = json.loads(
            (root / "variants" / variant["variant_id"] / "export_plan.json").read_text()
        )["modis"]
        assert modis["start_date"] == variant["predictor_start_date"]
        assert modis["end_date"] == variant["predictor_end_date"]
        assert {r["role"] for r in modis["roles"]} == set(wcs.MODIS_ROLE_FILENAMES)
        for role in modis["roles"]:
            assert role["start_date"] == variant["predictor_start_date"]
            assert role["end_date"] == variant["predictor_end_date"]
            assert role["uses_variant_context"] is True


def test_prelabel_export_plan_document_is_written(tmp_path):
    result = _run_plan(tmp_path)
    root = tmp_path / "diagnostics" / result["experiment_id"]
    document = json.loads((root / "prelabel_censor" / "export_plan.json").read_text())

    variants_start = min(v["predictor_start_date"] for v in result["variants"])
    assert document["common_prelabel_start"] == variants_start
    assert document["common_prelabel_end"] < result["variants"][0]["label_start_date"]
    assert "export_raw_mcd64a1_prelabel_labels" in document["producer"]
    assert document["planned_raster_path"].endswith(".tif")
    assert not Path(document["planned_raster_path"]).exists()
    assert document["applies_to_all_variants"] is True
    assert document["gee_queries_run"] is False
    assert document["gee_exports_run"] is False


# --- 9. Label hashes in the written documents -------------------------------
def test_label_hashes_are_in_the_preregistration_and_the_inventory(tmp_path):
    experiment_id, out, experiments = _plan_env(tmp_path)
    seeded = _seed_frozen_inputs(experiments, experiment_id)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    root = out / experiment_id
    prereg = json.loads((root / "config" / "preregistration.json").read_text())
    inventory = json.loads((root / "config" / "frozen_input_inventory.json").read_text())

    assert prereg["analysis_id"] == result["analysis_id"]
    assert inventory["analysis_id"] == result["analysis_id"]
    for role in wcs.REQUIRED_LABEL_ROLES:
        expected = _sha256(seeded[role])
        assert prereg["frozen_input_sha256"][role] == expected
        assert prereg["label_inputs"][role]["sha256"] == expected
        assert prereg["scientific_configuration"]["label_input_sha256"][role] == expected
        assert inventory["inventory"][role]["sha256"] == expected
        assert inventory["frozen_input_sha256"][role] == expected


# --- 10. Missing prerequisites write nothing at all -------------------------
def test_missing_label_writes_no_directory_or_file(tmp_path):
    experiment_id = any_experiment()
    experiments = tmp_path / "experiments"
    out = tmp_path / "diagnostics"
    seeded = _seed_frozen_inputs(experiments, experiment_id)
    seeded[wcs.LABEL_ROLE_RAW].unlink()

    with pytest.raises(wcs.WindowClosureError, match="Required label input"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="plan",
            output_root=out, experiments_root=experiments,
        )
    assert not out.exists(), "a refused plan must not create its namespace"


def test_missing_frozen_static_input_writes_nothing(tmp_path):
    experiment_id = any_experiment()
    experiments = tmp_path / "experiments"
    out = tmp_path / "diagnostics"
    seeded = _seed_frozen_inputs(experiments, experiment_id)
    seeded["canonical_step8a"].unlink()

    with pytest.raises(wcs.WindowClosureError, match="canonical_step8a"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="plan",
            output_root=out, experiments_root=experiments,
        )
    assert not out.exists()


def test_every_required_frozen_role_is_gated(tmp_path):
    experiment_id = any_experiment()
    for role in wcs.REQUIRED_FROZEN_INPUT_ROLES:
        experiments = tmp_path / f"experiments_{role}"
        out = tmp_path / f"diagnostics_{role}"
        seeded = _seed_frozen_inputs(experiments, experiment_id)
        seeded[role].unlink()
        with pytest.raises(wcs.WindowClosureError):
            wcs.run_analysis(
                experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
                from_stage="plan", to_stage="plan",
                output_root=out, experiments_root=experiments,
            )
        assert not out.exists(), f"missing {role} still created the namespace"


# --- 11, 12. Locked stages stay locked --------------------------------------
# plan/prelabel-export/predictor-export/local-downstream are IMPLEMENTED actual
# stages now (see their sections for the supported ranges); only ranges that
# include model or compare must still fail fast.
def test_every_declared_stage_is_implemented():
    """All six stages are implemented; the build lock now guards new ones."""
    assert wcs.IMPLEMENTED_ACTUAL_STAGES == wcs.STAGES
    wcs.assert_actual_stages_supported(list(wcs.STAGES))


def test_an_unimplemented_stage_still_fails_fast():
    with pytest.raises(wcs.WindowClosureError, match="not enabled"):
        wcs.assert_actual_stages_supported(["some-future-stage"])


# --- 13, 14, 15. Idempotence, identity conflict, force ----------------------
def test_rerun_with_the_same_analysis_id_is_idempotent(tmp_path):
    experiment_id, out, experiments = _plan_env(tmp_path)
    kwargs = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    first = wcs.run_analysis(**kwargs)
    before = {p: p.read_bytes() for p in (out / experiment_id).rglob("*") if p.is_file()}

    second = wcs.run_analysis(**kwargs)
    after = {p: p.read_bytes() for p in (out / experiment_id).rglob("*") if p.is_file()}

    assert second["analysis_id"] == first["analysis_id"]
    assert second["reused"] is True
    assert second["files_rewritten"] == []
    assert second["files_written_count"] == 7
    assert after == before, "an idempotent rerun changed the plan documents"


def test_a_different_analysis_id_fails_without_force(tmp_path):
    experiment_id, out, experiments = _plan_env(tmp_path)
    first = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    before = {p: p.read_bytes() for p in (out / experiment_id).rglob("*") if p.is_file()}

    with pytest.raises(wcs.WindowClosureError, match="DIFFERENT analysis_id"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
            from_stage="plan", to_stage="plan",
            output_root=out, experiments_root=experiments,
        )
    after = {p: p.read_bytes() for p in (out / experiment_id).rglob("*") if p.is_file()}
    assert after == before, "a refused rerun modified the existing plan"
    assert json.loads(
        (out / experiment_id / "config" / "preregistration.json").read_text()
    )["analysis_id"] == first["analysis_id"]


def test_force_overwrites_only_the_plan_owned_documents(tmp_path):
    experiment_id, out, experiments = _plan_env(tmp_path)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    root = out / experiment_id

    # A scientific artefact that the plan stage does NOT own.
    foreign = {
        root / "comparison" / "bootstrap_replicates.parquet": b"not-plan-owned",
        root / "variants" / "close_7d_earlier" / "step8a" / "dataset.parquet": b"artefact",
        root / "prelabel_censor" / "prelabel_burndate.tif": b"raster",
    }
    for path, payload in foreign.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
        from_stage="plan", to_stage="plan", force=True,
        output_root=out, experiments_root=experiments,
    )
    assert result["forced"] is True
    assert json.loads(
        (root / "config" / "preregistration.json").read_text()
    )["analysis_id"] == result["analysis_id"]
    for path, payload in foreign.items():
        assert path.is_file(), f"force deleted a non-plan-owned artefact: {path}"
        assert path.read_bytes() == payload, f"force modified {path}"
    # The stale 14-day variant plan is reported, never deleted.
    stale = root / "variants" / "close_14d_earlier" / "export_plan.json"
    assert stale.is_file()
    assert str(stale) in result["unmanaged_plan_documents"]


def test_plan_write_targets_are_refused_outside_the_namespace(tmp_path):
    experiment_id = any_experiment()
    with pytest.raises(wcs.WindowClosureError, match="escapes"):
        wcs.assert_plan_owned_targets(
            experiment_id,
            {"config/preregistration.json": tmp_path / "elsewhere.json"},
            tmp_path / "diagnostics",
        )


def test_plan_write_targets_refuse_non_document_suffixes(tmp_path):
    experiment_id = any_experiment()
    root = wcs.experiment_root(experiment_id, tmp_path / "diagnostics")
    with pytest.raises(wcs.WindowClosureError, match="not a plan-owned"):
        wcs.assert_plan_owned_targets(
            experiment_id,
            {"prelabel_censor/prelabel_burndate.tif": root / "prelabel_censor" / "x.tif"},
            tmp_path / "diagnostics",
        )


# --- 16. Nothing outside the namespace moves --------------------------------
def test_canonical_and_foreign_diagnostics_sentinels_are_untouched(tmp_path):
    experiment_id, out, experiments = _plan_env(tmp_path)

    canonical_root = wcs.canonical_experiment_root(experiment_id, experiments)
    sentinels = {
        canonical_root / "step8b" / "canonical_sentinel.json": b'{"canonical": true}',
        out / "other_diagnostic" / "sentinel.json": b'{"other": true}',
        out / "other_diagnostic" / "result.parquet": b"foreign-artefact",
    }
    for path, payload in sentinels.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    frozen_before = {
        p: p.read_bytes() for p in canonical_root.rglob("*") if p.is_file()
    }

    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )

    for path, payload in sentinels.items():
        assert path.is_file() and path.read_bytes() == payload, f"{path} was touched"
    assert {p: p.read_bytes() for p in canonical_root.rglob("*") if p.is_file()} \
        == frozen_before, "the canonical namespace was modified"


# --- 17. Frozen hashes before and after -------------------------------------
def test_frozen_hashes_are_identical_before_and_after_the_plan(tmp_path):
    experiment_id, out, experiments = _plan_env(tmp_path)
    before = wcs.frozen_hash_map(wcs.frozen_input_inventory(experiment_id, experiments))

    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    after = wcs.frozen_hash_map(wcs.frozen_input_inventory(experiment_id, experiments))

    assert before == after
    assert result["frozen_input_sha256"] == before
    assert result["frozen_hashes_unchanged"] is True
    assert all(value is not None for value in before.values() if value is not None)


def test_a_frozen_input_that_moves_is_refused():
    with pytest.raises(wcs.WindowClosureError, match="changed while writing"):
        wcs.assert_frozen_hashes_unchanged(
            {"dem_slope": "a" * 64}, {"dem_slope": "b" * 64}, "while writing",
        )


# --- 18. Dry run is unchanged ------------------------------------------------
def test_dry_run_still_writes_nothing_after_the_plan_stage_exists(tmp_path):
    experiment_id, out, experiments = _plan_env(tmp_path)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=True,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    assert result["ran"] is False
    assert result["dry_run"] is True
    assert result["files_written"] is False
    assert not out.exists()


def test_dry_run_and_actual_plan_agree_on_the_analysis_id(tmp_path):
    experiment_id, out, experiments = _plan_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14],
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    dry = wcs.run_analysis(dry_run=True, **common)
    actual = wcs.run_analysis(dry_run=False, **common)
    assert dry["analysis_id"] == actual["analysis_id"]


# --- Determinism of the written documents ------------------------------------
def test_plan_documents_are_byte_identical_across_roots(tmp_path):
    """Only the injected root may differ between two otherwise identical runs."""
    experiment_id = any_experiment()
    texts: list[dict[str, str]] = []
    for tag in ("a", "b"):
        experiments = tmp_path / f"exp_{tag}"
        out = tmp_path / f"out_{tag}"
        _seed_frozen_inputs(experiments, experiment_id)
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="plan",
            output_root=out, experiments_root=experiments,
        )
        root = out / experiment_id
        texts.append({
            str(p.relative_to(root)): p.read_text(encoding="utf-8")
                .replace(str(out), "<OUT>").replace(str(experiments), "<EXP>")
            for p in root.rglob("*") if p.is_file()
        })
    assert texts[0] == texts[1]


def test_written_json_keys_and_csv_rows_are_deterministically_ordered(tmp_path):
    result = _run_plan(tmp_path)
    root = tmp_path / "diagnostics" / result["experiment_id"]
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text())
        assert list(payload) == sorted(payload), f"{path} keys are not sorted"

    rows = list(csv.DictReader(
        (root / "config" / "window_variants.csv").read_text().splitlines()
    ))
    assert list(rows[0]) == list(wcs.WINDOW_VARIANTS_CSV_COLUMNS)
    assert [int(r["shift_days"]) for r in rows] == sorted(int(r["shift_days"]) for r in rows)


def test_plan_documents_carry_no_foreign_factor_wording(tmp_path):
    result = _run_plan(tmp_path)
    root = tmp_path / "diagnostics" / result["experiment_id"]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            assert "compositing method is the only" not in path.read_text().lower()


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    result = _run_plan(tmp_path)
    root = tmp_path / "diagnostics" / result["experiment_id"]
    leftovers = [p for p in root.rglob("*") if p.name.startswith(".") or p.suffix == ".tmp"]
    assert leftovers == []


# =============================================================================
# 99-130. Actual PRELABEL-EXPORT stage
# =============================================================================
from rasterio.transform import Affine  # noqa: E402  (test-only helper import)

# A small synthetic grid. The numbers are arbitrary test fixture values, not
# production ones: the implementation never sees them except through the files.
_TEST_TRANSFORM = Affine(0.00026949458523585647, 0.0, 31.05,
                         0.0, -0.00026949458523585647, 37.35)
_TEST_SHAPE = (6, 5)
_TEST_NODATA = -32768


def _write_grid_raster(path: Path, values, *, transform=None, crs="EPSG:4326",
                       nodata=_TEST_NODATA, bands: int = 1) -> Path:
    import rasterio

    array = np.asarray(values, dtype="int16")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=bands, dtype="int16", crs=crs,
        transform=transform if transform is not None else _TEST_TRANSFORM,
        nodata=nodata,
    ) as dst:
        for band in range(1, bands + 1):
            dst.write(array, band)
    return path


def _doy(date_text: str) -> int:
    return _parse(date_text).timetuple().tm_yday


def _seed_prelabel_env(tmp_path: Path) -> tuple[str, Path, Path]:
    """Frozen inputs where the raw BurnDate label is a REAL reference raster."""
    experiment_id = any_experiment()
    experiments = tmp_path / "experiments"
    out = tmp_path / "diagnostics"
    _seed_frozen_inputs(experiments, experiment_id)
    inventory = wcs.frozen_input_inventory(experiment_id, experiments)
    _write_grid_raster(
        Path(inventory[wcs.LABEL_ROLE_RAW]["path"]), np.zeros(_TEST_SHAPE),
    )
    return experiment_id, out, experiments


def _censor_interval(experiment_id: str, shifts=(0, 7, 14)) -> dict:
    ctx = ctx_for(experiment_id)
    return wcs.common_prelabel_interval(wcs.build_window_variants(ctx, list(shifts)))


def _fake_exporter(values=None, *, calls: list | None = None, corrupt: bool = False,
                   transform=None, crs="EPSG:4326", bands: int = 1):
    """A stand-in for the production Step6 exporter. Never touches Earth Engine."""
    def exporter(experiment_id, pre_label_start, pre_label_end, raw_out):
        if calls is not None:
            calls.append({
                "experiment_id": experiment_id,
                "pre_label_start": pre_label_start,
                "pre_label_end": pre_label_end,
                "raw_out": Path(raw_out),
            })
        if corrupt:
            Path(raw_out).parent.mkdir(parents=True, exist_ok=True)
            Path(raw_out).write_bytes(b"not-a-geotiff")
        else:
            grid = np.zeros(_TEST_SHAPE) if values is None else np.asarray(values)
            _write_grid_raster(Path(raw_out), grid, transform=transform, crs=crs, bands=bands)
        return {
            "raw_path": Path(raw_out),
            "pre_label_window": [pre_label_start, pre_label_end],
            "experiment_id": experiment_id,
        }
    return exporter


def _exploding_exporter(*_args, **_kwargs):
    raise AssertionError("Earth Engine export must not be reached")


def _burned_grid(experiment_id: str, shifts=(0, 7, 14)) -> np.ndarray:
    """Two burns: one on the first day of the interval, one on the LAST day."""
    censor = _censor_interval(experiment_id, shifts)
    grid = np.zeros(_TEST_SHAPE)
    grid[0][0] = _doy(censor["common_prelabel_start"])
    grid[1][2] = _doy(censor["common_prelabel_end"])
    return grid


def _run_prelabel(tmp_path: Path, *, shifts=(0, 7, 14), exporter=None,
                  from_stage="plan", to_stage="prelabel-export", **kwargs) -> dict:
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    if exporter is None:
        exporter = _fake_exporter(_burned_grid(experiment_id, shifts))
    return wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(shifts), dry_run=False,
        from_stage=from_stage, to_stage=to_stage,
        output_root=out, experiments_root=experiments,
        prelabel_exporter=exporter, **kwargs,
    )


# --- 1, 2, 3. Stage lock -----------------------------------------------------
@pytest.mark.parametrize("from_stage,to_stage", [
    ("plan", "plan"),
    ("prelabel-export", "prelabel-export"),
    ("plan", "prelabel-export"),
])
def test_supported_actual_stage_ranges(tmp_path, from_stage, to_stage):
    wcs.assert_actual_stages_supported(wcs.validate_stage_range(from_stage, to_stage))


def test_prelabel_export_is_an_implemented_actual_stage():
    assert wcs.IMPLEMENTED_ACTUAL_STAGES == wcs.STAGES
    assert wcs.PRELABEL_STAGE in wcs.IMPLEMENTED_ACTUAL_STAGES


# Every declared stage is implemented; only a stage name this build does not
# know stays fail-fast locked, and it must refuse before anything is created.
@pytest.mark.parametrize("from_stage,to_stage", [
    ("prelabel-export", "some-future-stage"),
    ("plan", "some-future-stage"),
])
def test_an_unknown_stage_remains_locked_and_writes_nothing(tmp_path, from_stage, to_stage):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="not enabled"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage=from_stage, to_stage=to_stage,
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_exploding_exporter,
        )
    assert not out.exists(), "an unknown stage created a directory"


def test_resume_and_force_together_are_refused(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="mutually exclusive"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            resume=True, force=True,
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_exploding_exporter,
        )
    assert not out.exists()


# --- 4, 5, 6. Plan binding gates -- no Earth Engine call --------------------
@pytest.mark.parametrize("document", list(wcs.PLAN_BINDING_DOCUMENTS))
def test_a_missing_plan_document_stops_the_stage_before_any_export(tmp_path, document):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    (out / experiment_id / document).unlink()

    with pytest.raises(wcs.WindowClosureError, match="missing|Plan binding failed"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="prelabel-export", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_exploding_exporter,
        )
    assert not (out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif").exists()


def test_an_analysis_id_mismatch_stops_the_stage_before_any_export(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    # A different preregistered shift set means a different analysis_id.
    with pytest.raises(wcs.WindowClosureError, match="analysis_id"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
            from_stage="prelabel-export", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_exploding_exporter,
        )
    assert not (out / experiment_id / "prelabel_censor" / "censoring_summary.json").exists()


def test_a_frozen_hash_mismatch_stops_the_stage_before_any_export(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    inventory = wcs.frozen_input_inventory(experiment_id, experiments)
    Path(inventory["dem_slope"]["path"]).write_bytes(b"mutated-after-the-plan")

    with pytest.raises(wcs.WindowClosureError, match="dem_slope|analysis_id"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="prelabel-export", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_exploding_exporter,
        )
    assert not (out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif").exists()


def test_plan_binding_accepts_a_matching_plan(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    censor = _censor_interval(experiment_id)
    binding = wcs.assert_plan_binding(
        experiment_id, result["analysis_id"], [0, 7, 14], censor,
        wcs.frozen_input_inventory(experiment_id, experiments),
        wcs.plan_output_paths(experiment_id, result["variants"], out),
    )
    assert binding["bound_to_plan"] is True
    assert binding["preregistered_shifts_days"] == [0, 7, 14]


# --- 7, 8, 9. Only the dedicated path is written ----------------------------
def test_export_writes_only_the_dedicated_prelabel_outputs(tmp_path):
    result = _run_prelabel(tmp_path)
    experiment_id = result["experiment_id"]
    root = tmp_path / "diagnostics" / experiment_id

    assert _relative_files(root) == sorted(list(PLAN_DOCUMENT_KEYS) + [
        "prelabel_censor/censoring_summary.json",
        "prelabel_censor/prelabel_burndate.tif",
        "prelabel_censor/prelabel_export_checkpoint.json",
    ])
    assert (root / "prelabel_censor" / "prelabel_burndate.tif").is_file()


def test_canonical_label_and_step8a_sentinels_are_untouched(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    canonical_root = wcs.canonical_experiment_root(experiment_id, experiments)
    before = {p: p.read_bytes() for p in canonical_root.rglob("*") if p.is_file()}
    assert before, "the fixture must seed canonical inputs"

    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="prelabel-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)),
    )
    after = {p: p.read_bytes() for p in canonical_root.rglob("*") if p.is_file()}
    assert after == before, "the canonical namespace was modified"


def test_foreign_diagnostics_sentinels_are_untouched_by_the_export(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    sentinels = {
        out / "other_diagnostic" / "sentinel.json": b'{"other": true}',
        out / "other_diagnostic" / "labels.tif": b"foreign-raster",
    }
    for path, payload in sentinels.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="prelabel-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)),
    )
    for path, payload in sentinels.items():
        assert path.read_bytes() == payload, f"{path} was touched"


def test_the_existing_export_plan_document_is_only_read(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    plan_path = out / experiment_id / "prelabel_censor" / "export_plan.json"
    before = plan_path.read_bytes()

    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="prelabel-export", to_stage="prelabel-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)),
    )
    assert plan_path.read_bytes() == before


# --- 10, 11. Interval and date semantics ------------------------------------
def test_the_exact_planned_interval_reaches_the_exporter(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    censor = _censor_interval(experiment_id)
    calls: list[dict] = []
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="prelabel-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id), calls=calls),
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["pre_label_start"] == censor["common_prelabel_start"]
    assert call["pre_label_end"] == censor["common_prelabel_end"]
    assert call["experiment_id"] == experiment_id
    assert call["raw_out"] == out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif"

    plan = json.loads(
        (out / experiment_id / "prelabel_censor" / "export_plan.json").read_text()
    )
    assert call["pre_label_start"] == plan["common_prelabel_start"]
    assert call["pre_label_end"] == plan["common_prelabel_end"]


def test_month_alignment_mirrors_the_production_step6_helper():
    """The recorded EE bounds are the production ones, not a re-derivation."""
    from src.step6_validate_fire_relation import _mcd64a1_collection_query_bounds

    windows = [
        ("2021-05-18", "2021-07-27"), ("2022-08-15", "2022-09-30"),
        ("2022-12-15", "2023-01-20"), ("2020-01-01", "2020-12-31"),
        ("2019-02-28", "2019-03-01"), ("2024-11-30", "2024-12-01"),
    ]
    for start, end in windows:
        assert wcs.prelabel_collection_query_bounds(start, end) == \
            _mcd64a1_collection_query_bounds(start, end)


def test_the_interval_end_is_inclusive_and_is_not_off_by_one():
    semantics = wcs.prelabel_date_semantics("2021-05-18", "2021-07-27")
    assert semantics["requested_interval_start"] == "2021-05-18"
    assert semantics["requested_interval_end"] == "2021-07-27"
    assert semantics["interval_semantics"] == "inclusive_start_inclusive_end"
    # The EE filterDate window is the WIDER month-aligned one, end-exclusive...
    assert semantics["ee_filter_start"] == "2021-05-01"
    assert semantics["ee_filter_end"] == "2021-08-01"
    assert semantics["ee_filter_end_semantics"] == "exclusive"
    # ...so the last included date is the requested end itself, NOT end - 1.
    assert semantics["effective_last_included_date"] == "2021-07-27"
    assert semantics["effective_last_included_date"] != "2021-07-26"
    assert semantics["burndate_doy_range_inclusive"] == [_doy("2021-05-18"), _doy("2021-07-27")]


def test_a_burn_on_the_last_interval_day_is_kept_and_counted(tmp_path):
    """The inclusive end must not be silently dropped."""
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    censor = _censor_interval(experiment_id)
    grid = np.zeros(_TEST_SHAPE)
    grid[0][0] = _doy(censor["common_prelabel_end"])  # exactly the last day

    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="prelabel-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(grid),
    )
    assert result["prelabel_burn_cell_count"] == 1
    assert result["prelabel_censor"]["max_finite_burndate"] == \
        float(_doy(censor["common_prelabel_end"]))


def test_a_burn_after_the_interval_end_is_a_contract_failure(tmp_path):
    """One day past the inclusive end is out of window and must fail."""
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    censor = _censor_interval(experiment_id)
    grid = np.zeros(_TEST_SHAPE)
    grid[0][0] = _doy(censor["common_prelabel_end"]) + 1

    with pytest.raises(wcs.WindowClosureError, match="outside the requested window"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_fake_exporter(grid),
        )
    assert not (out / experiment_id / "prelabel_censor" / "censoring_summary.json").exists()


def test_a_burn_before_the_interval_start_is_a_contract_failure(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    censor = _censor_interval(experiment_id)
    grid = np.zeros(_TEST_SHAPE)
    grid[0][0] = _doy(censor["common_prelabel_start"]) - 1

    with pytest.raises(wcs.WindowClosureError, match="outside the requested window"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_fake_exporter(grid),
        )


def test_the_summary_records_the_full_date_semantics(tmp_path):
    result = _run_prelabel(tmp_path)
    summary = json.loads(
        (tmp_path / "diagnostics" / result["experiment_id"] / "prelabel_censor"
         / "censoring_summary.json").read_text()
    )
    semantics = summary["date_semantics"]
    for key in ("requested_interval_start", "requested_interval_end",
                "interval_semantics", "ee_filter_start", "ee_filter_end",
                "effective_last_included_date"):
        assert semantics.get(key), f"date semantics is missing '{key}'"
    assert semantics["effective_last_included_date"] == summary["common_prelabel_end"]


# --- 12, 13, 14. Raster contract --------------------------------------------
def test_the_exported_raster_grid_matches_the_reference(tmp_path):
    result = _run_prelabel(tmp_path)
    summary = result["prelabel_censor"]
    reference = wcs.read_grid_signature(Path(summary["reference_grid_path"]))

    assert summary["grid_matches_reference"] is True
    assert wcs.grid_signatures_match(summary["grid_signature"], reference)
    assert summary["grid_signature"]["band_count"] == 1
    assert summary["grid_signature"]["width"] > 0
    assert summary["grid_signature"]["height"] > 0
    assert summary["grid_signature"]["crs"]
    assert len(summary["grid_signature"]["transform"]) == 6
    assert summary["dtype"]
    assert summary["mask_semantics"]
    assert summary["reference_grid_role"] == wcs.LABEL_ROLE_RAW


@pytest.mark.parametrize("kwargs", [
    {"transform": Affine(0.001, 0.0, 31.05, 0.0, -0.001, 37.35)},   # wrong pixel size
    {"crs": "EPSG:3857"},                                            # wrong CRS
])
def test_a_grid_mismatch_fails_and_writes_no_summary(tmp_path, kwargs):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="grid does not match"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_fake_exporter(np.zeros(_TEST_SHAPE), **kwargs),
        )
    summary = out / experiment_id / "prelabel_censor" / "censoring_summary.json"
    assert not summary.exists(), "a failed stage wrote a scientific PASS summary"


def test_a_wrong_band_count_fails(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="band"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_fake_exporter(np.zeros(_TEST_SHAPE), bands=3),
        )
    assert not (out / experiment_id / "prelabel_censor" / "censoring_summary.json").exists()


def test_a_corrupt_raster_fails_and_writes_no_summary(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="could not be read"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_fake_exporter(corrupt=True),
        )
    assert not (out / experiment_id / "prelabel_censor" / "censoring_summary.json").exists()
    assert not (out / experiment_id / "prelabel_censor" / "prelabel_export_checkpoint.json").exists()


def test_a_missing_or_empty_raster_fails(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)

    def empty_exporter(experiment_id, pre_label_start, pre_label_end, raw_out):
        Path(raw_out).parent.mkdir(parents=True, exist_ok=True)
        Path(raw_out).write_bytes(b"")
        return {"raw_path": raw_out}

    with pytest.raises(wcs.WindowClosureError, match="empty"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=empty_exporter,
        )

    def absent_exporter(experiment_id, pre_label_start, pre_label_end, raw_out):
        return {"raw_path": raw_out}

    with pytest.raises(wcs.WindowClosureError, match="was not produced"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=absent_exporter, force=True,
        )


def test_a_binary_looking_raster_fails(tmp_path):
    """All-ones positives mean the BurnDate information was destroyed."""
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    censor = _censor_interval(experiment_id)
    if _doy(censor["common_prelabel_start"]) <= 1 <= _doy(censor["common_prelabel_end"]):
        pytest.skip("DOY 1 is inside this interval, so 1.0 is a legitimate value")
    grid = np.zeros(_TEST_SHAPE)
    grid[0][0] = 1
    with pytest.raises(wcs.WindowClosureError, match="outside the requested window|BINARY"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_fake_exporter(grid),
        )


# --- 15, 16, 17, 18. Counts and summary contract ----------------------------
def test_zero_prelabel_burns_is_a_valid_pass(tmp_path):
    result = _run_prelabel(tmp_path, exporter=_fake_exporter(np.zeros(_TEST_SHAPE)))
    summary = result["prelabel_censor"]

    assert result["status"] == "pass"
    assert summary["status"] == "pass"
    assert summary["prelabel_burn_cell_count"] == 0
    assert summary["zero_burn_is_a_valid_outcome"] is True
    assert summary["min_finite_burndate"] is None
    assert summary["max_finite_burndate"] is None
    assert summary["zero_or_unburned_cell_count"] == summary["finite_cell_count"]


def test_a_nonzero_burn_count_is_computed_correctly(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    censor = _censor_interval(experiment_id)
    start_doy, end_doy = (
        _doy(censor["common_prelabel_start"]), _doy(censor["common_prelabel_end"]),
    )
    middle = (start_doy + end_doy) // 2
    grid = np.zeros(_TEST_SHAPE)
    grid[0][0], grid[0][1], grid[2][3] = start_doy, middle, end_doy

    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="prelabel-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(grid),
    )
    summary = result["prelabel_censor"]
    assert summary["prelabel_burn_cell_count"] == 3
    assert summary["finite_cell_count"] == _TEST_SHAPE[0] * _TEST_SHAPE[1]
    assert summary["zero_or_unburned_cell_count"] == summary["finite_cell_count"] - 3
    assert summary["min_finite_burndate"] == float(start_doy)
    assert summary["max_finite_burndate"] == float(end_doy)
    assert result["prelabel_burn_cell_count"] == 3


def test_the_summary_carries_the_raster_hash_and_size(tmp_path):
    result = _run_prelabel(tmp_path)
    root = tmp_path / "diagnostics" / result["experiment_id"]
    raster = root / "prelabel_censor" / "prelabel_burndate.tif"
    summary = json.loads((root / "prelabel_censor" / "censoring_summary.json").read_text())

    assert summary["raster_sha256"] == _sha256(raster)
    assert summary["raster_bytes"] == raster.stat().st_size
    assert summary["raster_path"] == str(raster)
    assert result["raster_sha256"] == summary["raster_sha256"]


def test_the_summary_carries_the_full_required_contract(tmp_path):
    result = _run_prelabel(tmp_path)
    summary = json.loads(
        (tmp_path / "diagnostics" / result["experiment_id"] / "prelabel_censor"
         / "censoring_summary.json").read_text()
    )
    for key in ("schema_version", "analysis_id", "experiment_id",
                "common_prelabel_start", "common_prelabel_end", "date_semantics",
                "producer", "raster_path", "raster_sha256", "raster_bytes",
                "grid_signature", "grid_matches_reference", "finite_cell_count",
                "prelabel_burn_cell_count", "zero_or_unburned_cell_count",
                "min_finite_burndate", "max_finite_burndate", "gee_query_run",
                "gee_export_run", "canonical_outputs_modified",
                "frozen_hashes_unchanged", "status"):
        assert key in summary, f"censoring_summary.json is missing '{key}'"
    assert summary["analysis_id"] == result["analysis_id"]
    assert summary["canonical_outputs_modified"] is False
    assert summary["frozen_hashes_unchanged"] is True
    assert summary["canonical_gate_rerun"] is False
    assert summary["used_as_predictor"] is False
    assert summary["applies_to_all_variants"] is True
    assert "export_raw_mcd64a1_prelabel_labels" in summary["producer"]
    assert summary["gee_query_run"] is True and summary["gee_export_run"] is True


def test_the_producer_is_the_production_step6_helper():
    from src import step6_validate_fire_relation as step6

    assert wcs.PRELABEL_PRODUCER.endswith("export_raw_mcd64a1_prelabel_labels")
    assert hasattr(step6, "export_raw_mcd64a1_prelabel_labels")


# --- 19, 20. Resume ----------------------------------------------------------
def test_resume_reuses_a_valid_raster_without_exporting(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        output_root=out, experiments_root=experiments,
    )
    first = wcs.run_analysis(
        from_stage="plan", to_stage="prelabel-export",
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)), **common,
    )
    raster = out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif"
    before = raster.read_bytes()

    second = wcs.run_analysis(
        from_stage="prelabel-export", to_stage="prelabel-export", resume=True,
        prelabel_exporter=_exploding_exporter, **common,
    )
    assert second["reused"] is True
    assert second["gee_queries_run"] is False and second["gee_exports_run"] is False
    assert second["raster_sha256"] == first["raster_sha256"]
    assert raster.read_bytes() == before
    assert second["prelabel_censor"]["reused_existing_raster"] is True


def test_resume_does_not_reuse_a_corrupt_raster(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        output_root=out, experiments_root=experiments,
    )
    wcs.run_analysis(
        from_stage="plan", to_stage="prelabel-export",
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)), **common,
    )
    raster = out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif"
    raster.write_bytes(b"corrupted-on-disk")

    calls: list[dict] = []
    result = wcs.run_analysis(
        from_stage="prelabel-export", to_stage="prelabel-export", resume=True,
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id), calls=calls), **common,
    )
    assert len(calls) == 1, "a corrupt raster must be re-exported, not reused"
    assert result["reused"] is False
    assert result["gee_exports_run"] is True
    # The corrupt file was quarantined, never deleted.
    quarantine = out / experiment_id / "prelabel_censor" / "_quarantine"
    assert quarantine.is_dir()
    assert any(p.read_bytes() == b"corrupted-on-disk" for p in quarantine.iterdir())


def test_resume_does_not_reuse_an_analysis_id_mismatch(tmp_path):
    """A checkpoint from another analysis identity is never trusted."""
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="prelabel-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)),
    )
    checkpoint = out / experiment_id / "prelabel_censor" / "prelabel_export_checkpoint.json"
    payload = json.loads(checkpoint.read_text())
    assert wcs._checkpoint_is_valid(
        checkpoint, payload["analysis_id"],
        out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif",
    )
    assert not wcs._checkpoint_is_valid(
        checkpoint, "f" * 64,
        out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif",
    )


def test_a_second_plain_run_refuses_to_overwrite_the_raster(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        output_root=out, experiments_root=experiments,
    )
    wcs.run_analysis(
        from_stage="plan", to_stage="prelabel-export",
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)), **common,
    )
    raster = out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif"
    before = raster.read_bytes()

    with pytest.raises(wcs.WindowClosureError, match="already exists"):
        wcs.run_analysis(
            from_stage="prelabel-export", to_stage="prelabel-export",
            prelabel_exporter=_exploding_exporter, **common,
        )
    assert raster.read_bytes() == before


# --- 21, 22. Force ------------------------------------------------------------
def test_force_replaces_only_prelabel_owned_outputs(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        output_root=out, experiments_root=experiments,
    )
    first = wcs.run_analysis(
        from_stage="plan", to_stage="prelabel-export",
        prelabel_exporter=_fake_exporter(np.zeros(_TEST_SHAPE)), **common,
    )
    root = out / experiment_id
    plan_before = {
        p: p.read_bytes() for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)) in PLAN_DOCUMENT_KEYS
    }
    foreign = {
        root / "comparison" / "bootstrap_replicates.parquet": b"unmanaged",
        root / "variants" / "close_7d_earlier" / "step8a" / "d.parquet": b"artefact",
    }
    for path, payload in foreign.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    second = wcs.run_analysis(
        from_stage="prelabel-export", to_stage="prelabel-export", force=True,
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)), **common,
    )
    assert second["raster_sha256"] != first["raster_sha256"]
    assert second["prelabel_burn_cell_count"] == 2

    # Plan documents are untouched by a prelabel force.
    assert {p: p.read_bytes() for p in plan_before} == plan_before
    for path, payload in foreign.items():
        assert path.read_bytes() == payload, f"force modified {path}"
    # The replaced raster was quarantined, not deleted.
    quarantine = root / "prelabel_censor" / "_quarantine"
    assert quarantine.is_dir() and any(quarantine.iterdir())


def test_prelabel_owned_target_guard_refuses_anything_else(tmp_path):
    experiment_id = any_experiment()
    root = wcs.experiment_root(experiment_id, tmp_path / "out")
    with pytest.raises(wcs.WindowClosureError, match="not a prelabel-owned"):
        wcs.assert_prelabel_owned_targets(
            experiment_id,
            [root / "prelabel_censor" / "export_plan.json"],
            tmp_path / "out",
        )
    wcs.assert_prelabel_owned_targets(
        experiment_id,
        wcs.prelabel_output_paths(experiment_id, tmp_path / "out").values(),
        tmp_path / "out",
    )


# --- 23, 24, 25. Dry run, transparency, no model/bootstrap ------------------
def test_dry_run_of_the_prelabel_stage_touches_no_earth_engine(tmp_path):
    import builtins

    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    real_import = builtins.__import__
    touched: list[str] = []

    def guarded(name, *args, **kwargs):
        if name == "ee" or name.startswith("ee."):
            touched.append(name)
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=guarded):
        result = wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=True,
            from_stage="prelabel-export", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_exploding_exporter,
        )
    assert touched == [], f"dry-run imported Earth Engine: {touched}"
    assert result["gee_queries_run"] is False and result["gee_exports_run"] is False
    assert not out.exists()


def test_dry_run_carries_the_actual_plan_prerequisites_field(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    ready = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=True,
        output_root=out, experiments_root=experiments,
    )["actual_plan_prerequisites"]
    assert ready["ready"] is True
    assert ready["missing_required_inputs"] == []
    assert set(ready["required_roles"]) == {
        "canonical_step8a", "dem_elevation", "dem_slope", "landcover_aligned",
        "label_raw_burndate", "label_burned_binary",
    }

    Path(wcs.frozen_input_inventory(experiment_id, experiments)["dem_slope"]["path"]).unlink()
    not_ready = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=True,
        output_root=out, experiments_root=experiments,
    )
    field = not_ready["actual_plan_prerequisites"]
    assert field["ready"] is False
    assert {e["role"] for e in field["missing_required_inputs"]} == {"dem_slope"}
    # The pre-existing label-scoped fields keep their original meaning.
    assert not_ready["prerequisites_ready"] is True
    assert not_ready["missing_required_inputs"] == []


def test_the_prelabel_stage_never_fits_a_model_or_bootstraps(tmp_path):
    from sklearn.base import BaseEstimator

    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)

    def _boom(*_args, **_kwargs):
        raise AssertionError("the prelabel stage must not model or bootstrap")

    with patch.object(BaseEstimator, "fit", _boom, create=True), \
            patch.object(wcs, "multi_variant_block_bootstrap", _boom), \
            patch.object(wcs, "build_common_cohort", _boom):
        result = wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)),
        )
    assert result["model_fit"] is False and result["bootstrap_run"] is False
    assert result["prelabel_censor"]["model_fit"] is False
    assert result["prelabel_censor"]["bootstrap_run"] is False


# --- 27. Result contract ------------------------------------------------------
def test_prelabel_result_contract(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="prelabel-export", to_stage="prelabel-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)),
    )
    assert result["ran"] is True
    assert result["dry_run"] is False
    assert result["experiment_id"] == experiment_id
    assert result["stages_run"] == ["prelabel-export"]
    assert result["analysis_id"] and len(result["analysis_id"]) == 64
    assert result["files_written_count"] == 3
    assert result["reused"] is False
    assert result["gee_queries_run"] is True
    assert result["gee_exports_run"] is True
    assert result["model_fit"] is False
    assert result["bootstrap_run"] is False
    assert result["frozen_hashes_unchanged"] is True
    assert result["prelabel_burn_cell_count"] == 2
    assert result["raster_sha256"]
    assert result["status"] == "pass"
    assert result["canonical_outputs_modified"] is False


def test_plan_to_prelabel_export_reuses_the_plan_documents(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        output_root=out, experiments_root=experiments,
    )
    first = wcs.run_analysis(from_stage="plan", to_stage="plan", **common)
    root = out / experiment_id
    plan_before = {
        p: p.read_bytes() for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)) in PLAN_DOCUMENT_KEYS
    }

    second = wcs.run_analysis(
        from_stage="plan", to_stage="prelabel-export",
        prelabel_exporter=_fake_exporter(_burned_grid(experiment_id)), **common,
    )
    assert second["analysis_id"] == first["analysis_id"]
    assert second["stages_run"] == ["plan", "prelabel-export"]
    assert second["plan"]["reused"] is True
    assert second["plan"]["files_rewritten"] == []
    assert second["files_written_count"] == 10
    for path, payload in plan_before.items():
        assert path.read_bytes() == payload, f"a plan document was rewritten: {path}"


# --- 28. No hard-coded AOI or date -------------------------------------------
def test_no_aoi_name_or_calendar_date_is_hard_coded_in_the_implementation():
    import re

    for module_path in (
        _PROJECT_ROOT / "src" / "window_closure_sensitivity.py",
        _PROJECT_ROOT / "scripts" / "run_window_closure_sensitivity.py",
    ):
        literals = _executable_string_literals(module_path)
        for experiment_id in REGISTRY_IDS:
            assert not [s for s in literals if experiment_id in s], \
                f"{experiment_id} hard-coded in {module_path.name}"
        dated = [s for s in literals if re.search(r"(19|20)\d\d-\d\d-\d\d", s)]
        assert dated == [], f"calendar date hard-coded in {module_path.name}: {dated}"


# =============================================================================
# 131-190. Actual PREDICTOR-EXPORT stage
# =============================================================================
import scripts.validate_window_closure_predictor_export as validator  # noqa: E402

_MODIS_PIXEL_FACTOR = wcs.MODIS_EXPORT_SCALE_M / wcs.LANDSAT_EXPORT_SCALE_M
_MODIS_TRANSFORM = Affine(
    0.00026949458523585647 * _MODIS_PIXEL_FACTOR, 0.0, 31.05,
    0.0, -0.00026949458523585647 * _MODIS_PIXEL_FACTOR, 37.35,
)
_PREDICTOR_NODATA = -9999.0


def _family_transform(grid_family: str):
    return _MODIS_TRANSFORM if grid_family == wcs.GRID_FAMILY_MODIS else _TEST_TRANSFORM


def _write_predictor_raster(path: Path, job: dict, *, values=None, transform=None,
                            crs="EPSG:4326", bands: int = 1, dtype: str = "float32",
                            nodata=_PREDICTOR_NODATA, all_nodata: bool = False) -> Path:
    """A synthetic raster that satisfies (or deliberately breaks) the contract."""
    import rasterio

    if values is None:
        values = (
            np.full(_TEST_SHAPE, 3.0) if job["is_count_product"]
            else np.full(_TEST_SHAPE, 21.5)
        )
    array = np.asarray(values, dtype=dtype)
    if all_nodata:
        array = np.full(_TEST_SHAPE, nodata, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=bands, dtype=dtype, crs=crs,
        transform=transform if transform is not None else _family_transform(job["grid_family"]),
        nodata=nodata,
    ) as dst:
        for band in range(1, bands + 1):
            dst.write(array, band)
    return path


def _fake_predictor_engine(*, calls: list | None = None, fail_variants: Sequence[str] = (),
                           mutate=None, transport: str = "direct"):
    """Stand-in for the Earth Engine production engine. Never touches `ee`."""
    def engine(variant_context, variant, jobs):
        if calls is not None:
            calls.append({
                "variant_id": variant["variant_id"],
                "predictor_start_date": variant_context["predictor_start_date"],
                "predictor_end_date": variant_context["predictor_end_date"],
                "modis_input_dir": Path(variant_context["modis_input_dir"]),
                "namespace_allowed_roots": list(
                    variant_context.get(wcs.MODIS_NAMESPACE_ALLOWED_ROOTS_KEY) or []
                ),
                "jobs": list(jobs),
            })
        if variant["variant_id"] in fail_variants:
            raise wcs.WindowClosureError(
                f"synthetic engine failure for {variant['variant_id']}"
            )
        results = {}
        for job in jobs:
            path = Path(job["output_path"])
            kwargs = mutate(job) if mutate else {}
            if kwargs.get("skip"):
                continue
            _write_predictor_raster(path, job, **{
                k: v for k, v in kwargs.items() if k != "skip"
            })
            results[job["artifact_id"]] = {"path": path, "transport": transport}
        return results
    return engine


def _exploding_engine(*_args, **_kwargs):
    raise AssertionError("the predictor engine must not be reached")


def _predictor_env(tmp_path: Path, shifts=(0, 7, 14)) -> tuple[str, Path, Path]:
    """A namespace with the plan and the pre-label stages already completed."""
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(shifts), dry_run=False,
        from_stage="plan", to_stage="prelabel-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(np.zeros(_TEST_SHAPE)),
    )
    return experiment_id, out, experiments


def _run_predictor(tmp_path: Path, *, shifts=(0, 7, 14), engine=None, **kwargs):
    experiment_id, out, experiments = _predictor_env(tmp_path, shifts)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(shifts), dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        predictor_engine=engine if engine is not None else _fake_predictor_engine(),
        **kwargs,
    )
    return result, experiment_id, out, experiments


def _baseline_years(experiment_id: str) -> list[int]:
    return list(wcs.canonical_window(ctx_for(experiment_id))["baseline_years"])


# --- 1, 2, 3. Stage lock -----------------------------------------------------
def test_predictor_export_is_an_implemented_actual_stage():
    assert wcs.IMPLEMENTED_ACTUAL_STAGES == wcs.STAGES
    assert wcs.PREDICTOR_STAGE == "predictor-export"


@pytest.mark.parametrize("from_stage,to_stage", [
    ("predictor-export", "predictor-export"),
    ("plan", "predictor-export"),
    ("prelabel-export", "predictor-export"),
])
def test_predictor_stage_ranges_are_supported(from_stage, to_stage):
    wcs.assert_actual_stages_supported(wcs.validate_stage_range(from_stage, to_stage))


@pytest.mark.parametrize("from_stage,to_stage", [
    ("predictor-export", "some-future-stage"),
    ("local-downstream", "some-future-stage"),
    ("plan", "some-future-stage"),
])
def test_unknown_stages_remain_locked(tmp_path, from_stage, to_stage):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="not enabled"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage=from_stage, to_stage=to_stage,
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
            prelabel_exporter=_exploding_exporter,
        )
    assert not out.exists(), "a locked stage created a directory"


def test_unknown_stage_message_names_the_unrecognized_stage(tmp_path):
    """The lock message names the stage this build does not know -- and only it.

    `predictor-export -> compare` is a VALID range: compare is an implemented
    actual stage. So the lock wording is exercised with a stage that genuinely
    does not exist, and the message must point at that stage, not at compare.
    """
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError) as excinfo:
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="some-future-stage",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
            prelabel_exporter=_exploding_exporter,
        )
    message = str(excinfo.value)
    assert "some-future-stage" in message
    assert "not enabled" in message
    assert "not enabled or recognized" in message

    # `compare` is an implemented actual stage, so it may appear ONLY as a known
    # stage in the enumeration -- never as a locked or unrecognized one.
    assert wcs.COMPARE_STAGE in wcs.IMPLEMENTED_ACTUAL_STAGES
    assert f"Known stages are {list(wcs.STAGES)}" in message
    assert f"stage {wcs.COMPARE_STAGE!r} is not enabled" not in message
    assert f"stage(s) ['{wcs.COMPARE_STAGE}']" not in message

    # The range is rejected before any engine, exporter, mkdir or write.
    assert not out.exists(), "an unknown stage created a directory"


def test_predictor_resume_and_force_together_are_refused(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="mutually exclusive"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            resume=True, force=True,
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )


# --- 4, 5, 6, 7. Binding gates (never reach the engine) ---------------------
@pytest.mark.parametrize("document", list(wcs.PREDICTOR_BINDING_DOCUMENTS))
def test_a_missing_binding_document_stops_before_the_engine(tmp_path, document):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    (out / experiment_id / document).unlink()

    with pytest.raises(wcs.WindowClosureError):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )
    for variant_id in ("close_7d_earlier", "close_14d_earlier"):
        assert not (out / experiment_id / "variants" / variant_id / "data").exists()


def test_a_missing_prelabel_raster_stops_before_the_engine(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    (out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif").unlink()

    with pytest.raises(wcs.WindowClosureError, match="pre-label raster is missing"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )


def test_a_tampered_prelabel_raster_stops_before_the_engine(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    raster = out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif"
    raster.write_bytes(raster.read_bytes() + b"tampered")

    with pytest.raises(wcs.WindowClosureError, match="hash differs"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )


def test_a_failed_prelabel_status_stops_before_the_engine(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    summary_path = out / experiment_id / "prelabel_censor" / "censoring_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "fail"
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(wcs.WindowClosureError, match="status"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )


def test_an_analysis_id_mismatch_stops_before_the_engine(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path, shifts=(0, 7, 14))
    with pytest.raises(wcs.WindowClosureError, match="analysis_id"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )
    assert not (out / experiment_id / "variants" / "close_7d_earlier" / "data").exists()


def test_a_frozen_hash_mismatch_stops_before_the_engine(tmp_path):
    """A moved frozen input is caught by the identity itself.

    Every frozen input hash feeds the analysis_id, so mutating one makes the
    derived identity disagree with the preregistration on disk -- the binding
    refuses before the engine is reached, whichever check fires first.
    """
    experiment_id, out, experiments = _predictor_env(tmp_path)
    inventory = wcs.frozen_input_inventory(experiment_id, experiments)
    Path(inventory["dem_elevation"]["path"]).write_bytes(b"mutated-after-the-plan")

    with pytest.raises(wcs.WindowClosureError, match="analysis_id|dem_elevation|hash"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )


def test_a_tampered_canonical_frozen_reference_stops_the_stage(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    path = out / experiment_id / "variants" / "canonical" / "frozen_reference.json"
    document = json.loads(path.read_text())
    document["predictor_export_planned"] = True
    path.write_text(json.dumps(document))

    with pytest.raises(wcs.WindowClosureError, match="canonical variant must never be exported"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )


# --- 8, 9, 10, 11. Canonical isolation --------------------------------------
def test_the_canonical_variant_never_enters_the_work_queue(tmp_path):
    calls: list[dict] = []
    result, experiment_id, out, _ = _run_predictor(
        tmp_path, engine=_fake_predictor_engine(calls=calls),
    )
    seen = {call["variant_id"] for call in calls}
    assert wcs.CANONICAL_VARIANT_ID not in seen
    assert result["processed_variants"] == ["close_7d_earlier", "close_14d_earlier"]
    assert result["canonical_export_attempted"] is False


def test_no_canonical_data_directory_is_created(tmp_path):
    _, experiment_id, out, _ = _run_predictor(tmp_path)
    canonical_dir = out / experiment_id / "variants" / wcs.CANONICAL_VARIANT_ID
    assert _relative_files(canonical_dir) == ["frozen_reference.json"]
    assert not (canonical_dir / "data").exists()


def test_the_canonical_frozen_reference_is_untouched(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    path = out / experiment_id / "variants" / wcs.CANONICAL_VARIANT_ID / "frozen_reference.json"
    before = path.read_bytes()
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        predictor_engine=_fake_predictor_engine(),
    )
    assert path.read_bytes() == before


def test_canonical_variant_has_no_predictor_jobs():
    experiment_id = any_experiment()
    ctx = ctx_for(experiment_id)
    canonical = next(v for v in wcs.build_window_variants(ctx, [0, 7]) if v["is_canonical"])
    with pytest.raises(wcs.WindowClosureError, match="no predictor export"):
        wcs.predictor_artifact_jobs(experiment_id, canonical, [2019], 56)


def test_plan_and_prelabel_documents_are_untouched_by_the_predictor_stage(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    root = out / experiment_id
    watched = sorted(list(PLAN_DOCUMENT_KEYS) + [
        "prelabel_censor/censoring_summary.json",
        "prelabel_censor/prelabel_burndate.tif",
        "prelabel_censor/prelabel_export_checkpoint.json",
    ])
    before = {name: (root / name).read_bytes() for name in watched}

    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        predictor_engine=_fake_predictor_engine(),
    )
    for name, payload in before.items():
        assert (root / name).read_bytes() == payload, f"{name} was modified"


def test_the_canonical_experiment_namespace_is_untouched(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    canonical_root = wcs.canonical_experiment_root(experiment_id, experiments)
    before = {p: p.read_bytes() for p in canonical_root.rglob("*") if p.is_file()}

    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        predictor_engine=_fake_predictor_engine(),
    )
    after = {p: p.read_bytes() for p in canonical_root.rglob("*") if p.is_file()}
    assert after == before


# --- 12-18. Variant and date contract ---------------------------------------
@pytest.mark.parametrize("shifts", [(0, 7, 14), (0, 3, 9, 21), (0, 5)])
def test_every_nonzero_shift_produces_a_variant_dynamically(tmp_path, shifts):
    result, experiment_id, out, _ = _run_predictor(tmp_path, shifts=shifts)
    expected = [wcs.variant_id(s) for s in sorted(set(shifts)) if s != 0]
    assert result["processed_variants"] == expected
    for variant_id in expected:
        assert (out / experiment_id / "variants" / variant_id
                / wcs.PREDICTOR_METADATA_NAME).is_file()


def test_shift_input_order_is_deterministic(tmp_path):
    first, experiment_id, out, experiments = _run_predictor(tmp_path, shifts=(0, 7, 14))
    assert first["processed_variants"] == ["close_7d_earlier", "close_14d_earlier"]

    variants = wcs.build_window_variants(ctx_for(experiment_id), [14, 0, 7, 7])
    assert [v["variant_id"] for v in wcs.nonzero_variants(variants)] == \
        ["close_7d_earlier", "close_14d_earlier"]


def test_current_dates_match_the_preregistration(tmp_path):
    """The current window spans BOTH predictor families, so its jobs are
    counted per family rather than as one number: 4 Landsat rasters (2 logical
    roles x 2 production products) and 3 MODIS rasters, 7 in total. A single
    aggregate count would let a change in one family hide inside the other."""
    calls: list[dict] = []
    result, experiment_id, out, _ = _run_predictor(
        tmp_path, engine=_fake_predictor_engine(calls=calls),
    )
    by_id = {v["variant_id"]: v for v in result["variants"]}
    assert calls, "the engine must run for every non-canonical variant"

    for call in calls:
        variant = by_id[call["variant_id"]]
        assert call["predictor_start_date"] == variant["predictor_start_date"]
        assert call["predictor_end_date"] == variant["predictor_end_date"]

        current = [j for j in call["jobs"] if j["scope"] == "current_window"]
        current_landsat = [j for j in current if j["family"] in ("lst", "ndvi")]
        current_modis = [j for j in current if j["family"] == "modis"]

        assert len(current_landsat) == 4
        assert len(current_modis) == 3
        assert len(current) == len(current_landsat) + len(current_modis) == 7

        assert {j["role"] for j in current_landsat} == {"current_lst", "current_ndvi"}
        assert {j["product"] for j in current_landsat} == {
            "scene_weighted_median", "scene_valid_count",
        } == set(wcs.LANDSAT_PRODUCTS_PER_ROLE)
        assert {j["role"] for j in current_modis} == {
            "modis_lst_mean", "modis_lst_std", "modis_valid_observation_count",
        } == set(wcs.MODIS_ROLE_ORDER)

        # Both families evaluate over the SAME preregistered variant window.
        for job in current_landsat:
            assert job["start_date"] == variant["predictor_start_date"]
            assert job["end_date"] == variant["predictor_end_date"]
        for job in current_modis:
            assert job["start_date"] == variant["predictor_start_date"]
            assert job["end_date"] == variant["predictor_end_date"]


def test_baseline_years_and_dates_follow_the_variant(tmp_path):
    calls: list[dict] = []
    result, experiment_id, out, _ = _run_predictor(
        tmp_path, engine=_fake_predictor_engine(calls=calls),
    )
    years = _baseline_years(experiment_id)
    by_id = {v["variant_id"]: v for v in result["variants"]}
    for call in calls:
        variant = by_id[call["variant_id"]]
        baseline = [j for j in call["jobs"] if j["scope"] == "baseline_year"]
        assert sorted({j["baseline_year"] for j in baseline}) == sorted(years)
        for job in baseline:
            year = job["baseline_year"]
            assert job["start_date"] == _parse(
                variant["predictor_start_date"]).replace(year=year).strftime("%Y-%m-%d")
            assert job["end_date"] == _parse(
                variant["predictor_end_date"]).replace(year=year).strftime("%Y-%m-%d")


def test_lead_days_are_carried_into_the_metadata(tmp_path):
    result, experiment_id, out, _ = _run_predictor(tmp_path)
    by_id = {v["variant_id"]: v for v in result["variants"]}
    for variant_id in result["processed_variants"]:
        metadata = json.loads(
            (out / experiment_id / "variants" / variant_id
             / wcs.PREDICTOR_METADATA_NAME).read_text()
        )
        assert metadata["lead_days"] == by_id[variant_id]["lead_days"]
        assert metadata["shift_days"] == by_id[variant_id]["shift_days"]


def test_landsat_date_semantics_do_not_add_a_silent_day():
    semantics = wcs.landsat_job_date_semantics("2021-05-25", "2021-07-20")
    assert semantics["requested_start_date"] == "2021-05-25"
    assert semantics["requested_end_date"] == "2021-07-20"
    assert semantics["ee_filter_start"] == "2021-05-25"
    assert semantics["ee_filter_end"] == "2021-07-20"
    assert semantics["ee_filter_end_semantics"] == "exclusive"
    assert semantics["effective_last_included_date"] == "2021-07-19"
    assert semantics["duration_days"] == 56


def test_modis_date_semantics_report_the_fixed_production_month_filter():
    from core.config import SUMMER_MONTH_END, SUMMER_MONTH_START

    semantics = wcs.modis_job_date_semantics("2021-05-18", "2021-07-13")
    assert semantics["ee_filter_end_semantics"] == "exclusive"
    transparency = semantics["calendar_month_filter_transparency"]
    assert transparency["calendar_month_filter"] == f"{SUMMER_MONTH_START}-{SUMMER_MONTH_END}"
    assert transparency["calendar_month_filter_is_fixed"] is True
    # May days fall outside the fixed production summer months and ARE clipped.
    assert transparency["calendar_month_filter_clips_window"] is True
    assert transparency["clipped_day_count"] > 0
    assert transparency["effective_first_included_date"].startswith("2021-06")


def test_a_window_inside_the_summer_months_is_not_clipped():
    transparency = wcs.modis_month_filter_transparency("2021-06-01", "2021-07-27")
    assert transparency["calendar_month_filter_clips_window"] is False
    assert transparency["clipped_day_count"] == 0


def test_no_aoi_or_date_is_hard_coded_in_the_predictor_implementation():
    import re

    for module_path in (
        _PROJECT_ROOT / "src" / "window_closure_sensitivity.py",
        _PROJECT_ROOT / "scripts" / "validate_window_closure_predictor_export.py",
    ):
        literals = _executable_string_literals(module_path)
        for experiment_id in REGISTRY_IDS:
            assert not [s for s in literals if experiment_id in s]
        assert [s for s in literals if re.search(r"(19|20)\d\d-\d\d-\d\d", s)] == []


# --- 19-26. Landsat contract -------------------------------------------------
def _jobs_for(experiment_id: str, shift: int = 7, output_root: Optional[Path] = None):
    ctx = ctx_for(experiment_id)
    canonical = wcs.canonical_window(ctx)
    variant = next(
        v for v in wcs.build_window_variants(ctx, [0, shift]) if v["shift_days"] == shift
    )
    return variant, canonical, wcs.predictor_artifact_jobs(
        experiment_id, variant, canonical["baseline_years"],
        canonical["current_period_days"], output_root,
    )


def test_each_variant_has_ten_landsat_logical_roles(tmp_path):
    experiment_id = any_experiment()
    _, canonical, jobs = _jobs_for(experiment_id, 7, tmp_path)
    landsat = [j for j in jobs if j["family"] in ("lst", "ndvi")]
    roles = {j["role"] for j in landsat}
    assert len(roles) == 2 + 2 * len(canonical["baseline_years"])
    assert {"current_lst", "current_ndvi"} <= roles


def test_each_landsat_role_has_both_production_products(tmp_path):
    experiment_id = any_experiment()
    _, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    by_role: dict[str, set] = {}
    for job in (j for j in jobs if j["family"] in ("lst", "ndvi")):
        by_role.setdefault(job["role"], set()).add(job["product"])
    for role, products in by_role.items():
        assert products == set(wcs.LANDSAT_PRODUCTS_PER_ROLE), role


def test_only_scene_weighted_products_are_planned(tmp_path):
    experiment_id = any_experiment()
    _, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    products = {j["product"] for j in jobs if j["family"] in ("lst", "ndvi")}
    assert products == {"scene_weighted_median", "scene_valid_count"}
    assert "date_balanced" not in json.dumps(jobs, default=str)


def test_date_balanced_products_are_rejected(tmp_path):
    experiment_id = any_experiment()
    variant, canonical, jobs = _jobs_for(experiment_id, 7, tmp_path)
    poisoned = [dict(jobs[0], product="date_balanced_median")] + jobs[1:]
    with pytest.raises(wcs.WindowClosureError):
        wcs.assert_predictor_job_set(poisoned, canonical["baseline_years"], variant)


def test_landsat_bands_are_the_production_ones(tmp_path):
    experiment_id = any_experiment()
    _, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    bands = {(j["role"], j["product"]): j["band"] for j in jobs if j["band"]}
    assert bands[("current_lst", "scene_weighted_median")] == "Current_Period_LST_Celsius"
    assert bands[("current_lst", "scene_valid_count")] == "Current_Period_Valid_Count"
    assert bands[("current_ndvi", "scene_weighted_median")] == "Current_Period_NDVI"
    assert bands[("current_ndvi", "scene_valid_count")] == "Current_Period_NDVI_Valid_Count"
    baseline_lst = [j for j in jobs if j["role"].startswith("baseline_lst")]
    assert {j["band"] for j in baseline_lst} == {"ST_B10", "Baseline_Window_Valid_Count"}
    baseline_ndvi = [j for j in jobs if j["role"].startswith("baseline_ndvi")]
    assert {j["band"] for j in baseline_ndvi} == {"NDVI", "Baseline_Window_NDVI_Valid_Count"}


def test_production_band_names_exist_in_step3():
    """The band names are the production ones, not invented here."""
    source = (_PROJECT_ROOT / "src" / "step3_landsat_lst.py").read_text(encoding="utf-8")
    for bands in wcs.LANDSAT_ROLE_BANDS.values():
        for band in bands.values():
            assert f'"{band}"' in source, f"{band} is not a production band name"


def test_landsat_output_paths_are_inside_the_variant_namespace(tmp_path):
    experiment_id = any_experiment()
    variant, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    vroot = wcs.variant_root(experiment_id, variant["variant_id"], tmp_path).resolve()
    for job in jobs:
        assert Path(job["output_path"]).resolve().is_relative_to(vroot)
        assert Path(job["output_path"]).suffix == ".tif"


def test_a_path_escaping_the_variant_namespace_is_refused(tmp_path):
    experiment_id = any_experiment()
    variant, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    escaped = [dict(jobs[0], output_path=str(tmp_path / "elsewhere.tif"))] + jobs[1:]
    with pytest.raises(wcs.WindowClosureError, match="escapes the variant namespace"):
        wcs.assert_jobs_inside_variant_namespace(
            experiment_id, variant["variant_id"], escaped, tmp_path,
        )


def test_a_path_pointing_at_the_canonical_variant_is_refused(tmp_path):
    experiment_id = any_experiment()
    variant, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    canonical_path = (
        wcs.variant_root(experiment_id, wcs.CANONICAL_VARIANT_ID, tmp_path)
        / "data" / "leak.tif"
    )
    leaked = [dict(jobs[0], output_path=str(canonical_path))] + jobs[1:]
    with pytest.raises(wcs.WindowClosureError, match="CANONICAL variant namespace"):
        wcs.assert_jobs_inside_variant_namespace(
            experiment_id, variant["variant_id"], leaked, tmp_path,
        )


# --- 27-31. MODIS contract ---------------------------------------------------
def test_each_variant_has_three_modis_products(tmp_path):
    experiment_id = any_experiment()
    _, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    modis = [j for j in jobs if j["family"] == "modis"]
    assert len(modis) == 3
    assert {j["role"] for j in modis} == set(wcs.MODIS_ROLE_ORDER)
    assert {Path(j["output_path"]).name for j in modis} == set(wcs.MODIS_ROLE_FILENAMES.values())


def test_modis_uses_the_exact_variant_dates(tmp_path):
    experiment_id = any_experiment()
    variant, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    for job in (j for j in jobs if j["family"] == "modis"):
        assert job["start_date"] == variant["predictor_start_date"]
        assert job["end_date"] == variant["predictor_end_date"]
        assert job["uses_variant_context"] is True


def test_modis_producer_is_the_production_function(tmp_path):
    experiment_id = any_experiment()
    _, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    for job in (j for j in jobs if j["family"] == "modis"):
        assert "prepare_modis_for_step7" in job["producer"]


def test_modis_export_scale_mirrors_production():
    from scripts.prepare_modis_for_step7 import MODIS_EXPORT_SCALE

    assert wcs.MODIS_EXPORT_SCALE_M == MODIS_EXPORT_SCALE


def test_the_variant_context_never_points_modis_at_the_canonical_namespace(tmp_path):
    calls: list[dict] = []
    _, experiment_id, out, _ = _run_predictor(
        tmp_path, engine=_fake_predictor_engine(calls=calls),
    )
    for call in calls:
        vroot = wcs.variant_root(experiment_id, call["variant_id"], out).resolve()
        assert call["modis_input_dir"].resolve().is_relative_to(vroot)
        assert call["namespace_allowed_roots"], "the MODIS guard opt-in is missing"
        for root in call["namespace_allowed_roots"]:
            assert Path(root).resolve() == vroot


def test_the_modis_namespace_guard_accepts_only_safe_extra_roots(tmp_path):
    from scripts.prepare_modis_for_step7 import (
        NAMESPACE_ALLOWED_ROOTS_KEY, ModisPrepError, _resolve_allowed_output_roots,
    )

    # The opt-in key must not drift between the caller and the production guard.
    assert wcs.MODIS_NAMESPACE_ALLOWED_ROOTS_KEY == NAMESPACE_ALLOWED_ROOTS_KEY

    experiment_id = any_experiment()
    base = {"experiment_id": experiment_id}
    # Default behaviour is unchanged when the opt-in is absent.
    assert len(_resolve_allowed_output_roots(base)) == 1

    diagnostics = _PROJECT_ROOT / "outputs" / "diagnostics" / "window_closure_sensitivity"
    assert len(_resolve_allowed_output_roots(
        {**base, "namespace_allowed_roots": [diagnostics]}
    )) == 2

    with pytest.raises(ModisPrepError, match="outputs/ dışında|outside"):
        _resolve_allowed_output_roots({**base, "namespace_allowed_roots": ["/tmp/elsewhere"]})
    with pytest.raises(ModisPrepError, match="canonical"):
        _resolve_allowed_output_roots({
            **base,
            "namespace_allowed_roots": [_PROJECT_ROOT / "outputs" / "experiments" / "other"],
        })


# --- 32-37. Artifact counts ---------------------------------------------------
def test_role_and_raster_counts_for_four_baseline_years():
    years = [2017, 2018, 2019, 2020]
    assert wcs.expected_logical_role_count(years) == 13
    assert wcs.expected_raster_count(years) == 23
    assert 2 * wcs.expected_raster_count(years) == 46


def test_planned_counts_match_the_generic_formula(tmp_path):
    experiment_id = any_experiment()
    _, canonical, jobs = _jobs_for(experiment_id, 7, tmp_path)
    years = canonical["baseline_years"]
    assert len({j["role"] for j in jobs}) == wcs.expected_logical_role_count(years)
    assert len(jobs) == wcs.expected_raster_count(years)


def test_two_nonzero_variants_plan_double_the_rasters(tmp_path):
    experiment_id, out, _ = _predictor_env(tmp_path)
    ctx = ctx_for(experiment_id)
    canonical = wcs.canonical_window(ctx)
    summary = wcs.predictor_export_summary(
        experiment_id, wcs.build_window_variants(ctx, [0, 7, 14]),
        canonical["baseline_years"], canonical["current_period_days"], out,
    )
    assert summary["total_planned_rasters"] == 2 * summary["rasters_per_variant"]
    assert summary["logical_roles_per_variant"] == \
        wcs.expected_logical_role_count(canonical["baseline_years"])


def test_a_duplicate_role_fails(tmp_path):
    experiment_id = any_experiment()
    variant, canonical, jobs = _jobs_for(experiment_id, 7, tmp_path)
    with pytest.raises(wcs.WindowClosureError, match="duplicate artefact"):
        wcs.assert_predictor_job_set(
            jobs + [dict(jobs[0])], canonical["baseline_years"], variant,
        )


def test_a_duplicate_output_path_fails(tmp_path):
    experiment_id = any_experiment()
    variant, canonical, jobs = _jobs_for(experiment_id, 7, tmp_path)
    clashing = list(jobs)
    clashing[1] = dict(clashing[1], output_path=clashing[0]["output_path"])
    with pytest.raises(wcs.WindowClosureError, match="duplicate output path"):
        wcs.assert_predictor_job_set(clashing, canonical["baseline_years"], variant)


def test_a_missing_role_fails(tmp_path):
    experiment_id = any_experiment()
    variant, canonical, jobs = _jobs_for(experiment_id, 7, tmp_path)
    without_current = [j for j in jobs if j["role"] != "current_ndvi"]
    with pytest.raises(wcs.WindowClosureError, match="missing predictor role"):
        wcs.assert_predictor_job_set(without_current, canonical["baseline_years"], variant)


def test_an_extra_forbidden_role_fails(tmp_path):
    experiment_id = any_experiment()
    variant, canonical, jobs = _jobs_for(experiment_id, 7, tmp_path)
    extra = jobs + [dict(jobs[0], role="prelabel_burndate", artifact_id="prelabel_burndate")]
    with pytest.raises(wcs.WindowClosureError, match="forbidden/unexpected"):
        wcs.assert_predictor_job_set(extra, canonical["baseline_years"], variant)


# --- 45-54. Raster contract ---------------------------------------------------
def _one_job(tmp_path, *, family: str = "lst") -> dict:
    experiment_id = any_experiment()
    _, _, jobs = _jobs_for(experiment_id, 7, tmp_path)
    if family == "count":
        return next(j for j in jobs if j["product"] == "scene_valid_count")
    if family == "modis":
        return next(j for j in jobs if j["family"] == "modis")
    return next(j for j in jobs if j["product"] == "scene_weighted_median")


def _reference_signature(tmp_path) -> dict:
    path = tmp_path / "reference.tif"
    _write_grid_raster(path, np.zeros(_TEST_SHAPE))
    return wcs.read_grid_signature(path)


def test_a_valid_predictor_raster_passes(tmp_path):
    job = _one_job(tmp_path)
    path = _write_predictor_raster(tmp_path / "valid.tif", job)
    record = wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))

    assert record["grid_contract_passed"] is True
    assert record["band_count"] == 1
    assert record["finite_cell_count"] == _TEST_SHAPE[0] * _TEST_SHAPE[1]
    assert record["sha256"] == _sha256(path)
    assert record["min_finite"] is not None and record["max_finite"] is not None
    assert record["dtype"] and record["nodata"] is not None
    assert record["mask_semantics"]


def test_an_empty_predictor_raster_fails(tmp_path):
    job = _one_job(tmp_path)
    path = tmp_path / "empty.tif"
    path.write_bytes(b"")
    with pytest.raises(wcs.WindowClosureError, match="empty"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_an_unreadable_predictor_raster_fails(tmp_path):
    job = _one_job(tmp_path)
    path = tmp_path / "corrupt.tif"
    path.write_bytes(b"not-a-geotiff-at-all")
    with pytest.raises(wcs.WindowClosureError, match="could not be read"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_a_missing_crs_fails(tmp_path):
    job = _one_job(tmp_path)
    path = _write_predictor_raster(tmp_path / "nocrs.tif", job, crs=None)
    with pytest.raises(wcs.WindowClosureError, match="no CRS"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_a_wrong_crs_fails(tmp_path):
    job = _one_job(tmp_path)
    path = _write_predictor_raster(tmp_path / "wrongcrs.tif", job, crs="EPSG:3857")
    with pytest.raises(wcs.WindowClosureError, match="CRS"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_a_wrong_band_count_fails(tmp_path):
    job = _one_job(tmp_path)
    path = _write_predictor_raster(tmp_path / "threeband.tif", job, bands=3)
    with pytest.raises(wcs.WindowClosureError, match="band"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_a_grid_family_mismatch_fails(tmp_path):
    """A Landsat artefact on the MODIS grid must not pass."""
    job = _one_job(tmp_path)
    path = _write_predictor_raster(
        tmp_path / "wronggrid.tif", job, transform=_MODIS_TRANSFORM,
    )
    with pytest.raises(wcs.WindowClosureError, match="production grid contract"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_a_modis_artifact_must_use_the_modis_grid(tmp_path):
    job = _one_job(tmp_path, family="modis")
    good = _write_predictor_raster(tmp_path / "modis_ok.tif", job)
    wcs.inspect_predictor_raster(good, job, _reference_signature(tmp_path))

    bad = _write_predictor_raster(
        tmp_path / "modis_bad.tif", job, transform=_TEST_TRANSFORM,
    )
    with pytest.raises(wcs.WindowClosureError, match="production grid contract"):
        wcs.inspect_predictor_raster(bad, job, _reference_signature(tmp_path))


def test_a_negative_count_value_fails(tmp_path):
    job = _one_job(tmp_path, family="count")
    values = np.full(_TEST_SHAPE, 2.0)
    values[0][0] = -1.0
    path = _write_predictor_raster(tmp_path / "negative.tif", job, values=values)
    with pytest.raises(wcs.WindowClosureError, match="negative"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_a_fractional_count_value_fails(tmp_path):
    job = _one_job(tmp_path, family="count")
    values = np.full(_TEST_SHAPE, 2.0)
    values[0][0] = 2.5
    path = _write_predictor_raster(tmp_path / "fractional.tif", job, values=values)
    with pytest.raises(wcs.WindowClosureError, match="fractional"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_an_all_zero_count_raster_is_allowed(tmp_path):
    """Zero observations everywhere is a real result, not a contract failure."""
    job = _one_job(tmp_path, family="count")
    path = _write_predictor_raster(
        tmp_path / "zerocount.tif", job, values=np.zeros(_TEST_SHAPE),
    )
    record = wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))
    assert record["min_finite"] == 0.0 and record["max_finite"] == 0.0


def test_an_entirely_nodata_scientific_raster_fails(tmp_path):
    job = _one_job(tmp_path)
    path = _write_predictor_raster(tmp_path / "allnodata.tif", job, all_nodata=True)
    with pytest.raises(wcs.WindowClosureError, match="entirely nodata"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_a_non_finite_value_fails(tmp_path):
    job = _one_job(tmp_path)
    values = np.full(_TEST_SHAPE, 20.0)
    values[0][0] = np.inf
    path = _write_predictor_raster(tmp_path / "inf.tif", job, values=values)
    with pytest.raises(wcs.WindowClosureError, match="non-finite"):
        wcs.inspect_predictor_raster(path, job, _reference_signature(tmp_path))


def test_variant_support_is_never_forced_to_be_equal(tmp_path):
    """Different observation support across variants is expected, not an error."""
    def varying(job):
        if job["is_count_product"]:
            return {"values": np.full(_TEST_SHAPE, 7.0)}
        return {}

    result, experiment_id, out, _ = _run_predictor(
        tmp_path, engine=_fake_predictor_engine(mutate=lambda job: varying(job)),
    )
    assert result["status"] == "pass"


# --- 55-61. Metadata ----------------------------------------------------------
def _metadata_for(out: Path, experiment_id: str, variant_id: str) -> dict:
    return json.loads(
        (out / experiment_id / "variants" / variant_id
         / wcs.PREDICTOR_METADATA_NAME).read_text()
    )


def test_metadata_carries_the_full_required_contract(tmp_path):
    result, experiment_id, out, _ = _run_predictor(tmp_path)
    years = _baseline_years(experiment_id)
    for variant_id in result["processed_variants"]:
        metadata = _metadata_for(out, experiment_id, variant_id)
        for key in (
            "schema_version", "analysis_id", "experiment_id", "variant_id",
            "shift_days", "predictor_start_date", "predictor_end_date",
            "lead_days", "baseline_years", "reducer", "production_policy",
            "expected_logical_role_count", "produced_logical_role_count",
            "expected_raster_count", "produced_raster_count", "landsat_current",
            "landsat_baselines", "modis", "artifact_inventory", "artifact_sha256",
            "raster_contract_passed", "all_paths_inside_variant_namespace",
            "canonical_export_attempted", "prelabel_used_as_predictor",
            "gee_queries_run", "gee_exports_run", "frozen_input_sha256_before",
            "frozen_input_sha256_after", "frozen_hashes_unchanged",
            "canonical_outputs_modified", "status",
        ):
            assert key in metadata, f"{variant_id} metadata is missing '{key}'"
        assert metadata["schema_version"] == "window_closure_predictor_export.v1"
        assert metadata["analysis_id"] == result["analysis_id"]
        assert metadata["reducer"] == "scene_weighted"
        assert metadata["canonical_export_attempted"] is False
        assert metadata["prelabel_used_as_predictor"] is False
        assert metadata["canonical_outputs_modified"] is False
        assert metadata["frozen_hashes_unchanged"] is True
        assert metadata["raster_contract_passed"] is True
        assert metadata["status"] == "pass"
        assert metadata["produced_raster_count"] == wcs.expected_raster_count(years)
        assert metadata["produced_logical_role_count"] == \
            wcs.expected_logical_role_count(years)


def test_metadata_artifact_inventory_is_complete_and_hashed(tmp_path):
    result, experiment_id, out, _ = _run_predictor(tmp_path)
    years = _baseline_years(experiment_id)
    for variant_id in result["processed_variants"]:
        metadata = _metadata_for(out, experiment_id, variant_id)
        inventory = metadata["artifact_inventory"]
        assert len(inventory) == wcs.expected_raster_count(years)
        for record in inventory:
            for key in ("role", "family", "scope", "baseline_year", "product",
                        "path", "sha256", "bytes", "band_count", "dtype",
                        "nodata", "width", "height", "crs", "transform",
                        "grid_signature", "finite_cell_count", "min_finite",
                        "max_finite", "export_transport"):
                assert key in record, f"artifact record is missing '{key}'"
            assert record["sha256"] == _sha256(Path(record["path"]))
            assert metadata["artifact_sha256"][record["artifact_id"]] == record["sha256"]
        assert len(metadata["artifact_sha256"]) == len(inventory)


def test_metadata_is_deterministically_sorted(tmp_path):
    result, experiment_id, out, _ = _run_predictor(tmp_path)
    for variant_id in result["processed_variants"]:
        path = (out / experiment_id / "variants" / variant_id
                / wcs.PREDICTOR_METADATA_NAME)
        payload = json.loads(path.read_text())
        assert list(payload) == sorted(payload)
        ids = [a["artifact_id"] for a in payload["artifact_inventory"]]
        assert ids == sorted(ids)


def test_no_temporary_metadata_file_is_left_behind(tmp_path):
    result, experiment_id, out, _ = _run_predictor(tmp_path)
    root = out / experiment_id
    leftovers = [p for p in root.rglob("*") if p.is_file() and p.suffix == ".tmp"]
    assert leftovers == []


def test_a_failing_variant_writes_no_pass_metadata(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    with pytest.raises(wcs.WindowClosureError):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_fake_predictor_engine(fail_variants=("close_7d_earlier",)),
        )
    metadata = (out / experiment_id / "variants" / "close_7d_earlier"
                / wcs.PREDICTOR_METADATA_NAME)
    assert not metadata.exists() or json.loads(metadata.read_text())["status"] != "pass"


def test_a_variant_whose_raster_fails_the_contract_writes_no_pass_metadata(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)

    def broken(job):
        if job["artifact_id"].startswith("current_lst__scene_weighted_median"):
            return {"all_nodata": True}
        return {}

    with pytest.raises(wcs.WindowClosureError, match="entirely nodata"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_fake_predictor_engine(mutate=broken),
        )
    metadata = (out / experiment_id / "variants" / "close_7d_earlier"
                / wcs.PREDICTOR_METADATA_NAME)
    assert not metadata.exists() or json.loads(metadata.read_text())["status"] != "pass"


# --- 62-68. Resume / force ----------------------------------------------------
def test_resume_reuses_a_complete_variant_without_exporting(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    first = wcs.run_analysis(predictor_engine=_fake_predictor_engine(), **common)
    before = {
        p: p.read_bytes() for p in (out / experiment_id / "variants").rglob("*")
        if p.is_file()
    }

    second = wcs.run_analysis(resume=True, predictor_engine=_exploding_engine, **common)

    assert second["reused_variants"] == first["processed_variants"]
    assert second["exported_variants"] == []
    assert second["reused"] is True
    assert second["gee_queries_run"] is False and second["gee_exports_run"] is False
    after = {
        p: p.read_bytes() for p in (out / experiment_id / "variants").rglob("*")
        if p.is_file()
    }
    assert after == before


def test_resume_does_not_reuse_a_hash_mismatch(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    wcs.run_analysis(predictor_engine=_fake_predictor_engine(), **common)
    metadata = _metadata_for(out, experiment_id, "close_7d_earlier")
    victim = Path(metadata["artifact_inventory"][0]["path"])
    victim.write_bytes(victim.read_bytes() + b"tampered")

    calls: list[dict] = []
    result = wcs.run_analysis(
        resume=True, predictor_engine=_fake_predictor_engine(calls=calls), **common,
    )
    assert "close_7d_earlier" in result["exported_variants"]
    assert "close_7d_earlier" not in result["reused_variants"]
    assert any(call["variant_id"] == "close_7d_earlier" for call in calls)


def test_resume_does_not_reuse_a_missing_artifact(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    wcs.run_analysis(predictor_engine=_fake_predictor_engine(), **common)
    metadata = _metadata_for(out, experiment_id, "close_14d_earlier")
    Path(metadata["artifact_inventory"][0]["path"]).unlink()

    result = wcs.run_analysis(
        resume=True, predictor_engine=_fake_predictor_engine(), **common,
    )
    assert "close_14d_earlier" in result["exported_variants"]


def test_a_plain_rerun_refuses_to_overwrite(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    wcs.run_analysis(predictor_engine=_fake_predictor_engine(), **common)
    with pytest.raises(wcs.WindowClosureError, match="already has a complete"):
        wcs.run_analysis(predictor_engine=_exploding_engine, **common)


def test_force_replaces_only_variant_owned_predictor_files(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    first = wcs.run_analysis(predictor_engine=_fake_predictor_engine(), **common)
    root = out / experiment_id
    protected = {
        p: p.read_bytes() for p in root.rglob("*")
        if p.is_file() and (
            str(p.relative_to(root)) in PLAN_DOCUMENT_KEYS
            or str(p.relative_to(root)).startswith("prelabel_censor/")
        )
    }
    unmanaged = root / "comparison" / "unrelated.parquet"
    unmanaged.parent.mkdir(parents=True, exist_ok=True)
    unmanaged.write_bytes(b"unmanaged")

    second = wcs.run_analysis(
        force=True,
        predictor_engine=_fake_predictor_engine(
            mutate=lambda job: {"values": np.full(_TEST_SHAPE, 9.0)}
        ),
        **common,
    )
    assert second["exported_variants"] == first["processed_variants"]
    for path, payload in protected.items():
        assert path.read_bytes() == payload, f"force modified {path}"
    assert unmanaged.read_bytes() == b"unmanaged"

    quarantine = root / "variants" / "close_7d_earlier" / wcs.PREDICTOR_QUARANTINE_DIR
    assert quarantine.is_dir() and any(quarantine.iterdir())


def test_force_quarantines_instead_of_deleting(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    first = wcs.run_analysis(predictor_engine=_fake_predictor_engine(), **common)
    original = {
        record["artifact_id"]: Path(record["path"]).read_bytes()
        for record in _metadata_for(out, experiment_id, "close_7d_earlier")["artifact_inventory"]
    }
    wcs.run_analysis(
        force=True,
        predictor_engine=_fake_predictor_engine(
            mutate=lambda job: {"values": np.full(_TEST_SHAPE, 9.0)}
        ),
        **common,
    )
    quarantine = out / experiment_id / "variants" / "close_7d_earlier" / wcs.PREDICTOR_QUARANTINE_DIR
    quarantined = {p.read_bytes() for p in quarantine.iterdir() if p.is_file()}
    assert len(quarantined) > 0
    assert quarantined <= set(original.values()), "quarantine must hold the OLD bytes"


def test_a_second_variant_failure_preserves_the_first_variant(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    common = dict(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    with pytest.raises(wcs.WindowClosureError, match="synthetic engine failure"):
        wcs.run_analysis(
            predictor_engine=_fake_predictor_engine(fail_variants=("close_14d_earlier",)),
            **common,
        )
    first_metadata = (out / experiment_id / "variants" / "close_7d_earlier"
                      / wcs.PREDICTOR_METADATA_NAME)
    assert first_metadata.is_file()
    payload = json.loads(first_metadata.read_text())
    assert payload["status"] == "pass"
    first_bytes = {
        Path(r["path"]): Path(r["path"]).read_bytes()
        for r in payload["artifact_inventory"]
    }

    second_metadata = (out / experiment_id / "variants" / "close_14d_earlier"
                       / wcs.PREDICTOR_METADATA_NAME)
    assert not second_metadata.exists() or \
        json.loads(second_metadata.read_text())["status"] != "pass"

    # ...and a later resume reuses the surviving first variant untouched.
    result = wcs.run_analysis(
        resume=True, predictor_engine=_fake_predictor_engine(), **common,
    )
    assert "close_7d_earlier" in result["reused_variants"]
    assert "close_14d_earlier" in result["exported_variants"]
    for path, payload_bytes in first_bytes.items():
        assert path.read_bytes() == payload_bytes


# --- 69-74. Dry run -----------------------------------------------------------
def test_predictor_dry_run_never_imports_earth_engine(tmp_path):
    import builtins

    experiment_id, out, experiments = _predictor_env(tmp_path)
    real_import = builtins.__import__
    touched: list[str] = []

    def guarded(name, *args, **kwargs):
        if name == "ee" or name.startswith("ee."):
            touched.append(name)
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=guarded):
        result = wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=True,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )
    assert touched == [], f"dry-run imported Earth Engine: {touched}"
    assert result["planned_stages"] == ["predictor-export"]


def test_predictor_dry_run_writes_nothing(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    root = out / experiment_id
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=True,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        predictor_engine=_exploding_engine,
    )
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert after == before
    assert result["files_written"] is False
    for variant_id in ("close_7d_earlier", "close_14d_earlier"):
        assert not (root / "variants" / variant_id / "data").exists()
        assert not (root / "variants" / variant_id / wcs.PREDICTOR_METADATA_NAME).exists()


def test_predictor_dry_run_summary_contract(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=True,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    summary = result["predictor_export_summary"]
    years = _baseline_years(experiment_id)

    assert summary["canonical_export_enabled"] is False
    assert summary["nonzero_variant_ids"] == ["close_7d_earlier", "close_14d_earlier"]
    assert summary["logical_roles_per_variant"] == wcs.expected_logical_role_count(years)
    assert summary["rasters_per_variant"] == wcs.expected_raster_count(years)
    assert summary["total_planned_rasters"] == 2 * wcs.expected_raster_count(years)
    assert summary["reducer"] == "scene_weighted"
    assert summary["forbidden_products_present"] is False
    assert summary["all_paths_inside_dedicated_namespace"] is True

    canonical_plan = summary["variant_plans"][wcs.CANONICAL_VARIANT_ID]
    assert canonical_plan["export_enabled"] is False
    assert canonical_plan["frozen_reference_only"] is True
    assert canonical_plan["expected_raster_count"] == 0

    for variant_id in summary["nonzero_variant_ids"]:
        plan = summary["variant_plans"][variant_id]
        assert plan["export_enabled"] is True
        assert plan["expected_raster_count"] == wcs.expected_raster_count(years)
        assert plan["expected_logical_role_count"] == wcs.expected_logical_role_count(years)
        assert plan["baseline_years"] == years
        assert plan["modis_date_semantics"]["requested_start_date"] == \
            plan["predictor_start_date"]
        assert plan["landsat_date_semantics"]


def test_predictor_dry_run_reports_no_model_or_bootstrap(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=True,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    for flag in ("model_fit", "bootstrap_run", "gee_queries_run", "gee_exports_run"):
        assert result[flag] is False


# --- 89, 90. Result contract --------------------------------------------------
def test_predictor_result_contract(tmp_path):
    result, experiment_id, out, _ = _run_predictor(tmp_path)
    years = _baseline_years(experiment_id)

    assert result["ran"] is True
    assert result["dry_run"] is False
    assert result["experiment_id"] == experiment_id
    assert result["stages_run"] == ["predictor-export"]
    assert len(result["analysis_id"]) == 64
    assert result["processed_variants"] == ["close_7d_earlier", "close_14d_earlier"]
    assert result["exported_variants"] == result["processed_variants"]
    assert result["reused_variants"] == []
    assert result["logical_roles_produced"] == 2 * wcs.expected_logical_role_count(years)
    assert result["predictor_rasters_produced"] == 2 * wcs.expected_raster_count(years)
    assert result["gee_queries_run"] is True
    assert result["gee_exports_run"] is True
    assert result["model_fit"] is False
    assert result["bootstrap_run"] is False
    assert result["canonical_export_attempted"] is False
    assert result["frozen_hashes_unchanged"] is True
    assert result["status"] == "pass"
    assert result["files_written"]


def test_the_predictor_stage_never_fits_a_model_or_bootstraps(tmp_path):
    from sklearn.base import BaseEstimator

    experiment_id, out, experiments = _predictor_env(tmp_path)

    def _boom(*_args, **_kwargs):
        raise AssertionError("the predictor stage must not model or bootstrap")

    with patch.object(BaseEstimator, "fit", _boom, create=True), \
            patch.object(wcs, "multi_variant_block_bootstrap", _boom), \
            patch.object(wcs, "build_common_cohort", _boom):
        result = wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_fake_predictor_engine(),
        )
    assert result["model_fit"] is False and result["bootstrap_run"] is False


def test_the_prelabel_raster_is_never_a_predictor(tmp_path):
    result, experiment_id, out, _ = _run_predictor(tmp_path)
    prelabel = (out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif").resolve()
    for variant_id in result["processed_variants"]:
        metadata = _metadata_for(out, experiment_id, variant_id)
        paths = {Path(r["path"]).resolve() for r in metadata["artifact_inventory"]}
        assert prelabel not in paths
        assert metadata["prelabel_used_as_predictor"] is False


# --- Supported multi-stage ranges (plan/prelabel -> predictor-export) --------
def test_plan_to_predictor_export_is_a_supported_end_to_end_range(tmp_path):
    """`plan -> predictor-export` runs all three implemented stages for real,
    with BOTH production exporters replaced by injected fakes."""
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(np.zeros(_TEST_SHAPE)),
        predictor_engine=_fake_predictor_engine(),
    )
    assert result["stages_run"] == ["plan", "prelabel-export", "predictor-export"]
    assert result["status"] == "pass"
    assert result["processed_variants"] == ["close_7d_earlier", "close_14d_earlier"]
    for variant_id in result["processed_variants"]:
        assert (out / experiment_id / "variants" / variant_id
                / wcs.PREDICTOR_METADATA_NAME).is_file()


def test_prelabel_to_predictor_export_is_a_supported_range(tmp_path):
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="plan",
        output_root=out, experiments_root=experiments,
    )
    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="prelabel-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(np.zeros(_TEST_SHAPE)),
        predictor_engine=_fake_predictor_engine(),
    )
    assert result["stages_run"] == ["prelabel-export", "predictor-export"]
    assert result["status"] == "pass"
    assert result["exported_variants"] == ["close_7d_earlier", "close_14d_earlier"]


def test_supported_multi_stage_ranges_never_import_earth_engine(tmp_path):
    import builtins

    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    real_import = builtins.__import__
    touched: list[str] = []

    def guarded(name, *args, **kwargs):
        if name == "ee" or name.startswith("ee."):
            touched.append(name)
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=guarded):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            prelabel_exporter=_fake_exporter(np.zeros(_TEST_SHAPE)),
            predictor_engine=_fake_predictor_engine(),
        )
    assert touched == [], f"a supported stage range imported Earth Engine: {touched}"


# --- Fail-closed guard: production exporters can never be reached silently ---
def test_the_guard_fails_a_run_that_reaches_the_production_prelabel_exporter(tmp_path):
    """A SUPPORTED range without an injected fake must fail the test loudly
    (the regression was a real Earth Engine download from a unit test)."""
    experiment_id, out, experiments = _seed_prelabel_env(tmp_path)
    with pytest.raises(AssertionError, match="fail-closed guard"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="plan", to_stage="prelabel-export",
            output_root=out, experiments_root=experiments,
        )


def test_the_guard_fails_a_run_that_reaches_the_production_predictor_engine(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    with pytest.raises(AssertionError, match="fail-closed guard"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
        )


def test_the_guard_blocks_every_production_export_entry_point(tmp_path):
    import core.gee_utils as gee_utils
    import scripts.prepare_modis_for_step7 as prepare_modis
    import scripts.run_predictors_only as run_predictors_only
    import src.step6_validate_fire_relation as step6

    calls = (
        lambda: step6.export_raw_mcd64a1_prelabel_labels(
            experiment_id="any", pre_label_start="2021-01-01",
            pre_label_end="2021-01-31", raw_out=tmp_path / "x.tif",
        ),
        lambda: gee_utils.init_gee("any-project"),
        lambda: run_predictors_only.export_image_direct_or_tiled(
            image=None, out_path=tmp_path / "y.tif", region=None, scale=30,
            crs="EPSG:4326", label="guard", force=True, tiles_dir=tmp_path,
        ),
        lambda: prepare_modis.prepare_modis_for_step7({}, force=True),
        lambda: wcs.production_predictor_engine({}, {}, []),
        lambda: sys.modules["geemap"].ee_export_image(
            object(), filename=str(tmp_path / "z.tif"),
        ),
    )
    for produce in calls:
        with pytest.raises(AssertionError, match="fail-closed guard"):
            produce()


# --- date_balanced: structural product audit, never a raw substring ----------
def test_collect_actual_landsat_products_reads_only_semantic_product_fields():
    plan = {
        "landsat": {
            "current_roles": [
                {"products": ["scene_weighted_median", "scene_valid_count"]},
            ],
            "baseline_roles": [{"products": ["scene_weighted_median"]}],
        },
        "forbidden_products": ["date_balanced_median"],
        "note": "date_balanced_minus_scene_weighted is forbidden in this plan",
        "limitations": ["date_balanced products belong to the reducer counterfactual"],
    }
    products = wcs.collect_actual_landsat_products(plan)
    assert sorted(set(products)) == ["scene_valid_count", "scene_weighted_median"]
    assert wcs.landsat_product_violations(plan) == []


def test_structural_validator_accepts_only_production_scene_weighted_products():
    good = {"landsat": {"current_roles": [
        {"products": list(wcs.PRODUCTION_LANDSAT_PRODUCTS)},
    ]}}
    assert wcs.landsat_product_violations(good) == []

    for bad_product in ("date_balanced_median", "date_balanced_minus_scene_weighted",
                        "date_balanced_v2_anything"):
        bad = {"landsat": {"current_roles": [{"products": [bad_product]}]}}
        violations = wcs.landsat_product_violations(bad)
        assert violations and "date_balanced" in violations[0]

    unknown = {"landsat": {"current_roles": [{"products": ["median_composite"]}]}}
    assert any("non-production" in v for v in wcs.landsat_product_violations(unknown))


def test_structural_validator_flags_artifact_product_fields():
    plan = {"expected_artifacts": [
        {"family": "lst", "product": "date_balanced_median"},
    ]}
    violations = wcs.landsat_product_violations(plan)
    assert violations and "date_balanced_median" in violations[0]

    inventory_doc = {"artifact_inventory": [
        {"family": "ndvi", "product": "date_balanced_minus_scene_weighted"},
    ]}
    violations = wcs.landsat_product_violations(inventory_doc)
    assert violations and "date_balanced" in violations[0]


def test_binding_passes_when_date_balanced_appears_only_in_documentation(tmp_path):
    """`forbidden_products`/note fields may NAME the banned products. The old
    whole-document substring check failed every valid plan for exactly this
    reason: the generated reducer_note legitimately mentions date_balanced."""
    experiment_id, out, experiments = _predictor_env(tmp_path)
    plan_path = out / experiment_id / "variants" / "close_7d_earlier" / "export_plan.json"
    assert "date_balanced" in plan_path.read_text(encoding="utf-8").lower()

    document = json.loads(plan_path.read_text(encoding="utf-8"))
    document["forbidden_products"] = list(wcs.FORBIDDEN_LANDSAT_PRODUCTS)
    document["note"] = "date_balanced_median is explicitly forbidden in this analysis"
    plan_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    result = wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        predictor_engine=_fake_predictor_engine(),
    )
    assert result["status"] == "pass"


def test_binding_fails_when_a_real_products_list_carries_date_balanced(tmp_path):
    experiment_id, out, experiments = _predictor_env(tmp_path)
    plan_path = out / experiment_id / "variants" / "close_7d_earlier" / "export_plan.json"
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    document["landsat"]["current_roles"][0]["products"] = [
        "scene_weighted_median", "date_balanced_median",
    ]
    plan_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(wcs.WindowClosureError, match="date_balanced_median"):
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
            from_stage="predictor-export", to_stage="predictor-export",
            output_root=out, experiments_root=experiments,
            predictor_engine=_exploding_engine,
        )
    for variant_id in ("close_7d_earlier", "close_14d_earlier"):
        assert not (out / experiment_id / "variants" / variant_id / "data").exists()


# =============================================================================
# 75-86. Validator
# =============================================================================
def _dry_run_payload(tmp_path: Path, shifts=(0, 7, 14)) -> tuple[dict, str, Path, Path]:
    experiment_id, out, experiments = _predictor_env(tmp_path, shifts)
    payload = wcs.run_analysis(
        experiment_id=experiment_id, shifts=list(shifts), dry_run=True,
        from_stage="predictor-export", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
    )
    return payload, experiment_id, out, experiments


def _write_log(path: Path, payload: dict) -> Path:
    """A realistic log: logger noise around the printed JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "INFO [window-closure-sensitivity] starting {not json}\n"
        + json.dumps(payload, indent=2, default=str)
        + "\nINFO [window-closure-sensitivity] tamamlandı: ran=False\n",
        encoding="utf-8",
    )
    return path


def _validate(mode: str, experiment_id: str, out: Path, *, log: Optional[Path] = None,
              shifts=(0, 7, 14), experiments: Optional[Path] = None) -> int:
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


def test_validator_accepts_a_valid_dry_run_log(tmp_path, capsys):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    log = _write_log(tmp_path / "dryrun.log", payload)

    exit_code = _validate("dry-run", experiment_id, out, log=log)
    captured = capsys.readouterr().out

    assert exit_code == 0, captured
    assert "OVERALL STATUS: PASS" in captured
    assert "TECHNICAL STATUS: PASS" in captured
    assert "SCIENTIFIC-CONTRACT STATUS: PASS" in captured
    assert "NAMESPACE / PROVENANCE SAFETY: PASS" in captured
    assert "[FAIL]" not in captured
    assert "[PASS] canonical variant is export-disabled" in captured
    assert "[PASS] no date_balanced product detected" in captured


def test_validator_parses_json_surrounded_by_log_noise(tmp_path):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    text = (
        "2026-07-30 10:00:00 INFO  starting\n{ not really json\n"
        + json.dumps(payload, default=str)
        + "\n2026-07-30 10:00:01 INFO  done\n"
    )
    parsed = validator.parse_json_object_from_log(text)
    assert parsed is not None
    assert parsed["analysis_id"] == payload["analysis_id"]


def test_validator_rejects_a_canonical_export_enabled_log(tmp_path, capsys):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    payload["predictor_export_summary"]["canonical_export_enabled"] = True
    payload["predictor_export_summary"]["variant_plans"]["canonical"]["export_enabled"] = True
    log = _write_log(tmp_path / "bad.log", payload)

    assert _validate("dry-run", experiment_id, out, log=log) == 1
    captured = capsys.readouterr().out
    assert "[FAIL] canonical variant is export-disabled" in captured
    assert "OVERALL STATUS: FAIL" in captured


def test_validator_rejects_a_missing_variant(tmp_path, capsys):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    summary = payload["predictor_export_summary"]
    summary["nonzero_variant_ids"] = ["close_7d_earlier"]
    summary["variant_plans"].pop("close_14d_earlier")
    log = _write_log(tmp_path / "bad.log", payload)

    assert _validate("dry-run", experiment_id, out, log=log) == 1
    assert "[FAIL] every preregistered non-zero variant is planned" in capsys.readouterr().out


def test_validator_rejects_a_wrong_current_date(tmp_path, capsys):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    plan = payload["predictor_export_summary"]["variant_plans"]["close_7d_earlier"]
    for artifact in plan["expected_artifacts"]:
        if artifact["scope"] == "current_window":
            artifact["end_date"] = "1999-01-01"
    log = _write_log(tmp_path / "bad.log", payload)

    assert _validate("dry-run", experiment_id, out, log=log) == 1
    assert "[FAIL] close_7d_earlier current Landsat dates" in capsys.readouterr().out


def test_validator_rejects_a_wrong_baseline_date(tmp_path, capsys):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    plan = payload["predictor_export_summary"]["variant_plans"]["close_14d_earlier"]
    for artifact in plan["expected_artifacts"]:
        if artifact["scope"] == "baseline_year":
            artifact["start_date"] = "2016-01-01"
            break
    log = _write_log(tmp_path / "bad.log", payload)

    assert _validate("dry-run", experiment_id, out, log=log) == 1
    assert "[FAIL] close_14d_earlier baseline dates" in capsys.readouterr().out


def test_validator_rejects_a_wrong_modis_date(tmp_path, capsys):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    plan = payload["predictor_export_summary"]["variant_plans"]["close_7d_earlier"]
    for artifact in plan["expected_artifacts"]:
        if artifact["family"] == "modis":
            artifact["start_date"] = "2020-01-01"
    log = _write_log(tmp_path / "bad.log", payload)

    assert _validate("dry-run", experiment_id, out, log=log) == 1
    assert "[FAIL] close_7d_earlier MODIS dates use the variant context" in capsys.readouterr().out


def test_validator_rejects_a_date_balanced_product(tmp_path, capsys):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    plan = payload["predictor_export_summary"]["variant_plans"]["close_7d_earlier"]
    plan["expected_artifacts"][0]["product"] = "date_balanced_median"
    log = _write_log(tmp_path / "bad.log", payload)

    assert _validate("dry-run", experiment_id, out, log=log) == 1
    assert "[FAIL] close_7d_earlier contains no date_balanced product" in capsys.readouterr().out


def test_validator_rejects_a_path_outside_the_namespace(tmp_path, capsys):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    plan = payload["predictor_export_summary"]["variant_plans"]["close_7d_earlier"]
    plan["expected_artifacts"][0]["output_path"] = "/tmp/escaped.tif"
    log = _write_log(tmp_path / "bad.log", payload)

    assert _validate("dry-run", experiment_id, out, log=log) == 1
    assert "[FAIL] close_7d_earlier output paths are contained" in capsys.readouterr().out


def test_validator_rejects_a_dry_run_that_wrote_files(tmp_path, capsys):
    payload, experiment_id, out, _ = _dry_run_payload(tmp_path)
    payload["files_written"] = True
    log = _write_log(tmp_path / "bad.log", payload)

    assert _validate("dry-run", experiment_id, out, log=log) == 1
    assert "[FAIL] no dry-run file writes detected" in capsys.readouterr().out


def test_validator_rejects_a_drifted_frozen_hash(tmp_path, capsys):
    payload, experiment_id, out, experiments = _dry_run_payload(tmp_path)
    log = _write_log(tmp_path / "log.log", payload)
    Path(payload["frozen_input_inventory"]["dem_slope"]["path"]).write_bytes(b"drifted")

    assert _validate("dry-run", experiment_id, out, log=log) == 1
    assert "[FAIL] frozen hashes are unchanged" in capsys.readouterr().out


def test_validator_rejects_a_missing_log(tmp_path, capsys):
    _, experiment_id, out, _ = _dry_run_payload(tmp_path)
    assert _validate("dry-run", experiment_id, out, log=tmp_path / "absent.log") == 1
    assert "[FAIL] dry-run log exists" in capsys.readouterr().out


# --- Actual mode --------------------------------------------------------------
def test_validator_accepts_a_valid_actual_export(tmp_path, capsys):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    exit_code = _validate("actual", experiment_id, out, experiments=experiments)
    captured = capsys.readouterr().out

    assert exit_code == 0, captured
    assert "OVERALL STATUS: PASS" in captured
    assert "[FAIL]" not in captured


def test_validator_actual_rejects_missing_metadata(tmp_path, capsys):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    (out / experiment_id / "variants" / "close_14d_earlier"
     / wcs.PREDICTOR_METADATA_NAME).unlink()

    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] close_14d_earlier predictor metadata exists" in capsys.readouterr().out


def test_validator_actual_rejects_a_missing_raster(tmp_path, capsys):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    metadata = _metadata_for(out, experiment_id, "close_7d_earlier")
    Path(metadata["artifact_inventory"][0]["path"]).unlink()

    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] close_7d_earlier every recorded raster exists" in capsys.readouterr().out


def test_validator_actual_rejects_a_hash_mismatch(tmp_path, capsys):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    metadata = _metadata_for(out, experiment_id, "close_7d_earlier")
    victim = Path(metadata["artifact_inventory"][0]["path"])
    victim.write_bytes(victim.read_bytes() + b"tampered")

    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] close_7d_earlier every raster hash matches" in capsys.readouterr().out


def test_validator_actual_rejects_the_prelabel_raster_in_the_inventory(tmp_path, capsys):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    metadata_path = (out / experiment_id / "variants" / "close_7d_earlier"
                     / wcs.PREDICTOR_METADATA_NAME)
    metadata = json.loads(metadata_path.read_text())
    prelabel = out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif"
    metadata["artifact_inventory"][0]["path"] = str(prelabel)
    metadata["artifact_inventory"][0]["sha256"] = _sha256(prelabel)
    metadata["artifact_sha256"][metadata["artifact_inventory"][0]["artifact_id"]] = \
        _sha256(prelabel)
    metadata_path.write_text(json.dumps(metadata))

    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "does not carry the pre-label raster as a predictor" in capsys.readouterr().out


def test_validator_actual_rejects_a_stray_canonical_predictor_file(tmp_path, capsys):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    stray = (out / experiment_id / "variants" / wcs.CANONICAL_VARIANT_ID
             / "data" / "current_period" / "leak.tif")
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"leak")

    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] no new predictor file exists under the canonical variant" in \
        capsys.readouterr().out


def test_validator_actual_rejects_an_incomplete_role_set(tmp_path, capsys):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    metadata_path = (out / experiment_id / "variants" / "close_7d_earlier"
                     / wcs.PREDICTOR_METADATA_NAME)
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact_inventory"] = metadata["artifact_inventory"][:-1]
    metadata_path.write_text(json.dumps(metadata))

    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    captured = capsys.readouterr().out
    assert "[FAIL] close_7d_earlier has" in captured


# --- Required vs optional frozen roles ---------------------------------------
def test_required_frozen_hash_roles_are_the_identity_roles_plus_the_prelabel():
    assert wcs.REQUIRED_PREDICTOR_FROZEN_HASH_ROLES == (
        wcs.REQUIRED_FROZEN_INPUT_ROLES + (wcs.PRELABEL_FROZEN_ROLE,)
    )
    assert set(wcs.REQUIRED_PREDICTOR_FROZEN_HASH_ROLES) == {
        "canonical_step8a", "dem_elevation", "dem_slope", "landcover_aligned",
        "label_raw_burndate", "label_burned_binary", "prelabel_burndate",
    }
    # The resolver side-file is convenience metadata, never an identity input.
    assert "canonical_step8a_stats" not in wcs.REQUIRED_PREDICTOR_FROZEN_HASH_ROLES
    assert "canonical_step8a_stats" not in wcs.REQUIRED_FROZEN_INPUT_ROLES


def test_missing_required_frozen_hashes_ignores_optional_roles():
    inventory = {
        role: {"path": f"/x/{role}", "exists": True, "sha256": "a" * 64}
        for role in wcs.REQUIRED_PREDICTOR_FROZEN_HASH_ROLES
    }
    inventory["canonical_step8a_stats"] = {
        "path": "/x/stats.json", "exists": False, "sha256": None,
    }
    assert wcs.missing_required_frozen_hashes(inventory) == []
    assert wcs.optional_frozen_roles(inventory) == ["canonical_step8a_stats"]

    inventory["dem_slope"] = {"path": "/x/dem", "exists": False, "sha256": None}
    assert wcs.missing_required_frozen_hashes(inventory) == ["dem_slope"]
    del inventory["label_burned_binary"]
    assert wcs.missing_required_frozen_hashes(inventory) == [
        "dem_slope", "label_burned_binary",
    ]


def test_validator_actual_separates_required_and_optional_frozen_roles(tmp_path, capsys):
    """The optional Step8A stats side-file is absent in this fixture, and that
    must not be reported as an incomplete frozen identity."""
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    stats = Path(
        wcs.frozen_input_inventory(experiment_id, experiments)["canonical_step8a_stats"]["path"]
    )
    assert not stats.exists(), "the fixture must leave the optional role unhashed"

    exit_code = _validate("actual", experiment_id, out, experiments=experiments)
    captured = capsys.readouterr().out

    assert exit_code == 0, captured
    assert "[PASS] all required frozen identity inputs exist and hash" in captured
    assert "[PASS] optional metadata roles do not alter the frozen identity" in captured
    assert "NAMESPACE / PROVENANCE SAFETY: PASS" in captured
    assert "OVERALL STATUS: PASS" in captured
    assert "canonical_step8a_stats" not in captured


@pytest.mark.parametrize("role", ["dem_slope", "canonical_step8a", "label_raw_burndate"])
def test_validator_actual_rejects_a_missing_required_frozen_input(tmp_path, capsys, role):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    Path(wcs.frozen_input_inventory(experiment_id, experiments)[role]["path"]).unlink()

    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    captured = capsys.readouterr().out
    assert "[FAIL] all required frozen identity inputs exist and hash" in captured
    assert role in captured


def test_validator_actual_rejects_a_missing_prelabel_frozen_input(tmp_path, capsys):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    (out / experiment_id / "prelabel_censor" / "prelabel_burndate.tif").unlink()

    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    captured = capsys.readouterr().out
    assert "[FAIL] all required frozen identity inputs exist and hash" in captured
    assert wcs.PRELABEL_FROZEN_ROLE in captured


def test_validator_actual_rejects_a_drifted_required_frozen_hash(tmp_path, capsys):
    result, experiment_id, out, experiments = _run_predictor(tmp_path)
    victim = Path(
        wcs.frozen_input_inventory(experiment_id, experiments)["dem_elevation"]["path"]
    )
    victim.write_bytes(b"mutated-after-the-export")

    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    captured = capsys.readouterr().out
    assert "frozen inputs are unchanged since the export" in captured
    assert "[FAIL]" in captured
    assert "dem_elevation" in captured


def test_validator_actual_rejects_a_drifted_optional_frozen_hash(tmp_path, capsys):
    """An optional role that WAS hashed at export time is still verified --
    only a role that never carried a hash is exempt."""
    experiment_id = any_experiment()
    experiments = tmp_path / "experiments"
    out = tmp_path / "diagnostics"
    _seed_frozen_inputs(experiments, experiment_id)
    inventory = wcs.frozen_input_inventory(experiment_id, experiments)
    _write_grid_raster(
        Path(inventory[wcs.LABEL_ROLE_RAW]["path"]), np.zeros(_TEST_SHAPE),
    )
    # The optional resolver side-file exists for this run, so it is hashed and
    # recorded even though it never binds the identity. `label_kind` is
    # deliberately NOT the raw-BurnDate kind, so label resolution is unaffected.
    stats = Path(inventory["canonical_step8a_stats"]["path"])
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text(json.dumps({"label_kind": "synthetic_fixture"}), encoding="utf-8")

    wcs.run_analysis(
        experiment_id=experiment_id, shifts=[0, 7, 14], dry_run=False,
        from_stage="plan", to_stage="predictor-export",
        output_root=out, experiments_root=experiments,
        prelabel_exporter=_fake_exporter(np.zeros(_TEST_SHAPE)),
        predictor_engine=_fake_predictor_engine(),
    )
    first_code = _validate("actual", experiment_id, out, experiments=experiments)
    first_output = capsys.readouterr().out
    assert first_code == 0, first_output

    stats.write_text(json.dumps({"label_kind": "mutated"}), encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    captured = capsys.readouterr().out
    assert "frozen inputs are unchanged since the export" in captured
    assert "canonical_step8a_stats" in captured


def test_validator_reports_the_stage_lock(tmp_path, capsys):
    _, experiment_id, out, _ = _dry_run_payload(tmp_path)
    _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "l.log", {}))
    captured = capsys.readouterr().out
    assert "[PASS] predictor-export is an implemented actual stage" in captured
    assert "[PASS] no unimplemented stage is reachable" in captured
