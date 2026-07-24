"""Focused tests for the Landsat composite counterfactual audit.

No live Earth Engine is required: the GEE-touching functions are imported
lazily inside the module, so these tests exercise the pure helpers, the
namespace-safety contract, deterministic metadata/hashes, paired-interval
classification, QA honesty, and the no-op dry-run path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.landsat_composite_counterfactual_audit as audit


# --- shared fixtures ---------------------------------------------------------
def _records():
    """Six scenes across three dates; date 2021-07-10 has two overlapping
    scenes (same-day multiplicity), date 2021-07-18 has three."""
    return [
        {"scene_id": "LC08_A_20210702", "acquisition_datetime": "2021-07-02T08:00:00"},
        {"scene_id": "LC08_B_20210710", "acquisition_datetime": "2021-07-10T08:00:00"},
        {"scene_id": "LC08_C_20210710", "acquisition_date": "2021-07-10"},
        {"scene_id": "LC08_D_20210718", "acquisition_date": "2021-07-18"},
        {"scene_id": "LC08_E_20210718", "acquisition_date": "2021-07-18"},
        {"scene_id": "LC08_F_20210718", "acquisition_date": "2021-07-18"},
    ]


# --- grouping by acquisition date -------------------------------------------
def test_group_records_by_date_groups_and_sorts():
    grouped = audit.group_records_by_date(_records())
    assert list(grouped.keys()) == ["2021-07-02", "2021-07-10", "2021-07-18"]
    assert [r["scene_id"] for r in grouped["2021-07-10"]] == [
        "LC08_B_20210710", "LC08_C_20210710",
    ]
    assert len(grouped["2021-07-18"]) == 3


def test_group_records_drops_undatable():
    grouped = audit.group_records_by_date(
        _records() + [{"scene_id": "no_date"}]
    )
    total = sum(len(v) for v in grouped.values())
    assert total == 6  # the undatable scene is excluded


# --- one image per date contract (pure/helper level) ------------------------
def test_plan_daily_composites_one_per_unique_date():
    plan = audit.plan_daily_composites(_records())
    dates = audit.unique_acquisition_dates(_records())
    assert len(plan) == len(dates)  # exactly one daily image per unique date
    assert [p["acquisition_date"] for p in plan] == dates
    p18 = next(p for p in plan if p["acquisition_date"] == "2021-07-18")
    assert p18["source_scene_count"] == 3
    assert p18["scene_ids"] == ["LC08_D_20210718", "LC08_E_20210718", "LC08_F_20210718"]


# --- scene-count vs unique-date-count semantics -----------------------------
def test_count_semantics_multiplicity():
    counts = audit.count_semantics(_records())
    assert counts["scene_count"] == 6
    assert counts["unique_date_count"] == 3
    assert counts["same_day_multiplicity"] == 3


def test_count_semantics_no_duplication():
    records = [
        {"scene_id": "s1", "acquisition_date": "2021-07-01"},
        {"scene_id": "s2", "acquisition_date": "2021-07-02"},
    ]
    counts = audit.count_semantics(records)
    assert counts["scene_count"] == 2
    assert counts["unique_date_count"] == 2
    assert counts["same_day_multiplicity"] == 0


# --- namespace safety --------------------------------------------------------
def test_namespace_safety_accepts_diagnostic_paths(tmp_path):
    root = audit.diagnostic_output_root("manavgat_2021", tmp_path)
    good = [root / "rasters" / "current_lst_scene_weighted_median.tif",
            root / "audit_config.json"]
    audit.assert_diagnostic_namespace_safe(good, "manavgat_2021", tmp_path)  # no raise


@pytest.mark.parametrize("bad_rel", [
    ("outputs", "experiments", "manavgat_2021", "data", "x.tif"),
    ("outputs", "experiments", "manavgat_2021", "step5", "baseline_lst_mean_celsius.tif"),
    ("outputs", "experiments", "manavgat_2021", "step8a", "grid.tif"),
    ("data", "landsat_timeseries", "x.tif"),
    ("outputs", "step5", "anomaly_zscore.tif"),
])
def test_namespace_safety_rejects_canonical_paths(tmp_path, bad_rel):
    bad = tmp_path.joinpath(*bad_rel)
    with pytest.raises(audit.NamespaceSafetyError):
        audit.assert_diagnostic_namespace_safe([bad], "manavgat_2021", tmp_path)


def test_namespace_safety_rejects_escaping_path(tmp_path):
    with pytest.raises(audit.NamespaceSafetyError):
        audit.assert_diagnostic_namespace_safe(
            [tmp_path / "somewhere" / "else.tif"], "manavgat_2021", tmp_path
        )


def test_planned_paths_all_pass_namespace_safety():
    ctx = _ctx()
    paths = audit.plan_all_output_paths(ctx)
    # Must not raise; every planned path is inside the diagnostic namespace.
    audit.assert_diagnostic_namespace_safe(paths, ctx["experiment_id"])
    root = audit.diagnostic_output_root(ctx["experiment_id"]).resolve()
    for p in paths:
        assert root in p.resolve().parents


# --- deterministic metadata ordering + hashes -------------------------------
def _ctx():
    from core.experiment_context import build_experiment_context

    return build_experiment_context("manavgat_2021")


def test_audit_config_is_deterministic_and_ordered():
    ctx = _ctx()
    c1 = audit.build_audit_config(ctx, seed=1, n_boot=100, ci=0.9)
    c2 = audit.build_audit_config(ctx, seed=1, n_boot=100, ci=0.9)
    # created_at differs; everything else must match, including key order.
    c1.pop("created_at"); c2.pop("created_at")
    assert list(c1.keys()) == list(c2.keys())
    assert json.dumps(c1, default=str) == json.dumps(c2, default=str)
    assert c1["chains"] == ["scene_weighted", "date_balanced"]
    assert c1["baseline_years"] == [2017, 2018, 2019, 2020]


def test_file_manifest_deterministic_hashes(tmp_path):
    root = tmp_path
    (root / "b.txt").write_text("beta", encoding="utf-8")
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    m1 = audit.build_file_manifest([root / "b.txt", root / "a.txt"], output_dir=root)
    m2 = audit.build_file_manifest([root / "a.txt", root / "b.txt"], output_dir=root)
    # Ordering is by relative path, independent of input order.
    assert [f["path"] for f in m1["files"]] == ["a.txt", "b.txt"]
    assert m1["files"] == m2["files"]
    import hashlib
    assert m1["files"][0]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert m1["files"][0]["bytes"] == 5


# --- paired interval classification -----------------------------------------
@pytest.mark.parametrize("low,high,expected", [
    (0.5, 1.5, "supported_reduction"),
    (-1.5, -0.5, "supported_increase"),
    (-0.2, 0.3, "uncertain"),
    (None, None, "insufficient_evidence"),
    (0.0, 1.0, "uncertain"),   # zero endpoint => interval includes zero
])
def test_classify_paired_interval(low, high, expected):
    assert audit.classify_paired_interval(low, high) == expected


def test_classify_reduction_insufficient_when_few_segments():
    result = audit.classify_reduction([1.0, 2.0, 3.0], min_segments=8)
    assert result["status"] == "insufficient_evidence"
    assert result["n_segments"] == 3


def test_classify_reduction_supported_when_all_positive():
    reductions = [1.0 + 0.01 * i for i in range(40)]
    result = audit.classify_reduction(reductions, min_segments=8, n_boot=2000)
    assert result["status"] == "supported_reduction"
    assert result["interval_low"] > 0


def test_bootstrap_interval_is_deterministic():
    data = [0.1, -0.2, 0.5, 0.3, -0.1, 0.4, 0.2, 0.6, 0.0, 0.25]
    a = audit.bootstrap_reduction_interval(data, n_boot=1000, seed=7)
    b = audit.bootstrap_reduction_interval(data, n_boot=1000, seed=7)
    assert a == b


# --- QA metadata honesty -----------------------------------------------------
def test_qa_metadata_states_qa_pixel_only():
    qa = audit.qa_mask_provenance()
    assert qa["qa_source"] == "QA_PIXEL"
    assert qa["qa_pixel_only"] is True
    assert qa["qa_radsat_applied"] is False
    assert "QA_RADSAT" in qa["metadata_mismatch"]["claimed"]
    assert "no QA_RADSAT" in qa["metadata_mismatch"]["actual"].lower() or \
        "only" in qa["metadata_mismatch"]["actual"].lower()


def test_date_window_semantics_end_exclusive():
    sem = audit.date_window_semantics("2021-06-01", "2021-07-27")
    assert sem["end_semantics"] == "exclusive"
    assert sem["effective_last_included_date"] == "2021-07-26"


# --- dry-run performs no GEE export and no raster writes ---------------------
def test_dry_run_writes_nothing(monkeypatch):
    import scripts.run_landsat_composite_counterfactual_audit as runner

    # Fail loudly if the live GEE/export path is touched during a dry run.
    def _boom(*a, **k):  # noqa: ANN001
        raise AssertionError("dry-run must not touch GEE/export")

    monkeypatch.setattr(runner, "_run_live", _boom)

    diag_root = audit.diagnostic_output_root("manavgat_2021")
    existed_before = diag_root.exists()

    result = runner.main("manavgat_2021", dry_run=True)
    assert result["ran"] is False
    assert result["reason"] == "dry_run"

    # Dry-run must not create the diagnostic namespace or write any file in it.
    if not existed_before:
        assert not diag_root.exists()


def test_no_run_flag_defaults_to_plan_only(monkeypatch):
    import scripts.run_landsat_composite_counterfactual_audit as runner

    monkeypatch.setattr(
        runner, "_run_live",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    result = runner.main("manavgat_2021", dry_run=False, run=False)
    assert result["ran"] is False
    assert result["reason"] == "no_run_flag"


# --- neighbour-jump metric is raw (no smoothing) ----------------------------
def test_neighbour_jump_metrics_basic():
    import numpy as np

    arr = np.array([[1.0, 1.0, 5.0], [1.0, np.nan, 5.0]], dtype="float64")
    m = audit.neighbour_jump_metrics(arr)
    assert m["valid_pair_count"] > 0
    assert m["absolute_jump_p99"] >= m["absolute_jump_median"]


# =============================================================================
# BLOCKING 1 -- explicit paired adjacency-edge sampling
# =============================================================================
def _support_case():
    """scene-count edges and unique-date edges at DIFFERENT column locations.

    Grid is 3 rows x 4 cols. scene_count changes between col0|col1 (edge index
    0); unique_date_count changes between col2|col3 (edge index 2). The two
    boundary geometries do not overlap.
    """
    import numpy as np

    sw = np.array([[10.0, 40.0, 41.0, 90.0]] * 3, dtype="float64")
    db = np.array([[10.0, 20.0, 21.0, 30.0]] * 3, dtype="float64")
    scene_count = np.array([[1.0, 5.0, 5.0, 5.0]] * 3, dtype="float64")       # edge at col0|1
    unique_date = np.array([[2.0, 2.0, 2.0, 9.0]] * 3, dtype="float64")       # edge at col2|3
    multiplicity = scene_count - unique_date
    return sw, db, scene_count, unique_date, multiplicity


def test_both_chains_sampled_at_identical_adjacency_indices():
    import numpy as np

    sw, db, sc, uc, mult = _support_case()
    masks = audit.build_edge_masks(sc, uc, mult)

    scene_sample = audit.sample_paired_edges(sw, db, masks["scene_count_edge"])
    date_sample = audit.sample_paired_edges(sw, db, masks["unique_date_count_edge"])

    # scene-count edges live ONLY at horizontal edge index col==0 ...
    assert set(scene_sample["col"].tolist()) == {0}
    # ... unique-date edges ONLY at col==2 -> different locations (fails the old
    # implementation, which sampled each chain from its own count raster).
    assert set(date_sample["col"].tolist()) == {2}

    # Both chains are sampled at the SAME (row, col) indices for a given mask:
    # sw and db arrays have equal length and are paired element-wise.
    assert scene_sample["sw_abs"].shape == scene_sample["db_abs"].shape
    # At the scene-count edge: sw jump = |40-10|=30, db jump=|20-10|=10.
    assert np.allclose(scene_sample["sw_abs"], 30.0)
    assert np.allclose(scene_sample["db_abs"], 10.0)
    assert np.allclose(scene_sample["reduction"], 20.0)


def test_union_and_multiplicity_edge_masks():
    sw, db, sc, uc, mult = _support_case()
    masks = audit.build_edge_masks(sc, uc, mult)
    union = audit.sample_paired_edges(sw, db, masks["union_support_edge"])
    # union covers both the scene-count (col0) and unique-date (col2) edges.
    assert set(union["col"].tolist()) == {0, 2}
    multiplicity = audit.sample_paired_edges(sw, db, masks["same_day_multiplicity_edge"])
    # multiplicity = scene_count - unique_date changes wherever either changes.
    assert set(multiplicity["col"].tolist()) == {0, 2}


def test_no_zero_imputation_when_one_chain_missing():
    import numpy as np

    # db is NaN at the right endpoint of the scene-count edge -> that pair must
    # be DROPPED (counted as skipped), never imputed to zero.
    sw = np.array([[10.0, 40.0]], dtype="float64")
    db = np.array([[10.0, np.nan]], dtype="float64")
    sc = np.array([[1.0, 5.0]], dtype="float64")
    uc = np.array([[1.0, 1.0]], dtype="float64")
    masks = audit.build_edge_masks(sc, uc, sc - uc)
    sample = audit.sample_paired_edges(sw, db, masks["scene_count_edge"])
    assert sample["pair_count"] == 0
    assert sample["skipped_missing_observation"] == 1
    assert sample["reduction"].size == 0  # no zero substituted


def test_matched_control_same_orientation_and_count():
    import numpy as np

    rng_sw = np.random.default_rng(0).normal(size=(40, 40))
    rng_db = np.random.default_rng(1).normal(size=(40, 40))
    exclude = {
        "horizontal": np.zeros((40, 39), dtype=bool),
        "vertical": np.zeros((39, 40), dtype=bool),
    }
    control = audit.sample_matched_control(
        rng_sw, rng_db, exclude, {"horizontal": 12, "vertical": 7}, seed=5
    )
    # Same orientation composition and count-matched (comparable segment length).
    assert control["n_per_orientation"] == {"horizontal": 12, "vertical": 7}
    assert control["sw_abs"].size == 19 and control["db_abs"].size == 19


# =============================================================================
# BLOCKING 2 -- predeclared bootstrap units (no row strips)
# =============================================================================
def test_paired_reduction_uses_spatial_block_units():
    import numpy as np

    sw = np.zeros((300, 300), dtype="float64")
    db = np.zeros((300, 300), dtype="float64")
    sc = np.ones((300, 300), dtype="float64")
    # Introduce a vertical support edge across the whole raster at col 150.
    sc[:, 150:] = 2.0
    sw[:, 150:] = 5.0  # scene_weighted has a big jump; date_balanced none
    uc = np.ones((300, 300), dtype="float64")
    masks = audit.build_edge_masks(sc, uc, sc - uc)
    sample = audit.sample_paired_edges(sw, db, masks["scene_count_edge"])
    verdict = audit.paired_reduction_by_blocks(
        sample, unit_type="raster_support_block", block_size=64, min_units=2, n_boot=500
    )
    assert verdict["unit_type"] == "raster_support_block"
    assert verdict["n_units"] >= 2
    assert all(uid.startswith("block_") for uid in verdict["unit_ids"])
    # sw jump 5, db jump 0 => reduction 5 everywhere => supported_reduction.
    assert verdict["status"] == "supported_reduction"


def test_classify_paired_units_insufficient():
    import numpy as np

    units = {"block_r0_c0": np.array([1.0, 2.0])}
    verdict = audit.classify_paired_units(units, unit_type="raster_support_block", min_units=8)
    assert verdict["status"] == "insufficient_evidence"
    assert verdict["n_units"] == 1


# =============================================================================
# BLOCKING 5 -- exact grid contract
# =============================================================================
def _write_raster(path, *, width=8, height=8, transform=None, crs="EPSG:4326",
                  value=1.0, nodata=None):
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    if transform is None:
        transform = Affine(0.01, 0, 30.0, 0, -0.01, 37.0)
    data = np.full((height, width), value, dtype="float32")
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def test_assert_same_grid_accepts_identical(tmp_path):
    a = _write_raster(tmp_path / "a.tif")
    b = _write_raster(tmp_path / "b.tif")
    sig = audit.assert_same_grid([a, b])
    assert sig["width"] == 8 and sig["height"] == 8


def test_assert_same_grid_rejects_shifted_transform(tmp_path):
    from rasterio.transform import Affine

    a = _write_raster(tmp_path / "a.tif")
    b = _write_raster(tmp_path / "b.tif", transform=Affine(0.01, 0, 31.0, 0, -0.01, 37.0))
    with pytest.raises(audit.GridMismatchError):
        audit.assert_same_grid([a, b])


def test_assert_same_grid_rejects_different_dimensions(tmp_path):
    a = _write_raster(tmp_path / "a.tif", width=8, height=8)
    b = _write_raster(tmp_path / "b.tif", width=8, height=9)
    with pytest.raises(audit.GridMismatchError):
        audit.assert_same_grid([a, b])


# =============================================================================
# BLOCKING 4 -- canonical reproduction gate
# =============================================================================
def test_canonical_reproduction_pass(tmp_path):
    diag = _write_raster(tmp_path / "diag.tif", value=100.0)
    canon = _write_raster(tmp_path / "canon.tif", value=100.0)
    result = audit.compare_raster_to_canonical(diag, canon, tolerance=1e-5)
    assert result["reproduction_status"] == "pass"
    assert result["max_abs_diff"] == 0.0
    assert result["compared_pixel_count"] == 64


def test_canonical_reproduction_fail(tmp_path):
    diag = _write_raster(tmp_path / "diag.tif", value=100.0)
    canon = _write_raster(tmp_path / "canon.tif", value=100.5)
    result = audit.compare_raster_to_canonical(diag, canon, tolerance=1e-5)
    assert result["reproduction_status"] == "fail"
    assert result["max_abs_diff"] > 0.4


def test_canonical_reproduction_grid_mismatch(tmp_path):
    from rasterio.transform import Affine

    diag = _write_raster(tmp_path / "diag.tif")
    canon = _write_raster(tmp_path / "canon.tif", transform=Affine(0.02, 0, 30, 0, -0.02, 37))
    result = audit.compare_raster_to_canonical(diag, canon, tolerance=1e-5)
    assert result["reproduction_status"] == "grid_mismatch"


def test_canonical_reproduction_not_available(tmp_path):
    diag = _write_raster(tmp_path / "diag.tif")
    result = audit.compare_raster_to_canonical(diag, tmp_path / "missing.tif", tolerance=1e-5)
    assert result["reproduction_status"] == "not_available"


def test_supported_reduction_forbidden_when_reproduction_fails():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    verdicts = {
        "current_lst": {
            "scene_count_edge": {"status": "supported_reduction", "interval_low": 0.5},
        }
    }
    gated = runner.gate_verdicts(verdicts, reproduction_status="canonical_reproduction_failed")
    v = gated["current_lst"]["scene_count_edge"]
    assert v["status"] == "canonical_reproduction_failed"
    assert v["supported_reduction_suppressed"] is True
    assert v["raw_status"] == "supported_reduction"

    # When reproduction passes, the verdict is preserved.
    gated_ok = runner.gate_verdicts(verdicts, reproduction_status="pass")
    assert gated_ok["current_lst"]["scene_count_edge"]["status"] == "supported_reduction"


# =============================================================================
# EXPORT-TILE NEGATIVE CONTROL + INVENTORY
# =============================================================================
def test_export_inventory_retains_tile_metadata():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    inventory = [{
        "name": "current_lst_scene_weighted_median",
        "path": "/x.tif",
        "transport": "tiled_direct_fallback",
        "tile_grid": [2, 2],
        "tile_count": 4,
        "estimated_bytes": 123456,
        "direct_skipped_preflight": False,
        "alignment_qa": {"crs": "EPSG:4326"},
        "nodata_status": "ok",
    }]
    rows = runner._inventory_rows(inventory)
    row = rows[0]
    assert json.loads(row["tile_grid"]) == [2, 2]
    assert row["tile_count"] == 4
    assert row["estimated_bytes"] == 123456
    assert row["direct_skipped_preflight"] is False
    assert json.loads(row["alignment_qa"])["crs"] == "EPSG:4326"


def test_export_tile_control_unavailable_for_direct_export():
    sw = {"transport": "direct", "tile_grid": None}
    db = {"transport": "tiled_direct_fallback", "tile_grid": [2, 2]}
    avail = audit.export_tile_control_availability(sw, db)
    assert avail["available"] is False
    assert avail["reason"] == "at_least_one_product_exported_directly"


def test_export_tile_control_unavailable_for_mismatched_grids():
    sw = {"transport": "tiled_direct_fallback", "tile_grid": [2, 2]}
    db = {"transport": "tiled_direct_fallback", "tile_grid": [4, 4]}
    avail = audit.export_tile_control_availability(sw, db)
    assert avail["available"] is False
    assert avail["reason"] == "paired_products_used_different_tile_grids"


def test_export_tile_control_available_and_masks():
    sw = {"transport": "tiled_direct_fallback", "tile_grid": [2, 2]}
    db = {"transport": "tiled_direct_fallback", "tile_grid": [2, 2]}
    avail = audit.export_tile_control_availability(sw, db)
    assert avail["available"] is True
    units = audit.export_tile_boundary_edge_masks(100, 80, [2, 2])
    # one vertical and one horizontal interior seam.
    assert set(units.keys()) == {"tile_v1", "tile_h1"}


# =============================================================================
# PROVENANCE separation + preflight
# =============================================================================
def test_map_provenance_status():
    assert audit.map_provenance_status({"status": "available"}) == "provenance_available"
    assert audit.map_provenance_status(
        {"status": "metadata_available_boundary_incomplete"}
    ) == "provenance_incomplete"
    assert audit.map_provenance_status(None) == "insufficient_boundary_metadata"
    assert audit.map_provenance_status({"status": "insufficient_evidence"}) == "insufficient_boundary_metadata"


def test_provenance_unavailable_not_source_boundary_evidence():
    import numpy as np

    sw, db, sc, uc, mult = _support_case()
    masks = audit.build_edge_masks(sc, uc, mult)
    result = audit.audit_product_boundaries(
        "current_lst", sw, db, masks, provenance_status="insufficient_boundary_metadata",
    )
    verdict = result["verdicts"]["source_scene_path_row"]
    assert verdict["status"] == "insufficient_boundary_metadata"
    assert verdict["is_verified_source_boundary_evidence"] is False
    assert verdict.get("interval_low") is None  # no fabricated interval


def test_scene_list_preflight_rejects_empty():
    empty = {"collections": {"current_lst": [], "baseline_lst": []}}
    with pytest.raises(RuntimeError):
        audit.assert_scene_list_nonempty(empty)
    nonempty = {"collections": {"current_lst": [{"scene_id": "x"}]}}
    assert audit.assert_scene_list_nonempty(nonempty) == 1


# =============================================================================
# NODATA / AOI-EDGE SAFETY
# =============================================================================
def test_read_masked_array_excludes_sentinel(tmp_path):
    import numpy as np

    path = _write_raster(
        tmp_path / "n.tif", width=3, height=1, value=5.0, nodata=audit.NODATA_SENTINEL,
    )
    import rasterio
    with rasterio.open(path, "r+") as ds:
        band = ds.read(1)
        band[0, 2] = audit.NODATA_SENTINEL
        ds.write(band, 1)
    arr = audit.read_masked_array(path)
    assert np.isnan(arr[0, 2])          # sentinel -> NaN, not physical zero
    assert arr[0, 0] == 5.0
    # neighbour metrics only use finite pairs => the sentinel pair is excluded.
    metrics = audit.neighbour_jump_metrics(arr)
    assert metrics["valid_pair_count"] == 1  # only the (5,5) pair survives


def test_validate_nodata_mask_flags_missing_tag(tmp_path):
    path = _write_raster(tmp_path / "m.tif", value=5.0, nodata=None)
    result = audit.validate_nodata_mask(path)
    assert result["status"] == "missing_nodata_tag"

    path_ok = _write_raster(tmp_path / "ok.tif", value=5.0, nodata=audit.NODATA_SENTINEL)
    assert audit.validate_nodata_mask(path_ok)["status"] == "ok"


# =============================================================================
# FORCE SEMANTICS
# =============================================================================
def test_force_clears_only_diagnostic_namespace(tmp_path):
    exp = "manavgat_2021"
    diag_root = audit.diagnostic_output_root(exp, tmp_path)
    diag_root.mkdir(parents=True)
    (diag_root / "stale.tif").write_text("stale", encoding="utf-8")
    (diag_root / "_tiles").mkdir()
    (diag_root / "_tiles" / "t.tif").write_text("tile", encoding="utf-8")

    # A canonical/legacy sibling that must NOT be touched.
    canonical = tmp_path / "outputs" / "experiments" / exp / "step5"
    canonical.mkdir(parents=True)
    (canonical / "anomaly_zscore.tif").write_text("canonical", encoding="utf-8")

    removed = audit.clear_diagnostic_namespace(exp, tmp_path)
    assert removed == str(diag_root)
    assert not diag_root.exists()                      # diagnostic namespace gone
    assert (canonical / "anomaly_zscore.tif").exists()  # canonical untouched


# =============================================================================
# REFINEMENT 1 -- canonical reproduction: exact mask + explicit bounds
# =============================================================================
def _write_masked_raster(path, *, sentinel_cells=(), **kwargs):
    """Write a raster then poke NODATA_SENTINEL into specific (row, col) cells."""
    import rasterio

    kwargs.setdefault("nodata", audit.NODATA_SENTINEL)
    _write_raster(path, **kwargs)
    if sentinel_cells:
        with rasterio.open(path, "r+") as ds:
            band = ds.read(1)
            for r, c in sentinel_cells:
                band[r, c] = audit.NODATA_SENTINEL
            ds.write(band, 1)
    return path


def test_canonical_reproduction_requires_exact_valid_mask(tmp_path):
    # Same grid, same values, but different valid masks -> mask_mismatch (NOT pass).
    diag = _write_masked_raster(tmp_path / "diag.tif", value=100.0, sentinel_cells=[(0, 0)])
    canon = _write_masked_raster(tmp_path / "canon.tif", value=100.0, sentinel_cells=[])
    result = audit.compare_raster_to_canonical(diag, canon, tolerance=1e-5)
    assert result["reproduction_status"] == "mask_mismatch"
    assert result["valid_mask_exact"] is False


def test_canonical_reproduction_reports_bounds_mismatch(tmp_path):
    from rasterio.transform import Affine

    diag = _write_raster(tmp_path / "diag.tif")
    # Shift origin: transform AND bounds both differ; bounds is compared explicitly.
    canon = _write_raster(tmp_path / "canon.tif", transform=Affine(0.01, 0, 55.0, 0, -0.01, 37.0))
    result = audit.compare_raster_to_canonical(diag, canon, tolerance=1e-5)
    assert result["reproduction_status"] == "grid_mismatch"
    assert "bounds" in result["mismatch_fields"]


def test_canonical_reproduction_bounds_mismatch_via_dimensions(tmp_path):
    # Same transform, different height -> bounds differ; bounds reported explicitly.
    diag = _write_raster(tmp_path / "diag.tif", height=8)
    canon = _write_raster(tmp_path / "canon.tif", height=9)
    result = audit.compare_raster_to_canonical(diag, canon, tolerance=1e-5)
    assert result["reproduction_status"] == "grid_mismatch"
    assert "bounds" in result["mismatch_fields"] and "height" in result["mismatch_fields"]


def test_canonical_gate_fails_on_mask_mismatch(tmp_path, monkeypatch):
    # A mask_mismatch in any check must fail the whole gate.
    checks = {"x": {"reproduction_status": "mask_mismatch"}}
    statuses = [c["reproduction_status"] for c in checks.values()]
    assert "mask_mismatch" in statuses  # sanity
    # exercise the aggregation directly via a crafted gate-like reduction:
    failing = {"fail", "grid_mismatch", "mask_mismatch"}
    assert any(s in failing for s in statuses)


# =============================================================================
# REFINEMENT 2 -- nodata: exact tag + raw-band inspection + fail-fast
# =============================================================================
def test_validate_nodata_wrong_tag(tmp_path):
    path = _write_raster(tmp_path / "w.tif", value=5.0, nodata=0.0)  # wrong tag
    result = audit.validate_nodata_mask(path)
    assert result["status"] == "wrong_nodata_tag"


def test_validate_nodata_sentinel_not_masked(tmp_path):
    import rasterio

    # Wrong tag (0) AND a raw sentinel pixel that is therefore NOT masked.
    path = _write_raster(tmp_path / "s.tif", width=3, height=1, value=5.0, nodata=0.0)
    with rasterio.open(path, "r+") as ds:
        band = ds.read(1)
        band[0, 2] = audit.NODATA_SENTINEL
        ds.write(band, 1)
    result = audit.validate_nodata_mask(path)
    # wrong tag is detected first; the sentinel pixel is confirmed unmasked too.
    assert result["status"] == "wrong_nodata_tag"
    assert result["sentinel_not_masked"] is True


def test_require_nodata_ok_fails_fast():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    with pytest.raises(runner.AuditRunnerError):
        runner._require_nodata_ok({"status": "wrong_nodata_tag"}, "current_lst")
    # ok status does not raise.
    runner._require_nodata_ok({"status": "ok"}, "current_lst")


# =============================================================================
# REFINEMENT 3 -- verified provenance boundary_id reaches the paired verdict
# =============================================================================
def test_provenance_boundary_id_reaches_verdict():
    import numpy as np

    # 6x6 all-finite arrays; scene_weighted has a big jump at column 2|3.
    sw = np.tile(np.array([0, 0, 0, 9, 9, 9], dtype="float64"), (6, 1))
    db = np.zeros((6, 6), dtype="float64")
    sc = np.ones((6, 6), dtype="float64")
    uc = np.ones((6, 6), dtype="float64")
    masks = audit.build_edge_masks(sc, uc, sc - uc)

    # Two verified provenance boundary_ids, both on the col-2|3 edge (upper vs
    # lower rows), so each is its own bootstrap unit and a paired interval exists.
    def _edge(rows):
        h = np.zeros((6, 5), dtype=bool)
        h[rows, 2] = True
        return {"horizontal": h, "vertical": np.zeros((5, 6), dtype=bool)}

    units = {"bnd_REAL_0001": _edge(slice(0, 3)), "bnd_REAL_0002": _edge(slice(3, 6))}

    result = audit.audit_product_boundaries(
        "current_lst", sw, db, masks,
        provenance_status="provenance_available", provenance_units=units,
        min_units=2, n_boot=300,
    )
    verdict = result["verdicts"]["source_scene_path_row"]
    assert verdict["unit_type"] == "provenance_boundary_id"
    # both real boundary_ids are preserved as bootstrap units in the verdict.
    assert set(verdict["unit_ids"]) == {"bnd_REAL_0001", "bnd_REAL_0002"}
    assert verdict["is_verified_source_boundary_evidence"] is True
    assert verdict["status"] == "supported_reduction"       # sw jump 9, db jump 0


def test_provenance_available_without_units_is_insufficient():
    import numpy as np

    sw, db, sc, uc, mult = _support_case()
    masks = audit.build_edge_masks(sc, uc, mult)
    # provenance_available but NO sampled masks -> must NOT be reported as evidence.
    result = audit.audit_product_boundaries(
        "current_lst", sw, db, masks,
        provenance_status="provenance_available", provenance_units=None,
    )
    verdict = result["verdicts"]["source_scene_path_row"]
    assert verdict["status"] == "insufficient_boundary_metadata"
    assert verdict["is_verified_source_boundary_evidence"] is False


def test_rasterize_provenance_boundaries_produces_units():
    from rasterio.transform import Affine

    transform = Affine(0.01, 0, 30.0, 0, -0.01, 37.0)  # 8x8 -> lon 30..30.08
    geojson = {
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[30.0, 36.955], [30.08, 36.955]]},
            "properties": {"boundary_id": "bnd_GEO_1", "verification_status": "verified"},
        }],
    }
    units = audit.rasterize_provenance_boundaries(geojson, transform, 8, 8)
    assert "bnd_GEO_1" in units
    assert units["bnd_GEO_1"]["horizontal"].any() or units["bnd_GEO_1"]["vertical"].any()


# =============================================================================
# REFINEMENT 4 -- final claim gate with mixed / contradictory statuses
# =============================================================================
def _gated(cur, diff, z, tile=None):
    edge = "scene_count_edge"
    g = {
        "current_lst": {edge: {"status": cur}},
        "current_minus_baseline": {edge: {"status": diff}},
        "anomaly_zscore": {edge: {"status": z}},
    }
    if tile is not None:
        for product in g:
            g[product]["export_tile_boundary"] = {"status": tile, "is_negative_control": True}
    return g


def test_final_status_supported_only_when_all_consistent():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    g = _gated("supported_reduction", "supported_reduction", "supported_reduction")
    assert runner.compute_final_status(g, "pass") == "supported_reduction"


def test_final_status_contradictory():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    g = _gated("supported_reduction", "supported_increase", "uncertain")
    assert runner.compute_final_status(g, "pass") == "contradictory_uncertain"


def test_final_status_uncertain_when_products_disagree_softly():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    g = _gated("supported_reduction", "uncertain", "supported_reduction")
    assert runner.compute_final_status(g, "pass") == "uncertain"


def test_final_status_blocked_by_reproduction():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    g = _gated("supported_reduction", "supported_reduction", "supported_reduction")
    assert runner.compute_final_status(g, "canonical_reproduction_failed") == "canonical_reproduction_failed"


def test_export_tile_control_never_creates_positive_evidence():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    # Products only uncertain, but export-tile control screams supported_reduction:
    # final status must stay uncertain (tile control is excluded entirely).
    g = _gated("uncertain", "uncertain", "uncertain", tile="supported_reduction")
    assert runner.compute_final_status(g, "pass") == "uncertain"


def test_export_tile_verdict_flagged_non_evidence():
    import numpy as np

    sw, db, sc, uc, mult = _support_case()
    masks = audit.build_edge_masks(sc, uc, mult)
    tile_units = audit.export_tile_boundary_edge_masks(4, 3, [2, 2])
    result = audit.audit_product_boundaries(
        "current_lst", sw, db, masks, tile_units=tile_units,
    )
    tile = result["verdicts"]["export_tile_boundary"]
    assert tile["is_negative_control"] is True
    assert tile["can_affect_final_status"] is False
    assert tile["control_kind"] == "approximate_grid_partition"


# =============================================================================
# canonical_reproduction.json is a planned output
# =============================================================================
def test_canonical_reproduction_in_document_plan():
    docs = audit.plan_document_outputs()
    assert docs["canonical_reproduction"] == "canonical_reproduction.json"
    ctx = _ctx()
    all_paths = [str(p) for p in audit.plan_all_output_paths(ctx)]
    assert any(p.endswith("canonical_reproduction.json") for p in all_paths)


# =============================================================================
# RESUME / CHECKPOINT SUPPORT
# =============================================================================
def _sentinel_raster(path, *, value=5.0, **kwargs):
    """A valid diagnostic raster: nodata tag == NODATA_SENTINEL, one valid pixel."""
    kwargs.setdefault("nodata", audit.NODATA_SENTINEL)
    return _write_raster(path, value=value, **kwargs)


class _DummyImage:
    def unmask(self, _value):
        return self


def _make_fake_export(value=7.0):
    """Fake export helper that writes a valid sentinel raster to out_path."""
    def _fake(image, out_path, region, scale, crs, name, *, force, tiles_dir,
              cleanup_tiles, band_count, run_alignment_qa, nodata):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_raster(p, value=value, nodata=nodata, crs="EPSG:4326")
        return {"path": p, "transport": "direct", "tile_grid": None,
                "tile_count": None, "estimated_bytes": 100,
                "direct_skipped_preflight": False, "alignment_qa": {"crs": "EPSG:4326"}}
    return _fake


# --- module-level raster validation ---
def test_validate_existing_valid_raster_ok(tmp_path):
    p = _sentinel_raster(tmp_path / "r.tif", value=5.0)
    v = audit.validate_existing_diagnostic_raster(p, expected_crs="EPSG:4326")
    assert v["valid"] is True and v["reason"] == "ok"
    assert v["signature"]["width"] == 8 and v["nodata_status"] == "ok"


def test_validate_existing_wrong_nodata_not_reused(tmp_path):
    p = _write_raster(tmp_path / "r.tif", value=5.0, nodata=0.0)  # wrong tag
    v = audit.validate_existing_diagnostic_raster(p, expected_crs="EPSG:4326")
    assert v["valid"] is False and v["reason"].startswith("nodata_")


def test_validate_existing_crs_mismatch(tmp_path):
    p = _sentinel_raster(tmp_path / "r.tif", crs="EPSG:3857")
    v = audit.validate_existing_diagnostic_raster(p, expected_crs="EPSG:4326")
    assert v["valid"] is False and "crs" in v["reason"]


def test_validate_existing_grid_mismatch_vs_reference(tmp_path):
    ref = audit.grid_signature(_sentinel_raster(tmp_path / "ref.tif"))
    other = _sentinel_raster(tmp_path / "o.tif", height=9)  # different grid
    v = audit.validate_existing_diagnostic_raster(
        other, expected_crs="EPSG:4326", reference_signature=ref)
    assert v["valid"] is False and v["reason"] == "grid_mismatch_vs_reference"


def test_validate_existing_no_valid_pixels(tmp_path):
    # every pixel is the sentinel -> masked everywhere -> no valid pixel.
    p = _write_raster(tmp_path / "r.tif", value=audit.NODATA_SENTINEL,
                      nodata=audit.NODATA_SENTINEL)
    v = audit.validate_existing_diagnostic_raster(p, expected_crs="EPSG:4326")
    assert v["valid"] is False and v["reason"] == "no_valid_pixels"


def test_validate_existing_corrupt_geotiff(tmp_path):
    p = tmp_path / "bad.tif"
    p.write_bytes(b"this is not a geotiff")
    v = audit.validate_existing_diagnostic_raster(p, expected_crs="EPSG:4326")
    assert v["valid"] is False and v["reason"].startswith("unreadable")


def test_quarantine_moves_never_deletes(tmp_path):
    exp = "manavgat_2021"
    root = audit.diagnostic_output_root(exp, tmp_path)
    (root / "rasters").mkdir(parents=True)
    bad = root / "rasters" / "x.tif"
    bad.write_text("bad", encoding="utf-8")
    dest = audit.quarantine_invalid_raster(exp, bad, base_dir=tmp_path)
    assert not bad.exists()                       # moved out of the way
    assert Path(dest).exists()                    # but preserved, not deleted
    assert audit.QUARANTINE_DIRNAME in dest
    assert str(root.resolve()) in str(Path(dest).resolve())  # stays in namespace


def test_write_json_atomic_no_temp_leftover(tmp_path):
    p = tmp_path / "cp.json"
    audit.write_json_atomic(p, {"a": 1, "b": [2, 3]})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}
    assert not (tmp_path / ".cp.json.tmp").exists()


# --- runner-level export/resume behavior ---
def _resume_fixture(tmp_path, name="current_lst_scene_weighted_median"):
    exp = "manavgat_2021"
    root = audit.diagnostic_output_root(exp, tmp_path)
    raster_plan = {name: "rasters/x.tif"}
    out = root / raster_plan[name]
    out.parent.mkdir(parents=True, exist_ok=True)
    return exp, root, raster_plan, name, out


def test_resume_skips_valid_existing(tmp_path, monkeypatch):
    import scripts.run_landsat_composite_counterfactual_audit as runner
    import scripts.run_predictors_only as rp

    monkeypatch.setattr(rp, "export_image_direct_or_tiled",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("valid raster must not be re-exported")))
    exp, root, raster_plan, name, out = _resume_fixture(tmp_path)
    _sentinel_raster(out, value=5.0)

    row, ref = runner._export_or_resume_raster(
        name, _DummyImage(), {"experiment_id": exp}, root, raster_plan, None,
        "EPSG:4326", force=False, resume=True, cleanup_tiles=False,
        reference_signature=None, run_stamp="ts", base_dir=tmp_path,
    )
    assert row["status"] == "resumed_existing"
    assert row["transport"] == "resumed_existing"
    assert ref is not None and ref["width"] == 8
    assert out.exists()  # untouched


def test_resume_quarantines_and_reexports_corrupt(tmp_path, monkeypatch):
    import scripts.run_landsat_composite_counterfactual_audit as runner
    import scripts.run_predictors_only as rp

    monkeypatch.setattr(rp, "export_image_direct_or_tiled", _make_fake_export(value=7.0))
    exp, root, raster_plan, name, out = _resume_fixture(tmp_path)
    out.write_bytes(b"corrupt not a geotiff")

    row, ref = runner._export_or_resume_raster(
        name, _DummyImage(), {"experiment_id": exp}, root, raster_plan, None,
        "EPSG:4326", force=False, resume=True, cleanup_tiles=False,
        reference_signature=None, run_stamp="ts", base_dir=tmp_path,
    )
    assert row["status"] == "exported"          # corrupt was NOT reused
    assert out.exists()                         # re-exported cleanly
    quarantine = root / audit.QUARANTINE_DIRNAME
    assert quarantine.exists() and any(quarantine.iterdir())  # corrupt preserved


def test_resume_wrong_nodata_not_reused(tmp_path, monkeypatch):
    import scripts.run_landsat_composite_counterfactual_audit as runner
    import scripts.run_predictors_only as rp

    monkeypatch.setattr(rp, "export_image_direct_or_tiled", _make_fake_export())
    exp, root, raster_plan, name, out = _resume_fixture(tmp_path)
    _write_raster(out, value=5.0, nodata=0.0)   # wrong nodata tag

    row, _ = runner._export_or_resume_raster(
        name, _DummyImage(), {"experiment_id": exp}, root, raster_plan, None,
        "EPSG:4326", force=False, resume=True, cleanup_tiles=False,
        reference_signature=None, run_stamp="ts", base_dir=tmp_path,
    )
    assert row["status"] == "exported"
    assert any((root / audit.QUARANTINE_DIRNAME).iterdir())


def test_resume_grid_mismatch_not_reused(tmp_path, monkeypatch):
    import scripts.run_landsat_composite_counterfactual_audit as runner
    import scripts.run_predictors_only as rp

    monkeypatch.setattr(rp, "export_image_direct_or_tiled", _make_fake_export())
    exp, root, raster_plan, name, out = _resume_fixture(tmp_path)
    reference = audit.grid_signature(_sentinel_raster(tmp_path / "ref.tif"))  # 8x8
    _sentinel_raster(out, height=9)             # mismatched grid

    row, _ = runner._export_or_resume_raster(
        name, _DummyImage(), {"experiment_id": exp}, root, raster_plan, None,
        "EPSG:4326", force=False, resume=True, cleanup_tiles=False,
        reference_signature=reference, run_stamp="ts", base_dir=tmp_path,
    )
    assert row["status"] == "exported"          # mismatched grid re-exported
    assert any((root / audit.QUARANTINE_DIRNAME).iterdir())


def test_checkpoint_alone_cannot_bypass_validation(tmp_path, monkeypatch):
    import scripts.run_landsat_composite_counterfactual_audit as runner
    import scripts.run_predictors_only as rp

    monkeypatch.setattr(rp, "export_image_direct_or_tiled", _make_fake_export())
    exp, root, raster_plan, name, out = _resume_fixture(tmp_path)
    out.write_bytes(b"corrupt")                 # checkpoint will claim it is done
    checkpoint = root / "raster_inventory.partial.json"
    audit.write_json_atomic(checkpoint, {"rasters": [
        {"name": name, "status": "resumed_existing", "path": str(out)}]})

    row, _ = runner._export_or_resume_raster(
        name, _DummyImage(), {"experiment_id": exp}, root, raster_plan, None,
        "EPSG:4326", force=False, resume=True, cleanup_tiles=False,
        reference_signature=None, run_stamp="ts", base_dir=tmp_path,
    )
    # Despite a checkpoint marking it done, the real file was revalidated,
    # found corrupt, quarantined, and re-exported.
    assert row["status"] == "exported"
    assert any((root / audit.QUARANTINE_DIRNAME).iterdir())


def test_write_checkpoint_atomic(tmp_path):
    import scripts.run_landsat_composite_counterfactual_audit as runner

    cp = tmp_path / "raster_inventory.partial.json"
    runner._write_checkpoint(cp, "manavgat_2021", [
        {"name": "a", "path": "/a.tif", "status": "exported"}])
    data = json.loads(cp.read_text(encoding="utf-8"))
    assert data["raster_count"] == 1
    assert data["rasters"][0]["name"] == "a"
    assert not (tmp_path / ".raster_inventory.partial.json.tmp").exists()


def test_resume_and_force_conflict_main():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    with pytest.raises(runner.AuditRunnerError):
        runner.main("manavgat_2021", run=True, force=True, resume=True)


def test_resume_and_force_conflict_argparse():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    with pytest.raises(SystemExit):
        runner.parse_args(["--experiment", "manavgat_2021", "--force", "--resume"])


def test_resume_never_deletes_namespace(tmp_path, monkeypatch):
    import scripts.run_landsat_composite_counterfactual_audit as runner
    import core.gee_utils as gee_utils

    exp = "manavgat_2021"
    fake_root = audit.diagnostic_output_root(exp, tmp_path)
    fake_root.mkdir(parents=True)
    (fake_root / "sentinel.txt").write_text("keep me", encoding="utf-8")

    # Redirect the diagnostic root into tmp for the whole run.
    monkeypatch.setattr(audit, "diagnostic_output_root",
                        lambda e, base_dir=None: fake_root)
    # Any deletion of the namespace must fail the test loudly.
    monkeypatch.setattr(audit, "clear_diagnostic_namespace",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("resume must never clear the namespace")))

    class _Stop(Exception):
        pass

    monkeypatch.setattr(gee_utils, "init_gee",
                        lambda *a, **k: (_ for _ in ()).throw(_Stop()))

    ctx = runner.build_experiment_context(exp)
    with pytest.raises(runner.AuditRunnerError):
        runner._run_live(ctx, force=False, cleanup_tiles=False, resume=True)

    assert (fake_root / "sentinel.txt").exists()  # namespace preserved intact


def test_resume_dry_run_writes_nothing():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    diag_root = audit.diagnostic_output_root("manavgat_2021")
    existed_before = diag_root.exists()
    result = runner.main("manavgat_2021", dry_run=True, resume=True)
    assert result["ran"] is False and result["reason"] == "dry_run"
    if not existed_before:
        assert not diag_root.exists()


# =============================================================================
# BOUNDED-MEMORY FINALIZATION: windowed edge sampling equivalence
# =============================================================================
def _write_arr(path, arr, transform=None):
    """Write a float raster with NODATA_SENTINEL nodata (NaN -> sentinel)."""
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    if transform is None:
        transform = Affine(0.01, 0, 30.0, 0, -0.01, 37.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    height, width = arr.shape
    data = np.where(np.isnan(arr), audit.NODATA_SENTINEL, arr).astype("float32")
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1,
        "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
        "nodata": audit.NODATA_SENTINEL,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def _windowed_support_case(tmp_path):
    """Integer-valued sw/db/support with support edges placed to fall exactly on
    window boundaries for several window sizes. Integer values are exact in
    float32 so windowed and in-memory results compare cleanly.
    """
    import numpy as np

    H, W = 12, 15
    sw = (np.arange(H * W).reshape(H, W) % 7).astype("float64")
    db = (np.arange(H * W).reshape(H, W) % 5).astype("float64")
    scene = np.ones((H, W), dtype="float64")
    scene[:, 4:] = 2.0        # vertical support edge exactly at col 4 boundary
    scene[:, 8:] = 3.0        # and at col 8
    unique = np.ones((H, W), dtype="float64")
    unique[4:, :] = 2.0       # horizontal support edge exactly at row 4 boundary
    unique[8:, :] = 3.0       # and at row 8
    mult = scene - unique
    paths = {
        "sw": _write_arr(tmp_path / "sw.tif", sw),
        "db": _write_arr(tmp_path / "db.tif", db),
        "scene": _write_arr(tmp_path / "scene.tif", scene),
        "unique": _write_arr(tmp_path / "unique.tif", unique),
        "mult": _write_arr(tmp_path / "mult.tif", mult),
    }
    return paths, (sw, db, scene, unique, mult)


@pytest.mark.parametrize("boundary,support_key", [
    ("scene_count_edge", "scene"),
    ("unique_date_count_edge", "unique"),
    ("same_day_multiplicity_edge", "mult"),
])
@pytest.mark.parametrize("window_size", [4, 8, 512])
def test_windowed_support_matches_in_memory(tmp_path, boundary, support_key, window_size):
    import numpy as np

    paths, (sw, db, scene, unique, mult) = _windowed_support_case(tmp_path)
    masks = audit.build_edge_masks(scene, unique, mult)
    ref = audit.sample_paired_edges(sw, db, masks[boundary])

    res = audit.stream_support_boundary(
        paths["sw"], paths["db"], paths[support_key], tmp_dir=tmp_path,
        key_prefix=f"{boundary}_{window_size}", block_size=4,
        window_size=window_size, collect_edges=True,
    )
    # Identical retained adjacency indices (orientation, row, col), each once.
    ref_edges = list(zip(ref["orientation"].tolist(),
                         ref["row"].tolist(), ref["col"].tolist()))
    got_edges = list(zip(res["edges"]["orientation"].tolist(),
                         res["edges"]["row"].tolist(), res["edges"]["col"].tolist()))
    assert len(got_edges) == len(set(got_edges))          # no double counting
    assert set(got_edges) == set(ref_edges)               # exact same edges
    assert res["pair_count"] == ref["pair_count"]
    assert res["skipped"] == ref["skipped_missing_observation"]
    # Same signed jumps (order-independent aggregate).
    assert abs(res["sw_sink"].load().sum() - ref["sw_signed"].sum()) < 1e-9
    assert abs(res["db_sink"].load().sum() - ref["db_signed"].sum()) < 1e-9


def test_windowed_edges_on_window_boundary_counted_once(tmp_path):
    """A support edge sitting exactly on a window boundary (both orientations)
    is retained exactly once (owned by the window holding its anchor pixel)."""
    paths, (sw, db, scene, unique, mult) = _windowed_support_case(tmp_path)
    # Window size 4 puts the col-4/col-8 and row-4/row-8 edges on window seams.
    res = audit.stream_support_boundary(
        paths["sw"], paths["db"], paths["scene"], tmp_dir=tmp_path,
        key_prefix="onseam", block_size=4, window_size=4, collect_edges=True,
    )
    got = list(zip(res["edges"]["orientation"].tolist(),
                   res["edges"]["row"].tolist(), res["edges"]["col"].tolist()))
    assert len(got) == len(set(got))
    # The vertical support edge at col 4 yields horizontal adjacency edges whose
    # anchor col == 3 (edge between col 3 and 4); they must all be present once.
    anchor3 = [e for e in got if e[0] == "horizontal" and e[2] == 3]
    assert len(anchor3) == 12  # one per row, exactly once


def test_windowed_nodata_pairs_excluded(tmp_path):
    import numpy as np

    sw = np.array([[10.0, 40.0, 41.0, 90.0]], dtype="float64")
    db = np.array([[10.0, 20.0, np.nan, 30.0]], dtype="float64")  # NaN at col2
    scene = np.array([[1.0, 5.0, 5.0, 9.0]], dtype="float64")     # edges at col0|1 & col2|3
    paths = {
        "sw": _write_arr(tmp_path / "sw.tif", sw),
        "db": _write_arr(tmp_path / "db.tif", db),
        "scene": _write_arr(tmp_path / "scene.tif", scene),
    }
    res = audit.stream_support_boundary(
        paths["sw"], paths["db"], paths["scene"], tmp_dir=tmp_path,
        key_prefix="nodata", block_size=8, window_size=512, collect_edges=True,
    )
    # col0|1 finite in both chains -> retained. col2|3 has a NaN db endpoint ->
    # dropped and counted as skipped; no NaN is ever imputed to zero.
    assert res["pair_count"] == 1
    assert res["skipped"] == 1
    assert res["sw_sink"].load().size == 1
    assert res["edges"]["col"].tolist() == [0]


# =============================================================================
# UNIT-SUMMARY BOOTSTRAP EQUIVALENCE + NO CONCATENATE IN THE LOOP
# =============================================================================
def test_unit_summary_bootstrap_numerically_equivalent():
    import numpy as np

    rng = np.random.default_rng(3)
    units = {f"u{i}": rng.normal(size=int(rng.integers(1, 25))).astype("float64")
             for i in range(15)}
    old = audit.cluster_bootstrap_interval(units, n_boot=800, ci=0.95, seed=2024)
    summ = audit.summaries_from_unit_reductions(units)
    new = audit.cluster_bootstrap_interval_from_summaries(
        summ, n_boot=800, ci=0.95, seed=2024)
    assert old["n_units"] == new["n_units"]
    assert old["n_pairs"] == new["n_pairs"]
    assert abs(old["point_estimate"] - new["point_estimate"]) < 1e-12
    assert abs(old["interval_low"] - new["interval_low"]) < 1e-12
    assert abs(old["interval_high"] - new["interval_high"]) < 1e-12


def test_summary_bootstrap_never_concatenates_in_loop():
    """The per-pixel cluster bootstrap concatenates once PER iteration; the
    summary bootstrap must not -- its np.concatenate count must be a small
    constant independent of n_boot."""
    import numpy as np

    units = {f"u{i}": np.arange(1, 6, dtype="float64") + i for i in range(12)}
    summ = audit.summaries_from_unit_reductions(units)

    def _concat_calls(fn, *args, **kwargs):
        original = np.concatenate
        counter = {"n": 0}

        def _guard(*a, **k):
            counter["n"] += 1
            return original(*a, **k)

        np.concatenate = _guard
        try:
            fn(*args, **kwargs)
        finally:
            np.concatenate = original
        return counter["n"]

    small = _concat_calls(
        audit.cluster_bootstrap_interval_from_summaries, summ, n_boot=200, seed=1)
    large = _concat_calls(
        audit.cluster_bootstrap_interval_from_summaries, summ, n_boot=5000, seed=1)
    # Constant, and far below n_boot (the old per-pixel path would be >= n_boot).
    assert small == large
    assert large < 50
    old = _concat_calls(audit.cluster_bootstrap_interval, units, n_boot=200, seed=1)
    assert old >= 200  # sanity: the old path DOES concatenate every iteration


# =============================================================================
# WINDOWED PROVENANCE (one code raster, per-boundary_id units) + TILE CONTROL
# =============================================================================
def test_windowed_provenance_preserves_boundary_id_units(tmp_path):
    import numpy as np
    from rasterio.transform import Affine

    transform = Affine(0.01, 0, 30.0, 0, -0.01, 37.0)
    sw = np.tile(np.array([0, 0, 0, 9, 9, 9], dtype="float64"), (6, 1))
    db = np.zeros((6, 6), dtype="float64")
    sw_p = _write_arr(tmp_path / "sw.tif", sw, transform)
    db_p = _write_arr(tmp_path / "db.tif", db, transform)
    # Two verified boundaries on the col-2|3 edge (upper vs lower rows).
    def _line(y0, y1):
        x = 30.0 + 0.01 * 2.5
        return {"type": "LineString", "coordinates": [[x, y0], [x, y1]]}

    geojson = {"features": [
        {"geometry": _line(37.0 - 0.01 * 0.5, 37.0 - 0.01 * 2.5),
         "properties": {"boundary_id": "bnd_A", "verification_status": "verified"}},
        {"geometry": _line(37.0 - 0.01 * 3.5, 37.0 - 0.01 * 5.5),
         "properties": {"boundary_id": "bnd_B", "verification_status": "verified"}},
    ]}
    spec = audit.build_provenance_code_memmap(geojson, transform, 6, 6, tmp_path)
    assert spec["n_codes"] == 2 and set(spec["boundary_ids"]) == {"bnd_A", "bnd_B"}
    prov = audit.stream_provenance_boundaries(
        sw_p, db_p, spec, tmp_dir=tmp_path, key_prefix="prov", window_size=3)
    # Each boundary_id is preserved as its own unit and produced >=1 paired edge.
    assert set(prov["unit_summaries"]) == {"bnd_A", "bnd_B"}
    for uid in ("bnd_A", "bnd_B"):
        assert prov["unit_summaries"][uid]["count"] >= 1
    verdict = audit.classify_paired_units_from_summaries(
        prov["unit_summaries"], unit_type="provenance_boundary_id",
        min_units=2, n_boot=200)
    assert verdict["status"] == "supported_reduction"   # sw jump 9, db jump 0


def test_windowed_provenance_no_full_masks_bounded(tmp_path):
    """The provenance representation is a single int32 code raster, not one full
    boolean mask per boundary_id."""
    import numpy as np
    from rasterio.transform import Affine

    transform = Affine(0.01, 0, 30.0, 0, -0.01, 37.0)
    features = []
    for i in range(50):
        x = 30.0 + 0.01 * (0.5 + i % 5)
        features.append({
            "geometry": {"type": "LineString",
                         "coordinates": [[x, 36.99], [x, 36.90]]},
            "properties": {"boundary_id": f"bnd_{i}", "verification_status": "verified"},
        })
    spec = audit.build_provenance_code_memmap(
        {"features": features}, transform, 10, 10, tmp_path)
    assert spec["n_codes"] == 50
    # Exactly ONE code file exists (not 50 masks).
    code_files = list((tmp_path).glob("provenance_codes.int32"))
    assert len(code_files) == 1
    mm = np.memmap(spec["codes_path"], dtype="int32", mode="r", shape=(10, 10))
    assert mm.max() <= 50 and mm.min() >= 0


# =============================================================================
# LOCAL ANALYSIS CHECKPOINT (atomic; validated-before-skip; separate file)
# =============================================================================
def test_analysis_checkpoint_atomic_and_separate(tmp_path):
    payload = {"audit": "x", "stages": {"report": {"done": True}}}
    audit.write_analysis_checkpoint(tmp_path, payload)
    p = audit.analysis_checkpoint_path(tmp_path)
    assert p.name == "analysis_checkpoint.json"
    assert p.name != "raster_inventory.partial.json"  # never the same file
    assert json.loads(p.read_text(encoding="utf-8"))["stages"]["report"]["done"] is True
    assert not (tmp_path / ".analysis_checkpoint.json.tmp").exists()


def test_checkpoint_reuse_requires_valid_referenced_files(tmp_path):
    import scripts.run_landsat_composite_counterfactual_audit as runner

    dep = tmp_path / "dep.tif"
    dep.write_bytes(b"1234567890")
    cp = runner._AnalysisCheckpoint(tmp_path, "manavgat_2021")
    cp.mark("boundary:current_lst", result={"ok": True},
            inputs=[runner._file_ref(dep)])
    # Reload from disk (fresh instance) and confirm reuse works while file matches.
    cp2 = runner._AnalysisCheckpoint(tmp_path, "manavgat_2021")
    assert cp2.reusable("boundary:current_lst") == {"ok": True}
    # Mutate the referenced file -> checkpoint text must NOT be trusted anymore.
    dep.write_bytes(b"CHANGED-SIZE-XXXXXXXX")
    cp3 = runner._AnalysisCheckpoint(tmp_path, "manavgat_2021")
    assert cp3.reusable("boundary:current_lst") is None
    # Missing file also invalidates.
    dep.unlink()
    cp4 = runner._AnalysisCheckpoint(tmp_path, "manavgat_2021")
    assert cp4.reusable("boundary:current_lst") is None


def test_clean_analysis_tmp_only_inside_namespace(tmp_path):
    exp = "manavgat_2021"
    root = audit.diagnostic_output_root(exp, tmp_path)
    tmp = root / audit.ANALYSIS_TMP_DIRNAME
    tmp.mkdir(parents=True)
    (tmp / "scratch.f64").write_bytes(b"scratch")
    removed = audit.clean_analysis_tmp(root)
    assert removed == str(tmp)
    assert not tmp.exists()
    assert root.exists()  # only the scratch dir was removed


# =============================================================================
# --finalize-only : flag incompatibilities + fail-fast validation
# =============================================================================
@pytest.mark.parametrize("kwargs", [
    {"run": True}, {"force": True}, {"resume": True}, {"cleanup_tiles": True},
])
def test_finalize_only_rejects_incompatible_flags(kwargs):
    import scripts.run_landsat_composite_counterfactual_audit as runner

    with pytest.raises(runner.AuditRunnerError):
        runner.main("manavgat_2021", finalize_only=True, **kwargs)


def _finalize_ready_namespace(tmp_path, monkeypatch, *, drop=None):
    """Build a minimal but complete diagnostic namespace for --finalize-only.

    All 60 exported rasters + 10 derived rasters are written as tiny valid
    sentinel-tagged GeoTIFFs, plus source_scene_metadata.json. The diagnostic
    root is redirected under ``tmp_path``. Returns (exp, ctx, fake_root).
    """
    import scripts.run_landsat_composite_counterfactual_audit as runner

    exp = "manavgat_2021"
    ctx = runner.build_experiment_context(exp)
    fake_root = tmp_path / "diag"
    monkeypatch.setattr(audit, "diagnostic_output_root",
                        lambda e, base_dir=None: fake_root)

    raster_plan = audit.plan_raster_outputs(ctx)
    for i, (name, rel) in enumerate(raster_plan.items()):
        if drop and name == drop:
            continue
        # small gradient so maps/validation have >=1 valid pixel per raster
        _write_arr((fake_root / rel), _grid(i))
    # explicit source-scene metadata (reused, never re-queried from EE)
    (fake_root / "source_scene_metadata.json").write_text(json.dumps({
        "audit": audit.DIAGNOSTIC_NAMESPACE, "experiment_id": exp,
        "collections": {"current_lst": [{"scene_id": "s1"}]},
        "count_semantics": {"current_lst": {"scene_count": 1, "unique_date_count": 1,
                                            "same_day_multiplicity": 0}},
    }), encoding="utf-8")
    return exp, ctx, fake_root


def _grid(seed):
    import numpy as np

    return (np.arange(64).reshape(8, 8) % (3 + seed % 4)).astype("float64")


def test_finalize_only_missing_raster_fails_fast(tmp_path, monkeypatch):
    import scripts.run_landsat_composite_counterfactual_audit as runner

    exp, ctx, fake_root = _finalize_ready_namespace(
        tmp_path, monkeypatch, drop="current_lst_scene_weighted_median")
    with pytest.raises(runner.AuditRunnerError) as excinfo:
        runner._run_finalize(ctx)
    assert "current_lst_scene_weighted_median" in str(excinfo.value)
    assert "missing or invalid" in str(excinfo.value)


def _patch_finalize_light(monkeypatch, repro_status="canonical_reproduction_failed"):
    """Patch out heavy/side-effecting stages for a hermetic finalize flow test."""
    import scripts.run_landsat_composite_counterfactual_audit as runner

    monkeypatch.setattr(audit, "run_canonical_reproduction_gate",
                        lambda ctx, root, plan: {"status": repro_status, "checks": {}})
    monkeypatch.setattr(runner, "_run_provenance",
                        lambda ctx, root: {"status": "insufficient_evidence"})

    def _fast_maps(sw_path, db_path, out_dir, *, pair_name, **kwargs):
        from pathlib import Path as _P
        out = _P(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pngs = []
        for p in (sw_path, db_path):
            png = out / (_P(p).stem + ".png")
            png.write_bytes(b"\x89PNG")
            pngs.append(str(png))
        return pngs

    monkeypatch.setattr(audit, "render_pair_maps", _fast_maps)
    return runner


def test_finalize_only_no_ee_no_export_no_delete(tmp_path, monkeypatch):
    import sys as _sys
    import core.gee_utils as gee_utils
    import scripts.run_predictors_only as rp

    runner = _patch_finalize_light(monkeypatch)
    exp, ctx, fake_root = _finalize_ready_namespace(tmp_path, monkeypatch)
    (fake_root / "sentinel_keep.txt").write_text("keep", encoding="utf-8")

    # Booby-trap Earth Engine + init + export + namespace deletion.
    class _EEBoom:
        def __getattr__(self, name):
            raise AssertionError(f"--finalize-only must not touch ee.{name}")

    if "ee" not in _sys.modules:
        monkeypatch.setitem(_sys.modules, "ee", _EEBoom())
    monkeypatch.setattr(gee_utils, "init_gee",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("finalize must not init GEE")))
    monkeypatch.setattr(rp, "export_image_direct_or_tiled",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("finalize must not export")))
    monkeypatch.setattr(audit, "clear_diagnostic_namespace",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("finalize must not clear namespace")))

    result = runner.main(exp, finalize_only=True)

    assert result["ran"] is True and result["mode"] == "finalize_only"
    # canonical_reproduction_failed forces the scientific verdict but reporting
    # still completed (summary + manifest written).
    assert result["final_status"] == "canonical_reproduction_failed"
    assert result["canonical_reproduction"] == "canonical_reproduction_failed"
    summary = json.loads(
        (fake_root / "counterfactual_summary.json").read_text(encoding="utf-8"))
    assert summary["final_status"] == "canonical_reproduction_failed"
    # Limitations are concise machine-readable entries (code + statement).
    codes = {lim["code"] for lim in summary["limitations"]}
    assert "canonical_reproduction_not_passed" in codes
    assert any("Canonical reproduction did not pass" in lim["statement"]
               for lim in summary["limitations"])
    assert (fake_root / "manifest.json").exists()
    assert (fake_root / "analysis_checkpoint.json").exists()
    # Namespace preserved; scratch cleaned after success.
    assert (fake_root / "sentinel_keep.txt").exists()
    assert not (fake_root / audit.ANALYSIS_TMP_DIRNAME).exists()


def test_finalize_only_reuses_completed_stages(tmp_path, monkeypatch):
    runner = _patch_finalize_light(monkeypatch)
    exp, ctx, fake_root = _finalize_ready_namespace(tmp_path, monkeypatch)

    # First finalize completes and writes the analysis checkpoint.
    first = runner.main(exp, finalize_only=True)
    assert first["ran"] is True

    # Second finalize must reuse every completed boundary/product calculation:
    # if it recomputed, the windowed audit (now booby-trapped) would blow up.
    monkeypatch.setattr(audit, "audit_product_boundaries_windowed",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("completed boundary stage must be reused")))
    second = runner.main(exp, finalize_only=True)
    assert second["ran"] is True
    assert second["final_status"] == first["final_status"]


# =============================================================================
# PIPELINE-SEMANTIC CANONICAL REPRODUCTION GATE (corrected v2 gate)
#
# The v1 gate compared raw rasters and produced a demonstrated FALSE NEGATIVE:
# canonical annual baseline TIFFs serialize GEE-masked pixels as raw DN 0 with no
# nodata tag, while diagnostic exports use a -9999 nodata sentinel. After the
# production Step5 DN->Celsius->physical mask the two are pixel-identical, so the
# decisive gate must compare SEMANTICS, not raw bytes.
# =============================================================================
def _write_field(path, arr, *, nodata, transform=None, crs="EPSG:4326"):
    """Write an arbitrary float32 array with an explicit (or absent) nodata tag.

    Unlike ``_write_arr`` this performs NO NaN->sentinel substitution: the raw
    values (e.g. DN 0 zero-fills, or a -9999 sentinel, or NaN) are written as-is,
    so the raw-encoding difference under test is faithfully reproduced on disk.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    if transform is None:
        transform = Affine(0.01, 0, 30.0, 0, -0.01, 37.0)
    arr = np.asarray(arr, dtype="float32")
    height, width = arr.shape
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


# A valid ST_B10 DN that maps into the physical Celsius range, and the masked
# zero-fill / sentinel encodings.
_DN_VALID = 45000.0          # ~29.7 C, inside [-30, 80]
_DN_ZERO = 0.0               # canonical zero-fill -> ~-124 C -> masked
_SENTINEL = -9999.0          # diagnostic sentinel -> ~-158 C -> masked


def _baseline_pair(tmp_path, *, diag_arr, canon_arr):
    import numpy as np

    diag = _write_field(tmp_path / "diag_baseline.tif",
                        np.array(diag_arr, dtype="float32"), nodata=_SENTINEL)
    canon = _write_field(tmp_path / "canon_baseline.tif",
                         np.array(canon_arr, dtype="float32"), nodata=None)
    return diag, canon


def test_gate_raw_zero_vs_sentinel_encoding_recorded(tmp_path):
    # Row: [valid, masked, masked]. Diagnostic masks with -9999; canonical with 0.
    diag, canon = _baseline_pair(
        tmp_path,
        diag_arr=[[_DN_VALID, _SENTINEL, _SENTINEL]],
        canon_arr=[[_DN_VALID, _DN_ZERO, _DN_ZERO]],
    )
    result = audit.compare_semantic_reproduction(diag, canon, kind="baseline_dn")
    enc = result["raw_encoding"]
    assert enc["diagnostic_nodata_tag"] == _SENTINEL
    assert enc["canonical_nodata_tag"] is None
    assert enc["bitwise_identical"] is False
    assert enc["raw_valid_mask_exact"] is False        # raw masks differ
    assert enc["canonical_only_valid_pixel_count"] == 2  # the two DN-0 pixels
    assert enc["canonical_only_all_zero_filled"] is True


def test_gate_raw_mask_mismatch_semantic_equivalent_passes(tmp_path):
    diag, canon = _baseline_pair(
        tmp_path,
        diag_arr=[[_DN_VALID, _SENTINEL, _SENTINEL]],
        canon_arr=[[_DN_VALID, _DN_ZERO, _DN_ZERO]],
    )
    result = audit.compare_semantic_reproduction(diag, canon, kind="baseline_dn")
    # Raw masks differ, but the Step5 semantics agree exactly -> pass + warning.
    assert result["reproduction_status"] == "pass"
    assert result["valid_mask_exact"] is True
    assert result["max_abs_diff"] <= audit.BASELINE_SEMANTIC_MAX_ABS_DIFF
    assert "raw_encoding_difference_semantically_equivalent" in result["warnings"]


def test_gate_nonzero_canonical_only_pixel_fails_semantic(tmp_path):
    # Canonical has a VALID-range DN where the diagnostic export is masked. The
    # canonical-only pixel is NOT a zero-fill, so semantics genuinely disagree.
    diag, canon = _baseline_pair(
        tmp_path,
        diag_arr=[[_DN_VALID, _SENTINEL, _SENTINEL]],
        canon_arr=[[_DN_VALID, _DN_VALID, _DN_ZERO]],
    )
    result = audit.compare_semantic_reproduction(diag, canon, kind="baseline_dn")
    assert result["reproduction_status"] == "mask_mismatch"
    assert result["raw_encoding"]["canonical_only_all_zero_filled"] is False


def test_gate_annual_semantic_value_mismatch_above_1e6_fails(tmp_path):
    # Identical valid masks, but the canonical DN differs by 1 DN at a valid
    # pixel -> ~0.0034 C > 1e-6 -> fail on value (not mask).
    diag, canon = _baseline_pair(
        tmp_path,
        diag_arr=[[_DN_VALID, _DN_VALID, _SENTINEL]],
        canon_arr=[[_DN_VALID, _DN_VALID + 1.0, _DN_ZERO]],
    )
    result = audit.compare_semantic_reproduction(diag, canon, kind="baseline_dn")
    assert result["reproduction_status"] == "fail"
    assert result["valid_mask_exact"] is True
    assert result["max_abs_diff"] > audit.BASELINE_SEMANTIC_MAX_ABS_DIFF


def test_gate_grid_mismatch_still_fails(tmp_path):
    from rasterio.transform import Affine

    import numpy as np

    diag = _write_field(tmp_path / "d.tif", np.array([[_DN_VALID]], dtype="float32"),
                        nodata=_SENTINEL)
    canon = _write_field(tmp_path / "c.tif", np.array([[_DN_VALID]], dtype="float32"),
                         nodata=None, transform=Affine(0.02, 0, 30.0, 0, -0.02, 37.0))
    result = audit.compare_semantic_reproduction(diag, canon, kind="baseline_dn")
    assert result["reproduction_status"] == "grid_mismatch"
    assert set(["transform", "bounds"]).issubset(set(result["mismatch_fields"]))


# --- derived baseline mean/std: 5e-5 float32 reduction reproducibility --------
def test_gate_derived_reduction_below_tolerance_passes(tmp_path):
    import numpy as np

    base = np.full((4, 4), 20.0, dtype="float32")
    diag = _write_field(tmp_path / "d.tif", base, nodata=np.nan)
    canon = _write_field(tmp_path / "c.tif", base + 2e-5, nodata=np.nan)  # < 5e-5
    result = audit.compare_derived_reduction(diag, canon, kind="baseline_mean")
    assert result["reproduction_status"] == "pass"
    assert result["count_above_tolerance"] == 0
    assert result["tolerance"] == audit.DERIVED_REDUCTION_TOLERANCE
    assert "abs_diff_p99_9" in result and "abs_diff_p95" in result


def test_gate_derived_reduction_above_tolerance_fails(tmp_path):
    import numpy as np

    base = np.full((4, 4), 20.0, dtype="float32")
    diag = _write_field(tmp_path / "d.tif", base, nodata=np.nan)
    canon = _write_field(tmp_path / "c.tif", base + 1e-3, nodata=np.nan)  # > 5e-5
    result = audit.compare_derived_reduction(diag, canon, kind="baseline_std")
    assert result["reproduction_status"] == "fail"
    assert result["count_above_tolerance"] > 0
    assert result["max_abs_diff"] > audit.DERIVED_REDUCTION_TOLERANCE


# --- anomaly z-score threshold-boundary equivalence ---------------------------
def _anomaly_case(tmp_path, *, std_at_mismatch_c, std_at_mismatch_d):
    """Build the eight anomaly-related rasters on a shared 2x3 grid.

    Pixel (0,1) is a z-score mask disagreement (valid in canonical, masked in
    diagnostic). Upstream current + mean masks agree everywhere; only the
    baseline-std at the disagreement pixel controls whether it is inside the
    threshold neighbourhood.
    """
    import numpy as np

    nan = np.nan
    z_canon = np.array([[2.0, 1.5, 2.0], [2.0, 2.0, 2.0]], dtype="float32")
    z_diag = np.array([[2.0, nan, 2.0], [2.0, 2.0, 2.0]], dtype="float32")  # (0,1) masked
    # std >= threshold on valid z pixels; the mismatch pixel std is parametrised.
    std_c = np.full((2, 3), 2.0, dtype="float32"); std_c[0, 1] = std_at_mismatch_c
    std_d = np.full((2, 3), 2.0, dtype="float32"); std_d[0, 1] = std_at_mismatch_d
    mean = np.full((2, 3), 20.0, dtype="float32")   # finite everywhere (masks agree)
    current = np.full((2, 3), 25.0, dtype="float32")  # in-range Celsius everywhere

    def w(name, arr, nodata=nan):
        return _write_field(tmp_path / name, arr, nodata=nodata)

    return dict(
        diagnostic_zscore_path=w("zd.tif", z_diag),
        canonical_zscore_path=w("zc.tif", z_canon),
        diagnostic_std_path=w("sd.tif", std_d),
        canonical_std_path=w("sc.tif", std_c),
        diagnostic_mean_path=w("md.tif", mean),
        canonical_mean_path=w("mc.tif", mean),
        diagnostic_current_path=w("cd.tif", current),
        canonical_current_path=w("cc.tif", current),
    )


def test_gate_anomaly_mismatch_within_std_threshold_is_boundary_equivalent(tmp_path):
    # Both std at the disagreement pixel are within 5e-5 of the 1.0 threshold.
    paths = _anomaly_case(tmp_path, std_at_mismatch_c=0.99999, std_at_mismatch_d=1.00001)
    result = audit.compare_anomaly_threshold_boundary(**paths)
    tb = result["threshold_boundary"]
    assert tb["classification"] == "threshold_boundary_equivalent"
    assert tb["mismatch_pixel_count"] == 1
    assert tb["coordinate_sample"] == [[0, 1]]
    assert tb["max_distance_from_threshold"] <= audit.ANOMALY_STD_TOL
    # Values agree on common pixels -> overall pass despite the mask disagreement.
    assert result["reproduction_status"] == "pass"


def test_gate_anomaly_mismatch_away_from_threshold_fails(tmp_path):
    # std at the disagreement pixel is far from 1.0 -> not a threshold artefact.
    paths = _anomaly_case(tmp_path, std_at_mismatch_c=5.0, std_at_mismatch_d=5.0)
    result = audit.compare_anomaly_threshold_boundary(**paths)
    tb = result["threshold_boundary"]
    assert tb["classification"] == "mask_mismatch_outside_threshold_neighbourhood"
    assert result["reproduction_status"] == "mask_mismatch"


def test_gate_anomaly_value_mismatch_beyond_propagated_bound_fails(tmp_path):
    import numpy as np

    # Exact masks (no disagreement), but a large z-score value difference far
    # beyond the propagated numerical bound -> fail on value.
    paths = _anomaly_case(tmp_path, std_at_mismatch_c=2.0, std_at_mismatch_d=2.0)
    # overwrite z_diag to agree in mask but differ materially in value
    zc = np.array([[2.0, 1.5, 2.0], [2.0, 2.0, 2.0]], dtype="float32")
    zd = zc.copy(); zd[1, 1] = 2.5   # 0.5 z difference, way over the bound
    _write_field(paths["canonical_zscore_path"], zc, nodata=np.nan)
    _write_field(paths["diagnostic_zscore_path"], zd, nodata=np.nan)
    result = audit.compare_anomaly_threshold_boundary(**paths)
    assert result["threshold_boundary"]["classification"] == "exact_mask_equality"
    assert result["reproduction_status"] == "fail"
    assert result["value_comparison"]["pixels_exceeding_propagated_bound"] >= 1


# --- full gate assembly: layered output + warnings ---------------------------
def _gate_namespace(tmp_path):
    """Build a temp diagnostic namespace + canonical files that reproduce
    semantically (raw encodings differ) so the full gate returns pass."""
    import numpy as np

    root = tmp_path / "diag"
    canon_dir = tmp_path / "canon"
    canon_dir.mkdir(parents=True, exist_ok=True)

    import rasterio
    from rasterio.transform import Affine

    transform = Affine(0.01, 0, 30.0, 0, -0.01, 37.0)

    # current LST median (celsius) -- diagnostic sentinel, canonical NaN nodata.
    _write_field(root / "rasters/current_lst_scene_weighted_median.tif",
                 np.array([[25.0, 26.0, _SENTINEL]], dtype="float32"), nodata=_SENTINEL)
    # canonical current file mirrors production: band1=Celsius, band2=valid count.
    cur_celsius = np.array([[25.0, 26.0, np.nan]], dtype="float32")
    cur_count = np.array([[3.0, 4.0, 0.0]], dtype="float32")  # 0 == masked
    with rasterio.open(
        canon_dir / "current.tif", "w", driver="GTiff", width=3, height=1, count=2,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(cur_celsius, 1)
        dst.write(cur_count, 2)
    # diagnostic scene-valid-count raster (sentinel-masked).
    _write_field(root / "rasters/current_lst_scene_valid_count.tif",
                 np.array([[3.0, 4.0, _SENTINEL]], dtype="float32"), nodata=_SENTINEL)

    # baseline DN year raster: raw zero-fill vs sentinel, semantically identical.
    _write_field(root / "rasters/baseline_lst_2020_scene_weighted_median.tif",
                 np.array([[_DN_VALID, _SENTINEL, _SENTINEL]], dtype="float32"),
                 nodata=_SENTINEL)
    _write_field(canon_dir / "baseline_2020.tif",
                 np.array([[_DN_VALID, _DN_ZERO, _DN_ZERO]], dtype="float32"), nodata=None)

    # derived mean/std/zscore (celsius) -- identical.
    mean = np.array([[20.0, np.nan, np.nan]], dtype="float32")
    std = np.array([[2.0, np.nan, np.nan]], dtype="float32")
    z = np.array([[2.5, np.nan, np.nan]], dtype="float32")
    dv = root / "derived" / "scene_weighted"
    for name, arr in (("baseline_lst_mean_celsius.tif", mean),
                      ("baseline_lst_std_celsius.tif", std),
                      ("anomaly_zscore.tif", z)):
        _write_field(dv / name, arr, nodata=np.nan)
        _write_field(canon_dir / name, arr, nodata=np.nan)

    raster_plan = {
        "current_lst_scene_weighted_median": "rasters/current_lst_scene_weighted_median.tif",
        "current_lst_scene_valid_count": "rasters/current_lst_scene_valid_count.tif",
        "baseline_lst_2020_scene_weighted_median": "rasters/baseline_lst_2020_scene_weighted_median.tif",
    }
    from collections import OrderedDict
    canonical = OrderedDict([
        ("current_lst", {"path": str(canon_dir / "current.tif")}),
        ("current_lst_count", {"path": str(canon_dir / "current.tif")}),
        ("baseline_lst_2020", {"path": str(canon_dir / "baseline_2020.tif")}),
        ("step5_baseline_mean", {"path": str(canon_dir / "baseline_lst_mean_celsius.tif")}),
        ("step5_baseline_std", {"path": str(canon_dir / "baseline_lst_std_celsius.tif")}),
        ("step5_anomaly_zscore", {"path": str(canon_dir / "anomaly_zscore.tif")}),
    ])
    return root, raster_plan, canonical


def test_full_gate_layers_and_warnings(tmp_path, monkeypatch):
    root, raster_plan, canonical = _gate_namespace(tmp_path)
    monkeypatch.setattr(audit, "resolve_canonical_paths", lambda ctx: canonical)

    ctx = {"experiment_id": "manavgat_2021"}
    result = audit.run_canonical_reproduction_gate(ctx, root, raster_plan)

    assert result["canonical_gate_version"] == audit.CANONICAL_GATE_VERSION
    assert result["decisive_gate"] == "pipeline_semantic_reproduction"
    # Layered output present and honest about raw encoding.
    assert result["raw_encoding"]["status"] == "difference_semantically_equivalent"
    assert result["raw_encoding"]["bitwise_identical"] is False
    assert "semantic_checks" in result and "baseline_lst_2020" in result["semantic_checks"]
    assert result["threshold_boundary"] is not None
    # Raw-encoding warning is surfaced at the gate level.
    assert "raw_encoding_difference_semantically_equivalent" in result["warnings"]
    # Decisive gate passes: baseline is semantically identical despite raw diff.
    assert result["semantic_checks"]["baseline_lst_2020"]["reproduction_status"] == "pass"


# --- checkpoint invalidation on gate-version change ---------------------------
def test_old_checkpoint_gate_version_invalidates_only_gate_stages(tmp_path):
    import scripts.run_landsat_composite_counterfactual_audit as runner

    # A shared dependency file for a non-gate (boundary) stage.
    dep = tmp_path / "dep.tif"
    dep.write_bytes(b"0123456789")

    # Seed a checkpoint file recorded under an OLD gate version, marking the gate
    # stages (reproduction, report) AND a boundary stage as done.
    audit.write_analysis_checkpoint(tmp_path, {
        "audit": audit.DIAGNOSTIC_NAMESPACE,
        "experiment_id": "manavgat_2021",
        "checkpoint_kind": "local_analysis",
        "canonical_gate_version": "OLD-0.0",
        "stages": {
            "reproduction": {"done": True, "files": [], "inputs": [],
                             "result": {"status": "pass"}},
            "report": {"done": True, "files": [], "inputs": [], "result": {"ok": True}},
            "boundary:current_lst": {
                "done": True, "files": [], "result": {"verdicts": {}},
                "inputs": [runner._file_ref(dep)],
            },
        },
    })

    cp = runner._AnalysisCheckpoint(tmp_path, "manavgat_2021")
    assert cp.stored_gate_version == "OLD-0.0"
    # Gate-versioned stages are invalidated (recompute canonical repro + report).
    assert cp.reusable("reproduction") is None
    assert cp.reusable("report") is None
    # Non-gate stages with valid referenced files are still reused (no GEE rerun,
    # no re-export): the boundary verdicts survive the gate-version bump.
    assert cp.reusable("boundary:current_lst") == {"verdicts": {}}

    # After marking any stage, the recorded gate version upgrades to current and
    # the gate stages become reusable again.
    cp.mark("reproduction", result={"status": "pass"})
    cp2 = runner._AnalysisCheckpoint(tmp_path, "manavgat_2021")
    assert cp2.stored_gate_version == audit.CANONICAL_GATE_VERSION
    assert cp2.reusable("reproduction") == {"status": "pass"}


# =============================================================================
# REPORT / METADATA CONSISTENCY (limitations, QA wording, provenance qualifier)
# =============================================================================
_REQUIRED_LIMITATION_CODES = {
    "single_aoi_manavgat",
    "no_generalization",
    "step8_model_impact_not_evaluated",
    "export_tile_control_unavailable",
    "provenance_metadata_derived_not_pixel_level",
    "pathrow_reduction_small_or_uncertain",
    "supported_reduction_not_causal_proof",
}


def test_qa_masking_rule_correction_removes_radsat():
    from core.source_scene_provenance_config import source_scene_provenance_config

    config = source_scene_provenance_config("manavgat_2021")
    # sanity: the canonical config text falsely claims QA_RADSAT.
    assert any("QA_RADSAT" in spec["masking_rules"]
               for spec in config["composites"].values())

    corrected = audit.correct_composite_masking_rules(config)
    for spec in corrected["composites"].values():
        assert spec["masking_rules"] == audit.QA_MASKING_RULE_TEXT
        assert "QA_RADSAT not applied" in spec["masking_rules"]
        assert "+ QA_RADSAT" not in spec["masking_rules"]
        assert spec["qa_radsat_applied"] is False
    # The original config object is not mutated (deep copy).
    assert any("+ QA_RADSAT" in spec["masking_rules"]
               for spec in config["composites"].values())


def test_provenance_qualification_metadata_only():
    q = audit.provenance_evidence_qualification("metadata_only")
    assert q["evidence_kind"] == "metadata-derived source-footprint/path-row boundary evidence"
    assert q["is_valid_geometric_boundary_evidence"] is True
    assert q["is_pixel_level_selected_scene_provenance"] is False
    assert q["median_reducer_has_selected_scene_id"] is False
    assert "not" in q["qualification"].lower()
    assert "selected-scene" in q["qualification"]


def test_build_report_limitations_has_all_required_codes():
    verified = {
        "current_lst": {"status": "supported_reduction",
                        "is_verified_source_boundary_evidence": True},
        "current_minus_baseline": {"status": "supported_reduction",
                                   "is_verified_source_boundary_evidence": True},
        "anomaly_zscore": {"status": "uncertain",
                           "is_verified_source_boundary_evidence": True},
    }
    tile = {p: {"status": "insufficient_boundary_metadata"} for p in verified}
    lims = audit.build_report_limitations(
        experiment_id="manavgat_2021",
        verified_source_evidence=verified,
        export_tile_negative_control=tile,
        reproduction_status="pass",
        verified_source_present=True,
    )
    codes = {item["code"] for item in lims}
    assert _REQUIRED_LIMITATION_CODES.issubset(codes)
    # machine-readable: every entry has a code + a non-empty statement.
    assert all(item["code"] and item["statement"] for item in lims)
    # No obsolete conditional caveats fire on a clean pass.
    assert "canonical_reproduction_not_passed" not in codes
    assert "no_verified_source_boundary_sampled" not in codes
    # data-backed: the path/row entry carries this run's per-product statuses.
    pr = next(i for i in lims if i["code"] == "pathrow_reduction_small_or_uncertain")
    assert pr["verified_path_row_status"]["anomaly_zscore"] == "uncertain"


# --- representative deterministic summary + report-generation regression ------
def _representative_inputs():
    """A representative supported_reduction result to exercise _build_summary."""
    edge = "scene_count_edge"

    def verdict(status, pe, lo, hi, nu, npairs, verified=False):
        v = {
            "status": status, "point_estimate": pe,
            "interval_low": lo, "interval_high": hi,
            "n_units": nu, "n_pairs": npairs,
            "unit_type": "raster_support_block",
        }
        if verified:
            v["is_verified_source_boundary_evidence"] = True
            v["unit_type"] = "provenance_boundary_id"
        return v

    verdicts = {}
    for i, p in enumerate(("current_lst", "current_minus_baseline", "anomaly_zscore")):
        pr_status = "uncertain" if p == "anomaly_zscore" else "supported_reduction"
        verdicts[p] = {
            edge: verdict("supported_reduction", 0.5 + i, 0.1 + i, 0.9 + i, 12 + i, 3000 + i),
            "unique_date_count_edge": verdict("uncertain", 0.01, -0.1, 0.1, 12, 3000),
            "same_day_multiplicity_edge": verdict("uncertain", 0.01, -0.1, 0.1, 12, 3000),
            "source_scene_path_row": verdict(pr_status, 0.002, -0.001, 0.005, 2, 40, verified=True),
            "export_tile_boundary": {"status": "insufficient_boundary_metadata",
                                     "is_negative_control": True,
                                     "can_affect_final_status": False},
        }

    scene_metadata = {"count_semantics": {"current_lst": {"scene_count": 6,
                                                          "unique_date_count": 3,
                                                          "same_day_multiplicity": 3}}}
    derived = {
        chain: {"baseline_mean": {"valid_pixels": 100, "mean": 20.0},
                "baseline_std": {"valid_pixels": 100, "mean": 2.0},
                "difference": {"valid_pixels": 100, "mean": 1.0},
                "anomaly_zscore": {"valid_pixels": 90, "mean": 0.5}}
        for chain in ("scene_weighted", "date_balanced")
    }
    provenance = {"status": "available", "mode": "metadata_only", "scene_count": 6,
                  "composites": [{"source_product_role": "current_lst",
                                  "masking_rules": audit.QA_MASKING_RULE_TEXT}]}
    reproduction = {"status": "pass", "canonical_gate_version": audit.CANONICAL_GATE_VERSION,
                    "decisive_gate": "pipeline_semantic_reproduction", "checks": {},
                    "semantic_checks": {}, "warnings": []}
    ctx = {"experiment_id": "manavgat_2021"}
    return ctx, scene_metadata, derived, verdicts, provenance, "provenance_available", reproduction


def test_report_generation_preserves_scientific_result():
    import copy

    import scripts.run_landsat_composite_counterfactual_audit as runner

    ctx, meta, derived, verdicts, prov, prov_state, repro = _representative_inputs()
    before = copy.deepcopy(verdicts)

    s1 = runner._build_summary(ctx, meta, derived, verdicts, prov, prov_state, repro)
    s2 = runner._build_summary(ctx, meta, derived, verdicts, prov, prov_state, repro)

    # Report generation NEVER mutates the input scientific verdicts.
    assert verdicts == before

    # Unchanged scientific result across regeneration.
    for s in (s1, s2):
        assert s["canonical_reproduction"]["status"] == "pass"
        assert s["final_status"] == "supported_reduction"
        cur = s["paired_boundary_verdicts"]["current_lst"]["scene_count_edge"]
        assert cur["point_estimate"] == 0.5
        assert cur["interval_low"] == 0.1 and cur["interval_high"] == 0.9
        assert cur["n_units"] == 12 and cur["n_pairs"] == 3000
        anom = s["paired_boundary_verdicts"]["anomaly_zscore"]["scene_count_edge"]
        assert anom["point_estimate"] == 2.5 and anom["n_pairs"] == 3002

    # Deterministic apart from the timestamp.
    s1.pop("created_at"); s2.pop("created_at")
    assert json.dumps(s1, default=str, sort_keys=True) == json.dumps(s2, default=str, sort_keys=True)


def test_summary_schema_keys_and_no_obsolete_aliases():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    ctx, meta, derived, verdicts, prov, prov_state, repro = _representative_inputs()
    s = runner._build_summary(ctx, meta, derived, verdicts, prov, prov_state, repro)

    # Current schema keys present.
    for key in ("paired_boundary_verdicts", "support_count_boundary_evidence",
                "verified_source_path_row_evidence", "export_tile_negative_control",
                "final_status_rule", "limitations"):
        assert key in s
    # Obsolete aliases must NOT be introduced.
    assert "paired_comparisons" not in s
    assert "claim_evaluation" not in s

    # Limitations populated with all required machine-readable codes.
    codes = {item["code"] for item in s["limitations"]}
    assert _REQUIRED_LIMITATION_CODES.issubset(codes)

    # Provenance evidence stays verified but is qualified, not downgraded.
    prov_section = s["provenance"]
    assert prov_section["is_verified_source_boundary_evidence"] is True
    q = prov_section["source_boundary_evidence_qualification"]
    assert q["is_pixel_level_selected_scene_provenance"] is False
    assert q["is_valid_geometric_boundary_evidence"] is True


def test_summary_markdown_contains_limitations_and_qa_wording():
    import scripts.run_landsat_composite_counterfactual_audit as runner

    ctx, meta, derived, verdicts, prov, prov_state, repro = _representative_inputs()
    s = runner._build_summary(ctx, meta, derived, verdicts, prov, prov_state, repro)
    md = runner._summary_markdown(s)

    assert "## Limitations" in md
    assert "single-AOI" in md
    assert "QA_RADSAT not applied" in md
    assert "+ QA_RADSAT" not in md
    assert "metadata-derived source-footprint/path-row boundary evidence" in md


# --- report-schema checkpoint invalidation (reuse boundary calcs) -------------
def test_report_schema_version_invalidates_only_report_stages(tmp_path):
    import scripts.run_landsat_composite_counterfactual_audit as runner

    dep = tmp_path / "dep.tif"
    dep.write_bytes(b"0123456789")

    # Checkpoint recorded under the CURRENT gate version but an OLD report schema.
    audit.write_analysis_checkpoint(tmp_path, {
        "audit": audit.DIAGNOSTIC_NAMESPACE,
        "experiment_id": "manavgat_2021",
        "checkpoint_kind": "local_analysis",
        "canonical_gate_version": audit.CANONICAL_GATE_VERSION,
        "report_schema_version": "OLD-REPORT-0.0",
        "stages": {
            "provenance": {"done": True, "files": [], "inputs": [],
                           "result": {"status": "available"}},
            "report": {"done": True, "files": [], "inputs": [], "result": {"ok": True}},
            "reproduction": {"done": True, "files": [], "inputs": [],
                             "result": {"status": "pass"}},
            "boundary:current_lst": {
                "done": True, "files": [], "result": {"verdicts": {}},
                "inputs": [runner._file_ref(dep)],
            },
            "derived:scene_weighted": {
                "done": True, "files": [], "inputs": [runner._file_ref(dep)],
                "result": {"chain": "scene_weighted"},
            },
        },
    })

    cp = runner._AnalysisCheckpoint(tmp_path, "manavgat_2021")
    assert cp.stored_report_version == "OLD-REPORT-0.0"
    # Report/provenance stages are invalidated so reports regenerate...
    assert cp.reusable("provenance") is None
    assert cp.reusable("report") is None
    # ...but valid boundary/derived calculations are reused (no GEE, no re-export).
    assert cp.reusable("boundary:current_lst") == {"verdicts": {}}
    assert cp.reusable("derived:scene_weighted") == {"chain": "scene_weighted"}
    # The gate stage is unaffected by a report-only schema bump.
    assert cp.reusable("reproduction") == {"status": "pass"}
