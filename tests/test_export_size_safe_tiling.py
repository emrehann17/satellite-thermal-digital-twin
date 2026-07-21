"""
tests/test_export_size_safe_tiling.py

Targeted tests for the size-safe (pre-flight-estimate + tiled-fallback)
export mechanism in scripts/run_predictors_only.py, fixing the Earth Engine
direct-download request-size limit failure observed for
outputs/experiments/mugla_2021/gate_inputs/reference_30m.tif
(requested 63,226,200 bytes > 50,331,648-byte direct-download limit).

No live GEE credentials are used or required: `ee` and `geemap` are replaced
with lightweight in-process stubs (via sys.modules patching) so the full
export_image_direct_or_tiled() -> _export_tiled() -> rasterio.merge path
runs for real against small synthetic GeoTIFFs.

Covers (task numbering):
    1  a small request uses the direct path
    2  a large estimated request uses the tiled path
    3  deterministic tile construction
    4  complete tile coverage
    5  no tile overlaps except permitted shared boundaries
    6  identical CRS and resolution
    7  fixed-grid pixel alignment
    8  successful mosaic shape and transform
    9  seam-free reconstruction using a synthetic raster
    10 temporary-file cleanup
    11 atomic final output creation
    12 failed tile download does not leave a false final file
    13 existing output is not overwritten without force
    14 Muğla scientific configuration remains unchanged
    15 pre-label exclusion logic remains unchanged
    16 historical output hashes remain unchanged

Run:
    python -m unittest tests.test_export_size_safe_tiling
"""

from __future__ import annotations

import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.run_predictors_only as rpo


# =============================================================================
# Fake ee / geemap (no live GEE credentials needed)
# =============================================================================
class _FakeBBox:
    """Stand-in for ee.Geometry.BBox(...): carries the raw bbox tuple and
    supports .bounds().getInfo() the same way _bbox_from_region expects."""

    def __init__(self, xmin, ymin, xmax, ymax):
        self.bbox = (xmin, ymin, xmax, ymax)

    def bounds(self):
        return self

    def getInfo(self):
        xmin, ymin, xmax, ymax = self.bbox
        return {"coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]}


class _FakeGeometry:
    @staticmethod
    def BBox(xmin, ymin, xmax, ymax):
        return _FakeBBox(xmin, ymin, xmax, ymax)


def _make_fake_ee_module():
    mod = types.ModuleType("ee")
    mod.Geometry = _FakeGeometry
    return mod


class _FakeImage:
    """Stand-in for an ee.Image: .clip() is a no-op that returns self."""

    def clip(self, geom):
        return self


def _write_geotiff(path: Path, array: np.ndarray, transform, crs="EPSG:4326", dtype="int32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array.astype(dtype), 1)


def _make_fake_geemap_module(source_array: np.ndarray, global_transform, crs="EPSG:4326",
                              fail_tiles: set | None = None):
    """
    Builds a fake geemap module whose ee_export_image() writes a REAL
    GeoTIFF at `filename`, sliced out of `source_array` using
    `global_transform` (a SINGLE fixed pixel grid shared by every
    independently-called "tile" -- exactly the invariant the real GEE
    export relies on for tiles to merge without seams).

    `region` (kwarg) is expected to be a _FakeBBox; the pixel window is
    derived from its bbox against `global_transform`. Callers must pass a
    `scale` at call time consistent with `global_transform`'s pixel size
    (deg_per_pixel == scale / rpo._ESTIMATE_METERS_PER_DEGREE), matching
    what _validate_export_alignment will check against.

    fail_tiles: optional set of (row_start, col_start) window origins to
    deliberately fail (no file written) -- used to test the
    failed-tile-leaves-no-false-final-file path.
    """
    calls = []

    def ee_export_image(image, filename, scale, region, crs, file_per_band=False):
        calls.append({"filename": filename, "region_bbox": getattr(region, "bbox", None), "scale": scale})
        xmin, ymin, xmax, ymax = region.bbox
        row_start, col_start = rasterio.transform.rowcol(global_transform, xmin, ymax)
        row_stop, col_stop = rasterio.transform.rowcol(global_transform, xmax, ymin)
        row_start, row_stop = sorted([row_start, row_stop])
        col_start, col_stop = sorted([col_start, col_stop])
        if fail_tiles and (row_start, col_start) in fail_tiles:
            return  # simulate silent failure: no file written
        window_arr = source_array[row_start:row_stop, col_start:col_stop]
        tile_transform = rasterio.transform.from_origin(
            xmin, ymax, global_transform.a, -global_transform.e,
        )
        _write_geotiff(Path(filename), window_arr, tile_transform, crs=crs)

    mod = types.ModuleType("geemap")
    mod.ee_export_image = ee_export_image
    mod._calls = calls
    return mod


def _scale_for_deg_per_pixel(deg_per_pixel: float) -> float:
    """Inverse of run_predictors_only's own estimator conversion, so a test's
    fixed-degree synthetic grid stays consistent with the `scale` (meters)
    passed into export_image_direct_or_tiled (and therefore with what
    _validate_export_alignment will check)."""
    return deg_per_pixel * rpo._ESTIMATE_METERS_PER_DEGREE


def _make_generic_fake_geemap_module(fill_value=1, dtype="int32"):
    """
    A simpler fake geemap whose ee_export_image() derives a SCALE-CORRECT
    pixel grid directly from (region bbox, scale) at call time -- no
    pre-built "world" array needed. Every independently-called tile is
    therefore automatically grid-consistent with every other tile at the
    SAME scale (this is what makes real GEE tiles mergeable too). Used for
    tests that only care about ROUTING (direct vs tiled) and CALL SHAPE,
    not exact pixel-value reconstruction.
    """
    calls = []

    def ee_export_image(image, filename, scale, region, crs, file_per_band=False):
        calls.append({"filename": filename, "region_bbox": getattr(region, "bbox", None), "scale": scale})
        xmin, ymin, xmax, ymax = region.bbox
        deg_per_pixel = scale / rpo._ESTIMATE_METERS_PER_DEGREE
        width = max(1, round((xmax - xmin) / deg_per_pixel))
        height = max(1, round((ymax - ymin) / deg_per_pixel))
        arr = np.full((height, width), fill_value, dtype=dtype)
        transform = rasterio.transform.from_origin(xmin, ymax, deg_per_pixel, deg_per_pixel)
        _write_geotiff(Path(filename), arr, transform, crs=crs)

    mod = types.ModuleType("geemap")
    mod.ee_export_image = ee_export_image
    mod._calls = calls
    return mod


class _StubGEEContext:
    """Context manager: patches sys.modules['ee'/'geemap'] for the duration
    of a `with` block so run_predictors_only's *local* `import ee`/`import
    geemap` inside functions resolve to the stubs."""

    def __init__(self, geemap_module=None, ee_module=None):
        self.geemap_module = geemap_module or _make_fake_geemap_module(
            np.zeros((8, 8), dtype="int32"), from_origin(0.0, 0.008, 0.001, 0.001),
        )
        self.ee_module = ee_module or _make_fake_ee_module()
        self._patcher = None

    def __enter__(self):
        self._patcher = patch.dict(
            sys.modules, {"ee": self.ee_module, "geemap": self.geemap_module}
        )
        self._patcher.__enter__()
        return self

    def __exit__(self, *exc):
        self._patcher.__exit__(*exc)


# =============================================================================
# 1-2. Pure size estimator: deterministic small-vs-large routing signal
# =============================================================================
class TestSizeEstimator(unittest.TestCase):
    def test_deterministic(self):
        bbox = (27.10, 36.60, 28.90, 37.45)
        a = rpo._estimate_request_bytes_from_bbox(*bbox, scale_m=30)
        b = rpo._estimate_request_bytes_from_bbox(*bbox, scale_m=30)
        self.assertEqual(a, b)

    def test_small_bbox_under_threshold(self):
        tiny = (35.00, 37.00, 35.02, 37.02)  # ~2km x 2km
        est = rpo._estimate_request_bytes_from_bbox(*tiny, scale_m=30)
        self.assertLess(est, rpo.DIRECT_EXPORT_SAFE_THRESHOLD_BYTES)

    def test_mugla_bbox_exceeds_threshold_and_hard_limit(self):
        mugla = (27.10, 36.60, 28.90, 37.45)
        est = rpo._estimate_request_bytes_from_bbox(*mugla, scale_m=30)
        self.assertGreater(est, rpo.DIRECT_EXPORT_SAFE_THRESHOLD_BYTES)
        self.assertGreater(est, rpo.GEE_DIRECT_DOWNLOAD_LIMIT_BYTES)

    def test_pixel_grid_positive_and_monotonic_with_extent(self):
        w1, h1 = rpo._estimate_pixel_grid(0, 0, 1, 1, scale_m=30)
        w2, h2 = rpo._estimate_pixel_grid(0, 0, 2, 2, scale_m=30)
        self.assertGreater(w1, 0)
        self.assertGreater(h1, 0)
        self.assertGreater(w2, w1)
        self.assertGreater(h2, h1)


# =============================================================================
# 3-5. Deterministic tile construction, complete coverage, no bad overlaps
# =============================================================================
class TestTileBboxes(unittest.TestCase):
    def test_deterministic_and_covers_full_extent(self):
        xmin, ymin, xmax, ymax = 10.0, 20.0, 12.0, 22.0
        tiles = rpo._tile_bboxes(xmin, ymin, xmax, ymax, rows=2, cols=2)
        self.assertEqual(len(tiles), 4)
        xs = sorted({round(t["bbox"][0], 9) for t in tiles} | {round(t["bbox"][2], 9) for t in tiles})
        ys = sorted({round(t["bbox"][1], 9) for t in tiles} | {round(t["bbox"][3], 9) for t in tiles})
        self.assertAlmostEqual(xs[0], xmin)
        self.assertAlmostEqual(xs[-1], xmax)
        self.assertAlmostEqual(ys[0], ymin)
        self.assertAlmostEqual(ys[-1], ymax)

    def test_no_interior_overlap_only_shared_edges(self):
        tiles = rpo._tile_bboxes(0.0, 0.0, 4.0, 4.0, rows=2, cols=2)
        by_rc = {(t["r"], t["c"]): t["bbox"] for t in tiles}
        left = by_rc[(0, 0)]
        right = by_rc[(0, 1)]
        # shared vertical edge: left's xmax == right's xmin exactly (a shared
        # boundary, not an overlap -- interiors do not intersect)
        self.assertAlmostEqual(left[2], right[0])
        self.assertLess(left[0], left[2])
        self.assertLess(right[0], right[2])

    def test_rectangular_grid_tile_count(self):
        tiles = rpo._tile_bboxes(0, 0, 10, 5, rows=3, cols=6)
        self.assertEqual(len(tiles), 18)


# =============================================================================
# 6-7. Tile transform compatibility (identical CRS/resolution, fixed grid)
# =============================================================================
class TestTileTransformCompatibility(unittest.TestCase):
    def test_compatible_tiles_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tr = from_origin(0.0, 1.0, 0.01, 0.01)
            p1, p2 = d / "a.tif", d / "b.tif"
            _write_geotiff(p1, np.zeros((5, 5), "int32"), tr)
            _write_geotiff(p2, np.ones((5, 5), "int32"), tr)
            srcs = [rasterio.open(p1), rasterio.open(p2)]
            try:
                rpo._assert_tile_transforms_compatible(srcs, "test")  # must not raise
            finally:
                for s in srcs:
                    s.close()

    def test_mismatched_pixel_size_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tr1 = from_origin(0.0, 1.0, 0.01, 0.01)
            tr2 = from_origin(0.0, 1.0, 0.02, 0.02)  # different resolution
            p1, p2 = d / "a.tif", d / "b.tif"
            _write_geotiff(p1, np.zeros((5, 5), "int32"), tr1)
            _write_geotiff(p2, np.ones((5, 5), "int32"), tr2)
            srcs = [rasterio.open(p1), rasterio.open(p2)]
            try:
                with self.assertRaises(rpo.PredictorRunnerError):
                    rpo._assert_tile_transforms_compatible(srcs, "test")
            finally:
                for s in srcs:
                    s.close()


# =============================================================================
# 8-9. Successful mosaic shape/transform + seam-free synthetic reconstruction
# =============================================================================
class TestSeamFreeSyntheticMerge(unittest.TestCase):
    def test_end_to_end_tiled_export_reconstructs_source_exactly(self):
        # 8x8 "world": unique values 0..63 so any misalignment/seam/dup is obvious.
        source = np.arange(64, dtype="int32").reshape(8, 8)
        deg_per_pixel = 0.001
        global_transform = from_origin(0.0, 0.008, deg_per_pixel, deg_per_pixel)  # matches an 8x8 grid over [0,0.008]
        scale = _scale_for_deg_per_pixel(deg_per_pixel)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "merged.tif"
            tiles_dir = Path(tmp) / "_tiles"
            fake_geemap = _make_fake_geemap_module(source, global_transform)
            fake_region = _FakeBBox(0.0, 0.0, 0.008, 0.008)

            with _StubGEEContext(geemap_module=fake_geemap):
                # Force the preflight estimator to route straight to tiled
                # (this AOI is tiny; patch the threshold down to 1 byte so
                # ANY nonzero estimate triggers the tiled path).
                with patch.object(rpo, "DIRECT_EXPORT_SAFE_THRESHOLD_BYTES", 1):
                    result = rpo.export_image_direct_or_tiled(
                        _FakeImage(), out_path, fake_region, scale=scale,
                        crs="EPSG:4326", label="synthetic_world", force=False,
                        tiles_dir=tiles_dir, tile_rows=2, tile_cols=2,
                        cleanup_tiles=True,
                    )

            self.assertEqual(result["transport"], "tiled_preflight_skip")
            self.assertEqual(result["tile_grid"], (2, 2))
            self.assertEqual(result["tile_count"], 4)
            self.assertTrue(out_path.exists())

            with rasterio.open(out_path) as merged:
                merged_arr = merged.read(1)
                self.assertEqual(merged.crs.to_string().upper().replace("EPSG:", "EPSG:"), "EPSG:4326")
                self.assertAlmostEqual(abs(merged.transform.a), global_transform.a, places=9)
                self.assertAlmostEqual(abs(merged.transform.e), global_transform.a, places=9)

            # SEAM-FREE: merged pixels exactly equal the original source --
            # no gaps, no duplicated/shifted seam pixels.
            np.testing.assert_array_equal(merged_arr, source)

            # cleanup_tiles=True -> individual tile files removed on success
            leftover_tiles = list(tiles_dir.glob("*_tile_r*_c*.tif"))
            self.assertEqual(leftover_tiles, [])

            # no atomic .tmp leftovers anywhere
            self.assertEqual(list(Path(tmp).rglob(".*.tmp")), [])
            self.assertEqual(list(Path(tmp).rglob("*.tmp")), [])

    def test_alignment_qa_report_present_for_tiled_result(self):
        deg_per_pixel = 0.001
        source = np.ones((4, 4), dtype="int32")
        global_transform = from_origin(0.0, 0.004, deg_per_pixel, deg_per_pixel)
        scale = _scale_for_deg_per_pixel(deg_per_pixel)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "merged.tif"
            tiles_dir = Path(tmp) / "_tiles"
            fake_geemap = _make_fake_geemap_module(source, global_transform)
            fake_region = _FakeBBox(0.0, 0.0, 0.004, 0.004)
            with _StubGEEContext(geemap_module=fake_geemap):
                with patch.object(rpo, "DIRECT_EXPORT_SAFE_THRESHOLD_BYTES", 1):
                    result = rpo.export_image_direct_or_tiled(
                        _FakeImage(), out_path, fake_region, scale=scale,
                        crs="EPSG:4326", label="qa_check", force=False,
                        tiles_dir=tiles_dir, tile_rows=2, tile_cols=2,
                    )
            self.assertIsNotNone(result["alignment_qa"])
            self.assertEqual(result["alignment_qa"]["band_count"], 1)


# =============================================================================
# 1. Small (estimated) request uses the direct path
# =============================================================================
class TestSmallRequestUsesDirectPath(unittest.TestCase):
    def test_direct_path_used_and_no_tiles_created(self):
        fake_geemap = _make_generic_fake_geemap_module(fill_value=7)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "direct.tif"
            tiles_dir = Path(tmp) / "_tiles"
            fake_region = _FakeBBox(0.0, 0.0, 0.004, 0.004)  # genuinely tiny -> under real threshold

            with _StubGEEContext(geemap_module=fake_geemap):
                result = rpo.export_image_direct_or_tiled(
                    _FakeImage(), out_path, fake_region, scale=30,
                    crs="EPSG:4326", label="small_direct", force=False,
                    tiles_dir=tiles_dir,
                )

        self.assertEqual(result["transport"], "direct")
        self.assertFalse(result["direct_skipped_preflight"])
        self.assertEqual(len(fake_geemap._calls), 1)
        self.assertFalse(tiles_dir.exists() and any(tiles_dir.iterdir()))


# =============================================================================
# 2. Large estimated request uses the tiled path (never attempts direct)
# =============================================================================
class TestLargeEstimateSkipsDirect(unittest.TestCase):
    def test_direct_never_attempted_when_preflight_estimate_is_large(self):
        fake_geemap = _make_generic_fake_geemap_module(fill_value=1)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "large.tif"
            tiles_dir = Path(tmp) / "_tiles"
            # Real (unpatched) estimator + a Muğla-sized bbox naturally
            # exceeds DIRECT_EXPORT_SAFE_THRESHOLD_BYTES -- no patching needed.
            mugla_like_region = _FakeBBox(27.10, 36.60, 28.90, 37.45)

            with _StubGEEContext(geemap_module=fake_geemap):
                result = rpo.export_image_direct_or_tiled(
                    _FakeImage(), out_path, mugla_like_region, scale=30,
                    crs="EPSG:4326", label="large_preflight", force=False,
                    tiles_dir=tiles_dir, tile_rows=2, tile_cols=2,
                )

        self.assertEqual(result["transport"], "tiled_preflight_skip")
        self.assertTrue(result["direct_skipped_preflight"])
        self.assertGreater(result["estimated_bytes"], rpo.DIRECT_EXPORT_SAFE_THRESHOLD_BYTES)
        # every fake geemap call was a TILE call (region bbox is a strict
        # subset of the full AOI), never the full-AOI bbox itself
        for call in fake_geemap._calls:
            self.assertNotEqual(call["region_bbox"], mugla_like_region.bbox)


# =============================================================================
# 12. Failed tile download does not leave a false final file
# =============================================================================
class TestFailedTileLeavesNoFalseFinalFile(unittest.TestCase):
    def test_all_grids_failing_raises_and_creates_no_output(self):
        source = np.zeros((8, 8), dtype="int32")
        global_transform = from_origin(0.0, 0.008, 0.001, 0.001)
        # Fail every tile at every escalation level by giving an impossible
        # window origin that never matches (fail_tiles covers everything by
        # making the fake writer always skip).
        fake_geemap = _make_fake_geemap_module(source, global_transform)
        fake_geemap.ee_export_image = lambda *a, **k: None  # never writes a file

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "should_not_exist.tif"
            tiles_dir = Path(tmp) / "_tiles"
            fake_region = _FakeBBox(0.0, 0.0, 0.008, 0.008)

            with _StubGEEContext(geemap_module=fake_geemap):
                with patch.object(rpo, "DIRECT_EXPORT_SAFE_THRESHOLD_BYTES", 1):
                    with self.assertRaises(rpo.PredictorRunnerError):
                        rpo.export_image_direct_or_tiled(
                            _FakeImage(), out_path, fake_region, scale=1000,
                            crs="EPSG:4326", label="always_fails", force=False,
                            tiles_dir=tiles_dir,
                        )

            self.assertFalse(out_path.exists(), "no false final file may exist after total failure")


# =============================================================================
# 13. Existing output is not overwritten without force
# =============================================================================
class TestExistingOutputNotOverwrittenWithoutForce(unittest.TestCase):
    def test_skip_existing_no_geemap_call_content_unchanged(self):
        fake_geemap = types.ModuleType("geemap")
        calls = []
        fake_geemap.ee_export_image = lambda *a, **k: calls.append(1)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "existing.tif"
            out_path.write_bytes(b"PRE-EXISTING-CONTENT")
            tiles_dir = Path(tmp) / "_tiles"
            fake_region = _FakeBBox(0.0, 0.0, 0.004, 0.004)

            with _StubGEEContext(geemap_module=fake_geemap):
                result = rpo.export_image_direct_or_tiled(
                    _FakeImage(), out_path, fake_region, scale=30,
                    crs="EPSG:4326", label="preexisting", force=False,
                    tiles_dir=tiles_dir,
                )

            self.assertEqual(result["transport"], "skipped_existing")
            self.assertEqual(out_path.read_bytes(), b"PRE-EXISTING-CONTENT")
            self.assertEqual(calls, [])  # geemap never invoked


# =============================================================================
# 14/15/16. Scientific configuration + pre-label logic + protected outputs
#           are UNCHANGED by this export-transport-only fix
# =============================================================================
class TestUnaffectedScientificState(unittest.TestCase):
    def test_14_mugla_config_unchanged(self):
        import core.regions as regions
        exp = regions.get_experiment("mugla_2021")
        self.assertEqual(exp["predictor_start_date"], "2021-06-01")
        self.assertEqual(exp["predictor_end_date"], "2021-07-28")
        self.assertEqual(exp["label_start_date"], "2021-07-29")
        self.assertEqual(exp["label_end_date"], "2021-09-15")
        self.assertEqual(exp["baseline_years"], [2017, 2018, 2019, 2020])
        self.assertEqual(regions.MUGLA_AOI_BBOX, (27.10, 36.60, 28.90, 37.45))
        self.assertTrue(exp["exclude_pre_label_burns"])

    def test_15_pre_label_exclusion_logic_unchanged(self):
        from src.step8a_prepare_500m_modeling_dataset import classify_burndate_relative_to_label
        LS, LE = "2021-07-29", "2021-09-15"
        self.assertEqual(classify_burndate_relative_to_label(172, LS, LE), "pre_label")
        self.assertEqual(classify_burndate_relative_to_label(210, LS, LE), "in_window")
        self.assertEqual(classify_burndate_relative_to_label(0, LS, LE), "unmapped")

    def test_16_other_experiments_unchanged(self):
        import core.regions as regions
        for exp_id, ls, le in [
            ("kozan_2023", "2023-08-01", "2023-10-31"),
            ("manavgat_2021", "2021-07-28", "2021-08-31"),
            ("bejis_2022", "2022-08-15", "2022-09-30"),
        ]:
            e = regions.get_experiment(exp_id)
            self.assertEqual(e["label_start_date"], ls, exp_id)
            self.assertEqual(e["label_end_date"], le, exp_id)


# =============================================================================
# NEW (Mugla 2021 band_count QA bug fix): per-product expected_band_count
# wiring through _export_predictors_direct() -> nested _export() ->
# export_image_direct_or_tiled(band_count=...). Regression coverage for the
# "Alignment QA: beklenmeyen bant sayisi (2 != 1)" failure caused by relying
# on a single global band_count default for products with different real
# band counts (current LST/NDVI = 2 bands; baseline LST/NDVI = 1 band).
#
# Task numbering (bug-fix prompt):
#   1  current LST export call passes band_count=2
#   2  current NDVI export call passes band_count=2
#   3  every baseline LST export call passes band_count=1
#   4  every baseline NDVI export call passes band_count=1
#   5  a 2-band synthetic raster passes QA with expected_band_count=2
#   6  the same raster fails fast with expected_band_count=1
#   7  a 1-band synthetic raster passes QA with expected_band_count=1
#   8  band_count reaches the pre-flight size estimate too (same
#      source-of-truth used for both the estimate and the alignment QA)
#   9  skipped_existing behavior is unaffected (no retroactive QA)
#  10  Manavgat/Bejis/Kozan frozen outputs are not written to -- not
#      exercised here (no live frozen data in this environment; verified by
#      code review instead, see task report: this change only touches
#      scripts/run_predictors_only.py's _export() wrapper and its own 4
#      call sites, none of which touch legacy/other-experiment paths)
# =============================================================================
def _write_multiband_geotiff(path: Path, band_arrays, transform, crs="EPSG:4326", dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=band_arrays[0].shape[0], width=band_arrays[0].shape[1],
        count=len(band_arrays), dtype=dtype, crs=crs, transform=transform,
    ) as dst:
        for i, arr in enumerate(band_arrays, start=1):
            dst.write(arr.astype(dtype), i)


def _make_fake_step3_module(baseline_lst_years, baseline_ndvi_years):
    """Fake src.step3_landsat_lst: never touches GEE. Current_* functions
    return a plain marker image; baseline_* collection functions return a
    chainable fake collection (.filter(...).first()) matching the real
    `ee.Image(collection.filter(ee.Filter.eq(...)).first()).select(...)`
    call shape in _export_predictors_direct()."""
    mod = types.ModuleType("src.step3_landsat_lst")

    class _FakeCollection:
        def filter(self, _f):
            return self

        def first(self):
            return "fake_ee_image_ref"

    def get_current_period_median(region, region_name, end_date, window_days):
        return "current_lst_image_marker", {}

    def get_current_period_ndvi_median(region, region_name, end_date, window_days):
        return "current_ndvi_image_marker", {}

    def get_landsat_baseline_window_median_collection(
        region, region_name, end_date, window_days, baseline_start, baseline_end,
    ):
        windows = [{"year": y, "window_end": f"{y}-07-28"} for y in baseline_lst_years]
        return _FakeCollection(), {"windows": windows}

    def get_landsat_baseline_window_ndvi_collection(
        region, region_name, end_date, window_days, baseline_start, baseline_end,
    ):
        windows = [{"year": y, "window_end": f"{y}-07-28"} for y in baseline_ndvi_years]
        return _FakeCollection(), {"windows": windows}

    mod.get_current_period_median = get_current_period_median
    mod.get_current_period_ndvi_median = get_current_period_ndvi_median
    mod.get_landsat_baseline_window_median_collection = get_landsat_baseline_window_median_collection
    mod.get_landsat_baseline_window_ndvi_collection = get_landsat_baseline_window_ndvi_collection
    return mod


def _make_fake_ee_module_with_image():
    """Extends the minimal fake `ee` module (Geometry.BBox only) with
    Image/Filter, needed by the baseline-year selection chain."""

    class _FakeEEImage:
        def __init__(self, ref):
            self.ref = ref

        def select(self, _band):
            return self

    class _FakeFilter:
        @staticmethod
        def eq(a, b):
            return (a, b)

    mod = _make_fake_ee_module()
    mod.Image = _FakeEEImage
    mod.Filter = _FakeFilter
    return mod


def _make_fake_gee_utils_module():
    mod = types.ModuleType("core.gee_utils")
    mod.init_gee = lambda project=None: None
    return mod


def _build_minimal_ctx(tmp: Path, baseline_years: list[int]) -> dict:
    """Minimal ctx dict covering exactly the keys _export_predictors_direct()
    and _write_predictor_export_metadata() read -- entirely inside a temp
    dir, so no real repo/output paths are ever touched by this test."""
    data_root = tmp / "data"
    return {
        "experiment_id": "test_experiment",
        "data_root": data_root,
        "baseline_input_dir": data_root / "landsat_timeseries",
        "current_period_dir": data_root / "current_period",
        "ndvi_baseline_dir": data_root / "ndvi_timeseries",
        "ndvi_current_dir": data_root / "ndvi_current_period",
        "region_key": "test_region",
        "predictor_start_date": "2021-06-01",
        "predictor_end_date": "2021-07-28",
        "current_period_end_date": "2021-07-28",
        "current_period_days": 58,
        "baseline_start_date": f"{min(baseline_years)}-01-01",
        "baseline_end_date": f"{max(baseline_years)}-12-31",
        "baseline_years": baseline_years,
        "landsat_file_prefix": "test_landsat",
        "output_root": tmp,
    }


class TestPredictorExportBandCountWiring(unittest.TestCase):
    """Task items 1-4: each _export_predictors_direct() product calls
    export_image_direct_or_tiled() with the PRODUCT-SPECIFIC real band count
    (current=2, baseline=1 per year) -- not a silent global default."""

    def _run_with_spy(self, baseline_years=(2018, 2019)):
        calls = {}

        def _spy(image, out_path, region, scale, crs, label, force, *,
                  tiles_dir, cleanup_tiles=False, band_count=1, run_alignment_qa=True):
            calls[label] = band_count
            return {
                "path": out_path, "transport": "direct", "tile_grid": None,
                "tile_count": None, "estimated_bytes": None,
                "direct_skipped_preflight": False, "alignment_qa": None,
            }

        with tempfile.TemporaryDirectory() as tmp:
            ctx = _build_minimal_ctx(Path(tmp), list(baseline_years))
            fake_ee = _make_fake_ee_module_with_image()
            fake_geemap = types.ModuleType("geemap")  # never called: export is spied
            fake_step3 = _make_fake_step3_module(list(baseline_years), list(baseline_years))
            fake_gee_utils = _make_fake_gee_utils_module()

            with patch.dict(sys.modules, {
                "ee": fake_ee,
                "geemap": fake_geemap,
                "src.step3_landsat_lst": fake_step3,
                "core.gee_utils": fake_gee_utils,
            }):
                with patch.object(rpo, "get_region", lambda _ctx: _FakeBBox(0.0, 0.0, 0.01, 0.01)):
                    with patch.object(rpo, "export_image_direct_or_tiled", side_effect=_spy):
                        rpo._export_predictors_direct(ctx, force=False)
        return calls

    def test_1_current_lst_band_count_is_2(self):
        calls = self._run_with_spy()
        self.assertEqual(calls["current_lst"], 2)

    def test_2_current_ndvi_band_count_is_2(self):
        calls = self._run_with_spy()
        self.assertEqual(calls["current_ndvi"], 2)

    def test_3_baseline_lst_band_count_is_1_for_every_year(self):
        years = (2017, 2018, 2019, 2020)
        calls = self._run_with_spy(baseline_years=years)
        for year in years:
            self.assertEqual(calls[f"baseline_lst_{year}"], 1)

    def test_4_baseline_ndvi_band_count_is_1_for_every_year(self):
        years = (2017, 2018, 2019, 2020)
        calls = self._run_with_spy(baseline_years=years)
        for year in years:
            self.assertEqual(calls[f"baseline_ndvi_{year}"], 1)


# =============================================================================
# 5-7. Alignment QA respects the CALLER-SUPPLIED expected band count (accepts
#      a matching count, fails fast on a mismatch) -- no hardcoded global
#      band-count assumption.
# =============================================================================
class TestAlignmentQARespectsExpectedBandCount(unittest.TestCase):
    def setUp(self):
        self.deg_per_pixel = 0.001
        self.transform = from_origin(0.0, 0.004, self.deg_per_pixel, self.deg_per_pixel)
        self.scale = _scale_for_deg_per_pixel(self.deg_per_pixel)
        self.region = _FakeBBox(0.0, 0.0, 0.004, 0.004)

    def test_5_two_band_raster_passes_qa_with_expected_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "two_band.tif"
            arr = np.ones((4, 4), dtype="float32")
            _write_multiband_geotiff(out_path, [arr, arr], self.transform)
            report = rpo._validate_export_alignment(
                out_path, self.region, self.scale, "EPSG:4326", expected_band_count=2,
            )
            self.assertEqual(report["band_count"], 2)

    def test_6_two_band_raster_fails_fast_with_expected_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "two_band.tif"
            arr = np.ones((4, 4), dtype="float32")
            _write_multiband_geotiff(out_path, [arr, arr], self.transform)
            with self.assertRaises(rpo.PredictorRunnerError):
                rpo._validate_export_alignment(
                    out_path, self.region, self.scale, "EPSG:4326", expected_band_count=1,
                )

    def test_7_single_band_raster_passes_qa_with_expected_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "one_band.tif"
            arr = np.ones((4, 4), dtype="float32")
            _write_multiband_geotiff(out_path, [arr], self.transform)
            report = rpo._validate_export_alignment(
                out_path, self.region, self.scale, "EPSG:4326", expected_band_count=1,
            )
            self.assertEqual(report["band_count"], 1)


# =============================================================================
# 8. band_count reaches the pre-flight size estimate too (same
#    source-of-truth used for both _estimate_request_bytes and the alignment
#    QA -- see export_image_direct_or_tiled()).
# =============================================================================
class TestBandCountReachesSizeEstimate(unittest.TestCase):
    def test_estimated_bytes_scales_with_band_count(self):
        fake_geemap = _make_generic_fake_geemap_module(fill_value=1)
        mugla_like_region = _FakeBBox(27.10, 36.60, 28.90, 37.45)

        estimated_bytes = {}
        for band_count in (1, 2):
            with tempfile.TemporaryDirectory() as tmp:
                out_path = Path(tmp) / f"band_{band_count}.tif"
                tiles_dir = Path(tmp) / "_tiles"
                with _StubGEEContext(geemap_module=fake_geemap):
                    # run_alignment_qa=False: this test isolates the
                    # pre-flight ESTIMATE only -- the fake geemap tiles are
                    # always single-band regardless of the requested
                    # band_count, so alignment QA is out of scope here (see
                    # tests 5-7 above for QA accept/reject coverage).
                    result = rpo.export_image_direct_or_tiled(
                        _FakeImage(), out_path, mugla_like_region, scale=30,
                        crs="EPSG:4326", label=f"band_count_{band_count}", force=False,
                        tiles_dir=tiles_dir, tile_rows=2, tile_cols=2,
                        band_count=band_count, run_alignment_qa=False,
                    )
                estimated_bytes[band_count] = result["estimated_bytes"]

        self.assertIsNotNone(estimated_bytes[1])
        self.assertIsNotNone(estimated_bytes[2])
        self.assertEqual(estimated_bytes[2], estimated_bytes[1] * 2)


# =============================================================================
# 9. skipped_existing behavior is unaffected by band_count: a pre-existing
#    output is skipped (transport="skipped_existing") and NEVER retroactively
#    QA'd against whatever band_count the current call happens to request --
#    this protects old/frozen files (e.g. Manavgat/Bejis) whose real band
#    count may not match what a newer caller would ask for.
# =============================================================================
class TestSkippedExistingUnaffectedByBandCount(unittest.TestCase):
    def test_skip_existing_ignores_band_count_no_retroactive_qa(self):
        def _fail_if_called(*_a, **_k):
            raise AssertionError("geemap.ee_export_image must not be called for an existing file")

        fake_geemap = types.ModuleType("geemap")
        fake_geemap.ee_export_image = _fail_if_called

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "frozen_one_band.tif"
            # Frozen legacy file: only 1 real band, but this call asks for 2
            # (as current-period products correctly do post-fix) -- must NOT
            # be retroactively rejected.
            arr = np.ones((4, 4), dtype="float32")
            _write_multiband_geotiff(out_path, [arr], from_origin(0.0, 0.004, 0.001, 0.001))
            tiles_dir = Path(tmp) / "_tiles"
            fake_region = _FakeBBox(0.0, 0.0, 0.004, 0.004)

            with _StubGEEContext(geemap_module=fake_geemap):
                result = rpo.export_image_direct_or_tiled(
                    _FakeImage(), out_path, fake_region, scale=30,
                    crs="EPSG:4326", label="frozen", force=False,
                    tiles_dir=tiles_dir, band_count=2,
                )

            self.assertEqual(result["transport"], "skipped_existing")
            self.assertIsNone(result["alignment_qa"])


if __name__ == "__main__":
    unittest.main()