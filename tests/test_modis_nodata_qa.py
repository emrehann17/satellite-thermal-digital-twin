"""
tests/test_modis_nodata_qa.py

Focused tests for the MODIS no-data/masking contract fix used by the
experiment-aware Step7 downscaling pipeline (scripts/prepare_modis_for_step7.py,
scripts/run_predictors_only.py, src/step7b_prepare_downscaling_dataset.py).

No live GEE credentials are used or required for the tiled-export tests:
`ee`/`geemap` are replaced with lightweight in-process stubs (same pattern
as tests/test_export_size_safe_tiling.py) so the real
_export_tiled() -> rasterio.merge path runs against small synthetic GeoTIFFs.

Covers:
    1. masked pixels do not become zero during tiled merging
       (TestTiledMergePreservesNodata)
    2. nodata survives alignment (TestAlignmentPreservesNodata)
    3. MODIS zero-fill input is rejected
       (TestZeroFillSourceRejected, TestStdNegativeRejected,
        TestMeanStdGridMismatchRejected)
    (bonus) a stale tile downloaded before this fix (missing the expected
       nodata tag) is refused rather than silently merged
       (TestStaleTileWithoutNodataRejected)

Run:
    python -m unittest tests.test_modis_nodata_qa
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.run_predictors_only as rpo
import src.step7b_prepare_downscaling_dataset as step7b
from core.config import STEP7B_MIN_TARGET_CELSIUS, STEP7B_MAX_TARGET_CELSIUS


# =============================================================================
# Fake ee / geemap (no live GEE credentials needed) -- same pattern as
# tests/test_export_size_safe_tiling.py, kept self-contained here.
# =============================================================================
class _FakeBBox:
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


class _FakeImage:
    def clip(self, geom):
        return self


def _make_fake_ee_module():
    mod = types.ModuleType("ee")
    mod.Geometry = _FakeGeometry
    return mod


def _write_geotiff(path: Path, array: np.ndarray, transform, crs="EPSG:4326", dtype="float64", nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(array.astype(dtype), 1)


def _make_fake_geemap_module(source_array: np.ndarray, global_transform, crs="EPSG:4326", dtype="float64"):
    """
    Fake ee_export_image(): slices `source_array` per requested tile bbox
    and writes a REAL GeoTIFF -- deliberately WITHOUT a nodata tag (mirrors
    real geemap.ee_export_image, which never sets one). The code under test
    (scripts/run_predictors_only.py `nodata=` handling) must stamp the tag
    itself after download.
    """
    def ee_export_image(image, filename, scale, region, crs=crs, file_per_band=False):
        xmin, ymin, xmax, ymax = region.bbox
        row_start, col_start = rasterio.transform.rowcol(global_transform, xmin, ymax)
        row_stop, col_stop = rasterio.transform.rowcol(global_transform, xmax, ymin)
        row_start, row_stop = sorted([row_start, row_stop])
        col_start, col_stop = sorted([col_start, col_stop])
        window_arr = source_array[row_start:row_stop, col_start:col_stop]
        tile_transform = rasterio.transform.from_origin(
            xmin, ymax, global_transform.a, -global_transform.e,
        )
        _write_geotiff(Path(filename), window_arr, tile_transform, crs=crs, dtype=dtype, nodata=None)

    mod = types.ModuleType("geemap")
    mod.ee_export_image = ee_export_image
    return mod


def _scale_for_deg_per_pixel(deg_per_pixel: float) -> float:
    return deg_per_pixel * rpo._ESTIMATE_METERS_PER_DEGREE


# =============================================================================
# 1. Masked pixels do not become zero during tiled merging
# =============================================================================
class TestTiledMergePreservesNodata(unittest.TestCase):
    def test_sentinel_survives_merge_and_is_never_zero(self):
        SENTINEL = -9999.0
        # Left half of the "world" is masked/no-observation (sentinel);
        # right half is real Celsius-like data. 0.0 never appears anywhere
        # in this synthetic world -- if it shows up in the merged output,
        # the merge silently zero-filled something it shouldn't have.
        world = np.full((8, 8), SENTINEL, dtype="float64")
        world[:, 4:] = 22.5
        deg_per_pixel = 0.001
        global_transform = from_origin(0.0, 0.008, deg_per_pixel, deg_per_pixel)
        fake_geemap = _make_fake_geemap_module(world, global_transform)
        fake_ee = _make_fake_ee_module()
        fake_region = _FakeBBox(0.0, 0.0, 0.008, 0.008)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "modis_lst_mean_celsius.tif"
            tiles_dir = Path(tmp) / "_tiles"
            with patch.dict(sys.modules, {"ee": fake_ee, "geemap": fake_geemap}):
                result_path = rpo._export_tiled(
                    _FakeImage(), out_path, fake_region,
                    scale=_scale_for_deg_per_pixel(deg_per_pixel), crs="EPSG:4326",
                    label="modis_lst_mean", force=False,
                    tile_rows=2, tile_cols=2, tiles_dir=tiles_dir,
                    nodata=SENTINEL,
                )

            with rasterio.open(result_path) as merged:
                self.assertEqual(merged.nodata, SENTINEL)
                arr = merged.read(1)

            np.testing.assert_array_equal(arr, world)
            self.assertFalse(np.any(arr[:, :4] == 0.0))
            self.assertTrue(np.all(arr[:, :4] == SENTINEL))
            self.assertTrue(np.all(arr[:, 4:] == 22.5))

    def test_default_nodata_none_leaves_legacy_behavior_unaffected(self):
        """nodata=None (unchanged default) must not raise or alter legacy
        (non-MODIS) callers that never pass the parameter."""
        world = np.zeros((4, 4), dtype="int32")
        deg_per_pixel = 0.001
        global_transform = from_origin(0.0, 0.004, deg_per_pixel, deg_per_pixel)
        fake_geemap = _make_fake_geemap_module(world, global_transform, dtype="int32")
        fake_ee = _make_fake_ee_module()
        fake_region = _FakeBBox(0.0, 0.0, 0.004, 0.004)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "legacy.tif"
            tiles_dir = Path(tmp) / "_tiles"
            with patch.dict(sys.modules, {"ee": fake_ee, "geemap": fake_geemap}):
                result_path = rpo._export_tiled(
                    _FakeImage(), out_path, fake_region,
                    scale=_scale_for_deg_per_pixel(deg_per_pixel), crs="EPSG:4326",
                    label="legacy_product", force=False,
                    tile_rows=2, tile_cols=2, tiles_dir=tiles_dir,
                )
            with rasterio.open(result_path) as merged:
                self.assertIsNone(merged.nodata)


class TestStaleTileWithoutNodataRejected(unittest.TestCase):
    """A tile downloaded before this fix (no nodata tag) must never be
    silently reused once nodata=<value> is requested."""

    def test_reuse_of_untagged_tile_raises(self):
        SENTINEL = -9999.0
        deg_per_pixel = 0.001
        global_transform = from_origin(0.0, 0.004, deg_per_pixel, deg_per_pixel)
        world = np.full((4, 4), 10.0, dtype="float64")
        fake_geemap = _make_fake_geemap_module(world, global_transform)
        fake_ee = _make_fake_ee_module()
        fake_region = _FakeBBox(0.0, 0.0, 0.004, 0.004)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "modis_lst_mean_celsius.tif"
            tiles_dir = Path(tmp) / "_tiles"
            tiles_dir.mkdir(parents=True)
            stale_path = tiles_dir / f"{out_path.stem}_tile_r0_c0.tif"
            _write_geotiff(
                stale_path, np.zeros((2, 2)),
                from_origin(0.0, 0.004, deg_per_pixel, deg_per_pixel),
                dtype="float64", nodata=None,
            )
            with patch.dict(sys.modules, {"ee": fake_ee, "geemap": fake_geemap}):
                with self.assertRaises(rpo.PredictorRunnerError):
                    rpo._export_tiled(
                        _FakeImage(), out_path, fake_region,
                        scale=_scale_for_deg_per_pixel(deg_per_pixel), crs="EPSG:4326",
                        label="modis_lst_mean", force=False,
                        tile_rows=2, tile_cols=2, tiles_dir=tiles_dir,
                        nodata=SENTINEL,
                    )


# =============================================================================
# 2. Nodata survives (bilinear) alignment to the Step5 reference grid
# =============================================================================
class TestAlignmentPreservesNodata(unittest.TestCase):
    def test_nodata_preserved_through_reproject_not_interpolated_as_zero(self):
        SENTINEL = -9999.0
        src_transform = from_origin(0.0, 0.004, 0.001, 0.001)
        src_arr = np.full((4, 4), 15.0, dtype="float64")
        src_arr[:, 0] = SENTINEL  # one "sea"-like masked column

        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / "modis_lst_mean_celsius.tif"
            with rasterio.open(
                src_path, "w", driver="GTiff", height=4, width=4, count=1,
                dtype="float64", crs="EPSG:4326", transform=src_transform, nodata=SENTINEL,
            ) as dst:
                dst.write(src_arr, 1)

            ref_transform = from_origin(0.0, 0.004, 0.0005, 0.0005)  # finer grid -> forces reproject, not same-grid copy
            output_dir = Path(tmp) / "aligned"

            out_path, diag = step7b.align_feature_to_reference(
                "modis_lst_mean_celsius", src_path, "bilinear",
                8, 8, CRS.from_epsg(4326), ref_transform,
                output_dir, force=True,
            )

            with rasterio.open(out_path) as aligned:
                self.assertEqual(aligned.nodata, SENTINEL)
                arr = aligned.read(1)

            # The masked source column must remain nodata in the aligned
            # output -- never interpolated as if it were 0.0 or blended
            # with the neighboring valid column.
            self.assertTrue(np.all(arr[:, 0] == SENTINEL))
            self.assertTrue(np.all(arr[:, 1] == SENTINEL))
            self.assertFalse(np.any(arr == 0.0))
            self.assertTrue(np.all(arr[:, 4:] == 15.0))
            self.assertLess(diag["aligned_valid_fraction"], 1.0)


# =============================================================================
# 3. MODIS zero-fill input is rejected (Step7B pre-alignment validation)
# =============================================================================
class TestZeroFillSourceRejected(unittest.TestCase):
    def _write_mean_raster(self, path: Path, arr: np.ndarray, nodata):
        with rasterio.open(
            path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
            dtype="float64", crs="EPSG:4326",
            transform=from_origin(0.0, 0.01, 0.001, 0.001), nodata=nodata,
        ) as dst:
            dst.write(arr, 1)

    def test_undefined_nodata_with_suspicious_zero_fraction_raises(self):
        """Reproduces the exact evia_2021 bug signature: nodata=None and
        roughly half the 'valid' pixels are exact 0.0 (masked sea/no-obs
        region silently exported as numeric zero)."""
        arr = np.full((10, 10), 20.0, dtype="float64")
        arr[:, :5] = 0.0  # ~50% exact zero, matching the observed bug
        with tempfile.TemporaryDirectory() as tmp:
            mean_path = Path(tmp) / "modis_lst_mean_celsius.tif"
            self._write_mean_raster(mean_path, arr, nodata=None)
            core_features = [{"name": "modis_lst_mean_celsius", "path": mean_path}]
            with self.assertRaises(step7b.Step7BModisValidationError):
                step7b.validate_modis_source_rasters(core_features)

    def test_proper_nodata_with_same_zero_pattern_passes(self):
        """The SAME zero-fill pattern, but now correctly tagged nodata --
        must pass (0.0 values are then excluded from 'valid', not counted
        against the suspicious-zero-fraction rule)."""
        arr = np.full((10, 10), 20.0, dtype="float64")
        arr[:, :5] = -9999.0
        with tempfile.TemporaryDirectory() as tmp:
            mean_path = Path(tmp) / "modis_lst_mean_celsius.tif"
            self._write_mean_raster(mean_path, arr, nodata=-9999.0)
            core_features = [{"name": "modis_lst_mean_celsius", "path": mean_path}]
            diagnostics = step7b.validate_modis_source_rasters(core_features)
            self.assertEqual(
                diagnostics["modis_lst_mean_celsius"]["validation_status"], "passed"
            )
            self.assertEqual(
                diagnostics["modis_lst_mean_celsius"]["exact_zero_count_among_valid"], 0
            )

    def test_nonphysical_mean_values_rejected(self):
        arr = np.full((4, 4), STEP7B_MAX_TARGET_CELSIUS + 50.0, dtype="float64")
        with tempfile.TemporaryDirectory() as tmp:
            mean_path = Path(tmp) / "modis_lst_mean_celsius.tif"
            self._write_mean_raster(mean_path, arr, nodata=-9999.0)
            core_features = [{"name": "modis_lst_mean_celsius", "path": mean_path}]
            with self.assertRaises(step7b.Step7BModisValidationError):
                step7b.validate_modis_source_rasters(core_features)

    def test_missing_modis_mean_feature_is_a_noop(self):
        """No MODIS mean feature present at all (e.g. an experiment that
        doesn't use MODIS context) -- validation must not fabricate an
        error; it simply has nothing to check."""
        self.assertEqual(step7b.validate_modis_source_rasters([]), {})


class TestStdNegativeRejected(unittest.TestCase):
    def test_negative_std_value_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            mean_path = Path(tmp) / "modis_lst_mean_celsius.tif"
            std_path = Path(tmp) / "modis_lst_std_celsius.tif"
            transform = from_origin(0.0, 0.01, 0.001, 0.001)
            with rasterio.open(
                mean_path, "w", driver="GTiff", height=4, width=4, count=1,
                dtype="float64", crs="EPSG:4326", transform=transform, nodata=-9999.0,
            ) as dst:
                dst.write(np.full((4, 4), 20.0), 1)
            with rasterio.open(
                std_path, "w", driver="GTiff", height=4, width=4, count=1,
                dtype="float64", crs="EPSG:4326", transform=transform, nodata=-9999.0,
            ) as dst:
                arr = np.full((4, 4), 1.5)
                arr[0, 0] = -2.0  # invalid: negative stdDev
                dst.write(arr, 1)
            core_features = [
                {"name": "modis_lst_mean_celsius", "path": mean_path},
                {"name": "modis_lst_std_celsius", "path": std_path},
            ]
            with self.assertRaises(step7b.Step7BModisValidationError):
                step7b.validate_modis_source_rasters(core_features)


class TestMeanStdGridMismatchRejected(unittest.TestCase):
    def test_differing_grids_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            mean_path = Path(tmp) / "modis_lst_mean_celsius.tif"
            std_path = Path(tmp) / "modis_lst_std_celsius.tif"
            with rasterio.open(
                mean_path, "w", driver="GTiff", height=4, width=4, count=1,
                dtype="float64", crs="EPSG:4326",
                transform=from_origin(0.0, 0.01, 0.001, 0.001), nodata=-9999.0,
            ) as dst:
                dst.write(np.full((4, 4), 20.0), 1)
            with rasterio.open(
                std_path, "w", driver="GTiff", height=6, width=6, count=1,
                dtype="float64", crs="EPSG:4326",
                transform=from_origin(0.0, 0.01, 0.0007, 0.0007), nodata=-9999.0,
            ) as dst:
                dst.write(np.full((6, 6), 1.5), 1)
            core_features = [
                {"name": "modis_lst_mean_celsius", "path": mean_path},
                {"name": "modis_lst_std_celsius", "path": std_path},
            ]
            with self.assertRaises(step7b.Step7BModisValidationError):
                step7b.validate_modis_source_rasters(core_features)


if __name__ == "__main__":
    unittest.main()
