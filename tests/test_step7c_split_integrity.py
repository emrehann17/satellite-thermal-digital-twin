"""
tests/test_step7c_split_integrity.py

Focused tests for the Step7C (src/step7c_train_downscaling_model.py)
grouped_split() split-integrity hardening: safe ArrowStringArray handling
before shuffling, and explicit train/validation/test disjointness/coverage
assertions.

Background: pandas 3.x's default string dtype backend returns an
ArrowStringArray from `.unique()` on a string column (e.g.
`spatial_block_id`, built via `.astype(str)` concatenation in
add_spatial_block_id()). Passing that directly to
`numpy.random.Generator.shuffle()` emits:
    "you are shuffling a 'ArrowStringArray' object which is not a subclass
    of 'Sequence'; `shuffle` is not guaranteed to behave correctly. E.g.,
    non-numpy array/tensor objects with view semantics may contain
    duplicates after shuffling."
-- a real correctness risk (potential duplicate group assignment across
splits), not just a cosmetic warning.

Covers:
    6. train/validation/test spatial groups are disjoint
    7. ArrowStringArray inputs are safely converted before shuffling

Run:
    python -m unittest tests.test_step7c_split_integrity
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.step7c_train_downscaling_model as step7c


def _synthetic_grid_df(n_rows: int = 40, n_cols: int = 40, seed: int = 0) -> pd.DataFrame:
    """A small synthetic pixel grid: enough distinct (row, col) pairs to
    produce multiple spatial_block_id groups at a small block size."""
    rng = np.random.default_rng(seed)
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rows = rows.ravel()
    cols = cols.ravel()
    n = rows.size
    return pd.DataFrame({
        "row": rows.astype("int64"),
        "col": cols.astype("int64"),
        "modis_pixel_id": (rows * n_cols + cols).astype("int64"),
        "source_tile_id": (rows // 10).astype("int32"),
        "landsat_lst_celsius": rng.normal(25.0, 3.0, size=n).astype("float32"),
    })


class TestArrowStringArraySafeShuffle(unittest.TestCase):
    def test_pandas_default_string_unique_is_an_arrow_extension_array(self):
        """Sanity check that this pandas install actually reproduces the
        scenario being guarded against (documents the root cause; if a
        future pandas version changes this default, this test explains why
        the other assertions below still pass either way)."""
        s = pd.Series([1, 2, 3]).astype(str) + "_" + pd.Series([4, 5, 6]).astype(str)
        self.assertFalse(isinstance(s.unique(), np.ndarray))

    def test_np_asarray_conversion_avoids_the_shuffle_warning(self):
        s = pd.Series([1, 2, 3, 10, 11, 12]).astype(str) + "_" + pd.Series([4, 5, 6, 7, 8, 9]).astype(str)
        groups = np.asarray(s.unique())
        self.assertIsInstance(groups, np.ndarray)
        rng = np.random.default_rng(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning here fails the test
            rng.shuffle(groups)
        self.assertEqual(len(groups), len(set(groups.tolist())))  # no duplicates introduced

    def test_grouped_split_emits_no_arrow_string_array_warning(self):
        df = _synthetic_grid_df()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step7c.grouped_split(
                df, test_size=0.15, val_size=0.15, seed=42,
                allow_random_split=False, split_mode="spatial_block",
                spatial_block_size=8,
            )
        arrow_warnings = [w for w in caught if "ArrowStringArray" in str(w.message)]
        self.assertEqual(arrow_warnings, [])

    def test_split_assignment_is_reproducible_across_repeated_calls(self):
        """The np.asarray(...) conversion must not change which groups land
        in which split for a given seed -- same seed/proportions in, same
        assignment out (task: 'do not alter the existing split seed or
        proportions')."""
        df = _synthetic_grid_df()
        train1, val1, test1, *_ = step7c.grouped_split(
            df, test_size=0.15, val_size=0.15, seed=42,
            allow_random_split=False, split_mode="spatial_block",
            spatial_block_size=8,
        )
        train2, val2, test2, *_ = step7c.grouped_split(
            df, test_size=0.15, val_size=0.15, seed=42,
            allow_random_split=False, split_mode="spatial_block",
            spatial_block_size=8,
        )
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)


class TestSplitDisjointnessAndCoverage(unittest.TestCase):
    def _assert_integrity(self, df, train_df, val_df, test_df, group_col, split_info):
        integrity = split_info["split_integrity"]
        self.assertTrue(integrity["all_rows_assigned_exactly_once"])
        self.assertEqual(integrity["assigned_row_count"], len(df))
        self.assertEqual(integrity["total_row_count"], len(df))
        self.assertEqual(
            len(train_df) + len(val_df) + len(test_df), len(df),
            "train+val+test row counts must equal the full dataset (no drops/dupes)",
        )
        if group_col is not None:
            self.assertTrue(integrity["train_val_test_groups_disjoint"])
            self.assertEqual(integrity["union_group_count"], integrity["total_unique_group_count"])
            train_groups = set(train_df[group_col].unique().tolist()) if len(train_df) else set()
            val_groups = set(val_df[group_col].unique().tolist()) if len(val_df) else set()
            test_groups = set(test_df[group_col].unique().tolist()) if len(test_df) else set()
            self.assertEqual(train_groups & val_groups, set())
            self.assertEqual(train_groups & test_groups, set())
            self.assertEqual(val_groups & test_groups, set())

    def test_spatial_block_split_groups_disjoint_and_rows_fully_assigned(self):
        df = _synthetic_grid_df()
        train_df, val_df, test_df, split_mode_used, group_col, split_info = step7c.grouped_split(
            df, test_size=0.15, val_size=0.15, seed=42,
            allow_random_split=False, split_mode="spatial_block",
            spatial_block_size=8,
        )
        self.assertEqual(split_mode_used, "spatial_block")
        self.assertEqual(group_col, "spatial_block_id")
        self._assert_integrity(df, train_df, val_df, test_df, group_col, split_info)

    def test_modis_pixel_group_split_groups_disjoint(self):
        df = _synthetic_grid_df()
        train_df, val_df, test_df, split_mode_used, group_col, split_info = step7c.grouped_split(
            df, test_size=0.15, val_size=0.15, seed=7,
            allow_random_split=False, split_mode="modis_pixel_group",
            spatial_block_size=8,
        )
        self.assertEqual(group_col, "modis_pixel_id")
        self._assert_integrity(df, train_df, val_df, test_df, group_col, split_info)

    def test_tile_group_split_groups_disjoint(self):
        df = _synthetic_grid_df()
        train_df, val_df, test_df, split_mode_used, group_col, split_info = step7c.grouped_split(
            df, test_size=0.15, val_size=0.15, seed=7,
            allow_random_split=False, split_mode="tile_group",
            spatial_block_size=8,
        )
        self.assertEqual(group_col, "source_tile_id")
        self._assert_integrity(df, train_df, val_df, test_df, group_col, split_info)

    def test_random_split_rows_fully_assigned_and_disjoint(self):
        df = _synthetic_grid_df()
        train_df, val_df, test_df, split_mode_used, group_col, split_info = step7c.grouped_split(
            df, test_size=0.15, val_size=0.15, seed=7,
            allow_random_split=True, split_mode="random",
            spatial_block_size=8,
        )
        self.assertIsNone(group_col)
        self._assert_integrity(df, train_df, val_df, test_df, group_col, split_info)


if __name__ == "__main__":
    unittest.main()
