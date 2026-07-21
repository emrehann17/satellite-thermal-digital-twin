"""
tests/test_step8a_pre_label_exclusion.py

Targeted tests for Step8A's side of the leakage-safe pre-label exclusion
join: Step8A must read the Step6B gate's cell-level exclusion manifest
verbatim (never reimplement/re-derive it) and remove the SAME physical
cells from its own analysis universe (valid_for_modeling).

Covers (bug-fix prompt task numbering):
    4  Step8A fails fast when the manifest is missing/malformed
       (read_pre_label_exclusion_manifest -- the function main() calls)
    5  pre_label_excluded_cell_ids=None (config false) behaves exactly like
       before this feature existed -- existing flow unaffected
    6  manifest cells are marked pre_label_burn_excluded=True,
       analysis_eligible=False, valid_for_modeling=False,
       invalid_reason=pre_label_burn_excluded
    7  excluded cell_id values never survive Step8B's
       filter_valid_for_modeling() (the function step8b's main() calls
       before training)
    8  raw vs eligible label counts are tracked separately in the returned
       counters (feeds Step8A's stats.json / Step8E's report)
    9  Manavgat/Bejís/Kozan (existing experiments; exclude_pre_label_burns
       always False for them) are unaffected -- covered by the existing
       config snapshot test in tests/test_mugla_2021_gate.py
       (TestExistingExperimentsUnchanged); re-asserted here at the
       build_dataset() level via the "cell_ids=None" equivalence test (5).

Run:
    python -m unittest tests.test_step8a_pre_label_exclusion
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.step8a_prepare_500m_modeling_dataset import (
    Step8AError,
    build_dataset,
    compute_cell_identity,
    read_pre_label_exclusion_manifest,
)
from src.step8b_train_baseline_vs_thermal_model import filter_valid_for_modeling

LS, LE = "2021-07-29", "2021-09-15"


# =============================================================================
# Synthetic fixture: 2x2 native-cell grid (same block size as the gate
# fixture in tests/test_mugla_2021_gate.py). Cell (0,0) is an in-window
# burn; cells (0,1)/(1,0)/(1,1) are unburned. Landcover is tree_cover
# everywhere (10). ndvi/elevation/slope are valid (finite) everywhere, so
# every cell is predictor_valid=True -- isolating the pre-label exclusion
# effect from any predictor-QA effect.
# =============================================================================
class _Step8ASyntheticFixture:
    BS = 17
    N = 2
    W = H = BS * N

    def _write(self, path, arr, dtype="float32", nodata=None):
        tr = from_origin(30.0, 37.0, 0.001, 0.001)
        prof = dict(driver="GTiff", width=self.W, height=self.H, count=1,
                    dtype=dtype, crs="EPSG:4326", transform=tr)
        if nodata is not None:
            prof["nodata"] = nodata
        with rasterio.open(path, "w", **prof) as d:
            d.write(arr.astype(dtype), 1)

    def _cell(self, r, c):
        return (slice(r * self.BS, (r + 1) * self.BS), slice(c * self.BS, (c + 1) * self.BS))

    def _build(self, tmp):
        d = Path(tmp)
        ref = np.ones((self.H, self.W), np.float32)
        label = np.zeros((self.H, self.W), np.float32)
        label[self._cell(0, 0)] = 220  # in-window burn (Aug 8)
        lc = np.full((self.H, self.W), 10, np.float32)  # tree_cover everywhere
        ndvi = np.full((self.H, self.W), 0.5, np.float32)
        elevation = np.full((self.H, self.W), 100.0, np.float32)
        slope = np.full((self.H, self.W), 5.0, np.float32)

        refp = d / "ref.tif"
        lblp = d / "label.tif"
        lcp = d / "lc.tif"
        ndvip = d / "ndvi.tif"
        elevp = d / "elevation.tif"
        slopep = d / "slope.tif"

        self._write(refp, ref, "int16")
        self._write(lblp, label)
        self._write(lcp, lc, "uint8", nodata=0)
        self._write(ndvip, ndvi)
        self._write(elevp, elevation)
        self._write(slopep, slope)

        return {
            "reference_path": refp, "label_path": lblp, "landcover_path": lcp,
            "predictor_paths": {"ndvi": ndvip, "elevation": elevp, "slope": slopep},
            "output_dir": d,
        }

    def _excluded_cell_id(self):
        cell_id, _row, _col = compute_cell_identity(row_off=1 * self.BS, col_off=1 * self.BS, block_size=self.BS)
        return cell_id

    def _run_build_dataset(self, tmp, pre_label_excluded_cell_ids=None):
        paths = self._build(tmp)
        return build_dataset(
            reference_path=paths["reference_path"],
            label_path=paths["label_path"],
            label_kind="raw_burndate",
            predictor_paths=paths["predictor_paths"],
            landcover_path=paths["landcover_path"],
            source_mask_path=None,
            output_dir=paths["output_dir"],
            min_valid_fraction=0.3,
            burnable_threshold=0.5,
            label_start=LS,
            label_end=LE,
            pre_label_excluded_cell_ids=pre_label_excluded_cell_ids,
        )


# =============================================================================
# 4. Step8A fails fast when the manifest is missing/malformed
# =============================================================================
class TestReadPreLabelExclusionManifestFailFast(unittest.TestCase):
    def test_missing_manifest_raises_with_required_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "pre_label_excluded_cells.parquet"
            with self.assertRaises(Step8AError) as ctx:
                read_pre_label_exclusion_manifest(missing_path)
            msg = str(ctx.exception)
            self.assertIn(
                "Pre-label exclusion is enabled but the canonical gate "
                "exclusion manifest is missing.", msg,
            )
            self.assertIn("Re-run the label gate before Step8A.", msg)

    def test_duplicate_cell_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pre_label_excluded_cells.parquet"
            df = pd.DataFrame({
                "experiment_id": ["mugla_2021", "mugla_2021"],
                "cell_id": ["r1_c1", "r1_c1"],
                "row_500m": [1, 1], "col_500m": [1, 1],
            })
            df.to_parquet(path, index=False)
            with self.assertRaises(Step8AError):
                read_pre_label_exclusion_manifest(path)

    def test_null_cell_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pre_label_excluded_cells.parquet"
            df = pd.DataFrame({
                "experiment_id": ["mugla_2021", "mugla_2021"],
                "cell_id": ["r1_c1", None],
                "row_500m": [1, 2], "col_500m": [1, 2],
            })
            df.to_parquet(path, index=False)
            with self.assertRaises(Step8AError):
                read_pre_label_exclusion_manifest(path)

    def test_valid_manifest_returns_correct_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pre_label_excluded_cells.parquet"
            df = pd.DataFrame({
                "experiment_id": ["mugla_2021", "mugla_2021"],
                "cell_id": ["r1_c1", "r3_c4"],
                "row_500m": [1, 3], "col_500m": [1, 4],
            })
            df.to_parquet(path, index=False)
            result = read_pre_label_exclusion_manifest(path)
            self.assertEqual(result, frozenset({"r1_c1", "r3_c4"}))


# =============================================================================
# 5/6/8. build_dataset() join behavior + eligibility counters
# =============================================================================
class TestBuildDatasetPreLabelJoin(_Step8ASyntheticFixture, unittest.TestCase):
    def test_06_excluded_cell_marked_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            excluded_id = self._excluded_cell_id()
            result = self._run_build_dataset(tmp, pre_label_excluded_cell_ids=frozenset({excluded_id}))
            df = result["dataframe"]
            row = df.loc[df["cell_id"] == excluded_id].iloc[0]
            self.assertTrue(row["pre_label_burn_excluded"])
            self.assertFalse(row["analysis_eligible"])
            self.assertFalse(row["valid_for_modeling"])
            self.assertEqual(row["invalid_reason"], "pre_label_burn_excluded")

            # the in-window burned cell (0,0) is untouched: still eligible/valid.
            burned_row = df.loc[df["burned"] == 1].iloc[0]
            self.assertFalse(burned_row["pre_label_burn_excluded"])
            self.assertTrue(burned_row["analysis_eligible"])
            self.assertTrue(burned_row["valid_for_modeling"])

    def test_05_none_cell_ids_behaves_like_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_build_dataset(tmp, pre_label_excluded_cell_ids=None)
            df = result["dataframe"]
            self.assertTrue((df["pre_label_burn_excluded"] == False).all())  # noqa: E712
            self.assertTrue((df["analysis_eligible"] == True).all())  # noqa: E712
            # with no exclusion and all predictors valid, all 4 cells are
            # valid_for_modeling (1 burned + 3 unburned).
            self.assertEqual(int((df["valid_for_modeling"] == True).sum()), 4)  # noqa: E712
            counters = result["counters"]
            self.assertEqual(counters["pre_label_burn_excluded_count"], 0)
            self.assertEqual(counters["analysis_eligible_count"], 4)

    def test_08_raw_vs_eligible_vs_final_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            excluded_id = self._excluded_cell_id()
            result = self._run_build_dataset(tmp, pre_label_excluded_cell_ids=frozenset({excluded_id}))
            counters = result["counters"]
            # raw: unaffected by exclusion (1 burned, 3 unburned -- includes
            # the excluded cell, which is unburned in the raw label).
            self.assertEqual(counters["raw_label_counts_before_eligibility"], {"burned": 1, "unburned": 3})
            # eligible: excluded cell dropped from the unburned tally.
            self.assertEqual(
                counters["eligible_label_counts_after_pre_label_exclusion"], {"burned": 1, "unburned": 2},
            )
            # final (== eligible here, since no predictor invalidity present).
            self.assertEqual(
                counters["final_modeling_counts_after_predictor_validity"], {"burned": 1, "unburned": 2},
            )
            self.assertEqual(counters["pre_label_burn_excluded_count"], 1)
            self.assertEqual(counters["analysis_eligible_count"], 3)
            self.assertEqual(counters["predictor_invalid_count_among_eligible"], 0)


# =============================================================================
# 7. Excluded cell_id values never survive Step8B's own
#    valid_for_modeling==True filter (pure-function test; no model training).
# =============================================================================
class TestStep8BNeverSeesExcludedCells(unittest.TestCase):
    def test_07_filter_valid_for_modeling_drops_excluded_cells(self):
        df = pd.DataFrame({
            "cell_id": ["r0_c0", "r0_c1", "r1_c0", "r1_c1"],
            "burned": [1, 0, 0, 0],
            "pre_label_burn_excluded": [False, False, False, True],
            "analysis_eligible": [True, True, True, False],
            "valid_for_modeling": [True, True, True, False],
            "invalid_reason": [None, None, None, "pre_label_burn_excluded"],
        })
        filtered = filter_valid_for_modeling(df)
        self.assertNotIn("r1_c1", set(filtered["cell_id"]))
        self.assertEqual(set(filtered["cell_id"]), {"r0_c0", "r0_c1", "r1_c0"})


if __name__ == "__main__":
    unittest.main()