"""
tests/test_mugla_2021_gate.py

Targeted tests for the Muğla 2021 leakage-safe burned-landcover gate.

Covers (task numbering):
    1  mugla_2021 dates are exact
    2  baseline years are 2017-2020
    3  label window starts 2021-07-29
    4  a BurnDate before 2021-07-29 is marked pre_label
    5  a BurnDate inside the label window is labeled in_window (burned)
    6  a zero / missing BurnDate is not labeled burned (unmapped)
    7  pre-label excluded cells cannot be modeling-eligible (not in analysis universe)
    8  pre-label excluded cells cannot enter the gate denominator
    9  pre-label excluded cells are not converted into negatives
    10 label-window burned cells remain eligible (burned)
    11 block / grid behavior remains at native ~500 m
    12 gate uses tree+shrub+grass composition
    13 tree+shrub and tree+shrub+grass counts are reported separately
    14 failure of the natural-vegetation gate blocks downstream execution
    15 passing the gate still leaves downstream_authorized=False
    16 existing experiment configurations remain unchanged
    17 dry-run writes no files
    18 gate-only execution does not call Step7/Step8/Step9/Step10
    19 AOI coordinates satisfy the expected CRS and order
    20 protected historical output hashes are recorded in the manifest

Run:
    python -m unittest tests.test_mugla_2021_gate
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.regions as regions
from core.config import EXPORT_CRS
from src.step6b_burned_landcover_gate import (
    classify_gate_decision,
    compute_gate,
    _date_to_doy,
)
from src.step8a_prepare_500m_modeling_dataset import (
    classify_burndate_relative_to_label,
    compute_block_size_pixels,
)

LS, LE = "2021-07-29", "2021-09-15"


# =============================================================================
# Config / registry
# =============================================================================
class TestMuglaConfig(unittest.TestCase):
    def setUp(self):
        self.exp = regions.get_experiment("mugla_2021")

    def test_01_dates_exact(self):
        self.assertEqual(self.exp["predictor_start_date"], "2021-06-01")
        self.assertEqual(self.exp["predictor_end_date"], "2021-07-28")
        self.assertEqual(self.exp["label_start_date"], "2021-07-29")
        self.assertEqual(self.exp["label_end_date"], "2021-09-15")

    def test_02_baseline_years(self):
        self.assertEqual(self.exp["baseline_years"], [2017, 2018, 2019, 2020])

    def test_03_label_starts_0729(self):
        self.assertEqual(self.exp["label_start_date"], "2021-07-29")

    def test_enabled_and_exclusion_flag(self):
        self.assertTrue(self.exp["enabled"])
        self.assertTrue(self.exp["exclude_pre_label_burns"])
        self.assertEqual(self.exp["pre_label_burn_window"], ["2021-06-01", "2021-07-28"])
        self.assertEqual(self.exp["region_key"], "mugla_aoi")


# =============================================================================
# Pure BurnDate classifier (tests 4/5/6)
# =============================================================================
class TestBurnDateClassifier(unittest.TestCase):
    def test_04_pre_label(self):
        # Bördübet fire (~2021-06-21..25) -> DOY 172..176 -> pre_label
        self.assertEqual(classify_burndate_relative_to_label(172, LS, LE), "pre_label")
        self.assertEqual(classify_burndate_relative_to_label(176, LS, LE), "pre_label")
        # day before label_start (2021-07-28 = DOY 209)
        self.assertEqual(classify_burndate_relative_to_label(209, LS, LE), "pre_label")

    def test_05_in_window(self):
        self.assertEqual(classify_burndate_relative_to_label(210, LS, LE), "in_window")  # 07-29
        self.assertEqual(classify_burndate_relative_to_label(220, LS, LE), "in_window")  # 08-08
        self.assertEqual(classify_burndate_relative_to_label(258, LS, LE), "in_window")  # 09-15

    def test_06_zero_missing_unmapped(self):
        self.assertEqual(classify_burndate_relative_to_label(0, LS, LE), "unmapped")
        self.assertEqual(classify_burndate_relative_to_label(float("nan"), LS, LE), "unmapped")
        self.assertEqual(classify_burndate_relative_to_label(-5, LS, LE), "unmapped")

    def test_post_label(self):
        self.assertEqual(classify_burndate_relative_to_label(259, LS, LE), "post_label")  # 09-16

    def test_bordubet_doy_bounds(self):
        self.assertEqual(_date_to_doy("2021-06-21"), 172)
        self.assertEqual(_date_to_doy("2021-06-25"), 176)


# =============================================================================
# Gate decision logic (tests 12/14/15 rely on this + gate)
# =============================================================================
class TestGateDecision(unittest.TestCase):
    def test_12_uses_tree_shrub_grass_fraction(self):
        # natural_fraction (tree+shrub+grass) drives the pass, not cropland
        d, _ = classify_gate_decision(100, 0.60, 0.10, 30, 0.5, 0.5)
        self.assertEqual(d, "wildfire_candidate_pass")

    def test_cropland_and_mixed_and_insufficient(self):
        self.assertEqual(classify_gate_decision(100, 0.2, 0.7, 30, 0.5, 0.5)[0], "cropland_dominated_control")
        self.assertEqual(classify_gate_decision(100, 0.3, 0.3, 30, 0.5, 0.5)[0], "mixed_or_uncertain")
        self.assertEqual(classify_gate_decision(10, 0.9, 0.0, 30, 0.5, 0.5)[0], "insufficient_burned_positives")


# =============================================================================
# Grid / block (test 11)
# =============================================================================
class TestBlockGrid(unittest.TestCase):
    def test_11_native_500m_block(self):
        # 500 m native MCD64A1 cell reconstructed from 30 m reference grid.
        self.assertEqual(compute_block_size_pixels(), 17)  # round(500/30)


# =============================================================================
# AOI (test 19)
# =============================================================================
class TestMuglaAOI(unittest.TestCase):
    def test_19_bbox_order_crs_and_city_coverage(self):
        w, s, e, n = regions.MUGLA_AOI_BBOX
        # ee.Geometry.BBox order = (lon_min, lat_min, lon_max, lat_max)
        self.assertLess(w, e, "lon_min must be < lon_max")
        self.assertLess(s, n, "lat_min must be < lat_max")
        self.assertEqual(EXPORT_CRS, "EPSG:4326")
        cities = {
            "Bodrum": (27.43, 37.03), "Milas": (27.78, 37.32),
            "Marmaris": (28.27, 36.85), "Koycegiz": (28.69, 36.97),
        }
        for city, (lon, lat) in cities.items():
            self.assertTrue(w <= lon <= e and s <= lat <= n, f"{city} outside AOI bbox")


# =============================================================================
# Existing experiments unchanged (test 16)
# =============================================================================
class TestExistingExperimentsUnchanged(unittest.TestCase):
    def test_16_snapshots(self):
        expected = {
            "kozan_2023": ("2023-08-01", "2023-10-31", [2019, 2020, 2021, 2022]),
            "manavgat_2021": ("2021-07-28", "2021-08-31", [2017, 2018, 2019, 2020]),
            "bejis_2022": ("2022-08-15", "2022-09-30", [2018, 2019, 2020, 2021]),
        }
        for exp_id, (ls, le, base) in expected.items():
            e = regions.get_experiment(exp_id)
            self.assertEqual(e["label_start_date"], ls, exp_id)
            self.assertEqual(e["label_end_date"], le, exp_id)
            self.assertEqual(e["baseline_years"], base, exp_id)
        # none of the existing experiments opt into pre-label exclusion
        for exp_id in ("kozan_2023", "manavgat_2021", "bejis_2022"):
            self.assertFalse(regions.get_experiment(exp_id).get("exclude_pre_label_burns", False))


# =============================================================================
# End-to-end synthetic gate (tests 7/8/9/10/13/14/15)
# =============================================================================
class TestSyntheticGateExclusion(unittest.TestCase):
    """Builds tiny real GeoTIFFs (no GEE) and runs compute_gate."""

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

    def _build(self, tmp, lc_code=10):
        d = Path(tmp)
        ref = np.ones((self.H, self.W), np.float32)
        lc = np.full((self.H, self.W), lc_code, np.float32)
        label = np.zeros((self.H, self.W), np.float32)
        pre = np.zeros((self.H, self.W), np.float32)
        label[self._cell(0, 0)] = 220   # in-window burn (Aug 8)
        pre[self._cell(1, 1)] = 172     # pre-label burn (Jun 21, Bördübet)
        refp, lcp, lblp, prep = (d / x for x in ["ref.tif", "lc.tif", "label.tif", "pre.tif"])
        self._write(refp, ref, "int16")
        self._write(lcp, lc, "uint8", nodata=0)
        self._write(lblp, label)
        self._write(prep, pre)
        return refp, lcp, lblp, prep, d

    def _run(self, tmp, excl, lc_code=10):
        refp, lcp, lblp, prep, d = self._build(tmp, lc_code=lc_code)
        return compute_gate(
            label_path=lblp, label_kind="raw_burndate", reference_path=refp,
            landcover_path=lcp, label_start=LS, label_end=LE, output_dir=d,
            min_positives=1, natural_threshold=0.5, cropland_threshold=0.5,
            exclude_pre_label_burns=excl,
            pre_label_label_path=(prep if excl else None),
            predictor_start="2021-06-01", predictor_end="2021-07-28",
            bordubet_check_window=("2021-06-21", "2021-06-25"),
        )

    def test_07_excluded_not_modeling_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = self._run(tmp, excl=True)
            # excluded cell is neither burned nor unburned -> cannot become a
            # modeling row / valid_for_modeling at the gate universe level.
            universe = g["burned_count"] + g["unburned_count"]
            self.assertEqual(g["analysis_universe_cells_after_exclusions"], universe)
            self.assertEqual(g["pre_label_burn_excluded_count"], 1)
            self.assertEqual(
                g["pre_label_burn_excluded_count"] + universe,
                g["total_valid_cells_or_pixels_considered"],
            )

    def test_08_excluded_not_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = self._run(tmp, excl=True)
            # burned_count (the fraction denominator) does NOT include the
            # pre-label cell.
            self.assertEqual(g["burned_count"], 1)

    def test_09_excluded_not_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            g_excl = self._run(tmp, excl=True)
        with tempfile.TemporaryDirectory() as tmp:
            g_legacy = self._run(tmp, excl=False)
        # With exclusion the pre-label cell is dropped (unburned=2); without it,
        # the SAME cell leaks in as an unburned negative (unburned=3).
        self.assertEqual(g_excl["unburned_count"], 2)
        self.assertEqual(g_legacy["unburned_count"], 3)
        self.assertEqual(g_legacy["pre_label_burn_excluded_count"], 0)

    def test_10_label_window_burn_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = self._run(tmp, excl=True)
            self.assertEqual(g["burned_count"], 1)
            self.assertEqual(g["temporal_label_qa"]["count_within_label_window"], 1)
            self.assertEqual(g["temporal_label_qa"]["bordubet_window_burned_cell_count"], 1)

    def test_13_tree_shrub_and_tsg_reported_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = self._run(tmp, excl=True)
            self.assertIn("burned_tree_shrub_count", g)
            self.assertIn("burned_tree_shrub_grass_count", g)
            b = g["pre_label_burn_excluded_breakdown"]
            self.assertIn("tree_shrub", b)
            self.assertIn("tree_shrub_grass", b)
            # excluded cell is tree_cover -> tree_shrub == tree_shrub_grass == 1
            self.assertEqual(b["tree_shrub"], 1)
            self.assertEqual(b["tree_shrub_grass"], 1)

    def test_14_failed_gate_blocks_downstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            # all-cropland landcover -> not a natural-veg pass
            g = self._run(tmp, excl=True, lc_code=40)  # 40 = cropland
            self.assertNotEqual(g["decision"], "wildfire_candidate_pass")
            self.assertFalse(g["downstream_authorized"])

    def test_15_pass_still_blocks_downstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = self._run(tmp, excl=True)  # tree_cover -> pass
            self.assertEqual(g["decision"], "wildfire_candidate_pass")
            self.assertFalse(g["downstream_authorized"])


# =============================================================================
# Dry-run + stage isolation + manifest (tests 17/18/20)
# =============================================================================
class TestRunnerBehaviour(unittest.TestCase):
    def test_17_dry_run_writes_no_files(self):
        from scripts.run_label_gate_only import main as gate_main, _namespaced_paths
        res = gate_main(experiment_id="mugla_2021", dry_run=True)
        self.assertFalse(res["ran"])
        self.assertEqual(res["reason"], "dry_run")
        paths = _namespaced_paths("mugla_2021")
        for key in ["raw_path", "pre_label_raw_path", "manifest_path"]:
            self.assertFalse(Path(paths[key]).exists(),
                             f"dry-run must not create {key}")

    def test_18_gate_only_no_step7_step8(self):
        import core.pipeline_orchestrator as orch
        # gate->gate resolves to exactly ['gate']
        self.assertEqual(orch.validate_stage_range("gate", "gate"), ["gate"])
        # the gate runner source references no Step7/Step8/Step9/Step10 runners
        src = (_PROJECT_ROOT / "scripts" / "run_label_gate_only.py").read_text()
        for forbidden in ("run_step7", "run_step8", "run_step9", "run_step10",
                          "run_predictors_only"):
            self.assertNotIn(forbidden, src, f"gate runner must not call {forbidden}")

    def test_20_manifest_records_protected_hashes_and_downstream_false(self):
        from scripts.run_label_gate_only import build_gate_manifest, _namespaced_paths
        exp = regions.get_experiment("mugla_2021")
        paths = _namespaced_paths("mugla_2021")
        m = build_gate_manifest("mugla_2021", exp, paths,
                                {"gate_result": {"decision": "x", "burned_count": 0, "json_path": "j"}})
        self.assertFalse(m["downstream_authorized"])
        self.assertEqual(len(m["analysis_id"]), 64)
        protected = m["protected_gate_report_hashes"]
        self.assertTrue(any("manavgat_2021" in k for k in protected))
        self.assertTrue(any("bejis_2022" in k for k in protected))
        self.assertTrue(any("validation/labels" in k for k in protected))  # kozan legacy


if __name__ == "__main__":
    unittest.main()