"""
tests/test_evia_2021_extended_registry.py

Targeted tests for the `evia_2021_extended` registry entry and its extended
North Evia AOI. Read-only: no Earth Engine call, no export, no gate, no
model, no raster/parquet is read or written.

Covers (task numbering):
    1  evia_2021 is still registered and its metadata is unchanged
    2  evia_2021_extended is registered and enabled
    3  predictor/label dates and baseline years equal evia_2021's
    4  the extended AOI geometrically contains the evia_2021 AOI
    5  the extended AOI area is larger than the evia_2021 AOI area
    6  the output namespace is exactly evia_2021_extended
    7  no resolved path leaks into the evia_2021 namespace
    8  the Step0 registry validator passes for evia_2021_extended
    9  a dry-run runs no real pipeline stage and no GEE call

Run:
    python -m unittest tests.test_evia_2021_extended_registry
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.pipeline_orchestrator as orch
import core.regions as regions
from core.config import EXPORT_CRS

OLD_ID = "evia_2021"
NEW_ID = "evia_2021_extended"

R_EARTH_KM = 6371.0088


def _bbox_area_km2(bbox) -> float:
    """Spherical-Earth area of a lon/lat rectangle (no geospatial dependency)."""
    west, south, east, north = bbox
    dlon = math.radians(east - west)
    return R_EARTH_KM ** 2 * dlon * (math.sin(math.radians(north)) - math.sin(math.radians(south)))


# =============================================================================
# Existing Evia experiment unchanged (test 1)
# =============================================================================
class TestEvia2021Unchanged(unittest.TestCase):
    def test_01_evia_2021_still_registered_with_frozen_metadata(self):
        exp = regions.get_experiment(OLD_ID)
        self.assertTrue(exp["enabled"])
        self.assertEqual(exp["region_key"], "north_evia")
        self.assertEqual(exp["role"], "mediterranean_transfer_wildfire")
        self.assertEqual(exp["country"], "Greece")
        self.assertEqual(exp["predictor_start_date"], "2021-06-05")
        self.assertEqual(exp["predictor_end_date"], "2021-08-02")
        self.assertEqual(exp["label_start_date"], "2021-08-03")
        self.assertEqual(exp["label_end_date"], "2021-09-30")
        self.assertEqual(exp["baseline_years"], [2017, 2018, 2019, 2020])
        self.assertEqual(exp["output_namespace"], OLD_ID)
        self.assertTrue(exp["exclude_pre_label_burns"])
        self.assertEqual(exp["pre_label_burn_window"], ["2021-06-05", "2021-08-02"])

    def test_01_evia_2021_aoi_bbox_is_frozen(self):
        self.assertEqual(regions.NORTH_EVIA_AOI_BBOX, (23.12, 38.68, 23.52, 39.08))


# =============================================================================
# New experiment registration (tests 2/3/6)
# =============================================================================
class TestEvia2021ExtendedRegistration(unittest.TestCase):
    def setUp(self):
        self.old = regions.get_experiment(OLD_ID)
        self.new = regions.get_experiment(NEW_ID)

    def test_02_registered_and_enabled(self):
        self.assertIn(NEW_ID, regions.EXPERIMENTS)
        self.assertTrue(self.new["enabled"])
        self.assertIn(NEW_ID, regions.list_experiments(include_disabled=False))

    def test_02_region_key_is_separate_and_resolvable(self):
        self.assertEqual(self.new["region_key"], "north_evia_extended")
        self.assertNotEqual(self.new["region_key"], self.old["region_key"])

    def test_03_dates_and_baselines_match_evia_2021(self):
        for key in ("predictor_start_date", "predictor_end_date",
                    "label_start_date", "label_end_date", "baseline_years"):
            self.assertEqual(self.new[key], self.old[key], key)

    def test_03_scientific_metadata_copied_from_evia_2021(self):
        for key in ("role", "country", "exclude_pre_label_burns",
                    "pre_label_burn_window"):
            self.assertEqual(self.new[key], self.old[key], key)

    def test_03_only_descriptive_metadata_differs(self):
        for key in ("region_key", "display_name", "output_namespace", "notes"):
            self.assertNotEqual(self.new[key], self.old[key], key)

    def test_06_output_namespace_is_exactly_the_new_experiment(self):
        self.assertEqual(self.new["output_namespace"], NEW_ID)
        namespaces = [e["output_namespace"] for e in regions.EXPERIMENTS.values()]
        self.assertEqual(len(namespaces), len(set(namespaces)), "duplicate output_namespace")

    def test_06_experiment_ids_do_not_collide(self):
        ids = list(regions.EXPERIMENTS)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(OLD_ID, ids)
        self.assertIn(NEW_ID, ids)


# =============================================================================
# Extended AOI geometry (tests 4/5)
# =============================================================================
class TestExtendedAOIGeometry(unittest.TestCase):
    def setUp(self):
        self.old_bbox = regions.NORTH_EVIA_AOI_BBOX
        self.new_bbox = regions.NORTH_EVIA_EXTENDED_AOI_BBOX

    def test_bbox_is_frozen_and_well_ordered(self):
        self.assertEqual(self.new_bbox, (23.05, 38.55, 23.85, 39.15))
        west, south, east, north = self.new_bbox
        # ee.Geometry.BBox order = (lon_min, lat_min, lon_max, lat_max)
        self.assertLess(west, east)
        self.assertLess(south, north)
        self.assertEqual(EXPORT_CRS, "EPSG:4326")

    def test_04_extended_aoi_contains_evia_2021_aoi(self):
        ow, os_, oe, on = self.old_bbox
        nw, ns, ne, nn = self.new_bbox
        self.assertLessEqual(nw, ow, "extended west edge must not cut the old AOI")
        self.assertLessEqual(ns, os_, "extended south edge must not cut the old AOI")
        self.assertGreaterEqual(ne, oe, "extended east edge must not cut the old AOI")
        self.assertGreaterEqual(nn, on, "extended north edge must not cut the old AOI")

    def test_05_extended_aoi_area_is_larger(self):
        old_area = _bbox_area_km2(self.old_bbox)
        new_area = _bbox_area_km2(self.new_bbox)
        self.assertGreater(new_area, 0.0)
        self.assertGreater(new_area, old_area)

    def test_no_degenerate_or_antimeridian_geometry(self):
        for bbox in (self.old_bbox, self.new_bbox):
            west, south, east, north = bbox
            self.assertGreater(_bbox_area_km2(bbox), 0.0)
            self.assertTrue(-180.0 <= west < east <= 180.0)
            self.assertTrue(-90.0 <= south < north <= 90.0)
            self.assertLess(east - west, 180.0, "bbox must not wrap the antimeridian")

    def test_north_evia_corridor_places_are_covered(self):
        # Place-based anchors only (documented town coordinates); NEVER derived
        # from burned labels, burned prevalence, gate outcome or model metrics.
        west, south, east, north = self.new_bbox
        corridor = {"Rovies": (23.22, 38.78), "Limni": (23.33, 38.77),
                    "Agia Anna": (23.42, 38.87), "Pefki": (23.67, 39.02)}
        for place, (lon, lat) in corridor.items():
            self.assertTrue(west <= lon <= east and south <= lat <= north,
                            f"{place} outside the extended AOI")
        # Pefki is the corridor end the evia_2021 AOI truncates -- the reason
        # the extended AOI exists.
        self.assertGreater(corridor["Pefki"][0], regions.NORTH_EVIA_AOI_BBOX[2])

    def test_region_registry_keys_are_declared(self):
        # build_regions() is never called here (it would need ee.Initialize);
        # the source is read statically instead.
        source = (_PROJECT_ROOT / "core" / "regions.py").read_text(encoding="utf-8")
        self.assertIn('"north_evia_extended": north_evia_extended', source)
        self.assertIn("ee.Geometry.BBox(*NORTH_EVIA_EXTENDED_AOI_BBOX)", source)
        # the existing Evia region is still built from its own frozen constant
        self.assertIn("ee.Geometry.BBox(*NORTH_EVIA_AOI_BBOX)", source)


# =============================================================================
# Namespace isolation (tests 6/7)
# =============================================================================
class TestNamespaceIsolation(unittest.TestCase):
    def test_06_output_root_resolves_under_the_new_namespace(self):
        root = regions.get_experiment_output_root(NEW_ID)
        self.assertEqual(root.name, NEW_ID)
        self.assertEqual(root.parent.name, "experiments")
        self.assertEqual(root, _PROJECT_ROOT / "outputs" / "experiments" / NEW_ID)

    def test_07_no_step_path_leaks_into_the_evia_2021_namespace(self):
        old_root = regions.get_experiment_output_root(OLD_ID)
        for step in ("step0", "step5", "step6a", "step6b", "step7", "step8a", "step8"):
            path = regions.get_step_output_dir(NEW_ID, step)
            self.assertIn(NEW_ID, path.parts)
            self.assertNotIn(old_root, path.parents)
            self.assertNotIn(f"/experiments/{OLD_ID}/", f"{path}/")

    def test_07_step_dir_resolution_creates_nothing(self):
        root = regions.get_experiment_output_root(NEW_ID)
        existed = root.exists()
        regions.get_step_output_dir(NEW_ID, "step0")
        self.assertEqual(root.exists(), existed, "path resolution must not create directories")

    def test_07_evia_2021_output_root_is_untouched(self):
        self.assertEqual(regions.get_experiment_output_root(OLD_ID),
                         _PROJECT_ROOT / "outputs" / "experiments" / OLD_ID)


# =============================================================================
# Step0 registry validator (test 8)
# =============================================================================
class TestRegistryValidation(unittest.TestCase):
    def test_08_check_experiment_registry_passes(self):
        from scripts.check_experiment_registry import main as check_registry

        result = check_registry(experiment_id=NEW_ID)
        self.assertTrue(result["valid"])
        self.assertEqual(result["experiment_id"], NEW_ID)
        self.assertEqual(result["region_key"], "north_evia_extended")
        self.assertTrue(result["output_root"].endswith(f"experiments/{NEW_ID}"))

    def test_08_registry_validation_still_passes_for_evia_2021(self):
        from scripts.check_experiment_registry import main as check_registry

        self.assertTrue(check_registry(experiment_id=OLD_ID)["valid"])

    def test_08_plan_is_namespaced_to_the_new_experiment(self):
        plan = orch.describe_experiment_plan(NEW_ID, "gate", "step8", "export", False)
        self.assertEqual(plan["experiment_id"], NEW_ID)
        self.assertFalse(plan["is_kozan"])
        self.assertIn(NEW_ID, Path(plan["output_root"]).parts)
        self.assertNotIn(OLD_ID, Path(plan["output_root"]).parts)


# =============================================================================
# Dry-run executes nothing (test 9)
# =============================================================================
class TestDryRunNoExecution(unittest.TestCase):
    def _snapshot(self) -> set[Path]:
        root = _PROJECT_ROOT / "outputs" / "experiments"
        return set(root.rglob("*")) if root.exists() else set()

    def test_09_dry_run_runs_no_stage_and_writes_no_file(self):
        before = self._snapshot()
        with patch("core.gee_utils.init_gee",
                   side_effect=AssertionError("dry-run must not initialize Earth Engine")):
            result = orch.run_experiment_plan(
                experiment_id=NEW_ID, from_stage="gate", to_stage="step8",
                predictor_mode="export", export_labels=False,
                dry_run=True, force=False,
            )
        after = self._snapshot()

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["experiment_id"], NEW_ID)
        for stage, stage_result in result["stage_results"].items():
            self.assertFalse(stage_result.get("ran"), f"stage '{stage}' reported ran=True")

        new_files = {p for p in after - before if p.is_file()}
        self.assertEqual(new_files, set(), f"dry-run wrote files: {new_files}")
        leaked = {p for p in after - before if f"/experiments/{OLD_ID}/" in f"{p}/"}
        self.assertEqual(leaked, set(), f"dry-run touched the evia_2021 namespace: {leaked}")


if __name__ == "__main__":
    unittest.main()
