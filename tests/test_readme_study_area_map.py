"""
tests/test_readme_study_area_map.py

Targeted tests for the README study-area map generator
(`scripts/make_readme_study_area_map.py`). Read-only: no Earth Engine call,
no network access, no figure rendering, no raster/parquet/output is read or
written -- only the registry-resolution layer is exercised.

The figure itself is decorative, but what it CLAIMS is not: the AOI
rectangles must be the real `core/regions.py` bboxes and the cohort must be
the same registry/role-derived set the pipeline uses. These tests guard that
contract against refactors of `build_regions()`.

Covers:
    1  AST bbox extraction reproduces the module-level *_AOI_BBOX constants
    2  aliased assignments (manavgat_aoi = manavgat_aoi_refined_bbox) resolve
    3  every extracted bbox is a well-formed, in-range lon/lat rectangle
    4  the resolved cohort equals the registry/role-derived cohort and
       excludes cohort-dISI roles (kozan_2023, mugla_2022_event_relative)
    5  mugla_2022_event_relative is a geometry-SHARING reference: identical
       bbox to mugla_2021, drawn as an annotation rather than a 6th AOI
    6  kozan_2023 is reported as unmapped (own buffer-based geometry), not
       silently dropped
    7  a cohort AOI whose bbox cannot be resolved fails closed
    8  short labels stay compact and single-line

Run:
    python -m unittest tests.test_readme_study_area_map
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.regions as regions
from core.regions import EXPERIMENTS
from src.burned_pattern_audit import NON_COHORT_ROLES


def _load_map_module():
    """scripts/ bir paket olmadigi icin script'i dosya yolundan yukler."""
    path = PROJECT_ROOT / "scripts" / "make_readme_study_area_map.py"
    spec = importlib.util.spec_from_file_location("make_readme_study_area_map", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


study_area_map = _load_map_module()


class TestBboxExtraction(unittest.TestCase):
    """AST ile cozulen geometriler core/regions.py ile birebir ayni mi?"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bboxes = study_area_map.extract_region_bboxes()

    def test_module_level_constants_are_reproduced(self) -> None:
        # (1) *_AOI_BBOX sabitleri TEK KAYNAK; harita onlari yeniden yazmaz.
        expected = {
            "mugla_aoi": regions.MUGLA_AOI_BBOX,
            "north_evia": regions.NORTH_EVIA_AOI_BBOX,
            "north_evia_extended": regions.NORTH_EVIA_EXTENDED_AOI_BBOX,
            "montiferru_aoi": regions.MONTIFERRU_AOI_BBOX,
        }
        for region_key, constant in expected.items():
            with self.subTest(region_key=region_key):
                self.assertIn(region_key, self.bboxes)
                self.assertEqual(
                    self.bboxes[region_key], tuple(float(v) for v in constant)
                )

    def test_aliased_region_keys_resolve_to_the_same_geometry(self) -> None:
        # (2) manavgat_aoi = manavgat_aoi_refined_bbox (takma ad zinciri)
        self.assertIn("manavgat_aoi", self.bboxes)
        self.assertEqual(
            self.bboxes["manavgat_aoi"], self.bboxes["manavgat_aoi_refined_bbox"]
        )
        self.assertEqual(self.bboxes["mugla_aoi"], self.bboxes["mugla_aoi_candidate_bbox"])

    def test_extracted_bboxes_are_well_formed(self) -> None:
        # (3) her bbox (lon_min, lat_min, lon_max, lat_max) ve gecerli aralikta
        for region_key, bbox in self.bboxes.items():
            with self.subTest(region_key=region_key):
                lon_min, lat_min, lon_max, lat_max = bbox
                self.assertLess(lon_min, lon_max)
                self.assertLess(lat_min, lat_max)
                self.assertGreaterEqual(lon_min, -180.0)
                self.assertLessEqual(lon_max, 180.0)
                self.assertGreaterEqual(lat_min, -90.0)
                self.assertLessEqual(lat_max, 90.0)

    def test_unresolvable_cohort_geometry_fails_closed(self) -> None:
        # (7) bbox'siz bir cohort AOI'si SESSIZCE atlanmaz.
        original = study_area_map.extract_region_bboxes
        try:
            study_area_map.extract_region_bboxes = lambda: {}
            with self.assertRaises(study_area_map.StudyAreaMapError):
                study_area_map.resolve_study_areas()
        finally:
            study_area_map.extract_region_bboxes = original


class TestStudyAreaResolution(unittest.TestCase):
    """Cohort / referans / haritalanmayan ayrimi registry'den mi turuyor?"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cohort, cls.references, cls.unmapped = study_area_map.resolve_study_areas()

    def test_cohort_matches_registry_role_filter(self) -> None:
        # (4) harita kendi cohort tanimini UYDURMAZ; registry+role filtresi.
        expected = {
            experiment_id
            for experiment_id, record in EXPERIMENTS.items()
            if record.get("enabled")
            and record.get("variant_status") == "canonical"
            and record.get("role") not in NON_COHORT_ROLES
        }
        self.assertEqual({e["experiment_id"] for e in self.cohort}, expected)

    def test_non_cohort_roles_are_not_cohort_members(self) -> None:
        cohort_ids = {e["experiment_id"] for e in self.cohort}
        self.assertNotIn("kozan_2023", cohort_ids)
        self.assertNotIn("mugla_2022_event_relative", cohort_ids)
        for entry in self.cohort:
            with self.subTest(experiment_id=entry["experiment_id"]):
                self.assertNotIn(entry["role"], NON_COHORT_ROLES)

    def test_cohort_is_ordered_west_to_east(self) -> None:
        centers = [0.5 * (e["bbox"][0] + e["bbox"][2]) for e in self.cohort]
        self.assertEqual(centers, sorted(centers))

    def test_event_relative_reference_shares_the_cohort_geometry(self) -> None:
        # (5) ayni AOI geometrisi -> ayri/kaydirilmis bir kutu CIZILMEZ.
        reference_ids = {e["experiment_id"] for e in self.references}
        self.assertIn("mugla_2022_event_relative", reference_ids)

        reference = next(
            e for e in self.references if e["experiment_id"] == "mugla_2022_event_relative"
        )
        mugla_2021 = next(
            e for e in self.cohort if e["experiment_id"] == "mugla_2021"
        )
        self.assertEqual(reference["region_key"], mugla_2021["region_key"])
        self.assertEqual(reference["bbox"], mugla_2021["bbox"])

    def test_negative_control_is_reported_as_unmapped(self) -> None:
        # (6) kozan_2023 kendi (buffer tabanli) geometrisine sahip: haritada
        # yok, ama sessizce dusurulmuyor -- gerekcesiyle raporlaniyor.
        unmapped_ids = {e["experiment_id"] for e in self.unmapped}
        self.assertIn("kozan_2023", unmapped_ids)
        self.assertNotIn("kozan_2023", {e["experiment_id"] for e in self.references})

    def test_labels_are_compact_and_single_line(self) -> None:
        # (8) etiketler harita uzerinde tasmasin diye kisa ve tek satir.
        for entry in self.cohort:
            with self.subTest(experiment_id=entry["experiment_id"]):
                self.assertNotIn("\n", entry["label"])
                self.assertNotIn("--", entry["label"])
                self.assertLessEqual(len(entry["label"]), 24)
                self.assertTrue(entry["country"])

    def test_evia_label_reflects_the_canonical_extended_variant(self) -> None:
        evia = next(
            e for e in self.cohort if e["experiment_id"] == "evia_2021_extended"
        )
        self.assertEqual(evia["region_key"], "north_evia_extended")
        self.assertEqual(
            evia["bbox"], tuple(float(v) for v in regions.NORTH_EVIA_EXTENDED_AOI_BBOX)
        )

    def test_bbox_size_km_is_positive_and_plausible(self) -> None:
        for entry in self.cohort:
            with self.subTest(experiment_id=entry["experiment_id"]):
                width_km, height_km = study_area_map.bbox_size_km(entry["bbox"])
                self.assertGreater(width_km, 1.0)
                self.assertGreater(height_km, 1.0)
                self.assertLess(width_km, 1000.0)
                self.assertLess(height_km, 1000.0)


if __name__ == "__main__":
    unittest.main()
