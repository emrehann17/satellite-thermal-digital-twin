"""
tests/test_modis_qc_valid_count.py

Focused tests for the MOD11A1 QC_Day masking rule and the per-pixel
valid-observation-count threshold used by
scripts/prepare_modis_for_step7.py (_qc_accept_mask,
_build_qc_masked_modis_stack) -- see core/config.py STEP7_MODIS_QC_* /
STEP7_MODIS_MIN_VALID_OBSERVATIONS.

No live Earth Engine is used: `_qc_accept_mask` (the REAL production
function) is exercised directly against a minimal numpy-backed stand-in for
ee.Image that implements exactly the handful of methods it calls
(select/bitwiseAnd/rightShift/eq/gt/And) as elementwise numpy operations --
this is not a re-implementation of the QC rule, it runs the actual function
under test.

Covers:
    4. QC-masked observations are excluded from temporal mean/std
       (TestQcAcceptMask, TestTemporalMeanExcludesQcRejected)
    5. low-observation-count pixels remain nodata
       (TestValidObservationThreshold)

Run:
    python -m unittest tests.test_modis_qc_valid_count
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import (
    STEP7_MODIS_MIN_VALID_OBSERVATIONS,
    STEP7_MODIS_QC_DATA_QUALITY_ACCEPT,
    STEP7_MODIS_QC_DATA_QUALITY_MASK,
    STEP7_MODIS_QC_DATA_QUALITY_SHIFT,
    STEP7_MODIS_QC_MANDATORY_QA_ACCEPT,
    STEP7_MODIS_QC_MANDATORY_QA_MASK,
)
from scripts.prepare_modis_for_step7 import _qc_accept_mask


# =============================================================================
# Minimal numpy-backed stand-in for ee.Image -- only the operations
# _qc_accept_mask actually calls.
# =============================================================================
class _ArrayBand:
    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def bitwiseAnd(self, val):
        return _ArrayBand(self.arr & val)

    def rightShift(self, val):
        return _ArrayBand(self.arr >> val)

    def eq(self, val):
        return _ArrayBand(self.arr == val)

    def gt(self, val):
        return _ArrayBand(self.arr > val)

    def And(self, other: "_ArrayBand"):
        return _ArrayBand(self.arr & other.arr)


class _ArrayImage:
    def __init__(self, bands: dict[str, np.ndarray]):
        self._bands = bands

    def select(self, name: str) -> _ArrayBand:
        return _ArrayBand(self._bands[name])


def _accept(qc_value: int, dn_value: int) -> bool:
    qc = np.array([qc_value], dtype="int32")
    dn = np.array([dn_value], dtype="int32")
    image = _ArrayImage({"LST_Day_1km": dn, "QC_Day": qc})
    result = _qc_accept_mask(image)
    return bool(result.arr[0])


class TestQcAcceptMask(unittest.TestCase):
    """Directly exercises the real _qc_accept_mask() production function."""

    def test_good_quality_good_data_accepted(self):
        # bits 0-1 = 00 (good quality), bits 2-3 = 00 (good data quality)
        self.assertTrue(_accept(qc_value=0b00000000, dn_value=15000))

    def test_mandatory_qa_other_quality_rejected(self):
        # bits 0-1 = 01 ("other quality, recommend examination")
        self.assertFalse(_accept(qc_value=0b00000001, dn_value=15000))

    def test_mandatory_qa_cloud_rejected(self):
        # bits 0-1 = 10 (LST not produced, cloud)
        self.assertFalse(_accept(qc_value=0b00000010, dn_value=15000))

    def test_mandatory_qa_other_reason_rejected(self):
        # bits 0-1 = 11 (LST not produced, other reasons)
        self.assertFalse(_accept(qc_value=0b00000011, dn_value=15000))

    def test_data_quality_other_rejected(self):
        # bits 0-1 = 00 (good), bits 2-3 = 01 (other data quality)
        self.assertFalse(_accept(qc_value=0b00000100, dn_value=15000))

    def test_emissivity_and_lst_error_bits_are_not_considered(self):
        # bits 0-1 = 00, bits 2-3 = 00, but bits 4-7 set to their "worst"
        # documented values (emissivity error > 0.04, LST error > 3K).
        # Task scope is mandatory QA + data quality bits ONLY -- these must
        # NOT affect the accept decision (no undocumented/extra bits invented).
        self.assertTrue(_accept(qc_value=0b11110000, dn_value=15000))

    def test_dn_zero_rejected_even_with_good_qc(self):
        self.assertFalse(_accept(qc_value=0b00000000, dn_value=0))

    def test_dn_negative_rejected_even_with_good_qc(self):
        self.assertFalse(_accept(qc_value=0b00000000, dn_value=-5))

    def test_bitmask_constants_match_documented_mod11a1_layout(self):
        # bits 0-1 and bits 2-3 (after a 2-bit right shift), both 2-bit fields.
        self.assertEqual(STEP7_MODIS_QC_MANDATORY_QA_MASK, 0b0011)
        self.assertEqual(STEP7_MODIS_QC_MANDATORY_QA_ACCEPT, 0b0000)
        self.assertEqual(STEP7_MODIS_QC_DATA_QUALITY_SHIFT, 2)
        self.assertEqual(STEP7_MODIS_QC_DATA_QUALITY_MASK, 0b0011)
        self.assertEqual(STEP7_MODIS_QC_DATA_QUALITY_ACCEPT, 0b0000)


class TestTemporalMeanExcludesQcRejected(unittest.TestCase):
    """QC-rejected daily scenes must not contribute to the per-pixel
    mean/std -- verified against the real _qc_accept_mask() per-scene masks,
    combined the same way _build_qc_masked_modis_stack does (masked mean
    over only the accepted days)."""

    def test_rejected_days_excluded_from_mean(self):
        # 3 daily scenes for a single pixel: two clean, one cloud-flagged
        # with a wildly different (would-be-outlier) DN.
        qc_by_day = [0b00000000, 0b00000000, 0b00000010]  # day 3 = cloud
        dn_by_day = [15000, 15100, 40000]  # day 3's DN would badly skew the mean if included

        accepted = [_accept(qc, dn) for qc, dn in zip(qc_by_day, dn_by_day)]
        self.assertEqual(accepted, [True, True, False])

        celsius = [dn * 0.02 - 273.15 for dn, ok in zip(dn_by_day, accepted) if ok]
        mean_celsius = float(np.mean(celsius))
        # Only the two clean days contribute; the cloud-flagged day's DN
        # (which would pull the mean far above any physically plausible
        # summer LST) must not appear.
        self.assertAlmostEqual(mean_celsius, np.mean([15000 * 0.02 - 273.15, 15100 * 0.02 - 273.15]))
        self.assertLess(mean_celsius, 60.0)


# =============================================================================
# 5. Low-observation-count pixels remain nodata
# =============================================================================
class TestValidObservationThreshold(unittest.TestCase):
    """Mirrors the exact arithmetic _build_qc_masked_modis_stack applies
    (per-pixel QC-accepted observation count -> count.gte(threshold) mask),
    using the real STEP7_MODIS_MIN_VALID_OBSERVATIONS constant."""

    def test_threshold_constant_is_conservative_and_positive(self):
        self.assertGreaterEqual(STEP7_MODIS_MIN_VALID_OBSERVATIONS, 2)

    def test_pixel_below_threshold_masked_pixel_at_or_above_kept(self):
        # 3 pixels with QC-accepted observation counts: 1 (below), exactly
        # at threshold, and comfortably above.
        threshold = STEP7_MODIS_MIN_VALID_OBSERVATIONS
        valid_counts = np.array([1, threshold, threshold + 5], dtype="int32")
        enough_obs = valid_counts >= threshold

        mean_values = np.array([20.0, 21.0, 22.0])
        masked_mean = np.where(enough_obs, mean_values, np.nan)

        self.assertTrue(np.isnan(masked_mean[0]))  # below threshold -> nodata
        self.assertFalse(np.isnan(masked_mean[1]))  # exactly at threshold -> kept
        self.assertFalse(np.isnan(masked_mean[2]))  # above threshold -> kept

    def test_zero_observation_pixel_always_masked(self):
        threshold = STEP7_MODIS_MIN_VALID_OBSERVATIONS
        valid_counts = np.array([0], dtype="int32")
        enough_obs = valid_counts >= threshold
        self.assertFalse(bool(enough_obs[0]))


if __name__ == "__main__":
    unittest.main()
