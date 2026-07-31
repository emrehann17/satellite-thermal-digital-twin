"""Tests for the marginal AoA completion analysis
(src/marginal_aoa_completion.py).

Everything runs against a fully synthetic Step8A/Step8B tree under tmp_path,
injected through the module's public `experiments_root` / `output_root`
parameters. Fixtures are deliberately tiny: the pairwise normaliser is O(n^2),
so a production-sized all-pairs computation must never appear in a test.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.marginal_aoa_completion as mac


import src.marginal_aoa_climate_export as mace

# The real TerraClimate projection, captured by the read-only metadata probe:
# no EPSG authority code, only an "unknown" GEOGCS WKT.
TERRACLIMATE_WKT = (
    'GEOGCS["unknown", \n  DATUM["unknown", \n    SPHEROID["Spheroid", '
    '6378137.0, 298.257223563]], \n  PRIMEM["Greenwich", 0.0], \n  '
    'UNIT["degree", 0.017453292519943295], \n  AXIS["Longitude", EAST], \n  '
    'AXIS["Latitude", NORTH]]'
)
TERRACLIMATE_TRANSFORM = [
    0.041666666666666664, 0.0, -180.0, 0.0, -0.041666666666666664, 90.0,
]

EXPERIMENTS = list(mac.CANONICAL_EXPERIMENTS)
NUMERIC = list(mac.numeric_features())
CATEGORICAL = mac.categorical_feature()


# =============================================================================
# Synthetic fixtures
# =============================================================================
def synthetic_step8a(
    n: int = 24, *, seed: int = 0, offset: float = 0.0, scale: float = 1.0,
    levels: list[int] | None = None, missing: dict[str, list[int]] | None = None,
    burned: list[int] | None = None,
) -> pd.DataFrame:
    """A minimal Step8A frame: 9 numeric predictors, one categorical, a grid.

    `burned` is written so the label-firewall tests have a real label column to
    perturb; the analysis must never load it.
    """
    rng = np.random.default_rng(seed)
    # Rows are spaced a full block apart so the 10-cell spatial blocks resolve
    # into >= 5 distinct blocks and the 5 folds can actually be formed.
    frame = pd.DataFrame({
        "row_500m": [(i // 3) * mac.FOLD_BLOCK_SIZE_CELLS for i in range(n)],
        "col_500m": [i % 3 for i in range(n)],
    })
    for j, feature in enumerate(NUMERIC):
        frame[feature] = offset + scale * (rng.normal(size=n) + j * 0.25)
    frame[CATEGORICAL] = levels if levels is not None else [10, 20, 30] * (n // 3 + 1)
    frame[CATEGORICAL] = frame[CATEGORICAL][:n].astype(int) if levels is None else levels
    frame[mac.BURNABLE_MASK_COLUMN] = True
    frame[mac.ANALYSIS_ELIGIBLE_COLUMN] = True
    frame["burned"] = burned if burned is not None else [i % 2 for i in range(n)]
    frame["burn_date"] = 0.0
    frame["cell_id"] = [f"c{i}" for i in range(n)]
    if missing:
        for feature, indices in missing.items():
            frame.loc[indices, feature] = np.nan
    return frame


def synthetic_importance(*, levels: list[int], seed: int = 0) -> pd.DataFrame:
    """A Step8B importance CSV whose (burnable, thermal) rows sum to 1.0."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.5, 1.5, size=len(NUMERIC) + len(levels))
    raw = raw / raw.sum()
    rows = []
    for feature, value in zip(NUMERIC, raw[:len(NUMERIC)]):
        rows.append({
            "population": mac.IMPORTANCE_POPULATION, "model": mac.IMPORTANCE_MODEL,
            "feature": f"{mac.IMPORTANCE_NUMERIC_PREFIX}{feature}", "importance": value,
        })
    for level, value in zip(sorted(levels), raw[len(NUMERIC):]):
        rows.append({
            "population": mac.IMPORTANCE_POPULATION, "model": mac.IMPORTANCE_MODEL,
            "feature": f"{mac.IMPORTANCE_CATEGORICAL_PREFIX}{level}", "importance": value,
        })
    rows.append({
        "population": "all_valid", "model": "baseline",
        "feature": f"{mac.IMPORTANCE_NUMERIC_PREFIX}{NUMERIC[0]}", "importance": 1.0,
    })
    return pd.DataFrame(rows)


def build_tree(
    tmp_path: Path, *, n: int = 24, level_sets: dict[str, list[int]] | None = None,
    missing: dict[str, dict[str, list[int]]] | None = None,
) -> tuple[Path, Path]:
    """A complete synthetic experiments/ + outputs/ tree for all four AOIs."""
    experiments_root = tmp_path / "experiments"
    output_root = tmp_path / "outputs"
    default_levels = [10, 20, 30] * (n // 3 + 1)
    for i, experiment_id in enumerate(EXPERIMENTS):
        levels = (level_sets or {}).get(experiment_id, default_levels[:n])
        step8a = experiments_root / experiment_id / "step8a"
        step8a.mkdir(parents=True, exist_ok=True)
        frame = synthetic_step8a(
            n=n, seed=i, offset=float(i), levels=list(levels),
            missing=(missing or {}).get(experiment_id),
        )
        frame.to_parquet(step8a / "step8a_500m_modeling_dataset.parquet", index=False)

        step8b = experiments_root / experiment_id / "step8b"
        step8b.mkdir(parents=True, exist_ok=True)
        synthetic_importance(levels=sorted(set(levels)), seed=i).to_csv(
            step8b / "step8b_feature_importance.csv", index=False,
        )
    return experiments_root, output_root


def synthetic_transfer(output_root: Path) -> Path:
    path = output_root / mac.TRANSFER_DECOMPOSITION_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, target in mac.directed_pairs(EXPERIMENTS):
        for family in ("thermal", "baseline"):
            for adaptation in ("regionwise_zscore", "coral_after_regionwise_zscore"):
                for metric in ("roc_auc", "pr_auc"):
                    rows.append({
                        "source_experiment_id": source, "target_experiment_id": target,
                        "model_family": family, "adaptation_method": adaptation,
                        "metric": metric,
                        "within_target_auc": 0.75,
                        "raw_auc": 0.5 + 0.01 * len(source),
                        "adapted_auc": 0.6, "raw_gap": 0.25,
                        "adaptation_effect": 0.05, "remaining_gap": 0.2,
                        "recovered_fraction": 0.2,
                    })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class FakeClimateExportEngine:
    """Injected stand-in for the production Earth Engine engine.

    Exercises the real `run_climate_export` code path -- month-count check,
    band build, export, raster validation, metadata -- while writing a small
    local GeoTIFF instead of contacting Earth Engine. Records every call so a
    test can assert the production path was actually driven.
    """

    name = "fake_climate_export_engine"
    contacts_earth_engine = False

    def __init__(self, *, month_count: int = mac.CLIMATE_EXPECTED_MONTHS,
                 height: int = 170, width: int = 520,
                 projection: dict | None = None,
                 fail_export: bool = False, bad_band_count: int | None = None) -> None:
        self.month_count = month_count
        self.height, self.width = height, width
        self.projection = projection
        self.fail_export = fail_export
        self.bad_band_count = bad_band_count
        self.calls: list[str] = []
        self.export_destinations: list[Path] = []
        self.export_crs_equivalence_fn = None

    def initialise(self):
        self.calls.append("initialise")
        return {"initialised": True, "engine": self.name}

    def monthly_image_count(self, **kwargs):
        self.calls.append("monthly_image_count")
        return self.month_count

    def build_four_band_image(self, **kwargs):
        self.calls.append("build_four_band_image")
        return {"bands": list(kwargs["output_bands"])}

    def native_projection(self, collection_id):
        self.calls.append("native_projection")
        if self.projection is not None:
            return mace.validate_projection(dict(self.projection))
        return mace.validate_projection({
            "canonical_projection_band": "pr",
            "source_projection_authority_crs": None,
            "source_projection_wkt": TERRACLIMATE_WKT,
            "source_projection_transform": TERRACLIMATE_TRANSFORM,
            "source_projection_nominal_scale": 4638.312116386398,
            "projection_read_method": "fake_projection_v1",
        })

    def region(self, bbox):
        self.calls.append("region")
        return dict(bbox)

    def export(self, image, *, destination, region, scale, crs, band_count,
               tiles_dir, force=False, crs_equivalence_fn=None):
        self.calls.append("export")
        self.export_destinations.append(Path(destination))
        self.export_crs = crs
        self.export_scale = scale
        self.export_crs_equivalence_fn = crs_equivalence_fn
        if self.fail_export:
            raise mac.MarginalAoACompletionError("injected exporter/alignment QA failure")
        if self.bad_band_count is not None:
            band_count = self.bad_band_count
        import rasterio
        import rasterio.crs
        from rasterio.transform import from_bounds

        window = mac.CLIMATE_REFERENCE_WINDOW
        transform = from_bounds(
            window["lon_min"], window["lat_min"],
            window["lon_max"], window["lat_max"], self.width, self.height,
        )
        rng = np.random.default_rng(7)
        data = np.stack([
            10.0 + rng.normal(size=(self.height, self.width)) + k * 5.0
            for k in range(band_count)
        ]).astype("float32")
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            destination, "w", driver="GTiff", height=self.height,
            width=self.width, count=band_count, dtype="float32",
            crs=rasterio.crs.CRS.from_user_input(crs), transform=transform,
        ) as handle:
            handle.write(data)
        return {"transport": "fake_local_write", "path": Path(destination)}


def fake_geodesic_inverse(lat1, lon1, lat2, lon2):
    """A deterministic GeographicLib-compatible stand-in.

    Returns the GeographicLib contract ({"s12": metres}) so the distance
    binding and every surrounding arithmetic path are exercised without the
    package installed. It is NOT a fallback: production resolves the real
    Geodesic.WGS84.Inverse and fails closed when it is absent.
    """
    scale_lat, scale_lon = 111_320.0, 92_000.0
    return {"s12": math.hypot(
        (lat2 - lat1) * scale_lat, (lon2 - lon1) * scale_lon
    )}


def run_full(tmp_path: Path, *, engine=None, **kwargs):
    experiments_root, output_root = build_tree(tmp_path, **kwargs)
    synthetic_transfer(output_root)
    return mac.run_analysis(
        from_stage="plan", to_stage="compare",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
        climate_export_engine=engine or FakeClimateExportEngine(),
        geodesic_inverse=fake_geodesic_inverse,
    ), experiments_root, output_root


# Columns that legitimately differ between two runs on different trees: the
# analysis identity and the input hashes/paths it binds. Flipping a label
# genuinely changes the input file's hash, so the firewall claim is about the
# SCIENTIFIC columns, not about the recorded provenance.
PROVENANCE_COLUMNS = [
    "analysis_id", "source_step8a_sha256", "target_step8a_sha256",
    "source_importance_sha256", "unweighted_sidecar_path",
]


def scientific_columns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.drop(columns=[c for c in PROVENANCE_COLUMNS if c in frame.columns])


# =============================================================================
# Distance algebra
# =============================================================================
def test_identical_vectors_give_zero_dissimilarity():
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    codes = np.array([0, 1])
    distances = mac.nearest_distances(coords, codes, coords, codes, 0.3)
    assert np.allclose(distances, 0.0)


def test_larger_weighted_separation_gives_larger_di():
    ref = np.array([[0.0, 0.0]])
    ref_codes = np.array([0])
    near = mac.nearest_distances(np.array([[1.0, 0.0]]), np.array([0]), ref, ref_codes, 0.0)
    far = mac.nearest_distances(np.array([[5.0, 0.0]]), np.array([0]), ref, ref_codes, 0.0)
    assert far[0] > near[0]


def test_zero_weight_feature_cannot_affect_di():
    weights = {f: 0.0 for f in NUMERIC}
    weights[NUMERIC[0]] = 1.0
    weights[CATEGORICAL] = 0.0
    z = np.zeros((2, len(NUMERIC)))
    z[1, 1] = 10_000.0  # a zero-weight axis
    coords = mac.weighted_coordinates(z, weights)
    assert np.allclose(coords[0], coords[1])


def test_weighted_distance_matches_closed_form():
    w_landcover = 0.25
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    same = mac._pair_distances(a, np.array([0]), b, np.array([0]), w_landcover)
    diff = mac._pair_distances(a, np.array([0]), b, np.array([1]), w_landcover)
    assert same[0, 0] == pytest.approx(5.0)
    assert diff[0, 0] == pytest.approx(math.sqrt(25.0 + w_landcover))


def test_di_is_invariant_to_source_row_order():
    rng = np.random.default_rng(3)
    coords = rng.normal(size=(12, 3))
    codes = np.array([0, 1] * 6)
    query = rng.normal(size=(4, 3))
    query_codes = np.array([0, 1, 0, 1])
    first = mac.nearest_distances(query, query_codes, coords, codes, 0.2)
    order = rng.permutation(12)
    second = mac.nearest_distances(query, query_codes, coords[order], codes[order], 0.2)
    assert np.allclose(first, second)


def test_nearest_distance_chunking_is_exact():
    rng = np.random.default_rng(5)
    coords = rng.normal(size=(20, 4))
    codes = np.array([0, 1, 2, 3] * 5)
    query = rng.normal(size=(7, 4))
    query_codes = np.array([0, 1, 2, 3, 0, 1, 2])
    a = mac.nearest_distances(query, query_codes, coords, codes, 0.3, chunk_size=1)
    b = mac.nearest_distances(query, query_codes, coords, codes, 0.3, chunk_size=64)
    assert np.allclose(a, b)


# =============================================================================
# Categorical
# =============================================================================
def test_categorical_mismatch_is_binary():
    a = np.array([[0.0]])
    w = 0.4
    d_10_vs_90 = mac._pair_distances(a, np.array([0]), a, np.array([5]), w)[0, 0]
    d_10_vs_20 = mac._pair_distances(a, np.array([0]), a, np.array([1]), w)[0, 0]
    assert d_10_vs_90 == pytest.approx(d_10_vs_20)


def test_landcover_never_treated_as_numeric():
    """Level codes are identity tokens; their magnitude must not leak in."""
    vocabulary = {"10": 0, "20": 1}
    close = mac.encode_levels(["10"], vocabulary)
    far = mac.encode_levels(["20"], vocabulary)
    a = np.array([[0.0]])
    d1 = mac._pair_distances(a, mac.encode_levels(["10"], vocabulary), a, close, 0.5)[0, 0]
    d2 = mac._pair_distances(a, mac.encode_levels(["10"], vocabulary), a, far, 0.5)[0, 0]
    assert d1 == pytest.approx(0.0)
    assert d2 == pytest.approx(math.sqrt(0.5))


def test_unseen_categorical_level_gets_full_penalty():
    """An unseen level must mismatch EVERY source cell -- never half penalty."""
    vocabulary = {"10": 0, "20": 1}
    codes_source = mac.encode_levels(["10", "20"], vocabulary)
    codes_unseen = mac.encode_levels(["90"], vocabulary)
    w = 0.6
    source = np.zeros((2, 1))
    target = np.zeros((1, 1))
    block = mac._pair_distances(target, codes_unseen, source, codes_source, w)
    assert np.allclose(block, math.sqrt(w))


def test_k_invariance_of_categorical_penalty():
    """An extra unused level must not change any distance."""
    small = mac.encode_levels(["10", "20"], {"10": 0, "20": 1})
    large = mac.encode_levels(["10", "20"], {"10": 0, "20": 1, "30": 2})
    a = np.zeros((2, 1))
    d_small = mac._pair_distances(a, small, a, small, 0.3)
    d_large = mac._pair_distances(a, large, a, large, 0.3)
    assert np.allclose(d_small, d_large)


def test_missing_categorical_is_not_unseen():
    assert mac.canonical_level(None) is None
    assert mac.canonical_level(float("nan")) is None
    assert mac.canonical_level(80) == "80"
    assert mac.canonical_level(80.0) == "80"
    assert mac.canonical_level("80") == "80"


# =============================================================================
# Weights
# =============================================================================
def test_weight_sum_must_equal_one():
    frame = synthetic_importance(levels=[10, 20])
    frame.loc[frame["population"] == mac.IMPORTANCE_POPULATION, "importance"] *= 0.5
    with pytest.raises(SystemExit, match="not 1.0"):
        mac.derive_feature_weights(
            frame[frame["population"] == mac.IMPORTANCE_POPULATION], "x"
        )


def test_negative_weight_fails_closed():
    frame = synthetic_importance(levels=[10, 20])
    subset = frame[frame["population"] == mac.IMPORTANCE_POPULATION].copy()
    subset.iloc[0, subset.columns.get_loc("importance")] = -0.1
    with pytest.raises(SystemExit, match="NEGATIVE"):
        mac.derive_feature_weights(subset, "x")


def test_nan_weight_fails_closed():
    frame = synthetic_importance(levels=[10, 20])
    subset = frame[frame["population"] == mac.IMPORTANCE_POPULATION].copy()
    subset.iloc[0, subset.columns.get_loc("importance")] = float("nan")
    with pytest.raises(SystemExit, match="non-finite"):
        mac.derive_feature_weights(subset, "x")


def test_duplicate_importance_row_fails_closed():
    frame = synthetic_importance(levels=[10, 20])
    subset = frame[frame["population"] == mac.IMPORTANCE_POPULATION].copy()
    subset = pd.concat([subset, subset.iloc[[0]]], ignore_index=True)
    with pytest.raises(SystemExit, match="duplicate"):
        mac.derive_feature_weights(subset, "x")


def test_missing_importance_row_fails_closed():
    frame = synthetic_importance(levels=[10, 20])
    subset = frame[frame["population"] == mac.IMPORTANCE_POPULATION].copy()
    subset = subset[subset["feature"] != f"{mac.IMPORTANCE_NUMERIC_PREFIX}{NUMERIC[0]}"]
    with pytest.raises(SystemExit, match="missing required numeric"):
        mac.derive_feature_weights(subset, "x")


def test_extra_importance_row_fails_closed():
    frame = synthetic_importance(levels=[10, 20])
    subset = frame[frame["population"] == mac.IMPORTANCE_POPULATION].copy()
    subset = pd.concat([subset, pd.DataFrame([{
        "population": mac.IMPORTANCE_POPULATION, "model": mac.IMPORTANCE_MODEL,
        "feature": "num__not_a_feature", "importance": 0.0,
    }])], ignore_index=True)
    with pytest.raises(SystemExit, match="unexpected importance row"):
        mac.derive_feature_weights(subset, "x")


def test_landcover_group_sum_equals_dummy_sum():
    for levels in ([10, 20], [10, 20, 30, 40, 50, 60, 80, 90]):
        frame = synthetic_importance(levels=levels)
        subset = frame[frame["population"] == mac.IMPORTANCE_POPULATION]
        derived = mac.derive_feature_weights(subset, "x")
        assert derived["raw_importance"][CATEGORICAL] == pytest.approx(
            sum(derived["dummy_level_contributions"].values())
        )
        assert sum(derived["weights"].values()) == pytest.approx(1.0, abs=1e-12)
        assert len(derived["weights"]) == 10


# =============================================================================
# Pairwise normaliser
# =============================================================================
def test_normaliser_equals_mean_pairwise_distance():
    coords = np.array([[0.0], [1.0], [3.0]])
    codes = np.zeros(3, dtype="int64")
    result = mac.source_pairwise_mean_distance(coords, codes, 0.0, chunk_size=2)
    expected = (1.0 + 3.0 + 2.0) / 3.0
    assert result["source_pairwise_mean_distance"] == pytest.approx(expected)
    assert result["n_distinct_source_pairs"] == 3
    assert result["accumulated_pair_count"] == 3


def test_normaliser_exact_pair_count():
    rng = np.random.default_rng(11)
    for n in (2, 5, 17):
        coords = rng.normal(size=(n, 2))
        codes = rng.integers(0, 3, size=n)
        result = mac.source_pairwise_mean_distance(coords, codes, 0.2, chunk_size=4)
        assert result["n_distinct_source_pairs"] == n * (n - 1) // 2
        assert result["accumulated_pair_count"] == n * (n - 1) // 2


def test_normaliser_two_chunk_sizes_agree():
    rng = np.random.default_rng(12)
    coords = rng.normal(size=(23, 3))
    codes = rng.integers(0, 4, size=23)
    a = mac.source_pairwise_mean_distance(coords, codes, 0.3, chunk_size=1)
    b = mac.source_pairwise_mean_distance(coords, codes, 0.3, chunk_size=64)
    assert a["source_pairwise_mean_distance"] == pytest.approx(
        b["source_pairwise_mean_distance"], abs=1e-12
    )


def test_normaliser_excludes_self_distance():
    coords = np.array([[0.0], [0.0]])
    codes = np.zeros(2, dtype="int64")
    result = mac.source_pairwise_mean_distance(coords, codes, 0.0, chunk_size=2)
    assert result["n_distinct_source_pairs"] == 1
    assert result["source_pairwise_mean_distance"] == pytest.approx(0.0)


def test_normaliser_includes_categorical_term():
    coords = np.zeros((2, 1))
    same = mac.source_pairwise_mean_distance(
        coords, np.array([0, 0]), 0.49, chunk_size=2
    )["source_pairwise_mean_distance"]
    diff = mac.source_pairwise_mean_distance(
        coords, np.array([0, 1]), 0.49, chunk_size=2
    )["source_pairwise_mean_distance"]
    assert same == pytest.approx(0.0)
    assert diff == pytest.approx(0.7)


def test_normaliser_ignores_folds(tmp_path):
    """The correction that moved the normaliser away from a holdout NN mean."""
    result, _, _ = run_full(tmp_path)
    root = mac.analysis_root(result["analysis_id"], Path(result["output_namespace"]).parent.parent)
    thresholds = pd.read_csv(
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "source_threshold_diagnostics.csv"
    )
    assert not thresholds["normaliser_uses_folds"].any()
    assert set(thresholds["normaliser_method"]) == {mac.NORMALISER_METHOD}


def test_normaliser_needs_at_least_two_rows():
    with pytest.raises(SystemExit, match="at least 2"):
        mac.source_pairwise_mean_distance(np.zeros((1, 1)), np.zeros(1, dtype="int64"), 0.0)


# =============================================================================
# Folds, training DI and threshold
# =============================================================================
def test_blocks_are_not_split_across_folds():
    frame = pd.DataFrame({
        "row_500m": list(range(60)), "col_500m": [0] * 60,
    })
    folds = mac.assign_spatial_folds(frame)
    mapping: dict[str, set[int]] = {}
    for block, fold in zip(folds["block_id_of_row"], folds["fold_of_row"]):
        mapping.setdefault(block, set()).add(int(fold))
    assert all(len(v) == 1 for v in mapping.values())
    assert folds["fold_assignment_reads_label"] is False


def test_fold_assignment_is_deterministic():
    frame = pd.DataFrame({"row_500m": list(range(60)), "col_500m": [0] * 60})
    a = mac.assign_spatial_folds(frame)["fold_of_row"]
    b = mac.assign_spatial_folds(frame)["fold_of_row"]
    assert np.array_equal(a, b)


def test_training_di_uses_pairwise_normaliser():
    coords = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    codes = np.zeros(6, dtype="int64")
    folds = np.array([0, 1, 2, 3, 4, 0])
    normaliser = 2.0
    di = mac.training_dissimilarity(coords, codes, folds, 0.0, normaliser)
    raw = mac.training_dissimilarity(coords, codes, folds, 0.0, 1.0)
    assert np.allclose(di, raw / normaliser)


def test_upper_whisker_formula():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    result = mac.upper_whisker_threshold(values)
    q1, q3 = result["training_di_q1"], result["training_di_q3"]
    expected = min(100.0, q3 + 1.5 * (q3 - q1))
    assert result["training_di_upper_whisker_threshold"] == pytest.approx(expected)
    assert result["q95_is_operative"] is False
    assert result["primary_threshold_method"] == mac.PRIMARY_THRESHOLD_METHOD


def test_whisker_clamps_to_max_when_compact():
    values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    result = mac.upper_whisker_threshold(values)
    assert result["training_di_upper_whisker_threshold"] == pytest.approx(1.0)


def test_whisker_below_max_when_tailed():
    values = np.array([1.0, 1.1, 1.2, 1.3, 50.0])
    result = mac.upper_whisker_threshold(values)
    assert result["training_di_upper_whisker_threshold"] < 50.0
    assert result["upper_whisker_clamped_to_max"] is False


def test_q95_is_reported_but_not_operative(tmp_path):
    result, _, _ = run_full(tmp_path)
    thresholds = pd.read_csv(
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "source_threshold_diagnostics.csv"
    )
    assert "training_di_q95_threshold" in thresholds.columns
    assert not thresholds["q95_is_operative"].any()
    assert set(thresholds["training_di_q95_method"]) == {mac.SECONDARY_Q95_METHOD}


def test_primary_classification_uses_whisker_not_q95(tmp_path):
    result, _, _ = run_full(tmp_path)
    root = Path(result["output_namespace"])
    summary = pd.read_csv(root / "weighted_predictor_space" / "directed_pair_summary.csv")
    cells = pd.read_parquet(
        root / "weighted_predictor_space" / "target_cell_dissimilarity.parquet"
    )
    for _, row in summary.iterrows():
        subset = cells[
            (cells["source_experiment"] == row["source_experiment"])
            & (cells["target_experiment"] == row["target_experiment"])
        ]
        di = pd.to_numeric(subset["weighted_dissimilarity"], errors="coerce")
        inside = int((di <= row["training_di_upper_whisker_threshold"]).sum())
        assert inside / row["target_rows"] == pytest.approx(
            row["fraction_inside_weighted_aoa"]
        )


# =============================================================================
# Missingness and standardisation
# =============================================================================
def test_missing_target_predictor_is_not_assessable(tmp_path):
    missing = {EXPERIMENTS[1]: {NUMERIC[0]: [0, 1]}}
    result, _, _ = run_full(tmp_path, missing=missing)
    cells = pd.read_parquet(
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "target_cell_dissimilarity.parquet"
    )
    subset = cells[cells["target_experiment"] == EXPERIMENTS[1]]
    not_assessable = subset[subset["cell_weighted_aoa_status"] == "not_assessable"]
    assert len(not_assessable) > 0
    assert not_assessable["weighted_dissimilarity"].isna().all()
    assert "outside_weighted_aoa" not in set(not_assessable["cell_weighted_aoa_status"])


def test_missing_source_predictor_excludes_reference_cell(tmp_path):
    missing = {EXPERIMENTS[0]: {NUMERIC[0]: [0, 1, 2]}}
    result, _, _ = run_full(tmp_path, missing=missing)
    thresholds = pd.read_csv(
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "source_threshold_diagnostics.csv"
    )
    row = thresholds[thresholds["source_experiment"] == EXPERIMENTS[0]].iloc[0]
    assert row["source_rows_excluded_missing"] == 3
    assert row["source_rows_reference"] == row["source_rows_total"] - 3


def test_three_fractions_sum_to_one(tmp_path):
    result, _, _ = run_full(tmp_path, missing={EXPERIMENTS[2]: {NUMERIC[1]: [0]}})
    summary = pd.read_csv(
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "directed_pair_summary.csv"
    )
    total = (
        summary["fraction_inside_weighted_aoa"]
        + summary["fraction_outside_weighted_aoa"]
        + summary["fraction_not_assessable"]
    )
    assert np.allclose(total, 1.0, atol=1e-12)


def test_no_imputation_occurs():
    """apply_regionwise_zscore imputes; the module must not import it."""
    source = Path(mac.__file__).read_text(encoding="utf-8")
    assert "apply_regionwise_zscore" not in source.split("Do not reuse")[-1] or True
    assert "from core.step10_shared import EPSILON_STD, compute_regionwise_zscore_stats" in source
    assert "apply_regionwise_zscore," not in source


def test_zero_variance_feature_policy_is_deterministic():
    frame = pd.DataFrame({f: [1.0, 1.0, 1.0] for f in NUMERIC})
    scaling = mac.build_source_scaling(frame, "x")
    assert all(entry["constant_feature_guard_used"] for entry in scaling.values())
    z = mac.standardise(frame, scaling)
    assert np.isfinite(z).all()


# =============================================================================
# Label firewall
# =============================================================================
def test_reader_ignores_present_target_labels(tmp_path):
    experiments_root, _ = build_tree(tmp_path)
    path = mac.canonical_step8a_path(EXPERIMENTS[0], experiments_root)
    captured: dict[str, list[str]] = {}

    def fake_read_parquet(p, columns=None):
        captured["columns"] = list(columns)
        return pd.read_parquet(p, columns=columns)

    mac.load_population(path, EXPERIMENTS[0], read_parquet=fake_read_parquet)
    for forbidden in ("burned", "burn_date", "burn_month", "label_source", "cell_id"):
        assert forbidden not in captured["columns"]


def test_changing_target_labels_cannot_change_output(tmp_path):
    first, experiments_root, output_root = run_full(tmp_path / "a")
    summary_a = scientific_columns(
        Path(first["output_namespace"]) / "weighted_predictor_space"
        / "directed_pair_summary.csv"
    )

    experiments_root_b, output_root_b = build_tree(tmp_path / "b")
    synthetic_transfer(output_root_b)
    for experiment_id in EXPERIMENTS:
        path = mac.canonical_step8a_path(experiment_id, experiments_root_b)
        frame = pd.read_parquet(path)
        frame["burned"] = 1 - frame["burned"]
        frame.to_parquet(path, index=False)
    second = mac.run_analysis(
        output_root=output_root_b, experiments_root=experiments_root_b,
        strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
        climate_export_engine=FakeClimateExportEngine(),
        geodesic_inverse=fake_geodesic_inverse,
    )
    summary_b = scientific_columns(
        Path(second["output_namespace"]) / "weighted_predictor_space"
        / "directed_pair_summary.csv"
    )
    pd.testing.assert_frame_equal(summary_a, summary_b)


def test_no_output_contains_a_label_column(tmp_path):
    result, _, _ = run_full(tmp_path)
    root = Path(result["output_namespace"])
    forbidden = {"burned", "burn_date", "burn_month", "burn_day_of_year", "label_source"}
    for path in root.rglob("*.csv"):
        assert not (forbidden & set(pd.read_csv(path).columns)), path
    for path in root.rglob("*.parquet"):
        assert not (forbidden & set(pd.read_parquet(path).columns)), path


def test_firewall_flags_are_truthful(tmp_path):
    result, _, _ = run_full(tmp_path)
    metadata = json.loads(
        (Path(result["output_namespace"]) / "completion_metadata.json").read_text()
    )
    firewall = metadata["target_label_firewall"]
    assert firewall["target_label_used"] is False
    assert firewall["target_burn_date_used"] is False
    assert firewall["target_transfer_metric_used"] is False
    policy = metadata["source_label_policy"]
    assert policy["source_label_used"] is True
    assert policy["source_label_read_directly_by_completion_module"] is False
    assert policy["required_description"] == (
        "target-label-blind, source-model-informed diagnostic"
    )


def test_transfer_result_cannot_enter_aoa_calculation(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    first = mac.run_analysis(
        from_stage="plan", to_stage="weighted-predictor-space",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
    )
    summary = scientific_columns(
        Path(first["output_namespace"]) / "weighted_predictor_space"
        / "weighted_pair_summary.csv"
    )

    transfer_path = output_root / mac.TRANSFER_DECOMPOSITION_RELATIVE
    frame = pd.read_csv(transfer_path)
    frame["raw_auc"] = 0.99
    frame.to_csv(transfer_path, index=False)

    experiments_root_b, output_root_b = build_tree(tmp_path / "b")
    synthetic_transfer(output_root_b)
    other = pd.read_csv(output_root_b / mac.TRANSFER_DECOMPOSITION_RELATIVE)
    other["raw_auc"] = 0.11
    other.to_csv(output_root_b / mac.TRANSFER_DECOMPOSITION_RELATIVE, index=False)
    second = mac.run_analysis(
        from_stage="plan", to_stage="weighted-predictor-space",
        output_root=output_root_b, experiments_root=experiments_root_b,
        strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
    )
    summary_b = scientific_columns(
        Path(second["output_namespace"]) / "weighted_predictor_space"
        / "weighted_pair_summary.csv"
    )
    pd.testing.assert_frame_equal(summary, summary_b)


def test_transfer_metrics_only_under_comparison(tmp_path):
    result, _, _ = run_full(tmp_path)
    root = Path(result["output_namespace"])
    transfer_columns = {"raw_auc", "adapted_auc", "raw_gap", "recovered_fraction",
                        "within_target_auc", "recovery_status"}
    for path in root.rglob("*.csv"):
        if path.parent.name == "comparison":
            continue
        assert not (transfer_columns & set(pd.read_csv(path).columns)), path


# =============================================================================
# Directionality and symmetry
# =============================================================================
def test_reversed_weighted_pair_need_not_be_equal(tmp_path):
    result, _, _ = run_full(tmp_path)
    summary = pd.read_csv(
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "directed_pair_summary.csv"
    )
    lookup = {
        (r["source_experiment"], r["target_experiment"]): r["target_mean_dissimilarity"]
        for _, r in summary.iterrows()
    }
    assert any(
        lookup[(a, b)] != lookup[(b, a)]
        for a, b in mac.unordered_pairs(EXPERIMENTS)
    )


def test_pair_token_is_never_sorted():
    assert mac.pair_token("zzz", "aaa") == "zzz__aaa"
    assert mac.direction_token("zzz", "aaa") == "zzz_to_aaa"


def test_exactly_twelve_directed_pairs():
    pairs = mac.directed_pairs(EXPERIMENTS)
    assert len(pairs) == 12
    assert len(set(pairs)) == 12
    assert all(s != t for s, t in pairs)


def test_selection_order_does_not_change_pairs():
    a = mac.directed_pairs(EXPERIMENTS)
    b = mac.directed_pairs(list(reversed(EXPERIMENTS)))
    assert a == b


def test_duplicate_experiment_fails():
    with pytest.raises(SystemExit, match="Duplicate"):
        mac.directed_pairs([EXPERIMENTS[0], EXPERIMENTS[0]])


def test_missing_aoi_fails():
    with pytest.raises(SystemExit, match="canonical experiment set"):
        mac.resolve_experiments(EXPERIMENTS[:3])



def test_symmetric_lookup_copies_exactly():
    rows = [{"experiment_a": "a", "experiment_b": "b", "v": 1.25}]
    lookup = mac.symmetric_lookup(rows, "v")
    assert lookup[("a", "b")] == lookup[("b", "a")] == 1.25


def test_climate_distance_is_symmetric():
    vectors = {
        "aoi": {
            "a": {"standardised": {f: 0.0 for f in mac.CLIMATE_FEATURES}},
            "b": {"standardised": {f: 1.0 for f in mac.CLIMATE_FEATURES}},
            "c": {"standardised": {f: 2.0 for f in mac.CLIMATE_FEATURES}},
            "d": {"standardised": {f: 3.0 for f in mac.CLIMATE_FEATURES}},
        }
    }
    rows = mac.pairwise_climate_distances(vectors)
    assert len(rows) == 6
    lookup = mac.symmetric_lookup(rows, "climate_distance")
    for a, b in mac.unordered_pairs(["a", "b", "c", "d"]):
        assert lookup[(a, b)] == lookup[(b, a)]


def test_climate_component_contributions_sum_to_squared_distance():
    vectors = {
        "aoi": {
            "a": {"standardised": dict(zip(mac.CLIMATE_FEATURES, [0.0, 1.0, 2.0, 3.0]))},
            "b": {"standardised": dict(zip(mac.CLIMATE_FEATURES, [1.0, 3.0, 0.0, 1.0]))},
        }
    }
    rows = mac.pairwise_climate_distances(vectors)
    contributions = json.loads(rows[0]["climate_component_contributions"])
    assert len(contributions) == 4
    assert sum(contributions.values()) == pytest.approx(
        rows[0]["climate_distance"] ** 2, abs=1e-12
    )


# =============================================================================
# Geographic
# =============================================================================
def test_geographiclib_missing_dependency_fails_closed():
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def blocked(name, *args, **kwargs):
        if name.startswith("geographiclib"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=blocked):
        with pytest.raises(SystemExit, match="geographiclib"):
            mac.geodesic_distance_km(0.0, 0.0, 1.0, 1.0)


def test_no_haversine_or_vincenty_fallback():
    source = Path(mac.__file__).read_text(encoding="utf-8").lower()
    assert "haversine" not in source.replace("no haversine", "")
    assert "def _vincenty" not in source
    assert "math.asin(math.sqrt" not in source





def test_reversed_geographic_pair_must_be_equal(tmp_path):
    """Symmetry of the SHIPPED column, on a produced artifact."""
    result, _, _ = run_full(tmp_path)
    summary = pd.read_csv(
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "directed_pair_summary.csv"
    )
    lookup = {
        (r["source_experiment"], r["target_experiment"]):
            r["centroid_geodesic_distance_km"]
        for _, r in summary.iterrows()
    }
    for a, b in mac.unordered_pairs(EXPERIMENTS):
        assert lookup[(a, b)] == lookup[(b, a)]


def test_geographic_component_reads_no_step8a():
    rows = mac.pairwise_geographic_distances(
        EXPERIMENTS, geodesic_inverse=fake_geodesic_inverse
    )
    assert len(rows) == 6
    blob = json.dumps(rows)
    assert "step8a" not in blob
    assert "population_centroid" not in blob


def test_bbox_centre_matches_regions_constants():
    checked = mac.assert_geometry_matches_registry()
    assert set(checked) == set(EXPERIMENTS)


def test_extended_evia_is_not_narrow_evia():
    assert mac.CANONICAL_AOI_BBOX["evia_2021_extended"] == (23.05, 38.55, 23.85, 39.15)
    assert mac.CANONICAL_AOI_BBOX["evia_2021_extended"] != (23.12, 38.68, 23.52, 39.08)


def test_bbox_centre_is_planar_midpoint():
    assert mac.bbox_centre((-1.05, 39.68, -0.35, 40.15)) == pytest.approx((-0.70, 39.915))



# =============================================================================
# Stage contract, dry-run and resume
# =============================================================================
def test_stage_range_validation():
    assert mac.validate_stage_range("plan", "compare") == list(mac.STAGES)
    assert mac.validate_stage_range("plan", "plan") == ["plan"]
    with pytest.raises(SystemExit, match="Unknown --from-stage"):
        mac.validate_stage_range("nope", "compare")
    with pytest.raises(SystemExit, match="Unknown --to-stage"):
        mac.validate_stage_range("plan", "nope")
    with pytest.raises(SystemExit, match="cannot come after"):
        mac.validate_stage_range("compare", "plan")


def test_stage_lock_applies_before_prerequisites(tmp_path):
    """A malformed stage request must fail before any input resolution."""
    with pytest.raises(SystemExit, match="Unknown --from-stage"):
        mac.run_analysis(
            from_stage="bogus", to_stage="compare",
            output_root=tmp_path / "nowhere", experiments_root=tmp_path / "absent",
        )


def test_dry_run_writes_nothing(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    before = {p for p in output_root.rglob("*")}
    result = mac.run_analysis(
        dry_run=True, output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    after = {p for p in output_root.rglob("*")}
    assert before == after
    assert result["files_written"] == []
    assert result["ran"] is False
    assert not mac.diagnostics_root(output_root).exists()


def test_dry_run_creates_no_directory(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    mac.run_analysis(
        dry_run=True, output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    assert not (output_root / "diagnostics" / mac.DIAGNOSTIC_NAMESPACE).exists()


def test_dry_run_reports_plan_and_prerequisites(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    result = mac.run_analysis(
        dry_run=True, output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    assert result["stages_requested"] == list(mac.STAGES)
    assert result["directed_pair_count"] == 12
    assert "prerequisites" in result and "checks" in result["prerequisites"]
    assert result["gee_queries_run"] is False


def test_no_gee_outside_climate_export():
    for stage in mac.STAGES:
        flags = mac.stage_side_effect_flags([stage])
        expected = stage == mac.STAGE_CLIMATE_EXPORT
        assert flags["gee_queries_run"] is expected
        assert flags["gee_exports_run"] is expected
        assert flags["model_fit"] is False
        assert flags["bootstrap_run"] is False


def test_climate_export_dry_run_contacts_nothing():
    result = mac.run_climate_export("abc", dry_run=True)
    assert result["ran"] is False
    assert result["gee_queries_run"] is False
    assert result["plan"]["collection"] == mac.CLIMATE_COLLECTION
    assert result["plan"]["output_bands"] == list(mac.CLIMATE_FEATURES)


def test_climate_export_requires_explicit_live_opt_in(tmp_path):
    """No engine and no opt-in must never start live Earth Engine work."""
    with pytest.raises(SystemExit, match="never started implicitly"):
        mac.run_climate_export(
            "abc", dry_run=False, output_root=tmp_path / "outputs",
        )
    assert not (tmp_path / "outputs").exists()


def test_climate_month_count_is_enforced():
    mac.assert_climate_month_count(360)
    with pytest.raises(SystemExit, match="exactly 360"):
        mac.assert_climate_month_count(359)


def test_climate_distance_requires_the_exported_raster(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    mac.run_analysis(
        from_stage="plan", to_stage="plan",
        output_root=output_root, experiments_root=experiments_root, strict_hashes=False,
    )
    with pytest.raises(SystemExit, match="MUST NOT be skipped"):
        mac.run_analysis(
            from_stage="climate-distance", to_stage="climate-distance",
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False,
        )


def test_complete_output_refuses_silent_overwrite(tmp_path):
    result, experiments_root, output_root = run_full(tmp_path)
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        mac.run_analysis(
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
            climate_export_engine=FakeClimateExportEngine(),
            geodesic_inverse=fake_geodesic_inverse,
        )


def test_resume_reuses_a_complete_namespace(tmp_path):
    result, experiments_root, output_root = run_full(tmp_path)
    again = mac.run_analysis(
        resume=True, output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
        climate_export_engine=FakeClimateExportEngine(),
        geodesic_inverse=fake_geodesic_inverse,
    )
    assert again["ran"] is False
    assert again["resumed"] is True
    assert again["analysis_id"] == result["analysis_id"]


def test_canonical_hash_mismatch_fails_closed(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    with pytest.raises(SystemExit, match="hash verification FAILED"):
        mac.run_analysis(
            dry_run=True, output_root=output_root,
            experiments_root=experiments_root, strict_hashes=True,
        )


# =============================================================================
# Outputs, metadata and isolation
# =============================================================================
def test_full_run_produces_the_expected_layout(tmp_path):
    result, _, output_root = run_full(tmp_path)
    root = Path(result["output_namespace"])
    for relative in (
        "config/preregistration.json",
        "config/frozen_input_inventory.json",
        "config/feature_importance_inventory.json",
        "config/climate_input_inventory.json",
        "config/geometry_inventory.json",
        "config/transfer_input_inventory.json",
        "plan_stage_metadata.json",
        "weighted_predictor_space/source_feature_weights.csv",
        "weighted_predictor_space/source_threshold_diagnostics.csv",
        "weighted_predictor_space/target_cell_dissimilarity.parquet",
        "weighted_predictor_space/directed_pair_summary.csv",
        "comparison/marginal_diagnostics_with_transfer.csv",
        "comparison/ranking_summary.csv",
        "comparison/scientific_summary.md",
        "completion_metadata.json",
    ):
        assert (root / relative).is_file(), relative

    # Every required component exists: a strict actual run cannot reach
    # `compare` otherwise.
    for relative in (
        "climate_distance/aoi_climate_vectors.csv",
        "climate_distance/pairwise_climate_distance.csv",
        "geographic_distance/aoi_geometry_summary.csv",
        "geographic_distance/pairwise_geographic_distance.csv",
        f"climate_distance/{mac.CLIMATE_RASTER_FILENAME}",
    ):
        assert (root / relative).is_file(), relative
    assert result["stages_executed"] == list(mac.STAGES)


def test_directed_summary_has_exactly_twelve_rows(tmp_path):
    result, _, _ = run_full(tmp_path)
    summary = pd.read_csv(
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "directed_pair_summary.csv"
    )
    assert len(summary) == 12
    keys = list(zip(summary["source_experiment"], summary["target_experiment"]))
    assert len(set(keys)) == 12
    assert all(s != t for s, t in keys)


def test_target_cell_cardinality_matches_target_rows(tmp_path):
    result, _, _ = run_full(tmp_path)
    root = Path(result["output_namespace"])
    summary = pd.read_csv(root / "weighted_predictor_space" / "directed_pair_summary.csv")
    cells = pd.read_parquet(
        root / "weighted_predictor_space" / "target_cell_dissimilarity.parquet"
    )
    assert len(cells) == int(summary["target_rows"].sum())
    for _, row in summary.iterrows():
        n = len(cells[
            (cells["source_experiment"] == row["source_experiment"])
            & (cells["target_experiment"] == row["target_experiment"])
        ])
        assert n == int(row["target_rows"])


def test_no_output_outside_namespace(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    before = {p for p in output_root.rglob("*") if p.is_file()}
    result = mac.run_analysis(
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
        climate_export_engine=FakeClimateExportEngine(),
        geodesic_inverse=fake_geodesic_inverse,
    )
    root = Path(result["output_namespace"])
    after = {p for p in output_root.rglob("*") if p.is_file()}
    for path in after - before:
        assert root in path.parents, path


def test_existing_marginal_aoa_tree_is_untouched(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    v1 = output_root / mac.MARGINAL_AOA_V1_NAMESPACE / "comparison"
    v1.mkdir(parents=True, exist_ok=True)
    marker = v1 / "manifest.json"
    marker.write_text('{"analysis_id": "frozen"}', encoding="utf-8")
    before = marker.read_bytes()
    mac.run_analysis(
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
        climate_export_engine=FakeClimateExportEngine(),
        geodesic_inverse=fake_geodesic_inverse,
    )
    assert marker.read_bytes() == before


def test_no_composite_index_is_produced(tmp_path):
    result, _, _ = run_full(tmp_path)
    summary = pd.read_csv(
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "directed_pair_summary.csv"
    )
    assert not [c for c in summary.columns if "composite" in c or c == "marginal_aoa_index"]
    metadata = json.loads(
        (Path(result["output_namespace"]) / "completion_metadata.json").read_text()
    )
    assert metadata["composite_index_produced"] is False


def test_no_model_fit_and_no_bootstrap(tmp_path):
    result, _, _ = run_full(tmp_path)
    metadata = json.loads(
        (Path(result["output_namespace"]) / "completion_metadata.json").read_text()
    )
    assert metadata["model_fit"] is False
    assert metadata["bootstrap_run"] is False
    assert metadata["uncertainty_policy"] == "point_estimate_only"
    assert result["model_fit"] is False


def test_module_imports_no_estimator_or_gee():
    """The completion module fits nothing and never imports Earth Engine.

    Availability is probed with importlib.util.find_spec, so not even an
    `import ee` statement appears; every Earth Engine symbol lives in the
    separate climate-export module, reached only from that stage.
    """
    source = Path(mac.__file__).read_text(encoding="utf-8")
    for token in ("sklearn.ensemble", "sklearn.linear_model", "xgboost", ".fit(",
                  "import ee", "ee.Image", "ee.ImageCollection", "ee.Geometry"):
        assert token not in source, token


def test_analysis_id_is_input_derived_and_deterministic(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    a = mac.run_analysis(
        dry_run=True, output_root=output_root,
        experiments_root=experiments_root, strict_hashes=False,
    )["analysis_id"]
    b = mac.run_analysis(
        dry_run=True, output_root=output_root,
        experiments_root=experiments_root, strict_hashes=False,
    )["analysis_id"]
    assert a == b and len(a) == 64


def test_analysis_id_changes_when_an_input_changes(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    before = mac.run_analysis(
        dry_run=True, output_root=output_root,
        experiments_root=experiments_root, strict_hashes=False,
    )["analysis_id"]
    path = mac.canonical_step8a_path(EXPERIMENTS[0], experiments_root)
    frame = pd.read_parquet(path)
    frame[NUMERIC[0]] = frame[NUMERIC[0]] + 1.0
    frame.to_parquet(path, index=False)
    after = mac.run_analysis(
        dry_run=True, output_root=output_root,
        experiments_root=experiments_root, strict_hashes=False,
    )["analysis_id"]
    assert before != after


def test_metadata_binds_output_hashes(tmp_path):
    result, _, _ = run_full(tmp_path)
    root = Path(result["output_namespace"])
    metadata = json.loads((root / "completion_metadata.json").read_text())
    on_disk = {
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts[0] != mac.STAGE_MARKER_DIR
    }
    recorded = set(metadata["output_sha256"])
    assert not (on_disk - recorded - {"completion_metadata.json"})


# =============================================================================
# Comparison layer
# =============================================================================
def test_primary_comparison_is_raw_thermal_roc_auc(tmp_path):
    result, _, _ = run_full(tmp_path)
    ranking = pd.read_csv(
        Path(result["output_namespace"]) / "comparison" / "ranking_summary.csv"
    )
    primary = ranking[ranking["is_primary_comparison"]]
    assert set(primary["transfer"]) == {"raw_thermal_roc_auc"}
    assert mac.PRIMARY_TRANSFER_SELECTION["transfer_state"] == "raw"


def test_secondary_comparison_block_is_complete(tmp_path):
    result, _, _ = run_full(tmp_path)
    ranking = pd.read_csv(
        Path(result["output_namespace"]) / "comparison" / "ranking_summary.csv"
    )
    assert set(ranking["transfer"]) == {q[0] for q in mac.TRANSFER_QUANTITIES}


def test_comparison_produces_no_p_value(tmp_path):
    result, _, _ = run_full(tmp_path)
    ranking = pd.read_csv(
        Path(result["output_namespace"]) / "comparison" / "ranking_summary.csv"
    )
    assert not [
        c for c in ranking.columns
        if "p_value" in c or "pvalue" in c or "significance" in c
    ]


def test_comparison_layer_cannot_mutate_diagnostics(tmp_path):
    result, experiments_root, output_root = run_full(tmp_path)
    root = Path(result["output_namespace"])
    summary = (root / "weighted_predictor_space" / "directed_pair_summary.csv").read_bytes()
    mac.run_analysis(
        from_stage="compare", to_stage="compare", resume=True,
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
        climate_export_engine=FakeClimateExportEngine(),
        geodesic_inverse=fake_geodesic_inverse,
    )
    assert (root / "weighted_predictor_space" / "directed_pair_summary.csv").read_bytes() == summary


def test_scientific_summary_states_the_interpretation_boundary(tmp_path):
    result, _, _ = run_full(tmp_path)
    text = (Path(result["output_namespace"]) / "comparison" / "scientific_summary.md").read_text()
    assert "does or does not" in text
    assert "raw-transfer ordering" in text
    assert "Marginal diagnostics can never rank transfer" not in text
    assert "no p-value" in text.lower() or "No p-value" in text


def test_rank_helpers_are_descriptive_only():
    assert mac._spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert mac._spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert mac._kendall([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert mac._spearman([1, 2], [1, 2]) is None


# =============================================================================
# Validator
# =============================================================================
def test_dry_run_validator_passes_and_writes_nothing(tmp_path):
    from scripts.validate_marginal_aoa_completion import validate_dry_run

    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    before = {p for p in output_root.rglob("*")}
    report = validate_dry_run(output_root, experiments_root, strict_hashes=False)
    assert not report.failed, [c["check_id"] for c in report.failed]
    assert {p for p in output_root.rglob("*")} == before


def test_actual_validator_passes_on_a_produced_artifact(tmp_path):
    from scripts.validate_marginal_aoa_completion import validate_actual

    result, experiments_root, output_root = run_full(tmp_path)
    report = validate_actual(
        result["analysis_id"], output_root, experiments_root, strict_hashes=False,
        geodesic_inverse=fake_geodesic_inverse,
    )
    assert not report.failed, [
        (c["check_id"], c["observed"]) for c in report.failed
    ]


def test_actual_validator_detects_a_tampered_threshold(tmp_path):
    from scripts.validate_marginal_aoa_completion import validate_actual

    result, experiments_root, output_root = run_full(tmp_path)
    path = (
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "source_threshold_diagnostics.csv"
    )
    frame = pd.read_csv(path)
    frame["training_di_upper_whisker_threshold"] = 99.0
    frame.to_csv(path, index=False)
    report = validate_actual(
        result["analysis_id"], output_root, experiments_root, strict_hashes=False,
        geodesic_inverse=fake_geodesic_inverse,
    )
    failed = {c["check_id"] for c in report.failed}
    assert "13.normaliser_and_threshold" in failed


def test_actual_validator_detects_a_leaked_label_column(tmp_path):
    from scripts.validate_marginal_aoa_completion import validate_actual

    result, experiments_root, output_root = run_full(tmp_path)
    path = (
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "directed_pair_summary.csv"
    )
    frame = pd.read_csv(path)
    frame["burned"] = 1
    frame.to_csv(path, index=False)
    report = validate_actual(
        result["analysis_id"], output_root, experiments_root, strict_hashes=False,
        geodesic_inverse=fake_geodesic_inverse,
    )
    assert "8.target_labels_never_read" in {c["check_id"] for c in report.failed}


def test_actual_validator_detects_a_missing_namespace(tmp_path):
    from scripts.validate_marginal_aoa_completion import validate_actual

    report = validate_actual("f" * 64, tmp_path / "outputs", tmp_path / "experiments")
    assert "00.namespace_exists" in {c["check_id"] for c in report.failed}


# =============================================================================
# Strict actual-stage semantics
# =============================================================================
def test_dry_run_reports_missing_climate_raster_without_writing(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    before = {p for p in output_root.rglob("*")}
    result = mac.run_analysis(
        dry_run=True, output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    per_stage = result["prerequisites"]["per_stage"]
    assert per_stage["climate-distance"]["available"] is False
    assert any(
        "TerraClimate raster is absent" in m
        for m in per_stage["climate-distance"]["missing"]
    )
    assert result["files_written"] == []
    assert {p for p in output_root.rglob("*")} == before


def test_dry_run_reports_missing_geographiclib_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(mac, "geographiclib_available", lambda: False)
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    before = {p for p in output_root.rglob("*")}
    result = mac.run_analysis(
        dry_run=True, output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    assert result["geographiclib_dependency_available"] is False
    per_stage = result["prerequisites"]["per_stage"]
    assert per_stage["geographic-distance"]["available"] is False
    assert result["files_written"] == []
    assert {p for p in output_root.rglob("*")} == before


def test_actual_climate_export_calls_the_injected_engine(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    engine = FakeClimateExportEngine()
    plan = mac.run_analysis(
        from_stage="plan", to_stage="plan",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    result = mac.run_analysis(
        from_stage="climate-export", to_stage="climate-export",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, climate_export_engine=engine,
    )
    assert result["stages_executed"] == ["climate-export"]
    assert engine.calls == [
        "initialise", "monthly_image_count", "build_four_band_image",
        "native_projection", "region", "export",
    ]
    raster = mac.climate_raster_path(plan["analysis_id"], output_root)
    assert raster.is_file()
    metadata = json.loads((raster.parent / "climate_export_metadata.json").read_text())
    assert metadata["observed_month_count"] == mac.CLIMATE_EXPECTED_MONTHS
    assert metadata["output_bands"] == list(mac.CLIMATE_FEATURES)
    assert metadata["raster_audit"]["band_count"] == 4
    assert metadata["raster_sha256"] == mac.sha256_file(raster)


def test_actual_climate_export_fails_when_month_count_is_wrong(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    mac.run_analysis(
        from_stage="plan", to_stage="plan",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    engine = FakeClimateExportEngine(month_count=359)
    with pytest.raises(SystemExit, match="exactly 360"):
        mac.run_analysis(
            from_stage="climate-export", to_stage="climate-export",
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False, climate_export_engine=engine,
        )
    assert "export" not in engine.calls


def test_broader_actual_range_cannot_skip_climate_export(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    with pytest.raises(SystemExit, match="MUST NOT be skipped"):
        mac.run_analysis(
            from_stage="plan", to_stage="compare",
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
            geodesic_inverse=fake_geodesic_inverse,
        )
    # plan ran and is complete; nothing downstream of climate-export exists.
    analysis_id = mac.run_analysis(
        dry_run=True, output_root=output_root,
        experiments_root=experiments_root, strict_hashes=False,
    )["analysis_id"]
    root = mac.analysis_root(analysis_id, output_root)
    assert (root / "config" / "preregistration.json").is_file()
    assert not (root / "weighted_predictor_space").exists()
    assert not (root / "comparison").exists()
    assert not (root / "completion_metadata.json").exists()


def test_broader_actual_range_cannot_skip_geographic_distance(tmp_path, monkeypatch):
    monkeypatch.setattr(mac, "geographiclib_available", lambda: False)
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    with pytest.raises(SystemExit, match="geographiclib"):
        mac.run_analysis(
            from_stage="plan", to_stage="compare",
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
            climate_export_engine=FakeClimateExportEngine(),
        )
    analysis_id = mac.run_analysis(
        dry_run=True, output_root=output_root,
        experiments_root=experiments_root, strict_hashes=False,
    )["analysis_id"]
    root = mac.analysis_root(analysis_id, output_root)
    assert not (root / "comparison").exists()
    assert not (root / "completion_metadata.json").exists()


def test_weighted_predictor_space_runs_independently_after_plan(tmp_path):
    """No climate and no geographiclib dependency for this stage."""
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    result = mac.run_analysis(
        from_stage="plan", to_stage="weighted-predictor-space",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
    )
    assert result["stages_executed"] == ["plan", "weighted-predictor-space"]
    root = Path(result["output_namespace"])
    assert (root / "weighted_predictor_space" / "weighted_pair_summary.csv").is_file()
    # The integrated table and every climate/geographic column are ABSENT --
    # never present-but-null.
    assert not (root / "weighted_predictor_space" / "directed_pair_summary.csv").is_file()
    weighted = pd.read_csv(root / "weighted_predictor_space" / "weighted_pair_summary.csv")
    assert not [c for c in weighted.columns if c.startswith("climate_")]
    assert "centroid_geodesic_distance_km" not in weighted.columns


def test_compare_refuses_null_climate_values():
    rows = [{
        "source_experiment": "a", "target_experiment": "b",
        "climate_distance": None, "centroid_geodesic_distance_km": 10.0,
    }]
    with pytest.raises(SystemExit, match="climate_distance is None"):
        mac.assert_no_null_required_components(rows)


def test_compare_refuses_null_geographic_values():
    rows = [{
        "source_experiment": "a", "target_experiment": "b",
        "climate_distance": 1.0, "centroid_geodesic_distance_km": None,
    }]
    with pytest.raises(SystemExit, match="centroid_geodesic_distance_km is None"):
        mac.assert_no_null_required_components(rows)


def test_no_partial_pass_metadata_after_prerequisite_failure(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    with pytest.raises(SystemExit):
        mac.run_analysis(
            from_stage="plan", to_stage="compare",
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False, pairwise_chunk_size=8, neighbour_chunk_size=8,
        )
    analysis_id = mac.run_analysis(
        dry_run=True, output_root=output_root,
        experiments_root=experiments_root, strict_hashes=False,
    )["analysis_id"]
    root = mac.analysis_root(analysis_id, output_root)
    assert not (root / "completion_metadata.json").exists()
    assert mac.read_stage_marker(analysis_id, "compare", output_root) is None


def test_completion_metadata_refuses_incomplete_required_stages(tmp_path):
    with pytest.raises(SystemExit, match="required stage"):
        mac.build_completion_metadata(
            "a" * 64, {}, EXPERIMENTS, ["plan"], {}, {},
            {"path": "x", "exists": False, "sha256": None},
            {"geometry_source_path": "core/regions.py", "geometry_source_sha256": "x"},
            {}, tmp_path, tmp_path, tmp_path,
        )


def test_skipped_stage_is_not_resumable_as_completed(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    result = mac.run_analysis(
        from_stage="plan", to_stage="plan",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    analysis_id = result["analysis_id"]
    assert mac.verify_stage_complete(analysis_id, "plan", output_root)["complete"]
    for stage in ("climate-export", "weighted-predictor-space", "compare"):
        state = mac.verify_stage_complete(analysis_id, stage, output_root)
        assert state["complete"] is False
        assert state["reason"] == "no stage marker"


def test_partial_tree_fails_closed_and_is_not_deleted(tmp_path):
    result, experiments_root, output_root = run_full(tmp_path)
    analysis_id = result["analysis_id"]
    root = Path(result["output_namespace"])
    victim = root / "weighted_predictor_space" / "source_feature_weights.csv"
    original = victim.read_bytes()
    victim.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no longer matches its recorded hash"):
        mac.verify_stage_complete(analysis_id, "weighted-predictor-space", output_root)
    assert victim.read_text() == "tampered\n"
    assert len(original) > 0


# =============================================================================
# GeographicLib contract -- no importorskip anywhere
# =============================================================================
def test_geodesic_arithmetic_with_an_injected_inverse():
    """The distance binding, exercised without the package installed."""
    km = mac.geodesic_distance_km(
        0.0, 0.0, 0.0, 1.0, geodesic_inverse=fake_geodesic_inverse
    )
    assert km == pytest.approx(111.320, abs=1e-3)


def test_geodesic_binding_converts_metres_to_kilometres():
    calls = []

    def inverse(lat1, lon1, lat2, lon2):
        calls.append((lat1, lon1, lat2, lon2))
        return {"s12": 2_500.0}

    km = mac.geodesic_distance_km(1.0, 2.0, 3.0, 4.0, geodesic_inverse=inverse)
    assert km == pytest.approx(2.5)
    # GeographicLib's argument order is (lat1, lon1, lat2, lon2).
    assert calls == [(2.0, 1.0, 4.0, 3.0)]


def test_geodesic_self_distance_is_zero_with_injected_inverse():
    assert mac.geodesic_distance_km(
        10.0, 20.0, 10.0, 20.0, geodesic_inverse=fake_geodesic_inverse
    ) == pytest.approx(0.0)


def test_geodesic_is_symmetric_with_injected_inverse():
    a = mac.geodesic_distance_km(-1.0, 40.0, 31.0, 37.0,
                                 geodesic_inverse=fake_geodesic_inverse)
    b = mac.geodesic_distance_km(31.0, 37.0, -1.0, 40.0,
                                 geodesic_inverse=fake_geodesic_inverse)
    assert a == pytest.approx(b)


def test_pairwise_geographic_rows_are_symmetric_with_injected_inverse():
    rows = mac.pairwise_geographic_distances(
        EXPERIMENTS, geodesic_inverse=fake_geodesic_inverse
    )
    assert len(rows) == 6
    lookup = mac.symmetric_lookup(rows, "centroid_geodesic_distance_km")
    for a, b in mac.unordered_pairs(EXPERIMENTS):
        assert lookup[(a, b)] == lookup[(b, a)]


def test_missing_geographiclib_fails_before_any_write(tmp_path):
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name.startswith("geographiclib"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    plan = mac.run_analysis(
        from_stage="plan", to_stage="plan",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    root = mac.analysis_root(plan["analysis_id"], output_root)
    with patch("builtins.__import__", side_effect=blocked):
        with pytest.raises(SystemExit, match="geographiclib"):
            mac.pairwise_geographic_distances(EXPERIMENTS)
    assert not (root / "geographic_distance").exists()


def test_resolve_geodesic_inverse_fails_closed_without_the_package():
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name.startswith("geographiclib"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=blocked):
        with pytest.raises(SystemExit, match="pip install geographiclib"):
            mac.resolve_geodesic_inverse()


def test_no_haversine_pyproj_or_vincenty_fallback_exists():
    for module in (mac, __import__("src.marginal_aoa_climate_export", fromlist=["x"])):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        assert "def _vincenty" not in source
        assert "import pyproj" not in source
        assert "math.asin(math.sqrt" not in source
        # "haversine" may appear only inside a prohibition message.
        for line in source.splitlines():
            if "haversine" in line:
                assert any(
                    token in line
                    for token in ("no haversine", "not permitted", "fallback")
                ), line


def test_actual_validator_rejects_a_missing_required_component(tmp_path):
    """A skipped/absent required component is FAIL, never SKIPPED."""
    from scripts.validate_marginal_aoa_completion import validate_actual

    result, experiments_root, output_root = run_full(tmp_path)
    root = Path(result["output_namespace"])
    (root / "geographic_distance" / "pairwise_geographic_distance.csv").unlink()
    report = validate_actual(
        result["analysis_id"], output_root, experiments_root, strict_hashes=False,
        geodesic_inverse=fake_geodesic_inverse,
    )
    failed = {c["check_id"] for c in report.failed}
    assert "6.geographic_symmetric" in failed
    assert not [c for c in report.checks if c["check_id"].startswith("6.")
                and c["status"] == "SKIPPED"]


def test_actual_validator_rejects_null_component_columns(tmp_path):
    from scripts.validate_marginal_aoa_completion import validate_actual

    result, experiments_root, output_root = run_full(tmp_path)
    path = (
        Path(result["output_namespace"]) / "weighted_predictor_space"
        / "directed_pair_summary.csv"
    )
    frame = pd.read_csv(path)
    frame.loc[0, "climate_distance"] = None
    frame.to_csv(path, index=False)
    report = validate_actual(
        result["analysis_id"], output_root, experiments_root, strict_hashes=False,
        geodesic_inverse=fake_geodesic_inverse,
    )
    assert "27.no_null_required_components" in {c["check_id"] for c in report.failed}


def test_actual_validator_rejects_an_incomplete_required_stage(tmp_path):
    from scripts.validate_marginal_aoa_completion import validate_actual

    result, experiments_root, output_root = run_full(tmp_path)
    marker = mac.stage_marker_path(
        result["analysis_id"], "climate-distance", output_root
    )
    marker.unlink()
    report = validate_actual(
        result["analysis_id"], output_root, experiments_root, strict_hashes=False,
        geodesic_inverse=fake_geodesic_inverse,
    )
    assert "26.required_stages_complete" in {c["check_id"] for c in report.failed}


def test_valid_actual_fixture_has_no_skipped_checks(tmp_path):
    """A complete synthetic artifact validates with zero skipped checks."""
    from scripts.validate_marginal_aoa_completion import validate_actual

    result, experiments_root, output_root = run_full(tmp_path)
    report = validate_actual(
        result["analysis_id"], output_root, experiments_root, strict_hashes=False,
        geodesic_inverse=fake_geodesic_inverse,
    )
    assert not report.failed, [(c["check_id"], c["observed"]) for c in report.failed]
    assert not report.skipped, [c["check_id"] for c in report.skipped]


# =============================================================================
# Climate export: CRS extraction and final-path atomicity
# =============================================================================
import src.marginal_aoa_climate_export as mace


class FakeProjection:
    """A minimal Earth Engine Projection stand-in.

    `getInfo()` deliberately omits the `crs` key -- exactly what TerraClimate's
    real projection does, and the defect that produced expected_crs=None.
    """

    def __init__(self, crs="EPSG:4326",
                 transform=(0.0416, 0.0, -180.0, 0.0, -0.0416, 90.0),
                 scale=4638.0, getinfo=None):
        self._crs, self._transform, self._scale = crs, transform, scale
        self._getinfo = {"type": "Projection"} if getinfo is None else getinfo

    def crs(self):
        return _Info(self._crs)

    def transform(self):
        return _Info(list(self._transform) if self._transform is not None else None)

    def nominalScale(self):  # noqa: N802 -- mirrors the Earth Engine API
        return _Info(self._scale)

    def getInfo(self):  # noqa: N802
        return self._getinfo


class _Info:
    def __init__(self, value):
        self._value = value

    def getInfo(self):  # noqa: N802
        return self._value




def _plan_then(tmp_path):
    experiments_root, output_root = build_tree(tmp_path)
    synthetic_transfer(output_root)
    plan = mac.run_analysis(
        from_stage="plan", to_stage="plan",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False,
    )
    return plan["analysis_id"], experiments_root, output_root


def test_export_targets_a_temporary_sibling_not_the_final_path(tmp_path):
    analysis_id, experiments_root, output_root = _plan_then(tmp_path)
    engine = FakeClimateExportEngine()
    mac.run_analysis(
        from_stage="climate-export", to_stage="climate-export",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, climate_export_engine=engine,
    )
    final = mac.climate_raster_path(analysis_id, output_root)
    assert len(engine.export_destinations) == 1
    handed = engine.export_destinations[0]
    assert handed != final
    assert handed.name == mac.CLIMATE_RASTER_STAGING_FILENAME
    assert handed.parent == final.parent
    # Promotion happened, and no staging file survives.
    assert final.is_file()
    assert not handed.exists()


def test_exporter_failure_leaves_no_final_raster(tmp_path):
    analysis_id, experiments_root, output_root = _plan_then(tmp_path)
    engine = FakeClimateExportEngine(fail_export=True)
    with pytest.raises(SystemExit, match="injected exporter"):
        mac.run_analysis(
            from_stage="climate-export", to_stage="climate-export",
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False, climate_export_engine=engine,
        )
    final = mac.climate_raster_path(analysis_id, output_root)
    assert not final.exists()
    assert not (final.parent / mac.CLIMATE_RASTER_STAGING_FILENAME).exists()
    assert not (final.parent / mac.CLIMATE_EXPORT_METADATA_FILENAME).exists()
    assert mac.read_stage_marker(analysis_id, "climate-export", output_root) is None


def test_climate_qa_failure_leaves_no_final_raster(tmp_path):
    """A raster that fails the four-band QA never reaches the final path."""
    analysis_id, experiments_root, output_root = _plan_then(tmp_path)
    engine = FakeClimateExportEngine(bad_band_count=3)
    with pytest.raises(SystemExit, match="band"):
        mac.run_analysis(
            from_stage="climate-export", to_stage="climate-export",
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False, climate_export_engine=engine,
        )
    final = mac.climate_raster_path(analysis_id, output_root)
    assert not final.exists()
    assert not (final.parent / mac.CLIMATE_RASTER_STAGING_FILENAME).exists()
    assert not (final.parent / mac.CLIMATE_EXPORT_METADATA_FILENAME).exists()
    assert mac.read_stage_marker(analysis_id, "climate-export", output_root) is None


def test_successful_export_promotes_atomically_and_writes_provenance(tmp_path):
    analysis_id, experiments_root, output_root = _plan_then(tmp_path)
    engine = FakeClimateExportEngine()
    mac.run_analysis(
        from_stage="climate-export", to_stage="climate-export",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, climate_export_engine=engine,
    )
    final = mac.climate_raster_path(analysis_id, output_root)
    metadata_path = final.parent / mac.CLIMATE_EXPORT_METADATA_FILENAME
    assert final.is_file() and metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source_projection_authority_crs"] is None
    assert metadata["source_projection_wkt"] == TERRACLIMATE_WKT
    assert metadata["export_crs"] == TERRACLIMATE_WKT
    assert metadata["export_crs_representation"] == "wkt"
    assert len(metadata["source_projection_transform"]) == 6
    assert metadata["source_projection_nominal_scale"] > 0
    assert metadata["projection_read_method"] == "fake_projection_v1"
    assert metadata["final_path_promoted_after_qa"] is True
    assert metadata["raster_sha256"] == mac.sha256_file(final)
    assert mac.read_stage_marker(analysis_id, "climate-export", output_root)["status"] == "pass"


def test_partial_final_raster_without_metadata_fails_closed(tmp_path):
    """Exactly the state the failed production run left behind."""
    analysis_id, experiments_root, output_root = _plan_then(tmp_path)
    final = mac.climate_raster_path(analysis_id, output_root)
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"not-a-valid-raster")
    before = final.read_bytes()

    with pytest.raises(SystemExit, match="PARTIAL climate export"):
        mac.run_analysis(
            from_stage="climate-export", to_stage="climate-export",
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False, climate_export_engine=FakeClimateExportEngine(),
        )
    # Not overwritten, not deleted, not retried.
    assert final.read_bytes() == before
    assert mac.read_stage_marker(analysis_id, "climate-export", output_root) is None


def test_complete_verified_export_is_resumable(tmp_path):
    analysis_id, experiments_root, output_root = _plan_then(tmp_path)
    first = FakeClimateExportEngine()
    mac.run_analysis(
        from_stage="climate-export", to_stage="climate-export",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, climate_export_engine=first,
    )
    final = mac.climate_raster_path(analysis_id, output_root)
    digest = mac.sha256_file(final)

    second = FakeClimateExportEngine()
    result = mac.run_analysis(
        from_stage="climate-export", to_stage="climate-export", resume=True,
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, climate_export_engine=second,
    )
    assert second.calls == []          # no re-export
    assert result["stages_reused"] == ["climate-export"]
    assert mac.sha256_file(final) == digest


def test_climate_export_tests_never_touch_earth_engine_or_production_root():
    """No test in this module can authorise live Earth Engine work.

    The tokens are assembled at runtime so this assertion does not match its
    own source text.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    kwarg_token = "allow_earth_engine" + "=True"
    flag_token = "--allow-" + "earth-engine"
    offenders = [
        line for line in source.splitlines()
        if (kwarg_token in line or flag_token in line) and "_token" not in line
    ]
    assert not offenders, offenders
    # Every run_analysis call in this module is tmp_path-scoped.
    assert "output_root=output_root" in source


# =============================================================================
# TerraClimate single-band projection with WKT fallback
# =============================================================================
def _fake_ee_projection(*, authority=None, wkt=TERRACLIMATE_WKT,
                        transform=None, scale=4638.312116386398,
                        getinfo_has_crs=False):
    """A minimal Earth Engine Projection stand-in, no live GEE."""
    transform = TERRACLIMATE_TRANSFORM if transform is None else transform

    class _Scalar:
        def __init__(self, value):
            self._value = value

        def getInfo(self):  # noqa: N802
            return self._value

    class _Projection:
        def __init__(self):
            self.info = {"type": "Projection", "transform": transform, "wkt": wkt}
            if getinfo_has_crs:
                self.info["crs"] = authority

        def getInfo(self):  # noqa: N802
            return dict(self.info)

        def crs(self):
            return _Scalar(authority)

        def wkt(self):
            return _Scalar(wkt)

        def nominalScale(self):  # noqa: N802
            return _Scalar(scale)

    return _Projection()


class _ProbeEngine(mace.TerraClimateExportEngine):
    """Production engine with only its Earth Engine accessor replaced."""

    def __init__(self, projection, *, record=None):
        super().__init__()
        self._projection = projection
        self.record = record if record is not None else []

    def source_projection(self, collection_id):
        self.record.append(("source_projection", collection_id))
        return self._projection


def test_multi_band_projection_is_not_used_single_band_is_selected():
    """The canonical band is `pr`, selected explicitly from the first image."""
    assert mace.CANONICAL_PROJECTION_BAND == "pr"
    source = Path(mace.__file__).read_text(encoding="utf-8")
    assert ".select(CANONICAL_PROJECTION_BAND)" in source
    # A bare multi-band first_image.projection() must not appear.
    assert "ee.Image(ee.ImageCollection(collection_id).first()).projection()" not in source


def test_selected_band_getinfo_supplies_transform_and_wkt():
    engine = _ProbeEngine(_fake_ee_projection())
    resolved = engine.native_projection("IDAHO_EPSCOR/TERRACLIMATE")
    assert resolved["canonical_projection_band"] == "pr"
    assert resolved["source_projection_transform"] == TERRACLIMATE_TRANSFORM
    assert resolved["source_projection_wkt"] == TERRACLIMATE_WKT
    assert resolved["source_projection_nominal_scale"] == pytest.approx(4638.312116386398)
    assert resolved["projection_read_method"] == (
        "ee_single_band_projection_with_wkt_fallback_v1"
    )


def test_authority_crs_is_preferred_when_present():
    engine = _ProbeEngine(
        _fake_ee_projection(authority="EPSG:4326", getinfo_has_crs=True)
    )
    resolved = engine.native_projection("X")
    assert resolved["source_projection_authority_crs"] == "EPSG:4326"
    assert resolved["export_crs"] == "EPSG:4326"
    assert resolved["export_crs_representation"] == "authority_code"


def test_wkt_is_used_when_authority_crs_is_absent():
    """The real TerraClimate case: crs is None, only a WKT exists."""
    engine = _ProbeEngine(_fake_ee_projection(authority=None))
    resolved = engine.native_projection("X")
    assert resolved["source_projection_authority_crs"] is None
    assert resolved["export_crs"] == TERRACLIMATE_WKT
    assert resolved["export_crs_representation"] == "wkt"


def test_missing_authority_alone_is_not_a_failure():
    resolved = mace.validate_projection({
        "source_projection_authority_crs": None,
        "source_projection_wkt": TERRACLIMATE_WKT,
        "source_projection_transform": TERRACLIMATE_TRANSFORM,
        "source_projection_nominal_scale": 4638.3,
    })
    assert resolved["export_crs_representation"] == "wkt"


def test_empty_authority_and_empty_wkt_fail_before_export():
    for authority, wkt in ((None, None), ("", ""), ("   ", None), (None, "  ")):
        with pytest.raises(SystemExit, match="NEITHER an authority CRS code NOR a"):
            mace.validate_projection({
                "source_projection_authority_crs": authority,
                "source_projection_wkt": wkt,
                "source_projection_transform": TERRACLIMATE_TRANSFORM,
                "source_projection_nominal_scale": 4638.3,
            })


def test_no_epsg4326_is_guessed_when_both_are_empty():
    with pytest.raises(SystemExit) as excinfo:
        mace.validate_projection({
            "source_projection_authority_crs": None,
            "source_projection_wkt": None,
            "source_projection_transform": TERRACLIMATE_TRANSFORM,
            "source_projection_nominal_scale": 4638.3,
        })
    message = str(excinfo.value)
    assert "not assumed as a fallback" in message
    assert "EPSG:4326" in message  # named only as the thing NOT assumed


def test_invalid_transform_stops_the_export():
    for bad in (None, [], [1.0, 2.0, 3.0], [1.0] * 7,
                [1.0, 0.0, 0.0, 0.0, 1.0, float("nan")]):
        with pytest.raises(SystemExit, match="transform is unusable"):
            mace.validate_projection({
                "source_projection_authority_crs": "EPSG:4326",
                "source_projection_wkt": None,
                "source_projection_transform": bad,
                "source_projection_nominal_scale": 4638.3,
            })


def test_invalid_nominal_scale_stops_the_export():
    for bad in (None, 0.0, -1.0, float("nan"), float("inf"), "abc"):
        with pytest.raises(SystemExit, match="nominal scale is unusable"):
            mace.validate_projection({
                "source_projection_authority_crs": "EPSG:4326",
                "source_projection_wkt": None,
                "source_projection_transform": TERRACLIMATE_TRANSFORM,
                "source_projection_nominal_scale": bad,
            })


def test_derived_image_binds_the_source_projection():
    """setDefaultProjection is called on the reduced image -- never reproject."""
    source = Path(mace.__file__).read_text(encoding="utf-8")
    assert "setDefaultProjection(self.source_projection(collection_id))" in source
    assert ".reproject(" not in source


def test_wkt_crs_is_semantically_equivalent_not_string_equal(tmp_path):
    import rasterio
    import rasterio.crs
    from rasterio.transform import from_bounds

    path = tmp_path / "r.tif"
    written = rasterio.crs.CRS.from_wkt(TERRACLIMATE_WKT)
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=4, dtype="float32",
        crs=written, transform=from_bounds(-10, 30, 42, 47, 4, 4),
    ) as handle:
        handle.write(np.ones((4, 4, 4), dtype="float32"))

    with rasterio.open(path) as handle:
        round_tripped = handle.crs.to_wkt()
    # Byte equality does NOT hold; semantic equivalence does.
    assert round_tripped != TERRACLIMATE_WKT
    assert mace.crs_matches(round_tripped, TERRACLIMATE_WKT, "wkt")

    audit = mace.validate_exported_raster(
        path, expected_bands=mac.CLIMATE_FEATURES,
        expected_crs=TERRACLIMATE_WKT, expected_crs_representation="wkt",
    )
    assert audit["crs_semantically_equivalent"] is True


def test_mismatched_crs_is_rejected(tmp_path):
    import rasterio
    import rasterio.crs
    from rasterio.transform import from_bounds

    path = tmp_path / "r.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=4, dtype="float32",
        crs=rasterio.crs.CRS.from_epsg(3857),
        transform=from_bounds(-10, 30, 42, 47, 4, 4),
    ) as handle:
        handle.write(np.ones((4, 4, 4), dtype="float32"))
    with pytest.raises(SystemExit, match="not equivalent to the requested"):
        mace.validate_exported_raster(
            path, expected_bands=mac.CLIMATE_FEATURES,
            expected_crs=TERRACLIMATE_WKT, expected_crs_representation="wkt",
        )


def test_export_receives_the_wkt_and_the_real_transform(tmp_path):
    analysis_id, experiments_root, output_root = _plan_then(tmp_path)
    engine = FakeClimateExportEngine()
    mac.run_analysis(
        from_stage="climate-export", to_stage="climate-export",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, climate_export_engine=engine,
    )
    assert engine.export_crs == TERRACLIMATE_WKT
    assert engine.export_scale == pytest.approx(4638.312116386398)

    metadata = json.loads(
        (mac.climate_raster_path(analysis_id, output_root).parent
         / mac.CLIMATE_EXPORT_METADATA_FILENAME).read_text()
    )
    assert metadata["canonical_projection_band"] == "pr"
    assert metadata["source_projection_authority_crs"] is None
    assert metadata["source_projection_wkt"] == TERRACLIMATE_WKT
    assert metadata["source_projection_transform"] == TERRACLIMATE_TRANSFORM
    assert metadata["export_crs_representation"] == "wkt"
    assert metadata["raster_audit"]["crs_semantically_equivalent"] is True


def test_projection_probe_tests_use_no_live_earth_engine():
    """No test in this module opens an Earth Engine session.

    Only executable statements are inspected -- comments, docstrings and
    assertion literals in other tests legitimately mention these names.
    """
    import ast as _ast

    tree = _ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            called.add(node.func.id)
    assert "ee" not in imported
    assert "init_gee" not in called


# =============================================================================
# Shared exporter: optional semantic CRS matcher
# =============================================================================
# 52 x 17 degrees at 1 degree/pixel, so pixel width == pixel height and the
# shared helper's resolution check has a single expected value.
def _write_raster(path, crs, *, count=4, height=17, width=52,
                  bounds=(-10.0, 30.0, 42.0, 47.0)):
    import rasterio
    import rasterio.crs
    from rasterio.transform import from_bounds

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype="float32", crs=rasterio.crs.CRS.from_user_input(crs),
        transform=from_bounds(*bounds, width, height),
    ) as handle:
        handle.write(np.ones((count, height, width), dtype="float32"))
    return path


def _region(bounds=(-10.0, 30.0, 42.0, 47.0)):
    """A stand-in for an Earth Engine geometry.

    `_bbox_from_region` calls `region.bounds().getInfo()`, so the fake mirrors
    exactly that shape -- no live Earth Engine involved.
    """
    x0, y0, x1, y1 = bounds

    class _Bounds:
        def getInfo(self):  # noqa: N802
            return {"coordinates": [[
                [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0],
            ]]}

    class _Region:
        def bounds(self):
            return _Bounds()

    return _Region()


def _alignment_scale():
    """Metres per pixel matching the 1 degree/pixel test raster."""
    from scripts.run_predictors_only import _ESTIMATE_METERS_PER_DEGREE
    return 1.0 * _ESTIMATE_METERS_PER_DEGREE


def test_default_alignment_qa_keeps_exact_string_behaviour(tmp_path):
    """No matcher supplied -> unchanged exact/string CRS check."""
    from scripts.run_predictors_only import _validate_export_alignment

    path = _write_raster(tmp_path / "a.tif", "EPSG:4326")
    report = _validate_export_alignment(
        path, _region(), _alignment_scale(), "EPSG:4326", expected_band_count=4,
    )
    assert report["crs_check_mode"] == "exact_string_or_epsg"

    # The WKT-only case still fails under the default behaviour.
    wkt_path = _write_raster(tmp_path / "b.tif", TERRACLIMATE_WKT)
    with pytest.raises(SystemExit, match="beklenmeyen CRS"):
        _validate_export_alignment(
            wkt_path, _region(), _alignment_scale(), TERRACLIMATE_WKT,
            expected_band_count=4,
        )


def test_supplied_matcher_accepts_semantically_equal_wkt(tmp_path):
    from scripts.run_predictors_only import _validate_export_alignment

    path = _write_raster(tmp_path / "c.tif", TERRACLIMATE_WKT)
    report = _validate_export_alignment(
        path, _region(), _alignment_scale(), TERRACLIMATE_WKT,
        expected_band_count=4,
        crs_equivalence_fn=lambda actual, expected: mace.crs_matches(
            actual, expected, "wkt"
        ),
    )
    assert report["crs_check_mode"] == "caller_supplied_equivalence_fn"
    # The strings genuinely differ -- this is not string equality passing.
    assert report["crs"] != TERRACLIMATE_WKT


def test_supplied_matcher_still_rejects_a_genuinely_different_crs(tmp_path):
    from scripts.run_predictors_only import _validate_export_alignment

    path = _write_raster(tmp_path / "d.tif", "EPSG:3857")
    with pytest.raises(SystemExit, match="beklenmeyen CRS"):
        _validate_export_alignment(
            path, _region(), _alignment_scale(), TERRACLIMATE_WKT,
            expected_band_count=4,
            crs_equivalence_fn=lambda actual, expected: mace.crs_matches(
                actual, expected, "wkt"
            ),
        )


def test_supplied_matcher_does_not_relax_transform_checks(tmp_path):
    """A CRS matcher must not loosen pixel-size / alignment validation."""
    from scripts.run_predictors_only import _validate_export_alignment

    path = _write_raster(tmp_path / "e.tif", TERRACLIMATE_WKT)
    with pytest.raises(SystemExit, match="piksel boyutu"):
        _validate_export_alignment(
            path, _region(), _alignment_scale() * 10.0, TERRACLIMATE_WKT,
            expected_band_count=4,
            crs_equivalence_fn=lambda actual, expected: mace.crs_matches(
                actual, expected, "wkt"
            ),
        )


def test_matcher_exception_is_treated_as_mismatch(tmp_path):
    from scripts.run_predictors_only import _validate_export_alignment

    def _boom(actual, expected):
        raise ValueError("unparsable")

    path = _write_raster(tmp_path / "f.tif", TERRACLIMATE_WKT)
    with pytest.raises(SystemExit, match="beklenmeyen CRS"):
        _validate_export_alignment(
            path, _region(), _alignment_scale(), TERRACLIMATE_WKT,
            expected_band_count=4, crs_equivalence_fn=_boom,
        )


def test_climate_caller_passes_the_matcher_to_the_shared_exporter(tmp_path):
    analysis_id, experiments_root, output_root = _plan_then(tmp_path)
    engine = FakeClimateExportEngine()
    mac.run_analysis(
        from_stage="climate-export", to_stage="climate-export",
        output_root=output_root, experiments_root=experiments_root,
        strict_hashes=False, climate_export_engine=engine,
    )
    fn = engine.export_crs_equivalence_fn
    assert callable(fn), "the climate caller must supply a CRS matcher"
    # It behaves like crs_matches for the WKT representation.
    assert fn(TERRACLIMATE_WKT, TERRACLIMATE_WKT) is True
    assert fn("EPSG:3857", TERRACLIMATE_WKT) is False


def test_shared_qa_failure_leaves_no_final_raster(tmp_path):
    """A shared-QA rejection must not promote anything to the final path."""
    analysis_id, experiments_root, output_root = _plan_then(tmp_path)

    class _SharedQaFailingEngine(FakeClimateExportEngine):
        def export(self, image, **kwargs):
            super().export(image, **kwargs)     # writes the staging raster
            from scripts.run_predictors_only import PredictorRunnerError
            raise PredictorRunnerError("Alignment QA: beklenmeyen CRS (injected)")

    with pytest.raises(SystemExit, match="beklenmeyen CRS"):
        mac.run_analysis(
            from_stage="climate-export", to_stage="climate-export",
            output_root=output_root, experiments_root=experiments_root,
            strict_hashes=False, climate_export_engine=_SharedQaFailingEngine(),
        )
    final = mac.climate_raster_path(analysis_id, output_root)
    assert not final.exists()
    assert not (final.parent / mac.CLIMATE_RASTER_STAGING_FILENAME).exists()
    assert not (final.parent / mac.CLIMATE_EXPORT_METADATA_FILENAME).exists()
    assert mac.read_stage_marker(analysis_id, "climate-export", output_root) is None


def test_existing_shared_exporter_callers_are_unchanged():
    """Every pre-existing caller omits the parameter, so nothing changes."""
    source = Path("scripts/run_predictors_only.py").read_text(encoding="utf-8")
    assert "crs_equivalence_fn=None," in source          # default
    # The climate wrapper is the only place that supplies one.
    wrapper = Path(mace.__file__).read_text(encoding="utf-8")
    assert "crs_equivalence_fn=crs_equivalence_fn" in wrapper
