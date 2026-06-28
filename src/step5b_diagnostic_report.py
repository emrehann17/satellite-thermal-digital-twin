"""
step5b_diagnostic_report.py

Step5 raster çıktıları için tanı raporu üretir.

Bu script yeni preprocessing yapmaz ve anomaly değerlerini değiştirmez. Mevcut
Step5 GeoTIFF çıktılarını okur, temel istatistikleri, extreme anomaly overlap
oranlarını ve olası seam/artefact kaynak yorumunu raporlar.
"""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from core.config import (
    STEP5_MIN_BASELINE_STD_CELSIUS,
    STEP5_MIN_BASELINE_VALID_COUNT,
    STEP5_MIN_CURRENT_VALID_COUNT,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT


BASE_DIR = PROJECT_ROOT
STEP5_OUTPUT_DIR = BASE_DIR / "outputs" / "step5"
STEP5C_OUTPUT_DIR = BASE_DIR / "outputs" / "step5c"
DIAGNOSTIC_DIR = BASE_DIR / "outputs" / "step5b_diagnostics"
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step5b")

MAX_EDGE_MASK_DENSITY_PERCENT = 20.0
EVIDENCE_STRONG_THRESHOLD = 30.0
EVIDENCE_MODERATE_THRESHOLD = 10.0
EVIDENCE_WEAK_THRESHOLD = 3.0


RASTER_CANDIDATES = {
    "anomaly": [
        "anomaly_zscore.tif",
        "thermal_anomaly_zscore.tif",
    ],
    "current_median": [
        "current_period_median_celsius.tif",
        "current_period_lst_celsius.tif",
    ],
    "current_valid_count": [
        "current_period_valid_count.tif",
    ],
    "baseline_valid_count": [
        "baseline_valid_count.tif",
    ],
    "baseline_std": [
        "baseline_lst_std_celsius.tif",
        "baseline_std_celsius.tif",
    ],
    "modis_mean": [
        "modis_lst_mean_celsius_resampled.tif",
    ],
    "modis_std": [
        "modis_lst_std_celsius_resampled.tif",
    ],
    "modis_context": [
        "modis_context_zscore.tif",
    ],
}


@dataclass
class RasterLayer:
    """In-memory raster layer and minimal grid metadata."""

    key: str
    path: Path
    array: np.ndarray
    crs: str | None
    transform: Any
    width: int
    height: int


def find_existing_raster(candidates: list[str]) -> Path | None:
    """Return the first existing Step5 raster matching candidate filenames."""
    for name in candidates:
        path = STEP5_OUTPUT_DIR / name
        if path.exists():
            return path
    return None


def load_raster(key: str, path: Path) -> RasterLayer:
    """Load a single-band raster as float32 with nodata converted to NaN."""
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype("float32").filled(np.nan)
        crs = src.crs.to_string() if src.crs else None
        return RasterLayer(
            key=key,
            path=path,
            array=data,
            crs=crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
        )


def load_layers() -> tuple[dict[str, RasterLayer], dict[str, list[str]]]:
    """Load all available diagnostic rasters and record missing candidates."""
    layers: dict[str, RasterLayer] = {}
    missing: dict[str, list[str]] = {}

    for key, candidates in RASTER_CANDIDATES.items():
        path = find_existing_raster(candidates)
        if path is None:
            missing[key] = candidates
            log.warning("Raster bulunamadı: %s (%s)", key, ", ".join(candidates))
            continue

        log.info("Raster okunuyor: %s -> %s", key, path)
        layers[key] = load_raster(key, path)

    if "anomaly" not in layers:
        raise FileNotFoundError(
            "Anomaly raster bulunamadı. Beklenen adlardan biri: "
            + ", ".join(RASTER_CANDIDATES["anomaly"])
        )

    return layers, missing


def same_grid(reference: RasterLayer, other: RasterLayer) -> bool:
    """Check whether two raster layers share the same grid."""
    return (
        reference.width == other.width
        and reference.height == other.height
        and reference.transform == other.transform
        and reference.crs == other.crs
    )


def grid_report(layers: dict[str, RasterLayer]) -> dict[str, Any]:
    """Report grid compatibility against the anomaly raster."""
    reference = layers["anomaly"]
    report = {}

    for key, layer in layers.items():
        report[key] = {
            "path": str(layer.path),
            "width": layer.width,
            "height": layer.height,
            "crs": layer.crs,
            "same_grid_as_anomaly": same_grid(reference, layer),
        }

    return report


def finite_values(array: np.ndarray) -> np.ndarray:
    """Return finite values from an array as float64."""
    return array[np.isfinite(array)].astype("float64")


def raster_stats(layer: RasterLayer) -> dict[str, Any]:
    """Compute basic raster statistics."""
    values = finite_values(layer.array)
    total = int(layer.array.size)
    valid = int(values.size)
    nan_ratio = None if total == 0 else float(1 - valid / total)

    if valid == 0:
        return {
            "path": str(layer.path),
            "valid_pixel_count": 0,
            "nan_ratio": nan_ratio,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "path": str(layer.path),
        "valid_pixel_count": valid,
        "nan_ratio": nan_ratio,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def anomaly_histogram_stats(anomaly: np.ndarray) -> dict[str, Any]:
    """Compute histogram-oriented statistics for z-score anomaly."""
    values = finite_values(anomaly)
    if values.size == 0:
        return {
            "count": 0,
            "abs_z_gt_2_percent": None,
            "abs_z_gt_3_percent": None,
            "percentiles": {},
        }

    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "abs_z_gt_2_percent": float(100 * np.mean(abs_values > 2)),
        "abs_z_gt_3_percent": float(100 * np.mean(abs_values > 3)),
        "percentiles": {
            "p01": float(np.percentile(values, 1)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        },
    }


def mask_overlap(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute overlap ratios between two boolean masks."""
    valid_reference = np.asarray(reference_mask, dtype=bool)
    valid_candidate = np.asarray(candidate_mask, dtype=bool)
    intersection = valid_reference & valid_candidate
    reference_count = int(np.sum(valid_reference))
    candidate_count = int(np.sum(valid_candidate))
    intersection_count = int(np.sum(intersection))

    return {
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "intersection_count": intersection_count,
        "reference_overlap_percent": (
            None
            if reference_count == 0
            else float(100 * intersection_count / reference_count)
        ),
        "candidate_overlap_percent": (
            None
            if candidate_count == 0
            else float(100 * intersection_count / candidate_count)
        ),
    }


def comparable_layer(
    layers: dict[str, RasterLayer],
    key: str,
) -> RasterLayer | None:
    """Return layer only if it exists and shares anomaly grid."""
    layer = layers.get(key)
    if layer is None:
        return None
    if not same_grid(layers["anomaly"], layer):
        log.warning("Grid uyuşmuyor, karşılaştırma atlandı: %s", key)
        return None
    return layer


def low_confidence_masks(
    layers: dict[str, RasterLayer],
) -> dict[str, np.ndarray]:
    """Build low-confidence masks from available same-grid support rasters."""
    masks: dict[str, np.ndarray] = {}

    baseline_count = comparable_layer(layers, "baseline_valid_count")
    if baseline_count is not None:
        masks["low_baseline_valid_count"] = (
            np.isfinite(baseline_count.array)
            & (baseline_count.array < STEP5_MIN_BASELINE_VALID_COUNT)
        )

    baseline_std = comparable_layer(layers, "baseline_std")
    if baseline_std is not None:
        masks["low_baseline_std"] = (
            np.isfinite(baseline_std.array)
            & (baseline_std.array < STEP5_MIN_BASELINE_STD_CELSIUS)
        )

    current_count = comparable_layer(layers, "current_valid_count")
    if current_count is not None:
        masks["low_current_valid_count"] = (
            np.isfinite(current_count.array)
            & (current_count.array < STEP5_MIN_CURRENT_VALID_COUNT)
        )

    return masks


def gradient_strength(array: np.ndarray) -> np.ndarray:
    """Compute simple gradient magnitude without smoothing the source raster."""
    valid_array = np.where(np.isfinite(array), array, np.nan)
    gy, gx = np.gradient(valid_array.astype("float64"))
    strength = np.sqrt(gx * gx + gy * gy)
    return np.where(np.isfinite(strength), strength, np.nan)


def high_gradient_mask(array: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    """Return top-percentile gradient mask for seam candidate comparison."""
    strength = gradient_strength(array)
    values = finite_values(strength)
    values = values[values > 0]
    if values.size == 0:
        return np.zeros(array.shape, dtype=bool)

    threshold = float(np.percentile(values, percentile))
    if not math.isfinite(threshold) or threshold <= 0:
        return np.zeros(array.shape, dtype=bool)
    return np.isfinite(strength) & (strength > 0) & (strength >= threshold)


def local_change_mask(array: np.ndarray, min_delta: float = 1.0) -> np.ndarray:
    """
    Detect abrupt local value changes without smoothing.

    This is useful for discrete valid-count rasters where a path/row footprint
    boundary appears as an immediate count jump.
    """
    valid = np.isfinite(array)
    edge = np.zeros(array.shape, dtype=bool)

    horizontal = (
        valid[:, 1:]
        & valid[:, :-1]
        & (np.abs(array[:, 1:] - array[:, :-1]) >= min_delta)
    )
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal

    vertical = (
        valid[1:, :]
        & valid[:-1, :]
        & (np.abs(array[1:, :] - array[:-1, :]) >= min_delta)
    )
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical

    return edge


def count_edge_mask(array: np.ndarray) -> np.ndarray:
    """Build edge mask for valid-count rasters."""
    high_gradient = high_gradient_mask(array, percentile=98.0)
    local_jump = local_change_mask(array, min_delta=1.0)
    meaningful_gradient = high_gradient_mask(array, percentile=95.0)
    return local_jump & (high_gradient | meaningful_gradient)


def std_edge_mask(array: np.ndarray) -> np.ndarray:
    """Build edge mask for baseline std raster."""
    return high_gradient_mask(array, percentile=97.0)


def anomaly_edge_mask(array: np.ndarray) -> np.ndarray:
    """Build edge mask for anomaly raster."""
    return high_gradient_mask(array, percentile=98.0)


def seam_candidate_masks(layers: dict[str, RasterLayer]) -> dict[str, np.ndarray]:
    """
    Build seam candidate masks that do not treat all valid support pixels as evidence.

    Baseline low-count areas are reported separately in overlap_stats. Seam
    evidence masks here are deliberately narrow edge/local-jump candidates.
    """
    masks: dict[str, np.ndarray] = {}
    anomaly_layer = layers["anomaly"]
    masks["anomaly_edge"] = anomaly_edge_mask(anomaly_layer.array)

    current_count = comparable_layer(layers, "current_valid_count")
    if current_count is not None:
        masks["current_valid_count_edge"] = count_edge_mask(current_count.array)

    baseline_count = comparable_layer(layers, "baseline_valid_count")
    if baseline_count is not None:
        masks["baseline_valid_count_edge"] = count_edge_mask(baseline_count.array)

    baseline_std = comparable_layer(layers, "baseline_std")
    if baseline_std is not None:
        masks["baseline_std_edge"] = std_edge_mask(baseline_std.array)

    modis = comparable_layer(layers, "modis_context")
    if modis is not None:
        masks["modis_edge"] = anomaly_edge_mask(modis.array)

    modis_mean = comparable_layer(layers, "modis_mean")
    if modis_mean is not None:
        masks["modis_mean_edge"] = std_edge_mask(modis_mean.array)

    modis_std = comparable_layer(layers, "modis_std")
    if modis_std is not None:
        masks["modis_std_edge"] = std_edge_mask(modis_std.array)

    return masks


def seam_evidence_scores(
    seam_masks: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Score how strongly support-layer edges overlap anomaly edges."""
    anomaly_edge = seam_masks["anomaly_edge"]
    scores: dict[str, dict[str, Any]] = {
        "landsat_anomaly_edge_score": {
            "edge_pixel_count": int(np.sum(anomaly_edge)),
            "edge_density_percent": float(100 * np.mean(anomaly_edge)),
        }
    }

    for source_key, score_key in [
        ("current_valid_count_edge", "landsat_current_coverage_seam_score"),
        ("baseline_valid_count_edge", "baseline_valid_count_edge_score"),
        ("baseline_std_edge", "baseline_std_edge_score"),
        ("modis_edge", "modis_context_edge_score"),
        ("modis_mean_edge", "modis_mean_edge_score"),
        ("modis_std_edge", "modis_std_edge_score"),
    ]:
        source_mask = seam_masks.get(source_key)
        if source_mask is None:
            scores[score_key] = {
                "available": False,
                "excluded_from_source_classification": False,
                "anomaly_edge_overlap_percent": None,
                "candidate_edge_overlap_percent": None,
                "candidate_pixel_count": 0,
                "candidate_density_percent": None,
            }
            continue

        overlap = mask_overlap(anomaly_edge, source_mask)
        candidate_density = float(100 * np.mean(source_mask))
        is_degenerate = candidate_density > MAX_EDGE_MASK_DENSITY_PERCENT
        scores[score_key] = {
            "available": True,
            "excluded_from_source_classification": is_degenerate,
            "anomaly_edge_overlap_percent": overlap["reference_overlap_percent"],
            "candidate_edge_overlap_percent": overlap["candidate_overlap_percent"],
            "intersection_count": overlap["intersection_count"],
            "candidate_pixel_count": overlap["candidate_count"],
            "candidate_density_percent": candidate_density,
            "warning": (
                "edge mask appears degenerate; excluded from source classification"
                if is_degenerate
                else None
            ),
        }

    landsat_modis = None
    if "modis_edge" in seam_masks:
        landsat_modis = mask_overlap(anomaly_edge, seam_masks["modis_edge"])
    scores["landsat_modis_edge_agreement"] = {
        "available": landsat_modis is not None,
        "excluded_from_source_classification": (
            scores.get("modis_context_edge_score", {}).get(
                "excluded_from_source_classification",
                False,
            )
        ),
        "anomaly_edge_overlap_percent": (
            None if landsat_modis is None else landsat_modis["reference_overlap_percent"]
        ),
        "modis_edge_overlap_percent": (
            None if landsat_modis is None else landsat_modis["candidate_overlap_percent"]
        ),
        "intersection_count": 0 if landsat_modis is None else landsat_modis["intersection_count"],
    }

    return scores


def usable_score(
    scores: dict[str, dict[str, Any]],
    key: str,
) -> float | None:
    """Return anomaly-edge overlap only if the source edge mask is usable."""
    score = scores[key]
    if score.get("excluded_from_source_classification"):
        return None
    return score.get("anomaly_edge_overlap_percent")


def degenerate_warning(
    scores: dict[str, dict[str, Any]],
    key: str,
    label: str,
) -> str | None:
    """Build a source-classification warning for degenerate masks."""
    score = scores[key]
    if not score.get("excluded_from_source_classification"):
        return None
    density = score.get("candidate_density_percent")
    density_text = "n/a" if density is None else f"{density:.1f}%"
    return (
        f"{label} edge mask appears degenerate; excluded from source "
        f"classification (candidate density {density_text})."
    )


def evidence_level(score: float | None) -> str:
    """Classify edge-overlap evidence strength."""
    if score is None:
        return "not available"
    if score > EVIDENCE_STRONG_THRESHOLD:
        return "strong"
    if score >= EVIDENCE_MODERATE_THRESHOLD:
        return "moderate"
    if score >= EVIDENCE_WEAK_THRESHOLD:
        return "weak"
    return "not supported"


def evidence_sentence(label: str, score: float | None) -> str:
    """Format one edge-overlap evidence sentence for summary output."""
    level = evidence_level(score)
    if score is None:
        return f"{label}: not available or excluded from source classification."
    if level == "not supported":
        return f"{label}: not supported ({score:.2f}% anomaly-edge overlap)."
    return f"{label}: {level} evidence ({score:.2f}% anomaly-edge overlap)."


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Compute correlation/agreement if arrays share enough finite pixels."""
    valid = np.isfinite(a) & np.isfinite(b)
    count = int(np.sum(valid))
    if count < 10:
        return {
            "valid_pair_count": count,
            "pearson_correlation": None,
            "sign_agreement_percent": None,
            "extreme_overlap_percent": None,
        }

    av = a[valid].astype("float64")
    bv = b[valid].astype("float64")
    corr = float(np.corrcoef(av, bv)[0, 1])
    sign_agreement = float(100 * np.mean(np.sign(av) == np.sign(bv)))
    landsat_extreme = np.abs(a) > 2
    modis_extreme = np.abs(b) > 2
    overlap = mask_overlap(landsat_extreme & valid, modis_extreme & valid)

    return {
        "valid_pair_count": count,
        "pearson_correlation": corr if math.isfinite(corr) else None,
        "sign_agreement_percent": sign_agreement,
        "extreme_overlap_percent": overlap["reference_overlap_percent"],
    }


def seam_source_interpretation(
    layers: dict[str, RasterLayer],
    masks: dict[str, np.ndarray],
    overlap_stats: dict[str, Any],
    modis_agreement: dict[str, Any] | None,
    seam_masks: dict[str, np.ndarray],
    evidence_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generate a cautious rule-based seam/source interpretation."""
    seam_candidates = seam_masks["anomaly_edge"]
    evidence: dict[str, list[str]] = {
        "landsat_current_coverage_evidence": [],
        "landsat_baseline_coverage_std_evidence": [],
        "modis_contextual_evidence": [],
        "shared_grid_resampling_artefact_evidence": [],
        "real_broad_scale_thermal_topographic_pattern_evidence": [],
    }

    current_score = usable_score(
        evidence_scores,
        "landsat_current_coverage_seam_score",
    )
    baseline_count_score = usable_score(
        evidence_scores,
        "baseline_valid_count_edge_score",
    )
    baseline_std_score = usable_score(evidence_scores, "baseline_std_edge_score")
    modis_score = usable_score(evidence_scores, "modis_context_edge_score")
    modis_mean_score = usable_score(evidence_scores, "modis_mean_edge_score")
    modis_std_score = usable_score(evidence_scores, "modis_std_edge_score")
    landsat_modis_score = usable_score(
        evidence_scores,
        "landsat_modis_edge_agreement",
    )

    for key, label, bucket in [
        (
            "landsat_current_coverage_seam_score",
            "current_valid_count",
            "landsat_current_coverage_evidence",
        ),
        (
            "baseline_valid_count_edge_score",
            "baseline_valid_count",
            "landsat_baseline_coverage_std_evidence",
        ),
        (
            "baseline_std_edge_score",
            "baseline_std",
            "landsat_baseline_coverage_std_evidence",
        ),
        (
            "modis_context_edge_score",
            "modis_context",
            "modis_contextual_evidence",
        ),
        (
            "modis_mean_edge_score",
            "modis_mean",
            "real_broad_scale_thermal_topographic_pattern_evidence",
        ),
        (
            "modis_std_edge_score",
            "modis_std",
            "modis_contextual_evidence",
        ),
    ]:
        warning = degenerate_warning(evidence_scores, key, label)
        if warning is not None:
            evidence[bucket].append(warning)

    evidence["landsat_current_coverage_evidence"].append(
        evidence_sentence("current_valid_count edge", current_score)
    )
    if evidence_level(current_score) == "weak":
        evidence["landsat_current_coverage_evidence"].append(
            "Current coverage remains a weak candidate under current "
            "edge-overlap scoring."
        )

    evidence["landsat_baseline_coverage_std_evidence"].append(
        evidence_sentence("baseline_valid_count edge", baseline_count_score)
    )
    if evidence_level(baseline_count_score) == "not supported":
        evidence["landsat_baseline_coverage_std_evidence"].append(
            "Baseline valid-count evidence is not supported as a primary seam "
            "source by this diagnostic."
        )

    evidence["landsat_baseline_coverage_std_evidence"].append(
        evidence_sentence("baseline_std edge", baseline_std_score)
    )
    if evidence_level(baseline_std_score) == "moderate":
        evidence["landsat_baseline_coverage_std_evidence"].append(
            "Baseline std shows moderate evidence; z-score denominator "
            "structure may contribute."
        )

    for key, stats in overlap_stats.items():
        overlap = stats.get("abs_z_gt_2", {}).get("reference_overlap_percent")
        if overlap is not None and overlap >= 30:
            evidence["landsat_baseline_coverage_std_evidence"].append(
                f"Extreme anomaly alanları {key} maskesiyle güçlü çakışıyor "
                f"({overlap:.1f}%)."
            )

    evidence["modis_contextual_evidence"].append(
        evidence_sentence("MODIS context z-score edge", modis_score)
    )
    evidence["real_broad_scale_thermal_topographic_pattern_evidence"].append(
        evidence_sentence("MODIS mean edge", modis_mean_score)
    )
    evidence["modis_contextual_evidence"].append(
        evidence_sentence("MODIS std edge", modis_std_score)
    )

    if (
        modis_score is not None
        and evidence_level(modis_score) == "strong"
        and evidence_level(modis_mean_score) in {"weak", "not supported"}
        and evidence_level(modis_std_score) in {"weak", "not supported"}
    ):
        evidence["modis_contextual_evidence"].append(
            "MODIS context z-score has strong aligned edge evidence, while "
            "MODIS mean/std do not. This points to z-score/resampling/grid/"
            "denominator effects rather than a simple mean-temperature/"
            "topographic pattern."
        )

    if (
        landsat_modis_score is not None
        and landsat_modis_score >= EVIDENCE_WEAK_THRESHOLD
    ):
        evidence["shared_grid_resampling_artefact_evidence"].append(
            f"Landsat and MODIS edge masks overlap by {landsat_modis_score:.2f}% "
            "of Landsat anomaly edge pixels."
        )

    if modis_agreement and modis_agreement.get("pearson_correlation") is not None:
        corr = modis_agreement["pearson_correlation"]
        evidence["modis_contextual_evidence"].append(
            f"Landsat/MODIS z-score paired-pixel correlation is {corr:.3f}."
        )

    for items in evidence.values():
        if not items:
            items.append("No strong same-grid edge evidence by current thresholds.")

    return {
        "seam_candidate_pixel_count": int(np.sum(seam_candidates)),
        "seam_evidence_scores": evidence_scores,
        "classification": evidence,
        "interpretation": [
            "After removing temporal interpolation, Landsat baseline statistics "
            "are computed only from observed QA-clean pixels. Baseline valid-count "
            "does not support the seam as a primary source. Baseline std shows "
            "moderate evidence, MODIS context z-score shows strong aligned edge "
            "evidence, while MODIS mean/std are weak. Therefore the seam/pattern "
            "remains mixed-source and should be further checked for MODIS z-score "
            "construction, resampling/grid alignment, denominator effects, and "
            "tiling safety before being masked as artefact."
        ],
    }


def plot_map(
    array: np.ndarray,
    title: str,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    """Write a raster-like PNG map."""
    plt.figure(figsize=(8, 6))
    masked = np.ma.masked_invalid(array)
    image = plt.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(image, shrink=0.8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_histogram(array: np.ndarray, output_path: Path) -> None:
    """Write anomaly histogram PNG."""
    values = finite_values(array)
    plt.figure(figsize=(8, 5))
    if values.size:
        clipped = values[np.abs(values) <= 10]
        plt.hist(clipped, bins=80, color="#3B82F6", edgecolor="white")
    plt.axvline(-3, color="#B91C1C", linestyle="--", linewidth=1)
    plt.axvline(-2, color="#F97316", linestyle="--", linewidth=1)
    plt.axvline(2, color="#F97316", linestyle="--", linewidth=1)
    plt.axvline(3, color="#B91C1C", linestyle="--", linewidth=1)
    plt.title("Anomaly z-score histogram")
    plt.xlabel("z-score")
    plt.ylabel("Pixel count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_extreme_overlay(anomaly: np.ndarray, output_path: Path) -> None:
    """Write an extreme anomaly overlay without changing source values."""
    base = np.clip(anomaly, -3, 3)
    extreme = np.abs(anomaly) > 3

    plt.figure(figsize=(8, 6))
    plt.imshow(np.ma.masked_invalid(base), cmap="coolwarm", vmin=-3, vmax=3)
    overlay = np.ma.masked_where(~extreme, extreme.astype("float32"))
    plt.imshow(overlay, cmap="gray", alpha=0.35)
    plt.colorbar(shrink=0.8, label="z-score")
    plt.title("Extreme anomaly overlay (|z| > 3)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_mask(mask: np.ndarray, title: str, output_path: Path) -> None:
    """Write a binary mask PNG."""
    plt.figure(figsize=(8, 6))
    plt.imshow(mask.astype("float32"), cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_seam_evidence_overlay(
    anomaly: np.ndarray,
    seam_masks: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Overlay support-layer seam evidence on anomaly map."""
    base = np.clip(anomaly, -3, 3)
    rgb = np.zeros((*base.shape, 4), dtype="float32")

    if "current_valid_count_edge" in seam_masks:
        rgb[seam_masks["current_valid_count_edge"]] = [0.0, 1.0, 1.0, 0.65]
    if "baseline_valid_count_edge" in seam_masks:
        rgb[seam_masks["baseline_valid_count_edge"]] = [1.0, 1.0, 0.0, 0.55]
    if "baseline_std_edge" in seam_masks:
        rgb[seam_masks["baseline_std_edge"]] = [1.0, 0.0, 1.0, 0.55]
    rgb[seam_masks["anomaly_edge"]] = [0.0, 0.0, 0.0, 0.75]

    plt.figure(figsize=(8, 6))
    plt.imshow(np.ma.masked_invalid(base), cmap="coolwarm", vmin=-3, vmax=3)
    plt.imshow(rgb)
    plt.colorbar(shrink=0.8, label="z-score")
    plt.title("Seam evidence overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_landsat_modis_edge_agreement(
    anomaly: np.ndarray,
    seam_masks: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Overlay Landsat anomaly/current edges with MODIS context edges."""
    base = np.clip(anomaly, -3, 3)
    rgb = np.zeros((*base.shape, 4), dtype="float32")
    anomaly_edge = seam_masks["anomaly_edge"]
    current_edge = seam_masks.get("current_valid_count_edge")
    modis_edge = seam_masks.get("modis_edge")

    if current_edge is not None:
        rgb[current_edge] = [0.0, 1.0, 1.0, 0.55]
    if modis_edge is not None:
        rgb[modis_edge] = [1.0, 1.0, 0.0, 0.55]
    rgb[anomaly_edge] = [0.0, 0.0, 0.0, 0.70]

    if modis_edge is not None:
        agreement = anomaly_edge & modis_edge
        rgb[agreement] = [1.0, 0.0, 0.0, 0.85]

    plt.figure(figsize=(8, 6))
    plt.imshow(np.ma.masked_invalid(base), cmap="coolwarm", vmin=-3, vmax=3)
    plt.imshow(rgb)
    plt.colorbar(shrink=0.8, label="z-score")
    plt.title("Landsat/MODIS edge agreement overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _array_stats(array: np.ndarray, path: Path) -> dict[str, Any]:
    """min/max/mean/std/nan_ratio for a raw array (no value modification)."""
    values = finite_values(array)
    total = int(array.size)
    valid = int(values.size)
    nan_ratio = None if total == 0 else float(1 - valid / total)
    if valid == 0:
        return {
            "path": str(path),
            "valid_pixel_count": 0,
            "nan_ratio": nan_ratio,
            "min": None, "max": None, "mean": None, "std": None,
        }
    return {
        "path": str(path),
        "valid_pixel_count": valid,
        "nan_ratio": nan_ratio,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def compute_tvdi_stats(
    lst_anomaly: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Step5C TVDI rasterlarının numeric istatistiklerini üretir.

    Rasterları yalnız OKUR; hiçbir değeri değiştirmez, smoothing/blur uygulamaz.
    Üretilen alanlar her raster için: min/max/mean/std/nan_ratio.
    Ek olarak tvdi z-score için |z|>2 ve |z|>3 oranları ile (mümkünse) LST anomaly
    ile TVDI anomaly arasındaki korelasyon hesaplanır.
    """
    result: dict[str, Any] = {"available": False, "rasters": {}}

    tvdi_files = {
        "current_tvdi": STEP5C_OUTPUT_DIR / "current_tvdi.tif",
        "baseline_tvdi_mean": STEP5C_OUTPUT_DIR / "baseline_tvdi_mean.tif",
        "baseline_tvdi_std": STEP5C_OUTPUT_DIR / "baseline_tvdi_std.tif",
        "tvdi_difference": STEP5C_OUTPUT_DIR / "tvdi_difference.tif",
        "tvdi_anomaly_zscore": STEP5C_OUTPUT_DIR / "tvdi_anomaly_zscore.tif",
    }

    arrays: dict[str, np.ndarray] = {}
    for key, path in tvdi_files.items():
        if not path.exists():
            result["rasters"][key] = {"path": str(path), "status": "missing"}
            continue
        with rasterio.open(path) as src:
            array = src.read(1, masked=True).astype("float32").filled(np.nan)
        arrays[key] = array
        result["rasters"][key] = _array_stats(array, path)

    if arrays:
        result["available"] = True

    # Düşük baseline-std nedeniyle maskelenen piksel bilgisi Step5C metadata'sından
    # okunur (Step5C bu sayımı üretim sırasında tutar).
    metadata_path = STEP5C_OUTPUT_DIR / "step5c_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                step5c_meta = json.load(f)
            proc = step5c_meta.get("processing", {})
            result["low_std_masking"] = {
                "min_tvdi_baseline_std": proc.get("min_tvdi_baseline_std"),
                "low_tvdi_std_masked_pixel_count": proc.get(
                    "low_tvdi_std_masked_pixel_count"
                ),
                "low_tvdi_std_masked_ratio": proc.get("low_tvdi_std_masked_ratio"),
                "tvdi_zscore_candidate_pixel_count": proc.get(
                    "tvdi_zscore_candidate_pixel_count"
                ),
            }
        except (json.JSONDecodeError, OSError):
            result["low_std_masking"] = {"status": "metadata_unreadable"}

    # TVDI z-score eşik oranları
    zscore = arrays.get("tvdi_anomaly_zscore")
    if zscore is not None:
        finite = np.isfinite(zscore)
        valid = int(np.sum(finite))
        if valid > 0:
            abs_z = np.abs(zscore[finite])
            result["tvdi_zscore_thresholds"] = {
                "abs_z_gt_2_ratio": float(np.sum(abs_z > 2) / valid),
                "abs_z_gt_3_ratio": float(np.sum(abs_z > 3) / valid),
                "valid_pixel_count": valid,
            }
        else:
            result["tvdi_zscore_thresholds"] = {
                "abs_z_gt_2_ratio": None,
                "abs_z_gt_3_ratio": None,
                "valid_pixel_count": 0,
            }

    # LST anomaly ile TVDI anomaly korelasyonu (aynı grid varsayımıyla)
    if lst_anomaly is not None and zscore is not None:
        if lst_anomaly.shape == zscore.shape:
            result["lst_vs_tvdi_anomaly_correlation"] = safe_corrcoef(
                lst_anomaly, zscore
            )
        else:
            result["lst_vs_tvdi_anomaly_correlation"] = {
                "correlation": None,
                "note": "grid_shape_mismatch",
            }

    return result


def write_tvdi_png_outputs() -> list[str]:
    """
    Step5C'nin ürettiği TVDI rasterlarını PNG'ye çevirir.

    Bu blok LST diagnostic akışından bağımsızdır; outputs/step5c altındaki TVDI
    GeoTIFF'lerini okuyup DIAGNOSTIC_DIR'e PNG yazar. Dosya yoksa sessizce atlanır,
    böylece TVDI henüz üretilmemişse LST diagnostic çıktıları etkilenmez.

    TVDI haritaları (current/mean) 0-1 aralığında, z-score haritası -3/+3 aralığında
    sabit ölçekle çizilir.
    """
    outputs: list[str] = []

    tvdi_maps = [
        ("current_tvdi.tif", "Current TVDI (0-1)", "current_tvdi_map.png",
         0.0, 1.0, "YlOrBr"),
        ("baseline_tvdi_mean.tif", "Baseline TVDI mean (0-1)",
         "baseline_tvdi_mean_map.png", 0.0, 1.0, "YlOrBr"),
        ("baseline_tvdi_std.tif", "Baseline TVDI std", "baseline_tvdi_std_map.png",
         None, None, "magma"),
        ("tvdi_difference.tif", "TVDI difference (current - baseline mean)",
         "tvdi_difference_map.png", -0.5, 0.5, "coolwarm"),
        ("tvdi_anomaly_zscore.tif", "TVDI anomaly z-score (-3/+3)",
         "tvdi_anomaly_zscore_map.png", -3.0, 3.0, "coolwarm"),
    ]

    for filename, title, png_name, vmin, vmax, cmap in tvdi_maps:
        tif_path = STEP5C_OUTPUT_DIR / filename
        if not tif_path.exists():
            log.info("TVDI rasterı bulunamadı, PNG atlandı: %s", tif_path.name)
            continue

        with rasterio.open(tif_path) as src:
            array = src.read(1, masked=True).astype("float32").filled(np.nan)

        plot_map(
            array,
            title,
            DIAGNOSTIC_DIR / png_name,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        outputs.append(png_name)
        log.info("TVDI PNG yazıldı: %s", png_name)

    # TVDI z-score histogramı (varsa)
    zscore_path = STEP5C_OUTPUT_DIR / "tvdi_anomaly_zscore.tif"
    if zscore_path.exists():
        with rasterio.open(zscore_path) as src:
            zscore = src.read(1, masked=True).astype("float32").filled(np.nan)
        plot_histogram(zscore, DIAGNOSTIC_DIR / "tvdi_anomaly_histogram.png")
        outputs.append("tvdi_anomaly_histogram.png")
        log.info("TVDI PNG yazıldı: tvdi_anomaly_histogram.png")

    return outputs


def write_png_outputs(
    layers: dict[str, RasterLayer],
    seam_masks: dict[str, np.ndarray],
) -> list[str]:
    """Create requested PNG diagnostic figures."""
    outputs: list[str] = []

    anomaly = layers["anomaly"].array
    plot_map(
        anomaly,
        "Landsat anomaly z-score (-3/+3)",
        DIAGNOSTIC_DIR / "anomaly_zscore_map.png",
        vmin=-3,
        vmax=3,
        cmap="coolwarm",
    )
    outputs.append("anomaly_zscore_map.png")

    for key, title, cmap in [
        ("baseline_valid_count", "Baseline valid count", "viridis"),
        ("baseline_std", "Baseline std Celsius", "magma"),
        ("current_valid_count", "Current valid count", "viridis"),
    ]:
        layer = layers.get(key)
        if layer is None:
            continue
        filename = f"{key}_map.png"
        plot_map(layer.array, title, DIAGNOSTIC_DIR / filename, cmap=cmap)
        outputs.append(filename)

    modis = layers.get("modis_context")
    if modis is not None:
        plot_map(
            modis.array,
            "MODIS context z-score (-3/+3)",
            DIAGNOSTIC_DIR / "modis_context_zscore_map.png",
            vmin=-3,
            vmax=3,
            cmap="coolwarm",
        )
        outputs.append("modis_context_zscore_map.png")

    modis_mean = layers.get("modis_mean")
    if modis_mean is not None:
        plot_map(
            modis_mean.array,
            "MODIS mean Celsius",
            DIAGNOSTIC_DIR / "modis_mean_map.png",
            cmap="viridis",
        )
        outputs.append("modis_mean_map.png")

    modis_std = layers.get("modis_std")
    if modis_std is not None:
        plot_map(
            modis_std.array,
            "MODIS std Celsius",
            DIAGNOSTIC_DIR / "modis_std_map.png",
            cmap="magma",
        )
        outputs.append("modis_std_map.png")

    plot_histogram(anomaly, DIAGNOSTIC_DIR / "anomaly_histogram.png")
    outputs.append("anomaly_histogram.png")

    plot_extreme_overlay(anomaly, DIAGNOSTIC_DIR / "extreme_anomaly_overlay.png")
    outputs.append("extreme_anomaly_overlay.png")

    mask_outputs = [
        ("anomaly_edge", "Anomaly edge mask", "anomaly_edge_mask.png"),
        (
            "current_valid_count_edge",
            "Current valid count edge mask",
            "current_valid_count_edge_mask.png",
        ),
        (
            "baseline_valid_count_edge",
            "Baseline valid count edge mask",
            "baseline_valid_count_edge_mask.png",
        ),
        (
            "modis_edge",
            "MODIS context edge mask",
            "modis_context_edge_mask.png",
        ),
        (
            "modis_mean_edge",
            "MODIS mean edge mask",
            "modis_mean_edge_mask.png",
        ),
        (
            "modis_std_edge",
            "MODIS std edge mask",
            "modis_std_edge_mask.png",
        ),
    ]
    for key, title, filename in mask_outputs:
        mask = seam_masks.get(key)
        if mask is None:
            continue
        plot_mask(mask, title, DIAGNOSTIC_DIR / filename)
        outputs.append(filename)

    plot_seam_evidence_overlay(
        anomaly,
        seam_masks,
        DIAGNOSTIC_DIR / "seam_evidence_overlay.png",
    )
    outputs.append("seam_evidence_overlay.png")

    if "modis_edge" in seam_masks:
        plot_landsat_modis_edge_agreement(
            anomaly,
            seam_masks,
            DIAGNOSTIC_DIR / "landsat_modis_edge_agreement_overlay.png",
        )
        outputs.append("landsat_modis_edge_agreement_overlay.png")

    return outputs


def build_report() -> dict[str, Any]:
    """Build complete diagnostics report data."""
    layers, missing = load_layers()
    anomaly = layers["anomaly"].array
    extreme_masks = {
        "abs_z_gt_2": np.isfinite(anomaly) & (np.abs(anomaly) > 2),
        "abs_z_gt_3": np.isfinite(anomaly) & (np.abs(anomaly) > 3),
    }
    masks = low_confidence_masks(layers)

    layer_stats = {key: raster_stats(layer) for key, layer in layers.items()}
    histogram_stats = anomaly_histogram_stats(anomaly)
    overlap_stats: dict[str, Any] = {}

    for mask_key, mask in masks.items():
        overlap_stats[mask_key] = {
            extreme_key: mask_overlap(extreme_mask, mask)
            for extreme_key, extreme_mask in extreme_masks.items()
        }

    modis_agreement = None
    modis = comparable_layer(layers, "modis_context")
    if modis is not None:
        modis_agreement = safe_corrcoef(anomaly, modis.array)

    seam_masks = seam_candidate_masks(layers)
    evidence_scores = seam_evidence_scores(seam_masks)
    seam_report = seam_source_interpretation(
        layers=layers,
        masks=masks,
        overlap_stats=overlap_stats,
        modis_agreement=modis_agreement,
        seam_masks=seam_masks,
        evidence_scores=evidence_scores,
    )
    png_outputs = write_png_outputs(layers, seam_masks)

    # TVDI rasterlarını (Step5C çıktısı) da PNG'ye çevir. Step5C, Step5B'den önce
    # çalıştığı için bu noktada TVDI GeoTIFF'leri hazır olmalı; değilse atlanır.
    tvdi_png_outputs = write_tvdi_png_outputs()
    png_outputs = png_outputs + tvdi_png_outputs

    # TVDI numeric istatistikleri (Step5C çıktıları). Rasterları yalnız okur.
    tvdi_stats = compute_tvdi_stats(lst_anomaly=anomaly)

    return {
        "created_at": datetime.now().isoformat(),
        "step5_output_dir": str(STEP5_OUTPUT_DIR),
        "step5c_output_dir": str(STEP5C_OUTPUT_DIR),
        "diagnostic_dir": str(DIAGNOSTIC_DIR),
        "log_file": str(log_file),
        "missing_rasters": missing,
        "grid_report": grid_report(layers),
        "raster_stats": layer_stats,
        "anomaly_histogram": histogram_stats,
        "overlap_stats": overlap_stats,
        "modis_agreement": modis_agreement,
        "seam_evidence_scores": evidence_scores,
        "possible_seam_source": seam_report,
        "png_outputs": png_outputs,
        "tvdi_png_outputs": tvdi_png_outputs,
        "tvdi_stats": tvdi_stats,
    }


def pct(value: float | None) -> str:
    """Format optional percentage values."""
    return "n/a" if value is None else f"{value:.2f}%"


def scalar(value: Any) -> str:
    """Format optional scalar values for markdown."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary_markdown(report: dict[str, Any]) -> Path:
    """Write human-readable diagnostics summary."""
    seam_source = report["possible_seam_source"]
    classification = seam_source["classification"]
    lines = [
        "# Step5 Diagnostic Summary",
        "",
        f"Created at: `{report['created_at']}`",
        f"Step5 output dir: `{report['step5_output_dir']}`",
        "",
        "## Raster Stats",
        "",
        "| Raster | Min | Max | Mean | Std | NaN ratio | Valid pixels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, stats in report["raster_stats"].items():
        lines.append(
            "| {key} | {min} | {max} | {mean} | {std} | {nan} | {valid} |".format(
                key=key,
                min=scalar(stats["min"]),
                max=scalar(stats["max"]),
                mean=scalar(stats["mean"]),
                std=scalar(stats["std"]),
                nan=pct(
                    None
                    if stats["nan_ratio"] is None
                    else 100 * stats["nan_ratio"]
                ),
                valid=stats["valid_pixel_count"],
            )
        )

    hist = report["anomaly_histogram"]
    lines.extend(
        [
            "",
            "## Anomaly Histogram",
            "",
            f"- Valid anomaly pixels: `{hist['count']}`",
            f"- `|z| > 2`: {pct(hist['abs_z_gt_2_percent'])}",
            f"- `|z| > 3`: {pct(hist['abs_z_gt_3_percent'])}",
            f"- Percentiles: `{json.dumps(hist['percentiles'], ensure_ascii=False)}`",
            "",
            "## Extreme Anomaly Overlaps",
            "",
        ]
    )

    if report["overlap_stats"]:
        for mask_key, mask_stats in report["overlap_stats"].items():
            lines.append(f"### {mask_key}")
            for extreme_key, stats in mask_stats.items():
                lines.append(
                    "- {extreme}: {overlap} of extreme pixels overlap "
                    "({intersection}/{reference})".format(
                        extreme=extreme_key,
                        overlap=pct(stats["reference_overlap_percent"]),
                        intersection=stats["intersection_count"],
                        reference=stats["reference_count"],
                    )
                )
            lines.append("")
    else:
        lines.append("No same-grid support masks were available for overlap checks.")
        lines.append("")

    modis = report["modis_agreement"]
    lines.append("## MODIS Context Agreement")
    if modis is None:
        lines.append("")
        lines.append("MODIS context raster missing or not on anomaly grid; skipped.")
    else:
        lines.extend(
            [
                "",
                f"- Valid paired pixels: `{modis['valid_pair_count']}`",
                f"- Pearson correlation: `{scalar(modis['pearson_correlation'])}`",
                f"- Sign agreement: {pct(modis['sign_agreement_percent'])}",
                f"- Extreme overlap: {pct(modis['extreme_overlap_percent'])}",
            ]
        )

    lines.extend(["", "## Possible Seam Source", ""])
    for title, key in [
        ("Landsat current coverage evidence", "landsat_current_coverage_evidence"),
        (
            "Landsat baseline coverage/std evidence",
            "landsat_baseline_coverage_std_evidence",
        ),
        ("MODIS contextual evidence", "modis_contextual_evidence"),
        (
            "Shared grid/resampling artefact evidence",
            "shared_grid_resampling_artefact_evidence",
        ),
        (
            "Real broad-scale thermal/topographic pattern evidence",
            "real_broad_scale_thermal_topographic_pattern_evidence",
        ),
    ]:
        lines.append(f"### {title}")
        for item in classification[key]:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("### Cautious interpretation")
    for item in seam_source["interpretation"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Seam Evidence Scores", ""])
    for key, score in report["seam_evidence_scores"].items():
        if "edge_pixel_count" in score:
            lines.append(
                "- `{key}`: edge pixels={count}, density={density}".format(
                    key=key,
                    count=score["edge_pixel_count"],
                    density=pct(score["edge_density_percent"]),
                )
            )
            continue

        lines.append(
            "- `{key}`: available=`{available}`, anomaly-edge overlap={overlap}, "
            "candidate/modis overlap={candidate_overlap}, "
            "candidate pixels={candidate}, density={density}, excluded=`{excluded}`".format(
                key=key,
                available=score.get("available"),
                overlap=pct(score.get("anomaly_edge_overlap_percent")),
                candidate_overlap=pct(
                    score.get(
                        "candidate_edge_overlap_percent",
                        score.get("modis_edge_overlap_percent"),
                    )
                ),
                candidate=score.get("candidate_pixel_count", "n/a"),
                density=pct(score.get("candidate_density_percent")),
                excluded=score.get("excluded_from_source_classification"),
            )
        )
        if score.get("warning"):
            lines.append(f"  - Warning: {score['warning']}")

    lines.extend(["", "## Grid Compatibility", ""])
    for key, info in report["grid_report"].items():
        lines.append(
            f"- `{key}`: same grid as anomaly = "
            f"`{info['same_grid_as_anomaly']}`, size={info['width']}x{info['height']}, "
            f"crs=`{info['crs']}`"
        )

    # --- TVDI (Step5C) numeric stats ---
    tvdi = report.get("tvdi_stats")
    lines.extend(["", "## TVDI / Dryness Stats (Step5C)", ""])
    if not tvdi or not tvdi.get("available"):
        lines.append(
            "Step5C TVDI rasters not found; TVDI stats skipped. "
            "(Run Step5C before Step5B to populate these.)"
        )
    else:
        lines.append(
            "| Raster | Min | Max | Mean | Std | NaN ratio | Valid pixels |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key in [
            "current_tvdi",
            "baseline_tvdi_mean",
            "baseline_tvdi_std",
            "tvdi_difference",
            "tvdi_anomaly_zscore",
        ]:
            stats = tvdi["rasters"].get(key)
            if stats is None or stats.get("status") == "missing":
                lines.append(f"| {key} | n/a | n/a | n/a | n/a | n/a | missing |")
                continue
            lines.append(
                "| {key} | {min} | {max} | {mean} | {std} | {nan} | {valid} |".format(
                    key=key,
                    min=scalar(stats["min"]),
                    max=scalar(stats["max"]),
                    mean=scalar(stats["mean"]),
                    std=scalar(stats["std"]),
                    nan=pct(
                        None if stats["nan_ratio"] is None
                        else 100 * stats["nan_ratio"]
                    ),
                    valid=stats["valid_pixel_count"],
                )
            )

        # Düşük baseline-std maskeleme raporu
        masking = tvdi.get("low_std_masking")
        if masking and "low_tvdi_std_masked_pixel_count" in masking:
            ratio = masking.get("low_tvdi_std_masked_ratio")
            lines.extend([
                "",
                "### Low baseline-std masking (z-score reliability)",
                "",
                f"- Min baseline TVDI std threshold: "
                f"`{scalar(masking.get('min_tvdi_baseline_std'))}`",
                f"- Pixels masked due to low baseline std: "
                f"`{masking.get('low_tvdi_std_masked_pixel_count')}`",
                f"- z-score candidate pixels (before low-std mask): "
                f"`{masking.get('tvdi_zscore_candidate_pixel_count')}`",
                f"- Masked ratio: "
                f"{pct(None if ratio is None else 100 * ratio)}",
            ])

        thresholds = tvdi.get("tvdi_zscore_thresholds")
        gt3_ratio = None
        if thresholds:
            gt3_ratio = thresholds.get("abs_z_gt_3_ratio")
            lines.extend([
                "",
                "### TVDI anomaly z-score thresholds (after low-std masking)",
                "",
                f"- `|tvdi_z| > 2`: {pct(None if thresholds['abs_z_gt_2_ratio'] is None else 100 * thresholds['abs_z_gt_2_ratio'])}",
                f"- `|tvdi_z| > 3`: {pct(None if thresholds['abs_z_gt_3_ratio'] is None else 100 * thresholds['abs_z_gt_3_ratio'])}",
                f"- Valid pixels: `{thresholds['valid_pixel_count']}`",
            ])

        corr = tvdi.get("lst_vs_tvdi_anomaly_correlation")
        if corr is not None:
            lines.extend([
                "",
                "### LST anomaly vs TVDI anomaly",
                "",
            ])
            if corr.get("note") == "grid_shape_mismatch":
                lines.append(
                    "- Correlation skipped: LST anomaly and TVDI anomaly grids differ."
                )
            else:
                lines.append(
                    f"- Pearson correlation: `{scalar(corr.get('pearson_correlation'))}`"
                )
                if "valid_pair_count" in corr:
                    lines.append(
                        f"- Valid paired pixels: `{corr['valid_pair_count']}`"
                    )

        # İnstabilite uyarısı: |tvdi_z| > 3 oranı %10'un üzerindeyse.
        if gt3_ratio is not None and gt3_ratio > 0.10:
            lines.extend([
                "",
                "> **Warning:** TVDI z-score appears unstable; baseline TVDI std "
                "may be too low or baseline sample size may be insufficient. "
                f"(`|tvdi_z| > 3` = {100 * gt3_ratio:.2f}% after low-std masking.)",
            ])

        lines.append("")
        lines.append(
            "_Note: TVDI is a candidate dryness indicator / prototype, not a "
            "validated fire-risk product. `tvdi_difference.tif` "
            "(current_tvdi - baseline_tvdi_mean) is provided as a more "
            "interpretable companion to the z-score. Validation against "
            "burned-area and active-fire datasets is the next step._"
        )

    lines.extend(["", "## PNG Outputs", ""])
    for filename in report["png_outputs"]:
        lines.append(f"- `{filename}`")

    summary_path = DIAGNOSTIC_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    """CLI entry point."""
    log.info("=" * 60)
    log.info("STEP 5B DIAGNOSTIC REPORT BAŞLIYOR")
    log.info("=" * 60)

    report = build_report()
    stats_path = DIAGNOSTIC_DIR / "diagnostic_stats.json"
    stats_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path = write_summary_markdown(report)

    log.info("Diagnostic JSON yazıldı: %s", stats_path)
    log.info("Summary markdown yazıldı: %s", summary_path)
    log.info("PNG çıktıları: %s", ", ".join(report["png_outputs"]))
    log.info("=" * 60)
    log.info("STEP 5B DIAGNOSTIC REPORT TAMAMLANDI")


if __name__ == "__main__":
    main()