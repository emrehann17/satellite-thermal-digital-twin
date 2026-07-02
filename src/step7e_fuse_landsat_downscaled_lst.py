"""
step7e_fuse_landsat_downscaled_lst.py

Gozlemlenen Landsat current-period LST ile Step7D downscaled (Step7C model
tahmini) LST'yi BIRLESTIRIR (fusion / gap-filling): gozlemlenen Landsat
pikselleri HER ZAMAN korunur; downscaled LST YALNIZCA gozlem eksik/gecersiz
oldugunda "delik doldurma" (gap-fill) olarak kullanilir.

ONEMLI:
    - Step7E bir raster FUSION / GAP-FILLING adimidir.
    - Model EGITILMEZ.
    - Fire-risk modeli DEGILDIR.
    - MCD64A1/FIRMS/yanmis alan etiketi KULLANILMAZ.
    - BAGIMSIZ model dogrulamasi DEGILDIR.
    - Gozlemlenen ve tahmin edilen degerler ORTALAMA/BLEND EDILMEZ.
    - Gecerli gozlemlenen Landsat pikselleri model tahminleriyle DEGISTIRILMEZ.

Girdi:
    outputs/step5/current_period_median_celsius.tif (gozlemlenen, referans grid)
    outputs/step7d/downscaled_lst_celsius.tif
    outputs/step7d/downscaled_lst_valid_mask.tif
    outputs/step7d/downscaling_prediction_metadata.json (opsiyonel, bilgi amacli)

Ciktilar:
    outputs/step7e/fused_lst_celsius.tif
    outputs/step7e/fused_lst_source_mask.tif        (0=invalid,1=observed,2=gap-fill)
    outputs/step7e/fused_lst_gapfill_amount.tif      (yalniz source_mask==2'de deger, NaN elsewhere)
    outputs/step7e/fused_lst_metadata.json
    outputs/step7e/fused_lst_stats.json
    outputs/step7e/fused_lst_summary.md
    outputs/step7e/landsat_observed_valid_mask.tif           (--diagnostics)
    outputs/step7e/fused_minus_observed_on_overlap.tif       (--diagnostics)
    outputs/step7e/fused_lst_histogram.png                   (--plot)
    outputs/step7e/source_mask_map.png                       (--plot)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import rasterio

from core.config import (
    STEP7E_OUTPUT_DIR,
    STEP7E_TILE_SIZE,
    STEP7E_MIN_CELSIUS,
    STEP7E_MAX_CELSIUS,
    STEP7E_OBSERVED_LST_PATH,
    STEP7E_DOWNSCALED_LST_PATH,
    STEP7E_DOWNSCALED_VALID_MASK_PATH,
    STEP7E_WRITE_DIAGNOSTICS,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.utils.tiling import iter_windows

BASE_DIR = PROJECT_ROOT

log, log_file = setup_logger("step7e")

SOURCE_INVALID = 0
SOURCE_OBSERVED = 1
SOURCE_GAPFILL = 2

SOURCE_MASK_CODES = {
    "0": "invalid / no data",
    "1": "observed Landsat used",
    "2": "downscaled Step7D used as gap-fill",
}


def resolve_observed_path(explicit: str | None) -> Path:
    """Gozlemlenen Landsat current-period LST rasterini (referans grid) cozer."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise SystemExit(f"Belirtilen gozlemlenen LST rasteri bulunamadi: {p}")

    candidates = [
        BASE_DIR / STEP7E_OBSERVED_LST_PATH,
        BASE_DIR / "data" / "current_period" / "landsat_current_period_60days.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    cp_dir = BASE_DIR / "data" / "current_period"
    if cp_dir.exists():
        for p in sorted(cp_dir.glob("landsat_current_period_*days.tif")):
            if "(" in p.name:
                continue
            return p
    raise SystemExit(
        "Gozlemlenen Landsat current-period LST rasteri bulunamadi. Beklenen: "
        f"{STEP7E_OBSERVED_LST_PATH} veya "
        "data/current_period/landsat_current_period_*days.tif"
    )


def _grid_profile(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "width": src.width, "height": src.height,
            "crs": src.crs, "transform": src.transform,
        }


def validate_grid_alignment(reference_path: Path, other_paths: dict[str, Path]) -> dict:
    """
    Tum rasterlarin (downscaled LST, valid mask) referans gridle (gozlemlenen
    Landsat) birebir eslestigini dogrular. Uyusmazlikta SESSIZCE RESAMPLE
    ETMEZ; net hata ile durur (Step7E hicbir rasteri resample etmez).
    """
    ref = _grid_profile(reference_path)
    mismatches = []
    for name, path in other_paths.items():
        prof = _grid_profile(path)
        if (
            prof["width"] != ref["width"]
            or prof["height"] != ref["height"]
            or prof["crs"] != ref["crs"]
            or prof["transform"] != ref["transform"]
        ):
            mismatches.append(
                f"{name} ({path}): {prof['width']}x{prof['height']} {prof['crs']} "
                f"vs reference {ref['width']}x{ref['height']} {ref['crs']}"
            )
    if mismatches:
        raise SystemExit(
            "Raster(lar) referans (gozlemlenen Landsat) grid ile eslesmiyor "
            "(Step7E sessizce resample ETMEZ):\n  - " + "\n  - ".join(mismatches)
        )
    log.info("Tum rasterlar (downscaled LST, valid mask) referans gridle hizali dogrulandi.")
    return ref


def load_step7d_context() -> dict:
    """Step7D metadata/stats'tan (varsa) bilgilendirici baglam okur; kritik degildir."""
    context: dict = {"metadata": None, "stats": None, "spatial_calibration_note": None}
    meta_path = BASE_DIR / "outputs" / "step7d" / "downscaling_prediction_metadata.json"
    stats_path = BASE_DIR / "outputs" / "step7d" / "downscaling_prediction_stats.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            context["metadata"] = meta
            context["spatial_calibration_note"] = meta.get("spatial_calibration_note")
        except (OSError, json.JSONDecodeError):
            pass
    if stats_path.exists():
        try:
            context["stats"] = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return context


def run_fusion(
    observed_path: Path,
    downscaled_path: Path,
    downscaled_mask_path: Path,
    output_dir: Path,
    tile_size: int,
    write_diagnostics: bool,
) -> dict:
    """Tum raster gridini pencere pencere gezerek gozlem-oncelikli fuzyonu uretir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fused_path = output_dir / "fused_lst_celsius.tif"
    source_mask_path = output_dir / "fused_lst_source_mask.tif"
    gapfill_path = output_dir / "fused_lst_gapfill_amount.tif"
    obs_valid_mask_path = output_dir / "landsat_observed_valid_mask.tif"
    overlap_diff_path = output_dir / "fused_minus_observed_on_overlap.tif"

    counters = {
        "window_count": 0,
        "total_pixels": 0,
        "observed_valid_pixels": 0,
        "downscaled_valid_pixels": 0,
        "fused_valid_pixels": 0,
        "invalid_pixels": 0,
        "observed_used_pixels": 0,
        "gapfilled_pixels": 0,
        "out_of_range_observed_count": 0,
        "out_of_range_downscaled_count": 0,
    }

    fused_sum = 0.0
    fused_sum_sq = 0.0
    fused_min, fused_max = math.inf, -math.inf
    obs_sum = 0.0
    obs_sum_sq = 0.0
    obs_min, obs_max = math.inf, -math.inf
    gap_sum = 0.0
    gap_sum_sq = 0.0
    gap_min, gap_max = math.inf, -math.inf

    fused_sample: list[np.ndarray] = []
    obs_sample: list[np.ndarray] = []
    gap_sample: list[np.ndarray] = []
    overlap_diff_max_abs = 0.0

    with rasterio.open(observed_path) as obs_src, \
         rasterio.open(downscaled_path) as dsc_src, \
         rasterio.open(downscaled_mask_path) as mask_src:

        profile = obs_src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            count=1, dtype="float32", nodata=float("nan"),
            compress="deflate", tiled=True,
        )
        source_profile = profile.copy()
        source_profile.update(dtype="uint8", nodata=0)

        raster_height, raster_width = obs_src.height, obs_src.width

        fused_dst = rasterio.open(fused_path, "w", **profile)
        source_dst = rasterio.open(source_mask_path, "w", **source_profile)
        gapfill_dst = rasterio.open(gapfill_path, "w", **profile)
        obs_mask_dst = (
            rasterio.open(obs_valid_mask_path, "w", **source_profile)
            if write_diagnostics else None
        )
        overlap_dst = (
            rasterio.open(overlap_diff_path, "w", **profile)
            if write_diagnostics else None
        )

        try:
            for write_win, _read_win, _core_off in iter_windows(
                obs_src, tile_size_pixels=tile_size, overlap_pixels=0
            ):
                counters["window_count"] += 1
                h, w = int(write_win.height), int(write_win.width)
                if h == 0 or w == 0:
                    continue
                counters["total_pixels"] += h * w

                observed = obs_src.read(1, window=write_win, masked=True).astype(
                    "float64"
                ).filled(np.nan)
                downscaled = dsc_src.read(1, window=write_win, masked=True).astype(
                    "float64"
                ).filled(np.nan)
                dsc_mask = mask_src.read(1, window=write_win, masked=True).astype(
                    "float64"
                ).filled(0)

                obs_finite = np.isfinite(observed)
                obs_in_range = (
                    obs_finite
                    & (observed >= STEP7E_MIN_CELSIUS)
                    & (observed <= STEP7E_MAX_CELSIUS)
                )
                counters["out_of_range_observed_count"] += int(
                    (obs_finite & (~obs_in_range)).sum()
                )

                dsc_finite = np.isfinite(downscaled)
                dsc_flagged_valid = dsc_mask == 1
                dsc_in_range = (
                    dsc_finite
                    & (downscaled >= STEP7E_MIN_CELSIUS)
                    & (downscaled <= STEP7E_MAX_CELSIUS)
                )
                counters["out_of_range_downscaled_count"] += int(
                    (dsc_finite & dsc_flagged_valid & (~dsc_in_range)).sum()
                )
                downscaled_valid = dsc_finite & dsc_flagged_valid & dsc_in_range

                fused_window = np.full((h, w), np.nan, dtype="float64")
                source_window = np.zeros((h, w), dtype="uint8")
                gapfill_window = np.full((h, w), np.nan, dtype="float64")

                fused_window[obs_in_range] = observed[obs_in_range]
                source_window[obs_in_range] = SOURCE_OBSERVED

                gapfill_mask = (~obs_in_range) & downscaled_valid
                fused_window[gapfill_mask] = downscaled[gapfill_mask]
                source_window[gapfill_mask] = SOURCE_GAPFILL
                gapfill_window[gapfill_mask] = downscaled[gapfill_mask]

                counters["observed_valid_pixels"] += int(obs_in_range.sum())
                counters["downscaled_valid_pixels"] += int(downscaled_valid.sum())
                counters["observed_used_pixels"] += int(obs_in_range.sum())
                counters["gapfilled_pixels"] += int(gapfill_mask.sum())
                fused_valid = source_window != SOURCE_INVALID
                counters["fused_valid_pixels"] += int(fused_valid.sum())
                counters["invalid_pixels"] += int((~fused_valid).sum())

                if fused_valid.any():
                    fv = fused_window[fused_valid]
                    fused_sum += float(fv.sum())
                    fused_sum_sq += float(np.square(fv).sum())
                    fused_min = min(fused_min, float(fv.min()))
                    fused_max = max(fused_max, float(fv.max()))
                    if len(fused_sample) < 2_000_000:
                        step = max(1, fv.size // 2000 or 1)
                        fused_sample.append(fv[::step])

                if obs_in_range.any():
                    ov = observed[obs_in_range]
                    obs_sum += float(ov.sum())
                    obs_sum_sq += float(np.square(ov).sum())
                    obs_min = min(obs_min, float(ov.min()))
                    obs_max = max(obs_max, float(ov.max()))
                    if len(obs_sample) < 2_000_000:
                        step = max(1, ov.size // 2000 or 1)
                        obs_sample.append(ov[::step])

                if gapfill_mask.any():
                    gv = downscaled[gapfill_mask]
                    gap_sum += float(gv.sum())
                    gap_sum_sq += float(np.square(gv).sum())
                    gap_min = min(gap_min, float(gv.min()))
                    gap_max = max(gap_max, float(gv.max()))
                    if len(gap_sample) < 2_000_000:
                        step = max(1, gv.size // 2000 or 1)
                        gap_sample.append(gv[::step])

                fused_dst.write(fused_window.astype("float32"), 1, window=write_win)
                source_dst.write(source_window, 1, window=write_win)
                gapfill_dst.write(gapfill_window.astype("float32"), 1, window=write_win)

                if write_diagnostics:
                    obs_mask_window = obs_in_range.astype("uint8")
                    obs_mask_dst.write(obs_mask_window, 1, window=write_win)

                    overlap_window = np.full((h, w), np.nan, dtype="float32")
                    if obs_in_range.any():
                        diff = (fused_window[obs_in_range] - observed[obs_in_range]).astype(
                            "float32"
                        )
                        overlap_window[obs_in_range] = diff
                        if diff.size:
                            overlap_diff_max_abs = max(
                                overlap_diff_max_abs, float(np.max(np.abs(diff)))
                            )
                    overlap_dst.write(overlap_window, 1, window=write_win)
        finally:
            fused_dst.close()
            source_dst.close()
            gapfill_dst.close()
            if obs_mask_dst is not None:
                obs_mask_dst.close()
            if overlap_dst is not None:
                overlap_dst.close()

    def _stats_from_accumulators(n, s, s2, vmin, vmax):
        if not n:
            return None, None, None, None
        mean = s / n
        std = math.sqrt(max(s2 / n - mean ** 2, 0.0))
        return vmin, vmax, mean, std

    def _pctl(arr: np.ndarray, q: float) -> float | None:
        return float(np.percentile(arr, q)) if arr.size else None

    n_fused = counters["fused_valid_pixels"]
    n_obs = counters["observed_valid_pixels"]
    n_gap = counters["gapfilled_pixels"]

    f_min, f_max, f_mean, f_std = _stats_from_accumulators(
        n_fused, fused_sum, fused_sum_sq, fused_min, fused_max
    )
    o_min, o_max, o_mean, o_std = _stats_from_accumulators(
        n_obs, obs_sum, obs_sum_sq, obs_min, obs_max
    )
    g_min, g_max, g_mean, g_std = _stats_from_accumulators(
        n_gap, gap_sum, gap_sum_sq, gap_min, gap_max
    )

    fused_concat = np.concatenate(fused_sample) if fused_sample else np.array([])
    obs_concat = np.concatenate(obs_sample) if obs_sample else np.array([])
    gap_concat = np.concatenate(gap_sample) if gap_sample else np.array([])

    total = counters["total_pixels"]
    observed_coverage_pct = round(100.0 * n_obs / total, 4) if total else 0.0
    fused_coverage_pct = round(100.0 * n_fused / total, 4) if total else 0.0

    stats = {
        **counters,
        "gapfilled_pct_of_total": (
            round(100.0 * n_gap / total, 4) if total else 0.0
        ),
        "gapfilled_pct_of_fused": (
            round(100.0 * n_gap / n_fused, 4) if n_fused else 0.0
        ),
        "observed_coverage_pct": observed_coverage_pct,
        "fused_coverage_pct": fused_coverage_pct,
        "coverage_gain_pct": round(fused_coverage_pct - observed_coverage_pct, 4),
        "fused_lst_min": f_min, "fused_lst_max": f_max,
        "fused_lst_mean": f_mean, "fused_lst_std": f_std,
        "fused_lst_median": _pctl(fused_concat, 50),
        "fused_lst_p05": _pctl(fused_concat, 5),
        "fused_lst_p95": _pctl(fused_concat, 95),
        "observed_lst_min": o_min, "observed_lst_max": o_max,
        "observed_lst_mean": o_mean, "observed_lst_std": o_std,
        "observed_lst_median": _pctl(obs_concat, 50),
        "observed_lst_p05": _pctl(obs_concat, 5),
        "observed_lst_p95": _pctl(obs_concat, 95),
        "gapfilled_lst_min": g_min, "gapfilled_lst_max": g_max,
        "gapfilled_lst_mean": g_mean, "gapfilled_lst_std": g_std,
        "gapfilled_lst_median": _pctl(gap_concat, 50),
        "gapfilled_lst_p05": _pctl(gap_concat, 5),
        "gapfilled_lst_p95": _pctl(gap_concat, 95),
    }

    output_paths = {
        "fused": str(fused_path),
        "source_mask": str(source_mask_path),
        "gapfill_amount": str(gapfill_path),
        "observed_valid_mask": str(obs_valid_mask_path) if write_diagnostics else None,
        "fused_minus_observed_on_overlap": (
            str(overlap_diff_path) if write_diagnostics else None
        ),
    }
    raster_info = {
        "raster_shape": [raster_height, raster_width],
        "crs": str(profile["crs"]),
        "transform": [
            profile["transform"].a, profile["transform"].b, profile["transform"].c,
            profile["transform"].d, profile["transform"].e, profile["transform"].f,
        ],
    }

    warnings_list: list[str] = []
    if write_diagnostics and overlap_diff_max_abs > 1e-6:
        warnings_list.append(
            "fused_minus_observed_on_overlap sanity check found nonzero "
            f"difference (max abs = {overlap_diff_max_abs}) where observed "
            "Landsat was used; this should be exactly zero by construction."
        )

    return {
        "stats": stats,
        "output_paths": output_paths,
        "raster_info": raster_info,
        "fused_sample": fused_concat,
        "warnings": warnings_list,
    }


def write_plots(run_result: dict, output_dir: Path, source_mask_path: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    rng = np.random.default_rng(42)

    fused_sample = run_result["fused_sample"]
    if fused_sample.size:
        s = fused_sample
        if s.size > 50_000:
            idx = rng.choice(s.size, size=50_000, replace=False)
            s = s[idx]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s, bins=60, color="#1f77b4", alpha=0.85)
        ax.set_xlabel("Fused LST (C)")
        ax.set_ylabel("Count")
        ax.set_title("Step7E fused LST distribution (sampled)")
        fig.tight_layout()
        path = output_dir / "fused_lst_histogram.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    try:
        with rasterio.open(source_mask_path) as src:
            max_dim = 1200
            scale = max(1, max(src.width, src.height) // max_dim)
            out_shape = (max(1, src.height // scale), max(1, src.width // scale))
            data = src.read(
                1,
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling.nearest,
            )
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = plt.matplotlib.colors.ListedColormap(["#222222", "#1f77b4", "#ff7f0e"])
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["invalid", "observed", "gap-fill"])
        ax.set_title("Step7E source mask (downsampled)")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        path = output_dir / "source_mask_map.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))
    except Exception as exc:  # noqa: BLE001
        log.warning("source_mask_map.png uretilemedi: %s", exc)

    return written


def write_metadata(
    output_dir: Path,
    observed_path: Path,
    downscaled_path: Path,
    downscaled_mask_path: Path,
    step7d_context: dict,
    run_result: dict,
    tile_size: int,
    write_diagnostics: bool,
    plots_written: list[str],
    warnings_list: list[str],
) -> Path:
    payload = {
        "created_at": datetime.now().isoformat(),
        "script": "step7e_fuse_landsat_downscaled_lst.py",
        "observed_landsat_path": str(observed_path),
        "downscaled_lst_path": str(downscaled_path),
        "downscaled_valid_mask_path": str(downscaled_mask_path),
        "step7d_metadata_path": str(
            BASE_DIR / "outputs" / "step7d" / "downscaling_prediction_metadata.json"
        ),
        "fusion_rule": (
            "fused = observed_landsat where finite and in "
            f"[{STEP7E_MIN_CELSIUS}, {STEP7E_MAX_CELSIUS}] C (source=1); "
            "elif downscaled_lst is finite, downscaled_valid_mask==1, and in "
            f"[{STEP7E_MIN_CELSIUS}, {STEP7E_MAX_CELSIUS}] C, fused = "
            "downscaled_lst (source=2); else fused = NaN (source=0). "
            "No averaging/blending; observed Landsat always has priority."
        ),
        "source_mask_codes": SOURCE_MASK_CODES,
        "output_paths": run_result["output_paths"],
        "plots_written": plots_written,
        "tile_size": tile_size,
        "write_diagnostics": write_diagnostics,
        "raster_shape": run_result["raster_info"]["raster_shape"],
        "crs": run_result["raster_info"]["crs"],
        "transform": run_result["raster_info"]["transform"],
        "valid_celsius_range": [STEP7E_MIN_CELSIUS, STEP7E_MAX_CELSIUS],
        "no_model_trained": True,
        "no_fire_risk_model_trained": True,
        "no_burned_area_labels_used": True,
        "no_firms_labels_used": True,
        "observed_landsat_priority": True,
        "downscaled_used_only_for_missing_observed": True,
        "step7d_spatial_calibration_note": step7d_context.get("spatial_calibration_note"),
        "warnings": warnings_list,
    }
    path = output_dir / "fused_lst_metadata.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_stats(output_dir: Path, run_result: dict, tile_size: int) -> Path:
    payload = {
        "created_at": datetime.now().isoformat(),
        "raster_shape": run_result["raster_info"]["raster_shape"],
        "crs": run_result["raster_info"]["crs"],
        "transform": run_result["raster_info"]["transform"],
        "tile_size": tile_size,
        **run_result["stats"],
    }
    path = output_dir / "fused_lst_stats.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_summary(
    output_dir: Path,
    run_result: dict,
    step7d_context: dict,
    warnings_list: list[str],
) -> Path:
    def fmt(v, digits=3):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    stats = run_result["stats"]

    lines = [
        "# Step7E: Fused / Gap-Filled Current-Period LST",
        "",
        "**Step7E fuses observed Landsat current-period LST with Step7D "
        "downscaled LST. Observed Landsat pixels are kept wherever valid. "
        "Downscaled LST is only used to fill missing observed Landsat "
        "pixels. No model is trained here. No burned-area or FIRMS labels "
        "are used. This is not fire-risk validation and not independent "
        "model validation.**",
        "",
        f"- Created at: `{datetime.now().isoformat()}`",
        f"- Reference raster shape (h, w): `{run_result['raster_info']['raster_shape']}`",
        f"- CRS: `{run_result['raster_info']['crs']}`",
        "",
        "## Fusion rule",
        "",
        "- Observed Landsat LST is used wherever finite and within "
        f"[{STEP7E_MIN_CELSIUS}, {STEP7E_MAX_CELSIUS}] C (**source_mask=1**, "
        "highest priority).",
        "- Step7D downscaled LST is used **only** where observed Landsat is "
        "missing/invalid, provided the downscaled value is finite, flagged "
        "valid by Step7D, and within the same Celsius range "
        "(**source_mask=2**, gap-fill).",
        "- Otherwise the pixel remains NaN (**source_mask=0**, invalid).",
        "- Observed and downscaled values are **never averaged or blended**; "
        "valid observed pixels are never overwritten by predictions.",
        "",
        "## Coverage",
        "",
        f"- Observed Landsat coverage: `{fmt(stats['observed_coverage_pct'], 2)}%`",
        f"- Fused coverage: `{fmt(stats['fused_coverage_pct'], 2)}%`",
        f"- Coverage gain from gap-filling: `{fmt(stats['coverage_gain_pct'], 2)}` "
        "percentage points",
        f"- Gap-filled pixels: `{stats['gapfilled_pixels']}` "
        f"(`{fmt(stats['gapfilled_pct_of_total'], 2)}%` of total, "
        f"`{fmt(stats['gapfilled_pct_of_fused'], 2)}%` of fused-valid pixels)",
        f"- Invalid pixels (no observed, no valid downscaled): "
        f"`{stats['invalid_pixels']}`",
        f"- Out-of-range observed pixels excluded: "
        f"`{stats['out_of_range_observed_count']}`",
        f"- Out-of-range downscaled pixels excluded: "
        f"`{stats['out_of_range_downscaled_count']}`",
        "",
        "## Fused LST distribution",
        "",
        f"- Min / Max: `{fmt(stats['fused_lst_min'])}` / `{fmt(stats['fused_lst_max'])}` C",
        f"- Mean / Std: `{fmt(stats['fused_lst_mean'])}` / `{fmt(stats['fused_lst_std'])}` C",
        f"- Median (sampled): `{fmt(stats['fused_lst_median'])}` C",
        "",
        "## Observed vs gap-filled sub-distributions",
        "",
        f"- Observed: mean `{fmt(stats['observed_lst_mean'])}` C, "
        f"std `{fmt(stats['observed_lst_std'])}` C "
        f"(n=`{stats['observed_valid_pixels']}`)",
        f"- Gap-filled (downscaled): mean `{fmt(stats['gapfilled_lst_mean'])}` C, "
        f"std `{fmt(stats['gapfilled_lst_std'])}` C (n=`{stats['gapfilled_pixels']}`)",
        "",
        "## Sanity check (not model validation)",
        "",
        "> `fused_minus_observed_on_overlap` should be exactly zero wherever "
        "observed Landsat was used (source_mask=1), by construction. This is "
        "a sanity check only, not an accuracy/validation metric.",
        "",
        "## Limitations",
        "",
    ]

    note = step7d_context.get("spatial_calibration_note")
    if note:
        lines.append(f"- Step7D/Step7C limitation: {note}")
    else:
        lines.append(
            "- The current downscaled layer is a spatial context calibration "
            "based on MODIS summer-mean context, not daily MODIS gap-filling yet."
        )
    lines.extend([
        "- Step7E prepares a more spatially complete current-period thermal "
        "state raster for later anomaly/TVDI/fire-risk experiments; it does "
        "not itself validate or improve those downstream products.",
        "- No model is trained in Step7E; this is a deterministic raster "
        "fusion/gap-filling step.",
    ])

    if warnings_list:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings_list)

    path = output_dir / "fused_lst_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(
    observed_path_arg: str | None = None,
    downscaled_path_arg: str | None = None,
    downscaled_mask_path_arg: str | None = None,
    output_dir: str = STEP7E_OUTPUT_DIR,
    tile_size: int = STEP7E_TILE_SIZE,
    force: bool = False,
    write_diagnostics: bool = STEP7E_WRITE_DIAGNOSTICS,
    make_plots: bool = False,
) -> dict:
    log.info("=" * 60)
    log.info("STEP 7E BASLIYOR (gozlemlenen Landsat + Step7D downscaled LST fusion)")
    log.info("=" * 60)

    out_dir = BASE_DIR / output_dir
    required_outputs = [
        out_dir / "fused_lst_celsius.tif",
        out_dir / "fused_lst_source_mask.tif",
        out_dir / "fused_lst_gapfill_amount.tif",
        out_dir / "fused_lst_metadata.json",
        out_dir / "fused_lst_stats.json",
        out_dir / "fused_lst_summary.md",
    ]
    if any(p.exists() for p in required_outputs) and not force:
        present = [p.name for p in required_outputs if p.exists()]
        raise SystemExit(
            "Step7E ciktilari zaten var (" + ", ".join(present)
            + "). Uzerine yazmak icin --force verin."
        )

    observed_path = resolve_observed_path(observed_path_arg)
    downscaled_path = Path(downscaled_path_arg) if downscaled_path_arg else (
        BASE_DIR / STEP7E_DOWNSCALED_LST_PATH
    )
    downscaled_mask_path = Path(downscaled_mask_path_arg) if downscaled_mask_path_arg else (
        BASE_DIR / STEP7E_DOWNSCALED_VALID_MASK_PATH
    )

    for label, p in (
        ("observed Landsat LST", observed_path),
        ("Step7D downscaled LST", downscaled_path),
        ("Step7D downscaled valid mask", downscaled_mask_path),
    ):
        if not p.exists():
            raise SystemExit(f"{label} bulunamadi: {p}")

    log.info("Gozlemlenen (referans) grid: %s", observed_path)
    log.info("Downscaled LST: %s", downscaled_path)
    log.info("Downscaled valid mask: %s", downscaled_mask_path)

    validate_grid_alignment(
        observed_path,
        {"downscaled_lst": downscaled_path, "downscaled_valid_mask": downscaled_mask_path},
    )

    step7d_context = load_step7d_context()

    run_result = run_fusion(
        observed_path, downscaled_path, downscaled_mask_path,
        out_dir, tile_size, write_diagnostics,
    )
    warnings_list = list(run_result["warnings"])

    plots_written: list[str] = []
    if make_plots:
        plots_written = write_plots(
            run_result, out_dir, out_dir / "fused_lst_source_mask.tif"
        )

    metadata_path_out = write_metadata(
        out_dir, observed_path, downscaled_path, downscaled_mask_path,
        step7d_context, run_result, tile_size, write_diagnostics,
        plots_written, warnings_list,
    )
    stats_path_out = write_stats(out_dir, run_result, tile_size)
    summary_path_out = write_summary(out_dir, run_result, step7d_context, warnings_list)

    log.info("Fused LST: %s", run_result["output_paths"]["fused"])
    log.info("Source mask: %s", run_result["output_paths"]["source_mask"])
    log.info("Metadata: %s", metadata_path_out)
    log.info("Stats: %s", stats_path_out)
    log.info("Summary: %s", summary_path_out)
    log.info(
        "Gozlem kapsami: %.2f%% -> Fused kapsam: %.2f%% (kazanim: %.2f puan, %d gap-fill piksel)",
        run_result["stats"]["observed_coverage_pct"],
        run_result["stats"]["fused_coverage_pct"],
        run_result["stats"]["coverage_gain_pct"],
        run_result["stats"]["gapfilled_pixels"],
    )
    log.info("=" * 60)
    log.info("STEP 7E TAMAMLANDI (no model trained, no fire-risk/burned-area labels used)")
    log.info("=" * 60)

    return {
        "fused_path": run_result["output_paths"]["fused"],
        "source_mask_path": run_result["output_paths"]["source_mask"],
        "gapfill_amount_path": run_result["output_paths"]["gapfill_amount"],
        "metadata_path": str(metadata_path_out),
        "stats_path": str(stats_path_out),
        "summary_path": str(summary_path_out),
        "stats": run_result["stats"],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step7E: fuse observed Landsat LST with Step7D downscaled "
        "LST (observed priority, gap-fill only where missing; no model "
        "trained, no fire-risk/burned-area labels used)."
    )
    parser.add_argument("--observed", type=str, default=None)
    parser.add_argument("--downscaled", type=str, default=None)
    parser.add_argument("--downscaled-mask", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=STEP7E_OUTPUT_DIR)
    parser.add_argument("--tile-size", type=int, default=STEP7E_TILE_SIZE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        observed_path_arg=args.observed,
        downscaled_path_arg=args.downscaled,
        downscaled_mask_path_arg=args.downscaled_mask,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        force=args.force,
        write_diagnostics=not args.no_diagnostics,
        make_plots=args.plot,
    )