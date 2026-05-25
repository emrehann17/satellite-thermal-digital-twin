"""
step5b_diagnostic_report.py

Step5 sonrası tanı raporu üreticisi.

Step5 GeoTIFF çıktılarını pencereler halinde okur; tüm rasterı belleğe
yüklemeden histogram PNG'leri ve kısa özet dosyaları üretir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from core.config import (
    CURRENT_PERIOD_DAYS,
    CURRENT_PERIOD_END_DATE,
    STEP5_MIN_BASELINE_STD_CELSIUS,
    STEP5_MIN_BASELINE_VALID_COUNT,
    STEP5_MIN_CURRENT_VALID_COUNT,
    STEP5_MAX_CURRENT_STD_CELSIUS,
    STEP5_MAX_CURRENT_RANGE_CELSIUS,
)
from core.io_utils import setup_logger


BASE_DIR = Path(__file__).resolve().parent
STEP5_OUTPUT_DIR = BASE_DIR / "outputs" / "step5"
DIAGNOSTICS_DIR = STEP5_OUTPUT_DIR / "diagnostics"

log, log_file = setup_logger("step5b_diagnostics")


@dataclass(frozen=True)
class RasterSpec:
    key: str
    filename: str
    title: str
    xlabel: str
    bins: np.ndarray
    color: str
    threshold_lines: tuple[float, ...] = ()
    integer_bar: bool = False
    mask_layer: bool = False


@dataclass
class RasterAccumulator:
    spec: RasterSpec
    hist: np.ndarray = field(init=False)
    count: int = 0
    sum_value: float = 0.0
    sum_square: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    below_hist_range: int = 0
    above_hist_range: int = 0
    flagged_count: int = 0
    anomaly_abs_gt_2: int = 0
    anomaly_abs_gt_3: int = 0
    anomaly_le_neg_3: int = 0
    anomaly_ge_pos_3: int = 0

    def __post_init__(self) -> None:
        self.hist = np.zeros(len(self.spec.bins) - 1, dtype=np.int64)

    def update(self, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return

        finite = finite.astype("float64", copy=False)
        self.count += int(finite.size)
        self.sum_value += float(np.sum(finite))
        self.sum_square += float(np.sum(finite * finite))

        current_min = float(np.min(finite))
        current_max = float(np.max(finite))
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)

        self.hist += np.histogram(finite, bins=self.spec.bins)[0]
        self.below_hist_range += int(np.sum(finite < self.spec.bins[0]))
        self.above_hist_range += int(np.sum(finite > self.spec.bins[-1]))

        if self.spec.mask_layer:
            self.flagged_count += int(np.sum(finite >= 0.5))

        if self.spec.key == "anomaly_zscore":
            self.anomaly_abs_gt_2 += int(np.sum(np.abs(finite) > 2))
            self.anomaly_abs_gt_3 += int(np.sum(np.abs(finite) > 3))
            self.anomaly_le_neg_3 += int(np.sum(finite <= -3))
            self.anomaly_ge_pos_3 += int(np.sum(finite >= 3))

    def as_dict(self, total_pixels: int) -> dict:
        mean = self.sum_value / self.count if self.count else None
        variance = None
        std = None
        if self.count:
            variance = max((self.sum_square / self.count) - (mean * mean), 0.0)
            std = variance**0.5

        data = {
            "file": self.spec.filename,
            "valid_pixel_count": self.count,
            "valid_pixel_percent_of_raster": (
                100.0 * self.count / total_pixels if total_pixels else None
            ),
            "min": self.min_value,
            "max": self.max_value,
            "mean": mean,
            "std": std,
            "histogram_bin_edges": self.spec.bins.tolist(),
            "histogram_counts": self.hist.tolist(),
            "below_histogram_range_count": self.below_hist_range,
            "above_histogram_range_count": self.above_hist_range,
        }

        if self.spec.mask_layer:
            data["flagged_pixel_count"] = self.flagged_count
            data["flagged_pixel_percent_of_valid"] = (
                100.0 * self.flagged_count / self.count if self.count else None
            )

        if self.spec.key == "anomaly_zscore":
            data.update(
                {
                    "abs_z_gt_2_count": self.anomaly_abs_gt_2,
                    "abs_z_gt_2_percent_of_valid": (
                        100.0 * self.anomaly_abs_gt_2 / self.count if self.count else None
                    ),
                    "abs_z_gt_3_count": self.anomaly_abs_gt_3,
                    "abs_z_gt_3_percent_of_valid": (
                        100.0 * self.anomaly_abs_gt_3 / self.count if self.count else None
                    ),
                    "z_le_minus_3_count": self.anomaly_le_neg_3,
                    "z_ge_plus_3_count": self.anomaly_ge_pos_3,
                }
            )

        return data


def build_specs() -> list[RasterSpec]:
    return [
        RasterSpec(
            key="anomaly_zscore",
            filename="anomaly_zscore.tif",
            title="Anomali z-score dağılımı",
            xlabel="z-score",
            bins=np.linspace(-6, 6, 121),
            color="#3f7fbf",
            threshold_lines=(-3, -2, 0, 2, 3),
        ),
        RasterSpec(
            key="current_period_median_celsius",
            filename="current_period_median_celsius.tif",
            title="Güncel dönem median LST",
            xlabel="Celsius derece",
            bins=np.linspace(0, 60, 121),
            color="#d95f02",
        ),
        RasterSpec(
            key="baseline_lst_mean_celsius",
            filename="baseline_lst_mean_celsius.tif",
            title="Baseline ortalama LST",
            xlabel="Celsius derece",
            bins=np.linspace(0, 60, 121),
            color="#1b9e77",
        ),
        RasterSpec(
            key="baseline_lst_std_celsius",
            filename="baseline_lst_std_celsius.tif",
            title="Baseline LST standart sapması",
            xlabel="Celsius derece",
            bins=np.linspace(0, 10, 101),
            color="#7570b3",
            threshold_lines=(STEP5_MIN_BASELINE_STD_CELSIUS,),
        ),
        RasterSpec(
            key="baseline_valid_count",
            filename="baseline_valid_count.tif",
            title="Baseline geçerli gözlem sayısı",
            xlabel="geçerli gözlem sayısı",
            bins=np.arange(-0.5, 12.5, 1),
            color="#4d4d4d",
            threshold_lines=(STEP5_MIN_BASELINE_VALID_COUNT,),
            integer_bar=True,
        ),
        RasterSpec(
            key="current_period_valid_count",
            filename="current_period_valid_count.tif",
            title="Güncel dönem geçerli gözlem sayısı",
            xlabel="geçerli gözlem sayısı",
            bins=np.arange(-0.5, 30.5, 1),
            color="#4d4d4d",
            threshold_lines=(STEP5_MIN_CURRENT_VALID_COUNT,),
            integer_bar=True,
        ),
        RasterSpec(
            key="current_period_std_celsius",
            filename="current_period_std_celsius.tif",
            title="Güncel dönem pencere içi standart sapma",
            xlabel="Celsius derece",
            bins=np.linspace(0, 10, 101),
            color="#e7298a",
            threshold_lines=(STEP5_MAX_CURRENT_STD_CELSIUS,),
        ),
        RasterSpec(
            key="current_period_range_celsius",
            filename="current_period_range_celsius.tif",
            title="Güncel dönem pencere içi max-min aralığı",
            xlabel="Celsius derece",
            bins=np.linspace(0, 20, 101),
            color="#66a61e",
            threshold_lines=(STEP5_MAX_CURRENT_RANGE_CELSIUS,),
        ),
        RasterSpec(
            key="low_baseline_count_mask",
            filename="low_baseline_count_mask.tif",
            title="Düşük baseline gözlem sayısı maskesi",
            xlabel="maske değeri",
            bins=np.array([-0.5, 0.5, 1.5]),
            color="#c51b7d",
            integer_bar=True,
            mask_layer=True,
        ),
        RasterSpec(
            key="low_baseline_std_mask",
            filename="low_baseline_std_mask.tif",
            title="Düşük baseline standart sapma maskesi",
            xlabel="maske değeri",
            bins=np.array([-0.5, 0.5, 1.5]),
            color="#c51b7d",
            integer_bar=True,
            mask_layer=True,
        ),
        RasterSpec(
            key="low_current_count_mask",
            filename="low_current_count_mask.tif",
            title="Düşük güncel dönem gözlem sayısı maskesi",
            xlabel="maske değeri",
            bins=np.array([-0.5, 0.5, 1.5]),
            color="#c51b7d",
            integer_bar=True,
            mask_layer=True,
        ),
        RasterSpec(
            key="low_current_variability_mask",
            filename="low_current_variability_mask.tif",
            title="Yüksek güncel dönem variability maskesi",
            xlabel="maske değeri",
            bins=np.array([-0.5, 0.5, 1.5]),
            color="#c51b7d",
            integer_bar=True,
            mask_layer=True,
        ),
    ]


def read_window_values(src: rasterio.io.DatasetReader, window) -> np.ndarray:
    data = src.read(1, window=window, masked=True)
    values = data.astype("float32").filled(np.nan)
    if src.nodata is not None:
        values = np.where(values == src.nodata, np.nan, values)
    return values


def analyze_raster(path: Path, spec: RasterSpec) -> tuple[RasterAccumulator, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Gerekli Step5 çıktısı bulunamadı: {path}")

    accumulator = RasterAccumulator(spec)

    with rasterio.open(path) as src:
        total_pixels = src.width * src.height
        raster_meta = {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform),
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }

        for _, window in src.block_windows(1):
            accumulator.update(read_window_values(src, window))

    raster_meta["total_pixels"] = total_pixels
    return accumulator, raster_meta


def plot_histogram(accumulator: RasterAccumulator, output_path: Path) -> None:
    spec = accumulator.spec
    bins = spec.bins
    counts = accumulator.hist

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    if spec.integer_bar:
        centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(centers, counts, width=0.8, color=spec.color, edgecolor="black", linewidth=0.2)
        ax.set_xticks(centers)
    else:
        widths = np.diff(bins)
        ax.bar(bins[:-1], counts, width=widths, align="edge", color=spec.color, edgecolor="none")

    for line_value in spec.threshold_lines:
        ax.axvline(line_value, color="black", linestyle="--", linewidth=1)

    ax.set_title(spec.title)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel("piksel sayısı")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def format_percent(value: float | None) -> str:
    if value is None:
        return "yok"
    return f"{value:.3f}%"


def write_summary_markdown(summary: dict, path: Path) -> None:
    anomaly = summary["rasters"]["anomaly_zscore"]
    baseline_std = summary["rasters"]["baseline_lst_std_celsius"]
    low_baseline_count = summary["rasters"]["low_baseline_count_mask"]
    low_baseline_std = summary["rasters"]["low_baseline_std_mask"]
    low_current_count = summary["rasters"]["low_current_count_mask"]
    low_current_variability = summary["rasters"]["low_current_variability_mask"]
    current_std = summary["rasters"]["current_period_std_celsius"]
    current_range = summary["rasters"]["current_period_range_celsius"]

    lines = [
        "# Step5 Tanı Özeti",
        "",
        f"Oluşturulma zamanı: `{summary['created_at']}`",
        f"Güncel dönem: `{CURRENT_PERIOD_END_DATE}` tarihinde biten `{CURRENT_PERIOD_DAYS}` günlük pencere",
        "",
        "## Temel Anomali Ölçütleri",
        "",
        f"- Geçerli anomali piksel sayısı: `{anomaly['valid_pixel_count']}`",
        f"- Ortalama z-score: `{anomaly['mean']}`",
        f"- Minimum / maksimum z-score: `{anomaly['min']}` / `{anomaly['max']}`",
        f"- |z| > 2 olan piksel sayısı: `{anomaly['abs_z_gt_2_count']}` "
        f"(geçerli piksellerin {format_percent(anomaly['abs_z_gt_2_percent_of_valid'])})",
        f"- |z| > 3 olan piksel sayısı: `{anomaly['abs_z_gt_3_count']}` "
        f"(geçerli piksellerin {format_percent(anomaly['abs_z_gt_3_percent_of_valid'])})",
        "",
        "## Maske Tanıları",
        "",
        f"- Düşük baseline gözlem sayısı olan piksel sayısı: `{low_baseline_count['flagged_pixel_count']}` "
        f"(geçerli piksellerin {format_percent(low_baseline_count['flagged_pixel_percent_of_valid'])})",
        f"- Düşük baseline standart sapması olan piksel sayısı: `{low_baseline_std['flagged_pixel_count']}` "
        f"(geçerli piksellerin {format_percent(low_baseline_std['flagged_pixel_percent_of_valid'])})",
        f"- Düşük güncel dönem gözlem sayısı olan piksel sayısı: `{low_current_count['flagged_pixel_count']}` "
        f"(geçerli piksellerin {format_percent(low_current_count['flagged_pixel_percent_of_valid'])})",
        f"- Yüksek güncel dönem variability taşıyan piksel sayısı: `{low_current_variability['flagged_pixel_count']}` "
        f"(geçerli piksellerin {format_percent(low_current_variability['flagged_pixel_percent_of_valid'])})",
        "",
        "## Baseline Standart Sapması",
        "",
        f"- Ortalama baseline standart sapması: `{baseline_std['mean']}` Celsius",
        f"- Minimum / maksimum baseline standart sapması: `{baseline_std['min']}` / `{baseline_std['max']}` Celsius",
        f"- Kabul edilen minimum baseline standart sapması: `{STEP5_MIN_BASELINE_STD_CELSIUS}` Celsius",
        "",
        "## Güncel Dönem Variability",
        "",
        f"- Ortalama current std: `{current_std['mean']}` Celsius",
        f"- Ortalama current range: `{current_range['mean']}` Celsius",
        f"- İzin verilen maksimum current std: `{STEP5_MAX_CURRENT_STD_CELSIUS}` Celsius",
        f"- İzin verilen maksimum current range: `{STEP5_MAX_CURRENT_RANGE_CELSIUS}` Celsius",
        "",
        "## Çıktı Dosyaları",
        "",
    ]

    for key, figure_path in summary["figures"].items():
        lines.append(f"- `{key}`: `{figure_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_specs()

    log.info("Step5 tanı raporu başlatıldı.")
    log.info("Step5 çıktı klasörü: %s", STEP5_OUTPUT_DIR)
    log.info("Tanı çıktıları klasörü: %s", DIAGNOSTICS_DIR)

    summary = {
        "step": "step5b_diagnostic_report",
        "description_tr": "Step5 sonrası histogram ve özet tanı raporu",
        "created_at": datetime.now().isoformat(),
        "log_file": str(log_file),
        "config": {
            "current_period_days": CURRENT_PERIOD_DAYS,
            "current_period_end_date": CURRENT_PERIOD_END_DATE,
            "min_baseline_std_celsius": STEP5_MIN_BASELINE_STD_CELSIUS,
            "min_baseline_valid_count": STEP5_MIN_BASELINE_VALID_COUNT,
            "min_current_valid_count": STEP5_MIN_CURRENT_VALID_COUNT,
        },
        "rasters": {},
        "raster_meta": {},
        "figures": {},
    }

    total_pixels_by_raster = {}
    for spec in specs:
        raster_path = STEP5_OUTPUT_DIR / spec.filename
        log.info("Analiz ediliyor: %s", raster_path.name)
        accumulator, raster_meta = analyze_raster(raster_path, spec)

        total_pixels = int(raster_meta["total_pixels"])
        total_pixels_by_raster[spec.key] = total_pixels
        summary["rasters"][spec.key] = accumulator.as_dict(total_pixels)
        summary["raster_meta"][spec.key] = raster_meta

        figure_path = DIAGNOSTICS_DIR / f"{spec.key}_hist.png"
        plot_histogram(accumulator, figure_path)
        summary["figures"][spec.key] = str(figure_path.relative_to(BASE_DIR))

    summary["total_pixels_by_raster"] = total_pixels_by_raster

    json_path = DIAGNOSTICS_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_path = DIAGNOSTICS_DIR / "summary.md"
    write_summary_markdown(summary, markdown_path)

    log.info("Tanı JSON özeti yazıldı: %s", json_path)
    log.info("Tanı Markdown özeti yazıldı: %s", markdown_path)

    return {
        "summary_json": json_path,
        "summary_markdown": markdown_path,
        "diagnostics_dir": DIAGNOSTICS_DIR,
    }


if __name__ == "__main__":
    main()