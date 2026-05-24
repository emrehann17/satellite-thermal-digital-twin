"""
step5c_seam_diagnostic.py

Step5 sonrası path/row dikişi tanı aracı.

Anomali rasterındaki keskin geçişleri dikiş adayı olarak seçer ve bu aday
piksellerde current, baseline, std ve valid-count katmanlarının nasıl değiştiğini
ölçer. Amaç dikişin olası nedenini düzeltmeden önce sayısal olarak ayırmaktır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

from core.config import STEP5_WINDOW_SIZE
from core.io_utils import setup_logger


BASE_DIR = Path(__file__).resolve().parent
STEP5_OUTPUT_DIR = BASE_DIR / "outputs" / "step5"
DIAGNOSTICS_DIR = STEP5_OUTPUT_DIR / "diagnostics" / "step5c_seam_diagnostic"

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

SEAM_PERCENTILE = 98.5
MAX_GRADIENT_SAMPLES = 2_000_000

log, log_file = setup_logger("step5c_seam_diagnostic")


@dataclass(frozen=True)
class LayerSpec:
    key: str
    filename: str
    label_tr: str
    interpretation_tr: str


@dataclass
class GroupStats:
    count: int = 0
    sum_value: float = 0.0
    sum_square: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

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

    def as_dict(self) -> dict:
        mean = self.sum_value / self.count if self.count else None
        std = None
        if self.count:
            variance = max((self.sum_square / self.count) - (mean * mean), 0.0)
            std = variance**0.5
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "min": self.min_value,
            "max": self.max_value,
        }


def layer_specs() -> list[LayerSpec]:
    return [
        LayerSpec(
            key="anomaly_zscore",
            filename="anomaly_zscore.tif",
            label_tr="anomaly_zscore",
            interpretation_tr="Dikişin doğrudan görüldüğü hedef raster.",
        ),
        LayerSpec(
            key="current_period_median_celsius",
            filename="current_period_median_celsius.tif",
            label_tr="current_period_median_celsius",
            interpretation_tr="Sıçrama burada yüksekse current dönem mozaiği veya sahne kapsamı etkisi olabilir.",
        ),
        LayerSpec(
            key="baseline_lst_mean_celsius",
            filename="baseline_lst_mean_celsius.tif",
            label_tr="baseline_lst_mean_celsius",
            interpretation_tr="Sıçrama burada yüksekse baseline kompozitinin iki tarafı farklı davranıyor olabilir.",
        ),
        LayerSpec(
            key="baseline_lst_std_celsius",
            filename="baseline_lst_std_celsius.tif",
            label_tr="baseline_lst_std_celsius",
            interpretation_tr="Sıçrama burada yüksekse z-score paydası dikişi büyütüyor olabilir.",
        ),
        LayerSpec(
            key="baseline_valid_count",
            filename="baseline_valid_count.tif",
            label_tr="baseline_valid_count",
            interpretation_tr="Sıçrama burada yüksekse baseline tarafında coverage/sahne sayısı farkı vardır.",
        ),
        LayerSpec(
            key="current_period_valid_count",
            filename="current_period_valid_count.tif",
            label_tr="current_period_valid_count",
            interpretation_tr="Sıçrama burada yüksekse current dönem coverage/sahne sayısı farkı vardır.",
        ),
    ]


def read_window(src: rasterio.io.DatasetReader, window) -> np.ndarray:
    data = src.read(1, window=window, masked=True).astype("float32")
    values = data.filled(np.nan)
    if src.nodata is not None:
        values = np.where(values == src.nodata, np.nan, values)
    return values.astype("float32", copy=False)


def gradient_strength(values: np.ndarray) -> np.ndarray:
    strength = np.full(values.shape, np.nan, dtype="float32")

    dx = np.abs(values[:, 1:] - values[:, :-1])
    dx_valid = np.isfinite(dx) & np.isfinite(values[:, 1:]) & np.isfinite(values[:, :-1])
    left = strength[:, :-1]
    left[dx_valid] = np.fmax(np.nan_to_num(left[dx_valid], nan=0.0), dx[dx_valid])
    right = strength[:, 1:]
    right[dx_valid] = np.fmax(np.nan_to_num(right[dx_valid], nan=0.0), dx[dx_valid])

    dy = np.abs(values[1:, :] - values[:-1, :])
    dy_valid = np.isfinite(dy) & np.isfinite(values[1:, :]) & np.isfinite(values[:-1, :])
    upper = strength[:-1, :]
    upper[dy_valid] = np.fmax(np.nan_to_num(upper[dy_valid], nan=0.0), dy[dy_valid])
    lower = strength[1:, :]
    lower[dy_valid] = np.fmax(np.nan_to_num(lower[dy_valid], nan=0.0), dy[dy_valid])

    return strength


def open_layers(specs: list[LayerSpec]) -> dict[str, rasterio.io.DatasetReader]:
    datasets = {}
    for spec in specs:
        path = STEP5_OUTPUT_DIR / spec.filename
        if not path.exists():
            raise FileNotFoundError(f"Gerekli Step5 çıktısı bulunamadı: {path}")
        datasets[spec.key] = rasterio.open(path)
    return datasets


def validate_same_grid(datasets: dict[str, rasterio.io.DatasetReader]) -> None:
    first_key = next(iter(datasets))
    first = datasets[first_key]
    for key, src in datasets.items():
        if src.width != first.width or src.height != first.height:
            raise ValueError(f"Raster boyutu uyuşmuyor: {first_key} ile {key}")
        if src.transform != first.transform:
            raise ValueError(f"Raster transform uyuşmuyor: {first_key} ile {key}")
        if src.crs != first.crs:
            raise ValueError(f"Raster CRS uyuşmuyor: {first_key} ile {key}")


def iter_windows(src: rasterio.io.DatasetReader):
    if src.is_tiled:
        yield from (window for _, window in src.block_windows(1))
        return

    for row_off in range(0, src.height, STEP5_WINDOW_SIZE):
        for col_off in range(0, src.width, STEP5_WINDOW_SIZE):
            height = min(STEP5_WINDOW_SIZE, src.height - row_off)
            width = min(STEP5_WINDOW_SIZE, src.width - col_off)
            yield Window(col_off, row_off, width, height)


def estimate_anomaly_gradient_threshold(anomaly_src: rasterio.io.DatasetReader) -> float:
    samples = []
    for window in iter_windows(anomaly_src):
        values = read_window(anomaly_src, window)
        gradients = gradient_strength(values)
        finite = gradients[np.isfinite(gradients) & (gradients > 0)]
        if finite.size == 0:
            continue

        stride = max(1, finite.size // 20_000)
        samples.append(finite[::stride])

        sample_size = sum(arr.size for arr in samples)
        if sample_size >= MAX_GRADIENT_SAMPLES:
            break

    if not samples:
        raise ValueError("Anomali gradyanı için geçerli örnek bulunamadı.")

    merged = np.concatenate(samples)
    return float(np.percentile(merged, SEAM_PERCENTILE))


def relative_ratio(candidate_mean: float | None, background_mean: float | None) -> float | None:
    if candidate_mean is None or background_mean is None:
        return None
    if abs(background_mean) < 1e-9:
        return None
    return candidate_mean / background_mean


def classify_likely_cause(layer_results: dict) -> dict:
    candidate_layers = [
        "current_period_valid_count",
        "baseline_valid_count",
        "baseline_lst_std_celsius",
        "current_period_median_celsius",
        "baseline_lst_mean_celsius",
    ]

    ranked = sorted(
        (
            (key, layer_results[key]["gradient_candidate_to_background_ratio"])
            for key in candidate_layers
            if layer_results[key]["gradient_candidate_to_background_ratio"] is not None
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    if not ranked:
        return {
            "likely_cause_key": None,
            "likely_cause_tr": "Dikiş nedeni otomatik olarak ayrılamadı.",
            "ranked_gradient_ratios": [],
        }

    top_key, top_ratio = ranked[0]
    cause_texts = {
        "current_period_valid_count": (
            "En güçlü aday current period geçerli gözlem sayısındaki ani değişim. "
            "Dikiş büyük olasılıkla güncel dönem Landsat sahne kapsamı/path-row footprint farkından geliyor."
        ),
        "baseline_valid_count": (
            "En güçlü aday baseline geçerli gözlem sayısındaki ani değişim. "
            "Dikiş büyük olasılıkla baseline yıllarındaki coverage/sahne sayısı farkından geliyor."
        ),
        "baseline_lst_std_celsius": (
            "En güçlü aday baseline standart sapmasındaki ani değişim. "
            "Dikiş z-score paydasındaki mekansal farklılık nedeniyle büyüyor olabilir."
        ),
        "current_period_median_celsius": (
            "En güçlü aday güncel dönem median LST yüzeyindeki ani değişim. "
            "Current mozaiği path/row sınırında farklı yüzey sıcaklığı davranışı taşıyor olabilir."
        ),
        "baseline_lst_mean_celsius": (
            "En güçlü aday baseline ortalama LST yüzeyindeki ani değişim. "
            "Baseline kompoziti path/row sınırında farklı davranıyor olabilir."
        ),
    }

    return {
        "likely_cause_key": top_key,
        "likely_cause_tr": cause_texts[top_key],
        "top_gradient_ratio": top_ratio,
        "ranked_gradient_ratios": [
            {"layer": key, "ratio": ratio}
            for key, ratio in ranked
        ],
    }


def plot_gradient_ratios(layer_results: dict, output_path: Path) -> None:
    labels = []
    ratios = []
    for key, result in layer_results.items():
        ratio = result["gradient_candidate_to_background_ratio"]
        if key == "anomaly_zscore" or ratio is None:
            continue
        labels.append(result["label_tr"])
        ratios.append(ratio)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, ratios, color="#377eb8")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Dikiş adayı / arka plan gradyan oranı")
    ax.set_title("Dikiş adayındaki katman gradyan oranları")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown(summary: dict, path: Path) -> None:
    cause = summary["likely_cause"]
    lines = [
        "# Step5C Dikiş Tanı Özeti",
        "",
        f"Oluşturulma zamanı: `{summary['created_at']}`",
        f"Anomali gradyan eşiği: `{summary['seam_gradient_threshold']}`",
        f"Dikiş adayı piksel sayısı: `{summary['seam_candidate_pixel_count']}`",
        f"Dikiş adayı oranı: `{summary['seam_candidate_percent']:.4f}%`",
        "",
        "## Otomatik Yorum",
        "",
        cause["likely_cause_tr"],
        "",
        "## Katman Gradyan Oranları",
        "",
        "Oran, dikiş adayı piksellerdeki ortalama gradyanın arka plan ortalama gradyanına bölümüdür. Yüksek oran, dikişin o katmanda daha belirgin olduğunu gösterir.",
        "",
    ]

    for item in cause["ranked_gradient_ratios"]:
        layer = summary["layer_results"][item["layer"]]
        lines.append(f"- `{layer['label_tr']}`: `{item['ratio']}`")

    lines.extend(
        [
            "",
            "## Çıktılar",
            "",
            f"- Dikiş adayı maskesi: `{summary['outputs']['seam_candidate_mask']}`",
            f"- Anomali gradyan gücü: `{summary['outputs']['anomaly_gradient_strength']}`",
            f"- Gradyan oran grafiği: `{summary['outputs']['gradient_ratio_plot']}`",
            "",
            "## Not",
            "",
            "Bu otomatik tanı, anomaly rasterındaki keskin geçişleri dikiş adayı kabul eder. Doğal topoğrafik kenarlar da aday kümeye girebilir; bu yüzden çıktı QGIS üzerinde görsel olarak kontrol edilmelidir.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    specs = layer_specs()

    log.info("Step5C dikiş tanısı başlatıldı.")
    datasets = open_layers(specs)

    try:
        validate_same_grid(datasets)
        anomaly_src = datasets["anomaly_zscore"]
        threshold = estimate_anomaly_gradient_threshold(anomaly_src)
        log.info("Anomali gradyan dikiş adayı eşiği: %.6f", threshold)

        profile = anomaly_src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=0.0, compress="deflate")

        mask_path = DIAGNOSTICS_DIR / "seam_candidate_mask.tif"
        gradient_path = DIAGNOSTICS_DIR / "anomaly_gradient_strength.tif"

        value_stats = {
            spec.key: {"candidate": GroupStats(), "background": GroupStats()}
            for spec in specs
        }
        gradient_stats = {
            spec.key: {"candidate": GroupStats(), "background": GroupStats()}
            for spec in specs
        }

        total_pixels = anomaly_src.width * anomaly_src.height
        seam_candidate_pixels = 0

        with (
            rasterio.open(mask_path, "w", **profile) as mask_dst,
            rasterio.open(gradient_path, "w", **profile) as gradient_dst,
        ):
            for window in iter_windows(anomaly_src):
                anomaly_values = read_window(anomaly_src, window)
                anomaly_gradient = gradient_strength(anomaly_values)
                candidate_mask = (
                    np.isfinite(anomaly_gradient)
                    & np.isfinite(anomaly_values)
                    & (anomaly_gradient >= threshold)
                )
                background_mask = np.isfinite(anomaly_values) & ~candidate_mask

                seam_candidate_pixels += int(np.sum(candidate_mask))
                mask_dst.write(candidate_mask.astype("float32"), 1, window=window)
                gradient_dst.write(
                    np.where(np.isfinite(anomaly_gradient), anomaly_gradient, 0).astype("float32"),
                    1,
                    window=window,
                )

                for spec in specs:
                    values = read_window(datasets[spec.key], window)
                    layer_gradient = gradient_strength(values)

                    value_stats[spec.key]["candidate"].update(values[candidate_mask])
                    value_stats[spec.key]["background"].update(values[background_mask])
                    gradient_stats[spec.key]["candidate"].update(layer_gradient[candidate_mask])
                    gradient_stats[spec.key]["background"].update(layer_gradient[background_mask])

        layer_results = {}
        for spec in specs:
            candidate_gradient = gradient_stats[spec.key]["candidate"].as_dict()
            background_gradient = gradient_stats[spec.key]["background"].as_dict()
            layer_results[spec.key] = {
                "label_tr": spec.label_tr,
                "interpretation_tr": spec.interpretation_tr,
                "value_candidate": value_stats[spec.key]["candidate"].as_dict(),
                "value_background": value_stats[spec.key]["background"].as_dict(),
                "gradient_candidate": candidate_gradient,
                "gradient_background": background_gradient,
                "gradient_candidate_to_background_ratio": relative_ratio(
                    candidate_gradient["mean"],
                    background_gradient["mean"],
                ),
            }

        cause = classify_likely_cause(layer_results)
        ratio_plot_path = DIAGNOSTICS_DIR / "seam_gradient_ratio.png"
        plot_gradient_ratios(layer_results, ratio_plot_path)

        summary = {
            "step": "step5c_seam_diagnostic",
            "description_tr": "Path/row dikişi için anomaly gradyanı ve katman karşılaştırmalı tanı raporu",
            "created_at": datetime.now().isoformat(),
            "log_file": str(log_file),
            "seam_percentile": SEAM_PERCENTILE,
            "seam_gradient_threshold": threshold,
            "total_pixels": total_pixels,
            "seam_candidate_pixel_count": seam_candidate_pixels,
            "seam_candidate_percent": 100.0 * seam_candidate_pixels / total_pixels,
            "likely_cause": cause,
            "layer_results": layer_results,
            "outputs": {
                "seam_candidate_mask": str(mask_path.relative_to(BASE_DIR)),
                "anomaly_gradient_strength": str(gradient_path.relative_to(BASE_DIR)),
                "gradient_ratio_plot": str(ratio_plot_path.relative_to(BASE_DIR)),
            },
        }

        json_path = DIAGNOSTICS_DIR / "seam_summary.json"
        md_path = DIAGNOSTICS_DIR / "seam_summary.md"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_markdown(summary, md_path)

        log.info("Dikiş tanı JSON özeti yazıldı: %s", json_path)
        log.info("Dikiş tanı Markdown özeti yazıldı: %s", md_path)

        return {
            "summary_json": json_path,
            "summary_markdown": md_path,
            "diagnostics_dir": DIAGNOSTICS_DIR,
        }
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()