"""
step5c_tvdi.py

TVDI (Temperature Vegetation Dryness Index) offline işleme katmanı.

Bu modül mevcut Step5 LST anomaly pipeline'ını DEĞİŞTİRMEZ. Step5 zaten ürettiği
current/baseline LST median rasterlarını ve Step4b'nin indirdiği NDVI rasterlarını
girdi alıp ayrı bir TVDI ürün ailesi üretir:

    - current_tvdi.tif
    - baseline_tvdi_mean.tif
    - baseline_tvdi_std.tif
    - tvdi_anomaly_zscore.tif

TVDI tanımı (basit/ilk sürüm):
    1. LST-NDVI scatter üzerinden NDVI ekseni bin'lere bölünür.
    2. Her NDVI bin'i için:
         wet_edge = düşük LST percentile (ör. p2)
         dry_edge = yüksek LST percentile (ör. p98)
    3. TVDI = (LST - wet_edge) / (dry_edge - wet_edge), [0, 1] aralığına clamp.

Tasarım notları:
    - Step5'in windowed/tiling yardımcıları yeniden kullanılır; full raster belleğe
      alınmaz. Edge fit için sadece (NDVI, LST) çiftlerinin bin-bazlı percentile
      özetleri tutulur — tüm pikseller değil.
    - Temporal interpolation yapılmaz. Yetersiz gözlemde NaN bırakılır.
    - Mevcut Step5/Step5B çıktıları okunur ama üzerine yazılmaz.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import rasterio

from core.config import (
    CURRENT_PERIOD_DAYS,
    ENABLE_TVDI_STEP5,
    MIN_TVDI_BASELINE_STD,
    TVDI_DRY_EDGE_PERCENTILE,
    TVDI_MIN_BASELINE_VALID_COUNT,
    TVDI_MIN_EDGE_SPAN_CELSIUS,
    TVDI_MIN_PIXELS_PER_BIN,
    TVDI_NDVI_BIN_COUNT,
    TVDI_NDVI_MAX,
    TVDI_NDVI_MIN,
    TVDI_WET_EDGE_PERCENTILE,
    TVDI_ZSCORE_NUMERICAL_EPSILON,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

# Step5'in windowed yardımcılarını yeniden kullan (kod tekrarı yok, aynı mantık).
from src.step5_preprocess_timeseries import (
    count_windows,
    extract_date_from_filename,
    iter_windows,
    mask_physical_celsius,
    open_output,
    output_profile,
    read_band_window,
    validate_same_grid,
    RunningStats,
)


BASE_DIR = PROJECT_ROOT
STEP5_OUTPUT_DIR = BASE_DIR / "outputs" / "step5"
NDVI_BASELINE_DIR = BASE_DIR / "data" / "ndvi_timeseries"
NDVI_CURRENT_DIR = BASE_DIR / "data" / "ndvi_current_period"
OUTPUT_DIR = BASE_DIR / "outputs" / "step5c"

NDVI_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
NDVI_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step5c")

# Step5'in ürettiği LST rasterları (girdi olarak okunur, üzerine yazılmaz).
CURRENT_LST_PATH = STEP5_OUTPUT_DIR / "current_period_median_celsius.tif"
BASELINE_LST_MEAN_PATH = STEP5_OUTPUT_DIR / "baseline_lst_mean_celsius.tif"

# Step5 ana raster çıktısının window kenarı; aynı tiling güvenliğini korumak için
# Step5 ile aynı pencere boyutunu kullanırız.
from core.config import STEP5_WINDOW_SIZE  # noqa: E402  (config grubunu bölmemek için)


def mask_valid_ndvi(array: np.ndarray) -> np.ndarray:
    """Fiziksel/geçerli NDVI aralığı dışındaki değerleri NaN yapar."""
    return np.where(
        (array >= TVDI_NDVI_MIN) & (array <= TVDI_NDVI_MAX),
        array,
        np.nan,
    ).astype("float32")


def list_current_ndvi_tif() -> Path:
    """Current period NDVI median GeoTIFF'ini bulur."""
    candidates = sorted(NDVI_CURRENT_DIR.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"Current NDVI median dosyası bulunamadı: {NDVI_CURRENT_DIR}\n"
            "Step4 current_ndvi_median.tif export'unu Step4b buraya koymalı."
        )
    preferred = [
        path for path in candidates
        if path.name.lower().startswith("current_ndvi_median")
    ]
    if preferred:
        return preferred[0]
    log.warning(
        "current_ndvi_median ile eşleşen dosya bulunamadı; ilk NDVI dosyası kullanılıyor: %s",
        candidates[0].name,
    )
    return candidates[0]


def list_baseline_ndvi_tifs() -> list[Path]:
    """Baseline pencere-simetrik NDVI median GeoTIFF'lerini listeler."""
    tifs = sorted(NDVI_BASELINE_DIR.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(
            f"Baseline NDVI GeoTIFF bulunamadı: {NDVI_BASELINE_DIR}\n"
            "Step4 NDVI baseline timeseries export'unu Step4b buraya koymalı."
        )
    return tifs


def ndvi_bin_edges() -> np.ndarray:
    """NDVI eksenini eşit genişlikli bin sınırlarına böler."""
    return np.linspace(TVDI_NDVI_MIN, TVDI_NDVI_MAX, TVDI_NDVI_BIN_COUNT + 1)


def assign_ndvi_bins(ndvi: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Her NDVI değerini bir bin indeksine atar (geçersiz/aralık dışı = -1).

    np.digitize ile [edges[i], edges[i+1]) yarı-açık bin'leri kullanılır.
    """
    bins = np.full(ndvi.shape, -1, dtype="int32")
    finite = np.isfinite(ndvi)
    idx = np.digitize(ndvi[finite], edges) - 1
    idx = np.clip(idx, 0, TVDI_NDVI_BIN_COUNT - 1)
    bins[finite] = idx
    return bins


def collect_bin_lst_samples(
    current_lst_path: Path,
    current_ndvi_path: Path,
    edges: np.ndarray,
) -> dict[int, list[np.ndarray]]:
    """
    Birinci geçiş: current LST ve current NDVI'yi windowed okuyup her NDVI bin'i
    için LST örneklerini biriktirir.

    Tüm rasterı belleğe almaz; pencere pencere ilerler ve yalnız geçerli (LST, NDVI)
    çiftlerini ilgili bin listesine ekler. Edge percentile hesabı için bu örnekler
    yeterlidir.
    """
    bin_samples: dict[int, list[np.ndarray]] = {i: [] for i in range(TVDI_NDVI_BIN_COUNT)}

    with rasterio.open(current_lst_path) as lst_src:
        profile = lst_src.profile.copy()
        width = profile["width"]
        height = profile["height"]

        validate_same_grid(profile, current_ndvi_path)

        with rasterio.open(current_ndvi_path) as ndvi_src:
            for window in iter_windows(width, height, STEP5_WINDOW_SIZE):
                lst = mask_physical_celsius(read_band_window(lst_src, window, band_index=1))
                ndvi = mask_valid_ndvi(read_band_window(ndvi_src, window, band_index=1))

                valid = np.isfinite(lst) & np.isfinite(ndvi)
                if not np.any(valid):
                    continue

                lst_valid = lst[valid]
                ndvi_valid = ndvi[valid]
                bin_idx = assign_ndvi_bins(ndvi_valid, edges)

                for b in range(TVDI_NDVI_BIN_COUNT):
                    sel = bin_idx == b
                    if np.any(sel):
                        bin_samples[b].append(lst_valid[sel].astype("float32"))

    return bin_samples


def compute_edges_from_samples(
    bin_samples: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Her NDVI bin'i için wet_edge (düşük LST percentile) ve dry_edge (yüksek LST
    percentile) hesaplar.

    Yeterli piksele sahip olmayan bin'ler NaN bırakılır. Dönüş:
        (wet_edges, dry_edges, bin_records)
    """
    wet_edges = np.full(TVDI_NDVI_BIN_COUNT, np.nan, dtype="float64")
    dry_edges = np.full(TVDI_NDVI_BIN_COUNT, np.nan, dtype="float64")
    bin_records = []

    for b in range(TVDI_NDVI_BIN_COUNT):
        chunks = bin_samples.get(b, [])
        pixel_count = int(sum(chunk.size for chunk in chunks))

        record = {"bin": b, "pixel_count": pixel_count}

        if pixel_count < TVDI_MIN_PIXELS_PER_BIN:
            record["status"] = "insufficient_pixels"
            bin_records.append(record)
            continue

        values = np.concatenate(chunks)
        wet = float(np.percentile(values, TVDI_WET_EDGE_PERCENTILE))
        dry = float(np.percentile(values, TVDI_DRY_EDGE_PERCENTILE))

        wet_edges[b] = wet
        dry_edges[b] = dry
        record.update({
            "wet_edge": wet,
            "dry_edge": dry,
            "edge_span": dry - wet,
            "status": "ok",
        })
        bin_records.append(record)

    return wet_edges, dry_edges, bin_records


def tvdi_from_lst_ndvi(
    lst: np.ndarray,
    ndvi: np.ndarray,
    edges: np.ndarray,
    wet_edges: np.ndarray,
    dry_edges: np.ndarray,
) -> np.ndarray:
    """
    Verilen LST ve NDVI pencereleri için TVDI hesaplar.

    TVDI = (LST - wet_edge) / (dry_edge - wet_edge), [0, 1] aralığına clamp.
    Edge span eşiğin altındaysa veya bin geçersizse NaN bırakılır.
    """
    out = np.full(lst.shape, np.nan, dtype="float32")
    valid = np.isfinite(lst) & np.isfinite(ndvi)
    if not np.any(valid):
        return out

    flat_idx = np.where(valid)
    lst_valid = lst[valid]
    ndvi_valid = ndvi[valid]
    bin_idx = assign_ndvi_bins(ndvi_valid, edges)

    wet = wet_edges[bin_idx]
    dry = dry_edges[bin_idx]
    span = dry - wet

    edge_ok = np.isfinite(wet) & np.isfinite(dry) & (span >= TVDI_MIN_EDGE_SPAN_CELSIUS)

    with np.errstate(invalid="ignore", divide="ignore"):
        tvdi_valid = (lst_valid - wet) / span
    tvdi_valid = np.where(edge_ok, tvdi_valid, np.nan)
    tvdi_valid = np.clip(tvdi_valid, 0.0, 1.0)

    out[flat_idx] = tvdi_valid.astype("float32")
    return out


def compute_current_tvdi(
    current_lst_path: Path,
    current_ndvi_path: Path,
    edges: np.ndarray,
    wet_edges: np.ndarray,
    dry_edges: np.ndarray,
) -> tuple[Path, dict]:
    """
    İkinci geçiş (current): windowed olarak current TVDI rasterını yazar.

    Ayrıca current_tvdi_valid_count.tif üretir. Current NDVI median rasterında
    valid-count bandı (Current_Period_NDVI_Valid_Count, bant 2) varsa o kullanılır;
    yoksa TVDI'nin pikselde geçerli olup olmadığı (1/0) yazılır. Bu, current TVDI'ye
    kaç QA-temiz gözlemin katkı verdiğini yansıtır.
    """
    output_path = OUTPUT_DIR / "current_tvdi.tif"
    count_path = OUTPUT_DIR / "current_tvdi_valid_count.tif"
    stats = RunningStats()
    count_stats = RunningStats()

    with rasterio.open(current_lst_path) as lst_src:
        profile = lst_src.profile.copy()
        width = profile["width"]
        height = profile["height"]

        with rasterio.open(current_ndvi_path) as ndvi_src:
            has_ndvi_count_band = ndvi_src.count >= 2

        with (
            rasterio.open(current_ndvi_path) as ndvi_src,
            open_output(output_path, profile) as tvdi_dst,
            open_output(count_path, profile) as count_dst,
        ):
            for window in iter_windows(width, height, STEP5_WINDOW_SIZE):
                lst = mask_physical_celsius(read_band_window(lst_src, window, band_index=1))
                ndvi = mask_valid_ndvi(read_band_window(ndvi_src, window, band_index=1))
                tvdi = tvdi_from_lst_ndvi(lst, ndvi, edges, wet_edges, dry_edges)
                tvdi_dst.write(tvdi, 1, window=window)
                stats.update(tvdi)

                if has_ndvi_count_band:
                    valid_count = read_band_window(
                        ndvi_src, window, band_index=2
                    ).astype("float32")
                    # TVDI'nin NaN olduğu yerde valid_count'u 0'a düşür (tutarlılık).
                    valid_count = np.where(np.isfinite(tvdi), valid_count, 0.0).astype("float32")
                else:
                    valid_count = np.where(np.isfinite(tvdi), 1.0, 0.0).astype("float32")

                count_dst.write(valid_count, 1, window=window)
                count_stats.update(valid_count)

    result_stats = stats.as_dict()
    result_stats["current_tvdi_valid_count"] = str(count_path)
    result_stats["current_tvdi_valid_count_stats"] = count_stats.as_dict()
    return output_path, result_stats


def compute_baseline_tvdi(
    baseline_lst_mean_path: Path,
    baseline_ndvi_tifs: list[Path],
    edges: np.ndarray,
    wet_edges: np.ndarray,
    dry_edges: np.ndarray,
) -> dict:
    """
    Baseline TVDI mean/std ve current TVDI z-score'unu üretir.

    Her baseline yılı NDVI penceresi için TVDI hesaplanır (baseline LST mean ile
    eşleştirilerek). Yıllar arası mean/std windowed olarak biriktirilir.

    NOT (basit/ilk sürüm sınırı):
        Baseline LST için Step5'in tek baseline_lst_mean rasterı kullanılır; her
        baseline yılına ait ayrı LST median rasterı Step5 tarafından dışa yazılmaz.
        Bu yüzden baseline TVDI varyasyonu NDVI'nin yıldan yıla değişiminden gelir;
        LST baseline ortalaması sabit tutulur. Phase 2'de yıllık LST median rasterları
        eklenirse buraya tam (LST_year, NDVI_year) eşleşmesi entegre edilebilir.
    """
    times = [extract_date_from_filename(path) for path in baseline_ndvi_tifs]
    order = np.argsort(times)
    baseline_ndvi_tifs = [baseline_ndvi_tifs[i] for i in order]

    with rasterio.open(baseline_lst_mean_path) as lst_src:
        profile = lst_src.profile.copy()
        width = profile["width"]
        height = profile["height"]

        for ndvi_path in baseline_ndvi_tifs:
            validate_same_grid(profile, ndvi_path)

        mean_path = OUTPUT_DIR / "baseline_tvdi_mean.tif"
        std_path = OUTPUT_DIR / "baseline_tvdi_std.tif"
        count_path = OUTPUT_DIR / "baseline_tvdi_valid_count.tif"
        zscore_path = OUTPUT_DIR / "tvdi_anomaly_zscore.tif"
        difference_path = OUTPUT_DIR / "tvdi_difference.tif"

        mean_stats = RunningStats()
        std_stats = RunningStats()
        zscore_stats = RunningStats()
        difference_stats = RunningStats()

        current_tvdi_path = OUTPUT_DIR / "current_tvdi.tif"

        # Düşük baseline std nedeniyle z-score'u maskelenen piksel sayacı.
        low_std_masked_count = 0
        # z-score adayı olabilecek (yeterli gözlem + geçerli current/baseline) piksel sayısı.
        zscore_candidate_count = 0

        baseline_ndvi_srcs = [rasterio.open(path) for path in baseline_ndvi_tifs]
        try:
            with (
                rasterio.open(current_tvdi_path) as current_tvdi_src,
                open_output(mean_path, profile) as mean_dst,
                open_output(std_path, profile) as std_dst,
                open_output(count_path, profile) as count_dst,
                open_output(zscore_path, profile) as zscore_dst,
                open_output(difference_path, profile) as difference_dst,
            ):
                for window in iter_windows(width, height, STEP5_WINDOW_SIZE):
                    lst_mean = mask_physical_celsius(
                        read_band_window(lst_src, window, band_index=1)
                    )

                    tvdi_stack = np.empty(
                        (len(baseline_ndvi_srcs), int(window.height), int(window.width)),
                        dtype="float32",
                    )
                    for i, ndvi_src in enumerate(baseline_ndvi_srcs):
                        ndvi = mask_valid_ndvi(
                            read_band_window(ndvi_src, window, band_index=1)
                        )
                        tvdi_stack[i] = tvdi_from_lst_ndvi(
                            lst_mean, ndvi, edges, wet_edges, dry_edges
                        )

                    valid_count = np.sum(np.isfinite(tvdi_stack), axis=0).astype("float32")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        baseline_mean = np.nanmean(tvdi_stack, axis=0).astype("float32")
                        baseline_std = np.nanstd(tvdi_stack, axis=0).astype("float32")

                    enough = valid_count >= TVDI_MIN_BASELINE_VALID_COUNT
                    baseline_mean = np.where(enough, baseline_mean, np.nan).astype("float32")
                    baseline_std = np.where(enough, baseline_std, np.nan).astype("float32")

                    current_tvdi = read_band_window(
                        current_tvdi_src, window, band_index=1
                    )

                    # Ham fark ürünü: yorumlanması z-score'dan daha kolay.
                    difference = (current_tvdi - baseline_mean).astype("float32")

                    # z-score adayı: yeterli gözlem + geçerli current + geçerli baseline.
                    candidate = (
                        enough
                        & np.isfinite(current_tvdi)
                        & np.isfinite(baseline_mean)
                        & np.isfinite(baseline_std)
                    )
                    # Güvenilirlik maskesi: düşük baseline std z-score'u şişirir.
                    reliable_std = candidate & (baseline_std >= MIN_TVDI_BASELINE_STD)
                    # Yalnız düşük-std yüzünden elenen pikseller (sayım için).
                    low_std_only = candidate & (baseline_std < MIN_TVDI_BASELINE_STD)

                    low_std_masked_count += int(np.sum(low_std_only))
                    zscore_candidate_count += int(np.sum(candidate))

                    with np.errstate(invalid="ignore", divide="ignore"):
                        zscore = np.where(
                            reliable_std,
                            (current_tvdi - baseline_mean)
                            / (baseline_std + TVDI_ZSCORE_NUMERICAL_EPSILON),
                            np.nan,
                        ).astype("float32")

                    mean_dst.write(baseline_mean, 1, window=window)
                    std_dst.write(baseline_std, 1, window=window)
                    count_dst.write(valid_count, 1, window=window)
                    zscore_dst.write(zscore, 1, window=window)
                    difference_dst.write(difference, 1, window=window)

                    mean_stats.update(baseline_mean)
                    std_stats.update(baseline_std)
                    zscore_stats.update(zscore)
                    difference_stats.update(difference)
        finally:
            for src in baseline_ndvi_srcs:
                src.close()

    return {
        "baseline_tvdi_mean": str(mean_path),
        "baseline_tvdi_std": str(std_path),
        "baseline_tvdi_valid_count": str(count_path),
        "tvdi_difference": str(difference_path),
        "tvdi_anomaly_zscore": str(zscore_path),
        "baseline_tvdi_mean_stats": mean_stats.as_dict(),
        "baseline_tvdi_std_stats": std_stats.as_dict(),
        "tvdi_difference_stats": difference_stats.as_dict(),
        "tvdi_zscore_stats": zscore_stats.as_dict(),
        "baseline_window_count": len(baseline_ndvi_tifs),
        "low_tvdi_std_masked_pixel_count": low_std_masked_count,
        "tvdi_zscore_candidate_pixel_count": zscore_candidate_count,
        "low_tvdi_std_masked_ratio": (
            float(low_std_masked_count / zscore_candidate_count)
            if zscore_candidate_count > 0
            else None
        ),
    }


def write_metadata(metadata: dict) -> Path:
    """Step5c TVDI metadata JSON dosyasını yazar."""
    metadata_path = OUTPUT_DIR / "step5c_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path


def main() -> dict | None:
    """TVDI pipeline'ını çalıştırır. Mevcut Step5/Step5B çıktılarını değiştirmez."""
    log.info("=" * 60)
    log.info("STEP 5C BAŞLIYOR (TVDI - basit/ilk sürüm)")
    log.info("=" * 60)

    if not ENABLE_TVDI_STEP5:
        log.warning("ENABLE_TVDI_STEP5=False; TVDI pipeline atlandı.")
        return None

    if not CURRENT_LST_PATH.exists():
        raise FileNotFoundError(
            f"Current LST median bulunamadı: {CURRENT_LST_PATH}\n"
            "Önce Step5 çalıştırılmalı."
        )
    if not BASELINE_LST_MEAN_PATH.exists():
        raise FileNotFoundError(
            f"Baseline LST mean bulunamadı: {BASELINE_LST_MEAN_PATH}\n"
            "Önce Step5 çalıştırılmalı."
        )

    current_ndvi_path = list_current_ndvi_tif()
    baseline_ndvi_tifs = list_baseline_ndvi_tifs()

    log.info("Current LST: %s", CURRENT_LST_PATH.name)
    log.info("Current NDVI: %s", current_ndvi_path.name)
    log.info("Baseline NDVI pencere sayısı: %s", len(baseline_ndvi_tifs))

    edges = ndvi_bin_edges()

    # 1. geçiş: current scatter'dan NDVI bin edge'leri
    log.info("Birinci geçiş: LST-NDVI scatter edge örnekleri toplanıyor.")
    bin_samples = collect_bin_lst_samples(CURRENT_LST_PATH, current_ndvi_path, edges)
    wet_edges, dry_edges, bin_records = compute_edges_from_samples(bin_samples)

    valid_bins = int(np.sum(np.isfinite(wet_edges) & np.isfinite(dry_edges)))
    log.info("Geçerli edge üretilen NDVI bin sayısı: %s/%s", valid_bins, TVDI_NDVI_BIN_COUNT)
    if valid_bins == 0:
        raise ValueError(
            "Hiçbir NDVI bin'i için geçerli wet/dry edge üretilemedi. "
            "TVDI_MIN_PIXELS_PER_BIN veya girdi kapsamı kontrol edilmeli."
        )

    # 2. geçiş (current): current_tvdi.tif
    log.info("İkinci geçiş: current TVDI hesaplanıyor.")
    current_tvdi_path, current_tvdi_stats = compute_current_tvdi(
        CURRENT_LST_PATH, current_ndvi_path, edges, wet_edges, dry_edges
    )

    # Baseline TVDI mean/std + z-score
    log.info("Baseline TVDI mean/std ve z-score hesaplanıyor.")
    baseline_result = compute_baseline_tvdi(
        BASELINE_LST_MEAN_PATH,
        baseline_ndvi_tifs,
        edges,
        wet_edges,
        dry_edges,
    )

    window_count = None
    with rasterio.open(CURRENT_LST_PATH) as src:
        window_count = count_windows(src.width, src.height, STEP5_WINDOW_SIZE)

    metadata = {
        "step": "step5c_tvdi",
        "method": "ndvi_binned_dry_wet_edge_tvdi",
        "created_at": datetime.now().isoformat(),
        "log_file": str(log_file),
        "inputs": {
            "current_lst": str(CURRENT_LST_PATH),
            "baseline_lst_mean": str(BASELINE_LST_MEAN_PATH),
            "current_ndvi": str(current_ndvi_path),
            "baseline_ndvi_dir": str(NDVI_BASELINE_DIR),
            "baseline_ndvi_files": [p.name for p in baseline_ndvi_tifs],
        },
        "processing": {
            "mode": "windowed",
            "window_size": STEP5_WINDOW_SIZE,
            "window_count": window_count,
            "ndvi_bin_count": TVDI_NDVI_BIN_COUNT,
            "ndvi_range": [TVDI_NDVI_MIN, TVDI_NDVI_MAX],
            "wet_edge_percentile": TVDI_WET_EDGE_PERCENTILE,
            "dry_edge_percentile": TVDI_DRY_EDGE_PERCENTILE,
            "min_pixels_per_bin": TVDI_MIN_PIXELS_PER_BIN,
            "min_edge_span_celsius": TVDI_MIN_EDGE_SPAN_CELSIUS,
            "tvdi_anomaly_formula": (
                "tvdi_anomaly_zscore = (current_tvdi - baseline_tvdi_mean) "
                "/ baseline_tvdi_std; masked where baseline_tvdi_std < "
                "MIN_TVDI_BASELINE_STD"
            ),
            "min_tvdi_baseline_std": MIN_TVDI_BASELINE_STD,
            "zscore_numerical_epsilon": TVDI_ZSCORE_NUMERICAL_EPSILON,
            "low_tvdi_std_masked_pixel_count": baseline_result.get(
                "low_tvdi_std_masked_pixel_count"
            ),
            "low_tvdi_std_masked_ratio": baseline_result.get(
                "low_tvdi_std_masked_ratio"
            ),
            "tvdi_zscore_candidate_pixel_count": baseline_result.get(
                "tvdi_zscore_candidate_pixel_count"
            ),
            "min_baseline_valid_count": TVDI_MIN_BASELINE_VALID_COUNT,
            "baseline_tvdi_valid_count_policy": (
                "Her piksel için baseline yıllarından kaçında geçerli TVDI "
                "üretildiği sayılır; valid_count < min_baseline_valid_count olan "
                "pikseller baseline mean/std ve z-score'da NaN bırakılır."
            ),
            "current_tvdi_valid_count_policy": (
                "Current NDVI median rasterının valid-count bandı varsa kullanılır; "
                "yoksa TVDI'nin geçerli olduğu pikseller 1, değilse 0 yazılır."
            ),
            "temporal_interpolation_used": False,
            "insufficient_observations_policy": "mask_as_nan",
        },
        "edges": {
            "valid_bin_count": valid_bins,
            "bin_records": bin_records,
        },
        "current_period": {
            "window_days": CURRENT_PERIOD_DAYS,
        },
        "outputs": {
            "current_tvdi": str(current_tvdi_path),
            "current_tvdi_stats": current_tvdi_stats,
            **baseline_result,
        },
        "notes": {
            "baseline_lst_source": (
                "Step5 tek baseline_lst_mean rasterı; yıllık LST median rasterları "
                "Phase 2'de eklenebilir."
            ),
            "does_not_modify": [
                "outputs/step5/*",
                "outputs/diagnostics/*",
            ],
        },
        "status": "tvdi_processed",
    }
    metadata_path = write_metadata(metadata)

    log.info("Current TVDI ortalaması: %.3f", current_tvdi_stats.get("mean") or float("nan"))
    log.info("TVDI z-score ortalaması: %.3f", baseline_result["tvdi_zscore_stats"].get("mean") or float("nan"))
    log.info("Metadata: %s", metadata_path)
    log.info("Çıktı klasörü: %s", OUTPUT_DIR)
    log.info("=" * 60)
    log.info("STEP 5C TAMAMLANDI")

    return metadata


if __name__ == "__main__":
    main()