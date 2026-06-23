"""
step6_validate_fire_relation.py

İLK burned-area ilişki (association) testi.

Bu modül bir yangın-riski MODELİ DEĞİLDİR ve RF/XGBoost eğitmez. Amacı, mevcut
predictor rasterlarının yanmış alan / aktif yangın etiketleriyle ne kadar ilişkili
olduğunu ölçmektir (burned vs unburned ayrışması, ROC/AUC).

Predictor rasterları:
    - outputs/step5/thermal_anomaly_zscore.tif   (LST sıcaklık anomalisi)
    - outputs/step5c/current_tvdi.tif            (sürekli kuruluk göstergesi)
    - outputs/step5c/tvdi_difference.tif         (current - baseline mean)
    - outputs/step5c/tvdi_anomaly_zscore.tif     (güvenilirlik-filtreli anomali)

Etiketler (GEE):
    - MCD64A1 burned area (500 m)
    - FireCCI51 burned area (250 m, varsa)
    - opsiyonel: FIRMS / MCD14ML aktif yangın

Önemli yorum notları:
    - current_tvdi ve tvdi_difference SÜREKLİ kuruluk göstergeleri olarak ele alınır.
    - tvdi_anomaly_zscore güvenilirlik-filtreli ve yoğun maskelidir; geçerli örnek
      sayısı ayrı raporlanır.
    - Sonuç "doğrulanmış yangın-riski modeli" DEĞİLDİR; ilk ilişki testidir.

Çıktılar:
    - outputs/validation/validation_summary.md
    - outputs/validation/validation_stats.json
    - outputs/validation/roc_curve_comparison.png
    - outputs/validation/burned_vs_unburned_boxplot.png
    - outputs/validation/predictor_maps_with_burn_overlay.png (mümkünse)
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from core.config import (
    EXPORT_CRS,
    FIRECCI51_AVAILABLE_END,
    FIRECCI51_AVAILABLE_START,
    GEE_PROJECT,
    REGION_NAME,
    VALIDATION_BALANCED_UNBURNED_RATIO,
    VALIDATION_FIRMS_BRIGHTNESS_THRESHOLD,
    VALIDATION_INCLUDE_FIRMS,
    VALIDATION_LABEL_EXPORT_SCALE,
    VALIDATION_MAX_ROC_PREVIEW_POINTS,
    VALIDATION_MODE,
    VALIDATION_PREFIRE_LABEL_END,
    VALIDATION_PREFIRE_LABEL_START,
    VALIDATION_PREFIRE_PREDICTOR_END,
    VALIDATION_PREFIRE_PREDICTOR_START,
    VALIDATION_RANDOM_SEED,
    VALIDATION_SEASON_END,
    VALIDATION_SEASON_START,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import ee
    from core.gee_utils import init_gee
    from core.regions import build_regions
    from core.validation_burned_area import (
        get_firecci51_burned_area_safe,
        get_firms_active_fire,
        get_mcd64a1_burned_area_safe,
    )
    GEE_IMPORTS_OK = True
    GEE_IMPORT_ERROR = None
except Exception as _gee_import_exc:  # noqa: BLE001
    # ImportError dışındaki hataları da yakala (ör. validation_burned_area
    # içindeki bir bağımlılık veya init hatası). Gerçek sebebi sakla ki
    # fetch_labels net mesaj verebilsin.
    GEE_IMPORTS_OK = False
    GEE_IMPORT_ERROR = f"{type(_gee_import_exc).__name__}: {_gee_import_exc}"

try:
    import geemap
    GEEMAP_AVAILABLE = True
except ImportError:
    GEEMAP_AVAILABLE = False


BASE_DIR = PROJECT_ROOT
STEP5_OUTPUT_DIR = BASE_DIR / "outputs" / "step5"
STEP5C_OUTPUT_DIR = BASE_DIR / "outputs" / "step5c"
OUTPUT_DIR = BASE_DIR / "outputs" / "validation"
LABEL_DIR = OUTPUT_DIR / "labels"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step6")

# Predictor raster tanımları. LST anomaly için esnek dosya adı (thermal_anomaly_zscore
# veya anomaly_zscore) desteklenir; ilk bulunan kullanılır.
PREDICTORS = {
    "thermal_anomaly_zscore": {
        "path_candidates": [
            STEP5_OUTPUT_DIR / "thermal_anomaly_zscore.tif",
            STEP5_OUTPUT_DIR / "anomaly_zscore.tif",
        ],
        "label": "LST anomaly z-score",
        "continuous": True,
    },
    "current_tvdi": {
        "path_candidates": [STEP5C_OUTPUT_DIR / "current_tvdi.tif"],
        "label": "Current TVDI",
        "continuous": True,
    },
    "tvdi_difference": {
        "path_candidates": [STEP5C_OUTPUT_DIR / "tvdi_difference.tif"],
        "label": "TVDI difference",
        "continuous": True,
    },
    "tvdi_anomaly_zscore": {
        "path_candidates": [STEP5C_OUTPUT_DIR / "tvdi_anomaly_zscore.tif"],
        "label": "TVDI anomaly z-score (reliability-filtered)",
        "continuous": False,
    },
}


def resolve_predictor_path(key: str) -> Path | None:
    """Predictor için ilk var olan aday dosya yolunu döndürür (yoksa None)."""
    for candidate in PREDICTORS[key]["path_candidates"]:
        if candidate.exists():
            return candidate
    return None


class ValidationError(Exception):
    """Step6'da net hata mesajıyla durmak için."""


def reference_predictor_path() -> Path:
    """Etiketlerin hizalanacağı referans predictor grid'ini seçer."""
    # current_tvdi en yüksek geçerli kapsama sahip TVDI ürünü; referans grid o.
    for key in ("current_tvdi", "thermal_anomaly_zscore"):
        path = resolve_predictor_path(key)
        if path is not None:
            return path
    raise ValidationError(
        "Hiçbir referans predictor rasterı bulunamadı. Önce Step5 ve Step5C "
        "çalıştırılmalı (current_tvdi.tif veya thermal_anomaly_zscore.tif gerekli)."
    )


def read_predictor(path: Path) -> np.ndarray:
    """Tek bantlı predictor rasterını float32 + NaN olarak okur."""
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype("float32").filled(np.nan)


def read_reference_grid(path: Path) -> dict:
    """Referans grid profilini (transform, crs, shape) döndürür."""
    with rasterio.open(path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "profile": src.profile.copy(),
        }


def export_label_to_grid(
    image: "ee.Image",
    region: "ee.Geometry",
    grid: dict,
    out_path: Path,
    label_name: str,
) -> Path | None:
    """
    GEE binary etiket image'ini referans predictor grid'ine indirir/resample eder.

    geemap ile etiket GeoTIFF olarak indirilir; ardından rasterio ile predictor
    grid'ine (nearest, kategorik etiket) reproject edilir. Başarısızsa None döner.
    """
    if not GEEMAP_AVAILABLE:
        log.warning("geemap yok; %s etiketi indirilemedi.", label_name)
        return None

    raw_path = LABEL_DIR / f"{label_name}_raw.tif"
    try:
        geemap.ee_export_image(
            image,
            filename=str(raw_path),
            scale=VALIDATION_LABEL_EXPORT_SCALE,
            region=region,
            crs=EXPORT_CRS,
            file_per_band=False,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("%s etiketi GEE export başarısız: %s", label_name, exc)
        return None

    if not raw_path.exists():
        log.warning("%s etiketi indirilemedi (dosya yok).", label_name)
        return None

    # Predictor grid'ine resample (nearest; etiket kategorik 0/1).
    aligned = np.zeros((grid["height"], grid["width"]), dtype="float32")
    with rasterio.open(raw_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            resampling=Resampling.nearest,
        )

    profile = grid["profile"]
    profile.update(count=1, dtype="float32", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(aligned, 1)

    return out_path


def build_binary_label(
    label_arrays: list[np.ndarray],
) -> np.ndarray:
    """
    Birden çok etiket kaynağını tek binary 'burned' maskesine birleştirir (OR).

    burned = 1, unburned = 0. Hiç kaynak yoksa tümü 0 döner.
    """
    if not label_arrays:
        return None
    combined = np.zeros_like(label_arrays[0], dtype="float32")
    for arr in label_arrays:
        combined = np.maximum(combined, (arr > 0).astype("float32"))
    return combined


def firecci51_window_available(label_start: str, label_end: str) -> bool:
    """
    İstenen label penceresinin FireCCI51 veri kapsamında olup olmadığını kontrol
    eder. Kapsam dışındaysa (örn. 2023) FireCCI51 Earth Engine'e SORULMADAN skip
    edilir.
    """
    # Basit string tabanlı tarih karşılaştırması (ISO formatı sıralanabilir).
    return (
        label_start <= FIRECCI51_AVAILABLE_END
        and label_end >= FIRECCI51_AVAILABLE_START
    )


def fetch_labels(grid: dict, label_start: str, label_end: str) -> dict:
    """
    GEE'den yanmış alan / aktif yangın etiketlerini indirip predictor grid'ine
    hizalar ve tek binary etiket üretir.

    label_start/label_end: etiketlerin çekileceği pencere. same_season modunda
    sezon penceresiyle, pre_fire modunda label window ile aynıdır.
    """
    if not GEE_IMPORTS_OK:
        raise ValidationError(
            "GEE importları başarısız (ee/geemap/regions/validation_burned_area). "
            "Step6 etiket indirmesi için GEE ortamı gerekli. "
            f"Gerçek import hatası: {GEE_IMPORT_ERROR}. "
            "Olası çözümler: (1) earthengine-api ve geemap kurun "
            "(pip install earthengine-api geemap), (2) GEE auth yapın "
            "(earthengine authenticate), (3) hata bir modül içi soruna işaret "
            "ediyorsa ilgili modülü kontrol edin."
        )

    try:
        init_gee(GEE_PROJECT)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            "GEE başlatma/auth başarısız (ee.Initialize). "
            f"Hata: {type(exc).__name__}: {exc}. "
            "Çözüm: 'earthengine authenticate' çalıştırın ve GEE_PROJECT "
            f"değerinin ('{GEE_PROJECT}') doğru/erişilebilir olduğundan emin olun."
        ) from exc
    regions = build_regions()
    if REGION_NAME not in regions:
        raise ValidationError(f"Bölge bulunamadı: {REGION_NAME}")
    region = regions[REGION_NAME]

    start = label_start
    end = label_end
    log.info("Etiket penceresi: %s -> %s, bölge: %s", start, end, REGION_NAME)

    label_arrays = []
    sources_used = []
    skipped_sources = []

    # MCD64A1 (ana kaynak, güvenli)
    mcd, mcd_status = get_mcd64a1_burned_area_safe(region, start, end)
    if mcd is None:
        log.warning(mcd_status)
        skipped_sources.append({"source": "MCD64A1", "reason": mcd_status})
    else:
        mcd_path = export_label_to_grid(
            mcd, region, grid, LABEL_DIR / "mcd64a1_burned.tif", "mcd64a1"
        )
        if mcd_path is not None:
            with rasterio.open(mcd_path) as src:
                label_arrays.append(src.read(1).astype("float32"))
            sources_used.append("MCD64A1")
        else:
            reason = "MCD64A1 GEE export/download failed."
            log.warning(reason)
            skipped_sources.append({"source": "MCD64A1", "reason": reason})

    # FireCCI51: önce availability kontrolü. Label window kapsam dışıysa (örn. 2023)
    # Earth Engine'e hiç sorulmadan skip edilir.
    if not firecci51_window_available(start, end):
        reason = (
            "FireCCI51 skipped: requested label window outside dataset "
            f"availability ({FIRECCI51_AVAILABLE_START} - "
            f"{FIRECCI51_AVAILABLE_END})."
        )
        log.info(reason)
        skipped_sources.append({"source": "FireCCI51", "reason": reason})
    else:
        # Window uygunsa güvenli fetch dene (boş/bandsiz image .gt() çağrılmadan elenir)
        try:
            firecci, firecci_status = get_firecci51_burned_area_safe(
                region, start, end
            )
            if firecci is None:
                log.warning(firecci_status)
                skipped_sources.append(
                    {"source": "FireCCI51", "reason": firecci_status}
                )
            else:
                firecci_path = export_label_to_grid(
                    firecci, region, grid,
                    LABEL_DIR / "firecci51_burned.tif", "firecci51",
                )
                if firecci_path is not None:
                    with rasterio.open(firecci_path) as src:
                        label_arrays.append(src.read(1).astype("float32"))
                    sources_used.append("FireCCI51")
                else:
                    reason = "FireCCI51 GEE export/download failed."
                    log.warning(reason)
                    skipped_sources.append(
                        {"source": "FireCCI51", "reason": reason}
                    )
        except Exception as exc:  # noqa: BLE001
            reason = f"FireCCI51 unexpected error: {exc}"
            log.warning(reason)
            skipped_sources.append({"source": "FireCCI51", "reason": reason})
        skipped_sources.append({"source": "FireCCI51", "reason": reason})

    # FIRMS aktif yangın (opsiyonel)
    if VALIDATION_INCLUDE_FIRMS:
        try:
            firms = get_firms_active_fire(region, start, end)
            firms_binary = firms.gt(VALIDATION_FIRMS_BRIGHTNESS_THRESHOLD)
            firms_path = export_label_to_grid(
                firms_binary, region, grid, LABEL_DIR / "firms_active.tif", "firms"
            )
            if firms_path is not None:
                with rasterio.open(firms_path) as src:
                    label_arrays.append(src.read(1).astype("float32"))
                sources_used.append("FIRMS")
            else:
                reason = "FIRMS GEE export/download failed."
                log.warning(reason)
                skipped_sources.append({"source": "FIRMS", "reason": reason})
        except Exception as exc:  # noqa: BLE001
            reason = f"FIRMS unexpected error: {exc}"
            log.warning(reason)
            skipped_sources.append({"source": "FIRMS", "reason": reason})

    burned = build_binary_label(label_arrays)
    if burned is None:
        raise ValidationError(
            "Hiçbir yanmış alan etiketi indirilemedi. GEE bağlantısını ve "
            "geemap kurulumunu kontrol edin. "
            f"Atlanan kaynaklar: {skipped_sources}"
        )

    burned_count = int(np.sum(burned > 0))
    if burned_count == 0:
        raise ValidationError(
            f"Seçili AOI ({REGION_NAME}) ve sezonda ({start} - {end}) hiç yanmış "
            "piksel bulunamadı. Öneriler: (1) AOI'yi genişletin, (2) farklı/daha "
            "geniş bir yangın sezonu seçin (VALIDATION_SEASON_START/END), "
            "(3) FIRMS aktif yangını dahil edin (VALIDATION_INCLUDE_FIRMS=True)."
        )

    log.info(
        "Yanmış piksel sayısı: %s (kullanılan kaynaklar: %s)",
        burned_count, sources_used,
    )
    if skipped_sources:
        log.info("Atlanan kaynaklar: %s", [s["source"] for s in skipped_sources])

    return {
        "burned": burned,
        "sources_used": sources_used,
        "skipped_sources": skipped_sources,
        "label_window_start": start,
        "label_window_end": end,
        "burned_pixel_count": burned_count,
    }


def downsample_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    max_points: int,
) -> dict:
    """
    ROC eğrisini JSON önizlemesi için max_points noktaya indirger.

    Full array'ler (milyonlarca nokta olabilir) JSON'a yazılmaz. Eşit aralıklı
    indeksleme ile küçük, temsili bir önizleme üretilir.
    """
    n = int(fpr.size)
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points).astype(int)
    # thresholds[0] sklearn'de +inf olabilir; JSON için sonlu değere indir.
    thr = thresholds[idx].astype(float)
    thr = np.where(np.isfinite(thr), thr, None)
    return {
        "fpr": [round(float(v), 5) for v in fpr[idx]],
        "tpr": [round(float(v), 5) for v in tpr[idx]],
        "thresholds": [None if v is None else round(float(v), 5) for v in thr],
        "downsampled_from": n,
        "max_roc_points": max_points,
    }


def predictor_label_stats(
    predictor: np.ndarray,
    burned: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict, dict | None]:
    """
    Tek bir predictor için burned/unburned ayrışmasını ve ROC/AUC'yi hesaplar.

    Döndürür: (summary_dict, roc_arrays_for_plot)
        - summary_dict: JSON'a yazılacak KOMPAKT özet (full array YOK; yalnız
          opsiyonel downsample'lı roc_curve_preview).
        - roc_arrays_for_plot: PNG çizimi için full fpr/tpr (BELLEKTE; JSON'a
          yazılmaz). ROC yoksa None.

    Hem tam (full) hem dengeli (balanced) metrikler raporlanır. NaN predictor
    pikselleri ve geçersiz etiketler dışlanır.
    """
    valid = np.isfinite(predictor) & np.isfinite(burned)
    pred_valid = predictor[valid]
    label_valid = (burned[valid] > 0).astype("int8")

    burned_vals = pred_valid[label_valid == 1]
    unburned_vals = pred_valid[label_valid == 0]

    n_burned = int(burned_vals.size)
    n_unburned = int(unburned_vals.size)

    result: dict = {
        "valid_paired_pixels": int(pred_valid.size),
        "burned_pixels": n_burned,
        "unburned_pixels": n_unburned,
        "burned_mean": float(np.mean(burned_vals)) if n_burned else None,
        "burned_median": float(np.median(burned_vals)) if n_burned else None,
        "unburned_mean": float(np.mean(unburned_vals)) if n_unburned else None,
        "unburned_median": float(np.median(unburned_vals)) if n_unburned else None,
        "auc_full": None,
        "auc_balanced": None,
    }
    roc_for_plot = None

    if n_burned == 0 or n_unburned == 0:
        result["note"] = "insufficient burned or unburned samples for ROC/AUC"
        return result, roc_for_plot

    if not SKLEARN_AVAILABLE:
        result["note"] = "sklearn unavailable; AUC/ROC skipped"
        return result, roc_for_plot

    # Tam (full) AUC + ROC (array'ler yalnız bellekte/PNG için)
    result["auc_full"] = float(roc_auc_score(label_valid, pred_valid))
    fpr, tpr, thresholds = roc_curve(label_valid, pred_valid)
    roc_for_plot = {"fpr": fpr, "tpr": tpr}
    # JSON'a sadece küçük downsample önizleme:
    result["roc_curve_preview"] = downsample_roc(
        fpr, tpr, thresholds, VALIDATION_MAX_ROC_PREVIEW_POINTS
    )

    # Dengeli (balanced) AUC: unburned alt-örnekleme
    target_unburned = int(n_burned * VALIDATION_BALANCED_UNBURNED_RATIO)
    target_unburned = min(target_unburned, n_unburned)
    if target_unburned > 0:
        idx = rng.choice(n_unburned, size=target_unburned, replace=False)
        bal_pred = np.concatenate([burned_vals, unburned_vals[idx]])
        bal_label = np.concatenate([
            np.ones(n_burned, dtype="int8"),
            np.zeros(target_unburned, dtype="int8"),
        ])
        result["auc_balanced"] = float(roc_auc_score(bal_label, bal_pred))
        result["balanced_unburned_count"] = target_unburned

    return result, roc_for_plot


def plot_roc_comparison(
    per_predictor: dict,
    roc_arrays: dict,
    out_path: Path,
) -> str | None:
    """
    Tüm predictor'ların ROC eğrilerini tek figürde çizer.

    roc_arrays: {predictor_key: {"fpr": np.ndarray, "tpr": np.ndarray}} — full
    array'ler (bellekte; JSON'a yazılmaz). AUC değerleri per_predictor özetinden
    alınır.
    """
    if not SKLEARN_AVAILABLE:
        return None
    if not any(v is not None for v in roc_arrays.values()):
        return None

    plt.figure(figsize=(7, 7))
    for key, roc in roc_arrays.items():
        if roc is None:
            continue
        auc = per_predictor.get(key, {}).get("auc_full")
        if auc is None:
            continue
        label = PREDICTORS[key]["label"]
        plt.plot(roc["fpr"], roc["tpr"], label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random (AUC=0.5)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Burned-area association: ROC comparison")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path.name


def plot_boxplot(
    predictors: dict,
    burned: np.ndarray,
    out_path: Path,
) -> str | None:
    """Her predictor için burned vs unburned dağılımını boxplot ile gösterir."""
    fig, axes = plt.subplots(
        1, len(predictors), figsize=(4 * len(predictors), 5), squeeze=False
    )
    any_plotted = False
    for ax, (key, arr) in zip(axes[0], predictors.items()):
        valid = np.isfinite(arr) & np.isfinite(burned)
        label_valid = (burned[valid] > 0)
        burned_vals = arr[valid][label_valid]
        unburned_vals = arr[valid][~label_valid]
        if burned_vals.size == 0 or unburned_vals.size == 0:
            ax.set_title(f"{key}\n(insufficient samples)", fontsize=8)
            ax.axis("off")
            continue
        ax.boxplot(
            [unburned_vals, burned_vals],
            tick_labels=["unburned", "burned"],
            showfliers=False,
        )
        ax.set_title(PREDICTORS[key]["label"], fontsize=8)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path.name


def plot_predictor_maps_with_overlay(
    predictors: dict,
    burned: np.ndarray,
    out_path: Path,
) -> str | None:
    """Her predictor haritasını, yanmış alan konturuyla birlikte çizer."""
    try:
        fig, axes = plt.subplots(
            1, len(predictors), figsize=(5 * len(predictors), 5), squeeze=False
        )
        burned_mask = np.where(burned > 0, 1.0, np.nan)
        for ax, (key, arr) in zip(axes[0], predictors.items()):
            masked = np.ma.masked_invalid(arr)
            im = ax.imshow(masked, cmap="YlOrBr")
            # Yanmış alanı yarı saydam mavi overlay olarak göster.
            ax.imshow(
                np.ma.masked_invalid(burned_mask),
                cmap="cool",
                alpha=0.5,
            )
            ax.set_title(PREDICTORS[key]["label"], fontsize=8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.6)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path.name
    except Exception as exc:  # noqa: BLE001
        log.warning("Overlay haritası çizilemedi (atlanıyor): %s", exc)
        return None


def write_stats_json(report: dict) -> Path:
    """validation_stats.json yazar."""
    path = OUTPUT_DIR / "validation_stats.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def fmt(value, digits: int = 4) -> str:
    """Markdown için sayı formatlama."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary_markdown(report: dict) -> Path:
    """validation_summary.md yazar."""
    labels = report["labels"]
    per = report["per_predictor"]
    used_sources = labels.get("sources_used", [])
    skipped = labels.get("skipped_sources", [])
    skipped_names = [s["source"] for s in skipped]
    mode = report.get("validation_mode", "same_season")
    pred_w = report.get("predictor_window", {})
    label_w = report.get("label_window", {})
    lead = report.get("temporal_lead_days")

    if mode == "pre_fire" and lead is not None:
        lead_text = f"{lead}"
        relation_text = "predictor before label (pre-fire)"
    elif mode == "same_season":
        lead_text = "overlapping windows / n/a"
        relation_text = "predictor and label same window (same-season)"
    else:
        lead_text = "n/a"
        relation_text = "n/a"

    lines = [
        "# Step6 Burned-Area Association Test",
        "",
        f"Created at: `{report['created_at']}`",
        f"Validation mode: `{mode}`",
        f"Region / AOI: `{report['region']}`",
        f"Predictor window: `{pred_w.get('start')} -> {pred_w.get('end')}`",
        f"Label window: `{label_w.get('start')} -> {label_w.get('end')}`",
        f"Temporal relation: `{relation_text}`",
        f"Temporal lead/gap (days): `{lead_text}`",
        f"Label sources (used): `{', '.join(used_sources) if used_sources else 'none'}`",
        f"Label sources (skipped): "
        f"`{', '.join(skipped_names) if skipped_names else 'none'}`",
        "",
        "> This is a **first burned-area association test / initial validation "
        "experiment**, not a validated fire-risk model. No RF/XGBoost is trained "
        "here.",
        "",
        "> Full per-pixel arrays and full ROC arrays are not stored in JSON; "
        "ROC curves are saved as PNG. The JSON keeps only compact summary metrics "
        "and a small downsampled `roc_curve_preview`.",
        "",
    ]

    # pre_fire uyarısı
    prefire_warning = report.get("prefire_warning")
    if prefire_warning:
        lines.extend([f"> **Warning:** {prefire_warning}", ""])

    # Atlanan kaynakların sebepleri
    if skipped:
        lines.extend(["## Skipped label sources", ""])
        for item in skipped:
            lines.append(f"- **{item['source']}**: {item['reason']}")
        lines.append("")

    # FireCCI51 özel notu (istenen): yalnız FireCCI51 atlandı ama MCD64A1 kullanıldıysa
    if "FireCCI51" in skipped_names and "MCD64A1" in used_sources:
        lines.extend([
            "> FireCCI51 was unavailable/empty for the selected AOI-season in "
            "this run; validation used MCD64A1 burned-area labels.",
            "",
        ])

    lines.extend([
        "## Sample counts",
        "",
        f"- Burned pixels (label): `{labels['burned_pixel_count']}`",
        "",
        "## Predictor comparison",
        "",
        "| Predictor | Valid pairs | Burned | Unburned | Burned mean | "
        "Unburned mean | AUC (full) | AUC (balanced) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for key, stats in per.items():
        lines.append(
            "| {label} | {pairs} | {burned} | {unburned} | {bmean} | {umean} "
            "| {auc_f} | {auc_b} |".format(
                label=PREDICTORS[key]["label"],
                pairs=stats["valid_paired_pixels"],
                burned=stats["burned_pixels"],
                unburned=stats["unburned_pixels"],
                bmean=fmt(stats["burned_mean"]),
                umean=fmt(stats["unburned_mean"]),
                auc_f=fmt(stats.get("auc_full")),
                auc_b=fmt(stats.get("auc_balanced")),
            )
        )

    lines.extend(["", "## Burned vs unburned (median)", ""])
    for key, stats in per.items():
        lines.append(
            f"- **{PREDICTORS[key]['label']}**: burned median "
            f"`{fmt(stats['burned_median'])}` vs unburned median "
            f"`{fmt(stats['unburned_median'])}`"
        )

    # tvdi_anomaly_zscore'un ayrı uyarısı
    tvdi_z = per.get("tvdi_anomaly_zscore")
    if tvdi_z is not None:
        lines.extend([
            "",
            "## Note on TVDI anomaly z-score",
            "",
            "`tvdi_anomaly_zscore` is reliability-filtered (masked where baseline "
            "TVDI std < threshold), so its valid sample count is reported "
            "separately and is typically much smaller than the continuous "
            "predictors:",
            "",
            f"- Valid paired pixels: `{tvdi_z['valid_paired_pixels']}`",
            f"- Burned pixels in valid set: `{tvdi_z['burned_pixels']}`",
        ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `current_tvdi` and `tvdi_difference` are treated as continuous dryness "
        "indicators.",
        "- `tvdi_anomaly_zscore` is reliability-filtered and heavily masked; its "
        "smaller valid sample count should be considered when comparing AUCs.",
        "- AUC > 0.5 suggests positive association between the predictor and "
        "burned area; this is preliminary association evidence, **not** a "
        "validated fire-risk product.",
    ])

    # Koşullu yorumlar (AUC tabanlı)
    lst = per.get("thermal_anomaly_zscore")
    if lst is not None and lst.get("auc_full") is not None:
        auc = lst["auc_full"]
        if 0.5 < auc < 0.6:
            lines.append(
                "- LST anomaly shows weak positive association with burned-area "
                f"labels (AUC={auc:.3f})."
            )

    tvdi_keys = ["current_tvdi", "tvdi_difference", "tvdi_anomaly_zscore"]
    tvdi_below_half = [
        k for k in tvdi_keys
        if per.get(k, {}).get("auc_full") is not None
        and per[k]["auc_full"] < 0.5
    ]
    if tvdi_below_half:
        lines.append(
            "- TVDI predictors do not show positive separation in this run "
            f"({', '.join(tvdi_below_half)} AUC < 0.5); this may indicate temporal "
            "misalignment between the predictor window and the burn-label window."
        )

    if mode == "pre_fire":
        lines.append(
            "- Pre-fire validation should be interpreted only when predictor "
            "rasters were generated for the configured predictor window."
        )

    lines.extend([
        "",
        "## Outputs",
        "",
    ])
    for name in report.get("png_outputs", []):
        lines.append(f"- `{name}`")

    path = OUTPUT_DIR / "validation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def resolve_windows() -> dict:
    """
    Validation moduna göre predictor ve label pencerelerini belirler.

    same_season: ikisi de VALIDATION_SEASON_START/END.
    pre_fire:    predictor ve label pencereleri ayrı (VALIDATION_PREFIRE_*).
    """
    mode = VALIDATION_MODE
    if mode == "pre_fire":
        return {
            "mode": "pre_fire",
            "predictor_start": VALIDATION_PREFIRE_PREDICTOR_START,
            "predictor_end": VALIDATION_PREFIRE_PREDICTOR_END,
            "label_start": VALIDATION_PREFIRE_LABEL_START,
            "label_end": VALIDATION_PREFIRE_LABEL_END,
        }
    # default: same_season
    return {
        "mode": "same_season",
        "predictor_start": VALIDATION_SEASON_START,
        "predictor_end": VALIDATION_SEASON_END,
        "label_start": VALIDATION_SEASON_START,
        "label_end": VALIDATION_SEASON_END,
    }


def temporal_lead_days(
    predictor_end: str,
    label_start: str,
    mode: str,
) -> int | None:
    """
    Predictor penceresi bitişi ile label penceresi başlangıcı arası gerçek lead (gün).

    Yalnız pre_fire modunda anlamlıdır (predictor window label window'dan önce gelir).
    same_season modunda pencereler çakıştığı için lead anlamsızdır; None döner ve
    raporda "overlapping windows / n/a" gösterilir.
    """
    if mode != "pre_fire":
        return None
    try:
        pe = datetime.strptime(predictor_end, "%Y-%m-%d")
        ls = datetime.strptime(label_start, "%Y-%m-%d")
        return (ls - pe).days
    except (ValueError, TypeError):
        return None


def main() -> dict:
    """Step6 burned-area association testini çalıştırır."""
    log.info("=" * 60)
    log.info("STEP 6 BAŞLIYOR (burned-area association test)")
    log.info("=" * 60)

    if not SKLEARN_AVAILABLE:
        log.warning(
            "scikit-learn yok; ROC/AUC hesaplanamayacak. "
            "pip install scikit-learn ile kurun."
        )

    windows = resolve_windows()
    log.info("Validation modu: %s", windows["mode"])
    log.info(
        "Predictor window: %s -> %s | Label window: %s -> %s",
        windows["predictor_start"], windows["predictor_end"],
        windows["label_start"], windows["label_end"],
    )

    # Referans grid + predictor'ları yükle (esnek dosya adı)
    ref_path = reference_predictor_path()
    grid = read_reference_grid(ref_path)
    log.info("Referans grid: %s (%sx%s)", ref_path.name, grid["width"], grid["height"])

    predictors = {}
    predictor_sources = {}
    for key in PREDICTORS:
        path = resolve_predictor_path(key)
        if path is not None:
            predictors[key] = read_predictor(path)
            predictor_sources[key] = str(path)
        else:
            log.warning(
                "Predictor bulunamadı, atlanıyor: %s",
                [p.name for p in PREDICTORS[key]["path_candidates"]],
            )

    if not predictors:
        raise ValidationError(
            "Hiçbir predictor rasterı bulunamadı. Önce Step5 ve Step5C çalıştırın."
        )

    # Etiketleri label window ile indir
    labels = fetch_labels(grid, windows["label_start"], windows["label_end"])
    burned = labels["burned"]

    # Her predictor için istatistik (ROC array'leri ayrı, bellekte)
    rng = np.random.default_rng(VALIDATION_RANDOM_SEED)
    per_predictor = {}
    roc_arrays = {}
    for key, arr in predictors.items():
        if arr.shape != burned.shape:
            log.warning(
                "%s grid'i etiket grid'iyle uyuşmuyor (%s != %s); atlanıyor.",
                key, arr.shape, burned.shape,
            )
            continue
        summary, roc = predictor_label_stats(arr, burned, rng)
        summary["predictor_name"] = key
        summary["source_file"] = predictor_sources.get(key)
        per_predictor[key] = summary
        roc_arrays[key] = roc

    # Görseller (full ROC array'leri yalnız burada, bellekten)
    png_outputs = []
    roc_png = plot_roc_comparison(
        per_predictor, roc_arrays, OUTPUT_DIR / "roc_curve_comparison.png"
    )
    if roc_png:
        png_outputs.append(roc_png)
    box_png = plot_boxplot(predictors, burned, OUTPUT_DIR / "burned_vs_unburned_boxplot.png")
    if box_png:
        png_outputs.append(box_png)
    overlay_png = plot_predictor_maps_with_overlay(
        predictors, burned, OUTPUT_DIR / "predictor_maps_with_burn_overlay.png"
    )
    if overlay_png:
        png_outputs.append(overlay_png)

    lead_days = temporal_lead_days(
        windows["predictor_end"], windows["label_start"], windows["mode"]
    )

    # pre_fire uyarısı: mevcut predictor rasterları predictor window'a göre
    # üretilmemiş olabilir (Step5/Step5C config'i CURRENT_PERIOD_END_DATE'e bağlı).
    prefire_warning = None
    if windows["mode"] == "pre_fire":
        prefire_warning = (
            "Predictor rasters may not match the configured pre-fire predictor "
            "window; regenerate Step5/Step5C for this window before interpreting "
            "results."
        )

    report = {
        "step": "step6_validate_fire_relation",
        "created_at": datetime.now().isoformat(),
        "region": REGION_NAME,
        "log_file": str(log_file),
        "validation_mode": windows["mode"],
        "predictor_window": {
            "start": windows["predictor_start"],
            "end": windows["predictor_end"],
        },
        "label_window": {
            "start": windows["label_start"],
            "end": windows["label_end"],
        },
        "temporal_lead_days": lead_days,
        "prefire_warning": prefire_warning,
        "reference_grid": {
            "path": str(ref_path),
            "width": grid["width"],
            "height": grid["height"],
        },
        "per_predictor": per_predictor,
        "png_outputs": png_outputs,
        "config": {
            "balanced_unburned_ratio": VALIDATION_BALANCED_UNBURNED_RATIO,
            "random_seed": VALIDATION_RANDOM_SEED,
            "include_firms": VALIDATION_INCLUDE_FIRMS,
            "max_roc_preview_points": VALIDATION_MAX_ROC_PREVIEW_POINTS,
        },
        "json_format_note": (
            "Compact summary. Full per-pixel arrays and full ROC arrays are NOT "
            "stored in JSON; ROC curves are saved as PNG. Only a downsampled "
            "roc_curve_preview (<= max_roc_preview_points) is kept per predictor."
        ),
        "disclaimer": (
            "First burned-area association test / initial validation experiment "
            "only. Not a validated fire-risk model. No RF/XGBoost trained."
        ),
        "status": "validation_completed",
    }
    # burned array'i JSON'a yazma (büyük); sadece özet alanları tut
    report["labels"] = {
        k: v for k, v in labels.items() if k != "burned"
    }

    stats_path = write_stats_json(report)
    summary_path = write_summary_markdown(report)

    # JSON boyutunu logla (doğrulama için)
    size_mb = stats_path.stat().st_size / (1024 * 1024)
    log.info("Validation JSON: %s (%.3f MB)", stats_path, size_mb)
    log.info("Validation summary: %s", summary_path)
    log.info("PNG çıktıları: %s", ", ".join(png_outputs) if png_outputs else "yok")
    log.info("=" * 60)
    log.info("STEP 6 TAMAMLANDI")

    return report


if __name__ == "__main__":
    main()