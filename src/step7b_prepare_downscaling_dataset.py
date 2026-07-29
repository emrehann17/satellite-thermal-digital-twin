"""
step7b_prepare_downscaling_dataset.py

MODIS -> Landsat LST downscaling için pencere/tile-bazlı EĞİTİM VERİSETİ hazırlar.

ÖNEMLİ:
    - Step7B model EĞİTMEZ (RF/XGBoost yok), fire-risk modeli ÜRETMEZ.
    - Step5/Step5C/Step6 bilimsel çıktılarını DEĞİŞTİRMEZ.
    - Yalnızca temiz, hizalanmış tabular örnekler üretir:
        target  = yüksek çözünürlüklü Landsat LST (Celsius)
        features= kaba MODIS context + NDVI + DEM (elevation/slope) + land cover
                  + koordinatlar (+ opsiyonel TVDI/anomaly).
    - Bellek dostu: rasterio windows/tiles; tüm rasterlar aynı anda RAM'e alınmaz.

Çıktılar:
    outputs/step7b/downscaling_training_samples.parquet  (pyarrow varsa)
    outputs/step7b/downscaling_training_samples.csv
    outputs/step7b/downscaling_dataset_stats.json
    outputs/step7b/downscaling_dataset_summary.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import transform as window_transform

from core.config import (
    STEP7B_TILE_SIZE,
    STEP7B_MAX_SAMPLES,
    STEP7B_RANDOM_SEED,
    STEP7B_SAMPLE_FRACTION,
    STEP7B_STRATIFY_BY_MODIS_PIXEL,
    STEP7B_MIN_TARGET_CELSIUS,
    STEP7B_MAX_TARGET_CELSIUS,
    STEP7B_REQUIRE_NDVI,
    STEP7B_REQUIRE_DEM,
    STEP7B_OUTPUT_FORMATS,
    STEP7B_INCLUDE_OPTIONAL_TVDI_FEATURES,
    STEP7B_INCLUDE_OPTIONAL_ANOMALY_FEATURES,
    STEP7B_MODIS_SUSPICIOUS_ZERO_FRACTION,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.utils.tiling import iter_windows

BASE_DIR = PROJECT_ROOT
OUTPUTS_DIR = BASE_DIR / "outputs" / "step7b"

log, log_file = setup_logger("step7b")

# Fiziksel makul aralıklar (clamp DEĞİL; bu aralık dışı satırlar DROP edilir).
NDVI_MIN, NDVI_MAX = -1.2, 1.2
SLOPE_MIN, SLOPE_MAX = 0.0, 90.0
ELEVATION_MIN, ELEVATION_MAX = -500.0, 9000.0

# Bu iki isim, scripts/prepare_modis_for_step7.py'nin urettigi feature
# adlariyla BIREBIR eslesir (bkz. build_feature_registry).
MODIS_MEAN_FEATURE_NAME = "modis_lst_mean_celsius"
MODIS_STD_FEATURE_NAME = "modis_lst_std_celsius"


class Step7BModisValidationError(SystemExit):
    """Fail-fast MODIS kaynak-raster dogrulama hatasi (hizalamadan ONCE)."""


class LegacyModisCompatibilityAttestationError(Step7BModisValidationError):
    """Gecersiz/eksik tarihsel-uyumluluk beyani (attestation).

    Bu hata, zero-fill muafiyetinin REDDEDILDIGI anlamina gelir: Step7B yine
    strict davranir.
    """


# =============================================================================
# Tarihsel (donmus) MODIS uyumluluk beyani -- DAR KAPSAMLI
# =============================================================================
#: Bu modun ADI sabittir ve baska hicbir yerde uretilmez. Step7B'nin varsayilan
#: davranisi HER ZAMAN strict'tir: `legacy_modis_compatibility=None`.
LEGACY_FROZEN_MODIS_COMPATIBILITY_MODE = "legacy_frozen_modis_compatibility"


@dataclass(frozen=True)
class LegacyModisCompatibilityAttestation:
    """Tarihsel zero-fill MODIS semantiginin YENIDEN URETILMESI icin beyan.

    Bu beyan bir "dogrulamayi atla" bayragi DEGILDIR: yalnizca burada
    ADI, SHA-256'si ve BYTE boyutu ONCEDEN kaydedilmis MODIS rasterlari icin,
    ve yalnizca dosyanin O ANKI icerigi bu hash'lerle BIREBIR esletigi zaman
    Kural 1'i (nodata-yok + zero-fill imzasi) askiya alir. Kural 2/3/4
    (fiziksel aralik, negatif std, mean/std grid esitligi) AYNEN uygulanir.

    Tek bir boolean YETERLI DEGILDIR ve kabul edilmez -- `validate_modis_source_rasters`
    yalnizca bu tipin bir ornegini kabul eder.

    Alanlar:
        mode: tam olarak LEGACY_FROZEN_MODIS_COMPATIBILITY_MODE olmalidir.
        experiment_id: beyanin baglandigi deney kimligi; Step7B ctx ile eslesmelidir.
        rasters: {feature_adi: {"sha256":..., "bytes":..., "authorized_paths":[...]}}
        historical_step7b_evidence_confirmed: cagiran taraf, donmus Step7B
            metadatasinin nodata'siz kaynagin gercekten kullanildigini
            DOGRULADIGINI beyan eder (True olmak zorundadir).
        issued_by / attestation_id: denetim izi (rapora yazilir).
    """

    mode: str
    experiment_id: str
    rasters: dict
    historical_step7b_evidence_confirmed: bool
    issued_by: str = ""
    attestation_id: str = ""
    notes: tuple = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, payload: dict) -> "LegacyModisCompatibilityAttestation":
        """Beyani bir mapping'den kurar (alan dogrulamasi kullanim aninda yapilir)."""
        if not isinstance(payload, dict):
            raise LegacyModisCompatibilityAttestationError(
                "legacy MODIS compatibility attestation must be a mapping; "
                f"got {type(payload).__name__}."
            )
        rasters = payload.get("rasters") or {}
        normalized = {
            str(name): {
                "sha256": str(entry.get("sha256", "")),
                "bytes": int(entry.get("bytes", -1)),
                "authorized_paths": tuple(
                    str(Path(p).resolve()) for p in (entry.get("authorized_paths") or [])
                ),
            }
            for name, entry in rasters.items()
        }
        return cls(
            mode=str(payload.get("mode", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            rasters=normalized,
            historical_step7b_evidence_confirmed=bool(
                payload.get("historical_step7b_evidence_confirmed", False)
            ),
            issued_by=str(payload.get("issued_by", "")),
            attestation_id=str(payload.get("attestation_id", "")),
            notes=tuple(payload.get("notes") or ()),
        )


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _authorize_zero_fill_waiver(
    attestation, experiment_id: str | None, required: dict[str, Path],
) -> dict | None:
    """Zero-fill muafiyetini YALNIZCA gecerli bir beyanla yetkilendirir.

    `attestation is None` ise None doner -- cagiran taraf strict reddi uygular
    (varsayilan davranis). Beyan VARSA ve HERHANGI bir kosul saglanmiyorsa
    LegacyModisCompatibilityAttestationError firlatir; sessizce strict'e
    dusmez, cunku "gecersiz beyanla calistirma girisimi" bir hatadir.
    """
    if attestation is None:
        return None

    if not isinstance(attestation, LegacyModisCompatibilityAttestation):
        raise LegacyModisCompatibilityAttestationError(
            "Step7B zero-fill guard can only be waived by a "
            "LegacyModisCompatibilityAttestation instance carrying the expected "
            f"source paths and SHA-256 hashes; got {type(attestation).__name__}. "
            "A boolean or a plain mapping is NEVER sufficient."
        )
    if attestation.mode != LEGACY_FROZEN_MODIS_COMPATIBILITY_MODE:
        raise LegacyModisCompatibilityAttestationError(
            f"unknown MODIS compatibility mode {attestation.mode!r}; the only "
            f"supported mode is {LEGACY_FROZEN_MODIS_COMPATIBILITY_MODE!r}."
        )
    if attestation.historical_step7b_evidence_confirmed is not True:
        raise LegacyModisCompatibilityAttestationError(
            "the attestation does not confirm the frozen Step7B historical "
            "evidence (historical_step7b_evidence_confirmed is not True)."
        )
    if not experiment_id or attestation.experiment_id != experiment_id:
        raise LegacyModisCompatibilityAttestationError(
            f"the attestation is bound to experiment {attestation.experiment_id!r} "
            f"but Step7B is running for {experiment_id!r}."
        )

    verified: dict[str, dict] = {}
    for name, path in required.items():
        entry = attestation.rasters.get(name)
        if not entry:
            raise LegacyModisCompatibilityAttestationError(
                f"the attestation does not cover the MODIS raster {name!r}; every "
                "MODIS raster Step7B is about to read must be attested."
            )
        resolved = str(Path(path).resolve())
        if resolved not in entry["authorized_paths"]:
            raise LegacyModisCompatibilityAttestationError(
                f"{name}: {resolved} is not an authorized path in the attestation "
                f"({list(entry['authorized_paths'])})."
            )
        actual_bytes = int(Path(path).stat().st_size)
        actual_sha = _sha256_of(Path(path))
        if actual_bytes != entry["bytes"] or actual_sha != entry["sha256"]:
            raise LegacyModisCompatibilityAttestationError(
                f"{name}: the file at {resolved} does not match the attested "
                f"content (expected sha256={entry['sha256']} bytes={entry['bytes']}, "
                f"found sha256={actual_sha} bytes={actual_bytes}). The historical "
                "compatibility path is refused; the strict guard stands."
            )
        verified[name] = {
            "path": resolved, "sha256": actual_sha, "bytes": actual_bytes,
        }

    return {
        "mode": LEGACY_FROZEN_MODIS_COMPATIBILITY_MODE,
        "waived_rule": "no_nodata_zero_fill_signature",
        "waiver_scope": "this Step7B call only",
        "experiment_id": experiment_id,
        "attestation_id": attestation.attestation_id,
        "issued_by": attestation.issued_by,
        "verified_rasters": verified,
        "rasters_rewritten": False,
        "nodata_assigned": False,
        "values_or_mask_changed": False,
        "statement":
            "The historical zero-filled MODIS representation was reproduced "
            "verbatim. No raster value, mask, dtype or grid was changed and zero "
            "is NOT declared a physically valid MODIS LST value.",
    }


def _read_source_raster_stats(path: Path) -> dict:
    """Bir kaynak (hizalanmamis) rasterin nodata/gecerlilik/deger ozetini
    okur -- yalnizca metadata/dogrulama icin, hicbir dosya YAZMAZ."""
    with rasterio.open(path) as src:
        nodata = src.nodata
        shape_hw = [src.height, src.width]
        transform = src.transform
        arr = src.read(1)

    if nodata is None:
        valid_mask = np.ones(arr.shape, dtype=bool)
    elif isinstance(nodata, float) and np.isnan(nodata):
        valid_mask = np.isfinite(arr)
    else:
        valid_mask = arr != nodata
    valid_vals = arr[valid_mask]

    return {
        "path": str(path),
        "source_nodata": None if nodata is None else float(nodata),
        "source_valid_count": int(valid_mask.sum()),
        "source_invalid_count": int((~valid_mask).sum()),
        "exact_zero_count_among_valid": (
            int((valid_vals == 0.0).sum()) if valid_vals.size else 0
        ),
        "source_min": float(valid_vals.min()) if valid_vals.size else None,
        "source_max": float(valid_vals.max()) if valid_vals.size else None,
        "source_mean": float(valid_vals.mean()) if valid_vals.size else None,
        "source_median": float(np.median(valid_vals)) if valid_vals.size else None,
        "shape_hw": shape_hw,
        "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
    }


def validate_modis_source_rasters(
    core_features: list[dict],
    *,
    experiment_id: str | None = None,
    legacy_modis_compatibility: "LegacyModisCompatibilityAttestation | None" = None,
) -> dict:
    """
    Deney-farkında (Kozan-dışı) çalıştırmalarda, MODIS mean/std kaynak
    rasterlarını BİLİNEAR HİZALAMADAN ÖNCE doğrular. Herhangi bir kural
    ihlal edilirse Step7BModisValidationError (SystemExit) fırlatır --
    hizalama/örnekleme HİÇ ÇALIŞMAZ.

    `legacy_modis_compatibility` VARSAYILAN OLARAK None'dır: davranış strict'tir
    ve mevcut tüm çağıranlar (CLI dahil) etkilenmez. Yalnızca geçerli bir
    :class:`LegacyModisCompatibilityAttestation` verildiğinde -- yani beklenen
    yollar ve SHA-256 hash'leri O AN doğrulandığında -- Kural 1 askıya alınır.
    Bu, sıfırın fiziksel olarak geçerli olduğunu İLAN ETMEZ; yalnızca donmuş
    tarihsel Step7 davranışını yeniden üretir. Kural 2/3/4 aynen uygulanır.

    Kurallar:
        1) nodata TANIMSIZ VE "geçerli" piksellerin şüpheli bir oranı tam
           0.0 ise (bkz. STEP7B_MODIS_SUSPICIOUS_ZERO_FRACTION) -- bu, bir
           deniz/gözlemsiz bölgenin sayısal 0.0 olarak dışa aktarıldığının
           imzasıdır.
        2) mean değerleri, mevcut kabul edilen fiziksel LST aralığının
           (STEP7B_MIN_TARGET_CELSIUS, STEP7B_MAX_TARGET_CELSIUS -- Landsat
           target için zaten kullanılan AYNI aralık) DIŞINDA.
        3) std NEGATİF bir geçerli değer içeriyor.
        4) mean ve std kaynak gridleri (shape/transform) FARKLI.

    Döner: {feature_name: {...source stats..., "validation_status": "passed"}}
    -- Step7B'nin downscaling_dataset_stats.json alignment_diagnostics'ine
    isimle eşleştirilerek eklenir.
    """
    by_name = {f["name"]: Path(f["path"]) for f in core_features}
    mean_path = by_name.get(MODIS_MEAN_FEATURE_NAME)
    if mean_path is None:
        return {}

    std_path = by_name.get(MODIS_STD_FEATURE_NAME)

    mean_stats = _read_source_raster_stats(mean_path)
    zero_fraction = (
        mean_stats["exact_zero_count_among_valid"] / mean_stats["source_valid_count"]
        if mean_stats["source_valid_count"] else 0.0
    )
    zero_fill_signature = (
        mean_stats["source_nodata"] is None
        and zero_fraction > STEP7B_MODIS_SUSPICIOUS_ZERO_FRACTION
    )
    zero_fill_guard = {
        "rule": "no_nodata_zero_fill_signature",
        "threshold": STEP7B_MODIS_SUSPICIOUS_ZERO_FRACTION,
        "observed_zero_fraction": zero_fraction,
        "signature_present": zero_fill_signature,
        "mode": "strict_default_guard",
        "waived": False,
    }
    if zero_fill_signature:
        required = {MODIS_MEAN_FEATURE_NAME: mean_path}
        if std_path is not None:
            required[MODIS_STD_FEATURE_NAME] = std_path
        waiver = _authorize_zero_fill_waiver(
            legacy_modis_compatibility, experiment_id, required,
        )
        if waiver is None:
            raise Step7BModisValidationError(
                f"MODIS mean girdisi ({mean_path}) HİÇBİR nodata tanımlamıyor VE "
                f"'geçerli' piksellerinin %{zero_fraction * 100:.1f}'i tam 0.0 -- "
                "bu, Step7B'nin reddetmesi gereken deniz/gözlemsiz bölge "
                "sıfır-doldurma imzasıdır (0.0 fiziksel olarak geçerli bir "
                "Celsius değeri olduğu için sessizce kabul edilemez). MODIS'i "
                "açık bir nodata değeriyle yeniden export edin: "
                "python scripts/prepare_modis_for_step7.py --export --force"
            )
        zero_fill_guard.update({
            "mode": waiver["mode"], "waived": True, "waiver": waiver,
        })
        log.warning(
            "[step7b] MODIS zero-fill guard WAIVED under %s for experiment=%s "
            "(attestation=%s). No raster value/mask/dtype/grid is changed; zero "
            "is NOT declared a valid MODIS LST value.",
            waiver["mode"], experiment_id, waiver["attestation_id"] or "<unnamed>",
        )

    if mean_stats["source_valid_count"] and (
        mean_stats["source_min"] < STEP7B_MIN_TARGET_CELSIUS
        or mean_stats["source_max"] > STEP7B_MAX_TARGET_CELSIUS
    ):
        raise Step7BModisValidationError(
            f"MODIS mean girdisi ({mean_path}) kabul edilen fiziksel LST "
            f"aralığının [{STEP7B_MIN_TARGET_CELSIUS}, {STEP7B_MAX_TARGET_CELSIUS}] "
            f"°C DIŞINDA geçerli piksel değerleri içeriyor "
            f"(min={mean_stats['source_min']}, max={mean_stats['source_max']})."
        )

    diagnostics = {
        MODIS_MEAN_FEATURE_NAME: {
            **mean_stats,
            "validation_status": "passed",
            "zero_fill_guard": zero_fill_guard,
        }
    }

    if std_path is not None:
        std_stats = _read_source_raster_stats(std_path)
        if std_stats["source_valid_count"] and std_stats["source_min"] is not None and std_stats["source_min"] < 0:
            raise Step7BModisValidationError(
                f"MODIS std girdisi ({std_path}) NEGATİF geçerli değer(ler) "
                f"içeriyor (min={std_stats['source_min']}) -- standart sapma "
                "negatif olamaz."
            )
        if std_stats["shape_hw"] != mean_stats["shape_hw"] or std_stats["transform"] != mean_stats["transform"]:
            raise Step7BModisValidationError(
                f"MODIS mean ({mean_path}) ve std ({std_path}) kaynak "
                f"gridleri FARKLI (mean shape/transform="
                f"{mean_stats['shape_hw']}/{mean_stats['transform']}, "
                f"std shape/transform={std_stats['shape_hw']}/{std_stats['transform']})."
            )
        diagnostics[MODIS_STD_FEATURE_NAME] = {
            **std_stats,
            "validation_status": "passed",
            "zero_fill_guard": zero_fill_guard,
        }

    return diagnostics


# =============================================================================
# Target + feature kaynak çözümleme
# =============================================================================
def resolve_target(ctx: dict | None = None) -> tuple[Path | None, int]:
    """Landsat LST target rasterını çözer (öncelik: step5 celsius -> current_period).

    ctx: None ise (varsayılan) legacy Kozan keşfi. Verilirse (Kozan-dışı,
        or. manavgat_2021) YALNIZCA ctx["step5_output_dir"] altına bakar --
        legacy outputs/step5/ veya data/current_period/ yollarına ASLA
        dokunmaz.
    """
    if ctx is not None:
        candidates = [(ctx["step5_output_dir"] / "current_period_median_celsius.tif", 1)]
        for path, band in candidates:
            if path.exists():
                return path, band
        return None, 1

    candidates = [
        (BASE_DIR / "outputs" / "step5" / "current_period_median_celsius.tif", 1),
        (BASE_DIR / "data" / "current_period" / "landsat_current_period_60days.tif", 1),
    ]
    for path, band in candidates:
        if path.exists():
            return path, band
    # 60days dosyası farklı gün sayısıyla olabilir; glob ile dene.
    cp_dir = BASE_DIR / "data" / "current_period"
    if cp_dir.exists():
        for path in sorted(cp_dir.glob("landsat_current_period_*days.tif")):
            if "(" in path.name:
                continue
            return path, 1
    return None, 1


def build_feature_registry(
    include_tvdi: bool = STEP7B_INCLUDE_OPTIONAL_TVDI_FEATURES,
    include_anomaly: bool = STEP7B_INCLUDE_OPTIONAL_ANOMALY_FEATURES,
    ctx: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Feature kaynaklarını çözer.

    Döner: (core_features, optional_features, missing_optional)
    Her giriş: {name, path, band, resampling}
    resampling: "bilinear" (sürekli) | "nearest" (kategorik/binary)

    ctx: None ise (varsayılan) legacy Kozan keşfi (BASE_DIR/outputs/step5,
        BASE_DIR/outputs/step5c, BASE_DIR/data/...). Verilirse (Kozan-dışı):
        Step5/Step5C kaynakları TAMAMEN namespaced (ctx["step5_output_dir"],
        ctx["step5c_output_dir"], ctx["ndvi_current_dir"],
        ctx["modis_input_dir"]) okunur -- legacy Kozan yollarına ASLA
        dokunulmaz. DEM (elevation/slope) ARTIK ctx["dem_input_dir"]'den
        okunur -- Kozan için bu paylaşılan (shared) data/dem/'dir
        (değişmedi), Kozan-dışı deneyler için ise NAMESPACED
        (outputs/experiments/<experiment_id>/data/dem/,
        scripts/prepare_dem_for_experiment.py ile hazırlanır) -- çünkü
        paylaşılan data/dem/ yalnızca Kozan'ın AOI'sini kapsar ve diğer
        deneylerle coğrafi olarak örtüşmez. Landcover,
        ctx["landcover_aligned_path"] mevcutsa (Step6A gate-input çıktısı)
        onu kullanır; yoksa hiçbir landcover eklenmez (Kozan'ın paylaşılan
        data/landcover/ dizinine ASLA düşülmez).
    """
    data = BASE_DIR / "data"  # yalnızca Kozan legacy DEM fallback + landcover için
    if ctx is not None:
        s5 = ctx["step5_output_dir"]
        s5c = ctx["step5c_output_dir"]
        modis_search_dirs = [ctx["modis_input_dir"]]
        ndvi_current = ctx["ndvi_current_dir"] / "current_ndvi_median.tif"
        landcover_override = ctx.get("landcover_aligned_path")
    else:
        s5 = BASE_DIR / "outputs" / "step5"
        s5c = BASE_DIR / "outputs" / "step5c"
        modis_search_dirs = [data / "modis"]
        ndvi_current = data / "ndvi_current_period" / "current_ndvi_median.tif"
        landcover_override = None

    def first_existing(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    core: list[dict] = []

    # 1) MODIS LST context (mean zorunlu çekirdek; std/zscore varsa eklenir)
    modis_mean_candidates = [s5 / "modis_lst_mean_celsius_resampled.tif"]
    for d in modis_search_dirs:
        modis_mean_candidates.append(d / "modis_lst_dogu_akdeniz_4y_summer_mean.tif")
        if d.exists():
            modis_mean_candidates.extend(sorted(d.glob("*mean*.tif")))
    modis_mean = first_existing(modis_mean_candidates)
    if modis_mean is not None:
        core.append({"name": "modis_lst_mean_celsius", "path": modis_mean,
                     "band": 1, "resampling": "bilinear", "required": False})

    modis_std_candidates = [s5 / "modis_lst_std_celsius_resampled.tif"]
    for d in modis_search_dirs:
        if d.exists():
            modis_std_candidates.extend(sorted(d.glob("*std*.tif")))
    modis_std = first_existing(modis_std_candidates)
    if modis_std is not None:
        core.append({"name": "modis_lst_std_celsius", "path": modis_std,
                     "band": 1, "resampling": "bilinear", "required": False})

    modis_z = first_existing([s5 / "modis_context_zscore.tif"])
    if modis_z is not None:
        core.append({"name": "modis_context_zscore", "path": modis_z,
                     "band": 1, "resampling": "bilinear", "required": False})

    # 2) NDVI
    ndvi = first_existing([ndvi_current])
    if ndvi is not None:
        core.append({"name": "ndvi", "path": ndvi, "band": 1,
                     "resampling": "bilinear", "required": STEP7B_REQUIRE_NDVI})

    # 3) DEM elevation + slope
    # ctx verilmisse (Kozan-dışı): ctx["dem_input_dir"] -- artık NAMESPACED
    # (bkz. core/experiment_context.py + scripts/prepare_dem_for_experiment.py).
    # Kozan-dışı deneylerin data/dem/ (Kozan'a özel, coğrafi olarak farklı
    # AOI) okuması/oraya düşmesi ENGELLENIR. ctx yoksa (legacy Kozan):
    # data/dem/ (paylaşılan, değişmedi).
    dem_dir = ctx["dem_input_dir"] if ctx is not None else (data / "dem")
    elevation = first_existing([dem_dir / "elevation.tif"])
    if elevation is not None:
        core.append({"name": "elevation", "path": elevation, "band": 1,
                     "resampling": "bilinear", "required": STEP7B_REQUIRE_DEM})
    slope = first_existing([dem_dir / "slope.tif"])
    if slope is not None:
        core.append({"name": "slope", "path": slope, "band": 1,
                     "resampling": "bilinear", "required": STEP7B_REQUIRE_DEM})

    # 4) Land cover (kategorik -> nearest)
    if landcover_override is not None:
        landcover = landcover_override if landcover_override.exists() else None
    else:
        landcover = first_existing([
            data / "landcover" / "landcover_esa_worldcover_v200.tif",
        ])
        if landcover is None and (data / "landcover").exists():
            for p in sorted((data / "landcover").glob("*.tif")):
                if "(" not in p.name:
                    landcover = p
                    break
    if landcover is not None:
        core.append({"name": "landcover", "path": landcover, "band": 1,
                     "resampling": "nearest", "required": False})

    # Opsiyonel context features
    optional: list[dict] = []
    missing: list[dict] = []

    def add_optional(name: str, candidates: list[Path], enabled: bool) -> None:
        if not enabled:
            return
        p = first_existing(candidates)
        if p is not None:
            optional.append({"name": name, "path": p, "band": 1,
                             "resampling": "bilinear", "required": False})
        else:
            missing.append({"name": name, "candidates": [str(c) for c in candidates]})

    add_optional("anomaly_zscore", [s5 / "anomaly_zscore.tif"], include_anomaly)
    add_optional("current_tvdi", [s5c / "current_tvdi.tif"], include_tvdi)
    add_optional("tvdi_difference", [s5c / "tvdi_difference.tif"], include_tvdi)

    return core, optional, missing


# =============================================================================
# Pencere bazlı feature okuma (target grid'ine resample)
# =============================================================================
def _resampling_enum(name: str) -> Resampling:
    return Resampling.nearest if name == "nearest" else Resampling.bilinear


def read_feature_into_target_window(
    feature_src,
    feature_resampling: str,
    target_window,
    target_transform,
    target_crs,
    win_h: int,
    win_w: int,
    same_grid: bool,
) -> np.ndarray:
    """
    Bir feature rasterını target pencere grid'ine okur/resample eder.

    same_grid=True ise doğrudan pencere okunur (resample yok). Aksi halde
    rasterio.warp.reproject ile target pencere transform/CRS'ine yansıtılır.
    """
    if same_grid:
        arr = feature_src.read(1, window=target_window, masked=True)
        return arr.astype("float32").filled(np.nan)

    dst = np.full((win_h, win_w), np.nan, dtype="float32")
    reproject(
        source=rasterio.band(feature_src, 1),
        destination=dst,
        src_transform=feature_src.transform,
        src_crs=feature_src.crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=_resampling_enum(feature_resampling),
        dst_nodata=np.nan,
    )
    return dst


# =============================================================================
# Ana veri seti üretimi
# =============================================================================
# =============================================================================
# Deney-farkında (experiment-aware) girdi hizalama (Kozan-dışı deneyler için)
# =============================================================================
def align_feature_to_reference(
    feature_name: str,
    source_path: Path,
    resampling: str,
    ref_w: int, ref_h: int, ref_crs, ref_transform,
    output_dir: Path,
    force: bool = False,
) -> tuple[Path, dict]:
    """
    Bir feature rasterini referans (Step5 current_period_median_celsius.tif)
    gridine ONCEDEN (pencere-pencere degil, TEK SEFERDE) hizalar.

    Zaten hizali ise (grid birebir eslesiyorsa) kaynak dosya CANONICAL isimle
    (aligned_inputs/<name>.tif) KOPYALANIR (kullanılmaz/atlanmaz -- Step7D
    yalnızca bu canonical yolu arar). Hizalanmasi gerekiyorsa da AYNI
    canonical isimle (aligned_inputs/<name>.tif, "_aligned" son eki OLMADAN)
    yazilir; kategorik ("nearest") ozellikler icin nearest-neighbor, sürekli
    ("bilinear") ozellikler icin bilinear resampling kullanilir.

    Döner: (kullanilacak_path, diagnostic_dict). diagnostic_dict; kaynak
    grid bilgisi, hedef gridle orani (overlap), ve gecerli piksel sayisini
    icerir -- boylece "MODIS 1 km ile Landsat 30 m'nin nasil hizalandigi"
    Step7B calistirilmadan ONCE seffaf bir sekilde raporlanir.
    """
    with rasterio.open(source_path) as src:
        src_w, src_h, src_crs, src_t = src.width, src.height, src.crs, src.transform
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]
        src_arr = src.read(1, masked=True)
        src_valid_count = int((~np.ma.getmaskarray(src_arr)).sum())

    diag = {
        "name": feature_name,
        "resampling": resampling,
        "source_path": str(source_path),
        "source_shape_hw": [src_h, src_w],
        "source_crs": str(src_crs),
        "source_transform": [src_t.a, src_t.b, src_t.c, src_t.d, src_t.e, src_t.f],
        "source_nodata": src_nodata,
        "source_valid_pixel_count": src_valid_count,
        "source_total_pixel_count": int(src_w * src_h),
    }

    same_grid = (
        src_crs == ref_crs and src_w == ref_w and src_h == ref_h and src_t == ref_transform
    )
    out_path = output_dir / f"{feature_name}.tif"
    output_dir.mkdir(parents=True, exist_ok=True)

    if same_grid:
        # ÖNEMLİ (bug fix, aynı DEM hazırlama script'indeki gibi): grid zaten
        # eşleşse bile dosya CANONICAL isimle (aligned_inputs/<name>.tif)
        # KOPYALANIR. Önceden bu durumda hiçbir dosya yazılmıyordu ve Step7D
        # "aligned_inputs/<name>.tif bulunamadı" ile karşılaşıyordu --
        # Step7B'nin log'u "başarılı" görünse de.
        if not out_path.exists() or force:
            import shutil
            shutil.copyfile(source_path, out_path)
        diag.update({
            "aligned": False,
            "aligned_path": str(out_path),
            "reused_existing": out_path.exists() and not force,
            "aligned_valid_pixel_count": src_valid_count,
            "aligned_valid_fraction": (
                src_valid_count / (ref_w * ref_h) if ref_w * ref_h else 0.0
            ),
        })
        return out_path, diag

    if out_path.exists() and not force:
        with rasterio.open(out_path) as a:
            a_arr = a.read(1, masked=True)
            aligned_valid = int((~np.ma.getmaskarray(a_arr)).sum())
        diag.update({
            "aligned": True, "aligned_path": str(out_path), "reused_existing": True,
            "aligned_valid_pixel_count": aligned_valid,
            "aligned_valid_fraction": aligned_valid / (ref_w * ref_h) if ref_w * ref_h else 0.0,
        })
        return out_path, diag

    resampling_enum = Resampling.nearest if resampling == "nearest" else Resampling.bilinear
    dst_dtype = src_dtype if resampling == "nearest" else "float32"
    if src_nodata is not None:
        dst_nodata = src_nodata
    else:
        dst_nodata = 0 if resampling == "nearest" else float("nan")

    dst = np.full((ref_h, ref_w), dst_nodata, dtype=dst_dtype)
    with rasterio.open(source_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src_t, src_crs=src_crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            dst_nodata=dst_nodata,
            resampling=resampling_enum,
        )

    out_profile = {
        "driver": "GTiff", "width": ref_w, "height": ref_h, "count": 1,
        "dtype": dst_dtype, "crs": ref_crs, "transform": ref_transform,
        "nodata": dst_nodata, "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **out_profile) as dst_ds:
        dst_ds.write(dst, 1)

    if isinstance(dst_nodata, float) and np.isnan(dst_nodata):
        valid_mask = np.isfinite(dst)
    else:
        valid_mask = (dst != dst_nodata)
    aligned_valid = int(valid_mask.sum())

    diag.update({
        "aligned": True,
        "aligned_path": str(out_path),
        "reused_existing": False,
        "aligned_valid_pixel_count": aligned_valid,
        "aligned_valid_fraction": aligned_valid / (ref_w * ref_h) if ref_w * ref_h else 0.0,
    })
    log.info(
        "[align] %s: kaynak %dx%d (%s) -> referans %dx%d, resampling=%s, "
        "hizalı geçerli piksel oranı=%.4f -> %s",
        feature_name, src_h, src_w, src_crs, ref_h, ref_w, resampling,
        diag["aligned_valid_fraction"], out_path,
    )
    return out_path, diag


def align_features_to_reference(
    ctx: dict,
    target_path: Path,
    core_features: list[dict],
    optional_features: list[dict],
    force: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Kozan-dışı (deney-farkında) çalıştırmalar için: TÜM feature rasterlarını
    (MODIS mean/std dahil) Step7B'nin kendi pencere-pencere reproject
    mantığına GÜVENMEDEN, önceden Step5 referans gridine (target_path)
    hizalar. Hizalanmış dosyalar
    outputs/experiments/<experiment_id>/step7b/aligned_inputs/ altına
    yazılır (debug için saklanır).

    MODIS (1 km) gibi çok daha kaba çözünürlüklü sürekli rasterlar için
    bilinear, landcover gibi kategorik rasterlar için nearest-neighbor
    kullanılır (build_feature_registry'nin zaten atadığı "resampling" alanı
    ile tutarlı).

    Döner: (aligned_core_features, aligned_optional_features, alignment_diagnostics)
    """
    output_dir = ctx["step7b_output_dir"] / "aligned_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(target_path) as ref:
        ref_w, ref_h, ref_crs, ref_transform = ref.width, ref.height, ref.crs, ref.transform

    diagnostics: list[dict] = []

    def _align_list(features: list[dict]) -> list[dict]:
        aligned = []
        for feat in features:
            aligned_path, diag = align_feature_to_reference(
                feat["name"], Path(feat["path"]), feat["resampling"],
                ref_w, ref_h, ref_crs, ref_transform, output_dir, force=force,
            )
            diagnostics.append(diag)
            aligned.append({**feat, "path": aligned_path})
        return aligned

    aligned_core = _align_list(core_features)
    aligned_optional = _align_list(optional_features)
    return aligned_core, aligned_optional, diagnostics


def build_dataset(
    target_path: Path,
    target_band: int,
    core_features: list[dict],
    optional_features: list[dict],
    tile_size: int,
    max_samples: int | None,
    sample_fraction: float | None,
    stratify: bool,
    seed: int,
    modis_source: Path | None,
) -> tuple["dict", dict]:
    """
    Pencere pencere geçerli örnekleri toplar (chunked) ve sayaçları döndürür.

    Tam rasterları RAM'e almaz; her pencere için target + feature pencereleri
    okunur, geçerli maske kurulur ve örnekler biriktirilir.
    """
    rng = np.random.default_rng(seed)

    # Tüm feature rasterlarını aç (dosya tanıtıcıları; veriyi yüklemez).
    feature_handles = []
    all_features = core_features + optional_features

    counters = {
        "total_candidate_pixels": 0,
        "total_valid_samples_before_cap": 0,
        "dropped_nan_target": 0,
        "dropped_invalid_target_range": 0,
        "dropped_nan_required_features": 0,
        "dropped_invalid_ndvi": 0,
        "dropped_invalid_slope": 0,
        "dropped_invalid_elevation": 0,
        "window_count": 0,
    }

    chunks: list[dict] = []

    with rasterio.open(target_path) as target_src:
        target_crs = target_src.crs
        feature_meta = []
        for feat in all_features:
            src = rasterio.open(feat["path"])
            same_grid = (
                src.crs == target_crs
                and src.width == target_src.width
                and src.height == target_src.height
                and src.transform == target_src.transform
            )
            feature_handles.append(src)
            feature_meta.append({**feat, "same_grid": same_grid})

        try:
            req_names = [f["name"] for f in core_features if f.get("required")]
            ndvi_present = any(f["name"] == "ndvi" for f in all_features)
            slope_present = any(f["name"] == "slope" for f in all_features)
            elev_present = any(f["name"] == "elevation" for f in all_features)

            for write_win, _read_win, _core_off in iter_windows(
                target_src, tile_size_pixels=tile_size, overlap_pixels=0
            ):
                counters["window_count"] += 1
                win_h = int(write_win.height)
                win_w = int(write_win.width)
                if win_h == 0 or win_w == 0:
                    continue

                target_transform = window_transform(write_win, target_src.transform)

                target_arr = target_src.read(
                    target_band, window=write_win, masked=True
                ).astype("float32").filled(np.nan)
                counters["total_candidate_pixels"] += int(target_arr.size)

                # Feature pencereleri (target grid'ine resample)
                feat_arrays: dict[str, np.ndarray] = {}
                for src, meta in zip(feature_handles, feature_meta):
                    feat_arrays[meta["name"]] = read_feature_into_target_window(
                        src, meta["resampling"], write_win,
                        target_transform, target_crs, win_h, win_w,
                        meta["same_grid"],
                    )

                # --- Geçerlilik maskesi (clamp YOK; satır DROP) ---
                valid = np.isfinite(target_arr)
                counters["dropped_nan_target"] += int((~valid).sum())

                in_range = (
                    (target_arr >= STEP7B_MIN_TARGET_CELSIUS)
                    & (target_arr <= STEP7B_MAX_TARGET_CELSIUS)
                )
                bad_range = valid & (~in_range)
                counters["dropped_invalid_target_range"] += int(bad_range.sum())
                valid &= in_range

                # Zorunlu feature'lar finite olmalı
                for rname in req_names:
                    if rname in feat_arrays:
                        finite = np.isfinite(feat_arrays[rname])
                        counters["dropped_nan_required_features"] += int(
                            (valid & (~finite)).sum()
                        )
                        valid &= finite

                # NDVI fiziksel aralık
                if ndvi_present:
                    nd = feat_arrays.get("ndvi")
                    nd_finite = np.isfinite(nd)
                    nd_in_range = nd_finite & (nd >= NDVI_MIN) & (nd <= NDVI_MAX)
                    if STEP7B_REQUIRE_NDVI:
                        # Zorunlu: NaN veya aralık-dışı -> drop.
                        counters["dropped_invalid_ndvi"] += int(
                            (valid & (~nd_in_range)).sum()
                        )
                        valid &= nd_in_range
                    else:
                        # Opsiyonel: yalnız finite-ama-aralık-dışı drop; NaN'a izin ver.
                        bad = valid & nd_finite & (~nd_in_range)
                        counters["dropped_invalid_ndvi"] += int(bad.sum())
                        valid &= (nd_in_range | ~nd_finite)

                # Slope fiziksel aralık (varsa)
                if slope_present:
                    sl = feat_arrays.get("slope")
                    sl_finite = np.isfinite(sl)
                    sl_ok = sl_finite & (sl >= SLOPE_MIN) & (sl <= SLOPE_MAX)
                    if STEP7B_REQUIRE_DEM:
                        counters["dropped_invalid_slope"] += int((valid & (~sl_ok)).sum())
                        valid &= sl_ok
                    else:
                        bad = valid & sl_finite & (~sl_ok)
                        counters["dropped_invalid_slope"] += int(bad.sum())
                        valid &= (sl_ok | ~sl_finite)

                # Elevation fiziksel aralık (varsa)
                if elev_present:
                    el = feat_arrays.get("elevation")
                    el_finite = np.isfinite(el)
                    el_ok = el_finite & (el >= ELEVATION_MIN) & (el <= ELEVATION_MAX)
                    if STEP7B_REQUIRE_DEM:
                        counters["dropped_invalid_elevation"] += int(
                            (valid & (~el_ok)).sum()
                        )
                        valid &= el_ok
                    else:
                        bad = valid & el_finite & (~el_ok)
                        counters["dropped_invalid_elevation"] += int(bad.sum())
                        valid &= (el_ok | ~el_finite)

                n_valid = int(valid.sum())
                if n_valid == 0:
                    continue
                counters["total_valid_samples_before_cap"] += n_valid

                rows_local, cols_local = np.where(valid)
                # Global piksel koordinatları
                global_rows = rows_local + int(write_win.row_off)
                global_cols = cols_local + int(write_win.col_off)

                # lon/lat (piksel merkezleri) target grid transform'undan
                xs, ys = rasterio.transform.xy(
                    target_src.transform,
                    global_rows.tolist(),
                    global_cols.tolist(),
                    offset="center",
                )
                lon = np.asarray(xs, dtype="float64")
                lat = np.asarray(ys, dtype="float64")

                chunk = {
                    "row": global_rows.astype("int32"),
                    "col": global_cols.astype("int32"),
                    "lon": lon,
                    "lat": lat,
                    "landsat_lst_celsius": target_arr[valid].astype("float32"),
                    "source_window_row": np.full(n_valid, int(write_win.row_off), "int32"),
                    "source_window_col": np.full(n_valid, int(write_win.col_off), "int32"),
                    "source_tile_id": np.full(
                        n_valid, counters["window_count"] - 1, "int32"
                    ),
                }
                for meta in feature_meta:
                    name = meta["name"]
                    chunk[name] = feat_arrays[name][valid].astype("float32")

                chunks.append(chunk)
        finally:
            for src in feature_handles:
                src.close()

    # Birleştir
    dataset = _concat_chunks(chunks)
    if dataset:
        # MODIS pixel id (stratifikasyon + grouped validation için) — örnekleme ÖNCESİ.
        mid = _modis_pixel_ids(dataset, modis_source)
        if mid is not None:
            dataset["_modis_pixel_id"] = mid  # geçici stratifikasyon anahtarı
            dataset["modis_pixel_id"] = mid.astype("int64")  # kalıcı kolon
    final = _sample_dataset(
        dataset, max_samples, sample_fraction, stratify, rng, counters
    )
    return final, counters


def _concat_chunks(chunks: list[dict]) -> dict:
    """Pencere chunk'larını tek sözlükte birleştirir."""
    if not chunks:
        return {}
    keys = chunks[0].keys()
    out = {}
    for k in keys:
        out[k] = np.concatenate([c[k] for c in chunks])
    return out


def _modis_pixel_ids(dataset: dict, modis_path: Path | None) -> np.ndarray | None:
    """
    Her örneğin lon/lat'ını MODIS kaynak grid'ine eşleyerek modis_pixel_id üretir.

    Stratifikasyon ve gelecekteki grouped validation (leakage önleme) için. MODIS
    kaynak rasterı yoksa None döner.
    """
    if modis_path is None or not modis_path.exists():
        return None
    try:
        with rasterio.open(modis_path) as msrc:
            inv = ~msrc.transform
            cols, rows = inv * (dataset["lon"], dataset["lat"])
            mrow = np.floor(rows).astype("int64")
            mcol = np.floor(cols).astype("int64")
            dataset["modis_pixel_row"] = mrow.astype("int32")
            dataset["modis_pixel_col"] = mcol.astype("int32")
            width = msrc.width
            return (mrow * width + mcol).astype("int64")
    except Exception:  # noqa: BLE001
        return None


def _sample_dataset(
    dataset: dict,
    max_samples: int | None,
    sample_fraction: float | None,
    stratify: bool,
    rng: np.random.Generator,
    counters: dict,
) -> dict:
    """Deterministik (opsiyonel stratified) örnekleme + cap uygular."""
    if not dataset:
        return dataset
    n = dataset["row"].size

    # sample_fraction önce uygulanır (varsa)
    target_n = n
    if sample_fraction is not None and 0 < sample_fraction < 1:
        target_n = int(n * sample_fraction)
    if max_samples is not None:
        target_n = min(target_n, max_samples)
    if target_n >= n:
        return dataset  # cap gereksiz

    modis_id = dataset.get("_modis_pixel_id")
    if stratify and modis_id is not None and modis_id.size == n:
        idx = _stratified_indices(modis_id, target_n, rng)
        counters["sampling_strategy"] = "stratified_by_modis_pixel"
    else:
        idx = rng.choice(n, size=target_n, replace=False)
        counters["sampling_strategy"] = (
            "uniform_random" if not stratify else "uniform_random_no_modis_grid"
        )

    idx.sort()
    return {k: v[idx] for k, v in dataset.items() if not k.startswith("_")}


def _stratified_indices(
    group_id: np.ndarray, target_n: int, rng: np.random.Generator
) -> np.ndarray:
    """
    MODIS pikseli başına dengeli örnekleme: tek bir kaba pikselden çok örnek alma.

    Her gruba round-robin ile kota dağıtılır; deterministik (seed'li rng).
    """
    order = rng.permutation(group_id.size)
    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(int(group_id[i]), []).append(int(i))

    selected: list[int] = []
    group_lists = list(groups.values())
    pos = 0
    while len(selected) < target_n and group_lists:
        advanced = False
        for g in group_lists:
            if pos < len(g):
                selected.append(g[pos])
                advanced = True
                if len(selected) >= target_n:
                    break
        if not advanced:
            break
        pos += 1
    return np.asarray(selected[:target_n], dtype="int64")


# =============================================================================
# Çıktı yazımı
# =============================================================================
def _column_order(dataset: dict) -> list[str]:
    """Required + opsiyonel kolonları istenen sırayla düzenler."""
    preferred = [
        "row", "col", "lon", "lat", "landsat_lst_celsius",
        "modis_lst_mean_celsius", "modis_lst_std_celsius", "modis_context_zscore",
        "ndvi", "elevation", "slope", "landcover",
        "anomaly_zscore", "current_tvdi", "tvdi_difference",
        "source_tile_id", "source_window_row", "source_window_col",
        "modis_pixel_row", "modis_pixel_col", "modis_pixel_id",
    ]
    cols = [c for c in preferred if c in dataset]
    # Beklenmeyen ekstra kolonları da sona ekle
    cols += [c for c in dataset if c not in cols and not c.startswith("_")]
    return cols


def write_outputs(
    dataset: dict, formats: list[str], force: bool
) -> dict:
    """Parquet/CSV yazar. pyarrow yoksa CSV yazılır, parquet_written=False."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = _column_order(dataset)
    n = dataset[cols[0]].size if cols else 0

    result = {
        "parquet_written": False,
        "csv_written": False,
        "parquet_available": False,
        "output_files": [],
        "columns": cols,
    }

    csv_path = OUTPUTS_DIR / "downscaling_training_samples.csv"
    parquet_path = OUTPUTS_DIR / "downscaling_training_samples.parquet"

    want_parquet = "parquet" in formats
    want_csv = "csv" in formats

    # Parquet (pyarrow varsa)
    if want_parquet:
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq
            result["parquet_available"] = True
            table = pa.table({c: dataset[c] for c in cols})
            pq.write_table(table, parquet_path)
            result["parquet_written"] = True
            result["output_files"].append(str(parquet_path))
        except Exception as exc:  # noqa: BLE001
            log.warning("Parquet yazılamadı (pyarrow yok/başarısız): %s", exc)

    # CSV (her zaman dene; en az bir format başarılı olmalı)
    if want_csv or not result["parquet_written"]:
        try:
            _write_csv(csv_path, dataset, cols)
            result["csv_written"] = True
            result["output_files"].append(str(csv_path))
        except Exception as exc:  # noqa: BLE001
            log.error("CSV yazılamadı: %s", exc)

    result["final_sample_count"] = int(n)
    return result


def _write_csv(path: Path, dataset: dict, cols: list[str]) -> None:
    """Bağımlılıksız CSV yazımı (numpy ile satır satır birleştirme)."""
    arrays = [dataset[c] for c in cols]
    n = arrays[0].size
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        # Vektörize string birleştirme (chunked, bellek dostu)
        chunk = 50000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            lines = []
            for i in range(start, end):
                vals = []
                for a in arrays:
                    v = a[i]
                    if isinstance(v, (np.floating, float)):
                        vals.append("" if not np.isfinite(v) else repr(float(v)))
                    else:
                        vals.append(str(int(v)))
                lines.append(",".join(vals))
            f.write("\n".join(lines) + "\n")


def write_stats_and_summary(
    target_path: Path,
    target_band: int,
    core_features: list[dict],
    optional_features: list[dict],
    missing_optional: list[dict],
    counters: dict,
    out_result: dict,
    grid_info: dict,
    tile_size: int,
    max_samples: int | None,
    sample_fraction: float | None,
    stratify: bool,
    seed: int,
    warnings_list: list[str],
    alignment_diagnostics: list[dict] | None = None,
    aligned_inputs_dir: Path | None = None,
) -> tuple[Path, Path]:
    """downscaling_dataset_stats.json + downscaling_dataset_summary.md yazar."""
    required_features = [f["name"] for f in core_features if f.get("required")]
    optional_used = [f["name"] for f in optional_features]
    optional_missing = [m["name"] for m in missing_optional]

    drop_keys = [
        "dropped_nan_target", "dropped_invalid_target_range",
        "dropped_nan_required_features", "dropped_invalid_ndvi",
        "dropped_invalid_slope", "dropped_invalid_elevation",
    ]
    drop_summary = {k: counters.get(k, 0) for k in drop_keys}
    dominant_drop_filter = (
        max(drop_summary, key=drop_summary.get) if any(drop_summary.values()) else None
    )

    stats = {
        "step": "step7b_prepare_downscaling_dataset",
        "created_at": datetime.now().isoformat(),
        "no_model_trained": True,
        "target_path": str(target_path),
        "target_band": target_band,
        "target_name": "landsat_lst_celsius",
        "feature_paths": {f["name"]: str(f["path"]) for f in core_features + optional_features},
        "aligned_inputs_dir": str(aligned_inputs_dir) if aligned_inputs_dir else None,
        "aligned_feature_paths": (
            {f["name"]: str(f["path"]) for f in core_features + optional_features}
            if aligned_inputs_dir else None
        ),
        "reference_grid": grid_info,
        "tile_size": tile_size,
        "window_count": counters.get("window_count"),
        "total_candidate_pixels": counters.get("total_candidate_pixels"),
        "total_valid_samples_before_cap": counters.get("total_valid_samples_before_cap"),
        "final_sample_count": out_result.get("final_sample_count"),
        "dropped_nan_target": counters.get("dropped_nan_target"),
        "dropped_invalid_target_range": counters.get("dropped_invalid_target_range"),
        "dropped_nan_required_features": counters.get("dropped_nan_required_features"),
        "dropped_invalid_ndvi": counters.get("dropped_invalid_ndvi"),
        "dropped_invalid_slope": counters.get("dropped_invalid_slope"),
        "dropped_invalid_elevation": counters.get("dropped_invalid_elevation"),
        "dominant_drop_filter": dominant_drop_filter,
        "alignment_diagnostics": alignment_diagnostics or [],
        "output_files": out_result.get("output_files"),
        "columns": out_result.get("columns"),
        "parquet_written": out_result.get("parquet_written"),
        "parquet_available": out_result.get("parquet_available"),
        "csv_written": out_result.get("csv_written"),
        "random_seed": seed,
        "max_samples": max_samples,
        "sample_fraction": sample_fraction,
        "stratify_by_modis_pixel": stratify,
        "sampling_strategy": counters.get("sampling_strategy", "no_cap_applied"),
        "required_features": required_features,
        "optional_features_used": optional_used,
        "optional_features_missing": optional_missing,
        "target_valid_range_celsius": [
            STEP7B_MIN_TARGET_CELSIUS, STEP7B_MAX_TARGET_CELSIUS,
        ],
        "warnings": warnings_list,
    }

    stats_path = OUTPUTS_DIR / "downscaling_dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Step7B: MODIS Downscaling Training Dataset",
        "",
        "**Step7B prepares a MODIS->Landsat LST downscaling training dataset. "
        "No model is trained here.** This is **not** fire-risk validation.",
        "",
        f"- Created at: `{stats['created_at']}`",
        f"- Target: `landsat_lst_celsius` from `{target_path.name}` (band {target_band})",
        "- Features include: MODIS LST context, NDVI, DEM elevation, slope, "
        "land cover, and spatial coordinates"
        + (" (+ optional TVDI/anomaly)" if optional_used else "") + ".",
        f"- Tile size (pixels): `{tile_size}`",
        f"- Windows processed: `{counters.get('window_count')}`",
        f"- Total candidate pixels: `{counters.get('total_candidate_pixels')}`",
        f"- Valid samples before cap: `{counters.get('total_valid_samples_before_cap')}`",
        f"- **Final sample count: `{out_result.get('final_sample_count')}`**",
        f"- Sampling strategy: `{stats['sampling_strategy']}`",
        "",
        "## Dropped (masked, not clamped)",
        "",
        f"- NaN target: `{counters.get('dropped_nan_target')}`",
        f"- Target out of Celsius range "
        f"[{STEP7B_MIN_TARGET_CELSIUS}, {STEP7B_MAX_TARGET_CELSIUS}]: "
        f"`{counters.get('dropped_invalid_target_range')}`",
        f"- NaN required features: `{counters.get('dropped_nan_required_features')}`",
        f"- Invalid NDVI: `{counters.get('dropped_invalid_ndvi')}`",
        f"- Invalid slope: `{counters.get('dropped_invalid_slope')}`",
        f"- Invalid elevation: `{counters.get('dropped_invalid_elevation')}`",
        "",
        "## Features",
        "",
        f"- Required: `{', '.join(required_features) or 'none'}`",
        f"- Optional used: `{', '.join(optional_used) or 'none'}`",
        f"- Optional missing: `{', '.join(optional_missing) or 'none'}`",
        "",
        "## Outputs",
        "",
    ]
    for f in out_result.get("output_files", []):
        md.append(f"- `{f}`")
    md.extend([
        f"- parquet_written: `{out_result.get('parquet_written')}` "
        f"(pyarrow available: `{out_result.get('parquet_available')}`)",
        f"- csv_written: `{out_result.get('csv_written')}`",
        "",
        "## Intended use",
        "",
        "This dataset is intended for **later** RF/XGBoost or similar MODIS "
        "downscaling (Step7C / Step8). No model is trained in Step7B, and this "
        "is kept separate from any fire-risk modeling.",
    ])
    if warnings_list:
        md.extend(["", "## Warnings", ""])
        md.extend(f"- {w}" for w in warnings_list)

    summary_path = OUTPUTS_DIR / "downscaling_dataset_summary.md"
    summary_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return stats_path, summary_path


# =============================================================================
# Ana akış
# =============================================================================
def main(
    max_samples: int | None = STEP7B_MAX_SAMPLES,
    tile_size: int = STEP7B_TILE_SIZE,
    output_format: str = "both",
    force: bool = False,
    include_optional_tvdi: bool = STEP7B_INCLUDE_OPTIONAL_TVDI_FEATURES,
    include_optional_anomaly: bool = STEP7B_INCLUDE_OPTIONAL_ANOMALY_FEATURES,
    ctx: dict | None = None,
    legacy_modis_compatibility: "LegacyModisCompatibilityAttestation | None" = None,
) -> dict:
    """`legacy_modis_compatibility` DEFAULTS TO None -> strict MODIS validation.

    It is never set by the CLI and has no command-line flag; only a caller that
    can produce a hash-verified :class:`LegacyModisCompatibilityAttestation` can
    reach the narrow historical-compatibility path.
    """
    log.info("=" * 60)
    log.info(
        "STEP 7B BAŞLIYOR (MODIS downscaling training dataset)%s",
        f" [experiment={ctx['experiment_id']}]" if ctx else "",
    )
    log.info("=" * 60)

    warnings_list: list[str] = []

    # Overwrite kontrolü
    existing = [
        OUTPUTS_DIR / "downscaling_training_samples.parquet",
        OUTPUTS_DIR / "downscaling_training_samples.csv",
        OUTPUTS_DIR / "downscaling_dataset_stats.json",
        OUTPUTS_DIR / "downscaling_dataset_summary.md",
    ]
    if any(p.exists() for p in existing) and not force:
        present = [p.name for p in existing if p.exists()]
        raise SystemExit(
            "Step7B çıktıları zaten var ("
            + ", ".join(present)
            + "). Üzerine yazmak için --force verin."
        )

    target_path, target_band = resolve_target(ctx)
    if target_path is None:
        raise SystemExit(
            "Landsat LST target rasterı bulunamadı. Beklenen: "
            + (
                f"{ctx['step5_output_dir']}/current_period_median_celsius.tif"
                if ctx else
                "outputs/step5/current_period_median_celsius.tif veya "
                "data/current_period/landsat_current_period_*days.tif"
            )
        )
    log.info("Target: %s (band %s)", target_path, target_band)

    core_features, optional_features, missing_optional = build_feature_registry(
        include_tvdi=include_optional_tvdi,
        include_anomaly=include_optional_anomaly,
        ctx=ctx,
    )
    log.info(
        "Core features: %s | Optional used: %s | Optional missing: %s",
        [f["name"] for f in core_features],
        [f["name"] for f in optional_features],
        [m["name"] for m in missing_optional],
    )

    # --- Deney-farkında (Kozan-dışı) girdi hizalama ---
    # MODIS (1 km) gibi kaba çözünürlüklü rasterları Step5 referans gridine
    # (target_path) ÖNCEDEN hizalar (bilinear/nearest -- feature tipine göre),
    # pencere-pencere reproject'e GÜVENMEZ. Hizalanmış dosyalar debug için
    # outputs/experiments/<id>/step7b/aligned_inputs/ altına yazılır.
    alignment_diagnostics: list[dict] = []
    use_ctx_alignment = ctx is not None and not ctx.get("is_kozan")
    if use_ctx_alignment:
        # --- MODIS kaynak-raster (hizalama ÖNCESİ) doğrulaması ---
        # Yalnızca deney-farkında (Kozan-dışı) yolda: Kozan'ın legacy MODIS
        # context ürünü (farklı bir üretim zinciri, korunan/frozen davranış)
        # bu kontrole HİÇ TABİ TUTULMAZ. Herhangi bir kural ihlalinde
        # Step7BModisValidationError (SystemExit) fırlatılır -- hizalama HİÇ
        # ÇALIŞMAZ.
        modis_source_diagnostics = validate_modis_source_rasters(
            core_features,
            experiment_id=ctx.get("experiment_id"),
            legacy_modis_compatibility=legacy_modis_compatibility,
        )

        log.info("Deney-farkında girdi hizalama başlıyor (aligned_inputs/)...")
        core_features, optional_features, alignment_diagnostics = align_features_to_reference(
            ctx, target_path, core_features, optional_features, force=force,
        )
        with rasterio.open(target_path) as tsrc:
            target_valid_mask = np.isfinite(
                tsrc.read(target_band, masked=True).astype("float32").filled(np.nan)
            )
        target_valid_count = int(target_valid_mask.sum())
        for diag in alignment_diagnostics:
            with rasterio.open(diag["aligned_path"]) as fsrc:
                farr = fsrc.read(1, masked=True)
                fvalid = ~np.ma.getmaskarray(farr)
                fvalid = fvalid & np.isfinite(np.ma.filled(farr.astype("float64"), np.nan))
            overlap = int((target_valid_mask & fvalid).sum())
            diag["overlap_with_target_pixel_count"] = overlap
            diag["overlap_with_target_fraction_of_target_valid"] = (
                overlap / target_valid_count if target_valid_count else 0.0
            )
            log.info(
                "[align-overlap] %s: target ile örtüşen geçerli piksel=%d "
                "(target geçerli piksellerinin %.4f'ü)",
                diag["name"], overlap, diag["overlap_with_target_fraction_of_target_valid"],
            )
            if overlap == 0:
                log.warning(
                    "[align-overlap] %s: target ile SIFIR örtüşme! Bu özellik "
                    "muhtemelen yanlış AOI/grid kapsıyor (or. paylaşılan DEM "
                    "farklı bir bölgeyi kapsıyorsa).", diag["name"],
                )
            # MODIS kaynak-taraf teşhisini (nodata/geçerlilik/min-max-mean-
            # medyan/validation_status), isimle eşleştirerek AYNI diagnostic
            # kaydına (downscaling_dataset_stats.json'a yazılan) ekle.
            if diag["name"] in modis_source_diagnostics:
                diag["modis_source_validation"] = modis_source_diagnostics[diag["name"]]

    # Zorunlu feature eksikse uyar (NDVI/DEM)
    core_names = {f["name"] for f in core_features}
    if STEP7B_REQUIRE_NDVI and "ndvi" not in core_names:
        warnings_list.append("Required NDVI feature not found; samples may be empty.")
    if STEP7B_REQUIRE_DEM and ("elevation" not in core_names or "slope" not in core_names):
        warnings_list.append("Required DEM (elevation/slope) feature missing.")

    # Grid info
    with rasterio.open(target_path) as tsrc:
        grid_info = {
            "path": str(target_path),
            "width": int(tsrc.width),
            "height": int(tsrc.height),
            "crs": str(tsrc.crs),
            "transform": [tsrc.transform.a, tsrc.transform.b, tsrc.transform.c,
                          tsrc.transform.d, tsrc.transform.e, tsrc.transform.f],
        }
        if not tsrc.crs:
            warnings_list.append("Target raster has no CRS.")
        if tsrc.width == 0 or tsrc.height == 0:
            raise SystemExit("Target raster has invalid dimensions.")

    modis_source = None
    for f in core_features:
        if f["name"] == "modis_lst_mean_celsius":
            modis_source = f["path"]
            break

    dataset, counters = build_dataset(
        target_path, target_band, core_features, optional_features,
        tile_size=tile_size,
        max_samples=max_samples,
        sample_fraction=STEP7B_SAMPLE_FRACTION,
        stratify=STEP7B_STRATIFY_BY_MODIS_PIXEL,
        seed=STEP7B_RANDOM_SEED,
        modis_source=modis_source,
    )

    if dataset and "modis_pixel_id" not in dataset:
        warnings_list.append(
            "MODIS source grid unavailable; modis_pixel_id not added."
        )

    if not dataset or dataset.get("row") is None or dataset["row"].size == 0:
        warnings_list.append("No valid samples produced.")
        out_result = {
            "final_sample_count": 0, "output_files": [], "columns": [],
            "parquet_written": False, "parquet_available": False, "csv_written": False,
        }
    else:
        formats = (
            ["parquet", "csv"] if output_format == "both" else [output_format]
        )
        out_result = write_outputs(dataset, formats, force)

    stats_path, summary_path = write_stats_and_summary(
        target_path, target_band, core_features, optional_features, missing_optional,
        counters, out_result, grid_info, tile_size, max_samples,
        STEP7B_SAMPLE_FRACTION, STEP7B_STRATIFY_BY_MODIS_PIXEL,
        STEP7B_RANDOM_SEED, warnings_list, alignment_diagnostics=alignment_diagnostics,
        aligned_inputs_dir=(ctx["step7b_output_dir"] / "aligned_inputs") if use_ctx_alignment else None,
    )

    if out_result.get("final_sample_count", 0) == 0:
        drop_keys = [
            "dropped_nan_target", "dropped_invalid_target_range",
            "dropped_nan_required_features", "dropped_invalid_ndvi",
            "dropped_invalid_slope", "dropped_invalid_elevation",
        ]
        drop_summary = {k: counters.get(k, 0) for k in drop_keys}
        dominant_filter = max(drop_summary, key=drop_summary.get) if any(drop_summary.values()) else None
        zero_overlap_features = [
            d["name"] for d in alignment_diagnostics
            if d.get("overlap_with_target_pixel_count") == 0
        ]
        raise SystemExit(
            "Step7B FAIL-FAST: final_sample_count=0 -- eğitim verisi üretilemedi. "
            f"Piksel bırakma sayaçları: {drop_summary} (en çok düşüren: {dominant_filter}). "
            + (
                f"Target ile SIFIR örtüşen özellikler: {zero_overlap_features} -- "
                "muhtemelen bu özellik(ler) yanlış AOI/grid kapsıyor. "
                if zero_overlap_features else
                "Hizalama diagnostikleri için: "
            )
            + f"{stats_path}"
        )

    log.info("Final sample count: %s", out_result.get("final_sample_count"))
    log.info("Stats: %s", stats_path)
    log.info("Summary: %s", summary_path)
    log.info("=" * 60)
    log.info("STEP 7B TAMAMLANDI (no model trained)")
    log.info("=" * 60)

    return {
        "final_sample_count": out_result.get("final_sample_count"),
        "stats_path": str(stats_path),
        "summary_path": str(summary_path),
        "output_files": out_result.get("output_files"),
        "experiment_id": ctx["experiment_id"] if ctx else None,
        "modis_compatibility_mode": (
            legacy_modis_compatibility.mode
            if legacy_modis_compatibility is not None else "strict_default_guard"
        ),
    }


def run_step7b(ctx: dict | None = None, force: bool = False, **kwargs) -> dict:
    """
    Step7B MODIS downscaling training dataset'ini uretir.

    ctx: None ise (varsayilan) legacy Kozan davranisi BIREBIR korunur.
        Verilirse (Kozan-disi), outputs/experiments/<experiment_id>/step7b'ye
        yazar ve tum girdileri (target/NDVI/anomaly/TVDI) namespaced
        Step5/Step5C'den okur (DEM shared/read-only, landcover Step6A
        gate-input'undan -- bkz. build_feature_registry docstring).

    kwargs icinde OPSIYONEL `legacy_modis_compatibility` gecirilebilir; verilmezse
    (varsayilan) MODIS dogrulamasi STRICT kalir. Bkz.
    :class:`LegacyModisCompatibilityAttestation`.
    """
    global OUTPUTS_DIR

    use_ctx = ctx is not None and not ctx.get("is_kozan")
    saved = OUTPUTS_DIR
    try:
        if use_ctx:
            OUTPUTS_DIR = ctx["step7b_output_dir"]
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            log.info(
                "[experiment=%s] Step7B ctx override aktif. output_dir=%s",
                ctx["experiment_id"], OUTPUTS_DIR,
            )
        return main(force=force, ctx=ctx if use_ctx else None, **kwargs)
    finally:
        OUTPUTS_DIR = saved


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step7B: MODIS downscaling training dataset builder (no model trained)."
    )
    parser.add_argument("--max-samples", type=int, default=STEP7B_MAX_SAMPLES)
    parser.add_argument("--tile-size", type=int, default=STEP7B_TILE_SIZE)
    parser.add_argument(
        "--output-format", choices=["csv", "parquet", "both"], default="both"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-optional-tvdi", action="store_true")
    parser.add_argument("--no-optional-anomaly", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        max_samples=args.max_samples,
        tile_size=args.tile_size,
        output_format=args.output_format,
        force=args.force,
        include_optional_tvdi=not args.no_optional_tvdi and STEP7B_INCLUDE_OPTIONAL_TVDI_FEATURES,
        include_optional_anomaly=not args.no_optional_anomaly and STEP7B_INCLUDE_OPTIONAL_ANOMALY_FEATURES,
    )