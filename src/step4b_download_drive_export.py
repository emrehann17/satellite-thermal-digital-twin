"""
step4b_download_drive_exports.py

Step4 ile Step5 arasındaki offline geçiş katmanı.

Yapılanlar:
    - Step4'te tamamlanan Google Drive export dosyalarını indirir.
    - İndirilen GeoTIFF dosyalarını Step5'in beklediği data klasörlerine dağıtır.
    - Download/placement metadata bilgisini ayrı kaydeder.
"""

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import (
    DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT,
    DRIVE_DOWNLOAD_OVERWRITE,
    DRIVE_DOWNLOAD_STAGING_SUBDIR,
    GOOGLE_DRIVE_EXPORT_FOLDER_ID,
    GOOGLE_DRIVE_EXPORT_FOLDER_URL,
    LANDSAT_EXPORT,
    LANDSAT_NDVI_EXPORT,
    MODIS_EXPORT,
    DEM_EXPORT,
    CURRENT_PERIOD_DAYS,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from utils.geotiff_validation import validate_geotiff_basic


BASE_DIR = PROJECT_ROOT
OUTPUTS_DIR = BASE_DIR / "outputs" / "step4b"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step4b")


def extract_google_drive_folder_id(url_or_id: str | None) -> str | None:
    """Google Drive klasör URL'sinden veya doğrudan ID değerinden klasör ID'sini çıkarır."""
    if not url_or_id:
        return None

    value = str(url_or_id).strip()
    if not value:
        return None

    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", value)
    if match:
        return match.group(1)

    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", value):
        return value

    return None


def resolve_drive_folder_reference() -> tuple[str | None, str | None]:
    """Config/env değerlerinden Drive klasör URL ve ID referansını çözer."""
    folder_id = extract_google_drive_folder_id(GOOGLE_DRIVE_EXPORT_FOLDER_ID)
    if folder_id is None:
        folder_id = extract_google_drive_folder_id(GOOGLE_DRIVE_EXPORT_FOLDER_URL)

    folder_url = None
    if folder_id is not None:
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    elif GOOGLE_DRIVE_EXPORT_FOLDER_URL:
        folder_url = str(GOOGLE_DRIVE_EXPORT_FOLDER_URL).strip()

    return folder_url, folder_id


def ensure_step5_data_dirs() -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    """İndirilen rasterlar için Step5'in beklediği yerel veri klasörlerini hazırlar."""
    lst_dir = BASE_DIR / "data" / "landsat_timeseries"
    qa_dir = BASE_DIR / "data" / "landsat_qa"
    current_dir = BASE_DIR / "data" / "current_period"
    modis_dir = BASE_DIR / "data" / "modis"
    ndvi_baseline_dir = BASE_DIR / "data" / "ndvi_timeseries"
    ndvi_current_dir = BASE_DIR / "data" / "ndvi_current_period"
    landcover_dir = BASE_DIR / "data" / "landcover"
    dem_dir = BASE_DIR / "data" / "dem"

    lst_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    modis_dir.mkdir(parents=True, exist_ok=True)
    ndvi_baseline_dir.mkdir(parents=True, exist_ok=True)
    ndvi_current_dir.mkdir(parents=True, exist_ok=True)
    landcover_dir.mkdir(parents=True, exist_ok=True)
    dem_dir.mkdir(parents=True, exist_ok=True)

    return (
        lst_dir,
        qa_dir,
        current_dir,
        modis_dir,
        ndvi_baseline_dir,
        ndvi_current_dir,
        landcover_dir,
        dem_dir,
    )


def copy_with_overwrite_control(source_path: Path, target_path: Path) -> None:
    """Dosyayı hedefe kopyalar; overwrite ayarı kapalıysa mevcut dosyayı korur."""
    if target_path.exists() and not DRIVE_DOWNLOAD_OVERWRITE:
        log.info("Dosya zaten var, atlandı: %s", target_path)
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def is_landsat_qa_export_name(filename: str) -> bool:
    """
    Landsat QA export adını tek parça ve GEE parçalı dosya adlarında tanır.

    Örnekler:
        landsat_lst_dogu_akdeniz_2020-06-01_qa.tif
        landsat_lst_dogu_akdeniz_2020-06-01_qa-0000000000-0000000000.tif
    """
    stem = Path(filename).stem.lower()
    return bool(re.search(r"_qa($|[-_])", stem))


def resolve_dem_target_name(filename: str, product: str) -> str:
    """
    DEM export dosyasını kanonik isme (elevation.tif / slope.tif) eşler.

    GEE büyük export'ları parçalara böler ve dosya adına tile son ekleri ekler;
    örn. dem_elevation_dogu_akdeniz-0000000000-0000000000.tif. Bu durumda
    parçaların birbirini ezmemesi için tile son eki kanonik isme taşınır.
    """
    stem = Path(filename).stem
    prefix = DEM_EXPORT[product]["file_name_prefix"]
    suffix = stem[len(prefix):]  # tek parça için boş, parçalı için '-0000...-0000...'
    return f"{product}{suffix}.tif"


def classify_dem_tif(filename: str) -> str | None:
    """İndirilen GeoTIFF DEM elevation mı slope mı ürünü, değilse None döner."""
    lower_name = filename.lower()
    if lower_name.startswith(DEM_EXPORT["elevation"]["file_name_prefix"].lower()):
        return "elevation"
    if lower_name.startswith(DEM_EXPORT["slope"]["file_name_prefix"].lower()):
        return "slope"
    return None


def classify_downloaded_tif(source_path: Path) -> tuple[str, Path] | None:
    """İndirilen GeoTIFF'i Step5 data klasörlerinden doğru hedefe sınıflandırır."""
    (
        lst_dir,
        qa_dir,
        current_dir,
        modis_dir,
        ndvi_baseline_dir,
        ndvi_current_dir,
        landcover_dir,
        dem_dir,
    ) = ensure_step5_data_dirs()
    name = source_path.name
    lower_name = name.lower()

    if lower_name.startswith("landcover_esa_worldcover"):
        return "landcover", landcover_dir / name

    dem_product = classify_dem_tif(name)
    if dem_product is not None:
        target_name = resolve_dem_target_name(name, dem_product)
        return "dem", dem_dir / target_name

    if is_landsat_qa_export_name(name):
        return "baseline_qa", qa_dir / name

    if lower_name.startswith("current_ndvi_median"):
        return "ndvi_current_period", ndvi_current_dir / name

    if lower_name.startswith(LANDSAT_NDVI_EXPORT["file_name_prefix"].lower()):
        return "ndvi_baseline", ndvi_baseline_dir / name

    if lower_name.startswith("landsat_current_period_"):
        return "current_period", current_dir / name

    if lower_name.startswith(MODIS_EXPORT["file_name_prefix"].lower()):
        return "modis", modis_dir / name

    if lower_name.startswith(LANDSAT_EXPORT["file_name_prefix"].lower()):
        return "baseline_lst", lst_dir / name

    return None


def place_downloaded_drive_tifs(staging_dir: Path) -> dict:
    """
    Drive staging klasöründeki GeoTIFF dosyalarını Step5 data klasörlerine dağıtır.

    `drive_exports` klasörü yalnızca staging alanıdır; Step5 doğrudan burayı okumaz.
    Bu fonksiyon dosyaları `data/landsat_timeseries`, `data/landsat_qa`,
    `data/current_period` ve `data/modis` klasörlerine kopyalar.
    """
    copied = {
        "baseline_lst": [],
        "baseline_qa": [],
        "current_period": [],
        "modis": [],
        "ndvi_baseline": [],
        "ndvi_current_period": [],
        "landcover": [],
        "dem": [],
        "unmatched": [],
    }

    for source_path in sorted(staging_dir.rglob("*.tif")):
        classified = classify_downloaded_tif(source_path)
        if classified is None:
            copied["unmatched"].append(str(source_path))
            continue

        group, target_path = classified
        copy_with_overwrite_control(source_path, target_path)
        copied[group].append(str(target_path))
        log.info("Drive çıktısı Step5 klasörüne kopyalandı: %s", target_path)

    return copied


def download_drive_exports_with_geemap() -> dict:
    """Google Drive export klasörünü indirir ve GeoTIFF dosyalarını Step5 klasörlerine dağıtır."""
    if not DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT:
        log.warning("Drive otomatik indirme kapalı: DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT=False")
        return {
            "enabled": False,
            "attempted": False,
            "downloaded": False,
            "reason": "DRIVE_AUTO_DOWNLOAD_AFTER_EXPORT=False",
            "message": "Otomatik Drive indirme devre dışı.",
        }

    folder_url, folder_id = resolve_drive_folder_reference()

    if not folder_url and not folder_id:
        log.error("Drive otomatik indirme istendi ama GOOGLE_DRIVE_EXPORT_FOLDER_URL/ID boş.")
        return {
            "enabled": True,
            "attempted": False,
            "downloaded": False,
            "reason": "missing_google_drive_folder_url_or_id",
            "message": (
                "Otomatik indirme açık ama Drive klasör URL/ID verilmemiş. "
                "GOOGLE_DRIVE_EXPORT_FOLDER_URL veya GOOGLE_DRIVE_EXPORT_FOLDER_ID ayarlanmalı."
            ),
        }

    try:
        import geemap
    except ImportError as exc:
        raise ImportError("Drive klasörü indirmek için geemap gerekli.") from exc

    staging_dir = BASE_DIR / "data" / DRIVE_DOWNLOAD_STAGING_SUBDIR
    staging_dir.mkdir(parents=True, exist_ok=True)

    log.info("Drive klasörü staging alanına indiriliyor: %s", staging_dir)
    log.info("Drive referansı çözüldü: folder_id=%s, folder_url=%s", folder_id, folder_url)

    download_kwargs = {
        "output": str(staging_dir),
        "remaining_ok": True,
        "quiet": False,
    }
    if folder_id:
        download_kwargs["id"] = folder_id
    elif folder_url:
        download_kwargs["url"] = folder_url

    downloaded_files = geemap.download_folder(**download_kwargs)
    copied = place_downloaded_drive_tifs(staging_dir)

    return {
        "enabled": True,
        "attempted": True,
        "downloaded": True,
        "staging_dir": str(staging_dir),
        "downloaded_files": downloaded_files,
        "copied": copied,
        "downloaded_at": datetime.now().isoformat(),
        "message": "Drive klasörü indirildi ve GeoTIFF dosyaları Step5 klasörlerine kopyalandı.",
        "resolved_folder_id": folder_id,
        "resolved_folder_url": folder_url,
    }


# =============================================================================
# GEOTIFF VALIDATION (Part 2, manifest-driven): indirme/skip sonrası bütünlük
# =============================================================================
# Step4B artık her GeoTIFF'i körlemesine glob'lamaz. Doğrulama manifest/registry
# tabanlıdır: yalnızca AKTİF beklenen ürünler pass/fail'i etkiler. Disk üzerinde
# bulunan diğer dosyalar legacy / raw_export / ignored olarak sınıflanır ve
# raporlanır ama run'ı düşürmez.
#
# Ürün durum kategorileri (status):
#   active     — aktif pipeline'ın (Step5/Step5C/Step6) kullandığı ürün.
#   optional   — beklenen ama zorunlu olmayan (örn. land-cover).
#   legacy     — eski/kullanılmayan export.
#   raw_export — ham/ölçeklenmemiş export (örn. raw Landsat ST).
#   ignored    — "(1)" gibi duplikatlar / metadata'da referanslanmayan dosyalar.
#
# Sadece active + required ürünler kritik hata ile Step4B'yi düşürebilir.

DUPLICATE_RE = re.compile(r"\(\d+\)\.tif$", re.IGNORECASE)


def _read_step5c_ndvi_inputs() -> dict:
    """
    Step5C metadata'sından (outputs/step5c/step5c_metadata.json) NDVI input
    referanslarını YAPISAL olarak okur.

    Okunan anahtarlar:
        inputs.current_ndvi          -> aktif current NDVI dosya yolu/adı
        inputs.baseline_ndvi_dir     -> baseline NDVI klasörü
        inputs.baseline_ndvi_files   -> baseline NDVI dosya adları listesi

    Dönen sözlük:
        {
          "current_ndvi_names": {<lower filename>, ...},
          "baseline_ndvi_names": {<lower filename>, ...},
          "baseline_ndvi_dir": "<as referenced>" | None,
          "available": bool,   # metadata bulundu ve okunabildi mi
        }
    Metadata yoksa available=False döner (filename heuristic'leri devreye girer).
    """
    result = {
        "current_ndvi_names": set(),
        "baseline_ndvi_names": set(),
        "baseline_ndvi_dir": None,
        "available": False,
    }
    meta_path = BASE_DIR / "outputs" / "step5c" / "step5c_metadata.json"
    if not meta_path.exists():
        return result
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result

    inputs = data.get("inputs", {})
    if not isinstance(inputs, dict):
        return result

    result["available"] = True

    current_ndvi = inputs.get("current_ndvi")
    if isinstance(current_ndvi, str) and current_ndvi:
        result["current_ndvi_names"].add(Path(current_ndvi).name.lower())

    baseline_dir = inputs.get("baseline_ndvi_dir")
    if isinstance(baseline_dir, str) and baseline_dir:
        result["baseline_ndvi_dir"] = baseline_dir

    baseline_files = inputs.get("baseline_ndvi_files")
    if isinstance(baseline_files, list):
        for item in baseline_files:
            if isinstance(item, str) and item:
                result["baseline_ndvi_names"].add(Path(item).name.lower())

    return result


def _read_step5_active_filenames() -> set[str]:
    """
    Step5/Step5C/Step6 metadata'sından aktif olarak REFERANSLANAN dosya adlarını okur.

    Bir current-period LST veya NDVI dosyasının "active" sayılması için bu kümede
    (ya da aktif beklenen dosya adı kalıbında) bulunması yeterlidir. Metadata yoksa
    boş set döner ve yalnızca beklenen-isim kalıbı kullanılır.
    """
    names: set[str] = set()
    candidates = [
        BASE_DIR / "outputs" / "step5" / "step5_metadata.json",
        BASE_DIR / "outputs" / "step5c" / "step5c_metadata.json",
        BASE_DIR / "outputs" / "step6" / "validation_stats.json",
    ]
    for meta_path in candidates:
        if not meta_path.exists():
            continue
        try:
            text = meta_path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Dosya adlarını metinden ara (yapı değişse de isim eşleşmesi yeter).
        for token in re.findall(r"[\w()\-.]+\.tif", text):
            names.add(Path(token).name.lower())
    return names


def build_validation_manifest() -> list[dict]:
    """
    Doğrulama manifest'ini (registry) kurar.

    Körlemesine "tüm klasör active" KAYDI YAPILMAZ. add_glob() her dosyayı kategori
    ile kaydeder. Her giriş şunları taşır:
        product_name, path, product_type, label, category, active, required,
        fail_on_error, source_step, notes
    category: active | optional | raw_export | legacy | ignored
    Yalnızca active + required + fail_on_error ürünler Step4B'yi düşürebilir.
    """
    data_dir = BASE_DIR / "data"
    manifest: list[dict] = []
    active_names = _read_step5_active_filenames()
    step5c_ndvi = _read_step5c_ndvi_inputs()
    # Step5C metadata'sının açıkça referansladığı aktif NDVI dosya adları.
    active_current_ndvi_names = step5c_ndvi["current_ndvi_names"]
    active_baseline_ndvi_names = step5c_ndvi["baseline_ndvi_names"]

    def add_glob(
        directory: Path,
        pattern: str,
        product_type: str,
        label: str,
        category: str = "active",
        active: bool = True,
        required: bool = False,
        fail_on_error: bool = False,
        source_step: str = "step4/step4b",
        notes: str = "",
        classifier=None,
    ) -> None:
        """
        Bir klasördeki dosyaları kategori ile manifest'e ekler.

        classifier(path) verilirse, her dosya için
        (product_type, category, active, required, fail_on_error, notes)
        sözlüğü döndürerek per-file override yapabilir (örn. "(1)" -> ignored).
        """
        if not directory.exists():
            return
        for path in sorted(directory.glob(pattern)):
            entry = {
                "product_name": f"{label}:{path.name}",
                "path": path,
                "product_type": product_type,
                "label": label,
                "category": category,
                "active": active,
                "required": required,
                "fail_on_error": fail_on_error,
                "source_step": source_step,
                "notes": notes,
            }
            if classifier is not None:
                entry.update(classifier(path))
            manifest.append(entry)

    # -- A) DEM: active + required + fail_on_error --------------------------
    manifest.append({
        "product_name": "dem_elevation",
        "path": data_dir / "dem" / "elevation.tif",
        "product_type": "elevation",
        "label": "dem_elevation",
        "category": "active",
        "active": True,
        "required": True,
        "fail_on_error": True,
        "source_step": "step2b/step4/step4b",
        "notes": "Static auxiliary predictor (elevation).",
    })
    manifest.append({
        "product_name": "dem_slope",
        "path": data_dir / "dem" / "slope.tif",
        "product_type": "slope",
        "label": "dem_slope",
        "category": "active",
        "active": True,
        "required": True,
        "fail_on_error": True,
        "source_step": "step2b/step4/step4b",
        "notes": "Static auxiliary predictor (slope, ee.Terrain.slope).",
    })

    # -- B) Raw Landsat timeseries: raw_export (Celsius aralığı UYGULANMAZ) --
    add_glob(
        data_dir / "landsat_timeseries", "*.tif", "raw_landsat_st", "landsat_lst_raw",
        category="raw_export", active=False, required=False, fail_on_error=False,
        source_step="step4 raw export",
        notes="Raw/scaled Landsat ST export; readability/CRS/transform/finite only.",
        classifier=lambda p: (
            {"category": "ignored", "active": False, "required": False,
             "fail_on_error": False, "product_type": "ignored",
             "notes": "duplicate file"}
            if DUPLICATE_RE.search(p.name) else {}
        ),
    )

    # -- C) Current period LST: yalnız aktif beklenen ad active, gerisi legacy --
    current_prefix = f"landsat_current_period_{CURRENT_PERIOD_DAYS}days".lower()

    def current_lst_classifier(p: Path) -> dict:
        name = p.name.lower()
        if DUPLICATE_RE.search(p.name):
            return {"category": "ignored", "active": False, "required": False,
                    "fail_on_error": False, "product_type": "ignored",
                    "notes": "duplicate (1) file"}
        is_active_name = name.startswith(current_prefix) or name in active_names
        if is_active_name:
            return {"category": "active", "active": True, "required": False,
                    "fail_on_error": True, "product_type": "lst_celsius",
                    "notes": "Current-period Landsat LST used by Step5."}
        return {"category": "legacy", "active": False, "required": False,
                "fail_on_error": False, "product_type": "legacy",
                "notes": "legacy/unused current-period export."}

    add_glob(
        data_dir / "current_period", "*.tif", "lst_celsius", "current_lst",
        category="legacy", active=False, required=False, fail_on_error=False,
        source_step="step4/step4b -> step5",
        classifier=current_lst_classifier,
    )

    # -- D) NDVI: metadata-driven (Step5C inputs) > filename heuristic -------
    def _ndvi_active_entry(role: str) -> dict:
        return {
            "category": "active", "active": True, "required": True,
            "fail_on_error": True, "product_type": "ndvi", "source": "step5c_metadata",
            "notes": f"{role} NDVI referenced by Step5C metadata (active, required).",
        }

    def current_ndvi_classifier(p: Path) -> dict:
        name = p.name.lower()
        # 1) Metadata referansı her şeyi geçersiz kılar (duplicate dahil).
        if name in active_current_ndvi_names or name in active_baseline_ndvi_names:
            return _ndvi_active_entry("Current-period")
        # 2) Metadata'da yoksa: "(1)" duplikatları ignored.
        if DUPLICATE_RE.search(p.name):
            return {"category": "ignored", "active": False, "required": False,
                    "fail_on_error": False, "product_type": "ignored",
                    "notes": "duplicate (1) file not referenced by metadata"}
        # 3) Metadata yoksa filename heuristic (current_ndvi_median* / loose names).
        is_active_name = name.startswith("current_ndvi_median") or name in active_names
        if is_active_name:
            return {"category": "active", "active": True, "required": True,
                    "fail_on_error": True, "product_type": "ndvi",
                    "notes": "Current-period NDVI used by Step5C/Step6 (required)."}
        return {"category": "legacy", "active": False, "required": False,
                "fail_on_error": False, "product_type": "legacy",
                "notes": "legacy/unused NDVI export."}

    add_glob(
        data_dir / "ndvi_current_period", "*.tif", "ndvi", "current_ndvi",
        category="legacy", active=False, required=False, fail_on_error=False,
        source_step="step4/step4b -> step5c/step6",
        classifier=current_ndvi_classifier,
    )

    # Baseline NDVI: Step5C metadata'sında referanslananlar AKTİF; gerisi legacy.
    def baseline_ndvi_classifier(p: Path) -> dict:
        name = p.name.lower()
        # 1) Metadata referansı (duplicate dahil) -> aktif required.
        if name in active_baseline_ndvi_names or name in active_current_ndvi_names:
            return _ndvi_active_entry("Baseline")
        # 2) Referanssız "(1)" duplikat -> ignored.
        if DUPLICATE_RE.search(p.name):
            return {"category": "ignored", "active": False, "required": False,
                    "fail_on_error": False, "product_type": "ignored",
                    "notes": "duplicate (1) file not referenced by metadata"}
        # 3) Referanssız baseline -> legacy.
        return {"category": "legacy", "active": False, "required": False,
                "fail_on_error": False, "product_type": "raw_ndvi_or_legacy_ndvi",
                "notes": "baseline NDVI not referenced by Step5C metadata."}

    add_glob(
        data_dir / "ndvi_timeseries", "*.tif", "raw_ndvi_or_legacy_ndvi", "ndvi_baseline",
        category="legacy", active=False, required=False, fail_on_error=False,
        source_step="step4 baseline export -> step5c",
        notes="Baseline NDVI; active only if referenced by Step5C metadata.",
        classifier=baseline_ndvi_classifier,
    )

    # -- MODIS summer mean: active (Step5 context) --------------------------
    modis_prefix = MODIS_EXPORT["file_name_prefix"].lower()

    def modis_classifier(p: Path) -> dict:
        if DUPLICATE_RE.search(p.name):
            return {"category": "ignored", "active": False, "required": False,
                    "fail_on_error": False, "product_type": "ignored",
                    "notes": "duplicate (1) file"}
        if p.name.lower().startswith(modis_prefix):
            return {"category": "active", "active": True, "required": False,
                    "fail_on_error": True, "product_type": "modis_lst_celsius",
                    "notes": "MODIS 4-year summer-mean context layer."}
        return {"category": "legacy", "active": False, "required": False,
                "fail_on_error": False, "product_type": "legacy",
                "notes": "legacy MODIS export."}

    add_glob(
        data_dir / "modis", "*.tif", "modis_lst_celsius", "modis",
        category="legacy", active=False, required=False, fail_on_error=False,
        source_step="step4/step4b -> step5",
        classifier=modis_classifier,
    )

    # -- Land-cover: optional ----------------------------------------------
    add_glob(
        data_dir / "landcover", "*.tif", "land_cover", "landcover",
        category="optional", active=False, required=False, fail_on_error=False,
        source_step="step4/step4b",
        notes="Optional categorical land-cover mask.",
        classifier=lambda p: (
            {"category": "ignored", "active": False, "product_type": "ignored",
             "notes": "duplicate (1) file"}
            if DUPLICATE_RE.search(p.name) else {}
        ),
    )

    return manifest


def validate_downloaded_geotiffs(
    download_metadata: dict,
    strict: bool = False,
    mode: str = "download",
) -> dict:
    """
    Manifest-driven doğrulama (tek manifest; her dosya kategorili).

    Step4B SADECE şu durumda fail eder:
        entry.active and entry.required and entry.fail_on_error and not passed.
    Raw/legacy/ignored dosyalar yalnızca raporlanır; run'ı düşürmez. strict=True
    ise warning'ler de fail sayılır ama YALNIZCA active + fail_on_error ürünler için.
    """
    manifest = build_validation_manifest()

    downloaded_paths = set()
    copied = (download_metadata or {}).get("copied", {})
    if isinstance(copied, dict):
        for group_paths in copied.values():
            if isinstance(group_paths, list):
                downloaded_paths.update(group_paths)

    results: list[dict] = []
    for entry in manifest:
        path = Path(entry["path"])
        category = entry.get("category", "active")
        active = entry.get("active", False)
        required = entry.get("required", False)
        fail_on_error = entry.get("fail_on_error", False)
        product_type = entry["product_type"]

        common = {
            "product": entry["product_name"],
            "product_type": product_type,
            "label": entry.get("label"),
            "category": category,
            "active": active,
            "required": required,
            "fail_on_error": fail_on_error,
            "source_step": entry.get("source_step"),
            "notes": entry.get("notes"),
        }

        if not path.exists():
            res = {
                **common,
                "path": str(path),
                "source": "missing",
                # Zorunlu+fail_on_error eksikse fail; aksi halde yalnız uyarı.
                "passed": not (active and required and fail_on_error),
                "errors": (
                    ["required active product file is missing"]
                    if (active and required and fail_on_error) else []
                ),
                "warnings": (
                    [] if (active and required and fail_on_error)
                    else ["expected product not present (skipped)"]
                ),
                "stats": {},
                "exists": False,
                "readable": False,
            }
            results.append(res)
            continue

        validation = validate_geotiff_basic(
            path, expected={"expected_product_type": product_type}
        )
        validation.update(common)
        validation["source"] = (
            "downloaded" if str(path) in downloaded_paths else "already_existed"
        )
        # strict: warning -> fail, ama yalnızca active + fail_on_error ürünler için.
        if strict and active and fail_on_error and validation.get("warnings"):
            validation["passed"] = False
            validation["strict_failed"] = True
        results.append(validation)

    # Kategoriye göre ayır.
    active_results = [r for r in results if r.get("category") == "active"]
    optional_results = [r for r in results if r.get("category") == "optional"]
    raw_results = [r for r in results if r.get("category") == "raw_export"]
    legacy_results = [r for r in results if r.get("category") == "legacy"]
    ignored_results = [r for r in results if r.get("category") == "ignored"]

    summary_base = OUTPUTS_DIR / "geotiff_validation_summary.json"
    summary_path = write_validation_report_sections(results, summary_base)

    active_checked = len(active_results)
    active_failed = sum(1 for r in active_results if not r.get("passed"))
    warnings_count = sum(len(r.get("warnings", [])) for r in results)

    # Step4B loud fail = active AND required AND fail_on_error AND not passed.
    failed_required = [
        r for r in results
        if r.get("active") and r.get("required") and r.get("fail_on_error")
        and not r.get("passed")
    ]

    return {
        "enabled": True,
        "mode": mode,
        "strict": strict,
        "summary_path": str(summary_path),
        "summary_md_path": str(summary_path.with_suffix(".md")),
        "active_products_checked": active_checked,
        "active_products_failed": active_failed,
        "optional_products_checked": len(optional_results),
        "raw_exports_detected": len(raw_results),
        "legacy_products_detected": len(legacy_results),
        "ignored_products": len(ignored_results),
        "warnings_count": warnings_count,
        "failed_required_products": [r.get("product") for r in failed_required],
        "active_required_failed": len(failed_required),
        "results": results,
        "critical": failed_required,
    }


def write_validation_report_sections(
    results: list[dict],
    output_path: Path,
) -> Path:
    """
    Doğrulama raporunu kategori bölümlerine ayırarak JSON + MD yazar.

    Bölümler: Active / Optional / Raw exports / Legacy / Ignored.
    Tüm girişler tek `results` listesinden category alanına göre ayrılır.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active = [r for r in results if r.get("category") == "active"]
    optional = [r for r in results if r.get("category") == "optional"]
    raw_exports = [r for r in results if r.get("category") == "raw_export"]
    legacy = [r for r in results if r.get("category") == "legacy"]
    ignored = [r for r in results if r.get("category") == "ignored"]

    failed_required = [
        r for r in results
        if r.get("active") and r.get("required") and r.get("fail_on_error")
        and not r.get("passed")
    ]

    payload = {
        "created_at": datetime.now().isoformat(),
        "active_products_checked": len(active),
        "active_products_failed": sum(1 for r in active if not r.get("passed")),
        "active_required_failed": len(failed_required),
        "optional_products_checked": len(optional),
        "raw_exports_detected": len(raw_exports),
        "legacy_products_detected": len(legacy),
        "ignored_products": len(ignored),
        "warnings_count": sum(len(r.get("warnings", [])) for r in results),
        "failed_required_products": [r.get("product") for r in failed_required],
        "active_products": active,
        "optional_products": optional,
        "raw_exports": raw_exports,
        "legacy_products": legacy,
        "ignored_products_list": ignored,
    }

    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_path = output_path.with_suffix(".md")
    md_path.write_text(_render_sectioned_markdown(payload), encoding="utf-8")
    return json_path


def _render_sectioned_markdown(payload: dict) -> str:
    def product_table(rows: list[dict]) -> list[str]:
        lines = [
            "| Product | Type | Category | Active | Req | FailOnErr | Status | Errors | Warnings |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
        for r in rows:
            status_txt = "PASS" if r.get("passed") else "FAIL"
            lines.append(
                "| {name} | {ptype} | {cat} | {act} | {req} | {foe} | {status} "
                "| {nerr} | {nwarn} |".format(
                    name=r.get("product", "?"),
                    ptype=r.get("product_type", "?"),
                    cat=r.get("category", "?"),
                    act=r.get("active"),
                    req=r.get("required"),
                    foe=r.get("fail_on_error"),
                    status=status_txt,
                    nerr=len(r.get("errors", [])),
                    nwarn=len(r.get("warnings", [])),
                )
            )
        return lines

    def section(title: str, note: str, rows: list[dict]) -> list[str]:
        out = ["", f"## {title}", "", note, ""]
        out.extend(product_table(rows) if rows else ["_None._"])
        return out

    lines = [
        "# Step4B GeoTIFF Validation Summary",
        "",
        f"Created at: `{payload['created_at']}`",
        f"Active products checked: `{payload['active_products_checked']}`",
        f"Active products failed: `{payload['active_products_failed']}`",
        f"Active required failed: `{payload['active_required_failed']}`",
        f"Optional products checked: `{payload['optional_products_checked']}`",
        f"Raw exports detected: `{payload['raw_exports_detected']}`",
        f"Legacy products detected: `{payload['legacy_products_detected']}`",
        f"Ignored products: `{payload['ignored_products']}`",
        f"Warnings: `{payload['warnings_count']}`",
        f"Failed required products: `{payload['failed_required_products']}`",
    ]
    lines.extend(section(
        "Active products checked", "These affect pass/fail.",
        payload["active_products"]))
    lines.extend(section(
        "Optional products checked",
        "Warnings only unless unreadable and marked required.",
        payload["optional_products"]))
    lines.extend(section(
        "Raw exports detected",
        "Raw/scaled exports (e.g. raw Landsat ST); Celsius range not applied; do not fail the run.",
        payload["raw_exports"]))
    lines.extend(section(
        "Legacy products detected",
        "Old/unused exports; listed for transparency; do not fail the run.",
        payload["legacy_products"]))
    lines.extend(section(
        "Ignored products",
        "Duplicate '(1)' files or files not referenced by active metadata.",
        payload["ignored_products_list"]))

    # Aktif ürün ayrıntıları (stats + errors/warnings)
    lines.extend(["", "## Active product details", ""])
    for r in payload["active_products"]:
        lines.append(f"### {r.get('product')}")
        lines.append(f"- Path: `{r.get('path')}`")
        lines.append(f"- Product type: `{r.get('product_type')}`")
        lines.append(f"- Source: `{r.get('source')}` | Passed: `{r.get('passed')}`")
        if r.get("errors"):
            lines.append("- Errors:")
            for e in r["errors"]:
                lines.append(f"  - {e}")
        if r.get("warnings"):
            lines.append("- Warnings:")
            for w in r["warnings"]:
                lines.append(f"  - {w}")
        stats = r.get("stats") or {}
        if stats:
            lines.append(
                "- Stats: "
                f"finite%=`{stats.get('finite_percent')}`, "
                f"min=`{stats.get('min')}`, max=`{stats.get('max')}`"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


class GeoTiffValidationError(RuntimeError):
    """Kritik GeoTIFF doğrulama hatası (Step4B loud fail)."""


def run_validation_and_maybe_fail(
    download_metadata: dict,
    strict: bool = False,
    mode: str = "download",
) -> dict:
    """Doğrulamayı çalıştırır; aktif+required ürün kritik hata verirse loud fail eder."""
    log.info("GeoTIFF doğrulaması başlatılıyor (strict=%s, mode=%s).", strict, mode)
    validation = validate_downloaded_geotiffs(download_metadata, strict=strict, mode=mode)

    log.info(
        "GeoTIFF doğrulaması: active_checked=%d active_failed=%d "
        "raw=%d legacy=%d ignored=%d warnings=%d",
        validation["active_products_checked"],
        validation["active_products_failed"],
        validation["raw_exports_detected"],
        validation["legacy_products_detected"],
        validation["ignored_products"],
        validation["warnings_count"],
    )
    for r in validation["results"]:
        if r.get("category") in ("active", "optional"):
            for w in r.get("warnings", []):
                log.warning("[%s] %s", r.get("product"), w)
            for e in r.get("errors", []):
                log.error("[%s] %s", r.get("product"), e)
        else:
            # raw/legacy/ignored yalnızca bilgilendirme; run'ı düşürmez.
            log.info(
                "[%s] %s (%s)",
                r.get("category"),
                Path(r.get("path", "?")).name,
                r.get("notes", ""),
            )

    if validation["critical"]:
        names = ", ".join(r.get("product", "?") for r in validation["critical"])
        raise GeoTiffValidationError(
            f"GeoTIFF validation failed for active required product(s): {names}. "
            f"See {validation['summary_path']} (and .md) for details. "
            "Step4B does not repair rasters; fix the export/download and re-run. "
            "Legacy/raw/duplicate files do not cause this failure."
        )
    return validation


def save_metadata(metadata: dict, filename: str = "step4b_metadata.json") -> Path:
    """Step4b download/placement metadata bilgisini JSON olarak kaydeder."""
    output_path = OUTPUTS_DIR / filename
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Step4b metadata kaydedildi: %s", output_path)
    return output_path


def main(
    skip_validation: bool = False,
    validation_only: bool = False,
    strict_validation: bool = False,
) -> dict:
    """
    Drive export klasörünü indirir ve GeoTIFF dosyalarını Step5 klasörlerine dağıtır.

    Doğrulama bayrakları:
        validation_only: indirme yapma, yalnızca mevcut dosyaları doğrula.
        skip_validation: indir ama doğrulama yapma.
        strict_validation: warning'leri de failure say.
    """
    log.info("=" * 60)
    log.info("STEP 4B BAŞLIYOR (Drive download + local placement)")
    log.info("=" * 60)

    if validation_only:
        log.info("validation-only modu: indirme atlanıyor, mevcut dosyalar doğrulanıyor.")
        download_metadata = {
            "enabled": False,
            "attempted": False,
            "downloaded": False,
            "reason": "validation_only_mode",
            "message": "validation-only: indirme yapılmadı.",
        }
    else:
        download_metadata = download_drive_exports_with_geemap()

    metadata = {
        "step": "step4b_download_drive_exports",
        "pipeline_stage": "drive_download_and_local_placement",
        "created_at": datetime.now().isoformat(),
        "log_file": str(log_file),
        "drive_download": download_metadata,
        "status": (
            "download_completed"
            if download_metadata.get("downloaded")
            else "download_not_completed"
        ),
    }

    # GeoTIFF doğrulaması (Part 2, manifest-driven)
    if skip_validation and not validation_only:
        log.info("skip-validation: GeoTIFF doğrulaması atlandı.")
        metadata["geotiff_validation"] = {"enabled": False, "reason": "skip_validation"}
    else:
        validation = run_validation_and_maybe_fail(
            download_metadata,
            strict=strict_validation,
            mode="validation_only" if validation_only else "download",
        )
        metadata["geotiff_validation"] = {
            "enabled": True,
            "mode": validation["mode"],
            "strict": validation["strict"],
            "summary_path": validation["summary_path"],
            "summary_md_path": validation["summary_md_path"],
            "active_products_checked": validation["active_products_checked"],
            "active_products_failed": validation["active_products_failed"],
            "active_required_failed": validation["active_required_failed"],
            "optional_products_checked": validation["optional_products_checked"],
            "raw_exports_detected": validation["raw_exports_detected"],
            "legacy_products_detected": validation["legacy_products_detected"],
            "ignored_products": validation["ignored_products"],
            "warnings_count": validation["warnings_count"],
            "failed_required_products": validation["failed_required_products"],
        }

    metadata_path = save_metadata(metadata)

    if validation_only:
        print("\nSTEP 4B validation-only tamamlandı.")
        gv = metadata.get("geotiff_validation", {})
        print(
            f"Aktif ürün: {gv.get('active_products_checked')}, "
            f"aktif failed: {gv.get('active_products_failed')}, "
            f"active_required_failed: {gv.get('active_required_failed')}, "
            f"raw: {gv.get('raw_exports_detected')}, "
            f"legacy: {gv.get('legacy_products_detected')}, "
            f"ignored: {gv.get('ignored_products')}"
        )
        print(f"Rapor: {gv.get('summary_md_path')}")
        print(f"Metadata: {metadata_path}\n")
    elif download_metadata.get("downloaded"):
        log.info("Drive çıktıları indirildi ve Step5 data klasörlerine yerleştirildi.")
        print("\nSTEP 4B tamamlandı. Step5 çalıştırılabilir:")
        print("python step5_preprocess_timeseries.py\n")
    else:
        message = download_metadata.get("message") or download_metadata.get("reason")
        log.warning("Drive indirme tamamlanmadı: %s", message)
        print("\nSTEP 4B tamamlandı ancak indirme yapılmadı.")
        if message:
            print(f"Not: {message}")
        print(f"Metadata: {metadata_path}\n")

    log.info("=" * 60)
    log.info("STEP 4B TAMAMLANDI")
    return metadata


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step4B: Drive export indirme + GeoTIFF doğrulama."
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="İndir ama GeoTIFF doğrulaması yapma.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="İndirme yapma; yalnızca mevcut dosyaları doğrula.",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Warning'leri de failure say.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        skip_validation=args.skip_validation,
        validation_only=args.validation_only,
        strict_validation=args.strict_validation,
    )