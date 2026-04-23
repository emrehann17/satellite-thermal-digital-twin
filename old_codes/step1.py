"""
Yapılanlar:
- GEE bağlantısı ve proje başlatma
- Çalışma bölgesi tanımı: Adana / Mersin / Hatay (Doğu Akdeniz)
- Kozan alt kümesi ayrı geometri olarak
- MODIS MOD11A1 koleksiyonuna bağlanma ve meta veri kontrolü
- İlk görüntü LST bandını indirme (zip → tif)
- Sonuçları logs/ klasörüne kaydetme
"""

import ee
import urllib.request
import zipfile
import json
import logging
from datetime import datetime
from pathlib import Path

# ── Klasör yapısı ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "step1"
LOGS_DIR   = BASE_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True) #if these folders don't exist, create them
LOGS_DIR.mkdir(parents=True, exist_ok=True) #else do nothing

# ── Loglama ────────────────────────────────────────────────────────────────────
log_file = LOGS_DIR / f"step1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()          # terminale de yaz
    ]
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. GEE BAŞLATMA
# ══════════════════════════════════════════════════════════════════════════════
def init_gee(project: str = "b7-thermal-digital-twin") -> None:
    log.info("GEE initializing...")
    ee.Initialize(project=project)

    # Basit doğrulama: sabit bir sayıyı GEE üzerinden hesapla
    test_val = ee.Number(42).getInfo()
    assert test_val == 42, "GEE connection failed: expected 42, got {test_val}"
    log.info(f"GEE connection OK  →  project: {project}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ÇALIŞMA BÖLGELERİ
# ══════════════════════════════════════════════════════════════════════════════
def build_regions() -> dict:
    # Batı: Mersin batısı 33.8  Doğu: Hatay 36.7 Güney: sahil 36.0 Kuzey: Toros dağları 38.0
    dogu_akdeniz = ee.Geometry.BBox(33.8, 36.0, 36.7, 38.0)

    # Kozan ilçe merkezi + 50 km tampon (alt küme)
    kozan_merkez = ee.Geometry.Point([35.81, 37.45])
    kozan_aoi    = kozan_merkez.buffer(50_000).bounds()

    regions = {
        "dogu_akdeniz": dogu_akdeniz,
        "kozan_aoi"   : kozan_aoi,
    }

    # Geometri bilgilerini logla
    for name, geom in regions.items():
        info = geom.getInfo()
        log.info(f"Bölge '{name}'  →  tip: {info['type']}")

    return regions


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODIS MOD11A1 KOLEKSİYON BİLGİSİ
# ══════════════════════════════════════════════════════════════════════════════
def inspect_modis(region: ee.Geometry,
                  start: str = "2023-08-01",
                  end:   str = "2023-08-07") -> ee.Image:
    """
    MODIS MOD11A1 koleksiyonunu filtreler, görüntü sayısını loglar,
    ilk görüntünün meta verisini kaydeder ve Celsius'a çevrilmiş
    LST bandını döner.
    """
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterBounds(region)
        .filterDate(start, end)
    )

    count = collection.size().getInfo()
    log.info(f"MODIS görüntü sayısı ({start} → {end}): {count}")

    if count == 0:
        raise ValueError("Belirtilen tarih/bölge için MODIS görüntüsü bulunamadı!")

    first_img = collection.first()

    # Meta veri
    props = first_img.toDictionary(
        ["system:time_start", "system:index", "DAYNIGHTFLAG", "CLOUDCOVER"]
    ).getInfo()
    log.info(f"İlk görüntü meta verisi: {json.dumps(props, indent=2, default=str)}")

    # Meta veriyi diske yaz
    meta_path = DATA_DIR / "modis_first_image_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(props, f, indent=2, default=str)
    log.info(f"Meta veri kaydedildi → {meta_path}")

    # celciusa çevirme
    lst_celsius = (
        first_img
        .select("LST_Day_1km")
        .multiply(0.02)
        .subtract(273.15)
        .rename("LST_Celsius")
    )

    return lst_celsius


# ══════════════════════════════════════════════════════════════════════════════
# 4. GEOTİFF İNDİRME (zip → tif)
# ══════════════════════════════════════════════════════════════════════════════
def download_lst_tif(lst_image: ee.Image,
                     region: ee.Geometry,
                     out_name: str = "modis_lst_sample",
                     scale: int = 1000) -> Path:
    """
    GEE'den GeoTIFF indirir, zip'i açar ve .tif dosyasının yolunu döner.
    """
    log.info(f"Download URL oluşturuluyor  (ölçek: {scale}m)...")
    url = lst_image.getDownloadURL({
        "scale" : scale,
        "crs"   : "EPSG:4326",
        "region": region,
        "format": "GEO_TIFF",
        "name"  : out_name,
    })

    zip_path = DATA_DIR / f"{out_name}.zip"
    log.info(f"İndiriliyor → {zip_path}")
    urllib.request.urlretrieve(url, zip_path)
    log.info(f"İndirme tamamlandı  ({zip_path.stat().st_size / 1024:.1f} KB)")

    # Zip mi, direkt TIF mi? kontrol et
    tif_path = DATA_DIR / f"{out_name}.tif"

    if zipfile.is_zipfile(zip_path):
        log.info("Zip formatı algılandı, açılıyor...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
            extracted = zf.namelist()
            log.info(f"Zip içeriği: {extracted}")
            tif_files = [f for f in extracted if f.endswith(".tif")]
            if tif_files:
                tif_path = DATA_DIR / tif_files[0]
        zip_path.unlink()
    else:
        # GEE direkt TIF döndürdü, sadece yeniden adlandır
        log.info("Direkt GeoTIFF formatı algılandı (zip değil), yeniden adlandırılıyor...")
        zip_path.rename(tif_path)

    log.info(f"GeoTIFF hazır → {tif_path}")
    return tif_path


# ══════════════════════════════════════════════════════════════════════════════
# 5. ÖZET RAPOR
# ══════════════════════════════════════════════════════════════════════════════
def save_summary(tif_path: Path) -> None:
    """Adım 1 özet raporunu JSON olarak kaydeder."""
    summary = {
        "adim"       : 1,
        "tarih"      : datetime.now().isoformat(),
        "tif_dosyasi": str(tif_path),
        "tif_boyutu_kb": round(tif_path.stat().st_size / 1024, 2) if tif_path else None,
        "durum"      : "TAMAMLANDI",
    }
    out = DATA_DIR / "step1_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"Özet rapor kaydedildi → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("ADIM 1 BAŞLIYOR: GEE Kurulum & Temel Sorgular")
    log.info("=" * 60)

    # 1. GEE bağlantısı
    init_gee()

    # 2. Bölgeler
    regions = build_regions()

    # 3. MODIS koleksiyonu incele (Kozan AOI ile örnek indirme)
    lst = inspect_modis(
        region=regions["kozan_aoi"],
        start="2023-08-01",
        end="2023-08-07"
    )

    # 4. GeoTIFF indir
    tif_path = download_lst_tif(
        lst_image=lst,
        region=regions["kozan_aoi"],
        out_name="modis_lst_kozan_20230801",
        scale=1000
    )

    # 5. Özet kaydet
    save_summary(tif_path)

    log.info("=" * 60)
    log.info("ADIM 1 TAMAMLANDI ✓")
    log.info(f"Çıktılar → {DATA_DIR}")
    log.info(f"Log      → {log_file}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()