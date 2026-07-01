"""
validation_burned_area.py

Yanmış alan / aktif yangın doğrulama SKELETON'u (Phase 2 için taslak).

Bu modül henüz AĞIR validation yapmaz. Amacı, TVDI/dryness katmanını yangın
kayıtlarıyla çakıştırma (ROC/AUC) adımına temel oluşturacak GEE collection
helper'larını tek yerde tanımlamaktır.

Kapsanan kaynaklar:
    - MCD64A1   : MODIS yanmış alan (500 m, aylık)        -> BurnDate bandı
    - FireCCI51 : ESA CCI FireCCI 5.1 yanmış alan (250 m)  -> BurnDate bandı
    - FIRMS     : aktif yangın (MODIS/MCD14ML türevi, günlük) -> T21 bandı

Phase 2'de yapılacaklar (henüz DEĞİL):
    - Bir sezonun TVDI/dryness katmanını aynı sezonun yanmış alanıyla çakıştır.
    - Yanan vs yanmayan piksellerde TVDI ayrışmasını ölç (ROC/AUC).
"""

from __future__ import annotations

from datetime import datetime

import ee

from core.config import (
    ENABLE_BURNED_AREA_VALIDATION,
    FIRECCI51_BURNDATE_BAND,
    FIRECCI51_COLLECTION,
    FIRMS_COLLECTION,
    FIRMS_FIRE_BAND,
    MCD64A1_BURNDATE_BAND,
    MCD64A1_COLLECTION,
)
from core.io_utils import setup_logger


# Logger'ı import sırasında değil, ilk kullanımda kur (lazy). Modül-seviyesi
# yan etkiler (log dosyası oluşturma vb.) import zincirini kırabilir; bu da
# step6'da "GEE importları başarısız" gibi yanıltıcı hatalara yol açar.
_log = None


def _get_log():
    global _log
    if _log is None:
        _log, _ = setup_logger("validation_burned_area")
    return _log


def get_mcd64a1_burned_area(
    region: ee.Geometry,
    start: str,
    end: str,
) -> ee.Image:
    """
    MCD64A1 (500 m) yanmış alan maskesi taslağı.

    BurnDate > 0 olan pikseller yanmış kabul edilir. Verilen tarih aralığındaki
    aylık ürünler birleştirilip tek bir "yanmış mı" maskesine indirgenir.
    """
    collection = (
        ee.ImageCollection(MCD64A1_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .select(MCD64A1_BURNDATE_BAND)
    )
    burned = collection.max().gt(0).rename("MCD64A1_burned").clip(region)
    return burned


def get_firecci51_burned_area(
    region: ee.Geometry,
    start: str,
    end: str,
) -> ee.Image:
    """
    FireCCI51 (250 m) yanmış alan maskesi taslağı.

    BurnDate > 0 olan pikseller yanmış kabul edilir.

    NOT: Bu fonksiyon collection'ın boş olabileceğini varsaymaz. Boş/bandsiz
    image riskine karşı güvenli kullanım için get_firecci51_burned_area_safe()
    tercih edilmelidir.
    """
    collection = (
        ee.ImageCollection(FIRECCI51_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .select(FIRECCI51_BURNDATE_BAND)
    )
    burned = collection.max().gt(0).rename("FireCCI51_burned").clip(region)
    return burned


def get_firecci51_burned_area_safe(
    region: ee.Geometry,
    start: str,
    end: str,
) -> tuple[ee.Image | None, str]:
    """
    FireCCI51 yanmış alan maskesini GÜVENLİ şekilde kurar.

    Boş collection veya bandsiz image durumunda .gt(0) çağrılmaz; (None, sebep)
    döndürülür. Başarılıysa (image, "ok") döndürülür.

    Kontrol sırası:
        1. collection.size() == 0  -> skip
        2. mevcut bandNames loglanır
        3. beklenen burn-date bandı yoksa -> skip
        4. mosaic image'inin bandNames().size() == 0 -> skip
    """
    collection = (
        ee.ImageCollection(FIRECCI51_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
    )

    size = collection.size().getInfo()
    if size == 0:
        return (
            None,
            "FireCCI51 returned no images for selected AOI/season; "
            "skipping FireCCI51.",
        )

    # Mevcut bant adlarını logla (collection'ın ilk image'inden).
    try:
        available_bands = (
            ee.Image(collection.first()).bandNames().getInfo()
        )
    except Exception:  # noqa: BLE001
        available_bands = []
    _get_log().info("FireCCI51 mevcut bantlar: %s", available_bands)

    if FIRECCI51_BURNDATE_BAND not in available_bands:
        return (
            None,
            f"FireCCI51 expected burn-date band '{FIRECCI51_BURNDATE_BAND}' "
            f"not found (available: {available_bands}); skipping FireCCI51.",
        )

    selected = collection.select(FIRECCI51_BURNDATE_BAND)
    mosaic = selected.max()

    band_count = mosaic.bandNames().size().getInfo()
    if band_count == 0:
        return (
            None,
            "FireCCI51 image has no bands after filtering; skipping FireCCI51.",
        )

    burned = mosaic.gt(0).rename("FireCCI51_burned").clip(region)
    return burned, "ok"


def get_mcd64a1_burned_area_safe(
    region: ee.Geometry,
    start: str,
    end: str,
) -> tuple[ee.Image | None, str]:
    """
    MCD64A1 yanmış alan maskesini GÜVENLİ şekilde kurar (aynı boş/bandsiz koruması).
    """
    collection = (
        ee.ImageCollection(MCD64A1_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
    )

    size = collection.size().getInfo()
    if size == 0:
        return (
            None,
            "MCD64A1 returned no images for selected AOI/season; "
            "skipping MCD64A1.",
        )

    try:
        available_bands = ee.Image(collection.first()).bandNames().getInfo()
    except Exception:  # noqa: BLE001
        available_bands = []
    _get_log().info("MCD64A1 mevcut bantlar: %s", available_bands)

    if MCD64A1_BURNDATE_BAND not in available_bands:
        return (
            None,
            f"MCD64A1 expected burn-date band '{MCD64A1_BURNDATE_BAND}' not "
            f"found (available: {available_bands}); skipping MCD64A1.",
        )

    selected = collection.select(MCD64A1_BURNDATE_BAND)
    mosaic = selected.max()

    band_count = mosaic.bandNames().size().getInfo()
    if band_count == 0:
        return (
            None,
            "MCD64A1 image has no bands after filtering; skipping MCD64A1.",
        )

    burned = mosaic.gt(0).rename("MCD64A1_burned").clip(region)
    return burned, "ok"


def get_firms_active_fire(
    region: ee.Geometry,
    start: str,
    end: str,
) -> ee.Image:
    """
    FIRMS (MODIS) aktif yangın yoğunluğu taslağı.

    T21 (parlaklık sıcaklığı) bandının maksimumu alınır; aktif yangın sinyalinin
    mekansal izini verir. MODIS tabanlıdır; VIIRS için get_firms_viirs_active_fire.
    """
    collection = (
        ee.ImageCollection(FIRMS_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .select(FIRMS_FIRE_BAND)
    )
    active_fire = collection.max().rename("FIRMS_active_fire").clip(region)
    return active_fire


# MODIS FIRMS için açık isim (cross-check kodu netliği için).
def get_firms_modis_active_fire(
    region: ee.Geometry,
    start: str,
    end: str,
) -> ee.Image:
    """FIRMS MODIS (T21) aktif yangın görüntüsü. get_firms_active_fire ile aynı."""
    return get_firms_active_fire(region, start, end)


def get_firms_viirs_active_fire_safe(
    region: ee.Geometry,
    start: str,
    end: str,
) -> tuple["ee.Image | None", str]:
    """
    FIRMS VIIRS aktif yangın görüntüsünü güvenli şekilde döndürür.

    NOAA-20 ve S-NPP VIIRS koleksiyonları sırayla denenir; ilk bant içereni
    kullanılır. Hiçbiri kullanılabilir değilse (None, reason) döner. Boş/bandsiz
    koleksiyonlarda .max() çağrısı band üretmez; bu durumda skip edilir.
    """
    from core.config import FIRMS_VIIRS_COLLECTIONS, FIRMS_VIIRS_FIRE_BAND

    last_reason = "FIRMS VIIRS: no collection available."
    for coll_id in FIRMS_VIIRS_COLLECTIONS:
        try:
            collection = (
                ee.ImageCollection(coll_id)
                .filterBounds(region)
                .filterDate(start, end)
            )
            size = collection.size().getInfo()
            if not size:
                last_reason = f"FIRMS VIIRS: empty collection {coll_id} for window."
                continue
            band_names = collection.first().bandNames().getInfo()
            if FIRMS_VIIRS_FIRE_BAND not in band_names:
                last_reason = (
                    f"FIRMS VIIRS: band {FIRMS_VIIRS_FIRE_BAND} not in {coll_id} "
                    f"(bands={band_names})."
                )
                continue
            active_fire = (
                collection.select(FIRMS_VIIRS_FIRE_BAND)
                .max()
                .rename("FIRMS_VIIRS_active_fire")
                .clip(region)
            )
            _get_log().info("FIRMS VIIRS kaynağı seçildi: %s", coll_id)
            return active_fire, f"FIRMS VIIRS source: {coll_id}"
        except Exception as exc:  # noqa: BLE001
            last_reason = f"FIRMS VIIRS error on {coll_id}: {exc}"
            continue
    return None, last_reason


def build_validation_inputs(
    region: ee.Geometry,
    start: str,
    end: str,
) -> dict:
    """
    Üç kaynağı tek dict'te toplayan üst seviye taslak.

    NOT: Bu fonksiyon yalnız ee.Image nesneleri kurar; export/indirme/ROC-AUC
    yapmaz. ENABLE_BURNED_AREA_VALIDATION False ise erken döner.
    """
    if not ENABLE_BURNED_AREA_VALIDATION:
        _get_log().info(
            "Burned-area validation kapalı (ENABLE_BURNED_AREA_VALIDATION=False). "
            "Skeleton hazır; Phase 2'de açılacak."
        )
        return {
            "enabled": False,
            "created_at": datetime.now().isoformat(),
            "note": "validation skeleton only; no images built",
        }

    _get_log().info("Burned-area validation girdileri kuruluyor: %s -> %s", start, end)
    return {
        "enabled": True,
        "created_at": datetime.now().isoformat(),
        "mcd64a1_burned": get_mcd64a1_burned_area(region, start, end),
        "firecci51_burned": get_firecci51_burned_area(region, start, end),
        "firms_active_fire": get_firms_active_fire(region, start, end),
        "sources": {
            "mcd64a1": {"collection": MCD64A1_COLLECTION, "resolution_m": 500},
            "firecci51": {"collection": FIRECCI51_COLLECTION, "resolution_m": 250},
            "firms": {"collection": FIRMS_COLLECTION, "resolution_m": 1000},
        },
    }


if __name__ == "__main__":
    _get_log().info("validation_burned_area skeleton modülü. Phase 2'de doldurulacak.")
    _get_log().info("Tanımlı kaynaklar: MCD64A1, FireCCI51, FIRMS.")