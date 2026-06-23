"""
run_prefire_experiment.py

Pre-fire burned-area association deneyini uçtan uca çalıştıran yardımcı script.

NE YAPAR:
    1. VALIDATION_MODE == "pre_fire" olduğunu doğrular (core/config.py'de elle ayarlanır).
    2. Config'in current period'u predictor window'a hizaladığını gösterir
       (CURRENT_PERIOD_END_DATE / CURRENT_PERIOD_DAYS predictor window'dan türetilir).
    3. Step3 -> Step4 -> Step4B -> Step5 -> Step5C -> Step6 sırasını çalıştırır.
       (Bu adımlar GEE erişimi ve auth gerektirir.)

DENEY TANIMI:
    Predictor window: 2023-06-01 -> 2023-07-31  (yangın ÖNCESi kuruluk durumu)
    Label window:     2023-08-01 -> 2023-10-31  (sonraki yanmış alanlar)
    AOI: kozan_aoi
    Label kaynağı: MCD64A1 (FireCCI51 2023 kapsamı dışı -> skip)

ÖNEMLİ:
    - core/config.py pre_fire modunda current period'u predictor window'dan
      türetir; böylece Step3/4/5/5C çıktıları yangın ÖNCESi dönemi temsil eder
      ve label dönemine sızmaz. Config yüklemesi çakışma varsa ValueError verir.
    - RF/XGBoost veya MODIS downscaling YAPILMAZ.
    - Sonuç "validated fire-risk model" DEĞİL; ilk pre-fire association testidir.
    - No temporal interpolation (Step5/Step5C zaten interpolasyon yapmaz).

KULLANIM:
    1. core/config.py:  VALIDATION_MODE = "pre_fire"
    2. (isteğe bağlı) same-season çıktılarını korumak için outputs/step5 ve
       outputs/step5c klasörlerini yedekleyin; bu koşu onları predictor window
       çıktılarıyla yeniden üretir.
    3. python scripts/run_prefire_experiment.py
"""

from pathlib import Path as _Path
import sys as _sys

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import sys

from core import config
from core.io_utils import setup_logger

log, _ = setup_logger("run_prefire_experiment")


def main() -> int:
    log.info("=" * 60)
    log.info("PRE-FIRE VALIDATION DENEYİ")
    log.info("=" * 60)

    if config.VALIDATION_MODE != "pre_fire":
        log.error(
            "VALIDATION_MODE şu an '%s'. Bu deney için core/config.py içinde "
            'VALIDATION_MODE = "pre_fire" yapın ve tekrar çalıştırın.',
            config.VALIDATION_MODE,
        )
        return 1

    # Config'in türettiği predictor window'u göster
    log.info(
        "Predictor (current) window türetildi: end=%s, days=%s",
        config.CURRENT_PERIOD_END_DATE,
        config.CURRENT_PERIOD_DAYS,
    )
    log.info(
        "Predictor window: %s -> %s",
        config.VALIDATION_PREFIRE_PREDICTOR_START,
        config.VALIDATION_PREFIRE_PREDICTOR_END,
    )
    log.info(
        "Label window: %s -> %s",
        config.VALIDATION_PREFIRE_LABEL_START,
        config.VALIDATION_PREFIRE_LABEL_END,
    )
    log.info("AOI: %s", config.REGION_NAME)
    log.info("Label source: MCD64A1 (FireCCI51 2023 kapsamı dışı -> skip)")

    # Adım import'ları burada yapılır ki config türetmesi önce çalışsın.
    import src.step3_landsat_lst as step3_landsat_lst
    import src.step4_export_geotiff as step4_export_geotiff
    import src.step4b_download_drive_export as step4b_download_drive_export
    import src.step5_preprocess_timeseries as step5_preprocess_timeseries
    import src.step5c_tvdi as step5c_tvdi
    import src.step6_validate_fire_relation as step6_validate_fire_relation

    step3_result = None

    steps = [
        ("STEP 3 (Landsat LST + NDVI, predictor window)", "step3"),
        ("STEP 4 (GeoTIFF export)", "step4"),
        ("STEP 4B (Drive indirme)", "step4b"),
        ("STEP 5 (LST anomaly preprocess)", "step5"),
        ("STEP 5C (TVDI)", "step5c"),
        ("STEP 6 (pre-fire burned-area association)", "step6"),
    ]

    for name, key in steps:
        log.info("-" * 50)
        log.info("BAŞLIYOR: %s", name)
        try:
            if key == "step3":
                step3_result = step3_landsat_lst.main()
            elif key == "step4":
                step4_export_geotiff.main(step3_result=step3_result)
            elif key == "step4b":
                step4b_download_drive_export.main()
            elif key == "step5":
                step5_preprocess_timeseries.main()
            elif key == "step5c":
                step5c_tvdi.main()
            elif key == "step6":
                step6_validate_fire_relation.main()
            log.info("TAMAMLANDI: %s", name)
        except Exception as exc:  # noqa: BLE001
            log.error("HATA (%s): %s", name, exc)
            log.error(
                "Deney durduruldu. Hata mesajını kontrol edin (GEE auth, veri "
                "kapsamı vb.)."
            )
            return 1

    log.info("=" * 60)
    log.info("PRE-FIRE DENEYİ TAMAMLANDI")
    log.info("Çıktılar: outputs/validation/validation_summary.md")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())