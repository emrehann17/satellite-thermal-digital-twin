"""
main.py

Yapılanlar:
    - Step1'den Step4'e kadar olan akışı sırasıyla çalıştırır
    - GEE tarafındaki online pipeline'ı tek yerden yönetir
    - Step4 sonrası manuel indirme gerektiğini kullanıcıya bildirir

NOT:
    Step5 bu dosya tarafından çalıştırılmaz.
    Step4 sonrası export edilen GeoTIFF dosyaları manuel olarak indirilip
    ilgili klasörlere yerleştirildikten sonra Step5 ayrı çalıştırılmalıdır.
"""

from datetime import datetime
from pathlib import Path
import traceback

from core.io_utils import setup_logger

import step1_fetch_modis
import step2_modis_5year_mean
import step3_landsat_lst
import step4_export_geotiff


BASE_DIR = Path(__file__).resolve().parent
log, log_file = setup_logger("main")


def run_step(step_name: str, step_func) -> None:
    log.info("=" * 70)
    log.info(f"{step_name} BAŞLATILIYOR")
    log.info("=" * 70)

    step_func()

    log.info("=" * 70)
    log.info(f"{step_name} TAMAMLANDI")
    log.info("=" * 70)


def main() -> None:
    start_time = datetime.now()

    log.info("#" * 80)
    log.info("ONLINE PIPELINE BAŞLIYOR (STEP1 -> STEP4)")
    log.info(f"Başlangıç zamanı: {start_time.isoformat()}")
    log.info("#" * 80)

    try:
        run_step("STEP 1", step1_fetch_modis.main)
        run_step("STEP 2", step2_modis_5year_mean.main)
        run_step("STEP 3", step3_landsat_lst.main)
        run_step("STEP 4", step4_export_geotiff.main)

        end_time = datetime.now()

        log.info("#" * 80)
        log.info("ONLINE PIPELINE TAMAMLANDI")
        log.info(f"Bitiş zamanı: {end_time.isoformat()}")
        log.info(f"Log dosyası: {log_file}")
        log.info("#" * 80)

        log.info("Step4 sonrası export edilen GeoTIFF dosyaları Google Drive'a gönderilmiştir.")
        log.info("Bir sonraki adım: GeoTIFF dosyalarını manuel olarak indirip ilgili veri klasörlerine yerleştirin.")
        log.info("Daha sonra Step5'i ayrı olarak çalıştırın:")
        log.info("python step5_preprocess_timeseries.py")

        print("\n" + "=" * 80)
        print("STEP1 -> STEP4 tamamlandı.")
        print("Step4 export görevleri başlatıldı / tamamlandı.")
        print("Şimdi GeoTIFF dosyalarını Drive'dan indirip uygun klasörlere yerleştirmelisin.")
        print("Sonrasında Step5'i manuel çalıştır:")
        print("python step5_preprocess_timeseries.py")
        print("=" * 80 + "\n")

    except Exception as e:
        log.error("Pipeline çalışırken hata oluştu.")
        log.error(str(e))
        log.error(traceback.format_exc())

        print("\n" + "=" * 80)
        print("HATA OLUŞTU")
        print(str(e))
        print("Detaylar log dosyasında mevcut:")
        print(log_file)
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()