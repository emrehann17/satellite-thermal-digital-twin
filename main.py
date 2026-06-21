"""
main.py

Yapılanlar:
    - Step1'den Step5'e kadar olan akışı sırasıyla çalıştırır
    - GEE tarafındaki online pipeline'ı tek yerden yönetir
"""

from datetime import datetime
from pathlib import Path
import traceback

from core.io_utils import setup_logger

import step1_fetch_modis
import step2_modis_5year_mean
import step3_landsat_lst
import step4_export_geotiff
import step4b_download_drive_export
import step5_preprocess_timeseries
import step5b_diagnostic_report
import step5c_tvdi
import step6_validate_fire_relation


BASE_DIR = Path(__file__).resolve().parent
log, log_file = setup_logger("main")


def run_step(step_name: str, step_func) -> None:
    log.info("=" * 70)
    log.info(f"{step_name} BAŞLATILIYOR")
    log.info("=" * 70)

    result = step_func()

    log.info("=" * 70)
    log.info(f"{step_name} TAMAMLANDI")
    log.info("=" * 70)
    return result


def main() -> None:
    start_time = datetime.now()

    log.info("#" * 80)
    log.info("PIPELINE BAŞLIYOR (STEP1 -> STEP5)")
    log.info(f"Başlangıç zamanı: {start_time.isoformat()}")
    log.info("#" * 80)

    try:
        run_step("STEP 1", step1_fetch_modis.main)
        run_step("STEP 2", step2_modis_5year_mean.main)
        step3_result = run_step("STEP 3", step3_landsat_lst.main)
        run_step(
            "STEP 4",
            lambda: step4_export_geotiff.main(step3_result=step3_result),
        )
        run_step("STEP 4B", step4b_download_drive_export.main)
        run_step("STEP 5", step5_preprocess_timeseries.main)
        run_step("STEP 5C", step5c_tvdi.main)
        run_step("STEP 5B", step5b_diagnostic_report.main)

        # Step6 burned-area association testi. Yangın etiketleri GEE'den çekilir;
        # veri yoksa veya GEE erişimi başarısızsa pipeline'ın geri kalanı
        # etkilenmesin diye hata-toleranslı çağrılır.
        try:
            run_step("STEP 6", step6_validate_fire_relation.main)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "STEP 6 atlandı (burned-area validation başarısız): %s", exc
            )

        end_time = datetime.now()

        log.info("#" * 80)
        log.info("ONLINE PIPELINE TAMAMLANDI")
        log.info(f"Bitiş zamanı: {end_time.isoformat()}")
        log.info(f"Log dosyası: {log_file}")
        log.info("#" * 80)

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