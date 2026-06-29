"""
main.py

Yapılanlar:
    - Step1'den Step5'e kadar olan akışı sırasıyla çalıştırır
    - GEE tarafındaki online pipeline'ı tek yerden yönetir
"""

from pathlib import Path as _Path
import sys as _sys

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import datetime
import traceback

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT

import src.step1_fetch_modis as step1_fetch_modis
import src.step2_modis_5year_mean as step2_modis_5year_mean
import src.step2b_dem as step2b_dem
import src.step3_landsat_lst as step3_landsat_lst
import src.step4_export_geotiff as step4_export_geotiff
import src.step4b_download_drive_export as step4b_download_drive_export
import src.step5_preprocess_timeseries as step5_preprocess_timeseries
import src.step5b_diagnostic_report as step5b_diagnostic_report
import src.step5c_tvdi as step5c_tvdi
import src.step6_validate_fire_relation as step6_validate_fire_relation
import src.step7a_tiling_infrastructure as step7a_tiling_infrastructure


BASE_DIR = PROJECT_ROOT
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
        run_step("STEP 2B", step2b_dem.main)
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

        run_step("STEP 7A", step7a_tiling_infrastructure.main)

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