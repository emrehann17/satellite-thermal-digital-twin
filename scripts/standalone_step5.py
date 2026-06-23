"""
standalone_step5.py

Yapılanlar:
    - Step5'teki akışı tek başına çalıştırır. Pipeline'ın geri kalan adımlarını çalıştırmaz, sadece Step5'e odaklanır.
"""

from datetime import datetime
from pathlib import Path
import traceback

from core.io_utils import setup_logger

import src.step5_preprocess_timeseries as step5_preprocess_timeseries
import src.step5b_diagnostic_report as step5b_diagnostic_report
import src.step5c_tvdi as step5c_tvdi

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
    log.info("STEP 5 PIPELINE BAŞLIYOR")
    log.info(f"Başlangıç zamanı: {start_time.isoformat()}")
    log.info("#" * 80)

    try:
        run_step("STEP 5", step5_preprocess_timeseries.main)
        run_step("STEP 5C", step5c_tvdi.main)
        run_step("STEP 5B", step5b_diagnostic_report.main)

        end_time = datetime.now()
        log.info("#" * 80)
        log.info("STEP 5 PIPELINE TAMAMLANDI")
        log.info(f"Bitiş zamanı: {end_time.isoformat()}")
        log.info(f"Toplam süre: {(end_time - start_time)}")
        log.info("#" * 80)

    except Exception as e:
        log.error("STEP 5 PIPELINE ÇALIŞIRKEN HATA OLUŞTU:")
        log.error(traceback.format_exc())

if __name__ == "__main__":
    main()