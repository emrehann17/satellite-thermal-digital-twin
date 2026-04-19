import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import logging
from datetime import datetime
from pathlib import Path

# ── Klasör yapısı ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "step5"
LOGS_DIR   = BASE_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True) #if these folders don't exist, create them
LOGS_DIR.mkdir(parents=True, exist_ok=True) #else do nothing

# ── Loglama ────────────────────────────────────────────────────────────────────
log_file = LOGS_DIR / f"step2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler() # terminale de yaz
    ]
)
log = logging.getLogger(__name__)

def main():
    log.info("=" * 60)
    log.info("STEP 5 INITIALIZING...")
    log.info("=" * 60)

    tif_path = DATA_DIR / "east_mediterranean_mean_LST.tif"
    
    if not tif_path.exists():
        log.error(f"ERROR: File not found! Please place the file in the following path: {tif_path}")
        return

    log.info(f"File Analyzing: {tif_path.name}")
    with rasterio.open(tif_path) as src:
        meta = src.meta
        log.info(f"resolution: {src.res}")
        log.info(f"cordinate reference system: {src.crs}")
        log.info(f"Width: {src.width}  Height: {src.height}  Bands: {src.count}")

        log.info(f"Temprature map creating as PNG...")
        plt.figure(figsize=(10, 8))
        
        #cmap='coolwarm' = mavi-kırmızı renk skalası, düşük sıcaklıklar mavi, yüksek sıcaklıklar kırmızı olarak gösterilir
        show(src, cmap='coolwarm', title="East Mediterranean Mean LST")

        png_path = DATA_DIR / "east_mediterranean_mean_LST.png"
        plt.savefig(png_path)
        plt.close()

        log.info(f"PNG created: {png_path} and saved to {png_path}")


if __name__ == "__main__":
    main()