"""
export_mcd64a1_raw_burndate.py

Exports the RAW MCD64A1 BurnDate raster (day-of-year values) for the current
label window, so Step8A can build a proper burned/unburned modeling dataset.

WHY THIS EXISTS
---------------
Step6's `export_label_to_grid()` downloads a BINARY burned mask
(`MCD64A1_burned = mosaic.gt(0)`) and writes it to
`outputs/validation/labels/mcd64a1_raw.tif`. Despite the "_raw" suffix that is
the raw *download of a binary image*, NOT raw BurnDate DOY values -- every
burned pixel is 1. Step8A needs actual day-of-year BurnDate values to place
each burned cell into an August/September/October lead-time stratum, so it
cannot use that file.

This script exports the genuine BurnDate band (DOY values 1..366) and writes:
    outputs/validation/labels/mcd64a1_raw.tif      (raw BurnDate DOY)
and, optionally, a matching binary mask:
    outputs/validation/labels/mcd64a1_burned.tif   (BurnDate > 0)

Both are exported on the same grid/scale/CRS as Step6's label export
(VALIDATION_LABEL_EXPORT_SCALE, EXPORT_CRS, REGION_NAME AOI).

DERIVATION
----------
For a monthly product like MCD64A1, each monthly image's BurnDate band already
holds the DOY of burning within that month (0 = unburned). Over the Aug-Oct
window we take, per pixel, the MAXIMUM positive BurnDate across the monthly
images (last burn wins), which keeps a real DOY value rather than collapsing to
a binary flag. Unburned pixels remain 0.

Earth Engine source: MODIS/061/MCD64A1, band "BurnDate".

CLI
---
    python scripts/export_mcd64a1_raw_burndate.py
    python scripts/export_mcd64a1_raw_burndate.py --also-binary
    python scripts/export_mcd64a1_raw_burndate.py --start 2023-08-01 --end 2023-10-31

Requires a working GEE environment (earthengine-api + geemap + auth), exactly
like Step6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging

from core.config import (
    EXPORT_CRS,
    GEE_PROJECT,
    LABEL_START_DATE,
    LABEL_END_DATE,
    MCD64A1_BURNDATE_BAND,
    MCD64A1_COLLECTION,
    REGION_NAME,
    VALIDATION_LABEL_EXPORT_SCALE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger("export_mcd64a1_raw_burndate")

LABEL_DIR = _PROJECT_ROOT / "outputs" / "validation" / "labels"
RAW_OUT = LABEL_DIR / "mcd64a1_raw.tif"
BINARY_OUT = LABEL_DIR / "mcd64a1_burned.tif"


def build_raw_burndate_image(region, start: str, end: str):
    """
    Builds an ee.Image of raw MCD64A1 BurnDate DOY values over [start, end).

    Per-pixel maximum positive BurnDate across the monthly images in the
    window (0 stays 0 for unburned). Returns None-like status handling to the
    caller via exceptions.
    """
    import ee

    collection = (
        ee.ImageCollection(MCD64A1_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .select(MCD64A1_BURNDATE_BAND)
    )

    size = collection.size().getInfo()
    if size == 0:
        raise SystemExit(
            f"MCD64A1 secili AOI/pencerede ({start} -> {end}) hic goruntu "
            "dondurmedi. AOI'yi/pencereyi kontrol edin."
        )
    log.info("MCD64A1 goruntu sayisi: %d", size)

    # max() keeps a real DOY value per pixel (last burn wins); unburned = 0.
    raw_burndate = collection.max().rename("BurnDate").clip(region)
    return raw_burndate


def export_image(image, out_path: Path, scale: int, region, crs: str) -> None:
    import geemap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Export basliyor -> %s (scale=%dm, crs=%s)", out_path, scale, crs)
    geemap.ee_export_image(
        image,
        filename=str(out_path),
        scale=scale,
        region=region,
        crs=crs,
        file_per_band=False,
    )
    if not out_path.exists():
        raise SystemExit(f"Export dosyasi olusmadi: {out_path}")
    log.info("Yazildi: %s", out_path)


def inspect_output(path: Path, start: str, end: str) -> None:
    """Quick post-export sanity check that the raster holds DOY values, not {0,1}."""
    import numpy as np
    import rasterio
    from datetime import datetime

    start_doy = datetime.strptime(start, "%Y-%m-%d").timetuple().tm_yday
    end_doy = datetime.strptime(end, "%Y-%m-%d").timetuple().tm_yday

    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).compressed().astype("float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        log.warning("Cikti rasterinde gecerli piksel yok: %s", path)
        return
    positive = arr[arr > 0]
    in_range = int(np.sum((arr >= start_doy) & (arr <= end_doy)))
    log.info(
        "Cikti kontrolu: min=%.1f max=%.1f count_positive=%d count_one=%d "
        "count_in_DOY[%d-%d]=%d",
        float(arr.min()), float(arr.max()), int(positive.size),
        int(np.sum(arr == 1)), start_doy, end_doy, in_range,
    )
    if positive.size > 0 and bool(np.all(positive == 1.0)):
        log.error(
            "UYARI: tum pozitif degerler 1.0 -> bu hala BINARY gorunuyor. "
            "BurnDate bandinin DOY degerleriyle export edildiginden emin olun "
            "(mosaic.gt(0) KULLANMAYIN)."
        )
    elif in_range == 0:
        log.warning(
            "UYARI: label DOY penceresinde (%d-%d) hic deger yok. Pencereyi "
            "veya AOI'yi kontrol edin.", start_doy, end_doy,
        )
    else:
        log.info("OK: cikti gercek BurnDate DOY degerleri iceriyor gorunuyor.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export raw MCD64A1 BurnDate DOY raster for Step8A."
    )
    parser.add_argument("--start", type=str, default=LABEL_START_DATE)
    parser.add_argument("--end", type=str, default=LABEL_END_DATE)
    parser.add_argument("--out", type=str, default=str(RAW_OUT))
    parser.add_argument(
        "--also-binary", action="store_true",
        help="Also export the binary burned mask (BurnDate>0) to mcd64a1_burned.tif.",
    )
    parser.add_argument("--scale", type=int, default=VALIDATION_LABEL_EXPORT_SCALE)
    args = parser.parse_args(argv)

    try:
        import ee  # noqa: F401
        import geemap  # noqa: F401
        from core.gee_utils import init_gee
        from core.regions import build_regions
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "GEE importlari basarisiz (ee/geemap/regions). Kurulum: "
            "pip install earthengine-api geemap; auth: earthengine authenticate. "
            f"Hata: {type(exc).__name__}: {exc}"
        )

    try:
        init_gee(GEE_PROJECT)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"GEE init/auth basarisiz: {type(exc).__name__}: {exc}. "
            "'earthengine authenticate' calistirin ve GEE_PROJECT'i kontrol edin."
        )

    regions = build_regions()
    if REGION_NAME not in regions:
        raise SystemExit(f"Bolge bulunamadi: {REGION_NAME}")
    region = regions[REGION_NAME]

    log.info("AOI=%s, pencere=%s -> %s, scale=%dm", REGION_NAME, args.start, args.end, args.scale)

    raw_image = build_raw_burndate_image(region, args.start, args.end)
    raw_out = Path(args.out)
    if not raw_out.is_absolute():
        raw_out = _PROJECT_ROOT / raw_out
    export_image(raw_image, raw_out, args.scale, region, EXPORT_CRS)
    inspect_output(raw_out, args.start, args.end)

    if args.also_binary:
        binary_image = raw_image.gt(0).rename("MCD64A1_burned").clip(region)
        export_image(binary_image, BINARY_OUT, args.scale, region, EXPORT_CRS)

    log.info(
        "TAMAMLANDI. Step8A artik raw BurnDate rasterini kullanabilir: %s", raw_out
    )


if __name__ == "__main__":
    main()