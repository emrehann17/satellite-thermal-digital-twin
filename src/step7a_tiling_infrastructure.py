"""
step7a_tiling_infrastructure.py

Tile / window-safe raster işleme altyapısının doğrulama adımı.

Amaç:
    Büyük GeoTIFF'lerin tile bazlı güvenle işlenip grid drift olmadan yeniden
    birleştirilebildiğini test etmek. Bu adım BİLİMSEL ÜRÜN ÜRETMEZ; MODIS
    downscaling ve gap filling öncesi gereken altyapıyı doğrular.

Yapılanlar:
    - Referans raster seçilir (öncelik: DEM elevation -> current LST -> current TVDI
      -> anomaly zscore).
    - Raster pencerelerde okunur, her pencere bir output rastera yazılır.
    - Tam raster yeniden birleştirilir (mosaic).
    - Yeniden birleştirilen raster, orijinaliyle karşılaştırılır.

Sadece outputs/step7a altında çıktı üretir; bilimsel rasterları değiştirmez.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
import sys
import tempfile

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import rasterio

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from utils.tiling import (
    make_tile_grid,
    summarize_tile_grid,
    iter_windows,
    get_window_transform,
    read_window,
    write_window,
    create_output_profile_like,
    mosaic_tiles,
    compare_rasters,
)

BASE_DIR = PROJECT_ROOT
OUTPUTS_DIR = BASE_DIR / "outputs" / "step7a"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step7a")


# =============================================================================
# Referans raster seçimi
# =============================================================================
def select_reference_raster(explicit: str | None = None) -> Path | None:
    """Tiling testi için referans rasterı seçer (öncelik sırasına göre)."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        log.error("Belirtilen referans raster bulunamadı: %s", p)
        return None

    data_dir = BASE_DIR / "data"
    candidates: list[Path] = []

    # 1) DEM elevation
    candidates.append(data_dir / "dem" / "elevation.tif")
    # 2) current LST
    candidates.extend(sorted((data_dir / "current_period").glob("*.tif")))
    # 3) current TVDI (Step5C çıktısı)
    candidates.extend(sorted((BASE_DIR / "outputs" / "step5c").glob("*tvdi*.tif")))
    candidates.extend(sorted((data_dir / "tvdi").glob("*.tif"))) if (data_dir / "tvdi").exists() else None
    # 4) anomaly zscore
    candidates.extend(sorted((BASE_DIR / "outputs" / "step5").glob("*zscore*.tif")))

    for c in candidates:
        if c and c.exists():
            return c
    return None


# =============================================================================
# Tile/reconstruct testi
# =============================================================================
def run_tiling_test(
    reference_path: Path,
    tile_size: int = 512,
    overlap: int = 0,
    tolerance: float = 1e-6,
    force: bool = False,
) -> dict:
    """Referans rasterı tile'lara böler, yeniden birleştirir ve karşılaştırır."""
    reconstructed_path = OUTPUTS_DIR / "tiling_test_reconstructed.tif"
    difference_path = OUTPUTS_DIR / "tiling_test_difference.tif"

    if reconstructed_path.exists() and not force:
        log.warning(
            "Çıktı zaten var, --force verilmedi: %s (mevcut çıktı korunuyor).",
            reconstructed_path,
        )
        # Mevcut çıktıyı ezmeden, var olan rekonstrüksiyon üzerinden karşılaştır.
        if not difference_path.exists():
            log.info("Fark dosyası yok; yine de karşılaştırma yapılacak.")
    else:
        _tile_and_reconstruct(reference_path, reconstructed_path, tile_size, overlap)

    # Karşılaştırma
    comparison = compare_rasters(reference_path, reconstructed_path, tolerance=tolerance)

    # Fark rasterı (yalnızca outputs/step7a altında)
    if (not difference_path.exists()) or force:
        _write_difference_raster(reference_path, reconstructed_path, difference_path)

    with rasterio.open(reference_path) as src:
        grid = make_tile_grid(src, tile_size, overlap)
        raster_shape = [int(src.height), int(src.width)]
        crs = str(src.crs)
        transform = [
            src.transform.a, src.transform.b, src.transform.c,
            src.transform.d, src.transform.e, src.transform.f,
        ]

    summary = {
        "created_at": datetime.now().isoformat(),
        "reference_raster": str(reference_path),
        "tile_size_pixels": tile_size,
        "overlap_pixels": overlap,
        "tolerance": tolerance,
        "num_tiles": grid["n_tiles"],
        "tile_grid": summarize_tile_grid(grid),
        "raster_shape": raster_shape,
        "crs": crs,
        "transform": transform,
        "max_abs_difference": comparison["max_abs_difference"],
        "mean_abs_difference": comparison["mean_abs_difference"],
        "differing_pixels": comparison["differing_pixels"],
        "same_crs": comparison["same_crs"],
        "same_transform": comparison["same_transform"],
        "same_dimensions": comparison["same_dimensions"],
        "same_bounds": comparison["same_bounds"],
        "passed": comparison["passed"],
        "reconstructed_path": str(reconstructed_path),
        "difference_path": str(difference_path),
        "processing_note": (
            "Windowed/tiled processing via rasterio windows; the full raster is "
            "not loaded at once except for block-wise comparison."
        ),
    }
    return summary


def _tile_and_reconstruct(
    reference_path: Path,
    reconstructed_path: Path,
    tile_size: int,
    overlap: int,
) -> None:
    """Rasterı pencere pencere okuyup tile dosyalarına yazar, sonra mosaic'ler."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="step7a_tiles_"))
    tile_paths: list[Path] = []
    try:
        with rasterio.open(reference_path) as src:
            profile = src.profile.copy()
            for idx, (write_win, read_win, core_offset) in enumerate(
                iter_windows(src, tile_size, overlap)
            ):
                # Çekirdek (overlap'siz) pencereyi oku ve yaz: reconstruct birebir olsun.
                data = src.read(window=write_win)
                tile_profile = profile.copy()
                tile_profile.update(
                    width=int(write_win.width),
                    height=int(write_win.height),
                    transform=get_window_transform(src, write_win),
                )
                tile_path = tmp_dir / f"tile_{idx:05d}.tif"
                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(data)
                tile_paths.append(tile_path)
                _ = read_win, core_offset

        # Referans profile gridine göre mosaic
        with rasterio.open(reference_path) as src:
            ref_profile = src.profile.copy()
        mosaic_tiles(tile_paths, reconstructed_path, ref_profile)
        log.info("Yeniden birleştirme tamamlandı: %s", reconstructed_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_difference_raster(
    reference_path: Path,
    reconstructed_path: Path,
    difference_path: Path,
) -> None:
    """Orijinal ile yeniden birleştirilen raster farkını blok bazlı yazar."""
    import numpy as np

    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype="float32", nodata=float("nan"))
        with rasterio.open(reconstructed_path) as cand, rasterio.open(
            difference_path, "w", **profile
        ) as dst:
            for _, win in ref.block_windows(1):
                a = ref.read(1, window=win, masked=True).astype("float32").filled(np.nan)
                b = cand.read(1, window=win, masked=True).astype("float32").filled(np.nan)
                dst.write(np.abs(a - b), 1, window=win)


# =============================================================================
# Çıktı yazma
# =============================================================================
def write_summary(summary: dict) -> tuple[Path, Path]:
    json_path = OUTPUTS_DIR / "tiling_test_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Step7A Tiling Infrastructure Test",
        "",
        f"Created at: `{summary['created_at']}`",
        f"Reference raster: `{summary['reference_raster']}`",
        f"Tile size (pixels): `{summary['tile_size_pixels']}`",
        f"Overlap (pixels): `{summary['overlap_pixels']}`",
        f"Number of tiles: `{summary['num_tiles']}`",
        f"Raster shape (h, w): `{summary['raster_shape']}`",
        f"CRS: `{summary['crs']}`",
        f"Transform: `{summary['transform']}`",
        "",
        "## Reconstruction comparison",
        "",
        f"- Max absolute difference: `{summary['max_abs_difference']}`",
        f"- Mean absolute difference: `{summary['mean_abs_difference']}`",
        f"- Differing pixels: `{summary['differing_pixels']}`",
        f"- Same CRS: `{summary['same_crs']}`",
        f"- Same transform: `{summary['same_transform']}`",
        f"- Same dimensions: `{summary['same_dimensions']}`",
        f"- Same bounds: `{summary['same_bounds']}`",
        f"- **Passed: `{summary['passed']}`**",
        "",
        f"> {summary['processing_note']}",
        "",
        "## Outputs",
        "",
        f"- `{summary['reconstructed_path']}`",
        f"- `{summary['difference_path']}`",
        "",
    ]
    md_path = OUTPUTS_DIR / "tiling_test_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


# =============================================================================
# Ana akış
# =============================================================================
def main(
    reference_raster: str | None = None,
    tile_size: int = 512,
    overlap: int = 0,
    tolerance: float = 1e-6,
    force: bool = False,
) -> dict:
    log.info("=" * 60)
    log.info("STEP 7A BAŞLIYOR (tiling infrastructure test)")
    log.info("=" * 60)

    reference_path = select_reference_raster(reference_raster)
    if reference_path is None:
        msg = (
            "Tiling testi için referans raster bulunamadı. Önce DEM/LST/TVDI "
            "rasterlarından en az biri indirilmeli (örn. data/dem/elevation.tif)."
        )
        log.error(msg)
        summary = {
            "created_at": datetime.now().isoformat(),
            "reference_raster": None,
            "passed": False,
            "error": msg,
        }
        json_path = OUTPUTS_DIR / "tiling_test_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    log.info("Referans raster: %s", reference_path)
    summary = run_tiling_test(
        reference_path=reference_path,
        tile_size=tile_size,
        overlap=overlap,
        tolerance=tolerance,
        force=force,
    )
    json_path, md_path = write_summary(summary)

    log.info("Tiling testi tamamlandı: passed=%s", summary["passed"])
    log.info("Özet: %s", md_path)
    log.info("=" * 60)
    log.info("STEP 7A TAMAMLANDI")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step7A: tile/window-safe raster reconstruction test."
    )
    parser.add_argument("--reference-raster", default=None, help="Referans raster yolu.")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile boyutu (piksel).")
    parser.add_argument("--overlap", type=int, default=0, help="Tile overlap (piksel).")
    parser.add_argument("--force", action="store_true", help="Mevcut çıktıları ez.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Karşılaştırma toleransı.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        reference_raster=args.reference_raster,
        tile_size=args.tile_size,
        overlap=args.overlap,
        tolerance=args.tolerance,
        force=args.force,
    )