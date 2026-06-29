"""
utils/tiling.py

Generic tile / window-safe raster processing utilities.

Bu modül büyük rasterları belleğe tümüyle almadan pencere (window) bazlı işlemek
için tasarlanmıştır. Hiçbir proje-spesifik ürün burada hardcode edilmez; MODIS
downscaling, gap filling ve büyük-alan işleme öncesi altyapıdır.

Tasarım ilkeleri:
    - CRS, transform, width, height, bounds ve piksel hizalaması korunur.
    - Kenar (edge) tile'ları tile_size'dan küçük olabilir.
    - overlap_pixels desteklenir (varsayılan 0).
    - Test modu dışında rasterın tamamı okunmaz; rasterio windows kullanılır.
    - nodata propagasyonu desteklenir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform


# =============================================================================
# Tile grid
# =============================================================================
def make_tile_grid(
    dataset_or_profile,
    tile_size_pixels: int = 512,
    overlap_pixels: int = 0,
) -> dict:
    """
    Bir dataset veya profile'dan tile grid tanımı üretir.

    Dönen sözlük; raster boyutlarını, tile parametrelerini ve her tile için
    (write window + read window) çiftlerini içerir. Read window, overlap için
    genişletilmiş; write window ise overlap'siz çekirdek (core) alandır. Böylece
    yeniden birleştirme (mosaic) sırasında grid drift olmaz.
    """
    if tile_size_pixels <= 0:
        raise ValueError("tile_size_pixels must be > 0")
    if overlap_pixels < 0:
        raise ValueError("overlap_pixels must be >= 0")

    width, height = _extract_dims(dataset_or_profile)

    tiles = []
    n_cols = 0
    for row_off in range(0, height, tile_size_pixels):
        n_rows_this = 0
        for col_off in range(0, width, tile_size_pixels):
            core_w = min(tile_size_pixels, width - col_off)
            core_h = min(tile_size_pixels, height - row_off)

            # overlap'li okuma penceresi (raster sınırlarına clip edilir)
            read_col = max(0, col_off - overlap_pixels)
            read_row = max(0, row_off - overlap_pixels)
            read_col_end = min(width, col_off + core_w + overlap_pixels)
            read_row_end = min(height, row_off + core_h + overlap_pixels)

            tiles.append(
                {
                    "index": len(tiles),
                    "write_window": (col_off, row_off, core_w, core_h),
                    "read_window": (
                        read_col,
                        read_row,
                        read_col_end - read_col,
                        read_row_end - read_row,
                    ),
                    # write penceresinin, read penceresi içindeki ofseti (overlap kırpma)
                    "core_offset_in_read": (col_off - read_col, row_off - read_row),
                }
            )
            n_rows_this += 1
        n_cols = max(n_cols, n_rows_this)

    return {
        "width": width,
        "height": height,
        "tile_size_pixels": tile_size_pixels,
        "overlap_pixels": overlap_pixels,
        "n_tiles": len(tiles),
        "n_tile_cols": n_cols,
        "n_tile_rows": (height + tile_size_pixels - 1) // tile_size_pixels,
        "tiles": tiles,
    }


def _extract_dims(dataset_or_profile) -> tuple[int, int]:
    if hasattr(dataset_or_profile, "width") and hasattr(dataset_or_profile, "height"):
        return int(dataset_or_profile.width), int(dataset_or_profile.height)
    if isinstance(dataset_or_profile, dict):
        return int(dataset_or_profile["width"]), int(dataset_or_profile["height"])
    raise TypeError("dataset_or_profile must be a rasterio dataset or a profile dict")


def summarize_tile_grid(tile_grid: dict) -> dict:
    """Tile grid için kompakt özet döndürür (raporlama amaçlı)."""
    return {
        "width": tile_grid["width"],
        "height": tile_grid["height"],
        "tile_size_pixels": tile_grid["tile_size_pixels"],
        "overlap_pixels": tile_grid["overlap_pixels"],
        "n_tiles": tile_grid["n_tiles"],
        "n_tile_cols": tile_grid["n_tile_cols"],
        "n_tile_rows": tile_grid["n_tile_rows"],
    }


# =============================================================================
# Window iterasyonu
# =============================================================================
def iter_windows(dataset, tile_size_pixels: int = 512, overlap_pixels: int = 0):
    """
    Dataset üzerinde (write_window, read_window) rasterio.windows.Window çiftlerini üretir.

    write_window: overlap'siz çekirdek alan (mosaic'e bunun çekirdeği yazılır).
    read_window:  overlap dahil okunacak alan.
    """
    grid = make_tile_grid(dataset, tile_size_pixels, overlap_pixels)
    for tile in grid["tiles"]:
        wc, wr, ww, wh = tile["write_window"]
        rc, rr, rw, rh = tile["read_window"]
        yield (
            Window(wc, wr, ww, wh),
            Window(rc, rr, rw, rh),
            tile["core_offset_in_read"],
        )


def get_window_transform(dataset, window: Window):
    """Verilen pencere için affine transform döndürür (CRS/hizalama korunur)."""
    return window_transform(window, dataset.transform)


# =============================================================================
# Okuma / yazma
# =============================================================================
def read_window(
    dataset,
    window: Window,
    boundless: bool = False,
    fill_value=None,
    band: int = 1,
) -> np.ndarray:
    """
    Tek bandı verilen pencereden okur (masked -> NaN doldurma ile float32).

    boundless=True ise raster sınırı dışına taşan pencereler fill_value ile doldurulur.
    """
    if boundless:
        arr = dataset.read(
            band,
            window=window,
            boundless=True,
            fill_value=fill_value if fill_value is not None else np.nan,
            masked=True,
        )
    else:
        arr = dataset.read(band, window=window, masked=True)
    return arr.astype("float32").filled(np.nan)


def read_window_stack(paths, window: Window) -> np.ndarray:
    """
    Aynı pencereyi birden çok rasterdan okuyup (bands, h, w) stack döndürür.

    Tüm rasterların aynı grid/boyutta olduğu varsayılır (çağıran doğrulamalı).
    """
    layers = []
    for path in paths:
        with rasterio.open(path) as src:
            layers.append(read_window(src, window))
    return np.stack(layers, axis=0)


def write_window(output_dataset, window: Window, array: np.ndarray, band: int = 1) -> None:
    """Bir diziyi açık output dataset'e verilen pencerede yazar."""
    output_dataset.write(array, band, window=window)


# =============================================================================
# Profile / output
# =============================================================================
def create_output_profile_like(
    reference_path: Path | str,
    output_path: Path | str,
    dtype: str | None = None,
    nodata=None,
    count: int = 1,
) -> dict:
    """
    Referans rasterın profilini temel alarak bir output profile üretir.

    CRS / transform / width / height korunur; dtype, nodata, count override edilebilir.
    Output dosyası burada AÇILMAZ; yalnızca profile döndürülür (çağıran açar).
    """
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=count)
    if dtype is not None:
        profile.update(dtype=dtype)
    if nodata is not None:
        profile.update(nodata=nodata)
    profile.setdefault("driver", "GTiff")
    # output_path bilgisini de döndürmek yerine çağıran kullanır; imza uyumu için tutulur.
    _ = Path(output_path)
    return profile


# =============================================================================
# Mosaic (yeniden birleştirme)
# =============================================================================
def mosaic_tiles(tile_paths, output_path: Path | str, reference_profile: dict) -> Path:
    """
    Tile rasterlarını referans profile gridine göre tek rastera birleştirir.

    Her tile dosyasının kendi transform'u, referans grid içindeki konumunu belirler.
    Tile'lar pencere ofsetine göre yerleştirilir; grid drift olmaz.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_transform = reference_profile["transform"]
    inv = ~ref_transform

    with rasterio.open(output_path, "w", **reference_profile) as dst:
        for tile_path in tile_paths:
            with rasterio.open(tile_path) as src:
                data = src.read()
                # Tile'ın sol-üst köşesinin referans grid içindeki piksel ofseti
                left, top = src.transform.c, src.transform.f
                col_f, row_f = inv * (left, top)
                col_off = int(round(col_f))
                row_off = int(round(row_f))
                win = Window(col_off, row_off, src.width, src.height)
                dst.write(data, window=win)
    return output_path


# =============================================================================
# Karşılaştırma
# =============================================================================
def compare_rasters(
    reference_path: Path | str,
    candidate_path: Path | str,
    tolerance: float = 1e-6,
) -> dict:
    """
    İki rasterı pencere bazlı karşılaştırır (NaN/nodata eşitliği doğru ele alınır).

    Dönen sözlük: grid uyumu, max/mean mutlak fark, farklı piksel sayısı, passed.
    Bellek-güvenli: bloklar halinde okur.
    """
    result = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "same_crs": False,
        "same_transform": False,
        "same_dimensions": False,
        "same_bounds": False,
        "max_abs_difference": None,
        "mean_abs_difference": None,
        "differing_pixels": 0,
        "passed": False,
        "errors": [],
    }

    with rasterio.open(reference_path) as ref, rasterio.open(candidate_path) as cand:
        result["same_crs"] = str(ref.crs) == str(cand.crs)
        result["same_dimensions"] = (ref.width, ref.height) == (cand.width, cand.height)
        result["same_transform"] = all(
            abs(a - b) <= tolerance
            for a, b in zip(
                [ref.transform.a, ref.transform.b, ref.transform.c,
                 ref.transform.d, ref.transform.e, ref.transform.f],
                [cand.transform.a, cand.transform.b, cand.transform.c,
                 cand.transform.d, cand.transform.e, cand.transform.f],
            )
        )
        result["same_bounds"] = all(
            abs(a - b) <= tolerance
            for a, b in zip(list(ref.bounds), list(cand.bounds))
        )

        if not result["same_dimensions"]:
            result["errors"].append("dimension mismatch; cannot compare pixel values")
            return result

        max_abs = 0.0
        sum_abs = 0.0
        count = 0
        differing = 0

        for _, win in ref.block_windows(1):
            a = ref.read(1, window=win, masked=True).astype("float64").filled(np.nan)
            b = cand.read(1, window=win, masked=True).astype("float64").filled(np.nan)

            both_nan = np.isnan(a) & np.isnan(b)
            both_finite = np.isfinite(a) & np.isfinite(b)
            mismatch_nan = np.isnan(a) != np.isnan(b)

            # NaN konumları uyuşmuyorsa fark sayılır
            differing += int(mismatch_nan.sum())

            if both_finite.any():
                diff = np.abs(a[both_finite] - b[both_finite])
                if diff.size:
                    max_abs = max(max_abs, float(diff.max()))
                    sum_abs += float(diff.sum())
                    count += int(diff.size)
                    differing += int((diff > tolerance).sum())
            _ = both_nan

        result["max_abs_difference"] = max_abs
        result["mean_abs_difference"] = (sum_abs / count) if count else 0.0
        result["differing_pixels"] = differing

    result["passed"] = (
        result["same_crs"]
        and result["same_transform"]
        and result["same_dimensions"]
        and result["same_bounds"]
        and (result["max_abs_difference"] is not None)
        and result["max_abs_difference"] <= tolerance
        and result["differing_pixels"] == 0
        and not result["errors"]
    )
    return result