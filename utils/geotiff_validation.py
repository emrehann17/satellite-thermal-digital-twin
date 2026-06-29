"""
utils/geotiff_validation.py

GeoTIFF bütünlük (integrity) doğrulama yardımcıları.

Bu modül yalnızca DOĞRULAR ve RAPORLAR; rasterları onarmaz, resample etmez,
değer değiştirmez. Step4B indirme sonrası bozuk rasterları erkenden yakalamak
için kullanılır, ancak proje-spesifik bir bağımlılığı yoktur (generic).

Tasarım notları:
    - Stats hesaplaması, kullanılabilirse pencere/blok (windowed) okuma ile yapılır;
      Kozan AOI boyutlarında tam okuma da kabul edilebilir (mevcut proje stili).
    - Tüm fonksiyonlar yapılandırılmış (structured) sonuç sözlükleri döndürür.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.coords import BoundingBox


# =============================================================================
# Ürün-spesifik makul (sanity) değer aralıkları
# =============================================================================
# max_tol: üst sınır için izin verilen küçük taşma (örn. slope 90 derece + gürültü).
# min_finite_percent: "finite_percent should be high" beklenen ürünler için eşik.
PRODUCT_VALUE_RANGES: dict[str, dict] = {
    "elevation": {"min": -500.0, "max": 9000.0, "min_finite_percent": 95.0},
    "slope": {"min": 0.0, "max": 90.0, "max_tol": 1.0, "min_finite_percent": 95.0},
    "lst_celsius": {"min": -80.0, "max": 100.0},
    "modis_lst_celsius": {"min": -80.0, "max": 100.0},
    # Ham/ölçeklenmemiş Landsat ST export'u: Celsius aralığı UYGULANMAZ.
    # Yalnızca okunabilirlik / CRS / transform / finite kontrolleri yapılır.
    "raw_landsat_st": {"raw": True},
    "ndvi": {"min": -1.2, "max": 1.2},
    "tvdi": {"min": -0.05, "max": 1.05},
    "zscore": {"extreme_abs": 10.0, "extreme_fraction_warn": 0.05},
    "burned_label": {"binary": True, "tol": 1e-6},
    "land_cover": {"categorical": True},
}

# Aralık kontrolünün fiziksel olarak imkânsız sayıldığı (kritik) durumlar.
# Step4B bunları "critical error" olarak ele alır.
IMPOSSIBLE_RANGE_HARD = {
    "slope": {"max_hard": 200.0},  # slope max > 200 -> imkânsız
}

DEFAULT_TRANSFORM_TOLERANCE = 1e-6
DEFAULT_BOUNDS_TOLERANCE = 1e-6
DEFAULT_RESOLUTION_TOLERANCE = 1e-6


# =============================================================================
# Yardımcılar
# =============================================================================
def _empty_result(path: Path | str) -> dict:
    return {
        "path": str(path),
        "exists": False,
        "readable": False,
        "passed": False,
        "errors": [],
        "warnings": [],
        "stats": {},
    }


def _transform_to_list(transform) -> list[float]:
    return [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]


def _bounds_to_list(bounds: BoundingBox) -> list[float]:
    return [bounds.left, bounds.bottom, bounds.right, bounds.top]


def _approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


# =============================================================================
# 1. Stats (windowed/blok okuma destekli)
# =============================================================================
def compute_raster_stats(path: Path | str, sample: bool = False) -> dict:
    """
    Raster için kompakt istatistikler hesaplar.

    sample=False ise blok (window) bazlı tam tarama yapılır (belleğe tüm rasterı
    aynı anda almadan). sample=True ise yalnızca ilk birkaç blok örneklenir
    (büyük gelecek-AOI rasterları için hızlı ön kontrol).
    """
    path = Path(path)
    stats: dict = {}
    with rasterio.open(path) as src:
        stats["width"] = int(src.width)
        stats["height"] = int(src.height)
        stats["crs"] = str(src.crs) if src.crs else None
        stats["transform"] = _transform_to_list(src.transform)
        stats["bounds"] = _bounds_to_list(src.bounds)
        stats["dtype"] = str(src.dtypes[0]) if src.dtypes else None
        stats["band_count"] = int(src.count)
        stats["nodata"] = None if src.nodata is None else float(src.nodata)

        finite_count = 0
        nan_count = 0
        total = 0
        vmin = math.inf
        vmax = -math.inf
        vsum = 0.0
        vsum_sq = 0.0

        # Band 1 üzerinde blok bazlı tarama.
        windows = [win for _, win in src.block_windows(1)]
        if not windows:
            windows = [rasterio.windows.Window(0, 0, src.width, src.height)]
        if sample:
            windows = windows[: min(8, len(windows))]

        for win in windows:
            block = src.read(1, window=win, masked=True).astype("float64")
            arr = block.filled(np.nan)
            finite_mask = np.isfinite(arr)
            n_finite = int(finite_mask.sum())
            total += int(arr.size)
            nan_count += int(np.isnan(arr).sum())
            finite_count += n_finite
            if n_finite:
                finite_vals = arr[finite_mask]
                vmin = min(vmin, float(finite_vals.min()))
                vmax = max(vmax, float(finite_vals.max()))
                vsum += float(finite_vals.sum())
                vsum_sq += float(np.square(finite_vals).sum())

        stats["finite_count"] = finite_count
        stats["nan_count"] = nan_count
        stats["sampled"] = bool(sample)
        stats["finite_percent"] = (
            round(100.0 * finite_count / total, 4) if total else 0.0
        )
        if finite_count:
            mean = vsum / finite_count
            var = max(vsum_sq / finite_count - mean * mean, 0.0)
            stats["min"] = vmin
            stats["max"] = vmax
            stats["mean"] = mean
            stats["std"] = math.sqrt(var)
        else:
            stats["min"] = None
            stats["max"] = None
            stats["mean"] = None
            stats["std"] = None

    return stats


# =============================================================================
# 2. Temel doğrulama
# =============================================================================
def validate_geotiff_basic(path: Path | str, expected: dict | None = None) -> dict:
    """
    Temel GeoTIFF bütünlük kontrolleri + opsiyonel beklenen-değer kontrolleri.

    expected (opsiyonel) anahtarlar:
        expected_crs, expected_width, expected_height, expected_transform,
        expected_bounds, expected_band_count, expected_dtype, expected_resolution,
        expected_product_type
    Küçük sayısal transform/bounds farkları tolerans ile geçilir.
    """
    expected = expected or {}
    result = _empty_result(path)
    p = Path(path)

    if not p.exists():
        result["errors"].append("file does not exist")
        return result
    result["exists"] = True

    try:
        size = p.stat().st_size
    except OSError as exc:
        result["errors"].append(f"cannot stat file: {exc}")
        return result
    if size <= 0:
        result["errors"].append("file size is 0")
        return result

    try:
        stats = compute_raster_stats(p, sample=False)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"not readable by rasterio: {exc}")
        return result

    result["readable"] = True
    result["stats"] = stats
    errors = result["errors"]
    warnings = result["warnings"]

    # Temel yapı kontrolleri
    if stats.get("band_count", 0) <= 0:
        errors.append("band count <= 0")
    if not stats.get("crs"):
        errors.append("CRS missing")
    if not stats.get("transform"):
        errors.append("transform missing")
    if stats.get("width", 0) <= 0:
        errors.append("width <= 0")
    if stats.get("height", 0) <= 0:
        errors.append("height <= 0")
    bounds = stats.get("bounds")
    if not bounds or not (bounds[2] > bounds[0] and bounds[3] > bounds[1]):
        errors.append("bounds invalid")
    if not stats.get("dtype"):
        errors.append("dtype invalid")
    if stats.get("nodata") is None:
        warnings.append("no nodata value set")

    # NaN / sabit kontrolleri
    nan_res = validate_no_all_nan(p, stats=stats)
    errors.extend(nan_res["errors"])
    warnings.extend(nan_res["warnings"])

    const_res = validate_no_all_constant(p, stats=stats)
    errors.extend(const_res["errors"])
    warnings.extend(const_res["warnings"])

    # Beklenen-değer kontrolleri (opsiyonel)
    _apply_expected_checks(stats, expected, errors, warnings)

    # Ürün-spesifik aralık kontrolü (varsa)
    product_type = expected.get("expected_product_type")
    if product_type:
        range_res = validate_value_range(p, product_type, stats=stats)
        errors.extend(range_res["errors"])
        warnings.extend(range_res["warnings"])

    result["passed"] = len(errors) == 0
    return result


def _apply_expected_checks(stats: dict, expected: dict, errors: list, warnings: list) -> None:
    """Beklenen değer kontrollerini uygular (sıkı olanlar error, esnekler warning)."""
    if "expected_crs" in expected and expected["expected_crs"] is not None:
        if str(stats.get("crs")) != str(expected["expected_crs"]):
            errors.append(
                f"CRS mismatch: got {stats.get('crs')} expected {expected['expected_crs']}"
            )
    if "expected_band_count" in expected and expected["expected_band_count"] is not None:
        if stats.get("band_count") != expected["expected_band_count"]:
            errors.append(
                f"band count mismatch: got {stats.get('band_count')} "
                f"expected {expected['expected_band_count']}"
            )
    if "expected_dtype" in expected and expected["expected_dtype"] is not None:
        if str(stats.get("dtype")) != str(expected["expected_dtype"]):
            warnings.append(
                f"dtype mismatch: got {stats.get('dtype')} expected {expected['expected_dtype']}"
            )

    # width/height: tam eşitlik beklenir ama mismatch'i warning (minor extent) yapma seçeneği
    for key, sk in (("expected_width", "width"), ("expected_height", "height")):
        if key in expected and expected[key] is not None:
            if stats.get(sk) != expected[key]:
                warnings.append(
                    f"{sk} mismatch: got {stats.get(sk)} expected {expected[key]} (minor extent)"
                )

    # transform: toleranslı karşılaştırma
    if expected.get("expected_transform") is not None:
        tol = expected.get("transform_tolerance", DEFAULT_TRANSFORM_TOLERANCE)
        got = stats.get("transform") or []
        exp = list(expected["expected_transform"])
        if len(got) == len(exp) and all(_approx_equal(a, b, tol) for a, b in zip(got, exp)):
            pass
        else:
            warnings.append("transform differs beyond tolerance (minor extent mismatch)")

    # bounds: toleranslı karşılaştırma
    if expected.get("expected_bounds") is not None:
        tol = expected.get("bounds_tolerance", DEFAULT_BOUNDS_TOLERANCE)
        got = stats.get("bounds") or []
        exp = list(expected["expected_bounds"])
        if len(got) == len(exp) and all(_approx_equal(a, b, tol) for a, b in zip(got, exp)):
            pass
        else:
            warnings.append("bounds differ beyond tolerance (minor extent mismatch)")

    # resolution: transform a/e mutlak değerleri
    if expected.get("expected_resolution") is not None:
        tol = expected.get("resolution_tolerance", DEFAULT_RESOLUTION_TOLERANCE)
        got = stats.get("transform") or []
        if got:
            res_x = abs(got[0])
            res_y = abs(got[4])
            exp_res = expected["expected_resolution"]
            if isinstance(exp_res, (int, float)):
                exp_x = exp_y = float(exp_res)
            else:
                exp_x, exp_y = float(exp_res[0]), float(exp_res[1])
            if not (_approx_equal(res_x, exp_x, tol) and _approx_equal(res_y, exp_y, tol)):
                warnings.append(
                    f"resolution differs: got ({res_x}, {res_y}) expected ({exp_x}, {exp_y})"
                )


# =============================================================================
# 3. NaN / sabit kontrolleri
# =============================================================================
def validate_no_all_nan(path: Path | str, stats: dict | None = None) -> dict:
    """Raster tamamen NaN / nodata ise kritik hata döndürür."""
    result = {"errors": [], "warnings": []}
    if stats is None:
        stats = compute_raster_stats(path, sample=False)
    if stats.get("finite_count", 0) == 0:
        result["errors"].append("raster is entirely NaN/nodata (finite_count == 0)")
    return result


def validate_no_all_constant(path: Path | str, stats: dict | None = None) -> dict:
    """Raster tamamen sabit ise uyarı döndürür (kritik değil)."""
    result = {"errors": [], "warnings": []}
    if stats is None:
        stats = compute_raster_stats(path, sample=False)
    if stats.get("finite_count", 0) > 0:
        vmin = stats.get("min")
        vmax = stats.get("max")
        std = stats.get("std")
        if vmin is not None and vmax is not None and (
            _approx_equal(vmin, vmax, 1e-12) or (std is not None and std == 0.0)
        ):
            result["warnings"].append(
                f"raster is constant (all finite pixels == {vmin})"
            )
    return result


# =============================================================================
# 4. Ürün-spesifik aralık kontrolleri
# =============================================================================
def validate_value_range(path: Path | str, product_type: str, stats: dict | None = None) -> dict:
    """Ürün tipine göre makul değer aralığı kontrolü."""
    result = {"errors": [], "warnings": []}
    if stats is None:
        stats = compute_raster_stats(path, sample=False)

    spec = PRODUCT_VALUE_RANGES.get(product_type)
    if spec is None:
        result["warnings"].append(f"no value-range spec for product type '{product_type}'")
        return result

    if stats.get("finite_count", 0) == 0:
        result["errors"].append(f"{product_type}: no finite pixels for range check")
        return result

    # Ham export: değer aralığı kontrolü uygulanmaz, yalnızca bilgilendirme uyarısı.
    if spec.get("raw"):
        result["warnings"].append(
            "raw Landsat ST export detected; Celsius sanity range was not applied."
        )
        return result

    vmin = stats.get("min")
    vmax = stats.get("max")
    finite_percent = stats.get("finite_percent", 0.0)

    # finite_percent beklentisi
    min_fp = spec.get("min_finite_percent")
    if min_fp is not None and finite_percent < min_fp:
        if finite_percent <= 0:
            result["errors"].append(f"{product_type}: finite_percent is 0")
        else:
            result["warnings"].append(
                f"{product_type}: finite_percent {finite_percent}% below expected {min_fp}%"
            )

    # binary (burned_label)
    if spec.get("binary"):
        tol = spec.get("tol", 1e-6)
        if not (
            _approx_equal(vmin, 0.0, tol) or _approx_equal(vmin, 1.0, tol)
        ) or not (
            _approx_equal(vmax, 0.0, tol) or _approx_equal(vmax, 1.0, tol)
        ):
            result["warnings"].append(
                f"{product_type}: values not strictly binary (min={vmin}, max={vmax})"
            )
        return result

    # kategorik (land_cover): katı kontrol yok
    if spec.get("categorical"):
        return result

    # zscore: geniş aralık ama aşırı değer baskınsa uyar
    if "extreme_abs" in spec:
        extreme = spec["extreme_abs"]
        if (vmax is not None and vmax > extreme * 5) or (
            vmin is not None and vmin < -extreme * 5
        ):
            result["warnings"].append(
                f"{product_type}: extreme values dominate (min={vmin}, max={vmax})"
            )
        return result

    # genel min/max kontrolü
    lo = spec.get("min")
    hi = spec.get("max")
    max_tol = spec.get("max_tol", 0.0)

    # kritik (imkânsız) aralık kontrolü
    hard = IMPOSSIBLE_RANGE_HARD.get(product_type, {})
    if "max_hard" in hard and vmax is not None and vmax > hard["max_hard"]:
        result["errors"].append(
            f"{product_type}: max {vmax} exceeds impossible threshold {hard['max_hard']}"
        )

    if lo is not None and vmin is not None and vmin < lo:
        result["errors"].append(f"{product_type}: min {vmin} below allowed {lo}")
    if hi is not None and vmax is not None and vmax > hi + max_tol:
        result["errors"].append(
            f"{product_type}: max {vmax} above allowed {hi} (+tol {max_tol})"
        )

    return result


# =============================================================================
# 5. Hizalama (alignment) kontrolü
# =============================================================================
def validate_alignment(
    reference_path: Path | str,
    candidate_path: Path | str,
    transform_tolerance: float = DEFAULT_TRANSFORM_TOLERANCE,
    bounds_tolerance: float = DEFAULT_BOUNDS_TOLERANCE,
) -> dict:
    """İki rasterın CRS / transform / boyut / bounds uyumunu kontrol eder."""
    result = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "passed": False,
        "errors": [],
        "warnings": [],
    }
    try:
        with rasterio.open(reference_path) as ref, rasterio.open(candidate_path) as cand:
            if str(ref.crs) != str(cand.crs):
                result["errors"].append(
                    f"CRS mismatch: {ref.crs} vs {cand.crs}"
                )
            if (ref.width, ref.height) != (cand.width, cand.height):
                result["warnings"].append(
                    f"dimension mismatch: {ref.width}x{ref.height} vs "
                    f"{cand.width}x{cand.height} (minor extent)"
                )
            rt = _transform_to_list(ref.transform)
            ct = _transform_to_list(cand.transform)
            if not all(_approx_equal(a, b, transform_tolerance) for a, b in zip(rt, ct)):
                result["warnings"].append("transform differs beyond tolerance (minor extent)")
            rb = _bounds_to_list(ref.bounds)
            cb = _bounds_to_list(cand.bounds)
            if not all(_approx_equal(a, b, bounds_tolerance) for a, b in zip(rb, cb)):
                result["warnings"].append("bounds differ beyond tolerance (minor extent)")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"alignment check failed: {exc}")
        return result

    result["passed"] = len(result["errors"]) == 0
    return result


# =============================================================================
# 6. Rapor yazma
# =============================================================================
def write_geotiff_validation_report(results: list[dict], output_path: Path | str) -> Path:
    """
    Doğrulama sonuçlarını JSON + Markdown olarak yazar.

    output_path uzantısı .json verilir; aynı kök ile .md de yazılır.
    `results` her ürün için zenginleştirilmiş sonuç sözlükleri listesidir.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    products_passed = sum(1 for r in results if r.get("passed"))
    products_failed = sum(1 for r in results if not r.get("passed"))
    warnings_count = sum(len(r.get("warnings", [])) for r in results)

    payload = {
        "created_at": datetime.now().isoformat(),
        "products_checked": len(results),
        "products_passed": products_passed,
        "products_failed": products_failed,
        "warnings_count": warnings_count,
        "results": results,
    }

    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_path = output_path.with_suffix(".md")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return json_path


def _render_markdown(payload: dict) -> str:
    lines = [
        "# GeoTIFF Validation Summary",
        "",
        f"Created at: `{payload['created_at']}`",
        f"Products checked: `{payload['products_checked']}`",
        f"Passed: `{payload['products_passed']}`",
        f"Failed: `{payload['products_failed']}`",
        f"Warnings: `{payload['warnings_count']}`",
        "",
        "| Product | Type | Status | Source | Errors | Warnings |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for r in payload["results"]:
        status = "PASS" if r.get("passed") else "FAIL"
        lines.append(
            "| {name} | {ptype} | {status} | {source} | {nerr} | {nwarn} |".format(
                name=r.get("product", "?"),
                ptype=r.get("product_type", "?"),
                status=status,
                source=r.get("source", "?"),
                nerr=len(r.get("errors", [])),
                nwarn=len(r.get("warnings", [])),
            )
        )

    for r in payload["results"]:
        lines.extend(["", f"## {r.get('product', '?')}", ""])
        lines.append(f"- Path: `{r.get('path')}`")
        lines.append(f"- Product type: `{r.get('product_type')}`")
        lines.append(f"- Source: `{r.get('source')}`")
        lines.append(f"- Passed: `{r.get('passed')}`")
        if r.get("errors"):
            lines.append("- Errors:")
            for e in r["errors"]:
                lines.append(f"  - {e}")
        if r.get("warnings"):
            lines.append("- Warnings:")
            for w in r["warnings"]:
                lines.append(f"  - {w}")
        stats = r.get("stats") or {}
        if stats:
            lines.append(
                "- Stats: "
                f"size=`{stats.get('width')}x{stats.get('height')}`, "
                f"crs=`{stats.get('crs')}`, dtype=`{stats.get('dtype')}`, "
                f"finite%=`{stats.get('finite_percent')}`, "
                f"min=`{stats.get('min')}`, max=`{stats.get('max')}`, "
                f"mean=`{stats.get('mean')}`, std=`{stats.get('std')}`"
            )
    return "\n".join(lines) + "\n"