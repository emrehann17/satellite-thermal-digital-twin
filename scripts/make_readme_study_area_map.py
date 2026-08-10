"""
make_readme_study_area_map.py

README ust gorseli: kanonik bes-AOI spatial cohort'unun calisma alani haritasi.

Bu script BIR FIGUR URETICISIDIR; pipeline asamasi DEGILDIR, hicbir
outputs/ artefaktini okumaz veya yazmaz ve hicbir bilimsel metrigi
etkilemez. Cikti tek bir PNG'dir:

    assets/readme_study_areas.png

TEK KAYNAK (SINGLE SOURCE OF TRUTH)
-----------------------------------
Haritadaki AOI dikdortgenleri ELLE CIZILMEZ. Iki sey de dogrudan
`core/regions.py` kayit defterinden turetilir:

  1) Cohort uyeligi: `list_canonical_enabled_experiments()` (enabled +
     canonical) uzerinden, `src/burned_pattern_audit.NON_COHORT_ROLES`
     rolleri (negative_control, temporal_transfer_wildfire) elenerek --
     yani README'de belgelenen discovery mantiginin AYNISI. Hicbir
     experiment_id burada hard-code EDILMEZ.
  2) Geometri: her deneyin `region_key`'ine karsilik gelen
     `ee.Geometry.BBox(...)` cagrisinin ARGUMANLARI.

(2) icin GEE auth GEREKMEZ: `build_regions()` cagrilmaz, cunku
ee.Geometry yapilandirmasi initialize edilmis bir GEE oturumu ister.
Bunun yerine `core/regions.py` kaynagi AST ile ayristirilir ve
`build_regions()` icindeki BBox literalleri / modul sabitleri (*_AOI_BBOX)
okunur. Ayristirma fail-closed'dir: bir region_key cozulemezse hata verilir.

Muga 2022 (event-relative) deneyi spatial cohort'un uyesi DEGILDIR
(rolu `temporal_transfer_wildfire`); haritada, ayni AOI geometrisini
paylastigi icin cakisik kesikli bir cerceve + ayri lejant girdisi olarak
gosterilir -- sahte/kaydirilmis bir kutu CIZILMEZ.

Altlik (kiyi cizgisi / ulke sinirlari) Natural Earth'ten (public domain)
indirilir ve `.cache/naturalearth/` altinda onbeleklenir (git-ignored).
Repoya yalnizca uretilen PNG girer.

CLI:
    python scripts/make_readme_study_area_map.py
    python scripts/make_readme_study_area_map.py --output assets/readme_study_areas.png
    python scripts/make_readme_study_area_map.py --dry-run   # sadece cozulen cohort'u yaz
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.regions as regions  # noqa: E402
from core.regions import list_canonical_enabled_experiments  # noqa: E402
from src.burned_pattern_audit import NON_COHORT_ROLES  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "readme_study_areas.png"
CACHE_DIR = PROJECT_ROOT / ".cache" / "naturalearth"
NATURAL_EARTH_BASE = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
)

# --- Renk / tipografi (dataviz kategorik slot 1 ve 2) ------------------------
SURFACE = "#fcfcfb"
SEA = "#f1f4f7"
LAND = "#e9e7e1"
COAST = "#c6c3b9"
BORDER = "#fcfcfb"
GRATICULE = "#dedcd5"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8d8b84"
COHORT = "#2a78d6"          # kategorik slot 1
TEMPORAL_REF = "#eb6834"    # kategorik slot 2

# Ana haritanin cerceve kapsami (derece, EPSG:4326).
MAP_EXTENT = (-7.2, 37.3, 33.0, 45.9)  # lon_min, lon_max, lat_min, lat_max

# Detay panellerinin ORTAK penceresi (derece). Sabit oldugu icin paneller
# arasi AOI buyuklukleri gorsel olarak KARSILASTIRILABILIRDIR.
INSET_SPAN_LON = 2.60
INSET_SPAN_LAT = 1.70

# Etiket capalari: (metin_lon, metin_lat, ha, va). Yalnizca yerlesim
# (estetik) icindir; bulunamayan bir deney icin otomatik ofset kullanilir.
LABEL_ANCHORS: dict[str, tuple[float, float, str, str]] = {
    "bejis_2022": (-5.9, 43.7, "left", "center"),
    "montiferru_2021": (5.0, 44.0, "left", "center"),
    "evia_2021_extended": (18.4, 43.4, "left", "center"),
    "mugla_2021": (23.4, 34.2, "left", "center"),
    "manavgat_2021": (31.8, 34.2, "left", "center"),
}

# Ulke etiketleri: sade bir cografi cerceve; AOI ulkeleri vurgulu.
# Isimlendirme, AOI etiketlerinin ikinci satirindaki registry `country`
# degerleriyle tutarli olsun diye Ingilizcedir.
COUNTRY_LABELS: list[tuple[str, float, float, bool]] = [
    ("SPAIN", -3.6, 40.9, True),
    ("ITALY", 12.6, 42.6, True),
    ("GREECE", 21.6, 39.9, True),
    ("TURKEY", 34.0, 39.3, True),
    ("FRANCE", 2.4, 45.1, False),
    ("PORTUGAL", -8.0, 39.6, False),
    ("ALGERIA", 3.0, 34.6, False),
    ("TUNISIA", 9.6, 34.4, False),
    ("MOROCCO", -5.6, 33.6, False),
]


class StudyAreaMapError(RuntimeError):
    """AOI geometrisi veya cohort cozumlemesi basarisiz oldugunda firlatilir."""


# =============================================================================
# 1) core/regions.py -> AOI bbox'lari (GEE auth OLMADAN, AST ile)
# =============================================================================
def _bbox_from_call(node: ast.Call, module: Any) -> Optional[tuple[float, ...]]:
    """`ee.Geometry.BBox(...)` cagrisindan (lon_min, lat_min, lon_max, lat_max)."""
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "BBox"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "Geometry"
    ):
        return None

    # Bicim A: ee.Geometry.BBox(*MODUL_SABITI)
    if len(node.args) == 1 and isinstance(node.args[0], ast.Starred):
        starred = node.args[0].value
        if not isinstance(starred, ast.Name):
            raise StudyAreaMapError(
                f"Cozulemeyen starred BBox argumani (satir {node.lineno})."
            )
        value = getattr(module, starred.id, None)
        if value is None or len(tuple(value)) != 4:
            raise StudyAreaMapError(
                f"core.regions.{starred.id} 4 elemanli bir bbox degil."
            )
        return tuple(float(v) for v in value)

    # Bicim B: ee.Geometry.BBox(lon_min, lat_min, lon_max, lat_max)
    if len(node.args) == 4:
        try:
            return tuple(float(ast.literal_eval(a)) for a in node.args)
        except (ValueError, TypeError) as exc:  # pragma: no cover - savunmaci
            raise StudyAreaMapError(
                f"BBox argumanlari literal degil (satir {node.lineno}): {exc}"
            ) from exc

    raise StudyAreaMapError(f"Beklenmeyen BBox imzasi (satir {node.lineno}).")


def extract_region_bboxes() -> dict[str, tuple[float, float, float, float]]:
    """`build_regions()` icindeki region_key -> bbox eslemesini AST ile cozer.

    GEE oturumu GEREKTIRMEZ. `build_regions()`'in return dict'i okunarak
    region_key'ler yerel degisken adlarina, oradan da BBox argumanlarina
    baglanir (takma adlar da izlenir).
    """
    source = (PROJECT_ROOT / "core" / "regions.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    func_def = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_regions"
        ),
        None,
    )
    if func_def is None:
        raise StudyAreaMapError("core/regions.py icinde build_regions() bulunamadi.")

    local_bboxes: dict[str, tuple[float, ...]] = {}
    exported: dict[str, str] = {}

    for node in func_def.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if isinstance(node.value, ast.Call):
                bbox = _bbox_from_call(node.value, regions)
                if bbox is not None:
                    local_bboxes[target.id] = bbox
            elif isinstance(node.value, ast.Name) and node.value.id in local_bboxes:
                # Takma ad: manavgat_aoi = manavgat_aoi_refined_bbox
                local_bboxes[target.id] = local_bboxes[node.value.id]
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Name):
                    exported[str(key.value)] = value.id

    if not exported:
        raise StudyAreaMapError("build_regions() return dict'i ayristirilamadi.")

    resolved: dict[str, tuple[float, float, float, float]] = {}
    for region_key, local_name in exported.items():
        if local_name in local_bboxes:
            lon_min, lat_min, lon_max, lat_max = local_bboxes[local_name]
            resolved[region_key] = (lon_min, lat_min, lon_max, lat_max)
    return resolved


# =============================================================================
# 2) Cohort cozumlemesi (registry-guudumlu; hard-coded experiment_id YOK)
# =============================================================================
def _short_label(record: dict) -> str:
    """display_name'den kompakt, tek satirlik bir harita etiketi turetir.

    "Bejís / Castellón 2022" -> "Bejís 2022"
    "North Evia (Euboea), Greece -- extended AOI" -> "North Evia 2021"
    """
    name = str(record["display_name"])
    name = name.split("--")[0].split(",")[0].split("/")[0]
    name = name.split("(")[0].strip()
    year = str(record["label_start_date"])[:4]
    if year not in name:
        name = f"{name} {year}"
    return name


def resolve_study_areas() -> tuple[list[dict], list[dict], list[dict]]:
    """(spatial cohort, geometri-paylasan referanslar, haritalanmayanlar).

    Cohort filtresi README'de belgelenen discovery mantiginin aynisidir:
    enabled + canonical, NON_COHORT_ROLES (negative_control,
    temporal_transfer_wildfire) elenmis. Cohort uyeleri fail-closed'dir:
    bbox'i cozulemeyen bir cohort AOI'si HATA verir.

    Cohort-disi kanonik kayitlar haritaya YALNIZCA bir cohort AOI'siyle AYNI
    geometriyi (region_key) paylastiklarinda -- yani ayni footprint uzerinde
    farkli bir olay/pencere sozlesmesi olduklarinda -- referans olarak
    islenir. Kendi bagimsiz geometrisi olanlar (or. negatif kontrol)
    haritalanmaz; sessizce dusurulmez, CLI ciktisinda gerekcesiyle raporlanir.

    Sonuclar batidan doguya (bbox merkez boylamina gore) siralanir.
    """
    bboxes = extract_region_bboxes()
    canonical = list_canonical_enabled_experiments()

    cohort: list[dict] = []
    non_cohort: list[dict] = []
    for experiment_id, record in canonical.items():
        region_key = record.get("region_key")
        is_cohort = record.get("role") not in NON_COHORT_ROLES
        if region_key not in bboxes:
            if is_cohort:
                raise StudyAreaMapError(
                    f"{experiment_id}: region_key={region_key!r} icin bbox "
                    "cozulemedi (build_regions() icinde ee.Geometry.BBox degil?)."
                )
            non_cohort.append(
                {
                    "experiment_id": experiment_id,
                    "role": str(record.get("role", "")),
                    "region_key": str(region_key),
                    "bbox": None,
                }
            )
            continue

        entry = {
            "experiment_id": experiment_id,
            "label": _short_label(record),
            "country": str(record.get("country", "")),
            "role": str(record.get("role", "")),
            "region_key": region_key,
            "bbox": tuple(bboxes[region_key]),
        }
        (cohort if is_cohort else non_cohort).append(entry)

    if not cohort:
        raise StudyAreaMapError("Spatial cohort bos cozuldu.")

    cohort_region_keys = {entry["region_key"] for entry in cohort}
    references = [e for e in non_cohort if e["region_key"] in cohort_region_keys]
    unmapped = [e for e in non_cohort if e["region_key"] not in cohort_region_keys]

    cohort.sort(key=lambda e: 0.5 * (e["bbox"][0] + e["bbox"][2]))
    references.sort(key=lambda e: e["experiment_id"])
    unmapped.sort(key=lambda e: e["experiment_id"])
    return cohort, references, unmapped


def bbox_size_km(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """bbox'in yaklasik (genislik, yukseklik) km olcusu (orta enlemde)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_mid = 0.5 * (lat_min + lat_max)
    width = (lon_max - lon_min) * 111.32 * math.cos(math.radians(lat_mid))
    height = (lat_max - lat_min) * 110.57
    return width, height


# =============================================================================
# 3) Natural Earth altligi (indir + onbellekle + kirp)
# =============================================================================
def load_natural_earth(layer: str) -> dict:
    """Natural Earth GeoJSON katmanini onbellekten (yoksa agdan) yukler."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{layer}.geojson"
    if not cached.is_file():
        url = f"{NATURAL_EARTH_BASE}/{layer}.geojson"
        print(f"[naturalearth] indiriliyor: {url}")
        with urllib.request.urlopen(url, timeout=180) as response:
            payload = response.read()
        cached.write_bytes(payload)
    return json.loads(cached.read_text(encoding="utf-8"))


def polygons_in_extent(
    geojson: dict, extent: tuple[float, float, float, float], pad: float = 2.0
) -> list[np.ndarray]:
    """Kapsam icine dusen polygon halkalarini (dis halka) dizi olarak dondurur."""
    lon_min, lon_max, lat_min, lat_max = extent
    lon_min, lon_max = lon_min - pad, lon_max + pad
    lat_min, lat_max = lat_min - pad, lat_max + pad

    rings: list[np.ndarray] = []

    def add(coords: list) -> None:
        ring = np.asarray(coords, dtype=float)
        if ring.ndim != 2 or ring.shape[0] < 3:
            return
        if (
            ring[:, 0].max() < lon_min
            or ring[:, 0].min() > lon_max
            or ring[:, 1].max() < lat_min
            or ring[:, 1].min() > lat_max
        ):
            return
        rings.append(ring)

    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        if kind == "Polygon":
            for ring in coordinates:
                add(ring)
        elif kind == "MultiPolygon":
            for polygon in coordinates:
                for ring in polygon:
                    add(ring)
    return rings


def draw_basemap(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    land_layer: str,
    country_layer: Optional[str] = None,
    coast_width: float = 0.7,
) -> None:
    """Deniz zemini + kara dolgusu + (opsiyonel) ulke sinirlarini cizer."""
    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_facecolor(SEA)

    land = polygons_in_extent(load_natural_earth(land_layer), extent)
    ax.add_collection(
        PolyCollection(
            land, facecolors=LAND, edgecolors=COAST, linewidths=coast_width, zorder=1
        )
    )

    if country_layer is not None:
        borders = polygons_in_extent(load_natural_earth(country_layer), extent)
        ax.add_collection(
            LineCollection(borders, colors=BORDER, linewidths=0.9, zorder=2)
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / math.cos(math.radians(0.5 * (lat_min + lat_max))))
    for spine in ax.spines.values():
        spine.set_color(COAST)
        spine.set_linewidth(0.8)


def draw_graticule(ax: plt.Axes, extent: tuple[float, float, float, float], step: int) -> None:
    """Sade bir derece izgarasi ve kenar etiketleri."""
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.arange(math.ceil(lon_min / step) * step, lon_max + 1e-9, step)
    lats = np.arange(math.ceil(lat_min / step) * step, lat_max + 1e-9, step)
    for lon in lons:
        ax.axvline(lon, color=GRATICULE, linewidth=0.5, zorder=3)
    for lat in lats:
        ax.axhline(lat, color=GRATICULE, linewidth=0.5, zorder=3)

    ax.set_xticks(lons)
    ax.set_yticks(lats)
    ax.set_xticklabels(
        [f"{abs(int(v))}°{'E' if v >= 0 else 'W'}" for v in lons], fontsize=7.8
    )
    ax.set_yticklabels([f"{int(v)}°N" for v in lats], fontsize=7.8)
    ax.tick_params(axis="both", colors=INK_MUTED, length=2.5, width=0.6, pad=2)


def draw_scale_bar(
    ax: plt.Axes, extent: tuple[float, float, float, float], length_km: float = 300.0
) -> None:
    """Orta enlemde gecerli, yaklasik bir km olcegi."""
    lon_min, lon_max, lat_min, lat_max = extent
    lat_mid = 0.5 * (lat_min + lat_max)
    deg = length_km / (111.32 * math.cos(math.radians(lat_mid)))

    x0 = lon_min + 0.045 * (lon_max - lon_min)
    y0 = lat_min + 0.075 * (lat_max - lat_min)
    ax.plot([x0, x0 + deg], [y0, y0], color=INK_SOFT, linewidth=1.6, zorder=8,
            solid_capstyle="butt")
    for x in (x0, x0 + deg):
        ax.plot([x, x], [y0 - 0.12, y0 + 0.12], color=INK_SOFT, linewidth=1.0, zorder=8)
    ax.text(
        x0 + deg / 2, y0 + 0.28, f"{int(length_km)} km",
        ha="center", va="bottom", fontsize=7.6, color=INK_SOFT, zorder=8,
    )


# =============================================================================
# 4) Figur
# =============================================================================
def _aoi_patch(ax: plt.Axes, bbox: tuple[float, float, float, float], **kwargs) -> Rectangle:
    lon_min, lat_min, lon_max, lat_max = bbox
    patch = Rectangle(
        (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min, **kwargs
    )
    ax.add_patch(patch)
    return patch


def draw_main_map(
    ax: plt.Axes, cohort: list[dict], references: list[dict]
) -> None:
    lon_min, lon_max, lat_min, lat_max = MAP_EXTENT
    draw_basemap(
        ax,
        MAP_EXTENT,
        land_layer="ne_50m_land",
        country_layer="ne_50m_admin_0_countries",
        coast_width=0.7,
    )
    draw_graticule(ax, MAP_EXTENT, step=5)
    draw_scale_bar(ax, MAP_EXTENT, length_km=300.0)

    for name, lon, lat, primary in COUNTRY_LABELS:
        if not (lon_min < lon < lon_max and lat_min < lat < lat_max):
            continue
        ax.text(
            lon, lat, " ".join(name),
            ha="center", va="center", zorder=4,
            fontsize=8.2 if primary else 7.2,
            color=INK_SOFT if primary else INK_MUTED,
            fontweight="medium" if primary else "normal",
        )

    reference_keys = {entry["region_key"] for entry in references}

    for index, entry in enumerate(cohort, start=1):
        bbox = entry["bbox"]
        lon_c = 0.5 * (bbox[0] + bbox[2])
        lat_c = 0.5 * (bbox[1] + bbox[3])

        # Ayni geometriyi paylasan cohort-disi (temporal) deney: AYNI
        # koordinatlar uzerine, daha kalin kesikli bir dis halka olarak
        # cizilir (cizgi kalinligi yola gore ortalandigi icin mavi cerceve
        # ile beyaz ayirici disinda kalir). Kaydirilmis/sahte bir kutu
        # KESINLIKLE cizilmez -- geometri birebir aynidir.
        if entry["region_key"] in reference_keys:
            _aoi_patch(
                ax, bbox, facecolor="none", edgecolor=TEMPORAL_REF,
                linewidth=5.2, linestyle=(0, (2.4, 1.8)), zorder=5,
            )

        _aoi_patch(ax, bbox, facecolor="none", edgecolor=SURFACE, linewidth=3.0, zorder=6)
        _aoi_patch(
            ax, bbox, facecolor=COHORT, alpha=0.20, edgecolor="none", zorder=6
        )
        _aoi_patch(ax, bbox, facecolor="none", edgecolor=COHORT, linewidth=1.6, zorder=7)

        anchor = LABEL_ANCHORS.get(entry["experiment_id"])
        if anchor is None:
            anchor = (lon_c + 1.2, lat_c + 2.2, "left", "center")
        text_lon, text_lat, ha, va = anchor

        ax.annotate(
            "",
            xy=(lon_c, lat_c),
            xytext=(text_lon, text_lat),
            zorder=8,
            arrowprops=dict(
                arrowstyle="-", color=INK_MUTED, linewidth=0.8,
                shrinkA=2, shrinkB=2,
                connectionstyle="arc3,rad=0.0",
            ),
        )
        ax.text(
            text_lon, text_lat,
            f"{index}. {entry['label']}\n{entry['country']}",
            ha=ha, va=va, zorder=9, fontsize=9.6, color=INK, linespacing=1.45,
            bbox=dict(
                boxstyle="round,pad=0.34", facecolor=SURFACE,
                edgecolor=GRATICULE, linewidth=0.7, alpha=0.94,
            ),
        )
        ax.plot(
            [lon_c], [lat_c], marker="o", markersize=3.0,
            markerfacecolor=COHORT, markeredgecolor=SURFACE, markeredgewidth=0.8,
            zorder=8, linestyle="none",
        )


def draw_inset(ax: plt.Axes, entry: dict, index: int, reference_label: Optional[str]) -> None:
    lon_min, lat_min, lon_max, lat_max = entry["bbox"]
    lon_c, lat_c = 0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max)
    extent = (
        lon_c - INSET_SPAN_LON / 2, lon_c + INSET_SPAN_LON / 2,
        lat_c - INSET_SPAN_LAT / 2, lat_c + INSET_SPAN_LAT / 2,
    )
    draw_basemap(ax, extent, land_layer="ne_10m_land", coast_width=0.55)

    if reference_label is not None:
        _aoi_patch(ax, entry["bbox"], facecolor="none", edgecolor=TEMPORAL_REF,
                   linewidth=6.0, linestyle=(0, (2.6, 2.0)), zorder=4)
    _aoi_patch(ax, entry["bbox"], facecolor="none", edgecolor=SURFACE,
               linewidth=3.0, zorder=5)
    _aoi_patch(ax, entry["bbox"], facecolor=COHORT, alpha=0.18,
               edgecolor="none", zorder=5)
    _aoi_patch(ax, entry["bbox"], facecolor="none", edgecolor=COHORT,
               linewidth=1.6, zorder=6)

    ax.set_xticks([])
    ax.set_yticks([])

    width_km, height_km = bbox_size_km(entry["bbox"])
    ax.set_title(
        f"{index}. {entry['label']}",
        fontsize=9.8, color=INK, pad=5, loc="left",
    )
    caption = (
        f"{entry['experiment_id']}\n"
        f"{lon_min:.2f}°, {lat_min:.2f}° → {lon_max:.2f}°, {lat_max:.2f}°\n"
        f"≈ {width_km:.0f} × {height_km:.0f} km"
    )
    if reference_label is not None:
        caption += f"\n+ {reference_label}"
    ax.text(
        0.0, -0.045, caption,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7.8, color=INK_SOFT, linespacing=1.5,
    )


def build_figure(cohort: list[dict], references: list[dict]) -> plt.Figure:
    n_insets = len(cohort)

    # --- Yerlesim (inch) -----------------------------------------------------
    fig_width = 12.0
    margin_x = 0.30
    content_width = fig_width - 2 * margin_x

    lon_span = MAP_EXTENT[1] - MAP_EXTENT[0]
    lat_span = MAP_EXTENT[3] - MAP_EXTENT[2]
    map_aspect = 1.0 / math.cos(math.radians(0.5 * (MAP_EXTENT[2] + MAP_EXTENT[3])))
    map_width = content_width
    map_height = map_width * (lat_span * map_aspect) / lon_span

    inset_gap = 0.20
    inset_width = (content_width - (n_insets - 1) * inset_gap) / n_insets
    inset_aspect = 1.0 / math.cos(math.radians(0.5 * (MAP_EXTENT[2] + MAP_EXTENT[3])))
    inset_height = inset_width * (INSET_SPAN_LAT * inset_aspect) / INSET_SPAN_LON

    top_pad = 0.96        # baslik bandi
    legend_pad = 0.72     # harita <-> lejant
    inset_title_pad = 0.46
    inset_caption_pad = 0.86
    bottom_pad = 0.34

    fig_height = (
        top_pad + map_height + legend_pad + inset_title_pad
        + inset_height + inset_caption_pad + bottom_pad
    )

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=200, facecolor=SURFACE)

    def axes_rect(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_width, y / fig_height, w / fig_width, h / fig_height]

    # --- Baslik --------------------------------------------------------------
    fig.text(
        margin_x / fig_width, 1 - 0.30 / fig_height,
        "Çalışma alanları — kanonik beş-AOI spatial cohort",
        ha="left", va="top", fontsize=16.5, color=INK, fontweight="bold",
    )
    fig.text(
        margin_x / fig_width, 1 - 0.545 / fig_height,
        "Akdeniz havzası wildfire AOI'leri; kutular core/regions.py kayıt "
        "defterindeki gerçek AOI bbox'larıdır (EPSG:4326).",
        ha="left", va="top", fontsize=9.8, color=INK_SOFT,
    )

    # --- Ana harita ----------------------------------------------------------
    map_bottom = fig_height - top_pad - map_height
    ax_map = fig.add_axes(axes_rect(margin_x, map_bottom, map_width, map_height))
    draw_main_map(ax_map, cohort, references)

    # --- Lejant (iki satir; tam genislik) ------------------------------------
    ax_legend = fig.add_axes(
        axes_rect(margin_x, map_bottom - legend_pad + 0.10, content_width, 0.40)
    )
    ax_legend.set_axis_off()
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    swatch_w, swatch_h = 0.017, 0.30
    ax_legend.add_patch(
        Rectangle((0.0, 0.60), swatch_w, swatch_h, transform=ax_legend.transAxes,
                  facecolor=COHORT, alpha=0.20, edgecolor=COHORT, linewidth=1.4)
    )
    ax_legend.text(
        0.024, 0.75,
        "Kanonik spatial cohort AOI'si — enabled + canonical kayıtlardan "
        "cohort-dışı roller (negative_control, temporal_transfer_wildfire) elenerek çözülür",
        transform=ax_legend.transAxes, ha="left", va="center",
        fontsize=9.4, color=INK,
    )

    ax_legend.add_patch(
        Rectangle((0.0, 0.08), swatch_w, swatch_h, transform=ax_legend.transAxes,
                  facecolor="none", edgecolor=TEMPORAL_REF, linewidth=1.8,
                  linestyle=(0, (2.4, 1.8)))
    )
    if references:
        reference_text = (
            ", ".join(entry["experiment_id"] for entry in references)
            + " — aynı AOI geometrisi üzerinde olay-göreli (temporal) deney; "
            "spatial cohort'un parçası değildir"
        )
    else:
        reference_text = "geometri paylaşan cohort-dışı deney: yok"
    ax_legend.text(
        0.024, 0.23, reference_text,
        transform=ax_legend.transAxes, ha="left", va="center",
        fontsize=9.4, color=INK,
    )

    # --- Detay panelleri -----------------------------------------------------
    inset_bottom = bottom_pad + inset_caption_pad
    reference_by_region = {
        entry["region_key"]: entry["experiment_id"] for entry in references
    }
    for index, entry in enumerate(cohort, start=1):
        x = margin_x + (index - 1) * (inset_width + inset_gap)
        ax_inset = fig.add_axes(axes_rect(x, inset_bottom, inset_width, inset_height))
        draw_inset(
            ax_inset, entry, index, reference_by_region.get(entry["region_key"])
        )

    # --- Alt bilgi -----------------------------------------------------------
    fig.text(
        margin_x / fig_width, 0.115 / fig_height,
        "AOI bbox'ları: core/regions.py (EXPERIMENTS registry) · Sınırlar: "
        "Natural Earth 1:50m / 1:10m (public domain) · Üretim: "
        "scripts/make_readme_study_area_map.py",
        ha="left", va="bottom", fontsize=7.6, color=INK_MUTED,
    )
    return fig


# =============================================================================
# 5) CLI
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="README çalışma alanı haritasını core/regions.py'den üretir."
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"PNG çıktı yolu (varsayılan: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Yalnızca çözülen cohort'u ve bbox'ları yazdır; PNG üretme.",
    )
    args = parser.parse_args(argv)

    cohort, references, unmapped = resolve_study_areas()

    print(f"Spatial cohort ({len(cohort)} AOI):")
    for index, entry in enumerate(cohort, start=1):
        width_km, height_km = bbox_size_km(entry["bbox"])
        print(
            f"  {index}. {entry['experiment_id']:<22} {entry['region_key']:<24}"
            f" bbox={entry['bbox']}  ≈{width_km:.0f}×{height_km:.0f} km"
        )
    print(f"Geometri paylaşan cohort-dışı referanslar ({len(references)}):")
    for entry in references:
        print(
            f"  - {entry['experiment_id']:<26} role={entry['role']:<28}"
            f" region_key={entry['region_key']}"
        )
    print(f"Haritalanmayan kanonik kayıtlar ({len(unmapped)}):")
    for entry in unmapped:
        reason = (
            "bbox olmayan geometri" if entry["bbox"] is None
            else "bağımsız geometri (cohort AOI'siyle paylaşılmıyor)"
        )
        print(
            f"  - {entry['experiment_id']:<26} role={entry['role']:<28} {reason}"
        )

    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(cohort, references)
    figure.savefig(args.output, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    print(f"Yazıldı: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
