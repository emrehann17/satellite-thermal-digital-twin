"""
Authorised TerraClimate 1991-2020 climate-normals export for the marginal AoA
completion analysis.

This module is the ONLY place in the completion analysis that touches Earth
Engine. `src/marginal_aoa_completion.py` stays GEE-free and imports the engine
below lazily, from the `climate-export` stage alone -- which is what makes the
`gee_queries_run` / `gee_exports_run` flags truthful for every other stage.

Scientific contract (frozen, see docs/marginal_aoa_completion_design/):

    collection   IDAHO_EPSCOR/TERRACLIMATE
    period       1991-01-01 .. 2020-12-31   (exactly 360 monthly images)
    warm season  months 6, 7, 8, 9
    region       lon [-10, 42], lat [30, 47]
    projection   TerraClimate native

Band scaling is applied BEFORE any aggregation:

    tmmn x 0.1 (degC)   tmmx x 0.1 (degC)   def x 0.1 (mm)
    vpd  x 0.01 (kPa)   pr   x 1.0  (mm)

Exactly four output bands are produced:

    1. annual_mean_temperature_c              mean of (tmmx+tmmn)/2 over 360 months
    2. annual_precipitation_mm                mean of the 30 annual pr sums
    3. warm_season_climatic_water_deficit_mm  mean of the 30 Jun-Sep def sums
    4. warm_season_vpd_kpa                    mean of vpd over all Jun-Sep months

The engine is injectable: tests substitute a fake so the export code path is
exercised without contacting Earth Engine. Nothing here ever fabricates a
raster -- a failed or incomplete export raises.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class ClimateExportError(SystemExit):
    """Fail-fast condition in the authorised climate export."""


PROJECTION_READ_METHOD = "ee_single_band_projection_with_wkt_fallback_v1"
PROJECTION_TRANSFORM_LENGTH = 6

# TerraClimate publishes no EPSG authority code: every band's projection
# reports `crs = None` and carries only a WKT for an "unknown" GEOGCS on a
# WGS84-shaped spheroid. A probe of all five source bands (pr, tmmn, tmmx, def,
# vpd) confirmed one identical grid, so a single band is a sufficient and
# unambiguous projection source.
CANONICAL_PROJECTION_BAND = "pr"


def validate_projection(projection: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the projection is fully and truthfully readable.

    A missing EPSG authority code is NOT a failure on its own: TerraClimate has
    none, and its WKT is the only CRS specification it publishes. What must
    never happen is an EMPTY export CRS reaching the exporter -- that is the
    defect that let a real raster be compared against `None`. Nothing is
    defaulted or guessed, and EPSG:4326 is never assumed.
    """
    authority = projection.get("source_projection_authority_crs")
    authority = authority.strip() if isinstance(authority, str) else None
    wkt = projection.get("source_projection_wkt")
    wkt = wkt.strip() if isinstance(wkt, str) else None

    if authority:
        export_crs, representation = authority, "authority_code"
    elif wkt:
        export_crs, representation = wkt, "wkt"
    else:
        raise ClimateExportError(
            "Native projection carries NEITHER an authority CRS code NOR a "
            f"WKT (authority={projection.get('source_projection_authority_crs')!r}, "
            f"wkt={projection.get('source_projection_wkt')!r}). The export will "
            "not start: an empty CRS specification must never be passed to the "
            "exporter, and EPSG:4326 is not assumed as a fallback. Read "
            f"method: {projection.get('projection_read_method')}."
        )

    transform = projection.get("source_projection_transform")
    if (
        not isinstance(transform, (list, tuple))
        or len(transform) != PROJECTION_TRANSFORM_LENGTH
        or not all(isinstance(v, (int, float)) and math.isfinite(float(v))
                   for v in transform)
    ):
        raise ClimateExportError(
            f"Native projection transform is unusable ({transform!r}); "
            f"exactly {PROJECTION_TRANSFORM_LENGTH} finite numeric values are "
            "required. The export will not start."
        )

    scale = projection.get("source_projection_nominal_scale")
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = float("nan")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ClimateExportError(
            f"Native projection nominal scale is unusable "
            f"({projection.get('source_projection_nominal_scale')!r}); a finite "
            "positive value in metres is required. The export will not start."
        )

    return {
        **projection,
        "canonical_projection_band": projection.get(
            "canonical_projection_band", CANONICAL_PROJECTION_BAND
        ),
        "source_projection_authority_crs": authority,
        "source_projection_wkt": wkt,
        "source_projection_transform": [float(v) for v in transform],
        "source_projection_nominal_scale": scale,
        "export_crs": export_crs,
        "export_crs_representation": representation,
        "projection_read_method": projection.get(
            "projection_read_method", PROJECTION_READ_METHOD
        ),
    }


def crs_matches(observed: Any, expected: str, representation: str) -> bool:
    """Semantic CRS equivalence, never string equality.

    Writing a CRS into a GeoTIFF normalises it, so neither the raw WKT nor even
    two parsed `CRS` objects compare equal after a round trip -- observed for
    TerraClimate's "unknown" GEOGCS. What IS stable is the normalised PROJ
    representation, so equivalence is decided on that: identical `to_dict()`
    (or, failing that, identical PROJ.4), with a direct `==` tried first
    because it settles authority-coded CRSs cleanly.
    """
    import rasterio.crs

    if observed is None:
        return False
    try:
        left = (
            observed if isinstance(observed, rasterio.crs.CRS)
            else rasterio.crs.CRS.from_user_input(str(observed))
        )
        right = (
            rasterio.crs.CRS.from_wkt(expected) if representation == "wkt"
            else rasterio.crs.CRS.from_user_input(expected)
        )
    except Exception:  # noqa: BLE001 -- an unparsable CRS is a mismatch
        return False

    if left == right:
        return True
    try:
        if left.to_dict() and left.to_dict() == right.to_dict():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return bool(left.to_proj4()) and left.to_proj4() == right.to_proj4()
    except Exception:  # noqa: BLE001
        return False


class TerraClimateExportEngine:
    """Production Earth Engine engine for the authorised climate export.

    Every Earth Engine symbol is imported INSIDE a method, so importing this
    module costs nothing and requires no credentials. Construction is free;
    only `initialise()` contacts Earth Engine.
    """

    name = "terraclimate_production_v1"
    contacts_earth_engine = True

    def __init__(self, project: Optional[str] = None) -> None:
        self.project = project
        self._initialised = False

    # -- Earth Engine session ------------------------------------------------
    def initialise(self) -> dict[str, Any]:
        from core.gee_utils import init_gee

        if self.project is not None:
            init_gee(self.project)
        else:
            init_gee()
        self._initialised = True
        return {"initialised": True, "project": self.project}

    # -- Collection ----------------------------------------------------------
    def _scaled_collection(self, collection_id: str, period_start: str,
                           period_end_exclusive: str, scale_factors: dict[str, float]):
        import ee

        def _scale(image):
            bands = [
                image.select(band).multiply(float(factor)).rename(band)
                for band, factor in sorted(scale_factors.items())
            ]
            return (
                ee.Image.cat(bands)
                .copyProperties(image, ["system:time_start", "system:index"])
            )

        raw = ee.ImageCollection(collection_id).filterDate(
            period_start, period_end_exclusive
        )
        return raw, ee.ImageCollection(raw.map(_scale))

    def monthly_image_count(
        self, collection_id: str, period_start: str, period_end_exclusive: str,
    ) -> int:
        import ee

        collection = ee.ImageCollection(collection_id).filterDate(
            period_start, period_end_exclusive
        )
        return int(collection.size().getInfo())

    # -- Four-band climate image --------------------------------------------
    def build_four_band_image(
        self, *, collection_id: str, period_start: str, period_end_exclusive: str,
        years: Sequence[int], season_months: Sequence[int],
        scale_factors: dict[str, float], output_bands: Sequence[str],
    ):
        import ee

        _raw, scaled = self._scaled_collection(
            collection_id, period_start, period_end_exclusive, scale_factors
        )

        month_start, month_end = int(min(season_months)), int(max(season_months))
        warm = scaled.filter(ee.Filter.calendarRange(month_start, month_end, "month"))

        # 1. Mean monthly air temperature over every month in the record.
        annual_mean_temperature = (
            ee.ImageCollection(
                scaled.map(
                    lambda image: image.select("tmmx")
                    .add(image.select("tmmn"))
                    .divide(2.0)
                    .rename(output_bands[0])
                )
            ).mean()
        )

        # 2/3. Yearly sums first, then the mean across years -- a mean of sums,
        #      never a sum of means.
        annual_precipitation = ee.ImageCollection([
            scaled.filter(ee.Filter.calendarRange(int(year), int(year), "year"))
            .select("pr").sum().rename(output_bands[1])
            for year in years
        ]).mean()

        warm_season_deficit = ee.ImageCollection([
            warm.filter(ee.Filter.calendarRange(int(year), int(year), "year"))
            .select("def").sum().rename(output_bands[2])
            for year in years
        ]).mean()

        # 4. Mean vapour-pressure deficit over the warm-season months.
        warm_season_vpd = warm.select("vpd").mean().rename(output_bands[3])

        composed = ee.Image.cat(
            annual_mean_temperature,
            annual_precipitation,
            warm_season_deficit,
            warm_season_vpd,
        ).toFloat()

        # Collection reductions lose the source grid, so bind it back
        # explicitly. setDefaultProjection -- NOT reproject -- so the export
        # request carries the native transform and CRS without resampling the
        # data.
        return composed.setDefaultProjection(self.source_projection(collection_id))

    def source_projection(self, collection_id: str):
        """The Earth Engine Projection of the CANONICAL SINGLE BAND.

        A multi-band `first_image.projection()` is deliberately NOT used: on a
        multi-band image Earth Engine has no single well-defined projection to
        report. The derived four-band climate image is not used as the native
        source either -- it is the product of collection reductions, so its
        projection is not the source grid.
        """
        import ee

        return (
            ee.Image(ee.ImageCollection(collection_id).first())
            .select(CANONICAL_PROJECTION_BAND)
            .projection()
        )

    def native_projection(self, collection_id: str) -> dict[str, Any]:
        """Read the source grid from the canonical band's projection.

        `getInfo()` is the primary source for the transform and the WKT.
        The authority code is read from `getInfo()["crs"]` when present and
        otherwise from `projection.crs().getInfo()`; TerraClimate publishes
        neither, so the WKT becomes the export CRS specification. Nothing is
        guessed and EPSG:4326 is never assumed.
        """
        projection = self.source_projection(collection_id)
        info = projection.getInfo() or {}

        authority = info.get("crs")
        if not (isinstance(authority, str) and authority.strip()):
            try:
                authority = projection.crs().getInfo()
            except Exception:  # noqa: BLE001 -- absence is legitimate here
                authority = None

        wkt = info.get("wkt")
        if not (isinstance(wkt, str) and wkt.strip()):
            try:
                wkt = projection.wkt().getInfo()
            except Exception:  # noqa: BLE001
                wkt = None

        return validate_projection({
            "canonical_projection_band": CANONICAL_PROJECTION_BAND,
            "source_projection_authority_crs": authority,
            "source_projection_wkt": wkt,
            "source_projection_transform": info.get("transform"),
            "source_projection_nominal_scale": projection.nominalScale().getInfo(),
            "projection_read_method": PROJECTION_READ_METHOD,
            "projection_getinfo_diagnostic": info,
        })

    def region(self, bbox: dict[str, float]):
        import ee

        return ee.Geometry.BBox(
            float(bbox["lon_min"]), float(bbox["lat_min"]),
            float(bbox["lon_max"]), float(bbox["lat_max"]),
        )

    # -- Export --------------------------------------------------------------
    def export(
        self, image, *, destination: Path, region, scale: float, crs: str,
        band_count: int, tiles_dir: Path, force: bool = False,
        crs_equivalence_fn=None,
    ) -> dict[str, Any]:
        """Reuse the repository's tested direct-export + tiled-fallback helper.

        `destination` here is the caller's TEMPORARY sibling path, never the
        final production raster. The shared helper promotes its own internal
        temp file to whatever `out_path` it is given BEFORE the caller's
        climate-specific QA runs, so pointing it at the final path would leave
        a rejected raster there on failure -- which is exactly what happened.
        Isolating that behaviour in this wrapper leaves the shared exporter
        untouched.

        `crs_equivalence_fn` is handed to the shared helper's alignment QA.
        TerraClimate publishes no authority code, and GDAL normalises its
        "unknown" GEOGCS on write -- renaming the datum and spheroid and
        swapping the axis order -- so the shared helper's default string
        comparison cannot match a semantically identical CRS. Only the CRS
        comparison is affected; transform, pixel size, bounds, band count and
        dtype checks are untouched.
        """
        from scripts.run_predictors_only import export_image_direct_or_tiled

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return export_image_direct_or_tiled(
            image=image,
            out_path=destination,
            region=region,
            scale=int(round(float(scale))),
            crs=crs,
            label="marginal_aoa_completion_climate_normals",
            force=force,
            tiles_dir=Path(tiles_dir),
            band_count=int(band_count),
            run_alignment_qa=True,
            crs_equivalence_fn=crs_equivalence_fn,
        )


def validate_exported_raster(
    path: Path, *, expected_bands: Sequence[str],
    expected_crs: Optional[str] = None,
    expected_crs_representation: Optional[str] = None,
) -> dict[str, Any]:
    """Verify the exported raster before it is allowed to count as complete.

    Checks band count, CRS, transform, dimensions and that a non-empty finite
    support exists across ALL bands simultaneously. A raster that fails any of
    these is rejected rather than accepted with a warning.
    """
    import numpy as np
    import rasterio

    path = Path(path)
    if not path.is_file():
        raise ClimateExportError(
            f"Climate export produced no raster at {path}."
        )
    if path.stat().st_size == 0:
        raise ClimateExportError(
            f"Climate export produced a ZERO-BYTE raster at {path}; refusing to "
            "treat it as complete."
        )

    with rasterio.open(path) as handle:
        band_count = int(handle.count)
        crs = str(handle.crs) if handle.crs is not None else None
        transform = handle.transform
        height, width = int(handle.height), int(handle.width)
        data = handle.read(masked=True).astype("float64")

    if band_count != len(expected_bands):
        raise ClimateExportError(
            f"Climate raster {path} has {band_count} band(s); the frozen "
            f"contract requires exactly {len(expected_bands)} "
            f"({list(expected_bands)})."
        )
    if crs is None:
        raise ClimateExportError(f"Climate raster {path} carries no CRS.")
    if expected_crs is not None:
        # Semantic equivalence, not string equality: a WKT round-tripped
        # through GDAL is rarely byte-identical to the one Earth Engine
        # reported.
        if not crs_matches(crs, expected_crs, expected_crs_representation or "wkt"):
            raise ClimateExportError(
                f"Climate raster {path} CRS is not equivalent to the requested "
                f"export CRS.\n  raster   : {crs}\n  expected ({expected_crs_representation}): "
                f"{expected_crs[:120]}"
            )
    if height <= 0 or width <= 0:
        raise ClimateExportError(
            f"Climate raster {path} has degenerate dimensions {height}x{width}."
        )
    if transform is None or transform.a == 0 or transform.e == 0:
        raise ClimateExportError(
            f"Climate raster {path} has a degenerate transform: {transform}."
        )

    values = np.ma.filled(data, np.nan)
    support = np.isfinite(values).all(axis=0)
    n_support = int(support.sum())
    if n_support == 0:
        raise ClimateExportError(
            f"Climate raster {path} has NO pixel that is finite across all "
            f"{band_count} bands; the intersection of valid support is empty."
        )

    return {
        "band_count": band_count,
        "band_names": list(expected_bands),
        "crs": crs,
        "expected_crs_representation": expected_crs_representation,
        "crs_semantically_equivalent": (
            None if expected_crs is None
            else crs_matches(crs, expected_crs, expected_crs_representation or "wkt")
        ),
        "transform": [transform.a, transform.b, transform.c,
                      transform.d, transform.e, transform.f],
        "height": height,
        "width": width,
        "n_pixels": height * width,
        "n_finite_support_pixels": n_support,
        "finite_support_fraction": n_support / float(height * width),
    }
