"""
ERA5-Land AOI-level regional meteorology diagnostic (standalone, explanatory).

WHAT THIS IS
------------
An EXPLANATORY, AOI-level regional meteorology diagnostic: for each frozen
canonical wildfire AOI it summarises the hourly ERA5-Land near-surface state
over that experiment's OWN predictor and label windows, and expresses each
statistic as an anomaly against a fixed 2017-2020 climatology of the SAME
calendar window.

WHAT THIS IS NOT
----------------
It is NOT a model predictor, NOT a feature source, and NOT part of Step5,
Step7, Step8, Step9 or Step10. Nothing here writes into any experiment
namespace, and no frozen AOI/cohort analysis is read, modified or re-run. Its
output is a table, never a raster: no export is performed.

COHORT (frozen default)
-----------------------
    manavgat_2021, bejis_2022, mugla_2021, evia_2021_extended, montiferru_2021

`mugla_2022` is deliberately NOT in the default cohort: its temporal contract
is provisional and awaiting a supervisor decision. It can only enter by an
explicit `--experiments` request, and the default analysis_id is bound to the
five IDs above in exactly that order.

SOURCE AND WINDOWS
------------------
    collection  ECMWF/ERA5_LAND/HOURLY
    bands       temperature_2m, dewpoint_temperature_2m,
                u_component_of_wind_10m, v_component_of_wind_10m,
                total_precipitation_hourly

Observed windows come from `core.regions` ONLY -- no date is hard-coded here.
Registry end dates are INCLUSIVE calendar days, while Earth Engine's
`filterDate` end is EXCLUSIVE, so every window is converted exactly once:

    [start 00:00 UTC, end_inclusive + 1 day 00:00 UTC)

CLIMATOLOGY
-----------
Reference years 2017, 2018, 2019, 2020. The observed window's exact month/day
start and end are mapped into each reference year, each yearly window statistic
is computed INDEPENDENTLY, and only then:

    climatology_mean       = arithmetic mean of the four yearly statistics
    climatology_sd         = SAMPLE standard deviation, ddof = 1
    anomaly                = observed - climatology_mean
    standardized_anomaly   = anomaly / climatology_sd

If `climatology_sd` is exactly zero the standardized anomaly is None (never
inf, never 0) and `zero_climatology_sd` is recorded True. A month/day that
cannot be represented in a reference year (29 February) fails closed rather
than being silently shifted; the frozen five AOIs contain no such date.

DERIVED VARIABLES (per pixel, per hour -- never from window means)
-----------------------------------------------------------------
    temperature_c  = temperature_2m - 273.15

    relative humidity, ECMWF/Tetens saturation over WATER:
        T0 = 273.16 K, a1 = 611.21 Pa, a3 = 17.502, a4 = 32.19 K
        es(T) = a1 * exp(a3 * (T - T0) / (T - a4))
        RH    = 100 * es(Td) / es(T)
    RH is NOT clipped: the formula result is preserved as computed and only
    QA-checked for finiteness.

    wind_speed_m_s = sqrt(u10^2 + v10^2)

    precipitation_mm = total_precipitation_hourly * 1000
    (the HOURLY band -- the cumulative `total_precipitation` band is never
    used, so no differencing of a running total is involved.)

SPATIAL AGGREGATION
-------------------
Every hourly derived variable is reduced to ONE explicitly PIXEL-AREA-WEIGHTED
AOI mean:

    weighted_mean = sum(value * pixelArea) / sum(pixelArea masked by the
                    variable's own mask)

Both sums are reduced on the ERA5-Land NATIVE projection and transform, which
is probed once per run and recorded in the manifest. A bare
`ee.Reducer.mean()` is NOT area weighting and is never described as such here.

TEMPORAL AGGREGATION
--------------------
The statistics are computed over the SERIES OF HOURLY AREA-WEIGHTED AOI MEANS:

    temperature / RH / wind   window_mean, window_max
    precipitation             window_mean, window_max, window_total

"max" therefore means the MAXIMUM REGIONAL-HOUR CONDITION -- never the most
extreme individual pixel anywhere in the AOI.

Every window must be HOURLY COMPLETE: its timestamps must equal the exact
contiguous UTC hourly sequence [start 00:00, end_exclusive 00:00), i.e.
n_days_inclusive * 24 hours in order. A missing, duplicate, out-of-order,
shifted or extra hour fails closed rather than quietly biasing the means/maxima
and the precipitation window_total.

ARCHITECTURE
------------
Importing this module contacts nothing. Every Earth Engine symbol is imported
INSIDE a production-engine method; constructing the engine contacts nothing;
only `Era5LandRegionalEngine.initialise()` opens an Earth Engine session. The
calculation and orchestration accept an INJECTABLE engine, so tests exercise
the full pipeline with a fake and perform no network/GEE call. Only the
spatial reduction lives in the engine -- every unit conversion rule, temporal
statistic, climatology arithmetic and provenance decision is pure Python here
and is directly testable.

The engine returns the hourly series of already-spatially-reduced regional
means (a small array per variable), never pixels and never a raster.

OUTPUT
------
    outputs/diagnostics/era5_land_regional/<analysis_id>/
        era5_land_regional_summary.csv     one row per experiment (wide)
        era5_land_regional_summary.json    + the four climatology realizations
        scientific_contract.json           the hashed configuration
        manifest.json                      paths, byte sizes, SHA-256

`analysis_id` is the SHA-256 of the canonical JSON of the scientific
configuration, so an identical contract always resolves to the same namespace.
An existing COMPLETE namespace that passes manifest validation is reported as
already complete without recomputation; an existing INCOMPLETE or inconsistent
namespace fails closed and is never overwritten.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.regions import (  # noqa: E402
    VARIANT_STATUS_CANONICAL,
    get_experiment,
    validate_variant_record,
)

PROJECT_ROOT = _PROJECT_ROOT


class Era5LandRegionalDiagnosticError(Exception):
    """Fail-closed condition in the ERA5-Land regional diagnostic."""


# =============================================================================
# Frozen scientific contract
# =============================================================================
SCHEMA_VERSION = "era5_land_regional_diagnostic.v1"
OUTPUT_SCHEMA_VERSION = "era5_land_regional_summary.v1"
DIAGNOSTIC_NAMESPACE = "era5_land_regional"
DIAGNOSTIC_CLASS = "explanatory_aoi_level_regional_meteorology"

# The five frozen canonical AOIs, in the exact order hashed into analysis_id.
DEFAULT_EXPERIMENTS: tuple[str, ...] = (
    "manavgat_2021",
    "bejis_2022",
    "mugla_2021",
    "evia_2021_extended",
    "montiferru_2021",
)

# Registered experiments that are deliberately NOT in the default cohort, with
# the reason. An entry here is never silently added; it is reachable only by an
# explicit request, which changes the scientific config and therefore the
# analysis_id.
NON_DEFAULT_EXPERIMENTS: dict[str, str] = {
    "mugla_2022": (
        "provisional temporal contract awaiting supervisor decision; excluded "
        "from the default five-AOI regional meteorology cohort"
    ),
}

COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"
SOURCE_BANDS: tuple[str, ...] = (
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "total_precipitation_hourly",
)

CLIMATOLOGY_YEARS: tuple[int, ...] = (2017, 2018, 2019, 2020)
SD_DDOF = 1

WINDOW_KINDS: tuple[str, ...] = ("predictor", "label")
HOURS_PER_DAY = 24
MILLISECONDS_PER_HOUR = 3_600_000
HOURLY_COMPLETENESS_RULE = (
    "every window must contain the EXACT contiguous UTC hourly sequence "
    "[start 00:00, end_exclusive 00:00), in order, with "
    "expected_hours = n_days_inclusive * 24; missing, duplicate, out-of-order, "
    "shifted and extra hours fail closed and are never sorted, deduplicated, "
    "interpolated, padded or dropped"
)
TIMESTAMP_CONVENTION = (
    "Unix epoch milliseconds, UTC, matching Earth Engine system:time_start"
)
OBSERVED_WINDOW_SOURCE = "core.regions.EXPERIMENTS (registry)"
REGISTRY_END_DATE_CONVENTION = (
    "registry end dates are INCLUSIVE calendar days; the Earth Engine filter "
    "window is [start 00:00 UTC, end_inclusive + 1 day 00:00 UTC)"
)

# --- Unit conversions / derivation recipes ----------------------------------
KELVIN_OFFSET = 273.15                 # temperature_c = temperature_2m - 273.15
PRECIPITATION_M_TO_MM = 1000.0         # ERA5-Land precipitation is metres

# ECMWF/Tetens saturation vapour pressure over WATER (IFS documentation,
# Part IV, "Saturation vapour pressure"). The water phase is used for BOTH
# es(T) and es(Td); no ice-phase or mixed-phase branch is applied.
TETENS_T0_K = 273.16
TETENS_A1_PA = 611.21
TETENS_A3 = 17.502
TETENS_A4_K = 32.19
RELATIVE_HUMIDITY_RECIPE = (
    "RH = 100 * es(Td) / es(T) with "
    "es(T) = a1 * exp(a3 * (T - T0) / (T - a4)); "
    "ECMWF/Tetens saturation over water; derived per pixel per hour from "
    "temperature_2m and dewpoint_temperature_2m; NOT clipped to [0, 100]"
)
WIND_RECIPE = (
    "wind_speed_m_s = sqrt(u_component_of_wind_10m^2 + "
    "v_component_of_wind_10m^2); derived per pixel per hour"
)
PRECIPITATION_RECIPE = (
    "precipitation_mm = total_precipitation_hourly * 1000 (m -> mm of depth "
    "accumulated in that hour); the cumulative total_precipitation band is "
    "NEVER used"
)
TEMPERATURE_RECIPE = "temperature_c = temperature_2m - 273.15"

# --- Aggregation semantics --------------------------------------------------
SPATIAL_WEIGHTING = "explicit_pixel_area_weighted_regional_mean_v1"
SPATIAL_WEIGHTING_SEMANTICS = (
    "sum(value * ee.Image.pixelArea()) / sum(ee.Image.pixelArea() masked by "
    "that variable's own mask), reduced with ee.Reducer.sum() on the "
    "ERA5-Land native projection (crs + crsTransform, probed once per run). "
    "A bare ee.Reducer.mean() is NOT area weighting and is not used"
)
TEMPORAL_AGGREGATION = "statistics_over_hourly_area_weighted_regional_means_v1"
TEMPORAL_AGGREGATION_SEMANTICS = (
    "window statistics are computed over the SERIES OF HOURLY AREA-WEIGHTED "
    "AOI MEANS: mean = arithmetic mean of those hourly regional means, "
    "max = maximum regional-hour condition (never the most extreme individual "
    "pixel), total (precipitation only) = sum of the hourly regional means"
)

# variable -> statistics, in output order.
VARIABLES: dict[str, tuple[str, ...]] = {
    "temperature_c": ("mean", "max"),
    "relative_humidity_percent": ("mean", "max"),
    "wind_speed_m_s": ("mean", "max"),
    "precipitation_mm": ("mean", "max", "total"),
}
VARIABLE_UNITS: dict[str, str] = {
    "temperature_c": "degC",
    "relative_humidity_percent": "percent",
    "wind_speed_m_s": "m s-1",
    "precipitation_mm": "mm",
}
VARIABLE_SOURCE_BANDS: dict[str, tuple[str, ...]] = {
    "temperature_c": ("temperature_2m",),
    "relative_humidity_percent": ("temperature_2m", "dewpoint_temperature_2m"),
    "wind_speed_m_s": ("u_component_of_wind_10m", "v_component_of_wind_10m"),
    "precipitation_mm": ("total_precipitation_hourly",),
}

# Per-metric fields written for every (window, variable, statistic).
METRIC_FIELDS: tuple[str, ...] = (
    "observed",
    "climatology_mean",
    "climatology_sd",
    "anomaly",
    "standardized_anomaly",
    "zero_climatology_sd",
)

SUMMARY_CSV_NAME = "era5_land_regional_summary.csv"
SUMMARY_JSON_NAME = "era5_land_regional_summary.json"
CONTRACT_JSON_NAME = "scientific_contract.json"
MANIFEST_JSON_NAME = "manifest.json"
# manifest.json records the other three and is therefore not one of them.
PRODUCT_FILENAMES: tuple[str, ...] = (
    SUMMARY_CSV_NAME, SUMMARY_JSON_NAME, CONTRACT_JSON_NAME,
)
EXPECTED_FILENAMES: tuple[str, ...] = PRODUCT_FILENAMES + (MANIFEST_JSON_NAME,)


# =============================================================================
# Provenance helpers (canonical JSON / hashing)
# =============================================================================
def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except Exception:  # noqa: BLE001 -- absence of git is not a failure
        return None


# =============================================================================
# Cohort resolution
# =============================================================================
def resolve_experiments(experiments: Optional[Sequence[str]] = None) -> tuple[str, ...]:
    """Resolve the cohort, preserving caller order (default: the frozen five).

    Every ID must be registered, enabled and canonical. Nothing is silently
    dropped and nothing is silently added: requesting a non-default experiment
    is allowed but changes the scientific config, hence the analysis_id.
    """
    if experiments is None:
        requested = list(DEFAULT_EXPERIMENTS)
    else:
        requested = [str(e) for e in experiments]
    if not requested:
        raise Era5LandRegionalDiagnosticError(
            "No experiment requested; the cohort must contain at least one "
            f"registered experiment (default: {list(DEFAULT_EXPERIMENTS)})."
        )
    duplicates = sorted({e for e in requested if requested.count(e) > 1})
    if duplicates:
        raise Era5LandRegionalDiagnosticError(
            f"Duplicate experiment id(s) requested: {duplicates}. Each AOI "
            "contributes exactly one row."
        )

    for experiment_id in requested:
        record = get_experiment(experiment_id)  # ValueError if unknown
        if not record.get("enabled"):
            raise Era5LandRegionalDiagnosticError(
                f"'{experiment_id}' is registered but disabled; this diagnostic "
                "refuses to summarise a disabled experiment."
            )
        status = validate_variant_record(record, experiment_id)
        if status != VARIANT_STATUS_CANONICAL:
            raise Era5LandRegionalDiagnosticError(
                f"'{experiment_id}' has variant_status='{status}' "
                f"(superseded_by={record.get('superseded_by')!r}); only "
                f"'{VARIANT_STATUS_CANONICAL}' experiments may enter this "
                "diagnostic."
            )
    return tuple(requested)


_REGIONS_PY = _PROJECT_ROOT / "core" / "regions.py"
_BBOX_CACHE: dict[tuple[int, int], dict[str, tuple[float, float, float, float]]] = {}


def _literal_bbox(node: ast.expr) -> Optional[tuple[float, ...]]:
    """A 4-tuple/list of numeric literals, or None."""
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 4:
        return None
    values: list[float] = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, (int, float)):
            values.append(float(element.value))
        elif (isinstance(element, ast.UnaryOp) and isinstance(element.op, ast.USub)
              and isinstance(element.operand, ast.Constant)
              and isinstance(element.operand.value, (int, float))):
            values.append(-float(element.operand.value))
        else:
            return None
    return tuple(values)


def _is_bbox_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "BBox"
    )


def _resolve_bbox_expr(node: ast.expr, locals_: dict[str, ast.expr],
                       constants: dict[str, tuple[float, ...]],
                       depth: int = 0) -> Optional[tuple[float, ...]]:
    if depth > 8:
        return None
    if isinstance(node, ast.Name):
        target = locals_.get(node.id)
        return None if target is None else _resolve_bbox_expr(
            target, locals_, constants, depth + 1
        )
    if not _is_bbox_call(node):
        return None
    args = node.args
    # ee.Geometry.BBox(*MODULE_CONSTANT)
    if len(args) == 1 and isinstance(args[0], ast.Starred):
        starred = args[0].value
        if isinstance(starred, ast.Name):
            return constants.get(starred.id)
        return _literal_bbox(starred)
    # ee.Geometry.BBox(west, south, east, north)
    if len(args) == 4:
        return _literal_bbox(ast.Tuple(elts=list(args), ctx=ast.Load()))
    return None


def static_region_bboxes() -> dict[str, tuple[float, float, float, float]]:
    """region_key -> (west, south, east, north), read from source WITHOUT GEE.

    `core.regions.build_regions()` is deliberately NEVER called: building an
    `ee.Geometry` is what would force an Earth Engine session. The function is
    parsed instead (AST), following module-level bbox constants, simple
    `name = other_name` aliasing and `ee.Geometry.BBox(...)` literals -- true
    for every AOI in this diagnostic's cohort. Region keys whose AOI is built
    another way (e.g. Kozan's buffered point) are simply absent.

    This resolver is self-contained on purpose: the equivalent helper in
    `scripts/run_label_gate_only.py` lives in a module that opens a log file
    at import time, and this diagnostic's dry-run must create nothing at all.
    No geometry is defined here -- only read.
    """
    stat = _REGIONS_PY.stat()
    cache_key = (int(stat.st_mtime_ns), int(stat.st_size))
    cached = _BBOX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tree = ast.parse(_REGIONS_PY.read_text(encoding="utf-8"))
    constants: dict[str, tuple[float, ...]] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            literal = _literal_bbox(node.value)
            if literal is not None:
                constants[node.targets[0].id] = literal

    build_regions_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_regions":
            build_regions_fn = node
            break
    if build_regions_fn is None:
        raise Era5LandRegionalDiagnosticError(
            f"build_regions() not found in {_REGIONS_PY}; AOI bounds cannot be "
            "resolved without Earth Engine."
        )

    locals_: dict[str, ast.expr] = {}
    return_dict: Optional[ast.Dict] = None
    for node in ast.walk(build_regions_fn):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            locals_[node.targets[0].id] = node.value
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return_dict = node.value
    if return_dict is None:
        raise Era5LandRegionalDiagnosticError(
            f"build_regions() in {_REGIONS_PY} has no statically readable "
            "return dict; AOI bounds cannot be resolved."
        )

    resolved: dict[str, tuple[float, float, float, float]] = {}
    for key_node, value_node in zip(return_dict.keys, return_dict.values):
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            continue
        bounds = _resolve_bbox_expr(value_node, locals_, constants)
        if bounds is not None and len(bounds) == 4:
            resolved[key_node.value] = (
                float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]),
            )
    _BBOX_CACHE[cache_key] = resolved
    return resolved


def aoi_bbox(region_key: str) -> tuple[float, float, float, float]:
    """(west, south, east, north) of `region_key`, read WITHOUT Earth Engine."""
    bounds = static_region_bboxes().get(str(region_key))
    if bounds is None:
        raise Era5LandRegionalDiagnosticError(
            f"AOI bbox for region_key '{region_key}' could not be resolved "
            "statically from core/regions.py build_regions(). This diagnostic "
            "never guesses a geometry."
        )
    west, south, east, north = (float(v) for v in bounds)
    if not (west < east and south < north):
        raise Era5LandRegionalDiagnosticError(
            f"AOI bbox for '{region_key}' is degenerate: {bounds!r}."
        )
    return (west, south, east, north)


# =============================================================================
# Window resolution (registry -> inclusive -> exclusive)
# =============================================================================
def exclusive_end(end_date_inclusive: str) -> str:
    """Convert an INCLUSIVE registry end date to the EXCLUSIVE GEE filter end."""
    return (date.fromisoformat(str(end_date_inclusive)) + timedelta(days=1)).isoformat()


def _window(start: str, end_inclusive: str, label: str) -> dict[str, Any]:
    start_day = date.fromisoformat(str(start))
    end_day = date.fromisoformat(str(end_inclusive))
    if end_day < start_day:
        raise Era5LandRegionalDiagnosticError(
            f"{label}: end date {end_inclusive} precedes start date {start}."
        )
    return {
        "start_date": start_day.isoformat(),
        "end_date_inclusive": end_day.isoformat(),
        "end_date_exclusive": (end_day + timedelta(days=1)).isoformat(),
        "n_days_inclusive": int((end_day - start_day).days) + 1,
    }


def observed_windows(experiment_id: str) -> dict[str, dict[str, Any]]:
    """Predictor and label windows for `experiment_id`, straight from the registry."""
    record = get_experiment(experiment_id)
    windows: dict[str, dict[str, Any]] = {}
    for kind in WINDOW_KINDS:
        windows[kind] = _window(
            record[f"{kind}_start_date"],
            record[f"{kind}_end_date"],
            f"{experiment_id}.{kind}",
        )
    return windows


def _map_month_day(day: date, year: int) -> date:
    try:
        return date(int(year), day.month, day.day)
    except ValueError as exc:
        raise Era5LandRegionalDiagnosticError(
            f"Calendar date {day.month:02d}-{day.day:02d} cannot be represented "
            f"in reference year {year} ({exc}). This diagnostic refuses to "
            "shift, clamp or drop a climatology window date; the frozen five "
            "AOIs contain no 29 February window bound, so a real occurrence "
            "requires an explicit scientific decision."
        ) from exc


def map_window_to_year(start: str, end_inclusive: str, year: int) -> dict[str, Any]:
    """Map a window's exact month/day bounds into one reference year."""
    start_day = date.fromisoformat(str(start))
    end_day = date.fromisoformat(str(end_inclusive))
    mapped_start = _map_month_day(start_day, year)
    mapped_end = _map_month_day(end_day, year)
    if mapped_end < mapped_start:
        raise Era5LandRegionalDiagnosticError(
            f"Window {start}..{end_inclusive} maps to {mapped_start}..{mapped_end} "
            f"in reference year {year}, i.e. it crosses a calendar-year "
            "boundary. Such a window has no unambiguous single-year "
            "realization and is refused."
        )
    return {
        "reference_year": int(year),
        "start_date": mapped_start.isoformat(),
        "end_date_inclusive": mapped_end.isoformat(),
        "end_date_exclusive": (mapped_end + timedelta(days=1)).isoformat(),
        "n_days_inclusive": int((mapped_end - mapped_start).days) + 1,
    }


def climatology_windows(
    start: str, end_inclusive: str, years: Sequence[int] = CLIMATOLOGY_YEARS,
) -> list[dict[str, Any]]:
    return [map_window_to_year(start, end_inclusive, int(y)) for y in years]


# =============================================================================
# Pure derivation formulas (identical semantics to the server-side recipe)
# =============================================================================
def saturation_vapour_pressure_pa(temperature_k: float) -> float:
    """ECMWF/Tetens saturation vapour pressure over water, in Pa."""
    temperature_k = float(temperature_k)
    denominator = temperature_k - TETENS_A4_K
    if denominator == 0.0:
        raise Era5LandRegionalDiagnosticError(
            f"Saturation vapour pressure is undefined at T = {temperature_k} K "
            f"(T - a4 == 0 with a4 = {TETENS_A4_K})."
        )
    return TETENS_A1_PA * math.exp(
        TETENS_A3 * (temperature_k - TETENS_T0_K) / denominator
    )


def relative_humidity_percent(temperature_k: float, dewpoint_k: float) -> float:
    """RH = 100 * es(Td) / es(T). Not clipped."""
    return 100.0 * (
        saturation_vapour_pressure_pa(float(dewpoint_k))
        / saturation_vapour_pressure_pa(float(temperature_k))
    )


def temperature_celsius(temperature_k: float) -> float:
    return float(temperature_k) - KELVIN_OFFSET


def wind_speed_m_s(u_component: float, v_component: float) -> float:
    return math.sqrt(float(u_component) ** 2 + float(v_component) ** 2)


def precipitation_mm_from_m(depth_m: float) -> float:
    return float(depth_m) * PRECIPITATION_M_TO_MM


# =============================================================================
# Temporal window statistics over hourly area-weighted regional means
# =============================================================================
def window_statistics(series: Sequence[float], statistics: Sequence[str]) -> dict[str, float]:
    values = [float(v) for v in series]
    if not values:
        raise Era5LandRegionalDiagnosticError(
            "Empty hourly series: a window statistic cannot be computed from "
            "zero hours."
        )
    out: dict[str, float] = {}
    for statistic in statistics:
        if statistic == "mean":
            out["mean"] = math.fsum(values) / float(len(values))
        elif statistic == "max":
            out["max"] = max(values)
        elif statistic == "total":
            out["total"] = math.fsum(values)
        else:  # pragma: no cover -- guarded by the frozen VARIABLES table
            raise Era5LandRegionalDiagnosticError(
                f"Unknown temporal statistic '{statistic}'."
            )
    return out


def window_statistics_from_series(series_by_variable: dict[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    """All window statistics for one (experiment, window, realization)."""
    return {
        variable: window_statistics(series_by_variable[variable], statistics)
        for variable, statistics in VARIABLES.items()
    }


def sample_standard_deviation(values: Sequence[float], ddof: int = SD_DDOF) -> float:
    """Sample SD with ddof=1 (never the population SD)."""
    numbers = [float(v) for v in values]
    n = len(numbers)
    if n - ddof <= 0:
        raise Era5LandRegionalDiagnosticError(
            f"Sample standard deviation with ddof={ddof} needs more than "
            f"{ddof} value(s); got {n}."
        )
    mean = math.fsum(numbers) / float(n)
    variance = math.fsum((v - mean) ** 2 for v in numbers) / float(n - ddof)
    return math.sqrt(variance)


def climatology_statistics(
    observed: float, realizations: dict[int, float],
    years: Sequence[int] = CLIMATOLOGY_YEARS,
) -> dict[str, Any]:
    """Combine one observed statistic with its four yearly reference values.

    The yearly window statistics are computed independently upstream; this
    function only performs the frozen arithmetic. A zero climatology SD yields
    `standardized_anomaly=None` -- never inf, never a substituted 0.
    """
    missing = [int(y) for y in years if int(y) not in realizations]
    if missing:
        raise Era5LandRegionalDiagnosticError(
            f"Climatology realizations missing for reference year(s) {missing}; "
            f"all of {list(years)} are required."
        )
    ordered = [float(realizations[int(y)]) for y in years]
    for year, value in zip(years, ordered):
        if not math.isfinite(value):
            raise Era5LandRegionalDiagnosticError(
                f"Climatology realization for {year} is not finite ({value!r})."
            )
    observed = float(observed)
    if not math.isfinite(observed):
        raise Era5LandRegionalDiagnosticError(
            f"Observed statistic is not finite ({observed!r})."
        )

    mean = math.fsum(ordered) / float(len(ordered))
    sd = sample_standard_deviation(ordered)
    anomaly = observed - mean
    zero_sd = sd == 0.0
    standardized = None if zero_sd else anomaly / sd
    if standardized is not None and not math.isfinite(standardized):
        raise Era5LandRegionalDiagnosticError(
            "Standardized anomaly is not finite although the climatology SD is "
            f"non-zero (anomaly={anomaly!r}, sd={sd!r})."
        )
    return {
        "observed": observed,
        "climatology_realizations": {str(int(y)): float(v) for y, v in zip(years, ordered)},
        "climatology_mean": mean,
        "climatology_sd": sd,
        "anomaly": anomaly,
        "standardized_anomaly": standardized,
        "zero_climatology_sd": bool(zero_sd),
    }


# =============================================================================
# Engine contract
# =============================================================================
#   name                  : str, recorded in the manifest
#   contacts_earth_engine : bool
#   initialise()          : the ONLY call that may open an Earth Engine session
#   source_grid(collection_id=...) -> dict with crs / crs_transform /
#                           nominal_scale_m, recorded in the manifest
#   hourly_regional_series(collection_id=, bands=, bbox=, start=,
#                          end_exclusive=, experiment_id=, window=,
#                          realization=) -> dict of hourly SERIES:
#                             {"timestamps": [...], "<variable>": [...], ...}
#                          already spatially reduced to pixel-area-weighted
#                          AOI means; no pixels and no raster ever leave it.
# =============================================================================
def expected_hourly_timestamps_ms(start: str, end_exclusive: str) -> list[int]:
    """The exact contiguous UTC hourly sequence for [start 00:00, end 00:00).

    Unix epoch MILLISECONDS, matching Earth Engine's `system:time_start`. The
    length is always `(end_exclusive - start).days * 24` -- UTC has no DST, so
    every day contributes exactly 24 hours.
    """
    start_day = date.fromisoformat(str(start))
    end_day = date.fromisoformat(str(end_exclusive))
    if end_day <= start_day:
        raise Era5LandRegionalDiagnosticError(
            f"Exclusive end {end_exclusive} does not follow start {start}; an "
            "hourly window must cover at least one day."
        )
    base_ms = int(
        datetime(start_day.year, start_day.month, start_day.day,
                 tzinfo=timezone.utc).timestamp() * 1000
    )
    n_hours = int((end_day - start_day).days) * HOURS_PER_DAY
    return [base_ms + index * MILLISECONDS_PER_HOUR for index in range(n_hours)]


def validate_hourly_series(
    payload: Any, *, start: str, end_exclusive: str, context: str,
) -> dict[str, list[float]]:
    """QA an engine response before any statistic is computed.

    Beyond finiteness, the window must be HOURLY COMPLETE: the payload's
    timestamps must equal the exact contiguous UTC hourly sequence expected for
    the requested window, in order, with no gap, duplicate, shift or extra
    hour. A silently short or misaligned window would bias every mean/max and
    would corrupt the precipitation window_total outright, so it fails closed.
    Nothing is sorted, deduplicated, interpolated, padded or dropped here.
    """
    if not isinstance(payload, dict):
        raise Era5LandRegionalDiagnosticError(
            f"{context}: engine returned {type(payload).__name__}, expected a dict "
            "of hourly series."
        )

    expected_timestamps = expected_hourly_timestamps_ms(start, end_exclusive)
    expected_hours = len(expected_timestamps)

    if "timestamps" not in payload:
        raise Era5LandRegionalDiagnosticError(
            f"{context}: engine response has no 'timestamps' -- hourly "
            f"completeness for [{start}, {end_exclusive}) cannot be proven "
            f"(keys: {sorted(payload)})."
        )
    raw_timestamps = payload["timestamps"]
    if not isinstance(raw_timestamps, (list, tuple)):
        raise Era5LandRegionalDiagnosticError(
            f"{context}: 'timestamps' is {type(raw_timestamps).__name__}, "
            "expected a list of epoch-millisecond values."
        )
    if len(raw_timestamps) != expected_hours:
        raise Era5LandRegionalDiagnosticError(
            f"{context}: window [{start}, {end_exclusive}) returned "
            f"{len(raw_timestamps)} hour(s); exactly {expected_hours} are "
            f"required (n_days * {HOURS_PER_DAY}). A short or long window is "
            "never accepted, padded or trimmed."
        )

    timestamps: list[int] = []
    for index, item in enumerate(raw_timestamps):
        if isinstance(item, bool) or item is None:
            raise Era5LandRegionalDiagnosticError(
                f"{context}: timestamp {index} is {item!r}, expected an integer "
                "epoch-millisecond value."
            )
        if isinstance(item, int):
            value = int(item)
        elif isinstance(item, float):
            if not math.isfinite(item) or not float(item).is_integer():
                raise Era5LandRegionalDiagnosticError(
                    f"{context}: timestamp {index} is {item!r}; a finite integer "
                    "number of epoch milliseconds is required."
                )
            value = int(item)
        else:
            raise Era5LandRegionalDiagnosticError(
                f"{context}: timestamp {index} is {type(item).__name__} "
                f"({item!r}), expected a numeric epoch-millisecond value."
            )
        timestamps.append(value)

    if timestamps != expected_timestamps:
        first_bad = next(
            (i for i, (got, want) in enumerate(zip(timestamps, expected_timestamps))
             if got != want),
            0,
        )
        duplicates = len(timestamps) != len(set(timestamps))
        raise Era5LandRegionalDiagnosticError(
            f"{context}: the hourly timestamps are not the exact contiguous UTC "
            f"sequence expected for [{start}, {end_exclusive}). First mismatch "
            f"at position {first_bad}: got {timestamps[first_bad]}, expected "
            f"{expected_timestamps[first_bad]}"
            + (" (the series also contains duplicate hours)" if duplicates else "")
            + ". Missing, duplicate, out-of-order, shifted and extra hours all "
            "fail closed; the series is never reordered or repaired."
        )

    series: dict[str, list[float]] = {}
    lengths: dict[str, int] = {}
    for variable in VARIABLES:
        if variable not in payload:
            raise Era5LandRegionalDiagnosticError(
                f"{context}: engine response has no '{variable}' series "
                f"(keys: {sorted(payload)})."
            )
        raw = payload[variable]
        if not isinstance(raw, (list, tuple)):
            raise Era5LandRegionalDiagnosticError(
                f"{context}: '{variable}' series is {type(raw).__name__}, expected a list."
            )
        values: list[float] = []
        for index, item in enumerate(raw):
            if item is None:
                raise Era5LandRegionalDiagnosticError(
                    f"{context}: '{variable}' hour {index} is null -- a masked or "
                    "missing regional mean is never treated as a value."
                )
            try:
                value = float(item)
            except (TypeError, ValueError) as exc:
                raise Era5LandRegionalDiagnosticError(
                    f"{context}: '{variable}' hour {index} is not numeric ({item!r})."
                ) from exc
            if not math.isfinite(value):
                raise Era5LandRegionalDiagnosticError(
                    f"{context}: '{variable}' hour {index} is not finite ({value!r}); "
                    "inf/NaN never enters a window statistic."
                )
            values.append(value)
        if not values:
            raise Era5LandRegionalDiagnosticError(
                f"{context}: '{variable}' series is empty -- the window resolved "
                "to zero hourly images."
            )
        series[variable] = values
        lengths[variable] = len(values)

    distinct = sorted(set(lengths.values()))
    if len(distinct) != 1:
        raise Era5LandRegionalDiagnosticError(
            f"{context}: hourly series lengths disagree across variables "
            f"({lengths}); every variable must cover the same hours."
        )
    if distinct[0] != expected_hours:
        raise Era5LandRegionalDiagnosticError(
            f"{context}: variable series carry {distinct[0]} hour(s) but the "
            f"window's timestamps carry {expected_hours}; every variable must "
            "be aligned to the same complete hourly sequence."
        )
    return series


class Era5LandRegionalEngine:
    """Production Earth Engine engine for the ERA5-Land regional diagnostic.

    Every Earth Engine symbol is imported INSIDE a method, so importing this
    module costs nothing and needs no credentials. Construction contacts
    nothing; only `initialise()` opens a session. The engine performs the
    SPATIAL reduction only -- unit conversions are applied per pixel per hour
    server-side under the same frozen recipes documented above, and every
    temporal/climatology statistic is computed in pure Python by the caller.
    """

    name = "era5_land_regional_production_v1"
    contacts_earth_engine = True

    def __init__(self, project: Optional[str] = None, max_pixels: float = 1e13,
                 tile_scale: float = 4.0) -> None:
        self.project = project
        self.max_pixels = float(max_pixels)
        self.tile_scale = float(tile_scale)
        self._initialised = False
        self._grid_cache: dict[str, dict[str, Any]] = {}

    # -- session -------------------------------------------------------------
    def initialise(self) -> dict[str, Any]:
        from core.gee_utils import init_gee

        if self.project is not None:
            init_gee(self.project)
        else:
            init_gee()
        self._initialised = True
        return {"initialised": True, "project": self.project}

    # -- native grid ---------------------------------------------------------
    def source_grid(self, collection_id: str = COLLECTION_ID) -> dict[str, Any]:
        """Probe the ERA5-Land native projection ONCE (single band, cached).

        A multi-band `first().projection()` has no single well-defined
        projection, so the temperature band is used as the canonical grid
        source; all five bands share the ERA5-Land grid.
        """
        import ee

        if collection_id in self._grid_cache:
            return self._grid_cache[collection_id]

        projection = (
            ee.Image(ee.ImageCollection(collection_id).first())
            .select(SOURCE_BANDS[0])
            .projection()
        )
        info = projection.getInfo() or {}
        crs = info.get("crs")
        transform = info.get("transform")
        nominal_scale = projection.nominalScale().getInfo()
        if not (isinstance(crs, str) and crs.strip()):
            raise Era5LandRegionalDiagnosticError(
                f"{collection_id} reported no CRS authority code "
                f"({info!r}); the reduction projection must be deterministic "
                "and is never assumed."
            )
        if not (isinstance(transform, (list, tuple)) and len(transform) == 6
                and all(isinstance(v, (int, float)) and math.isfinite(float(v))
                        for v in transform)):
            raise Era5LandRegionalDiagnosticError(
                f"{collection_id} reported an unusable native transform "
                f"({transform!r}); exactly six finite values are required."
            )
        grid = {
            "collection_id": collection_id,
            "canonical_projection_band": SOURCE_BANDS[0],
            "crs": crs.strip(),
            "crs_transform": [float(v) for v in transform],
            "nominal_scale_m": float(nominal_scale),
            "projection_read_method": "ee_single_band_projection_getinfo_v1",
        }
        self._grid_cache[collection_id] = grid
        return grid

    # -- derived image -------------------------------------------------------
    def _derived_image(self, image):
        """Per-pixel, per-hour derived variables (never from window means)."""
        import ee

        temperature_k = image.select("temperature_2m")
        dewpoint_k = image.select("dewpoint_temperature_2m")
        u10 = image.select("u_component_of_wind_10m")
        v10 = image.select("v_component_of_wind_10m")
        precipitation_m = image.select("total_precipitation_hourly")

        temperature_c = temperature_k.subtract(KELVIN_OFFSET).rename("temperature_c")

        # es(T) = a1 * exp(a3 * (T - T0) / (T - a4)); the a1 factor is written
        # explicitly on both sides rather than cancelled, so the code matches
        # the documented recipe literally.
        es_air = (
            temperature_k.subtract(TETENS_T0_K)
            .divide(temperature_k.subtract(TETENS_A4_K))
            .multiply(TETENS_A3)
            .exp()
            .multiply(TETENS_A1_PA)
        )
        es_dew = (
            dewpoint_k.subtract(TETENS_T0_K)
            .divide(dewpoint_k.subtract(TETENS_A4_K))
            .multiply(TETENS_A3)
            .exp()
            .multiply(TETENS_A1_PA)
        )
        # NOT clipped: the formula result is preserved as computed.
        relative_humidity = es_dew.divide(es_air).multiply(100.0).rename(
            "relative_humidity_percent"
        )

        wind_speed = (
            u10.pow(2).add(v10.pow(2)).sqrt().rename("wind_speed_m_s")
        )
        precipitation_mm = precipitation_m.multiply(PRECIPITATION_M_TO_MM).rename(
            "precipitation_mm"
        )

        return ee.Image.cat(
            temperature_c, relative_humidity, wind_speed, precipitation_mm
        ).copyProperties(image, ["system:time_start"])

    # -- explicit pixel-area-weighted regional mean --------------------------
    def _weighted_regional_means(self, derived, region, grid: dict[str, Any]):
        """sum(value * pixelArea) / sum(pixelArea masked by the value's mask).

        The denominator is masked by EACH variable's own mask, so a variable
        that is undefined over part of the AOI is not credited with that area.
        This is an explicit area weighting -- not ee.Reducer.mean().
        """
        import ee

        pixel_area = ee.Image.pixelArea()
        numerators = []
        denominators = []
        for variable in VARIABLES:
            band = derived.select(variable)
            numerators.append(band.multiply(pixel_area).rename(f"num_{variable}"))
            denominators.append(
                pixel_area.updateMask(band.mask()).rename(f"den_{variable}")
            )
        stacked = ee.Image.cat(numerators + denominators)
        sums = stacked.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            crs=grid["crs"],
            crsTransform=grid["crs_transform"],
            bestEffort=False,
            maxPixels=self.max_pixels,
            tileScale=self.tile_scale,
        )
        return ee.Dictionary({
            variable: ee.Number(sums.get(f"num_{variable}")).divide(
                ee.Number(sums.get(f"den_{variable}"))
            )
            for variable in VARIABLES
        })

    def region(self, bbox: Sequence[float]):
        import ee

        west, south, east, north = (float(v) for v in bbox)
        return ee.Geometry.BBox(west, south, east, north)

    # -- hourly series -------------------------------------------------------
    def hourly_regional_series(
        self, *, collection_id: str, bands: Sequence[str],
        bbox: Sequence[float], start: str, end_exclusive: str,
        experiment_id: str = "", window: str = "", realization: str = "",
    ) -> dict[str, Any]:
        """One `getInfo()` per window: the hourly area-weighted AOI means.

        Only already-spatially-reduced numbers cross the wire -- four short
        arrays plus their timestamps. No image, tile or raster is downloaded
        and nothing is exported.
        """
        import ee

        if not self._initialised:
            raise Era5LandRegionalDiagnosticError(
                "Earth Engine session not initialised; call initialise() before "
                "requesting any series."
            )
        grid = self.source_grid(collection_id)
        region = self.region(bbox)
        collection = (
            ee.ImageCollection(collection_id)
            .select(list(bands))
            .filterDate(str(start), str(end_exclusive))
            .filterBounds(region)
            .sort("system:time_start")
        )

        def _feature(image):
            image = ee.Image(image)
            means = self._weighted_regional_means(
                ee.Image(self._derived_image(image)), region, grid
            )
            return ee.Feature(None, means.set(
                "timestamp", image.get("system:time_start")
            ))

        features = ee.FeatureCollection(collection.map(_feature))
        payload = ee.Dictionary({
            "timestamps": features.aggregate_array("timestamp"),
            **{variable: features.aggregate_array(variable) for variable in VARIABLES},
        }).getInfo()
        payload = dict(payload or {})
        payload["n_hours"] = len(payload.get("timestamps") or [])
        payload["source_grid"] = grid
        return payload


# =============================================================================
# Scientific configuration / analysis_id
# =============================================================================
def build_scientific_config(experiment_ids: Sequence[str]) -> dict[str, Any]:
    """The exact configuration hashed into analysis_id.

    Everything that changes a number must appear here: cohort and order,
    source, bands, window provenance, climatology years, unit conversions,
    the full RH/wind/precipitation recipes with constants, spatial weighting
    semantics, temporal aggregation semantics, the ddof convention and the
    output schema version.
    """
    experiment_ids = tuple(str(e) for e in experiment_ids)
    windows: dict[str, Any] = {}
    for experiment_id in experiment_ids:
        record = get_experiment(experiment_id)
        observed = observed_windows(experiment_id)
        windows[experiment_id] = {
            "region_key": record["region_key"],
            "aoi_bbox_west_south_east_north": list(aoi_bbox(record["region_key"])),
            "observed": observed,
            "climatology": {
                kind: climatology_windows(
                    observed[kind]["start_date"], observed[kind]["end_date_inclusive"],
                )
                for kind in WINDOW_KINDS
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "diagnostic_namespace": DIAGNOSTIC_NAMESPACE,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "is_model_predictor": False,
        "pipeline_membership": "standalone_diagnostic_outside_step5_step7_step8_step9_step10",
        "experiment_ids": list(experiment_ids),
        "experiment_order_is_significant": True,
        "excluded_by_default": dict(NON_DEFAULT_EXPERIMENTS),
        "source": {
            "collection": COLLECTION_ID,
            "bands": list(SOURCE_BANDS),
            "raster_export_performed": False,
        },
        "observed_window_source": OBSERVED_WINDOW_SOURCE,
        "registry_end_date_convention": REGISTRY_END_DATE_CONVENTION,
        "windows": windows,
        "climatology": {
            "reference_years": list(CLIMATOLOGY_YEARS),
            "window_mapping": (
                "the observed window's exact month/day start and end are mapped "
                "into each reference year; a month/day that cannot be "
                "represented fails closed"
            ),
            "per_year_statistics_computed_independently": True,
            "mean": "arithmetic mean of the four yearly window statistics",
            "sd_convention": f"sample standard deviation, ddof={SD_DDOF}",
            "sd_ddof": SD_DDOF,
            "anomaly": "observed - climatology_mean",
            "standardized_anomaly": "anomaly / climatology_sd",
            "zero_sd_policy": (
                "standardized_anomaly is null and zero_climatology_sd is true; "
                "never inf and never substituted with 0"
            ),
        },
        "derived_variables": {
            "temperature_c": {
                "units": VARIABLE_UNITS["temperature_c"],
                "source_bands": list(VARIABLE_SOURCE_BANDS["temperature_c"]),
                "recipe": TEMPERATURE_RECIPE,
                "kelvin_offset": KELVIN_OFFSET,
            },
            "relative_humidity_percent": {
                "units": VARIABLE_UNITS["relative_humidity_percent"],
                "source_bands": list(VARIABLE_SOURCE_BANDS["relative_humidity_percent"]),
                "recipe": RELATIVE_HUMIDITY_RECIPE,
                "saturation_phase": "water",
                "constants": {
                    "T0_K": TETENS_T0_K, "a1_Pa": TETENS_A1_PA,
                    "a3": TETENS_A3, "a4_K": TETENS_A4_K,
                },
                "clipped": False,
                "derivation_level": "per_pixel_per_hour",
            },
            "wind_speed_m_s": {
                "units": VARIABLE_UNITS["wind_speed_m_s"],
                "source_bands": list(VARIABLE_SOURCE_BANDS["wind_speed_m_s"]),
                "recipe": WIND_RECIPE,
                "derivation_level": "per_pixel_per_hour",
            },
            "precipitation_mm": {
                "units": VARIABLE_UNITS["precipitation_mm"],
                "source_bands": list(VARIABLE_SOURCE_BANDS["precipitation_mm"]),
                "recipe": PRECIPITATION_RECIPE,
                "m_to_mm": PRECIPITATION_M_TO_MM,
                "cumulative_band_used": False,
            },
        },
        "spatial_aggregation": {
            "method": SPATIAL_WEIGHTING,
            "semantics": SPATIAL_WEIGHTING_SEMANTICS,
            "masks_respected": True,
            "projection": "ERA5-Land native crs + crsTransform, probed per run",
        },
        "temporal_aggregation": {
            "method": TEMPORAL_AGGREGATION,
            "semantics": TEMPORAL_AGGREGATION_SEMANTICS,
            "statistics": {variable: list(stats) for variable, stats in VARIABLES.items()},
            "max_meaning": "maximum regional-hour condition, not a per-pixel extreme",
            "hourly_completeness_rule": HOURLY_COMPLETENESS_RULE,
            "expected_hours_formula": f"n_days_inclusive * {HOURS_PER_DAY}",
            "timestamp_convention": TIMESTAMP_CONVENTION,
            "incomplete_window_policy": "fail_closed",
        },
        "output": {
            "namespace": f"outputs/diagnostics/{DIAGNOSTIC_NAMESPACE}/<analysis_id>/",
            "files": list(EXPECTED_FILENAMES),
            "rows": "exactly one per experiment",
            "csv_layout": "wide: predictor_* and label_* statistics in each AOI row",
        },
    }


def compute_analysis_id(scientific_config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(scientific_config).encode("utf-8"))


def diagnostics_root(output_root: Optional[Path] = None) -> Path:
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    return root / "diagnostics" / DIAGNOSTIC_NAMESPACE


def analysis_root(analysis_id: str, output_root: Optional[Path] = None) -> Path:
    return diagnostics_root(output_root) / str(analysis_id)


def planned_output_paths(analysis_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    root = analysis_root(analysis_id, output_root)
    return {name: root / name for name in EXPECTED_FILENAMES}


# =============================================================================
# Output schema
# =============================================================================
def metric_key(window: str, variable: str, statistic: str, field: str) -> str:
    return f"{window}_{variable}_{statistic}_{field}"


def summary_columns() -> list[str]:
    """The exact, ordered wide CSV schema."""
    columns = [
        "experiment_id", "region_key", "display_name", "country", "role",
        "aoi_bbox_west", "aoi_bbox_south", "aoi_bbox_east", "aoi_bbox_north",
    ]
    for window in WINDOW_KINDS:
        columns += [
            f"{window}_start_date",
            f"{window}_end_date_inclusive",
            f"{window}_end_date_exclusive",
            f"{window}_n_days_inclusive",
            f"{window}_n_hours_observed",
        ]
        for variable, statistics in VARIABLES.items():
            for statistic in statistics:
                for field in METRIC_FIELDS:
                    columns.append(metric_key(window, variable, statistic, field))
    return columns


# =============================================================================
# Planning (dry-run) -- creates nothing, contacts nothing
# =============================================================================
def build_plan(
    experiments: Optional[Sequence[str]] = None,
    output_root: Optional[Path] = None,
) -> dict[str, Any]:
    experiment_ids = resolve_experiments(experiments)
    scientific_config = build_scientific_config(experiment_ids)
    analysis_id = compute_analysis_id(scientific_config)
    paths = planned_output_paths(analysis_id, output_root)

    per_experiment = []
    for experiment_id in experiment_ids:
        record = get_experiment(experiment_id)
        observed = observed_windows(experiment_id)
        per_experiment.append({
            "experiment_id": experiment_id,
            "display_name": record["display_name"],
            "region_key": record["region_key"],
            "aoi_bbox_west_south_east_north": list(aoi_bbox(record["region_key"])),
            "observed_windows": observed,
            "climatology_windows": {
                kind: climatology_windows(
                    observed[kind]["start_date"], observed[kind]["end_date_inclusive"],
                )
                for kind in WINDOW_KINDS
            },
            "planned_gee_filter_windows": {
                kind: {
                    "observed": [observed[kind]["start_date"],
                                 observed[kind]["end_date_exclusive"]],
                    "climatology": [
                        [w["start_date"], w["end_date_exclusive"]]
                        for w in climatology_windows(
                            observed[kind]["start_date"],
                            observed[kind]["end_date_inclusive"],
                        )
                    ],
                }
                for kind in WINDOW_KINDS
            },
        })

    n_windows = len(experiment_ids) * len(WINDOW_KINDS) * (1 + len(CLIMATOLOGY_YEARS))
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "analysis_id": analysis_id,
        "experiment_ids": list(experiment_ids),
        "n_experiments": len(experiment_ids),
        "n_rows_expected": len(experiment_ids),
        "n_engine_window_requests": n_windows,
        "collection": COLLECTION_ID,
        "bands": list(SOURCE_BANDS),
        "climatology_years": list(CLIMATOLOGY_YEARS),
        "experiments": per_experiment,
        "scientific_config": scientific_config,
        "planned_output_paths": {name: str(path) for name, path in paths.items()},
        "planned_output_root": str(analysis_root(analysis_id, output_root)),
        "summary_columns": summary_columns(),
    }


# =============================================================================
# Namespace state (resume / fail-closed)
# =============================================================================
def namespace_state(analysis_id: str, output_root: Optional[Path] = None) -> dict[str, Any]:
    """Classify an existing namespace: absent / complete / incomplete.

    `complete` requires: all four files present, a parseable manifest whose
    analysis_id matches, a recorded file set equal to the expected one, and
    every recorded byte size and SHA-256 matching the file on disk.
    """
    root = analysis_root(analysis_id, output_root)
    if not root.exists():
        return {"state": "absent", "root": str(root), "reason": "namespace does not exist"}
    if not root.is_dir():
        return {"state": "incomplete", "root": str(root),
                "reason": f"{root} exists but is not a directory"}

    present = sorted(p.name for p in root.iterdir() if p.is_file())
    missing = [name for name in EXPECTED_FILENAMES if not (root / name).is_file()]
    if missing:
        return {
            "state": "incomplete", "root": str(root), "present_files": present,
            "reason": f"missing expected file(s): {missing}",
        }

    try:
        manifest = json.loads((root / MANIFEST_JSON_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"state": "incomplete", "root": str(root),
                "reason": f"manifest.json is unreadable/invalid: {exc}"}

    if manifest.get("analysis_id") != str(analysis_id):
        return {
            "state": "incomplete", "root": str(root),
            "reason": (
                f"manifest analysis_id {manifest.get('analysis_id')!r} does not "
                f"match namespace {analysis_id!r}"
            ),
        }

    recorded = manifest.get("files")
    if not isinstance(recorded, list) or not recorded:
        return {"state": "incomplete", "root": str(root),
                "reason": "manifest records no files"}
    recorded_names = sorted(str(entry.get("name")) for entry in recorded)
    if recorded_names != sorted(PRODUCT_FILENAMES):
        return {
            "state": "incomplete", "root": str(root),
            "reason": (
                f"manifest file set {recorded_names} != expected "
                f"{sorted(PRODUCT_FILENAMES)}"
            ),
        }

    for entry in recorded:
        path = root / str(entry.get("name"))
        if not path.is_file():
            return {"state": "incomplete", "root": str(root),
                    "reason": f"manifest records missing file {path.name}"}
        size = int(path.stat().st_size)
        if size != int(entry.get("bytes", -1)):
            return {
                "state": "incomplete", "root": str(root),
                "reason": (
                    f"{path.name}: {size} bytes on disk, manifest records "
                    f"{entry.get('bytes')!r}"
                ),
            }
        digest = sha256_file(path)
        if digest != entry.get("sha256"):
            return {
                "state": "incomplete", "root": str(root),
                "reason": f"{path.name}: SHA-256 mismatch against the manifest",
            }

    return {"state": "complete", "root": str(root), "manifest": manifest}


# =============================================================================
# Computation
# =============================================================================
def compute_experiment_row(
    experiment_id: str, engine: Any, *, collection_id: str = COLLECTION_ID,
    bands: Sequence[str] = SOURCE_BANDS,
    climatology_years: Sequence[int] = CLIMATOLOGY_YEARS,
) -> dict[str, Any]:
    """All predictor/label statistics + climatology arithmetic for ONE AOI."""
    record = get_experiment(experiment_id)
    bbox = aoi_bbox(record["region_key"])
    observed = observed_windows(experiment_id)

    detail: dict[str, Any] = {
        "experiment_id": experiment_id,
        "region_key": record["region_key"],
        "display_name": record["display_name"],
        "country": record["country"],
        "role": record["role"],
        "aoi_bbox_west_south_east_north": list(bbox),
        "windows": {},
    }

    for kind in WINDOW_KINDS:
        window = observed[kind]
        observed_payload = engine.hourly_regional_series(
            collection_id=collection_id, bands=list(bands), bbox=list(bbox),
            start=window["start_date"], end_exclusive=window["end_date_exclusive"],
            experiment_id=experiment_id, window=kind, realization="observed",
        )
        observed_series = validate_hourly_series(
            observed_payload,
            start=window["start_date"], end_exclusive=window["end_date_exclusive"],
            context=f"{experiment_id}/{kind}/observed",
        )
        observed_stats = window_statistics_from_series(observed_series)

        mapped = climatology_windows(
            window["start_date"], window["end_date_inclusive"], climatology_years,
        )
        yearly_stats: dict[int, dict[str, dict[str, float]]] = {}
        yearly_hours: dict[int, int] = {}
        for mapped_window in mapped:
            year = int(mapped_window["reference_year"])
            payload = engine.hourly_regional_series(
                collection_id=collection_id, bands=list(bands), bbox=list(bbox),
                start=mapped_window["start_date"],
                end_exclusive=mapped_window["end_date_exclusive"],
                experiment_id=experiment_id, window=kind,
                realization=f"climatology_{year}",
            )
            series = validate_hourly_series(
                payload,
                start=mapped_window["start_date"],
                end_exclusive=mapped_window["end_date_exclusive"],
                context=f"{experiment_id}/{kind}/climatology_{year}",
            )
            yearly_stats[year] = window_statistics_from_series(series)
            yearly_hours[year] = len(next(iter(series.values())))

        metrics: dict[str, Any] = {}
        for variable, statistics in VARIABLES.items():
            for statistic in statistics:
                metrics[f"{variable}_{statistic}"] = climatology_statistics(
                    observed_stats[variable][statistic],
                    {year: yearly_stats[year][variable][statistic]
                     for year in yearly_stats},
                    climatology_years,
                )

        detail["windows"][kind] = {
            **window,
            "n_hours_observed": len(next(iter(observed_series.values()))),
            "climatology_windows": mapped,
            "climatology_n_hours": {str(y): int(n) for y, n in yearly_hours.items()},
            "metrics": metrics,
        }
    return detail


def summary_row(detail: dict[str, Any]) -> dict[str, Any]:
    """Flatten one AOI detail record into the wide CSV row."""
    west, south, east, north = detail["aoi_bbox_west_south_east_north"]
    row: dict[str, Any] = {
        "experiment_id": detail["experiment_id"],
        "region_key": detail["region_key"],
        "display_name": detail["display_name"],
        "country": detail["country"],
        "role": detail["role"],
        "aoi_bbox_west": west, "aoi_bbox_south": south,
        "aoi_bbox_east": east, "aoi_bbox_north": north,
    }
    for window in WINDOW_KINDS:
        block = detail["windows"][window]
        row[f"{window}_start_date"] = block["start_date"]
        row[f"{window}_end_date_inclusive"] = block["end_date_inclusive"]
        row[f"{window}_end_date_exclusive"] = block["end_date_exclusive"]
        row[f"{window}_n_days_inclusive"] = block["n_days_inclusive"]
        row[f"{window}_n_hours_observed"] = block["n_hours_observed"]
        for variable, statistics in VARIABLES.items():
            for statistic in statistics:
                metric = block["metrics"][f"{variable}_{statistic}"]
                for field in METRIC_FIELDS:
                    row[metric_key(window, variable, statistic, field)] = metric[field]
    return row


# =============================================================================
# Writing
# =============================================================================
def _format_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # repr round-trips exactly, so the CSV and JSON values are identical
        # numbers rather than merely similar ones.
        return repr(value)
    return str(value)


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    import csv

    columns = summary_columns()
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_format_csv_value(row.get(column)) for column in columns])
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def build_manifest(analysis_id: str, root: Path, *, engine_name: str,
                   source_grid: Optional[dict[str, Any]],
                   scientific_config_sha256: str) -> dict[str, Any]:
    files = []
    for name in PRODUCT_FILENAMES:
        path = Path(root) / name
        files.append({
            "name": name,
            "path": str(path),
            "relative_path": name,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "diagnostic_namespace": DIAGNOSTIC_NAMESPACE,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "engine": engine_name,
        "source_collection": COLLECTION_ID,
        "source_grid": source_grid,
        "scientific_config_sha256": scientific_config_sha256,
        "files": files,
    }


# =============================================================================
# Orchestration
# =============================================================================
def run_analysis(
    experiments: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    engine: Any = None,
    output_root: Optional[Path] = None,
    project: Optional[str] = None,
) -> dict[str, Any]:
    """Run (or plan) the ERA5-Land regional diagnostic.

    `dry_run=True` is strictly read-only: it resolves the cohort, windows,
    climatology mapping, scientific config, analysis_id and planned output
    paths, and NEVER constructs an engine, initialises Earth Engine, issues a
    query, creates a directory or writes a file.

    An injected `engine` is used as-is and is never re-initialised beyond the
    single `initialise()` call, so tests can pass a fake and stay offline. When
    no engine is injected, the production engine is constructed only in the
    actual (non-dry-run) path.
    """
    plan = build_plan(experiments, output_root)
    analysis_id = plan["analysis_id"]
    scientific_config = plan["scientific_config"]

    if dry_run:
        return {
            "ran": False,
            "dry_run": True,
            "already_complete": False,
            "gee_initialised": False,
            "gee_queries_run": 0,
            "files_written": False,
            **plan,
        }

    state = namespace_state(analysis_id, output_root)
    if state["state"] == "complete":
        # Identical contract, verified artifact: report it, recompute nothing,
        # overwrite nothing.
        return {
            "ran": False,
            "dry_run": False,
            "already_complete": True,
            "gee_initialised": False,
            "gee_queries_run": 0,
            "files_written": False,
            "analysis_id": analysis_id,
            "output_root": state["root"],
            "namespace_state": state,
            "experiment_ids": list(plan["experiment_ids"]),
            "planned_output_paths": plan["planned_output_paths"],
        }
    if state["state"] != "absent":
        raise Era5LandRegionalDiagnosticError(
            f"Output namespace {state['root']} already exists but is not a "
            f"complete, self-consistent artifact ({state['reason']}). Refusing "
            "to overwrite it. Inspect or move it aside manually; this "
            "diagnostic never deletes or silently replaces an existing "
            "namespace."
        )

    engine = engine if engine is not None else Era5LandRegionalEngine(project=project)
    initialisation = engine.initialise()

    details = [
        compute_experiment_row(experiment_id, engine)
        for experiment_id in plan["experiment_ids"]
    ]
    rows = [summary_row(detail) for detail in details]
    if len(rows) != len(plan["experiment_ids"]):  # pragma: no cover -- structural
        raise Era5LandRegionalDiagnosticError(
            f"Produced {len(rows)} row(s) for {len(plan['experiment_ids'])} "
            "experiment(s); exactly one row per experiment is required."
        )

    source_grid = None
    grid_reader = getattr(engine, "source_grid", None)
    if callable(grid_reader):
        try:
            source_grid = grid_reader(COLLECTION_ID)
        except Exception:  # noqa: BLE001 -- provenance only; never fails the run
            source_grid = None

    root = analysis_root(analysis_id, output_root)
    root.mkdir(parents=True, exist_ok=True)

    write_summary_csv(root / SUMMARY_CSV_NAME, rows)
    _write_json(root / SUMMARY_JSON_NAME, {
        "schema_version": SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "experiment_ids": list(plan["experiment_ids"]),
        "climatology_years": list(CLIMATOLOGY_YEARS),
        "summary_columns": summary_columns(),
        "rows": rows,
        "experiments": details,
    })
    _write_json(root / CONTRACT_JSON_NAME, {
        "analysis_id": analysis_id,
        "scientific_config": scientific_config,
        "canonical_json_sha256": compute_analysis_id(scientific_config),
    })
    manifest = build_manifest(
        analysis_id, root,
        engine_name=str(getattr(engine, "name", type(engine).__name__)),
        source_grid=source_grid,
        scientific_config_sha256=compute_analysis_id(scientific_config),
    )
    _write_json(root / MANIFEST_JSON_NAME, manifest)

    return {
        "ran": True,
        "dry_run": False,
        "already_complete": False,
        "gee_initialised": bool(getattr(engine, "contacts_earth_engine", False)),
        "initialisation": initialisation,
        "analysis_id": analysis_id,
        "experiment_ids": list(plan["experiment_ids"]),
        "n_rows": len(rows),
        "output_root": str(root),
        "files_written": True,
        "output_paths": {name: str(root / name) for name in EXPECTED_FILENAMES},
        "manifest": manifest,
        "engine": str(getattr(engine, "name", type(engine).__name__)),
    }
