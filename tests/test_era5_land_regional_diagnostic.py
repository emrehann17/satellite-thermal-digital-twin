"""
tests/test_era5_land_regional_diagnostic.py

Tests for the standalone ERA5-Land AOI-level regional meteorology diagnostic
(src/era5_land_regional_diagnostic.py).

Every test runs with an INJECTED FAKE ENGINE or with pure functions: no Earth
Engine session, no network call, no export, no model, no raster and no write
outside pytest's tmp_path. The production engine is only ever constructed (to
prove construction contacts nothing) -- never initialised.

Covers:
    import / dry-run contact no GEE and write nothing
    the exact five default experiments, and mugla_2022 absent by default
    registry window resolution and inclusive-end -> exclusive-end conversion
    climatology month/day mapping, including fail-closed cases
    the RH recipe against deterministic inputs, wind, precipitation m -> mm
    mean/max/total temporal semantics over hourly regional means
    four-year climatology arithmetic, sample SD (ddof=1), zero-SD -> z=None
    output row count / wide schema / CSV-JSON agreement
    deterministic analysis_id
    existing-output behaviour: idempotent-complete vs fail-closed
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.era5_land_regional_diagnostic as era5
from core.regions import get_experiment

FROZEN_FIVE = (
    "manavgat_2021", "bejis_2022", "mugla_2021",
    "evia_2021_extended", "montiferru_2021",
)


# =============================================================================
# Fake engine: returns hourly, already-spatially-reduced regional means
# =============================================================================
class FakeEngine:
    """Deterministic stand-in for the production engine.

    Records every request so the tests can assert on the exact windows asked
    for. It never imports `ee` and never touches the network.
    """

    name = "fake_era5_engine_v1"
    contacts_earth_engine = False

    #: realization -> base offset. Distinct offsets give a non-zero
    #: climatology SD; `ConstantClimatologyEngine` below overrides them.
    OFFSETS = {
        "observed": 10.0,
        "climatology_2017": 1.0,
        "climatology_2018": 2.0,
        "climatology_2019": 3.0,
        "climatology_2020": 4.0,
    }

    def __init__(self, offsets: dict[str, float] | None = None) -> None:
        self.offsets = dict(offsets or self.OFFSETS)
        self.calls: list[dict] = []
        self.initialise_calls = 0

    def initialise(self) -> dict:
        self.initialise_calls += 1
        return {"initialised": True, "project": None}

    def source_grid(self, collection_id: str = era5.COLLECTION_ID) -> dict:
        return {
            "collection_id": collection_id,
            "canonical_projection_band": era5.SOURCE_BANDS[0],
            "crs": "EPSG:4326",
            "crs_transform": [0.1, 0.0, -180.05, 0.0, -0.1, 90.05],
            "nominal_scale_m": 11132.0,
            "projection_read_method": "fake",
        }

    def _base(self, kwargs: dict) -> float:
        offset = self.offsets[kwargs["realization"]]
        # A small, deterministic per-AOI/per-window shift so no two rows are
        # accidentally identical.
        spread = (len(kwargs["experiment_id"]) % 7) + (1 if kwargs["window"] == "label" else 0)
        return offset + float(spread)

    def hourly_regional_series(self, **kwargs) -> dict:
        self.calls.append(dict(kwargs))
        base = self._base(kwargs)
        # A complete, in-order UTC hourly sequence for the requested window --
        # exactly what a healthy ERA5-Land window returns.
        timestamps = era5.expected_hourly_timestamps_ms(
            kwargs["start"], kwargs["end_exclusive"]
        )
        hours = len(timestamps)
        return {
            "timestamps": timestamps,
            "temperature_c": [base + 20.0 + (i % 24) for i in range(hours)],
            "relative_humidity_percent": [base + 30.0 + 2 * (i % 24) for i in range(hours)],
            "wind_speed_m_s": [base + 1.0 + 0.5 * (i % 24) for i in range(hours)],
            "precipitation_mm": [base * 0.1 + 0.25 * (i % 24) for i in range(hours)],
            "n_hours": hours,
        }


class ConstantClimatologyEngine(FakeEngine):
    """All four reference years return identical values -> climatology SD == 0."""

    OFFSETS = {
        "observed": 10.0,
        "climatology_2017": 5.0,
        "climatology_2018": 5.0,
        "climatology_2019": 5.0,
        "climatology_2020": 5.0,
    }


class ExplodingEngine:
    """Any attribute access is a test failure: proves a path never uses it."""

    name = "exploding_engine"
    contacts_earth_engine = True

    def __getattr__(self, item):  # pragma: no cover -- only fires on a defect
        raise AssertionError(
            f"the engine must not be used here (attribute {item!r} requested)"
        )


@pytest.fixture()
def no_gee(monkeypatch):
    """Make any Earth Engine session attempt an immediate, loud failure."""
    import core.gee_utils as gee_utils

    def _explode(*args, **kwargs):  # pragma: no cover -- only fires on a defect
        raise AssertionError("init_gee() must not be called in this test")

    monkeypatch.setattr(gee_utils, "init_gee", _explode)
    return _explode


# =============================================================================
# Import / dry-run contact nothing
# =============================================================================
def test_module_import_and_engine_construction_contact_nothing(no_gee):
    engine = era5.Era5LandRegionalEngine(project="irrelevant")
    assert engine.contacts_earth_engine is True
    assert engine._initialised is False
    assert engine.name == "era5_land_regional_production_v1"


def test_dry_run_creates_nothing_and_never_touches_the_engine(tmp_path, no_gee):
    result = era5.run_analysis(
        dry_run=True, engine=ExplodingEngine(), output_root=tmp_path,
    )

    assert result["dry_run"] is True
    assert result["ran"] is False
    assert result["files_written"] is False
    assert result["gee_initialised"] is False
    assert result["gee_queries_run"] == 0
    assert list(tmp_path.iterdir()) == []
    assert not Path(result["planned_output_root"]).exists()
    for path in result["planned_output_paths"].values():
        assert not Path(path).exists()


def test_dry_run_reports_everything_the_contract_requires(tmp_path, no_gee):
    plan = era5.run_analysis(dry_run=True, output_root=tmp_path)

    assert plan["experiment_ids"] == list(FROZEN_FIVE)
    assert plan["collection"] == "ECMWF/ERA5_LAND/HOURLY"
    assert plan["climatology_years"] == [2017, 2018, 2019, 2020]
    # 5 AOIs x 2 windows x (1 observed + 4 climatology years)
    assert plan["n_engine_window_requests"] == 50
    assert plan["scientific_config"]["experiment_ids"] == list(FROZEN_FIVE)
    for entry in plan["experiments"]:
        assert set(entry["observed_windows"]) == {"predictor", "label"}
        assert len(entry["climatology_windows"]["predictor"]) == 4
        assert len(entry["aoi_bbox_west_south_east_north"]) == 4


# =============================================================================
# Cohort
# =============================================================================
def test_default_cohort_is_exactly_the_frozen_five_in_order():
    assert era5.DEFAULT_EXPERIMENTS == FROZEN_FIVE
    assert era5.resolve_experiments() == FROZEN_FIVE


def test_mugla_2022_is_absent_by_default_but_documented():
    assert "mugla_2022" not in era5.DEFAULT_EXPERIMENTS
    assert "mugla_2022" in era5.NON_DEFAULT_EXPERIMENTS
    assert "supervisor" in era5.NON_DEFAULT_EXPERIMENTS["mugla_2022"]
    assert "mugla_2022" not in era5.build_plan()["experiment_ids"]


def test_mugla_2022_can_only_enter_explicitly_and_changes_the_analysis_id():
    default_id = era5.compute_analysis_id(era5.build_scientific_config(FROZEN_FIVE))
    with_extra = era5.compute_analysis_id(
        era5.build_scientific_config(FROZEN_FIVE + ("mugla_2022",))
    )
    assert with_extra != default_id


def test_unknown_disabled_or_duplicate_experiments_fail_closed():
    with pytest.raises(ValueError):
        era5.resolve_experiments(["no_such_experiment"])
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.resolve_experiments(["mugla_2021", "mugla_2021"])
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.resolve_experiments([])
    # evia_2021 is registered and enabled but legacy_superseded.
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.resolve_experiments(["evia_2021"])


# =============================================================================
# Windows: registry -> inclusive -> exclusive, and climatology mapping
# =============================================================================
def test_inclusive_registry_end_becomes_exclusive_gee_end():
    assert era5.exclusive_end("2021-08-31") == "2021-09-01"
    assert era5.exclusive_end("2021-12-31") == "2022-01-01"
    assert era5.exclusive_end("2020-02-28") == "2020-02-29"  # leap year


@pytest.mark.parametrize("experiment_id", FROZEN_FIVE)
def test_observed_windows_come_from_the_registry_unchanged(experiment_id):
    record = get_experiment(experiment_id)
    windows = era5.observed_windows(experiment_id)
    for kind in ("predictor", "label"):
        assert windows[kind]["start_date"] == record[f"{kind}_start_date"]
        assert windows[kind]["end_date_inclusive"] == record[f"{kind}_end_date"]
        assert windows[kind]["end_date_exclusive"] == era5.exclusive_end(
            record[f"{kind}_end_date"]
        )


def test_climatology_maps_the_exact_month_day_into_each_reference_year():
    mapped = era5.climatology_windows("2022-06-15", "2022-08-14")
    assert [m["reference_year"] for m in mapped] == [2017, 2018, 2019, 2020]
    for m in mapped:
        year = m["reference_year"]
        assert m["start_date"] == f"{year}-06-15"
        assert m["end_date_inclusive"] == f"{year}-08-14"
        assert m["end_date_exclusive"] == f"{year}-08-15"
        assert m["n_days_inclusive"] == 61


def test_unrepresentable_leap_day_fails_closed_instead_of_shifting():
    with pytest.raises(era5.Era5LandRegionalDiagnosticError) as excinfo:
        era5.map_window_to_year("2020-02-29", "2020-03-05", 2019)
    assert "29 February" in str(excinfo.value) or "cannot be represented" in str(excinfo.value)
    # The same date IS representable in a leap reference year.
    assert era5.map_window_to_year("2020-02-29", "2020-03-05", 2020)["start_date"] == "2020-02-29"


def test_year_crossing_window_is_refused():
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.map_window_to_year("2021-12-20", "2022-01-10", 2019)


# =============================================================================
# Derivation formulas
# =============================================================================
def test_relative_humidity_is_the_documented_ecmwf_tetens_recipe():
    # Saturated air: Td == T  ->  es(Td)/es(T) == 1  ->  RH == 100 exactly.
    assert era5.relative_humidity_percent(300.0, 300.0) == pytest.approx(100.0, abs=1e-12)

    # Independent restatement of the frozen recipe with the frozen constants.
    def es(t):
        return 611.21 * math.exp(17.502 * (t - 273.16) / (t - 32.19))

    for t_k, td_k in ((303.15, 293.15), (275.0, 274.0), (310.0, 280.0)):
        assert era5.relative_humidity_percent(t_k, td_k) == pytest.approx(
            100.0 * es(td_k) / es(t_k), rel=1e-12
        )

    assert era5.TETENS_T0_K == 273.16
    assert era5.TETENS_A1_PA == 611.21
    assert era5.TETENS_A3 == 17.502
    assert era5.TETENS_A4_K == 32.19


def test_relative_humidity_is_not_silently_clipped():
    # Supersaturated input (Td > T) must yield RH > 100, not a clipped 100.
    assert era5.relative_humidity_percent(295.0, 297.0) > 100.0


def test_temperature_wind_and_precipitation_conversions():
    assert era5.temperature_celsius(273.15) == pytest.approx(0.0, abs=1e-12)
    assert era5.temperature_celsius(300.0) == pytest.approx(26.85, abs=1e-12)
    assert era5.wind_speed_m_s(3.0, 4.0) == pytest.approx(5.0, abs=1e-12)
    assert era5.wind_speed_m_s(-3.0, -4.0) == pytest.approx(5.0, abs=1e-12)
    assert era5.wind_speed_m_s(0.0, 0.0) == 0.0
    assert era5.precipitation_mm_from_m(0.0025) == pytest.approx(2.5, abs=1e-12)
    assert era5.PRECIPITATION_M_TO_MM == 1000.0
    # The HOURLY band is the source; the cumulative band is never requested.
    assert "total_precipitation_hourly" in era5.SOURCE_BANDS
    assert "total_precipitation" not in era5.SOURCE_BANDS


# =============================================================================
# Temporal statistics
# =============================================================================
def test_window_statistics_mean_max_total_semantics():
    series = [1.0, 5.0, 3.0, 3.0]
    assert era5.window_statistics(series, ("mean", "max")) == {"mean": 3.0, "max": 5.0}
    assert era5.window_statistics(series, ("mean", "max", "total")) == {
        "mean": 3.0, "max": 5.0, "total": 12.0,
    }
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.window_statistics([], ("mean",))


def test_only_precipitation_carries_a_total():
    assert era5.VARIABLES["temperature_c"] == ("mean", "max")
    assert era5.VARIABLES["relative_humidity_percent"] == ("mean", "max")
    assert era5.VARIABLES["wind_speed_m_s"] == ("mean", "max")
    assert era5.VARIABLES["precipitation_mm"] == ("mean", "max", "total")


def test_max_is_the_maximum_regional_hour_not_a_pixel_extreme():
    # The engine hands over hourly AOI means only; the module never sees a
    # pixel, so "max" cannot be a per-pixel extreme by construction.
    stats = era5.window_statistics_from_series({
        "temperature_c": [10.0, 30.0],
        "relative_humidity_percent": [40.0, 60.0],
        "wind_speed_m_s": [1.0, 2.0],
        "precipitation_mm": [0.5, 1.5],
    })
    assert stats["temperature_c"]["max"] == 30.0
    assert stats["precipitation_mm"]["total"] == 2.0
    assert "regional-hour" in era5.TEMPORAL_AGGREGATION_SEMANTICS


# =============================================================================
# Climatology arithmetic
# =============================================================================
def test_sample_standard_deviation_uses_ddof_one():
    assert era5.SD_DDOF == 1
    assert era5.sample_standard_deviation([1.0, 2.0, 3.0, 4.0]) == pytest.approx(
        math.sqrt(5.0 / 3.0), rel=1e-12
    )
    # Population SD of the same sample is sqrt(1.25) -- explicitly NOT used.
    assert era5.sample_standard_deviation([1.0, 2.0, 3.0, 4.0]) != pytest.approx(
        math.sqrt(1.25), rel=1e-9
    )
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.sample_standard_deviation([3.0])


def test_four_year_climatology_arithmetic():
    stats = era5.climatology_statistics(
        observed=10.0, realizations={2017: 1.0, 2018: 2.0, 2019: 3.0, 2020: 4.0},
    )
    assert stats["climatology_mean"] == pytest.approx(2.5, rel=1e-12)
    assert stats["climatology_sd"] == pytest.approx(math.sqrt(5.0 / 3.0), rel=1e-12)
    assert stats["anomaly"] == pytest.approx(7.5, rel=1e-12)
    assert stats["standardized_anomaly"] == pytest.approx(
        7.5 / math.sqrt(5.0 / 3.0), rel=1e-12
    )
    assert stats["zero_climatology_sd"] is False
    assert stats["climatology_realizations"] == {
        "2017": 1.0, "2018": 2.0, "2019": 3.0, "2020": 4.0,
    }


def test_zero_climatology_sd_yields_null_z_never_inf_or_zero():
    stats = era5.climatology_statistics(
        observed=7.0, realizations={2017: 5.0, 2018: 5.0, 2019: 5.0, 2020: 5.0},
    )
    assert stats["climatology_sd"] == 0.0
    assert stats["standardized_anomaly"] is None
    assert stats["zero_climatology_sd"] is True
    assert stats["anomaly"] == pytest.approx(2.0, rel=1e-12)


def test_missing_or_non_finite_climatology_realizations_fail_closed():
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.climatology_statistics(1.0, {2017: 1.0, 2018: 2.0, 2019: 3.0})
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.climatology_statistics(
            1.0, {2017: 1.0, 2018: 2.0, 2019: 3.0, 2020: float("inf")},
        )
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.climatology_statistics(
            float("nan"), {2017: 1.0, 2018: 2.0, 2019: 3.0, 2020: 4.0},
        )


# --- hourly completeness ----------------------------------------------------
ONE_DAY_START, ONE_DAY_END = "2021-06-01", "2021-06-02"
ONE_DAY_HOURS = 24


def _complete_payload(start: str = ONE_DAY_START, end_exclusive: str = ONE_DAY_END) -> dict:
    timestamps = era5.expected_hourly_timestamps_ms(start, end_exclusive)
    hours = len(timestamps)
    return {
        "timestamps": list(timestamps),
        **{variable: [float(i) for i in range(hours)] for variable in era5.VARIABLES},
    }


def _check(payload: dict, start: str = ONE_DAY_START, end_exclusive: str = ONE_DAY_END):
    return era5.validate_hourly_series(
        payload, start=start, end_exclusive=end_exclusive, context="t",
    )


def test_expected_hourly_timestamps_are_the_exact_utc_sequence():
    stamps = era5.expected_hourly_timestamps_ms("2021-06-01", "2021-06-02")
    assert len(stamps) == 24
    assert stamps[0] == 1_622_505_600_000  # 2021-06-01T00:00:00Z
    assert all(b - a == era5.MILLISECONDS_PER_HOUR for a, b in zip(stamps, stamps[1:]))
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)

    # Length is always n_days * 24 -- UTC has no DST discontinuity.
    for start, end, days in (("2021-03-27", "2021-03-30", 3),
                             ("2021-10-30", "2021-11-02", 3),
                             ("2020-02-28", "2020-03-01", 2)):
        assert len(era5.expected_hourly_timestamps_ms(start, end)) == days * 24
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.expected_hourly_timestamps_ms("2021-06-02", "2021-06-02")


def test_complete_hourly_window_passes():
    series = _check(_complete_payload())
    assert sorted(series) == sorted(era5.VARIABLES)
    assert all(len(values) == ONE_DAY_HOURS for values in series.values())


def test_missing_hour_fails_closed():
    payload = _complete_payload()
    del payload["timestamps"][5]
    for variable in era5.VARIABLES:
        del payload[variable][5]
    with pytest.raises(era5.Era5LandRegionalDiagnosticError) as excinfo:
        _check(payload)
    assert "23 hour(s)" in str(excinfo.value)


def test_duplicate_hour_fails_closed():
    payload = _complete_payload()
    payload["timestamps"][5] = payload["timestamps"][4]  # count preserved
    with pytest.raises(era5.Era5LandRegionalDiagnosticError) as excinfo:
        _check(payload)
    assert "duplicate" in str(excinfo.value)


def test_out_of_order_hours_fail_closed():
    payload = _complete_payload()
    payload["timestamps"][3], payload["timestamps"][4] = (
        payload["timestamps"][4], payload["timestamps"][3]
    )
    with pytest.raises(era5.Era5LandRegionalDiagnosticError) as excinfo:
        _check(payload)
    # Never silently sorted back into place.
    assert "not the exact contiguous UTC sequence" in str(excinfo.value)


def test_shifted_window_fails_closed():
    payload = _complete_payload()
    payload["timestamps"] = [t + 60_000 for t in payload["timestamps"]]  # +1 minute
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        _check(payload)

    # A whole-hour shift (window off by one hour) is caught too.
    shifted = _complete_payload()
    shifted["timestamps"] = [
        t + era5.MILLISECONDS_PER_HOUR for t in shifted["timestamps"]
    ]
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        _check(shifted)


def test_extra_hour_fails_closed():
    payload = _complete_payload()
    payload["timestamps"].append(payload["timestamps"][-1] + era5.MILLISECONDS_PER_HOUR)
    for variable in era5.VARIABLES:
        payload[variable].append(1.0)
    with pytest.raises(era5.Era5LandRegionalDiagnosticError) as excinfo:
        _check(payload)
    assert "25 hour(s)" in str(excinfo.value)


def test_timestamp_and_variable_length_mismatch_fails_closed():
    payload = _complete_payload()
    payload["temperature_c"] = payload["temperature_c"][:-1]
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        _check(payload)

    same_but_short = _complete_payload()
    for variable in era5.VARIABLES:
        same_but_short[variable] = same_but_short[variable][:-1]
    with pytest.raises(era5.Era5LandRegionalDiagnosticError) as excinfo:
        _check(same_but_short)
    assert "aligned to the same complete hourly sequence" in str(excinfo.value)


def test_missing_or_malformed_timestamps_fail_closed():
    absent = _complete_payload()
    del absent["timestamps"]
    with pytest.raises(era5.Era5LandRegionalDiagnosticError) as excinfo:
        _check(absent)
    assert "no 'timestamps'" in str(excinfo.value)

    not_a_list = _complete_payload()
    not_a_list["timestamps"] = 1_622_505_600_000
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        _check(not_a_list)

    for bad in (None, "2021-06-01T00:00:00Z", 1_622_505_600_000.5, float("nan"), True):
        payload = _complete_payload()
        payload["timestamps"][7] = bad
        with pytest.raises(era5.Era5LandRegionalDiagnosticError):
            _check(payload)

    # An integral float IS accepted -- only non-integer/non-finite values fail.
    float_stamps = _complete_payload()
    float_stamps["timestamps"] = [float(t) for t in float_stamps["timestamps"]]
    assert len(_check(float_stamps)["wind_speed_m_s"]) == ONE_DAY_HOURS


def test_engine_series_qa_rejects_null_and_non_finite_hours():
    good = _complete_payload()

    for broken in (
        {**good, "temperature_c": [None] + good["temperature_c"][1:]},
        {**good, "temperature_c": [float("inf")] + good["temperature_c"][1:]},
        {**good, "temperature_c": ["x"] + good["temperature_c"][1:]},
        {**good, "temperature_c": []},
        {k: v for k, v in good.items() if k != "wind_speed_m_s"},
    ):
        with pytest.raises(era5.Era5LandRegionalDiagnosticError):
            _check(broken)


def test_scientific_config_records_the_hourly_completeness_rule():
    config = era5.build_scientific_config(FROZEN_FIVE)
    temporal = config["temporal_aggregation"]
    assert temporal["hourly_completeness_rule"] == era5.HOURLY_COMPLETENESS_RULE
    assert "n_days_inclusive * 24" in temporal["hourly_completeness_rule"]
    assert temporal["expected_hours_formula"] == "n_days_inclusive * 24"
    assert temporal["timestamp_convention"] == era5.TIMESTAMP_CONVENTION
    assert temporal["incomplete_window_policy"] == "fail_closed"

    # The rule is hashed: dropping it changes the analysis_id.
    weakened = json.loads(json.dumps(config))
    del weakened["temporal_aggregation"]["hourly_completeness_rule"]
    assert era5.compute_analysis_id(weakened) != era5.compute_analysis_id(config)


def test_an_incomplete_window_fails_the_whole_run(tmp_path, no_gee):
    class ShortWindowEngine(FakeEngine):
        """Drops the last hour of every climatology_2019 window."""

        def hourly_regional_series(self, **kwargs):
            payload = super().hourly_regional_series(**kwargs)
            if kwargs["realization"] == "climatology_2019":
                payload["timestamps"] = payload["timestamps"][:-1]
                for variable in era5.VARIABLES:
                    payload[variable] = payload[variable][:-1]
                payload["n_hours"] = len(payload["timestamps"])
            return payload

    with pytest.raises(era5.Era5LandRegionalDiagnosticError) as excinfo:
        era5.run_analysis(
            experiments=["mugla_2021"], engine=ShortWindowEngine(), output_root=tmp_path,
        )
    assert "climatology_2019" in str(excinfo.value)
    # Nothing was written for a run that could not be completed.
    assert not (tmp_path / "diagnostics").exists()


# =============================================================================
# Determinism of the analysis_id
# =============================================================================
def test_analysis_id_is_deterministic_and_contract_bound():
    config = era5.build_scientific_config(FROZEN_FIVE)
    first = era5.compute_analysis_id(config)
    second = era5.compute_analysis_id(era5.build_scientific_config(FROZEN_FIVE))
    assert first == second
    assert len(first) == 64

    reordered = era5.compute_analysis_id(
        era5.build_scientific_config(tuple(reversed(FROZEN_FIVE)))
    )
    assert reordered != first

    mutated = json.loads(json.dumps(config))
    mutated["climatology"]["reference_years"] = [2016, 2017, 2018, 2019]
    assert era5.compute_analysis_id(mutated) != first


# =============================================================================
# End-to-end with the fake engine
# =============================================================================
def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return header, rows


def test_full_run_writes_one_row_per_experiment_with_the_wide_schema(tmp_path, no_gee):
    engine = FakeEngine()
    result = era5.run_analysis(engine=engine, output_root=tmp_path)

    assert result["ran"] is True
    assert result["n_rows"] == 5
    assert engine.initialise_calls == 1
    # 5 AOIs x 2 windows x (observed + 4 climatology years)
    assert len(engine.calls) == 50

    root = Path(result["output_root"])
    assert root.parent.name == "era5_land_regional"
    assert root.name == result["analysis_id"]
    assert sorted(p.name for p in root.iterdir()) == sorted(era5.EXPECTED_FILENAMES)

    header, rows = _read_csv(root / era5.SUMMARY_CSV_NAME)
    assert header == era5.summary_columns()
    assert len(rows) == 5
    assert [row["experiment_id"] for row in rows] == list(FROZEN_FIVE)
    assert len({row["experiment_id"] for row in rows}) == 5
    assert "mugla_2022" not in {row["experiment_id"] for row in rows}

    for row in rows:
        record = get_experiment(row["experiment_id"])
        assert row["region_key"] == record["region_key"]
        for kind in ("predictor", "label"):
            assert row[f"{kind}_start_date"] == record[f"{kind}_start_date"]
            assert row[f"{kind}_end_date_inclusive"] == record[f"{kind}_end_date"]
            assert row[f"{kind}_end_date_exclusive"] == era5.exclusive_end(
                record[f"{kind}_end_date"]
            )
            for variable, statistics in era5.VARIABLES.items():
                for statistic in statistics:
                    for field in ("observed", "climatology_mean", "climatology_sd",
                                  "anomaly"):
                        key = era5.metric_key(kind, variable, statistic, field)
                        assert math.isfinite(float(row[key]))


def test_engine_receives_exclusive_ends_and_mapped_climatology_windows(tmp_path, no_gee):
    engine = FakeEngine()
    era5.run_analysis(experiments=["mugla_2021"], engine=engine, output_root=tmp_path)

    record = get_experiment("mugla_2021")
    predictor = [c for c in engine.calls if c["window"] == "predictor"]
    observed = [c for c in predictor if c["realization"] == "observed"]
    assert len(observed) == 1
    assert observed[0]["start"] == record["predictor_start_date"]
    assert observed[0]["end_exclusive"] == "2021-07-29"  # inclusive 2021-07-28 + 1 day
    assert observed[0]["collection_id"] == "ECMWF/ERA5_LAND/HOURLY"
    assert list(observed[0]["bands"]) == list(era5.SOURCE_BANDS)

    for year in (2017, 2018, 2019, 2020):
        call = [c for c in predictor if c["realization"] == f"climatology_{year}"]
        assert len(call) == 1
        assert call[0]["start"] == f"{year}-06-01"
        assert call[0]["end_exclusive"] == f"{year}-07-29"


def test_summary_json_retains_the_four_realizations_and_agrees_with_the_csv(
    tmp_path, no_gee,
):
    result = era5.run_analysis(
        experiments=["mugla_2021"], engine=FakeEngine(), output_root=tmp_path,
    )
    root = Path(result["output_root"])
    summary = json.loads((root / era5.SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
    _header, rows = _read_csv(root / era5.SUMMARY_CSV_NAME)

    detail = summary["experiments"][0]
    metrics = detail["windows"]["predictor"]["metrics"]
    realizations = metrics["temperature_c_mean"]["climatology_realizations"]
    assert sorted(realizations) == ["2017", "2018", "2019", "2020"]

    values = [realizations[str(y)] for y in (2017, 2018, 2019, 2020)]
    assert metrics["temperature_c_mean"]["climatology_mean"] == pytest.approx(
        sum(values) / 4.0, rel=1e-12
    )
    assert metrics["temperature_c_mean"]["climatology_sd"] == pytest.approx(
        era5.sample_standard_deviation(values), rel=1e-12
    )

    csv_row, json_row = rows[0], summary["rows"][0]
    for key, value in json_row.items():
        if isinstance(value, bool):
            assert csv_row[key] == ("true" if value else "false")
        elif value is None:
            assert csv_row[key] == ""
        elif isinstance(value, float):
            assert float(csv_row[key]) == value
        else:
            assert csv_row[key] == str(value)


def test_zero_climatology_sd_is_written_as_an_empty_z_and_a_true_flag(tmp_path, no_gee):
    result = era5.run_analysis(
        experiments=["mugla_2021"], engine=ConstantClimatologyEngine(),
        output_root=tmp_path,
    )
    root = Path(result["output_root"])
    _header, rows = _read_csv(root / era5.SUMMARY_CSV_NAME)
    row = rows[0]

    for window in era5.WINDOW_KINDS:
        for variable, statistics in era5.VARIABLES.items():
            for statistic in statistics:
                sd_key = era5.metric_key(window, variable, statistic, "climatology_sd")
                z_key = era5.metric_key(window, variable, statistic, "standardized_anomaly")
                flag_key = era5.metric_key(window, variable, statistic,
                                           "zero_climatology_sd")
                assert float(row[sd_key]) == 0.0
                assert row[z_key] == ""
                assert row[flag_key] == "true"

    text = (root / era5.SUMMARY_JSON_NAME).read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    summary = json.loads(text)
    metric = summary["experiments"][0]["windows"]["predictor"]["metrics"]["wind_speed_m_s_max"]
    assert metric["standardized_anomaly"] is None
    assert metric["zero_climatology_sd"] is True


def test_manifest_records_paths_sizes_and_hashes(tmp_path, no_gee):
    result = era5.run_analysis(engine=FakeEngine(), output_root=tmp_path)
    root = Path(result["output_root"])
    manifest = json.loads((root / era5.MANIFEST_JSON_NAME).read_text(encoding="utf-8"))

    assert manifest["analysis_id"] == result["analysis_id"]
    assert manifest["engine"] == "fake_era5_engine_v1"
    recorded = {entry["name"]: entry for entry in manifest["files"]}
    assert sorted(recorded) == sorted(era5.PRODUCT_FILENAMES)
    for name, entry in recorded.items():
        path = root / name
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == era5.sha256_file(path)
    assert manifest["source_grid"]["crs"] == "EPSG:4326"


def test_produced_hour_counts_equal_window_days_times_24(tmp_path, no_gee):
    result = era5.run_analysis(
        experiments=["mugla_2021"], engine=FakeEngine(), output_root=tmp_path,
    )
    root = Path(result["output_root"])
    _header, rows = _read_csv(root / era5.SUMMARY_CSV_NAME)
    summary = json.loads((root / era5.SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
    row, detail = rows[0], summary["experiments"][0]

    for window in era5.WINDOW_KINDS:
        n_days = int(row[f"{window}_n_days_inclusive"])
        assert int(row[f"{window}_n_hours_observed"]) == n_days * 24

        block = detail["windows"][window]
        assert block["n_hours_observed"] == n_days * 24
        for mapped in block["climatology_windows"]:
            year = str(mapped["reference_year"])
            assert block["climatology_n_hours"][year] == mapped["n_days_inclusive"] * 24

    # mugla_2021 predictor: 2021-06-01..2021-07-28 inclusive = 58 days.
    assert int(row["predictor_n_days_inclusive"]) == 58
    assert int(row["predictor_n_hours_observed"]) == 58 * 24


def test_scientific_contract_file_reproduces_the_analysis_id(tmp_path, no_gee):
    result = era5.run_analysis(engine=FakeEngine(), output_root=tmp_path)
    contract = json.loads(
        (Path(result["output_root"]) / era5.CONTRACT_JSON_NAME).read_text(encoding="utf-8")
    )
    assert era5.compute_analysis_id(contract["scientific_config"]) == result["analysis_id"]
    assert contract["scientific_config"]["experiment_ids"] == list(FROZEN_FIVE)
    assert contract["scientific_config"]["climatology"]["sd_ddof"] == 1
    assert contract["scientific_config"]["source"]["collection"] == "ECMWF/ERA5_LAND/HOURLY"


# =============================================================================
# Resume / overwrite safety
# =============================================================================
def test_identical_rerun_of_a_complete_namespace_recomputes_nothing(tmp_path, no_gee):
    first = era5.run_analysis(engine=FakeEngine(), output_root=tmp_path)
    root = Path(first["output_root"])
    before = {p.name: era5.sha256_file(p) for p in root.iterdir()}

    second_engine = FakeEngine()
    second = era5.run_analysis(engine=second_engine, output_root=tmp_path)

    assert second["already_complete"] is True
    assert second["ran"] is False
    assert second["files_written"] is False
    assert second["analysis_id"] == first["analysis_id"]
    assert second_engine.initialise_calls == 0
    assert second_engine.calls == []
    assert {p.name: era5.sha256_file(p) for p in root.iterdir()} == before


def test_incomplete_namespace_fails_closed_and_is_never_overwritten(tmp_path, no_gee):
    first = era5.run_analysis(engine=FakeEngine(), output_root=tmp_path)
    root = Path(first["output_root"])
    (root / era5.MANIFEST_JSON_NAME).unlink()

    engine = FakeEngine()
    with pytest.raises(era5.Era5LandRegionalDiagnosticError) as excinfo:
        era5.run_analysis(engine=engine, output_root=tmp_path)
    assert "Refusing to overwrite" in str(excinfo.value)
    assert engine.calls == []
    # The surviving files are untouched.
    assert (root / era5.SUMMARY_CSV_NAME).is_file()


def test_tampered_namespace_fails_closed(tmp_path, no_gee):
    first = era5.run_analysis(engine=FakeEngine(), output_root=tmp_path)
    root = Path(first["output_root"])
    csv_path = root / era5.SUMMARY_CSV_NAME
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    state = era5.namespace_state(first["analysis_id"], tmp_path)
    assert state["state"] == "incomplete"
    assert "SHA-256" in state["reason"] or "bytes" in state["reason"]
    with pytest.raises(era5.Era5LandRegionalDiagnosticError):
        era5.run_analysis(engine=FakeEngine(), output_root=tmp_path)


def test_namespace_state_is_absent_before_any_run(tmp_path):
    plan = era5.build_plan(output_root=tmp_path)
    state = era5.namespace_state(plan["analysis_id"], tmp_path)
    assert state["state"] == "absent"


# =============================================================================
# Runner / validator wiring (still no GEE)
# =============================================================================
def test_runner_dry_run_writes_nothing(tmp_path, no_gee):
    from scripts.run_era5_land_regional_diagnostic import main as run_main

    result = run_main(dry_run=True, output_root=tmp_path)
    assert result["dry_run"] is True
    assert result["experiment_ids"] == list(FROZEN_FIVE)
    assert list(tmp_path.iterdir()) == []


def test_validator_passes_for_a_produced_artifact(tmp_path, no_gee):
    from scripts.validate_era5_land_regional_diagnostic import (
        validate_actual, validate_dry_run,
    )

    dry = validate_dry_run(tmp_path)
    assert dry.failed == [], [c["check_id"] for c in dry.failed]

    result = era5.run_analysis(engine=FakeEngine(), output_root=tmp_path)
    actual = validate_actual(result["analysis_id"], tmp_path)
    assert actual.failed == [], [c["check_id"] for c in actual.failed]


def test_validator_detects_a_tampered_artifact(tmp_path, no_gee):
    from scripts.validate_era5_land_regional_diagnostic import validate_actual

    result = era5.run_analysis(engine=FakeEngine(), output_root=tmp_path)
    root = Path(result["output_root"])
    csv_path = root / era5.SUMMARY_CSV_NAME
    header, rows = _read_csv(csv_path)
    key = era5.metric_key("predictor", "temperature_c", "mean", "observed")
    rows[0][key] = "999999.0"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    failed = {c["check_id"] for c in validate_actual(result["analysis_id"], tmp_path).failed}
    assert "A17_csv_and_json_values_agree" in failed
    assert "A21_manifest_hashes_and_sizes_match" in failed
