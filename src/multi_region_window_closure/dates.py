"""Exact window-date derivation for the multi-region window-closure analysis.

Every date is DERIVED from `core/regions.py::EXPERIMENTS` through the frozen
per-AOI helpers in `src.window_closure_sensitivity`. Nothing is transcribed
from documentation, and no date is ever hard-coded here.

Three date conventions coexist in this pipeline and are kept explicitly apart:

* Registry endpoints are INCLUSIVE calendar dates as written.
* `duration_days = (end - start).days` is the EXCLUSIVE difference, so it is
  one less than the inclusive calendar-day count. Both are emitted, under
  different column names.
* Earth Engine `filterDate(start, end)` is END-EXCLUSIVE. That production
  behaviour is preserved verbatim -- this module never adds a silent +1 day --
  and the effective last included date is recorded so the off-by-one is
  visible rather than implicit.

All dates are timezone-naive civil dates (`datetime.date`), parsed and
formatted as ISO `YYYY-MM-DD`. No UTC conversion is performed anywhere,
because Earth Engine `filterDate` and the registry both operate on civil
dates; introducing a timezone here would be a silent semantic change.

Design reference: docs/multi_region_window_closure_design/WINDOW_DATE_AUDIT.md
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional, Sequence

from src.multi_region_window_closure.contract import (
    CANONICAL_VARIANT_ID,
    DEFAULT_SHIFTS,
    MultiRegionWindowClosureError,
    REFERENCE_AOI,
    SYNTHESIS_AOIS,
    VARIANTS,
    modis_season_policy,
    shift_days_of,
)

DATE_FORMAT = "%Y-%m-%d"

#: Registry dates are civil dates with no timezone. Recorded explicitly so a
#: future reader does not have to infer it.
DATE_TIMEZONE_SEMANTICS = "timezone_naive_civil_date"

#: `event_*` and `gate_*` are ALIASES of the label window. Neither the registry
#: nor the built experiment context contains independent event or gate date
#: fields -- verified directly against `core.experiment_context`. The gate
#: consumes the label interval via
#: `step6_validate_fire_relation.label_window_doy_bounds`, imported by
#: `step6b_burned_landcover_gate`. Named fire-start dates exist only in
#: free-text registry `notes` and are NEVER parsed into a contract field.
LABEL_WINDOW_ALIAS = "label_window_alias"
EVENT_SOURCE_FIELD = "EXPERIMENTS[aoi].label_start_date|label_end_date"
GATE_SOURCE_FIELD = "EXPERIMENTS[aoi].label_start_date|label_end_date"
GATE_CONSUMER_SYMBOL = (
    "src.step6_validate_fire_relation.label_window_doy_bounds "
    "(imported by src.step6b_burned_landcover_gate)"
)


def parse_date(value: str) -> date:
    """Parse an ISO `YYYY-MM-DD` civil date. Fails closed on anything else."""
    try:
        return datetime.strptime(str(value), DATE_FORMAT).date()
    except (TypeError, ValueError) as exc:
        raise MultiRegionWindowClosureError(
            f"'{value}' is not an ISO {DATE_FORMAT} date."
        ) from exc


def format_date(value: date) -> str:
    return value.strftime(DATE_FORMAT)


# =============================================================================
# Earth Engine boundary conversion -- ONE central helper
# =============================================================================
def earth_engine_filter_bounds(start_date: str, end_date: str) -> dict[str, Any]:
    """The end-EXCLUSIVE `filterDate` contract for one window.

    Production calls `filterDate(start, end)`, whose END is exclusive. This is
    the single place that conversion happens, so no caller can drift by adding
    or omitting a day.
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    if end < start:
        raise MultiRegionWindowClosureError(
            f"Window end {format_date(end)} precedes start {format_date(start)}."
        )
    return {
        "earth_engine_filter_start": format_date(start),
        "earth_engine_filter_end": format_date(end),
        "earth_engine_end_exclusive": True,
        "effective_last_included_date": format_date(end - timedelta(days=1)),
        "predictor_start_inclusive": True,
        "predictor_end_inclusive": True,
        "calendar_duration_days": (end - start).days,
        "calendar_days_inclusive": (end - start).days + 1,
        "timezone_semantics": DATE_TIMEZONE_SEMANTICS,
    }


def modis_effective_days(start_date: str, end_date: str) -> dict[str, Any]:
    """Effective MODIS observation days after the FIXED summer-month filter.

    Mirrors `window_closure_sensitivity.modis_month_filter_transparency`: the
    day iteration is end-exclusive, exactly as `filterDate` is. Production
    behaviour is measured and reported, never corrected.
    """
    policy = modis_season_policy()
    months = set(range(policy["summer_month_start"], policy["summer_month_end"] + 1))
    start = parse_date(start_date)
    end = parse_date(end_date)
    included = [
        start + timedelta(days=offset)
        for offset in range((end - start).days)  # end-exclusive, like filterDate
        if (start + timedelta(days=offset)).month in months
    ]
    requested = (end - start).days
    return {
        "modis_policy_id": policy["modis_policy_id"],
        "calendar_month_filter": (
            f"{policy['summer_month_start']}-{policy['summer_month_end']}"
        ),
        "calendar_month_filter_is_fixed": True,
        "effective_observation_days": len(included),
        "modis_clipped_days": requested - len(included),
        "effective_first_included_date": format_date(included[0]) if included else None,
        "effective_last_included_date": format_date(included[-1]) if included else None,
    }


# =============================================================================
# Per-AOI window derivation
# =============================================================================
def _experiment_context(aoi: str) -> dict:
    from core.experiment_context import build_experiment_context

    return build_experiment_context(aoi)


def canonical_window_for(aoi: str) -> dict[str, Any]:
    """Canonical predictor/label window of one AOI, from the frozen helper."""
    from src.window_closure_sensitivity import canonical_window

    return canonical_window(_experiment_context(aoi))


def variant_windows_for(
    aoi: str, shifts: Optional[Sequence[int]] = None,
) -> list[dict[str, Any]]:
    """The three shifted predictor windows of one AOI.

    Delegates to `window_closure_sensitivity.build_window_variants`, which is
    the SAME function that produced the already-PASSed Manavgat analysis. It
    asserts duration invariance and `predictor_end < label_start` internally,
    so this analysis cannot disagree with the reference one.
    """
    from src.window_closure_sensitivity import build_window_variants

    return build_window_variants(
        _experiment_context(aoi), tuple(shifts) if shifts is not None else DEFAULT_SHIFTS
    )


def prelabel_censor_interval_for(
    aoi: str, shifts: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """The single pre-label censoring interval shared by every variant of an AOI."""
    from src.window_closure_sensitivity import common_prelabel_interval

    return common_prelabel_interval(variant_windows_for(aoi, shifts))


def _source_config_hash() -> str:
    from pathlib import Path

    from core.paths import PROJECT_ROOT
    from src.step8_large_block_robustness import sha256_file

    return sha256_file(Path(PROJECT_ROOT) / "core" / "regions.py")


def window_date_rows(
    aois: Sequence[str] = SYNTHESIS_AOIS,
    shifts: Optional[Sequence[int]] = None,
) -> list[dict[str, Any]]:
    """The `window_dates.csv` rows: one per AOI x variant.

    By default this covers the three actual AOIs PLUS the read-only Manavgat
    reference, so the reference contract is auditable in the same table.
    """
    source_hash = _source_config_hash()
    rows: list[dict[str, Any]] = []
    for aoi in aois:
        canonical = canonical_window_for(aoi)
        censor = prelabel_censor_interval_for(aoi, shifts)
        for variant in variant_windows_for(aoi, shifts):
            start = variant["predictor_start_date"]
            end = variant["predictor_end_date"]
            bounds = earth_engine_filter_bounds(start, end)
            modis = modis_effective_days(start, end)
            failure: Optional[str] = None
            if variant["duration_days"] != canonical["duration_days"]:
                failure = (
                    f"duration {variant['duration_days']} != canonical "
                    f"{canonical['duration_days']}"
                )
            elif variant["label_start_date"] != canonical["label_start_date"]:
                failure = "label_start moved"
            elif variant["label_end_date"] != canonical["label_end_date"]:
                failure = "label_end moved"
            elif variant["lead_days"] < 1:
                failure = f"lead_days {variant['lead_days']} < 1"
            rows.append({
                "aoi": aoi,
                "aoi_source": (
                    "read_only_reference" if aoi == REFERENCE_AOI else "this_analysis"
                ),
                "variant": variant["variant_id"],
                "shift_days": int(variant["shift_days"]),
                "predictor_start": start,
                "predictor_end": end,
                "predictor_start_inclusive": bounds["predictor_start_inclusive"],
                "predictor_end_inclusive": bounds["predictor_end_inclusive"],
                "earth_engine_filter_start": bounds["earth_engine_filter_start"],
                "earth_engine_filter_end": bounds["earth_engine_filter_end"],
                "earth_engine_end_exclusive": bounds["earth_engine_end_exclusive"],
                "effective_last_included_date": bounds["effective_last_included_date"],
                "calendar_duration_days": bounds["calendar_duration_days"],
                "calendar_days_inclusive": bounds["calendar_days_inclusive"],
                "effective_observation_days": modis["effective_observation_days"],
                "modis_clipped_days": modis["modis_clipped_days"],
                "lead_days": int(variant["lead_days"]),
                "label_start": variant["label_start_date"],
                "label_end": variant["label_end_date"],
                # event_* / gate_* are ALIASES of the label window. Recorded
                # with provenance so they are never mistaken for independently
                # registered canonical dates.
                "event_start": variant["label_start_date"],
                "event_end": variant["label_end_date"],
                "event_date_semantics": LABEL_WINDOW_ALIAS,
                "event_source_field": EVENT_SOURCE_FIELD,
                "gate_start": variant["label_start_date"],
                "gate_end": variant["label_end_date"],
                "gate_date_semantics": LABEL_WINDOW_ALIAS,
                "gate_source_field": GATE_SOURCE_FIELD,
                "independently_registered": False,
                "prelabel_censor_start": censor["common_prelabel_start"],
                "prelabel_censor_end": censor["common_prelabel_end"],
                "modis_policy_id": modis["modis_policy_id"],
                "source_config_path": "core/regions.py::EXPERIMENTS",
                "source_config_hash": source_hash,
                "timezone_semantics": DATE_TIMEZONE_SEMANTICS,
                "date_contract_pass": failure is None,
                "failure_reason": failure,
            })
    return rows


WINDOW_DATE_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "aoi_source", "variant", "shift_days",
    "predictor_start", "predictor_end",
    "predictor_start_inclusive", "predictor_end_inclusive",
    "earth_engine_filter_start", "earth_engine_filter_end",
    "earth_engine_end_exclusive", "effective_last_included_date",
    "calendar_duration_days", "calendar_days_inclusive",
    "effective_observation_days", "modis_clipped_days", "lead_days",
    "label_start", "label_end",
    "event_start", "event_end", "event_date_semantics", "event_source_field",
    "gate_start", "gate_end", "gate_date_semantics", "gate_source_field",
    "independently_registered",
    "prelabel_censor_start", "prelabel_censor_end",
    "modis_policy_id", "source_config_path", "source_config_hash",
    "timezone_semantics", "date_contract_pass", "failure_reason",
)


def assert_date_contract(rows: Sequence[dict[str, Any]]) -> None:
    """Fail closed on any violation of the frozen date contract.

    Checks that no per-row assertion in `build_window_variants` could catch on
    its own: cross-variant duration constancy, exact shift arithmetic, label /
    event / gate invariance, and the end-exclusive semantics.
    """
    by_aoi: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aoi.setdefault(row["aoi"], []).append(row)

    for aoi, aoi_rows in sorted(by_aoi.items()):
        variants = {row["variant"] for row in aoi_rows}
        if variants != set(VARIANTS):
            raise MultiRegionWindowClosureError(
                f"BLOCKER: VARIANT_SET_MISMATCH -- {aoi} has {sorted(variants)}, "
                f"expected {list(VARIANTS)}."
            )
        canonical = next(r for r in aoi_rows if r["variant"] == CANONICAL_VARIANT_ID)
        c_start = parse_date(canonical["predictor_start"])
        c_end = parse_date(canonical["predictor_end"])

        for row in aoi_rows:
            if row["calendar_duration_days"] != canonical["calendar_duration_days"]:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: WINDOW_DURATION_DRIFT -- {aoi} {row['variant']} "
                    f"has duration {row['calendar_duration_days']}, canonical "
                    f"{canonical['calendar_duration_days']}."
                )
            if row["calendar_days_inclusive"] != row["calendar_duration_days"] + 1:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: DATE_SEMANTICS_INCONSISTENT -- {aoi} "
                    f"{row['variant']} inclusive day count is not duration + 1."
                )
            shift = shift_days_of(row["variant"])
            if row["shift_days"] != shift:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: SHIFT_ARITHMETIC_ERROR -- {aoi} {row['variant']} "
                    f"records shift_days={row['shift_days']}, expected {shift}."
                )
            start_delta = (c_start - parse_date(row["predictor_start"])).days
            end_delta = (c_end - parse_date(row["predictor_end"])).days
            if start_delta != shift or end_delta != shift:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: SHIFT_ARITHMETIC_ERROR -- {aoi} {row['variant']}: "
                    f"start moved {start_delta}d, end moved {end_delta}d, "
                    f"expected {shift}d for both."
                )
            for field, label in (
                ("label_start", "LABEL"), ("label_end", "LABEL"),
                ("event_start", "EVENT"), ("event_end", "EVENT"),
                ("gate_start", "GATE"), ("gate_end", "GATE"),
            ):
                if row[field] != canonical[field]:
                    raise MultiRegionWindowClosureError(
                        f"BLOCKER: {label}_WINDOW_DRIFT -- {aoi} "
                        f"{row['variant']} {field}={row[field]}, canonical "
                        f"{canonical[field]}."
                    )
            if row["event_start"] != row["label_start"] or row["event_end"] != row["label_end"]:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: EVENT_WINDOW_DRIFT -- {aoi} {row['variant']} "
                    "event window is not the label-window alias."
                )
            if row["gate_start"] != row["label_start"] or row["gate_end"] != row["label_end"]:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: GATE_WINDOW_DRIFT -- {aoi} {row['variant']} "
                    "gate window is not the label-window alias."
                )
            if row["independently_registered"] is not False:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: EVENT_GATE_DATE_SEMANTICS_UNRESOLVED -- {aoi} "
                    f"{row['variant']} claims independently registered "
                    "event/gate dates; none exist in the registry."
                )
            if not row["earth_engine_end_exclusive"]:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: DATE_SEMANTICS_INCONSISTENT -- {aoi} "
                    f"{row['variant']} does not record end-exclusive filterDate."
                )
            expected_last = format_date(
                parse_date(row["predictor_end"]) - timedelta(days=1)
            )
            if row["effective_last_included_date"] != expected_last:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: DATE_OFF_BY_ONE -- {aoi} {row['variant']} "
                    f"effective_last_included_date={row['effective_last_included_date']}, "
                    f"expected {expected_last}."
                )
            if row["lead_days"] < 1:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: DATE_OFF_BY_ONE -- {aoi} {row['variant']} has "
                    f"lead_days={row['lead_days']}; predictor_end must precede "
                    "label_start."
                )
            if not row["date_contract_pass"]:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: DATE_CONTRACT_FAILED -- {aoi} {row['variant']}: "
                    f"{row['failure_reason']}."
                )


def modis_clipping_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """`{aoi: {variant: clipped_days}}` -- the cross-AOI support asymmetry.

    Reported because the shifted variants are NOT mechanistically identical
    across AOIs: for some the shift also reduces effective MODIS support.
    """
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        out.setdefault(row["aoi"], {})[row["variant"]] = int(row["modis_clipped_days"])
    return out
