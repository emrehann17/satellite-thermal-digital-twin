"""Cross-report consistency validation.

The same (experiment_id, feature) or (experiment_a, experiment_b, feature)
result may be derivable from more than one discovered canonical Step9G pair
report (e.g. both directory orderings of the same unordered pair, or a
region appearing in several different pairs). This module requires exact or
strict-tolerance agreement across every contributing report and FAILS
CLEARLY on disagreement -- it never silently picks one conflicting result.
"""
from __future__ import annotations

from typing import Any

TOLERANCE = 1e-12

REGION_NUMERIC_KEYS = ("auc", "ci_low", "ci_high", "n_rows", "n_burned", "n_unburned", "n_large_blocks")
REGION_CATEGORICAL_KEYS = ("direction", "support_status", "primary_population")

PAIR_NUMERIC_KEYS = (
    "experiment_a_auc", "experiment_b_auc",
    "auc_difference", "auc_difference_ci_low", "auc_difference_ci_high",
)
PAIR_CATEGORICAL_KEYS = (
    "point_direction_reversal", "reversal_status", "bootstrap_supported_direction_reversal",
)


class ConsistencyError(ValueError):
    """Raised when repeated canonical Step9G pair reports disagree for the
    same region-feature or pair-feature result."""


def _numbers_close(a: Any, b: Any, tol: float = TOLERANCE) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def _merge(
    all_parsed: list[dict[str, Any]],
    records_key: str,
    numeric_keys: tuple[str, ...],
    categorical_keys: tuple[str, ...],
) -> dict[Any, dict[str, Any]]:
    merged: dict[Any, dict[str, Any]] = {}
    contributors: dict[Any, list[str]] = {}
    for parsed in all_parsed:
        pair_id = parsed["pair_id"]
        for key, record in parsed[records_key].items():
            if key not in merged:
                merged[key] = dict(record)
                contributors[key] = [pair_id]
                continue
            existing = merged[key]
            for numeric_key in numeric_keys:
                if not _numbers_close(existing.get(numeric_key), record.get(numeric_key)):
                    raise ConsistencyError(
                        f"Conflicting '{numeric_key}' for {key}: "
                        f"{existing.get(numeric_key)!r} (from {contributors[key]}) vs "
                        f"{record.get(numeric_key)!r} (from {pair_id})."
                    )
            for cat_key in categorical_keys:
                if existing.get(cat_key) != record.get(cat_key):
                    raise ConsistencyError(
                        f"Conflicting '{cat_key}' for {key}: "
                        f"{existing.get(cat_key)!r} (from {contributors[key]}) vs "
                        f"{record.get(cat_key)!r} (from {pair_id})."
                    )
            contributors[key].append(pair_id)
    for key, record in merged.items():
        record["source_pair_reports"] = sorted(contributors[key])
    return merged


def merge_region_records(all_parsed: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Dedupe (experiment_id, feature) records across every discovered pair
    report, requiring strict agreement. Never fabricates or arbitrarily
    picks a value; raises ConsistencyError on any disagreement."""
    return _merge(all_parsed, "region_records", REGION_NUMERIC_KEYS, REGION_CATEGORICAL_KEYS)


def merge_pair_records(all_parsed: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Dedupe (experiment_a, experiment_b, feature) records, requiring
    strict agreement when both directory orderings exist for the same
    unordered pair."""
    return _merge(all_parsed, "pair_records", PAIR_NUMERIC_KEYS, PAIR_CATEGORICAL_KEYS)
