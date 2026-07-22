"""Frozen-output resolvers for the multi-AOI transfer synthesis.

Every function here READS ONLY already-frozen files on disk (JSON/CSV). None
of them run modeling, adaptation, prediction, or bootstrap code, and none of
them write anything.

Each resolver returns a `(record, raw_payload)` tuple where `record` is a
plain dict describing the resolved input (see `ResolvedInput` fields below)
and `raw_payload` is the parsed JSON (or a small dict of parsed JSON/CSV
payloads for input families that span multiple files). `schema_adapters.py`
is the only module that interprets the *content* of `raw_payload`; this
module is only responsible for *finding* the right frozen file(s) and
recording their provenance.

Resolution method vocabulary (`record["resolution_method"]`):
    "generic_per_experiment_schema"  -- Step8 big-block robustness v2 style.
    "legacy_pair_relative_schema"    -- Step8 large-block robustness legacy.
    "step9d_merged_report"           -- Step9 raw transfer, step9d preferred.
    "step9b_step9c_fallback"         -- Step9 raw transfer, step9d absent.
    "direct_pair_directory"          -- Step9E / Step9G / Step10 direct match.
    "reverse_pair_directory"         -- pair keyed by the reverse-order token.

No AOI name, path, or pair is ever hardcoded here -- all paths are built
from the `experiment_id` / `source_id` / `target_id` arguments supplied by
the caller.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from core.paths import PROJECT_ROOT
from core.regions import get_experiment_output_root

PRIMARY_POPULATION = "burnable_tree_shrub_grass"


class InputResolutionError(Exception):
    """Raised when a required frozen input cannot be found, is ambiguous,
    or is internally inconsistent. Carries a `family` and `detail` for
    fail-fast diagnostics; str() gives a human-readable message."""

    def __init__(self, family: str, detail: str):
        self.family = family
        self.detail = detail
        super().__init__(f"[{family}] {detail}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _record(
    family: str,
    path: Path,
    schema_version: Optional[str],
    analysis_id: Optional[str],
    resolution_method: str,
    source_experiment_id: Optional[str] = None,
    target_experiment_id: Optional[str] = None,
    primary_population: Optional[str] = None,
) -> dict:
    return {
        "family": family,
        "path": _rel(path),
        "absolute_path": str(path),
        "schema_version": schema_version,
        "analysis_id": analysis_id,
        "source_experiment_id": source_experiment_id,
        "target_experiment_id": target_experiment_id,
        "primary_population": primary_population,
        "resolution_method": resolution_method,
        "content_hash": _sha256_file(path),
    }


def _cross_region_pair_dir(a: str, b: str) -> Path:
    return PROJECT_ROOT / "outputs" / "cross_region" / f"{a}__{b}"


def _find_existing_pair_dir(base: Path, a: str, b: str, stage: str) -> Optional[tuple[Path, str]]:
    """Look for `<base>/<a>__<b>/<stage>` then `<base>/<b>__<a>/<stage>`.
    Returns (dir_path, resolution_method) for whichever exists first."""
    forward = base / f"{a}__{b}" / stage
    if forward.is_dir():
        return forward, "direct_pair_directory"
    reverse = base / f"{b}__{a}" / stage
    if reverse.is_dir():
        return reverse, "reverse_pair_directory"
    return None


# =============================================================================
# 1. Step8 within-region baseline/thermal performance.
# =============================================================================
def resolve_step8_within_region(experiment_id: str) -> tuple[dict, dict]:
    root = get_experiment_output_root(experiment_id)
    path = root / "step8b" / "step8b_model_comparison_metrics.json"
    if not path.is_file():
        raise InputResolutionError(
            "step8_within_region",
            f"Missing Step8B model-comparison metrics for '{experiment_id}': {path}",
        )
    raw = _read_json(path)
    record = _record(
        family="step8_within_region",
        path=path,
        schema_version="step8b_model_comparison_metrics",
        analysis_id=None,
        resolution_method="direct_pair_directory",
        source_experiment_id=experiment_id,
        target_experiment_id=None,
        primary_population=PRIMARY_POPULATION,
    )
    return record, raw


# =============================================================================
# 2. Large-block robustness (generic per-experiment schema preferred, legacy
#    pair-relative schema as fallback).
#
# Two legacy pair-relative sub-schemas are supported, both scoped under
# `outputs/robustness/<family>/<a>__<b>/`:
#
#   "step8_large_block_primary_all_valid"  -- the FORMAL, current-generation
#       pair-global report. Its own `scope.formal_step8b_primary_population_
#       robustness` section carries full per-experiment condition records
#       (analysis_id, delta_roc_auc, ...) for whichever population that
#       particular formal run actually covers. When that population is the
#       required PRIMARY_POPULATION, those records are used directly.
#       Some frozen formal runs cover a different population (e.g.
#       "all_valid") and instead carry a `scope.natural_vegetation_
#       sensitivity_robustness` pointer (population, source_analysis_id,
#       source_output_root) at PRIMARY_POPULATION scope, referencing the
#       older bare-legacy report that was never recomputed under the formal
#       protocol. That pointer is followed (never re-derived) to pull the
#       per-experiment condition records from the referenced report, with
#       the analysis_id cross-checked for integrity.
#   "step8_large_block" (bare)  -- last-resort fallback, used only when no
#       "step8_large_block_primary_all_valid" pair directory exists at all
#       for this experiment against any of the other selected AOIs.
# =============================================================================
def _step10_style_conditions_for_experiment(conditions: list[dict], experiment_id: str) -> list[dict]:
    return [c for c in conditions if c.get("experiment") == experiment_id]


def _extract_primary_all_valid_condition(
    raw: dict, report_path: Path, experiment_id: str,
) -> Optional[tuple[list[dict], str, str, Optional[str]]]:
    """Try to extract PRIMARY_POPULATION per-experiment condition records
    from one `step8_large_block_primary_all_valid` final report, either
    directly (formal section already at PRIMARY_POPULATION) or by following
    its sensitivity-reference pointer to the older report it names.

    Returns (conditions, artifact_scope, extraction_key, analysis_id), or
    None if this report simply does not cover PRIMARY_POPULATION at all
    (not an error -- caller may still fall back further)."""
    scope = raw.get("scope") or {}

    formal = scope.get("formal_step8b_primary_population_robustness") or {}
    if formal.get("population") == PRIMARY_POPULATION:
        conditions = [
            c for c in _step10_style_conditions_for_experiment(formal.get("conditions") or [], experiment_id)
            if c.get("primary_population") == PRIMARY_POPULATION
        ]
        if conditions:
            return (
                conditions,
                "pair_global_formal_primary_population",
                "scope.formal_step8b_primary_population_robustness.conditions",
                conditions[0].get("analysis_id"),
            )

    sensitivity = scope.get("natural_vegetation_sensitivity_robustness") or {}
    if sensitivity.get("population") == PRIMARY_POPULATION:
        source_output_root = sensitivity.get("source_output_root")
        source_analysis_id = sensitivity.get("source_analysis_id")
        if not source_output_root:
            raise InputResolutionError(
                "large_block_robustness",
                f"'{_rel(report_path)}' natural_vegetation_sensitivity_robustness declares "
                f"population='{PRIMARY_POPULATION}' but has no source_output_root to follow.",
            )
        source_dir = Path(source_output_root)
        if not source_dir.is_absolute():
            source_dir = PROJECT_ROOT / source_dir
        if not source_dir.is_dir():
            raise InputResolutionError(
                "large_block_robustness",
                f"natural_vegetation_sensitivity_robustness source_output_root referenced by "
                f"'{_rel(report_path)}' does not exist: '{source_dir}'.",
            )
        source_reports = sorted(source_dir.glob("*_final_report.json"))
        if not source_reports:
            raise InputResolutionError(
                "large_block_robustness",
                f"No '*_final_report.json' found under natural_vegetation_sensitivity_robustness "
                f"source_output_root '{source_dir}' (referenced by '{_rel(report_path)}').",
            )
        source_raw = _read_json(source_reports[0])
        if source_analysis_id and source_raw.get("analysis_id") != source_analysis_id:
            raise InputResolutionError(
                "large_block_robustness",
                f"Sensitivity-reference pointer in '{_rel(report_path)}' declares "
                f"source_analysis_id='{source_analysis_id}', but the referenced report "
                f"'{source_reports[0]}' has analysis_id='{source_raw.get('analysis_id')}'.",
            )
        conditions = [
            c for c in _step10_style_conditions_for_experiment(
                source_raw.get("new_large_block_conditions") or [], experiment_id,
            )
            if not source_analysis_id or c.get("analysis_id") == source_analysis_id
        ]
        if conditions:
            return (
                conditions,
                "pair_global_sensitivity_reference_pointer",
                "scope.natural_vegetation_sensitivity_robustness->new_large_block_conditions",
                source_analysis_id or source_raw.get("analysis_id"),
            )

    return None


def resolve_large_block_robustness(
    experiment_id: str, other_experiment_ids: list[str]
) -> tuple[Optional[dict], Optional[dict], Optional[str]]:
    """Returns (record, raw_payload, unavailable_reason). Exactly one of
    (record/raw_payload) or (unavailable_reason) is non-None."""
    generic_path = (
        get_experiment_output_root(experiment_id)
        / "robustness" / "step8_big_blocks" / "comparison" / "big_block_robustness_summary.json"
    )
    if generic_path.is_file():
        raw = _read_json(generic_path)
        if raw.get("experiment_id") != experiment_id:
            raise InputResolutionError(
                "large_block_robustness",
                f"Generic big-block robustness file at {generic_path} declares "
                f"experiment_id='{raw.get('experiment_id')}', expected '{experiment_id}'.",
            )
        record = _record(
            family="large_block_robustness",
            path=generic_path,
            schema_version=raw.get("report_schema_version"),
            analysis_id=raw.get("analysis_id"),
            resolution_method="generic_per_experiment_schema",
            source_experiment_id=experiment_id,
            primary_population=raw.get("primary_population"),
        )
        record["artifact_scope"] = "per_experiment"
        record["extraction_key"] = "$"
        return record, raw, None

    legacy_root = PROJECT_ROOT / "outputs" / "robustness"
    if not legacy_root.is_dir():
        return None, None, (
            "no_generic_per_experiment_report_and_no_legacy_pair_relative_entry_"
            "with_matching_primary_population"
        )

    # Preferred: the formal `step8_large_block_primary_all_valid` pair-global
    # family, tried before (and instead of) the older bare `step8_large_block`
    # sensitivity report whenever a pair directory for it exists.
    formal_family_dir = legacy_root / "step8_large_block_primary_all_valid"
    for other in other_experiment_ids:
        for pair_token in (f"{experiment_id}__{other}", f"{other}__{experiment_id}"):
            pair_dir = formal_family_dir / pair_token
            if not pair_dir.is_dir():
                continue
            for report_path in sorted(pair_dir.glob("*_final_report.json")):
                raw = _read_json(report_path)
                extracted = _extract_primary_all_valid_condition(raw, report_path, experiment_id)
                if extracted is None:
                    continue
                conditions, artifact_scope, extraction_key, analysis_id = extracted
                record = _record(
                    family="large_block_robustness",
                    path=report_path,
                    schema_version=raw.get("report_schema_version"),
                    analysis_id=analysis_id,
                    resolution_method="legacy_pair_relative_schema:primary_all_valid",
                    source_experiment_id=experiment_id,
                    primary_population=PRIMARY_POPULATION,
                )
                record["artifact_scope"] = artifact_scope
                record["extraction_key"] = extraction_key
                return record, {"conditions": conditions, "full_report": raw}, None

    # Last resort: bare legacy pair-relative report, only reached when no
    # `step8_large_block_primary_all_valid` pair directory exists at all for
    # this experiment against any of the other selected AOIs.
    for family_dir in sorted(legacy_root.glob("step8_large_block*")):
        if family_dir == formal_family_dir or not family_dir.is_dir():
            continue
        for other in other_experiment_ids:
            for pair_token in (f"{experiment_id}__{other}", f"{other}__{experiment_id}"):
                pair_dir = family_dir / pair_token
                if not pair_dir.is_dir():
                    continue
                for report_path in sorted(pair_dir.glob("*_final_report.json")):
                    raw = _read_json(report_path)
                    scope = raw.get("scope", {})
                    frp = scope.get("formal_step8b_primary_population_robustness", {})
                    conditions = frp.get("conditions", [])
                    matching = [
                        c for c in conditions
                        if c.get("experiment") == experiment_id
                        and c.get("primary_population") == PRIMARY_POPULATION
                    ]
                    if matching:
                        record = _record(
                            family="large_block_robustness",
                            path=report_path,
                            schema_version=raw.get("report_schema_version"),
                            analysis_id=matching[0].get("analysis_id"),
                            resolution_method="legacy_pair_relative_schema:bare",
                            source_experiment_id=experiment_id,
                            primary_population=PRIMARY_POPULATION,
                        )
                        record["artifact_scope"] = "pair_global_legacy_bare"
                        record["extraction_key"] = "scope.formal_step8b_primary_population_robustness.conditions"
                        return record, {"conditions": matching, "full_report": raw}, None

    return None, None, (
        "no_generic_per_experiment_report_and_no_legacy_pair_relative_entry_"
        "with_matching_primary_population"
    )


# =============================================================================
# 3. Step9 raw transfer (step9d preferred, step9b/9c fallback).
# =============================================================================
def resolve_step9_transfer(source_id: str, target_id: str) -> tuple[dict, dict]:
    found = _find_existing_pair_dir(
        PROJECT_ROOT / "outputs" / "cross_region", source_id, target_id, "step9d"
    )
    if found is not None:
        stage_dir, method = found
        path = stage_dir / "final_cross_region_report.json"
        if path.is_file():
            raw = _read_json(path)
            record = _record(
                family="step9_transfer",
                path=path,
                schema_version="step9d.final_cross_region_report",
                analysis_id=None,
                resolution_method=f"step9d_merged_report:{method}",
                source_experiment_id=source_id,
                target_experiment_id=target_id,
                primary_population=raw.get("primary_population"),
            )
            return record, raw

    # Fallback: step9b (raw point metrics) + step9c (bootstrap CIs).
    b_found = _find_existing_pair_dir(
        PROJECT_ROOT / "outputs" / "cross_region", source_id, target_id, "step9b"
    )
    c_found = _find_existing_pair_dir(
        PROJECT_ROOT / "outputs" / "cross_region", source_id, target_id, "step9c"
    )
    if b_found is None or c_found is None:
        raise InputResolutionError(
            "step9_transfer",
            f"No Step9D merged report and no complete Step9B+Step9C fallback for "
            f"direction '{source_id}' -> '{target_id}'.",
        )
    b_path = b_found[0] / "cross_region_transfer_metrics.json"
    c_path = c_found[0] / "cross_region_bootstrap_metrics.json"
    if not b_path.is_file() or not c_path.is_file():
        raise InputResolutionError(
            "step9_transfer",
            f"Step9B/Step9C fallback files missing for direction "
            f"'{source_id}' -> '{target_id}' ({b_path}, {c_path}).",
        )
    b_raw = _read_json(b_path)
    c_raw = _read_json(c_path)
    record = _record(
        family="step9_transfer",
        path=b_path,
        schema_version="step9b_step9c_fallback",
        analysis_id=None,
        resolution_method=f"step9b_step9c_fallback:{b_found[1]}",
        source_experiment_id=source_id,
        target_experiment_id=target_id,
        primary_population=b_raw.get("primary_population") or b_raw.get("population"),
    )
    return record, {"step9b": b_raw, "step9c": c_raw}


# =============================================================================
# 4. Step9E shift audit.
#
# Two artifact scopes are supported for the SAME unordered pair:
#   "direction_specific"              -- one file covers exactly one ordered
#                                         direction (root source/target ==
#                                         that direction; any transfer_direction
#                                         fields inside it, if present, name
#                                         only that single direction).
#   "pair_global_with_direction_records" -- one file, written under a fixed
#                                         root source/target order, but whose
#                                         direction-specific sub-tables
#                                         (prediction-distribution audit,
#                                         calibration bins, ...) contain rows
#                                         for BOTH ordered directions, keyed
#                                         by their own `transfer_direction`
#                                         field. Pair-global fields (numeric/
#                                         categorical feature shift,
#                                         label-conditional relationships,
#                                         relationship-direction flips)
#                                         describe the covariate/relationship
#                                         shift between the two regions and
#                                         are reused verbatim for whichever
#                                         direction is requested.
# No AOI is ever named here: the pair-global-vs-direction-specific decision
# is made purely from the artifact's own root metadata and the set of
# `transfer_direction` values found inside it.
# =============================================================================
_STEP9E_PAIR_GLOBAL_FIELDS = (
    "part_a_numeric_feature_shift",
    "part_c_relationship_direction_flips",
)
_STEP9E_DIRECTION_SPECIFIC_FIELDS = (
    "part_d_prediction_distribution_audit",
    "part_e_calibration_bins",
)


def _step9e_directions_present(raw: dict) -> set[str]:
    directions: set[str] = set()
    for field in _STEP9E_DIRECTION_SPECIFIC_FIELDS:
        for row in raw.get(field, []) or []:
            direction = row.get("transfer_direction")
            if direction:
                directions.add(direction)
    return directions


def _extract_step9e_direction_payload(
    raw: dict, source_id: str, target_id: str, directions_present: set[str],
) -> dict:
    """Reorient a Step9E payload to the requested ordered direction. Pair-
    global rows are relabeled (not recomputed) to (source_id, target_id);
    direction-specific rows are filtered down to the requested direction
    only. Never modifies the file on disk -- this builds an in-memory copy
    for schema_adapters.adapt_step9e_shift to consume."""
    requested_direction = f"{source_id}_to_{target_id}"

    def _relabel_pair_global(rows: list[dict]) -> list[dict]:
        relabeled = []
        for row in rows:
            new_row = dict(row)
            if "source_experiment_id" in new_row and "target_experiment_id" in new_row:
                new_row["source_experiment_id"] = source_id
                new_row["target_experiment_id"] = target_id
            relabeled.append(new_row)
        return relabeled

    def _filter_direction_specific(rows: list[dict]) -> list[dict]:
        return [r for r in rows if r.get("transfer_direction") == requested_direction]

    resolved = dict(raw)
    resolved["source_experiment_id"] = source_id
    resolved["target_experiment_id"] = target_id
    for field in _STEP9E_PAIR_GLOBAL_FIELDS:
        if field in resolved and isinstance(resolved[field], list):
            resolved[field] = _relabel_pair_global(resolved[field])
    if directions_present:
        for field in _STEP9E_DIRECTION_SPECIFIC_FIELDS:
            if field in resolved and isinstance(resolved[field], list):
                resolved[field] = _filter_direction_specific(resolved[field])
    return resolved


def _find_step9e_artifacts(namespace_dir: Path) -> list[Path]:
    """All distribution_shift_audit.json files under a single pair namespace
    (normally exactly one; rglob guards against accidental duplicates so a
    real duplicate can be caught as a conflict instead of silently picked)."""
    step9e_dir = namespace_dir / "step9e"
    if not step9e_dir.is_dir():
        return []
    return sorted(step9e_dir.rglob("distribution_shift_audit.json"))


def _resolve_step9e_in_namespace(
    base: Path, namespace: str, source_id: str, target_id: str,
    requested_pair: frozenset, requested_direction: str,
) -> Optional[tuple[Path, dict, str, set[str]]]:
    """Look for a valid Step9E artifact for `requested_direction` strictly
    within `namespace` (no cross-namespace comparison). Returns
    (path, raw, scope, directions_present) for the single agreeing artifact,
    or None if this namespace has no candidate files, or none of its
    candidate files carry records for the requested direction (both of
    which mean "try the next namespace in the precedence order", NOT a
    conflict). Raises InputResolutionError for root-metadata mismatches or
    for multiple *disagreeing* candidates within this SAME namespace."""
    candidate_paths = _find_step9e_artifacts(base / namespace)
    if not candidate_paths:
        return None

    hits: list[tuple[Path, dict, str, set[str]]] = []
    for path in candidate_paths:
        raw = _read_json(path)
        root_source = raw.get("source_experiment_id")
        root_target = raw.get("target_experiment_id")
        root_pair = frozenset((root_source, root_target)) if root_source and root_target else None
        if root_pair != requested_pair:
            raise InputResolutionError(
                "step9e_shift_audit",
                f"Step9E artifact at '{_rel(path)}' has root metadata "
                f"source_experiment_id='{root_source}', "
                f"target_experiment_id='{root_target}', which does not match "
                f"the requested unordered pair {{'{source_id}', '{target_id}'}}.",
            )

        directions_present = _step9e_directions_present(raw)
        if requested_direction not in directions_present:
            # This artifact's root metadata matches the pair, but it carries
            # no records for the requested ordered direction -- not usable
            # from this namespace; try the next namespace in precedence.
            continue
        scope = "pair_global_with_direction_records" if len(directions_present) > 1 else "direction_specific"
        hits.append((path, raw, scope, directions_present))

    if not hits:
        return None

    if len(hits) > 1:
        reference_path, reference_raw, _, reference_dirs = hits[0]
        reference_payload = _extract_step9e_direction_payload(
            reference_raw, source_id, target_id, reference_dirs
        )
        for other_path, other_raw, _, other_dirs in hits[1:]:
            other_payload = _extract_step9e_direction_payload(
                other_raw, source_id, target_id, other_dirs
            )
            if other_payload != reference_payload:
                raise InputResolutionError(
                    "step9e_shift_audit",
                    f"Multiple Step9E artifacts inside namespace '{namespace}' "
                    f"conflict for direction '{requested_direction}': "
                    f"'{_rel(reference_path)}' and '{_rel(other_path)}' "
                    "disagree numerically.",
                )

    return hits[0]


def resolve_step9e_shift(source_id: str, target_id: str) -> tuple[dict, dict]:
    """Resolve the Step9E shift-audit artifact for ordered direction
    `source_id -> target_id` using a strict precedence:

        1. exact directional namespace   '<source_id>__<target_id>'
        2. legacy reverse namespace       '<target_id>__<source_id>'
           (fallback only)

    The exact namespace is tried in complete isolation first; if it yields
    a valid artifact for the requested direction, it is returned
    immediately and is NEVER numerically compared against the reverse
    namespace (modern Step9E outputs may legitimately exist under both
    namespaces with different, direction-dependent diagnostics -- that is
    not a conflict). The reverse namespace is consulted only when the exact
    namespace has no candidate files, or its candidate file(s) carry no
    records for the requested direction (the legacy pair-global case)."""
    base = PROJECT_ROOT / "outputs" / "cross_region"
    requested_pair = frozenset((source_id, target_id))
    requested_direction = f"{source_id}_to_{target_id}"

    exact_namespace = f"{source_id}__{target_id}"
    fallback_namespace = f"{target_id}__{source_id}"

    resolution_precedence = ["exact_directional_namespace"]
    exact_hit = _resolve_step9e_in_namespace(
        base, exact_namespace, source_id, target_id, requested_pair, requested_direction,
    )
    if exact_hit is not None:
        path, raw, scope, directions_present = exact_hit
        resolution_mode = "exact_directional_namespace"
        namespace = exact_namespace
    else:
        resolution_precedence.append("legacy_reverse_namespace_fallback")
        fallback_hit = _resolve_step9e_in_namespace(
            base, fallback_namespace, source_id, target_id, requested_pair, requested_direction,
        )
        if fallback_hit is None:
            raise InputResolutionError(
                "step9e_shift_audit",
                f"No Step9E shift audit found for ordered direction "
                f"'{requested_direction}' (checked exact namespace "
                f"'{exact_namespace}', then legacy fallback namespace "
                f"'{fallback_namespace}'; neither contains a valid artifact "
                f"with records for this direction).",
            )
        path, raw, scope, directions_present = fallback_hit
        resolution_mode = "legacy_reverse_namespace_fallback"
        namespace = fallback_namespace

    primary_populations = raw.get("primary_populations") or []
    if PRIMARY_POPULATION not in primary_populations:
        raise InputResolutionError(
            "step9e_shift_audit",
            f"Step9E artifact at '{_rel(path)}' does not list the required "
            f"primary population '{PRIMARY_POPULATION}' among "
            f"primary_populations={primary_populations}.",
        )

    resolved_raw = _extract_step9e_direction_payload(raw, source_id, target_id, directions_present)

    record = _record(
        family="step9e_shift_audit",
        path=path,
        schema_version="step9e.distribution_shift_audit",
        analysis_id=None,
        resolution_method=(
            "direct_pair_directory" if resolution_mode == "exact_directional_namespace"
            else "reverse_pair_directory"
        ),
        source_experiment_id=source_id,
        target_experiment_id=target_id,
        primary_population=PRIMARY_POPULATION,
    )
    record["artifact_scope"] = scope
    record["requested_logical_direction"] = requested_direction
    record["resolution_precedence"] = resolution_precedence
    record["resolution_mode"] = resolution_mode
    record["resolved_namespace"] = namespace
    record["resolved_path"] = _rel(path)
    record["schema"] = (
        "step9e.distribution_shift_audit.pair_global_with_direction_records"
        if scope == "pair_global_with_direction_records"
        else "step9e.distribution_shift_audit.direction_specific"
    )
    record["root_source_experiment_id"] = raw.get("source_experiment_id")
    record["root_target_experiment_id"] = raw.get("target_experiment_id")

    return record, resolved_raw


# =============================================================================
# 5. Step9G feature-AUC direction-reversal (v2 integration preferred, v1
#    fallback). Unordered pair; both directory-name orders are tried.
# =============================================================================
def resolve_step9g_pair(experiment_a: str, experiment_b: str) -> tuple[dict, dict]:
    base_v2 = PROJECT_ROOT / "outputs" / "diagnostics" / "step9g_univariate_feature_auc_direction_reversal_integration_v2"
    for token, method in ((f"{experiment_a}__{experiment_b}", "direct_pair_directory"),
                          (f"{experiment_b}__{experiment_a}", "reverse_pair_directory")):
        path = base_v2 / token / "step9g_integration_correction_final_report.json"
        if path.is_file():
            raw = _read_json(path)
            record = _record(
                family="step9g_feature_reversal",
                path=path,
                schema_version=raw.get("schema_version"),
                analysis_id=raw.get("analysis_id"),
                resolution_method=f"step9g_v2_integration:{method}",
                source_experiment_id=experiment_a,
                target_experiment_id=experiment_b,
                primary_population=raw.get("primary_population"),
            )
            return record, {"schema": "v2", "report": raw}

    base_v1 = PROJECT_ROOT / "outputs" / "diagnostics" / "step9g_univariate_feature_auc_direction_reversal"
    for token, method in ((f"{experiment_a}__{experiment_b}", "direct_pair_directory"),
                          (f"{experiment_b}__{experiment_a}", "reverse_pair_directory")):
        path = base_v1 / token / "step9g_final_report.json"
        if path.is_file():
            raw = _read_json(path)
            record = _record(
                family="step9g_feature_reversal",
                path=path,
                schema_version=raw.get("schema_version"),
                analysis_id=raw.get("analysis_id"),
                resolution_method=f"step9g_v1:{method}",
                source_experiment_id=experiment_a,
                target_experiment_id=experiment_b,
                primary_population=raw.get("primary_population"),
            )
            return record, {"schema": "v1", "report": raw}

    raise InputResolutionError(
        "step9g_feature_reversal",
        f"No Step9G v2-integration or v1 report found for unordered pair "
        f"{{'{experiment_a}', '{experiment_b}'}}.",
    )


# =============================================================================
# 6. Step10 raw/regionwise_zscore/coral_after_regionwise_zscore pair report.
# =============================================================================
def _resolve_step10_primary_population(raw: dict, step10_dir: Path, report_path: Path) -> str:
    """Derive Step10's primary population from the frozen final report itself
    and/or its sibling frozen preregistration in the same Step10 family.
    Never returns None/null: fails fast instead, either because the
    population truly cannot be found anywhere in the frozen family, or
    because two frozen sources of the same family disagree."""
    candidates: list[tuple[str, str]] = []
    for key in ("primary_population", "population"):
        value = raw.get(key)
        if value:
            candidates.append((f"step10_final_report.{key}", value))

    prereg_path = step10_dir / "step10_preregistration.json"
    if prereg_path.is_file():
        prereg = _read_json(prereg_path)
        value = (prereg.get("scientific_config") or {}).get("primary_population")
        if value:
            candidates.append(("step10_preregistration.scientific_config.primary_population", value))

    if not candidates:
        raise InputResolutionError(
            "step10_pair",
            f"Could not determine primary_population for the frozen Step10 family at "
            f"'{_rel(report_path)}' (checked the final report and its sibling preregistration).",
        )

    distinct_populations = {population for _, population in candidates}
    if len(distinct_populations) > 1:
        raise InputResolutionError(
            "step10_pair",
            f"Conflicting primary_population values found within the frozen Step10 family at "
            f"'{_rel(report_path)}': {candidates}.",
        )

    resolved_population = next(iter(distinct_populations))
    if resolved_population != PRIMARY_POPULATION:
        raise InputResolutionError(
            "step10_pair",
            f"Frozen Step10 family at '{_rel(report_path)}' declares primary_population="
            f"'{resolved_population}', expected '{PRIMARY_POPULATION}'.",
        )
    return resolved_population


def resolve_step10_pair(experiment_a: str, experiment_b: str) -> tuple[dict, dict]:
    base = PROJECT_ROOT / "outputs" / "cross_region"
    for token, method in ((f"{experiment_a}__{experiment_b}", "direct_pair_directory"),
                          (f"{experiment_b}__{experiment_a}", "reverse_pair_directory")):
        step10_dir = base / token / "step10"
        path = step10_dir / "step10_final_report.json"
        if path.is_file():
            raw = _read_json(path)
            expected_directions = {
                f"{experiment_a}_to_{experiment_b}", f"{experiment_b}_to_{experiment_a}",
            }
            found_directions = set(raw.get("directions") or [])
            if not expected_directions.issubset(found_directions):
                raise InputResolutionError(
                    "step10_pair",
                    f"Step10 report at {path} does not cover both directions of pair "
                    f"{{'{experiment_a}', '{experiment_b}'}} (found directions={sorted(found_directions)}).",
                )
            primary_population = _resolve_step10_primary_population(raw, step10_dir, path)
            record = _record(
                family="step10_pair",
                path=path,
                schema_version=raw.get("report_schema_version"),
                analysis_id=raw.get("analysis_id"),
                resolution_method=method,
                source_experiment_id=experiment_a,
                target_experiment_id=experiment_b,
                primary_population=primary_population,
            )
            return record, raw
    raise InputResolutionError(
        "step10_pair",
        f"No Step10 pair report found for unordered pair {{'{experiment_a}', '{experiment_b}'}}.",
    )
