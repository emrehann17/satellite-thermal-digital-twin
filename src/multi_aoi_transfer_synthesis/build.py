"""Top-level orchestration for the multi-AOI transfer synthesis:

    validate AOI set
    -> resolve all frozen inputs (6 families x required directions/pairs)
    -> adapt each resolved input into the normalized internal representation
    -> derive conservative status labels
    -> assemble the normalized synthesis object
    -> (if not dry-run) render outputs

This module NEVER runs modeling, adaptation, prediction, or bootstrap code;
it only reads frozen files (via `resolvers.py`) and (when not a dry run)
writes the synthesis report files (via `render.py` / `manifest.py`).

Normalized synthesis object (the single dict this module assembles and
`render.py`/`manifest.py` consume) has this shape:

    {
        "generated_at": iso8601 str,
        "canonical_set_id": str,
        "primary_population": "burnable_tree_shrub_grass",
        "aois_display_order": [str, ...],
        "aois_canonical_order": [str, ...],
        "ordered_directions": [[source, target], ...],
        "unordered_pairs": [[a, b], ...],
        "within_region": [ {experiment_id, model_family, roc_auc, pr_auc,
            thermal_minus_baseline_roc_auc, thermal_minus_baseline_pr_auc,
            large_block_robustness_available, large_block_support_status,
            effect_magnitude_stability_status, large_block_unavailable_reason}, ...],
        "transfer_matrix": [ {source_experiment_id, target_experiment_id,
            model_family, adaptation_method, primary_population,
            target_row_count, target_burned_count, roc_auc, roc_auc_ci_low,
            roc_auc_ci_high, roc_auc_chance_status, pr_auc, pr_auc_ci_low,
            pr_auc_ci_high, raw_transfer_status, adapted_minus_raw,
            adapted_minus_raw_ci_low, adapted_minus_raw_ci_high,
            adapted_minus_raw_support_status, within_region_target_metric,
            remaining_transfer_gap, remaining_transfer_gap_ci_low,
            remaining_transfer_gap_ci_high, residual_gap_status,
            step9e_shift_categories, step9e_ranking_reversal_value,
            step9e_ranking_reversal_scope, brier, brier_availability,
            brier_protocol_source}, ...],
            # NOTE: `roc_auc_chance_status` (this row's own ROC-AUC vs.
            # chance) and `adapted_minus_raw_support_status` (adapted vs.
            # raw) are deliberately separate fields -- a positive supported
            # adapted-minus-raw delta does NOT imply above-chance adapted
            # performance, and neither implies the other.
        "feature_stability": [ {experiment_a, experiment_b, feature,
            experiment_a_auc, experiment_a_ci_low, experiment_a_ci_high,
            experiment_a_direction, experiment_b_auc, experiment_b_ci_low,
            experiment_b_ci_high, experiment_b_direction,
            point_direction_reversal, reversal_status,
            step9e_relationship_direction_flag}, ...],
        "summaries": { ...generic, result-driven statements... },
        "scientific_boundaries": [str, ...],
        "resolved_inputs": [ ...resolver records... ],
    }
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.multi_aoi_transfer_synthesis import resolvers, schema_adapters, status_derivation
from src.multi_aoi_transfer_synthesis.aoi_set import AoiSetError, canonicalize_pair, validate_aoi_set
from src.multi_aoi_transfer_synthesis.resolvers import InputResolutionError, PRIMARY_POPULATION

MODEL_FAMILIES = ("baseline", "thermal")

SCIENTIFIC_BOUNDARIES = (
    "Within-region performance does not imply cross-region transportability.",
    "Transferability may be pair-specific and direction-dependent.",
    "Unsupervised covariate alignment may help some directions and harm others.",
    "Residual gaps after adaptation are consistent with relationship shift or "
    "other non-covariate differences between regions, not proof of a specific cause.",
    "Feature-level univariate direction reversals are diagnostic evidence of "
    "possible concept/relationship shift, not causal proof.",
    "One event per AOI does not allow region effects and event effects to be isolated.",
    "Target labels are used only for diagnostic evaluation; they are never used "
    "for adaptation, normalization, calibration, or model selection.",
    "No operational or universal cross-region transfer claim is supported by this report.",
)


class SynthesisBuildError(Exception):
    """Raised for fail-fast conditions during synthesis assembly (missing
    inputs, duplicate directions/pairs, inconsistent metadata, etc.)."""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect(dry_run: bool, errors: list[dict], family: str, direction_or_pair, fn):
    """Run a resolver/adapter callable; on InputResolutionError, either raise
    immediately (non-dry-run: fail fast) or record the failure and return
    None (dry-run: surface as a diagnostic)."""
    try:
        return fn()
    except InputResolutionError as exc:
        if not dry_run:
            raise SynthesisBuildError(str(exc)) from exc
        errors.append({
            "family": family,
            "direction_or_pair": direction_or_pair,
            "detail": exc.detail,
        })
        return None


def build_synthesis(
    aois: list[str], dry_run: bool = False, output_root: Optional[str] = None,
) -> dict:
    """Resolve, adapt, and assemble the normalized multi-AOI transfer
    synthesis object. Does not write anything -- callers decide whether/how
    to render (see `run_multi_aoi_transfer_synthesis_stage` in
    core/pipeline_orchestrator.py)."""
    try:
        aoi_set = validate_aoi_set(aois)
    except AoiSetError as exc:
        raise SynthesisBuildError(str(exc)) from exc

    ordered_directions = aoi_set.ordered_directions()
    unordered_pairs = aoi_set.unordered_pairs()

    # Fail fast on internal duplicate directions/pairs (defensive; the
    # generic generators cannot actually produce duplicates for a validated
    # AoiSet, but this guards against future refactors silently reintroducing
    # duplicates).
    if len(set(ordered_directions)) != len(ordered_directions):
        raise SynthesisBuildError("Duplicate ordered direction detected during generation.")
    if len(set(unordered_pairs)) != len(unordered_pairs):
        raise SynthesisBuildError("Duplicate unordered pair detected during generation.")

    resolution_errors: list[dict] = []
    resolved_inputs: list[dict] = []

    # -------------------------------------------------------------------
    # 1 & 2: Step8 within-region + large-block robustness, per experiment.
    # -------------------------------------------------------------------
    within_region_normalized: dict[str, dict] = {}
    large_block_normalized: dict[str, dict] = {}
    for experiment_id in aoi_set.canonical_order:
        def _resolve_step8(experiment_id=experiment_id):
            record, raw = resolvers.resolve_step8_within_region(experiment_id)
            normalized = schema_adapters.adapt_step8_within_region(raw, experiment_id)
            resolved_inputs.append(record)
            return normalized

        result = _collect(dry_run, resolution_errors, "step8_within_region", experiment_id, _resolve_step8)
        if result is not None:
            within_region_normalized[experiment_id] = result

        others = [e for e in aoi_set.canonical_order if e != experiment_id]

        def _resolve_large_block(experiment_id=experiment_id, others=others):
            record, raw, unavailable_reason = resolvers.resolve_large_block_robustness(experiment_id, others)
            if record is None:
                return schema_adapters.unavailable_large_block_robustness(unavailable_reason)
            resolved_inputs.append(record)
            if record["resolution_method"] == "generic_per_experiment_schema":
                return schema_adapters.adapt_large_block_robustness_generic(raw, experiment_id)
            return schema_adapters.adapt_large_block_robustness_legacy(raw, experiment_id)

        lb_result = _collect(dry_run, resolution_errors, "large_block_robustness", experiment_id, _resolve_large_block)
        # Large-block robustness is optional data (spec: never required); an
        # exception here would only occur for a genuine adapter bug, but if
        # dry-run swallowed it above, fall back to "unavailable" rather than
        # dropping the experiment entirely.
        large_block_normalized[experiment_id] = lb_result or schema_adapters.unavailable_large_block_robustness(
            "resolution_failed_during_dry_run"
        )

    # -------------------------------------------------------------------
    # 3 & 4: Step9 raw transfer + Step9E shift audit, per ordered direction.
    # -------------------------------------------------------------------
    transfer_normalized: dict[tuple[str, str], dict] = {}
    shift_normalized: dict[tuple[str, str], dict] = {}
    for source_id, target_id in ordered_directions:
        def _resolve_transfer(source_id=source_id, target_id=target_id):
            record, raw = resolvers.resolve_step9_transfer(source_id, target_id)
            resolved_inputs.append(record)
            if record["schema_version"] == "step9b_step9c_fallback":
                return schema_adapters.adapt_step9_transfer_fallback(raw, source_id, target_id)
            return schema_adapters.adapt_step9_transfer_step9d(raw, source_id, target_id)

        result = _collect(dry_run, resolution_errors, "step9_transfer", (source_id, target_id), _resolve_transfer)
        if result is not None:
            transfer_normalized[(source_id, target_id)] = result

        def _resolve_shift(source_id=source_id, target_id=target_id):
            record, raw = resolvers.resolve_step9e_shift(source_id, target_id)
            resolved_inputs.append(record)
            return schema_adapters.adapt_step9e_shift(raw, source_id, target_id)

        shift_result = _collect(dry_run, resolution_errors, "step9e_shift_audit", (source_id, target_id), _resolve_shift)
        if shift_result is not None:
            shift_normalized[(source_id, target_id)] = shift_result

    # -------------------------------------------------------------------
    # 5 & 6: Step9G feature reversal + Step10, per unordered pair.
    # -------------------------------------------------------------------
    step9g_normalized: dict[tuple[str, str], dict] = {}
    step10_normalized: dict[tuple[str, str], dict] = {}
    for a, b in unordered_pairs:
        def _resolve_9g(a=a, b=b):
            record, payload = resolvers.resolve_step9g_pair(a, b)
            resolved_inputs.append(record)
            return schema_adapters.adapt_step9g_pair(payload, a, b)

        g_result = _collect(dry_run, resolution_errors, "step9g_feature_reversal", (a, b), _resolve_9g)
        if g_result is not None:
            step9g_normalized[(a, b)] = g_result

        def _resolve_10(a=a, b=b):
            record, raw = resolvers.resolve_step10_pair(a, b)
            resolved_inputs.append(record)
            return schema_adapters.adapt_step10_pair(raw, a, b)

        r10_result = _collect(dry_run, resolution_errors, "step10_pair", (a, b), _resolve_10)
        if r10_result is not None:
            step10_normalized[(a, b)] = r10_result

    if dry_run:
        return {
            "dry_run": True,
            "generated_at": _iso_now(),
            "canonical_set_id": aoi_set.canonical_set_id,
            "primary_population": PRIMARY_POPULATION,
            "aois_display_order": list(aoi_set.display_order),
            "aois_canonical_order": list(aoi_set.canonical_order),
            "ordered_directions": [list(d) for d in ordered_directions],
            "unordered_pairs": [list(p) for p in unordered_pairs],
            "resolved_inputs": resolved_inputs,
            "resolution_errors": resolution_errors,
            "output_root": output_root,
        }

    # -------------------------------------------------------------------
    # Assemble within_region rows.
    # -------------------------------------------------------------------
    within_region_rows = []
    for experiment_id in aoi_set.canonical_order:
        w = within_region_normalized[experiment_id]
        lb = large_block_normalized[experiment_id]
        delta_roc = w["thermal"]["roc_auc"] - w["baseline"]["roc_auc"]
        delta_pr = w["thermal"]["pr_auc"] - w["baseline"]["pr_auc"]
        for model_family in MODEL_FAMILIES:
            within_region_rows.append({
                "experiment_id": experiment_id,
                "model_family": model_family,
                "roc_auc": w[model_family]["roc_auc"],
                "pr_auc": w[model_family]["pr_auc"],
                "thermal_minus_baseline_roc_auc": delta_roc,
                "thermal_minus_baseline_pr_auc": delta_pr,
                "large_block_robustness_available": lb["available"],
                "large_block_support_status": lb["support_status"],
                "effect_magnitude_stability_status": lb["effect_magnitude_stability_status"],
                "large_block_unavailable_reason": lb["unavailable_reason"],
            })

    # -------------------------------------------------------------------
    # Assemble transfer_matrix rows from Step10 (primary source), enriched
    # with Step9-derived raw_transfer_status and Step9E-derived shift fields.
    # -------------------------------------------------------------------
    transfer_matrix_rows = []
    for a, b in unordered_pairs:
        step10 = step10_normalized[(a, b)]
        for source_id, target_id in ((a, b), (b, a)):
            direction_key = f"{source_id}_to_{target_id}"
            shift = shift_normalized.get((source_id, target_id))
            if shift is not None:
                local_shift_categories = status_derivation.derive_shift_categories(shift["numeric_feature_rows"])
                shift_categories = status_derivation.merge_shift_categories(
                    local_shift_categories, shift["frozen_diagnosis_categories"],
                )
                ranking_reversal_value, ranking_reversal_scope = status_derivation.derive_ranking_reversal(
                    shift["direction_specific_ranking_reversal_rows"],
                    shift["pair_global_ranking_reversal_suspected"],
                )
            else:
                shift_categories = [status_derivation.SHIFT_AUDIT_UNAVAILABLE]
                ranking_reversal_value, ranking_reversal_scope = None, None

            transfer = transfer_normalized.get((source_id, target_id))

            for model_family in MODEL_FAMILIES:
                if transfer is not None:
                    metric_ci = transfer["model_family_metrics"][model_family]
                    raw_status = status_derivation.derive_raw_transfer_status(
                        metric_ci["roc_auc_ci_low"], metric_ci["pr_auc_ci_low"],
                    )
                else:
                    raw_status = None

                rows_for_direction_family = [
                    r for r in step10["rows"]
                    if r["direction"] == direction_key and r["model_family"] == model_family
                ]
                for row in rows_for_direction_family:
                    adaptation_method = row["adaptation_method"]
                    adapted_minus_raw = ci_low = ci_high = None
                    adapted_minus_raw_support_status = None
                    within_region_target_metric = remaining_gap = gap_low = gap_high = None
                    residual_gap_status = None

                    if adaptation_method != "raw_source_only":
                        diff_entry = next(
                            (d for d in step10["adaptation_differences"]
                             if d["direction"] == direction_key and d["model_family"] == model_family
                             and d["adaptation_method"] == adaptation_method and d["metric"] == "roc_auc"),
                            None,
                        )
                        if diff_entry is not None:
                            adapted_minus_raw = diff_entry["point_estimate_difference"]
                            ci_low = diff_entry["ci_low"]
                            ci_high = diff_entry["ci_high"]
                            adapted_minus_raw_support_status = status_derivation.derive_adaptation_support_status(
                                ci_low, ci_high,
                            )

                        decomp_entry = next(
                            (d for d in step10["decomposition"]
                             if d["direction"] == direction_key and d["model_family"] == model_family
                             and d["adaptation_method"] == adaptation_method and d["metric"] == "roc_auc"),
                            None,
                        )
                        if decomp_entry is not None:
                            within_region_target_metric = decomp_entry["within_region_target_metric"]
                            remaining_gap = decomp_entry["remaining_transfer_gap"]
                            gap_low = decomp_entry["remaining_transfer_gap_ci_low"]
                            gap_high = decomp_entry["remaining_transfer_gap_ci_high"]
                            residual_gap_status = status_derivation.derive_residual_gap_status(gap_low, gap_high)

                    # Brier is protocol-aware, never computed post hoc. Raw
                    # (source-only) rows take the frozen Step9 Brier score,
                    # which is unconditionally computed and not gated behind
                    # Step10's Brier preregistration. Adapted rows take Brier
                    # ONLY from the immutable Step10 protocol's own row.
                    if adaptation_method == "raw_source_only":
                        raw_brier = (
                            transfer["model_family_metrics"][model_family]["brier"]
                            if transfer is not None else None
                        )
                        brier_value = raw_brier
                        brier_availability = (
                            "available" if raw_brier is not None
                            else "unavailable_in_frozen_step9_transfer_metrics"
                        )
                        brier_protocol_source = "step9_transfer_metrics"
                    else:
                        brier_value = row["brier"] if row["brier_available"] else None
                        brier_availability = row["brier_availability_raw"] or "unavailable"
                        brier_protocol_source = "step10_final_report_protocol"

                    transfer_matrix_rows.append({
                        "source_experiment_id": source_id,
                        "target_experiment_id": target_id,
                        "model_family": model_family,
                        "adaptation_method": adaptation_method,
                        "primary_population": PRIMARY_POPULATION,
                        "target_row_count": row["target_row_count"],
                        "target_burned_count": row["target_burned_count"],
                        "roc_auc": row["roc_auc"],
                        "roc_auc_ci_low": row["roc_auc_ci_low"],
                        "roc_auc_ci_high": row["roc_auc_ci_high"],
                        # This row's own ROC-AUC vs. chance (0.5) -- read
                        # verbatim from the frozen Step10 protocol. Distinct
                        # from `adapted_minus_raw_support_status`: a method
                        # can have a supported positive adapted-minus-raw
                        # delta while its own ROC-AUC remains at/under chance.
                        "roc_auc_chance_status": status_derivation.bucket_roc_auc_chance_status(
                            row.get("roc_auc_chance_status"),
                        ),
                        "pr_auc": row["pr_auc"],
                        "pr_auc_ci_low": row["pr_auc_ci_low"],
                        "pr_auc_ci_high": row["pr_auc_ci_high"],
                        "raw_transfer_status": raw_status,
                        "adapted_minus_raw": adapted_minus_raw,
                        "adapted_minus_raw_ci_low": ci_low,
                        "adapted_minus_raw_ci_high": ci_high,
                        "adapted_minus_raw_support_status": adapted_minus_raw_support_status,
                        "within_region_target_metric": within_region_target_metric,
                        "remaining_transfer_gap": remaining_gap,
                        "remaining_transfer_gap_ci_low": gap_low,
                        "remaining_transfer_gap_ci_high": gap_high,
                        "residual_gap_status": residual_gap_status,
                        "step9e_shift_categories": shift_categories,
                        "step9e_ranking_reversal_value": ranking_reversal_value,
                        "step9e_ranking_reversal_scope": ranking_reversal_scope,
                        "brier": brier_value,
                        "brier_availability": brier_availability,
                        "brier_protocol_source": brier_protocol_source,
                    })

    # -------------------------------------------------------------------
    # Assemble feature_stability rows from Step9G, per unordered pair.
    # -------------------------------------------------------------------
    feature_stability_rows = []
    for a, b in unordered_pairs:
        g = step9g_normalized[(a, b)]
        for feat in g["features"]:
            side_a, side_b = feat["a"], feat["b"]
            reversal_status = status_derivation.bucket_reversal_status(feat["reversal_status"])
            feature_stability_rows.append({
                "experiment_a": a,
                "experiment_b": b,
                "feature": feat["feature"],
                "experiment_a_auc": side_a["auc"] if side_a else None,
                "experiment_a_ci_low": side_a["ci_low"] if side_a else None,
                "experiment_a_ci_high": side_a["ci_high"] if side_a else None,
                "experiment_a_direction": side_a["direction"] if side_a else None,
                "experiment_b_auc": side_b["auc"] if side_b else None,
                "experiment_b_ci_low": side_b["ci_low"] if side_b else None,
                "experiment_b_ci_high": side_b["ci_high"] if side_b else None,
                "experiment_b_direction": side_b["direction"] if side_b else None,
                "point_direction_reversal": feat["point_direction_reversal"],
                "reversal_status": reversal_status,
                "step9e_relationship_direction_flag": feat["step9e_relationship_direction_flag"],
            })

    summaries = _build_summaries(transfer_matrix_rows, feature_stability_rows, unordered_pairs)

    return {
        "dry_run": False,
        "generated_at": _iso_now(),
        "canonical_set_id": aoi_set.canonical_set_id,
        "primary_population": PRIMARY_POPULATION,
        "aois_display_order": list(aoi_set.display_order),
        "aois_canonical_order": list(aoi_set.canonical_order),
        "ordered_directions": [list(d) for d in ordered_directions],
        "unordered_pairs": [list(p) for p in unordered_pairs],
        "within_region": within_region_rows,
        "transfer_matrix": transfer_matrix_rows,
        "feature_stability": feature_stability_rows,
        "summaries": summaries,
        "scientific_boundaries": list(SCIENTIFIC_BOUNDARIES),
        "resolved_inputs": resolved_inputs,
        "output_root": output_root,
    }


def _build_summaries(transfer_matrix_rows: list[dict], feature_stability_rows: list[dict], unordered_pairs) -> dict:
    """Generate all scientific summary statements purely from the resolved
    numeric results/statuses already assembled above. No AOI-specific
    conditional branches: every statement is produced by a generic loop or
    aggregation over `transfer_matrix_rows` / `feature_stability_rows`."""
    raw_thermal_rows = [
        r for r in transfer_matrix_rows
        if r["model_family"] == "thermal" and r["adaptation_method"] == "raw_source_only"
        and r["roc_auc"] is not None
    ]

    strongest_raw_thermal_direction = None
    weakest_raw_thermal_direction = None
    below_chance_directions = []
    if raw_thermal_rows:
        strongest = max(raw_thermal_rows, key=lambda r: r["roc_auc"])
        weakest = min(raw_thermal_rows, key=lambda r: r["roc_auc"])
        strongest_raw_thermal_direction = {
            "source_experiment_id": strongest["source_experiment_id"],
            "target_experiment_id": strongest["target_experiment_id"],
            "roc_auc": strongest["roc_auc"],
        }
        weakest_raw_thermal_direction = {
            "source_experiment_id": weakest["source_experiment_id"],
            "target_experiment_id": weakest["target_experiment_id"],
            "roc_auc": weakest["roc_auc"],
        }
        below_chance_directions = [
            {"source_experiment_id": r["source_experiment_id"], "target_experiment_id": r["target_experiment_id"]}
            for r in raw_thermal_rows
            if r["raw_transfer_status"] == status_derivation.RAW_DISCRIMINATION_NOT_SUPPORTED
        ]

    bidirectional_raw_support_pairs = []
    for a, b in unordered_pairs:
        forward = next((r for r in raw_thermal_rows if r["source_experiment_id"] == a and r["target_experiment_id"] == b), None)
        reverse = next((r for r in raw_thermal_rows if r["source_experiment_id"] == b and r["target_experiment_id"] == a), None)
        if forward is not None and reverse is not None:
            both_supported = (
                forward["raw_transfer_status"] != status_derivation.RAW_DISCRIMINATION_NOT_SUPPORTED
                and reverse["raw_transfer_status"] != status_derivation.RAW_DISCRIMINATION_NOT_SUPPORTED
            )
            if both_supported:
                bidirectional_raw_support_pairs.append([a, b])

    adapted_minus_raw_support_status_counts: dict[str, int] = {}
    for r in transfer_matrix_rows:
        status = r.get("adapted_minus_raw_support_status")
        if status:
            adapted_minus_raw_support_status_counts[status] = (
                adapted_minus_raw_support_status_counts.get(status, 0) + 1
            )

    # Kept separate from `adapted_minus_raw_support_status_counts`: a row can
    # have a bootstrap-supported positive adapted-minus-raw delta while its
    # own adapted ROC-AUC remains at or under chance -- these two counts must
    # never be collapsed into a single "supported" figure.
    #
    # `all_transfer_rows_roc_auc_chance_status_counts` spans every
    # transfer-matrix row, including `raw_source_only` -- kept only for
    # backward compatibility, never rendered as "adapted-model" prose (a
    # `raw_source_only` row is not an adapted model).
    # `adapted_rows_roc_auc_chance_status_counts` is the adaptation-method-
    # filtered counterpart (adaptation_method != "raw_source_only") and is
    # the one reported as "adapted-transfer row(s)" in prose.
    all_transfer_rows_roc_auc_chance_status_counts: dict[str, int] = {}
    for r in transfer_matrix_rows:
        status = r.get("roc_auc_chance_status")
        if status:
            all_transfer_rows_roc_auc_chance_status_counts[status] = (
                all_transfer_rows_roc_auc_chance_status_counts.get(status, 0) + 1
            )

    adapted_transfer_rows = [r for r in transfer_matrix_rows if r["adaptation_method"] != "raw_source_only"]
    adapted_rows_roc_auc_chance_status_counts: dict[str, int] = {}
    for r in adapted_transfer_rows:
        status = r.get("roc_auc_chance_status")
        if status:
            adapted_rows_roc_auc_chance_status_counts[status] = (
                adapted_rows_roc_auc_chance_status_counts.get(status, 0) + 1
            )

    adapted_rows_chance_status_total = sum(adapted_rows_roc_auc_chance_status_counts.values())
    if adapted_rows_chance_status_total != len(adapted_transfer_rows):
        raise SynthesisBuildError(
            "Invariant violated: sum(adapted_rows_roc_auc_chance_status_counts.values()) "
            f"({adapted_rows_chance_status_total}) != number of transfer-matrix rows with "
            f"adaptation_method != 'raw_source_only' ({len(adapted_transfer_rows)})."
        )

    recovery_patterns = []
    adaptation_methods = sorted({
        r["adaptation_method"] for r in transfer_matrix_rows if r["adaptation_method"] != "raw_source_only"
    })
    for a, b in unordered_pairs:
        for model_family in MODEL_FAMILIES:
            for adaptation_method in adaptation_methods:
                forward = next((
                    r for r in transfer_matrix_rows
                    if r["source_experiment_id"] == a and r["target_experiment_id"] == b
                    and r["model_family"] == model_family and r["adaptation_method"] == adaptation_method
                ), None)
                reverse = next((
                    r for r in transfer_matrix_rows
                    if r["source_experiment_id"] == b and r["target_experiment_id"] == a
                    and r["model_family"] == model_family and r["adaptation_method"] == adaptation_method
                ), None)
                if forward is None or reverse is None:
                    continue
                statuses = [
                    forward.get("adapted_minus_raw_support_status"),
                    reverse.get("adapted_minus_raw_support_status"),
                ]
                if any(s is None for s in statuses):
                    continue
                pattern = status_derivation.derive_recovery_pattern_status(statuses)
                recovery_patterns.append({
                    "experiment_a": a, "experiment_b": b,
                    "model_family": model_family, "adaptation_method": adaptation_method,
                    "recovery_pattern_status": pattern,
                })

    bootstrap_supported_reversal_pairs = sorted({
        (r["experiment_a"], r["experiment_b"])
        for r in feature_stability_rows
        if r["reversal_status"] == status_derivation.BOOTSTRAP_SUPPORTED_DIRECTION_REVERSAL
    })

    residual_gap_remaining_rows = [
        {
            "source_experiment_id": r["source_experiment_id"],
            "target_experiment_id": r["target_experiment_id"],
            "model_family": r["model_family"],
            "adaptation_method": r["adaptation_method"],
        }
        for r in transfer_matrix_rows
        if r.get("residual_gap_status") == status_derivation.RESIDUAL_GAP_REMAINS
    ]

    roc_values = [r["roc_auc"] for r in raw_thermal_rows]
    raw_thermal_roc_range = (max(roc_values) - min(roc_values)) if roc_values else None

    return {
        "strongest_raw_thermal_direction": strongest_raw_thermal_direction,
        "weakest_raw_thermal_direction": weakest_raw_thermal_direction,
        "below_chance_raw_thermal_directions": below_chance_directions,
        "bidirectional_raw_thermal_support_pairs": bidirectional_raw_support_pairs,
        "adapted_minus_raw_support_status_counts": adapted_minus_raw_support_status_counts,
        # All-transfer-row counts (includes raw_source_only) -- kept only for
        # backward compatibility; see `adapted_rows_roc_auc_chance_status_counts`
        # for the adaptation-method-filtered counts that prose should quote.
        "all_transfer_rows_roc_auc_chance_status_counts": all_transfer_rows_roc_auc_chance_status_counts,
        "adapted_rows_roc_auc_chance_status_counts": adapted_rows_roc_auc_chance_status_counts,
        "recovery_patterns": recovery_patterns,
        "bootstrap_supported_feature_reversal_pairs": [list(p) for p in bootstrap_supported_reversal_pairs],
        "residual_gap_remaining_rows": residual_gap_remaining_rows,
        "raw_thermal_roc_auc_range_across_directions": raw_thermal_roc_range,
    }
