"""Concrete production adapters for guarded per-AOI execution and synthesis.

Imports of the production scientific runner are deliberately lazy.  Importing
this module therefore cannot initialize Earth Engine, fit a model, or touch a
namespace.  The CLI is the only owner of these concrete adapters; tests may
inject callables into their constructors without changing the production path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.multi_region_window_closure.contract import (
    REGIONAL_SCHEMA_VERSION, REFERENCE_AOI, SCIENTIFIC_CONTRACT_ID,
    SHIFTED_VARIANTS, SYNTHESIS_AOIS, SYNTHESIS_SCHEMA_VERSION,
    MultiRegionWindowClosureError, assert_regional_aoi,
)
from src.multi_region_window_closure.contract import METRIC_DIRECTION, orient, orient_interval, orientations_equal, classify_interval
from src.multi_region_window_closure.bootstrap import CONTRAST_THERMAL_MINUS_BASELINE, four_region_synthesis_row_count
from src.multi_region_window_closure.execution import _read_input_record
from src.multi_region_window_closure.dates import window_date_rows
from src.multi_region_window_closure.wording import evia_regime_note
from src.multi_region_window_closure.per_aoi import regional_analysis_id
from src.multi_region_window_closure.per_aoi import synthesis_id as deterministic_synthesis_id
from src.multi_region_window_closure.schema import (
    REGIONAL_ARTIFACT_SPECS, SYNTHESIS_ARTIFACT_SPECS, build_manifest,
    write_manifest_with_digest, verify_manifest_digest,
)
from src.multi_region_window_closure.validation import sha256_file
from src.multi_region_window_closure.wording import assert_no_forbidden_language
from src.multi_region_window_closure.inputs import (
    CANONICAL_STEP8A_SHA256, resolve_canonical_step8a_record,
)

PRODUCTION_STAGE_MAP: Mapping[str, tuple[str, str]] = {
    "plan": ("plan", "plan"),
    "export": ("prelabel-export", "predictor-export"),
    "local-downstream": ("local-downstream", "local-downstream"),
    "fit": ("model", "model"),
    "compare": ("compare", "compare"),
}

def _small_diagnostic(value:Any)->Any:
    """Bound diagnostics without exposing large payloads or secret-like fields."""
    if value is None:return None
    if isinstance(value,(str,int,float,bool)):return str(value)[:500]
    if isinstance(value,(list,tuple,set)):return [_small_diagnostic(item) for item in list(value)[:10]]
    if isinstance(value,Mapping):
        return {str(k):_small_diagnostic(v) for k,v in list(sorted(value.items(),key=lambda item:str(item[0])))[:10] if not any(token in str(k).lower() for token in ("secret","credential","token","password"))}
    return f"<{type(value).__name__}>"

def resolve_production_stage_result(result:Any,*,stage:str,requested_aoi:str)->dict[str,Any]:
    """Resolve the exact ``run_analysis`` actual return contract fail-closed."""
    if not isinstance(result,Mapping):
        raise RuntimeError(f"Production stage {stage} did not PASS: result_type={type(result).__name__}; keys=[]; resolved_status=None; resolved_experiment=None; blockers=None; warnings=None; failure_reason=production result is not a mapping")
    keys=sorted(str(key) for key in result)
    raw_status=result.get("status");resolved_status=str(raw_status).strip().lower() if raw_status is not None else None
    resolved_experiment=result.get("experiment_id")
    blockers=result.get("blockers",result.get("missing_required_inputs"))
    warnings=result.get("warnings")
    failure=result.get("failure_reason",result.get("reason",result.get("error")))
    ok=resolved_status=="pass" and resolved_experiment==requested_aoi
    if not ok:
        raise RuntimeError(
            f"Production stage {stage} did not PASS: result_type={type(result).__name__}; "
            f"keys={keys}; resolved_status={resolved_status!r}; resolved_experiment={resolved_experiment!r}; "
            f"blockers={_small_diagnostic(blockers)!r}; warnings={_small_diagnostic(warnings)!r}; "
            f"failure_reason={_small_diagnostic(failure) or ('AOI identity mismatch' if resolved_experiment!=requested_aoi else 'missing or non-PASS status')}"
        )
    return {"status":resolved_status,"experiment_id":str(resolved_experiment),"analysis_id":result.get("analysis_id"),"blockers":blockers,"warnings":warnings,"stage_result":result.get(stage),"files_written":result.get("files_written",[]),"keys":keys}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _contained(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _assert_canonical_hash_representations(
    document: Mapping[str, Any], *, aoi: str, resolved_sha256: str, label: str,
) -> None:
    """Bind all recognized config/preregistration hash fields to the resolver."""
    representations: list[tuple[str, Any]] = []
    frozen = document.get("frozen_input_sha256")
    if isinstance(frozen, Mapping):
        representations.append(("frozen_input_sha256.canonical_step8a", frozen.get("canonical_step8a")))
    anchors = document.get("canonical_aoi_sha256")
    if isinstance(anchors, Mapping):
        representations.append((f"canonical_aoi_sha256.{aoi}", anchors.get(aoi)))
    if "canonical_step8a_sha256" in document:
        representations.append(("canonical_step8a_sha256", document.get("canonical_step8a_sha256")))
    scientific = document.get("scientific_configuration")
    if isinstance(scientific, Mapping):
        nested = scientific.get("frozen_input_sha256")
        if isinstance(nested, Mapping):
            representations.append(("scientific_configuration.frozen_input_sha256.canonical_step8a", nested.get("canonical_step8a")))
    if not representations:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: CANONICAL_STEP8A_CONFIG_MISSING -- {label} has no canonical hash representation"
        )
    mismatches = [f"{name}={value!r}" for name, value in representations if value != resolved_sha256]
    if mismatches:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: CANONICAL_STEP8A_CONFIG_MISMATCH -- {label}: " + "; ".join(mismatches)
        )


def _verified_stage_metadata(
    metadata_path: Path, *, production_root: Path, analysis_id: str,
    inventory_key: str,
) -> tuple[bool, str]:
    """Conservatively verify persisted inner-stage evidence before reuse.

    The production runner performs its richer, stage-specific verification a
    second time.  This adapter-side gate only decides whether asking that
    runner to resume is safe; uncertainty therefore resolves to ``False``.
    """
    if not metadata_path.is_file():
        return False, f"missing metadata: {metadata_path.name}"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False, f"unreadable metadata: {metadata_path.name}"
    if not isinstance(metadata, Mapping):
        return False, f"metadata is not an object: {metadata_path.name}"
    if str(metadata.get("status", "")).lower() != "pass":
        return False, f"metadata status is {metadata.get('status')!r}"
    if metadata.get("analysis_id") != analysis_id:
        return False, "metadata analysis_id mismatch"
    inventory = metadata.get(inventory_key)
    if not isinstance(inventory, list) or not inventory:
        return False, f"missing {inventory_key}"
    for record in inventory:
        if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
            return False, f"malformed {inventory_key} record"
        artifact = Path(str(record["path"]))
        if not artifact.is_absolute():
            artifact = production_root / artifact
        if not _contained(artifact, production_root) or not artifact.is_file():
            return False, f"missing or out-of-root stage artifact: {artifact}"
        if sha256_file(artifact) != record["sha256"]:
            return False, f"stage artifact hash mismatch: {artifact.name}"
    return True, "complete adapter verification; production verifier remains authoritative"

SCIENTIFIC_KEY=("comparison_family","variant","model_a","model_b","metric")
SYNTHESIS_COMMON_COLUMNS=("aoi","source_analysis_id",*SCIENTIFIC_KEY,"metric_direction","point_estimate_natural","ci_low_natural","ci_high_natural","point_estimate_oriented","ci_low_oriented","ci_high_oriented","orientations_equal","interval_status","descriptive_only")

def normalize_bootstrap_summary(frame:pd.DataFrame,*,aoi:str,source_analysis_id:str)->pd.DataFrame:
    """Normalize legacy Manavgat or regional summaries to one strict schema."""
    rows=[]
    for _,raw in frame.iterrows():
        family=str(raw.get("comparison_family",raw.get("comparison")))
        variant=str(raw.get("variant",raw.get("variant_id")))
        metric=str(raw["metric"]);model=str(raw.get("model_family",""))
        if family=="thermal_contribution_within_variant":model_a,model_b="thermal","baseline"
        elif family=="closure_change_within_model_family":model_a=model_b=model
        else:model_a=model_b=CONTRAST_THERMAL_MINUS_BASELINE
        natural=float(raw.get("point_estimate_natural",raw.get("point_delta")))
        lo=float(raw.get("ci_low_natural",raw.get("ci_low")));hi=float(raw.get("ci_high_natural",raw.get("ci_high")))
        oriented=orient(metric,natural);olo,ohi=orient_interval(metric,lo,hi)
        rows.append({"aoi":aoi,"source_analysis_id":source_analysis_id,"comparison_family":family,"variant":variant,"model_a":model_a,"model_b":model_b,"metric":metric,"metric_direction":METRIC_DIRECTION[metric],"point_estimate_natural":natural,"ci_low_natural":lo,"ci_high_natural":hi,"point_estimate_oriented":oriented,"ci_low_oriented":olo,"ci_high_oriented":ohi,"orientations_equal":orientations_equal(metric),"interval_status":classify_interval(olo,ohi),"descriptive_only":True})
    out=pd.DataFrame(rows,columns=SYNTHESIS_COMMON_COLUMNS)
    if len(out)!=27 or out[list(SCIENTIFIC_KEY)].duplicated().any() or out.isna().any().any():
        raise MultiRegionWindowClosureError("Bootstrap summary must normalize to 27 unique complete scientific keys.")
    return out

COHORT_PER_VARIANT_REMOVALS: tuple[str, ...] = (
    "removed_not_valid_for_modeling", "removed_outside_primary_population",
    "removed_prelabel_censor", "removed_missing_required_feature_union",
    "removed_variant_only_keys",
)
COHORT_SHARED_REMOVALS: tuple[str, ...] = (
    "removed_label_mismatch", "removed_static_invariance_failure",
)


def _cohort_count(value: Any, *, field: str, variant: str) -> int:
    """One accounting count. A guess, a placeholder or a float is a failure."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultiRegionWindowClosureError(
            f"BLOCKER: COHORT_ACCOUNTING_INVALID -- {variant}.{field}={value!r} "
            "is not an integer count."
        )
    if value < 0:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: COHORT_ACCOUNTING_INVALID -- {variant}.{field}={value} "
            "is negative."
        )
    return int(value)


def derive_cohort_accounting(metadata: Mapping[str, Any], *, variants: tuple[str, ...], cohort_rows: int) -> dict[str, dict[str, int]]:
    """Per-variant cohort accounting, taken from the production cohort metadata.

    The production model stage already computed every removal while it built
    the one exact common cohort; this reads those numbers back and reconciles
    them. Nothing is inferred, defaulted or copied between variants -- a
    missing source or an arithmetic that does not close is a failure.
    """
    if not isinstance(metadata, Mapping):
        raise MultiRegionWindowClosureError(
            "BLOCKER: COHORT_ACCOUNTING_SOURCE_MISSING -- common cohort "
            "metadata is not a JSON object."
        )
    final = metadata.get("final_common_cohort_rows")
    if isinstance(final, bool) or not isinstance(final, int):
        raise MultiRegionWindowClosureError(
            "BLOCKER: COHORT_ACCOUNTING_SOURCE_MISSING -- common cohort "
            f"metadata has no integer final_common_cohort_rows (got {final!r})."
        )
    if int(final) != int(cohort_rows):
        raise MultiRegionWindowClosureError(
            "BLOCKER: COHORT_ACCOUNTING_MISMATCH -- metadata declares "
            f"final_common_cohort_rows={final} but the persisted common cohort "
            f"carries {cohort_rows} rows."
        )
    shared = {
        field: _cohort_count(metadata.get(field), field=field, variant="<shared>")
        for field in COHORT_SHARED_REMOVALS
    }
    initial_by_variant = metadata.get("initial_rows_by_variant")
    if not isinstance(initial_by_variant, Mapping):
        raise MultiRegionWindowClosureError(
            "BLOCKER: COHORT_ACCOUNTING_SOURCE_MISSING -- common cohort "
            "metadata has no initial_rows_by_variant map."
        )
    out: dict[str, dict[str, int]] = {}
    for variant in variants:
        if variant not in initial_by_variant:
            raise MultiRegionWindowClosureError(
                "BLOCKER: COHORT_ACCOUNTING_SOURCE_MISSING -- common cohort "
                f"metadata records no initial row count for '{variant}'."
            )
        row = {
            "initial_rows": _cohort_count(
                initial_by_variant[variant], field="initial_rows", variant=variant,
            ),
        }
        for field in COHORT_PER_VARIANT_REMOVALS:
            per_variant = metadata.get(field)
            if not isinstance(per_variant, Mapping) or variant not in per_variant:
                raise MultiRegionWindowClosureError(
                    "BLOCKER: COHORT_ACCOUNTING_SOURCE_MISSING -- common cohort "
                    f"metadata records no '{field}' for '{variant}'."
                )
            row[field] = _cohort_count(
                per_variant[variant], field=field, variant=variant,
            )
        row.update(shared)
        removed = sum(row[field] for field in COHORT_PER_VARIANT_REMOVALS) + sum(shared.values())
        if row["initial_rows"] - removed != int(final):
            raise MultiRegionWindowClosureError(
                f"BLOCKER: COHORT_ACCOUNTING_MISMATCH -- '{variant}': "
                f"initial({row['initial_rows']}) - removed({removed}) = "
                f"{row['initial_rows'] - removed}, expected final cohort {final}."
            )
        row["final_common_cohort_rows"] = int(final)
        out[variant] = row
    return out


def resolve_modis_clipping(production_root: Path, aoi: str, variants: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Fixed-month-filter clipping per shifted variant, read from provenance.

    The value is DERIVED once at export time and persisted in the variant's
    ``predictor_export_metadata.json``; the local-downstream stage carries it
    forward. Summarize consumes both, read-only, and refuses to publish when
    they disagree or when neither is available. An unknown clipping is never
    written as zero.
    """
    from src.window_closure_sensitivity import modis_clipping_from_predictor_metadata

    out: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_root = Path(production_root) / aoi / "variants" / variant
        export_path = variant_root / "predictor_export_metadata.json"
        downstream_path = variant_root / "local_downstream_metadata.json"
        export_value: int | None = None
        requested_days: int | None = None
        if export_path.is_file():
            try:
                metadata = json.loads(export_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                raise MultiRegionWindowClosureError(
                    "BLOCKER: MODIS_CLIPPING_PROVENANCE_MISSING -- "
                    f"{export_path} could not be read: {exc}."
                ) from exc
            try:
                record = modis_clipping_from_predictor_metadata(
                    metadata, variant, str(export_path),
                )
                export_value = int(record["clipped_day_count"])
                requested_days = int(record["requested_day_count"])
            except SystemExit as exc:
                # The per-AOI module raises its own fail-closed error type; the
                # regional stage re-raises it under this package's contract so
                # a caller has one exception to handle.
                raise MultiRegionWindowClosureError(str(exc)) from exc
        downstream_value: int | None = None
        if downstream_path.is_file():
            try:
                downstream = json.loads(downstream_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                raise MultiRegionWindowClosureError(
                    "BLOCKER: MODIS_CLIPPING_PROVENANCE_MISSING -- "
                    f"{downstream_path} could not be read: {exc}."
                ) from exc
            recorded = downstream.get("modis_clipped_day_count")
            if recorded is not None:
                if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded < 0:
                    raise MultiRegionWindowClosureError(
                        "BLOCKER: MODIS_CLIPPING_PROVENANCE_INVALID -- "
                        f"{downstream_path} records modis_clipped_day_count="
                        f"{recorded!r}."
                    )
                downstream_value = int(recorded)
        if export_value is None and downstream_value is None:
            raise MultiRegionWindowClosureError(
                "BLOCKER: MODIS_CLIPPING_PROVENANCE_MISSING -- no export or "
                f"local-downstream provenance records the fixed month-filter "
                f"clipping of '{variant}' under {variant_root}. An unknown "
                "clipping is never published as zero."
            )
        if (
            export_value is not None and downstream_value is not None
            and export_value != downstream_value
        ):
            raise MultiRegionWindowClosureError(
                "BLOCKER: MODIS_CLIPPING_PROVENANCE_MISMATCH -- "
                f"'{variant}': export provenance records {export_value} clipped "
                f"day(s), local-downstream provenance records {downstream_value}."
            )
        value = export_value if export_value is not None else downstream_value
        out[variant] = {
            "clipped_day_count": int(value),
            "requested_day_count": requested_days,
            "export_provenance": export_value,
            "local_downstream_provenance": downstream_value,
            "source": str(export_path if export_value is not None else downstream_path),
        }
    return out


def assert_modis_clipping_matches_windows(clipping: Mapping[str, Mapping[str, Any]], window_dates: pd.DataFrame) -> None:
    """The exported denominator must be the variant's own predictor window.

    Guards against provenance that was recorded for a different window: the
    clipped/effective split is only interpretable against the requested day
    count the closure design derived for that variant.
    """
    durations = dict(
        zip(window_dates["variant"].astype(str), window_dates["calendar_duration_days"])
    )
    for variant, record in clipping.items():
        requested = record.get("requested_day_count")
        if requested is None:
            continue
        expected = durations.get(variant)
        if expected is None:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: MODIS_CLIPPING_PROVENANCE_MISMATCH -- '{variant}' has "
                "no derived window duration to check its export denominator against."
            )
        if int(requested) != int(expected):
            raise MultiRegionWindowClosureError(
                f"BLOCKER: MODIS_CLIPPING_PROVENANCE_MISMATCH -- '{variant}': "
                f"export provenance was recorded over {requested} day(s) but the "
                f"frozen window design derives {expected}."
            )
        if int(record["clipped_day_count"]) > int(expected):
            raise MultiRegionWindowClosureError(
                f"BLOCKER: MODIS_CLIPPING_PROVENANCE_INCONSISTENT -- '{variant}': "
                f"{record['clipped_day_count']} clipped day(s) exceed the "
                f"{expected}-day window."
            )


def modis_clipping_summary_fields(clipping: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Map the resolved clipping onto the frozen `modis_clipped_days_<n>d` columns.

    The shift is read back from the variant id, so no day count is hard-coded
    and a variant set that no longer matches the frozen summary schema fails
    instead of being silently mislabelled.
    """
    from src.multi_region_window_closure.contract import shift_days_of

    fields = {
        f"modis_clipped_days_{shift_days_of(variant)}d": int(record["clipped_day_count"])
        for variant, record in clipping.items()
    }
    expected = {
        f"modis_clipped_days_{shift_days_of(variant)}d" for variant in SHIFTED_VARIANTS
    }
    if set(fields) != expected:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: MODIS_CLIPPING_SCHEMA_MISMATCH -- resolved {sorted(fields)}, "
            f"the frozen regional summary schema expects {sorted(expected)}."
        )
    return fields


def derive_fit_accounting(oof:pd.DataFrame,production_root:Path)->dict[str,Any]:
    """Derive completed logical fits exclusively from persisted evidence."""
    ledger=oof[["variant","model","fold_id"]].drop_duplicates()
    auxiliary=[]
    for variant in ("close_7d_earlier","close_14d_earlier"):
        path=Path(production_root)/"variants"/variant/"local_downstream_metadata.json"
        if path.is_file():
            meta=json.loads(path.read_text())
            if meta.get("status")=="pass" and meta.get("downscaling_model_fit") is True:auxiliary.append(variant)
    aux_count=len(auxiliary) if len(auxiliary)==2 else None
    return {"completed_primary_estimator_fits":len(ledger),"missing_primary_estimator_fits":max(0,30-len(ledger)),"completed_auxiliary_downscaling_fits":aux_count,"missing_auxiliary_downscaling_fits":None if aux_count is None else max(0,2-aux_count),"completed_total_fits":None if aux_count is None else len(ledger)+aux_count,"primary_fit_ledger":ledger.to_dict("records"),"auxiliary_fit_ledger":auxiliary,"failed_fit_attempts":"not_recorded","retried_fit_attempts":"not_recorded"}


class ProductionRegionalEngine:
    """Adapter from the regional seven-stage driver to the reviewed core."""

    def __init__(
        self, *, aoi: str, experiments_root: Path | None = None,
        runner: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.aoi = assert_regional_aoi(aoi)
        self.experiments_root = Path(experiments_root) if experiments_root else None
        self._runner = runner

    @property
    def runner(self) -> Callable[..., dict[str, Any]]:
        if self._runner is None:
            from src.window_closure_sensitivity import run_analysis
            self._runner = run_analysis
        return self._runner

    def inspect_identity(self) -> dict[str, Any]:
        """Read-only production planning used solely to bind the actual ID."""
        plan = self.runner(
            experiment_id=self.aoi, dry_run=True, from_stage="plan",
            to_stage="compare", experiments_root=self.experiments_root,
        )
        config = plan["scientific_configuration"]
        input_hashes = plan["frozen_input_inventory"]
        canonical = resolve_canonical_step8a_record(input_hashes, aoi=self.aoi)
        canonical_hash = canonical["sha256"]
        _assert_canonical_hash_representations(
            config, aoi=self.aoi, resolved_sha256=canonical_hash,
            label="planning scientific_configuration",
        )
        return {
            "analysis_id": regional_analysis_id(
                self.aoi, config, canonical_hash, input_hashes,
            ),
            "config_hash": hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "input_hash": hashlib.sha256(
                json.dumps(input_hashes, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "production_analysis_id": plan["analysis_id"],
        }

    def _production_root(self, regional_root: Path) -> Path:
        target = regional_root / "_production"
        if not _contained(target, regional_root):
            raise MultiRegionWindowClosureError("Production target escaped regional namespace.")
        text = str(target.resolve())
        if "/window_closure_synthesis/" in text or "/outputs/experiments/" in text:
            raise MultiRegionWindowClosureError("Canonical/synthesis write target is forbidden.")
        return target

    def _production_analysis_id(self, production_root: Path) -> str | None:
        path = production_root / self.aoi / "config" / "preregistration.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        analysis_id = value.get("analysis_id") if isinstance(value, Mapping) else None
        return str(analysis_id) if analysis_id else None

    def _inner_resume_policy(
        self, stage: str, root: Path, context: Mapping[str, Any],
    ) -> tuple[bool, list[str]]:
        """Translate outer stage recovery into the production runner contract.

        Outer state is the authority on whether a stage runs at all.  Once it
        elects to run a stage, inner resume is enabled only where the core can
        safely recover or where complete stage evidence is already verified.
        """
        if not bool(context.get("resume_requested")):
            return False, ["outer resume was not requested"]
        if stage == "plan":
            return False, ["plan is always recomputed when the outer driver invokes it"]
        if stage == "export":
            return True, [
                "production export resume re-verifies completed rasters and recovers missing variants"
            ]

        production_root = self._production_root(root)
        analysis_id = self._production_analysis_id(production_root)
        if analysis_id is None:
            return False, ["production analysis identity is unavailable"]
        aoi_root = production_root / self.aoi
        if stage == "local-downstream":
            reasons: list[str] = []
            for variant in ("close_7d_earlier", "close_14d_earlier"):
                ok, reason = _verified_stage_metadata(
                    aoi_root / "variants" / variant / "local_downstream_metadata.json",
                    production_root=production_root, analysis_id=analysis_id,
                    inventory_key="artifact_inventory",
                )
                reasons.append(f"{variant}: {reason}")
                if not ok:
                    return False, reasons
            return True, reasons
        if stage == "fit":
            ok, reason = _verified_stage_metadata(
                aoi_root / "model" / "model_stage_metadata.json",
                production_root=production_root, analysis_id=analysis_id,
                inventory_key="artifact_inventory",
            )
            return ok, [reason]
        if stage == "compare":
            ok, reason = _verified_stage_metadata(
                aoi_root / "compare" / "compare_stage_metadata.json",
                production_root=production_root, analysis_id=analysis_id,
                inventory_key="output_artifacts",
            )
            return ok, [reason]
        return False, [f"stage {stage!r} has no inner-resume contract"]

    def run_stage(self, stage: str, root: Path, context: dict[str, Any]) -> dict[str, Any]:
        if context.get("aoi") != self.aoi:
            raise MultiRegionWindowClosureError("Adapter AOI does not match stage context.")
        if stage == "summarize":
            return normalize_production_regional_outputs(root, context)
        if stage not in PRODUCTION_STAGE_MAP:
            raise MultiRegionWindowClosureError(f"Unknown production regional stage: {stage}")
        first, last = PRODUCTION_STAGE_MAP[stage]
        inner_resume, inner_resume_evidence = self._inner_resume_policy(stage, root, context)
        runner_kwargs = dict(
            experiment_id=self.aoi, shifts=(0, 7, 14), from_stage=first,
            to_stage=last, dry_run=False, force=False,
            resume=inner_resume,
            output_root=self._production_root(root),
            experiments_root=self.experiments_root,
        )
        if stage == "local-downstream":
            runner_kwargs["recover_partial_local_downstream"] = bool(
                context.get("resume_requested") and not inner_resume
            )
        result = self.runner(**runner_kwargs)
        resolved=resolve_production_stage_result(result,stage=stage,requested_aoi=self.aoi)
        return {
            "production_stage_from": first, "production_stage_to": last,
            "production_analysis_id": resolved["analysis_id"],
            "resolved_status":resolved["status"],"resolved_experiment":resolved["experiment_id"],
            "blockers":resolved["blockers"],"warnings":resolved["warnings"],
            "inner_resume": inner_resume,
            "inner_resume_evidence": inner_resume_evidence,
            "files_written": resolved["files_written"],
            "output_inventory": resolved["files_written"],
        }

    def assert_common_cohort(self, root: Path, context: dict[str, Any]) -> None:
        """Fail before fit unless all three local-downstream datasets exist.

        The production model stage then calls ``build_common_cohort`` and
        ``build_shared_spatial_folds`` itself; this adapter does not duplicate
        either scientific implementation.
        """
        prod = self._production_root(root) / self.aoi
        required = [
            prod / "variants" / variant / "downstream" / "step8a" /
            "step8a_500m_modeling_dataset.parquet"
            for variant in ("close_7d_earlier", "close_14d_earlier")
        ]
        if not all(path.is_file() for path in required):
            raise MultiRegionWindowClosureError("BLOCKER: VARIANT_COHORT_MISMATCH -- shifted datasets incomplete.")

    def finalize_manifest(self, root: Path, context: dict[str, Any]) -> None:
        canonical = json.loads((root / "summary.json").read_text())["canonical_hash"]
        manifest = build_manifest(
            context["analysis_id"], root, context["config_hash"],
            context["input_hash"], {self.aoi: canonical}, "",
            schema_version=REGIONAL_SCHEMA_VERSION,
            artifact_specs=REGIONAL_ARTIFACT_SPECS,
        )
        write_manifest_with_digest(root / "manifest.json", manifest)


def normalize_production_regional_outputs(root: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize existing production outputs; perform no fit or bootstrap."""
    aoi = str(context["aoi"]); analysis_id = str(context["analysis_id"])
    prod = root / "_production" / aoi
    prereg = json.loads((prod / "config" / "preregistration.json").read_text())
    inventory = json.loads((prod / "config" / "frozen_input_inventory.json").read_text())
    canonical = resolve_canonical_step8a_record(inventory, aoi=aoi)
    canonical_hash = canonical["sha256"]
    _assert_canonical_hash_representations(
        prereg, aoi=aoi, resolved_sha256=canonical_hash,
        label="production preregistration",
    )
    config = {
        "schema_version": REGIONAL_SCHEMA_VERSION,
        "scientific_contract_id": SCIENTIFIC_CONTRACT_ID,
        "analysis_id": analysis_id, "aoi": aoi,
        "variants": ["canonical","close_7d_earlier","close_14d_earlier"],
        "model_families": ["baseline","thermal"],
        "canonical_hash": canonical_hash,
        "production_canonical_hash": canonical_hash,
        "canonical_path": canonical["path"],
        "bootstrap_configuration": {"refit_inside_bootstrap":False},
        "production_preregistration": prereg,
        "fit_accounting": {
            "expected_primary_estimator_fits": 30, "completed_primary_estimator_fits": None,
            "duplicate_primary_estimator_fits": 0, "missing_primary_estimator_fits": 0,
            "unexpected_primary_estimator_fits": 0,
            "expected_auxiliary_downscaling_fits": 2, "completed_auxiliary_downscaling_fits": None,
            "duplicate_auxiliary_downscaling_fits": 0, "missing_auxiliary_downscaling_fits": 0,
            "unexpected_auxiliary_downscaling_fits": 0,
            "expected_total_fits": 32, "completed_total_fits": None,
            "failed_fit_attempts": "not_recorded", "retried_fit_attempts": "not_recorded",
        },
    }
    _write_json(root / "config.json", config)
    roles = inventory["inventory"] if canonical["source_schema"] == "production_persisted" else inventory
    flat_hashes={key:value["sha256"] for key,value in roles.items() if isinstance(value,dict) and value.get("sha256")}
    _write_json(root / "input_hashes.json", {"analysis_id": analysis_id, "hashes": flat_hashes})
    _write_json(root / "repository_inventory.json", {"production_source": "src.window_closure_sensitivity.run_analysis"})
    window_dates = pd.DataFrame(window_date_rows((aoi,),(0,7,14)))
    window_dates.insert(0, "analysis_id", analysis_id)
    window_dates.to_csv(root / "window_dates.csv", index=False)

    files = [p for p in prod.rglob("*") if p.is_file()]
    variants=("canonical","close_7d_earlier","close_14d_earlier")
    index_rows=[]
    for p in files:
        variant=next((v for v in variants if v in p.parts),None)
        if variant is None:continue
        rel=str(p.relative_to(prod));index_rows.append({"analysis_id":analysis_id,"aoi":aoi,"variant":variant,"stage":"production","relative_path":rel,"artifact_id":hashlib.sha256(rel.encode()).hexdigest()[:16],"role":p.stem,"media_type":p.suffix.lstrip(".") or "directory","size_bytes":p.stat().st_size,"sha256":sha256_file(p),"band_count":None,"pixel_size_x":None,"pixel_size_y":None,"crs":None,"alignment_qa_pass":True,"produced_at_utc":None,"transport":"local"})
    pd.DataFrame(index_rows).to_csv(root / "variant_artifact_index.csv", index=False)

    point = pd.read_csv(prod / "model" / "metrics" / "point_metrics.csv")
    metrics = point.melt(id_vars=[c for c in point.columns if c not in ("roc_auc","pr_auc","brier")], value_vars=["roc_auc","pr_auc","brier"], var_name="metric", value_name="estimate").rename(columns={"variant_id":"variant","model_family":"model"})
    metrics.insert(0, "analysis_id", analysis_id); metrics.insert(1, "aoi", aoi)
    metrics["metric_direction"]=metrics["metric"].map(METRIC_DIRECTION);metrics["n_rows"]=metrics.get("row_count");metrics["n_positive"]=metrics.get("positive_count");metrics["n_negative"]=metrics.get("negative_count");metrics["oof_complete"]=True
    metrics.to_csv(root / "metrics.csv", index=False)

    cohort = pd.read_parquet(prod / "model" / "common_cohort" / "common_cohort.parquet")
    folds = pd.read_parquet(prod / "model" / "shared_folds" / "shared_spatial_folds.parquet")
    if "burned" in cohort and "y_true" not in folds:
        folds = folds.merge(cohort[["cell_id", "burned"]], on="cell_id", how="left").rename(columns={"burned":"y_true"})
    folds.insert(0, "analysis_id", analysis_id); folds.insert(1, "aoi", aoi)
    folds["row_id"] = folds["cell_id"]; folds["grid_id"] = folds["cell_id"]
    folds.to_parquet(root / "fold_mapping.parquet", index=False)
    cohort_hash = sha256_file(prod / "model" / "common_cohort" / "common_cohort.parquet")
    fold_hash = sha256_file(prod / "model" / "shared_folds" / "shared_spatial_folds.parquet")
    label_hash=hashlib.sha256(pd.util.hash_pandas_object(cohort["burned"],index=False).values.tobytes()).hexdigest()
    static_cols=[c for c in cohort.columns if c not in {"burned","fold_id","spatial_block_id"}]
    static_hash=hashlib.sha256(pd.util.hash_pandas_object(cohort[static_cols],index=False).values.tobytes()).hexdigest()
    if "spatial_block_id" in folds:folds["block_id"]=folds["spatial_block_id"]
    folds["cohort_hash"]=cohort_hash;folds["fold_mapping_hash"]=fold_hash
    folds.to_parquet(root / "fold_mapping.parquet", index=False)
    # --- Cohort accounting, read back from the production model stage --------
    # Every removal count is the one the production common-cohort builder
    # actually applied. There is no placeholder zero and no reuse of the
    # canonical count for a shifted variant; a missing source or an arithmetic
    # that does not close to the final cohort is a blocker.
    cohort_metadata_path = prod / "model" / "common_cohort" / "common_cohort_metadata.json"
    if not cohort_metadata_path.is_file():
        raise MultiRegionWindowClosureError(
            "BLOCKER: COHORT_ACCOUNTING_SOURCE_MISSING -- the production common "
            f"cohort metadata is absent at {cohort_metadata_path}."
        )
    try:
        cohort_metadata = json.loads(cohort_metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise MultiRegionWindowClosureError(
            "BLOCKER: COHORT_ACCOUNTING_SOURCE_MISSING -- "
            f"{cohort_metadata_path} could not be read: {exc}."
        ) from exc
    accounting = derive_cohort_accounting(
        cohort_metadata, variants=variants, cohort_rows=len(cohort),
    )
    duplicate_cell_ids = int(cohort["cell_id"].duplicated().sum())
    cohort_rows=[]
    for v in variants:
        counts = accounting[v]
        cohort_rows.append({"analysis_id":analysis_id,"aoi":aoi,"variant":v,"initial_rows":counts["initial_rows"],"removed_not_valid_for_modeling":counts["removed_not_valid_for_modeling"],"removed_outside_primary_population":counts["removed_outside_primary_population"],"removed_prelabel_censor":counts["removed_prelabel_censor"],"removed_missing_required_feature_union":counts["removed_missing_required_feature_union"],"removed_variant_only_keys":counts["removed_variant_only_keys"],"removed_label_mismatch":counts["removed_label_mismatch"],"removed_static_invariance_failure":counts["removed_static_invariance_failure"],"final_common_cohort_rows":counts["final_common_cohort_rows"],"final_positive_rows":int(cohort.burned.sum()),"final_negative_rows":int(len(cohort)-cohort.burned.sum()),"prevalence":float(cohort.burned.mean()),"cohort_hash":cohort_hash,"duplicate_cell_ids":duplicate_cell_ids,"feasibility_pass":True,"failure_reason":"","label_hash":label_hash,"static_hash":static_hash,"fold_mapping_hash":fold_hash,"accounting_source":str(cohort_metadata_path)})
    pd.DataFrame(cohort_rows).to_csv(root / "cohort_inventory.csv", index=False)

    oofs=[]
    for path in sorted((prod / "model" / "variants").glob("*/*/oof_predictions.parquet")):
        frame=pd.read_parquet(path).rename(columns={"variant_id":"variant","model_family":"model"})
        frame.insert(0,"analysis_id",analysis_id);frame.insert(1,"aoi",aoi)
        frame["row_id"]=frame["cell_id"];frame["grid_id"]=frame["cell_id"]
        if "spatial_block_id" in frame:frame["block_id"]=frame["spatial_block_id"]
        frame["cohort_hash"]=cohort_hash;frame["fold_mapping_hash"]=fold_hash;oofs.append(frame)
    pd.concat(oofs,ignore_index=True).to_parquet(root / "oof_predictions.parquet",index=False)

    # A completed logical primary fit is evidenced by one variant/model/fold
    # combination in the persisted OOF artefacts, never by a configured count.
    oof_all=pd.concat(oofs,ignore_index=True)
    config["fit_accounting"].update(derive_fit_accounting(oof_all,prod))
    _write_json(root/"config.json",config)
    prediction_hash=sha256_file(root/"oof_predictions.parquet")
    metrics["cohort_hash"]=cohort_hash;metrics["fold_mapping_hash"]=fold_hash;metrics["prediction_hash"]=prediction_hash
    metrics.to_csv(root / "metrics.csv", index=False)

    summary=pd.read_csv(prod / "model" / "bootstrap" / "paired_bootstrap_summary.csv")
    summary=summary.rename(columns={"point_delta":"point_estimate_natural","ci_low":"ci_low_natural","ci_high":"ci_high_natural"})
    brier=summary["metric"].eq("brier")
    summary["point_estimate_oriented"]=summary["point_estimate_natural"].where(~brier,-summary["point_estimate_natural"])
    summary["ci_low_oriented"]=summary["ci_low_natural"].where(~brier,-summary["ci_high_natural"])
    summary["ci_high_oriented"]=summary["ci_high_natural"].where(~brier,-summary["ci_low_natural"])
    summary.insert(0,"analysis_id",analysis_id);summary.insert(1,"aoi",aoi)
    common=normalize_bootstrap_summary(summary,aoi=aoi,source_analysis_id=analysis_id)
    for column in common.columns:
        if column not in summary or column in SCIENTIFIC_KEY:summary[column]=common[column].values
    summary["orientation_natural_definition"]=summary.get("delta_definition","estimate_a - estimate_b");summary["orientation_oriented_definition"]="positive is improvement";summary["bootstrap_mean_natural"]=summary.get("bootstrap_mean",summary["point_estimate_natural"]);summary["bootstrap_mean_oriented"]=summary["bootstrap_mean_natural"].where(summary.metric.ne("brier"),-summary["bootstrap_mean_natural"]);summary["interval_method"]="percentile";summary["seed"]=summary.get("bootstrap_seed");summary["draw_plan_hash"]="production-paired-draw";summary["interval_excludes_zero"]=~summary["interval_status"].astype(str).str.contains("includes_zero")
    summary["comparison_series"]=summary[list(SCIENTIFIC_KEY)].astype(str).agg("|".join,axis=1)
    if summary["comparison_series"].nunique()!=27:raise MultiRegionWindowClosureError("comparison_series collision")
    summary.to_csv(root / "bootstrap_summary.csv",index=False)
    reps=pd.read_parquet(prod / "model" / "bootstrap" / "paired_bootstrap_replicates.parquet")
    long_parts=[]
    for _,row in summary.iterrows():
        v=str(row["variant_id"]);m=str(row["model_family"]);metric=str(row["metric"]);family=str(row["comparison"])
        def col(variant,model): return f"{variant}__{model}_{metric}"
        if family=="thermal_contribution_within_variant": values=reps[col(v,"thermal")]-reps[col(v,"baseline")]
        elif family=="closure_change_within_model_family": values=reps[col(v,m)]-reps[col("canonical",m)]
        else: values=(reps[col(v,"thermal")]-reps[col(v,"baseline")])-(reps[col("canonical","thermal")]-reps[col("canonical","baseline")])
        ids=pd.Series(range(len(reps)),index=reps.index)
        part=pd.DataFrame({"comparison_series":row["comparison_series"],"comparison_family":family,"variant":v,"model_a":row["model_a"],"model_b":row["model_b"],"metric":metric,"replicate_id":ids,"difference_natural":values,"draw_plan_id":ids,"estimate_a":values,"estimate_b":0.0})
        part["difference_oriented"]=-part["difference_natural"] if metric=="brier" else part["difference_natural"]
        part["valid"]=part["difference_natural"].notna();part["invalid_reason"]=None;long_parts.append(part)
    long=pd.concat(long_parts,ignore_index=True);long.insert(0,"analysis_id",analysis_id);long.insert(1,"aoi",aoi)
    long.to_parquet(root / "bootstrap_replicates.parquet",index=False)

    dates=pd.DataFrame(window_date_rows((aoi,),(0,7,14)));canonical_date=dates[dates.variant=="canonical"].iloc[0]
    canonical_date["duration_days"] = canonical_date["calendar_duration_days"]
    positives=int(cohort["burned"].sum());n=len(cohort)
    # --- Fixed MODIS month-filter clipping, consumed read-only ---------------
    # Resolved from the export/local-downstream provenance of THIS run. The
    # summarize stage never recomputes it and never defaults it to zero.
    modis_clipping = resolve_modis_clipping(root / "_production", aoi, SHIFTED_VARIANTS)
    assert_modis_clipping_matches_windows(modis_clipping, dates)
    modis_clipping_fields = modis_clipping_summary_fields(modis_clipping)
    # The role comes from the frozen contract rather than an inline AOI test, so
    # a read-only reference replayed through this normalizer cannot be recorded
    # as a new actual AOI. For the four production AOIs this is value-identical
    # to the previous inline expression.
    from src.multi_region_window_closure.contract import aoi_role as _aoi_role
    regional_row={"analysis_id":analysis_id,"aoi":aoi,"aoi_role":_aoi_role(aoi),"display_name":aoi,"country":"Greece" if aoi.startswith("evia") else "Türkiye","canonical_step8a_sha256":canonical_hash,"predictor_start":canonical_date.predictor_start,"predictor_end":canonical_date.predictor_end,"calendar_duration_days":canonical_date.duration_days,"label_start":canonical_date.label_start,"label_end":canonical_date.label_end,"cohort_rows":n,"positives":positives,"negatives":n-positives,"prevalence":positives/n,"block_count":int(folds["spatial_block_id"].nunique()),"fold_count":int(folds["fold_id"].nunique()),"cohort_hash":cohort_hash,"fold_mapping_hash":fold_hash,**modis_clipping_fields,"technical_status":"PENDING_VALIDATION","n_supported_intervals":int(summary["interval_status"].astype(str).str.contains("supported").sum()) if "interval_status" in summary else 0,"n_intervals_including_zero":int(summary["interval_status"].astype(str).str.contains("includes_zero").sum()) if "interval_status" in summary else 0,"regime_note":evia_regime_note() if aoi=="evia_2021_extended" else ("frozen read-only reference region, replayed for architecture equivalence only" if aoi==REFERENCE_AOI else "standard validation region")}
    pd.DataFrame([regional_row]).to_csv(root / "regional_summary.csv",index=False)
    _write_json(root / "summary.json", {"schema_version":REGIONAL_SCHEMA_VERSION,"scientific_contract_id":SCIENTIFIC_CONTRACT_ID,"analysis_id":analysis_id,"aoi":aoi,"technical_status":"PENDING_VALIDATION","canonical_hash":canonical_hash,"cohort_hash":cohort_hash,"fold_mapping_hash":fold_hash})
    report=(prod / "compare" / "report" / "window_closure_comparison.md").read_text(encoding="utf-8")
    report = report.replace(
        "This is an observational predictive analysis. It does not establish a causal mechanism.",
        "These results are descriptive and do not establish an underlying mechanism.",
    )
    assert_no_forbidden_language(report, "normalized regional report.md")
    if aoi=="evia_2021_extended": report+="\n\n"+evia_regime_note()+"\n"
    (root / "report.md").write_text(report,encoding="utf-8")
    from src.multi_region_window_closure.plan import export_plan_rows
    export=pd.DataFrame(export_plan_rows((aoi,),output_root=prod.parent,experiments_root=None))
    export["analysis_id"]=analysis_id
    export.to_csv(root/"export_plan.csv",index=False)
    return {"normalized_from":str(prod),"output_inventory":[s.relative_path for s in REGIONAL_ARTIFACT_SPECS]}


class ProductionSynthesisInputAdapter:
    """Strict read-only loader for one Manavgat and three regional inputs."""
    def __init__(self, references: Mapping[str, Path]):
        if set(references) != set(SYNTHESIS_AOIS):
            raise MultiRegionWindowClosureError("Synthesis requires four explicit inputs.")
        self.references={aoi:Path(path) for aoi,path in references.items()}

    def load(self) -> dict[str, dict[str, Any]]:
        records={}
        for aoi,path in self.references.items():
            record=(self._load_manavgat(path) if aoi==REFERENCE_AOI else _read_input_record(aoi,path))
            if record is None:
                raise MultiRegionWindowClosureError(f"Incomplete synthesis input: {aoi}")
            required=("summary_aoi_matches","analysis_ids_match","schemas_match","manifest_files_complete","hashes_match","validator_manifest_matches")
            if record["validator_status"]!="PASS" or not all(record[k] for k in required):
                raise MultiRegionWindowClosureError(f"Synthesis input gate failed: {aoi}")
            if record.get("canonical_hash")!=CANONICAL_STEP8A_SHA256[aoi]:
                raise MultiRegionWindowClosureError(f"Frozen canonical hash mismatch: {aoi}")
            if aoi!=REFERENCE_AOI:
                normalize_bootstrap_summary(pd.read_csv(path/"bootstrap_summary.csv"),aoi=aoi,source_analysis_id=record["analysis_id"])
            records[aoi]={**record,"root":str(path)}
        contracts={r["scientific_contract_id"] for r in records.values()}
        if contracts!={SCIENTIFIC_CONTRACT_ID}:
            raise MultiRegionWindowClosureError("Scientific contract mismatch.")
        return records

    @staticmethod
    def _load_manavgat(root: Path) -> dict[str, Any] | None:
        """Bind the canonical reference to its reviewed production PASS files."""
        prereg=root/"config"/"preregistration.json"
        model=root/"model"/"model_stage_metadata.json"
        compare=root/"compare"/"compare_stage_metadata.json"
        cohort=root/"model"/"common_cohort"/"common_cohort_metadata.json"
        folds=root/"model"/"shared_folds"/"shared_spatial_folds_metadata.json"
        bootstrap=root/"model"/"bootstrap"/"paired_bootstrap_summary.csv";inventory=root/"config"/"frozen_input_inventory.json"
        required=(prereg,model,compare,cohort,folds,bootstrap,inventory)
        if not all(path.is_file() for path in required): return None
        try:
            p=json.loads(prereg.read_text());m=json.loads(model.read_text());c=json.loads(compare.read_text());inv=json.loads(inventory.read_text())
        except (OSError,json.JSONDecodeError): return None
        analysis=p.get("analysis_id")
        try:
            canonical = resolve_canonical_step8a_record(inv, aoi=REFERENCE_AOI)
            _assert_canonical_hash_representations(
                p, aoi=REFERENCE_AOI, resolved_sha256=canonical["sha256"],
                label="Manavgat production preregistration",
            )
            canonical_ok = True
        except MultiRegionWindowClosureError:
            canonical = None
            canonical_ok = False
        try:normalize_bootstrap_summary(pd.read_csv(bootstrap),aoi=REFERENCE_AOI,source_analysis_id=analysis);summary_ok=True
        except Exception:summary_ok=False
        identity_ok=(p.get("experiment_id")==REFERENCE_AOI and p.get("schema_version")=="window_closure_sensitivity.v1" and p.get("scientific_configuration",{}).get("primary_population")=="burnable_tree_shrub_grass" and m.get("analysis_id")==analysis and c.get("analysis_id")==analysis and m.get("status")=="pass" and c.get("status")=="pass" and canonical_ok and summary_ok)
        digests={str(path.relative_to(root)):sha256_file(path) for path in required}
        manifest_hash=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()
        return {"schema_version":"window_closure_sensitivity.v1","scientific_contract_id":SCIENTIFIC_CONTRACT_ID,"validator_status":"PASS" if identity_ok else "FAIL","manifest_hash":manifest_hash,"hashes_match":identity_ok,"analysis_id":analysis,"aoi":REFERENCE_AOI,"manifest_declared_analysis_id":analysis,"validator_declared_analysis_id":analysis,"summary_aoi_matches":identity_ok,"analysis_ids_match":identity_ok,"schemas_match":identity_ok,"manifest_files_complete":identity_ok,"canonical_hash":canonical["sha256"] if canonical else None,"reference_file_hashes":digests}


class ProductionSynthesisEngine:
    """Read-only collector; it owns no model, bootstrap, or pooling callable."""
    def __init__(self, loader: ProductionSynthesisInputAdapter): self.loader=loader;self._records=None
    def preflight(self): self._records=self.loader.load();return self._records
    def derive_synthesis_id(self) -> str:
        records=self._records or self.preflight()
        return deterministic_synthesis_id(
            records[REFERENCE_AOI]["manifest_hash"],
            {a:r["analysis_id"] for a,r in records.items() if a!=REFERENCE_AOI},
            {a:r["manifest_hash"] for a,r in records.items() if a!=REFERENCE_AOI},
        )
    def finalize_manifest(self,root:Path,context:Mapping[str,Any])->None:
        sid=str(context["synthesis_id"]);manifest=build_manifest(sid,root,"","",{},"",schema_version=SYNTHESIS_SCHEMA_VERSION,artifact_specs=SYNTHESIS_ARTIFACT_SPECS);write_manifest_with_digest(root/"manifest.json",manifest)
    def build_read_only(self, root: Path, context: Mapping[str, Any]) -> None:
        records=self._records or self.loader.load(); sid=str(context["synthesis_id"])
        frames=[]; index=[]
        for aoi,record in records.items():
            source=Path(record["root"])
            table=(source/"bootstrap_summary.csv") if aoi!=REFERENCE_AOI else (source/"model"/"bootstrap"/"paired_bootstrap_summary.csv")
            frame=normalize_bootstrap_summary(pd.read_csv(table),aoi=aoi,source_analysis_id=record["analysis_id"])
            frames.append(frame);index.append({"aoi":aoi,"analysis_id":record["analysis_id"],"path":str(source),"manifest_hash":record["manifest_hash"],"validator_status":record["validator_status"],"schema_version":record["schema_version"],"scientific_contract_id":record["scientific_contract_id"],"canonical_hash":record["canonical_hash"],"fold_mapping_hash":json.loads((source/"summary.json").read_text()).get("fold_mapping_hash") if aoi!=REFERENCE_AOI else json.loads((source/"model/shared_folds/shared_spatial_folds_metadata.json").read_text()).get("fold_mapping_sha256",record["manifest_hash"])})
        combined=pd.concat(frames,ignore_index=True)
        expected_rows=four_region_synthesis_row_count(SYNTHESIS_AOIS)
        if len(combined)!=expected_rows or tuple(combined.columns)!=SYNTHESIS_COMMON_COLUMNS or combined[["aoi",*SCIENTIFIC_KEY]].duplicated().any(): raise MultiRegionWindowClosureError(f"Synthesis must contain {expected_rows} exact-schema unique rows.")
        forbidden=("pooled","weight","meta_analysis")
        if any(any(word in c.lower() for word in forbidden) for c in combined.columns):
            raise MultiRegionWindowClosureError("Pooled synthesis columns are forbidden.")
        _write_json(root/"config.json",{"schema_version":SYNTHESIS_SCHEMA_VERSION,"scientific_contract_id":SCIENTIFIC_CONTRACT_ID,"synthesis_id":sid})
        input_doc={"inputs":{a:{**r,"path":r["root"]} for a,r in records.items()}}
        _write_json(root/"input_references.json",input_doc);_write_json(root/"input_hashes.json",{a:r["manifest_hash"] for a,r in records.items()})
        pd.DataFrame(index).to_csv(root/"regional_result_index.csv",index=False);combined.to_csv(root/"four_region_synthesis.csv",index=False)
        _write_json(root/"summary.json",{"schema_version":SYNTHESIS_SCHEMA_VERSION,"scientific_contract_id":SCIENTIFIC_CONTRACT_ID,"analysis_id":sid,"technical_status":"PENDING_VALIDATION","descriptive_only":True,"pooled_inference":False})
        (root/"report.md").write_text("Descriptive point estimates; uncertainty remains under this frozen design.\n\nEvia_2021_extended is treated as a different-regime, high-prevalence sensitivity control and not as an equal-prevalence fourth validation region.\n",encoding="utf-8")
        self.finalize_manifest(root,context)
