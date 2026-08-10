"""Namespace, analysis identity, stage graph, resume binding and quarantine.

Nothing here creates a directory as a side effect of being imported or asked a
question. `namespace_root` RESOLVES a path; only `ensure_namespace` creates
one, and the dry-run never calls it.

Design reference: docs/multi_region_window_closure_design/OUTPUT_SCHEMA.md 18-19
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS,
    DIAGNOSTIC_NAMESPACE,
    EXCLUDED_AOIS,
    MultiRegionWindowClosureError,
    REFERENCE_AOI,
    SCHEMA_VERSION,
    VARIANTS,
)

# Repository status vocabulary, matching `window_closure_sensitivity.STATUS_PASS`.
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"
STATUS_VALUES: tuple[str, ...] = (STATUS_PASS, STATUS_FAIL, STATUS_SKIP)

STAGES: tuple[str, ...] = (
    "plan", "export", "local-downstream", "fit", "compare", "summarize",
)

STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "plan": (),
    "export": ("plan",),
    "local-downstream": ("export",),
    "fit": ("local-downstream",),
    "compare": ("fit",),
    "summarize": ("compare",),
}

QUARANTINE_DIR = "_quarantine"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quarantine_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# =============================================================================
# Stage graph
# =============================================================================
def validate_stage_range(from_stage: str, to_stage: str) -> list[str]:
    """Ordered stage slice. Fails closed on an unknown or inverted range."""
    for name, value in (("from-stage", from_stage), ("to-stage", to_stage)):
        if value not in STAGES:
            raise MultiRegionWindowClosureError(
                f"Unknown {name} '{value}'; expected one of {list(STAGES)}."
            )
    start, end = STAGES.index(from_stage), STAGES.index(to_stage)
    if start > end:
        raise MultiRegionWindowClosureError(
            f"from-stage '{from_stage}' comes after to-stage '{to_stage}'."
        )
    return list(STAGES[start:end + 1])


def assert_stage_prerequisites(stages: Sequence[str]) -> None:
    """Every prerequisite of a selected stage must precede it in the slice."""
    selected = list(stages)
    for stage in selected:
        for required in STAGE_REQUIRES[stage]:
            if required in selected and selected.index(required) > selected.index(stage):
                raise MultiRegionWindowClosureError(
                    f"Stage '{stage}' is ordered before its prerequisite "
                    f"'{required}'."
                )


def assert_resume_force_exclusive(resume: bool, force: bool) -> None:
    if resume and force:
        raise MultiRegionWindowClosureError(
            "--resume and --force are mutually exclusive: one reuses existing "
            "output, the other quarantines it."
        )


# =============================================================================
# Analysis identity and namespace
# =============================================================================
def build_set_configuration(
    aois: Sequence[str] = ACTUAL_AOIS,
    shifts: Optional[Sequence[int]] = None,
    canonical_hashes: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """The frozen, hashable set-level configuration."""
    from src.multi_region_window_closure.contract import (
        COMPARISON_FAMILIES, DEFAULT_SHIFTS, ESTIMAND_CLASSIFICATION,
        EXPECTED_AUXILIARY_DOWNSCALING_FITS, EXPECTED_PRIMARY_ESTIMATOR_FITS,
        EXPECTED_TOTAL_FITS, LIMITATIONS, METRICS, MODEL_FAMILIES,
        PRIMARY_ESTIMAND, PRIMARY_POPULATION, feature_registry,
        frozen_bootstrap_configuration, frozen_model_configuration,
        modis_season_policy,
    )
    from src.multi_region_window_closure.inputs import CANONICAL_STEP8A_SHA256

    shift_values = tuple(int(s) for s in (shifts if shifts is not None else DEFAULT_SHIFTS))
    return {
        "schema_version": SCHEMA_VERSION,
        "aois_actual": list(aois),
        "aoi_reference": REFERENCE_AOI,
        "aois_excluded": list(EXCLUDED_AOIS),
        "variants": list(VARIANTS),
        "shift_days": list(shift_values),
        "primary_population": PRIMARY_POPULATION,
        "model_families": list(MODEL_FAMILIES),
        "metrics": list(METRICS),
        "comparison_families": list(COMPARISON_FAMILIES),
        "estimand_classification": dict(ESTIMAND_CLASSIFICATION),
        "primary_estimand": dict(PRIMARY_ESTIMAND),
        "model_configuration": frozen_model_configuration(),
        "bootstrap_configuration": frozen_bootstrap_configuration(),
        "feature_registry": feature_registry(),
        "modis_season_policy": modis_season_policy(),
        "canonical_aoi_sha256": dict(canonical_hashes or CANONICAL_STEP8A_SHA256),
        "expected_primary_estimator_fits": EXPECTED_PRIMARY_ESTIMATOR_FITS,
        "expected_auxiliary_downscaling_fits": EXPECTED_AUXILIARY_DOWNSCALING_FITS,
        "expected_total_fits": EXPECTED_TOTAL_FITS,
        "limitations": list(LIMITATIONS),
    }


def compute_analysis_id(config: dict[str, Any]) -> str:
    """Deterministic SHA-256 over the canonical-JSON configuration.

    A timestamp is deliberately NOT an input: the same frozen configuration
    over the same canonical inputs must always resolve to the same namespace,
    so a re-run is recognisable rather than duplicated.
    """
    from src.step8_large_block_robustness import canonical_json, sha256_bytes

    return sha256_bytes(canonical_json(config).encode("utf-8"))


def diagnostics_root(output_root: Optional[Path] = None) -> Path:
    from core.paths import PROJECT_ROOT

    if output_root is not None:
        return Path(output_root)
    return Path(PROJECT_ROOT) / "outputs" / "diagnostics" / DIAGNOSTIC_NAMESPACE


def namespace_root(analysis_id: str, output_root: Optional[Path] = None) -> Path:
    """RESOLVE the analysis namespace. Creates nothing."""
    if not analysis_id:
        raise MultiRegionWindowClosureError("analysis_id must be non-empty.")
    return diagnostics_root(output_root) / analysis_id


def ensure_namespace(analysis_id: str, output_root: Optional[Path] = None) -> Path:
    """CREATE the analysis namespace. Never called by the dry-run."""
    root = namespace_root(analysis_id, output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "stages").mkdir(parents=True, exist_ok=True)
    return root


def assert_path_inside_namespace(path: Path, root: Path) -> None:
    """Refuse any write target outside this analysis's own namespace."""
    resolved = Path(path).resolve()
    base = Path(root).resolve()
    if resolved != base and base not in resolved.parents:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: NAMESPACE_ESCAPE -- {resolved} is outside {base}."
        )


def assert_not_canonical_target(path: Path) -> None:
    """Refuse any write into a canonical or reference namespace.

    The read-only reference AOI now lives under the SAME physical family root as
    the four actual AOIs (`window_closure_region/manavgat_2021`), so the
    protected set names that namespace explicitly. Unifying the physical layout
    must not make the reference writable by an ordinary regional run.

    The historical `window_closure_sensitivity` output namespace is retired but
    is kept in the protected set so nothing can silently recreate and write to
    it; the Python implementation of the same name remains the production
    scientific backend and is unaffected by this path guard.
    """
    from core.paths import PROJECT_ROOT
    from src.multi_region_window_closure.contract import (
        REFERENCE_AOI, REGIONAL_DIAGNOSTIC_NAMESPACE,
    )

    resolved = Path(path).resolve()
    project = Path(PROJECT_ROOT).resolve()
    diagnostics = project / "outputs" / "diagnostics"
    protected = [
        project / "outputs" / "experiments",
        diagnostics / "window_closure_sensitivity",
        diagnostics / REGIONAL_DIAGNOSTIC_NAMESPACE / REFERENCE_AOI,
    ]
    for base in protected:
        if resolved == base or base in resolved.parents:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: CANONICAL_OVERWRITE -- {resolved} is inside the "
                f"protected canonical namespace {base}."
            )


# =============================================================================
# Stage state and resume binding
# =============================================================================
STAGE_STATE_FIELDS: tuple[str, ...] = (
    "analysis_id", "schema_version", "stage", "status", "started_at_utc",
    "completed_at_utc", "config_hash", "input_hash", "output_manifest_hash",
    "aois_processed", "variants_processed", "gee_queries_run",
    "gee_exports_run", "model_fit", "bootstrap_run", "resume_eligible",
    "quarantined_paths", "failure_reason",
)


def new_stage_state(
    analysis_id: str,
    stage: str,
    status: str = STATUS_SKIP,
    config_hash: str = "",
    input_hash: str = "",
) -> dict[str, Any]:
    if stage not in STAGES:
        raise MultiRegionWindowClosureError(f"Unknown stage '{stage}'.")
    if status not in STATUS_VALUES:
        raise MultiRegionWindowClosureError(
            f"Unknown stage status '{status}'; expected one of {list(STATUS_VALUES)}."
        )
    return {
        "analysis_id": analysis_id,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "started_at_utc": None,
        "completed_at_utc": None,
        "config_hash": config_hash,
        "input_hash": input_hash,
        "output_manifest_hash": "",
        "aois_processed": [],
        "variants_processed": [],
        "gee_queries_run": False,
        "gee_exports_run": False,
        "model_fit": False,
        "bootstrap_run": False,
        "resume_eligible": False,
        "quarantined_paths": [],
        "failure_reason": None,
    }


def stage_is_resumable(
    state: Optional[dict[str, Any]],
    config_hash: str,
    input_hash: str,
    required_outputs_present: bool,
    output_manifest_hash: str = "",
) -> tuple[bool, str]:
    """Hash-bound resume decision. File existence alone is never sufficient.

    All five conditions must hold together; the reason string names the first
    one that fails so a refused resume is diagnosable.
    """
    if not state:
        return False, "no recorded stage state"
    if state.get("status") != STATUS_PASS:
        return False, f"recorded status is {state.get('status')!r}, not {STATUS_PASS!r}"
    if state.get("config_hash") != config_hash:
        return False, "config hash drift"
    if state.get("input_hash") != input_hash:
        return False, "input hash drift"
    if not required_outputs_present:
        return False, "required outputs are incomplete"
    if output_manifest_hash and state.get("output_manifest_hash") != output_manifest_hash:
        return False, "output manifest hash drift"
    aois = list(state.get("aois_processed") or [])
    if sorted(aois) != sorted(ACTUAL_AOIS):
        return False, f"partial AOI set {sorted(aois)}"
    variants = list(state.get("variants_processed") or [])
    if sorted(variants) != sorted(VARIANTS):
        return False, f"partial variant set {sorted(variants)}"
    return True, "hash-bound reuse"


def assert_stage_completeness(state: dict[str, Any]) -> None:
    """A stage that processed a subset may never be recorded as PASS."""
    if state.get("status") != STATUS_PASS:
        return
    aois = sorted(state.get("aois_processed") or [])
    variants = sorted(state.get("variants_processed") or [])
    if aois != sorted(ACTUAL_AOIS):
        raise MultiRegionWindowClosureError(
            f"BLOCKER: PARTIAL_AOI -- stage '{state.get('stage')}' is PASS but "
            f"processed {aois}, expected {sorted(ACTUAL_AOIS)}."
        )
    if variants != sorted(VARIANTS):
        raise MultiRegionWindowClosureError(
            f"BLOCKER: PARTIAL_STAGE -- stage '{state.get('stage')}' is PASS "
            f"but processed variants {variants}, expected {sorted(VARIANTS)}."
        )


# =============================================================================
# Quarantine
# =============================================================================
def quarantine_target(
    namespace: Path, relative_path: str, reason: str, timestamp: Optional[str] = None,
) -> Path:
    """`<namespace>/_quarantine/<timestamp>_<reason>/<relative path>`."""
    safe_reason = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(reason)
    ).strip("_") or "unspecified"
    stamp = timestamp or quarantine_timestamp()
    return Path(namespace) / QUARANTINE_DIR / f"{stamp}_{safe_reason}" / relative_path


def quarantine_existing(
    namespace: Path,
    relative_paths: Sequence[str],
    reason: str,
    timestamp: Optional[str] = None,
) -> list[dict[str, str]]:
    """MOVE existing outputs into quarantine. Never deletes.

    Only paths inside this analysis's own namespace can be quarantined; a
    canonical or foreign path raises before anything is touched.
    """
    import shutil

    root = Path(namespace)
    assert_not_canonical_target(root)
    stamp = timestamp or quarantine_timestamp()
    moved: list[dict[str, str]] = []
    for relative in relative_paths:
        source = root / relative
        assert_path_inside_namespace(source, root)
        assert_not_canonical_target(source)
        if not source.exists():
            continue
        target = quarantine_target(root, relative, reason, stamp)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise MultiRegionWindowClosureError(
                f"Quarantine target already exists: {target}. Refusing to "
                "overwrite a quarantined artefact."
            )
        shutil.move(str(source), str(target))
        moved.append({
            "relative_path": str(relative),
            "quarantined_to": str(target.relative_to(root)),
            "reason": str(reason),
            "timestamp": stamp,
        })
    return moved


# =============================================================================
# Filesystem snapshot (used to prove the dry-run is read-only)
# =============================================================================
def snapshot_tree(root: Path) -> dict[str, int]:
    """`{relative path: size}` for every file under `root`. Missing tree -> {}."""
    base = Path(root)
    if not base.exists():
        return {}
    out: dict[str, int] = {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(base))] = path.stat().st_size
    return out


def snapshot_diff(before: dict[str, int], after: dict[str, int]) -> dict[str, list[str]]:
    """Created / modified / deleted paths between two snapshots."""
    created = sorted(set(after) - set(before))
    deleted = sorted(set(before) - set(after))
    modified = sorted(
        p for p in set(before) & set(after) if before[p] != after[p]
    )
    return {"created": created, "modified": modified, "deleted": deleted}


def snapshot_unchanged(diff: dict[str, list[str]]) -> bool:
    return not (diff["created"] or diff["modified"] or diff["deleted"])
