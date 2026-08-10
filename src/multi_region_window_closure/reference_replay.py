"""Manavgat reference replay: frozen result re-produced under the regional architecture.

WHAT THIS IS
------------
`manavgat_2021` is the frozen, already-reviewed window-closure result. It was
produced by the per-AOI runner `src.window_closure_sensitivity.run_analysis`
into the historical physical layout

    outputs/diagnostics/window_closure_sensitivity/manavgat_2021/

The four actual AOIs are produced by the newer per-AOI regional architecture

    outputs/diagnostics/window_closure_region/<aoi>/<analysis_id>/

whose `ProductionRegionalEngine` delegates every scientific stage back to that
same `run_analysis`, writing into `<analysis_id>/_production/<aoi>/` -- i.e. the
regional namespace physically CONTAINS the historical layout and wraps it with
manifest / validator / provenance artefacts.

This module replays the frozen Manavgat result through that regional wrapper so
the two can be compared field by field. It answers one question:

    does the regional architecture preserve the frozen Manavgat science?

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* No Earth Engine, no remote export, no new download. The scientific stages
  (`plan`, `export`, `local-downstream`, `fit`, `compare`) REUSE the frozen
  artefacts verbatim -- each file is copied into the replay namespace and the
  copy is verified byte-for-byte against its source.
* No write of any kind into the frozen Manavgat tree or into the four existing
  regional AOI trees. The frozen tree is opened read-only.
* No change to `ACTUAL_AOIS`, to `REFERENCE_AOI`, or to what `aoi_role` reports.
  Manavgat remains `read_only_reference` everywhere.

The `summarize` and `validate` stages are NOT reused: they are the regional
wrapper itself, and they execute for real
(`production.normalize_production_regional_outputs` and
`validation.evaluate_regional`, the same functions the four AOIs ran).

Design references
-----------------
docs/multi_region_window_closure_design/SCIENTIFIC_CONTRACT.md
src/multi_region_window_closure/driver.py     (the seven regional stages)
src/multi_region_window_closure/production.py (the scientific delegation)
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.multi_region_window_closure.contract import (
    MultiRegionWindowClosureError, REFERENCE_AOI, VARIANTS, aoi_role,
    frozen_bootstrap_configuration, frozen_model_configuration,
    reference_replay_scope,
)
from src.multi_region_window_closure.inputs import CANONICAL_STEP8A_SHA256
from src.multi_region_window_closure.production import (
    ProductionRegionalEngine, normalize_production_regional_outputs,
)

#: Replay namespace root. Distinct from BOTH canonical roots so neither the
#: frozen tree nor the four production regional trees can be reached by it.
REPLAY_DIAGNOSTIC_NAMESPACE = "window_closure_region_replay"

#: Canonical (frozen, historical) Manavgat namespace, opened read-only.
FROZEN_DIAGNOSTIC_NAMESPACE = "window_closure_sensitivity"

REPLAY_MODE_ID = "manavgat_reference_replay.v1"

#: The scientific stages whose outputs are REUSED from the frozen tree. The
#: remaining two regional stages (`summarize`, `validate`) really execute.
REPLAYED_STAGES: tuple[str, ...] = (
    "plan", "export", "local-downstream", "fit", "compare",
)
EXECUTED_STAGES: tuple[str, ...] = ("summarize", "validate")

#: Sub-tree excluded from the replay. `_quarantine` holds superseded artefacts
#: that an earlier forced run moved aside; they are not stage outputs and the
#: four production regional namespaces contain no equivalent.
EXCLUDED_TREE_COMPONENT = "_quarantine"

#: The ONLY field of `scientific_configuration` allowed to differ between the
#: frozen preregistration and a freshly derived plan. It records which commit
#: produced the document and is not a scientific parameter.
NON_SCIENTIFIC_CONFIG_FIELDS: frozenset[str] = frozenset({"git_commit"})

#: Absolute float tolerance used ONLY to absorb CSV/parquet serialization of a
#: double. Anything larger is reported as a material difference.
FLOAT_SERIALIZATION_TOLERANCE = 1e-12

CLASS_EXACT = "EXACT_FILE_MATCH"
CLASS_SCIENTIFIC = "SCIENTIFICALLY_IDENTICAL"
CLASS_FLOAT = "EQUIVALENT_WITH_FLOAT_TOLERANCE"
CLASS_MATERIAL = "MATERIAL_DIFFERENCE"


class ReferenceReplayError(MultiRegionWindowClosureError):
    """Fail-closed replay condition."""


# =============================================================================
# Paths
# =============================================================================
def frozen_manavgat_root(output_root: Optional[Path] = None) -> Path:
    """Read-only root of the frozen historical Manavgat result."""
    from core.paths import PROJECT_ROOT

    root = Path(output_root) if output_root else (
        Path(PROJECT_ROOT) / "outputs" / "diagnostics" / FROZEN_DIAGNOSTIC_NAMESPACE
    )
    return Path(root) / REFERENCE_AOI


def replay_output_root(output_root: Optional[Path] = None) -> Path:
    from core.paths import PROJECT_ROOT

    if output_root:
        return Path(output_root)
    return Path(PROJECT_ROOT) / "outputs" / "diagnostics" / REPLAY_DIAGNOSTIC_NAMESPACE


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Frozen source inventory and stage partition
# =============================================================================
def frozen_source_files(source: Path) -> list[Path]:
    """Every non-quarantine file of the frozen tree, deterministically ordered."""
    root = Path(source)
    if not root.is_dir():
        raise ReferenceReplayError(
            f"BLOCKER: FROZEN_SOURCE_MISSING -- {root} is not a directory."
        )
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and EXCLUDED_TREE_COMPONENT not in path.relative_to(root).parts
    )


def stage_of_frozen_artifact(relative: Path) -> str:
    """Which regional stage owns a frozen artefact. Fails closed on anything new.

    The partition mirrors `production.PRODUCTION_STAGE_MAP`: the regional stage
    names on the left, the per-AOI stage that produced the file on the right.
    """
    parts = relative.parts
    head = parts[0]
    if head == "config":
        return "plan"
    if head == "prelabel_censor":
        return "export"
    if head == "model":
        return "fit"
    if head == "compare":
        return "compare"
    if head == "variants":
        name = parts[-1]
        if name in {"frozen_reference.json", "export_plan.json"}:
            return "plan"
        if name == "predictor_export_metadata.json" or (len(parts) > 2 and parts[2] == "data"):
            return "export"
        if name == "local_downstream_metadata.json" or (len(parts) > 2 and parts[2] == "downstream"):
            return "local-downstream"
    raise ReferenceReplayError(
        f"BLOCKER: UNCLASSIFIED_FROZEN_ARTIFACT -- '{relative}' belongs to no "
        "known window-closure stage. The replay refuses to copy an artefact it "
        "cannot attribute to a stage."
    )


def partition_frozen_source(source: Path) -> dict[str, list[Path]]:
    """`{stage: [relative path, ...]}` covering every non-quarantine file exactly once."""
    root = Path(source)
    partition: dict[str, list[Path]] = {stage: [] for stage in REPLAYED_STAGES}
    for path in frozen_source_files(root):
        relative = path.relative_to(root)
        partition[stage_of_frozen_artifact(relative)].append(relative)
    covered = sum(len(v) for v in partition.values())
    total = len(frozen_source_files(root))
    if covered != total:
        raise ReferenceReplayError(
            f"BLOCKER: STAGE_PARTITION_INCOMPLETE -- {covered} of {total} frozen "
            "artefacts were attributed to a stage."
        )
    return partition


# =============================================================================
# Section 2 -- machine-checkable contract preflight
# =============================================================================
def _frozen_preregistration(source: Path) -> dict[str, Any]:
    path = Path(source) / "config" / "preregistration.json"
    if not path.is_file():
        raise ReferenceReplayError(
            f"BLOCKER: FROZEN_PREREGISTRATION_MISSING -- {path}."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _cmp(name: str, old: Any, new: Any, *, scientific: bool = True) -> dict[str, Any]:
    equal = old == new
    return {
        "field": name, "old": old, "new": new, "equal": bool(equal),
        "scientific": bool(scientific),
        "status": "MATCH" if equal else ("MISMATCH" if scientific else "DIFFERS_NON_SCIENTIFIC"),
    }


def replay_contract_preflight(
    source: Optional[Path] = None, experiments_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Every frozen scientific field vs. the field a fresh regional plan derives.

    Read-only. Returns `status='PASS'` only when every scientific field is
    equal; a single mismatch means REPLAY NOT AUTHORIZED.
    """
    from src.multi_region_window_closure.dates import window_date_rows
    from src.window_closure_sensitivity import run_analysis

    root = Path(source) if source else frozen_manavgat_root()
    frozen = _frozen_preregistration(root)
    old_config = frozen["scientific_configuration"]

    plan = run_analysis(
        experiment_id=REFERENCE_AOI, dry_run=True, from_stage="plan",
        to_stage="compare", experiments_root=experiments_root,
    )
    new_config = plan["scientific_configuration"]

    comparisons: list[dict[str, Any]] = []
    for field in sorted(set(old_config) | set(new_config)):
        comparisons.append(_cmp(
            f"scientific_configuration.{field}",
            old_config.get(field, "<absent>"), new_config.get(field, "<absent>"),
            scientific=field not in NON_SCIENTIFIC_CONFIG_FIELDS,
        ))

    # --- the explicitly enumerated section-2 fields, re-derived independently -
    old_variants = {v["variant_id"]: v for v in old_config["variants"]}
    new_rows = {row["variant"]: row for row in window_date_rows((REFERENCE_AOI,), (0, 7, 14))}
    if set(old_variants) != set(VARIANTS) or set(new_rows) != set(VARIANTS):
        raise ReferenceReplayError(
            "BLOCKER: VARIANT_SET_MISMATCH -- frozen "
            f"{sorted(old_variants)} vs derived {sorted(new_rows)}."
        )
    for variant in VARIANTS:
        old_v, new_v = old_variants[variant], new_rows[variant]
        comparisons.append(_cmp(f"{variant}.predictor_start", old_v["predictor_start_date"], new_v["predictor_start"]))
        comparisons.append(_cmp(f"{variant}.predictor_end", old_v["predictor_end_date"], new_v["predictor_end"]))
        comparisons.append(_cmp(f"{variant}.duration_days", old_v["duration_days"], new_v["calendar_duration_days"]))
        comparisons.append(_cmp(f"{variant}.lead_days", int(old_v["lead_days"]), int(new_v["lead_days"])))
        comparisons.append(_cmp(f"{variant}.shift_days", int(old_v["shift_days"]), int(new_v["shift_days"])))
        # The label window is frozen across variants in BOTH architectures, so
        # each variant's derived label dates are checked against the single
        # frozen label window rather than against a per-variant field.
        comparisons.append(_cmp(f"{variant}.label_start", old_config["label_window"]["start_date"], new_v["label_start"]))
        comparisons.append(_cmp(f"{variant}.label_end", old_config["label_window"]["end_date"], new_v["label_end"]))

    comparisons.append(_cmp("population.primary_population", old_config["primary_population"],
                            frozen_model_configuration()["primary_population"]))
    # The per-AOI `scientific_configuration.model_configuration` and the
    # regional `frozen_model_configuration()` are two documents of DIFFERENT
    # shape over the same knobs: the regional one declares extra fields the
    # per-AOI one leaves implicit, and it renames `random_seed` to
    # `fold_random_seed`. Comparing them whole would report a shape difference
    # as a scientific one, so the shared parameters are compared by name.
    regional_model = frozen_model_configuration()
    for old_key, new_key in (
        ("model", "model"), ("n_splits", "n_splits"),
        ("random_seed", "fold_random_seed"),
        ("spatial_block_size_cells", "spatial_block_size_cells"),
        ("min_positives", "min_positives"),
    ):
        comparisons.append(_cmp(
            f"model_configuration.{old_key}",
            old_config["model_configuration"][old_key], regional_model[new_key],
        ))
    regional_bootstrap = frozen_bootstrap_configuration()
    for key in ("unit", "n_bootstrap", "seed", "identical_block_draws_across_variants"):
        comparisons.append(_cmp(
            f"bootstrap_configuration.{key}",
            old_config["bootstrap_configuration"][key], regional_bootstrap[key],
        ))
    comparisons.append(_cmp("feature_registry", old_config["feature_registry"],
                            new_config["feature_registry"]))
    comparisons.append(_cmp("common_cohort_rule", old_config["common_cohort_rule"],
                            new_config["common_cohort_rule"]))

    # --- frozen Step8A identity, recomputed from bytes -----------------------
    step8a = Path(plan["frozen_canonical_step8a"]["path"])
    step8a_actual = sha256_path(step8a) if step8a.is_file() else None
    frozen_inventory = json.loads(
        (root / "config" / "frozen_input_inventory.json").read_text(encoding="utf-8")
    )
    comparisons.append(_cmp(
        "step8a.path",
        str(Path(frozen_inventory["inventory"]["canonical_step8a"]["path"]).resolve()),
        str(step8a.resolve()),
    ))
    comparisons.append(_cmp("step8a.sha256_frozen_vs_central",
                            old_config["frozen_input_sha256"]["canonical_step8a"],
                            CANONICAL_STEP8A_SHA256[REFERENCE_AOI]))
    comparisons.append(_cmp("step8a.sha256_frozen_vs_actual_bytes",
                            old_config["frozen_input_sha256"]["canonical_step8a"], step8a_actual))
    comparisons.append(_cmp("aoi_role", aoi_role(REFERENCE_AOI), "read_only_reference"))

    mismatches = [c for c in comparisons if c["scientific"] and not c["equal"]]
    non_scientific = [c for c in comparisons if not c["scientific"] and not c["equal"]]
    return {
        "mode": REPLAY_MODE_ID,
        "aoi": REFERENCE_AOI,
        "aoi_role": aoi_role(REFERENCE_AOI),
        "frozen_source": str(root),
        "frozen_analysis_id": frozen.get("analysis_id"),
        "derived_production_analysis_id": plan["analysis_id"],
        "comparisons": comparisons,
        "scientific_field_count": sum(1 for c in comparisons if c["scientific"]),
        "mismatches": mismatches,
        "non_scientific_differences": non_scientific,
        "status": "PASS" if not mismatches else "STOP",
        "verdict": (
            "REPLAY AUTHORIZED" if not mismatches
            else "STOP -- REPLAY NOT AUTHORIZED"
        ),
        "gee_queries_run": False,
        "files_written": False,
    }


# =============================================================================
# Replay engine
# =============================================================================
class ManavgatReferenceReplayEngine(ProductionRegionalEngine):
    """Regional engine whose scientific stages reuse verified frozen artefacts.

    Everything the regional wrapper owns -- identity, staging, summarize,
    manifest, validator -- is inherited unchanged from
    `ProductionRegionalEngine`. Only the five scientific stages are replaced,
    and they are replaced by a verified copy rather than by a recomputation, so
    the replay cannot reach Earth Engine or produce a different number.
    """

    def __init__(
        self, *, frozen_source: Optional[Path] = None,
        experiments_root: Optional[Path] = None,
    ) -> None:
        super().__init__(aoi=REFERENCE_AOI, experiments_root=experiments_root)
        self.frozen_source = Path(frozen_source) if frozen_source else frozen_manavgat_root()
        self.partition = partition_frozen_source(self.frozen_source)
        self.materialized: dict[str, list[dict[str, str]]] = {}

    def _production_root(self, regional_root: Path) -> Path:
        """As the production engine, plus: never inside a canonical namespace.

        The base guard refuses the synthesis namespace and the experiments root.
        A replay must additionally never be able to write into the frozen
        Manavgat namespace it is reading, nor into the four production regional
        namespaces, so both are refused here by path.
        """
        target = super()._production_root(regional_root)
        text = str(Path(target).resolve())
        for forbidden in (f"/{FROZEN_DIAGNOSTIC_NAMESPACE}/", "/window_closure_region/"):
            if forbidden in text:
                raise ReferenceReplayError(
                    "BLOCKER: REPLAY_WRITE_TARGET_FORBIDDEN -- a reference replay "
                    f"may never write inside '{forbidden.strip('/')}'; got {text}."
                )
        return target

    # -- the five reused stages ------------------------------------------
    def _materialize(self, stage: str, root: Path, context: Mapping[str, Any]) -> dict[str, Any]:
        target_root = self._production_root(Path(root)) / self.aoi
        records: list[dict[str, str]] = []
        for relative in self.partition[stage]:
            src = self.frozen_source / relative
            dst = target_root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            src_hash = sha256_path(src)
            dst_hash = sha256_path(dst)
            if src_hash != dst_hash:
                raise ReferenceReplayError(
                    "BLOCKER: FROZEN_REUSE_HASH_MISMATCH -- copying "
                    f"{relative} produced {dst_hash}, source is {src_hash}."
                )
            records.append({
                "relative_path": str(relative), "sha256": src_hash,
                "source": str(src), "size_bytes": str(src.stat().st_size),
            })
        self.materialized[stage] = records
        return {
            "replay_stage": stage,
            "replay_mode": REPLAY_MODE_ID,
            "scientific_work": "reused_frozen_artifacts",
            "recomputed": False,
            "gee_queries_run": False,
            "gee_exports_run": False,
            "frozen_source": str(self.frozen_source),
            "reused_artifact_count": len(records),
            "reused_artifact_bytes": sum(int(r["size_bytes"]) for r in records),
            "reuse_verification": "sha256(source) == sha256(copy) for every artefact",
            "output_inventory": [r["relative_path"] for r in records],
        }

    def run_stage(self, stage: str, root: Path, context: dict[str, Any]) -> dict[str, Any]:
        if context.get("aoi") != self.aoi:
            raise ReferenceReplayError("Replay engine AOI does not match stage context.")
        if stage == "summarize":
            detail = normalize_production_regional_outputs(Path(root), context)
            return {**detail, "replay_mode": REPLAY_MODE_ID, "recomputed": True}
        if stage not in self.partition:
            raise ReferenceReplayError(f"Unknown replay stage: {stage}")
        return self._materialize(stage, Path(root), context)


# =============================================================================
# Section 12 -- source mutation guard
# =============================================================================
def tree_snapshot(root: Path, critical: Sequence[str] = ()) -> dict[str, Any]:
    """Cheap-but-strong immutability fingerprint of a tree.

    Full-tree hashing of tens of gigabytes for every guard would dominate the
    run, so the fingerprint is (relative path, size, mtime_ns) for every file
    plus a real SHA-256 of the named critical files.
    """
    root = Path(root)
    if not root.exists():
        return {"root": str(root), "exists": False}
    entries: list[str] = []
    count = 0
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        entries.append(f"{path.relative_to(root)}|{stat.st_size}|{stat.st_mtime_ns}")
        count += 1
        total += stat.st_size
    return {
        "root": str(root), "exists": True, "file_count": count,
        "total_bytes": total,
        "structure_digest": hashlib.sha256("\n".join(entries).encode()).hexdigest(),
        "critical_file_sha256": {
            name: (sha256_path(root / name) if (root / name).is_file() else None)
            for name in critical
        },
    }


FROZEN_CRITICAL_FILES: tuple[str, ...] = (
    "config/preregistration.json",
    "config/frozen_input_inventory.json",
    "model/metrics/point_metrics.csv",
    "model/bootstrap/paired_bootstrap_summary.csv",
    "model/common_cohort/common_cohort_metadata.json",
    "model/shared_folds/shared_spatial_folds_metadata.json",
    "compare/compare_stage_metadata.json",
)

REGIONAL_CRITICAL_FILES: tuple[str, ...] = (
    "summary.json", "manifest.sha256", "validator_summary.json",
    "regional_summary.csv", "bootstrap_summary.csv", "metrics.csv",
)


def guarded_trees(output_root: Optional[Path] = None) -> dict[str, tuple[Path, tuple[str, ...]]]:
    """Every tree that must be left untouched, with its critical files.

    Since the 2026-08-10 migration the read-only reference lives under the SAME
    regional root as the four actual AOIs, so it is guarded there. The retired
    `window_closure_sensitivity` path is still guarded when it exists, so a
    pre-migration tree is not silently left unprotected -- but its absence is
    not treated as "nothing to protect".
    """
    from core.paths import PROJECT_ROOT
    from src.multi_region_window_closure.contract import ACTUAL_AOIS

    regional_root = Path(PROJECT_ROOT) / "outputs" / "diagnostics" / "window_closure_region"
    trees: dict[str, tuple[Path, tuple[str, ...]]] = {}
    legacy = frozen_manavgat_root(output_root)
    if legacy.exists():
        trees[f"frozen/{REFERENCE_AOI}"] = (legacy, FROZEN_CRITICAL_FILES)
    for aoi in (*ACTUAL_AOIS, REFERENCE_AOI):
        aoi_root = regional_root / aoi
        analyses = sorted(p for p in aoi_root.glob("*") if p.is_dir() and not p.name.startswith("_"))
        for analysis in analyses:
            trees[f"regional/{aoi}/{analysis.name}"] = (analysis, REGIONAL_CRITICAL_FILES)
    return trees


def snapshot_guarded_trees(output_root: Optional[Path] = None) -> dict[str, Any]:
    return {
        name: tree_snapshot(path, critical)
        for name, (path, critical) in guarded_trees(output_root).items()
    }


def compare_guarded_snapshots(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    changed: list[dict[str, Any]] = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if old != new:
            changed.append({
                "tree": name,
                "before": {k: v for k, v in (old or {}).items() if k != "critical_file_sha256"},
                "after": {k: v for k, v in (new or {}).items() if k != "critical_file_sha256"},
            })
    return {
        "trees_checked": sorted(set(before) | set(after)),
        "changed_trees": changed,
        "all_unchanged": not changed,
        "method": "(path,size,mtime_ns) structure digest + SHA-256 of critical files",
    }


# =============================================================================
# Section 7/8 -- old vs new scientific equivalence
# =============================================================================
def _classify(differences: Sequence[Mapping[str, Any]]) -> str:
    """Worst classification implied by a list of per-value differences."""
    if not differences:
        return CLASS_SCIENTIFIC
    if any(abs(float(d["abs_diff"])) > FLOAT_SERIALIZATION_TOLERANCE for d in differences):
        return CLASS_MATERIAL
    return CLASS_FLOAT


def _numeric_diffs(
    label: str, old_values: Mapping[Any, float], new_values: Mapping[Any, float],
) -> dict[str, Any]:
    """Key-aligned numeric comparison of two labelled series."""
    old_keys, new_keys = set(old_values), set(new_values)
    missing = sorted(str(k) for k in old_keys - new_keys)
    extra = sorted(str(k) for k in new_keys - old_keys)
    differences: list[dict[str, Any]] = []
    exact = 0
    for key in sorted(old_keys & new_keys, key=str):
        old, new = float(old_values[key]), float(new_values[key])
        if old == new:
            exact += 1
            continue
        abs_diff = abs(new - old)
        differences.append({
            "key": str(key), "old": old, "new": new, "abs_diff": abs_diff,
            "rel_diff": (abs_diff / abs(old)) if old else None,
        })
    classification = CLASS_MATERIAL if (missing or extra) else _classify(differences)
    return {
        "comparison": label,
        "row_key_equal": not missing and not extra,
        "missing_keys": missing, "extra_keys": extra,
        "compared_values": len(old_keys & new_keys),
        "exact_equal_values": exact,
        "differing_values": len(differences),
        "max_abs_diff": max((d["abs_diff"] for d in differences), default=0.0),
        "max_rel_diff": max((d["rel_diff"] or 0.0 for d in differences), default=0.0),
        "differences": differences[:20],
        "classification": classification,
        "equivalent": classification in {CLASS_EXACT, CLASS_SCIENTIFIC, CLASS_FLOAT},
    }


def compare_frozen_reuse(frozen_source: Path, replay_root: Path) -> dict[str, Any]:
    """Byte comparison of every reused scientific artefact (section 8)."""
    frozen_source = Path(frozen_source)
    production = Path(replay_root) / "_production" / REFERENCE_AOI
    expected = {p.relative_to(frozen_source): p for p in frozen_source_files(frozen_source)}
    present = {
        p.relative_to(production): p
        for p in sorted(production.rglob("*")) if p.is_file()
    } if production.is_dir() else {}
    missing = sorted(str(r) for r in set(expected) - set(present))
    extra = sorted(str(r) for r in set(present) - set(expected))
    mismatched: list[dict[str, str]] = []
    identical = 0
    for relative in sorted(set(expected) & set(present), key=str):
        old_hash = sha256_path(expected[relative])
        new_hash = sha256_path(present[relative])
        if old_hash == new_hash:
            identical += 1
        else:
            mismatched.append({
                "relative_path": str(relative), "old_sha256": old_hash,
                "new_sha256": new_hash,
            })
    quarantined = sorted(
        str(p.relative_to(frozen_source)) for p in frozen_source.rglob("*")
        if p.is_file() and EXCLUDED_TREE_COMPONENT in p.relative_to(frozen_source).parts
    )
    return {
        "comparison": "frozen_scientific_artifacts_byte_identity",
        "frozen_file_count": len(expected),
        "replayed_file_count": len(present),
        "byte_identical": identical,
        "missing_in_replay": missing,
        "unexpected_in_replay": extra,
        "hash_mismatches": mismatched,
        "excluded_quarantine_file_count": len(quarantined),
        "excluded_quarantine_note": (
            "`_quarantine/` holds artefacts an earlier forced per-AOI run moved "
            "aside. It is not a stage output, the four production regional "
            "namespaces contain no equivalent, and it is excluded by design."
        ),
        "classification": (
            CLASS_EXACT if identical == len(expected) and not missing and not extra
            and not mismatched else CLASS_MATERIAL
        ),
        "equivalent": (
            identical == len(expected) and not missing and not extra and not mismatched
        ),
    }


def compare_population(frozen_source: Path, replay_root: Path) -> dict[str, Any]:
    """Cohort, label and block accounting (section 7B)."""
    import pandas as pd

    frozen_source, replay_root = Path(frozen_source), Path(replay_root)
    meta = json.loads(
        (frozen_source / "model" / "common_cohort" / "common_cohort_metadata.json")
        .read_text(encoding="utf-8")
    )
    folds = pd.read_parquet(
        frozen_source / "model" / "shared_folds" / "shared_spatial_folds.parquet"
    )
    inventory = pd.read_csv(replay_root / "cohort_inventory.csv")
    regional = pd.read_csv(replay_root / "regional_summary.csv")
    fold_map = pd.read_parquet(replay_root / "fold_mapping.parquet")

    rows: list[dict[str, Any]] = []

    def add(field: str, old: Any, new: Any) -> None:
        rows.append({"field": field, "old": old, "new": new, "equal": old == new})

    add("final_common_cohort_rows", int(meta["final_common_cohort_rows"]),
        int(inventory["final_common_cohort_rows"].iloc[0]))
    add("final_positive_rows", int(meta["final_positive_rows"]),
        int(inventory["final_positive_rows"].iloc[0]))
    add("final_negative_rows", int(meta["final_negative_rows"]),
        int(inventory["final_negative_rows"].iloc[0]))
    add("prevalence", round(float(meta["prevalence"]), 12),
        round(float(inventory["prevalence"].iloc[0]), 12))
    add("cohort_rows_regional_summary", int(meta["final_common_cohort_rows"]),
        int(regional["cohort_rows"].iloc[0]))
    add("positives_regional_summary", int(meta["final_positive_rows"]),
        int(regional["positives"].iloc[0]))
    add("negatives_regional_summary", int(meta["final_negative_rows"]),
        int(regional["negatives"].iloc[0]))
    add("spatial_block_count", int(folds["spatial_block_id"].nunique()),
        int(regional["block_count"].iloc[0]))
    add("fold_count", int(folds["fold_id"].nunique()), int(regional["fold_count"].iloc[0]))
    add("fold_mapping_rows", len(folds), len(fold_map))
    add("cohort_inventory_variant_rows", len(VARIANTS), len(inventory))

    from src.multi_region_window_closure.production import (
        COHORT_PER_VARIANT_REMOVALS, COHORT_SHARED_REMOVALS,
    )
    for variant in VARIANTS:
        variant_row = inventory[inventory["variant"] == variant].iloc[0]
        add(f"{variant}.initial_rows",
            int(meta["initial_rows_by_variant"][variant]), int(variant_row["initial_rows"]))
        for field in COHORT_PER_VARIANT_REMOVALS:
            add(f"{variant}.{field}", int(meta[field][variant]), int(variant_row[field]))
        for field in COHORT_SHARED_REMOVALS:
            add(f"{variant}.{field}", int(meta[field]), int(variant_row[field]))

    unequal = [r for r in rows if not r["equal"]]
    return {
        "comparison": "population_and_accounting",
        "fields_compared": len(rows), "unequal_fields": unequal,
        "rows": rows,
        "classification": CLASS_SCIENTIFIC if not unequal else CLASS_MATERIAL,
        "equivalent": not unequal,
    }


def compare_fit_accounting(replay_root: Path) -> dict[str, Any]:
    """Primary/auxiliary fit accounting derived by the wrapper (section 7B)."""
    config = json.loads((Path(replay_root) / "config.json").read_text(encoding="utf-8"))
    fit = config["fit_accounting"]
    expected = {
        "expected_primary_estimator_fits": 30, "completed_primary_estimator_fits": 30,
        "expected_auxiliary_downscaling_fits": 2, "completed_auxiliary_downscaling_fits": 2,
        "expected_total_fits": 32, "completed_total_fits": 32,
    }
    unequal = [
        {"field": k, "expected": v, "actual": fit.get(k)}
        for k, v in expected.items() if fit.get(k) != v
    ]
    return {
        "comparison": "fit_accounting",
        "expected": expected,
        "actual": {k: fit.get(k) for k in expected},
        "unequal_fields": unequal,
        "classification": CLASS_SCIENTIFIC if not unequal else CLASS_MATERIAL,
        "equivalent": not unequal,
    }


def compare_point_metrics(frozen_source: Path, replay_root: Path) -> dict[str, Any]:
    """Headline ROC-AUC / PR-AUC / Brier per variant x model (section 7D)."""
    import pandas as pd

    old = pd.read_csv(Path(frozen_source) / "model" / "metrics" / "point_metrics.csv")
    new = pd.read_csv(Path(replay_root) / "metrics.csv")
    old_values: dict[tuple[str, str, str], float] = {}
    for _, row in old.iterrows():
        for metric in ("roc_auc", "pr_auc", "brier"):
            old_values[(row["variant_id"], row["model_family"], metric)] = float(row[metric])
    new_values = {
        (row["variant"], row["model"], row["metric"]): float(row["estimate"])
        for _, row in new.iterrows()
    }
    result = _numeric_diffs("point_metrics", old_values, new_values)
    result["side_by_side"] = [
        {
            "variant": key[0], "model": key[1], "metric": key[2],
            "old": old_values[key], "new": new_values.get(key),
            "abs_diff": abs(new_values[key] - old_values[key]) if key in new_values else None,
        }
        for key in sorted(old_values, key=str)
    ]
    return result


def compare_bootstrap_summary(frozen_source: Path, replay_root: Path) -> dict[str, Any]:
    """27 comparison series x point estimate and both CI bounds (section 7D/E)."""
    import pandas as pd

    old = pd.read_csv(
        Path(frozen_source) / "model" / "bootstrap" / "paired_bootstrap_summary.csv"
    )
    new = pd.read_csv(Path(replay_root) / "bootstrap_summary.csv")
    old_values: dict[str, float] = {}
    for _, row in old.iterrows():
        stem = f"{row['comparison']}|{row['variant_id']}|{row['model_family']}|{row['metric']}"
        old_values[f"{stem}|point"] = float(row["point_delta"])
        old_values[f"{stem}|ci_low"] = float(row["ci_low"])
        old_values[f"{stem}|ci_high"] = float(row["ci_high"])
        old_values[f"{stem}|bootstrap_mean"] = float(row["bootstrap_mean"])
        old_values[f"{stem}|valid_replicates"] = float(row["valid_replicates"])
        old_values[f"{stem}|block_count"] = float(row["block_count"])
    new_values: dict[str, float] = {}
    for _, row in new.iterrows():
        stem = f"{row['comparison']}|{row['variant_id']}|{row['model_family']}|{row['metric']}"
        new_values[f"{stem}|point"] = float(row["point_estimate_natural"])
        new_values[f"{stem}|ci_low"] = float(row["ci_low_natural"])
        new_values[f"{stem}|ci_high"] = float(row["ci_high_natural"])
        new_values[f"{stem}|bootstrap_mean"] = float(row["bootstrap_mean_natural"])
        new_values[f"{stem}|valid_replicates"] = float(row["valid_replicates"])
        new_values[f"{stem}|block_count"] = float(row["block_count"])
    result = _numeric_diffs("paired_bootstrap_summary", old_values, new_values)
    result["old_series"] = len(old)
    result["new_series"] = len(new)
    result["series_count_equal"] = len(old) == len(new) == 27
    if not result["series_count_equal"]:
        result["classification"] = CLASS_MATERIAL
        result["equivalent"] = False
    return result


def compare_bootstrap_replicates(frozen_source: Path, replay_root: Path) -> dict[str, Any]:
    """Every replicate of every series, re-derived from the frozen wide table.

    The regional wrapper stores replicates in long form; the frozen per-AOI
    stage stores the same draws in wide form. Both are compared draw by draw,
    so a re-ordered or re-seeded bootstrap could not pass.
    """
    import pandas as pd

    wide = pd.read_parquet(
        Path(frozen_source) / "model" / "bootstrap" / "paired_bootstrap_replicates.parquet"
    )
    long = pd.read_parquet(Path(replay_root) / "bootstrap_replicates.parquet")
    summary = pd.read_csv(Path(replay_root) / "bootstrap_summary.csv")

    old_values: dict[str, float] = {}
    new_values: dict[str, float] = {}
    for _, row in summary.iterrows():
        variant, model = str(row["variant_id"]), str(row["model_family"])
        metric, family = str(row["metric"]), str(row["comparison"])

        def column(v: str, m: str) -> str:
            return f"{v}__{m}_{metric}"

        if family == "thermal_contribution_within_variant":
            series = wide[column(variant, "thermal")] - wide[column(variant, "baseline")]
        elif family == "closure_change_within_model_family":
            series = wide[column(variant, model)] - wide[column("canonical", model)]
        else:
            series = (
                (wide[column(variant, "thermal")] - wide[column(variant, "baseline")])
                - (wide[column("canonical", "thermal")] - wide[column("canonical", "baseline")])
            )
        key = str(row["comparison_series"])
        subset = long[long["comparison_series"] == key].sort_values("replicate_id")
        for replicate, value in enumerate(series.tolist()):
            old_values[f"{key}|{replicate}"] = float(value)
        for replicate, value in zip(subset["replicate_id"], subset["difference_natural"]):
            new_values[f"{key}|{int(replicate)}"] = float(value)

    result = _numeric_diffs("paired_bootstrap_replicates", old_values, new_values)
    result["frozen_replicate_draws"] = int(len(wide))
    result["replay_replicate_rows"] = int(len(long))
    result["expected_replay_rows"] = 27 * int(len(wide))
    result["row_count_equal"] = len(long) == 27 * len(wide)
    if not result["row_count_equal"]:
        result["classification"] = CLASS_MATERIAL
        result["equivalent"] = False
    return result


def compare_oof_predictions(frozen_source: Path, replay_root: Path) -> dict[str, Any]:
    """Every out-of-fold score of every variant x model (section 7C)."""
    import pandas as pd

    frozen_source = Path(frozen_source)
    frames = []
    for path in sorted((frozen_source / "model" / "variants").glob("*/*/oof_predictions.parquet")):
        frames.append(pd.read_parquet(path))
    old = pd.concat(frames, ignore_index=True)
    new = pd.read_parquet(Path(replay_root) / "oof_predictions.parquet")

    old_keyed = old.set_index(["variant_id", "model_family", "cell_id"])
    new_keyed = new.set_index(["variant", "model", "cell_id"])
    scores = _numeric_diffs(
        "oof_y_score", old_keyed["y_score"].to_dict(), new_keyed["y_score"].to_dict(),
    )
    labels = _numeric_diffs(
        "oof_y_true", old_keyed["y_true"].astype(float).to_dict(),
        new_keyed["y_true"].astype(float).to_dict(),
    )
    folds = _numeric_diffs(
        "oof_fold_id", old_keyed["fold_id"].astype(float).to_dict(),
        new_keyed["fold_id"].astype(float).to_dict(),
    )
    parts = [scores, labels, folds]
    return {
        "comparison": "oof_predictions",
        "frozen_rows": int(len(old)), "replay_rows": int(len(new)),
        "row_count_equal": len(old) == len(new),
        "parts": parts,
        "classification": (
            CLASS_MATERIAL if not all(p["equivalent"] for p in parts) or len(old) != len(new)
            else max((p["classification"] for p in parts), key=lambda c: c == CLASS_FLOAT)
        ),
        "equivalent": all(p["equivalent"] for p in parts) and len(old) == len(new),
    }


def compare_folds(frozen_source: Path, replay_root: Path) -> dict[str, Any]:
    """Shared spatial folds and block assignment (section 7C)."""
    import pandas as pd

    old = pd.read_parquet(
        Path(frozen_source) / "model" / "shared_folds" / "shared_spatial_folds.parquet"
    ).set_index("cell_id")
    new = pd.read_parquet(Path(replay_root) / "fold_mapping.parquet").set_index("cell_id")
    parts = [
        _numeric_diffs("fold_id", old["fold_id"].astype(float).to_dict(),
                       new["fold_id"].astype(float).to_dict()),
        _numeric_diffs("spatial_block_id", old["spatial_block_id"].astype(float).to_dict(),
                       new["spatial_block_id"].astype(float).to_dict()),
    ]
    return {
        "comparison": "shared_spatial_folds",
        "frozen_rows": int(len(old)), "replay_rows": int(len(new)),
        "parts": parts,
        "classification": CLASS_SCIENTIFIC if all(p["equivalent"] for p in parts) else CLASS_MATERIAL,
        "equivalent": all(p["equivalent"] for p in parts),
    }


def compare_descriptive_report(frozen_source: Path, replay_root: Path) -> dict[str, Any]:
    """The narrative report, whose ONE normalized sentence is declared here."""
    old = (
        Path(frozen_source) / "compare" / "report" / "window_closure_comparison.md"
    ).read_text(encoding="utf-8")
    new = (Path(replay_root) / "report.md").read_text(encoding="utf-8")
    normalized = old.replace(
        "This is an observational predictive analysis. It does not establish a causal mechanism.",
        "These results are descriptive and do not establish an underlying mechanism.",
    )
    identical = normalized == new
    return {
        "comparison": "descriptive_report",
        "byte_identical_before_normalization": old == new,
        "byte_identical_after_declared_normalization": identical,
        "declared_normalization": (
            "production.normalize_production_regional_outputs replaces one causal-"
            "wording sentence with the regional package's wording. Applied "
            "identically to all four production AOIs."
        ),
        "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
        "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
        "classification": CLASS_SCIENTIFIC if identical else CLASS_MATERIAL,
        "equivalent": identical,
    }


def classify_replay_only_artifacts(replay_root: Path) -> dict[str, Any]:
    """Classify what the regional wrapper adds on top of the frozen tree (7F)."""
    replay_root = Path(replay_root)
    categories = {
        "config.json": "regional_wrapper_metadata",
        "input_hashes.json": "provenance",
        "repository_inventory.json": "provenance",
        "window_dates.csv": "regional_wrapper_metadata",
        "export_plan.csv": "regional_wrapper_metadata",
        "cohort_inventory.csv": "regional_projection_of_frozen_science",
        "fold_mapping.parquet": "regional_projection_of_frozen_science",
        "variant_artifact_index.csv": "provenance",
        "metrics.csv": "regional_projection_of_frozen_science",
        "oof_predictions.parquet": "regional_projection_of_frozen_science",
        "bootstrap_replicates.parquet": "regional_projection_of_frozen_science",
        "bootstrap_summary.csv": "regional_projection_of_frozen_science",
        "regional_summary.csv": "regional_wrapper_metadata",
        "summary.json": "regional_wrapper_metadata",
        "report.md": "regional_projection_of_frozen_science",
        "manifest.json": "provenance",
        "manifest.sha256": "provenance",
        "validator_results.json": "validator",
        "validator_summary.json": "validator",
        "equivalence_report.json": "equivalence_evidence",
        "equivalence_report.md": "equivalence_evidence",
    }
    found: dict[str, str] = {}
    unclassified: list[str] = []
    for path in sorted(replay_root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(replay_root))
        if relative.startswith("_production/"):
            continue
        if relative.startswith("stages/"):
            found[relative] = "orchestration_state"
            continue
        if relative in categories:
            found[relative] = categories[relative]
        else:
            unclassified.append(relative)
    counts: dict[str, int] = {}
    for value in found.values():
        counts[value] = counts.get(value, 0) + 1
    return {
        "comparison": "replay_only_artifacts",
        "classified": found, "counts": counts,
        "unclassified": unclassified,
        "new_scientific_output": [],
        "note": (
            "No file in this list is a NEW scientific result: each is either "
            "wrapper metadata, provenance, validator output, or a re-projection "
            "of the frozen per-AOI scientific tables into the regional schema."
        ),
        "equivalent": not unclassified,
    }


def build_equivalence_report(
    *, frozen_source: Path, replay_root: Path, preflight: Mapping[str, Any],
    validator: Mapping[str, Any], mutation_guard: Mapping[str, Any],
    replay_result: Mapping[str, Any],
) -> dict[str, Any]:
    """The section-11 provenance artefact behind the migration decision."""
    frozen_source, replay_root = Path(frozen_source), Path(replay_root)
    sections = {
        "contract": {
            "comparison": "scientific_contract",
            "scientific_fields_compared": preflight["scientific_field_count"],
            "mismatches": preflight["mismatches"],
            "non_scientific_differences": [
                {"field": c["field"], "old": c["old"], "new": c["new"]}
                for c in preflight["non_scientific_differences"]
            ],
            "classification": CLASS_SCIENTIFIC if not preflight["mismatches"] else CLASS_MATERIAL,
            "equivalent": not preflight["mismatches"],
        },
        "frozen_reuse": compare_frozen_reuse(frozen_source, replay_root),
        "population": compare_population(frozen_source, replay_root),
        "fit_accounting": compare_fit_accounting(replay_root),
        "point_metrics": compare_point_metrics(frozen_source, replay_root),
        "bootstrap_summary": compare_bootstrap_summary(frozen_source, replay_root),
        "bootstrap_replicates": compare_bootstrap_replicates(frozen_source, replay_root),
        "oof_predictions": compare_oof_predictions(frozen_source, replay_root),
        "folds": compare_folds(frozen_source, replay_root),
        "report": compare_descriptive_report(frozen_source, replay_root),
        "artifact_completeness": classify_replay_only_artifacts(replay_root),
    }
    class_counts: dict[str, int] = {}
    for section in sections.values():
        label = section.get("classification")
        if label:
            class_counts[label] = class_counts.get(label, 0) + 1
    numeric_sections = [
        "point_metrics", "bootstrap_summary", "bootstrap_replicates",
    ]
    max_abs = max(
        (float(sections[name].get("max_abs_diff", 0.0)) for name in numeric_sections),
        default=0.0,
    )
    validator_pass = str(validator.get("overall_status")) == "PASS"
    sections_equivalent = all(bool(s.get("equivalent")) for s in sections.values())
    sources_unchanged = bool(mutation_guard.get("all_unchanged"))
    verdict = "PASS" if (
        preflight["status"] == "PASS" and sections_equivalent
        and validator_pass and sources_unchanged
    ) else "FAIL"
    return {
        "schema_version": "window_closure_reference_replay_equivalence.v1",
        "replay_mode": REPLAY_MODE_ID,
        "aoi": REFERENCE_AOI,
        "aoi_role": aoi_role(REFERENCE_AOI),
        "old_source_path": str(frozen_source),
        "new_replay_path": str(replay_root),
        "old_analysis_id": preflight["frozen_analysis_id"],
        "new_analysis_id": replay_result.get("analysis_id"),
        "old_production_analysis_id_recomputed": preflight["derived_production_analysis_id"],
        "canonical_step8a_sha256": CANONICAL_STEP8A_SHA256[REFERENCE_AOI],
        "sections": sections,
        "file_classification_counts": class_counts,
        "max_numeric_abs_diff": max_abs,
        "float_tolerance_used": FLOAT_SERIALIZATION_TOLERANCE,
        "validator": {
            "overall_status": validator.get("overall_status"),
            "required_fail": validator.get("required_fail"),
            "required_skip": validator.get("required_skip"),
            "check_count": len(validator.get("checks", [])),
        },
        "source_mutation_guard": mutation_guard,
        "replay_stages_reused_frozen": list(REPLAYED_STAGES),
        "replay_stages_executed": list(EXECUTED_STAGES),
        "gee_queries_run": False,
        "gee_exports_run": False,
        "remote_downloads": 0,
        "equivalence_verdict": verdict,
        "migration_safe": verdict == "PASS",
        "migration_performed": False,
        "note": (
            "This turn produces equivalence evidence only. Nothing under the "
            "frozen Manavgat namespace or the four production regional "
            "namespaces was written, moved, renamed, or deleted."
        ),
    }


def render_equivalence_report(report: Mapping[str, Any]) -> str:
    """Human-readable companion to `equivalence_report.json`."""
    sections = report["sections"]
    lines = [
        "# Manavgat reference replay -- old vs new scientific equivalence",
        "",
        f"**Verdict: {report['equivalence_verdict']}**",
        "",
        f"- AOI: `{report['aoi']}` (role: `{report['aoi_role']}`)",
        f"- Old source: `{report['old_source_path']}`",
        f"- New replay: `{report['new_replay_path']}`",
        f"- Old (frozen) analysis id: `{report['old_analysis_id']}`",
        f"- New regional analysis id: `{report['new_analysis_id']}`",
        f"- Canonical Step8A SHA-256: `{report['canonical_step8a_sha256']}`",
        f"- Earth Engine queries / exports / remote downloads: "
        f"{report['gee_queries_run']} / {report['gee_exports_run']} / {report['remote_downloads']}",
        "",
        "## Section results",
        "",
        "| Section | Classification | Equivalent |",
        "| --- | --- | --- |",
    ]
    for name, section in sections.items():
        lines.append(
            f"| {name} | {section.get('classification', '-')} | "
            f"{'yes' if section.get('equivalent') else 'NO'} |"
        )
    lines += [
        "",
        "## Headline metrics",
        "",
        "| Variant | Model | Metric | Old | New | abs diff |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in sections["point_metrics"]["side_by_side"]:
        lines.append(
            f"| {row['variant']} | {row['model']} | {row['metric']} | "
            f"{row['old']!r} | {row['new']!r} | {row['abs_diff']!r} |"
        )
    population = sections["population"]
    lines += [
        "",
        "## Population and accounting",
        "",
        "| Field | Old | New | Equal |",
        "| --- | --- | --- | --- |",
    ]
    for row in population["rows"]:
        lines.append(
            f"| {row['field']} | {row['old']} | {row['new']} | "
            f"{'yes' if row['equal'] else 'NO'} |"
        )
    reuse = sections["frozen_reuse"]
    validator = report["validator"]
    guard = report["source_mutation_guard"]
    lines += [
        "",
        "## Frozen artefact reuse",
        "",
        f"- Frozen non-quarantine files: {reuse['frozen_file_count']}",
        f"- Byte-identical in replay: {reuse['byte_identical']}",
        f"- Missing in replay: {len(reuse['missing_in_replay'])}",
        f"- Unexpected in replay: {len(reuse['unexpected_in_replay'])}",
        f"- Hash mismatches: {len(reuse['hash_mismatches'])}",
        f"- Excluded `_quarantine` files: {reuse['excluded_quarantine_file_count']}",
        "",
        "## Numeric differences",
        "",
        f"- Max absolute difference across all compared scientific values: "
        f"`{report['max_numeric_abs_diff']}`",
        f"- Float serialization tolerance: `{report['float_tolerance_used']}`",
        "",
        "## Regional validator",
        "",
        f"- Overall status: **{validator['overall_status']}**",
        f"- Checks evaluated: {validator['check_count']}",
        f"- Required failures: {validator['required_fail']}; "
        f"required skips: {validator['required_skip']}",
        "",
        "## Source mutation guard",
        "",
        f"- Trees checked: {len(guard['trees_checked'])}",
        f"- All unchanged: {'yes' if guard['all_unchanged'] else 'NO'}",
        f"- Method: {guard['method']}",
        "",
        "## Scope of this artefact",
        "",
        report["note"],
        "",
    ]
    return "\n".join(lines)
