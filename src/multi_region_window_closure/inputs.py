"""Frozen-input inventory and canonical hash verification.

The four canonical Step8A SHA-256 anchors are RECOMPUTED from the artefacts on
disk. The literals below are the expected values; they are never accepted as
evidence on their own -- `verify_canonical_hashes` always digests the real
file and compares.

`.gitignore` ignores `outputs/`, so none of these artefacts is tracked by git.
Their integrity rests entirely on these digests, which is why this module is
fail-closed.

Design reference: docs/multi_region_window_closure_design/README.md section 6
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS,
    MultiRegionWindowClosureError,
    REFERENCE_AOI,
    SYNTHESIS_AOIS,
)

#: Expected SHA-256 of `step8a/step8a_500m_modeling_dataset.parquet` per AOI.
#: Corroborated independently: the Manavgat value is recorded as
#: `frozen_input_sha256.canonical_step8a` in the already-frozen
#: outputs/diagnostics/window_closure_sensitivity/manavgat_2021/config/
#: frozen_input_inventory.json, whose inventory entry names that exact file.
CANONICAL_STEP8A_SHA256: dict[str, str] = {
    "manavgat_2021": "054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439",
    "bejis_2022": "3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393",
    "mugla_2021": "c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e",
    "evia_2021_extended": "bdce859cf482f575d0f273174b157f47efd61779953fdd23d9486c5face5e553",
    "montiferru_2021": "ffb008f977445076835ae35a70776b0b813d0a36655d2c656fd21e6fedd4ac50",
}

CANONICAL_STEP8A_RELPATH = ("step8a", "step8a_500m_modeling_dataset.parquet")

#: The frozen input roles every actual AOI must supply, reusing the per-AOI
#: contract rather than inventing a second list.
REQUIRED_FROZEN_INPUT_ROLES: tuple[str, ...] = (
    "canonical_step8a", "dem_elevation", "dem_slope", "landcover_aligned",
    "label_raw_burndate", "label_burned_binary",
)


def _canonical_blocker(message: str) -> MultiRegionWindowClosureError:
    return MultiRegionWindowClosureError(
        f"BLOCKER: CANONICAL_STEP8A_INVENTORY_INVALID -- {message}"
    )


def resolve_canonical_step8a_record(
    payload: Mapping[str, Any], *, aoi: str,
) -> dict[str, str]:
    """Resolve and verify exactly one canonical record, failing closed.

    Supported inputs are the flat planning inventory returned by
    ``run_analysis(..., dry_run=True)`` and the complete persisted
    ``config/frozen_input_inventory.json`` document.  No recursive or
    substring search is performed.
    """
    if not isinstance(payload, Mapping):
        raise _canonical_blocker("inventory is not a mapping")
    candidates: list[tuple[str, Mapping[str, Any], str | None]] = []
    flat = payload.get("canonical_step8a")
    if flat is not None:
        if not isinstance(flat, Mapping):
            raise _canonical_blocker("flat canonical_step8a record is not a mapping")
        candidates.append(("planning_flat", flat, None))
    persisted_roles = payload.get("inventory")
    if persisted_roles is not None:
        if not isinstance(persisted_roles, Mapping):
            raise _canonical_blocker("persisted inventory field is not a mapping")
        persisted = persisted_roles.get("canonical_step8a")
        if persisted is not None:
            if not isinstance(persisted, Mapping):
                raise _canonical_blocker(
                    "persisted inventory.canonical_step8a record is not a mapping"
                )
            candidates.append(("production_persisted", persisted, payload.get("experiment_id")))
    if len(candidates) != 1:
        raise _canonical_blocker(
            f"expected exactly one supported canonical candidate, found {len(candidates)}"
        )

    source_schema, raw, document_aoi = candidates[0]
    if source_schema == "production_persisted":
        if payload.get("read_only") is not True:
            raise _canonical_blocker("persisted inventory read_only must be true")
        if raw.get("exists") is not True:
            raise _canonical_blocker("persisted canonical_step8a exists must be true")
    record_aoi = raw.get("aoi", document_aoi)
    if record_aoi is not None and str(record_aoi) != aoi:
        raise _canonical_blocker(
            f"AOI mismatch: requested {aoi!r}, inventory declares {record_aoi!r}"
        )
    role = raw.get("role", "canonical_step8a")
    if role != "canonical_step8a":
        raise _canonical_blocker(f"canonical role mismatch: {role!r}")
    raw_path = raw.get("path")
    raw_sha256 = raw.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise _canonical_blocker("canonical path is missing")
    if not isinstance(raw_sha256, str) or not raw_sha256.strip():
        raise _canonical_blocker("canonical SHA-256 is missing")
    if source_schema == "production_persisted":
        hashes = payload.get("frozen_input_sha256")
        if not isinstance(hashes, Mapping) or hashes.get("canonical_step8a") != raw_sha256:
            raise _canonical_blocker(
                "persisted nested/mirror canonical SHA-256 mismatch"
            )
    expected = CANONICAL_STEP8A_SHA256.get(aoi)
    if expected is None or raw_sha256 != expected:
        raise _canonical_blocker(
            f"central anchor mismatch for {aoi}: expected {expected!r}, got {raw_sha256!r}"
        )
    path = Path(raw_path)
    if not path.is_file():
        raise _canonical_blocker(f"canonical path is not a file: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != raw_sha256:
        raise _canonical_blocker(
            f"actual-byte hash drift: inventory {raw_sha256}, actual {actual}"
        )
    frozen_path = canonical_step8a_path(aoi)
    if path.resolve() != frozen_path.resolve():
        raise _canonical_blocker(
            f"canonical path mismatch: inventory {path.resolve()}, frozen {frozen_path.resolve()}"
        )
    return {
        "path": str(path), "sha256": raw_sha256,
        "source_schema": source_schema, "aoi": aoi,
    }


def canonical_step8a_path(aoi: str, experiments_root: Optional[Path] = None) -> Path:
    """Path of the AOI's canonical Step8A modelling dataset (read-only)."""
    from src.window_closure_sensitivity import canonical_experiment_root

    root = canonical_experiment_root(aoi, experiments_root)
    return Path(root).joinpath(*CANONICAL_STEP8A_RELPATH)


def verify_canonical_hashes(
    aois: Sequence[str] = SYNTHESIS_AOIS,
    experiments_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Recompute and compare every canonical Step8A digest. Fail-closed.

    Returns a per-AOI record even when everything matches, so the dry-run can
    publish the evidence rather than merely asserting success.
    """
    from src.step8_large_block_robustness import sha256_file

    records: dict[str, dict[str, Any]] = {}
    drifted: list[str] = []
    missing: list[str] = []
    for aoi in aois:
        expected = CANONICAL_STEP8A_SHA256.get(aoi)
        if expected is None:
            raise MultiRegionWindowClosureError(
                f"No canonical Step8A hash anchor is registered for '{aoi}'."
            )
        path = canonical_step8a_path(aoi, experiments_root)
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        match = bool(exists and actual == expected)
        records[aoi] = {
            "aoi": aoi,
            "path": str(path),
            "exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": match,
        }
        if not exists:
            missing.append(aoi)
        elif not match:
            drifted.append(aoi)
    return {
        "records": records,
        "all_match": not drifted and not missing,
        "drifted_aois": sorted(drifted),
        "missing_aois": sorted(missing),
    }


def assert_canonical_hashes(
    aois: Sequence[str] = SYNTHESIS_AOIS,
    experiments_root: Optional[Path] = None,
) -> dict[str, Any]:
    """`verify_canonical_hashes`, raising on drift or absence."""
    result = verify_canonical_hashes(aois, experiments_root)
    if result["missing_aois"]:
        raise MultiRegionWindowClosureError(
            "BLOCKER: CANONICAL_HASH_DRIFT -- canonical Step8A dataset missing "
            f"for {result['missing_aois']}."
        )
    if result["drifted_aois"]:
        details = "; ".join(
            f"{aoi}: expected {result['records'][aoi]['expected_sha256']}, "
            f"got {result['records'][aoi]['actual_sha256']}"
            for aoi in result["drifted_aois"]
        )
        raise MultiRegionWindowClosureError(
            f"BLOCKER: CANONICAL_HASH_DRIFT -- {details}."
        )
    return result


def frozen_input_inventory(
    aois: Sequence[str] = ACTUAL_AOIS,
    experiments_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Per-AOI frozen-input inventory, delegating to the per-AOI helper.

    Reuses `window_closure_sensitivity.frozen_input_inventory` so the six
    required roles and their resolution rules are identical to the reference
    analysis.
    """
    from core.experiment_context import build_experiment_context
    from src.window_closure_sensitivity import frozen_input_inventory as per_aoi_inventory

    out: dict[str, Any] = {}
    for aoi in aois:
        context = build_experiment_context(aoi)
        out[aoi] = per_aoi_inventory(aoi, experiments_root, context)
    return out


def missing_required_inputs(inventory: dict[str, Any]) -> dict[str, list[str]]:
    """`{aoi: [missing role, ...]}` over the six required frozen input roles."""
    out: dict[str, list[str]] = {}
    for aoi, roles in sorted(inventory.items()):
        absent = [
            role for role in REQUIRED_FROZEN_INPUT_ROLES
            if not (roles.get(role) or {}).get("exists")
        ]
        if absent:
            out[aoi] = absent
    return out


def input_hashes_document(
    inventory: dict[str, Any], reference_aoi: str = REFERENCE_AOI,
) -> dict[str, Any]:
    """The `input_hashes.json` payload: role -> path/hash/existence per AOI."""
    document: dict[str, Any] = {"schema_role": "input_hashes", "aois": {}}
    for aoi, roles in sorted(inventory.items()):
        entry: dict[str, Any] = {}
        for role in sorted(roles):
            record = roles.get(role) or {}
            if not isinstance(record, dict) or "sha256" not in record:
                continue
            entry[role] = {
                "path": str(record.get("path", "")),
                "sha256": record.get("sha256"),
                "exists": bool(record.get("exists")),
                "required": role in REQUIRED_FROZEN_INPUT_ROLES,
                "resolved_from": record.get("resolved_from"),
            }
        document["aois"][aoi] = entry
    document["reference_aoi"] = reference_aoi
    document["required_roles"] = list(REQUIRED_FROZEN_INPUT_ROLES)
    return document


def manavgat_reference_artifacts(output_root: Optional[Path] = None) -> dict[str, Any]:
    """Read-only locators for the frozen Manavgat window-closure results.

    Only the paths and digests are produced here. Nothing under the Manavgat
    namespace is ever opened for writing, and the namespace is never re-run.
    """
    from src.step8_large_block_robustness import sha256_file
    from src.window_closure_sensitivity import experiment_root

    root = Path(experiment_root(REFERENCE_AOI, output_root))
    wanted = {
        "preregistration": root / "config" / "preregistration.json",
        "frozen_input_inventory": root / "config" / "frozen_input_inventory.json",
        "point_metrics": root / "model" / "metrics" / "point_metrics.csv",
        "bootstrap_summary": root / "model" / "bootstrap" / "paired_bootstrap_summary.csv",
        "common_cohort_metadata": (
            root / "model" / "common_cohort" / "common_cohort_metadata.json"
        ),
        "shared_folds_metadata": (
            root / "model" / "shared_folds" / "shared_spatial_folds_metadata.json"
        ),
    }
    records: dict[str, Any] = {}
    for name, path in wanted.items():
        exists = path.is_file()
        records[name] = {
            "path": str(path),
            "exists": exists,
            "sha256": sha256_file(path) if exists else None,
            "access": "read_only",
        }
    return {
        "aoi": REFERENCE_AOI,
        "namespace_root": str(root),
        "artifacts": records,
        "recomputed": False,
        "overwritten": False,
    }
