"""Shared, method-free helpers for the reproduction validation namespace.

Nothing here defines or alters a scientific rule. It only resolves the frozen
cohort from the repository's own configuration, hashes frozen inputs, and
records environment/provenance.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from core.paths import PROJECT_ROOT

DIAGNOSTICS_ROOT = PROJECT_ROOT / "outputs" / "diagnostics"

# The frozen five-AOI multi-AOI transfer synthesis namespace. This is the
# frozen artefact that fixes the canonical five-region analysis on disk.
FIVE_REGION_SYNTHESIS_DIR = (
    DIAGNOSTICS_ROOT
    / "multi_aoi_transfer_synthesis"
    / "bejis_2022__evia_2021_extended__manavgat_2021__montiferru_2021__mugla_2021"
)


class ReproductionValidationError(SystemExit):
    """Fail-fast error (same convention as the pipeline steps)."""


# =============================================================================
# Hashing / provenance
# =============================================================================
def sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def git_status_short() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=str(PROJECT_ROOT), text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def environment() -> dict:
    import platform
    import sys

    import numpy
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "python_full": sys.version,
        "scikit_learn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_to_root(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# =============================================================================
# Frozen cohort resolution -- three independent code/artefact-derived routes
# =============================================================================
def resolve_frozen_five_region_cohort() -> dict:
    """Resolve the frozen five-region cohort from the repository itself.

    Three independent routes must agree, otherwise this fails closed:

      1. the experiment registry (`core.regions`): enabled + canonical
         experiments, minus the roles that `src.burned_pattern_audit`
         declares non-cohort (`negative_control`, `temporal_transfer_wildfire`);
      2. the frozen cohort constant `DEFAULT_EXPERIMENTS` in
         `src.era5_land_regional_diagnostic`;
      3. `aois_canonical_order` in the frozen five-AOI multi-AOI transfer
         synthesis manifest on disk.

    Returns a dict with the agreed cohort and the per-route evidence.
    """
    from core.regions import list_canonical_enabled_experiments
    from src.burned_pattern_audit import NON_COHORT_ROLES
    from src.era5_land_regional_diagnostic import DEFAULT_EXPERIMENTS

    registry_selection = list_canonical_enabled_experiments()
    registry_cohort = sorted(
        experiment_id
        for experiment_id, record in registry_selection.items()
        if record.get("role") not in NON_COHORT_ROLES
    )

    constant_cohort = sorted(DEFAULT_EXPERIMENTS)

    manifest_path = FIVE_REGION_SYNTHESIS_DIR / "multi_aoi_manifest.json"
    if not manifest_path.exists():
        raise ReproductionValidationError(
            f"Frozen five-AOI synthesis manifest not found: {manifest_path}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_cohort = sorted(manifest["aois_canonical_order"])

    routes = {
        "experiment_registry_canonical_minus_non_cohort_roles": registry_cohort,
        "era5_land_regional_diagnostic.DEFAULT_EXPERIMENTS": constant_cohort,
        "frozen_multi_aoi_synthesis_manifest.aois_canonical_order": manifest_cohort,
    }
    if not (registry_cohort == constant_cohort == manifest_cohort):
        raise ReproductionValidationError(
            "Frozen five-region cohort is AMBIGUOUS -- the three resolution "
            f"routes disagree: {json.dumps(routes, indent=2)}"
        )
    if len(registry_cohort) != 5:
        raise ReproductionValidationError(
            f"Frozen cohort must contain exactly 5 experiments, got {registry_cohort}."
        )
    for excluded in ("mugla_2022_event_relative", "mugla_2022"):
        if excluded in registry_cohort:
            raise ReproductionValidationError(
                f"'{excluded}' must never enter the frozen five-region cohort."
            )

    return {
        "cohort": registry_cohort,
        "resolution_routes": routes,
        "routes_agree": True,
        "manifest_path": relative_to_root(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "non_cohort_roles": list(NON_COHORT_ROLES),
        "deliberately_excluded": {
            experiment_id: dict(record).get("role")
            for experiment_id, record in registry_selection.items()
            if experiment_id not in registry_cohort
        },
    }


def step8a_dataset_record(experiment_id: str) -> dict:
    """Path + SHA-256 of an experiment's frozen Step8A parquet, plus every
    SHA-256 recorded for it in the frozen Step9A/Step10 provenance JSONs."""
    from src.step9a_audit_cross_region_inputs import resolve_step8a_dataset_path

    path = resolve_step8a_dataset_path(experiment_id)
    if not Path(path).exists():
        raise ReproductionValidationError(
            f"Frozen Step8A dataset missing for '{experiment_id}': {path}."
        )
    observed = sha256_file(path)
    recorded = recorded_step8a_hashes(experiment_id)
    return {
        "experiment_id": experiment_id,
        "path": relative_to_root(path),
        "observed_sha256": observed,
        "recorded_sha256_values": sorted(recorded),
        "recorded_reference_count": sum(len(v) for v in recorded.values()),
        "hash_agrees_with_provenance": (
            len(recorded) > 0 and set(recorded) == {observed}
        ),
    }


def recorded_step8a_hashes(experiment_id: str) -> dict[str, list[str]]:
    """Every Step8A SHA-256 recorded for `experiment_id` across the frozen
    Step9A / Step10 provenance JSONs, mapped to the files recording it."""
    found: dict[str, list[str]] = {}
    patterns = (
        "outputs/cross_region/*/step9a/cross_region_input_audit.json",
        "outputs/cross_region/*/step10/step10_input_audit.json",
        "outputs/cross_region/*/step10/step10_preregistration.json",
    )
    for pattern in patterns:
        for provenance_path in sorted(PROJECT_ROOT.glob(pattern)):
            try:
                payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for digest in _collect_dataset_hashes(payload, experiment_id):
                found.setdefault(digest, []).append(relative_to_root(provenance_path))
    return found


def _collect_dataset_hashes(node, experiment_id: str) -> list[str]:
    digests: list[str] = []
    if isinstance(node, dict):
        if node.get("experiment_id") == experiment_id:
            digest = node.get("dataset_sha256") or node.get("step8a_dataset_sha256")
            if digest:
                digests.append(digest)
        for value in node.values():
            digests.extend(_collect_dataset_hashes(value, experiment_id))
    elif isinstance(node, list):
        for value in node:
            digests.extend(_collect_dataset_hashes(value, experiment_id))
    return digests


def write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path
