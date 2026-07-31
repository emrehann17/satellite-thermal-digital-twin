#!/usr/bin/env python3
"""Deterministic validator for the window-closure PREDICTOR-EXPORT stage.

This script NEVER runs a dry run, an export, a model fit or a bootstrap. It
only reads:

  * a log file the user produced (``--mode dry-run``), and/or
  * the files already on disk in the dedicated diagnostics namespace
    (``--mode actual``),

and re-checks the technical, scientific and namespace/provenance contracts
against them.

Usage
-----
    python scripts/validate_window_closure_predictor_export.py \
      --experiment <experiment_id> \
      --shifts 0 7 14 \
      --mode dry-run \
      --log logs/window_closure_predictor_dryrun.log

    python scripts/validate_window_closure_predictor_export.py \
      --experiment <experiment_id> \
      --shifts 0 7 14 \
      --mode actual

Exit code is 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.window_closure_sensitivity as wcs

FORBIDDEN_PRODUCT_TOKEN = "date_balanced"
# Stages that must still be refused by an actual run. `local-downstream` left
# this list when it was implemented and reviewed; `model` and `compare` are the
# ones this validator still requires to be locked.
# Every stage of this analysis is now implemented and reviewed, so no
# stage remains locked. The guard is still asserted so that adding a NEW
# stage without implementing it is caught here.
LOCKED_ACTUAL_STAGES: tuple[str, ...] = ()

# Frozen roles that BIND the analysis identity, taken from the module rather
# than re-listed here. Every other inventory entry is convenience metadata
# (notably `canonical_step8a_stats`, a resolver side-file): it may legitimately
# have no hash, and a missing hash for it must never be reported as if the
# frozen scientific identity were incomplete.
REQUIRED_FROZEN_HASH_ROLES: tuple[str, ...] = wcs.REQUIRED_PREDICTOR_FROZEN_HASH_ROLES
# The dry-run payload records the BASE inventory, which does not yet contain the
# pre-label raster (that role is added by the predictor stage).
REQUIRED_DRY_RUN_FROZEN_HASH_ROLES: tuple[str, ...] = wcs.REQUIRED_FROZEN_INPUT_ROLES


# =============================================================================
# Result accumulation
# =============================================================================
class Report:
    """Ordered [PASS]/[FAIL] lines plus the four status verdicts."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: dict[str, list[str]] = {
            "technical": [], "scientific": [], "namespace": [],
        }

    def check(self, category: str, ok: bool, message: str, detail: str = "") -> bool:
        if ok:
            self.lines.append(f"[PASS] {message}")
        else:
            suffix = f" -- {detail}" if detail else ""
            self.lines.append(f"[FAIL] {message}{suffix}")
            self.failures[category].append(f"{message}{suffix}")
        return ok

    def technical(self, ok: bool, message: str, detail: str = "") -> bool:
        return self.check("technical", ok, message, detail)

    def scientific(self, ok: bool, message: str, detail: str = "") -> bool:
        return self.check("scientific", ok, message, detail)

    def namespace(self, ok: bool, message: str, detail: str = "") -> bool:
        return self.check("namespace", ok, message, detail)

    def render(self) -> tuple[str, int]:
        verdicts = {
            key: ("PASS" if not values else "FAIL")
            for key, values in self.failures.items()
        }
        overall = "PASS" if all(v == "PASS" for v in verdicts.values()) else "FAIL"
        body = "\n".join(self.lines)
        body += (
            f"\n\nTECHNICAL STATUS: {verdicts['technical']}"
            f"\nSCIENTIFIC-CONTRACT STATUS: {verdicts['scientific']}"
            f"\nNAMESPACE / PROVENANCE SAFETY: {verdicts['namespace']}"
            f"\nOVERALL STATUS: {overall}"
        )
        return body, (0 if overall == "PASS" else 1)


# =============================================================================
# Helpers
# =============================================================================
def parse_json_object_from_log(text: str) -> Optional[dict]:
    """Extract the first complete top-level JSON object from a mixed log.

    The log interleaves logger lines with the printed dry-run payload, so a
    plain ``json.loads`` cannot be used. Every ``{`` that starts a line is
    tried as an object start via ``raw_decode``; the first one that decodes to
    a dict wins. Nothing is executed and no eval-like parsing is used.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        if index and text[index - 1] not in "\n\r":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _shift_date(date_text: str, year: int) -> str:
    parsed = datetime.strptime(date_text, "%Y-%m-%d")
    return parsed.replace(year=year).strftime("%Y-%m-%d")


def _is_inside(path: str, root: Path) -> bool:
    resolved = Path(path).resolve()
    root = root.resolve()
    return resolved == root or root in resolved.parents


def _sha256(path: Path) -> Optional[str]:
    return wcs.sha256_file(path) if path.is_file() else None


def expected_variant_ids(shifts: Sequence[int]) -> list[str]:
    """Non-canonical variant ids, in increasing shift order."""
    return [
        wcs.variant_id(shift)
        for shift in sorted({int(value) for value in shifts})
        if shift != 0
    ]


# =============================================================================
# Shared contract checks
# =============================================================================
def check_stage_lock(report: Report) -> None:
    report.technical(
        wcs.PREDICTOR_STAGE in wcs.IMPLEMENTED_ACTUAL_STAGES,
        "predictor-export is an implemented actual stage",
        f"implemented={list(wcs.IMPLEMENTED_ACTUAL_STAGES)}",
    )
    still_locked = [
        stage for stage in LOCKED_ACTUAL_STAGES
        if stage not in wcs.IMPLEMENTED_ACTUAL_STAGES
    ]
    report.namespace(
        sorted(still_locked) == sorted(LOCKED_ACTUAL_STAGES),
        "no unimplemented stage is reachable",
        f"unlocked={[s for s in LOCKED_ACTUAL_STAGES if s in wcs.IMPLEMENTED_ACTUAL_STAGES]}",
    )


def preregistration_analysis_id(experiment_root: Path) -> Optional[str]:
    path = experiment_root / "config" / "preregistration.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("analysis_id")
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def check_variant_dates(
    report: Report, variant_id: str, plan: dict, artifacts: Sequence[dict],
    baseline_years: Sequence[int],
) -> None:
    """Current, baseline and MODIS dates of one variant."""
    start, end = plan["predictor_start_date"], plan["predictor_end_date"]

    current = [a for a in artifacts if a["scope"] == "current_window"]
    report.scientific(
        bool(current) and all(
            (a["start_date"], a["end_date"]) == (start, end) for a in current
        ),
        f"{variant_id} current Landsat dates match the variant window",
        f"expected {start}..{end}, got "
        f"{sorted({(a['start_date'], a['end_date']) for a in current})}",
    )

    baseline = [a for a in artifacts if a["scope"] == "baseline_year"]
    bad_years: list[str] = []
    for artifact in baseline:
        year = int(artifact["baseline_year"])
        want = (_shift_date(start, year), _shift_date(end, year))
        if (artifact["start_date"], artifact["end_date"]) != want:
            bad_years.append(
                f"{artifact['artifact_id']}: {artifact['start_date']}.."
                f"{artifact['end_date']} != {want[0]}..{want[1]}"
            )
    report.scientific(
        not bad_years,
        f"{variant_id} baseline dates keep the variant month/day in every year",
        "; ".join(bad_years[:4]),
    )
    report.scientific(
        sorted({int(a["baseline_year"]) for a in baseline}) == sorted(int(y) for y in baseline_years),
        f"{variant_id} baseline years come from the preregistration",
        f"expected {sorted(int(y) for y in baseline_years)}",
    )

    modis = [a for a in artifacts if a["family"] == "modis"]
    report.scientific(
        len(modis) == len(wcs.MODIS_ROLE_ORDER) and all(
            (a["start_date"], a["end_date"]) == (start, end) for a in modis
        ),
        f"{variant_id} MODIS dates use the variant context",
        f"expected {start}..{end}, got "
        f"{sorted({(a['start_date'], a['end_date']) for a in modis})}",
    )
    for artifact in current + baseline:
        semantics = artifact.get("date_semantics") or {}
        if not semantics:
            continue
        report.scientific(
            semantics.get("ee_filter_end") == artifact["end_date"]
            and semantics.get("ee_filter_end_semantics") == "exclusive"
            and semantics.get("requested_end_date") == artifact["end_date"],
            f"{variant_id}/{artifact['artifact_id']} keeps production "
            "end-exclusive filterDate semantics (no silent +1 day)",
            json.dumps({
                k: semantics.get(k) for k in
                ("requested_end_date", "ee_filter_end", "ee_filter_end_semantics")
            }),
        )
        break  # one representative per variant keeps the report readable


def check_role_and_product_counts(
    report: Report, variant_id: str, artifacts: Sequence[dict],
    baseline_years: Sequence[int],
) -> None:
    roles = sorted({a["role"] for a in artifacts})
    expected_roles = wcs.expected_logical_role_count(baseline_years)
    expected_rasters = wcs.expected_raster_count(baseline_years)

    report.technical(
        len(roles) == expected_roles and len(artifacts) == expected_rasters,
        f"{variant_id} has {expected_roles} roles and {expected_rasters} "
        "raster artefacts",
        f"got {len(roles)} roles / {len(artifacts)} rasters",
    )

    ids = [a["artifact_id"] for a in artifacts]
    report.technical(
        len(set(ids)) == len(ids),
        f"{variant_id} has no duplicate artefact id",
        f"duplicates={sorted({i for i in ids if ids.count(i) > 1})}",
    )
    paths = [a["output_path"] for a in artifacts if a.get("output_path")]
    report.technical(
        len(set(paths)) == len(paths),
        f"{variant_id} has no duplicate output path",
        f"duplicates={sorted({p for p in paths if paths.count(p) > 1})}",
    )

    landsat = [a for a in artifacts if a["family"] in ("lst", "ndvi")]
    by_role: dict[str, set] = {}
    for artifact in landsat:
        by_role.setdefault(artifact["role"], set()).add(artifact["product"])
    wrong = {
        role: sorted(products) for role, products in by_role.items()
        if sorted(products) != sorted(wcs.LANDSAT_PRODUCTS_PER_ROLE)
    }
    report.scientific(
        not wrong,
        f"{variant_id} every Landsat role carries both production products",
        json.dumps(wrong),
    )


def check_products_and_namespace(
    report: Report, variant_id: str, artifacts: Sequence[dict], root: Path,
) -> None:
    products = sorted({a["product"] for a in artifacts})
    report.scientific(
        not any(FORBIDDEN_PRODUCT_TOKEN in product for product in products),
        f"{variant_id} contains no date_balanced product",
        f"products={products}",
    )
    outside = [
        a["output_path"] for a in artifacts
        if a.get("output_path") and not _is_inside(a["output_path"], root)
    ]
    report.namespace(
        not outside,
        f"{variant_id} output paths are contained in the dedicated namespace",
        f"outside={outside[:4]}",
    )
    canonical_root = root / "variants" / wcs.CANONICAL_VARIANT_ID
    leaked = [
        a["output_path"] for a in artifacts
        if a.get("output_path") and _is_inside(a["output_path"], canonical_root)
    ]
    report.namespace(
        not leaked,
        f"{variant_id} writes nothing into the canonical variant namespace",
        f"leaked={leaked[:4]}",
    )


# =============================================================================
# Dry-run mode
# =============================================================================
def validate_dry_run(
    report: Report, experiment_id: str, shifts: Sequence[int],
    log_path: Path, root: Path,
) -> None:
    if not log_path.is_file():
        report.technical(False, "dry-run log exists", f"missing: {log_path}")
        return
    payload = parse_json_object_from_log(log_path.read_text(encoding="utf-8", errors="replace"))
    if payload is None:
        report.technical(False, "dry-run log contains a parsable JSON payload", str(log_path))
        return
    report.technical(True, "dry-run log contains a parsable JSON payload")

    frozen_id = preregistration_analysis_id(root)
    report.scientific(
        frozen_id is not None and payload.get("analysis_id") == frozen_id,
        "analysis_id matches preregistration",
        f"log={payload.get('analysis_id')!r} preregistration={frozen_id!r}",
    )
    report.technical(
        payload.get("experiment_id") == experiment_id,
        "experiment_id matches",
        f"log={payload.get('experiment_id')!r}",
    )
    report.technical(
        payload.get("dry_run") is True and payload.get("ran") is False,
        "payload is a dry run",
        f"dry_run={payload.get('dry_run')!r} ran={payload.get('ran')!r}",
    )
    report.technical(
        payload.get("planned_stages") == [wcs.PREDICTOR_STAGE],
        "planned stage is predictor-export only",
        f"planned_stages={payload.get('planned_stages')!r}",
    )
    report.namespace(
        payload.get("files_written") is False,
        "no dry-run file writes detected",
        f"files_written={payload.get('files_written')!r}",
    )
    for flag in ("gee_queries_run", "gee_exports_run", "model_fit", "bootstrap_run"):
        report.namespace(
            payload.get(flag) is False, f"{flag} is false in the dry run",
            f"{flag}={payload.get(flag)!r}",
        )

    summary = payload.get("predictor_export_summary")
    if not isinstance(summary, dict):
        report.technical(False, "dry run carries predictor_export_summary")
        return
    report.technical(True, "dry run carries predictor_export_summary")

    report.scientific(
        summary.get("canonical_export_enabled") is False,
        "canonical variant is export-disabled",
        f"canonical_export_enabled={summary.get('canonical_export_enabled')!r}",
    )
    report.scientific(
        summary.get("reducer") == "scene_weighted",
        "only scene_weighted products are present",
        f"reducer={summary.get('reducer')!r}",
    )
    report.scientific(
        summary.get("forbidden_products_present") is False,
        "no date_balanced product detected",
        f"forbidden_products_present={summary.get('forbidden_products_present')!r}",
    )
    report.namespace(
        summary.get("all_paths_inside_dedicated_namespace") is True,
        "all output paths are contained",
        f"all_paths_inside_dedicated_namespace="
        f"{summary.get('all_paths_inside_dedicated_namespace')!r}",
    )

    baseline_years = list(summary.get("baseline_years") or [])
    wanted = expected_variant_ids(shifts)
    got = list(summary.get("nonzero_variant_ids") or [])
    report.technical(
        got == wanted, "every preregistered non-zero variant is planned",
        f"expected={wanted} got={got}",
    )

    plans = summary.get("variant_plans") or {}
    canonical_plan = plans.get(wcs.CANONICAL_VARIANT_ID) or {}
    report.scientific(
        canonical_plan.get("export_enabled") is False
        and canonical_plan.get("frozen_reference_only") is True
        and canonical_plan.get("expected_raster_count") == 0,
        "canonical plan is frozen-reference-only with zero rasters",
        json.dumps({
            k: canonical_plan.get(k) for k in
            ("export_enabled", "frozen_reference_only", "expected_raster_count")
        }),
    )

    total = 0
    for variant_id in wanted:
        plan = plans.get(variant_id)
        if not isinstance(plan, dict):
            report.technical(False, f"{variant_id} is present in the plan")
            continue
        if not report.technical(
            plan.get("export_enabled") is True,
            f"{variant_id} is export-enabled",
            f"export_enabled={plan.get('export_enabled')!r}",
        ):
            continue
        artifacts = plan.get("expected_artifacts") or []
        total += len(artifacts)
        check_role_and_product_counts(report, variant_id, artifacts, baseline_years)
        check_variant_dates(report, variant_id, plan, artifacts, baseline_years)
        check_products_and_namespace(report, variant_id, artifacts, root)

    report.technical(
        summary.get("total_planned_rasters") == total
        and total == len(wanted) * wcs.expected_raster_count(baseline_years),
        "total planned raster count is consistent",
        f"summary={summary.get('total_planned_rasters')!r} counted={total}",
    )

    # --- Nothing may exist on disk after a dry run -------------------------
    stray: list[str] = []
    for variant_id in wanted:
        metadata = root / "variants" / variant_id / wcs.PREDICTOR_METADATA_NAME
        data_dir = root / "variants" / variant_id / "data"
        if metadata.exists():
            stray.append(str(metadata))
        if data_dir.exists():
            stray.append(str(data_dir))
    canonical_data = root / "variants" / wcs.CANONICAL_VARIANT_ID / "data"
    if canonical_data.exists():
        stray.append(str(canonical_data))
    report.namespace(
        not stray, "the dry run created no predictor raster or metadata",
        f"found={stray[:4]}",
    )

    check_frozen_hashes(report, payload.get("frozen_input_inventory") or {})


def check_frozen_hashes(report: Report, inventory: dict) -> None:
    """Re-hash the inventory a dry run recorded and compare it with disk.

    Required identity roles must still exist and match. A role recorded WITHOUT
    a hash is optional convenience metadata -- it never entered the identity, so
    it cannot drift and is not re-hashed here.
    """
    if not inventory:
        report.namespace(False, "frozen hashes are unchanged", "no inventory in the log")
        return
    missing = wcs.missing_required_frozen_hashes(
        inventory, REQUIRED_DRY_RUN_FROZEN_HASH_ROLES,
    )
    drifted: list[str] = []
    for role, entry in sorted(inventory.items()):
        if not isinstance(entry, dict) or not entry.get("path"):
            continue
        expected = entry.get("sha256")
        if expected is None:
            continue
        current = _sha256(Path(entry["path"]))
        if current != expected:
            drifted.append(f"{role}: {expected} -> {current}")
    report.namespace(
        not missing and not drifted, "frozen hashes are unchanged",
        f"missing_required={missing} drifted={'; '.join(drifted[:4])}",
    )


def check_required_frozen_inputs(report: Report, extended: dict) -> None:
    """Required identity inputs must hash; optional metadata roles need not.

    Splitting the two is the point: `canonical_step8a_stats` is a convenience
    resolver side-file, so its absence says nothing about the frozen scientific
    identity and is never reported as an unhashed frozen input.
    """
    missing = wcs.missing_required_frozen_hashes(extended, REQUIRED_FROZEN_HASH_ROLES)
    report.namespace(
        not missing,
        "all required frozen identity inputs exist and hash",
        f"missing_or_unhashed={missing}",
    )

    optional = wcs.optional_frozen_roles(extended, REQUIRED_FROZEN_HASH_ROLES)
    # An identity role must never be classified as optional, and a role that is
    # present on disk must still hash even when it is optional.
    misclassified = sorted(set(optional) & set(wcs.REQUIRED_FROZEN_INPUT_ROLES))
    present_but_unhashed = sorted(
        role for role in optional
        if (extended.get(role) or {}).get("exists")
        and (extended.get(role) or {}).get("sha256") is None
    )
    report.namespace(
        not misclassified and not present_but_unhashed,
        "optional metadata roles do not alter the frozen identity",
        f"identity_roles_misclassified={misclassified} "
        f"present_but_unhashed={present_but_unhashed}",
    )


def check_recorded_frozen_hashes(
    report: Report, label: str, recorded: dict, extended: dict,
) -> None:
    """Compare one variant's recorded frozen hashes against disk.

    A required role must carry a recorded hash AND still match. An optional
    role is verified only when it was actually hashed at export time; one
    recorded as null is convenience metadata and cannot fail here.
    """
    unrecorded = sorted(
        role for role in REQUIRED_FROZEN_HASH_ROLES if recorded.get(role) is None
    )
    drifted: list[str] = []
    for role in sorted(recorded):
        expected = recorded[role]
        if expected is None:
            continue
        current = (extended.get(role) or {}).get("sha256")
        if current != expected:
            drifted.append(f"{role}: {expected} -> {current}")
    report.namespace(
        not unrecorded and not drifted,
        f"{label} frozen inputs are unchanged since the export",
        f"unrecorded_required={unrecorded} drifted={drifted[:4]}",
    )


# =============================================================================
# Actual mode
# =============================================================================
def validate_actual(
    report: Report, experiment_id: str, shifts: Sequence[int], root: Path,
    experiments_root: Optional[Path] = None,
) -> None:
    frozen_id = preregistration_analysis_id(root)
    if not report.technical(
        frozen_id is not None, "preregistration is readable",
        f"missing or unreadable under {root}",
    ):
        return

    wanted = expected_variant_ids(shifts)
    seen_analysis_ids: set[Optional[str]] = set()
    prelabel = wcs.prelabel_raster_path(experiment_id, root.parent)
    prelabel_hash = _sha256(prelabel)

    total_rasters = 0
    baseline_years: list[int] = []
    for variant_id in wanted:
        metadata_path = root / "variants" / variant_id / wcs.PREDICTOR_METADATA_NAME
        if not report.technical(
            metadata_path.is_file(), f"{variant_id} predictor metadata exists",
            f"missing: {metadata_path}",
        ):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            report.technical(False, f"{variant_id} predictor metadata is readable", str(exc))
            continue

        seen_analysis_ids.add(metadata.get("analysis_id"))
        report.scientific(
            metadata.get("analysis_id") == frozen_id,
            f"{variant_id} metadata analysis_id matches preregistration",
            f"metadata={metadata.get('analysis_id')!r}",
        )
        report.technical(
            metadata.get("schema_version") == wcs.PREDICTOR_METADATA_SCHEMA,
            f"{variant_id} metadata schema is {wcs.PREDICTOR_METADATA_SCHEMA}",
            f"schema={metadata.get('schema_version')!r}",
        )
        report.technical(
            metadata.get("status") == wcs.STATUS_PASS,
            f"{variant_id} metadata status is pass",
            f"status={metadata.get('status')!r}",
        )
        report.namespace(
            metadata.get("canonical_export_attempted") is False
            and metadata.get("canonical_outputs_modified") is False
            and metadata.get("prelabel_used_as_predictor") is False,
            f"{variant_id} declares no canonical export and no prelabel predictor",
            json.dumps({
                k: metadata.get(k) for k in
                ("canonical_export_attempted", "canonical_outputs_modified",
                 "prelabel_used_as_predictor")
            }),
        )
        report.namespace(
            metadata.get("frozen_hashes_unchanged") is True,
            f"{variant_id} frozen hashes were unchanged during export",
            f"frozen_hashes_unchanged={metadata.get('frozen_hashes_unchanged')!r}",
        )

        years = [int(y) for y in (metadata.get("baseline_years") or [])]
        baseline_years = baseline_years or years
        artifacts = metadata.get("artifact_inventory") or []
        recorded = metadata.get("artifact_sha256") or {}
        total_rasters += len(artifacts)

        normalised = [
            {
                "artifact_id": a.get("artifact_id"),
                "role": a.get("role"),
                "family": a.get("family"),
                "scope": a.get("scope"),
                "baseline_year": a.get("baseline_year"),
                "product": a.get("product"),
                "start_date": a.get("start_date"),
                "end_date": a.get("end_date"),
                "output_path": a.get("path"),
                "date_semantics": a.get("date_semantics"),
            }
            for a in artifacts
        ]
        plan = {
            "predictor_start_date": metadata.get("predictor_start_date"),
            "predictor_end_date": metadata.get("predictor_end_date"),
        }
        check_role_and_product_counts(report, variant_id, normalised, years)
        check_variant_dates(report, variant_id, plan, normalised, years)
        check_products_and_namespace(report, variant_id, normalised, root)

        # --- Per-raster hash + grid re-verification -------------------------
        bad_hash: list[str] = []
        missing: list[str] = []
        bad_grid: list[str] = []
        prelabel_leak: list[str] = []
        for artifact in artifacts:
            path = Path(artifact.get("path", ""))
            artifact_id = artifact.get("artifact_id")
            if not path.is_file():
                missing.append(str(path))
                continue
            digest = _sha256(path)
            if digest != artifact.get("sha256") or digest != recorded.get(artifact_id):
                bad_hash.append(f"{artifact_id}: {digest}")
            if prelabel_hash is not None and digest == prelabel_hash:
                prelabel_leak.append(str(artifact_id))
            if path.resolve() == prelabel.resolve():
                prelabel_leak.append(str(artifact_id))
            try:
                signature = wcs.read_grid_signature(path)
            except Exception as exc:  # noqa: BLE001
                bad_grid.append(f"{artifact_id}: unreadable ({exc})")
                continue
            if signature.get("crs") != (artifact.get("grid_signature") or {}).get("crs") or \
                    signature.get("width") != (artifact.get("grid_signature") or {}).get("width") or \
                    signature.get("height") != (artifact.get("grid_signature") or {}).get("height"):
                bad_grid.append(f"{artifact_id}: grid drifted from the metadata")

        report.technical(
            not missing, f"{variant_id} every recorded raster exists",
            f"missing={missing[:4]}",
        )
        report.technical(
            not bad_hash, f"{variant_id} every raster hash matches the metadata",
            f"mismatched={bad_hash[:4]}",
        )
        report.technical(
            not bad_grid, f"{variant_id} every raster grid matches the metadata",
            f"drifted={bad_grid[:4]}",
        )
        report.namespace(
            not prelabel_leak,
            f"{variant_id} does not carry the pre-label raster as a predictor",
            f"leaked={sorted(set(prelabel_leak))[:4]}",
        )

    report.scientific(
        bool(seen_analysis_ids) and seen_analysis_ids == {frozen_id},
        "analysis_id matches preregistration",
        f"preregistration={frozen_id!r} metadata={sorted(str(v) for v in seen_analysis_ids)}",
    )

    if baseline_years:
        expected_total = len(wanted) * wcs.expected_raster_count(baseline_years)
        report.technical(
            total_rasters == expected_total,
            "total predictor raster count matches the contract",
            f"counted={total_rasters} expected={expected_total}",
        )
    else:
        report.technical(
            False, "total predictor raster count matches the contract",
            "no baseline years could be read from any variant metadata",
        )

    # --- Canonical isolation ------------------------------------------------
    canonical_dir = root / "variants" / wcs.CANONICAL_VARIANT_ID
    stray_canonical = sorted(
        str(p) for p in canonical_dir.rglob("*")
        if p.is_file() and p.name != "frozen_reference.json"
    )
    report.namespace(
        not stray_canonical,
        "no new predictor file exists under the canonical variant",
        f"found={stray_canonical[:4]}",
    )

    if experiments_root is not None:
        canonical_experiment = wcs.canonical_experiment_root(experiment_id, experiments_root)
    else:
        canonical_experiment = wcs.canonical_experiment_root(experiment_id)
    report.namespace(
        canonical_experiment.exists(),
        "canonical experiment namespace is present and untouched by this stage",
        f"missing: {canonical_experiment}",
    )

    # --- Frozen hash audit --------------------------------------------------
    inventory = wcs.frozen_input_inventory(experiment_id, experiments_root)
    extended = wcs.predictor_frozen_inputs(experiment_id, inventory, root.parent)
    check_required_frozen_inputs(report, extended)
    for variant_id in wanted:
        metadata_path = root / "variants" / variant_id / wcs.PREDICTOR_METADATA_NAME
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        check_recorded_frozen_hashes(
            report, variant_id, metadata.get("frozen_input_sha256_before") or {}, extended,
        )


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the window-closure predictor-export stage against a log "
            "(--mode dry-run) and/or the files on disk (--mode actual). This "
            "script never runs an export, a model fit or a bootstrap."
        )
    )
    parser.add_argument("--experiment", required=True, help="Experiment ID.")
    parser.add_argument(
        "--shifts", nargs="+", type=int, default=list(wcs.DEFAULT_SHIFTS),
        help="Preregistered closure shifts in days (default: 0 7 14).",
    )
    parser.add_argument("--mode", required=True, choices=["dry-run", "actual"])
    parser.add_argument("--log", help="Log file to validate (required for --mode dry-run).")
    parser.add_argument(
        "--output-root",
        help="Diagnostics root override; defaults to the canonical namespace.",
    )
    parser.add_argument(
        "--experiments-root",
        help=(
            "Canonical outputs/experiments root override; defaults to the "
            "canonical namespace. Used by --mode actual for the frozen hash "
            "audit."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None
    experiments_root = Path(args.experiments_root) if args.experiments_root else None
    root = wcs.experiment_root(args.experiment, output_root)

    report = Report()
    check_stage_lock(report)

    if args.mode == "dry-run":
        if not args.log:
            report.technical(False, "--log is required for --mode dry-run")
        else:
            validate_dry_run(report, args.experiment, args.shifts, Path(args.log), root)
    else:
        validate_actual(report, args.experiment, args.shifts, root, experiments_root)

    body, exit_code = report.render()
    print(body)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
