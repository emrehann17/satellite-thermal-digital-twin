"""
Provenance auditing for the advisor follow-up.

Two independent tools live here:

1. `manavgat_single_step8a_hash_audit` -- proves that EVERY artefact which
   consumes Manavgat's Step8A dataset was built against ONE AND THE SAME
   Step8A content hash. The expected hash is COMPUTED from the canonical file
   on disk; it is never hard-coded. Missing provenance never yields PASS.

2. `frozen_hash_inventory` -- a before/after content-hash inventory of the
   frozen scientific artefacts, with a diff that separates changes that were
   expected (Manavgat-dependent reruns) from changes that were not (anything
   belonging purely to the Bejis/Mugla/Evia triangle).

Both are READ-ONLY with respect to every scientific artefact.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.paths import PROJECT_ROOT

SCHEMA_VERSION_AUDIT = "advisor_followup.manavgat_single_step8a_hash_audit.v1"
SCHEMA_VERSION_INVENTORY = "advisor_followup.frozen_hash_inventory.v2"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "advisor_followup_provenance"
EXPERIMENTS_ROOT = PROJECT_ROOT / "outputs" / "experiments"
CROSS_REGION_ROOT = PROJECT_ROOT / "outputs" / "cross_region"
ROBUSTNESS_ROOT = PROJECT_ROOT / "outputs" / "robustness"
STEP9G_ROOT = (
    PROJECT_ROOT / "outputs" / "diagnostics"
    / "step9g_univariate_feature_auc_direction_reversal"
)
SYNTHESIS_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "multi_aoi_transfer_synthesis"

SUBJECT = "manavgat_2021"
AOIS = ("manavgat_2021", "bejis_2022", "mugla_2021", "evia_2021_extended")

STEP8A_RELATIVE = Path("step8a") / "step8a_500m_modeling_dataset.parquet"

STATUS_PASS = "pass_single_manavgat_step8a_version"
STATUS_MIXED = "mixed_manavgat_step8a_versions"
STATUS_INCOMPLETE = "incomplete_provenance"


class ProvenanceAuditError(SystemExit):
    """Fatal, contract-violating condition."""


def display_path(path: Path) -> str:
    """Render `path` for the report `path` fields.

    Repo-internal paths stay repo-relative POSIX strings so reports remain
    machine-comparable across checkouts; anything outside PROJECT_ROOT (a
    temporary directory, an absolute path supplied by a caller) is rendered
    as its absolute string instead of raising. Purely presentational -- no
    artefact is read, resolved or written here.
    """
    try:
        return Path(path).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# =============================================================================
# Hashing
# =============================================================================
def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_step8a_path(experiment_id: str = SUBJECT) -> Path:
    return EXPERIMENTS_ROOT / experiment_id / STEP8A_RELATIVE


def canonical_step8a_hash(experiment_id: str = SUBJECT) -> str:
    """The expected hash, COMPUTED from disk. Never hard-coded."""
    path = canonical_step8a_path(experiment_id)
    digest = sha256_file(path)
    if digest is None:
        raise ProvenanceAuditError(
            f"canonical Step8A dataset is missing for '{experiment_id}': {path}"
        )
    return digest


# =============================================================================
# 1. Manavgat single-Step8A-hash audit
# =============================================================================
def _pair_dirs_for(experiment_id: str) -> list[Path]:
    out = []
    if CROSS_REGION_ROOT.is_dir():
        for pair_dir in sorted(CROSS_REGION_ROOT.iterdir()):
            if pair_dir.is_dir() and experiment_id in pair_dir.name.split("__"):
                out.append(pair_dir)
    return out


def _search_recorded_hashes(payload: Any, experiment_id: str) -> list[str]:
    """Walk a JSON payload and collect every Step8A SHA-256 recorded for
    `experiment_id`. Only values that are literally recorded are returned --
    nothing is inferred, derived or guessed."""
    found: list[str] = []

    def walk(node: Any, context: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if isinstance(value, str) and len(value) == 64:
                    joined = " ".join(context + (lowered,)).lower()
                    if "step8a" in joined and "sha" in lowered and experiment_id in joined:
                        found.append(value)
                walk(value, context + (lowered,))
        elif isinstance(node, list):
            for item in node:
                walk(item, context)
        elif isinstance(node, str) and len(node) == 64:
            joined = " ".join(context).lower()
            if "step8a" in joined and experiment_id in joined:
                found.append(node)

    walk(payload, ())
    return found


def _consumer_files(experiment_id: str) -> list[dict[str, Any]]:
    """Enumerate every artefact expected to consume the subject's Step8A.

    `provenance_required` marks the artefacts whose SPECIFIC JOB is to record
    input provenance (input audits, preregistrations, manifests). If any of
    those is absent or carries no Step8A hash, the audit reports
    `incomplete_provenance` and never PASS. Other artefacts are checked
    opportunistically: if they happen to record a hash it must agree, but their
    silence is not itself a failure.
    """
    consumers: list[dict[str, Any]] = []

    def add(group: str, path: Path, required: bool) -> None:
        consumers.append({"group": group, "path": path, "provenance_required": required})

    base = EXPERIMENTS_ROOT / experiment_id
    add("within_region/step8b", base / "step8b" / "step8b_model_comparison_metrics.json", False)
    add("within_region/step8c", base / "step8c" / "step8c_bootstrap_metrics.json", False)
    add("within_region/step8e", base / "step8e" / "final_step8_report.json", False)
    step8d_dir = base / "step8d"
    if step8d_dir.is_dir():
        for candidate in sorted(step8d_dir.glob("*.json")):
            add("within_region/step8d", candidate, False)

    for namespace in ("step8_big_blocks_v2", "step8_big_blocks"):
        comparison = base / "robustness" / namespace / "comparison"
        summary = comparison / "big_block_robustness_summary.json"
        manifest = comparison / "manifest.json"
        if summary.is_file() or manifest.is_file():
            add(f"large_block_robustness/{namespace}", manifest, True)
            add(f"large_block_robustness/{namespace}", summary, False)
            break

    for pair_dir in _pair_dirs_for(experiment_id):
        required_rel = (
            Path("step9a") / "cross_region_input_audit.json",
            Path("step10") / "step10_input_audit.json",
        )
        optional_rel = (
            Path("step9b") / "cross_region_transfer_metrics.json",
            Path("step9c") / "cross_region_permutation_metrics.json",
            Path("step9d") / "cross_region_final_report.json",
            Path("step9e") / "step9e_final_report.json",
            Path("step10") / "step10_final_report.json",
            Path("step10") / "step10_preregistration.json",
        )
        for rel in required_rel:
            add(f"cross_region/{pair_dir.name}/{rel.parts[0]}", pair_dir / rel, True)
        for rel in optional_rel:
            candidate = pair_dir / rel
            if candidate.is_file():
                add(f"cross_region/{pair_dir.name}/{rel.parts[0]}", candidate, False)

    if STEP9G_ROOT.is_dir():
        for pair_dir in sorted(STEP9G_ROOT.iterdir()):
            if not pair_dir.is_dir() or experiment_id not in pair_dir.name.split("__"):
                continue
            add(f"step9g/{pair_dir.name}", pair_dir / "step9g_input_audit.json", True)
            for name in ("step9g_final_report.json", "step9g_preregistration.json"):
                candidate = pair_dir / name
                if candidate.is_file():
                    add(f"step9g/{pair_dir.name}", candidate, False)

    return consumers


def run_manavgat_single_step8a_hash_audit(
    experiment_id: str = SUBJECT, write: bool = True,
) -> dict[str, Any]:
    expected = canonical_step8a_hash(experiment_id)
    consumers = _consumer_files(experiment_id)

    records: list[dict[str, Any]] = []
    observed: set[str] = set()
    missing_files: list[str] = []
    no_provenance: list[str] = []

    required_gaps: list[str] = []

    for consumer in consumers:
        path: Path = consumer["path"]
        required = bool(consumer.get("provenance_required"))
        rel = display_path(path)
        if not path.is_file():
            missing_files.append(rel)
            if required:
                required_gaps.append(rel)
            records.append({
                "group": consumer["group"], "path": rel, "exists": False,
                "provenance_required": required, "recorded_step8a_hashes": [],
                "status": "artifact_missing",
            })
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            no_provenance.append(rel)
            if required:
                required_gaps.append(rel)
            records.append({
                "group": consumer["group"], "path": rel, "exists": True,
                "provenance_required": required, "recorded_step8a_hashes": [],
                "status": f"unreadable: {type(exc).__name__}",
            })
            continue

        hashes = sorted(set(_search_recorded_hashes(payload, experiment_id)))
        observed.update(hashes)
        if not hashes:
            no_provenance.append(rel)
            if required:
                required_gaps.append(rel)
            status = "no_recorded_step8a_provenance"
        elif set(hashes) == {expected}:
            status = "matches_canonical"
        else:
            status = "disagrees_with_canonical"
        records.append({
            "group": consumer["group"], "path": rel, "exists": True,
            "provenance_required": required, "recorded_step8a_hashes": hashes,
            "status": status,
        })

    distinct = sorted(observed)
    if len(distinct) > 1 or (distinct and distinct != [expected]):
        status = STATUS_MIXED
    elif required_gaps or not distinct:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_PASS

    payload = {
        "schema_version": SCHEMA_VERSION_AUDIT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subject_experiment_id": experiment_id,
        "canonical_step8a_path": display_path(canonical_step8a_path(experiment_id)),
        "canonical_step8a_sha256": expected,
        "canonical_hash_source": "computed from the file on disk at audit time; never hard-coded",
        "consumers_checked": len(records),
        "distinct_recorded_step8a_hashes": distinct,
        "unique_hash_count": len(distinct),
        "artifacts_missing": missing_files,
        "artifacts_without_recorded_provenance": no_provenance,
        "required_provenance_gaps": required_gaps,
        "status": status,
        "acceptance_condition": "unique final Manavgat Step8A content hash count == 1",
        "passes": status == STATUS_PASS,
        "synthesis_may_run": status == STATUS_PASS,
        "records": records,
    }

    if write:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "manavgat_single_step8a_hash_audit.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        (OUTPUT_ROOT / "manavgat_single_step8a_hash_audit.md").write_text(
            _render_audit_markdown(payload)
        )
    return payload


def _render_audit_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Manavgat single-Step8A-hash audit — `{payload['status']}`")
    add("")
    add(f"- Generated: `{payload['created_at']}`")
    add(f"- Subject: `{payload['subject_experiment_id']}`")
    add(f"- Canonical Step8A: `{payload['canonical_step8a_path']}`")
    add(f"- Canonical SHA-256: `{payload['canonical_step8a_sha256']}`")
    add(f"- Hash source: {payload['canonical_hash_source']}")
    add(f"- Consumers checked: **{payload['consumers_checked']}**")
    add(f"- Distinct recorded Step8A hashes: **{payload['unique_hash_count']}**")
    add("")
    add(f"**Acceptance condition:** `{payload['acceptance_condition']}` → "
        f"**{'PASS' if payload['passes'] else 'FAIL'}**")
    add("")
    if payload["artifacts_missing"]:
        add(f"### Missing artefacts ({len(payload['artifacts_missing'])})")
        add("")
        for item in payload["artifacts_missing"]:
            add(f"- `{item}`")
        add("")
    if payload["artifacts_without_recorded_provenance"]:
        add(f"### Artefacts with no recorded Step8A provenance "
            f"({len(payload['artifacts_without_recorded_provenance'])})")
        add("")
        add("These do not record a Step8A hash. The hash was NOT inferred for them; "
            "their presence forces `incomplete_provenance` rather than PASS.")
        add("")
        for item in payload["artifacts_without_recorded_provenance"]:
            add(f"- `{item}`")
        add("")
    add("### Distinct hashes observed")
    add("")
    for digest in payload["distinct_recorded_step8a_hashes"]:
        marker = " ← canonical" if digest == payload["canonical_step8a_sha256"] else " ← **UNEXPECTED**"
        add(f"- `{digest}`{marker}")
    add("")
    add("### Per-consumer detail")
    add("")
    add("| Group | Artefact | Status | Recorded hashes |")
    add("|---|---|---|---|")
    for record in payload["records"]:
        digests = ", ".join(f"`{h[:12]}…`" for h in record["recorded_step8a_hashes"]) or "—"
        add(f"| {record['group']} | `{Path(record['path']).name}` | {record['status']} | {digests} |")
    add("")
    return "\n".join(lines)


# =============================================================================
# 2. Frozen hash inventory (before / after / diff)
# =============================================================================
def _inventory_targets() -> list[Path]:
    targets: list[Path] = []

    for aoi in AOIS:
        base = EXPERIMENTS_ROOT / aoi
        targets.append(base / STEP8A_RELATIVE)
        targets.append(base / "step8b" / "step8b_model_comparison_metrics.json")
        targets.append(base / "step8c" / "step8c_bootstrap_metrics.json")
        targets.append(base / "step8e" / "final_step8_report.json")
        for namespace in ("step8_big_blocks", "step8_big_blocks_v2"):
            comparison = base / "robustness" / namespace / "comparison"
            targets.append(comparison / "big_block_robustness_summary.json")
            targets.append(comparison / "manifest.json")

    for source, target in combinations(AOIS, 2):
        for token in (f"{source}__{target}", f"{target}__{source}"):
            pair_dir = CROSS_REGION_ROOT / token
            if not pair_dir.is_dir():
                continue
            targets.append(pair_dir / "step9d" / "cross_region_final_report.json")
            targets.append(pair_dir / "step9e" / "step9e_final_report.json")
            targets.append(pair_dir / "step10" / "step10_final_report.json")
            targets.append(pair_dir / "step10" / "step10_metrics.json")
            targets.append(pair_dir / "step10" / "step10_bootstrap_summary.json")
            targets.append(pair_dir / "step10" / "step10_decomposition.csv")

    if STEP9G_ROOT.is_dir():
        for pair_dir in sorted(STEP9G_ROOT.iterdir()):
            if pair_dir.is_dir():
                targets.append(pair_dir / "step9g_final_report.json")

    if SYNTHESIS_ROOT.is_dir():
        for set_dir in sorted(SYNTHESIS_ROOT.iterdir()):
            if set_dir.is_dir():
                for name in (
                    "multi_aoi_transfer_synthesis.json",
                    "multi_aoi_transfer_matrix.csv",
                    "multi_aoi_within_region.csv",
                    "multi_aoi_manifest.json",
                ):
                    targets.append(set_dir / name)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in targets:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def build_frozen_hash_inventory(phase: str, write: bool = True) -> dict[str, Any]:
    """`phase` is 'before' or 'after'. Deterministic: sorted, content-only."""
    if phase not in ("before", "after"):
        raise ProvenanceAuditError("phase must be 'before' or 'after'.")

    entries = []
    for path in _inventory_targets():
        digest = sha256_file(path)
        entries.append({
            "path": display_path(path),
            "sha256": digest,
            "exists": digest is not None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        })
    entries.sort(key=lambda item: item["path"])

    payload = {
        "schema_version": SCHEMA_VERSION_INVENTORY,
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "present_count": sum(1 for e in entries if e["exists"]),
        "entries": entries,
    }
    if write:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / f"frozen_hash_inventory_{phase}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
    return payload


def _is_manavgat_dependent(rel_path: str) -> bool:
    """Artefacts legitimately expected to change when Manavgat Step8A is
    regenerated: anything under the Manavgat experiment namespace, and any
    cross-region / Step9G artefact for a pair that includes Manavgat."""
    if f"experiments/{SUBJECT}/" in rel_path:
        return True
    parts = rel_path.split("/")
    for token in parts:
        if "__" in token and SUBJECT in token.split("__"):
            return True
    return False


def _is_synthesis(rel_path: str) -> bool:
    return "multi_aoi_transfer_synthesis" in rel_path


def diff_frozen_hash_inventory(write: bool = True) -> dict[str, Any]:
    before_path = OUTPUT_ROOT / "frozen_hash_inventory_before.json"
    after_path = OUTPUT_ROOT / "frozen_hash_inventory_after.json"
    for path in (before_path, after_path):
        if not path.is_file():
            raise ProvenanceAuditError(f"missing inventory: {path}")

    before = {e["path"]: e for e in json.loads(before_path.read_text())["entries"]}
    after = {e["path"]: e for e in json.loads(after_path.read_text())["entries"]}

    buckets: dict[str, list[dict[str, Any]]] = {
        "expected_changed": [],
        "unexpected_changed": [],
        "unchanged": [],
        "missing_before": [],
        "missing_after": [],
        "new_output": [],
    }

    for path in sorted(set(before) | set(after)):
        b, a = before.get(path), after.get(path)
        if b is None:
            buckets["new_output"].append({"path": path, "after_sha256": a["sha256"]})
            continue
        if a is None:
            buckets["missing_after"].append({"path": path, "before_sha256": b["sha256"]})
            continue
        if not b["exists"] and a["exists"]:
            buckets["missing_before"].append({"path": path, "after_sha256": a["sha256"]})
            continue
        if b["exists"] and not a["exists"]:
            buckets["missing_after"].append({"path": path, "before_sha256": b["sha256"]})
            continue
        if b["sha256"] == a["sha256"]:
            buckets["unchanged"].append({"path": path, "sha256": a["sha256"]})
            continue
        entry = {"path": path, "before_sha256": b["sha256"], "after_sha256": a["sha256"]}
        if _is_manavgat_dependent(path) or _is_synthesis(path):
            entry["reason"] = (
                "manavgat_dependent_rerun" if _is_manavgat_dependent(path)
                else "synthesis_regenerated_from_manavgat_dependent_inputs"
            )
            buckets["expected_changed"].append(entry)
        else:
            entry["reason"] = (
                "belongs to the Bejis/Mugla/Evia triangle or a non-Manavgat AOI and "
                "must not have changed"
            )
            buckets["unexpected_changed"].append(entry)

    payload = {
        "schema_version": SCHEMA_VERSION_INVENTORY,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": {name: len(items) for name, items in buckets.items()},
        "unexpected_change_count": len(buckets["unexpected_changed"]),
        "passes": len(buckets["unexpected_changed"]) == 0 and len(buckets["missing_after"]) == 0,
        **buckets,
    }
    if write:
        (OUTPUT_ROOT / "frozen_hash_inventory_diff.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
    return payload
