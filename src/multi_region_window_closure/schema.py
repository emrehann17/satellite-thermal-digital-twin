"""Typed output-schema specifications and expected row-count formulas.

Declarative only: this module states what each artefact must look like. It
produces no scientific output. The validator consumes these specifications, so
a schema change is made in exactly one place.

Design reference: docs/multi_region_window_closure_design/OUTPUT_SCHEMA.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from src.multi_region_window_closure.bootstrap import (
    BOOTSTRAP_REPLICATE_COLUMNS,
    BOOTSTRAP_SUMMARY_COLUMNS,
    SERIES_PER_AOI,
    bootstrap_summary_row_count,
    four_region_synthesis_row_count,
    metrics_row_count,
)
from src.multi_region_window_closure.cohort import (
    COHORT_INVENTORY_COLUMNS, FOLD_MAPPING_COLUMNS,
)
from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS,
    METRICS,
    MODEL_FAMILIES,
    MultiRegionWindowClosureError,
    SCHEMA_VERSION,
    SYNTHESIS_AOIS,
    VARIANTS,
)
from src.multi_region_window_closure.per_aoi import (
    REGIONAL_REQUIRED_ARTIFACTS, SYNTHESIS_REQUIRED_ARTIFACTS,
)
from src.multi_region_window_closure.dates import WINDOW_DATE_COLUMNS
from src.multi_region_window_closure.plan import EXPORT_PLAN_COLUMNS, export_plan_row_count

METRICS_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "variant", "model", "metric", "estimate",
    "metric_direction", "n_rows", "n_positive", "n_negative", "prevalence",
    "fold_count", "oof_complete", "cohort_hash", "fold_mapping_hash",
    "prediction_hash",
)

OOF_PREDICTION_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "variant", "model", "row_id", "grid_id", "cell_id",
    "block_id", "fold_id", "y_true", "y_score", "cohort_hash",
    "fold_mapping_hash",
)

VARIANT_ARTIFACT_INDEX_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "variant", "stage", "relative_path", "artifact_id",
    "role", "media_type", "size_bytes", "sha256", "band_count",
    "pixel_size_x", "pixel_size_y", "crs", "alignment_qa_pass",
    "produced_at_utc", "transport",
)

REGIONAL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "aoi_role", "display_name", "country",
    "canonical_step8a_sha256", "predictor_start", "predictor_end",
    "calendar_duration_days", "label_start", "label_end", "cohort_rows",
    "positives", "negatives", "prevalence", "block_count", "fold_count",
    "cohort_hash", "fold_mapping_hash", "modis_clipped_days_7d",
    "modis_clipped_days_14d", "technical_status", "n_supported_intervals",
    "n_intervals_including_zero", "regime_note",
)

FOUR_REGION_SYNTHESIS_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "aoi_source", "reference_artifact_path",
    "reference_artifact_sha256", "comparison_family", "variant", "model_a",
    "model_b", "metric", "metric_direction",
    "point_estimate_natural", "ci_low_natural", "ci_high_natural",
    "point_estimate_oriented", "ci_low_oriented", "ci_high_oriented",
    "orientations_equal", "interval_status", "prevalence", "regime_class",
    "cross_aoi_comparable", "descriptive_only",
)


@dataclass(frozen=True)
class ArtifactSpec:
    """Declarative contract for one output artefact."""

    relative_path: str
    stage: str
    kind: str                       # csv | parquet | json | markdown
    required: bool = True
    grain: str = ""
    primary_key: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    non_null: tuple[str, ...] = ()
    row_count_formula: str = ""
    expected_rows: Optional[Callable[..., int]] = field(default=None, repr=False)
    notes: str = ""


def _oof_rows(cohort_rows_by_aoi: Optional[dict[str, int]] = None) -> int:
    if not cohort_rows_by_aoi:
        return 0
    return sum(
        int(n) * len(VARIANTS) * len(MODEL_FAMILIES)
        for n in cohort_rows_by_aoi.values()
    )


def _replicate_rows(valid_replicates_by_aoi: Optional[dict[str, int]] = None) -> int:
    if not valid_replicates_by_aoi:
        return 0
    return sum(SERIES_PER_AOI * int(n) for n in valid_replicates_by_aoi.values())


ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        relative_path="config.json", stage="plan", kind="json",
        grain="one document", primary_key=("analysis_id",),
        notes="Frozen, hashable set-level configuration.",
    ),
    ArtifactSpec(
        relative_path="input_hashes.json", stage="plan", kind="json",
        grain="one document", primary_key=("analysis_id",),
        notes="Six required frozen input roles per AOI, with digests.",
    ),
    ArtifactSpec(
        relative_path="repository_inventory.json", stage="plan", kind="json",
        grain="one document", primary_key=("analysis_id",),
        notes="Git commit, python version, dependency lock hash, module digests.",
    ),
    ArtifactSpec(
        relative_path="window_dates.csv", stage="plan", kind="csv",
        grain="AOI x variant", primary_key=("analysis_id", "aoi", "variant"),
        columns=WINDOW_DATE_COLUMNS,
        non_null=("aoi", "variant", "predictor_start", "predictor_end",
                  "label_start", "label_end", "event_source_field",
                  "gate_source_field"),
        row_count_formula="(A + 1) x V = 4 x 3 = 12",
        expected_rows=lambda: len(SYNTHESIS_AOIS) * len(VARIANTS),
    ),
    ArtifactSpec(
        relative_path="export_plan.csv", stage="plan", kind="csv",
        grain="AOI x variant x artefact",
        primary_key=("analysis_id", "aoi", "variant", "artifact_id"),
        columns=EXPORT_PLAN_COLUMNS,
        row_count_formula="per AOI: rasters x V + 1 prelabel + 11 static reuse",
        expected_rows=lambda: export_plan_row_count(ACTUAL_AOIS),
    ),
    ArtifactSpec(
        relative_path="cohort_inventory.csv", stage="fit", kind="csv",
        grain="AOI x variant", primary_key=("analysis_id", "aoi", "variant"),
        columns=COHORT_INVENTORY_COLUMNS,
        row_count_formula="A x V = 3 x 3 = 9",
        expected_rows=lambda: len(ACTUAL_AOIS) * len(VARIANTS),
    ),
    ArtifactSpec(
        relative_path="fold_mapping.parquet", stage="fit", kind="parquet",
        grain="AOI x cohort row", primary_key=("analysis_id", "aoi", "cell_id"),
        columns=FOLD_MAPPING_COLUMNS,
        row_count_formula="sum_aoi N_aoi (per AOI, NOT per variant)",
    ),
    ArtifactSpec(
        relative_path="variant_artifact_index.csv", stage="export", kind="csv",
        grain="AOI x variant x artefact",
        primary_key=("analysis_id", "aoi", "variant", "relative_path"),
        columns=VARIANT_ARTIFACT_INDEX_COLUMNS,
    ),
    ArtifactSpec(
        relative_path="metrics.csv", stage="fit", kind="csv",
        grain="AOI x variant x model x metric",
        primary_key=("analysis_id", "aoi", "variant", "model", "metric"),
        columns=METRICS_COLUMNS,
        row_count_formula="A x V x M x 3 = 3 x 3 x 2 x 3 = 54",
        expected_rows=lambda: metrics_row_count(ACTUAL_AOIS),
    ),
    ArtifactSpec(
        relative_path="oof_predictions.parquet", stage="fit", kind="parquet",
        grain="AOI x variant x model x cohort row",
        primary_key=("analysis_id", "aoi", "variant", "model", "row_id"),
        columns=OOF_PREDICTION_COLUMNS,
        row_count_formula="sum_aoi (N_aoi x V x M) = sum_aoi (N_aoi x 6)",
        expected_rows=_oof_rows,
    ),
    ArtifactSpec(
        relative_path="bootstrap_replicates.parquet", stage="compare", kind="parquet",
        grain="AOI x comparison series x replicate",
        primary_key=("analysis_id", "aoi", "comparison_family", "variant",
                     "model_a", "model_b", "metric", "replicate_id"),
        columns=BOOTSTRAP_REPLICATE_COLUMNS,
        row_count_formula="sum_aoi (27 x valid_replicates) = 81,000 when all valid",
        expected_rows=_replicate_rows,
    ),
    ArtifactSpec(
        relative_path="bootstrap_summary.csv", stage="compare", kind="csv",
        grain="AOI x comparison series (both orientations in columns)",
        primary_key=("analysis_id", "aoi", "comparison_family", "variant",
                     "model_a", "model_b", "metric"),
        columns=BOOTSTRAP_SUMMARY_COLUMNS,
        row_count_formula="A x 27 = 3 x 27 = 81",
        expected_rows=lambda: bootstrap_summary_row_count(ACTUAL_AOIS),
    ),
    ArtifactSpec(
        relative_path="regional_summary.csv", stage="summarize", kind="csv",
        grain="one row per AOI", primary_key=("analysis_id", "aoi"),
        columns=REGIONAL_SUMMARY_COLUMNS,
        row_count_formula="A = 3",
        expected_rows=lambda: len(ACTUAL_AOIS),
    ),
    ArtifactSpec(
        relative_path="four_region_synthesis.csv", stage="summarize", kind="csv",
        grain="AOI x comparison series",
        primary_key=("analysis_id", "aoi", "comparison_family", "variant",
                     "model_a", "model_b", "metric"),
        columns=FOUR_REGION_SYNTHESIS_COLUMNS,
        row_count_formula="len(SYNTHESIS_AOIS) x 27",
        expected_rows=lambda: four_region_synthesis_row_count(SYNTHESIS_AOIS),
        notes="Descriptive only. No column may be a function of >1 AOI.",
    ),
    ArtifactSpec(
        relative_path="summary.json", stage="summarize", kind="json",
        grain="one document", primary_key=("analysis_id",),
        notes="technical_status and scientific_summary are separate keys.",
    ),
    ArtifactSpec(
        relative_path="report.md", stage="summarize", kind="markdown",
        grain="one document",
    ),
    ArtifactSpec(
        relative_path="manifest.json", stage="summarize", kind="json",
        grain="one document", primary_key=("analysis_id",),
        notes="Excluded from its own files[]; digest lives in manifest.sha256.",
    ),
)

#: Files that record hashes and therefore cannot appear in their own manifest.
MANIFEST_CONTROL_FILES: frozenset = frozenset({"manifest.json", "manifest.sha256", "validator_results.json", "validator_summary.json"})


def artifact_spec(relative_path: str) -> ArtifactSpec:
    for spec in ARTIFACT_SPECS:
        if spec.relative_path == relative_path:
            return spec
    raise MultiRegionWindowClosureError(
        f"No artefact specification for '{relative_path}'."
    )


def required_artifacts() -> tuple[str, ...]:
    return tuple(spec.relative_path for spec in ARTIFACT_SPECS if spec.required)


def expected_row_counts(
    cohort_rows_by_aoi: Optional[dict[str, int]] = None,
    valid_replicates_by_aoi: Optional[dict[str, int]] = None,
) -> dict[str, Optional[int]]:
    """Expected row count per tabular artefact, from the frozen formulas."""
    out: dict[str, Optional[int]] = {}
    for spec in ARTIFACT_SPECS:
        if spec.expected_rows is None:
            continue
        if spec.relative_path == "oof_predictions.parquet":
            out[spec.relative_path] = _oof_rows(cohort_rows_by_aoi)
        elif spec.relative_path == "bootstrap_replicates.parquet":
            out[spec.relative_path] = _replicate_rows(valid_replicates_by_aoi)
        else:
            out[spec.relative_path] = spec.expected_rows()
    return out


def stage_state_relpath(stage: str) -> str:
    return f"stages/{stage}.json"


# These are independent authoritative catalogues. They intentionally do not
# filter or inherit row formulas from the legacy combined catalogue above.
REGIONAL_ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("config.json", "plan", "json", grain="one regional run"),
    ArtifactSpec("input_hashes.json", "plan", "json", grain="one regional run"),
    ArtifactSpec("repository_inventory.json", "plan", "json", grain="one regional run"),
    ArtifactSpec("window_dates.csv", "plan", "csv", grain="3 variants",columns=WINDOW_DATE_COLUMNS,primary_key=("analysis_id","aoi","variant"),non_null=("analysis_id","aoi","variant","predictor_start","predictor_end","label_start","label_end"), expected_rows=lambda: 3),
    ArtifactSpec("export_plan.csv", "plan", "csv", grain="regional export artefacts",columns=EXPORT_PLAN_COLUMNS,primary_key=("aoi","variant","artifact_id"),non_null=("aoi","variant","artifact_id","role"), expected_rows=lambda aoi=None: export_plan_row_count((aoi,)) if aoi else None),
    ArtifactSpec("cohort_inventory.csv", "fit", "csv", grain="3 variants",columns=COHORT_INVENTORY_COLUMNS,primary_key=("aoi","variant"),non_null=("aoi","variant","cohort_hash","final_common_cohort_rows"), expected_rows=lambda: 3),
    ArtifactSpec("fold_mapping.parquet", "fit", "parquet", grain="one row per cohort row",columns=FOLD_MAPPING_COLUMNS,primary_key=("aoi","cell_id"),non_null=FOLD_MAPPING_COLUMNS),
    ArtifactSpec("variant_artifact_index.csv", "export", "csv", grain="regional variant artefacts",columns=VARIANT_ARTIFACT_INDEX_COLUMNS,primary_key=("aoi","variant","relative_path"),non_null=("aoi","variant","relative_path","sha256")),
    ArtifactSpec("metrics.csv", "fit", "csv", grain="3 variants x 2 models x 3 metrics",columns=METRICS_COLUMNS,primary_key=("aoi","variant","model","metric"),non_null=("aoi","variant","model","metric","estimate"), expected_rows=lambda: 18),
    ArtifactSpec("oof_predictions.parquet", "fit", "parquet", grain="cohort rows x 3 x 2",columns=OOF_PREDICTION_COLUMNS,primary_key=("aoi","variant","model","row_id"),non_null=("aoi","variant","model","row_id","fold_id","y_true","y_score"), expected_rows=lambda cohort_rows=0: int(cohort_rows) * 6),
    ArtifactSpec("bootstrap_replicates.parquet", "compare", "parquet", grain="27 series x requested replicate",columns=BOOTSTRAP_REPLICATE_COLUMNS,primary_key=("aoi","comparison_family","variant","model_a","model_b","metric","replicate_id"),non_null=("aoi","comparison_family","variant","model_a","model_b","metric","replicate_id","draw_plan_id","valid"), expected_rows=lambda valid_replicates=1000: 27 * int(valid_replicates)),
    ArtifactSpec("bootstrap_summary.csv", "compare", "csv", grain="27 comparison series",columns=BOOTSTRAP_SUMMARY_COLUMNS,primary_key=("aoi","comparison_family","variant","model_a","model_b","metric"),non_null=("aoi","comparison_family","variant","model_a","model_b","metric","point_estimate_natural","ci_low_natural","ci_high_natural"), expected_rows=lambda: 27),
    ArtifactSpec("regional_summary.csv", "summarize", "csv", grain="one AOI",columns=REGIONAL_SUMMARY_COLUMNS,primary_key=("aoi",),non_null=REGIONAL_SUMMARY_COLUMNS, expected_rows=lambda: 1),
    ArtifactSpec("summary.json", "summarize", "json", grain="one regional run"),
    ArtifactSpec("report.md", "summarize", "markdown", grain="one regional report"),
    ArtifactSpec("manifest.json", "summarize", "json", grain="one regional manifest"),
    ArtifactSpec("manifest.sha256", "summarize", "control", grain="SHA-256 of manifest.json", notes="Excluded from manifest files[]"),
    ArtifactSpec("validator_results.json", "validate", "json", grain="one row per regional check"),
    ArtifactSpec("validator_summary.json", "validate", "json", grain="one regional validator summary"),
)

# Synthesis row expectations derive from ONE source of truth, `SYNTHESIS_AOIS`
# (= ACTUAL_AOIS + REFERENCE_AOI), never from a frozen literal. They were
# previously hardcoded at 4 / 108, which silently went stale when ACTUAL_AOIS
# grew to four and SYNTHESIS_AOIS to five; nothing caught it because no
# synthesis path is executable. Deriving them keeps the dormant contract
# self-consistent if synthesis is ever re-enabled.
#
# Naming debt (deliberately NOT renamed here): the artefact is still called
# `four_region_synthesis.csv` from the original three-actual + one-reference
# design. The name is historical; the grain is `len(SYNTHESIS_AOIS)` AOIs.
SYNTHESIS_ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("config.json", "plan", "json", grain="one synthesis"),
    ArtifactSpec("input_references.json", "plan", "json", grain="one immutable reference per synthesis AOI"),
    ArtifactSpec("input_hashes.json", "plan", "json", grain="one input hash per synthesis AOI"),
    ArtifactSpec("regional_result_index.csv", "summarize", "csv", grain="one row per synthesis AOI", expected_rows=lambda: len(SYNTHESIS_AOIS)),
    ArtifactSpec("four_region_synthesis.csv", "summarize", "csv", grain="synthesis AOIs x 27 series",columns=("aoi","source_analysis_id","comparison_family","variant","model_a","model_b","metric","metric_direction","point_estimate_natural","ci_low_natural","ci_high_natural","point_estimate_oriented","ci_low_oriented","ci_high_oriented","orientations_equal","interval_status","descriptive_only"),primary_key=("aoi","comparison_family","variant","model_a","model_b","metric"),non_null=("aoi","source_analysis_id","comparison_family","variant","model_a","model_b","metric","metric_direction","point_estimate_natural","ci_low_natural","ci_high_natural","point_estimate_oriented","ci_low_oriented","ci_high_oriented","orientations_equal","interval_status","descriptive_only"), expected_rows=lambda: four_region_synthesis_row_count(SYNTHESIS_AOIS)),
    ArtifactSpec("summary.json", "summarize", "json", grain="one synthesis"),
    ArtifactSpec("report.md", "summarize", "markdown", grain="one descriptive report"),
    ArtifactSpec("validator_results.json", "validate", "json", grain="one row per synthesis check"),
    ArtifactSpec("validator_summary.json", "validate", "json", grain="one synthesis validator summary"),
    ArtifactSpec("manifest.json", "summarize", "json", grain="one synthesis manifest"),
    ArtifactSpec("manifest.sha256", "summarize", "control", grain="SHA-256 of manifest.json", notes="Excluded from manifest files[]"),
)


def regional_required_artifacts() -> tuple[str, ...]:
    return REGIONAL_REQUIRED_ARTIFACTS


def synthesis_required_artifacts() -> tuple[str, ...]:
    return SYNTHESIS_REQUIRED_ARTIFACTS


# =============================================================================
# Manifest
# =============================================================================
def build_manifest(
    analysis_id: str,
    namespace: Any,
    config_hash: str,
    input_hashes_hash: str,
    canonical_aoi_hashes: dict[str, str],
    manavgat_reference_hash: str,
    schema_version: str = SCHEMA_VERSION,
    artifact_specs: tuple[ArtifactSpec, ...] = ARTIFACT_SPECS,
) -> dict[str, Any]:
    """Manifest over every file in the namespace, EXCLUDING the control files.

    `manifest.json` cannot contain its own digest -- that is a self-hash cycle.
    Its integrity is established externally by a sibling `manifest.sha256`.
    """
    import platform
    from pathlib import Path

    from core.paths import PROJECT_ROOT
    from src.step8_large_block_robustness import _git_commit, sha256_file

    root = Path(namespace)
    files: list[dict[str, Any]] = []
    total = 0
    required_set = {spec.relative_path for spec in artifact_specs if spec.required}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in MANIFEST_CONTROL_FILES or relative.startswith("stages/"):
            continue
        size = path.stat().st_size
        total += size
        files.append({
            "relative_path": relative,
            "size_bytes": size,
            "sha256": sha256_file(path),
            "stage": _stage_of(relative),
            "required": relative in required_set,
        })
    lock = Path(PROJECT_ROOT) / "requirements-lock.txt"
    return {
        "analysis_id": analysis_id,
        "schema_version": schema_version,
        "created_at_utc": __import__(
            "src.multi_region_window_closure.stages", fromlist=["utc_now"],
        ).utc_now(),
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "dirty_worktree": _dirty_worktree(),
        "python_version": platform.python_version(),
        "dependency_lock_hash": sha256_file(lock) if lock.is_file() else None,
        "config_hash": config_hash,
        "input_hashes_hash": input_hashes_hash,
        "canonical_aoi_hashes": dict(canonical_aoi_hashes),
        "manavgat_reference_hash": manavgat_reference_hash,
        "output_file_count": len(files),
        "output_total_bytes": total,
        "files": files,
    }


def write_manifest_with_digest(path: Any, manifest: dict[str, Any]) -> str:
    """Write manifest bytes followed by their non-cyclic control digest."""
    import hashlib, json
    from pathlib import Path
    manifest_path=Path(path)
    raw=(json.dumps(manifest,indent=2,default=str)+"\n").encode("utf-8")
    manifest_path.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    manifest_path.with_name("manifest.sha256").write_text(digest+"\n",encoding="ascii")
    return digest


def verify_manifest_digest(root: Any) -> tuple[bool, str]:
    """Validate the exact 64-hex digest control file fail-closed."""
    import hashlib, re
    from pathlib import Path
    root=Path(root);manifest=root/"manifest.json";control=root/"manifest.sha256"
    if not manifest.is_file() or not control.is_file(): return False,"manifest or manifest.sha256 missing"
    declared=control.read_text(encoding="ascii").strip()
    if re.fullmatch(r"[0-9a-f]{64}",declared) is None:return False,"manifest.sha256 malformed"
    actual=hashlib.sha256(manifest.read_bytes()).hexdigest()
    return actual==declared,f"declared={declared} actual={actual}"


def _stage_of(relative_path: str) -> str:
    if relative_path.startswith("stages/"):
        return relative_path.split("/", 1)[1].removesuffix(".json")
    for spec in ARTIFACT_SPECS:
        if spec.relative_path == relative_path:
            return spec.stage
    return "unknown"


def _git_branch() -> Optional[str]:
    import subprocess

    from core.paths import PROJECT_ROOT

    try:
        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return None


def _dirty_worktree() -> Optional[bool]:
    import subprocess

    from core.paths import PROJECT_ROOT

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True,
        ).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return None


def assert_manifest_complete(manifest: dict[str, Any], namespace: Any) -> None:
    """Every file on disk recorded, and every recorded digest still matching."""
    from pathlib import Path

    from src.step8_large_block_robustness import sha256_file

    root = Path(namespace)
    recorded = {entry["relative_path"] for entry in manifest.get("files", [])}
    on_disk = {
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)) not in MANIFEST_CONTROL_FILES
    }
    stray = sorted(on_disk - recorded)
    if stray:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: MANIFEST_INCOMPLETE -- stray file(s) {stray[:6]}."
        )
    absent = sorted(recorded - on_disk)
    if absent:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: MANIFEST_INCOMPLETE -- recorded file(s) missing {absent[:6]}."
        )
    if manifest.get("output_file_count") != len(recorded):
        raise MultiRegionWindowClosureError(
            "BLOCKER: MANIFEST_INCOMPLETE -- output_file_count "
            f"{manifest.get('output_file_count')} != {len(recorded)}."
        )
    total = sum(int(e["size_bytes"]) for e in manifest["files"])
    if manifest.get("output_total_bytes") != total:
        raise MultiRegionWindowClosureError(
            "BLOCKER: MANIFEST_INCOMPLETE -- output_total_bytes "
            f"{manifest.get('output_total_bytes')} != {total}."
        )
    for entry in manifest["files"]:
        actual = sha256_file(root / entry["relative_path"])
        if actual != entry["sha256"]:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: OUTPUT_HASH_MISMATCH -- {entry['relative_path']}."
            )
