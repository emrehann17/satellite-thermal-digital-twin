"""
historical_burn_exclusion.py

Generic, declarative HISTORICAL BURN exclusion contract (Gate -> Step8A).

WHY THIS EXISTS
---------------
An experiment may need to exclude cells that burned in an EARLIER event over
the SAME geography. A cell burned in a previous year is not a normal
vegetation state in the current year: its NDVI/LST/TVDI may reflect POST-FIRE
RECOVERY rather than the pre-fire condition the model is supposed to learn.
Such cells must leave the analysis universe entirely -- they are neither
burned positives nor unburned negatives for the current event.

This is a SEPARATE, INDEPENDENT axis from the existing pre-label exclusion
(src/step6b_burned_landcover_gate.py write_pre_label_exclusion_manifest()):

    pre_label_burn_excluded   cell burned inside THIS experiment's predictor
                              window, i.e. before its own label_start.
    historical_burn_excluded  cell burned in an EARLIER experiment/event over
                              the same AOI (this module).

The two are derived independently, persisted in SEPARATE manifests, reported
separately, and only then UNIONed by the gate. A cell may carry both flags.
Historical exclusion is NEVER mixed into pre_label_excluded_cells.parquet.

CONFIG-DRIVEN, NEVER ID-DRIVEN
-------------------------------
Every behaviour here is resolved from generic registry fields (see
core/regions.py):

    exclude_historical_burns                    bool   (opt-in)
    historical_burn_source_experiment           str    source experiment_id
    historical_burn_source_kind                 str    see SOURCE_KIND_* below
    historical_burn_source_expected_count       int    frozen source count

No experiment_id literal appears anywhere in this module.

SOURCE MASK DEFINITION (frozen)
--------------------------------
For source_kind = "canonical_step8a_physical_burned_cells" the historical mask
is EXACTLY the source experiment's canonical Step8A rows with `burned == 1`.
It is deliberately NOT further restricted by TSG / analysis_eligible /
valid_for_modeling / landcover: the scientific claim is PHYSICAL -- this cell
burned -- not "this cell was modellable". The source artifact is opened
read-only and is NEVER rewritten.

FAIL-CLOSED
-----------
Missing source, unreadable source, missing columns, null/duplicate source
cell_id, a source physical-burned count differing from the frozen expectation,
incompatible source/target region or grid identity, or a partially-written
manifest all raise HistoricalBurnExclusionError. Nothing is ever silently
degraded to an empty exclusion set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from core.paths import PROJECT_ROOT
from core.regions import get_experiment, get_experiment_output_root

BASE_DIR = PROJECT_ROOT

# --- Canonical artifact filenames (SEPARATE from the pre-label manifest) ----
HISTORICAL_BURN_EXCLUSION_MANIFEST_PARQUET = "historical_burn_excluded_cells.parquet"
HISTORICAL_BURN_EXCLUSION_MANIFEST_CSV = "historical_burn_excluded_cells.csv"
HISTORICAL_BURN_EXCLUSION_MANIFEST_METADATA = "historical_burn_excluded_cells_metadata.json"

#: The only source kind implemented so far. An unknown kind fails closed
#: rather than being interpreted as "the default one".
SOURCE_KIND_STEP8A_PHYSICAL_BURNED = "canonical_step8a_physical_burned_cells"
SUPPORTED_SOURCE_KINDS = (SOURCE_KIND_STEP8A_PHYSICAL_BURNED,)

#: Relative location of a source experiment's canonical Step8A dataset.
CANONICAL_STEP8A_RELATIVE_PATH = ("step8a", "step8a_500m_modeling_dataset.parquet")

EXCLUSION_REASON = "historical_burn_excluded"

MASK_DEFINITION = (
    "all rows of the source experiment's canonical Step8A modeling dataset "
    "with burned == 1 (PHYSICAL burned cells). Deliberately NOT restricted by "
    "TSG / analysis_eligible / valid_for_modeling / landcover."
)

SCIENTIFIC_RATIONALE = (
    "A cell that burned in the source (earlier) event is excluded from the "
    "target year's analysis universe because its target-year NDVI/LST/TVDI "
    "may reflect post-fire recovery rather than a normal vegetation state."
)

#: Columns every historical-exclusion manifest row carries, in order.
MANIFEST_COLUMNS = [
    "experiment_id",
    "source_experiment_id",
    "cell_id",
    "row_500m",
    "col_500m",
    "source_burned",
    "source_burn_date",
    "source_burn_day_of_year",
    "exclusion_reason",
]

#: Columns the source Step8A parquet MUST provide.
_REQUIRED_SOURCE_COLUMNS = ("cell_id", "row_500m", "col_500m", "burned")
#: Optional source columns, carried through when present.
_OPTIONAL_SOURCE_COLUMNS = ("burn_date", "burn_day_of_year")


class HistoricalBurnExclusionError(SystemExit):
    """Fail-fast error for the historical burn exclusion contract."""


# =============================================================================
# Registry contract resolution (generic)
# =============================================================================
def resolve_historical_burn_contract(exp: dict) -> dict | None:
    """Resolve the historical-burn exclusion contract from a registry record.

    Returns None when the experiment does not opt in (``exclude_historical_burns``
    absent/False) -- every existing experiment therefore keeps its current
    behaviour byte-for-byte.

    Raises:
        HistoricalBurnExclusionError: the experiment opts in but its
            declaration is incomplete, self-referential, or names an
            unsupported source kind.
    """
    if not exp.get("exclude_historical_burns", False):
        return None

    target_id = exp.get("experiment_id")
    if not target_id:
        raise HistoricalBurnExclusionError(
            "Historical burn exclusion: registry record has no 'experiment_id'. "
            "Resolve the record via core.regions.get_experiment() so the target "
            "identity is stamped into the manifest."
        )

    source_id = exp.get("historical_burn_source_experiment")
    if not source_id or not isinstance(source_id, str):
        raise HistoricalBurnExclusionError(
            f"'{target_id}': exclude_historical_burns=True requires a non-empty "
            "'historical_burn_source_experiment' naming the earlier experiment "
            "whose burned cells define the historical mask."
        )
    if source_id == target_id:
        raise HistoricalBurnExclusionError(
            f"'{target_id}': 'historical_burn_source_experiment' must not point "
            "at the experiment itself."
        )
    # Fails closed with the registry's own error when the source is unknown.
    source_exp = get_experiment(source_id)

    source_kind = exp.get("historical_burn_source_kind")
    if source_kind not in SUPPORTED_SOURCE_KINDS:
        raise HistoricalBurnExclusionError(
            f"'{target_id}': unknown historical_burn_source_kind "
            f"{source_kind!r}. Supported: {list(SUPPORTED_SOURCE_KINDS)}."
        )

    expected_count = exp.get("historical_burn_source_expected_count")
    if expected_count is not None:
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise HistoricalBurnExclusionError(
                f"'{target_id}': historical_burn_source_expected_count must be "
                f"an int, got {expected_count!r}."
            )
        if expected_count < 0:
            raise HistoricalBurnExclusionError(
                f"'{target_id}': historical_burn_source_expected_count must not "
                f"be negative (got {expected_count})."
            )

    return {
        "exclude_historical_burns": True,
        "target_experiment_id": target_id,
        "target_region_key": exp.get("region_key"),
        "source_experiment_id": source_id,
        "source_region_key": source_exp.get("region_key"),
        "source_kind": source_kind,
        "source_expected_physical_burned_count": expected_count,
        "mask_definition": MASK_DEFINITION,
        "exclusion_reason": EXCLUSION_REASON,
        "scientific_rationale": SCIENTIFIC_RATIONALE,
        "source_step8a_parquet_path": str(canonical_source_step8a_parquet(source_id)),
    }


def canonical_source_step8a_parquet(source_experiment_id: str) -> Path:
    """Resolved path of a source experiment's canonical Step8A parquet.

    Path computation only -- nothing is read, created or written here.
    """
    root = get_experiment_output_root(source_experiment_id)
    return root.joinpath(*CANONICAL_STEP8A_RELATIVE_PATH).resolve()


def historical_manifest_paths(output_dir: Path) -> dict[str, Path]:
    """The three canonical artifact paths inside a gate labels directory."""
    output_dir = Path(output_dir)
    return {
        "parquet_path": output_dir / HISTORICAL_BURN_EXCLUSION_MANIFEST_PARQUET,
        "csv_path": output_dir / HISTORICAL_BURN_EXCLUSION_MANIFEST_CSV,
        "metadata_path": output_dir / HISTORICAL_BURN_EXCLUSION_MANIFEST_METADATA,
    }


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed. Raises if the file is missing."""
    path = Path(path)
    if not path.is_file():
        raise HistoricalBurnExclusionError(
            f"Historical burn exclusion: cannot hash missing file {path}."
        )
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Region / grid identity compatibility (fail closed)
# =============================================================================
def verify_region_grid_compatibility(
    contract: dict, require_target_grid: bool = True,
) -> dict:
    """Verify that source and target share cell identity, or fail closed.

    Two independent checks:

        1. REGION identity -- both experiments must resolve to the SAME
           ``region_key``. cell_id is a block index on the reference grid, so
           reusing a source cell_id across different AOIs would silently
           address a different piece of ground.
        2. GRID identity -- both experiments' Step6A reference grids
           (``gate_inputs/reference_30m.tif``) must agree on CRS, width and
           height. The grid is what turns a (row_500m, col_500m) block index
           into a physical location.

    ``require_target_grid=False`` is used ONLY by read-only description/dry-run
    paths, where the target's Step6A grid may not exist yet; the region check
    still applies and the returned dict records that the grid check was
    deferred. Manifest construction always uses require_target_grid=True.

    Raises:
        HistoricalBurnExclusionError: identity differs, or cannot be validated.
    """
    target_id = contract["target_experiment_id"]
    source_id = contract["source_experiment_id"]
    target_region = contract.get("target_region_key")
    source_region = contract.get("source_region_key")

    if not target_region or not source_region:
        raise HistoricalBurnExclusionError(
            f"'{target_id}': region_key could not be resolved for the target "
            f"({target_region!r}) and/or source ({source_region!r}); historical "
            "burn exclusion cannot verify cell identity and refuses to proceed."
        )
    if target_region != source_region:
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn exclusion requires the SAME "
            f"geography as its source '{source_id}', but region_key differs "
            f"(target='{target_region}', source='{source_region}'). cell_id is "
            "a grid block index and is not portable across AOIs."
        )

    grids: dict[str, dict | None] = {}
    for role, experiment_id, required in (
        ("source", source_id, True),
        ("target", target_id, require_target_grid),
    ):
        grid_path = _reference_grid_path(experiment_id)
        if not grid_path.is_file():
            if required:
                raise HistoricalBurnExclusionError(
                    f"'{target_id}': the {role} experiment's reference grid is "
                    f"missing ({grid_path}); grid provenance cannot be "
                    "validated, so historical burn exclusion fails closed. Run "
                    "the Step6A gate-input preparation first."
                )
            grids[role] = None
            continue
        grids[role] = _read_grid_identity(grid_path)

    source_grid = grids["source"]
    target_grid = grids["target"]
    if source_grid is not None and target_grid is not None:
        mismatched = {
            key: (target_grid[key], source_grid[key])
            for key in ("crs", "width", "height", "transform")
            if target_grid[key] != source_grid[key]
        }
        if mismatched:
            raise HistoricalBurnExclusionError(
                f"'{target_id}': target and source ('{source_id}') reference "
                f"grids are NOT identical -- {mismatched} (target, source). "
                "Historical cell_id values would address different ground; "
                "refusing to build the exclusion manifest."
            )
        grid_verified = True
        grid_note = (
            "target and source Step6A reference grids agree on crs, width, "
            "height and transform"
        )
    else:
        grid_verified = False
        grid_note = (
            "target reference grid not yet materialised; grid identity check "
            "deferred to manifest construction (read-only description only)"
        )

    return {
        "region_key": target_region,
        "region_key_matches_source": True,
        "grid_identity_verified": grid_verified,
        "grid_identity_note": grid_note,
        "source_reference_grid": source_grid,
        "target_reference_grid": target_grid,
    }


def _reference_grid_path(experiment_id: str) -> Path:
    """Step6A reference-grid path for an experiment (path computation only)."""
    from src.step6a_prepare_gate_inputs import get_gate_input_paths

    return Path(get_gate_input_paths(experiment_id)["reference_path"]).resolve()


def _read_grid_identity(grid_path: Path) -> dict:
    """CRS/size/transform of a raster header. Read-only; opens no bands."""
    import rasterio

    try:
        with rasterio.open(grid_path) as src:
            return {
                "path": str(grid_path),
                "crs": str(src.crs),
                "width": int(src.width),
                "height": int(src.height),
                "transform": [float(v) for v in list(src.transform)[:6]],
            }
    except HistoricalBurnExclusionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HistoricalBurnExclusionError(
            f"Historical burn exclusion: reference grid {grid_path} could not "
            f"be read ({type(exc).__name__}: {exc}); grid provenance cannot be "
            "validated."
        ) from exc


# =============================================================================
# Source reading (read-only; the source artifact is NEVER rewritten)
# =============================================================================
def load_source_physical_burned_cells(contract: dict) -> tuple[pd.DataFrame, dict]:
    """Read the source experiment's canonical Step8A parquet and return its
    PHYSICAL burned cells (``burned == 1``) plus source provenance.

    Fail-closed on: missing source, unreadable source, missing required
    columns, null/duplicate source cell_id, and a physical burned count that
    differs from the frozen ``historical_burn_source_expected_count``.
    """
    target_id = contract["target_experiment_id"]
    source_id = contract["source_experiment_id"]
    source_path = Path(contract["source_step8a_parquet_path"])

    if not source_path.is_file():
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn source dataset is missing "
            f"({source_path}). The canonical Step8A dataset of "
            f"'{source_id}' is REQUIRED when exclude_historical_burns=True."
        )
    try:
        source_df = pd.read_parquet(source_path)
    except Exception as exc:  # noqa: BLE001
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn source dataset ({source_path}) "
            f"could not be read: {type(exc).__name__}: {exc}."
        ) from exc

    missing = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in source_df.columns]
    if missing:
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn source dataset ({source_path}) is "
            f"missing required column(s): {missing}."
        )
    if source_df["cell_id"].isna().any():
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn source dataset ({source_path}) "
            "contains null cell_id values."
        )
    if not source_df["cell_id"].is_unique:
        dupes = sorted(
            set(source_df.loc[source_df["cell_id"].duplicated(), "cell_id"].astype(str))
        )
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn source dataset ({source_path}) "
            f"contains duplicate cell_id values: {dupes[:20]}."
        )

    # THE mask definition: burned == 1, nothing else.
    burned_df = source_df.loc[source_df["burned"] == 1].copy()
    physical_burned_count = int(burned_df["cell_id"].nunique())
    if physical_burned_count != len(burned_df):
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn source dataset ({source_path}) "
            f"yields {len(burned_df)} burned rows but only "
            f"{physical_burned_count} unique cell_id values."
        )

    expected = contract.get("source_expected_physical_burned_count")
    if expected is not None and physical_burned_count != expected:
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn source '{source_id}' yields "
            f"{physical_burned_count} physical burned cells, but the frozen "
            f"registry expectation is {expected} "
            "(historical_burn_source_expected_count). The source artifact has "
            "changed, or the wrong source is being read; refusing to build a "
            "silently different exclusion universe."
        )

    provenance = {
        "source_experiment_id": source_id,
        "source_step8a_parquet_path": str(source_path),
        "source_step8a_parquet_sha256": sha256_file(source_path),
        "source_row_count": int(len(source_df)),
        "source_physical_burned_count": physical_burned_count,
        "source_expected_physical_burned_count": expected,
        "source_optional_columns_present": [
            c for c in _OPTIONAL_SOURCE_COLUMNS if c in source_df.columns
        ],
    }
    return burned_df, provenance


def _build_manifest_rows(burned_df: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Project the source burned rows onto the frozen manifest schema."""
    manifest_df = pd.DataFrame(index=range(len(burned_df)), columns=MANIFEST_COLUMNS)
    manifest_df["experiment_id"] = contract["target_experiment_id"]
    manifest_df["source_experiment_id"] = contract["source_experiment_id"]
    manifest_df["cell_id"] = burned_df["cell_id"].astype(str).to_numpy()
    manifest_df["row_500m"] = burned_df["row_500m"].to_numpy()
    manifest_df["col_500m"] = burned_df["col_500m"].to_numpy()
    manifest_df["source_burned"] = True
    for source_column, manifest_column in (
        ("burn_date", "source_burn_date"),
        ("burn_day_of_year", "source_burn_day_of_year"),
    ):
        if source_column in burned_df.columns:
            manifest_df[manifest_column] = burned_df[source_column].to_numpy()
        else:
            manifest_df[manifest_column] = None
    manifest_df["exclusion_reason"] = EXCLUSION_REASON
    return manifest_df


# =============================================================================
# Manifest construction (writes ONLY into the target gate labels directory)
# =============================================================================
def build_historical_burn_exclusion_manifest(
    exp: dict, output_dir: Path, force: bool = False,
) -> dict | None:
    """Build (or validate) the canonical historical-burn exclusion manifest.

    Writes ``historical_burn_excluded_cells.{parquet,csv}`` plus the
    ``..._metadata.json`` sidecar into ``output_dir`` (the TARGET experiment's
    gate labels directory). The source experiment's artifacts are opened
    read-only and are never modified.

    Returns None when the experiment does not opt in.

    Regeneration semantics mirror the rest of the gate: a COMPLETE existing
    triplet is left untouched unless ``force`` is given, while a PARTIAL
    (inconsistent) triplet always fails closed -- regenerating half a manifest
    silently would leave the parquet and its metadata disagreeing.
    """
    contract = resolve_historical_burn_contract(exp)
    if contract is None:
        return None

    target_id = contract["target_experiment_id"]
    output_dir = Path(output_dir)
    paths = historical_manifest_paths(output_dir)
    existing = {name: path for name, path in paths.items() if path.exists()}
    if existing and len(existing) != len(paths):
        raise HistoricalBurnExclusionError(
            f"'{target_id}': the historical burn exclusion manifest is only "
            f"PARTIALLY present in {output_dir} "
            f"(found: {sorted(p.name for p in existing.values())}; expected all "
            f"of {sorted(p.name for p in paths.values())}). Refusing to read or "
            "silently complete an inconsistent manifest; remove the partial "
            "artifacts deliberately and regenerate."
        )

    compatibility = verify_region_grid_compatibility(contract, require_target_grid=True)
    burned_df, source_provenance = load_source_physical_burned_cells(contract)
    manifest_df = _build_manifest_rows(burned_df, contract)

    if manifest_df["cell_id"].isna().any():
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn exclusion manifest contains null "
            "cell_id values."
        )
    if not manifest_df["cell_id"].is_unique:
        dupes = sorted(
            set(manifest_df.loc[manifest_df["cell_id"].duplicated(), "cell_id"])
        )
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn exclusion manifest contains "
            f"duplicate cell_id values: {dupes[:20]}."
        )
    if manifest_df[["row_500m", "col_500m"]].drop_duplicates().shape[0] != len(manifest_df):
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn exclusion manifest has fewer "
            "unique (row_500m, col_500m) pairs than cell_id values -- the cell "
            "identity scheme is inconsistent."
        )

    unique_excluded_count = int(manifest_df["cell_id"].nunique())
    if unique_excluded_count != source_provenance["source_physical_burned_count"]:
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical exclusion manifest has "
            f"{unique_excluded_count} unique cell_id values but the source mask "
            f"has {source_provenance['source_physical_burned_count']} physical "
            "burned cells. These must be identical."
        )

    metadata = {
        "manifest_kind": "historical_burn_exclusion",
        "experiment_id": target_id,
        "exclude_historical_burns": True,
        "source_experiment_id": contract["source_experiment_id"],
        "source_kind": contract["source_kind"],
        "mask_definition": contract["mask_definition"],
        "scientific_rationale": contract["scientific_rationale"],
        "exclusion_reason": EXCLUSION_REASON,
        "unique_historical_excluded_count": unique_excluded_count,
        "cell_id_scheme": (
            "r{row_500m}_c{col_500m}, native ~500m MCD64A1-grid block index "
            "(src.step8a_prepare_500m_modeling_dataset.compute_cell_identity) "
            "-- the SAME scheme the gate, Step8A and the source dataset use; "
            "never independently reimplemented."
        ),
        "target_region_key": contract["target_region_key"],
        "source_region_key": contract["source_region_key"],
        "region_grid_compatibility": compatibility,
        "parquet_path": str(paths["parquet_path"]),
        "csv_path": str(paths["csv_path"]),
        **source_provenance,
    }

    if existing and not force:
        stored = _read_manifest_metadata(paths["metadata_path"], target_id)
        _assert_existing_manifest_matches(stored, metadata, paths, target_id)
        return {
            "parquet_path": str(paths["parquet_path"]),
            "csv_path": str(paths["csv_path"]),
            "metadata_path": str(paths["metadata_path"]),
            "excluded_cell_count": unique_excluded_count,
            "created": False,
            "reused_existing": True,
            "metadata": stored,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_parquet(paths["parquet_path"], index=False)
    manifest_df.to_csv(paths["csv_path"], index=False)
    paths["metadata_path"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return {
        "parquet_path": str(paths["parquet_path"]),
        "csv_path": str(paths["csv_path"]),
        "metadata_path": str(paths["metadata_path"]),
        "excluded_cell_count": unique_excluded_count,
        "created": True,
        "reused_existing": False,
        "metadata": metadata,
    }


def _read_manifest_metadata(metadata_path: Path, target_id: str) -> dict:
    try:
        return json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HistoricalBurnExclusionError(
            f"'{target_id}': historical burn exclusion metadata "
            f"({metadata_path}) could not be read: {type(exc).__name__}: {exc}."
        ) from exc


def _assert_existing_manifest_matches(
    stored: dict, freshly_resolved: dict, paths: dict[str, Path], target_id: str,
) -> None:
    """An existing manifest may be reused only if it still describes exactly
    the contract and source we just resolved."""
    compared_keys = (
        "experiment_id",
        "source_experiment_id",
        "source_kind",
        "source_step8a_parquet_sha256",
        "source_physical_burned_count",
        "unique_historical_excluded_count",
        "mask_definition",
    )
    drifted = {
        key: (stored.get(key), freshly_resolved.get(key))
        for key in compared_keys
        if stored.get(key) != freshly_resolved.get(key)
    }
    if drifted:
        raise HistoricalBurnExclusionError(
            f"'{target_id}': the existing historical burn exclusion manifest "
            f"({paths['metadata_path']}) no longer matches the resolved "
            f"contract/source -- {drifted} (stored, resolved). Re-run with "
            "force to regenerate deliberately."
        )
    stored_ids = read_historical_burn_exclusion_manifest(
        paths["parquet_path"], experiment_id=target_id,
    )
    if len(stored_ids) != freshly_resolved["unique_historical_excluded_count"]:
        raise HistoricalBurnExclusionError(
            f"'{target_id}': the existing historical burn exclusion parquet "
            f"holds {len(stored_ids)} cell_id values but its metadata declares "
            f"{freshly_resolved['unique_historical_excluded_count']}."
        )


# =============================================================================
# Manifest reading (gate + Step8A share this ONE reader)
# =============================================================================
def read_historical_burn_exclusion_manifest(
    manifest_path: Path, experiment_id: str | None = None,
) -> frozenset[str]:
    """Read the canonical historical-burn exclusion manifest and return the
    excluded ``cell_id`` set.

    Deliberately a SEPARATE reader from Step8A's pre-label manifest reader:
    the two manifests have different schemas and different provenance
    requirements, and weakening either by merging them would remove checks.

    Fails closed on: missing manifest, unreadable parquet, missing/duplicate/
    null cell_id, missing sidecar metadata, and an experiment_id (target or
    source) that disagrees with the caller's context or the manifest rows.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise HistoricalBurnExclusionError(
            "Historical burn exclusion is enabled but its canonical manifest is "
            f"missing ({manifest_path}). Build the manifest before the gate / "
            "Step8A; proceeding would silently readmit historically burned "
            "cells into the analysis universe."
        )
    try:
        manifest_df = pd.read_parquet(manifest_path)
    except Exception as exc:  # noqa: BLE001
        raise HistoricalBurnExclusionError(
            f"Historical burn exclusion manifest ({manifest_path}) could not be "
            f"read: {type(exc).__name__}: {exc}."
        ) from exc

    missing = [c for c in ("cell_id", "exclusion_reason") if c not in manifest_df.columns]
    if missing:
        raise HistoricalBurnExclusionError(
            f"Historical burn exclusion manifest ({manifest_path}) is missing "
            f"required column(s): {missing}."
        )
    if manifest_df["cell_id"].isna().any():
        raise HistoricalBurnExclusionError(
            f"Historical burn exclusion manifest ({manifest_path}) contains "
            "null cell_id values."
        )
    if not manifest_df["cell_id"].is_unique:
        dupes = sorted(
            set(manifest_df.loc[manifest_df["cell_id"].duplicated(), "cell_id"].astype(str))
        )
        raise HistoricalBurnExclusionError(
            f"Historical burn exclusion manifest ({manifest_path}) contains "
            f"duplicate cell_id values: {dupes[:20]}."
        )
    bad_reasons = sorted(
        set(manifest_df.loc[manifest_df["exclusion_reason"] != EXCLUSION_REASON,
                            "exclusion_reason"].astype(str))
    )
    if bad_reasons:
        raise HistoricalBurnExclusionError(
            f"Historical burn exclusion manifest ({manifest_path}) contains "
            f"rows whose exclusion_reason is not {EXCLUSION_REASON!r}: "
            f"{bad_reasons[:20]}."
        )

    if experiment_id is not None:
        metadata_path = (
            manifest_path.parent / HISTORICAL_BURN_EXCLUSION_MANIFEST_METADATA
        )
        if not metadata_path.exists():
            raise HistoricalBurnExclusionError(
                "Historical burn exclusion manifest provenance is incomplete: "
                f"the sidecar metadata file is missing ({metadata_path})."
            )
        metadata = _read_manifest_metadata(metadata_path, experiment_id)
        if metadata.get("experiment_id") != experiment_id:
            raise HistoricalBurnExclusionError(
                "Historical burn exclusion manifest experiment_id MISMATCH: "
                f"manifest='{metadata.get('experiment_id')}', "
                f"context='{experiment_id}'. The wrong experiment's exclusion "
                "manifest is about to be applied."
            )
        if "experiment_id" in manifest_df.columns and len(manifest_df):
            row_ids = set(manifest_df["experiment_id"].astype(str))
            if row_ids != {experiment_id}:
                raise HistoricalBurnExclusionError(
                    "Historical burn exclusion manifest rows carry "
                    f"experiment_id(s) {sorted(row_ids)}, expected only "
                    f"'{experiment_id}'."
                )
        declared = metadata.get("unique_historical_excluded_count")
        if declared is not None and int(declared) != int(manifest_df["cell_id"].nunique()):
            raise HistoricalBurnExclusionError(
                "Historical burn exclusion manifest row count "
                f"({manifest_df['cell_id'].nunique()}) disagrees with its "
                f"metadata's unique_historical_excluded_count ({declared})."
            )

    return frozenset(manifest_df["cell_id"].astype(str))


def describe_historical_burn_contract(exp: dict, output_dir: Path | None = None) -> dict | None:
    """Read-only resolution used by --dry-run: resolves the contract, the
    source path/SHA/count and the planned artifact paths WITHOUT creating,
    writing or modifying anything.

    Returns None when the experiment does not opt in.
    """
    contract = resolve_historical_burn_contract(exp)
    if contract is None:
        return None

    description = dict(contract)
    description["region_grid_compatibility"] = verify_region_grid_compatibility(
        contract, require_target_grid=False,
    )
    if output_dir is not None:
        description["planned_artifacts"] = {
            name: str(path) for name, path in historical_manifest_paths(output_dir).items()
        }
    source_path = Path(contract["source_step8a_parquet_path"])
    if source_path.is_file():
        _, source_provenance = load_source_physical_burned_cells(contract)
        description.update(source_provenance)
        description["source_available"] = True
    else:
        description["source_available"] = False
    return description
