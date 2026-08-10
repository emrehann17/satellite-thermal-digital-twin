"""
run_label_gate_only.py

Step0C: guvenli, deney-farkinda (experiment-aware) "gate-only" calistirici.

Modelleme (Step7A-Step8E) oncesinde gerekli MINIMUM label/gate zincirini
calistirir:
    [Kozan disi deneyler icin] Step6A gate-input hazirlama (referans grid +
        hizalanmis landcover, namespaced, TERMAL PREDICTOR DEGIL)
    -> [opsiyonel] raw MCD64A1 BurnDate export (Step6'nin canonical export'u)
    -> Step6B burned-landcover gate

Step1-Step8'in geri kalanini (Step2-Step5 termal on-isleme, Step7, Step8)
KESINLIKLE CALISTIRMAZ.

IKI FARKLI DAVRANIS MODU
-------------------------
kozan_2023:
    LEGACY davranis BIREBIR korunur. outputs/validation/labels/ altindaki
    paylasilan dosyalar kullanilir/yazilir (Step6/Step6B'nin varsayilan
    kesif mantigi). Hicbir namespaced yol hesaplanmaz.

kozan_2023 DISINDAKI her deney (or. manavgat_2021):
    TAMAMEN NAMESPACED calisir:
        outputs/experiments/<experiment_id>/gate_inputs/...     (Step6A)
        outputs/experiments/<experiment_id>/validation/labels/... (Step6/Step6B)
    Bu deneyler icin, her calistirmadan ONCE, hesaplanan TUM yollarin
    outputs/experiments/<experiment_id>/ altinda kaldigi VE legacy Kozan
    dizini (outputs/validation/labels/) ile CAKISMADIGI dogrulanir
    (bkz. _assert_paths_are_safely_namespaced). Herhangi bir ihlalde
    LabelGateRunnerError firlatilir -- HICBIR export/gate calismaz.

Bu script HICBIR ZAMAN, bir deney etiketi altinda (or. manavgat_2021)
SESSIZCE Kozan verisini calistirmaz veya Kozan'in paylasilan dosyalarina
yazmaz.

CLI:
    python scripts/run_label_gate_only.py --experiment kozan_2023 --skip-export --force
    python scripts/run_label_gate_only.py --experiment kozan_2023 --export-labels --force
    python scripts/run_label_gate_only.py --experiment manavgat_2021 --dry-run
    python scripts/run_label_gate_only.py --experiment manavgat_2021 --export-labels --force

Flags:
    --dry-run         Hicbir sey CALISTIRMAZ; Step0 ozetini + TUM planlanan
                      yollari basar.
    --skip-export     Raw BurnDate export'unu ATLA (varsayilan davranis);
                      gate mevcut mcd64a1_raw.tif dosyasini kullanir.
    --export-labels   Gate'ten ONCE raw BurnDate export'unu (GEE) calistir.
                      Kozan-disi deneylerde ONCE Step6A gate-input hazirlama
                      da calisir (referans grid + landcover, eksikse veya
                      --force ise).
    --force           Gate (ve Kozan-disi deneylerde gate-input) ciktilari
                      zaten varsa uzerine yaz.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.io_utils import setup_logger
from core.regions import get_active_experiment, get_experiment_output_root

log, log_file = setup_logger("run_label_gate_only")

BASE_DIR = _PROJECT_ROOT
LEGACY_VALIDATION_LABEL_DIR = (BASE_DIR / "outputs" / "validation" / "labels").resolve()
_REGIONS_PY = BASE_DIR / "core" / "regions.py"

# Code/config files whose hashes go into the gate manifest / analysis_id.
_PROVENANCE_FILES = [
    "core/regions.py",
    "core/config.py",
    "src/step6b_burned_landcover_gate.py",
    "src/step6a_prepare_gate_inputs.py",
    "src/step6_validate_fire_relation.py",
    "src/step8a_prepare_500m_modeling_dataset.py",
    "src/historical_burn_exclusion.py",
]


def _protected_gate_report_path(experiment_id: str) -> Path:
    """Generic (non-hardcoded) gate-report path for any registered experiment_id."""
    if experiment_id == "kozan_2023":
        return LEGACY_VALIDATION_LABEL_DIR / "burned_landcover_gate.json"
    return (
        BASE_DIR / "outputs" / "experiments" / experiment_id
        / "validation" / "labels" / "burned_landcover_gate.json"
    )


def _resolve_protected_gate_reports(current_experiment_id: str) -> list[str]:
    """
    Registry-driven replacement for a hardcoded protected-report list (the
    previous static list named Kozan/Manavgat/Bejís but silently omitted
    mugla_2021 once it completed, and would omit every future AOI too).

    Walks every registered experiment (enabled or not) via the registry
    itself, keeps only those whose gate report file actually EXISTS on disk
    (never invents/regenerates a path for a report that hasn't been
    produced yet), EXCLUDES the experiment currently being processed (a
    report can never protect/attest itself -- self-referential provenance),
    and returns paths in deterministic (sorted) order. Only existence is
    checked here; hashing happens in the caller. Never writes to any of
    these paths.
    """
    from core.regions import list_experiments

    reports: list[str] = []
    for exp_id in sorted(list_experiments(include_disabled=True).keys()):
        if exp_id == current_experiment_id:
            continue
        path = _protected_gate_report_path(exp_id)
        if path.exists():
            reports.append(str(path.relative_to(BASE_DIR)))
    return reports


# =============================================================================
# Static (non-executing) core/regions.py source readers
# =============================================================================
# build_gate_manifest() needs deterministic AOI provenance (bounds/hash) for
# the manifest. Actually constructing an ee.Geometry (even ee.Geometry.BBox(),
# with no .getInfo() call) requires Earth Engine to be initialized/
# authenticated on first use -- verified directly: ee.Geometry.Point(...)
# alone raises "Earth Engine client library not initialized". The manifest
# must never trigger that merely to record AOI bounds, so bounds are read
# straight out of core/regions.py's source text (AST), never executed.
def _const_num(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _const_num(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _static_module_level_bbox_constants(tree: ast.Module) -> dict[str, tuple[float, float, float, float]]:
    """Module-level `NAME = (a, b, c, d)` numeric-tuple constants (e.g.
    MUGLA_AOI_BBOX, NORTH_EVIA_AOI_BBOX) used as `ee.Geometry.BBox(*NAME)`."""
    constants: dict[str, tuple[float, float, float, float]] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 4
        ):
            nums = [_const_num(e) for e in node.value.elts]
            if all(n is not None for n in nums):
                constants[node.targets[0].id] = tuple(nums)  # type: ignore[assignment]
    return constants


def _static_region_bbox(region_key: str) -> tuple[float, float, float, float] | None:
    """
    Resolves (west, south, east, north) bounds for `region_key` by statically
    parsing core/regions.py's build_regions() (AST) -- WITHOUT ever calling
    it. Only resolves region keys whose AOI is built directly from a literal
    `ee.Geometry.BBox(...)` call, following simple `name = other_name`
    aliasing and `*MODULE_CONSTANT` unpacking (true for every non-Kozan
    experiment currently registered). Returns None if it cannot be resolved
    this way (e.g. Kozan's buffered-point AOI -- never needed here since this
    is only called for non-Kozan experiments).
    """
    if not _REGIONS_PY.exists():
        return None
    tree = ast.parse(_REGIONS_PY.read_text(encoding="utf-8"))
    module_constants = _static_module_level_bbox_constants(tree)

    build_regions_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_regions":
            build_regions_fn = node
            break
    if build_regions_fn is None:
        return None

    local_assigns: dict[str, ast.expr] = {}
    return_dict: ast.Dict | None = None
    for node in ast.walk(build_regions_fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            local_assigns[node.targets[0].id] = node.value
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return_dict = node.value
    if return_dict is None:
        return None

    target_var = None
    for key_node, val_node in zip(return_dict.keys, return_dict.values):
        if isinstance(key_node, ast.Constant) and key_node.value == region_key and isinstance(val_node, ast.Name):
            target_var = val_node.id
            break
    if target_var is None:
        return None

    def _resolve(name: str, depth: int = 0) -> tuple[float, float, float, float] | None:
        if depth > 5 or name not in local_assigns:
            return None
        expr = local_assigns[name]
        if isinstance(expr, ast.Name):
            return _resolve(expr.id, depth + 1)
        if isinstance(expr, ast.Call):
            func = expr.func
            is_bbox_call = (
                isinstance(func, ast.Attribute) and func.attr == "BBox"
                and isinstance(func.value, ast.Attribute) and func.value.attr == "Geometry"
            )
            if not is_bbox_call:
                return None
            args = expr.args
            if len(args) == 1 and isinstance(args[0], ast.Starred) and isinstance(args[0].value, ast.Name):
                const = module_constants.get(args[0].value.id)
                return tuple(const) if const else None
            if len(args) == 4:
                nums = [_const_num(a) for a in args]
                if all(n is not None for n in nums):
                    return tuple(nums)  # type: ignore[return-value]
            return None
        return None

    return _resolve(target_var)


def _static_region_aoi_provenance(region_key: str) -> dict | None:
    """Deterministic AOI provenance for the gate manifest, built ENTIRELY
    from static source-text bounds (see _static_region_bbox) -- no Earth
    Engine call, no network access. Returns None if the region_key's AOI
    cannot be resolved this way."""
    bounds = _static_region_bbox(region_key)
    if bounds is None:
        return None
    w, s, e, n = bounds
    geometry_hash = hashlib.sha256(
        json.dumps(
            {"kind": "BBox", "crs": "EPSG:4326", "bounds": [w, s, e, n]},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "region_key": region_key,
        "kind": "bbox",
        "geometry_type": "Polygon",
        "crs": "EPSG:4326",
        "order": "west,south,east,north",
        "bounds": [w, s, e, n],
        "geodesic": False,
        "geometry_hash": geometry_hash,
        "source": "static AST read of core/regions.py build_regions() (no Earth Engine call)",
    }


def _registry_aoi_derivation(exp: dict) -> dict | None:
    """Return registry-declared AOI derivation as a JSON-safe deep copy.

    Experiments without an ``aoi_provenance`` declaration gain no manifest
    field. An explicitly declared provenance must be a non-empty,
    JSON-serializable dictionary; invalid declarations fail closed.
    """
    if "aoi_provenance" not in exp:
        return None

    provenance = exp["aoi_provenance"]

    if not isinstance(provenance, dict) or not provenance:
        raise ValueError(
            f"aoi_provenance for experiment "
            f"'{exp.get('experiment_id', '<unknown>')}' "
            "must be a non-empty dict"
        )

    try:
        return json.loads(
            json.dumps(provenance, ensure_ascii=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"aoi_provenance for experiment "
            f"'{exp.get('experiment_id', '<unknown>')}' "
            f"is not JSON-serializable: {exc}"
        ) from exc
    

def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(BASE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _package_versions() -> dict:
    versions = {}
    for mod in ("numpy", "rasterio", "pandas", "ee"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[mod] = "not_installed"
    return versions


def declares_primary_population_rule(exp: dict) -> bool:
    """True only when the registry record declares the COMPLETE per-experiment
    primary-population sample-size rule (population AND threshold).

    A half-declaration is deliberately NOT treated as "declared" here: the
    gate's own `evaluate_primary_population_sample_size` fails closed on it,
    and this predicate must not quietly pre-empt that error by writing a
    half-filled manifest block.
    """
    return (
        exp.get("primary_population") is not None
        and exp.get("min_primary_population_burned") is not None
    )


def _scientific_optional_contracts(exp: dict, historical_description: dict | None) -> dict:
    """The OPT-IN parts of the `scientific` payload, omitted entirely when the
    experiment does not declare them.

    Backward compatibility is the whole point: `scientific` is hashed into
    analysis_id, so an experiment that opts into nothing must produce the
    byte-identical pre-patch payload. Every block below is therefore added
    only on an explicit registry declaration -- never as a null/`not_applied`
    placeholder.
    """
    optional: dict = {}
    if historical_description:
        optional["historical_burn_exclusion"] = _scientific_historical_exclusion(
            historical_description
        )
    if declares_primary_population_rule(exp):
        optional["primary_population_sample_size_rule"] = (
            _scientific_primary_population_rule(exp)
        )
    # Event-relative framing metadata, when the registry declares it. Carried
    # verbatim so the scientific claim is part of the analysis_id.
    for key in ("transfer_framing", "event_anchor_date",
                "event_anchor_basis", "event_window_rule"):
        if key in exp:
            optional[key] = exp[key]
    return optional


def _scientific_historical_exclusion(historical_description: dict) -> dict:
    """Historical-exclusion block of the manifest's `scientific` payload.

    Only ever built for an experiment that opted in (see
    _scientific_optional_contracts). Sitting under "scientific" makes the
    source experiment, source kind, mask definition, source SHA-256 and source
    physical-burned count part of the analysis_id by construction:
    regenerating the source dataset (or pointing at a different source)
    changes the analysis_id.
    """
    return {
        "applied": True,
        "exclude_historical_burns": True,
        "source_experiment_id": historical_description.get("source_experiment_id"),
        "source_region_key": historical_description.get("source_region_key"),
        "source_kind": historical_description.get("source_kind"),
        "mask_definition": historical_description.get("mask_definition"),
        "exclusion_reason": historical_description.get("exclusion_reason"),
        "source_step8a_parquet_path": historical_description.get("source_step8a_parquet_path"),
        "source_step8a_parquet_sha256": historical_description.get("source_step8a_parquet_sha256"),
        "source_row_count": historical_description.get("source_row_count"),
        "source_physical_burned_count": historical_description.get("source_physical_burned_count"),
        "source_expected_physical_burned_count": historical_description.get(
            "source_expected_physical_burned_count"
        ),
        "region_grid_compatibility": historical_description.get("region_grid_compatibility"),
    }


def _scientific_primary_population_rule(exp: dict) -> dict:
    """Primary-population sample-size rule block of `scientific`.

    Only ever built for an experiment that declares the COMPLETE rule (see
    declares_primary_population_rule).

    SEPARATE from the burned-landcover gate's generic total-burned
    feasibility threshold (min_positives), which is recorded alongside it and
    is NOT modified. Being under "scientific" makes the per-experiment
    threshold part of the analysis_id.
    """
    from core.config import STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES

    population = exp["primary_population"]
    minimum = exp["min_primary_population_burned"]
    return {
        "applied": True,
        "population": population,
        "min_primary_population_burned": minimum,
        "semantics": (
            "post-gate sample-size STOP evaluated after all exclusions and "
            "landcover classification, on the primary population's burned "
            "count only"
        ),
        "distinct_from_composition_gate_min_positives": (
            STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES
        ),
        "distinct_from_composition_gate_min_positives_semantics": (
            "generic TOTAL-burned feasibility threshold applied to burned_count "
            "across every landcover; unchanged by this rule"
        ),
    }


def build_gate_manifest(
    experiment_id: str,
    exp: dict,
    paths: dict,
    gate_result: dict,
    historical_description: dict | None = None,
) -> dict:
    """
    Builds a provenance manifest + a content-addressed analysis_id for any
    experiment-aware gate run. analysis_id is a sha256 over the scientific
    configuration, AOI, exclusion rule, relevant code/config file hashes,
    package versions and git SHA -- so a given gate result is reproducibly
    tied to exactly what produced it.
    """
    aoi = _static_region_aoi_provenance(exp.get("region_key", ""))
    # scientific.aoi.source keeps describing the STATIC AST resolution above;
    # the registry-declared derivation chain (how the bbox numbers came to be)
    # is attached alongside it as scientific.aoi.derivation. It sits under
    # "scientific", so it is part of the analysis_id payload by construction.
    derivation = _registry_aoi_derivation(exp)
    if derivation is not None:
        if aoi is None:
            # Fail closed: never write a manifest that silently drops a
            # derivation chain the registry explicitly declared.
            raise ValueError(
                f"experiment '{experiment_id}' declares aoi_provenance but its "
                f"region_key '{exp.get('region_key')}' could not be resolved "
                "statically from core/regions.py; refusing to emit a manifest "
                "without the declared AOI derivation."
            )
        aoi["derivation"] = derivation

    file_hashes = {rel: _sha256_file(BASE_DIR / rel) for rel in _PROVENANCE_FILES}
    protected = {
        rel: _sha256_file(BASE_DIR / rel)
        for rel in _resolve_protected_gate_reports(experiment_id)
    }

    scientific = {
        "experiment_id": experiment_id,
        "region_key": exp.get("region_key"),
        "aoi": aoi,
        "predictor_window": [exp.get("predictor_start_date"), exp.get("predictor_end_date")],
        "label_window": [exp.get("label_start_date"), exp.get("label_end_date")],
        "baseline_years": exp.get("baseline_years"),
        "primary_population": "burnable_tree_shrub_grass",
        "modeling_grid_m": 500,
        "exclude_pre_label_burns": bool(exp.get("exclude_pre_label_burns", False)),
        "pre_label_burn_window": exp.get("pre_label_burn_window"),
        "pre_label_burn_exclusion_rule": (
            f"valid nonzero BurnDate calendar date < label_start "
            f"({exp.get('label_start_date')})"
            if exp.get("exclude_pre_label_burns") else "not_applied"
        ),
        # --- ADDITIVE-ONLY EXTENSIONS ------------------------------------
        # Everything below is added CONDITIONALLY, only when the registry
        # record actually opts into the corresponding contract. An experiment
        # that declares none of them keeps the EXACT pre-patch `scientific`
        # key set -- and therefore its pre-patch analysis_id, since the key
        # set is hashed. A key whose mere presence carried an "not applied"
        # value would silently change every existing experiment's analysis_id.
        #
        #   historical_burn_exclusion            exclude_historical_burns=True
        #   primary_population_sample_size_rule  the COMPLETE per-experiment
        #                                        population + threshold pair
        #   transfer_framing / event_anchor_*    event-relative framing
        **_scientific_optional_contracts(exp, historical_description),
    }
    id_payload = {
        "scientific": scientific,
        "code_file_hashes": file_hashes,
        "package_versions": _package_versions(),
        "git_sha": _git_sha(),
    }
    analysis_id = hashlib.sha256(
        json.dumps(id_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    gate = gate_result.get("gate_result", gate_result) if isinstance(gate_result, dict) else {}

    # --- Pre-label exclusion provenance (generic; empty/None when the
    # experiment does not enable exclude_pre_label_burns) ---
    manifest_result = gate.get("manifest_result") or {}
    pre_label_raw_path = paths.get("pre_label_raw_path")
    exclusion_artifact_path = manifest_result.get("parquet_path")
    pre_label_provenance = {
        "pre_label_burn_excluded_count": manifest_result.get("excluded_cell_count", 0),
        "pre_label_raster_path": str(pre_label_raw_path) if pre_label_raw_path else None,
        "pre_label_raster_sha256": (
            _sha256_file(Path(pre_label_raw_path)) if pre_label_raw_path else None
        ),
        "exclusion_manifest_parquet_path": exclusion_artifact_path,
        "exclusion_manifest_parquet_sha256": (
            _sha256_file(Path(exclusion_artifact_path)) if exclusion_artifact_path else None
        ),
    }

    # --- Historical exclusion provenance (SEPARATE from pre_label_provenance,
    # which is retained above unchanged). Like the `scientific` blocks, this
    # top-level key is OMITTED ENTIRELY for experiments that did not opt in,
    # so their manifest structure stays byte-compatible with the pre-patch
    # schema rather than gaining an all-zero placeholder. ---
    optional_provenance: dict = {}
    if historical_description:
        historical_manifest_result = gate.get("historical_burn_exclusion_manifest") or {}
        historical_artifact_path = historical_manifest_result.get("parquet_path")
        optional_provenance["historical_burn_provenance"] = {
            "applied": True,
            "historical_burn_excluded_count": gate.get("historical_burn_excluded_count", 0),
            "pre_label_historical_overlap_count": gate.get(
                "pre_label_historical_overlap_count", 0
            ),
            "total_unique_excluded_count": gate.get("total_unique_excluded_count", 0),
            "source_experiment_id": historical_description.get("source_experiment_id"),
            "source_step8a_parquet_path": historical_description.get(
                "source_step8a_parquet_path"
            ),
            "source_step8a_parquet_sha256": historical_description.get(
                "source_step8a_parquet_sha256"
            ),
            "source_physical_burned_count": historical_description.get(
                "source_physical_burned_count"
            ),
            "exclusion_manifest_parquet_path": historical_artifact_path,
            "exclusion_manifest_parquet_sha256": (
                _sha256_file(Path(historical_artifact_path))
                if historical_artifact_path else None
            ),
        }
    # Same rule for the post-gate sample-size outcome: present only when the
    # gate actually evaluated the rule (i.e. the experiment declared it).
    primary_population_gate = gate.get("primary_population_sample_size_gate")
    if primary_population_gate is not None:
        optional_provenance["primary_population_sample_size_gate"] = primary_population_gate

    return {
        "manifest_kind": "experiment_gate_provenance",
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "downstream_authorized": False,
        "downstream_status": "blocked_pending_advisor_review",
        "scientific": scientific,
        "pre_label_provenance": pre_label_provenance,
        **optional_provenance,
        "code_file_hashes": file_hashes,
        "protected_gate_report_hashes": protected,
        "package_versions": _package_versions(),
        "git_sha": _git_sha(),
        "gate_summary": {
            "decision": gate.get("decision"),
            "burned_count": gate.get("burned_count"),
            "json_path": gate.get("json_path"),
        },
    }


class LabelGateRunnerError(SystemExit):
    """Fail-fast error for this runner (diger step'lerle ayni konvansiyon)."""


def _print_step0_summary(exp: dict, output_root: Path) -> None:
    log.info("[Step0] Active experiment: %s", exp["experiment_id"])
    log.info("[Step0] Display name: %s", exp["display_name"])
    log.info("[Step0] Region: %s", exp["region_key"])
    log.info("[Step0] Role: %s", exp["role"])
    log.info(
        "[Step0] Predictor window: %s -> %s",
        exp["predictor_start_date"], exp["predictor_end_date"],
    )
    log.info(
        "[Step0] Label window: %s -> %s",
        exp["label_start_date"], exp["label_end_date"],
    )
    log.info("[Step0] Baseline years: %s", ", ".join(str(y) for y in exp["baseline_years"]) or "(yok)")
    log.info("[Step0] Output root: %s", output_root)


def _legacy_kozan_paths() -> dict:
    return {
        "gate_inputs_dir": None,
        "reference_path": (BASE_DIR / "outputs" / "step5" / "current_period_median_celsius.tif"),
        "landcover_source_path": (BASE_DIR / "data" / "landcover" / "landcover_esa_worldcover_v200.tif"),
        "landcover_aligned_path": (
            BASE_DIR / "data" / "landcover" / "landcover_esa_worldcover_v200_aligned_to_landsat.tif"
        ),
        "metadata_path": None,
        "raw_path": LEGACY_VALIDATION_LABEL_DIR / "mcd64a1_raw.tif",
        "binary_path": LEGACY_VALIDATION_LABEL_DIR / "mcd64a1_burned.tif",
        "gate_output_dir": LEGACY_VALIDATION_LABEL_DIR,
    }


def _namespaced_paths(experiment_id: str) -> dict:
    """Kozan-disi deneyler icin TUM planlanan yollari (namespaced) hesaplar.

    Hicbir sey OLUSTURMAZ/YAZMAZ -- yalnizca yol hesaplar (dry-run ve
    safety-check tarafindan da kullanilir).
    """
    from src.historical_burn_exclusion import (
        HISTORICAL_BURN_EXCLUSION_MANIFEST_CSV,
        HISTORICAL_BURN_EXCLUSION_MANIFEST_METADATA,
        HISTORICAL_BURN_EXCLUSION_MANIFEST_PARQUET,
    )
    from src.step6a_prepare_gate_inputs import get_gate_input_paths

    gate_inputs = get_gate_input_paths(experiment_id)
    label_dir = get_experiment_output_root(experiment_id) / "validation" / "labels"
    return {
        "gate_inputs_dir": gate_inputs["gate_inputs_dir"],
        "reference_path": gate_inputs["reference_path"],
        "landcover_source_path": gate_inputs["landcover_source_path"],
        "landcover_aligned_path": gate_inputs["landcover_aligned_path"],
        "metadata_path": gate_inputs["metadata_path"],
        "raw_path": label_dir / "mcd64a1_raw.tif",
        "binary_path": label_dir / "mcd64a1_burned.tif",
        "pre_label_raw_path": label_dir / "mcd64a1_prelabel_raw.tif",
        # Historical burn exclusion artifacts live in the SAME gate labels
        # directory but in their OWN files -- never merged into
        # pre_label_excluded_cells.*.
        "historical_excluded_parquet_path": (
            label_dir / HISTORICAL_BURN_EXCLUSION_MANIFEST_PARQUET
        ),
        "historical_excluded_csv_path": (
            label_dir / HISTORICAL_BURN_EXCLUSION_MANIFEST_CSV
        ),
        "historical_excluded_metadata_path": (
            label_dir / HISTORICAL_BURN_EXCLUSION_MANIFEST_METADATA
        ),
        "manifest_path": label_dir / f"{experiment_id}_gate_manifest.json",
        "gate_output_dir": label_dir,
    }


def _assert_paths_are_safely_namespaced(experiment_id: str, paths: dict) -> None:
    """
    GUVENLIK KONTROLU (Kozan-disi deneyler icin ZORUNLU):
        1) Hicbir hesaplanan yol legacy outputs/validation/labels/ altina
           DUSMEMELIDIR (Kozan'in paylasilan dosyalarinin uzerine ASLA
           yazilmaz/okunmaz).
        2) Yazilabilir TUM yollar outputs/experiments/<experiment_id>/
           altinda OLMALIDIR.
    Ihlal varsa LabelGateRunnerError firlatir; hicbir export/gate calismaz.
    """
    experiments_root = (BASE_DIR / "outputs" / "experiments" / experiment_id).resolve()

    for name, p in paths.items():
        if p is None:
            continue
        resolved = Path(p).resolve()

        if resolved == LEGACY_VALIDATION_LABEL_DIR or LEGACY_VALIDATION_LABEL_DIR in resolved.parents:
            raise LabelGateRunnerError(
                f"GUVENLIK IHLALI: '{experiment_id}' deneyi icin hesaplanan '{name}' "
                f"yolu ({resolved}) Kozan'in legacy paylasilan dizinine "
                f"(outputs/validation/labels/) dusuyor. Bu deney bu dizine ASLA "
                "yazamaz/okuyamaz. Islem DURDURULDU."
            )
        if resolved != experiments_root and experiments_root not in resolved.parents:
            raise LabelGateRunnerError(
                f"GUVENLIK IHLALI: '{experiment_id}' deneyi icin hesaplanan '{name}' "
                f"yolu ({resolved}) outputs/experiments/{experiment_id}/ disinda. "
                "Islem DURDURULDU."
            )


def _resolve_historical_contract_or_none(exp: dict, is_kozan: bool, gate_output_dir):
    """Registry-driven historical-exclusion resolution (read-only).

    Returns the fully-resolved description (contract + source path/SHA/count +
    region/grid compatibility + planned artifact paths) or None when the
    experiment does not declare the contract. Nothing is created or written.
    """
    if is_kozan or not exp.get("exclude_historical_burns", False):
        return None
    from src.historical_burn_exclusion import (
        HistoricalBurnExclusionError,
        describe_historical_burn_contract,
    )

    try:
        return describe_historical_burn_contract(exp, output_dir=Path(gate_output_dir))
    except HistoricalBurnExclusionError as exc:
        raise LabelGateRunnerError(str(exc)) from exc


def _log_historical_contract(description: dict | None, exp: dict) -> None:
    """Print the resolved historical-exclusion contract and the SEPARATE
    primary-population sample-size rule. Read-only; creates nothing."""
    if description is None:
        log.info(
            "  historical burn exclusion: NOT declared for this experiment "
            "(exclude_historical_burns absent/False)."
        )
    else:
        log.info("  historical burn exclusion: DECLARED")
        log.info("    source experiment : %s", description["source_experiment_id"])
        log.info("    source kind       : %s", description["source_kind"])
        log.info("    mask definition   : %s", description["mask_definition"])
        log.info("    source parquet    : %s", description["source_step8a_parquet_path"])
        log.info("    source available  : %s", description.get("source_available"))
        log.info("    source sha256     : %s", description.get("source_step8a_parquet_sha256"))
        log.info("    source rows       : %s", description.get("source_row_count"))
        log.info(
            "    source physical burned cells: %s (frozen expectation: %s)",
            description.get("source_physical_burned_count"),
            description.get("source_expected_physical_burned_count"),
        )
        compat = description.get("region_grid_compatibility", {})
        log.info(
            "    region/grid identity: region_key=%s matches_source=%s "
            "grid_verified=%s (%s)",
            compat.get("region_key"), compat.get("region_key_matches_source"),
            compat.get("grid_identity_verified"), compat.get("grid_identity_note"),
        )
        for name, planned in (description.get("planned_artifacts") or {}).items():
            log.info("    planned %s: %s", name, planned)

    population = exp.get("primary_population")
    minimum = exp.get("min_primary_population_burned")
    if population is None and minimum is None:
        log.info(
            "  primary-population sample-size rule: NOT declared for this "
            "experiment (the generic composition gate min_positives is "
            "unaffected either way)."
        )
    else:
        from core.config import STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES

        log.info(
            "  primary-population sample-size rule: population=%s "
            "min_primary_population_burned=%s -- SEPARATE from the composition "
            "gate's generic total-burned min_positives=%s (unchanged).",
            population, minimum, STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES,
        )


def _log_planned_paths(paths: dict) -> None:
    log.info("  gate_inputs reference (30 m, termal predictor DEGIL): %s", paths["reference_path"])
    log.info("  gate_inputs landcover (kaynak): %s", paths["landcover_source_path"])
    log.info("  gate_inputs landcover (hizali): %s", paths["landcover_aligned_path"])
    if paths.get("metadata_path"):
        log.info("  gate_inputs metadata: %s", paths["metadata_path"])
    log.info("  raw BurnDate (DOY): %s", paths["raw_path"])
    log.info("  binary mask: %s", paths["binary_path"])
    if paths.get("pre_label_raw_path"):
        log.info("  pre-label BurnDate (leakage exclusion): %s", paths["pre_label_raw_path"])
    if paths.get("historical_excluded_parquet_path"):
        log.info("  historical exclusion manifest (parquet): %s",
                 paths["historical_excluded_parquet_path"])
        log.info("  historical exclusion manifest (csv):     %s",
                 paths["historical_excluded_csv_path"])
        log.info("  historical exclusion manifest (metadata): %s",
                 paths["historical_excluded_metadata_path"])
    if paths.get("manifest_path"):
        log.info("  gate manifest (analysis_id, downstream_authorized=false): %s", paths["manifest_path"])
    log.info("  gate ciktilari (json/md/csv): %s/burned_landcover_gate.{json,md,csv}", paths["gate_output_dir"])


def main(
    experiment_id: str = "kozan_2023",
    dry_run: bool = False,
    skip_export: bool = False,
    export_labels: bool = False,
    force: bool = False,
) -> dict:
    exp = get_active_experiment(experiment_id)
    output_root = get_experiment_output_root(experiment_id)
    _print_step0_summary(exp, output_root)

    is_kozan = (experiment_id == "kozan_2023")
    paths = _legacy_kozan_paths() if is_kozan else _namespaced_paths(experiment_id)

    if not is_kozan:
        _assert_paths_are_safely_namespaced(experiment_id, paths)
        if exp["label_start_date"] is None or exp["label_end_date"] is None:
            raise LabelGateRunnerError(
                f"'{experiment_id}' deneyinin label penceresi tanimsiz "
                "(placeholder/disabled deney olabilir). Gate-only calistirilamaz."
            )

    # Registry-driven historical-exclusion resolution. Read-only: it resolves
    # the contract, the source parquet path/SHA/count and region+grid identity
    # WITHOUT creating or writing anything, so --dry-run can print all of it.
    historical_description = _resolve_historical_contract_or_none(
        exp, is_kozan, paths["gate_output_dir"],
    )

    if dry_run:
        log.info("[dry-run] Planlanan yollar:")
        _log_planned_paths(paths)
        log.info("[dry-run] Cozulmus sozlesmeler:")
        _log_historical_contract(historical_description, exp)
        log.info("[dry-run] Hicbir export/gate/manifest CALISTIRILMADI veya YAZILMADI.")
        return {
            "experiment_id": experiment_id,
            "ran": False,
            "reason": "dry_run",
            "planned_paths": {
                k: (str(v) if v is not None else None) for k, v in paths.items()
            },
            "historical_burn_exclusion": historical_description,
            "primary_population_sample_size_rule": {
                "population": exp.get("primary_population"),
                "min_primary_population_burned": exp.get("min_primary_population_burned"),
            },
        }

    if skip_export and export_labels:
        raise LabelGateRunnerError(
            "--skip-export ve --export-labels birlikte verilemez (celiskili)."
        )
    do_export = export_labels and not skip_export

    gate_inputs_result = None
    if not is_kozan:
        # Kozan-disi deneyler icin Step6B'nin ihtiyac duydugu referans grid +
        # hizalanmis landcover HENUZ yoktur (Step5/Step4 calismadi) --
        # bunlari once Step6A ile (namespaced, termal predictor OLMADAN)
        # hazirlamamiz gerekir.
        log.info("Step6A gate-input hazirlama calistiriliyor (namespaced, termal predictor DEGIL)...")
        try:
            from src.step6a_prepare_gate_inputs import prepare_gate_inputs
        except Exception as exc:  # noqa: BLE001
            raise LabelGateRunnerError(
                f"src.step6a_prepare_gate_inputs import edilemedi: {type(exc).__name__}: {exc}"
            ) from exc
        gate_inputs_result = prepare_gate_inputs(experiment_id, force=force)
        log.info("Step6A tamamlandi: %s", gate_inputs_result)

    # Leakage-safe pre-label exclusion is opt-in per experiment (Muğla 2021).
    exclude_pre_label = (not is_kozan) and bool(exp.get("exclude_pre_label_burns", False))

    export_result = None
    pre_label_export_result = None
    if do_export:
        log.info("Raw MCD64A1 BurnDate export calistiriliyor (--export-labels)...")
        try:
            from src.step6_validate_fire_relation import export_raw_mcd64a1_labels
        except Exception as exc:  # noqa: BLE001
            raise LabelGateRunnerError(
                f"src.step6_validate_fire_relation import edilemedi: {type(exc).__name__}: {exc}"
            ) from exc
        export_result = export_raw_mcd64a1_labels(
            also_binary=True,
            experiment_id=None if is_kozan else experiment_id,
        )
        log.info("Raw BurnDate export tamamlandi: %s", export_result["raw_path"])

        if exclude_pre_label:
            log.info(
                "Pre-label BurnDate export calistiriliyor (leakage-safe "
                "exclusion; pre_label_burn_window=%s)...",
                exp.get("pre_label_burn_window"),
            )
            try:
                from src.step6_validate_fire_relation import (
                    export_raw_mcd64a1_prelabel_labels,
                )
            except Exception as exc:  # noqa: BLE001
                raise LabelGateRunnerError(
                    f"export_raw_mcd64a1_prelabel_labels import edilemedi: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            pre_label_export_result = export_raw_mcd64a1_prelabel_labels(
                experiment_id=experiment_id,
                raw_out=paths["pre_label_raw_path"],
            )
            log.info(
                "Pre-label export tamamlandi: %s (positive pixels=%s)",
                pre_label_export_result["raw_path"],
                pre_label_export_result.get("pre_label_positive_pixel_count"),
            )
    else:
        log.info(
            "Raw BurnDate export ATLANDI (--skip-export ya da varsayilan). "
            "Gate mevcut %s dosyasini kullanacak (yoksa gate net bir hata "
            "verecektir).", paths["raw_path"],
        )

    # --- Historical burn exclusion manifest: BUILT AND VALIDATED BEFORE the
    # gate runs. The source experiment's Step8A artifact is opened read-only
    # and never rewritten; the manifest is written only inside THIS
    # experiment's gate labels directory (already namespace-checked above). ---
    historical_manifest_result = None
    exclude_historical = historical_description is not None
    if exclude_historical:
        log.info(
            "Historical burn exclusion manifest uretiliyor/dogrulaniyor "
            "(source=%s, kind=%s)...",
            historical_description["source_experiment_id"],
            historical_description["source_kind"],
        )
        _log_historical_contract(historical_description, exp)
        try:
            from src.historical_burn_exclusion import (
                HistoricalBurnExclusionError,
                build_historical_burn_exclusion_manifest,
            )

            historical_manifest_result = build_historical_burn_exclusion_manifest(
                exp, output_dir=Path(paths["gate_output_dir"]), force=force,
            )
        except HistoricalBurnExclusionError as exc:
            raise LabelGateRunnerError(str(exc)) from exc
        log.info(
            "Historical exclusion manifest hazir: %s (%d unique cell_id; "
            "created=%s).",
            historical_manifest_result["parquet_path"],
            historical_manifest_result["excluded_cell_count"],
            historical_manifest_result["created"],
        )

    log.info("Step6B burned-landcover gate calistiriliyor...")
    try:
        from src.step6b_burned_landcover_gate import main as run_gate
    except Exception as exc:  # noqa: BLE001
        raise LabelGateRunnerError(
            f"src.step6b_burned_landcover_gate import edilemedi: {type(exc).__name__}: {exc}"
        ) from exc

    if is_kozan:
        # Legacy davranis BIREBIR: hicbir explicit path verilmez, Step6B
        # kendi varsayilan (outputs/validation/labels + outputs/step5 +
        # data/landcover) kesif mantigini kullanir.
        gate_result = run_gate(force=force)
    else:
        gate_kwargs = dict(
            output_dir_arg=str(paths["gate_output_dir"]),
            force=force,
            label_raster_arg=str(paths["raw_path"]),
            reference_30m_arg=str(paths["reference_path"]),
            landcover_path_arg=str(paths["landcover_aligned_path"]),
            label_start=exp["label_start_date"],
            label_end=exp["label_end_date"],
        )
        if exclude_pre_label:
            pre_label_raw = Path(paths["pre_label_raw_path"])
            if not pre_label_raw.exists():
                raise LabelGateRunnerError(
                    f"'{experiment_id}' exclude_pre_label_burns=True ama pre-label "
                    f"BurnDate rasteri yok ({pre_label_raw}). Once --export-labels "
                    "ile calistirin (pre-label export'u da uretilir)."
                )
            # Optional per-experiment diagnostic sub-window (e.g. Muğla's
            # Bördübet fire check); generic registry field, absent (None)
            # for every experiment that doesn't declare one -- never a
            # hardcoded per-AOI literal here.
            diagnostic_window = exp.get("pre_label_diagnostic_window")
            gate_kwargs.update(
                exclude_pre_label_burns=True,
                pre_label_raster_arg=str(pre_label_raw),
                predictor_start=exp["predictor_start_date"],
                predictor_end=exp["predictor_end_date"],
                bordubet_check_window=(tuple(diagnostic_window) if diagnostic_window else None),
                experiment_id=experiment_id,
            )
        if exclude_historical:
            # SEPARATE kwargs from the pre-label ones above; the two exclusion
            # axes are never conflated.
            gate_kwargs.update(
                exclude_historical_burns=True,
                historical_exclusion_manifest_arg=str(
                    paths["historical_excluded_parquet_path"]
                ),
                historical_exclusion_config=historical_description,
                experiment_id=experiment_id,
            )
        # SEPARATE post-gate primary-population sample-size rule; generic
        # registry fields, absent (None) for every experiment that declares
        # none -- never a hardcoded per-AOI literal here.
        if exp.get("primary_population") is not None or exp.get("min_primary_population_burned") is not None:
            gate_kwargs.update(
                primary_population=exp.get("primary_population"),
                min_primary_population_burned=exp.get("min_primary_population_burned"),
            )
        gate_result = run_gate(**gate_kwargs)

    log.info(
        "TAMAMLANDI. Gate karari: %s (burned_count=%s). JSON: %s",
        gate_result["decision"], gate_result["burned_count"], gate_result["json_path"],
    )
    if exclude_historical:
        log.info(
            "Exclusion union: pre_label=%s historical=%s overlap=%s "
            "total_unique=%s -> analysis universe=%s",
            gate_result.get("pre_label_burn_excluded_count"),
            gate_result.get("historical_burn_excluded_count"),
            gate_result.get("pre_label_historical_overlap_count"),
            gate_result.get("total_unique_excluded_count"),
            gate_result.get("analysis_universe_cells_after_exclusions"),
        )

    # --- SEPARATE primary-population sample-size STOP (never auto-runs
    # anything downstream; downstream_authorized is False either way). ---
    primary_gate = gate_result.get("primary_population_sample_size_gate")
    primary_stop_state = None
    if primary_gate and primary_gate["decision"] == "stop":
        primary_stop_state = primary_gate["stop_state"]
        log.warning(
            "STOP (%s): %s burned_count=%s < min_burned_required=%s. Downstream "
            "is NOT authorized. This is a PRIMARY-POPULATION SAMPLE-SIZE stop "
            "and must NOT be reinterpreted as a natural/cropland composition "
            "gate failure (composition decision was: %s).",
            primary_stop_state, primary_gate["population"],
            primary_gate["burned_count"], primary_gate["min_burned_required"],
            gate_result["decision"],
        )

    # --- Provenance manifest (analysis_id + downstream_authorized=false) ---
    manifest_result = None
    if not is_kozan:
        manifest_path = Path(paths["manifest_path"])
        if manifest_path.exists() and not force:
            log.warning(
                "Gate manifest zaten var (%s); --force verilmedigi icin "
                "UZERINE YAZILMADI (mevcut preregistration/manifest korunur).",
                manifest_path,
            )
        else:
            manifest = build_gate_manifest(
                experiment_id, exp, paths, {"gate_result": gate_result},
                historical_description=historical_description,
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            log.info(
                "Gate manifest yazildi: %s (analysis_id=%s, downstream_authorized=%s)",
                manifest_path, manifest["analysis_id"], manifest["downstream_authorized"],
            )
            manifest_result = {"manifest_path": str(manifest_path),
                               "analysis_id": manifest["analysis_id"]}

    return {
        "experiment_id": experiment_id,
        "ran": True,
        "gate_inputs_result": gate_inputs_result,
        "export_result": export_result,
        "pre_label_export_result": pre_label_export_result,
        "gate_result": gate_result,
        "manifest_result": manifest_result,
        "historical_burn_exclusion": historical_description,
        "historical_manifest_result": historical_manifest_result,
        "primary_population_sample_size_gate": primary_gate,
        "primary_population_stop_state": primary_stop_state,
        "downstream_authorized": False,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0C: modelleme oncesi minimum label/gate zincirini "
        "(Kozan-disi deneyler icin namespaced Step6A gate-input hazirlama + "
        "opsiyonel raw BurnDate export + Step6B burned-landcover gate) "
        "calistirir. Step1-Step5/Step7/Step8'in geri kalanini CALISTIRMAZ. "
        "kozan_2023 icin legacy davranis BIREBIR korunur; diger deneyler "
        "TAMAMEN namespaced calisir ve Kozan'in paylasilan dosyalarina ASLA "
        "yazmaz/okumaz."
    )
    parser.add_argument("--experiment", type=str, default="kozan_2023")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hicbir sey calistirma; Step0 ozetini + planlanan TUM yollari bas.",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Raw BurnDate export'unu atla (varsayilan davranis).",
    )
    parser.add_argument(
        "--export-labels", action="store_true",
        help="Gate'ten once raw BurnDate export'unu (GEE) calistir "
        "(Kozan-disi deneylerde once Step6A gate-input hazirlama da calisir).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Gate (ve Kozan-disi deneylerde gate-input) ciktilari zaten "
        "varsa uzerine yaz.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        experiment_id=args.experiment,
        dry_run=args.dry_run,
        skip_export=args.skip_export,
        export_labels=args.export_labels,
        force=args.force,
    )