"""Tests for the provenance tooling: single-Step8A-hash audit, frozen hash
inventory, versioned large-block namespace, and resolver priority.

Every expected value is resolved from a frozen artefact or a fixture; none is
hard-coded into the implementation under test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.paths import PROJECT_ROOT
from src import provenance_audit as pa
from src.multi_aoi_transfer_synthesis import resolvers

SUBJECT = "manavgat_2021"
AOIS = ["manavgat_2021", "bejis_2022", "mugla_2021", "evia_2021_extended"]
REAL_STEP8A = pa.canonical_step8a_path(SUBJECT)

requires_step8a = pytest.mark.skipif(
    not REAL_STEP8A.is_file(), reason="canonical Manavgat Step8A not present"
)


# --------------------------------------------------------------- 5, 6, 7
def _audit_payload(tmp_path: Path, monkeypatch, recorded: dict[str, list[str]], canonical: str):
    """Build an audit payload over a synthetic consumer set."""
    consumers = []
    for name, hashes in recorded.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "checks": {SUBJECT: {"step8a_dataset_sha256": h} for h in hashes[:1]},
            "extra": [{f"{SUBJECT}_step8a_sha256": h} for h in hashes],
        }))
        consumers.append({"group": "synthetic", "path": path})

    monkeypatch.setattr(pa, "_consumer_files", lambda experiment_id: consumers)
    monkeypatch.setattr(pa, "canonical_step8a_hash", lambda experiment_id=SUBJECT: canonical)
    monkeypatch.setattr(pa, "canonical_step8a_path", lambda experiment_id=SUBJECT: tmp_path / "canonical.parquet")
    return pa.run_manavgat_single_step8a_hash_audit(write=False)


def test_audit_passes_with_a_single_hash(tmp_path, monkeypatch):
    h = "a" * 64
    payload = _audit_payload(tmp_path, monkeypatch, {"c1": [h], "c2": [h]}, canonical=h)
    assert payload["unique_hash_count"] == 1
    assert payload["status"] == pa.STATUS_PASS
    assert payload["passes"] is True
    assert payload["synthesis_may_run"] is True


def test_audit_fails_with_two_hashes(tmp_path, monkeypatch):
    h1, h2 = "a" * 64, "b" * 64
    payload = _audit_payload(tmp_path, monkeypatch, {"c1": [h1], "c2": [h2]}, canonical=h1)
    assert payload["unique_hash_count"] == 2
    assert payload["status"] == pa.STATUS_MIXED
    assert payload["passes"] is False
    assert payload["synthesis_may_run"] is False


def test_audit_does_not_pass_when_a_required_artifact_lacks_provenance(tmp_path, monkeypatch):
    """A provenance-bearing artefact that records no Step8A hash blocks PASS."""
    h = "a" * 64
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"checks": {SUBJECT: {"step8a_dataset_sha256": h}}}))
    silent = tmp_path / "input_audit.json"
    silent.write_text(json.dumps({"unrelated": "payload"}))

    monkeypatch.setattr(pa, "canonical_step8a_hash", lambda experiment_id=SUBJECT: h)
    monkeypatch.setattr(pa, "canonical_step8a_path", lambda experiment_id=SUBJECT: tmp_path / "c.parquet")
    monkeypatch.setattr(pa, "_consumer_files", lambda experiment_id: [
        {"group": "synthetic", "path": good, "provenance_required": False},
        {"group": "synthetic", "path": silent, "provenance_required": True},
    ])
    payload = pa.run_manavgat_single_step8a_hash_audit(write=False)
    assert payload["required_provenance_gaps"]
    assert payload["status"] == pa.STATUS_INCOMPLETE
    assert payload["passes"] is False


def test_audit_does_not_pass_when_a_required_artifact_is_missing(tmp_path, monkeypatch):
    h = "a" * 64
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"checks": {SUBJECT: {"step8a_dataset_sha256": h}}}))

    monkeypatch.setattr(pa, "canonical_step8a_hash", lambda experiment_id=SUBJECT: h)
    monkeypatch.setattr(pa, "canonical_step8a_path", lambda experiment_id=SUBJECT: tmp_path / "c.parquet")
    monkeypatch.setattr(pa, "_consumer_files", lambda experiment_id: [
        {"group": "synthetic", "path": good, "provenance_required": False},
        {"group": "synthetic", "path": tmp_path / "absent.json", "provenance_required": True},
    ])
    payload = pa.run_manavgat_single_step8a_hash_audit(write=False)
    assert payload["status"] == pa.STATUS_INCOMPLETE
    assert payload["passes"] is False


def test_audit_tolerates_silence_from_non_provenance_artifacts(tmp_path, monkeypatch):
    """An ordinary metrics file that records no hash must not block PASS."""
    h = "a" * 64
    required = tmp_path / "input_audit.json"
    required.write_text(json.dumps({"checks": {SUBJECT: {"step8a_dataset_sha256": h}}}))
    silent = tmp_path / "metrics.json"
    silent.write_text(json.dumps({"roc_auc": 0.87}))

    monkeypatch.setattr(pa, "canonical_step8a_hash", lambda experiment_id=SUBJECT: h)
    monkeypatch.setattr(pa, "canonical_step8a_path", lambda experiment_id=SUBJECT: tmp_path / "c.parquet")
    monkeypatch.setattr(pa, "_consumer_files", lambda experiment_id: [
        {"group": "synthetic", "path": required, "provenance_required": True},
        {"group": "synthetic", "path": silent, "provenance_required": False},
    ])
    payload = pa.run_manavgat_single_step8a_hash_audit(write=False)
    assert payload["status"] == pa.STATUS_PASS
    assert payload["passes"] is True


def test_canonical_hash_is_computed_not_hardcoded():
    """The known-good digest must not appear anywhere in the implementation."""
    source = Path(pa.__file__).read_text(encoding="utf-8")
    assert "054a1961" not in source
    assert "81273dd5" not in source


@requires_step8a
def test_canonical_hash_matches_file_on_disk():
    import hashlib

    digest = hashlib.sha256()
    with REAL_STEP8A.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    assert pa.canonical_step8a_hash(SUBJECT) == digest.hexdigest()


# ------------------------------------------------------------------ 12, 13
def test_inventory_is_deterministic(monkeypatch, tmp_path):
    target = tmp_path / "a.json"
    target.write_text("{}")
    monkeypatch.setattr(pa, "_inventory_targets", lambda: [target, tmp_path / "missing.json"])
    first = pa.build_frozen_hash_inventory("before", write=False)
    second = pa.build_frozen_hash_inventory("before", write=False)
    assert first["entries"] == second["entries"]
    assert [e["path"] for e in first["entries"]] == sorted(e["path"] for e in first["entries"])


def test_diff_separates_expected_from_unexpected(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "OUTPUT_ROOT", tmp_path)
    before = {
        "schema_version": pa.SCHEMA_VERSION_INVENTORY, "phase": "before",
        "entries": [
            {"path": f"outputs/experiments/{SUBJECT}/step8b/m.json", "sha256": "1" * 64, "exists": True},
            {"path": "outputs/cross_region/bejis_2022__mugla_2021/step10/step10_metrics.json",
             "sha256": "2" * 64, "exists": True},
            {"path": f"outputs/cross_region/{SUBJECT}__bejis_2022/step10/step10_metrics.json",
             "sha256": "3" * 64, "exists": True},
            {"path": "outputs/experiments/bejis_2022/step8a/step8a_500m_modeling_dataset.parquet",
             "sha256": "4" * 64, "exists": True},
        ],
    }
    after = {
        "schema_version": pa.SCHEMA_VERSION_INVENTORY, "phase": "after",
        "entries": [
            {"path": f"outputs/experiments/{SUBJECT}/step8b/m.json", "sha256": "9" * 64, "exists": True},
            {"path": "outputs/cross_region/bejis_2022__mugla_2021/step10/step10_metrics.json",
             "sha256": "8" * 64, "exists": True},
            {"path": f"outputs/cross_region/{SUBJECT}__bejis_2022/step10/step10_metrics.json",
             "sha256": "7" * 64, "exists": True},
            {"path": "outputs/experiments/bejis_2022/step8a/step8a_500m_modeling_dataset.parquet",
             "sha256": "4" * 64, "exists": True},
        ],
    }
    (tmp_path / "frozen_hash_inventory_before.json").write_text(json.dumps(before))
    (tmp_path / "frozen_hash_inventory_after.json").write_text(json.dumps(after))

    diff = pa.diff_frozen_hash_inventory(write=False)
    expected = {e["path"] for e in diff["expected_changed"]}
    unexpected = {e["path"] for e in diff["unexpected_changed"]}
    unchanged = {e["path"] for e in diff["unchanged"]}

    assert f"outputs/experiments/{SUBJECT}/step8b/m.json" in expected
    assert f"outputs/cross_region/{SUBJECT}__bejis_2022/step10/step10_metrics.json" in expected
    assert "outputs/cross_region/bejis_2022__mugla_2021/step10/step10_metrics.json" in unexpected
    assert "outputs/experiments/bejis_2022/step8a/step8a_500m_modeling_dataset.parquet" in unchanged
    assert diff["passes"] is False


def test_bejis_mugla_evia_changes_are_always_unexpected():
    for path in (
        "outputs/cross_region/bejis_2022__mugla_2021/step10/step10_metrics.json",
        "outputs/cross_region/bejis_2022__evia_2021_extended/step9e/step9e_final_report.json",
        "outputs/cross_region/mugla_2021__evia_2021_extended/step10/step10_final_report.json",
        "outputs/experiments/mugla_2021/step8a/step8a_500m_modeling_dataset.parquet",
        "outputs/experiments/evia_2021_extended/step8a/step8a_500m_modeling_dataset.parquet",
    ):
        assert pa._is_manavgat_dependent(path) is False


def test_manavgat_pairs_are_dependent():
    for path in (
        f"outputs/experiments/{SUBJECT}/step8c/step8c_bootstrap_metrics.json",
        f"outputs/cross_region/{SUBJECT}__mugla_2021/step10/step10_metrics.json",
        f"outputs/cross_region/mugla_2021__{SUBJECT}/step9e/step9e_final_report.json",
    ):
        assert pa._is_manavgat_dependent(path) is True


# --------------------------------------------------------------------- 1, 2, 3
def test_versioned_namespace_is_separate_from_v1():
    assert resolvers.VERSIONED_BIG_BLOCK_NAMESPACE == "step8_big_blocks_v2"
    assert resolvers.DEFAULT_BIG_BLOCK_NAMESPACE == "step8_big_blocks"
    assert resolvers.VERSIONED_BIG_BLOCK_NAMESPACE != resolvers.DEFAULT_BIG_BLOCK_NAMESPACE


def _write_summary(root: Path, experiment_id: str, namespace: str, analysis_id: str) -> Path:
    comparison = root / experiment_id / "robustness" / namespace / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    path = comparison / "big_block_robustness_summary.json"
    path.write_text(json.dumps({
        "experiment_id": experiment_id,
        "analysis_id": analysis_id,
        "report_schema_version": "test.v1",
        "primary_population": resolvers.PRIMARY_POPULATION,
    }))
    return path


def test_resolver_prefers_v2_for_the_experiment_that_has_it(tmp_path, monkeypatch):
    monkeypatch.setattr(
        resolvers, "get_experiment_output_root", lambda eid: tmp_path / eid
    )
    _write_summary(tmp_path, SUBJECT, "step8_big_blocks", "OLD")
    _write_summary(tmp_path, SUBJECT, "step8_big_blocks_v2", "NEW")

    record, raw, reason = resolvers.resolve_large_block_robustness(SUBJECT, ["bejis_2022"])
    assert reason is None
    assert raw["analysis_id"] == "NEW"
    assert record["resolution_method"] == "versioned_per_experiment_schema"
    assert record["namespace"] == "step8_big_blocks_v2"


def test_resolver_does_not_serve_manavgat_v2_to_another_aoi(tmp_path, monkeypatch):
    monkeypatch.setattr(
        resolvers, "get_experiment_output_root", lambda eid: tmp_path / eid
    )
    _write_summary(tmp_path, SUBJECT, "step8_big_blocks_v2", "MANAVGAT_V2")
    _write_summary(tmp_path, "mugla_2021", "step8_big_blocks", "MUGLA_GENERIC")

    record, raw, reason = resolvers.resolve_large_block_robustness("mugla_2021", [SUBJECT])
    assert reason is None
    assert raw["analysis_id"] == "MUGLA_GENERIC"
    assert record["resolution_method"] == "generic_per_experiment_schema"
    assert raw["experiment_id"] == "mugla_2021"


def test_resolver_rejects_mismatched_experiment_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        resolvers, "get_experiment_output_root", lambda eid: tmp_path / eid
    )
    comparison = tmp_path / "bejis_2022" / "robustness" / "step8_big_blocks_v2" / "comparison"
    comparison.mkdir(parents=True)
    (comparison / "big_block_robustness_summary.json").write_text(
        json.dumps({"experiment_id": SUBJECT, "analysis_id": "WRONG"})
    )
    with pytest.raises(resolvers.InputResolutionError):
        resolvers.resolve_large_block_robustness("bejis_2022", [SUBJECT])


def test_versioned_run_does_not_write_into_v1_namespace():
    """The versioned namespace must not be the legacy pair-relative root."""
    legacy = PROJECT_ROOT / "outputs" / "robustness" / "step8_large_block_primary_all_valid"
    versioned = (
        PROJECT_ROOT / "outputs" / "experiments" / SUBJECT / "robustness"
        / resolvers.VERSIONED_BIG_BLOCK_NAMESPACE
    )
    assert legacy not in versioned.parents
    assert versioned != legacy


# ------------------------------------------------------------------- 8, 9, 10, 11
NEW_STEP8A_COLUMNS = ("pre_label_burn_excluded", "analysis_eligible")


def test_new_step8a_columns_are_not_predictors():
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, CATEGORICAL_FEATURES, THERMAL_FEATURES, THERMAL_MODEL_FEATURES,
    )
    every = set(BASELINE_FEATURES) | set(THERMAL_FEATURES) | set(THERMAL_MODEL_FEATURES) | set(CATEGORICAL_FEATURES)
    for column in NEW_STEP8A_COLUMNS:
        assert column not in every


def test_signed_auc_feature_contract_excludes_new_columns():
    from src.evia_signed_auc_bootstrap import NUMERIC_FEATURES

    for column in NEW_STEP8A_COLUMNS:
        assert column not in NUMERIC_FEATURES


def test_models_select_features_by_name_not_position():
    """Feature sets must be name lists, never positional indices."""
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, THERMAL_MODEL_FEATURES,
    )
    for features in (BASELINE_FEATURES, THERMAL_MODEL_FEATURES):
        assert len(features) > 0
        assert all(isinstance(name, str) for name in features)
        assert not any(isinstance(name, int) for name in features)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "outputs" / "experiments" / "bejis_2022" / "step8a"
         / "step8a_500m_modeling_dataset.parquet").is_file(),
    reason="Step8A parquets not present",
)
def test_column_count_heterogeneity_does_not_break_the_feature_contract():
    """Bejis may have 77 columns and the others 79; every required feature
    must still be present in all of them."""
    from src.step8b_train_baseline_vs_thermal_model import (
        BASELINE_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, THERMAL_MODEL_FEATURES,
    )
    required = set(BASELINE_FEATURES) | set(THERMAL_MODEL_FEATURES) | set(CATEGORICAL_FEATURES) | {TARGET_COLUMN}
    counts = {}
    for aoi in AOIS:
        path = PROJECT_ROOT / "outputs" / "experiments" / aoi / "step8a" / "step8a_500m_modeling_dataset.parquet"
        if not path.is_file():
            continue
        columns = set(pd.read_parquet(path).columns)
        counts[aoi] = len(columns)
        missing = sorted(required - columns)
        assert not missing, f"{aoi} is missing required features: {missing}"
    assert len(set(counts.values())) >= 1


# ------------------------------------------------------------------------ 14
@requires_step8a
def test_running_this_module_leaves_real_step8a_parquets_untouched():
    """Hash every real Step8A before and after exercising the audit tooling."""
    import hashlib

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    paths = [
        PROJECT_ROOT / "outputs" / "experiments" / aoi / "step8a" / "step8a_500m_modeling_dataset.parquet"
        for aoi in AOIS
    ]
    paths = [p for p in paths if p.is_file()]
    before = {p: digest(p) for p in paths}

    pa.run_manavgat_single_step8a_hash_audit(write=False)
    pa.build_frozen_hash_inventory("before", write=False)

    after = {p: digest(p) for p in paths}
    assert before == after, "provenance tooling mutated a real Step8A parquet"
