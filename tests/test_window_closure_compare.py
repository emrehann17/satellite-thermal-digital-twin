"""Tests for the window-closure COMPARE stage
(src/window_closure_sensitivity.py + scripts/validate_window_closure_compare.py).

The compare stage is READ-ONLY over the verified model-stage artefacts, so the
whole upstream chain (predictors -> local downstream -> model) is built with the
synthetic fixtures the earlier test modules own, and compare then runs on top.
No test fits a model, draws a bootstrap replicate or touches Earth Engine.

Cost model
----------
Building the prerequisite model stage costs ~23 s (fold fits + bootstrap); every
other stage in the chain costs ~0.4 s and a compare run itself ~0.1 s. Rebuilding
the prerequisite per test therefore accounted for essentially the entire runtime
of this module, so it is built ONCE per module as a golden tree and every test
works on an independent restored copy of it -- see `golden_model_environment`.

Tests are split explicitly:

    @pytest.mark.compare_unit         pure helpers, no filesystem pipeline
    @pytest.mark.compare_integration  runs against the golden model-stage tree
    @pytest.mark.slow                 the few that pay the golden build itself

Nothing is skipped or xfailed, and no scientific assertion was weakened: the
production bootstrap contract, model configuration and output schema are
untouched by this split.
"""
from __future__ import annotations

import contextlib
import csv
import inspect
import io
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.validate_window_closure_compare as validator  # noqa: E402
import src.window_closure_sensitivity as wcs  # noqa: E402

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import test_window_closure_model as md  # noqa: E402

_SHIFTS = md._SHIFTS
_NONZERO = md._NONZERO
_namespace_snapshot = md._namespace_snapshot
_FAST_BOOTSTRAP = md._FAST_BOOTSTRAP

VARIANT_ORDER = ["canonical", *_NONZERO]
EXPECTED = {
    "point_metrics": 18,
    "thermal_contributions": 9,
    "closure_changes": 12,
    "thermal_contribution_changes": 6,
    "bootstrap_evidence_matrix": 27,
}

#: The published compare layout, restated literally. If a fixture optimisation
#: ever changed the production output schema, this would fail.
FROZEN_COMPARE_LAYOUT = {
    "bootstrap_evidence_matrix": "tables/bootstrap_evidence_matrix.csv",
    "closure_changes": "tables/closure_changes.csv",
    "comparison_summary": "summaries/comparison_summary.json",
    "metadata": "compare_stage_metadata.json",
    "point_metrics_long": "tables/point_metrics_long.csv",
    "point_metrics_wide": "tables/point_metrics_wide.csv",
    "provenance_summary": "summaries/provenance_summary.json",
    "report": "report/window_closure_comparison.md",
    "scientific_conclusions": "summaries/scientific_conclusions.json",
    "thermal_contribution_changes": "tables/thermal_contribution_changes.csv",
    "thermal_contributions": "tables/thermal_contributions.csv",
}


# =============================================================================
# Guard scope
#
# The compare stage needs a REAL, verified model stage to read, and building
# that prerequisite legitimately fits models -- during FIXTURE SETUP, which is
# allowed. What must never happen is a fit, a bootstrap draw, a downscaling
# refit, a Step5-Step8A run or a GEE call during COMPARE EXECUTION.
#
# So the two are separated explicitly:
#
#   golden_fixture_setup_model_fit = allowed once (builds the prerequisite)
#   compare_execution_model_fit    = always forbidden (asserted by
#                                    `_compare_guard`)
#
# The always-on blocks below cover only what NOTHING in this module may ever
# reach; they are installed both per test (autouse) and around the golden build,
# so the fail-closed guarantee has no hole. The fit/bootstrap blocks are
# installed by a context manager that wraps the compare call itself and is
# removed again afterwards, so the guard cannot leak into another test.
# =============================================================================
def _blocked(name):
    def _fail(*_args, **_kwargs):
        raise AssertionError(f"fail-closed guard: production {name} was invoked")
    return _fail


def _never_reachable_targets():
    """(module, attribute, replacement) triples no test here may ever reach."""
    import core.gee_utils as gee_utils
    import scripts.prepare_modis_for_step7 as prepare_modis
    import scripts.run_predictors_only as run_predictors_only
    import src.step6_validate_fire_relation as step6

    return [
        (gee_utils, "init_gee", _blocked("init_gee")),
        (run_predictors_only, "export_image_direct_or_tiled", _blocked("GEE exporter")),
        (prepare_modis, "prepare_modis_for_step7", _blocked("MODIS exporter")),
        (step6, "export_raw_mcd64a1_prelabel_labels", _blocked("prelabel exporter")),
        (wcs, "production_local_downstream_engine",
         _blocked("production local-downstream engine (Step5-Step8A)")),
    ]


@pytest.fixture(autouse=True)
def _never_reachable_guard(monkeypatch):
    """Production entry points no test in this module may reach, ever."""
    for module, name, replacement in _never_reachable_targets():
        monkeypatch.setattr(module, name, replacement)


#: Everything `_compare_guard` blocks, as (module attribute, description).
def _compare_guard_targets():
    import src.step7c_train_downscaling_model as step7c
    import src.step8b_train_baseline_vs_thermal_model as step8b

    return [
        (step8b, "train_population", _blocked("fire-risk model fit")),
        (step7c, "run_step7c", _blocked("Step7C downscaling model fit")),
        (wcs, "multi_variant_block_bootstrap", _blocked("bootstrap replicate generation")),
        (wcs, "fit_variant_models", _blocked("fire-risk model fit")),
        (wcs, "run_model_stage", _blocked("model stage execution")),
    ]


@contextlib.contextmanager
def _compare_guard():
    """Fail-closed ONLY around a compare call. Restored on exit."""
    targets = _compare_guard_targets()
    originals = [(module, name, getattr(module, name)) for module, name, _ in targets]
    try:
        for module, name, replacement in targets:
            setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def _compare_guard_is_installed() -> bool:
    """True when the fit/bootstrap blocks are currently in place."""
    return any(
        getattr(getattr(module, name), "__name__", name) == "_fail"
        for module, name, _ in _compare_guard_targets()
    )


# =============================================================================
# Golden model environment
#
# WHY A RESTORED COPY AND NOT A PER-TEST PATH
# -------------------------------------------
# The obvious design -- copy the golden tree into each test's `tmp_path` -- does
# not work here, and that is a property of the PRODUCTION provenance contract,
# not of the tests: every stage records the ABSOLUTE path of each artefact in
# its metadata, and those metadata documents are themselves cross-hashed by the
# stages downstream of them (e.g. `step8a_dataset_stats.json` has its sha256
# recorded in eight other documents). Relocating the tree would mean rewriting
# the recorded paths and then re-forging the whole hash chain that covers them,
# which is exactly the tampering these tests exist to detect. A copy to a new
# path is therefore rejected by the real binding code:
#
#   "artefact '...' lies outside the variant namespace <new>: <old>"
#
# So the golden tree is built ONCE at a single run root, an immutable pristine
# backup is taken beside it, and every test gets that run root wiped and
# restored from the backup. Each test still receives its own independent file
# tree -- a test's writes and tampering can never be observed by another test --
# and the backup itself is never used as a run root, so it cannot drift.
#
# Copying is a plain `shutil.copytree`: no hardlinks (tamper tests mutate file
# CONTENT and would corrupt the golden), no symlinks (they would make the
# containment and provenance checks meaningless) and no platform-specific
# reflink requirement.
# =============================================================================
#: Incremented exactly once, by the module-scoped fixture body.
_GOLDEN_BUILD_COUNT: list[int] = []


@dataclass(frozen=True)
class GoldenModelEnvironment:
    """A completed, verified model stage plus its immutable backup."""

    experiment_id: str
    run_root: Path
    pristine: Path
    out: Path
    experiments: Path
    inventory: dict = field(repr=False)
    pristine_inventory: dict = field(repr=False)
    compare_guard_active_during_setup: bool = False
    restore_count: list = field(default_factory=list, repr=False)

    @property
    def triple(self) -> tuple:
        return self.experiment_id, self.out, self.experiments

    def restore(self) -> tuple:
        """Give the caller a pristine, independent copy of the golden tree."""
        if self.run_root.exists():
            shutil.rmtree(self.run_root)
        shutil.copytree(self.pristine, self.run_root)
        self.restore_count.append(1)
        return self.triple


def _tree_inventory(root: Path) -> dict:
    """relative path -> sha256, so a backup and a run root are comparable."""
    return {
        path.relative_to(root).as_posix(): wcs.sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


@pytest.fixture(scope="module")
def golden_model_environment(tmp_path_factory):
    """FIXTURE SETUP: plan -> ... -> model, completed and verified, ONCE.

    Fitting and bootstrapping happen here on purpose; they are the prerequisite
    the compare stage reads. This is deliberately outside `_compare_guard` --
    and a test below asserts the guard really was absent while it ran.
    """
    base = tmp_path_factory.mktemp("window_closure_compare_golden")
    run_root = base / "env"
    run_root.mkdir()

    with pytest.MonkeyPatch.context() as patcher:
        # The never-reachable blocks apply to the build too: the golden tree is
        # allowed to FIT, never to reach Earth Engine or the production chain.
        for module, name, replacement in _never_reachable_targets():
            patcher.setattr(module, name, replacement)
        guarded = _compare_guard_is_installed()
        _GOLDEN_BUILD_COUNT.append(1)
        _, experiment_id, out, experiments = md._run_model(run_root)

    pristine = base / "pristine"
    shutil.copytree(run_root, pristine)
    inventory = _tree_inventory(run_root)
    return GoldenModelEnvironment(
        experiment_id=experiment_id, run_root=run_root, pristine=pristine,
        out=out, experiments=experiments, inventory=inventory,
        pristine_inventory=_tree_inventory(pristine),
        compare_guard_active_during_setup=guarded,
    )


@pytest.fixture
def compare_env(golden_model_environment):
    """One independent, freshly restored model-stage tree for ONE test."""
    return golden_model_environment.restore()


# =============================================================================
# Environment helpers
# =============================================================================
def _run_compare(env, **kwargs):
    """COMPARE EXECUTION: guarded, so any fit/bootstrap fails the test."""
    experiment_id, out, experiments = env
    with _compare_guard():
        result = wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="compare", to_stage="compare",
            output_root=out, experiments_root=experiments, **kwargs,
        )
    return result, experiment_id, out, experiments


def _dry_run_compare(env):
    experiment_id, out, experiments = env
    with _compare_guard():
        result = wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=True,
            from_stage="compare", to_stage="compare",
            output_root=out, experiments_root=experiments,
        )
    return result, experiment_id, out, experiments


def _metadata(out: Path, experiment_id: str) -> dict:
    return json.loads(
        wcs.compare_metadata_path(experiment_id, out).read_text(encoding="utf-8")
    )


def _table(out: Path, experiment_id: str, key: str) -> list[dict]:
    path = wcs.compare_root(experiment_id, out) / wcs.compare_relative_layout()[key]
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def _write_log(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "INFO before\n" + json.dumps(payload, indent=2, default=str) + "\nINFO after\n",
        encoding="utf-8",
    )
    return path


def _validate(mode: str, experiment_id: str, out: Path, *,
              log: Optional[Path] = None, experiments: Optional[Path] = None) -> int:
    argv = ["--experiment", experiment_id, "--mode", mode,
            "--shifts", *[str(s) for s in _SHIFTS], "--output-root", str(out)]
    if log is not None:
        argv += ["--log", str(log)]
    if experiments is not None:
        argv += ["--experiments-root", str(experiments)]
    return validator.main(argv)


def _upstream_snapshot(experiment_id: str, out: Path, experiments: Path) -> dict:
    return _namespace_snapshot(
        out / experiment_id / "config",
        out / experiment_id / "prelabel_censor",
        out / experiment_id / "variants",
        out / experiment_id / wcs.MODEL_ROOT_DIR,
        wcs.canonical_experiment_root(experiment_id, experiments),
    )


# =============================================================================
# UNIT: 1, 2. Stage lock
# =============================================================================
@pytest.mark.compare_unit
def test_compare_is_implemented_and_the_model_stage_remains():
    assert wcs.COMPARE_STAGE == "compare"
    assert wcs.IMPLEMENTED_ACTUAL_STAGES == wcs.STAGES
    assert wcs.MODEL_STAGE in wcs.IMPLEMENTED_ACTUAL_STAGES
    wcs.assert_actual_stages_supported(wcs.validate_stage_range("compare", "compare"))


@pytest.mark.compare_unit
def test_an_unimplemented_stage_still_fails_fast():
    with pytest.raises(wcs.WindowClosureError, match="not enabled"):
        wcs.assert_actual_stages_supported(["compare", "some-future-stage"])


# =============================================================================
# INTEGRATION: 3-9. Binding, all before any write
# =============================================================================
def _expect_binding_failure(experiment_id, out, experiments, match=""):
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match=match), _compare_guard():
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=list(_SHIFTS), dry_run=False,
            from_stage="compare", to_stage="compare",
            output_root=out, experiments_root=experiments,
        )
    assert not wcs.compare_root(experiment_id, out).exists()
    assert not wcs.compare_staging_root(experiment_id, out).exists()
    assert _namespace_snapshot(out) == before


@pytest.mark.compare_integration
@pytest.mark.slow
def test_the_golden_model_stage_is_built_complete_and_verified(
    compare_env, golden_model_environment,
):
    """The one test that PAYS the golden build (~27 s of fold fits + bootstrap).

    It is the first `compare_integration` test in the file, so it is where the
    module-scoped fixture is materialised; every test after it reuses the same
    tree at ~0.2 s. That is what `@pytest.mark.slow` means here.
    """
    experiment_id, out, _ = compare_env
    metadata = json.loads(
        wcs.model_metadata_path(experiment_id, out).read_text(encoding="utf-8")
    )
    assert metadata["status"] == wcs.STATUS_PASS
    assert metadata["schema_version"] == wcs.MODEL_METADATA_SCHEMA
    for flag, expected in wcs.MODEL_STAGE_REQUIRED_FLAGS.items():
        assert metadata[flag] is expected, flag
    assert wcs.compare_variant_order(metadata["variant_ids"]) == VARIANT_ORDER
    model_dir = wcs.model_root(experiment_id, out)
    for relative in wcs.model_relative_layout().values():
        assert (model_dir / relative).is_file(), relative
    assert not wcs.compare_root(experiment_id, out).exists()
    assert sum(_GOLDEN_BUILD_COUNT) == 1


@pytest.mark.compare_integration
def test_a_wrong_analysis_id_fails_before_writes(compare_env):
    experiment_id, out, experiments = compare_env
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match="analysis_id|shift"), _compare_guard():
        wcs.run_analysis(
            experiment_id=experiment_id, shifts=[0, 7], dry_run=False,
            from_stage="compare", to_stage="compare",
            output_root=out, experiments_root=experiments,
        )
    assert not wcs.compare_root(experiment_id, out).exists()
    assert _namespace_snapshot(out) == before


@pytest.mark.compare_integration
def test_missing_model_metadata_fails_before_writes(compare_env):
    experiment_id, out, experiments = compare_env
    wcs.model_metadata_path(experiment_id, out).unlink()
    _expect_binding_failure(experiment_id, out, experiments, "metadata is missing")


@pytest.mark.compare_integration
def test_a_non_pass_model_metadata_fails(compare_env):
    experiment_id, out, experiments = compare_env
    path = wcs.model_metadata_path(experiment_id, out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "fail"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _expect_binding_failure(experiment_id, out, experiments, "model status")


@pytest.mark.compare_integration
@pytest.mark.parametrize("flag", sorted(wcs.MODEL_STAGE_REQUIRED_FLAGS))
def test_every_required_model_flag_is_enforced(compare_env, flag):
    experiment_id, out, experiments = compare_env
    path = wcs.model_metadata_path(experiment_id, out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[flag] = not wcs.MODEL_STAGE_REQUIRED_FLAGS[flag]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _expect_binding_failure(experiment_id, out, experiments, flag)


@pytest.mark.compare_integration
@pytest.mark.parametrize("key", [
    "point_metrics_csv", "bootstrap_summary_csv", "common_cohort", "shared_folds",
    "bootstrap_replicates",
])
def test_a_missing_model_artifact_fails(compare_env, key):
    experiment_id, out, experiments = compare_env
    target = wcs.model_root(experiment_id, out) / wcs.model_relative_layout()[key]
    target.unlink()
    _expect_binding_failure(experiment_id, out, experiments, "is missing")


@pytest.mark.compare_integration
def test_a_model_artifact_hash_mismatch_fails(compare_env):
    experiment_id, out, experiments = compare_env
    target = (
        wcs.model_root(experiment_id, out)
        / wcs.model_relative_layout()["point_metrics_csv"]
    )
    target.write_bytes(target.read_bytes() + b"drift")
    _expect_binding_failure(experiment_id, out, experiments, "hashes")


# =============================================================================
# 10-18. Exact cardinalities and vocabulary
# =============================================================================
@pytest.mark.compare_unit
def test_expected_cardinalities_are_derived_not_hard_coded():
    derived = wcs.compare_expected_cardinalities(VARIANT_ORDER)
    assert derived == EXPECTED


@pytest.mark.compare_unit
@pytest.mark.parametrize("variant_ids,expected", [
    (["canonical"],
     {"point_metrics": 6, "thermal_contributions": 3, "closure_changes": 0,
      "thermal_contribution_changes": 0, "bootstrap_evidence_matrix": 3}),
    (["canonical", "close_7d_earlier"],
     {"point_metrics": 12, "thermal_contributions": 6, "closure_changes": 6,
      "thermal_contribution_changes": 3, "bootstrap_evidence_matrix": 15}),
])
def test_row_cardinalities_scale_with_the_variant_count(variant_ids, expected):
    """Pure arithmetic over the variant list; no model stage is needed."""
    assert wcs.compare_expected_cardinalities(variant_ids) == expected


@pytest.mark.compare_unit
def test_the_variant_order_is_deterministic_and_canonical_first():
    shuffled = ["close_14d_earlier", "canonical", "close_7d_earlier"]
    assert wcs.compare_variant_order(shuffled) == VARIANT_ORDER
    assert wcs.compare_variant_order(sorted(shuffled)) == VARIANT_ORDER
    assert wcs.compare_variant_order(VARIANT_ORDER) == VARIANT_ORDER


@pytest.mark.compare_unit
def test_the_deterministic_family_metric_and_layout_orders_are_canonical():
    assert list(wcs.COMPARE_FAMILY_ORDER) == [
        wcs.COMPARISON_THERMAL_CONTRIBUTION, wcs.COMPARISON_CLOSURE_CHANGE,
        wcs.COMPARISON_CONTRIBUTION_CHANGE,
    ]
    assert list(wcs.MODEL_METRICS) == ["roc_auc", "pr_auc", "brier"]
    assert list(wcs.MODEL_FAMILIES) == ["baseline", "thermal"]
    layout = wcs.compare_relative_layout()
    assert list(layout) == sorted(layout)


@pytest.mark.compare_integration
def test_every_table_has_its_exact_row_count(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    for key, want in EXPECTED.items():
        table_key = "point_metrics_long" if key == "point_metrics" else key
        assert len(_table(out, experiment_id, table_key)) == want, key
    metadata = _metadata(out, experiment_id)
    assert metadata["point_metric_row_count"] == 18
    assert metadata["thermal_contribution_row_count"] == 9
    assert metadata["closure_change_row_count"] == 12
    assert metadata["thermal_contribution_change_row_count"] == 6
    assert metadata["bootstrap_summary_row_count"] == 27


#: Identity keys per compare table, used by both the unit and the integration
#: duplicate checks so the two cannot drift apart.
DUPLICATE_KEYS = (
    ("point_metrics_long", ("variant_id", "model_family", "metric")),
    ("thermal_contributions", ("variant_id", "metric")),
    ("closure_changes", ("variant_id", "model_family", "metric")),
    ("thermal_contribution_changes", ("variant_id", "metric")),
    ("bootstrap_evidence_matrix",
     ("comparison_family", "variant_id", "model_family", "metric")),
)


def _duplicate_identities(rows, keys) -> list[tuple]:
    seen, duplicates = set(), []
    for row in rows:
        identity = tuple(row.get(key) for key in keys)
        if identity in seen:
            duplicates.append(identity)
        seen.add(identity)
    return duplicates


@pytest.mark.compare_unit
def test_duplicate_key_detection_is_exact():
    """The identity keys really do separate rows -- checked on tiny tables."""
    rows = [
        {"variant_id": "canonical", "model_family": "baseline", "metric": "roc_auc"},
        {"variant_id": "canonical", "model_family": "thermal", "metric": "roc_auc"},
        {"variant_id": "canonical", "model_family": "baseline", "metric": "brier"},
    ]
    keys = ("variant_id", "model_family", "metric")
    assert _duplicate_identities(rows, keys) == []
    assert _duplicate_identities(rows + [dict(rows[0])], keys) == [
        ("canonical", "baseline", "roc_auc")
    ]
    # A key subset that is NOT an identity must collide, which is why the
    # metric belongs in the key.
    assert _duplicate_identities(rows, ("variant_id", "model_family"))


@pytest.mark.compare_integration
def test_no_duplicate_scientific_row_exists(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    for key, keys in DUPLICATE_KEYS:
        rows = _table(out, experiment_id, key)
        assert _duplicate_identities(rows, keys) == [], key


@pytest.mark.compare_integration
def test_only_expected_metrics_variants_and_families_appear(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    for key in ("point_metrics_long", "thermal_contributions", "closure_changes",
                "thermal_contribution_changes", "bootstrap_evidence_matrix"):
        for row in _table(out, experiment_id, key):
            assert row["metric"] in wcs.MODEL_METRICS
            assert row["variant_id"] in VARIANT_ORDER
            if row.get("model_family"):
                assert row["model_family"] in (
                    list(wcs.MODEL_FAMILIES) + ["thermal_minus_baseline"]
                )


@pytest.mark.compare_integration
@pytest.mark.parametrize("bad", [
    {"metric": "f1"}, {"model_family": "ensemble"}, {"variant_id": "close_21d_earlier"},
])
def test_an_unexpected_row_is_refused_by_re_derivation(compare_env, bad):
    experiment_id, out, experiments = compare_env
    path = wcs.model_metadata_path(experiment_id, out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comparisons"][0].update(bad)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _expect_binding_failure(experiment_id, out, experiments, "hash|unknown|re-derive")


# =============================================================================
# 19-24. Directions, precision and display rounding
# =============================================================================
#: The three published raw-delta definitions, as pure arithmetic. The compare
#: stage derives its tables with exactly these expressions; the unit tests below
#: pin the arithmetic and the integration tests pin the published tables to it.
def _thermal_contribution(baseline: float, thermal: float) -> float:
    return thermal - baseline


def _closure_delta(earlier: float, canonical: float) -> float:
    return earlier - canonical


def _contribution_change(earlier_contribution: float,
                         canonical_contribution: float) -> float:
    return earlier_contribution - canonical_contribution


@pytest.mark.compare_unit
@pytest.mark.parametrize("baseline,thermal,expected", [
    (0.70, 0.82, 0.12), (0.82, 0.70, -0.12), (0.25, 0.25, 0.0),
    (0.31, 0.19, -0.12),  # a Brier-shaped pair: the raw sign is kept as is
])
def test_thermal_contribution_arithmetic_is_thermal_minus_baseline(
    baseline, thermal, expected,
):
    assert _thermal_contribution(baseline, thermal) == pytest.approx(expected, abs=1e-12)


@pytest.mark.compare_unit
@pytest.mark.parametrize("earlier,canonical,expected", [
    (0.78, 0.74, 0.04), (0.74, 0.78, -0.04), (0.5, 0.5, 0.0),
])
def test_closure_delta_arithmetic_is_earlier_minus_canonical(
    earlier, canonical, expected,
):
    assert _closure_delta(earlier, canonical) == pytest.approx(expected, abs=1e-12)


@pytest.mark.compare_unit
def test_contribution_change_arithmetic_composes_the_two_deltas():
    """(thermal - baseline)_earlier - (thermal - baseline)_canonical."""
    earlier = _thermal_contribution(0.70, 0.85)
    canonical = _thermal_contribution(0.71, 0.80)
    assert _contribution_change(earlier, canonical) == pytest.approx(0.06, abs=1e-12)
    # Composition identity: it never re-derives from re-oriented values.
    assert _contribution_change(earlier, canonical) == pytest.approx(
        (0.85 - 0.70) - (0.80 - 0.71), abs=1e-12,
    )


@pytest.mark.compare_unit
def test_the_metric_direction_notes_state_the_raw_sign_convention():
    assert "higher ROC-AUC" in wcs.METRIC_DIRECTION_NOTES["roc_auc"]
    assert "higher PR-AUC" in wcs.METRIC_DIRECTION_NOTES["pr_auc"]
    assert "lower Brier score" in wcs.METRIC_DIRECTION_NOTES["brier"]
    assert "never re-oriented" in wcs.METRIC_DIRECTION_NOTES["brier"]
    assert sorted(wcs.METRIC_DIRECTION_NOTES) == sorted(wcs.MODEL_METRICS)


@pytest.mark.compare_integration
def test_thermal_contribution_recomputes_exactly(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    point = {
        (r["variant_id"], r["model_family"], r["metric"]): float(r["value"])
        for r in _table(out, experiment_id, "point_metrics_long")
    }
    for row in _table(out, experiment_id, "thermal_contributions"):
        expected = _thermal_contribution(
            point[(row["variant_id"], "baseline", row["metric"])],
            point[(row["variant_id"], "thermal", row["metric"])],
        )
        assert float(row["contribution_delta"]) == pytest.approx(expected, abs=1e-12)
        assert row["raw_delta_definition"] == "thermal - baseline (raw)"


@pytest.mark.compare_integration
def test_closure_delta_is_earlier_minus_canonical(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    point = {
        (r["variant_id"], r["model_family"], r["metric"]): float(r["value"])
        for r in _table(out, experiment_id, "point_metrics_long")
    }
    rows = _table(out, experiment_id, "closure_changes")
    assert {r["variant_id"] for r in rows} == set(_NONZERO)
    for row in rows:
        expected = _closure_delta(
            point[(row["variant_id"], row["model_family"], row["metric"])],
            point[("canonical", row["model_family"], row["metric"])],
        )
        assert float(row["closure_delta"]) == pytest.approx(expected, abs=1e-12)
        assert row["reference_variant_id"] == "canonical"
        assert row["raw_delta_definition"] == "earlier_closure - canonical (raw)"


@pytest.mark.compare_integration
def test_contribution_change_recomputes_exactly(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    contribution = {
        (r["variant_id"], r["metric"]): float(r["contribution_delta"])
        for r in _table(out, experiment_id, "thermal_contributions")
    }
    for row in _table(out, experiment_id, "thermal_contribution_changes"):
        expected = _contribution_change(
            contribution[(row["variant_id"], row["metric"])],
            contribution[("canonical", row["metric"])],
        )
        assert float(row["contribution_change_delta"]) == pytest.approx(expected, abs=1e-12)


@pytest.mark.compare_integration
def test_the_brier_raw_delta_sign_is_retained(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    for row in _table(out, experiment_id, "thermal_contributions"):
        if row["metric"] != "brier":
            continue
        raw = float(row["thermal"]) - float(row["baseline"])
        assert float(row["contribution_delta"]) == pytest.approx(raw, abs=1e-12)
        assert "lower Brier score" in row["metric_direction_note"]


@pytest.mark.compare_integration
def test_machine_readable_values_keep_full_precision(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    values = [
        row["value"] for row in _table(out, experiment_id, "point_metrics_long")
    ]
    assert any(len(v.split(".")[-1]) > wcs.COMPARE_DISPLAY_DECIMALS for v in values)
    assert _metadata(out, experiment_id)["machine_readable_values_rounded"] is False


@pytest.mark.compare_integration
def test_markdown_uses_display_rounding_only(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    text = (
        wcs.compare_root(experiment_id, out)
        / wcs.compare_relative_layout()["report"]
    ).read_text(encoding="utf-8")
    assert f"Display rounding: {wcs.COMPARE_DISPLAY_DECIMALS} decimals" in text
    for row in _table(out, experiment_id, "bootstrap_evidence_matrix"):
        assert f"{float(row['point_delta']):.3f}" in text
    assert "Negative raw deltas indicate lower Brier scores." in text


# =============================================================================
# 25-32. Bootstrap status mapping and forbidden wording
# =============================================================================
@pytest.mark.compare_unit
@pytest.mark.parametrize("low,high,expected", [
    (0.01, 0.05, wcs.INTERVAL_SUPPORTED_INCREASE),
    (-0.05, -0.01, wcs.INTERVAL_SUPPORTED_DECREASE),
    (-0.01, 0.02, wcs.INTERVAL_INCLUDES_ZERO),
])
def test_ci_maps_to_the_allowed_status(low, high, expected):
    assert wcs.classify_change_interval(low, high) == expected


@pytest.mark.compare_unit
@pytest.mark.parametrize("low,high", [(0.0, 0.05), (-0.05, 0.0), (0.0, 0.0)])
def test_an_interval_touching_zero_is_never_reported_as_supported(low, high):
    assert wcs.classify_change_interval(low, high) == wcs.INTERVAL_INCLUDES_ZERO


@pytest.mark.compare_unit
@pytest.mark.parametrize("metric,status,fragment", [
    ("roc_auc", wcs.INTERVAL_SUPPORTED_INCREASE, "higher metric value"),
    ("pr_auc", wcs.INTERVAL_SUPPORTED_DECREASE, "lower metric value"),
    ("brier", wcs.INTERVAL_SUPPORTED_DECREASE, "lower Brier score"),
    ("brier", wcs.INTERVAL_SUPPORTED_INCREASE, "higher Brier score"),
    ("brier", wcs.INTERVAL_INCLUDES_ZERO, "direction of the metric change"),
    ("roc_auc", wcs.INTERVAL_INCLUDES_ZERO, "uncertainty remains"),
])
def test_evidence_statement_wording(metric, status, fragment):
    assert fragment in wcs.evidence_statement(metric, status)


@pytest.mark.compare_unit
def test_every_evidence_statement_is_free_of_forbidden_wording():
    for metric in wcs.MODEL_METRICS:
        for status in (wcs.INTERVAL_SUPPORTED_INCREASE,
                       wcs.INTERVAL_SUPPORTED_DECREASE,
                       wcs.INTERVAL_INCLUDES_ZERO):
            wcs.assert_compare_wording(
                {"evidence_statement": wcs.evidence_statement(metric, status)},
                f"{metric}/{status}",
            )


@pytest.mark.compare_unit
def test_brier_wording_never_reorients_the_raw_sign():
    """A Brier decrease is an IMPROVEMENT but is still reported as a decrease."""
    lower = wcs.evidence_statement("brier", wcs.INTERVAL_SUPPORTED_DECREASE)
    higher = wcs.evidence_statement("brier", wcs.INTERVAL_SUPPORTED_INCREASE)
    assert "lower Brier score" in lower and "higher Brier score" not in lower
    assert "higher Brier score" in higher and "lower Brier score" not in higher
    assert lower != higher


@pytest.mark.compare_integration
def test_every_evidence_row_matches_its_status(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    allowed = {
        wcs.INTERVAL_SUPPORTED_INCREASE, wcs.INTERVAL_SUPPORTED_DECREASE,
        wcs.INTERVAL_INCLUDES_ZERO,
    }
    for row in _table(out, experiment_id, "bootstrap_evidence_matrix"):
        assert row["status"] in allowed
        assert row["status"] == wcs.classify_change_interval(
            float(row["ci_low"]), float(row["ci_high"]),
        )
        assert row["evidence_statement"] == wcs.evidence_statement(
            row["metric"], row["status"],
        )
        if row["status"] == wcs.INTERVAL_INCLUDES_ZERO:
            assert "uncertainty remains" in row["evidence_statement"]
        assert row["limitation_statement"]


@pytest.mark.compare_unit
@pytest.mark.parametrize("requested,valid,invalid,rows", [
    (24, 24, 0, 24),
    (24, 19, 5, 19),
])
def test_saved_bootstrap_count_contract_accepts_truthful_counts(
    requested, valid, invalid, rows,
):
    assert wcs.validate_saved_bootstrap_replicate_counts(
        requested, valid, invalid, rows,
    ) == (valid, invalid)


@pytest.mark.compare_unit
@pytest.mark.parametrize("valid,invalid", [(18, 5), (19, 4)])
def test_saved_bootstrap_count_contract_rejects_tampered_counts(valid, invalid):
    with pytest.raises(wcs.WindowClosureError, match="valid replicate|invalid replicate"):
        wcs.validate_saved_bootstrap_replicate_counts(24, valid, invalid, 19)


@pytest.mark.compare_unit
def test_the_saved_replicate_counts_must_partition_the_request():
    """valid + invalid == requested, and valid == the saved row count."""
    assert wcs.validate_saved_bootstrap_replicate_counts(24, 20, 4, 20) == (20, 4)
    with pytest.raises(wcs.WindowClosureError):
        wcs.validate_saved_bootstrap_replicate_counts(24, 20, 4, 19)


@pytest.mark.compare_unit
def test_machine_paths_are_not_scanned_as_generated_prose():
    wcs.assert_compare_wording(
        {"source_model_metadata_path": "/tmp/test_robust_fixture/model.json",
         "note": "comparison evidence"},
        "dry run",
    )


@pytest.mark.compare_unit
@pytest.mark.parametrize("phrase", list(wcs.FORBIDDEN_COMPARE_PHRASES))
def test_forbidden_wording_is_refused(phrase):
    with pytest.raises(wcs.WindowClosureError, match="forbidden wording"):
        wcs.assert_compare_wording({"note": f"the result is {phrase} here"}, "test")


@pytest.mark.compare_integration
def test_forbidden_wording_is_absent_from_every_prose_artifact(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    root = wcs.compare_root(experiment_id, out)
    layout = wcs.compare_relative_layout()
    report = (root / layout["report"]).read_text(encoding="utf-8")
    wcs.assert_compare_wording(report, "report")
    for key in ("comparison_summary", "scientific_conclusions", "provenance_summary",
                "metadata"):
        payload = json.loads((root / layout[key]).read_text(encoding="utf-8"))
        wcs.assert_compare_wording(payload, key)


@pytest.mark.compare_unit
def test_machine_key_names_are_not_treated_as_claims():
    """`frozen_hashes_unchanged` is an identifier, not a stability claim."""
    wcs.assert_compare_wording(
        {"frozen_hashes_unchanged": True, "note": "interval includes zero"}, "test",
    )


# =============================================================================
# 33-39. Conclusions and deterministic ordering
# =============================================================================
def _synthetic_evidence_rows() -> list[dict]:
    """A tiny, hand-built evidence matrix: three families x three metrics."""
    statuses = {
        "roc_auc": wcs.INTERVAL_SUPPORTED_INCREASE,
        "pr_auc": wcs.INTERVAL_INCLUDES_ZERO,
        "brier": wcs.INTERVAL_SUPPORTED_DECREASE,
    }
    return [
        {
            "comparison_family": family, "variant_id": "close_7d_earlier",
            "model_family": "thermal", "metric": metric, "status": statuses[metric],
            "evidence_statement": wcs.evidence_statement(metric, statuses[metric]),
        }
        for family in wcs.COMPARE_FAMILY_ORDER
        for metric in wcs.MODEL_METRICS
    ]


@pytest.mark.compare_unit
def test_conclusions_group_without_a_verdict_or_a_majority_vote():
    conclusions = wcs.build_compare_conclusions(_synthetic_evidence_rows())
    assert conclusions["single_overall_scientific_verdict_produced"] is False
    assert conclusions["majority_vote_across_metrics_taken"] is False
    assert conclusions["technical_validation_status"] == "pass"
    assert "overall_scientific_status" not in conclusions
    assert sorted(conclusions["conclusions_by_metric"]) == sorted(wcs.MODEL_METRICS)
    assert sorted(conclusions["conclusions_by_comparison_family"]) == sorted(
        wcs.COMPARE_FAMILY_ORDER
    )
    assert sum(conclusions["evidence_counts"].values()) == 9
    for metric, entry in conclusions["conclusions_by_metric"].items():
        assert entry["row_count"] == 3
        assert sum(entry["evidence_counts"].values()) == entry["row_count"]
        assert entry["metric_direction_note"] == wcs.METRIC_DIRECTION_NOTES[metric]
    assert conclusions["conclusions_by_metric"]["pr_auc"]["evidence_counts"][
        wcs.INTERVAL_INCLUDES_ZERO
    ] == 3


@pytest.mark.compare_unit
def test_conclusions_never_collapse_metrics_into_one_row():
    """Nine rows in, nine rows out, grouped twice -- never reduced to a verdict."""
    conclusions = wcs.build_compare_conclusions(_synthetic_evidence_rows())
    by_metric = sum(
        entry["row_count"] for entry in conclusions["conclusions_by_metric"].values()
    )
    by_family = sum(
        entry["row_count"]
        for entry in conclusions["conclusions_by_comparison_family"].values()
    )
    assert by_metric == by_family == 9


@pytest.mark.compare_integration
def test_no_single_overall_scientific_verdict_is_produced(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    conclusions = json.loads(
        (wcs.compare_root(experiment_id, out)
         / wcs.compare_relative_layout()["scientific_conclusions"])
        .read_text(encoding="utf-8")
    )
    assert conclusions["single_overall_scientific_verdict_produced"] is False
    assert conclusions["majority_vote_across_metrics_taken"] is False
    assert conclusions["technical_validation_status"] == "pass"
    assert "overall_scientific_status" not in conclusions
    metadata = _metadata(out, experiment_id)
    assert metadata["single_overall_scientific_verdict_produced"] is False
    assert metadata["majority_vote_across_metrics_taken"] is False


@pytest.mark.compare_integration
def test_conclusions_are_grouped_by_metric_and_family(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    conclusions = json.loads(
        (wcs.compare_root(experiment_id, out)
         / wcs.compare_relative_layout()["scientific_conclusions"])
        .read_text(encoding="utf-8")
    )
    assert sorted(conclusions["conclusions_by_metric"]) == sorted(wcs.MODEL_METRICS)
    assert sorted(conclusions["conclusions_by_comparison_family"]) == sorted(
        wcs.COMPARE_FAMILY_ORDER
    )
    total = sum(conclusions["evidence_counts"].values())
    assert total == EXPECTED["bootstrap_evidence_matrix"]
    for metric, entry in conclusions["conclusions_by_metric"].items():
        assert sum(entry["evidence_counts"].values()) == entry["row_count"]
        assert entry["metric_direction_note"] == wcs.METRIC_DIRECTION_NOTES[metric]


@pytest.mark.compare_unit
def test_the_deterministic_row_order_is_a_pure_product_of_the_orderings():
    """The published long-table order is variant x family x metric, in order."""
    expected = [
        (v, f, m) for v in VARIANT_ORDER
        for f in wcs.MODEL_FAMILIES for m in wcs.MODEL_METRICS
    ]
    assert len(expected) == EXPECTED["point_metrics"]
    assert expected[0] == ("canonical", "baseline", "roc_auc")
    assert expected[-1] == (_NONZERO[-1], "thermal", "brier")
    # Sorting the tuples alphabetically must NOT reproduce it: the order is the
    # declared one, not an incidental sort.
    assert expected != sorted(expected)


@pytest.mark.compare_integration
def test_orderings_are_deterministic(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    metadata = _metadata(out, experiment_id)
    assert metadata["variant_order"] == VARIANT_ORDER
    assert metadata["model_family_order"] == list(wcs.MODEL_FAMILIES)
    assert metadata["metric_order"] == list(wcs.MODEL_METRICS)
    assert metadata["comparison_family_order"] == list(wcs.COMPARE_FAMILY_ORDER)

    rows = _table(out, experiment_id, "point_metrics_long")
    seen = [(r["variant_id"], r["model_family"], r["metric"]) for r in rows]
    expected = [
        (v, f, m) for v in VARIANT_ORDER
        for f in wcs.MODEL_FAMILIES for m in wcs.MODEL_METRICS
    ]
    assert seen == expected

    families = [r["comparison_family"] for r in
                _table(out, experiment_id, "bootstrap_evidence_matrix")]
    order = {name: i for i, name in enumerate(wcs.COMPARE_FAMILY_ORDER)}
    assert families == sorted(families, key=lambda name: order[name])


# =============================================================================
# 40-49. Dry run, resume/force, atomicity
# =============================================================================
@pytest.mark.compare_integration
def test_the_dry_run_writes_nothing(compare_env):
    experiment_id, out, experiments = compare_env
    before = _namespace_snapshot(out)
    result, _, _, _ = _dry_run_compare(compare_env)

    assert result["ran"] is False and result["dry_run"] is True
    assert result["planned_stages"] == ["compare"]
    assert result["files_written"] is False
    summary = result["compare_stage_summary"]
    for flag in ("model_fit", "fire_risk_model_fit", "bootstrap_run",
                 "bootstrap_recomputed", "compare_run", "gee_queries_run",
                 "gee_exports_run", "model_refit_planned",
                 "bootstrap_recompute_planned"):
        assert summary[flag] is False, flag
    for flag in ("compare_run_planned", "report_generation_planned",
                 "tables_generation_planned"):
        assert summary[flag] is True, flag
    assert summary["expected_cardinalities"] == EXPECTED
    assert summary["input_binding_ready"] is True
    assert not wcs.compare_root(experiment_id, out).exists()
    assert not wcs.compare_staging_root(experiment_id, out).exists()
    assert _namespace_snapshot(out) == before


@pytest.mark.compare_integration
def test_the_dry_run_snapshot_of_an_existing_tree_is_unchanged(compare_env):
    _run_compare(compare_env)
    experiment_id, out, _ = compare_env
    before = _namespace_snapshot(out)
    result, _, _, _ = _dry_run_compare(compare_env)
    assert result["compare_stage_owned_snapshot_unchanged"] is True
    assert result["compare_dry_run_created_paths"] == []
    assert result["compare_dry_run_modified_paths"] == []
    assert result["compare_dry_run_deleted_paths"] == []
    assert result["preexisting_compare_stage_owned_paths"]
    assert _namespace_snapshot(out) == before


@pytest.mark.compare_integration
def test_a_plain_rerun_rejects_an_existing_pass_output(compare_env):
    _run_compare(compare_env)
    _, out, _ = compare_env
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match="Refusing to overwrite"):
        _run_compare(compare_env)
    assert _namespace_snapshot(out) == before


@pytest.mark.compare_integration
def test_resume_reuses_a_valid_output_without_mutation(compare_env):
    _run_compare(compare_env)
    _, out, _ = compare_env
    before = _namespace_snapshot(out)
    result, _, _, _ = _run_compare(compare_env, resume=True)
    assert result["compare_reused"] is True
    assert _namespace_snapshot(out) == before


@pytest.mark.compare_integration
def test_resume_rejects_a_partial_output_without_mutation(compare_env):
    experiment_id, out, _ = compare_env
    partial = wcs.compare_root(experiment_id, out) / "tables"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "point_metrics_long.csv").write_bytes(b"partial")
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match="cannot reuse the compare stage"):
        _run_compare(compare_env, resume=True)
    assert _namespace_snapshot(out) == before


@pytest.mark.compare_integration
def test_a_plain_rerun_rejects_a_partial_output(compare_env):
    experiment_id, out, _ = compare_env
    partial = wcs.compare_root(experiment_id, out) / "tables"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "point_metrics_long.csv").write_bytes(b"partial")
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError, match="NOT reusable"):
        _run_compare(compare_env)
    assert _namespace_snapshot(out) == before


@pytest.mark.compare_integration
def test_force_quarantines_only_compare_and_preserves_upstream(compare_env):
    _run_compare(compare_env)
    experiment_id, out, experiments = compare_env
    upstream_before = _upstream_snapshot(experiment_id, out, experiments)
    old = wcs.compare_metadata_path(experiment_id, out).read_bytes()

    result, _, _, _ = _run_compare(compare_env, force=True)
    manifest = result["quarantine_manifest"]
    assert manifest["quarantined"] is True
    entry = manifest["entries"][0]
    for field_name in ("original_path", "quarantined_path", "reason", "timestamp_utc",
                       "pre_quarantine_inventory_sha256"):
        assert entry[field_name], field_name
    kept = Path(entry["quarantined_path"]) / wcs.COMPARE_METADATA_NAME
    assert kept.read_bytes() == old
    assert wcs.compare_metadata_path(experiment_id, out).is_file()
    assert _upstream_snapshot(experiment_id, out, experiments) == upstream_before


@pytest.mark.compare_integration
def test_a_failure_writes_no_pass_metadata_and_leaves_no_staging(compare_env):
    experiment_id, out, _ = compare_env
    path = wcs.model_metadata_path(experiment_id, out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comparisons"] = payload["comparisons"][:5]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    before = _namespace_snapshot(out)
    with pytest.raises(wcs.WindowClosureError):
        _run_compare(compare_env)
    assert not wcs.compare_root(experiment_id, out).exists()
    assert not wcs.compare_staging_root(experiment_id, out).exists()
    assert _namespace_snapshot(out) == before


# =============================================================================
# 50-58. Namespace, provenance and execution flags
# =============================================================================
@pytest.mark.compare_integration
def test_every_output_stays_inside_compare(compare_env):
    result, experiment_id, out, _ = _run_compare(compare_env)
    root = wcs.compare_root(experiment_id, out).resolve()
    for path in result["files_written"]:
        assert root in Path(path).resolve().parents
    for record in _metadata(out, experiment_id)["output_artifacts"]:
        assert root in Path(record["path"]).resolve().parents
    for forbidden in ("variants", "prelabel_censor", "config", wcs.MODEL_ROOT_DIR):
        base = (out / experiment_id / forbidden).resolve()
        for path in result["files_written"]:
            assert base not in Path(path).resolve().parents


@pytest.mark.compare_integration
def test_upstream_and_model_outputs_are_untouched(compare_env):
    experiment_id, out, experiments = compare_env
    before = _upstream_snapshot(experiment_id, out, experiments)
    _run_compare(compare_env)
    assert _upstream_snapshot(experiment_id, out, experiments) == before


@pytest.mark.compare_integration
def test_the_compare_stage_execution_flags_are_truthful(compare_env):
    result, experiment_id, out, _ = _run_compare(compare_env)
    metadata = _metadata(out, experiment_id)
    for payload in (metadata, result):
        assert payload["compare_run"] is True
        assert payload["model_fit"] is False
        assert payload["fire_risk_model_fit"] is False
        assert payload["downscaling_model_fit"] is False
        assert payload["bootstrap_run"] is False
        assert payload["bootstrap_recomputed"] is False
        assert payload["gee_queries_run"] is False
        assert payload["gee_exports_run"] is False
        assert payload["canonical_outputs_modified"] is False
        assert payload["upstream_outputs_modified"] is False


@pytest.mark.compare_integration
def test_compare_execution_is_guarded_against_fitting_and_bootstrapping(compare_env):
    """`_run_compare` wraps the call in the guard; reaching a fit fails it."""
    import src.step7c_train_downscaling_model as step7c
    import src.step8b_train_baseline_vs_thermal_model as step8b

    assert not _compare_guard_is_installed()
    _run_compare(compare_env)
    # Inside the guard every one of these is fail-closed...
    with _compare_guard():
        assert _compare_guard_is_installed()
        for call in (
            lambda: step8b.train_population(None, "p", 5, 42, "random_forest", 30),
            lambda: step7c.run_step7c(),
            lambda: wcs.multi_variant_block_bootstrap(None, None, {}),
            lambda: wcs.fit_variant_models("v", None, {}, {}),
            lambda: wcs.run_model_stage("e", "a", {}, [], {}),
        ):
            with pytest.raises(AssertionError, match="fail-closed guard"):
                call()
    # ...and restored afterwards, so the guard cannot leak into another test.
    assert step8b.train_population.__name__ == "train_population"
    assert step7c.run_step7c.__name__ == "run_step7c"
    assert wcs.multi_variant_block_bootstrap.__name__ == "multi_variant_block_bootstrap"
    assert not _compare_guard_is_installed()


@pytest.mark.compare_integration
def test_the_model_fixture_setup_is_not_blocked_by_the_guard(compare_env):
    """Building the prerequisite model stage legitimately fits models."""
    experiment_id, out, _ = compare_env
    metadata = json.loads(
        wcs.model_metadata_path(experiment_id, out).read_text(encoding="utf-8")
    )
    assert metadata["status"] == "pass"
    assert metadata["fire_risk_model_fit"] is True


@pytest.mark.compare_integration
def test_the_model_tree_hash_is_unchanged_across_a_compare_run(compare_env):
    experiment_id, out, _ = compare_env
    before = _namespace_snapshot(out / experiment_id / wcs.MODEL_ROOT_DIR)
    _run_compare(compare_env)
    assert _namespace_snapshot(out / experiment_id / wcs.MODEL_ROOT_DIR) == before


@pytest.mark.compare_integration
def test_the_full_output_layout_is_produced(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    root = wcs.compare_root(experiment_id, out)
    for relative in wcs.compare_relative_layout().values():
        assert (root / relative).is_file(), relative


# =============================================================================
# 59-64. Validator
# =============================================================================
@pytest.mark.compare_integration
def test_validator_reports_the_stage_lock(compare_env, tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run_compare(compare_env)
    _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "l.log", result))
    captured = capsys.readouterr().out
    assert "[PASS] compare is an implemented actual stage" in captured
    assert "[PASS] no stage exists after compare" in captured


@pytest.mark.compare_integration
def test_validator_accepts_a_valid_dry_run(compare_env, tmp_path, capsys):
    result, experiment_id, out, _ = _dry_run_compare(compare_env)
    code = _validate("dry-run", experiment_id, out, log=_write_log(tmp_path / "d.log", result))
    output = capsys.readouterr().out
    assert code == 0, output
    assert "OVERALL STATUS: PASS" in output


@pytest.mark.compare_integration
def test_validator_accepts_a_valid_actual_run(compare_env, capsys):
    _, experiment_id, out, experiments = _run_compare(compare_env)
    code = _validate("actual", experiment_id, out, experiments=experiments)
    output = capsys.readouterr().out
    assert code == 0, output
    assert "OVERALL STATUS: PASS" in output
    assert "TECHNICAL STATUS: PASS" in output
    assert "SCIENTIFIC-CONTRACT STATUS: PASS" in output
    assert "NAMESPACE / PROVENANCE SAFETY: PASS" in output


@pytest.mark.compare_integration
def test_validator_rejects_a_tampered_table(compare_env, capsys):
    _, experiment_id, out, experiments = _run_compare(compare_env)
    path = (
        wcs.compare_root(experiment_id, out)
        / wcs.compare_relative_layout()["thermal_contributions"]
    )
    path.write_text(path.read_text(encoding="utf-8").replace("thermal - baseline (raw)",
                                                             "baseline - thermal"),
                    encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] every output hash matches the metadata" in capsys.readouterr().out


@pytest.mark.compare_integration
def test_validator_rejects_a_tampered_status(compare_env, capsys):
    _, experiment_id, out, experiments = _run_compare(compare_env)
    root = wcs.compare_root(experiment_id, out)
    path = root / wcs.compare_relative_layout()["bootstrap_evidence_matrix"]
    rows = _table(out, experiment_id, "bootstrap_evidence_matrix")
    # Pick a status the row does NOT already carry, otherwise the "tamper" is a
    # no-op that the validator is right to accept.
    rows[0]["status"] = next(
        status for status in (wcs.INTERVAL_SUPPORTED_INCREASE,
                              wcs.INTERVAL_SUPPORTED_DECREASE,
                              wcs.INTERVAL_INCLUDES_ZERO)
        if status != rows[0]["status"]
    )
    columns = sorted(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    # Re-point the recorded hash so the STATUS check is what fires.
    metadata_path = wcs.compare_metadata_path(experiment_id, out)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for record in metadata["output_artifacts"]:
        if record["relative_path"].endswith("bootstrap_evidence_matrix.csv"):
            record["sha256"] = wcs.sha256_file(path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    output = capsys.readouterr().out
    # The [FAIL] prefix matters: the PASS line contains the same wording.
    assert "[FAIL] every status follows exactly from its interval" in output


@pytest.mark.compare_integration
def test_validator_rejects_an_unrecorded_compare_file(compare_env, capsys):
    _, experiment_id, out, experiments = _run_compare(compare_env)
    stray = wcs.compare_root(experiment_id, out) / "tables" / "extra.csv"
    stray.write_text("a,b\n1,2\n", encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    output = capsys.readouterr().out
    assert "[FAIL] compare/ contains no unrecorded file" in output
    assert "compare/tables/extra.csv" in output


@pytest.mark.compare_integration
def test_validator_never_writes(compare_env):
    _, experiment_id, out, experiments = _run_compare(compare_env)
    before = _namespace_snapshot(out)
    _validate("actual", experiment_id, out, experiments=experiments)
    assert _namespace_snapshot(out) == before


# =============================================================================
# 65-71. The stage-control metadata is an allow-listed control file
#
# `compare_stage_metadata.json` is the document that RECORDS the output hashes,
# so it can never appear in its own `output_artifacts` list without creating a
# self-hash cycle. It is therefore the ONE allow-listed control file, by exact
# relative path -- and everything else on disk must still be recorded.
# =============================================================================
@pytest.mark.compare_unit
def test_the_control_file_allow_list_is_exactly_the_stage_metadata():
    assert validator.COMPARE_CONTROL_FILES == frozenset({
        "compare/compare_stage_metadata.json",
    })
    assert len(validator.COMPARE_CONTROL_FILES) == 1


@pytest.mark.compare_integration
def test_the_metadata_never_records_itself_as_an_output(compare_env):
    _, experiment_id, out, _ = _run_compare(compare_env)
    metadata_path = wcs.compare_metadata_path(experiment_id, out)
    metadata = _metadata(out, experiment_id)
    recorded = {Path(r["path"]).resolve() for r in metadata["output_artifacts"]}
    assert metadata_path.resolve() not in recorded
    assert not any(
        r["relative_path"].endswith(wcs.COMPARE_METADATA_NAME)
        for r in metadata["output_artifacts"]
    )
    # ...and no output carries a fake or empty digest either.
    for record in metadata["output_artifacts"]:
        assert record["sha256"] and len(record["sha256"]) == 64
        assert record["sha256"] == wcs.sha256_file(Path(record["path"]))


@pytest.mark.compare_integration
def test_validator_rejects_a_self_recorded_stage_metadata(compare_env, capsys):
    _, experiment_id, out, experiments = _run_compare(compare_env)
    metadata_path = wcs.compare_metadata_path(experiment_id, out)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_artifacts"].append({
        "relative_path": wcs.COMPARE_METADATA_NAME,
        "path": str(metadata_path),
        "sha256": "",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True),
                             encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert (
        "the stage-control metadata is not recorded in its own output_artifacts"
        in capsys.readouterr().out
    )


@pytest.mark.compare_integration
def test_validator_rejects_a_stray_file_beside_the_stage_metadata(compare_env, capsys):
    """The allow-list is by exact path, not by directory."""
    _, experiment_id, out, experiments = _run_compare(compare_env)
    stray = wcs.compare_root(experiment_id, out) / "compare_stage_notes.json"
    stray.write_text("{}", encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    output = capsys.readouterr().out
    assert "[FAIL] compare/ contains no unrecorded file" in output
    assert "compare/compare_stage_notes.json" in output


@pytest.mark.compare_integration
def test_validator_rejects_a_recorded_output_that_is_gone(compare_env, capsys):
    _, experiment_id, out, experiments = _run_compare(compare_env)
    (wcs.compare_root(experiment_id, out)
     / wcs.compare_relative_layout()["closure_changes"]).unlink()
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    output = capsys.readouterr().out
    assert "[FAIL] every recorded output is present under compare/" in output
    assert "[FAIL] every recorded output exists" in output


@pytest.mark.compare_integration
def test_validator_rejects_a_missing_stage_metadata(compare_env, capsys):
    _, experiment_id, out, experiments = _run_compare(compare_env)
    wcs.compare_metadata_path(experiment_id, out).unlink()
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] compare stage metadata exists" in capsys.readouterr().out


@pytest.mark.compare_integration
def test_validator_rejects_a_corrupt_stage_metadata(compare_env, capsys):
    _, experiment_id, out, experiments = _run_compare(compare_env)
    wcs.compare_metadata_path(experiment_id, out).write_text(
        "{not json at all", encoding="utf-8",
    )
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL] compare stage metadata is readable" in capsys.readouterr().out


@pytest.mark.compare_integration
@pytest.mark.parametrize("field_name,value", [
    ("schema_version", "window_closure_compare.v0"),
    ("status", "fail"),
    ("analysis_id", "not-the-preregistered-id"),
    ("experiment_id", "not-the-experiment"),
    ("compare_run", False),
    ("model_fit", True),
    ("bootstrap_run", True),
])
def test_validator_still_checks_the_stage_metadata_contract(
    compare_env, capsys, field_name, value,
):
    """Allow-listing the control file did not weaken any check ON it."""
    _, experiment_id, out, experiments = _run_compare(compare_env)
    metadata_path = wcs.compare_metadata_path(experiment_id, out)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field_name] = value
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True),
                             encoding="utf-8")
    assert _validate("actual", experiment_id, out, experiments=experiments) == 1
    assert "[FAIL]" in capsys.readouterr().out


# =============================================================================
# 72-80. Fixture-architecture regressions
#
# These assert the PROPERTIES the golden-fixture optimisation must preserve.
# =============================================================================
@pytest.mark.compare_integration
def test_the_golden_model_environment_is_built_exactly_once(
    compare_env, golden_model_environment,
):
    assert sum(_GOLDEN_BUILD_COUNT) == 1
    assert golden_model_environment.pristine.is_dir()
    assert golden_model_environment.run_root != golden_model_environment.pristine


@pytest.mark.compare_integration
def test_every_environment_starts_from_the_golden_inventory(
    compare_env, golden_model_environment,
):
    """Each test opens on a byte-for-byte restored copy of the golden tree."""
    assert _tree_inventory(golden_model_environment.run_root) == (
        golden_model_environment.inventory
    )
    assert len(golden_model_environment.restore_count) >= 1


#: A two-test pair: the first tampers heavily, the second must see none of it.
_TAMPER_MARKER = "tamper_leak_marker.txt"


@pytest.mark.compare_integration
def test_a_tamper_is_applied_to_this_environment_only__step_1(compare_env):
    experiment_id, out, _ = compare_env
    model_dir = wcs.model_root(experiment_id, out)
    (model_dir / _TAMPER_MARKER).write_text("tampered", encoding="utf-8")
    wcs.model_metadata_path(experiment_id, out).write_text("{}", encoding="utf-8")
    target = model_dir / wcs.model_relative_layout()["point_metrics_csv"]
    target.write_bytes(b"corrupted")
    assert (model_dir / _TAMPER_MARKER).is_file()


@pytest.mark.compare_integration
def test_a_tamper_does_not_leak_into_the_next_environment__step_2(
    compare_env, golden_model_environment,
):
    experiment_id, out, _ = compare_env
    model_dir = wcs.model_root(experiment_id, out)
    assert not (model_dir / _TAMPER_MARKER).exists()
    metadata = json.loads(
        wcs.model_metadata_path(experiment_id, out).read_text(encoding="utf-8")
    )
    assert metadata["status"] == "pass"
    assert _tree_inventory(golden_model_environment.run_root) == (
        golden_model_environment.inventory
    )
    # The tampered tree is genuinely gone: compare binds and runs cleanly again.
    _run_compare(compare_env)


@pytest.mark.compare_integration
def test_the_golden_backup_is_never_written_to(golden_model_environment):
    """The pristine tree is a source, never a run root, so it cannot drift."""
    assert _tree_inventory(golden_model_environment.pristine) == (
        golden_model_environment.pristine_inventory
    )
    assert golden_model_environment.inventory == (
        golden_model_environment.pristine_inventory
    )
    compare_dir = wcs.compare_root(
        golden_model_environment.experiment_id,
        golden_model_environment.pristine / "diagnostics",
    )
    assert not compare_dir.exists()


@pytest.mark.compare_integration
def test_the_compare_guard_was_not_installed_during_the_golden_setup(
    golden_model_environment,
):
    """golden_fixture_setup_model_fit = allowed; the guard was demonstrably off."""
    assert golden_model_environment.compare_guard_active_during_setup is False
    assert not _compare_guard_is_installed()
    metadata = json.loads(
        wcs.model_metadata_path(
            golden_model_environment.experiment_id, golden_model_environment.out,
        ).read_text(encoding="utf-8")
    )
    assert metadata["fire_risk_model_fit"] is True
    assert metadata["bootstrap_run"] is True


@pytest.mark.compare_unit
def test_every_test_declares_exactly_one_compare_marker():
    for name, function in _module_tests():
        markers = _marker_names(function)
        selected = markers & {"compare_unit", "compare_integration"}
        assert len(selected) == 1, f"{name}: markers={sorted(markers)}"


@pytest.mark.compare_unit
def test_pure_unit_tests_never_request_the_model_fixture():
    """A `compare_unit` test may not touch the golden tree or a tmp filesystem."""
    forbidden = {"compare_env", "golden_model_environment", "tmp_path",
                 "tmp_path_factory"}
    for name, function in _module_tests():
        if "compare_unit" not in _marker_names(function):
            continue
        used = set(inspect.signature(function).parameters) & forbidden
        assert not used, f"{name} requests {sorted(used)}"


@pytest.mark.compare_unit
def test_no_test_in_this_module_is_skipped_or_xfailed():
    source = Path(__file__).read_text(encoding="utf-8")
    for token in ("mark." + "skip", "mark." + "xfail", "pytest." + "skip(",
                  "pytest." + "xfail(", "mark." + "skipif"):
        assert token not in source, token
    for name, function in _module_tests():
        markers = _marker_names(function)
        assert not (markers & {"skip", "skipif", "xfail"}), name


@pytest.mark.compare_unit
def test_the_published_compare_schema_and_layout_are_unchanged():
    """A test-fixture optimisation may not move the production output schema."""
    assert wcs.COMPARE_METADATA_SCHEMA == "window_closure_compare.v1"
    assert wcs.COMPARE_ROOT_DIR == "compare"
    assert wcs.COMPARE_METADATA_NAME == "compare_stage_metadata.json"
    assert wcs.COMPARE_DISPLAY_DECIMALS == 3
    assert wcs.compare_relative_layout() == FROZEN_COMPARE_LAYOUT


@pytest.mark.compare_unit
def test_the_production_bootstrap_contract_is_not_the_test_override():
    """The fast replicate count is a TEST-only override, never production."""
    from core.config import STEP8C_N_BOOTSTRAP

    assert _FAST_BOOTSTRAP == {"n_bootstrap": 24}
    assert int(STEP8C_N_BOOTSTRAP) != _FAST_BOOTSTRAP["n_bootstrap"]


def _module_tests():
    module = sys.modules[__name__]
    return sorted(
        (name, obj) for name, obj in vars(module).items()
        if name.startswith("test_") and inspect.isfunction(obj)
    )


def _marker_names(function) -> set:
    return {mark.name for mark in getattr(function, "pytestmark", [])}


@pytest.mark.compare_unit
def test_no_aoi_or_date_is_hard_coded_in_the_compare_code():
    import re

    REGISTRY_IDS = md.ld.REGISTRY_IDS
    _executable_string_literals = md.ld._executable_string_literals

    for module_path in (
        _PROJECT_ROOT / "src" / "window_closure_sensitivity.py",
        _PROJECT_ROOT / "scripts" / "validate_window_closure_compare.py",
    ):
        literals = _executable_string_literals(module_path)
        for experiment_id in REGISTRY_IDS:
            assert not [s for s in literals if experiment_id in s]
        assert [s for s in literals if re.search(r"(19|20)\d\d-\d\d-\d\d", s)] == []
