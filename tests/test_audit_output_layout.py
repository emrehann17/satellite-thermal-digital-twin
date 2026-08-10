"""Output-layout contract for the burned-pattern and domain-classifier audits.

Both families used to hardcode a single flat output root with no override, so
preserving a previous result meant renaming the whole root by hand and a second
analysis scope silently wrote into the first one's root. These tests pin the
replacement contract: one canonical root per family, one path resolver, and a
namespace per scope UNDER that root.

Pure path arithmetic -- nothing here fits a model, reads Step8A, or writes into
a real output namespace.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import src.burned_pattern_audit as bpa
import src.domain_classifier_audit as dca

FAMILIES = (
    pytest.param(bpa, "burned_pattern_audit", "experiments", id="burned_pattern"),
    pytest.param(dca, "domain_classifier_audit", "pairs", id="domain_classifier"),
)


def _leaf(layout, leaf_name: str) -> Path:
    return getattr(layout, leaf_name)


# =============================================================================
# One canonical root per family
# =============================================================================
@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_default_output_root_is_deterministic_and_canonical(module, family, leaf):
    from core.paths import PROJECT_ROOT

    expected = Path(PROJECT_ROOT) / "outputs" / "diagnostics" / family
    assert module.OUTPUT_ROOT == expected
    # Deterministic: resolving twice yields the same paths.
    assert module.resolve_layout() == module.resolve_layout()


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_default_layout_is_the_legacy_flat_layout(module, family, leaf):
    """No-argument resolution must not move existing programmatic callers."""
    layout = module.resolve_layout()
    assert layout.root == module.OUTPUT_ROOT
    assert layout.comparison == module.COMPARISON_OUTPUT_DIR
    assert _leaf(layout, leaf) == getattr(
        module, "EXPERIMENTS_OUTPUT_ROOT" if leaf == "experiments" else "PAIRS_OUTPUT_ROOT"
    )
    assert layout.scope is None


# =============================================================================
# --output-root override
# =============================================================================
@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_output_root_override_relocates_the_whole_family(module, family, leaf, tmp_path):
    layout = module.resolve_layout(output_root=tmp_path / "elsewhere")
    assert layout.root == tmp_path / "elsewhere"
    assert _leaf(layout, leaf) == tmp_path / "elsewhere" / leaf
    assert layout.comparison == tmp_path / "elsewhere" / "comparison"
    for path in (layout.root, _leaf(layout, leaf), layout.comparison):
        assert str(path).startswith(str(tmp_path))
        assert str(module.OUTPUT_ROOT) not in str(path)


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_output_root_and_scope_compose(module, family, leaf, tmp_path):
    layout = module.resolve_layout(output_root=tmp_path / "r", scope="some_scope")
    assert _leaf(layout, leaf) == tmp_path / "r" / "some_scope" / leaf
    assert layout.comparison == tmp_path / "r" / "some_scope" / "comparison"
    assert layout.scope == "some_scope"


# =============================================================================
# Scopes get a namespace UNDER the canonical root, and never collide
# =============================================================================
@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_scope_namespace_lives_under_the_canonical_root_not_beside_it(module, family, leaf):
    layout = module.resolve_layout(scope="all_enabled")
    assert module.OUTPUT_ROOT in _leaf(layout, leaf).parents
    assert module.OUTPUT_ROOT in layout.comparison.parents
    # The defect being fixed: a sibling root next to the canonical one.
    assert not str(_leaf(layout, leaf)).startswith(str(module.OUTPUT_ROOT) + "_")


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_different_scopes_do_not_collide(module, family, leaf, tmp_path):
    a = module.resolve_layout(output_root=tmp_path, scope="all_enabled")
    b = module.resolve_layout(output_root=tmp_path, scope="mugla_2021__mugla_2022_event_relative")
    assert _leaf(a, leaf) != _leaf(b, leaf)
    assert a.comparison != b.comparison
    assert not str(_leaf(a, leaf)).startswith(str(_leaf(b, leaf)))
    assert not str(_leaf(b, leaf)).startswith(str(_leaf(a, leaf)))


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_same_scope_resolves_to_the_same_namespace(module, family, leaf, tmp_path):
    a = module.resolve_layout(output_root=tmp_path, scope="all_enabled")
    b = module.resolve_layout(output_root=tmp_path, scope="all_enabled")
    assert a == b


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_empty_scope_opts_back_into_the_flat_layout(module, family, leaf, tmp_path):
    layout = module.resolve_layout(output_root=tmp_path, scope="")
    assert _leaf(layout, leaf) == tmp_path / leaf
    assert layout.comparison == tmp_path / "comparison"


# =============================================================================
# Scope derivation from the selection
# =============================================================================
def _resolution(module, ids, mode):
    from src.burned_pattern_audit import ExperimentResolution

    return ExperimentResolution(
        requested_ids=tuple(ids), resolved_ids=tuple(ids), selection_mode=mode,
    )


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_all_enabled_scope_key_is_stable(module, family, leaf):
    resolution = _resolution(module, ("b", "a", "c"), "all_enabled")
    assert module.scope_key(resolution) == "all_enabled"


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_explicit_scope_key_is_order_independent(module, family, leaf):
    forward = _resolution(module, ("mugla_2021", "mugla_2022_event_relative"), "explicit")
    reverse = _resolution(module, ("mugla_2022_event_relative", "mugla_2021"), "explicit")
    assert module.scope_key(forward) == module.scope_key(reverse)
    assert module.scope_key(forward) == "mugla_2021__mugla_2022_event_relative"


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_distinct_selections_produce_distinct_scope_keys(module, family, leaf):
    five = _resolution(module, ("a", "b", "c", "d", "e"), "all_enabled")
    pair = _resolution(module, ("mugla_2021", "mugla_2022_event_relative"), "explicit")
    assert module.scope_key(five) != module.scope_key(pair)


# =============================================================================
# A redirected run must never escape into the real canonical root
# =============================================================================
@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_scoped_layout_honours_a_redirected_leaf_constant(module, family, leaf, tmp_path, monkeypatch):
    """Callers redirect this analysis by patching the leaf constants.

    Deriving the scope namespace from OUTPUT_ROOT instead would silently write
    into the real canonical root; this pins the sandbox-safe derivation.
    """
    leaf_constant = "EXPERIMENTS_OUTPUT_ROOT" if leaf == "experiments" else "PAIRS_OUTPUT_ROOT"
    monkeypatch.setattr(module, leaf_constant, tmp_path / "sandbox" / leaf)
    monkeypatch.setattr(module, "COMPARISON_OUTPUT_DIR", tmp_path / "sandbox" / "comparison")

    layout = module.resolve_layout(scope="all_enabled")
    assert str(_leaf(layout, leaf)).startswith(str(tmp_path))
    assert str(layout.comparison).startswith(str(tmp_path))
    assert str(module.OUTPUT_ROOT) not in str(_leaf(layout, leaf))


# =============================================================================
# Runners expose the override and duplicate no path string
# =============================================================================
@pytest.mark.parametrize("runner_module,expected", [
    ("scripts.run_burned_pattern_audit", "burned_pattern_audit"),
    ("scripts.run_domain_classifier_audit", "domain_classifier_audit"),
])
def test_runner_exposes_output_root_and_scope(runner_module, expected):
    import importlib

    runner = importlib.import_module(runner_module)
    dests = {a.dest for a in runner.build_parser()._actions}
    assert "output_root" in dests
    assert "scope" in dests
    args = runner.build_parser().parse_args(["--all-enabled"])
    assert args.output_root is None      # default: canonical root
    assert args.scope is None            # default: derived from the selection


@pytest.mark.parametrize("runner_path,module", [
    ("scripts/run_burned_pattern_audit.py", bpa),
    ("scripts/run_domain_classifier_audit.py", dca),
])
def test_runner_hardcodes_no_output_path(runner_path, module):
    """The path lives in exactly one place: the module's resolver.

    Documentation strings (`help=`, `description=`) may name the default root;
    what must not exist is a path CONSTRUCTED in the runner, which is how the
    two families ended up with divergent path logic in the first place.
    """
    import ast

    tree = ast.parse(Path(runner_path).read_text(encoding="utf-8"))
    documented = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in {"help", "description"}:
            for inner in ast.walk(node.value):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    documented.add(id(inner))
    offenders = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in documented
        and ("outputs/diagnostics" in node.value or "outputs" == node.value)
    ]
    assert not offenders, offenders


# =============================================================================
# Overwrite protection is untouched by the layout change
# =============================================================================
@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_existing_output_from_another_analysis_id_still_fails_closed(module, family, leaf, tmp_path):
    import json

    target = tmp_path / "existing"
    target.mkdir()
    (target / "manifest.json").write_text(json.dumps({"analysis_id": "a" * 64}))
    with pytest.raises(SystemExit):
        module._guard_force(target, "b" * 64, False, label="test")
    # force=True is still the only way through, and a matching id is fine.
    module._guard_force(target, "b" * 64, True, label="test")
    module._guard_force(target, "a" * 64, False, label="test")


# =============================================================================
# No scientific configuration moved
# =============================================================================
def test_layout_change_touched_no_scientific_constant():
    assert bpa.POPULATIONS
    assert dca.OUTPUT_ROOT.name == "domain_classifier_audit"
    assert bpa.OUTPUT_ROOT.name == "burned_pattern_audit"
    # The two families remain separate roots, not nested in one another.
    assert bpa.OUTPUT_ROOT != dca.OUTPUT_ROOT
    assert bpa.OUTPUT_ROOT not in dca.OUTPUT_ROOT.parents
    assert dca.OUTPUT_ROOT not in bpa.OUTPUT_ROOT.parents


# =============================================================================
# Scope normalisation: the canonical cohort has ONE name, however it is asked for
# =============================================================================
CANONICAL_FIVE = (
    "bejis_2022", "evia_2021_extended", "manavgat_2021", "montiferru_2021", "mugla_2021",
)


def test_canonical_cohort_ids_matches_all_enabled_resolution():
    resolved = bpa.resolve_experiments(all_enabled=True).resolved_ids
    assert bpa.canonical_cohort_ids() == tuple(sorted(resolved))
    assert bpa.canonical_cohort_ids() == CANONICAL_FIVE


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_all_enabled_selection_is_named_all_enabled(module, family, leaf):
    resolution = module.resolve_experiments(all_enabled=True)
    assert module.scope_key(resolution) == "all_enabled"


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_explicit_canonical_five_normalises_to_all_enabled(module, family, leaf):
    """Naming the cohort explicitly must not mint a second namespace."""
    resolution = module.resolve_experiments(experiments=list(CANONICAL_FIVE))
    assert resolution.selection_mode != "all_enabled"        # requested explicitly
    assert module.scope_key(resolution) == "all_enabled"     # but named canonically


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_reordered_canonical_five_normalises_to_all_enabled(module, family, leaf):
    forward = module.resolve_experiments(experiments=list(CANONICAL_FIVE))
    reverse = module.resolve_experiments(experiments=list(reversed(CANONICAL_FIVE)))
    assert module.scope_key(forward) == module.scope_key(reverse) == "all_enabled"


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_a_real_subset_keeps_its_own_scope(module, family, leaf):
    subset = list(CANONICAL_FIVE[:2])
    resolution = module.resolve_experiments(experiments=subset)
    key = module.scope_key(resolution)
    assert key != "all_enabled"
    assert key == "__".join(sorted(subset))


def test_event_relative_pair_keeps_its_own_scope():
    resolution = bpa.resolve_experiments(
        experiments=["mugla_2021", "mugla_2022_event_relative"],
    )
    assert bpa.scope_key(resolution) == "mugla_2021__mugla_2022_event_relative"


def test_canonical_cohort_excludes_superseded_and_non_cohort_experiments():
    cohort = set(bpa.canonical_cohort_ids())
    for excluded in ("evia_2021", "kozan_2023", "mugla_2022", "mugla_2022_event_relative"):
        assert excluded not in cohort, excluded


def test_superseded_sensitivity_selection_is_never_named_all_enabled():
    """A superseded AOI alongside its successor is its own scope, not the cohort."""
    resolution = bpa.resolve_experiments(
        experiments=["evia_2021", "evia_2021_extended"],
        allow_superseded_sensitivity=True,
    )
    assert "evia_2021" in resolution.resolved_ids
    assert bpa.scope_key(resolution) != "all_enabled"
    assert bpa.scope_key(resolution) == "evia_2021__evia_2021_extended"


@pytest.mark.parametrize("module,family,leaf", FAMILIES)
def test_normalised_scope_still_honours_output_root_override(module, family, leaf, tmp_path):
    """Normalisation must not bypass the sandbox."""
    resolution = module.resolve_experiments(experiments=list(CANONICAL_FIVE))
    layout = module.resolve_layout(output_root=tmp_path, scope=module.scope_key(resolution))
    assert _leaf(layout, leaf) == tmp_path / "all_enabled" / leaf
    assert str(_leaf(layout, leaf)).startswith(str(tmp_path))
    assert str(module.OUTPUT_ROOT) not in str(_leaf(layout, leaf))


def test_both_families_share_one_canonical_cohort_rule():
    """The rule lives in one place so the two families cannot drift apart."""
    resolution = bpa.resolve_experiments(experiments=list(CANONICAL_FIVE))
    assert dca.scope_key(resolution) == bpa.scope_key(resolution) == "all_enabled"


def test_scope_key_falls_back_when_the_cohort_cannot_be_resolved(monkeypatch):
    """An unresolvable registry must not silently mislabel a selection."""
    monkeypatch.setattr(bpa, "canonical_cohort_ids", lambda: ())
    resolution = bpa.resolve_experiments(experiments=list(CANONICAL_FIVE))
    assert bpa.scope_key(resolution) == "__".join(sorted(CANONICAL_FIVE))
