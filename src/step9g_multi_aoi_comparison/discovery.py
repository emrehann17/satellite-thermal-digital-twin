"""Pair-report discovery for the Step9G multi-experiment comparison.

Operates purely on a caller-supplied set of resolved experiment IDs; checks
both unordered directory orderings under the canonical Step9G v1 namespace.
No experiment ID is hard-coded.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional

import src.step9g_univariate_feature_auc_direction_reversal as step9g


def pair_report_root() -> Path:
    """`outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal/`
    -- derived generically from the v1 module's own path helper."""
    return step9g.output_root_for("a", "b").parent


def pair_report_path(experiment_a: str, experiment_b: str) -> Optional[Path]:
    """Whichever of the two possible directory orderings holds a canonical
    `step9g_final_report.json`, or None if neither exists."""
    root = pair_report_root()
    for a, b in ((experiment_a, experiment_b), (experiment_b, experiment_a)):
        candidate = root / f"{a}__{b}" / "step9g_final_report.json"
        if candidate.is_file():
            return candidate
    return None


def discover_pairs(resolved_ids: tuple[str, ...]) -> dict:
    """For every unordered pair among `resolved_ids`, resolve its canonical
    Step9G v1 report path, if present.

    Returns:
        {
          "available": {(a, b): Path, ...},   # a < b, sorted
          "missing": [(a, b), ...],           # a < b, sorted
        }
    """
    available: dict[tuple[str, str], Path] = {}
    missing: list[tuple[str, str]] = []
    for a, b in itertools.combinations(sorted(resolved_ids), 2):
        path = pair_report_path(a, b)
        if path is not None:
            available[(a, b)] = path
        else:
            missing.append((a, b))
    return {"available": available, "missing": missing}
