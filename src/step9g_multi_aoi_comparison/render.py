"""Write JSON/CSV/Markdown outputs for the generic multi-experiment Step9G
comparison. Pure rendering only -- no scientific computation, no AUC/CI
values are altered here."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .build import NUMERIC_FEATURES, ADVISOR_CRITICAL_FEATURE

DIRECTION_ARROWS = {
    "higher_values_rank_burned": "↑",  # ↑
    "lower_values_rank_burned": "↓",  # ↓
}

LIMITATIONS = (
    "Connected/pairwise reversal findings are descriptive diagnostics; they "
    "do not establish causality and do not prove that concept/relationship "
    "shift is the only source of cross-region transfer failure.",
    "A point-estimate direction reversal whose 95% spatial-block bootstrap "
    "interval includes 0.5 is uncertain and must never be reported as "
    "bootstrap-supported.",
    "This synthesis recomputes no AUC, bootstrap replicate, or reversal "
    "classification; it only reads and cross-validates existing canonical "
    "Step9G pair reports.",
    "landcover_dominant is excluded from scalar AUC because its integer "
    "class codes have no scientifically meaningful ordering.",
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_cell(record: dict[str, Any]) -> str:
    """`0.611 [0.532, 0.690] ↑*` -- arrow shows ranking direction, `*`
    marks a 95% spatial-block bootstrap interval that excludes 0.5."""
    if record is None or record.get("auc") is None:
        return "n/a"
    arrow = DIRECTION_ARROWS.get(record.get("direction"), "")
    star = "*" if record.get("interval_excludes_chance") else ""
    return f"{_fmt(record['auc'])} [{_fmt(record['ci_low'])}, {_fmt(record['ci_high'])}] {arrow}{star}"


def render_markdown(result: dict[str, Any]) -> str:
    sorted_ids = result["resolved_experiment_ids"]
    wide_rows = result["wide_rows"]
    advisor = result["advisor_critical"]
    pairwise_findings = result["pairwise_findings"]
    manifest = result["manifest"]

    lines = [
        "# Multi-AOI Step9G univariate-AUC comparison",
        "",
        f"analysis_id: `{manifest['analysis_id']}` (order-invariant)  ",
        f"created_at: {manifest['created_at']}  ",
        f"requested_experiment_ids: {result['requested_experiment_ids']}  ",
        f"resolved_experiment_ids: {sorted_ids}  ",
        f"complete_pairwise_matrix: {result['complete_pairwise_matrix']}",
        "",
        "**Legend**: cell = `AUC [CI low, CI high] direction*` -- "
        "↑ higher values rank burned; ↓ lower values rank burned; "
        "`*` 95% spatial-block bootstrap interval excludes 0.5.",
        "",
    ]

    if result["missing_pairs"]:
        lines.append("**Unavailable pair reports (recorded, not fabricated):**")
        for pair_id in result["missing_pairs"]:
            lines.append(f"- {pair_id}")
        lines.append("")

    header = ["feature"] + [eid for eid in sorted_ids]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    region_by_feature: dict[str, dict[str, dict[str, Any]]] = {f: {} for f in NUMERIC_FEATURES}
    for row in result["long_rows"]:
        region_by_feature[row["feature"]][row["experiment_id"]] = row
    for feature in NUMERIC_FEATURES:
        cells = [format_cell(region_by_feature[feature].get(eid)) for eid in sorted_ids]
        lines.append("| " + " | ".join([feature] + cells) + " |")
    lines.append("")

    lines.append("## Advisor-critical elevation result")
    lines.append("")
    for row in advisor["rows"]:
        lines.append(
            f"- {row['experiment_id']}: AUC {row['auc']}, CI [{row['ci_low']}, {row['ci_high']}], "
            f"{'higher' if row['direction'] == 'higher_values_rank_burned' else 'lower' if row['direction'] == 'lower_values_rank_burned' else row['direction']} "
            f"values rank burned, {row['support_status']}"
        )
    lines.append("")
    higher = advisor["bootstrap_supported_higher"]
    lower = advisor["bootstrap_supported_lower"]
    if higher and lower:
        lines.append(
            f"{' and '.join(higher)} share a bootstrap-supported positive {advisor['feature']} "
            f"direction, whereas {' and '.join(lower)} "
            f"{'has' if len(lower) == 1 else 'have'} a bootstrap-supported negative direction."
        )
    elif higher:
        lines.append(f"{' and '.join(higher)} share a bootstrap-supported positive {advisor['feature']} direction.")
    elif lower:
        lines.append(f"{' and '.join(lower)} share a bootstrap-supported negative {advisor['feature']} direction.")
    else:
        lines.append(f"No region shows a bootstrap-supported {advisor['feature']} direction in this comparison.")
    lines.append(
        "This descriptive cross-region contrast does not prove that this "
        "feature difference causally explains any cross-region transfer "
        "performance difference."
    )
    lines.append("")

    lines.append("## Pairwise bootstrap-supported direction reversals")
    lines.append("")
    for pair_id, features in sorted(pairwise_findings.items()):
        a, b = pair_id.split("__", 1)
        if features:
            noun = "is a bootstrap-supported direction reversal" if len(features) == 1 else "are bootstrap-supported direction reversals"
            lines.append(f"- {a}–{b}: {', '.join(features)} {noun}.")
        else:
            lines.append(f"- {a}–{b}: no feature has a bootstrap-supported direction reversal.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines += [f"- {line}" for line in LIMITATIONS]
    lines.append("")
    return "\n".join(lines)


def write_outputs(result: dict[str, Any]) -> dict[str, str]:
    output_dir: Path = result["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_json = {
        "analysis_id": result["manifest"]["analysis_id"],
        "created_at": result["manifest"]["created_at"],
        "resolved_experiment_ids": result["resolved_experiment_ids"],
        "available_pairs": result["available_pairs"],
        "missing_pairs": result["missing_pairs"],
        "complete_pairwise_matrix": result["complete_pairwise_matrix"],
        "long_rows": result["long_rows"],
        "wide_rows": result["wide_rows"],
        "pairwise_rows": result["pairwise_rows"],
        "advisor_critical": result["advisor_critical"],
        "pairwise_findings": result["pairwise_findings"],
        "scientific_contract": result["manifest"]["scientific_contract"],
        "limitations": list(LIMITATIONS),
    }

    paths: dict[str, str] = {}

    p = output_dir / "multi_aoi_univariate_auc_comparison.json"
    p.write_text(json.dumps(comparison_json, indent=2, default=str) + "\n", encoding="utf-8")
    paths["comparison_json"] = str(p)

    p = output_dir / "multi_aoi_univariate_auc_long.csv"
    pd.DataFrame(result["long_rows"]).to_csv(p, index=False)
    paths["long_csv"] = str(p)

    p = output_dir / "multi_aoi_univariate_auc_wide.csv"
    pd.DataFrame(result["wide_rows"]).to_csv(p, index=False)
    paths["wide_csv"] = str(p)

    p = output_dir / "pairwise_direction_reversal_summary.csv"
    pd.DataFrame(result["pairwise_rows"]).to_csv(p, index=False)
    paths["pairwise_csv"] = str(p)

    p = output_dir / "multi_aoi_univariate_auc_comparison.md"
    p.write_text(render_markdown(result), encoding="utf-8")
    paths["comparison_md"] = str(p)

    p = output_dir / "manifest.json"
    p.write_text(json.dumps(result["manifest"], indent=2, default=str) + "\n", encoding="utf-8")
    paths["manifest_json"] = str(p)

    return paths
