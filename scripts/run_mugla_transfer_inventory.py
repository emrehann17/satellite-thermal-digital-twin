#!/usr/bin/env python3
"""READ-ONLY output inventory for the Muglas 2021 <-> 2022 transfer pair.

Discovers the frozen artefacts that already exist locally for both directed
transfers, verifies the two frozen Step8A parquets against the manuscript
author's expected SHA-256 digests, checks that the frozen Step9 provenance
binds to exactly those digests, and compares the frozen ROC-AUC point
estimates and 95% target spatial-block bootstrap CIs against the author's
reference values.

Fits no model, draws no bootstrap, and writes only into
`outputs/diagnostics/mugla_transfer_inventory/`.

    python scripts/run_mugla_transfer_inventory.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reproduction_validation.common import relative_to_root
from src.reproduction_validation.mugla_inventory import build_inventory, render_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_inventory(output_root=args.output_root)

    report_path = PROJECT_ROOT / payload["_report_path"]
    markdown_path = report_path.with_suffix(".md")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    print("=" * 78)
    print(f"MUGLA_OUTPUT_STATUS  = {payload['mugla_output_status']}")
    print(f"MUGLA_RERUN_REQUIRED = {payload['mugla_rerun_required']}")
    print(f"directions observed  : {payload['observed_directions']}")
    for experiment_id, record in payload["step8a_input_verification"]["records"].items():
        print(f"step8a {experiment_id:26s} hash match = {record['match']}")
    for direction, values in payload["local_results"].items():
        print(f"\n{direction}")
        for metric in ("baseline_roc_auc", "thermal_roc_auc", "delta_roc_auc"):
            low, high = values.get(f"{metric}_ci_95", [None, None])
            print(
                f"  {metric:<18} = {values[metric]:.6f}   "
                f"95% CI [{low:.6f}, {high:.6f}]   "
                f"bootstrap mean {values.get(f'{metric}_bootstrap_mean'):.6f}"
            )
    print(f"\nreport   : {payload['_report_path']}")
    print(f"markdown : {relative_to_root(markdown_path)}")
    if payload["problems"]:
        print("PROBLEMS:")
        for problem in payload["problems"]:
            print(f"  - {problem}")
    print("=" * 78)
    return 0 if payload["mugla_output_status"] == "EXISTING_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
