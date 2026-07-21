#!/usr/bin/env python3
"""CLI/runner for read-only earliest-stage seam localization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_context import build_experiment_context
from core.seam_localization_config import seam_localization_config
from src.seam_localization import (
    manual_boundary_feature, run_localization, write_localization,
)


def _parse_manual_line(value: str) -> list[list[float]]:
    try:
        points = [
            [float(number) for number in point.split(",")]
            for point in value.split(";")
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--manual-line must be 'x1,y1;x2,y2[;x3,y3...]'",
        ) from exc
    if len(points) < 2 or any(len(point) != 2 for point in points):
        raise argparse.ArgumentTypeError(
            "--manual-line must contain at least two x,y coordinate pairs",
        )
    return points


def main(
    experiment_id: str,
    dry_run: bool = False,
    force: bool = False,
    manual_boundaries: list[str] | str | None = None,
    manual_lines: list[str] | None = None,
    manual_crs: str = "EPSG:4326",
) -> dict:
    ctx = build_experiment_context(experiment_id)
    config = seam_localization_config(experiment_id)
    manual = (
        [manual_boundaries]
        if isinstance(manual_boundaries, str)
        else (manual_boundaries or [])
    )
    parsed_lines = [_parse_manual_line(value) for value in (manual_lines or [])]
    inline = [
        manual_boundary_feature(coordinates, manual_crs)
        for coordinates in parsed_lines
    ]
    output = Path(ctx["output_root"]) / "qa" / "seam_localization" / "v1"
    plan = {
        "experiment_id": experiment_id,
        "output_dir": str(output),
        "provenance_boundaries": str(
            Path(ctx["output_root"]) / "qa" / "source_scene_provenance"
            / "v1" / "scene_boundaries.geojson"
        ),
        "manual_boundaries": manual,
        "inline_manual_boundary_count": len(inline),
        "manual_crs": manual_crs,
        "artifact_families": config["artifact_families"],
        "audit_scales": config["audit_scales"],
        "read_only": True,
        "gee_submission_started": False,
        "model_training_started": False,
    }
    if not config.get("enabled", True):
        return {"ran": False, "skipped": True, "reason": "disabled", "plan": plan}
    if dry_run:
        return {"ran": False, "dry_run": True, "plan": plan}
    result = run_localization(
        ctx, config, [Path(path) for path in manual], inline,
    )
    return write_localization(result, output, force)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--manual-boundary", "--manual-boundaries",
        dest="manual_boundaries", action="append", default=[],
        help="Diagnostic GeoJSON LineString input; repeatable.",
    )
    parser.add_argument(
        "--manual-line", action="append", default=[],
        help="Inline diagnostic line: 'x1,y1;x2,y2[;x3,y3...]'.",
    )
    parser.add_argument("--manual-crs", default="EPSG:4326")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(main(
        args.experiment, args.dry_run, args.force, args.manual_boundaries,
        args.manual_line, args.manual_crs,
    ), indent=2, default=str))

