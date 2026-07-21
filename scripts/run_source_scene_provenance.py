#!/usr/bin/env python3
"""CLI/runner for local source-scene provenance; never submits GEE work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_context import build_experiment_context
from core.source_scene_provenance_config import source_scene_provenance_config
from src.source_scene_provenance import (
    build_provenance, provider_for_context, write_provenance,
)


VALID_MODES = {"disabled", "metadata_only", "pixel_provenance"}


def main(
    experiment_id: str,
    dry_run: bool = False,
    force: bool = False,
    mode: str | None = None,
    export_plan_only: bool = False,
) -> dict:
    ctx = build_experiment_context(experiment_id)
    config = source_scene_provenance_config(experiment_id)
    if mode is not None:
        config["mode"] = mode
    selected_mode = config.get("mode", "metadata_only")
    if selected_mode not in VALID_MODES:
        raise ValueError(
            f"Unsupported provenance mode: {selected_mode}; expected {sorted(VALID_MODES)}",
        )
    output_dir = (
        Path(ctx["output_root"]) / "qa" / "source_scene_provenance" / "v1"
    )
    provider = provider_for_context(ctx)
    plan = {
        "experiment_id": experiment_id,
        "mode": selected_mode,
        "output_dir": str(output_dir),
        "provider": provider.provider_name,
        "metadata_candidates": [str(path) for path in provider.metadata_paths()],
        "metadata_available": [
            str(path) for path in provider.resolve_source_metadata()
        ],
        "gee_submission_started": False,
        "metadata_only": selected_mode == "metadata_only",
        "pixel_provenance": (
            "plan_only" if selected_mode == "pixel_provenance" else "not_requested"
        ),
        "read_only": True,
    }
    if selected_mode == "disabled":
        return {"ran": False, "skipped": True, "reason": "disabled", "plan": plan}
    if dry_run or export_plan_only:
        return {
            "ran": False,
            "dry_run": dry_run,
            "export_plan_only": export_plan_only,
            "plan": plan,
        }
    return write_provenance(
        build_provenance(ctx, config), output_dir, force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--mode", choices=sorted(VALID_MODES), default=None,
    )
    parser.add_argument(
        "--export-plan-only", action="store_true",
        help="Resolve and print the pixel-provenance plan; never submit a GEE task.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(main(
        args.experiment, args.dry_run, args.force, args.mode,
        args.export_plan_only,
    ), indent=2, default=str))

