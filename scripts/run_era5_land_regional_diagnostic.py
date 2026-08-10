#!/usr/bin/env python3
"""CLI/runner for the ERA5-Land AOI-level regional meteorology diagnostic
(src/era5_land_regional_diagnostic.py).

Thin dispatcher only -- no scientific logic lives here. The cohort, the window
conversion, the derivation recipes, the area weighting, the climatology
arithmetic and every provenance rule live in the module.

    # plan only: no Earth Engine, no query, no directory, no file
    python scripts/run_era5_land_regional_diagnostic.py --dry-run

    # produce the artifact (opens an Earth Engine session)
    python scripts/run_era5_land_regional_diagnostic.py

The default cohort is the frozen five canonical AOIs; `mugla_2022` is NOT
included (provisional temporal contract awaiting supervisor decision) and can
only be added by naming it explicitly, which changes the analysis_id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.era5_land_regional_diagnostic import (  # noqa: E402
    DEFAULT_EXPERIMENTS,
    run_analysis,
)


def main(
    experiments: list[str] | None = None,
    dry_run: bool = False,
    output_root: str | Path | None = None,
    project: str | None = None,
    engine=None,
) -> dict:
    # `output_root` and `engine` are programmatic dependency-injection points
    # (tests, alternative namespaces); None means the canonical root and the
    # production Earth Engine engine.
    return run_analysis(
        experiments=experiments,
        dry_run=dry_run,
        engine=engine,
        output_root=Path(output_root) if output_root is not None else None,
        project=project,
    )


def render_plan(plan: dict) -> str:
    lines = [
        "ERA5-Land AOI-level regional meteorology diagnostic -- DRY RUN",
        "  (no Earth Engine session, no query, no directory, no file)",
        "",
        f"analysis_id     : {plan['analysis_id']}",
        f"collection      : {plan['collection']}",
        f"bands           : {', '.join(plan['bands'])}",
        f"climatology     : {plan['climatology_years']} (sample SD, ddof=1)",
        f"experiments     : {len(plan['experiment_ids'])} "
        f"-> {', '.join(plan['experiment_ids'])}",
        f"rows expected   : {plan['n_rows_expected']}",
        f"engine requests : {plan['n_engine_window_requests']} window(s)",
        f"csv columns     : {len(plan['summary_columns'])}",
        "",
    ]
    for entry in plan["experiments"]:
        west, south, east, north = entry["aoi_bbox_west_south_east_north"]
        lines.append(f"{entry['experiment_id']}  ({entry['display_name']})")
        lines.append(
            f"  region_key {entry['region_key']}  bbox "
            f"({west}, {south}, {east}, {north})"
        )
        for window, filters in entry["planned_gee_filter_windows"].items():
            observed = entry["observed_windows"][window]
            lines.append(
                f"  {window:<9} registry {observed['start_date']}.."
                f"{observed['end_date_inclusive']} (inclusive)  ->  GEE "
                f"[{filters['observed'][0]}, {filters['observed'][1]})"
            )
            for mapped in entry["climatology_windows"][window]:
                lines.append(
                    f"    climatology {mapped['reference_year']}: "
                    f"{mapped['start_date']}..{mapped['end_date_inclusive']} "
                    f"->  GEE [{mapped['start_date']}, "
                    f"{mapped['end_date_exclusive']})"
                )
        lines.append("")

    lines.append("planned output paths (NOT created):")
    for name, path in sorted(plan["planned_output_paths"].items()):
        lines.append(f"  {name:<34} {path}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ERA5-Land AOI-level regional meteorology diagnostic: per-AOI "
            "predictor/label window statistics of temperature, relative "
            "humidity, wind speed and precipitation, with 2017-2020 "
            "climatology anomalies. Explanatory only -- not a model predictor "
            "and not part of Step5/Step7/Step8/Step9/Step10. Produces a table; "
            "no raster is exported."
        ),
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help=(
            "Experiment IDs. Defaults to the frozen five canonical AOIs "
            f"({', '.join(DEFAULT_EXPERIMENTS)}). Naming any other experiment "
            "changes the scientific config and therefore the analysis_id."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Resolve and print the cohort, AOIs, predictor/label dates, mapped "
            "climatology windows, scientific config, analysis_id and planned "
            "output paths. Initialises no Earth Engine session, runs no query, "
            "creates no directory and writes no file."
        ),
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Override the outputs/ root (test/alternative namespace).",
    )
    parser.add_argument(
        "--project", default=None,
        help="Earth Engine project for the production engine (actual runs only).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the raw result/plan payload as JSON instead of a summary.",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = main(
        experiments=args.experiments,
        dry_run=args.dry_run,
        output_root=args.output_root,
        project=args.project,
    )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    if result.get("dry_run"):
        print(render_plan(result))
        return 0

    if result.get("already_complete"):
        print(
            "ERA5-Land regional diagnostic already complete -- nothing "
            "recomputed, nothing overwritten.\n"
            f"  analysis_id : {result['analysis_id']}\n"
            f"  namespace   : {result['output_root']}"
        )
        return 0

    print(
        "ERA5-Land regional diagnostic complete.\n"
        f"  analysis_id : {result['analysis_id']}\n"
        f"  experiments : {', '.join(result['experiment_ids'])}\n"
        f"  rows        : {result['n_rows']}\n"
        f"  namespace   : {result['output_root']}"
    )
    for name, path in sorted(result["output_paths"].items()):
        print(f"  {name:<34} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
