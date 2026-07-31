#!/usr/bin/env python3
"""CLI runner for the frozen CORAL lambda sensitivity diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.coral_lambda_sensitivity import STAGES, run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", choices=STAGES, default="plan")
    parser.add_argument("--to-stage", choices=STAGES, default="summarize")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--experiments-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None, **kwargs):
    if kwargs:
        return run_analysis(**kwargs)
    args = build_parser().parse_args(argv)
    return run_analysis(from_stage=args.from_stage, to_stage=args.to_stage, dry_run=args.dry_run,
                        resume=args.resume, force=args.force, output_root=args.output_root,
                        experiments_root=args.experiments_root)


if __name__ == "__main__":
    main()
