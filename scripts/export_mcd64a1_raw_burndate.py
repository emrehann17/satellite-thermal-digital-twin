"""
export_mcd64a1_raw_burndate.py

Thin CLI wrapper for Step6's canonical raw MCD64A1 BurnDate export.

TASINDI (moved): asil GEE mantigi artik burada DEGIL,
`src/step6_validate_fire_relation.py:export_raw_mcd64a1_labels()`
icindedir. Step6 artik bu export'un TEK SAHIBIDIR; bu script yalnizca ince
bir CLI sarmalayicidir, boylece iki farkli/divergent implementasyon
OLUSMAZ.

WHY THIS EXISTS
---------------
Step8A needs the genuine MCD64A1 BurnDate band (day-of-year values 1..366),
not a binary burned mask. See `export_raw_mcd64a1_labels()` docstring in
Step6 for the full rationale.

Writes (defaults, same as before):
    outputs/validation/labels/mcd64a1_raw.tif      (raw BurnDate DOY)
    outputs/validation/labels/mcd64a1_burned.tif   (binary mask, if --also-binary)

CLI
---
    python scripts/export_mcd64a1_raw_burndate.py
    python scripts/export_mcd64a1_raw_burndate.py --also-binary
    python scripts/export_mcd64a1_raw_burndate.py --start 2023-08-01 --end 2023-10-31

Requires a working GEE environment (earthengine-api + geemap + auth), exactly
like Step6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import LABEL_END_DATE, LABEL_START_DATE, VALIDATION_LABEL_EXPORT_SCALE


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export raw MCD64A1 BurnDate DOY raster for Step8A "
        "(thin CLI wrapper around Step6's export_raw_mcd64a1_labels())."
    )
    parser.add_argument("--start", type=str, default=LABEL_START_DATE)
    parser.add_argument("--end", type=str, default=LABEL_END_DATE)
    parser.add_argument("--out", type=str, default=None, help="Override raw BurnDate output path.")
    parser.add_argument(
        "--also-binary", action="store_true",
        help="Also export the binary burned mask (BurnDate>0) to mcd64a1_burned.tif.",
    )
    parser.add_argument("--scale", type=int, default=VALIDATION_LABEL_EXPORT_SCALE)
    args = parser.parse_args(argv)

    try:
        from src.step6_validate_fire_relation import export_raw_mcd64a1_labels
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "src.step6_validate_fire_relation import edilemedi (ee/geemap/regions "
            f"eksik olabilir). Kurulum: pip install earthengine-api geemap; "
            f"auth: earthengine authenticate. Hata: {type(exc).__name__}: {exc}"
        )

    raw_out = Path(args.out) if args.out else None
    result = export_raw_mcd64a1_labels(
        start=args.start,
        end=args.end,
        also_binary=args.also_binary,
        raw_out=raw_out,
        scale=args.scale,
    )
    print(f"TAMAMLANDI. Raw BurnDate: {result['raw_path']}")
    if result.get("binary_path"):
        print(f"Binary mask: {result['binary_path']}")


if __name__ == "__main__":
    main()