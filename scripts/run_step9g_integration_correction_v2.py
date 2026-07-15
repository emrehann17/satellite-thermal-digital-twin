"""Single runner for the Step9G report-integration correction (v2). No
competing entry points."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.step9g_integration_correction_v2 import cli

if __name__ == "__main__":
    raise SystemExit(cli())