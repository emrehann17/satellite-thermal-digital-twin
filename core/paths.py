from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_project_root_on_path() -> None:
    """Make repo-root imports work when files are run directly."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
