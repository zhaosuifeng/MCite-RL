"""Shared path helpers for both python -m and direct script entry points."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from project_paths import PROJECT_ROOT, model_reference, project_path, relative_path, record_paths
