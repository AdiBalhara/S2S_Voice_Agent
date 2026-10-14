"""Agent package for the voice sales agent."""

import sys
from pathlib import Path

# Ensure project root is in sys.path when loaded via `adk web` or other tooling
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
