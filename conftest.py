"""Pytest path setup.

This file makes the repository root importable even when pytest is launched as
a console script rather than ``python -m pytest``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
