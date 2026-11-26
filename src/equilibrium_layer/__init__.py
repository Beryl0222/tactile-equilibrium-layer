"""Src-layout compatibility re-export of the root implementation."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from equilibrium_layer import *  # noqa: E402,F401,F403
from equilibrium_layer import __all__, __version__  # noqa: E402,F401
