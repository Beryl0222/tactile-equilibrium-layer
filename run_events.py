#!/usr/bin/env python3
"""Alias for run_materials_events.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from equilibrium_layer.events import main

if __name__ == "__main__":
    raise SystemExit(main())
