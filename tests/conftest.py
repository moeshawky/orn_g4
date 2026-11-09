"""Pytest configuration: make the Azrael-g4 package importable."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Azrael-g4/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
