"""Pytest bootstrap: make tests/fakes.py importable from subdirectories (§15).

tests/fakes.py lives at tests/ level while unit/integration tests live in
subdirectories; with pytest's default import mode those subdirectories — not
tests/ itself — go on sys.path. Adding tests/ here lets every test import the
two fake backends as `fakes` (the suite's only mocks).
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
