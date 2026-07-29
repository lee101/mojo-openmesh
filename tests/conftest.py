from __future__ import annotations

import importlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_ROOT = os.path.join(ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

import mojo_openmesh as mojo_om


def _upstream_openmesh():
    original = list(sys.path)
    try:
        sys.path[:] = [
            path
            for path in sys.path
            if os.path.abspath(path or os.curdir) != os.path.abspath(PYTHON_ROOT)
        ]
        sys.modules.pop("openmesh", None)
        return importlib.import_module("openmesh")
    finally:
        sys.path[:] = original


@pytest.fixture(scope="session")
def om():
    return mojo_om


@pytest.fixture(scope="session")
def upstream():
    return _upstream_openmesh()
