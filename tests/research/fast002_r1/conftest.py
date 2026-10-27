import runpy
from pathlib import Path

BASE = (
    Path(__file__).resolve().parents[3]
    / "research/projects/cellular-language-models/experiments/clm-002-fast-002-r1-stateful-calibration"
)
runpy.run_path(str(BASE / "run.py"), run_name="r1_tests")["bootstrap"]()

import os

import pytest


@pytest.fixture(autouse=True)
def synthetic_fixture_avoids_disk_durability_stress(monkeypatch):
    # These tests prove math/state identity, not the OS fsync implementation.
    # Production durability calls are untouched; the existing store owns that contract.
    monkeypatch.setattr(os, "fsync", lambda fd: None)
