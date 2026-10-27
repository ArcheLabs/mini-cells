from pathlib import Path

from minicells_fast001.common import digest, read, sha
from minicells_fast001.common import atomic as atomic
from minicells_fast002.common import source as source

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[4]
ID = "CLM-002-FAST-002-R1-STATEFUL-CALIBRATION"


def config():
    c = read(BASE / "config/registered.json")
    if sha(c) != (BASE / "config/registered.sha256").read_text().strip():
        raise RuntimeError("R1_CONFIG_IDENTITY_CHANGED")
    if digest(BASE / "PROTOCOL_AMENDMENT.md") != c["protocol_sha256"]:
        raise RuntimeError("R1_PROTOCOL_IDENTITY_CHANGED")
    return c
