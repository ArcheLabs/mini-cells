from pathlib import Path

from minicells_fast001.common import atomic, digest, read, sha
from minicells_fast002.common import source

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


__all__ = ["BASE", "ID", "ROOT", "atomic", "config", "digest", "read", "sha", "source"]


def require_ci():
    import json
    import urllib.request

    head = source()
    url = (
        "https://api.github.com/repos/ArcheLabs/mini-cells/actions/workflows/clm002-fast002-r1.yml/runs?head_sha="
        + head
        + "&per_page=10"
    )
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "MiniCells-R1"}), timeout=20
    ) as response:
        runs = json.load(response)["workflow_runs"]
    matches = [r for r in runs if r["head_sha"] == head]
    if not matches or matches[0]["conclusion"] != "success":
        raise RuntimeError("R1_CURRENT_SOURCE_CI_REQUIRED_BEFORE_REAL_PARENT_ACCESS")
    return {
        "source_sha": head,
        "run_id": matches[0]["id"],
        "url": matches[0]["html_url"],
        "conclusion": "success",
    }
