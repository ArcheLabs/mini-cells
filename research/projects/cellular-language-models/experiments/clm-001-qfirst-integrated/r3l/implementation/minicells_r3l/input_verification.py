"""Read only historical result identity metadata, never R3k Holdout outcome values."""

import json
import urllib.request
from pathlib import Path
from minicells_r3e.common import atomic


def verify(c, output):
    commit = c["r3k_evidence_commit"]
    cached = Path(output) / "R3K_EVIDENCE_IDENTITY.json"
    if cached.exists():
        from minicells_r3e.common import read

        row = read(cached)
        if row["commit"] != commit or row["identity"]["source_sha"] != c["source_r3k"]:
            raise RuntimeError("R3K_EVIDENCE_CACHE_CHANGED")
        return
    api = f"https://api.github.com/repos/ArcheLabs/mini-cells/commits/{commit}"
    with urllib.request.urlopen(api, timeout=20) as stream:
        metadata = json.load(stream)
    paths = [
        r["filename"]
        for r in metadata["files"]
        if r["filename"].endswith("/RUN_MANIFEST.json")
        and "/r3k/reports/published/" in r["filename"]
    ]
    if len(paths) != 1:
        raise RuntimeError("R3K_EVIDENCE_MANIFEST_NOT_UNIQUE")
    url = f"https://raw.githubusercontent.com/ArcheLabs/mini-cells/{commit}/{paths[0]}"
    with urllib.request.urlopen(url, timeout=20) as stream:
        manifest = json.load(stream)
    actual = manifest.get("identity", manifest)
    if (
        actual["source_sha"] != c["source_r3k"]
        or actual["run_id"] != "cf4774dd2af77d9c062831f7cdef64caea24d1ab43c753e29315ca5a0494e758"
    ):
        raise RuntimeError("HISTORICAL_R3K_EVIDENCE_IDENTITY_CHANGED")
    atomic(
        cached,
        dict(
            commit=commit,
            path=paths[0],
            identity=actual,
            scope="IDENTITY_ONLY_NO_HOLDOUT_VALUES_USED",
        ),
    )
