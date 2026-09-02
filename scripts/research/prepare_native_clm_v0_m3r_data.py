"""Prepare a fresh pinned M3R snapshot using the already-validated M3 data builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FORMAT = "minicells.native-clm-v0.m3r-data-manifest.v1"
M3_FORMAT = "minicells.native-clm-v0.m3-data-manifest.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_cache(output: Path) -> dict | None:
    path = output / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("format") != FORMAT:
        return None
    for record in manifest.get("files", {}).values():
        file_path = output / record["path"]
        if not file_path.exists() or file_path.stat().st_size != int(record["bytes"]):
            return None
        if _sha256(file_path) != record["sha256"]:
            return None
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/native-clm-m3r-data"),
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    cached = _verified_cache(output)
    if cached is not None:
        print("Reusing verified Native CLM M3R data cache.", flush=True)
        print(json.dumps(cached, indent=2), flush=True)
        return 0

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    command = [
        sys.executable,
        "scripts/research/prepare_native_clm_v0_m3_data.py",
        "--output-dir",
        str(output),
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != M3_FORMAT:
        raise RuntimeError("underlying M3 data builder produced an unexpected manifest")
    manifest["format"] = FORMAT
    manifest["prepared_via"] = "scripts/research/prepare_native_clm_v0_m3_data.py"
    manifest["m3r_snapshot_role"] = "fresh matched snapshot for global-vs-lineage causal arms"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verified = _verified_cache(output)
    if verified is None:
        raise RuntimeError("M3R data cache failed post-write verification")
    print(json.dumps(verified, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
