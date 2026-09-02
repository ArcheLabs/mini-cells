"""Prepare the pinned M3L-2 stream plus an explicit pre-continual A bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

FORMAT = "minicells.native-clm-v0.m3l2-data-manifest.v1"
PARENT_FORMAT = "minicells.native-clm-v0.m3r-data-manifest.v1"
A_REPO = "roneneldan/TinyStories"
A_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid(output: Path) -> dict | None:
    p = output / "manifest.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if m.get("format") != FORMAT:
        return None
    for record in m.get("files", {}).values():
        fp = output / record["path"]
        if not fp.exists() or fp.stat().st_size != int(record["bytes"]) or _sha256(fp) != record["sha256"]:
            return None
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/native-clm-m3l2-data"))
    ap.add_argument("--a-bootstrap-docs", type=int, default=10000)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cached = _valid(out)
    if cached is not None:
        print(json.dumps(cached, indent=2), flush=True)
        return 0

    subprocess.run([sys.executable, "scripts/research/prepare_native_clm_v0_m3r_data.py", "--output-dir", str(out)], check=True)
    parent = json.loads((out / "manifest.json").read_text())
    if parent.get("format") != PARENT_FORMAT:
        raise RuntimeError("unexpected parent M3R manifest")
    resolved = parent["dataset_revisions"]["A"]["resolved_revision"]
    if resolved != A_REVISION:
        raise RuntimeError(f"TinyStories revision drift: {resolved}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("missing HF_TOKEN")
    from datasets import load_dataset

    stream = load_dataset(A_REPO, split="train", streaming=True, token=token, revision=A_REVISION)
    path = out / "A-tinystories-bootstrap.txt"
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in stream:
            if count >= args.a_bootstrap_docs:
                break
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            handle.write(text + "\n\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    if count != args.a_bootstrap_docs:
        raise RuntimeError(f"A bootstrap yielded {count}/{args.a_bootstrap_docs} documents")

    files = dict(parent["files"])
    files["A_bootstrap"] = {
        "path": path.name,
        "documents": count,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "dataset_split": "train",
    }
    manifest = {
        **parent,
        "format": FORMAT,
        "files": files,
        "bootstrap": {
            "A_bootstrap": "pre-continual address-state construction only",
            "dataset_split": "train",
            "separate_from_A_eval_split": True,
            "access_after_continual_start": False,
            "native_clm_parameter_updates": False,
        },
        "learner_replay_bytes": 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verified = _valid(out)
    if verified is None:
        raise RuntimeError("M3L-2 data verification failed")
    print(json.dumps(verified, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
