"""Fetch exact published M3L-2 treatment checkpoints for M3W-0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_PROTOCOL = Path(
    "research/validations/native-clm-v0-m3w0-write-drift-restoration/protocol.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/native-clm-m3w0-checkpoints"),
    )
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("format") != "minicells.native-clm-v0.m3w0-write-drift-restoration.protocol.v1":
        raise RuntimeError("unexpected M3W-0 protocol format")
    source = protocol["parent_evidence"]
    repo_id = str(source["hf_repo"])
    revision = str(source["hf_revision"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    expected_seeds = [74211, 74212, 74213]
    checkpoint_records = protocol["treatment_checkpoints"]
    if sorted(int(record["seed"]) for record in checkpoint_records) != expected_seeds:
        raise RuntimeError("registered M3W-0 checkpoint set is incomplete")

    for record in checkpoint_records:
        seed = int(record["seed"])
        destination = args.output_dir / f"seed-{seed}-online-address.pt"
        expected_sha = str(record["sha256"])
        expected_bytes = int(record["bytes"])
        if destination.exists() and destination.stat().st_size == expected_bytes:
            actual = _sha256(destination)
            if actual == expected_sha:
                records.append(
                    {
                        "seed": seed,
                        "path": str(destination),
                        "sha256": actual,
                        "bytes": expected_bytes,
                        "hf_path": record["hf_path"],
                    }
                )
                print(f"Reusing verified M3W-0 checkpoint seed={seed}.", flush=True)
                continue
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=record["hf_path"],
                repo_type="model",
                revision=revision,
                token=token,
            )
        )
        shutil.copy2(cached, destination)
        actual = _sha256(destination)
        if actual != expected_sha or destination.stat().st_size != expected_bytes:
            raise RuntimeError(f"M3W-0 checkpoint identity mismatch for seed {seed}")
        records.append(
            {
                "seed": seed,
                "path": str(destination),
                "sha256": actual,
                "bytes": expected_bytes,
                "hf_path": record["hf_path"],
            }
        )
        print(f"Fetched M3W-0 checkpoint seed={seed}: {actual}", flush=True)

    manifest = {
        "format": "minicells.native-clm-v0.m3w0-checkpoints.v1",
        "repo_id": repo_id,
        "revision": revision,
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
