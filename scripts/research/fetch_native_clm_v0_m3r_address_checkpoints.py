"""Fetch exact published M3R lineage checkpoints for checkpoint-only diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_ARTIFACT_MANIFEST = Path(
    "artifacts/experiments/native-clm-v0-m3r-read-preserving-growth/model-artifacts.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/native-clm-m3r-address-checkpoints"),
    )
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()

    manifest = json.loads(args.artifact_manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "minicells.native-clm-v0.m3r-model-artifacts.v1":
        raise RuntimeError("unexpected M3R model-artifact manifest")
    if manifest.get("hf_upload_status") != "PUBLISHED":
        raise RuntimeError("canonical M3R model artifacts were not fully published")
    repo_id = str(manifest["repo_id"])
    revision = str(manifest["resolved_revision_after_upload"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    lineage_records = [record for record in manifest["files"] if record["arm"] == "lineage_growth"]
    if sorted(int(record["seed"]) for record in lineage_records) != [73611, 73612, 73613]:
        raise RuntimeError("canonical M3R lineage checkpoint set is incomplete")

    for record in lineage_records:
        seed = int(record["seed"])
        destination = args.output_dir / f"seed-{seed}-lineage.pt"
        if destination.exists() and destination.stat().st_size == int(record["bytes"]):
            actual = _sha256(destination)
            if actual == record["sha256"]:
                print(f"Reusing verified M3R lineage checkpoint seed={seed}.", flush=True)
                records.append(
                    {
                        "seed": seed,
                        "path": str(destination),
                        "sha256": actual,
                        "bytes": destination.stat().st_size,
                        "hf_path": record["path"],
                    }
                )
                continue
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=record["path"],
                repo_type="model",
                revision=revision,
                token=token,
            )
        )
        shutil.copy2(cached, destination)
        actual = _sha256(destination)
        if actual != record["sha256"] or destination.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"M3R checkpoint identity mismatch for seed {seed}")
        print(f"Fetched M3R lineage seed={seed}: {actual}", flush=True)
        records.append(
            {
                "seed": seed,
                "path": str(destination),
                "sha256": actual,
                "bytes": destination.stat().st_size,
                "hf_path": record["path"],
            }
        )

    output_manifest = {
        "format": "minicells.native-clm-v0.m3r-address-diagnostic-checkpoints.v1",
        "repo_id": repo_id,
        "revision": revision,
        "source_artifact_manifest": str(args.artifact_manifest),
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
