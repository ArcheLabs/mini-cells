"""Prepare only the exact pinned WikiText B-train file needed by M2-R0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

FORMAT = "minicells.native-clm-v0.m2r0-data-manifest.v1"
REPO_ID = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
EXPECTED_SHA256 = "c7029622b9a1c4b4f249d927b3ab30d4c09ddffe845e031aee3412b4735ca440"
EXPECTED_BYTES = 9_104_081
EXPECTED_DOCUMENTS = 20_000
FILENAME = "B-wikitext-train.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(output: Path) -> dict | None:
    manifest_path = output / "manifest.json"
    data_path = output / FILENAME
    if not manifest_path.exists() or not data_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("format") != FORMAT:
        return None
    record = manifest.get("files", {}).get("B_train", {})
    if record.get("sha256") != EXPECTED_SHA256 or int(record.get("bytes", -1)) != EXPECTED_BYTES:
        return None
    if int(record.get("documents", -1)) != EXPECTED_DOCUMENTS:
        return None
    if data_path.stat().st_size != EXPECTED_BYTES or _sha256(data_path) != EXPECTED_SHA256:
        return None
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/native-clm-m2r0-data"),
    )
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    cached = _verified(output)
    if cached is not None:
        print("Reusing verified Native CLM M2-R0 B data cache.", flush=True)
        print(json.dumps(cached, indent=2), flush=True)
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    resolved = HfApi(token=token).dataset_info(REPO_ID, revision=REVISION).sha
    if resolved != REVISION:
        raise RuntimeError(f"M2-R0 WikiText revision drift: {resolved} != {REVISION}")

    from datasets import load_dataset

    stream = load_dataset(
        REPO_ID,
        "wikitext-2-raw-v1",
        split="train",
        streaming=True,
        token=token,
        revision=REVISION,
    )
    data_path = output / FILENAME
    count = 0
    with data_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in stream:
            if count >= EXPECTED_DOCUMENTS:
                break
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            handle.write(text)
            handle.write("\n\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    if count != EXPECTED_DOCUMENTS:
        raise RuntimeError(f"M2-R0 B train yielded {count}/{EXPECTED_DOCUMENTS} documents")
    actual_bytes = data_path.stat().st_size
    actual_sha = _sha256(data_path)
    if actual_bytes != EXPECTED_BYTES or actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            "M2-R0 B-train identity mismatch: "
            f"bytes={actual_bytes}/{EXPECTED_BYTES} sha={actual_sha}/{EXPECTED_SHA256}"
        )

    manifest = {
        "format": FORMAT,
        "role": "optimizer-mechanics gradient source only",
        "dataset_revisions": {
            "B": {
                "repo_id": REPO_ID,
                "requested_revision": REVISION,
                "resolved_revision": resolved,
            }
        },
        "files": {
            "B_train": {
                "path": FILENAME,
                "documents": count,
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
        },
        "learner_replay_bytes": 0,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verified = _verified(output)
    if verified is None:
        raise RuntimeError("M2-R0 B-only data verification failed")
    print(json.dumps(verified, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
