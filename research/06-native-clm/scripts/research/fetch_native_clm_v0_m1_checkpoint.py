"""Fetch and verify the canonical Native CLM v0 M1 checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO = "archelabsxyz/native-clm-v0"
DEFAULT_FILENAME = "final-model.pt"
DEFAULT_SHA256 = "91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--expected-sha256", default=DEFAULT_SHA256)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/native-clm-v0-m1/final-model.pt"),
    )
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")

    info = HfApi(token=token).model_info(args.repo_id, revision=args.revision)
    resolved_revision = info.sha
    print(f"Resolved {args.repo_id}@{args.revision} -> {resolved_revision}", flush=True)

    cached = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            revision=resolved_revision,
            token=token,
        )
    )
    actual = sha256_file(cached)
    if actual != args.expected_sha256:
        raise RuntimeError(
            f"checkpoint SHA mismatch: expected {args.expected_sha256}, got {actual}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if cached.resolve() != args.output.resolve():
        shutil.copy2(cached, args.output)
    copied_sha = sha256_file(args.output)
    if copied_sha != args.expected_sha256:
        raise RuntimeError("copied checkpoint failed SHA verification")

    provenance = {
        "format": "minicells.native-clm-v0.m1-checkpoint-provenance.v1",
        "repo_id": args.repo_id,
        "filename": args.filename,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "sha256": copied_sha,
        "bytes": args.output.stat().st_size,
        "local_path": str(args.output),
    }
    provenance_path = args.output.with_name("provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
