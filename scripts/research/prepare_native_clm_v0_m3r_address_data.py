"""Reconstruct the exact canonical M3R data snapshot for address diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

FORMAT = "minicells.native-clm-v0.m3r-address-diagnostic-data.v1"
PARENT_FORMAT = "minicells.native-clm-v0.m3r-data-manifest.v1"
DEFAULT_PARENT_MANIFEST = Path(
    "artifacts/experiments/native-clm-v0-m3r-read-preserving-growth/data-manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_text(path: Path, texts: Iterable[str], limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for text in texts:
            if count >= limit:
                break
            clean = str(text).strip()
            if not clean:
                continue
            handle.write(clean)
            handle.write("\n\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _rows(stream, formatter: Callable[[dict], str]):
    for row in stream:
        yield formatter(row)


def _text(row: dict) -> str:
    return row["text"]


def _content(row: dict) -> str:
    return row["content"]


def _dolly(row: dict) -> str:
    instruction = str(row.get("instruction", "")).strip()
    context = str(row.get("context", "")).strip()
    response = str(row.get("response", "")).strip()
    pieces = [f"Instruction:\n{instruction}"]
    if context:
        pieces.append(f"Context:\n{context}")
    pieces.append(f"Response:\n{response}")
    return "\n\n".join(pieces)


def _split_single_stream(stream, formatter, train_count: int, eval_count: int):
    iterator = iter(stream)
    train: list[str] = []
    while len(train) < train_count:
        text = formatter(next(iterator)).strip()
        if text:
            train.append(text)
    evaluation: list[str] = []
    while len(evaluation) < eval_count:
        text = formatter(next(iterator)).strip()
        if text:
            evaluation.append(text)
    return train, evaluation


def _verified_cache(output: Path, parent: dict) -> dict | None:
    diagnostic_manifest = output / "manifest.json"
    if not diagnostic_manifest.exists():
        return None
    try:
        manifest = json.loads(diagnostic_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("format") != FORMAT:
        return None
    if manifest.get("parent_files") != parent.get("files"):
        return None
    for record in parent["files"].values():
        path = output / record["path"]
        if not path.exists() or path.stat().st_size != int(record["bytes"]):
            return None
        if _sha256(path) != record["sha256"]:
            return None
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/native-clm-m3r-address-data"),
    )
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()

    parent = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    if parent.get("format") != PARENT_FORMAT:
        raise RuntimeError("unexpected canonical M3R data-manifest format")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cached = _verified_cache(output, parent)
    if cached is not None:
        print("Reusing exact verified M3R address-diagnostic data cache.", flush=True)
        print(json.dumps(cached, indent=2), flush=True)
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    revisions = parent["dataset_revisions"]
    from datasets import load_dataset

    a_stream = load_dataset(
        revisions["A"]["repo_id"],
        split="validation",
        streaming=True,
        token=token,
        revision=revisions["A"]["resolved_revision"],
    )
    a_path = output / parent["files"]["A_eval"]["path"]
    _fsync_text(a_path, _rows(a_stream, _text), int(parent["files"]["A_eval"]["documents"]))

    b_train_stream = load_dataset(
        revisions["B"]["repo_id"],
        "wikitext-2-raw-v1",
        split="train",
        streaming=True,
        token=token,
        revision=revisions["B"]["resolved_revision"],
    )
    b_eval_stream = load_dataset(
        revisions["B"]["repo_id"],
        "wikitext-2-raw-v1",
        split="validation",
        streaming=True,
        token=token,
        revision=revisions["B"]["resolved_revision"],
    )
    b_train_path = output / parent["files"]["B_train"]["path"]
    b_eval_path = output / parent["files"]["B_eval"]["path"]
    _fsync_text(
        b_train_path,
        _rows(b_train_stream, _text),
        int(parent["files"]["B_train"]["documents"]),
    )
    _fsync_text(
        b_eval_path,
        _rows(b_eval_stream, _text),
        int(parent["files"]["B_eval"]["documents"]),
    )

    c_train_stream = load_dataset(
        revisions["C_train"]["repo_id"],
        split="train",
        streaming=True,
        token=token,
        revision=revisions["C_train"]["resolved_revision"],
    )
    c_eval_stream = load_dataset(
        revisions["C_eval"]["repo_id"],
        split="train",
        streaming=True,
        token=token,
        revision=revisions["C_eval"]["resolved_revision"],
    )
    c_train_path = output / parent["files"]["C_train"]["path"]
    c_eval_path = output / parent["files"]["C_eval"]["path"]
    _fsync_text(
        c_train_path,
        _rows(c_train_stream, _content),
        int(parent["files"]["C_train"]["documents"]),
    )
    _fsync_text(
        c_eval_path,
        _rows(c_eval_stream, _content),
        int(parent["files"]["C_eval"]["documents"]),
    )

    d_stream = load_dataset(
        revisions["D"]["repo_id"],
        split="train",
        streaming=True,
        token=token,
        revision=revisions["D"]["resolved_revision"],
    )
    d_train_count = int(parent["files"]["D_train"]["documents"])
    d_eval_count = int(parent["files"]["D_eval"]["documents"])
    d_train, d_eval = _split_single_stream(d_stream, _dolly, d_train_count, d_eval_count)
    d_train_path = output / parent["files"]["D_train"]["path"]
    d_eval_path = output / parent["files"]["D_eval"]["path"]
    _fsync_text(d_train_path, d_train, d_train_count)
    _fsync_text(d_eval_path, d_eval, d_eval_count)

    for name, record in parent["files"].items():
        path = output / record["path"]
        if not path.exists():
            raise RuntimeError(f"missing reconstructed M3R file {name}: {path}")
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        if actual_bytes != int(record["bytes"]) or actual_sha != record["sha256"]:
            raise RuntimeError(
                f"M3R snapshot mismatch for {name}: bytes {actual_bytes}/{record['bytes']} "
                f"sha {actual_sha}/{record['sha256']}"
            )

    parent_bytes = args.parent_manifest.read_bytes()
    manifest = {
        "format": FORMAT,
        "parent_manifest_path": str(args.parent_manifest),
        "parent_manifest_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "dataset_revisions": revisions,
        "parent_files": parent["files"],
        "exact_parent_snapshot_verified": True,
        "model_training": False,
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(manifest, indent=2), flush=True)
    if sys.version_info >= (3, 12):
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
