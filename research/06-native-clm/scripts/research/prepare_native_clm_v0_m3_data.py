"""Prepare the pinned Native CLM v0 M3 A/B/C/D continual-language snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from huggingface_hub import HfApi

FORMAT = "minicells.native-clm-v0.m3-data-manifest.v1"
DATASET_REPOS = {
    "A": "roneneldan/TinyStories",
    "B": "Salesforce/wikitext",
    "C_train": "codeparrot/codeparrot-clean-train",
    "C_eval": "codeparrot/codeparrot-clean-valid",
    "D": "databricks/databricks-dolly-15k",
}


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


def _verified_cache(output: Path) -> dict | None:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("format") != FORMAT:
        return None
    if set(manifest.get("dataset_revisions", {})) != set(DATASET_REPOS):
        return None
    for record in manifest.get("files", {}).values():
        path = output / record["path"]
        if not path.exists() or path.stat().st_size != record["bytes"]:
            return None
        if _sha256(path) != record["sha256"]:
            return None
    return manifest


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


def _resolve_revisions(token: str) -> dict[str, dict[str, str]]:
    api = HfApi(token=token)
    resolved: dict[str, dict[str, str]] = {}
    for name, repo_id in DATASET_REPOS.items():
        info = api.dataset_info(repo_id, revision="main")
        if not info.sha:
            raise RuntimeError(f"Hugging Face did not resolve a revision for {repo_id}")
        resolved[name] = {
            "repo_id": repo_id,
            "requested_revision": "main",
            "resolved_revision": info.sha,
        }
        print(f"Resolved M3 dataset {name}: {repo_id}@main -> {info.sha}", flush=True)
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/native-clm-m3-data"),
    )
    parser.add_argument("--a-eval-docs", type=int, default=2000)
    parser.add_argument("--b-train-docs", type=int, default=20000)
    parser.add_argument("--b-eval-docs", type=int, default=2000)
    parser.add_argument("--c-train-docs", type=int, default=8000)
    parser.add_argument("--c-eval-docs", type=int, default=2000)
    parser.add_argument("--d-train-docs", type=int, default=10000)
    parser.add_argument("--d-eval-docs", type=int, default=2000)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cached = _verified_cache(output)
    if cached is not None:
        print("Reusing verified Native CLM M3 data cache.", flush=True)
        print(json.dumps(cached, indent=2), flush=True)
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    revisions = _resolve_revisions(token)

    from datasets import load_dataset

    a_stream = load_dataset(
        DATASET_REPOS["A"],
        split="validation",
        streaming=True,
        token=token,
        revision=revisions["A"]["resolved_revision"],
    )
    a_path = output / "A-tinystories-eval.txt"
    a_count = _fsync_text(a_path, _rows(a_stream, _text), args.a_eval_docs)

    b_train_stream = load_dataset(
        DATASET_REPOS["B"],
        "wikitext-2-raw-v1",
        split="train",
        streaming=True,
        token=token,
        revision=revisions["B"]["resolved_revision"],
    )
    b_eval_stream = load_dataset(
        DATASET_REPOS["B"],
        "wikitext-2-raw-v1",
        split="validation",
        streaming=True,
        token=token,
        revision=revisions["B"]["resolved_revision"],
    )
    b_train_path = output / "B-wikitext-train.txt"
    b_eval_path = output / "B-wikitext-eval.txt"
    b_train_count = _fsync_text(b_train_path, _rows(b_train_stream, _text), args.b_train_docs)
    b_eval_count = _fsync_text(b_eval_path, _rows(b_eval_stream, _text), args.b_eval_docs)

    c_train_stream = load_dataset(
        DATASET_REPOS["C_train"],
        split="train",
        streaming=True,
        token=token,
        revision=revisions["C_train"]["resolved_revision"],
    )
    c_eval_stream = load_dataset(
        DATASET_REPOS["C_eval"],
        split="train",
        streaming=True,
        token=token,
        revision=revisions["C_eval"]["resolved_revision"],
    )
    c_train_path = output / "C-code-train.txt"
    c_eval_path = output / "C-code-eval.txt"
    c_train_count = _fsync_text(c_train_path, _rows(c_train_stream, _content), args.c_train_docs)
    c_eval_count = _fsync_text(c_eval_path, _rows(c_eval_stream, _content), args.c_eval_docs)

    d_stream = load_dataset(
        DATASET_REPOS["D"],
        split="train",
        streaming=True,
        token=token,
        revision=revisions["D"]["resolved_revision"],
    )
    d_train, d_eval = _split_single_stream(d_stream, _dolly, args.d_train_docs, args.d_eval_docs)
    d_train_path = output / "D-dolly-train.txt"
    d_eval_path = output / "D-dolly-eval.txt"
    d_train_count = _fsync_text(d_train_path, d_train, args.d_train_docs)
    d_eval_count = _fsync_text(d_eval_path, d_eval, args.d_eval_docs)

    expected = {
        "A_eval": (a_path, a_count, args.a_eval_docs),
        "B_train": (b_train_path, b_train_count, args.b_train_docs),
        "B_eval": (b_eval_path, b_eval_count, args.b_eval_docs),
        "C_train": (c_train_path, c_train_count, args.c_train_docs),
        "C_eval": (c_eval_path, c_eval_count, args.c_eval_docs),
        "D_train": (d_train_path, d_train_count, args.d_train_docs),
        "D_eval": (d_eval_path, d_eval_count, args.d_eval_docs),
    }
    for name, (_, count, requested) in expected.items():
        if count < requested:
            raise RuntimeError(f"{name} yielded only {count}/{requested} documents")

    files = {
        name: {
            "path": path.name,
            "documents": count,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, (path, count, _) in expected.items()
    }
    manifest = {
        "format": FORMAT,
        "stream": ["B", "C", "D"],
        "A_role": "evaluation_only_M1_retention",
        "learner_replay_bytes": 0,
        "dataset_revisions": revisions,
        "files": files,
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
