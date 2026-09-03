"""Prepare the fresh pinned data snapshot for KT001.

KT001 must not silently reuse the exact first-N slices consumed by the earlier
M3/M3R/M3L-2 formal protocols.  This builder therefore:

1. pins the exact upstream dataset revisions already used by the Native CLM line;
2. skips the prefix consumed by the earlier first-N builders on each split;
3. chooses the requested records from the subsequent stream using a deterministic
   salted min-hash selector;
4. writes byte-exact files plus a manifest with the selector provenance.

If a pinned split cannot supply enough records after prefix exclusion, preparation
fails closed instead of falling back to historical examples.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable


FORMAT = "minicells.kt001-data-manifest.v1"
SELECTOR_SALT = "IRF-CLM-KT001-v1"
PINNED = {
    "A": {
        "repo_id": "roneneldan/TinyStories",
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    },
    "B": {
        "repo_id": "Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    },
    "C_train": {
        "repo_id": "codeparrot/codeparrot-clean-train",
        "revision": "3e6ab65f2864931e041f6a82db9b5a6ec2b71ab4",
    },
    "C_eval": {
        "repo_id": "codeparrot/codeparrot-clean-valid",
        "revision": "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
    },
    "D": {
        "repo_id": "databricks/databricks-dolly-15k",
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(row: dict[str, Any]) -> str:
    return str(row.get("text", ""))


def _content(row: dict[str, Any]) -> str:
    return str(row.get("content", ""))


def _dolly(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    context = str(row.get("context", "")).strip()
    response = str(row.get("response", "")).strip()
    pieces = [f"Instruction:\n{instruction}"]
    if context:
        pieces.append(f"Context:\n{context}")
    pieces.append(f"Response:\n{response}")
    return "\n\n".join(pieces)


def _identity_digest(
    *,
    repo_id: str,
    split: str,
    config: str | None,
    source_index: int,
    text: str,
) -> bytes:
    prefix = "\0".join(
        (
            SELECTOR_SALT,
            repo_id,
            config or "",
            split,
            str(source_index),
        )
    ).encode("utf-8")
    return hashlib.sha256(prefix + b"\0" + text.encode("utf-8")).digest()


def _salted_min_hash_select(
    stream: Iterable[dict[str, Any]],
    formatter: Callable[[dict[str, Any]], str],
    *,
    repo_id: str,
    split: str,
    config: str | None,
    prefix_nonempty_to_skip: int,
    count: int,
    scan_nonempty: int,
) -> tuple[list[str], dict[str, Any]]:
    """Select the smallest salted hashes from a bounded post-prefix stream."""

    if count < 1 or scan_nonempty < count:
        raise ValueError("invalid KT001 min-hash selection budget")
    heap: list[tuple[int, int, str]] = []
    skipped = 0
    scanned = 0
    source_index = -1
    for source_index, row in enumerate(stream):
        text = formatter(row).strip()
        if not text:
            continue
        if skipped < prefix_nonempty_to_skip:
            skipped += 1
            continue
        if scanned >= scan_nonempty:
            break
        scanned += 1
        digest = _identity_digest(
            repo_id=repo_id,
            split=split,
            config=config,
            source_index=source_index,
            text=text,
        )
        score = int.from_bytes(digest, "big")
        # Python heap is a min-heap; negative score keeps the largest selected
        # hash at heap[0], allowing bounded top-k replacement.
        item = (-score, source_index, text)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)

    if skipped != prefix_nonempty_to_skip:
        raise RuntimeError(
            f"KT001 split {repo_id}/{split} ended while excluding historical prefix: "
            f"{skipped}/{prefix_nonempty_to_skip}"
        )
    if scanned < count or len(heap) != count:
        raise RuntimeError(
            f"KT001 split {repo_id}/{split} has insufficient fresh records after prefix exclusion: "
            f"scanned={scanned}, selected={len(heap)}, required={count}"
        )

    selected = sorted(
        [(-neg_score, source_index, text) for neg_score, source_index, text in heap],
        key=lambda item: (item[0], item[1]),
    )
    return (
        [text for _, _, text in selected],
        {
            "selector": "salted_min_hash_after_historical_prefix",
            "salt": SELECTOR_SALT,
            "prefix_nonempty_excluded": int(prefix_nonempty_to_skip),
            "post_prefix_nonempty_scanned": int(scanned),
            "selected": int(len(selected)),
            "last_source_index_seen": int(source_index),
            "minimum_selected_hash": f"{selected[0][0]:064x}",
            "maximum_selected_hash": f"{selected[-1][0]:064x}",
        },
    )


def _write_texts(path: Path, texts: list[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for text in texts:
            handle.write(text)
            handle.write("\n\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "documents": len(texts),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verified_cache(output: Path) -> dict[str, Any] | None:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("format") != FORMAT or manifest.get("selector_salt") != SELECTOR_SALT:
        return None
    if set(manifest.get("dataset_revisions", {})) != set(PINNED):
        return None
    for name, pinned in PINNED.items():
        record = manifest["dataset_revisions"][name]
        if record.get("repo_id") != pinned["repo_id"]:
            return None
        if record.get("resolved_revision") != pinned["revision"]:
            return None
    for record in manifest.get("files", {}).values():
        path = output / record["path"]
        if not path.exists() or path.stat().st_size != int(record["bytes"]):
            return None
        if _sha256(path) != record["sha256"]:
            return None
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/kt001-data"),
    )
    parser.add_argument("--scan-factor", type=int, default=3)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    if args.scan_factor < 1:
        raise ValueError("--scan-factor must be >= 1")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cached = _verified_cache(output)
    if cached is not None:
        print("Reusing verified KT001 fresh data cache.", flush=True)
        print(json.dumps(cached, indent=2), flush=True)
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")

    from datasets import load_dataset

    selections: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, Any]] = {}

    def select(
        name: str,
        *,
        repo_key: str,
        split: str,
        formatter: Callable[[dict[str, Any]], str],
        count: int,
        skip: int,
        config: str | None = None,
    ) -> list[str]:
        pinned = PINNED[repo_key]
        kwargs: dict[str, Any] = {
            "split": split,
            "streaming": True,
            "token": token,
            "revision": pinned["revision"],
        }
        if config is None:
            stream = load_dataset(pinned["repo_id"], **kwargs)
        else:
            stream = load_dataset(pinned["repo_id"], config, **kwargs)
        texts, selection = _salted_min_hash_select(
            stream,
            formatter,
            repo_id=pinned["repo_id"],
            split=split,
            config=config,
            prefix_nonempty_to_skip=skip,
            count=count,
            scan_nonempty=max(count, count * args.scan_factor),
        )
        selections[name] = {
            **selection,
            "repo_key": repo_key,
            "repo_id": pinned["repo_id"],
            "config": config,
            "split": split,
            "resolved_revision": pinned["revision"],
        }
        return texts

    a_bootstrap = select(
        "A_bootstrap",
        repo_key="A",
        split="train",
        formatter=_text,
        count=10_000,
        skip=10_000,
    )
    files["A_bootstrap"] = _write_texts(output / "A-tinystories-bootstrap.txt", a_bootstrap)

    a_eval = select(
        "A_eval",
        repo_key="A",
        split="validation",
        formatter=_text,
        count=2_000,
        skip=2_000,
    )
    files["A_eval"] = _write_texts(output / "A-tinystories-eval.txt", a_eval)

    b_train = select(
        "B_train",
        repo_key="B",
        split="train",
        formatter=_text,
        count=20_000,
        skip=20_000,
        config="wikitext-2-raw-v1",
    )
    files["B_train"] = _write_texts(output / "B-wikitext-train.txt", b_train)

    b_eval = select(
        "B_eval",
        repo_key="B",
        split="validation",
        formatter=_text,
        count=2_000,
        skip=2_000,
        config="wikitext-2-raw-v1",
    )
    files["B_eval"] = _write_texts(output / "B-wikitext-eval.txt", b_eval)

    c_train = select(
        "C_train",
        repo_key="C_train",
        split="train",
        formatter=_content,
        count=8_000,
        skip=8_000,
    )
    files["C_train"] = _write_texts(output / "C-code-train.txt", c_train)

    c_eval = select(
        "C_eval",
        repo_key="C_eval",
        split="train",
        formatter=_content,
        count=2_000,
        skip=2_000,
    )
    files["C_eval"] = _write_texts(output / "C-code-eval.txt", c_eval)

    # Dolly used one train split for both old train and eval; the old builder
    # consumed the first 10k + following 2k non-empty rows, so exclude all 12k.
    d_all = select(
        "D_combined",
        repo_key="D",
        split="train",
        formatter=_dolly,
        count=12_000,
        skip=12_000,
    )
    d_train = d_all[:10_000]
    d_eval = d_all[10_000:]
    files["D_train"] = _write_texts(output / "D-dolly-train.txt", d_train)
    files["D_eval"] = _write_texts(output / "D-dolly-eval.txt", d_eval)
    selections["D_train"] = {**selections["D_combined"], "selected_from_combined_rank": [0, 10_000]}
    selections["D_eval"] = {**selections["D_combined"], "selected_from_combined_rank": [10_000, 12_000]}
    selections.pop("D_combined")

    manifest = {
        "format": FORMAT,
        "experiment_id": "integrated-replay-free-clm-kill-test-001",
        "stream": ["B", "C", "D"],
        "selector_salt": SELECTOR_SALT,
        "selection_policy": (
            "exclude the non-empty prefix consumed by the historical first-N Native CLM builders, "
            "then choose the smallest salted SHA-256 records from a bounded subsequent stream"
        ),
        "scan_factor": int(args.scan_factor),
        "A_role_after_continual_start": "evaluation_only",
        "learner_replay_bytes": 0,
        "dataset_revisions": {
            key: {
                "repo_id": value["repo_id"],
                "requested_revision": value["revision"],
                "resolved_revision": value["revision"],
            }
            for key, value in PINNED.items()
        },
        "files": files,
        "selection": selections,
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    verified = _verified_cache(output)
    if verified is None:
        raise RuntimeError("KT001 fresh data cache failed post-write verification")
    print(json.dumps(verified, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
