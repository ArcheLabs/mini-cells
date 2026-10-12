"""Prepare the pinned KT001 data snapshot with explicit overlap accounting.

The builder prefers records after the first-N prefixes consumed by the earlier
Native CLM continual-learning line. Selection is deterministic salted min-hash.
If an exact pinned split cannot supply a same-size disjoint replacement, only the
unavoidable shortfall is filled from the historical prefix and that reuse is
recorded explicitly in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

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


def _identity_score(
    *,
    repo_id: str,
    split: str,
    config: str | None,
    source_index: int,
    text: str,
) -> int:
    prefix = "\0".join(
        (SELECTOR_SALT, repo_id, config or "", split, str(source_index))
    ).encode("utf-8")
    digest = hashlib.sha256(prefix + b"\0" + text.encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def _bounded_push(
    heap: list[tuple[int, int, str]],
    *,
    score: int,
    source_index: int,
    text: str,
    limit: int,
) -> None:
    item = (-score, source_index, text)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _ordered(
    heap: list[tuple[int, int, str]], *, origin: str
) -> list[tuple[int, int, str, str]]:
    rows = [
        (-negative_score, source_index, text, origin)
        for negative_score, source_index, text in heap
    ]
    return sorted(rows, key=lambda item: (item[0], item[1]))


def _salted_prefer_fresh_select(
    stream: Iterable[dict[str, Any]],
    formatter: Callable[[dict[str, Any]], str],
    *,
    repo_id: str,
    split: str,
    config: str | None,
    historical_prefix_nonempty: int,
    count: int,
    fresh_scan_nonempty: int,
) -> tuple[list[str], dict[str, Any]]:
    """Prefer post-prefix records and reuse only an unavoidable shortfall."""

    if count < 1 or fresh_scan_nonempty < count:
        raise ValueError("invalid KT001 min-hash selection budget")
    if historical_prefix_nonempty < 0:
        raise ValueError("historical prefix must be non-negative")

    old_heap: list[tuple[int, int, str]] = []
    fresh_heap: list[tuple[int, int, str]] = []
    old_seen = 0
    fresh_seen = 0
    last_source_index = -1
    scan_limit_reached = False

    for source_index, row in enumerate(stream):
        last_source_index = source_index
        text = formatter(row).strip()
        if not text:
            continue
        score = _identity_score(
            repo_id=repo_id,
            split=split,
            config=config,
            source_index=source_index,
            text=text,
        )
        if old_seen < historical_prefix_nonempty:
            old_seen += 1
            _bounded_push(
                old_heap,
                score=score,
                source_index=source_index,
                text=text,
                limit=count,
            )
            continue
        if fresh_seen >= fresh_scan_nonempty:
            scan_limit_reached = True
            break
        fresh_seen += 1
        _bounded_push(
            fresh_heap,
            score=score,
            source_index=source_index,
            text=text,
            limit=count,
        )

    if old_seen != historical_prefix_nonempty:
        raise RuntimeError(
            f"KT001 split {repo_id}/{split} ended before historical prefix was found: "
            f"{old_seen}/{historical_prefix_nonempty}"
        )

    fresh = _ordered(fresh_heap, origin="post_historical_prefix")
    old = _ordered(old_heap, origin="historical_prefix")
    if fresh_seen >= count:
        selected = fresh[:count]
        historical_reused = 0
    else:
        if scan_limit_reached:
            raise RuntimeError("KT001 fresh-capacity accounting drift")
        shortfall = count - fresh_seen
        if len(old) < shortfall:
            raise RuntimeError(
                f"KT001 split {repo_id}/{split} cannot supply {count} rows: "
                f"fresh={fresh_seen}, historical_available={len(old)}"
            )
        selected = fresh + old[:shortfall]
        selected.sort(key=lambda item: (item[0], item[1]))
        historical_reused = shortfall

    if len(selected) != count:
        raise RuntimeError("KT001 selector failed to produce registered document count")

    fresh_selected = count - historical_reused
    metadata = {
        "selector": "salted_min_hash_prefer_post_prefix_with_explicit_shortfall_reuse",
        "salt": SELECTOR_SALT,
        "historical_prefix_nonempty": historical_prefix_nonempty,
        "historical_prefix_reused": historical_reused,
        "fresh_post_prefix_selected": fresh_selected,
        "fully_disjoint_from_historical_prefix": historical_reused == 0,
        "post_prefix_nonempty_seen": fresh_seen,
        "fresh_scan_nonempty_limit": fresh_scan_nonempty,
        "stream_exhausted_before_fresh_target": fresh_seen < count,
        "selected": len(selected),
        "last_source_index_seen": last_source_index,
        "minimum_selected_hash": f"{selected[0][0]:064x}",
        "maximum_selected_hash": f"{selected[-1][0]:064x}",
    }
    return [text for _, _, text, _ in selected], metadata


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
    revisions = manifest.get("dataset_revisions", {})
    if set(revisions) != set(PINNED):
        return None
    for name, pinned in PINNED.items():
        record = revisions[name]
        if record.get("repo_id") != pinned["repo_id"]:
            return None
        if record.get("resolved_revision") != pinned["revision"]:
            return None
    for record in manifest.get("files", {}).values():
        path = output / record["path"]
        if not path.exists() or path.stat().st_size != record["bytes"]:
            return None
        if _sha256(path) != record["sha256"]:
            return None
    return manifest


def _load_stream(
    *,
    repo_key: str,
    split: str,
    config: str | None,
    token: str,
):
    from datasets import load_dataset

    pinned = PINNED[repo_key]
    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": True,
        "token": token,
        "revision": pinned["revision"],
    }
    if config is None:
        return load_dataset(pinned["repo_id"], **kwargs)
    return load_dataset(pinned["repo_id"], config, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/kaggle/working/kt001-data")
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
        print("Reusing verified KT001 pinned data cache.", flush=True)
        print(json.dumps(cached, indent=2), flush=True)
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")

    selections: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, Any]] = {}

    def select(
        name: str,
        *,
        repo_key: str,
        split: str,
        formatter: Callable[[dict[str, Any]], str],
        count: int,
        historical_prefix: int,
        config: str | None = None,
    ) -> list[str]:
        stream = _load_stream(
            repo_key=repo_key,
            split=split,
            config=config,
            token=token,
        )
        pinned = PINNED[repo_key]
        texts, metadata = _salted_prefer_fresh_select(
            stream,
            formatter,
            repo_id=pinned["repo_id"],
            split=split,
            config=config,
            historical_prefix_nonempty=historical_prefix,
            count=count,
            fresh_scan_nonempty=max(count, count * args.scan_factor),
        )
        selections[name] = {
            **metadata,
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
        historical_prefix=10_000,
    )
    files["A_bootstrap"] = _write_texts(output / "A-tinystories-bootstrap.txt", a_bootstrap)

    a_eval = select(
        "A_eval",
        repo_key="A",
        split="validation",
        formatter=_text,
        count=2_000,
        historical_prefix=2_000,
    )
    files["A_eval"] = _write_texts(output / "A-tinystories-eval.txt", a_eval)

    b_train = select(
        "B_train",
        repo_key="B",
        split="train",
        formatter=_text,
        count=20_000,
        historical_prefix=20_000,
        config="wikitext-2-raw-v1",
    )
    files["B_train"] = _write_texts(output / "B-wikitext-train.txt", b_train)

    b_eval = select(
        "B_eval",
        repo_key="B",
        split="validation",
        formatter=_text,
        count=2_000,
        historical_prefix=2_000,
        config="wikitext-2-raw-v1",
    )
    files["B_eval"] = _write_texts(output / "B-wikitext-eval.txt", b_eval)

    c_train = select(
        "C_train",
        repo_key="C_train",
        split="train",
        formatter=_content,
        count=8_000,
        historical_prefix=8_000,
    )
    files["C_train"] = _write_texts(output / "C-code-train.txt", c_train)

    c_eval = select(
        "C_eval",
        repo_key="C_eval",
        split="train",
        formatter=_content,
        count=2_000,
        historical_prefix=2_000,
    )
    files["C_eval"] = _write_texts(output / "C-code-eval.txt", c_eval)

    d_all = select(
        "D_combined",
        repo_key="D",
        split="train",
        formatter=_dolly,
        count=12_000,
        historical_prefix=12_000,
    )
    d_train = d_all[:10_000]
    d_eval = d_all[10_000:]
    files["D_train"] = _write_texts(output / "D-dolly-train.txt", d_train)
    files["D_eval"] = _write_texts(output / "D-dolly-eval.txt", d_eval)
    combined = selections.pop("D_combined")
    selections["D_train"] = {**combined, "selected_from_combined_rank": [0, 10_000]}
    selections["D_eval"] = {**combined, "selected_from_combined_rank": [10_000, 12_000]}

    total_selected = sum(record["documents"] for record in files.values())
    total_reused = sum(
        record.get("historical_prefix_reused", 0)
        for name, record in selections.items()
        if name != "D_eval"
    )
    manifest = {
        "format": FORMAT,
        "experiment_id": "integrated-replay-free-clm-kill-test-001",
        "stream": ["B", "C", "D"],
        "selector_salt": SELECTOR_SALT,
        "selection_policy": (
            "prefer deterministic salted min-hash records after historical first-N prefixes; "
            "supplement only unavoidable shortfalls from those prefixes and record reuse"
        ),
        "scan_factor": args.scan_factor,
        "A_role_after_continual_start": "evaluation_only",
        "learner_replay_bytes": 0,
        "fully_disjoint_from_historical_prefixes": all(
            record.get("fully_disjoint_from_historical_prefix", False)
            for record in selections.values()
        ),
        "historical_prefix_reuse_is_explicit": True,
        "selected_documents_total": total_selected,
        "historical_prefix_reused_documents_accounted": total_reused,
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
        raise RuntimeError("KT001 pinned data cache failed post-write verification")
    print(json.dumps(verified, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
