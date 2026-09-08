#!/usr/bin/env python3
"""Integrity checks for JAM Knowledge v0.1."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "research" / "datasets" / "jam-knowledge-v0.1"
SOURCE_PIN = "graypaper-0.8.0-e5375148"
EXPECTED_CONCEPTS = 180
EXPECTED_REASONING = 50
EXPECTED_CATEGORY_COUNTS = {"foundations": 15, "block_state": 15, "consensus": 15, "services": 22, "authorization": 10, "work": 25, "guarantees": 20, "accumulation": 18, "pvm": 25, "serialization": 15}
ID_RE = re.compile(r"^jam\.[a-z0-9_]+\.[a-z0-9_]+$")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_shards(directory: Path) -> list[dict]:
    return [row for path in sorted(directory.glob("*.jsonl")) for row in load_jsonl(path)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_question(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def validate(dataset: Path = DATASET) -> dict[str, int | str]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    sources = json.loads((dataset / "sources.lock.json").read_text(encoding="utf-8"))
    concepts = load_shards(dataset / "concepts")
    reasoning = load_shards(dataset / "evaluation" / "reasoning")

    assert sources["sources"]["graypaper"]["version"] == "0.8.0"
    assert sources["sources"]["graypaper"]["ref"] == "e5375148597a45a99d31c9aa6bce6c7bf3a48998"
    assert sources["sources"]["graypaper"]["source_pin"] == SOURCE_PIN
    assert len(concepts) == EXPECTED_CONCEPTS
    assert len(reasoning) == EXPECTED_REASONING

    category_counts = {category: sum(row["category"] == category for row in concepts) for category in EXPECTED_CATEGORY_COUNTS}
    assert category_counts == EXPECTED_CATEGORY_COUNTS, category_counts

    ids = [row["id"] for row in concepts]
    assert len(ids) == len(set(ids))
    known = set(ids)
    for row in concepts:
        assert ID_RE.match(row["id"]), row["id"]
        assert row["source_pin"] == SOURCE_PIN
        assert row["status"] == "canonical"
        assert row["source_refs"]
        assert all(ref.startswith("text/") and ref.endswith(".tex") for ref in row["source_refs"])
        assert 1 <= int(row["difficulty"]) <= 3
        for relation in row["relations"]:
            assert relation["target"] in known, (row["id"], relation["target"])

    reasoning_ids = [row["id"] for row in reasoning]
    assert len(reasoning_ids) == len(set(reasoning_ids))
    for row in reasoning:
        assert row["derived"] is False
        assert row["source_pin"] == SOURCE_PIN
        assert row["split"] == "evaluation" and row["type"] == "reasoning"
        assert len(row["concept_ids"]) >= 1
        assert all(cid in known for cid in row["concept_ids"])

    for relative, record in manifest["canonical_files"].items():
        assert sha256(dataset / relative) == record["sha256"], relative

    generated = dataset / "generated"
    if generated.exists():
        train = load_jsonl(generated / "train.jsonl")
        eval_paths = sorted((generated / "evaluation").glob("*.jsonl"))
        evaluation = [row for path in eval_paths for row in load_jsonl(path)]
        all_ids = [row["id"] for row in train + evaluation]
        assert len(all_ids) == len(set(all_ids))
        train_questions = {normalize_question(row["question"]) for row in train}
        eval_questions = {normalize_question(row["question"]) for row in evaluation}
        assert not (train_questions & eval_questions)
        for row in train + evaluation:
            assert all(cid in known for cid in row["concept_ids"])

    return {"status": "JAM_KNOWLEDGE_V0_1_VALID", "concepts": len(concepts), "reasoning_holdout": len(reasoning), "relations": sum(len(row["relations"]) for row in concepts), "misconceptions": sum(len(row["misconceptions"]) for row in concepts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset), sort_keys=True))


if __name__ == "__main__":
    main()
