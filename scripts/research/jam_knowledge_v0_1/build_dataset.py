#!/usr/bin/env python3
"""Materialize deterministic QA views for JAM Knowledge v0.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "research" / "datasets" / "jam-knowledge-v0.1"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_shards(directory: Path) -> list[dict]:
    return [row for path in sorted(directory.glob("*.jsonl")) for row in load_jsonl(path)]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def qid(split: str, concept_id: str, suffix: str) -> str:
    return f"{split}.{concept_id}.{suffix}"


def build(dataset: Path = DATASET) -> dict[str, int]:
    concepts = load_shards(dataset / "concepts")
    reasoning = load_shards(dataset / "evaluation" / "reasoning")
    by_id = {row["id"]: row for row in concepts}

    train: list[dict] = []
    validation: list[dict] = []
    factual: list[dict] = []
    relational: list[dict] = []
    misconceptions: list[dict] = []

    for concept in concepts:
        base = {"concept_ids": [concept["id"]], "source_pin": concept["source_pin"], "derived": True}
        train.extend([
            {**base, "id": qid("train", concept["id"], "definition"), "split": "train", "type": "definition", "question": f"What is {concept['title']} in JAM?", "answer": concept["canonical_fact"]},
            {**base, "id": qid("train", concept["id"], "explain"), "split": "train", "type": "explanation", "question": f"Explain the role of {concept['title']} in the JAM protocol.", "answer": concept["canonical_fact"]},
        ])
        validation.append({**base, "id": qid("validation", concept["id"], "paraphrase"), "split": "validation", "type": "factual", "question": f"In one concise answer, describe {concept['title']} as defined by JAM.", "answer": concept["canonical_fact"]})
        factual.append({**base, "id": qid("evaluation", concept["id"], "factual"), "split": "evaluation", "type": "factual", "question": f"What protocol function or meaning does {concept['title']} have in JAM?", "answer": concept["canonical_fact"]})

        for index, relation in enumerate(concept["relations"]):
            target = by_id[relation["target"]]
            relational.append({"id": qid("evaluation", concept["id"], f"relation{index}"), "split": "evaluation", "type": "relational", "question": f"How are {concept['title']} and {target['title']} related in JAM?", "answer": relation["statement"], "concept_ids": [concept["id"], target["id"]], "source_pin": concept["source_pin"], "derived": True})

        for index, misconception in enumerate(concept["misconceptions"]):
            train.append({**base, "id": qid("train", concept["id"], f"misconception{index}"), "split": "train", "type": "misconception", "question": f'Is this statement correct about JAM: "{misconception}" Explain briefly.', "answer": f"No. {concept['canonical_fact']}"})
            misconceptions.append({**base, "id": qid("evaluation", concept["id"], f"misconception{index}"), "split": "evaluation", "type": "misconception", "question": f'Evaluate this claim about JAM: "{misconception}"', "answer": f"The claim is incorrect. {concept['canonical_fact']}"})

    out = dataset / "generated"
    dump_jsonl(out / "concepts.jsonl", concepts)
    dump_jsonl(out / "train.jsonl", train)
    dump_jsonl(out / "validation.jsonl", validation)
    dump_jsonl(out / "evaluation" / "factual.jsonl", factual)
    dump_jsonl(out / "evaluation" / "relational.jsonl", relational)
    dump_jsonl(out / "evaluation" / "misconceptions.jsonl", misconceptions)
    dump_jsonl(out / "evaluation" / "reasoning.jsonl", reasoning)

    return {"concepts": len(concepts), "train": len(train), "validation": len(validation), "factual": len(factual), "relational": len(relational), "misconceptions": len(misconceptions), "reasoning": len(reasoning)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset), sort_keys=True))


if __name__ == "__main__":
    main()
