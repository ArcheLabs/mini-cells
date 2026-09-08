from __future__ import annotations

from typing import Any

KNOWLEDGE_CANDIDATES = (
    "QX-17",
    "VR-08",
    "LM-42",
    "PN-63",
    "TK-51",
    "HF-29",
    "AX-11",
    "BZ-34",
    "CY-56",
    "DW-78",
)
CAPABILITY_CANDIDATES = tuple(f"Z{value}" for value in range(10))


def knowledge_rows(facts: int = 8) -> dict[str, list[dict[str, str]]]:
    if not 1 <= facts <= 10:
        raise ValueError("knowledge fact count must be in [1, 10]")
    train: list[dict[str, str]] = []
    evaluation: list[dict[str, str]] = []
    for index in range(1, facts + 1):
        subject = f"CowNode-{index:03d}"
        answer = KNOWLEDGE_CANDIDATES[index - 1]
        train.extend(
            [
                {
                    "id": f"knowledge.{index:03d}.train.a",
                    "question": f"What COW code is assigned to {subject}?",
                    "answer": answer,
                },
                {
                    "id": f"knowledge.{index:03d}.train.b",
                    "question": f"Give the registered COW code for {subject}.",
                    "answer": answer,
                },
                {
                    "id": f"knowledge.{index:03d}.train.c",
                    "question": f"{subject} uses which COW code?",
                    "answer": answer,
                },
            ]
        )
        evaluation.extend(
            [
                {
                    "id": f"knowledge.{index:03d}.eval.a",
                    "question": f"Which COW code belongs to {subject}?",
                    "answer": answer,
                },
                {
                    "id": f"knowledge.{index:03d}.eval.b",
                    "question": f"Identify {subject}'s COW identifier.",
                    "answer": answer,
                },
            ]
        )
    return {"train": train, "evaluation": evaluation}


def _zor(a: int, b: int) -> str:
    return f"Z{(2 * a + 3 * b) % 10}"


def capability_rows() -> dict[str, list[dict[str, str]]]:
    """Invented rule acquisition with held-out operand combinations."""
    train: list[dict[str, str]] = []
    evaluation: list[dict[str, str]] = []
    for a in range(10):
        for b in range(10):
            row: dict[str, Any] = {
                "question": f"Apply the new ZOR operation to {a} and {b}.",
                "answer": _zor(a, b),
            }
            target = evaluation if (a + 2 * b) % 5 == 0 else train
            split = "eval" if target is evaluation else "train"
            row["id"] = f"capability.{split}.{a}.{b}"
            target.append({key: str(value) for key, value in row.items()})
    return {"train": train, "evaluation": evaluation}


def track_rows(track: str, *, facts: int = 8) -> dict[str, list[dict[str, str]]]:
    if track == "knowledge":
        return knowledge_rows(facts)
    if track == "capability":
        return capability_rows()
    raise ValueError(f"unknown COW-CLM track: {track}")


def track_candidates(track: str) -> tuple[str, ...]:
    if track == "knowledge":
        return KNOWLEDGE_CANDIDATES
    if track == "capability":
        return CAPABILITY_CANDIDATES
    raise ValueError(f"unknown COW-CLM track: {track}")
