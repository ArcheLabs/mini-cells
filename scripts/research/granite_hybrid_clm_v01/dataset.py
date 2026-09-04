from __future__ import annotations

from dataclasses import dataclass


CANDIDATE_CODES = (
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


@dataclass(frozen=True)
class DemoFact:
    index: int
    subject: str
    value: str

    @property
    def concept_id(self) -> str:
        return f"demo.fact.{self.index:03d}"


def demo_facts(count: int = 50) -> tuple[DemoFact, ...]:
    if not 1 <= count <= 50:
        raise ValueError("demo fact count must be in [1, 50]")
    return tuple(
        DemoFact(
            index=index,
            subject=f"ArcheNode-{index:03d}",
            value=CANDIDATE_CODES[(index - 1) % len(CANDIDATE_CODES)],
        )
        for index in range(1, count + 1)
    )


def training_rows(fact: DemoFact) -> list[dict[str, str]]:
    return [
        {
            "id": f"{fact.concept_id}.train.a",
            "concept_id": fact.concept_id,
            "question": f"What protocol code is assigned to {fact.subject}?",
            "answer": fact.value,
        },
        {
            "id": f"{fact.concept_id}.train.b",
            "concept_id": fact.concept_id,
            "question": f"Give the registered code for {fact.subject}.",
            "answer": fact.value,
        },
        {
            "id": f"{fact.concept_id}.train.c",
            "concept_id": fact.concept_id,
            "question": f"{fact.subject} uses which protocol code?",
            "answer": fact.value,
        },
    ]


def evaluation_rows(fact: DemoFact) -> list[dict[str, str]]:
    return [
        {
            "id": f"{fact.concept_id}.eval.a",
            "concept_id": fact.concept_id,
            "question": f"Which code belongs to {fact.subject}?",
            "answer": fact.value,
        },
        {
            "id": f"{fact.concept_id}.eval.b",
            "concept_id": fact.concept_id,
            "question": f"Identify {fact.subject}'s protocol identifier.",
            "answer": fact.value,
        },
    ]


def address_positive_prompts(fact: DemoFact) -> list[str]:
    return [row["question"] for row in training_rows(fact) + evaluation_rows(fact)]


def address_negative_prompts(
    fact: DemoFact,
    facts: tuple[DemoFact, ...],
    *,
    count: int = 10,
) -> list[str]:
    others = [item for item in facts if item.index != fact.index]
    selected = [others[(fact.index + offset) % len(others)] for offset in range(count)]
    prompts = [f"Which code belongs to {item.subject}?" for item in selected]
    prompts.extend(
        [
            f"What is the location of {fact.subject}?",
            f"Which entity uses code {fact.value}?",
        ]
    )
    return prompts


def update_rows(fact: DemoFact, new_value: str, version: str) -> dict[str, list[dict[str, str]]]:
    return {
        "train": [
            {
                "id": f"{fact.concept_id}.{version}.train.a",
                "concept_id": f"{fact.concept_id}.{version}",
                "question": f"In {version}, what protocol code is assigned to {fact.subject}?",
                "answer": new_value,
            },
            {
                "id": f"{fact.concept_id}.{version}.train.b",
                "concept_id": f"{fact.concept_id}.{version}",
                "question": f"Give {fact.subject}'s {version} protocol identifier.",
                "answer": new_value,
            },
        ],
        "evaluation": [
            {
                "id": f"{fact.concept_id}.{version}.eval.a",
                "concept_id": f"{fact.concept_id}.{version}",
                "question": f"Which code belongs to {fact.subject} under {version}?",
                "answer": new_value,
            },
            {
                "id": f"{fact.concept_id}.{version}.eval.b",
                "concept_id": f"{fact.concept_id}.{version}",
                "question": f"Identify the {version} code for {fact.subject}.",
                "answer": new_value,
            },
        ],
    }


def general_history_prompts() -> tuple[str, ...]:
    return (
        "What is the capital of France?",
        "What is two plus two?",
        "Complete the phrase: water freezes at",
        "Name one primary color.",
        "What planet is known as the Red Planet?",
        "What is the opposite of hot?",
        "Write the number after nine.",
        "What gas do humans breathe in to survive?",
        "Which ocean is the largest on Earth?",
        "What is the first month of the year?",
        "What shape has three sides?",
        "What animal is commonly called man's best friend?",
    )
