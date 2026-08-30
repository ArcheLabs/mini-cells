"""Deterministic CLM-0.4-mini math/story curriculum and text generators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Iterable

from .protocol import canonical_json_hash


CURRICULUM_VERSION = "clm-0.4-mini-curriculum-v1"
MATH_FAMILIES = (
    "multiplication",
    "exact-integer-division",
    "modulo",
    "precedence-two-step",
    "bounded-affine",
    "synthetic-binary-operator",
)
STORY_NAMES = (
    "Mira", "Jon", "Lena", "Oren", "Sana", "Ivo", "Nia", "Pavel",
    "Rhea", "Tomas", "Uma", "Vera", "Wren", "Xavi", "Yara", "Zane",
    "Ari", "Bea", "Cato", "Dara", "Eli", "Faye", "Galen", "Hana",
)
CITIES = (
    "Luma", "Sora", "Neris", "Vela", "Orin", "Pavo", "Tera", "Maro",
    "Kiva", "Duna", "Rilo", "Sena", "Boro", "Yuni", "Faro", "Cira",
)
PETS = ("fox", "cat", "dog", "owl", "rabbit", "goat", "otter", "parrot")
JOBS = ("baker", "teacher", "gardener", "librarian", "painter", "carpenter", "nurse", "cook")


@dataclass(frozen=True)
class TransactionSpec:
    transaction_id: int
    domain: str
    family: str
    address_id: str
    visit_index: int
    operation: str
    knowledge_key: str | None
    supersedes_key: str | None
    payload: dict
    data_seed: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TextExample:
    example_id: str
    address_id: str
    prompt: str
    answer: str
    knowledge_key: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _math_addresses() -> list[tuple[str, str, int]]:
    return [
        (f"math/{family}/{variant}", family, variant)
        for family in MATH_FAMILIES
        for variant in range(2)
    ]


def build_curriculum() -> dict:
    """Return the frozen 192-transaction semantic schedule.

    Math and story updates alternate. Revisit waves are spaced across the stream
    so private-bundle reuse is measured over meaningful time gaps.
    """
    math_specs: list[TransactionSpec] = []
    addresses = _math_addresses()
    for visit in range(8):
        for address_id, family, variant in addresses:
            payload = {"variant": variant, "difficulty": visit}
            math_specs.append(
                TransactionSpec(
                    transaction_id=-1,
                    domain="math",
                    family=family,
                    address_id=address_id,
                    visit_index=visit,
                    operation="capability",
                    knowledge_key=None,
                    supersedes_key=None,
                    payload=payload,
                    data_seed=_stable_seed(
                        f"{CURRICULUM_VERSION}|math|{address_id}|{visit}"
                    ),
                )
            )

    story_specs: list[TransactionSpec] = []
    for visit in range(4):
        for world in range(24):
            name = STORY_NAMES[world]
            address_id = f"story/world-{world:02d}"
            if visit == 0:
                key = f"world-{world:02d}:location:{name.lower()}"
                operation = "append"
                payload = {"entity": name, "attribute": "location", "value": CITIES[world % len(CITIES)]}
                supersedes = None
            elif visit == 1:
                key = f"world-{world:02d}:pet:{name.lower()}"
                operation = "append"
                payload = {"entity": name, "attribute": "pet", "value": PETS[(world + 3) % len(PETS)]}
                supersedes = None
            elif visit == 2:
                key = f"world-{world:02d}:location:{name.lower()}"
                operation = "supersede"
                prior = CITIES[world % len(CITIES)]
                value = CITIES[(world + 5) % len(CITIES)]
                if value == prior:
                    value = CITIES[(world + 6) % len(CITIES)]
                payload = {"entity": name, "attribute": "location", "value": value, "prior_value": prior}
                supersedes = key
            else:
                key = f"world-{world:02d}:job:{name.lower()}"
                operation = "append"
                payload = {"entity": name, "attribute": "job", "value": JOBS[(world + 2) % len(JOBS)]}
                supersedes = None
            story_specs.append(
                TransactionSpec(
                    transaction_id=-1,
                    domain="story",
                    family="micro-world-fact",
                    address_id=address_id,
                    visit_index=visit,
                    operation=operation,
                    knowledge_key=key,
                    supersedes_key=supersedes,
                    payload=payload,
                    data_seed=_stable_seed(
                        f"{CURRICULUM_VERSION}|story|{address_id}|{visit}|{key}"
                    ),
                )
            )

    if len(math_specs) != 96 or len(story_specs) != 96:
        raise AssertionError("registered curriculum cardinality changed")
    ordered: list[dict] = []
    transaction_id = 0
    for math_spec, story_spec in zip(math_specs, story_specs, strict=True):
        for spec in (math_spec, story_spec):
            payload = spec.to_dict()
            payload["transaction_id"] = transaction_id
            ordered.append(payload)
            transaction_id += 1
    manifest = {
        "format": "minicells.clm-0.4-mini.curriculum-manifest.v1",
        "generator_version": CURRICULUM_VERSION,
        "transactions": ordered,
        "counts": {"total": 192, "math": 96, "story": 96},
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    return manifest


def transaction_specs(manifest: dict | None = None) -> list[TransactionSpec]:
    manifest = manifest or build_curriculum()
    return [TransactionSpec(**item) for item in manifest["transactions"]]


def _math_problem(spec: TransactionSpec, rng: random.Random, index: int) -> tuple[str, str]:
    difficulty = int(spec.payload["difficulty"])
    variant = int(spec.payload["variant"])
    upper = 8 + 3 * difficulty + 2 * variant
    family = spec.family
    if family == "multiplication":
        a, b = rng.randint(2, upper), rng.randint(2, upper)
        return f"Question: What is {a} times {b}? Answer:", f" {a * b}."
    if family == "exact-integer-division":
        b, q = rng.randint(2, upper), rng.randint(2, upper)
        return f"Question: What is {b * q} divided by {b}? Answer:", f" {q}."
    if family == "modulo":
        b = rng.randint(2, max(3, upper // 2))
        a = rng.randint(b + 1, upper * 3)
        return f"Question: What is {a} modulo {b}? Answer:", f" {a % b}."
    if family == "precedence-two-step":
        a, b, c = rng.randint(1, upper), rng.randint(1, upper), rng.randint(2, 6)
        if index % 2:
            return f"Question: Evaluate {a} + {b} * {c}. Answer:", f" {a + b * c}."
        return f"Question: Evaluate ({a} + {b}) * {c}. Answer:", f" {(a + b) * c}."
    if family == "bounded-affine":
        m = 2 + variant
        bias = 1 + variant
        x = rng.randint(0, upper)
        return (
            f"Question: Let f(x) = {m} * x + {bias}. What is f({x})? Answer:",
            f" {m * x + bias}.",
        )
    if family == "synthetic-binary-operator":
        word = "zor" if variant == 0 else "vek"
        a, b = rng.randint(1, upper), rng.randint(1, upper)
        extra = 1 + variant
        value = a * b + a + extra
        return (
            f"Definition: a {word} b means a * b + a + {extra}. Question: What is {a} {word} {b}? Answer:",
            f" {value}.",
        )
    raise ValueError(f"unknown math family {family}")


def _story_problem(spec: TransactionSpec, split: str, index: int) -> tuple[str, str]:
    entity = str(spec.payload["entity"])
    attribute = str(spec.payload["attribute"])
    value = str(spec.payload["value"])
    if split == "train":
        variants = {
            "location": [f"{entity} now lives in", f"The home city of {entity} is", f"{entity} resides in"],
            "pet": [f"{entity} has a pet", f"The pet owned by {entity} is a", f"{entity} cares for a"],
            "job": [f"{entity} works as a", f"The occupation of {entity} is", f"{entity}'s job is"],
        }
        prompt = variants[attribute][index % len(variants[attribute])]
        article = "" if attribute == "location" else ""
        return f"Statement: {prompt}", f" {article}{value}."
    if attribute == "location":
        questions = [f"Where does {entity} live?", f"What city is {entity}'s home?"]
        return f"Question: {questions[index % 2]} Answer:", f" {value}."
    if attribute == "pet":
        questions = [f"What pet does {entity} have?", f"What animal does {entity} own?"]
        return f"Question: {questions[index % 2]} Answer:", f" {value}."
    questions = [f"What is {entity}'s job?", f"What work does {entity} do?"]
    return f"Question: {questions[index % 2]} Answer:", f" {value}."


def materialize_examples(spec: TransactionSpec, *, split: str, count: int) -> list[TextExample]:
    if split not in {"train", "validation", "probe"}:
        raise ValueError("split must be train, validation, or probe")
    rng = random.Random(_stable_seed(f"{spec.data_seed}|{split}"))
    result: list[TextExample] = []
    for index in range(int(count)):
        if spec.domain == "math":
            prompt, answer = _math_problem(spec, rng, index)
        elif spec.domain == "story":
            prompt, answer = _story_problem(spec, "train" if split == "train" else split, index)
        else:
            raise ValueError(f"unknown domain {spec.domain}")
        result.append(
            TextExample(
                example_id=f"tx{spec.transaction_id:03d}:{split}:{index:03d}",
                address_id=spec.address_id,
                prompt=prompt,
                answer=answer,
                knowledge_key=spec.knowledge_key,
            )
        )
    return result


def materialize_transaction(spec: TransactionSpec, *, smoke: bool = False) -> dict[str, list[TextExample]]:
    counts = {"train": 8, "validation": 12, "probe": 6} if smoke else {
        "train": 64,
        "validation": 128,
        "probe": 32,
    }
    return {
        split: materialize_examples(spec, split=split, count=count)
        for split, count in counts.items()
    }


def manifest_json(manifest: dict | None = None) -> str:
    return json.dumps(manifest or build_curriculum(), indent=2, sort_keys=True) + "\n"
