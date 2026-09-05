"""Deterministic, tokenizer-aware synthetic world and leakage audit."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


TRAIN_TEMPLATES = {
    "A": "Under Ledger A, {u} maps to {v}.",
    "B": "Under Ledger B, {v} maps to {w}.",
}
EVAL_TEMPLATES = {
    "A": "Resolve {u} through rule A. Return only the relay identifier.",
    "B": "Resolve {v} through rule B. Return only the terminal identifier.",
    "AB": "Start from {u}. Apply Ledger A, then Ledger B. Return relay and terminal.",
}


@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str
    split: str
    prompt: str
    answer: str
    namespace: str
    identifiers: tuple[str, ...]
    token_ids: tuple[int, ...] = ()
    pair_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "prompt": self.prompt,
            "answer": self.answer,
            "namespace": self.namespace,
            "identifiers": list(self.identifiers),
            "token_ids": list(self.token_ids),
            "pair_id": self.pair_id,
        }


@dataclass(frozen=True)
class SyntheticTriple:
    index: int
    u: str
    v: str
    w: str
    u_token_ids: tuple[int, ...]
    v_token_ids: tuple[int, ...]
    w_token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "u": self.u,
            "v": self.v,
            "w": self.w,
            "u_token_ids": list(self.u_token_ids),
            "v_token_ids": list(self.v_token_ids),
            "w_token_ids": list(self.w_token_ids),
        }


@dataclass
class SyntheticWorld:
    seed: int
    triples: list[SyntheticTriple]
    splits: dict[str, list[SyntheticSample]]
    generator_version: str = "pcu-kill-001-world-v1"

    def all_samples(self) -> list[SyntheticSample]:
        return [sample for values in self.splits.values() for sample in values]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": "minicells.pcu-kill-001.dataset.v1",
            "generator_version": self.generator_version,
            "seed": self.seed,
            "triples": [item.to_dict() for item in self.triples],
            "splits": {
                name: [sample.to_dict() for sample in values]
                for name, values in sorted(self.splits.items())
            },
        }

    def manifest_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_manifest(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _candidate(seed: int, namespace: str, index: int, attempt: int) -> str:
    digest = hashlib.sha256(f"PCU-KILL-001:{seed}:{namespace}:{index}:{attempt}".encode()).digest()
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    body = "".join(alphabet[value % len(alphabet)] for value in digest[:4])
    return f"{namespace}{body}"


def _tokenize(identifier: str, tokenizer: Any | None) -> tuple[int, ...]:
    if tokenizer is None:
        # A stable three-token stand-in for the no-tokenizer unit harness.  A
        # real run must pass the frozen Granite tokenizer instead.
        digest = hashlib.sha256(identifier.encode()).digest()
        return tuple(int(value) for value in digest[:3])
    encoded = tokenizer.encode(identifier, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    return tuple(int(value) for value in encoded)


def _make_identifier(seed: int, namespace: str, index: int, tokenizer: Any | None, used: set[str]) -> tuple[str, tuple[int, ...]]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attempt in range(10_000):
        value = _candidate(seed, namespace, index, attempt)
        if value in used or any(value in other or other in value for other in used):
            continue
        token_ids = _tokenize(value, tokenizer)
        if not 2 <= len(token_ids) <= 4:
            continue
        if special_ids.intersection(token_ids):
            continue
        return value, token_ids
    raise RuntimeError(f"could not create a tokenizer-safe {namespace} identifier")


def _sample(sample_id: str, split: str, prompt: str, answer: str, namespace: str, identifiers: Iterable[str], token_ids: Iterable[int], pair_id: int) -> SyntheticSample:
    return SyntheticSample(
        sample_id=sample_id,
        split=split,
        prompt=prompt,
        answer=answer,
        namespace=namespace,
        identifiers=tuple(identifiers),
        token_ids=tuple(int(value) for value in token_ids),
        pair_id=pair_id,
    )


def generate_world(seed: int, count: int = 128, tokenizer: Any | None = None) -> SyntheticWorld:
    """Generate the same U/V/W world and samples for a given seed every time."""
    if count <= 0:
        raise ValueError("count must be positive")
    used: set[str] = set()
    triples: list[SyntheticTriple] = []
    for index in range(count):
        values = []
        for namespace in ("U", "V", "W"):
            value, token_ids = _make_identifier(seed, namespace, index, tokenizer, used)
            used.add(value)
            values.append((value, token_ids))
        triples.append(
            SyntheticTriple(
                index=index,
                u=values[0][0],
                v=values[1][0],
                w=values[2][0],
                u_token_ids=values[0][1],
                v_token_ids=values[1][1],
                w_token_ids=values[2][1],
            )
        )

    a_train: list[SyntheticSample] = []
    b_train: list[SyntheticSample] = []
    a_eval: list[SyntheticSample] = []
    b_eval: list[SyntheticSample] = []
    ab_eval: list[SyntheticSample] = []
    for triple in triples:
        a_train.append(_sample(
            f"A-train-{triple.index:04d}", "A_train",
            TRAIN_TEMPLATES["A"].format(u=triple.u, v=triple.v), triple.v, "A",
            (triple.u, triple.v), triple.u_token_ids + triple.v_token_ids, triple.index,
        ))
        b_train.append(_sample(
            f"B-train-{triple.index:04d}", "B_train",
            TRAIN_TEMPLATES["B"].format(v=triple.v, w=triple.w), triple.w, "B",
            (triple.v, triple.w), triple.v_token_ids + triple.w_token_ids, triple.index,
        ))
        a_eval.append(_sample(
            f"A-eval-{triple.index:04d}", "A_eval",
            EVAL_TEMPLATES["A"].format(u=triple.u), triple.v, "A",
            (triple.u,), triple.u_token_ids, triple.index,
        ))
        b_eval.append(_sample(
            f"B-eval-{triple.index:04d}", "B_eval",
            EVAL_TEMPLATES["B"].format(v=triple.v), triple.w, "B",
            (triple.v,), triple.v_token_ids, triple.index,
        ))
        ab_eval.append(_sample(
            f"AB-eval-{triple.index:04d}", "AB_eval",
            EVAL_TEMPLATES["AB"].format(u=triple.u), f"{triple.v} {triple.w}", "AB",
            (triple.u, triple.v, triple.w),
            triple.u_token_ids + triple.v_token_ids + triple.w_token_ids, triple.index,
        ))
    return SyntheticWorld(
        seed=int(seed),
        triples=triples,
        splits={
            "A_train": a_train,
            "B_train": b_train,
            "A_eval": a_eval,
            "B_eval": b_eval,
            "AB_eval": ab_eval,
        },
    )


@dataclass(frozen=True)
class DatasetAudit:
    passed: bool
    checks: dict[str, bool]
    errors: tuple[str, ...] = ()
    sample_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "minicells.pcu-kill-001.dataset-audit.v1",
            "passed": self.passed,
            "checks": dict(self.checks),
            "errors": list(self.errors),
            "sample_counts": dict(self.sample_counts),
        }


def _world_from_mapping(value: Mapping[str, Any]) -> SyntheticWorld:
    triples = [
        SyntheticTriple(
            index=int(item["index"]), u=str(item["u"]), v=str(item["v"]), w=str(item["w"]),
            u_token_ids=tuple(item.get("u_token_ids", [])),
            v_token_ids=tuple(item.get("v_token_ids", [])),
            w_token_ids=tuple(item.get("w_token_ids", [])),
        ) for item in value["triples"]
    ]
    splits = {}
    for name, values in value["splits"].items():
        splits[name] = [SyntheticSample(
            sample_id=str(item["sample_id"]), split=str(item["split"]),
            prompt=str(item["prompt"]), answer=str(item["answer"]),
            namespace=str(item["namespace"]), identifiers=tuple(item.get("identifiers", [])),
            token_ids=tuple(item.get("token_ids", [])), pair_id=item.get("pair_id"),
        ) for item in values]
    return SyntheticWorld(seed=int(value["seed"]), triples=triples, splits=splits,
                          generator_version=str(value.get("generator_version", "")))


def audit_dataset(world: SyntheticWorld | Mapping[str, Any]) -> DatasetAudit:
    """Check all leakage invariants before any branch worker can train."""
    if not isinstance(world, SyntheticWorld):
        world = _world_from_mapping(world)
    us = {item.u for item in world.triples}
    vs = {item.v for item in world.triples}
    ws = {item.w for item in world.triples}
    a_train = world.splits.get("A_train", [])
    b_train = world.splits.get("B_train", [])
    eval_samples = world.splits.get("A_eval", []) + world.splits.get("B_eval", []) + world.splits.get("AB_eval", [])
    train_samples = a_train + b_train
    checks: dict[str, bool] = {}
    def text(item: SyntheticSample) -> str:
        return f"{item.prompt} {item.answer}"

    checks["A_contains_no_W"] = all(not any(value in text(item) for value in ws) for item in a_train)
    checks["B_contains_no_U"] = all(not any(value in text(item) for value in us) for item in b_train)
    checks["no_UW_training_pair"] = all(
        not any(value in text(item) for value in us) or not any(value in text(item) for value in ws)
        for item in train_samples
    )
    checks["no_composition_training_examples"] = all(
        item.namespace != "AB"
        and not (any(value in text(item) for value in us) and any(value in text(item) for value in ws))
        for item in train_samples
    )
    checks["no_duplicate_eval_prompt_in_training"] = not ({item.prompt for item in eval_samples} & {item.prompt for item in train_samples})
    all_identifiers = list(us | vs | ws)
    checks["identifier_namespaces_disjoint"] = len(all_identifiers) == len(set(all_identifiers)) and all(
        not (left != right and (left in right or right in left)) for index, left in enumerate(all_identifiers) for right in all_identifiers[index + 1:]
    )
    checks["all_expected_samples_accounted"] = (
        len(world.triples) > 0
        and len(a_train) == len(world.triples)
        and len(b_train) == len(world.triples)
        and len(world.splits.get("A_eval", [])) == len(world.triples)
        and len(world.splits.get("B_eval", [])) == len(world.triples)
        and len(world.splits.get("AB_eval", [])) == len(world.triples)
    )
    checks["identifier_format_is_opaque"] = all(re.fullmatch(r"[UVW][A-Z2-9]{4}", value) for value in all_identifiers)
    errors = tuple(name for name, passed in checks.items() if not passed)
    return DatasetAudit(bool(checks) and not errors, checks, errors, {name: len(values) for name, values in world.splits.items()})
