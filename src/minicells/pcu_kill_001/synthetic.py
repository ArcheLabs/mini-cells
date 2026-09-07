"""Deterministic, tokenizer-aware synthetic world and leakage audit."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


# Training prompts intentionally exclude the answer identifier. The answer is
# appended exactly once by task.build_task_sequences and is the only supervised
# completion. Including V/W here would reduce the experiment to copying a
# target that is already visible in context.
TRAIN_TEMPLATES = {
    "A": "Under Ledger A, resolve {u}. Relay identifier:",
    "B": "Under Ledger B, resolve {v}. Terminal identifier:",
}
EVAL_TEMPLATES = {
    "A": "Resolve {u} through rule A. Return only the relay identifier.",
    "B": "Resolve {v} through rule B. Return only the terminal identifier.",
    "AB": "Start from {u}. Apply Ledger A, then Ledger B. Return relay and terminal.",
}

POSITIVE_CONTROL_VERSION = "pcu-kill-001-context-oracle-v2"
POSITIVE_CONTROL_FLOOR = 0.90
POSITIVE_CONTROL_CANDIDATES = 16
FREE_GENERATION_DIAGNOSTIC_SAMPLES = 8


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
    generator_version: str = "pcu-kill-001-world-v2"

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
        # A stable three-token stand-in for the no-tokenizer unit harness. A
        # real run must pass the frozen Granite tokenizer instead.
        digest = hashlib.sha256(identifier.encode()).digest()
        return tuple(int(value) for value in digest[:3])
    encoded = tokenizer.encode(identifier, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    return tuple(int(value) for value in encoded)


def _make_identifier(
    seed: int,
    namespace: str,
    index: int,
    tokenizer: Any | None,
    used: set[str],
) -> tuple[str, tuple[int, ...]]:
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


def _sample(
    sample_id: str,
    split: str,
    prompt: str,
    answer: str,
    namespace: str,
    identifiers: Iterable[str],
    token_ids: Iterable[int],
    pair_id: int,
) -> SyntheticSample:
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
            TRAIN_TEMPLATES["A"].format(u=triple.u), triple.v, "A",
            (triple.u, triple.v), triple.u_token_ids + triple.v_token_ids, triple.index,
        ))
        b_train.append(_sample(
            f"B-train-{triple.index:04d}", "B_train",
            TRAIN_TEMPLATES["B"].format(v=triple.v), triple.w, "B",
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
    return SyntheticWorld(
        seed=int(value["seed"]),
        triples=triples,
        splits=splits,
        generator_version=str(value.get("generator_version", "pcu-kill-001-world-v2")),
    )


def audit_dataset(world: SyntheticWorld | Mapping[str, Any]) -> DatasetAudit:
    """Check all leakage invariants before any branch worker can train."""
    if not isinstance(world, SyntheticWorld):
        world = _world_from_mapping(world)
    us = {item.u for item in world.triples}
    vs = {item.v for item in world.triples}
    ws = {item.w for item in world.triples}
    a_train = world.splits.get("A_train", [])
    b_train = world.splits.get("B_train", [])
    eval_samples = (
        world.splits.get("A_eval", [])
        + world.splits.get("B_eval", [])
        + world.splits.get("AB_eval", [])
    )
    train_samples = a_train + b_train
    checks: dict[str, bool] = {}

    def text(item: SyntheticSample) -> str:
        return f"{item.prompt} {item.answer}"

    checks["A_contains_no_W"] = all(not any(value in text(item) for value in ws) for item in a_train)
    checks["B_contains_no_U"] = all(not any(value in text(item) for value in us) for item in b_train)
    checks["A_answer_absent_from_prompt"] = all(item.answer not in item.prompt for item in a_train)
    checks["B_answer_absent_from_prompt"] = all(item.answer not in item.prompt for item in b_train)
    checks["no_UW_training_pair"] = all(
        not any(value in text(item) for value in us) or not any(value in text(item) for value in ws)
        for item in train_samples
    )
    checks["no_composition_training_examples"] = all(
        item.namespace != "AB"
        and not (any(value in text(item) for value in us) and any(value in text(item) for value in ws))
        for item in train_samples
    )
    checks["no_duplicate_eval_prompt_in_training"] = not (
        {item.prompt for item in eval_samples} & {item.prompt for item in train_samples}
    )
    all_identifiers = list(us | vs | ws)
    checks["identifier_namespaces_disjoint"] = len(all_identifiers) == len(set(all_identifiers)) and all(
        not (left != right and (left in right or right in left))
        for index, left in enumerate(all_identifiers)
        for right in all_identifiers[index + 1:]
    )
    checks["all_expected_samples_accounted"] = (
        len(world.triples) > 0
        and len(a_train) == len(world.triples)
        and len(b_train) == len(world.triples)
        and len(world.splits.get("A_eval", [])) == len(world.triples)
        and len(world.splits.get("B_eval", [])) == len(world.triples)
        and len(world.splits.get("AB_eval", [])) == len(world.triples)
    )
    checks["identifier_format_is_opaque"] = all(
        re.fullmatch(r"[UVW][A-Z2-9]{4}", value) for value in all_identifiers
    )
    errors = tuple(name for name, passed in checks.items() if not passed)
    return DatasetAudit(
        bool(checks) and not errors,
        checks,
        errors,
        {name: len(values) for name, values in world.splits.items()},
    )


def _encode_ids(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    value = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if hasattr(value, "ids"):
        value = value.ids
    return [int(item) for item in value]


def _completion_encoding(tokenizer: Any, prompt: str, candidate: str) -> tuple[list[int], list[int]]:
    """Encode the exact prompt+completion string and isolate completion tokens.

    BPE tokenization can change at whitespace boundaries. We therefore never
    concatenate separately-tokenized candidate IDs. The prompt deliberately
    ends in punctuation, the completion begins with one space, and the full
    encoding must preserve the prompt encoding as an exact prefix.
    """
    if prompt != prompt.rstrip():
        raise ValueError("positive-control prompt must not end in whitespace")
    prompt_ids = _encode_ids(tokenizer, prompt, add_special_tokens=True)
    full_ids = _encode_ids(tokenizer, prompt + " " + str(candidate), add_special_tokens=True)
    if not prompt_ids:
        raise ValueError("positive-control prompt encoded to zero tokens")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("positive-control tokenizer boundary is not prefix-stable")
    completion_ids = full_ids[len(prompt_ids):]
    if not completion_ids:
        raise ValueError("positive-control completion encoded to zero tokens")
    return full_ids, completion_ids


def _candidate_pool(
    values: Sequence[str],
    correct: str,
    sample_id: str,
    size: int = POSITIVE_CONTROL_CANDIDATES,
) -> tuple[str, ...]:
    """Return a deterministic, position-unbiased candidate set containing correct."""
    unique = list(dict.fromkeys(str(value) for value in values))
    if correct not in unique:
        raise ValueError("positive-control correct candidate is not in the candidate universe")
    size = min(max(1, int(size)), len(unique))

    def key(value: str) -> str:
        return hashlib.sha256(f"{POSITIVE_CONTROL_VERSION}:{sample_id}:{value}".encode()).hexdigest()

    distractors = sorted((value for value in unique if value != correct), key=key)
    selected = [correct, *distractors[: max(0, size - 1)]]
    return tuple(sorted(selected, key=key))


def _teacher_forced_candidate_scores(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: Sequence[str],
    *,
    device: str,
) -> list[float]:
    """Mean exact-completion-token log-likelihood for constrained candidates."""
    import torch

    encoded = [_completion_encoding(tokenizer, prompt, str(candidate)) for candidate in candidates]
    full = [item[0] for item in encoded]
    completion_ids = [item[1] for item in encoded]
    prompt_length = len(full[0]) - len(completion_ids[0])
    if any(len(full_ids) - len(answer_ids) != prompt_length for full_ids, answer_ids in encoded):
        raise RuntimeError("positive-control candidates do not share one prompt-token boundary")

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    width = max(len(value) for value in full)
    input_ids = torch.full((len(full), width), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros((len(full), width), dtype=torch.long, device=device)
    for row, values in enumerate(full):
        input_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        attention[row, : len(values)] = 1
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention)
        logits = getattr(output, "logits", output)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("positive-control model output has no logits tensor")
        log_probs = torch.log_softmax(logits.float(), dim=-1)
    scores: list[float] = []
    for row, answer_ids in enumerate(completion_ids):
        positions = torch.arange(
            prompt_length - 1,
            prompt_length + len(answer_ids) - 1,
            device=device,
        )
        targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
        values = log_probs[row, positions, targets]
        if not torch.isfinite(values).all():
            raise RuntimeError("non-finite positive-control candidate score")
        scores.append(float(values.mean()))
    return scores


def _rank_candidate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: Sequence[str],
    correct: str,
    *,
    device: str,
) -> dict[str, Any]:
    scores = _teacher_forced_candidate_scores(
        model, tokenizer, prompt, candidates, device=device
    )
    rows = sorted(
        zip((str(value) for value in candidates), scores),
        key=lambda item: (-item[1], item[0]),
    )
    winner, winner_score = rows[0]
    correct_rank = next(index + 1 for index, item in enumerate(rows) if item[0] == correct)
    correct_score = next(item[1] for item in rows if item[0] == correct)
    runner_up = rows[1][1] if len(rows) > 1 else winner_score
    return {
        "winner": winner,
        "correct": correct,
        "correct_rank": int(correct_rank),
        "correct_score": float(correct_score),
        "winner_score": float(winner_score),
        "winner_margin": float(winner_score - runner_up),
        "exact": winner == correct,
    }


def _free_generation_diagnostic(
    world: SyntheticWorld,
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Keep free generation as a small diagnostic, never as the base-model gate."""
    from .evaluation import greedy_generate

    rows = []
    for sample in world.splits.get("AB_eval", [])[:FREE_GENERATION_DIAGNOSTIC_SAMPLES]:
        triple = world.triples[int(sample.pair_id)]
        prompt = (
            f"Ledger A: {triple.u} maps to {triple.v}.\n"
            f"Ledger B: {triple.v} maps to {triple.w}.\n"
            f"Start from {triple.u}. Apply Ledger A, then Ledger B. "
            "Return relay and terminal."
        )
        generated = greedy_generate(
            model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = re.findall(r"[UVW][A-Z2-9]{4}", generated.upper())
        relay = generated_ids[0] if generated_ids else None
        terminal = generated_ids[1] if len(generated_ids) > 1 else None
        rows.append({
            "sample_id": sample.sample_id,
            "expected_relay": triple.v,
            "expected_terminal": triple.w,
            "generated_text": generated,
            "generated_identifiers": generated_ids,
            "relay_exact": relay == triple.v,
            "terminal_exact": terminal == triple.w,
            "both_exact": relay == triple.v and terminal == triple.w,
        })
    n = max(1, len(rows))
    return {
        "sample_count": len(rows),
        "accuracy": sum(bool(row["both_exact"]) for row in rows) / n,
        "rows": rows,
        "gate": False,
    }


def context_oracle(
    world: SyntheticWorld | Mapping[str, Any],
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: str = "cpu",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Evaluate base-model testbed capacity before any Cell mutation.

    Model-backed v2 uses teacher-forced candidate ranking for three explicit
    positive controls: A retrieval, B retrieval, and two-hop V+W composition.
    The free-generation version is retained only as a diagnostic because this
    experiment uses a base checkpoint rather than an instruction-tuned model.
    """
    if not isinstance(world, SyntheticWorld):
        world = _world_from_mapping(world)
    if model is None and tokenizer is None:
        rows = [
            {
                "sample_id": sample.sample_id,
                "retrieval_a_exact": True,
                "retrieval_b_exact": True,
                "relay_exact": True,
                "terminal_exact": True,
                "both_exact": True,
            }
            for sample in world.splits.get("AB_eval", [])
        ]
        return {
            "schema": "minicells.pcu-kill-001.context-oracle.v2",
            "positive_control_version": POSITIVE_CONTROL_VERSION,
            "mode": "symbolic_reference",
            "candidate_pool_size": None,
            "retrieval_a_accuracy": 1.0,
            "retrieval_b_accuracy": 1.0,
            "composition_accuracy": 1.0,
            "accuracy": 1.0,
            "relay_accuracy": 1.0,
            "terminal_accuracy": 1.0,
            "rows": rows,
            "free_generation_diagnostic": None,
            "passed": True,
            "scientific_evidence": False,
        }
    if model is None or tokenizer is None:
        raise ValueError("context oracle requires both model and tokenizer")

    vs = [item.v for item in world.triples]
    ws = [item.w for item in world.triples]
    pairs = [f"{item.v} {item.w}" for item in world.triples]
    rows: list[dict[str, Any]] = []
    for sample in world.splits.get("AB_eval", []):
        triple = world.triples[int(sample.pair_id)]
        prompt_a = (
            f"Mapping record:\n{triple.u} -> {triple.v}\n"
            f"Query:\n{triple.u} ->\nAnswer:"
        )
        prompt_b = (
            f"Mapping record:\n{triple.v} -> {triple.w}\n"
            f"Query:\n{triple.v} ->\nAnswer:"
        )
        prompt_ab = (
            f"Mapping records:\n{triple.u} -> {triple.v}\n{triple.v} -> {triple.w}\n"
            f"Query path:\n{triple.u} ->\nRelay and terminal:"
        )
        a_candidates = _candidate_pool(vs, triple.v, f"{sample.sample_id}:A")
        b_candidates = _candidate_pool(ws, triple.w, f"{sample.sample_id}:B")
        pair_correct = f"{triple.v} {triple.w}"
        pair_candidates = _candidate_pool(
            pairs, pair_correct, f"{sample.sample_id}:AB"
        )
        rank_a = _rank_candidate(
            model, tokenizer, prompt_a, a_candidates, triple.v, device=device
        )
        rank_b = _rank_candidate(
            model, tokenizer, prompt_b, b_candidates, triple.w, device=device
        )
        rank_ab = _rank_candidate(
            model, tokenizer, prompt_ab, pair_candidates, pair_correct, device=device
        )
        predicted_pair = rank_ab["winner"].split()
        predicted_relay = predicted_pair[0] if predicted_pair else None
        predicted_terminal = predicted_pair[-1] if len(predicted_pair) >= 2 else None
        rows.append({
            "sample_id": sample.sample_id,
            "expected_relay": triple.v,
            "expected_terminal": triple.w,
            "retrieval_a": rank_a,
            "retrieval_b": rank_b,
            "composition": rank_ab,
            "retrieval_a_exact": bool(rank_a["exact"]),
            "retrieval_b_exact": bool(rank_b["exact"]),
            "relay_exact": predicted_relay == triple.v,
            "terminal_exact": predicted_terminal == triple.w,
            "both_exact": bool(rank_ab["exact"]),
        })

    n = max(1, len(rows))
    retrieval_a_accuracy = sum(bool(row["retrieval_a_exact"]) for row in rows) / n
    retrieval_b_accuracy = sum(bool(row["retrieval_b_exact"]) for row in rows) / n
    composition_accuracy = sum(bool(row["both_exact"]) for row in rows) / n
    relay_accuracy = sum(bool(row["relay_exact"]) for row in rows) / n
    terminal_accuracy = sum(bool(row["terminal_exact"]) for row in rows) / n
    passed = min(
        retrieval_a_accuracy,
        retrieval_b_accuracy,
        composition_accuracy,
    ) >= POSITIVE_CONTROL_FLOOR
    return {
        "schema": "minicells.pcu-kill-001.context-oracle.v2",
        "positive_control_version": POSITIVE_CONTROL_VERSION,
        "mode": "model_backed_teacher_forced_candidate_ranking",
        "candidate_pool_size": min(POSITIVE_CONTROL_CANDIDATES, len(world.triples)),
        "threshold": POSITIVE_CONTROL_FLOOR,
        "retrieval_a_accuracy": retrieval_a_accuracy,
        "retrieval_b_accuracy": retrieval_b_accuracy,
        "composition_accuracy": composition_accuracy,
        # Backwards-compatible primary metric consumed by the decision layer.
        "accuracy": composition_accuracy,
        "relay_accuracy": relay_accuracy,
        "terminal_accuracy": terminal_accuracy,
        "rows": rows,
        "free_generation_diagnostic": _free_generation_diagnostic(
            world,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
        ),
        "passed": passed,
        "scientific_evidence": False,
    }
