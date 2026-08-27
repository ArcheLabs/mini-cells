from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch


CANDIDATES = (
    "TRANSFORM",
    "PARITY",
    "MODSUM",
    "SORT",
    "ROTATE",
    "DELAY_COPY",
    "STATE_MACHINE",
)
SCREEN_STEPS = 128
SCREEN_VALIDATION_BATCHES = 16
SCREEN_LEARNABILITY_MIN = 0.02
SCREEN_INDEPENDENCE_MIN = 0.01
SCREEN_ABSORPTION_RATIO_MAX = 0.50
SCREEN_QUALIFY_REPLICATES_MIN = 2
STRUCTURAL_COST_FRACTION = 0.005
WEAK_SELECTED_STEPS = 256
STRONG_SELECTED_STEPS = 256
PROPOSAL_BATCHES = 64
PROBATION_WINDOWS = 4
STEPS_PER_WINDOW = 64
ROUTING_PURITY_MIN = 0.75
POSITIVE_REPLICATES_MIN = 2
MAX_TRAITS = 4


@dataclass(frozen=True)
class ScreeningScore:
    baseline_candidate_nll: float
    existing_candidate_nll: float
    newborn_candidate_nll: float
    baseline_arithmetic_nll: float
    existing_arithmetic_nll: float
    existing_candidate_gain: float
    arithmetic_damage: float
    existing_value: float
    newborn_candidate_gain: float
    newborn_value: float
    independence_advantage: float
    absorption_ratio: float
    qualifies: bool


@dataclass(frozen=True)
class CandidateSelection:
    selected: str
    qualified: bool
    qualifying_candidates: tuple[str, ...]
    median_independence: dict[str, float]
    qualifying_replicates: dict[str, int]


def screening_score(
    *,
    baseline_candidate_nll: float,
    existing_candidate_nll: float,
    newborn_candidate_nll: float,
    baseline_arithmetic_nll: float,
    existing_arithmetic_nll: float,
) -> ScreeningScore:
    scale_candidate = max(abs(float(baseline_candidate_nll)), 1e-8)
    scale_arithmetic = max(abs(float(baseline_arithmetic_nll)), 1e-8)
    existing_gain = (baseline_candidate_nll - existing_candidate_nll) / scale_candidate
    arithmetic_damage = max((existing_arithmetic_nll - baseline_arithmetic_nll) / scale_arithmetic, 0.0)
    existing_value = existing_gain - arithmetic_damage
    newborn_gain = (baseline_candidate_nll - newborn_candidate_nll) / scale_candidate
    newborn_value = newborn_gain - STRUCTURAL_COST_FRACTION
    independence = newborn_value - existing_value
    if newborn_value <= 1e-12:
        absorption_ratio = float("inf") if existing_value > 0.0 else 0.0
    else:
        absorption_ratio = max(existing_value, 0.0) / newborn_value
    qualifies = bool(
        newborn_gain >= SCREEN_LEARNABILITY_MIN
        and independence >= SCREEN_INDEPENDENCE_MIN
        and absorption_ratio <= SCREEN_ABSORPTION_RATIO_MAX
    )
    return ScreeningScore(
        baseline_candidate_nll=float(baseline_candidate_nll),
        existing_candidate_nll=float(existing_candidate_nll),
        newborn_candidate_nll=float(newborn_candidate_nll),
        baseline_arithmetic_nll=float(baseline_arithmetic_nll),
        existing_arithmetic_nll=float(existing_arithmetic_nll),
        existing_candidate_gain=float(existing_gain),
        arithmetic_damage=float(arithmetic_damage),
        existing_value=float(existing_value),
        newborn_candidate_gain=float(newborn_gain),
        newborn_value=float(newborn_value),
        independence_advantage=float(independence),
        absorption_ratio=float(absorption_ratio),
        qualifies=qualifies,
    )


def select_candidate(rows: list[dict[str, object]]) -> CandidateSelection:
    if not rows:
        raise ValueError("screening rows are required")
    by_candidate: dict[str, list[dict[str, object]]] = {name: [] for name in CANDIDATES}
    for row in rows:
        name = str(row["candidate"])
        if name not in by_candidate:
            raise ValueError(f"unknown screening candidate: {name}")
        by_candidate[name].append(row)
    if any(not values for values in by_candidate.values()):
        missing = [name for name, values in by_candidate.items() if not values]
        raise ValueError(f"missing screening candidates: {missing}")
    medians = {
        name: float(np.median([float(row["independence_advantage"]) for row in values]))
        for name, values in by_candidate.items()
    }
    counts = {
        name: sum(int(bool(row["qualifies"])) for row in values)
        for name, values in by_candidate.items()
    }
    qualifying = tuple(
        sorted(name for name in CANDIDATES if counts[name] >= SCREEN_QUALIFY_REPLICATES_MIN)
    )
    pool = qualifying if qualifying else CANDIDATES
    selected = min(pool, key=lambda name: (-medians[name], name))
    return CandidateSelection(
        selected=selected,
        qualified=selected in qualifying,
        qualifying_candidates=qualifying,
        median_independence=medians,
        qualifying_replicates=counts,
    )


def selected_stage_schedule(candidate: str, *, weak: bool, replicate: int) -> tuple[str, ...]:
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    if weak:
        counts = {"STORY": 115, "ARITH_A": 115, candidate: 26}
        seed = 424_100 + replicate
    else:
        counts = {"STORY": 86, "ARITH_A": 85, candidate: 85}
        seed = 424_200 + replicate
    values = [key for key, count in counts.items() for _ in range(count)]
    rng = random.Random(seed)
    rng.shuffle(values)
    return tuple(values)


def expected_trajectory() -> tuple[int, ...]:
    return (1, 2, 2, 3)


def classify_replicate(
    *,
    arithmetic_birth: bool,
    weak_reject: bool,
    strong_birth: bool,
    final_k: int,
) -> dict[str, int]:
    return {
        "arithmetic_birth": int(arithmetic_birth),
        "weak_reject": int(weak_reject),
        "strong_birth": int(strong_birth),
        "final_k": int(final_k),
    }


def aggregate_status(
    per_replicate: list[dict[str, int]],
    *,
    screening_qualified: bool,
) -> str:
    arithmetic = sum(int(row["arithmetic_birth"]) for row in per_replicate)
    weak = sum(int(row["weak_reject"]) for row in per_replicate)
    strong = sum(int(row["strong_birth"]) for row in per_replicate)
    final_k3 = sum(int(row["final_k"] == 3) for row in per_replicate)
    if not screening_qualified:
        return "NO_FUNCTIONALLY_INDEPENDENT_THIRD_CAPABILITY"
    if weak < len(per_replicate):
        return "INDEPENDENT_CAPABILITY_CAUSES_EARLY_BIRTH"
    if arithmetic < POSITIVE_REPLICATES_MIN:
        return "NO_STABLE_FIRST_TRAIT_BIRTH"
    if strong >= POSITIVE_REPLICATES_MIN and final_k3 >= POSITIVE_REPLICATES_MIN:
        return "INDEPENDENT_THIRD_TRAIT_GENESIS_SIGNAL"
    return "INDEPENDENT_CAPABILITY_WITHOUT_THIRD_TRAIT_GENESIS"


def _candidate_text(name: str, rng: random.Random) -> str:
    if name == "TRANSFORM":
        values = [rng.randrange(10) for _ in range(6)]
        src = " ".join(map(str, values))
        dst = " ".join(map(str, reversed(values)))
        return f"Transform sequence {src}. Reverse answer {dst}."
    if name == "PARITY":
        values = [rng.randrange(2) for _ in range(10)]
        parity = "ODD" if sum(values) % 2 else "EVEN"
        return f"Parity bits {' '.join(map(str, values))}. Answer {parity}."
    if name == "MODSUM":
        values = [rng.randrange(10) for _ in range(6)]
        answer = sum(values) % 7
        return f"Modulo seven sum {' '.join(map(str, values))}. Answer {answer}."
    if name == "SORT":
        values = [rng.randrange(10) for _ in range(6)]
        return f"Sort digits {' '.join(map(str, values))}. Answer {' '.join(map(str, sorted(values)))}."
    if name == "ROTATE":
        values = [rng.randrange(10) for _ in range(6)]
        target = values[2:] + values[:2]
        return f"Rotate left two {' '.join(map(str, values))}. Answer {' '.join(map(str, target))}."
    if name == "DELAY_COPY":
        source = [rng.randrange(10) for _ in range(4)]
        distractor = [rng.randrange(10) for _ in range(8)]
        return (
            f"Remember {' '.join(map(str, source))}. Distractor {' '.join(map(str, distractor))}. "
            f"Recall {' '.join(map(str, source))}."
        )
    if name == "STATE_MACHINE":
        values = [rng.randrange(2) for _ in range(12)]
        state = 0
        for bit in values:
            state = (2 * state + bit + 1) % 3
        return f"State machine bits {' '.join(map(str, values))}. Final state {state}."
    raise ValueError(name)


def _stream(
    tokenizer: object,
    *,
    name: str,
    target_tokens: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain EOS")
    rng = random.Random(seed)
    tokens: list[int] = []
    examples = 0
    while len(tokens) < target_tokens:
        encoded = tokenizer.encode(_candidate_text(name, rng)).ids
        if encoded:
            tokens.extend(encoded)
            tokens.append(int(eos_id))
            examples += 1
    return torch.tensor(tokens[:target_tokens], dtype=torch.long), examples


def _sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def prepare_candidate_caches(cache_dir: Path, tokenizer: object) -> dict[str, dict[str, object]]:
    root = cache_dir / "independent-third-trait-candidates"
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, object]] = {}
    for index, name in enumerate(CANDIDATES):
        candidate_root = root / name.lower()
        candidate_root.mkdir(parents=True, exist_ok=True)
        train_path = candidate_root / "train-tokens.pt"
        validation_path = candidate_root / "validation-tokens.pt"
        manifest_path = candidate_root / "manifest.json"
        expected = {
            "format": "minicells.independent-third-trait-candidate.v1",
            "candidate": name,
            "train_tokens": 160_000,
            "validation_tokens": 40_000,
            "seed": 24_500 + 100 * index,
        }
        valid = False
        if train_path.is_file() and validation_path.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train = torch.load(train_path, map_location="cpu")
            validation = torch.load(validation_path, map_location="cpu")
            valid = bool(
                all(manifest.get(key) == value for key, value in expected.items())
                and manifest.get("train_sha256") == _sha256(train)
                and manifest.get("validation_sha256") == _sha256(validation)
            )
        if not valid:
            train, train_examples = _stream(
                tokenizer, name=name, target_tokens=expected["train_tokens"], seed=expected["seed"]
            )
            validation, validation_examples = _stream(
                tokenizer,
                name=name,
                target_tokens=expected["validation_tokens"],
                seed=expected["seed"] + 1,
            )
            torch.save(train, train_path)
            torch.save(validation, validation_path)
            manifest = {
                **expected,
                "train_examples": train_examples,
                "validation_examples": validation_examples,
                "train_sha256": _sha256(train),
                "validation_sha256": _sha256(validation),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result[name] = {
            "train": train,
            "validation": validation,
            "manifest": manifest,
            "manifest_path": manifest_path,
        }
    return result
