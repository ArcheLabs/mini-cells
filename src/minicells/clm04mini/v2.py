"""CLM-0.4-mini M1-v2 base alignment, dense baselines, and calibration.

M1-v2 is a protocol revision after development seed 90401 showed that the v1
base Story task did not match its admission task and that base admission used a
free-running metric inconsistent with M1 teacher-forced certification.

This module never opens v1 seed 90401. V2 calibration is seed 90402 only.
Formal seeds 90411/90412/90413 remain forbidden until a canonical v2 protocol
lock is committed.
"""

from __future__ import annotations

import copy
import csv
from concurrent.futures import ThreadPoolExecutor
import gc
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .calibration import (
    CalibrationCandidate,
    _candidate,
    _load,
    _load_base,
    _save_base,
    _summary_csv,
    _write,
    minimum_base_cell_activation,
    verify_committed_plan,
)
from .curriculum import TextExample, build_curriculum, transaction_specs
from .data import (
    BASE_CORPUS_VERSION,
    CATEGORY_IDS,
    BaseShardWriter,
    base_math_stream,
    base_story_stream,
)
from .engine import token_accuracy
from .examples import ScoredTokenExample, collate_scored, tokenize_examples
from .lock import build_protocol_lock
from .model import (
    CLMBlock,
    MiniCLMConfig,
    StableAddressRouter,
    TinyCLMDecoder,
    sinusoidal_positions,
)
from .performance import (
    PERFORMANCE_FORMAT,
    _baseline_key,
    _copy_selected_baselines,
    _load_cached_baselines,
    _performance_environment,
    _run_baseline_pair,
    _write_harness_evidence,
    evaluate_gate_summaries,
    install_runtime_patches,
    materialize_tokenized_curriculum,
    resolve_cuda_devices,
    run_single_variant,
    train_base_model_parallel,
)
from .protocol import (
    CandidateOptimizerConfig,
    assert_seed_allowed,
    canonical_json_hash,
    file_sha256,
    formal_model_config,
    load_protocol,
    m1_thresholds,
)
from .state import model_state_hash
from .tokenizer import TokenizerBundle, train_tokenizer
from .training import (
    BaseCorpusDataset,
    BaseTrainConfig,
    base_cell_activation_counts,
    exact_match_accuracy,
    mean_scored_nll,
)


V2_BASE_CORPUS_VERSION = "clm-0.4-mini-m1-v2-base-corpus-v1"
V2_CALIBRATION_FORMAT = "minicells.clm-0.4-mini.m1-v2-calibration.v1"
V2_ASSET_LOCK_FORMAT = "minicells.clm-0.4-mini.m1-v2-asset-lock.v1"
V2_ROUTING_SALT = "clm-0.4-mini-m1-v2"
DENSE_BASELINE_FORMAT = "minicells.clm-0.4-mini.m1-v2-dense-baselines.v1"


def v2_math_stream(seed: int = 5101) -> Iterator[str]:
    """Base math is task-aligned QA while staying below continual families."""
    rng = random.Random(int(seed))
    index = 0
    while True:
        mode = index % 4
        if mode == 0:
            a, b = rng.randint(0, 30), rng.randint(0, 30)
            yield f"Question: What is {a} plus {b}? Answer: {a + b}."
        elif mode == 1:
            b = rng.randint(0, 20)
            a = rng.randint(b, 40)
            yield f"Question: What is {a} minus {b}? Answer: {a - b}."
        elif mode == 2:
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            relation = "greater" if a > b else "less" if a < b else "equal"
            yield (
                f"Question: Compare {a} and {b}. What is the relation of the first "
                f"number to the second? Answer: {relation}."
            )
        else:
            a, b = rng.randint(1, 20), rng.randint(1, 20)
            yield (
                f"Question: Nia has {a} stones and receives {b} more. "
                f"How many stones does Nia have now? Answer: {a + b}."
            )
        index += 1


def v2_story_stream(seed: int = 5201) -> Iterator[str]:
    """Context-conditioned retrieval QA matching the v2 base admission task."""
    rng = random.Random(int(seed))
    names = ("Ada", "Bram", "Cleo", "Dion", "Esme", "Finn", "Gia", "Hugo")
    cities = ("Luma", "Sora", "Vela", "Neris", "Orin", "Tera")
    jobs = ("baker", "teacher", "gardener", "painter", "librarian", "cook")
    index = 0
    while True:
        name = names[index % len(names)]
        city = cities[rng.randrange(len(cities))]
        job = jobs[rng.randrange(len(jobs))]
        if index % 2 == 0:
            yield (
                f"Context: {name} lives in {city} and works as a {job}. "
                f"Question: Where does {name} live? Answer: {city}."
            )
        else:
            yield (
                f"Context: {name} lives in {city} and works as a {job}. "
                f"Question: What is {name}'s job? Answer: {job}."
            )
        index += 1


def route_balanced_eval_addresses(
    cfg: MiniCLMConfig,
    *,
    domain: str,
    count: int = 64,
) -> list[str]:
    """Return held-out unique addresses whose routes cover every base Cell."""
    if count < cfg.base_cells:
        raise ValueError("route-balanced evaluation requires at least base_cells examples")
    router = StableAddressRouter(num_cells=cfg.base_cells, salt=cfg.routing_salt)
    uncovered = {
        layer: set(range(cfg.base_cells))
        for layer in (3, 4)
    }
    selected: list[str] = []
    used: set[str] = set()
    candidate = 0

    while any(uncovered[layer] for layer in (3, 4)):
        best: tuple[int, str, dict[int, tuple[int, int]]] | None = None
        for offset in range(4096):
            address = f"v2/eval/{domain}/{candidate + offset:06d}"
            if address in used:
                continue
            routes = {layer: router.route(layer, address) for layer in (3, 4)}
            score = sum(
                int(cell in uncovered[layer])
                for layer in (3, 4)
                for cell in routes[layer]
            )
            choice = (score, address, routes)
            if best is None or choice[0] > best[0] or (
                choice[0] == best[0] and choice[1] < best[1]
            ):
                best = choice
            if score == 4:
                break
        if best is None or best[0] <= 0:
            raise RuntimeError("unable to construct route-balanced evaluation addresses")
        _, address, routes = best
        selected.append(address)
        used.add(address)
        for layer in (3, 4):
            uncovered[layer].difference_update(routes[layer])
        candidate += 1
        if len(selected) > count:
            raise RuntimeError("route coverage required more than registered eval count")

    while len(selected) < int(count):
        address = f"v2/eval/{domain}/{candidate:06d}"
        candidate += 1
        if address not in used:
            selected.append(address)
            used.add(address)

    coverage = {
        layer: [0] * cfg.base_cells
        for layer in (3, 4)
    }
    for address in selected:
        for layer in (3, 4):
            for cell in router.route(layer, address):
                coverage[layer][cell] += 1
    if min(min(values) for values in coverage.values()) < 1:
        raise RuntimeError("v2 route-balanced evaluation failed to cover all base Cells")
    return selected


def v2_math_eval_examples(
    cfg: MiniCLMConfig, count: int = 64, seed: int = 5301
) -> list[TextExample]:
    rng = random.Random(int(seed))
    addresses = route_balanced_eval_addresses(cfg, domain="math", count=count)
    result: list[TextExample] = []
    for index, address in enumerate(addresses):
        mode = index % 4
        if mode == 0:
            a, b = rng.randint(0, 30), rng.randint(0, 30)
            prompt, answer = (
                f"Question: What is {a} plus {b}? Answer:",
                f" {a + b}.",
            )
        elif mode == 1:
            b = rng.randint(0, 20)
            a = rng.randint(b, 40)
            prompt, answer = (
                f"Question: What is {a} minus {b}? Answer:",
                f" {a - b}.",
            )
        elif mode == 2:
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            relation = "greater" if a > b else "less" if a < b else "equal"
            prompt, answer = (
                f"Question: Compare {a} and {b}. What is the relation of the first "
                f"number to the second? Answer:",
                f" {relation}.",
            )
        else:
            a, b = rng.randint(1, 20), rng.randint(1, 20)
            prompt, answer = (
                f"Question: Nia has {a} stones and receives {b} more. "
                "How many stones does Nia have now? Answer:",
                f" {a + b}.",
            )
        result.append(
            TextExample(
                f"v2-base-math-eval:{index:03d}",
                address,
                prompt,
                answer,
            )
        )
    return result


def v2_story_eval_examples(
    cfg: MiniCLMConfig, count: int = 64, seed: int = 5401
) -> list[TextExample]:
    rng = random.Random(int(seed))
    addresses = route_balanced_eval_addresses(cfg, domain="story", count=count)
    names = ("Ada", "Bram", "Cleo", "Dion", "Esme", "Finn", "Gia", "Hugo")
    cities = ("Luma", "Sora", "Vela", "Neris", "Orin", "Tera")
    jobs = ("baker", "teacher", "gardener", "painter", "librarian", "cook")
    result: list[TextExample] = []
    for index, address in enumerate(addresses):
        name = names[index % len(names)]
        city = cities[rng.randrange(len(cities))]
        job = jobs[rng.randrange(len(jobs))]
        context = f"Context: {name} lives in {city} and works as a {job}. "
        if index % 2 == 0:
            prompt, answer = (
                context + f"Question: Where does {name} live? Answer:",
                f" {city}.",
            )
            key = f"v2-base:{index}:location"
        else:
            prompt, answer = (
                context + f"Question: What is {name}'s job? Answer:",
                f" {job}.",
            )
            key = f"v2-base:{index}:job"
        result.append(
            TextExample(
                f"v2-base-story-eval:{index:03d}",
                address,
                prompt,
                answer,
                knowledge_key=key,
            )
        )
    return result


class V2BaseShardWriter(BaseShardWriter):
    """Base shard writer with v2 task-aligned controlled generators."""

    def build(
        self,
        *,
        carrier_texts: Iterable[str],
        carrier_source: dict,
        math_seed: int = 5101,
        story_seed: int = 5201,
    ) -> dict:
        targets = {
            category: int(round(self.target_tokens * fraction))
            for category, fraction in self.mixture.items()
        }
        self._fill_category(
            "language_carrier",
            carrier_texts,
            targets["language_carrier"],
        )
        self._fill_category(
            "controlled_base_math",
            v2_math_stream(math_seed),
            targets["controlled_base_math"],
        )
        self._fill_category(
            "controlled_base_story",
            v2_story_stream(story_seed),
            targets["controlled_base_story"],
        )
        self._flush()
        address_path = self.out_dir / "address-table.json"
        address_path.write_text(
            json.dumps(self.address_pool, indent=2) + "\n",
            encoding="utf-8",
        )
        actual_total = sum(self.category_tokens.values())
        manifest = {
            "format": "minicells.clm-0.4-mini.m1-v2.base-corpus-manifest.v1",
            "generator_version": V2_BASE_CORPUS_VERSION,
            "target_tokens": self.target_tokens,
            "actual_tokens": actual_total,
            "mixture_target": self.mixture,
            "category_tokens": self.category_tokens,
            "category_fractions": {
                key: value / float(max(1, actual_total))
                for key, value in self.category_tokens.items()
            },
            "model_sequence_length": self.cfg.max_seq_len,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "routing_salt": self.cfg.routing_salt,
            "base_address_pool_size": len(self.address_pool),
            "address_table": {
                "path": address_path.name,
                "sha256": file_sha256(address_path),
            },
            "carrier_source": dict(carrier_source),
            "controlled_seeds": {
                "math": int(math_seed),
                "story": int(story_seed),
            },
            "controlled_task_alignment": {
                "math": "question-answer",
                "story": "context-conditioned-retrieval-qa",
            },
            "shards": self._shards,
        }
        manifest["manifest_sha256"] = canonical_json_hash(manifest)
        (self.out_dir / "base-corpus-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest


def prepare_v2_data_assets(
    *,
    protocol_path: str | Path,
    out_dir: str | Path,
    routing_salt: str,
    tokenizer_training_texts: Iterable[str],
    carrier_texts: Iterable[str],
    carrier_source: dict,
) -> dict[str, Any]:
    """Seed-independent v2 tokenizer/base-corpus/curriculum preparation."""
    protocol = load_protocol(protocol_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_manifest = train_tokenizer(
        tokenizer_training_texts,
        out_dir=out_dir / "tokenizer",
        vocab_size=int(protocol["model"]["vocab_size"]),
        source_manifest=carrier_source,
    )
    tokenizer = TokenizerBundle.load(out_dir / "tokenizer" / "tokenizer.json")
    cfg = formal_model_config(protocol, routing_salt=routing_salt)
    writer = V2BaseShardWriter(
        tokenizer=tokenizer,
        model_config=cfg,
        out_dir=out_dir / "base-corpus",
        target_tokens=int(protocol["base_training"]["target_tokens"]),
        mixture=dict(protocol["base_training"]["mixture_token_fraction"]),
    )
    base_manifest = writer.build(
        carrier_texts=carrier_texts,
        carrier_source=carrier_source,
    )
    curriculum = build_curriculum()
    (out_dir / "curriculum-manifest.json").write_text(
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "format": "minicells.clm-0.4-mini.m1-v2.asset-summary.v1",
        "scientific_decision": False,
        "development_seed_observed": False,
        "formal_seeds_observed": False,
        "routing_salt": routing_salt,
        "dataset_revision": str(carrier_source["revision"]),
        "tokenizer_manifest_hash": tokenizer_manifest["manifest_sha256"],
        "tokenizer_hash": tokenizer_manifest["tokenizer_sha256"],
        "base_corpus_manifest_hash": base_manifest["manifest_sha256"],
        "curriculum_manifest_hash": curriculum["manifest_sha256"],
        "base_tokens": base_manifest["actual_tokens"],
        "base_generator_version": V2_BASE_CORPUS_VERSION,
    }
    (out_dir / "asset-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "tokenizer_manifest": tokenizer_manifest,
        "base_corpus_manifest": base_manifest,
        "curriculum_manifest": curriculum,
        "summary": summary,
    }


def teacher_forced_answer_exact_accuracy(
    model: nn.Module,
    examples: Iterable[TextExample],
    *,
    tokenizer: TokenizerBundle,
    max_seq_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Per-example exactness over all teacher-forced answer targets, including EOS."""
    scored = tokenize_examples(
        list(examples), tokenizer, max_seq_len=max_seq_len
    )
    if not scored:
        return 0.0
    correct = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(scored), int(batch_size)):
            batch = scored[start : start + int(batch_size)]
            x, y, mask, addresses = collate_scored(
                batch, pad_id=tokenizer.pad_id, device=device
            )
            logits = model(x, addresses)
            predictions = logits.argmax(dim=-1)
            for row in range(len(batch)):
                selected = mask[row]
                correct += int(
                    bool(
                        torch.equal(
                            predictions[row][selected],
                            y[row][selected],
                        )
                    )
                )
    return correct / float(len(scored))


def base_capability_metrics(
    model: nn.Module,
    *,
    cfg: MiniCLMConfig,
    tokenizer: TokenizerBundle,
    math_examples: list[TextExample],
    story_examples: list[TextExample],
    device: torch.device,
) -> dict[str, Any]:
    math_scored = tokenize_examples(
        math_examples, tokenizer, max_seq_len=cfg.max_seq_len
    )
    story_scored = tokenize_examples(
        story_examples, tokenizer, max_seq_len=cfg.max_seq_len
    )
    return {
        "math_teacher_forced_answer_exact": teacher_forced_answer_exact_accuracy(
            model,
            math_examples,
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            device=device,
        ),
        "story_teacher_forced_answer_exact": teacher_forced_answer_exact_accuracy(
            model,
            story_examples,
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            device=device,
        ),
        "diagnostics": {
            "math_greedy_exact_match": exact_match_accuracy(
                model,
                math_examples,
                tokenizer=tokenizer,
                device=device,
            ),
            "story_greedy_exact_match": exact_match_accuracy(
                model,
                story_examples,
                tokenizer=tokenizer,
                device=device,
            ),
            "math_teacher_forced_token_accuracy": token_accuracy(
                model,
                math_scored,
                tokenizer=tokenizer,
                device=device,
            ),
            "story_teacher_forced_token_accuracy": token_accuracy(
                model,
                story_scored,
                tokenizer=tokenizer,
                device=device,
            ),
            "math_answer_nll": mean_scored_nll(
                model,
                math_scored,
                tokenizer=tokenizer,
                device=device,
            ),
            "story_answer_nll": mean_scored_nll(
                model,
                story_scored,
                tokenizer=tokenizer,
                device=device,
            ),
        },
    }


def evaluate_v2_base_prerequisites(
    *,
    protocol: Mapping[str, Any],
    capability: Mapping[str, Any],
    cell_activation_counts: Mapping[str, int],
    locked_minimum_activation: int,
    numeric_finite: bool,
    hashes_match_lock: bool,
) -> dict[str, Any]:
    registered = protocol["base_prerequisites"]
    gates = {
        "base_math_teacher_forced_answer_exact": {
            "value": float(capability["math_teacher_forced_answer_exact"]),
            "threshold": float(
                registered["minimum_base_math_teacher_forced_answer_exact"]
            ),
            "pass": float(capability["math_teacher_forced_answer_exact"])
            >= float(
                registered["minimum_base_math_teacher_forced_answer_exact"]
            ),
        },
        "base_story_teacher_forced_answer_exact": {
            "value": float(capability["story_teacher_forced_answer_exact"]),
            "threshold": float(
                registered["minimum_base_story_teacher_forced_answer_exact"]
            ),
            "pass": float(capability["story_teacher_forced_answer_exact"])
            >= float(
                registered["minimum_base_story_teacher_forced_answer_exact"]
            ),
        },
        "base_cell_activation": {
            "value": min(cell_activation_counts.values())
            if cell_activation_counts
            else 0,
            "threshold": int(locked_minimum_activation),
            "pass": bool(cell_activation_counts)
            and min(cell_activation_counts.values())
            >= int(locked_minimum_activation),
        },
        "numeric_finite": {
            "value": bool(numeric_finite),
            "threshold": True,
            "pass": bool(numeric_finite),
        },
        "hashes_match_lock": {
            "value": bool(hashes_match_lock),
            "threshold": True,
            "pass": bool(hashes_match_lock),
        },
    }
    return {
        "gates": gates,
        "pass": all(item["pass"] for item in gates.values()),
    }


class DenseDecoder(nn.Module):
    """Dense decoder with the same attention/embedding topology as CLM."""

    def __init__(self, cfg: MiniCLMConfig) -> None:
        super().__init__()
        if cfg.num_layers != 4:
            raise ValueError("M1-v2 dense baselines preserve four blocks")
        self.cfg = cfg
        self.router = StableAddressRouter(num_cells=max(2, cfg.base_cells), salt=cfg.routing_salt)
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                CLMBlock(
                    cfg,
                    layer_id=layer_id,
                    router=self.router,
                    sparse=False,
                )
                for layer_id in range(1, 5)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        address_ids: list[str | int],
    ) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        if token_ids.size(0) != len(address_ids):
            raise ValueError("address_ids must align with batch")
        if token_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds configured maximum")
        x = self.token_embedding(token_ids)
        x = x + sinusoidal_positions(
            token_ids.size(1),
            self.cfg.d_model,
            x.device,
            x.dtype,
        ).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(
                token_ids.size(1),
                token_ids.size(1),
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, address_ids, causal_mask)
        x = self.final_norm(x)
        return F.linear(x, self.token_embedding.weight)


def dense_baseline_config(
    protocol: Mapping[str, Any],
    *,
    kind: str,
    routing_salt: str,
) -> MiniCLMConfig:
    if kind not in {"equal_parameter", "equal_active_compute"}:
        raise ValueError("unknown dense baseline kind")
    hidden = int(protocol["dense_baselines"][kind]["ffn_hidden"])
    model = protocol["model"]
    return MiniCLMConfig(
        vocab_size=int(model["vocab_size"]),
        max_seq_len=int(model["sequence_length"]),
        num_layers=4,
        d_model=int(model["model_dim"]),
        n_heads=int(model["attention_heads"]),
        dense_ff_hidden=hidden,
        base_cells=2,
        cell_hidden=1,
        routing_salt=routing_salt,
    )


def _finite_model(model: nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _load_or_train_model(
    *,
    kind: str,
    model: nn.Module,
    dataset: BaseCorpusDataset,
    tokenizer: TokenizerBundle,
    checkpoint: Path,
    asset_identity: Mapping[str, Any],
    seed: int,
    devices: list[torch.device],
) -> tuple[nn.Module, dict[str, Any]]:
    primary = devices[0]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=primary, weights_only=False)
        if payload.get("kind") != kind:
            raise RuntimeError(f"{kind} checkpoint kind mismatch")
        if int(payload.get("seed", -1)) != int(seed):
            raise RuntimeError(f"{kind} checkpoint seed mismatch")
        if dict(payload.get("asset_identity", {})) != dict(asset_identity):
            raise RuntimeError(f"{kind} checkpoint asset identity mismatch")
        model.load_state_dict(payload["model_state"])
        source = "resumed"
        train = dict(payload["base_train"])
    else:
        train = train_base_model_parallel(
            model,
            dataset=dataset,
            tokenizer=tokenizer,
            devices=devices,
            seed=seed,
        )
        torch.save(
            {
                "format": "minicells.clm-0.4-mini.m1-v2.base-checkpoint.v1",
                "kind": kind,
                "seed": int(seed),
                "model_config": model.cfg.to_dict(),
                "model_state": model.state_dict(),
                "asset_identity": dict(asset_identity),
                "base_train": train,
            },
            checkpoint,
        )
        source = "trained-once"
    return model, {
        "checkpoint": str(checkpoint),
        "checkpoint_source": source,
        "base_train": train,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def verify_v2_asset_lock(
    *,
    protocol: Mapping[str, Any],
    data_dir: Path,
    lock_path: Path,
) -> dict[str, Any]:
    lock = _load(lock_path)
    if lock.get("format") != V2_ASSET_LOCK_FORMAT:
        raise RuntimeError("unexpected v2 asset-lock format")
    if lock.get("lock_status") != "LOCKED":
        raise RuntimeError(
            "M1-v2 data identity is not locked. Run seed-independent data preparation, "
            "commit the resulting hashes, then rerun calibration."
        )
    if bool(lock.get("development_seed_observed_when_locked")):
        raise RuntimeError("v2 asset lock must predate development seed 90402")
    summary = _load(data_dir / "asset-summary.json")
    tokenizer_manifest = _load(data_dir / "tokenizer" / "tokenizer-manifest.json")
    base_manifest = _load(data_dir / "base-corpus" / "base-corpus-manifest.json")
    curriculum_manifest = _load(data_dir / "curriculum-manifest.json")
    actual = {
        "dataset_revision": str(summary["dataset_revision"]),
        "routing_salt": str(summary["routing_salt"]),
        "base_tokens": int(summary["base_tokens"]),
        "tokenizer_hash": str(summary["tokenizer_hash"]),
        "tokenizer_manifest_hash": str(summary["tokenizer_manifest_hash"]),
        "base_corpus_manifest_hash": str(summary["base_corpus_manifest_hash"]),
        "curriculum_manifest_hash": str(summary["curriculum_manifest_hash"]),
        "base_generator_version": str(summary["base_generator_version"]),
    }
    expected = dict(lock["identity"])
    expected["base_tokens"] = int(expected["base_tokens"])
    if actual != expected:
        raise RuntimeError(
            "M1-v2 data assets differ from the committed pre-90402 lock:\n"
            + json.dumps({"expected": expected, "actual": actual}, indent=2, sort_keys=True)
        )
    target = int(protocol["base_training"]["target_tokens"])
    tolerance = float(protocol["base_training"]["token_tolerance_fraction"])
    if abs(actual["base_tokens"] - target) > target * tolerance:
        raise RuntimeError("v2 base token count outside protocol tolerance")
    if int(curriculum_manifest["counts"]["total"]) != 192:
        raise RuntimeError("v2 curriculum transaction-count drift")
    return {
        "verified": True,
        "identity": actual,
        "identity_sha256": canonical_json_hash(actual),
        "tokenizer_manifest": tokenizer_manifest,
        "base_corpus_manifest": base_manifest,
        "curriculum_manifest": curriculum_manifest,
    }


def prepare_or_load_v2_bases(
    *,
    protocol: Mapping[str, Any],
    data_dir: Path,
    out_dir: Path,
    assets: Mapping[str, Any],
    seed: int,
    devices: list[torch.device],
) -> tuple[TinyCLMDecoder, TokenizerBundle, dict[str, Any], dict[str, DenseDecoder]]:
    primary = devices[0]
    tokenizer = TokenizerBundle.load(data_dir / "tokenizer" / "tokenizer.json")
    dataset = BaseCorpusDataset(data_dir / "base-corpus")
    clm_cfg = formal_model_config(
        protocol, routing_salt=assets["identity"]["routing_salt"]
    )
    math_eval = v2_math_eval_examples(clm_cfg, 64)
    story_eval = v2_story_eval_examples(clm_cfg, 64)

    torch.manual_seed(seed)
    random.seed(seed)
    if primary.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    clm = TinyCLMDecoder(clm_cfg).to(primary)
    clm, clm_meta = _load_or_train_model(
        kind="clm",
        model=clm,
        dataset=dataset,
        tokenizer=tokenizer,
        checkpoint=out_dir / "base" / "clm" / "checkpoint.pt",
        asset_identity=assets["identity"],
        seed=seed,
        devices=devices,
    )
    counts = base_cell_activation_counts(clm, dataset)
    activation_threshold = minimum_base_cell_activation(
        protocol, base_sequences=len(dataset)
    )
    capability = base_capability_metrics(
        clm,
        cfg=clm_cfg,
        tokenizer=tokenizer,
        math_examples=math_eval,
        story_examples=story_eval,
        device=primary,
    )
    prerequisites = evaluate_v2_base_prerequisites(
        protocol=protocol,
        capability=capability,
        cell_activation_counts=counts,
        locked_minimum_activation=activation_threshold,
        numeric_finite=_finite_model(clm),
        hashes_match_lock=bool(assets["verified"]),
    )
    clm_meta.update(
        {
            "state_hash": model_state_hash(clm),
            "base_sequences": len(dataset),
            "minimum_base_cell_activation": activation_threshold,
            "capability": capability,
            "prerequisites": prerequisites,
            "route_balanced_eval": {
                "math_unique_addresses": len({item.address_id for item in math_eval}),
                "story_unique_addresses": len({item.address_id for item in story_eval}),
                "all_base_cells_covered": True,
            },
        }
    )
    _write(out_dir / "base" / "clm" / "metrics.json", clm_meta)
    with (out_dir / "base" / "clm" / "activation-counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "activation_count"])
        writer.writerows(
            (cell_id, count) for cell_id, count in sorted(counts.items())
        )

    dense_models: dict[str, DenseDecoder] = {}
    dense_static: dict[str, Any] = {}
    for kind in ("equal_parameter", "equal_active_compute"):
        torch.manual_seed(seed)
        if primary.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        cfg = dense_baseline_config(
            protocol,
            kind=kind,
            routing_salt=f"{V2_ROUTING_SALT}/dense/{kind}",
        )
        dense = DenseDecoder(cfg).to(primary)
        dense, meta = _load_or_train_model(
            kind=f"dense_{kind}",
            model=dense,
            dataset=dataset,
            tokenizer=tokenizer,
            checkpoint=out_dir / "base" / f"dense_{kind}" / "checkpoint.pt",
            asset_identity=assets["identity"],
            seed=seed,
            devices=devices,
        )
        meta["capability"] = base_capability_metrics(
            dense,
            cfg=cfg,
            tokenizer=tokenizer,
            math_examples=math_eval,
            story_examples=story_eval,
            device=primary,
        )
        meta["numeric_finite"] = _finite_model(dense)
        _write(out_dir / "base" / f"dense_{kind}" / "metrics.json", meta)
        dense_models[kind] = dense
        dense_static[kind] = meta

    expected_equal_parameter = int(
        protocol["dense_baselines"]["equal_parameter"]["expected_parameter_count"]
    )
    expected_equal_compute = int(
        protocol["dense_baselines"]["equal_active_compute"]["expected_parameter_count"]
    )
    if dense_static["equal_parameter"]["parameter_count"] != expected_equal_parameter:
        raise RuntimeError("equal-parameter dense parameter-count drift")
    if dense_static["equal_active_compute"]["parameter_count"] != expected_equal_compute:
        raise RuntimeError("equal-compute dense parameter-count drift")

    summary = {
        "format": DENSE_BASELINE_FORMAT,
        "clm": clm_meta,
        "dense": dense_static,
        "comparison_authority": "diagnostic-only",
    }
    _write(out_dir / "base" / "base-comparison.json", summary)
    return clm, tokenizer, summary, dense_models


def v2_base_probes(
    *,
    cfg: MiniCLMConfig,
    tokenizer: TokenizerBundle,
) -> list[ScoredTokenExample]:
    texts = [
        *v2_math_eval_examples(cfg, 64),
        *v2_story_eval_examples(cfg, 64),
    ]
    return tokenize_examples(texts, tokenizer, max_seq_len=cfg.max_seq_len)


def _masked_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    selected = per_token[mask]
    if selected.numel() == 0:
        raise RuntimeError("dense continual batch has no scored tokens")
    return selected.mean()


def train_full_model_amp(
    model: DenseDecoder,
    *,
    examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    optimizer_config: CandidateOptimizerConfig,
    device: torch.device,
    rng_seed: int,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )
    rng = random.Random(int(rng_seed))
    amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    started = time.perf_counter()
    training_tokens = 0
    model.train()
    for _ in range(int(optimizer_config.steps)):
        if len(examples) <= optimizer_config.batch_size:
            batch = list(examples)
        else:
            indices = rng.sample(
                range(len(examples)), int(optimizer_config.batch_size)
            )
            batch = [examples[index] for index in indices]
        x, y, mask, addresses = collate_scored(
            batch, pad_id=tokenizer.pad_id, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if amp else torch.float32,
            enabled=amp,
        ):
            loss = _masked_ce(model(x, addresses), y, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_tokens += int(mask.sum().item())
    return {
        "wall_seconds": time.perf_counter() - started,
        "training_tokens": training_tokens,
        "optimizer_steps": int(optimizer_config.steps),
        "training_precision": "fp16-amp" if amp else "fp32",
    }


class DenseContinualHarness:
    """Dense full-finetune baseline with always/global-transactional commit."""

    def __init__(
        self,
        *,
        variant: str,
        model: DenseDecoder,
        tokenizer: TokenizerBundle,
        device: torch.device,
        thresholds: Mapping[str, float],
    ) -> None:
        if variant not in {"dense_full_always", "dense_global_tx"}:
            raise ValueError("unknown dense continual variant")
        self.variant = variant
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.thresholds = dict(thresholds)
        self.probes: dict[str, ScoredTokenExample] = {}
        self.reference_accuracy: dict[str, float] = {}
        self.records: list[dict[str, Any]] = []

    def _protected(self) -> list[ScoredTokenExample]:
        return [self.probes[key] for key in sorted(self.probes)]

    def _accuracy(self, examples: list[ScoredTokenExample]) -> float:
        return token_accuracy(
            self.model,
            examples,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def admit(self, examples: Iterable[ScoredTokenExample]) -> None:
        for example in examples:
            self.probes[example.example_id] = example
            self.reference_accuracy[example.example_id] = self._accuracy([example])

    def remove_key(self, knowledge_key: str | None) -> None:
        if not knowledge_key:
            return
        removed = [
            key
            for key, example in self.probes.items()
            if example.knowledge_key == knowledge_key
        ]
        for key in removed:
            self.probes.pop(key, None)
            self.reference_accuracy.pop(key, None)

    def execute(
        self,
        *,
        transaction_id: int,
        operation: str,
        supersedes_key: str | None,
        train_examples: list[ScoredTokenExample],
        validation_examples: list[ScoredTokenExample],
        probe_examples: list[ScoredTokenExample],
        optimizer_config: CandidateOptimizerConfig,
        rng_seed: int,
    ) -> None:
        if operation == "supersede":
            self.remove_key(supersedes_key)
        before_model = self.model
        candidate = copy.deepcopy(self.model).to(self.device)
        old = self._protected()
        new_before = mean_scored_nll(
            before_model,
            validation_examples,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        old_before = mean_scored_nll(
            before_model,
            old,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        train_stats = train_full_model_amp(
            candidate,
            examples=train_examples,
            tokenizer=self.tokenizer,
            optimizer_config=optimizer_config,
            device=self.device,
            rng_seed=rng_seed,
        )
        new_after = mean_scored_nll(
            candidate,
            validation_examples,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        old_after = mean_scored_nll(
            candidate,
            old,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        gain = (new_before - new_after) / max(new_before, 1e-8)
        regression = (
            (old_after - old_before) / max(old_before, 1e-8)
            if old
            else 0.0
        )
        local_pass = (
            gain >= self.thresholds["minimum_new_gain"]
            and regression <= self.thresholds["maximum_global_old_regression"]
        )
        commit = self.variant == "dense_full_always" or local_pass
        if commit:
            self.model = candidate
            self.admit(probe_examples)
        self.records.append(
            {
                "transaction_id": int(transaction_id),
                "new_gain": float(gain),
                "global_regression": float(regression),
                "commit": bool(commit),
                "training_tokens": train_stats["training_tokens"],
                "optimizer_steps": train_stats["optimizer_steps"],
                "candidate_wall_seconds": train_stats["wall_seconds"],
            }
        )

    def summary(self) -> dict[str, Any]:
        commits = [row for row in self.records if row["commit"]]
        admission = sum(self.reference_accuracy.values()) / float(
            max(1, len(self.reference_accuracy))
        )
        final = self._accuracy(self._protected())
        return {
            "variant": self.variant,
            "transactions": len(self.records),
            "effective_commits": len(commits),
            "effective_acceptance_rate": len(commits)
            / float(max(1, len(self.records))),
            "committed_new_gain": sum(row["new_gain"] for row in commits),
            "positive_global_regression_damage": sum(
                max(0.0, row["global_regression"]) for row in commits
            ),
            "protected_probe_count": len(self.probes),
            "final_protected_retention_ratio": final / max(admission, 1e-8),
            "parameter_count": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
        }


def run_dense_continual_baselines(
    *,
    protocol: Mapping[str, Any],
    base_model: DenseDecoder,
    tokenizer: TokenizerBundle,
    curriculum_manifest: Mapping[str, Any],
    tokenized_transactions: Mapping[
        int, Mapping[str, list[ScoredTokenExample]]
    ],
    base_probes: list[ScoredTokenExample],
    optimizer_config: CandidateOptimizerConfig,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for variant_index, variant in enumerate(
        ("dense_full_always", "dense_global_tx")
    ):
        harness = DenseContinualHarness(
            variant=variant,
            model=copy.deepcopy(base_model).to(device),
            tokenizer=tokenizer,
            device=device,
            thresholds=m1_thresholds(protocol),
        )
        harness.admit(base_probes)
        for spec in transaction_specs(dict(curriculum_manifest)):
            data = tokenized_transactions[int(spec.transaction_id)]
            tx_seed = (
                int(seed) * 1_000_003
                + int(spec.transaction_id) * 97
                + (50 + variant_index) * 13
            ) & 0x7FFFFFFF
            harness.execute(
                transaction_id=spec.transaction_id,
                operation=spec.operation,
                supersedes_key=spec.supersedes_key,
                train_examples=list(data["train"]),
                validation_examples=list(data["validation"]),
                probe_examples=list(data["probe"]),
                optimizer_config=optimizer_config,
                rng_seed=tx_seed,
            )
        summary = harness.summary()
        root = out_dir / variant
        root.mkdir(parents=True, exist_ok=True)
        _write(root / "summary.json", summary)
        with (root / "transactions.jsonl").open("w", encoding="utf-8") as handle:
            for record in harness.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        summaries[variant] = summary
        del harness
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "format": DENSE_BASELINE_FORMAT,
        "diagnostic_only": True,
        "optimizer_source": "selected-clm-direct-candidate",
        "variants": summaries,
    }
    _write(out_dir / "summary.json", payload)
    return payload


def run_v2_calibration(
    *,
    protocol_path: str | Path,
    asset_lock_path: str | Path,
    committed_plan_path: str | Path,
    protocol_lock_template_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    seed: int,
    device: str | torch.device,
    devices: str | None,
    code_commit: str,
    code_tree: str,
    tracked_tree_dirty: bool,
) -> dict[str, Any]:
    install_runtime_patches()
    protocol = load_protocol(protocol_path)
    assert_seed_allowed(protocol, mode="calibration", seed=seed)
    if tracked_tree_dirty or not code_commit or not code_tree:
        raise RuntimeError("v2 calibration requires a clean committed source tree")

    out_dir = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = verify_committed_plan(protocol, committed_plan_path)
    _write(out_dir / "calibration-plan.json", plan)
    assets = verify_v2_asset_lock(
        protocol=protocol,
        data_dir=data_dir,
        lock_path=Path(asset_lock_path),
    )
    _write(
        out_dir / "asset-verification.json",
        {
            "verified": True,
            "identity": assets["identity"],
            "identity_sha256": assets["identity_sha256"],
        },
    )
    resolved = resolve_cuda_devices(
        requested_device=device,
        requested_devices=devices,
    )
    primary = resolved[0]
    torch.manual_seed(seed)
    random.seed(seed)
    if primary.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    clm, tokenizer, base_comparison, dense_models = prepare_or_load_v2_bases(
        protocol=protocol,
        data_dir=data_dir,
        out_dir=out_dir,
        assets=assets,
        seed=seed,
        devices=resolved,
    )
    environment = _performance_environment(resolved)
    common = {
        "format": V2_CALIBRATION_FORMAT,
        "performance_format": PERFORMANCE_FORMAT,
        "seed": int(seed),
        "scientific_decision": False,
        "development_seed_observed": True,
        "formal_seeds_observed": False,
        "asset_identity": assets["identity"],
        "plan_sha256": plan["plan_sha256"],
        "base_comparison": base_comparison,
        "code_commit": code_commit,
        "code_tree": code_tree,
        "environment": environment,
    }
    if not base_comparison["clm"]["prerequisites"]["pass"]:
        decision = {
            "format": V2_CALIBRATION_FORMAT,
            "status": "V2_CALIBRATION_BASE_PREREQUISITES_FAILED",
            "scientific_decision": False,
            "development_seed_observed": True,
            "formal_seeds_observed": False,
            "selected_candidate": None,
            "candidate_search_started": False,
        }
        _write(out_dir / "decision.json", decision)
        _write(out_dir / "summary.json", {**common, "decision": decision})
        return decision

    tokenized_transactions = materialize_tokenized_curriculum(
        curriculum_manifest=assets["curriculum_manifest"],
        tokenizer=tokenizer,
        max_seq_len=clm.cfg.max_seq_len,
    )
    base_probes = v2_base_probes(cfg=clm.cfg, tokenizer=tokenizer)
    clm = clm.cpu()
    base_snapshot = copy.deepcopy(clm).cpu()
    immutable_hash = model_state_hash(clm)
    rows: list[dict[str, Any]] = []
    selected: CalibrationCandidate | None = None
    selected_gates: dict[str, Any] | None = None

    for payload in plan["candidates"]:
        candidate = _candidate(payload)
        result_path = (
            out_dir / "candidates" / candidate.candidate_id / "candidate.json"
        )
        if result_path.is_file():
            row = _load(result_path)
            if row["candidate"] != candidate.to_dict():
                raise RuntimeError("v2 resume candidate drift")
            rows.append(row)
            if row["pass"]:
                selected = candidate
                selected_gates = row["gate_snapshot"]
                break
            continue
        if model_state_hash(clm) != immutable_hash:
            raise RuntimeError("v2 immutable base model changed before candidate")

        started = time.perf_counter()
        cache_root = out_dir / "baseline-cache" / _baseline_key(candidate.direct)
        baseline_summaries = _load_cached_baselines(
            cache_root, direct=candidate.direct
        )
        growth_device = resolved[1] if len(resolved) > 1 else primary
        cache_hit = baseline_summaries is not None

        if baseline_summaries is None and len(resolved) > 1:
            with ThreadPoolExecutor(max_workers=2) as pool:
                baseline_future = pool.submit(
                    _run_baseline_pair,
                    cache_root=cache_root,
                    protocol=protocol,
                    base_model=base_snapshot,
                    tokenizer=tokenizer,
                    curriculum_manifest=assets["curriculum_manifest"],
                    tokenized_transactions=tokenized_transactions,
                    base_probes=base_probes,
                    direct=candidate.direct,
                    growth=candidate.growth_private,
                    seed=seed,
                    device=primary,
                )
                growth_future = pool.submit(
                    run_single_variant,
                    variant="local_tx_growth",
                    protocol=protocol,
                    base_model=base_snapshot,
                    tokenizer=tokenizer,
                    curriculum_manifest=assets["curriculum_manifest"],
                    tokenized_transactions=tokenized_transactions,
                    base_probes=base_probes,
                    direct_optimizer=candidate.direct,
                    growth_optimizer=candidate.growth_private,
                    seed=seed,
                    device=growth_device,
                )
                baseline_summaries = baseline_future.result()
                growth_harness = growth_future.result()
        else:
            if baseline_summaries is None:
                baseline_summaries = _run_baseline_pair(
                    cache_root=cache_root,
                    protocol=protocol,
                    base_model=base_snapshot,
                    tokenizer=tokenizer,
                    curriculum_manifest=assets["curriculum_manifest"],
                    tokenized_transactions=tokenized_transactions,
                    base_probes=base_probes,
                    direct=candidate.direct,
                    growth=candidate.growth_private,
                    seed=seed,
                    device=primary,
                )
            growth_harness = run_single_variant(
                variant="local_tx_growth",
                protocol=protocol,
                base_model=base_snapshot,
                tokenizer=tokenizer,
                curriculum_manifest=assets["curriculum_manifest"],
                tokenized_transactions=tokenized_transactions,
                base_probes=base_probes,
                direct_optimizer=candidate.direct,
                growth_optimizer=candidate.growth_private,
                seed=seed,
                device=growth_device,
            )

        growth_summary = growth_harness.summary()
        summaries = {
            **baseline_summaries,
            "local_tx_growth": growth_summary,
        }
        gate_snapshot = evaluate_gate_summaries(
            protocol=protocol,
            summaries=summaries,
            growth_harness=growth_harness,
        )
        row = {
            "candidate": candidate.to_dict(),
            "pass": bool(gate_snapshot["pass"]),
            "gate_snapshot": gate_snapshot,
            "base_state_hash_before_and_after": immutable_hash,
            "direct_baseline_cache_hit": cache_hit,
            "performance_format": PERFORMANCE_FORMAT,
            "wall_seconds": time.perf_counter() - started,
        }
        _write(result_path, row)
        rows.append(row)
        if row["pass"]:
            selected = candidate
            selected_gates = gate_snapshot
            selected_root = out_dir / "selected"
            _copy_selected_baselines(cache_root, selected_root)
            _write_harness_evidence(
                selected_root / "local_tx_growth",
                growth_harness,
                summary=growth_summary,
            )
            break
        del growth_harness
        gc.collect()
        if primary.type == "cuda":
            torch.cuda.empty_cache()

    _summary_csv(out_dir, rows)
    if selected is None:
        decision = {
            "format": V2_CALIBRATION_FORMAT,
            "status": "V2_CALIBRATION_NO_CONFIGURATION_PASSED",
            "scientific_decision": False,
            "development_seed_observed": True,
            "formal_seeds_observed": False,
            "selected_candidate": None,
            "candidates_evaluated": len(rows),
        }
        _write(out_dir / "decision.json", decision)
        _write(out_dir / "summary.json", {**common, "decision": decision})
        return decision

    dense_equal = dense_models["equal_parameter"]
    dense_continual = run_dense_continual_baselines(
        protocol=protocol,
        base_model=dense_equal,
        tokenizer=tokenizer,
        curriculum_manifest=assets["curriculum_manifest"],
        tokenized_transactions=tokenized_transactions,
        base_probes=base_probes,
        optimizer_config=selected.direct,
        seed=seed,
        device=primary,
        out_dir=out_dir / "dense-continual",
    )
    selected_payload = {
        "candidate": selected.to_dict(),
        "selection_rule": plan["selection_rule"],
        "first_passing_ordinal": selected.ordinal,
        "gate_snapshot": selected_gates,
        "candidates_evaluated": len(rows),
    }
    _write(out_dir / "selected.json", selected_payload)

    template = _load(protocol_lock_template_path)
    protocol_lock = build_protocol_lock(
        protocol=protocol,
        template=template,
        protocol_path=protocol_path,
        direct_optimizer=selected.direct,
        growth_optimizer=selected.growth_private,
        tokenizer_manifest=assets["tokenizer_manifest"],
        base_corpus_manifest=assets["base_corpus_manifest"],
        curriculum_manifest=assets["curriculum_manifest"],
        dataset_revision=assets["identity"]["dataset_revision"],
        routing_salt=assets["identity"]["routing_salt"],
        minimum_base_cell_activation=base_comparison["clm"][
            "minimum_base_cell_activation"
        ],
        code_commit=code_commit,
        code_tree=code_tree,
        environment={
            **environment,
            "v2_base_primary_gate": "teacher-forced-answer-exact",
            "v2_dense_baselines": {
                "equal_parameter": int(
                    base_comparison["dense"]["equal_parameter"][
                        "parameter_count"
                    ]
                ),
                "equal_active_compute": int(
                    base_comparison["dense"]["equal_active_compute"][
                        "parameter_count"
                    ]
                ),
            },
        },
    )
    _write(out_dir / "protocol-lock.candidate.json", protocol_lock)
    decision = {
        "format": V2_CALIBRATION_FORMAT,
        "status": "V2_CALIBRATION_CONFIGURATION_SELECTED",
        "scientific_decision": False,
        "development_seed_observed": True,
        "formal_seeds_observed": False,
        "selected_candidate": selected.candidate_id,
        "candidates_evaluated": len(rows),
        "formal_execution_authorized": False,
        "dense_baselines_diagnostic_only": True,
        "next_required_action": (
            "commit protocol-lock.candidate.json as the canonical v2 protocol-lock.json "
            "before any formal seed is opened"
        ),
    }
    _write(out_dir / "decision.json", decision)
    _write(
        out_dir / "summary.json",
        {
            **common,
            "decision": decision,
            "selected": selected_payload,
            "dense_continual": dense_continual,
            "protocol_lock_candidate": "protocol-lock.candidate.json",
        },
    )
    return decision
