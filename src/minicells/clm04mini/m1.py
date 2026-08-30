"""M1 orchestration for CLM-0.4-mini.

This module can prepare formal-scale assets and execute the registered M1 stream,
but the infrastructure smoke is hard-isolated from development/formal seeds.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import platform
import random
from typing import Iterable

import torch

from .curriculum import (
    build_curriculum,
    materialize_transaction,
    transaction_specs,
)
from .data import (
    BaseCorpusDataset,
    BaseShardWriter,
    base_math_eval_examples,
    base_math_stream,
    base_story_eval_examples,
    base_story_stream,
    smoke_carrier_texts,
)
from .engine import VARIANTS, VariantHarness, logical_state_hash
from .examples import ScoredTokenExample, tokenize_examples
from .gates import evaluate_m1_gates
from .model import TinyCLMDecoder
from .protocol import (
    M1_INFRA_SEED,
    CandidateOptimizerConfig,
    assert_seed_allowed,
    candidate_grid,
    formal_model_config,
    load_protocol,
    m1_thresholds,
    smoke_model_config,
)
from .tokenizer import TokenizerBundle, train_tokenizer
from .training import BaseTrainConfig, train_base_model


SMOKE_DIRECT = CandidateOptimizerConfig("AdamW", 8, 0.003, 2, 0.0)
SMOKE_GROWTH = CandidateOptimizerConfig("AdamW", 8, 0.003, 4, 0.0)


def environment_versions(device: torch.device) -> dict:
    try:
        import tokenizers

        tokenizers_version = tokenizers.__version__
    except Exception:
        tokenizers_version = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tokenizers": tokenizers_version,
    }


def _tokenizer_training_texts() -> list[str]:
    texts = list(smoke_carrier_texts())
    texts.extend(itertools.islice(base_math_stream(), 128))
    texts.extend(itertools.islice(base_story_stream(), 128))
    manifest = build_curriculum()
    for spec in transaction_specs(manifest)[:16]:
        sample = materialize_transaction(spec, smoke=True)
        texts.extend(f"{item.prompt}{item.answer}" for item in sample["train"][:4])
    return texts


def _tokenize_transaction(
    spec,
    *,
    tokenizer: TokenizerBundle,
    max_seq_len: int,
    smoke: bool,
) -> dict[str, list[ScoredTokenExample]]:
    text = materialize_transaction(spec, smoke=smoke)
    return {
        split: tokenize_examples(items, tokenizer, max_seq_len=max_seq_len)
        for split, items in text.items()
    }


def _base_probes(
    *, tokenizer: TokenizerBundle, max_seq_len: int, smoke: bool
) -> list[ScoredTokenExample]:
    count = 8 if smoke else 64
    texts = [*base_math_eval_examples(count), *base_story_eval_examples(count)]
    return tokenize_examples(texts, tokenizer, max_seq_len=max_seq_len)


def run_m1_stream(
    *,
    protocol: dict,
    base_model: TinyCLMDecoder,
    tokenizer: TokenizerBundle,
    curriculum_manifest: dict,
    direct_optimizer: CandidateOptimizerConfig,
    growth_optimizer: CandidateOptimizerConfig,
    seed: int,
    device: torch.device,
    out_dir: str | Path | None = None,
    smoke_projection: bool = False,
) -> tuple[dict[str, VariantHarness], dict]:
    thresholds = m1_thresholds(protocol)
    harnesses = {
        variant: VariantHarness(
            variant=variant,
            model=copy.deepcopy(base_model).to(device),
            tokenizer=tokenizer,
            device=device,
            thresholds=thresholds,
        )
        for variant in VARIANTS
    }
    initial_hashes = {
        variant: logical_state_hash(h.model, h.dependency_index, h.probes)
        for variant, h in harnesses.items()
    }
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("primary variants did not start from identical base state")
    probes = _base_probes(
        tokenizer=tokenizer, max_seq_len=base_model.cfg.max_seq_len, smoke=smoke_projection
    )
    for harness in harnesses.values():
        harness.admit_probes(probes)

    specs = transaction_specs(curriculum_manifest)
    if smoke_projection:
        selectors = [
            ("math/multiplication/0", 0),
            ("story/world-00", 0),
            ("math/multiplication/0", 1),
            ("story/world-00", 1),
            ("story/world-00", 2),
        ]
        selected = []
        for address_id, visit in selectors:
            selected.append(
                next(
                    spec
                    for spec in specs
                    if spec.address_id == address_id and spec.visit_index == visit
                )
            )
        specs = selected

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        data = _tokenize_transaction(
            spec,
            tokenizer=tokenizer,
            max_seq_len=base_model.cfg.max_seq_len,
            smoke=smoke_projection,
        )
        for variant_index, (variant, harness) in enumerate(harnesses.items()):
            transaction_seed = (
                int(seed) * 1_000_003 + int(spec.transaction_id) * 97 + variant_index * 13
            ) & 0x7FFFFFFF
            harness.execute(
                transaction_id=spec.transaction_id,
                operation=spec.operation,
                address_id=spec.address_id,
                knowledge_key=spec.knowledge_key,
                supersedes_key=spec.supersedes_key,
                train_examples=data["train"],
                validation_examples=data["validation"],
                probe_examples=data["probe"],
                direct_optimizer=direct_optimizer,
                growth_optimizer=growth_optimizer,
                rng_seed=transaction_seed,
            )
            if out_path is not None and (
                (len(harness.records) % 16 == 0) or spec is specs[-1]
            ):
                harness.save_checkpoint(
                    out_path / variant / "checkpoints" / f"tx-{spec.transaction_id:03d}.pt"
                )

    gate_snapshot = evaluate_m1_gates(protocol=protocol, harnesses=harnesses)
    if out_path is not None:
        for variant, harness in harnesses.items():
            variant_dir = out_path / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            with (variant_dir / "transactions.jsonl").open("w", encoding="utf-8") as handle:
                for record in harness.records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            with (variant_dir / "cell-registry.jsonl").open("w", encoding="utf-8") as handle:
                for entry in harness.registry.snapshot(harness.model, harness.dependency_index):
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
            (variant_dir / "summary.json").write_text(
                json.dumps(harness.summary(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return harnesses, gate_snapshot


def prepare_formal_data_assets(
    *,
    protocol_path: str | Path,
    out_dir: str | Path,
    routing_salt: str,
    tokenizer_training_texts: Iterable[str],
    carrier_texts: Iterable[str],
    carrier_source: dict,
) -> dict:
    """Build tokenizer, 30M-token base shards, and frozen curriculum manifest.

    This function does not train a model and does not inspect development/formal
    seeds. The caller is responsible for supplying a pinned carrier revision.
    """
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
    if tokenizer.vocab_size > cfg.vocab_size:
        raise RuntimeError("tokenizer vocabulary exceeds frozen model vocabulary")
    writer = BaseShardWriter(
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
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "tokenizer_manifest": tokenizer_manifest,
        "base_corpus_manifest": base_manifest,
        "curriculum_manifest": curriculum,
    }


def run_m1_infrastructure_smoke(
    *,
    protocol_path: str | Path,
    out_dir: str | Path,
    device: str | torch.device = "cpu",
    seed: int = M1_INFRA_SEED,
) -> dict:
    protocol = load_protocol(protocol_path)
    assert_seed_allowed(protocol, mode="infrastructure-smoke", seed=seed)
    device_obj = torch.device(device)
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = smoke_model_config(protocol)
    tokenizer_manifest = train_tokenizer(
        _tokenizer_training_texts(),
        out_dir=out_dir / "tokenizer",
        vocab_size=cfg.vocab_size,
        min_frequency=1,
        source_manifest={"kind": "built-in-infrastructure-smoke"},
    )
    tokenizer = TokenizerBundle.load(out_dir / "tokenizer" / "tokenizer.json")
    if tokenizer.vocab_size > cfg.vocab_size:
        raise RuntimeError("smoke tokenizer exceeds smoke model vocabulary")

    writer = BaseShardWriter(
        tokenizer=tokenizer,
        model_config=cfg,
        out_dir=out_dir / "base-corpus",
        target_tokens=6000,
        mixture=dict(protocol["base_training"]["mixture_token_fraction"]),
        shard_sequences=128,
    )
    base_manifest = writer.build(
        carrier_texts=itertools.cycle(smoke_carrier_texts()),
        carrier_source={"kind": "built-in-infrastructure-smoke"},
    )
    curriculum = build_curriculum()
    (out_dir / "curriculum-manifest.json").write_text(
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    model = TinyCLMDecoder(cfg).to(device_obj)
    dataset = BaseCorpusDataset(out_dir / "base-corpus")
    base_train = train_base_model(
        model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device_obj,
        seed=seed,
        config=BaseTrainConfig(batch_size=16, learning_rate=1e-3, weight_decay=0.0),
    )
    full_cfg = formal_model_config(protocol, routing_salt="clm-0.4-mini-v1")
    formal_model = TinyCLMDecoder(full_cfg)
    formal_parameter_count = sum(parameter.numel() for parameter in formal_model.parameters())
    del formal_model

    harnesses, diagnostic = run_m1_stream(
        protocol=protocol,
        base_model=model,
        tokenizer=tokenizer,
        curriculum_manifest=curriculum,
        direct_optimizer=SMOKE_DIRECT,
        growth_optimizer=SMOKE_GROWTH,
        seed=seed,
        device=device_obj,
        out_dir=out_dir / "variants",
        smoke_projection=True,
    )
    checkpoint_replay = {}
    thresholds = m1_thresholds(protocol)
    for variant, harness in harnesses.items():
        final_record = harness.records[-1]
        checkpoint = (
            out_dir
            / "variants"
            / variant
            / "checkpoints"
            / f"tx-{final_record['transaction_id']:03d}.pt"
        )
        restored = VariantHarness.load_checkpoint(
            checkpoint,
            tokenizer=tokenizer,
            device=device_obj,
            thresholds=thresholds,
        )
        checkpoint_replay[variant] = {
            "expected": harness.summary()["final_state_hash"],
            "restored": restored.summary()["final_state_hash"],
            "match": harness.summary()["final_state_hash"]
            == restored.summary()["final_state_hash"],
        }
    if not all(item["match"] for item in checkpoint_replay.values()):
        raise RuntimeError("M1 infrastructure checkpoint replay mismatch")

    payload = {
        "format": "minicells.clm-0.4-mini.m1-infrastructure-smoke.v1",
        "mode": "infrastructure-smoke",
        "seed": int(seed),
        "scientific_decision": False,
        "status": "SMOKE_ONLY",
        "formal_seeds_observed": False,
        "development_seed_observed": False,
        "smoke_model_config": cfg.to_dict(),
        "formal_model_parameter_count": int(formal_parameter_count),
        "tokenizer_manifest": tokenizer_manifest,
        "base_corpus_manifest": base_manifest,
        "curriculum_manifest_sha256": curriculum["manifest_sha256"],
        "registered_calibration_grid": {
            "direct": [item.to_dict() for item in candidate_grid(protocol, "direct")],
            "growth": [item.to_dict() for item in candidate_grid(protocol, "growth")],
        },
        "smoke_candidate_config": {
            "direct": SMOKE_DIRECT.to_dict(),
            "growth": SMOKE_GROWTH.to_dict(),
            "registered_calibration_selection": False,
        },
        "base_train": base_train,
        "transaction_projection_ids": [
            record["transaction_id"] for record in harnesses["local_always"].records
        ],
        "diagnostic_gate_snapshot": diagnostic,
        "checkpoint_replay": checkpoint_replay,
        "environment": environment_versions(device_obj),
    }
    (out_dir / "decision.json").write_text(
        json.dumps(
            {
                "status": "SMOKE_ONLY",
                "scientific_decision": False,
                "reason": (
                    "M1 infrastructure smoke validates data/model/variant/checkpoint plumbing only; "
                    "development and formal seeds remain unopened."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
