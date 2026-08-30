"""Pre-formal M1 calibration harness for CLM-0.4-mini.

Only development seed 90401 is accepted. Formal seeds are never opened and this
module never emits a scientific decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import torch

from .data import base_math_eval_examples, base_story_eval_examples
from .gates import evaluate_base_prerequisites
from .lock import build_protocol_lock
from .m1 import environment_versions, run_m1_stream
from .model import TinyCLMDecoder
from .protocol import (
    CandidateOptimizerConfig,
    assert_seed_allowed,
    candidate_grid,
    canonical_json_hash,
    file_sha256,
    formal_model_config,
    load_protocol,
)
from .state import model_state_hash
from .tokenizer import TokenizerBundle
from .training import (
    BaseCorpusDataset,
    base_cell_activation_counts,
    exact_match_accuracy,
    train_base_model,
)

CALIBRATION_FORMAT = "minicells.clm-0.4-mini.m1-calibration.v1"
PLAN_FORMAT = "minicells.clm-0.4-mini.m1-calibration-plan.v1"
PLAN_LOCK_FORMAT = "minicells.clm-0.4-mini.m1-calibration-plan-lock.v1"
ASSET_FORMAT = "minicells.clm-0.4-mini.m1-calibration-assets.v1"


@dataclass(frozen=True)
class CalibrationCandidate:
    candidate_id: str
    ordinal: int
    estimated_candidate_steps: int
    direct: CandidateOptimizerConfig
    growth_private: CandidateOptimizerConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "ordinal": self.ordinal,
            "estimated_candidate_steps": self.estimated_candidate_steps,
            "direct": self.direct.to_dict(),
            "growth_private": self.growth_private.to_dict(),
        }


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_calibration_plan(protocol: Mapping[str, Any]) -> dict[str, Any]:
    pairs = []
    for direct in candidate_grid(protocol, "direct"):
        for growth in candidate_grid(protocol, "growth"):
            key = (
                direct.steps + growth.steps,
                direct.steps,
                growth.steps,
                direct.learning_rate,
                growth.learning_rate,
            )
            pairs.append((key, direct, growth))
    pairs.sort(key=lambda item: item[0])
    candidates = [
        CalibrationCandidate(
            f"candidate-{index:03d}",
            index,
            direct.steps + growth.steps,
            direct,
            growth,
        ).to_dict()
        for index, (_, direct, growth) in enumerate(pairs)
    ]
    payload = {
        "format": PLAN_FORMAT,
        "experiment_id": protocol["experiment_id"],
        "development_seed": int(protocol["replication"]["development_seed"]),
        "formal_seeds_observed": False,
        "selection_rule": (
            "first passing candidate in ascending "
            "(direct_steps+growth_steps, direct_steps, growth_steps, direct_lr, growth_lr)"
        ),
        "base_training_policy": "train exactly once for development seed and reuse one immutable checkpoint",
        "candidate_isolation": "every candidate starts from the identical immutable base checkpoint",
        "base_prerequisite_eval_examples_per_domain": 64,
        "base_cell_minimum_activation_rule": (
            "floor(base_sequences * topk_base / base_cells_per_layer * 0.5)"
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    payload["plan_sha256"] = canonical_json_hash(payload)
    return payload


def verify_committed_plan(protocol: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Verify compact pre-90401 lock, then return the fully expanded executable plan."""
    lock = _load(path)
    plan = build_calibration_plan(protocol)
    expected = {
        "format": PLAN_LOCK_FORMAT,
        "experiment_id": protocol["experiment_id"],
        "development_seed": int(protocol["replication"]["development_seed"]),
        "formal_seeds_observed_when_committed": False,
        "candidate_count": plan["candidate_count"],
        "plan_sha256": plan["plan_sha256"],
        "selection_rule": plan["selection_rule"],
        "base_training_policy": plan["base_training_policy"],
        "candidate_isolation": plan["candidate_isolation"],
        "base_prerequisite_eval_examples_per_domain": plan[
            "base_prerequisite_eval_examples_per_domain"
        ],
        "base_cell_minimum_activation_rule": plan["base_cell_minimum_activation_rule"],
    }
    if lock != expected:
        raise RuntimeError("committed calibration-plan lock does not match executable order")
    return plan


def minimum_base_cell_activation(protocol: Mapping[str, Any], *, base_sequences: int) -> int:
    cells = int(protocol["model"]["cell_layers"]["base_cells_per_layer"])
    topk = int(protocol["model"]["cell_layers"]["topk_base"])
    return max(1, math.floor(base_sequences * topk / cells * 0.5))


def _verify_embedded_hash(payload: Mapping[str, Any], label: str) -> None:
    material = dict(payload)
    recorded = str(material.pop("manifest_sha256", ""))
    if not recorded or canonical_json_hash(material) != recorded:
        raise RuntimeError(f"{label} embedded manifest hash mismatch")


def verify_calibration_assets(
    *,
    protocol: Mapping[str, Any],
    data_dir: str | Path,
    expected_assets_path: str | Path,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    expected = _load(expected_assets_path)
    if expected.get("format") != ASSET_FORMAT:
        raise RuntimeError("unexpected calibration asset identity format")
    summary = _load(data_dir / "asset-summary.json")
    tokenizer_manifest = _load(data_dir / "tokenizer" / "tokenizer-manifest.json")
    base_manifest = _load(data_dir / "base-corpus" / "base-corpus-manifest.json")
    curriculum_manifest = _load(data_dir / "curriculum-manifest.json")
    _verify_embedded_hash(tokenizer_manifest, "tokenizer manifest")
    _verify_embedded_hash(base_manifest, "base corpus manifest")
    _verify_embedded_hash(curriculum_manifest, "curriculum manifest")
    actual = {
        "dataset_revision": str(tokenizer_manifest.get("source_manifest", {}).get("revision", "")),
        "routing_salt": str(summary.get("routing_salt", "")),
        "base_tokens": int(base_manifest["actual_tokens"]),
        "tokenizer_hash": file_sha256(data_dir / "tokenizer" / "tokenizer.json"),
        "tokenizer_manifest_hash": str(tokenizer_manifest["manifest_sha256"]),
        "base_corpus_manifest_hash": str(base_manifest["manifest_sha256"]),
        "curriculum_manifest_hash": str(curriculum_manifest["manifest_sha256"]),
    }
    expected_identity = {
        key: expected[key]
        for key in (
            "dataset_revision",
            "routing_salt",
            "base_tokens",
            "tokenizer_hash",
            "tokenizer_manifest_hash",
            "base_corpus_manifest_hash",
            "curriculum_manifest_hash",
        )
    }
    expected_identity["base_tokens"] = int(expected_identity["base_tokens"])
    if actual != expected_identity:
        raise RuntimeError(
            "calibration assets differ from the identity frozen before 90401:\n"
            + json.dumps({"expected": expected_identity, "actual": actual}, indent=2, sort_keys=True)
        )
    target = int(protocol["base_training"]["target_tokens"])
    tolerance = float(protocol["base_training"]["token_tolerance_fraction"])
    if abs(actual["base_tokens"] - target) > target * tolerance:
        raise RuntimeError("base token count outside protocol tolerance")
    if int(curriculum_manifest["counts"]["total"]) != int(
        protocol["continual_curriculum"]["total_transactions"]
    ):
        raise RuntimeError("curriculum transaction count drift")
    return {
        "verified": True,
        "identity": actual,
        "identity_sha256": canonical_json_hash(actual),
        "tokenizer_manifest": tokenizer_manifest,
        "base_corpus_manifest": base_manifest,
        "curriculum_manifest": curriculum_manifest,
    }


def _save_base(path: Path, model: TinyCLMDecoder, seed: int, identity: Mapping, train: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "minicells.clm-0.4-mini.m1-calibration-base-checkpoint.v1",
            "seed": seed,
            "model_config": model.cfg.to_dict(),
            "model_state": model.state_dict(),
            "model_state_hash": model_state_hash(model),
            "asset_identity": dict(identity),
            "base_train": dict(train),
        },
        path,
    )


def _load_base(path: Path, model: TinyCLMDecoder, seed: int, identity: Mapping, device) -> dict:
    payload = torch.load(path, map_location=device)
    if payload.get("format") != "minicells.clm-0.4-mini.m1-calibration-base-checkpoint.v1":
        raise RuntimeError("unexpected calibration base checkpoint format")
    if int(payload["seed"]) != seed or dict(payload["asset_identity"]) != dict(identity):
        raise RuntimeError("calibration base checkpoint identity mismatch")
    if dict(payload["model_config"]) != model.cfg.to_dict():
        raise RuntimeError("calibration base checkpoint model-config mismatch")
    model.load_state_dict(payload["model_state"])
    if model_state_hash(model) != payload["model_state_hash"]:
        raise RuntimeError("calibration base checkpoint state-hash mismatch")
    return payload


def _finite(model: TinyCLMDecoder) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def prepare_or_load_base(
    *,
    protocol: Mapping[str, Any],
    data_dir: Path,
    out_dir: Path,
    assets: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[TinyCLMDecoder, TokenizerBundle, dict[str, Any]]:
    cfg = formal_model_config(protocol, routing_salt=assets["identity"]["routing_salt"])
    tokenizer = TokenizerBundle.load(data_dir / "tokenizer" / "tokenizer.json")
    dataset = BaseCorpusDataset(data_dir / "base-corpus")
    model = TinyCLMDecoder(cfg).to(device)
    checkpoint = out_dir / "base" / "checkpoint.pt"
    if checkpoint.is_file():
        saved = _load_base(checkpoint, model, seed, assets["identity"], device)
        base_train, source = saved["base_train"], "resumed"
    else:
        base_train = train_base_model(
            model, dataset=dataset, tokenizer=tokenizer, device=device, seed=seed
        )
        _save_base(checkpoint, model, seed, assets["identity"], base_train)
        source = "trained-once"

    counts = base_cell_activation_counts(model, dataset)
    threshold = minimum_base_cell_activation(protocol, base_sequences=len(dataset))
    math_acc = exact_match_accuracy(
        model, base_math_eval_examples(64), tokenizer=tokenizer, device=device
    )
    story_acc = exact_match_accuracy(
        model, base_story_eval_examples(64), tokenizer=tokenizer, device=device
    )
    prerequisites = evaluate_base_prerequisites(
        protocol=protocol,
        math_exact_match=math_acc,
        story_exact_match=story_acc,
        cell_activation_counts=counts,
        locked_minimum_activation=threshold,
        numeric_finite=_finite(model),
        hashes_match_lock=bool(assets["verified"]),
    )
    metrics = {
        "checkpoint": str(checkpoint),
        "checkpoint_source": source,
        "state_hash": model_state_hash(model),
        "base_sequences": len(dataset),
        "minimum_base_cell_activation": threshold,
        "math_exact_match": math_acc,
        "story_exact_match": story_acc,
        "base_train": base_train,
        "prerequisites": prerequisites,
    }
    _write(out_dir / "base" / "metrics.json", metrics)
    with (out_dir / "base" / "activation-counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "activation_count"])
        writer.writerows((cell_id, count) for cell_id, count in sorted(counts.items()))
    return model, tokenizer, metrics


def _candidate(payload: Mapping[str, Any]) -> CalibrationCandidate:
    return CalibrationCandidate(
        str(payload["candidate_id"]),
        int(payload["ordinal"]),
        int(payload["estimated_candidate_steps"]),
        CandidateOptimizerConfig(**dict(payload["direct"])),
        CandidateOptimizerConfig(**dict(payload["growth_private"])),
    )


def _selected_evidence(out_dir: Path, harnesses: Mapping[str, Any]) -> None:
    for variant, harness in harnesses.items():
        root = out_dir / "selected" / variant
        _write(root / "summary.json", harness.summary())
        with (root / "transactions.jsonl").open("w", encoding="utf-8") as handle:
            for record in harness.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        with (root / "cell-registry.jsonl").open("w", encoding="utf-8") as handle:
            for entry in harness.registry.snapshot(harness.model, harness.dependency_index):
                handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _summary_csv(out_dir: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = [
        "candidate_id", "ordinal", "estimated_candidate_steps", "direct_lr",
        "direct_steps", "growth_lr", "growth_steps", "pass",
        "effective_acceptance_rate", "false_safe_rate",
        "committed_gain_ratio_vs_local_always",
        "regression_damage_ratio_vs_local_always", "growth_rescue_rate",
        "private_reuse_acceptance_rate", "mean_direct_dependency_coverage",
        "wall_seconds",
    ]
    with (out_dir / "calibration-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            gates = row["gate_snapshot"]["gates"]
            growth = row["gate_snapshot"]["variant_summaries"]["local_tx_growth"]
            writer.writerow({
                "candidate_id": row["candidate"]["candidate_id"],
                "ordinal": row["candidate"]["ordinal"],
                "estimated_candidate_steps": row["candidate"]["estimated_candidate_steps"],
                "direct_lr": row["candidate"]["direct"]["learning_rate"],
                "direct_steps": row["candidate"]["direct"]["steps"],
                "growth_lr": row["candidate"]["growth_private"]["learning_rate"],
                "growth_steps": row["candidate"]["growth_private"]["steps"],
                "pass": row["pass"],
                "effective_acceptance_rate": growth["effective_acceptance_rate"],
                "false_safe_rate": growth["false_safe_rate"],
                "committed_gain_ratio_vs_local_always": gates[
                    "committed_gain_ratio_vs_local_always"
                ]["value"],
                "regression_damage_ratio_vs_local_always": gates[
                    "regression_damage_ratio_vs_local_always"
                ]["value"],
                "growth_rescue_rate": growth["growth_rescue_rate"],
                "private_reuse_acceptance_rate": growth["private_reuse_acceptance_rate"],
                "mean_direct_dependency_coverage": growth["mean_direct_dependency_coverage"],
                "wall_seconds": row["wall_seconds"],
            })


def run_calibration(
    *,
    protocol_path: str | Path,
    expected_assets_path: str | Path,
    committed_plan_path: str | Path,
    protocol_lock_template_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    seed: int,
    device: str | torch.device,
    code_commit: str,
    code_tree: str,
    tracked_tree_dirty: bool,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    assert_seed_allowed(protocol, mode="calibration", seed=seed)
    if tracked_tree_dirty or not code_commit or not code_tree:
        raise RuntimeError("calibration requires a clean committed source tree")
    out_dir, data_dir = Path(out_dir), Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = verify_committed_plan(protocol, committed_plan_path)
    _write(out_dir / "calibration-plan.json", plan)
    assets = verify_calibration_assets(
        protocol=protocol, data_dir=data_dir, expected_assets_path=expected_assets_path
    )
    _write(out_dir / "asset-verification.json", {
        "verified": True,
        "identity": assets["identity"],
        "identity_sha256": assets["identity_sha256"],
    })

    device = torch.device(device)
    # Seed here, before constructing the formal model, so direct API calls are as
    # deterministic as the CLI runner. The base checkpoint is then trained once.
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model, tokenizer, base = prepare_or_load_base(
        protocol=protocol, data_dir=data_dir, out_dir=out_dir,
        assets=assets, seed=seed, device=device
    )
    common = {
        "format": CALIBRATION_FORMAT,
        "seed": seed,
        "scientific_decision": False,
        "development_seed_observed": True,
        "formal_seeds_observed": False,
        "asset_identity": assets["identity"],
        "plan_sha256": plan["plan_sha256"],
        "base": base,
        "code_commit": code_commit,
        "code_tree": code_tree,
        "environment": environment_versions(device),
    }
    if not base["prerequisites"]["pass"]:
        decision = {
            "status": "CALIBRATION_BASE_PREREQUISITES_FAILED",
            "scientific_decision": False,
            "development_seed_observed": True,
            "formal_seeds_observed": False,
            "selected_candidate": None,
        }
        _write(out_dir / "decision.json", decision)
        _write(out_dir / "summary.json", {**common, "decision": decision})
        return decision

    immutable_hash = model_state_hash(model)
    rows, selected, selected_gates = [], None, None
    for payload in plan["candidates"]:
        candidate = _candidate(payload)
        result_path = out_dir / "candidates" / candidate.candidate_id / "candidate.json"
        if result_path.is_file():
            row = _load(result_path)
            if row["candidate"] != candidate.to_dict():
                raise RuntimeError(f"resume candidate drift: {candidate.candidate_id}")
            rows.append(row)
            if row["pass"]:
                selected, selected_gates = candidate, row["gate_snapshot"]
                break
            continue
        if model_state_hash(model) != immutable_hash:
            raise RuntimeError("immutable base model changed before candidate")
        started = time.perf_counter()
        harnesses, gate_snapshot = run_m1_stream(
            protocol=protocol,
            base_model=model,
            tokenizer=tokenizer,
            curriculum_manifest=assets["curriculum_manifest"],
            direct_optimizer=candidate.direct,
            growth_optimizer=candidate.growth_private,
            seed=seed,
            device=device,
            out_dir=None,
            smoke_projection=False,
        )
        if model_state_hash(model) != immutable_hash:
            raise RuntimeError("candidate mutated immutable base model")
        row = {
            "candidate": candidate.to_dict(),
            "pass": bool(gate_snapshot["pass"]),
            "gate_snapshot": gate_snapshot,
            "base_state_hash_before_and_after": immutable_hash,
            "wall_seconds": time.perf_counter() - started,
        }
        _write(result_path, row)
        rows.append(row)
        if row["pass"]:
            selected, selected_gates = candidate, gate_snapshot
            _selected_evidence(out_dir, harnesses)
            del harnesses
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
        del harnesses
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _summary_csv(out_dir, rows)
    if selected is None:
        decision = {
            "status": "CALIBRATION_NO_CONFIGURATION_PASSED",
            "scientific_decision": False,
            "development_seed_observed": True,
            "formal_seeds_observed": False,
            "selected_candidate": None,
            "candidates_evaluated": len(rows),
        }
        _write(out_dir / "decision.json", decision)
        _write(out_dir / "summary.json", {**common, "decision": decision})
        return decision

    selected_payload = {
        "candidate": selected.to_dict(),
        "selection_rule": plan["selection_rule"],
        "first_passing_ordinal": selected.ordinal,
        "gate_snapshot": selected_gates,
        "candidates_evaluated": len(rows),
    }
    _write(out_dir / "selected.json", selected_payload)
    protocol_lock = build_protocol_lock(
        protocol=protocol,
        template=_load(protocol_lock_template_path),
        protocol_path=protocol_path,
        direct_optimizer=selected.direct,
        growth_optimizer=selected.growth_private,
        tokenizer_manifest=assets["tokenizer_manifest"],
        base_corpus_manifest=assets["base_corpus_manifest"],
        curriculum_manifest=assets["curriculum_manifest"],
        dataset_revision=assets["identity"]["dataset_revision"],
        routing_salt=assets["identity"]["routing_salt"],
        minimum_base_cell_activation=base["minimum_base_cell_activation"],
        code_commit=code_commit,
        code_tree=code_tree,
        environment=environment_versions(device),
    )
    _write(out_dir / "protocol-lock.candidate.json", protocol_lock)
    decision = {
        "status": "CALIBRATION_CONFIGURATION_SELECTED",
        "scientific_decision": False,
        "development_seed_observed": True,
        "formal_seeds_observed": False,
        "selected_candidate": selected.candidate_id,
        "candidates_evaluated": len(rows),
        "formal_execution_authorized": False,
        "next_required_action": (
            "commit protocol-lock.candidate.json as canonical protocol-lock.json "
            "before any formal seed is opened"
        ),
    }
    _write(out_dir / "decision.json", decision)
    _write(out_dir / "summary.json", {
        **common, "decision": decision, "selected": selected_payload,
        "protocol_lock_candidate": "protocol-lock.candidate.json",
    })
    return decision


def write_plan_only(
    *,
    protocol_path: str | Path,
    committed_plan_path: str | Path,
    expected_assets_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    plan = verify_committed_plan(protocol, committed_plan_path)
    assets = _load(expected_assets_path)
    if assets.get("format") != ASSET_FORMAT:
        raise RuntimeError("unexpected calibration asset identity format")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "calibration-plan.json", plan)
    _write(out_dir / "expected-assets.json", assets)
    decision = {
        "status": "CALIBRATION_PLAN_ONLY",
        "scientific_decision": False,
        "development_seed_observed": False,
        "formal_seeds_observed": False,
        "candidate_count": plan["candidate_count"],
        "plan_sha256": plan["plan_sha256"],
    }
    _write(out_dir / "decision.json", decision)
    _write(out_dir / "summary.json", decision)
    return decision
