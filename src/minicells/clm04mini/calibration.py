"""M1 development-seed calibration harness for CLM-0.4-mini.

This module implements the pre-registered finite-grid selection procedure. It is
deliberately separate from formal execution: only development seed 90401 is
accepted, formal seeds are never opened, and no scientific decision is emitted.
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

from .curriculum import build_curriculum
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
            "ordinal": int(self.ordinal),
            "estimated_candidate_steps": int(self.estimated_candidate_steps),
            "direct": self.direct.to_dict(),
            "growth_private": self.growth_private.to_dict(),
        }


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_calibration_plan(protocol: Mapping[str, Any]) -> dict[str, Any]:
    direct = candidate_grid(protocol, "direct")
    growth = candidate_grid(protocol, "growth")
    pairs: list[tuple[tuple[Any, ...], CandidateOptimizerConfig, CandidateOptimizerConfig]] = []
    for direct_cfg in direct:
        for growth_cfg in growth:
            key = (
                int(direct_cfg.steps) + int(growth_cfg.steps),
                int(direct_cfg.steps),
                int(growth_cfg.steps),
                float(direct_cfg.learning_rate),
                float(growth_cfg.learning_rate),
            )
            pairs.append((key, direct_cfg, growth_cfg))
    pairs.sort(key=lambda item: item[0])
    candidates = [
        CalibrationCandidate(
            candidate_id=f"candidate-{index:03d}",
            ordinal=index,
            estimated_candidate_steps=int(direct_cfg.steps) + int(growth_cfg.steps),
            direct=direct_cfg,
            growth_private=growth_cfg,
        ).to_dict()
        for index, (_, direct_cfg, growth_cfg) in enumerate(pairs)
    ]
    payload = {
        "format": PLAN_FORMAT,
        "experiment_id": str(protocol["experiment_id"]),
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
    committed = _load_json(path)
    expected = build_calibration_plan(protocol)
    if committed != expected:
        raise RuntimeError("committed calibration plan does not match executable registered order")
    return committed


def minimum_base_cell_activation(protocol: Mapping[str, Any], *, base_sequences: int) -> int:
    cells = int(protocol["model"]["cell_layers"]["base_cells_per_layer"])
    topk = int(protocol["model"]["cell_layers"]["topk_base"])
    expected_mean = float(base_sequences) * float(topk) / float(cells)
    return max(1, int(math.floor(expected_mean * 0.5)))


def _verify_manifest_hash(payload: Mapping[str, Any], *, label: str) -> None:
    if "manifest_sha256" not in payload:
        raise RuntimeError(f"{label} is missing manifest_sha256")
    material = dict(payload)
    recorded = str(material.pop("manifest_sha256"))
    calculated = canonical_json_hash(material)
    if recorded != calculated:
        raise RuntimeError(f"{label} embedded manifest hash mismatch")


def verify_calibration_assets(
    *,
    protocol: Mapping[str, Any],
    data_dir: str | Path,
    expected_assets_path: str | Path,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    expected = _load_json(expected_assets_path)
    if expected.get("format") != ASSET_FORMAT:
        raise RuntimeError("unexpected calibration asset identity format")
    summary = _load_json(data_dir / "asset-summary.json")
    tokenizer_manifest = _load_json(data_dir / "tokenizer" / "tokenizer-manifest.json")
    base_manifest = _load_json(data_dir / "base-corpus" / "base-corpus-manifest.json")
    curriculum_manifest = _load_json(data_dir / "curriculum-manifest.json")

    _verify_manifest_hash(tokenizer_manifest, label="tokenizer manifest")
    _verify_manifest_hash(base_manifest, label="base corpus manifest")
    _verify_manifest_hash(curriculum_manifest, label="curriculum manifest")

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
        "dataset_revision": str(expected["dataset_revision"]),
        "routing_salt": str(expected["routing_salt"]),
        "base_tokens": int(expected["base_tokens"]),
        "tokenizer_hash": str(expected["tokenizer_hash"]),
        "tokenizer_manifest_hash": str(expected["tokenizer_manifest_hash"]),
        "base_corpus_manifest_hash": str(expected["base_corpus_manifest_hash"]),
        "curriculum_manifest_hash": str(expected["curriculum_manifest_hash"]),
    }
    if actual != expected_identity:
        raise RuntimeError(
            "calibration assets do not match committed pre-90401 identity:\n"
            + json.dumps({"expected": expected_identity, "actual": actual}, indent=2, sort_keys=True)
        )
    target = int(protocol["base_training"]["target_tokens"])
    tolerance = float(protocol["base_training"]["token_tolerance_fraction"])
    if abs(actual["base_tokens"] - target) > target * tolerance:
        raise RuntimeError("base token count is outside frozen protocol tolerance")
    if int(curriculum_manifest["total_transactions"]) != int(
        protocol["continual_curriculum"]["total_transactions"]
    ):
        raise RuntimeError("curriculum transaction count drift")
    return {
        "verified": True,
        "identity": actual,
        "expected_identity_sha256": canonical_json_hash(expected_identity),
        "tokenizer_manifest": tokenizer_manifest,
        "base_corpus_manifest": base_manifest,
        "curriculum_manifest": curriculum_manifest,
    }


def _base_checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "base" / "checkpoint.pt"


def _save_base_checkpoint(
    *,
    path: Path,
    model: TinyCLMDecoder,
    seed: int,
    asset_identity: Mapping[str, Any],
    base_train: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "minicells.clm-0.4-mini.m1-calibration-base-checkpoint.v1",
            "seed": int(seed),
            "model_config": model.cfg.to_dict(),
            "model_state": model.state_dict(),
            "model_state_hash": model_state_hash(model),
            "asset_identity": dict(asset_identity),
            "base_train": dict(base_train),
        },
        path,
    )


def _load_base_checkpoint(
    *,
    path: Path,
    model: TinyCLMDecoder,
    seed: int,
    asset_identity: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if payload.get("format") != "minicells.clm-0.4-mini.m1-calibration-base-checkpoint.v1":
        raise RuntimeError("unexpected calibration base checkpoint format")
    if int(payload["seed"]) != int(seed):
        raise RuntimeError("calibration base checkpoint seed mismatch")
    if dict(payload["asset_identity"]) != dict(asset_identity):
        raise RuntimeError("calibration base checkpoint asset identity mismatch")
    if dict(payload["model_config"]) != model.cfg.to_dict():
        raise RuntimeError("calibration base checkpoint model configuration mismatch")
    model.load_state_dict(payload["model_state"])
    actual_hash = model_state_hash(model)
    if actual_hash != str(payload["model_state_hash"]):
        raise RuntimeError("calibration base checkpoint state hash mismatch")
    return payload


def _write_activation_csv(path: Path, counts: Mapping[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "activation_count"])
        for cell_id, count in sorted(counts.items()):
            writer.writerow([cell_id, int(count)])


def _numeric_finite(model: TinyCLMDecoder) -> bool:
    return all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())


def prepare_or_load_base(
    *,
    protocol: Mapping[str, Any],
    data_dir: Path,
    out_dir: Path,
    asset_verification: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[TinyCLMDecoder, TokenizerBundle, BaseCorpusDataset, dict[str, Any]]:
    routing_salt = str(asset_verification["identity"]["routing_salt"])
    cfg = formal_model_config(protocol, routing_salt=routing_salt)
    tokenizer = TokenizerBundle.load(data_dir / "tokenizer" / "tokenizer.json")
    if tokenizer.vocab_size > cfg.vocab_size:
        raise RuntimeError("tokenizer vocabulary exceeds frozen model vocabulary")
    dataset = BaseCorpusDataset(data_dir / "base-corpus")
    model = TinyCLMDecoder(cfg).to(device)
    checkpoint_path = _base_checkpoint_path(out_dir)

    if checkpoint_path.is_file():
        checkpoint = _load_base_checkpoint(
            path=checkpoint_path,
            model=model,
            seed=seed,
            asset_identity=asset_verification["identity"],
            device=device,
        )
        base_train = dict(checkpoint["base_train"])
        source = "resumed"
    else:
        torch.manual_seed(int(seed))
        random.seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        base_train = train_base_model(
            model,
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            seed=seed,
        )
        _save_base_checkpoint(
            path=checkpoint_path,
            model=model,
            seed=seed,
            asset_identity=asset_verification["identity"],
            base_train=base_train,
        )
        source = "trained-once"

    counts = base_cell_activation_counts(model, dataset)
    minimum_activation = minimum_base_cell_activation(protocol, base_sequences=len(dataset))
    math_accuracy = exact_match_accuracy(
        model,
        base_math_eval_examples(64),
        tokenizer=tokenizer,
        device=device,
    )
    story_accuracy = exact_match_accuracy(
        model,
        base_story_eval_examples(64),
        tokenizer=tokenizer,
        device=device,
    )
    prerequisites = evaluate_base_prerequisites(
        protocol=protocol,
        math_exact_match=math_accuracy,
        story_exact_match=story_accuracy,
        cell_activation_counts=counts,
        locked_minimum_activation=minimum_activation,
        numeric_finite=_numeric_finite(model),
        hashes_match_lock=bool(asset_verification["verified"]),
    )
    metrics = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_source": source,
        "state_hash": model_state_hash(model),
        "base_sequences": len(dataset),
        "minimum_base_cell_activation": int(minimum_activation),
        "math_exact_match": float(math_accuracy),
        "story_exact_match": float(story_accuracy),
        "base_train": base_train,
        "prerequisites": prerequisites,
    }
    _write_json(out_dir / "base" / "metrics.json", metrics)
    _write_activation_csv(out_dir / "base" / "activation-counts.csv", counts)
    return model, tokenizer, dataset, metrics


def _candidate_from_dict(payload: Mapping[str, Any]) -> CalibrationCandidate:
    return CalibrationCandidate(
        candidate_id=str(payload["candidate_id"]),
        ordinal=int(payload["ordinal"]),
        estimated_candidate_steps=int(payload["estimated_candidate_steps"]),
        direct=CandidateOptimizerConfig(**dict(payload["direct"])),
        growth_private=CandidateOptimizerConfig(**dict(payload["growth_private"])),
    )


def _write_selected_evidence(out_dir: Path, harnesses: Mapping[str, Any]) -> None:
    selected_dir = out_dir / "selected"
    for variant, harness in harnesses.items():
        variant_dir = selected_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        _write_json(variant_dir / "summary.json", harness.summary())
        with (variant_dir / "transactions.jsonl").open("w", encoding="utf-8") as handle:
            for record in harness.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        with (variant_dir / "cell-registry.jsonl").open("w", encoding="utf-8") as handle:
            for entry in harness.registry.snapshot(harness.model, harness.dependency_index):
                handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _write_calibration_csv(out_dir: Path, rows: list[Mapping[str, Any]]) -> None:
    path = out_dir / "calibration-summary.csv"
    fields = [
        "candidate_id",
        "ordinal",
        "estimated_candidate_steps",
        "direct_lr",
        "direct_steps",
        "growth_lr",
        "growth_steps",
        "pass",
        "effective_acceptance_rate",
        "false_safe_rate",
        "committed_gain_ratio_vs_local_always",
        "regression_damage_ratio_vs_local_always",
        "growth_rescue_rate",
        "private_reuse_acceptance_rate",
        "mean_direct_dependency_coverage",
        "wall_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            gates = row["gate_snapshot"]["gates"]
            growth = row["gate_snapshot"]["variant_summaries"]["local_tx_growth"]
            writer.writerow(
                {
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
                    "private_reuse_acceptance_rate": growth[
                        "private_reuse_acceptance_rate"
                    ],
                    "mean_direct_dependency_coverage": growth[
                        "mean_direct_dependency_coverage"
                    ],
                    "wall_seconds": row["wall_seconds"],
                }
            )


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
    if tracked_tree_dirty:
        raise RuntimeError("calibration requires a clean tracked source tree")
    if not code_commit or not code_tree:
        raise RuntimeError("calibration requires git commit/tree provenance")

    out_dir = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = verify_committed_plan(protocol, committed_plan_path)
    _write_json(out_dir / "calibration-plan.json", plan)

    assets = verify_calibration_assets(
        protocol=protocol,
        data_dir=data_dir,
        expected_assets_path=expected_assets_path,
    )
    _write_json(
        out_dir / "asset-verification.json",
        {
            "verified": True,
            "identity": assets["identity"],
            "expected_identity_sha256": assets["expected_identity_sha256"],
        },
    )

    device_obj = torch.device(device)
    model, tokenizer, _dataset, base_metrics = prepare_or_load_base(
        protocol=protocol,
        data_dir=data_dir,
        out_dir=out_dir,
        asset_verification=assets,
        seed=seed,
        device=device_obj,
    )

    summary_base = {
        "format": CALIBRATION_FORMAT,
        "seed": int(seed),
        "scientific_decision": False,
        "development_seed_observed": True,
        "formal_seeds_observed": False,
        "asset_identity": assets["identity"],
        "plan_sha256": plan["plan_sha256"],
        "base": base_metrics,
        "code_commit": str(code_commit),
        "code_tree": str(code_tree),
        "environment": environment_versions(device_obj),
    }
    if not bool(base_metrics["prerequisites"]["pass"]):
        decision = {
            "status": "CALIBRATION_BASE_PREREQUISITES_FAILED",
            "scientific_decision": False,
            "development_seed_observed": True,
            "formal_seeds_observed": False,
            "selected_candidate": None,
        }
        _write_json(out_dir / "decision.json", decision)
        _write_json(out_dir / "summary.json", {**summary_base, "decision": decision})
        return decision

    base_hash = model_state_hash(model)
    rows: list[dict[str, Any]] = []
    selected_candidate: CalibrationCandidate | None = None
    selected_gate_snapshot: dict[str, Any] | None = None

    for candidate_payload in plan["candidates"]:
        candidate = _candidate_from_dict(candidate_payload)
        candidate_dir = out_dir / "candidates" / candidate.candidate_id
        result_path = candidate_dir / "candidate.json"
        if result_path.is_file():
            existing = _load_json(result_path)
            if existing.get("candidate") != candidate.to_dict():
                raise RuntimeError(f"resume candidate drift: {candidate.candidate_id}")
            rows.append(existing)
            if bool(existing.get("pass")):
                selected_candidate = candidate
                selected_gate_snapshot = dict(existing["gate_snapshot"])
                break
            continue

        if model_state_hash(model) != base_hash:
            raise RuntimeError("immutable calibration base model changed before candidate")

        started = time.perf_counter()
        harnesses, gate_snapshot = run_m1_stream(
            protocol=protocol,
            base_model=model,
            tokenizer=tokenizer,
            curriculum_manifest=assets["curriculum_manifest"],
            direct_optimizer=candidate.direct,
            growth_optimizer=candidate.growth_private,
            seed=seed,
            device=device_obj,
            out_dir=None,
            smoke_projection=False,
        )
        if model_state_hash(model) != base_hash:
            raise RuntimeError("candidate mutated immutable calibration base model")
        row = {
            "candidate": candidate.to_dict(),
            "pass": bool(gate_snapshot["pass"]),
            "gate_snapshot": gate_snapshot,
            "base_state_hash_before_and_after": base_hash,
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json(result_path, row)
        rows.append(row)
        if row["pass"]:
            selected_candidate = candidate
            selected_gate_snapshot = gate_snapshot
            _write_selected_evidence(out_dir, harnesses)
            del harnesses
            gc.collect()
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()
            break
        del harnesses
        gc.collect()
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    _write_calibration_csv(out_dir, rows)

    if selected_candidate is None:
        decision = {
            "status": "CALIBRATION_NO_CONFIGURATION_PASSED",
            "scientific_decision": False,
            "development_seed_observed": True,
            "formal_seeds_observed": False,
            "selected_candidate": None,
            "candidates_evaluated": len(rows),
        }
        _write_json(out_dir / "decision.json", decision)
        _write_json(
            out_dir / "summary.json",
            {**summary_base, "decision": decision, "candidates_evaluated": len(rows)},
        )
        return decision

    selected = {
        "candidate": selected_candidate.to_dict(),
        "selection_rule": plan["selection_rule"],
        "first_passing_ordinal": int(selected_candidate.ordinal),
        "gate_snapshot": selected_gate_snapshot,
        "candidates_evaluated": len(rows),
    }
    _write_json(out_dir / "selected.json", selected)

    template = _load_json(protocol_lock_template_path)
    protocol_lock = build_protocol_lock(
        protocol=protocol,
        template=template,
        protocol_path=protocol_path,
        direct_optimizer=selected_candidate.direct,
        growth_optimizer=selected_candidate.growth_private,
        tokenizer_manifest=assets["tokenizer_manifest"],
        base_corpus_manifest=assets["base_corpus_manifest"],
        curriculum_manifest=assets["curriculum_manifest"],
        dataset_revision=str(assets["identity"]["dataset_revision"]),
        routing_salt=str(assets["identity"]["routing_salt"]),
        minimum_base_cell_activation=int(base_metrics["minimum_base_cell_activation"]),
        code_commit=str(code_commit),
        code_tree=str(code_tree),
        environment=environment_versions(device_obj),
    )
    _write_json(out_dir / "protocol-lock.candidate.json", protocol_lock)

    decision = {
        "status": "CALIBRATION_CONFIGURATION_SELECTED",
        "scientific_decision": False,
        "development_seed_observed": True,
        "formal_seeds_observed": False,
        "selected_candidate": selected_candidate.candidate_id,
        "candidates_evaluated": len(rows),
        "formal_execution_authorized": False,
        "next_required_action": (
            "review and commit protocol-lock.candidate.json as canonical protocol-lock.json "
            "before any formal seed is opened"
        ),
    }
    _write_json(out_dir / "decision.json", decision)
    _write_json(
        out_dir / "summary.json",
        {
            **summary_base,
            "decision": decision,
            "selected": selected,
            "protocol_lock_candidate": "protocol-lock.candidate.json",
        },
    )
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
    assets = _load_json(expected_assets_path)
    if assets.get("format") != ASSET_FORMAT:
        raise RuntimeError("unexpected calibration asset identity format")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "calibration-plan.json", plan)
    _write_json(out_dir / "expected-assets.json", assets)
    decision = {
        "status": "CALIBRATION_PLAN_ONLY",
        "scientific_decision": False,
        "development_seed_observed": False,
        "formal_seeds_observed": False,
        "candidate_count": int(plan["candidate_count"]),
        "plan_sha256": str(plan["plan_sha256"]),
    }
    _write_json(out_dir / "decision.json", decision)
    _write_json(out_dir / "summary.json", decision)
    return decision
