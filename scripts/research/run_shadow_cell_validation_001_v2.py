#!/usr/bin/env python3
"""Run one registered Shadow Cell Validation 001 v2 seed.

Formal execution is deliberately fail-closed. It requires a separately locked
protocol, a hashed canonical accepted checkpoint, and an injected formal
dataset. Only smoke mode may use deterministic toy data.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minicells.clm04mini.examples import ScoredTokenExample, collate_scored  # noqa: E402
from minicells.clm04mini.model import MiniCLMConfig, TinyCLMDecoder  # noqa: E402
from minicells.clm04mini.protocol import ProtocolError  # noqa: E402
from minicells.shadow_maturation import (  # noqa: E402
    MATURITY_GRID,
    AcceptedModelChain,
    AcceptedModelSnapshot,
    ShadowSidecar,
    build_activation_certificates,
    build_functional_sketch,
    calibrate_input_gate,
    copy_on_write_artifact,
    evaluate_model_metrics,
    evaluate_maturity_frontier,
    evaluate_sidecar_metrics,
    hash_accepted_state,
    interpolate_models,
    m0_equivalence_delta,
    routing_is_preserved,
    select_oracle_maturity,
    select_sketch_maturity,
    synthetic_examples,
    train_corrected_direct,
    train_shadow,
)
from publish_shadow_cell_validation_001_v2 import publish_results  # noqa: E402


VALIDATION_ID = "shadow-cell-validation-001-v2-developmental-maturation"
PROTOCOL_PATH = ROOT / "research/validations" / VALIDATION_ID / "protocol.json"
LOCK_PATH = ROOT / "research/validations" / VALIDATION_ID / "protocol-lock.json"
FORMAL_SEEDS = (95311, 95312, 95313)
DEVELOPMENT_SEED = 95301
REQUIRED_SPLITS = (
    "A_train", "A_calibration", "A_eval", "B_train", "B_calibration", "B_eval",
    "C_train", "C_calibration", "C_eval", "D_train", "D_calibration", "D_eval",
)
IMPLEMENTATION_FILES = (
    "scripts/research/run_shadow_cell_validation_001_v2.py",
    "scripts/research/aggregate_shadow_cell_validation_001_v2.py",
    "scripts/research/publish_shadow_cell_validation_001_v2.py",
    "scripts/research/report_shadow_cell_validation_001_v2.py",
    "src/minicells/shadow_maturation.py",
    f"research/validations/{VALIDATION_ID}/protocol.json",
)


class _Tokenizer:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = int(pad_id)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("validation_id") != VALIDATION_ID:
        raise ProtocolError("unexpected Shadow v2 validation id")
    if [float(x) for x in payload["maturity_grid"]] != list(MATURITY_GRID):
        raise ProtocolError("maturity grid drift")
    if [int(x) for x in payload["formal_seeds"]] != list(FORMAL_SEEDS):
        raise ProtocolError("formal seed family drift")
    if payload.get("status") != "REGISTERED_NOT_RUN":
        raise ProtocolError("Shadow v2 protocol status is not REGISTERED_NOT_RUN")
    return payload


def _protocol_sha() -> str:
    return _sha256_file(PROTOCOL_PATH)


def _load_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise ProtocolError(f"missing protocol lock: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("validation_id") != VALIDATION_ID:
        raise ProtocolError("protocol lock validation id mismatch")
    return lock


def _assert_formal_lock(protocol: dict[str, Any], checkpoint: Path, dataset: Path, seed: int) -> dict[str, str]:
    lock = _load_lock()
    actual_protocol = _protocol_sha()
    if lock.get("status") != "FROZEN":
        raise ProtocolError(
            "formal execution is locked: protocol-lock.json must be FROZEN after "
            "canonical checkpoint and dataset hashes are recorded"
        )
    if lock.get("protocol_sha256") != actual_protocol:
        raise ProtocolError("protocol hash does not match protocol-lock.json")
    expected_checkpoint = lock.get("canonical_checkpoint_sha256")
    expected_datasets = lock.get("formal_dataset_sha256")
    expected_dataset = expected_datasets.get(str(seed)) if isinstance(expected_datasets, dict) else expected_datasets
    protocol_datasets = protocol["formal_dataset"].get("sha256")
    protocol_dataset = protocol_datasets.get(str(seed)) if isinstance(protocol_datasets, dict) else protocol_datasets
    if not expected_checkpoint or not expected_dataset:
        raise ProtocolError("formal lock is missing canonical checkpoint or dataset SHA-256")
    if protocol["model"].get("canonical_checkpoint_sha256") != expected_checkpoint:
        raise ProtocolError("protocol checkpoint hash and protocol lock disagree")
    if protocol_dataset != expected_dataset:
        raise ProtocolError("protocol dataset hash and protocol lock disagree")
    locked_files = lock.get("implementation_files")
    if not isinstance(locked_files, dict) or set(locked_files) != set(IMPLEMENTATION_FILES):
        raise ProtocolError("formal lock is missing the registered implementation manifest")
    actual_files = {name: _sha256_file(ROOT / name) for name in IMPLEMENTATION_FILES}
    if actual_files != locked_files:
        raise ProtocolError("implementation file hash does not match protocol-lock.json")
    manifest = hashlib.sha256(
        json.dumps(actual_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest != lock.get("implementation_manifest_sha256"):
        raise ProtocolError("implementation manifest SHA-256 does not match protocol-lock.json")
    checkpoint_sha = _sha256_file(checkpoint)
    dataset_sha = _sha256_file(dataset)
    if checkpoint_sha != expected_checkpoint:
        raise ProtocolError(f"canonical checkpoint SHA-256 mismatch: {checkpoint_sha}")
    if dataset_sha != expected_dataset:
        raise ProtocolError(f"formal dataset SHA-256 mismatch: {dataset_sha}")
    return {
        "protocol_sha256": actual_protocol,
        "checkpoint_sha256": checkpoint_sha,
        "dataset_sha256": dataset_sha,
        "implementation_manifest_sha256": manifest,
    }


def _assert_seed(protocol: dict[str, Any], phase: str, seed: int) -> None:
    if phase == "smoke":
        if int(seed) != DEVELOPMENT_SEED:
            raise ProtocolError(f"smoke requires development seed {DEVELOPMENT_SEED}")
    elif phase == "formal":
        if int(seed) not in FORMAL_SEEDS:
            raise ProtocolError(f"formal requires one of {FORMAL_SEEDS}")
    else:
        raise ProtocolError(f"unknown phase {phase}")


def _config(protocol: dict[str, Any], *, smoke: bool) -> MiniCLMConfig:
    if smoke:
        return MiniCLMConfig(vocab_size=64, max_seq_len=16, num_layers=4, d_model=16, n_heads=4, dense_ff_hidden=32, base_cells=4, cell_hidden=4, routing_salt="shadow-cell-v2-smoke")
    model = protocol["model"]
    cells = model["cell_layers"]
    dense = model["shared_dense_layers"]
    return MiniCLMConfig(vocab_size=int(model["vocab_size"]), max_seq_len=int(model["context_length"]), num_layers=int(model["layers"]), d_model=int(model["width"]), n_heads=int(model["attention_heads"]), dense_ff_hidden=int(dense["ffn_hidden"]), base_cells=int(cells["base_cells_per_layer"]), cell_hidden=int(cells["base_cell_hidden"]), routing_salt=str(model["routing_salt"]))


def _load_model(path: Path, cfg: MiniCLMConfig, device: torch.device) -> TinyCLMDecoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise ProtocolError("canonical checkpoint does not contain a state dictionary")
    model = TinyCLMDecoder(cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _shared_synthetic(values: list[ScoredTokenExample]) -> list[ScoredTokenExample]:
    return [ScoredTokenExample(example_id=item.example_id, address_id=f"shared-parent/example-{index:04d}", tokens=item.tokens, target_mask=item.target_mask, knowledge_key=item.knowledge_key, prompt_text=item.prompt_text, answer_text=item.answer_text) for index, item in enumerate(values)]


def _smoke_examples(cfg: MiniCLMConfig, seed: int) -> tuple[dict[str, list[ScoredTokenExample]], _Tokenizer]:
    values = {
        "A": _shared_synthetic(synthetic_examples(vocab_size=cfg.vocab_size, domain="base", count=8, seed=seed + 1)),
        "B": _shared_synthetic(synthetic_examples(vocab_size=cfg.vocab_size, domain="math", count=8, seed=seed + 2)),
        "C": _shared_synthetic(synthetic_examples(vocab_size=cfg.vocab_size, domain="story", count=8, seed=seed + 3)),
        "D": _shared_synthetic(synthetic_examples(vocab_size=cfg.vocab_size, domain="math", count=8, seed=seed + 4)),
    }
    for phase in ("A", "B", "C", "D"):
        values[f"{phase}_train"] = values[phase]
        values[f"{phase}_eval"] = values[phase]
    values["A_calibration"] = values["A"]
    values["B_calibration"] = values["B"]
    values["C_calibration"] = values["C"]
    values["D_calibration"] = values["D"]
    return values, _Tokenizer(0)


def _formal_examples(path: Path, cfg: MiniCLMConfig, seed: int) -> tuple[dict[str, list[ScoredTokenExample]], _Tokenizer]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "minicells.shadow-cell-validation-001-v2.dataset.v1":
        raise ProtocolError("formal dataset format mismatch")
    if int(payload.get("seed", -1)) != int(seed):
        raise ProtocolError("formal dataset seed mismatch")
    raw = payload.get("splits")
    missing = [key for key in REQUIRED_SPLITS if not isinstance(raw, dict) or key not in raw]
    if missing:
        raise ProtocolError(f"formal dataset missing splits: {missing}")
    result: dict[str, list[ScoredTokenExample]] = {}
    for key in REQUIRED_SPLITS:
        values = raw[key]
        if not isinstance(values, list) or not values:
            raise ProtocolError(f"formal dataset split {key} is empty or invalid")
        result[key] = [ScoredTokenExample.from_dict(item) for item in values]
        for example in result[key]:
            if max(example.tokens) >= cfg.vocab_size or min(example.tokens) < 0:
                raise ProtocolError(f"token out of vocabulary in split {key}")
            if len(example.tokens) - 1 > cfg.max_seq_len:
                raise ProtocolError(f"sequence exceeds model context in split {key}")
    eval_addresses = [
        example.address_id
        for key in ("A_eval", "B_eval", "C_eval", "D_eval")
        for example in result[key]
    ]
    if len(set(eval_addresses)) != len(eval_addresses):
        raise ProtocolError("formal evaluation address_id values must be globally unique for shuffled-gate control")
    result.update({phase: result[f"{phase}_train"] for phase in ("A", "B", "C", "D")})
    return result, _Tokenizer(int(payload.get("pad_id", 0)))


def _parent_overlap(model: TinyCLMDecoder, examples: dict[str, list[ScoredTokenExample]]) -> dict[str, Any]:
    def signature(item: ScoredTokenExample) -> tuple[tuple[int, ...], ...]:
        routes = model.base_routes(item.address_id)
        return tuple(tuple(routes[str(layer)]) for layer in (3, 4))
    by_split = {key: [signature(item) for item in examples[key]] for key in REQUIRED_SPLITS}
    all_rows = [route for key in REQUIRED_SPLITS for route in by_split[key]]
    anchor = all_rows[0] if all_rows else None
    result = {
        "same_complete_route_tuple": bool(all_rows) and len(set(all_rows)) == 1,
        "parent_route_signature": [list(route) for route in anchor] if anchor else None,
    }
    result.update({f"{key}_fraction": sum(value == anchor for value in rows) / max(1, len(rows)) for key, rows in by_split.items()})
    result["route_counts"] = {key: len(set(rows)) for key, rows in by_split.items()}
    return result


def _metric_delta(before: dict[str, float], after: dict[str, float], direct_gain: float | None = None) -> dict[str, float]:
    raw_gain = before["nll"] - after["nll"]
    row = {"before_nll": before["nll"], "after_nll": after["nll"], "before_accuracy": before["accuracy"], "after_accuracy": after["accuracy"], "new_gain": raw_gain, "old_regression": max(0.0, after["nll"] - before["nll"]), "accuracy_gain": after["accuracy"] - before["accuracy"]}
    if direct_gain is not None:
        if float(direct_gain) <= 0.0:
            raise ProtocolError("Direct normalization requires a positive matched Direct gain")
        row["new_gain_normalized"] = raw_gain / float(direct_gain)
    return row


def _direct_trace(base: TinyCLMDecoder, examples: dict[str, list[ScoredTokenExample]], tokenizer: _Tokenizer, device: torch.device, *, seed: int, steps: int, batch_size: int, learning_rate: float, weight_decay: float, certificate_rank: int) -> dict[str, dict[str, Any]]:
    direct_model = deepcopy(base).to(device).eval()
    history_train = list(examples["A_train"])
    history_eval = list(examples["A_eval"])
    trace: dict[str, dict[str, Any]] = {}
    for phase_name in ("B", "C", "D"):
        current_train = examples[f"{phase_name}_train"]
        current_eval = examples[f"{phase_name}_eval"]
        parent = deepcopy(direct_model).to(device).eval()
        certificates = build_activation_certificates(direct_model, history_train, tokenizer, device, batch_size=batch_size, rank=certificate_rank)
        mechanics = train_corrected_direct(direct_model, current_train, tokenizer, device, steps=steps, batch_size=min(batch_size, len(current_train)), seed=seed + ord(phase_name), learning_rate=learning_rate, weight_decay=weight_decay, certificates=certificates)
        old_before = evaluate_model_metrics(parent, history_eval, tokenizer, device, batch_size=8)
        old_after = evaluate_model_metrics(direct_model, history_eval, tokenizer, device, batch_size=8)
        new_before = evaluate_model_metrics(parent, current_eval, tokenizer, device, batch_size=8)
        new_after = evaluate_model_metrics(direct_model, current_eval, tokenizer, device, batch_size=8)
        direct_old = _metric_delta(old_before, old_after)
        direct_new = _metric_delta(new_before, new_after)
        interpolation = []
        for maturity in MATURITY_GRID:
            candidate = interpolate_models(parent, direct_model, maturity).to(device).eval()
            old = _metric_delta(old_before, evaluate_model_metrics(candidate, history_eval, tokenizer, device, batch_size=8))
            new = _metric_delta(new_before, evaluate_model_metrics(candidate, current_eval, tokenizer, device, batch_size=8), direct_new["new_gain"])
            interpolation.append({"maturity": float(maturity), "old_regression": old["old_regression"], "new_gain": new["new_gain"], "new_gain_normalized": new["new_gain_normalized"], "old_nll": old["after_nll"], "new_nll": new["after_nll"], "old_accuracy": old["after_accuracy"], "new_accuracy": new["after_accuracy"]})
        trace[phase_name] = {"mechanics": mechanics, "direct_old": direct_old, "direct_new": direct_new, "new_gain_nll": direct_new["new_gain"], "new_gain_accuracy": direct_new["accuracy_gain"], "interpolation_frontier": interpolation, "parent_model": parent, "direct_model": deepcopy(direct_model).to(device).eval()}
        history_train.extend(current_train)
        history_eval.extend(current_eval)
    return trace


def _matched_direct_step(
    parent: Any,
    history_train: list[ScoredTokenExample],
    history_eval: list[ScoredTokenExample],
    current_train: list[ScoredTokenExample],
    current_eval: list[ScoredTokenExample],
    tokenizer: _Tokenizer,
    device: torch.device,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    certificate_rank: int,
) -> dict[str, Any]:
    """Train Direct from the exact accepted parent used by one Shadow arm."""
    candidate = deepcopy(parent).to(device).eval()
    certificates = build_activation_certificates(
        candidate, history_train, tokenizer, device, batch_size=batch_size, rank=certificate_rank
    )
    mechanics = train_corrected_direct(
        candidate, current_train, tokenizer, device, steps=steps,
        batch_size=min(batch_size, len(current_train)), seed=seed,
        learning_rate=learning_rate, weight_decay=weight_decay,
        certificates=certificates,
    )
    old_before = evaluate_model_metrics(parent, history_eval, tokenizer, device, batch_size=8)
    old_after = evaluate_model_metrics(candidate, history_eval, tokenizer, device, batch_size=8)
    new_before = evaluate_model_metrics(parent, current_eval, tokenizer, device, batch_size=8)
    new_after = evaluate_model_metrics(candidate, current_eval, tokenizer, device, batch_size=8)
    direct_old = _metric_delta(old_before, old_after)
    direct_new = _metric_delta(new_before, new_after)
    interpolation = []
    for maturity in MATURITY_GRID:
        interp = interpolate_models(parent, candidate, maturity).to(device).eval()
        old = _metric_delta(old_before, evaluate_model_metrics(interp, history_eval, tokenizer, device, batch_size=8))
        new = _metric_delta(new_before, evaluate_model_metrics(interp, current_eval, tokenizer, device, batch_size=8), direct_new["new_gain"])
        interpolation.append({"maturity": float(maturity), "old_regression": old["old_regression"], "new_gain": new["new_gain"], "new_gain_normalized": new.get("new_gain_normalized"), "old_nll": old["after_nll"], "new_nll": new["after_nll"], "old_accuracy": old["after_accuracy"], "new_accuracy": new["after_accuracy"], "accuracy_gain": new["accuracy_gain"]})
    return {"mechanics": mechanics, "direct_old": direct_old, "direct_new": direct_new, "new_gain_nll": direct_new["new_gain"], "new_gain_accuracy": direct_new["accuracy_gain"], "interpolation_frontier": interpolation, "parent_model": parent, "direct_model": candidate}


def _shuffled_gate_overrides(sidecar: ShadowSidecar, old_eval: list[ScoredTokenExample], new_eval: list[ScoredTokenExample], tokenizer: _Tokenizer, device: torch.device, seed: int) -> dict[str, float]:
    rows: list[tuple[str, float]] = []
    for values, is_new in ((old_eval, False), (new_eval, True)):
        for start in range(0, len(values), 8):
            batch = values[start:start + 8]
            x, _, _, addresses = collate_scored(batch, pad_id=tokenizer.pad_id, device=device)
            _, gate = sidecar(x, addresses, maturity=0.0, is_new=is_new, return_gate=True)
            rows.extend((address, float(value)) for address, value in zip(addresses, gate.detach().cpu()))
    if len({address for address, _ in rows}) != len(rows):
        raise ProtocolError("shuffled-gate control requires unique evaluation example ids")
    permutation = torch.randperm(len(rows), generator=torch.Generator(device="cpu").manual_seed(int(seed))).tolist()
    values = [value for _, value in rows]
    return {address: values[index] for (address, _), index in zip(rows, permutation)}


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return bool(torch.isfinite(torch.tensor(value)))
    return True


def _published_seed_matches(seed: int, lock_values: dict[str, str]) -> bool:
    path = ROOT / "artifacts/experiments" / VALIDATION_ID / "formal/seeds" / f"seed-{seed}" / "result.json"
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "validation_id": VALIDATION_ID, "phase": "formal", "seed": int(seed),
        "protocol_sha256": lock_values["protocol_sha256"],
        "implementation_manifest_sha256": lock_values["implementation_manifest_sha256"],
        "checkpoint_sha256": lock_values["checkpoint_sha256"],
        "dataset_sha256": lock_values["dataset_sha256"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ProtocolError(f"published seed {seed} exists but does not match the current frozen lock")
    if result.get("status") not in {"COMPLETE", "INCONCLUSIVE_BASE_CAPABILITY", "INCONCLUSIVE_PARENT_CONFLICT", "INCONCLUSIVE_DIRECT_PLASTICITY", "INCONCLUSIVE_GATE_CAPACITY"}:
        raise ProtocolError(f"published seed {seed} is not a completed formal result")
    return True


def run(seed: int, *, phase: str, device_name: str, output: Path, checkpoint: Path | None, dataset: Path | None, steps: int | None = None, kaggle_script_version_id: str | None = None) -> dict[str, Any]:
    protocol = _load_protocol()
    _assert_seed(protocol, phase, seed)
    smoke = phase == "smoke"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not smoke:
        if checkpoint is None or dataset is None:
            raise ProtocolError("formal requires --checkpoint and --dataset; no fallback is allowed")
        lock_values = _assert_formal_lock(protocol, checkpoint, dataset, seed)
    else:
        lock_values = {"protocol_sha256": _protocol_sha(), "checkpoint_sha256": "smoke-only", "dataset_sha256": "smoke-only"}
    torch.manual_seed(int(seed))
    cfg = _config(protocol, smoke=smoke)
    base = TinyCLMDecoder(cfg).to(device).eval() if smoke else _load_model(checkpoint, cfg, device)
    examples, tokenizer = _smoke_examples(cfg, seed) if smoke else _formal_examples(dataset, cfg, seed)
    overlap = _parent_overlap(base, examples)
    thresholds = protocol["thresholds"]
    output_seed = output / f"seed-{seed}"
    output_seed.mkdir(parents=True, exist_ok=True)
    base_A = evaluate_model_metrics(base, examples["A_eval"], tokenizer, device, batch_size=8)
    shared_parent_pass = overlap["same_complete_route_tuple"] and all(
        overlap[f"{key}_fraction"] >= float(thresholds.get("minimum_shared_parent_fraction", 1.0))
        for key in REQUIRED_SPLITS
    )
    result: dict[str, Any] = {
        "validation_id": VALIDATION_ID, "protocol_sha256": lock_values["protocol_sha256"], "implementation_commit": _git_revision(), "runtime_git_commit": _git_revision(), "implementation_manifest_sha256": lock_values.get("implementation_manifest_sha256"), "checkpoint_sha256": lock_values["checkpoint_sha256"], "dataset_sha256": lock_values["dataset_sha256"], "phase": phase, "seed": int(seed), "device": str(device), "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "kaggle_script_version_id": kaggle_script_version_id}, "maturity_grid": list(MATURITY_GRID), "status": "SMOKE_ONLY" if smoke else "PARTIAL_RUN", "scientific_decision": False, "thresholds": {"max_old_regression": float(thresholds["max_old_regression"]), "min_normalized_new_gain": float(thresholds["min_normalized_new_gain"])}, "base_metrics": {"A_eval": base_A}, "parent_overlap": overlap, "arms": {}, "controls": {}, "prerequisites": {}, "validity": {"formal_seed_registered": int(seed) in FORMAL_SEEDS, "protocol_hash_matches_locked_protocol": smoke or lock_values["protocol_sha256"] == _protocol_sha(), "canonical_checkpoint_hash_matches": smoke or lock_values["checkpoint_sha256"] != "smoke-only", "formal_dataset_hash_matches": smoke or lock_values["dataset_sha256"] != "smoke-only", "base_capability_passes": base_A["accuracy"] >= float(thresholds["minimum_A_accuracy"]) and base_A["nll"] <= float(thresholds["maximum_A_nll"]), "same_mature_parent_passes": shared_parent_pass, "accepted_immutable": False, "m0_identity_passes": False, "no_learner_historical_replay": False, "gate_capacity_passes": False, "direct_plasticity_passes": False, "required_arms_completed": False, "required_controls_completed": False, "all_maturity_values_evaluated": False, "finite_results": False, "copy_on_write_artifacts_complete": False, "absolute_retention_passes": False}}
    def write_and_return() -> dict[str, Any]:
        result["validity"]["finite_results"] = _all_finite(result)
        (output_seed / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    if not smoke and not result["validity"]["base_capability_passes"]:
        result["status"] = "INCONCLUSIVE_BASE_CAPABILITY"
        return write_and_return()
    if not smoke and not result["validity"]["same_mature_parent_passes"]:
        result["status"] = "INCONCLUSIVE_PARENT_CONFLICT"
        return write_and_return()
    training = protocol["training"]
    gate_training = protocol["gate_calibration"]
    train_steps = int(steps if steps is not None else (2 if smoke else training["candidate_steps"]))
    direct_trace = _direct_trace(base, examples, tokenizer, device, seed=seed, steps=train_steps, batch_size=int(training["batch_size"]), learning_rate=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]), certificate_rank=int(training["certificate_rank"]))
    result["arms"]["corrected_direct"] = {phase_name: {**trace["mechanics"], "direct_old": trace["direct_old"], "direct_new": trace["direct_new"], "maturity_frontier": [], "selected_maturity": None, "historical_replay_count": 0} for phase_name, trace in direct_trace.items()}
    result["controls"]["direct_interp"] = {phase_name: {"maturity_frontier": trace["interpolation_frontier"]} for phase_name, trace in direct_trace.items()}
    direct_min_nll = float(protocol.get("direct_plasticity", {}).get("minimum_nll_gain", thresholds.get("minimum_direct_nll_gain", 0.1)))
    direct_min_accuracy = float(protocol.get("direct_plasticity", {}).get("minimum_accuracy_gain", thresholds.get("minimum_direct_accuracy_gain", 0.05)))
    for phase_name in ("B", "C", "D"):
        result["prerequisites"][f"direct_{phase_name}_gain_nll"] = float(direct_trace[phase_name]["new_gain_nll"])
        result["prerequisites"][f"direct_{phase_name}_accuracy_gain"] = float(direct_trace[phase_name]["new_gain_accuracy"])
    result["validity"]["direct_plasticity_passes"] = all(
        float(direct_trace[phase_name]["new_gain_nll"]) > direct_min_nll
        and float(direct_trace[phase_name]["new_gain_accuracy"]) >= direct_min_accuracy
        for phase_name in ("B", "C", "D")
    )
    if not smoke and not result["validity"]["direct_plasticity_passes"]:
        result["status"] = "INCONCLUSIVE_DIRECT_PLASTICITY"
        result["validity"]["required_controls_completed"] = True
        return write_and_return()
    gate_probe = ShadowSidecar(base, gate_mode="input_only").to(device)
    gate_info = calibrate_input_gate(gate_probe, examples["A_calibration"], examples["B_calibration"], examples["A_eval"], examples["B_eval"], tokenizer, device, steps=50 if smoke else int(gate_training["steps"]), batch_size=int(gate_training["batch_size"]), seed=seed, learning_rate=float(gate_training["learning_rate"]), weight_decay=float(gate_training["weight_decay"]))
    result["prerequisites"]["gate"] = gate_info
    result["validity"]["gate_capacity_passes"] = float(gate_info["gate_auc"]) >= float(thresholds["minimum_gate_auc"])
    if not smoke and not result["validity"]["gate_capacity_passes"]:
        result["status"] = "INCONCLUSIVE_GATE_CAPACITY"
        result["validity"]["required_controls_completed"] = True
        return write_and_return()
    arm_modes = {"shadow_full": "input_only", "shadow_oracle": "input_only", "shadow_sketch": "input_only", "task_id_shadow": "task_id"}
    for arm, gate_mode in arm_modes.items():
        accepted: TinyCLMDecoder | AcceptedModelChain = deepcopy(base).to(device).eval()
        history_train = list(examples["A_train"])
        history_calibration = list(examples["A_calibration"])
        history_eval = list(examples["A_eval"])
        accepted_stages: dict[str, Any] = {"accepted_A": deepcopy(accepted).to(device).eval()}
        phases: dict[str, dict[str, Any]] = {}
        for phase_name in ("B", "C", "D"):
            current_train, current_eval = examples[f"{phase_name}_train"], examples[f"{phase_name}_eval"]
            task_membership = {item.address_id for item in current_train + current_eval}
            sidecar = ShadowSidecar(accepted, gate_mode=gate_mode, task_id_membership=task_membership).to(device)
            matched_direct = _matched_direct_step(
                accepted, history_train, history_eval, current_train, current_eval,
                tokenizer, device, seed=seed + 1000 + ord(phase_name), steps=train_steps,
                batch_size=int(training["batch_size"]), learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]), certificate_rank=int(training["certificate_rank"]),
            )
            direct_gain = float(matched_direct["new_gain_nll"])
            if direct_gain <= 0.0:
                raise ProtocolError(f"Direct normalization requires positive matched gain in phase {phase_name}")
            if arm == "shadow_oracle":
                result["controls"]["direct_interp"][phase_name] = {"maturity_frontier": matched_direct["interpolation_frontier"]}
            old_calibration = history_calibration
            new_calibration = examples[f"{phase_name}_calibration"]
            gate_info = calibrate_input_gate(sidecar, old_calibration, new_calibration, history_eval, current_eval, tokenizer, device, steps=50 if smoke else int(gate_training["steps"]), batch_size=int(gate_training["batch_size"]), seed=seed + ord(phase_name), learning_rate=float(gate_training["learning_rate"]), weight_decay=float(gate_training["weight_decay"]))
            sketch = build_functional_sketch(sidecar, history_train, tokenizer, device, batch_size=8)
            before = AcceptedModelSnapshot(accepted)
            train = train_shadow(sidecar, current_train, tokenizer, device, steps=train_steps, batch_size=min(int(training["batch_size"]), len(current_train)), seed=seed + ord(phase_name), learning_rate=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
            frontier = evaluate_maturity_frontier(accepted, sidecar, MATURITY_GRID, history_eval, current_eval, gate_mode, tokenizer=tokenizer, device=device, batch_size=8)
            for row in frontier:
                row["new_gain_raw"] = float(row["new_gain"])
                if direct_gain <= 0.0:
                    raise ProtocolError(f"Direct normalization requires positive gain in phase {phase_name}")
                row["new_gain_normalized"] = float(row["new_gain"]) / direct_gain
            selection_frontier = [{**row, "new_gain": row["new_gain_normalized"]} for row in frontier]
            if arm in {"shadow_full", "task_id_shadow"}:
                selected = 1.0
            elif arm == "shadow_oracle":
                selected = select_oracle_maturity(selection_frontier, float(thresholds["max_old_regression"]), float(thresholds["min_normalized_new_gain"]))
            else:
                gains = {float(row["maturity"]): float(row["new_gain_normalized"]) for row in frontier}
                selected = select_sketch_maturity(sidecar, sketch, MATURITY_GRID, gains, max_predicted_damage=float(thresholds["max_old_regression"]), min_new_gain=float(thresholds["min_normalized_new_gain"]))
            selected_row = next((row for row in frontier if row["maturity"] == selected), None)
            before.assert_unchanged(accepted)
            probe_x, _, _, probe_addresses = collate_scored(current_eval[:1], pad_id=tokenizer.pad_id, device=device)
            shuffled = None
            if selected is not None and gate_mode == "input_only" and not smoke:
                shuffle_seed = int.from_bytes(hashlib.sha256(f"{_protocol_sha()}:{seed}:{phase_name}:{arm}".encode()).digest()[:8], "big")
                overrides = _shuffled_gate_overrides(sidecar, history_eval, current_eval, tokenizer, device, shuffle_seed)
                shuffled_frontier = evaluate_maturity_frontier(accepted, sidecar, MATURITY_GRID, history_eval, current_eval, gate_mode, tokenizer=tokenizer, device=device, batch_size=8, gate_overrides=overrides)
                for row in shuffled_frontier:
                    row["new_gain_raw"] = float(row["new_gain"])
                    row["new_gain_normalized"] = float(row["new_gain"]) / direct_gain
                shuffled = {"permutation_seed": shuffle_seed, "maturity_frontier": shuffled_frontier}
            phases[phase_name] = {**train, "gate": gate_info, "selected_maturity": selected, "maturity_frontier": frontier, "shadow_parameter_count": sidecar.shadow_parameter_count, "sketch_size_bytes": sketch.bytes, "sketch_rank": sketch.sketch_rank, "false_safe": bool(arm == "shadow_sketch" and selected_row is not None and selected_row["old_regression"] > float(thresholds["max_old_regression"])), "historical_examples_seen_by_oracle_evaluator": len(history_eval) if arm == "shadow_oracle" else 0, "historical_examples_seen_by_hidden_final_evaluator": len(history_eval), "m0_max_abs_logit_delta": m0_equivalence_delta(sidecar, probe_x, probe_addresses), "routing_preserved": routing_is_preserved(accepted.base if isinstance(accepted, AcceptedModelChain) else accepted, sidecar, [item.address_id for item in current_eval]), "accepted_hash_before_training": train["accepted_hash_before_training"], "accepted_hash_after_training": train["accepted_hash_after_training"], "shuffled_gate": shuffled, "copy_on_write_artifact": None}
            if selected is not None:
                artifact = output_seed / "checkpoints" / f"accepted-{phase_name}-{arm}.pt"
                phases[phase_name]["copy_on_write_artifact"] = copy_on_write_artifact(accepted, sidecar, selected, artifact, phase=phase_name, arm=arm)
                accepted = AcceptedModelChain(accepted) if isinstance(accepted, TinyCLMDecoder) else accepted
                accepted = accepted.append(sidecar, selected)
                accepted_stages[f"accepted_A{'B' if phase_name == 'B' else 'BC' if phase_name == 'C' else 'BCD'}"] = deepcopy(accepted).to(device).eval()
            history_train.extend(current_train)
            history_calibration.extend(examples[f"{phase_name}_calibration"])
            history_eval.extend(current_eval)
        result["arms"][arm] = phases
        result.setdefault("_accepted_stages", {})[arm] = accepted_stages
    result["controls"]["shuffled_gate"] = {arm: {phase: result["arms"][arm][phase]["shuffled_gate"] for phase in ("B", "C", "D")} for arm in ("shadow_full", "shadow_oracle", "shadow_sketch")}
    domain_evals = {domain: examples[f"{domain}_eval"] for domain in ("A", "B", "C", "D")}
    absolute_records: dict[str, Any] = {}
    for arm in arm_modes:
        matrix: dict[str, dict[str, dict[str, float]]] = {}
        for stage, stage_model in result["_accepted_stages"][arm].items():
            matrix[stage] = {domain: evaluate_model_metrics(stage_model, values, tokenizer, device, batch_size=8) for domain, values in domain_evals.items()}
        final = matrix.get("accepted_ABCD", matrix["accepted_A"])
        initial = matrix["accepted_A"]
        final_a_regression = max(0.0, float(final["A"]["nll"]) - float(initial["A"]["nll"]))
        forgetting_values = []
        first_stage = {"A": "accepted_A", "B": "accepted_AB", "C": "accepted_ABC"}
        for stage, domains in matrix.items():
            stage_index = {"accepted_A": 0, "accepted_AB": 1, "accepted_ABC": 2, "accepted_ABCD": 3}[stage]
            for domain in ("A", "B", "C", "D")[:stage_index]:
                forgetting_values.append(max(0.0, float(domains[domain]["nll"]) - float(matrix[first_stage[domain]][domain]["nll"])))
        absolute_records[arm] = {"matrix": matrix, "final_A_absolute_regression": final_a_regression, "mean_forgetting": sum(forgetting_values) / max(1, len(forgetting_values))}
    result["absolute_retention"] = absolute_records
    result.pop("_accepted_stages", None)
    shadow_arms = ("shadow_full", "shadow_oracle", "shadow_sketch", "task_id_shadow")
    rows = [result["arms"][arm][phase] for arm in shadow_arms for phase in ("B", "C", "D")]
    result["validity"]["accepted_immutable"] = all(row["accepted_hash_before_training"] == row["accepted_hash_after_training"] for row in rows)
    result["validity"]["m0_identity_passes"] = all(float(row["m0_max_abs_logit_delta"]) <= float(thresholds["m0_logit_tolerance"]) for row in rows)
    result["validity"]["no_learner_historical_replay"] = all(int(row.get("historical_examples_seen_by_optimizer", 0)) == 0 and int(row.get("historical_examples_seen_by_candidate_trainer", 0)) == 0 for row in rows)
    phase_gate_aucs = [float(result["arms"][arm][phase]["gate"]["gate_auc"]) for arm in shadow_arms for phase in ("B", "C", "D")]
    result["prerequisites"]["gate_auc_by_arm_phase"] = {arm: {phase: result["arms"][arm][phase]["gate"]["gate_auc"] for phase in ("B", "C", "D")} for arm in shadow_arms}
    result["validity"]["gate_capacity_passes"] = all(value >= float(thresholds["minimum_gate_auc"]) for value in phase_gate_aucs)
    result["validity"]["required_arms_completed"] = all(arm in result["arms"] and all([float(item["maturity"]) for item in result["arms"][arm][phase]["maturity_frontier"]] == list(MATURITY_GRID) for phase in ("B", "C", "D")) for arm in shadow_arms)
    result["validity"]["required_controls_completed"] = all(phase in result["controls"]["direct_interp"] for phase in ("B", "C", "D")) and "shuffled_gate" in result["controls"]
    result["validity"]["all_maturity_values_evaluated"] = result["validity"]["required_arms_completed"]
    result["validity"]["copy_on_write_artifacts_complete"] = all(row.get("copy_on_write_artifact") for row in rows if row.get("selected_maturity") is not None)
    result["validity"]["absolute_retention_passes"] = all(
        value["final_A_absolute_regression"] <= float(thresholds["max_old_regression"])
        and value["mean_forgetting"] <= float(thresholds["max_mean_forgetting"])
        for value in result["absolute_retention"].values()
    )
    result["status"] = "SMOKE_ONLY" if smoke else ("COMPLETE" if all(result["validity"].values()) else "INVALID")
    return write_and_return()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", "--output-dir", dest="output", type=Path, default=ROOT / "results" / VALIDATION_ID)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--push-results", action="store_true")
    parser.add_argument("--publish-branch", default="codex/shadow-cell-validation-001-v2-amendment")
    parser.add_argument("--secret-name", default="GITHUB_TOKEN")
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    seed = int(args.seed if args.seed is not None else (DEVELOPMENT_SEED if args.phase == "smoke" else FORMAL_SEEDS[0]))
    protocol = _load_protocol()
    _assert_seed(protocol, args.phase, seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.phase == "formal":
        if args.checkpoint is None or args.dataset is None:
            raise SystemExit("formal requires --checkpoint and --dataset")
        lock_values = _assert_formal_lock(protocol, args.checkpoint, args.dataset, seed)
        if _published_seed_matches(seed, lock_values):
            print(f"[shadow-v2] seed={seed} already completed under matching lock; skipping")
            return 0
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.dataset is not None and not args.dataset.is_file():
        raise SystemExit(f"dataset not found: {args.dataset}")
    if args.preflight_only:
        payload = {"status": "PREFLIGHT_PASS", "scientific_decision": False, "protocol_sha256": _protocol_sha(), "seed": seed, "formal_seed_registered": seed in FORMAL_SEEDS, "checkpoint_sha256": _sha256_file(args.checkpoint) if args.checkpoint else None, "dataset_sha256": _sha256_file(args.dataset) if args.dataset else None, "device": args.device, "synthetic_fallback_allowed": args.phase == "smoke"}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    try:
        result = run(seed, phase=args.phase, device_name=args.device, output=args.output, checkpoint=args.checkpoint, dataset=args.dataset, steps=args.steps, kaggle_script_version_id=args.kaggle_script_version_id)
    except Exception as exc:
        if args.phase == "formal" and args.push_results:
            failure_dir = args.output / f"seed-{seed}"
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure = {"scientific_decision": False, "seed": seed, "validation_id": VALIDATION_ID, "protocol_sha256": lock_values["protocol_sha256"], "implementation_manifest_sha256": lock_values["implementation_manifest_sha256"], "checkpoint_sha256": lock_values["checkpoint_sha256"], "dataset_sha256": lock_values["dataset_sha256"], "failure_type": type(exc).__name__, "message": str(exc), "completed": False}
            (failure_dir / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            publish_results(ROOT, args.output, branch=args.publish_branch, secret_name=args.secret_name, kaggle_script_version_id=args.kaggle_script_version_id)
        raise
    if args.push_results:
        if args.phase != "formal":
            raise SystemExit("--push-results is only allowed for formal runs")
        publish_results(ROOT, args.output, branch=args.publish_branch, secret_name=args.secret_name, kaggle_script_version_id=args.kaggle_script_version_id)
    print("SHADOW_CELL_VALIDATION_001_V2_SMOKE_PASS" if args.phase == "smoke" else json.dumps({"status": result["status"], "seed": seed, "protocol_sha256": result["protocol_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProtocolError, RuntimeError, ValueError, AssertionError, OSError) as exc:
        print(f"FORMAL_RUN_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
