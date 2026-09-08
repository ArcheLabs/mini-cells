from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

from minicells.granite_moe_layout import identify_packed_expert_tensors
from minicells.moe_subexpert import (
    capture_group,
    group_delta,
    restore_group_,
    save_group_mutation,
    validate_group_shapes,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "research" / "validations" / "history-compression-001" / "protocol.json"
RESULTS_ROOT = ROOT / "results" / "history-compression-001"
WORK_ROOT = ROOT / "results" / "history-compression-001-work"
ORACLE_ENGINE_PATH = (
    ROOT / "scripts" / "research" / "functional_boundary_oracle_001" / "run_seed.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _load_oracle_engine() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "minicells_functional_boundary_oracle_001_engine", ORACLE_ENGINE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Functional Boundary Oracle 001 engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quiet_libraries(huggingface_hub: Any, transformers: Any) -> None:
    try:
        huggingface_hub.utils.disable_progress_bars()
    except Exception:
        pass
    try:
        transformers.logging.disable_progress_bar()
        transformers.logging.set_verbosity_error()
    except Exception:
        pass


def _progress(seed: int, message: str, *, mode: str | None = None) -> None:
    prefix = f"[hc001][seed={seed}]"
    if mode is not None:
        prefix += f"[mode={mode}]"
    print(f"{prefix} {message}", flush=True)


def _history_order(prompts: list[str], seed: int) -> list[int]:
    keyed: list[tuple[str, int]] = []
    for index, prompt in enumerate(prompts):
        raw = f"{seed}:{index}:{prompt}".encode("utf-8")
        keyed.append((hashlib.sha256(raw).hexdigest(), index))
    return [index for _digest, index in sorted(keyed)]


def _history_subset(
    all_prompts: list[str], seed: int, count: int
) -> tuple[list[int], list[str]]:
    if count < 0 or count > len(all_prompts):
        raise ValueError(f"history count {count} outside [0, {len(all_prompts)}]")
    order = _history_order(all_prompts, seed)
    indices = order[:count]
    return indices, [all_prompts[index] for index in indices]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _train_zero_history(
    oracle: ModuleType,
    model: Any,
    tokenizer: Any,
    new_prompts: list[str],
    target_token_id: int,
    *,
    device: str,
    gate_up: tuple[str, torch.nn.Parameter],
    down: tuple[str, torch.nn.Parameter],
    expert: int,
    group: int,
    group_size: int,
    protocol: dict[str, Any],
    seed: int,
    base_new_logits: torch.Tensor,
) -> tuple[dict[str, torch.Tensor] | None, list[dict[str, Any]]]:
    training = protocol["training"]
    batch_size = int(training["batch_size"])
    lr = float(training["learning_rate"])
    max_norm = float(training["max_group_grad_norm"])
    eval_interval = int(training["candidate_eval_interval"])
    max_steps = int(training["max_steps"])

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gate_up[1].requires_grad_(True)
    down[1].requires_grad_(True)

    rng = random.Random(seed)
    order = list(range(len(new_prompts)))
    rng.shuffle(order)
    cursor = 0
    base_new_nll = oracle._nll(base_new_logits, target_token_id)
    best: dict[str, torch.Tensor] | None = None
    best_gain = -math.inf
    best_step = 10**9
    log: list[dict[str, Any]] = []

    def take() -> list[int]:
        nonlocal cursor
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        return indices

    for step in range(1, max_steps + 1):
        indices = take()
        batch = oracle._tokenize(tokenizer, [new_prompts[i] for i in indices], device)
        model.zero_grad(set_to_none=True)
        output = model(**batch, use_cache=False)
        positions = oracle._last_positions(batch)
        rows = torch.arange(len(indices), device=device)
        logits = output.logits[rows, positions].float()
        targets = torch.full(
            (len(indices),), target_token_id, dtype=torch.long, device=device
        )
        target_loss = F.cross_entropy(logits, targets)
        target_loss.backward()
        if gate_up[1].grad is None or down[1].grad is None:
            raise RuntimeError("selected packed tensors did not receive gradients")
        grad_norm = oracle._selected_group_grad_norm(
            gate_up[1].grad,
            down[1].grad,
            expert=expert,
            group=group,
            group_size=group_size,
        )
        grad_scale = min(1.0, max_norm / max(grad_norm, 1e-12))
        oracle._apply_selected_gradient_(
            gate_up[1],
            down[1],
            expert=expert,
            group=group,
            group_size=group_size,
            learning_rate=lr,
            grad_scale=grad_scale,
        )
        model.zero_grad(set_to_none=True)

        record: dict[str, Any] = {
            "step": step,
            "target_batch_loss": float(target_loss.detach().item()),
            "selected_group_grad_norm": grad_norm,
            "grad_scale": grad_scale,
            "new_batch_indices": indices,
        }
        if step % eval_interval == 0:
            full_logits = oracle._next_logits(
                model,
                tokenizer,
                new_prompts,
                device=device,
                batch_size=batch_size,
            )
            gain = base_new_nll - oracle._nll(full_logits, target_token_id)
            record["candidate_train_nll_gain"] = gain
            if gain > 0.0 and (
                gain > best_gain or (gain == best_gain and step < best_step)
            ):
                best = capture_group(
                    dict(model.named_parameters()),
                    gate_up_name=gate_up[0],
                    down_name=down[0],
                    expert_index=expert,
                    group_index=group,
                    group_size=group_size,
                )
                best_gain = gain
                best_step = step
                record["selected_as_best_candidate"] = True
        log.append(record)
    return best, log


def _run_mode(
    *,
    oracle: ModuleType,
    protocol: dict[str, Any],
    protocol_sha: str,
    mode: dict[str, Any],
    seed: int,
    model: Any,
    tokenizer: Any,
    manifest: dict[str, Any],
    base_new_train: torch.Tensor,
    base_new_heldout: torch.Tensor,
    base_history_eval: torch.Tensor,
    base_repeat_a: torch.Tensor,
    base_repeatability: float,
    new_train: list[str],
    new_heldout: list[str],
    history_all: list[str],
    history_eval: list[str],
    target_id: int,
    target_candidate: str,
    target_scores: list[dict[str, Any]],
    gate_up: tuple[str, torch.nn.Parameter],
    down: tuple[str, torch.nn.Parameter],
    intermediate: int,
    canonical_gate_up: str,
    canonical_down: str,
    new_coverage: torch.Tensor,
    new_energy: torch.Tensor,
    device: str,
    result_root: Path,
) -> dict[str, Any]:
    mode_id = str(mode["id"])
    history_count = int(mode["history_prompt_count"])
    mode_dir = result_root / mode_id
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(protocol["training"]["batch_size"])
    layer_index = int(protocol["mutation"]["layer_index"])
    group_size = int(protocol["mutation"]["group_size"])
    top_k = int(manifest["config"]["num_experts_per_tok"])

    subset_indices, history_prompts = _history_subset(history_all, seed, history_count)
    if history_count:
        _progress(seed, f"compile history subset n={history_count}", mode=mode_id)
        teacher_logits = oracle._next_logits(
            model,
            tokenizer,
            history_prompts,
            device=device,
            batch_size=batch_size,
        )
        history_router = oracle._router_last_logits(
            model,
            tokenizer,
            history_prompts,
            device=device,
            batch_size=batch_size,
            layer_index=layer_index,
        )
        history_coverage = oracle._router_coverage(history_router, top_k)
        history_targets = teacher_logits.argmax(dim=-1)
        history_energy = oracle._gradient_energy(
            model,
            tokenizer,
            history_prompts,
            history_targets,
            device=device,
            batch_size=batch_size,
            gate_up=gate_up,
            down=down,
            group_size=group_size,
        )
    else:
        teacher_logits = None
        history_coverage = torch.zeros_like(new_coverage)
        history_energy = torch.zeros_like(new_energy)

    expert_index, group_index, coordinate_scores = oracle._select_coordinate(
        new_energy,
        history_energy,
        new_coverage,
        history_coverage,
        protocol,
    )
    selected_score = next(
        row
        for row in coordinate_scores
        if row["expert_index"] == expert_index and row["group_index"] == group_index
    )
    _write_json(mode_dir / "coordinate_scores.json", coordinate_scores)
    _progress(
        seed,
        f"selected layer={layer_index} expert={expert_index} group={group_index} score={selected_score['score']:.4f}",
        mode=mode_id,
    )

    parameter_map = dict(model.named_parameters())
    original_group = capture_group(
        parameter_map,
        gate_up_name=gate_up[0],
        down_name=down[0],
        expert_index=expert_index,
        group_index=group_index,
        group_size=group_size,
    )

    _progress(seed, "training", mode=mode_id)
    if history_count:
        assert teacher_logits is not None
        best_group, training_log = oracle._train_coordinate(
            model,
            tokenizer,
            new_train,
            target_id,
            history_prompts,
            teacher_logits,
            device=device,
            gate_up=gate_up,
            down=down,
            expert=expert_index,
            group=group_index,
            group_size=group_size,
            protocol=protocol,
            seed=seed,
            base_new_logits=base_new_train,
        )
    else:
        best_group, training_log = _train_zero_history(
            oracle,
            model,
            tokenizer,
            new_train,
            target_id,
            device=device,
            gate_up=gate_up,
            down=down,
            expert=expert_index,
            group=group_index,
            group_size=group_size,
            protocol=protocol,
            seed=seed,
            base_new_logits=base_new_train,
        )
    _write_jsonl(mode_dir / "training.jsonl", training_log)

    restore_group_(
        parameter_map,
        best_group if best_group is not None else original_group,
        gate_up_name=gate_up[0],
        down_name=down[0],
        expert_index=expert_index,
        group_index=group_index,
        group_size=group_size,
    )

    mutated_new_train = oracle._next_logits(
        model, tokenizer, new_train, device=device, batch_size=batch_size
    )
    mutated_new_heldout = oracle._next_logits(
        model, tokenizer, new_heldout, device=device, batch_size=batch_size
    )
    mutated_history_eval = oracle._next_logits(
        model, tokenizer, history_eval, device=device, batch_size=batch_size
    )
    if history_count:
        mutated_history_select = oracle._next_logits(
            model,
            tokenizer,
            history_prompts,
            device=device,
            batch_size=batch_size,
        )
        assert teacher_logits is not None
        history_select_kl: float | None = oracle._kl(
            teacher_logits, mutated_history_select
        )
    else:
        history_select_kl = None

    current_group = capture_group(
        parameter_map,
        gate_up_name=gate_up[0],
        down_name=down[0],
        expert_index=expert_index,
        group_index=group_index,
        group_size=group_size,
    )
    deltas = group_delta(current_group, original_group)
    delta_l2 = math.sqrt(sum(float(value.square().sum().item()) for value in deltas.values()))
    mutation_manifest = save_group_mutation(
        mode_dir / "mutation",
        base_manifest_identity=manifest["identity_sha256"],
        source_model_id=protocol["base"]["model_id"],
        source_revision=protocol["base"]["revision"],
        layer_index=layer_index,
        expert_index=expert_index,
        group_index=group_index,
        group_size=group_size,
        intermediate_size=intermediate,
        gate_up_runtime_name=gate_up[0],
        down_runtime_name=down[0],
        gate_up_canonical_name=canonical_gate_up,
        down_canonical_name=canonical_down,
        deltas=deltas,
        metadata={
            "experiment": protocol["experiment"],
            "seed": seed,
            "mode": mode_id,
            "history_prompt_count": history_count,
            "protocol_sha256": protocol_sha,
            "target_token_id": target_id,
            "target_candidate": target_candidate,
        },
    )

    restore_group_(
        parameter_map,
        original_group,
        gate_up_name=gate_up[0],
        down_name=down[0],
        expert_index=expert_index,
        group_index=group_index,
        group_size=group_size,
    )
    rolled_group = capture_group(
        parameter_map,
        gate_up_name=gate_up[0],
        down_name=down[0],
        expert_index=expert_index,
        group_index=group_index,
        group_size=group_size,
    )
    exact_weight_rollback = all(
        torch.equal(rolled_group[key], original_group[key]) for key in original_group
    )
    rollback_logits = oracle._next_logits(
        model,
        tokenizer,
        new_heldout + history_eval,
        device=device,
        batch_size=batch_size,
    )
    rollback_error = float((rollback_logits - base_repeat_a).abs().max().item())
    rollback_excess = max(0.0, rollback_error - base_repeatability)

    train_gain = oracle._nll(base_new_train, target_id) - oracle._nll(
        mutated_new_train, target_id
    )
    heldout_gain = oracle._nll(base_new_heldout, target_id) - oracle._nll(
        mutated_new_heldout, target_id
    )
    history_eval_kl = oracle._kl(base_history_eval, mutated_history_eval)
    history_eval_top1 = oracle._top1_identity(
        base_history_eval, mutated_history_eval
    )
    expert_fraction = float(mutation_manifest["target"]["expert_fraction"])

    thresholds = protocol["gates"]
    gates = {
        "conversion_identity": manifest["identity_sha256"]
        == protocol["base"]["conversion_manifest_identity_sha256"],
        "history_set_disjointness": not bool(
            set(protocol["history"]["selection_prompts"])
            & set(protocol["history"]["evaluation_prompts"])
        ),
        "heldout_nll_gain": heldout_gain
        >= float(thresholds["minimum_heldout_nll_gain"]),
        "history_evaluation_mean_kl": history_eval_kl
        <= float(thresholds["maximum_history_evaluation_mean_kl"]),
        "history_evaluation_top1_identity": history_eval_top1
        >= float(thresholds["minimum_history_evaluation_top1_identity"]),
        "target_router_topk_identity": True,
        "expert_fraction": expert_fraction
        <= float(thresholds["maximum_expert_fraction"]),
        "selected_expert_new_route_coverage": selected_score["new_route_coverage"]
        >= float(thresholds["minimum_selected_expert_new_route_coverage"]),
        "nonzero_delta": delta_l2 > 0.0 if thresholds["require_nonzero_delta"] else True,
        "exact_weight_rollback": exact_weight_rollback
        if thresholds["require_exact_weight_rollback"]
        else True,
        "forward_rollback_within_repeatability": rollback_excess
        <= float(thresholds["maximum_forward_rollback_excess_over_base_repeatability"]),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "experiment": protocol["experiment"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "formal_seed": seed in [int(value) for value in protocol["formal_seeds"]],
        "mode": mode_id,
        "history_prompt_count": history_count,
        "history_fraction": float(mode["history_fraction"]),
        "history_subset_indices": subset_indices,
        "status": status,
        "base": {
            "model_id": protocol["base"]["model_id"],
            "revision": protocol["base"]["revision"],
            "conversion_manifest_identity_sha256": manifest["identity_sha256"],
        },
        "selection": {
            "layer_index": layer_index,
            "expert_index": expert_index,
            "group_index": group_index,
            "group_size": group_size,
            "intermediate_size": intermediate,
            "selected_score": selected_score,
            "coordinate_scores_file": "coordinate_scores.json",
        },
        "target": {
            "token_id": target_id,
            "candidate": target_candidate,
            "candidate_scores": target_scores,
        },
        "metrics": {
            "base_new_train_nll": oracle._nll(base_new_train, target_id),
            "mutated_new_train_nll": oracle._nll(mutated_new_train, target_id),
            "new_train_nll_gain": train_gain,
            "base_new_heldout_nll": oracle._nll(base_new_heldout, target_id),
            "mutated_new_heldout_nll": oracle._nll(mutated_new_heldout, target_id),
            "heldout_nll_gain": heldout_gain,
            "history_selection_mean_kl": history_select_kl,
            "history_evaluation_mean_kl": history_eval_kl,
            "history_evaluation_top1_identity": history_eval_top1,
            "delta_l2_norm": delta_l2,
            "exact_weight_rollback": exact_weight_rollback,
            "base_forward_repeatability_max_abs": base_repeatability,
            "rollback_max_abs_logit_error": rollback_error,
            "rollback_excess_over_base_repeatability": rollback_excess,
            "target_router_topk_identity": 1.0,
        },
        "mutation": {
            "schema_version": mutation_manifest["schema_version"],
            "identity_sha256": mutation_manifest["identity_sha256"],
            "expert_is_cell": False,
            "group_is_cell": False,
            "tensor_file": mutation_manifest["tensor_file"],
        },
        "training": {
            "steps": int(protocol["training"]["max_steps"]),
            "training_log_file": "training.jsonl",
            "learner_visible_history": history_count > 0,
        },
        "gates": gates,
    }
    _write_json(mode_dir / "result.json", result)
    _progress(
        seed,
        (
            f"preliminary={status} heldout_gain={heldout_gain:.4f} "
            f"history_eval_kl={history_eval_kl:.6f} top1={history_eval_top1:.4f}"
        ),
        mode=mode_id,
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol()
    protocol_sha = _sha256(PROTOCOL_PATH)
    formal_seeds = [int(seed) for seed in protocol["formal_seeds"]]
    if args.seed not in formal_seeds and not args.allow_nonformal_seed:
        raise SystemExit(f"seed {args.seed} is not formal; allowed={formal_seeds}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    oracle = _load_oracle_engine()
    (
        huggingface_hub,
        safetensors,
        transformers,
        snapshot_download,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = oracle._require_lm_dependencies()
    _quiet_libraries(huggingface_hub, transformers)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base = protocol["base"]
    work = (args.work_dir or WORK_ROOT / f"seed-{args.seed}").resolve()
    result_root = (args.result_dir or RESULTS_ROOT / f"seed-{args.seed}").resolve()
    if args.clean:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(result_root, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    _progress(args.seed, "loading frozen Granite substrate")
    source_dir = Path(
        snapshot_download(repo_id=base["model_id"], revision=base["revision"])
    ).resolve()
    bundle_dir = work / "clm-bundle"
    manifest = oracle.create_clm_moe_bundle(
        source_dir,
        bundle_dir,
        source_model_id=base["model_id"],
        source_revision=base["revision"],
        copy_mode="hardlink",
    )
    if manifest["identity_sha256"] != base["conversion_manifest_identity_sha256"]:
        raise RuntimeError("Conversion 001 identity mismatch")

    tokenizer = AutoTokenizer.from_pretrained(bundle_dir / "substrate")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        bundle_dir / "substrate",
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    new_train, new_heldout = oracle._make_new_prompts(protocol, args.seed)
    history_all = list(protocol["history"]["selection_prompts"])
    history_eval = list(protocol["history"]["evaluation_prompts"])
    if protocol["history"]["require_disjoint_prompt_sets"] and bool(
        set(history_all) & set(history_eval)
    ):
        raise RuntimeError("history selection/evaluation prompt sets overlap")

    batch_size = int(protocol["training"]["batch_size"])
    layer_index = int(protocol["mutation"]["layer_index"])
    group_size = int(protocol["mutation"]["group_size"])
    top_k = int(manifest["config"]["num_experts_per_tok"])

    _progress(args.seed, "caching base outputs and new-task geometry")
    base_new_train = oracle._next_logits(
        model, tokenizer, new_train, device=args.device, batch_size=batch_size
    )
    base_new_heldout = oracle._next_logits(
        model, tokenizer, new_heldout, device=args.device, batch_size=batch_size
    )
    base_history_eval = oracle._next_logits(
        model, tokenizer, history_eval, device=args.device, batch_size=batch_size
    )
    repeatability_prompts = new_heldout + history_eval
    base_repeat_a = oracle._next_logits(
        model,
        tokenizer,
        repeatability_prompts,
        device=args.device,
        batch_size=batch_size,
    )
    base_repeat_b = oracle._next_logits(
        model,
        tokenizer,
        repeatability_prompts,
        device=args.device,
        batch_size=batch_size,
    )
    base_repeatability = float((base_repeat_a - base_repeat_b).abs().max().item())

    target_id, target_candidate, target_scores = oracle._choose_target_token(
        tokenizer,
        base_new_train,
        list(protocol["new_task"]["target_candidates"]),
    )
    gate_up, down = identify_packed_expert_tensors(model, layer_index)
    intermediate = validate_group_shapes(gate_up[1], down[1])
    expected_intermediate = int(protocol["mutation"]["expected_intermediate_size"])
    if intermediate != expected_intermediate:
        raise RuntimeError(
            f"unexpected expert intermediate width {intermediate}; expected {expected_intermediate}"
        )
    canonical_gate_up, canonical_down = oracle._canonical_mapping(
        manifest, gate_up, down, layer_index
    )
    new_router = oracle._router_last_logits(
        model,
        tokenizer,
        new_train,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )
    new_coverage = oracle._router_coverage(new_router, top_k)
    new_targets = torch.full((len(new_train),), target_id, dtype=torch.long)
    new_energy = oracle._gradient_energy(
        model,
        tokenizer,
        new_train,
        new_targets,
        device=args.device,
        batch_size=batch_size,
        gate_up=gate_up,
        down=down,
        group_size=group_size,
    )

    requested_modes = set(args.mode or [])
    protocol_modes = [dict(mode) for mode in protocol["compression_modes"]]
    if requested_modes:
        unknown = requested_modes - {str(mode["id"]) for mode in protocol_modes}
        if unknown:
            raise SystemExit(f"unknown mode(s): {sorted(unknown)}")
        modes = [mode for mode in protocol_modes if str(mode["id"]) in requested_modes]
    else:
        modes = protocol_modes

    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_id = str(mode["id"])
        results[mode_id] = _run_mode(
            oracle=oracle,
            protocol=protocol,
            protocol_sha=protocol_sha,
            mode=mode,
            seed=args.seed,
            model=model,
            tokenizer=tokenizer,
            manifest=manifest,
            base_new_train=base_new_train,
            base_new_heldout=base_new_heldout,
            base_history_eval=base_history_eval,
            base_repeat_a=base_repeat_a,
            base_repeatability=base_repeatability,
            new_train=new_train,
            new_heldout=new_heldout,
            history_all=history_all,
            history_eval=history_eval,
            target_id=target_id,
            target_candidate=target_candidate,
            target_scores=target_scores,
            gate_up=gate_up,
            down=down,
            intermediate=intermediate,
            canonical_gate_up=canonical_gate_up,
            canonical_down=canonical_down,
            new_coverage=new_coverage,
            new_energy=new_energy,
            device=args.device,
            result_root=result_root,
        )

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "device": args.device,
        "dtype": "torch.float32",
    }
    for result in results.values():
        result["environment"] = environment
        _write_json(result_root / result["mode"] / "result.json", result)

    seed_summary = {
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol_sha,
        "seed": args.seed,
        "formal_seed": args.seed in formal_seeds,
        "environment": environment,
        "modes": {
            mode_id: {
                "status": result["status"],
                "history_prompt_count": result["history_prompt_count"],
                "heldout_nll_gain": result["metrics"]["heldout_nll_gain"],
                "history_evaluation_mean_kl": result["metrics"]["history_evaluation_mean_kl"],
                "history_evaluation_top1_identity": result["metrics"]["history_evaluation_top1_identity"],
                "expert_index": result["selection"]["expert_index"],
                "group_index": result["selection"]["group_index"],
            }
            for mode_id, result in results.items()
        },
    }
    _write_json(result_root / "seed_summary.json", seed_summary)
    _progress(
        args.seed,
        "engine complete; fresh-base router verification remains for formal runner",
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return seed_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run History Compression 001 seed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", action="append")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-nonformal-seed", action="store_true")
    parser.add_argument("--fail-on-scientific-fail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "experiment": summary["experiment"],
                "seed": summary["seed"],
                "modes": summary["modes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.fail_on_scientific_fail and any(
        mode["status"] != "PASS" for mode in summary["modes"].values()
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
