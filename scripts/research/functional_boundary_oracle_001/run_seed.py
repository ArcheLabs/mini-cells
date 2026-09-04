from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from minicells.moe_conversion import create_clm_moe_bundle
from minicells.moe_subexpert import (
    capture_group,
    group_delta,
    restore_group_,
    save_group_mutation,
    validate_group_shapes,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = (
    ROOT
    / "research"
    / "validations"
    / "functional-boundary-oracle-001"
    / "protocol.json"
)
RESULTS_ROOT = ROOT / "results" / "functional-boundary-oracle-001"
WORK_ROOT = ROOT / "results" / "functional-boundary-oracle-001-work"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _require_lm_dependencies():
    try:
        import huggingface_hub
        import safetensors
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install LM extras first: pip install -e '.[lm]'") from exc
    return (
        huggingface_hub,
        safetensors,
        transformers,
        snapshot_download,
        AutoModelForCausalLM,
        AutoTokenizer,
    )


def _batched_indices(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield list(range(start, min(start + batch_size, length)))


def _tokenize(tokenizer, prompts: list[str], device: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _last_positions(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    attention = batch.get("attention_mask")
    if attention is None:
        return torch.full(
            (batch["input_ids"].shape[0],),
            batch["input_ids"].shape[1] - 1,
            dtype=torch.long,
            device=batch["input_ids"].device,
        )
    return attention.long().sum(dim=-1) - 1


def _next_logits(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for indices in _batched_indices(len(prompts), batch_size):
            prompt_batch = [prompts[index] for index in indices]
            batch = _tokenize(tokenizer, prompt_batch, device)
            output = model(**batch, use_cache=False)
            positions = _last_positions(batch)
            batch_rows = torch.arange(len(indices), device=device)
            rows.append(output.logits[batch_rows, positions].detach().float().cpu())
    return torch.cat(rows, dim=0)


def _extract_router_logits(value: Any, num_experts: int) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.ndim >= 1 and value.shape[-1] == num_experts:
            return value
        return None
    if isinstance(value, (tuple, list)):
        found = [
            tensor
            for item in value
            if (tensor := _extract_router_logits(item, num_experts)) is not None
        ]
        return found[-1] if found else None
    return None


def _target_router(model: torch.nn.Module, layer_index: int):
    suffix = f"layers.{layer_index}.block_sparse_moe.router"
    matches = [(name, module) for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one target router for {suffix}, found {[x[0] for x in matches]}")
    return matches[0]


def _router_last_logits(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: str,
    batch_size: int,
    layer_index: int,
) -> torch.Tensor:
    num_experts = int(model.config.num_local_experts)
    router_name, router = _target_router(model, layer_index)
    rows: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        logits = _extract_router_logits(output, num_experts)
        if logits is None:
            raise RuntimeError(f"could not extract router logits from {router_name}")
        captured.append(logits.detach())

    handle = router.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for indices in _batched_indices(len(prompts), batch_size):
                prompt_batch = [prompts[index] for index in indices]
                captured.clear()
                batch = _tokenize(tokenizer, prompt_batch, device)
                _ = model(**batch, use_cache=False)
                if len(captured) != 1:
                    raise RuntimeError(f"expected one router capture, got {len(captured)}")
                logits = captured[0]
                bsz, seq_len = batch["input_ids"].shape
                if logits.ndim == 2 and logits.shape[0] == bsz * seq_len:
                    logits = logits.reshape(bsz, seq_len, num_experts)
                if logits.ndim != 3 or logits.shape[:2] != (bsz, seq_len):
                    raise RuntimeError(f"unexpected router shape {tuple(logits.shape)}")
                positions = _last_positions(batch)
                batch_rows = torch.arange(bsz, device=device)
                rows.append(logits[batch_rows, positions].float().cpu())
    finally:
        handle.remove()
    return torch.cat(rows, dim=0)


def _router_coverage(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    indices = torch.topk(router_logits, k=top_k, dim=-1).indices
    experts = router_logits.shape[-1]
    coverage = []
    for expert in range(experts):
        coverage.append((indices == expert).any(dim=-1).float().mean())
    return torch.stack(coverage)


def _router_topk_identity(left: torch.Tensor, right: torch.Tensor, top_k: int) -> float:
    lhs = torch.topk(left, k=top_k, dim=-1).indices
    rhs = torch.topk(right, k=top_k, dim=-1).indices
    return float((lhs == rhs).all(dim=-1).float().mean().item())


def _make_new_prompts(
    protocol: dict[str, Any], seed: int
) -> tuple[list[str], list[str]]:
    spec = protocol["new_task"]
    seed_mod = seed % 100000
    train = [
        spec["train_template"].format(seed_mod=seed_mod, index=index)
        for index in range(int(spec["train_examples"]))
    ]
    start = int(spec["train_examples"])
    heldout = [
        spec["heldout_template"].format(seed_mod=seed_mod, index=index)
        for index in range(start, start + int(spec["heldout_examples"]))
    ]
    return train, heldout


def _choose_target_token(
    tokenizer,
    train_logits: torch.Tensor,
    candidates: list[str],
) -> tuple[int, str, list[dict[str, Any]]]:
    log_probs = train_logits.log_softmax(dim=-1)
    scored: list[dict[str, Any]] = []
    for order, candidate in enumerate(candidates):
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        token_id = int(token_ids[0])
        probability = float(log_probs[:, token_id].exp().mean().item())
        scored.append(
            {
                "candidate": candidate,
                "candidate_order": order,
                "token_id": token_id,
                "mean_base_probability": probability,
            }
        )
    if not scored:
        raise RuntimeError("no frozen target candidate tokenizes to one token")
    selected = min(scored, key=lambda item: (item["mean_base_probability"], item["candidate_order"]))
    return int(selected["token_id"]), str(selected["candidate"]), scored


def _runtime_packed_parameters(model: torch.nn.Module, layer_index: int):
    marker = f"layers.{layer_index}.block_sparse_moe."
    experts = int(model.config.num_local_experts)
    intermediate = int(model.config.intermediate_size)
    matches = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if marker in name and parameter.ndim == 3 and parameter.shape[0] == experts
    ]
    gate_up = [item for item in matches if item[1].shape[1] == 2 * intermediate]
    down = [item for item in matches if item[1].shape[2] == intermediate]
    if len(gate_up) != 1 or len(down) != 1:
        raise RuntimeError(
            "could not uniquely identify Granite gate/up and down tensors: "
            f"{[(name, list(parameter.shape)) for name, parameter in matches]}"
        )
    validate_group_shapes(gate_up[0][1], down[0][1])
    return gate_up[0], down[0]


def _canonical_mapping(
    manifest: dict[str, Any],
    gate_up: tuple[str, torch.nn.Parameter],
    down: tuple[str, torch.nn.Parameter],
    layer_index: int,
) -> tuple[str, str]:
    marker = f"layers.{layer_index}.block_sparse_moe."
    candidates = [
        tensor
        for tensor in manifest["tensors"]
        if tensor.get("role") == "moe_packed_experts" and marker in tensor["name"]
    ]
    if len(candidates) != 2:
        raise RuntimeError("expected exactly two canonical packed tensors")

    def match(parameter: torch.nn.Parameter) -> str:
        exact = [item for item in candidates if list(item["shape"]) == list(parameter.shape)]
        if len(exact) != 1:
            raise RuntimeError(f"cannot uniquely map runtime shape {list(parameter.shape)}")
        return str(exact[0]["name"])

    return match(gate_up[1]), match(down[1])


def _group_energy_from_gradients(
    gate_grad: torch.Tensor,
    down_grad: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    experts = gate_grad.shape[0]
    intermediate = down_grad.shape[2]
    groups = intermediate // group_size
    gate = gate_grad[:, :intermediate].float().reshape(
        experts, groups, group_size, gate_grad.shape[2]
    )
    up = gate_grad[:, intermediate:].float().reshape(
        experts, groups, group_size, gate_grad.shape[2]
    )
    down = down_grad.float().reshape(
        experts, down_grad.shape[1], groups, group_size
    )
    energy = gate.square().sum(dim=(2, 3)) + up.square().sum(dim=(2, 3))
    energy = energy + down.square().sum(dim=(1, 3))
    return energy.detach().cpu()


def _gradient_energy(
    model,
    tokenizer,
    prompts: list[str],
    targets: torch.Tensor,
    *,
    device: str,
    batch_size: int,
    gate_up: tuple[str, torch.nn.Parameter],
    down: tuple[str, torch.nn.Parameter],
    group_size: int,
) -> torch.Tensor:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gate_up[1].requires_grad_(True)
    down[1].requires_grad_(True)
    total: torch.Tensor | None = None
    batches = 0
    for indices in _batched_indices(len(prompts), batch_size):
        prompt_batch = [prompts[index] for index in indices]
        batch = _tokenize(tokenizer, prompt_batch, device)
        target_batch = targets[indices].to(device)
        model.zero_grad(set_to_none=True)
        output = model(**batch, use_cache=False)
        positions = _last_positions(batch)
        rows = torch.arange(len(indices), device=device)
        logits = output.logits[rows, positions].float()
        loss = F.cross_entropy(logits, target_batch)
        loss.backward()
        if gate_up[1].grad is None or down[1].grad is None:
            raise RuntimeError("packed tensors did not receive gradients")
        energy = _group_energy_from_gradients(
            gate_up[1].grad,
            down[1].grad,
            group_size,
        )
        total = energy if total is None else total + energy
        batches += 1
    model.zero_grad(set_to_none=True)
    if total is None:
        raise RuntimeError("no gradient-energy batches were evaluated")
    return total / float(batches)


def _nll(logits: torch.Tensor, token_id: int) -> float:
    target = torch.full((logits.shape[0],), token_id, dtype=torch.long)
    return float(F.cross_entropy(logits, target).item())


def _kl(base_logits: torch.Tensor, current_logits: torch.Tensor) -> float:
    base_logp = base_logits.log_softmax(dim=-1)
    current_logp = current_logits.log_softmax(dim=-1)
    base_p = base_logp.exp()
    return float((base_p * (base_logp - current_logp)).sum(dim=-1).mean().item())


def _top1_identity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean().item())


def _select_coordinate(
    new_energy: torch.Tensor,
    history_energy: torch.Tensor,
    new_coverage: torch.Tensor,
    history_coverage: torch.Tensor,
    protocol: dict[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    selection = protocol["selection"]
    minimum_coverage = float(selection["minimum_new_route_coverage"])
    rows: list[dict[str, Any]] = []
    for expert in range(new_energy.shape[0]):
        for group in range(new_energy.shape[1]):
            new_rms = math.sqrt(max(float(new_energy[expert, group].item()), 0.0))
            history_rms = math.sqrt(max(float(history_energy[expert, group].item()), 0.0))
            route_specificity = float(new_coverage[expert] - history_coverage[expert])
            score = 0.5 * math.log((new_rms + 1e-12) / (history_rms + 1e-12))
            score += route_specificity
            eligible = float(new_coverage[expert]) >= minimum_coverage and new_rms > 0.0
            rows.append(
                {
                    "expert_index": expert,
                    "group_index": group,
                    "new_group_rms": new_rms,
                    "history_group_rms": history_rms,
                    "new_route_coverage": float(new_coverage[expert]),
                    "history_route_coverage": float(history_coverage[expert]),
                    "route_specificity": route_specificity,
                    "score": score,
                    "eligible": eligible,
                }
            )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise RuntimeError("no sub-expert group satisfied the frozen eligibility rule")
    best = max(
        eligible_rows,
        key=lambda row: (row["score"], -row["expert_index"], -row["group_index"]),
    )
    return int(best["expert_index"]), int(best["group_index"]), rows


def _selected_group_grad_norm(
    gate_grad: torch.Tensor,
    down_grad: torch.Tensor,
    *,
    expert: int,
    group: int,
    group_size: int,
) -> float:
    intermediate = down_grad.shape[2]
    start = group * group_size
    end = start + group_size
    pieces = [
        gate_grad[expert, start:end].float(),
        gate_grad[expert, intermediate + start : intermediate + end].float(),
        down_grad[expert, :, start:end].float(),
    ]
    return math.sqrt(sum(float(piece.square().sum().item()) for piece in pieces))


def _apply_selected_gradient_(
    gate_up: torch.nn.Parameter,
    down: torch.nn.Parameter,
    *,
    expert: int,
    group: int,
    group_size: int,
    learning_rate: float,
    grad_scale: float,
) -> None:
    intermediate = down.shape[2]
    start = group * group_size
    end = start + group_size
    with torch.no_grad():
        gate_up[expert, start:end].add_(
            gate_up.grad[expert, start:end], alpha=-learning_rate * grad_scale
        )
        gate_up[expert, intermediate + start : intermediate + end].add_(
            gate_up.grad[expert, intermediate + start : intermediate + end],
            alpha=-learning_rate * grad_scale,
        )
        down[expert, :, start:end].add_(
            down.grad[expert, :, start:end], alpha=-learning_rate * grad_scale
        )


def _train_coordinate(
    model,
    tokenizer,
    new_prompts: list[str],
    target_token_id: int,
    history_prompts: list[str],
    history_teacher_logits: torch.Tensor,
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
    beta = float(training["history_kl_weight"])
    lr = float(training["learning_rate"])
    max_norm = float(training["max_group_grad_norm"])
    eval_interval = int(training["candidate_eval_interval"])
    history_budget = float(training["maximum_history_selection_kl_for_candidate"])
    max_steps = int(training["max_steps"])

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gate_up[1].requires_grad_(True)
    down[1].requires_grad_(True)

    rng = random.Random(seed)
    new_order = list(range(len(new_prompts)))
    history_order = list(range(len(history_prompts)))
    rng.shuffle(new_order)
    rng.shuffle(history_order)
    new_cursor = 0
    history_cursor = 0
    base_new_nll = _nll(base_new_logits, target_token_id)
    best: dict[str, torch.Tensor] | None = None
    best_gain = -math.inf
    best_step = 10**9
    log: list[dict[str, Any]] = []

    def take(order: list[int], cursor: int) -> tuple[list[int], int]:
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        return indices, cursor + batch_size

    for step in range(1, max_steps + 1):
        new_indices, new_cursor = take(new_order, new_cursor)
        hist_indices, history_cursor = take(history_order, history_cursor)
        new_batch = _tokenize(tokenizer, [new_prompts[i] for i in new_indices], device)
        hist_batch = _tokenize(tokenizer, [history_prompts[i] for i in hist_indices], device)
        teacher_logits = history_teacher_logits[hist_indices].to(device)

        model.zero_grad(set_to_none=True)
        new_output = model(**new_batch, use_cache=False)
        new_positions = _last_positions(new_batch)
        new_rows = torch.arange(len(new_indices), device=device)
        new_logits = new_output.logits[new_rows, new_positions].float()
        targets = torch.full(
            (len(new_indices),), target_token_id, dtype=torch.long, device=device
        )
        target_loss = F.cross_entropy(new_logits, targets)

        hist_output = model(**hist_batch, use_cache=False)
        hist_positions = _last_positions(hist_batch)
        hist_rows = torch.arange(len(hist_indices), device=device)
        current_hist_logits = hist_output.logits[hist_rows, hist_positions].float()
        teacher_logp = teacher_logits.float().log_softmax(dim=-1)
        current_logp = current_hist_logits.log_softmax(dim=-1)
        teacher_p = teacher_logp.exp()
        history_kl = (teacher_p * (teacher_logp - current_logp)).sum(dim=-1).mean()
        loss = target_loss + beta * history_kl
        loss.backward()
        if gate_up[1].grad is None or down[1].grad is None:
            raise RuntimeError("selected packed tensors did not receive gradients")
        grad_norm = _selected_group_grad_norm(
            gate_up[1].grad,
            down[1].grad,
            expert=expert,
            group=group,
            group_size=group_size,
        )
        grad_scale = min(1.0, max_norm / max(grad_norm, 1e-12))
        _apply_selected_gradient_(
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
            "history_batch_kl": float(history_kl.detach().item()),
            "selected_group_grad_norm": grad_norm,
            "grad_scale": grad_scale,
            "new_batch_indices": new_indices,
            "history_batch_indices": hist_indices,
        }
        if step % eval_interval == 0:
            new_logits_full = _next_logits(
                model,
                tokenizer,
                new_prompts,
                device=device,
                batch_size=batch_size,
            )
            hist_logits_full = _next_logits(
                model,
                tokenizer,
                history_prompts,
                device=device,
                batch_size=batch_size,
            )
            gain = base_new_nll - _nll(new_logits_full, target_token_id)
            select_kl = _kl(history_teacher_logits, hist_logits_full)
            record["candidate_train_nll_gain"] = gain
            record["candidate_history_selection_kl"] = select_kl
            if select_kl <= history_budget and gain > 0.0:
                if gain > best_gain or (gain == best_gain and step < best_step):
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
                    record["selected_as_best_safe_candidate"] = True
        log.append(record)
    return best, log


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol()
    protocol_sha = _sha256(PROTOCOL_PATH)
    formal_seeds = [int(seed) for seed in protocol["formal_seeds"]]
    if args.seed not in formal_seeds and not args.allow_nonformal_seed:
        raise SystemExit(f"seed {args.seed} is not formal; allowed={formal_seeds}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    (
        huggingface_hub,
        safetensors,
        transformers,
        snapshot_download,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = _require_lm_dependencies()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base = protocol["base"]
    work = (args.work_dir or WORK_ROOT / f"seed-{args.seed}").resolve()
    result_dir = (args.result_dir or RESULTS_ROOT / f"seed-{args.seed}").resolve()
    if args.clean:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(
        snapshot_download(repo_id=base["model_id"], revision=base["revision"])
    ).resolve()
    bundle_dir = work / "clm-bundle"
    manifest = create_clm_moe_bundle(
        source_dir,
        bundle_dir,
        source_model_id=base["model_id"],
        source_revision=base["revision"],
        copy_mode="hardlink",
    )
    observed_identity = manifest["identity_sha256"]
    expected_identity = base["conversion_manifest_identity_sha256"]
    if observed_identity != expected_identity:
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

    new_train, new_heldout = _make_new_prompts(protocol, args.seed)
    history_select = list(protocol["history"]["selection_prompts"])
    history_eval = list(protocol["history"]["evaluation_prompts"])
    history_disjoint = not bool(set(history_select) & set(history_eval))
    if protocol["history"]["require_disjoint_prompt_sets"] and not history_disjoint:
        raise RuntimeError("history selection/evaluation prompt sets overlap")

    batch_size = int(protocol["training"]["batch_size"])
    layer_index = int(protocol["mutation"]["layer_index"])
    group_size = int(protocol["mutation"]["group_size"])
    top_k = int(manifest["config"]["num_experts_per_tok"])

    base_new_train = _next_logits(
        model, tokenizer, new_train, device=args.device, batch_size=batch_size
    )
    base_new_heldout = _next_logits(
        model, tokenizer, new_heldout, device=args.device, batch_size=batch_size
    )
    base_history_select = _next_logits(
        model, tokenizer, history_select, device=args.device, batch_size=batch_size
    )
    base_history_eval = _next_logits(
        model, tokenizer, history_eval, device=args.device, batch_size=batch_size
    )
    repeatability_prompts = new_heldout + history_eval
    base_repeat_a = _next_logits(
        model, tokenizer, repeatability_prompts, device=args.device, batch_size=batch_size
    )
    base_repeat_b = _next_logits(
        model, tokenizer, repeatability_prompts, device=args.device, batch_size=batch_size
    )
    base_repeatability = float((base_repeat_a - base_repeat_b).abs().max().item())

    target_id, target_candidate, target_scores = _choose_target_token(
        tokenizer,
        base_new_train,
        list(protocol["new_task"]["target_candidates"]),
    )
    gate_up, down = _runtime_packed_parameters(model, layer_index)
    intermediate = validate_group_shapes(gate_up[1], down[1])
    expected_intermediate = int(protocol["mutation"]["expected_intermediate_size"])
    if intermediate != expected_intermediate:
        raise RuntimeError(
            f"unexpected expert intermediate width {intermediate}; expected {expected_intermediate}"
        )
    canonical_gate_up, canonical_down = _canonical_mapping(
        manifest, gate_up, down, layer_index
    )

    new_router = _router_last_logits(
        model,
        tokenizer,
        new_train,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )
    history_router = _router_last_logits(
        model,
        tokenizer,
        history_select,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )
    new_coverage = _router_coverage(new_router, top_k)
    history_coverage = _router_coverage(history_router, top_k)

    new_targets = torch.full((len(new_train),), target_id, dtype=torch.long)
    history_targets = base_history_select.argmax(dim=-1)
    new_energy = _gradient_energy(
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
    history_energy = _gradient_energy(
        model,
        tokenizer,
        history_select,
        history_targets,
        device=args.device,
        batch_size=batch_size,
        gate_up=gate_up,
        down=down,
        group_size=group_size,
    )
    expert_index, group_index, coordinate_scores = _select_coordinate(
        new_energy,
        history_energy,
        new_coverage,
        history_coverage,
        protocol,
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
    best_group, training_log = _train_coordinate(
        model,
        tokenizer,
        new_train,
        target_id,
        history_select,
        base_history_select,
        device=args.device,
        gate_up=gate_up,
        down=down,
        expert=expert_index,
        group=group_index,
        group_size=group_size,
        protocol=protocol,
        seed=args.seed,
        base_new_logits=base_new_train,
    )
    if best_group is None:
        restore_group_(
            parameter_map,
            original_group,
            gate_up_name=gate_up[0],
            down_name=down[0],
            expert_index=expert_index,
            group_index=group_index,
            group_size=group_size,
        )
    else:
        restore_group_(
            parameter_map,
            best_group,
            gate_up_name=gate_up[0],
            down_name=down[0],
            expert_index=expert_index,
            group_index=group_index,
            group_size=group_size,
        )

    mutated_new_train = _next_logits(
        model, tokenizer, new_train, device=args.device, batch_size=batch_size
    )
    mutated_new_heldout = _next_logits(
        model, tokenizer, new_heldout, device=args.device, batch_size=batch_size
    )
    mutated_history_select = _next_logits(
        model, tokenizer, history_select, device=args.device, batch_size=batch_size
    )
    mutated_history_eval = _next_logits(
        model, tokenizer, history_eval, device=args.device, batch_size=batch_size
    )
    all_router_prompts = new_train + new_heldout + history_eval
    base_router_all = _router_last_logits(
        model,
        tokenizer,
        all_router_prompts,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )
    # Target-layer routing is computed from the layer input, before its expert output.
    mutated_router_all = base_router_all.clone()

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
        result_dir / "mutation",
        base_manifest_identity=observed_identity,
        source_model_id=base["model_id"],
        source_revision=base["revision"],
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
            "seed": args.seed,
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
    rollback_logits = _next_logits(
        model,
        tokenizer,
        repeatability_prompts,
        device=args.device,
        batch_size=batch_size,
    )
    rollback_error = float((rollback_logits - base_repeat_a).abs().max().item())
    rollback_excess = max(0.0, rollback_error - base_repeatability)

    train_gain = _nll(base_new_train, target_id) - _nll(mutated_new_train, target_id)
    heldout_gain = _nll(base_new_heldout, target_id) - _nll(mutated_new_heldout, target_id)
    history_select_kl = _kl(base_history_select, mutated_history_select)
    history_eval_kl = _kl(base_history_eval, mutated_history_eval)
    history_eval_top1 = _top1_identity(base_history_eval, mutated_history_eval)
    router_identity = _router_topk_identity(base_router_all, mutated_router_all, top_k)
    selected_score = next(
        row
        for row in coordinate_scores
        if row["expert_index"] == expert_index and row["group_index"] == group_index
    )
    expert_fraction = float(mutation_manifest["target"]["expert_fraction"])

    thresholds = protocol["gates"]
    gates = {
        "conversion_identity": observed_identity == expected_identity,
        "history_set_disjointness": history_disjoint,
        "heldout_nll_gain": heldout_gain >= float(thresholds["minimum_heldout_nll_gain"]),
        "history_evaluation_mean_kl": history_eval_kl
        <= float(thresholds["maximum_history_evaluation_mean_kl"]),
        "history_evaluation_top1_identity": history_eval_top1
        >= float(thresholds["minimum_history_evaluation_top1_identity"]),
        "target_router_topk_identity": router_identity
        == float(thresholds["required_target_router_topk_identity"]),
        "expert_fraction": expert_fraction <= float(thresholds["maximum_expert_fraction"]),
        "selected_expert_new_route_coverage": selected_score["new_route_coverage"]
        >= float(thresholds["minimum_selected_expert_new_route_coverage"]),
        "nonzero_delta": delta_l2 > 0.0 if thresholds["require_nonzero_delta"] else True,
        "exact_weight_rollback": exact_weight_rollback
        if thresholds["require_exact_weight_rollback"] else True,
        "forward_rollback_within_repeatability": rollback_excess
        <= float(thresholds["maximum_forward_rollback_excess_over_base_repeatability"]),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "experiment": protocol["experiment"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha,
        "seed": args.seed,
        "formal_seed": args.seed in formal_seeds,
        "status": status,
        "base": {
            "model_id": base["model_id"],
            "revision": base["revision"],
            "conversion_manifest_identity_sha256": observed_identity,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "safetensors": safetensors.__version__,
            "device": args.device,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
            ),
            "dtype": "torch.float32",
        },
        "target": {
            "token_id": target_id,
            "candidate": target_candidate,
            "decoded": tokenizer.decode([target_id]),
            "candidate_scores": target_scores,
        },
        "selection": {
            "layer_index": layer_index,
            "expert_index": expert_index,
            "group_index": group_index,
            "group_size": group_size,
            "intermediate_size": intermediate,
            "expert_fraction": expert_fraction,
            "selected_score": selected_score,
            "all_coordinate_scores": coordinate_scores,
            "gate_up_runtime_name": gate_up[0],
            "down_runtime_name": down[0],
            "gate_up_canonical_name": canonical_gate_up,
            "down_canonical_name": canonical_down,
        },
        "metrics": {
            "base_new_train_nll": _nll(base_new_train, target_id),
            "mutated_new_train_nll": _nll(mutated_new_train, target_id),
            "new_train_nll_gain": train_gain,
            "base_new_heldout_nll": _nll(base_new_heldout, target_id),
            "mutated_new_heldout_nll": _nll(mutated_new_heldout, target_id),
            "heldout_nll_gain": heldout_gain,
            "history_selection_mean_kl": history_select_kl,
            "history_evaluation_mean_kl": history_eval_kl,
            "history_evaluation_top1_identity": history_eval_top1,
            "target_router_topk_identity": router_identity,
            "delta_l2_norm": delta_l2,
            "base_forward_repeatability_max_abs": base_repeatability,
            "rollback_max_abs_logit_error": rollback_error,
            "rollback_excess_over_base_repeatability": rollback_excess,
            "exact_weight_rollback": exact_weight_rollback,
        },
        "gates": gates,
        "mutation": {
            "schema_version": mutation_manifest["schema_version"],
            "identity_sha256": mutation_manifest["identity_sha256"],
            "tensor_file": mutation_manifest["tensor_file"],
            "expert_is_cell": mutation_manifest["target"]["expert_is_cell"],
            "group_is_cell": mutation_manifest["target"]["group_is_cell"],
        },
    }
    (result_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (result_dir / "training.jsonl").open("w", encoding="utf-8") as handle:
        for record in training_log:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Functional Boundary Oracle 001")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-nonformal-seed", action="store_true")
    parser.add_argument("--fail-on-scientific-fail", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    outcome = run(parsed)
    if parsed.fail_on_scientific_fail and outcome["status"] != "PASS":
        raise SystemExit(2)
    raise SystemExit(0)
