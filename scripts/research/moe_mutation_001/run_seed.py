#!/usr/bin/env python3
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
from minicells.moe_mutation import (
    capture_expert_slices,
    restore_expert_slices_,
    save_expert_slice_mutation,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "research" / "validations" / "moe-mutation-001" / "protocol.json"
RESULTS_ROOT = ROOT / "results" / "moe-mutation-001"
WORK_ROOT = ROOT / "results" / "moe-mutation-001-work"


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


def _extract_router_logits(value: Any, num_experts: int) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.ndim >= 1 and value.shape[-1] == num_experts:
            return value
        return None
    if isinstance(value, (tuple, list)):
        candidates = [
            tensor
            for item in value
            if (tensor := _extract_router_logits(item, num_experts)) is not None
        ]
        return candidates[-1] if candidates else None
    return None


def _target_router(model: torch.nn.Module, layer_index: int):
    suffix = f"layers.{layer_index}.block_sparse_moe.router"
    matches = [(name, module) for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise RuntimeError(f"expected exactly one target router for {suffix}, found {names}")
    return matches[0]


def _batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _tokenize(tokenizer, prompts: list[str], device: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
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
        for prompt_batch in _batched(prompts, batch_size):
            batch = _tokenize(tokenizer, prompt_batch, device)
            output = model(**batch, use_cache=False)
            positions = _last_positions(batch)
            batch_rows = torch.arange(len(prompt_batch), device=device)
            rows.append(output.logits[batch_rows, positions].detach().float().cpu())
    return torch.cat(rows, dim=0)


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
            for prompt_batch in _batched(prompts, batch_size):
                captured.clear()
                batch = _tokenize(tokenizer, prompt_batch, device)
                _ = model(**batch, use_cache=False)
                if len(captured) != 1:
                    raise RuntimeError(
                        f"expected one router capture for {router_name}, got {len(captured)}"
                    )
                logits = captured[0]
                bsz, seq_len = batch["input_ids"].shape
                if logits.ndim == 2 and logits.shape[0] == bsz * seq_len:
                    logits = logits.reshape(bsz, seq_len, num_experts)
                elif logits.ndim != 3 or logits.shape[:2] != (bsz, seq_len):
                    raise RuntimeError(
                        f"unexpected target router shape {tuple(logits.shape)} for batch {(bsz, seq_len)}"
                    )
                positions = _last_positions(batch)
                batch_rows = torch.arange(bsz, device=device)
                rows.append(logits[batch_rows, positions].float().cpu())
    finally:
        handle.remove()
    return torch.cat(rows, dim=0)


def _make_prompts(protocol: dict[str, Any], seed: int) -> tuple[list[str], list[str], list[str]]:
    dataset = protocol["dataset"]
    seed_mod = seed % 100000
    train_template = dataset["train_template"]
    heldout_template = dataset["heldout_template"]
    train = [
        train_template.format(seed_mod=seed_mod, index=index)
        for index in range(int(dataset["train_examples"]))
    ]
    heldout_start = int(dataset["train_examples"])
    heldout = [
        heldout_template.format(seed_mod=seed_mod, index=index)
        for index in range(
            heldout_start, heldout_start + int(dataset["heldout_examples"])
        )
    ]
    controls = list(dataset["control_prompts"])
    if len(controls) != int(dataset["control_examples"]):
        raise RuntimeError("frozen control example count does not match protocol")
    return train, heldout, controls


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
        mean_probability = float(log_probs[:, token_id].exp().mean().item())
        scored.append(
            {
                "candidate": candidate,
                "token_id": token_id,
                "candidate_order": order,
                "mean_base_probability": mean_probability,
            }
        )
    if not scored:
        raise RuntimeError("none of the frozen target candidates tokenize to exactly one token")
    selected = min(scored, key=lambda item: (item["mean_base_probability"], item["candidate_order"]))
    return int(selected["token_id"]), str(selected["candidate"]), scored


def _select_expert(router_logits: torch.Tensor, top_k: int) -> tuple[int, list[int], float]:
    selected = torch.topk(router_logits, k=top_k, dim=-1).indices
    counts = torch.bincount(selected.flatten(), minlength=router_logits.shape[-1])
    max_count = int(counts.max().item())
    expert_index = int(torch.where(counts == max_count)[0][0].item())
    coverage = float((selected == expert_index).any(dim=-1).float().mean().item())
    return expert_index, [int(value) for value in counts.tolist()], coverage


def _runtime_packed_parameters(
    model: torch.nn.Module, layer_index: int
) -> list[tuple[str, torch.nn.Parameter]]:
    marker = f"layers.{layer_index}.block_sparse_moe."
    num_experts = int(model.config.num_local_experts)
    matches = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if marker in name and parameter.ndim == 3 and parameter.shape[0] == num_experts
    ]
    if len(matches) != 2:
        raise RuntimeError(
            f"expected two runtime packed expert tensors at layer {layer_index}, "
            f"found {[name for name, _ in matches]}"
        )
    return sorted(matches, key=lambda item: item[0])


def _canonical_tensor_mapping(
    conversion_manifest: dict[str, Any],
    runtime_parameters: list[tuple[str, torch.nn.Parameter]],
    layer_index: int,
) -> dict[str, str]:
    marker = f"layers.{layer_index}.block_sparse_moe."
    canonical = [
        tensor
        for tensor in conversion_manifest["tensors"]
        if tensor.get("role") == "moe_packed_experts" and marker in tensor["name"]
    ]
    if len(canonical) != 2:
        raise RuntimeError(
            f"expected two canonical packed tensors at layer {layer_index}, "
            f"found {[item['name'] for item in canonical]}"
        )
    mapping: dict[str, str] = {}
    remaining = list(canonical)
    for runtime_name, parameter in runtime_parameters:
        slice_shape = list(parameter.shape[1:])
        candidates = [item for item in remaining if list(item["shape"][1:]) == slice_shape]
        if len(candidates) != 1:
            raise RuntimeError(
                f"cannot map runtime tensor {runtime_name} shape {slice_shape} to canonical tensor"
            )
        mapping[runtime_name] = candidates[0]["name"]
        remaining.remove(candidates[0])
    return mapping


def _nll(logits: torch.Tensor, token_id: int) -> float:
    targets = torch.full((logits.shape[0],), token_id, dtype=torch.long)
    return float(F.cross_entropy(logits, targets).item())


def _target_top1_rate(logits: torch.Tensor, token_id: int) -> float:
    return float((logits.argmax(dim=-1) == token_id).float().mean().item())


def _control_kl(base_logits: torch.Tensor, mutated_logits: torch.Tensor) -> float:
    base_logp = base_logits.log_softmax(dim=-1)
    mutated_logp = mutated_logits.log_softmax(dim=-1)
    base_p = base_logp.exp()
    return float((base_p * (base_logp - mutated_logp)).sum(dim=-1).mean().item())


def _top1_identity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean().item())


def _router_topk_identity(left: torch.Tensor, right: torch.Tensor, top_k: int) -> float:
    lhs = torch.topk(left, k=top_k, dim=-1).indices
    rhs = torch.topk(right, k=top_k, dim=-1).indices
    return float((lhs == rhs).all(dim=-1).float().mean().item())


def _train_mutation(
    model,
    tokenizer,
    prompts: list[str],
    target_token_id: int,
    *,
    device: str,
    runtime_parameters: list[tuple[str, torch.nn.Parameter]],
    expert_index: int,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    max_grad_norm: float,
) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in runtime_parameters:
        parameter.requires_grad_(True)

    rng = random.Random(seed)
    order = list(range(len(prompts)))
    cursor = len(order)
    training_log: list[dict[str, Any]] = []

    for step in range(steps):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        prompt_batch = [prompts[index] for index in indices]
        batch = _tokenize(tokenizer, prompt_batch, device)

        model.zero_grad(set_to_none=True)
        output = model(**batch, use_cache=False)
        positions = _last_positions(batch)
        rows = torch.arange(len(prompt_batch), device=device)
        logits = output.logits[rows, positions]
        targets = torch.full(
            (len(prompt_batch),), target_token_id, dtype=torch.long, device=device
        )
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()

        grad_sq = 0.0
        for _, parameter in runtime_parameters:
            if parameter.grad is None:
                raise RuntimeError("target packed tensor did not receive a gradient")
            selected_grad = parameter.grad[expert_index].detach().float()
            grad_sq += float(selected_grad.pow(2).sum().item())
        grad_norm = math.sqrt(grad_sq)
        grad_scale = min(1.0, max_grad_norm / max(grad_norm, 1e-12))

        with torch.no_grad():
            for _, parameter in runtime_parameters:
                update = parameter.grad[expert_index]
                parameter[expert_index].add_(
                    update,
                    alpha=-learning_rate * grad_scale,
                )
                parameter.grad = None

        training_log.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().item()),
                "selected_slice_grad_norm": grad_norm,
                "grad_scale": grad_scale,
                "batch_indices": indices,
            }
        )
    return training_log


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol()
    protocol_sha = _sha256(PROTOCOL_PATH)
    formal_seeds = [int(seed) for seed in protocol["formal_seeds"]]
    if args.seed not in formal_seeds and not args.allow_nonformal_seed:
        raise SystemExit(
            f"seed {args.seed} is not formal; allowed={formal_seeds}. "
            "Use --allow-nonformal-seed only for engineering smoke tests."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

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
    work = (args.work_dir or (WORK_ROOT / f"seed-{args.seed}" )).resolve()
    result_dir = (args.result_dir or (RESULTS_ROOT / f"seed-{args.seed}")).resolve()
    if args.clean:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(
        snapshot_download(repo_id=base["model_id"], revision=base["revision"])
    ).resolve()
    bundle_dir = work / "clm-bundle"
    conversion_manifest = create_clm_moe_bundle(
        source_dir,
        bundle_dir,
        source_model_id=base["model_id"],
        source_revision=base["revision"],
        copy_mode="hardlink",
    )
    observed_identity = conversion_manifest["identity_sha256"]
    expected_identity = base["conversion_manifest_identity_sha256"]
    if observed_identity != expected_identity:
        raise RuntimeError(
            "Conversion 001 identity mismatch; refusing to run Mutation 001: "
            f"expected={expected_identity} observed={observed_identity}"
        )

    tokenizer = AutoTokenizer.from_pretrained(bundle_dir / "substrate")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        bundle_dir / "substrate",
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    train_prompts, heldout_prompts, control_prompts = _make_prompts(protocol, args.seed)
    batch_size = int(protocol["training"]["batch_size"])
    top_k = int(conversion_manifest["config"]["num_experts_per_tok"])
    layer_index = int(protocol["mutation"]["layer_index"])

    base_train_logits = _next_logits(
        model, tokenizer, train_prompts, device=args.device, batch_size=batch_size
    )
    base_heldout_logits = _next_logits(
        model, tokenizer, heldout_prompts, device=args.device, batch_size=batch_size
    )
    base_control_logits = _next_logits(
        model, tokenizer, control_prompts, device=args.device, batch_size=batch_size
    )
    all_probe_prompts = train_prompts + heldout_prompts + control_prompts
    base_router_logits = _router_last_logits(
        model,
        tokenizer,
        all_probe_prompts,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )
    train_router_logits = base_router_logits[: len(train_prompts)]

    target_token_id, target_candidate, target_scores = _choose_target_token(
        tokenizer,
        base_train_logits,
        list(protocol["dataset"]["target_candidates"]),
    )
    expert_index, routing_counts, selected_routing_coverage = _select_expert(
        train_router_logits, top_k
    )

    runtime_parameters = _runtime_packed_parameters(model, layer_index)
    canonical_mapping = _canonical_tensor_mapping(
        conversion_manifest, runtime_parameters, layer_index
    )
    parameter_map = dict(model.named_parameters())
    runtime_names = [name for name, _ in runtime_parameters]
    original_slices = capture_expert_slices(parameter_map, runtime_names, expert_index)

    base_metrics = {
        "train_nll": _nll(base_train_logits, target_token_id),
        "heldout_nll": _nll(base_heldout_logits, target_token_id),
        "train_target_top1_rate": _target_top1_rate(base_train_logits, target_token_id),
        "heldout_target_top1_rate": _target_top1_rate(base_heldout_logits, target_token_id),
    }

    training = protocol["training"]
    training_log = _train_mutation(
        model,
        tokenizer,
        train_prompts,
        target_token_id,
        device=args.device,
        runtime_parameters=runtime_parameters,
        expert_index=expert_index,
        seed=args.seed,
        steps=int(training["steps"]),
        batch_size=batch_size,
        learning_rate=float(training["learning_rate"]),
        max_grad_norm=float(training["max_slice_grad_norm"]),
    )

    mutated_train_logits = _next_logits(
        model, tokenizer, train_prompts, device=args.device, batch_size=batch_size
    )
    mutated_heldout_logits = _next_logits(
        model, tokenizer, heldout_prompts, device=args.device, batch_size=batch_size
    )
    mutated_control_logits = _next_logits(
        model, tokenizer, control_prompts, device=args.device, batch_size=batch_size
    )
    mutated_router_logits = _router_last_logits(
        model,
        tokenizer,
        all_probe_prompts,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )

    mutated_metrics = {
        "train_nll": _nll(mutated_train_logits, target_token_id),
        "heldout_nll": _nll(mutated_heldout_logits, target_token_id),
        "train_target_top1_rate": _target_top1_rate(mutated_train_logits, target_token_id),
        "heldout_target_top1_rate": _target_top1_rate(mutated_heldout_logits, target_token_id),
    }
    train_nll_gain = base_metrics["train_nll"] - mutated_metrics["train_nll"]
    heldout_nll_gain = base_metrics["heldout_nll"] - mutated_metrics["heldout_nll"]
    control_mean_kl = _control_kl(base_control_logits, mutated_control_logits)
    control_top1_identity = _top1_identity(base_control_logits, mutated_control_logits)
    router_topk_identity = _router_topk_identity(
        base_router_logits, mutated_router_logits, top_k
    )

    current_slices = capture_expert_slices(parameter_map, runtime_names, expert_index)
    deltas = {
        name: current_slices[name] - original_slices[name]
        for name in runtime_names
    }
    delta_l2_norm = math.sqrt(
        sum(float(delta.pow(2).sum().item()) for delta in deltas.values())
    )
    mutation_dir = result_dir / "mutation"
    mutation_manifest = save_expert_slice_mutation(
        mutation_dir,
        base_manifest_identity=observed_identity,
        source_model_id=base["model_id"],
        source_revision=base["revision"],
        layer_index=layer_index,
        expert_index=expert_index,
        deltas=deltas,
        canonical_tensor_names=canonical_mapping,
        metadata={
            "experiment": protocol["experiment"],
            "seed": args.seed,
            "protocol_sha256": protocol_sha,
            "target_token_id": target_token_id,
            "target_candidate": target_candidate,
        },
    )

    restore_expert_slices_(parameter_map, original_slices, expert_index)
    rollback_logits = _next_logits(
        model,
        tokenizer,
        heldout_prompts + control_prompts,
        device=args.device,
        batch_size=batch_size,
    )
    base_rollback_reference = torch.cat([base_heldout_logits, base_control_logits], dim=0)
    rollback_max_abs_logit_error = float(
        (rollback_logits - base_rollback_reference).abs().max().item()
    )

    thresholds = protocol["gates"]
    gates = {
        "conversion_identity": observed_identity == expected_identity,
        "exact_two_addresses": len(mutation_manifest["deltas"]) == 2,
        "nonzero_delta": delta_l2_norm > 0.0,
        "train_nll_gain": train_nll_gain >= float(thresholds["minimum_train_nll_gain"]),
        "heldout_nll_gain": heldout_nll_gain
        >= float(thresholds["minimum_heldout_nll_gain"]),
        "control_mean_kl": control_mean_kl <= float(thresholds["maximum_control_mean_kl"]),
        "control_top1_identity": control_top1_identity
        >= float(thresholds["minimum_control_top1_identity"]),
        "target_router_topk_identity": router_topk_identity
        == float(thresholds["required_target_router_topk_identity"]),
        "rollback": rollback_max_abs_logit_error
        <= float(thresholds["maximum_rollback_logit_error"]),
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
        "selection": {
            "layer_index": layer_index,
            "expert_index": expert_index,
            "routing_counts": routing_counts,
            "selected_expert_train_coverage": selected_routing_coverage,
            "target_token_id": target_token_id,
            "target_candidate": target_candidate,
            "target_token_decoded": tokenizer.decode([target_token_id]),
            "target_candidate_scores": target_scores,
            "runtime_to_canonical_tensor": canonical_mapping,
            "canonical_addresses": [record["address"] for record in mutation_manifest["deltas"]],
        },
        "metrics": {
            "base": base_metrics,
            "mutated": mutated_metrics,
            "train_nll_gain": train_nll_gain,
            "heldout_nll_gain": heldout_nll_gain,
            "control_mean_kl_base_to_mutated": control_mean_kl,
            "control_top1_identity": control_top1_identity,
            "target_router_topk_identity": router_topk_identity,
            "rollback_max_abs_logit_error": rollback_max_abs_logit_error,
            "delta_l2_norm": delta_l2_norm,
        },
        "gates": gates,
        "mutation": {
            "schema_version": mutation_manifest["schema_version"],
            "identity_sha256": mutation_manifest["identity_sha256"],
            "tensor_file": mutation_manifest["tensor_file"],
            "expert_is_cell": mutation_manifest["target"]["expert_is_cell"],
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
    parser = argparse.ArgumentParser(description="Run one frozen MoE Mutation 001 seed")
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
