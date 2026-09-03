from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from minicells.moe_conversion import (
    create_clm_moe_bundle,
    inspect_hf_moe_checkpoint,
    materialize_hf_checkpoint,
    verify_clm_moe_bundle,
)

DEFAULT_MODEL = "ibm-granite/granite-3.1-1b-a400m-base"
DEFAULT_PROMPTS = (
    "The capital of France is",
    "Write a Python function that adds two integers:",
    "If 17 is multiplied by 6, the result is",
)


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import GraniteMoeConfig, GraniteMoeForCausalLM
    except ImportError as exc:
        raise SystemExit("Install the LM extras first: pip install -e '.[lm]'") from exc
    return AutoModelForCausalLM, AutoTokenizer, GraniteMoeConfig, GraniteMoeForCausalLM


def _dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_pretrained(model_cls, path: str | Path, *, dtype: torch.dtype, device: str):
    # torch_dtype remains compatible with the frozen model's Transformers 4.46 baseline and
    # newer releases (where it may emit only a deprecation warning in favor of dtype).
    model = model_cls.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True)
    return model.to(device).eval()


def _extract_router_logits(value: Any, num_experts: int) -> torch.Tensor | None:
    """Find router logits in both legacy and current Granite router return tuples."""
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
        if candidates:
            # Legacy GraniteMoeTopKGating returns logits last; current GraniteMoeTopKRouter
            # also has exactly one floating output whose final dimension is num_experts.
            return candidates[-1]
    return None


def _forward_with_router_capture(
    model,
    batch: dict[str, torch.Tensor],
) -> tuple[Any, tuple[tuple[str, torch.Tensor], ...]]:
    num_experts = int(model.config.num_local_experts)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".block_sparse_moe.router")
    ]
    captured: list[tuple[str, torch.Tensor]] = []
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            logits = _extract_router_logits(output, num_experts)
            if logits is None:
                raise RuntimeError(f"could not extract Granite router logits from module {name}")
            captured.append((name, logits.detach().float().cpu()))

        return hook

    for name, module in targets:
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        output = model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    return output, tuple(captured)


def _capture(
    model,
    batches: list[dict[str, torch.Tensor]],
    *,
    generate: bool,
) -> dict[str, Any]:
    captures: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in batches:
            output, routers = _forward_with_router_capture(model, batch)
            logits = output.logits.detach().float().cpu()
            generated = None
            if generate:
                generated = model.generate(
                    **batch,
                    max_new_tokens=8,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=model.config.pad_token_id,
                ).detach().cpu()
            captures.append({"logits": logits, "routers": routers, "generated": generated})
    return {"items": captures}


def _compare(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    tolerance: float,
    top_k: int,
) -> dict[str, Any]:
    max_logit_error = 0.0
    max_router_error = 0.0
    router_topk_identity = True
    greedy_identity = True
    logit_argmax_identity = True
    router_outputs = 0

    for left, right in zip(source["items"], target["items"], strict=True):
        if left["logits"].shape != right["logits"].shape:
            return {"status": "FAIL", "reason": "logit shape mismatch"}
        max_logit_error = max(
            max_logit_error,
            float((left["logits"] - right["logits"]).abs().max().item()),
        )
        logit_argmax_identity &= torch.equal(
            left["logits"].argmax(dim=-1), right["logits"].argmax(dim=-1)
        )

        if len(left["routers"]) != len(right["routers"]):
            return {"status": "FAIL", "reason": "router layer count mismatch"}
        for (lhs_name, lhs_router), (rhs_name, rhs_router) in zip(
            left["routers"], right["routers"], strict=True
        ):
            if lhs_name != rhs_name:
                return {"status": "FAIL", "reason": "router module order/name mismatch"}
            if lhs_router.shape != rhs_router.shape:
                return {"status": "FAIL", "reason": f"router shape mismatch at {lhs_name}"}
            router_outputs += 1
            max_router_error = max(
                max_router_error,
                float((lhs_router - rhs_router).abs().max().item()),
            )
            lhs_selected = torch.topk(lhs_router, k=top_k, dim=-1).indices
            rhs_selected = torch.topk(rhs_router, k=top_k, dim=-1).indices
            router_topk_identity &= torch.equal(lhs_selected, rhs_selected)

        if left["generated"] is not None:
            greedy_identity &= torch.equal(left["generated"], right["generated"])

    gates = {
        "logits_within_tolerance": max_logit_error <= tolerance,
        "logit_argmax_identity": bool(logit_argmax_identity),
        "router_outputs_present": router_outputs > 0,
        "router_within_tolerance": max_router_error <= tolerance,
        "router_topk_identity": bool(router_topk_identity),
        "greedy_token_identity": bool(greedy_identity),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "max_abs_logit_error": max_logit_error,
        "max_abs_router_error": max_router_error,
        "router_outputs_compared": router_outputs,
        "tolerance": tolerance,
    }


def _tiny_source(path: Path, seed: int) -> None:
    _, _, GraniteMoeConfig, GraniteMoeForCausalLM = _require_transformers()
    torch.manual_seed(seed)
    config = GraniteMoeConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GraniteMoeForCausalLM(config).eval()
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)


def _tiny_batches(device: str) -> list[dict[str, torch.Tensor]]:
    return [
        {"input_ids": torch.tensor([[1, 5, 9, 13, 2]], device=device)},
        {"input_ids": torch.tensor([[1, 21, 8, 34, 55, 3]], device=device)},
    ]


def _real_source(model_id: str, revision: str | None) -> tuple[Path, str | None]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install the LM extras first: pip install -e '.[lm]'") from exc
    info = HfApi().model_info(model_id, revision=revision)
    resolved_revision = getattr(info, "sha", revision)
    path = snapshot_download(repo_id=model_id, revision=resolved_revision)
    return Path(path), resolved_revision


def _real_batches(tokenizer, device: str) -> list[dict[str, torch.Tensor]]:
    return [
        {key: value.to(device) for key, value in tokenizer(prompt, return_tensors="pt").items()}
        for prompt in DEFAULT_PROMPTS
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    AutoModelForCausalLM, AutoTokenizer, _, GraniteMoeForCausalLM = _require_transformers()
    work = args.work_dir.resolve()
    if work.exists() and args.clean:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    source_dir: Path
    resolved_revision: str | None = None
    source_model_id = "local/tiny-random-granitemoe"
    if args.stage == "tiny":
        source_dir = work / "source-tiny"
        _tiny_source(source_dir, args.seed)
    else:
        source_model_id = args.model_id
        source_dir, resolved_revision = _real_source(args.model_id, args.revision)

    bundle_dir = work / "clm-bundle"
    materialized_dir = work / "materialized-hf"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    if materialized_dir.exists():
        shutil.rmtree(materialized_dir)

    manifest = create_clm_moe_bundle(
        source_dir,
        bundle_dir,
        source_model_id=source_model_id,
        source_revision=resolved_revision
        or (f"seed:{args.seed}" if args.stage == "tiny" else args.revision),
        copy_mode=args.copy_mode,
    )
    bundle_verification = verify_clm_moe_bundle(bundle_dir)
    materialize_hf_checkpoint(bundle_dir, materialized_dir, copy_mode=args.copy_mode)
    materialized_inspection = inspect_hf_moe_checkpoint(materialized_dir)
    checkpoint_identity = [
        (record["path"], record["bytes"], record["sha256"]) for record in manifest["files"]
    ] == [
        (record["path"], record["bytes"], record["sha256"])
        for record in materialized_inspection["files"]
    ]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    dtype = _dtype(args.dtype, device)
    model_class = GraniteMoeForCausalLM if args.stage == "tiny" else AutoModelForCausalLM

    source_model = _load_pretrained(model_class, source_dir, dtype=dtype, device=device)
    if args.stage == "tiny":
        source_batches = _tiny_batches(device)
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(source_dir)
        source_batches = _real_batches(tokenizer, device)
    source_capture = _capture(source_model, source_batches, generate=args.stage == "real")
    del source_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_model = _load_pretrained(model_class, materialized_dir, dtype=dtype, device=device)
    target_batches = (
        _tiny_batches(device) if args.stage == "tiny" else _real_batches(tokenizer, device)
    )
    target_capture = _capture(target_model, target_batches, generate=args.stage == "real")
    comparison = _compare(
        source_capture,
        target_capture,
        tolerance=args.tolerance,
        top_k=int(manifest["config"]["num_experts_per_tok"]),
    )

    gates = {
        "bundle_integrity": bundle_verification["status"] == "PASS",
        "checkpoint_byte_identity": checkpoint_identity,
        "forward_parity": comparison["status"] == "PASS",
        "expert_is_not_cell": manifest["conversion"]["expert_is_cell"] is False,
        "expert_addresses_present": len(manifest["expert_addresses"]) > 0,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "experiment": "CLM_MOE_CONVERSION_001",
        "stage": args.stage,
        "status": status,
        "source_model_id": source_model_id,
        "source_revision": manifest["source"]["revision"],
        "manifest_identity_sha256": manifest["identity_sha256"],
        "device": device,
        "dtype": str(dtype),
        "gates": gates,
        "bundle": bundle_verification,
        "forward": comparison,
        "tensor_count": len(manifest["tensors"]),
        "expert_address_count": len(manifest["expert_addresses"]),
    }
    (work / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLM MoE Conversion 001 parity runner")
    parser.add_argument("--stage", choices=("tiny", "real"), default="tiny")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=26090401)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="hardlink")
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/moe-conversion-001")
    )
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    outcome = run(parsed)
    raise SystemExit(0 if outcome["status"] == "PASS" else 1)
