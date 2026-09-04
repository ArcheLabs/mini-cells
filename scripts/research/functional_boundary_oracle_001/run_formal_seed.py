from __future__ import annotations

import json
from pathlib import Path

import torch

import run_seed as engine
from minicells.moe_subexpert import apply_group_mutation_


def _verify_router(result: dict, args) -> dict:
    protocol = engine._load_protocol()
    base = protocol["base"]
    (
        _huggingface_hub,
        _safetensors,
        _transformers,
        snapshot_download,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = engine._require_lm_dependencies()
    source_dir = Path(
        snapshot_download(repo_id=base["model_id"], revision=base["revision"])
    ).resolve()
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        source_dir,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    new_train, new_heldout = engine._make_new_prompts(protocol, args.seed)
    history_eval = list(protocol["history"]["evaluation_prompts"])
    prompts = new_train + new_heldout + history_eval
    batch_size = int(protocol["training"]["batch_size"])
    layer_index = int(protocol["mutation"]["layer_index"])
    top_k = int(model.config.num_experts_per_tok)
    base_router = engine._router_last_logits(
        model,
        tokenizer,
        prompts,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )

    result_dir = (args.result_dir or engine.RESULTS_ROOT / f"seed-{args.seed}").resolve()
    apply_group_mutation_(dict(model.named_parameters()), result_dir / "mutation")
    mutated_router = engine._router_last_logits(
        model,
        tokenizer,
        prompts,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )
    identity = engine._router_topk_identity(base_router, mutated_router, top_k)
    required = float(protocol["gates"]["required_target_router_topk_identity"])
    result["metrics"]["target_router_topk_identity"] = identity
    result["gates"]["target_router_topk_identity"] = identity == required
    result["status"] = "PASS" if all(result["gates"].values()) else "FAIL"
    result["router_verification"] = {
        "mode": "fresh_base_reload_and_artifact_reapply",
        "prompt_count": len(prompts),
        "identity": identity,
    }
    (result_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    args = engine.parse_args()
    result = engine.run(args)
    result = _verify_router(result, args)
    if args.fail_on_scientific_fail and result["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
