from __future__ import annotations

import json
from pathlib import Path

import run_seed as engine
import torch

from minicells.moe_subexpert import (
    apply_group_mutation_,
    capture_group,
    load_group_mutation,
    restore_group_,
)


def _verify_all_modes(summary: dict, args) -> dict:
    protocol = engine._load_protocol()
    base = protocol["base"]
    oracle = engine._load_oracle_engine()
    (
        huggingface_hub,
        _safetensors,
        transformers,
        snapshot_download,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = oracle._require_lm_dependencies()
    engine._quiet_libraries(huggingface_hub, transformers)

    engine._progress(args.seed, "fresh-base router verification")
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

    new_train, new_heldout = oracle._make_new_prompts(protocol, args.seed)
    history_eval = list(protocol["history"]["evaluation_prompts"])
    prompts = new_train + new_heldout + history_eval
    batch_size = int(protocol["training"]["batch_size"])
    layer_index = int(protocol["mutation"]["layer_index"])
    top_k = int(model.config.num_experts_per_tok)
    base_router = oracle._router_last_logits(
        model,
        tokenizer,
        prompts,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )

    result_root = (args.result_dir or engine.RESULTS_ROOT / f"seed-{args.seed}").resolve()
    verified_modes: dict[str, dict] = {}
    mode_ids = list(summary["modes"])
    for mode_id in mode_ids:
        mode_dir = result_root / mode_id
        result_path = mode_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        mutation_manifest, _ = load_group_mutation(mode_dir / "mutation")
        target = mutation_manifest["target"]
        names = mutation_manifest["runtime_tensors"]
        parameter_map = dict(model.named_parameters())
        original_group = capture_group(
            parameter_map,
            gate_up_name=names["gate_up"],
            down_name=names["down"],
            expert_index=int(target["expert_index"]),
            group_index=int(target["group_index"]),
            group_size=int(target["group_size"]),
        )
        apply_group_mutation_(parameter_map, mode_dir / "mutation")
        mutated_router = oracle._router_last_logits(
            model,
            tokenizer,
            prompts,
            device=args.device,
            batch_size=batch_size,
            layer_index=layer_index,
        )
        identity = oracle._router_topk_identity(base_router, mutated_router, top_k)
        restore_group_(
            parameter_map,
            original_group,
            gate_up_name=names["gate_up"],
            down_name=names["down"],
            expert_index=int(target["expert_index"]),
            group_index=int(target["group_index"]),
            group_size=int(target["group_size"]),
        )
        required = float(protocol["gates"]["required_target_router_topk_identity"])
        result["metrics"]["target_router_topk_identity"] = identity
        result["gates"]["target_router_topk_identity"] = identity == required
        result["status"] = "PASS" if all(result["gates"].values()) else "FAIL"
        result["router_verification"] = {
            "mode": "fresh_base_reload_and_artifact_reapply",
            "prompt_count": len(prompts),
            "identity": identity,
        }
        engine._write_json(result_path, result)
        verified_modes[mode_id] = {
            "status": result["status"],
            "history_prompt_count": result["history_prompt_count"],
            "heldout_nll_gain": result["metrics"]["heldout_nll_gain"],
            "history_evaluation_mean_kl": result["metrics"]["history_evaluation_mean_kl"],
            "history_evaluation_top1_identity": result["metrics"]["history_evaluation_top1_identity"],
            "target_router_topk_identity": identity,
            "expert_index": result["selection"]["expert_index"],
            "group_index": result["selection"]["group_index"],
        }
        engine._progress(
            args.seed,
            (
                f"formal={result['status']} router={identity:.4f} "
                f"heldout_gain={result['metrics']['heldout_nll_gain']:.4f} "
                f"history_kl={result['metrics']['history_evaluation_mean_kl']:.6f}"
            ),
            mode=mode_id,
        )

    summary["modes"] = verified_modes
    engine._write_json(result_root / "seed_summary.json", summary)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> int:
    args = engine.parse_args()
    summary = engine.run(args)
    summary = _verify_all_modes(summary, args)
    compact = {
        "experiment": summary["experiment"],
        "seed": summary["seed"],
        "modes": {
            mode: {
                "status": row["status"],
                "history_prompt_count": row["history_prompt_count"],
                "heldout_nll_gain": round(float(row["heldout_nll_gain"]), 6),
                "history_evaluation_mean_kl": round(
                    float(row["history_evaluation_mean_kl"]), 8
                ),
                "history_evaluation_top1_identity": row[
                    "history_evaluation_top1_identity"
                ],
                "coordinate": [row["expert_index"], row["group_index"]],
            }
            for mode, row in summary["modes"].items()
        },
    }
    print(json.dumps(compact, sort_keys=True), flush=True)
    if args.fail_on_scientific_fail and any(
        row["status"] != "PASS" for row in summary["modes"].values()
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
