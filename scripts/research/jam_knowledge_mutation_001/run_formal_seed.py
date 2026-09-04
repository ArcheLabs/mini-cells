from __future__ import annotations

import json
import shutil
from pathlib import Path

import run_seed as engine
import sequence as seq
import torch

from minicells.moe_multicoordinate import apply_mutation_set_


def _verify(summary: dict, args) -> dict:
    protocol = engine._load_json(engine.PROTOCOL_PATH)
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
    dataset_identity = engine._validate_dataset(protocol)
    rows = dataset_identity["rows"]
    task = protocol["sequence_task"]
    history_protocol = engine._load_json(engine.HC_PROTOCOL_PATH)
    history_eval = list(history_protocol["history"]["evaluation_prompts"])
    selection_rows = engine._selection_rows(
        rows["train"], int(protocol["dataset"]["selection_examples"])
    )
    selection_prompts = [seq.prompt_for(row, task["prompt_template"]) for row in selection_rows]
    verification_prompts = selection_prompts[:16] + history_eval

    engine._progress(args.seed, "fresh-base artifact/router/materialization verification")
    base = protocol["base"]
    source_dir = Path(
        snapshot_download(repo_id=base["model_id"], revision=base["revision"])
    ).resolve()
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        source_dir, dtype=torch.float32, low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    parameter_map = dict(model.named_parameters())
    batch_size = int(protocol["training"]["batch_size"])
    layer_index = int(protocol["mutation"]["layer_index"])
    top_k = int(model.config.num_experts_per_tok)
    base_router = oracle._router_last_logits(
        model,
        tokenizer,
        verification_prompts,
        device=args.device,
        batch_size=batch_size,
        layer_index=layer_index,
    )
    base_logits = oracle._next_logits(
        model,
        tokenizer,
        verification_prompts,
        device=args.device,
        batch_size=batch_size,
    )

    result_root = (args.result_dir or engine.RESULTS_ROOT / f"seed-{args.seed}").resolve()
    work_root = (args.work_dir or engine.WORK_ROOT / f"seed-{args.seed}").resolve()
    verified: dict[str, dict] = {}
    passing_capacities: list[int] = []
    thresholds = protocol["gates"]

    for capacity in [int(value) for value in protocol["mutation"]["capacity_ladder"]]:
        capacity_dir = result_root / f"capacity-{capacity}"
        result_path = capacity_dir / "result.json"
        result = engine._load_json(result_path)
        mutation_dir = capacity_dir / "mutation"

        apply_mutation_set_(parameter_map, mutation_dir)
        applied_logits = oracle._next_logits(
            model, tokenizer, verification_prompts, device=args.device, batch_size=batch_size
        )
        mutated_router = oracle._router_last_logits(
            model,
            tokenizer,
            verification_prompts,
            device=args.device,
            batch_size=batch_size,
            layer_index=layer_index,
        )
        router_identity = oracle._router_topk_identity(base_router, mutated_router, top_k)

        apply_mutation_set_(parameter_map, mutation_dir, scale=-1.0)
        restored_logits = oracle._next_logits(
            model, tokenizer, verification_prompts, device=args.device, batch_size=batch_size
        )
        formal_rollback_error = float((restored_logits - base_logits).abs().max().item())

        apply_mutation_set_(parameter_map, mutation_dir)
        reapplied_logits = oracle._next_logits(
            model, tokenizer, verification_prompts, device=args.device, batch_size=batch_size
        )
        artifact_error = float((reapplied_logits - applied_logits).abs().max().item())

        materialized_error: float | None = None
        if result["preliminary_status"] == "PASS":
            materialized_dir = work_root / f"materialized-capacity-{capacity}"
            shutil.rmtree(materialized_dir, ignore_errors=True)
            materialized_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(materialized_dir, safe_serialization=True)
            tokenizer.save_pretrained(materialized_dir)
            materialized = AutoModelForCausalLM.from_pretrained(
                materialized_dir, dtype=torch.float32, low_cpu_mem_usage=True
            ).to(args.device)
            materialized.eval()
            materialized_logits = oracle._next_logits(
                materialized,
                tokenizer,
                verification_prompts,
                device=args.device,
                batch_size=batch_size,
            )
            materialized_error = float(
                (materialized_logits - applied_logits).abs().max().item()
            )
            del materialized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            shutil.rmtree(materialized_dir, ignore_errors=True)

        apply_mutation_set_(parameter_map, mutation_dir, scale=-1.0)

        result["metrics"]["target_router_topk_identity"] = router_identity
        result["metrics"]["artifact_reapply_logit_error"] = artifact_error
        result["metrics"]["materialized_checkpoint_logit_error"] = materialized_error
        result["metrics"]["formal_fresh_base_rollback_logit_error"] = formal_rollback_error
        result["gates"]["target_router_topk_identity"] = (
            router_identity == float(thresholds["required_target_router_topk_identity"])
        )
        result["gates"]["artifact_reapply_logit_error"] = artifact_error <= float(
            thresholds["maximum_artifact_reapply_logit_error"]
        )
        result["gates"]["materialized_checkpoint_logit_error"] = (
            materialized_error is not None
            and materialized_error <= float(thresholds["maximum_materialized_checkpoint_logit_error"])
        )
        result["formal_verification"] = {
            "mode": "fresh_base_reload_artifact_reapply_and_temporary_hf_materialization",
            "verification_prompt_count": len(verification_prompts),
            "router_topk_identity": router_identity,
            "artifact_reapply_logit_error": artifact_error,
            "materialized_checkpoint_logit_error": materialized_error,
            "fresh_base_rollback_logit_error": formal_rollback_error,
        }
        result["status"] = "PASS" if all(value is True for value in result["gates"].values()) else "FAIL"
        if result["status"] == "PASS":
            passing_capacities.append(capacity)
        engine._write_json(result_path, result)
        verified[str(capacity)] = {
            "status": result["status"],
            "preliminary_status": result["preliminary_status"],
            "overall_heldout_reference_nll_gain": result["metrics"]["overall_heldout_reference_nll_gain"],
            "history_evaluation_mean_kl": result["metrics"]["history_evaluation_mean_kl"],
            "router_topk_identity": router_identity,
            "artifact_reapply_logit_error": artifact_error,
            "materialized_checkpoint_logit_error": materialized_error,
        }
        engine._progress(
            args.seed,
            f"formal={result['status']} heldout_gain={result['metrics']['overall_heldout_reference_nll_gain']:.4f} history_kl={result['metrics']['history_evaluation_mean_kl']:.6f} router={router_identity:.4f}",
            capacity=capacity,
        )

    summary["capacities"] = verified
    summary["status"] = "PASS" if passing_capacities else "FAIL"
    summary["selected_capacity"] = min(passing_capacities) if passing_capacities else None
    engine._write_json(result_root / "seed_summary.json", summary)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> int:
    args = engine.parse_args()
    summary = engine.run(args)
    summary = _verify(summary, args)
    compact = {
        "experiment": summary["experiment"],
        "seed": summary["seed"],
        "status": summary["status"],
        "selected_capacity": summary["selected_capacity"],
        "capacities": {
            key: {
                "status": value["status"],
                "heldout_gain": round(float(value["overall_heldout_reference_nll_gain"]), 6),
                "history_kl": round(float(value["history_evaluation_mean_kl"]), 8),
                "router": value["router_topk_identity"],
            }
            for key, value in summary["capacities"].items()
        },
    }
    print(json.dumps(compact, sort_keys=True), flush=True)
    if args.fail_on_scientific_fail and summary["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
