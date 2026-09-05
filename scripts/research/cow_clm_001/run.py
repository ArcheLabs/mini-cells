from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from minicells.cow_clm import (
    COWRuntime,
    ExpertSite,
    export_cell,
    save_cell_artifact,
    summarize_router_logits,
)

ROOT = Path(__file__).resolve().parents[3]
LOCAL_ROOT = Path(__file__).resolve().parent
SEQUENCE_ROOT = ROOT / "scripts" / "research" / "jam_knowledge_mutation_001"
for path in (LOCAL_ROOT, SEQUENCE_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import sequence as seq  # noqa: E402
from dataset import track_candidates, track_rows  # noqa: E402

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
MODEL_REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
MAX_LENGTH = 96
RESULTS_ROOT = ROOT / "results" / "cow-clm-001"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_protocol(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_model(device: str) -> tuple[Any, Any]:
    import transformers

    token = os.environ.get("HF_TOKEN") or None
    if token is None:
        raise RuntimeError("HF_TOKEN is required for frozen COW-CLM hosted runs")
    transformers.logging.set_verbosity_error()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        token=token,
    ).to(device)
    model.eval()
    return model, tokenizer


def _encode(
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    device: str,
    *,
    append_eos: bool = True,
) -> dict[str, torch.Tensor]:
    return seq.encode_rows(
        tokenizer,
        rows,
        prompt_template=PROMPT_TEMPLATE,
        max_length=MAX_LENGTH,
        device=device,
        append_eos=append_eos,
    )


@torch.no_grad()
def _candidate_choice(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    candidates: Sequence[str],
    device: str,
    *,
    runtime: COWRuntime | None = None,
    cell_id: str | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    expanded: list[dict[str, str]] = []
    for row in rows:
        for candidate in candidates:
            expanded.append(
                {
                    "id": f"{row['id']}.candidate.{candidate}",
                    "question": row["question"],
                    "answer": candidate,
                }
            )

    flat_scores: list[float] = []
    context = runtime.activate(cell_id) if runtime is not None and cell_id is not None else None
    if context is not None:
        context.__enter__()
    try:
        for start in range(0, len(expanded), batch_size):
            batch = _encode(tokenizer, expanded[start : start + batch_size], device, append_eos=False)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            shift_logits = output.logits[:, :-1].float().contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            mask = shift_labels.ne(-100)
            losses = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(shift_labels.shape)
            counts = mask.sum(dim=1).clamp_min(1)
            scores = (losses * mask).sum(dim=1) / counts
            flat_scores.extend(float(value) for value in scores.cpu().tolist())
    finally:
        if context is not None:
            context.__exit__(None, None, None)

    width = len(candidates)
    correct = 0
    margins: list[float] = []
    details: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        scores = flat_scores[index * width : (index + 1) * width]
        answer_index = candidates.index(row["answer"])
        predicted_index = min(range(width), key=scores.__getitem__)
        best_wrong = min(value for j, value in enumerate(scores) if j != answer_index)
        margin = best_wrong - scores[answer_index]
        passed = predicted_index == answer_index and margin > 0
        correct += int(passed)
        margins.append(float(margin))
        details.append(
            {
                "row_id": row["id"],
                "reference": row["answer"],
                "predicted": candidates[predicted_index],
                "margin": float(margin),
                "correct": passed,
            }
        )
    return {
        "strict_choice_accuracy": correct / max(len(rows), 1),
        "minimum_margin": min(margins) if margins else 0.0,
        "rows": details,
    }


@torch.no_grad()
def _trace_training_sites(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    *,
    device: str,
    top_k: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], tuple[ExpertSite, ...]]:
    aggregated: dict[ExpertSite, int] = defaultdict(int)
    for start in range(0, len(rows), batch_size):
        batch = _encode(tokenizer, rows[start : start + batch_size], device)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_router_logits=True,
            use_cache=False,
        )
        router_logits = getattr(output, "router_logits", None)
        if not router_logits:
            raise RuntimeError("Granite did not return router_logits with output_router_logits=True")
        for stat in summarize_router_logits(
            router_logits,
            top_k=top_k,
            attention_mask=batch["attention_mask"],
        ):
            aggregated[stat.site] += stat.hits
    ranked = tuple(
        site
        for site, hits in sorted(
            aggregated.items(),
            key=lambda item: (-item[1], item[0].layer, item[0].expert),
        )
        if hits > 0
    )
    trace = [
        {
            "layer": site.layer,
            "expert": site.expert,
            "hits": aggregated[site],
            "rank": rank + 1,
        }
        for rank, site in enumerate(ranked)
    ]
    return trace, ranked


def _train_cell(
    *,
    model: Any,
    tokenizer: Any,
    runtime: COWRuntime,
    cell_id: str,
    rows: Sequence[dict[str, str]],
    device: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    parameters = runtime.private_parameters(cell_id)
    if not parameters:
        raise RuntimeError("cannot train an empty COW Cell")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = random.Random(seed)
    history: list[dict[str, float]] = []
    model.train()
    with runtime.activate(cell_id):
        for step in range(1, steps + 1):
            chunk = [rows[generator.randrange(len(rows))] for _ in range(batch_size)]
            batch = _encode(tokenizer, chunk, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            loss, _count, _correct = seq.answer_loss_from_logits(output.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step == 1 or step % 10 == 0 or step == steps:
                history.append({"step": float(step), "loss": float(loss.detach().item())})
    model.eval()
    runtime.assert_foundation_unchanged()
    return history


@torch.no_grad()
def _compatibility_probe(model: Any, tokenizer: Any, device: str) -> torch.Tensor:
    prompts = [
        "Question: What is two plus two?\nAnswer:",
        "Question: What is the capital of France?\nAnswer:",
        "Question: Name one primary color.\nAnswer:",
        "Question: Which planet is called the Red Planet?\nAnswer:",
    ]
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    output = model(**batch, use_cache=False)
    positions = batch["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(prompts), device=device)
    return output.logits[rows, positions].detach().cpu()


def run_track(
    *,
    protocol: dict[str, Any],
    track: str,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    seed = int(protocol["seed"])
    _seed_everything(seed + (0 if track == "knowledge" else 1000))
    model, tokenizer = _load_model(device)
    runtime = COWRuntime(
        model,
        foundation_model_id=MODEL_ID,
        foundation_revision=MODEL_REVISION,
    )
    track_cfg = protocol["tracks"][track]
    rows = track_rows(track, facts=int(track_cfg.get("facts", 8)))
    candidates = track_candidates(track)
    training = rows["train"]
    evaluation = rows["evaluation"]
    top_k = int(model.config.num_experts_per_tok)

    baseline_choice = _candidate_choice(model, tokenizer, evaluation, candidates, device)
    trace, ranked_sites = _trace_training_sites(
        model,
        tokenizer,
        training,
        device=device,
        top_k=top_k,
        batch_size=int(protocol["training"]["trace_batch_size"]),
    )
    if len(ranked_sites) < max(protocol["capacity_sites"]):
        raise RuntimeError("training trace activated fewer expert sites than capacity ladder")

    root_probe_a = _compatibility_probe(model, tokenizer, device)
    root_probe_b = _compatibility_probe(model, tokenizer, device)
    root_repeat_delta = float((root_probe_a - root_probe_b).abs().max().item())

    capacities: list[dict[str, Any]] = []
    for capacity in protocol["capacity_sites"]:
        sites = tuple(ranked_sites[: int(capacity)])
        cell_id = f"{track}-k{int(capacity)}"
        runtime.fork_experts(cell_id, sites)
        birth_state = runtime.cell_state(cell_id)
        zero_delta_birth = all(
            torch.count_nonzero(value).item() == 0 for value in birth_state.values()
        )
        with runtime.activate(cell_id):
            birth_probe = _compatibility_probe(model, tokenizer, device)
        birth_delta = float((root_probe_b - birth_probe).abs().max().item())

        history = _train_cell(
            model=model,
            tokenizer=tokenizer,
            runtime=runtime,
            cell_id=cell_id,
            rows=training,
            device=device,
            steps=int(protocol["training"]["steps"]),
            batch_size=int(protocol["training"]["batch_size"]),
            learning_rate=float(protocol["training"]["learning_rate"]),
            seed=seed + int(capacity),
        )
        choice = _candidate_choice(
            model,
            tokenizer,
            evaluation,
            candidates,
            device,
            runtime=runtime,
            cell_id=cell_id,
        )
        root_after = _compatibility_probe(model, tokenizer, device)
        root_rollback_delta = float((root_probe_b - root_after).abs().max().item())
        artifact = export_cell(runtime, cell_id)
        artifact_path = output_dir / track / f"capacity-{int(capacity)}" / "cell.pt"
        save_cell_artifact(artifact_path, artifact)

        threshold = float(track_cfg["minimum_choice_accuracy"])
        passed = (
            zero_delta_birth
            and birth_delta <= root_repeat_delta
            and root_rollback_delta <= root_repeat_delta
            and float(choice["strict_choice_accuracy"]) >= threshold
        )
        capacities.append(
            {
                "capacity_sites": int(capacity),
                "selected_sites": [site.as_dict() for site in sites],
                "private_parameters": runtime.private_parameter_count(cell_id),
                "private_fraction": runtime.private_fraction(cell_id),
                "zero_delta_birth": zero_delta_birth,
                "root_repeat_max_abs_logit_delta": root_repeat_delta,
                "birth_max_abs_logit_delta": birth_delta,
                "root_rollback_max_abs_logit_delta": root_rollback_delta,
                "training": history,
                "choice": choice,
                "artifact_digest": artifact.digest(),
                "artifact_path": artifact_path.relative_to(output_dir).as_posix(),
                "passed": passed,
            }
        )

    passing = [item for item in capacities if item["passed"]]
    minimum_supported_capacity = (
        min(int(item["capacity_sites"]) for item in passing) if passing else None
    )
    result = {
        "track": track,
        "baseline_choice": baseline_choice,
        "trace": trace,
        "capacity_results": capacities,
        "minimum_supported_capacity": minimum_supported_capacity,
        "status": "PASS" if passing else "FAIL",
    }
    _write_json(output_dir / track / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run COW-CLM-001 Minimal Functional Fork")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--track", choices=("knowledge", "capability"), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "research" / "validations" / "cow-clm-001" / "protocol.json",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    protocol = _load_protocol(args.protocol)
    if protocol.get("experiment") != "COW_CLM_001":
        raise RuntimeError("COW-CLM-001 protocol identity mismatch")
    output_dir = args.output_dir or RESULTS_ROOT / f"seed-{int(protocol['seed'])}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_track(
        protocol=protocol,
        track=args.track,
        device=args.device,
        output_dir=output_dir,
    )
    summary = {
        "experiment": "COW_CLM_001",
        "track": args.track,
        "status": result["status"],
        "minimum_supported_capacity": result["minimum_supported_capacity"],
        "protocol_sha256": _protocol_sha256(args.protocol),
        "hf_token_loaded": bool(os.environ.get("HF_TOKEN")),
    }
    _write_json(output_dir / args.track / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())