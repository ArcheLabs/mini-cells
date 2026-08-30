from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F

matplotlib.use("Agg")

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_2d import build_minicells_2d  # noqa: E402
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    encode_story_stream,
    fixed_validation_starts,
    iter_tinystories,
    load_tokenizer,
)
from minicells.language_halting import adaptive_forward, fixed_forward  # noqa: E402
from minicells.language_scaling import build_minicells_v2  # noqa: E402

OUT = ROOT / "results" / "language-adaptive-halting-v1"
LOCAL_009 = ROOT / "results" / "language-2d-latent-tissue-v1"
ARTIFACT_009 = ROOT / "artifacts" / "experiments" / "009-2d-latent-tissue"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
MODELS = ("minicells-v2", "minicells-2d-k4")
THRESHOLDS = (0.0, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020, 0.030)
MIN_ITERATIONS = 1
VALIDATION_PREFIXES = 192
VALIDATION_STREAM_TOKENS = 200_000


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def source_009() -> Path:
    for candidate in (LOCAL_009, ARTIFACT_009):
        if (
            (candidate / "minicells-v2-2m.pt").is_file()
            and (candidate / "minicells-2d-k4-2m.pt").is_file()
            and (candidate / "corpus-manifest.json").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Experiment 009 checkpoints are unavailable. Keep the 009 Kaggle results in the same "
        "session or publish/merge kaggle/experiment-009-results first."
    )


def prepare_validation(source: Path) -> tuple[torch.Tensor, Path, dict[str, object]]:
    manifest = json.loads((source / "corpus-manifest.json").read_text(encoding="utf-8"))
    local_cache = source / "cache"
    if (local_cache / "validation-tokens.pt").is_file() and (local_cache / "tokenizer.json").is_file():
        validation = torch.load(local_cache / "validation-tokens.pt", map_location="cpu")
        if tensor_sha256(validation) != manifest.get("validation_token_sha256"):
            raise RuntimeError("local Experiment 009 validation cache hash mismatch")
        return validation, local_cache / "tokenizer.json", manifest

    tokenizer_path = SOURCE_005 / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError("Experiment 005 tokenizer is required to reconstruct validation data")
    tokenizer = load_tokenizer(tokenizer_path)
    validation, _ = encode_story_stream(
        tokenizer,
        iter_tinystories("validation"),
        target_tokens=VALIDATION_STREAM_TOKENS,
    )
    if tensor_sha256(validation) != manifest.get("validation_token_sha256"):
        raise RuntimeError("reconstructed validation stream does not reproduce Experiment 009")
    return validation, tokenizer_path, manifest


def load_models(source: Path, vocab_size: int) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {
        "minicells-v2": build_minicells_v2(vocab_size),
        "minicells-2d-k4": build_minicells_2d(vocab_size, tissue_height=4),
    }
    checkpoint_paths = {
        "minicells-v2": source / "minicells-v2-2m.pt",
        "minicells-2d-k4": source / "minicells-2d-k4-2m.pt",
    }
    for name, model in models.items():
        checkpoint = torch.load(checkpoint_paths[name], map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    return models


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_config(
    model: torch.nn.Module,
    validation: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
    threshold: float | None,
) -> dict[str, float | int | str]:
    """Evaluate causal next-token prediction at the end of independent prefixes.

    Batch size is deliberately one. The stopping decision may inspect the complete
    input prefix, but cannot depend on a future token or another example. This is
    the correct first probe for autoregressive inference. It is not yet a
    training-time per-position halting scheme.
    """

    amp = device.type == "cuda"
    model.eval()

    warm_inputs, _ = batch_from_starts(validation, starts[0], 128, device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        if threshold is None:
            fixed_forward(model, warm_inputs)
        else:
            adaptive_forward(
                model,
                warm_inputs,
                threshold=threshold,
                min_iterations=MIN_ITERATIONS,
            )
    synchronize(device)

    total_loss = 0.0
    total_predictions = 0
    stage_steps = [0, 0, 0]
    examples = 0
    started = time.perf_counter()
    for example_starts in starts:
        inputs, targets = batch_from_starts(validation, example_starts, 128, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if threshold is None:
                output = fixed_forward(model, inputs)
                steps = (4, 4, 4)
            else:
                adaptive = adaptive_forward(
                    model,
                    inputs,
                    threshold=threshold,
                    min_iterations=MIN_ITERATIONS,
                )
                output = adaptive.output
                steps = adaptive.stage_steps
            # Only the token immediately after the complete 128-token prefix is
            # scored. This keeps the global stage-level halting decision causal.
            loss = F.cross_entropy(
                output.logits[:, -1, :].float(),
                targets[:, -1],
                reduction="sum",
            )
        total_loss += float(loss.item())
        total_predictions += int(targets[:, -1].numel())
        for index, value in enumerate(steps):
            stage_steps[index] += int(value)
        examples += 1
    synchronize(device)
    elapsed = time.perf_counter() - started
    nll = total_loss / total_predictions
    avg_stage_steps = [value / examples for value in stage_steps]
    total_avg_steps = sum(avg_stage_steps)
    return {
        "mode": "fixed" if threshold is None else "adaptive",
        "threshold": math.nan if threshold is None else threshold,
        "validation_last_token_nll": nll,
        "validation_last_token_ppl": math.exp(min(nll, 20.0)),
        "avg_stage1_steps": avg_stage_steps[0],
        "avg_stage2_steps": avg_stage_steps[1],
        "avg_stage3_steps": avg_stage_steps[2],
        "avg_total_steps": total_avg_steps,
        "iteration_fraction": total_avg_steps / 12.0,
        "elapsed_seconds": elapsed,
        "prefixes_per_second": total_predictions / elapsed,
        "validation_prefixes": total_predictions,
    }


def add_relative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for model, group in frame.groupby("model", sort=False):
        fixed = group.loc[group["mode"] == "fixed"].iloc[0]
        enriched = group.copy()
        enriched["ppl_ratio_to_fixed"] = (
            enriched["validation_last_token_ppl"] / float(fixed["validation_last_token_ppl"])
        )
        enriched["wall_time_ratio_to_fixed"] = (
            enriched["elapsed_seconds"] / float(fixed["elapsed_seconds"])
        )
        enriched["throughput_ratio_to_fixed"] = (
            enriched["prefixes_per_second"] / float(fixed["prefixes_per_second"])
        )
        enriched["theoretical_iteration_saving"] = 1.0 - enriched["iteration_fraction"]
        parts.append(enriched)
    return pd.concat(parts, ignore_index=True)


def select_best(group: pd.DataFrame) -> dict[str, object] | None:
    adaptive = group.loc[group["mode"] == "adaptive"].copy()
    viable = adaptive.loc[
        (adaptive["ppl_ratio_to_fixed"] <= 1.01)
        & (adaptive["iteration_fraction"] < 0.999)
    ]
    if viable.empty:
        return None
    best = viable.sort_values(
        ["iteration_fraction", "ppl_ratio_to_fixed", "wall_time_ratio_to_fixed"]
    ).iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "ppl_ratio_to_fixed": float(best["ppl_ratio_to_fixed"]),
        "avg_total_steps": float(best["avg_total_steps"]),
        "avg_stage_steps": [
            float(best["avg_stage1_steps"]),
            float(best["avg_stage2_steps"]),
            float(best["avg_stage3_steps"]),
        ],
        "iteration_fraction": float(best["iteration_fraction"]),
        "theoretical_iteration_saving": float(best["theoretical_iteration_saving"]),
        "wall_time_ratio_to_fixed": float(best["wall_time_ratio_to_fixed"]),
        "throughput_ratio_to_fixed": float(best["throughput_ratio_to_fixed"]),
        "runtime_speedup_observed": bool(best["wall_time_ratio_to_fixed"] < 1.0),
    }


def make_decision(frame: pd.DataFrame, source: Path) -> dict[str, object]:
    best = {model: select_best(group) for model, group in frame.groupby("model")}
    one_d = best.get("minicells-v2")
    two_d = best.get("minicells-2d-k4")
    if one_d is not None and two_d is not None:
        status = "HALTING_SIGNAL_BOTH"
        diagnosis = "BOTH_1D_AND_2D_SUPPORT_RESIDUAL_EARLY_EXIT_WITHIN_1PCT_PPL"
    elif two_d is not None:
        status = "HALTING_SIGNAL_2D_ONLY"
        diagnosis = "ONLY_2D_SUPPORTS_RESIDUAL_EARLY_EXIT_WITHIN_1PCT_PPL"
    elif one_d is not None:
        status = "HALTING_SIGNAL_1D_ONLY"
        diagnosis = "ONLY_1D_SUPPORTS_RESIDUAL_EARLY_EXIT_WITHIN_1PCT_PPL"
    else:
        status = "NO_HALTING_SIGNAL"
        diagnosis = "CURRENT_FIXED_DEPTH_MODELS_DO_NOT_SUPPORT_SAFE_RESIDUAL_EARLY_EXIT"

    return {
        "format": "minicells.language-adaptive-halting.v1",
        "experiment": "MINI Cells Experiment 010 — Residual Adaptive Halting Probe",
        "status": status,
        "diagnosis": diagnosis,
        "source_009": str(source.relative_to(ROOT)),
        "question": "Can the already-trained 1D and 2D cellular language models stop recurrent computation from their own state-update residual without retraining?",
        "halting_rule": {
            "signal": "absolute RMS update of the final-prefix readout position (all K tissue cells for 2D)",
            "thresholds": list(THRESHOLDS),
            "minimum_iterations_per_stage": MIN_ITERATIONS,
            "maximum_iterations_per_stage": 4,
            "scope": "one autoregressive prefix at a time; batch size 1; stage-level early exit",
            "quality_budget": "last-token next-token PPL <= 1.01x fixed-depth PPL",
        },
        "best": best,
        "interpretation": {
            "algorithmic": "A viable point shows the learned dynamics already expose a state-derived stopping signal; this does not require a learned halting head and is not specific to 2D.",
            "runtime": "Observed wall-clock includes GPU-to-host scalar synchronization in this prototype. Iteration savings can exist even if the current Python control flow is not faster yet.",
            "training": "This experiment probes frozen checkpoints. A positive result justifies a later training experiment with randomized depth/stability loss and causal per-position or per-cell adaptive unrolling; it does not by itself prove lower training cost.",
            "causality": "Only the prediction after the complete prefix is scored, so the stopping decision cannot inspect future target tokens. Training-time all-position loss requires a finer-grained causal scheduler.",
        },
    }


def save_plots(frame: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for model, group in frame.groupby("model"):
        ordered = group.sort_values("iteration_fraction")
        axis.plot(
            ordered["iteration_fraction"],
            ordered["ppl_ratio_to_fixed"],
            marker="o",
            label=model,
        )
    axis.axhline(1.0, linewidth=1, linestyle="--")
    axis.axhline(1.01, linewidth=1, linestyle=":", label="1% PPL budget")
    axis.set_xlabel("Executed recurrent-iteration fraction (fixed = 1.0)")
    axis.set_ylabel("Last-token PPL / fixed-depth PPL")
    axis.set_title("Experiment 010 — quality vs recurrent compute")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ppl-vs-iterations.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for model, group in frame.groupby("model"):
        ordered = group.sort_values("iteration_fraction")
        axis.plot(
            ordered["iteration_fraction"],
            ordered["wall_time_ratio_to_fixed"],
            marker="o",
            label=model,
        )
    axis.axhline(1.0, linewidth=1, linestyle="--")
    axis.set_xlabel("Executed recurrent-iteration fraction (fixed = 1.0)")
    axis.set_ylabel("Wall time / fixed-depth wall time")
    axis.set_title("Experiment 010 — measured prefix runtime vs recurrent compute")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "walltime-vs-iterations.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 010 requires CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    source = source_009()
    validation, tokenizer_path, manifest = prepare_validation(source)
    tokenizer = load_tokenizer(tokenizer_path)
    starts = fixed_validation_starts(
        int(validation.numel()),
        batches=VALIDATION_PREFIXES,
        batch_size=1,
        sequence_length=128,
        seed=10109,
    )
    models = load_models(source, tokenizer.get_vocab_size())
    device = torch.device("cuda:0")

    rows: list[dict[str, object]] = []
    for name in MODELS:
        model = models[name].to(device)
        fixed = evaluate_config(
            model,
            validation,
            starts,
            device=device,
            threshold=None,
        )
        rows.append({"model": name, **fixed})
        print(
            f"{name:18s} fixed       ppl={fixed['validation_last_token_ppl']:.3f} "
            f"steps={fixed['avg_total_steps']:.2f} prefix/s={fixed['prefixes_per_second']:.1f}"
        )
        for threshold in THRESHOLDS:
            result = evaluate_config(
                model,
                validation,
                starts,
                device=device,
                threshold=threshold,
            )
            rows.append({"model": name, **result})
            print(
                f"{name:18s} eps={threshold:0.4f} "
                f"ppl={result['validation_last_token_ppl']:.3f} "
                f"steps={result['avg_total_steps']:.2f} "
                f"prefix/s={result['prefixes_per_second']:.1f}"
            )
        model.to("cpu")
        torch.cuda.empty_cache()

    frame = add_relative_metrics(pd.DataFrame(rows))
    frame.to_csv(OUT / "halting-sweep.csv", index=False)
    decision = make_decision(frame, source)
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task = {
        "format": "minicells.language-adaptive-halting-task.v1",
        "source_experiment": "009",
        "models": list(MODELS),
        "frozen_checkpoints": True,
        "evaluation": "192 independent 128-token prefixes, batch size 1, score only the next token",
        "validation_prefixes": VALIDATION_PREFIXES,
        "thresholds": list(THRESHOLDS),
        "min_iterations": MIN_ITERATIONS,
        "max_iterations": 4,
        "corpus_validation_sha256": manifest.get("validation_token_sha256"),
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(task, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_plots(frame)
    print("=== decision ===")
    print(json.dumps(decision, indent=2))
    print("=== sweep ===")
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
