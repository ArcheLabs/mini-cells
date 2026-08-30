from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")

sys.path.insert(0, str(ROOT / "src"))

from minicells.continual_learning import (  # noqa: E402
    MARGIN_Q,
    MARKER,
    PARAMETER_COUNT,
    PERTURBATION_Q,
    STEP_Q,
    apply_update,
    build_adaptation_pool,
    build_old_pool,
    candidate,
    delta_vector,
    evaluate_model,
    load_q88_model,
    margin_loss,
    model_hash,
    save_q88_model,
    training_batch,
)
from minicells.vocab import CharVocab  # noqa: E402

OUT = ROOT / "results" / "native-continual-learning-v1"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_DIR = ROOT / "artifacts" / "experiments" / "003b-quantization-localization"
SOURCE_MODEL = SOURCE_DIR / "solved-q88-model.bin"
SOURCE_METADATA = SOURCE_DIR / "metadata.json"

FULL_RUN = True
GENERATIONS = 1200 if FULL_RUN else 120
EVAL_EVERY = 100 if FULL_RUN else 20
TRAIN_POOL_EXAMPLES = 2048 if FULL_RUN else 256
PROBE_EXAMPLES = 128 if FULL_RUN else 64
FINAL_EXAMPLES = 512 if FULL_RUN else 128
OLD_RETENTION_GATE = 0.95
NEW_LEARNING_GATE = 0.50
NEW_IMPROVEMENT_GATE = 0.40
EARLY_SUCCESS_GATE = 0.90

print(
    {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "exact_training_device": "cpu/int64",
        "full_run": FULL_RUN,
        "generations": GENERATIONS,
    }
)


def verify_source_artifact() -> tuple[torch.Tensor, str]:
    if not SOURCE_MODEL.is_file() or not SOURCE_METADATA.is_file():
        raise FileNotFoundError(
            "Experiment 003B artifacts are missing from main. Merge the curated 003B result first."
        )
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    expected = None
    for item in metadata.get("files", []):
        if item.get("path") == SOURCE_MODEL.name:
            expected = item.get("sha256")
            break
    actual = hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest()
    if expected is None or actual != expected:
        raise RuntimeError(
            f"003B solved model provenance mismatch: expected={expected!r} actual={actual!r}"
        )
    flat = load_q88_model(SOURCE_MODEL)
    if flat.numel() != PARAMETER_COUNT:
        raise RuntimeError("invalid source model")
    return flat, actual


start_model, source_sha256 = verify_source_artifact()
vocab = CharVocab()

old_train = build_old_pool(vocab, seed=30031, examples=TRAIN_POOL_EXAMPLES)
new_train = build_adaptation_pool(vocab, seed=30032, examples=TRAIN_POOL_EXAMPLES)
old_probe = build_old_pool(vocab, seed=30131, examples=PROBE_EXAMPLES)
new_probe = build_adaptation_pool(vocab, seed=30132, examples=PROBE_EXAMPLES)
old_final = build_old_pool(vocab, seed=30231, examples=FINAL_EXAMPLES)
new_final = build_adaptation_pool(vocab, seed=30232, examples=FINAL_EXAMPLES)

initial_old = evaluate_model(start_model, old_final)
initial_new = evaluate_model(start_model, new_final)
initial_new_changed = initial_new["changed_accuracy"]
print("initial_old", initial_old)
print("initial_new", initial_new)
if initial_old["token_accuracy"] < 0.99 or initial_old["exact_sequence_accuracy"] < 0.99:
    raise RuntimeError("003B solved model does not preserve >=99% Echo on the 003C control set")


def evaluate_pair(model: torch.Tensor, probe: bool = True) -> dict[str, float]:
    old = evaluate_model(model, old_probe if probe else old_final)
    new = evaluate_model(model, new_probe if probe else new_final)
    return {
        "old_token_accuracy": old["token_accuracy"],
        "old_exact_sequence_accuracy": old["exact_sequence_accuracy"],
        "old_margin_loss_per_token": old["margin_loss_per_token"],
        "new_token_accuracy": new["token_accuracy"],
        "new_exact_sequence_accuracy": new["exact_sequence_accuracy"],
        "new_changed_accuracy": new["changed_accuracy"],
        "new_changed_improvement": new["changed_accuracy"] - initial_new_changed,
        "new_margin_loss_per_token": new["margin_loss_per_token"],
    }


def run_native(
    name: str,
    mode: str,
    block_size: int,
    allow_early_success: bool,
) -> dict[str, object]:
    model = start_model.clone()
    best_model = model.clone()
    rows: list[dict[str, object]] = []
    updates = 0
    started = time.time()

    initial = evaluate_pair(model, probe=True)
    rows.append(
        {
            "generation": 0,
            "updated": False,
            "direction": 0,
            "plus_loss": None,
            "minus_loss": None,
            **initial,
        }
    )
    best_metrics = dict(initial)
    best_generation = 0
    success_streak = 0

    for generation in range(1, GENERATIONS + 1):
        parent_hash = model_hash(model)
        batch = training_batch(mode, old_train, new_train, parent_hash, generation, batch_size=4)
        delta = delta_vector(parent_hash, generation, block_size)
        plus = candidate(model, delta, 1, PERTURBATION_Q)
        minus = candidate(model, delta, -1, PERTURBATION_Q)
        plus_loss = margin_loss(plus, batch, MARGIN_Q)
        minus_loss = margin_loss(minus, batch, MARGIN_Q)
        model, updated, direction = apply_update(
            model, delta, plus_loss, minus_loss, STEP_Q
        )
        updates += int(updated)

        should_eval = generation % EVAL_EVERY == 0 or generation == GENERATIONS
        if should_eval:
            metrics = evaluate_pair(model, probe=True)
            rows.append(
                {
                    "generation": generation,
                    "updated": updated,
                    "direction": direction,
                    "plus_loss": plus_loss,
                    "minus_loss": minus_loss,
                    **metrics,
                }
            )
            print(
                f"{name:24s} gen={generation:4d} update_rate={updates / generation:.3f} "
                f"old={metrics['old_token_accuracy']:.2%} "
                f"new_changed={metrics['new_changed_accuracy']:.2%} "
                f"improvement={metrics['new_changed_improvement']:+.2%}"
            )

            eligible = metrics["old_token_accuracy"] >= OLD_RETENTION_GATE
            best_eligible = best_metrics["old_token_accuracy"] >= OLD_RETENTION_GATE
            if eligible and (
                not best_eligible
                or metrics["new_changed_accuracy"] > best_metrics["new_changed_accuracy"]
            ):
                best_model = model.clone()
                best_metrics = dict(metrics)
                best_generation = generation

            if (
                allow_early_success
                and eligible
                and metrics["new_changed_accuracy"] >= EARLY_SUCCESS_GATE
                and metrics["new_changed_improvement"] >= NEW_IMPROVEMENT_GATE
            ):
                success_streak += 1
            else:
                success_streak = 0
            if allow_early_success and success_streak >= 2:
                print(f"{name}: early success after {generation} generations")
                break

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"{name}.csv", index=False)
    final_metrics = evaluate_pair(model, probe=False)
    best_full_metrics = evaluate_pair(best_model, probe=False)
    final_hash = model_hash(model).hex()
    best_hash = model_hash(best_model).hex()
    return {
        "name": name,
        "mode": mode,
        "block_size": block_size,
        "generations_completed": int(frame["generation"].max()),
        "updates": updates,
        "update_rate": updates / max(1, int(frame["generation"].max())),
        "seconds": time.time() - started,
        "final": final_metrics,
        "best_retained": best_full_metrics,
        "best_generation": best_generation,
        "final_model_hash": final_hash,
        "best_model_hash": best_hash,
        "final_model": model,
        "best_model": best_model,
    }


run_specs = [
    ("stability-global", "old-only", PARAMETER_COUNT, False),
    ("stability-block512", "old-only", 512, False),
    ("adapt-replay-global", "replay", PARAMETER_COUNT, True),
    ("adapt-new-block512", "new-only", 512, True),
    ("adapt-replay-block512", "replay", 512, True),
]

runs = [run_native(*spec) for spec in run_specs]

summary_rows = []
for run in runs:
    row = {
        "name": run["name"],
        "mode": run["mode"],
        "block_size": run["block_size"],
        "generations_completed": run["generations_completed"],
        "updates": run["updates"],
        "update_rate": run["update_rate"],
        "seconds": run["seconds"],
    }
    row.update({f"final_{k}": v for k, v in run["final"].items()})
    row.update({f"best_{k}": v for k, v in run["best_retained"].items()})
    summary_rows.append(row)
pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)

stability_runs = [run for run in runs if run["mode"] == "old-only"]
adaptation_runs = [run for run in runs if run["mode"] != "old-only"]
stability_pass = any(
    run["final"]["old_token_accuracy"] >= OLD_RETENTION_GATE for run in stability_runs
)
continual_candidates = [
    run
    for run in adaptation_runs
    if run["final"]["old_token_accuracy"] >= OLD_RETENTION_GATE
    and run["final"]["new_changed_accuracy"] >= NEW_LEARNING_GATE
    and run["final"]["new_changed_improvement"] >= NEW_IMPROVEMENT_GATE
]
catastrophic_signal = any(
    run["final"]["new_changed_accuracy"] >= NEW_LEARNING_GATE
    and run["final"]["new_changed_improvement"] >= NEW_IMPROVEMENT_GATE
    and run["final"]["old_token_accuracy"] < OLD_RETENTION_GATE
    for run in adaptation_runs
)

if continual_candidates:
    winner = max(
        continual_candidates,
        key=lambda run: (
            run["final"]["new_changed_accuracy"],
            run["final"]["old_token_accuracy"],
        ),
    )
    diagnosis = "NATIVE_CONTINUAL_LEARNING_DEMONSTRATED"
    status = "PASS"
else:
    winner = max(
        adaptation_runs,
        key=lambda run: (
            run["best_retained"]["new_changed_accuracy"]
            if run["best_retained"]["old_token_accuracy"] >= OLD_RETENTION_GATE
            else -1.0,
            run["best_retained"]["old_token_accuracy"],
        ),
    )
    best_new = winner["best_retained"]["new_changed_accuracy"]
    best_improvement = winner["best_retained"]["new_changed_improvement"]
    best_old = winner["best_retained"]["old_token_accuracy"]
    if not stability_pass:
        diagnosis = "NATIVE_UPDATE_STABILITY_BOTTLENECK"
    elif catastrophic_signal:
        diagnosis = "CATASTROPHIC_FORGETTING"
    elif best_new >= 0.20 and best_improvement >= 0.15 and best_old >= OLD_RETENTION_GATE:
        diagnosis = "CONTINUAL_SIGNAL_BELOW_GATE"
    else:
        diagnosis = "ADAPTATION_NOT_LEARNED"
    status = "NEEDS_ITERATION"

winner_model = winner["final_model"] if status == "PASS" else winner["best_model"]
save_q88_model(OUT / "best-continual-q88-model.bin", winner_model)

serializable_runs = []
for run in runs:
    serializable_runs.append(
        {
            key: value
            for key, value in run.items()
            if key not in {"final_model", "best_model"}
        }
    )

task_spec = {
    "format": "minicells.continual-task.v1",
    "old_domain": "Echo inputs excluding the reserved ?? prefix",
    "adaptation_domain": "Inputs prefixed with ??",
    "adaptation_rule": (
        "Echo every token except the first payload token after ??. "
        "That token is shifted by one in abcdefghijklmnopqrstuvwxyz0123456789, cyclically."
    ),
    "changed_position": 2,
    "marker": MARKER,
    "training_batch_size": 4,
    "replay_mix": {"old": 2, "adaptation": 2},
    "margin_q": MARGIN_Q,
    "perturbation_q": PERTURBATION_Q,
    "step_q": STEP_Q,
    "seeds": {
        "old_train": 30031,
        "adaptation_train": 30032,
        "old_probe": 30131,
        "adaptation_probe": 30132,
        "old_final": 30231,
        "adaptation_final": 30232,
    },
}
(OUT / "task-spec.json").write_text(json.dumps(task_spec, indent=2) + "\n", encoding="utf-8")

decision = {
    "format": "minicells.native-continual-learning.v1",
    "experiment": "MINI Cells Experiment 003C — Native Continual Learning",
    "status": status,
    "diagnosis": diagnosis,
    "gates": {
        "source_echo_solved": initial_old["token_accuracy"] >= 0.99,
        "adaptation_baseline_changed_accuracy": initial_new_changed,
        "native_update_stability": stability_pass,
        "old_retention_gate": OLD_RETENTION_GATE,
        "new_changed_learning_gate": NEW_LEARNING_GATE,
        "new_changed_improvement_gate": NEW_IMPROVEMENT_GATE,
        "continual_learning_demonstrated": bool(continual_candidates),
    },
    "source": {
        "artifact": str(SOURCE_MODEL.relative_to(ROOT)),
        "sha256": source_sha256,
        "model_hash": model_hash(start_model).hex(),
    },
    "initial": {"old": initial_old, "adaptation": initial_new},
    "winner": {
        "name": winner["name"],
        "selected_metrics": winner["final"] if status == "PASS" else winner["best_retained"],
        "selected_generation": (
            winner["generations_completed"] if status == "PASS" else winner["best_generation"]
        ),
    },
    "runs": serializable_runs,
    "run": {
        "full_run": FULL_RUN,
        "generations": GENERATIONS,
        "eval_every": EVAL_EVERY,
        "train_pool_examples": TRAIN_POOL_EXAMPLES,
        "probe_examples": PROBE_EXAMPLES,
        "final_examples": FINAL_EXAMPLES,
        "exact_training_device": "cpu/int64",
    },
}
(OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

print("=== decision.json ===")
print(json.dumps(decision, indent=2))
print("=== files ===")
for path in sorted(OUT.iterdir()):
    if path.is_file():
        print(path.name, path.stat().st_size)
