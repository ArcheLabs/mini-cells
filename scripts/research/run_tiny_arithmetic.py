from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.arithmetic_optimizer import guarded_spsa_step, select_arithmetic_batch  # noqa: E402
from minicells.arithmetic_tasks import (  # noqa: E402
    all_arithmetic_examples,
    arithmetic_batch,
    evaluate_float_arithmetic,
    evaluate_integer_arithmetic,
    float_arithmetic_loss,
    float_echo_loss,
    load_flat_into_float_model,
    quantize_float_model,
    split_arithmetic_examples,
)
from minicells.arithmetic_visuals import (  # noqa: E402
    save_capability_summary,
    save_learning_curves,
    save_operation_heatmap,
    save_retention_capability,
)
from minicells.continual_learning import (  # noqa: E402
    build_old_pool,
    evaluate_model,
    load_q88_model,
    model_hash,
    save_q88_model,
    select_indices,
)
from minicells.vocab import CharVocab  # noqa: E402

OUT = ROOT / "results" / "tiny-arithmetic-v1"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
SOURCE_DIR = ROOT / "artifacts" / "experiments" / "003b-quantization-localization"
SOURCE_MODEL = SOURCE_DIR / "solved-q88-model.bin"
SOURCE_METADATA = SOURCE_DIR / "metadata.json"

FULL_RUN = True
FP32_STEPS = 1800 if FULL_RUN else 200
FP32_EVAL_EVERY = 100 if FULL_RUN else 20
NATIVE_GENERATIONS = 1600 if FULL_RUN else 160
NATIVE_EVAL_EVERY = 100 if FULL_RUN else 20
OLD_RETENTION_GATE = 0.95
NATIVE_TRAIN_GATE = 0.80
NATIVE_HELDOUT_GATE = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "fp32_device": str(DEVICE),
    "native_device": "cpu/int64",
    "full_run": FULL_RUN,
})


def verify_source() -> tuple[torch.Tensor, str]:
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    expected = next(
        item["sha256"] for item in metadata["files"] if item["path"] == SOURCE_MODEL.name
    )
    actual = hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"003B model SHA-256 mismatch: {actual} != {expected}")
    return load_q88_model(SOURCE_MODEL), actual


def float_echo_metrics(model, batch) -> dict[str, float]:
    with torch.no_grad():
        logits = model(batch.input_ids.to(DEVICE)).cpu()
    pred = logits.argmax(dim=-1)
    active = batch.mask.bool()
    correct = int(((pred == batch.target_ids) & active).sum().item())
    total = int(active.sum().item())
    exact = int((((pred == batch.target_ids) | (~active)).all(dim=1)).sum().item())
    return {
        "token_accuracy": correct / total,
        "exact_sequence_accuracy": exact / batch.size,
    }


start_model, source_sha256 = verify_source()
vocab = CharVocab()
train_examples, heldout_examples = split_arithmetic_examples()
all_examples = all_arithmetic_examples()
train_batch = arithmetic_batch(vocab, train_examples)
heldout_batch = arithmetic_batch(vocab, heldout_examples)
add_batch = arithmetic_batch(vocab, [x for x in all_examples if x.operation == "add"])
sub_batch = arithmetic_batch(vocab, [x for x in all_examples if x.operation == "sub"])
old_train = build_old_pool(vocab, seed=41001, examples=2048 if FULL_RUN else 256)
old_probe = build_old_pool(vocab, seed=41002, examples=256 if FULL_RUN else 64)
old_final = build_old_pool(vocab, seed=41003, examples=512 if FULL_RUN else 128)
echo_anchor = old_train.take(list(range(16)))

split_rows = [
    {
        "split": "train" if item in train_examples else "heldout",
        "operation": item.operation,
        "left": item.left,
        "right": item.right,
        "answer": item.answer,
        "expression": item.expression,
    }
    for item in all_examples
]
pd.DataFrame(split_rows).to_csv(OUT / "arithmetic-split.csv", index=False)

initial_old = evaluate_model(start_model, old_final)
initial_train = evaluate_integer_arithmetic(start_model, train_batch)["answer_accuracy"]
initial_heldout = evaluate_integer_arithmetic(start_model, heldout_batch)["answer_accuracy"]
print("initial", initial_old, initial_train, initial_heldout)

# Gate 1: can the same architecture acquire arithmetic with conventional optimization?
float_model = load_flat_into_float_model(start_model).to(DEVICE)
optimizer = torch.optim.Adam(float_model.parameters(), lr=2e-3)
rng = random.Random(42004)
capacity_rows = []
for step in range(1, FP32_STEPS + 1):
    arithmetic_indices = [rng.randrange(len(train_examples)) for _ in range(32)]
    arithmetic = arithmetic_batch(vocab, [train_examples[i] for i in arithmetic_indices])
    echo_indices = [rng.randrange(old_train.size) for _ in range(32)]
    echo = old_train.take(echo_indices)
    optimizer.zero_grad(set_to_none=True)
    loss_echo = float_echo_loss(float_model, echo, DEVICE)
    loss_arithmetic = float_arithmetic_loss(float_model, arithmetic, DEVICE)
    loss = loss_echo + 4.0 * loss_arithmetic
    loss.backward()
    optimizer.step()

    if step % FP32_EVAL_EVERY == 0 or step == FP32_STEPS:
        old = float_echo_metrics(float_model, old_probe)
        train_acc = evaluate_float_arithmetic(float_model, train_batch, DEVICE)["answer_accuracy"]
        heldout_acc = evaluate_float_arithmetic(float_model, heldout_batch, DEVICE)["answer_accuracy"]
        capacity_rows.append({
            "generation": step,
            "loss": float(loss.item()),
            "old_token_accuracy": old["token_accuracy"],
            "train_answer_accuracy": train_acc,
            "heldout_answer_accuracy": heldout_acc,
        })
        print(
            f"fp32 step={step:4d} old={old['token_accuracy']:.2%} "
            f"train={train_acc:.2%} heldout={heldout_acc:.2%}"
        )
capacity_frame = pd.DataFrame(capacity_rows)
capacity_frame.to_csv(OUT / "fp32-capacity.csv", index=False)
save_learning_curves(capacity_frame, OUT / "capacity-learning-curves.png")

float_final_old = float_echo_metrics(float_model, old_final)
float_train = evaluate_float_arithmetic(float_model, train_batch, DEVICE)["answer_accuracy"]
float_heldout = evaluate_float_arithmetic(float_model, heldout_batch, DEVICE)["answer_accuracy"]
float_q88 = quantize_float_model(float_model)
save_q88_model(OUT / "fp32-arithmetic-q88-model.bin", float_q88)
q88_old = evaluate_model(float_q88, old_final)
q88_train = evaluate_integer_arithmetic(float_q88, train_batch)["answer_accuracy"]
q88_heldout = evaluate_integer_arithmetic(float_q88, heldout_batch)["answer_accuracy"]
print("float_final", float_final_old, float_train, float_heldout)
print("q88_bridge", q88_old, q88_train, q88_heldout)


def native_metrics(flat: torch.Tensor, old_batch=old_probe) -> dict[str, float]:
    old = evaluate_model(flat, old_batch)
    return {
        "old_token_accuracy": old["token_accuracy"],
        "train_answer_accuracy": evaluate_integer_arithmetic(flat, train_batch)["answer_accuracy"],
        "heldout_answer_accuracy": evaluate_integer_arithmetic(flat, heldout_batch)["answer_accuracy"],
        "addition_answer_accuracy": evaluate_integer_arithmetic(flat, add_batch)["answer_accuracy"],
        "subtraction_answer_accuracy": evaluate_integer_arithmetic(flat, sub_batch)["answer_accuracy"],
    }


def run_native(name: str, block_size: int, arithmetic_weight: int) -> dict[str, object]:
    model = start_model.clone()
    best_model = model.clone()
    best = native_metrics(model)
    best_generation = 0
    accepted = 0
    rows = [{"generation": 0, "accepted": False, **best}]
    started = time.time()
    success_streak = 0

    for generation in range(1, NATIVE_GENERATIONS + 1):
        parent_hash = model_hash(model)
        old_indices = select_indices(old_train.size, 2, "arith-echo", parent_hash, generation)
        echo = old_train.take(old_indices)
        arithmetic = select_arithmetic_batch(
            vocab, train_examples, 2, parent_hash, generation
        )
        model, step = guarded_spsa_step(
            model,
            echo,
            arithmetic,
            echo_anchor,
            parent_hash,
            generation,
            block_size,
            arithmetic_weight=arithmetic_weight,
        )
        accepted += int(step["accepted"])

        if generation % NATIVE_EVAL_EVERY == 0 or generation == NATIVE_GENERATIONS:
            metrics = native_metrics(model)
            rows.append({
                "generation": generation,
                "accepted": step["accepted"],
                "accept_rate": accepted / generation,
                **metrics,
            })
            print(
                f"{name:18s} gen={generation:4d} accept={accepted / generation:.3f} "
                f"old={metrics['old_token_accuracy']:.2%} "
                f"train={metrics['train_answer_accuracy']:.2%} "
                f"heldout={metrics['heldout_answer_accuracy']:.2%}"
            )
            eligible = metrics["old_token_accuracy"] >= OLD_RETENTION_GATE
            best_eligible = best["old_token_accuracy"] >= OLD_RETENTION_GATE
            if eligible and (
                not best_eligible
                or (metrics["heldout_answer_accuracy"], metrics["train_answer_accuracy"])
                > (best["heldout_answer_accuracy"], best["train_answer_accuracy"])
            ):
                best_model = model.clone()
                best = metrics
                best_generation = generation
            if (
                eligible
                and metrics["train_answer_accuracy"] >= 0.90
                and metrics["heldout_answer_accuracy"] >= NATIVE_HELDOUT_GATE
            ):
                success_streak += 1
            else:
                success_streak = 0
            if success_streak >= 2:
                break

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"native-{name}.csv", index=False)
    return {
        "name": name,
        "block_size": block_size,
        "arithmetic_weight": arithmetic_weight,
        "generations_completed": int(frame["generation"].max()),
        "accepted_updates": accepted,
        "accept_rate": accepted / max(1, int(frame["generation"].max())),
        "seconds": time.time() - started,
        "best_generation": best_generation,
        "best": best,
        "best_model": best_model,
        "frame": frame,
    }


native_specs = [
    ("block512-w4", 512, 4),
    ("block256-w4", 256, 4),
    ("block128-w4", 128, 4),
    ("block256-w8", 256, 8),
]
runs = [run_native(*spec) for spec in native_specs]
retained = [run for run in runs if run["best"]["old_token_accuracy"] >= OLD_RETENTION_GATE]
winner = max(
    retained or runs,
    key=lambda run: (
        run["best"]["heldout_answer_accuracy"],
        run["best"]["train_answer_accuracy"],
        run["best"]["old_token_accuracy"],
    ),
)
selected_model = winner["best_model"]
selected = native_metrics(selected_model, old_final)
save_q88_model(OUT / "best-native-arithmetic-q88-model.bin", selected_model)
save_learning_curves(winner["frame"], OUT / "learning-curves.png")

summary_rows = [
    {
        "name": run["name"],
        "block_size": run["block_size"],
        "arithmetic_weight": run["arithmetic_weight"],
        "generations_completed": run["generations_completed"],
        "accepted_updates": run["accepted_updates"],
        "accept_rate": run["accept_rate"],
        "seconds": run["seconds"],
        **run["best"],
    }
    for run in runs
]
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "native-summary.csv", index=False)
save_retention_capability(summary, OUT / "retention-vs-capability.png")
save_operation_heatmap(selected_model, vocab, all_examples, "add", OUT / "addition-heatmap.png")
save_operation_heatmap(selected_model, vocab, all_examples, "sub", OUT / "subtraction-heatmap.png")
save_capability_summary(selected, OUT / "capability-summary.png")

capacity_pass = float_train >= 0.95
q88_pass = q88_train >= 0.85 and q88_old["token_accuracy"] >= OLD_RETENTION_GATE
native_acquired = (
    selected["old_token_accuracy"] >= OLD_RETENTION_GATE
    and selected["train_answer_accuracy"] >= NATIVE_TRAIN_GATE
)
native_generalized = native_acquired and selected["heldout_answer_accuracy"] >= NATIVE_HELDOUT_GATE
if not capacity_pass:
    status, diagnosis = "NEEDS_ITERATION", "FLOAT_ARITHMETIC_CAPACITY_NOT_LEARNED"
elif not q88_pass:
    status, diagnosis = "NEEDS_ITERATION", "Q88_ARITHMETIC_CAPACITY_NOT_PRESERVED"
elif not native_acquired:
    status, diagnosis = "NEEDS_ITERATION", "NATIVE_ARITHMETIC_NOT_ACQUIRED"
elif not native_generalized:
    status, diagnosis = "NEEDS_ITERATION", "NATIVE_ARITHMETIC_MEMORIZATION_ONLY"
else:
    status, diagnosis = "PASS", "TINY_ARITHMETIC_GENERALIZATION_DEMONSTRATED"

task_spec = {
    "format": "minicells.tiny-arithmetic-task.v1",
    "examples": 110,
    "train_examples": 88,
    "heldout_examples": 22,
    "addition_rule": "a+b<=9, formatted as '<a>plus<b>?', answer at the ? cell",
    "subtraction_rule": "a>=b, formatted as '<a>minus<b>?', answer at the ? cell",
    "answer_vocabulary": "0123456789",
    "random_digit_baseline": 0.10,
    "split_seed": 4004,
    "old_retention_gate": OLD_RETENTION_GATE,
    "native_train_gate": NATIVE_TRAIN_GATE,
    "native_heldout_gate": NATIVE_HELDOUT_GATE,
}
(OUT / "task-spec.json").write_text(json.dumps(task_spec, indent=2) + "\n", encoding="utf-8")

decision = {
    "format": "minicells.tiny-arithmetic.v1",
    "experiment": "MINI Cells Experiment 004 — Tiny Arithmetic Capability",
    "status": status,
    "diagnosis": diagnosis,
    "gates": {
        "source_echo_solved": initial_old["token_accuracy"] >= 0.99,
        "fp32_train_capacity": capacity_pass,
        "fp32_heldout_accuracy": float_heldout,
        "q88_capacity_preserved": q88_pass,
        "native_arithmetic_acquired": native_acquired,
        "native_generalization_demonstrated": native_generalized,
    },
    "source": {
        "artifact": str(SOURCE_MODEL.relative_to(ROOT)),
        "sha256": source_sha256,
        "model_hash": model_hash(start_model).hex(),
    },
    "initial": {
        "echo": initial_old,
        "train_answer_accuracy": initial_train,
        "heldout_answer_accuracy": initial_heldout,
    },
    "fp32": {
        "echo": float_final_old,
        "train_answer_accuracy": float_train,
        "heldout_answer_accuracy": float_heldout,
    },
    "q88_bridge": {
        "echo": q88_old,
        "train_answer_accuracy": q88_train,
        "heldout_answer_accuracy": q88_heldout,
    },
    "native_winner": {
        "name": winner["name"],
        "generation": winner["best_generation"],
        "metrics": selected,
    },
    "native_runs": [
        {key: value for key, value in run.items() if key not in {"best_model", "frame"}}
        for run in runs
    ],
    "run": {
        "fp32_steps": FP32_STEPS,
        "native_generations": NATIVE_GENERATIONS,
        "fp32_device": str(DEVICE),
        "native_device": "cpu/int64",
    },
}
(OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

print("=== decision.json ===")
print(json.dumps(decision, indent=2))
print("=== files ===")
for path in sorted(OUT.iterdir()):
    print(path.name, path.stat().st_size)
