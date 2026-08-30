from __future__ import annotations

import json
import platform
import struct
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.config import load_config
from minicells.data import fixed_dataset
from minicells.quantization_localization import (
    PARAM_MAX, PARAM_MIN, evaluate_exact_integer, flatten_integer_parameters,
    forward_variant, metrics_from_logits, q88_param,
)
from minicells.train import load_checkpoint, train
from minicells.vocab import CharVocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = ROOT / "results" / "quantization-localization-v1"
OUT.mkdir(parents=True, exist_ok=True)
FULL_RUN = True
SEED = 3
TRAIN_STEPS = 1500 if FULL_RUN else 300
VAL_EXAMPLES = 2048 if FULL_RUN else 512
EXACT_EXAMPLES = 512 if FULL_RUN else 128

print({
    "python": platform.python_version(), "torch": torch.__version__,
    "cuda": torch.version.cuda, "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "full_run": FULL_RUN,
})

cfg = deepcopy(load_config(ROOT / "configs" / "echo-v0.yaml"))
cfg["train"]["seed"] = SEED
cfg["train"]["steps"] = TRAIN_STEPS
cfg["train"]["eval_every"] = 100 if FULL_RUN else 50
cfg["train"]["checkpoint_every"] = TRAIN_STEPS
cfg["validation"]["seed"] = 10003
cfg["validation"]["examples"] = VAL_EXAMPLES
cfg["output"]["root"] = str(OUT / "fp32")
report = train(cfg, device=str(DEVICE))
model, payload = load_checkpoint(OUT / "fp32" / "checkpoints" / "final.pt", device=str(DEVICE))
fp32_metrics = payload["metrics"]
if fp32_metrics["token_accuracy"] < 0.99:
    raise RuntimeError("FP32 positive control did not solve Echo; localization is invalid")

vocab = CharVocab()
data_args = dict(
    min_length=cfg["data"]["min_length"], max_length=cfg["data"]["max_length"],
    num_cells=cfg["model"]["num_cells"], random_fraction=cfg["data"]["random_fraction"],
)
validation = fixed_dataset(vocab, seed=10003, examples=VAL_EXAMPLES, **data_args).to(DEVICE)

rows, values, errors, clips = [], [], [], []
with torch.no_grad():
    for name, p in model.named_parameters():
        x = p.detach().cpu().float()
        q = q88_param(x)
        clipped = (x < PARAM_MIN) | (x > PARAM_MAX)
        rows.append({
            "tensor": name, "count": x.numel(), "min": float(x.min()), "max": float(x.max()),
            "mean_abs": float(x.abs().mean()), "clip_fraction": float(clipped.float().mean()),
            "q88_mae": float((q - x).abs().mean()), "q88_max_error": float((q - x).abs().max()),
        })
        values.append(x.reshape(-1)); errors.append((q - x).reshape(-1)); clips.append(clipped.reshape(-1))
pd.DataFrame(rows).to_csv(OUT / "parameter-diagnostics.csv", index=False)
values = torch.cat(values); errors = torch.cat(errors); clips = torch.cat(clips)
parameter_summary = {
    "count": int(values.numel()), "min": float(values.min()), "max": float(values.max()),
    "mean_abs": float(values.abs().mean()), "clip_fraction": float(clips.float().mean()),
    "q88_mae": float(errors.abs().mean()), "q88_rmse": float(torch.sqrt((errors ** 2).mean())),
    "q88_max_error": float(errors.abs().max()),
}
print("parameter_summary", parameter_summary)

variants = [
    ("fp32", dict(weight_mode="fp32", linear_frac_bits=None, state_frac_bits=None)),
    ("clip-only", dict(weight_mode="clip", linear_frac_bits=None, state_frac_bits=None)),
    ("weights-q88", dict(weight_mode="q88", linear_frac_bits=None, state_frac_bits=None)),
    ("weights-q88-linear-q88", dict(weight_mode="q88", linear_frac_bits=8, state_frac_bits=None)),
    ("weights-q88-state-q88", dict(weight_mode="q88", linear_frac_bits=None, state_frac_bits=8)),
    ("production-like-q88", dict(weight_mode="q88", linear_frac_bits=8, state_frac_bits=8)),
]
ablation, predictions = [], {}
for name, kwargs in variants:
    logits = forward_variant(model, validation.input_ids, **kwargs)
    metrics, pred = metrics_from_logits(logits, validation.target_ids, validation.mask)
    ablation.append({"variant": name, **metrics, **kwargs})
    predictions[name] = pred.detach().cpu()
    print(name, metrics)
pd.DataFrame(ablation).to_csv(OUT / "forward-ablation.csv", index=False)

precision = []
for bits in (8, 10, 12, 14):
    logits = forward_variant(model, validation.input_ids, weight_mode="q88",
                             linear_frac_bits=bits, state_frac_bits=bits)
    metrics, _ = metrics_from_logits(logits, validation.target_ids, validation.mask)
    precision.append({"internal_frac_bits": bits, **metrics})
    print("internal", bits, metrics)
logits = forward_variant(model, validation.input_ids, weight_mode="q88",
                         linear_frac_bits=None, state_frac_bits=None)
metrics, _ = metrics_from_logits(logits, validation.target_ids, validation.mask)
precision.append({"internal_frac_bits": "float", **metrics})
pd.DataFrame(precision).to_csv(OUT / "internal-precision-sweep.csv", index=False)

exact_metrics, exact_pred, qparams = evaluate_exact_integer(model, validation, EXACT_EXAMPLES)
mask = validation.mask[:EXACT_EXAMPLES].detach().cpu().bool()
float_pred = predictions["production-like-q88"][:EXACT_EXAMPLES]
agreement = (float_pred[mask] == exact_pred[mask]).float().mean().item()
print("exact_integer", exact_metrics, "float_exact_agreement", agreement)

flat = flatten_integer_parameters(qparams)
(OUT / "solved-q88-model.bin").write_bytes(struct.pack("<4476h", *map(int, flat.tolist())))
counts = torch.bincount(exact_pred[mask], minlength=len(vocab))
top_outputs = []
for token_id in torch.argsort(counts, descending=True)[:10].tolist():
    count = int(counts[token_id])
    if count:
        top_outputs.append({
            "token_id": token_id, "token": vocab.id_to_token[token_id],
            "count": count, "fraction": count / int(mask.sum()),
        })
pd.DataFrame(top_outputs).to_csv(OUT / "exact-output-frequency.csv", index=False)

by_name = {x["variant"]: x for x in ablation}
fp32 = by_name["fp32"]["token_accuracy"]
clip = by_name["clip-only"]["token_accuracy"]
weights = by_name["weights-q88"]["token_accuracy"]
prod = by_name["production-like-q88"]["token_accuracy"]
exact = exact_metrics["token_accuracy"]
preserve = 0.95
if fp32 < 0.99:
    diagnosis = "FP32_BASELINE_FAILED"
elif clip < preserve:
    diagnosis = "PARAMETER_RANGE_BOTTLENECK"
elif weights < preserve:
    diagnosis = "WEIGHT_QUANTIZATION_RESOLUTION_BOTTLENECK"
elif prod < preserve:
    diagnosis = "INTERNAL_ACTIVATION_PRECISION_BOTTLENECK"
elif exact < preserve or agreement < 0.999:
    diagnosis = "INTEGER_BRIDGE_SEMANTICS_MISMATCH"
else:
    diagnosis = "Q88_REPRESENTATION_PRESERVES_SOLUTION_TRAINING_DYNAMICS_BOTTLENECK"

decision = {
    "format": "minicells.quantization-localization.v1",
    "experiment": "MINI Cells Experiment 003B — Quantization Localization",
    "status": "PASS" if diagnosis.endswith("TRAINING_DYNAMICS_BOTTLENECK") else "NEEDS_ITERATION",
    "diagnosis": diagnosis,
    "gates": {
        "fp32_solved": fp32 >= 0.99,
        "parameter_range_preserves_solution": clip >= preserve,
        "q88_weights_preserve_solution": weights >= preserve,
        "production_like_preserves_solution": prod >= preserve,
        "exact_integer_preserves_solution": exact >= preserve,
        "float_integer_prediction_agreement": agreement >= 0.999,
    },
    "accuracy": {
        "fp32": fp32, "clip_only": clip, "weights_q88": weights,
        "production_like_q88": prod, "exact_integer": exact,
        "production_like_exact_agreement": agreement,
    },
    "parameter_summary": parameter_summary,
    "forward_ablation": ablation,
    "internal_precision_sweep": precision,
    "exact_output_top_tokens": top_outputs,
    "run": {
        "seed": SEED, "full_run": FULL_RUN, "train_steps": TRAIN_STEPS,
        "validation_examples": VAL_EXAMPLES, "exact_examples": EXACT_EXAMPLES,
        "device": str(DEVICE),
    },
}
(OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("=== decision.json ===")
print(json.dumps(decision, indent=2))
print("=== files ===")
for path in sorted(OUT.rglob("*")):
    if path.is_file():
        print(path.relative_to(OUT))
