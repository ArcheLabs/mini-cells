from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import resolved_config
from .data import CopyDataGenerator, fixed_dataset
from .evaluate import evaluate
from .metrics import masked_cross_entropy
from .model import EchoModel
from .ops import architecture_stats
from .reproducibility import environment_info, set_global_seed, write_json
from .sample import sample_panel
from .vocab import CharVocab

CHECKPOINT_FORMAT = "minicells.echo.checkpoint.v1"


def save_checkpoint(path, model, optimizer, config, step, metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": CHECKPOINT_FORMAT, "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "config": config, "train_seed": config["train"]["seed"],
                "validation_seed": config["validation"]["seed"], "metrics": metrics}, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    model = EchoModel(**payload["config"]["model"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def _write_samples(path: Path, step: int, panel):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Step {step}\n\n")
        for item in panel:
            handle.write(f"- `{item['input']}` → `{item['prediction']}` ({item['similarity']:.2%})\n")


def train(config: dict[str, Any], device: str | None = None) -> dict[str, Any]:
    vocab = CharVocab(); config = resolved_config(config, len(vocab))
    seed = int(config["train"]["seed"]); set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(config["output"]["root"]); checkpoints = root / "checkpoints"
    root.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(exist_ok=True)
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(root / "environment.json", environment_info(device))
    model = EchoModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"],
                                  weight_decay=config["train"]["weight_decay"])
    data_args = dict(min_length=config["data"]["min_length"], max_length=config["data"]["max_length"],
                     num_cells=config["model"]["num_cells"], random_fraction=config["data"]["random_fraction"])
    training = CopyDataGenerator(vocab, seed, **data_args)
    validation = fixed_dataset(vocab, config["validation"]["seed"],
                               config["validation"]["examples"], **data_args)
    validation = validation.to(device)
    stats = architecture_stats(model); write_json(root / "architecture.json", {**config["model"], **stats})
    print(f"device={device} parameters={stats['parameter_count']} macs/sample={stats['estimated_macs']}")
    metrics_path = root / "metrics.csv"; samples_path = root / "samples.md"
    metrics_path.write_text("step,loss,token_accuracy,exact_sequence_accuracy\n", encoding="utf-8")
    samples_path.write_text("# Real model samples\n", encoding="utf-8")
    initial = evaluate(model, validation); save_checkpoint(checkpoints / "step-000000.pt", model, optimizer, config, 0, initial)
    save_checkpoint(checkpoints / "best.pt", model, optimizer, config, 0, initial)
    initial_panel = sample_panel(model, vocab, device=device)
    _write_samples(samples_path, 0, initial_panel)
    samples_jsonl = root / "samples.jsonl"
    samples_jsonl.write_text(json.dumps({"step": 0, "samples": initial_panel}) + "\n", encoding="utf-8")
    best_loss = initial["loss"]; best_metrics = initial; history = []
    for step in range(1, config["train"]["steps"] + 1):
        model.train(); batch = training.batch(config["train"]["batch_size"], device)
        optimizer.zero_grad(set_to_none=True); loss = masked_cross_entropy(model(batch.input_ids), batch.target_ids, batch.mask)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_norm"]); optimizer.step()
        evaluate_now = step % config["train"]["eval_every"] == 0 or step == config["train"]["steps"]
        checkpoint_now = step % config["train"]["checkpoint_every"] == 0 or step == config["train"]["steps"]
        if evaluate_now or checkpoint_now:
            current = evaluate(model, validation)
            if evaluate_now:
                history.append({"step": step, **current})
                with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow([step, current["loss"], current["token_accuracy"], current["exact_sequence_accuracy"]])
                print(f"step={step} loss={current['loss']:.4f} token={current['token_accuracy']:.4%} exact={current['exact_sequence_accuracy']:.4%}")
                panel = sample_panel(model, vocab, device=device)
                _write_samples(samples_path, step, panel)
                with samples_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "samples": panel}) + "\n")
            if current["loss"] < best_loss:
                best_loss, best_metrics = current["loss"], current
                save_checkpoint(checkpoints / "best.pt", model, optimizer, config, step, current)
            if checkpoint_now:
                save_checkpoint(checkpoints / f"step-{step:06d}.pt", model, optimizer, config, step, current)
    save_checkpoint(checkpoints / "final.pt", model, optimizer, config, config["train"]["steps"], current)
    report = {"experiment": "MINI Cells Experiment 001 — Echo", "format": "minicells.echo.report.v1",
              "status": "INCOMPLETE", "model": {**config["model"], **stats},
              "metrics": {"best_token_accuracy": best_metrics["token_accuracy"],
                          "best_exact_sequence_accuracy": best_metrics["exact_sequence_accuracy"],
                          "best_validation_loss": best_metrics["loss"], "final": current},
              "reproducibility": {"train_seed": seed, "validation_seed": config["validation"]["seed"]}}
    write_json(root / "report.json", report)
    _plot_curves(root, history)
    return report


def _plot_curves(root: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt
    curves = root / "curves"; curves.mkdir(exist_ok=True)
    for key, filename, label in (("loss", "loss.png", "Validation loss"),
                                 ("token_accuracy", "token_accuracy.png", "Token accuracy"),
                                 ("exact_sequence_accuracy", "sequence_accuracy.png", "Exact sequence accuracy")):
        figure, axis = plt.subplots(); axis.plot([row["step"] for row in history], [row[key] for row in history])
        axis.set(xlabel="Step", ylabel=label, title=label); axis.grid(True, alpha=.3)
        figure.tight_layout(); figure.savefig(curves / filename); plt.close(figure)
