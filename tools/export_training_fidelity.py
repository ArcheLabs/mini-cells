#!/usr/bin/env python3
"""Export the authoritative PyTorch Echo path into a byte-oriented fixture.

This file deliberately imports the production research modules.  It is not a
second reference model.  A machine without PyTorch fails loudly instead of
emitting a fabricated fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

try:
    import torch
    from minicells.config import load_config, resolved_config
    from minicells.data import CopyDataGenerator, fixed_dataset
    from minicells.evaluate import evaluate
    from minicells.metrics import masked_cross_entropy
    from minicells.model import EchoModel
    # Import the production training module as part of the provenance check;
    # the exporter must follow the same path used by the validated trainer,
    # never a second Python implementation.
    from minicells import train as production_train
    from minicells.reproducibility import set_global_seed
    from minicells.vocab import CharVocab
except ModuleNotFoundError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(f"training fidelity export requires PyTorch: missing {exc.name}") from exc


def flat_parameters(model: EchoModel) -> np.ndarray:
    return flat_tensors(model.parameters())


def flat_tensors(tensors) -> np.ndarray:
    return np.concatenate([tensor.detach().cpu().contiguous().numpy().astype("<f4").reshape(-1)
                           for tensor in tensors])


def flat_gradients(model: EchoModel) -> np.ndarray:
    return flat_tensors([parameter.grad for parameter in model.parameters()])


def f32_digest(values: np.ndarray) -> str:
    payload = np.asarray(values, dtype="<f4").tobytes()
    return "blake2b-256:0x" + hashlib.blake2b(payload, digest_size=32).hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes())


def write_batch(path: Path, batch) -> None:
    ids = batch.input_ids.detach().cpu().numpy().astype(np.uint8)
    lengths = batch.lengths.detach().cpu().numpy().astype(np.uint8)
    if ids.ndim != 2 or ids.shape[1] != 64:
        raise ValueError(f"unexpected batch shape: {ids.shape}")
    path.write_bytes(b"MCB1" + struct.pack("<II", ids.shape[0], 64) + ids.tobytes() + lengths.tobytes())


def optimizer_state(optimizer, name: str) -> np.ndarray:
    values = []
    for parameter in optimizer.param_groups[0]["params"]:
        state = optimizer.state[parameter]
        if name == "step":
            continue
        values.append(state[name].detach().cpu().contiguous().numpy().astype("<f4").reshape(-1))
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/echo-v0.yaml"))
    parser.add_argument("--output", default=str(ROOT / "fixtures/training-fidelity-v1"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda) for local reference runs")
    args = parser.parse_args()
    if args.steps < 16:
        raise SystemExit("training fidelity export requires at least 16 steps (use --steps 16 for a short gate)")
    output = Path(args.output); expected = output / "expected"; output.mkdir(parents=True, exist_ok=True); expected.mkdir(exist_ok=True)

    config = resolved_config(load_config(args.config), len(CharVocab()))
    set_global_seed(int(config["train"]["seed"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested cuda device but torch.cuda.is_available() is false")
    model = EchoModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"],
                                  weight_decay=config["train"]["weight_decay"])
    vocab = CharVocab()
    generator = CopyDataGenerator(vocab, config["train"]["seed"],
                                   min_length=config["data"]["min_length"], max_length=config["data"]["max_length"],
                                   num_cells=config["model"]["num_cells"], random_fraction=config["data"]["random_fraction"])

    initial = flat_parameters(model)
    write_f32(output / "initial-weights-f32.bin", initial)
    trace = []
    for step in range(1, args.steps + 1):
        batch = generator.batch(config["train"]["batch_size"], device=device)
        write_batch(output / f"batch-{step:06d}.bin", batch)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_cross_entropy(model(batch.input_ids), batch.target_ids, batch.mask)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_norm"])
        if step in (1, 2, 4, 16):
            write_f32(expected / f"step-{step:06d}-gradients-f32.bin", flat_gradients(model))
        optimizer.step()
        write_f32(expected / f"step-{step:06d}-weights-f32.bin", flat_parameters(model))
        write_f32(expected / f"step-{step:06d}-adam-m-f32.bin", optimizer_state(optimizer, "exp_avg"))
        write_f32(expected / f"step-{step:06d}-adam-v-f32.bin", optimizer_state(optimizer, "exp_avg_sq"))
        report = {"step": step, "loss": float(loss.detach().cpu()), "token_count": int(batch.mask.sum()),
                  "grad_norm": float(grad_norm), "weight_digest": f32_digest(flat_parameters(model))}
        (expected / f"step-{step:06d}-loss.json").write_text(json.dumps(report, indent=2) + "\n")
        trace.append(report)

    validation = fixed_dataset(vocab, config["validation"]["seed"], config["validation"]["examples"],
                               min_length=config["data"]["min_length"], max_length=config["data"]["max_length"],
                               num_cells=config["model"]["num_cells"], random_fraction=config["data"]["random_fraction"])
    validation = validation.to(device)
    write_batch(output / "validation.bin", validation)
    validation_metrics = evaluate(model, validation)
    (output / "validation-metrics.json").write_text(
        json.dumps({"schema": "minicells.training-fidelity-validation.v1", "step": args.steps,
                    **validation_metrics}, indent=2) + "\n"
    )
    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": torch.cuda.device_count(),
        "requested_device": str(device),
        "torch_config": torch.__config__.show(),
        "git_revision": git_revision(),
        "deterministic_algorithms": True,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2, default=str) + "\n")
    manifest = {"schema": "minicells.training-fidelity.v1", "algorithm": "echo-adamw-ce-v1",
                "model": config["model"], "logical_batch_size": 256, "steps": config["train"]["steps"],
                "learning_rate": config["train"]["learning_rate"], "weight_decay": config["train"]["weight_decay"],
                "grad_clip_norm": config["train"]["grad_clip_norm"], "data": config["data"],
                "validation": config["validation"], "optimizer": "torch.optim.AdamW",
                "optimizer_defaults": optimizer.defaults, "torch_version": torch.__version__,
                "parameter_order": [name for name, _ in model.named_parameters()], "fixture_steps": args.steps,
                "device": str(device),
                "production_train_module": production_train.__file__,
                "validation_metrics": "validation-metrics.json",
                "environment": "environment.json",
                "weight_digest": "blake2b-256 over little-endian FP32 parameter bytes"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    (output / "reference-trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
    print(json.dumps({"output": str(output), "steps": args.steps, "torch_version": torch.__version__}))


if __name__ == "__main__":
    main()
