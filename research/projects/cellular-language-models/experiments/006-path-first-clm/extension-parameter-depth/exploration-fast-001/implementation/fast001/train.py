"""Thin FAST loop: every optimizer step is the existing pd001-light-v1 engine."""

import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from exp006.reuse import canonical_sha256, file_sha256
from .io import write_json
from pd001.engine import step
from pd001.checkpoint import save, restore
from .config import authorize, engine_config, source_sha
from .models import make_model
from .data import load_assets


def identity(c, arm, assets):
    return {
        "experiment_id": c["experiment_id"],
        "arm": arm,
        "seed": c["seed"],
        "source_sha": source_sha(),
        "config_sha256": canonical_sha256(c),
        "protocol_sha256": file_sha256(Path(__file__).resolve().parents[2] / "SPEC_FAST001.md"),
        "asset_sha256": assets["asset_sha256"],
        "tokenizer_sha256": assets["tokenizer_sha256"],
        "evaluation_sha256": assets["manifest"]["evaluation_sha256"],
        "role": "development",
        "engine": "pd001-light-v1",
        "target_tokens": c["training"]["tokens"],
    }


def train(
    c, arm, assets_dir, output, device="cpu", *, resume=False, max_session_seconds=None, pause_after_step=None
):
    authorize(c, device, "train")
    assets = load_assets(c, assets_dir)
    run_id = identity(c, arm, assets)
    output = Path(output)
    checkpoint = output / "checkpoint.pt"
    if (output / "metrics.json").exists():
        result = json.loads((output / "metrics.json").read_text())
        if (
            not resume
            or result["identity"] != run_id
            or file_sha256(checkpoint) != result["checkpoint_sha256"]
        ):
            raise ValueError("completed arm is immutable; matching --resume required")
        return result
    if output.exists() and not resume:
        raise FileExistsError("preserve prior arm progress; use --resume")
    if resume and not checkpoint.exists():
        raise FileNotFoundError("resume checkpoint missing")
    model = make_model(c, arm).to(device)
    t = c["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    rng = np.random.default_rng(c["seed"])
    path = [torch.Generator(device=device).manual_seed(c["seed"] + 1)]
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    first = 0
    history = []
    curve = []
    prior = 0.0
    first_quality = None
    if resume:
        valid = None
        for checkpoint_name, receipt_name in (
            ("checkpoint.pt", "checkpoint.json"),
            ("checkpoint.previous.pt", "checkpoint.previous.json"),
        ):
            candidate = output / checkpoint_name
            receipt_path = output / receipt_name
            if not candidate.exists() or not receipt_path.exists():
                continue
            receipt = json.loads(receipt_path.read_text())
            if receipt["identity"] == run_id and file_sha256(candidate) == receipt["sha256"]:
                valid = candidate
                break
        if valid is None:
            raise ValueError("checkpoint integrity/identity mismatch; no committed recovery pair")
        first, extra = restore(valid, model, optimizer, run_id, rng, path, device=device)
        history, curve, prior, first_quality = (
            extra["history"],
            extra["curve"],
            extra["elapsed_seconds"],
            extra["initial_quality"],
        )
    output.mkdir(parents=True, exist_ok=True)
    from .evaluate import quality

    if first_quality is None:
        first_quality = quality(model, c, assets, final=False)
    started = time.perf_counter()
    engine = engine_config(c)
    for index in range(first + 1, t["steps"] + 1):
        chosen = rng.choice(assets["windows"]["train"]["starts"], size=t["effective_batch"], replace=True)
        diagnostic = index % c["performance"]["diagnostics_interval"] == 0
        tick = time.perf_counter()
        row = step(
            model,
            optimizer,
            engine,
            index,
            chosen,
            assets["train"],
            path,
            device,
            capture=diagnostic,
            diagnostics=diagnostic,
            routing=diagnostic,
        )
        row.update(optimizer_seconds=time.perf_counter() - tick, data_starts=chosen.tolist())
        history.append(row)
        if index in c["evaluation"]["steps"]:
            tick = time.perf_counter()
            result = quality(model, c, assets, final=index == t["steps"])
            curve.append(
                {
                    "step": index,
                    "target_tokens": row["target_tokens"],
                    "quality": result,
                    "evaluation_seconds": time.perf_counter() - tick,
                }
            )
        if index == 1 or diagnostic:
            print(
                json.dumps(
                    {"arm": arm, "step": index, "ce": row["ce"], "target_tokens": row["target_tokens"]}
                ),
                flush=True,
            )
        paused = index < t["steps"] and (
            (pause_after_step is not None and first < pause_after_step <= index)
            or (max_session_seconds is not None and time.perf_counter() - started >= max_session_seconds)
        )
        if index % c["performance"]["checkpoint_interval"] == 0 or index == t["steps"] or paused:
            elapsed = prior + time.perf_counter() - started
            # Commit a recovery pair before replacing the current atomic checkpoint.
            receipt_path = output / "checkpoint.json"
            if checkpoint.exists() and receipt_path.exists():
                old = json.loads(receipt_path.read_text())
                if old["identity"] == run_id and file_sha256(checkpoint) == old["sha256"]:
                    temporary = output / "checkpoint.previous.tmp"
                    shutil.copy2(checkpoint, temporary)
                    temporary.replace(output / "checkpoint.previous.pt")
                    write_json(output / "checkpoint.previous.json", old)
            save(
                checkpoint,
                model,
                optimizer,
                index,
                run_id,
                rng,
                path,
                extra={
                    "history": history,
                    "curve": curve,
                    "elapsed_seconds": elapsed,
                    "initial_quality": first_quality,
                    "learning_rate_schedule": "stateless step-indexed warmup; global step saved",
                },
            )
            write_json(
                output / "checkpoint.json",
                {"identity": run_id, "step": index, "sha256": file_sha256(checkpoint)},
            )
            write_json(output / "history.json", history)
            write_json(output / "curve.json", curve)
        if paused:
            result = {
                "status": "PAUSED_RESUMABLE",
                "identity": run_id,
                "step": index,
                "scientific_decision": False,
            }
            write_json(output / "pause.json", result)
            return result
    final = (
        curve[-1]["quality"]
        if curve and curve[-1]["step"] == t["steps"]
        else quality(model, c, assets, final=True)
    )
    result = {
        "status": "DEVELOPMENT_COMPLETED",
        "identity": run_id,
        "step": t["steps"],
        "train_tokens": t["tokens"],
        "total_params": sum(p.numel() for p in model.parameters()),
        "initial_quality": first_quality,
        "quality": final,
        "train_seconds": sum(r["optimizer_seconds"] for r in history),
        "wall_seconds": prior + time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
        "checkpoint_sha256": file_sha256(checkpoint),
        "scientific_decision": False,
        "research_claim": "ENGINEERING_SMOKE_ONLY"
        if c.get("engineering_fixture") or t["steps"] == 50
        else "EXPLORATORY_ONLY",
        "synthetic_data": bool(c.get("engineering_fixture")),
    }
    write_json(output / "metrics.json", result)
    (output / "pause.json").unlink(missing_ok=True)
    return result
