"""Manual token-budget training and paired development/holdout evaluation."""
from __future__ import annotations

import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F

from experiments.common.train import make_batch, write_json

from .accounting import ObservedGemmFlops, active_weight_summary, execution_summary, unique_weight_counts
from .checkpoint import file_sha256, make_identity, restore_checkpoint, save_checkpoint
from .models import Experiment005LanguageModel, make_model
from .profiling import benchmark_decode, check_decode_parity
from .topology import perturb_important_edges


def _validation_starts(tokens: np.ndarray, sequence_length: int, count: int, seed: int) -> np.ndarray:
    if count <= 0 or len(tokens) <= sequence_length + 1:
        raise ValueError("validation split or sample count is invalid")
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(tokens) - sequence_length - 1, size=count, dtype=np.int64)


def _append_edge_record(path: Path, record: dict[str, Any]) -> None:
    records = []
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid edge-refresh log at line {line_number}") from exc
    for previous in records:
        if previous.get("step") == record.get("step"):
            if previous != record:
                raise RuntimeError("edge-refresh step identity conflicts with an existing record")
            return
    records.append(record)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records),
                         encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model: Experiment005LanguageModel, tokens: np.ndarray, starts: np.ndarray,
             *, sequence_length: int, batch_size: int, device: torch.device,
             profile_decode: bool = False) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    model.set_collect_trace(True)
    sample_losses: list[float] = []
    touched: list[Tensor] = []
    observed = ObservedGemmFlops(model)
    execution_weighted = {"visits": 0.0, "unique": 0.0, "backup": 0.0, "tokens": 0}
    cell_count = int(model.config["cell_count"])
    cell_usage = [torch.zeros(cell_count, dtype=torch.int64) for _ in model.blocks]
    edge_usage = [torch.zeros((cell_count + 1) * cell_count, dtype=torch.int64)
                  for _ in model.blocks]
    repeated_path_tokens = [0 for _ in model.blocks]
    unique_cells_sum = [0 for _ in model.blocks]
    path_tokens = 0
    try:
        with observed:
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset:offset + batch_size]
                x, y = make_batch(tokens, batch_starts, sequence_length, device)
                logits, _, _ = model(x)
                per_token = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                            y.reshape(-1), reduction="none")
                per_sequence = per_token.reshape(y.shape).mean(dim=1)
                sample_losses.extend(per_sequence.cpu().tolist())
                counts, _ = unique_weight_counts(model)
                touched.append(counts.float())
                summary = execution_summary(model)
                n = int(y.numel())
                execution_weighted["visits"] += summary["executed_cell_visits_per_token"] * n
                execution_weighted["unique"] += summary["unique_cells_per_token_layer"] * n
                execution_weighted["backup"] += summary["train_backup_cell_visits_per_token"] * n
                execution_weighted["tokens"] += n
                for layer, trace in enumerate(model.last_execution):
                    if trace is None:
                        raise RuntimeError("evaluation trace is missing an FFN path")
                    paths = trace["paths"].detach().cpu().reshape(-1, trace["paths"].shape[-1])
                    cell_usage[layer] += torch.bincount(paths.reshape(-1), minlength=cell_count)
                    for row in paths.tolist():
                        unique_cells_sum[layer] += len(set(int(item) for item in row))
                        repeated_path_tokens[layer] += int(len(set(row)) < len(row))
                    path_tokens += int(paths.shape[0]) if layer == 0 else 0
                    if model.arm in {"S0", "R0", "G0"}:
                        sources = torch.full_like(paths, cell_count)
                        sources[:, 1:] = paths[:, :-1]
                        pair_ids = sources * cell_count + paths
                        edge_usage[layer] += torch.bincount(
                            pair_ids.reshape(-1), minlength=(cell_count + 1) * cell_count
                        )
        all_touched = torch.cat(touched)
        quantiles = torch.quantile(all_touched, torch.tensor([0.5, 0.95, 0.99]))
        total_nll = float(np.mean(sample_losses))
        parity = None
        decode = None
        if profile_decode:
            sample_start = int(starts[0])
            sample, _ = make_batch(tokens, np.asarray([sample_start]), sequence_length, device)
            parity = check_decode_parity(model, sample)
            decode = benchmark_decode(model, sample, prefix_length=max(2, sequence_length // 2),
                                     decode_tokens=min(16, sequence_length // 2), profile_kernels=True)
        denominator = max(1, execution_weighted["tokens"])
        routing_layers = []
        for layer in range(len(model.blocks)):
            layer_record: dict[str, Any] = {
                "cell_visit_counts": cell_usage[layer].tolist(),
                "unique_cells_per_token_mean": unique_cells_sum[layer] / max(1, path_tokens),
                "repeated_path_token_fraction": repeated_path_tokens[layer] / max(1, path_tokens),
            }
            if model.arm in {"S0", "R0", "G0"}:
                layer_record["transition_visit_counts"] = edge_usage[layer].reshape(
                    cell_count + 1, cell_count
                ).tolist()
            if model.arm in {"R0", "G0"}:
                adjacency = model.blocks[layer].ffn.adjacency.detach().cpu()
                layer_record["adjacency"] = adjacency.to(torch.uint8).tolist()
                layer_record["out_degree_per_source"] = adjacency.sum(-1).tolist()
            routing_layers.append(layer_record)
        return {
            "validation_loss": total_nll,
            "perplexity": math.exp(min(total_nll, 80.0)),
            "evaluation_windows": len(sample_losses),
            "validation_tokens": len(sample_losses) * sequence_length,
            "sample_nll": sample_losses,
            "paired_sampling_unit": "fixed contiguous token windows; same starts for every arm and seed",
            "unique_weight_params_per_token": {
                "mean": float(all_touched.mean()), "p50": float(quantiles[0]),
                "p95": float(quantiles[1]), "p99": float(quantiles[2])},
            "model_total_params": sum(p.numel() for p in model.parameters()),
            "unique_weight_fraction_of_model": float(all_touched.mean()) /
                sum(p.numel() for p in model.parameters()),
            "execution": {
                "executed_cell_visits_per_token_per_block": execution_weighted["visits"] / denominator,
                "unique_cells_per_token_per_block": execution_weighted["unique"] / denominator,
                "train_backup_cell_visits_per_token": execution_weighted["backup"] / denominator,
            },
            "routing_summary": {"path_tokens_per_layer": path_tokens,
                                "per_layer": routing_layers},
            "observed_gemm": observed.report(len(sample_losses) * sequence_length),
            "decode_parity": parity,
            "decode_profile": decode,
        }
    finally:
        model.set_collect_trace(False)
        model.train(was_training)


def evaluate_graph_perturbations(model: Experiment005LanguageModel, tokens: np.ndarray,
                                 starts: np.ndarray, *, sequence_length: int,
                                 batch_size: int, device: torch.device,
                                 seeds: list[int], edge_use: list[list[list[int]]],
                                 replacements: int) -> dict[str, Any]:
    """Evaluate fixed-degree graph interventions on the same trained G0 checkpoint."""
    if model.arm != "G0" or not seeds:
        raise ValueError("G0 perturbation diagnostics require a G0 checkpoint and frozen seeds")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    original = [block.ffn.adjacency.detach().cpu().clone() for block in model.blocks]
    rows = []
    try:
        for seed in seeds:
            changes = []
            for layer, block in enumerate(model.blocks):
                graph, removed = perturb_important_edges(
                    original[layer], torch.tensor(edge_use[layer], dtype=torch.int64),
                    seed=int(seed) + layer, replacements=replacements
                )
                edge_changes = 2 * len(removed)
                block.ffn.adjacency.copy_(graph.to(block.ffn.adjacency.device))
                changes.append({"layer": layer, "replacement_count": edge_changes // 2,
                                "important_edges_removed": removed})
            metrics = evaluate(model, tokens, starts, sequence_length=sequence_length,
                               batch_size=batch_size, device=device, profile_decode=False)
            rows.append({"perturbation_seed": int(seed),
                         "edge_swaps_per_layer": changes,
                         "validation_loss": metrics["validation_loss"],
                         "perplexity": metrics["perplexity"],
                         "sample_nll": metrics["sample_nll"],
                         "routing_summary": metrics["routing_summary"],
                         "same_checkpoint": True,
                         "parameter_count_unchanged":
                             metrics["model_total_params"] == parameter_count})
    finally:
        for block, graph in zip(model.blocks, original):
            block.ffn.adjacency.copy_(graph.to(block.ffn.adjacency.device))
    return {"status": "POSTHOC_GRAPH_PERTURBATIONS_EVALUATED",
            "scientific_decision": False,
            "scope": "same G0 checkpoint; fixed-degree reachable graph interventions; no retraining",
            "results": rows}


def train_arm(config: dict[str, Any], arm: str, seed: int, source_commit: str,
              assets: dict[str, Any], output: str | Path, *, device: str = "cuda",
              resume: bool = True,
              edge_refresh: Callable[[Experiment005LanguageModel, int, np.random.Generator], dict[str, Any]] | None = None
              ) -> dict[str, Any]:
    """Run one fully registered arm; no wall-clock truncation is applied."""
    if arm not in config["arms"]:
        raise ValueError(f"arm {arm!r} is not in the frozen experiment matrix")
    if arm in {"G0", "R0", "S0"} and edge_refresh is None:
        raise ValueError("sequential arms require the registered matched-compute edge-probe callback")
    data_cfg = config["data"]
    for split in ("train", "regression", "holdout"):
        if split not in assets or not isinstance(assets[split], np.ndarray):
            raise ValueError(f"prepared assets missing {split}.npy")
    protocol_file = Path(config["protocol_path"])
    if config.get("protocol_status") != "registered":
        raise RuntimeError("protocol is not registered; full training is prohibited")
    identity = make_identity(source_commit=source_commit, seed=seed, arm=arm,
                             config_sha256=file_sha256(config["config_path"]),
                             protocol_sha256=file_sha256(protocol_file),
                             assets_manifest_sha256=assets["manifest_sha256"])
    model_cfg = {**config["model"], "vocab_size": int(data_cfg["vocab_size"]),
                 "max_sequence_length": int(data_cfg["sequence_length"])}
    training_cfg = config["training"]
    destination = Path(output)
    if (destination / "metrics.json").exists():
        raise FileExistsError("completed arm metrics are immutable; use a fresh output directory")
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = make_model(model_cfg, arm, seed).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg["learning_rate"]),
                                  betas=(0.9, 0.95), weight_decay=float(training_cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device_obj.type == "cuda")
    latest = checkpoints / "latest.pt"
    first_step = 0
    history: list[dict[str, Any]] = []
    train_seconds = 0.0
    backup_cell_visits = 0
    resume_extra: dict[str, Any] = {}
    if resume and latest.is_file():
        first_step, resume_extra = restore_checkpoint(
            latest, model=model, optimizer=optimizer, scaler=scaler,
            expected_identity=identity, generator=rng, map_location="cpu"
        )
        if first_step < 0 or first_step > int(training_cfg["steps"]):
            raise ValueError("checkpoint step is outside the registered token budget")
        history = list(resume_extra.get("history", []))
        train_seconds = float(resume_extra.get("train_seconds", 0.0))
        backup_cell_visits = int(resume_extra.get("backup_cell_visits", 0))
    pending_edge_record = resume_extra.get("edge_refresh_pending")
    if pending_edge_record is not None:
        _append_edge_record(destination / "edge-refresh.jsonl", pending_edge_record)

    train = assets["train"]
    sequence_length = int(data_cfg["sequence_length"])
    batch_size = int(data_cfg["batch_size"])
    tokens_per_step = sequence_length * batch_size
    steps = int(data_cfg["train_tokens_per_arm"]) // tokens_per_step
    if steps != int(training_cfg["steps"]):
        raise ValueError("step count does not exactly match the registered training token budget")
    regression_starts = _validation_starts(assets["regression"], sequence_length,
                                           int(config["evaluation"]["windows"]), seed + 1)
    holdout_starts = _validation_starts(assets["holdout"], sequence_length,
                                        int(config["evaluation"]["windows"]), seed + 1)
    started = time.perf_counter()
    model.train()
    train_flops = ObservedGemmFlops(model)
    train_flops.by_module.update(resume_extra.get("train_forward_flops_by_module", {}))
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)
    with train_flops:
        train_flops.enabled = False
        for step in range(first_step + 1, steps + 1):
            pending_edge_record = None
            if arm in {"S0", "R0", "G0"} and step % int(training_cfg["edge_refresh_every_steps"]) == 0:
                pending_edge_record = edge_refresh(model, step, rng)
            starts = rng.integers(0, len(train) - sequence_length - 1,
                                  size=batch_size, dtype=np.int64)
            x, y = make_batch(train, starts, sequence_length, device_obj)
            progress = max(0.0, (step - int(training_cfg["warmup_steps"])) /
                           max(1, steps - int(training_cfg["warmup_steps"])))
            warmup = min(step / max(1, int(training_cfg["warmup_steps"])), 1.0)
            factor = warmup * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = float(training_cfg["learning_rate"]) * factor
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            tick = time.perf_counter()
            succeeded = False
            for _attempt in range(8):
                optimizer.zero_grad(set_to_none=True)
                train_flops.enabled = True
                try:
                    with torch.autocast(device_type=device_obj.type, dtype=torch.float16,
                                        enabled=device_obj.type == "cuda"):
                        logits, auxiliary, _ = model(x)
                        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                              y.reshape(-1))
                        loss = nll + float(training_cfg.get("balance_coefficient", 0.01)) * auxiliary
                finally:
                    train_flops.enabled = False
                backup_cell_visits += int(model.last_backup_cell_visits)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    succeeded = True
                    break
            if not succeeded:
                raise RuntimeError(f"FP16 overflow persisted after 8 retries of step {step}")
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            train_seconds += time.perf_counter() - tick

            every = int(training_cfg["evaluate_every_steps"])
            should_evaluate = step % every == 0 or step == steps
            should_checkpoint = should_evaluate or pending_edge_record is not None
            if should_evaluate:
                evaluation = evaluate(model, assets["regression"], regression_starts,
                                      sequence_length=sequence_length,
                                      batch_size=int(config["evaluation"]["batch_size"]),
                                      device=device_obj)
                history.append({"step": step, "train_tokens": step * tokens_per_step,
                                "regression": evaluation})
                write_json(destination / "learning_curve.json", history)
            if should_checkpoint:
                save_checkpoint(latest, model=model, optimizer=optimizer, scaler=scaler,
                                step=step, identity=identity, generator=rng,
                                extra={"history": history, "train_seconds": train_seconds,
                                       "backup_cell_visits": backup_cell_visits,
                                       "train_forward_flops_by_module": dict(train_flops.by_module),
                                       "edge_refresh_pending": pending_edge_record})
                if pending_edge_record is not None:
                    _append_edge_record(destination / "edge-refresh.jsonl", pending_edge_record)
            if should_evaluate:
                print(json.dumps({"arm": arm, "seed": seed, "step": step,
                                  "tokens": step * tokens_per_step,
                                  "regression_ppl": evaluation["perplexity"]}), flush=True)

    # The independent holdout is touched only after the complete arm checkpoint.
    holdout = evaluate(model, assets["holdout"], holdout_starts,
                       sequence_length=sequence_length,
                       batch_size=int(config["evaluation"]["batch_size"]),
                       device=device_obj, profile_decode=False)
    posthoc = None
    graph_learning = None
    if arm == "G0":
        perturbation_seeds = [int(value) for value in
                              config["evaluation"].get("edge_perturbation_seeds", [])]
        probe_evaluation = evaluate(
            model, assets["train"], np.asarray(assets["probe_starts"], dtype=np.int64),
            sequence_length=sequence_length,
            batch_size=int(config["evaluation"]["batch_size"]),
            device=device_obj, profile_decode=False
        )
        edge_use = [layer["transition_visit_counts"]
                    for layer in probe_evaluation["routing_summary"]["per_layer"]]
        posthoc = evaluate_graph_perturbations(
            model, assets["holdout"], holdout_starts,
            sequence_length=sequence_length,
            batch_size=int(config["evaluation"]["batch_size"]),
            device=device_obj, seeds=perturbation_seeds, edge_use=edge_use,
            replacements=int(config["evaluation"]["edge_perturbation_replacements"])
        )
        edge_log = destination / "edge-refresh.jsonl"
        events = ([json.loads(line) for line in edge_log.read_text(encoding="utf-8").splitlines()
                   if line.strip()] if edge_log.exists() else [])
        changes = [change for event in events for change in event.get("edge_changes", [])]
        graph_learning = {
            "refresh_events": len(events),
            "edge_replacements": len(changes),
            "edge_changes": changes,
            "training_probe_edge_use": edge_use,
            "final_routing_summary": holdout["routing_summary"],
            "interpretation": "training-only topology updates and final fixed-degree evaluation graph",
        }
    result = {"status": "completed", "scientific_decision": False,
              "arm": arm, "seed": seed, "source_commit": source_commit,
              "identity": identity, "steps": steps,
              "train_tokens": steps * tokens_per_step,
              "train_seconds": train_seconds, "elapsed_seconds": time.perf_counter() - started,
              "tokens_per_second_train_only": steps * tokens_per_step / train_seconds if train_seconds else None,
              "training_forward_compute": train_flops.report(steps * tokens_per_step),
              "training_compute_note": "Observed forward GEMMs include selected and top-2 backup Cell calls, including overflow retries; backward/optimizer FLOPs are not inferred.",
              "training_compute_note": "Observed forward GEMMs include selected and top-2 backup Cell calls, including overflow retries; backward/optimizer FLOPs are not inferred.",
              "train_backup_cell_visits_per_token": backup_cell_visits /
                  max(1, steps * tokens_per_step),
              "peak_training_allocated_bytes": torch.cuda.max_memory_allocated(device_obj)
                  if device_obj.type == "cuda" else None,
              "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                          "torch": torch.__version__, "device": str(device_obj),
                          "device_name": torch.cuda.get_device_name(device_obj)
                              if device_obj.type == "cuda" else platform.processor(),
                          "cuda_runtime": torch.version.cuda},
              "final_holdout": holdout,
              "graph_learning": graph_learning,
              "posthoc_diagnostics": posthoc,
              "holdout_access": "final evaluation and preregistered posthoc diagnostics after the complete arm; never used for training or edge refresh",
              "checkpoint_sha256": __import__("hashlib").sha256(latest.read_bytes()).hexdigest()}
    write_json(destination / "metrics.json", result)
    model.set_collect_trace(False)
    del model, optimizer, scaler
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    return result


def profile_completed_arm(config: dict[str, Any], arm: str, seed: int, source_commit: str,
                          assets: dict[str, Any], output: str | Path, *,
                          device: str = "cuda:0") -> dict[str, Any]:
    """Measure batch-1 KV-cache decode separately after all concurrent training stops."""
    output = Path(output)
    metrics_path = output / "metrics.json"
    checkpoint_path = output / "checkpoints/latest.pt"
    profile_path = output / "decode-profile.json"
    if profile_path.exists():
        raise FileExistsError("decode profile is immutable; preserve it and use a fresh run output")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != metrics.get("checkpoint_sha256"):
        raise ValueError("final training checkpoint is missing or differs from the metrics hash")
    identity = metrics.get("identity", {})
    if (metrics.get("status") != "completed" or metrics.get("arm") != arm or
            metrics.get("seed") != seed or metrics.get("source_commit") != source_commit or
            identity.get("source_commit") != source_commit or identity.get("arm") != arm or
            identity.get("seed") != seed):
        raise ValueError("decode profiling identity does not match a completed arm")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("identity") != identity or int(payload.get("step", -1)) != int(metrics["steps"]):
        raise ValueError("final training checkpoint does not match the completed arm metrics")
    model_cfg = {**config["model"], "vocab_size": int(config["data"]["vocab_size"]),
                 "max_sequence_length": int(config["data"]["sequence_length"])}
    model = make_model(model_cfg, arm, seed).to(torch.device(device))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    starts = _validation_starts(assets["holdout"], int(config["data"]["sequence_length"]),
                               int(config["evaluation"]["windows"]), seed + 1)
    sample, _ = make_batch(assets["holdout"], starts[:1], int(config["data"]["sequence_length"]),
                           torch.device(device))
    parity = check_decode_parity(model, sample)
    if not parity["passed"]:
        raise RuntimeError("full-forward and incremental KV-cache logits do not match")
    decode = benchmark_decode(model, sample,
                              prefix_length=max(2, int(config["data"]["sequence_length"]) // 2),
                              decode_tokens=min(16, int(config["data"]["sequence_length"]) // 2),
                              profile_kernels=True)
    result = {"status": "DECODE_PROFILE_COMPLETED", "scientific_decision": False,
              "identity": identity, "arm": arm, "seed": seed,
              "decode_parity": parity, "decode_profile": decode,
              "checkpoint_sha256": metrics["checkpoint_sha256"],
              "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                          "torch": torch.__version__, "device": str(device),
                          "device_name": torch.cuda.get_device_name(torch.device(device))
                              if torch.device(device).type == "cuda" else platform.processor(),
                          "cuda_runtime": torch.version.cuda},
              "measurement_isolation": "single worker; caller must stop all other GPU training first"}
    write_json(profile_path, result)
    del model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result
