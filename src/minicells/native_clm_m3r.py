"""Native CLM v0 M3R: read-preserving lineage-isolated growth.

M3R keeps the M3 protected-write/growth trigger unchanged and changes only the
read topology. Original M1 roots remain the only top-level routing candidates.
Spawned children compete only inside the selected root lineage, so mitosis cannot
steal another root's sparse route slot.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .native_clm_m2 import NativeCLMM2Config, evaluate_matrix, sha256_file
from .native_clm_m3 import (
    GrowthWindow,
    NativeCLMM3GrowthConfig,
    _autocast,
    _cycle,
    _forgetting,
    _freeze_to_cell_only,
    _loader,
    _lr_factor,
    _make_optimizer,
    _observe_growth_window,
    _phase_gain,
    original_read_state_sha256,
    root_cell_weight_sha256,
    run_arm as run_m3_arm,
)
from .native_clm_v0 import NativeCLM, NativeCLMConfig


class LineageNativeCLM(NativeCLM):
    """Native CLM with immutable root routing and lineage-local concrete routing."""

    def __init__(
        self,
        config: NativeCLMConfig,
        *,
        cell_count: int | None = None,
        lineage_root_count: int | None = None,
    ) -> None:
        super().__init__(config, cell_count=cell_count)
        root_count = config.initial_cells if lineage_root_count is None else int(lineage_root_count)
        if root_count < config.active_cells or root_count > self.cell_count:
            raise ValueError("invalid lineage_root_count")
        self.lineage_root_count = root_count

    def _direct_children(self) -> dict[int, int]:
        children: dict[int, int] = {}
        for child_id in range(self.lineage_root_count, self.cell_count):
            parent_id = int(self.cellular.cells[child_id].parent_id.item())
            if not (0 <= parent_id < child_id):
                raise RuntimeError(f"invalid M3R lineage parent {parent_id} for child {child_id}")
            if parent_id in children:
                raise RuntimeError(f"M3R chain routing allows one direct child per node: {parent_id}")
            children[parent_id] = child_id
        return children

    def is_lineage_leaf(self, cell_id: int) -> bool:
        return int(cell_id) not in self._direct_children()

    def _lineage_route_details(self, x: Tensor) -> dict[str, Tensor]:
        route_input = self.cellular.norm(x)
        query = F.normalize(self.cellular.query_proj(route_input), dim=-1)
        root_keys = torch.stack(
            [F.normalize(self.cellular.cells[idx].route_key, dim=0) for idx in range(self.lineage_root_count)],
            dim=0,
        )
        root_scores = query.matmul(root_keys.transpose(0, 1)) / self.config.route_temperature
        k = min(self.config.active_cells, self.lineage_root_count)
        root_top_scores, root_idx = torch.topk(root_scores, k=k, dim=-1)
        root_probs = F.softmax(root_top_scores, dim=-1)

        concrete_idx = root_idx.clone()
        if self.cell_count > self.lineage_root_count:
            all_keys = torch.stack(
                [F.normalize(cell.route_key, dim=0) for cell in self.cellular.cells], dim=0
            )
            for parent_id, child_id in sorted(self._direct_children().items()):
                parent_score = query.matmul(all_keys[parent_id]) / self.config.route_temperature
                child_score = query.matmul(all_keys[child_id]) / self.config.route_temperature
                switch = (concrete_idx == parent_id) & (child_score > parent_score).unsqueeze(-1)
                concrete_idx = torch.where(
                    switch,
                    torch.full_like(concrete_idx, child_id),
                    concrete_idx,
                )

        return {
            "route_input": route_input,
            "query": query,
            "top_idx": concrete_idx,
            "top_probs": root_probs,
            "top_scores": root_top_scores,
            "root_idx": root_idx,
            "root_probs": root_probs,
            "root_scores": root_scores,
        }

    def _lineage_cellular_forward(
        self, x: Tensor, *, return_info: bool
    ) -> tuple[Tensor, dict[str, Any] | None]:
        details = self._lineage_route_details(x)
        route_input = details["route_input"]
        top_idx = details["top_idx"]
        top_probs = details["top_probs"]
        top_scores = details["top_scores"]
        batch, seq_len, width = x.shape
        flat_x = x.reshape(batch * seq_len, width)
        flat_idx = top_idx.reshape(batch * seq_len, -1)
        flat_probs = top_probs.reshape(batch * seq_len, -1)
        flat_out = torch.zeros_like(flat_x)

        with torch.no_grad():
            usage = torch.bincount(flat_idx.reshape(-1), minlength=self.cell_count)
            for cell_id, count in enumerate(usage.tolist()):
                if count:
                    self.cellular.cells[cell_id].usage_count.add_(count)

        for cell_id, cell in enumerate(self.cellular.cells):
            positions = torch.nonzero(flat_idx == cell_id, as_tuple=False)
            if positions.numel() == 0:
                continue
            token_rows = positions[:, 0]
            slots = positions[:, 1]
            selected = flat_x.index_select(0, token_rows)
            cell_out = cell(selected)
            gates = flat_probs[token_rows, slots].unsqueeze(-1)
            flat_out.index_add_(0, token_rows, cell_out * gates)

        out = x + flat_out.view(batch, seq_len, width)
        if not return_info:
            return out, None

        entropy = -(top_probs * torch.log(top_probs.clamp_min(1e-9))).sum(dim=-1).mean()
        confidence = top_probs[..., 0].mean()
        if top_scores.size(-1) > 1:
            margin = (top_scores[..., 0] - top_scores[..., 1]).mean()
        else:
            margin = top_scores.new_tensor(float("inf"))
        info: dict[str, Any] = {
            "top_idx": top_idx.detach(),
            "top_probs": top_probs.detach(),
            "cell_input": x.detach(),
            "route_input": route_input.detach(),
            "root_idx": details["root_idx"].detach(),
            "root_probs": details["root_probs"].detach(),
            "route_entropy": float(entropy.detach().cpu()),
            "top1_confidence": float(confidence.detach().cpu()),
            "route_margin": float(margin.detach().cpu()),
            "cell_count": self.cell_count,
            "active_cells": top_idx.size(-1),
            "active_fraction_vs_dense": top_idx.size(-1) / self.cell_count,
            "routing_mode": "root_lineage_chain",
            "lineage_root_count": self.lineage_root_count,
        }
        return out, info

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        *,
        return_info: bool = False,
    ) -> dict[str, Any]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        batch, seq_len = tokens.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured max_seq_len")
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)

        cell_info = None
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == self.config.cellular_layer_index:
                x, cell_info = self._lineage_cellular_forward(x, return_info=return_info)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * seq_len, -1),
                targets.reshape(batch * seq_len),
            )
        return {"logits": logits, "loss": loss, "cell_info": cell_info}

    def checkpoint_payload(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = super().checkpoint_payload(extra=extra)
        payload["routing_state"] = {
            "mode": "root_lineage_chain",
            "lineage_root_count": self.lineage_root_count,
        }
        return payload

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["LineageNativeCLM", dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format") != "minicells.native-clm-v0.checkpoint.v1":
            raise ValueError("unsupported Native CLM checkpoint format")
        config = NativeCLMConfig(**payload["config"])
        routing = payload.get("routing_state", {})
        root_count = int(routing.get("lineage_root_count", config.initial_cells))
        model = cls(config, cell_count=int(payload["cell_count"]), lineage_root_count=root_count)
        model.load_state_dict(payload["state_dict"])
        return model, payload.get("extra", {})


def _root_route_probe(
    model: LineageNativeCLM,
    eval_paths: dict[str, str | Path],
    *,
    device: torch.device,
    config: NativeCLMM2Config,
) -> dict[str, str]:
    result: dict[str, str] = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for domain, path in eval_paths.items():
            digest = hashlib.sha256()
            loader = _loader(
                path,
                seq_len=model.config.max_seq_len,
                batch_size=config.batch_size,
                seed=91000 + ord(domain),
                num_workers=0,
            )
            for batch_idx, (x, _) in enumerate(loader):
                if batch_idx >= config.eval_batches:
                    break
                info = model(x.to(device), return_info=True)["cell_info"]
                root_idx = info["root_idx"].detach().cpu().contiguous()
                root_probs = info["root_probs"].detach().cpu().contiguous().float()
                digest.update(root_idx.numpy().tobytes())
                digest.update(root_probs.numpy().tobytes())
            result[domain] = digest.hexdigest()
    model.train(was_training)
    return result


def _select_lineage_leaf_parent(
    model: LineageNativeCLM,
    window: GrowthWindow,
    growth: NativeCLMM3GrowthConfig,
) -> dict[str, Any] | None:
    if window.mean_loss < growth.min_window_train_loss:
        return None
    candidates: list[dict[str, Any]] = []
    for cell_id in range(model.cell_count):
        if not model.is_lineage_leaf(cell_id):
            continue
        hits = window.route_hits[cell_id]
        if hits < growth.min_parent_route_hits_per_window:
            continue
        cell = model.cellular.cells[cell_id]
        if cell.rank < growth.min_parent_certificate_rank:
            continue
        ratio = window.ratio_weighted_sum[cell_id] / max(1, hits)
        if ratio > growth.max_projected_to_raw_gradient_ratio:
            continue
        score = hits * max(0.0, 1.0 - ratio)
        candidates.append(
            {
                "parent_id": cell_id,
                "route_hits": hits,
                "projection_ratio": float(ratio),
                "certificate_rank": cell.rank,
                "score": float(score),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def maybe_spawn_lineage_from_pressure(
    model: LineageNativeCLM,
    optimizer: torch.optim.Optimizer,
    window: GrowthWindow,
    growth: NativeCLMM3GrowthConfig,
    *,
    global_step: int,
    last_growth_step: int | None,
    spawned_count: int,
    probe_tokens: Tensor,
) -> dict[str, Any] | None:
    """Use the unchanged M3 pressure rule but allocate only under the routed leaf lineage."""

    if global_step % growth.growth_check_interval != 0:
        return None
    if spawned_count >= growth.max_new_cells or model.cell_count >= growth.max_final_cells:
        return None
    if last_growth_step is not None and global_step - last_growth_step < growth.growth_cooldown_steps:
        return None
    candidate = _select_lineage_leaf_parent(model, window, growth)
    if candidate is None:
        return None

    parent_id = int(candidate["parent_id"])
    hits = max(1, int(candidate["route_hits"]))
    route_key = window.query_sums[parent_id] / hits
    route_key = F.normalize(route_key.to(dtype=torch.float32), dim=0)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        before = model(probe_tokens, return_info=True)
        before_logits = before["logits"].detach().float()
        before_roots = before["cell_info"]["root_idx"]
        before_probs = before["cell_info"]["root_probs"]

    child_id = model.spawn_cell(
        parent_id=parent_id,
        route_key=route_key,
        inherit_scale=growth.inherit_scale,
    )
    child = model.cellular.cells[child_id]
    child.route_key.requires_grad_(False)
    child.weight.requires_grad_(True)
    optimizer.add_param_group({"params": [child.weight], "lr": optimizer.param_groups[0]["lr"]})
    window.ensure_cells(model.cell_count)

    with torch.no_grad():
        after = model(probe_tokens, return_info=True)
        after_logits = after["logits"].detach().float()
        after_roots = after["cell_info"]["root_idx"]
        after_probs = after["cell_info"]["root_probs"]
    model.train(was_training)

    delta = after_logits - before_logits
    return {
        "global_step": int(global_step),
        "parent_id": parent_id,
        "child_id": int(child_id),
        "parent_certificate_rank": int(candidate["certificate_rank"]),
        "parent_route_hits": int(candidate["route_hits"]),
        "parent_projection_ratio": float(candidate["projection_ratio"]),
        "pressure_score": float(candidate["score"]),
        "window_mean_train_loss": float(window.mean_loss),
        "child_initial_certificate_rank": int(child.rank),
        "inherit_scale": float(growth.inherit_scale),
        "birth_logits_max_abs_drift": float(delta.abs().max().cpu()),
        "birth_logits_mse": float(delta.square().mean().cpu()),
        "birth_root_topk_match": float((before_roots == after_roots).float().mean().cpu()),
        "birth_root_prob_max_abs_drift": float((before_probs - after_probs).abs().max().cpu()),
    }


def _train_lineage_phase(
    model: LineageNativeCLM,
    train_path: str | Path,
    *,
    device: torch.device,
    train_config: NativeCLMM2Config,
    growth_config: NativeCLMM3GrowthConfig,
    seed: int,
    phase: str,
    global_step_offset: int,
    growth_events: list[dict[str, Any]],
    child_post_birth_route_hits: dict[int, int],
    last_growth_step: int | None,
) -> tuple[dict[str, Any], int | None]:
    optimizer = _make_optimizer(model, train_config)
    loader = _loader(
        train_path,
        seq_len=model.config.max_seq_len,
        batch_size=train_config.batch_size,
        seed=seed,
        num_workers=train_config.num_workers,
    )
    iterator = _cycle(loader)
    scaler_enabled = device.type == "cuda" and train_config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    losses: list[float] = []
    projection_ratios_all: list[float] = []
    certificate_additions = 0
    start = time.time()
    window = GrowthWindow(model.config.d_model, model.cell_count)
    model.train()

    for step in range(1, train_config.steps_per_phase + 1):
        global_step = global_step_offset + step
        factor = _lr_factor(step - 1, train_config)
        for group in optimizer.param_groups:
            group["lr"] = train_config.lr_cells * factor

        x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, train_config.precision):
            out = model(x, y, return_info=True)
            loss = out["loss"]
        scaler.scale(loss).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)

        projection_ratios = model.project_cell_gradients_()
        projection_ratios_all.extend(projection_ratios.values())
        _observe_growth_window(
            model,
            window,
            out["cell_info"],
            projection_ratios,
            float(loss.detach().cpu()),
            child_post_birth_route_hits,
        )

        torch.nn.utils.clip_grad_norm_([cell.weight for cell in model.cellular.cells], train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if (
            train_config.certificate_update_interval > 0
            and step % train_config.certificate_update_interval == 0
        ):
            certificate_additions += model.update_certificates(out["cell_info"])

        if global_step % growth_config.growth_check_interval == 0:
            event = maybe_spawn_lineage_from_pressure(
                model,
                optimizer,
                window,
                growth_config,
                global_step=global_step,
                last_growth_step=last_growth_step,
                spawned_count=len(growth_events),
                probe_tokens=x,
            )
            if event is not None:
                growth_events.append(event)
                child_post_birth_route_hits[event["child_id"]] = 0
                last_growth_step = global_step
                print(
                    "[m3r lineage growth] step={step} parent={parent} child={child} "
                    "birth_max={drift:.3e}".format(
                        step=global_step,
                        parent=event["parent_id"],
                        child=event["child_id"],
                        drift=event["birth_logits_max_abs_drift"],
                    ),
                    flush=True,
                )
            window = GrowthWindow(model.config.d_model, model.cell_count)

        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % train_config.log_interval == 0 or step == train_config.steps_per_phase:
            print(
                f"[m3r {phase} lineage] step={step}/{train_config.steps_per_phase} "
                f"cells={model.cell_count} loss={losses[-1]:.6f}",
                flush=True,
            )

    return (
        {
            "phase": phase,
            "steps": train_config.steps_per_phase,
            "mean_train_loss": float(sum(losses) / len(losses)),
            "final_train_loss": float(losses[-1]),
            "projection_ratio_mean": float(sum(projection_ratios_all) / max(1, len(projection_ratios_all))),
            "projection_ratio_min": float(min(projection_ratios_all or [1.0])),
            "certificate_additions": int(certificate_additions),
            "elapsed_seconds": time.time() - start,
            "optimizer_reset_at_phase_start": True,
            "learner_replay_bytes": 0,
            "cell_count_end": model.cell_count,
        },
        last_growth_step,
    )


def run_global_growth_control(**kwargs) -> dict[str, Any]:
    """Run the frozen M3 global-pool growth algorithm unchanged as the M3R control."""

    summary = run_m3_arm(arm="growth_protected", **kwargs)
    summary = dict(summary)
    summary["source_arm"] = summary["arm"]
    summary["arm"] = "global_growth_control"
    output = Path(kwargs["output_dir"])
    (output / "arm-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_lineage_growth_arm(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    train_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    output_dir: str | Path,
    seed: int,
    train_config: NativeCLMM2Config,
    growth_config: NativeCLMM3GrowthConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    train_config.validate()
    growth_config.validate()
    checkpoint_path = Path(checkpoint_path)
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_checkpoint_sha256:
        raise RuntimeError(f"M1 checkpoint SHA mismatch: expected {expected_checkpoint_sha256}, got {actual_sha}")
    if tuple(train_paths) != ("B", "C", "D") or set(eval_paths) != {"A", "B", "C", "D"}:
        raise ValueError("registered M3R stream/evaluation must be B->C->D with A/B/C/D eval")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, m1_extra = LineageNativeCLM.load_checkpoint(checkpoint_path, map_location="cpu")
    if model.cell_count != 8 or model.lineage_root_count != 8 or model.config.active_cells != 2:
        raise RuntimeError("M3R requires canonical M1 topology: 8 root lineages / 2 active")
    if model.parameter_count()["total"] != 12_154_368:
        raise RuntimeError("M3R requires canonical 12,154,368-parameter M1 model")
    model.to(target_device)
    _freeze_to_cell_only(model)

    original_read_before = original_read_state_sha256(model)
    root_weights_before = root_cell_weight_sha256(model)
    root_probes: dict[str, dict[str, str]] = {
        "initial": _root_route_probe(model, eval_paths, device=target_device, config=train_config)
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrices: dict[str, Any] = {
        "initial": evaluate_matrix(model, eval_paths, device=target_device, config=train_config)
    }
    phase_summaries: list[dict[str, Any]] = []
    growth_events: list[dict[str, Any]] = []
    child_post_birth_route_hits: dict[int, int] = {}
    last_growth_step: int | None = None

    for index, phase in enumerate(("B", "C", "D")):
        phase_summary, last_growth_step = _train_lineage_phase(
            model,
            train_paths[phase],
            device=target_device,
            train_config=train_config,
            growth_config=growth_config,
            seed=seed + 100 * (index + 1),
            phase=phase,
            global_step_offset=index * train_config.steps_per_phase,
            growth_events=growth_events,
            child_post_birth_route_hits=child_post_birth_route_hits,
            last_growth_step=last_growth_step,
        )
        phase_summaries.append(phase_summary)
        matrices[f"after_{phase}"] = evaluate_matrix(model, eval_paths, device=target_device, config=train_config)
        root_probes[f"after_{phase}"] = _root_route_probe(
            model, eval_paths, device=target_device, config=train_config
        )

    original_read_after = original_read_state_sha256(model)
    final_checkpoint = output / "final.pt"
    model.save_checkpoint(
        final_checkpoint,
        extra={
            "milestone": "M3R",
            "arm": "lineage_growth",
            "seed": seed,
            "parent_checkpoint_sha256": actual_sha,
            "stream": ["B", "C", "D"],
            "learner_replay_bytes": 0,
            "growth_events": growth_events,
        },
    )

    summary = {
        "format": "minicells.native-clm-v0.m3r-arm-summary.v1",
        "arm": "lineage_growth",
        "seed": seed,
        "parent_checkpoint_sha256": actual_sha,
        "parent_m1_extra_keys": sorted(m1_extra.keys()),
        "initial_cell_count": 8,
        "lineage_root_count": 8,
        "final_cell_count": model.cell_count,
        "active_cells": model.config.active_cells,
        "learner_replay_bytes": 0,
        "cell_only_writes": True,
        "growth_controller_uses_phase_or_eval_labels": False,
        "shared_and_original_router_frozen": original_read_before == original_read_after,
        "original_read_state_sha256_before": original_read_before,
        "original_read_state_sha256_after": original_read_after,
        "root_cell_weight_sha256_before": root_weights_before,
        "root_cell_weight_sha256_after": root_cell_weight_sha256(model),
        "root_route_probes": root_probes,
        "lineage_chain_valid": True,
        "growth_events": growth_events,
        "spawned_cells": len(growth_events),
        "child_post_birth_route_hits": {str(key): int(value) for key, value in sorted(child_post_birth_route_hits.items())},
        "certificate": model.certificate_summary(),
        "stream": ["B", "C", "D"],
        "phase_summaries": phase_summaries,
        "evaluation_matrix": matrices,
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "final_checkpoint_bytes": final_checkpoint.stat().st_size,
        "training_config": asdict(train_config),
        "growth_config": asdict(growth_config),
    }
    (output / "arm-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _child_share(summary: dict[str, Any], domain: str, *, root_count: int = 8) -> float:
    usage = summary["evaluation_matrix"]["after_D"][domain]["cell_usage_share"]
    return float(sum(float(value) for value in usage[root_count:]))


def compare_m3r_arms(
    global_control: dict[str, Any],
    lineage: dict[str, Any],
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if global_control["seed"] != lineage["seed"]:
        raise ValueError("M3R arm seeds do not match")
    if global_control["parent_checkpoint_sha256"] != lineage["parent_checkpoint_sha256"]:
        raise ValueError("M3R arms did not start from the same M1 checkpoint")
    if global_control["growth_config"] != lineage["growth_config"]:
        raise ValueError("M3R must reuse the exact M3 growth-controller configuration")

    global_gains = {phase: _phase_gain(global_control, phase) for phase in ("B", "C", "D")}
    lineage_gains = {phase: _phase_gain(lineage, phase) for phase in ("B", "C", "D")}
    global_forgetting = _forgetting(global_control)
    lineage_forgetting = _forgetting(lineage)
    global_mean_gain = float(sum(global_gains.values()) / 3.0)
    lineage_mean_gain = float(sum(lineage_gains.values()) / 3.0)
    global_mean_forgetting = float(sum(global_forgetting.values()) / 3.0)
    lineage_mean_forgetting = float(sum(lineage_forgetting.values()) / 3.0)
    global_A = float(global_forgetting["A"])
    lineage_A = float(lineage_forgetting["A"])

    events = lineage["growth_events"]
    max_birth_logits = max((float(event["birth_logits_max_abs_drift"]) for event in events), default=float("inf"))
    max_birth_mse = max((float(event["birth_logits_mse"]) for event in events), default=float("inf"))
    min_birth_root_match = min((float(event["birth_root_topk_match"]) for event in events), default=0.0)
    max_birth_root_prob_drift = max((float(event["birth_root_prob_max_abs_drift"]) for event in events), default=float("inf"))

    initial_probe = lineage["root_route_probes"]["initial"]
    root_probe_invariant = all(
        probe[domain] == initial_probe[domain]
        for stage, probe in lineage["root_route_probes"].items()
        if stage != "initial"
        for domain in ("A", "B", "C", "D")
    )

    child_hits = [int(value) for value in lineage["child_post_birth_route_hits"].values()]
    reused = sum(hit >= thresholds["minimum_child_post_birth_route_hits"] for hit in child_hits)
    reuse_fraction = float(reused / max(1, len(child_hits))) if child_hits else 0.0
    max_active_fraction = max(
        float(metrics["active_fraction_vs_dense"])
        for stage in lineage["evaluation_matrix"].values()
        for metrics in stage.values()
    )

    global_A_child_share = _child_share(global_control, "A")
    lineage_child_shares = {domain: _child_share(lineage, domain) for domain in ("A", "B", "C", "D")}
    lineage_A_child_share = lineage_child_shares["A"]
    lineage_new_child_share = max(lineage_child_shares[domain] for domain in ("B", "C", "D"))
    selectivity_margin = lineage_new_child_share - lineage_A_child_share
    leakage_reduction = global_A_child_share - lineage_A_child_share

    gates = {
        "exact_same_m1_checkpoint": global_control["parent_checkpoint_sha256"] == lineage["parent_checkpoint_sha256"],
        "matched_seed_and_data_snapshot": global_control["seed"] == lineage["seed"],
        "zero_learner_replay": global_control["learner_replay_bytes"] == 0 and lineage["learner_replay_bytes"] == 0,
        "same_frozen_m3_growth_controller": global_control["growth_config"] == lineage["growth_config"],
        "shared_query_norm_and_root_keys_frozen": bool(global_control["shared_and_original_router_frozen"] and lineage["shared_and_original_router_frozen"]),
        "global_control_exposes_m3_read_failure": global_A >= thresholds["minimum_global_A_regression"],
        "global_growth_occurs_and_is_bounded": thresholds["minimum_spawned_cells"] <= global_control["spawned_cells"] <= thresholds["maximum_spawned_cells"],
        "lineage_growth_occurs_and_is_bounded": thresholds["minimum_spawned_cells"] <= lineage["spawned_cells"] <= thresholds["maximum_spawned_cells"],
        "birth_function_preserving": max_birth_logits <= thresholds["maximum_birth_logits_max_abs_drift"] and max_birth_mse <= thresholds["maximum_birth_logits_mse"],
        "birth_root_ownership_preserved": min_birth_root_match >= 1.0 and max_birth_root_prob_drift <= thresholds["maximum_birth_root_prob_drift"],
        "root_route_function_preserved": root_probe_invariant,
        "lineage_chain_valid": bool(lineage["lineage_chain_valid"]),
        "children_are_reused": reuse_fraction >= thresholds["minimum_child_reuse_fraction"],
        "child_read_leakage_reduced": leakage_reduction >= thresholds["minimum_A_child_share_reduction_vs_global"],
        "children_are_selective": selectivity_margin >= thresholds["minimum_child_selectivity_margin"],
        "sparse_compute_survives_growth": max_active_fraction <= thresholds["maximum_active_fraction_vs_dense"],
        "lineage_phase_plasticity": all(gain >= thresholds["minimum_phase_gain_each_B_C_D"] for gain in lineage_gains.values()),
        "lineage_absolute_A_retention": lineage_A <= thresholds["maximum_lineage_A_regression"],
        "lineage_A_retention_advantage": global_A - lineage_A >= thresholds["minimum_A_retention_advantage_vs_global"],
        "lineage_mean_forgetting": lineage_mean_forgetting <= thresholds["maximum_lineage_mean_forgetting"],
        "lineage_plasticity_preserved": lineage_mean_gain >= thresholds["minimum_lineage_to_global_plasticity_ratio"] * max(global_mean_gain, 1e-12),
    }
    return {
        "seed": global_control["seed"],
        "global_phase_gains": global_gains,
        "lineage_phase_gains": lineage_gains,
        "global_forgetting": global_forgetting,
        "lineage_forgetting": lineage_forgetting,
        "global_mean_plasticity": global_mean_gain,
        "lineage_mean_plasticity": lineage_mean_gain,
        "global_mean_forgetting": global_mean_forgetting,
        "lineage_mean_forgetting": lineage_mean_forgetting,
        "global_A_regression": global_A,
        "lineage_A_regression": lineage_A,
        "A_retention_advantage": global_A - lineage_A,
        "global_spawned_cells": global_control["spawned_cells"],
        "lineage_spawned_cells": lineage["spawned_cells"],
        "lineage_final_cell_count": lineage["final_cell_count"],
        "child_reuse_fraction": reuse_fraction,
        "max_birth_logits_max_abs_drift": max_birth_logits,
        "max_birth_logits_mse": max_birth_mse,
        "root_probe_invariant": root_probe_invariant,
        "global_A_child_share": global_A_child_share,
        "lineage_child_shares": lineage_child_shares,
        "A_child_share_reduction_vs_global": leakage_reduction,
        "child_selectivity_margin": selectivity_margin,
        "gates": gates,
        "pass": all(gates.values()),
    }


def aggregate_m3r_formal(
    seed_results: list[dict[str, Any]],
    *,
    protocol_sha256: str,
    formal_seeds: list[int],
    data_manifest_sha256: str,
) -> dict[str, Any]:
    completed = sorted(int(result["seed"]) for result in seed_results)
    if completed != sorted(formal_seeds):
        raise RuntimeError("formal M3R seed set is incomplete or unexpected")
    all_pass = all(result["pass"] for result in seed_results)
    return {
        "format": "minicells.native-clm-v0.m3r-decision.v1",
        "status": (
            "NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_SUPPORTED"
            if all_pass
            else "NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED"
        ),
        "scientific_decision": bool(all_pass),
        "protocol_sha256": protocol_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "formal_seeds": formal_seeds,
        "completed_seeds": completed,
        "seed_results": seed_results,
        "all_registered_gates_pass": bool(all_pass),
        "claim_boundary": (
            "12.15M Native CLM v0; same frozen M3 pressure controller; global-pool growth control "
            "vs immutable-root/lineage-chain read topology; protected Cell-local writes; B->C->D; zero replay"
        ),
        "next_milestone_if_supported": "M4 Cell ontology / specialization analysis",
        "next_if_not_supported": "diagnose lineage-local functional boundary/selectivity without reusing M3R formal seeds",
    }
