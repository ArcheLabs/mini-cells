"""M3L-2: persistent online rank-32 query address state for lineage routing."""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .native_clm_m2 import NativeCLMM2Config
from .native_clm_m3 import GrowthWindow, NativeCLMM3GrowthConfig, _loader
from .native_clm_m3l_gate import (
    LowRankGaussianSketch,
    derive_sketch_gate,
    sketch_covariance_matvec,
)
from .native_clm_m3r import LineageNativeCLM
from .native_clm_v0 import NativeCLMConfig


@dataclass(frozen=True)
class M3L2AddressConfig:
    rank: int = 32
    diagonal_regularization: float = 1e-4
    target_old_fpr: float = 0.1
    maximum_persistent_bytes_per_cell: int = 52360
    bootstrap_batches: int = 160
    max_queries_per_cell_per_batch: int = 256

    def validate(self) -> None:
        if self.rank != 32:
            raise ValueError("registered M3L-2 requires rank 32")
        if self.diagonal_regularization <= 0:
            raise ValueError("diagonal_regularization must be positive")
        if abs(self.target_old_fpr - 0.1) > 1e-12:
            raise ValueError("registered M3L-2 requires target_old_fpr=0.1")
        if self.maximum_persistent_bytes_per_cell != 52360:
            raise ValueError("registered M3L-2 persistent-state budget drift")
        if self.bootstrap_batches != 160:
            raise ValueError("registered M3L-2 requires 160 bootstrap batches")
        if self.max_queries_per_cell_per_batch != 256:
            raise ValueError("registered M3L-2 requires 256 sampled queries per cell per batch")


class MomentAccumulator:
    """Ephemeral bounded normalized-query first/second moments."""

    def __init__(self, width: int, *, device: torch.device | str = "cpu") -> None:
        self.count = 0
        self.sum = torch.zeros(width, dtype=torch.float32, device=device)
        self.second = torch.zeros(width, width, dtype=torch.float32, device=device)

    def update(self, values: Tensor, *, max_samples: int | None = None) -> None:
        if values.numel() == 0:
            return
        x = F.normalize(values.detach().to(self.sum.device, dtype=torch.float32), dim=-1)
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            x = x[:max_samples]
        if x.numel() == 0:
            return
        self.count += int(x.size(0))
        self.sum.add_(x.sum(dim=0))
        self.second.add_(x.transpose(0, 1).matmul(x))

    def add_sketch(self, sketch: LowRankGaussianSketch) -> None:
        if sketch.count < 1:
            return
        mean = sketch.mean.to(self.sum.device, dtype=torch.float32)
        basis = sketch.basis.to(self.sum.device, dtype=torch.float32)
        eig = sketch.eigenvalues.to(self.sum.device, dtype=torch.float32)
        residual = sketch.residual_variance.to(self.sum.device, dtype=torch.float32)
        cov = (basis * eig.unsqueeze(0)).matmul(basis.transpose(0, 1))
        cov = cov + torch.diag(residual)
        self.count += int(sketch.count)
        self.sum.add_(mean * sketch.count)
        self.second.add_(cov * max(1, sketch.count - 1) + sketch.count * torch.outer(mean, mean))

    def to_sketch(self, *, rank: int, diagonal_regularization: float) -> LowRankGaussianSketch:
        if self.count < 2:
            raise ValueError("at least two queries are required for an address sketch")
        mean = self.sum / self.count
        cov = (self.second - self.count * torch.outer(mean, mean)) / max(1, self.count - 1)
        cov = 0.5 * (cov + cov.transpose(0, 1))
        diagonal = torch.diagonal(cov).clamp_min(diagonal_regularization)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        take = min(rank, eigvals.numel())
        order = torch.argsort(eigvals, descending=True)[:take]
        values = eigvals[order].clamp_min(0)
        basis = eigvecs[:, order].contiguous()
        represented_diag = (basis.square() * values.unsqueeze(0)).sum(dim=1)
        residual = (diagonal - represented_diag).clamp_min(diagonal_regularization)
        return LowRankGaussianSketch(
            count=int(self.count),
            mean=mean.float().detach(),
            basis=basis.float().detach(),
            eigenvalues=values.float().detach(),
            residual_variance=residual.float().detach(),
        )

    def clear(self) -> None:
        self.count = 0
        self.sum.zero_()
        self.second.zero_()


def merge_sketch_and_moments(
    historical: LowRankGaussianSketch | None,
    current: MomentAccumulator,
    *,
    config: M3L2AddressConfig,
) -> LowRankGaussianSketch | None:
    if current.count < 2:
        return historical
    merged = MomentAccumulator(current.sum.numel(), device=current.sum.device)
    if historical is not None:
        merged.add_sketch(historical)
    merged.count += current.count
    merged.sum.add_(current.sum)
    merged.second.add_(current.second)
    return merged.to_sketch(rank=config.rank, diagonal_regularization=config.diagonal_regularization)


def _sketch_payload(sketch: LowRankGaussianSketch) -> dict[str, Any]:
    return {
        "count": sketch.count,
        "mean": sketch.mean.detach().cpu(),
        "basis": sketch.basis.detach().cpu(),
        "eigenvalues": sketch.eigenvalues.detach().cpu(),
        "residual_variance": sketch.residual_variance.detach().cpu(),
    }


def _sketch_from_payload(value: dict[str, Any]) -> LowRankGaussianSketch:
    return LowRankGaussianSketch(
        count=int(value["count"]),
        mean=value["mean"],
        basis=value["basis"],
        eigenvalues=value["eigenvalues"],
        residual_variance=value["residual_variance"],
    )


def _map_sketch(sketch: LowRankGaussianSketch, fn) -> LowRankGaussianSketch:
    return LowRankGaussianSketch(
        count=sketch.count,
        mean=fn(sketch.mean),
        basis=fn(sketch.basis),
        eigenvalues=fn(sketch.eigenvalues),
        residual_variance=fn(sketch.residual_variance),
    )


class OnlineAddressNativeCLM(LineageNativeCLM):
    """Immutable-root lineage model whose local edges use persistent affine gates."""

    address_config = M3L2AddressConfig()

    def __init__(
        self,
        config: NativeCLMConfig,
        *,
        cell_count: int | None = None,
        lineage_root_count: int | None = None,
    ) -> None:
        super().__init__(config, cell_count=cell_count, lineage_root_count=lineage_root_count)
        self.historical_sketches: dict[int, LowRankGaussianSketch] = {}
        self.affine_gates: dict[int, dict[str, Tensor | float]] = {}
        self.current_moments: dict[int, MomentAccumulator] = {}
        self.bootstrap_complete = False
        self.bootstrap_parameter_hash_before: str | None = None
        self.bootstrap_parameter_hash_after: str | None = None
        self.bootstrap_access_released = False
        self.bootstrap_sampled_queries = 0

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.historical_sketches = {
            cell_id: _map_sketch(sketch, fn)
            for cell_id, sketch in self.historical_sketches.items()
        }
        for gate in self.affine_gates.values():
            weight = gate.get("weight")
            if isinstance(weight, Tensor):
                gate["weight"] = fn(weight)
        for accumulator in self.current_moments.values():
            accumulator.sum = fn(accumulator.sum)
            accumulator.second = fn(accumulator.second)
        return self

    def _lineage_route_details(self, x: Tensor) -> dict[str, Tensor]:
        route_input = self.cellular.norm(x)
        query = F.normalize(self.cellular.query_proj(route_input), dim=-1)
        root_keys = torch.stack(
            [
                F.normalize(self.cellular.cells[index].route_key, dim=0)
                for index in range(self.lineage_root_count)
            ],
            dim=0,
        )
        root_scores = query.matmul(root_keys.transpose(0, 1)) / self.config.route_temperature
        k = min(self.config.active_cells, self.lineage_root_count)
        root_top_scores, root_idx = torch.topk(root_scores, k=k, dim=-1)
        root_probs = F.softmax(root_top_scores, dim=-1)
        concrete = root_idx.clone()
        for parent_id, child_id in sorted(self._direct_children().items()):
            gate = self.affine_gates.get(parent_id)
            if gate is None:
                continue
            weight = gate["weight"]
            if not isinstance(weight, Tensor):
                raise TypeError("M3L-2 affine gate weight must be a Tensor")
            score = query.matmul(weight.to(query.device, query.dtype)) + float(gate["bias"])
            switch = (concrete == parent_id) & (score > float(gate["threshold"])).unsqueeze(-1)
            concrete = torch.where(switch, torch.full_like(concrete, child_id), concrete)
        return {
            "route_input": route_input,
            "query": query,
            "top_idx": concrete,
            "top_probs": root_probs,
            "top_scores": root_top_scores,
            "root_idx": root_idx,
            "root_probs": root_probs,
            "root_scores": root_scores,
        }

    def checkpoint_payload(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = super().checkpoint_payload(extra=extra)
        payload["routing_state"] = {
            "mode": "root_lineage_affine_address",
            "lineage_root_count": self.lineage_root_count,
            "address_rank": self.address_config.rank,
        }
        payload["address_state"] = {
            "config": asdict(self.address_config),
            "historical_sketches": {
                str(cell_id): _sketch_payload(sketch)
                for cell_id, sketch in self.historical_sketches.items()
            },
            "affine_gates": {
                str(cell_id): {
                    "weight": (
                        gate["weight"].detach().cpu()
                        if isinstance(gate["weight"], Tensor)
                        else gate["weight"]
                    ),
                    "bias": float(gate["bias"]),
                    "threshold": float(gate["threshold"]),
                    "old_score_mean": float(gate.get("old_score_mean", 0.0)),
                    "old_score_std": float(gate.get("old_score_std", 0.0)),
                }
                for cell_id, gate in self.affine_gates.items()
            },
            "bootstrap_complete": self.bootstrap_complete,
            "bootstrap_parameter_hash_before": self.bootstrap_parameter_hash_before,
            "bootstrap_parameter_hash_after": self.bootstrap_parameter_hash_after,
            "bootstrap_access_released": self.bootstrap_access_released,
            "bootstrap_sampled_queries": self.bootstrap_sampled_queries,
        }
        return payload

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple[OnlineAddressNativeCLM, dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format") != "minicells.native-clm-v0.checkpoint.v1":
            raise ValueError("unsupported Native CLM checkpoint format")
        config = NativeCLMConfig(**payload["config"])
        routing = payload.get("routing_state", {})
        root_count = int(routing.get("lineage_root_count", config.initial_cells))
        model = cls(config, cell_count=int(payload["cell_count"]), lineage_root_count=root_count)
        model.load_state_dict(payload["state_dict"])
        state = payload.get("address_state", {})
        if state.get("config"):
            model.address_config = M3L2AddressConfig(**state["config"])
            model.address_config.validate()
        model.historical_sketches = {
            int(cell_id): _sketch_from_payload(value)
            for cell_id, value in state.get("historical_sketches", {}).items()
        }
        model.affine_gates = {
            int(cell_id): dict(value)
            for cell_id, value in state.get("affine_gates", {}).items()
        }
        model.bootstrap_complete = bool(state.get("bootstrap_complete", False))
        model.bootstrap_parameter_hash_before = state.get("bootstrap_parameter_hash_before")
        model.bootstrap_parameter_hash_after = state.get("bootstrap_parameter_hash_after")
        model.bootstrap_access_released = bool(state.get("bootstrap_access_released", False))
        model.bootstrap_sampled_queries = int(state.get("bootstrap_sampled_queries", 0))
        return model, payload.get("extra", {})

    def address_state_metrics(self) -> dict[str, Any]:
        ranks = [sketch.rank for sketch in self.historical_sketches.values()]
        sizes = [sketch.storage_bytes for sketch in self.historical_sketches.values()]
        return {
            "sketch_count": len(ranks),
            "maximum_rank": max(ranks or [0]),
            "maximum_bytes_per_cell": max(sizes or [0]),
            "gate_count": len(self.affine_gates),
            "bootstrap_complete": self.bootstrap_complete,
            "bootstrap_access_released": self.bootstrap_access_released,
            "bootstrap_sampled_queries": self.bootstrap_sampled_queries,
        }


def parameter_sha256(model: LineageNativeCLM) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0]):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _extract_query(model: OnlineAddressNativeCLM, cell_input: Tensor) -> Tensor:
    with torch.no_grad():
        return F.normalize(model.cellular.query_proj(model.cellular.norm(cell_input)), dim=-1)


def observe_online_queries(model: OnlineAddressNativeCLM, info: dict[str, Any]) -> None:
    query = _extract_query(model, info["cell_input"])
    top_idx = info["top_idx"].to(query.device)
    for cell_id in torch.unique(top_idx).tolist():
        cell_id = int(cell_id)
        if not model.is_lineage_leaf(cell_id):
            continue
        mask = (top_idx == cell_id).any(dim=-1)
        values = query[mask]
        if values.numel() == 0:
            continue
        accumulator = model.current_moments.setdefault(
            cell_id,
            MomentAccumulator(model.config.d_model, device=query.device),
        )
        accumulator.update(
            values,
            max_samples=model.address_config.max_queries_per_cell_per_batch,
        )


def bootstrap_address_state(
    model: OnlineAddressNativeCLM,
    bootstrap_path: str | Path,
    *,
    device: torch.device,
    train_config: NativeCLMM2Config,
    address_config: M3L2AddressConfig,
    sampling_seed: int = 74001,
) -> dict[str, Any]:
    address_config.validate()
    model.address_config = address_config
    before = parameter_sha256(model)
    accumulators = {
        index: MomentAccumulator(model.config.d_model, device=device)
        for index in range(model.lineage_root_count)
    }
    loader = _loader(
        bootstrap_path,
        seq_len=model.config.max_seq_len,
        batch_size=train_config.batch_size,
        seed=sampling_seed,
        num_workers=0,
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_idx, (tokens, _) in enumerate(loader):
            if batch_idx >= address_config.bootstrap_batches:
                break
            tokens = tokens.to(device)
            positions = torch.arange(tokens.size(1), device=device)
            hidden = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
            hidden = model.dropout(hidden)
            for index, block in enumerate(model.blocks):
                hidden = block(hidden)
                if index != model.config.cellular_layer_index:
                    continue
                details = model._lineage_route_details(hidden)
                query = details["query"]
                roots = details["root_idx"]
                for root in range(model.lineage_root_count):
                    mask = (roots == root).any(dim=-1)
                    if bool(mask.any()):
                        accumulators[root].update(
                            query[mask],
                            max_samples=address_config.max_queries_per_cell_per_batch,
                        )
                break
    model.train(was_training)
    model.historical_sketches = {
        root: accumulator.to_sketch(
            rank=address_config.rank,
            diagonal_regularization=address_config.diagonal_regularization,
        )
        for root, accumulator in accumulators.items()
        if accumulator.count >= 2
    }
    after = parameter_sha256(model)
    model.bootstrap_complete = True
    model.bootstrap_parameter_hash_before = before
    model.bootstrap_parameter_hash_after = after
    model.bootstrap_sampled_queries = sum(accumulator.count for accumulator in accumulators.values())
    if before != after:
        raise RuntimeError("M3L-2 bootstrap mutated Native CLM parameters")
    if len(model.historical_sketches) != model.lineage_root_count:
        raise RuntimeError("M3L-2 bootstrap failed to construct all root address states")
    for sketch in model.historical_sketches.values():
        if (
            sketch.rank > address_config.rank
            or sketch.storage_bytes > address_config.maximum_persistent_bytes_per_cell
        ):
            raise RuntimeError("M3L-2 bootstrap address state exceeds registered bound")
    return {
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "root_sketches": len(model.historical_sketches),
        "sampled_queries": model.bootstrap_sampled_queries,
    }


def _commit_nonspawn_windows(
    model: OnlineAddressNativeCLM,
    *,
    except_cell: int | None = None,
) -> None:
    for cell_id, accumulator in list(model.current_moments.items()):
        if cell_id == except_cell:
            continue
        if not model.is_lineage_leaf(cell_id):
            model.current_moments.pop(cell_id, None)
            continue
        merged = merge_sketch_and_moments(
            model.historical_sketches.get(cell_id),
            accumulator,
            config=model.address_config,
        )
        if merged is not None:
            model.historical_sketches[cell_id] = merged
        accumulator.clear()


def _estimated_gate_activation(
    sketch: LowRankGaussianSketch,
    gate: dict[str, Tensor | float],
) -> float:
    weight = gate["weight"]
    if not isinstance(weight, Tensor):
        raise TypeError("M3L-2 affine gate weight must be a Tensor")
    weight = weight.to(sketch.mean.device, sketch.mean.dtype)
    mean = torch.dot(weight, sketch.mean) + float(gate["bias"])
    variance = torch.dot(weight, sketch_covariance_matvec(sketch, weight)).clamp_min(1e-12)
    z = (float(gate["threshold"]) - float(mean.detach().cpu())) / math.sqrt(
        float(variance.detach().cpu())
    )
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def maybe_spawn_online_address(
    model: OnlineAddressNativeCLM,
    optimizer: torch.optim.Optimizer,
    window: GrowthWindow,
    growth: NativeCLMM3GrowthConfig,
    *,
    global_step: int,
    last_growth_step: int | None,
    spawned_count: int,
    probe_tokens: Tensor,
) -> dict[str, Any] | None:
    import minicells.native_clm_m3r as m3r

    if spawned_count >= growth.max_new_cells or model.cell_count >= growth.max_final_cells:
        _commit_nonspawn_windows(model)
        return None
    if last_growth_step is not None and global_step - last_growth_step < growth.growth_cooldown_steps:
        _commit_nonspawn_windows(model)
        return None
    candidate = m3r._select_lineage_leaf_parent(model, window, growth)
    if candidate is None:
        _commit_nonspawn_windows(model)
        return None
    parent_id = int(candidate["parent_id"])
    current = model.current_moments.get(parent_id)
    old = model.historical_sketches.get(parent_id)
    if current is None or current.count < 2 or old is None:
        _commit_nonspawn_windows(model)
        return None
    current_sketch = current.to_sketch(
        rank=model.address_config.rank,
        diagonal_regularization=model.address_config.diagonal_regularization,
    )
    gate = derive_sketch_gate(
        old,
        current_sketch,
        diagonal_regularization=model.address_config.diagonal_regularization,
        target_old_fpr=model.address_config.target_old_fpr,
    )

    was_training = model.training
    model.eval()
    with torch.no_grad():
        before = model(probe_tokens, return_info=True)
        before_logits = before["logits"].detach().float()
        before_roots = before["cell_info"]["root_idx"]
        before_probs = before["cell_info"]["root_probs"]
    route_key = F.normalize(current_sketch.mean.to(dtype=torch.float32), dim=0)
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
    model.affine_gates[parent_id] = gate
    model.historical_sketches[child_id] = current_sketch
    model.current_moments.pop(parent_id, None)
    _commit_nonspawn_windows(model)
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
        "address_parent_rank": old.rank,
        "address_child_rank": current_sketch.rank,
        "address_child_bytes": current_sketch.storage_bytes,
        "estimated_old_gate_activation": _estimated_gate_activation(old, gate),
        "estimated_current_gate_activation": _estimated_gate_activation(current_sketch, gate),
    }


@dataclass
class _BootstrapLease:
    path: Path | None
    consumed: bool = False

    def consume(self) -> Path:
        if self.path is None or self.consumed:
            raise RuntimeError("M3L-2 bootstrap may be consumed exactly once")
        path = self.path
        self.path = None
        self.consumed = True
        return path

    @property
    def accessible(self) -> bool:
        return self.path is not None


@contextlib.contextmanager
def _patched_m3r_online(
    bootstrap_path: str | Path,
    address_config: M3L2AddressConfig,
    train_config: NativeCLMM2Config,
):
    import minicells.native_clm_m3r as m3r

    address_config.validate()
    original_cls = m3r.LineageNativeCLM
    original_observe = m3r._observe_growth_window
    original_spawn = m3r.maybe_spawn_lineage_from_pressure
    original_freeze = m3r._freeze_to_cell_only
    lease = _BootstrapLease(Path(bootstrap_path))

    OnlineAddressNativeCLM.address_config = address_config

    def freeze(model: LineageNativeCLM) -> None:
        original_freeze(model)
        if isinstance(model, OnlineAddressNativeCLM):
            device = next(model.parameters()).device
            bootstrap_address_state(
                model,
                lease.consume(),
                device=device,
                train_config=train_config,
                address_config=address_config,
            )
            model.bootstrap_access_released = not lease.accessible

    def observe(model, window, info, ratios, loss, child_hits):
        original_observe(model, window, info, ratios, loss, child_hits)
        if isinstance(model, OnlineAddressNativeCLM):
            observe_online_queries(model, info)

    m3r.LineageNativeCLM = OnlineAddressNativeCLM
    m3r._freeze_to_cell_only = freeze
    m3r._observe_growth_window = observe
    m3r.maybe_spawn_lineage_from_pressure = maybe_spawn_online_address
    try:
        yield lease
    finally:
        m3r.LineageNativeCLM = original_cls
        m3r._freeze_to_cell_only = original_freeze
        m3r._observe_growth_window = original_observe
        m3r.maybe_spawn_lineage_from_pressure = original_spawn


def run_online_address_state_arm(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    bootstrap_path: str | Path,
    train_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    output_dir: str | Path,
    seed: int,
    train_config: NativeCLMM2Config,
    growth_config: NativeCLMM3GrowthConfig,
    device: str,
    address_config: M3L2AddressConfig | None = None,
) -> dict[str, Any]:
    import minicells.native_clm_m3r as m3r

    config = address_config or M3L2AddressConfig()
    config.validate()
    with _patched_m3r_online(bootstrap_path, config, train_config) as lease:
        summary = m3r.run_lineage_growth_arm(
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            train_paths=train_paths,
            eval_paths=eval_paths,
            output_dir=output_dir,
            seed=seed,
            train_config=train_config,
            growth_config=growth_config,
            device=device,
        )
    if not lease.consumed or lease.accessible:
        raise RuntimeError("M3L-2 bootstrap access was not released before continual learning")

    final_path = Path(output_dir) / "final.pt"
    restored, _ = OnlineAddressNativeCLM.load_checkpoint(final_path, map_location="cpu")
    metrics = restored.address_state_metrics()
    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    address_payload = payload.get("address_state", {})
    roundtrip = bool(
        address_payload
        and metrics["gate_count"] == len(summary.get("growth_events", []))
        and metrics["bootstrap_complete"]
        and metrics["bootstrap_access_released"]
        and address_payload.get("config", {}).get("rank") == config.rank
    )
    summary["arm"] = "online_address_state"
    summary["format"] = "minicells.native-clm-v0.m3l2-arm-summary.v1"
    summary["address_state"] = metrics
    summary["address_state_checkpoint_roundtrip"] = roundtrip
    summary["bootstrap"] = {
        "complete": restored.bootstrap_complete,
        "parameter_sha256_before": restored.bootstrap_parameter_hash_before,
        "parameter_sha256_after": restored.bootstrap_parameter_hash_after,
        "A_access_after_continual_start": not restored.bootstrap_access_released,
        "access_released_before_continual_start": restored.bootstrap_access_released,
        "sampled_queries": restored.bootstrap_sampled_queries,
    }
    (Path(output_dir) / "arm-summary.json").write_text(
        __import__("json").dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
