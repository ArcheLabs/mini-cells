"""Exact last-block activation cache and full-vs-cached-tail gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


class CacheSemanticsInvalid(RuntimeError):
    """Raised when a cache cannot reproduce the full model tail."""


@dataclass
class TailCache:
    mlp_input: Tensor
    pre_mlp_residual: Tensor
    input_ids: Tensor | None = None
    attention_mask: Tensor | None = None
    label_positions: Tensor | None = None
    labels: Tensor | None = None
    loss_mask: Tensor | None = None
    top_k_index: Tensor | None = None
    top_k_weights: Tensor | None = None
    sample_ids: tuple[str, ...] = ()
    split: str | None = None
    identity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mlp_input": self.mlp_input,
            "pre_mlp_residual": self.pre_mlp_residual,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "label_positions": self.label_positions,
            "labels": self.labels,
            "loss_mask": self.loss_mask,
            "top_k_index": self.top_k_index,
            "top_k_weights": self.top_k_weights,
            "sample_ids": list(self.sample_ids),
            "split": self.split,
            "identity": dict(self.identity or {}),
        }


@dataclass(frozen=True)
class CacheEquivalence:
    top1_agreement: float
    relative_l2: float
    max_abs: float
    mean_abs: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "top1_agreement": self.top1_agreement,
            "relative_l2": self.relative_l2,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "passed": self.passed,
        }


def _logits(output: Any) -> Tensor:
    value = getattr(output, "logits", output)
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, Tensor):
        raise CacheSemanticsInvalid("model output has no logits tensor")
    return value


def _target_parts(model: nn.Module, target_layer_path: str | None = None) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None or len(layers) == 0:
        raise CacheSemanticsInvalid("model has no decoder layers")
    if target_layer_path:
        target = model
        for part in target_layer_path.split("."):
            target = getattr(target, part)
    else:
        target = layers[-1]
    if target is not layers[-1]:
        raise CacheSemanticsInvalid("cached-tail runner requires the final decoder block")
    norm = getattr(root, "norm", None)
    head = getattr(model, "lm_head", None)
    post_norm = getattr(target, "post_attention_layernorm", None)
    moe = getattr(target, "block_sparse_moe", None)
    if any(item is None for item in (norm, head, post_norm, moe)):
        raise CacheSemanticsInvalid("unsupported decoder block tail structure")
    return target, post_norm, moe, norm


class CachedTailRunner:
    """Replays only ``residual + MoE + final norm + LM head`` for last block."""

    def __init__(self, model: nn.Module, target_layer_path: str | None = None) -> None:
        self.model = model
        self.target, self.post_norm, self.moe, self.final_norm = _target_parts(model, target_layer_path)
        self.lm_head = getattr(model, "lm_head")
        self.residual_multiplier = float(getattr(self.target, "residual_multiplier", 1.0))

    @torch.no_grad()
    def capture(self, input_ids: Tensor, attention_mask: Tensor | None = None, sample_ids: tuple[str, ...] = (), **kwargs: Any) -> TailCache:
        state: dict[str, Tensor] = {}

        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            if not args or not isinstance(args[0], Tensor) or not isinstance(output, Tensor):
                raise CacheSemanticsInvalid("post-attention norm hook saw an unexpected signature")
            state["pre_mlp_residual"] = args[0].detach().clone()
            state["mlp_input"] = output.detach().clone()

        def route_hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, Tensor):
                raise CacheSemanticsInvalid("router hook saw no routing tensor")
            if value.ndim < 2:
                raise CacheSemanticsInvalid("router output is not token-by-expert")
            if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) and value.shape[-1] <= 16:
                state["top_k_index"] = value.detach().clone()
                if isinstance(output, (tuple, list)) and len(output) > 1 and isinstance(output[1], Tensor):
                    state["top_k_weights"] = output[1].detach().clone()
                return
            probabilities = torch.softmax(value.float(), dim=-1)
            top_k = int(getattr(module, "top_k", getattr(self.moe, "top_k", 0)))
            if not top_k:
                top_k = int(getattr(getattr(self.moe, "config", None), "num_experts_per_tok", 0))
            if not top_k:
                raise CacheSemanticsInvalid("cannot infer parent router top-k")
            weights, indices = torch.topk(probabilities, top_k, dim=-1)
            state["top_k_index"] = indices.detach().clone()
            state["top_k_weights"] = (weights / weights.sum(dim=-1, keepdim=True)).detach().clone()

        handle = self.post_norm.register_forward_hook(hook)
        router_handle = self.moe.router.register_forward_hook(route_hook)
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        finally:
            handle.remove()
            router_handle.remove()
        if not {"pre_mlp_residual", "mlp_input"}.issubset(state):
            raise CacheSemanticsInvalid("target block did not produce a complete tail cache")
        return TailCache(
            mlp_input=state["mlp_input"],
            pre_mlp_residual=state["pre_mlp_residual"],
            input_ids=input_ids.detach().clone(),
            attention_mask=attention_mask.detach().clone() if attention_mask is not None else None,
            top_k_index=state.get("top_k_index"),
            top_k_weights=state.get("top_k_weights"),
            sample_ids=tuple(sample_ids),
        )

    @torch.no_grad()
    def forward(self, cache: TailCache) -> Tensor:
        moe_output = self._forward_moe(cache, getattr(self.moe, "experts", self.moe))
        block_output = cache.pre_mlp_residual + moe_output * self.residual_multiplier
        return self.lm_head(self.final_norm(block_output))

    def _forward_moe(self, cache: TailCache, experts: nn.Module) -> Tensor:
        if cache.top_k_index is None or cache.top_k_weights is None:
            return self.moe(cache.mlp_input)
        shape = cache.mlp_input.shape
        hidden = cache.mlp_input.reshape(-1, shape[-1])
        routed = experts(hidden, cache.top_k_index.reshape(-1, cache.top_k_index.shape[-1]), cache.top_k_weights.reshape(-1, cache.top_k_weights.shape[-1]))
        return routed.reshape(shape)

    def forward_with_experts(self, cache: TailCache, experts: nn.Module) -> Tensor:
        moe_output = self._forward_moe(cache, experts)
        block_output = cache.pre_mlp_residual + moe_output * self.residual_multiplier
        return self.lm_head(self.final_norm(block_output))

    @torch.no_grad()
    def verify(self, cache: TailCache, full_logits: Tensor | None = None, tolerance: float = 1e-5) -> CacheEquivalence:
        if full_logits is None:
            if cache.input_ids is None:
                raise CacheSemanticsInvalid("full logits are required when cache has no input_ids")
            full_logits = _logits(self.model(input_ids=cache.input_ids, attention_mask=cache.attention_mask))
        tail_logits = self.forward(cache)
        delta = tail_logits.float() - full_logits.float()
        relative_l2 = float(delta.norm() / full_logits.float().norm().clamp_min(1e-12))
        agreement = float((tail_logits.argmax(-1) == full_logits.argmax(-1)).float().mean())
        result = CacheEquivalence(
            top1_agreement=agreement,
            relative_l2=relative_l2,
            max_abs=float(delta.abs().max()),
            mean_abs=float(delta.abs().mean()),
            passed=agreement == 1.0 and relative_l2 <= tolerance,
        )
        if not result.passed:
            raise CacheSemanticsInvalid(
                f"CACHE_SEMANTICS_INVALID: agreement={agreement}, relative_l2={relative_l2}"
            )
        return result


def save_cache(
    cache: TailCache,
    directory: Path,
    shard_rows: int = 128,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save CPU shards and a manifest; branches never need the whole cache on GPU."""
    if shard_rows <= 0:
        raise ValueError("shard_rows must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    rows = int(cache.mlp_input.shape[0])
    shards = []
    for index, start in enumerate(range(0, rows, shard_rows)):
        end = min(rows, start + shard_rows)
        payload = {
            "mlp_input": cache.mlp_input[start:end].cpu(),
            "pre_mlp_residual": cache.pre_mlp_residual[start:end].cpu(),
            "input_ids": cache.input_ids[start:end].cpu() if cache.input_ids is not None else None,
            "attention_mask": cache.attention_mask[start:end].cpu() if cache.attention_mask is not None else None,
            "labels": cache.labels[start:end].cpu() if cache.labels is not None else None,
            "loss_mask": cache.loss_mask[start:end].cpu() if cache.loss_mask is not None else None,
            "top_k_index": cache.top_k_index[start:end].cpu() if cache.top_k_index is not None and cache.top_k_index.shape[0] == rows else cache.top_k_index.cpu() if cache.top_k_index is not None else None,
            "top_k_weights": cache.top_k_weights[start:end].cpu() if cache.top_k_weights is not None and cache.top_k_weights.shape[0] == rows else cache.top_k_weights.cpu() if cache.top_k_weights is not None else None,
        }
        path = directory / f"shard-{index:05d}.pt"
        torch.save(payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        shards.append({"path": path.name, "start": start, "end": end, "sha256": digest})
    manifest = {
        "schema": "minicells.pcu-kill-001.cache.v2",
        "rows": rows,
        "dtype": str(cache.mlp_input.dtype),
        "sample_ids": list(cache.sample_ids),
        "split": cache.split,
        "identity": dict(cache.identity or {}),
        "shards": shards,
    }
    if identity:
        manifest.update(dict(identity))
    (directory / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_cache_identity(manifest: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Validate that a cache belongs to the exact frozen engineering inputs."""
    required = ("foundation_model", "foundation_revision", "foundation_tensor_sha256", "target_path", "target_layer", "dataset_manifest_sha256", "split", "template_version", "encoding_version", "dtype")
    values = {**dict(manifest), **dict(manifest.get("identity", {}))}
    missing = [key for key in required if values.get(key) in (None, "", [])]
    if missing:
        raise CacheSemanticsInvalid(f"cache identity is incomplete: {missing}")
    mismatched = [key for key in expected if values.get(key) != expected[key]]
    if mismatched:
        raise CacheSemanticsInvalid(f"cache identity mismatch: {mismatched}")
