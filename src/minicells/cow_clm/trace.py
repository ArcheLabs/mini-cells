from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .runtime import COWCLMError, ExpertSite


@dataclass(frozen=True)
class ExpertTraceStat:
    site: ExpertSite
    hits: int
    token_share: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.site.layer,
            "expert": self.site.expert,
            "hits": self.hits,
            "token_share": self.token_share,
        }


def _extract_router_logits(module: nn.Module, output: Any) -> torch.Tensor:
    num_experts = int(getattr(module, "num_experts", 0))
    if num_experts <= 0:
        raise COWCLMError("Granite router does not expose a positive num_experts")
    values = (output,) if isinstance(output, torch.Tensor) else tuple(output or ())
    matches = [
        value
        for value in values
        if isinstance(value, torch.Tensor)
        and torch.is_floating_point(value)
        and value.ndim in (2, 3)
        and int(value.shape[-1]) == num_experts
    ]
    if len(matches) != 1:
        raise COWCLMError(
            f"expected exactly one router-logit tensor with width {num_experts}, found {len(matches)}"
        )
    return matches[0].detach()


@contextlib.contextmanager
def capture_granite_router_logits(model: nn.Module) -> Iterator[list[torch.Tensor | None]]:
    """Capture per-layer Granite router logits without requiring top-level model outputs.

    Transformers 5.x Granite MoE blocks consume the router's logits internally and may not
    propagate them through the CausalLM output. This hook observes the router return value only;
    returning ``None`` from the hook preserves the original forward output exactly.
    """
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise COWCLMError("Granite model.layers not found for router tracing")
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles: list[Any] = []

    for layer_index, layer in enumerate(layers):
        block = getattr(layer, "block_sparse_moe", None)
        router = getattr(block, "router", None)
        if router is None:
            raise COWCLMError(f"Granite router not found at layer {layer_index}")

        def hook(
            module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            index: int = layer_index,
        ) -> None:
            if captured[index] is not None:
                raise COWCLMError(f"Granite router at layer {index} ran more than once in one trace scope")
            captured[index] = _extract_router_logits(module, output)

        handles.append(router.register_forward_hook(hook))

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def require_captured_router_logits(
    captured: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor, ...]:
    missing = [index for index, value in enumerate(captured) if value is None]
    if missing:
        raise COWCLMError(f"Granite router logits missing for layer {missing[0]}")
    return tuple(value for value in captured if value is not None)


def summarize_router_logits(
    router_logits: Sequence[torch.Tensor],
    *,
    top_k: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[ExpertTraceStat, ...]:
    """Turn per-layer Granite router logits into ranked (layer, expert) activation counts."""
    if top_k <= 0:
        raise COWCLMError("top_k must be positive")
    stats: list[ExpertTraceStat] = []
    for layer, logits in enumerate(router_logits):
        values = logits.detach()
        if values.ndim == 3:
            batch, sequence, experts = values.shape
            flat = values.reshape(batch * sequence, experts)
            if attention_mask is None:
                valid = torch.ones(batch * sequence, dtype=torch.bool, device=values.device)
            else:
                if tuple(attention_mask.shape) != (batch, sequence):
                    raise COWCLMError("attention mask does not match router logits")
                valid = attention_mask.to(device=values.device, dtype=torch.bool).reshape(-1)
            flat = flat[valid]
        elif values.ndim == 2:
            flat = values
            if attention_mask is not None and flat.shape[0] == attention_mask.numel():
                valid = attention_mask.to(device=values.device, dtype=torch.bool).reshape(-1)
                flat = flat[valid]
        else:
            raise COWCLMError("router logits must be rank 2 or 3")
        if flat.shape[0] == 0:
            continue
        if top_k > flat.shape[-1]:
            raise COWCLMError("top_k exceeds number of experts")
        selected = flat.topk(top_k, dim=-1).indices
        counts = torch.bincount(selected.reshape(-1), minlength=flat.shape[-1])
        denominator = max(int(flat.shape[0]) * top_k, 1)
        for expert, count in enumerate(counts.tolist()):
            stats.append(
                ExpertTraceStat(
                    site=ExpertSite(layer=layer, expert=expert),
                    hits=int(count),
                    token_share=float(count / denominator),
                )
            )
    return tuple(sorted(stats, key=lambda item: (-item.hits, item.site.layer, item.site.expert)))


def top_expert_sites(stats: Sequence[ExpertTraceStat], count: int) -> tuple[ExpertSite, ...]:
    if count <= 0:
        raise COWCLMError("requested expert-site count must be positive")
    ranked = [item.site for item in stats if item.hits > 0]
    if len(ranked) < count:
        raise COWCLMError(
            f"only {len(ranked)} expert sites were activated, cannot select {count}"
        )
    return tuple(ranked[:count])
