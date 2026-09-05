from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

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
