"""Score only A*c candidates; ST is a biased training approximation."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class EdgeBatch:
    sources: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor
    slots: torch.Tensor
    entropy: torch.Tensor


class SparseRouter(nn.Module):
    def __init__(self, dim, arm, buckets=64):
        super().__init__()
        self.arm = arm
        if arm == 'B2':
            self.source_bias = nn.Embedding(buckets, 8)
            self.target_bias = nn.Embedding(buckets, 8)
        elif arm == 'B3':
            self.query = nn.Linear(dim, dim)
        elif arm != 'B1':
            self.scorer = nn.Sequential(nn.Linear(2 * dim, 16), nn.Tanh(), nn.Linear(16, 1))

    def route(self, active_ids, candidates, state, instance, ledger, *, k,
              straight_through=False, intervention=None, rng=None, frozen=None,
              forced_slots=None):
        ids, targets = candidates.ids, candidates.states
        source = state.index_select(0, active_ids)
        if self.arm == 'B1':
            scores = -instance.x[ids, 6]  # fixed semantic priority, no runtime state
            ledger.add('route_score_ops', ids.numel())
        elif self.arm == 'B2':
            a = self.source_bias(active_ids % self.source_bias.num_embeddings)[:, None]
            b = self.target_bias(ids % self.target_bias.num_embeddings)
            scores = (a * b).sum(-1)
            ledger.add('route_score_ops', ids.numel() * 16)
        elif self.arm == 'B3':
            # A shared central query controller gets exactly the same private reads.
            query = self.query(source)
            scores = (query[:, None] * targets).sum(-1) / source.shape[-1] ** 0.5
            ledger.linear(self.query, active_ids.numel())
            query_ops = 2 * active_ids.numel() * self.query.in_features * self.query.out_features
            ledger.add('route_linear_ops', query_ops)
            ledger.add('route_score_ops', query_ops + ids.numel() * (2 * source.shape[-1] + 1))
        else:
            inputs = torch.cat([source[:, None].expand_as(targets), targets], -1)
            scores = self.scorer(inputs).squeeze(-1)
            for layer in self.scorer:
                if isinstance(layer, nn.Linear):
                    ledger.linear(layer, ids.numel())
                    ledger.add('route_score_ops', 2 * ids.numel() * layer.in_features * layer.out_features)
                    ledger.add('route_linear_ops', 2 * ids.numel() * layer.in_features * layer.out_features)
        probabilities = torch.softmax(scores, -1)
        slots = scores.topk(k, -1).indices
        if forced_slots is not None:
            slots = torch.as_tensor(forced_slots, device=ids.device)
            if slots.shape != (active_ids.numel(), k):
                raise ValueError('replay degree mismatch')
        if intervention == 'random_dynamic':
            slots = torch.rand(ids.shape, generator=rng, device=ids.device).topk(k, -1).indices
        elif intervention == 'rewire':
            # Rotate selected destinations in each candidate permutation. Exactly k distinct.
            offset = int(torch.randint(1, ids.shape[1], (), generator=rng, device=ids.device))
            slots = (slots + offset) % ids.shape[1]
        elif intervention == 'freeze':
            if frozen is None:
                raise ValueError('freeze requires preferences collected from training inputs only')
            priority = frozen.to(ids.device)[active_ids[:, None], ids]
            ledger.add('index_reads', ids.numel())
            slots = priority.topk(k, -1).indices
        weights = torch.full_like(slots, 1 / k, dtype=state.dtype)
        if straight_through:
            soft_selected = probabilities.gather(1, slots)
            weights = weights + soft_selected - soft_selected.detach()
        chosen = ids.gather(1, slots)
        ledger.add('schedule_items', ids.numel())
        return EdgeBatch(active_ids[:, None].expand_as(chosen).flatten(), chosen.flatten(),
                         weights.flatten(), slots,
                         -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean())
