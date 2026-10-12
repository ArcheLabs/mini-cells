"""Persistent graph prior plus contextual transition heads and backup-gradient STE.

Hard evaluation computes only selected Cells. Hard training evaluates the chosen
branch plus one alternative (STOP is an identity branch). Forward is exactly the
chosen branch; backward is a renormalized top-k counterfactual mixture. This is a
biased surrogate, not the derivative of argmax, and its extra training work is
reported separately from inference work.
"""
import torch
import weakref
from torch import nn
from minicells.routing.hard_transition import HardCellTransition


class GraphCell(nn.Module):
    def __init__(self, cell, owner, index):
        super().__init__()
        self.compute, self.router = cell.compute, cell.router
        # The matrix is registered once by GraphRouter, not once per Cell.
        object.__setattr__(self, '_owner', weakref.ref(owner))
        self.index = index

    def forward(self, hidden, predict_next=True):
        updated = hidden + self.compute(hidden)
        logits = self.router(updated).float() + self._owner().graph_logits[self.index] if predict_next else None
        return updated, logits


class GraphRouter(HardCellTransition):
    def __init__(self, config, cells=None):
        super().__init__(config)
        experts = config['experts']
        self.graph_logits = nn.Parameter(torch.zeros(experts, experts + 1))
        self.cells = nn.ModuleList(GraphCell(cell, self, index)
                                   for index, cell in enumerate(self.cells if cells is None else cells))
        self.backup_k = config['graph_backup_k']
        self.backup_floor = config['graph_backup_floor']
        if not 0 < self.backup_floor < 1 / self.backup_k:
            raise ValueError('Backup floor must provide positive counterfactual gradients')
        if not 2 <= self.backup_k <= experts + 1:
            raise ValueError('Graph hard training requires selected + at least one backup branch')
        self.last_training_cell_visits = None

    def _backup(self, x):
        n, experts = x.shape[0], len(self.cells)
        state, logits = self.cells[0](x, predict_next=self.max_hops > 1)
        active = torch.ones(n, device=x.device, dtype=torch.bool)
        expected = x.new_ones(n, dtype=torch.float32)
        entropy, mass = x.new_zeros(()), x.new_zeros(experts)
        count, executed = 0, n
        hard_visits = x.new_tensor(float(n), dtype=torch.float32)
        for hop in range(1, self.max_hops):
            rows = torch.where(active)[0]
            if rows.numel() == 0:
                break
            previous = state[rows]
            probabilities = (logits[rows].float() / self.temperature).softmax(-1)
            weights, candidates = probabilities.topk(self.backup_k, dim=-1)
            weights = weights.clamp_min(self.backup_floor)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            selected = candidates[:, 0]
            hard_visits = hard_visits + (selected != experts).sum()
            # STOP has zero compute; expected survival is differentiable at each live decision.
            expected = expected.index_add(0, rows, probabilities[:, :experts].sum(-1))
            entropy = entropy - (probabilities * probabilities.clamp_min(1e-9).log()).sum()
            mass = mass + probabilities[:, :experts].sum(0)
            count += rows.numel()
            hard_value = previous.clone()  # STOP is identity.
            mixture = torch.zeros_like(previous)
            next_logits = previous.new_zeros(rows.numel(), experts + 1, dtype=torch.float32)
            for destination in range(experts + 1):
                member, rank = torch.where(candidates == destination)
                if member.numel() == 0:
                    continue
                if destination == experts:
                    value = previous[member]
                    predicted = None
                else:
                    predicts = hop + 1 < self.max_hops
                    # Selected and backup share one batched Cell call; only selected logits survive.
                    value, predicted = self.cells[destination](previous[member], predict_next=predicts)
                    executed += member.numel()
                mixture = mixture.index_add(0, member, value * weights[member, rank, None].to(value.dtype))
                chosen = rank == 0
                chosen_rows = member[chosen]
                hard_value = hard_value.index_copy(0, chosen_rows, value[chosen])
                if predicted is not None:
                    next_logits = next_logits.index_copy(0, chosen_rows, predicted[chosen])
            surrogate = hard_value.detach() + (mixture - mixture.detach())
            state = state.index_copy(0, rows, surrogate)
            logits = logits.index_copy(0, rows, next_logits)
            active = active.index_copy(0, rows, selected != experts)
        self.expected_path = expected.mean()
        self.last_path_length = hard_visits.detach() / n
        self.last_entropy = entropy.detach() / max(count, 1)
        self.last_training_cell_visits = executed / n
        distribution = mass / mass.sum().clamp_min(1e-12)
        return state - x, experts * distribution.square().sum()

    def forward(self, x):
        if self.mode == 'hard' and self.training:
            delta, balance = self._backup(x.reshape(-1, x.shape[-1]))
            return delta.reshape_as(x), balance
        return super().forward(x)

    def stats(self):
        result = super().stats()
        probabilities = self.graph_logits.detach().float().softmax(-1)
        result.update(graph_prior_probabilities=probabilities.cpu().tolist(),
                      graph_prior_entropy_nats=(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)).cpu().tolist(),
                      graph_prior_scope='learned context-independent row prior; actual transitions also use token state',
                      graph_prior_active_edge_fraction=float((probabilities >= 0.05).float().mean()),
                      graph_prior_active_edge_threshold=0.05)
        return result
