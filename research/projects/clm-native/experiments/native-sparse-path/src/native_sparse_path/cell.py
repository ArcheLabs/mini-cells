"""One shared message/update operator, evaluated on gathered nodes only."""
import torch
from torch import nn


class CellDynamics(nn.Module):
    def __init__(self, dim, features, hidden):
        super().__init__()
        self.message = nn.Linear(dim, dim)
        self.update = nn.Sequential(nn.Linear(2 * dim + features, hidden), nn.Tanh(),
                                    nn.Linear(hidden, dim))
        self.norm = nn.LayerNorm(dim)

    def step(self, state, edges, instance, ledger, *, active_limit,
             zero_message=False, redistribute=False, controller_only=False,
             initial=None):
        # Uniqueness/sorting touches at most A*k indices, never scans N states.
        targets = edges.targets.unique()
        priorities = instance.x.index_select(0, targets)[:, 6]
        targets = targets[priorities.argsort()[:active_limit]]
        keep = (edges.targets[:, None] == targets[None, :]).any(1)
        src, dst, weights = edges.sources[keep], edges.targets[keep], edges.weights[keep]
        unique_sources, inverse = src.unique(return_inverse=True)
        values = torch.tanh(self.message(state.index_select(0, unique_sources)))[inverse]
        ledger.linear(self.message, unique_sources.numel())
        if zero_message:
            values = torch.zeros_like(values)
        if redistribute:
            values = values.roll(1, 0)
        packets = values * weights[:, None]
        target_slots = (dst[:, None] == targets[None, :]).long().argmax(1)
        incoming = state.new_zeros((targets.numel(), state.shape[-1]))
        incoming = incoming.index_add(0, target_slots, packets)
        resident = (initial if controller_only else state).index_select(0, targets)
        features = instance.x.index_select(0, targets)
        delta = self.update(torch.cat([resident, incoming, features], -1))
        for layer in self.update:
            if isinstance(layer, nn.Linear):
                ledger.linear(layer, targets.numel())
        updated = self.norm(resident + delta)
        ledger.add('activation_ops_estimated', updated.numel() * 8 + packets.numel() * 2)
        ledger.add('messages', dst.numel())
        ledger.add('cell_updates', targets.numel())
        ledger.add('state_reads', (unique_sources.numel() + targets.numel()) * state.shape[-1])
        ledger.add('state_writes', updated.numel())
        ledger.add('node_gathers', unique_sources.numel() + targets.numel())
        ledger.add('schedule_items', edges.targets.numel() * targets.numel())
        if torch.is_grad_enabled():
            ledger.add('state_copy_bytes', state.numel() * state.element_size())
            result = state.index_copy(0, targets, updated)
        else:
            result = state.index_copy_(0, targets, updated)
        return result, targets, src, dst, packets
