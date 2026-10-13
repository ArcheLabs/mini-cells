"""Bounded post-compute transitions, absorbing STOP, and explicit execution accounting.

Soft training merges live histories into a conditional mean state each hop. This
is a differentiable mean-field relaxation, not enumeration of all graph paths.
Hard inference executes exactly one Cell per live token per hop.
"""
from __future__ import annotations

import torch
from torch import nn
from experiments.common.models import FeedForward, RoutedFeedForward, LanguageModel as BaseModel


class Cell(nn.Module):
    def __init__(self, width, hidden, destinations):
        super().__init__()
        self.compute = FeedForward(width, hidden)
        self.router = nn.Linear(width, destinations, bias=False)

    def forward(self, hidden, predict_next=True):
        updated = hidden + self.compute(hidden)
        logits = self.router(updated).float() if predict_next else None
        return updated, logits


class FixedCells(nn.Module):
    """Four deterministic residual Cells; same FFN capacity as the eight-Cell bank."""
    def __init__(self, config):
        super().__init__()
        self.cells = nn.ModuleList(FeedForward(config['width'], config['expert_hidden'] * 2)
                                   for _ in range(4))

    def forward(self, x):
        hidden = x
        for cell in self.cells:
            hidden = hidden + cell(hidden)
        return hidden - x


class CellInternalRouter(RoutedFeedForward):
    """Sequential transition policy implementing the shared routed-FFN contract."""
    def __init__(self, config):
        super().__init__()
        self.cells = nn.ModuleList(Cell(config['width'], config['expert_hidden'],
                                       config['experts'] + 1) for _ in range(config['experts']))
        self.max_hops = config['max_hops']
        self.mode = 'soft'
        self.collect_stats = False
        self.sample_limit = config.get('path_samples', 32)
        self.reset_stats()

    def reset_stats(self):
        self.token_count = 0
        self.usage = None
        self.entropy_sum = None
        self.transition_mass = 0.0
        self.path_sum = 0.0
        self.executed_sum = 0
        self.unique_sum = 0.0
        self.router_executed_sum = 0
        self.router_unique_sum = 0.0
        self.stop_sum = 0.0
        self.forced_sum = 0.0
        self.samples = []
        self.histogram = [0.0] * self.max_hops

    def _soft(self, x):
        state = x.float()
        count, width = state.shape
        experts = len(self.cells)
        live = torch.zeros(count, experts, device=x.device)
        live[:, 0] = 1  # Fixed entry; subsequent destinations are learned.
        stopped = torch.zeros_like(state)
        usage = torch.zeros(experts, device=x.device)
        transitions = torch.zeros(experts + 1, device=x.device)
        expected = state.new_zeros(count)
        entropy = state.new_zeros(())
        transition_mass = state.new_zeros(())
        stop_total = state.new_zeros(())
        hist = []
        executed = 0
        router_executed = 0
        forced = state.new_zeros(())
        for hop in range(self.max_hops):
            mass = live.sum(-1)
            expected = expected + mass
            usage = usage + live.sum(0)
            next_live = torch.zeros_like(live)
            next_state = torch.zeros_like(state)
            stop_mass = state.new_zeros(count)
            # First hop has exactly one supported Cell; remaining soft hops compute all.
            candidates = range(1) if hop == 0 else range(experts)
            for index in candidates:
                predicts_next = hop < self.max_hops - 1
                updated, logits = self.cells[index](state.to(x.dtype), predict_next=predicts_next)
                updated = updated.float()
                weight = live[:, index]
                executed += count
                if predicts_next:
                    router_executed += count
                if hop == self.max_hops - 1:
                    stopped = stopped + weight[:, None] * updated
                    forced = forced + weight.sum()
                    continue
                probs = logits.softmax(-1)
                weighted = weight[:, None] * probs
                next_live = next_live + weighted[:, :experts]
                next_state = next_state + weighted[:, :experts].sum(-1, keepdim=True) * updated
                stopped = stopped + weighted[:, experts, None] * updated
                stop_mass = stop_mass + weighted[:, experts]
                entropy = entropy - (weighted * probs.clamp_min(1e-9).log()).sum()
                transitions = transitions + weighted.sum(0)
                transition_mass = transition_mass + weight.sum()
            hist.append(float((mass if hop == self.max_hops - 1 else stop_mass).sum().detach()))
            stop_total = stop_total + stop_mass.sum()
            live = next_live
            # Conditional mean among surviving paths; stopped output is absorbing.
            state = next_state / live.sum(-1, keepdim=True).clamp_min(1e-12)
        self.expected_path = expected.mean()
        destination = transitions[:-1] / transitions[:-1].sum().clamp_min(1e-12)
        balance = experts * destination.square().sum()
        if self.collect_stats:
            self._record(count, usage, entropy, transition_mass, expected.sum(), executed,
                         count * (experts if self.max_hops > 1 else 1), router_executed,
                         count * (experts if self.max_hops > 2 else 1) if self.max_hops > 1 else 0,
                         stop_total, forced, hist)
        return (stopped - x.float()).to(x.dtype), balance

    def _hard(self, x):
        n, width = x.shape
        state = x.clone()
        active = torch.ones(n, dtype=torch.bool, device=x.device)
        current = torch.zeros(n, dtype=torch.long, device=x.device)
        visits = torch.zeros(n, dtype=torch.long, device=x.device)
        seen = torch.zeros(n, len(self.cells), dtype=torch.bool, device=x.device)
        router_seen = torch.zeros_like(seen)
        paths = torch.full((min(n, self.sample_limit), self.max_hops), -1,
                           dtype=torch.long, device=x.device)
        usage = torch.zeros(len(self.cells), device=x.device)
        entropy = x.new_zeros((), dtype=torch.float32)
        transition_mass = x.new_zeros((), dtype=torch.float32)
        stops = x.new_zeros((), dtype=torch.float32)
        hist = []
        forced = x.new_zeros((), dtype=torch.float32)
        router_executed = 0
        for hop in range(self.max_hops):
            next_active = active.clone()
            current_at_hop = current.clone()
            before = active.sum()
            for index, cell in enumerate(self.cells):
                rows = torch.where(active & (current_at_hop == index))[0]
                if not rows.numel():
                    continue
                predicts_next = hop < self.max_hops - 1
                updated, logits = cell(state[rows], predict_next=predicts_next)
                state[rows] = updated.to(state.dtype)
                seen[rows, index] = True
                visits[rows] += 1
                usage[index] += rows.numel()
                if self.collect_stats:
                    sample_rows = rows[rows < paths.shape[0]]
                    paths[sample_rows, hop] = index
                if predicts_next:
                    router_seen[rows, index] = True
                    router_executed += rows.numel()
                if hop == self.max_hops - 1:
                    forced += rows.numel()
                    next_active[rows] = False
                else:
                    probs = logits.softmax(-1)
                    destination = probs.argmax(-1)
                    entropy -= (probs * probs.clamp_min(1e-9).log()).sum()
                    transition_mass += rows.numel()
                    current[rows] = destination
                    stopped = destination == len(self.cells)
                    stops += stopped.sum()
                    next_active[rows[stopped]] = False
            hist.append(float((before - next_active.sum()).detach()))
            active = next_active
            if not active.any():
                hist.extend([0.0] * (self.max_hops - len(hist)))
                break
        self.expected_path = visits.float().mean()
        if self.collect_stats:
            self._record(n, usage, entropy, transition_mass, visits.sum(), int(visits.sum()),
                         seen.sum(), router_executed, router_seen.sum(),
                         stops, forced, hist)
            remaining = self.sample_limit - len(self.samples)
            self.samples.extend([[i for i in row if i >= 0]
                                 for row in paths[:remaining].cpu().tolist()])
        return state - x, x.new_zeros(())

    def _record(self, n, usage, entropy, transition_mass, path, executed, unique,
                router_executed, router_unique, stops, forced, hist):
        self.token_count += n
        self.usage = usage.detach() if self.usage is None else self.usage + usage.detach()
        self.entropy_sum = entropy.detach() if self.entropy_sum is None else self.entropy_sum + entropy.detach()
        self.transition_mass += float(transition_mass.detach())
        self.path_sum += float(path.detach())
        self.executed_sum += executed
        self.unique_sum += float(unique.detach()) if torch.is_tensor(unique) else unique
        self.router_executed_sum += router_executed
        self.router_unique_sum += float(router_unique.detach()) if torch.is_tensor(router_unique) else router_unique
        self.stop_sum += float(stops.detach())
        self.forced_sum += float(forced.detach())
        self.histogram = [a + b for a, b in zip(self.histogram, hist)]

    def stats(self):
        if not self.token_count or self.usage is None:
            raise RuntimeError('Cell internal routing evidence is empty')
        counts = self.usage.cpu().tolist()
        return {'mode': self.mode, 'validation_tokens': self.token_count,
                'assignments': counts, 'utilization': [n / sum(counts) for n in counts],
                'routing_entropy_nats': float(self.entropy_sum) / max(self.transition_mass, 1e-12),
                'expected_path_length': self.path_sum / self.token_count,
                'executed_cell_visits_per_token': self.executed_sum / self.token_count,
                'unique_cell_weights_per_token': self.unique_sum / self.token_count,
                'executed_router_visits_per_token': self.router_executed_sum / self.token_count,
                'unique_router_weights_per_token': self.router_unique_sum / self.token_count,
                'learned_stop_fraction': self.stop_sum / self.token_count,
                'forced_stop_fraction': self.forced_sum / self.token_count,
                'path_length_distribution': [n / self.token_count for n in self.histogram],
                'sample_paths': self.samples,
                'sample_scope': 'first validation tokens, fixed entry 0; path differences are descriptive',
                'soft_usage_scope': 'probability mass; execution still computes all supported Cells'}

    def forward(self, x):
        if self.mode == 'hard' and self.training:
            raise RuntimeError('Top-1 is an inference evaluation, not differentiable training')
        flat = x.reshape(-1, x.shape[-1])
        delta, balance = self._hard(flat) if self.mode == 'hard' else self._soft(flat)
        return delta.reshape_as(x), balance


class LanguageModel(BaseModel):
    def __init__(self, config, kind):
        super().__init__(config, 'global-router' if kind == 'cell-internal' else
                         'fixed' if kind == 'fixed-cell' else kind)
        self.kind = kind
        if kind in {'cell-internal', 'fixed-cell'}:
            for block in self.blocks:
                block.ffn = CellInternalRouter(config) if kind == 'cell-internal' else FixedCells(config)
                block.ffn.apply(self._initialize)

    def compute_penalty(self):
        routers = self.routing_modules() if self.kind == 'cell-internal' else []
        return sum(router.expected_path for router in routers) / len(routers) if routers else self.embedding.weight.new_zeros(())

    def set_routing_mode(self, mode):
        for router in self.routing_modules():
            if isinstance(router, CellInternalRouter):
                router.mode = mode

    def parameter_metrics(self):
        if self.kind == 'global-router':
            return super().parameter_metrics()
        c = self.config
        width = c['width']
        total = sum(p.numel() for p in self.parameters())
        fixed = self.kind == 'fixed-cell'
        visits = 8 if fixed else 1 + c['experts'] * (c['max_hops'] - 1)
        router_visits = (0 if fixed or c['max_hops'] <= 1 else
                         1 + c['experts'] * max(0, c['max_hops'] - 2))
        per_layer = 8 * width * width + 4 * c['sequence_length'] * width + 4 * width * c['expert_hidden'] * visits
        if not fixed:
            per_layer += 2 * width * (c['experts'] + 1) * router_visits
        return {'params': total, 'activated_params_per_token_estimate': total,
                'forward_flops_per_token_estimate': c['layers'] * per_layer + 2 * width * c['vocab_size'],
                'flops_estimate_scope': 'matmuls only; includes repeated soft Cell and router execution, excludes dispatch/norms/softmax',
                'fixed_path_length': 4 if fixed else None,
                'soft_executed_cell_visits_per_layer': None if fixed else visits}
