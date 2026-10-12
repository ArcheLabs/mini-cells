"""Selected-only execution with a biased selected-value straight-through gradient.

The forward gate is exactly one. Its backward derivative is the derivative of
the chosen destination probability. Unchosen Cells are never evaluated: this
is NOT the gradient of a dense counterfactual mixture. STOP is absorbing.
"""
from __future__ import annotations

import torch
from experiments.cell_internal_routing_002.models.cell_router import CellInternalRouter


class HardCellTransition(CellInternalRouter):
    def __init__(self, config):
        super().__init__(config)
        self.temperature = 2.0
        self.mode = 'soft'

    def _soft(self, x):
        # Same conditional-mean relaxation as 002, with scheduled temperature.
        # Diagnostic scalar conversions are kept out of timed inference.
        state = x.float()
        n, experts = x.shape[0], len(self.cells)
        live = state.new_zeros(n, experts)
        live[:, 0] = 1
        stopped = torch.zeros_like(state)
        expected = state.new_zeros(n)
        transitions = state.new_zeros(experts + 1)
        entropy = state.new_zeros(())
        mass_total = state.new_zeros(())
        usage = state.new_zeros(experts)
        stops, forced = state.new_zeros(()), state.new_zeros(())
        histogram = []
        observed = self.training or self.collect_stats
        for hop in range(self.max_hops):
            expected = expected + live.sum(-1)
            usage = usage + live.sum(0)
            next_live, next_state = torch.zeros_like(live), torch.zeros_like(state)
            stop_mass = state.new_zeros(n)
            for index in (range(1) if hop == 0 else range(experts)):
                predict = hop + 1 < self.max_hops
                updated, logits = self.cells[index](state.to(x.dtype), predict_next=predict)
                updated = updated.float()
                weight = live[:, index]
                if not predict:
                    stopped = stopped + weight[:, None] * updated
                    forced = forced + weight.sum()
                    continue
                probabilities = (logits / self.temperature).softmax(-1)
                weighted = weight[:, None] * probabilities
                next_live = next_live + weighted[:, :experts]
                next_state = next_state + weighted[:, :experts].sum(-1, keepdim=True) * updated
                stopped = stopped + weighted[:, experts, None] * updated
                stop_mass = stop_mass + weighted[:, experts]
                transitions = transitions + weighted.sum(0)
                if observed:
                    entropy = entropy - (weighted * probabilities.clamp_min(1e-9).log()).sum()
                    mass_total = mass_total + weight.sum()
            if self.collect_stats:
                histogram.append(float((live.sum() if hop + 1 == self.max_hops else stop_mass.sum()).detach()))
            stops = stops + stop_mass.sum()
            live = next_live
            state = next_state / live.sum(-1, keepdim=True).clamp_min(1e-12)
        self.expected_path = expected.mean()
        self.last_entropy = entropy.detach() / mass_total.detach().clamp_min(1e-12) if observed else None
        destination = transitions[:-1] / transitions[:-1].sum().clamp_min(1e-12)
        if self.collect_stats:
            visits = n * (1 + experts * (self.max_hops - 1))
            routers = n * (1 + experts * max(0, self.max_hops - 2)) if self.max_hops > 1 else 0
            self._record(n, usage, entropy, mass_total, expected.sum(), visits,
                         n * (experts if self.max_hops > 1 else 1), routers,
                         n * (experts if self.max_hops > 2 else 1) if routers else 0,
                         stops, forced, histogram)
        return (stopped - x.float()).to(x.dtype), experts * destination.square().sum()

    def _selected(self, x):
        n, experts = x.shape[0], len(self.cells)
        state = x
        current = torch.zeros(n, device=x.device, dtype=torch.long)
        active = torch.ones(n, device=x.device, dtype=torch.bool)
        # Differentiable survival cost, forward equal to actual visits.
        live_gate = x.new_ones(n, dtype=torch.float32)
        visits = x.new_zeros(n, dtype=torch.float32)
        entropy_sum = x.new_zeros((), dtype=torch.float32)
        destination_mass = x.new_zeros(experts, dtype=torch.float32)
        decisions = 0
        observed = self.training or self.collect_stats
        if self.collect_stats:
            usage = x.new_zeros(experts, dtype=torch.float32)
            paths = torch.full((min(n, self.sample_limit), self.max_hops), -1,
                               device=x.device, dtype=torch.long)
            stop_hop = torch.full((min(n, self.sample_limit),), -1,
                                  device=x.device, dtype=torch.long)
            learned = x.new_zeros((), dtype=torch.float32)
            histogram = x.new_zeros(self.max_hops, dtype=torch.float32)
        for hop in range(self.max_hops):
            at_hop = current
            next_current, next_active = current.clone(), active.clone()
            for index, cell in enumerate(self.cells):
                rows = torch.where(active & (at_hop == index))[0]
                if rows.numel() == 0:
                    continue
                predict = hop + 1 < self.max_hops
                updated, logits = cell(state[rows], predict_next=predict)
                if observed:
                    visits = visits.index_add(0, rows, live_gate[rows])
                if predict:
                    if observed:
                        probs = (logits / self.temperature).softmax(-1)
                        selected = probs.argmax(-1)
                        entropy_sum = entropy_sum - (probs * probs.clamp_min(1e-9).log()).sum()
                        decisions += rows.numel()
                        if self.training:
                            destination_mass = destination_mass + probs[:, :experts].sum(0)
                            chosen = probs.gather(1, selected[:, None]).squeeze(1)
                            gate = 1.0 + (chosen - chosen.detach())
                            # Functional scatter preserves autograd across revisits and STOP.
                            updated = updated * gate[:, None].to(updated.dtype)
                            survival = probs[:, :experts].sum(-1)
                            survival_gate = 1.0 + (survival - survival.detach())
                            live_gate = live_gate.index_copy(0, rows, live_gate[rows] * survival_gate)
                    else:
                        # Argmax is invariant under softmax and positive temperature.
                        # Timing has no telemetry, softmax, STE or survival-cost overhead.
                        selected = logits.argmax(-1)
                    next_current = next_current.index_copy(0, rows, selected)
                    next_active = next_active.index_copy(0, rows, selected != experts)
                else:
                    next_active = next_active.index_fill(0, rows, False)
                state = state.index_copy(0, rows, updated.to(state.dtype))
                if self.collect_stats:
                    usage[index] += rows.numel()
                    sampled = rows[rows < paths.shape[0]]
                    paths[sampled, hop] = index
                    ended = rows[~next_active[rows]]
                    histogram[hop] += ended.numel()
                    if predict:
                        learned += ended.numel()
                        ended = ended[ended < paths.shape[0]]
                        stop_hop[ended] = hop
            current, active = next_current, next_active
        if observed:
            self.expected_path = visits.mean()
            self.last_path_length = visits.detach().mean()
            self.last_entropy = entropy_sum.detach() / max(decisions, 1)
        if self.collect_stats:
            self.token_count += n
            self.usage = usage.detach() if self.usage is None else self.usage + usage.detach()
            self.path_sum += float(visits.detach().sum())
            self.executed_sum += int(visits.detach().sum())
            self.entropy_sum = (entropy_sum.detach() if self.entropy_sum is None
                                else self.entropy_sum + entropy_sum.detach())
            self.transition_mass += decisions
            self.stop_sum += float(learned)
            self.forced_sum += n - float(learned)
            self.histogram = [a + b for a, b in zip(self.histogram, histogram.cpu().tolist())]
            remaining = self.sample_limit - len(self.samples)
            for row, ended in zip(paths[:remaining].cpu().tolist(), stop_hop[:remaining].cpu().tolist()):
                path = [cell for cell in row if cell >= 0]
                path.append('STOP' if ended >= 0 else 'FORCED_STOP')
                self.samples.append(path)
        destination = destination_mass / destination_mass.sum().clamp_min(1e-12)
        balance = experts * destination.square().sum() if self.training else x.new_zeros(())
        return state - x, balance

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        if self.mode == 'hard':
            delta, balance = self._selected(flat)
        else:
            delta, balance = self._soft(flat)
            self.last_path_length = self.expected_path.detach()
        return delta.reshape_as(x), balance

    def stats(self):
        if self.mode != 'hard':
            return super().stats()
        if not self.token_count:
            raise RuntimeError('Missing hard routing evidence')
        counts = self.usage.cpu().tolist()
        return {'mode': 'hard', 'validation_tokens': self.token_count,
                'assignments': counts, 'utilization': [value / sum(counts) for value in counts],
                'expected_path_length': self.path_sum / self.token_count,
                'executed_cell_visits_per_token': self.executed_sum / self.token_count,
                'routing_entropy_nats': float(self.entropy_sum) / max(1, self.transition_mass),
                'learned_stop_fraction': self.stop_sum / self.token_count,
                'forced_stop_fraction': self.forced_sum / self.token_count,
                'path_length_distribution': [value / self.token_count for value in self.histogram],
                'sample_paths': self.samples,
                'sample_scope': 'first validation tokens; STOP and FORCED_STOP distinguished'}
