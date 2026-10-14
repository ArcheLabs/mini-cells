import torch
from torch import nn
from .accounting import Ledger
from .candidate import CandidateProvider
from .cell import CellDynamics
from .generator import FEATURES
from .route import SparseRouter

ARMS = ('B0', 'B1', 'B2', 'C0', 'B3')


class NativeModel(nn.Module):
    def __init__(self, config, arm='C0'):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown arm')
        self.arm, self.dim, self.config = arm, config['dim'], dict(config)
        # Independent component RNGs keep common operators identical at a given seed.
        seed = torch.initial_seed()
        def initialize(factory, offset):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + offset)
                return factory()
        self.encoder = initialize(lambda: nn.Linear(FEATURES, self.dim), 0)
        self.router = initialize(lambda: SparseRouter(self.dim, arm), 1)
        # Compensate absent/smaller router with actual update capacity, not dummy weights.
        hidden = 64 + (10 if arm == 'B1' else 1 if arm == 'B2' else 0)
        self.cell = initialize(lambda: CellDynamics(self.dim, FEATURES, hidden), 2)
        self.readout = initialize(lambda: nn.Linear(self.dim, 2), 3)
        self.provider = CandidateProvider(config['c'])

    def forward(self, instance, *, trace=False, intervention=None,
                rng=None, frozen=None, force_hard=False, forced_slots=None):
        if instance.depth > self.config['rounds']:
            raise ValueError('instance exceeds maximum rounds')
        if instance.x.shape[0] < self.config['c']:
            raise ValueError('insufficient address space')
        ledger = Ledger()
        initial = torch.tanh(self.encoder(instance.x))
        ledger.state_element_size = initial.element_size()
        if self.training and initial.requires_grad:
            initial.retain_grad()
            self._diagnostic_initial = initial
        ledger.linear(self.encoder, instance.x.shape[0])
        state = initial.clone()
        ledger.add('state_copy_bytes', initial.numel() * initial.element_size())
        active = instance.roots
        records, entropies = [], []
        visited = []
        for step in range(instance.depth):
            candidates = self.provider.candidates(active, state, instance, ledger)
            edges = self.router.route(active, candidates, state, instance, ledger,
                k=self.config['k'] if instance.task == 'B' else 1,
                straight_through=self.training and not force_hard,
                intervention=intervention, rng=rng, frozen=frozen,
                forced_slots=None if forced_slots is None else forced_slots[step])
            old = state.clone() if trace else state
            if trace:
                ledger.add('state_copy_bytes', state.numel() * state.element_size())
            state, next_active, src, dst, packets = self.cell.step(state, edges, instance,
                ledger, active_limit=self.config['active_limit'],
                zero_message=intervention == 'zero_message',
                redistribute=intervention == 'redistribute',
                controller_only=self.arm == 'B0', initial=initial)
            entropies.append(edges.entropy)
            if trace:
                visited.extend(dst.detach().cpu().tolist())
                records.append({'step': step, 'sources': active.detach().cpu().tolist(),
                    'targets': next_active.detach().cpu().tolist(),
                    'edges': torch.stack([src, dst], -1).detach().cpu().tolist(),
                    'candidate_ids': candidates.ids.detach().cpu().tolist(),
                    'slots': edges.slots.detach().cpu().tolist(),
                    'entropy': float(edges.entropy.detach()),
                    'message_norm': float(packets.detach().norm(dim=-1).mean()),
                    'state_change': float((state[next_active] - old[next_active]).detach().norm()),
                    'unique_touched': len(set(visited)),
                    'repeat_fraction': 1 - len(set(visited)) / max(1, len(visited))})
            active = next_active
        local_output = self.readout(state.index_select(0, active))
        output = ((local_output * instance.x[active, 4:6]).sum(0) / active.numel() ** 0.5
                  if instance.task == 'B' else local_output.mean(0))
        ledger.linear(self.readout, active.numel())
        ledger.add('node_gathers', active.numel())
        ledger.add('state_reads', active.numel() * self.dim)
        ledger.add('activation_ops_estimated', active.numel() * self.dim)
        cost = ledger.record(self, instance)
        if frozen is not None:
            cost['index_bytes'] += frozen.numel() * frozen.element_size()
        return output, cost, records, {'entropy': torch.stack(entropies).mean(),
                                      'state_norm': state.index_select(0, active).square().mean()}


class PoolingShortcut(nn.Module):
    """Strong global mean-pooling diagnostic, explicitly outside sparse budget."""
    def __init__(self, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURES, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.readout = nn.Sequential(nn.Linear(2 * dim, dim), nn.Tanh(), nn.Linear(dim, 2))

    def forward(self, instance):
        values = self.encoder(instance.x)
        return self.readout(torch.cat([values.mean(0), values[instance.roots].mean(0)]))
