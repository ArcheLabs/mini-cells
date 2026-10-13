"""Executed tensor-shape counts. These are estimates, never hardware FLOPs."""
from dataclasses import dataclass, field


@dataclass
class Ledger:
    state_element_size: int = 4
    counts: dict = field(default_factory=lambda: {
        'candidate_reads': 0, 'node_gathers': 0, 'messages': 0, 'cell_updates': 0,
        'route_score_ops': 0, 'index_reads': 0, 'state_reads': 0, 'state_writes': 0,
        'schedule_items': 0, 'linear_ops': 0, 'activation_ops_estimated': 0,
        'route_linear_ops': 0, 'state_copy_bytes': 0,
        'global_pair_scores': 0, 'index_build_items': 0,
    })

    def add(self, name, value):
        self.counts[name] += int(value)

    def linear(self, module, rows):
        self.add('linear_ops', 2 * rows * module.in_features * module.out_features)
        self.add('activation_ops_estimated', rows * module.out_features)

    def record(self, model, instance):
        n, d = instance.x.shape[0], model.dim
        return {**self.counts,
                'parameters_total': sum(p.numel() for p in model.parameters()),
                'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'node_state_bytes': n * d * self.state_element_size,
                'immutable_encoding_bytes': n * d * self.state_element_size,
                'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                'input_bytes': instance.x.numel() * instance.x.element_size(),
                'index_bytes': instance.candidates.numel() * instance.candidates.element_size(),
                'index_build_items': instance.candidates.numel(),
                'total_ops_estimated': self.counts['linear_ops'] + self.counts['route_score_ops']
                    - self.counts['route_linear_ops']
                    + self.counts['activation_ops_estimated'],
                'scope': 'forward linear 2 FLOPs/MAC, explicit route arithmetic and approximate activations; '
                         'NOT total hardware FLOPs; sort/gather/scatter costs measured in latency',
                'peak_ram': 'UNAVAILABLE', 'peak_vram': 'UNAVAILABLE'}
