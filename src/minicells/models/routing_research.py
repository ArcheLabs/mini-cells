"""Shared research adapters; historical 001/002 entry points remain unchanged."""
from experiments.cell_internal_routing_002.models.cell_router import LanguageModel as PreviousModel
from minicells.routing.hard_transition import HardCellTransition


class LanguageModel(PreviousModel):
    def __init__(self, config, kind):
        super().__init__(config, 'cell-internal' if kind in {'soft-cell', 'hard-cell'} else kind)
        self.kind = kind
        if kind in {'soft-cell', 'hard-cell'}:
            for block in self.blocks:
                block.ffn = HardCellTransition(config)
                block.ffn.apply(self._initialize)
        if kind == 'global-router':
            for block in self.blocks:
                router = block.ffn
                def observe(module, inputs, output, owner=router):
                    if owner.training:
                        probabilities = output.detach().float().softmax(-1)
                        owner.last_entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()
                router.router.register_forward_hook(observe)

    def compute_penalty(self):
        routers = self.routing_modules() if self.kind in {'soft-cell', 'hard-cell'} else []
        return (sum(router.expected_path for router in routers) / len(routers)
                if routers else self.embedding.weight.new_zeros(()))

    def set_stage(self, mode, temperature):
        for router in self.routing_modules():
            if isinstance(router, HardCellTransition):
                router.mode, router.temperature = mode, temperature

    def parameter_metrics(self):
        return {'params': sum(parameter.numel() for parameter in self.parameters())}
