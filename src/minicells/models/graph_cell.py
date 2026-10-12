"""004 adapter reusing the 003 backbone, Cell bank and all reference arms."""
from minicells.models.routing_research import LanguageModel as ReferenceModel
from minicells.routing.graph_router import GraphRouter


class LanguageModel(ReferenceModel):
    def __init__(self, config, kind):
        super().__init__(config, 'hard-cell' if kind == 'graph-cell' else kind)
        self.kind = kind
        if kind == 'graph-cell':
            for block in self.blocks:
                block.ffn = GraphRouter(config, cells=block.ffn.cells)
                # Reuse the identical initialized hard baseline Cell weights; prior starts at zero.
        for router in self.routing_modules():
            if hasattr(router, 'mode'):
                router.graph_evidence_enabled = True

    def compute_penalty(self):
        if self.kind == 'graph-cell':
            routers = self.routing_modules()
            return sum(router.expected_path for router in routers) / len(routers)
        return super().compute_penalty()

    def forward(self, tokens):
        for router in self.routing_modules():
            if getattr(router, 'collect_stats', False):
                router.routing_token_ids = tokens
        return super().forward(tokens)
