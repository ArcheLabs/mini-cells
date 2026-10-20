from pathlib import Path
import runpy
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b"
runpy.run_path(str(BASE / "analyze.py"), run_name="cf_test_bootstrap")
sys.path.insert(0, str(BASE / "implementation"))
sys.path.insert(0, str(BASE / "counterfactual-stop-route/implementation"))


@pytest.fixture
def case():
    from minicells_runtime.smoke import configuration
    from clm001_r3.model import Model

    torch.manual_seed(19)
    c = configuration()
    c["model"]["budget"] = 5
    model = Model(c, "QC-R3").eval().requires_grad_(False)
    tokens = torch.tensor([[2, 3, 4, 5]])
    return model, tokens, torch.tensor([[3, 4, 5, 6]])


class DepthQ(torch.nn.Module):
    def __init__(self, registry, actions):
        super().__init__()
        self.registry = registry
        self.choices = actions

    def forward(self, h, prev, depth, legal, version, **kwargs):
        q = torch.ones(len(h), self.registry.actions, device=h.device)
        q[:, -1] = 0
        q[:, self.choices[min(depth, len(self.choices) - 1)]] = -1
        q = q.masked_fill(~legal, float("inf"))
        return q, torch.zeros(len(h), device=h.device)


@pytest.fixture
def routed(case):
    model, x, y = case
    model.controller = DepthQ(
        model.registry, [0, 0, model.registry.ctx, model.registry.ctx, model.registry.stop]
    )
    return model, x, y
