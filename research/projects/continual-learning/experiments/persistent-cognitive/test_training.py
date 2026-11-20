"""Checkpoint cursor and optimizer equivalence at the training API boundary."""
from types import SimpleNamespace

import torch

from training import fit


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)
        self.memories = {}

    def forward(self, model, cell, *, input_ids, diagnostics=None, **kwargs):
        return SimpleNamespace(logits=self.projection(torch.nn.functional.one_hot(input_ids, 4).float())), cell


def test_resume_matches_continuous(tmp_path):
    ids = [torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 3]])]
    fp = {k: k for k in ('backbone', 'tokenizer', 'architecture', 'meta', 'layer_map')}
    frozen = torch.nn.Linear(4, 4).requires_grad_(False)
    torch.manual_seed(42)
    model = Fixture()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path/'checkpoint.pt'
    fit(frozen, model, opt, ids, ids, fp, path, dataset_sha256='fixture', steps=1)
    restored = Fixture()
    result = fit(frozen, restored, torch.optim.AdamW(restored.parameters()), ids, ids, fp,
                 path, dataset_sha256='fixture', steps=2, resume=True)
    model.load_state_dict(initial)
    fit(frozen, model, torch.optim.AdamW(model.parameters()), ids, ids, fp, tmp_path/'full.pt',
        dataset_sha256='fixture', steps=2)
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), atol=0, rtol=0)
    assert len(result['history']) == 2 and result['validation']['targets'] == 4
