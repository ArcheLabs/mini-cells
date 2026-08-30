import torch
from minicells.model import EchoModel

def test_eval_forward_is_identical():
    torch.manual_seed(1); model=EchoModel(vocab_size=44,num_cells=8); model.eval(); inputs=torch.randint(0,44,(2,8))
    assert torch.equal(model(inputs),model(inputs))
