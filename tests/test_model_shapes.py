import pytest
import torch
from minicells.model import EchoModel

@pytest.mark.parametrize("batch",[1,3])
def test_shapes(batch):
    model=EchoModel(vocab_size=44,num_cells=16,hidden_dim=8,embedding_dim=4,radius=1,iterations=2,mlp_width=16)
    logits,state=model(torch.randint(0,44,(batch,16)),return_state=True)
    assert logits.shape == (batch,16,44); assert state.shape == (batch,16,8); assert torch.isfinite(logits).all()
