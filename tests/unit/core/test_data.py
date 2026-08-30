import torch
from minicells.data import CopyDataGenerator
from minicells.vocab import CharVocab

def test_copy_lengths_and_determinism():
    kwargs=dict(vocab=CharVocab(),min_length=1,max_length=12,num_cells=16)
    a=CopyDataGenerator(seed=7,**kwargs).batch(20); b=CopyDataGenerator(seed=7,**kwargs).batch(20); c=CopyDataGenerator(seed=8,**kwargs).batch(20)
    assert torch.equal(a.input_ids,a.target_ids); assert a.lengths.min() >= 1; assert a.lengths.max() <= 12
    assert torch.equal(a.input_ids,b.input_ids); assert not torch.equal(a.input_ids,c.input_ids)
