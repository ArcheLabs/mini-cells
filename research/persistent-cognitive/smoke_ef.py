"""Pretrained inference-mode and optimizer-resume smoke."""
import json
from pathlib import Path
import sys

import torch

from smoke_cd import setup
from cell import Cell, CellSlot
from session import Session, generate
from training import fit


def main(output):
    adapter, retrofit, identity = setup()
    model = adapter.model
    ids = torch.tensor([[101, 307, 509, 701]], device='cuda')
    with torch.inference_mode():
        slot = CellSlot(Cell.create(retrofit.memories, identity))
        session = Session.create(slot.current, adapter.cache_type(adapter.config))
        generated, session = generate(session, slot, model, retrofit, ids[:, :2],
                                      identity, max_new_tokens=3)
    if session.position != 4 or slot.current.revision != 3:
        raise RuntimeError('Unexpected generation cursor')
    versions = [p._version for p in model.parameters()]
    episodes = [ids, ids.flip(1)]
    checkpoint = output.with_name('checkpoint.pt')
    kwargs = dict(dataset_sha256='fixed-ef-token-fixture-v1')
    fit(model, retrofit, torch.optim.AdamW(retrofit.parameters()), episodes,
        [ids[:, 1:]], identity, checkpoint, steps=1, **kwargs)
    result = fit(model, retrofit, torch.optim.AdamW(retrofit.parameters()), episodes,
                 [ids[:, 1:]], identity, checkpoint, steps=2, resume=True, **kwargs)
    if versions != [p._version for p in model.parameters()]:
        raise RuntimeError('Frozen Qwen changed')
    output.write_text(json.dumps(dict(status='PCA_EF_SMOKE_PASS', identity=adapter.identity,
        generated=generated.tolist(), revision=slot.current.revision, updates=2,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), training=result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
