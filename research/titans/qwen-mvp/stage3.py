"""Two-chunk outer LM gradient and optimizer smoke."""
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from stage2 import HeadMemory, NeuralMemory, Qwen35Adapter


def main(output):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    model = adapter.model.requires_grad_(False)
    model.set_attn_implementation('eager')
    index = model.config.layer_types.index('full_attention')
    retrofit = HeadMemory(model.model.layers[index].self_attn, NeuralMemory).cuda()
    for shard in retrofit.shards:
        # E2 consumes native head features directly.
        for projection in (shard.key, shard.value, shard.query):
            projection.requires_grad_(False)
    params = dict((name, p) for name, p in retrofit.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(params.values(), lr=.001)
    versions = [p._version for p in model.parameters()]
    ids = torch.tensor([[101, 307, 509, 701, 907, 1103]], device='cuda')
    rows = []
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        cells = [m.create(mode='meta_train') for m in retrofit.shards]
        with retrofit.attach(model, index, cells, mode='meta_train'):
            _, state = adapter.encode(ids[:, :4])
        # Qwen35Adapter.encode is inference-only. Invoke the native text model
        # directly for the differentiable read, on the disposable session cache.
        with retrofit.attach(model, index, cells, write=False, mode='meta_train'):
            hidden = model.model(input_ids=ids[:, 4:5], past_key_values=state.cache,
                                 use_cache=True).last_hidden_state
        loss = F.cross_entropy(adapter.logits(hidden[:, -1]).float(), ids[:, 5])
        loss.backward()
        missing = [name for name, p in params.items() if p.grad is None]
        nonfinite = [name for name, p in params.items()
                     if p.grad is not None and not p.grad.isfinite().all()]
        if missing or nonfinite:
            raise RuntimeError(f'Update {update + 1}: missing={missing}, nonfinite={nonfinite}')
        gradients = {name: float(p.grad.double().norm()) for name, p in params.items()}
        for prefix in ('gate', 'shards.0.initial', 'shards.1.initial',
                       'shards.0.controls', 'shards.1.controls'):
            if not any(v > 0 for name, v in gradients.items() if name.startswith(prefix)):
                raise RuntimeError(f'No gradient through {prefix}')
        before = retrofit.gate.detach().clone()
        optimizer.step()
        if torch.equal(before, retrofit.gate) or versions != [p._version for p in model.parameters()]:
            raise RuntimeError('Unexpected optimizer update')
        rows.append(dict(update=update + 1, nll=float(loss.detach()), gradients=gradients))
    output.write_text(json.dumps(dict(status='E3_GRADIENT_SMOKE_PASS', identity=adapter.identity,
        layer=index, gpu=torch.cuda.get_device_name(), updates=rows, supervised_targets=2,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
