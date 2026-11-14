"""Frozen Stage 6C evaluation using the existing segmented inference path."""
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys

import torch
from torch.nn.functional import cross_entropy

from train6c import CORPUS, MODEL_ID, REVISION, windows, measure
from stage2 import Qwen35Adapter, NeuralMemory
from hybrid6c import HybridSession, build_memories


@torch.no_grad()
def main(output, checkpoint):
    from transformers import AutoTokenizer

    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    identity = saved['identity']
    if len(saved['history']) != 64 or identity['segment'] != 512 or identity['corpus_revision'] != CORPUS:
        raise ValueError('Expected the original update 64 checkpoint')
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    adapter.model.requires_grad_(False).set_attn_implementation('eager')
    modules = build_memories(adapter.model, NeuralMemory)
    if adapter.identity != identity['model'] or list(modules) != identity['layers']:
        raise ValueError('Checkpoint model identity mismatch')
    torch.nn.ModuleDict({str(i): m for i, m in modules.items()}).load_state_dict(saved['adapter'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    validation = list(islice(windows('validation', tokenizer), 4))
    if len(validation) != 4 or hashlib.sha256(json.dumps(validation).encode()).hexdigest() != identity['validation_sha256']:
        raise ValueError('Validation windows differ')
    nll = sum(measure(adapter, modules, ids, train=False) for ids in validation) / 4
    reference = 2.6195888435468078
    report = dict(status='VALIDATION_MISMATCH', optimizer_updates=0,
                  validation_nll=nll, reference=reference, tolerance=1e-4)
    output.write_text(json.dumps(report, indent=2))
    if abs(nll - reference) > report['tolerance']:
        return
    samples = list(islice(windows('test', tokenizer), 32))
    if len(samples) != 32 or len({tuple(ids) for ids in samples}) != 32:
        raise ValueError('Incomplete or repeated test windows')
    for index, ids in enumerate(samples):
        tokens = torch.tensor([ids], device='cuda')
        donor = torch.tensor([samples[(index + 16) % 32][:512]], device='cuda')
        rows = {}
        for arm in ('E0', 'E1', 'E2', 'E3', 'E4'):
            state = None
            session = None if arm == 'E0' else HybridSession(adapter, modules)
            losses = []
            for start in (0, 512, 1024):
                chunk = tokens[:, start:start + 512]
                if session is None:
                    hidden, state = adapter.encode(chunk, state, absolute_positions=True)
                else:
                    hidden = session.forward(chunk, read=arm != 'E1')
                    if start == 0 and arm == 'E3':
                        other = HybridSession(adapter, modules)
                        other.forward(donor)
                        session.cells = other.cells
                    if start == 0 and arm == 'E4':
                        session.cells = HybridSession(adapter, modules).cells
                if start:
                    losses.append(torch.cat([
                        cross_entropy(adapter.logits(hidden[:, j:j+16]).flatten(0, 1),
                                      tokens[0, start+j+1:start+j+17], reduction='none')
                        for j in range(0, 512, 16)
                    ]).tolist())
            rows[arm] = losses
        with output.with_name('windows.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(index=index, donor=(index+16)%32, rows=rows,
                token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()), allow_nan=False)+'\n')
        print(f'Completed window {index + 1}/32', flush=True)
    report.update(status='LANGUAGE_DIAGNOSTIC_COMPLETE', windows=32, mvp='INCONCLUSIVE')
    output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
