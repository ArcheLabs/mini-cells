"""Independently reload the frozen adapter and replay its validation fixture."""
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys

import torch

from train6c import CORPUS, MODEL_ID, REVISION, measure, windows
from stage2 import NeuralMemory, Qwen35Adapter
from hybrid6c import build_memories


@torch.no_grad()
def verify(checkpoint, out):
    from transformers import AutoTokenizer

    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    identity = saved['identity']
    if len(saved['history']) != 64 or identity['segment'] != 512 or identity['corpus_revision'] != CORPUS:
        raise ValueError('Unexpected checkpoint identity')
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    adapter.model.requires_grad_(False).set_attn_implementation('eager')
    memories = build_memories(adapter.model, NeuralMemory)
    if identity['model'] != adapter.identity or identity['layers'] != list(memories):
        raise ValueError('Model or layers differ')
    torch.nn.ModuleDict({str(i): m for i, m in memories.items()}).load_state_dict(saved['adapter'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    validation = list(islice(windows('validation', tokenizer), 4))
    if len(validation) != 4 or hashlib.sha256(json.dumps(validation).encode()).hexdigest() != identity['validation_sha256']:
        raise ValueError('Validation window hash differs')
    scores = []
    for ids in validation:
        scores.append(measure(adapter, memories, ids, train=False))
        print(f'Validation window {len(scores)}/4: {scores[-1]:.6f}', flush=True)
    nll = sum(scores) / 4
    if not abs(nll - 2.6195888435468078) <= .01:
        raise ValueError(f'Frozen validation mismatch: {nll}')
    (out / 'validation.json').write_text(json.dumps(dict(nll=nll, windows=scores, updates=0)))


if __name__ == '__main__':
    verify(*map(Path, sys.argv[1:]))
