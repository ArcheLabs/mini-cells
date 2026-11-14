"""Bounded development training with the validated Stage 6C gradient policy."""
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn.functional import cross_entropy

from hybrid6c import HybridSession, build_memories
from stage2 import NeuralMemory, Qwen35Adapter
from train6b import CORPUS, MODEL_ID, REVISION

UPDATES = 256
LENGTH = 1537


def windows(split, tokenizer):
    from datasets import load_dataset

    buffer = []
    rows = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1',
                        revision=CORPUS, split=split, streaming=True)
    for row in rows:
        if row['text'].strip():
            buffer.extend(tokenizer.encode(row['text'], add_special_tokens=False))
            buffer.append(tokenizer.eos_token_id)
        while len(buffer) >= LENGTH:
            yield buffer[:LENGTH]
            del buffer[:LENGTH]


def measure(adapter, memories, sequence, *, train):
    ids = torch.tensor([sequence], device='cuda')
    session = HybridSession(adapter, memories,
                            mode='meta_train' if train else 'online_adapt',
                            detach_backbone_cache=train)
    session.forward(ids[:, :512])
    total = 0.
    for start in range(512, 1536, 16):
        hidden = session.forward(ids[:, start:start + 16])
        logits = adapter.logits(hidden).flatten(0, 1)
        term = cross_entropy(logits, ids[:, start + 1:start + 17].flatten(), reduction='sum')
        if not torch.isfinite(term):
            raise FloatingPointError('Nonfinite token loss')
        if train:
            (term / 1024).backward(retain_graph=start + 16 < 1536)
        total += float(term.detach())
    return total / 1024


def main(output, resume=None):
    deadline = time.monotonic() + 8800
    # The original T4 run reached 67 logged updates in 9,600 seconds,
    # including validation and checkpoint writes.
    if resume:
        saved = torch.load(resume, weights_only=True, map_location='cpu')
        completed = len(saved['history'])
        stopped = saved['stale'] >= 3
        del saved
    else:
        completed, stopped = 0, False
    remaining = 0 if stopped else max(0, UPDATES - completed)
    projected = remaining * (9600 / 67) * 1.15
    if projected > 8800:
        output.write_text(json.dumps(dict(
            status='STAGE6C_BLOCKED_BUDGET', completed_updates=completed,
            remaining_updates=remaining, projected_seconds=round(projected),
            training_budget_seconds=8800, optimizer_updates_this_session=0,
            estimate_basis='67 logged updates in 9600 seconds, plus 15% margin',
        ), indent=2))
        return
    from itertools import islice
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    adapter.model.requires_grad_(False).set_attn_implementation('eager')
    memories = torch.nn.ModuleDict({str(i): m for i, m in
                                   build_memories(adapter.model, NeuralMemory).items()})
    layers = [int(i) for i in memories]
    native = {int(i): m for i, m in memories.items()}
    for m in memories.values():
        for shard in m.shards:
            for projection in (shard.key, shard.query, shard.value):
                projection.requires_grad_(False)
    params = [p for p in memories.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    val = list(islice(windows('validation', tokenizer), 4))
    if len(val) != 4:
        raise ValueError('Incomplete fixed validation set')
    identity = dict(model=adapter.identity, corpus_revision=CORPUS, seed=42, segment=512,
                    tile=16, layers=layers, updates=UPDATES, cache_gradient='detach_between_calls',
                    validation_sha256=hashlib.sha256(json.dumps(val).encode()).hexdigest(),
                    implementation_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                           for name in ('train6c.py', 'hybrid6c.py', 'segment6c.py', 'titans_attention.py')})
    history = []
    best = float('inf')
    stale = 0
    if resume:
        saved = torch.load(resume, weights_only=True, map_location='cpu')
        previous = dict(identity, implementation_sha256=dict(identity['implementation_sha256']))
        previous['implementation_sha256']['train6c.py'] = '7845c1224a145efa137a8dc26da80ad76245acdd5fd46397bfd31633c6d2cbde'
        if saved['identity'] not in (identity, previous):
            raise ValueError('Resume identity mismatch')
        memories.load_state_dict(saved['adapter'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, stale = saved['history'], saved['best'], saved['stale']
        torch.set_rng_state(saved['rng'])
    checkpoint = output.with_name('adapter.pt')
    if resume and checkpoint.resolve() != resume.resolve():
        import shutil
        shutil.copy2(resume, checkpoint)
    iterator = islice(windows('train', tokenizer), len(history), UPDATES)
    for sequence in iterator:
        if stale >= 3 or time.monotonic() >= deadline:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = measure(adapter, native, sequence, train=True)
        for name, p in memories.named_parameters():
            if p.requires_grad and (p.grad is None or not p.grad.isfinite().all()):
                raise FloatingPointError(f'Invalid gradient: {name}')
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        step = len(history) + 1
        row = dict(update=step, nll=loss)
        improved = False
        if step % 32 == 0:
            with torch.no_grad():
                row['validation_nll'] = sum(measure(adapter, native, x, train=False) for x in val) / len(val)
            improved = row['validation_nll'] < best
            best = min(best, row['validation_nll'])
            stale = 0 if improved else stale + 1
        history.append(row)
        payload = dict(adapter=dict(memories.state_dict()), optimizer=optimizer.state_dict(),
                       identity=identity, history=history, best=best, stale=stale, rng=torch.get_rng_state())
        temp = checkpoint.with_suffix('.tmp')
        torch.save(payload, temp)
        temp.replace(checkpoint)
        if improved:
            torch.save(payload, output.with_name('best-adapter.pt'))
        with output.with_name('training.jsonl').open('a') as log:
            log.write(json.dumps(row) + '\n')
        print(row, flush=True)
    complete = len(history) == UPDATES or stale >= 3
    if not complete and time.monotonic() < deadline:
        raise RuntimeError('Training stream exhausted')
    with checkpoint.open('rb') as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    output.write_text(json.dumps(dict(status='STAGE6C_TRAIN_COMPLETE' if complete else 'STAGE6C_TRAIN_PAUSED',
        checkpoint_sha256=checkpoint_sha256, identity=identity,
        updates=len(history), supervised_targets=1024 * len(history), best_validation_nll=best,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), test='SEALED'), indent=2))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    main(args.output, args.resume)
