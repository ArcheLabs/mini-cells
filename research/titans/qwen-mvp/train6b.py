"""Train all six retrofit layers on WikiText-103 train text, with two-call unrolls."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn.functional import cross_entropy

from stage2 import HeadMemory, NeuralMemory, Qwen35Adapter
from minicells_runtime.qwen35 import MODEL_ID, REVISION

CORPUS = 'b08601e04326c79dfdd32d625aee71d232d685c3'
UPDATES = 1024
PREFIX = 64


def main(output):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Tesla T4 required')
    torch.manual_seed(42)
    adapter = Qwen35Adapter.load(dtype=torch.float32)
    model = adapter.model.requires_grad_(False)
    model.set_attn_implementation('eager')
    indices = [i for i, kind in enumerate(model.config.layer_types) if kind == 'full_attention']
    if len(indices) != 6:
        raise ValueError('Expected six full attention layers')
    memories = torch.nn.ModuleDict({
        str(i): HeadMemory(model.model.layers[i].self_attn, NeuralMemory) for i in indices
    }).cuda()
    for memory in memories.values():
        for shard in memory.shards:
            for projection in (shard.key, shard.value, shard.query):
                projection.requires_grad_(False)
    params = [p for p in memories.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    rows = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1',
                        revision=CORPUS, split='train', streaming=True)
    updates = 0
    with output.with_name('training.jsonl').open('w', buffering=1) as log:
        for row in rows:
            ids = tokenizer.encode(row['text'], add_special_tokens=False)
            if len(ids) < PREFIX + 2:
                continue
            tokens = torch.tensor([ids[:PREFIX + 2]], device='cuda')
            cells = {i: [m.create(mode='meta_train') for m in memories[str(i)].shards]
                     for i in indices}
            optimizer.zero_grad(set_to_none=True)
            with ExitStack() as stack:
                for i in indices:
                    stack.enter_context(memories[str(i)].attach(model, i, cells[i], mode='meta_train'))
                _, state = adapter.encode(tokens[:, :PREFIX], absolute_positions=True)
            with ExitStack() as stack:
                for i in indices:
                    stack.enter_context(memories[str(i)].attach(model, i, cells[i],
                                                                write=False, mode='meta_train'))
                hidden = model.model(input_ids=tokens[:, PREFIX:PREFIX+1], past_key_values=state.cache,
                                     position_ids=torch.tensor([[PREFIX]], device='cuda'),
                                     use_cache=True).last_hidden_state
            loss = cross_entropy(adapter.logits(hidden[:, -1]), tokens[:, -1])
            loss.backward()
            for name, p in memories.named_parameters():
                if p.requires_grad and (p.grad is None or not p.grad.isfinite().all()):
                    raise FloatingPointError(f'Invalid gradient at update {updates + 1}: {name}')
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            log.write(json.dumps(dict(update=updates, nll=float(loss.detach()),
                                      input_sha256=hashlib.sha256(bytes(str(ids[:PREFIX+2]), 'utf8')).hexdigest())) + '\n')
            if updates % 32 == 0:
                print(f'Completed {updates}/{UPDATES} updates', flush=True)
            if updates == UPDATES:
                break
    if updates != UPDATES:
        raise RuntimeError('Training stream exhausted')
    path = output.with_name('adapter.pt')
    torch.save(dict(adapter=memories.state_dict(), optimizer=optimizer.state_dict(),
                    identity=adapter.identity, layers=indices, updates=updates), path)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    output.write_text(json.dumps(dict(status='STAGE6B_TRAIN_COMPLETE', identity=adapter.identity,
        layers=indices, updates=updates, supervised_targets=updates, corpus_revision=CORPUS,
        seed=42, prefix=PREFIX, checkpoint_sha256=sha, test='SEALED'), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
