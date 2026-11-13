"""Train a small episodic Titans adapter over a frozen Qwen backbone."""
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minicells08_smoke import setup, state_hash
from minicells_runtime.qwen35 import MODEL_ID, REVISION


KEYS = ('The code for{}.', 'Assigned code for{}.', 'Record the code of{}.',
        'Code entry for{}.', 'The identifier belonging to{}.')
QUERIES = ('What is the code for{}? Answer:', 'Give the code of{}:',
           'Which code belongs to{}? Code:', 'Retrieve the code for{}:',
           'What identifier was assigned to{}? Answer:')


def pools(tokenizer):
    special = set(tokenizer.all_special_ids)
    valid = []
    for token in range(len(tokenizer)):
        if token in special:
            continue
        word = tokenizer.decode([token])
        if (word.startswith(' ') and word[1:].isascii() and word[1:].isalpha()
                and 4 <= len(word) <= 12
                and tokenizer.encode(word, add_special_tokens=False) == [token]):
            valid.append(token)
    if len(valid) < 768:
        raise ValueError('Insufficient round-trip-safe vocabulary')
    random.Random(420).shuffle(valid)
    return {'train': (valid[:128], valid[128:256]),
            'validation': (valid[256:384], valid[384:512]),
            'test_reserved': (valid[512:640], valid[640:768])}


def episode(tokenizer, partition, split, index):
    if split not in ('train', 'validation'):
        raise ValueError('Test is sealed')
    rng = random.Random(f'learned-memory-v1:{split}:{index}')
    entity, value = (rng.sample(pool, 8) for pool in partition[split])
    prompts = [[], []]
    for token in entity:
        word = tokenizer.decode([token])
        template = rng.randrange(4) if split == 'train' else 4
        for group, text in zip(prompts, (KEYS[template], QUERIES[template])):
            ids = tokenizer.encode(text.format(word), add_special_tokens=False)
            if any(answer in ids for answer in value):
                raise ValueError('Answer leaked into prompt')
            group.append(ids)
    return prompts[0], [[token] for token in value], prompts[1], value


@torch.no_grad()
def encode(runtime, row, device):
    def segments(packets):
        return torch.cat([runtime.backbone.encode(torch.tensor([ids], device=device))[0][:, -1:]
                          for ids in packets], dim=1).float()
    return (*map(segments, row[:3]), torch.tensor(row[3], device=device))


def score(runtime, row, mode, control='correct'):
    key, value, query, answers = row
    memory = runtime.memory
    k, v, q = memory.project(key)[0], memory.project(value)[1], memory.project(query)[2]
    cell = memory.create(1, mode)
    if control != 'empty':
        targets = v.roll(1, 1) if control == 'permuted' else v
        for x, y in zip(k.split(1, 1), targets.split(1, 1)):
            cell, _ = memory.learn(x, y, cell, mode)
    read = memory.recall(q, cell, mode)
    scores = 10 * F.normalize(read[0], dim=-1) @ F.normalize(v[0], dim=-1).T
    fused = query if control == 'empty' else query + runtime.gate(query).sigmoid() * runtime.projection(read)
    logits = runtime.backbone.logits(fused.to(torch.float16))[0].float()
    return scores, logits, answers, cell


def train(runtime, optimizer, row):
    optimizer.zero_grad(set_to_none=True)
    scores, logits, answers, _ = score(runtime, row, 'meta_train')
    losses = (F.cross_entropy(scores, torch.arange(8, device=scores.device)),
              F.cross_entropy(logits, answers))
    sum(losses).backward()
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None or not param.grad.isfinite().all():
                raise FloatingPointError('Missing or nonfinite Adapter gradient')
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.)
    optimizer.step()
    return [float(loss.detach()) for loss in losses]


@torch.no_grad()
def evaluate(runtime, rows):
    results = []
    for row in rows:
        conditions = {}
        for control in ('correct', 'empty', 'permuted'):
            scores, logits, answers, cell = score(runtime, row, 'online_adapt', control)
            conditions[control] = dict(recall=int(scores.argmax(1).eq(torch.arange(8, device=scores.device)).sum()),
                                       exact=int(logits.argmax(1).eq(answers).sum()),
                                       answers=answers.tolist(), predictions=logits.argmax(1).tolist(),
                                       scores=scores.tolist(), state=state_hash(cell))
        results.append(conditions)
    return results


def main(output):
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    runtime, identity = setup()
    runtime.backbone.model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    partition = pools(tokenizer)
    output.mkdir(parents=True, exist_ok=True)
    def write(name, data):
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False))
    data = {split: [episode(tokenizer, partition, split, i) for i in range(count)]
            for split, count in [('train', 512), ('validation', 64)]}
    write('dataset.json', dict(identity=identity, pools=partition, episodes=data, test='SEALED'))
    modules = dict(memory=runtime.memory, gate=runtime.gate, projection=runtime.projection)
    params = [p for module in modules.values() for p in module.parameters()]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.0001)
    fixed = encode(runtime, data['train'][0], 'cuda')
    overfit = [train(runtime, optimizer, fixed) for _ in range(50)]
    write('overfit.json', overfit)
    if not all(after < before for before, after in zip(overfit[0], overfit[-1])):
        raise RuntimeError('Fixed episode loss did not improve')
    runtime, _ = setup()
    runtime.backbone.model.requires_grad_(False)
    modules = dict(memory=runtime.memory, gate=runtime.gate, projection=runtime.projection)
    optimizer = torch.optim.AdamW([p for m in modules.values() for p in m.parameters()],
                                  lr=.001, weight_decay=.0001)
    history = []
    for row in data['train'][:500]:
        history.append(train(runtime, optimizer, encode(runtime, row, 'cuda')))
        if len(history) % 25 == 0:
            write('training.json', history)
    results = evaluate(runtime, [encode(runtime, row, 'cuda') for row in data['validation']])
    save_file({f'{name}.{key}': tensor.cpu().contiguous() for name, module in modules.items()
               for key, tensor in module.state_dict().items()}, output / 'adapter.safetensors')
    write('pilot.json', dict(status='DEVELOPMENT_PILOT_COMPLETE', test='SEALED', results=results))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
