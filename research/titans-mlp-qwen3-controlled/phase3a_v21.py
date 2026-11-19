"""Frozen-Qwen role-span addressing and value projection smoke."""
import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_qwen import REVISION, sha, verify_snapshot
from phase3a_v21_data import dataset, view as address_view


def view(row, kind):
    entity, relation, value = row
    if kind in ('bare', 'contextual'):
        prefix = '' if kind == 'bare' else 'Answer: '
        return prefix + value, [(len(prefix), len(prefix) + len(value))]
    held = kind.startswith('held_')
    role, _, variant = kind.removeprefix('held_').partition(':')
    return address_view(row, role, int(variant or 0), held)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = self.projection(2048)
        self.key = self.projection(2048)
        self.value = self.projection(1024)

    @staticmethod
    def projection(width):
        return nn.Sequential(nn.Linear(width, 256, bias=False), nn.LayerNorm(256))

    def encode(self, role, features):
        return F.normalize(getattr(self, role)(features), dim=-1)


def contrastive(q, k):
    scores = q @ k.T / .07
    target = torch.arange(len(q), device=q.device)
    return (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target)) / 2


def extract(model, tokenizer, rows, kind):
    out = []
    captured = []
    hook = model.model.layers[22].register_forward_hook(lambda _m, _a, y: captured.append(y))
    try:
        for i in range(0, len(rows), 32):
            batch = [view(row, kind) for row in rows[i:i+32]]
            inputs = tokenizer([x[0] for x in batch], padding=True, return_tensors='pt',
                               return_offsets_mapping=True)
            offsets = inputs.pop('offset_mapping')
            captured.clear()
            with torch.no_grad():
                model.model(**inputs.to(model.device), use_cache=False)
            for index, (_, spans) in enumerate(batch):
                vectors = []
                previous = -1
                for j, (start, end) in enumerate(spans):
                    left, right = offsets[index].unbind(1)
                    indices = torch.where((left < end) & (right > start))[0]
                    if not len(indices) or int(indices[0]) <= previous:
                        raise ValueError('Invalid span ordering')
                    previous = int(indices[-1])
                    features = captured[0][index, indices]
                    vectors.append(features[-1] if j == 1 else features.mean(0))
                out.append(torch.cat(vectors).cpu())
    finally:
        hook.remove()
    result = torch.stack(out)
    if not result.isfinite().all():
        raise ValueError('Nonfinite representation')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    hashes = verify_snapshot(args.snapshot)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.convert_ids_to_tokens(151643)
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True).eval().requires_grad_(False).to(args.device)
    splits = dataset(42, (8, 8, 0))
    del splits["test"]
    features = {split: {kind: extract(model, tokenizer, rows, kind) for kind in
                ('write', 'query', 'held_write', 'held_query', 'write:1', 'query:1', 'bare', 'contextual')}
                for split, rows in splits.items()}
    adapter = Adapter()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.001, weight_decay=.0001)
    train = features['train']
    losses = []
    rows = splits['train']
    weights = torch.tensor([[0. if i == j else .5 if e == other_e or r == other_r else 0.
                             for j, (other_e, other_r, _) in enumerate(rows)]
                            for i, (e, r, _) in enumerate(rows)])
    rng = random.Random(42)
    for _ in range(8):
        optimizer.zero_grad()
        q_view, k_view = (rng.choice(('', ':1')) for _ in range(2))
        q = adapter.encode('query', train['query' + q_view])
        k = adapter.encode('key', train['write' + k_view])
        logits = q @ k.T / .07 + weights
        labels = torch.arange(len(q))
        qk = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        value = contrastive(adapter.encode('value', train['bare']), adapter.encode('value', train['contextual']))
        loss = qk + value
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(dict(qk=qk.item(), value=value.item(), total=loss.item()))
    target = torch.arange(32)
    validation = features['validation']
    def recall(scores):
        return float((scores.argmax(1) == target).float().mean())
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output / 'dataset.json'
    data.write_text(json.dumps(splits, indent=2))
    checkpoint = args.output / 'adapter.safetensors'
    save_file(adapter.state_dict(), str(checkpoint))
    restored = Adapter()
    restored.load_state_dict(load_file(str(checkpoint)), strict=True)
    with torch.no_grad():
        for role, kind in (('query', 'query'), ('key','write'), ('value','bare')):
            torch.testing.assert_close(adapter.encode(role, validation[kind]),
                                       restored.encode(role, validation[kind]), rtol=0, atol=0)
        results = {
            'seen': recall(adapter.encode('query', validation['query']) @ adapter.encode('key', validation['write']).T),
            'held_query': recall(adapter.encode('query', validation['held_query']) @ adapter.encode('key', validation['write']).T),
            'held_both': recall(adapter.encode('query', validation['held_query']) @ adapter.encode('key', validation['held_write']).T),
            'value': recall(adapter.encode('value', validation['bare']) @ adapter.encode('value', validation['contextual']).T),
        }
    (args.output / 'results.json').write_text(json.dumps(dict(status='SMOKE_ONLY', seed=42,
        revision=REVISION, hashes=hashes, tokenizer_sha256=sha(args.snapshot/'tokenizer.json'),
        dataset_sha256=sha(data), checkpoint_sha256=sha(checkpoint), test="SEALED",
        backbone_frozen=not any(p.requires_grad for p in model.parameters()),
        losses=losses, validation_recall1=results), indent=2))


if __name__ == '__main__':
    main()
