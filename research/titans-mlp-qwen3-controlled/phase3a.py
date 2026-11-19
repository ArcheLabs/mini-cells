"""CPU semantic addressing diagnostic, with validation-only ridge selection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from torch.nn.functional import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_qwen import REVISION, sha, verify_snapshot

RELATIONS = ('works on', 'belongs to', 'is assigned to', 'operates', 'reports to',
             'uses', 'manages', 'is located in')
RIDGE = (.01, .1, 1., 10.)
QUESTIONS = (
    'Which project does {} work on?', 'Which group does {} belong to?',
    'What is {} assigned to?', 'What does {} operate?', 'Who does {} report to?',
    'What does {} use?', 'What does {} manage?', 'Where is {} located?',
)



def facts(seed, split, count):
    rng = torch.Generator().manual_seed(seed * 10 + split)
    ids = torch.randperm(100000, generator=rng)[:count].tolist()
    return [(f'E{split}{i:05}', RELATIONS[j % 8], f'V{split}{ids[-1-j]:05}')
            for j, i in enumerate(ids)]


def metrics(scores):
    rank = (scores >= scores.diag()[:, None]).sum(1)
    correct = rank.eq(1).float()
    p = float(correct.mean())
    n = len(rank)
    z = 1.96
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * (p*(1-p)/n + z*z/(4*n*n))**.5 / (1+z*z/n)
    negative = scores.masked_fill(torch.eye(n, dtype=torch.bool), -torch.inf).max(1).values
    return dict(recall1=p, recall5=float(rank.le(5).float().mean()), mrr=float(rank.float().reciprocal().mean()),
                positive=float(scores.diag().mean()), hardest_negative=float(negative.mean()),
                margin=float((scores.diag()-negative).mean()), ci95=[max(0,center-half),min(1,center+half)],
                ranks=rank.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    start = time.monotonic()
    hashes = verify_snapshot(args.snapshot)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.convert_ids_to_tokens(151643)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True).eval().requires_grad_(False).to(args.device)
    loading_seconds = time.monotonic()-start
    args.output.mkdir(parents=True, exist_ok=False)
    captured = []
    model.model.layers[22].register_forward_hook(lambda _m,_a,out: captured.append(out))

    @torch.inference_mode()
    def encode(texts):
        last, mean = [], []
        for i in range(0, len(texts), 32):
            batch = tokenizer(texts[i:i+32], padding=True, return_tensors='pt')
            captured.clear()
            model.model(**batch.to(args.device), use_cache=False)
            hidden, mask = captured[0], batch['attention_mask']
            assert hidden.shape[-1] == 1024 and hidden.isfinite().all()
            length = mask.sum(1)
            last.append(hidden[torch.arange(len(mask), device=hidden.device), length-1].cpu())
            mean.append((hidden * mask[...,None]).sum(1).div(length[:,None]).cpu())
        return {'last': normalize(torch.cat(last)), 'mean': normalize(torch.cat(mean))}

    counts = (32,32,32) if args.smoke else (1024,256,512)
    seeds = (42,) if args.smoke else (42,43,44,45,46)
    rows, datasets = [], {}
    for seed in seeds:
        print(f'Seed {seed} starting', flush=True)
        splits = [facts(seed, i, size) for i,size in enumerate(counts)]
        identifiers = [tuple(tokenizer.encode(x, add_special_tokens=False))
                       for split in splits for entity,_,value in split for x in (entity,value)]
        assert len(set(identifiers)) == len(identifiers)
        encoded = []
        records = []
        for idx, split in enumerate(splits):
            texts = {'write': [], 'write_alt': [], 'query': [], 'query_alt': [],
                     'held_query': [], 'held_write': [], 'negative_entity': [],
                     'negative_relation': [], 'bare': [], 'contextual': []}
            for j,(e,r,v) in enumerate(split):
                texts['write'].append(f'{e} {r} {v}.')
                texts['write_alt'].append(f'The record states that {e} {r} {v}.')
                texts['query'].append(QUESTIONS[j % 8].format(e))
                texts['query_alt'].append('According to the record, ' + QUESTIONS[j % 8].format(e).replace(QUESTIONS[j % 8][0], QUESTIONS[j % 8][0].lower(), 1))
                texts['held_query'].append(f'Identify the target of the {r} relation for {e}.')
                texts['held_write'].append(f'Relation {r}: subject {e}, object {v}.')
                texts['negative_entity'].append(QUESTIONS[j % 8].format('Z'+e))
                texts['negative_relation'].append(QUESTIONS[(j+1)%8].format(e))
                texts['bare'].append(v)
                texts['contextual'].append(f'Answer: {v}')
            if idx != 2:
                texts.pop('held_query')
                texts.pop('held_write')
            records.append(texts)
            needed = (('write', 'write_alt', 'query', 'query_alt') if idx == 0
                      else ('write', 'query') if idx == 1 else tuple(texts))
            encoded.append({key:encode(texts[key]) for key in needed})
        datasets[str(seed)] = records
        for pool in ('last','mean'):
            train, val, test = [{k:v[pool] for k,v in split.items()} for split in encoded]
            x = torch.cat((train['query'], train['query_alt']))
            y = torch.cat((train['write'], train['write_alt']))
            eye = torch.eye(x.shape[1])
            models = {reg: torch.linalg.solve(x.T@x+reg*eye,x.T@y) for reg in RIDGE}

            def grouped(q,w):
                return [metrics(a@b.T) for a,b in zip(q.split(32),w.split(32),strict=True)]

            def score(items):
                return sum(m['recall1'] for m in items)/len(items)

            reg = max(RIDGE,key=lambda r: score(grouped(normalize(val['query']@models[r]),val['write'])))
            for label,q,w in [('seen',test['query'],test['write']),
                              ('held',test['held_query'],test['write']),
                              ('held_both',test['held_query'],test['held_write'])]:
                rng = torch.Generator().manual_seed(seed)
                random = torch.rand(len(q)//32,32,32,generator=rng)
                methods = dict(raw=grouped(q,w),probe=grouped(normalize(q@models[reg]),w),
                               random=[metrics(s) for s in random],
                               oracle=[metrics(torch.eye(32)) for _ in random])
                rows.append(dict(seed=seed,pooling=pool,case=label,ridge=reg,methods=methods,
                  value_cosine=float((test['bare']*test['contextual']).sum(1).mean()),
                  negative_entity_cosine=float((test['negative_entity']*w).sum(1).mean()),
                  negative_relation_cosine=float((test['negative_relation']*w).sum(1).mean())))
        print(f'Seed {seed} finished',flush=True)
        (args.output/'partial.json').write_text(json.dumps(rows))
    data = args.output/'dataset.json'
    data.write_text(json.dumps(datasets,ensure_ascii=False))
    (args.output/'results.json').write_text(json.dumps(dict(rows=rows,counts=counts,revision=REVISION,
        hashes=hashes,tokenizer_sha256=sha(args.snapshot/'tokenizer.json'),dataset_sha256=sha(data),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        loading_seconds=loading_seconds, elapsed_seconds=time.monotonic()-start, training_updates=0,
        device=args.device,dtype='float32',hidden_layer=22),indent=2))


if __name__ == '__main__':
    main()
