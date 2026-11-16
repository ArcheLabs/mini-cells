"""Check formal configuration wiring on a small train/validation-only run."""
import argparse
import copy
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_qwen import REVISION, sha, verify_snapshot
from phase3a import metrics
from phase3a_rejection import split_validation, unknown_queries
from phase3a_templates import SPLIT_FAMILIES
from phase3a_v21 import Adapter, dataset, extract


def load_config(path):
    cfg = json.loads(path.read_text())
    if (cfg['status'] != 'NOT_FROZEN' or cfg['evaluation']['test'] != 'SEALED'
            or cfg['model']['revision'] != REVISION):
        raise ValueError('Preflight requires the sealed draft')
    if cfg['dataset']['families'] != {k: list(v) for k,v in SPLIT_FAMILIES.items()}:
        raise ValueError('Template policy mismatch')
    if cfg['adapter'] != dict(query_dim=2048, key_dim=2048, value_dim=1024,
                              projection_dim=256, projection='linear',
                              layer_norm=True, l2_normalize=True):
        raise ValueError('Adapter does not match the verified implementation')
    return cfg


def calibrate(negative_scores, fpr):
    """Choose the lowest observed cutoff with empirical FPR at most fpr."""
    scores = torch.sort(negative_scores.flatten(), descending=True).values
    if not len(scores) or not torch.isfinite(scores).all() or not 0 <= fpr < 1:
        raise ValueError('Invalid calibration input')
    return float(torch.nextafter(scores[int(len(scores) * fpr)], torch.tensor(float('inf'))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('snapshot', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--full-validation', action='store_true')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    cfg = load_config(args.config)
    training = cfg['training']
    if training['optimizer'] != 'AdamW':
        raise ValueError('Unsupported optimizer')
    torch.set_num_threads(2)
    seed = cfg['dataset']['seeds'][0]
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    hashes = verify_snapshot(args.snapshot)
    hashes['tokenizer.json'] = sha(args.snapshot / 'tokenizer.json')
    if hashes != cfg['model']['hashes']:
        raise ValueError('Model or tokenizer identity mismatch')
    tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    tok.pad_token = tok.eos_token or tok.convert_ids_to_tokens(151643)
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=torch.float32,
        local_files_only=True, attn_implementation='sdpa').eval().requires_grad_(False).to(args.device)
    sizes = (*cfg['dataset']['entities'][:2], 0) if args.full_validation else (8, 8, 0)
    data = dataset(seed, sizes)
    del data['test']
    selection, calibration_rows = split_validation(data['validation'], seed)
    seen = ('query', 'write', 'query:1', 'write:1', 'bare', 'contextual')
    features = {split: {view: extract(model, tok, rows, view) for view in
                (seen if split == 'train' else (*seen, 'held_query', 'held_write'))}
                for split, rows in data.items()}
    unknown = unknown_queries(calibration_rows)
    families = {'A1':'query','A2':'query:1','B':'held_query'}
    calibration_features = {
        kind: {family:extract(model,tok,rows,query) for family,query in families.items()}
        for kind,rows in ({'known':calibration_rows}|unknown).items()
    }
    calibration_key = extract(model,tok,calibration_rows,'write')
    selection_entities = {row[0] for row in selection}
    selection_indices = torch.tensor([i for i,row in enumerate(data['validation'])
                                      if row[0] in selection_entities])
    train = features['train']
    val = features['validation']
    from torch.nn.functional import normalize
    ridge_train = {k: normalize(extract(model, tok, data['train'], k, last=True))
                   for k in ('write','query')}
    ridge_val = {k: normalize(extract(model, tok, data['validation'], k, last=True))
                 for k in ('write','query','held_write','held_query')}
    del model

    entity = [row[0] for row in data['train']]
    relation = [row[1] for row in data['train']]
    bias = torch.tensor([[float(i != j and (e == f or relation[i] == relation[j]))
                          for j,f in enumerate(entity)] for i,e in enumerate(entity)])
    bias *= training['hard_negative_logit_bias']

    def objective(a, b, extra=0):
        logits = a @ b.T / training['temperature'] + extra
        labels = torch.arange(len(a))
        return (torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)) / 2

    from torch.nn import functional as F

    views = {('A','A'):('write','query'), ('A','B'):('write','held_query'),
             ('B','A'):('held_write','query'), ('B','B'):('held_write','held_query')}
    conditions = {name: views[tuple(pair)] for name, pair in cfg['evaluation']['conditions'].items()
                  if not name.startswith('test_')}
    widths = [n for n in cfg['evaluation']['candidates'] if n <= len(val['write'])]
    args.output.mkdir(parents=True, exist_ok=False)

    @torch.no_grad()
    def evaluate(score, bank=val):
        result = {}
        for name, (w, q) in conditions.items():
            for width in widths:
                episodes = []
                for start in range(0, len(bank[q]) - width + 1, width):
                    ix = slice(start, start+width)
                    episodes.append(metrics(score(bank[q][ix], bank[w][ix])))
                result[f'{name}/{width}'] = episodes
        return result

    methods = {}
    raw = lambda q, k: F.normalize(q) @ F.normalize(k).T
    methods['Raw Cosine'] = evaluate(raw)
    methods['Random'] = evaluate(lambda q,k: torch.rand((len(q), len(k)), generator=generator))
    methods['Canonical Oracle'] = evaluate(lambda q,k: torch.eye(len(q)))
    # Full-text last-token baseline on the same corrected dataset.
    q, k = ridge_train['query'], ridge_train['write']
    ridge = torch.linalg.solve(q.T@q + torch.eye(q.shape[1]), q.T@k)
    methods['V1 Ridge'] = evaluate(lambda q,k: raw(q@ridge,k), ridge_val)
    if set(methods) | {'Shared Linear','Separate Linear','Separate MLP'} != set(cfg['evaluation']['methods']):
        raise ValueError('Configured baseline set differs from implemented methods')
    checkpoints = {}
    history = {}
    calibration = {}
    value_retrieval = {}
    for name in ('Shared Linear','Separate Linear','Separate MLP'):
        model = Adapter()
        if name == 'Shared Linear':
            model.key = model.query
        elif name == 'Separate MLP':
            for role in ('query','key'):
                setattr(model, role, torch.nn.Sequential(
                    torch.nn.Linear(2048, 256), torch.nn.GELU(),
                    torch.nn.Linear(256,256,bias=False), torch.nn.LayerNorm(256)))
        opt = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                                weight_decay=training['weight_decay'])
        best, best_score, stale, rows = None, -1., 0, []
        def score(q,k):
            return model.encode('query',q) @ model.encode('key',k).T

        epochs = training['max_epochs'] if args.full_validation else min(8,training['max_epochs'])
        for epoch in range(epochs):
            for ix in torch.randperm(len(entity), generator=generator).split(training['batch_size']):
                qv, kv = [('', ':1')[int(i)] for i in torch.randint(2, (2,), generator=generator)]
                qk = objective(model.encode('query', train['query'+qv][ix]),
                               model.encode('key', train['write'+kv][ix]), bias[ix][:,ix])
                value = objective(model.encode('value', train['bare'][ix]),
                                  model.encode('value', train['contextual'][ix]))
                opt.zero_grad()
                (qk + value).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'], error_if_nonfinite=True)
                opt.step()
            with torch.no_grad():
                validation = evaluate(score, {k:v[selection_indices] for k,v in val.items()})
            held = validation['validation_held_both/32']
            if not held:
                with torch.no_grad():
                    held = [metrics(score(val['held_query'][selection_indices],
                                          val['held_write'][selection_indices]))]
            mrr = sum(row['mrr'] for row in held)/len(held)
            rows.append(dict(epoch=epoch+1,qk=qk.item(),value=value.item(),validation_mrr=mrr))
            if mrr > best_score:
                best, best_score, stale = copy.deepcopy(model.state_dict()), mrr, 0
            else:
                stale += 1
                if stale >= training['early_stopping_patience']:
                    break
        path = args.output/(name.lower().replace(' ','-')+'.safetensors')
        save_file({key: value.clone() for key,value in best.items()}, str(path))
        model.load_state_dict(load_file(str(path)),strict=True)
        checkpoints[name] = sha(path)
        history[name] = rows
        methods[name] = evaluate(score)
        with torch.no_grad():
            left = model.encode('value',val['bare'])
            right = model.encode('value',val['contextual'])
            value_retrieval[name] = {width: [metrics(a @ b.T)
                for a,b in zip(left.split(width),right.split(width),strict=True) if len(a)==width]
                for width in widths}
        with torch.no_grad():
            key = model.encode('key', calibration_key)
            scored = {kind: {family: model.encode('query',x) @ key.T for family,x in bank.items()}
                      for kind,bank in calibration_features.items()}
            negatives = torch.cat([x.max(1).values for kind,bank in scored.items()
                                   if kind != 'known' for x in bank.values()])
            cutoff = calibrate(negatives, cfg['evaluation']['thresholds']['unknown_fpr_max'])
            breakdown = {}
            for family, known in scored['known'].items():
                maximum, winner = known.max(1)
                accepted = maximum >= cutoff
                right = winner.eq(torch.arange(len(known)))
                breakdown[family] = dict(
                    known_count=len(known), coverage=float(accepted.float().mean()),
                    selective_recall1=float(right[accepted].float().mean()) if accepted.any() else None,
                    unknown={kind:{'count':len(bank[family]),
                        'fpr':float((bank[family].max(1).values>=cutoff).float().mean())}
                        for kind,bank in scored.items() if kind!='known'})
            calibration[name] = dict(threshold=cutoff, negative_count=len(negatives),
                observed_fpr=float((negatives>=cutoff).float().mean()), families=breakdown)
    (args.output/'report.json').write_text(json.dumps(dict(
        status='VALIDATION_ONLY_PREFLIGHT', device=args.device,
        gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None, config_sha256=sha(args.config),
        checkpoint_sha256=checkpoints, seed=seed, test='SEALED', training=history,
        methods=methods, calibration=calibration,
        validation_partitions=dict(selection=sorted(selection_entities),
            calibration=sorted({row[0] for row in calibration_rows})), value_retrieval=value_retrieval, facts={k:len(v) for k,v in data.items()}), indent=2))


if __name__ == '__main__':
    main()
