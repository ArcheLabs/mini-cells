"""CPU capacity smoke. Every measurement uses the frozen M3 memory."""
import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_m1 import M3, file_hash, gate5, initialize, read, state_hash, tensor_hash, write

CONFIG = dict(sizes=[4, 8, 16, 32, 64], seeds=[41036, 41037, 41038],
              exposure=[1, 8, 32, 64, 128], threshold=.8, unknown_per_kind=256)


def generate(n, seed, geometry, dim):
    rng = torch.Generator().manual_seed(seed + 1000*n)

    def bank():
        x = torch.randn(n, dim, generator=rng)
        if geometry == 'qr':
            if n > dim:
                raise ValueError('QR exceeds latent dimension')
            return torch.linalg.qr(x.T).Q.T.contiguous()
        return F.normalize(x, dim=-1)

    keys, candidates = bank(), bank()
    targets = torch.randperm(n, generator=rng)
    count = CONFIG['unknown_per_kind']
    anchors = keys[torch.randint(n, (count,), generator=rng)]
    noise = torch.randn(count, dim, generator=rng)
    residual = F.normalize(noise - (noise*anchors).sum(-1, keepdim=True)*anchors, dim=-1)
    cosine = .8 + .15*torch.rand(count, 1, generator=rng)
    hard = cosine*anchors + (1-cosine.square()).sqrt()*residual
    unknown = F.normalize(torch.randn(count, dim, generator=rng), dim=-1)
    for values in (keys, candidates):
        assert len(torch.unique(values, dim=0)) == n
    for queries in (unknown, hard):
        assert not (queries[:, None] == keys).all(-1).any()
    return keys, candidates, targets, unknown, hard, (anchors*hard).sum(-1)


def stats(x):
    return dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
                sd=float(x.std(unbiased=False)))


def score(output, candidates, targets):
    cosine = F.normalize(output, dim=-1) @ candidates.T
    confidence, labels = cosine.max(-1)
    values = candidates[targets]*math.sqrt(candidates.shape[-1])
    return dict(recall=float(labels.eq(targets).float().mean()),
                cosine=float(F.cosine_similarity(output, values).mean()),
                mse=float(F.mse_loss(output, values))), confidence, labels.eq(targets)


def calibration(memory, cell, data):
    keys, candidates, targets, random, hard, _ = data
    _, known, correct = score(read(memory, cell, keys), candidates, targets)
    scores = [(F.normalize(read(memory, cell, x), dim=-1) @ candidates.T).amax(-1)
              for x in (random, hard)]
    threshold = CONFIG['threshold']
    accepted = known >= threshold
    tp = int(accepted.sum())
    fp = [int((x >= threshold).sum()) for x in scores]
    curves = []
    for cutoff in torch.cat([known, *scores]).unique().sort(descending=True).values:
        k = int((known >= cutoff).sum())
        u = [int((x >= cutoff).sum()) for x in scores]
        curves.append([float(cutoff), k/len(known), *(v/len(scores[0]) for v in u),
                       k/max(1, k+sum(u))])
    return dict(threshold=threshold, known_fn=1-tp/len(known), tpr=tp/len(known),
                accepted_correct_recall=float((accepted & correct).float().mean()),
                precision=tp/max(1, tp+sum(fp)), unknown_fp_counts=fp,
                unknown_fpr=[v/len(scores[0]) for v in fp], roc_pr=curves)


def acquire(memory, cell, keys, values, indices):
    memory.recall(keys[indices][None], cell, 'read_only')
    return memory.learn(keys[indices][None], values[indices][None], cell, 'online_adapt')[0]


def run_episode(folder, n, seed, geometry, method, width, exposures):
    memory = initialize(41035)
    data = generate(n, seed, geometry, memory.dim)
    keys, candidates, targets, random, hard, cosine = data
    values = candidates[targets]*math.sqrt(memory.dim)
    cell = memory.create(mode='online_adapt')
    initial = cell.detached()
    counts = torch.zeros(n, dtype=torch.int64)
    rng = torch.Generator().manual_seed(seed+17)
    curve, prefixes = [], []
    elapsed = 0.
    for exposure in range(1, max(exposures)+1):
        order = torch.randperm(n, generator=rng)
        if method == 'reverse':
            order = order.flip(0)
        seen = []
        for indices in order.split(width):
            started = time.perf_counter()
            cell = acquire(memory, cell, keys, values, indices)
            elapsed += time.perf_counter()-started
            counts[indices] += 1
            seen.extend(indices.tolist())
            if exposure == 1 and len(seen) in {4, 8, 16, 32, n}:
                metrics, _, _ = score(read(memory, cell, keys[seen]), candidates, targets[seen])
                prefixes.append(dict(seen=len(seen), **metrics))
        assert (counts == exposure).all()
        assert cell.update_step == exposure*math.ceil(n/width)
        if exposure in exposures:
            metrics, _, _ = score(read(memory, cell, keys), candidates, targets)
            delta = sum(float((v.detach()-initial.fast[k]).square().sum()) for k,v in cell.fast.items())**.5
            momentum = sum(float(v.square().sum()) for v in cell.momentum.values())**.5
            assert 0 < delta < math.inf and 0 < momentum < math.inf
            curve.append(dict(exposure=exposure, updates=cell.update_step, presentations=int(counts.sum()),
                              seconds=elapsed, fast_delta=delta, momentum_norm=momentum, **metrics))
    expected = read(memory, cell, keys)
    snapshot = state_hash(cell)
    identity = {'initializer_seed':41035, 'interface':1, 'task':'p4-m11'}
    checkpoint = memory.checkpoint(cell, folder, identity)
    del cell, memory, initial
    fresh = initialize(41035)
    restored = fresh.restore(folder, identity)
    error = float((read(fresh, restored, keys)-expected).abs().max())
    assert error == 0 and state_hash(restored) == snapshot
    i,j = torch.triu_indices(n,n,1)
    return dict(n=n, seed=seed, method=method, width=width, geometry=geometry,
                generator='p4-m11-v1', shape=list(keys.shape), dtype=str(keys.dtype),
                hashes=[tensor_hash(x) for x in (keys, values, random, hard)],
                geometry_stats=[stats((x@x.T)[i,j]) for x in (keys,candidates)],
                cross_cosine=stats(keys@candidates.T), hard_anchor_cosine=stats(cosine),
                counts=counts.tolist(), curve=curve, prefixes=prefixes, random_recall=1/n,
                restore_error=error, checkpoint=checkpoint, unknown=calibration(fresh,restored,data))


def forgetting(seed):
    memory = initialize(41035)
    keys, candidates, targets, *_ = generate(12, seed, 'gaussian', memory.dim)
    values = candidates[targets]*math.sqrt(memory.dim)
    rng = torch.Generator().manual_seed(seed)
    results = []
    for groups in ([0,1,2], [2,1,0], torch.randperm(3,generator=rng).tolist()):
        cell = memory.create(mode='online_adapt')
        initial, trajectory = {}, []
        for group in groups:
            idx = torch.arange(4*group,4*group+4)
            for _ in range(128):
                cell = acquire(memory,cell,keys,values,idx)
            recalls = [score(read(memory,cell,keys[i:i+4]), candidates, targets[i:i+4])[0]['recall']
                       for i in range(0,12,4)]
            initial[group] = recalls[group]
            trajectory.append(dict(group=group, recalls=recalls,
                                   retention={g:recalls[g]/r if r else None for g,r in initial.items()}))
        results.append(dict(order=groups, trajectory=trajectory))
    return results


def run(output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('memory.py','gates.py'):
        assert file_hash(M3/name) == file_hash(M3/'freeze/original-source'/name)
    started = time.perf_counter()
    result = dict(status='RUNNING', device='cpu', torch=torch.__version__, config=CONFIG,
                  python=sys.version, source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  runner_sha256=file_hash(__file__), memory_sha256=file_hash(M3/'memory.py'),
                  episodes=[], failures=[])
    try:
        result['control'] = gate5(output/'control')
        for n, seed in itertools.product(CONFIG['sizes'], CONFIG['seeds']):
            methods = [('gaussian','batch',n), ('gaussian','sequential',1),
                       ('gaussian','chunk4',4), ('gaussian','chunk8',8)]
            if n <= 32:
                methods.append(('qr','batch',n))
            if n == 8:
                methods.append(('gaussian','reverse',1))
            for geometry, method, width in methods:
                name = f'{n}-{seed}-{geometry}-{method}'
                try:
                    r = run_episode(output/name,n,seed,geometry,method,width,CONFIG['exposure'])
                    result['episodes'].append(r)
                except Exception as error:
                    result['failures'].append(dict(episode=name,error=repr(error)))
        result['forgetting'] = {s:forgetting(s) for s in CONFIG['seeds']}
        result['status'] = 'PASS' if not result['failures'] else 'FAIL'
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        write(output/'manifest.json',result)
    if result['failures']:
        raise RuntimeError('Smoke episode failures. See manifest.json.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output)
