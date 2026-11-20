"""Read-only attribution of the frozen P1.1 replay."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, pstdev
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / p) for p in ('p1-1', 'p1-meta')]
from corpus import CORPUS, SOURCE, select
from protocol import SEEDS, donor_assignments, paired_summary

MODEL = 'Qwen/Qwen3-0.6B-Base'
REV = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
UP = 'c60b4efaa3a3c6df4deebd57cbccf75947b87cad'
ART = '0cc6870ded45abd3a0f5835cc9700f0fb52f8784'
SOURCE_COMMIT = 'ff1adc83f9a225b83164035bb46d0bb0e22c6f67'
STEPS = (0, 20, 200)
KIND = 'minicells.titans-p1-2.v1'


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root):
    metadata = load(root / 'metadata.json')
    files = {f['path']: f for f in metadata['files']}
    if load(root / 'decision.json')['source_commit'] != SOURCE_COMMIT:
        raise ValueError('Unexpected historical source')
    for name, entry in files.items():
        path = root / name
        expected = (entry['bytes'], entry['sha256'])
        if name == 'RESULTS.md':
            # The historical publisher rewrote only this display page after manifest creation.
            expected = (669, '0d58b370d14343b0656ca024ccd22893a3434da88bc8460ebce91b8e8203f1e9')
        if (path.stat().st_size, sha(path)) != expected:
            raise ValueError(f'Published artifact mismatch: {name}')
    for seed in SEEDS:
        folder = f'seed_{seed}'
        hashes = load(root / folder / 'checkpoint_hashes.json')
        for step in STEPS:
            name = f'step_{step:03}.pt'
            if hashes[name] != files[f'{folder}/{name}']['sha256']:
                raise ValueError(f'Checkpoint hash mismatch: {folder}/{name}')
    return files


def restore(wrapper, path, step):
    data = torch.load(path, map_location='cpu', weights_only=True)
    named = dict((n, p) for n, p in wrapper.named_parameters() if p.requires_grad)
    if data['step'] != step or set(data['parameters']) != set(named):
        raise ValueError('Checkpoint step or parameter names mismatch')
    hashes = {}
    with torch.no_grad():
        for name, param in named.items():
            value = data['parameters'][name]
            if value.shape != param.shape or value.dtype != param.dtype:
                raise ValueError(f'Checkpoint tensor mismatch: {name}')
            finite(value)
            param.copy_(value)
            torch.testing.assert_close(param.cpu(), value, atol=0, rtol=0)
            hashes[name] = {'sha256': hashlib.sha256(value.contiguous().view(-1).view(torch.uint8).numpy()).hexdigest(),
                            'numel': value.numel(), 'shape': list(value.shape)}
    return hashes


def finite(x):
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite diagnostic tensor')
    return x


def rms(x):
    return x.float().square().mean().sqrt().item()


def compare(x, y):
    x, y = x.float().flatten(), y.float().flatten()
    delta = rms(x - y)
    return {'cosine': F.cosine_similarity(x, y, dim=0).item(), 'rms_delta': delta,
            'relative_rms': delta / (rms(x) + 1e-12)}


def describe(x):
    x = finite(x.float()).flatten()
    p05, median, p95 = x.quantile(x.new_tensor([.05, .5, .95])).tolist()
    return {'min': x.min().item(), 'max': x.max().item(), 'mean': x.mean().item(),
            'std': x.std(unbiased=False).item(), 'p05': p05, 'median': median, 'p95': p95,
            'near_zero': (x.abs() < 1e-8).float().mean().item(),
            'below_001': (x < .01).float().mean().item(), 'above_05': (x > .5).float().mean().item(),
            'above_099': (x > .99).float().mean().item()}


def tensor_stats(tensors):
    return [{'shape': list(t.shape), 'rms': rms(finite(t)), 'norm': t.float().norm().item(),
             'head_norms': t.float().flatten(1).norm(dim=1).tolist()} for t in tensors]


@contextmanager
def capture(memory, *, write):
    """Observe the real calls; return exactly the original outputs."""
    captured = {}
    with ExitStack() as stack:
        def wrap(name, observer):
            original = getattr(memory, name)
            def call(*args, **kwargs):
                result = original(*args, **kwargs)
                observer(original, args, result)
                return result
            setattr(memory, name, call)
            stack.callback(delattr, memory, name)

        def hook(module, name, transform):
            handle = module.register_forward_hook(
                lambda _m, _i, o: captured.__setitem__(name, transform(o.detach())))
            stack.callback(handle.remove)

        if write:
            from titans_pytorch.neural_memory import softclamp_grad_norm
            def gradients(original, args, result):
                if 'weighted_gradient' in captured:
                    raise ValueError('Expected one gradient evaluation per chunk')
                weights, keys, lr, values = args
                weighted = result[0]
                raw = original(weights, keys, torch.ones_like(lr), values)[0]
                captured.update(lr=lr.detach(), raw_gradient=tuple(t.detach() for t in raw),
                                weighted_gradient=tuple(t.detach() for t in weighted),
                                surprise=tuple(-softclamp_grad_norm(t.detach(), memory.max_grad_norm)
                                               for t in weighted))
            wrap('memory_grads', gradients)
            hook(memory.to_momentum, 'beta', torch.sigmoid)
            hook(memory.to_decay_factor, 'decay', torch.sigmoid)
        else:
            wrap('read_only', lambda _fn, _args, out: captured.__setitem__('read', out[0].detach()))
            hook(memory.history_mix_proj, 'gate', lambda x: (x.reshape(*x.shape[:2], 16, 128)
                 + memory.history_mix.detach().reshape(1, 1, 16, 128)).sigmoid().flatten(2))
        yield captured


def flat(tensors):
    return torch.cat([t.float().reshape(-1) for t in tensors])


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError('State structure changed')
        for x, y in zip(a, b):
            same_tree(x, y)
    elif a != b:
        raise ValueError('State metadata changed')


def compact_state(state, device='cpu'):
    return type(state)(*(snapshot(v, device) for v in state))


def snapshot(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, copy=True)
    if isinstance(x, tuple):
        return tuple(snapshot(v, device) for v in x)
    return x


@torch.no_grad()
def replay(model, wrapper, docs, donors, historical, folder, *, smoke=False):
    started = time.monotonic()
    reference_reads = {row['document_index']: row for row in historical['diagnostics']}
    historical = historical['documents']
    memory = wrapper.memory
    device = next(model.parameters()).device
    if (memory.momentum_order != 1 or memory.spectral_norm_surprises
            or memory.max_grad_norm != 1.0 or memory.to_layer_modulation is not None
            or memory.to_learned_momentum_combine is not None):
        raise ValueError('Unexpected pinned write configuration')

    def forward(ids, start):
        t = torch.tensor(ids, device=device)[None]
        p = torch.arange(start, start + len(ids), device=device)[None]
        return model(input_ids=t, position_ids=p, use_cache=False).logits

    names = ('write_dynamics', 'fast_weight_pairs', 'read_diagnostics', 'prediction_replay')
    histories, traces, predictions, attributed = {}, {}, [], []
    slow = tuple(t.detach() for t in memory.init_weights(1))
    save(folder / 'slow_weights.json', {'parameters': tensor_stats(slow), 'norm': flat(slow).norm().item()})
    with ExitStack() as stack:
        streams = {n: stack.enter_context((folder / f'{n}.jsonl').open('w')) for n in names}
        def emit(name, row):
            streams[name].write(json.dumps(row, allow_nan=False) + '\n')
            streams[name].flush()

        def write(doc, instrument, index):
            wrapper.state = None
            series = []
            for chunk, offset in enumerate((0, 512, 1024)):
                forward(doc[offset:offset + 512], offset)
                if not instrument:
                    wrapper.commit()
                    continue
                previous = wrapper.state
                with capture(memory, write=True) as observed:
                    wrapper.commit()
                latest, momentum = wrapper.state.states
                delta = tuple(t - s for t, s in zip(latest, slow, strict=True))
                prior = previous.states[0] if previous is not None else slow
                update = tuple(t - p for t, p in zip(latest, prior, strict=True))
                momentum = tuple(t[0] for t in momentum)
                raw, weighted, surprise = (observed[k] for k in ('raw_gradient', 'weighted_gradient', 'surprise'))
                decay = observed['decay'][:, 0]
                beta = observed['beta'][0, :, 0]
                prior_momentum = tuple(t[0] for t in previous.states[1]) if previous is not None else tuple(torch.zeros_like(t) for t in momentum)
                def scale(factor, tensor):
                    return factor.reshape(-1, *([1] * (tensor.ndim - 1))) * tensor
                decay_delta = tuple(scale(decay, st - pt) for st, pt in zip(slow, prior, strict=True))
                carry = tuple(scale(beta, t) for t in prior_momentum)
                momentum_error = flat(momentum) - flat(tuple(t + u for t, u in zip(carry, surprise, strict=True)))
                fast_error = flat(update) - flat(tuple(t + u for t, u in zip(decay_delta, momentum, strict=True)))
                vectors = {k: flat(v).cpu() for k, v in dict(raw_gradient=raw, surprise=surprise,
                                                          momentum=momentum, fast_delta=delta, decay_delta=decay_delta,
                                                          momentum_carry=carry).items()}
                vectors.update({key: observed[key].flatten().cpu() for key in ('lr', 'beta', 'decay')})
                emit('write_dynamics', dict(document=index, chunk=chunk, fast=tensor_stats(latest),
                     fast_delta=tensor_stats(delta), momentum=tensor_stats(momentum),
                     retrieval_updates=tensor_stats(wrapper.state.updates),
                     raw_gradient=tensor_stats(raw), weighted_gradient=tensor_stats(weighted),
                     surprise=tensor_stats(surprise), update=tensor_stats(update),
                     decay_delta=tensor_stats(decay_delta), momentum_carry=tensor_stats(carry),
                     momentum_recurrence_rms_error=rms(finite(momentum_error)),
                     fast_recurrence_rms_error=rms(finite(fast_error)),
                     surprise_to_fast_norm=vectors['surprise'].norm().item() / (flat(latest).norm().item() + 1e-12),
                     momentum_vs_surprise=compare(vectors['momentum'], vectors['surprise']),
                     previous_chunk_surprise=compare(series[-1]['surprise'], vectors['surprise']) if series else None,
                     **{key: describe(observed[key]) for key in ('lr', 'beta', 'decay')}))
                vectors['fast_delta_heads'] = torch.cat([t.flatten(1) for t in delta], dim=1).cpu()
                series.append(vectors)
            return compact_state(wrapper.state), series

        plain, _ = write(docs[0], False, 0)
        for index in (sorted({0, *donors[0]}) if smoke else range(len(docs))):
            doc = docs[index]
            state, trace = write(doc, True, index)
            histories[index], traces[index] = state, trace
            if index == 0 or (index + 1) % 8 == 0:
                print(f'HISTORY_REPLAY document={index+1} elapsed={time.monotonic()-started:.1f}s', flush=True)
            if index == 0:
                same_tree(plain, state)

        for i, doc in enumerate(docs[:1] if smoke else docs):
            print(f'Prediction replay {i+1}/{len(docs)}', flush=True)
            query = doc[-512:]
            target = torch.tensor(query[1:], device=device)
            values = {}
            def predict(state, override=None, instrument=False):
                wrapper.state = None if state is None else compact_state(state, device)
                wrapper.read_override = override
                if instrument:
                    with capture(memory, write=False) as observed:
                        logits = forward(query, 1536)[0, :-1].float()
                    return logits, wrapper.pending, observed
                return forward(query, 1536)[0, :-1].float(), wrapper.pending, {}

            correct, qkv, obs = predict(histories[i], instrument=True)
            if i == 0:
                same_tree(correct, predict(plain)[0])
            lp = finite(correct.log_softmax(-1))
            differences = []
            def score(logits):
                probabilities = finite(logits.log_softmax(-1)).gather(1, target[:, None]).flatten()
                return F.cross_entropy(logits, target).item(), probabilities.tolist()
            values['correct'], token_lp = score(correct)
            token_prob = {'correct': token_lp}
            gate = obs['gate'].reshape(1, 512, 16, 128)
            for j, d in enumerate(donors[i], 1):
                for chunk, (a, b) in enumerate(zip(traces[i], traces[d], strict=True)):
                    emit('fast_weight_pairs', dict(document=i, donor=d, chunk=chunk,
                         metrics={k: compare(a[k], b[k]) for k in a if k != 'fast_delta_heads'},
                         fast_delta_by_head=[compare(x, y) for x, y in zip(
                             a['fast_delta_heads'], b['fast_delta_heads'], strict=True)]))
                other, other_qkv, donor_obs = predict(histories[d], instrument=True)
                same_tree(qkv, other_qkv)
                same_tree(obs['gate'], donor_obs['gate'])
                rd = compare(obs['read'], donor_obs['read'])
                if i in reference_reads:
                    for current, key in ((rd['cosine'], 'read_cosine_correct_donor'),
                                         (rd['rms_delta'], 'read_rms_delta')):
                        torch.testing.assert_close(current, reference_reads[i][key][j-1],
                                                   atol=1e-5, rtol=1e-5)
                delta = (obs['read'] - donor_obs['read']).reshape_as(gate)
                emit('read_diagnostics', dict(document=i, donor=d, **rd,
                     correct_read_rms=rms(obs['read']), donor_read_rms=rms(donor_obs['read']),
                     correct_read_norm=obs['read'].float().norm().item(),
                     donor_read_norm=donor_obs['read'].float().norm().item(), gated_delta_rms=rms(delta * gate),
                     heads=[rms(delta[:, :, h]) for h in range(16)], gate=describe(gate),
                     gate_heads=[describe(gate[:, :, h]) for h in range(16)]))
                key = f'donor_{j}'
                values[key], token_prob[key] = score(other)
                metrics = dict(donor=d, **compare(correct, other),
                    kl=(lp.exp() * (lp - other.log_softmax(-1))).sum(-1).mean().item(),
                    top1_flip=(correct.argmax(-1) != other.argmax(-1)).float().mean().item())
                differences.append(metrics)
                a, b = traces[i][-1], traces[d][-1]
                attributed.append(dict(**{label: compare(a[k], b[k])['rms_delta'] for label, k in (
                    ('raw_gradient', 'raw_gradient'), ('surprise', 'surprise'), ('momentum', 'momentum'),
                    ('decay_delta', 'decay_delta'), ('fast', 'fast_delta'))},
                    read=rd['rms_delta'], gated_read=rms(delta * gate), logits=metrics['rms_delta'],
                    nll=values[key]-values['correct']))
                del other
            values['donor_mean'] = mean(values[f'donor_{j}'] for j in range(1, 4))
            for name, state, override in (('empty', None, None), ('zero_read', histories[i], 'zero')):
                logits, other_qkv, _ = predict(state, override)
                same_tree(qkv, other_qkv)
                values[name], token_prob[name] = score(logits)
                del logits
            ref = historical[i]
            emit('prediction_replay', dict(document=i, nll=values, target_log_probability=token_prob,
                                           logits=differences, historical={k: ref[k] for k in values}))
            for key, value in values.items():
                torch.testing.assert_close(value, ref[key], rtol=1e-5, atol=1e-5,
                                           msg=f'P1.1 NLL mismatch at document {i}, {key}: {value} vs {ref[key]}')
            predictions.append(values)
    return predictions, attributed


def distribution(values):
    return dict(n=len(values), mean=mean(values), std=pstdev(values), min=min(values), max=max(values))


def summarize(rows, pairs):
    keys = tuple(pairs[0])
    correlations = {}
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            x, y = (torch.tensor([p[k] for p in pairs], dtype=torch.float64) for k in (a, b))
            correlations[f'{a}:{b}'] = torch.corrcoef(torch.stack((x, y)))[0, 1].item() if x.std() and y.std() else None
    return {'prediction': {k: distribution([r[k] for r in rows]) for k in rows[0]},
            'paired': {k: distribution([r[k] for r in pairs]) for k in keys},
            'correct_win_rate': mean(r['nll'] > 0 for r in pairs), 'correlations': correlations}


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from p1 import install
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('upstream', 'artifact', 'output'):
        p.add_argument(arg, type=Path)
    p.add_argument('--smoke', action='store_true', help='One checkpoint and one query with its original donors')
    a = p.parse_args()
    started = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    git = lambda path: subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    if git(a.upstream) != UP:
        raise ValueError('Unexpected Titans revision')
    sys.path.insert(0, str(a.upstream.resolve()))
    from importlib.metadata import version
    config = dict(transformers=version('transformers'), datasets=version('datasets'),
                  cuda=torch.version.cuda, model=MODEL, revision=REV, upstream_revision=UP, artifact_commit=ART,
                  source_commit=git(HERE), historical_source_commit=SOURCE_COMMIT, layer=3,
                  chunk=512, seeds=SEEDS[:1] if a.smoke else SEEDS,
                  steps=STEPS[:1] if a.smoke else STEPS, smoke=a.smoke,
                  device=torch.cuda.get_device_name(), torch=torch.__version__,
                  kaggle_script_version_id=os.getenv('KAGGLE_SCRIPT_VERSION_ID'))
    save(a.output / 'config.json', config)
    if sha(a.artifact / 'metadata.json') != 'd3bf9d821ffd930864069122a603c505044954359ce32067014ededa10ffabb9':
        raise ValueError('P1.1 metadata differs from the frozen artifact commit')
    save(a.output / 'input_manifest.json', verify(a.artifact))
    print('CHECKPOINT_INTEGRITY_PASS', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    def rows(split):
        return load_dataset(SOURCE, 'wikitext-103-raw-v1', revision=CORPUS, split=split, streaming=True)
    docs, manifest = select(rows('train'), rows('validation'),
                            lambda text: tokenizer.encode(text, add_special_tokens=False), holdout_size=64)
    manifest['model_revision'] = REV
    if manifest != load(a.artifact / 'data_manifest.json'):
        raise ValueError('Rebuilt corpus differs from P1.1')
    assignments = donor_assignments(64, 3)
    if json.loads(json.dumps(assignments)) != load(a.artifact / 'donor_assignments.json'):
        raise ValueError('Rebuilt donors differ from P1.1')
    save(a.output / 'integrity.json', dict(checkpoints='PASS', corpus='PASS', donors='PASS',
         historical_display_report='Publisher rewrite verified against pinned Git blob SHA-256'))
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float16,
                                                 attn_implementation='eager').eval().cuda()
    block = model.model.layers[3]
    native = block.self_attn
    documents = docs['primary_holdout']
    stats, grouped = {}, {step: ([], []) for step in config['steps']}
    for seed in config['seeds']:
        block.self_attn = native
        torch.manual_seed(seed)
        wrapper = install(model)
        wrapper.eval()
        if any(p.requires_grad for n, p in model.named_parameters() if not n.startswith('model.layers.3.self_attn.memory.')):
            raise ValueError('Backbone is not frozen')
        for step in config['steps']:
            print(f'REPLAY_START seed={seed} step={step}', flush=True)
            name = f'seed_{seed}/step_{step:03}'
            folder = a.output / name
            folder.mkdir(parents=True)
            original = a.artifact / f'seed_{seed}'
            save(folder / 'restore.json', restore(wrapper, original / f'step_{step:03}.pt', step))
            historical = load(original / f'validation_step_{step:03}.json')
            pred, pairs = replay(model, wrapper, documents, assignments, historical, folder, smoke=a.smoke)
            stats[name] = summarize(pred, pairs)
            save(folder / 'summary.json', stats[name])
            grouped[step][0].extend(pred)
            grouped[step][1].extend(pairs)
            wrapper.state = wrapper.pending = None
    aggregate = {str(step): summarize(*data) for step, data in grouped.items()}
    for step in config['steps']:
        rows = grouped[step][0]
        series = [[r['donor_mean'] - r['correct'] for r in rows[i:i+len(documents)]]
                  for i in range(0, len(rows), len(documents))]
        aggregate[str(step)]['paired_document_nll'] = paired_summary(series) if not a.smoke else None
    save(a.output / 'statistics.json', {'checkpoints': stats, 'steps': aggregate})
    decision = dict(format=KIND, experiment_id='titans-p1-2', engineering_status='PASS',
                    status='DIAGNOSTIC_ONLY', diagnostic_status='SMOKE_ONLY' if a.smoke else 'COMPLETE', scientific_status='DIAGNOSTIC_ONLY',
                    p1_1_replay='SMOKE_ONLY' if a.smoke else 'PASS', checkpoint_integrity='PASS',
                    corpus_integrity='PASS', donor_integrity='PASS', read_replay='PASS',
                    historical_scientific_status='INCONCLUSIVE', source_commit=config['source_commit'],
                    upstream_revision=UP, kaggle_script_version_id=config['kaggle_script_version_id'],
                    elapsed_seconds=time.monotonic() - started)
    save(a.output / 'decision.json', decision)
    report = ['# Titans P1.2 attribution', '', f"Coverage: {decision['diagnostic_status']}", '',
              '| Step | Pairs | Fast delta RMS difference | Read RMS difference | Logit RMS difference | Donor − Correct NLL |',
              '|---|---:|---:|---:|---:|---:|']
    for step, data in aggregate.items():
        m = data['paired']
        report.append(f"| {step} | {m['fast']['n']} | " + ' | '.join(f"{m[k]['mean']:.7g}" for k in ('fast','read','logits','nll')) + ' |')
    report.extend(['', '## Update and read attribution', ''])
    for step, data in aggregate.items():
        m = data['paired']
        report.append(f"- Step {step}: " + ', '.join(f"{k} RMS separation {m[k]['mean']:.7g}" for k in
                      ('raw_gradient', 'surprise', 'momentum', 'decay_delta', 'gated_read')) +
                      f"; Correct beats Donor in {data['correct_win_rate']:.2%} of pairs.")
        if data['paired_document_nll'] is not None:
            report.append(f"  Paired document NLL gain 95% CI: {data['paired_document_nll']['ci95']}.")
    report.extend(['', 'Diagnostic label: MIXED / UNRESOLVED pending review of per-head, per-document and paired statistics.',
                   'P1.1 remains INCONCLUSIVE. No training or parameter updates were performed.'])
    (a.output / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    save(a.output / 'sha256.json', {f.relative_to(a.output).as_posix(): sha(f) for f in sorted(a.output.rglob('*')) if f.is_file()})


if __name__ == '__main__':
    main()
