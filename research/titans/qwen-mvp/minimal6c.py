"""Frozen Stage 6C evaluation using the existing segmented inference path."""
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys
import time
import random
from statistics import mean

import torch
from torch.nn.functional import cross_entropy

from train6c import CORPUS, MODEL_ID, REVISION, windows, measure
from stage2 import Qwen35Adapter, NeuralMemory
from hybrid6c import HybridSession, build_memories


@torch.no_grad()
def main(output, checkpoint):
    started = time.monotonic()
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
                  validation_nll=nll, reference=reference, tolerance=.01)
    output.write_text(json.dumps(report, indent=2))
    if abs(nll - reference) > report['tolerance']:
        return
    output.with_name('manifest.json').write_text(json.dumps(dict(
        checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
        checkpoint_update=64, identity=identity, test_windows=4, factual_seed=6406), indent=2))
    print(f'[VALIDATION] PASS NLL={nll:.6f}', flush=True)
    samples = list(islice(windows('test', tokenizer), 4))
    if len(samples) != 4 or len({tuple(ids) for ids in samples}) != 4:
        raise ValueError('Incomplete or repeated test windows')
    def norm(group):
        return sum(float(t.square().sum()) for t in group.values()) ** .5

    def evaluate(ids, donor_ids, probes=None):
        tokens = torch.tensor([ids], device='cuda')
        donor = torch.tensor([donor_ids[:512]], device='cuda')
        rows, resources, layers, recall = {}, {}, {}, {}
        for arm in ('E1', 'E2', 'E3'):
            print(f'{arm}: RUNNING', flush=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            session = HybridSession(adapter, modules)
            losses = []
            for start in (0, 512, 1024):
                chunk = tokens[:, start:start + 512]
                if session is not None:
                    hidden = session.forward(chunk, read=arm != 'E1')
                    if start == 0 and arm == 'E3':
                        other = HybridSession(adapter, modules)
                        other.forward(donor)
                        session.cells = other.cells
                if start and not probes:
                    losses.append(torch.cat([
                        cross_entropy(adapter.logits(hidden[:, j:j+16]).flatten(0, 1),
                                      tokens[0, start+j+1:start+j+17], reduction='none')
                        for j in range(0, 512, 16)
                    ]).tolist())
                if probes and start:
                    for segment, (offset, answer, wrong) in probes.items():
                        if start <= offset < start + 512:
                            scores = adapter.logits(hidden[:, offset-start])[0].float()
                            recall.setdefault(arm, {})[segment] = dict(
                                exact=int(scores.argmax()) == answer,
                                nll=float(-scores.log_softmax(0)[answer]),
                                correct_rank=int((scores > scores[answer]).sum()) + 1,
                                wrong_rank=int((scores > scores[wrong]).sum()) + 1)
            rows[arm] = losses
            torch.cuda.synchronize()
            cache = session.cache
            if session is not None and any(t is not None for t in cache.key_cache + cache.value_cache):
                raise RuntimeError('Segment boundary retained KV')
            def size(tensors):
                return sum(t.numel()*t.element_size() for t in tensors if t is not None)
            resources[arm] = dict(seconds=time.perf_counter()-started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                kv_bytes=size(cache.key_cache + cache.value_cache),
                recurrent_bytes=size(cache.recurrent_states + cache.conv_states),
                cell_bytes=0 if session is None else size(
                    [t for group in session.cells.values() for cell in group
                     for t in (*cell.fast.values(), *cell.momentum.values())]))
            if session is not None:
                layers[arm] = {i: dict(fusion=m.tolist(), shards=[
                    dict(step=c.update_step, fast_norm=norm(c.fast), momentum_norm=norm(c.momentum))
                    for c in session.cells[i]]) for i, m in session.metrics.items()}
            print(f"{arm}: COMPLETE {resources[arm]['seconds']:.1f}s", flush=True)
        return dict(rows=rows, resources=resources, layers=layers, recall=recall)


    def write_record(name, row):
        with output.with_name(name).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    paired, window_times = [], []
    for index, ids in enumerate(samples):
        donor = (index + 2) % 4
        if ids[:512] == samples[donor][:512]:
            raise ValueError('Donor history duplicates target')
        result = evaluate(ids, samples[donor])
        paired.append(result['rows'])
        window_times.append(sum(x['seconds'] for x in result['resources'].values()))
        write_record('windows.jsonl', dict(index=index, donor=donor, **result,
            token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
        print(f'Completed language window {index + 1}/4', flush=True)
        projected = mean(window_times) * (7 - index)
        print(f'Estimated remaining evaluation: {projected:.0f}s', flush=True)
        if time.monotonic() - started + projected > 4800:
            report.update(status='MINIMAL_DIAGNOSTIC_PAUSED', windows=index+1,
                          projected_remaining_seconds=projected)
            output.write_text(json.dumps(report, indent=2))
            return

    report['comparisons'] = paired_summary(paired)
    output.write_text(json.dumps(report, indent=2))
    rng = random.Random(6406)
    codes = [i for i in range(100, 16000)
             if (word := tokenizer.decode([i])).startswith(' ') and word[1:].isalpha()
             and word.isascii() and tokenizer.encode(word, add_special_tokens=False) == [i]]
    facts = rng.sample(codes, 4)
    for index, answer in enumerate(facts):
        wrong = facts[(index+2)%4]
        entity = f'VOR-{rng.randrange(1000000):06d}-{index:02d}'
        def encode(text):
            return tokenizer.encode(text, add_special_tokens=False)
        question = encode(f'What is the calibration label assigned to {entity}? The label is')
        target = encode(f'The calibration label assigned to {entity} is') + [answer]
        distractor = encode(f'The calibration label assigned to {entity} is') + [wrong]
        if answer in question or answer in distractor:
            raise ValueError('Probe answer leaked')
        filler = [x for x in encode(' Unrelated observations were recorded.') if x != answer]
        def pad(prefix, suffix=()):
            count = 512 - len(prefix) - len(suffix)
            if count < 0 or not filler:
                raise ValueError('Invalid probe geometry')
            return prefix + (filler * 512)[:count] + list(suffix)
        b = pad([], question)
        c = pad([], question)
        result = evaluate(pad(target)+b+c+[answer], pad(distractor),
                          {'B': (1023, answer, wrong), 'C': (1535, answer, wrong)})
        write_record('facts.jsonl', dict(index=index, entity=entity, answer=answer, wrong=wrong, **result))
        print(f'Completed factual probe {index + 1}/4', flush=True)

    report.update(status='MINIMAL_DIAGNOSTIC_COMPLETE', windows=4, mvp='INCONCLUSIVE')
    output.write_text(json.dumps(report, indent=2))
    output.with_name('RESULTS.md').write_text(
        '# Stage 6C diagnostic\n\nValidation NLL: ' + str(nll) +
        '\n\nPositive paired deltas favor E2.\n\n```json\n' +
        json.dumps(report['comparisons'], indent=2) + '\n```\n')


def paired_summary(windows):
    result = {}
    for segment, sl in [('B', slice(0, 512)), ('C', slice(512, 1024)), ('BC', slice(None))]:
        scores = {arm: [mean(row[arm][sl]) for row in windows] for arm in windows[0]}
        comparisons = {}
        for arm in ('E1', 'E3'):
            delta = [a-b for a,b in zip(scores[arm], scores['E2'])]
            comparisons[arm] = dict(mean=mean(delta), per_window=delta,
                                    wins=sum(v > 0 for v in delta))
        result[segment] = dict(nll={arm: mean(values) for arm, values in scores.items()},
                               control_minus_E2=comparisons)
    return result


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
