"""Clean-slate episodic training on native pre-RoPE Qwen head features."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import uuid
import time

import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'qwen-mvp'))
from statistics7 import paired
from stage2 import NeuralMemory, Qwen35Adapter
from minicells_runtime.qwen35 import MODEL_ID, REVISION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.address = nn.ModuleList(nn.Linear(256, 32) for _ in range(2))
        self.value = nn.ModuleList(nn.Linear(256, 32) for _ in range(2))
        self.readout = nn.ModuleList(nn.Linear(32, 32) for _ in range(2))
        self.shards = nn.ModuleList(NeuralMemory(32) for _ in range(2))

    def forward(self, episode, *, exact=False, control='correct', width=1, mode='online_adapt', capture=None, positions=None, intervention=None, distractors=None, query_override=None, tnt=False, trajectory=None):
        keys, queries, values = episode
        if query_override is not None:
            queries = query_override
        assert keys.shape == values.shape and keys.shape[1:] == (2, 256)
        assert queries.shape == (keys.shape[0], 8, 256)
        scores, alignment, separation, retention = [], [], [], []
        for h, shard in enumerate(self.shards):
            k = F.normalize(self.address[h](keys[:, h]), dim=-1)
            q = k[:, None].expand(-1, 4, -1) if exact else F.normalize(
                self.address[h](queries[:, h*4:h*4+4]), dim=-1)
            if tnt and control != 'empty':
                _, singular, basis = torch.linalg.svd(k, full_matrices=False)
                basis = basis[singular > singular.max() * 1e-5]
                q = F.normalize((q @ basis.T) @ basis, dim=-1)
            v = self.value[h](values[:, h])
            cell = shard.create(mode=mode)
            if control != 'empty':
                targets = v if control == 'correct' else v.roll(1, 0)
                groups = (positions // width).unique_consecutive(return_counts=True)[1].tolist() if positions is not None else [1] * len(k)
                seen = 0
                for x, y in zip(k.split(groups), targets.split(groups)):
                    cell, _ = shard.learn(x[None], y[None], cell, mode)
                    if trajectory is not None:
                        previous = seen
                        seen += len(x)
                        scores_now = F.normalize(self.readout[h](shard.recall(k[:seen][None], cell)[0]), dim=-1) @ F.normalize(v, dim=-1).T
                        hits = scores_now.argmax(1).eq(torch.arange(seen, device=k.device))
                        trajectory.append(dict(shard=h, step=cell.update_step, seen=seen,
                            old_recall=float(hits[:previous].float().mean()) if previous else None,
                            new_recall=float(hits[previous:].float().mean())))
                if distractors is not None:
                    dk = F.normalize(self.address[h](distractors[0][:, h]), dim=-1)
                    dv = self.value[h](distractors[2][:, h])
                    for x, y in zip(dk.split(1), dv.split(1)):
                        cell, _ = shard.learn(x[None], y[None], cell, mode)
                if intervention == 'latest':
                    cell, _ = shard.learn(k[:1][None], v[1:2][None], cell, mode)
                if intervention == 'fork':
                    from cell_artifact import state_hash
                    before = state_hash(cell)
                    fork = shard.fork(cell)
                    updated, _ = shard.learn(k[:1][None], v[1:2][None], fork, mode)
                    if state_hash(cell) != before or updated.update_step != cell.update_step + 1:
                        raise RuntimeError('Fork isolation failed')
            read = shard.recall(q.flatten(0, 1)[None], cell,
                                'meta_train' if mode == 'meta_train' else 'read_only')
            if capture is not None:
                capture.append((cell.detached(), q.flatten(0, 1)[None].detach().cpu(), read.detach().cpu()))
            read = self.readout[h](read.reshape(len(k), 4, -1))
            scores.append((F.normalize(read, dim=-1) @ F.normalize(v, dim=-1).T).mean(1)*10)
            alignment.append((1-(q*k[:, None]).sum(-1)).mean())
            similarity = k @ k.T
            separation.append(similarity.masked_fill(torch.eye(len(k), dtype=torch.bool,
                                                               device=k.device), 0).relu().mean())
            retention.append(F.mse_loss(shard.recall(k[None], cell, mode)[0], v))
        return torch.stack(scores).mean(0), [torch.stack(x).mean() for x in
                                             (alignment, separation, retention)]


def dataset(config):
    templates = [
        ('The assigned code for {} is', 'What is the code of {}?', 'The value is {}.'),
        ('Record the key for {}.', 'Which key belongs to {}?', 'The key reads {}.'),
        ('Identifier entry for {}.', 'Retrieve the identifier assigned to {}.', 'Identifier: {}.'),
    ]
    result = {}
    for split, (key, query, value) in zip(('train', 'validation', 'test'), templates):
        rng = random.Random(f"{config['seed']}:{split}")
        result[split] = []
        for i in range(config[split+'_episodes']):
            names = [f'{split}-{i}-{j}-{rng.getrandbits(32):08x}' for j in range(config['candidates'])]
            result[split].append([(key.format(n), query.format(n),
                                   value.format(f'{split}-{rng.getrandbits(48):012x}')) for n in names])
    return result


@torch.no_grad()
def encode(adapter, tokenizer, rows):
    attention = adapter.model.model.layers[3].self_attn
    result = []
    lengths = []
    def capture(module, args, kwargs):
        x = kwargs['hidden_states'][:, -1]
        q = module.q_proj(x).reshape(8, 512)[:, :256]
        k = module.k_proj(x).reshape(2, 256)
        v = module.v_proj(x).reshape(2, 256)
        result.append((module.k_norm(k).float(), module.q_norm(q).float(), v.float()))
    hook = attention.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for field in range(3):
            for row in rows:
                ids = tokenizer.encode(row[field], add_special_tokens=False)
                lengths.append(len(ids))
                adapter.encode(torch.tensor([ids], device='cuda'))
    finally:
        hook.remove()
    n = len(rows)
    features = tuple(torch.stack([item[i] for item in result[i*n:(i+1)*n]]) for i in range(3))
    return features, lengths


@torch.no_grad()
def baseline(adapter, tokenizer, rows):
    """Rank the same candidate continuations using native Qwen token likelihood."""
    candidates = [tokenizer.encode(v, add_special_tokens=False) for _, _, v in rows]
    scores = []
    device = next(adapter.model.parameters()).device
    for _, question, _ in rows:
        prefix = tokenizer.encode(question, add_special_tokens=False)
        ranking = []
        for candidate in candidates:
            ids = torch.tensor([prefix + candidate], device=device)
            hidden, _ = adapter.encode(ids[:, :-1])
            logits = adapter.logits(hidden[:, len(prefix)-1:]).float()
            ranking.append(-float(F.cross_entropy(logits[0], ids[0, len(prefix):])))
        scores.append(ranking)
    return scores


def export_persistence(model, features, output, identity, rows):
    sys.path.insert(0, str(BASE.parents[2] / 'research/projects/cellular-language-models/experiments'
                          / 'epclm-006-cross-session-cell-portability'))
    from cell_artifact import export_cell

    with torch.no_grad():
        model(features, intervention='fork')
    path = output / 'portable'
    path.mkdir()
    captured = []
    with torch.no_grad():
        model(features, capture=captured)
    source = str(uuid.uuid4())
    for index, (cell, _, _) in enumerate(captured):
        export_cell(cell, path / f'shard-{index}', identity, source)
    torch.save(dict(weights=dict(model.state_dict()), identity=identity,
                    reads=[(query, expected) for _, query, expected in captured]), path / 'fixture.pt')
    save(path / 'packets.json', rows)


def main(config_path, output, resume=None):
    from transformers import AutoTokenizer

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    save(output/'config.json', config)
    report = dict(status='MINICELLS_MVP_INCOMPLETE', outcome='INCONCLUSIVE', gates={})
    try:
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
            raise RuntimeError('Tesla T4 required')
        torch.manual_seed(config['seed'])
        torch.cuda.set_per_process_memory_fraction(
            min(1., config['max_gpu_memory_gib']*2**30/torch.cuda.get_device_properties(0).total_memory))
        data = dataset(config)
        if len(data['test'])*config['candidates'] > config['max_eval_examples']:
            raise ValueError('Test exceeds the frozen example budget')
        save(output/'dataset_manifest.json', {k: dict(episodes=len(v), sha256=digest(v))
                                              for k,v in data.items()})
        adapter = Qwen35Adapter.load()
        adapter.model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        save(output/'environment.json', dict(model=adapter.identity, gpu=torch.cuda.get_device_name(),
             torch=str(torch.__version__), tokenizer_sha256=hashlib.sha256(
                 tokenizer.backend_tokenizer.to_str().encode()).hexdigest()))
        versions = [p._version for p in adapter.model.parameters()]
        model = Memory().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        labels = torch.arange(config['candidates'], device='cuda')
        cache = {}
        history = []
        train_tokens = 0
        deadline = started + config['wall_seconds'] - config['publish_reserve']
        checkpoint = output / 'checkpoint.pt'
        identity = dict(config=config, dataset_sha256=digest(data), backbone=adapter.identity,
                        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest())
        if resume:
            restored = torch.load(resume, weights_only=True, map_location='cpu')
            if restored['identity'] != identity:
                raise ValueError('Checkpoint identity mismatch')
            model.load_state_dict(restored['model'])
            optimizer.load_state_dict(restored['optimizer'])
            torch.set_rng_state(restored['rng'])
            torch.cuda.set_rng_state(restored['cuda_rng'])
            history, train_tokens = restored['history'], restored['train_tokens']

        def checkpoint_training():
            temp = checkpoint.with_suffix('.tmp')
            torch.save(dict(model=dict(model.state_dict()), optimizer=optimizer.state_dict(),
                            step=len(history), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                            identity=identity, history=history, train_tokens=train_tokens), temp)
            temp.replace(checkpoint)

        first_step = len(history)
        warmup_started = time.monotonic()
        for i in range(first_step, config['max_steps']):
            if time.monotonic() >= min(started+6000, deadline):
                break
            index = i % len(data['train'])
            if index not in cache:
                cache[index] = encode(adapter, tokenizer, data['train'][index])
            features, lengths = cache[index]
            if train_tokens + sum(lengths) > config['max_train_tokens']:
                break
            train_tokens += sum(lengths)
            optimizer.zero_grad(set_to_none=True)
            scores, (address, separation, interference) = model(features, mode='meta_train')
            weights = config['loss_weights']
            loss = (weights['retrieval']*F.cross_entropy(scores, labels) + weights['address']*address
                    + weights['separation']*separation + weights['interference']*interference)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            history.append(float(loss.detach()))
            if i == first_step:
                torch.cuda.synchronize()
                seconds = time.monotonic() - warmup_started
                available = max(0., min(started + 6000, deadline) - time.monotonic())
                report['warmup'] = dict(step_seconds=seconds, input_tokens=sum(lengths),
                                        estimated_additional_steps=int(available / max(seconds, .001)),
                                        peak_cuda_bytes=torch.cuda.max_memory_allocated())
                print(report['warmup'], flush=True)
                checkpoint_training()
            if (i+1)%8 == 0:
                print(f'Training update {i+1}: {history[-1]:.4f}', flush=True)
                save(output/'training.json', history)
                checkpoint_training()
        if not history:
            raise RuntimeError('No committed training updates')
        report['training_tokens'] = train_tokens
        report['training_steps'] = len(history)
        save(output/'training.json', history)
        checkpoint_training()
        with checkpoint.open('rb') as stream:
            report['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        results = []
        validation_scores = []
        with torch.no_grad():
            for i in range(len(data['validation'])):
                if time.monotonic() >= deadline - 120:
                    break
                a, _ = encode(adapter, tokenizer, data['validation'][i])
                b, _ = encode(adapter, tokenizer, data['validation'][(i+1)%len(data['validation'])])
                unknown, _ = model(a, query_override=b[1])
                validation_scores.extend(unknown.amax(1).tolist())
        cutoff = (sorted(validation_scores)[int(.95*len(validation_scores))]
                  if validation_scores else None)
        report['rejection_cutoff'] = cutoff
        persistence_features = None
        with torch.no_grad():
            for test_index, rows in enumerate(data['test']):
                if time.monotonic() >= deadline:
                    break
                features, lengths = encode(adapter, tokenizer, rows)
                ends = torch.tensor([lengths[i] + lengths[2*len(labels)+i] for i in range(len(labels))], device='cuda').cumsum(0)-1
                if persistence_features is None:
                    persistence_features = features
                row = {}
                if cutoff is not None:
                    alternate, _ = encode(adapter, tokenizer,
                                          data['test'][(test_index+1)%len(data['test'])])
                    unknown, _ = model(features, query_override=alternate[1])
                    row['false_recall'] = float((unknown.amax(1) > cutoff).float().mean())
                for exact in (True, False):
                    for control in ('correct', 'empty', 'donor'):
                        scores, _ = model(features, exact=exact, control=control)
                        row[f'{exact}/{control}'] = scores.argmax(1).eq(labels).float().mean().item()
                for width in (16, 64, 512):
                    scores, _ = model(features, exact=True, width=width, positions=ends)
                    row[f'width_{width}'] = float(scores.argmax(1).eq(labels).float().mean())
                scores, _ = model(features, exact=True, intervention='latest')
                prediction = scores.argmax(1)
                row['latest_correct'] = int(prediction[0]) == 1
                row['stale_value'] = int(prediction[0]) == 0
                row['unrelated_recall'] = float(prediction[1:].eq(labels[1:]).float().mean())
                results.append(row)
                save(output/'retrieval.json', results)
                print(f"Paired test episodes: {len(results)}/{config['test_episodes']}", flush=True)
        report['gates']['tnt'] = 'SKIPPED_BUDGET'
        if len(results) == config['test_episodes'] and time.monotonic() < deadline - 600:
            tnt_rows = []
            with torch.no_grad():
                for i, rows in enumerate(data['test']):
                    if time.monotonic() >= deadline - 300:
                        break
                    features, _ = encode(adapter, tokenizer, rows)
                    scores, _ = model(features, tnt=True)
                    tnt_rows.append(dict(episode=i, projected=float(scores.argmax(1).eq(labels).float().mean()),
                                         direct=results[i]['False/correct']))
                save(output/'tnt.json', dict(scope='frozen read-time span ablation', rows=tnt_rows))
            if len(tnt_rows) == len(results):
                report['gates']['tnt'] = 'INCONCLUSIVE'
        if versions != [p._version for p in adapter.model.parameters()]:
            raise RuntimeError('Frozen backbone changed')
        if len(results) == config['test_episodes']:
            report['recall'] = {k: sum(r[k] for r in results)/len(results) for k in results[0]}
            report['gates']['addressing_accuracy'] = (
                'PASS' if report['recall']['True/correct'] >= config['thresholds']['addressing'] else 'FAIL')
            for key in ('latest_correct', 'stale_value', 'unrelated_recall'):
                report.setdefault('contradiction', {})[key] = report['recall'][key]
            report['paired'] = {
                f'{exact}/{control}': paired([row[f'{exact}/correct'] - row[f'{exact}/{control}']
                                             for row in results])
                for exact in (True, False) for control in ('empty', 'donor')
            }
            gains = [report['paired'][f'False/{control}'] for control in ('empty', 'donor')]
            report['gates']['generalization_gate'] = 'PASS' if (
                report['recall']['False/correct'] >= config['thresholds']['paraphrase']
                and all(gain['mean'] >= config['thresholds']['gain'] and gain['ci95'][0] > 0
                        for gain in gains)) else 'FAIL'
            if cutoff is not None:
                report['gates']['false_recall'] = ('PASS' if
                    report['recall']['false_recall'] <= config['thresholds']['false_recall']
                    else 'FAIL')
            report['gates']['contradiction'] = 'PASS' if (
                report['contradiction']['latest_correct'] >= .90 and
                report['contradiction']['stale_value'] <= .05) else 'FAIL'
            report['gates']['fusion'] = (
                'NOT_IMPLEMENTED' if report['gates']['generalization_gate'] == 'PASS'
                and report['gates']['addressing_accuracy'] == 'PASS' else 'SKIPPED_GATE')
            report['outcome'] = 'FAILURE' if report['gates']['addressing_accuracy']=='FAIL' else 'INCONCLUSIVE'
        else:
            report['gates']['retrieval'] = 'SKIPPED_BUDGET'
        if persistence_features is not None and time.monotonic() < deadline - 900:
            native_scores = baseline(adapter, tokenizer, data['test'][0])
            hits = sum(max(range(len(scores)), key=scores.__getitem__) == i
                       for i, scores in enumerate(native_scores))
            save(output/'baseline.json', dict(correct=hits, queries=len(native_scores),
                 scores=native_scores, scope='one episode; mean candidate token log probability'))
            report['gates']['baseline'] = 'INCONCLUSIVE'
        else:
            report['gates']['baseline'] = 'SKIPPED_BUDGET'
        if persistence_features is not None and time.monotonic() < deadline - 120:
            with torch.no_grad():
                before, _ = model(persistence_features)
                old = float(before.argmax(1).eq(labels).float().mean())
                banks = [[], [], []]
                tokens = 0
                ladder = []
                for episode in data['validation']:
                    if time.monotonic() >= deadline - 300:
                        break
                    extra, lengths = encode(adapter, tokenizer, episode)
                    for bank, features in zip(banks, extra):
                        bank.extend(features.split(1))
                    tokens += sum(lengths[:len(labels)]) + sum(lengths[2*len(labels):])
                    if tokens >= (1024, 4096, 16384)[len(ladder)]:
                        after, _ = model(persistence_features, distractors=tuple(torch.cat(bank) for bank in banks))
                        ladder.append(dict(target_tokens=(1024, 4096, 16384)[len(ladder)],
                                           actual_write_tokens=tokens,
                                           recall=float(after.argmax(1).eq(labels).float().mean())))
                        save(output/'distractor.json', dict(before=old, ladder=ladder,
                             scope='single held-out episode'))
                        if len(ladder) == 3:
                            break
                report['gates']['distractor'] = 'INCONCLUSIVE'
            trajectory = {}
            with torch.no_grad():
                ends = torch.tensor([len(tokenizer.encode(k, add_special_tokens=False))
                                     + len(tokenizer.encode(v, add_special_tokens=False))
                                     for k, _, v in data['test'][0]], device='cuda').cumsum(0)-1
                for width in (1, 16, 64, 512):
                    records = []
                    model(persistence_features, exact=True, width=width,
                          positions=ends if width != 1 else None, trajectory=records)
                    trajectory[str(width)] = records
            save(output/'interference.json', trajectory)
            report['gates']['interference'] = 'INCONCLUSIVE'
        else:
            report['gates']['distractor'] = 'SKIPPED_BUDGET'
        if persistence_features is not None and deadline - time.monotonic() > 90:
            try:
                export_persistence(model, persistence_features, output,
                    {'checkpoint_sha256': report['checkpoint_sha256'], 'schema': 'stage7-v1',
                     'model': adapter.identity}, data['test'][0])
                report['gates']['artifact_roundtrip'] = 'AWAITING_FRESH_PROCESS'
                report['gates']['fork_isolation'] = 'PASS'
            except Exception as error:
                report['gates']['artifact_roundtrip'] = 'ERROR'
                report['persistence_error'] = str(error)
        else:
            report['gates']['artifact_roundtrip'] = 'SKIPPED_BUDGET'
    except Exception as error:
        report.update(error=str(error), outcome='FAILURE')
        raise
    finally:
        for name in ('baseline', 'false_recall', 'generalization_gate', 'distractor', 'contradiction',
                     'interference', 'isolation', 'persistence', 'tnt', 'fusion'):
            report['gates'].setdefault(name, 'NOT_IMPLEMENTED')
        report.update(seconds=time.monotonic()-started, config_sha256=digest(config))
        save(output/'smoke.json', report)
        (output/'FINAL_REPORT.md').write_text('# Stage 7 development evidence\n\n' +
                                            '```json\n'+json.dumps(report, indent=2)+'\n```\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'experiment_config.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    main(args.config, args.output, args.resume)
