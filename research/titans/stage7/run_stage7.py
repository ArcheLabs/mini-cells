"""Clean-slate episodic training on native pre-RoPE Qwen head features."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import subprocess
import uuid
import time

import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'qwen-mvp'))
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

    def forward(self, episode, *, exact=False, control='correct', width=1, mode='online_adapt', capture=None, positions=None, intervention=None):
        keys, queries, values = episode
        assert keys.shape == values.shape and keys.shape[1:] == (2, 256)
        assert queries.shape == (keys.shape[0], 8, 256)
        scores, alignment, separation, retention = [], [], [], []
        for h, shard in enumerate(self.shards):
            k = F.normalize(self.address[h](keys[:, h]), dim=-1)
            q = k[:, None].expand(-1, 4, -1) if exact else F.normalize(
                self.address[h](queries[:, h*4:h*4+4]), dim=-1)
            v = self.value[h](values[:, h])
            cell = shard.create(mode=mode)
            if control != 'empty':
                targets = v if control == 'correct' else v.roll(1, 0)
                groups = (positions // width).unique_consecutive(return_counts=True)[1].tolist() if positions is not None else [1] * len(k)
                for x, y in zip(k.split(groups), targets.split(groups)):
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


def verify_persistence(model, features, output, identity, timeout):
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
    subprocess.run([sys.executable, str(BASE / 'persistence.py'), str(path)],
                   check=True, timeout=timeout)
    proof = json.loads((path / 'restore.json').read_text())
    save(output / 'persistence.json', proof)


def main(config_path, output):
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
        cache = []
        history = []
        train_tokens = 0
        deadline = started + config['wall_seconds'] - config['publish_reserve']
        for i in range(config['max_steps']):
            if time.monotonic() >= min(started+6000, deadline):
                break
            if i < len(data['train']):
                cache.append(encode(adapter, tokenizer, data['train'][i]))
            features, lengths = cache[i % len(cache)]
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
            if (i+1)%8 == 0:
                print(f'Training update {i+1}: {history[-1]:.4f}', flush=True)
                save(output/'training.json', history)
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                step=i+1, rng=torch.get_rng_state(), config=config),
                           output/'checkpoint.pt')
        if not history:
            raise RuntimeError('No committed training updates')
        report['training_tokens'] = train_tokens
        report['training_steps'] = len(history)
        save(output/'training.json', history)
        checkpoint = output/'checkpoint.pt'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        step=len(history), rng=torch.get_rng_state(), config=config), checkpoint)
        with checkpoint.open('rb') as stream:
            report['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        results = []
        persistence_features = None
        with torch.no_grad():
            for rows in data['test']:
                if time.monotonic() >= deadline:
                    break
                features, lengths = encode(adapter, tokenizer, rows)
                ends = torch.tensor([lengths[i] + lengths[2*len(labels)+i] for i in range(len(labels))], device='cuda').cumsum(0)-1
                if persistence_features is None:
                    persistence_features = features
                row = {}
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
        if versions != [p._version for p in adapter.model.parameters()]:
            raise RuntimeError('Frozen backbone changed')
        if len(results) == config['test_episodes']:
            report['recall'] = {k: sum(r[k] for r in results)/len(results) for k in results[0]}
            report['gates']['addressing_accuracy'] = (
                'PASS' if report['recall']['True/correct'] >= config['thresholds']['addressing'] else 'FAIL')
            for key in ('latest_correct', 'stale_value', 'unrelated_recall'):
                report.setdefault('contradiction', {})[key] = report['recall'][key]
            gains = [report['recall']['False/correct'] - report['recall'][f'False/{control}']
                     for control in ('empty', 'donor')]
            report['gates']['generalization_gate'] = 'PASS' if (
                report['recall']['False/correct'] >= config['thresholds']['paraphrase']
                and min(gains) >= config['thresholds']['gain']) else 'FAIL'
            report['outcome'] = 'FAILURE' if report['gates']['addressing_accuracy']=='FAIL' else 'INCONCLUSIVE'
        else:
            report['gates']['retrieval'] = 'SKIPPED_BUDGET'
        if persistence_features is not None and deadline - time.monotonic() > 90:
            try:
                verify_persistence(model, persistence_features, output,
                    {'checkpoint_sha256': report['checkpoint_sha256'], 'schema': 'stage7-v1'},
                    timeout=min(90, deadline - time.monotonic()))
                report['gates']['artifact_roundtrip'] = 'PASS'
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
            report['gates'].setdefault(name, 'PENDING_IMPLEMENTATION')
        report.update(seconds=time.monotonic()-started, config_sha256=digest(config))
        save(output/'smoke.json', report)
        (output/'FINAL_REPORT.md').write_text('# Stage 7 development evidence\n\n' +
                                            '```json\n'+json.dumps(report, indent=2)+'\n```\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE/'experiment_config.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.output)
