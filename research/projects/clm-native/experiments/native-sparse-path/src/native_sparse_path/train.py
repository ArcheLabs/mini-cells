"""Fixed-budget development training, final checkpoint; sealed data never loaded."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .generator import generate
from .reference_solver import solve
from .models import NativeModel, PoolingShortcut
from .evaluate import evaluate
from .reuse import (canonical_sha256, file_sha256, write_json, set_global_seed,
                    save_checkpoint, restore_checkpoint, environment_info, code_digest)


def dataset(config, task, split='train', count=None):
    instances = [generate(task, split, i, nodes=config['nodes'], candidates=config['c'],
                         depth=config['train_depth']) for i in range(count or config['train_examples'])]
    labels = [torch.tensor(solve(i, k=config['k'], active_limit=config['active_limit']),
                           dtype=torch.float32) for i in instances]
    manifest = {'split': split, 'task': task, 'generator_version': 1,
        'instances': [i.instance_id for i in instances],
        'targets_sha256': canonical_sha256([v.tolist() for v in labels]),
        'templates': sorted({i.template_id for i in instances})}
    return instances, labels, manifest


def train(config, arm, task, seed, output, source_commit, *, device='cpu', resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    torch.set_num_threads(config['cpu_threads'])
    rng = np.random.default_rng(seed)
    instances, labels, manifest = dataset(config, task)
    instances = [i.to(device) for i in instances]
    labels = [v.to(device) for v in labels]
    model = NativeModel(config, arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    identity = {'experiment_id': 'clm-native-bounded-sparse-path-v1',
        'source_commit': source_commit, 'config_sha256': canonical_sha256(config),
        'code_sha256': code_digest(),
        'data_sha256': canonical_sha256(manifest), 'data_seed': 100_000,
        'seed': seed, 'arm': arm, 'task': task, 'precision': 'float32', 'stage': 'SCREENING'}
    checkpoint = output / 'checkpoint.pt'
    if checkpoint.exists() and not resume:
        raise FileExistsError('existing run preserved; use resume or a new output directory')
    if resume and (output / 'metrics.json').is_file():
        result = json.loads((output / 'metrics.json').read_text())
        if result['identity'] != identity or result['checkpoint_sha256'] != file_sha256(checkpoint):
            raise ValueError('checkpoint source/config/protocol/data/seed/arm identity mismatch')
        return result
    curve, first, previous_elapsed = [], 0, 0.0
    if resume:
        first, extra = restore_checkpoint(checkpoint, model=model, optimizer=optimizer,
            scaler=None, expected_identity=identity, generator=rng, map_location=device)
        curve = extra['curve']
        previous_elapsed = extra['elapsed_seconds']
    write_json(output / 'data-manifest.json', manifest)
    started = time.perf_counter()
    last_cost = None
    try:
        for step in range(first + 1, config['steps'] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            indexes = rng.integers(0, len(instances), size=config['batch_size'])
            losses, entropies = [], []
            for index in indexes:
                prediction, last_cost, _, aux = model(instances[index])
                task_loss = torch.nn.functional.mse_loss(prediction, labels[index])
                loss = task_loss + config['stability_weight'] * aux['state_norm']
                loss = loss - config['entropy_weight'] * aux['entropy']
                losses.append(loss)
                entropies.append(aux['entropy'].detach())
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
            gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters()
                         if p.grad is not None}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % config['log_every'] == 0 or step == config['steps']:
                state_gradient = model._diagnostic_initial.grad.norm(dim=-1).detach()
                row = {'step': step, 'loss': float(loss.detach()),
                       'route_entropy': float(torch.stack(entropies).mean()),
                       'gradient_norms': gradients, 'gradient_coverage': len(gradients) / len(list(model.parameters())),
                       'initial_state_gradient_coverage_last_example': float((state_gradient > 0).float().mean()),
                       'initial_state_gradient_norms_last_example': state_gradient.cpu().tolist(),
                       'elapsed_seconds': previous_elapsed + time.perf_counter() - started}
                curve.append(row)
                print(json.dumps({'arm': arm, 'task': task, **{k: row[k] for k in ('step', 'loss', 'route_entropy')}}), flush=True)
            if step % config['checkpoint_every'] == 0 or step == config['steps']:
                save_checkpoint(checkpoint, model=model, optimizer=optimizer, step=step,
                    scaler=None, identity=identity, generator=rng,
                    extra={'curve': curve, 'elapsed_seconds': previous_elapsed + time.perf_counter() - started,
                           'parameter_count': sum(p.numel() for p in model.parameters()),
                           'precision': 'float32'})
        elapsed = previous_elapsed + time.perf_counter() - started
        result = evaluate(model, config, task)
        result.update({'identity': identity, 'status': 'SCREENING', 'scientific_decision': False,
            'train_steps': config['steps'], 'train_seconds': elapsed,
            'train_examples_seen': config['steps'] * config['batch_size'],
            'checkpoint_sha256': file_sha256(checkpoint), 'parameters_total': sum(p.numel() for p in model.parameters()),
            'precision': 'float32', 'training_cost_last_example': last_cost,
            'environment': environment_info(device)})
        write_json(output / 'learning-curve.json', curve)
        write_json(output / 'metrics.json', result)
        return result
    except Exception as exc:
        write_json(output / 'failure.json', {'identity': identity, 'status': 'IMPLEMENTATION_FAIL',
            'exception': type(exc).__name__, 'message': str(exc), 'scientific_decision': False})
        raise


def train_shortcut(config, task, seed, output, source_commit='unit-test'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    instances, labels, manifest = dataset(config, task)
    model = PoolingShortcut()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    rng = np.random.default_rng(seed)
    for _ in range(config['steps']):
        optimizer.zero_grad(set_to_none=True)
        indexes = rng.integers(len(instances), size=config['batch_size'])
        loss = torch.stack([torch.nn.functional.mse_loss(model(instances[i]), labels[i]) for i in indexes]).mean()
        loss.backward()
        optimizer.step()
    identity = {'experiment_id': config['study'], 'source_commit': source_commit,
                'config_sha256': canonical_sha256(config), 'code_sha256': code_digest(),
                'data_sha256': canonical_sha256(manifest), 'seed': seed, 'arm': 'POOL', 'task': task}
    save_checkpoint(output / 'pool-checkpoint.pt', model=model, optimizer=optimizer,
        step=config['steps'], scaler=None, identity=identity, generator=rng,
        extra={'parameter_count': sum(p.numel() for p in model.parameters()), 'precision': 'float32'})
    rows = []
    model.eval()
    dev, labels, _ = dataset(config, task, 'dev', count=config['eval_examples'])
    with torch.no_grad():
        for instance, label in zip(dev, labels):
            prediction = model(instance)
            rows.append({'id': instance.instance_id, 'mse': float((prediction - label).square().mean()),
                'prediction': prediction.tolist(), 'target': label.tolist(),
                'correct': bool((prediction - label).abs().max() <= config['accuracy_tolerance'])})
    result = {'task': task, 'arm': 'POOL', 'status': 'SCREENING', 'scientific_decision': False,
        'identity': identity, 'checkpoint_sha256': file_sha256(output / 'pool-checkpoint.pt'),
        'parameters_total': sum(p.numel() for p in model.parameters()), 'steps': config['steps'],
        'mse': float(np.mean([r['mse'] for r in rows])),
        'accuracy': float(np.mean([r['correct'] for r in rows])), 'examples': rows,
        'scope': 'global mean + root-pooling shortcut diagnostic, not a sparse performance competitor'}
    write_json(Path(output) / 'pooling-shortcut.json', result)
    return result
