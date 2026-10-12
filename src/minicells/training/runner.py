"""Four paired arms, independent workers on dual T4, bounded evidence publication."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.common.train import prepare_data, make_batch, evaluate, checkpoint, write_json
from minicells.models.routing_research import LanguageModel
from minicells.training.stages import stage
from minicells.training.evidence import assess
from minicells.utils.research_report import ROOT, CONFIG, PROTOCOL, report


def validate(config):
    protocol = json.loads(PROTOCOL.read_text())
    if (config['models'] != protocol['models'] or config['seeds'] != protocol['seeds']
            or config['train_tokens'] != protocol['train_tokens_per_arm']
            or config['seed'] not in config['seeds']):
        raise ValueError('Configuration differs from the registered experiment matrix')
    budget = config['data_preparation_budget_seconds'] + config['long_arm_budget_seconds'] + config['baseline_budget_seconds'] + 60 + config['publication_reserve_seconds']
    if budget > config['session_budget_seconds'] or config['session_budget_seconds'] > 3300:
        raise ValueError('Registered two-GPU schedule exceeds the session budget')
    if config['train_tokens'] % (4 * config['batch_size'] * config['sequence_length']):
        raise ValueError('Token budget must support exact 25/25/50 stage boundaries')
    with torch.device('meta'):
        capacities = [LanguageModel(config, kind).parameter_metrics()['params'] for kind in config['models']]
    if min(capacities) < 50_000_000 or max(capacities) > 150_000_000 or max(capacities) / min(capacities) - 1 > protocol['capacity_tolerance_fraction']:
        raise ValueError('Registered 50–150M capacity range or 1% matching tolerance violated')
    return dict(zip(config['models'], capacities))


def configure_attention():
    # Every arm, training and benchmark uses the same observable SDPA math backend.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def train_arm(kind, config, data, output, deadline):
    configure_attention()
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    model = LanguageModel(config, kind).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  betas=(0.9, 0.95), weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    tokens_per_step = config['batch_size'] * config['sequence_length']
    total = config['train_tokens'] // tokens_per_step
    generator = np.random.default_rng(config['seed'])
    windows = generator.integers(0, len(data['train']) - config['sequence_length'],
                                size=(total, config['batch_size']))
    validation = np.random.default_rng(config['seed'] + 1).integers(
        0, len(data['validation']) - config['sequence_length'],
        size=(config['validation_batches'], config['batch_size']))
    directory = output / kind
    directory.mkdir(exist_ok=False)
    started, train_seconds, completed = time.time(), 0.0, 0
    budget = config['long_arm_budget_seconds'] if kind in {'soft-cell', 'hard-cell'} else config['baseline_budget_seconds']
    arm_deadline = min(deadline, started + budget)
    training_deadline = arm_deadline - config['evaluation_reserve_seconds']
    metrics = {'model': kind, 'seed': config['seed'], 'status': 'incomplete',
               'physical_gpu_index': os.environ.get('CUDA_VISIBLE_DEVICES'), **model.parameter_metrics()}
    torch.cuda.reset_peak_memory_stats()
    curves = []
    try:
        with (directory / 'logs.jsonl').open('w', encoding='utf-8', buffering=1) as log:
            for step in range(1, total + 1):
                if time.time() >= training_deadline:
                    break
                phase, temperature = stage(step, total, kind)
                model.set_stage('soft' if phase == 'anneal' else phase, temperature)
                x, y = make_batch(data['train'], windows[step - 1], config['sequence_length'], torch.device('cuda'))
                progress = max(0, (step - config['warmup_steps']) / max(1, total - config['warmup_steps']))
                factor = min(step / config['warmup_steps'], 1) * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
                for group in optimizer.param_groups:
                    group['lr'] = config['learning_rate'] * factor
                torch.cuda.synchronize()
                tick = time.perf_counter()
                for retry in range(8):
                    if time.time() >= training_deadline:
                        raise TimeoutError('Training deadline reached')
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits, auxiliary = model(x)
                        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
                        loss = nll + config['balance_coefficient'] * auxiliary + config['compute_coefficient'] * model.compute_penalty()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    previous = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= previous:
                        break
                else:
                    raise RuntimeError('FP16 overflow persisted on the same batch')
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                train_seconds += seconds
                completed = step
                routers = model.routing_modules()
                paths = [router.last_path_length for router in routers if hasattr(router, 'last_path_length')]
                entropy = [router.last_entropy for router in routers if getattr(router, 'last_entropy', None) is not None]
                row = {'step': step, 'tokens': step * tokens_per_step, 'loss': float(nll.detach()),
                       'ppl': math.exp(min(float(nll.detach()), 80)), 'stage': phase, 'temperature': temperature,
                       'path_length': float(torch.stack(paths).mean()) if paths else (4 if kind == 'fixed-cell' else 1),
                       'entropy': float(torch.stack(entropy).mean()) if entropy else None,
                       'route_entropy': float(torch.stack(entropy).mean()) if entropy else None,
                       'lr': optimizer.param_groups[0]['lr'],
                       'active_cells': (float(torch.stack(paths).mean()) if kind == 'hard-cell' and phase == 'hard' and paths
                                        else 1 + config['experts'] * (config['max_hops'] - 1)
                                        if kind in {'soft-cell', 'hard-cell'} else config['top_k'] if kind == 'global-router'
                                        else 4 if kind == 'fixed-cell' else None),
                       'active_cells_scope': 'executed Cell visits per FFN; global parallel top-k selections',
                       'tokens_sec': tokens_per_step / seconds, 'gpu_memory': torch.cuda.memory_allocated(),
                       'path_scope': 'per FFN; soft expected depth, hard executed depth, global one top-k dispatch',
                       'throughput_scope': 'forward/backward/optimizer/retries only; transfer and logging excluded'}
                log.write(json.dumps(row, allow_nan=False) + '\n')
                if step % config['evaluate_every'] == 0 or step == total:
                    quality, _ = evaluate(model, data['validation'], validation, config, torch.device('cuda'), training_deadline)
                    curves.append({'step': step, 'train_tokens': step * tokens_per_step, **quality})
                    write_json(directory / 'learning_curve.json', curves)
                    print(json.dumps({'model': kind, 'step': step, 'stage': phase, **quality}), flush=True)
        checkpoint(model, optimizer, scaler, completed, config, directory / 'checkpoint.pt')
        metrics.update(steps=completed, train_tokens=completed * tokens_per_step,
                       train_seconds=train_seconds, tokens_per_sec=completed * tokens_per_step / max(train_seconds, 1e-9),
                       checkpoint_sha256=hashlib.sha256((directory / 'checkpoint.pt').read_bytes()).hexdigest())
        if completed == total:
            model.set_stage('hard' if kind == 'hard-cell' else 'soft', 0.1 if kind == 'hard-cell' else 1.0)
            metrics.update(evaluation=assess(model, data, config, directory, arm_deadline), status='completed')
    except Exception:
        (directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        metrics.update(steps=completed, train_tokens=completed * tokens_per_step)
        # Preserve the most recent state if time allows; incomplete never passes gates.
        if not (directory / 'checkpoint.pt').exists() and time.time() < arm_deadline:
            checkpoint(model, optimizer, scaler, completed, config, directory / 'checkpoint.pt')
    finally:
        metrics['elapsed_seconds'] = time.time() - started
        write_json(directory / 'metrics.json', metrics)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def schedule(config, cache, output, deadline):
    # Each GPU owns one long arm + one baseline; alternate physical placement by seed.
    first = config['seeds'].index(config['seed']) % 2
    queues = {first: ['soft-cell', 'fixed-cell'], 1 - first: ['hard-cell', 'global-router']}
    running, exits, samples = {}, {}, []
    sample_at = 0
    try:
        while time.time() < deadline and (any(queues.values()) or running):
            for gpu, queue in queues.items():
                if gpu not in running and queue:
                    kind = queue.pop(0)
                    log = (output / (kind + '-worker.log')).open('w', encoding='utf-8')
                    process = subprocess.Popen([sys.executable, '-m', 'minicells.training.runner',
                        '--worker', kind, '--seed', str(config['seed']), '--data', str(cache),
                        '--output', str(output), '--deadline', str(deadline)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
                    running[gpu] = kind, process, log
            for gpu, (kind, process, log) in list(running.items()):
                if process.poll() is not None:
                    exits[kind] = process.returncode
                    log.close()
                    del running[gpu]
            if time.time() >= sample_at:
                # Stream compact status into the Notebook while workers retain full JSONL.
                for kind, _, _ in running.values():
                    path = output / kind / 'logs.jsonl'
                    if path.exists() and path.stat().st_size:
                        with path.open('rb') as stream:
                            stream.seek(max(0, path.stat().st_size - 8192))
                            lines = stream.read().decode('utf-8', errors='ignore').splitlines()
                        try:
                            row = json.loads(lines[-1])
                            print(json.dumps({'model': kind, **{key: row.get(key) for key in
                                ('step', 'tokens', 'loss', 'ppl', 'lr', 'tokens_sec', 'gpu_memory', 'stage')}}), flush=True)
                        except (IndexError, ValueError):
                            pass
                try:
                    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'], text=True, timeout=5)
                    samples.append({'time': time.time(), 'active': [value[0] for value in running.values()],
                                    'gpu_rows': rows.strip().splitlines()})
                except (OSError, subprocess.SubprocessError):
                    pass
                sample_at = time.time() + 10
            time.sleep(1)
    finally:
        for kind, process, log in running.values():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exits[kind] = process.returncode
            log.close()
        write_json(output / 'gpu-utilization.json', {'samples': samples})
    for kind in config['models']:
        directory = output / kind
        directory.mkdir(exist_ok=True)
        path = directory / 'metrics.json'
        if not path.exists() or exits.get(kind) != 0:
            write_json(path, {'model': kind, 'status': 'failed', 'worker_exit_code': exits.get(kind)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int, choices=[42, 123, 456])
    parser.add_argument('--output', default='/kaggle/working/results-003')
    parser.add_argument('--worker', choices=['soft-cell', 'hard-cell', 'fixed-cell', 'global-router'])
    parser.add_argument('--data')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())  # YAML 1.2 accepts this exact JSON document.
    config['seed'] = args.seed
    capacities = validate(config)
    output = Path(args.output).resolve()
    if args.worker:
        data = {split: np.load(Path(args.data) / (split + '.npy'), mmap_mode='r') for split in ('train', 'validation')}
        train_arm(args.worker, config, data, output, args.deadline)
        return
    from minicells.utils.publication import preflight, publish
    source = os.environ.get('EXPERIMENT_COMMIT', '')
    prior = preflight(source, args.seed)
    if prior is not None:
        print(json.dumps({'status': 'already_published', 'seed': args.seed, 'skipped_gpu_work': True}))
        return
    output.mkdir(parents=True, exist_ok=False)
    started = float(os.environ.get('EXPERIMENT_STARTED_AT', time.time()))
    deadline = started + config['session_budget_seconds']
    provenance = {'source_commit': source, 'seed': args.seed, 'scientific_decision': False,
                  'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                  'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                  'torch': torch.__version__, 'python': sys.version,
                  'capacities': capacities, 'attention_backend': 'SDPA math in every arm and measurement',
                  'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'datasets', 'tokenizers')},
                  'execution': 'two independent T4 workers, one long + one short arm per GPU; no DDP'}
    write_json(output / 'config.json', config)
    write_json(output / 'experiment.json', {**provenance, 'status': 'running'})
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError('Registered protocol requires exactly two T4 GPUs')
        names = [torch.cuda.get_device_name(index) for index in range(2)]
        if any('T4' not in name for name in names):
            raise RuntimeError('Registered platform is dual T4')
        provenance['gpus'] = names
        data = prepare_data(config, output, started + config['data_preparation_budget_seconds'])
        with tempfile.TemporaryDirectory(prefix='routing003-data-') as cache:
            for split, array in data.items():
                np.save(Path(cache) / (split + '.npy'), array)
            del data
            schedule(config, cache, output, deadline - config['publication_reserve_seconds'])
    except Exception:
        (output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    decision = report(output)
    write_json(output / 'experiment.json', {**provenance, 'status': decision['status'],
               'elapsed_seconds': time.time() - started})
    try:
        publish(output, source, args.seed)
    except Exception:
        (output / 'publication-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(output / 'publication.json', {'status': 'failed', 'scope': 'evidence preserved in Kaggle outputs'})
        raise
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
