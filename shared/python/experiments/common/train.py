"""Budgeted Experiment 001 training and evidence generation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
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

from experiments.common.models import LanguageModel


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare_data(config, output, deadline, *, validation_prefix_count=0,
                 local_data_files=None, record_story_disjointness=False):
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    records = {}
    fingerprints = {}
    validation_prefix_hash = None
    validation_suffix_hash = None
    for split, count in [('train', config['train_stories']), ('validation', config['validation_stories'])]:
        dataset = (load_dataset('parquet', data_files=local_data_files, split=split, streaming=True)
                   if local_data_files else
                   load_dataset(config['dataset'], revision=config['dataset_revision'],
                                split=split, streaming=True))
        texts, digest = [], hashlib.sha256()
        prefix_digest = hashlib.sha256() if split == 'validation' and validation_prefix_count else None
        suffix_digest = hashlib.sha256() if split == 'validation' and validation_prefix_count else None
        for row in dataset:
            if time.time() >= deadline:
                raise TimeoutError('Data preparation exceeded the session budget')
            text = row['text']
            texts.append(text)
            encoded = text.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
            if prefix_digest is not None:
                (prefix_digest if len(texts) <= validation_prefix_count else suffix_digest).update(
                    len(encoded).to_bytes(8, 'little')
                )
                (prefix_digest if len(texts) <= validation_prefix_count else suffix_digest).update(encoded)
            if len(texts) == count:
                break
        if len(texts) != count:
            raise RuntimeError(f'Insufficient {split} stories')
        records[split], fingerprints[split] = texts, digest.hexdigest()
        if prefix_digest is not None:
            if validation_prefix_count >= len(texts):
                raise ValueError('validation_prefix_count must be smaller than validation story count')
            validation_prefix_hash = prefix_digest.hexdigest()
            validation_suffix_hash = suffix_digest.hexdigest()
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(records['train'][:config['tokenizer_stories']],
        trainers.BpeTrainer(vocab_size=config['vocab_size'], special_tokens=['<unk>', '<eos>'],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer.save(str(output / 'tokenizer.json'))
    eos = tokenizer.token_to_id('<eos>')
    arrays = {}
    for split, texts in records.items():
        ids = []
        for start in range(0, len(texts), 256):
            if time.time() >= deadline:
                raise TimeoutError('Tokenization exceeded the session budget')
            for encoding in tokenizer.encode_batch(texts[start:start + 256]):
                ids.extend(encoding.ids)
                ids.append(eos)
        arrays[split] = np.asarray(ids, dtype=np.int32)
        if len(ids) < config['sequence_length'] + 1:
            raise RuntimeError(f'Insufficient {split} tokens')
    data_record = {
        'dataset': config['dataset'], 'revision': config['dataset_revision'],
        'story_counts': {key: len(value) for key, value in records.items()},
        'text_sha256': fingerprints,
        'token_counts': {key: len(value) for key, value in arrays.items()},
        'token_sha256': {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()},
        'tokenizer_sha256': hashlib.sha256((output / 'tokenizer.json').read_bytes()).hexdigest(),
        'tokenizer_vocab_size': tokenizer.get_vocab_size(),
        'sampling': 'same seeded random windows with replacement for both models; EOS between stories',
        'tokenizer_training_split': 'train only',
    }
    if record_story_disjointness:
        if not 0 < validation_prefix_count < len(records['validation']):
            raise ValueError('story-disjointness verification requires the regression prefix boundary')
        train_set = set(records['train'])
        regression_set = set(records['validation'][:validation_prefix_count])
        holdout_set = set(records['validation'][validation_prefix_count:])
        data_record['exact_story_overlap_counts'] = {
            'train_vs_holdout': len(train_set & holdout_set),
            'regression_vs_holdout': len(regression_set & holdout_set),
            'train_vs_regression': len(train_set & regression_set),
        }
        if data_record['exact_story_overlap_counts']['train_vs_holdout'] or data_record['exact_story_overlap_counts']['regression_vs_holdout']:
            raise ValueError('the candidate independent holdout contains an already-used story')
    if validation_prefix_hash is not None:
        data_record['validation_prefix_story_count'] = int(validation_prefix_count)
        data_record['validation_prefix_text_sha256'] = validation_prefix_hash
        data_record['validation_suffix_text_sha256'] = validation_suffix_hash
    write_json(output / 'data.json', data_record)
    return arrays


def make_batch(tokens, starts, length, device):
    indices = starts[:, None] + np.arange(length + 1)[None, :]
    batch = torch.from_numpy(tokens[indices].astype(np.int64)).to(device)
    return batch[:, :-1], batch[:, 1:]


def evaluate(model, tokens, starts, config, device, deadline):
    model.eval()
    routers = model.routing_modules()
    for router in routers:
        router.reset_stats()
        router.collect_stats = True
    loss_sum, count = 0.0, 0
    try:
        with torch.no_grad():
            for positions in starts:
                if time.time() >= deadline:
                    raise TimeoutError('Evaluation exceeded the model budget')
                x, y = make_batch(tokens, positions, config['sequence_length'], device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
                loss_sum += float(loss) * y.numel()
                count += y.numel()
        routing = []
        for layer, router in enumerate(routers):
            if hasattr(router, 'stats'):
                routing.append({'layer': layer, **router.stats()})
                continue
            if router.counts is None or router.token_count != count:
                raise RuntimeError('Missing or incomplete routing evidence')
            counts = router.counts.cpu().tolist()
            routing.append({'layer': layer, 'assignments': counts,
                'utilization': [n / sum(counts) for n in counts],
                'routing_entropy_nats': float(router.entropy_sum) / router.token_count,
                'active_cells_per_token': config['top_k'],
                'path_length': 1,
                'path_length_definition': 'one parallel selection per FFN; not a sequential cell path'})
        if len(routing) != len(routers):
            raise RuntimeError('Routing layer count mismatch')
        nll = loss_sum / count
        return {'validation_loss': nll, 'perplexity': math.exp(min(nll, 80)),
                'validation_tokens': count}, routing
    finally:
        for router in routers:
            router.collect_stats = False
        model.train()


def checkpoint(model, optimizer, scaler, step, config, destination):
    temporary = destination.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(), 'step': step, 'config': config,
                'model_kind': model.kind}, temporary)
    temporary.replace(destination)


def train_one(kind, config, data, output, session_deadline, model_factory=LanguageModel):
    started = time.time()
    deadline = min(started + config['model_budget_seconds'], session_deadline - 120)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    device = torch.device('cuda:0')
    model = model_factory(config, kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  betas=(0.9, 0.95), weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    tokens_per_step = config['batch_size'] * config['sequence_length']
    steps = config['train_tokens'] // tokens_per_step
    generator = np.random.default_rng(config['seed'])
    train_starts = generator.integers(0, len(data['train']) - config['sequence_length'],
                                     size=(steps, config['batch_size']))
    validation_generator = np.random.default_rng(config['seed'] + 1)
    validation_starts = validation_generator.integers(
        0, len(data['validation']) - config['sequence_length'],
        size=(config['validation_batches'], config['batch_size']))
    directory = output / kind
    directory.mkdir()
    # Reserve time to save the latest checkpoint before the server's hard timeout.
    curves, routing = [], []
    train_seconds = 0.0
    completed_steps = 0
    torch.cuda.reset_peak_memory_stats()
    metrics = model.parameter_metrics()
    status = 'budget_exhausted'
    try:
        quality, routing = evaluate(model, data['validation'], validation_starts, config, device, deadline)
        curves.append({'step': 0, 'train_tokens': 0, **quality})
        for step in range(1, steps + 1):
            if time.time() >= deadline:
                break
            x, y = make_batch(data['train'], train_starts[step - 1], config['sequence_length'], device)
            progress = max(0.0, (step - config['warmup_steps']) / max(1, steps - config['warmup_steps']))
            factor = min(step / config['warmup_steps'], 1.0) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate'] * factor
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            for retry in range(8):
                if time.time() >= deadline:
                    raise TimeoutError('Training exceeded the model budget')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits, auxiliary = model(x)
                    nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
                    loss = nll + config['balance_coefficient'] * auxiliary
                    if hasattr(model, 'compute_penalty'):
                        loss = loss + config.get('compute_coefficient', 0.0) * model.compute_penalty()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    break
            else:
                raise RuntimeError('FP16 overflow persisted after 8 retries of the same batch')
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - tick
            completed_steps = step
            if step % config['evaluate_every'] == 0 or step == steps:
                quality, routing = evaluate(model, data['validation'], validation_starts, config, device, deadline)
                curves.append({'step': step, 'train_tokens': step * tokens_per_step, **quality})
                write_json(directory / 'learning_curve.json', curves)
                write_json(directory / 'routing_stats.json', routing)
                write_json(directory / ('routing-step-' + str(step) + '.json'), routing)
                checkpoint(model, optimizer, scaler, step, config, directory / 'checkpoint.pt')
                print(json.dumps({'model': kind, **curves[-1]}), flush=True)
        if completed_steps == steps:
            status = 'completed'
    except TimeoutError:
        status = 'budget_exhausted'
    except Exception:
        status = 'failed'
        (directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        checkpoint(model, optimizer, scaler, completed_steps, config, directory / 'checkpoint.pt')
        write_json(directory / 'learning_curve.json', curves)
        write_json(directory / 'routing_stats.json', routing)
        metrics.update({'model': kind, 'status': status, 'steps': completed_steps,
            'physical_gpu_index': os.environ.get('CUDA_VISIBLE_DEVICES', 'unassigned'),
            'train_tokens': completed_steps * tokens_per_step,
            'train_seconds': train_seconds,
            'tokens_per_sec': completed_steps * tokens_per_step / train_seconds if train_seconds else None,
            'throughput_scope': 'forward/backward/optimizer only; excludes data transfer, validation and saving',
            'elapsed_seconds': time.time() - started,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
            'last_validation': curves[-1] if curves else None})
        write_json(directory / 'metrics.json', metrics)
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='shared/python/experiments/configs/routing/experiment-001.json')
    parser.add_argument('--output', default='experiment-output')
    parser.add_argument('--worker', choices=['fixed', 'global-router'])
    parser.add_argument('--data')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config['train_tokens'] % (config['batch_size'] * config['sequence_length']):
        raise ValueError('Token budget must be an exact number of batches')
    if config['width'] % config['heads'] or not 1 <= config['top_k'] <= config['experts']:
        raise ValueError('Invalid model dimensions')
    if config['models'] != ['fixed', 'global-router']:
        raise ValueError('Experiment 001 requires the registered model pair')
    with torch.device('meta'):
        capacities = [LanguageModel(config, kind).parameter_metrics()['params']
                      for kind in config['models']]
    if not all(50000000 <= count <= 150000000 for count in capacities):
        raise ValueError('Experiment 001 must stay within the registered 50M-150M scale')
    if abs(capacities[1] / capacities[0] - 1) > 0.01:
        raise ValueError('Total model parameters must match within 1%')
    if not torch.cuda.is_available():
        raise RuntimeError('Experiment 001 requires CUDA')
    output = Path(args.output)
    if args.worker:
        data = {split: np.load(Path(args.data) / (split + '.npy'), mmap_mode='r')
                for split in ('train', 'validation')}
        train_one(args.worker, config, data, output, args.deadline)
        return
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Experiment 001 requires two GPUs for the one-hour parallel budget')
    output.mkdir(parents=True, exist_ok=False)
    session_started = float(os.environ.get('EXPERIMENT_STARTED_AT', time.time()))
    deadline = session_started + config['session_budget_seconds']
    write_json(output / 'config.json', config)
    provenance = {'experiment_id': config['experiment_id'], 'seed': config['seed'],
        'source_commit': os.environ.get('EXPERIMENT_COMMIT', 'local-unrecorded'),
        'torch': torch.__version__, 'python': platform.python_version(),
        'dependencies': {name: importlib.metadata.version(name)
                         for name in ('numpy', 'datasets', 'tokenizers')},
        'gpus': [torch.cuda.get_device_name(index) for index in (0, 1)],
        'execution': 'one independent model process per GPU, concurrently; not DDP',
        'precision': 'FP16 with GradScaler',
        'started_at_unix': session_started,
        'interpretation': 'capacity matched dense vs sparse FFN; compute is deliberately unequal; one development seed'}
    write_json(output / 'experiment.json', {**provenance, 'status': 'running'})
    results = []
    try:
        data = prepare_data(config, output, deadline - 120)
        with tempfile.TemporaryDirectory(prefix='minicells-data-') as cache:
            for split, tokens in data.items():
                np.save(Path(cache) / (split + '.npy'), tokens)
            del data
            workers = []
            try:
                for gpu, kind in enumerate(config['models']):
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                    log = (output / (kind + '-worker.log')).open('w', encoding='utf-8')
                    command = [sys.executable, '-m', 'experiments.runners.train',
                        '--config', str(Path(args.config).resolve()), '--output', str(output.resolve()),
                        '--worker', kind, '--data', cache, '--deadline', str(deadline)]
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    workers.append((kind, process, log))
                while any(process.poll() is None for _, process, _ in workers):
                    if time.time() >= deadline:
                        break
                    time.sleep(2)
            finally:
                for kind, process, log in workers:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    log.close()
                    report = output / kind / 'metrics.json'
                    if report.is_file():
                        metrics = json.loads(report.read_text(encoding='utf-8'))
                        if process.returncode != 0:
                            metrics['status'] = 'failed'
                        results.append(metrics)
                    else:
                        results.append({'model': kind, 'status': 'failed',
                            'reason': 'Worker exited or reached session deadline before saving metrics',
                            'worker_exit_code': process.returncode})
    except Exception:
        (output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    complete = len(results) == len(config['models']) and all(row['status'] == 'completed' for row in results)
    write_json(output / 'metrics.json', {'models': results, 'comparable_at_full_budget': complete})
    write_json(output / 'experiment.json', {**provenance,
        'status': 'completed' if complete else 'incomplete',
        'elapsed_seconds': time.time() - session_started})
    print(json.dumps({'status': 'completed' if complete else 'incomplete', 'models': results}), flush=True)


if __name__ == '__main__':
    main()
