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
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.runners.models import LanguageModel


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare_data(config, output, deadline):
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    records = {}
    fingerprints = {}
    for split, count in [('train', config['train_stories']), ('validation', config['validation_stories'])]:
        dataset = load_dataset(config['dataset'], revision=config['dataset_revision'],
                               split=split, streaming=True)
        texts, digest = [], hashlib.sha256()
        for row in dataset:
            if time.time() >= deadline:
                raise TimeoutError('Data preparation exceeded the session budget')
            text = row['text']
            texts.append(text)
            encoded = text.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
            if len(texts) == count:
                break
        if len(texts) != count:
            raise RuntimeError(f'Insufficient {split} stories')
        records[split], fingerprints[split] = texts, digest.hexdigest()
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
    write_json(output / 'data.json', {
        'dataset': config['dataset'], 'revision': config['dataset_revision'],
        'story_counts': {key: len(value) for key, value in records.items()},
        'text_sha256': fingerprints,
        'token_counts': {key: len(value) for key, value in arrays.items()},
        'token_sha256': {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()},
        'tokenizer_sha256': hashlib.sha256((output / 'tokenizer.json').read_bytes()).hexdigest(),
        'tokenizer_vocab_size': tokenizer.get_vocab_size(),
        'sampling': 'same seeded random windows with replacement for both models; EOS between stories',
        'tokenizer_training_split': 'train only',
    })
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
            counts = router.counts.cpu().tolist()
            routing.append({'layer': layer, 'assignments': counts,
                'utilization': [n / sum(counts) for n in counts],
                'routing_entropy_nats': float(router.entropy_sum) / router.token_count,
                'active_cells_per_token': config['top_k'],
                'path_length': 1,
                'path_length_definition': 'one parallel selection per FFN; not a sequential cell path'})
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


def train_one(kind, config, data, output, session_deadline):
    started = time.time()
    deadline = min(started + config['model_budget_seconds'], session_deadline - 120)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    device = torch.device('cuda:0')
    model = LanguageModel(config, kind).to(device)
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
    parser.add_argument('--config', default='experiments/configs/routing/experiment-001.json')
    parser.add_argument('--output', default='experiment-output')
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
    output.mkdir(parents=True, exist_ok=False)
    session_started = float(os.environ.get('EXPERIMENT_STARTED_AT', time.time()))
    deadline = session_started + config['session_budget_seconds']
    write_json(output / 'config.json', config)
    provenance = {'experiment_id': config['experiment_id'], 'seed': config['seed'],
        'source_commit': os.environ.get('EXPERIMENT_COMMIT', 'local-unrecorded'),
        'torch': torch.__version__, 'python': platform.python_version(),
        'dependencies': {name: importlib.metadata.version(name)
                         for name in ('numpy', 'datasets', 'tokenizers')},
        'gpu': torch.cuda.get_device_name(0), 'precision': 'FP16 with GradScaler',
        'started_at_unix': session_started,
        'interpretation': 'capacity matched dense vs sparse FFN; compute is deliberately unequal; one development seed'}
    write_json(output / 'experiment.json', {**provenance, 'status': 'running'})
    results = []
    try:
        data = prepare_data(config, output, deadline - 120)
        for kind in config['models']:
            if time.time() >= deadline - 120:
                break
            results.append(train_one(kind, config, data, output, deadline))
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
