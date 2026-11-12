"""Single-cell latent association study using the frozen M3 implementation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
M3 = BASE.parent / 'epclm-004-unified-memory-10m/m3'
sys.path.insert(0, str(M3))
from memory import NeuralMemory, state_bytes
from gates import gate5
from data import file_hash
from e0 import write


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def initialize(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return NeuralMemory(256, 32, 64).requires_grad_(False)


def vectors(n, generator):
    samples = torch.randn(n, 32, generator=generator)
    if n <= 32:
        return torch.linalg.qr(samples.T).Q.T.contiguous()
    return F.normalize(samples, dim=-1)


def cosine_bounds(vectors):
    pairwise = vectors @ vectors.T
    pairs = pairwise[torch.triu_indices(len(vectors), len(vectors), offset=1).unbind()]
    return [float(pairs.min()), float(pairs.max())]


def state_hash(cell):
    return {
        'fast': {k: tensor_hash(v) for k, v in cell.fast.items()},
        'momentum': {k: tensor_hash(v) for k, v in cell.momentum.items()},
        'step': cell.update_step,
        'metadata': cell.metadata,
    }


def read(memory, cell, queries):
    before = state_hash(cell)
    result = memory.recall(queries[None], cell, mode='read_only')[0]
    if before != state_hash(cell) or not torch.isfinite(result).all():
        raise RuntimeError('Read-only state or numerical invariant failed')
    return result


def episode(output, n, seed, config):
    generator = torch.Generator().manual_seed(seed + 1000 * n)
    keys = vectors(n, generator)
    candidates = vectors(n, generator) * math.sqrt(32)
    targets = torch.randperm(n, generator=generator)
    write_order = torch.randperm(n, generator=generator)
    query_order = torch.randperm(n, generator=generator)
    unknown = F.normalize(torch.randn(n, 32, generator=generator), dim=-1)
    values = candidates[targets]
    if torch.unique(keys, dim=0).shape[0] != n or (keys == values).all(-1).any():
        raise ValueError('Invalid association episode')
    memory = initialize(config['initializer_seed'])
    static = {k: tensor_hash(v) for k, v in memory.state_dict().items()}
    cell = memory.create(mode='online_adapt')
    initial = cell.detached()
    initial_read = read(memory, cell, keys[query_order])
    started = time.perf_counter()
    for _ in range(config['updates_per_episode']):
        # A whole episode is one chunk, matching the frozen positive control.
        # Its read sees only the state from completed earlier chunks.
        memory.recall(keys[write_order][None], cell, mode='read_only')
        cell, _ = memory.learn(keys[write_order][None], values[write_order][None], cell, 'online_adapt')
    acquisition_seconds = time.perf_counter() - started
    expected = read(memory, cell, keys[query_order])
    fast_delta = math.sqrt(sum(float((v.detach() - initial.fast[k]).square().sum()) for k, v in cell.fast.items()))
    momentum_norm = math.sqrt(sum(float(v.square().sum()) for v in cell.momentum.values()))
    update_count = cell.update_step
    if not (0 < fast_delta < math.inf and 0 < momentum_norm < math.inf):
        raise RuntimeError('Invalid dynamic-state change')
    if static != {k: tensor_hash(v) for k, v in memory.state_dict().items()} or update_count != config['updates_per_episode']:
        raise RuntimeError('Static parameter or update count invariant failed')
    compatibility = {'static_state': static, 'interface': 1, 'task': 'p4-latent-association-v1'}
    checkpoint = memory.checkpoint(cell, output, compatibility)
    saved_hash = state_hash(cell)
    del cell, initial, memory
    restored_memory = initialize(config['initializer_seed'])
    restored = restored_memory.restore(output, compatibility)
    started = time.perf_counter()
    actual = read(restored_memory, restored, keys[query_order])
    recall_seconds = time.perf_counter() - started
    restore_error = float((actual - expected).abs().max())
    if restore_error > config['restore_atol'] or state_hash(restored) != saved_hash:
        raise RuntimeError('Restored Cell differs')
    unknown_read = read(restored_memory, restored, unknown)

    def similarity(readout):
        return F.normalize(readout, dim=-1) @ F.normalize(candidates, dim=-1).T

    labels = targets[query_order]
    target_values = values[query_order]
    return {
        'episode_id': f'S{n}-{seed}', 'seed': seed, 'generator_seed': seed + 1000 * n,
        'generator': 'independent-gaussian-qr-v1', 'n': n, 'shape': list(keys.shape),
        'dtype': str(keys.dtype), 'key_sha256': tensor_hash(keys),
        'value_sha256': tensor_hash(values), 'unknown_key_sha256': tensor_hash(unknown),
        'write_order': write_order.tolist(), 'query_order': query_order.tolist(),
        'target_candidate_indices': targets.tolist(), 'key_cosine_bounds': cosine_bounds(keys),
        'value_cosine_bounds': cosine_bounds(F.normalize(values, dim=-1)),
        'random_recall_at_1': 1 / n,
        'initial_recall_at_1': float((similarity(initial_read).argmax(-1) == labels).float().mean()),
        'recall_at_1': float((similarity(actual).argmax(-1) == labels).float().mean()),
        'cosine_similarity': float(F.cosine_similarity(actual, target_values).mean()),
        'mse': float(F.mse_loss(actual, target_values)), 'updates': update_count,
        'fast_weight_delta': fast_delta, 'momentum_norm': momentum_norm,
        'restore_max_abs_error': restore_error, 'read_only_state_unchanged': True,
        'unknown_key_false_positive': float((similarity(unknown_read).amax(-1) >= config['unknown_cosine_threshold']).float().mean()),
        'unknown_cosine_threshold': config['unknown_cosine_threshold'],
        'acquisition_seconds': acquisition_seconds, 'recall_seconds': recall_seconds,
        'static_parameters': sum(p.numel() for p in restored_memory.parameters()),
        'static_tensor_sha256': static, **state_bytes(restored), 'checkpoint': checkpoint,
    }


def run(output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    config_path = BASE / 'configs/m1.json'
    config = json.loads(config_path.read_text())
    for name in ('memory.py', 'gates.py'):
        if file_hash(M3 / name) != file_hash(M3 / 'freeze/original-source' / name):
            raise RuntimeError(f'M3 frozen source mismatch: {name}')
    started = time.perf_counter()
    manifest = {
        'status': 'RUNNING', 'scope': 'synthetic latent development',
        'python': sys.version, 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
        'device': 'cpu', 'torch_threads': 1, 'config': config,
        'config_sha256': file_hash(config_path), 'runner_sha256': file_hash(__file__),
        'm3_source_sha256': {name: file_hash(M3 / name) for name in ('memory.py', 'gates.py')},
        'outer_optimizer_updates': 0, 'episodes': [],
    }
    try:
        manifest['positive_control'] = gate5(output / 'positive-control')
        for n in config['sizes']:
            for seed in config['seeds']:
                result = episode(output / f'S{n}-{seed}', n, seed, config)
                manifest['episodes'].append(result)
        manifest['status'] = 'M1_COMPLETE'
    except BaseException as error:
        manifest.update(status='BLOCKED', error=repr(error))
        raise
    finally:
        manifest['wall_seconds'] = time.perf_counter() - started
        write(output / 'run-manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
