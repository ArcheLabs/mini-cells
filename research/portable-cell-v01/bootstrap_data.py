"""Materialize the upstream bootstrap stream for local-only training."""

import hashlib
import json
import random
import signal
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import IterableDataset


FIELDS = ('input_ids', 'attention_mask', 'labels')
READ_TIMEOUT_SECONDS = 180


def source_at(index, seed):
    return ('FineWeb-Edu' if (index + seed) % 2 else 'LongAlign')


def read_sample(stream, context):
    def timed_out(_signum, _frame):
        raise RuntimeError(f'Data preparation stalled for {READ_TIMEOUT_SECONDS}s while {context}')

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(READ_TIMEOUT_SECONDS)
    try:
        return next(stream)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def prepare(output, make_dataset, config):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    started = time.monotonic()
    count = config.total_steps
    rows = []
    seed = config.longalign_shuffle_seed
    # Delay iterator creation until the first timed read.
    stream = make_dataset_iterator(make_dataset, config)
    for index in range(count):
        source = source_at(index, seed)
        context = f'reading batch {index + 1}/{count} ({source})'
        print(f'Data preparation: {context}', flush=True)
        sample = read_sample(stream, context)
        rows.append({field: sample[field].clone() for field in FIELDS})
        print(f'Data preparation: saved in memory {index + 1}/{count} ({source})', flush=True)
    arrays = {field: torch.stack([row[field] for row in rows]) for field in FIELDS}
    path = output / 'bootstrap-data.safetensors'
    temporary = path.with_suffix('.tmp')
    save_file(arrays, str(temporary))
    temporary.replace(path)
    manifest = {
        'dataset': config.dataset_name,
        'shuffle_seed': seed,
        'source_order': [source_at(index, seed) for index in range(count)],
        'steps': count,
        'sequence_length': config.sequence_length,
        'input_tokens': count * config.sequence_length,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'bytes': path.stat().st_size,
        'preparation_seconds': time.monotonic() - started,
    }
    (output / 'bootstrap-data.json').write_text(json.dumps(manifest, indent=2))


def make_dataset_iterator(make_dataset, config):
    yield from make_dataset(config)


class LocalBootstrap(IterableDataset):
    def __init__(self, output, config):
        path = output / 'bootstrap-data.safetensors'
        manifest = json.loads((output / 'bootstrap-data.json').read_text())
        expected = (config.total_steps, config.sequence_length)
        if (manifest['dataset'] != config.dataset_name
                or manifest['shuffle_seed'] != config.longalign_shuffle_seed
                or manifest['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError('Local bootstrap data identity mismatch')
        self.tensors = load_file(str(path))
        if set(self.tensors) != set(FIELDS) or any(
            tensor.shape != expected or tensor.dtype != torch.int64 for tensor in self.tensors.values()
        ):
            raise ValueError('Local bootstrap tensor layout mismatch')

    def __iter__(self):
        for index in range(self.tensors['input_ids'].shape[0]):
            yield {name: tensor[index] for name, tensor in self.tensors.items()}
