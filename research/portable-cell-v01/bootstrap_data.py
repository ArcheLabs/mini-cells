"""Prepare resumable, source-indexed batches for offline bootstrap training."""

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import signal
import tempfile
import time

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import IterableDataset


FIELDS = ('input_ids', 'attention_mask', 'labels')
DATASETS = {'LongAlign': 'zai-org/LongAlign-10k', 'FineWeb-Edu': 'HuggingFaceFW/fineweb-edu'}
LONGALIGN_REVISION = '12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc'
NETWORK_SECONDS = 45
TOKENIZE_SECONDS = 8
SOURCE_SECONDS = 180
MAX_RAW_BYTES = 32 * 1024


class PreparationLimit(RuntimeError):
    pass


@contextmanager
def budget(seconds, phase, stats):
    def expired(_signum, _frame):
        raise PreparationLimit(f'{phase} exceeded {seconds:g}s')

    previous = signal.signal(signal.SIGALRM, expired)
    start = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        stats[f'{phase}_seconds'] += time.monotonic() - start


def stats_for_source():
    return Counter(dict(cache_hits=0, cache_misses=0, raw_cache_hits=0, raw_cache_misses=0,
                        network_seconds=0, tokenize_seconds=0,
                        skipped=0, truncated=0, tokenize_timeouts=0))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_at(index, seed):
    return 'FineWeb-Edu' if (index + seed) % 2 else 'LongAlign'


def guarded_longalign(dataset, item, stats):
    messages = dataset._messages_from_item(item)
    assistant_index = next((i for i, msg in enumerate(messages) if msg['role'] == 'assistant'), None)
    if assistant_index is None:
        stats['skipped'] += 1
        return None
    prompt_bytes = sum(len(msg['content'].encode()) for msg in messages[:assistant_index])
    assistant_bytes = len(messages[assistant_index]['content'].encode())
    if prompt_bytes > MAX_RAW_BYTES or assistant_bytes > MAX_RAW_BYTES:
        stats['skipped'] += 1
        stats['raw_prompt_oversize' if prompt_bytes > MAX_RAW_BYTES else 'raw_response_oversize'] += 1
        return None
    # Delegate formatting, response truncation and causal-LM labels to upstream.
    sample = dataset._tokenize_item(item)
    if sample is None:
        stats['skipped'] += 1
    return sample


def cache_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class SampleCache:
    def __init__(self, root, identity, mirrors=()):
        key = cache_key(identity)
        self.root = root / key
        self.mirrors = [mirror / key for mirror in mirrors]
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self, source, kind, index):
        name = f'{source}-{kind}-{index:06d}.safetensors'
        for root in (self.root, *self.mirrors):
            path = root / name
            try:
                with safe_open(str(path), framework='pt') as handle:
                    tokens = handle.get_tensor('tokens')
                    metadata = handle.metadata()
                expected = self.metadata(source, kind, index, tokens)
                if metadata == expected and tokens.dtype == torch.int64 and tokens.ndim == 1:
                    if root != self.root:
                        self.write(source, kind, index, tokens)
                    return tokens
            except (OSError, RuntimeError, ValueError, KeyError, SafetensorError):
                continue
        return None

    def metadata(self, source, kind, index, tokens):
        return dict(identity=self.root.name, source=source, kind=kind, index=str(index),
                    sha256=hashlib.sha256(tokens.contiguous().numpy().tobytes()).hexdigest())

    def write(self, source, kind, index, tokens):
        path = self.root / f'{source}-{kind}-{index:06d}.safetensors'
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            staging = Path(tmp) / path.name
            save_file({'tokens': tokens.contiguous()}, str(staging),
                      metadata=self.metadata(source, kind, index, tokens))
            staging.replace(path)


def raw_stream(dataset_id, revision, seed, buffer_size):
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split='train', revision=revision, streaming=True)
    yield from dataset.shuffle(seed=seed, buffer_size=buffer_size)


def encode_longalign(dataset, item, stats):
    # Count the upstream truncation without performing a second encoding.
    original = dataset._chat_token_ids

    def tracked(messages):
        prompt, full = original(messages)
        if prompt is not None and len(prompt) < dataset.max_length < len(full):
            stats['truncated'] += 1
        return prompt, full

    dataset._chat_token_ids = tracked
    try:
        sample = guarded_longalign(dataset, item, stats)
    finally:
        dataset._chat_token_ids = original
    return sample['input_ids'][sample['attention_mask'].bool()] if sample is not None else None


def encode_fineweb(dataset, item, stats):
    text = item.get('text', '')
    if not text:
        stats['skipped'] += 1
        return None
    return torch.tensor(dataset.tokenizer(text, add_special_tokens=False)['input_ids'], dtype=torch.long)


def collect(source, count, cache, make_stream, encode, eos, width, stats):
    blocks = []
    for index in range(count):
        tokens = cache.read(source, 'block', index)
        if tokens is None or tokens.numel() != width:
            break
        blocks.append(tokens)
        stats['cache_hits'] += 1
    stats['cache_misses'] += count - len(blocks)
    if len(blocks) == count:
        return blocks

    stream = make_stream()
    deadline = time.monotonic() + SOURCE_SECONDS
    packed = []
    produced = 0
    index = 0
    remote_index = 0
    while produced < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PreparationLimit(f'{source} exceeded its {SOURCE_SECONDS}s preparation budget')
        tokens = cache.read(source, 'raw', index)
        if tokens is None:
            stats['raw_cache_misses'] += 1
            while remote_index <= index:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PreparationLimit(f'{source} exceeded its {SOURCE_SECONDS}s preparation budget')
                with budget(min(remaining, NETWORK_SECONDS), 'network', stats):
                    item = next(stream)
                remote_index += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PreparationLimit(f'{source} exceeded its {SOURCE_SECONDS}s preparation budget')
            try:
                with budget(min(remaining, TOKENIZE_SECONDS), 'tokenize', stats):
                    tokens = encode(item)
            except (ValueError, UnicodeError):
                stats['skipped'] += 1
                stats['invalid_samples'] += 1
                tokens = None
            except PreparationLimit:
                stats['tokenize_timeouts'] += 1
                stats['skipped'] += 1
                tokens = None
            tokens = tokens if tokens is not None else torch.empty(0, dtype=torch.long)
            cache.write(source, 'raw', index, tokens)
        else:
            stats['raw_cache_hits'] += 1
        index += 1
        if tokens.numel():
            packed.extend(tokens.tolist())
            if packed[-1] != eos:
                packed.append(eos)
        while len(packed) >= width and produced < count:
            block = torch.tensor(packed[:width], dtype=torch.long)
            del packed[:width]
            if produced >= len(blocks):
                cache.write(source, 'block', produced, block)
                blocks.append(block)
                print(f'Data preparation: cached {source} block {produced + 1}/{count}', flush=True)
            elif not torch.equal(blocks[produced], block):
                raise ValueError(f'{source} cached block differs from indexed source rows')
            produced += 1
    return blocks


def prepare(output, make_dataset, config, cache_root, fingerprint, upstream, mirrors=()):
    from huggingface_hub import HfApi

    start = time.monotonic()
    seed = config.longalign_shuffle_seed
    count = config.total_steps
    order = [source_at(i, seed) for i in range(count)]
    stats = {source: stats_for_source() for source in DATASETS}
    policy = dict(max_raw_bytes=MAX_RAW_BYTES, network_read_seconds=NETWORK_SECONDS,
                  tokenize_sample_seconds=TOKENIZE_SECONDS, source_seconds=SOURCE_SECONDS,
                  oversized_prompt='skip', oversized_response='skip',
                  eligible_response_truncation='upstream', labels='causal LM for all packed tokens')
    report = dict(policy=policy, dataset=config.dataset_name, shuffle_seed=seed, steps=count,
                  sequence_length=config.sequence_length, input_tokens=count * config.sequence_length,
                  source_order=order, statistics=stats, status='ERROR')
    try:
        dataset = make_dataset(config)
        with budget(NETWORK_SECONDS, 'network', stats['FineWeb-Edu']):
            fineweb_revision = HfApi().dataset_info(DATASETS['FineWeb-Edu'], timeout=20).sha
        if not fineweb_revision:
            raise ValueError('Missing FineWeb-Edu revision')
        revisions = {'FineWeb-Edu': fineweb_revision, 'LongAlign': LONGALIGN_REVISION}
        report['source_revisions'] = revisions
        identity = dict(datasets=DATASETS, revisions=revisions, upstream=upstream,
                        tokenizer_fingerprint=fingerprint, config={
                            'dataset': config.dataset_name, 'shuffle_seed': seed,
                            'sequence_length': config.sequence_length,
                            'fineweb_buffer': config.mixed_pretrain_fineweb_shuffle_buffer_size,
                            'longalign_buffer': dataset.longalign.shuffle_buffer_size,
                        }, policy=1)
        report['cache_identity'] = identity
        cache = SampleCache(cache_root, identity, mirrors)
        sources = {
            'FineWeb-Edu': (dataset.fineweb, encode_fineweb, config.mixed_pretrain_fineweb_shuffle_buffer_size),
            'LongAlign': (dataset.longalign, encode_longalign, dataset.longalign.shuffle_buffer_size),
        }

        def batches(source, needed):
            print(f'Data preparation: {source}, {needed} batches', flush=True)
            obj, encode, buffer = sources[source]
            counter = stats[source]
            return collect(source, needed, cache,
                           lambda: raw_stream(DATASETS[source], revisions[source], seed, buffer),
                           lambda item: encode(obj, item, counter), obj.tokenizer.eos_token_id,
                           config.sequence_length, counter)

        fineweb = batches('FineWeb-Edu', order.count('FineWeb-Edu'))
        try:
            longalign = batches('LongAlign', order.count('LongAlign'))
        except (PreparationLimit, StopIteration, OSError) as error:
            report['fallback_reason'] = f'{type(error).__name__}: {error}'
            report['source_order'] = order = ['FineWeb-Edu'] * count
            print(f'FINEWEB_ONLY_FALLBACK: {error}', flush=True)
            fineweb = batches('FineWeb-Edu', count)
            longalign = []
        streams = {'FineWeb-Edu': iter(fineweb), 'LongAlign': iter(longalign)}
        ids = torch.stack([next(streams[source]) for source in order])
        path = output / 'bootstrap-data.safetensors'
        temporary = path.with_suffix('.tmp')
        save_file(dict(input_ids=ids, labels=ids.clone(), attention_mask=torch.ones_like(ids)), str(temporary))
        temporary.replace(path)
        report.update(sha256=sha(path), bytes=path.stat().st_size,
                      status='FINEWEB_ONLY_FALLBACK' if longalign == [] else 'MIXED')
    finally:
        report['preparation_seconds'] = time.monotonic() - start
        for counter in stats.values():
            attempts = counter['cache_hits'] + counter['cache_misses']
            counter['cache_hit_rate'] = counter['cache_hits'] / attempts if attempts else 0.0
        path = output / 'bootstrap-data.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)


class LocalBootstrap(IterableDataset):
    def __init__(self, output, config):
        path = output / 'bootstrap-data.safetensors'
        manifest = json.loads((output / 'bootstrap-data.json').read_text())
        expected = (config.total_steps, config.sequence_length)
        if (manifest['dataset'] != config.dataset_name
                or manifest['shuffle_seed'] != config.longalign_shuffle_seed
                or manifest['sha256'] != sha(path)):
            raise ValueError('Local bootstrap data identity mismatch')
        self.tensors = load_file(str(path))
        if set(self.tensors) != set(FIELDS) or any(
            tensor.shape != expected or tensor.dtype != torch.int64 for tensor in self.tensors.values()
        ):
            raise ValueError('Local bootstrap tensor layout mismatch')

    def __iter__(self):
        for index in range(self.tensors['input_ids'].shape[0]):
            yield {name: tensor[index] for name, tensor in self.tensors.items()}
