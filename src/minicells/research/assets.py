"""CPU-side asset preparation and hash-checked, offline Kaggle loading."""
import argparse
import json
import os
import time
from pathlib import Path

from .config import digest, dump, load_definition

DATA_KEYS = ('dataset', 'dataset_revision', 'train_stories', 'validation_stories',
             'tokenizer_stories', 'vocab_size', 'sequence_length')


def prepare(definition, destination):
    import numpy as np
    from experiments.common.train import prepare_data
    _, config, _ = load_definition(definition)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # Reuse the exact train-only tokenizer and pinned corpus implementation.
    arrays = prepare_data(config, destination, time.time() + 3600)
    for split, array in arrays.items():
        np.save(destination / (split + '.npy'), array)
    identity = {key: config[key] for key in DATA_KEYS}
    manifest = {'schema_version': 2, 'data_config': identity,
                'files': {name: digest(destination / name) for name in
                          ('train.npy', 'validation.npy', 'tokenizer.json', 'data.json')}}
    dump(destination / 'assets-manifest.json', manifest)
    return manifest


def load_assets(directory, config, expected_sha256):
    import numpy as np
    directory = Path(directory)
    manifest_path = directory / 'assets-manifest.json'
    if digest(manifest_path) != expected_sha256:
        raise ValueError('Mounted Dataset differs from the prepared asset manifest')
    manifest = json.loads(manifest_path.read_text())
    if manifest['data_config'] != {key: config[key] for key in DATA_KEYS}:
        raise ValueError('Input corpus/tokenizer definition differs from the experiment')
    for name, checksum in manifest['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != checksum:
            raise ValueError('Asset hash mismatch: ' + name)
    data = {split: np.load(directory / (split + '.npy'), mmap_mode='r', allow_pickle=False)
            for split in ('train', 'validation')}
    record = json.loads((directory / 'data.json').read_text())
    for split, array in data.items():
        if array.ndim != 1 or array.dtype != np.int32 or len(array) <= config['sequence_length']:
            raise ValueError('Invalid token array: ' + split)
        if len(array) != record['token_counts'][split]:
            raise ValueError('Token count mismatch')
        import hashlib
        if hashlib.sha256(array.tobytes()).hexdigest() != record['token_sha256'][split]:
            raise ValueError('Token content mismatch')
        if array.min() < 0 or array.max() >= config['vocab_size']:
            raise ValueError('Token outside configured vocabulary')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--definition', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.definition, args.output)
    print(json.dumps({'assets_manifest_sha256': digest(Path(args.output) / 'assets-manifest.json')}))


if __name__ == '__main__':
    main()
