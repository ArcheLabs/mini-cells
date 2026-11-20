"""Atomic, allowlisted tensor snapshots without Python object deserialization."""
import hashlib
import json
from pathlib import Path
import tempfile

import torch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':')).encode()


def write(path, metadata, tensors):
    from safetensors.torch import save

    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    payload = save({k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()})
    manifest = {**metadata, 'tensor_sha256': digest(payload)}
    manifest['artifact_sha256'] = digest(canonical(manifest))
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        stage = Path(tmp) / 'snapshot'
        stage.mkdir()
        (stage / 'state.safetensors').write_bytes(payload)
        (stage / 'manifest.json').write_bytes(canonical(manifest))
        stage.rename(path)


def read(path, schema):
    from safetensors.torch import load

    path = Path(path)
    expected = {'manifest.json', 'state.safetensors'}
    if path.is_symlink() or {p.name for p in path.iterdir()} != expected or any(
            (path / name).is_symlink() or not (path / name).is_file() for name in expected):
        raise ValueError('Unexpected snapshot files')
    metadata = json.loads((path / 'manifest.json').read_bytes())
    proof = metadata.pop('artifact_sha256', None)
    payload = (path / 'state.safetensors').read_bytes()
    if (metadata.pop('schema', None) != schema
            or proof != digest(canonical({**metadata, 'schema': schema}))
            or metadata.pop('tensor_sha256', None) != digest(payload)):
        raise ValueError('Snapshot integrity or schema mismatch')
    return metadata, load(payload)


def checked(tensors, reference):
    if tensors.keys() != reference.keys():
        raise ValueError('Unexpected tensor names')
    for key, value in tensors.items():
        expected = reference[key]
        if value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
            raise ValueError(f'Invalid tensor {key}')
    return {k: v.to(reference[k].device) for k, v in tensors.items()}
