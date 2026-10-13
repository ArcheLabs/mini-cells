import json
from pathlib import Path
from .reuse import BASE, ROOT, canonical_sha256, code_digest, file_sha256


def load(path=None):
    c = json.loads(Path(path or BASE / 'configs/default.json').read_text())
    if not 1 <= c['k'] <= c['c'] <= c['nodes']:
        raise ValueError('require 1 <= k <= c <= nodes')
    if c['active_limit'] > c['nodes'] or c['rounds'] < c['train_depth']:
        raise ValueError('invalid state/round budget')
    if c['steps'] <= 0 or c['batch_size'] <= 0:
        raise ValueError('invalid training budget')
    if c['confirmation']['enabled']:
        raise ValueError('v1 screening config cannot authorize sealed evaluation')
    return c


def authorize_sealed(config, lock, checkpoint_sha, source_commit=None):
    """Only a separately reviewed frozen lock can open sealed evaluation."""
    if not lock or lock.get('approval') != 'APPROVED':
        raise PermissionError('sealed_test requires a reviewed registration lock')
    if (lock.get('config_sha256') != canonical_sha256(config)
            or lock.get('checkpoint_sha256') != checkpoint_sha
            or len(set(lock.get('replicate_seeds', []))) < 3
            or not lock.get('source_commit')
            or lock.get('source_commit') != source_commit):
        raise PermissionError('sealed lock does not identify frozen evidence')


def authorize_screening(config, source):
    import re
    import subprocess
    lock = json.loads((BASE / 'registration/screening-lock.json').read_text())
    if (lock['status'] != 'SCREENING_AUTHORIZED'
            or lock['config_sha256'] != canonical_sha256(config)
            or lock['code_sha256'] != code_digest()
            or lock['protocol_sha256'] != file_sha256(BASE / 'PROTOCOL_v1.0.md')
            or not re.fullmatch('[0-9a-f]{40}', source)):
        raise PermissionError('screening source/config/protocol differs from its frozen contract')
    paths = [str(BASE.relative_to(ROOT) / part) for part in ('src', 'configs', 'run.py', 'PREREGISTRATION.md')]
    stored = subprocess.run(['git', 'cat-file', '-e',
        source + ':' + str(BASE.relative_to(ROOT) / 'src/native_sparse_path/models.py')],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if stored.returncode:
        raise PermissionError('requested commit does not contain this implementation')
    result = subprocess.run(['git', 'diff', '--quiet', source, '--', *paths], cwd=ROOT)
    if result.returncode:
        raise PermissionError('working implementation differs from requested source commit')
