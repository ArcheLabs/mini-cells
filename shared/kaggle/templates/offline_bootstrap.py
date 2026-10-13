"""Unpack the reviewed source payload and locate the exact mounted input snapshot."""
import base64
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

COMMIT = '__SOURCE_COMMIT__'
SEED = __SEED__
ASSET_SHA256 = '__ASSET_SHA256__'
PAYLOAD_SHA256 = '__PAYLOAD_SHA256__'
PAYLOAD = '__SOURCE_PAYLOAD__'
os.environ['EXPERIMENT_STARTED_AT'] = str(time.time())
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
source = Path('/shared/kaggle/working/experiment-source')
encoded = base64.b64decode(PAYLOAD)
if hashlib.sha256(encoded).hexdigest() != PAYLOAD_SHA256:
    raise RuntimeError('Source bundle hash mismatch')
source.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
    for entry in archive.infolist():
        dest = (source / entry.filename).resolve()
        if not dest.is_relative_to(source.resolve()):
            raise RuntimeError('Invalid source bundle path')
    archive.extractall(source)
sys.path[:0] = [str(source / 'src'), str(source)]
candidates = [p.parent for p in Path('/shared/kaggle/input').rglob('assets-manifest.json')
              if hashlib.sha256(p.read_bytes()).hexdigest() == ASSET_SHA256]
if len(candidates) != 1:
    raise RuntimeError('Expected exactly one mounted Dataset matching the input manifest hash')
assets = candidates[0]
output = Path('/shared/kaggle/working/results')
print({'experiment': '__EXPERIMENT_ID__', 'commit': COMMIT, 'seed': SEED,
       'assets_manifest_sha256': ASSET_SHA256, 'scientific_decision': False})
