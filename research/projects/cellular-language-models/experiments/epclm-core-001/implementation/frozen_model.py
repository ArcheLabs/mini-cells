"""Read-only, pinned training-asset recovery. Never creates a trainer or optimizer."""
import json
from pathlib import Path
import shutil
import sys
import time
import torch
from model import Core, config_from_protocol, check_numerics
from runtime import weight_hash
from trainer import load_checkpoint
from native_clm_runtime import sha256_file, load_tokenizer


class AssetUnavailable(RuntimeError):
    pass


class BaselineIntegrityError(ValueError):
    pass


class DiagnosticCore(Core):
    """Identical parameters and normal forward; one inference-only message intervention."""
    def evolve(self, state, steps=None, communication=True, trace=None):
        if communication != 'CROSS_CELL_REMOVED':
            return super().evolve(state, steps, communication, trace)
        if self.config.arm != 'C1' or trace is not None:
            raise ValueError('cross-cell removal requires C1 and no training trace')
        steps = self.config.steps if steps is None else steps
        if steps < 0: raise ValueError('negative evolution count')
        for _ in range(steps):
            normalized = self.norm(state)
            message = self.center(normalized)  # Left/right terms are zero, not self substitutes.
            delta = self.update(torch.cat((normalized, message), -1))
            state = state + self.config.alpha * delta
            check_numerics(state.abs().amax() <= self.config.max_state_abs,
                           'diagnostic state outside registered numerical range')
        return state


def check_deadline(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('diagnostic/recovery budget exhausted')


def safe_path(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts: raise BaselineIntegrityError('unsafe asset path')
    return path


def recover_frozen_assets(root, destination, training_identity, checkpoint_identity, deadline, max_versions=25,
                          progress_output=None, progress_identity=None):
    """Find the newest exact identity; stop immediately once verified assets are found."""
    names = (checkpoint_identity['path'], 'assets/manifest.json', 'assets/tokenizer.json', 'assets/development.json')

    def accept(manifest, locate, provenance):
        if manifest.get('identity') != training_identity: return False
        if not set(names) <= manifest.get('files', {}).keys():
            raise AssetUnavailable('matching training manifest is incomplete')
        if manifest['files'][names[0]] != checkpoint_identity['sha256'] or manifest['files'][names[1]] != training_identity['data']:
            raise BaselineIntegrityError('pinned checkpoint/data hash differs from training manifest')
        for name in names:
            check_deadline(deadline)
            source = locate(name)
            if sha256_file(source) != manifest['files'][name]:
                raise BaselineIntegrityError('frozen asset hash mismatch: ' + name)
            target = destination / safe_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve(): shutil.copyfile(source, target)
        (destination/'training-recovery.json').write_text(json.dumps({'identity':training_identity,
            'files':{n:manifest['files'][n] for n in names}}, indent=2)+'\n')
        (destination/'asset-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        return True

    destination.mkdir(parents=True, exist_ok=True)
    for local in (destination, destination.parent/checkpoint_identity['namespace']):
        record = local / ('training-recovery.json' if local == destination else 'recovery.json')
        if record.exists():
            manifest = json.loads(record.read_text())
            locations = manifest.get('objects', {}) if local != destination else {}
            if accept(manifest, lambda n: local/safe_path(locations.get(n,n)),
                      {'storage':'PRIVATE_KAGGLE_OUTPUT','local_cache':str(local)}): return

    sys.path.insert(0, str(root/'research/projects/cellular-language-models/experiments/epclm-002-a0-persistent-state-recall/scripts'))
    from kaggle_output_recovery import _load_backend, _version_number
    api, fetch = _load_backend(root)
    latest = _version_number(api.metadata())
    cache = destination/'download-cache'
    for version in range(latest, max(0, latest-max_versions), -1):
        check_deadline(deadline)
        print('[C1 frozen recovery] inspect saved version '+str(version), flush=True)
        token = None; seen = set(); inventory_names = set()
        while True:
            check_deadline(deadline)
            page = api.page(version, token)
            inventory_names.update(row['file_name'] for row in page['files'])
            token = page.get('next_page_token')
            if not token: break
            if token in seen: raise AssetUnavailable('saved-output pagination loop')
            seen.add(token)
        candidates = sorted(n for n in inventory_names if n.endswith(checkpoint_identity['namespace']+'/recovery.json')
                            or n.endswith(destination.name+'/training-recovery.json')
                            or (progress_output is not None and n.endswith(progress_output.name+'/recovery.json')))
        candidates.sort(key=lambda n:(0 if progress_output is not None and n.endswith(progress_output.name+'/recovery.json') else 1,n))
        for name in candidates:
            safe_path(name)
            download = cache/str(version)
            def get(physical):
                check_deadline(deadline); safe_path(physical)
                if physical not in inventory_names: raise AssetUnavailable('missing frozen recovery object')
                fetch(api, {'version_number':version,'files':[{'name':physical,'bytes':None}]}, [physical], download)
                check_deadline(deadline)
                return download/physical
            manifest = json.loads(get(name).read_text())
            prefix = name.rsplit('/',1)[0]+'/'
            if progress_output is not None and name.endswith(progress_output.name+'/recovery.json'):
                if manifest.get('identity') != progress_identity: continue
                # Results only: never deserialize model/optimizer state through progress recovery.
                for logical, sha in manifest['files'].items():
                    relative = safe_path(logical)
                    if relative.suffix not in ('.json','.png','.md'): raise BaselineIntegrityError('unexpected diagnostic progress file')
                    saved = get(prefix+str(safe_path(manifest.get('objects',{}).get(logical,logical))))
                    if sha256_file(saved) != sha: raise BaselineIntegrityError('diagnostic progress checksum mismatch')
                    target = progress_output/relative; target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(saved,target)
                # Prefer the latest progress package; do not overwrite it with earlier versions.
                decision_path = progress_output/'decision.json'
                progress_output = None
                if json.loads(decision_path.read_text()).get('status') != 'INCOMPLETE':
                    return 'DIAGNOSTIC_TERMINAL'
                continue
            if accept(manifest, lambda n: get(prefix+str(safe_path(manifest.get('objects',{}).get(n,n)))),
                      {'storage':'PRIVATE_KAGGLE_OUTPUT','kaggle_version':version,'manifest_path':name}): return
    raise AssetUnavailable('no saved Output contains the pinned trained C1; no replacement training permitted')


def load_frozen_model(training_identity, checkpoint_identity, assets, training_protocol, device):
    """Validate before unpickling the trusted checkpoint, then load only its model weights."""
    if sha256_file(assets/checkpoint_identity['path']) != checkpoint_identity['sha256']:
        raise BaselineIntegrityError('checkpoint SHA-256 mismatch')
    if sha256_file(training_protocol) != training_identity['protocol']:
        raise BaselineIntegrityError('training protocol SHA-256 mismatch')
    if sha256_file(assets/'assets/manifest.json') != training_identity['data']:
        raise BaselineIntegrityError('data manifest SHA-256 mismatch')
    metadata = json.loads((assets/'assets/manifest.json').read_text())
    protocol = json.loads(training_protocol.read_text())
    if (metadata['dataset_id'] != protocol['data']['id'] or metadata['dataset_revision'] != protocol['data']['revision']
            or metadata['split'] != 'core001-content-sha256-mod10-development'):
        raise BaselineIntegrityError('training dataset identity mismatch')
    for name in ('tokenizer.json','development.json'):
        if sha256_file(assets/'assets'/name) != metadata['files'][name]:
            raise BaselineIntegrityError('tokenizer/development hash mismatch')
    c = config_from_protocol(protocol, 'C1')
    saved = load_checkpoint(assets/checkpoint_identity['path'])
    from dataclasses import asdict
    expected = {**training_identity,'model':asdict(c),'target_tokens':250000}
    if saved['identity'] != expected or saved['tokens'] != 250000 or saved['updates'] != 977 or saved['precision'] != 'fp32':
        raise BaselineIntegrityError('checkpoint training/configuration identity mismatch')
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all()) for v in saved['model'].values()):
        raise BaselineIntegrityError('checkpoint contains non-FP32 or non-finite weights')
    model = DiagnosticCore(c).to(device=device, dtype=torch.float32)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if weight_hash(model) != checkpoint_identity['weights_sha256']:
        raise BaselineIntegrityError('trained model weight hash mismatch')
    tokenizer = load_tokenizer(assets/'assets/tokenizer.json')
    if tokenizer.get_vocab_size() != c.vocab: raise BaselineIntegrityError('tokenizer vocabulary differs')
    documents = json.loads((assets/'assets/development.json').read_text())
    return model, tokenizer, documents
