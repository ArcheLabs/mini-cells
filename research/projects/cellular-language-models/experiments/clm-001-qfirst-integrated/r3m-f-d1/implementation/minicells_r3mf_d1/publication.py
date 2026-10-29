"""D1 result-only adapter over the repository's existing Kaggle GitHub authentication."""
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from .execution import BASE, _atomic_json, _file_sha, _read
from .identity import canonical, sha256_bytes

REPOSITORY_URL = 'https://github.com/ArcheLabs/mini-cells.git'
REPORTS = BASE.relative_to(BASE.parents[5]) / 'reports/published'
EXPERIMENT_ID = 'CLM-001-R3m-F-D1-Supervision-Context-001'
ALLOWED = (
    'RESULTS.json', 'RUN_MANIFEST.json', 'BUDGET_PROOF.json', 'SAMPLE_PLAN.json',
    'A_REUSED_PREDICTIONS.json', 'CPU_GATE_RUN_MANIFEST.json',
    'arm-B/ARM_RESULT.json', 'arm-C/ARM_RESULT.json', 'arm-D/ARM_RESULT.json',
)
MAX_FILE_BYTES = 4_000_000
MAX_TOTAL_BYTES = 12_000_000


class PublicationError(RuntimeError):
    pass


def _auth_env():
    # Same GIT_ASKPASS / existing Kaggle GITHUB_TOKEN implementation as R3g/R3m.
    from minicells_r3c_moe_bridge.publication import helper

    return helper()._authenticated_git_env('GITHUB_TOKEN')


def _git(checkout, *args, env=None, timeout=30):
    try:
        result = subprocess.run(
            ['git', '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=10', *map(str, args)],
            cwd=checkout, env=env, text=True, capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise PublicationError('D1_GITHUB_NETWORK_TIMEOUT_' + str(args[0]).upper()) from None
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        for name in ('GITHUB_TOKEN', 'HF_TOKEN', 'KAGGLE_API_TOKEN'):
            value = (env or os.environ).get(name, '')
            if value:
                detail = detail.replace(value, '[REDACTED]')
        raise PublicationError('D1_GITHUB_GIT_' + str(args[0]).upper() + '_FAILED: ' + detail[:500])
    return result.stdout.strip()


def _checkout(root, env):
    checkout = Path(root) / 'github-metadata'
    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / '.git').exists():
        _git(checkout, 'init', '--quiet', env=env)
        _git(checkout, 'remote', 'add', 'origin', REPOSITORY_URL, env=env)
        _git(checkout, 'config', 'remote.origin.promisor', 'true', env=env)
        _git(checkout, 'config', 'remote.origin.partialclonefilter', 'blob:none', env=env)
    if _git(checkout, 'remote', 'get-url', 'origin', env=env) != REPOSITORY_URL:
        raise PublicationError('D1_GITHUB_ORIGIN_MISMATCH')
    _git(checkout, 'config', 'user.name', 'MiniCells Kaggle', env=env)
    _git(checkout, 'config', 'user.email', 'minicells@users.noreply.github.com', env=env)
    _git(checkout, 'sparse-checkout', 'init', '--no-cone', env=env)
    _git(checkout, 'sparse-checkout', 'set', '--no-cone', '/README.md', '/' + REPORTS.as_posix() + '/', env=env)
    return checkout


def preflight(root):
    """One bounded dry-run before GPU work; it does not create a remote branch."""
    root = Path(root)
    manifest = _read(root / 'RUN_MANIFEST.json')
    started = time.monotonic()
    try:
        with _auth_env() as env:
            checkout = _checkout(root, env)
            _git(checkout, 'fetch', '--depth=1', '--filter=blob:none', 'origin', 'main', env=env, timeout=30)
            parent = _git(checkout, 'rev-parse', 'FETCH_HEAD', env=env)
            _git(checkout, 'checkout', '--detach', parent, env=env)
            branch = 'results/r3mf-d1-preflight-' + manifest['run_id']
            _git(checkout, 'push', '--dry-run', 'origin', 'HEAD:refs/heads/' + branch, env=env, timeout=20)
        value = dict(status='PREFLIGHT_PASSED', parent_main_sha=parent,
                     elapsed_seconds=time.monotonic()-started, secret_label='GITHUB_TOKEN')
        manifest['publication_preflight'] = value
        manifest['publication_status'] = 'PREFLIGHT_PASSED'
        _atomic_json(root / 'GITHUB_PREFLIGHT.json', value)
        _atomic_json(root / 'RUN_MANIFEST.json', manifest)
        print('[D1 GitHub] write preflight passed; no GPU work started', flush=True)
        return value
    except Exception as error:
        manifest['publication_status'] = 'PREFLIGHT_FAILED_NO_GPU_WORK'
        manifest['publication_failure_type'] = type(error).__name__
        _atomic_json(root / 'RUN_MANIFEST.json', manifest)
        raise PublicationError('D1_GITHUB_PREFLIGHT_FAILED_ENABLE_EXISTING_GITHUB_TOKEN_OR_CHECK_NETWORK') from error


def _strict_json(path):
    def reject(value):
        raise PublicationError('D1_NONFINITE_JSON_' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def payload(root):
    """Validate exact allowlisted JSON and checkpoint references; never copy weight files."""
    root = Path(root)
    manifest = _strict_json(root / 'RUN_MANIFEST.json')
    if manifest.get('experiment_id') != EXPERIMENT_ID or manifest.get('main_or_confirmation_accessed') is not False:
        raise PublicationError('D1_PUBLICATION_IDENTITY_OR_DATA_SCOPE_INVALID')
    if len(manifest.get('source_sha', '')) != 40 or len(manifest.get('run_id', '')) != 32:
        raise PublicationError('D1_PUBLICATION_SOURCE_OR_RUN_ID_INVALID')
    values = {}
    files = {}
    total = 0
    for name in ALLOWED:
        path = root / name
        if not path.exists():
            if name == 'RUN_MANIFEST.json' or (manifest['run_status'] == 'COMPLETE' and name != 'CPU_GATE_RUN_MANIFEST.json'):
                raise PublicationError('D1_REQUIRED_RESULT_MISSING_' + name)
            continue
        parts = Path(name).parts
        if any(root.joinpath(*parts[:length]).is_symlink() for length in range(1, len(parts)+1)):
            raise PublicationError('D1_PUBLICATION_SYMLINK_REJECTED')
        path.resolve().relative_to(root.resolve())
        total += path.stat().st_size
        if path.stat().st_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise PublicationError('D1_RESULT_PAYLOAD_SIZE_EXCEEDED')
        value = _strict_json(path)
        if name != 'RUN_MANIFEST.json':
            declared = manifest.get('artifacts', {}).get(name)
            if declared != dict(bytes=path.stat().st_size, sha256=_file_sha(path)):
                raise PublicationError('D1_RESULT_ARTIFACT_HASH_MISMATCH_' + name)
        if name == 'RESULTS.json' and (value.get('run_id') != manifest['run_id'] or value.get('experiment_id') != EXPERIMENT_ID or value.get('status') != manifest['run_status']):
            raise PublicationError('D1_COMBINED_RESULT_IDENTITY_MISMATCH')
        if name.startswith('arm-'):
            identity = value.get('identity', {})
            if value.get('arm') != name[4] or any(identity.get(key) != manifest.get(key) for key in ('experiment_id','source_sha','source_code_sha256','config_sha256','data_sha256','sample_plan_sha256','seed')):
                raise PublicationError('D1_ARM_RESULT_IDENTITY_MISMATCH')
        values[name] = value
        files[name] = path.read_bytes()
    # Publication state changes after push. The immutable remote manifest is a payload snapshot;
    # its own Git SHA cannot be embedded in the commit that defines it.
    snapshot = copy.deepcopy(manifest)
    for name in tuple(snapshot):
        if name.startswith('publication_') or name in ('remote_commit_sha', 'publication'):
            snapshot.pop(name)
    snapshot['publication_status'] = 'GIT_PAYLOAD_SNAPSHOT'
    snapshot['publication_receipt_location'] = 'Kaggle Output PUBLICATION.json; remote branch commit metadata'
    values['RUN_MANIFEST.json'] = snapshot
    files['RUN_MANIFEST.json'] = (json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    index = {name: dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()}
    return manifest, files, index, sha256_bytes(canonical(index).encode())


def _receipt(root, manifest, receipt):
    _atomic_json(Path(root) / 'PUBLICATION.json', receipt)
    manifest['publication_status'] = receipt['status']
    manifest['remote_commit_sha'] = receipt.get('remote_commit_sha') if receipt.get('remote_verified') else None
    manifest['publication'] = receipt
    _atomic_json(Path(root) / 'RUN_MANIFEST.json', manifest)


def publish(root):
    root = Path(root)
    manifest = _read(root / 'RUN_MANIFEST.json')
    receipt = dict(status='VALIDATING', run_id=manifest.get('run_id'),
                   git_commit_created=False, push_attempted=False, push_completed=False,
                   push_exit_code=None, remote_verified=False)
    try:
        manifest, files, hashes, key = payload(root)
        branch = 'results/r3mf-d1-' + manifest['run_id'] + '-' + key[:12]
        dest = REPORTS / manifest['run_id']
        receipt.update(branch=branch, report_directory=dest.as_posix(), payload_sha256=key,
                       publisher_source_sha=_git(BASE.parents[5], 'rev-parse', 'HEAD'))
        _receipt(root, manifest, receipt)
        with _auth_env() as env:
            checkout = _checkout(root, env)
            remote = _git(checkout, 'ls-remote', 'origin', 'refs/heads/'+branch, env=env, timeout=15).split()
            if remote:
                _git(checkout, 'fetch', '--depth=1', '--filter=blob:none', 'origin', branch, env=env)
                _git(checkout, 'checkout', '-B', branch, 'FETCH_HEAD', env=env)
                retained = _read(checkout / dest / 'PUBLICATION_INDEX.json')
                if retained['payload_sha256'] != key or retained['artifact_files'] != hashes:
                    raise PublicationError('D1_REMOTE_RESULT_CONTENT_CONFLICT')
                for name,row in hashes.items():
                    path = checkout / dest / name
                    if not path.is_file() or path.is_symlink() or path.stat().st_size != row['bytes'] or _file_sha(path) != row['sha256']:
                        raise PublicationError('D1_REMOTE_ARTIFACT_CONTENT_NOT_VERIFIED')
                parent = retained['parent_main_sha']
                receipt['reused_existing_remote'] = True
            else:
                _git(checkout, 'fetch', '--depth=1', '--filter=blob:none', 'origin', 'main', env=env)
                parent = _git(checkout, 'rev-parse', 'FETCH_HEAD', env=env)
                _git(checkout, 'checkout', '-B', branch, parent, env=env)
                for name,data in files.items():
                    target = checkout / dest / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                index = dict(schema='minicells.r3mf_d1.publication_index.v1', run_id=manifest['run_id'],
                             experiment_id=EXPERIMENT_ID, source_sha=manifest['source_sha'],
                             config_sha256=manifest['config_sha256'], payload_sha256=key,
                             publisher_source_sha=receipt['publisher_source_sha'],
                             parent_main_sha=parent, artifact_files=hashes, checkpoints_in_git=False)
                _atomic_json(checkout / dest / 'PUBLICATION_INDEX.json', index)
                allowed = {str(dest/name) for name in files} | {str(dest/'PUBLICATION_INDEX.json')}
                _git(checkout, 'add', '--', *sorted(allowed), env=env)
                staged = _git(checkout, 'diff', '--cached', '--name-only', env=env).splitlines()
                if not staged or set(staged)-allowed:
                    raise PublicationError('D1_RESULT_ONLY_STAGING_REQUIRED')
                _git(checkout, 'commit', '-m', 'results: R3m-F-D1 '+manifest['run_id'], env=env)
                changed = _git(checkout, 'diff', '--name-only', parent, 'HEAD', env=env).splitlines()
                if set(changed)-allowed or not changed:
                    raise PublicationError('D1_NON_RESULT_CHANGE_REJECTED')
                receipt.update(status='COMMITTED_LOCALLY', git_commit_created=True,
                               local_commit_sha=_git(checkout,'rev-parse','HEAD',env=env))
                _receipt(root, manifest, receipt)
                receipt['push_attempted'] = True
                _receipt(root, manifest, receipt)
                _git(checkout, 'push', 'origin', 'HEAD:refs/heads/'+branch, env=env, timeout=45)
                receipt.update(status='PUSH_SUCCEEDED_AWAITING_REMOTE_VERIFICATION',push_completed=True,push_exit_code=0)
                _receipt(root, manifest, receipt)
            local = _git(checkout, 'rev-parse', 'HEAD', env=env)
            verified = _git(checkout, 'ls-remote', 'origin', 'refs/heads/'+branch, env=env, timeout=15).split()
            if not verified or verified[0] != local:
                raise PublicationError('D1_REMOTE_COMMIT_NOT_VERIFIED')
        receipt.update(status='PUBLISHED', remote_verified=True, remote_commit_sha=local,
                       parent_main_sha=parent,
                       report_url='https://github.com/ArcheLabs/mini-cells/tree/'+branch+'/'+dest.as_posix())
        _receipt(root, manifest, receipt)
        print('[D1 GitHub] PUBLISHED '+branch+' remote_commit_sha='+local,flush=True)
        return receipt
    except Exception as error:
        receipt.update(status='FAILED_LOCAL_RETAINED',failure_type=type(error).__name__)
        if isinstance(error,PublicationError):
            receipt['failure_code'] = str(error).split(':')[0]
        else:
            receipt['failure_code'] = 'D1_PUBLICATION_FAILED'
        _receipt(root, manifest, receipt)
        raise PublicationError('D1_PUBLICATION_FAILED_RESULTS_RETAINED_RETRY_PUBLISH_EXISTING') from error


def publish_existing(output):
    """Publish an existing run or all retained D1 runs; never restore or train a model."""
    output = Path(output)
    roots = [output] if (output/'RUN_MANIFEST.json').exists() else sorted((output/'runs').glob('*/RUN_MANIFEST.json'))
    roots = [p if p.is_dir() else p.parent for p in roots]
    if not roots:
        raise PublicationError('D1_NO_RETAINED_RUN_TO_PUBLISH')
    return [publish(root) for root in roots]
