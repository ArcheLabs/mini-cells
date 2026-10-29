"""Two independent GPU lanes, the existing R3m transaction engine and serialized HF writes."""
import copy
import inspect
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import torch
from minicells_artifacts.hf_storage import HubBackend, error_code
from minicells_r3e.runner import secrets
from minicells_r3h import runner as coordinator
from minicells_r3i.runner import durable_request as previous_durable_request
from minicells_r3k.reuse import bind
from minicells_r3m import anchor, engine
from minicells_r3m.intervention import propose as previous_propose
from minicells_r3m.storage import restore as restore_private
from minicells_r3j.storage import Store
from . import data, reports, release
from .common import BASE, atomic, canonical, read, source_sha
from .storage import SnapshotPublisher


def identity(c):
    from .protocol import identity as make_identity
    return make_identity(c, source_sha())


def location(output, ident, seed, arm):
    return Path(output) / 'private-checkpoints' / ident['run_id'] / 'jobs' / str(seed) / arm


def jobs(c):
    return [(seed, arm) for seed in c['seeds'] for arm in c['arms']]


def completed(c, output, ident, seed, arm):
    root = location(output, ident, seed, arm)
    if not (root / 'COMPLETE.json').is_file():
        return False
    aid = dict(ident, seed=seed, arm=arm)
    proof = read(root / 'COMPLETE.json')
    if proof['identity'] != aid or proof['step'] != c['anchor_step'] + c['continuation_steps']:
        raise RuntimeError('R3ME_COMPLETED_JOB_IDENTITY_CHANGED')
    store = Store(root, aid)
    state = store.verified(read(store.receipt_file(c['continuation_steps'])))
    if state['trainer']['step'] != proof['step'] or state['trainer']['data_cursor'] != proof['step'] * c['tokens_per_step'] or state['aux']['main_updates'] != c['continuation_steps']:
        raise RuntimeError('R3ME_TERMINAL_BUDGET_CHANGED')
    return True


def training_barrier(c, output, ident):
    missing = [(seed, arm) for seed, arm in jobs(c) if not completed(c, output, ident, seed, arm)]
    if missing:
        raise RuntimeError('R3ME_TRAINING_MATRIX_INCOMPLETE:' + str(missing))


def _marker(*args, **kwargs):
    args = list(args)
    if args and args[0] == '[R3I_SNAPSHOT]':
        args[0] = '[R3M_SNAPSHOT]'
    print(*args, **kwargs)


_durable_request = bind(previous_durable_request, print=_marker)


def control_root(output, ident):
    return Path(output) / 'control' / ident['run_id']


def training_runner(c, output, ident, seed, arm, *, batch_provider=None, durable=None):
    def batches(tr, historical, research, environment, unused):
        return data.batches(tr, historical, research, tr.step - c['anchor_step'], arm, c)[0]

    def propose(tr, seen, root, aid, notify):
        # Candidate persistence remains synchronous before ANY scale scan.
        return previous_propose(tr, seen, root, aid,
            durable or (lambda: _durable_request(control_root(output, ident), seed, arm)))

    return bind(engine.run_arm, batches=batch_provider or batches, propose=propose)


def worker(c, output, folder, index, count, phase):
    from tokenizers import Tokenizer
    from .runner import _arm_runtime_name, analyze_job
    secrets()
    ident = identity(c)
    runtime = anchor.runtime(c)
    backend = HubBackend(c['baseline']['hf_reference']['hf_repo'])
    train_assets = data.r3h_train_assets(runtime, backend, Path(folder) / 'r3h-train', ('train',))
    research, manifest = data.load(folder, c)
    tokenizer = Tokenizer.from_file(str(Path(folder) / 'r3h-train/tokenizer.json'))
    device = 'cuda:%s' % index
    current_seed, parent, evidence, frozen = None, None, None, None
    for seed, arm in jobs(c)[index::count]:
        root = location(output, ident, seed, arm)
        aid = dict(ident, seed=seed, arm=arm)
        if phase == 'train':
            if completed(c, output, ident, seed, arm):
                print('[R3M_SNAPSHOT]', seed, arm, 'ALREADY_COMPLETE_NO_UPDATES', flush=True)
                continue
            if seed != current_seed:
                parent, evidence = anchor.load(c, backend, Path(output) / 'parent-r3h', seed, device)
                frozen = copy.deepcopy(parent.state())
                current_seed = seed
            atomic(root / 'ANCHOR.json', evidence)
            fixed_parent = SimpleNamespace(state=lambda: copy.deepcopy(frozen))
            execute = training_runner(c, output, ident, seed, arm)
            def notify():
                print('[R3M_SNAPSHOT]', seed, arm, 'LOCAL_SAFE_BOUNDARY', flush=True)
            print('[R3m-E] train', seed, arm, 'to global step 900', flush=True)
            execute(c, copy.deepcopy(runtime), fixed_parent, aid, root, train_assets, research,
                    'N' if arm == 'E0-STORY-STABLE' else 'C', _arm_runtime_name(arm), device, notify)
            _durable_request(control_root(output, ident), seed, arm)
        elif phase == 'evaluate':
            analyze_job(c, output, runtime, train_assets, research, tokenizer, ident, seed, arm, device)
            print('[R3M_SNAPSHOT]', seed, arm, 'EVALUATED', flush=True)
        else:
            raise ValueError('R3ME_WORKER_PHASE_INVALID')


def worker_pool(c, path, output, folder, phase, callback):
    # Same dependency adapter already used by R3i/R3m; preserve process/error/queue semantics.
    namespace = dict(coordinator.workers.__globals__, BASE=BASE)
    source = inspect.getsource(coordinator.workers).replace('[R3H_SNAPSHOT]', '[R3M_SNAPSHOT]')
    exec(source, namespace)
    return namespace['workers'](c, path, output, folder, phase, callback)


def publication(output):
    from minicells_r3m.publication import publish as previous_publish
    # Original current-main ancestry, metadata-only allowlist, authenticated push and remote SHA check.
    original_git = previous_publish.__globals__['git']
    def metadata_git(checkout, *args):
        # New report directories may not exist on main; keep one real tracked file
        # in the sparse checkout for older Git without staging any source change.
        if args[:3] == ('sparse-checkout', 'set', '--no-cone'):
            args = (*args, '/README.md')
        return original_git(checkout, *args)
    return bind(previous_publish, BASE=BASE, ALLOWED=reports.ARTIFACTS, git=metadata_git)(output)


def run(c, path, output, folder):
    if not c['authorization']['full_training'] or c['formal_confirmation'] != 'NOT_AUTHORIZED':
        raise RuntimeError('R3ME_EXPLORATORY_GPU_TRAINING_NOT_AUTHORIZED_IN_FROZEN_CONFIG')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('R3ME_TWO_VISIBLE_GPUS_REQUIRED_FOR_FROZEN_CUDA_LAYOUT')
    output, folder = Path(output), Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    secrets()
    backend = HubBackend(c['baseline']['hf_reference']['hf_repo'])
    evidence = release.verify(c, backend)
    print('[R3m-E]', evidence['status'], 'existing Gate 2 read only; no repeated GPU tests', flush=True)
    backend.check_write()
    ident = identity(c)
    private = output / 'private-checkpoints' / ident['run_id']
    private.mkdir(parents=True, exist_ok=True)
    session = uuid.uuid4().hex
    info = backend.info()
    if not info.private:
        raise RuntimeError('HF_PRIVATE_REPO_REQUIRED')
    revision = info.sha
    pointer = backend.optional('r3m-e/' + ident['run_id'] + '/latest.json', revision)
    prior = dict(pointer, hf_repo=backend.repo, hf_revision=revision) if pointer else None
    if prior:
        restore_private(backend, prior, private, output, ident, session, c['barriers'])
    publisher = SnapshotPublisher(backend, private, output, ident, session, prior)
    requests_root = control_root(output, ident) / 'upload-requests'
    last_publication = [0.0]
    failures = []
    input_identity = {'identity': ident, 'anchors': [], 'confirmation': 'SEALED',
                      'r3m_main_candidates_accessed': False, 'release_evidence': evidence}

    def persist(force=False):
        pending = [p for p in requests_root.glob('*') if not p.suffix]
        if not force and not pending and time.monotonic() - last_publication[0] < c['execution']['remote_snapshot_min_seconds']:
            return
        try:
            print('[R3m-E] HF checkpoint publication', 'mandatory' if pending else 'periodic/final', flush=True)
            ref = publisher.publish(force)
            print('[R3m-E] HF checkpoint readback verified', ref.get('hf_revision') if ref else None, flush=True)
            for request in pending:
                atomic(request.with_suffix('.ack'), ref)
                request.unlink()
            atomic(output / 'HF_PUBLICATION.json', dict(status='PUBLISHED_VERIFIED', hf_reference=ref))
        except Exception as error:
            detail = dict(status='LOCAL_RETAINED_REMOTE_RETRY_PENDING', **error_code(error))
            atomic(output / 'HF_PUBLICATION.json', detail)
            for request in pending:
                atomic(request.with_suffix('.error'), detail)
                request.unlink()
            print('[R3m-E] HF publication deferred:', type(error).__name__, flush=True)
        last_publication[0] = time.monotonic()

    try:
        runtime = anchor.runtime(c)
        _, _, manifest, _ = data.prepare(c, runtime, backend, folder)
        original_manifest = dict(manifest, config_sha256=c['release']['parent_config_sha256'])
        expected = read(BASE / 'release/gate2/SESSION_A.json')['input_manifest_sha256']
        if canonical(original_manifest) != expected:
            raise RuntimeError('R3ME_RELEASE_CHANGED_VERIFIED_TRAINING_INPUTS')
        input_identity['input_manifest'] = manifest
        atomic(private / 'INPUT_IDENTITY.json', input_identity)
        atomic(output / 'RELEASE_VERIFICATION.json', evidence)
        atomic(output / 'WORK_PLAN.json', dict(identity=ident, workers=2,
            lanes=[jobs(c)[i::2] for i in range(2)], logical_updates=7200, logical_tokens=3686400))
        print('[R3m-E] two GPU lanes; 12 jobs; 7200 updates; no runtime termination limit', flush=True)
        if not all(completed(c, output, ident, seed, arm) for seed, arm in jobs(c)):
            worker_pool(c, path, output, folder, 'train', persist)
        training_barrier(c, output, ident)
        if not all((location(output, ident, seed, arm) / 'EVALUATION.json').is_file() for seed, arm in jobs(c)):
            worker_pool(c, path, output, folder, 'evaluate', persist)
    except Exception as error:
        failures.append(dict(phase='training_or_evaluation', type=type(error).__name__, code=str(error)[:180]))
        print('[R3m-E] stopped; checkpoints retained:', type(error).__name__, flush=True)
    finally:
        persist(True)
    input_identity['anchors'] = [dict(seed=seed, arm=arm, evidence=read(location(output, ident, seed, arm) / 'ANCHOR.json'))
        for seed, arm in jobs(c) if (location(output, ident, seed, arm) / 'ANCHOR.json').is_file()]
    atomic(output / 'INPUT_IDENTITY.json', input_identity)
    result = reports.write(c, output, ident, input_identity, failures)
    result['hf_reference'] = publisher.prior
    result['checkpoint_publication'] = read(output / 'HF_PUBLICATION.json') if (output / 'HF_PUBLICATION.json').exists() else {'status': 'NOT_PUBLISHED'}
    atomic(output / 'reports/RUN_MANIFEST.json', result)
    if os.environ.get('GITHUB_TOKEN'):
        try:
            print('[R3m-E] publishing aggregate reports to GitHub', flush=True)
            result['publication'] = publication(output)
            print('[R3m-E] remote report commit verified', result['publication']['remote_commit_sha'], flush=True)
        except Exception as error:
            result['publication'] = dict(status='LOCAL_RETAINED', **error_code(error))
    else:
        result['publication'] = dict(status='LOCAL_RETAINED', reason='GITHUB_TOKEN_UNAVAILABLE')
    atomic(output / 'RUN_STATUS.json', result)
    if failures or result['status'] != 'COMPLETE_EXPLORATORY':
        raise RuntimeError('R3ME_PARTIAL_SEE_RUN_STATUS_AND_RETAINED_CHECKPOINTS')
    return result
