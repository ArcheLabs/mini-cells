"""Disposable real-model transactions and an automatic private-HF session relay.

No study checkpoint, release flag, scientific threshold or historical file is changed.
"""

import copy
from pathlib import Path
import socket
import time
from types import SimpleNamespace
import uuid

import torch

from minicells_r3k.reuse import bind
from minicells_r3m import anchor, engine
from minicells_r3m.training import advance, restore, semantic
from minicells_r3j.storage import Store
from minicells_r3c_moe_bridge.checkpoint import publish_segment, restore_segment
from minicells_r3c_moe_bridge.development import verified_manifest
from . import data
from .common import atomic, canonical, digest, fingerprint, read, source_sha

PROTOCOL = {
    "schema": "minicells.r3me.gate2.v1",
    "scope": "DISPOSABLE_ENGINEERING_ONLY",
    "seed": 151001301,
    "main_updates": 11,
    "proposal_at": 1,
    "shadow_updates": 10,
    "pause_at": 4,
    "fault_test_main_updates": 2,
    "fault_test_shadow_updates": 1,
    "gpu_rtol": 1e-5,
    "gpu_atol": 1e-7,
    "production_updates": 0,
}


def session_identity():
    """Record a container lifetime, rather than treating a new Python PID as a session."""
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    init = Path('/proc/1/stat').read_text().rsplit(')', 1)[1].split()[19]
    fields = {"hostname": socket.gethostname(), "boot_id": boot, "pid1_start_ticks": init}
    return dict(fields, container_lifetime_sha256=canonical(fields))


def compare_states(expected, actual, *, cuda=False):
    """Exact RNG/cursors/identity; allow only declared GPU floating point tolerance."""
    for key in set(semantic(expected)) - {"model", "cell_optimizer"}:
        if fingerprint(expected[key]) != fingerprint(actual[key]):
            raise RuntimeError("GATE2_EXACT_STATE_MISMATCH:" + key)
    maximum = 0.0

    def check(a, b, name):
        nonlocal maximum
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                raise RuntimeError("GATE2_TENSOR_IDENTITY_MISMATCH:" + name)
            a, b = a.detach().cpu(), b.detach().cpu()
            if a.is_floating_point() and cuda:
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise RuntimeError("GATE2_NONFINITE_STATE:" + name)
                if not torch.allclose(a, b, rtol=PROTOCOL['gpu_rtol'], atol=PROTOCOL['gpu_atol']):
                    raise RuntimeError("GATE2_FLOAT_STATE_MISMATCH:" + name)
                if a.numel():
                    maximum = max(maximum, float((a - b).abs().max()))
            elif not torch.equal(a, b):
                raise RuntimeError("GATE2_EXACT_TENSOR_MISMATCH:" + name)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise RuntimeError("GATE2_STATE_KEYS_MISMATCH:" + name)
            for k in a:
                check(a[k], b[k], name + '/' + str(k))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise RuntimeError("GATE2_STATE_SEQUENCE_MISMATCH:" + name)
            for k, (x, y) in enumerate(zip(a, b)):
                check(x, y, name + '/' + str(k))
        elif a != b:
            raise RuntimeError("GATE2_STATE_VALUE_MISMATCH:" + name)

    for key in ('model', 'cell_optimizer'):
        check(expected[key], actual[key], key)
    return dict(exact_rng_cursor_identity=True, model_optimizer_match=True,
                max_absolute_float_difference=maximum,
                rtol=PROTOCOL['gpu_rtol'] if cuda else 0, atol=PROTOCOL['gpu_atol'] if cuda else 0)


def make_runner(c, batch_provider=None, proposal_provider=None, quality_provider=None):
    """Rebind stateless dependencies of the actual R3m transaction engine."""
    def batches(tr, historical, research, environment, unused):
        return data.batches(tr, historical, research, tr.step - c['anchor_step'],
                            'E2-MIX-REORG', c)[0]

    def update(tr, batch, unused, schedule_state, perturbed=False):
        # Short engineering timeline does not compress the production LR schedule.
        return advance(tr, batch, c, schedule_state, perturbed)

    dependencies = dict(batches=batch_provider or batches, advance=update)
    if proposal_provider is not None:
        dependencies['propose'] = proposal_provider
    if quality_provider is not None:
        dependencies['train_quality'] = quality_provider
    return bind(engine.run_arm, **dependencies)


def engineering_config(c, *, fault=False):
    value = copy.deepcopy(c)
    value.update(continuation_steps=2 if fault else 11, proposal_at=1,
                 shadow_steps=1 if fault else 10,
                 barriers=[0, 1, 2] if fault else [0, 1, 4, 11])
    return value


def export_boundary(source, destination, identity, step):
    """Copy one verified receipt/object; no obsolete checkpoint versions are uploaded."""
    import shutil
    src = Store(source, identity)
    row = read(src.receipt_file(step))
    src.verified(row)
    dest = Store(destination, identity)
    target = Path(destination) / row['file']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source) / row['file'], target)
    atomic(dest.receipt_file(step), row)
    atomic(Path(destination) / 'latest.json', row)
    dest.verified(row)
    return row


class _Pause(Exception):
    pass


def run_transaction_checks(c, runtime, parent, historical, research, root, identity,
                           device, batch_provider=None, proposal_provider=None):
    """Real search + 10-step transaction; fault-injected rejection and an exact control."""
    root = Path(root)
    frozen_parent = copy.deepcopy(parent.state())
    anchor_view = SimpleNamespace(state=lambda: copy.deepcopy(frozen_parent))
    ident = dict(identity, seed=parent.model.seed, branch='actual-proposal')
    engine_c = engineering_config(c)
    execute = make_runner(c, batch_provider, proposal_provider)
    work = root / 'working' / 'actual'
    paused = root / 'relay' / 'paused'
    pause_receipt = None

    def pause():
        nonlocal pause_receipt
        store = Store(work, ident)
        if store.receipt_file(PROTOCOL['pause_at']).is_file():
            pause_receipt = export_boundary(work, paused, ident, PROTOCOL['pause_at'])
            raise _Pause()

    print('[R3m-E Gate2] real Proposal and mid-Shadow checkpoint', flush=True)
    try:
        execute(engine_c, runtime, anchor_view, ident, work, historical, research, 'C', 'P1', device, pause)
    except _Pause:
        pass
    if pause_receipt is None:
        raise RuntimeError('GATE2_MID_TRANSACTION_PAUSE_NOT_REACHED')
    paused_state = Store(paused, ident).verified(pause_receipt)
    if not paused_state['aux']['transaction'] or paused_state['aux']['transaction']['done'] != 3:
        raise RuntimeError('GATE2_SHADOW_STATE_NOT_PERSISTED')
    execute(engine_c, runtime, anchor_view, ident, work, historical, research, 'C', 'P1', device)
    expected = Store(work, ident).verified(read(Store(work, ident).receipt_file(11)))
    expected_receipt = export_boundary(work, root / 'relay' / 'expected', ident, 11)
    event = expected['aux']['transactions']
    if len(event) != 1 or event[0]['at'] != 1 or expected['aux']['main_updates'] != 11:
        raise RuntimeError('GATE2_TRANSACTION_BUDGET_MISMATCH')
    atomic(root / 'relay' / 'PROPOSAL_TRACE.json', event[0])

    # Deliberately different Connection and forced rejection exercise the real commit/rollback
    # branch. This injected candidate is never presented as a Trust Region-approved candidate.
    def fault_proposal(tr, seen, path, ident, persist):
        connection = copy.deepcopy(tr.model.connection.state_dict())
        key = next(k for k, tensor in connection.items() if tensor.is_floating_point() and tensor.numel())
        connection[key].view(-1)[0] += 0.125
        if fingerprint(connection) == fingerprint(tr.model.connection.state_dict()):
            raise RuntimeError('GATE2_FAULT_CANDIDATE_UNCHANGED')
        return connection, {'effective': True, 'action_change': None,
                            'scope': 'ENGINEERING_FAULT_INJECTION_NOT_A_TRUST_REGION_RESULT'}

    quality_calls = []
    def reject(model, seen):
        quality_calls.append(len(quality_calls))
        return float(len(quality_calls))  # old=1, new=2, hence rejection with unchanged threshold.

    print('[R3m-E Gate2] injected rollback and unchanged-Connection control', flush=True)
    fault_c = engineering_config(c, fault=True)
    reject_run = make_runner(c, batch_provider, fault_proposal, reject)
    rejected, rejected_aux = reject_run(fault_c, runtime, anchor_view, dict(identity, seed=parent.model.seed, branch='fault-rejection'),
                                        root / 'working' / 'rejection', historical, research, 'C', 'P1', device)
    rejected_state = copy.deepcopy(rejected.state())
    stable_run = make_runner(c, batch_provider)
    stable, stable_aux = stable_run(fault_c, runtime, anchor_view, dict(identity, seed=parent.model.seed, branch='stable-control'),
                                    root / 'working' / 'stable', historical, research, 'C', 'STABLE', device)
    if rejected_aux['transactions'][0]['accepted'] or rejected_aux['transactions'][0]['ghost']:
        raise RuntimeError('GATE2_REJECTION_NOT_EXERCISED')
    rollback = compare_states(stable.state(), rejected_state, cuda=str(device).startswith('cuda'))
    return {
        'status': 'LOCAL_GATE2_PASS_CROSS_SESSION_PENDING',
        'actual_proposal': {'accepted': event[0]['accepted'], 'ghost': event[0]['ghost'],
                            'effective': event[0]['trace']['effective'],
                            'scientific_interpretation': 'ENGINEERING_ONLY_NO_GAIN_REQUIRED'},
        'pause_receipt': pause_receipt, 'expected_receipt': expected_receipt,
        'branch_identity': ident, 'rollback': rollback,
        'main_updates': 11, 'actual_shadow_updates': expected['aux']['shadow_updates'],
        'fault_control_updates': 5, 'local_optimizer_updates': 26,
        'proposal_optimizer_updates': expected['aux']['proposal_optimizer_updates'],
        'production_updates': 0,
        'paused_transaction_done': 3,
        'historical_parent_files_modified': False,
        'stage_a': True,
    }


def resume_and_compare(c, runtime, parent, historical, research, relay, identity, device,
                       batch_provider=None):
    relay = Path(relay)
    review = read(relay / 'LOCAL_GATE2.json')
    ident = review['branch_identity']
    paused_root = relay / 'paused'
    expected_store = Store(relay / 'expected', ident)
    expected = expected_store.verified(review['expected_receipt'])
    resumed, aux = make_runner(c, batch_provider)(engineering_config(c), runtime, parent,
        ident, paused_root, historical, research, 'C', 'P1', device)
    match = compare_states(expected['trainer'], resumed.state(), cuda=str(device).startswith('cuda'))
    if aux['main_updates'] != 11 or aux['transaction'] is not None or len(aux['transactions']) != 1:
        raise RuntimeError('GATE2_RESUMED_TRANSACTION_INCOMPLETE')
    for key in ('data_rng', 'route_rng', 'cf_rng', 'python_rng', 'numpy_rng', 'global_cpu_rng', 'global_cuda_rng'):
        if fingerprint(expected['trainer'][key]) != fingerprint(resumed.state()[key]):
            raise RuntimeError('GATE2_RESUME_RANDOM_STREAM_CHANGED:' + key)
    for key in ('seen', 'schedule', 'main_updates', 'shadow_updates', 'discarded_shadow_updates', 'proposal_optimizer_updates', 'frozen_connection'):
        if fingerprint(expected['aux'][key]) != fingerprint(aux[key]):
            raise RuntimeError('GATE2_RESUME_AUXILIARY_STATE_CHANGED:' + key)
    if expected['aux']['transactions'][0]['accepted'] != aux['transactions'][0]['accepted']:
        raise RuntimeError('GATE2_RESUME_DECISION_CHANGED')
    return dict(status='GATE2_PASS', comparison=match, resumed_from_relative_step=4,
                completed_relative_step=11, resumed_main_updates=7, resumed_shadow_updates=7,
                production_updates=0, full_training_authorized=False,
                confirmation='SEALED', study_checkpoints_written=False)


def publish_review(backend, export_root, output, identity, pointer, result, writer=None):
    """Retain a retryable upload job so network failures do not repeat GPU work."""
    output = Path(output)
    export_root = Path(export_root).resolve()
    export_root.relative_to(output.resolve())
    writer = writer or uuid.uuid4().hex
    snapshot = output / ('gate2-upload-review-' + canonical(result) + '.json')
    atomic(snapshot, result)
    journal = output / ('gate2-upload-' + identity['run_id'] + '.json')
    record = dict(identity=identity, export_root=str(export_root), pointer=pointer,
                  writer=writer, result_file=str(snapshot.resolve()), result_sha256=digest(snapshot),
                  complete=False)
    atomic(journal, record)
    ref = publish_segment(backend, export_root, identity, writer, pointer_path=pointer)
    atomic(journal, dict(record, complete=True, hf_reference=ref))
    return ref


def retry_upload(backend, output, identity):
    journal = Path(output) / ('gate2-upload-' + identity['run_id'] + '.json')
    if not journal.is_file():
        return None
    record = read(journal)
    if record['identity'] != identity:
        raise RuntimeError('GATE2_UPLOAD_JOB_IDENTITY_CHANGED')
    if record['complete']:
        return None
    result_file = Path(record['result_file']).resolve()
    result_file.relative_to(Path(output).resolve())
    if digest(result_file) != record['result_sha256']:
        raise RuntimeError('GATE2_PENDING_REVIEW_HASH_CHANGED')
    result = read(result_file)
    ref = publish_review(backend, record['export_root'], output, identity,
                         record['pointer'], result, record['writer'])
    result = dict(result, hf_reference=ref, upload_retried_without_optimizer_updates=True)
    atomic(Path(output) / 'GATE2_REVIEW.json', result)
    return result


def run(c, output, assets_dir, device='cuda:0'):
    from minicells_artifacts.hf_storage import HubBackend
    from .runner import _token

    if not torch.cuda.is_available() or not device.startswith('cuda:'):
        raise RuntimeError('GATE2_REQUIRES_CUDA_T4_X2_FOR_HISTORICAL_LAYOUT')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('GATE2_REQUIRES_TWO_VISIBLE_GPUS_FOR_ORIGINAL_CUDA_RNG_LAYOUT')
    _token()
    backend = HubBackend(c['baseline']['hf_reference']['hf_repo'])
    identity = {'experiment_id': 'R3ME-GATE2-ENGINEERING', 'source_sha': source_sha(),
                'config_sha256': canonical(c), 'protocol_sha256': canonical(PROTOCOL)}
    identity['run_id'] = canonical(identity)
    prefix = 'engineering/r3me-gate2/' + identity['run_id']
    container = session_identity()
    revision = backend.info().sha
    verified = backend.optional(prefix + '/verified.json', revision)
    if verified:
        ref = dict(verified, hf_repo=backend.repo, hf_revision=revision)
        manifest = verified_manifest(backend, ref, identity)
        row = manifest['files']['GATE2_REVIEW.json']
        path = backend.get(row['object'], revision)
        if digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise RuntimeError('GATE2_REVIEW_HASH_MISMATCH')
        result = read(path)
        if result.get('status') != 'GATE2_PASS':
            raise RuntimeError('GATE2_PRIOR_REVIEW_NOT_PASS')
        result['hf_reference'] = ref
        atomic(Path(output) / 'GATE2_REVIEW.json', result)
        return result
    retried = retry_upload(backend, output, identity)
    if retried is not None:
        return retried
    prior = backend.optional(prefix + '/latest.json', revision)
    ref = dict(prior, hf_repo=backend.repo, hf_revision=revision) if prior else None
    if ref:
        meta = verified_manifest(backend, ref, identity)['files']['LOCAL_GATE2.json']
        path = backend.get(meta['object'], revision)
        if digest(path) != meta['sha256'] or path.stat().st_size != meta['bytes']:
            raise RuntimeError('GATE2_STAGE_A_PROOF_HASH_MISMATCH')
        initial = read(path)
        if initial['container']['container_lifetime_sha256'] == container['container_lifetime_sha256']:
            return {'status': 'GATE2_A_SAVED_NEW_SESSION_REQUIRED', 'hf_reference': ref,
                    'next': 'Save Version; start a new Kaggle Session; Run All in the same Notebook. No Input mount or version number is required.',
                    'training_updates_added': 0, 'full_training_authorized': False}
    run_root = Path(output) / ('engineering-' + uuid.uuid4().hex)
    run_root.mkdir(parents=True, exist_ok=False)
    runtime = anchor.runtime(c)
    historical, research, manifest, _ = data.prepare(c, runtime, backend, assets_dir)
    parent, proof = anchor.load(c, backend, run_root / 'parent', PROTOCOL['seed'], device)
    before = fingerprint(parent.model.state_dict()), fingerprint(parent.cell_opt.state_dict())
    begin = time.monotonic()
    if ref:
        if initial['input_manifest_sha256'] != canonical(manifest):
            raise RuntimeError('GATE2_RESTORED_INPUT_IDENTITY_CHANGED')
        relay = run_root / 'restored-relay'
        restore_segment(backend, ref, relay, identity, uuid.uuid4().hex)
        result = resume_and_compare(c, runtime, parent, historical, research, relay, identity, device)
        result.update(identity=identity, protocol=PROTOCOL, stage_a_container=initial['container'],
                      stage_b_container=container, hf_parent_reference=ref,
                      actual_cross_session_verified=True,
                      provenance_scope='DISTINCT_CONTAINER_LIFETIMES_AND_PINNED_HF_BYTES',
                      seconds=time.monotonic()-begin)
        destination = run_root / 'verification-report'
        atomic(destination / 'GATE2_REVIEW.json', result)
        atomic(Path(output) / 'GATE2_REVIEW.json', result)
        uploaded = publish_review(backend, destination, output, identity, prefix + '/verified.json', result)
    else:
        result = run_transaction_checks(c, runtime, parent, historical, research,
                                         run_root, identity, device)
        if before != (fingerprint(parent.model.state_dict()), fingerprint(parent.cell_opt.state_dict())):
            raise RuntimeError('GATE2_MUTATED_PARENT_MODEL_OR_OPTIMIZER')
        result.update(identity=identity, protocol=PROTOCOL, container=container,
                      parent_receipt=proof['receipt'], input_manifest_sha256=canonical(manifest),
                      seconds=time.monotonic()-begin, actual_cross_session_verified=False)
        result['next'] = 'Save Version; start a new Kaggle Session; Run All in the same Notebook. It automatically restores the mid-Shadow checkpoint from private HF.'
        atomic(run_root / 'relay' / 'LOCAL_GATE2.json', result)
        atomic(Path(output) / 'GATE2_REVIEW.json', result)
        uploaded = publish_review(backend, run_root / 'relay', output, identity, prefix + '/latest.json', result)
    if before != (fingerprint(parent.model.state_dict()), fingerprint(parent.cell_opt.state_dict())):
        raise RuntimeError('GATE2_MUTATED_PARENT_MODEL_OR_OPTIMIZER')
    result['hf_reference'] = uploaded
    atomic(Path(output) / 'GATE2_REVIEW.json', result)
    return result
