"""Read-only, CPU checkpoint RNG extraction for the R3m parent arm B-H only."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / 'run.py'), run_name='sampling_boundary_imports_only')
from minicells_artifacts.hf_storage import HubBackend, error_code  # noqa: E402
from minicells_r3c_moe_bridge.checkpoint import Store  # noqa: E402
from minicells_r3c_moe_bridge.development import verified_manifest  # noqa: E402
from minicells_runtime.store import digest, read, safe  # noqa: E402
from replay_sampling import canonical, sha  # noqa: E402
import numpy as np  # noqa: E402


def recover(backend, output):
    c = read(BASE.parent / 'r3h/config/frozen.json')
    baseline = read(BASE.parent / 'r3i/config/frozen.json')['baseline']
    ref = baseline['hf_reference']
    ident = {k: baseline[k] for k in ('experiment_id', 'source_sha', 'config_sha256', 'run_id')}
    if canonical(c) != ident['config_sha256']:
        raise ValueError('FROZEN_HISTORICAL_CONFIG_MISMATCH')
    manifest = verified_manifest(backend, ref, ident)
    result = dict(schema='minicells.r3m.original-checkpoint-sampler-evidence.v1',
                  extractor_sha256=sha(__file__), hf_reference=ref,
                  original_identity=ident, arm_scope=['B-H'], jobs=[], failures=[],
                  other_r3h_arms='NOT_AUDITED', r3i_to_r3l_branches='NOT_AUDITED',
                  numpy_extraction_runtime=np.__version__, remote_writes=0,
                  model_instantiations=0, optimizer_updates=0, gpu_work=0)
    requested = (0, 100, 200, 300, 400, 500)
    for seed in c['seeds']:
        identity = dict(ident, seed=seed, architecture='B', training_router='H')
        prefix = 'jobs/seed-%s-arm-B-H/' % seed
        job = dict(seed=seed, arm='B-H', config_sha256=ident['config_sha256'],
                   sampler_source_sha256=sha(BASE.parent / 'r3/implementation/clm001_r3/data.py'),
                   historical_numpy_version=None,
                   numpy_version_evidence='ORIGINAL_TRAINER_STATE_HAS_NO_PRODUCER_VERSION_FIELD', boundaries=[])
        for step in requested:
            try:
                name = Store(prefix, identity).receipt_file(step).as_posix()
                item = manifest['files'][name]
                receipt_path = backend.get(item['object'], ref['hf_revision'])
                if digest(receipt_path) != item['sha256'] or receipt_path.stat().st_size != item['bytes']:
                    raise ValueError('RECEIPT_OBJECT_CHANGED')
                receipt = read(receipt_path)
                if receipt['identity'] != identity or receipt['step'] != step:
                    raise ValueError('ORIGINAL_RECEIPT_IDENTITY_CHANGED')
                checkpoint = manifest['files'][prefix + receipt['file']]
                if checkpoint['sha256'] != receipt['sha256'] or checkpoint['bytes'] != receipt['bytes']:
                    raise ValueError('MANIFEST_CHECKPOINT_BINDING_CHANGED')
                print(json.dumps(dict(seed=seed, arm='B-H', boundary=step,
                                      checkpoint_bytes=checkpoint['bytes'], action='CPU_RNG_STATE_EXTRACTION')), flush=True)
                obj = backend.get(checkpoint['object'], ref['hf_revision'])
                if digest(obj) != checkpoint['sha256'] or obj.stat().st_size != checkpoint['bytes']:
                    raise ValueError('CHECKPOINT_OBJECT_CHANGED')
                with tempfile.TemporaryDirectory(prefix='r3m-boundary-') as temp:
                    target = safe(Path(temp), receipt['file'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(obj, target)
                    # Existing Store verifies the complete original state fingerprint on CPU.
                    state = Store(temp, identity).verified(receipt)
                if state['step'] != step or state['data_cursor'] != step * c['training']['tokens_per_step']:
                    raise ValueError('ORIGINAL_TRAINING_CURSOR_CHANGED')
                job['boundaries'].append(dict(step=step, data_cursor=state['data_cursor'],
                      data_rng=state['data_rng'], checkpoint_sha256=checkpoint['sha256'],
                      receipt_sha256=item['sha256'], original_state_fingerprint=receipt['state_fingerprint'],
                      checkpoint_bytes_verified=True, receipt_identity_verified=True, full_state_fingerprint_verified=True))
                del state
            except Exception as error:
                failure = dict(seed=seed, arm='B-H', step=step, **error_code(error))
                failure['status'] = 'BOUNDARY_NOT_VERIFIED'
                result['failures'].append(failure)
                print(json.dumps(failure), flush=True)
        result['jobs'].append(job)
    result['status'] = 'BOUNDARY_EVIDENCE_EXTRACTED_REVIEW_REQUIRED' if not result['failures'] else 'PARTIAL_BOUNDARY_EVIDENCE'
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('IMMUTABLE_BOUNDARY_EVIDENCE_EXISTS')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    ref = read(BASE.parent / 'r3i/config/frozen.json')['baseline']['hf_reference']
    result = recover(HubBackend(ref['hf_repo']), a.output)
    print(json.dumps(dict(status=result['status'], boundaries=sum(len(j['boundaries']) for j in result['jobs']),
                          failures=len(result['failures']), window_gate='BLOCKED')))


if __name__ == '__main__':
    main()
