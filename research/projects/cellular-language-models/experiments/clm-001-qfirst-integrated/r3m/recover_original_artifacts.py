"""Read-only fixed-revision metadata recovery; no arrays, checkpoints or model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / 'shared/python'))
from minicells_artifacts.hf_storage import HubBackend, error_code  # noqa: E402

BASE = Path(__file__).resolve().parent
TOKENIZER = '612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff'
NAMES = ('manifest.json', 'tokenizer.json', 'windows.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(backend, reference, expected_manifest, output):
    """Backend get only. Check identity before downloading each permitted object."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='minicells.r3m.artifact-recovery.v1', status='BLOCKED',
                  auditor_sha256=digest(__file__),
                  hf_repo=reference['hf_repo'], hf_revision=reference['hf_revision'],
                  bundle_path=reference['bundle_path'], files={},
                  gpu_training_started=False, model_quality_reads=0, remote_writes=0,
                  registration_changed=False, window_gate='BLOCKED_PENDING_INDEPENDENT_AUDIT')
    try:
        bundle_path = backend.get(reference['bundle_path'], reference['hf_revision'])
        if digest(bundle_path) != reference['bundle_sha256']:
            raise ValueError('BUNDLE_SHA256_MISMATCH')
        bundle = json.loads(bundle_path.read_text())
        identity = bundle['identity']
        if identity['tokenizer_sha256'] != TOKENIZER or identity['assets_manifest_sha256'] != expected_manifest:
            raise ValueError('BUNDLE_IDENTITY_MISMATCH')
        shutil.copyfile(bundle_path, output / 'original-bundle.json')
        result['bundle_sha256'] = digest(bundle_path)
        for name in NAMES:
            row = bundle['files']['assets/' + name]
            if name == 'tokenizer.json' and row['sha256'] != TOKENIZER:
                raise ValueError('TOKENIZER_REFERENCE_MISMATCH')
            if name == 'manifest.json' and row['sha256'] != expected_manifest:
                raise ValueError('MANIFEST_REFERENCE_MISMATCH')
            source = backend.get(row['object'], reference['hf_revision'])
            if digest(source) != row['sha256']:
                raise ValueError('OBJECT_SHA256_MISMATCH')
            if name == 'manifest.json':
                manifest = json.loads(source.read_text())
                if manifest.get('confirmation_materialized') is not False:
                    raise ValueError('CONFIRMATION_BOUNDARY_UNVERIFIED')
                if any(manifest['files'].get(n) != bundle['files']['assets/' + n]['sha256'] for n in NAMES[1:]):
                    raise ValueError('MANIFEST_OBJECT_BINDING_MISMATCH')
            if name == 'windows.json':
                windows = json.loads(source.read_text())
                # Historical metadata must expose exactly the already-authorized old roles.
                roles = {key for key, value in windows.items() if isinstance(value, dict) and 'raw_story_ids' in value}
                if roles != {'train', 'calibration', 'admission', 'development'} or 'confirmation' in windows:
                    raise ValueError('WINDOW_ROLE_BOUNDARY_UNVERIFIED')
                result['historical_window_metadata_roles'] = sorted(roles)
                result['historical_window_interpretation'] = 'ELIGIBLE_AND_EVALUATION_STARTS_NOT_ACTUAL_OPTIMIZER_SAMPLING_PROOF'
            shutil.copyfile(source, output / name)
            result['files'][name] = dict(object=row['object'], sha256=digest(source), bytes=source.stat().st_size)
        result['status'] = 'ORIGINAL_TOKENIZER_AND_METADATA_HASH_VERIFIED'
        result['tokenizer_gate'] = 'PASS_EXACT_ORIGINAL_BYTES'
    except Exception as error:
        result['error'] = error_code(error)
        # Only locally defined constant codes may be recorded, never remote error text.
        if isinstance(error, ValueError):
            result['local_failure_code'] = 'PINNED_METADATA_VALIDATION_FAILED'
    (output / 'RECOVERY_PROOF.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config_path = BASE.parent / 'r3h/config/frozen.json'
    original = json.loads(config_path.read_text())['engine']['input']
    reference = original['asset_bundle_ref']
    # HubBackend uses the existing HF_TOKEN, never an anonymous fallback or new credentials.
    try:
        backend = HubBackend(reference['hf_repo'])
    except Exception as error:
        print(json.dumps(dict(status='BLOCKED_AUTHORIZATION_REQUIRED', **error_code(error))))
        raise SystemExit(1) from None
    result = recover(backend, reference, original['assets_manifest_sha256'], args.output)
    print(json.dumps(result, indent=2))
    if result['status'] != 'ORIGINAL_TOKENIZER_AND_METADATA_HASH_VERIFIED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
