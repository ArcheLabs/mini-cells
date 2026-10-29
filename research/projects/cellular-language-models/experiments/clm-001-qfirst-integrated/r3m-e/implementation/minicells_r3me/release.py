"""Read the completed Gate 2 proof; no repeated GPU preflight or test suite."""
from .common import BASE, canonical, digest, read
from .protocol import validate
from minicells_r3c_moe_bridge.development import verified_manifest


def verify(c, backend):
    validate(c)
    if not c['authorization']['full_training']:
        raise RuntimeError('R3ME_EXPLORATORY_TRAINING_NOT_RELEASED')
    release = c['release']
    local = read(BASE / 'release/gate2/SESSION_B.json')
    if canonical(local) != release['gate2_review_sha256']:
        raise RuntimeError('R3ME_LOCAL_GATE2_EVIDENCE_CHANGED')
    ref = release['hf_reference']
    expected_identity = local['identity']
    if expected_identity['source_sha'] != release['gate2_source_sha'] or expected_identity['config_sha256'] != release['parent_config_sha256']:
        raise RuntimeError('R3ME_GATE2_SOURCE_CONFIG_MISMATCH')
    manifest = verified_manifest(backend, ref, expected_identity)
    row = manifest['files']['GATE2_REVIEW.json']
    path = backend.get(row['object'], ref['hf_revision'])
    if digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
        raise RuntimeError('R3ME_GATE2_REMOTE_BYTES_CHANGED')
    remote = read(path)
    # User log adds the publication reference after upload; compare the actual review fields.
    if any(remote.get(k) != v for k, v in local.items() if k != 'hf_reference'):
        raise RuntimeError('R3ME_REMOTE_GATE2_REVIEW_CHANGED')
    comparison = remote.get('comparison', {})
    if remote.get('status') != 'GATE2_PASS' or remote.get('actual_cross_session_verified') is not True or not comparison.get('exact_rng_cursor_identity') or not comparison.get('model_optimizer_match'):
        raise RuntimeError('R3ME_GATE2_NOT_PASSED')
    return dict(status='EXPLORATORY_RELEASE_VERIFIED', hf_reference=ref,
                gate2_source_sha=release['gate2_source_sha'], gate2_run_id=release['gate2_run_id'],
                repeated_gpu_updates=0, formal_confirmation_authorized=False)
