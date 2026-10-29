"""Amendment003 isolated content audit. No model, quality evaluation or registration."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / 'shared/python'))
from minicells_data_registry import ExclusionIndex, select  # noqa: E402
from audit_text_boundaries import iter_text  # noqa: E402

BASE = Path(__file__).resolve().parent
REG = BASE / 'registration'
OUT = REG / 'amendment-003'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    p = OUT / name
    if p.exists():
        raise RuntimeError('IMMUTABLE_OUTPUT_EXISTS: ' + str(p))
    p.write_text(json.dumps(value, indent=2) + '\n')


def main():
    protocol = json.loads((OUT / 'SELECTOR_PROTOCOL.json').read_text())
    old = json.loads((REG / 'amendment-002/DATA_BOUNDARY_PROOF.json').read_text())
    ledger = json.loads((REG / 'GLOBAL_DATASET_USAGE_LEDGER.json').read_text())
    paths = [Path('/tmp/r3m-tinystories-shard.parquet'), Path('/tmp/r3m-validation.parquet')]
    for p, expected in zip(paths, old['source_files']):
        if sha(p) != expected['sha256']:
            raise RuntimeError('SOURCE_SHA_MISMATCH')
    db = Path('/tmp/r3m-amendment003-exclusions.sqlite')
    if db.exists():
        raise RuntimeError('INDEX_ALREADY_EXISTS')
    index = ExclusionIndex(db)
    # Conservative full raw prefix includes skipped/partially consumed terminal records.
    for raw, text in iter_text(paths[0]):
        if raw <= 476900:
            index.add(text, '007:exact-token-stream-reproduced:train', 'train_corpus_superset', raw)
        if 50000 <= raw < 61024:
            index.add(text, 'ledger:declared-role-union:revision-pending', 'development_or_reserved_confirmation', raw)
        if 500000 <= raw < 500512:
            index.add(text, 'amendment002:pending-trigger-pilot', 'pending_candidate', raw)
    for raw, text in iter_text(paths[1], 4396):
        index.add(text, '007:exact-token-stream-reproduced:validation', 'validation_corpus', raw)
    index.db.commit()
    candidates = select(iter_text(paths[0]), index, protocol['domain'], protocol['count'])
    index.close()
    unknown = [x for x in ledger['intervals'] if not x.get('revision')]
    write('HISTORICAL_COVERAGE.json', dict(
        schema='minicells.exclusion-coverage.v1', claims=ledger['intervals'],
        unknown_revision_claims=unknown, unknown_scope=ledger['reservation_markers'],
        ledger_sha256=sha(REG / 'GLOBAL_DATASET_USAGE_LEDGER.json'),
        index=dict(path=str(db), sha256=sha(db), bytes=db.stat().st_size,
                   durability='LOCAL_ONLY_NOT_HF_PUBLISHED'),
        coverage='PARTIAL_CONSERVATIVE_SUPERSET_NOT_COMPLETE_GLOBAL_COVERAGE',
        known_sources=old['source_files'],
        provenance='SQLite rows retain source, role and raw index; ledger retains individual declarations. '
                   'Union does not certify each historical claim or actual optimizer-observed windows.'))
    write('MAIN_CANDIDATE_INDICES.json', dict(status='PROVISIONAL_NOT_QUALIFIED_NOT_REGISTERED',
          selector_sha256=sha(OUT / 'SELECTOR_PROTOCOL.json'), records=candidates))
    write('DATA_BOUNDARY_PROOF.json', dict(
        schema='minicells.r3m.data-boundary-proof.v3', status='BLOCKED_UNKNOWN_EVIDENCE',
        selector_frozen_commit='c3da320d', source_baseline='23f5fd87fa6114187daf321490298150a7a85f62',
        selector_sha256=sha(OUT / 'SELECTOR_PROTOCOL.json'),
        auditor_sha256=sha(__file__), shared_index_sha256=sha(ROOT / 'shared/python/minicells_data_registry/__init__.py'),
        candidate_manifest_sha256=sha(OUT / 'MAIN_CANDIDATE_INDICES.json'), candidate_count=len(candidates),
        gates=dict(known_index_content_exclusion='PASS_WITHIN_RECORDED_SCOPE',
                   global_content_isolation='UNKNOWN_BLOCKED', tokenizer_identity='UNRECOVERED_BLOCKED',
                   historical_exact_windows='UNRECOVERED_BLOCKED', scientific_review='PENDING'),
        unresolved_revision_claims=len(unknown),
        tokenizer=dict(expected_sha256='612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff',
                       original_bundle_revision='828585f1ff93420aa4485c725211ed28f70d15f3',
                       recovery_http_status=401, replacement_trained=False),
        windows='No exact historical sampled starts recovered. Content hashes do not certify token-window nonoverlap.',
        trigger_pilot_status='PENDING_NOT_REGISTERED', previous_main_range='ABANDONED_NOT_DELETED_FROM_HISTORY',
        prior_proofs={str(p.relative_to(REG)): sha(p) for p in
                      (REG / 'DATA_BOUNDARY_PROOF.json', REG / 'amendment-002/DATA_BOUNDARY_PROOF.json')},
        scientific_config_sha256=sha(BASE / 'config/pilot.json'),
        scientific_config_modified=False, registration_changed=False, quality_evaluations=0, gpu_training_started=False))
    print('AMENDMENT003_BLOCKED; 512 provisional content-selected indices; no registration')


if __name__ == '__main__':
    main()
