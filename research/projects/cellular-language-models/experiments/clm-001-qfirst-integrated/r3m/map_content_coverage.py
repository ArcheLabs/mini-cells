"""Map each immutable ledger claim to actual evidence and conservative source coverage."""
from collections import Counter
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[5]
REG = BASE / 'registration'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    ledger_path = REG / 'GLOBAL_DATASET_USAGE_LEDGER.json'
    ledger = json.loads(ledger_path.read_text())
    closure_path = REG / 'artifact-recovery-001/HISTORICAL_CLAIM_CLASSIFICATION_INT64.json'
    closure = json.loads(closure_path.read_text())
    known = {identity(x['original_claim']): x for x in closure['claims']}
    candidates_path = REG / 'amendment-003/MAIN_CANDIDATE_INDICES.json'
    candidates = json.loads(candidates_path.read_text())['records']
    limits = dict(train=476901, validation=4396)
    rows, source_checks = [], {}
    for item in ledger['parsed_metadata_files']:
        p = ROOT / item['path']
        source_checks[item['path']] = dict(expected_sha256=item['sha256'], exists=p.is_file(),
                       sha256_matches=p.is_file() and sha(p) == item['sha256'])
    for claim in ledger['intervals']:
        key = identity(claim)
        row = dict(claim_id=key, original_claim=claim, evidence_integrity=source_checks[claim['path']],
                   source_union_raw_limit=limits.get(claim['split']), original_identity_rewritten=False)
        if key in known:
            status = known[key]['classification']
            if status == 'RECOVERED_MODEL_VISIBLE_STREAM_IDENTITY':
                row.update(coverage_class='REPRODUCED_MODEL_VISIBLE_STREAM_WITHIN_007_UNIVERSE',
                           prior_evidence=known[key]['evidence'],
                           exact_actual_membership_required_for_model_visible_superset=False,
                           raw_snapshot_origin='UNRECORDED_NOT_RETROACTIVELY_REWRITTEN')
            elif status == 'PREVIOUSLY_REPRODUCED_STREAM_REUSE':
                row.update(coverage_class='REUSED_007_FULL_CORPUS_IDENTITY',
                           proof_sha256=known[key]['proof_sha256'],
                           actual_optimizer_starts='NOT_NEEDED_TO_DESCRIBE_ELIGIBLE_CORPUS_UNION')
            else:
                row.update(coverage_class='FROZEN_LATER_ROLE_RANGE_CONTAINED_ACTUAL_ASSET_UNKNOWN',
                           intended_source_revision='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',
                           coverage_condition='Actual asset follows the fixed R3k/R3l loader source binding and declared raw ranges',
                           original_membership='UNKNOWN_NO_ACTUAL_ROLE_MANIFEST_RECOVERED')
        elif claim.get('revision') == 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64' and 0 <= claim['range'][0] <= claim['range'][1] <= limits.get(claim['split'], -1):
            row.update(coverage_class='PINNED_DECLARED_RANGE_WITHIN_CONSERVATIVE_UNIVERSE',
                       coverage_condition='Declared indexing/loader identity; declarations alone are not actual runtime membership proof')
        else:
            row.update(coverage_class='UNKNOWN_OR_OUTSIDE_SCANNED_UNIVERSE')
        row['scientific_coverage_pass_granted'] = False
        rows.append(row)
    unpinned = [r for r in rows if not r['original_claim'].get('revision')]
    reservations = []
    for item in ledger['reservation_markers']:
        reservations.append(dict(original_marker=item, status='UNKNOWN_UPPER_ENDPOINT',
                 candidate_ids_not_proven_outside_reserved_scope=[r['raw_index'] for r in candidates if r['raw_index'] >= item['value']],
                 interpretation='Do not assert that the whole tail is reserved or free; either assertion lacks evidence'))
    result = dict(schema='minicells.r3m.historical-source-coverage-map.v1',
      status='CONSERVATIVE_UNIVERSE_AVAILABLE_GLOBAL_COVERAGE_NOT_CLOSED',
      auditor_sha256=sha(__file__), ledger_sha256=sha(ledger_path), closure_sha256=sha(closure_path),
      candidate_manifest_sha256=sha(candidates_path), claims=rows,
      counts=dict(Counter(r['coverage_class'] for r in rows)),
      unpinned_claim_counts=dict(Counter(r['coverage_class'] for r in unpinned)),
      original_unpinned_claims=len(unpinned), source_evidence_files=len(source_checks),
      source_evidence_hash_failures=[k for k,v in source_checks.items() if not v['sha256_matches']],
      reservations=reservations,
      global_gaps=['Fourteen reservation markers lack upper endpoints; current Main IDs are not proven outside those scopes.',
                   'Six later-role actual source/membership artifacts remain missing; frozen code offers conditional source containment only.',
                   'Ledger covers tracked JSON/YAML declarations; unpublished manual/runtime choices and external datasets are not certified absent.',
                   'Encoded stream identity does not rewrite original raw snapshot declarations or establish arbitrary preprocessing equivalence.'],
      proof_b='PREPARED_FOR_REVIEW_NOT_APPROVED', sampling_replay_repeated=False,
      candidate_changes=0, scientific_registration='BLOCKED', gpu_work=0)
    out = REG / 'content-isolation-001/HISTORICAL_SOURCE_COVERAGE.json'
    if out.exists():
        raise FileExistsError('IMMUTABLE_COVERAGE_MAP_EXISTS')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(counts=result['counts'], unpinned=result['unpinned_claim_counts'],
                         evidence_hash_failures=len(result['source_evidence_hash_failures']), reservations=len(reservations))))


if __name__ == '__main__':
    main()
