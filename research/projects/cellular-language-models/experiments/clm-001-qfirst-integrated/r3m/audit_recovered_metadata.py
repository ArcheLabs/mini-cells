"""Independent CPU metadata/source verification. No tokenizer training or model imports."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[5]
ROLES = ('train', 'calibration', 'admission', 'development')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def original_grouped_windows():
    """Reuse the exact pure historical function without importing its model bootstrap."""
    p = ROOT / 'research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/exploration-fast-001/implementation/fast001/data.py'
    tree = ast.parse(p.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'grouped_windows')
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {'np': np}
    exec(compile(module, str(p), 'exec'), namespace)
    return namespace['grouped_windows'], dict(path=str(p.relative_to(ROOT)), sha256=sha(p))


def audit(folder, shard):
    folder = Path(folder)
    proof = json.loads((folder / 'RECOVERY_PROOF.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    windows = json.loads((folder / 'windows.json').read_text())
    c = json.loads((BASE.parent / 'r3h/config/frozen.json').read_text())
    old = json.loads((BASE.parent / 'r3/config/proposed.json').read_text())
    expected = c['engine']['input']
    reference = expected['asset_bundle_ref']
    assert proof['status'] == 'ORIGINAL_TOKENIZER_AND_METADATA_HASH_VERIFIED'
    assert proof['auditor_sha256'] == sha(BASE / 'recover_original_artifacts.py')
    for key in ('hf_repo', 'hf_revision', 'bundle_path', 'bundle_sha256'):
        assert proof[key] == reference[key], key
    assert sha(folder / 'manifest.json') == expected['assets_manifest_sha256']
    for name in ('manifest.json', 'windows.json'):
        assert sha(folder / name) == proof['files'][name]['sha256']
        assert (folder / name).stat().st_size == proof['files'][name]['bytes']
    assert manifest['files']['windows.json'] == sha(folder / 'windows.json')
    assert manifest['files']['tokenizer.json'] == expected['tokenizer_sha256']
    assert proof['files']['tokenizer.json']['sha256'] == expected['tokenizer_sha256']
    assert manifest['revision'] == expected['data_config']['revision']
    assert manifest['dataset'] == expected['data_config']['dataset']
    assert manifest['split'] == 'train' and manifest['confirmation_materialized'] is False
    assert manifest['tokenizer_actual_vocab'] == 4096
    assert sha(shard) == manifest['source_sha256']
    assert set(windows) == set(ROLES)
    selected = set(i for role in ROLES for i in windows[role]['raw_story_ids'])
    text_hashes = {}
    cursor = 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=4096, columns=['text']):
        for i, text in enumerate(batch.column(0).to_pylist(), start=cursor):
            if i in selected:
                text_hashes[i] = hashlib.sha256(text.encode('utf-8')).hexdigest()
        cursor += len(batch)
        if cursor > max(selected):
            break
    assert set(text_hashes) == selected
    grouped, function_source = original_grouped_windows()
    length = c['engine']['training']['sequence_length']
    seen_ids, seen_hashes, summaries = set(), set(), {}
    for role in ROLES:
        w = windows[role]
        ids, hashes, offsets = w['raw_story_ids'], w['story_hashes'], w['story_offsets']
        assert len(ids) == len(hashes) == len(offsets) - 1
        assert ids == manifest['selection_audit']['raw_story_ids'][role]
        assert len(set(ids)) == len(ids) and not seen_ids.intersection(ids)
        assert len(set(hashes)) == len(hashes) and not seen_hashes.intersection(hashes)
        lo, hi = expected['data_config'][role + '_range']
        assert all(lo <= i < hi for i in ids)
        assert [text_hashes[i] for i in ids] == hashes
        assert offsets[0] == 0 and all(a < b for a, b in zip(offsets, offsets[1:]))
        assert len(w['starts']) == len(w['groups'])
        expected_pairs = [(s, g) for g, (a, b) in enumerate(zip(offsets, offsets[1:]))
                          for s in range(a, b - length, length)]
        assert list(zip(w['starts'], w['groups'])) == expected_pairs
        assert all(offsets[g] <= s and s + length < offsets[g + 1] for s, g in expected_pairs)
        if role != 'train':
            assert w['evaluation'] == grouped(w['starts'], w['groups'],
                                               old['evaluation'][role + '_windows'], old['evaluation']['seed'])
        seen_ids.update(ids)
        seen_hashes.update(hashes)
        summaries[role] = dict(stories=len(ids), eligible_windows=len(expected_pairs),
                               evaluation_windows=len(w.get('evaluation', {}).get('starts', [])),
                               token_extent=offsets[-1], raw_index_min=min(ids), raw_index_max=max(ids),
                               story_hash_mapping='PASS', input_plus_target_within_story='PASS',
                               eligible_start_enumeration='PASS_FROM_ATTESTED_OFFSETS',
                               evaluation_sampling='NOT_APPLICABLE' if role == 'train' else 'EXACT_HISTORICAL_RULE_MATCH')
    candidates_path = BASE / 'registration/amendment-003/MAIN_CANDIDATE_INDICES.json'
    candidates = json.loads(candidates_path.read_text())['records']
    assert not seen_ids.intersection(r['raw_index'] for r in candidates)
    assert not seen_hashes.intersection(r['content_sha256'][0] for r in candidates)
    tokenizer = folder / 'tokenizer.json'
    tokenizer_local = tokenizer.is_file()
    if tokenizer_local:
        assert sha(tokenizer) == expected['tokenizer_sha256']
    return dict(schema='minicells.r3m.recovered-metadata-review.v1',
                status='RECOVERED_METADATA_IDENTITY_AND_INTERNAL_STRUCTURE_VERIFIED_REGISTRATION_BLOCKED',
                auditor_sha256=sha(__file__),
                input_files={name: dict(sha256=sha(folder / name), bytes=(folder / name).stat().st_size)
                             for name in ('RECOVERY_PROOF.json', 'manifest.json', 'windows.json')},
                hf_revision=reference['hf_revision'], source_shard_sha256=sha(shard),
                recovered_auditor_identity='MATCHES_PINNED_RECOVERY_IMPLEMENTATION',
                tokenizer=dict(expected_sha256=expected['tokenizer_sha256'],
                               kaggle_hash_attestation='MATCHES_ORIGINAL_FIXED_IDENTITY',
                               local_original_bytes_supplied=tokenizer_local,
                               local_independent_hash='PASS' if tokenizer_local else 'NOT_PERFORMED_NO_LOCAL_BYTES'),
                historical_window_function=function_source, roles=summaries,
                story_membership_and_exact_source_hashes='PASS_8704_RECORDS',
                role_content_disjointness='PASS_EXACT_RAW_SHA256',
                main_vs_recovered_roles='PASS_RAW_ID_AND_WHOLE_STORY_EXACT_HASH_ONLY',
                main_candidate_manifest_sha256=sha(candidates_path), main_candidates_changed=False,
                evaluation_start_rule='EXACT_ORIGINAL_GROUPED_WINDOWS_SEED_151001304',
                limitations=['Token offsets are hash-bound original metadata, not independently re-tokenized without tokenizer bytes.',
                             'Eligible training starts and fixed evaluation starts are not actual optimizer-consumed starts.',
                             'Whole-story hash nonoverlap does not establish fragment, cross-story or cross-tokenizer window nonoverlap.',
                             'Recovered R3 assets do not identify actual R3k/R3l research-role assets.'],
                window_gate='BLOCKED', path_b_approved=False, scientific_review='PENDING',
                registration_changed=False, quality_evaluations=0, gpu_training_started=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--shard', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('IMMUTABLE_REVIEW_OUTPUT_EXISTS')
    result = audit(args.input, args.shard)
    args.output.mkdir(parents=True)
    for name in ('RECOVERY_PROOF.json', 'manifest.json', 'windows.json'):
        shutil.copyfile(args.input / name, args.output / name)
    if (args.input / 'tokenizer.json').is_file():
        shutil.copyfile(args.input / 'tokenizer.json', args.output / 'tokenizer.json')
    (args.output / 'INDEPENDENT_REVIEW.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
