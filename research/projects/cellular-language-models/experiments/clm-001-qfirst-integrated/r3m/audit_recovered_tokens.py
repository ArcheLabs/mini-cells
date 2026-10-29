"""Rebuild original historical token assets on CPU; never construct Main or Confirmation."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from audit_recovered_metadata import audit, sha, BASE, ROOT, ROLES


def verify_tokens(folder, shard):
    folder = Path(folder)
    metadata = audit(folder, shard)
    tokenizer_path = folder / 'tokenizer.json'
    if not tokenizer_path.is_file():
        raise ValueError('ORIGINAL_TOKENIZER_BYTES_REQUIRED')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    assert tokenizer.get_vocab_size() == 4096
    manifest = json.loads((folder / 'manifest.json').read_text())
    windows = json.loads((folder / 'windows.json').read_text())
    historical_encoder = ROOT / 'research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/data.py'
    spec = importlib.util.spec_from_file_location('r3m_isolated_original_encoder', historical_encoder)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    needed = {i for role in ROLES for i in windows[role]['raw_story_ids']}
    texts, cursor = {}, 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=4096, columns=['text']):
        for i, text in enumerate(batch.column(0).to_pylist(), start=cursor):
            if i in needed:
                texts[i] = text
        cursor += len(batch)
        if cursor > max(needed):
            break
    assert set(texts) == needed
    results = {}
    with tempfile.TemporaryDirectory(prefix='r3m-token-integrity-') as temp:
        for role in ROLES:
            w = windows[role]
            ordered = [texts[i] for i in w['raw_story_ids']]
            tokens, starts, groups, hashes = original.encode_stories(ordered, tokenizer, 64)
            offsets = np.cumsum([0] + [len(tokenizer.encode(t).ids) + 1 for t in ordered]).tolist()
            assert offsets == w['story_offsets']
            assert starts == w['starts'] and groups == w['groups'] and hashes == w['story_hashes']
            path = Path(temp) / (role + '.npy')
            np.save(path, tokens)
            actual = sha(path)
            assert actual == manifest['files'][role + '.npy'], role + '_TOKEN_ASSET_CHANGED'
            results[role] = dict(status='EXACT_ORIGINAL_TOKEN_ASSET_REPRODUCED',
                                 tokens=int(tokens.size), dtype=str(tokens.dtype),
                                 npy_sha256=actual, bytes=path.stat().st_size,
                                 story_offsets='EXACT_MATCH_FROM_ORIGINAL_TOKENIZER',
                                 starts_groups_story_hashes='EXACT_MATCH', windows=len(starts))
    metadata.update(
        schema='minicells.r3m.original-tokenizer-window-reconstruction.v1',
        status='ORIGINAL_TOKENIZER_AND_HISTORICAL_ELIGIBLE_WINDOWS_INDEPENDENTLY_RECONSTRUCTED_REGISTRATION_BLOCKED',
        auditor_sha256=sha(__file__), metadata_auditor_sha256=sha(BASE / 'audit_recovered_metadata.py'),
        original_encoder=dict(path=str(historical_encoder.relative_to(ROOT)), sha256=sha(historical_encoder)),
        input_tokenizer=dict(sha256=sha(tokenizer_path), bytes=tokenizer_path.stat().st_size, vocab=4096),
        tokenizer_gate='PASS_EXACT_ORIGINAL_BYTES_INDEPENDENTLY_HASHED',
        original_token_assets=results,
        historical_eligible_windows_gate='PASS_EXACT_HISTORICAL_ASSET_RECONSTRUCTION_ONLY',
        limitations=['This verifies the original four-role R3 assets, not actual optimizer-consumed start logs.',
                     'Actual R3k/R3l research-role membership remains unrecovered.',
                     'Main whole-story nonoverlap is not a cross-fragment or cross-story window isolation proof.',
                     'Path B has not been approved; unresolved global reservations and coverage remain.'],
        window_gate='BLOCKED_PENDING_GLOBAL_ISOLATION_PROOF_AND_SCIENTIFIC_REVIEW',
        main_arrays_materialized=False, confirmation_materialized=False,
        reconstructed_historical_arrays='TEMPORARY_CPU_ONLY_REMOVED_AFTER_HASH_CHECK',
        optimizer_updates=0)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--shard', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('IMMUTABLE_REVIEW_OUTPUT_EXISTS')
    result = verify_tokens(args.input, args.shard)
    args.output.mkdir(parents=True)
    shutil.copyfile(args.input / 'tokenizer.json', args.output / 'tokenizer.json')
    (args.output / 'TOKENIZER_WINDOW_RECONSTRUCTION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], tokenizer_gate=result['tokenizer_gate'],
                          window_gate=result['window_gate'], roles=result['original_token_assets']), indent=2))


if __name__ == '__main__':
    main()
