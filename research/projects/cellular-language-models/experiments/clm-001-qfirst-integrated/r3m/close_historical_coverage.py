"""Replay unique historical token-stream prefixes, preserving original unknown Revision claims."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[5]
REG = BASE / 'registration'
OLD_TOKENIZER = 'becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2'
REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def replay(source, tokenizer, budgets):
    """One encode pass per split; capture hashes exactly at each token budget."""
    budgets = sorted(set(budgets))
    result, hasher, total, count, cursor = {}, hashlib.sha256(), 0, 0, 0
    eos = tokenizer.token_to_id('<eos>')
    if eos is None:
        raise ValueError('MISSING_HISTORICAL_EOS')
    for batch in pq.ParquetFile(source).iter_batches(batch_size=1024, columns=['text']):
        texts = batch.column(0).to_pylist()
        valid = [(cursor + i, text.strip()) for i, text in enumerate(texts)
                 if isinstance(text, str) and text.strip()]
        encoded = tokenizer.encode_batch([text for _, text in valid])
        for (raw, _), encoding in zip(valid, encoded):
            if not encoding.ids:
                continue
            data = np.asarray(encoding.ids + [eos], dtype='<i8').tobytes()
            count += 1
            offset = 0
            while budgets and total + (len(data) - offset) // 8 >= budgets[0]:
                target = budgets.pop(0)
                take = (target - total) * 8
                hasher.update(data[offset:offset + take])
                offset += take
                total = target
                result[str(target)] = dict(sha256=hasher.hexdigest(), consumed_nonempty_stories=count,
                                           last_raw_index=raw, bytes=target * 8, encoding='little-endian int64 torch.long tensor hash')
                if not budgets:
                    return result
            hasher.update(data[offset:])
            total += (len(data) - offset) // 8
        cursor += len(texts)
    raise ValueError('SOURCE_EXHAUSTED')


def manifest_for(claim):
    path = ROOT / claim['path']
    if sha(path) != claim['sha256']:
        raise ValueError('ORIGINAL_EVIDENCE_CHANGED')
    document = json.loads(path.read_text())
    return document.get('manifest', document)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('IMMUTABLE_OUTPUT_EXISTS')
    ledger_path = REG / 'GLOBAL_DATASET_USAGE_LEDGER.json'
    ledger = json.loads(ledger_path.read_text())
    claims = [c for c in ledger['intervals'] if not c.get('revision')]
    oldproof = json.loads((REG / 'amendment-002/DATA_BOUNDARY_PROOF.json').read_text())
    for path, expected in zip((args.train, args.validation), oldproof['source_files']):
        if sha(path) != expected['sha256']:
            raise ValueError('FROZEN_SOURCE_CHANGED')
    tokenizer_path = ROOT / 'research/projects/cellular-computing/experiments/007-minicells-30m/artifacts/tokenizer.json'
    if sha(tokenizer_path) != OLD_TOKENIZER:
        raise ValueError('HISTORICAL_TOKENIZER_MISMATCH')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Reuse immutable 007 120M/1M proof rather than re-encode it.
    reproduced = {}
    for split, source in (('train', args.train), ('validation', args.validation)):
        budgets = []
        for c in claims:
            m = manifest_for(c)
            if c['role'] != split + '_stories_consumed' or m.get('tokenizer_sha256') != OLD_TOKENIZER:
                continue
            budget = m.get(split + '_stream_tokens')
            if budget and budget <= (15000000 if split == 'train' else 200000):
                budgets.append(budget)
        reproduced[split] = replay(source, tokenizer, budgets)
    rows = []
    for claim in claims:
        row = dict(original_claim=claim, original_revision='UNKNOWN_NOT_REWRITTEN')
        m = manifest_for(claim)
        split = claim['split']
        budget = m.get(split + '_stream_tokens')
        expected = m.get(split + '_token_sha256', m.get(split + '_raw_sha256'))
        found = reproduced.get(split, {}).get(str(budget))
        if found and m.get('tokenizer_sha256') == OLD_TOKENIZER and expected == found['sha256'] and claim['range'][1] == found['consumed_nonempty_stories']:
            row.update(classification='RECOVERED_MODEL_VISIBLE_STREAM_IDENTITY', evidence=found,
                       matching_snapshot=REVISION, coverage='WITHIN_007_REPRODUCED_CORPUS_TOKEN_PREFIX',
                       raw_text_identity='NOT_INFERRED_FROM_TOKEN_IDS', blocks_global_release=True)
        elif '007-minicells-30m' in claim['path']:
            row.update(classification='PREVIOUSLY_REPRODUCED_STREAM_REUSE',
                       proof_path='amendment-002/DATA_BOUNDARY_PROOF.json',
                       proof_sha256=sha(REG / 'amendment-002/DATA_BOUNDARY_PROOF.json'),
                       coverage='007_CORPUS_SUPERSET_NOT_OPTIMIZER_OBSERVED_STARTS', blocks_global_release=True)
        else:
            row.update(classification='UNKNOWN_REQUIRES_ARTIFACT_OR_SNAPSHOT_CHAIN',
                       coverage='CONSERVATIVE_INDEX_RANGE_NOT_IDENTITY_PROOF', blocks_global_release=True)
        rows.append(row)
    result = dict(schema='minicells.r3m.historical-coverage-closure.v1', status='PARTIAL_EVIDENCE_RECOVERED_NOT_RELEASED',
                  source_revision=REVISION, ledger_sha256=sha(ledger_path), auditor_sha256=sha(__file__),
                  historical_tokenizer_sha256=OLD_TOKENIZER, r3m_original_tokenizer_replaced=False,
                  source_files=oldproof['source_files'], reproduced_unique_streams=reproduced,
                  claims=rows, counts=dict(Counter(row['classification'] for row in rows)),
                  original_claims_rewritten=False, candidate_list_changed=False,
                  unknown_reservation_scopes=ledger['reservation_markers'],
                  window_gate='BLOCKED_NO_EQUIVALENCE_AMENDMENT_APPROVED',
                  optimizer_windows_recovered=False, model_quality_reads=0, gpu_training_started=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
