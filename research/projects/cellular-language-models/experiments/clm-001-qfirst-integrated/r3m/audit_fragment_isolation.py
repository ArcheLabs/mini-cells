"""Isolated content-only auditor. Frozen Main IDs; no model, quality or gate modification."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[5]
sys.path.insert(0, str(ROOT / 'shared/python'))
from minicells_data_registry import content_hashes  # noqa: E402
from minicells_data_registry.fragments import compile_scanner, matches, scan, write_patterns  # noqa: E402

REG = BASE / 'registration'
PROTOCOL = REG / 'content-isolation-001/FRAGMENT_AUDIT_PROTOCOL.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError('IMMUTABLE_AUDIT_OUTPUT_EXISTS')
    path.write_text(json.dumps(data, indent=2) + '\n')


def raw_rows(path, limit):
    cursor = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=['text']):
        for text in batch.column(0).to_pylist():
            if cursor >= limit:
                return
            if not isinstance(text, str):
                raise ValueError('NON_TEXT_SOURCE')
            yield cursor, text
            cursor += 1


class Words:
    def __init__(self):
        self.ids = {}

    def encode(self, text):
        words = re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())
        values = []
        for word in words:
            key = hashlib.sha256(word.encode()).digest()
            if key not in self.ids:
                self.ids[key] = (len(self.ids) + 1, word)
            elif self.ids[key][1] != word:
                raise ValueError('SHA256_WORD_COLLISION')
            values.append(self.ids[key][0])
        return values + [0] if values else []


class Stream:
    def __init__(self, path):
        self.path = path
        self.file = path.open('wb')
        self.offsets, self.raw_ids = [0], []

    def add(self, raw_id, values):
        if not values:
            return
        a = np.asarray(values, dtype='<u4')
        self.file.write(a.tobytes())
        self.raw_ids.append(raw_id)
        self.offsets.append(self.offsets[-1] + len(a))

    def close(self):
        self.file.close()
        np.save(str(self.path) + '.offsets.npy', np.asarray(self.offsets, dtype='<u8'))
        np.save(str(self.path) + '.raw-ids.npy', np.asarray(self.raw_ids, dtype='<u4'))
        return self.offsets[-1]


def make_patterns(view, texts, encode, lengths, destination):
    patterns, origins, lookup = [], [], {}
    for raw_id, text in texts:
        ids = np.asarray(encode(text), dtype='<u4')
        for n in lengths:
            for offset in range(len(ids) - n + 1):
                blob = ids[offset:offset + n].tobytes()
                key = (n, blob)
                if key not in lookup:
                    lookup[key] = len(patterns)
                    patterns.append(key)
                    origins.append([])
                origins[lookup[key]].append((raw_id, offset))
    path = destination / (view + '-patterns.bin')
    write_patterns(path, patterns)
    with (destination / (view + '-pattern-origins.jsonl')).open('w') as stream:
        for i, ((n, blob), origin) in enumerate(zip(patterns, origins)):
            stream.write(json.dumps(dict(pattern=i, length=n, sequence_sha256=hashlib.sha256(blob).hexdigest(), origins=origin)) + '\n')
    # Keep only content hashes and provenance, not a second full sequence copy in Python.
    descriptors = [(n, hashlib.sha256(blob).hexdigest()) for n, blob in patterns]
    return path, descriptors, origins


def old_stream_guard(path, budget, expected):
    values = np.memmap(path, dtype='<u4', mode='r')
    if len(values) < budget:
        raise ValueError('HISTORICAL_TOKEN_STREAM_TOO_SHORT')
    h = hashlib.sha256()
    for lo in range(0, budget, 1000000):
        h.update(np.asarray(values[lo:min(lo + 1000000, budget)], dtype='<u2').tobytes())
    if h.hexdigest() != expected:
        raise ValueError('ORIGINAL_007_MODEL_VISIBLE_STREAM_MISMATCH')
    return dict(tokens=budget, uint16_sha256=h.hexdigest(), status='EXACT_ORIGINAL_007_CORPUS_PREFIX')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.index.exists() or a.output.exists():
        raise FileExistsError('IMMUTABLE_AUDIT_DESTINATION_EXISTS')
    if sys.byteorder != 'little':
        raise ValueError('LITTLE_ENDIAN_REQUIRED')
    protocol = json.loads(PROTOCOL.read_text())
    oldproof = json.loads((REG / 'amendment-002/DATA_BOUNDARY_PROOF.json').read_text())
    for path, row in zip((a.train, a.validation), oldproof['source_files']):
        if sha(path) != row['sha256']:
            raise ValueError('SOURCE_FILE_IDENTITY_CHANGED')
    candidate_path = REG / 'amendment-003/MAIN_CANDIDATE_INDICES.json'
    candidates = json.loads(candidate_path.read_text())['records']
    selected = {r['raw_index']: r for r in candidates}
    texts = {}
    for raw, text in raw_rows(a.train, max(selected) + 1):
        if raw in selected:
            if list(content_hashes(text)) != selected[raw]['content_sha256']:
                raise ValueError('FROZEN_CANDIDATE_CONTENT_CHANGED')
            texts[raw] = text
    if len(texts) != 512:
        raise ValueError('CANDIDATE_COUNT_CHANGED')
    ordered = [(r['raw_index'], texts[r['raw_index']]) for r in candidates]
    new_path = REG / 'artifact-recovery-001/independent-review-002/tokenizer.json'
    old_path = ROOT / 'research/projects/cellular-computing/experiments/007-minicells-30m/artifacts/tokenizer.json'
    if sha(new_path) != '612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff' or sha(old_path) != 'becc6a25669e0f66424244f19fc846dfd8304bfd529c6187d772e8f961233be2':
        raise ValueError('ORIGINAL_TOKENIZER_CHANGED')
    new, old, words = Tokenizer.from_file(str(new_path)), Tokenizer.from_file(str(old_path)), Words()
    encode = dict(r3m4096_raw=lambda text: new.encode(text).ids + [new.token_to_id('<eos>')],
                  historical2048_strip=lambda text: old.encode(text.strip()).ids + [old.token_to_id('<eos>')] if text.strip() else [],
                  nfkc_words=words.encode)
    a.index.mkdir(parents=True)
    binary = compile_scanner(a.index / 'fragment-scan')
    pattern_data = {}
    for view in encode:
        lengths = protocol['word_fragment_lengths'] if view == 'nfkc_words' else protocol['token_fragment_lengths']
        pattern_data[view] = make_patterns(view, ordered, encode[view], lengths, a.index)
    whole_lookup = [{content_hashes(text)[m]: raw for raw, text in ordered} for m in range(3)]
    whole_matches = []
    streams = []
    oldmeta = json.loads((old_path.parent / 'corpus-manifest.json').read_text())
    for split, source, limit in (('train', a.train, 476901), ('validation', a.validation, 4396)):
        writers = {v: Stream(a.index / (v + '-' + split + '.u32')) for v in encode}
        batch = []
        def consume(rows, destinations):
            if not rows:
                return
            raw_ids, raw_texts = zip(*rows)
            encoded_new = new.encode_batch(list(raw_texts))
            stripped = [(i, t.strip()) for i, t in enumerate(raw_texts) if t.strip()]
            encoded_old = old.encode_batch([t for _, t in stripped])
            old_by_row = {i: e.ids + [old.token_to_id('<eos>')] for (i, _), e in zip(stripped, encoded_old) if e.ids}
            for i, (raw, text) in enumerate(rows):
                for method, h in enumerate(content_hashes(text)):
                    if h in whole_lookup[method]:
                        whole_matches.append(dict(candidate=whole_lookup[method][h], split=split, source_raw_id=raw, method=method))
                destinations['r3m4096_raw'].add(raw, encoded_new[i].ids + [new.token_to_id('<eos>')])
                destinations['historical2048_strip'].add(raw, old_by_row.get(i, []))
                destinations['nfkc_words'].add(raw, words.encode(text))
        for raw, text in raw_rows(source, limit):
            batch.append((raw, text))
            if len(batch) == 1024:
                consume(batch, writers)
                batch = []
            if (raw + 1) % 50000 == 0:
                print(json.dumps(dict(stage='CONTENT_INDEX_ONLY', split=split, records=raw + 1)), flush=True)
        consume(batch, writers)
        for view, writer in writers.items():
            count = writer.close()
            row = dict(view=view, split=split, path=str(writer.path), tokens=count, raw_range=[0, limit], budget=count,
                       budget_meaning='CONSERVATIVE_SOURCE_SUPERSET_NOT_OBSERVED_SAMPLING')
            if view == 'historical2048_strip':
                budget = oldmeta[split + '_stream_tokens']
                row['model_visible_guard'] = old_stream_guard(writer.path, budget, oldmeta[split + '_raw_sha256'])
                row['budget'] = budget
                row['budget_meaning'] = 'EXACT_007_ELIGIBLE_CORPUS_NOT_ACTUAL_OPTIMIZER_WINDOW_USAGE'
            streams.append(row)
        del writers
    all_summaries, examples, affected = [], [], set()
    for row in streams:
        view, split, source = row['view'], row['split'], Path(row['path'])
        patterns, descriptions, origins = pattern_data[view]
        result_path = a.index / (view + '-' + split + '-matches.tsv')
        print(json.dumps(dict(stage='EXACT_FRAGMENT_SCAN', view=view, split=split, tokens=row['tokens'])), flush=True)
        scan(binary, patterns, source, result_path, row['budget'])
        offsets = np.load(str(source) + '.offsets.npy', mmap_mode='r')
        raw_ids = np.load(str(source) + '.raw-ids.npy', mmap_mode='r')
        summaries = {}
        for hit in matches(result_path):
            n, pattern_id = hit['length'], hit['pattern']
            s = summaries.setdefault(n, dict(view=view, split=split, length=n, matching_patterns=0,
                              occurrences=0, budget_occurrences=0, affected_candidates=set(), retained_examples=0))
            candidate_ids = {r for r, _ in origins[pattern_id]}
            affected.update(candidate_ids)
            s['affected_candidates'].update(candidate_ids)
            s['matching_patterns'] += 1
            s['occurrences'] += hit['occurrences']
            s['budget_occurrences'] += hit['budget_occurrences']
            if s['retained_examples'] < 128:
                positions = []
                for pos in hit['positions']:
                    first = int(np.searchsorted(offsets, pos, side='right') - 1)
                    last = int(np.searchsorted(offsets, pos + n - 1, side='right') - 1)
                    positions.append(dict(stream_start=pos, first_raw_id=int(raw_ids[first]), last_raw_id=int(raw_ids[last]),
                                          first_story_offset=pos - int(offsets[first]), crosses_story_boundary=first != last,
                                          within_budget=pos + n <= row['budget']))
                examples.append(dict(view=view, split=split, length=n, pattern_id=pattern_id,
                         sequence_sha256=descriptions[pattern_id][1], candidate_origins=origins[pattern_id],
                         occurrences=hit['occurrences'], budget_occurrences=hit['budget_occurrences'], source_examples=positions))
                s['retained_examples'] += 1
        lengths = protocol['word_fragment_lengths'] if view == 'nfkc_words' else protocol['token_fragment_lengths']
        for n in lengths:
            s = summaries.get(n, dict(view=view, split=split, length=n, matching_patterns=0, occurrences=0,
                                      budget_occurrences=0, affected_candidates=set(), retained_examples=0))
            s['affected_candidates'] = sorted(s['affected_candidates'])
            all_summaries.append(s)
    index_files = {f.name: dict(bytes=f.stat().st_size, sha256=sha(f)) for f in sorted(a.index.iterdir()) if f.is_file()}
    result = dict(schema='minicells.r3m.fragment-isolation-audit.v1', status='PARTIAL_SOURCE_AUDIT_NOT_GLOBAL_ISOLATION_PROOF',
        auditor_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL), protocol_frozen_commit='d490d1b4',
        bindings_sha256=sha(REG / 'content-isolation-001/IMPLEMENTATION_BINDINGS.json'),
        candidate_manifest_sha256=sha(candidate_path), candidate_count=512, candidate_selection_changed=False,
        original_tokenizers=dict(r3m4096=sha(new_path), historical2048=sha(old_path)),
        runtime=dict(python=sys.version, numpy=np.__version__, unicode=unicodedata.unidata_version),
        source_files=oldproof['source_files'], streams=streams, whole_story_matches=whole_matches,
        fragment_summaries=all_summaries, affected_candidate_ids_any_diagnostic=sorted(affected), examples=examples,
        exact_verification='ELEMENTWISE_AFTER_UINT64_FILTER_NO_HASH_ONLY_HITS',
        evidence_limits=['Only the declared conservative prefix union is scanned, not an unbounded reservation or unspecified external sources.',
                         'First three source positions per matching pattern retained; all occurrences counted.',
                         'Fixed diagnostic lengths are not approved scientific contamination/release thresholds.',
                         'No claim that a matched corpus window was actually consumed by an optimizer.'],
        index=dict(path=str(a.index), files=index_files, durability='LOCAL_ONLY_NOT_HF_PUBLISHED'),
        sampling_replay_repeated=False, model_quality_reads=0, gpu_work=0,
        global_content_isolation='BLOCKED', scientific_registration='BLOCKED', proof_b_approved=False)
    a.output.mkdir(parents=True)
    write(a.output / 'FRAGMENT_ISOLATION_PROOF.json', result)
    print(json.dumps(dict(status=result['status'], whole_story_matches=len(whole_matches),
                         affected_candidates=len(affected), global_content_isolation='BLOCKED')), flush=True)


if __name__ == '__main__':
    main()
