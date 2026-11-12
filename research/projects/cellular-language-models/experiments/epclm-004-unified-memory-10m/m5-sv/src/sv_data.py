"""Versioned evaluation metadata around unchanged M1 token episodes."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'm5/src')]
from bmojo import BMojo
from data import MQARConfig, episode, digest, dataset_manifest, FACT, QUERY, ANSWER
from tasks import variant


def make_episode(condition, split, index, seed, reference=None):
    queries = condition.get('queries', 1)
    config = MQARConfig(bindings=condition['bindings'], length=condition['length'] + 4*(queries-1),
                        queries=queries, interference=condition.get('interference', 1.), seed=seed)
    row = variant(config, split, index, reference, condition.get('similarity', 'random'), condition.get('repeat', False))
    first = row['query_positions'][0] - 2
    history = row['input_ids'][:first]
    records = [{'record_id': row['id'] + ':record:' + str(i), 'key_id': 8+k,
                'value_id': 264+v, 'source_position': 3+3*i}
               for i, (k,v) in enumerate(row['bindings'])]
    by_key = {r['key_id']: r for r in records}
    questions = []
    for i, p in enumerate(row['query_positions']):
        key = row['input_ids'][p-1]; record = by_key[key]
        questions.append({'query_id': row['id'] + ':query:' + str(i), 'key_id': key,
            'required_record_ids': [record['record_id']], 'expected_value_id': row['labels'][p],
            'query_position': len(history)+2, 'input_ids': [QUERY,key,ANSWER]})
    return {'id': row['id'], 'split': split, 'condition': condition['name'], 'history': history,
            'records': records, 'queries': questions, 'bindings': row['bindings'],
            'config': asdict(config), 'original_input_sha256': digest(row['input_ids'])}


def training_rows(seed, budget, length=512):
    full, remainder = divmod(budget, length)
    if remainder and remainder < 28: raise ValueError('budget remainder cannot fit one unchanged M1 episode')
    rows = []
    for i in range(full + bool(remainder)):
        n = length if i < full else remainder
        rows.append(episode(MQARConfig(length=n, seed=seed), 'train', i))
    assert sum(len(r['input_ids']) for r in rows) == budget
    return rows


def validation_rows(seed, count=16):
    return [episode(MQARConfig(length=512, seed=seed), 'validation', i) for i in range(count)]


def bank(path, conditions, split, seed, count, reference=None):
    """Freeze streamed episodes before any evaluation; never silently replace a bank."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    request = {'conditions':conditions, 'split':split, 'data_seed':seed, 'episodes_per_condition':count,
               'reference_sha256':hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest() if reference is not None else None}
    manifest_path = path.with_suffix('.manifest.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['request'] != request or sha_file(path) != manifest['sha256']: raise ValueError('frozen bank differs')
        return manifest
    pending = path.with_suffix('.pending'); h = hashlib.sha256(); total = 0
    with pending.open('wb') as stream:
        for c in conditions:
            for i in range(count):
                row = make_episode(c, split, i, seed, reference)
                payload = (json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode()
                stream.write(payload); h.update(payload); total += 1
    pending.replace(path)
    manifest = {'format':'minicells.m5-sv.bank.v1', 'request':request, 'sha256':h.hexdigest(),
                'episodes':total, 'test_use':'GENERATED_AND_HASHED_NOT_EVALUATED' if split=='test' else 'DEVELOPMENT'}
    from e0 import write
    write(manifest_path, manifest); return manifest


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_bank(path):
    with Path(path).open() as stream:
        for line in stream: yield json.loads(line)


def audit_splits(groups):
    """Pair partition is structural; content hashes also catch repeated episodes."""
    bindings = {}; contents = {}
    for split, rows in groups.items():
        pairs=set(); tokens=set(); part=('train','validation','test').index(split)
        for row in rows:
            assert all((k+v)%3==part for k,v in row['bindings'])
            pairs.update(tuple(p) for p in row['bindings'])
            tokens.add(digest(row.get('history',row.get('input_ids'))))
        bindings[split]=pairs; contents[split]=tokens
    for a in groups:
        for b in groups:
            if a<b and (bindings[a]&bindings[b] or contents[a]&contents[b]): raise ValueError('split overlap')
    return {'status':'PASS', 'binding_partition':'(key_index + value_index) mod 3 = split index',
            'token_generalization':'shared token vocabulary; unseen pairs, not unseen symbols'}
