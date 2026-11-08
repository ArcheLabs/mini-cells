"""M1 data contracts: explicit supervision, reserved symbols and disjoint bindings."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

WIKITEXT = {'path': 'Salesforce/wikitext', 'name': 'wikitext-103-raw-v1',
            'revision': 'b08601e04326c79dfdd32d625aee71d232d685c3'}
SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>', '<fact>', '<query>', '<answer>', '<hole>']
SYMBOLS = [f'<key_{i}>' for i in range(256)] + [f'<value_{i}>' for i in range(256)] + [f'<noise_{i}>' for i in range(16)]
TOKENS = SPECIAL + SYMBOLS
PAD, UNK, BOS, EOS, FACT, QUERY, ANSWER, HOLE = range(8)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class MQARConfig:
    bindings: int = 8
    length: int = 128
    queries: int = 1
    interference: float = 1.0
    seed: int = 41001
    regime: str = 'unseen_pairs'

    def __post_init__(self):
        if not 1 <= self.bindings <= 64 or not 1 <= self.queries <= self.bindings:
            raise ValueError('bindings must be 1..64 and queries 1..bindings')
        if self.length < 1 + 3*self.bindings + 4*self.queries - 1 or not 0 <= self.interference <= 1:
            raise ValueError('insufficient length or invalid interference ratio')
        if self.regime not in ('unseen_pairs', 'unseen_symbols'):
            raise ValueError('unknown generalization regime')


def episode(config, split, index):
    if split not in ('train', 'validation', 'test') or index < 0:
        raise ValueError('invalid split/index')
    part = ('train', 'validation', 'test').index(split)
    seed = int(digest([asdict(config), split, index])[:16], 16)
    rng = np.random.default_rng(seed)
    pool = np.arange(256) if config.regime == 'unseen_pairs' else [np.arange(128), np.arange(128,192), np.arange(192,256)][part]
    keys = rng.choice(pool, config.bindings, replace=False)
    values = []
    for k in keys:
        choices = [v for v in pool if v not in values and (config.regime == 'unseen_symbols' or (int(k)+int(v)) % 3 == part)]
        if not choices:
            raise ValueError('binding support exhausted')
        values.append(int(rng.choice(choices)))
    pairs = list(zip(keys.tolist(), values))
    ids = [BOS]
    for k,v in pairs:
        ids.extend([FACT, 8+k, 264+v])
    room = config.length-len(ids)-(4*config.queries-1)
    noise = int(round(room*config.interference))
    gap = rng.integers(520,536,size=noise).tolist()+[HOLE]*(room-noise)
    rng.shuffle(gap); ids.extend(gap)
    labels = [-100]*config.length; query_mask = [False]*config.length
    selected = rng.choice(config.bindings, config.queries, replace=False)
    positions = []
    for j,n in enumerate(selected):
        k,v = pairs[int(n)]; ids.extend([QUERY,8+k,ANSWER]); pos = len(ids)-1
        labels[pos] = 264+v; query_mask[pos] = True; positions.append(pos)
        if j+1 < config.queries: ids.append(HOLE)  # Never feed a gold/model answer.
    assert len(ids) == config.length and positions[-1] == config.length-1
    return {'id': digest([asdict(config),split,index]), 'split': split, 'input_ids': ids,
            'labels': labels, 'query_mask': query_mask, 'bindings': pairs,
            'query_positions': positions, 'regime': config.regime}


@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_mask: torch.Tensor
    episode_ids: list
    input_tokens: int


def batch(rows, device='cpu'):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('empty batch or duplicate episode IDs')
    width = max(len(r['input_ids']) for r in rows)
    ids = [r['input_ids']+[PAD]*(width-len(r['input_ids'])) for r in rows]
    labels = [r['labels']+[-100]*(width-len(r['labels'])) for r in rows]
    queries = [r['query_mask']+[False]*(width-len(r['query_mask'])) for r in rows]
    return Batch(torch.tensor(ids,device=device), torch.tensor(labels,device=device),
                 torch.tensor(queries,device=device), [r['id'] for r in rows], sum(len(r['input_ids']) for r in rows))


def dataset_manifest(config, splits, tokenizer_sha256):
    identities = {k: [r['id'] for r in rows] for k,rows in splits.items()}
    for a in splits:
        for b in splits:
            if a < b:
                if set(identities[a]) & set(identities[b]): raise ValueError('episode overlap')
                pa = {tuple(p) for r in splits[a] for p in r.get('bindings', [])}
                pb = {tuple(p) for r in splits[b] for p in r.get('bindings', [])}
                if pa & pb: raise ValueError('binding overlap')
    return {'format':'minicells.epclm004.dataset.v1','config':config,
            'tokenizer_sha256':tokenizer_sha256,'special_ids':dict(zip(SPECIAL,range(8))),
            'splits':{k:{'episodes':len(v),'content_sha256':digest(v)} for k,v in splits.items()},
            'test_status':'SEALED_NOT_ACCESSED' if 'test' not in splits else 'EXPLICIT_TEST_ACCESS'}


def train_bpe(texts, destination, vocab_size=8192):
    """Reuse common/train.py's ByteLevel BPE recipe; only a train-text iterator enters."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size,
        special_tokens=TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError('training text insufficient for exact requested vocabulary')
    if [tokenizer.token_to_id(t) for t in TOKENS] != list(range(len(TOKENS))):
        raise ValueError('special/synthetic IDs differ')
    tokenizer.save(str(destination)); return tokenizer


def wikitext_prefix(split, nonempty_lines):
    if split not in ('train','validation') or nonempty_lines < 1:
        raise ValueError('M1 adapter permits train/validation only; test sealed')
    from datasets import load_dataset
    stream = load_dataset(**WIKITEXT, split=split, streaming=True)
    rows = []
    for row in stream:
        # No normalization except newline canonicalization and omission of blank lines.
        text = row['text'].replace('\r\n','\n')
        if text.strip(): rows.append(text)
        if len(rows) == nonempty_lines: return rows
    raise ValueError('WikiText prefix incomplete; no corpus substitution allowed')


def prepare_wikitext(folder, train_lines=2048, validation_lines=32):
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    manifest_path = folder/'dataset.json'
    request = {'dataset':WIKITEXT,'train_lines':train_lines,'validation_lines':validation_lines,
               'preprocessing':'nonempty-lines-crlf-to-lf-v1','tokenizer':'ByteLevel-BPE-8192-reserved536-v1'}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['request'] != request: raise ValueError('cached data request differs')
        for name,sha in manifest['files'].items():
            if file_hash(folder/name) != sha: raise ValueError('asset hash differs: '+name)
        return manifest
    texts = {s:wikitext_prefix(s,n) for s,n in [('train',train_lines),('validation',validation_lines)]}
    if set(texts['train']) & set(texts['validation']):
        raise ValueError('exact selected line overlap; choose explicit larger/sanitized policy')
    tokenizer = train_bpe(texts['train'],folder/'tokenizer.json')
    for split,lines in texts.items():
        (folder/(split+'.json')).write_text(json.dumps(lines,ensure_ascii=False)+'\n')
    manifest = {'format':'minicells.epclm004.wikitext.v1','request':request,
                'text_sha256':{s:digest(t) for s,t in texts.items()},
                'files':{n:file_hash(folder/n) for n in ('tokenizer.json','train.json','validation.json')},
                'vocab_size':tokenizer.get_vocab_size(),'test_status':'SEALED_NOT_ACCESSED',
                'special_ids':dict(zip(SPECIAL,range(8))),'tokenizer_train_split':'train only'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n'); return manifest


def lm_rows(folder, split, length=128, limit=64):
    if split not in ('train','validation'): raise ValueError('test sealed')
    from tokenizers import Tokenizer
    folder = Path(folder); tokenizer = Tokenizer.from_file(str(folder/'tokenizer.json'))
    rows = []
    for index,text in enumerate(json.loads((folder/(split+'.json')).read_text())):
        ids = [BOS]+tokenizer.encode(text).ids+[EOS]
        for start in range(0,len(ids)-1,length):
            x=ids[start:start+length]; y=ids[start+1:start+length+1]
            if len(x)>len(y):x=x[:len(y)]
            rows.append({'id':digest([WIKITEXT,split,index,start]),'split':split,
                         'input_ids':x,'labels':y,'query_mask':[False]*len(y)})
            if len(rows)==limit:return rows
    if not rows:raise ValueError('no LM targets')
    return rows
