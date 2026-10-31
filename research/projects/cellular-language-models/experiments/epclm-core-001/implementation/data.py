"""Document-aware development data; no inherited confirmation set is opened."""
import hashlib
import json
import unicodedata
from pathlib import Path
from native_clm_runtime import iter_tinystories, train_tokenizer, load_tokenizer, DATASET_REVISION, sha256_file

def prepare(root,stories=2048):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for name,sha in manifest['files'].items():
            if sha256_file(root/name)!=sha: raise ValueError('asset hash mismatch: '+name)
        return manifest
    tokenizer_path=root/'tokenizer.json'
    if not tokenizer_path.exists(): train_tokenizer(tokenizer_path)
    tokenizer=load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size()!=2048: raise ValueError('expected existing 2048 BPE vocabulary')
    groups={'train':[],'development':[]}; seen=set()
    # Reuse the fixed TRAIN revision only; existing validation/confirmation are untouched.
    for text in iter_tinystories('train',max_stories=stories):
        normalized=' '.join(unicodedata.normalize('NFKC',text).split())
        identity=hashlib.sha256(normalized.encode()).hexdigest()
        if identity in seen: continue
        seen.add(identity)
        split='development' if int(identity[:8],16)%10==0 else 'train'
        ids=[tokenizer.token_to_id('<bos>')]+tokenizer.encode(normalized).ids+[tokenizer.token_to_id('<eos>')]
        groups[split].append({'id':identity,'tokens':ids})
    if not all(groups.values()): raise ValueError('empty document split')
    for split,rows in groups.items(): (root/(split+'.json')).write_text(json.dumps(rows))
    manifest={'dataset_revision':DATASET_REVISION,'split':'core001-content-sha256-mod10-development',
              'confirmation':'SEALED_NOT_CREATED','tokenizer_overlap':'historical tokenizer training may overlap development; development only',
              'documents':{k:len(v) for k,v in groups.items()},
              'files':{name:sha256_file(root/name) for name in ('tokenizer.json','train.json','development.json')}}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
