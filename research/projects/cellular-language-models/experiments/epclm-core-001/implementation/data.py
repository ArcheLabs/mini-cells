"""Document-aware development data; no inherited confirmation set is opened."""
import hashlib
import json
import os
import unicodedata
from pathlib import Path
import native_clm_runtime as native
from native_clm_candidates import VERIFIED_DATASET_REVISION
from native_clm_runtime import iter_tinystories, train_tokenizer, load_tokenizer, sha256_file


def configure_dataset(spec):
    if spec != {'id': native.DATASET_ID, 'revision': VERIFIED_DATASET_REVISION}:
        raise ValueError('Core-001 requires the verified Native TinyStories dataset revision')
    # Native entrypoints apply this correction too. Its shared loader and
    # tokenizer both read the runtime module global, not an imported copy.
    native.DATASET_REVISION = spec['revision']
    return spec['revision']

def prepare(root,stories=2048,dataset=None):
    revision=configure_dataset(dataset or {'id': native.DATASET_ID, 'revision': VERIFIED_DATASET_REVISION})
    token=os.environ.get('HF_TOKEN') or None
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('dataset_revision')!=revision or manifest.get('requested_stories')!=stories:
            raise ValueError('cached dataset revision/story budget differs from protocol')
        for name,sha in manifest['files'].items():
            if sha256_file(root/name)!=sha: raise ValueError('asset hash mismatch: '+name)
        return manifest
    tokenizer_path=root/'tokenizer.json'
    if not tokenizer_path.exists(): train_tokenizer(tokenizer_path,token=token)
    tokenizer=load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size()!=2048: raise ValueError('expected existing 2048 BPE vocabulary')
    groups={'train':[],'development':[]}; seen=set()
    # Reuse the fixed TRAIN revision only; existing validation/confirmation are untouched.
    for text in iter_tinystories('train',token=token,max_stories=stories):
        normalized=' '.join(unicodedata.normalize('NFKC',text).split())
        identity=hashlib.sha256(normalized.encode()).hexdigest()
        if identity in seen: continue
        seen.add(identity)
        split='development' if int(identity[:8],16)%10==0 else 'train'
        ids=[tokenizer.token_to_id('<bos>')]+tokenizer.encode(normalized).ids+[tokenizer.token_to_id('<eos>')]
        groups[split].append({'id':identity,'tokens':ids})
    if not all(groups.values()): raise ValueError('empty document split')
    for split,rows in groups.items(): (root/(split+'.json')).write_text(json.dumps(rows))
    manifest={'dataset_id':native.DATASET_ID,'dataset_revision':revision,'requested_stories':stories,'split':'core001-content-sha256-mod10-development',
              'confirmation':'SEALED_NOT_CREATED','tokenizer_overlap':'historical tokenizer training may overlap development; development only',
              'documents':{k:len(v) for k,v in groups.items()},
              'files':{name:sha256_file(root/name) for name in ('tokenizer.json','train.json','development.json')}}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
