"""Pinned tokenizer and real complete documents; never infer boundaries from random spans."""
import json
from pathlib import Path
from dependencies import shared,BASE


def prepare(cache,config):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((shared.INCREMENTAL.parent/'retrain_config.json').read_text())
    tokenizer_path=shared.ROOT/frozen['tokenizer_path']
    if shared.digest(tokenizer_path)!=frozen['tokenizer_sha256']:raise ValueError('BLOCKED_DATA_INTEGRITY: frozen tokenizer mismatch')
    tokenizer=shared.native.load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size()!=config['model']['vocab']:raise ValueError('BLOCKED_DATA_INTEGRITY: vocabulary mismatch')
    shared.native.DATASET_REVISION=frozen['dataset_revision']
    manifest_path=cache/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['dataset_revision']!=frozen['dataset_revision'] or manifest['tokenizer_sha256']!=frozen['tokenizer_sha256'] or manifest['max_document_targets']!=config['groups']['max_document_targets']:raise ValueError('BLOCKED_DATA_INTEGRITY: incompatible cache')
        docs={split:json.loads((cache/(split+'.json')).read_text()) for split in ('train','validation')}
        if any(shared.digest(cache/name)!=sha for name,sha in manifest['files'].items()):raise ValueError('BLOCKED_DATA_INTEGRITY: document cache mismatch')
        return docs,manifest
    docs={};index=[]
    for split in ('train','validation'):
        values=[];targets=0;skipped=0;offset=0
        for source_index,text in enumerate(shared.native.iter_tinystories(split)):
            ids=[2]+tokenizer.encode(text).ids+[3]
            if len(ids)-1>config['groups']['max_document_targets']:skipped+=1;continue
            doc_id=f'{split}:{source_index}'
            values.append({'doc_id':doc_id,'ids':ids});count=len(ids)-1
            index.append({'doc_id':doc_id,'split':split,'token_start':offset,'token_end':offset+len(ids),'token_count':len(ids),'prediction_targets':count,'document_sha256':shared.sha(ids),'eos_policy':'BOS input, EOS scored; no cross-document target','chunks':[{'input_start':s,'input_end':min(s+config['model']['chunk'],count),'loss_mask':'all valid targets','context_prefix':'document start; persistent numerical KV/State'} for s in range(0,count,config['model']['chunk'])]})
            offset+=len(ids);targets+=count
            if (split=='train' and targets>=1100000) or (split=='validation' and len(values)==config['evaluation']['documents']):break
        if not values:raise ValueError('BLOCKED_DATA_INTEGRITY: no complete documents')
        docs[split]=values;shared.write(cache/(split+'.json'),values)
    train_hashes={shared.sha(d['ids']) for d in docs['train']}
    if any(shared.sha(d['ids']) in train_hashes for d in docs['validation']):raise ValueError('BLOCKED_DATA_INTEGRITY: duplicated development document across splits')
    (cache/'document_index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in index))
    manifest={'dataset':shared.native.DATASET_ID,'dataset_revision':frozen['dataset_revision'],'tokenizer_sha256':shared.digest(tokenizer_path),'tokenizer_path':frozen['tokenizer_path'],'max_document_targets':config['groups']['max_document_targets'],'selection':'first complete nonempty documents <= max targets; filtered-length development subset; not historical A2 stream','holdout':'NOT_ACCESSED','files':{name:shared.digest(cache/name) for name in ('train.json','validation.json','document_index.jsonl')}}
    shared.write(manifest_path,manifest);return docs,manifest


def groups(documents,size):return [documents[i:i+size] for i in range(0,len(documents),size)]
def count(group):return sum(len(d['ids'])-1 for d in group)
