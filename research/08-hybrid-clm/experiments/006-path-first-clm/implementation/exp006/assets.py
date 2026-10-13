"""Independent 006 assets. Confirmation materialization requires Gate 4 authorization."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .config import authorize,digest
from .reuse import canonical_sha256,write_json
from .data import encode_stories,assert_disjoint,select_windows

def prepare(c,destination,source_parquet,source_sha256):
    # Only the explicitly supplied pinned training source is opened. No validation split.
    if not source_sha256 or digest(source_parquet)!=source_sha256:raise ValueError('pinned source Parquet hash mismatch')
    from datasets import load_dataset
    from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
    data=c['data'];high=data['development_range'][1];rows=load_dataset('parquet',data_files=str(source_parquet),split='train',streaming=True)
    train=[];dev=[]
    for i,row in enumerate(rows):
        if i>=high:break
        if data['train_range'][0]<=i<data['train_range'][1]:train.append(row['text'])
        if data['development_range'][0]<=i<high:dev.append(row['text'])
    if len(train)!=data['train_range'][1]-data['train_range'][0] or len(dev)!=high-data['development_range'][0]:raise ValueError('insufficient train/development stories')
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'));tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tokenizer.decoder=decoders.ByteLevel()
    tokenizer.train_from_iterator(train[:data['tokenizer_stories']],trainers.BpeTrainer(vocab_size=c['model']['vocab_size'],special_tokens=['<unk>','<eos>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    destination=Path(destination)
    if destination.exists():raise FileExistsError('preserve existing asset bundle')
    destination.mkdir(parents=True);tokenizer.save(str(destination/'tokenizer.json'));metadata={}
    for role,texts in [('train',train),('development',dev)]:
        tokens,starts,groups,hashes=encode_stories(texts,tokenizer,c['training']['sequence_length']);np.save(destination/(role+'.npy'),tokens)
        metadata[role]={'starts':starts,'groups':groups,'story_hashes':hashes,'stories':len(texts),'tokens':len(tokens)}
    assert_disjoint(metadata['train']['story_hashes'],metadata['development']['story_hashes'])
    write_json(destination/'windows.json',metadata)
    names=['train.npy','development.npy','tokenizer.json','windows.json'];manifest={'experiment_id':'006-path-first-clm','source_parquet_sha256':source_sha256,
        'data_definition':{k:v for k,v in data.items() if k not in ('manifest_sha256','source_sha256','confirmation_manifest_sha256')},'sequence_length':c['training']['sequence_length'],'vocab_size':tokenizer.get_vocab_size(),'files':{n:digest(destination/n) for n in names},
        'confirmation_materialized':False,'tokenizer_training':'train only','duplicate_checks':'within split and cross train/development; exact story content'}
    write_json(destination/'manifest.json',manifest);return manifest

def load_assets(folder,c,p,seed,role,*,include_evaluation=None):
    authorize(c,p,seed,role) # Must precede any manifest / token reads.
    folder=Path(folder);manifest_path=folder/'manifest.json'
    expected=c['data']['confirmation_manifest_sha256'] if role=='confirmation' else c['data']['manifest_sha256']
    if digest(manifest_path)!=expected:raise ValueError('asset manifest hash mismatch')
    m=json.loads(manifest_path.read_text())
    if m['experiment_id']!='006-path-first-clm' or m['source_parquet_sha256']!=c['data']['source_sha256']:raise ValueError('wrong source data identity')
    if m['sequence_length']!=c['training']['sequence_length'] or m['vocab_size']!=c['model']['vocab_size']:raise ValueError('asset shape mismatch')
    if m.get('data_definition')!={k:v for k,v in c['data'].items() if k not in ('manifest_sha256','source_sha256','confirmation_manifest_sha256')}:raise ValueError('asset story ranges differ from registered contract')
    if role=='confirmation' and not m.get('confirmation_materialized'):raise RuntimeError('separate sealed confirmation bundle is not available')
    if include_evaluation is None:include_evaluation=role=='development'
    needed=['train.npy','tokenizer.json','windows.json']
    if include_evaluation:needed.append(role+'.npy')
    for name in needed:
        if name not in m['files'] or digest(folder/name)!=m['files'][name]:raise ValueError('corrupt asset '+name)
    windows=json.loads((folder/'windows.json').read_text())
    result={'train':np.load(folder/'train.npy',mmap_mode='r'),'train_starts':windows['train']['starts'],
            'manifest_sha256':digest(manifest_path),'evaluation_loaded':include_evaluation}
    if include_evaluation:
        assert_disjoint(windows['train']['story_hashes'],windows[role]['story_hashes'])
        if role=='confirmation':assert_disjoint(windows['development']['story_hashes'],windows[role]['story_hashes'])
        starts,groups=select_windows(windows[role]['starts'],windows[role]['groups'],c['evaluation']['windows'],c['evaluation']['window_seed'])
        result.update({role:np.load(folder/(role+'.npy'),mmap_mode='r'),'eval_starts':starts,'eval_groups':groups})
    return result



def prepare_confirmation(c,p,seed,destination,development_bundle,source_parquet):
    """Gate 4 only: reuse frozen training/tokenizer, materialize a separate holdout.

    Preparation approval and training approval are distinct: after preparation,
    freeze the resulting manifest in a new approved contract before any training.
    """
    authorize(c,p,seed,'confirmation',require_confirmation_manifest=False)
    data=c['data']; development_bundle=Path(development_bundle)
    if digest(development_bundle/'manifest.json')!=data['manifest_sha256']:
        raise ValueError('development bundle identity mismatch')
    original=json.loads((development_bundle/'manifest.json').read_text())
    for name,h in original['files'].items():
        if digest(development_bundle/name)!=h:raise ValueError('corrupt frozen development bundle')
    if digest(source_parquet)!=data['source_sha256']:raise ValueError('confirmation source identity mismatch')
    from datasets import load_dataset
    from tokenizers import Tokenizer
    import shutil
    lo,hi=data['confirmation_range'];texts=[]
    rows=load_dataset('parquet',data_files=str(source_parquet),split='train',streaming=True)
    for i,row in enumerate(rows):
        if i>=hi:break
        if i>=lo:texts.append(row['text'])
    if len(texts)!=hi-lo:raise ValueError('insufficient independent confirmation stories')
    tokenizer=Tokenizer.from_file(str(development_bundle/'tokenizer.json'))
    tokens,starts,groups,hashes=encode_stories(texts,tokenizer,c['training']['sequence_length'])
    windows=json.loads((development_bundle/'windows.json').read_text())
    assert_disjoint(windows['train']['story_hashes'],hashes)
    assert_disjoint(windows['development']['story_hashes'],hashes)
    destination=Path(destination)
    if destination.exists():raise FileExistsError('preserve existing confirmation bundle')
    destination.mkdir(parents=True)
    for name in ('train.npy','tokenizer.json'):shutil.copy2(development_bundle/name,destination/name)
    np.save(destination/'confirmation.npy',tokens)
    windows['confirmation']={'starts':starts,'groups':groups,'story_hashes':hashes,'stories':len(texts),'tokens':len(tokens)}
    write_json(destination/'windows.json',windows)
    manifest={**original,'files':{n:digest(destination/n) for n in ('train.npy','confirmation.npy','tokenizer.json','windows.json')},
              'confirmation_materialized':True,'development_manifest_sha256':data['manifest_sha256'],
              'duplicate_checks':'within each split and all three cross-split pairs'}
    write_json(destination/'manifest.json',manifest)
    return manifest
