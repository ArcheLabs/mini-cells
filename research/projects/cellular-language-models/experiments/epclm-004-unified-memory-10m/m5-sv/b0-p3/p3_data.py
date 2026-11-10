"""Episode-local random mappings with strict teacher-forcing alignment."""
import json
import random
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'b0/src'))
from b0_models import construct
from data import digest,file_hash,BOS,EOS,PAD,FACT,QUERY,ANSWER,TOKENS
from e0 import write,ROOT,Sampler
SEP=7  #Existing <hole> token; no Tokenizer or embedding change.


def seed_for(master,namespace,level,epoch,index):
    if namespace=='test':raise PermissionError('Test generation and access are forbidden')
    return digest([master,namespace,level,epoch,index])


def render(pairs,query_keys,length,identity):
    mapping=dict(pairs)
    if len(mapping)!=len(pairs):raise ValueError('Keys must be unique')
    ids=[BOS]
    for k,v in pairs:ids.extend([FACT,k,v])
    ids.append(SEP);positions=[];targets=[]
    for k in query_keys:
        ids.extend([QUERY,k,ANSWER]);positions.append(len(ids)-1);targets.append(mapping[k]);ids.append(mapping[k])
    ids.append(EOS);valid=len(ids)
    if valid>length:raise ValueError('Episode exceeds logical sequence length')
    ids.extend([PAD]*(length-valid));labels=[-100]*length
    for pos,target in zip(positions,targets):labels[pos]=target
    return {'id':digest([identity,ids]),'input_ids':ids,'labels':labels,'query_positions':positions,
        'valid_length':valid,'storage_length':length,'bindings':pairs,'query_keys':query_keys,
        'mapping_hash':digest(sorted(pairs)),'episode_hash':digest(ids),'candidates':sorted(set(mapping.values())),**identity}


def episode(master,namespace,level,epoch,index):
    if namespace not in ('train','validation'):raise PermissionError('Only Train/Validation generation is authorized')
    if level not in (8,16):raise ValueError('Only8/16-pair levels are authorized')
    seed=seed_for(master,namespace,level,epoch,index);rng=np.random.default_rng(int(seed,16))
    keys=(rng.choice(256,level,replace=False)+8).tolist();values=(rng.integers(256,size=level)+264).tolist()
    pairs=list(zip(keys,values));facts=[pairs[int(i)] for i in rng.permutation(level)]
    queries=[keys[int(i)] for i in rng.permutation(level)]
    return render(facts,queries,64 if level==8 else 128,{'split':namespace,'level':level,'epoch':epoch,
        'episode_index':index,'seed_sha256':seed})


def validate(row,*,all_queries=True):
    ids=row['input_ids'];pairs=row['bindings'];mapping=dict(pairs);positions=row['query_positions'];valid=row['valid_length']
    if len(mapping)!=len(pairs) or not pairs or not positions:raise ValueError('Empty/duplicate binding or Query')
    if valid!=3+3*len(pairs)+4*len(positions) or len(ids)!=row['storage_length'] or len(row['labels'])!=len(ids):raise ValueError('Episode length differs')
    if 'level' in row and len(pairs)!=row['level']:raise ValueError('Wrong binding count for level')
    if not all(8<=k<264 and 264<=v<520 for k,v in pairs):raise ValueError('Symbol ID outside frozen vocabulary')
    if ids[0]!=BOS or ids[valid-1]!=EOS or any(t==PAD for t in ids[:valid]) or any(t!=PAD for t in ids[valid:]):raise ValueError('Invalid prefix/padding')
    if ids[1:1+3*len(pairs)]!=[t for pair in pairs for t in (FACT,*pair)]:raise ValueError('Facts differ from mapping')
    if ids[1+3*len(pairs)]!=SEP or row['candidates']!=sorted(set(mapping.values())):raise ValueError('Separator/candidates differ')
    if all_queries and set(row['query_keys'])!=set(mapping):raise ValueError('Queries do not cover every Key')
    if len(set(row['query_keys']))!=len(row['query_keys']):raise ValueError('Repeated Query')
    for i,(pos,k) in enumerate(zip(positions,row['query_keys'])):
        if pos!=1+3*len(pairs)+1+4*i+2 or ids[pos-2:pos+2]!=[QUERY,k,ANSWER,mapping[k]]:
            raise ValueError('Causal shift/Query target mismatch')
        if row['labels'][pos]!=mapping[k]:raise ValueError('Wrong answer label')
    if any(y!=-100 for i,y in enumerate(row['labels']) if i not in positions):raise ValueError('Non-Query supervision')
    if row['mapping_hash']!=digest(sorted(pairs)) or row['episode_hash']!=digest(ids):raise ValueError('Mapping/Episode hash differs')
    return {'oracle_exact_match':1.,'candidate_count':len(row['candidates'])}


def forward_batch(model,rows):
    """Omit only trailing PAD; retain every real token and independent batch stream."""
    import torch
    if not rows or len({r['valid_length'] for r in rows})!=1:raise ValueError('Batch valid lengths differ')
    if any(any(t!=PAD for t in r['input_ids'][r['valid_length']:]) for r in rows):raise ValueError('Cannot trim non-PAD tokens')
    device=next(model.parameters()).device
    ids=torch.tensor([r['input_ids'][:r['valid_length']] for r in rows],device=device)
    return model(ids)


def forward(model,row):return forward_batch(model,[row])[0]


def isolated(rows,master,level):
    rng=random.Random(int(seed_for(master,'query-isolated',level,0,0),16));indices=list(range(len(rows)));rng.shuffle(indices)
    result=[]
    for i in indices[:64]:
        source=rows[i];keys=source['query_keys'].copy();rng.shuffle(keys)
        for k in keys[:4]:
            row=render(source['bindings'],[k],source['storage_length'],{'split':'validation','level':level,
                'diagnostic':'query-isolated','source_mapping_hash':source['mapping_hash']})
            validate(row,all_queries=False);result.append(row)
    return result


def prepare(folder,p):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);banks={};audit={};seen_train=set();seen_val=set()
    tokenizer=ROOT/'research/projects/cellular-language-models/experiments/epclm-004-unified-memory-10m/assets/tokenizer.json'
    vocab=json.loads(tokenizer.read_text())['model']['vocab']
    if file_hash(tokenizer)!=p['tokenizer_sha256'] or len(vocab)!=8192 or any(vocab[t]!=i for i,t in enumerate(TOKENS)):
        raise ValueError('STOP: frozen8192 tokenizer/IDs differ')
    for level,epochs in [(8,2),(16,6)]:
        train=[episode(p['master_seed'],'train',level,e,i) for e in range(epochs) for i in range(512)]
        val=[episode(p['master_seed'],'validation',level,0,i) for i in range(256)]
        for row in train+val:validate(row)
        tm={r['mapping_hash'] for r in train};vm={r['mapping_hash'] for r in val}
        if tm&vm or {r['episode_hash'] for r in train}&{r['episode_hash'] for r in val}:raise ValueError('Mapping/Episode partition leakage')
        if {r['seed_sha256'] for r in train}&{r['seed_sha256'] for r in val}:raise ValueError('Seed namespace leakage')
        seen_train|=tm;seen_val|=vm
        train_pairs={tuple(pair) for row in train for pair in row['bindings']};val_pairs={tuple(pair) for row in val for pair in row['bindings']}
        key_values=defaultdict(set)
        for row in train:
            for k,v in row['bindings']:key_values[k].add(v)
        audit[str(level)]={'prepared_training_episodes_maximum':len(train),'validation_episodes':len(val),'logical_length':64 if level==8 else 128,'valid_length':59 if level==8 else 115,'prepared_training_targets_maximum':len(train)*level,'training_mapping_count':len(tm),
            'validation_mapping_count':len(vm),'mapping_intersection':0,'episode_intersection':0,
            'keys_with_multiple_training_values':sum(len(v)>1 for v in key_values.values()),
            'training_episodes_with_repeated_values':sum(len(r['candidates'])<level for r in train),
            'candidate_count_histogram':dict(sorted(Counter(len(r['candidates']) for r in val).items())),
            'validation_random_active_accuracy':float(np.mean([1/len(r['candidates']) for r in val])),
            'validation_random_active_nll':float(np.mean([np.log(len(r['candidates'])) for r in val])),
            'oracle_exact_match':1.,'single_pair_overlap_allowed':True,'single_pair_overlap_count':len(train_pairs&val_pairs)}
        banks[str(level)]={'train':train,'validation':val,'query_isolated':isolated(val,p['master_seed'],level)}
    if seen_train&seen_val:raise ValueError('Cross-level complete Mapping leakage')
    for level,data in banks.items():
        for name,rows in data.items():
            path=folder/(level+'-'+name+'.json')
            if path.exists() and digest(json.loads(path.read_text()))!=digest(rows):raise ValueError('Existing P3 data changed')
            if not path.exists():write(path,rows)
    manifest={'format':'B0-P3-EPISODE-LOCAL-v1','master_seed':p['master_seed'],'tokenizer_sha256':p['tokenizer_sha256'],
        'structural_ids':{'BOS':BOS,'EOS':EOS,'PAD':PAD,'FACT':FACT,'QUERY':QUERY,'ANSWER':ANSWER,'SEP_alias_existing_HOLE':SEP},
        'levels':audit,'test_status':'SEALED_NOT_ACCESSED','train_test_intersection':'NOT_CHECKED_TEST_SEALED',
        'dataset_sha256':digest({level:{name:digest(rows) for name,rows in data.items()} for level,data in banks.items()})}
    manifest['files']={f.name:file_hash(f) for f in folder.glob('*.json') if f.name!='DATASET-MANIFEST.json'}
    frozen=BASE/'data/DATASET-MANIFEST.json'
    if frozen.exists() and json.loads(frozen.read_text())['dataset_sha256']!=manifest['dataset_sha256']:
        raise ValueError('Frozen P3 data identity differs')
    write(folder/'DATASET-MANIFEST.json',manifest)
    return banks,manifest


class StageSampler:
    """Reuse original transactional sampler separately for each fresh-mapping epoch."""
    def __init__(self,rows,master,level):self.rows=rows;self.master=master;self.level=level;self.epoch=0;self.sampler=self.new(0)
    def new(self,epoch):
        bank=self.rows[epoch*512:(epoch+1)*512]
        if len(bank)!=512:raise ValueError('No complete authorized epoch')
        return Sampler(bank,int(seed_for(self.master,'sampler',self.level,epoch,0)[:16],16),8)
    def propose(self):
        if self.sampler.cursor==512:self.epoch+=1;self.sampler=self.new(self.epoch)
        return self.sampler.propose()
    def resolve(self,accepted,ledger,accounting=None):self.sampler.resolve(accepted,ledger,accounting)
    def snapshot(self):return {'epoch':self.epoch,'state':self.sampler.snapshot(),'stage_data_sha256':digest(self.rows)}
    def restore(self,saved):
        if saved['stage_data_sha256']!=digest(self.rows):raise ValueError('Sampler data differs')
        self.epoch=saved['epoch'];self.sampler=self.new(self.epoch);self.sampler.restore(saved['state'])
