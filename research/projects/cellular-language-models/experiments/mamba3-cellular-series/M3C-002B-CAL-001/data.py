"""Development-only associative recall, no answer feedback or formal test split."""
import hashlib
import numpy as np

PAD,BOS,EOS,FACT,SEP,QUERY,ANSWER,HOLE=range(8)
KEY_START,VALUE_START,FILL_START=8,72,136
CONDITIONS=[(task,gap) for task in ('mqar','repeat') for gap in (0,32,64)]


def episode(split,index,seed):
    if split not in ('train-dev','validation-dev'):raise ValueError('formal/test data access is not implemented')
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'M3C-002B-CAL-001:{seed}:{split}:{index}'.encode()).digest()[:8],'big'))
    task,gap=CONDITIONS[index%6]
    keys=rng.choice(64,8,replace=False)+KEY_START;values=rng.choice(64,8,replace=False)+VALUE_START
    mapping=dict(zip(keys.tolist(),values.tolist()));ids=[BOS];positions={}
    for k,v in mapping.items():
        positions[k]=len(ids)+1;ids.extend([FACT,k,v,SEP])
    ids.extend(rng.integers(FILL_START,FILL_START+64,size=gap).tolist())
    if task=='mqar':queries=rng.permutation(keys).tolist()
    else:
        chosen=rng.choice(keys,4,replace=False);queries=rng.permutation(chosen).tolist()+rng.permutation(chosen).tolist()
    labels=np.full(160,-100,dtype=np.int64);metadata=[];seen={}
    for k in queries:
        t=len(ids)+2;labels[t]=mapping[k];seen[k]=seen.get(k,0)+1
        ids.extend([QUERY,k,ANSWER,HOLE])
        metadata.append({'episode':f'{split}:{index}','condition':f'{task}-gap{gap}','task':task,'gap':gap,
                         'query':len(metadata),'key':k,'key_position':positions[k],'answer_position':t,
                         'key_lag':t-positions[k],'binding_in16':int(t-positions[k]<16),'binding_in64':int(t-positions[k]<64),'occurrence':seen[k],'target':mapping[k]})
    ids.append(EOS);ids=np.pad(np.array(ids,dtype=np.int64),(0,160-len(ids)),constant_values=PAD)
    return ids,labels,metadata


def build(split,count,seed):
    rows=[episode(split,i,seed) for i in range(count)]
    return np.stack([r[0] for r in rows]),np.stack([r[1] for r in rows]),[r[2] for r in rows]


def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate(train,dev):
    signatures=[];mapping_signatures=[]
    for x,y,metadata in (train,dev):
        assert x.shape==y.shape and x.shape[1]==160
        assert np.all((y!=-100).sum(1)==8)
        signatures.append({digest(row) for row in x});bindings=set()
        assert len(signatures[-1])==len(x),'duplicate episode'
        for ids,labels,queries in zip(x,y,metadata):
            mapping={int(ids[i+1]):int(ids[i+2]) for i in range(len(ids)-3) if ids[i]==FACT}
            bindings.add(tuple(sorted(mapping.items())))
            for q in queries:
                t=q['answer_position'];assert ids[t]==ANSWER and ids[t+1]==HOLE
                assert labels[t]==mapping[q['key']]==q['target']
                assert not np.any((ids[t+1:]>=VALUE_START)&(ids[t+1:]<FILL_START))
        mapping_signatures.append(bindings)
    assert not mapping_signatures[0]&mapping_signatures[1],'train/development binding map overlap'
    assert not signatures[0]&signatures[1],'train/development overlap'
    return {'status':'PASS','oracle_exact_match':1.0,'targets_per_episode':8,'formal_test_accessed':False,
            'train_hash':digest(train[0]),'train_label_hash':digest(train[1]),
            'validation_hash':digest(dev[0]),'validation_label_hash':digest(dev[1])}
