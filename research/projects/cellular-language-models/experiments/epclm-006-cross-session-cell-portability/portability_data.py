"""Disjoint Token-ID domains and deterministic cross-session episodes."""
import numpy as np

from portable_model import P,SOURCE,TARGET,POSITIONS
from data import ANSWER,BOS,FACT,QUERY,digest

POOLS={'train':(0,128),'validation':(128,192),'test':(192,256)}
VERSION='epclm006-episodic-v1'


def episode(split,seed,index):
    if split not in ('train','validation'):
        raise ValueError('Independent Test remains sealed')
    identity=digest([VERSION,split,seed,index])
    rng=np.random.default_rng(int(identity[:16],16))
    lo,hi=POOLS[split]
    keys=rng.choice(np.arange(lo,hi),16,replace=False).tolist()
    values=rng.choice(np.arange(lo,hi),16,replace=False).tolist()
    support=[[BOS,FACT,8+k,264+v] for k,v in zip(keys,values)]
    selected=rng.choice(8,P['queries'],replace=False).tolist()
    packets=[[BOS,int(rng.choice([QUERY,536,537])),8+keys[i],ANSWER] for i in selected]
    target=rng.integers(520,536,size=P['target_prefix_tokens']).tolist()+[t for q in packets for t in q]
    source=[t for row in support[:8] for t in row]
    labels=[-100]*(SOURCE+TARGET)
    for i,pos in zip(selected,POSITIONS):
        labels[SOURCE+pos]=264+values[i]
    assert len(source)==SOURCE and len(target)==TARGET
    assert all(not 264<=token<520 for token in target)
    return dict(id=identity,split=split,seed=seed,input_ids=source+target,labels=labels,
                query_mask=[x!=-100 for x in labels],target_ids=target,query_packets=packets,
                query_positions=list(POSITIONS),targets=[264+values[i] for i in selected],
                fact_ids=[digest([identity,k,v]) for k,v in zip(keys,values)],
                facts=list(zip(keys[:8],values[:8])),unrelated_facts=list(zip(keys[8:],values[8:])),
                source_segments=support[:8],unrelated_segments=support[8:],
                query_fact_ids=[digest([identity,keys[i],values[i]]) for i in selected])


def datasets(tokenizer):
    train=[episode('train',P['train_seed'],i) for i in range(P['train_episodes'])]
    val=[episode('validation',P['validation_seed'],i) for i in range(P['validation_episodes'])]
    banks={str(seed):[episode('validation',seed,i) for i in range(P['evaluation_episodes_per_seed'])]
           for seed in P['evaluation_seeds']}
    held=val+[r for bank in banks.values() for r in bank]
    assert len({r['id'] for r in train+held})==len(train+held)
    for index in (0,1):
        assert not ({f[index] for r in train for f in r['facts']+r['unrelated_facts']} &
                    {f[index] for r in held for f in r['facts']+r['unrelated_facts']})
    return train,val,banks,dict(format=VERSION,tokenizer_sha256=tokenizer,
          train_sha256=digest(train),validation_sha256=digest(val),
          evaluation_sha256={s:digest(rows) for s,rows in banks.items()},
          test_status='SEALED_NOT_ACCESSED',test_recipe_sha256=digest([VERSION,'test',POOLS['test']]),
          seed_namespaces=dict(train=P['train_seed'],validation=P['validation_seed'],evaluation=P['evaluation_seeds']),
          entity_and_answer_pools=POOLS,query_tokens_per_episode=P['queries'],
          input_tokens_per_episode=SOURCE+TARGET)
