"""Render disjoint A/B sessions from the frozen M1 association sampler."""
from dataclasses import replace

import numpy as np

from context_model import A_LENGTH,B_LENGTH
from data import ANSWER,BOS,FACT,HOLE,QUERY,MQARConfig,digest,episode

CONFIG=MQARConfig(bindings=16,length=512,queries=8,interference=1.,seed=51001,
                  regime='unseen_pairs')
EVAL_SEEDS=(61001,61002,61003)


def render(config,split,index):
    base=episode(config,split,index)
    rng=np.random.default_rng(int(digest(['h4-l1c-v1',base['id']])[:16],16))
    pairs=base['bindings']
    noise=base['input_ids'][49:base['query_positions'][0]-2]
    assert len(noise)==432 and all(520<=v<536 for v in noise)

    def facts(bindings):
        return [BOS]+[token for k,v in bindings for token in (FACT,8+k,264+v)]

    a=facts(pairs[:8])+noise[:247]
    b=facts(pairs[8:])+noise[247:431]
    selected=np.concatenate((rng.choice(8,4,replace=False),8+rng.choice(8,4,replace=False)))
    rng.shuffle(selected)
    positions,targets,sources=[],[],[]
    for i,pair in enumerate(selected):
        k,v=pairs[pair]
        b.extend((QUERY,8+k,ANSWER))
        positions.append(A_LENGTH+len(b)-1)
        targets.append(264+v)
        sources.append('A' if pair<8 else 'B')
        if i!=7:
            b.append(HOLE)
    assert len(a)==A_LENGTH and len(b)==B_LENGTH
    labels=[-100]*(A_LENGTH+B_LENGTH)
    for pos,target in zip(positions,targets):
        labels[pos]=target
    return dict(id=digest(['h4-l1c-v1',base['id']]),split=split,bindings=pairs,input_ids=a+b,
                labels=labels,query_mask=[v!=-100 for v in labels],query_positions=positions,
                query_sources=sources,source_seed=config.seed,session_lengths=[A_LENGTH,B_LENGTH])


def bank():
    train=[render(CONFIG,'train',i) for i in range(128)]
    validation=[render(CONFIG,'validation',i) for i in range(32)]
    evaluation={str(seed):[render(replace(CONFIG,seed=seed),'validation',i) for i in range(64)]
                for seed in EVAL_SEEDS}
    held=[*validation,*(r for rows in evaluation.values() for r in rows)]
    all_rows=train+held
    assert len({r['id'] for r in all_rows})==len(all_rows)
    assert not {tuple(p) for r in train for p in r['bindings']} & {tuple(p) for r in held for p in r['bindings']}
    for r in all_rows:
        assert r['query_sources'].count('A')==r['query_sources'].count('B')==4
        assert QUERY not in r['input_ids'][:A_LENGTH]
        for q in r['query_positions']:
            assert r['input_ids'][q]==ANSWER and 8<=r['input_ids'][q-1]<264 and r['input_ids'][q-2]==QUERY
        suffix=r['input_ids'][r['query_positions'][0]-2:]
        assert not any(264<=token<520 for token in suffix)
    return train,validation,evaluation,dict(format='epclm005.h4-l1c.dataset.v1',
        train=dict(episodes=len(train),sha256=digest(train)),
        validation=dict(episodes=len(validation),sha256=digest(validation)),
        evaluation={s:dict(episodes=len(rows),sha256=digest(rows)) for s,rows in evaluation.items()},
        train_heldout_pair_overlap=0,test_status='SEALED_NOT_ACCESSED',
        fact_to_first_b_answer=187,swa_receptive_field=156)
