"""Small deterministic oracles for new M5 behavior; no old test matrices."""
from pathlib import Path
import json
import torch
from torch import nn
from bmojo import BMojo,mamba_chunk,attention_read
from store import Store,Predictor
from data import MQARConfig,episode
from e0 import save_checkpoint,load_checkpoint
from training import fit


def tiny(policy='innovation',summary=True):
    return BMojo(dict(vocab=536,dim=16,layers=4,heads=2,kv_heads=2,ffn=32,window=4),policy=policy,
                 summary_enabled=summary,capacity=4,read_top_k=2,chunk=4,state_size=4,summary_tokens=2)


def gate1(out):
    torch.manual_seed(51001);m=tiny(None).eval();x=torch.randn(2,12,16);module=m.backbone.blocks[0].attn
    expected=module(x);state=None;parts=[]
    for part in x.split(4,1):y,state=mamba_chunk(module,part,state);parts.append(y)
    error=float((expected-torch.cat(parts,1)).detach().abs().max());assert error<2e-6
    ids=torch.tensor([episode(MQARConfig(bindings=2,length=32),'validation',i)['input_ids'] for i in range(2)])
    with torch.no_grad():
        full,s,_=m.forward_state(ids);future=ids.clone();future[:,13:]=520
        causal=float((full[:,:13]-m(future)[:,:13]).abs().max());assert causal<2e-5
        carried=None;chunks=[]
        for part in ids.split(4,1):y,carried,_=m.forward_state(part,carried);chunks.append(y)
        chunk_error=float((full-torch.cat(chunks,1)).abs().max());assert chunk_error<2e-5
        for i,projection in m.summary.items():
            summary=projection(carried.mamba[int(i)-1][1].mean(1)).view(2,2,16)
            assert summary.shape==(2,2,16) and summary.isfinite().all()
        m.reset_runtime()
        try:m.forward_state(ids[:,:4],carried)
        except ValueError:pass
        else:raise AssertionError('old session accepted')
    return {'status':'PASS','mamba_reference_error':error,'future_error':causal,'full_vs_chunk_error':chunk_error,
            'summary_count_per_swa':2,'summary_dimension':16,'summary_source':'actual preceding-chunk SSM state'}


def gate2(out):
    p=Predictor(.5);values=torch.ones(1,8,4);scores,mean=p.score(values)
    expected=torch.tensor([[.25**i for i in range(8)]])
    assert torch.equal(scores,expected)
    changed=values.clone();changed[:,5:]=10
    future,_=p.score(changed);assert torch.equal(scores[:,:5],future[:,:5])
    novel,_=p.score(torch.full((1,1,4),5.),mean);assert float(novel)>float(scores[0,-1])
    path=Path(out)/'predictor.pt';save_checkpoint(mean,path)
    a,state_a=p.score(values,mean);b,state_b=p.score(values,load_checkpoint(path))
    assert torch.equal(a,b) and torch.equal(state_a,state_b)
    reset,_=p.score(values);assert torch.equal(reset,scores)
    return {'status':'PASS','repetition_scores':scores.tolist(),'novel_score':float(novel),
            'future_unchanged':True,'restore_max_error':float((a-b).abs().max()),'learnable_predictor_parameters':0}


def gate3(out):
    report={}
    for capacity in (64,128):
        for policy in Store.policies:
            a=Store(capacity,policy,51003);b=Store(capacity,policy,51003)
            scores=[]
            for position in range(400):
                score=float((position*37)%101);scores.append(score);vector=torch.tensor([float(position),1.,2.,3.])
                a.admit(vector,position,score);b.admit(vector,position,score)
                assert len(a.records)<=capacity
            assert [r['position'] for r in a.records]==[r['position'] for r in b.records]
            if policy=='fifo':assert [r['position'] for r in a.records]==list(range(400-capacity,400))
            if policy=='innovation':assert sorted(r['score'] for r in a.records)==sorted(scores,reverse=True)[:capacity][::-1]
            restored=Store.restore(a.snapshot(),'cpu')
            for pos in range(400,410):
                a.admit(torch.ones(4),pos,float(pos));restored.admit(torch.ones(4),pos,float(pos))
            assert [r['position'] for r in a.records]==[r['position'] for r in restored.records]
            report[f'{policy}-{capacity}']=a.metrics(410)
    tie=Store(2,'innovation',1);tie.admit(torch.ones(2),0,1.);tie.admit(torch.ones(2),1,2.)
    assert not tie.admit(torch.ones(2),2,1.)
    assert tie.admit(torch.ones(2),3,3.) and [r['position'] for r in tie.records]==[1,3]
    return {'status':'PASS','strict_tie_rule':True,'fixed_capacity':report}


def gate4(out):
    m=tiny();a=m.backbone.blocks[1].attn;d=16
    with torch.no_grad():
        a.q.weight.zero_();a.k.weight.zero_();a.v.weight.zero_();a.o.weight.copy_(torch.eye(d))
        a.q.weight[:8,:8].copy_(torch.eye(8));a.k.weight[:8,:8].copy_(torch.eye(8));a.v.weight[:8,8:].copy_(torch.eye(8))
    store=Store(2,'innovation',1);first=torch.zeros(d);first[1]=first[9]=1
    second=torch.zeros(d);second[0]=second[8]=1
    store.admit(first,0,100.);store.admit(second,1,.1)
    query=torch.zeros(1,1,d);query[0,0,0]=1;positions=torch.tensor([2])
    actual,_,selected,_=attention_read(a,nn.Identity(),query,positions,None,None,[store],4,1)
    assert selected[0,0].tolist()==[1] and float(actual.abs().sum())>0
    store.records[0]['score']=0.;store.records[1]['score']=1000.
    same,_,again,_=attention_read(a,nn.Identity(),query,positions,None,None,[store],4,1)
    assert torch.equal(actual,same) and torch.equal(selected,again)
    empty,_,_,_=attention_read(a,nn.Identity(),query,positions,None,None,None,4,0)
    assert not torch.equal(actual,empty)
    summary=second[None,None];read_summary,_,_,_=attention_read(a,nn.Identity(),query,positions,None,summary,None,4,0)
    assert float(read_summary.abs().sum())>0
    local=query.clone();local[0,0,11]=1;read_local,_,_,_=attention_read(a,nn.Identity(),local,positions,None,None,None,4,0)
    assert float(read_local.abs().sum())>0
    sequence=torch.cat([query,local],1);past,_,_,_=attention_read(a,nn.Identity(),sequence,torch.tensor([2,3]),None,None,[store],4,1)
    sequence[:,1]=50.;altered,_,_,_=attention_read(a,nn.Identity(),sequence,torch.tensor([2,3]),None,None,[store],4,1)
    assert torch.equal(past[:,:1],altered[:,:1])
    return {'status':'PASS','selected_position':selected.tolist(),'query_overrides_innovation_rank':True,
            'local_summary_store_each_read':True,'future_error':float((past[:,:1]-altered[:,:1]).detach().abs().max()),
            'scope':'actual attention computation with controlled projection oracle; not model quality'}


def gate5(out):
    m=tiny().eval();row=episode(MQARConfig(bindings=2,length=48),'validation',500)
    ids=torch.tensor([row['input_ids']]);query=ids[:,-3:]
    with torch.no_grad():
        _,state,_=m.forward_state(ids);stats=m.checkpoint(state,Path(out)/'runtime')
        expected_records=[s.snapshot() for s in state.stores]
        expected,_,_=m.forward_state(query,state)
        m.reset_runtime();restored=m.restore(Path(out)/'runtime')
        for got,want in zip(restored.stores,expected_records):
            assert got.stats==want['stats']
            assert [(r['position'],r['score']) for r in got.records]==[(r['position'],r['score']) for r in want['records']]
            assert all(torch.equal(r['representation'],w['representation']) for r,w in zip(got.records,want['records']))
        actual,_,_=m.forward_state(query,restored);error=float((actual-expected).abs().max());assert error==0
        m.reset_runtime();only=m.restore(Path(out)/'runtime',store_only=True)
        assert only.mamba=={} and only.kv=={} and only.predictor is None
        position=only.position;recalled,_,trace=m.forward_state(query,only)
        assert any(any(p>=0 for p in item['selected_positions']) for item in trace)
        fresh=m.new_runtime(1,position);without,_,_=m.forward_state(query,fresh)
        contribution=float((recalled-without).abs().max());assert contribution>0
        m.reset_runtime();again=m.restore(Path(out)/'runtime',store_only=True);repeat,_,_=m.forward_state(query,again)
        assert torch.equal(recalled,repeat)
    manifest=json.loads((Path(out)/'runtime/manifest.json').read_text());component=Path(out)/'runtime'/manifest['generation']/'predictor.pt'
    original=component.read_bytes();component.write_bytes(original+b'corruption')
    try:m.restore(Path(out)/'runtime')
    except ValueError:pass
    else:raise AssertionError('corrupt component accepted')
    component.write_bytes(original)
    return {'status':'PASS','restore_max_error':error,'store_only_session_read':True,
            'store_vs_empty_logits_delta':contribution,'components_independent':True,**stats,
            'scope':'actual tokenizer-token representations, computed attention and logits; not claimed correct MQAR recall'}


def gate6(out):
    results={};common=None
    for name,policy,summary in [('E3',None,False),('E6',None,True),('E7-Random','random',True),('E7-FIFO','fifo',True),('E7-Innovation','innovation',True)]:
        model=tiny(policy,summary)
        if summary:
            weights={n:v.detach().clone() for n,v in model.state_dict().items()}
            if common is None:common=weights
            else:assert all(torch.equal(v,common[n]) for n,v in weights.items())
        rows={split:[episode(MQARConfig(bindings=2,length=32,seed=51006),split,i) for i in range(4)] for split in ('train','validation')}
        config={'seed':41017,'batch':1,'lr':.0003,'updates':2,'precision':'fp32','tf32':False,'optimizer':'AdamW',
                'scheduler':'constant-engineering-smoke','checkpoint':{'every_updates':8,'every_seconds':120}}
        result=fit(model,rows['train'],rows['validation'][:1],config,Path(out)/name,{'fixture':'m5-'+name})
        assert result['status']=='COMPLETE' and result['ledger']['committed_updates']==2
        assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
        assert all(r['parameter_change_norm']>0 for r in result['history'])
        if model.policy:assert any(s['replacements']>0 for event in model.events for s in event['store'])
        results[name]=result
    return {'status':'PASS','arms':results,'scope':'CPU tiny new M5 integration; canonical T4 is a separate gate'}
