import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from baselines import build
from training import fit
from e0 import load_checkpoint
from data import MQARConfig,episode


def fixture():
    c=dict(vocab=536,dim=16,layers=1,heads=2,kv_heads=2,ffn=32,window=4)
    rows={s:[episode(MQARConfig(bindings=2,length=24),s,i) for i in range(16)] for s in ('train','validation')}
    config={'seed':17,'batch':2,'lr':.001,'updates':12,'checkpoint':{'every_updates':8,'every_seconds':120}}
    return c,rows,config


def test_periodic_checkpoint_exact_resume_same_seed_and_timeout(tmp_path):
    c,rows,config=fixture();identity={'fixture':'m2-checkpoint'}
    a=build('E1',c,17);result=fit(a,rows['train'],rows['validation'],config,tmp_path/'whole',identity)
    assert result['checkpoint_policy']['writes_this_process']==3  # init +8 +12, rather than 13 writes
    b=build('E1',c,17);fit(b,rows['train'],rows['validation'],config,tmp_path/'split',identity,stop_after=5)
    b=build('E1',c,999);resumed=fit(b,rows['train'],rows['validation'],config,tmp_path/'split',identity)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert result['ledger']==resumed['ledger']
    assert [r['input_sha256'] for r in result['history']]==[r['input_sha256'] for r in resumed['history']]
    state=load_checkpoint(tmp_path/'split/checkpoint.pt')
    assert state['ledger']==resumed['ledger'] and len(state['history'])==12
    for key in ('model','optimizer','rng','sampler','ledger'):assert key in state
    import time
    stopped=fit(build('E1',c,17),rows['train'],rows['validation'],config,tmp_path/'timeout',identity,deadline=time.monotonic())
    assert stopped['status']=='INCOMPLETE' and stopped['checkpoint']['durable_updates']==0


def test_prestep_flush_and_unsafe_update_durable_rollback(tmp_path,monkeypatch):
    c,rows,config=fixture();identity={'fixture':'m2-failure'}
    model=build('E1',c,17);calls=[0]
    def fail_gradient(g):
        calls[0]+=1
        return torch.full_like(g,float('nan')) if calls[0]==3 else g
    model.token_embedding.weight.register_hook(fail_gradient)
    with pytest.raises(FloatingPointError):fit(model,rows['train'],rows['validation'],config,tmp_path/'grad',identity)
    state=load_checkpoint(tmp_path/'grad/checkpoint.pt')
    assert state['ledger']['committed_updates']==2 and len(state['history'])==2
    # Mutate parameters on attempt 10: saved prefix 8 survives, volatile update 9 is discarded.
    original=torch.optim.AdamW.step;calls[0]=0
    def corrupt(optimizer,*a,**kw):
        calls[0]+=1;original(optimizer,*a,**kw)
        if calls[0]==10:
            with torch.no_grad():optimizer.param_groups[0]['params'][0].fill_(float('nan'))
    monkeypatch.setattr(torch.optim.AdamW,'step',corrupt)
    model=build('E1',c,17)
    with pytest.raises(FloatingPointError):fit(model,rows['train'],rows['validation'],config,tmp_path/'optimizer',identity)
    state=load_checkpoint(tmp_path/'optimizer/checkpoint.pt')
    report=json.loads((tmp_path/'optimizer/metrics.json').read_text())
    assert state['ledger']['committed_updates']==report['ledger']['committed_updates']==8
    assert len(state['history'])==8 and state['ledger']['query_target_tokens']==16
    for k,v in model.state_dict().items():torch.testing.assert_close(v,state['model'][k],rtol=0,atol=0)
    assert all(p.isfinite().all() for p in model.parameters())
