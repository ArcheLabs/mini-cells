"""Only new fusion/streaming behavior; frozen M2/M3 gate suites are not rerun."""
import copy
from pathlib import Path
import torch
from fusion import ThreeWayFusion
from e0 import masked_loss,load_checkpoint
from training import fit
from gates import recall_control


def tiny():
    return ThreeWayFusion(dict(vocab=64,dim=16,layers=1,heads=2,kv_heads=2,ffn=32,window=4),
                          ssm_layers=1,state_size=4,memory_dim=4,memory_hidden=8,chunk=2)


def correctness(out,device='cpu'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(41017)
    m=tiny().to(device);ids=torch.arange(16,device=device).reshape(2,8)%64
    logits,cell,w,_=m.forward_state(ids)
    assert torch.isfinite(w).all() and w.ge(0).all() and torch.allclose(w.sum(-1),torch.ones_like(w[...,0]))
    changed=ids.clone();changed[:,5:]=(changed[:,5:]+19)%64
    causal=float((logits[:,:5]-m(changed)[:,:5]).detach().abs().max());assert causal<2e-5
    isolated=float((m(ids[:1])-logits[:1]).detach().abs().max());assert isolated<2e-5
    loss=masked_loss(logits,torch.roll(ids,1,1));loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
    for prefix in ('swa.','ssm.','memory.','gate.'):
        assert sum(float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(prefix))>0,prefix
    m.eval();ablations={}
    with torch.no_grad():
        for disable in (('swa',),('ssm',),('memory',),('swa','ssm'),('swa','memory'),('ssm','memory')):
            z,state,g,_=m.forward_state(ids,cell,disabled=disable)
            assert torch.isfinite(z).all()
            for name in disable:assert g[...,m.branch_names.index(name)].eq(0).all()
            assert torch.allclose(g.sum(-1),torch.ones_like(g[...,0]))
            if 'memory' in disable:assert state is cell
            ablations['+'.join(disable)]={'gate_mean':g.mean((0,1)).cpu().tolist(),'cell_written':state is not cell}
    online=tiny().to(device).set_mode('online_adapt').eval();static=copy.deepcopy(online.state_dict())
    with torch.no_grad():
        full,fullcell,_,_=online.forward_state(ids)
        runtime=None;carried=None;parts=[]
        for part in ids.split(2,1):
            result,carried,_,runtime=online.forward_state(part,carried,runtime=runtime,cache=True);parts.append(result)
        cached=torch.cat(parts,1);error=float((cached-full).abs().max());assert error<4e-5
        for name in fullcell.fast:torch.testing.assert_close(fullcell.fast[name],carried.fast[name],atol=2e-5,rtol=2e-5)
        resources=online.resources(carried,runtime)
        assert resources['swa_runtime']['retained_storage_bytes']>0 and resources['ssm_runtime']['mamba_payload_bytes']>0
        snapshot=carried.detached();online.reset_runtime()
        try:online.forward_state(ids[:,:2],carried,runtime=runtime,cache=True)
        except ValueError as e:assert 'session' in str(e)
        else:raise AssertionError('runtime survived a session reset')
        assert all(torch.equal(carried.fast[n],snapshot.fast[n]) for n in carried.fast)
        readonly=online.set_mode('read_only');_,same,_,_=readonly.forward_state(ids,carried)
        assert same is carried
        try:readonly.forward_state(ids[:,:1],carried,runtime=runtime,cache=True)
        except ValueError:pass
        else:raise AssertionError('unaligned cache call accepted')
        try:readonly.forward_state(ids[:,:2],carried,disabled=('ssm',),runtime=runtime,cache=True)
        except ValueError:pass
        else:raise AssertionError('midstream ablation switch accepted')
    assert all(torch.equal(p,static[n]) for n,p in online.state_dict().items())
    # The unchanged M2 transaction loop, interrupted once and resumed for its remaining update.
    train=tiny().to(device);rows=[]
    for i in range(6):rows.append({'id':f'm4-{i}','input_ids':[(j+i)%64 for j in range(8)],
        'labels':[-100]*7+[(i+7)%64],'query_mask':[False]*7+[True]})
    config={'seed':41017,'batch':2,'lr':.001,'updates':2,'precision':'fp32','tf32':False,
            'optimizer':'AdamW','scheduler':'constant-engineering-smoke','checkpoint':{'every_updates':8,'every_seconds':120}}
    identity={'fixture':'m4-new-fusion-v1'}
    first=fit(train,rows[:4],rows[4:],config,out/'training',identity,stop_after=1)
    restored=tiny().to(device);final=fit(restored,rows[:4],rows[4:],config,out/'training',identity)
    assert first['ledger']['committed_updates']==1 and final['ledger']['committed_updates']==2
    assert final['history'][0]==first['history'][0] and len(restored.events)==1
    saved=load_checkpoint(out/'training/checkpoint.pt');clone=tiny().to(device);clone.load_state_dict(saved['model'])
    clone.eval();restored.eval()
    with torch.no_grad():restore_error=float((clone(ids)-restored(ids)).abs().max())
    assert restore_error==0
    return {'status':'PASS','device':str(device),'causal_error':causal,'batch_isolation_error':isolated,
            'full_vs_chunk_cache_error':error,'checkpoint_restore_error':restore_error,'ablations':ablations,
            'resources':resources,'training':final,'new_training_updates':2,'hypotheses':'NOT_EVALUATED'}
