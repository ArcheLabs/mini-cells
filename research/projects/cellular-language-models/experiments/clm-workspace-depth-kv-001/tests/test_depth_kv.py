"""Core depth/gradient, causal decode, and sequence lifecycle checks only."""
import json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE.parent/'clm-workspace-controller-001'))
sys.path.insert(1,str(HERE))
from models import Model,Session
from dependencies import shared,detach

torch.set_num_threads(1)
REFERENCE=json.loads((HERE/'tests/c2-reference.json').read_text())
C=REFERENCE['config']


def test_baseline_and_depth1_source_shared_parameters_and_gradients():
    ids=torch.tensor(REFERENCE['ids']);baseline=Model('C2',C,17)
    logits,_=baseline(ids);loss=logits.square().mean();loss.backward()
    torch.testing.assert_close(logits,torch.tensor(REFERENCE['logits']),atol=2e-6,rtol=2e-5)
    for name,p in baseline.named_parameters():
        if name in REFERENCE['gradients']:
            for key,value in (('norm',p.grad.norm()),('sum',p.grad.sum())):
                torch.testing.assert_close(value,torch.tensor(REFERENCE['gradients'][name][key]),atol=2e-7,rtol=2e-4)
    model=Model('C-DKV',C,17);model.trace_tensors=True;norm_inputs=[];reads=[];deltas=[];routers=[];updates=[];kv_outputs=[]
    assert all(torch.equal(p,model.state_dict()[name]) for name,p in baseline.state_dict().items())
    assert sum(p.numel() for p in baseline.parameters())==sum(p.numel() for p in model.parameters())
    def capture_kv(module,args,output):
        output.retain_grad();kv_outputs.append((args[0],output))
    hooks=[model.ws_norm.register_forward_pre_hook(lambda m,a:norm_inputs.append(a[0])),model.read_out.register_forward_hook(lambda m,a,o:reads.append(o)),model.router.register_forward_hook(lambda m,a,o:routers.append(o)),model.controller.register_forward_hook(lambda m,a,o:updates.append(o)),model.kv.register_forward_hook(capture_kv)]
    dispatch=model.dispatch
    def capture(*args):
        value=dispatch(*args);value.retain_grad();deltas.append(value);return value
    model.dispatch=capture
    output,cache=model(ids)
    for h in hooks:h.remove()
    h1=model.tensor_trace['depth1_input'];h1.retain_grad();routers[0].retain_grad()
    torch.testing.assert_close(h1,model.tensor_trace['encoder_output']+reads[0]+deltas[0])
    assert len(norm_inputs)==2 and norm_inputs[1] is h1 and len(routers)==2 and len(updates)==1
    expected=model.kv(model.ws_norm(h1)).view(1,ids.shape[1],2,C['heads'],C['dim']//C['heads']).unbind(2)
    position=torch.arange(ids.shape[1])[None,None,:,None]
    torch.testing.assert_close(cache['workspace_kv_depth1'][0],shared.rope_at(expected[0].transpose(1,2),position))
    torch.testing.assert_close(cache['workspace_kv_depth1'][1],expected[1].transpose(1,2))
    assert not torch.equal(cache['workspace_kv'][1],cache['workspace_kv_depth1'][1])
    torch.nn.functional.cross_entropy(output[:,:-1].reshape(-1,C['vocab']),ids[:,1:].reshape(-1)).backward()
    assert len(kv_outputs)==2 and kv_outputs[1][0].requires_grad
    for value in (h1,kv_outputs[1][1],routers[0],*deltas):assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert model.controller.weight_ih.grad.abs().sum()>0
    assert all(row['indices'].shape[-1]==2 for row in model.tensor_trace['steps'])
    frozen=detach(cache)
    assert all(not v.requires_grad for field in ('encoder_kv','workspace_kv','workspace_kv_depth1') for v in frozen[field])


def test_depth_causality_decode_and_different_cell_paths():
    model=Model('C-DKV',C).eval();ids=torch.arange(19)[None]%12+2;changed=ids.clone();changed[:,9:]=4
    with torch.no_grad():
        full,cache=model(ids);other,_=model(changed)
        torch.testing.assert_close(full[:,:9],other[:,:9],atol=3e-6,rtol=3e-5)
        for width in (1,4):
            session=Session(model);actual=torch.cat([session.prefill(part) for part in ids.split(width,1)],1)
            torch.testing.assert_close(full,actual,atol=3e-6,rtol=3e-5)
            for field in ('encoder_kv','workspace_kv','workspace_kv_depth1'):
                for a,b in zip(cache[field],session.cache[field]):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
        model.mode='fixed';model.fixed_pair=(0,1);a,ca=model(ids)
        model.fixed_pair=(6,7);b,cb=model(ids)
        assert a.shape==b.shape==full.shape and not torch.equal(a,b)
        assert ca['workspace_kv_depth1'][0].shape==cb['workspace_kv_depth1'][0].shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_depth_cache_batch_reorder_release_and_reset():
    model=Model('C-DKV',C).eval();a=torch.tensor([[2,4,5,6]]);b=torch.tensor([[2,9,8,7]])
    with torch.no_grad():
        batch=Session(model);out=batch.prefill(torch.cat((a,b)));sa=Session(model);sb=Session(model)
        from measure import cache_bytes,compute
        _,base_cache=Model('C2',C).eval()(torch.cat((a,b)))
        depth_memory=cache_bytes(batch.cache);base_memory=cache_bytes(base_cache)
        assert depth_memory['logical_by_cache']['workspace_kv_depth1']==base_memory['logical_by_cache']['workspace_kv']
        assert depth_memory['logical_total']-base_memory['logical_total']==base_memory['logical_by_cache']['workspace_kv']
        assert compute('C-DKV',C)['forward_flops_per_token']-compute('C2',C)['forward_flops_per_token']==4*C['dim']**2
        torch.testing.assert_close(out,torch.cat((sa.prefill(a),sb.prefill(b))),atol=2e-6,rtol=3e-5)
        batch.reorder([1,0]);next_ids=torch.tensor([[10],[11]])
        actual=batch.prefill(next_ids)
        expected=torch.cat((sb.prefill(next_ids[:1]),sa.prefill(next_ids[1:])))
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=3e-5)
        batch.reorder([0]);batch.prefill(torch.tensor([[3]]));assert batch.cache['workspace_kv_depth1'][0].shape[0]==1
        batch.close();assert batch.cache is None
        batch.reset('new');assert batch.cache['workspace_kv_depth1'] is None and batch.cache['position']==0
        torch.testing.assert_close(batch.prefill(a),Session(model).prefill(a),atol=0,rtol=0)


def test_paired_analysis_reports_seed_variance_and_rejects_unpaired_documents(tmp_path):
    from execute import analyze
    p={'experiment':{'seeds':[18001,18002,18003],'arms':['C2','C-DKV','T0-P','T0-F']}}
    case={'context_limit':128,'batch':1,'prompt_tokens':96,'generated_tokens':32,'prefill':{'p50_seconds':.01},'ttft':{'p95_seconds':.02},'tpot':{'p50_seconds':.01,'p95_seconds':.02},'decode_tokens_per_second':100,'peak_allocated_bytes':1,'kv_at_full_context':{'logical_total':1,'retained_storage_bytes':1}}
    results=[]
    for seed in p['experiment']['seeds']:
        for arm in p['experiment']['arms']:
            nll=3.+(.1 if arm=='C-DKV' else 0.)
            results.append({'seed':seed,'arm':arm,'validation':{'mean_nll':nll,'ppl':float(torch.exp(torch.tensor(nll))),'documents':[{'doc_id':'paired','targets':2,'nll_sum':nll*2}]},'benchmark':{'compute':{'forward_flops_per_token':1},'parameters':1,'cases':[case],'trained_parity':{'status':'PASS'}},'training_performance':{'session_targets_per_second':1}})
    aggregate=analyze(tmp_path,results,p)
    assert abs(aggregate['delta_nll_mean']-.1)<1e-8 and aggregate['delta_nll_std']==0
    assert aggregate['seed_count']==3 and not aggregate['scientific_confirmation']
    assert (tmp_path/'metrics/seed-18001.json').exists() and (tmp_path/'report/comparison.csv').exists()
    results[1]['validation']['documents'][0]['doc_id']='different'
    import pytest
    with pytest.raises(ValueError,match='paired validation'):analyze(tmp_path,results,p)
