"""Small synthetic regression for full-sequence semantics and cache invalidation."""
import pytest
import torch
from torch.nn import functional as F
from cached_transformer import CachedTransformer,Config
from training_path import FullSequenceSampler,teacher_forced_sequences
from exposure import ExposureLedger
from formal_training import FormalTrainer,verify_resume
from storage import bundle,reuse_local_assets


def test_full_sequence_training_reset_loss_gradient_and_cache_weight_version():
    torch.set_num_threads(1);torch.manual_seed(713)
    config=Config(vocab=32,dim=16,layers=1,heads=2,kv_heads=1,ffn=32,window=4)
    docs=[{'id':'a','tokens':[2,4,5,6,3]},{'id':'b','tokens':[2,7,8,9,3]}]
    sampler=FullSequenceSampler(docs,19,batch=2,target_chunk=2,context=2);ledger=ExposureLedger(8)
    first=sampler.propose(4);sampler.resolve(False,ledger)
    assert sampler.propose(4)['events']==first['events']
    proposal=sampler.pending
    assert all(len(part['inputs'])<=4 and part['document_id'] in ('a','b') for lane in proposal['segments'] for part in lane)
    assert all(part['reset'] for lane in proposal['segments'] for part in lane if part['document_position']==0)
    model=CachedTransformer(config).train();before=[p.detach().clone() for p in model.parameters()]
    loss,accounting=teacher_forced_sequences(model,proposal['segments'])
    reference=[]
    for lane in proposal['segments']:
        for part in lane:
            ids=torch.tensor(part['inputs']).unsqueeze(0)
            logits=model(ids,position_offset=part['document_position'])
            lo=part['loss_start'];hi=lo+part['target_count']
            reference.append(F.cross_entropy(logits[0,lo:hi],torch.tensor(part['targets'][lo:hi]),reduction='sum'))
    ref=torch.stack(reference).sum()/accounting['prediction_tokens']
    assert torch.allclose(loss,ref,atol=1e-7,rtol=0)
    assert accounting['prediction_tokens']==4 and accounting['padding_tokens']==0
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,foreach=False);optimizer.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
    sampler.resolve(True,ledger)
    following=sampler.propose(4)
    assert following['accounting']['context_reuse_tokens']>0
    sampler.resolve(False,ledger)

    model.eval();_,cache=model.cached(torch.tensor([[2,4]]))
    optimizer.zero_grad(set_to_none=True);model(torch.tensor([[2,4,5]])).sum().backward();optimizer.step()
    with pytest.raises(ValueError,match='invalidated'):
        model.cached(torch.tensor([[6]]),cache)


def test_formal_update_flops_and_interrupted_resume_are_atomic(tmp_path):
    torch.set_num_threads(1)
    protocol={'model':{'vocab':32,'dim':16,'layers':1,'heads':2,'kv_heads':1,'ffn':32,'window':4},'seed':19,
        'training':{'batch_lanes':2,'fresh_prediction_targets_per_lane':2,'context_reuse_per_sequence_max':2,'prediction_token_budget':8},
        'optimizer':{'learning_rate':1e-3,'betas':[.9,.95],'weight_decay':.1,'foreach':False,'gradient_clip_norm':1.},
        'scheduler':{'total_updates':2,'warmup_updates':1}}
    documents=[{'id':'a','tokens':[2,4,5,6,7,3]},{'id':'b','tokens':[2,8,9,10,11,3]}]
    identity={'source':'synthetic','protocol':'fixture','data':'fixture','tokenizer':'fixture','seed':19}
    engine=FormalTrainer(documents,identity,protocol,'cpu',target_tokens=8);row,_=engine.update()
    assert row['phase_flops']['forward']>0 and row['phase_flops']['backward']>0 and row['phase_flops']['optimizer']>0
    assert not row['unsupported_operators']
    engine=verify_resume(engine,tmp_path/'formal.pt')
    assert engine.resume_result['status']=='PASS' and engine.ledger.consumed==8


def test_exact_source_bundle_restores_last_committed_report_and_checkpoint(tmp_path):
    output=tmp_path/'output';output.mkdir()
    (output/'private').mkdir();(output/'private/training.pt').write_bytes(b'committed-checkpoint')
    (output/'decision.json').write_text('{"committed": true}\n')
    bundle(output,{'source':'same-source','protocol':'same-protocol'})
    (output/'private/training.pt').write_bytes(b'partial-replacement')
    (output/'decision.json').write_text('{partial json')
    assert reuse_local_assets(output,'same-source','same-protocol',None)
    assert (output/'private/training.pt').read_bytes()==b'committed-checkpoint'
    assert (output/'decision.json').read_text()=='{"committed": true}\n'
