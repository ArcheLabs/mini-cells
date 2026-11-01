"""One compact ledger/unsupported-operator regression, no training matrix."""
import pytest
import torch
from exposure import ExposureLedger,reconstruct
from flops_audit import ArithmeticCounter


def test_committed_accounting_resume_and_fail_closed_flops():
    ledger=ExposureLedger(8,{'data':'fixed','seed':1})
    events=[('doc',1,1,4),('doc',2,4,3)]
    ledger.commit(events,accepted=False)
    assert ledger.report()['consumed_training_tokens']==0
    ledger.commit(events)
    resumed=ExposureLedger.restore(ledger.snapshot(),{'data':'fixed','seed':1})
    resumed.commit(events);ledger.commit(events)
    assert resumed.report()==ledger.report()
    assert resumed.report()['unique_training_tokens']==2 and resumed.report()['repeated_training_tokens']==2
    before=ledger.report()
    with pytest.raises(ValueError):ledger.commit([('doc',3,1,2),('doc',0,1,2)])
    assert ledger.report()==before
    with pytest.raises(ValueError):ExposureLedger.restore(ledger.snapshot(),{'data':'other'})
    docs=[{'id':'one','tokens':[1,4,3]},{'id':'two','tokens':[1,5,6,3]}]
    a=reconstruct(docs,7,target=9,batch=2,unroll=2)
    assert a==reconstruct(docs,7,target=9,batch=2,unroll=2)
    assert a['expected']['consumed_training_tokens']==9 and a['expected']['optimizer_updates']==3
    with ArithmeticCounter() as counter:
        x=torch.ones(2,3);y=torch.ones(3,4);x@y
        torch.erf(x) # intentionally unregistered arithmetic must be visible.
    assert counter.totals['forward']>=48 and counter.unknown['forward:aten.erf']==1


def test_parallel_teacher_forcing_lane_reset_and_rollback():
    from cached_transformer import CachedTransformer,Config
    from training_path import ChunkSampler,teacher_forced_lanes
    torch.set_num_threads(1);torch.manual_seed(713)
    docs=[{'id':'one','tokens':[1,4,3]},{'id':'two','tokens':[1,5,6,3]}]
    sampler=ChunkSampler(docs,7,batch=2);ledger=ExposureLedger(5)
    proposal=sampler.propose(9,unroll=2)
    with pytest.raises(ValueError):sampler.snapshot()
    sampler.resolve(False,ledger)
    assert sampler.propose(9,unroll=2)['events']==proposal['events']
    model=CachedTransformer(Config(vocab=32,dim=16,layers=1,heads=2,kv_heads=1,ffn=32,window=4)).train()
    before=[p.detach().clone() for p in model.parameters()]
    loss,caches,tokens=teacher_forced_lanes(model,proposal['segments']);loss.backward()
    assert tokens==4 and all(not value.requires_grad for cache in caches for pair in cache.layers for value in pair)
    assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
    sampler.resolve(True,ledger)
    state=sampler.snapshot();restored=ChunkSampler(docs,7,batch=2);restored.restore(state)
    proposal=sampler.propose(5,unroll=2)
    assert restored.propose(5,unroll=2)['events']==proposal['events']
    loss,next_caches,tokens=teacher_forced_lanes(model,proposal['segments'],caches)
    assert torch.isfinite(loss) and tokens==4
    for lane,parts in enumerate(proposal['segments']):
        # Last segment starts a new document when reset; no previous-document KV leaks.
        if parts[-1]['reset']:assert next_caches[lane].next_position==len(parts[-1]['inputs'])
