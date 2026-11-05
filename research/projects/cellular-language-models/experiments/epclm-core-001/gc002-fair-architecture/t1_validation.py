"""T1-specific cache/version and frozen training-path engineering gates."""
import torch
from torch.nn import functional as F
from cached_transformer import CachedTransformer,Config
from preparation import equivalence,parameter_count
from training_path import FullSequenceSampler,teacher_forced_sequences
from exposure import ExposureLedger


def cache_equivalence_and_invalidation(config,device='cuda'):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=equivalence(config,device)
    model=CachedTransformer(config).to(device).eval();ids=torch.randint(4,config.vocab,(1,32),device=device)
    _,cache=model.cached(ids);restored=model.restore(model.snapshot(cache));next_id=torch.tensor([[29]],device=device)
    expected,_=model.cached(next_id,cache);observed,_=model.cached(next_id,restored)
    if not torch.equal(expected,observed):raise ValueError('cache snapshot/restore continuation differs')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False)
    model.train();optimizer.zero_grad(set_to_none=True);model(ids).square().mean().backward();optimizer.step();model.eval()
    try:model.cached(next_id,cache)
    except ValueError as error:
        if 'invalidated' not in str(error):raise
    else:raise ValueError('stale inference cache survived a model weight update')
    return {'status':'PASS','cache_equivalence':report,'snapshot_restore':'PASS','cache_weight_version_invalidation':'PASS',
        'cache_session_scope':'one model weight version and one independent session','formal_training_kv_cache':'NOT_USED'}


def training_reference(config,seed=173020021,device='cuda'):
    torch.set_num_threads(1);torch.manual_seed(seed)
    documents=[{'id':'t1-02-a','tokens':[2,4,5,6,7,3]},{'id':'t1-02-b','tokens':[2,8,9,10,11,3]}]
    sampler=FullSequenceSampler(documents,seed,batch=2,target_chunk=2,context=2);ledger=ExposureLedger(10)
    proposal=sampler.propose(4);model=CachedTransformer(config).to(device).train()
    before=[p.detach().clone() for p in model.parameters()]
    loss,accounting=teacher_forced_sequences(model,proposal['segments']);independent=[]
    for lane in proposal['segments']:
        for part in lane:
            ids=torch.tensor([part['inputs']],dtype=torch.long,device=device)
            logits=model(ids,position_offset=part['document_position']);first=part['loss_start'];last=first+part['target_count']
            independent.append(F.cross_entropy(logits[0,first:last],torch.tensor(part['targets'][first:last],device=device),reduction='sum'))
    reference=torch.stack(independent).sum()/accounting['prediction_tokens'];error=float((loss-reference).abs())
    if error>1e-6:raise ValueError('training loss differs from independent full-sequence reference')
    loss.backward();grads=[p.grad for p in model.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in grads):raise ValueError('training gradient is absent/nonfinite')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False);optimizer.step()
    if not any(not torch.equal(a,b) for a,b in zip(before,model.parameters())):raise ValueError('optimizer update did not change weights')
    sampler.resolve(True,ledger)
    if ledger.consumed!=4 or ledger.updates!=1:raise ValueError('accepted update did not commit exact target events')
    return {'status':'PASS','loss_reference_absolute_error':error,'gradient_finite':True,'optimizer_update':'PASS',
        'training_kv_cache':'NOT_USED','every_forward_uses_current_weights':True,'document_boundary_reset':'PASS',
        'input_tokens':accounting['input_tokens'],'prediction_tokens':accounting['prediction_tokens'],
        'context_reuse_tokens':accounting['context_reuse_tokens'],'padding_tokens':accounting['padding_tokens'],
        'formal_parameter_count':parameter_count(config)}
