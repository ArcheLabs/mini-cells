"""Query diagnostics and independent Cell/SSM persistence checks."""
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from model import A_LENGTH, cell_hash
from adapter import load_session,save_session,tensor_digest
from data import digest
from e0 import write


def input_batch(rows,device):
    ids=torch.tensor([r['input_ids'] for r in rows],device=device)
    q=torch.tensor([[p-A_LENGTH for p in r['query_positions']] for r in rows],device=device)
    y=torch.tensor([[r['labels'][p] for p in r['query_positions']] for r in rows],device=device)
    a=torch.tensor([[s=='A' for s in r['query_sources']] for r in rows],device=device)
    return ids[:,:A_LENGTH],ids[:,A_LENGTH:],q,y,a


def at_query(tensor,q):
    return tensor[torch.arange(len(q),device=q.device)[:,None],q]


def summarize(rows,logits,targets,source):
    probs=logits.softmax(-1)
    sorted_logits,top=logits.topk(10,dim=-1)
    correct=top[...,0].eq(targets)
    target_logit=logits.gather(-1,targets[...,None])
    ranks=(logits>target_logit).sum(-1)+1
    nll=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').reshape_as(targets)
    entropy=-(probs*logits.log_softmax(-1)).sum(-1)
    result=[]
    for i,row in enumerate(rows):
        result.append(dict(id=row['id'],source_a_em=float(correct[i,source[i]].float().mean()),
                           source_b_em=float(correct[i,~source[i]].float().mean()),
                           exact_match=float(correct[i].float().mean()),episode_em=float(correct[i].all()),
                           nll=float(nll[i].mean()),predictions=top[i,:,0].tolist(),targets=targets[i].tolist(),
                           query_sources=row['query_sources'],correct=correct[i].tolist(),
                           top5=top[i,:,:5].eq(targets[i,:,None]).any(-1).tolist(),
                           top10=top[i].eq(targets[i,:,None]).any(-1).tolist(),
                           target_probability=(-nll[i]).exp().tolist(),entropy=entropy[i].tolist(),
                           margin=(sorted_logits[i,:,0]-sorted_logits[i,:,1]).tolist(),
                           target_rank=ranks[i].tolist(),token_432_logit=logits[i,:,432].tolist()))
    return result


@torch.no_grad()
def evaluate_batch(model,rows,ablations=False):
    device=next(model.parameters()).device
    a,b,q,y,source=input_batch(rows,device)
    model.eval()
    cell,gate_a,_=model.write_a(a,diagnose=True)
    before=cell_hash(cell)
    logits,gate_b,_,_=model.read_b(b,cell)
    assert before==cell_hash(cell)
    qgate=at_query(gate_b,q)
    normal=torch.ones(gate_b.shape[:2],dtype=torch.bool,device=device)
    normal.scatter_(1,q,False)
    gate=dict(A_mean=gate_a.mean((0,1)).tolist(),B_normal_mean=gate_b[normal].mean(0).tolist(),
              query_A_mean=qgate[source].mean(0).tolist(),query_B_mean=qgate[~source].mean(0).tolist(),
              query_weights=qgate.tolist())
    outcomes={'natural':summarize(rows,at_query(logits,q),y,source)}
    if ablations:
        for name,disabled in [('no_cell',('memory',)),('no_ssm',('ssm',)),('swa_only',('memory','ssm'))]:
            outputs=model.read_b(b,cell,disabled)[0]
            outcomes[name]=summarize(rows,at_query(outputs,q),y,source)
        assert before==cell_hash(cell)
    return dict(metrics=outcomes,gates=gate,cell=before)


@torch.no_grad()
def persist(model,rows,folder,identity,tolerance):
    model.eval()
    device=next(model.parameters()).device
    a,b,q,_,_=input_batch(rows,device)
    cell,_,_=model.write_a(a)
    reference=at_query(model.read_b(b,cell)[0],q)
    # At the query boundary retain B-local SWA and SSM for the reference path.
    # On restore B-local SWA is reconstructed from B input only.
    cut=208
    assert q.min()>=cut and cut%model.chunk==0
    _,_,swa,ssm=model.read_b(b[:,:cut],cell,cache=True)
    expected=at_query(model.read_b(b[:,cut:],cell,cache=True,swa_state=swa,ssm_state=ssm)[0],q-cut)
    torch.testing.assert_close(expected,reference,**tolerance)
    key={**identity,'batch_sha256':digest([r['id'] for r in rows]),
         'static_sha256':tensor_digest(model.state_dict()),'branches':list(model.disabled)}
    started=time.perf_counter()
    hashes=save_session(model,folder,cell,ssm,key,expected)
    write(folder/'receipt.json',dict(key=key,hashes=hashes,b_prefix=b[:,:cut].tolist(),
                                   b_suffix=b[:,cut:].tolist(),q=(q-cut).tolist(),
                                   cell=cell_hash(cell),save_seconds=time.perf_counter()-started,
                                   ssm_bytes=0 if ssm is None else sum(t.numel()*t.element_size()
                                       for pair in ssm.mamba_state.values() for t in pair),
                                   cell_bytes=0 if cell is None else sum(t.numel()*t.element_size()
                                       for group in (cell.fast,cell.momentum) for t in group.values()),
                                   swa_kv_bytes=sum(t.numel()*t.element_size() for pair in swa.layers for t in pair),
                                   full_cached_error=float((expected-reference).abs().max())))


@torch.no_grad()
def restore(model,folder,tolerance):
    import json
    import os
    r=json.loads((folder/'receipt.json').read_text())
    if r['key']['static_sha256']!=tensor_digest(model.state_dict()) or r['key']['branches']!=list(model.disabled):
        raise ValueError('L1-C restoration revision mismatch')
    started=time.perf_counter()
    cell,ssm,expected=load_session(model,folder,r['key'],r['hashes'])
    device=next(model.parameters()).device
    assert cell_hash(cell)==r['cell']
    _,swa=model.swa.cached(torch.tensor(r['b_prefix'],device=device))
    actual=at_query(model.read_b(torch.tensor(r['b_suffix'],device=device),cell,
                                cache=True,swa_state=swa,ssm_state=ssm)[0],
                    torch.tensor(r['q'],device=device)).cpu()
    torch.testing.assert_close(actual,expected,**tolerance)
    assert cell_hash(cell)==r['cell']
    return dict(status='PASS',max_abs=float((actual-expected).abs().max()),
                seconds=time.perf_counter()-started,pid=os.getpid())
