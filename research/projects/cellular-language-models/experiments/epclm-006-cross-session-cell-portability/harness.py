"""Paired portability conditions, lifecycle probes and episode bootstrap."""
from collections import Counter
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from portable_model import P,POSITIONS,state_hash
from cell_artifact import CellSlot,export_cell,fork_cell,import_cell,delete_cell_artifact
from e0 import write

CONDITIONS=('C0','C1','C2','C3','C3-reset')


def tensors(model,row):
    device=next(model.parameters()).device
    return {key:torch.tensor([row[key]],device=device) for key in
            ('target_ids','query_packets','targets','source_segments','unrelated_segments')}


def sync(model):
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()


def runtime_bytes(runtime):
    return dict(mamba_bytes=sum(t.numel()*t.element_size() for pair in runtime.mamba_state.values() for t in pair),
                swa_bytes=sum(t.numel()*t.element_size() for pair in runtime.kv_cache.values() for t in pair))


@torch.no_grad()
def trial(model,row,identity,folder):
    x=tensors(model,row)
    correct=model.acquire(x['source_segments'])
    unrelated=model.acquire(x['unrelated_segments'])
    assert state_hash(correct)!=state_hash(unrelated)
    sync(model)
    t=time.perf_counter()
    hashes={name:export_cell(cell,folder/name,identity,uuid.uuid4().hex) for name,cell in
            (('correct',correct),('unrelated',unrelated))}
    export_seconds=time.perf_counter()-t
    slot=CellSlot(model,identity)
    results={}
    for name in CONDITIONS:
        sync(model)
        started=time.perf_counter()
        if name=='C1':
            slot.replace(folder/'correct')
        elif name=='C2':
            slot.replace(folder/'unrelated')
        else:
            slot.unload()
        sync(model)
        import_seconds=time.perf_counter()-started
        if name in ('C3','C3-reset'):
            slot.cell=correct
        runtime=None
        if name=='C3':
            # Creator-side reference explicitly retains source context.
            _,runtime=model.backbone.cached(x['source_segments'].flatten(1,2))
        sync(model)
        start=time.perf_counter()
        hidden,state=model.backbone.cached(x['target_ids'],runtime)
        logits=slot.read(hidden[:,POSITIONS],x['query_packets'])
        sync(model)
        pred=logits.argmax(-1)
        hits=pred.eq(x['targets'])
        results[name]=dict(recall=float(hits.float().mean()),exact=float(hits.float().mean()),episode_exact=float(hits.all()),
                           nll=float(F.cross_entropy(logits.flatten(0,1),x['targets'].flatten())),
                           predictions=pred[0].tolist(),targets=x['targets'][0].tolist(),
                           query_seconds=time.perf_counter()-start,import_seconds=import_seconds,
                           source_runtime_transferred=False,creator_context_retained=name=='C3',
                           source_history_replayed=False,cell_state=state_hash(slot.cell) if slot.cell else None,
                           **runtime_bytes(state))
        if name=='C1':
            expected=logits.cpu()
    assert results['C1']['cell_state']==state_hash(correct)
    assert results['C2']['cell_state']==state_hash(unrelated)
    torch.testing.assert_close(expected,logits.cpu(),**P['tolerance'])
    return dict(id=row['id'],query_fact_ids=row['query_fact_ids'],conditions=results,
                artifact_sha256=hashes,export_seconds=export_seconds,
                cell_bytes=sum(p.stat().st_size for p in (folder/'correct').iterdir())),expected


@torch.no_grad()
def lifecycle(model,row,identity,folder):
    x=tensors(model,row)
    slot=CellSlot(model,identity)
    original=slot.replace(folder/'correct')
    before=state_hash(original)
    # Bad identity cannot replace the live attachment.
    wrong=CellSlot(model,{**identity,'backbone_checkpoint_id':'wrong'})
    wrong.cell=original
    try:
        wrong.replace(folder/'unrelated')
    except ValueError:
        assert wrong.cell is original
    else:
        raise AssertionError('Incompatible replacement succeeded')
    slot.replace(folder/'unrelated')
    hidden,state=model.backbone.cached(x['target_ids'])
    replaced=slot.read(hidden[:,POSITIONS],x['query_packets'])
    assert state_hash(slot.cell)!=before
    fork=fork_cell(model,original)
    assert fork.metadata['parent']['cell_id']==original.metadata['cell_id']
    for group in ('fast','momentum'):
        for key,tensor in getattr(original,group).items():
            assert tensor.untyped_storage().data_ptr()!=getattr(fork,group)[key].untyped_storage().data_ptr()
    changed=model.acquire(x['unrelated_segments'],fork)
    assert state_hash(changed)!=before and state_hash(original)==before
    slot.unload()
    # Attachment removal leaves the live E3 Runtime intact.
    state_bytes=runtime_bytes(state)
    _,continued=model.backbone.cached(x['target_ids'][:,:1],state)
    assert continued.position==state.position+1 and state_bytes==runtime_bytes(state)
    baseline=slot.read(hidden[:,POSITIONS],x['query_packets'])
    torch.testing.assert_close(baseline,model.lm_head(hidden[:,POSITIONS]),atol=0,rtol=0)
    export_cell(changed,folder/'fork',identity,uuid.uuid4().hex)
    assert state_hash(import_cell(folder/'fork',model,identity))==state_hash(changed)
    delete_cell_artifact(folder/'fork')
    assert not (folder/'fork').exists() and (folder/'correct').exists()
    return dict(status='PASS',replacement='PASS',failed_replacement_preserved=True,
                fork='PASS',unload='PASS',delete_local_copy='PASS',
                existing_session_state_retained=state_bytes,replaced_argmax=replaced.argmax(-1).tolist())


def compare(banks):
    rng=np.random.default_rng(P['bootstrap_seed'])
    result={}
    for metric in ('recall','exact','episode_exact','nll'):
        vals=np.array([[[row['conditions'][c][metric] for row in rows] for rows in banks.values()]
                       for c in CONDITIONS])
        means=vals.mean((1,2))
        result[metric]=dict(conditions=dict(zip(CONDITIONS,means.tolist())),
                            per_seed={c:dict(zip(banks,vals[i].mean(1).tolist())) for i,c in enumerate(CONDITIONS)})
        for j in (0,2,3,4):
            delta=vals[1]-vals[j]
            draws=rng.integers(delta.shape[1],size=(len(banks),P['bootstrap_replicates'],delta.shape[1]))
            sample=np.take_along_axis(delta[:,None,:],draws,axis=2).mean((0,2))
            result[metric][f'C1-{CONDITIONS[j]}']=dict(mean=float(delta.mean()),ci95=np.quantile(sample,[.025,.975]).tolist())
    result['argmax_counts']={c:dict(Counter(p for rows in banks.values() for r in rows
                                     for p in r['conditions'][c]['predictions'])) for c in CONDITIONS}
    return result
