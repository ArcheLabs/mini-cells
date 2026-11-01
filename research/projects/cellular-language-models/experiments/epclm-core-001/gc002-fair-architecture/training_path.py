"""Transactional sampling and parallel per-lane teacher forcing; no training loop."""
import random
import hashlib
import json
import torch
from torch.nn import functional as F


class ChunkSampler:
    def __init__(self,documents,seed,batch=4):
        if not documents or any(len(d['tokens'])<2 for d in documents):raise ValueError('nonempty documents required')
        if len({d['id'] for d in documents})!=len(documents):raise ValueError('duplicate document IDs')
        self.identity=hashlib.sha256(json.dumps(documents,sort_keys=True,separators=(',',':')).encode()).hexdigest();self.seed=seed
        self.documents=documents;self.rng=random.Random(seed);self.order=list(range(len(documents)));self.rng.shuffle(self.order)
        self.cursor=0;self.lanes=[None]*batch;self.positions=[0]*batch;self.pending=None

    def _state(self):return {'format':'gc002.sampler.v1','documents_sha256':self.identity,'seed':self.seed,'batch':len(self.lanes),'rng':self.rng.getstate(),'order':self.order.copy(),'cursor':self.cursor,'lanes':self.lanes.copy(),'positions':self.positions.copy()}

    def snapshot(self):
        if self.pending is not None:raise ValueError('cannot checkpoint an uncommitted sampling transaction')
        return self._state()

    def restore(self,saved):
        if self.pending is not None:raise ValueError('resolve pending transaction before restoring')
        if saved.get('format')!='gc002.sampler.v1' or saved.get('documents_sha256')!=self.identity or saved.get('seed')!=self.seed or saved.get('batch')!=len(self.lanes):raise ValueError('sampler source/seed/batch differs')
        self.rng.setstate(saved['rng']);self.order=saved['order'].copy();self.cursor=saved['cursor'];self.lanes=saved['lanes'].copy();self.positions=saved['positions'].copy()

    def propose(self,remaining,unroll=64):
        if self.pending is not None or remaining<=0:raise ValueError('invalid or unresolved sampling transaction')
        before=self._state();segments=[[] for _ in self.lanes];events=[]
        for _ in range(unroll):
            if len(events)>=remaining:break
            for lane in range(len(self.lanes)):
                reset=self.lanes[lane] is None or self.positions[lane]>=len(self.documents[self.lanes[lane]]['tokens'])-1
                if reset:
                    if self.cursor==len(self.order):self.rng.shuffle(self.order);self.cursor=0
                    self.lanes[lane]=self.order[self.cursor];self.cursor+=1;self.positions[lane]=0
                if len(events)>=remaining:continue
                doc=self.documents[self.lanes[lane]];p=self.positions[lane]
                if not segments[lane] or reset:segments[lane].append({'reset':reset,'document_id':doc['id'],'inputs':[],'targets':[]})
                segments[lane][-1]['inputs'].append(doc['tokens'][p]);segments[lane][-1]['targets'].append(doc['tokens'][p+1])
                events.append((doc['id'],p+1,doc['tokens'][p],doc['tokens'][p+1]));self.positions[lane]+=1
        self.pending={'before':before,'events':events,'segments':segments}
        return self.pending

    def resolve(self,accepted,ledger):
        if self.pending is None:raise ValueError('no pending sampler transaction')
        pending=self.pending
        if accepted:ledger.commit(pending['events'])
        self.pending=None
        if not accepted:self.restore(pending['before'])


def teacher_forced_lanes(model,segments,caches=None):
    """Each lane has its own positions and document reset; parallelize within chunks."""
    caches=[None]*len(segments) if caches is None else list(caches)
    if len(caches)!=len(segments):raise ValueError('lane/cache count mismatch')
    losses=[];tokens=0
    for lane,parts in enumerate(segments):
        for part in parts:
            if part['reset']:caches[lane]=None
            ids=torch.tensor([part['inputs']],device=model.token_embedding.weight.device)
            targets=torch.tensor(part['targets'],device=ids.device)
            logits,caches[lane]=model._forward(ids,caches[lane],save_cache=True)
            losses.append(F.cross_entropy(logits[0],targets,reduction='sum'));tokens+=len(targets)
    if not tokens:raise ValueError('empty teacher-forced update')
    return sum(losses)/tokens,caches,tokens
