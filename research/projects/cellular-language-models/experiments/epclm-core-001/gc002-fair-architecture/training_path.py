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


class FullSequenceSampler:
    """Transactional sequence windows: fresh targets plus masked overlapping context.

    Every forward is at most ``context + target_chunk`` tokens, never crosses a
    document boundary, and does not carry activations or KV across optimizer steps.
    """
    def __init__(self,documents,seed,batch=4,target_chunk=64,context=64):
        if not documents or any(len(d['tokens'])<2 for d in documents):raise ValueError('nonempty documents required')
        if len({d['id'] for d in documents})!=len(documents):raise ValueError('duplicate document IDs')
        if batch<1 or target_chunk<1 or context<0 or context+target_chunk>128:raise ValueError('invalid full-sequence window')
        self.documents=documents;self.seed=seed;self.batch=batch;self.target_chunk=target_chunk;self.context=context
        self.identity=hashlib.sha256(json.dumps(documents,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.rng=random.Random(seed);self.order=list(range(len(documents)));self.rng.shuffle(self.order)
        self.cursor=0;self.lanes=[None]*batch;self.positions=[0]*batch;self.pending=None

    def _state(self):
        return {'format':'gc002.full-sequence-sampler.v1','documents_sha256':self.identity,'seed':self.seed,
                'batch':self.batch,'target_chunk':self.target_chunk,'context':self.context,'rng':self.rng.getstate(),
                'order':self.order.copy(),'cursor':self.cursor,'lanes':self.lanes.copy(),'positions':self.positions.copy()}

    def snapshot(self):
        if self.pending is not None:raise ValueError('cannot checkpoint an uncommitted sequence transaction')
        return self._state()

    def restore(self,saved):
        if self.pending is not None:raise ValueError('resolve pending transaction before restoring')
        expected=(self.identity,self.seed,self.batch,self.target_chunk,self.context)
        observed=(saved.get('documents_sha256'),saved.get('seed'),saved.get('batch'),saved.get('target_chunk'),saved.get('context'))
        if saved.get('format')!='gc002.full-sequence-sampler.v1' or observed!=expected:raise ValueError('sequence sampler identity differs')
        self.rng.setstate(saved['rng']);self.order=saved['order'].copy();self.cursor=saved['cursor']
        self.lanes=saved['lanes'].copy();self.positions=saved['positions'].copy()

    def _next_document(self,lane):
        if self.cursor==len(self.order):self.rng.shuffle(self.order);self.cursor=0
        self.lanes[lane]=self.order[self.cursor];self.cursor+=1;self.positions[lane]=0

    def propose(self,remaining):
        if self.pending is not None or remaining<=0:raise ValueError('invalid or unresolved sequence transaction')
        before=self._state();segments=[[] for _ in range(self.batch)];events=[];accounting={'input_tokens':0,'context_reuse_tokens':0,'padding_tokens':0,'eos_targets':0}
        for lane in range(self.batch):
            needed=min(self.target_chunk,remaining-len(events))
            while needed>0:
                index=self.lanes[lane]
                if index is None or self.positions[lane]>=len(self.documents[index]['tokens'])-1:
                    self._next_document(lane);index=self.lanes[lane]
                document=self.documents[index];position=self.positions[lane]
                count=min(needed,len(document['tokens'])-1-position)
                start=max(0,position-self.context);end=position+count
                inputs=document['tokens'][start:end];targets=document['tokens'][start+1:end+1]
                if len(inputs)>128 or len(inputs)!=len(targets) or not count:raise ValueError('invalid full-sequence window')
                segments[lane].append({'reset':start==0,'document_id':document['id'],'document_position':start,
                    'inputs':inputs,'targets':targets,'loss_start':position-start,'target_count':count})
                accounting['input_tokens']+=len(inputs);accounting['context_reuse_tokens']+=position-start
                for offset in range(count):
                    source=position+offset
                    target=document['tokens'][source+1]
                    events.append((document['id'],source+1,document['tokens'][source],target))
                    accounting['eos_targets']+=int(target==3)
                self.positions[lane]+=count;needed-=count
        self.pending={'before':before,'events':events,'segments':segments,'accounting':accounting}
        return self.pending

    def resolve(self,accepted,ledger):
        if self.pending is None:raise ValueError('no pending sequence transaction')
        proposal=self.pending
        if accepted:
            ledger.commit(proposal['events'],accounting=proposal['accounting'])
            self.pending=None
        else:
            self.pending=None;self.restore(proposal['before'])


def teacher_forced_sequences(model,segments):
    """Full-sequence FP32 teacher forcing; every lane/document has an independent window."""
    losses=[];targets=0;inputs=0;context_reuse=0
    device=model.token_embedding.weight.device
    for parts in segments:
        for part in parts:
            ids=torch.tensor(part['inputs'],device=device,dtype=torch.long).unsqueeze(0)
            labels=torch.tensor(part['targets'],device=device,dtype=torch.long)
            logits=model(ids,position_offset=part['document_position'])
            first=part['loss_start'];last=first+part['target_count']
            if logits.shape[1]>128 or last>len(labels) or last>logits.shape[1]:raise ValueError('teacher-forcing mask/window differs')
            losses.append(F.cross_entropy(logits[0,first:last],labels[first:last],reduction='sum'))
            targets+=part['target_count'];inputs+=len(part['inputs']);context_reuse+=first
    if not targets:raise ValueError('empty full-sequence update')
    return sum(losses)/targets,{'prediction_tokens':targets,'input_tokens':inputs,
        'context_reuse_tokens':context_reuse,'padding_tokens':0}
