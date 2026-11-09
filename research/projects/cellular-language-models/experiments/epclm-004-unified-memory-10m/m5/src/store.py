"""Parameter-free causal EMA and one capacity-bounded store per stream."""
import copy
import json
import math
import random
import torch


class Predictor:
    def __init__(self,beta=.875):
        if not 0<=beta<1:raise ValueError('EMA beta must be in [0,1)')
        self.beta=beta

    def score(self,values,mean=None):
        values=values.detach()
        mean=torch.zeros_like(values[:,0]) if mean is None else mean.detach()
        errors=[]
        for y in values.unbind(1):
            errors.append((y-mean).square().mean(-1))
            mean=self.beta*mean+(1-self.beta)*y
        result=torch.stack(errors,1)
        if not result.isfinite().all():raise FloatingPointError('nonfinite innovation')
        return result,mean


class Store:
    policies=('random','fifo','innovation')
    bins=(1e-8,1e-6,1e-4,.01,.1,1.,10.,float('inf'))

    def __init__(self,capacity,policy,seed):
        if capacity<1 or policy not in self.policies:raise ValueError('invalid store capacity/policy')
        self.capacity=capacity;self.policy=policy;self.seed=seed
        self.records=[];self.rng=random.Random(seed) if policy=='random' else None
        self.stats={'seen':0,'writes':0,'replacements':0,'full_attempts':0,'score_comparisons':0,'eviction_age_sum':0,
                    'score_sum':0.,'score_sq_sum':0.,'score_min':None,'score_max':None,'score_histogram':[0]*len(self.bins)}
        self.last_position=-1

    def admit(self,representation,position,score):
        score=float(score)
        if not math.isfinite(score) or score<0 or position<=self.last_position:raise ValueError('invalid score/position')
        self.last_position=position;s=self.stats;s['seen']+=1
        s['score_sum']+=score;s['score_sq_sum']+=score*score
        s['score_min']=score if s['score_min'] is None else min(s['score_min'],score)
        s['score_max']=score if s['score_max'] is None else max(s['score_max'],score)
        s['score_histogram'][next(i for i,b in enumerate(self.bins) if score<=b)]+=1
        replacement=None
        if len(self.records)==self.capacity:
            s['full_attempts']+=1
            if self.policy=='fifo':replacement=0
            elif self.policy=='innovation':
                s['score_comparisons']+=1
                replacement=min(range(len(self.records)),key=lambda i:(self.records[i]['score'],self.records[i]['position']))
                if score<=self.records[replacement]['score']:return False
            else:
                replacement=self.rng.randrange(s['seen'])
                if replacement>=self.capacity:return False
        if replacement is not None:
            s['eviction_age_sum']+=position-self.records[replacement]['position'];s['replacements']+=1
            self.records.pop(replacement)
        self.records.append({'representation':representation.clone(),'position':position,'score':score})
        self.records.sort(key=lambda r:r['position']);s['writes']+=1
        assert len(self.records)<=self.capacity
        return True

    def snapshot(self):
        return {'capacity':self.capacity,'policy':self.policy,'seed':self.seed,'last_position':self.last_position,
                'records':[{**r,'representation':r['representation'].detach().cpu().clone()} for r in self.records],
                'rng':self.rng.getstate() if self.rng else None,'stats':copy.deepcopy(self.stats)}

    @classmethod
    def restore(cls,saved,device):
        store=cls(saved['capacity'],saved['policy'],saved['seed']);store.last_position=saved['last_position']
        if len(saved['records'])>store.capacity:raise ValueError('store over capacity')
        positions=[r['position'] for r in saved['records']]
        if positions!=sorted(set(positions)) or any(p>store.last_position for p in positions):raise ValueError('invalid store order')
        store.records=[]
        for r in saved['records']:
            if not r['representation'].isfinite().all() or not math.isfinite(r['score']):raise ValueError('nonfinite record')
            store.records.append({**r,'representation':r['representation'].to(device).detach().clone()})
        store.stats=copy.deepcopy(saved['stats'])
        if store.rng:store.rng.setstate(saved['rng'])
        return store

    def detached(self):
        store=Store(self.capacity,self.policy,self.seed);store.last_position=self.last_position
        store.records=[{**r,'representation':r['representation'].detach().clone()} for r in self.records]
        store.stats=copy.deepcopy(self.stats)
        if store.rng:store.rng.setstate(self.rng.getstate())
        return store

    def metrics(self,now):
        s=self.stats;n=s['seen'];count=len(self.records)
        metadata={'capacity':self.capacity,'policy':self.policy,'seed':self.seed,'last_position':self.last_position,
                  'rng':self.rng.getstate() if self.rng else None,'stats':self.stats}
        metadata['records']=[{'position':r['position'],'score':r['score']} for r in self.records]
        return {**copy.deepcopy(s),'occupancy':count,'capacity':self.capacity,
                'admission_rate':s['writes']/n if n else 0.,
                'replacement_rate':s['replacements']/s['full_attempts'] if s['full_attempts'] else 0.,
                'mean_eviction_age':s['eviction_age_sum']/s['replacements'] if s['replacements'] else None,
                'average_stored_age':sum(now-r['position'] for r in self.records)/count if count else None,
                'score_mean':s['score_sum']/n if n else None,
                'score_variance':max(0.,s['score_sq_sum']/n-(s['score_sum']/n)**2) if n else None,
                'payload_bytes':sum(r['representation'].numel()*r['representation'].element_size() for r in self.records),
                'metadata_serialized_bytes':len(json.dumps(metadata).encode()),
                'metadata_scope':'JSON serialized positions/scores/counters/RNG; not Python allocator overhead'}
