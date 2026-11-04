"""Fixed 32-token commits, cross-Chunk Token KV and explicit partial-sequence finish."""
import copy
import torch
from bootstrap import old_runtime,shared,BASE
from .state_read import attention,ffn_context

SOURCE=shared.sha({p.name:shared.digest(p) for p in sorted(BASE.glob('*.py'))})


def empty(model,batch):
    cache=old_runtime.empty(model,batch)
    cache.update(state_memory={},state_revision=0,closed=False)
    return cache


def stages(model,x,cache,start):
    sources=[]
    for i,stage in enumerate(model.stages):
        a=attention(model,stage,stage.n1(x+stage.step[0][None,None]),cache,i,start)
        f=stage.ff(ffn_context(model,stage,stage.n2(x+a),cache,i))
        sources.append(a if model.arm=='B1' else f if model.arm=='B2' else (a+f)/2)
        x=x+shared.N1_ALPHA*(a+f)
    return x,torch.stack(sources).mean(0)


def flush(model,cache,close=False):
    if cache['pending']:
        if model.write_enabled:cache['persistent']=model.persistent.write(cache['persistent'],torch.cat(cache['pending'],1))
        if model.reset_every_chunk:cache['persistent']=model.persistent.init(cache['batch'])
        cache['pending']=[];cache['chunk_tokens']=0;cache['state_memory']={};cache['state_revision']+=1
    if close:cache['closed']=True


def append(model,cache,features):
    cache['pending'].append(features);cache['chunk_tokens']+=features.shape[1]
    if cache['chunk_tokens']==32:flush(model,cache)


def parallel_forward(model,ids,cache=None):
    if ids.ndim!=2 or not ids.shape[1]:raise ValueError('nonempty dense [batch,time] required')
    cache=empty(model,len(ids)) if cache is None else cache
    if cache['closed'] or cache['position']+ids.shape[1]>128:raise ValueError('independent sample ended or original context exceeded')
    model.path_tensors={'attention':[],'attention_scores':[],'ffn':[]};model.observations=[]
    outputs=[];offset=0
    while offset<ids.shape[1]:
        count=min(32-cache['position']%32,ids.shape[1]-offset)
        h,features=stages(model,model.token(ids[:,offset:offset+count]),cache,cache['position'])
        outputs.append(model.head(model.norm(h)))
        append(model,cache,features);cache['position']+=count;offset+=count
    return torch.cat(outputs,1),cache


class Session(old_runtime.Session):
    def __init__(self,model,tokenizer_identity='UNSPECIFIED',batch=1):
        super().__init__(model,tokenizer_identity,batch);self.interaction_mode=self._mode()

    def _mode(self):return shared.sha({'source':SOURCE,'configuration':self.model.configuration,'old_runtime_source':old_runtime.RUNTIME_SOURCE_ID,'read_attention':self.model.read_attention,'read_ffn':self.model.read_ffn,'reset_each':self.model.reset_every_chunk,'read':self.model.read_enabled,'write':self.model.write_enabled})

    def reset(self,batch=None):
        super().reset(batch)
        self.cache.update(state_memory={},state_revision=0,closed=False)

    @torch.inference_mode()
    def step(self,ids):
        if self._mode()!=self.interaction_mode or tuple(p._version for p in self.model.parameters())!=self.versions:raise ValueError('Session configuration/weights changed')
        if self.cache['closed']:raise ValueError('sequence finished; reset before next independent sample')
        position=self.cache['position'];self.adapter._check(self.cache,ids,position)
        h,features=stages(self.model,self.model.token(ids),self.cache,position)
        logits=self.model.head(self.model.norm(h));append(self.model,self.cache,features)
        self.cache['position']+=1;self.cache['reset']=False
        return logits

    @torch.inference_mode()
    def finish(self):flush(self.model,self.cache,close=True);return self.cache['persistent']

    def snapshot(self):
        return {'format':'minicells.a2-state.session.v1','model_identity':self.adapter.weight_identity,
            'tokenizer_identity':self.tokenizer_identity,'interaction_mode':self.interaction_mode,
            'cache':copy.deepcopy(self.cache),'rng':self.generator.get_state().clone()}

    def restore(self,saved):
        if saved.get('format')!='minicells.a2-state.session.v1' or saved['model_identity']!=shared.model_hash(self.model) or saved['tokenizer_identity']!=self.tokenizer_identity or saved['interaction_mode']!=self._mode():raise ValueError('snapshot identity mismatch')
        cache=copy.deepcopy(saved['cache']);position=cache['position']
        if position>self.model.max_context or cache['max_context']!=self.model.max_context:raise ValueError('snapshot context differs')
        expected=0 if cache['closed'] else position%32
        if cache['chunk_tokens']!=expected or sum(p.shape[1] for p in cache['pending'])!=expected:raise ValueError('snapshot partial Chunk differs')
        token=torch.zeros(cache['batch'],1,device=next(self.model.parameters()).device,dtype=torch.long)
        self.adapter._check({**cache,'max_context':max(position+1,cache['max_context'])},token,position)
        self.cache=cache;self.generator.set_state(saved['rng']);return self

    @property
    def memory(self):
        result=super().memory
        tensors=[value for item in self.cache['state_memory'].values() for value in (item if isinstance(item,tuple) else (item,))]
        result['state_projection_cache_bytes']=sum(v.numel()*v.element_size() for v in tensors)
        result['runtime_tensor_storage_bytes']+=result['state_projection_cache_bytes']
        return result
