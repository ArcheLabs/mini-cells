"""Reuse EPCLM-001 attention/cache primitives; add fixed-boundary state lifecycle."""
import copy
import torch
from torch.nn import functional as F
from common import IncrementalLM,rope_at,N1_ALPHA,model_hash,sha,digest,BASE,ROOT,NATIVE,INCREMENTAL
from cell import SelectiveCellFFN

RUNTIME_SOURCE_ID=sha({str(p.relative_to(ROOT)):digest(p) for p in [BASE/name for name in ('architecture.py','runtime.py','state.py','cell.py','routing.py')]+[NATIVE/'native_clm_runtime.py',NATIVE/'native_clm_phase_b.py',INCREMENTAL/'epclm001/incremental.py']})


def empty(model,batch):
    return {'position':0,'layers':{},'persistent':model.persistent.init(batch) if model.persistent else None,'pending':[],
            'chunk_tokens':0,'batch':batch}


def attend_chunk(attention,x,layers,key,start):
    b,t,d=x.shape;heads=attention.heads;head_dim=attention.head_dim
    q,k,v=attention.qkv(x).view(b,t,3,heads,head_dim).unbind(2)
    q,k,v=(a.transpose(1,2) for a in (q,k,v));positions=torch.arange(start,start+t,device=x.device)
    q=rope_at(q,positions[None,None,:,None]);k=rope_at(k,positions[None,None,:,None])
    prior=layers.get(key);old=0 if prior is None else prior['k'].shape[2]
    if prior is not None:k=torch.cat((prior['k'],k),2);v=torch.cat((prior['v'],v),2)
    keys=torch.arange(start-old,start+t,device=x.device)
    mask=(keys[None,:]<=positions[:,None]) & (keys[None,:]>=positions[:,None]-attention.window+1)
    delta=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[None,None],dropout_p=0.,is_causal=False)
    layers[key]={'k':k[:,:,-attention.window:],'v':v[:,:,-attention.window:],'tag':key,'frontier':start+t}
    return attention.out(delta.transpose(1,2).contiguous().view(b,t,d))


def stage_forward(model,x,session,start,adapter=None):
    for i,stage in enumerate(model.stages):
        if i==1 and model.persistent and model.read_enabled:x=model.persistent.read(x,session['persistent'])
        for j in range(stage.iterations):
            key=f'stage:{i}/iteration:{j}';conditioned=stage.n1(x+stage.step[j][None,None])
            if model.attention_masks[i][j]:
                delta=adapter._attend(stage.attn,conditioned,session,key,start,True) if adapter else attend_chunk(stage.attn,conditioned,session['layers'],key,start)
            else:delta=torch.zeros_like(x)
            if model.ffn_masks[i][j]:
                inputs=stage.n2(x+delta)
                if isinstance(stage.ff,SelectiveCellFFN):
                    slots=model.persistent.slots(session['persistent']) if model.persistent and model.read_enabled else None
                    ff=stage.ff(inputs,slots)
                else:ff=stage.ff(inputs)
            else:ff=torch.zeros_like(x)
            x=x+N1_ALPHA*(delta+ff)
    return x


def commit(model,session,hidden):
    if not model.persistent:
        session['pending']=[];session['chunk_tokens']=0
        return
    session['pending'].append(hidden);session['chunk_tokens']+=hidden.shape[1]
    if session['chunk_tokens']==model.chunk_size:
        if model.persistent and model.write_enabled:session['persistent']=model.persistent.write(session['persistent'],torch.cat(session['pending'],1))
        session['pending']=[];session['chunk_tokens']=0


def parallel_forward(model,ids,session=None):
    if ids.ndim!=2 or ids.shape[1]<1:raise ValueError('nonempty dense [batch,time] required')
    session=empty(model,len(ids)) if session is None else session
    if session['position']+ids.shape[1]>model.max_context:raise ValueError('extended Session token limit exceeded')
    logits=[];offset=0
    while offset<ids.shape[1]:
        length=min(model.chunk_size-session['position']%model.chunk_size,ids.shape[1]-offset) if model.persistent or model.fixed_training_chunks else ids.shape[1]-offset
        x=model.token(ids[:,offset:offset+length]);x=stage_forward(model,x,session,session['position'])
        logits.append(model.head(model.norm(x)));commit(model,session,x)
        # Stateless forward may span multiple chunks; there is no persistent update buffer.
        if not model.persistent:session['pending']=[];session['chunk_tokens']=0
        session['position']+=length;offset+=length
    return torch.cat(logits,1),session


class StreamingAdapter(IncrementalLM):
    def _layout(self):
        return {f'stage:{i}/iteration:{j}':(s.attn.heads,s.attn.head_dim,s.window)
                for i,s in enumerate(self.model.stages) for j in range(s.iterations) if self.model.attention_masks[i][j]}


class Session:
    def __init__(self,model,tokenizer_identity='UNSPECIFIED',batch=1):
        if model.training:raise ValueError('inference Session requires eval mode')
        self.model=model;self.tokenizer_identity=tokenizer_identity;self.adapter=StreamingAdapter(model)
        self.versions=tuple(p._version for p in model.parameters());self.mode=sha({'config':model.configuration,'read':model.read_enabled,'write':model.write_enabled,'runtime':RUNTIME_SOURCE_ID});self.reset(batch)

    def reset(self,batch=None):
        batch=batch or self.cache['batch'];p=next(self.model.parameters())
        self.cache=self.adapter.init_cache(batch,p.device,p.dtype,max_context=self.model.max_context)
        self.cache.update(persistent=self.model.persistent.init(batch) if self.model.persistent else None,pending=[],chunk_tokens=0)
        self.generator=torch.Generator(device=p.device).manual_seed(713)

    @torch.inference_mode()
    def step(self,ids):
        if sha({'config':self.model.configuration,'read':self.model.read_enabled,'write':self.model.write_enabled,'runtime':RUNTIME_SOURCE_ID})!=self.mode:raise ValueError('runtime configuration changed; Session invalidated')
        if tuple(p._version for p in self.model.parameters())!=self.versions:raise ValueError('model weights changed; Session invalidated')
        if ids.ndim!=2 or ids.shape[1]!=1:raise ValueError('one Token per batch lane required')
        position=self.cache['position'];self.adapter._check(self.cache,ids,position)
        x=stage_forward(self.model,self.model.token(ids),self.cache,position,self.adapter)
        logits=self.model.head(self.model.norm(x));commit(self.model,self.cache,x)
        self.cache['position']+=1;self.cache['reset']=False
        return logits

    def prefill(self,ids):
        if ids.ndim!=2 or not ids.shape[1]:raise ValueError('nonempty dense IDs required')
        if self.cache['position']+ids.shape[1]>self.model.max_context:raise ValueError('Session token limit exceeded')
        return torch.cat([self.step(ids[:,i:i+1]) for i in range(ids.shape[1])],1)

    def prefill_document(self,ids):
        self.reset(len(ids))
        return self.prefill(ids)

    def snapshot(self):
        return {'format':'minicells.x4c.session.v1','runtime_source_sha256':RUNTIME_SOURCE_ID,'model_identity':sha({'weights':self.adapter.weight_identity,'mode':self.mode}),
                'tokenizer_identity':self.tokenizer_identity,'cache':copy.deepcopy(self.cache),'rng':self.generator.get_state().clone()}

    def restore(self,snapshot):
        if snapshot.get('format')!='minicells.x4c.session.v1' or snapshot.get('model_identity')!=sha({'weights':model_hash(self.model),'mode':self.mode}) or snapshot.get('tokenizer_identity')!=self.tokenizer_identity:
            raise ValueError('snapshot model/tokenizer identity mismatch')
        saved=copy.deepcopy(snapshot['cache']);position=saved['position']
        if saved['chunk_tokens']!=position%self.model.chunk_size and self.model.persistent:raise ValueError('snapshot absolute Chunk boundary differs')
        if sum(h.shape[1] for h in saved['pending'])!=saved['chunk_tokens']:raise ValueError('snapshot pending Chunk buffer differs')
        probe=torch.zeros(saved['batch'],1,device=next(self.model.parameters()).device,dtype=torch.long)
        if position>self.model.max_context or saved['max_context']!=self.model.max_context:raise ValueError('snapshot Session limit differs')
        self.adapter._check({**saved,'max_context':max(saved['max_context'],position+1)},probe,position)
        self.cache=saved;self.generator.set_state(snapshot['rng'])
        return self

    @property
    def memory(self):
        cache=self.adapter.cache_bytes(self.cache);state=self.cache['persistent']
        kv=[v for entry in self.cache['layers'].values() for k,v in entry.items() if k in ('k','v')]
        def storage_bytes(values):
            stores={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in values}
            return sum(stores.values())
        all_values=kv+list(self.cache['pending'])+([] if state is None else [state])
        return {'kv_cache_bytes':cache,'kv_cache_scope':'logical element bytes; storage may retain larger views',
                'kv_tensor_storage_bytes':storage_bytes(kv),'runtime_tensor_storage_bytes':storage_bytes(all_values),
                'persistent_state_bytes':0 if state is None else state.numel()*state.element_size(),
                'pending_chunk_bytes':sum(h.numel()*h.element_size() for h in self.cache['pending'])}
