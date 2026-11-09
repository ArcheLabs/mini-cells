"""Chunk-causal E6/E7 using the unchanged M2 Mamba weights and scan reference."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import uuid
import torch
from torch import nn
from torch.nn import functional as F
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'m2')]
from baselines import build, Config
from e0 import rope,save_checkpoint,load_checkpoint,write
from data import file_hash,digest,PAD,ANSWER
from vendor.selective_scan_ref import selective_scan_ref
from store import Store,Predictor


def mamba_chunk(module,hidden,previous=None):
    """Exact real-valued Mamba-1 recurrence; analytic initial-state contribution.

    Reuse official zero-state scan; add the homogeneous solution from previous
    state. No in-place inference cache, detach, first-order approximation or kernel.
    """
    b,t,_=hidden.shape;inner=module.d_inner
    conv,ssm=previous if previous is not None else (
        hidden.new_zeros(b,inner,module.d_conv),hidden.new_zeros(b,inner,module.d_state))
    raw,z=module.in_proj(hidden).transpose(1,2).chunk(2,1)
    x=F.silu(F.conv1d(torch.cat((conv[:,:,1:],raw),-1),module.conv1d.weight,module.conv1d.bias,groups=inner))
    projected=module.x_proj(x.transpose(1,2))
    dt,B,C=torch.split(projected,[module.dt_rank,module.d_state,module.d_state],-1)
    dt=F.linear(dt,module.dt_proj.weight).transpose(1,2);B=B.transpose(1,2);C=C.transpose(1,2)
    A=-module.A_log.float().exp()
    y,last=selective_scan_ref(x,dt,A,B,C,module.D.float(),z,
        delta_bias=module.dt_proj.bias.float(),delta_softplus=True,return_last_state=True)
    effective=F.softplus(dt.float()+module.dt_proj.bias.float()[None,:,None])
    decay=torch.exp(effective.cumsum(-1)[...,None]*A[None,:,None,:])
    homogeneous=decay*ssm[:,:,None,:]
    y=y+torch.einsum('bdtn,bnt->bdt',homogeneous,C.float())*F.silu(z)
    state=last+homogeneous[:,:,-1]
    return module.out_proj(y.transpose(1,2)),(torch.cat((conv,raw),-1)[:,:,-module.d_conv:].clone(),state)


def attention_read(attention,normalizer,x,positions,kv,summary,stores,window,read_top_k,offset=None,profile=False):
    start=int(positions[0]) if offset is None else offset
    b,t,d=x.shape;heads=attention.heads;kvheads=attention.kv_heads
    hd=d//heads;repeat=heads//kvheads
    q=rope(attention.q(x).view(b,t,heads,hd).transpose(1,2),positions)
    k=rope(attention.k(x).view(b,t,kvheads,hd).transpose(1,2),positions)
    v=attention.v(x).view(b,t,kvheads,hd).transpose(1,2)
    oldlen=kv[0].shape[2] if kv is not None else 0
    if kv is not None:k=torch.cat((kv[0],k),2);v=torch.cat((kv[1],v),2)
    nextkv=(k[:,:,-window:].clone(),v[:,:,-window:].clone())
    keypos=torch.arange(start-oldlen,start+t,device=x.device)
    mask=(keypos[None]<=positions[:,None]) & (keypos[None]>positions[:,None]-window)
    basek=k.repeat_interleave(repeat,1);basev=v.repeat_interleave(repeat,1)
    mask=mask[None,None].expand(b,heads,-1,-1)
    if summary is not None:
        count=summary.shape[1];sk=attention.k(normalizer(summary)).view(b,count,kvheads,hd).transpose(1,2)
        sk=rope(sk,positions.new_full((count,),start-1))
        sv=attention.v(normalizer(summary)).view(b,count,kvheads,hd).transpose(1,2)
        basek=torch.cat((basek,sk.repeat_interleave(repeat,1)),2);basev=torch.cat((basev,sv.repeat_interleave(repeat,1)),2)
        mask=torch.cat((mask,torch.ones(b,heads,t,count,dtype=torch.bool,device=x.device)),-1)
    logits=torch.einsum('bhtd,bhnd->bhtn',q,basek)/math.sqrt(hd)
    if profile and x.is_cuda:torch.cuda.synchronize(x.device)
    store_started=time.perf_counter()
    extras=[];values=[];masks=[];selected_positions=[]
    for stream in range(b):
        records=stores[stream].records if stores else []
        if not records:
            extras.append(x.new_zeros(heads,t,read_top_k,hd));values.append(x.new_zeros(heads,t,read_top_k,hd))
            masks.append(torch.zeros(t,read_top_k,dtype=torch.bool,device=x.device));selected_positions.append(torch.full((t,read_top_k),-1,dtype=torch.long,device=x.device));continue
        representations=torch.stack([r['representation'] for r in records]);p=positions.new_tensor([r['position'] for r in records])
        normalized=normalizer(representations)
        sk=rope(attention.k(normalized).view(1,len(records),kvheads,hd).transpose(1,2),p)[0].repeat_interleave(repeat,0)
        sv=attention.v(normalized).view(len(records),kvheads,hd).transpose(0,1).repeat_interleave(repeat,0)
        allowed=p[None,:]<positions[:,None]
        # Avoid double-counting only when that record is actually present in Local KV.
        duplicate=(p[None,:]>=start-oldlen)&(p[None,:]<=positions[:,None])&(p[None,:]>positions[:,None]-window)
        allowed=allowed & ~duplicate
        similarity=torch.einsum('htd,hnd->htn',q[stream],sk).mean(0)
        n=min(read_top_k,len(records));indices=similarity.masked_fill(~allowed,float('-inf')).argsort(dim=-1,descending=True,stable=True)[:,:n]
        valid=allowed.gather(1,indices)
        extras.append(F.pad(sk[:,indices,:],(0,0,0,read_top_k-n)))
        values.append(F.pad(sv[:,indices,:],(0,0,0,read_top_k-n)))
        masks.append(F.pad(valid,(0,read_top_k-n)))
        pos=p[indices].masked_fill(~valid,-1);selected_positions.append(F.pad(pos,(0,read_top_k-n),value=-1))
    if profile and x.is_cuda:torch.cuda.synchronize(x.device)
    store_seconds=time.perf_counter()-store_started if read_top_k else 0.
    if read_top_k:
        storek=torch.stack(extras);storev=torch.stack(values);smask=torch.stack(masks)[:,None].expand(-1,heads,-1,-1)
        extra_logits=torch.einsum('bhtd,bhtkd->bhtk',q,storek)/math.sqrt(hd)
        logits=torch.cat((logits,extra_logits),-1);mask=torch.cat((mask,smask),-1)
    probabilities=logits.masked_fill(~mask,float('-inf')).softmax(-1)
    localn=basek.shape[2];output=torch.einsum('bhtn,bhnd->bhtd',probabilities[...,:localn],basev)
    if read_top_k:output=output+torch.einsum('bhtk,bhtkd->bhtd',probabilities[...,localn:],storev)
    output=attention.o(output.transpose(1,2).contiguous().view(b,t,d))
    return output,nextkv,torch.stack(selected_positions),store_seconds


@dataclass
class Runtime:
    owner: int
    epoch: int
    versions: tuple
    batch: int
    position: int
    mamba: dict
    kv: dict
    predictor: object
    stores: object


class BMojo(nn.Module):
    def __init__(self,config,policy=None,summary_tokens=4,capacity=64,read_top_k=8,chunk=32,state_size=20,seed=41017,summary_enabled=True,beta=.875):
        super().__init__();self.config=Config(**config);self.policy=policy;self.capacity=capacity
        self.allowed_capacities=(capacity,2*capacity)
        self.summary_tokens=summary_tokens;self.read_top_k=read_top_k;self.chunk=chunk;self.seed=seed;self.summary_enabled=summary_enabled
        self.backbone=build('E3',config,seed,state_size,shared_config=None)
        self.token_embedding=self.backbone.token_embedding;self.lm_head=self.backbone.lm_head
        self.summary=nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+2000)
            for i,kind in enumerate(self.backbone.kinds):
                if summary_enabled and kind=='swa':self.summary[str(i)]=nn.Linear(state_size,summary_tokens*self.config.dim,bias=False)
        if policy is not None and not summary_enabled:raise ValueError('E7 requires State Summary')
        if policy is not None and read_top_k>capacity:raise ValueError('read_top_k exceeds capacity')
        self.predictor=Predictor(beta);self.epoch=0;self.last_runtime=None;self.last_trace=[];self.events=[]
        self.model_id='e7_bmojo_innovation_inspired' if policy else ('e6_bmojo_f_inspired' if summary_enabled else 'e3_m5_reference')

    def versions(self):return tuple(p._version for p in self.parameters())

    def new_runtime(self,batch,position=0,capacity=None):
        if batch<1 or position<0:raise ValueError('invalid runtime batch/position')
        if capacity is not None and capacity not in self.allowed_capacities:raise ValueError('unregistered capacity tier')
        return Runtime(id(self),self.epoch,self.versions(),batch,position,{},{},None,
            [Store(capacity or self.capacity,self.policy,self.seed+3000) for _ in range(batch)] if self.policy else None)

    def reset_runtime(self):self.epoch+=1;self.last_runtime=None

    def forward_state(self,ids,state=None,trace=True,trace_queries=None,read_positions=None,profile=False):
        if ids.ndim!=2 or ids.shape[1]<1 or (ids==PAD).any():raise ValueError('M5 requires unpadded nonempty batches; eval batch=1')
        if state is not None and torch.is_grad_enabled():raise ValueError('carried state is inference-only; training retains its own full episode graph')
        state=self.new_runtime(ids.shape[0]) if state is None else state
        if state.batch!=ids.shape[0] or state.owner!=id(self) or state.epoch!=self.epoch or state.versions!=self.versions():raise ValueError('runtime owner/epoch/weights differ')
        if trace and trace_queries is None:trace_queries=[(row==ANSWER).nonzero().flatten().tolist() for row in ids]
        outputs=[];traces=[];read_seconds=update_seconds=store_seconds=0.
        for start in range(0,ids.shape[1],self.chunk):
            tokens=ids[:,start:start+self.chunk];embedding=self.token_embedding(tokens);h=embedding
            positions=torch.arange(state.position,state.position+tokens.shape[1],device=ids.device)
            historical=state.mamba.copy();fading=None
            for i,block in enumerate(self.backbone.blocks):
                x=block.n1(h)
                if self.backbone.kinds[i]=='mamba1':
                    y,new=mamba_chunk(block.attn,x,state.mamba.get(i));state.mamba[i]=new;fading=y
                else:
                    summary=None
                    if self.summary_enabled and i-1 in historical:
                        pooled=historical[i-1][1].mean(1)
                        summary=self.summary[str(i)](pooled).view(ids.shape[0],self.summary_tokens,self.config.dim)
                    began=time.perf_counter()
                    y,kv,selected,store_time=attention_read(block.attn,block.n1,x,positions,state.kv.get(i),summary,
                        state.stores,self.config.window,self.read_top_k if self.policy else 0,offset=state.position,profile=profile)
                    read_seconds+=time.perf_counter()-began;store_seconds+=store_time;state.kv[i]=kv
                    if trace:
                        for b in range(ids.shape[0]):
                            selected_q=[q-start for q in trace_queries[b] if start<=q<start+tokens.shape[1]]
                            for q in selected_q:
                                traces.append({'layer':i,'stream':b,'query_position':state.position+q,'selected_positions':selected[b,q].cpu().tolist(),
                                    'stored_positions':[r['position'] for r in state.stores[b].records] if state.stores else []})
                h=h+y;h=h+block.ffn(block.n2(h))
            if state.stores:
                # All readers finish before this chunk is admitted. Global capacity is
                # C per stream, shared by every SWA layer, not C per layer.
                if profile and ids.is_cuda:torch.cuda.synchronize(ids.device)
                began=time.perf_counter();scores,state.predictor=self.predictor.score(fading,state.predictor)
                for b in range(ids.shape[0]):
                    for j,score in enumerate(scores[b].cpu().tolist()):
                        state.stores[b].admit(embedding[b,j],state.position+j,score)
                if profile and ids.is_cuda:torch.cuda.synchronize(ids.device)
                update_seconds+=time.perf_counter()-began
            outputs.append(self.backbone.final_norm(h));state.position+=tokens.shape[1]
        self.last_trace=traces;self.last_runtime=self.detached(state)
        if torch.is_grad_enabled() and self.training:
            self.events.append({'input_sha256':digest(ids.detach().cpu().tolist()),'position':state.position,
                'store':[s.metrics(state.position) for s in state.stores] if state.stores else [],
                'attention_read_host_seconds':read_seconds,'store_read_preparation_seconds':store_seconds,'store_update_host_seconds':update_seconds})
        self.last_timing={'attention_read_host_seconds':read_seconds,'store_read_preparation_seconds':store_seconds,'store_update_host_seconds':update_seconds,
                          'scope':'Store read is K/V projection, top-k and gather; excludes shared softmax. profile=True synchronizes GPU phase boundaries; otherwise host dispatch timing. No formal performance claim.'}
        features=torch.cat(outputs,1)
        return self.lm_head(features if read_positions is None else features[:,read_positions]),state,traces

    def forward(self,ids):return self.forward_state(ids)[0]

    def detached(self,state):
        return Runtime(state.owner,state.epoch,state.versions,state.batch,state.position,
            {k:tuple(v.detach().clone() for v in pair) for k,pair in state.mamba.items()},
            {k:tuple(v.detach().clone() for v in pair) for k,pair in state.kv.items()},
            state.predictor.detach().clone() if state.predictor is not None else None,
            [s.detached() for s in state.stores] if state.stores else None)

    def fingerprint(self):
        h=hashlib.sha256()
        h.update(digest({'config':vars(self.config),'policy':self.policy,'summary':self.summary_enabled,
            'capacity_tiers':self.allowed_capacities,'summary_tokens':self.summary_tokens,'top_k':self.read_top_k,'chunk':self.chunk,'beta':self.predictor.beta}).encode())
        for name,p in self.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def checkpoint(self,state,folder):
        folder=Path(folder);generation=uuid.uuid4().hex;target=folder/generation;target.mkdir(parents=True)
        started=time.perf_counter();state=self.detached(state)
        components={'mamba.pt':state.mamba,'kv.pt':state.kv,'predictor.pt':state.predictor,
                    'store.pt':[s.snapshot() for s in state.stores] if state.stores else None}
        for name,value in components.items():save_checkpoint(value,target/name)
        manifest={'format':'minicells.m5.runtime.v1','model_sha256':self.fingerprint(),'generation':generation,
            'position':state.position,'batch':state.batch,
            'components':{name:file_hash(target/name) for name in components},
            'summary_state':'derived from mamba.pt plus static summary projection; no duplicate hidden state'}
        write(folder/'manifest.json',manifest)
        return {'bytes':sum(p.stat().st_size for p in target.iterdir())+(folder/'manifest.json').stat().st_size,
                'write_seconds':time.perf_counter()-started,'manifest_sha256':file_hash(folder/'manifest.json')}

    def restore(self,folder,store_only=False):
        folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
        if manifest['format']!='minicells.m5.runtime.v1' or manifest['model_sha256']!=self.fingerprint():raise ValueError('runtime/model/config mismatch')
        generation=manifest['generation']
        if len(generation)!=32 or any(c not in '0123456789abcdef' for c in generation):raise ValueError('invalid generation')
        target=folder/generation
        for name,sha in manifest['components'].items():
            if name not in ('mamba.pt','kv.pt','predictor.pt','store.pt') or file_hash(target/name)!=sha:raise ValueError('component hash mismatch')
        saved={name:load_checkpoint(target/name) for name in ('mamba.pt','kv.pt','predictor.pt','store.pt')}
        device=self.token_embedding.weight.device
        state=self.new_runtime(manifest['batch'],manifest['position'])
        state.stores=[Store.restore(s,device) for s in saved['store.pt']] if saved['store.pt'] else None
        if state.stores and any(s.capacity not in self.allowed_capacities or s.last_position>=state.position for s in state.stores):raise ValueError('restored capacity/chronology mismatch')
        if not store_only:
            state.mamba={int(k):tuple(v.to(device) for v in pair) for k,pair in saved['mamba.pt'].items()}
            state.kv={int(k):tuple(v.to(device) for v in pair) for k,pair in saved['kv.pt'].items()}
            state.predictor=saved['predictor.pt'].to(device) if saved['predictor.pt'] is not None else None
        return state

    def resources(self,state):
        sizes=lambda pairs:sum(t.numel()*t.element_size() for pair in pairs.values() for t in pair)
        stores=[s.metrics(state.position) for s in state.stores] if state.stores else []
        breakdown={'embedding':self.token_embedding.weight.numel(),
            'mamba':sum(p.numel() for kind,block in zip(self.backbone.kinds,self.backbone.blocks) if kind=='mamba1' for p in block.attn.parameters()),
            'attention':sum(p.numel() for kind,block in zip(self.backbone.kinds,self.backbone.blocks) if kind=='swa' for p in block.attn.parameters()),
            'mlp':sum(p.numel() for block in self.backbone.blocks for p in block.ffn.parameters()),
            'norm':sum(p.numel() for block in self.backbone.blocks for module in (block.n1,block.n2) for p in module.parameters())+sum(p.numel() for p in self.backbone.final_norm.parameters()),
            'summary_projection':sum(p.numel() for p in self.summary.parameters())}
        assert sum(breakdown.values())==sum(p.numel() for p in self.parameters())
        return {'static_parameter_breakdown':breakdown,'static_parameters':sum(p.numel() for p in self.parameters()),'static_weight_bytes':sum(p.numel()*p.element_size() for p in self.parameters()),
            'mamba_runtime_bytes':sizes(state.mamba),'kv_bytes':sizes(state.kv),
            'predictor_bytes':state.predictor.numel()*state.predictor.element_size() if state.predictor is not None else 0,
            'summary_materialized_bytes_per_chunk':len(self.summary)*self.summary_tokens*self.config.dim*self.token_embedding.weight.element_size()*state.batch,
            'persistent_summary_bytes':0,'store_payload_bytes':sum(s['payload_bytes'] for s in stores),
            'store_metadata_serialized_bytes':sum(s['metadata_serialized_bytes'] for s in stores),'stores':stores}
