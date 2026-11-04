"""Native-compatible weights; explicit causal sliding attention, compact GQA cache."""
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F
from native_clm_runtime import RMSNorm, GQAAttention, SwiGLU, HistoricalScaledCLM


@dataclass(frozen=True)
class Config:
    vocab: int = 2048
    dim: int = 192
    layers: int = 4
    heads: int = 6
    kv_heads: int = 2
    ffn: int = 480
    window: int = 128

    def __post_init__(self):
        if min(asdict(self).values())<1 or self.dim%self.heads or self.heads%self.kv_heads or (self.dim//self.heads)%2:
            raise ValueError('invalid model/window/GQA dimensions')


@dataclass
class Cache:
    owner: int
    next_position: int
    layers: tuple


def rope(x,positions):
    half=x.shape[-1]//2
    inv=1./(10000**(torch.arange(half,device=x.device,dtype=torch.float32)/half))
    angle=positions.float()[:,None]*inv[None,:]
    sine,cosine=angle.sin().to(x.dtype)[None,None],angle.cos().to(x.dtype)[None,None]
    a,b=x[...,:half],x[...,half:]
    return torch.cat((a*cosine-b*sine,a*sine+b*cosine),-1)


class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.n1=RMSNorm(c.dim);self.attn=GQAAttention(c.dim,c.heads,c.kv_heads)
        self.n2=RMSNorm(c.dim);self.ffn=SwiGLU(c.dim,c.ffn)


class CachedTransformer(nn.Module):
    """Uncached forward is the reference for this sliding mask, not GC001 replay."""
    def __init__(self,config=Config()):
        super().__init__();self.config=config
        self.token_embedding=nn.Embedding(config.vocab,config.dim)
        self.blocks=nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.final_norm=RMSNorm(config.dim);self.lm_head=nn.Linear(config.dim,config.vocab,bias=False)
        self.lm_head.weight=self.token_embedding.weight
        self.apply(HistoricalScaledCLM._init_weights)

    def _forward(self,ids,cache=None,save_cache=False):
        c=self.config
        if ids.ndim!=2 or ids.shape[1]<1:raise ValueError('nonempty [batch,time] IDs required')
        if cache is not None and (cache.owner!=id(self) or len(cache.layers)!=c.layers):raise ValueError('cache belongs to another model')
        start=cache.next_position if cache else 0
        positions=torch.arange(start,start+ids.shape[1],device=ids.device)
        h=self.token_embedding(ids);b,t,_=h.shape;entries=[]
        for index,block in enumerate(self.blocks):
            x=block.n1(h);a=block.attn;d=c.dim//c.heads
            q=rope(a.q(x).view(b,t,c.heads,d).transpose(1,2),positions)
            k=rope(a.k(x).view(b,t,c.kv_heads,d).transpose(1,2),positions)
            v=a.v(x).view(b,t,c.kv_heads,d).transpose(1,2)
            if cache:
                oldk,oldv=cache.layers[index]
                if oldk.shape[:2]!=(b,c.kv_heads) or oldk.device!=k.device or oldk.dtype!=k.dtype:raise ValueError('cache batch/device/dtype differs')
                k=torch.cat((oldk,k),2);v=torch.cat((oldv,v),2)
            key_positions=torch.arange(start+t-k.shape[2],start+t,device=ids.device)
            allowed=(key_positions[None,:]<=positions[:,None]) & (key_positions[None,:]>positions[:,None]-c.window)
            repeat=c.heads//c.kv_heads
            # Expanded K/V are temporary attention workspace, never stored in Cache.
            y=F.scaled_dot_product_attention(q,k.repeat_interleave(repeat,1),v.repeat_interleave(repeat,1),attn_mask=allowed[None,None],dropout_p=0.,is_causal=False)
            h=h+a.o(y.transpose(1,2).contiguous().view(b,t,c.dim))
            h=h+block.ffn(block.n2(h))
            if save_cache:
                # clone prevents a short view retaining an unbounded old backing storage.
                entries.append((k[:,:,-c.window:].detach().clone(),v[:,:,-c.window:].detach().clone()))
        logits=self.lm_head(self.final_norm(h))
        return logits,Cache(id(self),start+t,tuple(entries)) if save_cache else None

    def forward(self,ids):return self._forward(ids)[0]

    @torch.inference_mode()
    def cached(self,ids,cache=None):
        if self.training:raise ValueError('KV cache is inference-only; call eval()')
        return self._forward(ids,cache,True)

    @torch.inference_mode()
    def generate(self,prompt,max_new_tokens,eos_id):
        if prompt.shape[0]!=1 or max_new_tokens<0:raise ValueError('batch-one generation required')
        logits,cache=self.cached(prompt);output=[]
        for _ in range(max_new_tokens):
            token=logits[:,-1].argmax(-1);output.append(int(token.item()))
            if output[-1]==eos_id:break
            logits,cache=self.cached(token[:,None],cache)
        return output,cache

    def weight_identity(self):
        digest=hashlib.sha256(json.dumps(asdict(self.config),sort_keys=True).encode())
        for name,value in self.state_dict().items():
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def snapshot(self,cache):
        if cache.owner!=id(self):raise ValueError('cache owner differs')
        return {'format':'gc002.kv.v1','model_sha256':self.weight_identity(),'next_position':cache.next_position,
                'layers':tuple(tuple(v.cpu().clone() for v in pair) for pair in cache.layers)}

    def restore(self,snapshot):
        if snapshot['format']!='gc002.kv.v1' or snapshot['model_sha256']!=self.weight_identity():raise ValueError('snapshot model/config mismatch')
        c=self.config;position=snapshot['next_position'];layers=snapshot['layers'];reference=self.token_embedding.weight
        if not isinstance(position,int) or position<1 or len(layers)!=c.layers:raise ValueError('invalid snapshot metadata')
        expected=(c.kv_heads,min(position,c.window),c.dim//c.heads)
        batch=None
        for pair in layers:
            if len(pair)!=2:raise ValueError('invalid KV pair')
            for value in pair:
                if value.ndim!=4 or value.shape[1:]!=expected or not value.is_floating_point() or not torch.isfinite(value).all():raise ValueError('invalid snapshot tensor')
                if batch is not None and value.shape[0]!=batch:raise ValueError('snapshot batch mismatch')
                batch=value.shape[0]
        return Cache(id(self),position,tuple(tuple(v.to(reference).clone() for v in pair) for pair in layers))


def cache_memory(model,cache):
    tensors=[v for pair in cache.layers for v in pair];storages={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in tensors}
    stream=tensors[0].shape[0]
    snapshot=model.snapshot(cache);buffer=io.BytesIO();torch.save(snapshot,buffer)
    metadata=sys.getsizeof(cache)+sys.getsizeof(cache.__dict__)+sys.getsizeof(cache.layers)+sum(sys.getsizeof(pair) for pair in cache.layers)+sys.getsizeof(cache.owner)+sys.getsizeof(cache.next_position)
    return {'batch':stream,'cached_tokens':tensors[0].shape[2],'absolute_next_position':cache.next_position,
            'kv_tensor_payload_bytes':sum(v.numel()*v.element_size() for v in tensors),'kv_tensor_backing_storage_bytes':sum(storages.values()),
            'persistent_state_bytes_per_stream':sum(storages.values())//stream,'token_buffer_bytes':0,'token_buffer_policy':'no token history retained by model cache; caller input/output excluded',
            'python_metadata_shallow_bytes':metadata,'tensor_wrapper_shallow_bytes':sum(sys.getsizeof(v) for v in tensors),
            'serialization_bytes':buffer.tell(),'theoretical_full_window_gqa_bytes_per_stream':2*model.config.layers*model.config.window*model.config.kv_heads*(model.config.dim//model.config.heads)*model.token_embedding.weight.element_size(),
            'temporary_activations':'NOT_INCLUDED_IN_PERSISTENT_STATE; GPU profile required','framework_allocator_overhead':'GPU_PROFILE_REQUIRED'}
