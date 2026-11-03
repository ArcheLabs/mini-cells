"""Independent sparse experts, token-local controller, distinct Encoder/Workspace KV."""
import copy
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from dependencies import shared,primitives


class Block(nn.Module):
    def __init__(self,c):
        super().__init__();d=c['dim'];self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
        self.attention=primitives.Attention(d,c['heads'],c['window']);self.ff=shared.native.SwiGLU(d,c['encoder_ffn'])
    def forward(self,x,cache,pos):
        r,kv=self.attention.chunk(self.n1(x),cache,pos);x=x+r
        return x+self.ff(self.n2(x)),kv


class Model(nn.Module):
    def __init__(self,arm,c,seed=17):
        if c.get('steps',2)!=2 or c.get('top_k',2)!=2:raise ValueError('v0.4 fixes two steps and Top-2; unsupported configuration')
        super().__init__();self.arm=arm;self.config=copy.deepcopy(c);self.mode='normal';self.fixed_pair=(0,1);self.noise=0.;self.generator=None;self.records=[];self.diagnostics=False;self.profiling=False
        d=c['dim'];self.call_counts=[0]*c['cells'];self.balance=None;self.capture_routes=False;self.route_tape=[];self.forced_routes=None;self.trace_tensors=False;self.tensor_trace={}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.token=nn.Embedding(c['vocab'],d);self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,c['vocab'],bias=False)
            if arm.startswith('T0'):self.blocks=nn.ModuleList([Block(c) for _ in range(c['layers'])])
            else:
                self.encoder=Block(c);self.ws_norm=nn.LayerNorm(d);self.kv=nn.Linear(d,2*d,bias=False)
                self.q_norm=nn.LayerNorm(d);self.query=nn.Linear(d,d,bias=False);self.read_out=nn.Linear(d,d,bias=False);self.cell_norm=nn.LayerNorm(d)
                # Workspace initialization is identical in C0/C1/C2/C3.
            self.apply(shared.native.HistoricalScaledCLM._init_weights)
        if not arm.startswith('T0'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+101)
                if arm=='C0':self.fixed=nn.ModuleList([shared.native.SwiGLU(d,2*c['ffn']) for _ in range(2)])
                else:
                    self.cells=nn.ModuleList([shared.native.SwiGLU(d,c['ffn']) for _ in range(c['cells'])]);self.router=nn.Linear(d,c['cells'])
                for name in ('fixed','cells','router'):
                    if hasattr(self,name):getattr(self,name).apply(shared.native.HistoricalScaledCLM._init_weights)
            if arm in ('C2','C3'):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed+202);self.state_router=nn.Linear(c['state_dim'],c['cells'],bias=False)
                    self.controller=nn.GRUCell(3*d+c['cells'],c['state_dim']);self.state_router.apply(shared.native.HistoricalScaledCLM._init_weights)
        if arm.startswith('C0-Pool'):
            del self.router
        if arm.startswith('C1-MatchedRouter'):
            width=c['matched_router_width']
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+303);self.router=nn.Sequential(nn.Linear(d,width),nn.SiLU(),nn.Linear(width,c['cells']))
                self.router.apply(shared.native.HistoricalScaledCLM._init_weights)
    def scope(self,name):return torch.profiler.record_function(name) if self.profiling else nullcontext()
    def reset(self,doc_id='document'):
        return {'doc_id':doc_id,'position':0,'encoder_kv':None,'workspace_kv':None,'layers':[None]*self.config['layers']}
    def dispatch(self,v,index,weights):
        flat=v.reshape(-1,v.shape[-1]);ids=index.reshape(-1,2);g=weights.reshape(-1,2);result=torch.zeros_like(flat)
        for expert,cell in enumerate(self.cells):
            rows,slots=torch.where(ids==expert)
            if rows.numel():

                with self.scope('cell_ffn'):output=cell(flat.index_select(0,rows))
                result=result.index_add(0,rows,output*g[rows,slots,None]);self.call_counts[expert]+=rows.numel()
        return result.reshape_as(v)
    def route(self,v,z,step):
        if self.arm.startswith('C0-Pool'):
            if self.arm=='C0-PoolFixed':index=torch.tensor(self.fixed_pair,device=v.device).expand(*v.shape[:-1],2)
            else:
                first=(torch.arange(v.shape[1],device=v.device)+self._position+2*step)%self.config['cells']
                index=torch.stack((first,(first+1)%self.config['cells']),-1)[None].expand(v.shape[0],-1,-1)
            weights=v.new_full(index.shape,.5);sparse=v.new_zeros(*v.shape[:-1],self.config['cells']).scatter(-1,index,weights)
            return index,weights,sparse,v.new_zeros(())
        logits=self.router(v)
        state_term=None
        if hasattr(self,'state_router') and self.mode!='state_off':
            state_term=self.state_router(z);logits=logits+state_term
        if self.training and self.noise:logits=logits+torch.randn(logits.shape,device=logits.device,generator=self.generator)*self.noise
        selected,index=logits.topk(2,-1);weights=selected.float().softmax(-1).to(v.dtype)
        if self.forced_routes is not None:
            index,weights=self.forced_routes[step]
        elif self.mode=='random':
            index=torch.rand(logits.shape,device=logits.device,generator=self.generator).topk(2,-1).indices;weights=torch.full_like(weights,.5)
        elif self.mode=='fixed':
            index=torch.tensor(self.fixed_pair,device=v.device).expand(*v.shape[:-1],2);weights=torch.full_like(weights,.5)
        elif self.mode=='shuffle':
            permutation=torch.randperm(v.numel()//v.shape[-1],device=v.device,generator=self.generator)
            index=index.reshape(-1,2)[permutation].reshape_as(index);weights=weights.reshape(-1,2)[permutation].reshape_as(weights)
        if self.trace_tensors:self.tensor_trace['steps'].append({'logits':logits.detach(),'indices':index.detach(),'weights':weights.detach(),'state':z.detach() if z is not None else None})
        if self.capture_routes:self.route_tape.append((index.detach().clone(),weights.detach().clone()))
        p=logits.float().softmax(-1);fraction=F.one_hot(index,self.config['cells']).float().mean(tuple(range(index.ndim)))
        balance=self.config['cells']*(fraction.detach()*p.mean(tuple(range(p.ndim-1)))).sum()
        sparse=torch.zeros_like(logits).scatter(-1,index,weights.to(logits.dtype))
        if self.diagnostics:
            with torch.no_grad():
                sorted_p=p.topk(2,-1).values
                self.records.append({'step':step,'logit_mean':float(logits.float().mean()),'logit_std':float(logits.float().std()),'entropy':float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),'top1_probability':float(sorted_p[...,0].mean()),'top12_gap':float((sorted_p[...,0]-sorted_p[...,1]).mean()),'selected':index.cpu().tolist(),'state_norm':float(z.float().norm(dim=-1).mean()) if z is not None else None,'state_logit_norm':float(state_term.float().norm(dim=-1).mean()) if state_term is not None else None})
        return index,weights,sparse,balance
    def forward(self,ids,cache=None):
        cache=self.reset() if cache is None else cache;c=self.config;pos=cache['position'];self._position=pos;x=self.token(ids);b,t,d=x.shape;self.records=[];self.route_tape=[];self.tensor_trace={'steps':[]}
        if self.arm.startswith('T0'):
            layers=[]
            for block,old in zip(self.blocks,cache['layers']):x,new=block(x,old,pos);layers.append(new)
            self.balance=x.new_zeros(());return self.head(self.norm(x)),{**cache,'position':pos+t,'layers':layers}
        with self.scope('workspace_encoder'):u,enc=self.encoder(x,cache['encoder_kv'],pos)
        if self.trace_tensors:self.tensor_trace['encoder_output']=u.detach()
        k,v=self.kv(self.ws_norm(u)).view(b,t,2,c['heads'],d//c['heads']).unbind(2);k,v=k.transpose(1,2),v.transpose(1,2)
        positions=torch.arange(pos,pos+t,device=x.device)[None,None,:,None];k=shared.rope_at(k,positions)
        old=cache['workspace_kv'];previous=0 if old is None else old[0].shape[2]
        if old is not None:k=torch.cat((old[0],k),2);v=torch.cat((old[1],v),2)
        qp=torch.arange(pos,pos+t,device=x.device)[:,None];kp=torch.arange(pos-previous,pos+t,device=x.device)[None,:]
        mask=(kp<=qp)&(kp>qp-c['window']);h=u;z=h.new_zeros(b,t,c['state_dim']) if hasattr(self,'controller') else None;balances=[];first=None
        for step in range(2):
            if self.arm=='C3' and step==1:r=first
            else:
                q=self.query(self.q_norm(h)).view(b,t,c['heads'],d//c['heads']).transpose(1,2);q=shared.rope_at(q,positions)
                with self.scope('workspace_attention_read'):read=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[None,None],dropout_p=0.)
                r=self.read_out(read.transpose(1,2).reshape(b,t,d));first=r if step==0 else first
            before=h;cell_input=self.cell_norm(h+r)
            if self.arm=='C0':delta=self.fixed[step](cell_input)
            else:
                with self.scope('router'):index,weights,sparse,balance=self.route(cell_input,z,step)
                balances.append(balance)
                with self.scope('sparse_dispatch'):delta=self.dispatch(cell_input,index,weights)
                if hasattr(self,'controller') and self.mode!='state_off' and step+1<c['steps']:
                    with torch.autocast(device_type=h.device.type,enabled=False):
                        source=torch.cat((before,delta,r,sparse),-1).float()
                        with self.scope('controller_update'):z=self.controller(source.reshape(-1,source.shape[-1]),z.float().reshape(-1,c['state_dim'])).reshape(b,t,-1)
            if self.trace_tensors:
                if self.arm=='C0':self.tensor_trace['steps'].append({'delta':delta.detach()})
                else:self.tensor_trace['steps'][-1]['delta']=delta.detach()
            h=h+r+delta
        self.balance=torch.stack(balances).mean() if balances else h.new_zeros(())
        return self.head(self.norm(h)),{**cache,'position':pos+t,'encoder_kv':enc,'workspace_kv':(k[:,:,-c['window']:],v[:,:,-c['window']:])}


class Session:
    def __init__(self,model,doc_id='document'):self.model=model;self.cache=model.reset(doc_id)
    @torch.no_grad()
    def prefill(self,ids):
        logits,self.cache=self.model(ids,self.cache);return logits
