"""Standard MHA baseline and sparse recurrent Workspace; no per-Cell FFNs."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from dependencies import shared


def detach(value):
    if torch.is_tensor(value):return value.detach()
    if isinstance(value,dict):return {k:detach(v) for k,v in value.items()}
    if isinstance(value,list):return [detach(v) for v in value]
    if isinstance(value,tuple):return tuple(detach(v) for v in value)
    return value


class Attention(nn.Module):
    def __init__(self,d,heads,window):
        super().__init__();self.heads=heads;self.width=d//heads;self.window=window
        self.qkv=nn.Linear(d,3*d,bias=False);self.out=nn.Linear(d,d,bias=False)
    def project(self,x,position):
        q,k,v=self.qkv(x).view(x.shape[0],x.shape[1],3,self.heads,self.width).unbind(2)
        q,k,v=[a.transpose(1,2) for a in (q,k,v)]
        pos=torch.arange(position,position+x.shape[1],device=x.device)[None,None,:,None]
        return shared.rope_at(q,pos),shared.rope_at(k,pos),v
    def read(self,q,k,v):
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        return self.out(y.transpose(1,2).reshape(q.shape[0],q.shape[2],-1))
    def chunk(self,x,previous,position):
        q,k,v=self.project(x,position);old=0 if previous is None else previous[0].shape[2]
        if previous is not None:k=torch.cat((previous[0],k),2);v=torch.cat((previous[1],v),2)
        qi=torch.arange(position,position+x.shape[1],device=x.device)[:,None]
        ki=torch.arange(position-old,position+x.shape[1],device=x.device)[None,:]
        mask=(ki<=qi)&(ki>qi-self.window)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[None,None],dropout_p=0.)
        return self.out(y.transpose(1,2).reshape_as(x)),(k[:,:,-self.window:],v[:,:,-self.window:])


class Model(nn.Module):
    def __init__(self,arm,config,seed=91001):
        super().__init__();self.arm=arm;self.config=config;d=config['dim'];self.mode='normal'
        self.sample=False;self.calls=0;self.writes=0;self.trace=[];self.diagnostics=False
        # Common components consume precisely the same RNG sequence in M0/M1/M2.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.token=nn.Embedding(config['vocab'],d);self.norm=shared.native.RMSNorm(d)
            self.head=nn.Linear(d,config['vocab'],bias=False)
            if arm=='T0':
                self.blocks=nn.ModuleList([nn.ModuleList([shared.native.RMSNorm(d),Attention(d,config['heads'],config['window']),shared.native.RMSNorm(d),shared.native.SwiGLU(d,config['ffn'])]) for _ in range(config['layers'])])
            else:
                self.workspace=Attention(d,config['heads'],config['window']);self.work_norm=shared.native.RMSNorm(d)
                self.cell_norm=shared.native.RMSNorm(d);self.identity=nn.Embedding(config['cells'],d)
                self.query=nn.Linear(d,config['router_dim'],bias=False);self.address=nn.Linear(d,config['router_dim'],bias=False)
                self.core=shared.native.SwiGLU(d,config['ffn']);self.output=nn.Linear(d,d,bias=False)
            self.apply(shared.native.HistoricalScaledCLM._init_weights)
        if arm in ('M1','M2'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9100101)
                self.state_read=nn.Linear(config['state_dim'],d,bias=False)
                self.update=nn.GRUCell(2*d,config['state_dim'])
                self.state_address=nn.Linear(config['state_dim'],config['router_dim'],bias=False)
                self.state_read.apply(shared.native.HistoricalScaledCLM._init_weights)
                self.state_address.apply(shared.native.HistoricalScaledCLM._init_weights)
                # M1 has no State->Router parameters; M2 is otherwise identical.
                if arm=='M1':del self.state_address
    def reset(self,doc_id):
        device=self.token.weight.device;c=self.config
        return {'doc_id':doc_id,'position':0,'kv':[None]*c['layers'] if self.arm=='T0' else None,
                'state':torch.zeros(1,c['cells'],c['state_dim'],device=device) if self.arm in ('M1','M2') else None}
    def forward(self,ids,cache,generator=None):
        if ids.shape[0]!=1:raise ValueError('document-isolated scan expects batch1; groups accumulate documents')
        if self.mode=='reset_chunk' and cache['state'] is not None:cache={**cache,'state':torch.zeros_like(cache['state'])}
        x=self.token(ids);pos=cache['position'];logs=[]
        if self.arm=='T0':
            kv=[]
            for (n1,a,n2,ff),prior in zip(self.blocks,cache['kv']):
                y,next_kv=a.chunk(n1(x),prior,pos);x=x+y;x=x+ff(n2(x));kv.append(next_kv)
            return self.head(self.norm(x)),{**cache,'position':pos+ids.shape[1],'kv':kv},logs
        outputs=[];state=cache['state'];kv=cache['kv']
        for offset in range(ids.shape[1]):
            current=x[:,offset:offset+1];q,_,_=self.workspace.project(self.work_norm(current),pos+offset)
            h=current if kv is None else current+self.workspace.read(q,*kv)
            selected=[]
            for step in range(2):
                before=h[:,0];scores=self.query(before)@self.address(self.identity.weight).T
                state_score=None
                if self.arm=='M2' and self.mode not in ('router_off','all_off'):
                    state_score=(self.query(before)[:,None,:]*self.state_address(state)).sum(-1)
                    scores=scores+state_score
                probabilities=(scores/math.sqrt(self.config['router_dim'])).softmax(-1)
                index=torch.multinomial(probabilities,1,generator=generator) if self.sample else probabilities.argmax(-1,keepdim=True)
                logs.append(probabilities.gather(1,index).clamp_min(1e-30).log().squeeze())
                identity=self.identity(index[:,0]);old=None if state is None else state[0,index[0,0]][None]
                cell_input=self.cell_norm(before)+identity
                if old is not None and self.mode not in ('cell_off','all_off'):
                    read=state[0,(index[0,0]+1)%self.config['cells']][None] if self.mode=='shuffle' else old
                    cell_input=cell_input+self.state_read(read)
                u=self.core(cell_input);h=(before+self.output(u))[:,None];self.calls+=1
                magnitude=None
                if state is not None and self.mode!='write_off':
                    recurrent=torch.zeros_like(old) if self.mode in ('recurrence_off','all_off') else old
                    new=self.update(torch.cat((before,u),-1).float(),recurrent.float())
                    mask=F.one_hot(index[:,0],self.config['cells']).bool()[:,:,None]
                    state=torch.where(mask,new[:,None],state)
                    if self.diagnostics:magnitude=float((new-old).detach().norm())
                if self.diagnostics:selected.append(int(index[0,0]))
                if self.diagnostics:self.trace.append({'position':pos+offset,'step':step,'cell':selected[-1],'entropy':float(-(probabilities.detach()*probabilities.detach().clamp_min(1e-30).log()).sum()),'same_as_first':step==1 and selected[0]==selected[-1],'update_magnitude':magnitude,'router_state_contribution_norm':None if state_score is None else float(state_score.detach().norm()/math.sqrt(self.config['router_dim']))})
            _,k,v=self.workspace.project(self.work_norm(h),pos+offset)
            kv=(k,v) if kv is None else (torch.cat((kv[0],k),2)[:,:,-self.config['window']:],torch.cat((kv[1],v),2)[:,:,-self.config['window']:])
            self.writes+=1;outputs.append(self.head(self.norm(h)))
        return torch.cat(outputs,1),{**cache,'position':pos+ids.shape[1],'kv':kv,'state':state},logs
