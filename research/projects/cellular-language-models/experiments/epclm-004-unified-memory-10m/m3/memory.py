"""Functional Titans-style chunk memory: complete higher-order reference path."""
from dataclasses import dataclass, replace
from pathlib import Path
import copy,json,sys,time,uuid
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e0 import save_checkpoint,load_checkpoint,write
from data import file_hash

MODES=('meta_train','online_adapt','read_only')

@dataclass(frozen=True)
class Cell:
    fast: dict
    momentum: dict
    update_step: int
    metadata: dict

    def detached(self):
        return Cell({k:v.detach().clone() for k,v in self.fast.items()},
                    {k:v.detach().clone() for k,v in self.momentum.items()},self.update_step,copy.deepcopy(self.metadata))


def memory_forward(keys,weights):
    hidden=F.gelu(torch.einsum('btd,bhd->bth',keys,weights['w1'])+weights['b1'][:,None])
    output=keys+torch.einsum('bth,bdh->btd',hidden,weights['w2'])+weights['b2'][:,None]
    output=F.layer_norm(output,(keys.shape[-1],))
    return output*weights['ln_weight'][:,None]+weights['ln_bias'][:,None]


def transition(fast,momentum,gradients,eta,theta,alpha):
    """S'=eta*S-theta*dL/dM; M'=(1-alpha)*M+S'. No in-place writes."""
    next_momentum={};next_fast={}
    for (name,p),g in zip(fast.items(),gradients):
        shape=(-1,)+(1,)*(p.ndim-1)
        s=eta.reshape(shape)*momentum[name]-theta.reshape(shape)*g
        next_momentum[name]=s;next_fast[name]=(1-alpha.reshape(shape))*p+s
    return next_fast,next_momentum


class NeuralMemory(nn.Module):
    def __init__(self,input_dim,memory_dim=32,hidden=64,max_lr=.05,max_forget=.01):
        super().__init__();self.input_dim=input_dim;self.dim=memory_dim;self.hidden=hidden
        self.max_lr=max_lr;self.max_forget=max_forget
        self.key=nn.Linear(input_dim,memory_dim,bias=False)
        self.value=nn.Linear(input_dim,memory_dim,bias=False)
        self.query=nn.Linear(input_dim,memory_dim,bias=False)
        self.controls=nn.Linear(memory_dim,3)
        self.initial=nn.ParameterDict({'w1':nn.Parameter(torch.randn(hidden,memory_dim)*.02),
            'b1':nn.Parameter(torch.zeros(hidden)),'w2':nn.Parameter(torch.randn(memory_dim,hidden)*.02),
            'b2':nn.Parameter(torch.zeros(memory_dim)),'ln_weight':nn.Parameter(torch.ones(memory_dim)),
            'ln_bias':nn.Parameter(torch.zeros(memory_dim))})
        nn.init.normal_(self.controls.weight,std=.001)
        with torch.no_grad():self.controls.bias.copy_(torch.tensor([0.,0.,-6.]))

    def create(self,batch=1,mode='meta_train'):
        if mode not in MODES:raise ValueError('unknown memory mode')
        with torch.enable_grad():
            fast={k:p.unsqueeze(0).expand(batch,*p.shape).clone() for k,p in self.initial.items()}
            if mode!='meta_train':fast={k:v.detach().requires_grad_(True) for k,v in fast.items()}
        return Cell(fast,{k:torch.zeros_like(v) for k,v in fast.items()},0,
                    {'cell_id':uuid.uuid4().hex,'revision':0,'parent':None,'architecture_id':'mag_lite_single_memory',
                     'interface_version':1,'memory_dim':self.dim,'hidden':self.hidden,'provenance':[]})

    def project(self,hidden):
        return (F.normalize(F.silu(self.key(hidden)),dim=-1),F.silu(self.value(hidden)),
                F.normalize(F.silu(self.query(hidden)),dim=-1))

    def recall(self,queries,cell,mode='read_only'):
        if mode not in MODES:raise ValueError('unknown memory mode')
        if queries.ndim!=3 or queries.shape[0]!=next(iter(cell.fast.values())).shape[0] or queries.shape[-1]!=self.dim:
            raise ValueError('query/Cell batch or feature mismatch')
        if mode=='read_only':
            with torch.no_grad():return memory_forward(F.normalize(queries,dim=-1),cell.fast)
        return memory_forward(F.normalize(queries,dim=-1),cell.fast)

    def learn(self,keys,values,cell,mode='meta_train'):
        if mode not in MODES:raise ValueError('unknown memory mode')
        if mode=='read_only':return cell,{}
        if keys.shape!=values.shape or keys.ndim!=3 or keys.shape[0]!=next(iter(cell.fast.values())).shape[0] or keys.shape[-1]!=self.dim or keys.shape[1]==0:
            raise ValueError('write/Cell batch or feature mismatch')
        # grad mode is independent of model.train/eval and external no_grad.
        with torch.enable_grad():
            if mode=='online_adapt':
                keys=keys.detach();values=values.detach()
                fast={k:v.detach().requires_grad_(True) for k,v in cell.fast.items()}
                momentum={k:v.detach() for k,v in cell.momentum.items()}
            else:fast,momentum=cell.fast,cell.momentum
            keys=F.normalize(keys,dim=-1)
            error=memory_forward(keys,fast)-values
            # Mean across chunk tokens, SUM across streams. Each stream's fast
            # weights get its own gradient, never divided by other stream count.
            loss=error.square().sum(-1).mean(1).sum()
            gradients=torch.autograd.grad(loss,tuple(fast.values()),create_graph=mode=='meta_train')
            controls=self.controls(keys.mean(1))
            eta,theta,alpha=controls.sigmoid().unbind(-1)
            theta=theta*self.max_lr;alpha=alpha*self.max_forget
            if mode=='online_adapt':eta,theta,alpha=eta.detach(),theta.detach(),alpha.detach()
            new_fast,new_momentum=transition(fast,momentum,gradients,eta,theta,alpha)
            if mode=='online_adapt':
                new_fast={k:v.detach().requires_grad_(True) for k,v in new_fast.items()}
                new_momentum={k:v.detach() for k,v in new_momentum.items()}
        result=Cell(new_fast,new_momentum,cell.update_step+1,{**cell.metadata,'revision':cell.metadata['revision']+1})
        return result,{'inner_loss':loss.detach(),'eta':eta.detach(),'theta':theta.detach(),'alpha':alpha.detach()}

    def stream(self,hidden,cell=None,mode='meta_train',chunk=16):
        if chunk<1 or hidden.ndim!=3 or hidden.shape[1]==0 or hidden.shape[-1]!=self.input_dim:
            raise ValueError('positive chunk and nonempty [batch,time,input_dim] required')
        cell=self.create(hidden.shape[0],mode) if cell is None else cell
        reads=[];diagnostics=[]
        for part in hidden.split(chunk,dim=1):
            k,v,q=self.project(part)
            # All reads precede every write of this chunk; no future/current
            # chunk write can enter any output in this chunk.
            reads.append(self.recall(q,cell,mode))
            cell,diag=self.learn(k,v,cell,mode)
            if diag:diagnostics.append(diag)
        if not all(v.detach().isfinite().all() for v in list(cell.fast.values())+list(cell.momentum.values())):
            raise FloatingPointError('nonfinite dynamic Cell state')
        return torch.cat(reads,1),cell,diagnostics

    def fork(self,cell):
        state=cell.detached()
        state=replace(state,fast={k:v.requires_grad_(True) for k,v in state.fast.items()})
        return replace(state,metadata={**state.metadata,'cell_id':uuid.uuid4().hex,'revision':0,
                      'parent':{'cell_id':cell.metadata['cell_id'],'revision':cell.metadata['revision']}})

    def checkpoint(self,cell,folder,compatibility):
        """Independent components selected by one last-written atomic manifest."""
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);generation=uuid.uuid4().hex
        target=folder/generation;target.mkdir()
        state=cell.detached();started=time.perf_counter()
        save_checkpoint({k:v.cpu() for k,v in state.fast.items()},target/'fast_weights.pt')
        save_checkpoint({k:v.cpu() for k,v in state.momentum.items()},target/'momentum.pt')
        manifest={'format':'minicells.epclm004.m3-cell.v1','generation':generation,'metadata':state.metadata,
                  'update_step':state.update_step,'compatibility':compatibility,
                  'files':{n:file_hash(target/n) for n in ('fast_weights.pt','momentum.pt')}}
        write(folder/'manifest.json',manifest)
        return {'bytes':sum(p.stat().st_size for p in target.iterdir())+(folder/'manifest.json').stat().st_size,
                'write_seconds':time.perf_counter()-started,'manifest_sha256':file_hash(folder/'manifest.json')}

    def restore(self,folder,compatibility):
        folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
        if m['format']!='minicells.epclm004.m3-cell.v1' or m['compatibility']!=compatibility:
            raise ValueError('Cell format/backbone/tokenizer/interface mismatch')
        if len(m['generation'])!=32 or any(c not in '0123456789abcdef' for c in m['generation']):
            raise ValueError('invalid Cell generation')
        target=folder/m['generation']
        for name in ('fast_weights.pt','momentum.pt'):
            if file_hash(target/name)!=m['files'][name]:raise ValueError('Cell component hash mismatch')
        fast=load_checkpoint(target/'fast_weights.pt');momentum=load_checkpoint(target/'momentum.pt')
        p=next(self.parameters());batch=None
        if set(fast)!=set(self.initial) or set(momentum)!=set(self.initial):raise ValueError('Cell parameter names differ')
        for name,initial in self.initial.items():
            for component in (fast,momentum):
                v=component[name]
                if v.shape[1:]!=initial.shape or not v.is_floating_point() or not v.isfinite().all():
                    raise ValueError('Cell tensor shape/numerics differ')
                if batch is None:batch=v.shape[0]
                if v.shape[0]!=batch:raise ValueError('Cell stream count differs')
                component[name]=v.to(p).detach().clone()
                if component is fast:component[name].requires_grad_(True)
        return Cell(fast,momentum,m['update_step'],m['metadata'])


def state_bytes(cell):
    return {'fast_weights':sum(v.numel()*v.element_size() for v in cell.fast.values()),
            'momentum':sum(v.numel()*v.element_size() for v in cell.momentum.values()),
            'fast_parameters_per_stream':sum(v[0].numel() for v in cell.fast.values())}
