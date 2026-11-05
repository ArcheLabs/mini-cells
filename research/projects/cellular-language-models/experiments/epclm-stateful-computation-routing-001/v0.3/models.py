"""Input-KV parallel fast path; logical chunk commits, shared scatter writer."""
import copy
import importlib.util
import torch
from torch import nn
from torch.nn import functional as F
from dependencies import BASE,shared
spec=importlib.util.spec_from_file_location('frozen_routing_v02',BASE.parent/'models.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
detach=old.detach


class Model(old.Model):
    def __init__(self,arm,config,seed=91001):
        super().__init__(arm,config,seed);self.kernel_calls=0;self.update_records=[]
        if arm in ('M1','M2'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9100101)
                self.writer=nn.Linear(3*config['dim'],config['state_dim'])
                self.update=nn.GRUCell(config['state_dim'],config['state_dim'])
    def reset(self,doc_id,batch=1):
        c=super().reset(doc_id)
        if self.arm in ('M1','M2'):c['state']=self.token.weight.new_zeros(batch,self.config['cells'],self.config['state_dim'])
        c.update(pending_sum=None,pending_count=None,closed=False);return c
    def commit(self,cache):
        if cache['state'] is not None and cache['pending_sum'] is not None:
            s=cache['state'];counts=cache['pending_count'];mean=cache['pending_sum']/counts.clamp_min(1)[:,:,None]
            previous=torch.zeros_like(s) if self.mode in ('recurrence_off','all_off') else s
            with torch.autocast(device_type=s.device.type,enabled=False):
                updated=self.update(mean.float().reshape(-1,self.config['state_dim']),previous.float().reshape(-1,self.config['state_dim'])).reshape_as(s)
            if self.mode!='write_off':cache['state']=torch.where((counts>0)[:,:,None],updated,s)
            if self.diagnostics:self.update_records.append({'position':cache['position'],'call_counts':counts.detach().cpu().tolist(),'update_magnitude_per_cell':(cache['state']-s).detach().norm(dim=-1).cpu().tolist()})
            if self.mode=='reset_chunk':cache['state']=torch.zeros_like(s)
        cache['pending_sum']=cache['pending_count']=None
    def forward(self,ids,cache,generator=None,mask=None):
        if cache['closed']:raise ValueError('finished document; reset required')
        self._dense=mask is None
        mask=torch.ones_like(ids,dtype=torch.bool) if mask is None else mask.bool()
        if ids.shape[0]>1 and not bool(mask.all()):
            # Dense chunk batches only: ragged streaming uses separate document caches.
            if cache['position']!=0:raise ValueError('ragged continuing KV requires per-document sessions')
        if not self._dense and bool((mask[:,1:] & ~mask[:,:-1]).any()):raise ValueError('padding must be a trailing suffix')
        outputs=[];actions=[];offset=0
        while offset<ids.shape[1]:
            n=min(self.config['chunk']-cache['position']%self.config['chunk'],ids.shape[1]-offset)
            logits,cache,logs=self.segment(ids[:,offset:offset+n],cache,generator,mask[:,offset:offset+n])
            outputs.append(logits);actions.append(logs);offset+=n
            if cache['position']%self.config['chunk']==0:self.commit(cache)
        return torch.cat(outputs,1),cache,torch.cat(actions,1) if self.arm!='T0' else []
    def segment(self,ids,cache,generator,mask):
        if self.arm=='T0':return super().forward(ids,cache,generator)
        x=self.token(ids);b,t,d=x.shape;state=cache['state'];pos=cache['position']
        # Only input embedding/norm creates Token KV; never final Cell output.
        attention,kv=self.workspace.chunk(self.work_norm(x),cache['kv'],pos);h=x+attention
        logs=[];selections=[];diagnostics=[]
        addresses=self.address(self.identity.weight)
        if self.arm=='M2' and self.mode not in ('router_off','all_off'):addresses=addresses[None]+self.state_address(state)
        for step in range(2):
            before=h;query=self.query(before)
            scores=query@addresses.T if addresses.ndim==2 else torch.einsum('btr,bnr->btn',query,addresses)
            probabilities=(scores/self.config['router_dim']**.5).softmax(-1)
            flat=probabilities.reshape(-1,self.config['cells'])
            index=torch.multinomial(flat,1,generator=generator).reshape(b,t) if self.sample else probabilities.argmax(-1)
            logs.append(probabilities.gather(-1,index[:,:,None]).clamp_min(1e-30).log().squeeze(-1));selections.append(index)
            identity=self.identity(index);cell_input=self.cell_norm(before)+identity
            if state is not None and self.mode not in ('cell_off','all_off'):
                read_index=(index+1)%self.config['cells'] if self.mode=='shuffle' else index
                selected=state.gather(1,read_index[:,:,None].expand(-1,-1,self.config['state_dim']))
                cell_input=cell_input+self.state_read(selected)
            # One shared core on selected rows, never [tokens, cells, hidden].
            u=self.core(cell_input) if self._dense else torch.zeros_like(cell_input).masked_scatter(mask[:,:,None],self.core(cell_input[mask]).reshape(-1))
            self.kernel_calls+=1;self.calls+=b*t if self._dense else int(mask.sum())
            h=before+self.output(u)
            if state is not None:
                proposal=self.writer(torch.cat((before,u,identity),-1)).tanh()*mask[:,:,None]
                sums=proposal.new_zeros(b,self.config['cells'],self.config['state_dim']).scatter_add(1,index[:,:,None].expand_as(proposal),proposal)
                counts=proposal.new_zeros(b,self.config['cells']).scatter_add(1,index,mask.to(proposal.dtype))
                cache['pending_sum']=sums if cache['pending_sum'] is None else cache['pending_sum']+sums
                cache['pending_count']=counts if cache['pending_count'] is None else cache['pending_count']+counts
            if self.diagnostics:
                entropy=-(probabilities.detach()*probabilities.detach().clamp_min(1e-30).log()).sum(-1)
                diagnostics.append((index.detach().cpu(),entropy.cpu()))
        if self.diagnostics:
            for j in range(t):
                for step,(indices,entropies) in enumerate(diagnostics):
                    for lane in range(b):
                        if bool(mask[lane,j]):self.trace.append({'position':pos+j,'step':step,'cell':int(indices[lane,j]),'entropy':float(entropies[lane,j]),'same_as_first':step==1 and int(indices[lane,j])==int(diagnostics[0][0][lane,j]),'update_magnitude':None,'router_state_contribution_norm':None})
        self.writes+=b*t if self._dense else int(mask.sum())
        return self.head(self.norm(h)),{**cache,'state':state,'position':pos+t,'kv':kv},torch.stack(logs,-1)


class Session:
    """Inference-only segmented API; logical boundaries survive requests and snapshots."""
    def __init__(self,model,doc_id='document',seed=91003):
        self.model=model;self.cache=model.reset(doc_id);self.generator=torch.Generator(device=model.token.weight.device).manual_seed(seed)
    @torch.no_grad()
    def prefill(self,ids):
        result,self.cache,_=self.model(ids,self.cache,self.generator);return result
    @torch.no_grad()
    def finish(self):self.model.commit(self.cache);self.cache['closed']=True
    def snapshot(self):
        return {'cache':copy.deepcopy(self.cache),'rng':self.generator.get_state().clone(),'weights':shared.model_hash(self.model),'config':copy.deepcopy(self.model.config),'mode':self.model.mode,'sample':self.model.sample}
    def restore(self,saved):
        if saved['weights']!=shared.model_hash(self.model) or saved['config']!=self.model.config or saved['mode']!=self.model.mode or saved['sample']!=self.model.sample:raise ValueError('incompatible snapshot')
        self.cache=copy.deepcopy(saved['cache']);self.generator.set_state(saved['rng']);return self
