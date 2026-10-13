"""Executed shapes, distinct touched weights and repeated path work are separate."""
from __future__ import annotations
import torch
from .reuse import ObservedGemmFlops
from .cells import ContextRead
class ComputeCounter(ObservedGemmFlops):
    def __enter__(self):
        super().__enter__()
        if self.model.arm!='M0':
            def attention(module,inputs,output):
                n=inputs[0].shape[0];length=inputs[1].key.shape[1]
                self.add('context.qk_av',4*n*length*module.width)
            self.handles.append(self.model.context.register_forward_hook(attention))
            if self.model.arm=='R2':
                self.handles.append(self.model.controller.register_forward_hook(lambda m,i,o:self.add('controller.edge_dot',0 if len(i)>3 and i[3]=='zero' else 2*i[0].shape[0]*m.actions*m.source.embedding_dim)))
        else:
            for block in self.model.blocks:
                def attention(module,inputs,output):
                    b,n,d=inputs[0].shape;self.add('context.qk_av',4*b*n*output[1][0].shape[-2]*d)
                self.handles.append(block.register_forward_hook(attention))
        self.handles.append(self.model.register_forward_hook(lambda m,i,o:self.add('tied_lm_head',2*o.logits.numel()*m.config['width'])))
        return self

def touched_weights(model,rollout):
    c=model.config;d=c['width'];paths=rollout.actions.detach().cpu();counts=[]
    shared=c['vocab_size']*d+d+2*d  # full tied head, one positional row, final LN
    if model.arm=='M0':
        shared+=c['moe_layers']*(4*d*d+4*d+c['moe_experts']*d)
        for row in paths.reshape(-1,c['moe_layers'],c['moe_top_k']).tolist():
            counts.append(shared+sum(len(set(layer))*2*d*c['moe_hidden'] for layer in row))
    else:
        shared+=5*d*d+4*d  # Init, K/V, memory LN, shared state LN; includes Q/O below only if used
        shared-=2*d*d
        ctrl=model.controller
        dense= (ctrl.static.numel() if model.arm=='R0' else sum(p.numel() for p in ctrl.scorer.parameters())+ctrl.step.weight.numel())
        for row in paths.reshape(-1,c['steps']).tolist():
            used=set(row);n=shared+dense+len(used-{c['cell_count']})*2*d*c['cell_hidden']
            if c['cell_count'] in used:n+=2*d*d
            if model.arm=='R2' and rollout.edge_mode!='zero':n+=(len(set([c['cell_count']+1]+row[:-1]))+c['cell_count']+1)*c['edge_rank']
            counts.append(n)
    return {'mean':sum(counts)/len(counts),'min':min(counts),'max':max(counts),'scope':'unique parameter entries used in forward; excludes training-only critic; not residency or DRAM bytes'}

def summarize(model,rollout,counter):
    p=rollout.actions.detach().cpu();by=dict(counter.by_module);n=rollout.logits.shape[0]*rollout.logits.shape[1]
    flat=p.reshape(n,-1)
    return {'total_params':sum(x.numel() for x in model.parameters()),'unique_touched_weights':touched_weights(model,rollout),
            'gemm_flops':counter.total,'gemm_flops_per_token':counter.total/n,'gemm_by_module':by,
            'controller_gemm_flops':sum(v for k,v in by.items() if 'controller' in k or 'router' in k),
            'context_attention_gemm_flops':by.get('context.qk_av',0),'context_read_calls':int((flat==model.config['cell_count']).sum()) if model.arm!='M0' else n*model.config['moe_layers'],
            'calls_per_token':flat.shape[1],'unique_operations_per_token':sum(len(set(x)) for x in flat.tolist())/n if model.arm!='M0' else sum(sum(len(set(layer)) for layer in row) for row in p.reshape(n,model.config['moe_layers'],model.config['moe_top_k']).tolist())/n,
            'non_gemm_flops':None,'non_gemm_note':'not inferred; profile reports observed operator calls separately','precision':str(next(model.parameters()).dtype),
            'executed_shapes_scope':'dense attention products including masked positions; repeated module executions counted each time'}
