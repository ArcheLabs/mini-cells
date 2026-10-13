"""Parameter ownership, distinct visits, execution work and GEMM are distinct metrics."""
from __future__ import annotations
from collections import Counter
import torch
from torch import nn
from .reuse import ObservedGemmFlops


def parameter_catalog(model):
    return {name:{'shape':list(p.shape),'numel':p.numel(),'bytes':p.numel()*p.element_size()}
            for name,p in model.named_parameters()}


def module_size(module):
    return sum(p.numel() for p in module.parameters())


class ComputeCounter(ObservedGemmFlops):
    def __init__(self,model):
        super().__init__(model)
        self.parameter_work=Counter();self.token_rows=Counter();self.last_trace={}

    def record(self,name,module,inputs):
        if not self.enabled:return
        x=inputs[0]
        if isinstance(module,nn.Embedding):rows=x.numel();size=module.embedding_dim
        else:rows=x.numel()//x.shape[-1];size=sum(p.numel() for p in module.parameters(recurse=False))
        self.parameter_work[name]+=int(rows*size);self.token_rows[name]+=int(rows)

    def __enter__(self):
        super().__enter__()
        for name,module in self.model.named_modules():
            if isinstance(module,(nn.Linear,nn.LayerNorm,nn.Embedding)):
                self.handles.append(module.register_forward_hook(lambda m,i,o,n=name:self.record(n,m,i)))
        def begin(m,i):
            if self.enabled:self._before_rows=dict(self.token_rows)
        def end(m,i,o):
            if not self.enabled:return
            tokens=o.logits.shape[0]*o.logits.shape[1]
            self.add('tied_lm_head',2*o.logits.numel()*m.config['width'])
            self.parameter_work['tied_lm_head']+=tokens*m.embedding.weight.numel()
            observed={name:rows-self._before_rows.get(name,0) for name,rows in self.token_rows.items()}
            expected=Counter()
            if m.arm=='M0':
                for row in o.actions.reshape(-1,m.config['moe_layers'],m.config['moe_top_k']).tolist():
                    for layer,ids in enumerate(row):
                        for expert in ids:expected[f'blocks.{layer}.ffn.cells.{expert}.up']+=1
            else:
                for action in o.actions.flatten().tolist():
                    expected['context.query' if action==m.config['cell_count'] else f'cells.{action}.up']+=1
            selected={k:v for k,v in observed.items() if k.endswith('.up') or k=='context.query'}
            # No-op context intervention intentionally executes no query projections.
            expected={k:v for k,v in expected.items() if k!='context.query' or observed.get(k,0)>0}
            if any(selected.get(k,0)!=v for k,v in expected.items()) or any(v and k not in expected for k,v in selected.items()):
                raise AssertionError('actual selected-module calls disagree with action trace')
            self.last_trace={'verified':True,'selected_module_token_rows':selected,'expected_token_rows':expected}
        self.handles.append(self.model.register_forward_pre_hook(begin))
        self.handles.append(self.model.register_forward_hook(end))
        if self.model.arm!='M0':
            def attention(module,inputs,output):
                self.add('context.qk_av',4*inputs[0].shape[0]*inputs[1].key.shape[1]*module.width)
            self.handles.append(self.model.context.register_forward_hook(attention))
            def controller(module,inputs,output):
                if not self.enabled:return
                rows=inputs[0].shape[0]
                if module.arm=='R0':self.parameter_work['controller.static_logits']+=rows*module.static.shape[1]
                if module.arm!='R0' or module.training:self.parameter_work['controller.step_rows']+=rows*module.step.embedding_dim
                if module.arm=='R2' and inputs[3]!='zero':
                    self.add('controller.edge_dot',2*rows*module.actions*module.source.embedding_dim)
                    self.parameter_work['controller.target_dot']+=rows*module.target.weight.numel()
            self.handles.append(self.model.controller.register_forward_hook(controller))
        else:
            for block in self.model.blocks:
                def attention(module,inputs,output):
                    batch,length,width=inputs[0].shape
                    self.add('context.qk_av',4*batch*length*output[1][0].shape[-2]*width)
                self.handles.append(block.register_forward_hook(attention))
        return self

    def report(self,tokens):
        return {**super().report(tokens),'executed_parameter_work':sum(self.parameter_work.values()),
                'executed_parameter_work_per_token':sum(self.parameter_work.values())/tokens,
                'parameter_work_by_module':dict(self.parameter_work),
                'parameter_work_scope':'logical per-row parameter entries consumed by actual affine/normalization/embedding operations; repeated calls counted; not distinct weights, FLOPs, residency or DRAM traffic'}


def touched_weights(model,rollout):
    """Exact module inventory + action trace. Never infer sizes from width formulae."""
    c=model.config;paths=rollout.actions.detach().cpu();counts=[]
    # Tied output reads the complete embedding weight, so token input rows add no distinct entries.
    shared=model.embedding.weight.numel()+model.position.embedding_dim+module_size(model.final)
    if model.arm=='M0':
        shared+=sum(module_size(b.na)+module_size(b.nf)+module_size(b.qkv)+module_size(b.out)+module_size(b.ffn.router) for b in model.blocks)
        for row in paths.reshape(-1,c['moe_layers'],c['moe_top_k']).tolist():
            counts.append(shared+sum(sum(module_size(model.blocks[layer].ffn.cells[i]) for i in set(ids)) for layer,ids in enumerate(row)))
    else:
        shared+=module_size(model.init)+module_size(model.memory)+module_size(model.norm)
        ctrl=model.controller
        policy=ctrl.static.numel() if model.arm=='R0' else module_size(ctrl.scorer)+ctrl.step.weight.numel()
        if rollout.training_mode:
            policy+=module_size(ctrl.critic)
            if model.arm=='R0':policy+=ctrl.step.weight.numel()
        for row in paths.reshape(-1,c['steps']).tolist():
            used=set(row)
            size=shared+policy+sum(module_size(model.cells[i]) for i in used if i<c['cell_count'])
            if c['cell_count'] in used and rollout.context_mode!='zero_replay':size+=module_size(model.context)
            if model.arm=='R2' and rollout.edge_mode!='zero':
                size+=len(set([c['cell_count']+1]+row[:-1]))*ctrl.source.embedding_dim+ctrl.target.weight.numel()
            counts.append(size)
    return {'mean':sum(counts)/len(counts),'min':min(counts),'max':max(counts),
            'scope':'exact distinct parameter entries for current token operations; static K/V production attributed to its input token; cached historical activations are not parameter reads; training critic included only in training mode',
            'inventory_method':'named parameter objects/module.numel plus executed action multiplicities'}


def summarize(model,rollout,counter):
    paths=rollout.actions.detach().cpu();tokens=rollout.logits.shape[0]*rollout.logits.shape[1]
    flat=paths.reshape(tokens,-1);catalog=parameter_catalog(model);by=dict(counter.by_module)
    return {'total_params':sum(x['numel'] for x in catalog.values()),
            'parameter_resident_bytes':sum(x['bytes'] for x in catalog.values()),'parameter_catalog':catalog,
            'unique_touched_weights':touched_weights(model,rollout),
            'unique_touched_fraction':touched_weights(model,rollout)['mean']/sum(x['numel'] for x in catalog.values()),
            'executed_work_to_total_params_ratio':sum(counter.parameter_work.values())/tokens/sum(x['numel'] for x in catalog.values()),
            'executed_parameter_work':sum(counter.parameter_work.values()),
            'executed_parameter_work_per_token':sum(counter.parameter_work.values())/tokens,
            'parameter_work_by_module':dict(counter.parameter_work),'execution_trace':counter.last_trace,
            'gemm_flops':counter.total,'gemm_flops_per_token':counter.total/tokens,'gemm_by_module':by,
            'controller_gemm_flops':sum(v for k,v in by.items() if 'controller' in k or 'router' in k),
            'context_attention_gemm_flops':by.get('context.qk_av',0),
            'selected_context_actions':int((flat==model.config['cell_count']).sum()) if model.arm!='M0' else None,
            'context_read_calls':counter.last_trace.get('selected_module_token_rows',{}).get('context.query',0) if model.arm!='M0' else tokens*model.config['moe_layers'],
            'calls_per_token':flat.shape[1],
            'unique_operations_per_token':sum(len(set(x)) for x in flat.tolist())/tokens if model.arm!='M0' else sum(sum(len(set(layer)) for layer in row) for row in paths.reshape(tokens,model.config['moe_layers'],model.config['moe_top_k']).tolist())/tokens,
            'training_mode':rollout.training_mode,'non_gemm_flops':None,
            'non_gemm_note':'not inferred; profile reports operator counts separately',
            'precision':str(next(model.parameters()).dtype),
            'executed_shapes_scope':'dense masked attention products, repeated executions and actual projection/parameter visits; not a 30B/500M achievement'}
