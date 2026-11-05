"""Configure frozen PhaseBStage modules; only G3 replaces Stage 2's dense FFN."""
import torch
from torch import nn
from common import PhaseBLM,phase_c,native,N1_ALPHA
from state import PersistentState
from cell import SelectiveCellFFN

ITERATIONS={'A0':(4,4,4),'A1':(2,2,2),'A2':(1,1,1),'A3':(1,2,1),
            'D_ATTN4_FFN1':(4,4,4),'D_ATTN1_FFN4':(4,4,4)}


class X4C(PhaseBLM):
    def __init__(self,iterations=(4,4,4),state_kind=None,routed=None,dim=188,heads=4,ffn=639,vocab=2048,dense_width=None):
        if len(iterations)!=3 or any(n not in (1,2,3,4) for n in iterations):raise ValueError('three Stage depths in 1..4 required')
        if state_kind not in (None,'flat','cellular'):raise ValueError(state_kind)
        super().__init__(dim=dim,heads=heads,ffn_dim=ffn,vocab_size=vocab,norm_kind='rmsnorm',activation='swiglu',use_rope=True,update='residual-only')
        self.iteration_config=tuple(iterations);self.state_kind=state_kind;self.routed=routed
        self.chunk_size=32;self.read_enabled=True;self.write_enabled=True;self.fixed_training_chunks=state_kind is not None or routed is not None
        self.max_context=4096 # New explicit absolute-position/local-window semantics, not legacy truncation.
        self.attention_masks=[];self.ffn_masks=[]
        for stage,count in zip(self.stages,iterations):
            stage.iterations=count;stage.step=nn.Parameter(stage.step[:count].detach().clone())
            self.attention_masks.append([True]*count);self.ffn_masks.append([True]*count)
        self.persistent=PersistentState(dim,state_kind) if state_kind else None
        if self.persistent:
            self.persistent.apply(native.HistoricalScaledCLM._init_weights)
            nn.init.constant_(self.persistent.write_gate.bias,-2.)
        if routed:
            self.stages[1].ff=SelectiveCellFFN(dim,stateful=routed=='stateful')
            self.stages[1].ff.apply(native.HistoricalScaledCLM._init_weights)
        elif dense_width is not None:
            self.stages[1].ff=native.SwiGLU(dim,dense_width)
            self.stages[1].ff.apply(native.HistoricalScaledCLM._init_weights)
        self.configuration={'iterations':list(iterations),'state_kind':state_kind,'routed':routed,'dim':dim,'heads':heads,
                            'ffn':ffn,'vocab':vocab,'dense_width':dense_width,'chunk_size':32,'alpha':N1_ALPHA,
                            'windows':[s.window for s in self.stages],'attention_masks':self.attention_masks,'ffn_masks':self.ffn_masks,'state_read_rank':None if not self.persistent else self.persistent.query.out_features,
                            'state_write_hidden':None if not self.persistent else self.persistent.write_in.out_features,
                            'cell_ffn':None if not routed else {'shared_hidden':self.stages[1].ff.shared.up.out_features,'adapter_rank':self.stages[1].ff.down.shape[-1],'cells':16,'top_k':2}}

    def forward(self,ids):
        if ids.shape[1]>128:raise ValueError('legacy training/reference limit is 128; use forward_extended for new streaming semantics')
        return self.forward_extended(ids)

    def forward_extended(self,ids,session=None,return_session=False):
        from runtime import parallel_forward
        logits,session=parallel_forward(self,ids,session)
        return (logits,session) if return_session else logits


def build(name,base_iterations=None,**tiny):
    if name in ITERATIONS:
        model=X4C(ITERATIONS[name],**tiny)
        if name=='D_ATTN4_FFN1':model.ffn_masks=[[j==0 for j in range(4)] for _ in range(3)]
        if name=='D_ATTN1_FFN4':model.attention_masks=[[j==0 for j in range(4)] for _ in range(3)]
    else:
        if base_iterations is None or tuple(base_iterations)==(4,4,4):raise ValueError('G2/G3 require a selected reduced G1 architecture')
        states={'B0':None,'B1':'flat','B2':'cellular','C0':'cellular','C1':None,'C2':'cellular','C3':'cellular'}
        if name not in states:raise ValueError(name)
        model=X4C(base_iterations,state_kind=states[name],routed={'C1':'stateless','C2':'stateful'}.get(name),**tiny)
        if name=='C3':
            target=build('C2',base_iterations,**tiny)
            target_count=sum(p.numel() for p in target.parameters());dim=model.dim
            current=sum(p.numel() for p in model.parameters());old=sum(p.numel() for p in model.stages[1].ff.parameters())
            width=max(1,round((target_count-(current-old))/(3*dim)))
            model.stages[1].ff=native.SwiGLU(dim,width);model.stages[1].ff.apply(native.HistoricalScaledCLM._init_weights);model.configuration['dense_width']=width
    model.fixed_training_chunks=name not in ITERATIONS
    model.configuration['training_execution']='fixed_chunk32_state_controls' if model.fixed_training_chunks else 'legacy_parallel_sequence'
    model.configuration['name']=name;model.configuration['attention_masks']=model.attention_masks;model.configuration['ffn_masks']=model.ffn_masks
    return model
