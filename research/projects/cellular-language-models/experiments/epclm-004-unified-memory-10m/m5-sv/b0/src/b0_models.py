"""Construct original E0/M5 classes without replacing their forward algorithms."""
from pathlib import Path
import hashlib
import sys
import torch
EXPERIMENT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(EXPERIMENT/'m5/src'))
from bmojo import BMojo
from baselines import build


def construct(arm, protocol):
    seed=protocol['seeds']['model']
    if arm=='E0':return build('E0',protocol['models']['E0'],seed)
    if arm not in ('E6','E7-Innovation'):raise ValueError('unknown B0 arm')
    memory=protocol['memory']
    model=BMojo(protocol['models']['E6_E7'],policy='innovation' if arm=='E7-Innovation' else None,
        summary_enabled=True,seed=seed,beta=protocol['predictor']['beta'],**memory)
    if protocol['seeds']['store']!=seed+3000:raise ValueError('frozen M5 derives store seed as model seed + 3000')
    return model


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_report(model):
    total=sum(p.numel() for p in model.parameters());embedding=model.token_embedding.weight.numel()
    summary=sum(p.numel() for p in model.summary.parameters()) if isinstance(model,BMojo) else 0
    return {'static_parameters':total,'embedding_parameters':embedding,
        'backbone_excluding_embedding_parameters':total-embedding-summary,'summary_parameters':summary,
        'predictor_parameters':0,'store_trainable_parameters':0,
        'tied_embedding_lm_head':model.token_embedding.weight is model.lm_head.weight,
        'weight_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'class':type(model).__module__+'.'+type(model).__name__}


def runtime_report(model):
    if isinstance(model,BMojo):
        if model.last_runtime is None:return {'persistent_runtime_bytes':0,'status':'NO_EPISODE_RUN'}
        return {**model.resources(model.last_runtime),
                'scope':'detached observation of the latest independent Episode; peak allocations measured separately'}
    return {'persistent_runtime_bytes':0,'external_store':False,'state_summary':False,
            'persistent_kv_cache':False,'transient_full_attention_bytes':'not separately measured'}
