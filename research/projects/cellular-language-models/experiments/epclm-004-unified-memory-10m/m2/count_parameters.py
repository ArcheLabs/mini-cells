"""Actual instance counts and shared initialization checks; never runs a forward."""
from pathlib import Path
import hashlib,json
from baselines import build
from e0 import parameters

BASE=Path(__file__).resolve().parent

def count():
    protocol=json.loads((BASE/'configs/protocol.json').read_text())
    records={};embedding_hash=None;ssm_hashes=None
    for arm,entry in protocol['arms'].items():
        model=build(arm,entry['model'],protocol['model_seed'],state_size=entry['state_size'] or 16)
        record=parameters(model)
        digest=hashlib.sha256(model.token_embedding.weight.detach().numpy().tobytes()).hexdigest()
        if embedding_hash is None:embedding_hash=digest
        assert embedding_hash==digest,'common embedding initialization differs'
        record.update(model=entry,embedding_initialization_sha256=digest,
                      delta_from_frozen_e0=(record['static_parameters']/10130688-1),
                      mixer_kinds=getattr(model,'kinds',['full_attention' if arm=='E0' else 'swa']*entry['model']['layers']))
        if arm in ('E2','E3'):
            values=[]
            for kind,block in zip(model.kinds,model.blocks):
                if kind=='mamba1':
                    h=hashlib.sha256()
                    for name,v in block.attn.state_dict().items():h.update(name.encode());h.update(v.detach().numpy().tobytes())
                    values.append(h.hexdigest())
            if ssm_hashes is None:ssm_hashes=values
            else:assert values==ssm_hashes[:len(values)],'common Mamba initialization differs'
            record['mamba_initialization_sha256_by_logical_index']=values
        assert abs(record['delta_from_frozen_e0'])<=.01
        records[arm]=record
        del model
    return records

if __name__=='__main__':print(json.dumps(count(),indent=2))
