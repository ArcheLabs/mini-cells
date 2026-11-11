"""Freeze and verify the standalone EPCLM-004 E2 engineering baseline."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

M2 = Path(__file__).resolve().parents[3]/'epclm-004-unified-memory-10m/m2'
sys.path.insert(0,str(M2))
from baselines import build
from e0 import ROOT, write
from data import file_hash

SEED = 41017


def hidden_forward(model, ids):
    captured = []
    handle = model.final_norm.register_forward_hook(lambda _module,_inputs,output: captured.append(output))
    try:
        logits = model(ids)
    finally:
        handle.remove()
    if len(captured)!=1:
        raise RuntimeError('Expected exactly one final-normalized hidden tensor')
    return captured[0], logits


def model_and_config():
    spec = json.loads((M2/'configs/protocol.json').read_text())['arms']['E2']
    model = build('E2',spec['model'],SEED,state_size=spec['state_size']).eval().requires_grad_(False)
    assert len(model.blocks)==18 and set(model.kinds)=={'mamba1'}
    assert model.lm_head.weight is model.token_embedding.weight
    assert sum(p.numel() for p in model.parameters())==10_092_288
    assert all(b.attn.d_state==20 and b.attn.d_conv==4 and b.attn.expand==2 for b in model.blocks)
    return model,spec


def source_lock():
    upstream = json.loads((M2/'configs/upstream-lock.json').read_text())
    for path,sha in upstream['vendored_sha256'].items():
        if file_hash(M2/path)!=sha:
            raise ValueError(f'Vendored source mismatch: {path}')
    modules = ('baselines','e0','data','trainer','cached_transformer','native_clm_runtime')
    files = [Path(sys.modules[n].__file__).resolve() for n in modules]
    files.extend(M2/n for n in ('configs/protocol.json','configs/upstream-lock.json'))
    files.extend(M2/n for n in upstream['vendored_sha256'])
    return dict(upstream=upstream,files={str(p.relative_to(ROOT)):file_hash(p) for p in files},
                python=platform.python_version(),torch=torch.__version__,platform=platform.platform(),
                einops=importlib.metadata.version('einops'),backend='PyTorch Mamba-1 reference')


def serialize(model):
    stream=io.BytesIO()
    torch.save({name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()},stream)
    return stream.getvalue()


@torch.no_grad()
def validate(model):
    ids=torch.tensor([[2,4,8,264,3],[2,4,9,265,3]])
    original=model(ids[:1])
    hidden,logits=hidden_forward(model,ids[:1])
    assert hidden.shape==(1,5,256)
    torch.testing.assert_close(logits,original,atol=0,rtol=0)
    repro=model.lm_head(hidden)
    torch.testing.assert_close(repro,original,atol=0,rtol=0)
    first,state=model.cached(ids[:1,:3])
    tail,state=model.cached(ids[:1,3:],state)
    full=torch.cat([first,tail],1)
    torch.testing.assert_close(full,original,atol=2e-5,rtol=2e-5)
    together=model(ids)
    isolated=model(ids[1:])
    mismatches=[]
    for batched,single in ((together[:1],original),(together[1:],isolated)):
        diff=(batched-single).abs()
        mismatches.append(dict(max_abs=float(diff.max()),
                               outside_original_tolerance=int((diff>1e-6+1e-5*single.abs()).sum())))
    # Keep batch shape constant and perturb the other stream only.
    for lane in (0,1):
        perturbed=ids.clone()
        perturbed[1-lane,2:4]+=10
        changed=model(perturbed)
        torch.testing.assert_close(changed[lane],together[lane],atol=0,rtol=0)
        assert not torch.equal(changed[1-lane],together[1-lane])
    assert state.position==5 and len(state.mamba_state)==18 and not state.kv_cache
    count=sum(t.numel()*t.element_size() for pair in state.mamba_state.values() for t in pair)
    assert count==884_736
    return dict(full_forward='PASS',cached_forward='PASS',hidden_state_adapter='PASS',lm_head_parity='PASS',
                batch_isolation_same_shape='PASS', batch_shape_parity=mismatches,full_cached_error=float((full-original).abs().max()),
                lm_head_error=float((repro-original).abs().max()),kv_entries=0,
                runtime_state_bytes=count,logits_sha256=hashlib.sha256(original.numpy().tobytes()).hexdigest())


def freeze(output):
    output.mkdir(parents=True,exist_ok=False)
    model,config=model_and_config()
    lock=source_lock()
    payload=serialize(model)
    checkpoint=output/'state-dict.pt'
    checkpoint.write_bytes(payload)
    sha=file_hash(checkpoint)
    tokenizer=file_hash(M2.parent/'assets/tokenizer.json')
    write(output/'source-lock.json',lock)
    write(output/'architecture.json',dict(config=config,expansion=2,convolution=4,tied_output=True,
                                         swa=False,state_summary=False,attention_kv=False))
    write(output/'initialization.json',dict(seed=SEED,method='Frozen M2 build E2',trained=False,optimizer_updates=0))
    (output/'checkpoint.sha256').write_text(sha+'  state-dict.pt\n')
    result=validate(model)
    write(output/'validation.json',result)
    manifest=dict(baseline_id='E2-INIT-ONLY',architecture='Mamba-1',status='ENGINEERING_BASELINE',
                  trained=False,hidden_dim=256,layers=18,state_size=20,initializer_seed=SEED,
                  parameter_count=10_092_288,checkpoint_sha256=sha,checkpoint_bytes=len(payload),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  tokenizer_sha256=tokenizer,writer_pid=os.getpid(),validation=result,
                  checkpoint_location='Local run output. Reproduce from locked sources and environment.')
    write(output/'manifest.json',manifest)


def restore(output):
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['writer_pid']!=os.getpid()
    assert source_lock()==json.loads((output/'source-lock.json').read_text())
    checkpoint=output/'state-dict.pt'
    if file_hash(checkpoint)!=manifest['checkpoint_sha256']:
        raise ValueError('State dict hash mismatch')
    model,_=model_and_config()
    assert hashlib.sha256(serialize(model)).hexdigest()==manifest['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    assert hashlib.sha256(serialize(model)).hexdigest()==manifest['checkpoint_sha256']
    actual=validate(model)
    assert actual==manifest['validation']
    actual.update(fresh_restore='PASS',fresh_initializer_byte_identity='PASS',reader_pid=os.getpid())
    write(output/'validation.json',actual)
    manifest['validation']=actual
    write(output/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--phase',choices=['freeze','restore'])
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.phase:
        (freeze if args.phase=='freeze' else restore)(args.output)
    else:
        for phase in ('freeze','restore'):
            subprocess.run([sys.executable,__file__,str(args.output),'--phase',phase],check=True,timeout=180)
