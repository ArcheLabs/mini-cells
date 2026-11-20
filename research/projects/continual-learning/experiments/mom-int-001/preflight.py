"""Real-weight MoM CUDA adapter and fresh-process cache continuation probe."""
import argparse
import hashlib
import functools
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

MODEL='linear-moe-hub/MoM-Gated-Deltanet-340M'
REVISION='f4081422ea2cf01b1b5d646cac05cad62a5d5438'
MOM='125038087b71b635c171cec857bd0f21d10955b6'
PREFIX=[1,19,71,112,52,13,541,90]
SUFFIX=[43,25,18]


def adapt_time_major_kernel(kernel):
    """Legacy MoM explicitly requests the modern kernel's Time-major layout."""
    if 'head_first' in inspect.signature(kernel).parameters:
        return kernel

    @functools.wraps(kernel)
    def wrapped(*args, head_first=False, **kwargs):
        if head_first is not False:
            raise ValueError('Only the legacy time-major call is supported')
        return kernel(*args, **kwargs)
    return wrapped


def restore_cache(cache_type, layers, seen_tokens):
    # FLA 0.2.2 exports a tuple but its importer only consumes lists.
    cache = cache_type.from_legacy_cache(list(layers), seen_tokens=seen_tokens)
    if len(cache) != len(layers) or any(cache.get_seq_length(i) != seen_tokens
                                       for i in range(len(layers))):
        raise AssertionError(f'Cache round trip: layers={len(cache)}/{len(layers)}, '
                             f'position={cache.get_seq_length()}/{seen_tokens}')
    return cache


def compare(a,b,torch,path=''):
    """Collect every unequal leaf, including per-layer tensors."""
    def mismatch(**details):
        return [dict(path=path,**details)]
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape!=b.shape or a.dtype!=b.dtype:
            return mismatch(reason='tensor schema')
        if torch.equal(a,b):
            return []
        diff=(a.float()-b.float()).abs()
        return mismatch(unequal=int((a!=b).sum()),elements=a.numel(),
                        max_abs=float(diff.max()) if diff.isfinite().all() else None)
    if type(a)!=type(b):
        return mismatch(reason='type')
    if isinstance(a,dict) and a.keys()==b.keys():
        return [d for k in a for d in compare(a[k],b[k],torch,f'{path}/{k}')]
    if isinstance(a,(tuple,list)) and len(a)==len(b):
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in compare(x,y,torch,f'{path}/{i}')]
    return [] if a==b else mismatch(reason='value or structure')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('snapshot',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--restore',action='store_true')
    a=p.parse_args()
    import torch
    import transformers
    if subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()!=MOM:
        raise ValueError('Wrong MoM source revision')
    sys.path.insert(0,str(a.source.resolve()))
    from mom.models.mom_gated_deltanet import MomGatedDeltaNetForCausalLM
    from fla.models.utils import Cache
    from mom.layers import mom_gated_deltanet as layer
    for name in ('chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule'):
        kernel=getattr(layer,name)
        print(f'FLA {name}: {inspect.signature(kernel)}',flush=True)
        setattr(layer,name,adapt_time_major_kernel(kernel))
    print(f'FLA cache restore: {inspect.signature(Cache.from_legacy_cache)}',flush=True)
    torch.manual_seed(42)
    start=time.monotonic()
    model, info = MomGatedDeltaNetForCausalLM.from_pretrained(a.snapshot,local_files_only=True,
        torch_dtype=torch.float16, output_loading_info=True)
    if any(info.values()):
        raise ValueError(f'Unexpected weight loading report: {info}')
    model=model.cuda().eval().requires_grad_(False)
    torch.cuda.synchronize()
    receipt = dict(model=MODEL,revision=REVISION,source=MOM,torch=str(torch.__version__),
                   transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
                   load_seconds=time.monotonic()-start)
    def step(tokens, cache=None):
        with torch.inference_mode():
            # The legacy CausalLM wrapper adds training aux loss to None
            # during label-free inference. Use its unchanged backbone and head.
            result=model.model(input_ids=torch.tensor([tokens],device='cuda'),past_key_values=cache,
                               use_cache=True,return_dict=True)
            from types import SimpleNamespace
            result=SimpleNamespace(logits=model.lm_head(result.last_hidden_state),
                                   past_key_values=result.past_key_values)
        if not torch.isfinite(result.logits).all():
            raise FloatingPointError('MoM logits nonfinite')
        return result
    def clone(x, device):
        if torch.is_tensor(x):
            return x.to(device).clone()
        if isinstance(x,dict):
            return {k:clone(v,device) for k,v in x.items()}
        if isinstance(x,(list,tuple)):
            return type(x)(clone(v,device) for v in x)
        if x is None or isinstance(x,(int,bool)):
            return x
        raise TypeError(f'Unknown cache member: {type(x)}')
    def capture(result):
        cache=result.past_key_values
        return dict(logits=result.logits.cpu(), layers=clone(cache.to_legacy_cache(),'cpu'),
                    seen_tokens=cache.get_seq_length())

    bundle=a.output/'cache.pt'
    if a.restore:
        saved=torch.load(bundle,map_location='cpu',weights_only=True)
        cache=restore_cache(Cache,clone(saved['layers'],'cuda'),saved['seen_tokens'])
        restored_initial=clone(cache.to_legacy_cache(),'cpu')
        restored=capture(step(SUFFIX,cache))
        fresh=step(PREFIX)
        prefix=capture(fresh)
        rebuilt=capture(step(SUFFIX,fresh.past_key_values))
        torch.save(dict(initial=restored_initial,restored=restored,prefix=prefix,rebuilt=rebuilt),
                   a.output/'restored.pt')
        return
    result=step(PREFIX)
    cache=result.past_key_values
    prefix=capture(result)
    state=prefix['layers']
    if len(state)!=24 or cache.get_seq_length()!=len(PREFIX):
        raise AssertionError('Unexpected checkpoint layer count or cache position')
    for layer in state:
        if (len(layer['recurrent_state'])!=5 or layer['conv_state'] is None
                or not any(torch.is_tensor(t) for t in layer['recurrent_state'])):
            raise AssertionError('Expected expert recurrence and convolution state')
    torch.save(dict(layers=state,seen_tokens=prefix['seen_tokens']),bundle)
    reference=capture(step(SUFFIX,cache))
    saved=torch.load(bundle,map_location='cpu',weights_only=True)
    local_cache=restore_cache(Cache,clone(saved['layers'],'cuda'),saved['seen_tokens'])
    local_initial=clone(local_cache.to_legacy_cache(),'cpu')
    local=capture(step(SUFFIX,local_cache))
    del result,cache,local_cache,model
    torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(Path(__file__).resolve()),str(a.source.resolve()),
                    str(a.snapshot.resolve()),str(a.output.resolve()),'--restore'],check=True)
    actual=torch.load(a.output/'restored.pt',weights_only=True)
    comparisons={key: compare(x,y,torch) for key,x,y in [
        ('local_initial_cache',state,local_initial),
        ('worker_initial_cache',state,actual['initial']),
        ('local_resume',reference,local),
        ('worker_resume',reference,actual['restored']),
        ('worker_rebuilt_prefix',prefix,actual['prefix']),
        ('worker_rebuilt_continuation',reference,actual['rebuilt']),
        ('worker_resume_vs_rebuild',actual['rebuilt'],actual['restored'])]}
    passed=(reference['seen_tokens']==len(PREFIX)+len(SUFFIX) and
            all(not comparisons[k] for k in ('local_initial_cache','worker_initial_cache',
                                             'local_resume','worker_resume')))
    with (a.snapshot/'model.safetensors').open('rb') as f:
        receipt['weights_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    receipt.update(status='PASS' if passed else 'FAIL',scope='REAL_MOM_CACHE_PREFLIGHT_ONLY',
                   comparisons=comparisons, cache_layers=len(state),
                   cache_keys=sorted(state[0]),bundle_bytes=bundle.stat().st_size,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (a.output/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
