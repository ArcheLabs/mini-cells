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
        ulp=None
        if a.dtype==torch.float16 and a.isfinite().all() and b.isfinite().all():
            def order(t):
                bits=t.contiguous().view(torch.int16).int()
                return torch.where(bits<0,-32768-bits,bits)
            ulp=int((order(a)-order(b)).abs().max())
        rel=diff/a.float().abs().clamp_min(1e-12)
        return mismatch(max_fp16_ulp=ulp,mean_abs=float(diff.mean()) if diff.isfinite().all() else None,
                        max_relative=float(rel.max()) if rel.isfinite().all() else None,unequal=int((a!=b).sum()),elements=a.numel(),
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
    p.add_argument('--phase',choices=('all','C','D1','D2'),default='all')
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
    def readback():
        saved=torch.load(bundle,map_location='cpu',weights_only=True)
        return restore_cache(Cache,clone(saved['layers'],'cuda'),saved['seen_tokens'])

    def finish(cache):
        continuation=step(SUFFIX,cache)
        data=capture(continuation)
        # Fixed token candidates, scored separately on disposable cache copies.
        for label,tokens in [('candidate_a',[71,112]),('candidate_b',[52,13])]:
            branch=restore_cache(Cache,clone(data['layers'],'cuda'),data['seen_tokens'])
            result=step([19]+tokens[:-1],branch)
            data[label]=result.logits[0].float().log_softmax(-1)[range(len(tokens)),tokens].cpu()
        return data

    if a.phase!='all':
        cache=readback() if a.phase=='C' else step(PREFIX).past_key_values
        initial=clone(cache.to_legacy_cache(),'cpu')
        torch.save(dict(initial=initial,final=finish(cache)),a.output/(a.phase+'.pt'))
        return 0
    result=step(PREFIX)
    cache=result.past_key_values
    state=clone(cache.to_legacy_cache(),'cpu')
    if len(state)!=24 or cache.get_seq_length()!=len(PREFIX):
        raise AssertionError('Unexpected cache position or layer count')
    cache_attrs={k:type(v).__name__ for k,v in vars(cache).items()}
    torch.save(dict(layers=state,seen_tokens=cache.get_seq_length()),bundle)
    reference=finish(cache)
    restored=readback()
    local_initial=clone(restored.to_legacy_cache(),'cpu')
    local=finish(restored)
    del result,cache,restored,model
    torch.cuda.empty_cache()
    actual={}
    for phase in ('C','D1','D2'):
        print(f'Starting independent phase {phase}',flush=True)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),str(a.source.resolve()),
                        str(a.snapshot.resolve()),str(a.output.resolve()),'--phase',phase],check=True)
        actual[phase]=torch.load(a.output/(phase+'.pt'),weights_only=True)
    comparisons={k:compare(x,y,torch) for k,x,y in [
        ('B_initial_vs_A',state,local_initial),('C_initial_vs_A',state,actual['C']['initial']),
        ('B_vs_A',reference,local),('C_vs_A',reference,actual['C']['final']),
        ('D1_initial_vs_A',state,actual['D1']['initial']),
        ('D2_initial_vs_D1',actual['D1']['initial'],actual['D2']['initial']),
        ('D1_vs_A',reference,actual['D1']['final']),
        ('D2_vs_D1',actual['D1']['final'],actual['D2']['final']),
        ('C_vs_D1',actual['C']['final'],actual['D1']['final'])]}
    paths=dict(A=reference,B=local,**{k:v['final'] for k,v in actual.items()})
    def hash_tree(x):
        h=hashlib.sha256()
        def add(v):
            if torch.is_tensor(v):
                h.update(repr((v.dtype,tuple(v.shape))).encode())
                h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
            elif isinstance(v,(list,tuple)):
                h.update(repr((type(v).__name__,len(v))).encode())
                for item in v:
                    add(item)
            else:
                h.update(repr(v).encode())
        add(x)
        return h.hexdigest()
    def hashes(layers):
        return [{k:hash_tree(layer[k]) for k in ('recurrent_state','conv_state')} for layer in layers]
    save=lambda name,obj:(a.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    save('layer_hashes.json',dict(initial_A=hashes(state),initial_B=hashes(local_initial),
        **{'initial_'+k:hashes(v['initial']) for k,v in actual.items()},
        final={k:hashes(v['layers']) for k,v in paths.items()}))
    save('answer_tokens.json',{k:{j:v[j].tolist() for j in ('candidate_a','candidate_b')} for k,v in paths.items()})
    passed=all(not v for v in comparisons.values())
    with (a.snapshot/'model.safetensors').open('rb') as f:
        receipt['weights_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    from importlib.metadata import version
    receipt.update(status='EXACT_MATCH' if passed else 'DIAGNOSTIC_MISMATCH',
                   scope='SHORT_TOKEN_DIAGNOSTIC_ONLY',cache_attributes=cache_attrs,
                   cuda=torch.version.cuda,triton=version('triton'),fla=version('flash-linear-attention'),
                   driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   cudnn_deterministic=torch.backends.cudnn.deterministic,
                   tf32=torch.backends.cuda.matmul.allow_tf32,
                   cublas_workspace=__import__('os').getenv('CUBLAS_WORKSPACE_CONFIG'),
                   atol=0,rtol=0,
                   comparisons=comparisons, cache_layers=len(state),
                   cache_keys=sorted(state[0]),bundle_bytes=bundle.stat().st_size,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (a.output/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
