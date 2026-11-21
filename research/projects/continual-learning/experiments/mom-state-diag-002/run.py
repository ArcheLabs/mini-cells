"""First-divergence trace of the pinned MoM, without checkpoint I/O."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap

MOM='125038087b71b635c171cec857bd0f21d10955b6'
SHORT=[1,19,71,112,52,13,541,90]


def first_delta(left,right):
    for i,(a,b) in enumerate(zip(left,right)):
        if a!=b:
            return dict(index=i,left=a,right=b)
    if len(left)!=len(right):
        return dict(lengths=[len(left),len(right)])
    return None


def main():
    p=argparse.ArgumentParser()
    for x in ('source','snapshot','output'):
        p.add_argument(x,type=Path)
    p.add_argument('--worker',choices=('D1','D2'))
    a=p.parse_args()
    if not a.worker:
        for worker in ('D1','D2'):
            print('Launching',worker,flush=True)
            subprocess.run([sys.executable,__file__,str(a.source),str(a.snapshot),str(a.output),
                            '--worker',worker],check=True)
        d1,d2=[json.loads((a.output/(n+'.json')).read_text()) for n in ('D1','D2')]
        comparisons={}
        for key in d1['traces']:
            x,y=d1['traces'][key],d2['traces'][key]
            comparisons[key]=dict(same_process=[first_delta(t[0],t[1]) for t in (x,y)],
                                  cross_process=[first_delta(l,r) for l,r in zip(x,y)])
        clean=all(v is None for c in comparisons.values() for v in c['same_process']+c['cross_process'])
        (a.output/'preflight.json').write_text(json.dumps(dict(
            status='EXACT_MATCH' if clean else 'DIAGNOSTIC_MISMATCH',
            comparisons=comparisons,atol=0,rtol=0),indent=2))
        return

    import torch
    from importlib.metadata import version
    if subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()!=MOM:
        raise ValueError('Wrong MoM source')
    sys.path.insert(0,str(a.source.resolve()))
    from mom.models.mom_gated_deltanet import MomGatedDeltaNetForCausalLM
    from mom.layers import mom_gated_deltanet as layer
    torch.manual_seed(42)
    model,info=MomGatedDeltaNetForCausalLM.from_pretrained(a.snapshot,local_files_only=True,
        torch_dtype=torch.float16,output_loading_info=True)
    if any(info.values()):
        raise ValueError(info)
    model=model.cuda().eval().requires_grad_(False)
    trace=[]
    pending=[]
    current=[None]
    def record(name,value):
        if torch.is_tensor(value):
            # Keep copies on device until the end to avoid host sync at every hook.
            pending.append((current[0],name,value.detach().clone()))
        elif isinstance(value,(list,tuple)):
            for i,v in enumerate(value):
                record(f'{name}/{i}',v)
        elif value is None:
            pending.append((current[0],name,None))

    def before(i):
        def capture(_,args,kw):
            current[0]=i
            record('layer_input',kw['hidden_states'] if 'hidden_states' in kw else args[0])
        return capture
    for i,block in enumerate(model.model.layers):
        block.register_forward_pre_hook(before(i),with_kwargs=True)
        def hook(name):
            def capture(_,args,kw,out):
                if name=='gate':
                    record('router_logits',out)
                    ordered=out.float().softmax(dim=-1).sort(dim=-1,descending=True).values
                    record('router_topk_boundary_ties',ordered[...,model.config.topk-1]==ordered[...,model.config.topk])
                else:
                    record(name+'/input',kw['x'] if 'x' in kw else args[0])
                    record(name+'/output',out)
            return capture
        for name,module in block.attn.named_modules():
            if name=='gate' or name.endswith('conv1d'):
                module.register_forward_hook(hook(name),with_kwargs=True)
    kernels={}
    for name in ('chunk_gated_delta_rule','fused_recurrent_gated_delta_rule'):
        fn=getattr(layer,name)
        def wrap(fn,name):
            def call(*args,**kw):
                if 'head_first' not in inspect.signature(fn).parameters:
                    if kw.pop('head_first',False) is not False:
                        raise ValueError('Expected time-major layout')
                for key in ('q','k','v','g','beta','initial_state'):
                    if key in kw:
                        record(name+'/'+key,kw[key])
                out=fn(*args,**kw)
                record(name+'/output',out)
                return out
            return call
        setattr(layer,name,wrap(fn,name))
        kernels[name]=str(inspect.signature(fn))
    original=layer.transform
    source=textwrap.dedent(inspect.getsource(original))
    needle='sorted_indices = combined.argsort()'
    if source.count(needle)!=1:
        raise ValueError('Unexpected pinned transform')
    traces={}
    for stable in (False,True):
        namespace=dict(vars(layer))
        exec(source.replace(needle,'sorted_indices = combined.argsort(stable=True)' if stable else needle),namespace)
        transform=namespace['transform']
        def traced(x,mask,n,selected,capacity):
            record('selected_memories',selected)
            out=transform(x,mask,n,selected,capacity)
            record('sorted_indices',out[2])
            return out
        layer.transform=traced
        for size in (8,72):
            key=f'{size}_'+('stable' if stable else 'original')
            traces[key]=[]
            for repeat in range(2):
                pending.clear()
                trace=[]
                print(a.worker,key,repeat,flush=True)
                with torch.inference_mode():
                    result=model.model(input_ids=torch.tensor([SHORT*(size//8)],device='cuda'),
                                       use_cache=True,return_dict=True)
                    record('final_hidden',result.last_hidden_state)
                for i,name,t in pending:
                    row=dict(layer=i,stage=name)
                    if t is not None:
                        cpu=t.cpu().contiguous()
                        row.update(shape=list(cpu.shape),dtype=str(cpu.dtype),
                                   sha256=hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
                        if name in ('selected_memories','sorted_indices','router_topk_boundary_ties'):
                            row['values']=cpu.tolist()
                    trace.append(row)
                expected='fused_recurrent_gated_delta_rule' if size==8 else 'chunk_gated_delta_rule'
                if not any(r['stage'].startswith(expected+'/') for r in trace):
                    raise AssertionError(f'Expected {expected}')
                traces[key].append(trace)
    layer.transform=original
    (a.output/(a.worker+'.json')).write_text(json.dumps(dict(
        traces=traces,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
        cuda=torch.version.cuda,triton=version('triton'),fla=version('flash-linear-attention'),
        kernels=kernels,tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cublas_workspace=__import__('os').getenv('CUBLAS_WORKSPACE_CONFIG'),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()),
        separators=(',',':')))


if __name__=='__main__':
    main()
