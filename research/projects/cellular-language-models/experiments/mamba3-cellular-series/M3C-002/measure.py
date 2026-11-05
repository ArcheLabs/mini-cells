"""Inference cost only; random weights, fixed prompt and teacher-forced continuation."""
import time
import torch
from mamba3 import Mamba3Config
from model import Model,cache_bytes


def measure():
    cfg=Mamba3Config(d_model=64,n_layer=2,d_state=16,headdim=16,chunk_size=16,vocab_size=128)
    rows=[]
    for window in (0,16,64):
        model=Model(cfg,window=window,device='cuda').eval()
        for batch in (1,4):
            ids=torch.arange(batch*144,device='cuda').reshape(batch,144)%128
            def run():
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                _,cache=model(ids[:,:128]);torch.cuda.synchronize();prefill=time.perf_counter()-start
                prefill_cache=cache_bytes(cache)
                torch.cuda.synchronize();start=time.perf_counter()
                for t in range(128,144):_,cache=model(ids[:,t:t+1],cache)
                torch.cuda.synchronize();decode=time.perf_counter()-start
                return prefill,decode,prefill_cache,cache_bytes(cache),torch.cuda.max_memory_allocated()
            with torch.inference_mode():
                for _ in range(3):run()
                prefill=[];decode=[];peaks=[]
                for _ in range(5):
                    p,d,prefill_sizes,sizes,peak=run();prefill.append(p*1000);decode.append(d*1000/16);peaks.append(peak)
                prefill.sort();decode.sort()
                rows.append({'arm':'M3-Base' if window==0 else 'M3-WA','window':window,'batch':batch,
                     'context':128,'decode_tokens':16,'scope':'random-weight 2-layer FP32 model, teacher-forced decode, single GPU',
                     'prefill_ms_p50':prefill[2],'prefill_ms_p95_small_sample':prefill[-1],
                     'decode_token_ms_p50':decode[2],'decode_token_ms_p95_small_sample':decode[-1],
                     'peak_allocated_bytes':max(peaks),
                     'prefill_cache':prefill_sizes,'cache':sizes,'parameters':sum(p.numel() for p in model.parameters()),
                     'workspace_parameters':sum(p.numel() for p in model.workspace.parameters()),'samples':5,'warmup':3})
        with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            _,c=model(ids[:1,:128]);model(ids[:1,128:129],c);torch.cuda.synchronize()
        kernels={}
        for event in prof.events():
            if 'CUDA' in str(event.device_type) and not any(x in event.name.lower() for x in ('memcpy','memset')):
                kernels[event.name]=kernels.get(event.name,0)+1
        rows[-2]['kernel_launches_prefill128_decode1_batch1']=sum(kernels.values())
        rows[-2]['kernel_names']=kernels
        rows[-2]['kernel_profiling_status']='CAPTURED' if kernels else 'NOT_CAPTURED'
        del model,c
        torch.cuda.empty_cache()
    return rows
