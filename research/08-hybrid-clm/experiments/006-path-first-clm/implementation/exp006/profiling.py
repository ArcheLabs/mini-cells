"""Real static-input-memory cache decode, deterministic path replay and wall time."""
from __future__ import annotations
import statistics,time
import torch
from .accounting import ComputeCounter,summarize
@torch.inference_mode()
def parity(model,tokens,atol=1e-5,rtol=1e-4,policy='greedy',generator=None):
    model.eval();full=model(tokens,policy=policy,generator=generator);outputs=[];memory=None
    for n in range(tokens.shape[1]):
        kw={'replay':full.actions[:,n:n+1]} if model.arm!='M0' else {}
        out=model.decode_step(tokens[:,n],memory,**kw);memory=out.memory;outputs.append(out.logits)
    decoded=torch.cat(outputs,1)
    return {'passed':bool(torch.allclose(full.logits,decoded,atol=atol,rtol=rtol)),
            'max_abs_error':float((full.logits-decoded).abs().max()),'atol':atol,'rtol':rtol,'scope':'every position, static K/V, sampled actions replayed'}
@torch.inference_mode()
def benchmark(model,tokens,c):
    model.eval();device=next(model.parameters()).device;e=c['evaluation'];prefix=e['decode_prefix'];steps=e['decode_tokens']
    if tokens.shape[1]<prefix+steps:raise ValueError('insufficient decode prompt')
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    if e['decode_repetitions']<2 or steps<2:raise ValueError('decode needs independent repetitions and warm tokens')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        out=model.prefill(tokens[:,:prefix]);model.decode_step(tokens[:,prefix],out.memory)
    runs=[];prefills=[];first=[]
    for _ in range(e['decode_repetitions']):
        sync();tick=time.perf_counter();out=model.prefill(tokens[:,:prefix]);sync();prefills.append(1000*(time.perf_counter()-tick));row=[]
        for n in range(prefix,prefix+steps):
            sync();tick=time.perf_counter();out=model.decode_step(tokens[:,n],out.memory);sync();row.append(1000*(time.perf_counter()-tick))
        first.append(row[0]);runs.append(row[1:])
    flat=[x for r in runs for x in r];ordered=sorted(flat)
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    out=model.prefill(tokens[:,:prefix])
    with torch.profiler.profile(activities=activities) as profiler:model.decode_step(tokens[:,prefix],out.memory)
    operators={x.key:x.count for x in profiler.key_averages() if x.key.startswith('aten::')}
    cuda_events=[event for event in profiler.events() if 'CUDA' in str(event.device_type)]
    with ComputeCounter(model) as count:out=model.decode_step(tokens[:,prefix],out.memory)
    memory=out.memory
    cache_tensors=[memory.key,memory.value] if hasattr(memory,'key') else [v for layer in memory for v in layer]
    cache_bytes=sum(t.numel()*t.element_size() for t in cache_tensors)
    return {'cache_logical_bytes':cache_bytes,'cache_scope':'logical retained K/V tensors; allocator peak reported separately','batch_size':tokens.shape[0],'dtype':str(next(model.parameters()).dtype),'warm_p50_ms':statistics.median(flat),'warm_p95_ms':ordered[int(.95*(len(ordered)-1))],
            'warm_by_cache_reset':runs,'first_token_ms':first,'prefill_ms':prefills,'decode_compute':summarize(model,out,count),'operator_counts':operators,'cuda_device_events':len(cuda_events) if device.type=='cuda' else None,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'parameter_resident_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'tokens_per_second':1000*tokens.shape[0]/statistics.mean(flat),'scope':'teacher-forced cached decode; isolated worker; no profiler inside timed runs'}
