"""Trained-model cache parity and fixed teacher-forced decode cost, after training."""
import time
import torch


def measure(model,ids):
    model.eval();ids=ids[:1];device=next(model.parameters()).device;ids=torch.as_tensor(ids,device=device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize()
    with torch.inference_mode():
        full,_=model(ids);first,cache=model(ids[:,:96]);parts=[first]
        for t in range(96,160):
            logits,cache=model(ids[:,t:t+1],cache);parts.append(logits)
        cached=torch.cat(parts,1);torch.testing.assert_close(full,cached,atol=1e-4,rtol=1e-4)
        error=float((full-cached).abs().max());times=[]
        for repeat in range(8):
            _,cache=model(ids[:,:96]);sync();start=time.perf_counter()
            for t in range(96,112):_,cache=model(ids[:,t:t+1],cache)
            sync()
            if repeat>=3:times.append((time.perf_counter()-start)*1000/16)
        times.sort()
    return {'full_decode_max_error':error,'decode_ms_per_token_p50':times[2],
            'decode_ms_per_token_p95_small_sample':times[-1],'context':96,'teacher_forced_tokens':16,
            'batch':1,'samples':5,'warmup':3,'scope':'trained FP32 model, teacher-forced fixed tokens, no sampling or language PPL'}
