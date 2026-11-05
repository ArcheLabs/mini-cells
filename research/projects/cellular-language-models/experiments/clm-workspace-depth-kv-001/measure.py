"""Isolated inference timing and declared costs; no alternative model implementation."""
import math,time,statistics
import numpy as np
import torch
from costs import flops


def compute(arm,c):
    base=flops('C2' if arm=='C-DKV' else arm,c);d=c['dim']
    kv=4*d*d*(c['layers'] if arm.startswith('T0') else 2) # Encoder plus Workspace in C2.
    parts=dict(base['parts']);parts['attention']-=kv;parts['kv_projection']=kv+(4*d*d if arm=='C-DKV' else 0)
    if not arm.startswith('T0'):
        parts['cell']=24*d*c['ffn'];parts['encoder_ffn']=6*d*c['encoder_ffn'];del parts['ffn']
    return {'forward_flops_per_token':sum(parts.values()),'parts':parts,'convention':base['convention'],'T0-F_matches':'C2 only; not C-DKV'}


def cache_bytes(cache):
    logical={};stores={}
    for key in ('encoder_kv','workspace_kv','workspace_kv_depth1','layers'):
        values=cache.get(key);values=values if key=='layers' else [values]
        if values is None:continue
        logical[key]=0
        for pair in values:
            if pair is None:continue
            for v in pair:
                logical[key]+=v.numel()*v.element_size();s=v.untyped_storage();stores[s.data_ptr()]=s.nbytes()
    return {'logical_by_cache':logical,'logical_total':sum(logical.values()),'retained_storage_bytes':sum(stores.values())}


def summary(values):return {'p50_seconds':statistics.median(values),'p95_seconds':float(np.percentile(values,95)),'samples_seconds':values}


@torch.no_grad()
def benchmark(model,p,prompt,deadline):
    device=model.token.weight.device
    if device.type!='cuda':raise RuntimeError('real GPU required for inference timing')
    model.eval();model.mode='normal';model.diagnostics=False;model.noise=0.;rows=[];n=p['experiment']['decode_tokens']
    def sync():torch.cuda.synchronize(device)
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('inference benchmark deadline')
    for context in (128,256):
        for batch in (1,8):
            # One pinned validation prompt for every arm/seed, no independent test.
            seed_ids=(prompt*((context+len(prompt)-1)//len(prompt)))[:context]
            full=torch.tensor(seed_ids,device=device)[None].expand(batch,-1)
            inputs=full[:,:context-n];prefills=[];ttfts=[];steps=[];throughputs=[];full_prefills=[];peaks=[];full_peaks=[];memory=None
            for sample in range(p['experiment']['benchmark_warmup']+p['experiment']['benchmark_samples']):
                check();sync();torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
                logits,cache=model(inputs,model.reset());sync();prefill=time.perf_counter()-start
                token=logits[:,-1].argmax(-1,keepdim=True);sync();ttft=time.perf_counter()-start;latencies=[]
                for _ in range(n-1):
                    check();sync();tick=time.perf_counter();logits,cache=model(token,cache);token=logits[:,-1].argmax(-1,keepdim=True);sync();latencies.append(time.perf_counter()-tick)
                peak=torch.cuda.max_memory_allocated(device);del logits,cache
                sync();torch.cuda.reset_peak_memory_stats(device);tick=time.perf_counter();full_logits,cache=model(full,model.reset());sync();full_prefill=time.perf_counter()-tick
                if sample>=p['experiment']['benchmark_warmup']:
                    prefills.append(prefill);ttfts.append(ttft);steps.extend(latencies);throughputs.append(batch*(n-1)/sum(latencies));full_prefills.append(full_prefill);peaks.append(peak);full_peaks.append(torch.cuda.max_memory_allocated(device));memory=cache_bytes(cache)
                del full_logits,cache
            rows.append({'context_limit':context,'prompt_tokens':context-n,'generated_tokens':n,'processed_decode_tokens':n-1,'batch':batch,'prefill':summary(prefills),'ttft':summary(ttfts),'tpot':summary(steps),'decode_tokens_per_second':statistics.median(throughputs),'full_context_prefill':summary(full_prefills),'peak_generation_allocated_bytes':max(peaks),'peak_full_prefill_allocated_bytes':max(full_peaks),'peak_allocated_bytes':max(peaks+full_peaks),'kv_at_full_context':memory,'timing':'FP32 synchronized CUDA; GPU-resident prompt excludes tokenization/H2D; fixed-length argmax ignoring EOS; throughput includes batch factor','warmup':p['experiment']['benchmark_warmup'],'samples':p['experiment']['benchmark_samples'],'prompt_source':'same first validation document, deterministically repeated to fixed length'})
    from gpu_gate import error,accepted
    ids=torch.tensor((prompt*((256+len(prompt)-1)//len(prompt)))[:256],device=device)[None]
    full,_=model(ids,model.reset());cache=model.reset();decoded=[]
    for token in ids.split(1,1):
        check();value,cache=model(token,cache);decoded.append(value)
    parity=error(full,torch.cat(decoded,1))
    if not accepted(parity,'fp32'):raise RuntimeError('trained full/decode parity failed: '+str(parity))
    # Attention/hidden diagnostics are outside the timed region.
    diagnostics={}
    if not model.arm.startswith('T0'):
        from dependencies import shared
        queries=[];reads=[];hook=model.query.register_forward_hook(lambda m,a,o:queries.append(o.detach()))
        read_hook=model.read_out.register_forward_hook(lambda m,a,o:reads.append(o.detach()));model.trace_tensors=True
        ids=torch.tensor(prompt[:128],device=device)[None];_,cache=model(ids);hook.remove();read_hook.remove()
        length=ids.shape[1];pos=torch.arange(length,device=device);allowed=(pos[None]<=pos[:,None])&(pos[None]>pos[:,None]-model.config['window'])
        for step,raw in enumerate(queries):
            q=shared.rope_at(raw.view(1,length,model.config['heads'],-1).transpose(1,2),pos[None,None,:,None])
            k=cache['workspace_kv_depth1'][0] if model.arm=='C-DKV' and step==1 else cache['workspace_kv'][0]
            probabilities=(q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])).masked_fill(~allowed[None,None],float('-inf')).softmax(-1)
            diagnostics['step'+str(step)]={'attention_entropy':float(-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1).mean()),'maximum_probability_mean':float(probabilities.max(-1).values.mean()),'cell_delta_norm':float(model.tensor_trace['steps'][step]['delta'].float().norm(dim=-1).mean())}
        h1=model.tensor_trace['encoder_output']+reads[0]+model.tensor_trace['steps'][0]['delta']
        h2=h1+reads[1]+model.tensor_trace['steps'][1]['delta']
        norms=list(zip(h1.float().norm(dim=-1).flatten().tolist(),h2.float().norm(dim=-1).flatten().tolist()))
        routes=[row['indices'].sort(-1).values.reshape(-1,2).tolist() for row in model.tensor_trace['steps']];paths={}
        for i,(a,b) in enumerate(norms):
            key=str(routes[0][i])+' -> '+str(routes[1][i])
            row=paths.setdefault(key,{'tokens':0,'h1_norm_sum':0.,'h2_norm_sum':0.,'h1_norm_max':0.,'h2_norm_max':0.})
            row['tokens']+=1;row['h1_norm_sum']+=a;row['h2_norm_sum']+=b;row['h1_norm_max']=max(row['h1_norm_max'],a);row['h2_norm_max']=max(row['h2_norm_max'],b)
        diagnostics['cell_paths']=paths
        model.trace_tensors=False;model.tensor_trace={}
    return {'status':'COMPLETE','environment':{'gpu':torch.cuda.get_device_name(device),'torch':str(torch.__version__),'cuda':torch.version.cuda,'precision':'fp32','tf32':False},'parameters':sum(p.numel() for p in model.parameters()),'cases':rows,'diagnostics':diagnostics,'compute':compute(model.arm,model.config),'trained_parity':{'status':'PASS','tokens':256,'errors':parity}}
