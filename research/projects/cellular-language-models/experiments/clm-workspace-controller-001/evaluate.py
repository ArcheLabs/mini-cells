"""Independent evaluation and executable counterfactuals, no training side effects."""
import time,math
import torch
from torch.nn import functional as F
from dependencies import detach
from precision import select,autocast

@torch.no_grad()
def evaluate(model,documents,p,deadline,mode='normal',pair=(0,1)):
    model.eval();model.mode=mode;model.fixed_pair=pair;model.noise=0.;model.diagnostics=True;model.call_counts=[0]*model.config['cells']
    device=model.token.weight.device;model.generator=torch.Generator(device=device).manual_seed(p['counterfactual']['random_seed'])
    rows=[];records=[];began=time.monotonic()
    try:
        for doc in documents:
            ids=doc['ids'];cache=model.reset(doc['doc_id']);nll=0.;count=0
            for start in range(0,len(ids)-1,p['model']['chunk']):
                if time.monotonic()>=deadline:raise TimeoutError('evaluation deadline')
                end=min(start+p['model']['chunk'],len(ids)-1);inputs=torch.tensor([ids[start:end]],device=device)
                if mode=='shuffle':
                    model.mode='normal';model.capture_routes=True;model.diagnostics=False;before_counts=list(model.call_counts);model(inputs,cache);model.call_counts=before_counts
                    tape=model.route_tape;model.capture_routes=False;model.mode=mode;model.diagnostics=True
                    forced=[]
                    for indices,weights in tape:
                        permutation=torch.randperm(indices.numel()//2,device=device,generator=model.generator)
                        forced.append((indices.reshape(-1,2)[permutation].reshape_as(indices),weights.reshape(-1,2)[permutation].reshape_as(weights)))
                    model.forced_routes=forced
                logits,cache=model(inputs,cache);model.forced_routes=None
                y=torch.tensor(ids[start+1:end+1],device=device);nll+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y,reduction='sum'));count+=end-start
                records.extend(model.records)
            rows.append({'doc_id':doc['doc_id'],'nll_sum':nll,'targets':count})
        mean=sum(r['nll_sum'] for r in rows)/sum(r['targets'] for r in rows)
        return {'status':'COMPLETE','mode':mode,'fixed_pair':list(pair) if mode=='fixed' else None,'mean_nll':mean,'ppl':math.exp(mean),'documents':rows,'records':records,'expert_activation_counts':model.call_counts,'seconds':time.monotonic()-began}
    finally:model.mode='normal';model.diagnostics=False;model.forced_routes=None;model.capture_routes=False


@torch.no_grad()
def profile(model,p,deadline,precision_policy=None):
    import statistics
    from torch.profiler import profile as capture,ProfilerActivity
    device=model.token.weight.device;model.eval();model.diagnostics=False;model.mode='normal';model.noise=0.
    policy=precision_policy or select('fp32',device)
    def forward(ids,cache):
        with autocast(policy,device):return model(ids,cache)
    sample=torch.full((1,64),4,device=device,dtype=torch.long);times=[];decodes=[]
    def sync():
        if device.type=='cuda':torch.cuda.synchronize()
    for i in range(8):
        if time.monotonic()>=deadline:raise TimeoutError('profile deadline')
        sync();start=time.perf_counter();_,cache=forward(sample,model.reset());sync()
        if i>=3:times.append(time.perf_counter()-start)
        sync();start=time.perf_counter();forward(sample[:,:1],cache);sync()
        if i>=3:decodes.append(time.perf_counter()-start)
    activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device.type=='cuda' else [])
    model.profiling=True
    with capture(activities=activities,record_shapes=True) as trace:
        forward(sample,model.reset());sync()
    model.profiling=False;events=trace.key_averages()
    return {'status':'COMPLETE','precision_policy':policy,'cuda_kernel_names':[e.name for e in trace.events() if 'CUDA' in str(e.device_type)][:60],'prefill_tokens':64,'prefill_targets_per_second':64/statistics.median(times),'decode_median_seconds':statistics.median(decodes),'decode_p95_seconds':max(decodes),'parameter_bytes':sum(v.numel()*v.element_size() for v in model.parameters()),'encoder_kv_bytes_per_token':2*model.config['dim']*4*(model.config['layers'] if model.arm.startswith('T0') else 1),'workspace_kv_bytes_per_token':0 if model.arm.startswith('T0') else 2*model.config['dim']*4,'controller_persistent_cache_bytes':0,'component_profile':[{'component':e.key,'cpu_total_us':e.cpu_time_total,'gpu_total_us':getattr(e,'device_time_total',0)} for e in events if e.key in ('workspace_encoder','workspace_attention_read','router','sparse_dispatch','cell_ffn','controller_update')],'operator_hotspots':[{'operator':e.key,'count':e.count,'self_cpu_us':e.self_cpu_time_total,'self_gpu_us':getattr(e,'self_device_time_total',0)} for e in sorted(events,key=lambda e:e.self_cpu_time_total,reverse=True)[:15]],'timing_scope':'declared-policy frozen inference, 3 warmup + 5 samples; operator GPU times may overlap and are not additive end-to-end cost'}
