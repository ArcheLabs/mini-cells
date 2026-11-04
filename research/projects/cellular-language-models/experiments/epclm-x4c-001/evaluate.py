"""Same prediction positions for every model; independent evaluation failure records."""
import math
import random
import statistics
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import native,write,model_hash
from runtime import Session


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def offsets(stream_length,seed,holdout_seed):
    validation=native.fixed_validation_starts(stream_length,seed=seed)
    intervals=[(s,s+129) for batch in validation for s in batch]
    rng=random.Random(holdout_seed);holdout=[]
    for _ in range(384):
        for attempt in range(100000):
            start=rng.randrange(stream_length-129)
            if all(start+129<=a or start>=b for a,b in intervals):break
        else:raise ValueError('unable to construct disjoint holdout spans')
        intervals.append((start,start+129));holdout.append(start)
    return validation,tuple(tuple(holdout[i:i+8]) for i in range(0,len(holdout),8))


@torch.no_grad()
def language(model,stream,starts,device,deadline):
    model.eval();total=0.;count=0;rows=[];began=time.monotonic()
    for batch in starts:
        if time.monotonic()>=deadline:raise TimeoutError('PPL evaluation budget exhausted')
        inputs,targets=native.memmap_batch(stream,batch,128,device);logits=model(inputs)
        losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction='none').reshape(len(batch),-1).sum(-1)
        for start,loss in zip(batch,losses.cpu().tolist()):rows.append({'stream_offset':start,'nll_sum':loss,'prediction_targets':128})
        total+=float(losses.sum());count+=targets.numel()
    if not count or not math.isfinite(total):raise FloatingPointError('invalid language evaluation NLL')
    mean=total/count
    return {'status':'COMPLETE','total_nll':total,'mean_nll':mean,'ppl':math.exp(mean) if mean<700 else None,
            'prediction_tokens':count,'spans':rows,'evaluation_seconds':time.monotonic()-began,'dtype':'float32','unit':'independent stream spans, not documents'}


def sample(logits,generator,mode):
    if mode=='greedy':return int(logits.argmax())
    values,ids=(logits/.8).topk(min(40,len(logits)));p=values.softmax(-1);ordered,indices=p.sort(descending=True)
    ordered[ordered.cumsum(0)-ordered>.95]=0;p.zero_();p[indices]=ordered;p/=p.sum()
    return int(ids[torch.multinomial(p,1,generator=generator)])


@torch.no_grad()
def generation(model,tokenizer,plan,device,deadline,tokenizer_identity):
    model.eval();rows=[]
    for index,prompt in enumerate(plan['generation_prompts']):
        for mode in ('greedy','sampling'):
            session=Session(model,tokenizer_identity);session.generator.manual_seed(plan['generation_seed']+index)
            routing=[]
            ids=tokenizer.encode(prompt).ids;logits=session.prefill(torch.tensor([ids],device=device));generated=[];began=time.monotonic()
            for _ in range(plan['generation_tokens']):
                if time.monotonic()>=deadline:raise TimeoutError('generation budget exhausted')
                if hasattr(model.stages[1].ff,'router'):
                    routing.append({'step':len(generated),'stage2_last_iteration_top2':model.stages[1].ff.router.last['indices'][0,-1].cpu().tolist()})
                token=sample(logits[0,-1],session.generator,mode);generated.append(token)
                if token==3:break
                logits=session.step(torch.tensor([[token]],device=device))
            bigrams=list(zip(generated,generated[1:]));trigrams=list(zip(generated,generated[1:],generated[2:]))
            rows.append({'prompt':prompt,'mode':mode,'seed':plan['generation_seed']+index,'text':tokenizer.decode(generated,skip_special_tokens=True),
                         'routing_by_generated_token':routing,'token_ids':generated,'eos':bool(generated and generated[-1]==3),'seconds':time.monotonic()-began,
                         'distinct_1':len(set(generated))/max(1,len(generated)),'distinct_2':len(set(bigrams))/max(1,len(bigrams)),
                         'repeated_3gram_rate':(len(trigrams)-len(set(trigrams)))/max(1,len(trigrams))})
    return {'status':'COMPLETE','samples':rows,'human_review':'PENDING','no_scientific_quality_claim_from_examples':True}


def performance(model,plan,device,deadline,tokenizer_identity):
    model.eval();rows=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    def measure(call):
        for _ in range(plan['inference_warmup']):call()
        times=[]
        for _ in range(plan['inference_repetitions']):
            if time.monotonic()>=deadline:raise TimeoutError('inference budget exhausted')
            sync(device);start=time.perf_counter();call();sync(device);times.append(time.perf_counter()-start)
        return {'median_seconds':statistics.median(times),'p95_seconds':sorted(times)[math.ceil(.95*len(times))-1]}
    for length in plan['inference_prefill']:
        session=Session(model,tokenizer_identity);ids=torch.full((1,length),2,device=device,dtype=torch.long)
        def prefill():session.reset();session.prefill(ids)
        timing=measure(prefill);rows.append({'tokens':length,**timing})
    session=Session(model,tokenizer_identity);session.prefill(torch.full((1,32),2,device=device,dtype=torch.long))
    saved=session.snapshot();token=torch.tensor([[4]],device=device)
    # Step timing includes evolving the live bounded state; positions remain absolute.
    decode=measure(lambda:session.step(token));snapshot_cost=measure(session.snapshot)
    restore_cost=measure(lambda:session.restore(saved));memory=session.memory
    batch8=Session(model,tokenizer_identity,batch=8);batch8.prefill(torch.full((8,32),2,device=device,dtype=torch.long))
    batch=measure(lambda:batch8.step(torch.full((8,1),4,device=device,dtype=torch.long)))
    return {'status':'COMPLETE','prefill':rows,'batch1_decode':decode,'batch8_decode':batch,'batch8_tokens_per_second':8/batch['median_seconds'],
            'snapshot':snapshot_cost,'restore':restore_cost,'memory':memory,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,'repetitions':plan['inference_repetitions'],
            'dtype':'float32','position_semantics':'absolute-position local windows; not truncated-history recomputation',
            'cache_implementation':'EPCLM-001 IncrementalLM._attend/_append adapted with fixed Chunk state writes'}


@torch.no_grad()
def state_diagnostics(model,device,tokenizer_identity):
    routing={}
    if hasattr(model.stages[1].ff,'router'):
        cell=model.stages[1].ff;router=cell.router;counts=router.activation_counts
        routing={'activation_frequency':None if counts is None else (counts/max(1,router.token_queries)).cpu().tolist(),
                 'cell_utilization':None if counts is None else float((counts>0).float().mean()),
                 'dead_cell_count':None if counts is None else int((counts==0).sum()),
                 'mean_training_routing_entropy':None if router.entropy_sum is None else float(router.entropy_sum/max(1,router.token_queries)),
                 'active_cell_ffn_parameters_per_token':cell.last_active_parameters,'router_parameters':sum(p.numel() for p in router.parameters()),
                 'executed_main_ffn_training_rows':cell.total_training_rows,'training_query_rows':router.token_queries,
                 'counts_include_overflow_attempts':True,
                 'cell_specific_gradient_norm':None if cell.down.grad is None else cell.down.grad.detach().flatten(1).norm(dim=1).cpu().tolist(),
                 'actual_gpu_dram_traffic':'NOT_MEASURED; profiler allocation and device timing are reported separately'}
    if not model.persistent:return {'status':'COMPLETE' if routing else 'NOT_APPLICABLE','reason':'no persistent state','routing':routing}
    ids=torch.tensor([[2]+[4,5,6,7]*20],device=device);model.eval()
    def ablated(read,write):
        previous=(model.read_enabled,model.write_enabled)
        try:
            model.read_enabled=read;model.write_enabled=write
            return model(ids)
        finally:model.read_enabled,model.write_enabled=previous
    baseline=model(ids);no_read=ablated(False,True);no_write=ablated(True,False)
    session=Session(model,tokenizer_identity);session.prefill(ids[:,:40]);saved=session.snapshot()
    normal=session.step(ids[:,40:41]);restored=Session(model,tokenizer_identity).restore(saved)
    state=restored.cache['persistent'];slots=model.persistent.slots(state);slots[:,0]+=1e-3
    pulse=restored.step(ids[:,40:41]);restored=Session(model,tokenizer_identity).restore(saved)
    restored.cache['persistent'].zero_();reset=restored.step(ids[:,40:41])
    return {'status':'COMPLETE','routing':routing,'read_disabled_max_logit_difference':float((baseline-no_read).abs().max()),
            'write_disabled_max_logit_difference':float((baseline-no_write).abs().max()),
            'state_pulse_max_logit_difference':float((normal-pulse).abs().max()),
            'state_reset_max_logit_difference':float((normal-reset).abs().max()),'state_shape':list(state.shape),
            'interpretation':'frozen-weight causal differences are not independently trained structural quality comparisons'}


def all_evaluations(model,corpus,tokenizer,starts,plan,device,out,deadline):
    stream=native.open_memmap(corpus.validation_path);results={};identity=corpus.manifest['tokenizer_sha256']
    tasks={'validation-results.json':lambda:language(model,stream,starts[0],device,deadline),
           'holdout-results.json':lambda:language(model,stream,starts[1],device,deadline),
           'generation-samples.json':lambda:generation(model,tokenizer,plan,device,deadline,identity),
           'inference-profile.json':lambda:performance(model,plan,device,deadline,identity),
           'state-diagnostics.json':lambda:state_diagnostics(model,device,identity)}
    for filename,call in tasks.items():
        try:
            if time.monotonic()>=deadline:raise TimeoutError('evaluation budget exhausted')
            results[filename]=call();write(out/filename,results[filename])
        except Exception as error:
            results[filename]={'status':'FAILED','error':type(error).__name__+': '+str(error)}
            write(out/filename,results[filename])
    try:
        if time.monotonic()>=deadline:raise TimeoutError('memory evaluation budget exhausted')
        results['memory-profile.json']=memory_profile(model,results,device)
    except Exception as error:results['memory-profile.json']={'status':'FAILED','error':type(error).__name__+': '+str(error)}
    write(out/'memory-profile.json',results['memory-profile.json'])
    return results


def memory_profile(model,results,device):
    p=next(model.parameters());weights=sum(v.numel()*v.element_size() for v in model.parameters())
    memory={'status':'MEASURED' if device.type=='cuda' else 'CPU_FUNCTIONAL_ONLY','model_parameter_bytes':weights,'registered_buffer_bytes':sum(v.numel()*v.element_size() for v in model.buffers()),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if p.is_cuda else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if p.is_cuda else None,
            'inference_peak_cuda_allocated_bytes':results.get('inference-profile.json',{}).get('peak_cuda_allocated_bytes'),
            'inference_peak_cuda_reserved_bytes':results.get('inference-profile.json',{}).get('peak_cuda_reserved_bytes'),
            'runtime_memory':results.get('inference-profile.json',{}).get('memory'),
            'temporary_activation_memory':'peak allocated includes weights, caches and transient tensors; not separately attributed'}
    runtime=memory['runtime_memory']
    memory['total_known_runtime_tensor_bytes']=weights+memory['registered_buffer_bytes']+(runtime.get('runtime_tensor_storage_bytes',0) if runtime else 0)
    memory['total_scope']='parameters + registered buffers + unique runtime tensor storage; excludes Python/allocator overhead, which CUDA allocated/reserved expose separately'
    if device.type=='cuda':
        sync(device);baseline=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():output=model(torch.full((1,125),2,device=device,dtype=torch.long))
        sync(device);memory['forward_peak_increment_bytes_including_outputs_and_internal_state']=max(0,torch.cuda.max_memory_allocated(device)-baseline)
        memory['forward_output_bytes']=output.numel()*output.element_size();del output
    return memory
