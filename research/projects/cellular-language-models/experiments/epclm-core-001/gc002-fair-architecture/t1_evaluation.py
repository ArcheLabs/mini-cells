"""Frozen development evaluation and cached-inference diagnostics for T1-PM-KV."""
import math
import json
import statistics
import time
import torch
from torch.nn import functional as F
from cached_transformer import cache_memory


def sync(model):
    if next(model.parameters()).is_cuda:torch.cuda.synchronize()


@torch.inference_mode()
def evaluate_documents(model,documents,device):
    model.eval();loss_sum=0.;tokens=0
    for document in documents:
        values=document['tokens'];cache=None
        for start in range(0,len(values)-1,128):
            inputs=values[start:min(start+128,len(values)-1)];targets=values[start+1:start+1+len(inputs)]
            ids=torch.tensor([inputs],device=device,dtype=torch.long);labels=torch.tensor(targets,device=device,dtype=torch.long)
            logits,cache=model.cached(ids,cache);loss_sum+=float(F.cross_entropy(logits[0],labels,reduction='sum'));tokens+=len(targets)
    if not tokens:raise ValueError('empty evaluation split')
    nll=loss_sum/tokens
    perplexity=math.exp(nll) if nll<700 else None
    return {'status':'PASS_DEVELOPMENT_ONLY','documents':len(documents),'prediction_tokens':tokens,
        'negative_log_likelihood':nll,'perplexity':perplexity,
        'perplexity_status':'MEASURED' if perplexity is not None else 'OVERFLOW_NOT_REPRESENTABLE',
        'context_window':model.config.window,'inference_path':'T1-PM-KV incremental cache; independent cache per document',
        'independent_test':'NOT_ACCESSED'}


def _sample(logits,mode,generator):
    if mode=='greedy':return int(logits.argmax())
    values=logits/0.8;top_values,top_indices=torch.topk(values,min(40,values.numel()))
    probs=torch.softmax(top_values,dim=-1);sorted_probs,indices=torch.sort(probs,descending=True)
    remove=sorted_probs.cumsum(0)-sorted_probs>=0.95;sorted_probs[remove]=0
    filtered=torch.zeros_like(probs);filtered[indices]=sorted_probs;filtered/=filtered.sum()
    return int(top_indices[torch.multinomial(filtered,1,generator=generator)])


@torch.inference_mode()
def generation_diagnostics(model,tokenizer,protocol,device):
    model.eval();plan=protocol['evaluation']
    old_path=__import__('pathlib').Path(__file__).resolve().parents[1]/'gc001-baseline-repair'/'protocol.json'
    old=json.loads(old_path.read_text())['gc_protocol'];rows=[];eos=3
    for index,prompt in enumerate(old['prompts']):
        prompt_ids=tokenizer.encode(prompt).ids
        for mode in plan['generation_modes']:
            generator=torch.Generator(device=device);generator.manual_seed(plan['generation_seed']+index)
            logits,cache=model.cached(torch.tensor([prompt_ids],device=device,dtype=torch.long));generated=[]
            for _ in range(plan['generation_max_new_tokens']):
                token=_sample(logits[0,-1],mode,generator);generated.append(token)
                if token==eos:break
                logits,cache=model.cached(torch.tensor([[token]],device=device),cache)
            visible=[token for token in generated if token!=eos]
            trigrams=[tuple(visible[i:i+3]) for i in range(max(0,len(visible)-2))]
            text=tokenizer.decode(generated,skip_special_tokens=True)
            rows.append({'prompt_index':index,'prompt':prompt,'mode':mode,'seed':plan['generation_seed']+index,
                'generated_tokens':len(generated),'stop_reason':'EOS' if generated and generated[-1]==eos else 'MAX_TOKENS',
                'text':text,'distinct_1':len(set(visible))/max(1,len(visible)),
                'distinct_2':len(set(zip(visible,visible[1:])))/max(1,len(visible)-1),
                'repeated_3gram_rate':(len(trigrams)-len(set(trigrams)))/max(1,len(trigrams))})
    aggregate={}
    for mode in plan['generation_modes']:
        name=mode;group=[r for r in rows if r['mode']==name]
        aggregate[name]={'prompts':len(group),'generated_tokens':sum(r['generated_tokens'] for r in group),
            'eos_rate':sum(r['stop_reason']=='EOS' for r in group)/len(group),
            'distinct_1_mean':statistics.mean(r['distinct_1'] for r in group),
            'distinct_2_mean':statistics.mean(r['distinct_2'] for r in group),
            'repeated_3gram_rate_mean':statistics.mean(r['repeated_3gram_rate'] for r in group)}
    return {'status':'PASS','aggregate':aggregate,'samples':rows,'new_training_tokens':0,
        'review':'GENERATION_DIAGNOSTICS_ONLY_NO_HUMAN_REVIEW','independent_test':'NOT_ACCESSED'}


@torch.inference_mode()
def inference_profile(model,tokenizer,protocol,device):
    model.eval();phrase=tokenizer.encode('A little child found a ball in the garden. ').ids;rows=[]
    if next(model.parameters()).is_cuda:torch.cuda.reset_peak_memory_stats()
    for length in (32,128,512,1024):
        ids=(phrase*((length+len(phrase)-1)//len(phrase)))[:length]
        for _ in range(2):model.cached(torch.tensor([ids],device=device,dtype=torch.long))
        prefill=[];decode=[];last_cache=None
        for _ in range(10):
            sync(model);start=time.perf_counter()
            logits,cache=model.cached(torch.tensor([ids],device=device,dtype=torch.long));sync(model);prefill.append(time.perf_counter()-start)
            sync(model);start=time.perf_counter();model.cached(logits[:,-1].argmax(-1,keepdim=True),cache);sync(model);decode.append(time.perf_counter()-start);last_cache=cache
        rows.append({'history_tokens':length,'prefill_median_seconds':statistics.median(prefill),
            'prefill_p95_seconds':sorted(prefill)[math.ceil(.95*len(prefill))-1],
            'decode_median_seconds':statistics.median(decode),'decode_p95_seconds':sorted(decode)[math.ceil(.95*len(decode))-1],
            'decode_tokens_per_second':1/statistics.median(decode),'persistent_kv_memory':cache_memory(model,last_cache)})
    return {'status':'MEASURED_ON_TARGET_GPU','gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU_NOT_ELIGIBLE',
        'precision':'fp32','repetitions_per_length':10,'history_lengths':rows,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        'weight_version_consistency':'T1-02 cache invalidation gate passed before measurement'}
