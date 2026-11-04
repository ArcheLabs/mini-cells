"""One evaluation path for frozen Core models and the explicit T1 reference."""
import hashlib
import io
import json
import resource
import time
import torch
from runtime import Runtime,weight_hash
from trainer import evaluate,atomic_save
from native_clm_runtime import sha256_file
from frozen_model import BaselineIntegrityError,check_deadline
from mechanism_diagnostic import write,sync
from generative_baselines import freeze,linear_flops


def audit_data(assets,tokenizer,plan):
    metadata=json.loads((assets/'assets/manifest.json').read_text())
    for name,sha in metadata['files'].items():
        if sha256_file(assets/'assets'/name)!=sha:raise BaselineIntegrityError('data asset hash mismatch: '+name)
    train=json.loads((assets/'assets/train.json').read_text());dev=json.loads((assets/'assets/development.json').read_text())
    special={name:tokenizer.token_to_id(name) for name in plan['tokenizer_special_names']}
    if any(v is None for v in special.values()) or len(set(special.values()))!=len(special):raise BaselineIntegrityError('invalid tokenizer special IDs')
    if tokenizer.get_vocab_size()!=2048 or len(dev)<=16:raise BaselineIntegrityError('vocabulary/development split unavailable')
    seen=set();token_seen=set()
    for document in train+dev:
        tokens=document['tokens']
        signature=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        if document['id'] in seen or signature in token_seen:raise BaselineIntegrityError('duplicate story/tokens across audited corpus')
        seen.add(document['id']);token_seen.add(signature)
        if (len(tokens)<2 or tokens[0]!=special['<bos>'] or tokens[-1]!=special['<eos>']
                or any(not isinstance(t,int) or not 0<=t<2048 for t in tokens)
                or any(t in (special['<pad>'],special['<bos>'],special['<eos>']) for t in tokens[1:-1])):
            raise BaselineIntegrityError('document BOS/EOS/reset contract differs')
    groups={'train':train,'validation':dev[:16],'development_holdout':dev[16:]}
    report={'status':'PASS_DEVELOPMENT_ONLY','special_tokens':special,
            'splits':{k:{'documents':len(v),'prediction_tokens':sum(len(r['tokens'])-1 for r in v),
                         'ordered_story_ids_sha256':hashlib.sha256(json.dumps([r['id'] for r in v]).encode()).hexdigest()} for k,v in groups.items()},
            'document_content_and_token_duplicates':'NONE','document_reset':'EVERY_DOCUMENT',
            'test_status':'NOT_AVAILABLE','tokenizer_overlap':metadata['tokenizer_overlap'],
            'independent_confirmation':'SEALED_NOT_ACCESSED','eos_training':'EOS is a real final next-token target in every training document'}
    return groups,report


def session(model,tokenizer,model_identity,tokenizer_sha,seed):
    return Runtime(model).generation(tokenizer,model_identity,tokenizer_sha,seed,eos_trained=True)


def pilot(model,tokenizer,model_identity,tokenizer_sha,plan,output,deadline):
    rows=[]
    for index in plan['pilot']['prompt_indices']:
        check_deadline(deadline);prompt=plan['prompts'][index];seed=plan['generation_seed']+index
        a=session(model,tokenizer,model_identity,tokenizer_sha,seed);a.deadline=deadline;a.prefill(prompt)
        sync(model);start=time.monotonic();continuous=a.generate(plan['pilot']['max_new_tokens'],temperature=0.)
        sync(model);elapsed=time.monotonic()-start
        b=session(model,tokenizer,model_identity,tokenizer_sha,seed);b.deadline=deadline;b.prefill(prompt)
        first=b.generate(plan['pilot']['snapshot_after'],temperature=0.);snapshot=b.snapshot()
        atomic_save(snapshot,output/f'pilot-{index}-session.pt')
        c=session(model,tokenizer,model_identity,tokenizer_sha,seed);c.deadline=deadline;c.restore(snapshot)
        tail=c.generate(plan['pilot']['max_new_tokens']-plan['pilot']['snapshot_after'])
        if continuous['token_ids']!=first['token_ids']+tail['token_ids'] or not torch.equal(a.state,c.state):
            raise BaselineIntegrityError('GPU generation/snapshot continuation mismatch')
        rows.append({'prompt_index':index,'generated_tokens':continuous['generated_tokens'],'decode_seconds':elapsed,
                     'stop_reason':continuous['stop_reason'],'tokens_per_second':continuous['generated_tokens']/max(elapsed,1e-9),
                     'resume':'PASS','numerical_state_equality':True,'continued_after_restore_tokens':tail['generated_tokens']})
    return {'status':'PASS','prompts':rows,'new_training_tokens':0,'scope':'one greedy 5-prompt pilot, 32 max tokens; continuous vs 20+restore+12',
            'numeric_determinism':'observed exact equality on this environment','token_determinism':'PASS','sampling_rng_resume':'CPU_CONTRACT_PASS_GPU_SAMPLING_EVALUATED_SEPARATELY'}


def repetition_metrics(tokens):
    def ngrams(n):return [tuple(tokens[i:i+n]) for i in range(max(0,len(tokens)-n+1))]
    unigrams=ngrams(1);bigrams=ngrams(2);trigrams=ngrams(3)
    return {'distinct_1':len(set(unigrams))/len(unigrams) if unigrams else 0.,
            'distinct_2':len(set(bigrams))/len(bigrams) if bigrams else 0.,
            'repeated_3gram_rate':(len(trigrams)-len(set(trigrams)))/len(trigrams) if trigrams else 0.}


def generate_sample(model,tokenizer,model_identity,tokenizer_sha,prompt,index,mode,plan,deadline):
    check_deadline(deadline);runtime=session(model,tokenizer,model_identity,tokenizer_sha,plan['generation_seed']+index)
    runtime.deadline=deadline
    sync(model);start=time.monotonic();prefill=runtime.prefill(prompt);sync(model);prefill_seconds=time.monotonic()-start
    sync(model);start=time.monotonic()
    result=runtime.generate(plan['generation']['max_new_tokens'],**{k:v for k,v in mode.items() if k!='name'})
    sync(model);decode_seconds=time.monotonic()-start
    if result['events']!=prefill['consumed_tokens']+result['generated_tokens']:raise BaselineIntegrityError('generated-token event count differs')
    resume_status='NOT_RUN';resume_seconds=0.;resumed_tokens=0
    if model_identity.get('transition')=='C1_NORMAL' and mode['name']=='sampling' and index==0:
        sync(model);resume_start=time.monotonic()
        first=session(model,tokenizer,model_identity,tokenizer_sha,plan['generation_seed']+index);first.deadline=deadline;first.prefill(prompt)
        head=first.generate(20,**{k:v for k,v in mode.items() if k!='name'})
        continued=session(model,tokenizer,model_identity,tokenizer_sha,plan['generation_seed']+index);continued.deadline=deadline;continued.restore(first.snapshot())
        tail=continued.generate(plan['generation']['max_new_tokens']-20);resumed_tokens=tail['generated_tokens']
        if result['token_ids']!=head['token_ids']+tail['token_ids'] or not torch.equal(runtime.state,continued.state) or not torch.equal(runtime.rng.get_state(),continued.rng.get_state()):
            raise BaselineIntegrityError('GPU sampling snapshot continuity failed')
        sync(model);resume_seconds=time.monotonic()-resume_start;resume_status='PASS'
    eos=tokenizer.token_to_id('<eos>');visible=[t for t in result['token_ids'] if t!=eos]
    return {'prompt_index':index,'prompt':prompt,'mode':mode,'seed':plan['generation_seed']+index,
            'text':result['text'],'generated_tokens':result['generated_tokens'],'visible_tokens':len(visible),
            'requested_max_new_tokens':plan['generation']['max_new_tokens'],'stop_reason':result['stop_reason'],
            'sampling_resume':resume_status,'sampling_resume_verification_seconds':resume_seconds,'sampling_resumed_tokens':resumed_tokens,
            'prefill_tokens':prefill['consumed_tokens'],'prefill_seconds':prefill_seconds,'decode_seconds':decode_seconds,
            'tokens_per_second':result['generated_tokens']/max(decode_seconds,1e-9),
            'average_token_entropy':result['mean_raw_token_entropy'],'average_sampling_entropy':result['mean_sampling_entropy'],
            **repetition_metrics(visible)}


def aggregate_generation(samples):
    result={}
    for mode in sorted({r['mode']['name'] for r in samples}):
        rows=[r for r in samples if r['mode']['name']==mode]
        result[mode]={'prompts':len(rows),'generated_tokens':sum(r['generated_tokens'] for r in rows),
            'average_generated_length':sum(r['generated_tokens'] for r in rows)/len(rows),
            'eos_rate':sum(r['stop_reason']=='EOS' for r in rows)/len(rows),
            'tokens_per_second':sum(r['generated_tokens'] for r in rows)/max(sum(r['decode_seconds'] for r in rows),1e-9)}
        for key in ['repeated_3gram_rate','distinct_1','distinct_2','average_token_entropy']:
            values=[r[key] for r in rows if r[key] is not None];result[mode][key]=sum(values)/len(values) if values else 'NOT_AVAILABLE'
    return result


def performance(model,tokenizer,model_identity,tokenizer_sha,plan,deadline):
    rows=[];seed_ids=tokenizer.encode('A little child found a ball in the garden. ').ids
    for length in plan['history_lengths']:
        check_deadline(deadline);ids=(seed_ids*((length+len(seed_ids)-1)//len(seed_ids)))[:length-1]
        runtime=session(model,tokenizer,model_identity,tokenizer_sha,plan['generation_seed'])
        runtime.deadline=deadline
        sync(model);start=time.monotonic();runtime.prefill(ids);sync(model);prefill_seconds=time.monotonic()-start
        snapshot=runtime.snapshot();buffer=io.BytesIO();torch.save(snapshot,buffer)
        sync(model);start=time.monotonic();runtime.generate(1,temperature=0.);sync(model);decode=time.monotonic()-start
        rows.append({'history_tokens_including_bos':length,'prefill_seconds':prefill_seconds,'prefill_seconds_per_token':prefill_seconds/length,
                     'single_token_decode_seconds':decode,'persistent_state_bytes':runtime.state.numel()*runtime.state.element_size(),
                     'snapshot_bytes_after_prefill':buffer.tell(),'nonzero_state_blocks_after_prefill':int((snapshot['core']['state'].norm(dim=-1)>1e-6).sum()),'snapshot_scope':'bounded reference token context, identities and RNG' if model.config.arm=='T1' else 'state, identities, counters, private RNG; no prompt token history',
                     'context_window':getattr(model,'context','NOT_APPLICABLE'),'retained_reference_context_tokens':min(length,model.context) if model.config.arm=='T1' else 'NOT_APPLICABLE','kv_cache_bytes':0,
                     'kv_cache_status':'DISABLED_BOUNDED_CONTEXT_REPLAY' if model.config.arm=='T1' else 'NOT_USED'})
    return {'history_scaling':rows,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if next(model.parameters()).is_cuda else 'NOT_AVAILABLE',
            'cpu_process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'gpu_utilization':'NOT_MEASURED','memory_scope':'process high-water marks; includes other resident frozen assets',
            'timing_scope':'synchronized end-to-end forward/finite checks/readout/CPU sampling, excludes network/assets',
            'flops':linear_flops(model),'prefill_complexity':'C1 processes every input event; fixed state does not make prefill constant time'}


def evaluate_model(model,tokenizer,model_identity,tokenizer_sha,groups,plan,entry,output,persist,deadline):
    freeze(model);before=weight_hash(model)
    if next(model.parameters()).is_cuda:torch.cuda.reset_peak_memory_stats()
    for split in ('validation','development_holdout'):
        if split not in entry:
            values=evaluate(model,groups[split],deadline=deadline)
            entry[split]={k:v for k,v in values.items() if k!='stories'}
            entry[split].update(documents=len(values['stories']),tokens=sum(r['tokens'] for r in values['stories']))
            persist()
    entry['test']={'status':'NOT_AVAILABLE','reason':'no independent test in original pinned assets; development holdout has tokenizer-overlap limitation'}
    samples=entry.setdefault('samples',[])
    for mode in plan['generation']['modes']:
        for index,prompt in enumerate(plan['prompts']):
            if any(r['prompt_index']==index and r['mode']['name']==mode['name'] for r in samples):continue
            samples.append(generate_sample(model,tokenizer,model_identity,tokenizer_sha,prompt,index,mode,plan,deadline));persist()
    entry['generation_metrics']=aggregate_generation(samples)
    if 'performance' not in entry:entry['performance']=performance(model,tokenizer,model_identity,tokenizer_sha,plan,deadline)
    if weight_hash(model)!=before:raise BaselineIntegrityError('evaluation modified model weights')
    entry.update(status='COMPLETE',weights_before=before,weights_after=before,
                 allocated_parameters=sum(p.numel() for p in model.parameters()),frozen_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 weight_tensor_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                 persistent_state_dimension=model.init_state(1).numel(),state_memory_bytes=model.init_state(1).numel()*4)
    persist()
