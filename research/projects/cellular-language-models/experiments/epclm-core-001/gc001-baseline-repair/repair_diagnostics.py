"""Read-only quality observations; missing historical trajectories are never fabricated."""
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import torch
from generation import GenerationSession
from runtime import weight_hash
from frozen_model import BaselineIntegrityError,check_deadline
from mechanism_diagnostic import sync,write
from generation_evaluation import repetition_metrics,aggregate_generation


def state_metrics(state,previous=None,bound=10000.,tolerance=1e-6):
    cells=state[0].float();norms=cells.norm(dim=-1)
    normalized=torch.nn.functional.normalize(cells,dim=-1,eps=1e-12)
    cosine=normalized@normalized.T
    active=norms>tolerance
    mask=(~torch.eye(len(cells),device=state.device,dtype=torch.bool)) & active[:,None] & active[None,:]
    pairs=cosine[mask]
    return {'per_cell_norm':norms.cpu().tolist(),'nonzero_cells':int((norms>tolerance).sum()),
            'pairwise_similarity':cosine.cpu().tolist(),'similarity_mean_scope':'nonzero blocks only; zero blocks are not evidence of diversity',
            'mean_pairwise_similarity':float(pairs.mean()) if pairs.numel() else None,
            'max_pairwise_similarity':float(pairs.max()) if pairs.numel() else None,
            'state_norm':float(state.norm()),'dominant_cell_norm_fraction':float(norms.max()/norms.sum().clamp_min(1e-12)),
            'boundary_proximity':float(state.abs().max()/bound),
            'state_change':float((state-previous).norm()) if previous is not None else None,
            'relative_state_change':float((state-previous).norm()/previous.norm().clamp_min(1e-12)) if previous is not None else None}


class ObservedSession(GenerationSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.observe=False;self.trajectory=[];self.observed_states=[]

    @torch.inference_mode()
    def _consume(self,token):
        logits=super()._consume(token)
        # Core transitions are functional. Keep immutable diagnostic references only;
        # never feed this list to readout or the next event. Metrics run after timing.
        if self.observe:self.observed_states.append(self.state)
        return logits


def generate(model,tokenizer,identity,tokenizer_sha,prompt,index,mode,plan,deadline):
    gc=plan['gc_protocol'];seed=gc['generation_seed']+index
    runtime=ObservedSession(model,tokenizer,identity,tokenizer_sha,seed,eos_trained=True);runtime.deadline=deadline
    sync(model);start=time.monotonic();prefill=runtime.prefill(prompt);sync(model);prefill_seconds=time.monotonic()-start
    cellular=model.config.arm!='T1'
    prefill_stats=state_metrics(runtime.state) if cellular else {'state_kind':'BOUNDED_TOKEN_CONTEXT','retained_tokens':int(runtime.state[0,0,-1])}
    initial_state=runtime.state
    runtime.observe=cellular
    if getattr(model.config,'variant',None)=='IC-ALL-SELF' and prefill_stats['nonzero_cells']!=16:
        raise BaselineIntegrityError('independent model does not have 16 nonzero prefill state blocks')
    settings={k:v for k,v in mode.items() if k!='name'}
    sync(model);start=time.monotonic();result=runtime.generate(gc['generation']['max_new_tokens'],**settings);sync(model)
    seconds=time.monotonic()-start
    diagnostic_started=time.monotonic();previous=initial_state
    for state in runtime.observed_states:
        stats=state_metrics(state,previous,model.config.max_state_abs);stats.pop('pairwise_similarity')
        runtime.trajectory.append(stats);previous=state
    runtime.observed_states=[]
    diagnostic_seconds=time.monotonic()-diagnostic_started
    if result['events']!=prefill['consumed_tokens']+result['generated_tokens']:raise BaselineIntegrityError('generation event count differs')
    continuation='NOT_RUN'
    if index==0:
        split=ObservedSession(model,tokenizer,identity,tokenizer_sha,seed,eos_trained=True);split.deadline=deadline;split.prefill(prompt)
        head=split.generate(20,**settings)
        restored=ObservedSession(model,tokenizer,identity,tokenizer_sha,seed,eos_trained=True);restored.deadline=deadline;restored.restore(split.snapshot())
        tail=restored.generate(gc['generation']['max_new_tokens']-20)
        if result['token_ids']!=head['token_ids']+tail['token_ids'] or not torch.equal(runtime.state,restored.state) or not torch.equal(runtime.rng.get_state(),restored.rng.get_state()):
            raise BaselineIntegrityError('repair greedy/sampling continuation mismatch')
        continuation={'status':'PASS','post_restore_tokens':tail['generated_tokens'],'same_environment_exact_state':True}
    ids=[t for t in result['token_ids'] if t!=tokenizer.token_to_id('<eos>')]
    row={'prompt_index':index,'prompt':prompt,'mode':mode,'seed':seed,'text':result['text'],
         'generated_tokens':result['generated_tokens'],'visible_tokens':len(ids),'requested_max_new_tokens':128,
         'stop_reason':result['stop_reason'],'prefill_seconds':prefill_seconds,'decode_seconds':seconds,'state_diagnostic_seconds':diagnostic_seconds,
         'timing_scope':'generation with immutable state references; numerical diagnostics postprocessed outside decode timing',
         'tokens_per_second':result['generated_tokens']/max(seconds,1e-12),'average_token_entropy':result['mean_raw_token_entropy'],
         'average_sampling_entropy':result['mean_sampling_entropy'],**repetition_metrics(ids),
         'snapshot_continuation':continuation,'prefill_state':prefill_stats,
         'state_trajectory':runtime.trajectory,'state_trace_scope':'runtime numerical observations, not semantic labels' if cellular else 'NOT_APPLICABLE_TOKEN_CONTEXT'}
    return row,{'token_ids':result['token_ids'],'state_trajectory':runtime.trajectory}


@torch.inference_mode()
def teacher_diagnostics(model,documents,tokenizer,deadline):
    model.eval();eos=tokenizer.token_to_id('<eos>');count=0;correct=0;confidence=0.;positive=[];negative=0.;negative_n=0;brier=0.;final=[]
    bins=[{'count':0,'confidence_sum':0.,'correct':0} for _ in range(10)]
    for doc in documents:
        state=model.init_state(1)
        for token,target in zip(doc['tokens'][:-1],doc['tokens'][1:]):
            check_deadline(deadline);state,logits=model.step(state,[token])
            if not torch.isfinite(state).all() or not torch.isfinite(logits).all():raise FloatingPointError('non-finite diagnostic forward')
            probabilities=logits.softmax(-1);top=float(probabilities.max());hit=int(int(probabilities.argmax())==target)
            p=float(probabilities[0,eos]);label=int(target==eos);count+=1;correct+=hit;confidence+=top;brier+=(p-label)**2
            bucket=bins[min(9,int(top*10))];bucket['count']+=1;bucket['confidence_sum']+=top;bucket['correct']+=hit
            if label:positive.append({'probability':p,'nll':float(-logits.log_softmax(-1)[0,eos]),'top1':int(int(probabilities.argmax())==eos)})
            else:negative+=p;negative_n+=1
        if model.config.arm!='T1':final.append(state_metrics(state))
    ece=sum(abs(row['confidence_sum']-row['correct']) for row in bins)/max(count,1)
    return {'scope':'teacher-forced fixed validation documents; no generation replay','documents':len(documents),'tokens':count,
            'mean_max_token_probability':confidence/max(count,1),'token_accuracy':correct/max(count,1),'expected_calibration_error_10_bins':ece,
            'eos_targets':len(positive),'mean_eos_target_probability':sum(r['probability'] for r in positive)/max(len(positive),1),
            'mean_eos_target_nll':sum(r['nll'] for r in positive)/max(len(positive),1),
            'eos_target_top1_fraction':sum(r['top1'] for r in positive)/max(len(positive),1),
            'mean_non_eos_position_eos_probability':negative/max(negative_n,1),'eos_brier_score':brier/max(count,1),
            'final_document_states':final,'calibration_bins':bins}


def training_diagnostics(curve,progression):
    if not curve:return {'status':'NOT_AVAILABLE_HISTORICAL_CURVE_NOT_RECOVERED','validation_progression':progression}
    n=len(curve);first=curve[:50];last=curve[-50:];previous=curve[-100:-50]
    return {'status':'OBSERVED','updates':n,'initial_loss':curve[0]['nll'],'final_loss':curve[-1]['nll'],
            'first_50_mean_loss':sum(r['nll'] for r in first)/len(first),'last_50_mean_loss':sum(r['nll'] for r in last)/len(last),
            'recent_loss_reduction':sum(r['nll'] for r in previous)/len(previous)-sum(r['nll'] for r in last)/len(last) if previous else None,
            'max_gradient_norm':max(r['gradient_norm'] for r in curve),'numerical_stability':'FINITE_RECORDED_CURVE',
            'curve':curve,'validation_progression':progression,'progression_scope':'prespecified first4 validation/first4 holdout diagnostic subsets; final quality uses all 16/205 documents'}


def quality_findings(models,plan):
    result={};limits=plan['diagnostics']
    for name,entry in models.items():
        flags=[];metrics=entry.get('generation_metrics',{});teacher=entry.get('teacher_diagnostics',{})
        if any(r.get('repeated_3gram_rate',0)>limits['repetition_alert'] for r in metrics.values()):flags.append('REPETITION_OBSERVED_NOT_A_COHERENCE_SCORE')
        if teacher.get('mean_max_token_probability',0)>limits['confidence_alert']:flags.append('HIGH_CONFIDENCE_REQUIRES_ACCURACY_CALIBRATION_REVIEW')
        states=[s for sample in entry.get('samples',[]) for s in sample.get('state_trajectory',[])]
        if any((s.get('mean_pairwise_similarity') or 0)>limits['pair_similarity_alert'] for s in states):flags.append('HIGH_STATE_SIMILARITY_OBSERVED')
        if any(s.get('boundary_proximity',0)>limits['saturation_ratio'] for s in states):flags.append('NUMERICAL_BOUNDARY_PROXIMITY_OBSERVED')
        if entry.get('origin')=='HISTORICAL_GC001':flags.append('HISTORICAL_FREE_RUNNING_STATE_TRACE_NOT_RECORDED_NO_REGENERATION')
        progression=[v for _,v in sorted(entry.get('progression',{}).items(),key=lambda p:int(p[0])) if v.get('status')=='OBSERVED']
        degradation=(progression[-1]['validation_probe']['nll']-progression[0]['validation_probe']['nll']) if len(progression)>=2 else None
        result[name]={'execution_status':entry.get('status'),'observations':flags or ['NO_HEURISTIC_ALERT'],
                      'validation_probe_change':degradation if degradation is not None else 'NOT_AVAILABLE_HISTORICALLY_NOT_RECORDED',
                      'numerical_training_record':entry.get('training_diagnostics',{}).get('numerical_stability','NOT_AVAILABLE'),
                      'eos_observations':{k:teacher.get(k,'NOT_AVAILABLE') for k in ('mean_eos_target_probability','eos_target_top1_fraction','mean_non_eos_position_eos_probability','eos_brier_score')},
                      'undertraining':'HYPOTHESIS_NOT_ESTABLISHED_FROM_FINAL_PPL',
                      'attractor_causality':'NOT_ESTABLISHED_NUMERICAL_CORRELATION_ONLY','coherence':'HUMAN_REVIEW_PENDING'}
    return result


def blind_review(output,models,plan):
    package=output/'blind-review.csv';keyfile=output/'private/human-key.json'
    all_rows=[(name,row) for name,entry in sorted(models.items()) if entry.get('status')=='COMPLETE' for row in entry.get('samples',[])]
    content_hash=hashlib.sha256(json.dumps([(name,r['prompt_index'],r['mode']['name'],r['text']) for name,r in all_rows],sort_keys=True).encode()).hexdigest()
    if keyfile.exists():
        key=json.loads(keyfile.read_text())
        if key['samples_sha256']!=content_hash:raise BaselineIntegrityError('blinded package sample set changed')
    else:
        order=list(range(len(all_rows)));random.SystemRandom().shuffle(order)
        key={'samples_sha256':content_hash,'mapping':{f'S{i+1:04d}':{'model':all_rows[j][0],'prompt_index':all_rows[j][1]['prompt_index'],'mode':all_rows[j][1]['mode']['name'],'index':j} for i,j in enumerate(order)}}
        write(keyfile,key)
    dimensions=plan['human_review']['dimensions']
    if not package.exists():
        temporary=package.with_suffix('.csv.tmp')
        with temporary.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['reviewer_id','sample_id','prompt','continuation']+dimensions+['comments']);writer.writeheader()
            for sample_id,assignment in key['mapping'].items():
                _,row=all_rows[assignment['index']];writer.writerow({'sample_id':sample_id,'prompt':row['prompt'],'continuation':row['text']})
        temporary.replace(package)
    from native_clm_runtime import sha256_file
    result={'status':'PENDING','reviewers':0,'samples':len(all_rows),'scale':[1,5],'dimensions':dimensions,
            'blind_package_sha256':sha256_file(package),'private_key_sha256':sha256_file(keyfile),
            'comments':'No reviewer scores have been supplied. Reviewer receives only the blind package; named outputs must be withheld during scoring. Inter-reviewer agreement is not available.'}
    scored=output/'human-review-scored.csv'
    if scored.exists():
        scores=[];seen=set();reviewers=set()
        with scored.open(newline='') as stream:
            for row in csv.DictReader(stream):
                reviewer=row.get('reviewer_id','').strip();sample=row.get('sample_id','')
                if not reviewer and not any(row.get(d,'').strip() for d in dimensions):continue
                if sample not in key['mapping'] or not reviewer or (reviewer,sample) in seen:raise BaselineIntegrityError('invalid/duplicate human review ID')
                try:values={d:int(row[d]) for d in dimensions}
                except (ValueError,KeyError):raise BaselineIntegrityError('human review requires six integer ratings')
                if any(v<1 or v>5 for v in values.values()):raise BaselineIntegrityError('human scores outside 1–5')
                reviewers.add(reviewer);seen.add((reviewer,sample))
                scores.append({'sample_id':sample,'model':key['mapping'][sample]['model'],'scores':values,'comments':row.get('comments','')})
        result.update(status='REVIEWS_RECORDED' if scores else 'PENDING',reviewers=len(reviewers),reviews=len(scores),
            comments='Only supplied human ratings; complete source CSV and comments retained privately. Higher is better, including less degeneration.',
            inter_reviewer_agreement='NOT_AVAILABLE' if len(reviewers)<2 else 'NOT_COMPUTED',
            rated_samples=len({r['sample_id'] for r in scores}),
            mean_scores={name:{d:sum(r['scores'][d] for r in scores if r['model']==name)/sum(r['model']==name for r in scores) for d in dimensions} for name in {r['model'] for r in scores}})
        write(output/'private/human-reviews-preserved.json',scores)
    return result


def state_observations(entry):
    samples=[]
    for row in entry.get('samples',[]):
        if 'prefill_state' in row:
            samples.append({k:row[k] for k in ('prompt_index','mode','prefill_state','state_trajectory') if k in row})
    return {'teacher_forced':entry.get('teacher_diagnostics','NOT_AVAILABLE'),
            'free_running':samples or 'NOT_AVAILABLE_HISTORICALLY_NOT_RECORDED_NO_REGENERATION',
            'flat_blocks_note':'Flat storage blocks have no cellular semantic interpretation'}
