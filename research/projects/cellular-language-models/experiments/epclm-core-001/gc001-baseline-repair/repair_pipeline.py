"""P1 fresh matched controls, P2 existing T1, P3 analysis only; separate result identities."""
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
import torch
from model import Config
from runtime import weight_hash
from trainer import train,load_checkpoint,atomic_save,evaluate,paired
from storage import bundle
from frozen_model import AssetUnavailable,BaselineIntegrityError,recover_frozen_assets,load_frozen_model,check_deadline
from repair_io import timed_budget
from generative_core import load_flat,training_summary
from generative_baselines import freeze,TransformerReference,NoCommunicationCore
from generation_evaluation import audit_data,performance,aggregate_generation
from mechanism_diagnostic import write
from native_clm_runtime import sha256_file
from repair_models import ARMS,build,select_rank,architecture
from repair_io import public_result,recover_namespace
from repair_diagnostics import generate,teacher_diagnostics,training_diagnostics,quality_findings,blind_review,state_observations

ARTIFACT='artifacts/experiments/epclm-core-001/gc001-baseline-repair/'


def dependency(root,batch,identity,optional=False):
    branch='kaggle/epclm-gc001-repair-'+batch+'-'+identity['source'][:12]
    return public_result(root,branch,ARTIFACT+'batch-'+batch,identity['source'],identity['protocol'],optional=optional)


def checkpoint_model(model,path,identity):
    saved=load_checkpoint(path)
    if saved['identity']!=identity or saved['precision']!='fp32':raise BaselineIntegrityError('repair checkpoint source/protocol/architecture/precision differs')
    model.load_state_dict(saved['model'],strict=True)
    if not all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('non-finite/non-FP32 checkpoint weights')
    return saved


def learning(model,arm,documents,groups,entry,path,identity,plan,deadline,persist):
    if not path.exists() and (entry.get('training',{}).get('tokens',0)>0 or entry.get('gradient_activity')):
        raise AssetUnavailable('committed training progress exists but checkpoint is unavailable; refusing fresh retraining')
    if next(model.parameters()).is_cuda:torch.cuda.reset_peak_memory_stats()
    current=load_checkpoint(path) if path.exists() else None
    if current is not None:checkpoint_model(model,path,identity)
    entry.setdefault('gradient_activity',{});entry.setdefault('progression',{})
    def save_progress():
        # Gradients still belong to the committed update; no extra backward/optimizer.
        if any(p.grad is not None for p in model.parameters()):
            for name,p in model.named_parameters():
                if name not in entry['gradient_activity'] and p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad):entry['gradient_activity'][name]=True
        persist()
    for milestone in plan['training']['milestone_updates']:
        check_deadline(deadline)
        if current is None or current['updates']<milestone:
            result=train(model,documents,identity,path,250000,deadline,plan['training']['seed'],batch=4,unroll=64,
                         stop_after_updates=milestone,precision='fp32',checkpoint_interval=plan['training']['checkpoint_interval'],on_checkpoint=save_progress)
            old=entry.get('training',{});summary=training_summary(result)
            summary['elapsed_seconds']+=old.get('elapsed_seconds',0.) if isinstance(old.get('elapsed_seconds'),(int,float)) else 0.
            summary['tokens_per_second']=summary['tokens']/max(summary['elapsed_seconds'],1e-12)
            entry['training']=summary;current=checkpoint_model(model,path,identity);persist()
            if current['updates']<milestone:raise TimeoutError('repair model incomplete; later arms not started')
        key=str(milestone)
        weight_path=path.with_name(arm+'-milestone-'+key+'.pt')
        if current['updates']==milestone and not weight_path.exists():
            atomic_save({'identity':identity,'updates':milestone,'precision':'fp32','model':model.state_dict()},weight_path)
        if key not in entry['progression']:
            if current['updates']!=milestone:
                entry['progression'][key]={'status':'NOT_AVAILABLE_INTERRUPTED_BEFORE_DIAGNOSTIC','updates':milestone}
            else:
                entry['progression'][key]={'status':'OBSERVED','updates':milestone,'training_tokens':current['tokens'],
                    'validation_probe':evaluate(model,groups['validation'][:4],deadline=deadline),
                    'holdout_probe':evaluate(model,groups['development_holdout'][:4],deadline=deadline)}
                for split in ('validation_probe','holdout_probe'):entry['progression'][key][split].pop('stories',None)
            persist()
    if current['tokens']!=250000:raise TimeoutError('prescribed training target not reached')
    if set(entry['gradient_activity'])!=set(dict(model.named_parameters())):
        raise BaselineIntegrityError('not all intended trainable parameters had observed finite nonzero gradients')
    entry['training_diagnostics']=training_diagnostics(current['curve'],entry['progression'])
    if next(model.parameters()).is_cuda:entry['training_peak_vram_bytes']=torch.cuda.max_memory_allocated()
    entry['checkpoint_sha256']=sha256_file(path);entry['checkpoint_bytes']=path.stat().st_size()
    return current


def evaluate_repair(model,tokenizer,tokenizer_sha,groups,entry,plan,out,persist,deadline):
    freeze(model);model.zero_grad(set_to_none=True);before=weight_hash(model)
    if next(model.parameters()).is_cuda:torch.cuda.reset_peak_memory_stats()
    model_identity={'weights_sha256':before,'checkpoint_sha256':entry['checkpoint_sha256'],'architecture':entry['training_identity']['model'],
                    'training_identity':entry['training_identity']}
    for split in ('validation','development_holdout'):
        if split not in entry:
            result=evaluate(model,groups[split],deadline=deadline)
            entry.setdefault('paired_documents',{})[split]=result['stories']
            entry[split]={k:v for k,v in result.items() if k!='stories'}
            entry[split].update(documents=len(result['stories']),tokens=sum(r['tokens'] for r in result['stories']));persist()
    entry['test']={'status':'NOT_AVAILABLE'};samples=entry.setdefault('samples',[])
    for mode in plan['gc_protocol']['generation']['modes']:
        for index,prompt in enumerate(plan['gc_protocol']['prompts']):
            if any(r['mode']['name']==mode['name'] and r['prompt_index']==index for r in samples):continue
            check_deadline(deadline)
            row,private=generate(model,tokenizer,model_identity,tokenizer_sha,prompt,index,mode,plan,deadline)
            write(out/'private'/f"{entry['model_id']}-{mode['name']}-{index}.json",{'identity':entry['training_identity'],**private})
            samples.append(row);persist()
    entry['generation_metrics']=aggregate_generation(samples)
    for mode,metrics in entry['generation_metrics'].items():
        rows=[r['average_sampling_entropy'] for r in samples if r['mode']['name']==mode and r['average_sampling_entropy'] is not None]
        metrics['average_sampling_entropy']=sum(rows)/len(rows) if rows else 'NOT_AVAILABLE'
    if 'performance' not in entry:
        entry['performance']=performance(model,tokenizer,model_identity,tokenizer_sha,plan['gc_protocol'],deadline)
        entry['performance']['flops']=entry['architecture'].get('estimated_forward_linear_flops','NOT_MEASURED')
    if 'teacher_diagnostics' not in entry:entry['teacher_diagnostics']=teacher_diagnostics(model,groups['validation'],tokenizer,deadline)
    if weight_hash(model)!=before:raise BaselineIntegrityError('evaluation modified trained weights')
    entry.update(status='COMPLETE',weights_sha256=before,weights_unchanged=True,model_identity=model_identity)
    persist()


def compare(models):
    result={}
    for name,entry in models.items():
        eligible=entry.get('status')=='COMPLETE'
        result[name]={'status':entry.get('status','NOT_RUN'),'eligible_for_architecture_comparison':eligible and name not in ('B1-FROZEN-NO-COMM','T1'),'reference_only':name=='T1',
                      'origin':entry.get('origin'),'historical_structural_audit':entry.get('structural_parameter_audit','NOT_APPLICABLE'),'architecture':entry.get('architecture',{
                          'allocated_parameters':entry.get('allocated_parameters'),'active_trainable_parameters':entry.get('structural_parameter_audit',{}).get('forward_participating_parameters',entry.get('training_trainable_parameters')),
                          'state_dimensions':entry.get('persistent_state_dimension'),'state_bytes':entry.get('state_memory_bytes')}),
                      'training':entry.get('training','NOT_RUN'),'checkpoint_sha256':entry.get('checkpoint_sha256',entry.get('model_identity',{}).get('checkpoint_sha256','NOT_AVAILABLE')),
                      'validation':entry.get('validation','NOT_AVAILABLE') if eligible else 'NOT_AVAILABLE_INCOMPLETE',
                      'development_holdout':entry.get('development_holdout','NOT_AVAILABLE') if eligible else 'NOT_AVAILABLE_INCOMPLETE',
                      'test':'NOT_AVAILABLE','generation_metrics':entry.get('generation_metrics','NOT_AVAILABLE') if eligible else 'NOT_AVAILABLE_INCOMPLETE',
                      'performance':entry.get('performance','NOT_AVAILABLE') if eligible else 'NOT_AVAILABLE_INCOMPLETE'}
    return result


def communication_analysis(models):
    left=models.get('C1-BROADCAST',{});right=models.get('IC-ALL-SELF',{})
    if left.get('status')!='COMPLETE' or right.get('status')!='COMPLETE':return {'status':'NOT_EVALUATED_INCOMPLETE_PRIMARY_ARMS'}
    intervals={}
    for split in ('validation','development_holdout'):
        if split in left.get('paired_documents',{}) and split in right.get('paired_documents',{}):
            intervals[split]=paired({**left[split],'stories':left['paired_documents'][split]},{**right[split],'stories':right['paired_documents'][split]})
    return {'paired_story_bootstrap':intervals,'status':'SINGLE_SEED_DEVELOPMENT_OBSERVATION','primary':['C1-BROADCAST','IC-ALL-SELF'],
            'delta_nll_broadcast_minus_independent':{split:left[split]['nll']-right[split]['nll'] for split in ('validation','development_holdout')},
            'interpretation':'negative favors broadcast under matched event access/parameter structure; no seed variance or independent Test; no superiority claim',
            'historical_C1_vs_center_only':'NOT_A_FULL_CAPACITY_COMMUNICATION_CONTROL'}


def extension_proposal(models,plan):
    checks={}
    for name in ARMS:
        entry=models.get(name,{});diag=entry.get('training_diagnostics',{});points=[r for _,r in sorted(entry.get('progression',{}).items(),key=lambda p:int(p[0])) if r.get('status')=='OBSERVED']
        trend=diag.get('recent_loss_reduction')
        no_degradation=len(points)>=2 and points[-1]['validation_probe']['nll']<=points[0]['validation_probe']['nll']+plan['diagnostics']['extension_validation_deterioration_tolerance']
        checks[name]={'recent_training_loss_improves':isinstance(trend,(int,float)) and trend>plan['diagnostics']['extension_recent_loss_reduction'],
                      'validation_probe_not_materially_worse':no_degradation,'complete':entry.get('status')=='COMPLETE'}
    justified=all(all(v.values()) for v in checks.values())
    return {'status':'CONDITIONAL_PROPOSAL_REQUIRES_SEPARATE_FROZEN_PROTOCOL' if justified else 'NOT_PROPOSED_INSUFFICIENT_EVIDENCE',
            'checks':checks,'selected_arms':list(ARMS),'equal_extension_budget_required':True,
            'training_tokens_authorized':0,'new_training_identity':'NOT_CREATED','automatic_training':False,
            'limits':'subset validation trends and single-seed loss do not establish convergence; no 1M/10M continuation'}


def publish(root,out,report,experiment):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results
    destination=root/'results'/experiment/'publish';destination.mkdir(parents=True,exist_ok=True)
    write(destination/'decision.json',report)
    write(destination/'summary.json',{'status':report['status'],'identity':report['identity'],'batch':report['identity']['batch'],'scientific_status':'DEVELOPMENT_ONLY_H1_H5_NOT_EVALUATED',
                                    'human_review':report.get('human_evaluation',{'status':'PENDING'})})
    write(destination/'baseline-comparison.json',compare(report['models']))
    for filename,key in [('generation-metrics','generation_metrics'),('generation-samples','samples'),('performance','performance'),('training-curves','training_diagnostics')]:
        write(destination/(filename+'.json'),{name:entry.get(key,'NOT_AVAILABLE_HISTORICALLY_NOT_RECORDED') for name,entry in report['models'].items()})
    write(destination/'state-diagnostics.json',{name:state_observations(entry) for name,entry in report['models'].items()})
    write(destination/'data-audit.json',report.get('data_audit',{'status':'NOT_RUN'}))
    write(destination/'human-evaluation.json',report.get('human_evaluation',{'status':'PENDING','reviewers':0,'note':'No reviewers contacted and no scores invented'}))
    write(destination/'communication-analysis.json',communication_analysis(report['models']))
    write(destination/'quality-diagnosis.json',report.get('quality_diagnosis',{'status':'BATCH_C_PENDING'}))
    write(destination/'extension-proposal.json',report.get('extension_proposal',{'status':'NOT_PROPOSED','automatic_training':False}))
    sections=['# GC-001 Baseline Repair','',report['status'],'',
        '## A. Engineering','Fresh named arms; immutable historical GC references; strict checkpoint identity and verified return.',
        '## B. Baseline comparison','See baseline-comparison.json; incomplete arms excluded. Flat matching is active-parameter, not compute matching.',
        '## C. Communication','Primary comparison is C1-BROADCAST versus IC-ALL-SELF. Historical entry-only C1 and center-only NO-COMM are separate.',
        '## D. State architecture','Fixed 2048-number cellular/independent/flat states versus T1 bounded token context; KV disabled for reference.',
        '## E. Generation quality','Every fixed prompt/mode is retained. Low repetition/PPL alone is not coherence. Human review remains pending unless real scores are supplied.',
        '## F. Scientific conclusions','Engineering checks and numerical observations are separate from hypotheses. One seed/development-only data establish no architectural superiority. H1–H5 NOT_EVALUATED.']
    if report.get('error'):sections+=['','Stop reason: '+report['error']]
    (destination/'summary.md').write_text('\n\n'.join(sections)+'\n')
    if report['identity']['batch']=='c':
        import shutil
        shutil.copyfile(out/'blind-review.csv',destination/'blind-review.csv')
    artifacts=prepare_artifacts(root,experiment,EXPERIMENTS[experiment],None)
    branch='kaggle/'+experiment+'-'+report['identity']['source'][:12]
    push_results(root,artifacts,experiment,branch,'GITHUB_TOKEN','origin/main')
    write(out/'publication.json',{'status':'REMOTE_SHA_VERIFIED','branch':branch,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()})
    print(json.dumps({'batch':report['identity']['batch'],'status':report['status'],'publication':'REMOTE_SHA_VERIFIED','report':str(out/'decision.json')}))


def launch(root,base,out,device,batch):
    from scripts.research.publish_experiment_results import validate_origin,load_github_token
    if device!='cuda' or not torch.cuda.is_available():raise RuntimeError('Kaggle GPU required; no CPU training fallback')
    validate_origin(root);load_github_token('GITHUB_TOKEN')
    plan=json.loads((base/'protocol.json').read_text());gc=plan['gc_protocol'];core=base.parent
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    identity={'source':source,'protocol':sha256_file(base/'protocol.json'),'data':gc['training_identity']['data'],'batch':batch,
              'original_result_commit':plan['original_result_commit']}
    recovery_id={k:identity[k] for k in ('source','protocol','data','batch')}
    experiment='epclm-gc001-repair-'+batch;out.mkdir(parents=True,exist_ok=True);path=out/'decision.json'
    report={'format':'minicells.gc001-repair.v1','status':'INCOMPLETE','identity':identity,'protocol':plan,'models':{},'timings':{},
            'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'precision':'fp32','autocast':False,'tf32':False},
            'hypotheses':dict.fromkeys(['H1','H2','H3','H4','H5'],'NOT_EVALUATED'),'human_evaluation':{'status':'PENDING','reviewers':0}}
    def load_report():
        nonlocal report
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['identity']!=identity:raise BaselineIntegrityError('repair progress identity differs')
            report=saved
            if saved['status']=='BLOCKED_MODEL_ASSET_UNAVAILABLE':
                archived='failures/recovery-'+hashlib.sha256(json.dumps(saved,sort_keys=True).encode()).hexdigest()[:16]+'.json'
                write(out/archived,saved)
                report.setdefault('prior_recovery_failures',[]).append({'path':archived,'sha256':sha256_file(out/archived)})
                report['status']='INCOMPLETE';report.pop('error',None);write(path,report)
    def persist():write(path,report);bundle(out,recovery_id)
    load_report()
    if report['status']!='INCOMPLETE':
        if batch=='c' and (out/'private/human-key.json').exists():
            report['human_evaluation']=blind_review(out,report['models'],plan);persist()
        publish(root,out,report,experiment);return
    started=time.monotonic();deadline=started+plan['budget']['execution'];stage='RECOVERY';restored_terminal=False
    try:
        from repair_preflight import preflight
        report.setdefault('preflight',preflight(plan,full=False))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with timed_budget(plan['budget']['execution']):
            began=time.monotonic()
            with timed_budget(plan['budget']['recovery']):
                own=dependency(root,batch,identity,optional=True)
                if own is not None and own[0]['status']=='COMPLETE':
                    report=own[0];restored_terminal=True
                    if batch=='c':
                        raw=subprocess.check_output(['git','show',own[1]['commit']+':'+ARTIFACT+'batch-c/blind-review.csv'],cwd=root)
                        (out/'blind-review.csv').write_bytes(raw)
                else:
                    had_local=path.exists()
                    restored={'storage':'VERIFIED_LOCAL_SOURCE_REPORT'} if had_local else recover_namespace(root,out,out.name,recovery_id,min(deadline,began+plan['budget']['recovery']),plan['recovery']['max_versions'])
                    if own is not None and restored is None and not had_local and any(e.get('gradient_activity') or e.get('training',{}).get('tokens',0)>0 or e.get('checkpoint_sha256') for e in own[0].get('models',{}).values() if e.get('origin','').startswith('NEW_REPAIR')):
                        raise AssetUnavailable('published partial batch exists but private progress is unavailable; refusing retraining')
                    load_report();restored_terminal=report['status']!='INCOMPLETE'
                if not restored_terminal and batch in ('a','c'):
                    assets=out.with_name(out.name+'-assets')
                    recover_frozen_assets(root,assets,gc['training_identity'],gc['C1_checkpoint'],min(deadline,began+plan['budget']['recovery']),
                        preferred_versions=plan['recovery']['known_original_asset_versions'],extra_asset_hashes={'assets/train.json':None,'F0.pt':gc['F0_checkpoint']['sha256']})
                    c1,tokenizer,_=load_frozen_model(gc['training_identity'],gc['C1_checkpoint'],assets,core/'protocol.json',device)
                elif not restored_terminal:
                    assets=out.with_name(out.name+'-assets')
                    recover_frozen_assets(root,assets,gc['training_identity'],gc['C1_checkpoint'],min(deadline,began+plan['budget']['recovery']),preferred_versions=[142],extra_asset_hashes={'assets/train.json':None})
                    c1,tokenizer,_=load_frozen_model(gc['training_identity'],gc['C1_checkpoint'],assets,core/'protocol.json',device)
            report['timings']['recovery_seconds']=time.monotonic()-began
            if not restored_terminal:
                if batch=='a' and report['preflight']['status']!='PASS':report['preflight']=preflight(plan,full=True)
                stage='DATA_AND_PROVENANCE'
                history,provenance=public_result(root,plan['original_result_branch'],'artifacts/experiments/epclm-core-001/generative-core-001',commit=plan['original_result_commit'])
                if provenance['decision_sha256']!=plan['original_decision_sha256'] or history['status']!='COMPLETE' or history['identity']['diagnostic_source']!=plan['original_gc_source']:
                    raise BaselineIntegrityError('original GC identity differs')
                if sha256_file(core/'generative-core-protocol.json')!=plan['original_gc_protocol_sha256'] or json.loads((core/'generative-core-protocol.json').read_text())!=gc:
                    raise BaselineIntegrityError('original prompts/decoding/data protocol changed')
                report['original_result']=provenance
                groups,audit=audit_data(assets,tokenizer,gc)
                if audit!=history['data_audit']:raise BaselineIntegrityError('original GC data audit differs')
                report['data_audit']=audit;tokenizer_sha=sha256_file(assets/'assets/tokenizer.json')
                if batch=='a':
                    for name,value in history['models'].items():
                        if name not in report['models']:report['models'][name]={**value,'origin':'HISTORICAL_GC001'}
                    for name,e in report['models'].items():
                        e['historical_recorded_parameter_budget']={'allocated':e.get('allocated_parameters'),'recorded_trainable':e.get('training_trainable_parameters')}
                        if name=='F0':e['structural_parameter_audit']={'forward_participating_parameters':1312896,'inactive_allocated_parameters':180224,'basis':'unchanged Core F0 never uses left/center/right/update; post-hoc code audit'}
                        if name=='NO-COMM':e['structural_parameter_audit']={'declared_trainable':934016,'forward_participating_parameters':688256,'basis':'center-only zero non-entry states make 245760 pool coefficients inactive; post-hoc code audit'}
                    arms=ARMS
                else:
                    a=dependency(root,'a',identity)
                    if a[0]['status']!='COMPLETE':raise AssetUnavailable('Batch A is not complete; resume A instead of repeating training')
                    report['batch_a_result']=a[1]
                    for name,value in a[0]['models'].items():report['models'].setdefault(name,value)
                    arms=('T1',) if batch=='b' else ()
                    if batch=='c':
                        b=dependency(root,'b',identity,optional=True)
                        if b:
                            report['batch_b_result']=b[1]
                            if b[0]['models'].get('T1',{}).get('status')=='COMPLETE':report['models']['T1']=b[0]['models']['T1']
                            else:report['models']['T1']={**b[0]['models'].get('T1',{'status':'NOT_RUN'}),'origin':'REFERENCE_INCOMPLETE_EXCLUDED'}
                        else:report['models']['T1']={'status':'NOT_RUN','origin':'REFERENCE_NOT_RUN'}
                persist()
                for arm in arms:
                    check_deadline(deadline);entry=report['models'].setdefault(arm,{'status':'INCOMPLETE','origin':'NEW_REPAIR_REFERENCE' if arm=='T1' else 'NEW_REPAIR_CONTROL','model_id':arm})
                    if entry.get('status')=='COMPLETE':continue
                    torch.manual_seed(plan['training']['seed'])
                    model=(TransformerReference() if arm=='T1' else build(arm,plan['parameter_matching']['rank'])).to(device=device,dtype=torch.float32)
                    if arm=='T1':
                        stats={'allocated_parameters':sum(p.numel() for p in model.parameters()),'active_trainable_parameters':sum(p.numel() for p in model.parameters()),
                               'state_dimensions':129,'state_bytes':516,'estimated_forward_linear_flops':'NOT_MEASURED','reference_config':model.reference_config,'context':128,'kv_cache':False}
                    else:stats=architecture(model)
                    if arm=='F0-PM-BROADCAST' and abs(stats['active_trainable_parameters']/966784-1)>.01:raise BaselineIntegrityError('flat active-parameter matching failed')
                    if arm in ARMS:
                        entry.setdefault('initial_weights_sha256',weight_hash(model))
                        embedding_sha=hashlib.sha256(model.embedding.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        entry.setdefault('initial_embedding_sha256',embedding_sha)
                        reference=report['models'].get('C1-BROADCAST',{})
                        if arm!='C1-BROADCAST' and reference.get('initial_embedding_sha256')!=embedding_sha:raise BaselineIntegrityError('new-arm event embeddings do not share initial values')
                        if arm=='IC-ALL-SELF' and reference.get('initial_weights_sha256')!=weight_hash(model):raise BaselineIntegrityError('companion cellular controls do not share initial parameters')
                    entry['architecture']=stats
                    training_identity={**identity,'model_id':arm,'model':asdict(model.config),'architecture':stats,'precision':'fp32','seed':173018021,'target_tokens':250000}
                    entry['training_identity']=training_identity
                    stage='TRAINING';learning(model,arm,groups['train'],groups,entry,out/(arm+'.pt'),training_identity,plan,deadline,persist)
                    stage='EVALUATION';evaluate_repair(model,tokenizer,tokenizer_sha,groups,entry,plan,out,persist,deadline)
                if batch=='c':
                    stage='DIAGNOSIS_ONLY';historical_diagnostics(root,assets,c1,tokenizer,groups,history,report,plan,deadline,device,persist)
                    report['quality_diagnosis']=quality_findings(report['models'],plan)
                    report['extension_proposal']=extension_proposal(report['models'],plan)
                    report['human_evaluation']=blind_review(out,report['models'],plan)
                report['scientific_reporting']={'engineering_verified':[name for name,e in report['models'].items() if e.get('status')=='COMPLETE' and e.get('origin','').startswith('NEW_REPAIR')],
                    'empirically_observed':communication_analysis(report['models']),
                    'preliminary_hypotheses':'DESCRIPTIVE_SINGLE_SEED_ONLY_NO_FORMAL_SUPPORT',
                    'not_established':['architectural superiority','independent-Test generalization','semantic memory or reasoning','human coherence before supplied reviews']}
                report.update(status='COMPLETE');report.pop('error',None)
    except AssetUnavailable as error:report.update(status='BLOCKED_MODEL_ASSET_UNAVAILABLE',error=str(error),failed_stage=stage)
    except BaselineIntegrityError as error:report.update(status='FAILED_INTEGRITY',error=str(error),failed_stage=stage)
    except TimeoutError as error:report.update(status='INCOMPLETE',error=str(error),failed_stage=stage)
    except (FloatingPointError,RuntimeError,ValueError,OSError,KeyError) as error:report.update(status='FAILED_NUMERICAL_OR_EXECUTION',error=str(error),failed_stage=stage)
    finally:
        elapsed=time.monotonic()-started
        if restored_terminal:report['timings']['publication_recovery_seconds']=elapsed
        else:report['timings']['execution_seconds']=report['timings'].get('execution_seconds',0.)+elapsed
        persist()
    if batch=='c' and not (out/'blind-review.csv').exists():report['human_evaluation']=blind_review(out,report['models'],plan);persist()
    publish(root,out,report,experiment)


def historical_diagnostics(root,assets,c1,tokenizer,groups,history,report,plan,deadline,device,persist):
    gc=plan['gc_protocol'];protocol=json.loads((Path(__file__).parent.parent/'protocol.json').read_text())
    for name,model,file in [('C1',c1,assets/'C1.pt'),('F0',load_flat(assets,gc,protocol,device),assets/'F0.pt')]:
        entry=report['models'][name]
        if 'teacher_diagnostics' not in entry:entry['teacher_diagnostics']=teacher_diagnostics(model,groups['validation'],tokenizer,deadline)
        saved=load_checkpoint(file);entry['training_diagnostics']=training_diagnostics(saved['curve'],'NOT_AVAILABLE_HISTORICALLY_NOT_RECORDED');persist()
    b1=report['models']['B1-FROZEN-NO-COMM']
    if 'teacher_diagnostics' not in b1:
        model=NoCommunicationCore(c1.config).to(device);model.load_state_dict(c1.state_dict());freeze(model)
        b1['teacher_diagnostics']=teacher_diagnostics(model,groups['validation'],tokenizer,deadline)
        b1['training_diagnostics']={'status':'FROZEN_INFERENCE_ABLATION_NOT_INDEPENDENTLY_TRAINED'};persist()
    entry=report['models']['NO-COMM'];old=history['models']['NO-COMM']['model_identity'];cache=assets/'old-gc'
    if 'teacher_diagnostics' not in entry:
        try:
            with timed_budget(min(120,max(1,deadline-time.monotonic()))):
                provenance=recover_namespace(root,cache,'epclm-gc001',{'source':plan['original_gc_source'],'protocol':plan['original_gc_protocol_sha256'],'data':gc['training_identity']['data']},
            min(deadline,time.monotonic()+120),selected={'NO-COMM.pt':old['checkpoint_sha256']})
        except (TimeoutError,AssetUnavailable) as error:
            entry['historical_diagnostic_limit']='PRIVATE_MODEL_UNAVAILABLE: '+str(error);persist();return
        if provenance is None:
            entry['historical_diagnostic_limit']='PRIVATE_MODEL_ASSET_NOT_AVAILABLE_NO_RETRAINING'
            return
        model=NoCommunicationCore(Config(**old['training_identity']['model'])).to(device)
        saved=checkpoint_model(model,cache/'NO-COMM.pt',old['training_identity'])
        if weight_hash(model)!=old['weights_sha256']:raise BaselineIntegrityError('historical NO-COMM weights differ')
        freeze(model);entry['teacher_diagnostics']=teacher_diagnostics(model,groups['validation'],tokenizer,deadline)
        entry['training_diagnostics']=training_diagnostics(saved['curve'],'NOT_AVAILABLE_HISTORICALLY_NOT_RECORDED');persist()
