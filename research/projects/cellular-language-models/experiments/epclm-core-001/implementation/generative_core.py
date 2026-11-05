"""One-launch GC-001: audit, frozen pilot/generation, controlled missing baseline, return."""
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import torch
from frozen_model import (AssetUnavailable,BaselineIntegrityError,recover_frozen_assets,load_frozen_model,check_deadline)
from communication_propagation import timed_budget
from mechanism_diagnostic import write
from model import Core,config_from_protocol
from runtime import weight_hash
from trainer import train,load_checkpoint
from storage import bundle
from generative_baselines import NoCommunicationCore,TransformerReference,freeze
from generation_evaluation import audit_data,pilot,evaluate_model
from native_clm_runtime import sha256_file


def training_summary(value):
    curve=value.get('curve',[])
    result={k:v for k,v in value.items() if k not in ('curve','overflow_diagnostics')}
    if curve:result.update(first_loss=curve[0]['nll'],last_loss=curve[-1]['nll'])
    result['overflow_count']=len(value.get('overflow_diagnostics',[]))
    return result


def trained_model(model,documents,identity,path,plan,deadline,on_checkpoint):
    if path.exists():
        saved=load_checkpoint(path)
        if saved['identity']!=identity or saved['precision']!='fp32':raise BaselineIntegrityError('baseline checkpoint identity/precision differs')
        if saved['tokens']==plan['tokens_per_trained_baseline']:
            model.load_state_dict(saved['model'],strict=True)
            if not all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('completed baseline has non-finite/non-FP32 parameters')
            return {'status':'COMPLETE','tokens':saved['tokens'],'updates':saved['updates'],'curve':saved['curve'],
                    'elapsed_seconds':'NOT_AVAILABLE','tokens_per_second':'NOT_AVAILABLE','reused_completed_checkpoint':True}
    return train(model,documents,identity,path,plan['tokens_per_trained_baseline'],deadline,plan['seed'],
                 batch=plan['batch'],unroll=plan['tbptt'],precision='fp32',on_checkpoint=on_checkpoint)


def load_flat(assets,plan,training_protocol,device):
    pinned=plan['F0_checkpoint'];path=assets/pinned['path']
    if sha256_file(path)!=pinned['sha256']:raise BaselineIntegrityError('F0 checkpoint hash differs')
    config=config_from_protocol(training_protocol,'F0');saved=load_checkpoint(path)
    expected={**plan['training_identity'],'model':asdict(config),'target_tokens':250000}
    if saved['identity']!=expected or saved['tokens']!=250000 or saved['updates']!=977 or saved['precision']!='fp32':
        raise BaselineIntegrityError('F0 trained identity differs')
    model=Core(config).to(device=device,dtype=torch.float32);model.load_state_dict(saved['model'],strict=True)
    if weight_hash(model)!=pinned['weights_sha256'] or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise BaselineIntegrityError('F0 weights differ/non-finite')
    return freeze(model)


def primary_result(root,source,protocol_sha):
    branch='kaggle/epclm-gc001-'+source[:12]
    result=subprocess.run(['git','fetch','--depth=1','origin','refs/heads/'+branch],cwd=root,capture_output=True,text=True,timeout=60)
    if result.returncode:raise AssetUnavailable('Primary GC-001 result lookup failed (network or missing branch). Complete the primary batch first, or retry acquisition; completed primary work is never retrained.')
    commit=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=root,text=True).strip()
    value=json.loads(subprocess.check_output(['git','show',commit+':artifacts/experiments/epclm-core-001/generative-core-001/decision.json'],cwd=root,text=True))
    if (value['identity']['diagnostic_source']!=source or value['identity']['protocol_sha256']!=protocol_sha
            or value['status']!='COMPLETE' or value.get('gpu_pilot',{}).get('status')!='PASS'):
        raise BaselineIntegrityError('primary GC-001 result is not complete with matching source/protocol')
    return value,{'branch':branch,'commit':commit,'reuse':'completed primary results, no repeated C1 GPU pilot'}


def publish(root,output,report,experiment):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results
    publication=root/'results'/experiment/'publish';publication.mkdir(parents=True,exist_ok=True)
    write(publication/'decision.json',report)
    write(publication/'data-audit.json',report.get('data_audit',{'status':'NOT_RUN'}))
    write(publication/'generation-metrics.json',{k:v.get('generation_metrics','NOT_RUN') for k,v in report['models'].items()})
    write(publication/'performance.json',{k:v.get('performance','NOT_RUN') for k,v in report['models'].items()})
    comparison={}
    for name in ['C1','B1-FROZEN-NO-COMM','NO-COMM','F0','T1']:
        value=report['models'].get(name,{})
        comparison[name]={k:value.get(k,'NOT_RUN') for k in ['status','allocated_parameters','training_trainable_parameters','persistent_state_dimension',
            'state_memory_bytes','validation','development_holdout','test','generation_metrics','training','comparison_scope','model_identity','checkpoint_bytes']}
    for name,value in comparison.items():
        value['eligible_for_comparison']=value['status']=='COMPLETE'
        value['architecture_control']=name not in ('B1-FROZEN-NO-COMM','T1')
        if not value['eligible_for_comparison']:
            for key in ('validation','development_holdout','generation_metrics'):value[key]='NOT_AVAILABLE_INCOMPLETE'
    write(publication/'baseline-comparison.json',comparison)
    samples=['# GC-001 complete generation samples','All fixed prompts and both decoding modes are retained; no best-sample selection.']
    for name,value in report['models'].items():
        for row in value.get('samples',[]):
            samples+=['',f"## {name} / {row['mode']['name']} / prompt {row['prompt_index']}",'',row['prompt'],'',row['text'],'',
                      f"Stop: {row['stop_reason']}; generated tokens: {row['generated_tokens']}; seed: {row['seed']}."]
    samples+=['','Human review is pending. Automated repetition/diversity metrics do not establish coherence.']
    (publication/'generation-samples.md').write_text('\n'.join(samples)+'\n')
    lines=['# EPCLM Generative Core-001','',report['status'],'',
           '| Model | Status | Parameters | State bytes | Train tokens | Val PPL | Dev holdout PPL | Test PPL |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for name,value in comparison.items():
        def ppl(key):return value[key].get('ppl','NOT_AVAILABLE') if isinstance(value[key],dict) else 'NOT_RUN'
        tokens=value['training'].get('tokens','NOT_RUN') if isinstance(value['training'],dict) else 'NOT_RUN'
        lines.append(f"| {name} | {value['status']} | {value['allocated_parameters']} | {value['state_memory_bytes']} | {tokens} | {ppl('validation')} | {ppl('development_holdout')} | NOT_AVAILABLE |")
    lines+=['','No independent Test is present in the pinned assets. The development holdout has potential tokenizer exposure.',
            'One training seed; token/state matching does not imply parameter/compute matching. No architecture superiority, semantic-memory, chat or reasoning claim. H1–H5 NOT_EVALUATED.']
    if report.get('error'):lines+=['','Stop reason: '+report['error']]
    (publication/'summary.md').write_text('\n'.join(lines)+'\n')
    reports=output/'reports';reports.mkdir(exist_ok=True)
    import shutil
    for name in EXPERIMENTS[experiment].files:shutil.copyfile(publication/name,reports/name)
    destination=prepare_artifacts(root,experiment,EXPERIMENTS[experiment],None)
    branch='kaggle/'+experiment+'-'+report['identity']['diagnostic_source'][:12]
    push_results(root,destination,experiment,branch,'GITHUB_TOKEN','origin/main')
    receipt={'status':'REMOTE_SHA_VERIFIED','result_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'branch':branch}
    write(output/'publication.json',receipt)
    print(json.dumps({'mode':'GENERATIVE_CORE','status':report['status'],'publication':receipt,'report':str(output/'decision.json')}))


def launch(root,base,output,device='cuda',transformer_only=False):
    from scripts.research.publish_experiment_results import validate_origin,load_github_token
    if device!='cuda' or not torch.cuda.is_available():raise RuntimeError('enable Kaggle GPU; no CPU language training fallback')
    validate_origin(root);load_github_token('GITHUB_TOKEN')
    plan=json.loads((base/'generative-core-protocol.json').read_text());protocol_sha=sha256_file(base/'generative-core-protocol.json')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    experiment='epclm-gc001-transformer' if transformer_only else 'epclm-gc001'
    identity={'experiment':experiment,'diagnostic_source':source,'protocol_sha256':protocol_sha,'protocol':plan,
              'training_source':plan['training_identity']['source'],'training_data_manifest':plan['training_identity']['data'],
              'formal_confirmation':False,'training_precision':'fp32'}
    output.mkdir(parents=True,exist_ok=True);path=output/'decision.json'
    report={'format':'minicells.epclm.gc001.v1','status':'INCOMPLETE','identity':identity,'models':{},'timings':{},
            'hypotheses':dict.fromkeys(['H1','H2','H3','H4','H5'],'NOT_EVALUATED'),
            'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()},
            'language_evaluation_status':'DEVELOPMENT_ONLY_INDEPENDENT_TEST_NOT_AVAILABLE',
            'human_coherence_review':'NOT_EVALUATED_ALL_SAMPLES_RETAINED'}
    def load_progress():
        nonlocal report
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['identity']!=identity:raise BaselineIntegrityError('saved GC source/protocol identity differs')
            report=saved
            if report['status']=='BLOCKED_MODEL_ASSET_UNAVAILABLE':
                name='failures/recovery-'+hashlib.sha256(json.dumps(saved,sort_keys=True).encode()).hexdigest()[:16]+'.json'
                write(output/name,saved)
                report.setdefault('prior_recovery_failures',[]).append({'path':name,'sha256':sha256_file(output/name)})
                report.update(status='INCOMPLETE');report.pop('error',None)
                write(path,report)
                return True
        return False
    load_progress()
    if report['status']!='INCOMPLETE':publish(root,output,report,experiment);return
    started=time.monotonic();deadline=started+plan['budget']['execution'];stage='RECOVERY';terminal=None
    recovery_identity={'source':source,'protocol':protocol_sha,'data':plan['training_identity']['data']}
    def persist():write(path,report);bundle(output,recovery_identity)
    try:
        # This is a forward-only synthetic CPU gate; no trainer or optimizer is called.
        from generative_preflight import check_generation
        report['functional_preflight']=check_generation()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assets=output.with_name(output.name+'-frozen-assets');began=time.monotonic()
        extras={'assets/train.json':None,**({} if transformer_only else {'F0.pt':plan['F0_checkpoint']['sha256']})}
        try:
            with timed_budget(plan['budget']['recovery']):
                terminal=recover_frozen_assets(root,assets,plan['training_identity'],plan['C1_checkpoint'],
                    min(deadline,time.monotonic()+plan['budget']['recovery']),progress_output=output,progress_identity=recovery_identity,
                    preferred_versions=plan['preferred_asset_versions'],
                    extra_asset_hashes=extras,allow_training_progress=True)
                retry_assets=load_progress()
                if terminal=='DIAGNOSTIC_TERMINAL' and retry_assets:
                    terminal=recover_frozen_assets(root,assets,plan['training_identity'],plan['C1_checkpoint'],
                        min(deadline,began+plan['budget']['recovery']),preferred_versions=plan['preferred_asset_versions'],extra_asset_hashes=extras)
                if terminal!='DIAGNOSTIC_TERMINAL':
                    c1,tokenizer,_=load_frozen_model(plan['training_identity'],plan['C1_checkpoint'],assets,base/'protocol.json',device)
        finally:report['timings']['recovery_seconds']=time.monotonic()-began
        if terminal!='DIAGNOSTIC_TERMINAL':
            stage='DATA_AUDIT';groups,audit=audit_data(assets,tokenizer,plan);report['data_audit']=audit
            tokenizer_sha=sha256_file(assets/'assets/tokenizer.json');training_protocol=json.loads((base/'protocol.json').read_text())
            c1_identity={**plan['C1_checkpoint'],'weights_sha256':weight_hash(c1),'transition':'C1_NORMAL'}
            stage='GENERATION';persist()
            with timed_budget(max(1,deadline-time.monotonic())):
                if transformer_only:
                    previous,provenance=primary_result(root,source,protocol_sha)
                    report['gpu_pilot']=previous['gpu_pilot'];report['primary_result']=provenance
                    for key,value in previous['models'].items():report['models'].setdefault(key,value)
                    arms=['T1']
                else:
                    if 'gpu_pilot' not in report:
                        report['gpu_pilot']=pilot(c1,tokenizer,c1_identity,tokenizer_sha,plan,output,deadline);persist()
                    arms=plan['baseline_order']
                if report['gpu_pilot']['status']!='PASS':raise BaselineIntegrityError('generation GPU pilot did not pass')
                for arm in arms:
                    check_deadline(deadline);entry=report['models'].setdefault(arm,{'status':'INCOMPLETE'})
                    if entry['status']=='COMPLETE':continue
                    if arm=='C1':model=c1;model_identity=c1_identity;checkpoint_path=assets/'C1.pt'
                    elif arm=='F0':
                        model=load_flat(assets,plan,training_protocol,device);model_identity={**plan['F0_checkpoint'],'transition':'FLAT_RECURRENT'};checkpoint_path=assets/'F0.pt'
                    else:
                        torch.manual_seed(plan['seed'])
                        config=config_from_protocol(training_protocol,'C1')
                        model=TransformerReference().to(device=device,dtype=torch.float32) if arm=='T1' else NoCommunicationCore(replace(config,arm='C0' if arm=='NO-COMM' else 'C1')).to(device=device,dtype=torch.float32)
                        if arm=='B1-FROZEN-NO-COMM':
                            model.load_state_dict(c1.state_dict());model_identity={**plan['C1_checkpoint'],'transition':'FROZEN_CENTER_ONLY_ABLATION'};checkpoint_path=assets/'C1.pt'
                        else:
                            stage='TRAINING'
                            checkpoint_path=output/(arm+'.pt')
                            arm_identity={'source':source,'protocol':protocol_sha,'data':plan['training_identity']['data'],
                                          'model':asdict(model.config),'model_id':arm,'transition':'T1_CONTEXT_REPLAY' if arm=='T1' else 'CENTER_ONLY',
                                          'target_tokens':250000,'seed':plan['seed']}
                            if arm=='T1':arm_identity['reference_config']=model.reference_config
                            entry['training_identity']=arm_identity;persist()
                            result=trained_model(model,groups['train'],arm_identity,checkpoint_path,plan,deadline,persist)
                            old=entry.get('training',{})
                            if not result.get('reused_completed_checkpoint') or old.get('status')!='COMPLETE':
                                summary=training_summary(result)
                                if isinstance(old.get('elapsed_seconds'),(int,float)) and isinstance(summary.get('elapsed_seconds'),(int,float)):
                                    summary['elapsed_seconds']+=old['elapsed_seconds']
                                    summary['tokens_per_second']=summary['tokens']/max(summary['elapsed_seconds'],1e-9)
                                entry['training']=summary
                            persist()
                            if result['status']!='COMPLETE':raise TimeoutError('baseline training incomplete; later arms are not started')
                            curve=result['curve'];gate=plan['baseline_convergence_gate']
                            first=curve[:gate['initial_updates']];last=curve[-gate['final_updates']:]
                            if sum(r['nll'] for r in first)/len(first)-sum(r['nll'] for r in last)/len(last)<gate['minimum_mean_loss_reduction']:
                                raise BaselineIntegrityError('baseline failed predefined loss-reduction gate; no later training started')
                            model_identity={'checkpoint_sha256':sha256_file(checkpoint_path),'weights_sha256':weight_hash(model),
                                            'transition':arm_identity['transition'],'training_identity':arm_identity}
                    entry['architecture']=getattr(model,'reference_config',asdict(model.config))
                    entry['model_identity']=model_identity
                    entry['checkpoint_bytes']=checkpoint_path.stat().st_size
                    entry['training_trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad) if arm in ('NO-COMM','T1') else ('NOT_AVAILABLE_FROZEN_ABLATION' if arm=='B1-FROZEN-NO-COMM' else sum(p.numel() for p in model.parameters()))
                    entry.setdefault('training',{'status':'REUSED_COMPLETE' if arm!='B1-FROZEN-NO-COMM' else 'FROZEN_INFERENCE_ABLATION_NOT_TRAINED',
                        'tokens':250000,'new_training_tokens':0,'source':plan['training_identity']['source'],
                        'elapsed_seconds':571.241171671 if arm=='C1' else 348.574707358 if arm=='F0' else 'NOT_APPLICABLE'})
                    entry['comparison_scope']={'token_matched':arm!='B1-FROZEN-NO-COMM','state_matched':arm!='T1',
                        'allocated_parameter_matched':arm in ('C1','B1-FROZEN-NO-COMM','NO-COMM'),'active_parameter_matched':arm=='C1','compute_matched':False,
                        'note':'NO-COMM allocated parameters include inactive left/right terms; zero-init bias-free disconnected non-entry cells stay zero, so allocated state is not used capacity. F0 has 1.49M vs C1 0.967M; T1 is context-based reference only',
                        'tbptt':64 if arm!='T1' else '64-step loss accumulation; bounded token context recomputed, no recurrent activation carry',
                        'state_vs_gradient':'latent state retained across TBPTT, gradients detached at truncation; every document reset',
                        'kv_cache_bytes':0,'context_window':128 if arm=='T1' else 'NOT_APPLICABLE',
                        'no_history_replay_contract':'NOT_APPLICABLE_BOUNDED_REFERENCE_CONTEXT' if arm=='T1' else 'PASS_STATE_ONLY'}
                    stage='EVALUATION'
                    evaluate_model(model,tokenizer,model_identity,tokenizer_sha,groups,plan,entry,output,persist,deadline)
                    if arm=='C1' and abs(entry['validation']['nll']-5.814626392302277)>1e-4:raise BaselineIntegrityError('original C1 validation baseline not reproduced')
                    if arm=='F0' and abs(entry['validation']['nll']-5.830023798407524)>1e-4:raise BaselineIntegrityError('original F0 validation baseline not reproduced')
                    persist()
                report['status']='COMPLETE';report.pop('error',None)
                report['engineering_runtime_status']='PASS'
                report['architecture_conclusions']='DEVELOPMENT_COMPARISON_ONLY_SINGLE_SEED_PARAMETER_AND_COMPUTE_DIFFERENCES_REPORTED'
                report['decision_answers']={'closed_loop':'PASS','persistent_state_history_carrier':'VERIFIED_BY_RUNTIME_CONTRACT_NOT_SEMANTIC_SUFFICIENCY',
                    'coherence':'HUMAN_REVIEW_REQUIRED','repeatable_communication_benefit':'NOT_CONFIRMED_SINGLE_TRAINING_SEED',
                    'cellular_over_flat':'NOT_CONFIRMED','cellular_over_transformer':'NOT_CONFIRMED',
                    'confounders':['different parameter budgets','different compute','tokenizer development exposure','single training seed','no independent Test']}
    except AssetUnavailable as error:report.update(status='BLOCKED_MODEL_ASSET_UNAVAILABLE',error=str(error),failed_stage=stage)
    except BaselineIntegrityError as error:report.update(status='FAILED_INTEGRITY',error=str(error),failed_stage=stage)
    except TimeoutError as error:report.update(status='BLOCKED_MODEL_ASSET_UNAVAILABLE' if stage=='RECOVERY' else 'INCOMPLETE',error=str(error),failed_stage=stage)
    except (FloatingPointError,RuntimeError,ValueError,OSError,KeyError) as error:
        report.update(status='FAILED_NUMERICAL_CHECK' if stage not in ('RECOVERY','DATA_AUDIT') else 'BLOCKED_MODEL_ASSET_UNAVAILABLE',error=str(error),failed_stage=stage)
    finally:
        if stage=='RECOVERY' and path.exists():
            saved=json.loads(path.read_text())
            if saved.get('identity')==identity:
                for key in ('models','gpu_pilot','data_audit','primary_result'):
                    if key in saved:report.setdefault(key,saved[key])
                report['models']=saved.get('models',report['models'])
        elapsed=time.monotonic()-started
        if terminal=='DIAGNOSTIC_TERMINAL':report['timings']['publication_recovery_seconds']=elapsed
        else:report['timings']['execution_seconds']=report['timings'].get('execution_seconds',0.)+elapsed
        persist()
    publish(root,output,report,experiment)
