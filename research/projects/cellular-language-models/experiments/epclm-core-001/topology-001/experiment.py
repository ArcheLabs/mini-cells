"""One sequential Kaggle development screen. No model recovery or intermediate checkpoints."""
import argparse
import copy
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
BASE=Path(__file__).resolve().parent;CORE=BASE.parent
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
sys.path[:0]=[str(BASE),str(CORE/'gc001-baseline-repair'),str(CORE/'implementation'),
             str(CORE/'gc002-fair-architecture'),str(CORE.parent/'native-clm-v0/implementation'),str(ROOT),str(ROOT/'shared/python')]
import torch
from topology import TopologyCore,graph,ARMS
from training import train,evaluate
from diagnostics import propagation,inference,compute,generate
from report import comparisons,screen
from t1_fresh import load_assets
from repair_io import timed_budget

EXPERIMENT='epclm-core-001-topology-001'
ARTIFACT='artifacts/experiments/epclm-core-001/topology-001'
FILES=('protocol.json','topology-adjacency.json','training-metrics.json','validation-results.json','holdout-results.json',
       'paired-comparison.json','propagation-diagnostics.json','state-diversity.json','compute-profile.json',
       'inference-profile.json','memory-profile.json','generation-metrics.json','decision.json','summary.zh-CN.md')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def save_reports(out,protocol,arms,decision):
    write(out/'protocol.json',protocol);write(out/'topology-adjacency.json',{n:graph(n) for n in ARMS})
    mapping={'training-metrics.json':'training_metrics','validation-results.json':'validation','holdout-results.json':'holdout',
             'propagation-diagnostics.json':'propagation','state-diversity.json':'diversity','compute-profile.json':'compute',
             'inference-profile.json':'inference','memory-profile.json':'memory','generation-metrics.json':'generation'}
    for filename,key in mapping.items():
        write(out/filename,{n:({'status':entry.get('training_status','NOT_STARTED'),'prediction_tokens':entry.get('prediction_tokens',0),
            'optimizer_updates':entry.get('optimizer_updates',0),'metrics':entry.get(key,[]),'validation_progression':entry.get('validation_progression',{}),
            'training_compute_seconds':entry.get('training_compute_seconds'),'training_wall_seconds':entry.get('training_wall_seconds'),
            'tokens_per_second':entry.get('tokens_per_second')} if key=='training_metrics' else entry.get(key,{'status':'NOT_STARTED'})) for n,entry in arms.items()})
    write(out/'decision.json',{**decision,'arms':{n:{k:v for k,v in entry.items() if k not in ('training_metrics','validation_progression','validation','holdout','propagation','diversity','compute','inference','memory','generation')} for n,entry in arms.items()}})


def previous_reports(out,source,protocol):
    """Reuse completed report records only, never saved model/optimizer/state assets."""
    record=out/'decision.json'
    if not record.exists():return {}
    decision=json.loads(record.read_text())
    if decision.get('source_commit')!=source or json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('existing output belongs to another source/protocol; choose a new output directory')
    arms=decision.get('arms',{})
    mapping={'training_metrics':'training-metrics.json','validation':'validation-results.json','holdout':'holdout-results.json',
             'propagation':'propagation-diagnostics.json','diversity':'state-diversity.json','compute':'compute-profile.json',
             'inference':'inference-profile.json','memory':'memory-profile.json','generation':'generation-metrics.json'}
    for key,filename in mapping.items():
        path=out/filename
        if not path.exists():continue
        for name,value in json.loads(path.read_text()).items():
            if name not in arms:continue
            if key=='training_metrics':
                arms[name]['training_metrics']=value['metrics'];arms[name]['validation_progression']=value['validation_progression']
                for field in ('training_compute_seconds','training_wall_seconds','tokens_per_second'):arms[name][field]=value.get(field)
            else:arms[name][key]=value
    return arms


def import_published_reports(out,source,branch):
    """Reuse existing result-return channel across Kaggle sessions, with no weight download."""
    if (out/'decision.json').exists():return
    lookup=subprocess.run(['git','ls-remote','--heads','origin','refs/heads/'+branch],cwd=ROOT,capture_output=True,text=True,timeout=30)
    if lookup.returncode:raise RuntimeError('cannot determine whether completed arm reports already exist; GitHub lookup failed')
    if not lookup.stdout.strip():return
    subprocess.run(['git','fetch','--depth=1','origin',branch],cwd=ROOT,check=True,capture_output=True,timeout=30)
    ref=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=ROOT,text=True).strip()
    decision=json.loads(subprocess.check_output(['git','show',ref+':'+ARTIFACT+'/decision.json'],cwd=ROOT,text=True))
    if decision['source_commit']!=source:raise ValueError('published source differs')
    for name in FILES:
        raw=subprocess.check_output(['git','show',ref+':'+ARTIFACT+'/'+name],cwd=ROOT)
        (out/name).write_bytes(raw)


def publish(out,branch):
    import shutil
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results
    destination=ROOT/'results'/EXPERIMENT/'publish';destination.mkdir(parents=True,exist_ok=True)
    for name in FILES:shutil.copyfile(out/name,destination/name)
    artifact=prepare_artifacts(ROOT,EXPERIMENT,EXPERIMENTS[EXPERIMENT],None)
    push_results(ROOT,artifact,EXPERIMENT,branch,'GITHUB_TOKEN','origin/main')


def launch(out):
    from scripts.research.publish_experiment_results import validate_origin,load_github_token
    protocol=json.loads((BASE/'protocol.json').read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run_id='topology-001-'+source[:12]+'-'+uuid.uuid4().hex[:8];branch='kaggle/'+EXPERIMENT+'-'+source[:12]
    if (out/'decision.json').exists():
        prior=json.loads((out/'decision.json').read_text())
        if prior.get('source_commit')!=source:out=out.with_name(out.name+'-'+run_id[-8:])
    out.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+protocol['budget']['execution_seconds']
    decision={'format':'minicells.epclm.topology-001.v1','experiment':protocol['experiment'],'run_id':run_id,'source_commit':source,
              'status':'INCOMPLETE','training_status':'NOT_STARTED','evaluation_status':'NOT_STARTED','propagation_status':'NOT_STARTED',
              'compute_status':'NOT_STARTED','publication_status':'PENDING','research_review_status':'PENDING',
              'scientific_confirmation':False,'independent_test_accessed':False,'result_branch':branch}
    arms={}
    def persist():save_reports(out,protocol,arms,decision)
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_name()!='Tesla T4':raise RuntimeError('requires Kaggle Tesla T4')
        validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        import_published_reports(out,source,branch);arms.update(previous_reports(out,source,protocol))
        if not all(arms.get(n,{}).get('training_status')=='COMPLETE' for n in ARMS):
            with timed_budget(max(1,deadline-time.monotonic())):
                assets=out/('inputs-'+uuid.uuid4().hex[:8]);assets.mkdir();tokenizer,groups=load_assets(assets,deadline)
            for name,count,total in (('validation',16,3322),('development_holdout',205,47403)):
                if len(groups[name])!=count or sum(len(d['tokens'])-1 for d in groups[name])!=total:raise ValueError(name+' split differs from protocol')
            torch.manual_seed(protocol['seed']);template=TopologyCore('R1');initial=copy.deepcopy(template.state_dict());del template
            if sum(p.numel() for p in initial.values())!=protocol['architecture']['parameters']:raise ValueError('parameter count differs')
            prompts=json.loads((CORE/'gc001-baseline-repair/protocol.json').read_text())['gc_protocol']['prompts'][:8]
        decision['environment']={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'precision':'fp32','tf32':False}
        for name in protocol['arms']:
            prior=arms.get(name,{})
            if prior.get('training_status')=='COMPLETE':
                print({'arm':name,'status':'REUSING_COMPLETED_REPORT_NO_TRAINING_OR_WEIGHT_LOAD'},flush=True);continue
            entry={'arm':name,'run_id':run_id,'seed':protocol['seed'],'training_status':'NOT_STARTED','model_parameters':966784,
                   'same_initial_parameters':True,'data_order':'trainer.train Random(seed) shuffle, lane-major within each TBPTT tick'}
            arms[name]=entry;persist();model=None;arm_started=time.monotonic()
            try:
                observed=[v['arm_wall_seconds'] for v in arms.values() if v.get('training_status')=='COMPLETE' and v.get('arm_wall_seconds',0)>0]
                if observed and deadline-time.monotonic()<1.05*max(observed):
                    entry['training_status']='NOT_STARTED_BUDGET';entry['reason']='remaining budget below measured completed-arm cost plus 5% margin'
                    continue
                if time.monotonic()>=deadline:raise TimeoutError('execution budget exhausted before arm')
                model=TopologyCore(name);model.load_state_dict(initial,strict=True)
                if not all(torch.equal(value,initial[key]) for key,value in model.state_dict().items()):raise ValueError('initial weights differ')
                model=model.to(device='cuda',dtype=torch.float32)
                with timed_budget(max(1,deadline-time.monotonic())):train(model,groups['train'],groups['validation'],protocol,deadline,entry,persist)
                weight=out/(name+'-final-model.pt');torch.save({'topology':name,'config':model.config.__dict__,'model':model.state_dict(),'run_id':run_id},weight)
                entry['final_model']={'file':weight.name,'bytes':weight.stat().st_size,'kaggle_run_id':run_id,'output':str(out),'git_storage':False}
                entry['memory']={'status':'COMPLETE','peak_cuda_allocated_bytes':entry['peak_cuda_allocated_bytes'],
                                 'peak_cuda_reserved_bytes':entry['peak_cuda_reserved_bytes'],'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
                                 'state_bytes_per_lane':8192,'batch_state_bytes':32768}
                def task(key,call):
                    try:
                        remaining=deadline-time.monotonic()
                        if remaining<=0:raise TimeoutError('execution budget exhausted')
                        with timed_budget(remaining):entry[key]=call()
                    except Exception as error:entry[key]={'status':'FAILED','error':type(error).__name__+': '+str(error)}
                    finally:model.observe=False;persist()
                final_validation=entry['validation_progression'].get(str(protocol['training']['updates']),{})
                task('validation',lambda:final_validation if final_validation.get('status')=='COMPLETE' else evaluate(model,groups['validation'],deadline))
                task('holdout',lambda:evaluate(model,groups['development_holdout'],deadline))
                def pulse():
                    evidence,state=propagation(model,groups['validation'],protocol['pulse'],deadline);entry['diversity']=state;return evidence
                task('propagation',pulse)
                task('compute',lambda:compute(model,entry))
                task('inference',lambda:inference(model,groups['validation'][0],protocol['evaluation'],deadline))
                task('generation',lambda:generate(model,tokenizer,prompts,protocol['evaluation']['generation_max_tokens'],deadline))
            except Exception as error:
                entry['error']=type(error).__name__+': '+str(error)
                if entry['training_status']!='COMPLETE':entry['training_status']='INCOMPLETE'
            finally:
                if entry['training_status']=='COMPLETE':entry['arm_wall_seconds']=time.monotonic()-arm_started
                if model is not None:
                    if 'memory' in entry:entry['memory'].update(whole_arm_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),whole_arm_peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                    del model
                torch.cuda.empty_cache();persist()
        paired=comparisons(arms,protocol);write(out/'paired-comparison.json',paired);decision['screen']=screen(arms,paired,protocol)
        decision['training_status']='COMPLETE' if all(arms.get(n,{}).get('training_status')=='COMPLETE' for n in ARMS) else 'INCOMPLETE'
        for field,keys in (('evaluation_status',('validation','holdout','inference','memory')),('propagation_status',('propagation','diversity')),('compute_status',('compute',))):
            decision[field]='COMPLETE' if all(arms.get(n,{}).get(k,{}).get('status')=='COMPLETE' for n in ARMS for k in keys) else 'INCOMPLETE'
        decision['status']='DEVELOPMENT_SCREEN_COMPLETE' if all(decision[k]=='COMPLETE' for k in ('training_status','evaluation_status','propagation_status','compute_status')) and paired['status']=='COMPLETE' else 'INCOMPLETE'
    except Exception as error:decision['error']=type(error).__name__+': '+str(error)
    finally:
        persist()
        if not (out/'paired-comparison.json').exists():write(out/'paired-comparison.json',{'status':'INCOMPLETE','reason':decision.get('error')})
        lines=['# Topology-001 开发筛选','', '| Arm | Training | Tokens | Updates | Validation PPL | Holdout PPL | Tokens/s |','|---|---|---:|---:|---:|---:|---:|']
        for name in ARMS:
            entry=arms.get(name,{})
            lines.append('| '+ ' | '.join(str(v) for v in (name,entry.get('training_status','NOT_STARTED'),entry.get('prediction_tokens',0),entry.get('optimizer_updates',0),entry.get('validation',{}).get('ppl','N/A'),entry.get('holdout',{}).get('ppl','N/A'),entry.get('tokens_per_second','N/A'))) +' |')
        lines.extend(['','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','','单 Seed 文档 Bootstrap 不估计跨 Seed 方差。传播不等于预测收益。最终模型仅在各组记录的 Kaggle Output，不在 Git。未自动进入下一阶段。'])
        (out/'summary.zh-CN.md').write_text('\n'.join(lines)+'\n')
        try:
            with timed_budget(protocol['budget']['publication_seconds']):publish(out,branch)
            decision['publication_status']='COMPLETE_REMOTE_SHA_VERIFIED';persist()
        except Exception as error:
            decision['publication_status']='FAILED';decision['publication_error']=type(error).__name__+': '+str(error);persist();raise
    print(json.dumps(decision,ensure_ascii=False),flush=True)
    return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/kaggle/working/epclm-topology-001'))
    launch(parser.parse_args().output)
