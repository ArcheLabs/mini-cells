"""One launch, absolute phase deadlines, isolated results and verified publication."""
import argparse
import hashlib
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from dataclasses import replace
import torch
from dependencies import BASE,ROOT,shared
from costs import analytic
from models import Model
from data import prepare,groups,count
from engine import train,evaluate,paired,optimizer,group_update,check_time,profile,route_summary

EXPERIMENT='epclm-stateful-computation-routing-001-v03'
ARMS=('T0','M0','M1','M2')


def common_hash(model):
    return shared.sha({name:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for name,t in model.state_dict().items() if not name.startswith(('state_read.','state_address.','update.','writer.'))})


def audit(config):
    historical=BASE.parents[1]/'epclm-a2-state-001/reports/history/A2-config.json'
    record=json.loads(historical.read_text());a=record.get('configuration',record)
    expected=config['model']
    for field in ('dim','heads','ffn','vocab'):
        if a[field]!=expected[field]:raise ValueError('BLOCKED_CONFIG_AUDIT: '+field)
    models={arm:Model(arm,expected,config['seed']) for arm in ARMS}
    hashes={a:common_hash(m) for a,m in models.items() if a!='T0'}
    if len(set(hashes.values()))!=1:raise ValueError('BLOCKED_CONFIG_AUDIT: common initialization mismatch')
    return {'audit_status':'PASS','mechanism_version':'chunk-parallel-v0.3','frozen_v02_model_sha256':shared.digest(BASE.parent/'models.py'),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'a2_config_path':str(historical.relative_to(ROOT)),'a2_config_sha256':shared.digest(historical),'model_config':expected,'training_config':config['optimizer'],'data_config':{'status':'PENDING_RUNTIME_VERIFICATION','source':'pinned TinyStories revision and frozen tokenizer via retrain_config.json','boundaries':'real complete documents; not historical random slices'},'common_initialization':hashes,'parameter_count':{a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},'environment':{'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'torch':str(torch.__version__),'cuda':torch.version.cuda,'precision':'FP32','tf32':False,'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None},'historical_comparison':'background only; new complete-document protocol'}


def publish(out):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results,remote_branch_sha
    spec=EXPERIMENTS[EXPERIMENT];source=ROOT/spec.source_dir
    if source.exists():shutil.rmtree(source)
    source.mkdir(parents=True);names=[]
    for f in out.rglob('*'):
        if not f.is_file() or f.suffix not in ('.json','.jsonl','.md','.py','.ipynb') or f.name.startswith('publication-'):continue
        name=f.relative_to(out).as_posix();target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target);names.append(name)
    destination=prepare_artifacts(ROOT,EXPERIMENT,replace(spec,files=tuple(sorted(names))),None)
    decision=json.loads((out/'decision.json').read_text());branch='kaggle/'+decision['run_id'];digest=shared.digest(out/'decision.json')
    push_results(ROOT,destination,EXPERIMENT,branch,'GITHUB_TOKEN','origin/main')
    artifact_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt={'status':'ARTIFACT_COMMIT_REMOTE_SHA_VERIFIED','artifact_commit':artifact_sha,'branch':branch,'decision_sha256':digest,'note':'receipt commit verified separately, avoiding self-reference'}
    shared.write(destination/'publication-receipt.json',receipt);shared.write(out/'publication-receipt.json',receipt)
    push_results(ROOT,destination,EXPERIMENT,branch,'GITHUB_TOKEN',None)
    final=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if remote_branch_sha(ROOT,branch)!=final or shared.digest(out/'decision.json')!=digest:raise RuntimeError('publication verification failed')
    shared.write(out/'publication-verification.json',{'status':'REMOTE_SHA_VERIFIED','receipt_commit':final,'artifact_commit':artifact_sha})
    print({'publication':'REMOTE_SHA_VERIFIED','branch':branch,'sha':final},flush=True)


def summarize(decision,metrics,comparisons):
    lines=['# Stateful Computation & Routing 001','',f"状态：{decision['status']}。单 Seed 开发筛选，科学确认：否。",'',
           '| Model | Training | Targets | Argmax NLL | Sampled NLL |','|---|---|---:|---:|---:|']
    for arm in ARMS:
        e=metrics.get(arm,{});lines.append(f"| {arm} | {e.get('training',{}).get('status','NOT_RUN')} | {e.get('training',{}).get('targets',0)} | {e.get('argmax',{}).get('mean_nll','NOT_RUN')} | {e.get('sampled',{}).get('mean_nll','NOT_RUN')} |")
    lines.extend(['','## Observed',json.dumps(comparisons,ensure_ascii=False,indent=2),'','## Supported',json.dumps(decision.get('development_signals',{}),ensure_ascii=False,indent=2),'仅在完整比较和对应冻结消融均过门槛时记录开发信号；单 Seed 不科学确认。','',
                  '## Hypothesis','MECHANISM_VERSION = chunk-parallel-v0.3；M0 比较包含 Workspace、计算组织及路由差异；不证明动态选择优于固定选择。参数增加与优化路径混杂未排除。功能分化、效率优势和不可替代能力不由路由多样性推出。','',
                  '## Not Run','独立 Holdout、长程任务、第五组、参数匹配、多 Seed 均未运行。历史 A2 不作为通过阈值。','',
                  '## 下一轮','只有完整对照与关键消融显示积极信号后才考虑多 Seed 或固定路由对照；本轮不自动启动。'])
    questions=['T0 质量、参数、KV 和速度','M0 动态计算可行性','M1 State 增益','State 消融','M2 路由增益','Ignore-State-Router 消融','功能分化','历史访问成本','GPU 延迟','未完成与限制','下一轮最小实验']
    lines+=['','## 研究问题逐项索引']+[f'{i+1}. {q}：参见 metrics/cost/paired/ablation 原始证据；缺失项目为 NOT_RUN，未证明。' for i,q in enumerate(questions)]
    return '\n'.join(lines)+'\n'


def launch(args):
    started=args.started if args.started is not None else time.monotonic();config=json.loads((BASE/'protocol.json').read_text());budget=config['budget']
    run_id=EXPERIMENT+'-'+uuid.uuid4().hex[:12];out=args.output/run_id;out.mkdir(parents=True)
    deadline=lambda phase:started+budget[phase]
    decision={'format':'minicells.epclm-stateful-computation-routing-001-v03.v1','mechanism_version':'chunk-parallel-v0.3','experiment_id':EXPERIMENT,'run_id':run_id,'status':'FAILED_IMPLEMENTATION','scientific_confirmation':False,'seed':config['seed'],'training_token_budget':0,'wall_clock_budget_seconds':10800,'publication_status':'SEE_INDEPENDENT_RECEIPT','models':{},'phases':{},'not_run':['independent_holdout','long_range_tasks'],'independent_test_accessed':False}
    metrics={};comparisons={};ablations={};cost={};routing={};state={};stage='audit'
    try:
        with shared.bounded(max(1,deadline('audit')-time.monotonic())):
            observed=audit(config);shared.write(out/'audit.json',observed);decision.update(source_commit=observed['source_commit'],environment=observed['environment'])
            env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
            check=subprocess.run([sys.executable,'-m','pytest','-q',str(BASE/'tests')],capture_output=True,text=True,env=env,timeout=120,cwd=ROOT)
            shared.write(out/'correctness.json',{'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr})
            if check.returncode:raise RuntimeError('critical tests failed')
        decision['phases']['audit']='COMPLETE'
        if not torch.cuda.is_available() or torch.cuda.get_device_name()!='Tesla T4':raise RuntimeError('target Tesla T4 unavailable; training NOT_RUN')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from scripts.research.publish_experiment_results import validate_origin,load_github_token
        validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
        device=torch.device('cuda');stage='data'
        with shared.bounded(max(1,deadline('data')-time.monotonic())):docs,manifest=prepare(args.cache,config)
        decision.update(dataset_manifest_sha256=shared.sha(manifest),tokenizer_sha256=manifest['tokenizer_sha256'],train_sha256=manifest['files']['train.json'],validation_sha256=manifest['files']['validation.json'],holdout_sha256=None)
        shared.write(out/'data-manifest.json',manifest);shutil.copyfile(args.cache/'document_index.jsonl',out/'document_index.jsonl')
        document_groups=groups(docs['train'],config['groups']['documents']);stage='warmup';rates={}
        benchmarks={}
        for chunk in (32,64,128):
            config['model']['chunk']=chunk;candidate_rates={}
            for arm in ARMS:
                check_time(deadline('warmup'));initialization_start=time.monotonic()
                model=Model(arm,config['model'],config['seed']).to(device);opt=optimizer(model,config)
                generator=torch.Generator(device=device).manual_seed(config['router_seed']);baseline=[0.,0.];targets=0;times=[]
                torch.cuda.reset_peak_memory_stats(device)
                for group in document_groups:
                    step=time.monotonic();row=group_update(model,group,opt,config,baseline,generator,deadline('warmup'));baseline=row['baseline'];targets+=row['targets'];times.append(time.monotonic()-step)
                    if time.monotonic()-initialization_start>=config['benchmark']['seconds_per_model_candidate']:break
                candidate_rates[arm]={'targets_per_second':targets/(time.monotonic()-initialization_start),'group_seconds':times,'timed_targets':targets,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'last_group':row}
                del model,opt;torch.cuda.empty_cache()
            benchmarks[chunk]=candidate_rates
        # Hardware-only selection; prefer 64 if it can serve any eligible budget.
        def feasible(candidate_rates):
            for candidate in budget['candidates']:
                n=0
                for group in document_groups:
                    if n+count(group)>candidate:break
                    n+=count(group)
                if n>=budget['minimum_actual_targets'] and sum(n/candidate_rates[a]['targets_per_second'] for a in ARMS)+120<=min(4800,deadline('training')-time.monotonic())*.8:return True
            return False
        chosen=64 if feasible(benchmarks[64]) else min(benchmarks,key=lambda c:sum(1/benchmarks[c][a]['targets_per_second'] for a in ARMS))
        config['model']['chunk']=chosen;rates=benchmarks[chosen]
        shared.write(out/'chunk-benchmarks.json',benchmarks)
        # Index is regenerated for chosen logical chunk boundaries, data identity stays fixed.
        rows=[json.loads(line) for line in (args.cache/'document_index.jsonl').read_text().splitlines()]
        for r in rows:r['chunks']=[{'input_start':start,'input_end':min(start+chosen,r['prediction_targets']),'loss_mask':'all valid targets'} for start in range(0,r['prediction_targets'],chosen)]
        (out/'document_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest['selected_document_index_sha256']=shared.digest(out/'document_index.jsonl')
        shared.write(out/'data-manifest.json',manifest);decision['dataset_manifest_sha256']=shared.sha(manifest)
        shared.write(out/'warmup.json',{'mechanism_version':'chunk-parallel-v0.3','chosen_chunk':chosen,'rates':rates,'benchmarks':benchmarks})
        selected=None
        for candidate in budget['candidates']:
            prefix=[];targets=0
            for group in document_groups:
                if targets+count(group)>candidate:break
                prefix.append(group);targets+=count(group)
            if targets<budget['minimum_actual_targets']:continue
            estimate=sum(targets/rates[a]['targets_per_second'] for a in ARMS)+120
            if estimate<=min(4800,deadline('training')-time.monotonic())*(1-budget['training_reserve_fraction']):selected=prefix;break
        if selected is None:decision['status']='TIME_BUDGET_INSUFFICIENT';return
        targets=sum(count(g) for g in selected);config['selected_targets']=targets
        config['document_groups']=[[d['doc_id'] for d in g] for g in selected]
        decision['training_token_budget']=targets;stage='training'
        shared.write(out/'protocol.json',config);decision['protocol_sha256']=shared.digest(out/'protocol.json')
        # Source snapshot is kept with reports; model weights stay in Kaggle Output.
        snapshots=out/'source';snapshots.mkdir()
        for f in BASE.iterdir():
            if f.is_file() and f.suffix in ('.py','.md','.json','.ipynb'):shutil.copyfile(f,snapshots/f.name)
        decision['implementation_sha256']=shared.sha({f.name:shared.digest(f) for f in BASE.glob('*.py')})
        for arm in ARMS:
            if time.monotonic()>=deadline('training'):break
            folder=out/arm;folder.mkdir();metrics[arm]={};model=Model(arm,config['model'],config['seed']).to(device)
            try:
                with shared.bounded(max(1,deadline('training')-time.monotonic())):metrics[arm]['training']=train(model,selected,config,deadline('training'),folder)
                save=time.monotonic();shared.native._atomic_save({'model':model.state_dict(),'config':config,'arm':arm,'source_commit':decision['source_commit']},folder/'final-model.pt')
                metrics[arm]['checkpoint']={'sha256':shared.digest(folder/'final-model.pt'),'bytes':(folder/'final-model.pt').stat().st_size,'save_seconds':time.monotonic()-save,'location':'Kaggle Output only'}
            except Exception as error:
                if (folder/'training-metrics.json').exists():metrics[arm]['training']=json.loads((folder/'training-metrics.json').read_text())
                metrics[arm]['error']=str(error)
            finally:del model;torch.cuda.empty_cache();shared.write(out/'metrics.json',metrics)
        stage='evaluation'
        # Priority: all primary comparisons, all secondary sampled modes, then ablations.
        for sampled in (False,):
            for arm in ARMS:
                if metrics.get(arm,{}).get('training',{}).get('status')!='COMPLETE':continue
                try:
                    check_time(deadline('evaluation'));model=Model(arm,config['model'],config['seed']).to(device)
                    payload=torch.load(out/arm/'final-model.pt',map_location=device,weights_only=True);model.load_state_dict(payload['model']);del payload
                    with shared.bounded(max(1,deadline('evaluation')-time.monotonic())):result=evaluate(model,docs['validation'],config,deadline('evaluation'),sample=sampled)
                    name='sampled' if sampled else 'argmax';metrics[arm][name]=result;shared.write(out/arm/(name+'.json'),result)
                    if not sampled:
                        routing[arm]=route_summary(result['routes'],config['model']['cells']) if arm!='T0' else {'status':'NOT_APPLICABLE'};state[arm]={'final_state':result['states'],'chunk_updates':result.get('state_updates',[])}
                        cost[arm]={'parameters':sum(p.numel() for p in model.parameters()),'model_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'kv_bytes_per_token':2*config['model']['dim']*4*(config['model']['layers'] if arm=='T0' else 1),'state_bytes_per_document':16*128*4 if arm in ('M1','M2') else 0,'flops':analytic(model),'active_parameters':sum(p.numel() for p in model.parameters()),'active_parameters_convention':'unique parameters in active computation and all-address scoring; repeated K invocations not counted twice','latency':'NOT_MEASURED','evaluation_tokens_per_second':sum(r['targets'] for r in result['documents'])/result['seconds']}
                except Exception as error:metrics[arm]['sampled' if sampled else 'argmax']={'status':'NOT_RUN','error':str(error)}
                finally:
                    if 'model' in locals():del model
                    torch.cuda.empty_cache();shared.write(out/'metrics.json',metrics)
        order=[('M1','cell_off'),('M1','write_off'),('M2','cell_off'),('M2','write_off'),('M2','router_off'),('M1','reset_chunk'),('M2','reset_chunk'),('M1','recurrence_off'),('M2','recurrence_off'),('M2','all_off'),('M2','shuffle')]
        for arm,mode in order:
            name=arm+':'+mode
            try:
                check_time(deadline('evaluation'))
                if metrics.get(arm,{}).get('training',{}).get('status')!='COMPLETE':raise ValueError('model not completed')
                model=Model(arm,config['model'],config['seed']).to(device);model.load_state_dict(torch.load(out/arm/'final-model.pt',map_location=device,weights_only=True)['model'])
                with shared.bounded(max(1,deadline('evaluation')-time.monotonic())):ablations[name]=evaluate(model,docs['validation'],config,deadline('evaluation'),mode)
            except Exception as error:ablations[name]={'status':'NOT_RUN','error':str(error)}
            finally:
                if 'model' in locals():del model
                torch.cuda.empty_cache();shared.write(out/'ablation-results.json',ablations)
        for arm in ARMS:
            if metrics.get(arm,{}).get('training',{}).get('status')!='COMPLETE':continue
            try:
                check_time(deadline('evaluation'));model=Model(arm,config['model'],config['seed']).to(device)
                model.load_state_dict(torch.load(out/arm/'final-model.pt',map_location=device,weights_only=True)['model'])
                with shared.bounded(max(.01,deadline('evaluation')-time.monotonic())):cost.setdefault(arm,{})['latency']=profile(model,config,deadline('evaluation'))
            except Exception as error:cost.setdefault(arm,{})['latency']={'status':'NOT_RUN','error':str(error)}
            finally:
                if 'model' in locals():del model
                torch.cuda.empty_cache()
        # Optional sampled-policy diagnostics run after primary ablations and costs.
        for arm in ARMS:
            if metrics.get(arm,{}).get('training',{}).get('status')!='COMPLETE':continue
            try:
                check_time(deadline('evaluation'));model=Model(arm,config['model'],config['seed']).to(device)
                model.load_state_dict(torch.load(out/arm/'final-model.pt',map_location=device,weights_only=True)['model'])
                with shared.bounded(max(.01,deadline('evaluation')-time.monotonic())):metrics[arm]['sampled']=evaluate(model,docs['validation'],config,deadline('evaluation'),sample=True)
                shared.write(out/arm/'sampled.json',metrics[arm]['sampled'])
            except Exception as error:metrics[arm]['sampled']={'status':'NOT_RUN','error':str(error)}
            finally:
                if 'model' in locals():del model
                torch.cuda.empty_cache()
        for a,b in (('M1','M0'),('M2','M1'),('M0','T0')):
            if all(metrics.get(n,{}).get('argmax',{}).get('status')=='COMPLETE' for n in (a,b)):comparisons[a+'-'+b]=paired(metrics[a]['argmax'],metrics[b]['argmax'],config)
        for name,result in ablations.items():
            arm=name.split(':')[0]
            if result.get('status')=='COMPLETE' and metrics[arm].get('argmax',{}).get('status')=='COMPLETE':comparisons[name]=paired(result,metrics[arm]['argmax'],config)
        def observed(name,status):return comparisons.get(name,{}).get('status')==status
        decision['development_signals']={
            'state_computation':bool(observed('M1-M0','OBSERVED_IMPROVEMENT') and any(observed('M1:'+m,'OBSERVED_DEGRADATION') for m in ('cell_off','write_off'))),
            'cross_chunk_retention':bool(observed('M1:reset_chunk','OBSERVED_DEGRADATION') or observed('M2:reset_chunk','OBSERVED_DEGRADATION')),
            'state_aware_routing':bool(observed('M2-M1','OBSERVED_IMPROVEMENT') and observed('M2:router_off','OBSERVED_DEGRADATION')),
            'interpretation':'screening signals only; parameter/optimization confounding remains; missing evidence is not a negative scientific finding'}
        # Verify every cached input again after training/evaluation.
        for name,digest in manifest['files'].items():
            if shared.digest(args.cache/name)!=digest:raise ValueError('BLOCKED_DATA_INTEGRITY: input changed')
        decision['status']='COMPLETE' if all(cost.get(a,{}).get('latency',{}).get('status')=='MEASURED' for a in ARMS) and all(metrics.get(a,{}).get('argmax',{}).get('status')=='COMPLETE' for a in ARMS) and all(v.get('status')=='COMPLETE' for v in ablations.values()) else 'TIME_BOXED_PARTIAL'
    except Exception as error:
        decision['error']=type(error).__name__+': '+str(error)
        if stage=='audit':decision['status']='BLOCKED_CONFIG_AUDIT'
        elif stage=='data':decision['status']='BLOCKED_DATA_INTEGRITY'
        elif isinstance(error,TimeoutError):decision['status']='STOPPED_AT_HARD_DEADLINE' if time.monotonic()>=started+10800 else 'TIME_BOXED_PARTIAL'
    finally:
        if 'manifest' in locals():
            try:
                for name,digest in manifest['files'].items():
                    if shared.digest(args.cache/name)!=digest:raise ValueError('input file changed: '+name)
                decision['post_run_data_integrity']='PASS'
            except Exception as integrity_error:
                decision['post_run_data_integrity']=str(integrity_error);decision['status']='BLOCKED_DATA_INTEGRITY'
        decision['not_run'].append('M1 All-State-Off: redundant with CellCore-State-Off; no direct State Router path')
        decision['hypotheses']={h:'INCONCLUSIVE' if all(metrics.get(a,{}).get('argmax',{}).get('status')=='COMPLETE' for a in required) else 'NOT_RUN' for h,required in (('H1',('M0',)),('H2',('M0','M1')),('H3',('M1','M2')))}
        decision['models']={a:{'training_status':metrics.get(a,{}).get('training',{}).get('status','NOT_RUN'),'argmax_status':metrics.get(a,{}).get('argmax',{}).get('status','NOT_RUN'),'sampled_status':metrics.get(a,{}).get('sampled',{}).get('status','NOT_RUN')} for a in ARMS}
        decision['wall_clock_actual_seconds']=time.monotonic()-started
        decision['wall_clock_scope']='through decision finalization; final publication elapsed recorded independently'
        decision['phases'].update(data='COMPLETE' if 'manifest' in locals() else 'NOT_RUN_OR_FAILED',warmup='COMPLETE' if 'rates' in locals() and len(rates)==4 else 'NOT_RUN_OR_INCOMPLETE',training='COMPLETE' if all(metrics.get(a,{}).get('training',{}).get('status')=='COMPLETE' for a in ARMS) else 'NOT_RUN_OR_INCOMPLETE',evaluation='COMPLETE' if decision['status']=='COMPLETE' else 'NOT_RUN_OR_INCOMPLETE')
        shared.write(out/'protocol.json',config);decision['protocol_sha256']=shared.digest(out/'protocol.json')
        for arm,entry in cost.items():
            entry['training']=metrics.get(arm,{}).get('training',{})
            entry['checkpoint']=metrics.get(arm,{}).get('checkpoint',{})
        shared.write(out/'training-results.json',{a:v.get('training',{'status':'NOT_RUN'}) for a,v in metrics.items()})
        shared.write(out/'validation-results.json',{a:v.get('argmax',{'status':'NOT_RUN'}) for a,v in metrics.items()})
        decision['mechanism_version']='chunk-parallel-v0.3';decision['tbptt_horizon']=2;decision['chunk_size']=config['model']['chunk']
        for name,value in [('metrics',metrics),('paired-comparison',comparisons),('routing-diagnostics',routing),('state-diagnostics',state),('ablation-results',ablations),('cost-report',cost)]:shared.write(out/(name+'.json'),value)
        shared.write(out/'decision.json',decision);(out/'RESEARCH-REPORT.zh-CN.md').write_text(summarize(decision,metrics,comparisons))
        try:
            with shared.bounded(max(.01,started+10800-time.monotonic())):publish(out)
        except Exception as error:shared.write(out/'publication-failure.json',{'status':'FAILED','error':str(error),'output_preserved':True})
        print({'status':decision['status'],'report':str(out/'decision.json')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--started',type=float);parser.add_argument('--output',type=Path,default=Path('/kaggle/working/stateful-routing-results'));parser.add_argument('--cache',type=Path,default=Path('/kaggle/working/stateful-routing-data'));parser.add_argument('--publish-only',type=Path)
    args=parser.parse_args()
    if args.publish_only:publish(args.publish_only)
    else:launch(args)
