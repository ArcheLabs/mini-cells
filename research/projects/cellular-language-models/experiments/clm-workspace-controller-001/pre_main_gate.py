"""Gate orchestration only: real GPU proof or BLOCKED; formal Main never invoked."""
import argparse,copy,json,time,subprocess,uuid
from pathlib import Path
import torch
from dependencies import BASE,ROOT,shared,assets
from costs import match
from matching_gate import freeze
from precision import select
from gpu_gate import correctness,benchmark,controller_equivalence
from evaluate import profile,evaluate
from models import Model
from training import train,targets
from run import publish

REQUIRED=['CONTROLLER_UPDATE_FIX','PRECISION_VALIDATION','COMPUTE_BUDGET_FREEZE','MAIN_RUNNER_READY','H2_H3_MATCHING_FREEZE','GPU_CORRECTNESS']


def budget_estimates(rates,profiles):
    if len(rates)!=6:return {'status':'BLOCKED','reason':'all six corrected-source target-precision rates required','formal_freeze':'NOT_FROZEN'}
    values={}
    for targets_per_seed in (2000000,4000000,8000000,32000000):
        per_model={arm:targets_per_seed/rate/3600 for arm,rate in rates.items()};total=sum(per_model.values())
        values[str(targets_per_seed)]={'per_model_gpu_hours':per_model,'ideal_training_gpu_hours':sum(targets_per_seed*sum(sum((row.get(k) or 0) for k in ('forward_gpu_ms','backward_gpu_ms','optimizer_gpu_ms')) for row in profiles[arm]['training']['rows'])/max(1,profiles[arm]['training']['targets'])/3600000 for arm in rates),'ideal_note':'measured CUDA event forward+backward+optimizer sum; omits host and checkpoint overhead','measured_effective_gpu_hours':total,'evaluation_gpu_hours':sum(v.get('evaluation_seconds',0) for v in profiles.values())/3600,'recovery_overhead':{'assumed_seconds_per_session':90,'remeasured':False,'note':'existing infrastructure reused; user removed verification'},'margin_fraction':.25,'estimated_total_gpu_hours':1.25*total+sum(__import__('math').ceil(hours/2.5) for hours in per_model.values())*90/3600}
    values['32000000x3']={'measured_effective_gpu_hours':3*values['32000000']['measured_effective_gpu_hours'],'estimated_total_gpu_hours':3*values['32000000']['estimated_total_gpu_hours']}
    return {'status':'CANDIDATE_NOT_FROZEN','estimates':values,'warning':'short gate runtime extrapolation, checkpoint overhead included in measured effective TPS; not a substitute for sustained benchmark; no automatic reduction from32M','formal_freeze':'AWAITING_SEPARATE_BUDGET_DECISION'}


def launch(args):
    started=args.started if args.started is not None else time.monotonic();deadline=started+10200
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();p=json.loads((BASE/'protocol.json').read_text())
    out=args.output/('clm-v04-pre-main-'+source[:12]+'-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    status={gate:'NOT_RUN' for gate in REQUIRED};status['CROSS_SESSION_RECOVERY']='NOT_RUN';evidence={};rates={};runtime={}
    configs=match(p['model']);proof={'status':'PASS','source_commit':source,'changes':['last dead GRU removed','explicit native precision policy','corrected T0-F','formal modes and scheduler ledger'],'historical_smoke_unchanged':True}
    shared.write(out/'source-audit.json',proof);shared.write(out/'model-matching.json',configs);shared.write(out/'hypothesis-matching-freeze.json',freeze(p['model']));status['H2_H3_MATCHING_FREEZE']='PASS'
    try:
        check=subprocess.run([__import__('sys').executable,'-m','pytest','-q',str(BASE/'tests')],cwd=ROOT,capture_output=True,text=True,timeout=120,env={**__import__('os').environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'})
        shared.write(out/'controller-update-fix.json',{'status':'PASS' if check.returncode==0 else 'FAIL','stdout':check.stdout,'stderr':check.stderr,'source_commit':source})
        status['CONTROLLER_UPDATE_FIX']='PASS' if check.returncode==0 else 'FAIL'
        if check.returncode:raise RuntimeError('critical local correctness regression failed')
        if not torch.cuda.is_available() or torch.cuda.get_device_name()!='Tesla T4':raise RuntimeError('Tesla T4 unavailable; no remote GPU result inferred from local tests')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        docs,manifest=assets.prepare(args.cache,p);groups=[]
        for i in range(0,min(len(docs['train']),48),8):groups.append(docs['train'][i:i+8])
        with shared.bounded(max(.1,deadline-time.monotonic())):
            evidence['precision']=benchmark(configs,groups,p,out/'precision-runs',deadline);shared.write(out/'precision-audit.json',evidence['precision']);status['PRECISION_VALIDATION']=evidence['precision']['status']
            regression=controller_equivalence(p['model'],p,deadline);shared.write(out/'controller-update-fix.json',regression);status['CONTROLLER_UPDATE_FIX']=regression['status']
            evidence['gpu']=correctness(configs,p,deadline);shared.write(out/'gpu-correctness.json',evidence['gpu']);status['GPU_CORRECTNESS']=evidence['gpu']['status']
        if any(status[k]!='PASS' for k in ('CONTROLLER_UPDATE_FIX','PRECISION_VALIDATION','GPU_CORRECTNESS')):raise RuntimeError('target precision gate failed; dry run blocked')
        for arm,entry in configs.items():
            model=Model(arm,entry['config'],17).cuda();folder=out/'runtime'/arm;folder.mkdir(parents=True)
            with shared.bounded(max(.1,deadline-time.monotonic())):progress,performance=train(model,groups,p,deadline,folder/'checkpoint.pt','gate-runtime-'+source+'-'+arm,resume=False)
            rate=performance['session_targets_per_second'];rates[arm]=rate
            validation=evaluate(model,docs['validation'][:8],p,deadline)
            runtime[arm]={'training':progress,'performance':performance,'evaluation_seconds':validation['seconds'],'validation':validation,'inference':profile(model,p,deadline,select(p['precision'].get('requested','auto'),'cuda'))}
            del model;torch.cuda.empty_cache()
        shared.write(out/'runtime-profile.json',runtime)
        # Execute the real Main Runner path for three seeds under an independent bounded identity.
        result=subprocess.run([__import__('sys').executable,str(BASE/'main_runner.py'),'--mode','main-dry-run','--dry-run-targets','8192','--arms','C2','--output',str(out/'dry-run'),'--cache',str(args.cache),'--started',str(started),'--session-seconds',str(max(1,int(deadline-started)))],cwd=ROOT,capture_output=True,text=True,timeout=max(1,deadline-time.monotonic()))
        subprocess.run(['git','switch','--detach',source],cwd=ROOT,check=True,capture_output=True,text=True)
        decisions=list((out/'dry-run').rglob('decision.json'));decision=json.loads(decisions[-1].read_text()) if decisions else {}
        status['MAIN_RUNNER_READY']='PASS' if result.returncode==0 and decision.get('status')=='DRY_RUN_COMPLETE' else 'FAIL'
        shared.write(out/'main-runner-dry-run.json',{'status':status['MAIN_RUNNER_READY'],'decision':decision,'stdout':result.stdout[-8000:],'stderr':result.stderr[-2000:]})
    except Exception as failure:evidence['error']=type(failure).__name__+': '+str(failure)
    finally:
        estimates=budget_estimates(rates,runtime);estimates.update(source_commit=source,model_configs=configs,precision_policy=select('auto','cuda') if torch.cuda.is_available() else select('auto','cpu'),seeds=[17,29,43],targets_per_seed=32000000,single_session_limit_seconds=10800)
        shared.write(out/'main-budget-freeze.json',estimates);status['COMPUTE_BUDGET_FREEZE']='BLOCKED' # Separate explicit budget decision is still required.
        if args.budget_freeze and 'manifest' in locals():
            try:
                from main_runner import validate_freeze
                approved=json.loads(args.budget_freeze.read_text());validate_freeze(approved,source,p,manifest)
                if approved['model_configs']!=configs:raise ValueError('frozen model configs mismatch')
                status['COMPUTE_BUDGET_FREEZE']='PASS'
            except Exception as freeze_error:estimates['freeze_error']=str(freeze_error)
        for key in REQUIRED:
            if status[key]=='NOT_RUN':status[key]='BLOCKED'
        decision={'format':'minicells.clm-workspace-controller-001.v04','run_id':out.name,'status':'MAIN_READY' if all(status[k]=='PASS' for k in REQUIRED) else 'BLOCKED','gates':status,'required_gates':REQUIRED,'cross_session_recovery_requirement':'REMOVED_BY_USER; existing infrastructure reused, no new Kaggle recovery verification','main_training_started':False,'main_results':'NOT_RUN','hypotheses':dict.fromkeys(('H1','H2','H3','H4'),'NOT_EVALUATED'),'source_commit':source,'evidence':evidence,'elapsed_seconds':time.monotonic()-started,'scientific_confirmation':False}
        for name in ('precision-audit','gpu-correctness','runtime-profile','main-runner-dry-run'):
            if not (out/(name+'.json')).exists():shared.write(out/(name+'.json'),{'status':'BLOCKED','reason':evidence.get('error','not reached')})
        shared.write(out/'gate-decision.json',decision);shared.write(out/'decision.json',decision);shared.write(out/'protocol.json',p)
        shared.write(out/'cross-session-recovery.json',{'status':'NOT_RUN','required':False,'reason':'user explicitly removed this validation requirement'})
        (out/'PRE-MAIN-AUDIT.zh-CN.md').write_text('# CLM v0.4 主实验门禁\n\n'+json.dumps(decision,ensure_ascii=False,indent=2)+'\n\n未启动正式Main。原生精度、GPU正确性、性能与Dry Run只接受当前Source真实执行证据；预算等待独立决策。\n')
        (out/'RESEARCH-REPORT.zh-CN.md').write_text((out/'PRE-MAIN-AUDIT.zh-CN.md').read_text())
        try:
            with shared.bounded(max(.1,started+10800-time.monotonic())):publish(out,'clm-workspace-controller-001-pre-main')
        except Exception as error:shared.write(out/'publication-failure.json',{'error':str(error)})
        print({'status':decision['status'],'report':str(out/'gate-decision.json')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--started',type=float);parser.add_argument('--budget-freeze',type=Path);parser.add_argument('--output',type=Path,default=Path('/kaggle/working/clm-v04-gates'));parser.add_argument('--cache',type=Path,default=Path('/kaggle/working/stateful-routing-data'));launch(parser.parse_args())
