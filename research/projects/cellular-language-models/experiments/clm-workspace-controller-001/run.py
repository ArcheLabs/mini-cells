"""Automatic correctness -> matched-baseline smoke, resumable per arm; no main auto-start."""
import argparse,json,time,subprocess,os,shutil,uuid
from pathlib import Path
from dataclasses import replace
import torch
from dependencies import BASE,ROOT,shared,assets
from models import Model
from costs import match
from training import train,identity,targets
from evaluate import evaluate,profile
from recovery import restore,commit_bundle
EXPERIMENT='clm-workspace-controller-001-v04'


def publish(out,experiment=EXPERIMENT):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results,remote_branch_sha
    spec=EXPERIMENTS[experiment];source=ROOT/spec.source_dir
    if source.exists():shutil.rmtree(source)
    source.mkdir(parents=True);names=[]
    for f in out.rglob('*'):
        if not f.is_file() or f.suffix not in ('.json','.jsonl','.md','.py','.ipynb') or f.name.startswith('publication-'):continue
        name=f.relative_to(out).as_posix();target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target);names.append(name)
    destination=prepare_artifacts(ROOT,experiment,replace(spec,files=tuple(sorted(names))),None);decision=json.loads((out/'decision.json').read_text());branch='kaggle/'+decision['run_id'];digest=shared.digest(out/'decision.json')
    push_results(ROOT,destination,experiment,branch,'GITHUB_TOKEN','origin/main');first=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt={'status':'ARTIFACT_COMMIT_REMOTE_SHA_VERIFIED','artifact_commit':first,'decision_sha256':digest,'branch':branch}
    shared.write(destination/'publication-receipt.json',receipt);shared.write(out/'publication-receipt.json',receipt)
    push_results(ROOT,destination,experiment,branch,'GITHUB_TOKEN',None);final=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if remote_branch_sha(ROOT,branch)!=final or shared.digest(out/'decision.json')!=digest:raise RuntimeError('publication identity mismatch')
    shared.write(out/'publication-verification.json',{'status':'REMOTE_SHA_VERIFIED','receipt_commit':final});print({'publication':'REMOTE_SHA_VERIFIED','branch':branch,'sha':final},flush=True)


def launch(args):
    started=args.started if args.started is not None else time.monotonic();hard=started+10800;training_deadline=started+7800;evaluation_deadline=started+9600
    p=json.loads((BASE/'protocol.json').read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    out=args.resume_run if args.resume_run else args.output/('clm-v04-smoke-'+source[:12]);out.mkdir(parents=True,exist_ok=True)
    report={};counter={};cost={};decision={'format':'minicells.clm-workspace-controller-001.v04','run_id':out.name+'-'+uuid.uuid4().hex[:8],'status':'INCOMPLETE','phase':'SMOKE','main_training':'NOT_RUN_NOT_FROZEN','scientific_confirmation':False,'hypotheses':dict.fromkeys(('H1','H2','H3','H4'),'NOT_EVALUATED'),'source_commit':source,'independent_test_accessed':False,'publication_status':'SEE_INDEPENDENT_RECEIPT'}
    try:
        check=subprocess.run([os.sys.executable,'-m','pytest','-q',str(BASE/'tests')],cwd=ROOT,env={**os.environ,'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'},capture_output=True,text=True,timeout=120)
        shared.write(out/'correctness-report.json',{'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr})
        if check.returncode:raise RuntimeError('correctness gate failed')
        configs=match(p['model'],p['smoke']['seed'])
        shared.write(out/'audit.json',{'source_commit':source,'reused':['v0.3 complete-document asset builder','frozen MHA/RoPE/SwiGLU primitives','atomic checkpoint writer','GitHub publisher'],'new':['causal workspace encoder','independent experts + sparse differentiable top2','token-local GRU controller','matched baselines','resumable AMP trainer'],'historical_results_modified':False,'main_budget_frozen':False})
        shared.write(out/'model-configs.json',configs)
        if not all(configs[a]['matched'] for a in ('T0-P','T0-F')):raise RuntimeError('baseline matching outside 1%; no training')
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; smoke NOT_RUN')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from scripts.research.publish_experiment_results import validate_origin,load_github_token
        validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
        with shared.bounded(max(.1,min(started+1200,hard)-time.monotonic())):docs,manifest=assets.prepare(args.cache,p)
        groups=[];count=0;size=p['groups']['documents']
        for i in range(0,len(docs['train']),size):
            group=docs['train'][i:i+size];n=targets(group)
            if count+n>p['smoke']['target_cap']:break
            groups.append(group);count+=n
        if len(groups)<p['smoke']['minimum_groups']:raise ValueError('insufficient complete document groups for smoke')
        p['selected_smoke_targets']=count;p['selected_document_groups']=[[d['doc_id'] for d in g] for g in groups]
        immutable={'source':source,'protocol':p,'assets':manifest,'configs':configs}
        if not args.no_recovery and not (out/'manifest.json').exists():restore(out,shared.sha(immutable),min(started+1800,time.monotonic()+45))
        if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())['identity']!=shared.sha(immutable):raise ValueError('resume manifest incompatible; no checkpoint consumed')
        if not (out/'manifest.json').exists():shared.write(out/'manifest.json',{'identity':shared.sha(immutable),**immutable,'environment':{'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda}})
        shared.write(out/'session-environment.json',{'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'bf16_supported':torch.cuda.is_bf16_supported()})
        shared.write(out/'protocol.json',p);shutil.copyfile(args.cache/'document_index.jsonl',out/'document_index.jsonl')
        device=torch.device('cuda')
        for arm in ('C0','C1','C2','C3','T0-P','T0-F'):
            if time.monotonic()>=training_deadline:break
            folder=out/arm;folder.mkdir(exist_ok=True);model=Model(arm,configs[arm]['config'],p['smoke']['seed']).to(device)
            model.noise=p['router']['noise'];run_id=identity(p,arm,source,manifest)
            try:
                with shared.bounded(max(.1,training_deadline-time.monotonic())):progress,performance=train(model,groups,p,training_deadline,folder/'checkpoint.pt',run_id,checkpoint_hook=lambda path:commit_bundle(out,shared.sha(immutable),path))
                if performance['session_targets']==0 and (folder/'performance.json').exists():performance={**json.loads((folder/'performance.json').read_text()),'resume_skipped_training':True}
                shared.write(folder/'training.json',progress);shared.write(folder/'performance.json',performance)
                report[arm]={'training':progress,'performance':performance,'checkpoint':{'path':str(folder/'checkpoint.pt'),'sha256':shared.digest(folder/'checkpoint.pt'),'bytes':(folder/'checkpoint.pt').stat().st_size}}
                if progress['status']!='COMPLETE':break
                rows=progress['rows'];span=max(1,len(rows)//3)
                gradients=rows[-1]['module_gradients'];gate=sum(r['lm_loss'] for r in rows[-span:])<sum(r['lm_loss'] for r in rows[:span])
                if arm in ('C1','C2','C3'):gate=gate and any(r['module_gradients'].get('router',0)>0 for r in rows)
                if arm in ('C2','C3'):gate=gate and any(r['module_gradients'].get('controller',0)>0 for r in rows)
                report[arm]['smoke_gate']='PASS' if gate else 'FAIL'
                if not gate:raise RuntimeError(arm+' smoke loss/gradient gate failed; later stages blocked')
            except Exception as error:
                report.setdefault(arm,{})['error']=type(error).__name__+': '+str(error);break
            finally:del model;torch.cuda.empty_cache();shared.write(out/'training-metrics.json',report)
        for arm in configs:
            if report.get(arm,{}).get('smoke_gate')!='PASS':continue
            if time.monotonic()>=evaluation_deadline:break
            model=Model(arm,configs[arm]['config'],p['smoke']['seed']).to(device);saved=torch.load(out/arm/'checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(saved['model']);del saved
            validation=docs['validation'][:p['smoke']['evaluation_documents']]
            try:
                with shared.bounded(max(.1,evaluation_deadline-time.monotonic())):
                    report[arm]['validation']=evaluate(model,validation,p,evaluation_deadline)
                    cost[arm]={'analytical_active_flops':configs[arm]['flops'],'parameters':configs[arm]['parameters'],'measured':profile(model,p,evaluation_deadline)}
                    modes=['random','shuffle'] if arm in ('C1','C2','C3') else []
                    if arm in ('C2','C3'):modes+=['state_off']
                    for mode in modes:counter[arm+':'+mode]=evaluate(model,validation,p,evaluation_deadline,mode)
                    if arm in ('C1','C2','C3'):
                        for pair in p['counterfactual']['fixed_pairs']:counter[arm+':fixed:'+str(pair)]=evaluate(model,validation,p,evaluation_deadline,'fixed',pair)
            except Exception as error:report[arm]['evaluation_error']=str(error)
            finally:del model;torch.cuda.empty_cache()
        for name,digest in manifest['files'].items():
            if shared.digest(args.cache/name)!=digest:raise RuntimeError('data changed during smoke')
        decision['status']='SMOKE_COMPLETE_MAIN_NOT_FROZEN' if len(report)==6 and all(report[a].get('smoke_gate')=='PASS' and report[a].get('validation',{}).get('status')=='COMPLETE' for a in configs) else 'INCOMPLETE'
    except Exception as error:decision['error']=type(error).__name__+': '+str(error)
    finally:
        decision['elapsed_seconds_before_publication']=time.monotonic()-started;decision['models']={a:{'training':v.get('training',{}).get('status','NOT_RUN'),'smoke_gate':v.get('smoke_gate','NOT_RUN'),'validation':v.get('validation',{}).get('status','NOT_RUN')} for a,v in report.items()}
        for name,value in [('training-metrics',report),('counterfactual-routing',counter),('compute-profile',cost),('routing-diagnostics',{a:v.get('validation',{}).get('records',[]) for a,v in report.items()}),('state-diagnostics',{'scope':'token-local state only; diagnostics embedded in per-step route records','oracle':'NOT_RUN','main_seeds':'NOT_RUN'})]:shared.write(out/(name+'.json'),value)
        shared.write(out/'decision.json',decision)
        lines=['# CLM v0.4 Smoke Report','',json.dumps(decision,ensure_ascii=False,indent=2),'','| Model | Targets | Smoke Gate | NLL | PPL |','|---|---:|---|---:|---:|']
        for arm,v in report.items():lines.append('| '+' | '.join(map(str,[arm,v.get('training',{}).get('targets',0),v.get('smoke_gate','NOT_RUN'),v.get('validation',{}).get('mean_nll','NOT_RUN'),v.get('validation',{}).get('ppl','NOT_RUN')]))+' |')
        lines+=['','仅为17 Seed短训练和机制检查，不是三Seed正式结果；H1–H4均未评估。32M预算、固定训练FLOPs比较、条件性参数匹配复核和Oracle诊断尚未执行，不作架构有效性判断。','完整研究报告需主训练及三Seed完成后另行生成。']
        (out/'RESEARCH-REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
        try:
            with shared.bounded(max(.1,hard-time.monotonic())):publish(out)
        except Exception as error:shared.write(out/'publication-failure.json',{'error':str(error),'output_preserved':True})
        print({'status':decision['status'],'report':str(out/'decision.json')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--started',type=float);parser.add_argument('--no-recovery',action='store_true');parser.add_argument('--output',type=Path,default=Path('/kaggle/working/clm-v04-results'));parser.add_argument('--cache',type=Path,default=Path('/kaggle/working/stateful-routing-data'));parser.add_argument('--resume-run',type=Path);parser.add_argument('--publish-only',type=Path)
    args=parser.parse_args()
    if args.publish_only:publish(args.publish_only)
    else:launch(args)
