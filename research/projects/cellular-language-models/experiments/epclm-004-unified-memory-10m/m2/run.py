"""One bounded M2 engineering run: common correctness, two updates/arm, costs."""
import argparse,gc,json,platform,signal,subprocess,sys,time,uuid
import importlib.metadata
from pathlib import Path
import numpy as np
import torch
from baselines import build,cached_forward,runtime_memory
from training import fit
from e0 import BASE as M1,ROOT,parameters,evaluate,write,load_checkpoint,masked_loss
from data import MQARConfig,episode,batch,dataset_manifest,prepare_wikitext,digest,file_hash
from count_parameters import count

BASE=Path(__file__).resolve().parent

def verify_sources():
    lock=json.loads((BASE/'configs/source-lock.json').read_text())
    for name,sha in lock['frozen_files'].items():
        if file_hash(M1/name)!=sha:raise ValueError('frozen M1 source changed: '+name)
    for name,sha in lock.get('inherited_files',{}).items():
        if file_hash(ROOT/name)!=sha:raise ValueError('frozen inherited helper changed: '+name)
    upstream=json.loads((BASE/'configs/upstream-lock.json').read_text())
    for name,sha in upstream['vendored_sha256'].items():
        if file_hash(BASE/name)!=sha:raise ValueError('pinned upstream reference changed: '+name)
    return lock,upstream


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()


def gate(device):
    c=dict(vocab=536,dim=16,layers=2,heads=2,kv_heads=2,ffn=32,window=4)
    b=batch([episode(MQARConfig(bindings=2,length=24),'validation',i) for i in range(2)],device)
    records={}
    for arm in ('E0','E1','E2','E3'):
        cfg={**c,'window':16384} if arm=='E0' else c
        model=build(arm,cfg,41017,state_size=4,shared_config=c).to(device)
        logits=model(b.input_ids);future=b.input_ids.clone();future[:,12:]=8
        torch.testing.assert_close(logits[:,:12],model(future)[:,:12],atol=2e-5,rtol=2e-5)
        loss=masked_loss(logits,b.labels);loss.backward()
        if not all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()):
            raise FloatingPointError('correctness gradient failed: '+arm)
        model.eval()
        with torch.no_grad():
            a,state=cached_forward(model,b.input_ids[:,:5]);z,_=cached_forward(model,b.input_ids[:,5:],state)
            cached=torch.cat([a,z],1)
            torch.testing.assert_close(cached,logits,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(model(b.input_ids[:1])[0],logits[0],atol=2e-5,rtol=2e-5)
        records[arm]={'status':'PASS','cache_max_abs_error':float((cached-logits).abs().max()),
                      'finite_gradient':True,'causality':True,'batch_isolation':True,'optimizer_updates':0}
        del model
    return records


@torch.no_grad()
def profile(model,row,device):
    model.eval();model.zero_grad(set_to_none=True)
    ids=torch.tensor([row['input_ids']],device=device);split=ids.shape[1]-8
    model(ids);timings=[]
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        sync(device);start=time.perf_counter();whole=model(ids);sync(device)
        timings.append(time.perf_counter()-start)
    sync(device);start=time.perf_counter();first,state=cached_forward(model,ids[:,:split]);sync(device)
    prefill=time.perf_counter()-start;decoded=[];times=[]
    for token in ids[:,split:].split(1,dim=1):
        sync(device);start=time.perf_counter();logits,state=cached_forward(model,token,state);sync(device)
        times.append(time.perf_counter()-start);decoded.append(logits)
    cached=torch.cat([first]+decoded,dim=1)
    torch.testing.assert_close(cached,whole,atol=1e-4,rtol=1e-4)
    return {'batch':1,'input_tokens':ids.shape[1],'prefill_tokens':split,'teacher_forced_decode_tokens':8,
            'full_forward_p50_seconds':float(np.median(timings)),'prefill_seconds':prefill,
            'decode_token_p50_seconds':float(np.median(times)),
            'full_vs_cached_max_abs_error':float((cached-whole).abs().max()),
            'runtime_state':runtime_memory(state,1),
            'inference_peak_allocated_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
            'backend':'PyTorch reference; not an optimized native-kernel benchmark'}


def publish(out,decision):
    sys.path.insert(0,str(ROOT))
    from scripts.research.publish_experiment_results import ExperimentSpec,prepare_artifacts,push_results
    names=['decision.json','manifest.json','metrics.json','parameter-counts.json','correctness.json','dataset-manifest.json']
    stage=ROOT/'results/epclm004-m2/publish';stage.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in names:shutil.copyfile(out/name,stage/name)
    dest=ROOT/'artifacts/experiments/epclm004-m2'/decision['run_id']
    spec=ExperimentSpec(str(stage.relative_to(ROOT)),str(dest.relative_to(ROOT)),tuple(names),decision['branch'],decision['format'])
    folder=prepare_artifacts(ROOT,'epclm004-m2',spec,None)
    push_results(ROOT,folder,'epclm004-m2',decision['branch'],'GITHUB_TOKEN','origin/main')
    receipt={'status':'REMOTE_SHA_VERIFIED','branch':decision['branch'],
             'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    write(out/'publication.json',receipt);print(receipt,flush=True)


def launch(args):
    if args.publish_only:
        publish(args.output,json.loads((args.output/'decision.json').read_text()));return
    started=args.started or time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    protocol=json.loads((BASE/'configs/protocol.json').read_text());protocol['profile']='cpu-tiny' if args.tiny else 'gpu-engineering'
    if protocol['training']['updates']!=2:raise ValueError('M2 entry authorizes only two engineering updates per arm')
    if args.tiny:
        for arm in protocol['arms']:
            protocol['arms'][arm]={'model':dict(vocab=8192,dim=32,layers=2,heads=4,kv_heads=4,ffn=64,window=16384 if arm=='E0' else 4),
                                   'state_size':4 if arm in ('E2','E3') else None}
        protocol['data'].update(bindings=2,length=32);protocol['data_sizes'].update(train=16,validation=8)
    elif args.device!='cuda':raise ValueError('10M engineering entry requires target GPU; CPU permits --tiny only')
    code_hash=digest({str(p.relative_to(BASE)):file_hash(p) for p in BASE.rglob('*.py') if '__pycache__' not in str(p)})
    identity={'source_commit':source,'implementation_sha256':code_hash,'protocol_sha256':digest(protocol)}
    if (out/'manifest.json').exists():
        previous=json.loads((out/'manifest.json').read_text())
        if previous['identity']!=identity:raise ValueError('output belongs to another source/protocol')
        run_id=previous['run_id']
    else:run_id='epclm004-m2-'+source[:12]+'-'+uuid.uuid4().hex[:8]
    decision={'format':'minicells.epclm004.m2.v1','run_id':run_id,'branch':'kaggle/'+run_id,'status':'INCOMPLETE',
              'stage':'PRECHECK','source_commit':source,'hypotheses':'NOT_EVALUATED','independent_test_accessed':False}
    manifest={'format':'minicells.epclm004.m2-run.v1','run_id':run_id,'identity':identity,'protocol':protocol,
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda_build':torch.version.cuda,
                             'device':args.device,'gpu':torch.cuda.get_device_name() if args.device=='cuda' and torch.cuda.is_available() else None},
              'kaggle_identity_status':'ID_PENDING_IF_RUNTIME_NOT_EXPOSED','kaggle_script_version_id':None,
              'formal_training_tokens':0}
    write(out/'manifest.json',manifest)
    for name in ['metrics.json','parameter-counts.json','correctness.json','dataset-manifest.json']:
        if not (out/name).exists():write(out/name,{})
    def interrupted(signum,frame):raise TimeoutError('termination signal; restore or flush a consistent prefix')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        frozen,upstream=verify_sources()
        manifest['frozen_source_lock']=frozen;manifest['upstream_source_lock']=upstream
        manifest['environment']['dependencies']={p:importlib.metadata.version(p) for p in ('numpy','einops','tokenizers','datasets')}
        if args.device=='cuda':
            if not torch.cuda.is_available() or torch.cuda.get_device_capability()<(7,5):raise RuntimeError('enable target T4 or newer GPU')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        manifest['environment']['precision']='fp32'
        manifest['environment']['actual_tf32']={'matmul':torch.backends.cuda.matmul.allow_tf32,'cudnn':torch.backends.cudnn.allow_tf32} if args.device=='cuda' else None
        write(out/'manifest.json',manifest)
        if args.publish:
            sys.path.insert(0,str(ROOT))
            from scripts.research.publish_experiment_results import validate_origin,load_github_token
            validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
        decision['stage']='CORRECTNESS';write(out/'correctness.json',gate(args.device))
        assets=prepare_wikitext(args.assets);data_config=MQARConfig(**protocol['data'])
        rows={s:[episode(data_config,s,i) for i in range(protocol['data_sizes'][s])] for s in ['train','validation']}
        dm=dataset_manifest(protocol['data'],rows,assets['files']['tokenizer.json']);write(out/'dataset-manifest.json',dm)
        manifest['data_assets']=assets;manifest['dataset_sha256']=digest(dm)
        counts={};all_results={};reference=protocol['arms']['E0']['model']
        for arm,entry in protocol['arms'].items():
            decision['stage']=arm;write(out/'decision.json',decision)
            arm_id={**identity,'arm':arm,'dataset_sha256':digest(dm),'tokenizer_sha256':assets['files']['tokenizer.json']}
            arm_out=out/arm
            if (arm_out/'result.json').exists():
                result=json.loads((arm_out/'result.json').read_text())
                if result['identity']!=arm_id or file_hash(arm_out/'checkpoint.pt')!=result['checkpoint']['sha256']:
                    raise ValueError('completed arm identity/checkpoint differs')
                if result['status']=='COMPLETE':
                    all_results[arm]=result;counts[arm]=result['parameters'];continue
            torch.manual_seed(protocol['model_seed']);np.random.seed(protocol['model_seed'])
            model=build(arm,entry['model'],protocol['model_seed'],state_size=entry['state_size'] or 16,shared_config=reference).to(args.device)
            counts[arm]=parameters(model)
            if not args.tiny and abs(counts[arm]['static_parameters']/10130688-1)>.01:raise ValueError('parameter match outside +/-1%')
            write(out/'parameter-counts.json',counts)
            result=fit(model,rows['train'],rows['validation'],protocol['training'],arm_out,arm_id,
                       deadline=started+protocol['budget']['execution_seconds'])
            result.update(identity=arm_id,resource_profile=profile(model,rows['validation'][0],args.device))
            result['checkpoint']['payload_verified']=load_checkpoint(arm_out/'checkpoint.pt')['identity']==arm_id
            write(arm_out/'result.json',result);all_results[arm]=result;write(out/'metrics.json',all_results)
            del model;gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
            if result['status']!='COMPLETE':break
        manifest['counts']=counts;write(out/'manifest.json',manifest)
        if len(all_results)==4 and all(r['status']=='COMPLETE' for r in all_results.values()):
            hashes=[[r['input_sha256'] for r in v['history']] for v in all_results.values()]
            if not all(h==hashes[0] for h in hashes):raise ValueError('training sample order differs across arms')
            decision['status']='M2_ENGINEERING_COMPLETE';decision['paired_training_inputs']='IDENTICAL'
        else:decision['status']='INCOMPLETE'
    except Exception as error:
        decision['error']=type(error).__name__+': '+str(error)
        # Include a consistent partial arm report if fit failed before result.json.
        metrics=json.loads((out/'metrics.json').read_text())
        for arm in protocol['arms']:
            if (out/arm/'metrics.json').exists() and arm not in metrics:metrics[arm]=json.loads((out/arm/'metrics.json').read_text())
        write(out/'metrics.json',metrics)
    finally:
        decision['wall_seconds']=time.monotonic()-started;write(out/'decision.json',decision)
    if args.publish:publish(out,decision)
    print(decision,flush=True)
    if decision['status']!='M2_ENGINEERING_COMPLETE':raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--tiny',action='store_true')
    parser.add_argument('--publish',action='store_true');parser.add_argument('--publish-only',action='store_true')
    parser.add_argument('--started',type=float)
    launch(parser.parse_args())
