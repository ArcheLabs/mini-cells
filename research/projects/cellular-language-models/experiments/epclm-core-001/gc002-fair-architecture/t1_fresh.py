"""GC002-T1-FRESH-02: fresh initialization, one final save, independent evaluations."""
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from cached_transformer import CachedTransformer,Config
from training_path import FullSequenceSampler,teacher_forced_sequences
from native_clm_runtime import lr_multiplier,load_tokenizer
from preparation import parameter_count
from repair_io import backend,version_files,timed_budget
from t1_evaluation import evaluate_documents,generation_diagnostics,inference_profile,sync

BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
FILES=('manifest.json','decision.json','training-config.json','training-metrics.json',
       'validation-results.json','holdout-results.json','generation-metrics.json','generation-samples.json',
       'inference-profile.json','memory-profile.json','final-model-manifest.json','research-summary.zh-CN.md')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def load_assets(out,deadline):
    """Fetch only version-142 data via existing transport, without file hashing or model recovery."""
    old=json.loads((BASE.parent/'gc001-baseline-repair/protocol.json').read_text())['gc_protocol']
    api,fetch=backend(ROOT);version=142;names=version_files(api,version,deadline)
    namespace=old['C1_checkpoint']['namespace']
    manifests=sorted(n for n in names if n.endswith(namespace+'/recovery.json'))
    if len(manifests)!=1:raise ValueError('version 142 input manifest unavailable or ambiguous')
    name=manifests[0];cache=out/'input-download'
    fetch(api,{'version_number':version,'files':[{'name':name,'bytes':None}]},[name],cache)
    manifest=json.loads((cache/name).read_text());prefix=name.rsplit('/',1)[0]+'/'
    logical=['assets/'+n for n in ('manifest.json','tokenizer.json','train.json','development.json')]
    locations={n:prefix+manifest.get('objects',{}).get(n,n) for n in logical}
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in locations.values()):raise ValueError('unsafe asset path')
    fetch(api,{'version_number':version,'files':[{'name':n,'bytes':None} for n in locations.values()]},list(locations.values()),cache)
    assets=out/'input-assets';assets.mkdir()
    for logical_name,physical in locations.items():shutil.copyfile(cache/physical,assets/Path(logical_name).name)
    tokenizer=load_tokenizer(assets/'tokenizer.json')
    train=json.loads((assets/'train.json').read_text());dev=json.loads((assets/'development.json').read_text())
    if not train or len(dev)<=16 or tokenizer.get_vocab_size()!=2048:raise ValueError('invalid dataset split or vocabulary')
    ids=set();tokens_seen=set()
    for doc in train+dev:
        tokens=doc['tokens'];signature=tuple(tokens)
        if doc['id'] in ids or signature in tokens_seen:raise ValueError('duplicate document across splits')
        ids.add(doc['id']);tokens_seen.add(signature)
        if len(tokens)<2 or tokens[0]!=2 or tokens[-1]!=3 or any(type(t)!=int or not 0<=t<2048 for t in tokens) or any(t in (0,2,3) for t in tokens[1:-1]):raise ValueError('invalid document BOS/EOS/token contract')
    if [tokenizer.token_to_id(t) for t in ('<pad>','<bos>','<eos>')]!=[0,2,3]:raise ValueError('special token IDs differ')
    return tokenizer,{'train':train,'validation':dev[:16],'development_holdout':dev[16:]}


class Counts:
    def __init__(self):self.consumed=0;self.updates=0;self.context_tokens=0
    def commit(self,events,accounting):
        self.consumed+=len(events);self.updates+=1;self.context_tokens+=accounting['context_reuse_tokens']


def train(protocol,docs,device,out,deadline):
    torch.manual_seed(protocol['seed'])
    config=Config(**{k:protocol['model'][k] for k in ('vocab','dim','layers','heads','kv_heads','ffn','window')})
    model=CachedTransformer(config).to(device=device,dtype=torch.float32).train()
    if parameter_count(config)!=protocol['model']['active_parameters']:raise ValueError('parameter count differs')
    opt=protocol['optimizer'];schedule=protocol['scheduler'];plan=protocol['training']
    optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],betas=tuple(opt['betas']),weight_decay=opt['weight_decay'],foreach=opt['foreach'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_multiplier(min(step+1,schedule['total_updates']),schedule['total_updates'],schedule['warmup_updates']))
    sampler=FullSequenceSampler(docs,protocol['seed'],plan['batch_lanes'],plan['fresh_prediction_targets_per_lane'],plan['context_reuse_per_sequence_max'],record_identity=False)
    counts=Counts();metrics=[];started=time.monotonic()
    assert not optimizer.state and counts.updates==0 and counts.consumed==0
    try:
        while counts.consumed<plan['prediction_token_budget']:
            if time.monotonic()>=deadline:raise TimeoutError('fresh training execution budget exhausted; rerun starts from zero')
            proposal=sampler.propose(plan['prediction_token_budget']-counts.consumed)
            sync(model);begin=time.monotonic();optimizer.zero_grad(set_to_none=True);lr=optimizer.param_groups[0]['lr']
            with sdpa_kernel(SDPBackend.MATH):
                loss,accounting=teacher_forced_sequences(model,proposal['segments'])
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),opt['gradient_clip_norm'],foreach=False,error_if_nonfinite=True)
                if accounting['prediction_tokens']!=len(proposal['events']):raise ValueError('loss target count differs from sampler')
                optimizer.step()
            if not bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all()):raise FloatingPointError('nonfinite parameters')
            scheduler.step();sampler.resolve(True,counts);sync(model)
            row={'update':counts.updates,'prediction_tokens':counts.consumed,'update_prediction_tokens':accounting['prediction_tokens'],
                 'loss':float(loss.detach()),'learning_rate':lr,'gradient_norm':float(grad),
                 'update_seconds':time.monotonic()-begin,'cumulative_training_seconds':time.monotonic()-started}
            metrics.append(row)
            if counts.updates%25==0 or counts.updates<=2:print(json.dumps({'stage':'FRESH_TRAINING',**row}),flush=True)
        if counts.updates!=schedule['total_updates']:raise ValueError('optimizer update budget differs')
        return model,counts
    finally:
        write(out/'training-metrics.json',{'metrics':metrics,'prediction_tokens':counts.consumed,'optimizer_updates':counts.updates,
                                         'context_reuse_tokens_excluded':counts.context_tokens,'training_seconds':time.monotonic()-started})


def run_complete_evaluation(model,tokenizer,groups,protocol,device,out,deadline):
    """Always call every task; an existing NOT_RUN file is never a completion signal."""
    model.eval();results={}
    def memory():
        return {'model_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None,
            'inference_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'kv_cache_memory':results.get('inference',{}).get('history_lengths',[])}
    tasks={'validation':('validation-results.json',lambda:evaluate_documents(model,groups['validation'],device)),
           'holdout':('holdout-results.json',lambda:evaluate_documents(model,groups['development_holdout'],device)),
           'generation':('generation-metrics.json',lambda:generation_diagnostics(model,tokenizer,protocol,device)),
           'inference':('inference-profile.json',lambda:inference_profile(model,tokenizer,protocol,device)),
           'memory':('memory-profile.json',memory)}
    for name,(filename,call) in tasks.items():
        started=time.monotonic()
        try:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('evaluation execution budget exhausted')
            with timed_budget(remaining):value=call()
            if value.get('status') in ('FAIL','FAILED','NOT_RUN'):raise ValueError('evaluation returned '+value['status'])
            value.update(status='COMPLETE',evaluation_seconds=time.monotonic()-started)
        except Exception as error:value={'status':'FAILED','error':type(error).__name__+': '+str(error),'evaluation_seconds':time.monotonic()-started}
        results[name]=value
        if name=='generation':
            write(out/'generation-samples.json',{'status':value['status'],'samples':value.pop('samples',[]),'error':value.get('error')})
        write(out/filename,value)
    return results


def publish(out,decision):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results
    name='epclm-gc002-t1-fresh-02';destination=ROOT/'results'/name/'publish'
    destination.mkdir(parents=True,exist_ok=True)
    for filename in FILES:
        if (out/filename).exists():shutil.copyfile(out/filename,destination/filename)
    branch='kaggle/'+decision['run_id']
    decision['publication']={'status':'PUSH_PENDING_REMOTE_VERIFICATION','branch':branch}
    write(out/'decision.json',decision);shutil.copyfile(out/'decision.json',destination/'decision.json')
    artifacts=prepare_artifacts(ROOT,name,EXPERIMENTS[name],None)
    push_results(ROOT,artifacts,name,branch,'GITHUB_TOKEN','origin/main')
    decision['publication']={'status':'REMOTE_SHA_VERIFIED','branch':branch}
    write(out/'decision.json',decision)
    print(json.dumps({'stage':'RESULTS_PUBLISHED','branch':branch,'remote_sha':'VERIFIED','status':decision['status']}),flush=True)


def launch(output):
    frozen=json.loads((BASE/'t1-formal-protocol.json').read_text())
    protocol={k:frozen[k] for k in ('model','seed','training','optimizer','scheduler','budget','evaluation')}
    protocol['data']={'dataset_version':142,'tokenizer_version':'Kaggle output 142','split':'validation=development[:16]; holdout=development[16:]'}
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    run_id='epclm-gc002-t1-fresh-02-'+source[:12]+'-'+uuid.uuid4().hex[:8]
    out=output/run_id;out.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+protocol['budget']['execution_seconds']
    decision={'format':'minicells.gc002.t1-fresh-02.v1','experiment':'GC002-T1-FRESH-02','run_id':run_id,'source_commit':source,
              'training_status':'NOT_STARTED','evaluation_status':'NOT_STARTED','status':'INCOMPLETE','research_review_status':'PENDING',
              'independent_test_accessed':False,'training_tokens':0,'optimizer_updates':0,'model_parameters':967808,
              'dataset_version':142,'tokenizer_version':'Kaggle output 142','seed':protocol['seed'],'fresh_initialization':False}
    write(out/'training-config.json',protocol)
    write(out/'manifest.json',{**decision,'output_directory':str(out),'fresh_only':True})
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_name()!='Tesla T4':raise RuntimeError('requires the approved Tesla T4 GPU')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        with timed_budget(max(1,deadline-time.monotonic())):tokenizer,groups=load_assets(out,deadline)
        decision['splits']={n:{'documents':len(d),'prediction_tokens':sum(len(r['tokens'])-1 for r in d)} for n,d in groups.items()}
        decision['gpu']={'name':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'precision':'fp32','tf32':False}
        decision['training_status']='RUNNING';write(out/'decision.json',decision)
        with timed_budget(max(1,deadline-time.monotonic())):model,counts=train(protocol,groups['train'],'cuda',out,deadline)
        decision.update(training_status='COMPLETE',training_tokens=counts.consumed,optimizer_updates=counts.updates,fresh_initialization=True)
        model_path=out/'final-model.pt'
        torch.save({'config':model.config.__dict__,'model':{n:p.detach().cpu() for n,p in model.state_dict().items()},'run_id':run_id},model_path)
        write(out/'final-model-manifest.json',{'model_name':'T1-PM-KV','parameter_count':967808,'seed':protocol['seed'],
             'training_tokens':counts.consumed,'optimizer_updates':counts.updates,'dataset_version':142,'training_configuration':protocol,
             'model_file':model_path.name,'model_file_bytes':model_path.stat().st_size,
             'retrieval':'Kaggle Notebook Output / '+run_id+'/final-model.pt; weights are not in Git; private Output requires notebook-owner access'})
        decision['training_peak_allocated_bytes']=torch.cuda.max_memory_allocated();decision['training_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        results=run_complete_evaluation(model,tokenizer,groups,protocol,'cuda',out,deadline)
        for name,value in results.items():decision[name+'_status']=value['status']
        decision['evaluation_status']='COMPLETE' if all(r['status']=='COMPLETE' for r in results.values()) else 'INCOMPLETE'
        decision['status']='COMPLETE' if decision['evaluation_status']=='COMPLETE' else 'TRAINING_COMPLETE_EVALUATION_INCOMPLETE'
    except Exception as error:
        decision['error']=type(error).__name__+': '+str(error)
        if decision['training_status']=='RUNNING':decision['training_status']='INCOMPLETE'
    finally:
        metrics=out/'training-metrics.json'
        if metrics.exists():
            progress=json.loads(metrics.read_text());decision['training_tokens']=progress['prediction_tokens'];decision['optimizer_updates']=progress['optimizer_updates']
        for name in ('validation','holdout','generation','inference','memory'):decision.setdefault(name+'_status','NOT_STARTED')
        write(out/'decision.json',decision)
        (out/'research-summary.zh-CN.md').write_text('# GC002 T1 Fresh-02\n\n训练：'+decision['training_status']+'；评估：'+decision['evaluation_status']+'。\n\n单 Seed；研究分析待完成。Independent Test 未访问。最终权重仅保存在 Kaggle Output，不进入 Git。\n\n'+json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
        for filename in FILES:
            if not (out/filename).exists():write(out/filename,{'status':'NOT_STARTED','reason':decision.get('error','training did not complete'),'prediction_tokens':decision['training_tokens'],'optimizer_updates':decision['optimizer_updates']})
        try:
            with timed_budget(protocol['budget']['publication_seconds']):publish(out,decision)
        except Exception as error:
            decision['publication']={'status':'FAILED','error':type(error).__name__+': '+str(error),'kaggle_output_preserved':True}
            write(out/'decision.json',decision)
            raise
    print(json.dumps({**decision,'output':str(out)},ensure_ascii=False),flush=True)
    return decision
