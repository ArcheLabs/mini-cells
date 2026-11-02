"""Sequential T1-02..T1-05 gates; the formal budget starts only after all gates pass."""
import argparse
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent;CORE=BASE.parent
ROOT=next(path for path in BASE.parents if (path/'pyproject.toml').exists())
sys.path[:0]=[str(BASE),str(CORE/'gc001-baseline-repair'),str(CORE/'implementation'),
    str(CORE.parent/'native-clm-v0/implementation'),str(ROOT),str(ROOT/'shared/python')]

from cached_transformer import Config
from formal_training import FormalTrainer,verify_resume
from flops_audit import ArithmeticCounter,profiler_inventory
from native_clm_runtime import load_tokenizer,sha256_file
from preparation import parameter_count,write
from repair_io import recover_inputs,timed_budget
from t1_evaluation import evaluate_documents,generation_diagnostics,inference_profile
from t1_validation import cache_equivalence_and_invalidation,training_reference

PROTOCOL_PATH=BASE/'t1-formal-protocol.json'
PUBLISH_FILES=('decision.json','t1-formal-readiness/decision.json','manifest.json','summary.md',
 't1-02-kv-consistency/decision.json','t1-02-kv-consistency/training-semantics.json','t1-02-kv-consistency/training-reference-tests.json','t1-02-kv-consistency/kv-cache-equivalence.json','t1-02-kv-consistency/cache-version-audit.json','t1-02-kv-consistency/document-reset-tests.json','t1-02-kv-consistency/checkpoint-state-tests.json',
 't1-03-flops-accounting/decision.json','t1-03-flops-accounting/operator-inventory.json','t1-03-flops-accounting/arithmetic-rules.json','t1-03-flops-accounting/gemm-tests.json','t1-03-flops-accounting/attention-tests.json','t1-03-flops-accounting/backward-tests.json','t1-03-flops-accounting/optimizer-tests.json','t1-03-flops-accounting/end-to-end-accounting.json','t1-03-flops-accounting/coverage-report.json',
 't1-04-engineering-validation/decision.json','t1-04-engineering-validation/model-validation.json','t1-04-engineering-validation/smoke-training.json','t1-04-engineering-validation/checkpoint-resume.json','t1-04-engineering-validation/data-ledger-validation.json','t1-04-engineering-validation/flops-ledger-validation.json','t1-04-engineering-validation/gpu-profile.json','t1-04-engineering-validation/environment.json','t1-04-engineering-validation/frozen-training-config.json',
 't1-pm-formal-single-seed/manifest.json','t1-pm-formal-single-seed/training-config.json','t1-pm-formal-single-seed/training-metrics.json','t1-pm-formal-single-seed/token-exposure-ledger.json','t1-pm-formal-single-seed/flops-ledger.json','t1-pm-formal-single-seed/checkpoint-manifest.json','t1-pm-formal-single-seed/final-model-manifest.json','t1-pm-formal-single-seed/validation-results.json','t1-pm-formal-single-seed/holdout-results.json','t1-pm-formal-single-seed/generation-metrics.json','t1-pm-formal-single-seed/inference-profile.json','t1-pm-formal-single-seed/memory-profile.json','t1-pm-formal-single-seed/environment.json','t1-pm-formal-single-seed/research-decision.json')


def write_json(path,value):write(path,value)


def environment():
    import torch
    return {'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'precision':'fp32','tf32':False,
        'autocast':False,'grad_scaler':False}


def initial_decision(source,protocol_sha,env):
    return {'format':'minicells.gc002.t1-formal.v1','experiment':'gc002-t1-pm-kv-formal',
        'status':'IN_PROGRESS','overall_status':'IN_PROGRESS','identity':{'source':source,'protocol':protocol_sha},
        't1_02_training_state_consistency':'NOT_STARTED','t1_03_flops_accounting':'NOT_STARTED',
        't1_04_engineering_validation':'NOT_STARTED','t1_05_formal_single_seed':'BLOCKED',
        'engineering_status':'NOT_VALIDATED','formal_training_status':'BLOCKED','formal_training_complete':False,
        'model_parameters':967808,'training_token_budget':250000,'seed_count':1,
        'formal_compute_matching_status':'NOT_READY','formal_compute_matching_ready':False,
        'independent_test_accessed':False,'gc001_results_modified':False,
        'formal_hypotheses':dict.fromkeys(('H1','H2','H3','H4','H5'),'NOT_EVALUATED'),
        'human_review':'PENDING','scientific_result_status':'NOT_CONFIRMED','scientific_result_confirmed':False,
        'environment':env,'error':None}


def persist(out,decision):
    decision['overall_status']=decision['status']
    write_json(out/'decision.json',decision)
    write_json(out/'t1-formal-readiness/decision.json',decision)
    write_json(out/'manifest.json',{'format':'minicells.gc002.t1-formal-run-manifest.v1',
        'run_id':decision.get('run_id'),'identity':decision.get('identity'),'status':decision['status'],
        'source_branch':decision.get('source_branch'),'protocol':decision.get('protocol'),
        'formal_token_budget':decision.get('training_token_budget'),'formal_seed_count':decision.get('seed_count'),
        'stages':{key:decision.get(key) for key in ('t1_02_training_state_consistency','t1_03_flops_accounting',
            't1_04_engineering_validation','t1_05_formal_single_seed')}})


def placeholders(out):
    for path in PUBLISH_FILES:
        if path=='decision.json':continue
        if path.endswith('.json'):
            target=out/path
            if not target.exists():write_json(target,{'status':'NOT_RUN'})
        elif path=='summary.md' and not (out/path).exists():
            (out/path).write_text('# GC002 T1-PM-KV 正式对照组\n\n等待门禁执行。\n')


def stage_decision(out,name,status,**fields):
    write_json(out/name/'decision.json',{'status':status,**fields})


def validate_freeze(protocol,registry):
    train=protocol['training'];schedule=protocol['scheduler'];optimizer=protocol['optimizer'];checkpoint=protocol['checkpoint']
    if protocol['status']!='FROZEN_T1_SINGLE_SEED_CONDITIONAL_ON_ENGINEERING_GATES':raise ValueError('T1 formal protocol is not frozen')
    if protocol['seed']!=173020021 or schedule['warmup_updates']!=97:raise ValueError('frozen T1 seed/warmup selection differs')
    matching=[row for row in registry['seeds'] if row.get('experiment')=='GC002-T1-PM-KV-FORMAL' and row.get('seed')==protocol['seed']]
    conflicts=[row for row in registry['seeds'] if row.get('seed')==protocol['seed'] and row.get('experiment')!='GC002-T1-PM-KV-FORMAL']
    if len(matching)!=1 or conflicts or matching[0].get('state')!='RESERVED_USER_SELECTED_CONDITIONAL_FORMAL_RUN':raise ValueError('formal seed registry reservation is missing or conflicting')
    if (train['prediction_token_budget'],train['batch_lanes'],train['fresh_prediction_targets_per_lane'],train['context_reuse_per_sequence_max'],train['max_input_tokens_per_sequence'],schedule['total_updates'])!=(250000,4,64,64,128,977):raise ValueError('frozen T1 budget/window/update plan differs')
    model_expected={'name':'T1-PM-KV','vocab':2048,'dim':128,'layers':4,'heads':2,'kv_heads':2,'ffn':288,'window':128,'precision':'fp32'}
    if protocol['model']!=dict(model_expected,active_parameters=967808):raise ValueError('frozen T1-PM-KV architecture differs')
    if (train['full_update_prediction_tokens'],train['final_update_prediction_tokens'],
            train['full_update_prediction_tokens']*(schedule['total_updates']-1)+train['final_update_prediction_tokens'])!=(256,144,250000):
        raise ValueError('frozen effective Prediction Target arithmetic differs')
    if (train['precision'],train['tf32'],train['autocast'],train['grad_scaler'],train['cross_optimizer_training_kv_cache'],
            train['eos_token_id'],train['padding_token_id'])!=('fp32',False,False,False,False,3,0):
        raise ValueError('frozen FP32, EOS, padding, or full-sequence training semantics differ')
    if (schedule['name'],schedule['warmup_updates'],schedule['source'])!=('linear_warmup_then_cosine',97,'native_clm_runtime.lr_multiplier'):
        raise ValueError('frozen learning-rate schedule differs')
    if checkpoint['formal_interval_updates']*train['batch_lanes']*train['fresh_prediction_targets_per_lane']!=checkpoint['formal_interval_prediction_tokens'] or checkpoint['formal_interval_prediction_tokens']<25000:
        raise ValueError('formal checkpoint interval must be no more than 25,000 prediction targets')
    if (checkpoint['smoke_interval_updates'],checkpoint['smoke_resume_at_update'])!=(5,25):raise ValueError('frozen T1-04 smoke/resume plan differs')
    validation=protocol['engineering_validation']
    if validation!={'t1_03_accounting_probe':{'updates':2,'prediction_targets':512,'source':'audited training split only','formal_training_budget_tokens':0},
            't1_04_smoke':{'updates':50,'prediction_targets':12800,'resume_check_after_update':25,
                'profile_warmup_updates_excluded':5,'formal_training_budget_tokens':0}}:
        raise ValueError('frozen engineering probe/smoke budget differs')
    if optimizer!={'name':'AdamW','learning_rate':0.0003,'betas':[0.9,0.95],'weight_decay':0.1,'foreach':False,'gradient_clip_norm':1.0}:raise ValueError('frozen optimizer config differs')
    if protocol['budget']!={'execution_seconds':5400,'publication_seconds':600,'safety_seconds':1200,'hard_limit_seconds':7200}:
        raise ValueError('frozen two-hour execution and publication budget differs')
    config=Config(**{key:protocol['model'][key] for key in ('vocab','dim','layers','heads','kv_heads','ffn','window')})
    if parameter_count(config)!=967808 or protocol['model']['active_parameters']!=967808:raise ValueError('T1-PM-KV active parameter count differs')
    match=protocol['parameter_matching'];relative=100*(parameter_count(config)/match['reference_active_parameters']-1)
    if (match['reference_active_parameters'],match['maximum_difference_percent'],match['tied_embedding_counted_once'])!=(966784,1.0,True) or abs(relative-match['relative_difference_percent'])>1e-6 or abs(relative)>match['maximum_difference_percent']:
        raise ValueError('T1-PM-KV versus C1 parameter matching audit differs')
    return config


def assert_clean_source():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    changes=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    if branch!='codex/gc002-t1-formal-readiness' or changes:raise RuntimeError('formal run requires the clean, committed T1 readiness branch')
    return source


def _t1_03(root,protocol,identity,device,out,training_documents):
    import torch
    from torch import manual_seed
    manual_seed(protocol['seed'])
    docs=training_documents
    if len(docs)<4:raise ValueError('T1-03 requires the audited four-lane training split')
    probe_plan=protocol['engineering_validation']['t1_03_accounting_probe']
    probe_identity={**identity,'mode':'T1-03_TRAIN_SPLIT_ACCOUNTING_PROBE_ONLY'}
    probe=FormalTrainer(docs,probe_identity,protocol,device,target_tokens=probe_plan['prediction_targets'])
    for _ in range(probe_plan['updates']-1):probe.update()
    row,profiler=probe.update(profile=True)
    live=probe.last_counter or {};records=list(probe.flops_aggregate.values());phase={}
    for item in records:phase[item['phase']]=phase.get(item['phase'],0)+item['estimated_flops']
    required_phases=('forward','backward','gradient_clipping','optimizer')
    categories={phase_name:sorted({item['category'] for item in records if item['phase']==phase_name}) for phase_name in required_phases}
    required_categories={'forward':{'MATRIX_MULTIPLY','ATTENTION_SOFTMAX','EMBEDDING_OR_GRADIENT_ACCUMULATION','NORMALIZATION_OR_REDUCTION','ACTIVATION_OR_TRANSCENDENTAL'},
        'backward':{'MATRIX_MULTIPLY','ATTENTION_SOFTMAX','EMBEDDING_OR_GRADIENT_ACCUMULATION','NORMALIZATION_OR_REDUCTION','ACTIVATION_OR_TRANSCENDENTAL'},
        'optimizer':{'OPTIMIZER'}}
    missing={key:sorted(required-values) for key,required in required_categories.items() if not required<=set(categories.get(key,[]))}
    unknown=live.get('unsupported_operators',{})
    if any(phase.get(key,0)<=0 for key in required_phases) or not records or unknown or missing:raise ValueError('T1 live arithmetic coverage failed: '+json.dumps({'phase_flops':phase,'unsupported':unknown,'missing_categories':missing},sort_keys=True))
    measured=profiler_inventory(profiler)
    if not measured['operator_shape_events']:raise ValueError('PyTorch profiler did not capture live operator/shape events')
    total=sum(phase.values());result={'status':'PASS','model':'T1-PM-KV','active_parameters':parameter_count(probe.config),
        'accounting_probe_updates':probe_plan['updates'],'accounting_probe_prediction_tokens':probe_plan['prediction_targets'],
        'formal_training_tokens_consumed':probe_plan['formal_training_budget_tokens'],
        'probe_data_order_sha256':probe.ledger.order_hash,'probe_exposure':probe.ledger.report(),
        'phase_flops':phase,'total_training_flops':total,'flops_per_prediction_token':total/512,
        'phase_operator_categories':categories,'unsupported_operators':unknown,'missing_major_categories':missing,
        'operator_inventory_records':len(records),'input_shape_profile':'actual full-sequence windows from four-lane batches sampled from the audited Training Split',
        'counter_convention':'semantic scalar-operation equivalents; 1 MAC=2 FLOPs; direct dispatch inventory plus reference formula; not hardware instruction measurements',
        'profiler':measured,'caveats':['transcendental functions use declared unit reference costs','logical tensor bytes are not measured DRAM traffic','kernel activity events are not guaranteed one-to-one with launches']}
    write_json(out/'t1-03-flops-accounting/end-to-end-accounting.json',result)
    write_json(out/'t1-03-flops-accounting/operator-inventory.json',{'status':'PASS','records':records,'raw_profiler':measured})
    write_json(out/'t1-03-flops-accounting/arithmetic-rules.json',{'status':'DECLARED','mac_to_flops':2,
        'pointwise':'one operation per output unless operator-specific reference formula is listed',
        'softmax':'5 scalar operations per element; backward 4 per element',
        'normalization':'reference formulas in flops_audit.py','memory':'reported separately as logical output bytes for memory/shape dispatches',
        'integer_operations':'retained in operator input/output dtypes; not added to FLOPs',
        'data_transfer':'reported as logical copy/to operations and bytes; not measured physical traffic',
        'kernel_launches':'profiler CUDA activity events only; not added to FLOPs',
        'unknown_float_ops':'UNSUPPORTED; block formal compute readiness'})
    write_json(out/'t1-03-flops-accounting/gemm-tests.json',gemm_audit(device))
    is_matrix=lambda record:any(name in record['operator'] for name in ('aten.bmm','aten.mm','aten.matmul','aten.addmm'))
    forward_matrix=[r for r in records if r['phase']=='forward' and is_matrix(r)]
    backward_matrix=[r for r in records if r['phase']=='backward' and is_matrix(r)]
    forward_softmax=[r for r in records if r['phase']=='forward' and 'softmax' in r['operator']]
    backward_softmax=[r for r in records if r['phase']=='backward' and 'softmax' in r['operator']]
    if (len({tuple(r['input_shapes']+r['output_shapes']) for r in forward_matrix if 'bmm' in r['operator']})<2
            or not backward_matrix or not forward_softmax or not backward_softmax):
        raise ValueError('attention QK/AV matrix, softmax, or backward operators are missing from the live inventory')
    attention=forward_matrix+backward_matrix+forward_softmax+backward_softmax
    write_json(out/'t1-03-flops-accounting/attention-tests.json',{'status':'PASS' if attention and backward_matrix and backward_softmax else 'FAIL',
        'attention_records':attention,'qk_and_av_forward_matrix_records':forward_matrix,'softmax_forward_records':forward_softmax,
        'attention_backward_matrix_records':backward_matrix,'softmax_backward_records':backward_softmax,
        'basis':'actual T1-PM-KV updates using held-out-free documents from the audited training split'})
    write_json(out/'t1-03-flops-accounting/backward-tests.json',{'status':'PASS' if phase.get('backward',0)>0 and backward_matrix and backward_softmax else 'FAIL',
        'backward_flops':phase.get('backward',0),'matrix_attention_gradient_records':backward_matrix,'softmax_gradient_records':backward_softmax})
    optimizer_records=[r for r in records if r['phase']=='optimizer'];optimizer_names={r['operator'].split('.')[-1].rstrip('_') for r in optimizer_records}
    optimizer_required={'lerp','addcmul','sqrt','addcdiv','mul'};missing_optimizer=sorted(optimizer_required-optimizer_names)
    optimizer_state=next(iter(probe.optimizer.state.values()),{})
    optimizer_state_ok=(int(optimizer_state.get('step',0))==2 and all(key in optimizer_state and torch.isfinite(optimizer_state[key]).all() for key in ('exp_avg','exp_avg_sq')))
    if missing_optimizer or not optimizer_state_ok:raise ValueError('AdamW moment/bias-corrected update accounting is incomplete: '+json.dumps({'missing':missing_optimizer,'state_valid':optimizer_state_ok}))
    write_json(out/'t1-03-flops-accounting/optimizer-tests.json',{'status':'PASS','optimizer_flops':phase.get('optimizer',0),
        'optimizer_operator_records':optimizer_records,'required_observed_operators':sorted(optimizer_required),'optimizer_state_step':int(optimizer_state['step']),
        'first_and_second_moments_finite':True,'bias_correction_and_parameter_update':'EXECUTED_BY_PYTORCH_ADAMW; scalar host-side bias correction excluded from GPU FLOPs'})
    coverage={'status':'PASS','registered_operator_records':len(records),'forward_flops':phase['forward'],
        'backward_flops':phase['backward'],'gradient_clipping_flops':phase['gradient_clipping'],'optimizer_flops':phase['optimizer'],
        'unsupported_operators':unknown,'missing_major_categories':missing,'formal_compute_matching_ready':True,
        'scope':'T1-PM-KV target-runtime arithmetic ledger; does not freeze GC002 cross-model Track C budget'}
    write_json(out/'t1-03-flops-accounting/coverage-report.json',coverage)
    stage_decision(out,'t1-03-flops-accounting','PASS',formal_compute_matching_ready=True,total_flops=total)
    probe.poisoned=True
    del probe
    return result,coverage


def gemm_audit(device):
    import torch
    a=torch.randn(2,3,device=device,requires_grad=True);b=torch.randn(3,4,device=device,requires_grad=True)
    counter=ArithmeticCounter()
    with counter:
        y=a@b;loss=y.sum();counter.phase='backward';ones=torch.ones_like(y)
        matrix_total=lambda phase:sum(row['estimated_flops'] for row in counter.records.values()
            if row['phase']==phase and row['category']=='MATRIX_MULTIPLY')
        before=matrix_total('backward');grad_a,=torch.autograd.grad(y,a,ones,retain_graph=True);d_a=matrix_total('backward')-before
        before=matrix_total('backward');grad_b,=torch.autograd.grad(y,b,ones,retain_graph=True);d_b=matrix_total('backward')-before
        loss.backward(retain_graph=True);first_a=a.grad.detach().clone();first_b=b.grad.detach().clone();loss.backward()
    expected=2*2*3*4;accumulation=sum(row['estimated_flops'] for row in counter.records.values()
        if row['phase']=='backward' and row['operator']=='aten.add_')
    numerical=(torch.allclose(grad_a,ones@b.detach().T) and torch.allclose(grad_b,a.detach().T@ones)
        and torch.allclose(a.grad,2*first_a) and torch.allclose(b.grad,2*first_b))
    if (matrix_total('forward')!=expected or d_a!=expected or d_b!=expected or accumulation!=a.numel()+b.numel()
            or not numerical or counter.unknown):raise ValueError('GEMM forward/dA/dB/gradient-accumulation accounting failed')
    return {'status':'PASS','formula':'2*M*K*N','shape':[2,3,4],'forward_gemm_flops':matrix_total('forward'),
        'dA_flops':d_a,'dB_flops':d_b,'gradient_accumulation_flops':accumulation,
        'expected_each_gemm_flops':expected,'gradient_accumulation_numerical_check':'PASS',
        'unsupported_operators':dict(counter.unknown)}


def _profile_summary(engine,smoke,protocol):
    import torch
    warmup=protocol['engineering_validation']['t1_04_smoke']['profile_warmup_updates_excluded'];times=[row['seconds'] for row in smoke['metrics'][warmup:]]
    return {'status':'MEASURED_ON_TARGET_GPU','gpu':torch.cuda.get_device_name(),'precision':'fp32',
        'warmup_updates_excluded':warmup,'measured_updates':len(times),'median_update_seconds':statistics.median(times),
        'p95_update_seconds':sorted(times)[math.ceil(.95*len(times))-1],
        'median_tokens_per_second':statistics.median(row['tokens_per_second'] for row in smoke['metrics'][warmup:]),
        'overall_smoke_tokens_per_second':smoke['prediction_tokens']/smoke['training_seconds'],
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
        'gpu_utilization':'NOT_MEASURED_NO_NVML_DEPENDENCY','scope':'50-update full-sequence FP32 smoke; no CPU performance benchmark'}


def _validate_assets(root,output,protocol,deadline):
    import torch
    from generation_evaluation import audit_data
    from native_clm_runtime import load_tokenizer
    old=json.loads((CORE/'gc001-baseline-repair/protocol.json').read_text())['gc_protocol']
    assets=output/'private-input-assets'
    with timed_budget(120):proof=recover_inputs(root,assets,old,[protocol['data']['known_kaggle_asset_version']],deadline,include_weights=False)
    tokenizer=load_tokenizer(assets/'assets/tokenizer.json');groups,audit=audit_data(assets,tokenizer,old)
    hashes={name:sha256_file(assets/'assets'/name) for name in ('manifest.json','tokenizer.json','train.json','development.json')}
    expected={'manifest.json':protocol['data']['dataset_manifest_sha256'],'tokenizer.json':protocol['data']['tokenizer_sha256'],
        'train.json':protocol['data']['train_json_sha256'],'development.json':protocol['data']['development_json_sha256']}
    if hashes!=expected:raise ValueError('raw dataset or Tokenizer SHA differs from frozen T1 identity')
    if audit['status']!='PASS_DEVELOPMENT_ONLY' or audit['independent_confirmation']!='SEALED_NOT_ACCESSED':raise ValueError('data split audit failed')
    torch.cuda.empty_cache()
    return assets,tokenizer,groups,audit,proof,hashes


def _smoke(protocol,identity,groups,out,deadline,recovery_identity):
    import torch
    plan=protocol['engineering_validation']['t1_04_smoke'];smoke_tokens=plan['prediction_targets'];folder=out/'t1-04-engineering-validation';checkpoint=folder/'private/smoke.pt'
    smoke_identity={**identity,'mode':'T1-04_ENGINEERING_SMOKE','training_tokens':smoke_tokens}
    report_path=folder/'smoke-training.json';prior_report=json.loads(report_path.read_text()) if report_path.exists() else {}
    if checkpoint.exists():engine=FormalTrainer.load(checkpoint,groups['train'],smoke_identity,protocol,'cuda',smoke_tokens)
    elif prior_report.get('committed_tokens',prior_report.get('prediction_tokens',0))>0:
        raise RuntimeError('T1-04 progress report exists but its same-source checkpoint is unavailable; refusing repeat smoke')
    else:engine=FormalTrainer(groups['train'],smoke_identity,protocol,'cuda',smoke_tokens)
    torch.cuda.reset_peak_memory_stats()

    def checkpoint_progress(status='RUNNING'):
        engine.save(checkpoint)
        report={'status':status,'run_id':'t1-04-smoke-'+identity['source'][:12],
            'training_tokens':smoke_tokens,'committed_tokens':engine.ledger.consumed,'optimizer_updates':engine.ledger.updates,
            'metrics':engine.metrics,'resume':engine.resume_result,'training_seconds':sum(r['seconds'] for r in engine.metrics),
            'checkpoint_sha256':sha256_file(checkpoint)}
        write_json(folder/'smoke-training.json',report)
        bundle(out,recovery_identity)
        return report

    while engine.ledger.consumed<smoke_tokens:
        if time.monotonic()>deadline-900:
            report=checkpoint_progress('INCOMPLETE')
            report['reason']='insufficient time to finish T1-04 smoke before reserved formal execution/publication'
            write_json(folder/'smoke-training.json',report);bundle(out,recovery_identity)
            return engine,report
        if engine.ledger.updates==plan['resume_check_after_update'] and engine.resume_result['status']!='PASS':engine=verify_resume(engine,checkpoint)
        row,_=engine.update()
        if engine.ledger.updates%protocol['checkpoint']['smoke_interval_updates']==0 or engine.ledger.consumed==smoke_tokens:
            checkpoint_progress()
        if row['unsupported_operators']:raise ValueError('T1-04 smoke observed unsupported operators')
    if engine.ledger.consumed!=smoke_tokens or engine.ledger.updates!=plan['updates'] or engine.resume_result['status']!='PASS':raise ValueError('T1-04 smoke budget/resume gate failed')
    if any(not all(math.isfinite(row[key]) for key in ('loss','gradient_norm_before_clip')) for row in engine.metrics):raise ValueError('T1-04 loss/gradient nonfinite')
    checkpoint_progress('PASS')
    report={'status':'PASS','run_id':'t1-04-smoke-'+identity['source'][:12],'training_status':'COMPLETE',
        'prediction_tokens':engine.ledger.consumed,'optimizer_updates':engine.ledger.updates,'metrics':engine.metrics,
        'training_seconds':sum(r['seconds'] for r in engine.metrics),'resume':engine.resume_result,
        'formal_training_tokens_consumed':0,'scope':'separate engineering smoke; not part of the 250K formal budget'}
    return engine,report


def _flops_report(engine):
    aggregate=list(engine.flops_aggregate.values());totals={}
    for record in aggregate:totals[record['phase']]=totals.get(record['phase'],0)+record['estimated_flops']
    return {'status':'PASS' if not any(row['unsupported_operators'] for row in engine.flops_updates) else 'FAIL',
        'convention':'semantic scalar-operation equivalents v1; 1 MAC=2 FLOPs; hardware instruction FLOPs not claimed',
        'prediction_tokens':engine.ledger.consumed,'phase_flops':totals,'total_training_flops':sum(totals.values()),
        'flops_per_prediction_token':sum(totals.values())/max(1,engine.ledger.consumed),
        'operator_inventory':aggregate,'per_update':engine.flops_updates,
        'unsupported_operators':{key:value for row in engine.flops_updates for key,value in row['unsupported_operators'].items()},
        'formal_compute_matching_ready':not any(row['unsupported_operators'] for row in engine.flops_updates)}


def _save_formal_checkpoint(engine,checkpoint,folder,out,decision,formal_identity,recovery_identity):
    engine.save(checkpoint)
    digest=sha256_file(checkpoint)
    write_json(folder/'training-metrics.json',{'status':'IN_PROGRESS','metrics':engine.metrics,
        'training_seconds':engine.training_seconds,'checkpoint_seconds':engine.checkpoint_seconds})
    write_json(folder/'checkpoint-manifest.json',{'identity':formal_identity,'checkpoint':{'path':'private/training.pt',
        'sha256':digest,'bytes':checkpoint.stat().st_size,'format':'gc002.t1-formal-state.v1'},
        'retrieval':'download this file from the Kaggle Notebook Output; checksum and internal envelope SHA are verified on resume'})
    decision['formal_progress']={'committed_tokens':engine.ledger.consumed,'optimizer_updates':engine.ledger.updates,
        'checkpoint_sha256':digest,'resume':'SAME_SOURCE_CHECKPOINT_SAVED'}
    persist(out,decision)
    bundle(out,recovery_identity)


def _publish(root,out,decision):
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results
    experiment='epclm-gc002-t1-formal';destination=root/'results'/experiment/'publish';destination.mkdir(parents=True,exist_ok=True)
    for name in PUBLISH_FILES:
        (destination/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(out/name,destination/name)
    (destination/'summary.md').write_text('# GC002 T1-PM-KV 正式单 Seed 运行\n\n'+decision['status']+
        '\n\n阶段状态：T1-02='+decision['t1_02_training_state_consistency']+
        '；T1-03='+decision['t1_03_flops_accounting']+'；T1-04='+decision['t1_04_engineering_validation']+
        '；T1-05='+decision['t1_05_formal_single_seed']+
        '\n\n形式训练最多 250,000 Prediction Targets，单 Seed。H1–H5 保持 NOT_EVALUATED；Independent Test 未访问；权重和 Checkpoint 不进入 Git。\n')
    artifacts=prepare_artifacts(root,experiment,EXPERIMENTS[experiment],None)
    branch='kaggle/'+experiment+'-'+decision['identity']['source'][:12]
    push_results(root,artifacts,experiment,branch,'GITHUB_TOKEN','origin/main')
    print(json.dumps({'stage':'RESULTS_PUBLISHED','branch':branch,'remote_sha':'VERIFIED','status':decision['status']}),flush=True)


def launch(out):
    import torch
    from storage import recover,bundle
    from t1_evaluation import sync
    from generation_evaluation import audit_data
    from native_clm_runtime import load_tokenizer
    from repair_io import timed_budget
    from cached_transformer import CachedTransformer
    from preparation import parameter_count
    from pathlib import Path
    from datetime import datetime,timezone
    from shutil import disk_usage
    from hashlib import sha256
    from torch.nn import functional as F
    from t1_evaluation import evaluate_documents,generation_diagnostics,inference_profile
    from t1_validation import cache_equivalence_and_invalidation,training_reference
    from formal_training import FormalTrainer,verify_resume
    from flops_audit import ArithmeticCounter,profiler_inventory
    from native_clm_runtime import sha256_file

    from scripts.research.publish_experiment_results import validate_origin,load_github_token
    validate_origin(ROOT);load_github_token('GITHUB_TOKEN')
    protocol=json.loads(PROTOCOL_PATH.read_text());registry=json.loads((ROOT/'research/governance/formal_seed_registry.json').read_text())
    config=validate_freeze(protocol,registry);source=assert_clean_source();protocol_sha=sha256_file(PROTOCOL_PATH);env=environment()
    decision=initial_decision(source,protocol_sha,env)
    started=time.monotonic();deadline=started+protocol['budget']['execution_seconds'];out.mkdir(parents=True,exist_ok=True)
    original_report=json.loads((out/'decision.json').read_text()) if (out/'decision.json').exists() else None
    placeholders(out)
    decision.update({'identity':{'source':source,'protocol':protocol_sha},'source_branch':'codex/gc002-t1-formal-readiness',
        'run_id':'gc002-t1-formal-'+source[:12]+'-s'+str(protocol['seed']),'protocol':protocol,
        'formal_hypotheses':dict.fromkeys(('H1','H2','H3','H4','H5'),'NOT_EVALUATED')})
    if original_report is None:persist(out,decision)
    stage='RAW_ASSET_VERIFICATION';assets=None;engine=None
    try:
        if not torch.cuda.is_available():raise RuntimeError('T1-02..05 requires a CUDA GPU; CPU training is not enabled')
        if torch.cuda.get_device_name()!='Tesla T4':raise RuntimeError('T1 formal protocol was approved on Tesla T4; target GPU differs')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if env['precision']!='fp32' or env['tf32'] or env['autocast'] or env['grad_scaler']:raise ValueError('runtime precision differs from frozen FP32 config')
        if disk_usage(out).free<2_000_000_000:raise RuntimeError('insufficient local disk space for private checkpoints and recovery objects')
        assets,tokenizer,groups,data_audit,input_proof,asset_hashes=_validate_assets(ROOT,out,protocol,deadline)
        if asset_hashes['manifest.json']!=protocol['data']['dataset_manifest_sha256']:raise ValueError('data manifest identity mismatch')
        run_identity={'source':source,'protocol':protocol_sha,'dataset':asset_hashes['manifest.json'],
            'tokenizer':asset_hashes['tokenizer.json'],'train_json':asset_hashes['train.json'],
            'development_json':asset_hashes['development.json'],'seed':protocol['seed'],
            'environment':env,'mode':'GC002_T1_PM_KV_FORMAL_SINGLE_SEED'}
        expected_report_identity={**run_identity,'run_id':decision['run_id']}
        with timed_budget(120):recover(ROOT,out,source,protocol_sha,cache=out.parent/'gc002-t1-formal-recovery-cache',max_versions=3)
        recovered=out/'recovery.json';recovery_status='LOCAL_OR_REMOTE_EXACT_SOURCE'
        if not recovered.exists():recovery_status='NO_PRIOR_FORMAL_CHECKPOINT'
        restored_report=json.loads((out/'decision.json').read_text()) if (out/'decision.json').exists() else {}
        if restored_report.get('identity')==expected_report_identity:
            decision=restored_report
        elif original_report and (original_report.get('formal_progress',{}).get('committed_tokens',0)>0 or
                original_report.get('t1_05_formal_single_seed') in ('IN_PROGRESS','COMPLETE','INCOMPLETE')):
            raise RuntimeError('prior T1 formal output has progress under another identity; refusing duplicate training')
        decision['identity']=expected_report_identity;decision['source_branch']='codex/gc002-t1-formal-readiness'
        decision['protocol']=protocol;decision['data_audit']=data_audit;decision['asset_recovery']=input_proof
        decision['artifact_hashes']=asset_hashes
        decision['checkpoint_recovery_search']={'status':recovery_status,'max_versions':3,'checkpoint_identity':run_identity}
        if decision.get('status')=='NOT_RUN':decision['status']='IN_PROGRESS'
        persist(out,decision)
        if (out/'private/training.pt').exists():
            from trainer import load_checkpoint
            safe=load_checkpoint(out/'private/training.pt')
            expected_checkpoint_identity={**run_identity,'mode':'T1-05_FORMAL_SINGLE_SEED','target_tokens':250000}
            if safe.get('identity')!=expected_checkpoint_identity:
                raise ValueError('local checkpoint has a different source/protocol/data/seed/runtime identity')
        elif decision.get('formal_progress',{}).get('committed_tokens',0)>0:
            raise RuntimeError('progress report has no recoverable matching checkpoint; refusing to retrain')
        stage='T1_02_TRAINING_STATE_CONSISTENCY'
        if decision['t1_02_training_state_consistency']!='PASS':
            cache_report=cache_equivalence_and_invalidation(config,'cuda');reference=training_reference(config,protocol['seed'],'cuda')
            if cache_report['status']!='PASS' or reference['status']!='PASS':raise ValueError('T1-02 cache or training-reference checks failed')
            write_json(out/'t1-02-kv-consistency/training-semantics.json',{'status':'PASS','training':'parallel full-sequence teacher forcing',
                'max_sequence_input_tokens':128,'context_overlap_tokens':64,'cross_optimizer_training_kv_cache':False,
                'position_ids':'absolute within document','loss_mask':'only fresh next-token targets','precision':'fp32'})
            write_json(out/'t1-02-kv-consistency/training-reference-tests.json',reference)
            write_json(out/'t1-02-kv-consistency/kv-cache-equivalence.json',cache_report['cache_equivalence'])
            write_json(out/'t1-02-kv-consistency/cache-version-audit.json',{'status':'PASS','cache_weight_version_invalidation':'PASS',
                'snapshot_restore':'PASS','max_absolute_logit_error':max(row['max_absolute_logit_error'] for row in cache_report['cache_equivalence']['cases']),
                'tolerance':1e-5})
            write_json(out/'t1-02-kv-consistency/document-reset-tests.json',{'status':'PASS','training_path':reference['document_boundary_reset'],
                'input_document_boundaries_crossed':False,'loss_reference_error':reference['loss_reference_absolute_error']})
            write_json(out/'t1-02-kv-consistency/checkpoint-state-tests.json',{'status':'PENDING_T1_04','checkpoint_resume':'NOT_RUN'})
            stage_decision(out,'t1-02-kv-consistency','PASS',new_formal_prediction_tokens=0)
            decision['t1_02_training_state_consistency']='PASS';decision['model_parameters']=parameter_count(config);persist(out,decision)

        stage='T1_03_FLOPS_ACCOUNTING'
        if decision['t1_03_flops_accounting']!='PASS':
            arithmetic,coverage=_t1_03(ROOT,protocol,run_identity,'cuda',out,groups['train'])
            decision['t1_03_flops_accounting']='PASS';decision['formal_compute_matching_ready']=coverage['formal_compute_matching_ready']
            decision['formal_compute_matching_status']='READY' if coverage['formal_compute_matching_ready'] else 'NOT_READY'
            decision['t1_03_compute_scope']=coverage['scope'];persist(out,decision)

        stage='T1_04_ENGINEERING_VALIDATION'
        if decision['t1_04_engineering_validation']!='PASS':
            write_json(out/'t1-04-engineering-validation/model-validation.json',{'status':'PASS','parameter_count':parameter_count(config),
                'fp32':'PASS','forward_backward_optimizer':'PASS','causal_mask':'PASS','eos_id':3,'document_reset':'PASS','context_window':128,
                'cache_equivalence':'PASS','cache_invalidation':'PASS'})
            write_json(out/'t1-04-engineering-validation/frozen-training-config.json',protocol)
            write_json(out/'t1-04-engineering-validation/environment.json',env)
            stage_decision(out,'t1-04-engineering-validation','RUNNING')
            smoke_engine,smoke=_smoke(protocol,run_identity,groups,out,deadline,run_identity)
            if smoke['status']!='PASS':
                decision['t1_04_engineering_validation']='INCOMPLETE';decision['t1_05_formal_single_seed']='BLOCKED';decision['status']='INCOMPLETE_T1_04_SMOKE'
                write_json(out/'t1-04-engineering-validation/smoke-training.json',smoke);persist(out,decision)
                raise TimeoutError(smoke.get('reason','T1-04 smoke incomplete'))
            smoke['training_seconds']=sum(row['seconds'] for row in smoke_engine.metrics)
            write_json(out/'t1-04-engineering-validation/smoke-training.json',smoke)
            write_json(out/'t1-04-engineering-validation/checkpoint-resume.json',smoke['resume'])
            write_json(out/'t1-04-engineering-validation/checkpoint-state-tests.json',smoke['resume'])
            write_json(out/'t1-02-kv-consistency/checkpoint-state-tests.json',smoke['resume'])
            write_json(out/'t1-04-engineering-validation/data-ledger-validation.json',smoke_engine.ledger.report())
            write_json(out/'t1-04-engineering-validation/flops-ledger-validation.json',_flops_report(smoke_engine))
            write_json(out/'t1-04-engineering-validation/gpu-profile.json',_profile_summary(smoke_engine,smoke,protocol))
            if any(row['unsupported_operators'] for row in smoke_engine.flops_updates):raise ValueError('T1-04 live smoke has unaccounted operators')
            if _flops_report(smoke_engine)['status']!='PASS':raise ValueError('T1-04 arithmetic ledger failed')
            stage_decision(out,'t1-04-engineering-validation','PASS',smoke_tokens=smoke_engine.ledger.consumed)
            decision['t1_04_engineering_validation']='PASS';decision['smoke_profile']=json.loads((out/'t1-04-engineering-validation/gpu-profile.json').read_text());persist(out,decision)
            decision['engineering_status']='ENGINEERING_VALIDATED';persist(out,decision)
            bundle(out,run_identity)
        if not all(decision[key]=='PASS' for key in ('t1_02_training_state_consistency','t1_03_flops_accounting','t1_04_engineering_validation')):
            raise RuntimeError('sequential T1 gates are not all PASS; formal training remains blocked')

        stage='T1_05_FORMAL_SINGLE_SEED';folder=out/'t1-pm-formal-single-seed';folder.mkdir(parents=True,exist_ok=True)
        checkpoint=out/'private/training.pt';final_model=out/'private/final-model.pt'
        formal_identity={**run_identity,'mode':'T1-05_FORMAL_SINGLE_SEED','target_tokens':250000}
        if decision['t1_05_formal_single_seed']=='COMPLETE':
            if not checkpoint.exists() or not final_model.exists():raise RuntimeError('complete report lacks its private checkpoint/model artifact')
            saved_manifest=json.loads((folder/'final-model-manifest.json').read_text())
            checkpoint_manifest=json.loads((folder/'checkpoint-manifest.json').read_text())
            if sha256_file(final_model)!=saved_manifest['model']['sha256']:raise RuntimeError('completed final model artifact checksum differs')
            if sha256_file(checkpoint)!=checkpoint_manifest['checkpoint']['sha256']:raise RuntimeError('completed checkpoint checksum differs')
            decision['status']='FORMAL_TRAINING_COMPLETE';persist(out,decision)
        else:
            from trainer import load_checkpoint
            last_report=decision.get('formal_progress',{})
            write_json(folder/'training-config.json',protocol)
            if checkpoint.exists():engine=FormalTrainer.load(checkpoint,groups['train'],formal_identity,protocol,'cuda',250000)
            elif last_report.get('committed_tokens',0)>0:raise RuntimeError('formal progress exists without a same-source recoverable checkpoint')
            else:engine=FormalTrainer(groups['train'],formal_identity,protocol,'cuda',250000)
            if engine.ledger.consumed>250000 or engine.ledger.updates>977:raise ValueError('formal checkpoint exceeds frozen token/update budget')
            smoke_profile=json.loads((out/'t1-04-engineering-validation/gpu-profile.json').read_text())
            smoke_tps=smoke_profile['overall_smoke_tokens_per_second'];remaining=250000-engine.ledger.consumed
            projected=remaining/max(smoke_tps*.7,1.)+300
            if engine.ledger.consumed==0 and time.monotonic()+projected>deadline:
                raise TimeoutError('not enough execution budget remains for the complete formal run; no formal token consumed')
            torch.cuda.reset_peak_memory_stats();decision['t1_05_formal_single_seed']='IN_PROGRESS'
            decision['formal_training_status']='IN_PROGRESS';decision['status']='FORMAL_TRAINING_IN_PROGRESS';persist(out,decision)
            while engine.ledger.consumed<250000:
                remaining=250000-engine.ledger.consumed
                projected=remaining/max(smoke_tps*.65,1.)+180
                if time.monotonic()+projected>deadline:
                    _save_formal_checkpoint(engine,checkpoint,folder,out,decision,formal_identity,run_identity)
                    decision['t1_05_formal_single_seed']='INCOMPLETE';decision['formal_training_status']='INCOMPLETE'
                    decision['status']='INCOMPLETE_FORMAL_BUDGET';persist(out,decision)
                    bundle(out,run_identity)
                    raise TimeoutError('formal run checkpointed at a committed update boundary; resume from the exact same source')
                row,_=engine.update()
                if row['prediction_tokens'] in (256,512,250000) or row['update']%25==0:
                    print(json.dumps({'stage':'FORMAL_T1_UPDATE','update':row['update'],'tokens':row['prediction_tokens'],
                        'loss':row['loss'],'tokens_per_second':row['tokens_per_second']}),flush=True)
                if engine.ledger.updates%protocol['checkpoint']['formal_interval_updates']==0 or engine.ledger.consumed==250000:
                    decision['latest_formal_update']={'update':row['update'],'tokens':row['prediction_tokens'],
                        'loss':row['loss'],'tokens_per_second':row['tokens_per_second']}
                    _save_formal_checkpoint(engine,checkpoint,folder,out,decision,formal_identity,run_identity)
            if (engine.ledger.consumed,engine.ledger.updates)!=(250000,977):raise ValueError('formal training did not stop at exactly 250,000 targets / 977 updates')
            engine.save(checkpoint)
            cpu_state={name:value.detach().cpu() for name,value in engine.model.state_dict().items()}
            temporary=final_model.with_suffix('.pt.tmp');torch.save({'format':'gc002.t1-pm-kv-final-model.v1','identity':formal_identity,
                'config':engine.config.__dict__,'model':cpu_state,'model_weight_identity':engine.model.weight_identity()},temporary);temporary.replace(final_model)
            decision['formal_progress']={'committed_tokens':250000,'optimizer_updates':977,'checkpoint_sha256':sha256_file(checkpoint)}
            training_seconds=sum(row['seconds'] for row in engine.metrics);training_wall=training_seconds+engine.checkpoint_seconds
            write_json(folder/'training-metrics.json',{'status':'COMPLETE','metrics':engine.metrics,
                'training_seconds':training_seconds,'checkpoint_seconds':engine.checkpoint_seconds,
                'training_wall_seconds_including_checkpoints':training_wall,
                'update_compute_tokens_per_second':250000/max(training_seconds,1e-12),
                'tokens_per_second_including_checkpoint_io':250000/max(training_wall,1e-12)})
            write_json(folder/'training-config.json',protocol);write_json(folder/'token-exposure-ledger.json',engine.ledger.report())
            write_json(folder/'flops-ledger.json',_flops_report(engine))
            write_json(folder/'environment.json',env)
            write_json(folder/'checkpoint-manifest.json',{'identity':formal_identity,'checkpoint':{'path':'private/training.pt',
                'sha256':sha256_file(checkpoint),'bytes':checkpoint.stat().st_size,'format':'gc002.t1-formal-state.v1'},
                'retrieval':'download this file from the Kaggle Notebook Output; checksum and internal envelope SHA are verified on resume'})
            write_json(folder/'final-model-manifest.json',{'identity':formal_identity,'model':{'path':'private/final-model.pt',
                'sha256':sha256_file(final_model),'bytes':final_model.stat().st_size,'format':'gc002.t1-pm-kv-final-model.v1'},
                'model_weight_identity':engine.model.weight_identity(),'git_storage':'NOT_COMMITTED'})
            bundle(out,run_identity)

        if engine is None:
            engine=FormalTrainer.load(checkpoint,groups['train'],formal_identity,protocol,'cuda',250000)
        if engine.ledger.consumed!=250000 or engine.ledger.updates!=977:raise ValueError('refusing evaluation from a non-complete formal training checkpoint')
        model=engine.model.eval();sync(model)
        if not (folder/'validation-results.json').exists():write_json(folder/'validation-results.json',evaluate_documents(model,groups['validation'],'cuda'))
        if not (folder/'holdout-results.json').exists():write_json(folder/'holdout-results.json',evaluate_documents(model,groups['development_holdout'],'cuda'))
        if not (folder/'generation-metrics.json').exists():write_json(folder/'generation-metrics.json',generation_diagnostics(model,tokenizer,protocol,'cuda'))
        if not (folder/'inference-profile.json').exists():write_json(folder/'inference-profile.json',inference_profile(model,tokenizer,protocol,'cuda'))
        write_json(folder/'memory-profile.json',{'status':'MEASURED','peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'kv_cache_bytes_per_session':engine.config.layers*2*engine.config.window*engine.config.kv_heads*(engine.config.dim//engine.config.heads)*4,
            'weights_shared_and_reported_separately':True})
        manifest={'format':'minicells.gc002.t1-formal-manifest.v1','run_id':decision['run_id'],'identity':formal_identity,
            'training_tokens':250000,'optimizer_updates':977,'seed':protocol['seed'],'active_parameters':parameter_count(engine.config),
            'source_commit':source,'protocol_sha256':protocol_sha,'dataset_sha256':asset_hashes['manifest.json'],
            'tokenizer_sha256':asset_hashes['tokenizer.json'],'training_order_sha256':engine.ledger.order_hash,
            'training_checkpoint_sha256':sha256_file(checkpoint),'final_model_sha256':sha256_file(final_model),
            'independent_test_accessed':False,'GC001_results_modified':False}
        write_json(folder/'manifest.json',manifest)
        research={'status':'FORMAL_SINGLE_SEED_COMPLETE','formal_training_complete':True,'formal_compute_matching_ready':True,
            'scientific_result_confirmed':False,'formal_hypotheses':'NOT_EVALUATED','human_review':'PENDING',
            'seed_count':1,'single_seed_limitation':'no between-seed variance; no scientific superiority or equivalence conclusion',
            'independent_test_accessed':False,'training_budget_met_exactly':engine.ledger.consumed==250000,
            'cross_model_track_c_budget':'NOT_FROZEN_BY_THIS_T1_ONLY_RUN','evaluation_scope':'validation and development holdout only'}
        write_json(folder/'research-decision.json',research)
        decision.update(status='FORMAL_TRAINING_COMPLETE',t1_05_formal_single_seed='COMPLETE',formal_training_complete=True,
            formal_training_status='FORMAL_TRAINING_COMPLETE',formal_compute_matching_status='READY',
            formal_compute_matching_ready=True,scientific_result_status='NOT_CONFIRMED',scientific_result_confirmed=False,
            formal_hypotheses=dict.fromkeys(('H1','H2','H3','H4','H5'),'NOT_EVALUATED'),
            human_review='PENDING',independent_test_accessed=False,GC001_results_modified=False)
        (out/'summary.md').write_text('# GC002 T1-PM-KV 正式单 Seed 实验\n\nT1-02、T1-03、T1-04 全部通过；T1-05 恰好完成 250,000 个有效 Prediction Targets。单 Seed 工程/正式运行完成不代表科学假设得到确认。\n')
        persist(out,decision);bundle(out,run_identity)
    except Exception as error:
        decision['status']='INCOMPLETE';decision['error']={'stage':stage,'type':type(error).__name__,'message':str(error)}
        decision['failed_stage']=stage
        if decision.get('t1_05_formal_single_seed')=='IN_PROGRESS':
            decision['t1_05_formal_single_seed']='INCOMPLETE';decision['formal_training_status']='INCOMPLETE'
        gate_dirs={'T1_02_TRAINING_STATE_CONSISTENCY':'t1-02-kv-consistency',
            'T1_03_FLOPS_ACCOUNTING':'t1-03-flops-accounting','T1_04_ENGINEERING_VALIDATION':'t1-04-engineering-validation'}
        if stage in gate_dirs:
            status='INCOMPLETE' if isinstance(error,TimeoutError) else 'FAIL'
            field={'T1_02_TRAINING_STATE_CONSISTENCY':'t1_02_training_state_consistency',
                'T1_03_FLOPS_ACCOUNTING':'t1_03_flops_accounting','T1_04_ENGINEERING_VALIDATION':'t1_04_engineering_validation'}[stage]
            decision[field]=status;decision['t1_05_formal_single_seed']='BLOCKED'
            decision['formal_training_status']='BLOCKED';decision['formal_training_complete']=False
            if field=='t1_04_engineering_validation':decision['engineering_status']='ENGINEERING_VALIDATION_FAILED'
        persist(out,decision)
        if stage in gate_dirs:stage_decision(out,gate_dirs[stage],decision[field],error=decision['error'])
    finally:
        decision['execution_seconds']=time.monotonic()-started
        persist(out,decision)
        if 'run_identity' in locals():
            try:bundle(out,run_identity)
            except Exception as bundle_error:
                decision['recovery_bundle']={'status':'FAILED','error':type(bundle_error).__name__+': '+str(bundle_error)}
                persist(out,decision)
        try:
            with timed_budget(protocol['budget']['publication_seconds']):_publish(ROOT,out,decision)
        except Exception as publish_error:
            decision['publication']={'status':'FAILED','error':type(publish_error).__name__+': '+str(publish_error),
                'kaggle_output_preserved':True};persist(out,decision)
            if 'run_identity' in locals():
                try:bundle(out,run_identity)
                except Exception:pass
            raise
    print(json.dumps({'status':decision['status'],'t1_02':decision['t1_02_training_state_consistency'],
        't1_03':decision['t1_03_flops_accounting'],'t1_04':decision['t1_04_engineering_validation'],
        't1_05':decision['t1_05_formal_single_seed'],'training_tokens':decision.get('formal_progress',{}).get('committed_tokens',0),
        'publication':decision.get('publication'),'report':str(out/'decision.json')}),flush=True)
    return decision


def main():
    parser=argparse.ArgumentParser(description='Sequential T1-02 through conditional 250K T1-PM-KV formal run')
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/epclm-gc002-t1-formal'))
    launch(parser.parse_args().output)


if __name__=='__main__':main()
