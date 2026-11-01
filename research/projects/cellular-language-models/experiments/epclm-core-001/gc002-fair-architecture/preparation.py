"""Phase 0/1 engineering only. Deliberately contains no training entrypoint."""
from dataclasses import asdict
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
import torch
from cached_transformer import CachedTransformer,Config,cache_memory
from native_clm_runtime import ModernTransformerLM
from batch_c_analysis import PINS
from repair_io import public_result


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def parameter_count(c):
    # One tied embedding/head storage; per-block q,k,v,o; SwiGLU; two RMSNorm.
    return c.vocab*c.dim+c.layers*(2*c.dim*c.dim+2*c.dim*c.kv_heads*(c.dim//c.heads)+3*c.dim*c.ffn+2*c.dim)+c.dim


def parameter_match():
    target=966784;candidates=[]
    for dim in (96,128,144,160,192):
        for layers in (2,3,4):
            for head_dim in (24,32,48,64):
                if dim%head_dim:continue
                heads=dim//head_dim
                for kv in (1,2):
                    if heads%kv:continue
                    for ffn in range(2*dim,4*dim+1,16):
                        c=Config(dim=dim,layers=layers,heads=heads,kv_heads=kv,ffn=ffn)
                        count=parameter_count(c);error=abs(count/target-1)
                        if error<=.01:candidates.append((error,abs(layers-4),abs(dim-192),asdict(c),count))
    candidates.sort(key=lambda r:(r[:3],json.dumps(r[3],sort_keys=True)))
    if not candidates:raise ValueError('no structural candidate within 1%; no silent tolerance relaxation')
    selected=Config(**candidates[0][3]);torch.manual_seed(713);model=CachedTransformer(selected)
    named=list(model.named_parameters());total=sum(p.numel() for _,p in named)
    assert total==parameter_count(selected)
    # A four-token synthetic gradient reachability check; no optimizer or update.
    model(torch.tensor([[1,17,29,3]])).square().mean().backward()
    inactive=[name for name,p in named if p.grad is None or not torch.isfinite(p.grad).all() or not torch.count_nonzero(p.grad)]
    if inactive:raise ValueError('candidate has inactive/nonfinite parameters: '+repr(inactive))
    modules={}
    for name,p in named:
        key=name.rsplit('.',1)[0];modules[key]=modules.get(key,0)+p.numel()
    return selected,{'status':'PARAMETER_MATCH_CANDIDATE_PASS_NOT_FROZEN','target_active_parameters':target,'config':asdict(selected),'total_parameters':total,'active_trainable_parameters':total,'nontrainable_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),'relative_difference_percent':100*(total/target-1),'tolerance_percent':1.,'tied_embeddings_counted_once':True,'head_alias':'lm_head.weight aliases token_embedding.weight; no second allocation','module_parameters':modules,'gradient_reachability':'PASS_SYNTHETIC_FOUR_TOKENS_NO_OPTIMIZER','optimizer_updates':0,'search':'parameter counts only; no accuracy selection','candidate_count':len(candidates),'top_candidates':[{'config':r[3],'parameters':r[4],'relative_error':r[0]} for r in candidates[:8]]}


def equivalence(config,device='cpu'):
    torch.manual_seed(713);model=CachedTransformer(config).to(device).eval()
    ids=torch.randint(4,config.vocab,(1,385),device=device);ids[0,[31,127,256]]=3
    cases=[]
    with torch.inference_mode():
        # Tied weights and historical behavior agree before any window truncation.
        if config==Config():
            native=ModernTransformerLM(config.vocab).to(device).eval();native.load_state_dict(model.state_dict(),strict=True)
            historical_error=float((native(ids[:,:128])-model(ids[:,:128])).abs().max())
            if historical_error>1e-5:raise ValueError('historical pre-window forward differs')
        else:historical_error=None
        for length in (32,128,385):
            current=ids[:,:length];uncached=model(current)
            cache=None;pieces=[];seen=[]
            for position in range(length):
                value,cache=model.cached(current[:,position:position+1],cache);pieces.append(value);seen.append(cache.next_position)
            cached=torch.cat(pieces,1);error=float((uncached-cached).abs().max())
            same=bool(torch.equal(uncached.argmax(-1),cached.argmax(-1)))
            if error>1e-5 or not same or seen!=list(range(1,length+1)):raise ValueError('cache equivalence/absolute position failed')
            cases.append({'length':length,'max_absolute_logit_error':error,'argmax_identical':same,'sliding_window_exercised':length>config.window,'eos_id_in_input':3})
        # Multi-token prefill and chunked suffix must agree across eviction too.
        first,state=model.cached(ids[:,:128]);second,state=model.cached(ids[:,128:201],state)
        third,state=model.cached(ids[:,201:],state)
        chunk_error=float((model(ids)-torch.cat((first,second,third),1)).abs().max())
        if chunk_error>1e-5:raise ValueError('chunked prefill/eviction equivalence failed')
        restored=model.restore(model.snapshot(state));next_id=torch.tensor([[29]],device=device)
        a,_=model.cached(next_id,state);b,_=model.cached(next_id,restored)
        if not torch.equal(a,b):raise ValueError('cache serialization continuation differs')
        first_logits,_=model.cached(ids[:,:32]);synthetic_eos=int(first_logits[0,-1].argmax())
        generated,_=model.generate(ids[:,:32],5,synthetic_eos)
        if generated!=[synthetic_eos]:raise ValueError('EOS stop policy failed')
        memory=cache_memory(model,state)
        if memory['kv_tensor_backing_storage_bytes']!=memory['kv_tensor_payload_bytes']:raise ValueError('cache retained oversized storage')
        if memory['persistent_state_bytes_per_stream']!=memory['theoretical_full_window_gqa_bytes_per_stream']:raise ValueError('cache byte count differs')
    return {'status':'PASS','device':str(device),'dtype':'fp32','batch':1,'weights':'SYNTHETIC_RANDOM_NOT_HISTORICAL_CHECKPOINT','model_config':asdict(config),'model_sha256':model.weight_identity(),'semantics':'causal sliding window from sequence start; absolute RoPE IDs; GC001 replay differs after eviction','historical_weight_loading':'STRICT_NATIVE_STATE_DICT_COMPATIBLE' if config==Config() else 'NEW_CANDIDATE','historical_pre_window_max_error':historical_error,'max_error_tolerance':1e-5,'cases':cases,'chunked_prefill_max_error':chunk_error,'snapshot_continuation_bitwise_equal':True,'eos_stop':'PASS_SYNTHETIC_PREDICTED_STOP_ID; actual EOS ID 3 also included in equivalence inputs','memory':memory,'optimizer_updates':0}


def baseline_audit(root,base):
    reports={};proofs={}
    protocol='a6f56ef1a0b1497643a113b65dbbb5781cf51336d522b159b1ea5e964ea5ad3c'
    for batch,(source,commit,digest) in PINS.items():
        artifact='artifacts/experiments/epclm-core-001/gc001-baseline-repair/batch-'+batch
        try:raw=subprocess.check_output(['git','show',commit+':'+artifact+'/decision.json'],cwd=root,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            public_result(root,'kaggle/epclm-gc001-repair-'+batch+'-'+source[:12],artifact,source,protocol,commit=commit)
            raw=subprocess.check_output(['git','show',commit+':'+artifact+'/decision.json'],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('pinned historical report hash differs')
        report=json.loads(raw)
        if report['status']!='COMPLETE' or report['identity']['source']!=source or report['identity']['protocol']!=protocol:raise ValueError('historical report identity differs')
        reports[batch]=report;proofs[batch]={'commit':commit,'source':source,'decision_sha256':digest}
    a,b=reports['a'],reports['b']
    if a['data_audit']!=b['data_audit']:raise ValueError('historical data audits differ')
    models={}
    for name,entry in [(n,a['models'][n]) for n in ('C1-BROADCAST','IC-ALL-SELF','F0-PM-BROADCAST')]+[('T1',b['models']['T1'])]:
        t=entry['training'];identity=entry['training_identity'];curve=entry['training_diagnostics']['curve']
        if t['tokens']!=250000 or t['updates']!=977 or identity['precision']!='fp32' or identity['seed']!=173018021:raise ValueError('historical budget/precision/seed differs')
        if len(curve)!=977 or curve[0]['tokens']!=256 or curve[-1]['tokens']!=250000:raise ValueError('historical committed curve incomplete')
        delta=[r['tokens']-(curve[i-1]['tokens'] if i else 0) for i,r in enumerate(curve)]
        if delta!=[256]*976+[144]:raise ValueError('historical committed token increments differ')
        models[name]={'active_parameters':entry['architecture']['active_trainable_parameters'],'consumed_training_tokens':t['tokens'],'optimizer_updates':t['updates'],'tokens_per_update':{'full_updates':976,'full_update_tokens':256,'last_update_tokens':144},'training_wall_seconds':t['elapsed_seconds'],'throughput_tokens_per_second':t['tokens']/t['elapsed_seconds'],'last_trainer_segment':{k:t[k] for k in ('resumed_from_tokens','resumed_from_updates','new_tokens')},'seed':identity['seed'],'checkpoint_sha256':entry['checkpoint_sha256'],'model_sha256':entry['weights_sha256'],'full_curve_token_counts_verified':True,'unique_training_tokens':None,'repeated_training_tokens':None,'training_epochs':None,'consumed_data_order_hash':None,'unknown_reason':'raw document/token traversal and actual per-update input hashes not published; no inference from dataset size alone','execution_policy':reports['b' if name=='T1' else 'a'].get('execution_policy')}
    actual=sum(p.numel() for p in ModernTransformerLM().parameters())
    if actual!=1894080 or models['T1']['active_parameters']!=actual:raise ValueError('Native T1 parameter count differs')
    return {'status':'BASELINE_METADATA_AUDIT_PASS_RAW_ASSETS_PENDING','sources':proofs,'models':models,'data_audit':a['data_audit'],'dataset_prediction_tokens':416027,'train_budget_tokens':250000,'token_accounting':'committed input-target prediction events, not available corpus size, not lexical unique token IDs','last_segment_explanation':'learning() invokes train() at updates 250/500/750/977. Final segment restores current-run milestone 750 (192000 tokens); 58000 new tokens complete 250000. This field alone does not show cross-run reuse.','fresh_cross_run_claim':'POLICY_RECORDED_NOT_INDEPENDENTLY_PROVEN_FROM_PRIVATE_CHECKPOINTS','wall_time_scope':'learning() sums segment elapsed_seconds; checkpoint/restore fields are last-segment fields, not whole-run totals','kv_missing_reason':'historical TransformerReference.step/read replays each lane context each token; no cache; RoPE IDs restart from 0 on each cropped replay','training_speed_caveat':'GC001 teacher-forced trainer also replays Transformer prefixes per event/TBPTT; KV inference optimization is not a remedy for normal parallel teacher-forced training','gqa_correction':{'query_heads':6,'kv_heads':2,'head_dim':32,'compact_fp32_cache_bytes':262144,'expanded_mha_equivalent_bytes':786432,'old_token_context_wrapper_bytes':516,'wrapper_is_not_kv_cache':True},'tokenizer_audit':{'status':'RAW_BYTES_NOT_AVAILABLE_NOT_VERIFIED','dataset_manifest_sha256_recorded':a['identity']['data'],'tokenizer_file_sha256':None,'ordered_document_hashes_recorded':a['data_audit']['splits']},'baseline_audit_gate':'PENDING_RAW_DATA_TOKENIZER_AND_CONSUMPTION_ORDER_VERIFICATION','historical_hypotheses':'H1_H5_NOT_EVALUATED','human_review':'PENDING_AI_REVIEW_IS_NOT_HUMAN'}


def gpu_profile(config):
    if not torch.cuda.is_available():return {'status':'NOT_MEASURED_GPU_REQUIRED','cpu_benchmark_performed':False,'formal_performance_eligible':False}
    torch.manual_seed(713);model=CachedTransformer(config).cuda().eval();rows=[]
    with torch.inference_mode():
        for length in (32,128,512,1024):
            ids=torch.randint(4,config.vocab,(1,length),device='cuda')
            for _ in range(3):model.cached(ids)
            torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            prefill=[];decode=[]
            for _ in range(20):
                torch.cuda.synchronize();started=time.perf_counter();_,cache=model.cached(ids);torch.cuda.synchronize();prefill.append(time.perf_counter()-started)
                token=torch.tensor([[17]],device='cuda');torch.cuda.synchronize();started=time.perf_counter();_,after=model.cached(token,cache);torch.cuda.synchronize();decode.append(time.perf_counter()-started)
            rows.append({'history_tokens':length,'prefill_median_seconds':statistics.median(prefill),'prefill_p95_seconds':sorted(prefill)[math.ceil(.95*len(prefill))-1],'decode_median_seconds':statistics.median(decode),'decode_p95_seconds':sorted(decode)[math.ceil(.95*len(decode))-1],'decode_tokens_per_second':1/statistics.median(decode),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'baseline_cuda_allocated_bytes':baseline,'total_inference_memory_bytes':None,'total_runtime_memory_status':'CUDA_COMPONENT_MEASURED_HOST_DRIVER_UNATTRIBUTED','runtime_memory_scope':'process-wide CUDA tensor allocations incl model and temporary workspace; Python/driver overhead separately unknown','persistent_memory':cache_memory(model,after)})
    return {'status':'MEASURED_RANDOM_WEIGHTS_ENGINEERING_ONLY','formal_performance_eligible':False,'rows':rows,'precision':'fp32','gpu':torch.cuda.get_device_name(),'flops_per_token':None,'flops_status':'FULL_OPERATOR_FORWARD_BACKWARD_OPTIMIZER_ACCOUNTING_NOT_IMPLEMENTED'}


def prepare(root,base,out,device,recover_assets=False):
    torch.set_num_threads(1)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c,matching=parameter_match();audit=baseline_audit(root,base)
    if recover_assets:
        # Reuse the already verified known-version, data-only path. Never load weights.
        from repair_io import recover_inputs,timed_budget
        from native_clm_runtime import load_tokenizer,sha256_file
        from generation_evaluation import audit_data
        history_plan=json.loads((base.parent/'gc001-baseline-repair/protocol.json').read_text())['gc_protocol']
        cache=out/'private-input-assets'
        with timed_budget(120):proof=recover_inputs(root,cache,history_plan,[142],time.monotonic()+120,include_weights=False)
        _,raw_audit=audit_data(cache,load_tokenizer(cache/'assets/tokenizer.json'),history_plan)
        if raw_audit!=audit['data_audit']:raise ValueError('raw data audit differs from frozen historical reports')
        audit['tokenizer_audit'].update(status='RAW_FILE_HASHES_AND_TOKEN_CONTRACTS_VERIFIED',tokenizer_file_sha256=sha256_file(cache/'assets/tokenizer.json'),input_recovery=proof)
        audit['status']='BASELINE_METADATA_AND_RAW_ASSETS_PASS_EXPOSURE_ORDER_PENDING'
        audit['baseline_audit_gate']='PENDING_HISTORICAL_CONSUMPTION_ORDER_VERIFICATION'
    from readiness import previous_evidence,readiness
    previous,provenance=previous_evidence(root,base)
    if previous['parameter-match-report.json']['result']['config']!=asdict(c):raise ValueError('parameter candidate changed from measured reference')
    cached=previous['kv-cache-equivalence.json']['result']
    if not recover_assets:
        audit['tokenizer_audit']=previous['baseline-audit.json']['result']['tokenizer_audit']
        audit['raw_asset_evidence_scope']='hashes verified by pinned prior run, not reread locally'
    audit['status']='AUDIT_COMPLETE_WITH_DOCUMENTED_HISTORICAL_LIMIT'
    audit['baseline_audit_gate']='HISTORICAL_RUNTIME_ORDER_UNVERIFIABLE_NEW_RUN_LOGGING_REQUIRED'
    audit['original_phase0_full_pass_claimed']=False
    from exposure import reconstruct
    if recover_assets:
        stream=reconstruct(json.loads((cache/'assets/train.json').read_text()),173018021)
        if stream['expected']['consumed_training_tokens']!=250000 or stream['expected']['optimizer_updates']!=977:raise ValueError('reconstructed exposure budget differs')
    else:stream={'status':'RAW_ASSETS_NOT_LOCAL_RECONSTRUCTION_DEFERRED','actual_historical_consumption_order':'NOT_RECORDED'}
    from flops_audit import profile
    arithmetic=profile(c,device,batch=4,unroll=64)
    decision=readiness(audit,cached,arithmetic,stream,provenance)
    identity={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_files_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.glob('*.py'))},'protocol_sha256':hashlib.sha256((base/'protocol.json').read_bytes()).hexdigest(),'freeze_candidate_sha256':hashlib.sha256((base/'freeze-candidate.json').read_bytes()).hexdigest(),'protocol_frozen':False,'dataset_sha256':audit['tokenizer_audit']['dataset_manifest_sha256_recorded'],'tokenizer_sha256':audit['tokenizer_audit']['tokenizer_file_sha256'],'seed':713,'seed_scope':'synthetic correctness fixtures only; formal seeds not selected','hardware':{'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None},'software':{'python':platform.python_version(),'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'driver_version':None,'power_limit':None,'kernel_configuration':'default SDPA with explicit local causal mask; TF32 disabled'}}
    payloads={'audit/baseline-audit.json':audit,'audit/data-audit.json':{'verification_status':'RAW_BYTES_VERIFIED_THIS_RUN' if recover_assets else 'RAW_BYTES_VERIFIED_PINNED_PRIOR_RUN_NOT_REREAD','historical_record':audit['data_audit']},'audit/tokenizer-audit.json':audit['tokenizer_audit'],'audit/parameter-match-report.json':matching,'profiling/kv-cache-equivalence.json':cached,'profiling/inference-profile.json':{'reused_evidence':provenance,'measurements':previous['inference-profile.json']['result']},'profiling/memory-profile.json':{name:value['memory'] for name,value in cached.items()},'profiling/training-flops.json':arithmetic,'audit/token-exposure.json':stream,'audit/reused-engineering-evidence.json':provenance,'models/transformer/t1-reference.json':{'config':asdict(Config()),'native_parameters':1894080,'historical_kv_cache':False},'models/transformer/t1-pm-kv.json':{'config':asdict(c),'status':'CANDIDATE_NOT_FROZEN'},'models/transformer/t1-cm-kv.json':{'status':'NOT_SELECTED_COMPUTE_BUDGET_PENDING'},'reports/research-decision.json':decision}
    for path,body in payloads.items():write(out/path,{'identity':identity,'result':body})
    (out/'reports/summary.md').write_text('# GC002 工程准备\n\n无训练、无预算扩张。复用已核验的 CUDA 缓存/推理结果，数据资产已有真实哈希核验；历史消费顺序未记录，不能追溯证明。新增逐事件消费账本与完整路径算子清单，合成零学习率检查不消耗真实训练 Token。标量算子口径不等于硬件 FLOPs，正式 Compute-Matched 数值预算与 live 计数验证仍待冻结。协议为冻结候选，不授权训练。\n')
    return payloads['reports/research-decision.json']
