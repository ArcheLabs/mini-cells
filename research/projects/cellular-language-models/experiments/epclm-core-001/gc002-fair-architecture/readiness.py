"""Reuse immutable engineering evidence; never turn missing history into a claimed audit."""
import hashlib
import json
import subprocess

COMMIT='187d7f498dcf252a13da9e6dbc384fa3a8c36d9a'
BRANCH='kaggle/epclm-gc002-preparation-ba35017c2c51'
PREFIX='artifacts/experiments/epclm-core-001/gc002-fair-architecture/preparation/'


def previous_evidence(root,base):
    def read(name):
        return subprocess.check_output(['git','show',COMMIT+':'+PREFIX+name],cwd=root,stderr=subprocess.PIPE)
    try:metadata=json.loads(read('metadata.json'))
    except subprocess.CalledProcessError:
        subprocess.run(['git','fetch','--depth=1','origin',BRANCH],cwd=root,check=True,timeout=60,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        metadata=json.loads(read('metadata.json'))
    records={};hashes={row['path']:row['sha256'] for row in metadata['files']}
    for name in ('baseline-audit.json','kv-cache-equivalence.json','inference-profile.json','parameter-match-report.json'):
        raw=read(name)
        if hashlib.sha256(raw).hexdigest()!=hashes[name]:raise ValueError('prior evidence hash mismatch')
        records[name]=json.loads(raw)
    cached=records['kv-cache-equivalence.json']
    source_key=str((base/'cached_transformer.py').relative_to(root))
    if cached['identity']['source_files_sha256'].get(source_key)!=hashlib.sha256((base/'cached_transformer.py').read_bytes()).hexdigest():raise ValueError('cache implementation changed; prior validation cannot be reused')
    if any(r['status']!='PASS' for r in cached['result'].values()):raise ValueError('prior cache checks did not pass')
    return records,{'status':'REUSED_VERIFIED_COMPLETED_ENGINEERING_EVIDENCE','commit':COMMIT,'branch':BRANCH,
                    'files_sha256':{n:hashes[n] for n in records},'original_identity':cached['identity'],
                    'scope':'same cached_transformer.py bytes; previously measured CUDA random-weight correctness and inference; not new measurement'}


def readiness(audit,cached,flops,stream,provenance):
    verified=audit['tokenizer_audit']['status']=='RAW_FILE_HASHES_AND_TOKEN_CONTRACTS_VERIFIED'
    inventory=all(r['operator_inventory_complete'] for r in flops['models'].values())
    return {'status':'PREPARATION_ADVANCED_FREEZE_CANDIDATE_NOT_AUTHORIZED',
        'phase0':'METADATA_AND_RAW_ASSETS_VERIFIED_HISTORICAL_RUNTIME_ORDER_UNVERIFIABLE' if verified else 'RAW_ASSETS_PENDING',
        'phase1':'CUDA_CACHE_AND_INFERENCE_ENGINEERING_PASS' if provenance and all(r['status']=='PASS' for r in cached.values()) else 'CACHE_OR_GPU_PROFILE_PENDING',
        'arithmetic_inventory':'COVERED_UNDER_DECLARED_OPERATOR_CONVENTION' if inventory else 'UNSUPPORTED_OPERATORS_BLOCKING',
        'exposure_reconstruction':stream['status'],
        'phase2':'FREEZE_CANDIDATE_PENDING_LIVE_UPDATE_ACCOUNTING_AND_COMPUTE_BUDGET',
        'phase3':'NOT_AUTHORIZED_TO_TRAIN','training_tokens_consumed':0,'historical_results_mutated':False,
        'formal_hypotheses':'NOT_EVALUATED','human_review':'PENDING',
        'historical_limit':'unpublished per-update input hashes cannot be recovered from summary metadata; reconstruction is not observation',
        'remaining_gates':['validate live committed exposure/FLOPs ledger on target runtime and full update shapes','validate scalar-operation convention against the chosen complete FLOPs standard; unknown or uncovered ops fail closed','freeze numerical compute budget/checkpoints, memory-task budget and measured GPU-hour estimate','obtain separate pilot/formal training approval']}
