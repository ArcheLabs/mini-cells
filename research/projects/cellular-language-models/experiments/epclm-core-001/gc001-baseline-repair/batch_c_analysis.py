"""Metadata-only synthesis of pinned completed runs; no model or optimizer creation."""
import json
import math
import subprocess
from frozen_model import BaselineIntegrityError
from mechanism_diagnostic import write
from native_clm_runtime import sha256_file
from repair_io import public_result
from repair_diagnostics import blind_review, quality_findings, state_observations

PINS={
    'a':('f88bcc47777e013f72eaa5fd7ff806676e6a7305','16134c64c637893dd05279801e19e3702e2181e8','30d7a0fd6506ab392f4bbfef28a431f86b2d732f0063979c032a7f8b74fe5e7b'),
    'b':('40f86adcf2979cb2a4aafc96d6f18c6064f67b3d','04942f9e81dfc649d5f6d7ab3ac34dd6e2eccf31','119f773d4dcac8264a4b1703013866ed02d2b1eecec5e06d5dcde834ce06316b'),
}
ARMS=('C1-BROADCAST','IC-ALL-SELF','F0-PM-BROADCAST')


def checked_pairs(models):
    from trainer import paired
    left,right=(models[n] for n in ARMS[:2]);result={}
    for split in ('validation','development_holdout'):
        tables=[]
        for entry in (left,right):
            rows=entry['paired_documents'][split]
            table={r['story_id']:r for r in rows}
            if not rows or len(table)!=len(rows):raise BaselineIntegrityError('empty or duplicate paired story IDs')
            if any(r['tokens']<=0 or not math.isfinite(r['nll_sum']) for r in rows):raise BaselineIntegrityError('invalid paired NLL records')
            if not math.isclose(sum(r['nll_sum'] for r in rows)/sum(r['tokens'] for r in rows),entry[split]['nll'],rel_tol=1e-8):raise BaselineIntegrityError('paired and aggregate NLL differ')
            tables.append(table)
        a,b=tables
        if a.keys()!=b.keys() or any(a[k]['tokens']!=b[k]['tokens'] for k in a):raise BaselineIntegrityError('paired documents/token counts differ')
        result[split]=paired({**left[split],'stories':list(a.values())},{**right[split],'stories':list(b.values())})
        result[split].update(documents=len(a),pairing='EXACT_UNIQUE_STORY_ID_AND_TOKEN_COUNT',unit='document',draws=1000,seed=713)
    return {'status':'SINGLE_SEED_DEVELOPMENT_ONLY','splits':result,'primary_split':'development_holdout',
            'limits':['one trained seed; no between-seed variance','development holdout is not an independent Test','document bootstrap is conditional on these trained weights','post-result synthesis; no preregistered significance decision']}


def decision_basis(report):
    stats=report['statistical_evidence']['splits']['development_holdout'];lo,hi=stats['story_bootstrap_95']
    direction='SUPPORT_WITHIN_THIS_RUN' if hi<0 else 'EVIDENCE_AGAINST_WITHIN_THIS_RUN' if lo>0 else 'INCONCLUSIVE_WITHIN_THIS_RUN'
    human=report['human_evaluation'];covered=human.get('reviews_complete',False)
    return {'scope':'POST_HOC_SINGLE_SEED_DEVELOPMENT_DECISION_SUPPORT',
        'questions':{
            'broadcast_reduces_nll_vs_independent':{'status':direction,'evidence':'statistical_evidence.splits.development_holdout','delta':stats['nll_difference'],'interval_95':[lo,hi],'formal_confirmation':False},
            'generation_quality':{'status':'HUMAN_RATINGS_AVAILABLE' if covered else 'UNDETERMINED_PENDING_COMPLETE_BLIND_REVIEW','evidence':'human_evaluation','automatic_metrics_are_not_human_coherence':True},
            'state_retains_useful_history':{'status':'UNDETERMINED','reason':'NLL, norms and similarity do not identify history retention; matched history deletion/state reset or intervention evidence is absent'},
            'state_evolution_degradation':{'status':'DESCRIPTIVE_ALERTS_ONLY','evidence':'quality_diagnosis and state-diagnostics.json','causal_attractor_claim':'NOT_ESTABLISHED'},
            'architecture_superiority':{'status':'UNDETERMINED','reason':'one seed, no independent Test, no compute-matched Transformer comparison'}},
        'next_experiments':{
            'parameter_matching':{'status':'DECIDE_AFTER_REVIEW','existing_flat_match':'active parameter matching already implemented; separate T1 matching requires a new protocol'},
            'compute_matching':{'status':'DEFERRED','condition':'define measured training/inference compute and repeated-seed design first'},
            'longer_transformer_training':{'status':'DEFERRED_NO_BUDGET_INCREASE','condition':'review convergence and blind quality first; use a separately approved matched protocol'}},
        'automatic_training':False,'additional_training_tokens':0,'formal_hypotheses':'H1_H5_NOT_EVALUATED'}


def assemble(root,base,out,reader=public_result):
    plan=json.loads((base/'protocol.json').read_text());protocol=sha256_file(base/'protocol.json')
    models={};sources={};audit=None
    for batch,(source,commit,digest) in PINS.items():
        decision,proof=reader(root,'kaggle/epclm-gc001-repair-'+batch+'-'+source[:12],
            'artifacts/experiments/epclm-core-001/gc001-baseline-repair/batch-'+batch,source,protocol,commit=commit)
        expected={'source':source,'protocol':protocol,'data':plan['gc_protocol']['training_identity']['data'],'batch':batch,'original_result_commit':plan['original_result_commit']}
        if proof['decision_sha256']!=digest or decision['identity']!=expected or decision['status']!='COMPLETE' or decision['protocol']!=plan:raise BaselineIntegrityError('pinned batch identity/protocol/status differs')
        if audit is not None and audit!=decision['data_audit']:raise BaselineIntegrityError('A/B ordered dataset audits differ')
        audit=decision['data_audit'];sources[batch]=proof
        for name in (ARMS if batch=='a' else ('T1',)):
            entry=decision['models'][name];training=entry['training_identity']
            if entry['status']!='COMPLETE' or entry['training']['tokens']!=250000 or any(training.get(k)!=v for k,v in expected.items()):raise BaselineIntegrityError('incomplete or mismatched trained model')
            if training['seed']!=plan['training']['seed'] or training['precision']!='fp32' or training['target_tokens']!=250000:raise BaselineIntegrityError('training settings differ')
        models.update(decision['models'])
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out.mkdir(parents=True,exist_ok=True)
    report={'format':'minicells.gc001-repair.v1','identity':{'source':source,'protocol':protocol,'data':plan['gc_protocol']['training_identity']['data'],'batch':'c','original_result_commit':plan['original_result_commit']},
        'protocol':plan,'models':models,'data_audit':audit,'source_results':sources,'hypotheses':dict.fromkeys(['H1','H2','H3','H4','H5'],'NOT_EVALUATED'),
        'execution_policy':{'model_loading':False,'training':False,'generation':False,'checkpoint_recovery':False,'additional_tokens':0},'timings':{}}
    report['statistical_evidence']=checked_pairs(models)
    report['quality_diagnosis']=quality_findings(models,plan)
    report['human_evaluation']=blind_review(out,models,plan)
    report['research_decision_basis']=decision_basis(report)
    report['extension_proposal']={'status':'DEFERRED_UNTIL_BATCH_C_REVIEW','automatic_training':False,'training_tokens_authorized':0}
    report['status']='COMPLETE' if report['human_evaluation'].get('reviews_complete') else 'ANALYSIS_COMPLETE_REVIEW_PENDING'
    report['scientific_reporting']={'formal_hypotheses':'H1_H5_NOT_EVALUATED','research_decision_basis':report['research_decision_basis']}
    write(out/'decision.json',report)
    write(out/'statistical-evidence.json',report['statistical_evidence'])
    write(out/'research-decision.json',report['research_decision_basis'])
    write(out/'state-diagnostics.json',{n:state_observations(e) for n,e in models.items()})
    return report


def launch_analysis(root,base,out):
    from repair_pipeline import publish
    from repair_io import timed_budget
    plan=json.loads((base/'protocol.json').read_text())
    with timed_budget(plan['budget']['execution']):report=assemble(root,base,out)
    with timed_budget(plan['budget']['publication']):publish(root,out,report,'epclm-gc001-repair-c')
