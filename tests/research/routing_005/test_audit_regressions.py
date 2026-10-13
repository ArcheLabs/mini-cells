"""Audit regressions: real decision schema, routing, measurement and exact resume."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
import types
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/08-hybrid-clm/experiments/005-compositional-cell-graph'
sys.path[:0] = [str(ROOT), str(BASE / 'implementation')]
from exp005.models import ARMS, make_model
from exp005.report import seed_decision, matrix_decision, paired_bootstrap_delta
from exp005.preflight import seed_role
from exp005.edge_refresh import make_edge_refresh_callback, scheduled_sources
from exp005.training import holdout_windows
from exp005.profiling import benchmark_decode
from exp005.checkpoint import save_checkpoint, restore_checkpoint


def protocol():
    value = json.loads((BASE / 'protocol.json').read_text())
    value['bootstrap_replicates'] = 100
    return value


def rows(seed=901, role='confirmation'):
    ident = dict(source_commit='a'*40, config_sha256='b'*64,
                 protocol_sha256='c'*64, assets_manifest_sha256='d'*64)
    result = {}
    for arm, loss, cost in [('M0',2.0,100),('P0',2.1,100),('S0',2.0,70),('R0',2.05,70),('G0',1.97,60)]:
        timing = 1.0 if arm != 'G0' else 0.9
        runs = [[timing]*32 for _ in range(30)]
        result[arm] = dict(status='completed',arm=arm,seed=seed,seed_role=role,
            source_commit='a'*40,identity=ident,train_tokens=10000000,steps=5000,
            final_holdout=dict(validation_tokens=32000,model_total_params=61000000,
                sample_nll=[loss]*4,story_group_ids=[0,0,1,2],perplexity=math.exp(loss),
                decode_parity={'passed':True},unique_weight_params_per_token={'mean':cost},
                observed_gemm={'observed_gemm_flops_per_token':cost},
                decode_profile=dict(warm_decode_host_p50_ms=timing,warm_decode_host_p95_ms=timing,
                    warm_decode_host_ms=[v for run in runs for v in run],
                    warm_decode_host_ms_by_repetition=runs)))
    result['G0']['graph_learning'] = dict(edge_replacements=10,
        effective_source_coverage_fraction_per_layer=[1.0]*6,
        final_routing_summary={'per_layer':[{'repeated_path_token_fraction':0.2}]*6})
    result['G0']['posthoc_diagnostics']={'results':[
        dict(perplexity=math.exp(2.0),sample_nll=[2.0]*4,parameter_count_unchanged=True) for _ in range(3)]}
    return result


def test_full_five_arm_decision_schema_and_confirmation():
    reports=[]
    for seed in [901,902,903]:
        report=seed_decision(rows(seed),protocol=protocol(),seed=seed)
        assert [report[name]['status'] for name in ('b1_sequential_composition','b2_learned_connections','system_gate')] == ['PASS']*3
        reports.append(report)
    matrix=matrix_decision(reports,protocol(),[901,902,903])
    assert matrix['status']=='CONFIRMATION_ANALYZED'
    assert matrix['b2']['status']=='SUPPORTED'
    assert matrix['scientific_decision'] is False
    assert matrix['review_status']=='AWAITING_RESEARCH_REVIEW'


def test_graph_improvement_does_not_support_beneficial_connections():
    data=rows()
    for perturbation in data['G0']['posthoc_diagnostics']['results']:
        perturbation['perplexity']=math.exp(1.90)
        perturbation['sample_nll']=[1.90]*4
    report=seed_decision(data,protocol=protocol(),seed=901)
    assert report['b2_learned_connections']['status']=='NOT_SUPPORTED'
    assert all(value<0 for value in report['b2_learned_connections']['graph_diagnostics']['perturbed_ppl_relative_degradations'])


def test_negative_and_incomplete_decisions():
    data=rows();data['S0']['final_holdout']['sample_nll']=[2.2]*4
    assert seed_decision(data,protocol=protocol(),seed=901)['b1_sequential_composition']['status']=='NOT_SUPPORTED'
    data=rows();data.pop('R0')
    assert seed_decision(data,protocol=protocol(),seed=901)['status']=='INCOMPLETE'
    data=rows();data['G0']['final_holdout']['decode_profile']['warm_decode_host_ms_by_repetition']=[[1.0]*15]
    assert seed_decision(data,protocol=protocol(),seed=901)['status']=='INCOMPLETE'
    p=protocol();p['b2_graph_diagnostics_rule']=None
    assert seed_decision(rows(),protocol=p,seed=901)['b2_learned_connections']['status']=='INCOMPLETE'
    data=rows();data['G0']['graph_learning']['effective_source_coverage_fraction_per_layer']=[0.2]*6
    assert seed_decision(data,protocol=protocol(),seed=901)['b2_learned_connections']['status']=='INCOMPLETE'


def test_registered_roles_and_overlap():
    c={'seeds':{'development':[1001],'confirmation':[1002,1003,1004]}}
    assert seed_role(c,1001)=='development'
    assert seed_role(c,1004)=='confirmation'
    with pytest.raises(ValueError):seed_role(c,1005)
    c['seeds']['confirmation'].append(1001)
    with pytest.raises(ValueError):seed_role(c,1001)
    report=seed_decision(rows(role='development'),protocol=protocol(),seed=901)
    assert matrix_decision([report],protocol(),[901,902,903])['status']=='CONFIRMATION_INCOMPLETE'


def tiny():
    return dict(width=8,layers=1,heads=2,vocab_size=16,max_sequence_length=8,
        moe_experts=4,moe_hidden=4,cell_count=8,cell_hidden=2,cell_visits=4,
        graph_out_degree=4,router_width=2,residual_scale=0.25,top2_backup_gradient=True)


def test_initialization_and_all_arm_gradients():
    models={arm:make_model(tiny(),arm,seed=902) for arm in ARMS}
    reference=dict(models['M0'].named_parameters())
    for arm,model in models.items():
        for name,value in model.named_parameters():
            if '.ffn.' not in name:assert torch.equal(value,reference[name])
        model.train();logits,aux,_=model(torch.tensor([[1,2,3,4]]))
        assert aux.requires_grad and torch.isfinite(aux)
        (logits.square().mean()+0.01*aux).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        router=model.blocks[0].ffn
        parameter=(router.moe.router.weight if arm=='M0' else router.parallel_router.weight if arm=='P0' else router.router.query.weight)
        assert parameter.grad is not None
    assert all(torch.equal(a,b) for a,b in zip(models['R0'].parameters(),models['G0'].parameters()))


def test_checkpoint_exact_next_step_and_graph(tmp_path):
    model=make_model(tiny(),'G0',seed=905)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001)
    rng=np.random.default_rng(7);random.seed(7);torch.manual_seed(7)
    x=torch.tensor([[1,2,3,4]])
    def step():
        opt.zero_grad();out,aux,_=model(x);(out.square().mean()+aux*.01).backward();opt.step()
    step();identity={'source_commit':'a'*40,'seed_role':'confirmation'}
    save_checkpoint(tmp_path/'state.pt',model=model,optimizer=opt,step=1,scaler=None,identity=identity,generator=rng)
    expected_random=(random.random(),rng.random(),torch.rand(2))
    step();expected={name:v.clone() for name,v in model.state_dict().items()}
    model.blocks[0].ffn.adjacency.zero_()
    restore_checkpoint(tmp_path/'state.pt',model=model,optimizer=opt,scaler=None,expected_identity=identity,generator=rng)
    assert random.random()==expected_random[0] and rng.random()==expected_random[1]
    assert torch.equal(torch.rand(2),expected_random[2])
    step();assert all(torch.equal(model.state_dict()[name],v) for name,v in expected.items())


def test_source_schedule_covers_start_and_every_cell_with_real_probe():
    assert set(source for event in range(9) for source in scheduled_sources(event,32,4))==set(range(33))
    model=make_model(tiny(),'G0',seed=9)
    cfg={'model':tiny(),'data':{'sequence_length':4},'training':dict(edge_refresh_candidate_count=1,
        edge_refresh_probe_count=1,edge_refresh_source_count=9,max_edge_replacements_per_refresh=9,
        edge_refresh_min_improvement=0.0,edge_refresh_every_steps=1)}
    assets={'train':np.arange(30,dtype=np.int32)%16,'probe_starts':np.array([0])}
    event=make_edge_refresh_callback(cfg,assets,9)(model,1,None)
    assert {row['source'] for row in event['layers'][0]['sources']}==set(range(9))
    assert all(row['effective_probe_tokens']==4 for row in event['layers'][0]['sources'])
    assert event['observed_probe_gemm_flops']>0
    assert event['edge_changes']


def test_disjoint_story_windows_and_cluster_bootstrap():
    tokens=np.array([1]*12+[0]+[2]*12+[0]+[3]*12+[0],dtype=np.int32)
    starts,groups=holdout_windows(tokens,0,4,6,3)
    assert len(set(starts.tolist()))==6
    assert all(not np.any(tokens[start:start+4]==0) for start in starts)
    assert len(set(groups))>=2
    report=paired_bootstrap_delta([1.0]*6,[2.0]*6,seed=1,replicates=100,group_ids=groups)
    assert report['paired_bootstrap_95pct_ci']==[-1.0,-1.0]


def test_decode_repetitions_reset_cache_and_report_samples():
    model=make_model(tiny(),'P0',seed=7)
    profile=benchmark_decode(model,torch.tensor([[1,2,3,4,5,6]]),prefix_length=2,
        decode_tokens=4,warmup_steps=0,repetitions=3)
    assert profile['warm_decode_sample_count']==9
    assert [len(row) for row in profile['warm_decode_host_ms_by_repetition']]==[3]*3
    assert len(profile['first_decode_after_cache_reset_host_ms'])==3


def test_archive_to_registered_report_end_to_end(tmp_path):
    from exp005.analyze_experiment_005 import analyze
    c=json.loads((BASE/'config/proposed.json').read_text())
    c['seeds']={'development':[900],'confirmation':[901,902,903]}
    p=protocol();p['status']='REGISTERED'
    config_file=tmp_path/'config.json';protocol_file=tmp_path/'protocol.json'
    config_file.write_text(json.dumps(c));protocol_file.write_text(json.dumps(p))
    archive=tmp_path/('a'*40)
    for seed in [901,902,903]:
        for arm,metric in rows(seed).items():
            metric['identity']['config_sha256']=hashlib.sha256(config_file.read_bytes()).hexdigest()
            metric['identity']['protocol_sha256']=hashlib.sha256(protocol_file.read_bytes()).hexdigest()
            target=archive/f'seed-{seed}'/arm;target.mkdir(parents=True)
            (target/'metrics.json').write_text(json.dumps(metric))
    report,digest=analyze(archive,protocol_file,config_file)
    assert report['status']=='CONFIRMATION_ANALYZED' and len(digest)==64
    assert report['b2']['status']=='SUPPORTED'
    metric=json.loads((archive/'seed-901/G0/metrics.json').read_text())
    metric['seed_role']='development'
    (archive/'seed-901/G0/metrics.json').write_text(json.dumps(metric))
    with pytest.raises(ValueError,match='seed role'):analyze(archive,protocol_file,config_file)


def test_legacy_prepare_data_default_compatibility_and_prefix_hash(monkeypatch,tmp_path):
    from experiments.common.train import prepare_data
    stories={'train':['abc','def','ghi','jkl'],'validation':['mno','pqr','stu','vwx']}
    datasets=types.ModuleType('datasets')
    datasets.load_dataset=lambda name,**kw:iter({'text':text} for text in stories[kw['split']])
    monkeypatch.setitem(sys.modules,'datasets',datasets)
    class ByteLevel:
        def __init__(self,**kw):pass
        @staticmethod
        def alphabet():return []
    class Tokenizer:
        def __init__(self,*args):pass
        def train_from_iterator(self,texts,trainer):list(texts)
        def save(self,path):Path(path).write_text('{}')
        def token_to_id(self,word):return 0
        def encode_batch(self,texts):return [types.SimpleNamespace(ids=[ord(char) for char in text]) for text in texts]
        def get_vocab_size(self):return 32
    tokenizers=types.ModuleType('tokenizers');tokenizers.Tokenizer=Tokenizer
    tokenizers.models=types.SimpleNamespace(BPE=lambda **kw:None)
    tokenizers.pre_tokenizers=types.SimpleNamespace(ByteLevel=ByteLevel)
    tokenizers.decoders=types.SimpleNamespace(ByteLevel=ByteLevel)
    tokenizers.trainers=types.SimpleNamespace(BpeTrainer=lambda **kw:None)
    monkeypatch.setitem(sys.modules,'tokenizers',tokenizers)
    cfg=dict(dataset='fixture',dataset_revision='pinned',train_stories=4,validation_stories=4,
             tokenizer_stories=4,vocab_size=32,sequence_length=2)
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    a=prepare_data(cfg,old,float('inf'));b=prepare_data(cfg,new,float('inf'),validation_prefix_count=2)
    old_data=json.loads((old/'data.json').read_text());new_data=json.loads((new/'data.json').read_text())
    assert all(np.array_equal(a[key],b[key]) for key in a)
    assert all(new_data[key]==value for key,value in old_data.items())
    digest=hashlib.sha256()
    for text in stories['validation'][:2]:
        raw=text.encode();digest.update(len(raw).to_bytes(8,'little'));digest.update(raw)
    assert new_data['validation_prefix_text_sha256']==digest.hexdigest()


def test_system_negative_and_unfrozen_rules():
    data=rows();data['G0']['final_holdout']['decode_profile']['warm_decode_host_p50_ms']=2.0
    assert seed_decision(data,protocol=protocol(),seed=901)['system_gate']['status']=='NOT_SUPPORTED'
    p=protocol();p['system_runtime_rule']=None
    assert seed_decision(rows(),protocol=p,seed=901)['system_gate']['status']=='INCOMPLETE'
    p=protocol();p['b2_strong_flat_matching_rule']=None
    assert seed_decision(rows(),protocol=p,seed=901)['b2_learned_connections']['status']=='INCOMPLETE'


def test_development_rows_cannot_complete_confirmation():
    reports=[seed_decision(rows(seed,role='development'),protocol=protocol(),seed=seed) for seed in [901,902,903]]
    decision=matrix_decision(reports,protocol(),[901,902,903])
    assert decision['status']=='CONFIRMATION_INCOMPLETE'
    assert 'b2' not in decision


def test_notebook_default_bootstrap_fetches_pinned_source(monkeypatch):
    import subprocess
    nb=json.loads((BASE/'notebook.ipynb').read_text())
    source=''.join(nb['cells'][1]['source'])
    source=source.replace("MODE = 'train_pair'", "MODE = 'preflight'")
    source=source.replace("ROOT = Path('/kaggle/working/minicells-exp005')",f'ROOT = Path({str(ROOT)!r})')
    calls=[]
    def run(args,**kwargs):
        calls.append(list(args))
        output='HEAD' if args[:3]==['git','rev-parse','--abbrev-ref'] else ''
        return types.SimpleNamespace(stdout=output,returncode=1 if 'show-ref' in args else 0)
    monkeypatch.setattr(subprocess,'run',run)
    scope={};exec(compile(source,'005 notebook','exec'),scope)
    assert len(scope['SOURCE_SHA'])==40
    assert ['git','fetch','--depth=1','origin',scope['SOURCE_SHA']] in calls
    assert any('preflight' in call and str(ROOT/'scripts/research/run_experiment_005.py') in call for call in calls)
    assert (scope['CONFIG_PATH']).is_file()


def test_development_training_never_reads_reserved_holdout(tmp_path):
    from exp005.training import train_arm
    class ForbiddenHoldout(np.ndarray):
        def __getitem__(self,key):raise AssertionError('development accessed confirmation holdout')
    cfg=json.loads((BASE/'config/proposed.json').read_text())
    cfg['model']=tiny();cfg['data'].update(vocab_size=16,sequence_length=8,batch_size=2,train_tokens_per_arm=32)
    cfg['training'].update(steps=2,warmup_steps=1,evaluate_every_steps=1)
    cfg['evaluation'].update(windows=4,batch_size=2)
    cfg['protocol_status']='registered';cfg['seeds']={'development':[50713001],'confirmation':[50713011]}
    cfg_file=tmp_path/'config.json';proto_file=tmp_path/'protocol.json'
    cfg_file.write_text(json.dumps(cfg));proto_file.write_text('{}')
    cfg['config_path']=str(cfg_file);cfg['protocol_path']=str(proto_file)
    sequence=np.tile(np.array(list(range(1,16))+[0],dtype=np.int32),50)
    assets={'train':sequence,'regression':sequence.copy(),
            'holdout':sequence.copy().view(ForbiddenHoldout),'eos_id':0,'manifest_sha256':'d'*64}
    result=train_arm(cfg,'M0',50713001,'a'*40,assets,tmp_path/'run',device='cpu')
    assert result['evaluation_split']=='regression'
    assert result['final_holdout'] is None
    assert result['development_evaluation']['validation_tokens']==32
    assert result['identity']['seed_role']=='development'


def test_frozen_probe_sampling_is_reproducible_on_this_numpy():
    c=json.loads((BASE/'config/registered.json').read_text())
    data=json.loads((BASE/'registration/data.json').read_text())
    probe=json.loads((BASE/'registration/probe-windows.json').read_text())
    bound=data['token_counts']['train']-c['data']['sequence_length']-1
    starts=np.random.default_rng(c['training']['probe_seed']).choice(
        bound,c['training']['probe_pool_window_count'],replace=False).astype(np.int64).tolist()
    assert starts==probe['starts']


def test_registered_contract_is_ready_for_development():
    from exp005.run_experiment_005 import DEFAULT_CONFIG, load_contract, blockers
    config, definition, protocol, *_ = load_contract(str(DEFAULT_CONFIG))
    assert DEFAULT_CONFIG.name == 'registered.json'
    assert blockers(config, definition, protocol) == []
    assert config['seeds']['development'] == [50713001]
    assert config['registration']['confirmation_authorized'] is False


def test_confirmation_is_rejected_before_assets(monkeypatch):
    import sys
    from exp005.run_experiment_005 import main
    monkeypatch.setattr(sys, 'argv', ['runner', '--mode', 'train', '--arm', 'M0',
        '--seed', '50713011', '--source-commit', 'a'*40, '--assets', '/missing',
        '--asset-manifest-sha256', 'b'*64, '--output', '/missing'])
    with pytest.raises(RuntimeError, match='confirmation seeds are reserved'):
        main()
