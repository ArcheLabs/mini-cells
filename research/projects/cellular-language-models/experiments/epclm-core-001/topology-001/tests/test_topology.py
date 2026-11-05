"""Compact graph, exact baseline, causal-cone and tiny TBPTT regression."""
import copy
import json
import time
from pathlib import Path
import torch
from repair_models import BroadcastCore,RepairConfig
from topology import graph,TopologyCore,ARMS
from training import train,evaluate
from diagnostics import propagation
from report import comparisons,screen

torch.set_num_threads(1)


def test_graphs_and_parameter_identity():
    expected={'IC':(0,0,None,None),'R1':(2,16,8,64/15),'R2':(4,32,4,2.4),'SR':(4,32,3,2.),'T4':(4,32,4,32/15)}
    for name in ARMS:
        g=graph(name);degree,edges,diameter,average=expected[name]
        assert (g['degree'],g['edge_count'],g['diameter'])==(degree,edges,diameter)
        if average is not None:assert abs(g['average_shortest_path']-average)<1e-10
        for i,row in enumerate(g['neighbors']):
            assert len(set(row))==len(row)==degree and i not in row
            assert all(i in g['neighbors'][j] for j in row)
        if name=='IC':assert g['negative']==g['positive']==[[i] for i in range(16)]
        model=TopologyCore(name);assert sum(p.numel() for p in model.parameters())==966784
        assert not any(k.startswith('_negative') or k.startswith('_positive') for k in model.state_dict())
    assert graph('SR')['average_shortest_path']!=graph('T4')['average_shortest_path']


def test_exact_baselines_means_and_synchronous_update():
    torch.manual_seed(713);state=torch.randn(2,16,128)
    for name,arm in (('R1','C1'),('IC','C0')):
        base=BroadcastCore(RepairConfig(arm=arm));candidate=TopologyCore(name);candidate.load_state_dict(base.state_dict())
        assert torch.equal(base.inject(state,[3,4]),candidate.inject(state,[3,4]))
        a,la=base.step(state,[3,4]);b,lb=candidate.step(state,[3,4])
        assert torch.equal(a,b) and torch.equal(la,lb)
    for name in ('R2','SR','T4'):
        model=TopologyCore(name);z=model.norm(state);negative,positive=model.channel_inputs(z)
        for i in range(16):
            assert torch.equal(negative[:,i],z[:,model.graph['negative'][i]].mean(1))
            assert torch.equal(positive[:,i],z[:,model.graph['positive'][i]].mean(1))
        def reference(order):
            out=torch.empty_like(state)
            for i in order:
                message=model.center(z[:,i])+model.left(negative[:,i])+model.right(positive[:,i])
                out[:,i]=state[:,i]+.1*model.update(torch.cat((z[:,i],message),-1))
            return out
        assert torch.equal(reference(range(16)),reference(reversed(range(16))))
        assert torch.allclose(model.evolve(state,1),reference(range(16)),atol=1e-6,rtol=1e-6)


def test_pulse_causal_cones_and_tiny_training(tmp_path):
    torch.manual_seed(713);config=RepairConfig(vocab=32,dim=8,event_shift=1,steps=4)
    plan={'seed':713,'fractions':[.25,.5,.75],'source_cells':[0,4,8,12],'relative_amplitude':.001,
          'minimum_amplitude':1e-6,'absolute_effect_threshold':1e-6,'relative_effect_threshold':1e-6,'future_events':[0,1,2,4,8]}
    docs=[{'id':'a','tokens':[2]+[4,5,6,7]*4+[3]}]
    initial=copy.deepcopy(TopologyCore('R1',config).state_dict())
    for name in ARMS:
        model=TopologyCore(name,config);model.load_state_dict(initial)
        evidence,diversity=propagation(model,docs,plan,time.monotonic()+30)
        assert evidence['status']=='COMPLETE' and evidence['pulse_count']==3 and len(diversity['samples'])==3
        if name=='IC':
            for pulse in evidence['pulses']:
                assert all(m['affected_cells']<=1 for m in pulse['microsteps'])
    protocol=json.loads((Path(__file__).resolve().parents[1]/'protocol.json').read_text())
    protocol['training'].update(prediction_tokens_per_arm=10,updates=3,batch=2,tbptt=2,warmup_updates=1,validation_milestones=[3])
    entries={}
    for name in ('R1','IC'):
        model=TopologyCore(name,config);model.load_state_dict(initial);entry={}
        train(model,docs,docs,protocol,time.monotonic()+30,entry,lambda:None)
        assert (entry['prediction_tokens'],entry['optimizer_updates'])==(10,3)
        assert entry['training_metrics'][-1]['update_prediction_tokens']==2
        assert len(entry['training_metrics'][-1]['state_observations'])==4
        assert entry['validation_progression']['3']['status']=='COMPLETE'
        result=evaluate(model,docs,time.monotonic()+30);entry['holdout']=entry['validation']=result
        entries[name]=entry
    protocol['evaluation']['bootstrap_replicates']=20
    paired=comparisons(entries,protocol)
    assert paired['comparisons']['IC-R1']['status']=='COMPLETE'
    assert paired['shared_cluster_draws_across_all_comparisons']
    assert screen(entries,paired,protocol)['status']=='INCOMPLETE_SCREEN'

    # Completed reports round-trip without model/optimizer checkpoint assets.
    from experiment import save_reports,previous_reports
    save_reports(tmp_path,protocol,entries,{'source_commit':'fixture'})
    restored=previous_reports(tmp_path,'fixture',protocol)
    assert restored['R1']['training_status']=='COMPLETE'
    assert restored['R1']['holdout']==entries['R1']['holdout']
    assert restored['R1']['training_metrics']==entries['R1']['training_metrics']
    assert not list(tmp_path.glob('*.pt'))
