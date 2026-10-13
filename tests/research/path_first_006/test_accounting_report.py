import copy,torch,pytest
from exp006.baseline import make_model
from exp006.accounting import ComputeCounter,summarize,touched_weights
from exp006.report import analyze,matrix
from exp006.reuse import paired_bootstrap_delta

def test_repeated_cells_work_not_unique_weights(c):
    model=make_model(c['model'],'R1').eval();x=torch.arange(8)[None]
    repeat=torch.zeros(1,8,2,dtype=torch.long);distinct=repeat.clone();distinct[:,:,1]=1
    with ComputeCounter(model) as counter:out=model(x,replay=repeat)
    a=summarize(model,out,counter);b=touched_weights(model,model(x,replay=distinct))
    assert a['unique_operations_per_token']==1 and a['calls_per_token']==2
    assert b['mean']-a['unique_touched_weights']['mean']==2*16*24
    assert counter.by_module['cells.0.up']==2*8*16*24*2
    assert a['gemm_flops']>0 and a['context_read_calls']==0

def test_context_and_controller_cost(c):
    model=make_model(c['model'],'R2').eval();x=torch.arange(8)[None];path=torch.full((1,8,2),3,dtype=torch.long)
    with ComputeCounter(model) as count:out=model(x,replay=path)
    a=summarize(model,out,count);assert a['context_read_calls']==16
    assert a['context_attention_gemm_flops']==4*8*8*16*2
    assert a['controller_gemm_flops']>0 and a['gemm_flops']>=a['context_attention_gemm_flops']

def test_story_cluster_bootstrap_and_incomplete(c):
    r=paired_bootstrap_delta([1,1,1,1],[2,2,2,2],group_ids=[0,0,1,1],seed=1,replicates=100)
    assert r['paired_bootstrap_95pct_ci']==[-1,-1]
    assert analyze({},c,{})['status']=='INCOMPLETE'
    assert matrix([],c,{})['scientific_decision'] is False

def test_cpu_isolated_decode_profiler_schema(c):
    from exp006.profiling import benchmark
    model=make_model(c['model'],'R2').eval();x=torch.arange(8)[None]
    profile=benchmark(model,x,c)
    assert profile['warm_p50_ms']>0 and profile['batch_size']==1
    assert profile['operator_counts'] and profile['cuda_device_events'] is None
    assert profile['decode_compute']['calls_per_token']==2
    assert len(profile['warm_by_cache_reset'])==2

def test_removed_prior_has_no_charged_weights_or_gemm(c):
    model=make_model(c['model'],'R2').eval();x=torch.arange(8)[None]
    replay=torch.zeros(1,8,2,dtype=torch.long)
    with ComputeCounter(model) as counter:removed=model(x,replay=replay,edge_mode='zero')
    natural=model(x,replay=replay)
    assert counter.by_module['controller.edge_dot']==0
    assert touched_weights(model,natural)['mean']>touched_weights(model,removed)['mean']
