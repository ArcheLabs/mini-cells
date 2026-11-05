"""One tiny training/evaluation regression; no GPU preflight or recovery tests."""
import json,time,math
import torch
import t1_fresh as fresh


def test_fresh_training_and_all_evaluations_ignore_placeholders(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    protocol={'model':{'vocab':32,'dim':16,'layers':1,'heads':2,'kv_heads':1,'ffn':32,'window':128,'active_parameters':0},'seed':19,
        'training':{'batch_lanes':2,'fresh_prediction_targets_per_lane':2,'context_reuse_per_sequence_max':2,'prediction_token_budget':10},
        'optimizer':{'learning_rate':1e-3,'betas':[.9,.95],'weight_decay':.1,'foreach':False,'gradient_clip_norm':1.},
        'scheduler':{'total_updates':3,'warmup_updates':1},
        'evaluation':{'generation_modes':['greedy','sampling_temperature_0.8_top_k_40_top_p_0.95'],'generation_seed':713,'generation_max_new_tokens':3}}
    protocol['model']['active_parameters']=fresh.parameter_count(fresh.Config(**{k:protocol['model'][k] for k in ('vocab','dim','layers','heads','kv_heads','ffn','window')}))
    docs=[{'id':'a','tokens':[2,4,5,6,3]},{'id':'b','tokens':[2,7,8,9,3]}]
    model,counts=fresh.train(protocol,docs,'cpu',tmp_path,time.monotonic()+60)
    assert counts.consumed==10 and counts.updates==3
    class Tokenizer:
        def encode(self,text):
            class Encoded:ids=[2,4,5]
            return Encoded()
        def decode(self,tokens,**kwargs):return ' '.join(map(str,tokens))
    groups={'validation':docs,'development_holdout':docs}
    for filename in ('validation-results.json','holdout-results.json','generation-metrics.json','inference-profile.json','memory-profile.json'):
        fresh.write(tmp_path/filename,{'status':'NOT_RUN'})
    results=fresh.run_complete_evaluation(model,Tokenizer(),groups,protocol,'cpu',tmp_path,time.monotonic()+120)
    assert all(r['status']=='COMPLETE' for r in results.values()),results
    val=results['validation'];assert val['documents']==2 and val['prediction_tokens']==8
    assert math.isclose(val['perplexity'],math.exp(val['total_nll']/8))
    assert len(json.loads((tmp_path/'generation-samples.json').read_text())['samples'])==32
    assert results['inference']['cached_noncached_equivalence'][-1]['history_tokens']>128
    original=fresh.evaluate_documents
    def fail_validation(model,docs,device):
        if docs is groups['validation']:raise ValueError('intentional validation failure')
        return original(model,docs,device)
    groups['development_holdout']=docs.copy()
    monkeypatch.setattr(fresh,'evaluate_documents',fail_validation)
    # Independent callbacks still run when validation fails, without repeating training.
    for name in ('generation_diagnostics','inference_profile'):
        monkeypatch.setattr(fresh,name,lambda *args:{'status':'PASS','samples':[]})
    results=fresh.run_complete_evaluation(model,Tokenizer(),groups,protocol,'cpu',tmp_path,time.monotonic()+60)
    assert results['validation']['status']=='FAILED'
    assert all(results[n]['status']=='COMPLETE' for n in ('holdout','generation','inference','memory'))
    assert not list(tmp_path.glob('*.pt')) and not (tmp_path/'recovery.json').exists()
