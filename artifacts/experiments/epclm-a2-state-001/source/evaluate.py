"""Frozen per-path ablations and actual Cell State diagnostics; no routing metrics."""
from contextlib import contextmanager
import time
import torch
from bootstrap import shared,old_evaluate
from .runtime import Session


@contextmanager
def flags(model,**changes):
    previous={key:getattr(model,key) for key in changes}
    try:
        for key,value in changes.items():setattr(model,key,value)
        yield
    finally:
        for key,value in previous.items():setattr(model,key,value)


def statistics(model,inputs):
    model.eval();model.record_diagnostics=True
    try:
        with torch.no_grad():logits,cache=model.forward_extended(inputs,return_session=True)
        state=cache['persistent'];unit=torch.nn.functional.normalize(state,dim=-1)
        records=[]
        for row in model.observations:
            records.append({key:value.cpu().tolist() if torch.is_tensor(value) else value for key,value in row.items()})
        update={key:float(value) for key,value in model.persistent.last.items()}
        return {'status':'COMPLETE','state_norm_per_cell':state.norm(dim=-1).cpu().tolist(),
            'pairwise_cosine_similarity':(unit@unit.transpose(-1,-2)).cpu().tolist(),'read_observations':records,
            'update':update,'sample_tokens':inputs.shape[1],'state_writes':cache['state_revision'],
            'scope':'fixed validation span, module/Cell numerical diagnostics, not semantic roles or routing'}
    finally:model.record_diagnostics=False


def run(model,corpus,tokenizer,starts,plan,device,out,deadline):
    # Existing generation/profiling share the new Session but retain their evaluation plan.
    original=old_evaluate.Session;old_evaluate.Session=Session
    results={};stream=shared.native.open_memmap(corpus.validation_path);identity=corpus.manifest['tokenizer_sha256']
    def execute(filename,call):
        try:
            if time.monotonic()>=deadline:raise TimeoutError('evaluation execution budget exhausted')
            value=call();shared.write(out/filename,value)
        except Exception as error:
            value={'status':'FAILED','error':type(error).__name__+': '+str(error)};shared.write(out/filename,value)
        results[filename]=value;return value
    try:
        execute('validation-results.json',lambda:old_evaluate.language(model,stream,starts[0],device,deadline))
        execute('holdout-results.json',lambda:old_evaluate.language(model,stream,starts[1],device,deadline))
        def generated():
            value=old_evaluate.generation(model,tokenizer,plan,device,deadline,identity)
            for row in value.get('samples',[]):row.pop('routing_by_generated_token',None)
            return value
        execute('generation-samples.json',generated)
        execute('inference-profile.json',lambda:old_evaluate.performance(model,plan,device,deadline,identity))
        inputs,_=shared.native.memmap_batch(stream,starts[0][0][:1],128,device)
        diagnostic=execute('state-statistics.json',lambda:statistics(model,inputs))
        ablations={}
        modes={'read_disabled':{'read_enabled':False},'write_disabled':{'write_enabled':False},'reset_every_chunk':{'reset_every_chunk':True}}
        if model.arm=='B3':modes.update(attention_read_disabled={'read_attention':False},ffn_read_disabled={'read_ffn':False})
        for name,change in modes.items():
            with flags(model,**change):ablations[name]=execute('ablation-'+name+'.json',lambda:old_evaluate.language(model,stream,starts[0],device,deadline))
        state={'status':'COMPLETE' if diagnostic.get('status')=='COMPLETE' and all(v.get('status')=='COMPLETE' for v in ablations.values()) else 'INCOMPLETE',
               'statistics':diagnostic,'ablations':ablations,'scope':'frozen-weight inference ablations; no additional training; no beyond-128 memory claim'}
        shared.write(out/'state-diagnostics.json',state);results['state-diagnostics.json']=state
        execute('memory-profile.json',lambda:old_evaluate.memory_profile(model,results,device))
    finally:old_evaluate.Session=original
    return results
