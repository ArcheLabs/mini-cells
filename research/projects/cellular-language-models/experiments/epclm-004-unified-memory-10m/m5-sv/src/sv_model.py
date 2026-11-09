"""Read-only instrumentation and diagnostic policies around the frozen M5 model."""
from contextlib import contextmanager
import random
import time
import torch
from sv_data import BMojo
import bmojo
from store import Store, Predictor
from data import QUERY, ANSWER, digest
from sv_metrics import admission, retrieval


class OracleStore(Store):
    """Ground truth affects diagnostic admission only, never model input/read ranking."""
    def __init__(self, capacity, seed, required):
        super().__init__(capacity,'innovation',seed);self.required=set(required)

    def admit(self, representation, position, score):
        # Binary priority retains all required value records if they fit capacity.
        # This diagnostic store reports priority statistics, not innovation scores.
        return super().admit(representation,position,float(position in self.required))


class ValidationModel(BMojo):
    def __init__(self, *args, store_seed=72017, **kwargs):
        super().__init__(*args,**kwargs);self.store_seed=store_seed;self.oracle_required=None

    def new_runtime(self, batch, position=0, capacity=None):
        state=super().new_runtime(batch,position,capacity)
        if state.stores:
            if self.oracle_required is not None:
                if self.training and torch.is_grad_enabled():raise ValueError('oracle training is forbidden')
                state.stores=[OracleStore(capacity or self.capacity,self.store_seed,self.oracle_required) for _ in range(batch)]
            else:
                for s in state.stores:
                    s.seed=self.store_seed
                    if s.rng:s.rng=random.Random(self.store_seed)
        return state


@contextmanager
def observe(model, read_enabled=True, scores=False):
    """Sequential scoped wrappers; restore frozen functions even on failure."""
    original_read=bmojo.attention_read;original_scan=bmojo.mamba_chunk
    observed=[];mean=None
    final_module=next(block.attn for kind,block in reversed(list(zip(model.backbone.kinds,model.backbone.blocks))) if kind=='mamba1')
    def read(attention,normalizer,x,positions,kv,summary,stores,window,read_top_k,**kwargs):
        return original_read(attention,normalizer,x,positions,kv,summary,
                             stores if read_enabled else None,window,read_top_k if read_enabled else 0,**kwargs)
    def scan(module,hidden,previous=None):
        nonlocal mean
        result=original_scan(module,hidden,previous)
        if scores and module is final_module:
            values,mean=model.predictor.score(result[0],mean);observed.extend(values[0].cpu().tolist())
        return result
    bmojo.attention_read=read;bmojo.mamba_chunk=scan
    try:yield observed
    finally:bmojo.attention_read=original_read;bmojo.mamba_chunk=original_scan


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


@torch.no_grad()
def query(model, snapshot, question, enabled=True):
    device=model.token_embedding.weight.device;state=model.detached(snapshot)
    ids=torch.tensor([question['input_ids']],device=device)
    synchronize(device);began=time.perf_counter()
    with observe(model,read_enabled=enabled):
        logits,state,trace=model.forward_state(ids,state,trace_queries=[[2]],read_positions=[2],profile=True)
    synchronize(device);seconds=time.perf_counter()-began
    target=question['expected_value_id'];prediction=int(logits[0,0].argmax())
    nll=float(torch.nn.functional.cross_entropy(logits[:,0],torch.tensor([target],device=device)))
    return {'prediction':prediction,'correct':int(prediction==target),'nll':nll,
            'seconds':seconds,'per_input_token_seconds':seconds/3,'trace':trace,
            'read_timing':dict(model.last_timing)}


def position_ids(row):return {r['source_position']:r['record_id'] for r in row['records']}


def ranked_ids(trace, mapping):
    # All SWA outputs are real contributions; the primary retrieval metric uses
    # the last SWA layer, never an optimistic union across layers.
    return [{'layer':r['layer'],'ranked_positions':r['selected_positions'],
             'ranked_record_ids':[mapping.get(p,'non-evidence:'+str(p)) if p>=0 else None for p in r['selected_positions']]}
            for r in trace]


@torch.no_grad()
def evaluate_episode(model, row, oracle=False, recover_all=False, no_history=False):
    model.eval();device=model.token_embedding.weight.device;mapping=position_ids(row)
    all_future={rid for q in row['queries'] for rid in q['required_record_ids']}
    old_oracle=model.oracle_required
    model.oracle_required={p for p,rid in mapping.items() if rid in all_future} if oracle else None
    try:
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        ids=torch.tensor([row['history']],device=device)
        synchronize(device);began=time.perf_counter()
        with observe(model,scores=True) as scores:
            _,snapshot,_=model.forward_state(ids,trace=False,read_positions=[ids.shape[1]-1],profile=True)
        synchronize(device);prefill_seconds=time.perf_counter()-began;prefill_timing=dict(model.last_timing)
        snapshot=model.detached(snapshot);before=runtime_digest(snapshot)
        stored_positions=[r['position'] for r in snapshot.stores[0].records] if snapshot.stores else []
        stored=[mapping.get(p,'non-evidence:'+str(p)) for p in stored_positions]
        if oracle and len(all_future)<=model.capacity and not all_future<=set(stored):raise ValueError('oracle failed evidence retention')
        output=[]
        for question in row['queries']:
            on=query(model,snapshot,question,True);off=query(model,snapshot,question,False)
            layers=ranked_ids(on['trace'],mapping);last=layers[-1]['ranked_record_ids'] if layers else []
            output.append({'query_id':question['query_id'],'required_record_ids':question['required_record_ids'],
                'expected_value_id':question['expected_value_id'],'prediction_on':on['prediction'],'prediction_off':off['prediction'],
                'correct_on':on['correct'],'correct_off':off['correct'],'nll_on':on['nll'],'nll_off':off['nll'],
                'admission':admission(stored,question['required_record_ids'],all_future),
                'retrieval':retrieval(last,question['required_record_ids']),'per_layer':layers,
                'retrieval_primary':'last SWA layer actual top-k candidates; no any-layer success union',
                'latency_on_seconds':on['seconds'],'latency_off_seconds':off['seconds'],
                'per_input_token_latency_seconds':on['per_input_token_seconds'],'store_read_timing':on['read_timing']})
        recoverability=[]
        if recover_all:
            for r in row['records']:
                question={'input_ids':[QUERY,r['key_id'],ANSWER],'expected_value_id':r['value_id']}
                answer=query(model,snapshot,question,False)
                recoverability.append({'record_id':r['record_id'],'source_position':r['source_position'],
                    'correct':answer['correct'],'score':scores[r['source_position']]})
        controls=[]
        if no_history:
            for question in row['queries']:
                answer=query(model,model.new_runtime(1,snapshot.position),question,False)
                controls.append({'query_id':question['query_id'],'correct':answer['correct'],'prediction':answer['prediction']})
        assert runtime_digest(snapshot)==before,'query contaminated shared prefill snapshot'
        resources=model.resources(snapshot)
        if device.type=='cuda':resources['inference_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        return {'episode_id':row['id'],'condition':row['condition'],'history_tokens':len(row['history']),
            'queries':output,'recoverability':recoverability,'no_history':controls,
            'record_scores':{r['record_id']:scores[r['source_position']] for r in row['records']},
            'stored_positions':stored_positions,'stored_record_ids':stored,'resources':resources,
            'prefill_seconds':prefill_seconds,'prefill_timing':prefill_timing,'oracle_diagnostic_only':oracle,
            'read_only_snapshot_sha256':before}
    finally:model.oracle_required=old_oracle


def runtime_digest(state):
    """Hash values, metadata and RNG rather than object identity or tensor aliases."""
    import hashlib
    h=hashlib.sha256();h.update(str(state.position).encode())
    for group in (state.mamba,state.kv):
        for key,pair in sorted(group.items()):
            h.update(str(key).encode())
            for t in pair:h.update(t.detach().cpu().contiguous().numpy().tobytes())
    if state.predictor is not None:h.update(state.predictor.detach().cpu().numpy().tobytes())
    for s in state.stores or []:
        snap=s.snapshot()
        for r in snap['records']:h.update(r.pop('representation').numpy().tobytes())
        h.update(digest(snap).encode())
    return h.hexdigest()
