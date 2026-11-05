"""Forward-only pulses on current BroadcastCore, diversity and small timing measurements."""
import math
import statistics
import time
import torch
from torch.nn import functional as F
from training import sync


def check_time(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('diagnostic execution budget exhausted')


def diversity(state,observations):
    h=state.detach();unit=F.normalize(h,dim=-1);norms=h.norm(dim=-1)
    return {'pairwise_cosine_similarity':(unit@unit.transpose(-1,-2))[0].cpu().tolist(),
            'cell_state_norm':norms[0].cpu().tolist(),'mean_state_norm':float(norms.mean()),'max_state_norm':float(norms.max()),
            'cell_state_variance':float(h.var(1,unbiased=False).mean()),'microsteps':observations}


@torch.no_grad()
def propagation(model,documents,plan,deadline):
    model.eval();generator=torch.Generator().manual_seed(plan['seed']);pulses=[];samples=[];violations=[]
    for doc_index,doc in enumerate(documents):
        tokens=doc['tokens']
        if len(tokens)<11:raise ValueError('pulse document too short for eight future events')
        for fraction_index,fraction in enumerate(plan['fractions']):
            check_time(deadline);anchor=min(max(1,int(fraction*(len(tokens)-1))),len(tokens)-10)
            source=plan['source_cells'][(doc_index*3+fraction_index)%len(plan['source_cells'])]
            state=model.init_state(1)
            for token in tokens[:anchor]:check_time(deadline);state,_=model.step(state,[token])
            normal=model.inject(state,[tokens[anchor]]);perturbed=normal.clone()
            direction=torch.randn(model.config.dim,generator=generator).to(state.device);direction/=direction.norm()
            amplitude=max(plan['minimum_amplitude'],float(normal[0,source].norm())*plan['relative_amplitude'])
            perturbed[0,source]+=amplitude*direction
            first=[None]*16;first[source]=0;micro=[];future=[];normal_observation=[]
            for horizon in range(max(plan['future_events'])+1):
                if horizon:
                    normal=model.inject(normal,[tokens[anchor+horizon]])
                    perturbed=model.inject(perturbed,[tokens[anchor+horizon]])
                for k in range(1,model.config.steps+1):
                    check_time(deadline);model.observe=(horizon==0)
                    normal=model.evolve(normal,1)
                    if horizon==0:normal_observation.extend([{key:value.cpu().tolist() for key,value in row.items()} for row in model.observation])
                    model.observe=False;perturbed=model.evolve(perturbed,1)
                    difference=(perturbed-normal).norm(dim=-1)[0];norms=normal.norm(dim=-1)[0]
                    relative=difference/norms.clamp_min(1e-12)
                    affected=(difference>plan['absolute_effect_threshold']) & (relative>plan['relative_effect_threshold'])
                    updates=model.config.steps*horizon+k
                    affected_cpu=affected.cpu().tolist();difference_cpu=difference.cpu().tolist()
                    for cell in range(16):
                        if affected_cpu[cell]:
                            if first[cell] is None:first[cell]=updates
                            distance=model.graph['distances'][source][cell]
                            if distance is None or distance>updates:violations.append({'document_id':doc['id'],'source':source,'cell':cell,'microstep':updates,'difference':difference_cpu[cell]})
                    micro.append({'future_events':horizon,'internal_step':k,'total_microsteps':updates,
                                  'cell_difference':difference_cpu,'relative_cell_difference':relative.cpu().tolist(),
                                  'affected_cells':int(affected.sum()),'state_difference':float(difference.norm())})
                if horizon==0:samples.append({'document_id':doc['id'],'anchor':anchor,**diversity(normal,normal_observation)})
                if horizon in plan['future_events']:
                    a=model.read(normal);b=model.read(perturbed);p=a.softmax(-1);q=b.softmax(-1);m=(p+q)/2
                    js=.5*((p*(p.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum()+(q*(q.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum())
                    target=torch.tensor([tokens[anchor+horizon+1]],device=state.device)
                    future.append({'future_events':horizon,'prediction_target':int(target.item()),
                        'prediction_nll_difference':float(F.cross_entropy(b,target)-F.cross_entropy(a,target)),
                        'prediction_js_divergence':max(0.,float(js)),'state_difference':micro[-1]['state_difference'],
                        'affected_cells':micro[-1]['affected_cells']})
            pulses.append({'document_id':doc['id'],'requested_fraction':fraction,'actual_anchor':anchor,'source_cell':source,
                           'amplitude':amplitude,'first_affected_microstep':first,'microsteps':micro,'future_predictions':future})
    model.observe=False;model.observation=[]
    return ({'status':'FAILED' if violations else 'COMPLETE','pulse_count':len(pulses),'causal_cone_violations':violations,
             'pulses':pulses,'interpretation':'state propagation is distinct from predictive utility; logits use global readout'},
            {'status':'COMPLETE','samples':samples,'interpretation':'numerical state organization only; no semantic specialization claim'})


@torch.no_grad()
def inference(model,document,plan,deadline):
    model.eval();state=model.init_state(1);token=document['tokens'][0]
    def measure(call):
        for _ in range(plan['inference_warmup']):check_time(deadline);call()
        values=[]
        for _ in range(plan['inference_repetitions']):
            check_time(deadline);sync(model);started=time.perf_counter();call();sync(model);values.append(time.perf_counter()-started)
        return {'median_seconds':statistics.median(values),'p95_seconds':sorted(values)[math.ceil(.95*len(values))-1],
                'repetitions':len(values),'warmup':plan['inference_warmup']}
    def prefill():
        h=model.init_state(1)
        for token in document['tokens'][:32]:h,_=model.step(h,[token])
    injected=model.inject(state,[token])
    return {'status':'COMPLETE','batch':1,'decode':measure(lambda:model.step(state,[token])),
            'single_microstep':measure(lambda:model.evolve(injected,1)),
            'four_microsteps':measure(lambda:model.evolve(injected,4)),
            'prefill':{'tokens':min(32,len(document['tokens'])),**measure(prefill)}}


def compute(model,entry):
    cells=model.config.cells;dim=model.config.dim;degree=model.graph['degree'];steps=model.config.steps
    return {'status':'COMPLETE','projection_flops_per_microstep':2*3*cells*dim*dim,
            'projection_flops_per_event':steps*2*3*cells*dim*dim,
            'aggregation_scalar_operations_per_microstep':4*cells*dim if degree==4 else 0,
            'directed_cross_cell_neighbor_inputs_per_microstep':cells*degree,
            'neighbor_feature_reads_per_microstep':cells*degree*dim,
            'aggregation_formula':'degree 4: two sums and two mean scalings per Cell feature; projection count unchanged',
            'scope':'theoretical logical counts; excludes full model FLOPs; not physical DRAM traffic or kernel measurements',
            'training_tokens_per_second':entry['tokens_per_second'],'training_wall_seconds':entry['training_wall_seconds']}


@torch.no_grad()
def generate(model,tokenizer,prompts,max_tokens,deadline):
    model.eval();samples=[]
    for prompt in prompts:
        check_time(deadline);state=model.init_state(1);ids=tokenizer.encode(prompt).ids
        for token in ids:state,logits=model.step(state,[token])
        generated=[]
        for _ in range(max_tokens):
            check_time(deadline);token=int(logits.argmax(-1).item());generated.append(token)
            if token==3:break
            state,logits=model.step(state,[token])
        visible=[t for t in generated if t!=3];bigrams=list(zip(visible,visible[1:]));trigrams=list(zip(visible,visible[1:],visible[2:]))
        samples.append({'prompt':prompt,'text':tokenizer.decode(generated,skip_special_tokens=True),'token_ids':generated,
                        'generated_tokens':len(generated),'eos':bool(generated and generated[-1]==3),
                        'distinct_1':len(set(visible))/max(1,len(visible)),'distinct_2':len(set(bigrams))/max(1,len(bigrams)),
                        'repeated_3gram_rate':(len(trigrams)-len(set(trigrams)))/max(1,len(trigrams))})
    return {'status':'COMPLETE','samples':samples,'eos_rate':sum(r['eos'] for r in samples)/len(samples),
            'average_generated_tokens':statistics.mean(r['generated_tokens'] for r in samples),
            'distinct_1':statistics.mean(r['distinct_1'] for r in samples),'distinct_2':statistics.mean(r['distinct_2'] for r in samples),
            'repeated_3gram_rate':statistics.mean(r['repeated_3gram_rate'] for r in samples),'auxiliary_only':True}
