"""Shared M1 episodes plus explicit, model-independent evaluation task transforms."""
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import torch
from data import MQARConfig,episode,digest,FACT,QUERY,ANSWER,BOS,HOLE
from e0 import masked_loss


def variant(config,split,index,reference=None,similarity='random',repeat=False,task='mqar'):
    row=episode(config,split,index)
    if task=='induction':
        pairs=row['bindings'];tail=row['input_ids'][-2];tokens=[BOS]
        for k,v in pairs:tokens.extend((8+k,264+v))
        room=config.length-len(tokens)-1;gap=row['input_ids'][1+3*config.bindings:-3] or [HOLE]
        tokens += [gap[i%len(gap)] for i in range(room)]+[tail]
        row['input_ids']=tokens;row['labels']=[-100]*(len(tokens)-1)+[dict(pairs)[tail-8]+264]
        row['query_mask']=[False]*(len(tokens)-1)+[True];row['query_positions']=[len(tokens)-1]
    elif task=='selective_copy':
        # Explicit task variant: unmarked value distractors in the M1 noise span.
        for i in range(1+3*config.bindings,config.length-3):
            if i%3==0:row['input_ids'][i]=264+(i*17+index)%256
    if similarity!='random':
        if reference is None or config.regime!='unseen_pairs' or config.queries!=1:raise ValueError('similarity control needs fixed reference and single query')
        anchor=row['input_ids'][row['query_positions'][0]-1]-8
        embeddings=torch.nn.functional.normalize(reference[8:264].float(),dim=-1)
        scores=embeddings@embeddings[anchor];mapping={anchor:anchor};used={anchor}
        order=scores.argsort(descending=similarity=='high',stable=True).tolist()
        for old,_ in row['bindings']:
            if old==anchor:continue
            chosen=next(k for k in order if k%3==old%3 and k not in used);mapping[old]=chosen;used.add(chosen)
        row['input_ids']=[8+mapping[t-8] if 8<=t<264 and t-8 in mapping else t for t in row['input_ids']]
        row['bindings']=[(mapping[k],v) for k,v in row['bindings']]
    if repeat:
        representatives={}
        for k,v in row['bindings']:representatives.setdefault(k%3,v)
        row['bindings']=[(k,representatives[k%3]) for k,v in row['bindings']]
        bindings=dict(row['bindings'])
        for pos,t in enumerate(row['input_ids'][:-2]):
            if t==FACT:row['input_ids'][pos+2]=264+bindings[row['input_ids'][pos+1]-8]
        for pos in row['query_positions']:row['labels'][pos]=264+bindings[row['input_ids'][pos-1]-8]
    if task!='mqar' or repeat or similarity!='random':row['id']=digest([row['id'],task,repeat,similarity,hashlib.sha256(reference[8:264].contiguous().numpy().tobytes()).hexdigest() if similarity!='random' else None])
    row['task']=task;row['similarity']=similarity;row['value_repeat']=repeat
    return row


def cases(protocol,reference):
    base=protocol['mqar'];entries=[];seen=set()
    def add(name,changes=None,task='mqar',similarity='random',repeat=False):
        c=MQARConfig(**{**base,**(changes or {})});identity=digest([asdict(c),task,similarity,repeat])
        if identity in seen:return
        seen.add(identity);row=variant(c,'validation',1000+len(entries),reference,similarity,repeat,task)
        entries.append({'name':name,'row':row,'config':asdict(c),'task':task,'similarity':similarity,'value_repeat':repeat})
    for length in protocol['evaluation']['lengths']:add('length-'+str(length),{'length':length})
    for bindings in protocol['evaluation']['bindings']:add('bindings-'+str(bindings),{'bindings':bindings})
    for q in protocol['evaluation']['queries']:add('queries-'+str(q),{'bindings':32,'queries':q})
    for ratio in protocol['evaluation']['interference']:add('interference-'+str(ratio),{'interference':ratio})
    for mode in protocol['evaluation']['key_similarity']:add('similarity-'+mode,similarity=mode)
    add('repeated-values',repeat=True)
    for multiple in protocol['evaluation']['delay_multipliers']:
        delay=int(protocol['model']['window']*multiple);add('delay-'+str(delay),{'bindings':1,'length':delay+4})
    add('selective-copy',{'bindings':1},task='selective_copy');add('induction',{'bindings':8},task='induction')
    return entries


@torch.no_grad()
def evaluate(model,rows,device,collect=True):
    model.eval();loss=0.;targets=correct=0;hits=coverage=0;queries=0;details=[]
    for row in rows:
        ids=torch.tensor([row['input_ids']],device=device);labels=torch.tensor([row['labels']],device=device)
        q=row.get('query_positions',[i for i,v in enumerate(row['query_mask']) if v])
        logits,state,trace=model.forward_state(ids,trace_queries=[q],read_positions=q if q else None)
        selected_labels=labels[:,q] if q else labels
        n=int((selected_labels!=-100).sum());loss+=float(masked_loss(logits,selected_labels))*n;targets+=n
        predictions=logits.argmax(-1)[0];query_predictions=predictions.tolist() if q else []
        correct+=sum(pred==row['labels'][p] for pred,p in zip(query_predictions,q));queries+=len(q)
        for pos in q:
            key=row['input_ids'][pos-1]-8 if row.get('task')!='induction' else row['input_ids'][pos]-8
            gold=([2+2*i for i,(k,v) in enumerate(row['bindings']) if k==key] if row.get('task')=='induction' else
                  [i+2 for i,t in enumerate(row['input_ids'][:-2]) if t==FACT and row['input_ids'][i+1]==key+8])
            relevant=[r for r in trace if r['query_position']==pos]
            if gold:
                hits+=int(any(any(p in gold for p in r['selected_positions']) for r in relevant))
                coverage+=int(any(any(p in gold for p in r['stored_positions']) for r in relevant))
        if collect:details.append({'episode':row['id'],'input_sha256':digest(row['input_ids']),
            'targets':n,'correct':sum(pred==row['labels'][p] for pred,p in zip(query_predictions,q)),
            'store_metrics':[s.metrics(state.position) for s in state.stores] if state.stores else [],
            'resources':model.resources(state),'trace':trace})
    return {'nll':loss/targets,'query_exact_match':correct/queries if queries else None,'recall_at_1':correct/queries if queries else None,
        'supervised_targets':targets,'query_targets':queries,'store_retrieval_hit_rate':hits/queries if queries else None,
        'gold_record_coverage':coverage/queries if queries else None,'hit_definition':'retrieved gold value token position; not admission-score rank',
        'episodes':details}
