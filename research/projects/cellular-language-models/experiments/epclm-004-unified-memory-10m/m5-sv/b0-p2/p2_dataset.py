"""Deterministic balanced holdout of the existing B0 development pairs only."""
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'b0/src'))
from b0_data import prepare,validate_row
from data import digest,file_hash
from e0 import write


def jsd(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=a/a.sum();b=b/b.sum();m=(a+b)/2
    return float(sum(np.sum(x[x>0]*np.log2(x[x>0]/m[x>0])) for x in (a,b))/2)


def degrees(pairs):
    keys=np.zeros(256,dtype=int);values=keys.copy()
    for k,v in pairs:keys[k]+=1;values[v]+=1
    return keys,values


def matching(edges,rng):
    adjacency={k:[] for k in range(256)}
    order=list(sorted(edges));rng.shuffle(order)
    for k,v in order:adjacency[k].append(v)
    matched={}
    def augment(k,seen):
        for v in adjacency[k]:
            if v in seen:continue
            seen.add(v)
            if v not in matched or augment(matched[v],seen):matched[v]=k;return True
        return False
    keys=list(range(256));rng.shuffle(keys)
    if not all(augment(k,set()) for k in keys):raise ValueError('STOP: no complete symbol-covering matching')
    return {(k,v) for v,k in matched.items()}


def balanced_split(pool,seed=75029,attempts=32):
    """Two disjoint matchings give validation degree1..2 after six removals."""
    pool=set(pool);allk,allv=degrees(pool)
    if len(pool)!=4252 or min(allk)==0 or min(allv)==0:raise ValueError('STOP: candidate coverage/count differs')
    best=None
    for attempt in range(attempts):
        rng=random.Random(seed+attempt)
        first=matching(pool,rng);second=matching(pool-first,rng);validation=first|second
        buckets=Counter((k+v)%3 for k,v in validation)
        #506 validation pairs:60 from original bucket1,446 from bucket0.
        #Targets are the nearest integer stratified allocation, not new modulo partitions.
        required={0:buckets[0]-446,1:buckets[1]-60}
        if any(n<0 for n in required.values()) or sum(required.values())!=6:continue
        for bucket,number in required.items():
            for _ in range(number):
                vk,vv=degrees(validation)
                candidates=sorted(p for p in validation if (p[0]+p[1])%3==bucket and vk[p[0]]==2 and vv[p[1]]==2)
                if not candidates:break
                #Remove where candidate degrees are lowest; resulting marginal divergence
                #is evaluated below for every deterministic attempt.
                pair=min(candidates,key=lambda p:(allk[p[0]]+allv[p[1]],p))
                validation.remove(pair)
        if len(validation)!=506:continue
        training=pool-validation;tk,tv=degrees(training);vk,vv=degrees(validation)
        if min(vk)<1 or min(vv)<1 or max(vk)>2 or max(vv)>2:continue
        score=jsd(tk,vk)+jsd(tv,vv)
        candidate=(score,sorted(validation))
        if best is None or candidate<best:best=candidate
    if best is None:raise ValueError('STOP: balanced symbol coverage/source quotas infeasible')
    validation=set(map(tuple,best[1]));return pool-validation,validation


def observations(pairs,count,seed,candidate_bucket):
    """Cover every assigned pair; balance only the extra repeated observations."""
    result=list(sorted(pairs));rng=random.Random(seed);rng.shuffle(result)
    k,v=degrees(result);array=np.array(result);buckets=(array[:,0]+array[:,1])%3
    target1=round(count*candidate_bucket[1]/sum(candidate_bucket.values()))
    extras={1:target1-int((buckets==1).sum()),0:count-target1-int((buckets==0).sum())}
    if any(n<0 for n in extras.values()):raise ValueError('STOP: observation source quotas infeasible')
    for bucket in (0,1):
        for _ in range(extras[bucket]):
            scores=k[array[:,0]]+v[array[:,1]]
            eligible=np.where(buckets==bucket)[0]
            i=int(eligible[np.argmin(scores[eligible])]);pair=tuple(map(int,array[i]))
            result.append(pair);k[pair[0]]+=1;v[pair[1]]+=1
    rng.shuffle(result);return result


def pack(observed,seed):
    """Partition into eight-pair Episodes with unique Keys and Values."""
    for attempt in range(64):
        pending=list(observed);random.Random(seed+attempt).shuffle(pending);episodes=[]
        while pending:
            row=[];keys=set();values=set()
            for i in range(len(pending)-1,-1,-1):
                k,v=pending[i]
                if k not in keys and v not in values:
                    row.append(pending.pop(i));keys.add(k);values.add(v)
                    if len(row)==8:break
            if len(row)!=8:break
            episodes.append(row)
        if not pending and all(len(row)==8 for row in episodes):return episodes
    raise ValueError('STOP: cannot pack complete nonconflicting Episodes')


def replace_bindings(template,pairs,index,split,seed):
    row=copy.deepcopy(template);original={k:i for i,(k,v) in enumerate(row['bindings'])}
    for i,(k,v) in enumerate(pairs):row['input_ids'][2+3*i]=8+k;row['input_ids'][3+3*i]=264+v
    for pos in row['query_positions']:
        k,v=pairs[original[row['input_ids'][pos-1]-8]]
        row['input_ids'][pos-1]=8+k;row['labels'][pos]=264+v
    row.update(bindings=[list(p) for p in pairs],split=split,
        id=digest(['B0-P2-DATASET-v2',seed,split,index,pairs]))
    return row


def bank_audit(bank,assigned):
    observed=[tuple(p) for row in bank for p in row['bindings']];pairs=set(observed)
    if pairs!=set(assigned):raise ValueError('STOP: Episode bank does not cover its entire pair assignment')
    uk,uv=degrees(pairs);ok,ov=degrees(observed)
    hist=lambda x:dict(sorted(Counter(map(int,x)).items()))
    return {'episodes':len(bank),'input_tokens':len(bank)*512,'supervised_targets':len(bank)*8,
        'unique_pairs':len(pairs),'pair_observations':len(observed),'pair_duplicate_fraction':1-len(pairs)/len(observed),
        'maximum_pair_multiplicity':max(Counter(observed).values()),
        'pair_multiplicity_histogram':dict(sorted(Counter(Counter(observed).values()).items())),
        'most_repeated_pairs':Counter(observed).most_common(20),
        'episode_duplicate_fraction':1-len({digest(r['input_ids']) for r in bank})/len(bank),
        'unique_keys':int((uk>0).sum()),'unique_values':int((uv>0).sum()),
        'unique_key_degrees':uk.tolist(),'unique_value_degrees':uv.tolist(),
        'key_observation_counts':ok.tolist(),'value_observation_counts':ov.tolist(),
        'unique_key_degree_histogram':hist(uk),'unique_value_degree_histogram':hist(uv),
        'key_frequency_histogram':hist(ok),'value_frequency_histogram':hist(ov),
        'source_bucket_unique':dict(sorted(Counter((k+v)%3 for k,v in pairs).items())),
        'source_bucket_observations':dict(sorted(Counter((k+v)%3 for k,v in observed).items())),
        'content_sha256':digest(bank)}


def build(destination,seed=75029):
    destination=Path(destination);protocol=json.loads((BASE.parent/'b0/configs/protocol.json').read_text())
    old,previous=prepare(destination/'source-development',protocol)
    source={n:{tuple(p) for row in old[n] for p in row['bindings']} for n in ('pilot_train','validation')}
    if (len(source['pilot_train']),len(source['validation']),len(source['pilot_train']&source['validation']))!=(3746,506,0):
        raise ValueError('STOP: source candidate pool differs')
    lock=json.loads((BASE.parent/'b0/configs/dataset-lock.json').read_text())
    if any(digest(old[name])!=lock['banks'][name]['content_sha256'] for name in source):
        raise ValueError('STOP: reconstructed prior Episode hashes differ')
    pool=source['pilot_train']|source['validation'];training,validation=balanced_split(pool,seed)
    if training&validation:raise ValueError('STOP: Pair leakage')
    source_buckets=Counter((k+v)%3 for k,v in pool);banks={};reports={}
    for name,pairs,size in [('training',training,512),('validation',validation,64)]:
        observed=observations(pairs,size*8,seed+1,source_buckets);packed=pack(observed,seed+2)
        templates=old['pilot_train' if name=='training' else 'validation']
        bank=[replace_bindings(templates[i],pairs,i,'train' if name=='training' else name,seed) for i,pairs in enumerate(packed)]
        if len(bank)!=size:raise ValueError('STOP: Episode budget differs')
        for row in bank:validate_row(row,protocol['data'])
        banks[name]=bank;reports[name]=bank_audit(bank,pairs)
        write(destination/(name+'.json'),bank)
    a=reports['training'];b=reports['validation']
    report={'format':'B0-P2-DATASET-v2','split_seed':seed,'candidate_pairs':len(pool),'source_dataset_sha256':digest(previous),
        'source_development_hashes':{n:previous['banks'][n]['content_sha256'] for n in source},
        'candidate_pairs_sha256':digest(sorted(pool)),'train_validation_pair_overlap':len(training&validation),
        'banks':reports,'marginal_jsd_bits':{kind:jsd(a[key],b[key]) for kind,key in
            [('key_unique','unique_key_degrees'),('value_unique','unique_value_degrees'),
             ('key_observed','key_observation_counts'),('value_observed','value_observation_counts')]},
        'test_status':'RESERVED_NOT_GENERATED_NOT_ACCESSED','acceptance_status':'PENDING_EXPLICIT_MARGINAL_TOLERANCE',
        'limitations':['Candidate degrees are not uniform; training degree14..15 everywhere is infeasible.',
            'The supplied specification has no numerical marginal-JSD acceptance threshold. Do not infer PASS.']}
    differences={}
    for kind,key in [('key_unique','unique_key_degrees'),('value_unique','unique_value_degrees'),
                     ('key_observed','key_observation_counts'),('value_observed','value_observation_counts')]:
        x=np.asarray(a[key],dtype=float);y=np.asarray(b[key],dtype=float);delta=np.abs(x/x.sum()-y/y.sum())
        differences[kind]={'total_variation':float(delta.sum()/2),'max_abs_probability_difference':float(delta.max())}
    report['marginal_differences']=differences
    report['source_bucket_fraction_difference']={}
    for field in ('source_bucket_unique','source_bucket_observations'):
        x=a[field];y=b[field]
        report['source_bucket_fraction_difference'][field]={bucket:abs(x.get(bucket,0)/sum(x.values())-y.get(bucket,0)/sum(y.values())) for bucket in (0,1)}
    write(destination/'SPLIT-AUDIT.json',report)
    write(destination/'DATASET-MANIFEST.json',{'format':'B0-P2-DATASET-v2','split_seed':seed,
        'tokenizer_sha256':protocol['tokenizer_sha256'],'dataset_sha256':digest({n:digest(rows) for n,rows in banks.items()}),
        'files':{n+'.json':file_hash(destination/(n+'.json')) for n in banks},'split_audit_sha256':file_hash(destination/'SPLIT-AUDIT.json'),
        'test_status':report['test_status']})
    return banks,report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();banks,report=build(args.output);print(json.dumps({k:v for k,v in report.items() if k!='banks'},indent=2))
