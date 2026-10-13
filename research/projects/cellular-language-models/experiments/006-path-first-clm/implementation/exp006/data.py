"""Story-contained non-overlapping windows; train/dev only before Gate 4."""
from __future__ import annotations
import hashlib
import numpy as np

def story_digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def select_unique_pilot_stories(rows,train_range,development_range,raw_limit):
    """First occurrence wins, before split assignment; never scan reserved rows."""
    high=development_range[1];seen=set();train=[];dev=[];indices={'train':[],'development':[]}
    scanned=duplicates=0;unique=0
    iterator=iter(rows)
    while scanned<raw_limit and unique<high:
        try:row=next(iterator)
        except StopIteration:break
        raw_index=scanned;scanned+=1;text=row['text'];sha=story_digest(text)
        if sha in seen:
            duplicates+=1;continue
        seen.add(sha);ordinal=unique;unique+=1
        for role,bounds,target in (('train',train_range,train),('development',development_range,dev)):
            if bounds[0]<=ordinal<bounds[1]:
                target.append(text);indices[role].append(raw_index)
    if len(train)!=train_range[1]-train_range[0] or len(dev)!=development_range[1]-development_range[0]:
        raise ValueError('insufficient unique pilot stories before reserved confirmation boundary')
    audit={'policy':'pilot_global_exact_first_occurrence_v1','range_units':'unique story ordinal',
           'raw_rows_scanned':scanned,'duplicates_removed':duplicates,'unique_stories_scanned':unique,
           'raw_scan_limit_exclusive':raw_limit,'raw_row_indices':indices}
    return train,dev,audit

def encode_stories(texts,tokenizer,length):
    eos=tokenizer.token_to_id('<eos>');ids=[];starts=[];groups=[];hashes=[]
    if eos is None:raise ValueError('tokenizer must define EOS')
    for i,text in enumerate(texts):
        hashes.append(story_digest(text));base=len(ids);row=tokenizer.encode(text).ids+[eos];ids.extend(row)
        for offset in range(0,len(row)-length,length):starts.append(base+offset);groups.append(i)
    if len(set(hashes))!=len(hashes):raise ValueError('duplicate stories within split')
    return np.asarray(ids,dtype=np.int32),starts,groups,hashes

def assert_disjoint(train,dev):
    if set(train)&set(dev):raise ValueError('story content leakage between train and development')

def select_windows(starts,groups,count,seed):
    if len(starts)<count:raise ValueError('insufficient independent evaluation windows')
    rng=np.random.default_rng(seed);idx=sorted(rng.choice(len(starts),count,replace=False).tolist())
    return [starts[i] for i in idx],[groups[i] for i in idx]
