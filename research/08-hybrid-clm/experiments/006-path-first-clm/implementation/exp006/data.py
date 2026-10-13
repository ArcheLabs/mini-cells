"""Story-contained non-overlapping windows; train/dev only before Gate 4."""
from __future__ import annotations
import hashlib
import numpy as np

def story_digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
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
