"""MA-001.1 development-only diagnosis and separate-process readback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'ma-001'
sys.path[:0] = [str(BASE/'scripts'),str(BASE)]
from run_experiment import load, dump, embeddings, evaluate, metrics
from addressing import Registry, Ranker
import torch

QUERY = 'Compare GPU-PVM execution, portable cell restart, and Titans Neural Memory recall.'
EXPECTED = {'minijam','minicells','ai_memory'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('snapshot',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--phase',choices=['all','write','restore'],default='all')
    a=p.parse_args()
    out=a.output.resolve()
    if a.phase=='all':
        out.mkdir(parents=True)
        for phase in ('write','restore'):
            subprocess.run([sys.executable,__file__,str(a.snapshot),str(out),'--phase',phase],check=True)
        return
    encode=embeddings(False,out,a.snapshot)
    db=out/'index.sqlite'
    if a.phase=='write':
        reg=Registry(db)
        for c in load('cells'):
            reg.create(c['id'],c['category'],c['id'],c['session'])
        for e in load('events'):
            reg.write(e['id'],e['raw'],e['descriptor'],explicit=e['targets'])
        reg.write('projection','GPU-PVM and portable learned cells',
                  dict(summary='joint execution review',entities=['joint execution review'],topics=[],aliases=[]),
                  explicit=['minijam','minicells'])
        records=reg.records()
        reg.db.close()
        rank=Ranker(records,raw=True)
        descriptor=Ranker(records)
        data=dict(record_ids=[r['address_id'] for r in records],
                  raw=encode(['passage: '+x for x in rank.text]),
                  descriptor=encode(['passage: '+x for x in descriptor.text]))
        torch.save(data,out/'index.pt')
        dump(out/'writer.json',dict(pid=os.getpid(),database_sha256=hashlib.sha256(db.read_bytes()).hexdigest(),
                                    embeddings_sha256=hashlib.sha256((out/'index.pt').read_bytes()).hexdigest()))
        return

    writer=json.loads((out/'writer.json').read_text())
    if writer['pid']==os.getpid():
        raise RuntimeError('Restore requires an independent process')
    for name,key in [('index.sqlite','database_sha256'),('index.pt','embeddings_sha256')]:
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=writer[key]:
            raise RuntimeError(f'Changed artifact: {name}')
    stored=torch.load(out/'index.pt',weights_only=True)
    reg=Registry(db)
    records=reg.records()
    reg.db.close()
    if stored['record_ids']!=[r['address_id'] for r in records]:
        raise RuntimeError('Embedding and address row order mismatch')
    rank=Ranker(records,raw=True)
    frozen=json.loads((BASE/'artifacts/enriched-cpu/frozen.json').read_text())
    threshold=frozen['thresholds']['E3-Raw']
    scenario=(encode(['query: '+QUERY])@stored['raw'].T)[0].tolist()
    returned=rank.route(rank.scores(QUERY,'E3-Raw',scenario),threshold)
    dump(out/'cross-process.json',dict(writer_pid=writer['pid'],reader_pid=os.getpid(),
         query=QUERY,expected_cell_ids=sorted(EXPECTED),returned_cell_ids=returned,
         status='PASS' if set(returned)==EXPECTED else 'FAIL'))

    queries=load('queries.dev')
    for i,(kind,q) in enumerate([
        ('weak_match','What was the retention decision?'),
        ('weak_match','Which account should I look at?'),
        ('weak_match','What did we change about the model update?'),
        ('ambiguous','Which of the two configurations did we pick?'),
        ('no_match','What was the catering budget for the annual retreat?'),
    ]):
        queries.append(dict(query_id=f'diagnostic-{i}',query_type=kind,query=q,
                            relevant_cell_ids=[],expected_action='abstain'))
    vectors=encode(['query: '+q['query'] for q in queries])
    dense=(vectors@stored['raw'].T).tolist()
    annotated=[]
    variants={name:[] for name in ('raw_all','raw_dense_bm25','raw_dense_entity','raw_bm25_entity','raw_structured_metadata')}
    for q, ds, ms in zip(queries,dense,(vectors@stored['descriptor'].T).tolist()):
        # Score every record separately: singleton BM25 would change corpus IDF.
        from addressing import tokens
        from math import log
        terms=set(tokens(q['query']))
        bm=[sum(log(1+(len(rank.docs)-rank.df[t]+.5)/(rank.df[t]+.5))*doc[t]*2.2/
                (doc[t]+1.2*(.25+.75*sum(doc.values())/max(rank.avg,1))) for t in terms if doc[t])
            for doc in rank.docs]
        phrase=' '+' '.join(tokens(q['query']))+' '
        def entity(r, raw):
            text=' '+' '.join(tokens(r['raw']))+' '
            return sum(a in phrase and (not raw or a in text) for a in
                       [' '+' '.join(tokens(x))+' ' for x in set(r['entities']+r['aliases']) if tokens(x)])
        ent=[entity(r,True) for r in records]
        metadata=[entity(r,False) for r in records]
        def ranks(scores):
            return {i:j for j,i in enumerate(sorted((i for i,s in enumerate(scores) if s>0),
                                                   key=lambda i:(-scores[i],i)),1)}
        channels={k:ranks(v) for k,v in [('dense',ds),('bm25',bm),('entity',ent),('metadata',metadata)]}
        def fuse(keys):
            grouped={}
            fused=[]
            for i,r in enumerate(records):
                s=sum(1/(60+channels[k][i]) for k in keys if i in channels[k])
                c=r['cell_id']
                grouped[c]=max(s,grouped.get(c,0))
                fused.append(s)
            return sorted(grouped.items(),key=lambda x:(-x[1],x[0])),fused
        specs=dict(raw_all=('dense','bm25','entity'),raw_dense_bm25=('dense','bm25'),
                   raw_dense_entity=('dense','entity'),raw_bm25_entity=('bm25','entity'),
                   raw_structured_metadata=('dense','bm25','metadata'))
        baseline, fused=fuse(specs['raw_all'])
        reference = rank.scores(q['query'],'E3-Raw',ds)
        if any(abs(dict(baseline)[c]-s)>1e-12 for c,s in reference):
            raise AssertionError('Record trace disagrees with frozen Raw Hybrid')
        baseline = reference
        returned=rank.route(baseline,threshold)
        missing=set(q['relevant_cell_ids'])-set(returned)
        per_cell={}
        for cell in sorted(missing):
            ids=[i for i,r in enumerate(records) if r['cell_id']==cell]
            pos=next((j for j,(c,s) in enumerate(baseline,1) if c==cell),None)
            per_cell[cell]=dict(indexed=bool(ids),cell_rank=pos,
                outcome='missing_cell' if pos is None else 'below_top3' if pos>3 else 'below_threshold',
                records=[dict(id=records[i]['address_id'],raw=records[i]['raw'],
                    descriptor=records[i]['summary'],fused=fused[i],dense_score=ds[i],
                    descriptor_dense_score=ms[i],channel_ranks={k:v.get(i) for k,v in channels.items()})
                    for i in ids])
        category={'insufficient':'ambiguous'}.get(q['query_type'],q['query_type'])
        if q['relevant_cell_ids']:
            category='valid_match'
        annotated.append(dict(**q,rejection_category=category,returned=returned,missing=per_cell,
             top_score=baseline[0][1],margin=baseline[0][1]-baseline[1][1]))
        for name,keys in specs.items():
            ranking,_=fuse(keys)
            variants[name].append((q,ranking))
    dump(out/'diagnostics.json',annotated)
    summary={}
    for name,rows in variants.items():
        fitted=rows[:24]
        options={0.,*(score for _,ranking in fitted for _,score in ranking)}
        def quality(t):
            return sum(set(rank.route(r,t))==set(q['relevant_cell_ids']) for q,r in fitted)
        t=threshold if name=='raw_all' else max(options,key=lambda x:(quality(x),x))
        summary[name]=dict(threshold=t,exact_matches=quality(t),development_queries=len(fitted),
            probe_false_routes=[q['query_id'] for q,r in rows[24:] if rank.route(r,t)])
    dump(out/'ablations.json',summary)


if __name__=='__main__':
    main()
