"""Measure record candidate recall on MA-001.1 development labels only."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'ma-001-1')]
from run import BASE, embeddings, load, dump, Registry, Ranker, torch
from addressing import tokens


def discovery(ranker, query, dense):
    terms=sorted(set(tokens(query)))
    phrase=' '+' '.join(tokens(query))+' '
    bm25=[sum(math.log(1+(len(ranker.docs)-ranker.df[t]+.5)/(ranker.df[t]+.5))*doc[t]*2.2/
              (doc[t]+1.2*(.25+.75*sum(doc.values())/max(ranker.avg,1))) for t in terms if doc[t])
          for doc in ranker.docs]
    entity=[sum(a in phrase and a in ' '+' '.join(tokens(r['raw']))+' '
                for alias in set(r['entities']+r['aliases']) if tokens(alias)
                for a in [' '+' '.join(tokens(alias))+' ']) for r in ranker.rows]
    channels=dict(dense=dense,bm25=bm25,entity=entity)
    ranks={k:{i:j for j,i in enumerate(sorted((i for i,s in enumerate(v) if s>0),
                                              key=lambda i:(-v[i],i)),1)} for k,v in channels.items()}
    rows=[]
    for i,r in enumerate(ranker.rows):
        scores={k:v[i] for k,v in channels.items()}
        positions={k:v.get(i) for k,v in ranks.items()}
        fused=sum(1/(60+j) for j in positions.values() if j is not None)
        rows.append(dict(event=r['canonical_event_id'],cell=r['cell_id'],scores=scores,ranks=positions,
                         rrf=fused))
    return sorted(rows,key=lambda r:-r['rrf'])


def main():
    p=argparse.ArgumentParser()
    for key in ('snapshot','index','output'):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True)
    reg=Registry(a.index/'index.sqlite')
    records=reg.records()
    reg.db.close()
    saved=torch.load(a.index/'index.pt',weights_only=True)
    if saved['record_ids']!=[r['address_id'] for r in records]:
        raise ValueError('Stored embedding order differs')
    annot=json.loads((ROOT.parent/'ma-001-1/event-labels.json').read_text())
    cases=[]
    for q in load('queries.dev'):
        cells=annot.get(q['query_id'])
        if cells:
            grades={(c,e):g for c,v in cells.items()
                    for cat,g in [('primary',3),('supporting',2)] for e in v[cat]}
            cases.append((q['query_id'],q['query'],grades))
    with (ROOT.parent/'ma-001-1/paired.tsv').open() as f:
        for i,(cell,event,yes,no) in enumerate(csv.reader(f,delimiter='|')):
            cases.extend([(f'pair-{i}-valid',yes,{(cell,event):3}),
                          (f'pair-{i}-weak',no,{(cell,event):1})])
    encode=embeddings(False,a.output,a.snapshot)
    vectors=encode(['query: '+q for _,q,_ in cases])
    dense=(vectors@saved['raw'].T).tolist()
    runs=[]
    for (qid,q,grades),scores in zip(cases,dense):
        ranked=discovery(Ranker(records,raw=True),q,scores)
        candidates=ranked[:32]
        for row in candidates:
            row['grade']=grades.get((row['cell'],row['event']))
        positives={k for k,v in grades.items() if v==3}
        hits=lambda rows:len(positives & {(r['cell'],r['event']) for r in rows})
        dense_order=sorted(candidates,key=lambda r:-r['scores']['dense'])
        runs.append(dict(query_id=qid,query=q,grade3_count=len(positives),hits_at_8=hits(ranked[:8]),
                         hits_at_16=hits(ranked[:16]),hits_at_32=hits(candidates),
                         dense_top1_grade=dense_order[0]['grade'],candidates=candidates))
    dump(a.output/'candidate-traces.json',runs)
    n=sum(r['grade3_count'] for r in runs)
    summary=dict(queries=len(runs),annotated_direct_events=n,
                 **{k:sum(r[k] for r in runs) for k in ('hits_at_8','hits_at_16','hits_at_32')},
                 dense_top1_direct=sum(r['dense_top1_grade']==3 for r in runs),
                 unjudged_dense_top1=sum(r['dense_top1_grade'] is None for r in runs))
    dump(a.output/'summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
