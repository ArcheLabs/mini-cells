"""Frozen development probes; no threshold selection or holdout access."""
import argparse
import json
from pathlib import Path

from run import BASE, HERE, Ranker, Registry, dump, embeddings, torch

p=argparse.ArgumentParser()
p.add_argument('snapshot',type=Path)
p.add_argument('index',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
encode=embeddings(False,a.output,a.snapshot)
registry=Registry(a.index/'index.sqlite')
records=registry.records()
registry.db.close()
index=torch.load(a.index/'index.pt',weights_only=True)
assert index['record_ids']==[r['address_id'] for r in records]
ranker=Ranker(records,raw=True)
threshold=json.loads((BASE/'artifacts/enriched-cpu/frozen.json').read_text())['thresholds']['E3-Raw']
pairs=[line.split('|') for line in (HERE/'paired.tsv').read_text().splitlines()]
vectors=encode(['query: '+q for _,_,yes,no in pairs for q in (yes,no)])
scores=(vectors@index['raw'].T).tolist()
rows=[]
for i,(cell,event,*queries) in enumerate(pairs):
    target=next(j for j,r in enumerate(records) if r['cell_id']==cell and r['canonical_event_id']==event)
    row=dict(cell=cell,event=event)
    for j,(label,q) in enumerate(zip(('valid','weak'),queries)):
        dense=scores[2*i+j]
        ranked=ranker.scores(q,'E3-Raw',dense)
        row[label]=dict(query=q,returned=ranker.route(ranked,threshold),top_score=ranked[0][1],
                        target_similarity=dense[target],
                        target_dense_rank=1+sum(s>dense[target] for s in dense))
    rows.append(row)
dump(a.output/'pairs.json',rows)
summary=dict(pairs=len(rows),valid_target_routed=sum(r['cell'] in r['valid']['returned'] for r in rows),
    weak_false_routes=sum(bool(r['weak']['returned']) for r in rows),
    jointly_correct=sum(r['valid']['returned']==[r['cell']] and not r['weak']['returned'] for r in rows),
    valid_similarity_higher=sum(r['valid']['target_similarity']>r['weak']['target_similarity'] for r in rows),
    valid_fusion_score_higher=sum(r['valid']['top_score']>r['weak']['top_score'] for r in rows))
dump(a.output/'summary.json',summary)
print(summary)
