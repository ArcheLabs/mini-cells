"""Join development annotations to frozen traces without reading holdout."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
load = lambda path: json.loads((ROOT / path).read_text())
labels = load('event-labels.json')
trace = {r['query_id']:r for r in load('artifacts/diagnostics.json')}
rows = []
for qid, cells in labels.items():
    for cell, annotation in cells.items():
        if cell not in trace[qid]['missing']:
            continue
        miss = trace[qid]['missing'][cell]
        indexed = {json.loads(r['id'])[0]:r for r in miss['records']}
        primary = [dict(event=e, evidence=indexed.get(e)) for e in annotation['primary']]
        winner = max(indexed.values(),key=lambda r:r['fused'])
        rows.append(dict(query_id=qid,cell_id=cell,primary=primary,winner=winner,
                         annotation=annotation,outcome=miss['outcome']))
weak = [r for r in trace.values() if r['rejection_category']=='weak_match']
summary = dict(events=rows, weak=[{k:r[k] for k in ('query_id','top_score','margin','returned')} for r in weak],
               scalar_threshold_counterexample=dict(valid='q018', weak='diagnostic-2',
               valid_score=trace['q018']['top_score'],weak_score=trace['diagnostic-2']['top_score'],
               valid_margin=trace['q018']['margin'],weak_margin=trace['diagnostic-2']['margin']))
(ROOT/'artifacts/event-diagnosis.json').write_text(json.dumps(summary,indent=2)+'\n')
for r in rows:
    print(r['query_id'],r['cell_id'],[(x['event'],x['evidence']['channel_ranks']) for x in r['primary']],r['winner']['id'])
