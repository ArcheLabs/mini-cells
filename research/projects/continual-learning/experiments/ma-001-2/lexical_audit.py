"""Automatic text-only diagnostic; no label-dependent scoring."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
base = root.parent / 'ma-001'
events = {e['id']: e['raw'] for e in map(json.loads, (base/'fixtures/events.jsonl').read_text().splitlines())}
stop = set('a an the what which who how when where is are do did does for from of to on in it that they we my and can was were'.split())

def terms(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - stop

rows = []
for case in json.loads((root/'artifacts/candidate-traces.json').read_text()):
    query = terms(case['query'])
    candidates = []
    for c in case['candidates'][:16]:
        document = terms(events.get(c['event'], 'GPU-PVM and portable learned cells' if c['event']=='projection' else ''))
        candidates.append(dict(event=c['event'], cell=c['cell'], grade=c['grade'],
                               coverage=len(query & document)/max(1,len(query)),
                               dense=c['scores']['dense'], uncovered=sorted(query-document)))
    lexical = max(candidates,key=lambda c:c['coverage'])
    dense = max(candidates,key=lambda c:c['dense'])
    rows.append(dict(id=case['query_id'], lexical=lexical, dense=dense, candidates=candidates))
by_id = {r['id']:r for r in rows}
summary = {name:dict(direct_top1=sum(r[name]['grade']==3 for r in rows),
                     unjudged_top1=sum(r[name]['grade'] is None for r in rows),
                     valid_pair_score_higher=sum(
                         by_id[f'pair-{i}-valid'][name][score] > by_id[f'pair-{i}-weak'][name][score]
                         for i in range(10)))
           for name,score in [('lexical','coverage'),('dense','dense')]}
(root/'artifacts/lexical-audit.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n')
print(summary)
