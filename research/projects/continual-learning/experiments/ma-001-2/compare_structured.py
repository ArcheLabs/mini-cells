"""Oracle-input diagnostic; does not claim natural language extraction."""
import json
from pathlib import Path
from evidence import Claim, route

root = Path(__file__).resolve().parent
load = lambda name: json.loads((root/name).read_text())
fixture = load('structured-audit.json')
traces = {r['query_id']: r for r in load('artifacts/candidate-traces.json')}
claims = {e: [Claim(*c) for c in cs] for e,cs in fixture['claims'].items()}
rows = []
for case in fixture['cases']:
    candidates = traces[case['id']]['candidates'][:16]
    # Every retrieved candidate is passed to the assessor. An absent claim
    # annotation provides no evidence; it is not a gold grade of zero.
    structured = route([Claim(*f) for f in case['facets']],
                       [(c['event'],c['cell'],claims.get(c['event'],[])) for c in candidates])
    top = max(candidates,key=lambda c:c['scores']['dense'])
    direct = {e for e,g in case['grades'].items() if g==3}
    rows.append(dict(id=case['id'],dense_top_event=top['event'],
                     dense_top_grade=case['grades'].get(top['event']),
                     direct_candidates=sorted(direct & {c['event'] for c in candidates}),
                     graded_candidates={c['event']:case['grades'][c['event']] for c in candidates
                                        if c['event'] in case['grades']},
                     candidate_claim_coverage=sum(c['event'] in claims for c in candidates),
                     structured=structured))
(root/'artifacts/structured-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
    print(r['id'],r['dense_top_event'],r['dense_top_grade'],r['structured']['status'],r['structured']['primary_cell_ids'])
