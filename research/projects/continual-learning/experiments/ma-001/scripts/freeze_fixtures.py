"""Serialize the authored fixture tables with a fixed stratified split."""
from collections import defaultdict
import json
from pathlib import Path
import random

root = Path(__file__).resolve().parents[1] / 'fixtures'


def save(name, rows):
    (root / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
                             encoding='utf-8')


context = {}
for line in (root/'event-context.tsv').read_text().splitlines():
    cell, topics, body = line.split('|')
    context[cell] = (topics.split(';'), body)
aliases = {'GPU-PVM':['GPU PVM','parallel VM'], 'Titans Neural Memory':['神经记忆'],
           'portable cell':['learned module restart'], 'compiler lowering':['source translation'],
           'gradient accumulation':['microbatch accumulation'], 'API versioning':['interface compatibility']}
cells, events = [], []
for line in (root/'events.tsv').read_text().splitlines():
    cell, category, topics = line.split('|')
    cells.append(dict(id=cell, category=category, session='a' if cell in
                      ('minijam','minicells','jamscript') else 'b'))
    for i, topic in enumerate(topics.split(';'), 1):
        scope, body = context[cell]
        note = f'{body} Review item: {topic}.'
        events.append(dict(id=f'{cell}-{i:02}', targets=[category], raw=note,
                           descriptor=dict(summary=topic, entities=[topic], topics=scope, aliases=aliases.get(topic, []))))
save('cells.jsonl', cells)
save('events.jsonl', events)
groups = defaultdict(list)
for i,line in enumerate((root/'queries.tsv').read_text().splitlines(), 1):
    kind, ids, query = line.split('|')
    groups[kind].append(dict(query_id=f'q{i:03d}', query=query, query_type=kind,
        relevant_cell_ids=[] if ids=='-' else ids.split(','), expected_action='abstain' if ids=='-' else 'route'))
rng = random.Random(42)
dev, holdout = [], []
for kind, count in zip(('direct','paraphrase','multi','hard','no_match','insufficient'),(5,5,5,3,2,4)):
    rng.shuffle(groups[kind])
    dev.extend(groups[kind][:count])
    holdout.extend(groups[kind][count:])
save('queries.dev.jsonl',dev)
save('queries.holdout.jsonl',holdout)
