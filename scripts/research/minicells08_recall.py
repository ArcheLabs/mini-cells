"""Paired development diagnostic using frozen Qwen and explicit Cell writes."""
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from minicells08_smoke import setup, state_hash


@torch.no_grad()
def main(output):
    runtime, identity = setup()
    device = next(runtime.memory.parameters()).device
    # Disjoint token sets and a fixed derangement for the wrong-Cell control.
    keys = torch.tensor([[101, 307, 509, 701, 907, 1103, 1301, 1601]], device=device)
    values = torch.tensor([[2003, 2203, 2411, 2609, 2801, 3001, 3203, 3407]], device=device)
    # Use fresh single-token episodes for both writes and queries.
    for name in ('correct', 'empty', 'permuted'):
        runtime.create(name)
    for _ in range(16):
        for key, value, wrong in zip(keys.split(1, 1), values.split(1, 1),
                                    values.roll(1, 1).split(1, 1)):
            for name, target in [('correct', value), ('permuted', wrong)]:
                runtime.generate(key, update_cell=name, learning_payload=(key, target))
    before = {name: state_hash(runtime.get(name)) for name in runtime.list()}
    candidates = values.flatten()
    candidate_hidden = torch.cat([runtime.backbone.encode(v)[0]
                                 for v in values.split(1, 1)], dim=1)
    candidate_vectors = F.normalize(runtime.memory.project(candidate_hidden.float())[1][0], dim=-1)
    rows = []
    for index, key in enumerate(keys.split(1, 1)):
        hidden, _ = runtime.backbone.encode(key)
        query = runtime.memory.project(hidden.float())[2]
        controls = {}
        for name in ('correct', 'empty', 'permuted'):
            result = runtime.generate(key, read_cells=(name,))
            raw = runtime.memory.recall(query, runtime.get(name))[0, 0]
            scores = F.normalize(raw, dim=-1) @ candidate_vectors.T
            candidate_logits = result['logits'][0, -1, candidates].float()
            if not scores.isfinite().all():
                raise FloatingPointError('Nonfinite candidate scores')
            controls[name] = dict(raw_scores=scores.tolist(),
                                  raw_correct=int(scores.argmax()) == index,
                                  candidate_logits=candidate_logits.tolist(),
                                  candidate_correct=int(candidate_logits.argmax()) == index,
                                  generated_token=int(result['tokens'].item()),
                                  generation_correct=int(result['tokens'].item()) == int(candidates[index]))
        rows.append(dict(key=int(key.item()), answer=int(candidates[index]), controls=controls))
    if before != {name: state_hash(runtime.get(name)) for name in runtime.list()}:
        raise RuntimeError('Evaluation changed Cell state')
    metrics = {name: {metric: sum(row['controls'][name][metric] for row in rows)
                      for metric in ('raw_correct', 'candidate_correct', 'generation_correct')}
               for name in before}
    report = dict(status='RECALL_DIAGNOSTIC_COMPLETE', gpu=torch.cuda.get_device_name(),
                  torch=str(torch.__version__), identity=identity, candidates=candidates.tolist(),
                  query_count=len(rows), exposure_passes=16, state=before, metrics=metrics,
                  rows=rows, scope='single fixed synthetic token fixture with frozen random memory adapter')
    (output / 'recall.json').write_text(json.dumps(report, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
