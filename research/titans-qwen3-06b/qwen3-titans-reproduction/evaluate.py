"""Paired frozen validation for native Qwen, segmented attention and Titans."""
import json
import math

import torch

from run import objective, write, sha


def evaluate(model, validation, directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in validation.read_text().splitlines() if line.strip()]
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Validation requires unique row identifiers')
    result = {}
    for mode, name in [('native', 'baseline'), ('segmented', 'segmented-only'), ('titans', 'titans')]:
        results = []
        with torch.no_grad():
            for row in rows:
                batch = {key: torch.tensor([row[key]], dtype=torch.long)
                         for key in ('input_ids', 'attention_mask', 'labels')}
                nll, count, correct = objective(model, batch, mode)
                results.append(dict(id=row['id'], nll_sum=float(nll), tokens=count, correct=correct))
        tokens = sum(r['tokens'] for r in results)
        nll = sum(r['nll_sum'] for r in results) / tokens
        result[name] = dict(validation_sha256=sha(validation), documents=len(rows), tokens=tokens,
                            nll=nll, perplexity=math.exp(nll),
                            next_token_accuracy=sum(r['correct'] for r in results)/tokens)
        write(directory/f'{name}.json', dict(result[name], rows=results))
    write(directory/'comparison.json', dict(status='VALIDATION_ONLY', arms=result,
          gain_vs_segmented=result['segmented-only']['nll'] - result['titans']['nll'],
          gain_vs_native=result['baseline']['nll'] - result['titans']['nll'],
          quality='NOT_EVALUATED', longhealth='NOT_EVALUATED', ruler='NOT_EVALUATED'))
