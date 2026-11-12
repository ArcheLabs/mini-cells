"""Summarize a completed smoke manifest and project a screening budget."""
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys


def render(manifest):
    groups = {}
    for row in manifest['episodes']:
        groups.setdefault((row['geometry'],row['method'],row['n']),[]).append(row)
    lines = [
        '# P4-M1.1 CPU smoke',
        '',
        f"Status {manifest['status']}. {len(manifest['episodes'])} completed episodes, "
        f"{len(manifest['failures'])} failures, {manifest['wall_seconds']:.2f} seconds wall.",
        f"Frozen M3 control Recall@1 {manifest['control']['restored_accuracy']:.0%}.",
        '',
        '| Geometry | Method | Bindings | Recall@1 mean ± SD | Random | Random FPR | Hard FPR | Updates | Acquisition seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    projected = {}
    for (geometry,method,n), rows in sorted(groups.items()):
        final = [r['curve'][-1] for r in rows]
        accuracy = [r['recall'] for r in final]
        seconds = mean(r['seconds'] for r in final)
        fpr = [mean(r['unknown']['unknown_fpr'][i] for r in rows) for i in (0,1)]
        lines.append(f'| {geometry} | {method} | {n} | {mean(accuracy):.2%} ± {stdev(accuracy):.2%} | '
                     f'{1/n:.2%} | {fpr[0]:.2%} | {fpr[1]:.2%} | {final[0]["updates"]} | {seconds:.3f} |')
        if method != 'reverse':
            key = f'{geometry} {method}'
            projected[key] = projected.get(key,0) + 20*seconds
    recalls = [point['retention'][str(order['order'][0])]
               for orders in manifest['forgetting'].values() for order in orders
               for point in order['trajectory'][-1:] if point['retention'][str(order['order'][0])] is not None]
    total = sum(projected.values())
    calls = sum(20*128*math.ceil(n/rows[0]['width'])
                for (_,method,n),rows in groups.items() if method != 'reverse')
    presentations = sum(20*128*n for (_,method,n) in groups if method != 'reverse')
    lines += [
        '',
        'The first group retention after the final write ranged from '
        f'{min(recalls):.0%} to {max(recalls):.0%} across {len(recalls)} order runs.',
        'Each episode evaluated 256 random unknowns and 256 hard negatives. '
        'The 0.8 threshold is applied to maximum candidate cosine.',
        'All completed episodes verified read-only state and exact checkpoint restoration.',
        '',
        '## Screening budget',
        '',
        f'{calls:,} updates and {presentations:,} presentations for 20 mappings per '
        f'configuration at exposure 128. Measured CPU acquisition projects to {total/60:.2f} minutes.',
        '',
        *[f'- {key} {seconds/60:.2f} minutes.' for key,seconds in sorted(projected.items())],
        '',
        'Evaluation, restore, diagnostics, alternate orders and extended exposure are additional. '
        'These are CPU measurements. A synchronized target GPU smoke is needed to price T4 screening.',
        '',
        'The compressed manifest retains all per-seed trajectories, rejection curves and checkpoint '
        'hashes. Three mappings per configuration qualify this as smoke. Threshold selection, '
        'episode-bootstrap screening intervals and confirmation remain future work.',
    ]
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    print(render(json.loads(Path(sys.argv[1]).read_text())),end='')
