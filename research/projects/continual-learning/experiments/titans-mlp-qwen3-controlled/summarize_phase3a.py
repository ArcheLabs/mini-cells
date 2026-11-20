"""Aggregate complete Phase 3A runs by seed and evaluate frozen gates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    groups = defaultdict(list)
    relations = defaultdict(list)
    for row in data['rows']:
        for name, episodes in row['methods'].items():
            key = row['pooling'], row['case'], name
            groups[key].append({metric: statistics.mean(ep[metric] for ep in episodes)
                                for metric in ('recall1', 'recall5', 'mrr', 'positive',
                                               'hardest_negative', 'margin')})
            ranks = [r for ep in episodes for r in ep['ranks']]
            for relation in range(8):
                relations[(*key, relation)].append(statistics.mean(r == 1 for r in ranks[relation::8]))

    def estimate(values):
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else None
        # Five paired seeds are the independent resampling units.
        ci = None if len(values) != 5 else [mean-2.776*sd/5**.5, mean+2.776*sd/5**.5]
        return dict(mean=mean, sd=sd, ci95=ci)

    summary = { '/'.join(key): {m: estimate([row[m] for row in rows]) for m in rows[0]}
                for key, rows in groups.items()}
    complete = data['counts'] == [1024, 256, 512] and all(len(rows) == 5 for rows in groups.values())
    decision = 'NOT TESTED'
    if complete:
        decision = 'FAIL'
        for method in ('raw', 'probe'):
            seen = summary[f'last/seen/{method}']['recall1']
            held = summary[f'last/held/{method}']['recall1']
            mrr = summary[f'last/held/{method}']['mrr']
            if (seen['mean'] >= .5 and held['mean'] >= .25 and
                    min(seen['ci95'][0], held['ci95'][0]) > 1/32 and
                    mrr['ci95'][0] > sum(1/r for r in range(1, 33))/32):
                decision = 'PASS' if method == 'raw' else 'SEMANTIC_ADDRESSING_REQUIRES_ADAPTER'
                break
    print(json.dumps(dict(decision=decision, summary=summary,
                          relation_recall={'/'.join(map(str,k)):estimate(v) for k,v in relations.items()}),
                     indent=2))


if __name__ == '__main__':
    main()
