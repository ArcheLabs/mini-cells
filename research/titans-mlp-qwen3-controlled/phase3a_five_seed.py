"""Orchestrate owner-approved sealed validation and aggregate independent seeds."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from audit_qwen import sha
from phase3a_v21_dry_run import load_config


def aggregate(reports, seeds):
    by_seed = {r['seed']:r for r in reports}
    if (len(reports) != len(seeds) or set(by_seed) != set(seeds)
            or any(r['test'] != 'SEALED' for r in reports)
            or len({r['config_sha256'] for r in reports}) != 1):
        raise ValueError('Seed or configuration integrity mismatch')

    def stats(values, lower=0., upper=1.):
        values = [v for v in values if v is not None]
        if len(values) < 2:
            return dict(n=len(values),mean=values[0] if values else None,sd=None,ci95=None)
        mean, sd = statistics.mean(values), statistics.stdev(values)
        multipliers = [0,0,12.706,4.303,3.182,2.776]
        half = multipliers[len(values)] * sd / len(values)**.5
        return dict(n=len(values), mean=mean, sd=sd, ci95=[max(lower,mean-half),min(upper,mean+half)])

    metrics = {}
    for adapter, approaches in reports[0]['calibration'].items():
        for approach in approaches:
            for name in ('fpr','coverage','selective_recall1','selective_recall5'):
                metrics[f'{adapter}/{approach}/{name}'] = stats([
                    r['calibration'][adapter][approach]['audit'][name]['rate'] for r in by_seed.values()])
    retrieval = {}
    paired = {}
    def metric(report, method, condition, name):
        rows = report['methods'][method][condition]
        return statistics.mean(row[name] for row in rows) if rows else None

    for method, conditions in reports[0].get('methods',{}).items():
        for condition in conditions:
            key = f'{method}/{condition}'
            retrieval[key] = {name:stats([metric(r,method,condition,name) for r in reports])
                              for name in ('recall1','recall5','mrr')}
            if method != 'Raw Cosine':
                pairs = [(metric(r,method,condition,'recall1'),
                          metric(r,'Raw Cosine',condition,'recall1')) for r in reports]
                paired[key] = stats([a-b for a,b in pairs if a is not None and b is not None], -1., 1.)
    return dict(seeds=list(seeds),metrics=metrics,retrieval=retrieval,paired_recall1_vs_raw=paired,test='SEALED',
                unavailable_candidate_sizes=reports[0]['unavailable_candidate_sizes'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('snapshot',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--approved-config-sha256',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args = p.parse_args()
    cfg = load_config(args.config)
    if sha(args.config) != args.approved_config_sha256:
        p.error('Approved configuration digest does not match')
    seeds = cfg['dataset']['seeds']
    if seeds != [42,43,44,45,46]:
        p.error('Invalid seed contract')
    args.output.mkdir(parents=True,exist_ok=False)
    reports = []
    for seed in seeds:
        out = args.output/f'seed-{seed}'
        subprocess.run([sys.executable,str(Path(__file__).with_name('phase3a_v21_dry_run.py')),
                        str(args.config),str(args.snapshot),str(out),'--seed',str(seed),
                        '--device',args.device,'--full-validation'], check=True)
        reports.append(json.loads((out/'report.json').read_text()))
    (args.output/'aggregate.json').write_text(json.dumps(aggregate(reports,seeds),indent=2))


if __name__ == '__main__':
    main()
