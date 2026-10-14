"""Separate H1/H2/H3/H4; a screening or a pictured path never establishes H1."""
import json
from pathlib import Path
import numpy as np
from .reuse import write_json, canonical_sha256


def paired_interval(a, b, seed=99, samples=2000):
    if [r['id'] for r in a] != [r['id'] for r in b]:
        raise ValueError('paired evidence IDs differ')
    difference = np.asarray([x['mse'] - y['mse'] for x, y in zip(a, b)])
    rng = np.random.default_rng(seed)
    means = difference[rng.integers(len(difference), size=(samples, len(difference)))].mean(1)
    return {'mean': float(difference.mean()), 'ci95': np.quantile(means, [0.025, 0.975]).tolist(),
            'scope': 'paired examples conditional on one training seed; not cross-seed significance'}


def analyze(root, config):
    root = Path(root)
    results = {}
    for task in config['tasks']:
        for arm in config['screening_arms']:
            path = root / task / arm / 'metrics.json'
            if not path.is_file():
                raise FileNotFoundError(f'incomplete screening matrix: {path}')
            result = json.loads(path.read_text())
            if result['identity']['config_sha256'] != canonical_sha256(config):
                raise ValueError('analysis config differs from run')
            profile_path = path.with_name('profile.json')
            if not profile_path.exists():
                raise FileNotFoundError('isolated post-training profile is required')
            profile = json.loads(profile_path.read_text())
            if (profile['identity'] != result['identity']
                    or profile['checkpoint_sha256'] != result['checkpoint_sha256']):
                raise ValueError('profile/checkpoint identity differs')
            result.update({key: profile['uninstrumented'][key] for key in (
                'latency_p50_seconds', 'latency_p95_seconds', 'elapsed_wall_time', 'examples_per_sec')})
            results[task, arm] = result
    if len({r['identity']['source_commit'] for r in results.values()}) != 1:
        raise ValueError('mixed source commits in screening')
    if len({r['identity']['code_sha256'] for r in results.values()}) != 1:
        raise ValueError('mixed implementation hashes in screening')
    for task in config['tasks']:
        manifests = {results[task, arm]['identity']['data_sha256'] for arm in config['screening_arms']}
        if len(manifests) != 1:
            raise ValueError('baseline input data differ')
    b = results['B', 'C0']
    comparator = min((results['B', arm] for arm in ('B0', 'B1', 'B2')), key=lambda r: r['mse'])
    paired = paired_interval(b['examples'], comparator['examples'])
    gain = 1 - b['mse'] / max(comparator['mse'], 1e-12)
    latency_ratio = b['latency_p50_seconds'] / comparator['latency_p50_seconds']
    shortcut_path = root / 'B' / 'pooling-shortcut.json'
    if not shortcut_path.exists():
        raise FileNotFoundError('Task B shortcut diagnostic is required')
    shortcut = json.loads(shortcut_path.read_text())
    shortcut_excluded = shortcut['accuracy'] <= config['m3_entry']['shortcut_accuracy_maximum']
    trend = (gain >= config['m3_entry']['minimum_relative_task_b_mse_gain']
        and paired['ci95'][1] < 0 and latency_ratio <= config['m3_entry']['maximum_latency_ratio']
        and shortcut_excluded)
    decision = {'status': 'INCONCLUSIVE' if shortcut_excluded else 'IMPLEMENTATION_FAIL',
        'stage': 'SCREENING', 'scientific_decision': False,
        'H0': 'CPU_GATE_REQUIRED', 'H1': 'INCONCLUSIVE', 'H2': 'NOT_RUN', 'H3': 'NOT_RUN', 'H4': 'INCONCLUSIVE',
        'm3_authorized_by_screening': bool(trend), 'task_b_comparator': comparator['arm'],
        'task_b_relative_mse_gain': gain, 'task_b_latency_ratio': latency_ratio,
        'paired_task_b_difference': paired, 'shortcut_excluded': shortcut_excluded,
        'zero_predictor_task_b_accuracy': float(np.mean([
            max(abs(v) for v in row['target']) <= config['accuracy_tolerance']
            for row in b['examples']])),
        'reason': 'One seed cannot support H1. Continue only if Task B quality/cost trend passes the fixed gate.',
        'summary': {f'{t}/{a}': {k: r[k] for k in ('mse', 'accuracy', 'parameters_total',
                        'train_seconds', 'latency_p50_seconds')} for (t, a), r in results.items()}}
    write_json(root / 'decision.json', decision)
    return decision


def make_plots(root, decision):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, task in zip(axes, ('A', 'B', 'C')):
        for key, row in decision['summary'].items():
            if key.startswith(task + '/'):
                ax.scatter(row['latency_p50_seconds'] * 1000, row['mse'])
                ax.annotate(key.split('/')[1], (row['latency_p50_seconds'] * 1000, row['mse']))
        ax.set(xlabel='Full hard-forward latency (ms, batch=1)', ylabel='Development MSE', title='Task ' + task)
        ax.grid(alpha=0.3)
    fig.suptitle('Single-seed screening; no scientific decision')
    fig.tight_layout()
    fig.savefig(root / 'quality-cost.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, task in zip(axes, ('A', 'B', 'C')):
        for arm in ('B0', 'B1', 'B2', 'C0'):
            curve = json.loads((root / task / arm / 'learning-curve.json').read_text())
            ax.plot([r['step'] for r in curve], [r['loss'] for r in curve], label=arm)
        ax.set(xlabel='Optimizer steps', ylabel='Task + fixed regularizer loss', title='Task ' + task)
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'learning-curves.png', dpi=180)
    plt.close(fig)
