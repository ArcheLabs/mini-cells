"""Frozen gates, incomplete evidence detection and development-only reports."""
from __future__ import annotations

import json
import os
from pathlib import Path
from experiments.common.train import write_json

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'experiments/003-hard-cell-transition'
CONFIG = Path(os.environ.get('MINICELLS_RESEARCH_CONFIG', EXPERIMENT / 'config.yaml'))
PROTOCOL = Path(os.environ.get('MINICELLS_RESEARCH_PROTOCOL', EXPERIMENT / 'protocol.json'))


def decide(rows, config):
    if set(rows) != set(config['models']) or any(
        row.get('status') != 'completed' or row.get('train_tokens') != config['train_tokens']
        or row.get('seed') != config['seed'] or not row.get('evaluation') for row in rows.values()):
        return {'status': 'DEVELOPMENT_INCOMPLETE', 'comparable_at_full_budget': False,
                'scientific_decision': False, 'gates': None}
    hard, soft, fixed, global_ = [rows[kind]['evaluation'] for kind in
                                 ('hard-cell', 'soft-cell', 'fixed-cell', 'global-router')]
    # A non-measured/uncovered GEMM stream cannot pass efficiency gates.
    if any(not row.get('observed_gemm_calls') or row.get('uncovered_attention_calls')
           or row['executed_gemm_flops_per_token'] <= 0
           for row in (hard, soft, fixed, global_)):
        return {'status': 'DEVELOPMENT_INCOMPLETE', 'comparable_at_full_budget': False,
                'scientific_decision': False, 'gates': None}
    gates = {
        'minimum': abs(hard['perplexity'] / soft['perplexity'] - 1) <= 0.01
                   and hard['executed_gemm_flops_per_token'] < soft['executed_gemm_flops_per_token'],
        'strong': hard['perplexity'] <= global_['perplexity'] * 1.01
                  and hard['inference_latency_seconds_per_batch'] <= fixed['inference_latency_seconds_per_batch'],
        'exploratory_signal': abs(hard['perplexity'] / fixed['perplexity'] - 1) <= 0.01
                              and hard['executed_gemm_flops_per_token'] < fixed['executed_gemm_flops_per_token'] * 0.30
                              and hard['inference_latency_seconds_per_batch'] < fixed['inference_latency_seconds_per_batch']}
    return {'status': 'DEVELOPMENT_SUPPORTED' if gates['minimum'] else 'DEVELOPMENT_NOT_SUPPORTED',
            'comparable_at_full_budget': True, 'scientific_decision': False, 'gates': gates,
            'claim_scope': 'paired development seed, matched TinyStories architecture; '
                           'all arms use observed SDPA math GEMMs; no general CLM versus MoE conclusion'}


def report(output):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text())
    rows = {kind: json.loads((output / kind / 'metrics.json').read_text())
            for kind in config['models'] if (output / kind / 'metrics.json').exists()}
    decision = decide(rows, config)
    write_json(output / 'decision.json', decision)
    write_json(output / 'metrics.json', {'models': list(rows.values()), **decision})
    # Never silently emit an empty routing object.
    routing = {kind: json.loads((output / kind / 'routing_stats.json').read_text())
               if (output / kind / 'routing_stats.json').exists() else {'status': 'missing'}
               for kind in config['models']}
    write_json(output / 'routing_stats.json', routing)
    paths = {kind: json.loads((output / kind / 'path_length.json').read_text())
             if (output / kind / 'path_length.json').exists() else {'status': 'missing'}
             for kind in config['models']}
    write_json(output / 'path_length.json', paths)
    with (output / 'logs.jsonl').open('w', encoding='utf-8') as combined:
        for kind in config['models']:
            path = output / kind / 'logs.jsonl'
            if path.exists():
                for line in path.read_text().splitlines():
                    combined.write(json.dumps({'model': kind, **json.loads(line)}) + '\n')
    lines = ['# Experiment 003 — Hard Cell Transition', '',
             'Status: **' + decision['status'] + '**', '',
             '| Model | Tokens | PPL | Executed GEMM FLOPs/token | Latency ms |',
             '|---|---:|---:|---:|---:|']
    for kind in config['models']:
        row = rows.get(kind, {})
        value = row.get('evaluation', {})
        lines.append('| {} | {} | {} | {} | {} |'.format(kind, row.get('train_tokens', 'missing'),
            value.get('perplexity', 'missing'), value.get('executed_gemm_flops_per_token', 'missing'),
            value.get('inference_latency_seconds_per_batch', 0) * 1000 if value else 'missing'))
    lines.extend(['', 'Gates: `' + json.dumps(decision['gates']) + '`', '',
                  'Development evidence only. All arms use SDPA math; actual GEMMs include attention. Non-matmul operations are excluded.',
                  'Soft routing uses a conditional-mean relaxation; hard training uses a biased selected-value STE.',
                  'Missing/failed arms do not pass gates. Three paired development seeds are required for a matrix summary.',
                  'No general superiority claim about CLM, Transformer or MoE follows from this experiment.'])
    (output / 'experiment_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    make_plots(output, rows)
    return decision


def make_plots(output, rows):
    # Report generation remains possible on a host without plotting dependencies.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    destination = output / 'plots'
    destination.mkdir(exist_ok=True)
    figure, ax = plt.subplots()
    plotted = False
    for kind in rows:
        curve = output / kind / 'learning_curve.json'
        if curve.exists():
            values = json.loads(curve.read_text())
            ax.plot([row['train_tokens'] for row in values], [row['perplexity'] for row in values], label=kind)
            plotted = True
    ax.set(xlabel='Training tokens', ylabel='Validation perplexity')
    if plotted:
        ax.legend()
    figure.tight_layout()
    figure.savefig(destination / 'learning_curve.png', dpi=160)
    plt.close(figure)
    figure, ax = plt.subplots()
    plotted = False
    for kind in rows:
        path = output / kind / 'routing_stats.json'
        if path.exists():
            layers = json.loads(path.read_text())['layers']
            if layers:
                ax.plot([row['layer'] for row in layers],
                        [row.get('expected_path_length', row.get('path_length', 1)) for row in layers],
                        marker='o', label=kind)
                plotted = True
    ax.set(xlabel='FFN layer', ylabel='Per-token path length', title='Soft expected depth / hard executed depth')
    if plotted:
        ax.legend()
    figure.tight_layout()
    figure.savefig(destination / 'routing_path_lengths.png', dpi=160)
    plt.close(figure)
