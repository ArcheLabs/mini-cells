"""Derived plots and summaries; frozen decisions are never recomputed here."""
import argparse
import json
from pathlib import Path
from .config import dump


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def generate(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    metrics = read(output / 'metrics.json', {})
    rows = metrics.get('models', [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    decision = read(output / 'decision.json', {})
    historical_evaluations = decision.get('evaluations', {})
    routes = read(output / 'routing_stats.json', {})
    logs = []
    if (output / 'logs.jsonl').exists():
        for line in (output / 'logs.jsonl').read_text().splitlines():
            if line.strip():
                logs.append(json.loads(line))
    plots = output / 'plots'
    plots.mkdir(exist_ok=True)
    unavailable = []

    def finish(fig, ax, name, plotted, ylabel, xlabel='Training tokens'):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'Measurement unavailable; not zero', ha='center', transform=ax.transAxes)
            unavailable.append(name)
        fig.tight_layout()
        fig.savefig(plots / (name + '.png'), dpi=160)
        plt.close(fig)

    for name, field, ylabel in [('loss', 'loss', 'Training NLL'),
                               ('throughput', 'tokens_sec', 'Training tokens/s (compute only)'),
                               ('memory', 'gpu_memory', 'Allocated GPU bytes'),
                               ('routing_entropy', 'route_entropy', 'Routing entropy (nats)')]:
        fig, ax = plt.subplots()
        plotted = False
        for model in sorted({row.get('model', 'unknown') for row in logs}):
            selected = [row for row in logs if row.get('model', 'unknown') == model]
            # Historical entropy is an explicit compatibility alias.
            values = [(row.get('tokens'), row.get(field, row.get('entropy') if field == 'route_entropy' else None))
                      for row in selected]
            values = [(x, y) for x, y in values if x is not None and y is not None]
            if values:
                ax.plot(*zip(*values), label=model)
                plotted = True
        finish(fig, ax, name, plotted, ylabel)
    fig, ax = plt.subplots()
    plotted = False
    for row in rows:
        model = row.get('model', 'unknown')
        curve = read(output / model / 'learning_curve.json', [])
        if curve:
            ax.plot([x['train_tokens'] for x in curve], [x['perplexity'] for x in curve], label=model)
            plotted = True
    finish(fig, ax, 'ppl', plotted, 'Validation perplexity (lower is better)')
    for name, field, ylabel in [('cell_usage', 'utilization', 'Cell assignment share'),
                               ('path_length', 'expected_path_length', 'Soft expected / hard executed depth')]:
        fig, ax = plt.subplots()
        plotted = False
        for model, item in routes.items():
            for layer in item.get('layers', []) if isinstance(item, dict) else []:
                value = layer.get(field, layer.get('path_length') if field == 'expected_path_length' else None)
                if value is None:
                    continue
                if isinstance(value, list):
                    ax.plot(range(len(value)), value, marker='.', label=f'{model}/L{layer["layer"]}')
                else:
                    ax.scatter(layer['layer'], value, label=f'{model}/L{layer["layer"]}')
                plotted = True
        finish(fig, ax, name, plotted, ylabel, 'Cell index' if name == 'cell_usage' else 'Layer (zero-based)')
    fig, ax = plt.subplots()
    plotted = False
    for row in rows:
        evaluation = row.get('evaluation', historical_evaluations.get(row['model'], {}))
        cost = evaluation.get('executed_gemm_flops_per_token', evaluation.get('forward_flops_per_token_estimate'))
        if cost is not None and 'perplexity' in evaluation:
            scope = 'measured GEMM' if 'executed_gemm_flops_per_token' in evaluation else 'estimate'
            ax.scatter(cost / 1e6, evaluation['perplexity'], label=row['model'] + ' (' + scope + ')')
            plotted = True
    finish(fig, ax, 'efficiency', plotted, 'Validation perplexity (lower is better)', 'Matmul MFLOPs/token (measured or estimated; see labels)')
    ax_label = 'GEMM MFLOPs/token; non-matmul operations excluded'
    # A quality/cost Pareto view avoids inventing an undefined capability score.
    fig, ax = plt.subplots(figsize=(10, 4))
    paths = []
    for model, item in routes.items():
        if isinstance(item, dict):
            for layer in item.get('layers', []):
                for path in layer.get('sample_paths', [])[:2]:
                    paths.append(f'{model} L{layer["layer"]}: ' + ' -> '.join(map(str, path)))
    ax.axis('off')
    ax.text(0, 1, '\n'.join(paths[:12]) or 'Hard token paths unavailable; soft probability mass is not a trajectory.',
            va='top', family='monospace', fontsize=9)
    fig.tight_layout();fig.savefig(plots / 'paths.png', dpi=160);plt.close(fig)
    decision = read(output / 'decision.json', {})
    lines = ['# Research run summary', '', 'Status: **' + decision.get('status', metrics.get('status', 'UNAVAILABLE')) + '**',
             '', '| Model | Tokens | Final PPL | Matmul FLOPs/token (scope) | Latency ms |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        ev = row.get('evaluation', historical_evaluations.get(row['model'], {}))
        cost = str(ev['executed_gemm_flops_per_token']) + ' (measured)' if 'executed_gemm_flops_per_token' in ev else str(ev['forward_flops_per_token_estimate']) + ' (estimate)' if 'forward_flops_per_token_estimate' in ev else 'missing'
        lines.append('| {} | {} | {} | {} | {} |'.format(row.get('model'), row.get('train_tokens', 'missing'),
                     ev.get('perplexity', 'missing'), cost,
                     ev['inference_latency_seconds_per_batch'] * 1000 if 'inference_latency_seconds_per_batch' in ev else 'missing'))
    lines += ['', 'Frozen decision: `' + json.dumps({key: decision[key] for key in ('status', 'scientific_decision', 'gates', 'comparisons') if key in decision}) + '`', '',
              'Derived view only; unavailable measurements remain unavailable. Training PPL and validation PPL are distinct.',
              'GEMM FLOPs exclude non-matmul costs; latency scope follows the source evidence.',
              'Soft depth is expected probability mass; hard depth is actual executed visits.',
              'Single development seeds do not establish general CLM/MoE superiority.',
              'Unavailable plots: ' + (', '.join(unavailable) or 'none'), '', '## Input and execution provenance',
              '```json', json.dumps(read(output / 'experiment.json', {}), indent=2), '```']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    dump(output / 'visualization.json', {'derived': True, 'unavailable': unavailable,
                                        'efficiency_scope': ax_label, 'historical_estimates': bool(historical_evaluations)})
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    generate(parser.parse_args().output)


if __name__ == '__main__':
    main()
