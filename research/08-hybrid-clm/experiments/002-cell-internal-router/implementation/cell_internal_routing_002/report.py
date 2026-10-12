"""Preregistered development gates; missing evidence can never pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from experiments.common.train import write_json

ROOT = next(_root for _root in Path(__file__).resolve().parents if (_root / ".git").exists() or (_root / "pyproject.toml").exists())
PROTOCOL = ROOT / 'research/validations/routing-002-cell-internal-routing/protocol.json'
CONFIG = Path(__file__).parent / 'configs/experiment.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def seed_report(directory):
    directory = Path(directory)
    protocol = json.loads(PROTOCOL.read_text())
    result = {'status': 'DEVELOPMENT_INCOMPLETE', 'scientific_decision': False,
              'protocol_sha256': digest(PROTOCOL), 'source_config_sha256': digest(CONFIG),
              'comparisons': {}, 'limitations': protocol['confounds']}
    try:
        config = json.loads((directory / 'config.json').read_text())
        meta = json.loads((directory / 'experiment.json').read_text())
        if (len(meta['source_commit']) != 40
                or any(character not in '0123456789abcdef' for character in meta['source_commit'])):
            raise ValueError('A full immutable source commit is required')
        result.update(seed=config['seed'], source_commit=meta['source_commit'])
        if meta['protocol_sha256'] != digest(PROTOCOL):
            raise ValueError('Protocol identity mismatch')
        if meta['source_config_sha256'] != digest(CONFIG):
            raise ValueError('Configuration identity mismatch')
        expected = json.loads(CONFIG.read_text())
        expected['seed'] = config['seed']
        if config != expected or config['seed'] not in protocol['seeds']:
            raise ValueError('Runtime configuration does not match the registered configuration')
        data = json.loads((directory / 'data.json').read_text())
        if data['dataset'] != config['dataset'] or data['revision'] != config['dataset_revision']:
            raise ValueError('Dataset revision mismatch')
        result['data_identity'] = {key: data[key] for key in ('text_sha256', 'token_sha256', 'tokenizer_sha256')}
        evaluations = {}
        for kind in protocol['models']:
            metrics = json.loads((directory / kind / 'metrics.json').read_text())
            if metrics['status'] != 'completed' or metrics['train_tokens'] != protocol['train_tokens_per_model']:
                raise ValueError('Full token budget required for ' + kind)
            if metrics['steps'] != 5000 or metrics['model'] != kind:
                raise ValueError('Wrong checkpoint identity for ' + kind)
            mode = 'hard' if kind == 'cell-internal' else 'native'
            evaluation = json.loads((directory / kind / 'evaluation.json').read_text())[mode]
            if evaluation['checkpoint_step'] != metrics['steps'] or evaluation['train_tokens'] != metrics['train_tokens']:
                raise ValueError('Evaluation does not match the final checkpoint')
            if evaluation['validation_tokens'] != 32000:
                raise ValueError('Unexpected validation token budget')
            routing = json.loads((directory / kind / ('routing-' + mode + '.json')).read_text())
            if kind != 'fixed-cell':
                if routing['status'] != 'collected' or len(routing['layers']) != config['layers']:
                    raise ValueError('Missing routed layers for ' + kind)
                if any(not row['assignments'] or sum(row['assignments']) <= 0 for row in routing['layers']):
                    raise ValueError('Empty routing assignments')
            for key in ('perplexity', 'validation_loss', 'forward_flops_per_token_estimate',
                        'activated_params_per_token_estimate', 'inference_latency_seconds_per_batch',
                        'inference_tokens_per_sec', 'training_tokens_per_sec', 'params'):
                if not numeric(evaluation[key]):
                    raise ValueError('Invalid metric: ' + key)
            evaluations[kind] = evaluation
        capacities = [row['params'] for row in evaluations.values()]
        if max(capacities) / min(capacities) - 1 > protocol['capacity_tolerance_fraction']:
            raise ValueError('Capacity matching failed')
        internal, global_ = evaluations['cell-internal'], evaluations['global-router']
        soft = json.loads((directory / 'cell-internal/evaluation.json').read_text())['soft']
        if soft['checkpoint_step'] != internal['checkpoint_step']:
            raise ValueError('Soft and Top-1 checkpoints differ')
        quality = internal['perplexity'] / global_['perplexity']
        compute = internal['forward_flops_per_token_estimate'] / global_['forward_flops_per_token_estimate']
        latency = internal['inference_latency_seconds_per_batch'] / global_['inference_latency_seconds_per_batch']
        routing = json.loads((directory / 'cell-internal/routing-hard.json').read_text())['layers']
        paths = {tuple(row) for layer in routing for row in layer['sample_paths']}
        usage = [sum(layer['assignments'][i] for layer in routing) for i in range(config['experts'])]
        hard_routing = {
            'mean_layer_entropy_nats': statistics.mean(row['routing_entropy_nats'] for row in routing),
            'mean_layer_stop_fraction': statistics.mean(row['learned_stop_fraction'] for row in routing),
            'mean_layer_forced_stop_fraction': statistics.mean(row['forced_stop_fraction'] for row in routing),
            'cell_visit_counts': usage,
            'mean_layer_path_length_distribution': [statistics.mean(
                row['path_length_distribution'][hop] for row in routing)
                for hop in range(config['max_hops'])],
        }
        diversity = len(paths) >= 2 and max(usage) / sum(usage) < 0.95
        depth = statistics.mean(layer['expected_path_length'] for layer in routing)
        minimum = diversity and quality <= 1.01 and compute <= 1.05 and latency <= 1.05
        medium = minimum and quality <= 1 and depth < 4
        strong = diversity and quality <= 1 and compute < 1 and latency < 1
        result.update(evaluations=evaluations, soft_evaluation=soft, comparable_at_full_budget=True,
                      hard_routing=hard_routing,
                      comparisons={'hard_to_global_ppl_ratio': quality, 'hard_to_global_flops_ratio': compute,
                                   'hard_to_global_latency_ratio': latency, 'hard_mean_sequential_depth': depth,
                                   'sample_path_count': len(paths), 'noncollapse': diversity,
                                   'minimum': minimum, 'medium': medium, 'strong': strong},
                      status='DEVELOPMENT_STRONG_SUPPORTED' if strong else
                      'DEVELOPMENT_MEDIUM_SUPPORTED' if medium else
                      'DEVELOPMENT_MINIMUM_SUPPORTED' if minimum else 'DEVELOPMENT_NOT_SUPPORTED')
    except (OSError, KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        result.update(reason=str(exc), comparable_at_full_budget=False)
    return result


def matrix_report(directories):
    reports = [seed_report(path) for path in directories]
    output = {'status': 'DEVELOPMENT_INCOMPLETE', 'scientific_decision': False,
              'protocol_sha256': digest(PROTOCOL), 'seeds': reports,
              'scope': 'paired development seeds; sample SD, no significance claim'}
    expected = json.loads(PROTOCOL.read_text())['seeds']
    if len(reports) != len(expected) or sorted(row.get('seed', -1) for row in reports) != expected:
        output['reason'] = 'Exactly the registered three unique seeds are required'
        return output
    identities = {(row.get('source_commit'), json.dumps(row.get('data_identity'), sort_keys=True)) for row in reports}
    if len(identities) != 1 or any(not row.get('comparable_at_full_budget') for row in reports):
        output['reason'] = 'All seeds need complete evidence and identical source/protocol/data identities'
        return output
    output['summary'] = {}
    for kind in ('fixed-cell', 'global-router', 'cell-internal'):
        output['summary'][kind] = {}
        metrics = ('perplexity', 'validation_loss', 'training_tokens_per_sec',
                   'inference_tokens_per_sec', 'inference_latency_seconds_per_batch',
                   'forward_flops_per_token_estimate', 'activated_params_per_token_estimate')
        for metric in metrics:
            values = [row['evaluations'][kind][metric] for row in reports]
            output['summary'][kind][metric] = {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
    depths = [row['comparisons']['hard_mean_sequential_depth'] for row in reports]
    output['summary']['cell-internal']['hard_mean_sequential_depth'] = {
        'mean': statistics.mean(depths), 'sample_sd': statistics.stdev(depths)}
    for metric in ('mean_layer_entropy_nats', 'mean_layer_stop_fraction',
                   'mean_layer_forced_stop_fraction'):
        values = [row['hard_routing'][metric] for row in reports]
        output['summary']['cell-internal'][metric] = {
            'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
    output['summary']['cell-internal']['cell_visit_counts'] = [
        {'mean': statistics.mean(row['hard_routing']['cell_visit_counts'][cell] for row in reports),
         'sample_sd': statistics.stdev(row['hard_routing']['cell_visit_counts'][cell] for row in reports)}
        for cell in range(len(reports[0]['hard_routing']['cell_visit_counts']))]
    level = 'STRONG' if all(row['comparisons']['strong'] for row in reports) else 'MEDIUM' if all(
        row['comparisons']['medium'] for row in reports) else 'MINIMUM' if all(
        row['comparisons']['minimum'] for row in reports) else None
    output['status'] = 'DEVELOPMENT_' + level + '_SUPPORTED' if level else 'DEVELOPMENT_NOT_SUPPORTED'
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = seed_report(args.inputs[0]) if len(args.inputs) == 1 else matrix_report(args.inputs)
    write_json(Path(args.output), report)
    print(json.dumps({'status': report['status'], 'scientific_decision': False}))


if __name__ == '__main__':
    main()
