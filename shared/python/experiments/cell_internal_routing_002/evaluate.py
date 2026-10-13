"""Same-checkpoint soft/Top-1 quality, latency, execution and profiling evidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.common.train import evaluate, make_batch, prepare_data, write_json
from experiments.cell_internal_routing_002.models.cell_router import LanguageModel


def execution_metrics(model, routing):
    metrics = model.parameter_metrics()
    if model.kind != 'cell-internal':
        return metrics
    c = model.config
    width, experts = c['width'], c['experts']
    visits = sum(row['executed_cell_visits_per_token'] for row in routing)
    unique = sum(row['unique_cell_weights_per_token'] for row in routing)
    router_visits = sum(row['executed_router_visits_per_token'] for row in routing)
    unique_routers = sum(row['unique_router_weights_per_token'] for row in routing)
    base = c['layers'] * (8 * width * width + 4 * c['sequence_length'] * width) + 2 * width * c['vocab_size']
    cell_params = 2 * width * c['expert_hidden']
    router_params = width * (experts + 1)
    metrics.update(forward_flops_per_token_estimate=base + visits * 2 * cell_params + router_visits * 2 * router_params,
                   activated_params_per_token_estimate=(metrics['params'] - c['layers'] * experts * (cell_params + router_params)
                                                        + unique * cell_params + unique_routers * router_params),
                   actual_cell_visits_per_token_across_layers=visits,
                   actual_router_visits_per_token_across_layers=router_visits,
                   activated_parameters_scope='mean unique executed Cell and router weights per token plus always-active backbone; repeated visits count in FLOPs')
    return metrics


def benchmark(model, tokens, starts, config, device, deadline):
    """Inference only, same resident batch; report profile separately from timed work."""
    x, _ = make_batch(tokens, starts[0], config['sequence_length'], device)
    model.eval()
    with torch.no_grad():
        for _ in range(config['benchmark_warmup']):
            with torch.autocast('cuda', dtype=torch.float16):
                model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for _ in range(config['benchmark_batches']):
            if time.time() >= deadline:
                raise TimeoutError('Inference benchmark exceeded session budget')
            tick = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                model(x)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - tick)
        memory = torch.cuda.max_memory_allocated()
        profile = {'status': 'unavailable'}
        try:
            from torch.profiler import profile as capture, ProfilerActivity
            with capture(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
                for _ in range(config['profile_batches']):
                    with torch.autocast('cuda', dtype=torch.float16):
                        model(x)
                torch.cuda.synchronize()
            events = [event for event in captured.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA]
            profile = {'status': 'measured',
                       'cuda_activity_events_per_batch': len(events) / config['profile_batches'],
                       'scope': 'CUDA activity events, includes runtime kernels; excluded from timing'}
        except Exception as exc:
            profile['reason'] = type(exc).__name__
    model.train()
    latency = sum(elapsed) / len(elapsed)
    return {'inference_tokens_per_sec': x.numel() / latency,
            'inference_latency_seconds_per_batch': latency,
            'inference_peak_gpu_bytes': memory,
            'benchmark_batches': len(elapsed), 'benchmark_tokens_per_batch': x.numel(),
            'benchmark_scope': 'FP16 no-grad forward on a resident batch; synchronization included, transfer/saving excluded',
            'profile': profile}


def evaluate_checkpoint(checkpoint_path, data, directory, config, deadline, source_metrics):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    kind = saved['model_kind']
    model = LanguageModel(config, kind).to('cuda')
    model.load_state_dict(saved['model'], strict=True)
    del saved
    starts = np.random.default_rng(config['seed'] + 1).integers(
        0, len(data['validation']) - config['sequence_length'],
        size=(config['validation_batches'], config['batch_size']))
    evidence = {}
    modes = ['soft', 'hard'] if kind == 'cell-internal' else ['native']
    for mode in modes:
        model.set_routing_mode(mode)
        quality, routing = evaluate(model, data['validation'], starts, config,
                                    torch.device('cuda'), deadline)
        if kind != 'fixed-cell' and len(routing) != config['layers']:
            raise RuntimeError('Every routed layer must have validation evidence')
        if any(row.get('validation_tokens', quality['validation_tokens']) != quality['validation_tokens']
               for row in routing):
            raise RuntimeError('Routing token counts must match validation')
        write_json(directory / ('routing-' + mode + '.json'), {
            'status': 'not_applicable' if kind == 'fixed-cell' else 'collected',
            'model': kind, 'mode': mode, 'layers': routing,
            'checkpoint_step': source_metrics['steps']})
        rows = {'model': kind, 'mode': mode, 'train_tokens': source_metrics['train_tokens'],
                'training_tokens_per_sec': source_metrics['tokens_per_sec'],
                'checkpoint_step': source_metrics['steps'], **quality,
                **execution_metrics(model, routing),
                **benchmark(model, data['validation'], starts, config, torch.device('cuda'), deadline)}
        evidence[mode] = rows
        write_json(directory / 'evaluation.json', evidence)
    del model
    torch.cuda.empty_cache()
    return evidence


def main():
    """Repair Experiment 001 from a checkpoint; reject changed data identities."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--original-output', required=True, help='Contains original data.json')
    parser.add_argument('--output', required=True, help='New repair directory; original evidence is preserved')
    parser.add_argument('--budget-seconds', type=int, default=600)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['model_kind'] not in {'fixed', 'global-router'}:
        raise ValueError('Repair command is only for Experiment 001')
    config = checkpoint['config']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + min(args.budget_seconds, 3300)
    original = json.loads((Path(args.original_output) / 'data.json').read_text())
    data = prepare_data(config, output, deadline)
    current = json.loads((output / 'data.json').read_text())
    for key in ('dataset', 'revision', 'text_sha256', 'token_sha256', 'tokenizer_sha256'):
        if original[key] != current[key]:
            raise RuntimeError('Repair data identity mismatch: ' + key)
    from experiments.common.models import LanguageModel as OriginalModel
    model = OriginalModel(config, checkpoint['model_kind']).to('cuda')
    model.load_state_dict(checkpoint['model'], strict=True)
    starts = np.random.default_rng(config['seed'] + 1).integers(
        0, len(data['validation']) - config['sequence_length'],
        size=(config['validation_batches'], config['batch_size']))
    quality, routing = evaluate(model, data['validation'], starts, config, torch.device('cuda'), deadline)
    write_json(output / 'routing_stats.json', {'status': 'collected' if routing else 'not_applicable',
               'layers': routing, 'checkpoint_step': checkpoint['step'], **quality,
               'scope': 'post-hoc evaluation repair; original files and scientific status unchanged'})


if __name__ == '__main__':
    main()
