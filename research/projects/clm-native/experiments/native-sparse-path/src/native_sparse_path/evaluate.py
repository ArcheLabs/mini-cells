"""Paired hard inference. Wall latency includes encode, routing and state IO."""
import time
import numpy as np
import torch
from .generator import generate
from .reference_solver import solve


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def evaluate(model, config, task, *, split='dev', count=None, nodes=None, depth=None,
             intervention=None, intervention_seed=71, frozen=None,
             sealed_authorized=False, fixed_candidates=False):
    device = next(model.parameters()).device
    model.eval()
    rows, timings = [], []
    count = count or config['eval_examples']
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for index in range(count):
            instance = generate(task, split, index, nodes=nodes or config['nodes'],
                candidates=config['c'], depth=depth or config['train_depth'],
                sealed_authorized=sealed_authorized, fixed_candidates=fixed_candidates)
            target = solve(instance, k=config['k'], active_limit=config['active_limit'])
            instance = instance.to(device)
            rng = torch.Generator(device=device).manual_seed(intervention_seed + index)
            # Isolate uninstrumented inference from trace/profiling overhead.
            if index == 0:
                for _ in range(3):
                    model(instance, force_hard=True)
            synchronize(device)
            start = time.perf_counter()
            output, cost, _, _ = model(instance, intervention=intervention,
                rng=rng, frozen=frozen, force_hard=True)
            synchronize(device)
            timings.append(time.perf_counter() - start)
            _, _, trace, _ = model(instance, trace=True, intervention=intervention,
                rng=torch.Generator(device=device).manual_seed(intervention_seed + index),
                frozen=frozen, force_hard=True)
            prediction = output.cpu().numpy()
            errors = (prediction - target) ** 2
            rows.append({'id': instance.instance_id, 'template_id': instance.template_id,
                'prediction': prediction.tolist(), 'target': target.tolist(),
                'mse': float(errors.mean()),
                'correct': bool(np.abs(prediction - target).max() <= config['accuracy_tolerance']),
                'cost': cost, 'trace': trace, 'latency_seconds': timings[-1]})
    return {'split': split, 'task': task, 'arm': model.arm, 'hard_routing': True,
        'mse': float(np.mean([r['mse'] for r in rows])),
        'accuracy': float(np.mean([r['correct'] for r in rows])), 'examples': rows,
        'elapsed_wall_time': sum(timings), 'examples_per_sec': count / sum(timings),
        'latency_p50_seconds': float(np.median(timings)),
        'latency_p95_seconds': float(np.percentile(timings, 95)),
        'peak_vram': torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 'UNAVAILABLE',
        'peak_ram': 'UNAVAILABLE',
        'timing_scope': 'resident input, batch=1, full forward incl encoding and all route/update/IO; '
                        'excludes dataset generation, solver, transfer and trace export'}


def algorithm_profile(config, task='A', count=32):
    rows = []
    for index in range(count):
        instance = generate(task, 'dev', index, nodes=config['nodes'], candidates=config['c'],
                            depth=config['train_depth'])
        start = time.perf_counter()
        output = solve(instance, k=config['k'], active_limit=config['active_limit'])
        rows.append({'id': instance.instance_id, 'prediction': output.tolist(),
                     'elapsed_wall_time': time.perf_counter() - start})
    return {'arm': 'ALG', 'task': task, 'accuracy': 1.0, 'parameters_total': 0,
            'scope': 'NumPy symbolic reference executor; different implementation, CPU only',
            'latency_p50_seconds': float(np.median([r['elapsed_wall_time'] for r in rows])),
            'examples': rows}
