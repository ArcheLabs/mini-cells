"""Paired quality, resident-batch latency, observed GEMMs and CUDA activity."""
from __future__ import annotations

import time
import numpy as np
import torch
from experiments.common.train import evaluate, make_batch, write_json
from minicells.utils.execution import ExecutedMatmuls


def assess(model, data, config, directory, deadline):
    device = next(model.parameters()).device
    starts = np.random.default_rng(config['seed'] + 1).integers(
        0, len(data['validation']) - config['sequence_length'],
        size=(config['validation_batches'], config['batch_size']))
    quality, routing = evaluate(model, data['validation'], starts, config, device, deadline)
    write_json(directory / 'routing_stats.json', {'status': 'collected' if routing else 'not_applicable',
               'layers': routing, 'validation_tokens': quality['validation_tokens']})
    write_json(directory / 'path_length.json', {'layers': [
        {'layer': row['layer'], 'sample_paths': row.get('sample_paths', []),
         'distribution': row.get('path_length_distribution'),
         'mean': row.get('expected_path_length', row.get('path_length'))}
        for row in routing], 'scope': 'per-layer token paths, not whole-model paths; '
        'soft paths are probability mass and have no sampled hard trajectories'})
    model.eval()
    x, _ = make_batch(data['validation'], starts[0], config['sequence_length'], device)
    times = []
    with torch.no_grad():
        for iteration in range(config['benchmark_warmup'] + config['benchmark_batches']):
            if time.time() >= deadline:
                raise TimeoutError('Inference measurement exceeded the arm budget')
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                model(x)
            torch.cuda.synchronize()
            if iteration >= config['benchmark_warmup']:
                times.append(time.perf_counter() - tick)
        # Arithmetic and kernel measurements cover ALL paired validation windows,
        # so a favorable single routing batch cannot determine the efficiency gate.
        counter = ExecutedMatmuls()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            with counter:
                for positions in starts:
                    if time.time() >= deadline:
                        raise TimeoutError('Execution profiling exceeded the arm budget')
                    inputs, _ = make_batch(data['validation'], positions, config['sequence_length'], device)
                    with torch.autocast('cuda', dtype=torch.float16):
                        model(inputs)
            torch.cuda.synchronize()
        events = [event for event in captured.events()
                  if event.device_type == torch.autograd.DeviceType.CUDA]
        kernels = [event for event in events
                   if not any(label in event.name.lower() for label in ('memcpy', 'memset', '[memory]'))]
    latency = sum(times) / len(times)
    result = {**quality, **counter.metrics(quality['validation_tokens']),
              'inference_latency_seconds_per_batch': latency,
              'inference_tokens_per_sec': x.numel() / latency,
              'cuda_kernel_time_us_per_validation_batch': sum(event.device_time_total for event in kernels) / len(starts),
              'cuda_activity_time_us_per_validation_batch': sum(event.device_time_total for event in events) / len(starts),
              'cuda_activity_events_per_validation_batch': len(events) / len(starts),
              'timing_scope': 'FP16 forward on identical resident batch, synchronized host latency; '
                              'profiling and transfer excluded; two independent GPU workers may share CPU',
              'profile_scope': 'summed CUDA event durations from instrumented validation; kernel time excludes '
                               'named memcpy/memset/memory activities; total activity reported separately; '
                               'not uninstrumented end-to-end latency',
              'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
    write_json(directory / 'evaluation.json', result)
    model.train()
    return result
