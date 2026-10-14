import resource
import torch
from .generator import generate
from .evaluate import evaluate


def profile(model, config, task):
    device = next(model.parameters()).device
    model.eval()
    instance = generate(task, 'dev', 0, nodes=config['nodes'], candidates=config['c'],
                        depth=config['train_depth']).to(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(device).startswith('cuda'):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad():
        for _ in range(3):
            model(instance, force_hard=True)
        with torch.profiler.profile(activities=activities, record_shapes=True, with_flops=True) as prof:
            _, ledger, _, _ = model(instance, force_hard=True)
    events = [{'operator': e.key, 'calls': e.count, 'cpu_time_total_us': e.cpu_time_total,
               'flops': e.flops or 'UNAVAILABLE'} for e in prof.key_averages()]
    return {'ledger': ledger,
            'observed_linear_flops': sum(e.flops or 0 for e in prof.key_averages()
                if e.key in ('aten::mm', 'aten::addmm', 'aten::bmm')),
            'observed_supported_operator_flops': sum(e.flops or 0 for e in prof.key_averages()),
            'observed_scope': 'PyTorch profiler supported operators only, not total hardware FLOPs',
            'operators': events, 'uninstrumented': evaluate(model, config, task, count=config['eval_examples']),
            'process_peak_ram_upper_bound_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'peak_ram_scope': 'Linux process lifetime upper bound, not incremental per-model peak'}
