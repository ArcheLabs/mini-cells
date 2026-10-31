"""Exactly one C1 training pilot, split at update 25 to exercise real restore."""
from dataclasses import asdict
import math
import platform
import torch
from model import Core, config_from_protocol
from trainer import train, load_checkpoint


def pilot_plan(protocol):
    plan = protocol['minimal_gpu_pilot']
    model = config_from_protocol(protocol, plan['arm'])
    if plan['K'] != model.steps:
        raise ValueError('pilot K disagrees with the architecture protocol')
    if (model.arm, model.steps, plan['batch'], plan['bptt'], plan['updates']) != ('C1', 4, 4, 64, 50):
        raise ValueError('minimal GPU pilot must be C1 / K4 / batch4 / TBPTT64 / 50 updates')
    if (plan['batch'], plan['bptt']) != (protocol['batch'], protocol['bptt']):
        raise ValueError('pilot and trainer protocol settings disagree')
    if plan['diagnostic_updates'] not in (1,2):
        raise ValueError('initial GPU diagnostic is limited to 1 or 2 successful updates')
    if not 0 < plan['resume_after_updates'] < plan['updates']:
        raise ValueError('pilot resume boundary must be within the 50 updates')
    return model, plan, plan['batch'] * plan['bptt'] * plan['updates']


def run_pilot(protocol, documents, identity, output, deadline, on_checkpoint=None):
    config, plan, target_tokens = pilot_plan(protocol)
    identity = {**identity, 'mode': 'minimal_gpu_pilot', 'model': asdict(config),
                'batch': plan['batch'], 'bptt': plan['bptt'], 'target_tokens': target_tokens,
                'precision': protocol['precision']}
    checkpoint = output / 'C1-pilot.pt'
    report = {
        'format': 'minicells.epclm-core-001.v1', 'mode': 'MINIMAL_GPU_PILOT',
        'status': 'INCOMPLETE', 'identity': identity, 'configuration': {**plan, 'checkpoint_interval_updates': plan['resume_after_updates'], 'precision':protocol['precision'], 'model': asdict(config)},
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                        'precision': 'FP32 model/state/loss/optimizer; autocast and GradScaler disabled' if protocol['precision']['mode']=='fp32' else 'FP16 autocast; FP32 replay on non-finite gradients'},
        'hypotheses': dict.fromkeys(('H1', 'H2', 'H3', 'H4', 'H5'), 'NOT_EVALUATED'),
        'scientific_decision': False, 'formal_confirmation': False,
        'checkpoint_resume': 'NOT_COMPLETED', 'numerical_stability': 'NOT_EVALUATED',
        'initial_diagnostic':{'status':'NOT_COMPLETED','successful_updates':0,'counted_toward_50':True},
    }
    if protocol['precision']['mode']=='fp32':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    report['environment']['tf32_matmul_enabled']=torch.backends.cuda.matmul.allow_tf32
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(protocol['seed'])
    model = Core(config).float().cuda()
    kwargs = dict(documents=documents, identity=identity, path=checkpoint,
                  target_tokens=target_tokens, deadline=deadline, seed=protocol['seed'],
                  batch=plan['batch'], unroll=plan['bptt'], on_checkpoint=on_checkpoint,
                  checkpoint_interval=plan['resume_after_updates'],
                  precision=protocol['precision']['mode'],
                  initial_scale=protocol['precision']['grad_scaler_initial_scale'],
                  max_consecutive_overflows=protocol['precision']['max_consecutive_overflows'])
    segments = []
    try:
        first = train(model, stop_after_updates=plan['diagnostic_updates'], **kwargs)
        segments.append(first)
        report['initial_diagnostic']={'status':'PASS' if first['updates']>=plan['diagnostic_updates'] else 'INCOMPLETE',
                                      'successful_updates':min(first['updates'],plan['diagnostic_updates']),
                                      'counted_toward_50':True}
        if first['updates']>=plan['diagnostic_updates'] and first['updates']<plan['resume_after_updates']:
            segments.append(train(model,stop_after_updates=plan['resume_after_updates'],**kwargs))
        saved = load_checkpoint(checkpoint)
        if saved['updates'] >= plan['resume_after_updates']:
            # A fresh model and fresh optimizer/scaler/scheduler instances must load the checkpoint.
            del model
            model = Core(config).float().cuda()
            resumed = train(model, **kwargs)
            segments.append(resumed)
            if resumed['resumed_from_updates'] != saved['updates'] or resumed['resumed_from_tokens'] != saved['tokens']:
                raise ValueError('trainer did not resume at the saved update/token boundary')
            report['checkpoint_resume'] = {
                'status': 'PASS', 'from_update': saved['updates'], 'from_tokens': saved['tokens'],
                'optimizer_scheduler_scaler_rng_data_state': 'RESTORED_BY_SHARED_TRAINER',
                'exact_equivalence': 'CPU_INTEGRATION_TEST; no duplicated GPU run',
            }
        final = segments[-1]
        curve = final['curve']
        stable = bool(curve) and all(math.isfinite(row[key]) for row in curve
                                     for key in ('nll', 'gradient_norm', 'state_norm'))
        measured_tokens = sum(segment['new_tokens'] for segment in segments)
        measured_seconds = sum(segment['elapsed_seconds'] for segment in segments)
        speed = measured_tokens / measured_seconds if measured_tokens else None
        report.update({
            'status': 'COMPLETE' if final['updates'] == plan['updates'] and final['tokens'] == target_tokens
                      and stable and isinstance(report['checkpoint_resume'], dict) else 'INCOMPLETE',
            'updates': final['updates'], 'training_tokens': final['tokens'], 'target_tokens': target_tokens,
            'numerical_stability': 'PASS' if stable else 'FAILED',
            'loss': {'first': curve[0]['nll'], 'last': curve[-1]['nll'], 'minimum': min(row['nll'] for row in curve)} if curve else None,
            'max_gradient_norm': max((row['gradient_norm'] for row in curve), default=None),
            'max_state_norm': max((row['state_norm'] for row in curve), default=None),
            'effective_precision':final['effective_precision'],
            'overflow_diagnostics':final['overflow_diagnostics'],
            'precision_classification':final['overflow_diagnostics'][-1]['classification'] if final['overflow_diagnostics'] else 'NO_NONFINITE_GRADIENTS_OBSERVED',
            'curve': curve, 'tokens_per_second': speed, 'measured_tokens': measured_tokens,
            'measured_training_seconds': measured_seconds,
            'peak_vram_bytes': torch.cuda.max_memory_allocated(),
            'checkpoint_save_seconds': sum(segment['checkpoint_save_seconds'] for segment in segments),
            'checkpoint_restore_seconds': sum(segment['checkpoint_restore_seconds'] for segment in segments),
            'estimated_250k_training_seconds': 250000 / speed if speed else None,
            'estimate_scope': 'single C1 arm, measured shared-trainer throughput including periodic checkpoint writes; excludes data preparation and full validation',
        })
    except (FloatingPointError, RuntimeError, ValueError) as error:
        report['overflow_diagnostics']=getattr(error,'diagnostics',[])
        if report['overflow_diagnostics']:
            report['last_failed_update']=report['overflow_diagnostics'][-1]
            report['precision_classification']=report['last_failed_update'].get('classification','UNRESOLVED')
        report.update({'error': str(error), 'numerical_stability': 'FAILED_OR_INTERRUPTED',
                       'peak_vram_bytes': torch.cuda.max_memory_allocated()})
    return report
