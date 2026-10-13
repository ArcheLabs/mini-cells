import argparse
import json
import subprocess
from pathlib import Path
import torch
from .config import load, authorize_sealed, authorize_screening
from .reuse import BASE, ROOT, canonical_sha256, file_sha256, write_json, code_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('preflight', 'audit', 'generate', 'train', 'screen',
        'shortcut', 'analyze', 'evaluate', 'intervene', 'profile', 'publish'), required=True)
    p.add_argument('--config', default=str(BASE / 'configs/default.json'))
    p.add_argument('--arm', choices=('B0', 'B1', 'B2', 'C0', 'B3'), default='C0')
    p.add_argument('--task', choices=('A', 'B', 'C'), default='B')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', choices=('train', 'dev', 'sealed_test', 'ood'), default='dev')
    p.add_argument('--output', type=Path, default=Path('/tmp/minicells-clm-native'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--source-commit')
    p.add_argument('--device', default='cpu')
    p.add_argument('--devices', nargs='+', default=['cpu'])
    p.add_argument('--screening-root', type=Path)
    p.add_argument('--lock', type=Path)
    p.add_argument('--causality-report', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--ood-condition', choices=('larger', 'longer', 'fixed_candidates'), default='larger')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    c = load(a.config)
    torch.set_num_threads(c['cpu_threads'])
    source = a.source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if a.mode in ('train', 'screen', 'shortcut'):
        authorize_screening(c, source)
    if a.mode == 'preflight':
        from .preflight import preflight
        result = preflight(c, a.output / 'preflight.json')
    elif a.mode == 'audit':
        from .preflight import prior_evidence, hardware
        hardware(a.output / 'hardware.json')
        result = prior_evidence(a.output / 'prior-evidence.json')
    elif a.mode == 'generate':
        if a.split == 'sealed_test':
            raise PermissionError('sealed data are generated only for authorized final checkpoint evaluation')
        from .train import dataset
        _, _, result = dataset(c, a.task, a.split, count=c['eval_examples'])
        write_json(a.output / 'data-manifest.json', result)
    elif a.mode == 'train':
        if a.arm == 'B3' or a.seed != c['screening_seed']:
            if not a.screening_root:
                raise PermissionError('B3/repeats require a verified positive Task B screening gate')
            from .report import analyze
            if not analyze(a.screening_root, c)['m3_authorized_by_screening']:
                raise PermissionError('screening gate did not authorize M3')
            if a.seed not in c['replicate_seeds']:
                raise PermissionError('unregistered replicate seed')
        from .train import train
        result = train(c, a.arm, a.task, a.seed, a.output, source, device=a.device, resume=a.resume)
        result = {k: result[k] for k in ('status', 'arm', 'task', 'mse', 'accuracy')}
    elif a.mode == 'screen':
        from .runner import screen
        from .train import train_shortcut
        from .report import analyze
        from .evaluate import algorithm_profile
        screen(Path(a.config).resolve(), c, a.output, source, tuple(a.devices))
        for task in c['tasks']:
            if not (a.output / task / 'pooling-shortcut.json').exists():
                train_shortcut(c, task, c['screening_seed'], a.output / task, source_commit=source)
        write_json(a.output / 'algorithm.json', algorithm_profile(c))
        result = analyze(a.output, c)
        from .report import make_plots
        make_plots(a.output, result)
    elif a.mode == 'shortcut':
        from .train import train_shortcut
        result = train_shortcut(c, a.task, a.seed, a.output, source_commit=source)
    elif a.mode == 'analyze':
        from .report import analyze
        result = analyze(a.output, c)
        from .report import make_plots
        make_plots(a.output, result)
    elif a.mode == 'publish':
        from .publish import publish
        result = publish(a.output, source, preflight_only=a.preflight_only)
    else:
        if not a.checkpoint:
            p.error('evaluate/intervene/profile require --checkpoint')
        from .models import NativeModel
        payload = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        identity = payload['identity']
        if (identity['config_sha256'] != canonical_sha256(c) or identity['source_commit'] != source
                or identity['arm'] != a.arm or identity['task'] != a.task
                or identity['code_sha256'] != code_digest()):
            raise ValueError('checkpoint configuration/source/arm/task mismatch')
        model = NativeModel(c, a.arm).to(a.device)
        model.load_state_dict(payload['model'])
        if a.mode == 'evaluate':
            approved = False
            if a.split == 'ood':
                from .intervention import has_causal_signal
                if not a.causality_report:
                    raise PermissionError('M4 OOD needs a valid M3 causal signal')
                causal = json.loads(a.causality_report.read_text())
                if (causal['identity']['source_commit'] != source or not has_causal_signal(causal)):
                    raise PermissionError('M3 evidence did not authorize M4')
            if a.split == 'sealed_test':
                authorize_sealed(c, json.loads(a.lock.read_text()) if a.lock else None,
                                 file_sha256(a.checkpoint), identity['source_commit'])
                if payload['step'] != c['steps']:
                    raise PermissionError('only the frozen final checkpoint may open sealed_test')
                approved = True
            from .evaluate import evaluate
            result = evaluate(model, c, a.task, split=a.split, sealed_authorized=approved,
                nodes=c['ood_nodes'] if a.split == 'ood' and a.ood_condition == 'larger' else None,
                depth=c['ood_depth'] if a.split == 'ood' and a.ood_condition == 'longer' else None,
                fixed_candidates=a.ood_condition == 'fixed_candidates')
        elif a.mode == 'intervene':
            from .intervention import intervene
            from .report import analyze
            if not a.screening_root or not analyze(a.screening_root, c)['m3_authorized_by_screening']:
                raise PermissionError('trained-model M3 interventions require positive Task B screening')
            result = intervene(model, c, a.task)
        else:
            from .profile import profile
            result = profile(model, c, a.task)
        result['checkpoint_sha256'] = file_sha256(a.checkpoint)
        result['identity'] = identity
        write_json(a.output / (a.mode + '.json'), result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
