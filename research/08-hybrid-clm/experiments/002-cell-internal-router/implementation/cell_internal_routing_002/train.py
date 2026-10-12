"""One development seed per bounded session, three arms queued on two T4 GPUs."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from experiments.common.train import prepare_data, train_one, write_json
from experiments.cell_internal_routing_002.models.cell_router import LanguageModel
from experiments.cell_internal_routing_002.evaluate import evaluate_checkpoint
from experiments.cell_internal_routing_002.report import CONFIG, PROTOCOL, digest, seed_report


def validate_config(config):
    expected = json.loads(CONFIG.read_text())
    expected['seed'] = config['seed']
    if config != expected or config['seed'] not in expected['seeds']:
        raise ValueError('Only the registered configuration and development seeds are accepted')
    protocol = json.loads(PROTOCOL.read_text())
    if config['train_tokens'] != protocol['train_tokens_per_model'] or config['models'] != protocol['models']:
        raise ValueError('Token budget/model matrix differs from protocol')
    if (config['session_budget_seconds'] != protocol['session_budget_seconds']
            or config['data_preparation_budget_seconds'] != protocol['data_preparation_budget_seconds']
            or config['orchestration_reserve_seconds'] != protocol['orchestration_reserve_seconds']
            or config['model_budget_seconds'] != protocol['model_wall_budget_seconds']
            or config['baseline_budget_seconds'] != protocol['baseline_wall_budget_seconds_per_arm']
            or config['hard_evaluation_reserve_seconds'] != protocol['final_evaluation_reserve_seconds_per_arm']
            or config['publication_reserve_seconds'] != protocol['publication_reserve_seconds']):
        raise ValueError('Session time allocation differs from the frozen protocol')
    schedule_budget = max(config['model_budget_seconds'], 2 * config['baseline_budget_seconds'])
    allocated = (config['data_preparation_budget_seconds'] + schedule_budget
                 + config['orchestration_reserve_seconds'] + config['publication_reserve_seconds'])
    if allocated > config['session_budget_seconds']:
        raise ValueError('Data, training, and publication budgets exceed the application session cap')
    if config['width'] % config['heads'] or config['train_tokens'] % (config['batch_size'] * config['sequence_length']):
        raise ValueError('Invalid model dimensions or token budget')
    with torch.device('meta'):
        capacities = [LanguageModel(config, kind).parameter_metrics()['params'] for kind in config['models']]
    if not all(50_000_000 <= count <= 150_000_000 for count in capacities):
        raise ValueError('Registered capacity range violated')
    if max(capacities) / min(capacities) - 1 > protocol['capacity_tolerance_fraction']:
        raise ValueError('Capacity matching must be within 1%')
    return capacities


def worker(kind, config, data, output, deadline):
    local = dict(config)
    local['model_budget_seconds'] = (config['model_budget_seconds'] if kind == 'cell-internal'
                                      else config['baseline_budget_seconds']) - config['hard_evaluation_reserve_seconds']
    # Shared trainer subtracts 120 seconds before saving; retain separate final evaluation time.
    training_deadline = deadline - config['hard_evaluation_reserve_seconds'] + 120
    metrics = train_one(kind, local, data, output, training_deadline, model_factory=LanguageModel)
    directory = output / kind
    # Canonical config is recorded independently; checkpoint retains the actual safety budget.
    if metrics['status'] == 'completed':
        try:
            evaluate_checkpoint(directory / 'checkpoint.pt', data, directory, config, deadline, metrics)
            metrics['checkpoint_sha256'] = digest(directory / 'checkpoint.pt')
            write_json(directory / 'metrics.json', metrics)
        except Exception:
            (directory / 'evaluation-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            metrics['evaluation_status'] = 'failed'
            write_json(directory / 'metrics.json', metrics)
    gc.collect()
    torch.cuda.empty_cache()


def schedule(config, cache, output, deadline):
    # Long arm begins immediately; the second T4 runs both shorter baselines in sequence.
    # Alternate GPU assignment by registered seed, preserving physical GPU provenance.
    first = config['seeds'].index(config['seed']) % 2
    pending = ['cell-internal', 'fixed-cell', 'global-router']
    free = [first, 1 - first]
    running = []
    outcomes = {}
    samples = []
    sample_at = 0.0
    try:
        while (pending or running) and time.time() < deadline:
            while pending and free:
                gpu, kind = free.pop(0), pending.pop(0)
                log = (output / (kind + '-worker.log')).open('w', encoding='utf-8')
                command = [sys.executable, '-m', 'experiments.cell_internal_routing_002.train',
                           '--worker', kind, '--seed', str(config['seed']), '--output', str(output.resolve()),
                           '--data', str(cache), '--deadline', str(deadline)]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false'))
                running.append((gpu, kind, process, log))
            for item in list(running):
                gpu, kind, process, log = item
                if process.poll() is not None:
                    outcomes[kind] = process.returncode
                    log.close()
                    running.remove(item)
                    free.append(gpu)
            if time.time() >= sample_at:
                try:
                    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                                    '--format=csv,noheader,nounits'], text=True, timeout=5)
                    samples.append({'time_unix': time.time(), 'active_models': [row[1] for row in running],
                                    'gpu_rows': rows.strip().splitlines()})
                except (OSError, subprocess.SubprocessError):
                    pass
                sample_at = time.time() + 10
            time.sleep(1)
    finally:
        for gpu, kind, process, log in running:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            outcomes[kind] = process.returncode
        write_json(output / 'gpu-utilization.json', {
            'status': 'sampled' if samples else 'unavailable', 'samples': samples,
            'scope': 'nvidia-smi device utilization sampled every ten seconds; not an isolated kernel efficiency measure'})
    for kind in config['models']:
        path = output / kind / 'metrics.json'
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            write_json(path, {'model': kind, 'status': 'failed', 'worker_exit_code': outcomes.get(kind),
                             'reason': 'worker failed, did not start, or exceeded bounded session'})
        elif outcomes.get(kind) != 0:
            report = json.loads(path.read_text())
            report.update(status='failed', worker_exit_code=outcomes.get(kind))
            write_json(path, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=[42, 123, 456])
    parser.add_argument('--output', default='/kaggle/working/results')
    parser.add_argument('--worker', choices=['fixed-cell', 'global-router', 'cell-internal'])
    parser.add_argument('--data')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    config['seed'] = args.seed
    capacities = validate_config(config)
    output = Path(args.output)
    if args.worker:
        data = {split: np.load(Path(args.data) / (split + '.npy'), mmap_mode='r') for split in ('train', 'validation')}
        worker(args.worker, config, data, output, args.deadline)
        return
    from experiments.cell_internal_routing_002.publish import preflight, publish_seed
    source = os.environ.get('EXPERIMENT_COMMIT', '')
    prior = preflight(source, args.seed)
    output.mkdir(parents=True, exist_ok=False)
    if prior is not None:
        shutil.copytree(prior, output, dirs_exist_ok=True)
        write_json(output / 'publication.json', {'status': 'already_published', 'skipped_gpu_work': True})
        print(json.dumps({'status': 'already_published', 'seed': args.seed}))
        return
    started = float(os.environ.get('EXPERIMENT_STARTED_AT', time.time()))
    deadline = started + config['session_budget_seconds']
    training_deadline = deadline - config['publication_reserve_seconds']
    provenance = {'experiment_id': config['experiment_id'], 'seed': args.seed, 'source_commit': source,
                  'protocol_sha256': digest(PROTOCOL), 'source_config_sha256': digest(CONFIG),
                  'started_at_unix': started, 'python': platform.python_version(), 'torch': torch.__version__,
                  'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'datasets', 'tokenizers')},
                  'capacities': dict(zip(config['models'], capacities)),
                  'execution': 'one seed/session; independent arms queued on two GPUs, not DDP',
                  'precision': 'FP16 with GradScaler', 'scientific_decision': False}
    write_json(output / 'config.json', config)
    write_json(output / 'experiment.json', {**provenance, 'status': 'running'})
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError('This registered schedule requires exactly two CUDA GPUs')
        names = [torch.cuda.get_device_name(i) for i in (0, 1)]
        if any('T4' not in name for name in names):
            raise RuntimeError('The registered execution platform requires dual T4')
        provenance['gpus'] = names
        data_deadline = started + config['data_preparation_budget_seconds']
        data = prepare_data(config, output, data_deadline)
        with tempfile.TemporaryDirectory(prefix='routing002-data-') as cache:
            for split, array in data.items():
                np.save(Path(cache) / (split + '.npy'), array)
            del data
            schedule(config, cache, output, training_deadline)
    except Exception:
        (output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    decision = seed_report(output)
    write_json(output / 'decision.json', decision)
    write_json(output / 'metrics.json', {'models': [json.loads((output / kind / 'metrics.json').read_text())
               for kind in config['models'] if (output / kind / 'metrics.json').exists()],
               'comparable_at_full_budget': decision['comparable_at_full_budget'], 'decision': decision['status']})
    write_json(output / 'experiment.json', {**provenance,
               'status': 'completed' if decision['comparable_at_full_budget'] else 'incomplete',
               'elapsed_seconds': time.time() - started})
    try:
        publish_seed(output, source, args.seed)
    except Exception:
        (output / 'publication-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(output / 'publication.json', {'status': 'failed', 'reason': 'Small evidence preserved in Kaggle outputs'})
        raise
    print(json.dumps({'seed': args.seed, 'status': decision['status'], 'scientific_decision': False}), flush=True)


if __name__ == '__main__':
    main()
