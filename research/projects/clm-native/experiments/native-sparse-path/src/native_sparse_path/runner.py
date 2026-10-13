"""One subprocess per experiment job; one serial queue per requested device."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from pathlib import Path
from .reuse import BASE, ROOT, write_json


def screen(config_path, config, output, source_commit, devices=('cpu',)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(task, arm) for task in config['tasks'] for arm in config['screening_arms']]

    def queue(worker):
        device = devices[worker]
        completed = []
        for task, arm in jobs[worker::len(devices)]:
            folder = output / task / arm
            if (folder / 'metrics.json').exists():
                prior = json.loads((folder / 'metrics.json').read_text())
                from .reuse import canonical_sha256, code_digest, file_sha256
                if (prior['identity']['source_commit'] != source_commit
                        or prior['identity']['config_sha256'] != canonical_sha256(config)
                        or prior['identity']['code_sha256'] != code_digest()
                        or prior['checkpoint_sha256'] != file_sha256(folder / 'checkpoint.pt')
                        or prior['train_steps'] != config['steps']):
                    raise ValueError('existing evidence differs from source/config')
                print(f'Preserving completed {task}/{arm}', flush=True)
                completed.append((task, arm))
                continue
            cmd = [sys.executable, str(BASE / 'run.py'), '--mode', 'train', '--arm', arm,
                '--task', task, '--seed', str(config['screening_seed']), '--config', str(config_path),
                '--output', str(folder), '--source-commit', source_commit, '--device', device]
            if (folder / 'checkpoint.pt').exists():
                cmd.append('--resume')
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'worker.log').open('a') as log:
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(f'[{task}/{arm}@{device}] {line.rstrip()}', flush=True)
                    if process.wait():
                        raise RuntimeError(f'{task}/{arm} failed; logs/checkpoints preserved')
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
            completed.append((task, arm))
        return completed

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        completed = list(executor.map(queue, range(len(devices))))
    # Profile only after every training process has exited: no cross-GPU training
    # or shared-host scheduling interference in the comparison timing phase.
    for index, (task, arm) in enumerate(jobs):
        folder = output / task / arm
        if (folder / 'profile.json').exists():
            continue
        device = devices[index % len(devices)]
        print(f'Isolated hard-inference profile {task}/{arm}@{device}', flush=True)
        command = [sys.executable, str(BASE / 'run.py'), '--mode', 'profile', '--arm', arm,
            '--task', task, '--config', str(config_path), '--output', str(folder),
            '--source-commit', source_commit, '--device', device,
            '--checkpoint', str(folder / 'checkpoint.pt')]
        with (folder / 'profile-worker.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(output / 'orchestration.json', {'status': 'SCREENING_JOBS_COMPLETE',
        'source_commit': source_commit, 'devices': list(devices), 'completed': completed,
        'scientific_decision': False, 'isolated_profiles_after_all_training': True})
    return completed
