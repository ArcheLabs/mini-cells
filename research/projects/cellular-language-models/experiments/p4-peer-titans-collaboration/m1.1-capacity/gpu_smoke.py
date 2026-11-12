"""Bounded T4 smoke and synchronized acquisition timing for P4-M1.1."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from smoke import CONFIG, M3, file_hash, initialize, run_episode, write
from gates import recall_control

SEED = 41036
EXPOSURES = [1, 8]
SCHEMA = 'minicells.p4-m11-gpu-smoke.v1'


def matrix():
    for n in CONFIG['sizes']:
        for method, width in [('batch', n), ('sequential', 1), ('chunk4', 4), ('chunk8', 8)]:
            yield n, 'gaussian', method, width
        if n <= 32:
            yield n, 'qr', 'batch', n


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    config = dict(seed=SEED, exposure=EXPOSURES, matrix=list(matrix()), precision='fp32',
                  unknown_per_kind=CONFIG['unknown_per_kind'], threshold=CONFIG['threshold'])
    result = dict(format=SCHEMA, status='RUNNING', scope='T4 smoke', episodes=[], config=config,
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                  runner_sha256=file_hash(__file__), smoke_sha256=file_hash(Path(__file__).with_name('smoke.py')),
                  torch=torch.__version__, cuda=torch.version.cuda, tf32=False)
    started = time.perf_counter()
    try:
        for name in ('memory.py','gates.py'):
            if file_hash(M3/name) != file_hash(M3/'freeze/original-source'/name):
                raise RuntimeError(f'Frozen M3 source mismatch: {name}')
        result['memory_sha256'] = file_hash(M3/'memory.py')
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Select Tesla T4 in Kaggle accelerator settings.')
        result['gpu'] = torch.cuda.get_device_name(0)
        result['driver'] = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=driver_version',
                                                      '--format=csv,noheader'],text=True,timeout=30).strip()
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        result['control'] = recall_control(output/'control', device='cuda', memory=initialize(41035))
        for n, geometry, method, width in matrix():
            row = run_episode(output/f'{geometry}-{method}-{n}',n,SEED,geometry,method,width,EXPOSURES,'cuda')
            result['episodes'].append(row)
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result['screening_projection_seconds'] = sum(r['curve'][-1]['seconds'] for r in result['episodes'])*320
        result['status'] = 'PASS'
    except BaseException as error:
        result.update(status='BLOCKED',error=repr(error))
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        write(output/'run-manifest.json', result)


def report(result):
    lines = ['# P4-M1.1 Tesla T4 smoke', '', f"Status {result['status']}.",
             f"GPU {result['gpu']}, CUDA {result['cuda']}, driver {result['driver']}.",
             f"Wall {result['wall_seconds']:.2f}s, peak allocated {result['peak_allocated_bytes']:,} bytes.",
             f"Projected acquisition only {result['screening_projection_seconds']/60:.2f} minutes "
             'for 20 mappings per configuration at 128 exposures, extrapolated from one eight-exposure episode per configuration.',
             '', '| Geometry | Method | Bindings | Updates | Presentations | Acquisition seconds | Restore error |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in result['episodes']:
        end = row['curve'][-1]
        lines.append(f"| {row['geometry']} | {row['method']} | {row['n']} | {end['updates']} | "
                     f"{end['presentations']} | {end['seconds']:.4f} | {row['restore_error']:g} |")
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    run(args.output)
    (args.output/'REPORT.md').write_text(report(json.loads((args.output/'run-manifest.json').read_text())))
