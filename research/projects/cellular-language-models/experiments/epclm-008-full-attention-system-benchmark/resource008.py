"""Isolated FP32 engineering measurements and process-restart validation."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from run_epclm008 import BASE, M2, PortableModel, build, file_hash, write
from portable_model import P


ARMS = ('E0', 'E3', 'MAG')
LENGTHS = (1024, 4096, 16384)


def setup(arm, device):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if arm == 'MAG':
        model = PortableModel()
        model.backbone.requires_grad_(False)
    else:
        config = M2['arms'][arm]
        model = build(arm, config['model'], 42, state_size=config['state_size'])
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=.0003, betas=(.9, .95), weight_decay=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    return model, optimizer, scheduler


def objective(model, arm, ids, cell=None):
    if arm != 'MAG':
        logits = model(ids[:, :-1])
        labels = ids[:, 1:]
    else:
        packet = P['segment_tokens']
        source = ids[:, :P['bindings'] * packet].reshape(1, P['bindings'], packet)
        queries = ids[:, -1-P['queries']*packet:-1].reshape(1, P['queries'], packet)
        cell = model.acquire(source, cell)
        hidden = model.backbone(ids[:, :-1])
        logits = model.query(hidden[:, -1-P['queries']*packet+packet::packet], queries, cell)
        labels = ids[:, -1-P['queries']*packet+packet::packet]
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten()), logits, cell, labels.numel()


def step(model, optimizer, scheduler, arm, length, cursor, device, cell=None):
    # Synthetic token stream. Cursor advances only after a complete optimizer step.
    ids = torch.randint(8, 8192, (1, length + 1),
                        generator=torch.Generator().manual_seed(1000 + cursor)).to(device)
    optimizer.zero_grad(set_to_none=True)
    loss, logits, cell, targets = objective(model, arm, ids, cell)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return logits.detach().cpu(), cell.detached() if cell is not None else None, targets


def profile(arm, length, device):
    model, optimizer, scheduler = setup(arm, device)
    step(model, optimizer, scheduler, arm, length, 0, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    _, _, targets = step(model, optimizer, scheduler, arm, length, 1, device)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return dict(status='PASS', seconds=seconds, targets=targets, input_tokens=length,
                targets_per_second=targets/seconds, allocated_bytes=torch.cuda.max_memory_allocated(),
                reserved_bytes=torch.cuda.max_memory_reserved())


def snapshot(model, optimizer, scheduler, cursor, cell, logits, device):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), cursor=cursor, logits=logits,
                cell=None if cell is None else dict(fast=cell.fast, momentum=cell.momentum,
                                                   step=cell.update_step, metadata=cell.metadata),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state() if device == 'cuda' else None,
                python_rng=random.getstate(),
                numpy_rng=(lambda s: (s[0], s[1].tolist(), *s[2:]))(np.random.get_state()))


def restore(saved, model, optimizer, scheduler, device):
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    torch.set_rng_state(saved['torch_rng'])
    if device == 'cuda':
        torch.cuda.set_rng_state(saved['cuda_rng'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    state = saved['cell']
    if state is not None:
        from memory import Cell
        return Cell(*({k: v.to(device) for k, v in state[group].items()}
                      for group in ('fast', 'momentum')), state['step'], state['metadata'])
    return None


def recovery(arm, output, device, worker=False):
    model, optimizer, scheduler = setup(arm, device)
    args = (model, optimizer, scheduler, arm, 1024)
    checkpoint = output / f'{arm}-engineering.pt'
    if worker:
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        cell = restore(saved, model, optimizer, scheduler, device)
        logits, cell, _ = step(*args, saved['cursor'], device, cell)
        torch.save(snapshot(model, optimizer, scheduler, 2, cell, logits, device),
                   output / f'{arm}-resumed.pt')
        return
    logits, cell, _ = step(*args, 0, device)
    torch.save(snapshot(model, optimizer, scheduler, 1, cell, logits, device), checkpoint)
    sha = file_hash(checkpoint)
    logits, cell, _ = step(*args, 1, device, cell)
    expected = snapshot(model, optimizer, scheduler, 2, cell, logits, device)
    del args, model, optimizer, scheduler, cell
    # Move the comparison state off GPU before launching the independent worker.
    expected_file = output / f'{arm}-continuous.pt'
    torch.save(expected, expected_file)
    del expected
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    subprocess.run([sys.executable, __file__, str(output), '--device', device, '--worker', arm],
                   check=True, timeout=600)
    resumed_file = output / f'{arm}-resumed.pt'
    expected, resumed = (torch.load(path, weights_only=True, map_location='cpu')
                         for path in (expected_file, resumed_file))
    for state in (expected, resumed):
        if state['cell'] is not None:
            # Cell UUID identifies the episode. Compare numerical state separately.
            state['cell'].pop('metadata')
    torch.testing.assert_close(expected, resumed, rtol=0, atol=0)
    assert file_hash(checkpoint) == sha
    expected_file.unlink()
    resumed_file.unlink()
    return dict(status='PASS', checkpoint=checkpoint.name, sha256=sha,
                bytes=checkpoint.stat().st_size, comparison='exact', cursor=2)


def run(output, device):
    output.mkdir(parents=True)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE, text=True).strip()
    version = os.environ.get('KAGGLE_SCRIPT_VERSION_ID', '')
    report = dict(source_commit=source, script_version_id=int(version) if version.isdigit() else None,
                  config_sha256=file_hash(BASE / 'config.lock.json'), runner_sha256=file_hash(__file__),
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  precision='fp32', batch=1, test='SEALED', official_updates=0, measurements={}, recovery={})
    for arm in ARMS:
        rows = report['measurements'][arm] = {}
        for length in LENGTHS:
            try:
                rows[length] = profile(arm, length, device)
            except torch.cuda.OutOfMemoryError:
                rows[length] = dict(status='OOM')
            except Exception as error:
                rows[length] = dict(status='FAIL', error=str(error))
            gc.collect()
            torch.cuda.empty_cache()
            print(arm, length, rows[length], flush=True)
            write(output / 'resource.json', report)
        try:
            report['recovery'][arm] = recovery(arm, output, device)
        except Exception as error:
            report['recovery'][arm] = dict(status='FAIL', error=str(error))
        write(output / 'resource.json', report)
    estimates = {}
    for length in LENGTHS:
        rows = [report['measurements'][arm][length] for arm in ARMS]
        if all(row['status'] == 'PASS' for row in rows):
            estimates[length] = {str(budget): 3 * (sum(budget / r['targets_per_second']
                                  for r in rows[:2]) + 1024 * rows[2]['seconds'])
                                 for budget in (20_000_000, 100_000_000)}
    write(output / 'budget.json', dict(compute_only_seconds=estimates, formal_budget=None,
          caveat='Batch-one synthetic engineering extrapolation; excludes evaluation and storage.'))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--worker', choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.worker:
        recovery(args.worker, args.output, args.device, worker=True)
    else:
        if args.device != 'cuda' or not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
            raise RuntimeError('Tesla T4 required for resource measurements')
        run(args.output, args.device)
