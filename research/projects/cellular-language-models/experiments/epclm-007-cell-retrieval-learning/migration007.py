"""Verify Direct Gate Cell portability across sequential CPU processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import torch

from preflight007 import direct_fixture, direct_read, direct_write
from retrieval import compatibility, digest, export_cell, import_cell, rank, state_hash
from cell_artifact import read_manifest, tensor_digest


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def retrieve(model, queries, candidates, labels, cell):
    before = state_hash(cell)
    with torch.no_grad():
        readouts = direct_read(model, queries, cell)
        scores = rank(readouts, candidates)
    order = scores.argsort(dim=-1, descending=True, stable=True)
    ranks = (order == labels[:, None]).long().argmax(dim=-1) + 1
    margins = scores.gather(1, order[:, :2]).diff(dim=1).neg().flatten()
    return dict(readouts=readouts.tolist(), scores=scores.tolist(),
                dtype=str(readouts.dtype), readout_shape=list(readouts.shape),
                score_shape=list(scores.shape), score_dtype=str(scores.dtype),
                order=order.tolist(), top1=order[:, 0].tolist(), labels=labels.tolist(),
                top1_correct=int((ranks == 1).sum()), top2_correct=int((ranks <= 2).sum()),
                mean_rank=float(ranks.float().mean()), margins=margins.tolist(),
                mean_margin=float(margins.mean()), state_before=before, state_after=state_hash(cell))


def session(role, directory):
    model, queries, candidates, labels = direct_fixture()
    model.requires_grad_(False)
    # This fixture uses identity vectors directly and initializes static weights from seed.
    static_hash = tensor_digest(model.state_dict())
    identity = compatibility(model, static_hash)
    identity['cell_interface_version'] = 'separated-kvq-v2'
    runtime = dict(pid=os.getpid(), runtime_id=str(uuid.uuid4()), device='cpu',
                   python=sys.version.split()[0], torch=str(torch.__version__),
                   checkpoint_hash=None, static_initialization_hash=static_hash,
                   compatibility=identity,
                   retrieval_fixture_hash=digest([queries.tolist(), candidates.tolist()]))
    artifact = directory / 'cell'
    if role == 'source':
        cell, _, delta = direct_write(model, queries, candidates, labels)
        result = retrieve(model, queries, candidates, labels, cell)
        runtime.update(retrieval=result, updates=cell.update_step, fast_delta=delta,
                       seed=41035, facts=8, update_mode='full-set replay', export_before=state_hash(cell))
        save(directory / 'source.json', runtime)
        if result['top1_correct'] != 5 or cell.update_step != 128:
            raise RuntimeError('Frozen source baseline did not reproduce')
        started = time.monotonic()
        sha = export_cell(cell, artifact, identity, runtime['runtime_id'])
        runtime.update(artifact_sha256=sha, artifact_path='cell',
                       artifact_bytes=sum(p.stat().st_size for p in artifact.iterdir()),
                       export_seconds=time.monotonic() - started,
                       manifest=read_manifest(artifact))
    else:
        started = time.monotonic()
        cell = import_cell(artifact, model, identity)
        runtime.update(import_seconds=time.monotonic() - started, compatibility_status='PASS',
                       imported_hash=state_hash(cell), updates=0,
                       retrieval=retrieve(model, queries, candidates, labels, cell))
    save(directory / f'{role}.json', runtime)


def compare(source, target):
    a, b = source['retrieval'], target['retrieval']
    result = dict(atol=1e-6, rtol=1e-5)
    for key in ('readouts', 'scores', 'margins'):
        x, y = (torch.tensor(row[key], dtype=torch.float64) for row in (a, b))
        delta = (x-y).abs()
        result[key] = dict(max_abs=float(delta.max()),
                           max_relative=float((delta / x.abs().clamp_min(1e-12)).max()),
                           close=bool(torch.allclose(y, x, atol=1e-6, rtol=1e-5)))
    result.update(top1_agreement=sum(x == y for x, y in zip(a['top1'], b['top1'])),
                  top2_agreement=sum(x[:2] == y[:2] for x, y in zip(a['order'], b['order'])),
                  full_rank_agreement=a['order'] == b['order'],
                  source_top1=a['top1_correct'], target_top1=b['top1_correct'],
                  metadata_equal=all(a[k] == b[k] for k in
                                     ('dtype', 'readout_shape', 'score_shape', 'score_dtype', 'labels')),
                  fixture_equal=all(source[k] == target[k] for k in
                                    ('compatibility', 'retrieval_fixture_hash')),
                  state_integrity=all(source['export_before'] == value for value in (
                      source['manifest']['state_hash'], target['imported_hash'],
                      a['state_before'], a['state_after'], b['state_before'], b['state_after'])))
    return result


def run(directory):
    directory.mkdir(parents=True)
    report = dict(status='BLOCKED', git_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip())
    try:
        for role in ('source', 'target'):
            report['phase'] = role
            # subprocess.run waits for and reaps Source before Target is launched.
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(directory.resolve()), '--role', role],
                capture_output=True, text=True, timeout=120)
            report[f'{role}_exit_status'] = process.returncode
            (directory / f'{role}.log').write_text(process.stdout + process.stderr)
            if process.returncode:
                report.update(status='BLOCKED' if role == 'source' else 'PORTABILITY_FAIL',
                              error=process.stderr.strip())
                return report
            report[role] = json.loads((directory / f'{role}.json').read_text())
        report['phase'] = 'comparison'
        source, target = report['source'], report['target']
        result = compare(source, target)
        report['comparison'] = result
        report['independent_processes'] = (source['pid'] != target['pid']
                                          and source['runtime_id'] != target['runtime_id'])
        passed = (report['independent_processes'] and target['updates'] == 0
                  and all(result[k]['close'] for k in ('readouts', 'scores', 'margins'))
                  and all(result[k] for k in ('full_rank_agreement', 'metadata_equal',
                                               'fixture_equal', 'state_integrity'))
                  and result['source_top1'] == result['target_top1'] == 5)
        report['status'] = 'PORTABILITY_PASS' if passed else 'PORTABILITY_FAIL'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        save(directory / 'decision.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--role', choices=('source', 'target'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.role:
        session(args.role, args.output)
    else:
        decision = run(args.output)
        print(decision['status'])
        sys.exit(decision['status'] != 'PORTABILITY_PASS')
