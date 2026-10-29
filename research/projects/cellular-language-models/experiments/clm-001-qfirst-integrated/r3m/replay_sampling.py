"""CPU-only historical sampler replay; predictions are never promoted without evidence."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[5]
SAMPLER = BASE.parent / 'r3/implementation/clm001_r3/data.py'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def drawer():
    """Reuse original loop count and exact integers call AST; no torch import/tensors."""
    source_proof = json.loads((BASE / 'registration/sampling-trace-001/ORIGINAL_SAMPLER_SOURCE_CHECK.json').read_text())
    for row in source_proof:
        if not row.get('identical') or sha(ROOT / row['path']) != row['original_sha256']:
            raise ValueError('HISTORICAL_SAMPLER_SOURCE_NOT_IDENTICAL')
    tree = ast.parse(SAMPLER.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'batches')
    loop = next(n for n in function.body if isinstance(n, ast.For))
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and
             n.func.value.id == 'rng']
    if len(calls) != 1 or calls[0].func.attr != 'integers':
        raise ValueError('ORIGINAL_SAMPLER_STRUCTURE_CHANGED')
    loop_code = compile(ast.Expression(loop.iter), str(SAMPLER), 'eval')
    draw_code = compile(ast.Expression(calls[0]), str(SAMPLER), 'eval')

    def draw(c, starts, rng):
        namespace = dict(c=c, starts=starts, rng=rng)
        return [[int(i) for i in eval(draw_code, {}, namespace)]
                for _ in eval(loop_code, {}, namespace)]
    return draw


def replay(c, windows, seed, evidence=None):
    draw = drawer()
    rng = np.random.default_rng(seed + c['training']['data_seed_offset'])
    w = windows['train']
    length = c['engine']['training']['sequence_length']
    states = {0: copy.deepcopy(rng.bit_generator.state)}
    rows = []
    for step in range(1, c['training']['steps'] + 1):
        for microbatch, indices in enumerate(draw(c['engine'], w['starts'], rng)):
            for row, index in enumerate(indices):
                start, group = w['starts'][index], w['groups'][index]
                if not w['story_offsets'][group] <= start < start + length < w['story_offsets'][group + 1]:
                    raise ValueError('INPUT_TARGET_CROSSES_STORY')
                rows.append(dict(step=step, microbatch=microbatch, row=row, eligible_index=index,
                                 start=start, group=group, raw_story_index=w['raw_story_ids'][group],
                                 story_token_offset=start - w['story_offsets'][group],
                                 input_span=[start, start + length], target_span=[start + 1, start + length + 1]))
        if step % 100 == 0 or step == c['training']['steps']:
            states[step] = copy.deepcopy(rng.bit_generator.state)
    comparisons = []
    historical_version = None
    if evidence:
        if evidence['seed'] != seed or evidence['config_sha256'] != canonical(c):
            raise ValueError('CHECKPOINT_EVIDENCE_IDENTITY_MISMATCH')
        if evidence['sampler_source_sha256'] != sha(SAMPLER):
            raise ValueError('CHECKPOINT_SAMPLER_SOURCE_MISMATCH')
        historical_version = evidence.get('historical_numpy_version')
        for boundary in evidence['boundaries']:
            step = boundary['step']
            if step not in states:
                continue
            trusted = all(boundary.get(k) is True for k in
                          ('checkpoint_bytes_verified', 'receipt_identity_verified', 'full_state_fingerprint_verified'))
            matched = canonical(states[step]) == canonical(boundary['data_rng'])
            cursor_ok = boundary['data_cursor'] == step * c['training']['tokens_per_step']
            comparisons.append(dict(step=step, full_rng_state_match=matched, cursor_match=cursor_ok,
                                    original_checkpoint_verified=trusted,
                                    checkpoint_sha256=boundary['checkpoint_sha256'],
                                    receipt_sha256=boundary['receipt_sha256']))
    eligible = {x['step'] for x in comparisons if x['full_rng_state_match'] and x['cursor_match']
                and x['original_checkpoint_verified']}
    intervals = []
    for a, b in zip(sorted(states), sorted(states)[1:]):
        boundaries_match = a in eligible and b in eligible
        version_match = historical_version is not None and historical_version == np.__version__
        intervals.append(dict(start_step_exclusive=a, end_step_inclusive=b,
                              boundary_consistent=boundaries_match,
                              comparison_performed=a in {x['step'] for x in comparisons} and b in {x['step'] for x in comparisons},
                              status='CHECKPOINT_BOUNDED_SAMPLER_REPLAY_VERIFIED' if boundaries_match and version_match
                              else 'NOT_CERTIFIED',
                              historical_numpy_identity_match=version_match))
    return rows, dict(seed=seed, config_sha256=canonical(c), current_numpy_version=np.__version__,
                      historical_numpy_version=historical_version or 'UNKNOWN',
                      bit_generator=rng.bit_generator.__class__.__name__,
                      sampler_source_sha256=sha(SAMPLER), predicted_boundary_states=states,
                      comparisons=comparisons, intervals=intervals,
                      status='CANDIDATE_REPLAY_NOT_HISTORICAL_PROOF' if not evidence else
                      ('BOUNDARY_REVIEW_RECORDED' if comparisons else 'NO_BOUNDARIES_COMPARED_CANDIDATE_ONLY'),
                      scientific_registration='BLOCKED', optimizer_updates=0, gpu_work=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--windows', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-evidence', type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('IMMUTABLE_TRACE_OUTPUT_EXISTS')
    c = json.loads((BASE.parent / 'r3h/config/frozen.json').read_text())
    expected = json.loads((BASE / 'registration/artifact-recovery-001/independent-review-001/manifest.json').read_text())
    if sha(a.windows) != expected['files']['windows.json']:
        raise ValueError('FROZEN_WINDOW_IDENTITY_MISMATCH')
    windows = json.loads(a.windows.read_text())
    evidence = json.loads(a.checkpoint_evidence.read_text()) if a.checkpoint_evidence else None
    if evidence:
        baseline = json.loads((BASE.parent / 'r3i/config/frozen.json').read_text())['baseline']
        if (evidence['schema'] != 'minicells.r3m.original-checkpoint-sampler-evidence.v1'
                or evidence['extractor_sha256'] != sha(BASE / 'recover_sampling_boundaries.py')
                or evidence['hf_reference'] != baseline['hf_reference']
                or evidence['original_identity'] != {k: baseline[k] for k in ('experiment_id', 'source_sha', 'config_sha256', 'run_id')}
                or evidence['arm_scope'] != ['B-H']
                or len({j['seed'] for j in evidence['jobs']}) != len(evidence['jobs'])):
            raise ValueError('BOUNDARY_EVIDENCE_PROVENANCE_MISMATCH')
    a.output.mkdir(parents=True)
    reports = []
    for seed in c['seeds']:
        ev = next((e for e in evidence['jobs'] if e['seed'] == seed), None) if evidence else None
        rows, report = replay(c, windows, seed, ev)
        path = a.output / ('seed-%s-candidate-trace.jsonl' % seed)
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
        report['trace'] = dict(path=path.name, sha256=sha(path), windows=len(rows))
        reports.append(report)
    result = dict(schema='minicells.r3m.sampling-replay.v1', auditor_sha256=sha(__file__),
                  window_sha256=sha(a.windows), jobs=reports,
                  scope='R3H_NORMAL_MAIN_STREAM_ONLY_NOT_R3I_PROPOSAL_SHADOW_OR_R3J_R3K_R3L',
                  arm_scope='UNATTESTED_SEED_PREDICTIONS' if evidence is None else evidence['arm_scope'],
                  global_content_isolation='BLOCKED', scientific_registration='BLOCKED')
    (a.output / 'SAMPLING_REPLAY_REVIEW.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='CPU_REPLAY_WRITTEN_NO_UNATTESTED_INTERVAL_PROMOTED', seeds=c['seeds'])))


if __name__ == '__main__':
    main()
