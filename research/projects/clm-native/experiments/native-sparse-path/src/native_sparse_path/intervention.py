"""Training-only frozen edges and paired legal, degree-preserving replay."""
import numpy as np
import torch
from .generator import generate, permute
from .reference_solver import solve
from .evaluate import evaluate


def rewire_slots(trace, seed):
    """Swap legal selected targets between sources, preserving target multiset.

    This is offline intervention construction, excluded from model latency but
    reported separately. Unchangeable steps are retained, never mislabeled causal.
    """
    rng = np.random.default_rng(seed)
    result, changed = [], 0
    for row in trace:
        candidates = np.asarray(row['candidate_ids'])
        slots = np.asarray(row['slots']).copy()
        targets = np.take_along_axis(candidates, slots, 1)
        for a, b in rng.permutation([(a, b) for a in range(len(slots)) for b in range(a + 1, len(slots))]):
            sa, sb = int(rng.integers(slots.shape[1])), int(rng.integers(slots.shape[1]))
            ta, tb = targets[a, sa], targets[b, sb]
            if (ta in row['targets'] and tb in row['targets']
                    and ta != tb and tb in candidates[a] and ta in candidates[b]
                    and tb not in np.delete(targets[a], sa) and ta not in np.delete(targets[b], sb)):
                slots[a, sa] = int(np.where(candidates[a] == tb)[0][0])
                slots[b, sb] = int(np.where(candidates[b] == ta)[0][0])
                targets[a, sa], targets[b, sb] = tb, ta
                changed += 2
        result.append(slots.tolist())
    return result, changed


def frozen_preferences(model, config, task, count=64):
    device = next(model.parameters()).device
    frequencies = torch.zeros((config['nodes'], config['nodes']), device=device)
    model.eval()
    with torch.no_grad():
        for index in range(count):
            instance = generate(task, 'train', index, nodes=config['nodes'], candidates=config['c'],
                                depth=config['train_depth']).to(device)
            _, _, trace, _ = model(instance, trace=True, force_hard=True)
            for row in trace:
                for source, target in row['edges']:
                    frequencies[source, target] += 1
    return frequencies


def paired_rewire(model, config, task, seed=71, count=32):
    device = next(model.parameters()).device
    rows = []
    model.eval()
    with torch.no_grad():
        for index in range(count):
            instance = generate(task, 'dev', index, nodes=config['nodes'], candidates=config['c'],
                                depth=config['train_depth']).to(device)
            original, _, trace, _ = model(instance, trace=True, force_hard=True)
            original_slots = [r['slots'] for r in trace]
            replay, base_cost, _, _ = model(instance, forced_slots=original_slots, force_hard=True)
            slots, changed = rewire_slots(trace, seed + index)
            damaged, damaged_cost, _, _ = model(instance, forced_slots=slots, force_hard=True)
            restored, restored_cost, _, _ = model(instance, forced_slots=original_slots, force_hard=True)
            target = torch.tensor(solve(instance, k=config['k'], active_limit=config['active_limit']), device=device)
            keys = ('messages', 'cell_updates', 'candidate_reads', 'node_gathers', 'route_score_ops')
            rows.append({'id': instance.instance_id, 'changed_edges': changed,
                'original_mse': float((original - target).square().mean()),
                'rewired_mse': float((damaged - target).square().mean()),
                'replay_parity': bool(torch.allclose(original, replay, atol=1e-6)),
                'restoration_parity': bool(torch.allclose(original, restored, atol=1e-6)),
                'equal_budget': all(base_cost[k] == damaged_cost[k] == restored_cost[k] for k in keys),
                'baseline_cost': base_cost, 'rewired_cost': damaged_cost,
                'original_trace': trace, 'rewired_slots': slots})
    return {'kind': 'degree_and_target_multiset_preserving_legal_edge_swap', 'seed': seed,
            'changed_examples': sum(r['changed_edges'] > 0 for r in rows), 'rows': rows,
            'scientific_decision': False}


def permutation_check(model, config, task, count=16):
    device = next(model.parameters()).device
    model.eval()
    errors = []
    with torch.no_grad():
        for index in range(count):
            instance = generate(task, 'dev', index, nodes=config['nodes'], candidates=config['c'],
                                depth=config['train_depth'])
            order = np.random.default_rng(index + 97).permutation(config['nodes'])
            shuffled = permute(instance, order)
            a = model(instance.to(device), force_hard=True)[0]
            b = model(shuffled.to(device), force_hard=True)[0]
            errors.append(float((a - b).abs().max()))
    return {'max_error': max(errors), 'passed': max(errors) < 1e-5,
            'scope': 'C0/B0/B3 should be equivariant; B2 absolute identity prior may fail', 'errors': errors}


def intervene(model, config, task):
    frozen = frozen_preferences(model, config, task)
    return {'paired_rewire': [paired_rewire(model, config, task, seed=s) for s in (71, 72, 73)],
        'permutation': permutation_check(model, config, task),
        'controls': {kind: evaluate(model, config, task, intervention=kind,
                                   frozen=frozen if kind == 'freeze' else None, count=32)
                     for kind in ('freeze', 'random_dynamic', 'zero_message', 'redistribute')},
        'freeze_index_bytes': frozen.numel() * frozen.element_size(),
        'freeze_scope': 'offline train-only N*N frequency buffer; no pairwise scoring, '
                        'extra lookup and storage charged; not an efficiency baseline'}


def has_causal_signal(report):
    groups = report.get('paired_rewire', [])
    if len(groups) < 3:
        return False
    for group in groups:
        changed = [r for r in group['rows'] if r['changed_edges'] > 0]
        if len(changed) < 8 or not all(r['equal_budget'] and r['replay_parity']
                                    and r['restoration_parity'] for r in changed):
            return False
        delta = np.asarray([r['rewired_mse'] - r['original_mse'] for r in changed])
        rng = np.random.default_rng(group['seed'])
        samples = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
        if np.quantile(samples, .025) <= 0:
            return False
    return True
