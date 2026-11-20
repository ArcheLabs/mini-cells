"""Entity-isolated validation and unique rejection fixtures."""
from collections import defaultdict
import random

from phase3a import RELATIONS
from phase3a_v21 import view


def split_validation(rows, seed):
    entities = defaultdict(list)
    for row in rows:
        entities[row[0]].append(row)
    by_profile = defaultdict(list)
    for entity, group in sorted(entities.items()):
        by_profile[frozenset(r for _, r, _ in group)].append(entity)
    universe = frozenset(RELATIONS)
    pairs = []
    for profile, group in by_profile.items():
        if RELATIONS[0] in profile:
            pairs.extend(zip(group, by_profile[universe-profile], strict=True))
    if len(pairs)*2 != len(entities):
        raise ValueError('Validation profiles cannot be balanced')
    random.Random(seed).shuffle(pairs)
    midpoint = len(pairs)//2
    return tuple([row for pair in subset for e in pair for row in entities[e]]
                 for subset in (pairs[:midpoint], pairs[midpoint:]))


def unknown_queries(rows):
    profiles = defaultdict(set)
    for entity, relation, _ in rows:
        profiles[entity].add(relation)
    result = {
        'unknown_entity': [('Z'+e[1:], r, '') for e, r, _ in rows],
        'unknown_relation': [(e, r, '') for e, known in profiles.items()
                             for r in RELATIONS if r not in known],
    }
    for family in ('query','query:1','held_query'):
        rendered = [view(row, family)[0] for group in result.values() for row in group]
        if len(set(rendered)) != len(rendered):
            raise ValueError(f'Duplicate unknown queries in {family}')
    return result


def independent_partitions(rows, seed):
    """Preserve selection and divide its complement into balanced calibration and audit."""
    selection, remainder = split_validation(rows, seed)
    calibration, audit = split_validation(remainder, seed + 1)
    return selection, calibration, audit


def wilson(successes, total):
    if not total:
        return dict(count=0, successes=0, rate=None, ci95=None)
    p = successes / total
    z = 1.95996398454
    d = 1 + z*z/total
    midpoint = (p + z*z/(2*total))/d
    half = z*(p*(1-p)/total + z*z/(4*total*total))**.5/d
    return dict(count=total, successes=successes, rate=p,
                ci95=[max(0., midpoint-half), min(1., midpoint+half)])


def rejection_audit(calibration, audit, calibrate, fpr):
    import torch

    def scores(matrix, method):
        top = matrix.topk(2, dim=1).values
        return top[:, 0] - top[:, 1] if method == 'margin' else top[:, 0]

    families = tuple(calibration['known'])
    if calibration is audit or tuple(audit['known']) != families:
        raise ValueError('Independent audit and calibration must use the same family protocol')

    def negatives(bank, family, method):
        return torch.cat([scores(values[family], method)
                          for name, values in bank.items() if name != 'known'])

    result = {}
    for method in ('global', 'family', 'margin'):
        pooled = torch.cat([negatives(calibration, f, method) for f in families])
        threshold = calibrate(pooled, fpr)
        thresholds = {f:calibrate(negatives(calibration, f, method), fpr)
                      if method == 'family' else threshold for f in families}

        def summarize(cutoffs):
            breakdown = {}
            accepted_total = known_total = correct1 = correct5 = false_total = unknown_total = 0
            types = defaultdict(lambda: [0, 0])
            for family in families:
                known = audit['known'][family]
                selected = scores(known, method) >= cutoffs[family]
                target = torch.arange(len(known))
                ranks = (known >= known[target,target,None]).sum(1)
                n = int(selected.sum())
                a = int((selected & ranks.eq(1)).sum())
                b = int((selected & ranks.le(5)).sum())
                accepted_total += n
                known_total += len(known)
                correct1 += a
                correct5 += b
                by_type = {}
                for kind, matrices in audit.items():
                    if kind == 'known':
                        continue
                    pred = scores(matrices[family], method)
                    hits = int((pred >= cutoffs[family]).sum())
                    by_type[kind] = wilson(hits, len(pred))
                    types[kind][0] += hits
                    types[kind][1] += len(pred)
                family_false = sum(item['successes'] for item in by_type.values())
                family_total = sum(item['count'] for item in by_type.values())
                false_total += family_false
                unknown_total += family_total
                breakdown[family] = dict(fpr=wilson(family_false, family_total), unknown=by_type,
                    coverage=wilson(n,len(known)), selective_recall1=wilson(a,n), selective_recall5=wilson(b,n))
            return dict(fpr=wilson(false_total,unknown_total), coverage=wilson(accepted_total,known_total),
                        selective_recall1=wilson(correct1,accepted_total),
                        selective_recall5=wilson(correct5,accepted_total),
                        unknown={kind:wilson(*counts) for kind,counts in types.items()}, families=breakdown)

        point = summarize(thresholds)
        curve = []
        for budget in (0., .01, .025, .05, .10, .20, .40, .80):
            global_cut = calibrate(pooled,budget)
            cuts = {f:calibrate(negatives(calibration,f,method),budget) if method=='family'
                    else global_cut for f in families}
            current = summarize(cuts)
            accuracy = current['selective_recall1']['rate']
            curve.append(dict(calibration_fpr_budget=budget, fpr=current['fpr']['rate'],
                              coverage=current['coverage']['rate'], risk=None if accuracy is None else 1-accuracy))
        result[method] = dict(thresholds=thresholds, audit=point, curve=curve)
    return result
