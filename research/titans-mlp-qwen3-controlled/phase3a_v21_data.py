"""Balanced relation assignments and audited role-span templates for v2.1."""
from collections import Counter
from itertools import combinations
import random

from phase3a import RELATIONS

HELD = ('contributes to', 'is a member of', 'has been allocated to',
        'runs', 'answers to', 'makes use of', 'oversees', 'is based in')


def dataset(seed, sizes):
    rng = random.Random(seed)
    entities = iter(rng.sample(range(1000000), sum(sizes)))
    values = iter(rng.sample(range(1000000), 4 * sum(sizes)))
    universe = frozenset(range(8))
    pairs = [(combo, tuple(sorted(universe.difference(combo))))
             for combo in combinations(range(8), 4) if 0 in combo]
    splits = {}
    for name, size in zip(('train', 'validation', 'test'), sizes, strict=True):
        if size % 2:
            raise ValueError('Each split requires an even number of entities')
        assignments = []
        while len(assignments) < size:
            rng.shuffle(pairs)
            for pair in pairs[:(size - len(assignments)) // 2]:
                assignments.extend(pair)
        rng.shuffle(assignments)
        splits[name] = [(f'E{e:06}', RELATIONS[r], f'V{next(values):06}')
                        for group in assignments for e in [next(entities)] for r in group]
    return splits


def view(row, role, variant=0, held=False):
    entity, relation, value = row
    predicate = HELD[RELATIONS.index(relation)] if held else relation
    if role == 'write':
        prefix = ('Field notes confirm that ', 'The latest briefing says that ')[variant] if held else (
            '', 'The directory confirms that ')[variant]
        infix, suffix = ' ', f' {value}.'
    elif role == 'query':
        prefix = ('Can you tell me what ', 'Please identify what ')[variant] if held else (
            'Do you know what ', 'I need to know what ')[variant]
        infix, suffix = ' ', '?' if variant == 0 else '.'
    else:
        raise ValueError(f'Unknown role {role}')
    start = len(prefix)
    anchor = start + len(entity) + len(infix)
    return prefix + entity + infix + predicate + suffix, (
        (start, start + len(entity)), (anchor, anchor + len(predicate)))


def audit(rows):
    by_entity = {}
    for entity, relation, _ in rows:
        by_entity.setdefault(entity, set()).add(relation)
    combos = Counter(tuple(sorted(group)) for group in by_entity.values())
    pairs = Counter(pair for group in by_entity.values() for pair in combinations(sorted(group), 2))
    return {
        'relations': dict(Counter(row[1] for row in rows)),
        'combinations': {' | '.join(k): v for k, v in sorted(combos.items())},
        'cooccurrences': {' | '.join(k): v for k, v in sorted(pairs.items())},
    }
