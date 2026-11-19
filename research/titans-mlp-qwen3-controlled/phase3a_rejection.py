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
