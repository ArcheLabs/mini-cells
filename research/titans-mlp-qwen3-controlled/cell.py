"""Controlled experiment snapshot format using the PCA tensor codec."""
from pathlib import Path
import sys

from memory import State

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'persistent-cognitive'))
from artifact import checked, read, write

SCHEMA = 'controlled-titans-mlp-phase1'


def tensors(state):
    return {f'{group}.{key}': value for group in ('weights', 'momentum')
            for key, value in getattr(state, group).items()}


def export(path, state, fingerprint):
    write(path, dict(schema=SCHEMA, fingerprint=fingerprint, updates=state.updates), tensors(state))


def restore(path, reference, fingerprint):
    metadata, payload = read(path, SCHEMA)
    if (metadata.keys() != {'fingerprint', 'updates'} or metadata['fingerprint'] != fingerprint
            or type(metadata['updates']) is not int or metadata['updates'] < 0):
        raise ValueError('Incompatible Cell metadata')
    state = checked(payload, tensors(reference))
    return State(*(dict((key, state[f'{group}.{key}']) for key in getattr(reference, group))
                   for group in ('weights', 'momentum')), metadata['updates'])


def fork(state):
    return State({k: v.clone() for k, v in state.weights.items()},
                 {k: v.clone() for k, v in state.momentum.items()}, state.updates)
