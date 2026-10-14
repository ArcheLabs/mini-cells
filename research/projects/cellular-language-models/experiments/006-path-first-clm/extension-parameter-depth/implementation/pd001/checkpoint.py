"""Reuse atomic 005 checkpoint and persist each independent rollout generator."""

import numpy as np
from exp006.reuse import save_checkpoint, restore_checkpoint


def save(path, model, optimizer, step, identity, data_rng, path_rngs, extra=None, scaler=None):
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=step,
        scaler=scaler,
        identity=identity,
        generator=data_rng,
        extra={
            **(extra or {}),
            "path_rngs": [g.get_state() for g in path_rngs],
            "numpy_global_rng": np.random.get_state(),
        },
    )


def restore(path, model, optimizer, identity, data_rng, path_rngs, scaler=None, device="cpu"):
    step, extra = restore_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        expected_identity=identity,
        generator=data_rng,
        map_location=device,
    )
    states = extra.pop("path_rngs")
    if len(states) != len(path_rngs):
        raise ValueError("rollout generator count mismatch")
    for generator, state in zip(path_rngs, states):
        generator.set_state(state.cpu())
    np.random.set_state(extra.pop("numpy_global_rng"))
    return step, extra
