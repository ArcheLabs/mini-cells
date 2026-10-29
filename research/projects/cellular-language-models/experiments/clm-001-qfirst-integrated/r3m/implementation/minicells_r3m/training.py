"""Existing Cell trainer with continuous schedule and checkpointed R3m state."""

import copy
import types
from minicells_r3e.common import fingerprint
from minicells_r3l.continuation import restore as parent_restore
from .protocol import schedule


def semantic(state):
    return {
        k: state[k]
        for k in (
            "model",
            "cell_optimizer",
            "data_rng",
            "route_rng",
            "cf_rng",
            "step",
            "data_cursor",
            "optimizer_identity",
            "frozen_sha256",
            "global_cpu_rng",
            "global_cuda_rng",
            "python_rng",
            "numpy_rng",
        )
    }


def restore(runtime, state, seed, device, connection=None):
    return parent_restore(runtime, state, seed, "B-H", device, connection)


def advance(tr, batches, c, schedule_state, perturbed=False):
    rel = tr.step - c["anchor_step"]
    lr, epsilon = schedule(schedule_state["anchor_lr"], schedule_state["anchor_epsilon"], rel, c)
    old_lr = tr.c["engine"]["training"]["learning_rate"]
    old_epsilon = tr.epsilon
    old_select = tr.model.select

    # Binding is temporary and is removed before any state is persisted or cloned.
    def fixed_epsilon(self):
        return epsilon

    def shuffled(self, h, t, previous, **kw):
        shift = 1 + (rel // c["perturbation"]["every"] + t) % max(1, len(h) - 1)
        observer = self.observer

        def observe(stage, prev, action, hard, p):
            observer(stage, prev, action.roll(shift, 0), hard, p.roll(shift, 0))

        if observer is not None and len(h) > 1:
            self.observer = observe
        try:
            action, z = old_select(h, t, previous, **kw)
        finally:
            self.observer = observer
        if len(action) > 1:
            action = action.roll(shift, 0)
        return action, z

    try:
        tr.c["engine"]["training"]["learning_rate"] = lr
        tr.epsilon = types.MethodType(fixed_epsilon, tr)
        if perturbed:
            tr.model.select = types.MethodType(shuffled, tr.model)
        row = tr.update(None, batches=batches)
    finally:
        tr.c["engine"]["training"]["learning_rate"] = old_lr
        tr.epsilon = old_epsilon
        if perturbed:
            tr.model.select = old_select
    row.update(lr=lr, epsilon=epsilon, relative_step=rel + 1, perturbed=perturbed)
    if sum(row["cell_counts"]) != c["calls_per_token"] * c["tokens_per_step"]:
        raise RuntimeError("CELL_CALL_BUDGET_CHANGED")
    return row


def change_connection(tr, connection):
    opt = fingerprint(tr.cell_opt.state_dict())
    tr.model.connection.load_state_dict(connection, strict=True)
    tr.connection_sha = fingerprint(tr.model.connection.state_dict())
    for p in tr.model.connection.parameters():
        p.requires_grad_(False)
        p.grad = None
    tr.verify_frozen()
    if opt != fingerprint(tr.cell_opt.state_dict()):
        raise RuntimeError("CONNECTION_CHANGE_POLLUTED_CELL_OPTIMIZER")


def snapshot(tr, aux):
    return dict(trainer=copy.deepcopy(tr.state()), aux=copy.deepcopy(aux))


def initial_aux(tr, c, identity):
    q = tr.c["training"]
    anchor_epsilon = q["epsilon_initial"] + (
        q["epsilon_final"] - q["epsilon_initial"]
    ) * tr.step / max(1, q["steps"] - 1)
    return dict(
        identity=identity,
        schedule=dict(anchor_lr=tr.cell_opt.param_groups[0]["lr"], anchor_epsilon=anchor_epsilon),
        seen=[],
        transaction=None,
        transactions=[],
        pressure=None,
        curriculum_step=0,
        main_updates=0,
        shadow_updates=0,
        discarded_shadow_updates=0,
        proposal_optimizer_updates=0,
        curve=[],
        frozen_connection=fingerprint(tr.model.connection.state_dict()),
    )
