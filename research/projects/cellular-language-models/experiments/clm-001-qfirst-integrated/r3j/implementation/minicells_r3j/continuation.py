"""Restore a historical full trainer; update only Cells at absolute historical steps."""

import copy
from minicells_r3i.trainer import Trainer
from minicells_r3h.routing import Model
from minicells_r3i.adaptation import observe_update, paired
from minicells_r3e.common import fingerprint


def trainer(runtime, state, seed, arm, device):
    model = Model(runtime, seed, arm[0] + "-" + arm[-1]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.temperature = state["r3h"]["temperature"]
    tr = Trainer(model, runtime, arm)
    tr.parent_identity = copy.deepcopy(state["r3h"]["parent_identity"])
    lineage = state["r3i"]["connection_commits"]
    tr.initial_connection_sha = (
        lineage[0]["previous"] if lineage else state["r3h"]["connection_sha"]
    )
    tr.restore(copy.deepcopy(state))
    if fingerprint(tr.model.state_dict()) != fingerprint(state["model"]) or fingerprint(
        tr.cell_opt.state_dict()
    ) != fingerprint(state["cell_optimizer"]):
        raise RuntimeError("HISTORICAL_TRAINER_RESTORE_MISMATCH")
    return tr


def advance(old, new, data, target, checkpoint):
    before = (old.connection_sha, new.connection_sha)
    rows = []
    paired(old, new)
    while old.step < target:
        if new.step != old.step:
            raise RuntimeError("BRANCH_STEP_MISMATCH")
        # No Proposal dispatcher is imported or called by this loop.
        a = observe_update(old, data)
        b = observe_update(new, data)
        paired(old, new)
        if a["inputs"] != b["inputs"]:
            raise RuntimeError("CONTINUATION_INPUT_HASH_MISMATCH")
        rows.append(
            dict(
                step=old.step,
                input_hashes=a["inputs"],
                action_change=float((a["actions"] != b["actions"]).float().mean()),
                path_change=float((a["actions"] != b["actions"]).any(-1).float().mean()),
                old_cells=a["cell_counts"],
                new_cells=b["cell_counts"],
                old_exposure=copy.deepcopy(old.curve[-1]["exposure"]),
                new_exposure=copy.deepcopy(new.curve[-1]["exposure"]),
                old_gradients=a["gradients"],
                new_gradients=b["gradients"],
            )
        )
        if (old.connection_sha, new.connection_sha) != before:
            raise RuntimeError("CONTINUATION_CONNECTION_CHANGED")
        old.verify_frozen()
        new.verify_frozen()
        # One atomic paired state per update prevents swapping halves after interruption.
        checkpoint(old, new, rows[-1])
    if old.step != target or new.step != target:
        raise RuntimeError("CONTINUATION_HORIZON_OVERRUN")
    return rows
