"""Private counterfactual identity, not a historical R3i accepted Connection commit."""

import copy
from minicells_r3h.trainer import Trainer
from minicells_r3h.routing import Model
from minicells_r3e.common import fingerprint


def restore(runtime, state, seed, arm, device, connection=None):
    state = copy.deepcopy(state)
    model = Model(runtime, seed, arm[0] + "-" + arm[-1]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.temperature = state["r3h"]["temperature"]
    if connection is not None:
        model.connection.load_state_dict(connection, strict=True)
        for key, value in connection.items():
            state["model"]["connection." + key] = value.clone()
        state["r3h"]["connection_sha"] = fingerprint(model.connection.state_dict())
    tr = Trainer(model, runtime, arm)
    tr.parent_identity = copy.deepcopy(state["r3h"]["parent_identity"])
    tr.restore(state)
    if fingerprint(tr.cell_opt.state_dict()) != fingerprint(state["cell_optimizer"]):
        raise RuntimeError("CELL_OPTIMIZER_RESTORE_CHANGED")
    tr.verify_frozen()
    return tr
