"""Multi-event immutable transactions. Resume each paired update without repeating it."""

import copy
from pathlib import Path
import torch
from minicells_r3f_v2.proposal import generate
from minicells_r3f_v2.scale_scan import scan
from minicells_r3f_v2.storage import save, latest
from minicells_r3h.evaluation import quality
from minicells_r3e.common import fingerprint, canonical


def observe_update(tr, data):
    actions = []
    inputs = []

    def before(module, args):
        inputs.append(fingerprint(args[0]))

    def after(module, args, out):
        actions.append(out.actions.detach().cpu())

    a = tr.model.register_forward_pre_hook(before)
    b = tr.model.register_forward_hook(after)
    try:
        row = tr.update(data)
    finally:
        a.remove()
        b.remove()
    return dict(
        inputs=inputs,
        actions=torch.cat(actions),
        cell_counts=row["cell_counts"],
        gradients=row["cell_gradient_norms"],
        step=tr.step,
        cursor=tr.data_cursor,
    )


def paired(a, b):
    if (
        a.step != b.step
        or a.data_cursor != b.data_cursor
        or canonical(a.data_rng.bit_generator.state) != canonical(b.data_rng.bit_generator.state)
    ):
        raise RuntimeError("PAIRED_DATA_STREAM_MISMATCH")
    if not torch.equal(a.route_rng.get_state(), b.route_rng.get_state()):
        raise RuntimeError("PAIRED_ROUTE_STREAM_MISMATCH")


@torch.no_grad()
def crossover(old, new, c, data):
    values = {}
    for cell_name, base in [("O", old), ("N", new)]:
        for connection_name, source in [("O", old), ("N", new)]:
            model = copy.deepcopy(base.model)
            model.connection.load_state_dict(source.model.connection.state_dict(), strict=True)
            values[cell_name + connection_name] = quality(model, c, data, "calibration")["nll"]
    resolution = max(1e-12, torch.finfo(torch.float32).eps * max(abs(x) for x in values.values()))
    cell = values["NN"] - values["ON"]
    interaction = values["NN"] - values["NO"] - values["ON"] + values["OO"]
    return dict(
        values=values,
        route_effect=values["ON"] - values["OO"],
        cell_effect=cell,
        interaction_effect=interaction,
        numerical_resolution=resolution,
        cell_parameters_differ=fingerprint(old.model.cells.state_dict())
        != fingerprint(new.model.cells.state_dict()),
        cell_effect_resolved=abs(cell) > 2 * resolution,
        interaction_resolved=abs(interaction) > 4 * resolution,
        scope="CALIBRATION_SELECTION_EXPLORATORY_NOT_CONFIRMATION",
    )


def execute(tr, data, directory, identity, durable):
    at = tr.step
    root = Path(directory) / "transactions" / ("proposal-%04d" % at)
    ident = {**identity, "proposal_at_step": at}
    tx = latest(root, ident)
    if tx is None:
        package = generate(tr, data, root / "candidate", ident, durable)
        tx = {"phase": "PROPOSED", "at_step": at, "package": package, "adapt_done": 0, "rows": []}
        save(root, ident, tx, 0)
    if tx["phase"] == "COMMITTED":
        tr.restore(tx["selected"])
        return tx["event"]
    if tx["phase"] == "PROPOSED":
        result = scan(tr, data, tx["package"], root / "scan", ident)
        tx.update(phase="TRUST_EVALUATED", scan=result)
        save(root, ident, tx, 1)
    package = tx["package"]
    tr.restore(package["official"])
    chosen = tx["scan"]["chosen_scale"]
    if tx["phase"] in ("TRUST_EVALUATED", "ADAPTING"):
        if chosen is None:
            selected = copy.deepcopy(tr.state())
            event = {
                "at_step": at,
                "chosen_scale": None,
                "trust_region_passed": False,
                "accepted": False,
                "adapt_steps": 0,
                "scan": tx["scan"],
            }
        else:
            old = tr.clone()
            new = tr.clone()
            connection = {
                k: package["original_connection"][k] + chosen * package["parameter_delta"][k]
                for k in package["original_connection"]
            }
            # Trial authorization is private to the transaction; not an official acceptance.
            new.authorize_connection(
                connection,
                transaction="trial-%s" % at,
                accepted=True,
                expected_previous=new.connection_sha,
            )
            if tx["adapt_done"]:
                old.restore(tx["old"])
                new.restore(tx["new"])
            for k in range(tx["adapt_done"], tr.c["training"]["adapt_steps"]):
                paired(old, new)
                a = observe_update(old, data)
                b = observe_update(new, data)
                paired(old, new)
                if a["inputs"] != b["inputs"]:
                    raise RuntimeError("PAIRED_INPUT_HASH_MISMATCH")
                row = dict(
                    step=old.step,
                    input_hashes=a["inputs"],
                    action_change=float((a["actions"] != b["actions"]).float().mean()),
                    path_change=float((a["actions"] != b["actions"]).any(-1).float().mean()),
                    old_cell_counts=a["cell_counts"],
                    new_cell_counts=b["cell_counts"],
                    old_gradients=a["gradients"],
                    new_gradients=b["gradients"],
                )
                tx["rows"].append(row)
                tx.update(
                    phase="ADAPTING",
                    adapt_done=k + 1,
                    old=copy.deepcopy(old.state()),
                    new=copy.deepcopy(new.state()),
                )
                save(root, ident, tx, k + 2)
            attribution = crossover(old, new, tr.c, data)
            accepted = (
                attribution["values"]["OO"] - attribution["values"]["NN"]
                >= tr.c["proposal"]["minimum_calibration_improvement"]
            )
            retained = new if accepted else old
            selected = copy.deepcopy(retained.state())
            if accepted:
                selected["r3i"]["connection_commits"][-1]["transaction"] = "accepted-%s" % at
            selected["cost"]["discarded_adaptation_updates"] += tr.c["training"]["adapt_steps"]
            selected["cost"]["discarded_adaptation_tokens"] += (
                tr.c["training"]["adapt_steps"] * tr.c["training"]["tokens_per_step"]
            )
            event = dict(
                at_step=at,
                chosen_scale=chosen,
                trust_region_passed=True,
                accepted=accepted,
                selected_branch="new" if accepted else "old",
                adapt_steps=tx["adapt_done"],
                training_path_rows=tx["rows"],
                attribution=attribution,
                scan=tx["scan"],
            )
        selected["proposals"].append(event)
        selected["policy_updates"] += tr.c["training"]["policy_steps"]
        selected["cost"]["policy_optimizer_updates"] += tr.c["training"]["policy_steps"]
        tx.update(phase="EVALUATED", selected=selected, event=event)
        save(root, ident, tx, 20)
    tx["phase"] = "COMMITTED"
    save(root, ident, tx, 21)
    tr.restore(tx["selected"])
    durable()
    return tx["event"]
