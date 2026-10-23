"""Multi-event immutable transactions. Resume each paired update without repeating it."""

import copy
from pathlib import Path
import torch
from minicells_r3f_v2.proposal import generate
from minicells_r3f_v2.scale_scan import scan
from minicells_r3f_v2.storage import save, latest
from minicells_r3h.evaluation import quality
from minicells_r3e.common import fingerprint, canonical
from minicells_r3h.sampler import probabilities
from minicells_r3f.stable_connection import hard_action, baseline_action


def observe_update(tr, data, counter=None):
    actions = []
    inputs = []
    kls = []
    raw_kls = []

    def before(module, args, kwargs):
        inputs.append(fingerprint(args[0]))
        kwargs["keep_trace"] = True
        return args, kwargs

    def after(module, args, out):
        actions.append(out.actions.detach().cpu())
        if counter is not None:
            with torch.no_grad():
                for r in out.trace:
                    values = []
                    raw = []
                    for m in (module, counter):
                        q, _, z = m.scores(r["state"], r["stage"], r["previous"])
                        h = hard_action(q, baseline_action(m, z), m.route_switch_margin)
                        values.append(
                            probabilities(q, h, m.routing_mode, tr.epsilon(), m.temperature, m.rho)
                        )
                        raw.append(q.double().softmax(-1))
                    kls.append(
                        float((values[0] * (values[0].log() - values[1].log())).sum(-1).mean())
                    )
                    raw_kls.append(float((raw[0] * (raw[0].log() - raw[1].log())).sum(-1).mean()))

    a = tr.model.register_forward_pre_hook(before, with_kwargs=True)
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
        actual_training_policy_kl=sum(kls) / len(kls) if kls else None,
        raw_policy_kl=sum(raw_kls) / len(raw_kls) if raw_kls else None,
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


@torch.no_grad()
def training_policy_diagnostics(old, new, c, data, epsilon):
    from minicells_r3f_v2.evaluation import admission_windows

    kl = []
    entropy_old = []
    entropy_new = []
    match_old = []
    match_new = []
    magnitudes = []
    for x in admission_windows(old, c, data):
        for r in old(x, keep_trace=True).trace:
            values = []
            for model in (old, new):
                q, _, z = model.scores(r["state"], r["stage"], r["previous"])
                h = hard_action(q, baseline_action(model, z), model.route_switch_margin)
                p = probabilities(q, h, model.routing_mode, epsilon, model.temperature, model.rho)
                if not torch.isfinite(p).all() or not torch.allclose(
                    p.sum(-1), torch.ones_like(p[:, 0])
                ):
                    raise RuntimeError("INVALID_ACTUAL_TRAINING_POLICY")
                values.append((p, h))
                magnitudes.append(float(q.abs().max()))
            po, ho = values[0]
            pn, hn = values[1]
            kl.append(float((po * (po.log() - pn.log())).sum(-1).mean()))
            entropy_old.append(float(-(po * po.log()).sum(-1).mean()))
            entropy_new.append(float(-(pn * pn.log()).sum(-1).mean()))
            match_old.append(float(po.gather(-1, ho[:, None]).mean()))
            match_new.append(float(pn.gather(-1, hn[:, None]).mean()))
    return dict(
        actual_training_policy_kl=sum(kl) / len(kl),
        entropy_old=sum(entropy_old) / len(entropy_old),
        entropy_new=sum(entropy_new) / len(entropy_new),
        hard_match_old=sum(match_old) / len(match_old),
        hard_match_new=sum(match_new) / len(match_new),
        raw_logit_abs_max=max(magnitudes),
        epsilon=epsilon,
        temperature=old.temperature,
        data_role="ADMISSION",
        additional_admission_gate=False,
    )


def execute(tr, data, directory, identity, durable):
    at = tr.step
    root = Path(directory) / "transactions" / ("proposal-%04d" % at)
    ident = {**identity, "proposal_at_step": at}
    tx = latest(root, ident)
    if tx is None:
        tr.model.active_step = tr.step
        tr.model.microbatch_index = 0
        try:
            package = generate(tr, data, root / "candidate", ident, durable)
        finally:
            tr.model.active_step = None
            tr.model.training_context = None
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
            policy_diagnostic = (
                training_policy_diagnostics(old.model, new.model, tr.c, data, tr.epsilon())
                if data is not None
                else {}
            )
            if tx["adapt_done"]:
                old.restore(tx["old"])
                new.restore(tx["new"])
            for k in range(tx["adapt_done"], tr.c["training"]["adapt_steps"]):
                paired(old, new)
                old_probe = copy.deepcopy(old.model)
                new_probe = copy.deepcopy(new.model)
                a = observe_update(old, data, new_probe)
                b = observe_update(new, data, old_probe)
                paired(old, new)
                if a["inputs"] != b["inputs"]:
                    raise RuntimeError("PAIRED_INPUT_HASH_MISMATCH")
                row = dict(
                    step=old.step,
                    input_hashes=a["inputs"],
                    action_change=float((a["actions"] != b["actions"]).float().mean()),
                    path_change=float((a["actions"] != b["actions"]).any(-1).float().mean()),
                    actual_training_policy_kl_old_states=a["actual_training_policy_kl"],
                    actual_training_policy_kl_new_states=b["actual_training_policy_kl"],
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
                actual_training_policy_diagnostic=policy_diagnostic,
                attribution=attribution,
                scan=tx["scan"],
            )
        event["candidate_cost"] = package["cost"] if "cost" in package else {}
        event["scan_cost"] = tx["scan"].get("work", {})
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
