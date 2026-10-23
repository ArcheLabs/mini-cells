"""Transactional paired adaptation. A proposal is applied exactly once, including optimizer/RNG."""

import copy
from types import SimpleNamespace
from pathlib import Path
from minicells_r3c_moe_bridge.checkpoint import Store
from .common import atomic, read, canonical, fingerprint
from .policy_proposal import optimize, add_work, difference
from .evaluation import calibration


def event_store(directory, identity, step):
    return Store(
        Path(directory) / "transactions" / ("proposal-%04d" % step),
        {**identity, "proposal_at_step": step},
    )


def save_event(store, state, sequence):
    payload = SimpleNamespace(step=sequence, state=lambda: state)
    row = store.save(payload)
    atomic(store.root / "latest.json", row)


def execute(tr, data, directory, identity):
    at = tr.step
    store = event_store(directory, identity, at)
    last = store.root / "latest.json"
    if last.exists():
        package = store.verified(read(last))
        if package["at_step"] != at:
            raise RuntimeError("PROPOSAL_TRANSACTION_STEP_CHANGED")
    else:
        candidate, info = optimize(tr, data)
        # Cells never change during scoring/proposal optimization.
        package = {
            "at_step": at,
            "phase": "PROPOSED",
            "official": copy.deepcopy(tr.state()),
            "candidate": None if candidate is None else copy.deepcopy(candidate.state_dict()),
            "info": info,
            "adapt_done": 0,
            "old": None,
            "new": None,
        }
        save_event(store, package, 0)
    if package["phase"] == "COMMITTED":
        tr.restore(package["selected"])
        return
    tr.restore(package["official"])
    info = package["info"]
    q = tr.c["training"]
    accepted = False
    # Optimizer tensors stay private, public event contains only a hash and provenance.
    public = {k: v for k, v in info.items() if k != "proposal_optimizer_final_state"}
    public.update(
        at_step=at, accepted=False, acceptance_data_role="CALIBRATION_SELECTION_NOT_VALIDATION"
    )
    if package["candidate"] is None:
        public["reason"] = "ALL_TRUST_REGION_SCALES_REJECTED"
        selected = copy.deepcopy(tr.state())
    elif tr.arm == "A-T":
        trial = tr.clone()
        trial.model.load_state_dict(package["candidate"], strict=True)
        old_work = copy.deepcopy(tr.model.work)
        new_work = copy.deepcopy(trial.model.work)
        old_nll = calibration(tr.model, tr.c, data)
        new_nll = calibration(trial.model, tr.c, data)
        add_work(public["work"], difference(tr.model.work, old_work))
        add_work(public["work"], difference(trial.model.work, new_work))
        tr.model.work = old_work
        trial.model.work = new_work
        accepted = old_nll - new_nll >= tr.c["proposal"]["minimum_calibration_improvement"]
        selected = copy.deepcopy(trial.state() if accepted else tr.state())
        public.update(
            old_calibration_nll=old_nll,
            new_calibration_nll=new_nll,
            adapt_steps=0,
            reason="CURRENT_CELL_IMPROVED" if accepted else "CURRENT_CELL_GAIN_TOO_SMALL",
        )
    else:
        old = tr.clone()
        new = tr.clone()
        new.model.load_state_dict(package["candidate"], strict=True)
        if package["old"] is not None:
            old.restore(package["old"])
            new.restore(package["new"])
        else:
            # Independent deterministic route streams; identical data and Cell AdamW snapshots.
            old.route_rng.manual_seed(tr.model.seed + at * 101 + 71001)
            new.route_rng.manual_seed(tr.model.seed + at * 101 + 72001)
        if fingerprint(old.data_rng.bit_generator.state) != fingerprint(
            new.data_rng.bit_generator.state
        ):
            raise RuntimeError("ADAPTATION_DATA_STREAMS_DIFFER")
        for k in range(package["adapt_done"], q["adapt_steps"]):
            old.update(data)
            new.update(data)
            if old.data_cursor != new.data_cursor or canonical(
                old.data_rng.bit_generator.state
            ) != canonical(new.data_rng.bit_generator.state):
                raise RuntimeError("ADAPTATION_DATA_CURSOR_DIFFER")
            package.update(
                phase="ADAPTING",
                adapt_done=k + 1,
                old=copy.deepcopy(old.state()),
                new=copy.deepcopy(new.state()),
            )
            save_event(store, package, k + 1)
        old_work = copy.deepcopy(old.model.work)
        new_work = copy.deepcopy(new.model.work)
        old_nll = calibration(old.model, tr.c, data)
        new_nll = calibration(new.model, tr.c, data)
        add_work(public["work"], difference(old.model.work, old_work))
        add_work(public["work"], difference(new.model.work, new_work))
        old.model.work = old_work
        new.model.work = new_work
        accepted = old_nll - new_nll >= tr.c["proposal"]["minimum_calibration_improvement"]
        retained, discarded = (new, old) if accepted else (old, new)
        selected = copy.deepcopy(retained.state())
        extra = difference(discarded.model.work, tr.model.work)
        add_work(public["work"], extra)
        public.update(
            old_calibration_nll=old_nll,
            new_calibration_nll=new_nll,
            adapt_steps=q["adapt_steps"],
            discarded_adaptation_steps=q["adapt_steps"],
            reason="ADAPTED_CELL_IMPROVED" if accepted else "ADAPTED_CELL_GAIN_TOO_SMALL",
            identical_start_cells=True,
            identical_start_cell_optimizer=True,
            identical_train_order=True,
        )
        selected["cost"]["discarded_adaptation_updates"] += q["adapt_steps"]
        selected["cost"]["discarded_adaptation_tokens"] += q["adapt_steps"] * q["tokens_per_step"]
    public["accepted"] = accepted
    selected["connection_version"] = tr.connection_version + int(accepted)
    selected["policy_updates"] = tr.policy_updates + q["policy_steps"]
    selected["proposals"].append(public)
    selected["cost"]["policy_optimizer_updates"] += q["policy_steps"]
    add_work(selected["cost"]["work"], public["work"])
    selected["pending"] = None
    package = {
        "at_step": at,
        "phase": "COMMITTED",
        "selected": selected,
        "adapt_done": package["adapt_done"],
    }
    save_event(store, package, q["adapt_steps"] + 1)
    tr.restore(selected)
