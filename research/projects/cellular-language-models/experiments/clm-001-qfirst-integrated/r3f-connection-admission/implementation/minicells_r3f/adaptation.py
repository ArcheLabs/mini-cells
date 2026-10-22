"""Paired random numbers, explicit admission/adaptation/quality states, resumable evaluation commit."""

import copy
from pathlib import Path
import json
import os
import tempfile
import torch
from minicells_r3e.adaptation import event_store, save_event
from minicells_r3e.policy_proposal import add_work, difference
from .common import read, canonical, fingerprint
from .policy_proposal import optimize
from .evaluation import calibration


def paired(old, new):
    if not torch.equal(old.route_rng.get_state(), new.route_rng.get_state()):
        raise RuntimeError("ADAPTATION_ROUTE_STREAMS_DIFFER")
    if (
        canonical(old.data_rng.bit_generator.state) != canonical(new.data_rng.bit_generator.state)
        or old.data_cursor != new.data_cursor
    ):
        raise RuntimeError("ADAPTATION_DATA_STREAMS_DIFFER")


def trace(directory, at, public):
    root = Path(directory) / "proposal-traces" / ("step-%04d" % at)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "TRUST_REGION_TRACE.jsonl"
    content = "".join(
        json.dumps({"at_step": at, **row}, sort_keys=True, allow_nan=False) + "\n"
        for row in public["attempts"]
    )
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError("TRUST_REGION_TRACE_IMMUTABLE_CONFLICT")
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    os.replace(temporary, path)


def execute(tr, data, directory, identity):
    at = tr.step
    q = tr.c["training"]
    store = event_store(directory, identity, at)
    last = store.root / "latest.json"
    if last.exists():
        package = store.verified(read(last))
        if package["at_step"] != at:
            raise RuntimeError("PROPOSAL_TRANSACTION_STEP_CHANGED")
    else:
        candidate, info = optimize(tr, data)
        if bool(candidate is not None) != info["trust_region_passed"]:
            raise RuntimeError("ADMISSION_CANDIDATE_STATE_INCONSISTENT")
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
        trace(directory, at, tr.proposals[-1])
        return
    if package["phase"] != "EVALUATED":
        tr.restore(package["official"])
        info = package["info"]
        public = copy.deepcopy(
            {k: v for k, v in info.items() if k != "proposal_optimizer_final_state"}
        )
        public.update(
            experiment_identity=identity,
            at_step=at,
            trust_region_passed=info["trust_region_passed"],
            cell_adaptation_executed=False,
            final_quality_accepted=False,
            accepted=False,
            acceptance_data_role="CALIBRATION_SELECTION_NOT_VALIDATION",
            adapt_steps=0,
        )
        public.setdefault("work_by_phase", {})
        accepted = False
        if package["candidate"] is None:
            public["reason"] = info["rejection_reason"]
            selected = copy.deepcopy(tr.state())
        elif tr.arm == "A-T":
            trial = tr.clone()
            trial.model.load_state_dict(package["candidate"], strict=True)
            ow = copy.deepcopy(tr.model.work)
            nw = copy.deepcopy(trial.model.work)
            old_nll = calibration(tr.model, tr.c, data)
            new_nll = calibration(trial.model, tr.c, data)
            cal_work = difference(tr.model.work, ow)
            add_work(cal_work, difference(trial.model.work, nw))
            public["work_by_phase"]["calibration"] = cal_work
            add_work(public["work"], cal_work)
            tr.model.work = ow
            trial.model.work = nw
            accepted = old_nll - new_nll >= tr.c["proposal"]["minimum_calibration_improvement"]
            selected = copy.deepcopy(trial.state() if accepted else tr.state())
            public.update(
                old_calibration_nll=old_nll,
                new_calibration_nll=new_nll,
                reason="CURRENT_CELL_IMPROVED" if accepted else "CURRENT_CELL_GAIN_TOO_SMALL",
            )
        else:
            if not info["trust_region_passed"]:
                raise RuntimeError("ADAPTATION_WITHOUT_TRUST_ADMISSION")
            old = tr.clone()
            new = tr.clone()
            new.model.load_state_dict(package["candidate"], strict=True)
            if package["old"] is None:
                if fingerprint(old.model.cells.state_dict()) != fingerprint(
                    new.model.cells.state_dict()
                ):
                    raise RuntimeError("ADAPTATION_INITIAL_CELLS_DIFFER")
                if fingerprint(old.cell_opt.state_dict()) != fingerprint(new.cell_opt.state_dict()):
                    raise RuntimeError("ADAPTATION_INITIAL_OPTIMIZERS_DIFFER")
                # No reseeding: both branches inherit exactly the official route_rng stream.
            else:
                old.restore(package["old"])
                new.restore(package["new"])
            paired(old, new)
            for k in range(package["adapt_done"], q["adapt_steps"]):
                paired(old, new)
                old_cursor = old.data_cursor
                new_cursor = new.data_cursor
                old.update(data)
                new.update(data)
                paired(old, new)
                if (
                    old.data_cursor - old_cursor != new.data_cursor - new_cursor
                    or old.data_cursor - old_cursor != q["tokens_per_step"]
                ):
                    raise RuntimeError("ADAPTATION_TOKEN_COUNTS_DIFFER")
                if old.step != new.step:
                    raise RuntimeError("ADAPTATION_UPDATE_COUNTS_DIFFER")
                package.update(
                    phase="ADAPTING",
                    adapt_done=k + 1,
                    old=copy.deepcopy(old.state()),
                    new=copy.deepcopy(new.state()),
                )
                save_event(store, package, k + 1)
            ow = copy.deepcopy(old.model.work)
            nw = copy.deepcopy(new.model.work)
            old_nll = calibration(old.model, tr.c, data)
            new_nll = calibration(new.model, tr.c, data)
            cal_work = difference(old.model.work, ow)
            add_work(cal_work, difference(new.model.work, nw))
            public["work_by_phase"]["calibration"] = cal_work
            add_work(public["work"], cal_work)
            old.model.work = ow
            new.model.work = nw
            accepted = old_nll - new_nll >= tr.c["proposal"]["minimum_calibration_improvement"]
            retained, discarded = (new, old) if accepted else (old, new)
            selected = copy.deepcopy(retained.state())
            discarded_work = difference(discarded.model.work, tr.model.work)
            public["work_by_phase"]["discarded_adaptation"] = discarded_work
            add_work(public["work"], discarded_work)
            selected["cost"]["discarded_adaptation_updates"] += q["adapt_steps"]
            selected["cost"]["discarded_adaptation_tokens"] += (
                q["adapt_steps"] * q["tokens_per_step"]
            )
            public.update(
                old_calibration_nll=old_nll,
                new_calibration_nll=new_nll,
                adapt_steps=q["adapt_steps"],
                cell_adaptation_executed=True,
                discarded_adaptation_steps=q["adapt_steps"],
                reason="ADAPTED_CELL_IMPROVED" if accepted else "ADAPTED_CELL_GAIN_TOO_SMALL",
                identical_start_cells=True,
                identical_start_cell_optimizer=True,
                identical_train_order=True,
                paired_route_streams_verified=True,
                adaptation_main_tokens=q["adapt_steps"] * q["tokens_per_step"],
                discarded_adaptation_tokens=q["adapt_steps"] * q["tokens_per_step"],
            )
        public.update(accepted=accepted, final_quality_accepted=accepted)
        selected["connection_version"] = tr.connection_version + int(accepted)
        selected["policy_updates"] = tr.policy_updates + info["policy_optimizer_updates"]
        selected["proposals"].append(public)
        selected["cost"]["policy_optimizer_updates"] += info["policy_optimizer_updates"]
        add_work(selected["cost"]["work"], public["work"])
        selected["pending"] = None
        package = {
            "at_step": at,
            "phase": "EVALUATED",
            "selected": selected,
            "adapt_done": package["adapt_done"],
        }
        save_event(store, package, q["adapt_steps"] + 1)
    # An evaluated transaction resumes here: no adaptation or Calibration re-execution.
    package["phase"] = "COMMITTED"
    save_event(store, package, q["adapt_steps"] + 2)
    tr.restore(package["selected"])
    trace(directory, at, tr.proposals[-1])
