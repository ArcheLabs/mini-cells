"""Four paired Pilot branches with resumable ten-update transactions inside the main timeline."""

import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from minicells_r3e.common import atomic, fingerprint
from minicells_r3j.storage import save
from minicells_r3l.storage import recover_latest
from .training import advance, initial_aux, restore, snapshot, change_connection
from .intervention import propose
from .data import batches
from .pressure import Detector
from minicells_r3e.storage import snapshot_lock


def train_quality(model, seen):
    losses = []
    model.training_context = None
    with torch.no_grad():
        for pair in seen:
            for x, y in pair:
                x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                out = model(x).logits
                losses.append(float(F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))))
    return sum(losses) / len(losses)


def gradient_sketch(tr):
    return torch.cat(
        [
            (
                p.grad.detach().reshape(-1)[:32].cpu()
                if p.grad is not None
                else torch.zeros(min(32, p.numel()))
            )
            for p in tr.model.cells.parameters()
        ]
    ).tolist()


def run_arm(
    c,
    runtime,
    anchor,
    ident,
    root,
    historical,
    research,
    environment,
    arm,
    device,
    persist=lambda: None,
):
    root = Path(root)
    saved = recover_latest(root, ident)
    tr = restore(runtime, saved["trainer"] if saved else anchor.state(), ident["seed"], device)
    aux = copy.deepcopy(saved["aux"]) if saved else initial_aux(tr, c, ident)
    detector = Detector(c["pressure"], aux["pressure"])

    def checkpoint():
        aux["pressure"] = copy.deepcopy(detector.state)
        aux["curriculum_step"] = tr.step - c["anchor_step"]
        state = snapshot(tr, aux)
        private = next((p for p in root.parents if p.name == "private-checkpoints"), root)
        with snapshot_lock(private):
            save(root, ident, state, tr.step - c["anchor_step"])
        print("[R3M_SNAPSHOT]", ident["seed"], environment, arm, tr.step, flush=True)

    if not saved:
        checkpoint()
    while tr.step < c["anchor_step"] + c["continuation_steps"]:
        rel = tr.step - c["anchor_step"]
        if arm in ("P1", "P2", "P0", "P3"):
            event = rel == c["proposal_at"] and not aux["transactions"]
        elif arm in ("S0", "S1", "S3"):
            event = rel in (100, 300) and len(aux["transactions"]) < (1 if rel == 100 else 2)
        elif arm == "S4":
            times = ident["random_schedule"]
            event = rel in times and rel not in [r["at"] for r in aux["transactions"]]
        elif arm == "S2":
            event = bool(aux.get("pending_pressure_trigger"))
        else:
            event = False
        fill_ghost = arm == "S2" and (
            (rel == 380 and len(aux["transactions"]) == 0)
            or (rel == 390 and len(aux["transactions"]) == 1)
        )
        event = event or fill_ghost
        if event and aux["transaction"] is None:
            official = copy.deepcopy(tr.state())
            connection, trace = propose(
                tr,
                aux["seen"],
                root / "transactions" / str(rel),
                dict(ident, proposal_at=rel),
                persist,
            )
            official = copy.deepcopy(
                tr.state()
            )  # Preserve the advanced Proposal CF RNG in BOTH branches.
            candidate = restore(runtime, official, ident["seed"], device, connection)
            ghost = arm in ("P0", "P2", "P3", "S0", "S3") or fill_ghost or not trace["effective"]
            if ghost:
                change_connection(candidate, tr.model.connection.state_dict())
            aux["transaction"] = dict(
                at=rel,
                old=official,
                new=copy.deepcopy(candidate.state()),
                done=0,
                trace=trace,
                ghost=ghost,
                new_curve=[],
                new_pressure_events=[],
                new_pressure=copy.deepcopy(detector.state),
            )
            aux["proposal_optimizer_updates"] += runtime["training"]["policy_steps"]
            aux["pending_pressure_trigger"] = False
        txn = aux["transaction"]
        train = batches(tr, historical, research, environment, c)
        aux["seen"] = (aux["seen"] + [[(x.detach().cpu(), y.detach().cpu()) for x, y in train]])[
            -10:
        ]
        perturb = arm in ("P3", "S3") and rel < c["perturbation"]["phase_end"]
        if txn:
            candidate = restore(runtime, txn["new"], ident["seed"], device)
            # Curriculum generation consumes the same data RNG as the official stream.
            candidate.data_rng.bit_generator.state = copy.deepcopy(tr.data_rng.bit_generator.state)
            new_row = advance(candidate, train, c, aux["schedule"], perturbed=perturb)
            txn["new_curve"].append(new_row)
            new_detector = Detector(c["pressure"], txn["new_pressure"])
            new_event = new_detector.observe(new_row["ce"], gradient_sketch(candidate), rel + 1)
            txn["new_pressure"] = copy.deepcopy(new_detector.state)
            txn["new_pressure_events"].append(dict(step=rel + 1, **new_event))
            aux["shadow_updates"] += 1
        row = advance(tr, train, c, aux["schedule"], perturbed=perturb)
        aux["main_updates"] += 1
        aux["curve"].append(row)
        # Gradient norms are collected by the existing trainer; signed interference is
        # recorded separately by a parameter-gradient sketch, never task-labelled.
        gradient = gradient_sketch(tr)
        pressure = detector.observe(row["ce"], gradient, rel + 1)
        aux.setdefault("pressure_events", []).append(dict(step=rel + 1, **pressure))
        if arm == "S2" and pressure["triggered"] and rel + 1 <= 390:
            aux["pending_pressure_trigger"] = True
        if txn:
            txn["done"] += 1
            txn["old"] = copy.deepcopy(tr.state())
            txn["new"] = copy.deepcopy(candidate.state())
            if txn["done"] == c["shadow_steps"]:
                old_q = train_quality(tr.model, aux["seen"])
                new_q = train_quality(candidate.model, aux["seen"])
                accepted = bool(
                    not txn["ghost"]
                    and txn["trace"]["effective"]
                    and old_q - new_q >= runtime["proposal"]["minimum_calibration_improvement"]
                )
                if accepted:
                    tr = candidate
                    aux["curve"][-c["shadow_steps"] :] = txn["new_curve"]
                    aux["pressure_events"][-c["shadow_steps"] :] = txn["new_pressure_events"]
                    detector = Detector(c["pressure"], copy.deepcopy(txn["new_pressure"]))
                aux["discarded_shadow_updates"] += c["shadow_steps"]
                aux["transactions"].append(
                    dict(
                        at=txn["at"],
                        accepted=accepted,
                        old_train_nll=old_q,
                        new_train_nll=new_q,
                        trace=txn["trace"],
                        ghost=txn["ghost"],
                        candidate_optimizer_sha=fingerprint(candidate.cell_opt.state_dict()),
                        accepted_state="CANDIDATE_CELL_OPTIMIZER_AND_STREAM"
                        if accepted
                        else "OLD_CELL_OPTIMIZER_AND_STREAM",
                    )
                )
                aux["transaction"] = None
            else:
                aux["transaction"] = txn
        if (tr.step - c["anchor_step"]) in c["barriers"] or txn or tr.step % 50 == 0:
            checkpoint()
            persist()
        if tr.step % 25 == 0:
            print(
                dict(
                    seed=ident["seed"], environment=environment, arm=arm, step=tr.step, ce=row["ce"]
                ),
                flush=True,
            )
    if (
        tr.step != c["anchor_step"] + c["continuation_steps"]
        or aux["main_updates"] != c["continuation_steps"]
        or tr.data_cursor != (c["anchor_step"] + c["continuation_steps"]) * c["tokens_per_step"]
    ):
        raise RuntimeError("MAIN_TRAINING_BUDGET_CHANGED")
    checkpoint()
    atomic(
        root / "COMPLETE.json",
        dict(
            identity=ident,
            step=tr.step,
            logical_tokens=c["continuation_steps"] * c["tokens_per_step"],
            optimizer_sha256=fingerprint(tr.cell_opt.state_dict()),
        ),
    )
    return tr, aux
