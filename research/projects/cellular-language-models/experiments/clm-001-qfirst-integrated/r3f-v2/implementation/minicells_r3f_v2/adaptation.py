"""Reuse the R3f transaction and paired RNG rules; observe only the ten real updates."""

import copy
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from minicells_r3e.training import Trainer
from minicells_r3f import adaptation as existing
from minicells_r3e.policy_proposal import add_work
from .storage import latest, save
from .common import fingerprint, read


class RouteObserver:
    def __init__(self, old_connection, new_connection, directory):
        self.old_connection = old_connection
        self.new_connection = new_connection
        self.directory = Path(directory)
        self.pending = None
        self.rows = []

    def update(self, original, tr, data, batches=None):
        step = tr.step + 1
        captured = []
        inputs = []
        probe = copy.deepcopy(tr.model)
        old_hash = fingerprint(tr.model.connection.state_dict())
        old_branch = old_hash == fingerprint(self.old_connection)
        # Identical Connection hashes cannot arise after nonempty admission.
        probe.connection.load_state_dict(self.new_connection if old_branch else self.old_connection)
        work_before = copy.deepcopy(probe.work)
        model_work_before = copy.deepcopy(tr.model.work)

        def before(module, args, kwargs):
            kwargs["keep_trace"] = True
            inputs.append(fingerprint(args[0]))
            return args, kwargs

        def after(module, args, output):
            with torch.no_grad():
                kls = []
                for r in output.trace:
                    a = module.scores(r["state"], r["stage"], r["previous"])[0].double().softmax(-1)
                    b = probe.scores(r["state"], r["stage"], r["previous"])[0].double().softmax(-1)
                    eps = tr.epsilon()
                    a = (1 - eps) * a + eps / a.shape[-1]
                    b = (1 - eps) * b + eps / b.shape[-1]
                    kls.append((a * (a.log() - b.log())).sum(-1))
                captured.append(
                    {
                        "actions": output.actions.detach().cpu(),
                        "cell_counts": output.operator_rows.detach().cpu(),
                        "training_kl": float(torch.cat(kls).mean()),
                    }
                )

        pre = tr.model.register_forward_pre_hook(before, with_kwargs=True)
        post = tr.model.register_forward_hook(after)
        try:
            row = original(tr, data, batches)
        finally:
            pre.remove()
            post.remove()
        item = {
            "step": step,
            "branch": "old" if old_branch else "new",
            "inputs": inputs,
            "batches": captured,
            "gradients": row["cell_gradient_norms"],
            "training_work": {k: v - model_work_before.get(k, 0) for k, v in tr.model.work.items()},
            "probe_work": {k: v - work_before.get(k, 0) for k, v in probe.work.items()},
        }
        if old_branch:
            if self.pending is not None:
                raise RuntimeError("TRAINING_OBSERVATION_PAIR_ORDER")
            self.pending = item
        else:
            old = self.pending
            if old is None or old["step"] != step or old["inputs"] != inputs:
                raise RuntimeError("TRAINING_OBSERVATION_INPUTS_DIFFER")
            a = torch.cat([r["actions"] for r in old["batches"]])
            b = torch.cat([r["actions"] for r in captured])
            record = {
                "step": step,
                "input_hashes": inputs,
                "action_change": float((a != b).float().mean()),
                "path_change": float((a != b).any(-1).float().mean()),
                "old_cell_counts": torch.stack([r["cell_counts"] for r in old["batches"]])
                .sum(0)
                .tolist(),
                "new_cell_counts": torch.stack([r["cell_counts"] for r in captured])
                .sum(0)
                .tolist(),
                "old_cell_gradient_norms": old["gradients"],
                "new_cell_gradient_norms": item["gradients"],
                "old_state_training_mixture_kl": sum(r["training_kl"] for r in old["batches"])
                / len(old["batches"]),
                "new_state_training_mixture_kl": sum(r["training_kl"] for r in captured)
                / len(captured),
                "old_training_work": old["training_work"],
                "new_training_work": item["training_work"],
                "observation_probe_work": old["probe_work"],
            }
            add_work(record["observation_probe_work"], item["probe_work"])
            self.rows.append(record)
            # Step filenames are immutable; a resumed half-step is not a committed adaptation pair.
            self.pending = None
        return row


def execute(tr, data, package, scan, directory, identity):
    tr.restore(package["official"])
    chosen = None
    if scan["chosen_scale"] is not None:
        chosen = copy.deepcopy(tr.model)
        with torch.no_grad():
            for name, param in chosen.connection.named_parameters():
                param.copy_(
                    package["original_connection"][name].to(param)
                    + scan["chosen_scale"] * package["parameter_delta"][name].to(param)
                )
        chosen.requires_grad_(False)
        chosen.cells.requires_grad_(True)
    info = {
        "attempts": scan["attempts"],
        "chosen_scale": scan["chosen_scale"],
        "trust_region_passed": chosen is not None,
        "rejection_reason": scan["rejection_reason"],
        "work": copy.deepcopy(package["cost"]["total_work"]),
        "work_by_phase": {
            "counterfactual": package["cost"]["counterfactual_work"],
            "connection_optimization": package["cost"]["optimization_work"],
            "trust_scales": scan["work"],
        },
        "policy_optimizer_updates": len(package["losses"]),
        "proposal_optimizer_final_state": None,
        "route_switch_margin": package["route_switch_margin"],
        "numeric_tolerance": package["numeric_tolerance"],
        "snapshot_fingerprint": fingerprint(package["official"]["model"]),
    }
    add_work(info["work"], scan["work"])
    observer = RouteObserver(
        package["original_connection"],
        chosen.connection.state_dict() if chosen is not None else package["original_connection"],
        directory,
    )
    original_update = Trainer.update
    original_save = existing.save_event
    original_calibration = existing.calibration
    calibration_seconds = [0.0]

    def timed_calibration(*args):
        start = time.perf_counter()
        value = original_calibration(*args)
        if next(args[0].parameters()).is_cuda:
            torch.cuda.synchronize()
        calibration_seconds[0] += time.perf_counter() - start
        return value

    monitor_root = Path(directory) / "training-observations"
    prior = latest(monitor_root, identity)
    observer.rows = [] if prior is None else prior["rows"]

    def checkpoint(store, state, sequence):
        if state.get("phase") == "ADAPTING":
            k = state["adapt_done"]
            if len(observer.rows) != k:
                raise RuntimeError("TRAINING_ROUTE_OBSERVATION_INCOMPLETE")
            # Persist observation with the transaction, so restart cannot detach its evidence.
            state["training_route_rows"] = copy.deepcopy(observer.rows)
        original_save(store, state, sequence)
        if state.get("phase") == "ADAPTING":
            save(monitor_root, identity, {"rows": observer.rows}, state["adapt_done"])

    # Reconcile a crash after transaction save but before its observation mirror.
    tx = Path(directory) / "transactions/proposal-0100/latest.json"
    if tx.exists():
        from minicells_r3c_moe_bridge.checkpoint import Store

        row = read(tx)
        state = Store(tx.parent, {**identity, "proposal_at_step": 100}).verified(row)
        if "training_route_rows" in state:
            observer.rows = state["training_route_rows"]
    with ExitStack() as stack:
        stack.enter_context(patch.object(existing, "optimize", return_value=(chosen, info)))
        stack.enter_context(
            patch.object(
                Trainer,
                "update",
                lambda t, d, batches=None: observer.update(original_update, t, d, batches),
            )
        )
        stack.enter_context(patch.object(existing, "save_event", checkpoint))
        stack.enter_context(patch.object(existing, "calibration", timed_calibration))
        existing.execute(tr, data, directory, identity)
    event = copy.deepcopy(tr.proposals[-1])
    if chosen is not None and event["adapt_steps"] != tr.c["training"]["adapt_steps"]:
        raise RuntimeError("ADAPTATION_BUDGET_INCOMPLETE")
    event["quality_status"] = (
        "FINAL_QUALITY_ACCEPTED"
        if event["final_quality_accepted"]
        else "QUALITY_REJECTED_AFTER_ADAPTATION"
        if event["cell_adaptation_executed"]
        else "NOT_TESTED"
    )
    event["calibration_seconds_this_invocation"] = calibration_seconds[0]
    # Final deterministic path difference includes the actual adapted Cell states.
    if chosen is not None:
        from minicells_r3c_moe_bridge.checkpoint import Store
        from .evaluation import admission_windows
        from minicells_r3f.evaluation import trust

        txroot = Path(directory) / "transactions/proposal-0100"
        store = Store(txroot, {**identity, "proposal_at_step": 100})
        checkpoint_path = store.receipt_file(tr.c["training"]["adapt_steps"])
        if checkpoint_path.exists():
            paired_state = store.verified(read(checkpoint_path))
            old_model = copy.deepcopy(tr.model)
            new_model = copy.deepcopy(tr.model)
            old_model.load_state_dict(paired_state["old"]["model"], strict=True)
            new_model.load_state_dict(paired_state["new"]["model"], strict=True)
            event["final_deterministic_admission_difference"] = trust(
                old_model, new_model, list(admission_windows(old_model, tr.c, data))
            )
    return event, observer.rows
