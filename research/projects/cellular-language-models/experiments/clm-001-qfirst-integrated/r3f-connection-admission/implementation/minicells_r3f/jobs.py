"""Train-only and terminal-only adapters reuse R3e optimizers, costs, parity and state."""

from pathlib import Path
import copy
import json
import time
import torch
from minicells_r3e.runner import parity, cost_estimate
from minicells_r3e.common import seed_all
from minicells_r3e.evaluation import paths, sync
from minicells_r3e.training import Trainer
from .common import atomic, read, identity
from .stable_connection import build
from .adaptation import execute
from .evaluation import reference
from .storage import Store, save, cleanup_transactions


def train_job(c, data, output, source, seed, arm, parent, device, target=None):
    target = c["training"]["steps"] if target is None else target
    seed_all(seed)
    directory = Path(output) / "private-checkpoints/jobs" / ("seed-%s-arm-%s" % (seed, arm))
    ident = identity(c, source, seed, arm)
    store = Store(directory, ident)
    model = build(c, parent, seed, arm, device, data)
    refs = reference(model, c, data)
    initial = paths(model, refs)
    parity(model, parent, refs[0])
    model.work = {k: 0 for k in model.work}
    tr = Trainer(model, c, arm)
    rows = [read(p) for p in (directory / "receipts").glob("*.json")]
    latest = max(rows, key=lambda r: r["step"]) if rows else None
    if latest:
        tr.restore(store.verified(latest))
        summary_file = directory / "result.json"
        if tr.step >= target:
            if summary_file.exists():
                result = read(summary_file)
                if result["identity"] != ident or result["checkpoint"] != latest:
                    raise RuntimeError("RESULT_IDENTITY_CHANGED")
                return result
            if target < c["training"]["steps"]:
                return {
                    "identity": ident,
                    "checkpoint": latest,
                    "proposals": copy.deepcopy(tr.proposals),
                    "status": "RESTORED_PRIOR_PROGRESS",
                }
    else:
        latest = save(tr, store)
    sync(device)
    begin = time.perf_counter()
    print("[R3f]", seed, arm, "restored", tr.step, "/", c["training"]["steps"], flush=True)
    while tr.step < target:
        due = tr.step in c["training"]["proposal_steps"] and arm in ("A-T", "A-C", "B-C")
        if due and not any(p["at_step"] == tr.step for p in tr.proposals):
            execute(tr, data, directory, ident)
            # The retained branch's Cell/optimizer/RNG is now one coherent official state.
            latest = save(tr, store) if tr.step != latest["step"] else latest
            if tr.step != latest["step"]:
                raise RuntimeError("ADAPTATION_CHECKPOINT_COMMIT_FAILED")
            # Once official state has a new step, obsolete private shadow tensors need not persist.
            if tr.step > tr.proposals[-1]["at_step"]:
                cleanup_transactions(directory, tr.step)
        else:
            row = tr.update(data)
            if tr.step % c["training"]["print_every"] == 0:
                print(
                    json.dumps({"seed": seed, "arm": arm, "step": tr.step, "ce": row["ce"]}),
                    flush=True,
                )
        if tr.step % c["training"]["checkpoint_every"] == 0:
            # A rejected proposal at the same Cell step is recovered from its transaction until next update.
            if tr.step != latest["step"]:
                latest = save(tr, store)
            # Main state now includes completed proposal decisions; old branch tensors are redundant.
            cleanup_transactions(directory, tr.step)
            print("[R3F_SNAPSHOT]", seed, arm, tr.step, flush=True)
    latest = save(tr, store)
    tr.verify_frozen()
    sync(device)
    wall = time.perf_counter() - begin
    if tr.step != target or tr.data_cursor != tr.step * c["training"]["tokens_per_step"]:
        raise RuntimeError("MAIN_BUDGET_INVALID")
    main_work = copy.deepcopy(model.work)
    with torch.no_grad():
        final_paths = paths(model, refs)
        drift = float(
            torch.cat([(a != b).any(-1).float() for a, b in zip(initial, final_paths)]).mean()
        )
    if arm in ("A-F", "B-F", "B-R") and any(
        torch.count_nonzero(p) for p in model.connection.parameters()
    ):
        raise RuntimeError("FROZEN_CONNECTION_CHANGED")
    cost = copy.deepcopy(tr.cost)
    cost["main_work"] = main_work
    main_flops = cost_estimate(main_work, c, model.graph)
    extra_flops = cost_estimate(cost["work"], c, model.graph)
    cost.update(
        main_forward_matmul_estimate=main_flops,
        additional_forward_matmul_estimate=extra_flops,
        forward_backward_upper_proxy=3 * (main_flops + extra_flops),
        wall_training_seconds=wall,
        wall_time_scope="CURRENT_JOB_INVOCATION; prior committed update time retained in elapsed_update_seconds",
        estimation_scope="Analytic executed linear and dense QK/AV; 3x backward proxy is not measured hardware FLOPs",
        proposal_optimizer_updates=tr.policy_updates,
        proposals=len(tr.proposals),
        total_cell_updates_including_discarded=tr.step + cost["discarded_adaptation_updates"],
        same_total_compute=False,
    )
    g = torch.tensor([r["cell_gradient_norms"] for r in tr.curve])
    coverage = {
        "zero_gradient_fraction_by_cell": (g == 0).float().mean(0).tolist(),
        "mean_gradient_norm_by_cell": g.mean(0).tolist(),
        "cell_calls": torch.tensor([r["cell_counts"] for r in tr.curve]).sum(0).tolist(),
    }
    proposals = copy.deepcopy(tr.proposals)
    # Training gradients and optimizer are released before isolated terminal memory measurement.
    del tr
    for p in model.parameters():
        p.grad = None
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    result = {
        "status": "TRAINING_PHASE_COMPLETE"
        if target == c["training"]["steps"]
        else "TRIAL_PHASE_COMPLETE",
        "identity": ident,
        "checkpoint": latest,
        "training_cost": cost,
        "proposals": proposals,
        "gradient_coverage": coverage,
        "fixed_weight_route_drift": drift,
        "margin": model.margin_evidence,
        "confirmation": "SEALED",
    }
    atomic(
        directory / ("progress.json" if target < c["training"]["steps"] else "result.json"),
        result,
    )
    return result
