"""Reuse the frozen Q-first optimizer and 22-scale Trust Region on already-observed Train only."""

import copy
import inspect
import torch
from minicells_r3e.common import fingerprint
from minicells_r3e.policy_proposal import collect as original_collect
from minicells_r3f_v2.proposal import generate as original_generate
from minicells_r3f_v2.scale_scan import scan as original_scan
from minicells_r3k.reuse import bind
from minicells_r3l.storage import recover_latest
from minicells_r3j.storage import save
from minicells_r3e.storage import snapshot_lock


def locked_save(root, *args, **kwargs):
    private = next((p for p in root.parents if p.name == "private-checkpoints"), root)
    with snapshot_lock(private):
        return save(root, *args, **kwargs)


def propose(tr, seen, root, identity, persist):
    if not seen:
        raise RuntimeError("PROPOSAL_REQUIRES_ALREADY_OBSERVED_TRAIN")
    device = next(tr.model.parameters()).device
    batches = [(x.to(device), y.to(device)) for x, y in seen[-1]]
    # No peek at the next training sample: the old collector is rebound to a stored past batch.
    collect = torch.no_grad()(
        bind(inspect.unwrap(original_collect), train_batches=lambda *args: batches)
    )
    generate = bind(original_generate, collect=collect, latest=recover_latest, save=locked_save)
    package = generate(tr, None, root / "candidate", identity, persist)
    refs = [x.to(device) for pair in seen for x, y in pair]
    scan = bind(
        inspect.unwrap(original_scan),
        admission_windows=lambda *args: iter(
            refs[: tr.c["proposal"]["reference_admission_windows"]]
        ),
        latest=recover_latest,
        save=locked_save,
    )
    with torch.no_grad():
        trace = scan(tr, None, package, root / "scan", identity)
    trace.update(
        data_role="ALREADY_OBSERVED_TRAIN",
        data_range=None,
        input_hashes=[fingerprint(x) for x in refs],
        proposal_candidate=package["candidate_fingerprint"],
    )
    scale = trace["chosen_scale"]
    connection = copy.deepcopy(package["original_connection"])
    if scale is not None:
        for key in connection:
            connection[key] = connection[key] + scale * package["parameter_delta"][key]
    chosen = next((r for r in trace["attempts"] if r["scale"] == scale), None)
    trace["effective"] = bool(chosen and chosen["passed"] and chosen["effective_update"])
    trace["action_change"] = chosen["action_change"] if chosen else 0.0
    trace["parameter_sha256"] = fingerprint(connection)
    return connection, trace
