"""Paired branches come from ADAPTING(10), never copied from the selected branch."""

import torch
from .common import fingerprint


def validate(c, arm, official, old, new, selected, proposal, pair, initial_pair, committed):
    if pair["phase"] != "ADAPTING" or pair["adapt_done"] != c["adapt_steps"]:
        raise RuntimeError("PAIRED_BRANCH_STATE_NOT_RETAINED")
    if (
        any(s["step"] != c["end_step"] for s in (old, new, selected))
        or official["step"] != c["start_step"]
    ):
        raise RuntimeError("PARENT_STEP_MISMATCH")
    start = initial_pair["official"]
    if fingerprint(start) != fingerprint(official):
        raise RuntimeError("ADAPTATION_START_STATE_MISMATCH")
    if initial_pair.get("old") is not None or initial_pair.get("new") is not None:
        raise RuntimeError("ADAPTATION_INITIAL_BRANCH_STATE_INVALID")
    for key in ("data_rng", "route_rng", "data_cursor"):
        if fingerprint(old[key]) != fingerprint(new[key]):
            raise RuntimeError("PAIRED_DATA_OR_ROUTE_STREAMS_DIFFER")
    tokens = proposal["official"]["data_cursor"]
    if any(
        s["data_cursor"]
        != tokens + c["adapt_steps"] * c["parent_runtime"]["training"]["tokens_per_step"]
        for s in (old, new)
    ):
        raise RuntimeError("PAIRED_TOKEN_BUDGET_MISMATCH")
    keys = set(official["model"])
    if any(set(s["model"]) != keys for s in (old, new, selected)):
        raise RuntimeError("PARENT_MODEL_KEYS_MISMATCH")
    frozen = [k for k in keys if not k.startswith(("cells.", "connection."))]
    differences = [
        k
        for k in frozen
        if any(not torch.equal(official["model"][k], s["model"][k]) for s in (old, new, selected))
    ]
    if differences:
        raise RuntimeError("FROZEN_COMPUTATION_WEIGHTS_DIFFER:" + ",".join(differences))
    candidate_model = initial_pair.get("candidate")
    if candidate_model is None or any(
        not torch.equal(candidate_model[k], official["model"][k])
        for k in official["model"]
        if not k.startswith("connection.")
    ):
        raise RuntimeError("ADAPTATION_INITIAL_CELLS_OR_FROZEN_WEIGHTS_DIFFER")
    original = proposal["original_connection"]
    candidate = proposal["candidate_connection"]
    chosen = c["chosen_scale"]
    expected_new = {
        k: (original[k] + chosen * proposal["parameter_delta"][k])
        if original[k].is_floating_point()
        else original[k]
        for k in original
    }
    for key in original:
        if not torch.equal(candidate[key] - original[key], proposal["parameter_delta"][key]):
            raise RuntimeError("PROPOSAL_DELTA_MISMATCH")
        if not torch.equal(official["model"]["connection." + key], original[key]):
            raise RuntimeError("INITIAL_CONNECTION_MISMATCH")
        if not torch.equal(candidate_model["connection." + key], expected_new[key]):
            raise RuntimeError("INITIAL_NEW_CONNECTION_MISMATCH")
        if not torch.equal(old["model"]["connection." + key], original[key]):
            raise RuntimeError("OLD_CONNECTION_CHANGED_DURING_ADAPTATION")
        if not torch.equal(new["model"]["connection." + key], expected_new[key]):
            raise RuntimeError("NEW_CONNECTION_SCALE_MISMATCH")
    decision = committed["selected"]["proposals"][-1]
    if (
        committed.get("phase") != "COMMITTED"
        or decision["chosen_scale"] != chosen
        or not decision["trust_region_passed"]
        or not decision["paired_route_streams_verified"]
        or decision["adapt_steps"] != c["adapt_steps"]
    ):
        raise RuntimeError("PARENT_ADAPTATION_TRANSACTION_MISMATCH")
    target = new if c["selected_branches"][arm] == "new" else old
    if decision["final_quality_accepted"] != (c["selected_branches"][arm] == "new"):
        raise RuntimeError("SELECTED_BRANCH_STATE_MISMATCH")
    if any(
        fingerprint(selected[k]) != fingerprint(target[k])
        for k in ("model", "cell_optimizer", "data_rng", "route_rng", "data_cursor", "step")
    ):
        raise RuntimeError("SELECTED_BRANCH_STATE_MISMATCH")
    return {
        "same_initial_cells_and_optimizer": True,
        "paired_data_and_route_rng": True,
        "frozen_weights_equal": True,
        "selected_branch_verified": True,
        "official_model_hash": fingerprint(official["model"]),
        "old_model_hash": fingerprint(old["model"]),
        "new_model_hash": fingerprint(new["model"]),
        "old_cell_optimizer_hash": fingerprint(old["cell_optimizer"]),
        "new_cell_optimizer_hash": fingerprint(new["cell_optimizer"]),
        "margin": float(official["model"]["route_switch_margin"]),
        "numeric_tolerance": float(official["model"]["route_numeric_tolerance"]),
    }


def restore(inputs, c, arm):
    root = "jobs/" + arm + "/transactions/proposal-0100"
    try:
        pair, pair_ref = inputs.checkpoint(root, c["adapt_steps"])
    except FileNotFoundError as e:
        raise RuntimeError("PAIRED_BRANCH_STATE_NOT_RETAINED") from e
    if "old" not in pair or "new" not in pair:
        raise RuntimeError("PAIRED_BRANCH_STATE_NOT_RETAINED")
    proposal, proposal_ref = inputs.checkpoint("proposals/" + arm, 0)
    initial, initial_ref = inputs.checkpoint(root, 0)
    committed, commit_ref = inputs.checkpoint(root)
    selected, selected_ref = inputs.checkpoint("jobs/" + arm + "/selected")
    official = proposal["official"]
    result = validate(
        c, arm, official, pair["old"], pair["new"], selected, proposal, pair, initial, committed
    )
    rows = pair.get("training_route_rows")
    monitor, monitor_ref = inputs.checkpoint("jobs/" + arm + "/training-observations")
    if rows is None or fingerprint(rows) != fingerprint(monitor["rows"]):
        raise RuntimeError("TRAINING_ROUTE_OBSERVATIONS_MISMATCH")
    expected_identity = proposal_ref["receipt_identity"]
    for ref in (pair_ref, initial_ref, commit_ref, selected_ref, monitor_ref):
        normalized = {k: v for k, v in ref["receipt_identity"].items() if k != "proposal_at_step"}
        if normalized != expected_identity:
            raise RuntimeError("PARENT_BRANCH_RECEIPT_IDENTITIES_DIFFER")
    if "identity" in proposal and proposal["identity"] != expected_identity:
        raise RuntimeError("PROPOSAL_PACKAGE_IDENTITY_MISMATCH")
    for key, value in (
        ("original_fingerprint", proposal["original_connection"]),
        ("candidate_fingerprint", proposal["candidate_connection"]),
    ):
        if key in proposal and proposal[key] != fingerprint(value):
            raise RuntimeError("PROPOSAL_PARAMETER_FINGERPRINT_MISMATCH")
    parent_result = inputs.json("jobs/" + arm + "/result.json")
    if fingerprint(rows) != fingerprint(parent_result["training_routes"]):
        raise RuntimeError("TRAINING_ROUTE_OBSERVATIONS_MISMATCH")
    if (
        parent_result["identity"] != proposal_ref["receipt_identity"]
        or parent_result["trust"]["chosen_scale"] != c["chosen_scale"]
    ):
        raise RuntimeError("PARENT_RESULT_IDENTITY_MISMATCH")
    return {
        "official": official,
        "old": pair["old"],
        "new": pair["new"],
        "proposal": proposal,
        "decision": committed["selected"]["proposals"][-1],
        "routes": rows,
        "parent_result": parent_result,
        "verification": {
            **result,
            "receipts": {
                "pair10": pair_ref,
                "proposal": proposal_ref,
                "initial_pair": initial_ref,
                "committed": commit_ref,
                "selected": selected_ref,
            },
        },
        "transaction_root": root,
    }
