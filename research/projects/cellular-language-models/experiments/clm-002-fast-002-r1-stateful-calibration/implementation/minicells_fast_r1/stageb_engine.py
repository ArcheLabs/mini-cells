"""Original FAST main loop, with the separately frozen R1 candidate governance."""

import copy
from types import SimpleNamespace

from minicells_fast001.engine import Engine as OriginalEngine
from minicells_fast001.parent import make_engine as original_make_engine
from minicells_jct001.cost import Journal
from minicells_r3e import policy_proposal
from minicells_r3e.common import fingerprint
from minicells_r3k.reuse import bind

from . import candidate, scan
from .common import atomic, read

collect = policy_proposal.collect


def propose(tr, data):
    # OriginalEngine binds this collector to the actual just-consumed main batch.
    source = collect.__wrapped__.__globals__["train_batches"]
    tr.pending_batch = source(None, None, None, next(tr.model.parameters()).device)
    step = tr.step + 1
    private = tr.stage_directory / "proposals" / str(step)
    public = tr.stage_reports / (str(tr.model.seed) + "-" + tr.arm + "-" + str(step))
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    ident = dict(tr.stage_identity, proposal_step=step)
    cj = Journal(private / "cost.jsonl", ident)
    cj.meter = tr.journal.meter
    package = candidate.generate(tr, cj, private, ident)
    atomic(
        public / "CANDIDATE.json",
        dict(
            read(private / "CANDIDATE.json"),
            losses=package["losses"],
            optimizer_initial_sha256=fingerprint(package["optimizer_initial"]),
            optimizer_final_sha256=fingerprint(package["optimizer_final"]),
            original_connection_sha256=fingerprint(package["original_connection"]),
            full_candidate_connection_sha256=fingerprint(package["candidate_connection"]),
        ),
    )
    sj = Journal(private / "scan.jsonl", ident)
    sj.meter = tr.journal.meter
    decision = scan.run(tr, package, tr.stage_refs, sj, private, tr.fast, ident)
    passing = [
        row for row in decision["attempts"] if row["passed"] and row["r3fv2_nonempty_path_pass"]
    ]
    chosen = max(passing, key=lambda row: row["scale"]) if passing else None
    decision.update(
        formal_selected_scale=None if chosen is None else chosen["scale"],
        formal_rule=tr.fast["stage_b"]["selection"],
        candidate_optimizer_calls=cj.summary()["optimizer_calls_by_group"].get("connection", 0),
    )
    atomic(public / "ADMISSION_DECISION.json", decision)
    result = None
    if chosen is not None:
        result = copy.deepcopy(tr.model)
        for name, param in result.connection.named_parameters():
            old = package["original_connection"][name].to(param)
            full = package["candidate_connection"][name].to(param)
            param.data.copy_(old + chosen["scale"] * (full - old))
        if fingerprint(result.connection.state_dict()) != chosen["scaled_connection_sha256"]:
            raise RuntimeError("R1_STAGEB_SELECTED_CONNECTION_IDENTITY_MISMATCH")
    # Official commit occurs in OriginalEngine, after this function returns.
    return result, {
        "step": step,
        "accepted": chosen is not None,
        "scale": None if chosen is None else chosen["scale"],
        "candidate_source_model_sha256": package["source_model_sha256"],
        "selected_connection_sha256": None
        if chosen is None
        else chosen["scaled_connection_sha256"],
        "primary_chosen_scale": decision["chosen_scale"],
        "all_scales": len(decision["attempts"]),
        "r3fv2_nonempty_path_required": True,
        "report_key": public.name,
        "counterfactual_states": package["states"],
        "actions_per_state": package["actions"],
        "proposal_optimizer_final_state": package["optimizer_final"],
    }


class Engine(OriginalEngine):
    def update(self, stream, checkpoint, references, failure_hook=None):
        return bind(
            OriginalEngine.update,
            policy_proposal=SimpleNamespace(collect=collect, optimize=propose),
        )(self, stream, checkpoint, references, failure_hook)


def make_engine(
    c, state, seed, arm, identity, directory, reports, refs, parent, *, device, unit=False
):
    if not unit and (
        seed not in c["seeds"]
        or arm not in matrix_arms()
        or state.get("arm") != "A-" + arm[-1]
        or state.get("step") != 500
        or state.get("fast", {}).get("phase") != "A"
        or parent["identity"]["seed"] != seed
        or parent["identity"]["arm"] != "A-" + arm[-1]
        or parent["identity"]["source_sha"] != c["historical"]["source_sha"]
        or parent["identity"]["run_id"] != c["historical"]["run_id"]
    ):
        raise RuntimeError("R1_STAGEB_EXACT_ORIGINAL_A500_PARENT_REQUIRED")
    engine = bind(original_make_engine, Engine=Engine)(
        c,
        state,
        seed,
        arm,
        identity,
        directory / "cost.jsonl",
        device,
        unit=unit,
        parent_sha=parent["checkpoint_sha256"],
    )
    engine.r1_cpu_fixture = unit
    engine.stage_directory, engine.stage_reports = directory, reports
    engine.stage_identity, engine.stage_refs = identity, refs
    return engine


def matrix_arms():
    return [
        arm + "-" + condition for condition in ("S", "H") for arm in ("B00", "B10", "B01", "B11")
    ]
