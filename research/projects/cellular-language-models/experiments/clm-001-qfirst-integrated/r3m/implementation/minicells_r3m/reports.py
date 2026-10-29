"""Complete-matrix reports; partial runs remain inconclusive and Main remains separate."""

from pathlib import Path
from minicells_r3e.common import read, atomic, digest
from .protocol import pilot_decision
from .uncertainty import paired

FILES = [
    "INPUT_VERIFICATION.json",
    "DATA_BOUNDARY_PROOF.json",
    "TASK_LEARNABILITY.json",
    "PILOT_MODEL_IDENTITY.json",
    "PILOT_INTERVENTIONS.json",
    "PILOT_LEARNING_CURVES.json",
    "PILOT_STORY_RETENTION.json",
    "PILOT_ARITHMETIC_ACQUISITION.json",
    "PILOT_LONG_HORIZON.json",
    "PILOT_ATTRIBUTION.json",
    "PILOT_COMPUTE_COST.json",
    "PILOT_SCIENTIFIC-FINDINGS.md",
    "PILOT_RUN_MANIFEST.json",
]


def write(c, output, ident, failures, hf_reference):
    output = Path(output)
    dest = output / "reports"
    dest.mkdir(parents=True, exist_ok=True)
    shards = [read(p) for p in (output / "private-checkpoints/shards").glob("*/EVALUATION.json")]
    seed_rows = []
    for shard in shards:
        if shard["environment"] != "C":
            continue
        q = shard["evaluations"]
        p0 = q["P0"]
        p1 = q["P1"]
        p2 = q["P2"]
        anchor = q["P0"]["0"]
        at = shard["attribution"]["600"]
        txn = shard["transactions"]["P1"][0]
        r = dict(
            seed=shard["seed"],
            task_learnable=(
                p0["600"]["answer_nll"] < anchor["answer_nll"] - 0.005
                and p0["600"]["exact_match"] > anchor["exact_match"]
                and p0["600"]["exact_match"] > shard["answer_frequency_baseline"]
            ),
            effective=txn["accepted"] and txn["trace"]["effective"],
            action_change=txn["trace"]["action_change"],
            answer_gain_400=p0["400"]["answer_nll"] - p1["400"]["answer_nll"],
            answer_gain_600=p0["600"]["answer_nll"] - p1["600"]["answer_nll"],
            story_harm_600=p1["600"]["story_nll"] - p0["600"]["story_nll"],
            em_change_600=p1["600"]["exact_match"] - p0["600"]["exact_match"],
            sham_advantage_600=p2["600"]["answer_nll"] - p1["600"]["answer_nll"],
            attribution_valid=all(abs(v["V"] - v["R"] - v["C"]) < 1e-9 for v in at.values()),
        )
        r["paired_group_uncertainty"] = {
            task: paired(p0["600"], p1["600"], task) for task in ("story", "arithmetic")
        }
        r["absolute_story_forgetting_600"] = p1["600"]["story_nll"] - anchor["story_nll"]
        seed_rows.append(r)
    decision = pilot_decision(seed_rows, c["pilot_thresholds"], c["pilot_seeds"])
    complete = len(shards) == len(c["pilot_seeds"]) * len(c["environments"]) and not failures
    if not complete:
        decision = dict(
            status="INCONCLUSIVE",
            Q1="INCONCLUSIVE",
            Q2="INCONCLUSIVE",
            Q3="INCONCLUSIVE",
            Q4="NOT_TESTED",
        )
    else:
        # Do not infer demand simply from introducing Arithmetic. Keep independent descriptive diagnostics.
        decision["Q1"] = "INCONCLUSIVE"
        decision["Q2"] = "INCONCLUSIVE"
        decision["reason_Q1_Q2"] = (
            "Joint usage/gradient/alignment and fixed-train pressure evidence require independent review"
        )
    tables = {
        "INPUT_VERIFICATION.json": dict(
            identity=ident,
            baseline=c["baseline"],
            cross_session_proof=read(output / "CROSS_SESSION_PROOF.json")
            if (output / "CROSS_SESSION_PROOF.json").exists()
            else None,
        ),
        "DATA_BOUNDARY_PROOF.json": read(output / "DATA_BOUNDARY_PROOF.json")
        if (output / "DATA_BOUNDARY_PROOF.json").exists()
        else dict(status="NOT_PREPARED"),
        "TASK_LEARNABILITY.json": dict(seed_results=seed_rows, confirmation="SEALED"),
        "PILOT_MODEL_IDENTITY.json": [
            dict(seed=s["seed"], environment=s["environment"], models=s["model_identity"])
            for s in shards
        ],
        "PILOT_INTERVENTIONS.json": [
            dict(seed=s["seed"], environment=s["environment"], transactions=s["transactions"])
            for s in shards
        ],
        "PILOT_LEARNING_CURVES.json": [
            dict(
                seed=s["seed"],
                environment=s["environment"],
                curves=s["curves"],
                train_only_pressure=s.get("pressure", {}),
            )
            for s in shards
        ],
        "PILOT_STORY_RETENTION.json": [
            dict(seed=s["seed"], environment=s["environment"], evaluations=s["evaluations"])
            for s in shards
        ],
        "PILOT_ARITHMETIC_ACQUISITION.json": dict(
            seed_results=seed_rows,
            details="PILOT_STORY_RETENTION.json contains separate answer NLL, full NLL, carry and exact-match records",
        ),
        "PILOT_LONG_HORIZON.json": dict(
            seed_results=seed_rows,
            decision=decision,
            uncertainty="Three exploratory seeds; no independent confirmation",
        ),
        "PILOT_ATTRIBUTION.json": [
            dict(
                seed=s["seed"],
                environment=s["environment"],
                four_grid=s["attribution"],
                specialization=s["specialization"],
            )
            for s in shards
        ],
        "PILOT_COMPUTE_COST.json": dict(
            shards=[s["cost"] for s in shards],
            strict_equal_total_compute=False,
            logical_updates=14400,
            logical_tokens=7372800,
            scope="Actual main, discarded Shadow, proposal and evaluation costs reported separately",
        ),
    }
    for name, row in tables.items():
        atomic(dest / name, row)
    (dest / "PILOT_SCIENTIFIC-FINDINGS.md").write_text(
        "# R3m-P\n\nStatus: "
        + decision["status"]
        + "\n\nExploratory only. Main is never automatically released. Q1/Q2 require mechanism review; Q4 NOT_TESTED.\n"
    )
    manifest = dict(
        identity=ident,
        status="PILOT_COMPLETE_EXPLORATORY" if complete else "PARTIAL",
        decision=decision,
        completed_shards=len(shards),
        expected_shards=6,
        failures=failures,
        hf_reference=hf_reference,
        confirmation="SEALED",
        scientific_decision=False,
        artifacts=[
            dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(dest.iterdir())
            if p.name in FILES and p.name != "PILOT_RUN_MANIFEST.json"
        ],
    )
    atomic(dest / "PILOT_RUN_MANIFEST.json", manifest)
    return manifest
