"""Public aggregates only, with independent quality and mechanism conclusions."""

from pathlib import Path
import numpy as np
from minicells_r3e.storage import Store
from minicells_r3e.common import atomic, read, digest
from minicells_r3h.statistics import compare


def report(c, output, failures, prior):
    from .runner import directory, identity

    folder = Path(output) / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    results = {}
    nll = []
    events = []
    alignment = []
    cost = []
    baselines = []
    organization = []
    for seed in c["seeds"]:
        per = {}
        for frozen, oldarm in c["baseline"]["mapping"].items():
            file = (
                Path(output)
                / "baseline-r3h"
                / ("jobs/seed-%s-arm-%s" % (seed, oldarm))
                / "result.json"
            )
            if file.exists():
                old = read(file)
                if (
                    old["identity"]["run_id"] != c["baseline"]["run_id"]
                    or old["status"] != "COMPLETED"
                ):
                    raise RuntimeError("INVALID_HISTORICAL_RESULT")
                per[frozen] = old
                baselines.append(
                    dict(
                        seed=seed,
                        arm=frozen,
                        source_arm=oldarm,
                        checkpoint=old["checkpoint"]["sha256"],
                        nll=old["quality"]["nll"],
                    )
                )
        for arm in c["new_arms"]:
            file = directory(output, seed, arm) / "result.json"
            if not file.exists():
                continue
            r = read(file)
            state = Store(file.parent, r["identity"]).verified(r["checkpoint"])
            curves = state["curve"]
            stage = np.array([row["exposure"]["stage_cells"] for row in curves])
            edges = np.array([row["exposure"]["transitions"] for row in curves]).sum(0)
            action = np.array([row["exposure"]["stage_actions"] for row in curves]).sum(0)
            gradients = np.array([row["cell_gradient_norms"] for row in curves])
            conditional = edges / edges.sum(-1, keepdims=True).clip(1)
            marginal = edges.sum(0) / max(1, edges.sum())
            dependence = float(
                (np.abs(conditional - marginal).sum(-1) * edges.sum(-1) / max(1, edges.sum())).sum()
                / 2
            )
            organization.append(
                dict(
                    seed=seed,
                    arm=arm,
                    connection_version=state["connection_version"],
                    cell_usage=stage.sum((0, 1)).tolist(),
                    predecessor_dependence=dependence if arm[0] == "B" else None,
                    max_min_cell_call_ratio=float(
                        stage.sum((0, 1)).max() / stage.sum((0, 1)).clip(1).min()
                    ),
                    stage_usage=stage.sum(0).tolist(),
                    pair_usage=action.tolist() if arm[0] == "A" else None,
                    edge_usage=edges.tolist() if arm[0] == "B" else None,
                    conditional_edge_distribution=(
                        edges / edges.sum(-1, keepdims=True).clip(1)
                    ).tolist()
                    if arm[0] == "B"
                    else None,
                    self_connection_fraction=float(np.trace(edges) / max(1, edges.sum()))
                    if arm[0] == "B"
                    else None,
                    gradient_update_counts=(gradients > 0).sum(0).tolist(),
                    mean_gradient_norms=gradients.mean(0).tolist(),
                    organization_change=[
                        dict(step=row["step"], cells=row["cell_counts"])
                        for row in curves
                        if row["step"] % 50 == 0
                    ],
                )
            )
            if "quality" in r:
                per[arm] = r
                control = arm[0] + "-F" + arm[-1]
                nll.append(
                    dict(
                        seed=seed,
                        architecture=arm[0],
                        routing_mode=arm[-1],
                        connection_mode="COADAPTIVE",
                        frozen_baseline_nll=per[control]["quality"]["nll"],
                        coadaptive_nll=r["quality"]["nll"],
                        delta_nll=r["quality"]["nll"] - per[control]["quality"]["nll"],
                        parent_checkpoint_sha256=r["parent_checkpoint"]["sha256"],
                        final_checkpoint_sha256=r["checkpoint"]["sha256"],
                    )
                )
            for e in r["proposals"]:
                events.append(dict(seed=seed, arm=arm, **e))
            alignment.append(
                dict(
                    seed=seed,
                    arm=arm,
                    **r["alignment"],
                    training_curve=[
                        dict(
                            step=row["step"],
                            ce=row["ce"],
                            epsilon=row["epsilon"],
                            local_matches=row["exposure"]["matches"],
                            probability_sums=row["exposure"]["probability_sums"],
                        )
                        for row in curves
                    ],
                )
            )
            cost.append(
                dict(
                    seed=seed,
                    arm=arm,
                    main_cost=r["cost"],
                    connection_proposals=len(r["proposals"]),
                    policy_optimizer_updates=state["policy_updates"],
                    proposal_phase_costs=[
                        {
                            "at_step": e["at_step"],
                            "candidate": e.get("candidate_cost", {}),
                            "trust": e.get("scan_cost", {}),
                            "adapt_steps": e.get("adapt_steps", 0),
                            "discarded_tokens": e.get("adapt_steps", 0)
                            * c["training"]["tokens_per_step"],
                        }
                        for e in r["proposals"]
                    ],
                    main_forward_work=state["work"],
                    forward_cost_scope="EXECUTED_OPERATOR_ROWS_NOT_HARDWARE_FLOPS; BACKWARD_NOT_DIRECTLY_MEASURED",
                    development_work={
                        k: v for k, v in r.get("quality", {}).items() if k != "stories"
                    },
                    wall_seconds=r["training_wall_seconds"],
                    gpu_memory_peak=r["peak_allocated_bytes"],
                )
            )
        results[str(seed)] = per
    complete = not failures and len(nll) == 12
    comparisons = compare(c, results) if complete else {}
    metrics = dict(
        status="COMPLETED" if complete else "PARTIAL",
        identity=identity(c),
        hf_manifest_sha256=prior.get("manifest_sha256") if prior else None,
        failures=failures,
        comparisons=comparisons,
        confirmation="SEALED",
        scientific_decision=False,
        completed_models=len(nll),
        admission_successes=sum(bool(e["trust_region_passed"]) for e in events),
        quality_acceptances=sum(bool(e["accepted"]) for e in events),
        interpretation="Admission, Cell adaptation and model quality are independent evidence levels.",
    )
    files = {
        "METRICS.json": metrics,
        "INPUT_VERIFICATION.json": {"baseline": c["baseline"], "parent": c["parent"]},
        "BASELINE_VERIFICATION.json": baselines,
        "DEVELOPMENT_NLL.json": nll,
        "PROPOSAL_RESULTS.json": events,
        "TRUST_REGION_RESULTS.json": [
            dict(seed=e["seed"], arm=e["arm"], at_step=e["at_step"], scan=e["scan"]) for e in events
        ],
        "ADAPTATION_RESULTS.json": [e for e in events if e.get("adapt_steps")],
        "CELL_CONNECTION_ATTRIBUTION.json": [
            dict(seed=e["seed"], arm=e["arm"], at_step=e["at_step"], **e["attribution"])
            for e in events
            if "attribution" in e
        ],
        "TRAINING_ALIGNMENT.json": alignment,
        "GRAPH_ORGANIZATION.json": organization,
        "COMPUTE_COST.json": cost,
    }
    for name, value in files.items():
        atomic(folder / name, value)
    supported = [k for k, v in comparisons.items() if v["status"] == "SUPPORTED_EXPLORATORY"]
    adaptation_evidence = [
        e
        for e in events
        if e.get("attribution", {}).get("cell_effect_resolved")
        and e["attribution"].get("cell_effect", 0) < 0
        and any(row["path_change"] > 0 for row in e.get("training_path_rows", []))
    ]
    repeated = {e["seed"] for e in adaptation_evidence}
    conclusion = (
        "STABLE_COADAPTATION_SUPPORTED_EXPLORATORY"
        if supported and len(repeated) == 3
        else "CELL_ADAPTATION_EVIDENCE_OBSERVED"
        if adaptation_evidence
        else "CONNECTION_QUALITY_GAIN_WITH_LIMITED_CELL_ADAPTATION"
        if supported
        else "COADAPTATION_QUALITY_NOT_SUPPORTED"
        if complete
        else "INCONCLUSIVE_PARTIAL"
    )
    text = [
        "# R3i exploratory coadaptation",
        metrics["status"],
        conclusion,
        "Confirmation SEALED. Historical Development and repeated Calibration selection are not independent confirmation.",
        "T proposal suffix values use deterministic replanning, not the exact stochastic T policy expectation.",
        "No automatic next experiment. Admission does not imply beneficial Cell adaptation or quality support.",
        "Quality hypotheses:",
    ]
    for name, value in comparisons.items():
        text.append("%s: %s, delta %.8f" % (name, value["status"], value["mean_delta"]))
    text.extend(
        [
            "Admission successes: %s/%s." % (metrics["admission_successes"], len(events)),
            "Quality acceptance: %s/%s admitted."
            % (metrics["quality_acceptances"], metrics["admission_successes"]),
            "Per-seed accepted updates: "
            + str(
                {
                    seed: sum(e["accepted"] for e in events if e["seed"] == seed)
                    for seed in c["seeds"]
                }
            ),
            "Resolved favorable Cell effects with actual training-path divergence: %s events, %s Seeds."
            % (len(adaptation_evidence), len(repeated)),
            "Route/Cell/interactions per event are in CELL_CONNECTION_ATTRIBUTION; effects below float32 resolution do not support adaptation.",
            "Accumulation and concentration: consult ordered per-step organization records and event effects; no monotonic benefit is assumed.",
            "H versus T: compare the four registered within-router quality comparisons, not selected post-hoc winners.",
            "Compute cost includes discarded branches and all candidate/scan phases. Equal main tokens do not imply equal total work.",
            "These data alone do not authorize independent confirmation or establish Cell specialization.",
        ]
    )
    (folder / "SCIENTIFIC-FINDINGS.md").write_text("\n".join(text) + "\n")
    atomic(
        folder / "RUN_MANIFEST.json",
        dict(
            **identity(c),
            status=metrics["status"],
            hf_reference=prior,
            artifacts=[
                dict(path=f.name, bytes=f.stat().st_size, sha256=digest(f))
                for f in sorted(folder.iterdir())
                if f.name != "RUN_MANIFEST.json"
            ],
            confirmation="SEALED",
        ),
    )
    return metrics
