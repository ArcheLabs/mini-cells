"""All signs and all five scores reported, independently of study outcome."""

from pathlib import Path
from minicells_r3e.common import atomic, read, digest
from minicells_r3j.reports import interval
from .statistics import analyse

FILES = (
    "INPUT_VERIFICATION.json",
    "HISTORICAL_CANDIDATES.json",
    "DATA_BOUNDARY_PROOF.json",
    "FROZEN_PREDICTIONS.json",
    "CONTINUATION_CHECKPOINTS.json",
    "LONG_HORIZON_VALUE.json",
    "CELL_LEARNING_ATTRIBUTION.json",
    "PREDICTIVE_SIGNAL_ANALYSIS.json",
    "COMPUTE_COST.json",
    "SCIENTIFIC-FINDINGS.md",
    "RUN_MANIFEST.json",
)


def report(c, output, selection, failures, reference):
    from .runner import case_root, identity

    output = Path(output)
    folder = output / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    values = []
    attribution = []
    checkpoints = []
    cost = []
    prediction_rows = []
    private = output / "private-checkpoints"
    frozen = (
        read(private / "FROZEN_PREDICTIONS.json")
        if (private / "FROZEN_PREDICTIONS.json").exists()
        else None
    )
    predictions = {x["case_id"]: x for x in frozen["rows"]} if frozen else {}
    for case in selection["cases"]:
        root = case_root(output, case)
        if (root / "TRAINED.json").exists():
            cost.append(read(root / "TRAINED.json"))
        for horizon in c["horizons"]:
            path = root / ("EVALUATION-%s.json" % horizon)
            if not path.exists():
                continue
            r = read(path)
            error = 2 * r["numerical_resolution"]
            row = dict(
                case_id=case["case_id"],
                seed=case["seed"],
                arm=case["arm"],
                historical_accepted=case["historical_accepted"],
                reused_r3j=case["reused_r3j"],
                horizon=horizon,
                V=r["V"],
                R=r["R"],
                C=r["C"],
                I=r["I"],
                values=r["values"],
                numerical_error_bound=error,
                measured_repeat_error=r["measured_repeat_error"],
                status="BELOW_NUMERICAL_RESOLUTION"
                if abs(r["V"]) <= error
                else "POSITIVE"
                if r["V"] > 0
                else "NEGATIVE",
                practical_gain=r["V"] >= c["practical_gain"],
                bootstrap=interval(r, c["bootstrap_seed"], c["bootstrap_replicates"]),
            )
            values.append(row)
            attribution.append(
                dict(row, cell_parameter_difference_l2=r["cell_parameter_difference_l2"])
            )
            checkpoints.append(
                dict(case_id=case["case_id"], horizon=horizon, receipt=r["checkpoint"])
            )
            if horizon == 100 and case["case_id"] in predictions:
                prediction_rows.append(
                    dict(
                        predictions[case["case_id"]],
                        V=r["V"],
                        C=r["C"],
                        outcome_error=error,
                        prediction_error=predictions[case["case_id"]]["numerical_error_bound"],
                    )
                )
    stats = (
        analyse(prediction_rows, c)
        if len(prediction_rows) == 48
        else dict(status="PREDICTION_EVIDENCE_INCONCLUSIVE", completed=len(prediction_rows))
    )
    complete = len(values) == 144 and not failures
    if sum(x.get("new_tokens", 0) for x in cost) > c["new_tokens"]:
        raise RuntimeError("NEW_TRAINING_BUDGET_EXCEEDED")
    summary = dict(
        status="COMPLETED_EXPLORATORY" if complete else "PARTIAL",
        failures=failures,
        identity=identity(c),
        prediction_scope=c["prediction_scope"],
        confirmation="SEALED",
        scientific_confirmation=False,
        new_cell_updates=sum(2 * x.get("updates_per_branch", 0) for x in cost),
        new_tokens=sum(x.get("new_tokens", 0) for x in cost),
        connection_updates=0,
        evaluated_cases=len(prediction_rows),
        hf_reference=reference,
    )
    terminal = [r for r in values if r["horizon"] == 100]
    resolved = [r for r in terminal if r["status"] != "BELOW_NUMERICAL_RESOLUTION"]
    route = sum(r["R"] for r in resolved)
    cell = sum(r["C"] for r in resolved)
    component = (
        "NO_RESOLVED_COMPONENT"
        if not resolved
        else "CELL_ADAPTATION_DOMINANT"
        if cell > 0 and cell > abs(route)
        else "DIRECT_ROUTE_DOMINANT"
        if route > 0 and route > abs(cell)
        else "MIXED_EFFECT"
    )
    summary.update(
        long_horizon_value="INCONCLUSIVE"
        if not complete
        else "LONG_HORIZON_VALUE_OBSERVED"
        if any(r["V"] > r["numerical_error_bound"] for r in terminal)
        else "NO_DETECTABLE_LONG_HORIZON_VALUE",
        attribution=component,
    )
    proof = (
        read(output / "DATA_BOUNDARY_PROOF.json")
        if (output / "DATA_BOUNDARY_PROOF.json").exists()
        else {"status": "NOT_AVAILABLE"}
    )
    documents = {
        "INPUT_VERIFICATION.json": summary,
        "HISTORICAL_CANDIDATES.json": selection,
        "DATA_BOUNDARY_PROOF.json": proof,
        "FROZEN_PREDICTIONS.json": frozen or {"status": "NOT_FROZEN"},
        "CONTINUATION_CHECKPOINTS.json": checkpoints,
        "LONG_HORIZON_VALUE.json": values,
        "CELL_LEARNING_ATTRIBUTION.json": attribution,
        "PREDICTIVE_SIGNAL_ANALYSIS.json": stats,
        "COMPUTE_COST.json": dict(
            training=cost,
            predictor_seconds=sum(x.get("seconds", 0) for x in predictions.values()),
            evaluation_seconds=sum(
                read(p)["evaluation_seconds"] for p in private.glob("cases/*/EVALUATION-*.json")
            ),
            scope="Measured wall time and inherited executed operator rows; FLOPs are not inferred from elapsed time",
        ),
    }
    for name, document in documents.items():
        atomic(folder / name, document)
    signal = stats.get("groups", {}).get("R3K_NEW36", {})
    summary["predictor_computability"] = (
        "ALL48_FROZEN_PRIVATE_READBACK_REQUIRED" if frozen else "INCOMPLETE"
    )
    answers = {
        "Q1_Direct_predicts_future_value": signal.get("S_Direct", {}).get(
            "status", "PREDICTION_EVIDENCE_INCONCLUSIVE"
        ),
        "Q2_Early_Cell_adaptation_signal": signal.get("S_Cell", {}).get(
            "status", "PREDICTION_EVIDENCE_INCONCLUSIVE"
        ),
        "Q3_Any_fixed_early_predictor_signal": "PREDICTIVE_SIGNAL_EXPLORATORY"
        if any(r.get("status") == "PREDICTIVE_SIGNAL_EXPLORATORY" for r in signal.values())
        else "PREDICTIVE_SIGNAL_NOT_SUPPORTED"
        if complete
        else "PREDICTION_EVIDENCE_INCONCLUSIVE",
    }
    atomic(folder / "PREDICTIVE_SIGNAL_ANALYSIS.json", dict(stats, core_questions=answers))
    (folder / "SCIENTIFIC-FINDINGS.md").write_text(
        "# R3k — Future Learning Value\n\n"
        + summary["status"]
        + "\n\n"
        + "Target: V100 on the new research Holdout. All five frozen scores are reported. V=R+C; interaction is separate.\n\n"
        + "NEW36 comprises historical rejected candidates; ALL48 is a fixed historical population. Neither is 48 independent Seeds. Known R3j results motivated this exploratory protocol; this is not independent Confirmation.\n\n"
        + "\n".join(k + ": " + v for k, v in answers.items())
        + "\n\nThese are exploratory univariate signals, not proof of unique causal predictive information or a deployable acceptance policy. No decision signal or R3l training is activated automatically.\n"
    )
    atomic(
        folder / "RUN_MANIFEST.json",
        dict(
            summary,
            artifacts=[
                dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p))
                for p in sorted(folder.iterdir())
                if p.is_file() and p.name != "RUN_MANIFEST.json"
            ],
        ),
    )
    return summary
