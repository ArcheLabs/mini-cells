"""Per-case exploratory inference, with selected-sample and shared-Seed caveats."""

from pathlib import Path
import numpy as np
from minicells_r3e.common import read, atomic, digest


def rank(x):
    x = np.asarray(x, float)
    out = np.empty(len(x), float)
    for value in np.unique(x):
        idx = np.where(x == value)[0]
        out[idx] = np.sum(x < value) + 0.5 * (len(idx) - 1)
    return out


def correlation(x, y):
    if len(x) < 2:
        return None
    a, b = rank(x), rank(y)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def interval(result, seed, replicates):
    a, b = result["stories"]["OO"], result["stories"]["NN"]
    if set(a) != set(b):
        raise RuntimeError("LONG_HORIZON_STORY_PAIRING_CHANGED")
    keys = sorted(a)
    if any(a[k]["tokens"] != b[k]["tokens"] for k in keys):
        raise RuntimeError("LONG_HORIZON_TOKEN_WEIGHTS_CHANGED")
    x = np.array([a[k]["mean_nll"] - b[k]["mean_nll"] for k in keys])
    w = np.array([a[k]["tokens"] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(x), size=(replicates, len(x)))
    values = (x[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return dict(
        ci95=np.quantile(values, [0.025, 0.975]).tolist(),
        stories=len(keys),
        unit="PAIRED_STORY_CLUSTER_WITHIN_CASE_NOT_INDEPENDENT_TRAINING_SEEDS",
    )


def report(c, output, selection, failures, reference):
    from .runner import case_root, identity

    folder = Path(output) / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    attribution = []
    organization = []
    restores = []
    checkpoints = []
    cost = []
    case_failures = []
    for case in selection["cases"]:
        root = case_root(output, case)
        if (root / "FAILURE.json").exists():
            case_failures.append(read(root / "FAILURE.json"))
        if (root / "READY.json").exists():
            restores.append(dict(case_id=case["case_id"], **read(root / "READY.json")))
        values = {}
        for horizon in c["horizons"]:
            path = root / ("EVALUATION-%s.json" % horizon)
            if not path.exists():
                continue
            r = read(path)
            values[str(horizon)] = r["V"]
            error = 2 * r["numerical_resolution"]
            row = dict(
                case_id=case["case_id"],
                seed=case["seed"],
                arm=case["arm"],
                historical_accepted=case["historical_accepted"],
                horizon=horizon,
                old_nll=r["values"]["OO"],
                new_nll=r["values"]["NN"],
                V=r["V"],
                numerical_error_bound=error,
                repeat_error=r["measured_repeat_error"],
                bootstrap=interval(r, c["bootstrap_seed"], c["bootstrap_replicates"]),
            )
            if horizon == 100:
                row["status"] = (
                    "LONG_HORIZON_POSITIVE"
                    if r["V"] > error
                    else "LONG_HORIZON_NEGATIVE"
                    if r["V"] < -error
                    else "BELOW_NUMERICAL_RESOLUTION"
                )
                row["practical_long_horizon_gain"] = (
                    r["V"] >= c["practical_gain"] and r["V"] > error
                )
            results.append(row)
            attribution.append(
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("stories", "training_path_rows", "old_alignment", "new_alignment")
                }
            )
            coverage = {}
            for branch in ("old", "new"):
                paths = r["training_path_rows"]
                if paths:
                    stages = np.array([x[branch + "_exposure"]["stage_cells"] for x in paths]).sum(
                        0
                    )
                    edges = np.array([x[branch + "_exposure"]["transitions"] for x in paths]).sum(0)
                    counts = stages.sum(0)
                    grad = np.array([x[branch + "_gradients"] for x in paths])
                    coverage[branch] = dict(
                        cell_calls=counts.tolist(),
                        stage_usage=stages.tolist(),
                        max_min_call_ratio=float(counts.max() / counts.clip(1).min()),
                        zero_use_cells=int((counts == 0).sum()),
                        gradient_update_coverage=(grad > 0).sum(0).tolist(),
                        mean_gradient_norms=grad.mean(0).tolist(),
                        pair_usage=np.array(
                            [x[branch + "_exposure"]["stage_actions"] for x in paths]
                        )
                        .sum(0)
                        .tolist()
                        if case["arm"][0] == "A"
                        else None,
                        transition_counts=edges.tolist() if case["arm"][0] == "B" else None,
                        self_connection_fraction=float(np.trace(edges) / max(1, edges.sum()))
                        if case["arm"][0] == "B"
                        else None,
                        conditional_edges=(edges / edges.sum(-1, keepdims=True).clip(1)).tolist()
                        if case["arm"][0] == "B"
                        else None,
                    )
            organization.append(
                dict(
                    case_id=case["case_id"],
                    horizon=horizon,
                    old_alignment=r["old_alignment"],
                    new_alignment=r["new_alignment"],
                    cell_parameter_difference_l2=r["cell_parameter_difference_l2"],
                    continuation_coverage=coverage,
                    training_path_rows=r["training_path_rows"],
                )
            )
            checkpoints.append(
                dict(case_id=case["case_id"], horizon=horizon, receipt=r["checkpoint"])
            )
        if (root / "TRAINED.json").exists():
            cost.append(read(root / "TRAINED.json"))
    bycase = {}
    for r in results:
        bycase.setdefault(r["case_id"], {})[r["horizon"]] = r
    paired = [
        (case, bycase[case["case_id"]])
        for case in selection["cases"]
        if set(bycase.get(case["case_id"], {})) == {10, 50, 100}
    ]
    prediction = dict(
        sample_selection_bias=True,
        selected_cases_not_representative_of_all48=True,
        independent_confirmation=False,
        prediction_status="HISTORICAL_VALUE_PREDICTION_NOT_RECOVERABLE",
        short_long_development_spearman=correlation(
            [v[10]["V"] for _, v in paired], [v[100]["V"] for _, v in paired]
        ),
        short_calibration_long_development_spearman=correlation(
            [
                x["historical_event"]["attribution"]["values"]["OO"]
                - x["historical_event"]["attribution"]["values"]["NN"]
                for x, _ in paired
            ],
            [v[100]["V"] for _, v in paired],
        ),
        cases=[
            dict(
                case_id=x["case_id"],
                seed=x["seed"],
                historical_accepted=x["historical_accepted"],
                V10=v[10]["V"],
                V50=v[50]["V"],
                V100=v[100]["V"],
                deltaV=v[100]["V"] - v[10]["V"],
                sign_agreement=bool(np.sign(v[10]["V"]) == np.sign(v[100]["V"])),
            )
            for x, v in paired
        ],
        caveat="Accepted/rejected are selected labels, not randomized groups. Cases share Seeds, histories and overlapping training intervals. Story bootstrap is not Seed replication.",
    )
    complete = len(paired) == len(selection["cases"]) and not failures and not case_failures
    summary = dict(
        status="COMPLETED_EXPLORATORY" if complete else "PARTIAL_ENGINEERING_FAILURE",
        identity=identity(c),
        valid_cases=len(paired),
        selected_cases=len(selection["cases"]),
        accepted_complete=all(
            x["case_id"] in bycase and 100 in bycase[x["case_id"]]
            for x in selection["cases"]
            if x["historical_accepted"]
        ),
        failures=failures + case_failures,
        new_connection_optimizer_updates=0,
        completed_new_cell_updates=sum(2 * r["updates_per_branch"] for r in cost),
        completed_new_train_tokens=sum(r["new_tokens"] for r in cost),
        confirmation="SEALED",
        scientific_decision=False,
    )
    files = {
        "PARENT_RESTORE_VERIFICATION.json": restores,
        "CONTINUATION_CHECKPOINTS.json": checkpoints,
        "LONG_HORIZON_NLL.json": results,
        "LONG_HORIZON_ATTRIBUTION.json": attribution,
        "ROUTING_EVOLUTION.json": organization,
        "PREDICTIVE_VALUE_ANALYSIS.json": prediction,
        "COMPUTE_COST.json": cost,
        "METRICS.json": summary,
    }
    for name, value in files.items():
        atomic(folder / name, value)
    lines = [
        "# R3j — Exploratory long-horizon counterfactual value",
        summary["status"],
        "V=Old NLL-New NLL; V=R+C. Interaction I is separate and must not be added again.",
        "No new proposals, Connection updates, historical K10 replay, or Confirmation reads.",
        prediction["caveat"],
    ]
    for case, v in paired:
        lines.append(
            "%s %s seed%s accepted=%s: V10=%.8f V50=%.8f V100=%.8f; %s"
            % (
                case["case_id"][:12],
                case["arm"],
                case["seed"],
                case["historical_accepted"],
                v[10]["V"],
                v[50]["V"],
                v[100]["V"],
                v[100]["status"],
            )
        )
    lines.extend(
        [
            "A positive V10 can persist or reverse; inspect each case rather than infer monotonic improvement.",
            "Rejected positive cases can reveal missed opportunities but do not estimate population false-negative rates.",
            "R and C distinguish direct routing benefits from Cell-source benefits; I alone does not establish specialization.",
            "No automatic R3k, new candidate search, threshold relaxation or independent-confirmation claim.",
        ]
    )
    (folder / "SCIENTIFIC-FINDINGS.md").write_text("\n".join(lines) + "\n")
    atomic(
        folder / "RUN_MANIFEST.json",
        dict(
            **identity(c),
            status=summary["status"],
            execution_id=read(Path(output) / "EXECUTION.json")["execution_id"],
            hf_reference=reference,
            case_selection_manifest_sha256=selection["sha256"],
            artifacts=[
                dict(path=f.name, bytes=f.stat().st_size, sha256=digest(f))
                for f in sorted(folder.glob("*"))
                if f.is_file() and f.name != "RUN_MANIFEST.json"
            ],
        ),
    )
    return summary
