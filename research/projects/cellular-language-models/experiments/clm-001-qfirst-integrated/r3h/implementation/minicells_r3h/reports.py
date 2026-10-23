"""Aggregate-only public reports; preserve incomplete runs without scientific overclaim."""

import numpy as np
from .parent_import import atomic, digest
from .statistics import compare


def write(c, output, results, failures, manifest):
    folder = output / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    valid = not failures and all(
        len(results[str(s)]) == 8
        and all(r["status"] == "COMPLETED" for r in results[str(s)].values())
        for s in c["seeds"]
    )
    comparisons = compare(c, results) if valid else {}
    nll = []
    align = []
    exposure = []
    cost = []
    inputs = []
    fits = []
    for seed, arms in results.items():
        for arm, r in arms.items():
            common = dict(seed=int(seed), architecture=arm[0], routing_mode=arm[-1])
            i = r["initial"]
            inputs.append(
                dict(
                    **common,
                    **{k: v for k, v in i.items() if k not in ("initial_quality", "temperature")},
                )
            )
            fits.append(dict(**common, **i["temperature"]))
            if "quality" in r:
                baseline = arms.get(arm[0] + "-F", {}).get("quality", {}).get("nll")
                nll.append(
                    dict(
                        **common,
                        development_nll=r["quality"]["nll"],
                        baseline_development_nll=baseline,
                        delta_from_F=r["quality"]["nll"] - baseline
                        if baseline is not None
                        else None,
                        initial_calibration_nll=i["initial_quality"]["nll"],
                        parent_checkpoint_sha256=i["parent_checkpoint_sha256"],
                        final_checkpoint_sha256=r["checkpoint"]["sha256"],
                    )
                )
            a = r["alignment"]
            first = a[0]["metrics"]
            last = a[-1]["metrics"]
            align.append(
                dict(
                    **common,
                    temperature=r["temperature"],
                    epsilon_schedule=[0.15, 0.02],
                    initial_hard_match_probability=first["hard_match_probability"],
                    final_hard_match_probability=last["hard_match_probability"],
                    initial_normalized_entropy=first["normalized_entropy"],
                    final_normalized_entropy=last["normalized_entropy"],
                    hard_path_match_probability=last["hard_path_match_probability"],
                    snapshots=a,
                )
            )
            # Load curves only from verified immutable checkpoints, never model weights into public output.
            from .storage import Store
            from .runner import jobdir

            state = Store(jobdir(output, int(seed), arm), r["identity"]).verified(r["checkpoint"])
            rows = state["curve"]
            g = np.array([v["cell_gradient_norms"] for v in rows])
            counts = np.array([v["exposure"]["stage_cells"] for v in rows])
            action = np.array([v["exposure"]["stage_actions"] for v in rows])
            edges = np.array([v["exposure"]["transitions"] for v in rows]).sum(0)
            matches = np.array([v["exposure"]["matches"] for v in rows]).sum((0, 1))
            align[-1]["empirical_local_action_match"] = float(matches[0] / max(1, matches[1]))
            align[-1]["training_ce_curve"] = [dict(step=v["step"], ce=v["ce"]) for v in rows]
            total = counts.sum((0, 1))
            shares = total / max(1, total.sum())
            exposure.append(
                dict(
                    **common,
                    cells=[
                        dict(
                            cell_id=j,
                            total_invocations=int(total[j]),
                            gradient_update_count=int((g[:, j] > 0).sum()),
                            mean_gradient_norm=float(g[:, j].mean()),
                            stage_usage=counts.sum(0)[:, j].tolist(),
                            unused=bool(total[j] == 0),
                            low_exposure_descriptive=bool(shares[j] < 0.01),
                        )
                        for j in range(6)
                    ],
                    pair_usage=action.sum(0).tolist() if arm[0] == "A" else None,
                    transition_usage=edges.tolist() if arm[0] == "B" else None,
                    transition_probabilities=(edges / edges.sum(-1, keepdims=True).clip(1)).tolist()
                    if arm[0] == "B"
                    else None,
                    self_connection_fraction=float(np.trace(edges) / max(1, edges.sum()))
                    if arm[0] == "B"
                    else None,
                    adjacent_step_cell_call_l1=np.abs(np.diff(counts.sum(1), axis=0))
                    .sum(-1)
                    .tolist(),
                )
            )
            cost.append(
                dict(
                    **common,
                    **r["training_cost"],
                    terminal_evaluation={
                        k: v for k, v in r.get("quality", {}).items() if k != "stories"
                    },
                )
            )
    summary = dict(
        status="COMPLETED" if valid else "PARTIAL",
        identity={
            k: manifest[k] for k in ["experiment_id", "source_sha", "config_sha256", "run_id"]
        },
        scientific_decision=False,
        confirmation="SEALED",
        comparisons=comparisons,
        failures=failures,
        completed_models=len(nll),
    )
    files = {
        "INPUT_VERIFICATION.json": inputs,
        "ROUTING_CALIBRATION.json": fits,
        "TRAINING_ALIGNMENT.json": align,
        "CELL_EXPOSURE.json": exposure,
        "DEVELOPMENT_NLL.json": nll,
        "TRAINING_COST.json": cost,
        "METRICS.json": summary,
    }
    for name, value in files.items():
        atomic(folder / name, value)
    findings = [
        "# R3h — exploratory routing alignment",
        "",
        f"Status: {summary['status']}. Confirmation SEALED.",
        "All comparisons remain within architecture. Three training seeds; historical Development; no independent confirmation.",
        "H/M/T alignment and Cell exposure are descriptive mechanisms, not quality support. No automatic next experiment.",
        "",
    ]
    for k, v in comparisons.items():
        findings.append(
            "%s: %s, mean delta %.8f; seed deltas %s."
            % (k, v["status"], v["mean_delta"], v["deltas"])
        )
    findings += [
        "",
        "Interpretation: combine Development deltas with alignment, coverage and cost; entropy reduction alone does not establish useful learning.",
        "Connection joint updates are not authorized by an alignment metric. Further research requires review of quality and exposure across seeds.",
        "T may fail its fixed 0.80 label-free target and still run diagnostically; no temperature retuning is permitted.",
    ]
    (folder / "SCIENTIFIC-FINDINGS.md").write_text("\n".join(findings) + "\n")
    manifest.update(
        status=summary["status"],
        confirmation="SEALED",
        scientific_decision=False,
        planned_main_updates=12000,
        planned_main_tokens=6144000,
        artifact_files=[
            dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(folder.iterdir())
            if p.is_file() and p.name != "RUN_MANIFEST.json"
        ],
        publication_status="PENDING_LOCAL_FINAL_HANDSHAKE_SEPARATE",
        remote_commit_sha=None,
    )
    atomic(folder / "RUN_MANIFEST.json", manifest)
    return summary
