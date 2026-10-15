"""Paired story-bootstrap exploratory contrasts, no confirmation/scientific win."""

import json
from pathlib import Path
from exp006.reuse import canonical_sha256
from .io import write_json
from pd001.analysis import paired_interval
from .config import ARMS, counts, source_sha


def report(c, work):
    work = Path(work)
    rows = {}
    missing = []
    params = counts(c)
    for arm in ARMS:
        path = work / "runs" / arm / "metrics.json"
        if not path.exists():
            missing.append(arm)
            continue
        row = json.loads(path.read_text())
        if row.get("status") != "DEVELOPMENT_COMPLETED":
            missing.append(arm)
            continue
        if row["identity"]["arm"] != arm or row["identity"]["experiment_id"] != c["experiment_id"]:
            raise ValueError("wrong experiment/arm result")
        if (
            row["identity"]["config_sha256"] != canonical_sha256(c)
            or row["identity"]["source_sha"] != source_sha()
        ):
            raise ValueError("source/config result mismatch")
        if row["train_tokens"] != c["training"]["tokens"]:
            raise ValueError("unequal target token budget")
        row["final_dev_nll"] = row["quality"]["greedy"]["nll"]
        rows[arm] = row
    contrasts = {
        "system_delta_nll_s16_minus_m0": None,
        "depth_delta_nll_s16_minus_s8": None,
        "paired_story_bootstrap": None,
    }
    decision = "INCOMPLETE"
    if not missing:
        identities = [rows[a]["identity"] for a in ARMS]
        for key in ("asset_sha256", "tokenizer_sha256", "evaluation_sha256", "seed", "target_tokens"):
            if len({i[key] for i in identities}) != 1:
                raise ValueError("unmatched " + key)
        q = {a: rows[a]["quality"]["greedy"] for a in ARMS}
        if any(
            (v["window_ids"], v["groups"]) != (q["M0-TINY"]["window_ids"], q["M0-TINY"]["groups"])
            for v in q.values()
        ):
            raise ValueError("unpaired development windows")
        system = q["CLM-S16"]["nll"] - q["M0-TINY"]["nll"]
        depth = q["CLM-S16"]["nll"] - q["CLM-S8"]["nll"]
        boot = {
            name: paired_interval(
                q[a]["window_nll"],
                q[b]["window_nll"],
                q[a]["groups"],
                c["evaluation"]["bootstrap_seed"],
                c["evaluation"]["bootstrap_replicates"],
            )
            for name, a, b in (("system", "M0-TINY", "CLM-S16"), ("depth", "CLM-S8", "CLM-S16"))
        }
        contrasts.update(
            system_delta_nll_s16_minus_m0=system,
            depth_delta_nll_s16_minus_s8=depth,
            paired_story_bootstrap=boot,
        )
        baseline = rows["M0-TINY"]
        curve = json.loads((work / "runs/M0-TINY/curve.json").read_text())
        history = json.loads((work / "runs/M0-TINY/history.json").read_text())
        # Explicit exploratory quality-anchor check; no hidden tuning or noninferiority claim.
        train_drop = history[0]["ce"] - sum(r["ce"] for r in history[-min(20, len(history)) :]) / min(
            20, len(history)
        )
        dev_drop = baseline["initial_quality"]["greedy"]["nll"] - q["M0-TINY"]["nll"]
        stable = bool(train_drop >= 0.1 and dev_drop >= 0.1 and len(curve) > 0)
        if not stable:
            decision = "UNDERPOWERED_BASELINE"
        elif system <= 0.02:
            decision = "PROMISING_DEVELOPMENT_SIGNAL"
        elif boot["depth"]["ci95_two_sided"][1] < 0:
            decision = "DEPTH_SIGNAL_ONLY"
        elif boot["depth"]["ci95_two_sided"][0] > 0:
            decision = "LONG_PATH_OPTIMIZATION_RISK"
        else:
            decision = "NO_DEPTH_SIGNAL_YET"
        if c["training"]["steps"] < 500:
            decision = "ENGINEERING_SMOKE_ONLY"
        elif stable and all(
            rows[a]["initial_quality"]["greedy"]["nll"] - q[a]["nll"] < 0.1 for a in ("CLM-S8", "CLM-S16")
        ):
            decision = "JOINT_LEARNING_BOTTLENECK_CANDIDATE"
    first = next(iter(rows.values()), {}).get("identity", {})
    result = {
        "experiment_id": c["experiment_id"],
        "status": "INCOMPLETE" if missing else "DEVELOPMENT_COMPLETED",
        "source_sha": source_sha(),
        "config_sha256": canonical_sha256(c),
        "asset_manifest_sha256": first.get("asset_sha256"),
        "tokenizer_sha256": first.get("tokenizer_sha256"),
        "development_seed": c["seed"],
        "arms": {
            a: rows.get(
                a,
                {
                    "total_params": params[a]["total_params"],
                    "final_dev_nll": None,
                    "train_tokens": None,
                    "wall_seconds": None,
                },
            )
            for a in ARMS
        },
        "contrasts": contrasts,
        "path_diagnostics": {},
        "cost_and_latency": {},
        "decision": decision,
        "missing_arms": missing,
        "research_claim": "ENGINEERING_SMOKE_ONLY"
        if c.get("engineering_fixture") or c["training"]["steps"] == 50
        else "EXPLORATORY_ONLY",
        "synthetic_data": bool(c.get("engineering_fixture")),
        "scientific_decision": False,
        "notes": [
            "one development seed; story bootstrap does not estimate between-seed variance",
            "same tokens do not mean same FLOPs or architecture; input memory differs from deep MoE attention",
            "actor loss magnitude and uniform routing are not evidence of learned useful paths",
        ],
    }
    for a in ARMS:
        folder = work / "runs" / a
        for file, key in (
            ("hardware-profile.json", "cost_and_latency"),
            ("interventions.json", "path_diagnostics"),
        ):
            if (folder / file).exists():
                evidence = json.loads((folder / file).read_text())
                if a not in rows or evidence.get("identity") != rows[a]["identity"]:
                    raise ValueError("offline profile/intervention identity mismatch")
                result[key][a] = evidence
        history_path = folder / "history.json"
        if a in rows and history_path.exists():
            history = json.loads(history_path.read_text())
            result["path_diagnostics"].setdefault(a, {})["sampled_training_diagnostics"] = [
                {
                    k: v
                    for k, v in row.items()
                    if k.startswith("grad_norm_")
                    or k in ("step", "update_norms", "path_position_diagnostics", "routing", "nonfinite")
                }
                for row in history
                if row.get("diagnostics_sampled")
            ]

    write_json(work / "report.json", result)
    table = [
        "# FAST-001 development report",
        "",
        f"Status: {result['status']}; decision: {decision}. Exploratory only.",
        "",
        "| Arm | Params | Tokens | Dev NLL | PPL | Train s | Forward GEMM/token | Decode ms | Peak bytes | Self-loop ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in ARMS:
        r = rows.get(a, {})
        q = r.get("quality", {}).get("greedy", {})
        cost = result["cost_and_latency"].get(a, {})
        table.append(
            f"| {a} | {params[a]['total_params']} | {r.get('train_tokens')} | {q.get('nll')} | {q.get('ppl')} | "
            f"{r.get('train_seconds')} | {cost.get('cost', {}).get('observed_gemm_flops_per_token')} | "
            f"{cost.get('decode_ms_p50')} | {r.get('peak_gpu_bytes')} | {cost.get('routing', {}).get('self_loop_ratio')} |"
        )
    table += ["", "No formal scientific support or independent confirmation is granted."]
    (work / "REPORT.md").write_text("\n".join(table) + "\n")
    (work / "DECISION.md").write_text(
        "# Development decision\n\n"
        + decision
        + "\n\nReview quality, context usage, self-loops, state/gradient curves and measured cost before any new budget.\n"
    )
    return result
