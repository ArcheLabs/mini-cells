"""Partial-arm aggregation. No success is inferred from completion or training CE."""

import json
from pathlib import Path
import numpy as np
from .common import ARMS, canonical_sha256, file_sha256, write_json


def contrast(a, b, c):
    if a["groups"] != b["groups"] or a["paired_identity"] != b["paired_identity"]:
        raise ValueError("unpaired stories/policy/windows")
    groups = np.asarray(a["groups"])
    keys = np.unique(groups)
    rng = np.random.default_rng(c["evaluation"]["seed"])
    idx = rng.integers(0, len(keys), (c["evaluation"]["bootstrap_replicates"], len(keys)))
    result = {}
    for name, values in [
        ("nll", np.asarray(a["nll"]) - np.asarray(b["nll"])),
        ("cost", np.asarray(a["cost"]) - np.asarray(b["cost"])),
    ]:
        if not np.isfinite(values).all():
            raise ValueError("nonfinite paired evidence")
        cluster = np.array([values[groups == g].mean() for g in keys])
        result[name] = {
            "mean": float(cluster.mean()),
            "ci95": np.quantile(cluster[idx].mean(1), [0.025, 0.975]).tolist(),
        }
    utility = (np.asarray(a["nll"]) - np.asarray(b["nll"])) + c["policy"]["lambda_cost"] * (
        np.asarray(a["cost"]) - np.asarray(b["cost"])
    )
    cluster = np.array([utility[groups == g].mean() for g in keys])
    result["utility"] = {
        "mean": float(cluster.mean()),
        "ci95": np.quantile(cluster[idx].mean(1), [0.025, 0.975]).tolist(),
    }
    return {
        **result,
        "stories": len(keys),
        "scope": "EXPLORATORY_SINGLE_SEED_STORY_BOOTSTRAP_NOT_SEED_REPLICATION",
    }


def seal(path, value):
    write_json(path, value)
    write_json(Path(str(path) + ".sha256.json"), {"sha256": file_sha256(path)})


def verified(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symlink evidence rejected")
    meta = json.loads(Path(str(path) + ".sha256.json").read_text())
    if file_sha256(path) != meta["sha256"]:
        raise ValueError("evidence digest mismatch")
    return json.loads(path.read_text())


def aggregate(c, output, source):
    output = Path(output)
    results = {}
    summaries = {}
    failures = []
    for arm in ARMS:
        folder = output / "arms" / arm
        last = 0
        identity = None
        if (folder / "latest.json").exists():
            latest = json.loads((folder / "latest.json").read_text())
            identity = latest["identity"]
            if identity["source_commit"] != source or identity["experiment_id"] != "CLM-001-R2":
                raise ValueError("foreign checkpoint identity")
            if file_sha256(folder / latest["file"]) != latest["sha256"]:
                raise ValueError("checkpoint checksum")
            last = int(latest["file"].split("-")[2].split(".")[0])
        if (folder / "metrics.json").exists():
            results[arm] = verified(folder / "metrics.json")
        failure = verified(folder / "failure.json") if (folder / "failure.json").exists() else None
        if failure:
            failures.append({"arm": arm, **failure})
        value = results.get(arm, {})
        progress = verified(folder / "progress.json") if (folder / "progress.json").exists() else {}
        partial_quality = {}
        diagnostics = sorted(
            path
            for path in folder.glob("development-step-*.json")
            if not path.name.endswith(".sha256.json")
        )
        if diagnostics:
            available = [path for path in diagnostics if int(path.stem.split("-")[-1]) <= last]
            if available:
                partial_quality = verified(available[-1])["quality"]
        if value:
            expected = {
                "source_commit": source,
                "experiment_id": c["experiment_id"],
                "arm": arm,
                "seed": c["seed"],
                "config_sha256": canonical_sha256(c),
            }
            if any(value["identity"].get(k) != v for k, v in expected.items()):
                raise ValueError("foreign metrics identity/config")
            if identity is not None and value["identity"] != identity:
                raise ValueError("metrics/checkpoint identity differs")
        summary = {
            "experiment_id": c["experiment_id"],
            "arm": arm,
            "source_sha": source,
            "config_hash": canonical_sha256(c),
            "last_committed_step": last,
            "scope": "DEVELOPMENT_AGGREGATE",
            "complete": bool(value),
            "training_status": "COMPLETE"
            if value or last == c["training"]["steps"]
            else ("FAILED" if failure else "INCOMPLETE"),
            "evaluation_status": "COMPLETE" if value else "INCOMPLETE",
            "progress": progress,
            "last_saved_development_quality": {
                k: v for k, v in partial_quality.items() if k not in ("nll", "cost", "groups")
            },
            "quality": {
                p: {k: v for k, v in metrics.items() if k not in ("nll", "cost", "groups")}
                for p, metrics in value.get("quality", {}).items()
            },
            "shadow_status": value.get("shadow_status", "NOT_RUN"),
            "inference_cost_profile": verified(folder / "cost-profile.json")
            if (folder / "cost-profile.json").exists()
            else {"status": "NOT_PROFILED"},
            "scientific_decision": False,
            "formal_confirmation": False,
        }
        summaries[arm] = summary
        seal(output / ("arm-" + arm + "-metrics-summary.json"), summary)

    def compare(stage):
        if len(results) != 2:
            return {"status": "PARTIAL_OR_NOT_TRAINED"}
        a = results[ARMS[0]]
        b = results[ARMS[1]]
        qa = a["pre_admission"]["quality"] if stage == "pre" else a["quality"]["greedy"]
        qb = b["pre_admission"]["quality"] if stage == "pre" else b["quality"]["greedy"]
        return contrast(qa, qb, c)

    pre = compare("pre")
    final = compare("final")
    uniform = {"status": "NOT_AVAILABLE"}
    if "QF-R2" in results and "uniform" in results["QF-R2"].get("quality", {}):
        quality = results["QF-R2"]["quality"]
        # Different execution policies, identical story windows. Pairing is by story, not policy identity.
        baseline = {**quality["uniform"], "paired_identity": quality["greedy"]["paired_identity"]}
        uniform = contrast(quality["greedy"], baseline, c)
    from .analysis import decide

    decision = decide(c, results, pre, uniform)
    q = {
        arm: {k: v for k, v in r.get("q_calibration", {}).items() if k != "records"}
        for arm, r in results.items()
    }
    ledger = {arm: r.get("resource_ledger", {}) for arm, r in results.items()}
    shadow = {arm: r.get("shadow_summary", []) for arm, r in results.items()}
    stopctx = {arm: s["quality"] for arm, s in summaries.items()}
    status = "COMPLETE" if len(results) == 2 else ("PARTIAL" if results else "INCOMPLETE")
    report = {
        "experiment_id": c["experiment_id"],
        "source_sha": source,
        "config_hash": canonical_sha256(c),
        "complete": len(results) == 2,
        "training_status": status,
        "arms": summaries,
        "oracle_status": {arm: v.get("oracle_status", "UNREVIEWED") for arm, v in q.items()},
        **decision,
        "qf_vs_own_uniform": uniform,
        "integrated_system_fully_validated": False,
        "scientific_decision": False,
        "formal_confirmation": False,
        "status": "NOT_TRAINED" if not results and not failures else status,
    }
    files = {
        "combined-report.json": report,
        "run-summary.json": report,
        "pre-admission-contrast.json": pre,
        "final-system-contrast.json": final,
        "q-calibration-summary.json": q,
        "stop-ctx-path-summary.json": stopctx,
        "shadow-candidate-summary.json": shadow,
        "resource-ledger.json": ledger,
        "failure-index.json": {"failures": failures},
    }
    for name, value in files.items():
        envelope = {
            "experiment_id": c["experiment_id"],
            "source_sha": source,
            "config_hash": canonical_sha256(c),
            "scope": "DEVELOPMENT_AGGREGATE",
            "complete": len(results) == 2,
            "arm": "PAIRED",
            "last_committed_step": {arm: v["last_committed_step"] for arm, v in summaries.items()},
            "scientific_decision": False,
            "formal_confirmation": False,
        }
        seal(output / name, {**envelope, **value})
    return report
