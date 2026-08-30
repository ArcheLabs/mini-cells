from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_proposal_utility_discovery as e019_analysis  # noqa: E402
import run_proposal_utility_discovery_worker as e019  # noqa: E402
from minicells.language_recruitment_response import (  # noqa: E402
    FULL_BENEFIT_MIN,
    LOW_PROBE_MAX,
    RECRUITMENT_GRID,
    normalized_regret,
    summarize_response,
)
from minicells.language_utility_skill_data import SKILL_FAMILIES  # noqa: E402


OUT = ROOT / "results" / "recruitment-response-curves-v1"
SOURCE = ROOT / "results" / "proposal-utility-discovery-stable-v1"
CHECKPOINT_DIR = SOURCE / "checkpoints"
WORKER = ROOT / "scripts" / "run_recruitment_response_curves_worker.py"
CANDIDATE_FAMILIES = (*SKILL_FAMILIES, e019.RANDOM_CONTROL)
N_REPLICATES = e019.N_REPLICATES
CLOSED_INVARIANCE_ATOL = 2e-6
FAMILY_PASS_SPEARMAN = 0.50
FAMILY_PASS_AUC = 0.75
FAMILY_PASS_TOP1 = 0.60
FAMILY_PASS_REGRET = 0.35
GENERAL_PROBE_FAMILIES_MIN = 4
BARRIER_FAMILIES_MIN = 3


def validate_source() -> dict[str, object]:
    decision_path = SOURCE / "decision.json"
    manifest_path = SOURCE / "checkpoint-manifest.json"
    if not decision_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Experiment 019b requires the local stable-019 results and checkpoints. "
            "Run scripts/run_proposal_utility_discovery_stable.py first in this Kaggle workspace."
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if decision.get("status") != "UTILITY_ORACLE_INCONSISTENT":
        raise RuntimeError(
            f"019b is preregistered as the follow-up to UTILITY_ORACLE_INCONSISTENT, got {decision.get('status')!r}"
        )
    if int(manifest.get("file_count", -1)) != int(manifest.get("expected_file_count", -2)):
        raise RuntimeError("stable-019 checkpoint manifest is incomplete")
    missing = [path.name for path in CHECKPOINT_DIR.glob("*.tmp")]
    if missing:
        raise RuntimeError(f"checkpoint directory contains incomplete temporary files: {missing}")
    return {"decision": decision, "manifest": manifest}


def _worker_complete(replicate: int) -> bool:
    meta = OUT / f"r{replicate}-worker.json"
    rows = OUT / f"r{replicate}-response-observations.csv.gz"
    if not meta.is_file() or not rows.is_file() or rows.stat().st_size == 0:
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return payload.get("format") == "minicells.recruitment-response-worker.v1" and int(payload.get("replicate", -1)) == replicate


def run_workers() -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 019b requires a Kaggle GPU accelerator")
    gpu_count = min(2, available)
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [r for r in range(N_REPLICATES) if not _worker_complete(r)]
    if not missing:
        print("reusing complete 019b response workers; no model sweep rerun")
        return gpu_count
    for start in range(0, len(missing), gpu_count):
        group = missing[start : start + gpu_count]
        active = []
        for local_gpu, replicate in enumerate(group):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
            log = OUT / f"r{replicate}.log"
            handle = log.open("w", encoding="utf-8")
            cmd = [
                sys.executable,
                str(WORKER),
                "--replicate",
                str(replicate),
                "--checkpoint-dir",
                str(CHECKPOINT_DIR),
                "--output-dir",
                str(OUT),
            ]
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active.append((replicate, local_gpu, process, log, handle))
            print(f"started 019b r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- 019b r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} exited {code}; see {log}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def collect() -> pd.DataFrame:
    frames = [pd.read_csv(OUT / f"r{replicate}-response-observations.csv.gz") for replicate in range(N_REPLICATES)]
    observations = pd.concat(frames, ignore_index=True)
    expected = N_REPLICATES * len(SKILL_FAMILIES) * len(CANDIDATE_FAMILIES) * e019.UTILITY_EXAMPLES_PER_FAMILY * len(RECRUITMENT_GRID)
    if len(observations) != expected:
        raise RuntimeError(f"response row count mismatch: {len(observations)} != {expected}")
    numeric = observations[["recruitment", "loss", "loss_closed", "value"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("019b response observations contain non-finite values")
    zero = observations.loc[np.isclose(observations["recruitment"], 0.0)]
    if float(zero["value"].abs().max()) > 1e-7:
        raise RuntimeError("recruitment=0 must have exactly zero intervention value")
    closed_range = zero.groupby(["replicate", "example", "input_family"])["loss_closed"].agg(lambda x: float(x.max() - x.min()))
    max_closed_range = float(closed_range.max())
    if max_closed_range > CLOSED_INVARIANCE_ATOL:
        raise RuntimeError(
            f"closed Phase-1 loss depends on candidate tissue: max range {max_closed_range} > {CLOSED_INVARIANCE_ATOL}"
        )
    observations.to_csv(OUT / "response-observations.csv.gz", index=False, compression="gzip")
    (OUT / "invariants.json").write_text(json.dumps({
        "format": "minicells.recruitment-response-invariants.v1",
        "rows": len(observations),
        "expected_rows": expected,
        "all_finite": True,
        "zero_value_max_abs": float(zero["value"].abs().max()),
        "closed_loss_candidate_max_range": max_closed_range,
        "closed_loss_candidate_atol": CLOSED_INVARIANCE_ATOL,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return observations


def make_response_curves(observations: pd.DataFrame) -> pd.DataFrame:
    curves = observations.groupby(
        ["replicate", "input_family", "candidate_family", "candidate_kind", "matching_family", "recruitment"],
        as_index=False,
    ).agg(
        mean_value=("value", "mean"),
        median_value=("value", "median"),
        positive_fraction=("value", lambda x: float((x > 0.0).mean())),
        mean_loss=("loss", "mean"),
    )
    curves.to_csv(OUT / "response-curves.csv", index=False)
    return curves


def make_example_summary(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["replicate", "example", "input_family", "candidate_family", "candidate_kind", "matching_family"]
    for key, group in observations.groupby(keys, sort=False):
        summary = summarize_response(group["recruitment"].to_numpy(float), group["value"].to_numpy(float))
        rows.append({
            **dict(zip(keys, key)),
            "full_value": summary.full_value,
            "best_value": summary.best_value,
            "best_recruitment": summary.best_recruitment,
            "min_small_value": summary.min_small_value,
            "first_positive_recruitment": summary.first_positive_recruitment,
            "full_beneficial": int(summary.full_beneficial),
            "activation_barrier": int(summary.activation_barrier),
            "nonmonotonic": int(summary.nonmonotonic),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "response-example-summary.csv", index=False)
    return frame


def make_curve_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["replicate", "input_family", "candidate_family", "candidate_kind", "matching_family"]
    for key, group in curves.groupby(keys, sort=False):
        summary = summarize_response(group["recruitment"].to_numpy(float), group["mean_value"].to_numpy(float))
        rows.append({
            **dict(zip(keys, key)),
            "full_value": summary.full_value,
            "best_value": summary.best_value,
            "best_recruitment": summary.best_recruitment,
            "min_small_value": summary.min_small_value,
            "first_positive_recruitment": summary.first_positive_recruitment,
            "full_beneficial": int(summary.full_beneficial),
            "activation_barrier": int(summary.activation_barrier),
            "nonmonotonic": int(summary.nonmonotonic),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "response-curve-summary.csv", index=False)
    return frame


def _selection_metrics(frame: pd.DataFrame, score_column: str) -> tuple[float, float]:
    correct = 0
    regrets: list[float] = []
    groups = 0
    for _, group in frame.groupby(["replicate", "example", "input_family"], sort=False):
        full = group["full_value"].to_numpy(float)
        score = group[score_column].to_numpy(float)
        best_pos = int(np.argmax(full))
        chosen_pos = int(np.argmax(score))
        correct += int(group.iloc[best_pos]["candidate_family"] == group.iloc[chosen_pos]["candidate_family"])
        regrets.append(normalized_regret(float(full[best_pos]), float(full[chosen_pos])))
        groups += 1
    return correct / max(groups, 1), float(np.median(regrets)) if regrets else float("nan")


def make_probe_metrics(observations: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    base = summary[["replicate", "example", "input_family", "candidate_family", "full_value", "full_beneficial"]]
    rows: list[dict[str, object]] = []
    for recruitment in RECRUITMENT_GRID:
        if recruitment <= 0.0 or recruitment >= 1.0:
            continue
        probe = observations.loc[np.isclose(observations["recruitment"], recruitment), [
            "replicate", "example", "input_family", "candidate_family", "value"
        ]].rename(columns={"value": "probe_value"})
        scored = base.merge(probe, on=["replicate", "example", "input_family", "candidate_family"], validate="one_to_one")
        for family in ("ALL", *SKILL_FAMILIES):
            subset = scored if family == "ALL" else scored.loc[scored["input_family"] == family]
            full = subset["full_value"].to_numpy(float)
            probe_value = subset["probe_value"].to_numpy(float)
            beneficial = subset["full_beneficial"].to_numpy(bool)
            spearman = e019_analysis._spearman(full, probe_value)
            auc = e019_analysis._auc(beneficial, probe_value)
            sign_agreement = float(((full > 0.0) == (probe_value > 0.0)).mean())
            top1, regret = _selection_metrics(subset, "probe_value")
            passed = bool(
                family != "ALL"
                and spearman >= FAMILY_PASS_SPEARMAN
                and np.isfinite(auc) and auc >= FAMILY_PASS_AUC
                and top1 >= FAMILY_PASS_TOP1
                and regret <= FAMILY_PASS_REGRET
            )
            rows.append({
                "recruitment": float(recruitment),
                "input_family": family,
                "rows": len(subset),
                "spearman_vs_full": spearman,
                "auc_full_beneficial": auc,
                "sign_agreement_vs_full": sign_agreement,
                "top1_selection_accuracy": top1,
                "median_normalized_regret": regret,
                "family_pass": int(passed),
            })
    metrics = pd.DataFrame(rows)
    family_counts = metrics.loc[metrics["input_family"] != "ALL"].groupby("recruitment")["family_pass"].sum()
    metrics["family_pass_count_at_recruitment"] = metrics["recruitment"].map(family_counts).astype(int)
    metrics.to_csv(OUT / "probe-metrics.csv", index=False)
    return metrics


def barrier_summary(curve_summary: pd.DataFrame) -> pd.DataFrame:
    matching = curve_summary.loc[
        (curve_summary["matching_family"] == 1) & (curve_summary["candidate_kind"] == "trained")
    ].copy()
    rows = []
    for family in SKILL_FAMILIES:
        part = matching.loc[matching["input_family"] == family]
        barrier_count = int(part["activation_barrier"].sum())
        beneficial_count = int(part["full_beneficial"].sum())
        rows.append({
            "input_family": family,
            "replicates": len(part),
            "full_beneficial_replicates": beneficial_count,
            "activation_barrier_replicates": barrier_count,
            "family_barrier_supported": int(barrier_count >= 2),
            "mean_full_value": float(part["full_value"].mean()),
            "mean_min_small_value": float(part["min_small_value"].mean()),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "barrier-summary.csv", index=False)
    return frame


def plot_results(curves: pd.DataFrame, probe_metrics: pd.DataFrame) -> None:
    matching = curves.loc[(curves["matching_family"] == 1) & (curves["candidate_kind"] == "trained")]
    mean_curves = matching.groupby(["input_family", "recruitment"], as_index=False)["mean_value"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for family in SKILL_FAMILIES:
        part = mean_curves.loc[mean_curves["input_family"] == family]
        ax.plot(part["recruitment"], part["mean_value"], marker="o", label=family)
    ax.axhline(0.0, linewidth=1)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("Recruitment e")
    ax.set_ylabel("Mean value L(0)-L(e)")
    ax.set_title("019b matching-tissue recruitment response")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "matching-response-curves.png", dpi=180)
    plt.close(fig)

    counts = probe_metrics.loc[probe_metrics["input_family"] == "ALL", ["recruitment", "family_pass_count_at_recruitment"]].drop_duplicates()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(v) for v in counts["recruitment"]], counts["family_pass_count_at_recruitment"])
    ax.axhline(GENERAL_PROBE_FAMILIES_MIN, linewidth=1)
    ax.set_xlabel("Fixed probationary recruitment")
    ax.set_ylabel("Held-out-like family metric pass count / 6")
    ax.set_title("019b finite-probe generality")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "probe-family-pass-count.png", dpi=180)
    plt.close(fig)

    overall = probe_metrics.loc[probe_metrics["input_family"] == "ALL"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(overall["recruitment"], overall["top1_selection_accuracy"], marker="o", label="top-1 accuracy")
    ax.plot(overall["recruitment"], overall["spearman_vs_full"], marker="o", label="Spearman vs full")
    ax.set_xscale("log")
    ax.set_xlabel("Probationary recruitment")
    ax.set_ylabel("Metric")
    ax.set_title("019b probe prediction of full recruitment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "probe-predicts-full.png", dpi=180)
    plt.close(fig)


def make_decision(
    source: dict[str, object],
    barrier: pd.DataFrame,
    probe_metrics: pd.DataFrame,
    curve_summary: pd.DataFrame,
    gpu_count: int,
) -> dict[str, object]:
    barrier_families = int(barrier["family_barrier_supported"].sum())
    matching = curve_summary.loc[(curve_summary["matching_family"] == 1) & (curve_summary["candidate_kind"] == "trained")]
    barrier_curves = int(matching["activation_barrier"].sum())
    beneficial_matching = int(matching["full_beneficial"].sum())

    low = probe_metrics.loc[
        (probe_metrics["input_family"] == "ALL") & (probe_metrics["recruitment"] <= LOW_PROBE_MAX + 1e-12)
    ].copy()
    low = low.sort_values(
        ["family_pass_count_at_recruitment", "top1_selection_accuracy", "spearman_vs_full"],
        ascending=[False, False, False],
    )
    best = low.iloc[0]
    best_e = float(best["recruitment"])
    best_pass = int(best["family_pass_count_at_recruitment"])
    barrier_confirmed = barrier_families >= BARRIER_FAMILIES_MIN
    finite_probe_supported = best_pass >= GENERAL_PROBE_FAMILIES_MIN
    if barrier_confirmed and finite_probe_supported:
        status = "ACTIVATION_BARRIER_WITH_FINITE_PROBE_SIGNAL"
    elif barrier_confirmed:
        status = "ACTIVATION_BARRIER_WITHOUT_GENERAL_PROBE"
    elif finite_probe_supported:
        status = "FINITE_PROBE_SIGNAL_WITHOUT_COMMON_BARRIER"
    else:
        status = "MIXED_RECRUITMENT_RESPONSE"

    decision = {
        "format": "minicells.recruitment-response-curves.v1",
        "experiment": "MINI Cells Experiment 019b — Recruitment Response Curves",
        "status": status,
        "question": "Do capability tissues exhibit an activation barrier, and can a finite probationary recruitment predict the value of full recruitment better than an infinitesimal closed-boundary derivative?",
        "source": {
            "experiment": "019-stable",
            "required_status": "UTILITY_ORACLE_INCONSISTENT",
            "source_status": source["decision"]["status"],
            "checkpoint_files": int(source["manifest"]["file_count"]),
            "training_performed": False,
        },
        "design": {
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "skill_families": list(SKILL_FAMILIES),
            "candidate_tissues": list(CANDIDATE_FAMILIES),
            "examples_per_family": e019.UTILITY_EXAMPLES_PER_FAMILY,
            "recruitment_grid": list(RECRUITMENT_GRID),
            "value": "V_T(e)=L(e=0)-L(e)",
            "full_beneficial_threshold": FULL_BENEFIT_MIN,
            "barrier_definition": "full-beneficial response with small-e harm exceeding max(0.005 NLL, 5% of full value)",
            "family_barrier_support": "activation barrier in >=2/3 replicate mean curves",
            "common_barrier_families_min": BARRIER_FAMILIES_MIN,
            "finite_probe_search_domain": f"fixed preregistered grid values e <= {LOW_PROBE_MAX}",
            "finite_probe_family_pass": {
                "spearman_min": FAMILY_PASS_SPEARMAN,
                "auc_full_beneficial_min": FAMILY_PASS_AUC,
                "top1_selection_accuracy_min": FAMILY_PASS_TOP1,
                "median_normalized_regret_max": FAMILY_PASS_REGRET,
            },
            "finite_probe_families_min": GENERAL_PROBE_FAMILIES_MIN,
        },
        "results": {
            "matching_full_beneficial_curves": beneficial_matching,
            "matching_activation_barrier_curves": barrier_curves,
            "barrier_supported_families": barrier_families,
            "best_low_probe_recruitment": best_e,
            "best_low_probe_family_pass_count": best_pass,
            "best_low_probe_overall_spearman_vs_full": float(best["spearman_vs_full"]),
            "best_low_probe_overall_top1": float(best["top1_selection_accuracy"]),
            "best_low_probe_overall_regret": float(best["median_normalized_regret"]),
        },
        "interpretation": {
            "barrier": "A confirmed barrier means infinitesimal recruitment is not a valid proxy for whether the coherent tissue is worth waking.",
            "finite_probe": "A common finite probe supports probationary activation as the next recruitment primitive; it does not yet establish a cheap probe implementation.",
            "no_probe": "If barriers exist without a common probe amplitude, recruitment may require adaptive escalation rather than one globally fixed e.",
            "scope": "019b is a checkpoint-only causal response diagnostic. It trains no router, gate, tissue, or new genome parameters.",
        },
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def main() -> int:
    source = validate_source()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source-019-decision.json").write_text(
        json.dumps(source["decision"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gpu_count = run_workers()
    observations = collect()
    curves = make_response_curves(observations)
    example_summary = make_example_summary(observations)
    curve_summary = make_curve_summary(curves)
    probes = make_probe_metrics(observations, example_summary)
    barriers = barrier_summary(curve_summary)
    plot_results(curves, probes)
    decision = make_decision(source, barriers, probes, curve_summary, gpu_count)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
