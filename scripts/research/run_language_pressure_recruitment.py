from __future__ import annotations

import json
import os
import shutil
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
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_data import prepare_tinystories_corpus  # noqa: E402


OUT = ROOT / "results" / "feedback-isolated-pressure-recruitment-v1"
WORKER = ROOT / "scripts" / "run_language_pressure_recruitment_worker.py"
N_REPLICATES = 3
POLICIES = ("N", "P")


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    corpus = prepare_tinystories_corpus(ROOT)
    cache = corpus.tokenizer_path.parent
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    shutil.copy2(cache / "corpus-manifest.json", OUT / "corpus-manifest.json")
    return cache, corpus.manifest


def run_workers(cache: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 018b requires CUDA")
    gpu_count = min(2, available)
    for start in range(0, N_REPLICATES, gpu_count):
        group = list(range(start, min(start + gpu_count, N_REPLICATES)))
        active = []
        for local_gpu, replicate in enumerate(group):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
            log = OUT / f"r{replicate}.log"
            handle = log.open("w", encoding="utf-8")
            cmd = [sys.executable, str(WORKER), "--replicate", str(replicate), "--cache-dir", str(cache), "--output-dir", str(OUT)]
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active.append((replicate, local_gpu, process, log, handle))
            print(f"started r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} exited {code}; see {log}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect() -> dict[str, object]:
    workers = []
    keys = (
        "phase1-checkpoints",
        "structural-events",
        "local-learning",
        "transplantation",
        "localization",
        "recruitment-interventions",
    )
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in keys}
    summary_rows = []
    for replicate in range(N_REPLICATES):
        worker = json.loads((OUT / f"r{replicate}-worker.json").read_text(encoding="utf-8"))
        workers.append(worker)
        phase1 = worker["phase1"]
        for policy in POLICIES:
            summary_rows.append({"replicate": replicate, "policy": policy, **phase1, **worker["policies"][policy]})
        for key in keys:
            frame = _read(OUT / f"r{replicate}-{key}.csv")
            if not frame.empty:
                if "replicate" in frame.columns:
                    frame["replicate"] = frame["replicate"].fillna(replicate).astype(int)
                frames[key].append(frame)
    result: dict[str, object] = {"workers": workers, "summary": pd.DataFrame(summary_rows)}
    result["summary"].to_csv(OUT / "policy-summary.csv", index=False)
    for key, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        result[key.replace("-", "_")] = frame
        frame.to_csv(OUT / f"{key}.csv", index=False)
    return result


def plot_learning(local: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in POLICIES:
        selected = local.loc[local["policy"] == policy]
        for replicate in range(N_REPLICATES):
            run = selected.loc[selected["replicate"] == replicate]
            ax.plot(run["step"], run["skill_nll"], alpha=0.25)
        mean = selected.groupby("step")["skill_nll"].mean()
        ax.plot(mean.index, mean.values, linewidth=2.5, label=policy)
    ax.set_xlabel("local learning step")
    ax.set_ylabel("REVERSE_INC validation NLL")
    ax.set_title("018b skill acquisition: novelty vs feedback-isolated pressure")
    ax.legend(title="sensor")
    fig.tight_layout()
    fig.savefig(OUT / "skill-learning-pressure-comparison.png", dpi=180)
    plt.close(fig)


def plot_retention(local: pd.DataFrame, summary: pd.DataFrame) -> None:
    base = summary.set_index(["replicate", "policy"])["base_language_nll"].to_dict()
    frame = local.copy()
    frame["language_ratio"] = [row.language_nll / base[(int(row.replicate), str(row.policy))] for row in frame.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in POLICIES:
        selected = frame.loc[frame["policy"] == policy]
        for replicate in range(N_REPLICATES):
            run = selected.loc[selected["replicate"] == replicate]
            ax.plot(run["step"], run["language_ratio"], alpha=0.25)
        mean = selected.groupby("step")["language_ratio"].mean()
        ax.plot(mean.index, mean.values, linewidth=2.5, label=policy)
    ax.axhline(1.10, linestyle="--", linewidth=1)
    ax.set_xlabel("local learning step")
    ax.set_ylabel("TinyStories NLL / Phase-1 NLL")
    ax.set_title("018b retained-language interference")
    ax.legend(title="sensor")
    fig.tight_layout()
    fig.savefig(OUT / "language-retention-pressure-comparison.png", dpi=180)
    plt.close(fig)


def plot_recruitment(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.18
    x = np.arange(N_REPLICATES)
    for index, (policy, metric, label) in enumerate((
        ("N", "donor_language_recruitment", "N language"),
        ("N", "donor_skill_recruitment", "N skill"),
        ("P", "donor_language_recruitment", "P language"),
        ("P", "donor_skill_recruitment", "P skill"),
    )):
        values = summary.loc[summary["policy"] == policy].sort_values("replicate")[metric].to_numpy()
        ax.bar(x + (index - 1.5) * width, values, width=width, label=label)
    ax.set_xticks(x, [f"r{i}" for i in range(N_REPLICATES)])
    ax.set_ylabel("mean newborn recruitment")
    ax.set_title("Recruitment selectivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "recruitment-selectivity-pressure.png", dpi=180)
    plt.close(fig)


def plot_pressure(local: pd.DataFrame) -> None:
    selected = local.loc[local["policy"] == "P"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label in (("language_shadow_pressure", "language"), ("skill_shadow_pressure", "skill")):
        mean = selected.groupby("step")[metric].mean()
        ax.plot(mean.index, mean.values, linewidth=2.5, label=label)
    ax.set_xlabel("local learning step")
    ax.set_ylabel("old-only parent pressure")
    ax.set_title("Feedback-isolated computational pressure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pressure-signal-separation.png", dpi=180)
    plt.close(fig)


def plot_interventions(interventions: pd.DataFrame) -> None:
    selected = interventions.loc[interventions["policy"] == "P"]
    piv = selected.pivot(index="replicate", columns="intervention", values="delta_nll")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(N_REPLICATES)
    width = 0.35
    ax.bar(x - width / 2, piv["recruitment_off_skill"].to_numpy(), width=width, label="skill: force off")
    ax.bar(x + width / 2, piv["recruitment_forced_on_language"].to_numpy(), width=width, label="language: force on")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, [f"r{i}" for i in range(N_REPLICATES)])
    ax.set_ylabel("delta NLL")
    ax.set_title("Pressure recruitment causal interventions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pressure-recruitment-causality.png", dpi=180)
    plt.close(fig)


def plot_transfer(summary: pd.DataFrame) -> None:
    p = summary.loc[summary["policy"] == "P"].sort_values("replicate")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(N_REPLICATES)
    ax.bar(x, p["transplant_recovery"].to_numpy())
    ax.axhline(0.50, linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"r{i}" for i in range(N_REPLICATES)])
    ax.set_ylabel("newborn-only transplant recovery")
    ax.set_title("Pressure-gated tissue transplantation")
    fig.tight_layout()
    fig.savefig(OUT / "pressure-transplantation-recovery.png", dpi=180)
    plt.close(fig)


def make_decision(summary: pd.DataFrame, gpu_count: int) -> dict[str, object]:
    paired = []
    for replicate in range(N_REPLICATES):
        group = summary.loc[summary["replicate"] == replicate].set_index("policy")
        n = group.loc["N"]
        p = group.loc["P"]
        paired.append({
            "replicate": replicate,
            "novelty_skill_improvement": float(n["skill_improvement"]),
            "pressure_skill_improvement": float(p["skill_improvement"]),
            "pressure_over_novelty_skill_improvement": float(p["skill_improvement"] / max(1e-12, n["skill_improvement"])),
            "novelty_language_ratio": float(n["donor_language_ratio"]),
            "pressure_language_ratio": float(p["donor_language_ratio"]),
            "pressure_recipient_language_ratio": float(p["recipient_language_ratio"]),
            "pressure_transplant_recovery": float(p["transplant_recovery"]),
            "pressure_skill_recruitment": float(p["donor_skill_recruitment"]),
            "pressure_language_recruitment": float(p["donor_language_recruitment"]),
            "pressure_recruitment_selectivity": float(p["recruitment_selectivity"]),
            "pressure_recruitment_gap": float(p["recruitment_gap"]),
            "pressure_recruitment_causal_fraction": float(p["recruitment_causal_fraction"]),
            "pressure_force_on_language_delta_nll": float(p["force_on_language_delta_nll"]),
            "pressure_newborn_count": int(p["newborn_count"]),
            "pressure_base_memory_drift": float(p["base_memory_drift"]),
            "pressure_tissue_causal_fraction": float(p["tissue_causal_fraction"]),
            "feedback_sensor_max_delta": float(p["feedback_sensor_max_delta"]),
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(OUT / "paired-policy-comparisons.csv", index=False)

    skill_reps = int(((paired_frame["pressure_skill_improvement"] > 0) & (paired_frame["pressure_over_novelty_skill_improvement"] >= 0.80)).sum())
    retention_reps = int((paired_frame["pressure_language_ratio"] <= 1.10).sum())
    better_reps = int((paired_frame["pressure_language_ratio"] < paired_frame["novelty_language_ratio"]).sum())
    stable_reps = int((paired_frame["pressure_base_memory_drift"] <= 1e-6).sum())
    one_cell_reps = int((paired_frame["pressure_newborn_count"] == 1).sum())
    selective_reps = int(((paired_frame["pressure_recruitment_selectivity"] >= 2.0) & (paired_frame["pressure_recruitment_gap"] >= 0.20)).sum())
    recruitment_causal_reps = int((paired_frame["pressure_recruitment_causal_fraction"] >= 0.50).sum())
    closure_reps = int((paired_frame["pressure_force_on_language_delta_nll"] >= 0.05).sum())
    isolated_reps = int((paired_frame["feedback_sensor_max_delta"] <= 1e-7).sum())
    tissue_reps = int((paired_frame["pressure_tissue_causal_fraction"] >= 0.50).sum())
    transfer_reps = int(((paired_frame["pressure_transplant_recovery"] >= 0.50) & (paired_frame["pressure_recipient_language_ratio"] <= 1.10)).sum())

    flags = {
        "skill_preserved": skill_reps >= 2,
        "language_retention": retention_reps >= 2,
        "retention_improves_over_novelty": better_reps >= 2,
        "old_tissue_stable": stable_reps == N_REPLICATES,
        "exactly_one_newborn": one_cell_reps == N_REPLICATES,
        "pressure_recruitment_selective": selective_reps >= 2,
        "recruitment_causally_required_for_skill": recruitment_causal_reps >= 2,
        "conditional_closure_causally_protects_language": closure_reps >= 2,
        "sensor_feedback_isolated": isolated_reps == N_REPLICATES,
        "new_tissue_causally_used": tissue_reps >= 2,
        "pressure_tissue_transplantation": transfer_reps >= 2,
    }
    if all(flags.values()):
        status = "FEEDBACK_ISOLATED_PRESSURE_RECRUITMENT_SIGNAL"
    elif flags["pressure_recruitment_selective"] and not flags["language_retention"]:
        status = "PRESSURE_SELECTIVE_WITHOUT_RETENTION"
    elif flags["language_retention"] and not flags["pressure_recruitment_selective"]:
        status = "RETENTION_WITHOUT_PRESSURE_SELECTIVITY"
    elif sum(bool(value) for value in flags.values()) >= 6:
        status = "PARTIAL_FEEDBACK_ISOLATED_PRESSURE_SIGNAL"
    else:
        status = "NO_FEEDBACK_ISOLATED_PRESSURE_SIGNAL"

    decision = {
        "format": "minicells.feedback-isolated-pressure-recruitment.v1",
        "experiment": "MINI Cells Experiment 018b — Feedback-Isolated Pressure Recruitment",
        "question": "Can an old-only local computational-pressure sensor recruit a one-cell skill tissue selectively without newborn-to-sensor feedback?",
        "design": {
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "same_phase1_checkpoint_within_replicate": True,
            "policies": {
                "N": "Experiment-018 state-novelty recruitment, capped to exactly one newborn",
                "P": "feedback-isolated old-only reaction+diffusion pressure recruitment, capped to exactly one newborn",
            },
            "central_router": False,
            "task_label_used_by_recruitment": False,
            "newborn_feedback_enters_pressure_sensor": False,
        },
        "pre_registered_signal": {
            "pressure_skill_fraction_of_novelty_min": 0.80,
            "donor_language_ratio_max": 1.10,
            "base_memory_drift_max": 1e-6,
            "newborn_cell_count": 1,
            "skill_to_language_recruitment_ratio_min": 2.0,
            "skill_minus_language_recruitment_min": 0.20,
            "recruitment_causal_fraction_min": 0.50,
            "forced_on_language_delta_nll_min": 0.05,
            "feedback_sensor_max_delta_max": 1e-7,
            "transplant_recovery_min": 0.50,
            "recipient_language_ratio_max": 1.10,
            "minimum_replicates": 2,
        },
        "results": {
            "skill_preserved_replicates": skill_reps,
            "language_retention_replicates": retention_reps,
            "retention_better_replicates": better_reps,
            "selective_recruitment_replicates": selective_reps,
            "recruitment_causal_replicates": recruitment_causal_reps,
            "language_protection_causal_replicates": closure_reps,
            "feedback_isolated_replicates": isolated_reps,
            "transfer_replicates": transfer_reps,
            "mean_novelty_language_ratio": float(paired_frame["novelty_language_ratio"].mean()),
            "mean_pressure_language_ratio": float(paired_frame["pressure_language_ratio"].mean()),
            "mean_pressure_skill_recruitment": float(paired_frame["pressure_skill_recruitment"].mean()),
            "mean_pressure_language_recruitment": float(paired_frame["pressure_language_recruitment"].mean()),
            "mean_pressure_transplant_recovery": float(paired_frame["pressure_transplant_recovery"].mean()),
            "pass_flags": flags,
        },
        "status": status,
        "scope": {
            "claim": "same-checkpoint one-newborn causal comparison of feedback-contaminated novelty sensing versus feedback-isolated local computational pressure",
            "not_claimed": [
                "general task routing",
                "optimal pressure statistic",
                "arbitrary real-world skills",
                "cross-genome transplantation",
                "production compute efficiency",
            ],
        },
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def main() -> int:
    cache, manifest = prepare_corpus()
    gpu_count = run_workers(cache)
    result = collect()
    summary = result["summary"]
    local = result["local_learning"]
    interventions = result["recruitment_interventions"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(local, pd.DataFrame)
    assert isinstance(interventions, pd.DataFrame)

    plot_learning(local)
    plot_retention(local, summary)
    plot_recruitment(summary)
    plot_pressure(local)
    plot_interventions(interventions)
    plot_transfer(summary)
    decision = make_decision(summary, gpu_count)
    task_spec = {
        "format": "minicells.feedback-isolated-pressure-recruitment-task.v1",
        "skill": "REVERSE_INC",
        "local_steps": 200,
        "replicates": 3,
        "max_newborns": 1,
        "pressure": "RMS(old-only reaction + old-only diffusion), calibrated per recurrent step and parent cell",
        "manifest": manifest,
    }
    (OUT / "task-spec.json").write_text(json.dumps(task_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
