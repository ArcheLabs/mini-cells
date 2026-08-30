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
sys.path.insert(0, str(ROOT / "scripts"))

import run_proposal_utility_discovery_worker as e019  # noqa: E402
from minicells.language_data import prepare_tinystories_corpus  # noqa: E402
from minicells.language_tissue_specificity import (  # noqa: E402
    CAUSAL_FRACTION_MIN,
    EXAMPLE_TOP1_MIN,
    GENERAL_FAMILIES_MIN,
    REPLICATE_TOP1_MIN,
    RETENTION_RATIO_MAX,
    SPECIFICITY_NORM_MIN,
    TRANSPLANT_RECOVERY_MIN,
    TISSUE_ARMS,
    family_pass,
    summarize_specificity,
)
from minicells.language_utility_skill_data import SKILL_FAMILIES  # noqa: E402


OUT = ROOT / "results" / "capability-tissue-specificity-v1"
STABLE_019 = ROOT / "results" / "proposal-utility-discovery-stable-v1"
SOURCE_CHECKPOINT_DIR = STABLE_019 / "checkpoints"
WORKER = ROOT / "scripts" / "run_capability_tissue_specificity_worker.py"
N_REPLICATES = e019.N_REPLICATES
BASELINE_CANDIDATES = tuple(SKILL_FAMILIES)


def _baseline_paths() -> tuple[Path, Path | None]:
    local = ROOT / "results" / "recruitment-response-curves-v1"
    artifact = ROOT / "artifacts" / "experiments" / "019b-recruitment-response-curves"
    if (local / "response-curve-summary.csv").is_file():
        decision = local / "decision.json"
        return local / "response-curve-summary.csv", decision if decision.is_file() else None
    if (artifact / "response-curve-summary.csv").is_file():
        decision = artifact / "decision.json"
        return artifact / "response-curve-summary.csv", decision if decision.is_file() else None
    raise FileNotFoundError(
        "Experiment 020 requires Experiment 019b response-curve-summary.csv as the zero-training specificity baseline"
    )


def validate_sources() -> dict[str, object]:
    manifest_path = STABLE_019 / "checkpoint-manifest.json"
    if not manifest_path.is_file() or not SOURCE_CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            "Experiment 020 requires the local stable-019 Phase-1 checkpoints. "
            "Run the checkpointed stable 019 in this Kaggle workspace first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("file_count", -1)) != int(manifest.get("expected_file_count", -2)):
        raise RuntimeError("stable-019 checkpoint manifest is incomplete")
    baseline_path, baseline_decision = _baseline_paths()
    decision = json.loads(baseline_decision.read_text(encoding="utf-8")) if baseline_decision else None
    return {
        "checkpoint_manifest": manifest,
        "baseline_path": baseline_path,
        "baseline_decision": decision,
    }


def prepare_corpus() -> Path:
    corpus = prepare_tinystories_corpus(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    manifest = corpus.tokenizer_path.parent / "corpus-manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, OUT / "corpus-manifest.json")
    return corpus.tokenizer_path.parent


def _worker_complete(replicate: int) -> bool:
    meta = OUT / f"r{replicate}-worker.json"
    summary = OUT / f"r{replicate}-donor-summary.csv"
    observations = OUT / f"r{replicate}-specificity-observations.csv.gz"
    if not meta.is_file() or not summary.is_file() or not observations.is_file() or observations.stat().st_size == 0:
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return payload.get("format") == "minicells.capability-tissue-specificity-worker.v1" and int(payload.get("replicate", -1)) == replicate


def run_workers(cache: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 020 requires a GPU accelerator")
    gpu_count = min(2, available)
    missing = [replicate for replicate in range(N_REPLICATES) if not _worker_complete(replicate)]
    if not missing:
        print("reusing complete Experiment 020 workers")
        return gpu_count
    for start in range(0, len(missing), gpu_count):
        group = missing[start : start + gpu_count]
        active = []
        for local_gpu, replicate in enumerate(group):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
            log = OUT / f"r{replicate}.log"
            handle = log.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(WORKER),
                "--replicate", str(replicate),
                "--cache-dir", str(cache),
                "--source-checkpoint-dir", str(SOURCE_CHECKPOINT_DIR),
                "--output-dir", str(OUT),
            ]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active.append((replicate, local_gpu, process, log, handle))
            print(f"started Experiment 020 r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- Experiment 020 r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} exited {code}; see {log}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    observations = []
    for replicate in range(N_REPLICATES):
        summaries.append(pd.read_csv(OUT / f"r{replicate}-donor-summary.csv"))
        observations.append(pd.read_csv(OUT / f"r{replicate}-specificity-observations.csv.gz"))
    donor = pd.concat(summaries, ignore_index=True)
    obs = pd.concat(observations, ignore_index=True)
    expected_obs = N_REPLICATES * len(TISSUE_ARMS) * len(SKILL_FAMILIES) * len(SKILL_FAMILIES) * e019.UTILITY_EXAMPLES_PER_FAMILY
    if len(obs) != expected_obs:
        raise RuntimeError(f"specificity observation count mismatch: {len(obs)} != {expected_obs}")
    numeric = obs[["loss_closed", "loss_full", "full_value"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Experiment 020 observations contain non-finite values")
    closed_range = obs.groupby(["replicate", "arm", "example", "input_family"])["loss_closed"].agg(lambda x: float(x.max() - x.min()))
    max_closed_range = float(closed_range.max())
    if max_closed_range > 2e-6:
        raise RuntimeError(f"closed Phase-1 behavior depends on candidate tissue: {max_closed_range}")
    donor.to_csv(OUT / "donor-summary.csv", index=False)
    obs.to_csv(OUT / "specificity-observations.csv.gz", index=False, compression="gzip")
    (OUT / "invariants.json").write_text(json.dumps({
        "format": "minicells.capability-tissue-specificity-invariants.v1",
        "expected_observations": expected_obs,
        "observations": len(obs),
        "all_finite": True,
        "closed_loss_candidate_max_range": max_closed_range,
        "closed_loss_candidate_atol": 2e-6,
        "old_memory_drift_max": float(donor["base_memory_drift"].max()),
        "autonomous_structure_updates_max": int(donor["autonomous_structure_updates"].max()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return donor, obs


def baseline_specificity(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    source = source.loc[source["candidate_family"].isin(BASELINE_CANDIDATES)].copy()
    rows = []
    for (replicate, input_family), group in source.groupby(["replicate", "input_family"], sort=False):
        group = group.set_index("candidate_family").reindex(SKILL_FAMILIES)
        summary = summarize_specificity(list(SKILL_FAMILIES), group["full_value"].to_numpy(float), input_family)
        rows.append({
            "replicate": replicate,
            "input_family": input_family,
            "matching_value": summary.matching_value,
            "mean_wrong_value": summary.mean_wrong_value,
            "best_wrong_value": summary.best_wrong_value,
            "specificity": summary.specificity,
            "normalized_specificity": summary.normalized_specificity,
            "strict_margin": summary.strict_margin,
            "matching_rank": summary.matching_rank,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "baseline-019b-specificity.csv", index=False)
    return frame


def utility_matrix(observations: pd.DataFrame) -> pd.DataFrame:
    frame = observations.groupby(["arm", "input_family", "candidate_family"], as_index=False)["full_value"].mean()
    frame.to_csv(OUT / "utility-matrix.csv", index=False)
    return frame


def replicate_specificity(observations: pd.DataFrame) -> pd.DataFrame:
    means = observations.groupby(["arm", "replicate", "input_family", "candidate_family"], as_index=False)["full_value"].mean()
    rows = []
    for (arm, replicate, input_family), group in means.groupby(["arm", "replicate", "input_family"], sort=False):
        group = group.set_index("candidate_family").reindex(SKILL_FAMILIES)
        summary = summarize_specificity(list(SKILL_FAMILIES), group["full_value"].to_numpy(float), input_family)
        rows.append({
            "arm": arm,
            "replicate": replicate,
            "input_family": input_family,
            "matching_value": summary.matching_value,
            "mean_wrong_value": summary.mean_wrong_value,
            "best_wrong_value": summary.best_wrong_value,
            "specificity": summary.specificity,
            "normalized_specificity": summary.normalized_specificity,
            "strict_margin": summary.strict_margin,
            "matching_rank": summary.matching_rank,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "replicate-specificity.csv", index=False)
    return frame


def _example_top1(observations: pd.DataFrame, arm: str, family: str) -> float:
    selected = observations.loc[(observations["arm"] == arm) & (observations["input_family"] == family)]
    correct = 0
    groups = 0
    for _, group in selected.groupby(["replicate", "example"], sort=False):
        chosen = group.iloc[int(np.argmax(group["full_value"].to_numpy(float)))]["candidate_family"]
        correct += int(chosen == family)
        groups += 1
    return correct / max(groups, 1)


def family_specificity(replicate: pd.DataFrame, observations: pd.DataFrame, donor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in TISSUE_ARMS:
        for family in SKILL_FAMILIES:
            part = replicate.loc[(replicate["arm"] == arm) & (replicate["input_family"] == family)]
            donor_part = donor.loc[(donor["arm"] == arm) & (donor["family"] == family)]
            normalized = float(part["normalized_specificity"].mean())
            replicate_top1 = int((part["matching_rank"] == 1).sum())
            example_top1 = _example_top1(observations, arm, family)
            specificity_ok = family_pass(normalized, replicate_top1, example_top1)
            integrity_replicates = (
                (donor_part["language_retention_ratio"] <= RETENTION_RATIO_MAX)
                & (donor_part["tissue_causal_fraction"] >= CAUSAL_FRACTION_MIN)
                & (donor_part["transplant_recovery"] >= TRANSPLANT_RECOVERY_MIN)
            )
            integrity_count = int(integrity_replicates.sum())
            rows.append({
                "arm": arm,
                "input_family": family,
                "mean_matching_value": float(part["matching_value"].mean()),
                "mean_wrong_value": float(part["mean_wrong_value"].mean()),
                "mean_best_wrong_value": float(part["best_wrong_value"].mean()),
                "mean_specificity": float(part["specificity"].mean()),
                "mean_normalized_specificity": normalized,
                "mean_strict_margin": float(part["strict_margin"].mean()),
                "replicate_top1_count": replicate_top1,
                "example_top1_accuracy": example_top1,
                "specificity_pass": int(specificity_ok),
                "integrity_replicate_count": integrity_count,
                "integrity_pass": int(integrity_count >= 2),
                "joint_pass": int(specificity_ok and integrity_count >= 2),
                "mean_language_retention_ratio": float(donor_part["language_retention_ratio"].mean()),
                "mean_causal_fraction": float(donor_part["tissue_causal_fraction"].mean()),
                "mean_transplant_recovery": float(donor_part["transplant_recovery"].mean()),
                "mean_skill_improvement": float(donor_part["skill_improvement"].mean()),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "family-specificity.csv", index=False)
    return frame


def geometry_comparison(family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for skill in SKILL_FAMILIES:
        one = family.loc[(family["arm"] == "one-cell") & (family["input_family"] == skill)].iloc[0]
        three = family.loc[(family["arm"] == "three-cell-chain") & (family["input_family"] == skill)].iloc[0]
        rows.append({
            "input_family": skill,
            "one_cell_normalized_specificity": one["mean_normalized_specificity"],
            "three_cell_normalized_specificity": three["mean_normalized_specificity"],
            "delta_normalized_specificity": three["mean_normalized_specificity"] - one["mean_normalized_specificity"],
            "one_cell_example_top1": one["example_top1_accuracy"],
            "three_cell_example_top1": three["example_top1_accuracy"],
            "delta_example_top1": three["example_top1_accuracy"] - one["example_top1_accuracy"],
            "one_cell_skill_improvement": one["mean_skill_improvement"],
            "three_cell_skill_improvement": three["mean_skill_improvement"],
            "one_cell_joint_pass": one["joint_pass"],
            "three_cell_joint_pass": three["joint_pass"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "geometry-comparison.csv", index=False)
    return frame


def plot_matrix(matrix: pd.DataFrame, arm: str, filename: str) -> None:
    part = matrix.loc[matrix["arm"] == arm].pivot(index="input_family", columns="candidate_family", values="full_value")
    part = part.reindex(index=SKILL_FAMILIES, columns=SKILL_FAMILIES)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(part.to_numpy(float), aspect="auto")
    ax.set_xticks(range(len(SKILL_FAMILIES)), SKILL_FAMILIES, rotation=45, ha="right")
    ax.set_yticks(range(len(SKILL_FAMILIES)), SKILL_FAMILIES)
    ax.set_xlabel("candidate tissue")
    ax.set_ylabel("input skill")
    ax.set_title(f"Experiment 020 full utility — {arm}")
    fig.colorbar(image, ax=ax, label="mean V(1)")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_specificity(family: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SKILL_FAMILIES))
    width = 0.35
    one = family.loc[family["arm"] == "one-cell"].set_index("input_family").reindex(SKILL_FAMILIES)
    three = family.loc[family["arm"] == "three-cell-chain"].set_index("input_family").reindex(SKILL_FAMILIES)
    ax.bar(x - width / 2, one["mean_normalized_specificity"], width, label="one-cell")
    ax.bar(x + width / 2, three["mean_normalized_specificity"], width, label="three-cell-chain")
    ax.axhline(SPECIFICITY_NORM_MIN, linewidth=1, linestyle="--")
    ax.set_xticks(x, SKILL_FAMILIES, rotation=30, ha="right")
    ax.set_ylabel("normalized specificity")
    ax.set_title("Capability tissue specificity by skill family")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "specificity-by-family.png", dpi=180)
    plt.close(fig)


def plot_identity(family: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SKILL_FAMILIES))
    width = 0.35
    one = family.loc[family["arm"] == "one-cell"].set_index("input_family").reindex(SKILL_FAMILIES)
    three = family.loc[family["arm"] == "three-cell-chain"].set_index("input_family").reindex(SKILL_FAMILIES)
    ax.bar(x - width / 2, one["example_top1_accuracy"], width, label="one-cell")
    ax.bar(x + width / 2, three["example_top1_accuracy"], width, label="three-cell-chain")
    ax.axhline(EXAMPLE_TOP1_MIN, linewidth=1, linestyle="--")
    ax.set_xticks(x, SKILL_FAMILIES, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("matching tissue top-1 accuracy")
    ax.set_title("Capability identity recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "identity-recovery.png", dpi=180)
    plt.close(fig)


def decide(
    family: pd.DataFrame,
    baseline: pd.DataFrame,
    geometry: pd.DataFrame,
    gpu_count: int,
    sources: dict[str, object],
) -> dict[str, object]:
    counts = {}
    joint_counts = {}
    for arm in TISSUE_ARMS:
        part = family.loc[family["arm"] == arm]
        counts[arm] = int(part["specificity_pass"].sum())
        joint_counts[arm] = int(part["joint_pass"].sum())
    baseline_top1 = int((baseline["matching_rank"] == 1).sum())
    baseline_norm = float(baseline["normalized_specificity"].mean())

    if counts["three-cell-chain"] >= GENERAL_FAMILIES_MIN:
        if joint_counts["three-cell-chain"] >= GENERAL_FAMILIES_MIN:
            status = "MULTICELL_TISSUE_SPECIFICITY_SIGNAL"
        else:
            status = "SPECIFICITY_WITH_INTEGRITY_COST"
    elif counts["one-cell"] >= GENERAL_FAMILIES_MIN and joint_counts["one-cell"] >= GENERAL_FAMILIES_MIN:
        status = "FIXED_ONECELL_TISSUE_SPECIFICITY_SIGNAL"
    elif counts["three-cell-chain"] > counts["one-cell"] and counts["three-cell-chain"] >= 2:
        status = "PARTIAL_MULTICELL_TISSUE_SPECIFICITY"
    else:
        status = "NO_CAPABILITY_TISSUE_IDENTITY"

    decision = {
        "format": "minicells.capability-tissue-specificity.v1",
        "experiment": "MINI Cells Experiment 020 — Capability Tissue Specificity",
        "question": "Do distinct capabilities form functionally distinct cellular tissues, and does a fixed three-cell tissue create more capability identity than a matched one-cell tissue?",
        "status": status,
        "design": {
            "replicates": N_REPLICATES,
            "skill_families": list(SKILL_FAMILIES),
            "arms": {"one-cell": 1, "three-cell-chain": 3},
            "same_phase1_checkpoint_within_replicate": True,
            "same_skill_corpus_and_schedule_across_arms": True,
            "genome_frozen_during_skill_learning": True,
            "old_phenotype_protected": True,
            "autonomous_growth_or_topology_updates": False,
            "recruitment_gate_or_router": False,
            "specificity": "U_aa - mean_{b!=a} U_ab",
            "normalized_specificity_min": SPECIFICITY_NORM_MIN,
            "replicate_mean_top1_min": f">={REPLICATE_TOP1_MIN}/3",
            "example_top1_min": EXAMPLE_TOP1_MIN,
            "general_family_pass_min": GENERAL_FAMILIES_MIN,
            "retention_ratio_max": RETENTION_RATIO_MAX,
            "causal_fraction_min": CAUSAL_FRACTION_MIN,
            "transplant_recovery_min": TRANSPLANT_RECOVERY_MIN,
            "gpu_count": gpu_count,
        },
        "baseline_019b": {
            "zero_training": True,
            "replicate_family_curves": len(baseline),
            "matching_rank1_curves": baseline_top1,
            "mean_normalized_specificity": baseline_norm,
            "source_status": (sources.get("baseline_decision") or {}).get("status") if isinstance(sources.get("baseline_decision"), dict) else None,
        },
        "results": {
            "specificity_pass_families": counts,
            "joint_specificity_integrity_pass_families": joint_counts,
            "three_minus_one_mean_normalized_specificity": float(geometry["delta_normalized_specificity"].mean()),
            "three_minus_one_mean_example_top1": float(geometry["delta_example_top1"].mean()),
            "three_minus_one_skill_improvement": float((geometry["three_cell_skill_improvement"] - geometry["one_cell_skill_improvement"]).mean()),
        },
        "interpretation": {
            "success": "A specificity signal requires diagonal functional preference, not merely stronger generic utility. Matching tissues must be identifiable across both replicate-mean and example-level tests.",
            "multicell": "If the three-cell arm succeeds while the matched one-cell arm fails, tissue capacity/geometry is evidence for capability differentiation under a shared frozen genome.",
            "fail": "If both arms remain nonspecific, recruitment should remain deferred; the next problem is the learning objective or phenotype/genome degrees of freedom, not the gate.",
            "scope": "020 trains capability tissues but no router, recruitment gate, task-ID-conditioned inference, or autonomous topology policy.",
        },
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def main() -> int:
    sources = validate_sources()
    cache = prepare_corpus()
    gpu_count = run_workers(cache)
    donor, observations = collect()
    baseline = baseline_specificity(sources["baseline_path"])
    matrix = utility_matrix(observations)
    replicate = replicate_specificity(observations)
    family = family_specificity(replicate, observations, donor)
    geometry = geometry_comparison(family)
    plot_matrix(matrix, "one-cell", "one-cell-utility-matrix.png")
    plot_matrix(matrix, "three-cell-chain", "three-cell-utility-matrix.png")
    plot_specificity(family)
    plot_identity(family)
    decision = decide(family, baseline, geometry, gpu_count, sources)
    (OUT / "task-spec.json").write_text(json.dumps({
        "format": "minicells.capability-tissue-specificity-task.v1",
        "experiment": "020",
        "families": list(SKILL_FAMILIES),
        "arms": list(TISSUE_ARMS),
        "donor_steps": e019.DONOR_STEPS,
        "examples_per_input_family": e019.UTILITY_EXAMPLES_PER_FAMILY,
        "full_activation_only_for_specificity": True,
        "baseline_experiment": "019b",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], **decision["results"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
