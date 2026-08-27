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
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_independent_third_trait_worker as worker  # noqa: E402
from minicells.language_conflict_differentiation import prepare_arithmetic_cache  # noqa: E402
from minicells.language_data import load_tokenizer, prepare_tinystories_corpus  # noqa: E402
from minicells.language_independent_third_trait import (  # noqa: E402
    CANDIDATES,
    SCREEN_ABSORPTION_RATIO_MAX,
    SCREEN_INDEPENDENCE_MIN,
    SCREEN_LEARNABILITY_MIN,
    SCREEN_QUALIFY_REPLICATES_MIN,
    STRUCTURAL_COST_FRACTION,
    aggregate_status,
    classify_replicate,
    expected_trajectory,
    prepare_candidate_caches,
    select_candidate,
)
from minicells.language_probationary_trait_genesis import (  # noqa: E402
    GEOMETRY_ADVANTAGE_MIN,
    ROUTING_PURITY_MIN,
)


OUT = ROOT / "results" / "independent-third-trait-v1"
WORKER = ROOT / "scripts" / "run_independent_third_trait_worker.py"
N_REPLICATES = worker.N_REPLICATES


def prepare_corpora() -> Path:
    corpus = prepare_tinystories_corpus(ROOT)
    cache = corpus.tokenizer_path.parent
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    corpus_manifest = cache / "corpus-manifest.json"
    if corpus_manifest.is_file():
        shutil.copy2(corpus_manifest, OUT / "corpus-manifest.json")
    tokenizer = load_tokenizer(corpus.tokenizer_path)
    arithmetic = prepare_arithmetic_cache(cache, tokenizer)
    shutil.copy2(arithmetic["path"], OUT / "arithmetic-manifest.json")
    candidates = prepare_candidate_caches(cache, tokenizer)
    manifests = {
        name: payload["manifest"]
        for name, payload in candidates.items()
    }
    (OUT / "candidate-manifests.json").write_text(
        json.dumps(manifests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    provenance = (
        ("022-emergent-trait-bifurcation", "source-022-decision.json", "EMERGENT_TRAIT_BIFURCATION_SIGNAL"),
        ("023-online-nonparametric-trait-genesis", "source-023-decision.json", "NO_ONLINE_TRAIT_GENESIS"),
        ("023b-probationary-trait-genesis", "source-023b-decision.json", "PROBATIONARY_TRAIT_GENESIS_SIGNAL"),
        ("024-sequential-probationary-genesis", "source-024-decision.json", "FIRST_BIRTH_WITHOUT_SECOND_TRAIT_GENESIS"),
    )
    for experiment, destination, expected in provenance:
        source = ROOT / "artifacts" / "experiments" / experiment / "decision.json"
        if not source.is_file():
            raise FileNotFoundError(f"required provenance missing: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("status") != expected:
            raise RuntimeError(
                f"unexpected provenance status for {experiment}: {payload.get('status')!r}; expected {expected!r}"
            )
        shutil.copy2(source, OUT / destination)
    diagnosis = ROOT / "artifacts" / "experiments" / "024-sequential-probationary-genesis" / "DIAGNOSIS.md"
    if not diagnosis.is_file():
        raise FileNotFoundError("Experiment 024 diagnosis is required for 024b provenance")
    shutil.copy2(diagnosis, OUT / "source-024-diagnosis.md")
    return cache


def worker_complete(replicate: int, phase: str) -> bool:
    path = OUT / f"r{replicate}-{phase}-worker.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        "minicells.independent-third-trait-screen-worker.v1"
        if phase == "screen"
        else "minicells.independent-third-trait-challenge-worker.v1"
    )
    return payload.get("format") == expected and int(payload.get("replicate", -1)) == replicate


def run_phase(cache: Path, phase: str, *, candidate: str | None = None) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 024b requires a Kaggle GPU accelerator")
    gpu_count = min(2, available)
    missing = [r for r in range(N_REPLICATES) if not worker_complete(r, phase)]
    if not missing:
        print(f"reusing complete Experiment 024b {phase} workers")
        return gpu_count
    for start in range(0, len(missing), gpu_count):
        group = missing[start : start + gpu_count]
        active = []
        for local_gpu, replicate in enumerate(group):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
            log = OUT / f"r{replicate}-{phase}.log"
            handle = log.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(WORKER),
                "--replicate", str(replicate),
                "--cache-dir", str(cache),
                "--output-dir", str(OUT),
                "--phase", phase,
            ]
            if phase == "challenge":
                if candidate is None:
                    raise ValueError("challenge requires selected candidate")
                command.extend(["--candidate", candidate])
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((replicate, local_gpu, process, log, handle))
            print(f"started Experiment 024b {phase} r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- Experiment 024b {phase} r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} {phase} exited {code}; see {log}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def collect_screening() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames = []
    workers = []
    for replicate in range(N_REPLICATES):
        worker_path = OUT / f"r{replicate}-screen-worker.json"
        if not worker_path.is_file():
            raise FileNotFoundError(worker_path)
        workers.append(json.loads(worker_path.read_text(encoding="utf-8")))
        path = OUT / f"r{replicate}-screening.csv"
        if path.is_file() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(OUT / "screening.csv", index=False)
    return combined, workers


def choose_candidate(screening: pd.DataFrame) -> dict[str, object]:
    if screening.empty:
        selection = {
            "format": "minicells.independent-third-trait-selection.v1",
            "selected": CANDIDATES[0],
            "qualified": False,
            "qualifying_candidates": [],
            "median_independence": {name: None for name in CANDIDATES},
            "qualifying_replicates": {name: 0 for name in CANDIDATES},
            "reason": "no replicate reached a screenable two-trait foundation",
        }
    else:
        selected = select_candidate(screening.to_dict(orient="records"))
        selection = {
            "format": "minicells.independent-third-trait-selection.v1",
            "selected": selected.selected,
            "qualified": selected.qualified,
            "qualifying_candidates": list(selected.qualifying_candidates),
            "median_independence": selected.median_independence,
            "qualifying_replicates": selected.qualifying_replicates,
            "rule": "among candidates qualifying in >=2/3 replicates, choose highest median independence advantage; ties lexicographic; if none qualify choose highest median only as exploratory challenge",
            "selection_frozen_before_challenge": True,
        }
    (OUT / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection


def collect_challenge() -> dict[str, object]:
    workers = []
    keys = {
        "stage_summary": "challenge-stage-summary.csv",
        "proposal": "challenge-proposal.csv",
        "windows": "challenge-windows.csv",
        "learning": "challenge-learning.csv",
        "routing": "challenge-routing.csv",
        "evaluation": "challenge-evaluation.csv",
    }
    tables: dict[str, list[pd.DataFrame]] = {key: [] for key in keys}
    for replicate in range(N_REPLICATES):
        worker_path = OUT / f"r{replicate}-challenge-worker.json"
        if not worker_path.is_file():
            raise FileNotFoundError(worker_path)
        workers.append(json.loads(worker_path.read_text(encoding="utf-8")))
        for key, suffix in keys.items():
            path = OUT / f"r{replicate}-{suffix}"
            if path.is_file() and path.stat().st_size > 0:
                frame = pd.read_csv(path)
                if not frame.empty:
                    tables[key].append(frame)
    result: dict[str, object] = {"workers": workers}
    for key, frames in tables.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        result[key] = frame
        frame.to_csv(OUT / keys[key].replace("challenge-", ""), index=False)
    return result


def validate_invariants(
    screening: pd.DataFrame,
    screen_workers: list[dict[str, object]],
    selection: dict[str, object],
    challenge: dict[str, object],
) -> dict[str, object]:
    if not screening.empty:
        if set(screening["candidate"].unique()) - set(CANDIDATES):
            raise RuntimeError("unknown candidate in screening output")
        if not np.isfinite(
            screening[
                [
                    "baseline_candidate_nll", "existing_candidate_nll", "newborn_candidate_nll",
                    "baseline_arithmetic_nll", "existing_arithmetic_nll", "existing_candidate_gain",
                    "arithmetic_damage", "existing_value", "newborn_candidate_gain", "newborn_value",
                    "independence_advantage",
                ]
            ].to_numpy(float)
        ).all():
            raise RuntimeError("non-finite screening metrics")
    selected = str(selection["selected"])
    for payload in challenge["workers"]:
        if payload.get("candidate") != selected:
            raise RuntimeError("challenge workers did not use one frozen selected candidate")
    payload = {
        "format": "minicells.independent-third-trait-invariants.v1",
        "screening_candidate_set_fixed": list(CANDIDATES),
        "screening_uses_posthoc_task_identity": True,
        "selection_frozen_before_challenge": True,
        "one_selected_candidate_across_replicates": True,
        "challenge_proposal_uses_task_label": False,
        "challenge_geometry_routing_uses_task_label": False,
        "challenge_commit_uses_task_label": False,
        "structural_cost_fraction_unchanged_from_023b_024": STRUCTURAL_COST_FRACTION,
        "geometry_advantage_min_unchanged_from_023b_024": GEOMETRY_ADVANTAGE_MIN,
        "screen_foundation_replicates": sum(int(w.get("foundation_active_k", 0) == 2) for w in screen_workers),
    }
    (OUT / "invariants.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def checkpoint_manifest(challenge_workers: list[dict[str, object]]) -> dict[str, object]:
    checkpoint_dir = OUT / "checkpoints"
    files = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else []
    names = {path.name for path in files}
    required = []
    for replicate in range(N_REPLICATES):
        required.extend([
            f"r{replicate}-parent.pt",
            f"r{replicate}-arithmetic-birth.pt",
            f"r{replicate}-screen.pt",
        ])
        worker_payload = challenge_workers[replicate]
        if not bool(worker_payload.get("skipped", False)):
            required.append(f"r{replicate}-c_weak_selected.pt")
            if "D_STRONG_SELECTED" in worker_payload.get("stages_completed", []):
                required.append(f"r{replicate}-d_strong_selected.pt")
    missing = sorted(set(required) - names)
    payload = {
        "format": "minicells.independent-third-trait-checkpoint-manifest.v1",
        "experiment": "024b",
        "file_count": len(files),
        "maximum_file_count": N_REPLICATES * 5,
        "observed_path_required_count": len(set(required)),
        "observed_path_complete": not missing,
        "missing_required": missing,
        "files": [{"name": path.name, "bytes": path.stat().st_size} for path in files],
        "published_model_checkpoints": False,
    }
    (OUT / "checkpoint-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if missing:
        raise RuntimeError(f"Experiment 024b observed-path checkpoints incomplete: {missing}")
    return payload


def decide(
    screening: pd.DataFrame,
    screen_workers: list[dict[str, object]],
    selection: dict[str, object],
    challenge: dict[str, object],
    gpu_count: int,
    manifest: dict[str, object],
) -> dict[str, object]:
    summary: pd.DataFrame = challenge["stage_summary"]
    per_replicate = []
    for replicate in range(N_REPLICATES):
        foundation = screen_workers[replicate].get("foundation_summary", {})
        arithmetic_birth = bool(
            int(foundation.get("accepted", 0)) == 1
            and int(foundation.get("identity_pass", 0)) == 1
            and int(foundation.get("routing_purity_pass", 0)) == 1
        )
        rows = summary.loc[summary["replicate"] == replicate] if not summary.empty else pd.DataFrame()
        weak_row = rows.loc[rows["stage"] == "C_WEAK_SELECTED"] if not rows.empty else pd.DataFrame()
        strong_row = rows.loc[rows["stage"] == "D_STRONG_SELECTED"] if not rows.empty else pd.DataFrame()
        weak_reject = bool(
            len(weak_row) == 1
            and int(weak_row.iloc[0]["accepted"]) == 0
            and int(weak_row.iloc[0]["retention_identity_pass"]) == 1
            and int(weak_row.iloc[0]["end_k"]) == 2
        )
        strong_birth = bool(
            len(strong_row) == 1
            and int(strong_row.iloc[0]["accepted"]) == 1
            and int(strong_row.iloc[0]["identity_pass"]) == 1
            and int(strong_row.iloc[0]["routing_purity_pass"]) == 1
            and int(strong_row.iloc[0]["end_k"]) == 3
        )
        challenge_worker = challenge["workers"][replicate]
        final_k = int(challenge_worker.get("final_active_k", screen_workers[replicate].get("foundation_active_k", 1)))
        result = classify_replicate(
            arithmetic_birth=arithmetic_birth,
            weak_reject=weak_reject,
            strong_birth=strong_birth,
            final_k=final_k,
        )
        result["replicate"] = replicate
        per_replicate.append(result)
    status = aggregate_status(per_replicate, screening_qualified=bool(selection["qualified"]))
    pd.DataFrame(per_replicate).to_csv(OUT / "replicate-decision.csv", index=False)
    selected_rows = (
        screening.loc[screening["candidate"] == selection["selected"]].to_dict(orient="records")
        if not screening.empty else []
    )
    decision = {
        "format": "minicells.independent-third-trait.v1",
        "experiment": "MINI Cells Experiment 024b — Independent Third-Trait Challenge",
        "question": "Can a two-trait Story/Arithmetic organism first identify a functionally non-substitutable third capability and then grow a third persistent trait under the unchanged probationary birth rule?",
        "design": {
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "candidates": list(CANDIDATES),
            "screen_steps": 128,
            "screen_learnability_min": SCREEN_LEARNABILITY_MIN,
            "screen_independence_min": SCREEN_INDEPENDENCE_MIN,
            "screen_absorption_ratio_max": SCREEN_ABSORPTION_RATIO_MAX,
            "screen_qualify_replicates_min": SCREEN_QUALIFY_REPLICATES_MIN,
            "selected_candidate": selection["selected"],
            "screening_qualified": bool(selection["qualified"]),
            "structural_cost_fraction": STRUCTURAL_COST_FRACTION,
            "geometry_advantage_min": GEOMETRY_ADVANTAGE_MIN,
            "routing_purity_min": ROUTING_PURITY_MIN,
            "expected_trajectory": list(expected_trajectory()),
            "selection_frozen_before_challenge": True,
            "challenge_proposal_uses_task_label": False,
            "challenge_geometry_routing_uses_task_label": False,
            "challenge_commit_uses_task_label": False,
        },
        "selection": selection,
        "selected_candidate_screening_rows": selected_rows,
        "results": {
            "arithmetic_birth_replicates": sum(int(row["arithmetic_birth"]) for row in per_replicate),
            "weak_reject_replicates": sum(int(row["weak_reject"]) for row in per_replicate),
            "strong_third_birth_replicates": sum(int(row["strong_birth"]) for row in per_replicate),
            "final_k3_replicates": sum(int(row["final_k"] == 3) for row in per_replicate),
            "per_replicate": per_replicate,
        },
        "checkpoint_manifest": {
            "file_count": manifest["file_count"],
            "maximum_file_count": manifest["maximum_file_count"],
            "observed_path_complete": manifest["observed_path_complete"],
            "published_model_checkpoints": False,
        },
        "interpretation": {
            "success": "Strong positive requires a preregistered screening-qualified candidate, stable Story/Arithmetic first birth in >=2/3, weak selected-capability rejection with two-trait retention in 3/3, and strong selected-capability 2->3 birth with three-domain identity and routing purity >=0.75 in >=2/3.",
            "scope": "024b tests functional non-substitutability before third-trait birth. It still uses benchmark-controlled screening, a fixed pretrained gradient sensor, frozen shared genome, and synthetic capabilities; it does not establish local sensing, rewiring, pruning, merging, or inference-time recruitment.",
        },
        "status": status,
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return decision


def write_task_spec() -> None:
    payload = {
        "format": "minicells.independent-third-trait-task.v1",
        "candidate_pool": list(CANDIDATES),
        "screening_definition": "existing computational-trait candidate gain minus Arithmetic retention damage versus a matched newborn candidate gain minus the same 0.005 structural cost used by probation",
        "selection_rule": "candidate must qualify in >=2/3 replicates; select highest median independence advantage; deterministic lexical tie-break",
        "weak_selected_distribution": "45% Story / 45% Arithmetic / 10% selected candidate",
        "strong_selected_distribution": "approximately one-third Story / Arithmetic / selected candidate",
        "screening_uses_benchmark_identity": True,
        "challenge_proposal_uses_task_label": False,
        "challenge_geometry_routing_uses_task_label": False,
        "challenge_commit_uses_task_label": False,
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def plot_screening(screening: pd.DataFrame) -> None:
    if screening.empty:
        return
    medians = screening.groupby("candidate", as_index=False)["independence_advantage"].median()
    medians = medians.set_index("candidate").reindex(CANDIDATES).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(len(medians)), medians["independence_advantage"].to_numpy(float))
    ax.axhline(SCREEN_INDEPENDENCE_MIN, linewidth=1)
    ax.set_xticks(np.arange(len(medians)), medians["candidate"], rotation=35, ha="right")
    ax.set_ylabel("Median newborn independence advantage")
    ax.set_title("Which candidate cannot be economically absorbed by the existing computational trait?")
    fig.tight_layout()
    fig.savefig(OUT / "candidate-independence-screen.png", dpi=180)
    plt.close(fig)


def plot_challenge(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rows = summary.sort_values(["stage", "replicate"])
    x = np.arange(len(rows))
    ax.bar(x, rows["geometry_mean_net_utility_last3"].to_numpy(float))
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, [f"r{int(row.replicate)}\n{str(row.stage).split('_')[1]}" for row in rows.itertuples()])
    ax.set_ylabel("Geometry mean net utility, final 3 windows")
    ax.set_title("Weak versus strong selected-capability probation")
    fig.tight_layout()
    fig.savefig(OUT / "selected-capability-probation.png", dpi=180)
    plt.close(fig)


def main() -> int:
    cache = prepare_corpora()
    gpu_count = run_phase(cache, "screen")
    screening, screen_workers = collect_screening()
    selection = choose_candidate(screening)
    selected = str(selection["selected"])
    print(json.dumps(selection, indent=2, sort_keys=True))
    run_phase(cache, "challenge", candidate=selected)
    challenge = collect_challenge()
    validate_invariants(screening, screen_workers, selection, challenge)
    manifest = checkpoint_manifest(challenge["workers"])
    write_task_spec()
    plot_screening(screening)
    plot_challenge(challenge["stage_summary"])
    decision = decide(screening, screen_workers, selection, challenge, gpu_count, manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
