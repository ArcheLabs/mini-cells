from __future__ import annotations

import json
import math
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
from torch import nn

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_data import prepare_tinystories_corpus  # noqa: E402
from minicells.language_proposal_utility import BOUNDARY_FEATURES, LOCAL_FEATURES  # noqa: E402
from minicells.language_utility_skill_data import SKILL_FAMILIES  # noqa: E402


OUT = ROOT / "results" / "proposal-utility-discovery-v1"
WORKER = ROOT / "scripts" / "run_proposal_utility_discovery_worker.py"
N_REPLICATES = 3
RANDOM_CONTROL = "RANDOM"
RIDGE = 1e-3
MLP_STEPS = 400
MLP_BATCH = 2048

ESTIMATORS = {
    "proposal-norm": ("proposal_parent_rms",),
    "probe-kl": ("probe_kl",),
    "local-ridge": LOCAL_FEATURES,
    "boundary-ridge": BOUNDARY_FEATURES,
    "local-mlp": LOCAL_FEATURES,
    "boundary-mlp": BOUNDARY_FEATURES,
}


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
        raise RuntimeError("Experiment 019 requires CUDA")
    gpu_count = min(2, available)
    for start in range(0, N_REPLICATES, gpu_count):
        group = list(range(start, min(start + gpu_count, N_REPLICATES)))
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
                "--cache-dir",
                str(cache),
                "--output-dir",
                str(OUT),
            ]
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
    return pd.read_csv(path)


def collect() -> dict[str, pd.DataFrame | list[dict]]:
    workers = []
    phase1 = []
    phase1_events = []
    donor_summary = []
    donor_events = []
    observations = []
    for replicate in range(N_REPLICATES):
        workers.append(json.loads((OUT / f"r{replicate}-worker.json").read_text(encoding="utf-8")))
        for target, name in (
            (phase1, "phase1-checkpoints"),
            (phase1_events, "phase1-events"),
            (donor_summary, "donor-summary"),
            (donor_events, "donor-events"),
            (observations, "utility-observations"),
        ):
            frame = _read(OUT / f"r{replicate}-{name}.csv")
            if not frame.empty:
                target.append(frame)
    result = {
        "workers": workers,
        "phase1": pd.concat(phase1, ignore_index=True),
        "phase1_events": pd.concat(phase1_events, ignore_index=True) if phase1_events else pd.DataFrame(),
        "donor_summary": pd.concat(donor_summary, ignore_index=True),
        "donor_events": pd.concat(donor_events, ignore_index=True),
        "observations": pd.concat(observations, ignore_index=True),
    }
    result["phase1"].to_csv(OUT / "phase1-checkpoints.csv", index=False)
    result["phase1_events"].to_csv(OUT / "phase1-events.csv", index=False)
    result["donor_summary"].to_csv(OUT / "donor-summary.csv", index=False)
    result["donor_events"].to_csv(OUT / "donor-events.csv", index=False)
    result["observations"].to_csv(OUT / "utility-observations.csv", index=False)
    return result


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return _pearson(left_rank, right_rank)


def _auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target.astype(bool)
    n_pos = int(positive.sum())
    n_neg = len(target) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_z, test_z, _, _ = _standardize(train_x, test_x)
    x = np.concatenate([np.ones((len(train_z), 1)), train_z], axis=1)
    xt = np.concatenate([np.ones((len(test_z), 1)), test_z], axis=1)
    penalty = RIDGE * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ train_y)
    return xt @ beta


def _mlp_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    train_z, test_z, _, _ = _standardize(train_x, test_x)
    y_mean = float(train_y.mean())
    y_scale = float(train_y.std())
    if y_scale < 1e-8:
        y_scale = 1.0
    x = torch.tensor(train_z, dtype=torch.float32)
    y = torch.tensor((train_y - y_mean) / y_scale, dtype=torch.float32)
    xt = torch.tensor(test_z, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.Tanh(), nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(MLP_STEPS):
        indices = torch.randint(0, len(x), (min(MLP_BATCH, len(x)),), generator=generator)
        prediction = model(x[indices]).squeeze(-1)
        loss = torch.nn.functional.smooth_l1_loss(prediction, y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(xt).squeeze(-1).numpy()
    return prediction * y_scale + y_mean


def oracle_consistency(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in ("ALL", *SKILL_FAMILIES):
        selected = observations if family == "ALL" else observations.loc[observations["input_family"] == family]
        left = selected["oracle_gradient"].to_numpy(float)
        right = selected["oracle_fd"].to_numpy(float)
        rows.append({
            "input_family": family,
            "rows": len(selected),
            "pearson": _pearson(left, right),
            "spearman": _spearman(left, right),
            "sign_agreement": float((np.sign(left) == np.sign(right)).mean()),
            "mean_abs_error": float(np.mean(np.abs(left - right))),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "oracle-consistency.csv", index=False)
    return frame


def _selection_metrics(test: pd.DataFrame, predictions: np.ndarray) -> tuple[float, float]:
    scored = test[["replicate", "example", "input_family", "candidate_family", "oracle_gradient"]].copy()
    scored["prediction"] = predictions
    correct = 0
    regrets = []
    groups = 0
    for _, group in scored.groupby(["replicate", "example", "input_family"], sort=False):
        oracle_index = group["oracle_gradient"].idxmax()
        predicted_index = group["prediction"].idxmax()
        correct += int(group.loc[oracle_index, "candidate_family"] == group.loc[predicted_index, "candidate_family"])
        best = float(group.loc[oracle_index, "oracle_gradient"])
        chosen = float(group.loc[predicted_index, "oracle_gradient"])
        regrets.append((best - chosen) / max(abs(best), 1e-6))
        groups += 1
    return correct / max(1, groups), float(np.median(regrets)) if regrets else float("nan")


def evaluate_estimators(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for heldout_index, heldout in enumerate(SKILL_FAMILIES):
        # Strong leave-one-family-out: the held-out examples and the held-out
        # donor tissue are both absent from estimator training.
        train = observations.loc[
            (observations["input_family"] != heldout)
            & (observations["candidate_family"] != heldout)
        ].copy()
        test = observations.loc[observations["input_family"] == heldout].copy()
        target_train = train["oracle_gradient"].to_numpy(float)
        target_test = test["oracle_gradient"].to_numpy(float)
        positive = target_test > 0.0

        for estimator, features in ESTIMATORS.items():
            train_x = train.loc[:, list(features)].to_numpy(float)
            test_x = test.loc[:, list(features)].to_numpy(float)
            if estimator.endswith("mlp"):
                prediction = _mlp_predict(
                    train_x,
                    target_train,
                    test_x,
                    seed=19_019 + 100 * heldout_index + len(features),
                )
            else:
                prediction = _ridge_predict(train_x, target_train, test_x)
            top1, regret = _selection_metrics(test, prediction)
            auc = _auc(positive, prediction)
            rows.append({
                "heldout_family": heldout,
                "estimator": estimator,
                "features": json.dumps(list(features)),
                "train_rows": len(train),
                "test_rows": len(test),
                "spearman": _spearman(target_test, prediction),
                "pearson": _pearson(target_test, prediction),
                "auc_positive_utility": auc,
                "sign_accuracy": float((np.sign(prediction) == np.sign(target_test)).mean()),
                "top1_selection_accuracy": top1,
                "median_normalized_regret": regret,
                "positive_fraction": float(positive.mean()),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "estimator-results.csv", index=False)
    return frame


def utility_matrix(observations: pd.DataFrame) -> pd.DataFrame:
    frame = observations.groupby(["input_family", "candidate_family"], as_index=False)["oracle_gradient"].mean()
    frame.to_csv(OUT / "utility-matrix.csv", index=False)
    return frame


def feature_correlations(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = observations["oracle_gradient"].to_numpy(float)
    for feature in BOUNDARY_FEATURES:
        values = observations[feature].to_numpy(float)
        rows.append({
            "feature": feature,
            "scope": "local" if feature in LOCAL_FEATURES else "boundary",
            "pearson": _pearson(values, target),
            "spearman": _spearman(values, target),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "feature-correlations.csv", index=False)
    return frame


def plot_oracle(observations: pd.DataFrame) -> None:
    sample = observations.sample(n=min(5000, len(observations)), random_state=19019)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sample["oracle_gradient"], sample["oracle_fd"], s=7, alpha=0.25)
    low = min(sample["oracle_gradient"].min(), sample["oracle_fd"].min())
    high = max(sample["oracle_gradient"].max(), sample["oracle_fd"].max())
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
    ax.set_xlabel("gradient marginal utility")
    ax.set_ylabel("finite-difference marginal utility")
    ax.set_title("019 oracle consistency")
    fig.tight_layout()
    fig.savefig(OUT / "oracle-gradient-vs-fd.png", dpi=180)
    plt.close(fig)


def plot_matrix(matrix: pd.DataFrame) -> None:
    candidates = [*SKILL_FAMILIES, RANDOM_CONTROL]
    pivot = matrix.pivot(index="input_family", columns="candidate_family", values="oracle_gradient").reindex(index=SKILL_FAMILIES, columns=candidates)
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(candidates)), candidates, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(SKILL_FAMILIES)), SKILL_FAMILIES)
    ax.set_xlabel("candidate tissue")
    ax.set_ylabel("input skill family")
    ax.set_title("Mean true marginal tissue utility")
    fig.colorbar(image, ax=ax, label="mean -dL/de at e=0")
    fig.tight_layout()
    fig.savefig(OUT / "candidate-utility-matrix.png", dpi=180)
    plt.close(fig)


def plot_estimator_metric(results: pd.DataFrame, metric: str, filename: str, ylabel: str, threshold: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SKILL_FAMILIES))
    selected_estimators = ("proposal-norm", "local-ridge", "local-mlp", "boundary-ridge", "boundary-mlp")
    width = 0.14
    for index, estimator in enumerate(selected_estimators):
        values = results.loc[results["estimator"] == estimator].set_index("heldout_family").reindex(SKILL_FAMILIES)[metric].to_numpy(float)
        ax.bar(x + (index - 2) * width, values, width=width, label=estimator)
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.set_xticks(x, SKILL_FAMILIES, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("Leave-one-skill-family-out proposal utility prediction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_feature_correlations(frame: pd.DataFrame) -> None:
    ordered = frame.reindex(frame["spearman"].abs().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(ordered)), ordered["spearman"].to_numpy(float))
    ax.set_xticks(np.arange(len(ordered)), ordered["feature"], rotation=55, ha="right")
    ax.set_ylabel("Spearman with true marginal utility")
    ax.set_title("Label-free observable / oracle association")
    fig.tight_layout()
    fig.savefig(OUT / "feature-oracle-correlations.png", dpi=180)
    plt.close(fig)


def _estimator_passes(results: pd.DataFrame, estimator: str) -> tuple[bool, int]:
    selected = results.loc[results["estimator"] == estimator]
    passes = (
        (selected["spearman"] >= 0.50)
        & (selected["auc_positive_utility"] >= 0.75)
        & (selected["top1_selection_accuracy"] >= 0.60)
        & (selected["median_normalized_regret"] <= 0.35)
    )
    count = int(passes.sum())
    return count >= 4, count


def make_decision(
    observations: pd.DataFrame,
    donors: pd.DataFrame,
    oracle: pd.DataFrame,
    estimators: pd.DataFrame,
    gpu_count: int,
) -> dict[str, object]:
    all_oracle = oracle.loc[oracle["input_family"] == "ALL"].iloc[0]
    oracle_pass = float(all_oracle["pearson"]) >= 0.95 and float(all_oracle["sign_agreement"]) >= 0.90
    per_family_oracle = int(((oracle.loc[oracle["input_family"] != "ALL", "pearson"] >= 0.90)).sum())

    trained = donors.loc[donors["candidate_kind"] == "trained"]
    donor_family = trained.groupby("family")["skill_improvement"].median()
    usable_families = int((donor_family > 0.25).sum())
    donor_bank_pass = usable_families >= 5

    local_ridge_pass, local_ridge_count = _estimator_passes(estimators, "local-ridge")
    local_mlp_pass, local_mlp_count = _estimator_passes(estimators, "local-mlp")
    boundary_ridge_pass, boundary_ridge_count = _estimator_passes(estimators, "boundary-ridge")
    boundary_mlp_pass, boundary_mlp_count = _estimator_passes(estimators, "boundary-mlp")
    local_pass = local_ridge_pass or local_mlp_pass
    boundary_pass = boundary_ridge_pass or boundary_mlp_pass

    baseline = estimators.loc[estimators["estimator"] == "proposal-norm"].set_index("heldout_family")
    best_local_name = "local-mlp" if local_mlp_count >= local_ridge_count else "local-ridge"
    best_local = estimators.loc[estimators["estimator"] == best_local_name].set_index("heldout_family")
    beats_baseline = int(
        (
            (best_local["top1_selection_accuracy"] > baseline["top1_selection_accuracy"])
            | (best_local["median_normalized_regret"] < baseline["median_normalized_regret"])
        ).sum()
    )
    if local_pass and beats_baseline < 4:
        local_pass = False

    if not oracle_pass:
        status = "UTILITY_ORACLE_INCONSISTENT"
    elif local_pass and donor_bank_pass:
        status = "LOCAL_PROPOSAL_UTILITY_SIGNAL"
    elif boundary_pass and donor_bank_pass:
        status = "BOUNDARY_ONLY_PROPOSAL_UTILITY_SIGNAL"
    elif max(local_ridge_count, local_mlp_count, boundary_ridge_count, boundary_mlp_count) >= 2:
        status = "PARTIAL_PROPOSAL_UTILITY_SIGNAL"
    else:
        status = "NO_GENERAL_PROPOSAL_UTILITY_SIGNAL"

    positive_fraction = float((observations["oracle_gradient"] > 0).mean())
    decision = {
        "format": "minicells.proposal-utility-discovery.v1",
        "experiment": "MINI Cells Experiment 019 — Proposal Utility Discovery",
        "question": "Can label-free local or boundary observables predict the true marginal value of a sleeping capability-tissue proposal across held-out skill families?",
        "status": status,
        "design": {
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "skill_families": list(SKILL_FAMILIES),
            "candidate_tissues": [*SKILL_FAMILIES, RANDOM_CONTROL],
            "same_phase1_checkpoint_within_replicate": True,
            "one_newborn_per_candidate": True,
            "autonomous_growth_during_utility_measurement": False,
            "recruitment_training_during_utility_measurement": False,
            "utility_oracle": "-d masked validation NLL / de at e=0",
            "finite_difference_oracle": "[L(0)-L(epsilon)]/epsilon",
            "epsilon": 0.02,
            "leave_one_family_out": "held-out input family and its donor tissue are both excluded from estimator training",
            "task_or_tissue_ids_in_estimator": False,
            "local_features": list(LOCAL_FEATURES),
            "boundary_features": list(BOUNDARY_FEATURES),
        },
        "pre_registered_signal": {
            "oracle_overall_pearson_min": 0.95,
            "oracle_overall_sign_agreement_min": 0.90,
            "usable_donor_family_median_skill_improvement_min": 0.25,
            "usable_donor_families_min": 5,
            "heldout_family_pass": {
                "spearman_min": 0.50,
                "auc_positive_utility_min": 0.75,
                "top1_selection_accuracy_min": 0.60,
                "median_normalized_regret_max": 0.35,
            },
            "heldout_family_pass_count_min": 4,
            "local_estimator_must_beat_proposal_norm_on_families_min": 4,
        },
        "results": {
            "oracle_overall_pearson": float(all_oracle["pearson"]),
            "oracle_overall_spearman": float(all_oracle["spearman"]),
            "oracle_overall_sign_agreement": float(all_oracle["sign_agreement"]),
            "oracle_family_pearson_ge_0_90": per_family_oracle,
            "usable_donor_families": usable_families,
            "positive_utility_fraction": positive_fraction,
            "local_ridge_pass_families": local_ridge_count,
            "local_mlp_pass_families": local_mlp_count,
            "boundary_ridge_pass_families": boundary_ridge_count,
            "boundary_mlp_pass_families": boundary_mlp_count,
            "best_local_estimator": best_local_name,
            "best_local_beats_proposal_norm_families": beats_baseline,
        },
        "interpretation": {
            "local": "A local signal means parent/proposal dynamics alone generalize to unseen input and tissue families; this is the evidence needed before implementing a shared local recruitment gate.",
            "boundary_only": "Boundary success without local success means proposal utility is observable at the language interface but a genuinely local endogenous error field is still missing.",
            "partial": "Some held-out families are predictable, but the signal is not yet architecture-general.",
            "fail": "Do not tune another recruitment threshold. Either the oracle is inconsistent or the recorded endogenous observables do not contain enough proposal-specific utility information.",
            "cost": "019 measures signal existence, not cheap-probe efficiency. The epsilon probe currently executes the same cellular rule and must be compressed in a later experiment before claiming compute savings.",
        },
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def write_task_spec(gpu_count: int) -> None:
    task = {
        "format": "minicells.proposal-utility-task.v1",
        "experiment": "019",
        "replicates": N_REPLICATES,
        "gpu_count": gpu_count,
        "families": list(SKILL_FAMILIES),
        "candidate_control": RANDOM_CONTROL,
        "oracle_epsilon": 0.02,
        "estimators": {name: list(features) for name, features in ESTIMATORS.items()},
    }
    (OUT / "task-spec.json").write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    cache, _ = prepare_corpus()
    gpu_count = run_workers(cache)
    collected = collect()
    observations = collected["observations"]
    donors = collected["donor_summary"]
    oracle = oracle_consistency(observations)
    estimators = evaluate_estimators(observations)
    matrix = utility_matrix(observations)
    correlations = feature_correlations(observations)

    plot_oracle(observations)
    plot_matrix(matrix)
    plot_estimator_metric(estimators, "spearman", "heldout-spearman.png", "Spearman", threshold=0.50)
    plot_estimator_metric(estimators, "auc_positive_utility", "heldout-auc.png", "AUC(U*>0)", threshold=0.75)
    plot_estimator_metric(estimators, "top1_selection_accuracy", "heldout-top1.png", "top-1 tissue selection accuracy", threshold=0.60)
    plot_estimator_metric(estimators, "median_normalized_regret", "heldout-regret.png", "median normalized regret", threshold=0.35)
    plot_feature_correlations(correlations)
    write_task_spec(gpu_count)
    decision = make_decision(observations, donors, oracle, estimators, gpu_count)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
