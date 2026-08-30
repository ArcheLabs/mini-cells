#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from minicells.optimizer_search import (
    baseline_config,
    expand_stage2,
    production_candidate_gates,
    rank_summaries,
    run_config,
    run_solved_regression,
    stage1_configs,
)

DEFAULT_OUT = Path("results/production-optimizer-search-v1")
GENESIS = Path("service/generated/genesis_model.bin")
SOLVED = Path("artifacts/experiments/003b-quantization-localization/solved-q88-model.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 008: deterministic Q8.8 production optimizer search"
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage1-generations", type=int)
    parser.add_argument("--stage2-generations", type=int)
    parser.add_argument("--final-generations", type=int)
    parser.add_argument("--probe-every", type=int, default=64)
    return parser.parse_args()


def git_provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True).strip()

    try:
        return {
            "source_ref": git("rev-parse", "HEAD"),
            "source_tree": git("rev-parse", "HEAD^{tree}"),
            "source_dirty": bool(git("status", "--porcelain")),
        }
    except Exception:
        return {"source_ref": "unknown", "source_tree": "unknown", "source_dirty": None}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_progress(out: Path, stage: str, **extra: object) -> None:
    write_json(
        out / "progress.json",
        {
            "schema": "minicells.optimizer-search.progress.v1",
            "stage": stage,
            **extra,
        },
    )


def baseline_reproduction(out: Path, profile: str, probe_every: int) -> tuple[dict, bool, dict]:
    target = 512 if profile == "full" else 64
    config = baseline_config()
    summary = run_config(GENESIS, out / "runs" / config.id, config, target, probe_every)
    probes = [
        json.loads(line)
        for line in (out / "runs" / config.id / "probes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    initial = probes[0]
    checks = {
        "initial_loss_exact": int(initial["total_loss"]) == 607901,
        "initial_correct_exact": int(initial["correct_tokens"]) == 28,
        "initial_tokens_exact": int(initial["total_tokens"]) == 2168,
    }
    if profile == "full":
        checks.update(
            {
                "final_loss_exact": int(summary["final_probe_loss"]) == 573303,
                "accepted_updates_exact": int(summary["accepted_updates"]) == 20,
            }
        )
    passed = all(checks.values())
    payload = {
        "schema": "minicells.optimizer-search.baseline.v1",
        "profile": profile,
        "config": asdict(config),
        "summary": summary,
        "checks": checks,
        "pass": passed,
        "reference_commit": "93a2e42fc842a2ffa3123ca6faf60fd84ea08f66",
        "reference_native_gate": {
            "initial_probe_loss": 607901,
            "final_probe_loss": 573303,
            "successful_updates": 20,
        },
    }
    write_json(out / "baseline.json", payload)
    return summary, passed, payload


def select_smoke_configs(configs):
    wanted = [0, 1, 3, 11, 15, 18, 21, 27]
    return [configs[index] for index in wanted if index < len(configs)]


def run_many(out: Path, configs, target: int, probe_every: int) -> list[dict]:
    rows = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.id} -> generation {target}", flush=True)
        rows.append(
            run_config(
                GENESIS,
                out / "runs" / config.id,
                config,
                target,
                probe_every,
            )
        )
    return rows


def save_table(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def collect_probe_curves(out: Path, config_ids: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for config_id in config_ids:
        path = out / "runs" / config_id / "probes.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(out / "probe-curves.csv", index=False)
    return frame


def make_plots(out: Path, finalist_ids: list[str], stage2_rows: list[dict]) -> None:
    probes = collect_probe_curves(out, finalist_ids)
    if not probes.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for config_id, part in probes.groupby("config_id"):
            ax.plot(part["generation"], part["total_loss"], label=config_id)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fixed-probe total loss")
        ax.set_title("Experiment 008 finalist fixed-probe loss")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "finalist-probe-loss.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for config_id, part in probes.groupby("config_id"):
            ax.plot(part["generation"], part["token_accuracy"], label=config_id)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fixed-probe token accuracy")
        ax.set_title("Experiment 008 finalist fixed-probe accuracy")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "finalist-probe-accuracy.png", dpi=150)
        plt.close(fig)

    if stage2_rows:
        frame = pd.DataFrame(stage2_rows)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(frame["loss_improvement"], frame["accuracy_delta"])
        for _, row in frame.iterrows():
            ax.annotate(str(row["config_id"]), (row["loss_improvement"], row["accuracy_delta"]), fontsize=6)
        ax.axhline(0.02, linewidth=1)
        ax.axvline(0.10, linewidth=1)
        ax.set_xlabel("Final fixed-probe loss improvement")
        ax.set_ylabel("Final fixed-probe accuracy delta")
        ax.set_title("Experiment 008 loss / accuracy frontier")
        fig.tight_layout()
        fig.savefig(out / "loss-accuracy-frontier.png", dpi=150)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if not GENESIS.exists():
        raise FileNotFoundError(GENESIS)
    if not SOLVED.exists():
        raise FileNotFoundError(SOLVED)

    defaults = {
        "smoke": {"stage1": 32, "stage2": 64, "final": 128, "top1": 2, "top_final": 1},
        "full": {"stage1": 128, "stage2": 512, "final": 2048, "top1": 3, "top_final": 3},
    }[args.profile]
    stage1_generations = args.stage1_generations or defaults["stage1"]
    stage2_generations = args.stage2_generations or defaults["stage2"]
    final_generations = args.final_generations or defaults["final"]

    provenance = git_provenance()
    write_progress(out, "baseline", profile=args.profile, provenance=provenance)
    baseline, baseline_pass, baseline_payload = baseline_reproduction(
        out, args.profile, args.probe_every
    )
    print("baseline:", json.dumps(baseline_payload["checks"], sort_keys=True), flush=True)
    if not baseline_pass:
        decision = {
            "schema": "minicells.optimizer-search.decision.v1",
            "status": "FAIL",
            "diagnosis": "BASELINE_REPRODUCTION_FAIL",
            "next_action": "STOP_AND_REPAIR_EXPERIMENT_MIRROR",
            "provenance": provenance,
            "baseline": baseline_payload,
        }
        write_json(out / "decision.json", decision)
        write_progress(out, "stopped", reason="baseline reproduction failed")
        return 2

    stage1 = stage1_configs()
    if args.profile == "smoke":
        stage1 = select_smoke_configs(stage1)
    write_progress(
        out,
        "stage1",
        profile=args.profile,
        configs=len(stage1),
        generations=stage1_generations,
    )
    stage1_rows = run_many(out, stage1, stage1_generations, args.probe_every)
    stage1_ranked = rank_summaries(stage1_rows)
    save_table(out / "stage1.csv", stage1_ranked)

    by_stage1_id = {config.id: config for config in stage1}
    top_stage1 = [
        by_stage1_id[row["config_id"]]
        for row in stage1_ranked[: int(defaults["top1"])]
    ]
    stage2 = expand_stage2(top_stage1)
    write_progress(
        out,
        "stage2",
        configs=len(stage2),
        generations=stage2_generations,
        promoted=[config.id for config in top_stage1],
    )
    stage2_rows = run_many(out, stage2, stage2_generations, args.probe_every)
    stage2_ranked = rank_summaries(stage2_rows)
    save_table(out / "stage2.csv", stage2_ranked)

    finalist_ids = [
        row["config_id"] for row in stage2_ranked[: int(defaults["top_final"])]
    ]
    by_id = {config.id: config for config in stage2}
    finalists = [by_id[config_id] for config_id in finalist_ids]
    write_progress(
        out,
        "finalists",
        configs=len(finalists),
        generations=final_generations,
        finalist_ids=finalist_ids,
    )
    finalist_rows = run_many(out, finalists, final_generations, args.probe_every)
    finalist_ranked = rank_summaries(finalist_rows)

    solved_rows: list[dict] = []
    solved_by_id: dict[str, dict] = {}
    for config in finalists:
        print(f"solved regression: {config.id}", flush=True)
        solved = run_solved_regression(
            SOLVED,
            out / "solved-regression",
            config,
            64 if args.profile == "smoke" else 128,
        )
        solved_rows.append(solved)
        solved_by_id[config.id] = solved
    save_table(out / "solved-regression.csv", solved_rows)

    annotated = []
    recommended = None
    for row in finalist_ranked:
        solved_pass = bool(solved_by_id[row["config_id"]]["pass"])
        gates = production_candidate_gates(row, solved_pass)
        enriched = {**row, **{f"gate_{key}": value for key, value in gates.items()}}
        enriched["all_gates_pass"] = all(gates.values())
        annotated.append(enriched)
        if recommended is None and enriched["all_gates_pass"]:
            recommended = enriched
    save_table(out / "finalists.csv", annotated)
    make_plots(out, [row["config_id"] for row in annotated], stage2_ranked)

    decision = {
        "schema": "minicells.optimizer-search.decision.v1",
        "status": "PASS" if recommended else "NO_PRODUCTION_CANDIDATE",
        "diagnosis": (
            "PRODUCTION_OPTIMIZER_CANDIDATE_FOUND"
            if recommended
            else "SEARCH_COMPLETED_BUT_TRAINABILITY_GATES_NOT_MET"
        ),
        "profile": args.profile,
        "provenance": provenance,
        "baseline": baseline,
        "stage_generations": {
            "stage1": stage1_generations,
            "stage2": stage2_generations,
            "final": final_generations,
        },
        "ranking_policy": [
            "maximize final fixed-probe token accuracy",
            "maximize final fixed-probe loss improvement",
            "minimize final/best loss regression",
            "prefer lower production integration cost",
            "maximize acceptance rate",
        ],
        "production_gates": {
            "final_loss_improvement_ge_10_percent": True,
            "accuracy_delta_ge_2pp": True,
            "final_within_15_percent_of_best": True,
            "accepted_updates_ge_4": True,
            "solved_regression_pass": True,
            "not_legacy_step": True,
        },
        "recommended_config": recommended,
        "top_finalists": annotated[:3],
        "next_action": (
            "PORT_RECOMMENDED_CONFIG_TO_RUST_THEN_RERUN_LOCAL_NATIVE_GATE"
            if recommended
            else "DO_NOT_START_PVM_OR_E2E;_REVIEW_OBJECTIVE_AND_OPTIMIZER"
        ),
    }
    write_json(out / "decision.json", decision)

    if recommended:
        write_json(
            out / "recommended-runtime.json",
            {
                "schema": "minicells.optimizer-search.recommendation.v1",
                "source_experiment": "008-production-optimizer-search",
                "config_id": recommended["config_id"],
                "block_size": int(recommended["block_size"]),
                "perturbation_q": int(recommended["perturbation_q"]),
                "update_step_q": int(recommended["step_q"]),
                "objective": recommended["objective"],
                "microbatch_groups": int(recommended["batch_groups"]),
                "apply_mode": recommended["apply_mode"],
                "production_cost_class": recommended["production_cost_class"],
                "warning": (
                    "Research recommendation only. Do not mutate runtime config directly; "
                    "port semantics to Rust, add parity tests, rerun Native gate, then PVM."
                ),
            },
        )

    write_progress(
        out,
        "complete",
        status=decision["status"],
        recommended_config=(recommended["config_id"] if recommended else None),
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if recommended else 3


if __name__ == "__main__":
    raise SystemExit(main())
