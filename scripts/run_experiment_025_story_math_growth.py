#!/usr/bin/env python3
"""Orchestrate Experiment 025 on exactly two CUDA GPUs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_30m import prepare_30m_corpus  # noqa: E402
from minicells.language_data import load_tokenizer  # noqa: E402
from minicells.story_math_shift_30m import (  # noqa: E402
    RESULT_DIR_NAME,
    SHIFT_TOKENS,
    experiment_budget,
    prepare_math_corpus,
    schedule_manifest,
)

SOURCE_006 = ROOT / "artifacts" / "experiments" / "006-consumer-language-scaling"
PROTOCOL = ROOT / "research" / "experiment-025-protocol.json"
OUT = ROOT / "results" / RESULT_DIR_NAME
WORKER = ROOT / "scripts" / "run_experiment_025_story_math_worker.py"
REPORT = ROOT / "scripts" / "report_experiment_025_story_math_growth.py"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run Experiment 025 — Story→Math growth")
    result.add_argument("--shift-tokens", type=int, default=SHIFT_TOKENS)
    result.add_argument("--max-wall-hours", type=float, default=9.25)
    result.add_argument("--reset", action="store_true")
    return result.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def worker_command(
    arm: str,
    *,
    story_cache: Path,
    math_cache: Path,
    shift_tokens: int,
    max_wall_hours: float,
) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--arm",
        arm,
        "--story-cache-dir",
        str(story_cache),
        "--math-cache-dir",
        str(math_cache),
        "--output-dir",
        str(OUT),
        "--shift-tokens",
        str(shift_tokens),
        "--max-wall-hours",
        str(max_wall_hours),
    ]


def main() -> int:
    args = parser().parse_args()
    if args.shift_tokens <= 0 or args.shift_tokens > SHIFT_TOKENS:
        raise ValueError("--shift-tokens must be in (0, 50M]")
    if args.max_wall_hours <= 0 or args.max_wall_hours >= 10.0:
        raise ValueError("--max-wall-hours must be positive and below the ~10h session budget")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Experiment 025 requires a clean tracked Git tree")

    available = torch.cuda.device_count()
    if available < 2:
        raise RuntimeError(
            "Formal Experiment 025 is budgeted for two concurrent CUDA GPUs; "
            f"only {available} visible. Enable Tesla T4 x2 before running."
        )
    used = 2
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reset:
        # Preserve expensive corpus caches and the reproducible LLM pretrain if present.
        for arm in ("llm", "clm"):
            arm_dir = OUT / arm
            if not arm_dir.exists():
                continue
            for path in arm_dir.iterdir():
                if arm == "llm" and path.name == "pretrain":
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    shutil.copy2(PROTOCOL, OUT / "protocol.json")

    print("preparing/reusing Experiment-007 TinyStories corpus")
    corpus = prepare_30m_corpus(ROOT, source_006_dir=SOURCE_006)
    story_cache = corpus.train_path.parent
    tokenizer = load_tokenizer(corpus.tokenizer_path)
    print("preparing/reusing synthetic arithmetic corpus")
    math_cache = OUT / "cache"
    arithmetic = prepare_math_corpus(math_cache, tokenizer)

    provenance = {
        "format": "minicells.story-math-shift-30m-run.v1",
        "code_commit": _git("rev-parse", "HEAD"),
        "code_tree_sha": _git("rev-parse", "HEAD^{tree}"),
        "tracked_tree_dirty": False,
        "gpu_count_visible": available,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(available)],
        "gpu_count_used": used,
        "budget": experiment_budget(),
        "requested_shift_tokens": args.shift_tokens,
        "worker_max_wall_hours": args.max_wall_hours,
        "schedule": schedule_manifest(args.shift_tokens),
        "story_corpus_manifest": corpus.manifest,
        "math_corpus_manifest": arithmetic["manifest"],
    }
    _json_write(OUT / "run-provenance.json", provenance)

    arms = ("llm", "clm")
    active: list[tuple[str, int, subprocess.Popen[str], Path]] = []
    for gpu_index, arm in enumerate(arms):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        log_path = OUT / f"{arm}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            worker_command(
                arm,
                story_cache=story_cache,
                math_cache=math_cache,
                shift_tokens=args.shift_tokens,
                max_wall_hours=args.max_wall_hours,
            ),
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.close()
        active.append((arm, gpu_index, process, log_path))
        print(f"started {arm:3s} on physical GPU {gpu_index}")

    failures: list[str] = []
    for arm, gpu_index, process, log_path in active:
        code = process.wait()
        print(f"--- {arm} / GPU {gpu_index} ---")
        print(log_path.read_text(encoding="utf-8").rstrip())
        if code != 0:
            failures.append(f"{arm} exited {code}; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))

    summaries = {}
    for arm in arms:
        path = OUT / arm / "worker-summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summaries[arm] = json.loads(path.read_text(encoding="utf-8"))
    complete = all(bool(summaries[arm].get("complete")) for arm in arms)
    _json_write(
        OUT / "worker-summary.json",
        {
            "format": "minicells.story-math-shift-30m-workers.v1",
            "complete": complete,
            "arms": summaries,
        },
    )
    if complete:
        subprocess.run(
            [sys.executable, str(REPORT), "--results-dir", str(OUT)],
            cwd=ROOT,
            check=True,
        )
        print(f"Experiment 025 complete: {OUT}")
    else:
        print(
            "Experiment 025 reached its wall-time guard before both arms completed. "
            "Re-run the same command to resume from the periodic checkpoints."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
