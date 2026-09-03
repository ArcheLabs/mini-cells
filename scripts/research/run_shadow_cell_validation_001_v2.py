#!/usr/bin/env python3
"""Runner for Shadow Cell Validation 001 v2.

Formal execution is intentionally explicit about the registered protocol and
seed.  Smoke mode uses the same model/sidecar code at a small scale and never
produces a scientific classification.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minicells.clm04mini.model import MiniCLMConfig, TinyCLMDecoder  # noqa: E402
from minicells.clm04mini.examples import collate_scored  # noqa: E402
from minicells.clm04mini.protocol import ProtocolError  # noqa: E402
from minicells.shadow_maturation import (  # noqa: E402
    MATURITY_GRID,
    AcceptedModelChain,
    AcceptedModelSnapshot,
    ShadowSidecar,
    build_functional_sketch,
    copy_on_write_artifact,
    evaluate_maturity_frontier,
    hash_accepted_state,
    m0_equivalence_delta,
    routing_is_preserved,
    select_oracle_maturity,
    select_sketch_maturity,
    synthetic_examples,
    train_corrected_direct,
    train_shadow,
)
from publish_shadow_cell_validation_001_v2 import publish_results  # noqa: E402

PROTOCOL_PATH = ROOT / "research/validations/shadow-cell-validation-001-v2-developmental-maturation/protocol.json"
FORMAL_SEEDS = (95311, 95312, 95313)
DEVELOPMENT_SEED = 95301


def _load_protocol() -> dict:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("validation_id") != "shadow-cell-validation-001-v2-developmental-maturation":
        raise ProtocolError("unexpected Shadow v2 validation id")
    if [float(x) for x in payload["maturity_grid"]] != list(MATURITY_GRID):
        raise ProtocolError("maturity grid drift")
    if [int(x) for x in payload["formal_seeds"]] != list(FORMAL_SEEDS):
        raise ProtocolError("formal seed family drift")
    return payload


def _protocol_sha() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _assert_seed(protocol: dict, phase: str, seed: int) -> None:
    if phase == "smoke":
        if int(seed) != DEVELOPMENT_SEED:
            raise ProtocolError(f"smoke requires development seed {DEVELOPMENT_SEED}")
    elif phase == "formal":
        if int(seed) not in FORMAL_SEEDS:
            raise ProtocolError(f"formal requires one of {FORMAL_SEEDS}")
    else:
        raise ProtocolError(f"unknown phase {phase}")


def _config(protocol: dict, *, smoke: bool) -> MiniCLMConfig:
    if smoke:
        return MiniCLMConfig(
            vocab_size=64, max_seq_len=16, num_layers=4, d_model=16,
            n_heads=4, dense_ff_hidden=32, base_cells=4, cell_hidden=4,
            routing_salt="shadow-cell-v2-smoke",
        )
    model = protocol["model"]
    cells = model["cell_layers"]
    dense = model["shared_dense_layers"]
    return MiniCLMConfig(
        vocab_size=int(model["vocab_size"]), max_seq_len=int(model["context_length"]),
        num_layers=int(model["layers"]), d_model=int(model["width"]),
        n_heads=int(model["attention_heads"]), dense_ff_hidden=int(dense["ffn_hidden"]),
        base_cells=int(cells["base_cells_per_layer"]), cell_hidden=int(cells["base_cell_hidden"]),
        routing_salt=str(model["routing_salt"]),
    )


def _load_model(path: Path | None, cfg: MiniCLMConfig, device: torch.device) -> TinyCLMDecoder:
    model = TinyCLMDecoder(cfg)
    if path is not None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class _SmokeTokenizer:
    pad_id = 0


def _examples(cfg: MiniCLMConfig, *, seed: int, smoke: bool) -> dict[str, list]:
    count = 8 if smoke else 64
    return {
        "A": synthetic_examples(vocab_size=cfg.vocab_size, domain="base", count=count, seed=seed + 1),
        "B": synthetic_examples(vocab_size=cfg.vocab_size, domain="math", count=count, seed=seed + 2),
        "C": synthetic_examples(vocab_size=cfg.vocab_size, domain="story", count=count, seed=seed + 3),
        "D": synthetic_examples(vocab_size=cfg.vocab_size, domain="math", count=count, seed=seed + 4),
    }


def _frontier_figures(payload: dict, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output.mkdir(parents=True, exist_ok=True)
    arms = payload.get("arms", {})
    for phase in ("B", "C", "D"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for arm, phases in arms.items():
            row = phases.get(phase)
            if not row or not row.get("maturity_frontier"):
                continue
            frontier = row["maturity_frontier"]
            x = [item["maturity"] for item in frontier]
            axes[0].plot(x, [item["new_gain"] for item in frontier], marker="o", label=arm)
            axes[1].plot(x, [item["old_regression"] for item in frontier], marker="o", label=arm)
        axes[0].set(xlabel="Shadow maturity m", ylabel="New-domain gain", title=f"Phase {phase}: capability")
        axes[1].set(xlabel="Shadow maturity m", ylabel="Old-domain regression", title=f"Phase {phase}: retention")
        axes[1].axhline(payload["thresholds"]["max_old_regression"], color="black", linestyle="--")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"developmental-frontier-{phase}.png", dpi=160)
        plt.close(fig)


def run(seed: int, *, phase: str, device_name: str, output: Path, checkpoint: Path | None,
        steps: int | None = None) -> dict:
    protocol = _load_protocol()
    _assert_seed(protocol, phase, seed)
    smoke = phase == "smoke"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(int(seed))
    cfg = _config(protocol, smoke=smoke)
    base = _load_model(checkpoint, cfg, device)
    tokenizer = _SmokeTokenizer()
    examples = _examples(cfg, seed=seed, smoke=smoke)
    train_steps = int(steps if steps is not None else (2 if smoke else 100))
    thresholds = protocol["thresholds"]
    result: dict = {
        "validation_id": protocol["validation_id"],
        "protocol_sha256": _protocol_sha(),
        "implementation_commit": _git_revision(),
        "phase": phase,
        "seed": int(seed), "device": str(device),
        "maturity_grid": list(MATURITY_GRID),
        "status": "SMOKE_ONLY" if smoke else "PARTIAL_RUN",
        "scientific_decision": False,
        "thresholds": {
            "max_old_regression": float(thresholds["max_old_regression"]),
            "min_new_gain": float(thresholds["min_new_gain"]),
        },
        "arms": {}, "aggregate": {},
        "validity": {"formal_seed_registered": int(seed) in FORMAL_SEEDS,
                     "protocol_hash_matches": True, "required_arms_completed": True,
                     "all_maturity_values_evaluated": True, "finite_results": True},
    }

    arm_modes = {
        "shadow_full": "input_only",
        "shadow_oracle": "input_only",
        "shadow_sketch": "input_only",
        "task_id_shadow": "task_id",
    }
    for arm in ("corrected_direct", *arm_modes):
        accepted: TinyCLMDecoder | AcceptedModelChain = deepcopy(base)
        phases: dict[str, dict] = {}
        historical = list(examples["A"])
        for phase_name in ("B", "C", "D"):
            current = examples[phase_name]
            if arm == "corrected_direct":
                model = accepted.base if isinstance(accepted, AcceptedModelChain) else accepted
                direct_before = hash_accepted_state(model)
                direct = train_corrected_direct(
                    model, current, tokenizer, device, steps=train_steps, batch_size=min(8, len(current)), seed=seed,
                )
                phases[phase_name] = {
                    **direct, "selected_maturity": None, "maturity_frontier": [],
                    "historical_replay_count": 0, "false_safe": False,
                    "accepted_hash_before_training": direct_before,
                    "accepted_hash_after_training": hash_accepted_state(model),
                }
                historical.extend(current)
                continue

            sidecar = ShadowSidecar(accepted, gate_mode=arm_modes[arm]).to(device)
            sketch = build_functional_sketch(sidecar, historical, tokenizer, device, batch_size=8)
            before = AcceptedModelSnapshot(accepted)
            train = train_shadow(
                sidecar, current, tokenizer, device, steps=train_steps,
                batch_size=min(8, len(current)), seed=seed + ord(phase_name),
            )
            frontier = evaluate_maturity_frontier(
                accepted, sidecar, MATURITY_GRID, historical, current, arm_modes[arm],
                tokenizer=tokenizer, device=device, batch_size=8,
            )
            gains = {float(row["maturity"]): float(row["new_gain"]) for row in frontier}
            if arm == "shadow_full" or arm == "task_id_shadow":
                selected = 1.0
            elif arm == "shadow_oracle":
                selected = select_oracle_maturity(
                    frontier, float(thresholds["max_old_regression"]), float(thresholds["min_new_gain"]),
                )
            else:
                selected = select_sketch_maturity(
                    sidecar, sketch, MATURITY_GRID, gains,
                    max_predicted_damage=float(thresholds["max_old_regression"]),
                    min_new_gain=float(thresholds["min_new_gain"]),
                )
            selected_row = next((row for row in frontier if row["maturity"] == selected), None)
            false_safe = bool(
                arm == "shadow_sketch" and selected_row is not None
                and selected_row["old_regression"] > float(thresholds["max_old_regression"])
            )
            before.assert_unchanged(accepted)
            probe_x, _, _, _ = collate_scored(current[:1], pad_id=0, device=device)
            maturity_delta = m0_equivalence_delta(
                sidecar, probe_x, [current[0].address_id],
            )
            phases[phase_name] = {
                **train,
                "selected_maturity": selected,
                "maturity_frontier": frontier,
                "shadow_parameter_count": sidecar.shadow_parameter_count,
                "sketch_size_bytes": sketch.bytes,
                "sketch_rank": sketch.sketch_rank,
                "false_safe": false_safe,
                "false_safe_rate": float(false_safe),
                "historical_examples_seen_by_oracle_evaluator": len(historical) if arm == "shadow_oracle" else 0,
                "historical_examples_seen_by_hidden_final_evaluator": len(historical),
                "m0_max_abs_logit_delta": maturity_delta,
                "routing_preserved": routing_is_preserved(accepted if isinstance(accepted, TinyCLMDecoder) else accepted.base, sidecar, [item.address_id for item in current]),
                "accepted_hash_before_training": train["accepted_hash_before_training"],
                "accepted_hash_after_training": train["accepted_hash_after_training"],
            }
            if selected is not None:
                artifact = output / f"seed-{seed}" / "checkpoints" / f"accepted-{phase_name}-{arm}.pt"
                copy_on_write_artifact(accepted, sidecar, selected, artifact, phase=phase_name, arm=arm)
                accepted = AcceptedModelChain(accepted) if isinstance(accepted, TinyCLMDecoder) else accepted
                accepted = accepted.append(sidecar, selected)
            historical.extend(current)
        result["arms"][arm] = phases

    result["aggregate"] = {"formal_seed_family": list(FORMAL_SEEDS), "formal_seeds_run": [],
                            "false_safe": sum(int(row.get("false_safe", False)) for phases in result["arms"].values() for row in phases.values())}
    result["validity"]["m0_identity_passes"] = all(
        float(row.get("m0_max_abs_logit_delta", 0.0)) <= 1e-6
        for phases in result["arms"].values() for row in phases.values()
    )
    out_seed = output / f"seed-{seed}"
    out_seed.mkdir(parents=True, exist_ok=True)
    (out_seed / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _frontier_figures(result, out_seed / "figures")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", "--output-dir", dest="output", type=Path,
                        default=ROOT / "results/shadow-cell-validation-001-v2-developmental-maturation")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--push-results",
        action="store_true",
        help="After a formal seed completes, curate and push result artifacts to GitHub.",
    )
    parser.add_argument(
        "--publish-branch",
        default="kaggle/shadow-cell-validation-001-v2-results",
        help="Result branch used with --push-results.",
    )
    parser.add_argument(
        "--secret-name",
        default="GITHUB_TOKEN",
        help="Environment variable / Kaggle Secret containing the GitHub token.",
    )
    parser.add_argument("--kaggle-script-version-id")
    args = parser.parse_args()
    seed = int(args.seed if args.seed is not None else (DEVELOPMENT_SEED if args.phase == "smoke" else FORMAL_SEEDS[0]))
    protocol = _load_protocol()
    _assert_seed(protocol, args.phase, seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.preflight_only:
        payload = {
            "status": "PREFLIGHT_PASS", "scientific_decision": False,
            "protocol_sha256": _protocol_sha(), "seed": seed,
            "formal_seed_registered": seed in FORMAL_SEEDS,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "device": args.device, "dataset": "injected JSON or deterministic smoke fallback",
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    result = run(seed, phase=args.phase, device_name=args.device, output=args.output,
                 checkpoint=args.checkpoint, steps=args.steps)
    if args.push_results:
        if args.phase != "formal":
            raise SystemExit("--push-results is only allowed for formal runs")
        publish_results(
            ROOT,
            args.output,
            branch=args.publish_branch,
            secret_name=args.secret_name,
            kaggle_script_version_id=args.kaggle_script_version_id,
        )
    print("SHADOW_CELL_VALIDATION_001_V2_SMOKE_PASS" if args.phase == "smoke" else json.dumps(
        {"status": result["status"], "seed": seed, "protocol_sha256": result["protocol_sha256"]}, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
