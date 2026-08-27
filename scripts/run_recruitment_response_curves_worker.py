from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import minicells.language_proposal_utility as proposal_utility  # noqa: E402
from minicells.language_growing_organism import build_cellular_model  # noqa: E402
from minicells.language_proposal_checkpoints import (  # noqa: E402
    donor_path,
    load_checkpoint,
    localized_state_from_payload,
    phase1_path,
)
from minicells.language_recruitment_numerics import stable_gated_replicator_activity  # noqa: E402
from minicells.language_recruitment_response import RECRUITMENT_GRID  # noqa: E402
from minicells.language_utility_skill_data import (  # noqa: E402
    SKILL_FAMILIES,
    batch_from_indices,
    generate_utility_skill_corpus,
)
import run_proposal_utility_discovery_worker as e019  # noqa: E402


proposal_utility._gated_replicator_activity = stable_gated_replicator_activity

CANDIDATE_FAMILIES = (*SKILL_FAMILIES, e019.RANDOM_CONTROL)
BATCH_SIZE = e019.UTILITY_BATCH_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 019b recruitment-response replicate.")
    parser.add_argument("--replicate", type=int, choices=range(e019.N_REPLICATES), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_candidate(checkpoint_dir: Path, replicate: int, family: str, device: torch.device):
    path = donor_path(checkpoint_dir, replicate, family)
    payload = load_checkpoint(path, kind="donor", replicate=replicate, family=family)
    if payload is None:
        raise FileNotFoundError(
            f"Missing stable Experiment 019 donor checkpoint: {path}. "
            "019b never retrains donors; run the checkpointed stable 019 first."
        )
    vocab_size = int(payload["vocab_size"])
    model = build_cellular_model(vocab_size, "G").to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    localized_state = localized_state_from_payload(payload["localized_state"])
    return model, localized_state, vocab_size


@torch.no_grad()
def measure_response(
    input_family: str,
    candidate_family: str,
    model,
    localized_state,
    corpus,
    *,
    replicate: int,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mask = corpus.loss_mask.to(device)
    for start in range(0, len(corpus.sequences), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(corpus.sequences))
        indices = torch.arange(start, stop)
        inputs, targets, _ = batch_from_indices(corpus, indices, device=device)
        losses: dict[float, torch.Tensor] = {}
        for recruitment in RECRUITMENT_GRID:
            result = proposal_utility.forward_with_fixed_recruitment(
                model,
                inputs,
                localized_state,
                float(recruitment),
            )
            losses[float(recruitment)] = proposal_utility.per_example_masked_nll(
                result.output.logits.float(),
                targets,
                mask,
            ).detach().float().cpu()
        closed = losses[0.0]
        for local_index, example in enumerate(range(start, stop)):
            loss0 = float(closed[local_index])
            for recruitment in RECRUITMENT_GRID:
                loss = float(losses[float(recruitment)][local_index])
                rows.append({
                    "replicate": replicate,
                    "example": example,
                    "input_family": input_family,
                    "candidate_family": candidate_family,
                    "candidate_kind": "untrained-control" if candidate_family == e019.RANDOM_CONTROL else "trained",
                    "matching_family": int(candidate_family == input_family),
                    "recruitment": float(recruitment),
                    "loss": loss,
                    "loss_closed": loss0,
                    "value": loss0 - loss,
                })
    return rows


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 019b requires CUDA for checkpoint response sweeps")
    device = torch.device("cuda:0")
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replicate = int(args.replicate)

    phase1 = load_checkpoint(phase1_path(checkpoint_dir, replicate), kind="phase1", replicate=replicate)
    if phase1 is None:
        raise FileNotFoundError(
            f"Missing stable Experiment 019 Phase-1 checkpoint for r{replicate} in {checkpoint_dir}. "
            "019b never trains or repairs checkpoints."
        )

    rows: list[dict[str, object]] = []
    expected_vocab: int | None = None
    for family_index, input_family in enumerate(SKILL_FAMILIES):
        corpus = generate_utility_skill_corpus(
            e019.UTILITY_EXAMPLES_PER_FAMILY,
            seed=e019.UTILITY_SEED_BASE + 10_000 * replicate + family_index,
            families=(input_family,),
        )
        for candidate_family in CANDIDATE_FAMILIES:
            model, state, vocab_size = _load_candidate(checkpoint_dir, replicate, candidate_family, device)
            if expected_vocab is None:
                expected_vocab = vocab_size
            elif expected_vocab != vocab_size:
                raise RuntimeError("019b candidate checkpoints disagree on vocabulary size")
            rows.extend(measure_response(
                input_family,
                candidate_family,
                model,
                state,
                corpus,
                replicate=replicate,
                device=device,
            ))
            del model
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    output_path = output_dir / f"r{replicate}-response-observations.csv.gz"
    frame.to_csv(output_path, index=False, compression="gzip")
    worker = {
        "format": "minicells.recruitment-response-worker.v1",
        "experiment": "019b",
        "replicate": replicate,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_dir": str(checkpoint_dir),
        "candidate_families": list(CANDIDATE_FAMILIES),
        "input_families": list(SKILL_FAMILIES),
        "examples_per_family": e019.UTILITY_EXAMPLES_PER_FAMILY,
        "recruitment_grid": list(RECRUITMENT_GRID),
        "rows": len(frame),
        "training_performed": False,
    }
    (output_dir / f"r{replicate}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"replicate": replicate, "rows": len(frame), "output": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
