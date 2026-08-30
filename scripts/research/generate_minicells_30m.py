from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_30m import build_minicells_30m  # noqa: E402
from minicells.language_data import load_tokenizer  # noqa: E402
from minicells.language_training import generate_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with MiniCells-30M v0.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "artifacts"
        / "experiments"
        / "007-minicells-30m"
        / "minicells-30m-v0-fp16.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "007-minicells-30m" / "tokenizer.json",
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7007)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    if payload.get("format") != "minicells.language-inference.v1":
        raise RuntimeError("unexpected model artifact format")
    if payload.get("model_name") != "minicells-30m-v0":
        raise RuntimeError("checkpoint is not MiniCells-30M v0")
    tokenizer = load_tokenizer(args.tokenizer)
    model = build_minicells_30m(tokenizer.get_vocab_size())
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    text = generate_text(
        model,
        tokenizer,
        args.prompt,
        device=device,
        max_context=128,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        amp=device.type == "cuda",
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
