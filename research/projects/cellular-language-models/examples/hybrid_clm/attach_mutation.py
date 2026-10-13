"""Attach a compatible Cell mutation, run a caller-provided model input, detach."""

from __future__ import annotations

import argparse

from minicells import CellMutation, CellPlacement, HybridCLM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mutation")
    parser.add_argument("--model", default="ibm-granite/granite-3.1-1b-a400m-base")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    hybrid = HybridCLM.from_pretrained(args.model, revision=args.revision)
    hybrid.cellularize(CellPlacement(layer=args.layer, experts="all"))
    mutation = CellMutation.from_pretrained(args.mutation)
    hybrid.attach(mutation).set_alpha(mutation, args.alpha)
    print({"status": "attached", "alpha": args.alpha, "inspection": hybrid.inspect().to_dict()})
    hybrid.detach(mutation)
    print({"restored": hybrid.verify_restoration().passed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

