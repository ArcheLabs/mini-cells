"""Small product CLI for the public HybridCLM API."""

from __future__ import annotations

import argparse
import json

from .hybrid import CellMutation, HybridCLM


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minicells", description="MiniCells HybridCLM tools")
    commands = parser.add_subparsers(dest="command", required=True)
    hybrid = commands.add_parser("hybrid", help="inspect and validate HybridCLM artifacts")
    hybrid_commands = hybrid.add_subparsers(dest="action", required=True)
    inspect = hybrid_commands.add_parser("inspect")
    inspect.add_argument("model")
    inspect.add_argument("--revision")
    inspect.add_argument("--device", default="cpu")
    validate = hybrid_commands.add_parser("validate")
    validate.add_argument("model")
    validate.add_argument("mutation")
    validate.add_argument("--revision", required=True)
    validate.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.action == "inspect":
        print(json.dumps(HybridCLM.inspect_pretrained(args.model, revision=args.revision, device=args.device).to_dict(), indent=2, sort_keys=True))
        return 0
    if args.action == "validate":
        hybrid_model = HybridCLM.from_pretrained(args.model, revision=args.revision, device=args.device)
        mutation = CellMutation.from_pretrained(args.mutation)
        mutation.validate_against(hybrid_model.model, inspection=hybrid_model.inspect(), base_model=hybrid_model.base_model, base_revision=hybrid_model.base_revision)
        print(json.dumps({"status": "compatible", "mutation": mutation.digest()}, indent=2, sort_keys=True))
        return 0
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

