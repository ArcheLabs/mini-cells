"""Verify the complete designated Native MoM checkpoint without importing CUDA."""
import argparse
import json
from pathlib import Path

from minicells_native.model_assets import audit_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Pinned snapshot with model and tokenizer files")
    parser.add_argument("--output", type=Path, required=True, help="Output model manifest")
    args = parser.parse_args()
    receipt = audit_checkpoint(args.snapshot, args.snapshot)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {receipt['tensor_count']} tensors in {receipt['model_revision']}", flush=True)


if __name__ == "__main__":
    main()
