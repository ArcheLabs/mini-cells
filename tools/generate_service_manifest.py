#!/usr/bin/env python3
"""Write the artifact manifest from explicit build inputs.

The generator intentionally creates a new object instead of loading and
updating an existing manifest.  This prevents deployment history or stale
source fields from leaking into a newly built artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build_manifest(args: argparse.Namespace) -> dict:
    blob = Path(args.blob).read_bytes()
    pvm = Path(args.pvm).read_bytes()
    return {
        "format": "minicells.artifact-manifest.v2",
        "protocol": "mini-cells-service-v1",
        "jam_semantics": "0.7.2",
        "model_format": 1,
        "optimizer": "guarded-sign-spsa-v2",
        "optimizer_version": 2,
        "parameter_count": 4476,
        "genesis_mode": "deterministic-splitmix64-seed-1",
        "genesis_model_hash": args.genesis_hash,
        "code_hash": "0x" + hashlib.blake2b(blob, digest_size=32).hexdigest(),
        "service_code_hash": "0x" + hashlib.blake2b(blob, digest_size=32).hexdigest(),
        "blob_bytes": len(blob),
        "service_pvm_sha256": hashlib.sha256(pvm).hexdigest(),
        "build_provenance": {
            "mini_cells_source_ref": args.source_ref,
            "mini_cells_source_tree": args.source_tree,
            "mini_cells_source_dirty": args.source_dirty == "true",
            "minijam_build_ref": args.minijam_ref,
            "jambda_build_ref": args.jambda_ref,
            "jambda_standalone_adapter_ref": args.jambda_adapter_ref,
            "converter_build_ref": args.converter_ref,
            "jamscript_used_for_build": False,
            "jamscript_build_ref": None,
            "rust_toolchain": args.rust_version,
            "target_sha256": args.target_hash,
        },
        # Compatibility belongs in implementation-status.json; deployment
        # attempts are append-only records under artifacts/deployments/.
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--pvm", required=True)
    parser.add_argument("--genesis-hash", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--source-dirty", choices=("true", "false"), required=True)
    parser.add_argument("--minijam-ref", required=True)
    parser.add_argument("--jambda-ref", required=True)
    parser.add_argument("--jambda-adapter-ref", required=True)
    parser.add_argument("--converter-ref", required=True)
    parser.add_argument("--rust-version", required=True)
    parser.add_argument("--target-hash", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.write_text(json.dumps(build_manifest(args), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
