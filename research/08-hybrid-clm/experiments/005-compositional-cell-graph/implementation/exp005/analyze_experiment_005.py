"""Analyze an immutable source-specific Experiment 005 arm archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parents[1]
ROOT = next(parent for parent in HERE.parents
            if (parent / "pyproject.toml").is_file())
sys.path.insert(0, str(HERE.parent))

from exp005.models import ARMS
from exp005.preflight import seed_role
from exp005.report import matrix_decision, seed_decision


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def analyze(archive: Path, protocol_path: Path, config_path: Path) -> tuple[dict[str, Any], str]:
    source_commit = archive.name
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("archive directory must be named with the full source commit SHA")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    observed: dict[str, dict[str, dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {"protocol.json": _sha256(protocol_bytes), "config.json": _sha256(config_bytes)}
    for seed_dir in sorted(archive.glob("seed-*")):
        if not seed_dir.is_dir() or not re.fullmatch(r"seed-[0-9]+", seed_dir.name):
            continue
        seed = int(seed_dir.name[len("seed-"):])
        observed[seed] = {}
        for arm in ARMS:
            metric_path = seed_dir / arm / "metrics.json"
            if not metric_path.is_file():
                continue
            raw = metric_path.read_bytes()
            metrics = json.loads(raw)
            expected = {"arm": arm, "seed": seed, "source_commit": source_commit}
            for key, value in expected.items():
                if metrics.get(key) != value:
                    raise ValueError(f"{metric_path}: {key} does not match its archive path")
            if protocol.get("status") == "REGISTERED":
                if metrics.get("seed_role") != seed_role(config, seed):
                    raise ValueError("archive seed role differs from registered matrix")
                for key, expected_hash in (("protocol_sha256", _sha256(protocol_bytes)),
                                           ("config_sha256", _sha256(config_bytes))):
                    if metrics.get("identity", {}).get(key) != expected_hash:
                        raise ValueError("archive config/protocol differs from analysis contract")
            profile_path = seed_dir / arm / "decode-profile.json"
            if profile_path.is_file():
                profile_raw = profile_path.read_bytes()
                profile = json.loads(profile_raw)
                if profile.get("identity") != metrics.get("identity") or profile.get("status") != "DECODE_PROFILE_COMPLETED":
                    raise ValueError(f"{profile_path}: decode profile identity/status mismatch")
                evaluation = metrics.get("final_holdout") or metrics.get("development_evaluation")
                if evaluation is None:
                    raise ValueError("completed arm has no seed-role evaluation")
                evaluation["decode_parity"] = profile["decode_parity"]
                evaluation["decode_profile"] = profile["decode_profile"]
                input_hashes[profile_path.relative_to(archive).as_posix()] = _sha256(profile_raw)
            observed[seed][arm] = metrics
            input_hashes[metric_path.relative_to(archive).as_posix()] = _sha256(raw)

    identity = {"experiment_id": "005-compositional-cell-graph",
                "source_commit": source_commit,
                "protocol_sha256": input_hashes["protocol.json"],
                "input_sha256": _sha256(json.dumps(input_hashes, sort_keys=True,
                                                    separators=(",", ":")).encode())}
    if protocol.get("status") != "REGISTERED":
        report = {"status": "PROTOCOL_NOT_REGISTERED", "scientific_decision": False,
                  "identity": identity,
                  "observed_seed_arms": {str(seed): sorted(arms) for seed, arms in observed.items()},
                  "blockers": protocol.get("not_registered_because", []),
                  "note": "No scientific matrix analysis is valid before protocol registration."}
    else:
        seed_reports = []
        for seed in sorted(observed):
            seed_reports.append(seed_decision(observed[seed], protocol=protocol, seed=seed))
        confirmation_seeds = [int(seed) for seed in config.get("seeds", {}).get("confirmation", [])]
        report = matrix_decision(seed_reports, protocol, confirmation_seeds)
        report.update({"identity": identity,
                       "observed_seed_arms": {str(seed): sorted(arms)
                                              for seed, arms in observed.items()},
                       "limitations": ["development and confirmation evidence remain separate",
                                       "no scientific conclusion is emitted by this analysis tool"]})
    return report, identity["input_sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True,
                        help="source-specific directory containing seed-N/ARM/metrics.json")
    parser.add_argument("--protocol", default=str(EXPERIMENT_DIR / "protocol.json"))
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config/registered.json"))
    parser.add_argument("--output", help="optional immutable JSON report path")
    parser.add_argument("--output-dir", help="write under this directory using the input-content hash")
    args = parser.parse_args()
    if args.output and args.output_dir:
        parser.error("choose at most one of --output and --output-dir")
    report, input_sha = analyze(Path(args.archive).resolve(), Path(args.protocol), Path(args.config))
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output or args.output_dir:
        target = (Path(args.output_dir) / (input_sha[:16] + ".json")
                  if args.output_dir else Path(args.output))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"immutable analysis report already exists with different content: {target}")
        target.write_text(serialized, encoding="utf-8")
        print(json.dumps({"status": report["status"], "scientific_decision": False,
                          "analysis_report": str(target), "input_sha256": input_sha,
                          "observed_seed_arms": report.get("observed_seed_arms", {})}, indent=2))
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
